"""One manifest schema for data, models and results.

This exists because several figures reached a slide with nothing behind them: a throughput
pair that had only ever been printed to a terminal, a horizon figure with no measurement
at all, and a corpus cost that was a hardcoded literal inside the script that appeared to
compute it. The failure mode is identical each time -- a number that is true-sounding,
repeated, and unattached.

A manifest makes the attachment mechanical. Given one, the bytes can be located and
verified, the inputs walked back, and the command re-run.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path


def _git(repo: Path, *args) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance must never break the run
        return ""


def git_state(repo: Path) -> dict:
    """Commit, branch, and whether the tree is dirty.

    `dirty` is recorded rather than refused, because iterating with uncommitted changes is
    normal and useful. What it disqualifies is QUOTING the result: a dirty manifest
    describes a run that cannot be reproduced from any commit, and the tools that publish
    a number check this field.
    """
    commit = _git(repo, "rev-parse", "HEAD")
    # OUTSIDE A REPO, SAY SO -- do not report clean. This used to return
    # `dirty: bool(git status output)`, and with no repository the output is empty, so a
    # staged copy with no version control at all reported `dirty: False`: clean, and
    # therefore publishable, while identifying no code whatsoever. Every corpus collected
    # on hpcfund was recorded that way, because the tree there is an rsync'd copy rather
    # than a checkout. `dirty: None` means unknown, and `require_clean` refuses it.
    if not commit:
        # A staged copy can still say where it came from: the staging step writes the
        # source commit to .source_commit beside the tree. That is a claim about the copy,
        # not a checked fact -- so it is recorded as source_commit, never as commit, and
        # the code fingerprint below is what lets it be verified against that commit.
        marker = repo / ".source_commit"
        src = marker.read_text().strip() if marker.exists() else ""
        return {"commit": "", "branch": "", "dirty": None, "in_repo": False,
                "source_commit": src}
    return {
        "commit": commit,
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git(repo, "status", "--porcelain")),
        "in_repo": True,
    }


# The source files whose content determines what a run DID. Hashed individually so a
# manifest identifies its code even where there is no git -- and so two manifests can be
# compared file by file to see exactly which part of the pipeline differed between them,
# which a single commit hash cannot tell you.
_CODE = ("collect.py", "train.py", "finetune.py", "evaluate.py", "doctor.py",
         "lib/*.py", "params/*.py", "params/*.yaml")


def code_fingerprint(root: Path) -> dict:
    """sha256 of every source file that shapes a run, plus one combined hash.

    Written because the 800-episode collection ran ACROSS a code change -- each array task
    loads collect.py when it starts, so tasks begun before a fix and after it ran different
    code -- and nothing in the manifests could say which was which. Timestamps could, but
    only by inference. This records it.
    """
    files = {}
    for pat in _CODE:
        for f in sorted(root.glob(pat)):
            if f.is_file():
                files[str(f.relative_to(root))] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    pkg = root.parent / "src" / "nedm" / "quadruped"
    if pkg.is_dir():
        for f in sorted(pkg.glob("*.py")):
            files["src/nedm/quadruped/" + f.name] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    combined = hashlib.sha256("".join(f"{k}:{v}" for k, v in sorted(files.items()))
                              .encode()).hexdigest()[:16]
    return {"combined": combined, "files": files}


def sha256(path: str | Path, cap_mb: int | None = None) -> str:
    """Content hash. `cap_mb` hashes only a prefix, for artifacts where a full pass is
    not worth the minutes; the cap is recorded alongside so the two are never confused."""
    h = hashlib.sha256()
    n = 0
    limit = None if cap_mb is None else cap_mb * 2**20
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(2**20), b""):
            h.update(chunk)
            n += len(chunk)
            if limit and n >= limit:
                return h.hexdigest() + f"#first{cap_mb}MiB"
    return h.hexdigest()


def chrono_provenance() -> dict:
    """The Chrono binary that will actually run, not the one on disk you assume.

    Resolved through the importing interpreter, because the build under a host's home
    directory is frequently NOT what the environment imports -- that gap is exactly how a
    capacity arm came to read -9.5% on one box where another read -39.6% on the same
    policy.
    """
    try:
        import pychrono
        d = Path(pychrono.__file__).parent
        so = d / "_core.so"
        return {"path": str(d),
                "md5": hashlib.md5(so.read_bytes()).hexdigest()[:16] if so.exists() else None}
    except Exception as e:  # noqa: BLE001
        return {"path": None, "md5": None, "error": f"{type(e).__name__}: {e}"}


def versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("torch", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    return out


def new_run_id(prefix: str = "") -> str:
    """`YYYYmmdd-HHMMSS-<short>`; the primary key every artifact is addressed by.

    The salt is drawn from os.urandom, NOT hashed from timestamp+pid+host. That earlier
    version collided: fifty calls inside the same second on one process produced one id,
    because every input to the hash was constant over that second. A colliding primary key
    silently overwrites an artifact, which is the worst failure this module could have.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    salt = hashlib.sha256(os.urandom(16)).hexdigest()[:7]
    return f"{prefix + '-' if prefix else ''}{stamp}-{salt}"


def manifest(kind: str, repo: Path, *, run_id: str | None = None,
             inputs: list[dict] | None = None, outputs: list[dict] | None = None,
             metric_defs: dict | None = None, extra: dict | None = None,
             notes: str = "") -> dict:
    """Assemble the record. `kind` is corpus | model | result."""
    if kind not in ("corpus", "model", "result"):
        raise ValueError(f"kind must be corpus|model|result, got {kind!r}")
    m = {
        "run_id": run_id or new_run_id(kind),
        "kind": kind,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_state(repo),
        "code": code_fingerprint(repo if (repo / "collect.py").exists()
                                 else repo / "quadruped"),
        "command": sys.argv,
        "host": {"name": socket.gethostname(), "platform": platform.platform()},
        "chrono_build": chrono_provenance(),
        "versions": versions(),
        "inputs": inputs or [],
        "outputs": outputs or [],
        # Versioned because a metric silently changed meaning mid-study -- val_loss moved
        # from a prefix of the validation split to a random sample -- and invalidated every
        # comparison spanning that date with nothing looking wrong. A stamped definition
        # turns that into a visible mismatch instead of a wrong number.
        "metric_defs": metric_defs or {},
        "notes": notes,
    }
    if extra:
        m.update(extra)
    return m


def write(path: str | Path, m: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2, sort_keys=False) + "\n")
    return p


def require_clean(m: dict, what: str = "this artifact") -> None:
    """Raise unless the manifest pins the code that produced it. Call before publishing.

    Refuses three states, not one. A DIRTY tree cannot be reproduced from any commit. A
    tree that is NOT A REPO and names no source commit cannot be reproduced at all -- and
    used to pass this check, because `dirty` read False for it. A staged copy that DOES name
    its source commit passes, since its code fingerprint can be compared against that
    commit to confirm it.
    """
    g = m.get("git", {})
    if g.get("in_repo") is False and not g.get("source_commit"):
        raise RuntimeError(
            f"{what} was produced outside a git repository with no recorded source "
            f"commit, so nothing identifies the code behind it beyond the per-file hashes "
            f"in manifest['code']. Stage with .source_commit written, or match those "
            f"hashes to a commit, before quoting it.")
    if g.get("dirty"):
        raise RuntimeError(
            f"{what} was produced from a dirty working tree (commit "
            f"{m['git'].get('commit', '?')[:8]}), so it cannot be reproduced from any "
            f"commit. Commit the tree and regenerate before quoting it."
        )


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    m = manifest("corpus", repo, notes="self-test",
                 metric_defs={"errdist": "v2"},
                 extra={"excitation": {"action_injection": {"kind": "ou"}}})
    assert m["kind"] == "corpus" and m["run_id"].startswith("corpus-")
    assert "commit" in m["git"] and "dirty" in m["git"]
    assert "chrono_build" in m and "metric_defs" in m
    ids = {new_run_id() for _ in range(5000)}
    assert len(ids) == 5000, f"run_id collision: {5000 - len(ids)} duplicates"
    try:
        require_clean({"git": {"dirty": True, "commit": "abc123"}})
        raise AssertionError("require_clean did not raise on a dirty tree")
    except RuntimeError:
        pass
    print("provenance self-test OK")
    print("  run_id:", m["run_id"])
    print("  git:", m["git"]["branch"], m["git"]["commit"][:8], "dirty:", m["git"]["dirty"])
    print("  chrono:", m["chrono_build"]["md5"])
    print("  versions:", m["versions"])
