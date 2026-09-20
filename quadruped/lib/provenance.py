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
    return {
        "commit": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git(repo, "status", "--porcelain")),
    }


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
    """Raise if the manifest records a dirty tree. Call before publishing a number."""
    if m.get("git", {}).get("dirty"):
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
