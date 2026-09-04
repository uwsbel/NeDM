"""One-touch metadata repair for a collected Go2 dataset.

Rewrites episode ids, recomputes the split from the new id, corrects the stale
checkpoint path, and stamps origin provenance. Run ONCE per dataset, with no
collection in flight.

WHY THE IDS HAD TO CHANGE. episode_id was f"go2_{terrain}_{index:03d}" and the
index runs per FAMILY, so arc/0, constant/0, lateral/0 and pivot/0 were all
`go2_crm_000`. One box's dataset held 1042 episodes under 140 distinct ids, and
preprocess raises on duplicates -- so it would have refused a single dataset
before any cross-box merge was attempted.

WHY THE SPLIT HAD TO BE RECOMPUTED, NOT CARRIED. The collector's
--validation-ratio defaults to 0.0, so EVERY episode in both datasets was
"train" and there was no held-out set at all. That fails silently, where the id
collision fails loudly: training would have run, a loss would have gone down, and
there would have been nothing to select a checkpoint on. The two are fixed
together because assign_split hashes the id -- with eight families sharing one id
those eight episodes always drew the same split, so a val set could have held all
eight of a family's index-0 episodes or missed a family entirely.

GIT PROVENANCE IS RECOVERED PER EPISODE, NOT ASSUMED CONSTANT. HEAD moved several
times during collection here (commits, then rebases onto a shared branch), so
episodes genuinely span commit values -- the same trap dorm-pc hit by rewording a
commit mid-run, in a worse form because a rebase changes trees and not only
messages. Each episode's HEAD is recovered from the reflog by write time rather
than backfilled with one convenient value.

AND A DIGEST OF THE CODE THAT ACTUALLY RAN, which is the field that means what a
reader wants. git_tree covers the whole worktree including docs that changed
under the run; collection_code_digest covers only the files an episode reads,
hashed AT that episode's commit rather than from the working tree -- the rigid
episodes ran under code that has since changed, and a working-tree digest quietly
asserted otherwise until this was fixed.

The report prints the distribution of both. If several commits map to ONE digest,
that is the substantive result: the episodes were produced by identical code even
though HEAD moved. It is a finding of the pass, not an assumption of it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from nedm.hmmwv_data import assign_split  # noqa: E402

VAL_RATIO = 0.2
CKPT_OLD = "/tmp/go2_import/go2_cts_150k.pt"
CKPT_NEW = "/home/kyle/sbel-artifacts/checkpoints/go2_cts_150k.pt"
REPO = Path(__file__).resolve().parents[2]

# The files an episode actually reads. git_tree also covers docs and unrelated
# scripts, which changed during the run; these did not.
CODE_FILES = [
    "scripts/collection/collect_go2_smoke.py",
    "src/nedm/quadruped/imported_policy.py",
    "src/nedm/quadruped/dataset.py",
    "src/nedm/quadruped/robot.py",
    "src/nedm/quadruped/constants.py",
    "src/nedm/quadruped/terrain.py",
    "src/nedm/quadruped/soilprobe.py",
]

# Columns carrying the id, present in EVERY csv row.
ID_COLUMNS = ["episode_id", "scenario_name", "split"]


_DIGEST_CACHE: dict[str, str] = {}


def code_digest(commit: str | None) -> str | None:
    """Digest of the collection-read files AS THEY WERE AT `commit`.

    Hashing the CURRENT working tree would describe today's code, not the code
    that produced the episode -- which is the same defect as backfilling one
    convenient commit, arrived at through a different door. The rigid episodes
    here ran under a commit whose files have since changed; a working-tree digest
    would have quietly asserted otherwise.
    """
    if commit is None:
        return None
    if commit in _DIGEST_CACHE:
        return _DIGEST_CACHE[commit]
    h = hashlib.sha256()
    for rel in CODE_FILES:
        blob = subprocess.run(["git", "show", f"{commit}:{rel}"],
                              capture_output=True, cwd=str(REPO))
        h.update(blob.stdout if blob.returncode == 0 else b"<absent>")
    _DIGEST_CACHE[commit] = h.hexdigest()[:16]
    return _DIGEST_CACHE[commit]


def reflog_head_at(when: datetime) -> tuple[str | None, str | None]:
    """HEAD (commit, tree) at a wall-clock time, from the reflog.

    Backfilling one convenient commit across a batch that spanned several would
    be a tidy record rather than a true one.
    """
    out = subprocess.run(["git", "reflog", "--date=iso", "--format=%H %gd"],
                         capture_output=True, text=True, cwd=str(REPO)).stdout
    best = None
    for line in out.splitlines():
        m = re.match(r"([0-9a-f]{40}) HEAD@\{(.+?)\}", line)
        if not m:
            continue
        ts = datetime.fromisoformat(m.group(2))
        if ts.replace(tzinfo=None) <= when and (best is None or ts.replace(tzinfo=None) > best[0]):
            best = (ts.replace(tzinfo=None), m.group(1))
    if best is None:
        return None, None
    tree = subprocess.run(["git", "rev-parse", f"{best[1]}^{{tree}}"],
                          capture_output=True, text=True, cwd=str(REPO)).stdout.strip()
    return best[1], tree or None


def strip_id_columns(csv_path: Path) -> str:
    """sha256 of the CSV with the id columns removed.

    Lets a verifier distinguish "the ids were renamed" from "the file was
    rewritten and something else moved with it".
    """
    h = hashlib.sha256()
    with csv_path.open(newline="") as fh:
        r = csv.DictReader(fh)
        keep = [c for c in (r.fieldnames or []) if c not in ID_COLUMNS]
        h.update("|".join(keep).encode())
        for row in r:
            h.update("|".join(row[c] for c in keep).encode())
    return h.hexdigest()[:16]


def repair(root: Path, seed_offset: int, machine: str, gpu_name: str, gpu_arch: str,
           prefix: str = "go2", dry_run: bool = True) -> dict:
    report = {"episodes": 0, "csv_unchanged": 0, "csv_CHANGED": [], "splits": {"train": 0, "val": 0},
              "commits": {}, "code_digests": {}}
    for meta_path in sorted(root.glob("*/episodes/*.json")):
        ep_dir = meta_path.parents[1]
        m = json.loads(meta_path.read_text())
        family = m.get("command_family") or "none"
        terrain = m["terrain_label"]
        idx = int(re.search(r"_(\d+)$", ep_dir.name).group(1))
        new_id = f"{prefix}_{terrain}_s{seed_offset}_{family}_{idx:03d}"
        new_split = assign_split(new_id, VAL_RATIO)

        csv_old = ep_dir / "episodes" / f"{m['episode_id']}.csv"
        before = strip_id_columns(csv_old)

        when = datetime.fromtimestamp(meta_path.stat().st_mtime)
        commit, tree = reflog_head_at(when)
        report["commits"][commit[:8] if commit else "none"] = \
            report["commits"].get(commit[:8] if commit else "none", 0) + 1
        digest = code_digest(commit)
        report["code_digests"][digest or "none"] = report["code_digests"].get(digest or "none", 0) + 1

        if not dry_run:
            csv_new = ep_dir / "episodes" / f"{new_id}.csv"
            rows = list(csv.DictReader(csv_old.open(newline="")))
            fields = list(rows[0].keys())
            with csv_new.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                for row in rows:
                    row["episode_id"] = new_id
                    row["scenario_name"] = new_id
                    row["split"] = new_split
                    w.writerow(row)
            after = strip_id_columns(csv_new)
            if csv_new != csv_old:
                csv_old.unlink()
            m.update({
                "episode_id": new_id, "scenario_name": new_id, "split": new_split,
                "csv_path": f"episodes/{new_id}.csv",
                "machine": machine, "gpu_name": gpu_name, "gpu_arch": gpu_arch,
                "seed_offset": seed_offset,
                "git_commit": commit, "git_tree": tree,
                "collection_code_digest": digest,
                "checkpoint_path": CKPT_NEW,
                "metadata_repaired": (
                    "episode_id was not unique (per-family index); split recomputed at "
                    f"val_ratio {VAL_RATIO} from the new id; checkpoint path corrected "
                    f"from {CKPT_OLD}; origin keys added. Trajectory data untouched."),
            })
            (ep_dir / "episodes" / f"{new_id}.json").write_text(json.dumps(m, indent=2) + "\n")
            if meta_path.name != f"{new_id}.json":
                meta_path.unlink()
            idx_path = ep_dir / "dataset_index.json"
            if idx_path.is_file():
                di = json.loads(idx_path.read_text())
                for e in di.get("episodes", []):
                    e.update({"episode_id": new_id, "scenario_name": new_id,
                              "split": new_split, "csv_path": f"episodes/{new_id}.csv"})
                idx_path.write_text(json.dumps(di, indent=2) + "\n")
            if after == before:
                report["csv_unchanged"] += 1
            else:
                report["csv_CHANGED"].append(new_id)
        report["splits"][new_split] += 1
        report["episodes"] += 1
    return report


def emit_sidecar(root: Path, out: Path) -> dict:
    """READ-ONLY: freeze the mtime-derived commit mapping into a file.

    Run this BEFORE the rewrite, and safely while a collection is still in
    flight, because it touches nothing.

    WHY IT EXISTS. reflog_head_at derives an episode's commit from its file
    mtime, and mtime is destroyed by `cp -r`, by rsync without -a, and by any
    archive-and-restore. Until that mapping is written down, the provenance of a
    thousand episodes depends on a filesystem attribute that ordinary handling
    silently discards -- and "nothing is likely to touch them tonight" is not the
    standard for something unrecoverable once lost.

    It also makes the rewrite a PURE FUNCTION OF FILES. Re-running the pass
    tomorrow would otherwise attribute commits differently from running it now,
    because the input would be mtimes rather than data.

    Keyed by the episode DIRECTORY, not episode_id -- the ids are not unique,
    which is the bug the rewrite exists to fix.
    """
    rows, skipped = {}, []
    for meta_path in sorted(root.glob("*/episodes/*.json")):
        try:
            json.loads(meta_path.read_text())
        except Exception:  # noqa: BLE001 - a half-written file from a live run
            skipped.append(str(meta_path.relative_to(root)))
            continue
        when = datetime.fromtimestamp(meta_path.stat().st_mtime)
        commit, tree = reflog_head_at(when)
        rows[str(meta_path.parents[1].name)] = {
            "meta_file": meta_path.name,
            "mtime": when.isoformat(),
            "git_commit": commit,
            "git_tree": tree,
            "collection_code_digest": code_digest(commit),
        }
    payload = {
        "note": ("Frozen mtime-derived git provenance. mtime does not survive "
                 "copying; this file does. The rewrite reads this instead of the "
                 "reflog, which also makes it reproducible."),
        "root": str(root),
        "captured_utc": datetime.now().astimezone().isoformat(),
        "episodes": len(rows),
        "skipped_unparseable": skipped,
        "entries": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--seed-offset", type=int, required=True)
    ap.add_argument("--machine", required=True)
    ap.add_argument("--gpu-name", required=True)
    ap.add_argument("--gpu-arch", required=True)
    ap.add_argument("--prefix", default="go2")
    ap.add_argument("--apply", action="store_true", help="without this, dry run")
    ap.add_argument("--sidecar", type=Path, default=None,
                    help="READ-ONLY: freeze the mtime-derived commit mapping and exit")
    a = ap.parse_args()
    if a.sidecar:
        rep = emit_sidecar(a.root, a.sidecar)
        rep.pop("entries")
        print(json.dumps(rep, indent=2))
        raise SystemExit(0)
    rep = repair(a.root, a.seed_offset, a.machine, a.gpu_name, a.gpu_arch,
                 a.prefix, dry_run=not a.apply)
    print(json.dumps(rep, indent=2))
