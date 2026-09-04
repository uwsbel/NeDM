"""Repair + consolidate a Go2 collection into the layout preprocess expects.

WHY CONSOLIDATION IS PART OF THE REPAIR AND NOT A SEPARATE STEP. The collector
writes ONE ROOT PER EPISODE -- each with its own dataset_index.json listing a
single episode -- because it was built to run one episode per process. preprocess
reads dataset_root/dataset_index.json and merges roots, so the collection as
written would need ~1,120 roots on one command line, and compute_dt_s would run
per root. The pipeline's expected shape is one root per terrain holding many
episodes, which is what the HMMWV collector produces.

Since the repair is already rewriting every episode id and moving every CSV, the
consolidation costs nothing extra here and avoids a second full pass.

WRITES TO A NEW ROOT. Originals are left untouched, so this is reversible and the
pre-repair data remains available for comparison. Disk is cheap; an irreversible
in-place rewrite of 1,120 episodes is not.
"""
from __future__ import annotations
import csv, hashlib, json, sys, shutil
from datetime import datetime, timezone
from pathlib import Path
# Derived from __file__, NOT hardcoded. The absolute path that was here named
# this box's checkout and made the script unimportable on dorm-pc, whose repo
# is at ~/sbel/NeDM -- so the one canonical repair script could not be run on
# half the dataset it is canonical for.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from nedm.hmmwv_data import assign_split

VAL_RATIO = 0.2
CKPT_NEW = "/home/kyle/sbel-artifacts/checkpoints/go2_cts_150k.pt"
ID_COLUMNS = ["episode_id", "scenario_name", "split"]

def strip_hash(path, fields=None):
    h = hashlib.sha256()
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        keep = [c for c in (r.fieldnames or []) if c not in ID_COLUMNS]
        h.update("|".join(keep).encode())
        for row in r:
            h.update("|".join(row[c] for c in keep).encode())
    return h.hexdigest()[:16]

def strip_rows_hash(rows, fields):
    """Same digest as strip_hash, over rows held in memory rather than a file."""
    h = hashlib.sha256()
    keep = [c for c in fields if c not in ID_COLUMNS]
    h.update("|".join(keep).encode())
    for row in rows:
        h.update("|".join(row[c] for c in keep).encode())
    return h.hexdigest()[:16]


def run(src: Path, out_root: Path, sidecar: Path, seed_offset: int,
        machine: str, gpu_name: str, gpu_arch: str, prefix="go2", apply=False):
    side = json.loads(sidecar.read_text())["entries"]
    report = {"per_terrain": {}, "commits": {}, "digests": {}, "csv_changed": [],
              "skipped_already_repaired": 0}
    groups = {}
    for meta in sorted(src.glob("*/episodes/*.json")):
        m = json.loads(meta.read_text())
        groups.setdefault(m["terrain_label"], []).append((meta, m))
    for terrain, items in groups.items():
        out = out_root / terrain
        if apply:
            (out / "episodes").mkdir(parents=True, exist_ok=True)
        entries, splits = [], {"train": 0, "val": 0}
        for meta, m in items:
            ep_dir = meta.parents[1].name
            # IDEMPOTENCE. The pass rewrites the JSON whose mtime the sidecar was
            # built from. Without this guard, a crash at episode 700 followed by a
            # re-run would recover today's HEAD for the first 700 and silently
            # replace their provenance with code that did not produce them -- the
            # exact defect this pass exists to prevent, through a door its own
            # docstring did not cover.
            if m.get("metadata_repaired"):
                report["skipped_already_repaired"] += 1
                continue
            fam = m.get("command_family") or "none"
            tail = ep_dir.rsplit("_", 1)
            if len(tail) != 2 or not tail[1].isdigit():
                raise ValueError(f"cannot parse episode index from {ep_dir!r}")
            idx = int(tail[1])
            new_id = f"{prefix}_{terrain}_s{seed_offset}_{fam}_{idx:03d}"
            split = assign_split(new_id, VAL_RATIO)
            splits[split] += 1
            sc = side.get(ep_dir, {})
            report["commits"][(sc.get("git_commit") or "none")[:8]] = \
                report["commits"].get((sc.get("git_commit") or "none")[:8], 0) + 1
            report["digests"][sc.get("collection_code_digest") or "none"] = \
                report["digests"].get(sc.get("collection_code_digest") or "none", 0) + 1
            csv_src = meta.parent / f"{m['episode_id']}.csv"
            # THE COMPARISON RUNS ON A DRY RUN TOO. Previously `before` was
            # computed inside `if apply:`, so a dry run reported csv_changed == []
            # because the check never happened -- a success path reachable without
            # the thing it checks having occurred. The rewritten rows are built in
            # memory and hashed either way; only the WRITE is conditional.
            rows = list(csv.DictReader(csv_src.open(newline="")))
            if not rows:
                raise ValueError(f"zero-row csv: {csv_src}")
            fields = list(rows[0].keys())
            before = strip_hash(csv_src)
            for row in rows:
                row["episode_id"] = new_id
                row["scenario_name"] = new_id
                row["split"] = split
            after = strip_rows_hash(rows, fields)
            if after != before:
                report["csv_changed"].append(new_id)
            if apply:
                dst = out / "episodes" / f"{new_id}.csv"
                with dst.open("w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=fields)
                    w.writeheader()
                    w.writerows(rows)
                m.update({
                    "episode_id": new_id, "scenario_name": new_id, "split": split,
                    # scenario_family was a CONSTANT -- "go2_crm_constant_command"
                    # for all eight families -- a true-looking field naming one
                    # family for eight. Not cosmetic: trainer._select_rollout_
                    # episodes round-robins across scenario_family to choose the
                    # rollout episodes that select every deployed checkpoint. One
                    # bucket means no balancing, so the twelve selection rollouts
                    # could all come from whichever family sorts first. Derived
                    # from the source directory, never from scenario_family.
                    "scenario_family": f"go2_{terrain}_{fam}_command",
                    "csv_path": f"episodes/{new_id}.csv",
                    "machine": machine, "gpu_name": gpu_name, "gpu_arch": gpu_arch,
                    "seed_offset": seed_offset,
                    "git_commit": sc.get("git_commit"), "git_tree": sc.get("git_tree"),
                    "collection_code_digest": sc.get("collection_code_digest"),
                    "checkpoint_path": CKPT_NEW,
                    "source_episode_dir": ep_dir,
                    "metadata_repaired": (
                        "episode_id was not unique (index ran per family); split "
                        f"recomputed at {VAL_RATIO} from the new id; git provenance "
                        "from the frozen sidecar, not re-derived from mtimes; "
                        "checkpoint path corrected. Trajectory data untouched."),
                })
                (out / "episodes" / f"{new_id}.json").write_text(json.dumps(m, indent=2) + "\n")
                # KEEP THE PER-EPISODE CONFIG. Not archival: patch_y_m lives ONLY
                # here, and it is 8.0 for lateral against 4.0 for every other
                # family, so dropping these would erase which episodes ran on the
                # wide bed. Named by episode id, so nothing collides at the root.
                cfg_src = meta.parents[1] / "collector_config.resolved.json"
                if cfg_src.is_file():
                    shutil.copy(cfg_src, out / "episodes" / f"{new_id}.config.json")
            # terminated_near_boundary: the Go2 collector never wrote it, so
            # build_combined_flat_crm_rl_references' DEFAULT-ON exclusion
            # (`if ep.get("terminated_near_boundary") and not allow_boundary`)
            # got None, and the filter never fired. The data existed all along in
            # the episode JSON's status; only the index copy was missing.
            # warmup_s: also missing against the HMMWV entry. No live consumer in
            # the training path, but the two schemas were meant to match and a
            # field absent for no reason is how the next one hides.
            boundary = m.get("status") == "bed_boundary"
            assert boundary == (m.get("bed_boundary_at_s") is not None), new_id
            entries.append({"episode_id": new_id, "scenario_name": new_id,
                            "scenario_family": f"go2_{terrain}_{fam}_command",
                            "warmup_s": m.get("warmup_s"),
                            "terminated_near_boundary": boundary,
                            "split": split, "csv_path": f"episodes/{new_id}.csv",
                            "rows": m["rows"], "duration_s": m["duration_s"],
                            "terrain_label": terrain})
        if apply:
            (out / "dataset_index.json").write_text(json.dumps({
                "dataset_name": f"go2_{terrain}_s{seed_offset}",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "episode_count": len(entries), "episodes": entries,
            }, indent=2) + "\n")
            # Root-level copy exists ONLY because compute_dt_s reads
            # record_step_s from it. It is a representative, not an authority:
            # terrain.patch_y_m differs per family, so read the per-episode
            # configs for anything terrain-specific.
            rep = None
            for meta_p, mm in items:
                c = meta_p.parents[1] / "collector_config.resolved.json"
                if c.is_file() and (mm.get("command_family") != "lateral" or rep is None):
                    rep = c
                    if mm.get("command_family") != "lateral":
                        break
            if rep:
                shutil.copy(rep, out / "collector_config.resolved.json")
            # The provenance note belongs WITH the data it explains. Each root
            # documents its own half's events; the roots are separate, so a
            # reader of either has the note that applies to it.
            note = src / "PROVENANCE_NOTE.md"
            if note.is_file():
                shutil.copy(note, out / "PROVENANCE_NOTE.md")
        report["per_terrain"][terrain] = {
            "episodes": len(entries), "splits": splits,
            "transitions": sum(e["rows"] for e in entries),
            "val_pct": round(100 * splits["val"] / max(len(entries), 1), 1),
            "distinct_ids": len({e["episode_id"] for e in entries}),
        }
    return report

if __name__ == "__main__":
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("src", type=Path); a.add_argument("out", type=Path)
    a.add_argument("--sidecar", type=Path, required=True)
    a.add_argument("--seed-offset", type=int, required=True)
    a.add_argument("--machine", required=True); a.add_argument("--gpu-name", required=True)
    a.add_argument("--gpu-arch", required=True); a.add_argument("--prefix", default="go2")
    a.add_argument("--apply", action="store_true")
    n = a.parse_args()
    print(json.dumps(run(n.src, n.out, n.sidecar, n.seed_offset, n.machine,
                         n.gpu_name, n.gpu_arch, n.prefix, n.apply), indent=2))
