#!/usr/bin/env python3
"""Seal the complete train/validation cohort and its exact H20 prefix ablation.

The full command refuses anything except 360 train routes, 90 validation routes,
and their 30 shared snapshots with valid atomic batch completion contracts.
Success refers to recording integrity, never to a vehicle's physical outcome.
No test observations, routes, or outcomes are opened.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import PIL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import traverse_fdm_rgbd_diverse_prepare as prep

HORIZON_KEYS = frozenset((
    "commands", "nominal_pose", "trajectory", "trajectory_mask", "work", "work_mask",
    "events", "event_mask", "bounded_motion", "bounded_motion_mask",
    "sustained_stall", "sustained_stall_mask", "asset_contact", "asset_contact_mask",
    "chassis_contact", "chassis_contact_mask", "attitude", "attitude_mask",
    "attitude_peak_abs", "attitude_peak_abs_mask"))
FIXED_KEYS = frozenset(("history", "global_features", "episode_index", "scene_index",
                        "image_index", "anchor", "source_domain"))


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected archived relative path: {value}")
    return path


def current_sources():
    paths = {Path(__file__).resolve()}
    paths.update(Path(module.__file__).resolve() for module in sys.modules.values()
        if getattr(module, "__file__", None) and Path(module.__file__).suffix == ".py"
        and Path(module.__file__).resolve().is_relative_to(ROOT))
    paths.add(ROOT / "slurm/traverse_fdm_rgbd_diverse_pack.sbatch")
    return {str(path.relative_to(ROOT)): prep.sha(path) for path in sorted(paths)}


def snapshot(out):
    sources = current_sources()
    out.mkdir(parents=True, exist_ok=False)
    for name, digest in sources.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        prep.require_hash(target, digest)
    prep.dump(out / "source_manifest.json", {"schema": "fdm_diverse_packing_source_v1",
        "scope": "All imported preparation/prefix/checker repository dependencies and Slurm entrypoint; collection archive remains separate and immutable",
        "files": sources})
    print(json.dumps({"snapshot": str(out), "files": len(sources),
                      "manifest_sha256": prep.sha(out / "source_manifest.json")}))


def full_records(manifest):
    records = json.loads(manifest.read_text())["records"]
    selected = [row for row in records if row["split"] in ("train", "val")]
    if len({row["scene_id"] for row in records}) != len(records):
        raise ValueError("Duplicate scene declaration")
    counts = {split: sum(row["split"] == split for row in selected) for split in ("train", "val")}
    if counts != {"train": 24, "val": 6}:
        raise ValueError(f"Full cohort requires 24 train + 6 val scenes: {counts}")
    if len({row["arena_bmp_sha256"] for row in selected}) != 30:
        raise ValueError("Full cohort requires 30 independently authored heightmaps")
    for row in selected:
        if len(row["routes"]) != 15 or len(row["route_sha256"]) != 15:
            raise ValueError("Full cohort requires every one of 15 declared routes per scene")
        if [Path(value).stem for value in row["routes"]] != [f"family_{i:02d}" for i in range(15)]:
            raise ValueError("Route inventory/order differs from the frozen full protocol")
    return selected


def validate_marker(path, expected, manifest_sha256):
    marker_path = path / "batch_complete.json"
    marker = json.loads(marker_path.read_text())
    common, task = marker["contract"]["common"], marker["contract"]["task"]
    if common.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"Batch declaration hash mismatch: {path}")
    for name, value in expected.items():
        if task.get(name) != value:
            raise ValueError(f"Task {name} differs from full cohort protocol: {path}")
    outputs = marker["output_sha256"]
    inventory = {str(item.relative_to(path)) for item in path.rglob("*")
                 if item.is_file() and item.name != "batch_complete.json"}
    if inventory != set(outputs):
        raise ValueError(f"Final directory inventory differs from its completion marker: {path}")
    for name, digest in outputs.items():
        prep.require_hash(path / safe_relative(name), digest)
    required = prep.REQUIRED_RAW if expected["stage"] == "collect" else prep.REQUIRED_OBS
    if not set(required).issubset(outputs):
        raise ValueError(f"Required completed data products are missing: {path}")
    provenance = json.loads((path / "simulation_provenance.json").read_text())
    if provenance["runtime_sha256"] != common["runtime"]["file_sha256"]:
        raise ValueError(f"Runtime contract differs from recording provenance: {path}")
    for name, digest in provenance["source_sha256"].items():
        if common["source_sha256"].get(name) != digest:
            raise ValueError(f"Recording source differs from batch source: {path}")
    return {"path": str(path), "marker_sha256": prep.sha(marker_path), "common": common,
            "bytes": sum((path / name).stat().st_size for name in outputs), "files": len(outputs)}


def preflight(manifest, collected, source_root, workers=8):
    records, tasks = full_records(manifest), []
    manifest_sha256 = prep.sha(manifest)
    for row in records:
        prep.require_hash(manifest.parent / safe_relative(row["case"]), row["case_sha256"])
        shared = {key: row[key] for key in ("scene_id", "split", "case_sha256", "arena_meta_sha256", "arena_bmp_sha256")}
        shared["case"] = str((manifest.parent / row["case"]).resolve())
        tasks.append((collected / "observations" / row["scene_id"],
                      shared | {"stage": "observe", "task_id": row["scene_id"] + "__observe"}))
        for index, (route, digest) in enumerate(zip(row["routes"], row["route_sha256"], strict=True)):
            prep.require_hash(manifest.parent / safe_relative(route), digest)
            tasks.append((collected / "raw" / row["scene_id"] / Path(route).stem,
                shared | {"stage": "collect", "route": str((manifest.parent / route).resolve()), "route_index": index,
                    "route_name": Path(route).stem, "task_id": row["scene_id"] + "__" + Path(route).stem,
                    "route_sha256": digest, "horizon_s": 180., "rich_telemetry": True, "render_parity": False}))
    missing = [str(path / "batch_complete.json") for path, _ in tasks if not (path / "batch_complete.json").is_file()]
    if missing:
        raise ValueError(f"Full collection is incomplete: {len(missing)} missing atomic markers; first={missing[0]}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        checked = list(pool.map(lambda task: validate_marker(*task, manifest_sha256), tasks))
    common = checked[0]["common"]
    for result in checked:
        if result.pop("common") != common:
            raise ValueError("The complete cohort mixes batch source/runtime contracts")
    for name, digest in common["source_sha256"].items():
        prep.require_hash(source_root / safe_relative(name), digest)
    report = {"schema": "fdm_diverse_full_integrity_v1", "complete": True,
        "manifest_sha256": manifest_sha256, "train_routes": 360, "val_routes": 90,
        "observations": 30, "atomic_markers_verified": len(checked),
        "test_data_opened": False, "outcome_filtering": False,
        "total_source_bytes": sum(item["bytes"] for item in checked),
        "common_contract": common, "outputs": checked}
    print(json.dumps({key: value for key, value in report.items() if key not in ("common_contract", "outputs")}), flush=True)
    return report


def validate_pack(pack):
    manifest = json.loads((pack / "manifest.json").read_text())
    if set(manifest["splits"]) != {"train", "val"} or "test" in manifest["scene_splits"].values():
        raise ValueError("Only train/validation packs are allowed")
    prep.require_hash(pack / "normalization.json", manifest["normalization_sha256"])
    for split, meta in manifest["splits"].items():
        for name, digest in ((f"{split}.npz", meta["sha256"]),
            (meta["rgbd_file"], meta["rgbd_sha256"]), (f"{split}_episodes.json", meta["episodes_sha256"])):
            prep.require_hash(pack / name, digest)
    return manifest


def derive_prefix(source, out, horizon=20):
    parent = validate_pack(source)
    if not 0 < horizon < parent["horizon_steps"]:
        raise ValueError("Derived horizon must be a strict measured prefix")
    out.mkdir(parents=True, exist_ok=False)
    manifest = copy.deepcopy(parent)
    manifest.update(horizon_steps=horizon, horizon_s=horizon*parent["output_dt_s"],
        horizon_derivation={"method": "Exact prefix of every H60 future tensor; identical anchors, histories, scene images and episode metadata",
            "source": str(source.resolve()), "manifest_sha256": prep.sha(source / "manifest.json"),
            "command_normalization": "Recomputed on the shorter train command prefix only; history/global normalization unchanged"},
        derivation_sha256=current_sources())
    normalization = json.loads((source / "normalization.json").read_text())
    for split, meta in manifest["splits"].items():
        for name in (meta["rgbd_file"], f"{split}_episodes.json"):
            shutil.copyfile(source / name, out / name)
        with np.load(source / f"{split}.npz", allow_pickle=False) as archive, zipfile.ZipFile(out / f"{split}.npz", "w", compression=zipfile.ZIP_STORED, allowZip64=True) as target:
            if set(archive.files) != HORIZON_KEYS | FIXED_KEYS:
                raise ValueError(f"Unreviewed tensor schema for prefix derivation: {set(archive.files) ^ (HORIZON_KEYS | FIXED_KEYS)}")
            for key in archive.files:
                value = archive[key]
                if key in HORIZON_KEYS:
                    if value.shape[1] != parent["horizon_steps"]:
                        raise ValueError(f"Wrong temporal dimension for {key}")
                    value = np.ascontiguousarray(value[:, :horizon])
                if not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite derived tensor: {key}")
                with target.open(key + ".npy", "w", force_zip64=True) as entry:
                    np.lib.format.write_array(entry, value, allow_pickle=False)
                meta["shapes"][key] = list(value.shape)
                if split == "train" and key == "commands":
                    flat = value.reshape(-1, value.shape[-1])
                    normalization["commands"] = {"mean": flat.mean(0, dtype=np.float64).tolist(),
                        "std": np.maximum(flat.std(0, dtype=np.float64), .01).tolist()}
        with np.load(out / f"{split}.npz", allow_pickle=False) as archive:
            episodes = json.loads((out / f"{split}_episodes.json").read_text())
            meta["event_support"] = prep.event_support(archive, episodes)
        meta["sha256"] = prep.sha(out / f"{split}.npz")
    prep.dump(out / "normalization.json", normalization)
    manifest["normalization_sha256"] = prep.sha(out / "normalization.json")
    prep.dump(out / "manifest.json", manifest)
    return verify_pair(source, out)


def verify_pair(long_pack, short_pack):
    longer, shorter = validate_pack(long_pack), validate_pack(short_pack)
    short_h = shorter["horizon_steps"]
    report = {"schema": "fdm_diverse_horizon_pair_v1", "complete": True,
        "long_manifest_sha256": prep.sha(long_pack / "manifest.json"),
        "short_manifest_sha256": prep.sha(short_pack / "manifest.json"), "splits": {}}
    for split, meta in longer["splits"].items():
        other = shorter["splits"][split]
        for key in ("windows", "episodes", "scenes", "scene_ids", "rgbd_sha256", "episodes_sha256"):
            if meta[key] != other[key]:
                raise ValueError(f"Unmatched horizon cohorts: {split}/{key}")
        checked = {}
        with np.load(long_pack / f"{split}.npz", allow_pickle=False) as a, np.load(short_pack / f"{split}.npz", allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise ValueError("Unmatched tensor inventories")
            for key in a.files:
                left, right = a[key], b[key]
                if key in HORIZON_KEYS:
                    left = left[:, :short_h]
                if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
                    raise ValueError(f"Horizon prefix changed values: {split}/{key}")
                checked[key] = {"exact": True, "shape": list(right.shape)}
        report["splits"][split] = {"windows": meta["windows"], "episodes": meta["episodes"],
            "scenes": meta["scenes"], "images_identical": True, "episodes_identical": True, "tensors": checked}
    norms = [json.loads((path / "normalization.json").read_text()) for path in (long_pack, short_pack)]
    for key in ("history", "global_features"):
        if norms[0][key] != norms[1][key]:
            raise ValueError("Horizon-independent normalization changed")
    report["normalization"] = "Identical history/global; command statistics recomputed from each horizon's same training anchors"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot")
    snap.add_argument("--out", type=Path, required=True)
    prefix = commands.add_parser("prefix")
    prefix.add_argument("--source", type=Path, required=True)
    prefix.add_argument("--out", type=Path, required=True)
    full = commands.add_parser("full")
    full.add_argument("--manifest", type=Path, required=True)
    full.add_argument("--collected", type=Path, required=True)
    full.add_argument("--source-snapshot-root", type=Path, required=True)
    full.add_argument("--out", type=Path, required=True)
    full.add_argument("--audit-only", action="store_true")
    full.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "snapshot":
        return snapshot(args.out)
    if args.command == "prefix":
        report = derive_prefix(args.source, args.out)
        prep.dump(args.out / "horizon_pair_validation.json", report)
        print(json.dumps(report))
        return
    started = time.monotonic()
    if args.out.exists():
        raise ValueError("Packing uses a fresh output; existing packs are never overwritten")
    archived = json.loads((ROOT / "source_manifest.json").read_text())
    if archived["files"] != current_sources():
        raise ValueError("Packing must run from the complete immutable preparation snapshot")
    integrity = preflight(args.manifest, args.collected, args.source_snapshot_root, args.workers)
    args.out.mkdir(parents=True, exist_ok=False)
    prep.dump(args.out / "collection_integrity.json", integrity)
    if args.audit_only:
        return
    long_pack, short_pack = args.out / "h60", args.out / "h20"
    subprocess.run([sys.executable, "-u", str(ROOT / "scripts/traverse_fdm_rgbd_diverse_prepare.py"),
        "--cohort-manifest", str(args.manifest), "--raw-root", str(args.collected / "raw"),
        "--observation-root", str(args.collected / "observations"),
        "--source-snapshot-root", str(args.source_snapshot_root), "--arena-root", str(args.source_snapshot_root),
        "--out", str(long_pack), "--horizon", "60", "--workers", str(args.workers)], check=True)
    pair = derive_prefix(long_pack, short_pack)
    for split, expected in (("train", (360, 24)), ("val", (90, 6))):
        result = pair["splits"][split]
        if (result["episodes"], result["scenes"]) != expected:
            raise ValueError("Sealed pack counts differ from full cohort")
    if archived["files"] != current_sources():
        raise ValueError("Preparation snapshot changed during packing")
    prep.dump(args.out / "horizon_pair_validation.json", pair)
    status = {"schema": "fdm_diverse_pack_pair_complete_v1", "complete": True,
        "collection_integrity_sha256": prep.sha(args.out / "collection_integrity.json"),
        "horizon_pair_validation_sha256": prep.sha(args.out / "horizon_pair_validation.json"),
        "long_manifest_sha256": pair["long_manifest_sha256"], "short_manifest_sha256": pair["short_manifest_sha256"],
        "source_snapshot": str(ROOT), "source_manifest_sha256": prep.sha(ROOT / "source_manifest.json"),
        "wall_s": time.monotonic()-started, "peak_self_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_preparation_subprocess_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "runtime": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__, "pillow": PIL.__version__},
        "test_data_opened": False, "outcome_filtering": False}
    prep.dump(args.out / "pair_complete.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
