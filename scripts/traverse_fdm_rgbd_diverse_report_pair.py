#!/usr/bin/env python3
"""CPU-only frozen H60/H20 validation evidence and small metadata export."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import traverse_fdm_rgbd_diverse_report as report

ARMS = (("rgbd", 11), ("blank", 11), ("rgbd", 29), ("blank", 29))
MATCH_ARGS = ("seed", "steps", "batch", "learning_rate", "weight_decay", "positive_sampling_fraction",
    "max_event_pos_weight", "candidate_patches", "rotation_augmentation", "progress_target",
    "xy_weight", "yaw_weight", "work_weight", "event_weight", "attitude_weight", "eval_every")


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sources():
    files = {Path(__file__).resolve(), ROOT / "slurm/traverse_fdm_rgbd_diverse_report.sbatch"}
    files.update(Path(module.__file__).resolve() for module in sys.modules.values()
        if getattr(module, "__file__", None) and Path(module.__file__).suffix == ".py"
        and Path(module.__file__).resolve().is_relative_to(ROOT))
    return {str(path.relative_to(ROOT)): report.sha(path) for path in sorted(files)}


def freeze(out):
    hashes = sources()
    out.mkdir(parents=True, exist_ok=False)
    for name, digest in hashes.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        report.require_hash(target, digest)
    dump(out / "source_manifest.json", {"schema": "fdm_diverse_report_source_v1", "files": hashes,
        "scope": "All imported repository metric/target/report dependencies and CPU entrypoint; no trainer, model, Torch, or test-data dependency"})
    print(json.dumps({"files": len(hashes), "manifest_sha256": report.sha(out / "source_manifest.json")}))


def differences(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return {key: differences(left[key], right[key]) for key in left.keys() & right.keys()}
    if type(left) in (int, float) and type(right) in (int, float):
        return left-right
    return None


def compare(longer, shorter):
    if (longer["scene_count"], longer["episode_count"], longer["window_count"]) != (shorter["scene_count"], shorter["episode_count"], shorter["window_count"]):
        raise ValueError("Horizon reports do not use identical validation cohorts")
    result = {"schema": "fdm_diverse_horizon_comparison_v1", "endpoint_s": 4,
        "comparison": "H60 first4s versus H20 full4s at identical measured anchors; best and fixed-final checkpoints separate",
        "primary_planning_model_predeclared": "H60 rgbd seed11 validation-best; no seed chosen from protected test",
        "limitations": ["A horizon ablation also changes the number of supervised future labels and command normalization; shared prefixes and anchor draws are checked.",
            "Overlapping windows are not independent trials; per-scene and pre-onset counts accompany risk metrics.",
            "Known-failure recognition does not demonstrate pre-onset route discrimination or closed-loop success."], "pairs": {}}
    for name, first in longer["runs"].items():
        second = shorter["runs"].get(name, {})
        if not first.get("available") or not second.get("available"):
            raise ValueError(f"Incomplete trained pair: {name}")
        mismatches = [key for key in MATCH_ARGS if first["arguments"].get(key) != second["arguments"].get(key)]
        pair = {"arm": first["arm"], "seed": first["seed"], "setting_mismatches": mismatches,
            "supported_events_h60": first["supported_events"], "supported_events_h20": second["supported_events"],
            "training_event_counts_h60": first["training_event_counts"], "training_event_counts_h20": second["training_event_counts"], "checkpoints": {}}
        for label in ("last", "best"):
            a, b = first["checkpoints"][label], second["checkpoints"][label]
            if not a.get("available") or not b.get("available"):
                raise ValueError("Checkpoint exports not complete")
            matched_draws = a["step"] == b["step"] and a.get("draw_digest") == b.get("draw_digest")
            entry = {"step_h60": a["step"], "step_h20": b["step"],
                "same_update_and_draw_digest": matched_draws,
                "same_augmentation_digest": a.get("augmentation_digest") == b.get("augmentation_digest"),
                "controlled_fixed_budget": label == "last" and not mismatches and matched_draws,
                "controls": {}}
            for control in ("normal", "blank", "shuffle"):
                x, y = a["controls"].get(control, {}), b["controls"].get(control, {})
                if not x.get("available") or not y.get("available"):
                    entry["controls"][control] = {"available": False, "reason_h60": x.get("reason"), "reason_h20": y.get("reason")}
                    continue
                x, y = x["horizons"]["4s"], y["horizons"]["4s"]
                entry["controls"][control] = {"available": True, "h60": x, "h20": y, "deltas_h60_minus_h20": differences(x, y)}
            pair["checkpoints"][label] = entry
        result["pairs"][name] = pair
    return result


def markdown(value):
    lines = ["# Matched forecast horizon comparison", "", value["comparison"], "",
        "Primary planning model fixed before full training: H60 RGB-D, seed11, validation-best.", "",
        "| Arm | Seed | Checkpoint | Group | H60 FDE m | H20 FDE m | H60 work MAE kJ | H20 work MAE kJ |", "|---|---:|---|---|---:|---:|---:|---:|"]
    fmt = lambda x: "—" if x is None else f"{x:.3f}"
    for pair in value["pairs"].values():
        for label, checkpoint in pair["checkpoints"].items():
            normal = checkpoint["controls"]["normal"]
            for group in ("all", "pre_first_observed_failure", "prior_observed_failure"):
                a, b = normal["h60"]["groups"].get(group), normal["h20"]["groups"].get(group)
                if a is None or b is None:
                    continue
                lines.append(f"| {pair['arm']} | {pair['seed']} | {label} | {group} | {fmt(a['motion']['endpoint_fde_m'])} | {fmt(b['motion']['endpoint_fde_m'])} | {fmt(a['work_kj']['mae'])} | {fmt(b['work_kj']['mae'])} |")
    lines += ["", "Per-scene causal strata, supported hazard counts, endpoint/prefix risk metrics, signed attitude errors, and image controls are retained in comparison.json and each horizon's report.json.", "",
              "These are validation forecast measurements. Full-route completion, failure, time, and mechanical work require the separate Chrono planner evaluation.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run")
    for key in ("pair", "runs-h60", "runs-h20", "raw-root", "out"):
        run.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        return freeze(args.out)
    started = time.monotonic()
    source_manifest = report.read_json(ROOT / "source_manifest.json")
    if sources() != source_manifest["files"]:
        raise ValueError("Run reports from the complete immutable source snapshot")
    complete = report.read_json(args.pair / "pair_complete.json")
    if not complete.get("complete"):
        raise ValueError("Matched data pair has not completed")
    report.require_hash(args.pair / "horizon_pair_validation.json", complete["horizon_pair_validation_sha256"])
    for horizon, field in (("h60", "long_manifest_sha256"), ("h20", "short_manifest_sha256")):
        report.require_hash(args.pair / horizon / "manifest.json", complete[field])
        run_root = getattr(args, "runs_"+horizon)
        for arm, seed in ARMS:
            status = report.read_json(run_root / f"patch_{arm}_s{seed}" / "status.json")
            if status.get("state") != "complete" or status.get("step") != 5000:
                raise ValueError(f"The declared 5000-update run is incomplete: {horizon}/{arm}/{seed}")
    args.out.mkdir(parents=True, exist_ok=False)
    def one(horizon):
        command = [sys.executable, "-u", str(ROOT / "scripts/traverse_fdm_rgbd_diverse_report.py"),
            "--pack", str(args.pair / horizon), "--raw-root", str(args.raw_root), "--out", str(args.out / horizon)]
        for arm, seed in ARMS:
            command += ["--run", f"{arm}_s{seed}=" + str(getattr(args, "runs_"+horizon) / f"patch_{arm}_s{seed}")]
        subprocess.run(command, check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(one, ("h60", "h20")))
    both = {horizon: report.read_json(args.out / horizon / "report.json") for horizon in ("h60", "h20")}
    comparison = compare(both["h60"], both["h20"])
    dump(args.out / "comparison.json", comparison)
    (args.out / "comparison.md").write_text(markdown(comparison))
    for horizon, result in both.items():
        metadata = args.out / "pack_metadata" / horizon
        metadata.mkdir(parents=True)
        for name in ("manifest.json", "normalization.json", "train_episodes.json", "val_episodes.json"):
            shutil.copyfile(args.pair / horizon / name, metadata / name)
        for name, entry in result["runs"].items():
            target = args.out / "run_metadata" / horizon / name
            target.mkdir(parents=True)
            source = Path(entry["path"])
            for file in ("config.json", "normalization.json", "provenance.json", "status.json", "metrics_best.json", "metrics_last.json", "best_controls.json", "last_controls.json", "array_task_result.json", "array_task_provenance.json"):
                shutil.copyfile(source / file, target / file)
            dump(target / "checkpoint_sha256.json", {label+".pt": entry["checkpoints"][label]["checkpoint_sha256"] for label in ("best", "last")})
    if sources() != source_manifest["files"]:
        raise ValueError("Reporter source changed during execution")
    dump(args.out / "complete.json", {"complete": True, "source_manifest_sha256": report.sha(ROOT / "source_manifest.json"),
        "comparison_sha256": report.sha(args.out / "comparison.json"), "wall_s": time.monotonic()-started,
        "test_data_opened": False, "training_or_inference_performed": False,
        "h60_report_sha256": report.sha(args.out / "h60/report.json"), "h20_report_sha256": report.sha(args.out / "h20/report.json")})
    print(json.dumps({"complete": True, "out": str(args.out)}))


if __name__ == "__main__":
    main()
