#!/usr/bin/env python3
"""Report frozen diverse FDM predictions, with no training or test-data access.

Compare fixed-budget last and validation-selected best separately. Validation
windows overlap: scene/episode support and equal-scene summaries are reported,
not window-level significance claims. Only saved predictions are inspected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from nedm.traverse.fdm_diverse_targets import prepare_episode_labels
from nedm.traverse.fdm_report_metrics import binary_metrics

ARMS = ("rgbd", "rgb_only", "depth_only", "blank")
EVENT_NAMES = ("contact", "rollover", "progress_event")
THRESHOLDS = (.35, .35, .5)
PREDICTION_KEYS = ("trajectory", "work", "event_logits", "attitude")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key].copy() for key in f.files}


def require_hash(path, digest):
    if sha(path) != digest:
        raise ValueError(f"Frozen bytes changed: {path}")


def load_pack(path, split):
    manifest = read_json(path / "manifest.json")
    if split not in ("train", "val") or split not in manifest["splits"]:
        raise ValueError("This report reads declared train/validation only; no test-data path is supported")
    entry = manifest["splits"][split]
    require_hash(path / f"{split}.npz", entry["sha256"])
    require_hash(path / f"{split}_rgbd.npy", entry["rgbd_sha256"])
    require_hash(path / f"{split}_episodes.json", entry["episodes_sha256"])
    require_hash(path / "normalization.json", manifest["normalization_sha256"])
    data = read_npz(path / f"{split}.npz")
    episodes = read_json(path / f"{split}_episodes.json")
    if len(episodes) != entry["episodes"] or len(data["anchor"]) != entry["windows"]:
        raise ValueError("Sealed manifest counts do not match data")
    if any(record["split"] != split for record in episodes):
        raise ValueError("Episode list crosses declared split")
    for key, shape in entry["shapes"].items():
        if list(data[key].shape) != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Bad sealed tensor: {key}")
    return manifest, data, episodes


def causal_groups(data, episodes, raw_root=None):
    """Distinguish new consequences from failures already observed at anchor."""
    count = len(data["anchor"])
    groups = {"all": np.ones(count, bool), "anchor0": data["anchor"] == 0}
    known = np.zeros(count, bool)
    ever = {key: np.zeros(count, bool) for key in ("contact", "rollover", "bounded", "sustained")}
    current = {key: np.zeros(count, bool) for key in ("bounded", "sustained")}
    sources, unavailable = [], []
    for index, record in enumerate(episodes):
        source = (raw_root / record["scene_id"] / Path(record["source"]).name) if raw_root else Path(record["source"])
        names = ("trajectory.npz", "rich_intervals.npz", "collection_meta.json", "outcome.json")
        if not all((source / name).is_file() for name in names):
            unavailable.append(record["id"])
            continue
        for name in names:
            require_hash(source / name, record["source_sha256"][name])
        raw, rich = read_npz(source / "trajectory.npz"), read_npz(source / "rich_intervals.npz")
        metadata, outcome = read_json(source / "collection_meta.json"), read_json(source / "outcome.json")
        pose = np.vstack([raw["pose"], raw["terminal_pose"]])
        state = np.vstack([raw["state"], raw["terminal_state"]])
        parked = np.r_[raw["parked"].astype(bool), bool(raw["terminal_parked"])]
        ep = prepare_episode_labels(pose, state, raw["action"], parked, rich,
            goal_xy=np.asarray(metadata["route"]["waypoints"][-1]), goal_radius_m=outcome["goal_radius_m"])
        rows = np.flatnonzero(data["episode_index"] == index)
        anchors = data["anchor"][rows]
        # Interval i is only a previously observed contact after endpoint i+1.
        endpoint_events = {
            "contact": np.r_[False, np.any(ep["contact"] > 1., axis=1)],
            "rollover": np.r_[False, np.any(ep["attitude_peaks"] > np.deg2rad(60.), axis=1)],
            "bounded": ep["bounded_endpoints"], "sustained": ep["sustained_endpoints"],
        }
        for key, endpoints in endpoint_events.items():
            ever[key][rows] = np.maximum.accumulate(endpoints)[anchors]
            if key in current:
                current[key][rows] = endpoints[anchors]
        known[rows] = True
        sources.append({"episode": record["id"], "source": str(source), "sha256": {name: record["source_sha256"][name] for name in names}})
    any_prior = np.any(np.stack(list(ever.values())), axis=0)
    groups["pre_first_observed_failure"] = known & ~any_prior
    groups["prior_observed_failure"] = known & any_prior
    groups["currently_bounded"] = known & current["bounded"]
    groups["currently_sustained_stall"] = known & current["sustained"]
    for key in ("contact", "bounded", "sustained"):
        groups[f"before_first_{key}"] = known & ~ever[key]
    return groups, {"known_windows": int(known.sum()), "unavailable_episodes": unavailable, "sources": sources,
        "definitions": {"pre_first_observed_failure": "No completed contact, rollover, bounded-motion or sustained-stall event at or before anchor; never-event episodes included.",
            "prior_observed_failure": "At least one failure event already completed; does not imply currently immobile.",
            "currently_bounded": "The two-second diameter/effort window ending exactly at anchor is positive.",
            "currently_sustained_stall": "The two-second low-speed/effort window ending exactly at anchor is positive.",
            "scope": "Strata use only the measured prefix; membership is not restricted to eventually positive routes. Missing raw sources make these strata unavailable, not negative."}}


def select_targets(data, target):
    result = dict(data)
    if target == "net_progress":
        return result
    if target not in ("bounded_motion", "sustained_stall"):
        raise ValueError(f"Unknown checkpoint progress target: {target}")
    result["events"] = data["events"].copy()
    result["event_mask"] = data["event_mask"].copy()
    result["events"][..., 2] = data[target][..., 0]
    result["event_mask"][..., 2] = data[target+"_mask"][..., 0]
    return result


def error_summary(error, truth=None, prediction=None):
    error = np.asarray(error, float)
    if not len(error):
        return {"count": 0, "mae": None, "rmse": None, "bias": None, "p90_absolute_error": None}
    result = {"count": len(error), "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
              "bias": float(error.mean()), "p90_absolute_error": float(np.quantile(np.abs(error), .9))}
    if truth is not None:
        result.update(truth_mean=float(np.mean(truth)), prediction_mean=float(np.mean(prediction)),
            weighted_absolute_percentage_error=float(np.abs(error).sum()/np.abs(truth).sum()) if np.abs(truth).sum() > 1e-9 else None)
    return result


def hazard_metrics(label, probability, episode_ids, scene_ids, supported, threshold):
    label, probability = np.asarray(label, int), np.asarray(probability, float)
    counts = {"known_windows": len(label), "positive_windows": int(label.sum()), "negative_windows": int((label == 0).sum()),
        "known_episodes": len(np.unique(episode_ids)), "positive_episodes": len(np.unique(episode_ids[label > 0])),
        "known_scenes": len(np.unique(scene_ids)), "positive_scenes": len(np.unique(scene_ids[label > 0])),
        "training_supported": bool(supported)}
    if not supported:
        return {**counts, "available": False, "reason": "Training lacks positive or negative labels for this head; raw logits are not an operational risk estimate."}
    metrics = binary_metrics(label, probability)
    accepted = probability <= threshold
    positive, negative = label > 0, label == 0
    metrics.pop("calibration_bins", None)
    metrics["planner_threshold"] = {"threshold": threshold, "comparison": "probability <= threshold accepted", "accepted": int(accepted.sum()),
        "false_accepts": int((accepted & positive).sum()), "false_rejects": int((~accepted & negative).sum()),
        "failure_recall": float((~accepted)[positive].mean()) if positive.any() else None,
        "negative_retention": float(accepted[negative].mean()) if negative.any() else None,
        "accepted_failure_fraction": float(positive[accepted].mean()) if accepted.any() else None}
    return {**counts, "available": True, "metrics": metrics,
        "scarce_positive_support": counts["positive_episodes"] < 10 or counts["positive_scenes"] < 3}


def summarize_group(prediction, data, chosen, index, dt, supported):
    end = index+1
    valid = (data["trajectory_mask"][:, :end, 0] > 0) & chosen[:, None]
    complete = chosen & (data["trajectory_mask"][:, index, 0] > 0)
    xy_error = np.linalg.norm(prediction["trajectory"][:, :end, :2]-data["trajectory"][:, :end, :2], axis=-1)
    predicted_yaw = np.arctan2(prediction["trajectory"][:, :end, 2], prediction["trajectory"][:, :end, 3])
    true_yaw = np.arctan2(data["trajectory"][:, :end, 2], data["trajectory"][:, :end, 3])
    yaw_delta = np.degrees(np.arctan2(np.sin(predicted_yaw-true_yaw), np.cos(predicted_yaw-true_yaw)))
    result = {"windows": int(chosen.sum()), "episodes": len(np.unique(data["episode_index"][chosen])),
        "scenes": len(np.unique(data["scene_index"][chosen])),
        "motion": {"observed_prefix_points": int(valid.sum()), "complete_horizon_windows": int(complete.sum()),
            "observed_prefix_ade_m": float(xy_error[valid].mean()) if valid.any() else None,
            "complete_prefix_ade_m": float(xy_error[complete].mean()) if complete.any() else None,
            "endpoint_fde_m": float(xy_error[complete, index].mean()) if complete.any() else None,
            "endpoint_fde_p90_m": float(np.quantile(xy_error[complete, index], .9)) if complete.any() else None,
            "endpoint_yaw_error_deg": error_summary(yaw_delta[complete, index])}}
    work_valid = chosen & (data.get("work_mask", data["trajectory_mask"])[:, index, 0] > 0)
    truth, predicted = data["work"][work_valid, index, 0], prediction["work"][work_valid, index, 0]
    result["work_kj"] = error_summary(predicted-truth, truth, predicted)
    result["attitude_deg"] = {}
    for axis, name in enumerate(("roll", "pitch")):
        selected = chosen & (data["attitude_mask"][:, index, axis] > 0)
        delta = prediction["attitude"][selected, index, axis]-data["attitude"][selected, index, axis]
        result["attitude_deg"][name] = error_summary(np.degrees(np.arctan2(np.sin(delta), np.cos(delta))))
        if "attitude_peak_abs" in data:
            known = chosen & (data["attitude_peak_abs_mask"][:, index, axis] > 0)
            forecast_peak = np.max(np.abs(prediction["attitude"][known, :end, axis]), axis=1)
            measured_peak = data["attitude_peak_abs"][known, index, axis]
            error = np.degrees(forecast_peak-measured_peak)
            result["attitude_deg"][name+"_endpoint_forecast_peak_vs_solver_peak"] = error_summary(error, np.degrees(measured_peak), np.degrees(forecast_peak))
    probability = 1./(1.+np.exp(-np.clip(prediction["event_logits"][:, :end], -80., 80.)))
    result["events"] = {}
    for event, name in enumerate(EVENT_NAMES):
        known = chosen & (data["event_mask"][:, index, event] > 0)
        label = data["events"][known, index, event]
        ep, scene = data["episode_index"][known], data["scene_index"][known]
        eligible = np.ones(end, bool) if event != 2 else np.arange(1, end+1)*dt >= 2.-1e-6
        peak = np.where(eligible[None], probability[:, :, event], 0.).max(1)
        result["events"][name] = {"horizon_probability": hazard_metrics(label, probability[known, index, event], ep, scene, supported[event], THRESHOLDS[event]),
            "planner_prefix_max_probability": hazard_metrics(label, peak[known], ep, scene, supported[event], THRESHOLDS[event])}
    return result


def summarize(prediction, data, groups, manifest, supported, scene_names):
    for key, target in (("trajectory", "trajectory"), ("work", "work"), ("event_logits", "events"), ("attitude", "attitude")):
        if key not in prediction or prediction[key].shape != data[target].shape or not np.isfinite(prediction[key]).all():
            raise ValueError(f"Prediction shape/finite check failed: {key}")
    dt, horizon = float(manifest["output_dt_s"]), data["trajectory"].shape[1]
    horizons = {}
    for seconds in (4., 8., 12.):
        index = int(round(seconds/dt))-1
        if index >= horizon or not np.isclose((index+1)*dt, seconds):
            horizons[f"{seconds:g}s"] = {"available": False, "reason": "Beyond this checkpoint's declared forecast horizon"}
            continue
        scopes = {name: summarize_group(prediction, data, group, index, dt, supported) for name, group in groups.items()}
        scenes = {scene_names[int(scene)]: summarize_group(prediction, data, data["scene_index"] == scene, index, dt, supported)
                  for scene in np.unique(data["scene_index"])}
        scene_causal = {scene_names[int(scene)]: {
            name: summarize_group(prediction, data, (data["scene_index"] == scene) & groups[name], index, dt, supported)
            for name in ("anchor0", "pre_first_observed_failure", "prior_observed_failure") if name in groups}
            for scene in np.unique(data["scene_index"])}
        def macro(section, metric):
            values = [entry[section][metric] for entry in scenes.values() if entry[section][metric] is not None]
            return {"mean": float(np.mean(values)) if values else None, "scenes_with_metric": len(values)}
        event_macro = {}
        for event in EVENT_NAMES:
            event_macro[event] = {}
            for metric in ("brier", "auroc"):
                values = [entry["events"][event]["planner_prefix_max_probability"].get("metrics", {}).get(metric) for entry in scenes.values()]
                values = [value for value in values if value is not None]
                event_macro[event][metric] = {"mean": float(np.mean(values)) if values else None, "scenes_with_metric": len(values)}
        horizons[f"{seconds:g}s"] = {"available": True, "groups": scopes, "per_scene": scenes, "per_scene_causal": scene_causal,
            "equal_scene_macro": {"endpoint_fde_m": macro("motion", "endpoint_fde_m"), "work_mae_kj": macro("work_kj", "mae"), "events": event_macro}}
    return horizons


def read_run(name, path, pack, manifest, data, groups, split):
    required = ("config.json", "normalization.json", "provenance.json", "status.json")
    if not all((path / file).exists() for file in required):
        return {"available": False, "path": str(path), "reason": "Run metadata not complete yet"}
    config, norm, provenance, status = [read_json(path / file) for file in required]
    if status.get("state") != "complete":
        return {"available": False, "path": str(path), "reason": "Training/evaluation export not complete", "status": status}
    expected = provenance["data"]
    for file in ("manifest.json", f"{split}.npz", f"{split}_rgbd.npy"):
        if sha(pack / file) != expected.get(file):
            raise ValueError(f"Run {name} used a different sealed pack: {file}")
    progress_target = config["model"]["progress_event_definition"]
    selected_data = select_targets(data, progress_target)
    supported = np.asarray(norm["supported_events"], bool)
    if supported.shape != (3,):
        raise ValueError("Checkpoint needs three explicit event support flags")
    result = {"available": True, "path": str(path.resolve()), "arm": config["model"]["arm"], "seed": config["arguments"]["seed"],
        "progress_target": progress_target, "supported_events": supported.tolist(), "training_event_counts": norm["event_counts"],
        "arguments": config["arguments"], "model_config": config["model"], "status": status,
        "source_hashes": {file: sha(path / file) for file in required}, "checkpoints": {}}
    for label in ("last", "best"):
        metric_path, control_path = path / f"metrics_{label}.json", path / f"{label}_controls.json"
        checkpoint = path / f"{label}.pt"
        if not all(file.exists() for file in (metric_path, control_path)):
            result["checkpoints"][label] = {"available": False, "reason": "Checkpoint exports missing"}
            continue
        metric, controls = read_json(metric_path), read_json(control_path)
        if controls["checkpoint_step"] != metric["step"]:
            raise ValueError(f"Stale control exports for {name}/{label}")
        hash_sidecar = path / "checkpoint_sha256.json"
        recorded_checkpoint_hash = read_json(hash_sidecar).get(checkpoint.name) if hash_sidecar.exists() else None
        local_checkpoint_hash = sha(checkpoint) if checkpoint.exists() else None
        if local_checkpoint_hash and recorded_checkpoint_hash and local_checkpoint_hash != recorded_checkpoint_hash:
            raise ValueError("Local checkpoint differs from recorded remote hash")
        check = {"available": True, "step": metric["step"], "selection": "fixed final update" if label == "last" else "minimum fixed validation masked loss",
            "draw_digest": metric.get("draw_digest"), "augmentation_digest": metric.get("augmentation_digest"),
            "checkpoint_sha256": local_checkpoint_hash or recorded_checkpoint_hash,
            "checkpoint_binary_local": checkpoint.exists(), "checkpoint_hash_source": "local binary" if local_checkpoint_hash else "remote sidecar" if recorded_checkpoint_hash else "unavailable; saved exports and matching checkpoint-step metadata only",
            "metrics_sha256": sha(metric_path), "controls_sha256": sha(control_path), "controls": {}}
        for control in ("normal", "blank", "shuffle"):
            control_info = controls["controls"].get(control)
            if control_info is None or control_info.get("available") is False:
                check["controls"][control] = {"available": False, "reason": (control_info or {}).get("reason", "Control not exported")}
                continue
            if control == "shuffle" and len(np.unique(data["image_index"])) < 2:
                raise ValueError("A one-scene validation pack cannot provide an image shuffle control")
            prefix = "last_" if label == "last" else ""
            prediction_path = path / f"{split}_predictions_{prefix}{control}.npz"
            if not prediction_path.exists():
                check["controls"][control] = {"available": False, "reason": "Saved control predictions missing"}
                continue
            prediction = read_npz(prediction_path)
            prediction_hash = sha(prediction_path)
            remote_manifest = path.parent / "prediction_download_verification.json"
            remote_verified = False
            if remote_manifest.exists():
                expected_download = read_json(remote_manifest).get(f"{path.name}/{prediction_path.name}")
                if expected_download:
                    if prediction_hash != expected_download["sha256"]:
                        raise ValueError("Prediction differs from verified remote export")
                    remote_verified = True
            check["controls"][control] = {"available": True, "prediction_sha256": prediction_hash, "remote_export_hash_verified": remote_verified,
                "horizons": summarize(prediction, selected_data, groups, manifest, supported, manifest["splits"][split]["scene_ids"]),
                "shuffle": control_info.get("shuffle")}
        check["image_dependence_control_minus_normal"] = {}
        normal = check["controls"].get("normal", {})
        if normal.get("available"):
            for control in ("blank", "shuffle"):
                altered = check["controls"].get(control, {})
                if not altered.get("available"):
                    check["image_dependence_control_minus_normal"][control] = {"available": False, "reason": altered.get("reason")}
                    continue
                changes = {}
                for horizon, base in normal["horizons"].items():
                    other = altered["horizons"][horizon]
                    if not (base.get("available") and other.get("available")):
                        continue
                    a, b = other["groups"]["all"], base["groups"]["all"]
                    delta = lambda x, y: None if x is None or y is None else float(x-y)
                    changes[horizon] = {"endpoint_fde_m": delta(a["motion"]["endpoint_fde_m"], b["motion"]["endpoint_fde_m"]),
                        "work_mae_kj": delta(a["work_kj"]["mae"], b["work_kj"]["mae"]),
                        "roll_mae_deg": delta(a["attitude_deg"]["roll"]["mae"], b["attitude_deg"]["roll"]["mae"]),
                        "pitch_mae_deg": delta(a["attitude_deg"]["pitch"]["mae"], b["attitude_deg"]["pitch"]["mae"])}
                check["image_dependence_control_minus_normal"][control] = {"available": True, "same_checkpoint": True, "horizons": changes}
        result["checkpoints"][label] = check
    return result


def paired_comparisons(runs):
    comparisons = []
    matching_args = ("seed", "steps", "batch", "learning_rate", "weight_decay", "positive_sampling_fraction", "max_event_pos_weight",
                     "candidate_patches", "rotation_augmentation", "progress_target", "xy_weight", "yaw_weight", "work_weight", "event_weight", "attitude_weight", "eval_every")
    for name, rgbd in runs.items():
        if not rgbd.get("available") or rgbd["arm"] != "rgbd":
            continue
        for other_name, blank in runs.items():
            if not blank.get("available") or blank["arm"] != "blank" or rgbd["seed"] != blank["seed"]:
                continue
            mismatches = [key for key in matching_args if rgbd["arguments"].get(key) != blank["arguments"].get(key)]
            row = {"rgbd": name, "blank": other_name, "same_seed": True, "training_setting_mismatches": mismatches, "checkpoints": {}}
            for label in ("last", "best"):
                first, second = rgbd["checkpoints"].get(label, {}), blank["checkpoints"].get(label, {})
                if not (first.get("available") and second.get("available")):
                    row["checkpoints"][label] = {"available": False}
                    continue
                a, b = first["controls"].get("normal", {}), second["controls"].get("normal", {})
                if not (a.get("available") and b.get("available")):
                    row["checkpoints"][label] = {"available": False}
                    continue
                fixed_match = first["step"] == second["step"] and first.get("draw_digest") == second.get("draw_digest")
                compared = {"available": True, "same_update_and_draw_digest": fixed_match,
                    "controlled_fixed_budget": label == "last" and not mismatches and fixed_match,
                    "caution": "Best checkpoints can be different updates; overlapping validation windows are not independent trials.", "deltas_rgbd_minus_blank": {}}
                for horizon in ("4s", "8s", "12s"):
                    ah, bh = a["horizons"][horizon], b["horizons"][horizon]
                    if not (ah.get("available") and bh.get("available")):
                        continue
                    am, bm = ah["groups"]["all"], bh["groups"]["all"]
                    delta = lambda x, y: None if x is None or y is None else float(x-y)
                    compared["deltas_rgbd_minus_blank"][horizon] = {"endpoint_fde_m": delta(am["motion"]["endpoint_fde_m"], bm["motion"]["endpoint_fde_m"]),
                        "work_mae_kj": delta(am["work_kj"]["mae"], bm["work_kj"]["mae"]),
                        "roll_mae_deg": delta(am["attitude_deg"]["roll"]["mae"], bm["attitude_deg"]["roll"]["mae"]),
                        "pitch_mae_deg": delta(am["attitude_deg"]["pitch"]["mae"], bm["attitude_deg"]["pitch"]["mae"])}
                row["checkpoints"][label] = compared
            comparisons.append(row)
    return comparisons


def markdown(report):
    lines = ["# Diverse FDM evidence report", "", f"This report evaluates saved forecasts on the frozen {report['split']} pack. It does not establish closed-loop MPPI success or full-route energy efficiency.", "",
        f"Pack: `{report['pack']}`. Scenes: {report['scene_count']}; episodes: {report['episode_count']}; overlapping windows: {report['window_count']}.", "",
        "Fixed-budget final checkpoints and validation-selected best checkpoints are shown separately. Unobserved tails are masked; unsupported event heads are not counted as valid risk predictors.", ""]
    fmt = lambda value: "—" if value is None else f"{value:.3f}"
    for label in ("last", "best"):
        lines += [f"## {'Fixed final update' if label == 'last' else 'Validation-selected best'}", "",
                  "| Run | Step | Horizon | Complete windows | FDE (m) | Work MAE (kJ) | Roll MAE (deg) | Pitch MAE (deg) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, run in report["runs"].items():
            checkpoint = run.get("checkpoints", {}).get(label, {})
            normal = checkpoint.get("controls", {}).get("normal", {})
            if not normal.get("available"):
                continue
            for horizon, entry in normal["horizons"].items():
                if not entry.get("available"):
                    continue
                values = entry["groups"]["all"]
                lines.append(f"| {name} | {checkpoint['step']} | {horizon} | {values['motion']['complete_horizon_windows']} | {fmt(values['motion']['endpoint_fde_m'])} | {fmt(values['work_kj']['mae'])} | {fmt(values['attitude_deg']['roll']['mae'])} | {fmt(values['attitude_deg']['pitch']['mae'])} |")
        lines.append("")
    lines += ["## Support and image controls", ""]
    for name, run in report["runs"].items():
        if not run.get("available"):
            lines.append(f"- {name}: unavailable ({run['reason']}).")
            continue
        lines.append(f"- {name}: supported contact/rollover/progress heads = `{run['supported_events']}`; progress target `{run['progress_target']}`.")
        for label, checkpoint in run["checkpoints"].items():
            if not checkpoint.get("available"):
                continue
            shuffle = checkpoint["controls"].get("shuffle", {})
            if not shuffle.get("available"):
                lines.append(f"  - {label} image shuffle unavailable: {shuffle.get('reason', 'missing export')}.")
    lines += ["", f"Hash-verified raw prefixes support pre-onset/already-observed-event strata for {report['causal_strata']['known_windows']} windows. Detailed horizon, per-scene, hazard, image-control and matched-arm results are in `report.json`.", "",
        "For physical MPPI evaluation, compare completion/failure/abstention first. Compare time and mechanical work only for paired safe full-goal completions with the same start, goal and time cap. A stalled vehicle can consume less total energy and must not win an efficiency comparison by failing early.", "",
        "Signed attitude endpoint predictions and solver-step peak attitude are different targets; both comparisons are retained. Reported full-route work/time extrapolations remain planning heuristics until measured in Chrono.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIRECTORY; repeat for the four arms/seeds")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    args = parser.parse_args()
    if args.out.exists() or args.out.resolve().is_relative_to(Path("/home/harry/NeDM")):
        raise ValueError("Use a new isolated report directory")
    manifest, data, episodes = load_pack(args.pack, args.split)
    groups, strata = causal_groups(data, episodes, args.raw_root)
    runs = {}
    for declaration in args.run:
        if "=" not in declaration:
            parser.error("Each --run must be LABEL=RUN_DIRECTORY")
        name, directory = declaration.split("=", 1)
        if name in runs or not name:
            raise ValueError("Run labels must be nonempty and unique")
        runs[name] = read_run(name, Path(directory), args.pack, manifest, data, groups, args.split)
    report = {"schema": "fdm_diverse_report_v1", "pack": str(args.pack.resolve()), "pack_manifest_sha256": sha(args.pack / "manifest.json"),
        "split": args.split, "scene_count": manifest["splits"][args.split]["scenes"], "episode_count": len(episodes), "window_count": len(data["anchor"]),
        "event_schema": manifest["event_schema"], "observation_semantics": manifest["observation"], "causal_strata": strata,
        "runs": runs, "rgbd_vs_blank": paired_comparisons(runs),
        "limitations": ["Overlapping windows are not independent trials; no window-level significance claim.",
            "Prediction accuracy does not establish planner success. Compare full Chrono outcomes separately.",
            "Censored trajectories/work/attitude require measured endpoints; cumulative known positives can remain valid after a missing tail.",
            "Unsupported event heads have no operational risk metric. A missing shuffle control is not zero image dependence.",
            "Per-scene metrics and equal-scene macro means prevent long episodes from silently dominating every comparison."],
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in sorted({Path(__file__).resolve()} | {
            Path(module.__file__).resolve() for module in sys.modules.values() if getattr(module, "__file__", None)
            and Path(module.__file__).suffix == ".py" and Path(module.__file__).resolve().is_relative_to(ROOT)})}}
    args.out.mkdir(parents=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    (args.out / "report.md").write_text(markdown(report))
    print(json.dumps({"out": str(args.out), "complete_runs": sum(run.get("available", False) for run in runs.values()), "report_sha256": sha(args.out / "report.json")}))


if __name__ == "__main__":
    main()
