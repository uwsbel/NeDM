#!/usr/bin/env python3
"""Audit and compare saved RGB-D FDM runs without training or test-split access.

Best-checkpoint image interventions and causal-anchor strata are reported
separately from fixed-budget last-checkpoint aggregates. This analysis does not
select a new checkpoint or claim that overlapping validation windows are
independent trials. It reads only saved predictions, fixed validation labels,
and hash-verified metadata of those same validation episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from traverse_fdm_train import binary_metrics, json_clean, pose_metrics, retention_metrics, sha256, write_json


ARMS = ("rgbd", "rgb_only", "depth_only", "blank")
EVENTS = ("contact", "rollover", "low_progress")
FOCUS_GROUPS = ("all", "moving", "already_low_progress", "before_first_contact", "moving_before_first_contact")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def data_digest(data: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in ("episode_index", "anchor", "events", "event_mask", "trajectory", "trajectory_mask"):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(data[name]).tobytes())
    return digest.hexdigest()


def build_strata(data: dict[str, np.ndarray], records: list[dict], source_root: Path,
                 cache_path: Path | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """All group membership uses information available at the anchor only."""
    history, anchors, episodes = data["history"], data["anchor"], data["episode_index"]
    count = len(anchors)
    signature = data_digest(data)
    metadata_hashes = [record["meta_sha256"] for record in records]
    metadata_signature = hashlib.sha256(json.dumps(metadata_hashes).encode()).hexdigest()
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if str(cache["validation_identity_sha256"].item()) != signature:
                raise ValueError("Strata cache validation identities/labels differ")
            if str(cache["metadata_hashes_sha256"].item()) != metadata_signature:
                raise ValueError("Strata cache metadata hashes differ")
            groups = {key[6:]: cache[key].astype(bool) for key in cache.files if key.startswith("group_")}
            details = json.loads(str(cache["details_json"].item()))
        details["loaded_cache"] = str(cache_path)
        details["metadata_files_read"] = 0
        return groups, details
    displacement = np.linalg.norm(history[:, -1, 20:22] - history[:, 0, 20:22], axis=1)
    vx = history[:, -1, 0]
    effort = history[:, :, 18].mean(axis=1)
    full = anchors >= 15
    moving = full & (vx >= 1.0) & (displacement >= 0.5)
    already_low = full & (displacement < 0.15 * 0.75) & (effort > 0.3)
    past, recent = np.zeros(count, bool), np.zeros(count, bool)
    past_known, recent_known = np.ones(count, bool), np.ones(count, bool)
    for episode_index, record in enumerate(records):
        path = source_root / record["source"] / "meta.json"
        if sha256(path) != record["meta_sha256"]:
            raise ValueError(f"Source validation metadata changed: {path}")
        metadata = read_json(path)
        listed = np.asarray(metadata.get("contact", {}).get("events", []), dtype=float).reshape(-1, 3)
        frames = listed[listed[:, 2] > 1.0, 0] if len(listed) else np.array([])
        rows = np.flatnonzero(episodes == episode_index)
        for row in rows:
            anchor = anchors[row]
            past[row] = np.any(frames < anchor)
            recent[row] = np.any((frames >= anchor - 4) & (frames < anchor))
            if len(listed) >= 2000:
                # The collector caps the event list. A missing later event is
                # not proof of no recent contact. Observed positives remain known.
                past_known[row] = past[row] or anchor <= listed[-1, 0] + 1
                recent_known[row] = recent[row] or anchor <= listed[-1, 0] + 1
    before = ~past & past_known
    groups = {
        "all": np.ones(count, bool), "full_history": full, "moving": moving,
        "slow": full & (np.abs(vx) < 0.5), "already_low_progress": already_low,
        "not_already_low_progress": full & ~already_low, "startup_padded_history": ~full,
        "before_first_contact": before, "recent_contact": recent,
        "past_contact_not_recent": past & ~recent & recent_known,
        "moving_before_first_contact": moving & before,
    }
    details = {
        "protocol": "Exploratory causal-anchor strata; same definitions as the earlier profile pilot where observable",
        "validation_identity_sha256": signature, "metadata_hashes_sha256": metadata_signature,
        "metadata_files_read": len(records), "raw_state_files_read": 0,
        "unknown_prior_contact_windows": int((~past_known).sum()),
        "unknown_recent_contact_windows": int((~recent_known).sum()),
        "definitions": {
            "full_history": "anchor >=15, so all 16 historical frames are measured",
            "moving": "full history, current body vx>=1m/s, past0.75s net displacement>=0.5m",
            "slow": "full history and abs(current body vx)<0.5m/s",
            "already_low_progress": "full history, past0.75s net displacement<0.1125m, mean previous throttle>0.3",
            "before_first_contact": "no prior recorded force>1N asset-contact interval; capped-list unknowns excluded",
            "recent_contact": "recorded force>1N asset-contact interval in the past0.2s before anchor",
            "moving_before_first_contact": "moving and before_first_contact",
            "low_progress_target": "frozen four-second net-displacement/throttle label; may include rollback or out-and-back motion, not necessarily sustained stall",
            "event_score": "four-second probability; structurally unsupervised early low-progress logits are never used",
            "groups": "overlapping; their counts must not be added as independent trials",
        },
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, validation_identity_sha256=np.array(signature),
                            metadata_hashes_sha256=np.array(metadata_signature),
                            details_json=np.array(json.dumps(details)),
                            **{f"group_{key}": value for key, value in groups.items()})
    return groups, details


def event_stratum(label: np.ndarray, probability: np.ndarray, episode_ids: np.ndarray) -> dict:
    label = np.asarray(label, np.int64)
    result = binary_metrics(label, probability)
    result["known_unique_episodes"] = int(np.unique(episode_ids).size)
    result["positive_unique_episodes"] = int(np.unique(episode_ids[label > 0]).size)
    result["negative_unique_episodes"] = int(np.unique(episode_ids[label == 0]).size)
    result["scarce_positive_support"] = result["positive_unique_episodes"] < 10
    result["false_accepts_at_total_retention"] = retention_metrics(label, probability)
    if len(label) and label.any() and (label == 0).any():
        # A matched feasible-class retention diagnostic is distinct from the
        # total-coverage tables. This threshold is fitted inside development
        # validation solely to compare ROC operating points, not deployed.
        threshold = float(np.quantile(probability[label == 0], 0.95, method="higher"))
        accepted = probability <= threshold
        result["diagnostic_95pct_negative_retention"] = {
            "threshold": threshold, "threshold_source": "this development-validation stratum; not a deployable gate",
            "negative_retention": float(accepted[label == 0].mean()),
            "failure_recall": float((~accepted[label > 0]).mean()),
            "false_accepts": int((accepted & (label > 0)).sum()),
            "positive_windows": int(label.sum()),
        }
    return result


def summarize_prediction(prediction: dict[str, np.ndarray], data: dict[str, np.ndarray],
                         groups: dict[str, np.ndarray]) -> dict[str, Any]:
    if prediction["trajectory"].shape != data["trajectory"].shape or prediction["event_logits"].shape != data["events"].shape:
        raise ValueError("Saved prediction shapes differ from fixed validation labels")
    if any(not np.isfinite(value).all() for value in prediction.values()):
        raise ValueError("Saved predictions contain non-finite values")
    probability = 1 / (1 + np.exp(-np.clip(prediction["event_logits"], -80, 80)))
    episodes = data["episode_index"]
    result = {}
    for name, group in groups.items():
        row = {"windows": int(group.sum()), "unique_episodes": int(np.unique(episodes[group]).size),
               "pose": pose_metrics(prediction["trajectory"][group], data["trajectory"][group], data["trajectory_mask"][group]),
               "events_4s": {}}
        final = group & (data["trajectory_mask"][:, -1, 0] > 0)
        row["pose"]["complete_horizon_unique_episodes"] = int(np.unique(episodes[final]).size)
        for event_index in (0, 2):
            known = group & (data["event_mask"][:, -1, event_index] > 0)
            row["events_4s"][EVENTS[event_index]] = event_stratum(
                data["events"][known, -1, event_index], probability[known, -1, event_index], episodes[known])
        result[name] = row
    return result


def read_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key].copy() for key in ("trajectory", "work", "event_logits")}


def metric_delta(first: dict, second: dict) -> dict:
    """First minus second; errors/Brier lower is better and AUC higher is better."""
    difference = lambda a, b: None if a is None or b is None else float(a - b)
    return {
        "ade_delta_m": difference(first["pose"]["ade_m"], second["pose"]["ade_m"]),
        "fde_delta_m": difference(first["pose"]["fde_m"], second["pose"]["fde_m"]),
        "events_4s": {name: {"auroc_delta": difference(first["events_4s"][name]["auroc"], second["events_4s"][name]["auroc"]),
            "brier_delta": difference(first["events_4s"][name]["brier"], second["events_4s"][name]["brier"]),
            "positive_unique_episodes": first["events_4s"][name]["positive_unique_episodes"],
            "scarce_positive_support": first["events_4s"][name].get("scarce_positive_support",
                first["events_4s"][name]["positive_unique_episodes"] < 10)}
            for name in ("contact", "low_progress")},
    }


def aggregate_row(metrics: dict) -> dict:
    val = metrics["validation"]
    return {"step": metrics["step"], "validation_loss": val["loss"]["total"],
            "pose": val["pose"], "work": val["work"], "nominal_kinematics": val["nominal_kinematics"],
            "draw_digest": metrics["draw_digest"],
            "events_4s": {name: val["events"][name]["last_horizon"] for name in ("contact", "low_progress")}}


def paired_motion(first: np.ndarray, second: np.ndarray, data: dict, group: np.ndarray) -> dict:
    valid = group & (data["trajectory_mask"][:, -1, 0] > 0)
    a = np.linalg.norm(first[valid, -1, :2] - data["trajectory"][valid, -1, :2], axis=1)
    b = np.linalg.norm(second[valid, -1, :2] - data["trajectory"][valid, -1, :2], axis=1)
    episode_ids = data["episode_index"][valid]
    delta = a - b
    episode_delta = [float(delta[episode_ids == ep].mean()) for ep in np.unique(episode_ids)]
    return {"complete_windows": int(valid.sum()), "unique_episodes": len(episode_delta),
            "mean_fde_delta_m": float(delta.mean()) if len(delta) else None,
            "median_fde_delta_m": float(np.median(delta)) if len(delta) else None,
            "fraction_windows_improved": float((delta < 0).mean()) if len(delta) else None,
            "episode_equal_weighted_mean_fde_delta_m": float(np.mean(episode_delta)) if episode_delta else None,
            "fraction_episodes_improved": float((np.asarray(episode_delta) < 0).mean()) if episode_delta else None}


def prediction_sensitivity(altered: dict, normal: dict, data: dict) -> dict:
    """Measure use of pixels separately from whether their use improves errors."""
    valid = data["trajectory_mask"][..., 0] > 0
    change = np.linalg.norm(altered["trajectory"][..., :2] - normal["trajectory"][..., :2], axis=-1)
    final = valid[:, -1]
    result = {"mean_pose_change_m": float(change[valid].mean()) if valid.any() else None,
              "max_pose_change_m": float(change[valid].max()) if valid.any() else None,
              "mean_final_pose_change_m": float(change[final, -1].mean()) if final.any() else None,
              "max_final_pose_change_m": float(change[final, -1].max()) if final.any() else None,
              "events_4s": {}}
    sigmoid = lambda logits: 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    probability_change = np.abs(sigmoid(altered["event_logits"]) - sigmoid(normal["event_logits"]))
    for index in (0, 2):
        known = data["event_mask"][:, -1, index] > 0
        values = probability_change[known, -1, index]
        result["events_4s"][EVENTS[index]] = {"mean_probability_change": float(values.mean()) if len(values) else None,
                                              "max_probability_change": float(values.max()) if len(values) else None}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="JSON report; companion Markdown uses the same stem")
    parser.add_argument("--source-root", type=Path, default=Path("/home/harry/NeDM"))
    parser.add_argument("--strata-cache", type=Path, help="Read/create small verified NPZ of causal group membership")
    parser.add_argument("--strata-metadata", type=Path, help="Reuse earlier profile-pilot strata.json's raw motion diagnostics")
    parser.add_argument("--allow-partial", action="store_true", help="Report available completed runs without asserting the full eight-run suite")
    args = parser.parse_args()
    manifest = read_json(args.data / "manifest.json")
    records = read_json(args.data / "val_episodes.json")
    with np.load(args.data / "val.npz", allow_pickle=False) as handle:
        data = {key: handle[key].copy() for key in handle.files}
    if "candidate" in data or data["commands"].shape[-1] != 5 or data["global_features"].shape[-1] != 8:
        raise ValueError("Expected RGB-D geometry-only command cache, not profile-pilot terrain inputs")
    val_hash = sha256(args.data / "val.npz")
    if val_hash != manifest["splits"]["val"]["sha256"]:
        raise ValueError("Validation cache hash differs from its manifest")
    image_hash = sha256(args.data / manifest["splits"]["val"]["rgbd_file"])
    if image_hash != manifest["splits"]["val"]["rgbd_sha256"]:
        raise ValueError("Validation RGB-D hash differs from its manifest")
    manifest_hash = sha256(args.data / "manifest.json")
    groups, strata_details = build_strata(data, records, args.source_root, args.strata_cache)
    nominal = {key: pose_metrics(data["nominal_pose"][group], data["trajectory"][group], data["trajectory_mask"][group])
               for key, group in groups.items()}
    report: dict[str, Any] = {
        "protocol": "Fixed-budget last aggregates; separately, saved best-validation-loss checkpoints and image interventions, with exploratory causal-anchor strata",
        "training_performed": False, "protected_test_read": False, "runs_root": str(args.runs.resolve()),
        "data": {"path": str(args.data.resolve()), "validation_sha256": val_hash, "rgbd_sha256": image_hash,
                 "manifest_sha256": manifest_hash, "validation_windows": len(data["anchor"]), "validation_episodes": len(records),
                 "observation": manifest["observation"], "camera": manifest["camera"], "image_encoding": manifest["image_encoding"],
                 "controller_domain": manifest["controller_domain"], "localization": manifest.get("localization"),
                 "reference_confound": manifest.get("reference_confound"), "split_rule": manifest["split_rule"]},
        "strata": strata_details, "nominal_kinematics_by_stratum": nominal, "runs": {},
        "paired_modalities": [], "within_model_image_interventions": [],
        "limitations": [
            "Validation selected the best checkpoints. Best-checkpoint strata are development diagnostics, not independent confirmatory results.",
            "Causal anchor strata overlap. Windows overlap within episodes; exact positive episode counts matter more than window counts.",
            "One arena and measured absolute anchor pose allow location-based map memorization. Pixel interventions and blank-arm comparisons do not establish new-arena generalization.",
            "Known reference speeds may encode authored terrain restrictions. Image inputs contain no BMP or authored clearance, but the command distribution remains a possible shortcut.",
            "Low progress is a net-displacement/throttle label, not a guarantee of sustained stall. Prospective moving cases are reported separately.",
            "Rollover has no positive examples and is unsupported. It is excluded from comparisons.",
            "Fixed-negative-retention thresholds are fitted within these validation strata for ROC diagnostics only, not calibrated deployment thresholds.",
            "This report evaluates recorded trajectories. Closed-loop MPPI and unseen-route Chrono outcomes must be reported separately.",
        ],
    }
    predictions: dict[tuple[str, str], dict] = {}
    for status_path in sorted(args.runs.glob("*/status.json")):
        folder, status = status_path.parent, read_json(status_path)
        if status.get("state") != "complete":
            if args.allow_partial:
                continue
            raise ValueError(f"Incomplete run: {folder}")
        if status.get("model_family") != "rgbd_reference_gru":
            raise ValueError(f"Run is not the RGB-D forward-GRU model: {folder}")
        provenance, config = read_json(folder / "provenance.json"), read_json(folder / "config.json")
        expected_hashes = {"val.npz": val_hash, "val_rgbd.npy": image_hash, "manifest.json": manifest_hash}
        if any(provenance["data"].get(key) != value for key, value in expected_hashes.items()):
            raise ValueError(f"Run used different validation pixels/labels: {folder}")
        best, last = read_json(folder / "metrics_best.json"), read_json(folder / "metrics_last.json")
        controls = read_json(folder / "best_controls.json")
        if controls["checkpoint_step"] != best["step"]:
            raise ValueError(f"Interventions use a different best checkpoint: {folder}")
        normalization = read_json(folder / "normalization.json")
        row = {"arm": status["arm"], "seed": status["seed"], "parameter_count": config["parameter_count"],
               "status": status, "best": aggregate_row(best), "fixed_last": aggregate_row(last),
               "supported_events": normalization["supported_events"], "training_event_counts": normalization["event_counts"],
               "code_hashes": provenance["code"], "data_hashes": provenance["data"],
               "controls": {}, "fixed_last_strata": None,
               "fixed_last_strata_note": "No last-checkpoint predictions saved; only original aggregate scope is available."}
        for control in ("normal", "shuffle", "blank", "current_history"):
            if control not in controls["controls"]:
                continue
            path = folder / f"val_predictions_{control}.npz"
            if control == "normal" and not path.exists():
                path = folder / "val_predictions_best.npz"
            if not path.exists():
                raise ValueError(f"Missing saved {control} predictions: {folder}")
            prediction = read_prediction(path)
            predictions[(folder.name, control)] = prediction
            strata = summarize_prediction(prediction, data, groups)
            ordinary = controls["controls"][control]
            for name in ("ade_m", "fde_m"):
                actual, expected = strata["all"]["pose"][name], ordinary["pose"][name]
                if actual is not None and (expected is None or not np.isclose(actual, expected, atol=1e-5, rtol=1e-5)):
                    raise ValueError(f"Prediction/metric {name} mismatch: {folder} {control}")
            for event in ("contact", "low_progress"):
                actual, expected = strata["all"]["events_4s"][event]["auroc"], ordinary["events"][event]["last_horizon"]["auroc"]
                if actual is not None and (expected is None or abs(actual - expected) > 1e-7):
                    raise ValueError(f"Prediction/metric AUC mismatch: {folder} {control} {event}")
            row["controls"][control] = {"saved_predictions_sha256": sha256(path), "strata": strata,
                                        "aggregate": ordinary}
        last_path = folder / "val_predictions_last.npz"
        if last_path.exists():
            row["fixed_last_strata"] = summarize_prediction(read_prediction(last_path), data, groups)
            row["fixed_last_strata_note"] = "Explicit last-checkpoint predictions supplied."
        report["runs"][folder.name] = row
    if not report["runs"]:
        raise ValueError("No completed RGB-D runs found")
    suite = list(report["runs"].values())
    seeds = sorted({row["seed"] for row in suite})
    if not args.allow_partial and (len(suite) != 8 or len(seeds) != 2 or
                                  any({r["arm"] for r in suite if r["seed"] == seed} != set(ARMS) for seed in seeds)):
        raise ValueError("Expected the complete four-arm, two-seed RGB-D suite")
    parity = []
    for seed in seeds:
        rows = [row for row in suite if row["seed"] == seed]
        matched = len({(row["fixed_last"]["step"], row["fixed_last"]["draw_digest"]) for row in rows}) == 1
        same_capacity = len({row["parameter_count"] for row in rows}) == 1
        same_sources = len({json.dumps(row["code_hashes"], sort_keys=True) for row in rows}) == 1
        same_data = len({json.dumps(row["data_hashes"], sort_keys=True) for row in rows}) == 1
        parity.append({"seed": seed, "matched_final_sampler_and_steps": matched, "matched_parameters": same_capacity,
                       "matched_code_hashes": same_sources, "matched_data_hashes": same_data,
                       "arms": [row["arm"] for row in rows], "step": rows[0]["fixed_last"]["step"],
                       "draw_digest": rows[0]["fixed_last"]["draw_digest"]})
        if not (matched and same_capacity and same_sources and same_data):
            raise ValueError(f"Seed {seed} has unmatched model capacity, data, source, draws, or budget")
    report["matching_audit"] = parity
    for name, row in report["runs"].items():
        if "normal" not in row["controls"]:
            continue
        normal = row["controls"]["normal"]["strata"]
        for control in ("shuffle", "blank", "current_history"):
            if control not in row["controls"]:
                continue
            altered = row["controls"][control]["strata"]
            report["within_model_image_interventions"].append({
                "run": name, "arm": row["arm"], "seed": row["seed"], "best_step": row["best"]["step"],
                "comparison": f"{control} minus normal",
                "prediction_sensitivity": prediction_sensitivity(predictions[(name, control)], predictions[(name, "normal")], data),
                "strata": {
                    group: {**metric_delta(altered[group], normal[group]), "paired_motion": paired_motion(
                        predictions[(name, control)]["trajectory"], predictions[(name, "normal")]["trajectory"], data, groups[group])}
                    for group in FOCUS_GROUPS}})
        if row["arm"] != "rgbd":
            continue
        for alternative in ("blank", "rgb_only", "depth_only"):
            matches = [(other_name, other) for other_name, other in report["runs"].items()
                       if other["arm"] == alternative and other["seed"] == row["seed"]]
            if not matches:
                continue
            other_name, other = matches[0]
            other_strata = other["controls"]["normal"]["strata"]
            report["paired_modalities"].append({
                "seed": row["seed"], "comparison": f"rgbd minus {alternative}",
                "fixed_last": metric_delta(row["fixed_last"], other["fixed_last"]),
                "best_steps": {"rgbd": row["best"]["step"], alternative: other["best"]["step"]},
                "best_strata": {group: {**metric_delta(normal[group], other_strata[group]), "paired_motion": paired_motion(
                    predictions[(name, "normal")]["trajectory"], predictions[(other_name, "normal")]["trajectory"], data, groups[group])}
                    for group in FOCUS_GROUPS}})
    # Simple causal comparators expose already-low-speed shortcuts in pooled
    # low-progress discrimination. They are ranking scores, not probabilities.
    vx = data["history"][:, -1, 0]
    displacement = np.linalg.norm(data["history"][:, -1, 20:22] - data["history"][:, 0, 20:22], axis=1)
    from traverse_fdm_strata import auc
    report["causal_low_progress_comparators"] = {}
    for name, score in (("negative_current_abs_speed", -np.abs(vx)), ("negative_past_displacement", -displacement)):
        report["causal_low_progress_comparators"][name] = {}
        for group_name in FOCUS_GROUPS:
            known = groups[group_name] & (data["event_mask"][:, -1, 2] > 0)
            labels = data["events"][known, -1, 2] > 0
            report["causal_low_progress_comparators"][name][group_name] = {
                "auroc": auc(labels, score[known]), "known_windows": int(known.sum()), "positive_windows": int(labels.sum()),
                "positive_unique_episodes": int(np.unique(data["episode_index"][known][labels]).size)}
    # Reuse existing raw motion diagnostics only after exact row/identity checks.
    old_cases = {}
    if args.strata_metadata is not None:
        old = read_json(args.strata_metadata)
        for case in old.get("moving_low_progress_cases", []):
            index = case["validation_row"]
            if records[int(data["episode_index"][index])]["id"] != case["episode"] or int(data["anchor"][index]) != case["anchor_frame"]:
                raise ValueError("Earlier stratum motion diagnostic identity mismatch")
            old_cases[index] = case
        report["reused_motion_diagnostics"] = {"path": str(args.strata_metadata), "sha256": sha256(args.strata_metadata),
                                               "verified_cases": len(old_cases)}
    case_groups = {
        "moving_future_low_progress": groups["moving"] & (data["event_mask"][:, -1, 2] > 0) & (data["events"][:, -1, 2] > 0),
        "moving_future_first_contact": groups["moving_before_first_contact"] & (data["event_mask"][:, -1, 0] > 0) & (data["events"][:, -1, 0] > 0),
    }
    report["prospective_positive_cases"] = {}
    for group_name, case_mask in case_groups.items():
        rows = []
        for index in np.flatnonzero(case_mask):
            case = {"validation_row": int(index), "episode": records[int(data["episode_index"][index])]["id"],
                    "anchor_frame": int(data["anchor"][index]), "current_vx_mps": float(vx[index]),
                    "past_075s_displacement_m": float(displacement[index]),
                    "before_first_contact": bool(groups["before_first_contact"][index]),
                    "target_contact_4s": bool(data["events"][index, -1, 0]),
                    "target_low_progress_4s": bool(data["events"][index, -1, 2]),
                    "target_displacement_4s_m": float(np.linalg.norm(data["trajectory"][index, -1, :2])),
                    "nominal_displacement_4s_m": float(np.linalg.norm(data["nominal_pose"][index, -1, :2])), "predictions": {}}
            if index in old_cases:
                case["reused_raw_future_motion_diagnostics"] = old_cases[index].get("future_motion_diagnostics")
            for (run_name, control), prediction in predictions.items():
                if control not in ("normal", "blank", "shuffle"):
                    continue
                probability = 1 / (1 + np.exp(-np.clip(prediction["event_logits"][index, -1], -80, 80)))
                case["predictions"][f"{run_name}/{control}"] = {
                    "displacement_4s_m": float(np.linalg.norm(prediction["trajectory"][index, -1, :2])),
                    "fde_m": float(np.linalg.norm(prediction["trajectory"][index, -1, :2] - data["trajectory"][index, -1, :2])),
                    "contact_probability_4s": float(probability[0]), "low_progress_probability_4s": float(probability[2])}
            rows.append(case)
        report["prospective_positive_cases"][group_name] = {"windows": len(rows),
            "unique_episodes": int(np.unique(data["episode_index"][case_mask]).size), "cases": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    format_number = lambda value: "unsupported" if value is None else f"{value:.4f}"
    lines = ["# RGB-D finite-horizon validation comparison", "",
             "Fixed-budget last checkpoints are compared first. Separate selected-checkpoint image interventions and prospective hazard strata follow; all remain development validation on one arena.", "",
             "| Arm | Seed | Last step | ADE / FDE (m) | Contact AUC at 4s | Low-progress AUC at 4s | Best step |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in suite:
        last = row["fixed_last"]
        lines.append(f"| {row['arm']} | {row['seed']} | {last['step']} | {format_number(last['pose']['ade_m'])} / {format_number(last['pose']['fde_m'])} | "
                     f"{format_number(last['events_4s']['contact']['auroc'])} | {format_number(last['events_4s']['low_progress']['auroc'])} | {row['best']['step']} |")
    exemplar = next(iter(report["runs"].values()))["controls"]["normal"]["strata"]
    lines += ["", "| Causal anchor stratum | Contact positives / episodes | Low-progress positives / episodes |", "|---|---:|---:|"]
    for name in FOCUS_GROUPS:
        contact, low = [exemplar[name]["events_4s"][event] for event in ("contact", "low_progress")]
        lines.append(f"| {name} | {contact['positive']} / {contact['positive_unique_episodes']} | {low['positive']} / {low['positive_unique_episodes']} |")
    lines += ["", "| RGB-D comparison (same seed, best checkpoints) | Seed | Moving FDE delta (m) | Prospective-contact AUC delta | Moving low-progress AUC delta |",
              "|---|---:|---:|---:|---:|"]
    for comparison in report["paired_modalities"]:
        strata = comparison["best_strata"]
        lines.append(f"| {comparison['comparison']} | {comparison['seed']} | {format_number(strata['moving']['fde_delta_m'])} | "
                     f"{format_number(strata['moving_before_first_contact']['events_4s']['contact']['auroc_delta'])} | "
                     f"{format_number(strata['moving']['events_4s']['low_progress']['auroc_delta'])} |")
    lines += ["", "For the preceding modality table, negative error deltas and positive AUC deltas favor RGB-D. Best steps may differ; fixed-last results above retain equal update budgets.", "",
              "| RGB-D image intervention minus normal (same checkpoint) | Seed | Moving FDE delta (m) | Prospective-contact AUC delta | Moving low-progress AUC delta |",
              "|---|---:|---:|---:|---:|"]
    for comparison in report["within_model_image_interventions"]:
        if comparison["arm"] != "rgbd":
            continue
        strata = comparison["strata"]
        lines.append(f"| {comparison['comparison']} | {comparison['seed']} | {format_number(strata['moving']['fde_delta_m'])} | "
                     f"{format_number(strata['moving_before_first_contact']['events_4s']['contact']['auroc_delta'])} | "
                     f"{format_number(strata['moving']['events_4s']['low_progress']['auroc_delta'])} |")
    lines += ["", "Intervention error increases or AUC decreases suggest useful image information in the selected checkpoint. They do not establish unseen-terrain or closed-loop route-choice success.", "",
              "The JSON includes per-episode support, calibration, fixed-retention false accepts, paired episode-weighted motion errors, and every prospective positive case. Moving low-progress examples are scarce and can include rollback. Rollover is unsupported.", "",
              "No model training, checkpoint selection, protected test reading, or raw state-file rereading occurred during this report."]
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(json_clean({"out": str(args.out), "runs": len(suite), "matching_audit": parity,
                                "prospective_case_support": {key: {k: value[k] for k in ("windows", "unique_episodes")}
                                    for key, value in report["prospective_positive_cases"].items()}}), indent=2))


if __name__ == "__main__":
    main()
