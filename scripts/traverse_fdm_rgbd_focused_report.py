#!/usr/bin/env python3
"""Compare focused RGB-D runs and offline rankings of recorded sibling routes.

No training, new checkpoint selection, MPPI refinement, or simulator execution
occurs here. Predicted costs select routes; SHA-verified Chrono outcomes provide
the separate safety and time ground truth. Validation is a designed development
cohort, not natural deployment prevalence or an untouched test set.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch

from nedm.traverse.fdm_rgbd_planner import RGBDCostConfig, RGBDReferenceScorer
from nedm.traverse.fdm_rgbd_model import PROGRESS_TARGET_VERSIONS
from traverse_fdm_rgbd_train import load_low_dim, select_progress_target
from traverse_fdm_train import binary_metrics, outcome_metrics, sha256, write_json


def read(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify_data(path, progress_target="sustained_stall"):
    manifest = read(path / "manifest.json")
    if progress_target == "bounded_motion" and manifest.get("bounded_motion_version") != 2:
        raise ValueError("Bounded-motion reporting requires the explicit version2 data definition")
    hashes = {name: sha256(path / name) for name in
              ("manifest.json", "train.npz", "val.npz", "train_rgbd.npy", "val_rgbd.npy")}
    for split in ("train", "val"):
        info = manifest["splits"][split]
        if hashes[split + ".npz"] != info["sha256"] or hashes[split + "_rgbd.npy"] != info["rgbd_sha256"]:
            raise ValueError(f"Changed {split} low-dimensional or observed-image cache")
    data = select_progress_target(load_low_dim(path / "val.npz"), progress_target)
    records = read(path / "val_episodes.json")
    if set(np.unique(data["episode_index"])) != set(range(len(records))):
        raise ValueError("Validation episode indices differ from their records")
    outcomes, source_checks = [], []
    for index, record in enumerate(records):
        if record["split"] != "val" or manifest["scene_splits"][record["scene_id"]] != "val":
            raise ValueError("Scene or episode is outside the declared validation split")
        source = Path(record["source"])
        path_outcome = source / "outcome.json"
        actual_hash = sha256(path_outcome)
        if actual_hash != record["source_sha256"]["outcome.json"]:
            raise ValueError(f"Changed measured outcome: {path_outcome}")
        outcome = read(path_outcome)
        if (outcome["case_id"] != record["scene_id"] or outcome["case_sha256"] != record["case_sha256"]
                or outcome["route_sha256"] != record["route_sha256"]):
            raise ValueError("Measured outcome has a different scene or commanded reference")
        rows = np.flatnonzero(data["episode_index"] == index)
        if data["anchor"][rows].tolist() != record["anchors"]:
            raise ValueError("Validation anchor ordering differs from source records")
        outcomes.append(outcome)
        source_checks.append({"episode": record["id"], "outcome_path": str(path_outcome), "sha256": actual_hash})
        if "bounded_motion_ever" in record:
            trajectory_hash = sha256(source / "trajectory.npz")
            if trajectory_hash != record["source_sha256"]["trajectory.npz"]:
                raise ValueError("Changed source of the packed bounded-motion diagnosis")
            outcome["bounded_motion"] = bool(record["bounded_motion_ever"])
            source_checks[-1]["bounded_motion_diagnosis"] = {
                "source": "version2 packed episode diagnosis from recorded poses and controls",
                "trajectory_sha256": trajectory_hash, "first_observed_frame": record["first_observed_bounded_motion_frame"]}
        elif progress_target == "bounded_motion":
            raise ValueError("Bounded-motion report requires its separate full-route episode diagnosis")
    return manifest, hashes, data, records, outcomes, source_checks


def compact_metrics(metrics):
    result = {key: metrics.get(key) for key in ("pose", "nominal_kinematics", "work", "loss", "progress_event_definition", "progress_event_version")}
    result["events"] = {}
    keys = ("count", "positive", "negative", "auroc", "average_precision", "brier", "ece", "unsupported",
            "known_unique_episodes", "positive_unique_episodes", "false_accepts_at_retention")
    for old_name, metric in metrics.get("events", {}).items():
        name = metrics.get("progress_event_definition", "sustained_stall") if old_name == "low_progress" else old_name
        result["events"][name] = {key: metric.get(key) for key in
            ("training_supported", "operational_use", "observed_unique_episodes", "positive_unique_episodes")}
        for kind in ("last_horizon", "window"):
            result["events"][name][kind] = {key: metric.get(kind, {}).get(key) for key in keys}
    result["any_event_window"] = {key: metrics.get("any_event_window", {}).get(key) for key in keys}
    return result


def strata(data, records, progress_target="sustained_stall"):
    episode = data["episode_index"]
    anchor = data["anchor"]
    before_contact = np.array([records[i]["first_contact_frame"] is None or a < records[i]["first_contact_frame"]
                               for i, a in zip(episode, anchor)])
    onset_key = f"first_observed_{progress_target}_frame"
    before_stall = np.array([records[i][onset_key] is None or a < records[i][onset_key] for i, a in zip(episode, anchor)])
    return {"all": np.ones(len(anchor), bool), "frame0": anchor == 0, "anchor_lt2s": anchor < 40,
            "before_first_contact": before_contact, f"before_observed_{progress_target}": before_stall,
            "moving_before_first_contact": before_contact & (np.abs(data["history"][:, -1, 0]) > .5)}


def stratified_metrics(prediction, data, groups, supported, dt, progress_target="sustained_stall"):
    result = {}
    for name, mask in groups.items():
        if not mask.any():
            result[name] = {"windows": 0, "unavailable": "No observed anchors in this stratum"}
            continue
        subset = {key: value[mask] for key, value in data.items()}
        measured = outcome_metrics({key: value[mask] for key, value in prediction.items()}, subset, supported, dt)
        measured["progress_event_definition"] = progress_target
        result[name] = {"windows": int(mask.sum()), "episodes": int(np.unique(subset["episode_index"]).size),
                        "scenes": int(np.unique(subset["scene_index"]).size), **compact_metrics(measured)}
        for head, public in ((0, "contact"), (2, progress_target)):
            positive = (subset["events"][:, -1, head] > 0) & (subset["event_mask"][:, -1, head] > 0)
            result[name]["events"][public]["last_horizon"]["positive_scenes"] = int(np.unique(subset["scene_index"][positive]).size)
    return result


def prediction_costs(prediction, data, supported, dt, config):
    """Delegate directly to the deployed scorer's cost, without running a model."""
    result = {}
    for index, global_ in enumerate(data["global_features"]):
        sine, cosine = global_[6:8]
        anchor = np.array([40 * global_[4], 40 * global_[5], np.arctan2(sine, cosine)])
        goal = anchor[:2] + np.array([cosine * global_[1] - sine * global_[2],
                                      sine * global_[1] + cosine * global_[2]])
        context = SimpleNamespace(cost_config=config, anchor_pose=anchor, goal_xy=goal,
            model=SimpleNamespace(config=SimpleNamespace(dt=dt), supported_events=torch.as_tensor(supported)))
        outputs = {"trajectory": prediction["trajectory"][index:index + 1],
                   "event_probability": 1 / (1 + np.exp(-np.clip(prediction["event_logits"][index:index + 1], -80., 80.)))}
        row = RGBDReferenceScorer.cost_breakdown(context, outputs)
        for key, value in row.items():
            result.setdefault(key, []).append(value[0])
    return {key: np.asarray(value) for key, value in result.items()}


def measured_outcome(outcome, progress_target="sustained_stall"):
    target_safe = (bool(outcome["goal_reached"] and not outcome["asset_contact"] and not outcome["bounded_motion"])
                   if progress_target == "bounded_motion" else bool(outcome["safe_goal_reached"]))
    return {"safe_goal": bool(outcome["safe_goal_reached"]), "goal_reached": bool(outcome["goal_reached"]),
            "target_safe_goal": target_safe, "bounded_motion": outcome.get("bounded_motion"),
            "contact": bool(outcome["asset_contact"]), "sustained_stall": bool(outcome["sustained_near_stop"]),
            "status": outcome["status"], "goal_time_s": outcome["goal_time_s"], "elapsed_s": outcome["elapsed_s"],
            "goal_progress_m": outcome["goal_progress_m"]}


def rank_siblings(prediction, data, records, outcomes, supported, dt, config, progress_target="sustained_stall"):
    costs = prediction_costs(prediction, data, supported, dt, config)
    scenes = []
    for scene in sorted({record["scene_id"] for record in records}):
        episodes = [i for i, record in enumerate(records) if record["scene_id"] == scene]
        if len(episodes) != 6:
            raise ValueError("The predeclared focused comparison requires six siblings per scene")
        rows = []
        for episode in episodes:
            row = np.flatnonzero((data["episode_index"] == episode) & (data["anchor"] == 0))
            if len(row) != 1:
                raise ValueError("Each recorded sibling needs exactly one frame-0 observation")
            rows.append(int(row[0]))
        order = sorted(range(len(rows)), key=lambda j: (float(costs["cost"][rows[j]]), records[episodes[j]]["id"]))
        allowed = [j for j in order if np.isfinite(costs["cost"][rows[j]])]
        choice = allowed[0] if allowed else None
        safe_times = [float(outcomes[i]["goal_time_s"]) for i in episodes
                      if measured_outcome(outcomes[i], progress_target)["target_safe_goal"] and outcomes[i]["goal_time_s"] is not None]
        oracle_time = min(safe_times) if safe_times else None
        selected = measured_outcome(outcomes[episodes[choice]], progress_target) if choice is not None else None
        candidates = []
        for rank, j in enumerate(order):
            row, episode = rows[j], episodes[j]
            candidates.append({"episode": records[episode]["id"], "validation_row": row, "predicted_rank": rank + 1,
                "selected": j == choice, "prediction": {key: value[row].item() for key, value in costs.items()},
                "measured": measured_outcome(outcomes[episode], progress_target)})
        regret = (selected["goal_time_s"] - oracle_time
                  if selected is not None and selected["target_safe_goal"] and oracle_time is not None else None)
        scenes.append({"scene": scene, "candidate_count": len(rows), "abstained": choice is None,
            "selected_episode": records[episodes[choice]]["id"] if choice is not None else None,
            "selected_measured": selected, "safe_completed_alternatives": len(safe_times),
            "oracle_safe_time_s": oracle_time, "safe_completion_time_regret_s": regret,
            "safe_alternative_missed": bool(safe_times and (selected is None or not selected["target_safe_goal"])),
            "candidates": candidates})
    selected = [scene["selected_measured"] for scene in scenes if scene["selected_measured"] is not None]
    safe_times = [row["goal_time_s"] for row in selected if row["target_safe_goal"]]
    regret = [scene["safe_completion_time_regret_s"] for scene in scenes if scene["safe_completion_time_regret_s"] is not None]
    summary = {"scenes": len(scenes), "selected": len(selected), "abstentions": len(scenes) - len(selected),
        "safe_goal": sum(row["safe_goal"] for row in selected), "contacts": sum(row["contact"] for row in selected),
        "target_safe_goal": sum(row["target_safe_goal"] for row in selected),
        "sustained_stalls": sum(row["sustained_stall"] for row in selected),
        "bounded_motion": (sum(row["bounded_motion"] for row in selected)
                           if all("bounded_motion" in outcome for outcome in outcomes) else None),
        "goal_reached": sum(row["goal_reached"] for row in selected),
        "unsafe_or_incomplete_selections": sum(not row["target_safe_goal"] for row in selected),
        "incomplete_without_contact_or_stall": sum(not row["goal_reached"] and not row["contact"] and not row["sustained_stall"] for row in selected),
        "safe_alternatives_missed": sum(scene["safe_alternative_missed"] for scene in scenes),
        "mean_safe_goal_time_s": float(np.mean(safe_times)) if safe_times else None,
        "mean_safe_completion_time_regret_s": float(np.mean(regret)) if regret else None,
        "regret_evaluable_scenes": len(regret)}
    return {"summary": summary, "scenes": scenes, "time_regret_safety_criterion": progress_target,
            "recorded_safe_goal_preserved": "safe_goal remains the original outcome.json field; target_safe_goal uses the named report event"}


def terrain_frame0_diagnostic(prediction, data, records, outcomes, ranking, config, supported, dt):
    """Separate short-horizon motion prediction from eventual terrain blockage."""
    episodes = [i for i, record in enumerate(records) if record["scene_id"].startswith("focus_terrain_")]
    if not episodes:
        return {"unavailable": "No terrain scenes in this validation cohort"}
    rows = np.array([int(np.flatnonzero((data["episode_index"] == i) & (data["anchor"] == 0))[0]) for i in episodes])
    goal = data["global_features"][rows, 1:3]
    initial_distance = np.linalg.norm(goal, axis=1)
    measured_progress = initial_distance - np.linalg.norm(goal - data["trajectory"][rows, -1, :2], axis=1)
    predicted_progress = initial_distance - np.linalg.norm(goal - prediction["trajectory"][rows, -1, :2], axis=1)
    probability = 1 / (1 + np.exp(-np.clip(prediction["event_logits"][rows, :, 2], -80., 80.)))
    eligible = (np.arange(probability.shape[1]) + 1) * dt >= 2. - 1e-6
    bounded_probability = np.where(eligible[None], probability, 0.).max(axis=1)
    observed = data["trajectory_mask"][rows, -1, 0] > 0
    eventual_blocked = np.array([outcomes[i].get("bounded_motion", False) and not outcomes[i]["goal_reached"] for i in episodes])
    eventual_safe = np.array([measured_outcome(outcomes[i], "bounded_motion")["target_safe_goal"] for i in episodes])
    groups = {"eventual_blocked": eventual_blocked, "eventual_safe_goal": eventual_safe,
              "other_or_incomplete": ~eventual_blocked & ~eventual_safe}

    def distribution(values):
        return {"count": int(len(values)), "mean": float(np.mean(values)) if len(values) else None,
                "min": float(np.min(values)) if len(values) else None, "max": float(np.max(values)) if len(values) else None}

    result = {"windows": len(rows), "scenes": len({records[i]["scene_id"] for i in episodes}),
        "definition": "Frame0 forecasts only; progress means initial minus 4-second goal distance, not path length or net displacement.",
        "bounded_head_training_supported": bool(supported[2]),
        "observed_4s_bounded_positive": int(((data["events"][rows, -1, 2] > 0) & (data["event_mask"][rows, -1, 2] > 0)).sum()),
        "observed_4s_bounded_labels": int((data["event_mask"][rows, -1, 2] > 0).sum()),
        "interpretation": "The bounded-motion head is trained on the 4-second event, not eventual full-route blockage. Association with later outcomes is descriptive and cannot validate a long-horizon calibrated risk claim.",
        "groups": {}, "route_details": [], "selection_without_risk_terms": []}
    for name, mask in groups.items():
        result["groups"][name] = {"routes": int(mask.sum()), "scenes": len({records[episodes[j]]["scene_id"] for j in np.flatnonzero(mask)}),
            "measured_4s_goal_progress_m": distribution(measured_progress[mask & observed]),
            "predicted_4s_goal_progress_m": distribution(predicted_progress[mask]),
            "goal_progress_mae_m": float(np.abs(predicted_progress[mask & observed] - measured_progress[mask & observed]).mean()) if (mask & observed).any() else None,
            "predicted_bounded_probability_at4s": distribution(probability[mask, -1]),
            "predicted_bounded_probability_max2to4s": distribution(bounded_probability[mask])}
    for j, episode in enumerate(episodes):
        result["route_details"].append({"episode": records[episode]["id"], "validation_row": int(rows[j]),
            "eventual_blocked": bool(eventual_blocked[j]), "eventual_safe_goal": bool(eventual_safe[j]),
            "measured_4s_goal_progress_m": float(measured_progress[j]) if observed[j] else None,
            "predicted_4s_goal_progress_m": float(predicted_progress[j]),
            "predicted_bounded_probability_at4s": float(probability[j, -1]),
            "predicted_bounded_probability_max2to4s": float(bounded_probability[j])})
    comparable = eventual_blocked | eventual_safe
    result["descriptive_eventual_block_auroc"] = {
        "from_negative_predicted4s_progress": binary_metrics(eventual_blocked[comparable], -predicted_progress[comparable])["auroc"],
        "from_predicted4s_bounded_probability": binary_metrics(eventual_blocked[comparable], probability[comparable, -1])["auroc"],
        "caution": "Different label horizon; no threshold fitting and no claim that the supervised 4-second risk head predicts full-route failure."}
    for scene in ranking["scenes"]:
        if not scene["scene"].startswith("focus_terrain_"):
            continue
        candidate = min(scene["candidates"], key=lambda c: (
            c["prediction"]["estimated_time_to_goal_s"] - config.terminal_progress_weight * c["prediction"]["predicted_goal_progress_m"], c["episode"]))
        result["selection_without_risk_terms"].append({"scene": scene["scene"],
            "full_cost_selected": scene["selected_episode"], "full_cost_target_safe": scene["selected_measured"]["target_safe_goal"] if scene["selected_measured"] else None,
            "motion_only_selected": candidate["episode"], "motion_only_target_safe": candidate["measured"]["target_safe_goal"],
            "choice_changed": candidate["episode"] != scene["selected_episode"],
            "definition": "Reuse the same forecasts; remove both contact/bounded penalties and predicted-risk rejection thresholds. No parameters or predictions are refit."})
    return result


def load_prediction(path, data):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as file:
        keys = ("trajectory", "work", "event_logits")
        if any(key not in file for key in keys):
            return None
        result = {key: file[key].copy() for key in keys}
    expected = {"trajectory": data["trajectory"].shape, "work": data["work"].shape, "event_logits": data["events"].shape}
    if any(result[key].shape != expected[key] or not np.isfinite(result[key]).all() for key in result):
        raise ValueError(f"Prediction shape/order or finiteness mismatch: {path}")
    return result


def parity(runs):
    available = [run for run in runs if run.get("available")]
    same = lambda values: bool(values) and len({fingerprint(value) for value in values}) == 1
    report = {"available_runs": len(available), "expected_runs": 8,
              "all_eight_complete": len(available) == 8 and all(run["state"] == "complete" for run in available),
              "same_data": same([run["provenance"]["data"] for run in available]),
              "same_code": same([run["provenance"]["code"] for run in available]),
              "progress_event_definitions": sorted({run["progress_event_definition"] for run in available}),
              "same_progress_target": same([run["progress_event_definition"] for run in available]),
              "all_rotation_enabled": all(run["rotation_augmentation"] for run in available), "by_seed": {}, "by_architecture": {}}
    for seed in (11, 29):
        subset = [run for run in available if run["seed"] == seed]
        report["by_seed"][str(seed)] = {"runs": len(subset),
            "same_final_step": same([run["last_step"] for run in subset]),
            "same_sample_draws": same([run["draw_digest"] for run in subset]) and all(run["draw_digest"] for run in subset),
            "same_rotation_draws": same([run["augmentation_digest"] for run in subset]) and all(run["augmentation_digest"] for run in subset)}
    for architecture in ("global", "patch"):
        subset = [run for run in available if run["architecture"] == architecture]
        report["by_architecture"][architecture] = {"parameter_counts": sorted({run["parameters"] for run in subset}),
            "modality_capacity_matched": same([run["parameters"] for run in subset])}
    return report


def render(report):
    lines = ["# Focused RGB-D experiment", "", "Offline frame-0 selection among six recorded sibling routes per held-out development scene. No MPPI refinement or new execution occurs in this report.", "",
             f"Progress target: `{report['progress_event_definition']}`, semantic version {report['progress_event_version']}. Target-safe goal requires measured goal completion without contact or the named progress event; the original recorded safe-goal field is retained separately in JSON.", "",
             "| Run | Checkpoint | Step | Target-safe goal | Contact | Strict stall | Bounded motion | Abstain | Safe time (s) | Safe regret (s) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda x: "—" if x is None else f"{x:.3f}"
    for run in report["runs"]:
        if not run.get("available"):
            continue
        for checkpoint in ("last", "best"):
            item = run["checkpoints"].get(checkpoint, {})
            normal = item.get("controls", {}).get("normal", {})
            if "ranking" not in normal:
                continue
            row = normal["ranking"]["summary"]
            lines.append(f"| {run['name']} | {checkpoint} | {item['step']} | {row['target_safe_goal']}/{row['scenes']} | {row['contacts']} | {row['sustained_stalls']} | {row['bounded_motion'] if row['bounded_motion'] is not None else '—'} | {row['abstentions']} | {fmt(row['mean_safe_goal_time_s'])} | {fmt(row['mean_safe_completion_time_regret_s'])} |")
    lines.extend(["", "Time regret is defined only for actually safe completed selections, relative to the fastest actually safe completed sibling in that scene. Failure and abstention receive no invented travel time.", "", "Image interventions (same fixed-last checkpoint):", "", "| Run | Control | Target-safe goal | Contact | Strict stall | Bounded motion | Abstain | Choices changed | FDE change (m) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for run in report["runs"]:
        for control, item in run.get("checkpoints", {}).get("last", {}).get("controls", {}).items():
            if "ranking" not in item:
                continue
            row = item["ranking"]["summary"]; change = item.get("relative_to_normal", {})
            lines.append(f"| {run['name']} | {control} | {row['target_safe_goal']}/{row['scenes']} | {row['contacts']} | {row['sustained_stalls']} | {row['bounded_motion'] if row['bounded_motion'] is not None else '—'} | {row['abstentions']} | {change.get('scene_choices_changed', 0)} | {fmt(change.get('fde_delta_m'))} |")
    terrain_runs = [run for run in report["runs"] if "groups" in run.get("checkpoints", {}).get("last", {}).get("controls", {}).get("normal", {}).get("terrain_frame0", {})]
    if terrain_runs:
        truth = terrain_runs[0]["checkpoints"]["last"]["controls"]["normal"]["terrain_frame0"]
        blocked, safe = truth["groups"]["eventual_blocked"], truth["groups"]["eventual_safe_goal"]
        lines.extend(["", "Terrain frame-0 forecasts, fixed-last normal images:", "",
            f"Measured validation outcomes: {blocked['routes']} eventual blocked routes and {safe['routes']} safe completed routes. Mean four-second goal progress is {fmt(blocked['measured_4s_goal_progress_m']['mean'])} m versus {fmt(safe['measured_4s_goal_progress_m']['mean'])} m. Bounded-event positives within four seconds: {truth['observed_4s_bounded_positive']}/{truth['observed_4s_bounded_labels']}.", "",
            "| Run | Predicted 4s progress blocked / safe (m) | Bounded probability blocked / safe | Choices changed without risk terms |",
            "|---|---:|---:|---:|"])
        for run in terrain_runs:
            diagnostic = run["checkpoints"]["last"]["controls"]["normal"]["terrain_frame0"]
            blocked, safe = diagnostic["groups"]["eventual_blocked"], diagnostic["groups"]["eventual_safe_goal"]
            progress = " / ".join(fmt(group["predicted_4s_goal_progress_m"]["mean"]) for group in (blocked, safe))
            probability = " / ".join("—" if group["predicted_bounded_probability_at4s"]["mean"] is None
                else f"{group['predicted_bounded_probability_at4s']['mean']:.3g}" for group in (blocked, safe))
            changed = sum(scene["choice_changed"] for scene in diagnostic["selection_without_risk_terms"])
            lines.append(f"| {run['name']} | {progress} | {probability} | {changed}/{diagnostic['scenes']} |")
        lines.extend(["", "These groups use eventual measured outcomes. Terrain frame-0 four-second bounded-event labels and their positive count are reported separately; later blockage is outside that supervised horizon when no positive is yet observed."])
    lines.extend(["", "Limits:", ""] + ["- " + value for value in report["limitations"]])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cost-config", type=Path, help="Previously declared RGBDCostConfig JSON; default matches deployed scorer")
    parser.add_argument("--progress-target", choices=("sustained_stall", "bounded_motion"), default="sustained_stall",
                        help="Explicit semantics; bounded_motion requires the separately recorded version2 target")
    parser.add_argument("--terrain-support", type=Path, help="Optional audited terrain horizon-support JSON; known adjacent v2 audit is used if present")
    args = parser.parse_args()
    config = RGBDCostConfig(**read(args.cost_config)) if args.cost_config else RGBDCostConfig()
    manifest, data_hashes, data, records, outcomes, checks = verify_data(args.data, args.progress_target)
    groups = strata(data, records, args.progress_target)
    dt = float(manifest["output_dt_s"])
    report = {"schema": 1, "evaluation": "offline recorded-family selection at frame0; no MPPI refinement",
        "data": str(args.data.resolve()), "data_hashes": data_hashes, "validation_episode_records_sha256": sha256(args.data / "val_episodes.json"),
        "progress_event_definition": args.progress_target, "progress_event_version": PROGRESS_TARGET_VERSIONS[args.progress_target],
        "progress_event_definition_text": manifest.get(args.progress_target + "_definition"), "cost_config": asdict(config),
        "cost_config_source": {"path": str(args.cost_config.resolve()), "sha256": sha256(args.cost_config)} if args.cost_config else "RGBDCostConfig defaults",
        "cost_implementation": {"path": "src/nedm/traverse/fdm_rgbd_planner.py", "sha256": sha256(ROOT / "src/nedm/traverse/fdm_rgbd_planner.py")},
        "measured_outcomes_verified": checks, "support": manifest["splits"]["val"]["event_support"], "runs": [],
        "limitations": [f"Only {manifest['splits']['val']['scenes']} held-out development scenes; their overlapping windows and six siblings are not independent trials.",
            "Validation was used for best-checkpoint selection. Fixed-last is a separate predeclared update count; neither is an untouched test evaluation.",
            "Failure-focused validation prevalence is designed, so Brier/ECE and risk thresholds are not natural deployment calibration.",
            "Four-second target labels and measured full-route events have different time coverage; later failures may be outside the model horizon.",
            "This ranks six actually recorded reference families; it does not evaluate MPPI-refined unseen routes or receding-horizon replanning."]}
    rollover_counts = {split: manifest["splits"][split]["event_support"]["rollover"]["all"]["positive_horizon_labels"]
                       for split in ("train", "val")}
    report["rollover_positive_horizon_labels"] = rollover_counts
    if not any(rollover_counts.values()):
        report["limitations"].append("Rollover has no positive training or validation support and is excluded from operational cost.")
    else:
        report["limitations"].append(f"Rollover has {rollover_counts['train']} positive training horizon labels and {rollover_counts['val']} validation labels; the declared scorer still omits rollover cost, so this evaluation cannot establish rollover-safe planning.")
    frame0 = groups["frame0"]
    frame0_positive = int(((data["events"][:, -1, 2] > 0) & (data["event_mask"][:, -1, 2] > 0) & frame0).sum())
    if frame0_positive == 0:
        report["limitations"].append(f"Frame0 validation has zero {args.progress_target}-positive 4-second labels; event discrimination cannot be validated at those anchors.")
    report["measured_goal_radii_m"] = sorted({outcome["goal_radius_m"] for outcome in outcomes})
    if any(not np.isclose(radius, config.goal_radius_m) for radius in report["measured_goal_radii_m"]):
        report["limitations"].append("Planning and measured goal radii differ; the exact criteria are archived separately.")
    support_path = args.terrain_support or args.data.parent / "fdm_rgbd_focused_v1/support_audit/terrain_horizon_support_v2.json"
    if args.progress_target == "bounded_motion" and support_path.exists():
        audit = read(support_path)
        audit_val = {record["id"]: record for record in audit["episodes"] if record["split"] == "val"}
        terrain = {record["id"]: i for i, record in enumerate(records) if record["scene_id"].startswith("focus_terrain_")}
        if set(audit_val) != set(terrain):
            raise ValueError("Terrain support audit has different validation episodes")
        for name, index in terrain.items():
            row = int(np.flatnonzero((data["episode_index"] == index) & (data["anchor"] == 0))[0])
            goal = data["global_features"][row, 1:3]
            progress = np.linalg.norm(goal) - np.linalg.norm(goal - data["trajectory"][row, -1, :2])
            if not np.isclose(progress, audit_val[name]["goal_progress4s_m"], atol=1e-4) or bool(audit_val[name]["bounded"]) != outcomes[index]["bounded_motion"]:
                raise ValueError("Terrain horizon audit differs from packed measured progress or full-route diagnosis")
        report["terrain_horizon_support"] = {"path": str(support_path.resolve()), "sha256": sha256(support_path),
            "summary": audit["summary"], "validation_cross_checked_against_pack_and_outcomes": True}
    images = np.load(args.data / "val_rgbd.npy", mmap_mode="r")
    report["frame0_observation_parity"] = {}
    for scene in sorted({record["scene_id"] for record in records}):
        rows = [int(np.flatnonzero((data["episode_index"] == i) & (data["anchor"] == 0))[0])
                for i, record in enumerate(records) if record["scene_id"] == scene]
        report["frame0_observation_parity"][scene] = {
            "history_max_difference": float(np.abs(data["history"][rows] - data["history"][rows[0]]).max()),
            "anchor_pose_max_difference": float(np.abs(data["global_features"][rows, 4:8] - data["global_features"][rows[0], 4:8]).max()),
            "rgbd_max_difference": float(np.abs(images[rows].astype(np.float32) - images[rows[0]].astype(np.float32)).max())}
    nominal = {"trajectory": data["nominal_pose"], "work": np.zeros_like(data["work"]), "event_logits": np.full_like(data["events"], -80.)}
    report["nominal_kinematics_baseline"] = rank_siblings(nominal, data, records, outcomes, np.zeros(3, bool), dt, config, args.progress_target)
    for seed in (11, 29):
        for architecture in ("global", "patch"):
            for arm in ("rgbd", "blank"):
                name = f"{architecture}_{arm}_s{seed}"; path = args.runs / name
                run = {"name": name, "architecture": architecture, "arm": arm, "seed": seed, "available": False}
                report["runs"].append(run)
                required = ("config.json", "provenance.json", "normalization.json", "metrics_last.json", "status.json")
                if not all((path / filename).exists() for filename in required):
                    run["unavailable"] = "Run or required saved artifacts are not yet available"; continue
                cfg, provenance, norm = read(path / "config.json"), read(path / "provenance.json"), read(path / "normalization.json")
                if provenance["data"] != data_hashes:
                    raise ValueError(f"{name}: training data hashes differ from evaluated focused pack")
                model = cfg["model"]
                if (model["arm"] != arm or cfg["arguments"]["seed"] != seed or bool(model["candidate_patches"]) != (architecture == "patch")
                        or model["progress_event_definition"] != args.progress_target):
                    raise ValueError(f"{name}: architecture, seed, modality, or target differs from named experiment")
                last = read(path / "metrics_last.json")
                run.update(available=True, state=read(path / "status.json")["state"], parameters=cfg["parameter_count"],
                    progress_event_definition=model["progress_event_definition"], rotation_augmentation=cfg["arguments"]["rotation_augmentation"],
                    provenance={"data": provenance["data"], "code": provenance["code"]}, last_step=last["step"],
                    draw_digest=last["draw_digest"], augmentation_digest=last.get("augmentation_digest"), checkpoints={})
                run["declared_final_steps"] = cfg["arguments"]["steps"]
                if run["state"] == "complete" and last["step"] != run["declared_final_steps"]:
                    raise ValueError(f"{name}: completed run is not at its declared final update")
                for checkpoint in ("last", "best"):
                    metric_path = path / f"metrics_{checkpoint}.json"
                    if not metric_path.exists():
                        run["checkpoints"][checkpoint] = {"unavailable": "Saved metrics missing"}; continue
                    metric = read(metric_path)
                    item = {"step": metric["step"], "aggregate": compact_metrics(metric["validation"]), "controls": {}}
                    run["checkpoints"][checkpoint] = item
                    controls_path = path / f"{checkpoint}_controls.json"
                    if not controls_path.exists():
                        item["controls_unavailable"] = "Matching checkpoint intervention metadata is not yet saved"; continue
                    control_metadata = read(controls_path)
                    if (control_metadata["checkpoint_step"] != metric["step"]
                            or control_metadata["progress_event_definition"] != args.progress_target):
                        raise ValueError(f"{name}: {checkpoint} control predictions have the wrong checkpoint or target")
                    item["control_metadata_sha256"] = sha256(controls_path)
                    predictions = {}
                    for control in ("normal", "shuffle", "blank"):
                        filename = f"val_predictions_{'last_' if checkpoint == 'last' else ''}{control}.npz"
                        prediction_path = path / filename
                        prediction = load_prediction(prediction_path, data)
                        if prediction is None:
                            item["controls"][control] = {"unavailable": f"Missing complete trajectory/work/risk outputs: {filename}"}; continue
                        predictions[control] = prediction
                        item["controls"][control] = {"prediction_sha256": sha256(prediction_path),
                            "strata": stratified_metrics(prediction, data, groups, norm["supported_events"], dt, args.progress_target),
                            "ranking": rank_siblings(prediction, data, records, outcomes, norm["supported_events"], dt, config, args.progress_target)}
                        if args.progress_target == "bounded_motion":
                            item["controls"][control]["terrain_frame0"] = terrain_frame0_diagnostic(
                                prediction, data, records, outcomes, item["controls"][control]["ranking"], config, norm["supported_events"], dt)
                    if "normal" in predictions:
                        normal = item["controls"]["normal"]
                        for control in ("shuffle", "blank"):
                            if control not in predictions:
                                continue
                            target = item["controls"][control]
                            fde, base_fde = target["strata"]["all"]["pose"]["fde_m"], normal["strata"]["all"]["pose"]["fde_m"]
                            target["relative_to_normal"] = {
                                "fde_delta_m": fde - base_fde if fde is not None and base_fde is not None else None,
                                "max_xy_prediction_change_m": float(np.abs(predictions[control]["trajectory"][..., :2] - predictions["normal"]["trajectory"][..., :2]).max()),
                                "scene_choices_changed": sum(a["selected_episode"] != b["selected_episode"] for a, b in zip(target["ranking"]["scenes"], normal["ranking"]["scenes"]))}
    report["parity"] = parity(report["runs"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, report)
    args.out.with_suffix(".md").write_text(render(report))
    print(json.dumps({"report": str(args.out), "available_runs": report["parity"]["available_runs"],
                      "complete": report["parity"]["all_eight_complete"], "verified_outcomes": len(checks)}))


if __name__ == "__main__":
    main()
