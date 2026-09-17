#!/usr/bin/env python3
"""Compare four frozen finite-horizon forecasts with their actual Chrono runs.

Role names are scene-designed hypotheses. Event probabilities are never edited
to match them. Four-second supervision and later full-run outcomes are reported
separately, with independently checked shared initial observations and hashes.
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
from nedm.traverse.fdm_bounded_targets import bounded_motion_endpoints, BOUNDED_MOTION_DEFINITION
from nedm.traverse.fdm_rgbd_diagnostics import route_digest, serializable


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as file:
        return {key: file[key].copy() for key in file.files}


def verify_frozen(root):
    prediction = root / "prediction"
    frozen_path = prediction / "frozen_artifacts.json"
    frozen = read(frozen_path)
    execution = read(root / "execution_protocol.json")
    if sha(frozen_path) != execution["frozen_prediction_manifest_sha256"] or frozen != execution["frozen_prediction_artifacts"]:
        raise ValueError("Prediction manifest differs from pre-execution protocol")
    for name, expected in frozen.items():
        path = (prediction / name).resolve()
        if not path.is_relative_to(prediction.resolve()) or sha(path) != expected:
            raise ValueError(f"Changed frozen prediction artifact: {name}")
    selection, candidates = read(prediction / "selection.json"), read(prediction / "candidates.json")
    provenance = selection["provenance"]
    observation_path = Path(provenance["observation"])
    if sha(observation_path) != provenance["observation_sha256"] or provenance["observation_sha256"] != execution["observation_sha256"]:
        raise ValueError("Scored observation differs from frozen execution context")
    checkpoint = Path(provenance["checkpoint"])
    checkpoint_check = "not locally available; expected SHA retained"
    if checkpoint.exists():
        if sha(checkpoint) != provenance["checkpoint_sha256"]:
            raise ValueError("Checkpoint differs from the frozen predictor")
        checkpoint_check = "verified"
    records = {record["id"]: record for record in candidates["candidates"]}
    for selected in selection["routes"]:
        route_path = Path(selected["reference_json"])
        route = read(route_path)
        if sha(route_path) != selected["route_sha256"] or route_digest(route) != selected["candidate_id"]:
            raise ValueError("Selected reference differs from the frozen scored candidate")
        if route_digest(records[selected["candidate_id"]]["route"]) != selected["candidate_id"]:
            raise ValueError("Forecast record has a different candidate command")
    return selection, candidates, records, load_npz(observation_path), execution, {
        "prediction_manifest_sha256": sha(frozen_path), "prediction_artifacts": frozen,
        "checkpoint_sha256": provenance["checkpoint_sha256"], "checkpoint_status": checkpoint_check,
        "observation_sha256": provenance["observation_sha256"], "execution_protocol_sha256": sha(root / "execution_protocol.json")}


def check_anchor(actual, expected, equality, expected_sha, trajectory):
    result = {"reported_match": bool(equality["matched"]),
        "expected_observation_sha256_matches": equality["anchor_observation_sha256"] == expected_sha,
        "rgb_equal": bool(np.array_equal(actual["rgb"], expected["rgb"])),
        "depth_max_abs_difference_m": float(np.abs(actual["depth_m"] - expected["depth_m"]).max()),
        "state_max_abs_difference": float(np.abs(actual["state"] - expected["state"]).max()),
        "pose_max_abs_difference": float(np.abs(actual["pose"] - expected["pose"]).max()),
        "history_max_abs_difference": float(np.abs(actual["history"] - expected["history"]).max()),
        "goal_max_abs_difference": float(np.abs(actual["goal_xy"] - expected["goal_xy"]).max()),
        "first_telemetry_pose_matches": bool(np.allclose(trajectory["pose"][0], expected["pose"], rtol=0., atol=2e-6)),
        "first_telemetry_state_matches": bool(np.allclose(trajectory["state"][0], expected["state"], rtol=1e-6, atol=1e-5))}
    result["verified"] = bool(result["reported_match"] and result["expected_observation_sha256_matches"]
        and result["rgb_equal"] and result["depth_max_abs_difference_m"] <= 1e-4
        and np.allclose(actual["state"], expected["state"], rtol=1e-6, atol=1e-5)
        and result["pose_max_abs_difference"] <= 2e-6 and np.allclose(actual["history"], expected["history"], rtol=1e-6, atol=1e-5)
        and result["goal_max_abs_difference"] <= 2e-6 and result["first_telemetry_pose_matches"] and result["first_telemetry_state_matches"])
    if not result["verified"]:
        raise ValueError("Actual execution does not share the scored initial observation")
    return result


def event_comparison(probability, threshold, first_time, horizon, complete_horizon):
    positive = first_time is not None and first_time <= horizon + 1e-7
    observed = positive or complete_horizon
    predicted = probability > threshold
    return {"predicted_probability": probability, "declared_threshold": threshold,
        "predicted_high_risk": predicted, "measured_within_horizon": positive if observed else None,
        "horizon_label_observed": observed, "first_measured_time_s": first_time,
        "within_horizon_false_negative": bool(positive and not predicted) if observed else None,
        "within_horizon_false_positive": bool(not positive and predicted) if observed else None,
        "later_event_outside_forecast": bool(first_time is not None and first_time > horizon + 1e-7)}


def compare_route(root, selected, forecast, expected, execution, cost):
    folder = root / "actual" / selected["id"]
    required = ("trajectory.npz", "outcome.json", "observation.npz", "anchor_equality.json", "simulation_provenance.json")
    result = {"id": selected["id"], "role_hypothesis": selected["label"], "role_semantics": selected["role_semantics"],
              "candidate_id": selected["candidate_id"], "available": False}
    if not all((folder / name).exists() for name in required):
        result["unavailable"] = "Complete actual telemetry/outcome/anchor artifacts have not arrived"
        return result
    telemetry, actual_obs = load_npz(folder / "trajectory.npz"), load_npz(folder / "observation.npz")
    outcome, simulation = read(folder / "outcome.json"), read(folder / "simulation_provenance.json")
    if outcome["route_sha256"] != selected["route_sha256"] or outcome["case_sha256"] != execution["case_sha256"]:
        raise ValueError("Actual run executed a different reference or scene")
    legacy_physics_hash = execution.get("physics_script_sha256")
    source_physics_hash = execution.get("source_sha256", {}).get("scripts/traverse_fdm_rgbd_chrono.py")
    if legacy_physics_hash and source_physics_hash and legacy_physics_hash != source_physics_hash:
        raise ValueError("Conflicting frozen physics script hashes")
    physics_hash = legacy_physics_hash or source_physics_hash
    if not physics_hash:
        raise ValueError("Frozen execution protocol does not identify the physics script")
    if simulation["case_sha256"] != execution["case_sha256"] or simulation["script_sha256"] != physics_hash:
        raise ValueError("Actual simulation differs from the frozen execution protocol")
    equality = check_anchor(actual_obs, expected, read(folder / "anchor_equality.json"), execution["observation_sha256"], telemetry)
    n = len(telemetry["action"])
    dt = float(telemetry["dt_s"])
    if not np.isclose(dt, .05) or len(telemetry["pose"]) != n or len(telemetry["state"]) != n:
        raise ValueError("Expected aligned20Hz pre-interval telemetry")
    poses = np.concatenate((telemetry["pose"], telemetry["terminal_pose"][None]))
    parked = np.r_[telemetry["parked"].astype(bool), bool(telemetry["terminal_parked"])]
    if not np.isfinite(poses).all() or not np.isfinite(telemetry["state"]).all() or not np.isfinite(telemetry["action"]).all():
        raise ValueError("Nonfinite measured telemetry")
    times = np.asarray(forecast["forecast_times_s"], float)
    horizon = float(forecast["forecast_horizon_s"])
    offsets = np.rint(times / dt).astype(int)
    if not np.allclose(offsets * dt, times, atol=1e-6):
        raise ValueError("Learned output times do not align to measured telemetry")
    observed = offsets <= n
    predicted = np.asarray(forecast["forecast_world_xy"], float)[1:]
    errors = np.linalg.norm(predicted[observed] - poses[offsets[observed], :2], axis=1)
    initial_distance = float(np.linalg.norm(expected["pose"][:2] - expected["goal_xy"]))
    complete_horizon = bool(observed[-1])
    measured_progress = initial_distance - float(np.linalg.norm(poses[offsets[-1], :2] - expected["goal_xy"])) if complete_horizon else None
    first_contact = np.flatnonzero(telemetry["contact_n"] > 1.)
    contact_interval_start = int(first_contact[0]) * dt if len(first_contact) else None
    contact_time = contact_interval_start
    contact_resolution = "First20Hz interval with >1N contact; event time is interval-localized"
    contact_log = folder / "contact_events.json"
    if contact_log.exists():
        events = read(contact_log)["events"]
        samples = [float(frame) * dt + (float(substep) + 1) * simulation["physics_dt_s"]
                   for frame, substep, _asset, force in events if force > 1.]
        if samples:
            contact_time = min(samples)
            contact_resolution = "First recorded post-physics-substep force sample >1N"
    endpoints = bounded_motion_endpoints(poses, telemetry["action"], parked,
        goal_xy=expected["goal_xy"], goal_radius_m=float(expected["goal_radius_m"]))
    bounded = np.flatnonzero(endpoints)
    first_bounded = int(bounded[0]) * dt if len(bounded) else None
    # A second diagnostic retains the original strict near-zero-body-vx label.
    strict = (np.abs(telemetry["state"][:, 0]) < .3) & (telemetry["action"][:, 1] > .3) & ~parked[:-1]
    run = 0; first_strict = None
    for index, flag in enumerate(strict):
        run = run + 1 if flag else 0
        if run >= 40 and first_strict is None:
            first_strict = (index + 1) * dt
    contact_probability = float(forecast["scoring"]["contact_probability"])
    bounded_probability = float(forecast["scoring"]["low_progress_probability"])
    progress_predicted = float(forecast["predicted_4s_goal_progress_m"])
    contact_report = event_comparison(contact_probability, cost["max_contact_probability"], contact_time, horizon, complete_horizon)
    contact_report.update(first_contact_interval_start_s=contact_interval_start, time_resolution=contact_resolution)
    bounded_report = event_comparison(bounded_probability, cost["max_low_progress_probability"], first_bounded, horizon, complete_horizon)
    bounded_report["first_qualifying_window_start_s"] = first_bounded - 2. if first_bounded is not None else None
    full_contact, full_bounded = bool(len(first_contact)), bool(len(bounded))
    if full_contact != bool(outcome["asset_contact"]):
        raise ValueError("Per-interval contact telemetry disagrees with saved actual outcome")
    if (first_strict is not None) != bool(outcome["sustained_near_stop"]):
        raise ValueError("Strict stall recomputation disagrees with saved actual outcome")
    result.update(available=True, actual_folder=str(folder.resolve()),
        source_sha256={name: sha(folder / name) for name in required}, shared_initial_observation=equality,
        forecast_horizon_s=horizon, measured_duration_s=n * dt,
        forecast_comparison={"observed_predicted_points": int(observed.sum()), "complete_horizon_observed": complete_horizon,
            "predicted_goal_progress_m": progress_predicted, "measured_goal_progress_m": measured_progress,
            "goal_progress_error_predicted_minus_measured_m": progress_predicted - measured_progress if measured_progress is not None else None,
            "xy_ade_m": float(errors.mean()) if len(errors) else None, "xy_fde_m": float(errors[-1]) if complete_horizon else None,
            "contact": contact_report, "bounded_motion": bounded_report},
        full_run={"status": outcome["status"], "goal_reached": bool(outcome["goal_reached"]), "goal_time_s": outcome["goal_time_s"],
            "contact": full_contact, "bounded_motion_v2": full_bounded, "strict_stall": first_strict is not None,
            "first_strict_stall_confirmed_s": first_strict, "first_bounded_motion_confirmed_s": first_bounded,
            "first_contact_s": contact_time, "goal_progress_m": float(outcome["goal_progress_m"]),
            "target_safe_goal": bool(outcome["goal_reached"] and not full_contact and not full_bounded and outcome["status"] != "rollover"),
            "recorded_safe_goal": bool(outcome["safe_goal_reached"]),
            "legacy_bounded_blockage_v1": outcome.get("bounded_blockage_v1")},
        estimated_time_to_goal={"value_s": float(forecast["scoring"]["estimated_time_to_goal_s"]),
            "interpretation": "Analytic extrapolation from four-second predicted progress; not a learned full-route travel-time forecast"})
    if contact_log.exists():
        result["source_sha256"]["contact_events.json"] = sha(contact_log)
    return result


def render(report):
    number = lambda value: "—" if value is None else f"{value:.3f}"
    yes = lambda value: "unobserved" if value is None else ("yes" if value else "no")
    lines = ["# Four-route forecast / Chrono comparison", "", "One shared measured starting observation. Route labels are frozen test hypotheses, not positive neural-network classifications.", "",
        "| Route | Predicted4s progress (m) | Actual4s progress (m) | Progress error (m) | FDE (m) | Predicted contact | Contact≤4s | Predicted bounded | Bounded≤4s |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|"]
    for route in report["routes"]:
        if not route["available"]:
            lines.append(f"| {route['role_hypothesis']} | Awaiting actual run | | | | | | | |"); continue
        row = route["forecast_comparison"]
        lines.append(f"| {route['role_hypothesis']} | {number(row['predicted_goal_progress_m'])} | {number(row['measured_goal_progress_m'])} | {number(row['goal_progress_error_predicted_minus_measured_m'])} | {number(row['xy_fde_m'])} | {100 * row['contact']['predicted_probability']:.4g}% | {yes(row['contact']['measured_within_horizon'])} | {100 * row['bounded_motion']['predicted_probability']:.4g}% | {yes(row['bounded_motion']['measured_within_horizon'])} |")
    lines.extend(["", "Full actual runs:", "", "| Route | Goal | Goal time (s) | First contact (s) | Bounded motion confirmed (s) | Strict stall confirmed (s) |", "|---|---|---:|---:|---:|---:|"])
    for route in report["routes"]:
        if not route["available"]:
            continue
        row = route["full_run"]
        lines.append(f"| {route['role_hypothesis']} | {yes(row['goal_reached'])} | {number(row['goal_time_s'])} | {number(row['first_contact_s'])} | {number(row['first_bounded_motion_confirmed_s'])} | {number(row['first_strict_stall_confirmed_s'])} |")
    lines.extend(["", "Declared high-risk thresholds: contact35%, bounded motion50%. Bounded motion requires a fully observed, wholly future two-second interval; it can be confirmed later than slowdown begins.", "",
        "An event after4s is outside the learned forecast horizon. The displayed time-to-goal cost extrapolates analytically beyond4s and is not a full-route learned prediction.", "",
        report["four_route_selection_assessment"]["statement"], "", f"Shared initial observation independently verified: {report['shared_initial_observation_verified']}. Frozen predictions, selected references, and execution provenance passed SHA checks."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Fourway experiment directory containing prediction/ and actual/")
    parser.add_argument("--out", type=Path, help="Default ROOT/comparison.json; Markdown is written beside it")
    args = parser.parse_args()
    selected, cloud, records, observation, execution, hashes = verify_frozen(args.root)
    routes = [compare_route(args.root, route, records[route["candidate_id"]], observation, execution, cloud["cost_config"])
              for route in selected["routes"]]
    complete = all(route["available"] for route in routes)
    safe = [route for route in routes if route["available"] and route["full_run"]["target_safe_goal"]]
    best_id = next(route["id"] for route in selected["routes"] if route["category"] == "best")
    best = next(route for route in routes if route["id"] == best_id)
    fastest = min(safe, key=lambda route: route["full_run"]["goal_time_s"]) if safe and complete else None
    regret = (best["full_run"]["goal_time_s"] - fastest["full_run"]["goal_time_s"]
              if fastest and best["available"] and best["full_run"]["target_safe_goal"] else None)
    assessment = {"all_four_runs_available": complete, "selected_route": best_id,
        "fastest_tested_safe_route": fastest["id"] if fastest else None, "selected_time_regret_s": regret,
        "selected_is_fastest_among_tested_safe_routes": bool(abs(regret) < 1e-6) if regret is not None else None,
        "scope": "Only these four preselected, physically executed references; no global route-optimality claim",
        "statement": "Comparison pending complete actual runs."}
    if complete:
        assessment["statement"] = (f"Among these four executed routes, {fastest['id']} was the fastest safe goal completion; selected-route time regret was {regret:.3f}s. This does not establish optimality over all possible routes."
            if regret is not None else "The selected route did not achieve an observed safe goal, or no tested route did. No travel-time regret is assigned to an incomplete or unsafe run.")
    roles = {route["id"]: route["category"] for route in selected["routes"]}
    for route in routes:
        if not route["available"]:
            continue
        actual = route["full_run"]
        category = roles[route["id"]]
        supported = None
        basis = "Post-execution check of the frozen scene-role hypothesis; not a neural-network risk classification."
        if category == "best":
            supported = assessment["selected_is_fastest_among_tested_safe_routes"]
        elif category == "collision":
            supported = actual["contact"]
        elif category == "stall":
            supported = actual["bounded_motion_v2"] and not actual["contact"]
        elif category == "low_progress" and best["available"]:
            actual_progress = route["forecast_comparison"]["measured_goal_progress_m"]
            best_progress = best["forecast_comparison"]["measured_goal_progress_m"]
            if actual_progress is not None and best_progress is not None:
                supported = actual["target_safe_goal"] and actual_progress < best_progress
                basis += " Lower progress is relative to the selected route at the same four-second endpoint, while completing safely."
        route["role_hypothesis_assessment"] = {"supported_by_actual_outcome": supported, "basis": basis}
    report = serializable({"schema": 1, "comparison_script_sha256": sha(__file__), "frozen_verification": hashes,
        "unique_frozen_candidates": cloud["unique_candidates"], "model_event_definition": cloud["model_config"]["progress_event_definition"],
        "bounded_motion_v2_definition": BOUNDED_MOTION_DEFINITION,
        "cost_config": cloud["cost_config"], "shared_initial_observation_verified": complete and all(route["shared_initial_observation"]["verified"] for route in routes),
        "routes": routes, "four_route_selection_assessment": assessment})
    output = args.out or args.root / "comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    output.with_suffix(".md").write_text(render(report))
    print(json.dumps({"comparison": str(output), "available_runs": sum(route["available"] for route in routes), "complete": complete}))


if __name__ == "__main__":
    main()
