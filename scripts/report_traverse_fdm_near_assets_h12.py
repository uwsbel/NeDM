#!/usr/bin/env python3
"""Compare four frozen near-asset 12 s forecasts with measured Chrono trials.

No inference, training, physics, outcome-based route selection or invented video
frames. ``--manifest`` names a pre-execution reporting_manifest.json. Absolute
cluster paths can be relocated with repeated ``--path-map REMOTE=LOCAL``; the
manifest and payload hashes stay unchanged. ``--metrics-only`` avoids plotting
imports on AMD. ``--encode-video`` requires real timestamped PNGs for all four
routes. Early terminal states remain censored, never extrapolated to 12 seconds.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARRAY_KEYS = ("waypoints", "stations", "headings", "speeds")
COLORS = ("#168c73", "#367db3", "#d65a42", "#9062a9")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as file:
        return {key: file[key].copy() for key in file.files}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def require_hash(path, expected, description):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"{description} SHA mismatch: {path}")
    return actual


class Paths:
    def __init__(self, manifest, mappings):
        self.base = Path(manifest).resolve().parent
        self.mappings = []
        for value in mappings:
            source, sep, target = value.partition("=")
            if not sep or not source or not target:
                raise ValueError("Path mappings must be REMOTE=LOCAL")
            self.mappings.append((Path(source), Path(target).resolve()))
        self.mappings.sort(key=lambda pair: len(str(pair[0])), reverse=True)

    def __call__(self, value):
        path = Path(value)
        if path.is_absolute():
            for source, target in self.mappings:
                if path.is_relative_to(source):
                    return (target / path.relative_to(source)).resolve()
            return path.resolve()
        return (self.base / path).resolve()


def exact_reference(left, right):
    return all(np.array_equal(np.asarray(left[key], np.float64),
                              np.asarray(right[key], np.float64)) for key in ARRAY_KEYS)


def geometry_separation(a, b):
    """Independent declared selection geometry: 61 fractions of first <=72 m."""
    sampled = []
    for route in (a, b):
        xy = np.asarray(route["waypoints"], np.float64)
        station = np.r_[0., np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
        query = np.linspace(0., min(72., float(station[-1])), 61)
        sampled.append(np.column_stack([np.interp(query, station, xy[:, axis]) for axis in (0, 1)]))
    return float(np.linalg.norm(sampled[0] - sampled[1], axis=1).max())


def recompute_cost(predictions, obs, manifest, source_root):
    """Use the frozen scalar scorer on saved outputs, without a model forward."""
    sys.path.insert(0, str(source_root / "src"))
    import torch
    from nedm.traverse.fdm_diverse_planner import RGBDReferenceScorer, RGBDCostConfig
    support = np.asarray(manifest["supported_events"], bool)
    if support.shape != (3,):
        raise ValueError("Expected declared support for contact, rollover, bounded motion")
    scorer = object.__new__(RGBDReferenceScorer)
    scorer.anchor_pose = np.asarray(obs["pose"], np.float64)
    scorer.goal_xy = np.asarray(obs["goal_xy"], np.float64)
    scorer.cost_config = RGBDCostConfig(**manifest["cost_config"])
    scorer.model = SimpleNamespace(config=SimpleNamespace(dt=float(manifest["dt"])),
                                   supported_events=torch.from_numpy(support))
    return scorer.cost_breakdown(predictions), scorer


def normalized_risk(components, config, supported):
    ratios = []
    for key, threshold, enabled in (("contact_probability", "max_contact_probability", supported[0]),
                                    ("low_progress_probability", "max_low_progress_probability", supported[2]),
                                    ("rollover_probability", "max_rollover_probability", supported[1])):
        if enabled:
            limit = float(getattr(config, threshold))
            if limit <= 0.:
                raise ValueError("Normalized-risk selection requires positive declared probability caps")
            ratios.append(np.asarray(components[key]) / limit)
    ratios.extend([components["predicted_peak_roll_deg"] / config.hard_roll_deg,
                   components["predicted_peak_pitch_deg"] / config.hard_pitch_deg])
    return np.max(np.stack(ratios), axis=0)


def verify_selection(candidates, selection, components, risks):
    records = candidates["candidates"]
    if [int(row["prediction_index"]) for row in records] != list(range(len(records))):
        raise ValueError("Candidates must appear in prediction-index order")
    ids = [row["id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate IDs")
    threshold = float(selection["geometric_distinctness_threshold_m"])
    if threshold != 2.:
        raise ValueError("Unexpected declared 2 m within-group separation")
    selected = selection["selected"]
    if len(selected) != 4 or len({row["candidate_id"] for row in selected}) != 4:
        raise ValueError("Exactly four globally distinct selected candidate IDs are required")
    selected_by_group = {group: [row for row in selected if row["predicted_group"] == group]
                         for group in ("preferred", "risky")}
    if any(len(rows) != 2 for rows in selected_by_group.values()):
        raise ValueError("Require two preferred and two higher-predicted-risk references")
    picked = []
    for group in ("preferred", "risky"):
        order = sorted(range(len(records)), key=(lambda i: (float(components["cost"][i]), i))
                       if group == "preferred" else (lambda i: (-float(risks[i]), i)))
        group_picks = []
        for index in order:
            if index in picked or (group == "preferred" and not np.isfinite(components["cost"][index])):
                continue
            if any(geometry_separation(records[index]["route"], records[j]["route"]) < threshold
                   for j in group_picks):
                continue
            group_picks.append(index)
            if len(group_picks) == 2:
                break
        if len(group_picks) != 2:
            raise ValueError(f"Insufficient declared distinct candidates for {group}")
        actual = [int(row["prediction_index"]) for row in selected_by_group[group]]
        if actual != group_picks:
            raise ValueError(f"Frozen {group} selection disagrees with independent greedy ranking: {actual} vs {group_picks}")
        for row, index in zip(selected_by_group[group], group_picks):
            if row["candidate_id"] != records[index]["id"]:
                raise ValueError("Selected prediction index and candidate ID disagree")
            if bool(row["exceeds_risk_gate"]) != (not bool(components["allowed_by_predicted_risk"][index])):
                raise ValueError("Selected risk-gate label disagrees with frozen scorer")
        picked.extend(group_picks)
    return selected


def risk_eligibility(prediction, anchor, goal, dt, radius):
    local = prediction["trajectory"][..., :2]
    c, s = np.cos(anchor[2]), np.sin(anchor[2])
    world = anchor[:2] + np.stack((c*local[:, 0] - s*local[:, 1],
                                  s*local[:, 0] + c*local[:, 1]), axis=-1)
    distance = np.linalg.norm(world - goal, axis=1)
    reached = distance <= radius
    first = int(np.argmax(reached)) if reached.any() else len(distance) - 1
    prefix = np.arange(len(distance)) <= first
    bounded = prefix & ((np.arange(len(distance)) + 1)*dt >= 2. - 1e-6)
    if reached.any():
        bounded &= np.arange(len(distance)) < first
    return world, prefix, bounded, first, bool(reached.any())


def compare_measured(trajectory, rich, intervals, obs, prediction, scorer, component):
    """Measured N intervals/N+1 endpoints; targets only cover observed futures."""
    from nedm.traverse.fdm_data import build_history
    from nedm.traverse.fdm_diverse_targets import prepare_episode_labels, build_diverse_targets
    n = len(trajectory["action"])
    dt = float(trajectory["dt_s"])
    if not np.isclose(dt, .05, rtol=0., atol=1e-12) or n < 1 or n > 240:
        raise ValueError("Expected 1..240 actual 20 Hz intervals for a 12 s trial")
    if trajectory["state"].shape != (n, 17) or trajectory["pose"].shape != (n, 3):
        raise ValueError("Malformed measured state/pose arrays")
    poses = np.concatenate((trajectory["pose"], trajectory["terminal_pose"][None]))
    states = np.concatenate((trajectory["state"], trajectory["terminal_state"][None]))
    parked = np.r_[trajectory["parked"], trajectory["terminal_parked"]].astype(bool)
    if len(rich["time_s"]) != n + 1 or len(intervals["duration_s"]) != n:
        raise ValueError("Rich endpoint/interval alignment mismatch")
    if not np.allclose(rich["time_s"], np.arange(n + 1)*dt, atol=1e-7, rtol=0.):
        raise ValueError("Measured timestamps do not match 20 Hz recording intervals")
    if not np.allclose(intervals["start_time_s"], rich["time_s"][:-1], atol=1e-9, rtol=0.) or not np.allclose(intervals["end_time_s"], rich["time_s"][1:], atol=1e-9, rtol=0.):
        raise ValueError("Rich interval boundaries disagree with measured endpoints")
    if not np.array_equal(trajectory["state"][0], obs["state"]) or not np.array_equal(trajectory["pose"][0], obs["pose"]):
        raise ValueError("Physical state/pose at launch differs from the scored observation")
    history = build_history(trajectory["state"][:1], trajectory["action"][:1], trajectory["pose"][:1], 0)
    if not np.array_equal(history, obs["history"]):
        raise ValueError("Actual causal startup history differs from the scored observation")
    ep = prepare_episode_labels(poses, states, trajectory["action"], parked, intervals,
                                goal_xy=obs["goal_xy"], goal_radius_m=float(obs["goal_radius_m"]))
    attitude = np.column_stack((rich["roll_rad"], rich["pitch_rad"]))
    target = build_diverse_targets(ep, 0, horizon=60, output_dt=.2, attitude=attitude)
    target["events"][:, 2] = target["bounded_motion"][:, 0]
    target["event_mask"][:, 2] = target["bounded_motion_mask"][:, 0]
    world, prefix, bounded, arrival, predicted_arrival = risk_eligibility(prediction, np.asarray(obs["pose"]),
        np.asarray(obs["goal_xy"]), .2, scorer.cost_config.goal_radius_m)
    valid = target["trajectory_mask"][:, 0].astype(bool)
    indices = np.flatnonzero(valid)
    errors = np.linalg.norm(prediction["trajectory"][:, :2] - target["trajectory"][:, :2], axis=-1)
    initial_distance = float(np.linalg.norm(np.asarray(obs["pose"])[:2] - obs["goal_xy"]))
    progress = initial_distance - np.linalg.norm(world - obs["goal_xy"], axis=1)
    observed_progress = initial_distance - np.linalg.norm(poses[:, :2] - obs["goal_xy"], axis=1)
    contact = (ep["contact"] > 1.).any(1)
    asset = ep["contact"][:, 0] > 1.
    chassis = ep["contact"][:, 1] > 1.
    roll = (ep["attitude_peaks"] > np.deg2rad(60.)).any(1)
    first_interval = lambda flag: None if not flag.any() else [float(intervals["start_time_s"][np.flatnonzero(flag)[0]]), float(intervals["end_time_s"][np.flatnonzero(flag)[0]])]
    bounded_indices = np.flatnonzero(ep["bounded_endpoints"])
    events = {}
    for axis, (name, key, cap, mask) in enumerate((("contact", "contact_probability", scorer.cost_config.max_contact_probability, prefix),
                                                  ("rollover", "rollover_probability", scorer.cost_config.max_rollover_probability, prefix),
                                                  ("bounded_motion", "low_progress_probability", scorer.cost_config.max_low_progress_probability, bounded))):
        support = bool(scorer.model.supported_events[axis])
        measured = bool(target["events"][-1, axis]) if target["event_mask"][-1, axis] else None
        probability = float(component[key]) if support else None
        high = None if probability is None else probability > cap
        events[name] = {"supported": support, "planner_eligible_probability": probability, "threshold": cap,
            "predicted_high_risk": high, "measured_within_12s": measured,
            "false_accept_for_event": bool(measured and not high) if measured is not None and high is not None else None,
            "false_reject_for_event": bool(not measured and high) if measured is not None and high is not None else None,
            "eligible_prefix_times_s": (np.flatnonzero(mask) + 1)*.2,
            "raw_unmasked_peak_probability": float(prediction["event_probability"][:, axis].max()) if support else None,
            "target_mask": target["event_mask"][:, axis]}
    endpoint_attitude_mask = target["attitude_mask"].astype(bool)
    attitude_error = np.abs(np.degrees(prediction["attitude"] - target["attitude"]))
    measured_work = float(ep["work"].sum(dtype=np.float64)) if np.isfinite(ep["work"]).all() else None
    summary = {"measured_duration_s": n*dt, "complete_12s_measured": n == 240,
        "observed_forecast_endpoints": len(indices), "causal_launch_history_sha256": hashlib.sha256(history.tobytes()).hexdigest(),
        "predicted_12s_goal_progress_m": float(progress[-1]),
        "measured_12s_goal_progress_m": float(observed_progress[-1]) if n == 240 else None,
        "measured_terminal_goal_progress_m": float(observed_progress[-1]),
        "predicted_12s_positive_engine_interface_work_kj": float(prediction["work"][-1, 0]),
        "measured_12s_positive_engine_interface_work_kj": measured_work if n == 240 else None,
        "measured_terminal_positive_engine_interface_work_kj": measured_work,
        "xy_ade_m_over_observed_endpoints": float(errors[valid].mean()) if len(indices) else None,
        "xy_fde_m_at_12s": float(errors[-1]) if valid[-1] else None,
        "last_observed_forecast_endpoint_s": float((indices[-1] + 1)*.2) if len(indices) else None,
        "xy_error_m_at_last_observed_forecast_endpoint": float(errors[indices[-1]]) if len(indices) else None,
        "signed_attitude_mae_deg_over_observed_endpoints": [float(attitude_error[:, axis][endpoint_attitude_mask[:, axis]].mean()) if endpoint_attitude_mask[:, axis].any() else None for axis in (0, 1)],
        "predicted_peak_abs_roll_pitch_deg_at_endpoints": np.abs(np.degrees(prediction["attitude"])).max(0),
        "measured_peak_abs_roll_pitch_deg_at_endpoints": np.abs(np.degrees(attitude)).max(0),
        "measured_peak_abs_roll_pitch_deg_at_solver_steps": np.degrees(ep["attitude_peaks"].max(0)),
        "first_contact_interval_s": first_interval(contact), "first_asset_contact_interval_s": first_interval(asset),
        "first_chassis_contact_interval_s": first_interval(chassis), "first_rollover_interval_s": first_interval(roll),
        "first_bounded_confirmed_s": float(bounded_indices[0]*dt) if len(bounded_indices) else None,
        "first_bounded_window_s": [float((bounded_indices[0] - 40)*dt), float(bounded_indices[0]*dt)] if len(bounded_indices) else None,
        "planner_pause_s": float(np.dot(np.asarray(rich["command_planner_paused"][:-1], float), intervals["duration_s"])),
        "predicted_goal_arrival_s": float((arrival + 1)*.2) if predicted_arrival else None,
        "events": events, "contact_present_at_launch_endpoint": bool(max(rich.get("asset_contact_max_resultant_n", [np.nan])[0], rich.get("chassis_contact_resultant_n", [np.nan])[0]) > 1.),
        "attitude_scope": "Signed endpoint errors; solver-step peaks reported separately, never equated to endpoint predictions",
        "work_scope": "Positive engine-interface mechanical work in kJ, not fuel or battery energy"}
    return summary, target, {"poses": poses, "states": states, "times_s": np.arange(n + 1)*dt,
                            "progress_m": observed_progress, "attitude": attitude,
                            "cumulative_work_kj": np.r_[0., np.cumsum(ep["work"], dtype=np.float64)], "forecast_world_xy": world}


def route_fingerprint(route):
    h = hashlib.sha256()
    for key in ARRAY_KEYS:
        array = np.asarray(route[key], dtype="<f8")
        h.update(str(array.shape).encode()); h.update(array.tobytes())
    h.update(np.asarray([route.get("meta", {}).get("fdm_station", 0.)], dtype="<f8").tobytes())
    return h.hexdigest()


def verify_execution(folder, route, manifest, manifest_sha, selected, obs, source_files):
    sidecar_path = folder / "near_assets_execution.json"
    sidecar = read(sidecar_path)
    expected = {"start_id": manifest["start_id"], "candidate_id": selected["candidate_id"],
        "prediction_index": route["prediction_index"], "reference_sha256": route["reference_sha256"],
        "reporting_manifest_sha256": manifest_sha, "predictions_sha256": manifest["predictions_sha256"],
        "observation_sha256": manifest["observation_sha256"], "checkpoint_sha256": manifest["checkpoint_sha256"],
        "source_manifest_sha256": manifest["source_manifest_sha256"], "horizon_s": 12.}
    for key, value in expected.items():
        if sidecar.get(key) != value:
            raise ValueError(f"Physical sidecar {key} disagrees with frozen prediction: {folder}")
    if sidecar.get("anchor_exact") is not True:
        raise ValueError("Physical wrapper did not certify its exact launch context")
    files = sidecar["artifacts_sha256"]
    required = ("trajectory.npz", "rich_intervals.npz", "rich_telemetry.npz", "anchor_state.npz",
                "outcome.json", "online_protocol.json", "simulation_provenance.json", "anchor_equality.json")
    for name in required:
        require_hash(folder/name, files[name], f"Actual {name}")
    for name, expected_sha in files.items():
        path = (folder/name).resolve()
        if not path.is_relative_to(folder):
            raise ValueError("Physical artifact digest path escapes trial directory")
        require_hash(path, expected_sha, "Physical artifact")
    if sidecar["runtime_sha256"] != manifest["runtime_sha256"]:
        raise ValueError("Actual and observed runtime fingerprints differ")
    for name, expected_sha in sidecar["source_sha256"].items():
        if name in source_files and source_files[name] != expected_sha:
            raise ValueError(f"Actual frozen source differs: {name}")
    trajectory = load_npz(folder/"trajectory.npz")
    rich = load_npz(folder/"rich_telemetry.npz")
    intervals = load_npz(folder/"rich_intervals.npz")
    anchor = load_npz(folder/"anchor_state.npz")
    outcome = read(folder/"outcome.json")
    equality = read(folder/"anchor_equality.json")
    if not equality.get("matched"):
        raise ValueError("Saved launch equality did not pass")
    for key in ("state", "pose", "history", "goal_xy"):
        if not np.array_equal(anchor[key], obs[key]):
            raise ValueError(f"Measured launch {key} differs from the scored context")
    if outcome.get("planning_decisions") != 1 or outcome.get("planning_mode") != "plan_once":
        raise ValueError("Require exactly one frozen launch reference and no replanning")
    reference = route["reference"]
    if not exact_reference(outcome["final_reference"], reference):
        raise ValueError("Actual final reference differs from the scored selected reference")
    if sidecar["executed_reference_arrays_sha256"] != route_fingerprint(reference):
        raise ValueError("Executed canonical reference fingerprint differs")
    return trajectory, rich, intervals, outcome, {"sidecar_sha256": sha(sidecar_path),
        "verified_artifacts_sha256": files, "runtime_sha256": sidecar["runtime_sha256"],
        "source_sha256": sidecar["source_sha256"], "anchor_exact": True,
        "execution_semantics": "One previously scored frozen reference; sidecar identifies intervention, not a fresh neural decision"}


def verify_video_frames(run):
    """Independent measured state/action/timing checks before existing encoder."""
    folder = run["folder"]
    metadata = read(folder/"frame_metadata.json")
    rows = metadata["frames"]
    if not rows or len(rows) != int(metadata["frame_count"]):
        raise ValueError("Native video frame inventory is missing/inconsistent")
    trajectory = run["trajectory"]
    n = len(trajectory["state"])
    states = np.concatenate((trajectory["state"], trajectory["terminal_state"][None]))
    poses = np.concatenate((trajectory["pose"], trajectory["terminal_pose"][None]))
    previous_time = -np.inf
    for index, row in enumerate(rows):
        frame = int(row["telemetry_frame"])
        if row["video_index"] != index or not 0 <= frame <= n:
            raise ValueError("Camera index is outside measured trajectory")
        if not np.array_equal(np.asarray(row["actual_state17"], np.float32), states[frame]) or not np.array_equal(np.asarray(row["actual_pose"], np.float64), poses[frame]):
            raise ValueError("Actual video state/pose differs from physical telemetry")
        action = trajectory["action"][min(frame, n-1)]
        if not np.array_equal(np.asarray(row["applied_action"], np.float32), action):
            raise ValueError("Actual video action differs from physical telemetry")
        if not np.isclose(row["recording_time_s"], frame*.05, atol=1e-7, rtol=0.) or row["recording_time_s"] <= previous_time:
            raise ValueError("Camera timestamps do not match measured recording time")
        previous_time = row["recording_time_s"]
        path = (folder/row["file"]).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise ValueError("Camera pixel file missing or outside trial directory")
    if rows[0]["telemetry_frame"] != 0 or rows[-1]["telemetry_frame"] != n or not rows[-1]["terminal"]:
        raise ValueError("Video must include actual launch and terminal endpoints")
    return {"passed": True, "frames": len(rows), "nominal_fps": metadata["nominal_fps"],
            "frame_metadata_sha256": sha(folder/"frame_metadata.json")}


def renderer_module():
    path = Path(__file__).with_name("render_traverse_fdm_diverse_online_report.py")
    spec = importlib.util.spec_from_file_location("_near_asset_shared_renderer", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module, path


def render_overview(runs, obs, camera, candidates, manifest, out, shared):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    half = float(manifest["mppi_config"]["arena_half_extent_m"])
    image_map = shared.observed_map(obs, camera, half)
    points = [np.asarray(obs["pose"])[None, :2]]
    for run in runs:
        ref = run["reference"]; ss = np.asarray(ref["stations"]); xy = np.asarray(ref["waypoints"])
        limit = min(float(ss[0])+72., float(ss[-1]))
        points.append(xy[ss <= limit])
        points.append(run["measured"]["poses"][:, :2]); points.append(run["measured"]["forecast_world_xy"])
    extent_points = np.concatenate(points)
    low, high = extent_points.min(0)-5., extent_points.max(0)+5.
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), layout="constrained")
    for ax in axes:
        ax.imshow(image_map["rgb"], origin="lower", extent=image_map["extent"])
        ax.set_xlim(low[0], high[0]); ax.set_ylim(low[1], high[1]); ax.set_aspect("equal")
        ax.set_xlabel("World X (m)"); ax.set_ylabel("World Y (m)")
        ax.scatter(*obs["pose"][:2], marker="o", s=50, c="white", edgecolors="black", zorder=10)
        if (low <= obs["goal_xy"]).all() and (obs["goal_xy"] <= high).all():
            ax.scatter(*obs["goal_xy"], marker="*", s=120, c="#14a477", edgecolors="white", zorder=10)
    for record in candidates["candidates"]:
        xy = np.asarray(record["route"]["waypoints"])
        ss = np.asarray(record["route"]["stations"])
        xy = xy[ss <= min(float(ss[0])+72., float(ss[-1]))]
        axes[0].plot(xy[:, 0], xy[:, 1], color="gray", lw=.45, alpha=.08)
    for color, run in zip(COLORS, runs):
        ref = run["reference"]; ss = np.asarray(ref["stations"]); xy = np.asarray(ref["waypoints"])
        xy = xy[ss <= min(float(ss[0])+72., float(ss[-1]))]
        predicted = np.vstack([obs["pose"][:2], run["measured"]["forecast_world_xy"]])
        measured = run["measured"]["poses"][:, :2]
        axes[0].plot(xy[:, 0], xy[:, 1], color=color, lw=1.6, label=run["label"])
        axes[1].plot(predicted[:, 0], predicted[:, 1], color=color, lw=1.8)
        axes[1].plot(measured[:, 0], measured[:, 1], color=color, lw=2., ls="--")
        axes[1].scatter(*predicted[-1], marker="x", s=35, color=color)
        axes[1].scatter(*measured[-1], marker="o", s=20, color=color)
    axes[0].set_title("Frozen references: first 72 m; grey = evaluated candidates")
    axes[1].set_title("12 s predictions and actual Chrono traversals")
    axes[0].legend(fontsize=8, loc="best")
    axes[1].legend(handles=[Line2D([0], [0], color="black", label="FDM prediction", lw=1.8),
        Line2D([0], [0], color="black", label="Actual measured trajectory", ls="--", lw=2.),
        Line2D([0], [0], color="black", marker="x", ls="", label="Predicted 12 s endpoint"),
        Line2D([0], [0], color="black", marker="o", ls="", label="Actual terminal endpoint")], fontsize=8)
    fig.suptitle(f"{manifest['start_id']} | two model-preferred and two higher-predicted-risk references\n"
                 "Labels were frozen before physics; they do not guarantee the observed outcome", fontsize=13)
    fig.savefig(out/"overview.png", dpi=180); fig.savefig(out/"overview.pdf"); plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), layout="constrained")
    forecast_times = np.arange(1, 61)*.2
    for color, run in zip(COLORS, runs):
        p = run["prediction"]; m = run["measured"]; target = run["target"]
        initial = np.linalg.norm(obs["pose"][:2]-obs["goal_xy"])
        progress = initial-np.linalg.norm(m["forecast_world_xy"]-obs["goal_xy"], axis=1)
        axes[0, 0].plot(forecast_times, progress, color=color, label=run["label"])
        axes[0, 0].plot(m["times_s"], m["progress_m"], color=color, ls="--")
        axes[0, 1].plot(forecast_times, p["work"][:, 0], color=color)
        axes[0, 1].plot(m["times_s"], m["cumulative_work_kj"], color=color, ls="--")
        for axis, col in ((0, 0), (2, 1)):
            eligible = forecast_times >= 2.-1e-6 if axis == 2 else np.ones(60, bool)
            axes[1, col].plot(forecast_times[eligible], p["event_probability"][eligible, axis], color=color)
            observed = target["event_mask"][:, axis].astype(bool)
            axes[1, col].step(forecast_times[observed], target["events"][observed, axis], color=color, ls="--", where="post", alpha=.65)
        for axis in (0, 1):
            axes[2, axis].plot(forecast_times, np.degrees(p["attitude"][:, axis]), color=color)
            axes[2, axis].plot(m["times_s"], np.degrees(m["attitude"][:, axis]), color=color, ls="--")
    titles = (("Goal progress", "m"), ("Positive engine-interface work", "kJ"),
              ("Contact probability / measured cumulative event", "probability / event"),
              ("Bounded-motion probability / measured cumulative event", "probability / event"),
              ("Signed roll", "degrees"), ("Signed pitch", "degrees"))
    for ax, (title, unit) in zip(axes.flat, titles):
        ax.set_title(title); ax.set_xlabel("Time after measured launch (s)"); ax.set_ylabel(unit); ax.grid(alpha=.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle("Solid = frozen FDM output; dashed = actual measured Chrono values\n"
                 "Endpoint attitudes and mechanical work; censored futures remain unobserved", fontsize=12)
    fig.savefig(out/"horizon_comparison.png", dpi=175); fig.savefig(out/"horizon_comparison.pdf"); plt.close(fig)
    return {k: v for k, v in image_map.items() if k not in ("rgb", "height")}


def report_markdown(manifest, runs, out):
    number = lambda value: "unobserved" if value is None else f"{value:.2f}"
    truth = lambda value: "unobserved" if value is None else ("yes" if value else "no")
    lines = ["**Near-asset 12-second prediction / Chrono comparison**", "",
        f"Start: `{manifest['start_id']}`. Four references were frozen before physics: two model-preferred and two with higher predicted risk. A preferred fallback may still be rejected by the model; a higher-risk reference may remain below its rejection limits. These labels do not guarantee an actual safe, colliding or stalled outcome.", "",
        "| Reference | Model gate | Predicted / actual 12 s progress (m) | Predicted / actual 12 s work (kJ) | Position error at 12 s (m) | Predicted contact / measured event | Predicted bounded motion / measured event |",
        "|---|---|---:|---:|---:|---|---|"]
    for run in runs:
        row = run["comparison"]; events = row["events"]
        probability = lambda key: "unsupported" if events[key]["planner_eligible_probability"] is None else f"{events[key]['planner_eligible_probability']:.1%}"
        lines.append(f"| {run['label']} | {'accepted' if run['components']['allowed_by_predicted_risk'] else 'rejected'} | {number(row['predicted_12s_goal_progress_m'])} / {number(row['measured_12s_goal_progress_m'])} | {number(row['predicted_12s_positive_engine_interface_work_kj'])} / {number(row['measured_12s_positive_engine_interface_work_kj'])} | {number(row['xy_fde_m_at_12s'])} | {probability('contact')} / {truth(events['contact']['measured_within_12s'])} | {probability('bounded_motion')} / {truth(events['bounded_motion']['measured_within_12s'])} |")
    lines += ["", "| Reference | Measured duration | First contact interval | First confirmed bounded-motion window | Measured solver-step peak roll / pitch | Media |", "|---|---:|---|---|---:|---|"]
    for run in runs:
        row = run["comparison"]
        interval = lambda value: "none" if value is None else f"{value[0]:.2f}–{value[1]:.2f} s"
        peak = row["measured_peak_abs_roll_pitch_deg_at_solver_steps"]
        video = run["video"]
        media = f"[Actual video]({Path(video['file']).resolve()})" if video.get("available") else video.get("reason", "not encoded")
        lines.append(f"| {run['label']} | {row['measured_duration_s']:.2f} s | {interval(row['first_contact_interval_s'])} | {interval(row['first_bounded_window_s'])} | {peak[0]:.2f}° / {peak[1]:.2f}° | {media} |")
    lines += ["", "Contact is asset or chassis resultant force above 1 N, localized to its measured 50 ms interval. Bounded motion is confirmed only after a fully observed two-second window with XY diameter ≤0.25 m, throttle >0.3 throughout, and no deliberate parking. Rollover and signed attitude results are retained in the machine-readable report.", "",
        "Only measured futures are compared. Early goal/rollover termination leaves the remaining 12-second trajectory, work and negative event labels unobserved; a known positive event remains known. A normal 12-second cap does not mean failure to complete a full route. Positive work is mechanical engine-interface work, not fuel consumption. Endpoint attitude predictions and solver-step attitude peaks are different quantities.", "",
        "Model scores apply the exact frozen goal-arrival mask and the bounded-motion mask for horizons of at least two seconds. The overview uses only the measured depth/RGB map and shows the first 72 m of commanded references. Forecasts end at 12 seconds; full references are commands, not predicted actual trajectories. Actual video frames are native Chrono camera pixels with verified measured states, actions and timestamps, with no synthetic interpolation.", "",
        f"[Machine-readable comparison]({(out/'report.json').resolve()}) · [Verification and artifact hashes]({(out/'report_manifest.json').resolve()})"]
    if (out/"overview.png").exists():
        lines += ["", f"[Four-reference overview]({(out/'overview.png').resolve()}) · [Horizon telemetry comparison]({(out/'horizon_comparison.png').resolve()})"]
    return "\n".join(lines) + "\n"


def run_report(args):
    manifest_path = args.manifest.resolve(); manifest_sha = sha(manifest_path)
    manifest = read(manifest_path); resolve = Paths(manifest_path, args.path_map)
    if manifest.get("schema") != "fdm_near_assets_reporting_v1":
        raise ValueError("Unsupported reporting manifest schema")
    if manifest["horizon_s"] != 12. or manifest["dt"] != .2:
        raise ValueError("Require the frozen 60-step, 12-second forecast")
    if not manifest.get("prediction_frozen_before_physical_execution"):
        raise ValueError("Prediction selection was not declared frozen before physics")
    file_fields = (("observation_npz", "observation_sha256"), ("observation_json", "observation_json_sha256"),
        ("observation_provenance_json", "observation_provenance_sha256"), ("predictions_npz", "predictions_sha256"),
        ("candidates_json", "candidates_sha256"), ("selection_json", "selection_sha256"),
        ("case", "case_sha256"), ("cost_config_file", "cost_config_sha256"))
    verified = {}
    for field, digest in file_fields:
        verified[field] = require_hash(resolve(manifest[field]), manifest[digest], field)
    if resolve(manifest["checkpoint"]).exists():
        verified["checkpoint"] = require_hash(resolve(manifest["checkpoint"]), manifest["checkpoint_sha256"], "Checkpoint")
    source_root = args.code_root.resolve() if args.code_root else resolve(manifest["source_root"])
    require_hash(source_root/"source_manifest.json", manifest["source_manifest_sha256"], "Frozen source inventory")
    source_inventory = read(source_root/"source_manifest.json")
    source_files = source_inventory.get("files", source_inventory.get("file_sha256", {}))
    if not source_files:
        raise ValueError("Missing frozen source file hashes")
    for name, expected in source_files.items():
        if name.endswith(".py"):
            require_hash(source_root/name, expected, "Frozen Python source")
    observation = load_npz(resolve(manifest["observation_npz"]))
    observation_meta = read(resolve(manifest["observation_json"]))
    observation_provenance = read(resolve(manifest["observation_provenance_json"]))
    if observation_provenance["runtime_sha256"] != manifest["runtime_sha256"]:
        raise ValueError("Observation runtime differs from frozen manifest")
    selection = read(resolve(manifest["selection_json"]))
    if selection.get("physical_results_used_for_selection") is not False:
        raise ValueError("Selection must explicitly exclude measured physical future outcomes")
    if selection["prediction_outputs_sha256"] != manifest["predictions_sha256"]:
        raise ValueError("Selection was generated from different prediction tensors")
    if not np.array_equal(np.asarray(selection["anchor_pose"]), observation["pose"]) or not np.array_equal(np.asarray(selection["anchor_state17"], np.float32), observation["state"]):
        raise ValueError("Scoring and observation launch state/pose differ")
    if hashlib.sha256(observation["history"].tobytes()).hexdigest() != selection["history_sha256"]:
        raise ValueError("Scoring and observation launch histories differ")
    goal = np.asarray(selection["goal_xy"], np.float64)
    if not np.array_equal(goal.astype(np.float32), observation["goal_xy"]):
        raise ValueError("Physical/scored supplied goal does not match the stored observation copy")
    goal_copy_difference = float(np.max(np.abs(goal-observation["goal_xy"])))
    obs = dict(observation); obs["goal_xy"] = goal
    candidates = read(resolve(manifest["candidates_json"]))
    predictions = load_npz(resolve(manifest["predictions_npz"]))
    count = len(candidates["candidates"])
    for key, width in (("trajectory", 4), ("event_probability", 3), ("work", 1), ("attitude", 2)):
        if predictions[key].shape != (count, 60, width) or not np.isfinite(predictions[key]).all():
            raise ValueError(f"Invalid frozen prediction shape/value: {key}")
    if (predictions["work"] < -1e-8).any() or (predictions["event_probability"] < 0.).any() or (predictions["event_probability"] > 1.).any():
        raise ValueError("Prediction work/probability outside its declared physical range")
    components, scorer = recompute_cost(predictions, obs, manifest, source_root)
    risks = normalized_risk(components, scorer.cost_config, manifest["supported_events"])
    for index, record in enumerate(candidates["candidates"]):
        if route_fingerprint(record["route"]) != record["route_sha256"]:
            raise ValueError("Candidate canonical route fingerprint differs")
        for key, saved in record["components"].items():
            actual = components[key][index]
            if saved is None:
                if np.isfinite(actual):
                    raise ValueError(f"Saved missing/infinite cost component became finite: {key}")
            elif isinstance(saved, bool):
                if saved != bool(actual):
                    raise ValueError(f"Saved candidate gate/support flag differs: {key}")
            elif not np.isclose(saved, actual, atol=1e-6, rtol=1e-7):
                raise ValueError(f"Saved candidate cost component differs: {key}")
        if not np.isclose(record["normalized_risk"], risks[index], atol=1e-7, rtol=1e-7):
            raise ValueError("Saved normalized risk differs from independent computation")
    selected = verify_selection(candidates, selection, components, risks)
    selected_by_id = {row["id"]: row for row in selected}
    if len(manifest["routes"]) != 4 or set(selected_by_id) != {row["id"] for row in manifest["routes"]}:
        raise ValueError("Reporting manifest does not contain the four selected routes")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty output directory; preserve earlier reports")
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for route in manifest["routes"]:
        route = dict(route); chosen = selected_by_id[route["id"]]
        for key in ("candidate_id", "prediction_index", "predicted_group"):
            if route[key] != chosen[key]:
                raise ValueError(f"Reporting route {key} differs from selected record")
        index = int(route["prediction_index"])
        ref_path = resolve(route["reference_json"])
        require_hash(ref_path, route["reference_sha256"], "Selected reference file")
        route["reference"] = read(ref_path)
        if not exact_reference(route["reference"], candidates["candidates"][index]["route"]) or route_fingerprint(route["reference"]) != candidates["candidates"][index]["route_sha256"]:
            raise ValueError("Selected file is not the scored reference")
        folder = resolve(route["actual_dir"])
        trajectory, rich, intervals, outcome, checks = verify_execution(folder, route, manifest, manifest_sha, chosen, obs, source_files)
        prediction = {key: value[index] for key, value in predictions.items()}
        component = {key: value[index] for key, value in components.items()}
        comparison, target, measured = compare_measured(trajectory, rich, intervals, obs, prediction, scorer, component)
        for field, actual in (("schema_contact", comparison["first_contact_interval_s"] is not None),
                              ("schema_rollover", comparison["first_rollover_interval_s"] is not None),
                              ("bounded_blockage_v1", comparison["first_bounded_confirmed_s"] is not None)):
            if bool(outcome[field]) != actual:
                raise ValueError(f"Recomputed measured event differs from outcome: {field}")
        if not np.isclose(outcome["positive_work_kj"], comparison["measured_terminal_positive_engine_interface_work_kj"], atol=1e-7, rtol=1e-10) or not np.isclose(outcome["elapsed_s"], comparison["measured_duration_s"], atol=1e-7, rtol=0.):
            raise ValueError("Recomputed duration/work differs from saved outcome")
        role_dir = out/route["id"]
        if not role_dir.is_relative_to(out) or not re.fullmatch(r"[A-Za-z0-9_-]+", route["id"]):
            raise ValueError("Invalid selected route output ID")
        role_dir.mkdir()
        np.savez_compressed(role_dir/"comparison_tensors.npz", **{"predicted_"+k: v for k, v in prediction.items()},
                            **{"target_"+k: v for k, v in target.items()}, **{"measured_"+k: v for k, v in measured.items()})
        run = {"id": route["id"], "label": route["label"], "predicted_group": route["predicted_group"],
            "candidate_id": route["candidate_id"], "prediction_index": index, "reference": route["reference"],
            "folder": folder, "trajectory": trajectory, "rich": rich, "prediction": prediction,
            "target": target, "measured": measured, "comparison": comparison,
            "components": component, "normalized_risk": risks[index], "outcome": outcome, "checks": checks,
            "video": {"available": False, "reason": "Encoding not requested"}}
        runs.append(run)
    renderer = None; renderer_path = None; observed_map = None
    if not args.metrics_only or args.encode_video:
        renderer, renderer_path = renderer_module()
    if args.encode_video:
        for run in runs:
            run["checks"]["native_frame_alignment"] = verify_video_frames(run)
            run["video"] = renderer.encode_video(run, out/run["id"], args.ffmpeg, args.ffprobe)
            if not run["video"].get("available"):
                raise ValueError("Requested video encoding has no actual native frames")
            run["video"]["physics_scope"] = "Same measured trial supplying the reported outcomes; no separate replay-parity claim"
    if not args.metrics_only:
        observed_map = render_overview(runs, obs, observation_meta["camera"], candidates, manifest, out, renderer)
    report_runs = [{key: value for key, value in run.items() if key not in ("trajectory", "rich", "prediction", "target", "measured", "reference")} for run in runs]
    provenance = {"script_sha256": sha(__file__), "reporting_manifest_sha256": manifest_sha,
        "source_manifest_sha256": manifest["source_manifest_sha256"], "verified_input_sha256": verified,
        "source_python_files_verified": sum(name.endswith('.py') for name in source_files),
        "checkpoint_sha256": manifest["checkpoint_sha256"], "goal_float32_observation_copy_max_abs_difference_m": goal_copy_difference,
        "scoring_goal_semantics": "Exact float64 supplied goal from frozen selection; float32 observation copy independently checked",
        "supported_events": manifest["supported_events"], "python": platform.python_version(), "numpy": np.__version__,
        "renderer_source_sha256": None if renderer_path is None else sha(renderer_path),
        "boundaries": "No model forward, training, physics stepping, authored terrain visualization, outcome-based route selection or synthesized video frames",
        "selection_recomputed_independently": True, "preferred_fallback_required": selection["preferred_fallback_required"],
        "observed_map": observed_map}
    report = {"schema": "fdm_near_assets_h12_report_v1", "start_id": manifest["start_id"], "scene_id": manifest["scene_id"],
              "horizon_s": 12., "routes": report_runs, "provenance": provenance}
    dump(out/"report.json", report)
    (out/"report.md").write_text(report_markdown(manifest, runs, out))
    output_files = [path for path in out.rglob("*") if path.is_file()]
    dump(out/"report_manifest.json", {"passed": True, "provenance": provenance,
        "artifacts_sha256": {str(path.relative_to(out)): sha(path) for path in output_files}})
    print(json.dumps({"out": str(out), "routes": 4, "videos": sum(run["video"].get("available", False) for run in runs),
                      "complete_12s_trials": sum(run["comparison"]["complete_12s_measured"] for run in runs)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--path-map", action="append", default=[], help="REMOTE=LOCAL; preserve frozen manifest bytes")
    parser.add_argument("--code-root", type=Path, help="Relocated exact source snapshot; all frozen Python hashes are verified")
    parser.add_argument("--metrics-only", action="store_true", help="Skip plotting imports; suitable for AMD runtime without matplotlib")
    parser.add_argument("--encode-video", action="store_true", help="Encode all four actual native-frame sequences with verified timestamps")
    parser.add_argument("--ffmpeg", default="ffmpeg"); parser.add_argument("--ffprobe", default="ffprobe")
    run_report(parser.parse_args())


if __name__ == "__main__":
    main()
