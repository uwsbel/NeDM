#!/usr/bin/env python3
"""Read-only physical mechanism audit of completed passive smooth-hill probes."""
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

WHEELS = ("tire_fl", "tire_fr", "tire_rl", "tire_rr")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def window_stats(f, begin, end):
    mask = (f["time_s"] >= begin - 1e-8) & (f["time_s"] < end - 1e-8)
    if not mask.any():
        return None
    wheels = {}
    for wheel in WHEELS:
        force = f[wheel + "_surface_normal_force_n"][mask]
        slip = f[wheel + "_longitudinal_slip"][mask]
        speed = f[wheel + "_surface_tangent_speed_mps"][mask]
        circum = f[wheel + "_circumferential_speed_mps"][mask]
        loaded = force > 200.
        wheels[wheel] = {
            "loaded_fraction_fn_gt200n": float(loaded.mean()),
            "median_normal_force_n": float(np.median(force)),
            "median_tangent_force_n": float(np.median(f[wheel + "_surface_tangent_force_n"][mask])),
            "median_native_slip": float(np.median(slip)),
            "median_native_slip_while_loaded": float(np.median(slip[loaded])) if loaded.any() else None,
            "median_circumferential_speed_mps": float(np.median(circum)),
            "median_surface_tangent_hub_speed_mps": float(np.median(speed)),
            "loaded_stationary_wheelspin_fraction": float(np.mean(loaded & (np.abs(slip) > .5) & (circum > .5) & (np.abs(speed) < .3))),
        }
    return {"begin_s": begin, "end_s": end, "samples": int(mask.sum()), "wheels": wheels,
        "median_pitch_deg": float(np.degrees(np.median(f["pitch_rad"][mask]))),
        "median_runner_shaft_power_kw": float(np.median(f["runner_shaft_power_kw"][mask])),
        "median_chassis_reference_height_above_ground_m": float(np.median(f["chassis_ref_height_above_terrain_m"][mask])),
        "gears": np.unique(f["transmission_current_gear"][mask]).tolist(),
        "minimum_throttle": float(f["driver_throttle"][mask].min()),
        "maximum_brake": float(f["driver_braking"][mask].max()),
        "max_chassis_contact_resultant_n": float(f["chassis_contact_resultant_n"][mask].max())}


def audit(directory):
    t = load(directory / "trajectory.npz")
    f = load(directory / "slope_diagnostics.npz")
    outcome, collection = read(directory / "outcome.json"), read(directory / "collection_meta.json")
    observer = read(directory / "slope_diagnostics.json")
    n, dt = len(t["state"]), float(collection["frame_dt_s"])
    poses = np.vstack([t["pose"], t["terminal_pose"]])
    parked = np.r_[t["parked"], t["terminal_parked"]]
    route = collection["route"]
    goal = np.asarray(route["waypoints"][-1], float)
    radius = float(outcome["goal_radius_m"])
    assert len(f["time_s"]) == n + 1 and np.allclose(f["time_s"], np.arange(n + 1) * dt)
    assert np.allclose(f["pos_x_m"], poses[:, 0]) and np.allclose(f["pos_y_m"], poses[:, 1])
    assert observer["case_sha256"] == collection["case_sha256"]
    endpoints = bounded_motion_endpoints(poses, t["action"], parked, goal_xy=goal, goal_radius_m=radius)
    hits = np.flatnonzero(endpoints)
    first_bounded = int(hits[0]) if len(hits) else None
    arrived = np.maximum.accumulate(np.linalg.norm(poses[:, :2] - goal, axis=1) <= radius)
    effort = (t["action"][:, 1] > .3) & ~parked[:-1] & ~arrived[:-1]
    strict_runs = runs(effort & (np.abs(t["state"][:, 0]) < .3))
    required = int(round(2. / dt))
    strict_hits = [begin + required for begin, end in strict_runs if end - begin >= required]
    last_xy = poses[max(0, n - int(round(5. / dt))):, :2]
    diameter = float(np.sqrt(np.sum((last_xy[:, None] - last_xy[None, :]) ** 2, axis=-1)).max())
    state0 = t["state"][0]
    xy_error = float(np.linalg.norm(poses[0, :2] - np.asarray(route["waypoints"])[0]))
    initial_checks = {"xy_error_lt_point1m": xy_error < .1,
        "abs_body_vx_lt_point1mps": bool(abs(state0[0]) < .1),
        "abs_body_vy_lt_point1mps": bool(abs(state0[1]) < .1),
        "abs_roll_pitch_lt5deg": bool(np.max(np.abs(state0[2:4])) < np.radians(5.)),
        "four_world_vertical_tire_forces_gt200n": bool(np.all(state0[7:11] > 200.))}
    windows = {"last5s": window_stats(f, max(0., n * dt - 5.), n * dt)}
    if first_bounded is not None:
        windows["first_bounded_interval"] = window_stats(f, (first_bounded - required) * dt, first_bounded * dt)
    if strict_hits:
        windows["first_strict_interval"] = window_stats(f, (strict_hits[0] - required) * dt, strict_hits[0] * dt)
    return {"case": directory.parent.name, "family": directory.name, "source_dir": str(directory),
        "input_sha256": {name: sha(directory / name) for name in ("trajectory.npz", "slope_diagnostics.npz", "outcome.json", "collection_meta.json", "slope_diagnostics.json")},
        "observer_sha256": observer["observer_sha256"], "optional_getter_errors": observer["optional_getter_errors"],
        "hill": observer["smooth_hill_design"], "route_meta": route.get("meta", {}),
        "status": outcome["status"], "goal_time_s": outcome["goal_time_s"], "elapsed_s": n * dt,
        "goal_progress_m": outcome["goal_progress_m"], "asset_contact": outcome["asset_contact"],
        "maximum_sampled_chassis_contact_resultant_n": float(f["chassis_contact_resultant_n"].max()),
        "first_bounded_confirmation_s": first_bounded * dt if first_bounded is not None else None,
        "first_strict_stop_confirmation_s": strict_hits[0] * dt if strict_hits else None,
        "longest_strict_stop_s": max((end - begin for begin, end in strict_runs), default=0) * dt,
        "effort_time_s": float(effort.sum() * dt),
        "rollback_time_body_vx_below_minus_point3_s": float(np.sum(t["state"][:, 0] < -.3) * dt),
        "min_body_vx_mps": float(t["state"][:, 0].min()), "last5s_xy_diameter_m": diameter,
        "initial_pose": poses[0].tolist(), "initial_body_vxy_mps": state0[:2].tolist(),
        "initial_roll_pitch_deg": np.degrees(state0[2:4]).tolist(),
        "initial_world_vertical_tire_forces_n": state0[7:11].tolist(), "initial_xy_error_m": xy_error,
        "descriptive_clean_launch_checks": initial_checks, "descriptive_clean_launch_all_pass": all(initial_checks.values()),
        "windows": windows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=Path, required=True, help="Completed probe group containing case/family directories")
    parser.add_argument("--out", type=Path, required=True, help="JSON destination; adjacent Markdown also written")
    args = parser.parse_args()
    directories = sorted(p.parent for p in args.group.glob("*/*/outcome.json"))
    if not directories:
        raise ValueError("No completed probe outcomes")
    rows = [audit(directory) for directory in directories]
    report = {"schema": 1, "group": str(args.group), "analysis_script_sha256": sha(Path(__file__)),
        "bounded_motion_definition": BOUNDED_MOTION_DEFINITION,
        "strict_stop_definition": "At least2s contiguous recorded intervals with abs(body_vx)<.3m/s and throttle>.3; no parking or measured goal arrival",
        "diagnostic_note": "Launch and wheelspin thresholds are descriptive audit checks, not learned event definitions. Wheelspin statistic: normal load>200N, abs(native slip)>.5, forward circumference speed>.5m/s and abs(hub tangent speed)<.3m/s.",
        "limitations": ["Zero20Hz chassis resultant cannot exclude every transient/cancelling contact; no contact-pair census or exact underbody clearance.",
            "Large slip of an unloaded wheel does not establish traction saturation. Report loaded slip separately.",
            "The runner shaft torque-speed product and gear are diagnostics, not proof of an engine-power limit.",
            "Geometry and speed were screened to construct a demonstration; these runs are not an independent performance test set."],
        "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Independent smooth-hill probe audit", "", "Physical outcomes below use measured telemetry; no NN scoring or geometry changes.", "",
        "| Case | Speed | Outcome | Bounded confirmation | Strict stop | Max chassis contact | Clean launch | Last5s diameter |",
        "|---|---:|---|---:|---:|---:|---|---:|"]
    for row in rows:
        lines.append(f"| {row['case']} | {row['route_meta'].get('cruise_speed_mps')} | {row['status']} | {row['first_bounded_confirmation_s']} | {row['first_strict_stop_confirmation_s']} | {row['maximum_sampled_chassis_contact_resultant_n']:.1f}N | {row['descriptive_clean_launch_all_pass']} | {row['last5s_xy_diameter_m']:.3f}m |")
    lines += ["", "All times are seconds from the measured launch. See JSON for source hashes, exact definitions, initial state, effort, rollback, loaded-wheel slip and force measurements.", "", *report["limitations"]]
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(args.out), "rows": len(rows)}))


if __name__ == "__main__":
    main()
