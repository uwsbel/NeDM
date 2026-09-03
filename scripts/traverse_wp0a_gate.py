#!/usr/bin/env python
"""WP0a batch gate: oracle plans tracked by a scripted driver in Chrono (plan §14, G0a).

Runs N independent episodes (sampled layout -> privileged oracle plan ->
ChPathFollowerDriver, no rendering, no learning) in parallel worker processes
and aggregates the G0a gate numbers: approach-pose reach rate (target >= 95%),
collision-free rate (target: zero contact episodes), and layout/plan
feasibility statistics.

Usage (from repo root, nedm env):
  PYTHONPATH=src python scripts/traverse_wp0a_gate.py --episodes 100 --procs 12
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SUCCESS_RADIUS_M = 2.0
SUCCESS_HOLD_S = 0.5
SETTLE_S = 0.8
CONTACT_EPS_N = 1.0  # below this, treat asset contact force as solver noise


def nearest_index(waypoints: np.ndarray, pos: np.ndarray, last: int, window: int = 60) -> int:
    lo = last
    hi = min(len(waypoints), last + window)
    d = np.hypot(waypoints[lo:hi, 0] - pos[0], waypoints[lo:hi, 1] - pos[1])
    return lo + int(np.argmin(d))


def cross_track_m(waypoints: np.ndarray, pos: np.ndarray, idx: int) -> float:
    best = math.inf
    for i in range(max(0, idx - 1), min(len(waypoints) - 1, idx + 1)):
        a, b = waypoints[i], waypoints[i + 1]
        ab = b - a
        denom = float(ab @ ab)
        t = 0.0 if denom < 1e-9 else float(np.clip((pos - a) @ ab / denom, 0.0, 1.0))
        best = min(best, float(np.hypot(*(pos - (a + t * ab)))))
    return best


def run_one(task: tuple[int, int, str, float]) -> dict:
    idx, seed, arena, timeout_scale = task
    import pychrono as chrono
    import pychrono.vehicle as veh

    from nedm.traverse.layout import sample_episode
    from nedm.traverse.scene import build_config, build_scene
    from nedm.traverse.terrain import TerrainMap

    wall0 = time.time()
    arena_dir = (REPO_ROOT / arena).resolve()
    tmap = TerrainMap.from_dir(arena_dir)
    row: dict = {"episode": idx, "seed": seed}

    try:
        layout, plan = sample_episode(tmap, f"gate_{idx:03d}", seed)
    except RuntimeError as exc:
        row.update(status="no_feasible_layout", error=str(exc), wall_s=time.time() - wall0)
        return row
    row.update(
        plan_length_m=float(plan.meta["length_m"]),
        plan_est_duration_s=float(plan.meta["est_duration_s"]),
        layout_attempt=int(plan.meta.get("layout_attempt", 0)),
        n_assets=len(layout.assets),
    )
    approach = np.asarray(plan.meta["approach_pose"][:2])

    start_z = float(tmap.height(*layout.start_xy)) + 0.75
    config = build_config(arena_dir, (*layout.start_xy, start_z), layout.start_yaw)
    scene = build_scene(config, layout, tmap, arena_dir, plan=plan, render=None)
    hmmwv, system, terrain = scene.hmmwv, scene.system, scene.terrain
    vehicle = hmmwv.GetVehicle()

    pts = chrono.vector_ChVector3d()
    last_s = -10.0
    for (x, y), s in zip(plan.waypoints, plan.stations):
        if s - last_s < 2.0 and s != plan.stations[-1]:
            continue
        last_s = s
        pts.append(chrono.ChVector3d(float(x), float(y), float(tmap.height(x, y)) + 0.5))
    path = chrono.ChBezierCurve(pts)
    driver = veh.ChPathFollowerDriver(vehicle, path, "oracle_plan", float(plan.speeds[0]))
    driver.GetSteeringController().SetLookAheadDistance(5.0)
    driver.GetSteeringController().SetGains(0.8, 0.0, 0.0)
    driver.GetSpeedController().SetGains(0.6, 0.05, 0.0)
    driver.Initialize()

    dt = float(config["simulation"]["step_size_s"])
    record_every = max(1, int(round(float(config["simulation"]["record_step_s"]) / dt)))
    timeout_s = SETTLE_S + float(plan.meta["est_duration_s"]) * timeout_scale + 8.0

    xtracks: list[float] = []
    wp_idx = 0
    status = "timeout"
    success_time = None
    max_contact_n = 0.0
    min_clearance_m = float("inf")
    min_dist_goal = math.inf
    step = 0
    # Repo convention: steering_rate_limit 0.1 per 20 Hz step = 2.0 full-scale/s.
    steer_rate_per_s = 2.0
    prev_steer = 0.0

    while True:
        t = float(system.GetChTime())
        if t >= timeout_s:
            break

        ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
        pos = np.array([ref.GetPos().x, ref.GetPos().y])
        wp_idx = nearest_index(plan.waypoints, pos, wp_idx)
        v_cmd = 0.0 if t < SETTLE_S else float(plan.speeds[wp_idx])
        driver.SetDesiredSpeed(v_cmd)

        driver.Synchronize(t)
        inputs = driver.GetInputs()
        max_d = steer_rate_per_s * dt
        prev_steer = min(max(float(inputs.m_steering), prev_steer - max_d), prev_steer + max_d)
        inputs.m_steering = prev_steer
        terrain.Synchronize(t)
        hmmwv.Synchronize(t, inputs, terrain)

        contact = max((body.GetContactForce().Length() for _, body in scene.asset_bodies), default=0.0)
        max_contact_n = max(max_contact_n, contact)
        # Distance from the vehicle reference to the nearest obstacle EDGE.
        # Without this the zero-contact criterion cannot be audited from its own
        # output: a pass is indistinguishable from a run where collision was
        # never active, which is exactly how the original G0a result came to be
        # vacuously true. Recording clearance turns "no contact" into "no contact
        # AND the closest approach was X m", which is a claim with content.
        clearance = min(
            (float(np.hypot(pos[0] - a.x_m, pos[1] - a.y_m)) - a.footprint_radius_m
             for a, _ in scene.asset_bodies),
            default=float("inf"),
        )
        min_clearance_m = min(min_clearance_m, clearance)
        dist_goal = float(np.hypot(*(pos - approach)))
        min_dist_goal = min(min_dist_goal, dist_goal)
        roll, pitch = float(vehicle.GetRoll()), float(vehicle.GetPitch())

        if step % record_every == 0:
            xtracks.append(cross_track_m(plan.waypoints, pos, wp_idx))

        if abs(roll) > math.radians(60) or abs(pitch) > math.radians(60):
            status = "rollover"
            break
        if success_time is None and dist_goal < SUCCESS_RADIUS_M:
            success_time = t
        if success_time is not None and t - success_time > SUCCESS_HOLD_S:
            status = "success"
            break

        driver.Advance(dt)
        terrain.Advance(dt)
        hmmwv.Advance(dt)
        step += 1

    xt = np.asarray(xtracks) if xtracks else np.zeros(1)
    row.update(
        status=status,
        final_time_s=float(system.GetChTime()),
        success_time_s=success_time,
        cross_track_mean_m=float(xt.mean()),
        cross_track_max_m=float(xt.max()),
        max_asset_contact_n=float(max_contact_n),
        min_asset_clearance_m=float(min_clearance_m),
        min_dist_goal_m=float(min_dist_goal),
        wall_s=time.time() - wall0,
    )
    return row


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena", default="assets/traverse/arena_v1")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed0", type=int, default=20260901)
    parser.add_argument("--seeds", default=None, help="comma-separated explicit seed list (overrides --episodes/--seed0)")
    parser.add_argument("--procs", type=int, default=12)
    parser.add_argument("--timeout-scale", type=float, default=2.5)
    parser.add_argument("--out", default="artifacts/traverse/wp0a_gate")
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = [args.seed0 + i for i in range(args.episodes)]
    args.episodes = len(seeds)
    tasks = [(i, seed, args.arena, args.timeout_scale) for i, seed in enumerate(seeds)]
    rows: list[dict] = []
    wall0 = time.time()
    with get_context("spawn").Pool(args.procs) as pool:
        for row in pool.imap_unordered(run_one, tasks):
            rows.append(row)
            print(
                f"[{len(rows):3d}/{args.episodes}] ep {row['episode']:3d} "
                f"{row['status']:>18s}  t={row.get('final_time_s', 0.0) or 0.0:5.1f}s  "
                f"xtrack_max={row.get('cross_track_max_m', float('nan')):5.2f} m  "
                f"contact={row.get('max_asset_contact_n', float('nan')):7.1f} N  "
                f"wall={row['wall_s']:5.1f}s",
                flush=True,
            )
    rows.sort(key=lambda r: r["episode"])
    with (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    n = len(rows)
    n_success = sum(r["status"] == "success" for r in rows)
    by_status = {s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}}
    contact_rows = [r for r in rows if r.get("max_asset_contact_n", 0.0) > CONTACT_EPS_N]
    driven = [r for r in rows if "cross_track_max_m" in r]
    lo, hi = wilson_interval(n_success, n)
    summary = {
        "episodes": n,
        "success": n_success,
        "success_rate": n_success / n if n else 0.0,
        "success_rate_wilson95": [lo, hi],
        "by_status": by_status,
        "collision_episodes": [r["episode"] for r in contact_rows],
        "max_contact_overall_n": max((r.get("max_asset_contact_n", 0.0) for r in rows), default=0.0),
        "cross_track_mean_m": float(np.mean([r["cross_track_mean_m"] for r in driven])) if driven else None,
        "cross_track_max_m": float(np.max([r["cross_track_max_m"] for r in driven])) if driven else None,
        "layout_resample_episodes": sum(r.get("layout_attempt", 0) > 0 for r in rows),
        "plan_length_m_minmax": [
            float(np.min([r["plan_length_m"] for r in driven])),
            float(np.max([r["plan_length_m"] for r in driven])),
        ] if driven else None,
        "wall_total_s": time.time() - wall0,
        "gate_G0a_pass": bool(n_success / n >= 0.95 and not contact_rows) if n else False,
        "seed0": args.seed0,
        "timeout_scale": args.timeout_scale,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0 if summary["gate_G0a_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
