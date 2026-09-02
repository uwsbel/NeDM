#!/usr/bin/env python
"""Traversal data collection: driver mixture -> episode-chunked store (plan §6).

Per episode: sample a layout (oracle-proven feasible), build the rendered
scene (256^2 RGB + depth at 20 Hz, collection light), drive the §6.2 mixture
family for a fixed duration, and stream frames + full state rows into the
§6.1 compressed episode store (schema v1, ``nedm.traverse.storage``).

Smoke tier (default): 10 episodes x 20 s = 4k frames.

Usage (from repo root, nedm env):
  PYTHONPATH=src python scripts/traverse_collect.py --tier smoke --procs 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SETTLE_S = 0.8
CTRL_DT_S = 0.05  # 20 Hz control/record/camera interval (plan §3.4)
STEER_RATE_PER_S = 2.0  # repo convention: 0.1 per 20 Hz step
CONTACT_EPS_N = 1.0
ROLL_PITCH_ABORT_RAD = math.radians(60.0)


def nearest_index(waypoints: np.ndarray, pos: np.ndarray, last: int, window: int = 60) -> int:
    lo = last
    hi = min(len(waypoints), last + window)
    d = np.hypot(waypoints[lo:hi, 0] - pos[0], waypoints[lo:hi, 1] - pos[1])
    return lo + int(np.argmin(d))


def run_one(task: dict) -> dict:
    idx = task["index"]
    seed = task["seed"]
    import pychrono as chrono
    import pychrono.vehicle as veh

    from nedm.hmmwv_data import WHEEL_SPECS, capture_row
    from nedm.traverse.drivers import MeanderController, assign_family, build_driver_route
    from nedm.traverse.layout import sample_episode
    from nedm.traverse.oracle import PlanCandidate
    from nedm.traverse.scene import RenderSpec, build_config, build_scene
    from nedm.traverse.storage import EpisodeWriter, verify_episode
    from nedm.traverse.terrain import TerrainMap

    wall0 = time.time()
    arena_dir = (REPO_ROOT / task["arena"]).resolve()
    tmap = TerrainMap.from_dir(arena_dir)
    family = assign_family(idx)
    episode_id = f"ep_{idx:04d}_{family}"
    row: dict = {"episode": idx, "seed": seed, "family": family, "episode_id": episode_id}

    try:
        layout, oracle_plan = sample_episode(tmap, episode_id, seed)
    except RuntimeError as exc:
        row.update(status="no_feasible_layout", error=str(exc), wall_s=time.time() - wall0)
        return row

    drv_rng = np.random.default_rng([seed, 77])
    contact_intended = family == "near_obstacle" and (idx // 10) % 2 == 0
    route = build_driver_route(
        family, tmap, layout, oracle_plan, drv_rng, contact_intended, duration_s=task["duration_s"]
    )
    family_actual = family
    if route is None:  # generation failed; oracle route always exists
        route = oracle_plan
        route.meta["family"] = "oracle_fallback"
        family_actual = "oracle_fallback"
    row["family_actual"] = family_actual
    meander = route if isinstance(route, MeanderController) else None
    plan: PlanCandidate | None = None if meander is not None else route

    start_z = float(tmap.height(*layout.start_xy)) + 0.75
    config = build_config(arena_dir, (*layout.start_xy, start_z), layout.start_yaw)
    render = RenderSpec(width=task["res"], height=task["res"], plan_markers=False)
    scene = build_scene(config, layout, tmap, arena_dir, plan=None, render=render)
    hmmwv, system, terrain = scene.hmmwv, scene.system, scene.terrain
    vehicle = hmmwv.GetVehicle()
    engine = vehicle.GetEngine()
    transmission = vehicle.GetTransmission()

    driver = None
    if plan is not None:
        pts = chrono.vector_ChVector3d()
        last_s = -10.0
        for (x, y), s in zip(plan.waypoints, plan.stations):
            if s - last_s < 2.0 and s != plan.stations[-1]:
                continue
            last_s = s
            pts.append(chrono.ChVector3d(float(x), float(y), float(tmap.height(x, y)) + 0.5))
        driver = veh.ChPathFollowerDriver(vehicle, chrono.ChBezierCurve(pts), "route", float(plan.speeds[0]))
        driver.GetSteeringController().SetLookAheadDistance(5.0)
        driver.GetSteeringController().SetGains(0.8, 0.0, 0.0)
        driver.GetSpeedController().SetGains(0.6, 0.05, 0.0)
        driver.Initialize()

    tire_radii = {
        name: float(vehicle.GetTire(axle, side).GetRadius()) for name, axle, side in WHEEL_SPECS
    }

    dt = float(config["simulation"]["step_size_s"])
    substeps = max(1, int(round(CTRL_DT_S / dt)))
    n_frames = int(round(task["duration_s"] / CTRL_DT_S))
    out_dir = (REPO_ROOT / task["out"]).resolve() / episode_id
    writer = EpisodeWriter(out_dir, task["res"], task["res"], chunk_frames=task["chunk_frames"])
    raw_rgb: list[np.ndarray] = []
    raw_depth: list[np.ndarray] = []

    manual = veh.DriverInputs()
    prev_steer = 0.0
    wp_idx = 0
    status = "complete"
    contact_events: list[list] = []
    max_contact_n = 0.0
    speeds_seen: list[float] = []

    frame = -int(round(SETTLE_S / CTRL_DT_S))  # negative frames = settle, unrecorded
    while frame < n_frames:
        ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
        pos = np.array([ref.GetPos().x, ref.GetPos().y])
        yaw = float(ref.GetRot().GetCardanAnglesZYX().z)
        speed = float(vehicle.GetSpeed())

        # --- 20 Hz decisions: desired speed (path families) / held inputs (meander) ---
        if meander is not None:
            if frame < 0:
                steer_cmd, throttle, braking = 0.0, 0.0, 1.0
            else:
                steer_cmd, throttle, braking = meander(CTRL_DT_S, pos[0], pos[1], yaw, speed)
            max_d = STEER_RATE_PER_S * CTRL_DT_S
            prev_steer = min(max(steer_cmd, prev_steer - max_d), prev_steer + max_d)
            manual.m_steering = prev_steer
            manual.m_throttle = throttle
            manual.m_braking = braking
            inputs = manual
        else:
            wp_idx = nearest_index(plan.waypoints, pos, wp_idx)
            driver.SetDesiredSpeed(0.0 if frame < 0 else float(plan.speeds[wp_idx]))

        # --- physics substeps (gate-validated loop: sync + rate limit per substep) ---
        frame_contact = 0.0
        for sub in range(substeps):
            ts = float(system.GetChTime())
            if driver is not None:
                driver.Synchronize(ts)
                inputs = driver.GetInputs()
                max_d = STEER_RATE_PER_S * dt
                prev_steer = min(max(float(inputs.m_steering), prev_steer - max_d), prev_steer + max_d)
                inputs.m_steering = prev_steer
            terrain.Synchronize(ts)
            hmmwv.Synchronize(ts, inputs, terrain)

            # Capture at substep 0: obs + state at ts, action starting the interval
            # (tire forces are valid here — computed inside hmmwv.Synchronize).
            if sub == 0 and frame >= 0:
                scene.manager.Update()
                rgb = scene.rgb_tap.take()
                depth = scene.depth_tap.take()
                state = capture_row(
                    hmmwv, terrain, family_actual, family, episode_id, "unassigned",
                    frame, ts, inputs, include_tires=True, tire_radii=tire_radii,
                )
                state["engine_motor_speed_radps"] = float(engine.GetMotorSpeed())
                state["engine_motorshaft_torque_nm"] = float(engine.GetOutputMotorshaftTorque())
                state["trans_driveshaft_torque_nm"] = float(transmission.GetOutputDriveshaftTorque())
                state["trans_motorshaft_speed_radps"] = float(transmission.GetOutputMotorshaftSpeed())
                writer.append(rgb, depth, state)
                raw_rgb.append(rgb)
                raw_depth.append(depth)
                speeds_seen.append(speed)

            if driver is not None:
                driver.Advance(dt)
            terrain.Advance(dt)
            hmmwv.Advance(dt)
            for a_idx, (_, body) in enumerate(scene.asset_bodies):
                force = float(body.GetContactForce().Length())
                if force > frame_contact:
                    frame_contact = force
                if force > CONTACT_EPS_N and frame >= 0:
                    if not contact_events or contact_events[-1][0] != frame or contact_events[-1][1] != a_idx:
                        contact_events.append([frame, a_idx, round(force, 1)])
                    elif force > contact_events[-1][2]:
                        contact_events[-1][2] = round(force, 1)
        max_contact_n = max(max_contact_n, frame_contact)

        roll, pitch = float(vehicle.GetRoll()), float(vehicle.GetPitch())
        if abs(roll) > ROLL_PITCH_ABORT_RAD or abs(pitch) > ROLL_PITCH_ABORT_RAD:
            status = "rollover"
            break
        frame += 1

    meta = writer.finalize(
        {
            "episode_id": episode_id,
            "tier": task["tier"],
            "seed": seed,
            "family": family,
            "family_actual": family_actual,
            "status": status,
            "t0_s": SETTLE_S,
            "frame_dt_s": CTRL_DT_S,
            "layout": layout.to_json(),
            "route": None if meander is not None else plan.to_json(),
            "meander": None
            if meander is None
            else {"target_speed_mps": meander.target_speed_mps, "keep_within_m": meander.keep_within_m},
            "oracle_approach_pose": oracle_plan.meta["approach_pose"],
            "contact": {"max_n": max_contact_n, "events": contact_events[:2000]},
            "collected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    verify_episode(out_dir, np.stack(raw_rgb), np.stack(raw_depth), np.random.default_rng(seed))

    disk = sum(meta["bytes"][k] for k in ("rgb_bin", "depth_bin", "states_npz"))
    row.update(
        status=status,
        frames=writer.frames,
        disk_mb=round(disk / 2**20, 2),
        raw_mb=round(meta["bytes"]["raw_frames"] / 2**20, 2),
        ratio=round(meta["bytes"]["raw_frames"] / max(disk, 1), 2),
        max_contact_n=round(max_contact_n, 1),
        n_contact_events=len(contact_events),
        mean_speed_mps=round(float(np.mean(speeds_seen)), 2) if speeds_seen else 0.0,
        verified=True,
        wall_s=round(time.time() - wall0, 1),
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena", default="assets/traverse/arena_v1")
    parser.add_argument("--tier", default="smoke", choices=["smoke", "pilot", "full"])
    parser.add_argument("--episodes", type=int, default=None, help="default: smoke 10 / pilot 200")
    parser.add_argument(
        "--indices", default=None,
        help="comma-separated episode indices to (re-)collect into an existing store; "
        "rows merge with episodes.jsonl and the manifest is rebuilt",
    )
    parser.add_argument("--seed0", type=int, default=20260910)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--chunk-frames", type=int, default=20)
    parser.add_argument("--procs", type=int, default=3)
    parser.add_argument("--out", default=None, help="default: artifacts/traverse/<tier>_v1")
    args = parser.parse_args()

    episodes = args.episodes or {"smoke": 10, "pilot": 200, "full": 1000}[args.tier]
    out = args.out or f"artifacts/traverse/{args.tier}_v1"
    out_dir = (REPO_ROOT / out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = [int(s) for s in args.indices.split(",")] if args.indices else list(range(episodes))
    tasks = [
        {
            "index": i,
            "seed": args.seed0 + i,
            "arena": args.arena,
            "out": out,
            "tier": args.tier,
            "duration_s": args.duration_s,
            "res": args.res,
            "chunk_frames": args.chunk_frames,
        }
        for i in indices
    ]

    rows: list[dict] = []
    wall0 = time.time()
    with get_context("spawn").Pool(min(args.procs, len(tasks))) as pool:
        for row in pool.imap_unordered(run_one, tasks):
            rows.append(row)
            print(
                f"[{len(rows):3d}/{len(tasks)}] {row['episode_id']:>24s} {row['status']:>10s}  "
                f"frames={row.get('frames', 0):3d}  disk={row.get('disk_mb', 0.0):6.1f} MB  "
                f"ratio={row.get('ratio', 0.0):5.1f}x  contact={row.get('max_contact_n', 0.0):7.1f} N  "
                f"wall={row['wall_s']:6.1f}s",
                flush=True,
            )
    # Merge with any existing store (--indices repair mode replaces its rows).
    by_index: dict[int, dict] = {}
    episodes_path = out_dir / "episodes.jsonl"
    if episodes_path.is_file():
        with episodes_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                old = json.loads(line)
                by_index[old["episode"]] = old
    for row in rows:
        by_index[row["episode"]] = row
    rows = [by_index[i] for i in sorted(by_index)]
    with episodes_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    arena_dir = (REPO_ROOT / args.arena).resolve()
    with (arena_dir / "arena_meta.json").open("r", encoding="utf-8") as handle:
        arena_meta = json.load(handle)
    bmp_hash = hashlib.sha256((arena_dir / arena_meta["bmp"]).read_bytes()).hexdigest()

    ok = [r for r in rows if r.get("frames")]
    manifest = {
        "schema_version": 1,
        "tier": args.tier,
        "seed0": args.seed0,
        "episodes": len(rows),
        "episodes_ok": len(ok),
        "duration_s": args.duration_s,
        "frame_dt_s": CTRL_DT_S,
        "arena": {"dir": args.arena, "bmp_sha256": bmp_hash, **arena_meta},
        "camera": {
            "width": args.res,
            "height": args.res,
            "hfov_deg": 47.0,
            "cam_height_m": 100.0,
            "image_up": "+Y world (north-up)",
            "depth_convention": "ray",
            "depth_ray_scale": 1.200,  # WP0b fit vs calibrated heightmap
            "max_depth_m": 250.0,
            "light_elevation_deg": 55.0,
        },
        "driver_mixture": {r["episode_id"]: r.get("family_actual", r["family"]) for r in rows},
        "processed_caches": "reference",  # §6.1: caches point at this store, never duplicate
        "total_disk_mb": round(sum(r.get("disk_mb", 0.0) for r in rows), 1),
        "total_raw_mb": round(sum(r.get("raw_mb", 0.0) for r in rows), 1),
        "wall_total_s": round(time.time() - wall0, 1),
        "procs": args.procs,
        "collected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    n_bad = sum(r.get("status") not in ("complete",) for r in rows)
    print(
        f"\ncollected {len(ok)}/{len(rows)} episodes ({n_bad} abnormal) -> {out_dir}\n"
        f"disk {manifest['total_disk_mb']:.0f} MB vs raw {manifest['total_raw_mb']:.0f} MB "
        f"({manifest['total_raw_mb'] / max(manifest['total_disk_mb'], 0.1):.1f}x), "
        f"wall {manifest['wall_total_s'] / 60:.1f} min with {args.procs} procs"
    )
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
