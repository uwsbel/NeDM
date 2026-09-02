#!/usr/bin/env python
"""Profile the traversal collection loop: where do the 35 ms/substep go?

Rebuilds the exact pilot scene for one episode and drives the same
substep loop as traverse_collect, accumulating Chrono's per-step timers
(collision broad/narrow, setup, solve, update) plus wall-clock for the
Python-side segments (sync, advance, contact poll, render+taps).

One variant per invocation (fresh process; OptiX contexts don't like
rebuilds):
  PYTHONPATH=src python scripts/traverse_profile.py --chassis HULLS --render --poll
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

CTRL_DT_S = 0.05
STEER_RATE_PER_S = 2.0
SETTLE_S = 0.8

TIMER_NAMES = (
    "TimerCollisionBroad",
    "TimerCollisionNarrow",
    "TimerSetup",
    "TimerLSsetup",
    "TimerLSsolve",
    "TimerJacobian",
    "TimerUpdate",
    "TimerAdvance",
    "TimerStep",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arena", default="assets/traverse/arena_v1")
    parser.add_argument("--seed", type=int, default=20261000)  # pilot episode 0
    parser.add_argument("--chassis", choices=["HULLS", "NONE"], default="HULLS")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--poll", action="store_true", help="per-substep asset contact polling")
    parser.add_argument("--no-assets", action="store_true")
    parser.add_argument("--sim-s", type=float, default=4.0)
    parser.add_argument("--res", type=int, default=256)
    args = parser.parse_args()

    import pychrono as chrono
    import pychrono.vehicle as veh

    from nedm.traverse.layout import sample_episode
    from nedm.traverse.scene import RenderSpec, build_config, build_scene
    from nedm.traverse.terrain import TerrainMap

    arena_dir = (REPO_ROOT / args.arena).resolve()
    tmap = TerrainMap.from_dir(arena_dir)
    layout, plan = sample_episode(tmap, "profile", args.seed)
    if args.no_assets:
        layout.assets.clear()

    start_z = float(tmap.height(*layout.start_xy)) + 0.75
    config = build_config(arena_dir, (*layout.start_xy, start_z), layout.start_yaw)
    config["vehicle"]["chassis_collision"] = args.chassis
    render = RenderSpec(width=args.res, height=args.res, plan_markers=False) if args.render else None
    scene = build_scene(config, layout, tmap, arena_dir, plan=None, render=render)
    hmmwv, system, terrain = scene.hmmwv, scene.system, scene.terrain
    vehicle = hmmwv.GetVehicle()

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

    dt = float(config["simulation"]["step_size_s"])
    substeps = max(1, int(round(CTRL_DT_S / dt)))
    n_frames = int(round(args.sim_s / CTRL_DT_S))

    timers = {name: 0.0 for name in TIMER_NAMES}
    walls = {"sync": 0.0, "advance": 0.0, "poll": 0.0, "render": 0.0}
    n_sub_measured = 0
    prev_steer = 0.0
    wp_idx = 0

    frame = -int(round(SETTLE_S / CTRL_DT_S))
    wall_loop0 = None
    while frame < n_frames:
        ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
        pos = np.array([ref.GetPos().x, ref.GetPos().y])
        lo = wp_idx
        hi = min(len(plan.waypoints), lo + 60)
        d = np.hypot(plan.waypoints[lo:hi, 0] - pos[0], plan.waypoints[lo:hi, 1] - pos[1])
        wp_idx = lo + int(np.argmin(d))
        driver.SetDesiredSpeed(0.0 if frame < 0 else float(plan.speeds[wp_idx]))

        if frame == 0:
            wall_loop0 = time.perf_counter()
        for sub in range(substeps):
            ts = float(system.GetChTime())
            t0 = time.perf_counter()
            driver.Synchronize(ts)
            inputs = driver.GetInputs()
            if frame < 0:
                prev_steer = 0.0
            else:
                max_d = STEER_RATE_PER_S * dt
                prev_steer = min(max(float(inputs.m_steering), prev_steer - max_d), prev_steer + max_d)
            inputs.m_steering = prev_steer
            terrain.Synchronize(ts)
            hmmwv.Synchronize(ts, inputs, terrain)
            t1 = time.perf_counter()

            if sub == 0 and frame >= 0 and scene.manager is not None:
                scene.manager.Update()
                scene.rgb_tap.take()
                scene.depth_tap.take()
            t2 = time.perf_counter()

            driver.Advance(dt)
            terrain.Advance(dt)
            hmmwv.Advance(dt)
            t3 = time.perf_counter()

            if args.poll:
                for _, body in scene.asset_bodies:
                    float(body.GetContactForce().Length())
            t4 = time.perf_counter()

            if frame >= 0:
                walls["sync"] += t1 - t0
                walls["render"] += t2 - t1
                walls["advance"] += t3 - t2
                walls["poll"] += t4 - t3
                for name in TIMER_NAMES:
                    timers[name] += getattr(system, "Get" + name)()
                n_sub_measured += 1
        frame += 1

    wall_loop = time.perf_counter() - wall_loop0
    result = {
        "variant": {
            "chassis": args.chassis,
            "render": bool(args.render),
            "poll": bool(args.poll),
            "assets": len(scene.asset_bodies),
        },
        "sim_s": args.sim_s,
        "substeps": n_sub_measured,
        "wall_s": round(wall_loop, 2),
        "rtf": round(args.sim_s / wall_loop, 4),
        "ms_per_substep": round(1e3 * wall_loop / n_sub_measured, 3),
        "chrono_ms_per_substep": {
            name: round(1e3 * total / n_sub_measured, 3) for name, total in timers.items()
        },
        "python_ms_per_substep": {
            name: round(1e3 * total / n_sub_measured, 3) for name, total in walls.items()
        },
        "speed_end_mps": round(float(vehicle.GetSpeed()), 2),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
