"""Consecutive-waypoint (multi-goal) sim-to-sim transfer test in Chrono.

Unlike ``eval_tracked_rl_goal_chrono.py`` (one goal, then reset to origin), this drives
the frozen goal-reaching policy through a *route* of waypoints in a SINGLE Chrono rollout
without ever recreating the sim. The policy is a memoryless MLP over the 11-D goal-relative
observation, so all we do to chain goals is: keep the one M113 scene alive, and when the
vehicle reaches the active waypoint, swap the target to the next one (recompute the same
body-frame obs against the new goal). Building the scene once side-steps the repeated-sim
stack-smash that has bitten past eval loops.

Default route is the square (0,0) -> (30,0) -> (30,30) -> (0,30) -> (0,0), given in the
vehicle's start frame. The first waypoint is the start pose; the rest are targets, hit in
order. By default every waypoint uses the strict training success tolerance (0.75 m): a
fast (~1.9 m/s) corner approach may overshoot and U-turn back before hitting the point —
that is expected and honest. Pass ``--capture-radius-m`` to fly through corners instead.

Saves the full trajectory + per-leg outcomes to an .npz for plotting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pychrono as chrono  # noqa: E402
import pychrono.vehicle as veh  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from nedm.arm_data import build_scene, make_vis, SETTLE_TIME, STEP_SIZE  # noqa: E402
from nedm.rl.dynamics import resolve_dynamics_checkpoint_path  # noqa: E402
from nedm.rl.tracked_goal_env import TrackedGoalReachingEnv, merge_env_cfg  # noqa: E402
from nedm.tracked_vehicle_data import TERRAIN_SIZE_M, _advance, _sync, capture_row  # noqa: E402

# The single-goal Chrono eval already owns the goal-marker helpers; reuse them verbatim.
from eval_tracked_rl_goal_chrono import add_goal_marker, remove_goal_marker  # noqa: E402

DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts/rl_runs/tracked_goal_v2_far_rollsel_rom_20260721/model_1499.pt"
DEFAULT_WAYPOINTS = "0,0 30,0 30,30 0,30 0,0"


def parse_waypoints(text: str) -> list[tuple[float, float]]:
    """Parse ``"0,0 30,0 30,30"`` (start-frame x,y pairs) into a list of tuples."""
    pts = []
    for token in text.replace(";", " ").split():
        x_str, y_str = token.split(",")
        pts.append((float(x_str), float(y_str)))
    if len(pts) < 2:
        raise ValueError("need at least a start and one target waypoint")
    return pts


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Drive a tracked goal-reach policy through consecutive waypoints in Chrono.")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--waypoints", type=str, default=DEFAULT_WAYPOINTS,
                   help='Space-separated start-frame x,y pairs; first is the start pose. Default is the 30 m square.')
    p.add_argument("--capture-radius-m", type=float, default=None,
                   help="Distance to count an intermediate waypoint as reached (default: the ROM success tolerance).")
    p.add_argument("--final-tol-m", type=float, default=None,
                   help="Tolerance for the LAST waypoint (default: the ROM success tolerance).")
    p.add_argument("--max-steps-per-leg", type=int, default=500,
                   help="Max policy steps per leg (each = action_repeat*0.02 s sim); room for a corner overshoot + U-turn.")
    p.add_argument("--stop-on-leg-failure", dest="stop_on_leg_failure", action="store_true", default=True,
                   help="Stop the run when a leg times out (default). The loop is not 'finished' if a leg fails.")
    p.add_argument("--continue-on-leg-failure", dest="stop_on_leg_failure", action="store_false",
                   help="Keep driving to later waypoints even after a leg times out.")
    p.add_argument("--output", type=Path, required=True, help="Output .npz path for the trajectory + per-leg outcomes.")
    p.add_argument("--render", action="store_true", help="Open the Chrono Irrlicht viewer during the rollout.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    waypoints = parse_waypoints(args.waypoints)
    targets = waypoints[1:]                      # the first waypoint is the start pose

    checkpoint = resolve_dynamics_checkpoint_path(args.checkpoint)
    run_dir = checkpoint.parent
    env_cfg = json.loads((run_dir / "env_cfg.json").read_text())
    train_cfg = json.loads((run_dir / "train_cfg.json").read_text())
    env_cfg.update({"num_envs": 1, "auto_reset": False, "device": args.device})

    # ROM env: source of the exact obs/action constants + a VecEnv for the runner.
    rom = TrackedGoalReachingEnv(merge_env_cfg(env_cfg), device=args.device)
    runner = OnPolicyRunner(rom, train_cfg, log_dir=None, device=args.device)
    loaded = torch.load(checkpoint, map_location=torch.device(args.device), weights_only=False)
    runner.alg.actor_critic.load_state_dict(loaded["model_state_dict"])
    if getattr(runner.alg, "rnd", None):
        runner.alg.rnd.load_state_dict(loaded["rnd_state_dict"])
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded["critic_obs_norm_state_dict"])
    policy = runner.get_inference_policy(device=args.device)

    dev = torch.device(args.device)
    state_std = rom.state_std.detach().clone()            # [vx, vy, r]
    action_mean = rom.action_mean.detach().clone()
    action_std = rom.action_std.detach().clone()
    action_center = rom.action_center.detach().cpu().numpy()
    cart = rom.cart_scale
    default_tol = rom.success_tolerance
    capture_radius = float(args.capture_radius_m) if args.capture_radius_m is not None else default_tol
    final_tol = float(args.final_tol_m) if args.final_tol_m is not None else default_tol
    action_repeat = rom.action_repeat
    step_dt = rom.step_dt
    n_sub = int(round(rom.dt_s / STEP_SIZE))

    # ---- Chrono scene (built once, reused for every waypoint) ----
    m113, vehicle, terrain, _g = build_scene(terrain_size_m=TERRAIN_SIZE_M)
    system = m113.GetSystem()
    vis = make_vis(vehicle, "Tracked waypoint route") if args.render else None
    if vis is not None:
        vehicle.EnableRealtime(True)
    render_every = n_sub
    tick = 0
    window_closed = False

    di = veh.DriverInputs()
    di.m_throttle, di.m_steering, di.m_braking = 0.0, 0.0, 1.0
    while system.GetChTime() < SETTLE_TIME - 1e-9:        # settle onto tracks, braked
        do_render = vis is not None and tick % render_every == 0
        if not _sync(m113, terrain, di, system.GetChTime(), vis, do_render):
            window_closed = True
            break
        _advance(m113, terrain, vis)
        tick += 1

    def read_state():
        row = capture_row(vehicle, "s", "f", "e", "eval", 0, 0.0, di)
        return (row["vel_body_x_mps"], row["vel_body_y_mps"], row["yaw_rate_radps"],
                row["pos_x_m"], row["pos_y_m"], row["yaw_rad"])

    vx, vy, r, x0, y0, yaw0 = read_state()
    c0, s0 = math.cos(yaw0), math.sin(yaw0)
    # transform every start-frame waypoint into world coords, once
    targets_w = [(x0 + c0 * wx - s0 * wy, y0 + s0 * wx + c0 * wy) for (wx, wy) in targets]

    last_driver = action_center.astype(np.float32).copy()
    poses = [(0.0, 0.0)]                                  # trajectory in the start frame (origin = start)
    active_wp = [0]                                       # which target index each pose was driving toward
    driver_log = []
    vel_log = []
    leg_reached, leg_steps, leg_time_s, leg_min_dist = [], [], [], []
    global_step = 0
    wall = time.time()

    for j, (gx_w, gy_w) in enumerate(targets_w):
        is_final = j == len(targets_w) - 1
        tol_j = final_tol if is_final else capture_radius
        marker = add_goal_marker(system, vis, gx_w, gy_w, float(tol_j)) if vis is not None else None

        leg_min = math.inf
        reached = False
        steps_this_leg = 0
        for _ in range(args.max_steps_per_leg):
            vx, vy, r, x, y, yaw = read_state()
            dx, dy = gx_w - x, gy_w - y
            dist = math.hypot(dx, dy)
            cy, sy = math.cos(yaw), math.sin(yaw)
            gx_b = cy * dx + sy * dy
            gy_b = -sy * dx + cy * dy
            he = math.atan2(gy_b, gx_b)
            vel = torch.tensor([vx, vy, r], dtype=torch.float32, device=dev)
            last_norm = (torch.tensor(last_driver, device=dev) - action_mean) / action_std
            obs = torch.cat([
                vel / state_std,
                torch.tensor([gx_b / cart, gy_b / cart, dist / cart, math.sin(he), math.cos(he)],
                             dtype=torch.float32, device=dev),
                last_norm,
            ]).unsqueeze(0)
            with torch.no_grad():
                raw = policy(obs)
            driver = rom._scale_policy_actions(raw)[0].detach().cpu().numpy()  # [steer, throttle, brake]
            di.m_steering, di.m_throttle, di.m_braking = float(driver[0]), float(driver[1]), float(driver[2])

            for _ in range(action_repeat * n_sub):
                do_render = vis is not None and tick % render_every == 0
                if not _sync(m113, terrain, di, system.GetChTime(), vis, do_render):
                    window_closed = True
                    break
                _advance(m113, terrain, vis)
                tick += 1
            if window_closed:
                break
            last_driver = driver.astype(np.float32)        # keep momentum continuous across legs

            _, _, _, x, y, yaw = read_state()
            rx = c0 * (x - x0) + s0 * (y - y0)             # record pose in the start frame
            ry = -s0 * (x - x0) + c0 * (y - y0)
            poses.append((rx, ry))
            active_wp.append(j)
            driver_log.append(driver.tolist())
            vel_log.append([vx, vy, r])
            global_step += 1
            steps_this_leg += 1
            leg_min = min(leg_min, dist)
            if dist < tol_j:
                reached = True
                break

        if marker is not None:
            remove_goal_marker(system, vis, marker)

        leg_reached.append(reached)
        leg_steps.append(steps_this_leg)
        leg_time_s.append(steps_this_leg * step_dt)
        leg_min_dist.append(leg_min)
        status = f"reached in {steps_this_leg * step_dt:5.1f}s" if reached else f"TIMEOUT ({args.max_steps_per_leg} steps)"
        print(f"  leg {j+1}/{len(targets_w)} -> ({targets[j][0]:.0f},{targets[j][1]:.0f})m  "
              f"tol {tol_j:.2f}m  {status}  min_dist={leg_min:.3f}m")
        if window_closed:
            break
        if not reached and args.stop_on_leg_failure:
            break

    elapsed = time.time() - wall
    finished = bool(len(leg_reached) == len(targets_w) and all(leg_reached))
    total_time_s = float(sum(leg_time_s))

    poses = np.asarray(poses, dtype=np.float32)
    waypoints_arr = np.asarray(waypoints, dtype=np.float32)     # includes the start (0,0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        poses=poses, active_wp=np.asarray(active_wp, dtype=np.int64),
        waypoints=waypoints_arr,
        leg_reached=np.asarray(leg_reached, dtype=bool),
        leg_steps=np.asarray(leg_steps, dtype=np.int64),
        leg_time_s=np.asarray(leg_time_s, dtype=np.float32),
        leg_min_dist=np.asarray(leg_min_dist, dtype=np.float32),
        tol=np.float32(final_tol), capture_radius=np.float32(capture_radius),
        step_dt=np.float32(step_dt), finished=np.bool_(finished),
        driver=np.asarray(driver_log, dtype=np.float32), vel=np.asarray(vel_log, dtype=np.float32),
    )

    verdict = f"FINISHED loop in {total_time_s:.1f}s" if finished else (
        f"FAILED at leg {len(leg_reached)}/{len(targets_w)}" if not window_closed else "ABORTED (window closed)")
    print(f"{verdict}  ({sum(leg_reached)}/{len(targets_w)} legs)  "
          f"steps={len(poses)-1}  wall={elapsed:.0f}s  -> {args.output.name}")
    return 0 if finished else 1


if __name__ == "__main__":
    raise SystemExit(main())
