"""Sim-to-sim transfer test: run a ROM-trained goal-reaching policy in Chrono.

Runs ONE goal per process (fresh M113+arm scene; avoids repeated torch+Chrono
sim re-creation, which has stack-smashed in past eval loops). The policy is a
memoryless MLP over the 11-D goal-relative observation, so Chrono needs no
dynamics context window: read the real vehicle state -> build the same obs the
policy trained on -> scale to a driver command -> advance Chrono action_repeat
control steps -> repeat. Saves the trajectory to an .npz for aggregate plotting.
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

import pychrono as chrono
import pychrono.vehicle as veh
from rsl_rl.runners import OnPolicyRunner

from nedm.arm_data import build_scene, make_vis, SETTLE_TIME, STEP_SIZE
from nedm.rl.dynamics import resolve_dynamics_checkpoint_path
from nedm.rl.tracked_goal_env import TrackedGoalReachingEnv, merge_env_cfg
from nedm.tracked_vehicle_data import TERRAIN_SIZE_M, _advance, _sync, capture_row


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run a tracked goal-reaching policy in Chrono (one goal).")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--goal-x", type=float, required=True, help="Goal x in the vehicle start frame (m).")
    p.add_argument("--goal-y", type=float, required=True, help="Goal y in the vehicle start frame (m).")
    p.add_argument("--max-steps", type=int, default=400, help="Max policy steps (each = action_repeat*0.02 s sim).")
    p.add_argument("--output", type=Path, required=True, help="Output .npz path for the trajectory.")
    p.add_argument("--render", action="store_true", help="Open the Chrono Irrlicht viewer during the rollout.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def _rebind_vis(vis, system) -> None:
    """Re-bind the viewer so it picks up bodies added/removed after make_vis()."""
    for method_name, args in (("AttachSystem", (system,)), ("BindAll", ())):
        method = getattr(vis, method_name, None)
        if method is not None:
            try:
                method(*args)
            except Exception:
                pass


def add_goal_marker(system, vis, gx_w, gy_w, radius):
    """Add a fixed, non-colliding green sphere at the goal (radius = success tolerance).

    A fixed collision-disabled body is inert (it does not enter the dynamics), matching
    the arm Chrono env's marker pattern for this same M113 Irrlicht scene. AttachSystem/
    BindAll re-bind the viewer so it picks up the body added after make_vis().

    Returns the marker body so callers can remove it later (see ``remove_goal_marker``).
    """
    body = chrono.ChBody()
    body.SetFixed(True)
    body.EnableCollision(False)
    body.SetPos(chrono.ChVector3d(gx_w, gy_w, radius))
    shape = chrono.ChVisualShapeSphere(float(radius))
    shape.SetColor(chrono.ChColor(0.1, 0.75, 0.1))
    body.AddVisualShape(shape)
    system.Add(body)
    _rebind_vis(vis, system)
    return body


def remove_goal_marker(system, vis, body) -> None:
    """Remove a goal marker body added by ``add_goal_marker`` and re-bind the viewer."""
    if body is None:
        return
    try:
        system.Remove(body)
    except Exception:
        pass
    _rebind_vis(vis, system)


def main(argv=None) -> int:
    args = parse_args(argv)
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
    tol = rom.success_tolerance
    action_repeat = rom.action_repeat
    dt_s = rom.dt_s
    step_dt = rom.step_dt
    n_sub = int(round(dt_s / STEP_SIZE))

    # ---- Chrono scene ----
    m113, vehicle, terrain, _g = build_scene(terrain_size_m=TERRAIN_SIZE_M)
    system = m113.GetSystem()
    vis = make_vis(vehicle, f"Tracked goal reach  goal=({args.goal_x:.0f}, {args.goal_y:.0f}) m") if args.render else None
    if vis is not None:
        vehicle.EnableRealtime(True)
    render_every = n_sub                                 # draw one frame per 0.02 s control step
    tick = 0
    window_closed = False

    di = veh.DriverInputs()
    di.m_throttle, di.m_steering, di.m_braking = 0.0, 0.0, 1.0
    while system.GetChTime() < SETTLE_TIME - 1e-9:      # settle onto tracks, braked
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
    # goal defined in the vehicle's start frame -> world
    gx_w = x0 + math.cos(yaw0) * args.goal_x - math.sin(yaw0) * args.goal_y
    gy_w = y0 + math.sin(yaw0) * args.goal_x + math.cos(yaw0) * args.goal_y
    if vis is not None:
        add_goal_marker(system, vis, gx_w, gy_w, float(rom.success_tolerance))

    last_driver = action_center.astype(np.float32).copy()
    poses = [(0.0, 0.0)]                                  # trajectory in start frame (origin)
    driver_log = []
    vel_log = []
    reached_step = -1
    min_dist = math.inf
    wall = time.time()

    for step in range(args.max_steps):
        vx, vy, r, x, y, yaw = read_state()
        dx, dy = gx_w - x, gy_w - y
        dist = math.hypot(dx, dy)
        c, s = math.cos(yaw), math.sin(yaw)
        gx_b = c * dx + s * dy
        gy_b = -s * dx + c * dy
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
        last_driver = driver.astype(np.float32)

        _, _, _, x, y, yaw = read_state()
        # record pose in the start frame
        rx = math.cos(yaw0) * (x - x0) + math.sin(yaw0) * (y - y0)
        ry = -math.sin(yaw0) * (x - x0) + math.cos(yaw0) * (y - y0)
        poses.append((rx, ry))
        driver_log.append(driver.tolist())
        vel_log.append([vx, vy, r])
        min_dist = min(min_dist, dist)
        if dist < tol and reached_step < 0:
            reached_step = step + 1
            break

    elapsed = time.time() - wall
    poses = np.asarray(poses, dtype=np.float32)
    goal_local = np.asarray([args.goal_x, args.goal_y], dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, poses=poses, goal=goal_local, reached_step=np.int64(reached_step),
        min_dist=np.float32(min_dist), tol=np.float32(tol), step_dt=np.float32(step_dt),
        driver=np.asarray(driver_log, dtype=np.float32), vel=np.asarray(vel_log, dtype=np.float32),
    )
    status = f"REACHED in {reached_step * step_dt:.1f}s" if reached_step >= 0 else "TIMEOUT"
    print(f"goal=({args.goal_x:.1f},{args.goal_y:.1f}) dist0={math.hypot(*goal_local):.1f}m  "
          f"{status}  min_dist={min_dist:.3f}m  steps={len(poses) - 1}  wall={elapsed:.0f}s  "
          f"-> {args.output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
