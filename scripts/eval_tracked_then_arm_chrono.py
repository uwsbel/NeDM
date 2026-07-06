"""Combined tracked-drive + arm-reach Chrono demo (one shared scene, one window).

This fuses the two existing single-policy Chrono evals into a single continuous
simulation on ONE M113+arm rig:

    scripts/eval_tracked_rl_goal_chrono.py   (drive the base to a 2-D goal)
    scripts/eval_arm_rl_chrono_reaching.py   (reach the gripper to 3-D goals)

Both already build the very same scene via ``nedm.arm_data.build_scene`` (a
tracked M113 base with the LRV arm welded to its front deck), so instead of
running them in two processes with two windows, this script builds the rig once
and sequences three phases in it:

  Phase 1 - DRIVE.  The frozen-ROM tracked goal policy drives the base to a 2-D
            goal on the terrain. The arm is swapped to the compliant PD actuator
            up front and rides at home (qcmd=0) under PD for the whole drive --
            the regime it was trained on. (Holding it rigid during the drive and
            swapping to PD afterward leaves the arm at a shifted settled
            equilibrium that measurably degrades the Phase-3 reach.)

  Phase 2 - BRAKE.  On reaching the goal we do NOT reset the env; we hold full
            brake until the base is at rest (early-stop once speed < a threshold,
            capped at ``--brake-seconds``) and then PIN the chassis. The base
            arrives still moving (~2 m/s: the goal policy has no stop objective),
            so the brake bleeds that off; but a braked-yet-unpinned M113 keeps a
            slow residual yaw creep (arm-mass asymmetry + track slip), so we pin
            the instant it is translationally at rest rather than watching it
            creep. Right before braking we swap the arm to the compliant PD
            actuator, so this window ALSO settles the arm at home. This replaces
            the arm env's own 6 s pre-roll settle (``pre_roll_time_s``): the arm
            phase below runs with ``pre_roll_time_s = 0`` and reuses this
            already-settled rig, so the two settle windows never both run.

  Phase 3 - REACH.  We pin the chassis (matching how the arm data was collected /
            the arm eval runs — base fixed, arm-only dynamics) and run the arm
            reaching policy for ``--arm-num-goals`` (default 10) consecutive
            random goals in the SAME scene, then exit.

Because the Chrono scene is created exactly once and the arm phase reuses it via
``reset_goal_idx`` (never rebuilding a sim), this avoids the repeated
sim-recreation that has stack-smashed past full-loop evals.

Run in the NeDM conda env, e.g.:

    conda run -n nedm python scripts/eval_tracked_then_arm_chrono.py --render \
        --tracked-goal-x 6 --tracked-goal-y 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import pychrono as chrono  # noqa: F401  # load Chrono before torch/libstdc++ users
import pychrono.vehicle as veh
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = Path(__file__).resolve().parent
for _p in (str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nedm.arm_data import (
    ArmPdActuator,
    CONTROL_DT,
    SETTLE_TIME,
    STEP_SIZE,
    build_scene,
    gripper_center,
    make_vis,
    setup_arm_collision,
    _substep,
)
from nedm.rl.arm_reaching_chrono_env import ArmReachingChronoEnv, ChronoArmSim
from nedm.rl.dynamics import resolve_dynamics_checkpoint_path
from nedm.rl.tracked_goal_env import TrackedGoalReachingEnv, merge_env_cfg
from nedm.tracked_vehicle_data import TERRAIN_SIZE_M, capture_row

# Reuse the standalone evals' helpers rather than re-deriving them.
import eval_arm_rl_chrono_reaching as arm_eval
from eval_tracked_rl_goal_chrono import add_goal_marker

DEFAULT_TRACKED_CHECKPOINT = REPO_ROOT / "artifacts/rl_runs/tracked_goal_v2_far/model_1500.pt"
DEFAULT_ARM_RUN_DIR = (
    REPO_ROOT
    / "artifacts/rl_runs_arm_goal_reach"
    / "arm_reach_adaptivekl005_lr1e4_tol005_ep150_bonus150_sigma015_luffy_repro_20260702"
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Phase 1 (drive)
    p.add_argument("--tracked-checkpoint", type=Path, default=DEFAULT_TRACKED_CHECKPOINT,
                   help="Tracked goal-reaching policy checkpoint (.pt).")
    p.add_argument("--tracked-goal-x", type=float, default=25.0, help="Base goal x in the start frame (m).")
    p.add_argument("--tracked-goal-y", type=float, default=0.0, help="Base goal y in the start frame (m).")
    p.add_argument("--tracked-max-steps", type=int, default=600,
                   help="Max drive policy steps (each = action_repeat*0.02 s); matches v2_far's 600-step horizon.")
    # Phase 2 (brake to rest, then pin)
    p.add_argument("--brake-seconds", type=float, default=4.0,
                   help="MAX seconds of full brake after reaching the base goal; braking stops early once the "
                        "base is at rest. Replaces the arm env's 6 s pre-roll.")
    p.add_argument("--min-brake-seconds", type=float, default=0.5,
                   help="Always brake at least this long before the early-stop check (lets the arm PD settle).")
    p.add_argument("--brake-stop-speed-mps", type=float, default=0.15,
                   help="Pin the chassis once base speed drops below this (m/s). A braked-but-unpinned M113 keeps "
                        "a slow residual yaw creep, so pinning as soon as it is at rest ends the visible spin.")
    # Phase 3 (reach)
    p.add_argument("--arm-run-dir", type=Path, default=DEFAULT_ARM_RUN_DIR,
                   help="Arm reaching RSL-RL run directory (holds env/train cfg + model_*.pt).")
    p.add_argument("--arm-policy-checkpoint", type=Path, default=None,
                   help="Defaults to the latest model_*.pt in --arm-run-dir.")
    p.add_argument("--arm-num-goals", type=int, default=10)
    p.add_argument("--arm-max-steps", type=int, default=None, help="Per-goal step cap (default: env max).")
    p.add_argument("--arm-goal-duration-s", type=float, default=None,
                   help="Hold each arm goal segment for this long (success does not end it; failures still do).")
    # Shared
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/combined_tracked_arm_demo")
    p.add_argument("--render", action="store_true", help="Open the Chrono Irrlicht viewer for the whole demo.")
    p.add_argument("--no-plots", action="store_true", help="Skip per-goal arm reach PNGs.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------
def load_tracked_policy(checkpoint: Path, device: str):
    """Build the frozen-ROM tracked goal env (num_envs=1) + load its policy.

    Returns (rom_env, policy). Mirrors scripts/eval_tracked_rl_goal_chrono.py.
    """
    checkpoint = resolve_dynamics_checkpoint_path(checkpoint)
    run_dir = checkpoint.parent
    env_cfg = json.loads((run_dir / "env_cfg.json").read_text())
    train_cfg = json.loads((run_dir / "train_cfg.json").read_text())
    env_cfg.update({"num_envs": 1, "auto_reset": False, "device": device})

    rom = TrackedGoalReachingEnv(merge_env_cfg(env_cfg), device=device)
    runner = OnPolicyRunner(rom, train_cfg, log_dir=None, device=device)
    loaded = torch.load(checkpoint, map_location=torch.device(device), weights_only=False)
    runner.alg.actor_critic.load_state_dict(loaded["model_state_dict"])
    if getattr(runner.alg, "rnd", None):
        runner.alg.rnd.load_state_dict(loaded["rnd_state_dict"])
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded["critic_obs_norm_state_dict"])
    return rom, runner.get_inference_policy(device=device)


def load_arm_env_and_policy(run_dir: Path, policy_checkpoint: Path | None, render: bool, device: str):
    """Build the Chrono arm env (deferred, no scene, no pre-roll) + load its policy.

    ``defer_reset=True`` means __init__ does NOT build a Chrono scene; we inject
    the already-driven shared rig into ``env.sims[0]`` later. ``pre_roll_time_s=0``
    documents that the Phase-2 brake replaces the arm env's own settle.
    """
    run_dir = run_dir.resolve()
    env_cfg = json.loads((run_dir / "env_cfg.json").read_text())
    train_cfg = json.loads((run_dir / "train_cfg.json").read_text())
    env_cfg.update({
        "num_envs": 1,
        "device": device,
        "auto_reset": False,
        "warm_start_context": True,
        "defer_reset": True,
        "render": bool(render),
        "pre_roll_time_s": 0.0,
        "save_render_frames": False,
    })
    env = ArmReachingChronoEnv(env_cfg, device=device)
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=device)
    checkpoint = policy_checkpoint.resolve() if policy_checkpoint else arm_eval.latest_policy_checkpoint(run_dir)
    arm_eval.load_runner_checkpoint(runner, checkpoint, device=device)
    return env, runner.get_inference_policy(device=device), checkpoint


# ---------------------------------------------------------------------------
# Phase 1: drive the base to a 2-D goal (arm held compliant under PD)
# ---------------------------------------------------------------------------
def drive_to_goal(rom, policy, m113, vehicle, terrain, actuator, vis, goal_x, goal_y, max_steps, device):
    """Drive the M113 base to (goal_x, goal_y) with the tracked policy.

    Reuses the exact obs/action contract of scripts/eval_tracked_rl_goal_chrono.py.
    Steps with ``arm_data._substep`` so the arm is held COMPLIANT under PD
    (qcmd=home) for the whole drive -- the regime the arm dynamics model/policy
    were trained on. This matters for the Phase-3 reach: holding the arm rigid
    during the drive and only swapping to PD afterward bakes in a loaded
    configuration, leaving the arm at a shifted settled equilibrium (~0.1 m EE
    offset) that measurably degrades reaching on marginal goals. Returns a summary.
    """
    dev = torch.device(device)
    state_std = rom.state_std.detach().clone()
    action_mean = rom.action_mean.detach().clone()
    action_std = rom.action_std.detach().clone()
    action_center = rom.action_center.detach().cpu().numpy()
    cart = rom.cart_scale
    tol = rom.success_tolerance
    action_repeat = rom.action_repeat
    n_sub = int(round(rom.dt_s / STEP_SIZE))

    di = veh.DriverInputs()
    di.m_throttle, di.m_steering, di.m_braking = 0.0, 0.0, 1.0
    system = m113.GetSystem()
    for _ in range(int(round(SETTLE_TIME / CONTROL_DT))):  # settle onto tracks, braked (arm under PD)
        _substep(m113, terrain, actuator, di, n_sub, vis=vis)

    def read_state():
        row = capture_row(vehicle, "s", "f", "e", "eval", 0, 0.0, di)
        return (row["vel_body_x_mps"], row["vel_body_y_mps"], row["yaw_rate_radps"],
                row["pos_x_m"], row["pos_y_m"], row["yaw_rad"])

    vx, vy, r, x0, y0, yaw0 = read_state()
    gx_w = x0 + math.cos(yaw0) * goal_x - math.sin(yaw0) * goal_y
    gy_w = y0 + math.sin(yaw0) * goal_x + math.cos(yaw0) * goal_y
    if vis is not None:
        add_goal_marker(system, vis, gx_w, gy_w, float(tol))

    last_driver = action_center.astype(np.float32).copy()
    reached_step = -1
    min_dist = math.inf

    for step in range(max_steps):
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

        for _ in range(action_repeat):
            _substep(m113, terrain, actuator, di, n_sub, vis=vis)
        last_driver = driver.astype(np.float32)

        vx, vy, r, x, y, yaw = read_state()
        min_dist = min(min_dist, dist)
        if dist < tol and reached_step < 0:
            reached_step = step + 1
            break

    status = "REACHED" if reached_step >= 0 else "TIMEOUT"
    return {
        "status": status,
        "reached_step": reached_step,
        "min_dist_m": float(min_dist),
        "tol_m": float(tol),
        "goal_local": [float(goal_x), float(goal_y)],
        "goal_world": [float(gx_w), float(gy_w)],
        "window_closed": False,
    }


# ---------------------------------------------------------------------------
# Phase 3: build the injectable arm sim from the shared rig
# ---------------------------------------------------------------------------
def make_arm_sim(env, m113, vehicle, terrain, gripper, actuator, collision_links, vis):
    """Wrap the shared, already-driven rig in a ChronoArmSim + set up markers.

    Mirrors ArmReachingChronoEnv._create_sim's marker handling so the arm env's
    goal/EE markers show up in the shared Irrlicht window.
    """
    driver_inputs = veh.DriverInputs()
    driver_inputs.m_throttle, driver_inputs.m_steering, driver_inputs.m_braking = 0.0, 0.0, 1.0
    ee_marker = goal_marker = None
    if env.visual_markers_enabled:
        system = m113.GetSystem()
        ee_marker = env._create_marker_body(system, radius_m=env.marker_radius_m,
                                             color=chrono.ChColor(1.0, 0.55, 0.05))
        goal_marker = env._create_marker_body(system, radius_m=env.marker_radius_m,
                                               color=chrono.ChColor(1.0, 0.05, 0.05))
        ee_world = gripper_center(gripper)
        ee_marker.SetPos(ee_world)
        goal_marker.SetPos(ee_world)
        env._bind_marker_visuals(vis, system)
    return ChronoArmSim(
        m113=m113, vehicle=vehicle, terrain=terrain, gripper=gripper, actuator=actuator,
        collision_links=collision_links, driver_inputs=driver_inputs, vis=vis,
        frame_recorder=None, ee_marker=ee_marker, goal_marker=goal_marker,
    )


def run_arm_phase(env, policy, sim, num_goals, arm_max_steps, goal_duration_s, output_dir, make_plots):
    """Run the arm reaching policy for consecutive goals in the injected rig."""
    env.sims[0] = sim
    env._initialize_history(0, sim)  # warm-start the context window from the settled rig
    env.actions[0] = env.action_hist[0, -1, :]
    env.last_actions[0] = env.actions[0].clone()
    env.policy_actions[0] = 0.0
    env.last_policy_actions[0] = 0.0

    goal_duration_steps = None
    if goal_duration_s is not None:
        step_dt_s = float(env.dt_s) * int(env.action_repeat)
        goal_duration_steps = max(1, int(round(float(goal_duration_s) / step_dt_s)))
    max_steps = int(arm_max_steps) if arm_max_steps is not None else int(env.max_episode_length)
    if goal_duration_steps is not None:
        max_steps = max(max_steps if arm_max_steps is not None else 0, goal_duration_steps)
    success_tolerance_m = float(env.cfg["reward"]["success_tolerance_m"])
    dt_s = env.dt_s * env.action_repeat

    summary = []
    for goal_index in range(int(num_goals)):
        record = arm_eval.rollout_one_goal(
            env, policy, max_steps=max_steps, ignore_dones=False,
            reset_scene=False, fixed_duration_steps=goal_duration_steps,
        )
        metrics = arm_eval.compute_metrics(record, success_tolerance_m=success_tolerance_m)
        metrics["rollout_index"] = goal_index
        summary.append(metrics)
        np.savez_compressed(
            output_dir / f"arm_reach_{goal_index:02d}.npz",
            ee_base=record["ee_base"], q=record["q"], qcmd=record["qcmd"],
            goal_base=record["goal_base"], error_m=record["error_m"], action=record["action"],
            reward=record["reward"], done=record["done"], clearance_m=record["clearance_m"],
            unsafe_action=record["unsafe_action"],
        )
        if make_plots:
            arm_eval.plot_record(record, output_dir / f"arm_reach_{goal_index:02d}.png",
                                 success_tolerance_m=success_tolerance_m, dt_s=dt_s)
        print(f"  arm goal {goal_index}: {json.dumps(metrics)}")
        if not arm_eval.can_continue_consecutive(metrics["done_reason"]):
            print(f"  stopping arm phase early after goal {goal_index} "
                  f"(unsafe done_reason={metrics['done_reason']}); cannot continue consecutively.")
            break
    return summary, success_tolerance_m


def main(argv=None) -> int:
    args = parse_args(argv)
    device = arm_eval.resolve_device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- policies ----
    print("[load] tracked goal policy + frozen ROM ...")
    rom, tracked_policy = load_tracked_policy(args.tracked_checkpoint, device)
    print("[load] arm reaching policy + Chrono arm env (deferred, no scene) ...")
    arm_env, arm_policy, arm_checkpoint = load_arm_env_and_policy(
        args.arm_run_dir, args.arm_policy_checkpoint, args.render, device)

    # ---- one shared M113 + arm rig ----
    print(f"[scene] building shared M113+arm rig (terrain {TERRAIN_SIZE_M:.0f} m) ...")
    m113, vehicle, terrain, gripper = build_scene(terrain_size_m=TERRAIN_SIZE_M)
    vis = make_vis(vehicle, "Tracked drive -> arm reach (combined demo)") if args.render else None
    if vis is not None:
        vehicle.EnableRealtime(True)
    # Swap to the compliant PD actuator + add arm collision geometry BEFORE driving, so
    # the arm rides compliant (PD, qcmd=home) throughout the drive -- the regime it was
    # trained on. Holding it rigid during the drive and swapping afterward shifts the
    # arm's settled equilibrium and degrades the Phase-3 reach (see drive_to_goal).
    print("[scene] swapping arm to PD actuator + adding arm collision geometry ...")
    actuator = ArmPdActuator(gripper)          # remove hard angle motors -> torque + PD, qcmd = home
    collision_links = setup_arm_collision(gripper)
    wall = time.time()

    # ---- Phase 1: drive (arm compliant under PD) ----
    print(f"[phase 1] driving base to goal ({args.tracked_goal_x:.1f}, {args.tracked_goal_y:.1f}) m ...")
    drive_info = drive_to_goal(rom, tracked_policy, m113, vehicle, terrain, actuator, vis,
                               args.tracked_goal_x, args.tracked_goal_y, args.tracked_max_steps, device)
    reached_s = drive_info["reached_step"] * rom.step_dt if drive_info["reached_step"] >= 0 else None
    print(f"[phase 1] {drive_info['status']}  min_dist={drive_info['min_dist_m']:.3f} m"
          + (f"  (t={reached_s:.1f} s)" if reached_s is not None else ""))

    # ---- Phase 2: brake to rest, then pin (replaces the arm 6 s pre-roll) ----
    brake_di = veh.DriverInputs()
    brake_di.m_throttle, brake_di.m_steering, brake_di.m_braking = 0.0, 0.0, 1.0
    n_sub = max(1, int(round(CONTROL_DT / STEP_SIZE)))
    # The goal policy has no stop objective, so the base arrives still moving (~2 m/s).
    # Bleed that off under full brake, but pin the chassis the moment it is at rest:
    # a braked-but-UNPINNED M113 keeps a slow residual yaw creep (arm-mass asymmetry +
    # track-ground slip) that reads as the vehicle slowly spinning, so the shorter this
    # unpinned window, the better. Pinning also needs the tracks ~stopped to avoid a jolt.
    max_brake_steps = int(round(float(args.brake_seconds) / CONTROL_DT))
    min_brake_steps = max(1, int(round(float(args.min_brake_seconds) / CONTROL_DT)))
    stop_speed = float(args.brake_stop_speed_mps)
    dbg = os.environ.get("BRAKE_DEBUG")
    print(f"[phase 2] braking to rest (<= {args.brake_seconds:.1f} s, stop at {stop_speed:.2f} m/s) ...")
    braked_steps = 0
    for bi in range(max_brake_steps):
        _substep(m113, terrain, actuator, brake_di, n_sub, vis=vis)
        braked_steps = bi + 1
        speed = abs(vehicle.GetSpeed())
        if dbg and bi % 50 == 0:
            row = capture_row(vehicle, "s", "f", "e", "dbg", 0, 0.0, brake_di)
            print(f"    [brake t={bi*CONTROL_DT:5.1f}s] speed={row['speed_mps']:+.4f} "
                  f"yaw_rate={row['yaw_rate_radps']:+.5f} yaw={row['yaw_rad']:+.4f}", flush=True)
        if braked_steps >= min_brake_steps and speed < stop_speed:
            break
    brake_seconds_used = braked_steps * CONTROL_DT

    # Pin the chassis so the arm phase sees a fixed base (arm-only dynamics), matching
    # how the arm data was collected and how the standalone arm eval runs. Once fixed,
    # the base cannot creep, so the arm phase is perfectly static.
    print(f"[phase 2] at rest after {brake_seconds_used:.1f} s "
          f"(speed {abs(vehicle.GetSpeed()):.3f} m/s); pinning chassis for the arm phase ...")
    vehicle.GetChassisBody().SetFixed(True)

    # ---- Phase 3: arm reach on the shared, settled rig ----
    print(f"[phase 3] arm reaching {args.arm_num_goals} consecutive goals "
          f"(policy={arm_checkpoint.name}) ...")
    sim = make_arm_sim(arm_env, m113, vehicle, terrain, gripper, actuator, collision_links, vis)
    arm_summary, success_tolerance_m = run_arm_phase(
        arm_env, arm_policy, sim, args.arm_num_goals, args.arm_max_steps,
        args.arm_goal_duration_s, output_dir, make_plots=not args.no_plots)

    # ---- summary ----
    aggregate = {
        "demo": "tracked_drive_then_arm_reach",
        "wall_seconds": round(time.time() - wall, 1),
        "device": device,
        "tracked": {
            "policy_checkpoint": str(resolve_dynamics_checkpoint_path(args.tracked_checkpoint)),
            "brake_seconds_cap": float(args.brake_seconds),
            "brake_seconds_used": round(brake_seconds_used, 2),
            **drive_info,
        },
        "arm": {
            "policy_checkpoint": str(arm_checkpoint),
            "success_tolerance_m": success_tolerance_m,
            "requested_goals": int(args.arm_num_goals),
            "num_rollouts": len(arm_summary),
            "success_rate": float(np.mean([r["success"] for r in arm_summary])) if arm_summary else None,
            "mean_final_ee_error_m": (
                float(np.mean([r["final_ee_error_m"] for r in arm_summary])) if arm_summary else None),
            "rollouts": arm_summary,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2))
    print(f"[done] wall={aggregate['wall_seconds']} s  ->  {output_dir/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
