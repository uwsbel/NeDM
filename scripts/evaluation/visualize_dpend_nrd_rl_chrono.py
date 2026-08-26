"""Watch a trained NRD reaching policy chase consecutive goals in Chrono (Irrlicht).

The policy (trained entirely inside the frozen NRD) drives the true Chrono double
pendulum through a chain of goals. The current goal is a GREEN sphere whose radius
equals the success tolerance; a reached goal leaves a small GRAY ball behind, a
timed-out goal a RED one. Goals are the same held-out (context, goal) pairs the
paired evaluation uses, so per-goal outcomes are comparable with
``nrd_chrono_transfer_eval_iter*/per_pair.json``.

By default the state is NOT reset between goals (one continuous rollout, the next
goal appears where the previous one ended); ``--reset-between`` teleports to each
pair's recorded context state instead, exactly as the paired evaluation does.

    PYTHONPATH=src DISPLAY=:1 python scripts/evaluation/visualize_dpend_nrd_rl_chrono.py \
        --run-dir artifacts/rl_runs/<run> --num-goals 10

Marker spheres are fixed, collision-free bodies added before the first reset; the
physics is unchanged (``--check-markers`` verifies bitwise-identical states with
and without them).
"""

from __future__ import annotations

import pychrono as chrono  # noqa: F401  # load Chrono before torch/libstdc++ users

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = REPO_ROOT / "scripts" / "evaluation"
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import nedm.double_pendulum_data as dp  # noqa: E402
from eval_dpend_nrd_rl_reach import latest_policy_checkpoint, load_runner_checkpoint  # noqa: E402
from nedm.nrd.context_bank import load_context_bank  # noqa: E402
from nedm.rl.dpend_nrd_reach_env import DEFAULT_EVAL_CONTEXT_BANK, DPEND_STATE_FIELDS, DPendNRDReachEnv, make_eval_pairs  # noqa: E402

GOAL_RGB = (0.15, 0.9, 0.3)
REACHED_RGB = (0.75, 0.75, 0.75)
TIMEOUT_RGB = (0.9, 0.2, 0.2)
PARKED = chrono.ChVector3d(0.0, 0.0, -50.0)  # markers wait here, far below the camera's view


# ---------------------------------------------------------------------------
# Goal markers: fixed, collision-free bodies (no DOFs -> no effect on the solver)
# ---------------------------------------------------------------------------
class GoalMarkers:
    def __init__(self, scene: dp.PendulumScene, count: int, goal_radius_m: float) -> None:
        self.goal = self._marker(scene, goal_radius_m, GOAL_RGB)
        self.reached = [self._marker(scene, 0.6 * goal_radius_m, REACHED_RGB) for _ in range(count)]
        self.timed_out = [self._marker(scene, 0.6 * goal_radius_m, TIMEOUT_RGB) for _ in range(count)]
        self._reached_used = 0
        self._timeout_used = 0

    @staticmethod
    def _marker(scene: dp.PendulumScene, radius: float, rgb: tuple[float, float, float]) -> chrono.ChBody:
        body = chrono.ChBody()
        body.SetFixed(True)
        body.EnableCollision(False)
        body.SetPos(PARKED)
        sphere = chrono.ChVisualShapeSphere(radius)
        sphere.SetColor(chrono.ChColor(*rgb))
        body.AddVisualShape(sphere)
        scene.system.Add(body)
        return body

    def show_goal(self, x: float, z: float) -> None:
        self.goal.SetPos(chrono.ChVector3d(x, 0.0, z))

    def mark_reached(self, x: float, z: float) -> None:
        self.reached[self._reached_used % len(self.reached)].SetPos(chrono.ChVector3d(x, 0.0, z))
        self._reached_used += 1

    def mark_timeout(self, x: float, z: float) -> None:
        self.timed_out[self._timeout_used % len(self.timed_out)].SetPos(chrono.ChVector3d(x, 0.0, z))
        self._timeout_used += 1


# ---------------------------------------------------------------------------
def make_visual_system(scene: dp.PendulumScene, width: int, height: int, title: str):
    import pychrono.irrlicht as irr

    vis = irr.ChVisualSystemIrrlicht()
    vis.AttachSystem(scene.system)
    vis.SetWindowSize(width, height)
    vis.SetWindowTitle(title)
    vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
    vis.Initialize()
    # Same viewpoint as the Chrono::Sensor camera used for z2: on -Y looking at the pivot.
    vis.AddCamera(chrono.ChVector3d(0.0, -dp.CAMERA_DISTANCE_M, 0.0), chrono.ChVector3d(0.0, 0.0, 0.0))
    vis.AddTypicalLights()
    vis.AddLightDirectional()
    try:
        vis.BindAll()
    except Exception:
        pass
    return vis


def render(vis) -> bool:
    if vis is None:
        return True
    if not vis.Run():
        return False
    vis.BeginScene()
    vis.Render()
    vis.EndScene()
    return True


# ---------------------------------------------------------------------------
@torch.no_grad()
def run_goal(
    scene: dp.PendulumScene,
    tap: dp.FrameTap | None,
    env: DPendNRDReachEnv,
    policy,
    goal_xz: np.ndarray,
    max_steps: int,
    vis,
    realtime_factor: float,
) -> dict[str, Any]:
    """Drive the policy toward one goal from the CURRENT Chrono state. Renders every control period."""
    device = env.device
    tolerance = env.success_tolerance
    goal_t = torch.as_tensor(goal_xz, dtype=torch.float32, device=device).view(1, 2)
    timer = chrono.ChRealtimeStepTimer() if realtime_factor > 0 else None
    substeps = dp.SUBSTEPS_PER_CONTROL

    current = dp.read_state(scene)
    dists = [math.hypot(current["tip_x_m"] - goal_xz[0], current["tip_z_m"] - goal_xz[1])]
    actions: list[float] = []
    success = False
    success_time_s = float("nan")
    spin = False
    window_closed = False
    steps_taken = 0

    for step in range(max_steps):
        frame_latent = None
        if tap is not None:
            scene.manager.Update()
            frame = tap.take()
            frame_latent = env.model.encode_frame_sequence(torch.from_numpy(frame)[None, None].to(device))[:, 0, :]
        z1 = torch.tensor([[current[field] for field in DPEND_STATE_FIELDS]], dtype=torch.float32, device=device)
        action = float(policy(env.build_observation(z1, frame_latent, goal_t)).flatten()[0].clamp(-1.0, 1.0).item())
        actions.append(action)
        scene.elbow_torque.SetSetpoint(action * dp.TAU_MAX_NM, scene.system.GetChTime())
        steps_taken = step + 1

        terminated = False
        for _ in range(env.action_repeat):
            for _ in range(substeps):
                scene.system.DoStepDynamics(dp.DT_SIM_S)
            current = dp.read_state(scene)
            dist = math.hypot(current["tip_x_m"] - goal_xz[0], current["tip_z_m"] - goal_xz[1])
            dists.append(dist)
            if not render(vis):
                window_closed = True
                terminated = True
                break
            if timer is not None:
                timer.Spin(dp.CONTROL_DT_S / realtime_factor)
            if dist <= tolerance:
                success = True
                success_time_s = len(dists[1:]) * dp.CONTROL_DT_S
                terminated = True
                break
            if abs(current["omega1_radps"]) > env.omega_limit or abs(current["omega2_radps"]) > env.omega_limit:
                spin = True
                terminated = True
                break
        if terminated:
            break

    return {
        "success": success,
        "spin": spin,
        "time_out": not success and not spin and not window_closed,
        "window_closed": window_closed,
        "policy_steps": steps_taken,
        "elapsed_s": (len(dists) - 1) * dp.CONTROL_DT_S,
        "success_time_s": success_time_s,
        "initial_distance_m": dists[0],
        "final_distance_m": dists[-1],
        "min_distance_m": min(dists),
        "action_abs_mean": float(np.mean(np.abs(actions))) if actions else 0.0,
        "final_state": current,
    }


def teleport_to_context(scene: dp.PendulumScene, context_state: np.ndarray) -> None:
    q1 = math.atan2(float(context_state[1]), float(context_state[0]))
    q2 = math.atan2(float(context_state[3]), float(context_state[2]))
    scene.elbow_torque.SetSetpoint(0.0, scene.system.GetChTime())
    dp.reset_state(scene, q1, q2, float(context_state[4]), float(context_state[5]))


# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, default=None)
    parser.add_argument("--context-bank", type=Path, default=DEFAULT_EVAL_CONTEXT_BANK)
    parser.add_argument("--num-goals", type=int, default=10)
    parser.add_argument("--pairs-seed", type=int, default=20260826)
    parser.add_argument("--first-pair", type=int, default=0, help="start at this index of the held-out pair list")
    parser.add_argument("--reset-between", action="store_true", help="teleport to each pair's context state (paired-eval protocol)")
    parser.add_argument("--max-steps", type=int, default=None, help="policy steps per goal (default: run's max_episode_steps)")
    parser.add_argument("--realtime-factor", type=float, default=1.0, help="1 = real time, 2 = twice as fast, 0 = unthrottled")
    parser.add_argument("--window", type=int, nargs=2, default=[1280, 960])
    parser.add_argument("--cycles", type=int, default=1, help="repeat the chain this many times, moving on to the next pairs each cycle")
    parser.add_argument("--no-hold", action="store_true", help="exit when the chain ends instead of keeping the window open")
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--no-markers", action="store_true")
    parser.add_argument("--check-markers", action="store_true",
                        help="headless: run the chain with and without marker bodies and compare the final states bitwise")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def run_chain(args: argparse.Namespace, env: DPendNRDReachEnv, policy, context_ids, goals_np, with_markers: bool, with_vis: bool):
    scene = dp.build_scene(with_camera=env.observe_z2)
    tap = dp.FrameTap(scene.camera) if env.observe_z2 else None
    markers = GoalMarkers(scene, args.num_goals, env.success_tolerance) if with_markers else None
    max_steps = int(args.max_steps) if args.max_steps is not None else int(env.max_episode_length)
    bank_states = load_context_bank(args.context_bank)["states"]
    vis = None
    if with_vis:
        vis = make_visual_system(scene, *args.window, f"{args.run_dir.name}: {args.num_goals} consecutive goals")

    results = []
    total_goals = int(args.num_goals) * max(1, int(args.cycles))
    for index in range(total_goals):
        if index % args.num_goals == 0:
            # New cycle: start from that pair's recorded context state (as in the paired evaluation).
            teleport_to_context(scene, bank_states[int(context_ids[index]), -1, :])
            if index > 0:
                print(f"--- cycle {index // args.num_goals + 1}/{args.cycles} ---", flush=True)
        elif args.reset_between:
            teleport_to_context(scene, bank_states[int(context_ids[index]), -1, :])
        goal = goals_np[index]
        if markers is not None:
            markers.show_goal(float(goal[0]), float(goal[1]))
        started = time.time()
        record = run_goal(scene, tap, env, policy, goal, max_steps, vis, float(args.realtime_factor))
        record["wall_s"] = time.time() - started
        record["pair"] = int(args.first_pair + index)
        record["context_id"] = int(context_ids[index])
        record["goal_x_m"], record["goal_z_m"] = float(goal[0]), float(goal[1])
        outcome = "SUCCESS" if record["success"] else ("SPIN" if record["spin"] else ("CLOSED" if record["window_closed"] else "TIMEOUT"))
        print(
            f"goal {index % args.num_goals + 1:2d}/{args.num_goals} pair {record['pair']:3d} at ({goal[0]:+.3f}, {goal[1]:+.3f}) m: {outcome:7s} "
            f"t={record['elapsed_s']:.2f}s closest={record['min_distance_m'] * 1000:5.1f} mm "
            f"final={record['final_distance_m'] * 1000:5.1f} mm |a|={record['action_abs_mean']:.2f}",
            flush=True,
        )
        if markers is not None:
            if record["success"]:
                markers.mark_reached(float(goal[0]), float(goal[1]))
            elif record["time_out"] or record["spin"]:
                markers.mark_timeout(float(goal[0]), float(goal[1]))
        results.append(record)
        if record["window_closed"]:
            break
        # One quiet control period between goals (the collector's convention).
        scene.elbow_torque.SetSetpoint(0.0, scene.system.GetChTime())
        dp._advance_to_next_boundary(scene)
        render(vis)
    if markers is not None:
        markers.goal.SetPos(PARKED)
    if vis is not None and not args.no_hold and not (results and results[-1]["window_closed"]):
        print("chain finished -- window stays open (close it or Ctrl-C to exit)", flush=True)
        while render(vis):
            time.sleep(0.02)
    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    device = args.device
    torch.set_float32_matmul_precision("high")
    env_cfg = json.loads((run_dir / "env_cfg.json").read_text())
    train_cfg = json.loads((run_dir / "train_cfg.json").read_text())
    env_cfg.update({"num_envs": 1, "device": device, "auto_reset": False, "context_bank": str(args.context_bank)})
    env = DPendNRDReachEnv(env_cfg, device=device)
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=device)
    checkpoint = args.policy_checkpoint.resolve() if args.policy_checkpoint else latest_policy_checkpoint(run_dir)
    iteration = load_runner_checkpoint(runner, checkpoint, device)
    policy = runner.get_inference_policy(device=device)

    bank = load_context_bank(args.context_bank)
    context_ids, goals = make_eval_pairs(
        bank, args.first_pair + args.num_goals * max(1, int(args.cycles)), args.pairs_seed, env.cfg["goal"], env.link_lengths, env.success_tolerance
    )
    context_ids = context_ids[args.first_pair :]
    goals_np = goals.numpy()[args.first_pair :]
    print(
        f"run={run_dir.name} checkpoint={checkpoint.name} (iter {iteration}) policy_obs={'z1z2' if env.observe_z2 else 'z1'} "
        f"tolerance={env.success_tolerance * 1000:.0f} mm goals={args.num_goals} "
        f"mode={'reset-between' if args.reset_between else 'continuous'} realtime x{args.realtime_factor}",
        flush=True,
    )

    if args.check_markers:
        saved = args.realtime_factor
        args.realtime_factor = 0.0
        args.cycles = 1
        a = run_chain(args, env, policy, context_ids, goals_np, with_markers=True, with_vis=False)
        b = run_chain(args, env, policy, context_ids, goals_np, with_markers=False, with_vis=False)
        args.realtime_factor = saved
        identical = all(
            ra["final_state"][key] == rb["final_state"][key]
            for ra, rb in zip(a, b)
            for key in ("q1_rad", "q2_rad", "omega1_radps", "omega2_radps")
        ) and [r["success"] for r in a] == [r["success"] for r in b]
        print(f"marker check: final states {'BITWISE IDENTICAL' if identical else 'DIFFER'} with vs without marker bodies")
        return 0 if identical else 1

    results = run_chain(args, env, policy, context_ids, goals_np, with_markers=not args.no_markers, with_vis=not args.no_vis)
    successes = sum(int(r["success"]) for r in results)
    print(f"\n{successes}/{len(results)} goals reached; median closest approach "
          f"{np.median([r['min_distance_m'] for r in results]) * 1000:.1f} mm; "
          f"total simulated time {sum(r['elapsed_s'] for r in results):.1f} s", flush=True)
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / f"chrono_consecutive_goals_iter{iteration}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in results:
        record.pop("final_state", None)
    (output_dir / "consecutive_goals.json").write_text(json.dumps(
        {"checkpoint": str(checkpoint), "iteration": iteration, "mode": "reset-between" if args.reset_between else "continuous",
         "tolerance_m": env.success_tolerance, "results": results}, indent=2))
    print(f"wrote {output_dir / 'consecutive_goals.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
