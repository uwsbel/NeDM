"""Paired NRD-vs-Chrono evaluation of a double-pendulum NRD reaching policy (plan section 10).

For a fixed held-out set of (recorded context, goal) pairs the SAME policy is run

  * inside the frozen NRD: reset to the recorded 16-step [z1, z2, a] context, then
    autonomous prediction with the policy acting every ``action_repeat`` steps; and
  * in Chrono: the mechanism is reset to the context's final state, the policy
    reads the TRUE z1 and (for the z1+z2 policy) the camera frame encoded by the
    frozen encoder, and holds each action for the same 0.1 s.

Metrics are computed from the predicted z1 (NRD) and from the true Chrono state.
The decoder is never used.

    PYTHONPATH=src python scripts/evaluation/eval_dpend_nrd_rl_reach.py \
        --run-dir artifacts/rl_runs/<run> --num-pairs 100 --gif-count 4
"""

from __future__ import annotations

import pychrono as chrono  # noqa: F401  # load Chrono before torch/libstdc++ users

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import nedm.double_pendulum_data as dp  # noqa: E402
from nedm.nrd.context_bank import load_context_bank  # noqa: E402
from nedm.rl.dpend_nrd_reach_env import (  # noqa: E402
    DEFAULT_EVAL_CONTEXT_BANK,
    DPEND_STATE_FIELDS,
    DPendNRDReachEnv,
    make_eval_pairs,
)

TOLERANCE_CURVE_M = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1]


# ---------------------------------------------------------------------------
# Policy loading (rsl_rl runner checkpoint)
# ---------------------------------------------------------------------------
def latest_policy_checkpoint(run_dir: Path) -> Path:
    candidates = []
    for path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"no model_*.pt in {run_dir}")
    return max(candidates)[1]


def load_runner_checkpoint(runner: OnPolicyRunner, checkpoint_path: Path, device: str) -> int:
    loaded = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)
    runner.alg.actor_critic.load_state_dict(loaded["model_state_dict"])
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded["critic_obs_norm_state_dict"])
    runner.current_learning_iteration = int(loaded["iter"])
    return int(loaded["iter"])


# ---------------------------------------------------------------------------
# NRD-side rollouts
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_nrd_rollouts(
    env: DPendNRDReachEnv, policy, context_ids: torch.Tensor, goals: torch.Tensor, max_steps: int
) -> dict[str, np.ndarray]:
    num = context_ids.numel()
    env_ids = torch.arange(num, device=env.device)
    env.reset_idx(env_ids, context_ids, goals)
    obs, _ = env.get_observations()
    tips = [env.tip_positions(env.state_hist[:, -1, :]).cpu().numpy()]
    actions = []
    done_step = np.full(num, -1, dtype=np.int64)
    for step in range(max_steps):
        obs, _, dones, _ = env.step(policy(obs))
        actions.append(env.actions.squeeze(-1).cpu().numpy())
        tips.append(env.tip_positions(env.state_hist[:, -1, :]).cpu().numpy())
        newly = (dones.cpu().numpy() > 0) & (done_step < 0)
        done_step[newly] = step + 1
        if bool(dones.all()):
            break
    records = {key: value.cpu().numpy() for key, value in env.episode_records().items()}
    records["episode_steps"] = np.where(done_step > 0, done_step, len(actions)).astype(np.float32)
    records["tips"] = np.stack(tips, axis=1)
    records["actions"] = np.stack(actions, axis=1)
    records["min_distance_fine_m"] = records["min_distance_m"].copy()  # no sub-transition resolution in the NRD
    return records


# ---------------------------------------------------------------------------
# Chrono-side rollouts
# ---------------------------------------------------------------------------
def _tip_world(scene: dp.PendulumScene) -> tuple[float, float]:
    tip = scene.link2.TransformPointLocalToParent(chrono.ChVector3d(0, 0, -dp.LINK2_LENGTH_M / 2.0))
    return float(tip.x), float(tip.z)


@torch.no_grad()
def run_chrono_episode(
    scene: dp.PendulumScene,
    tap: dp.FrameTap,
    env: DPendNRDReachEnv,
    policy,
    context_state: np.ndarray,
    goal_xz: np.ndarray,
    max_steps: int,
    render_every_boundary: bool = False,
) -> dict[str, Any]:
    tolerance = env.success_tolerance
    omega_limit = env.omega_limit
    omega_sigma = env.omega_sigma.cpu().numpy()
    substeps = dp.SUBSTEPS_PER_CONTROL
    device = env.device
    goal_t = torch.as_tensor(goal_xz, dtype=torch.float32, device=device).view(1, 2)

    q1 = math.atan2(float(context_state[1]), float(context_state[0]))
    q2 = math.atan2(float(context_state[3]), float(context_state[2]))
    w1, w2 = float(context_state[4]), float(context_state[5])
    scene.elbow_torque.SetSetpoint(0.0, scene.system.GetChTime())
    dp.reset_state(scene, q1, q2, w1, w2)

    frames: list[np.ndarray] = []
    tips: list[tuple[float, float]] = []
    actions: list[float] = []
    raw_actions: list[float] = []
    dists: list[float] = []
    domega_sq_sum = 0.0
    domega_max = 0.0
    boundary_count = 0
    min_fine = float("inf")
    success = False
    success_time_s = float("nan")
    spin = False
    steps_taken = 0
    start_state = dp.read_state(scene)
    prev_omega = np.array([start_state["omega1_radps"], start_state["omega2_radps"]])
    tips.append((start_state["tip_x_m"], start_state["tip_z_m"]))
    dists.append(float(math.hypot(start_state["tip_x_m"] - goal_xz[0], start_state["tip_z_m"] - goal_xz[1])))
    min_fine = min(min_fine, dists[-1])
    current = start_state

    for step in range(max_steps):
        # Manual trigger: exactly one render of the CURRENT state per Update().
        scene.manager.Update()
        frame = tap.take()
        frames.append(frame)
        z1 = torch.tensor([[current[field] for field in DPEND_STATE_FIELDS]], dtype=torch.float32, device=device)
        z2 = None
        if env.observe_z2:
            z2 = env.model.encode_frame_sequence(torch.from_numpy(frame)[None, None].to(device))[:, 0, :]
        obs = env.build_observation(z1, z2, goal_t)
        raw = float(policy(obs).flatten()[0].item())
        action = max(-1.0, min(1.0, raw))
        raw_actions.append(raw)
        actions.append(action)
        scene.elbow_torque.SetSetpoint(action * dp.TAU_MAX_NM, scene.system.GetChTime())
        steps_taken = step + 1

        terminated = False
        for k in range(env.action_repeat):
            for _ in range(substeps):
                scene.system.DoStepDynamics(dp.DT_SIM_S)
                tx, tz = _tip_world(scene)
                min_fine = min(min_fine, math.hypot(tx - goal_xz[0], tz - goal_xz[1]))
            current = dp.read_state(scene)
            omega = np.array([current["omega1_radps"], current["omega2_radps"]])
            domega = (omega - prev_omega) / omega_sigma
            prev_omega = omega
            domega_sq_sum += float((domega**2).sum())
            domega_max = max(domega_max, float(np.abs(domega).max()))
            boundary_count += 1
            dist = float(math.hypot(current["tip_x_m"] - goal_xz[0], current["tip_z_m"] - goal_xz[1]))
            tips.append((current["tip_x_m"], current["tip_z_m"]))
            dists.append(dist)
            if dist <= tolerance:
                success = True
                success_time_s = (step * env.action_repeat + k + 1) * env.dt_s
                terminated = True
            elif bool((np.abs(omega) > omega_limit).any()):
                spin = True
                terminated = True
            if terminated:
                break
            if render_every_boundary and k < env.action_repeat - 1:
                scene.manager.Update()
                frames.append(tap.take())
        if terminated:
            break

    # Final frame of the terminal state (GIF), then one quiet control period.
    scene.manager.Update()
    frames.append(tap.take())
    scene.elbow_torque.SetSetpoint(0.0, scene.system.GetChTime())
    dp._advance_to_next_boundary(scene)

    actions_np = np.asarray(actions, dtype=np.float32)
    raw_np = np.asarray(raw_actions, dtype=np.float32)
    prev = np.concatenate([[actions_np[0]], actions_np[:-1]]) if len(actions_np) else actions_np
    return {
        "success": float(success),
        "spin": float(spin),
        "ood": 0.0,
        "nonfinite": 0.0,
        "time_out": float(not success and not spin),
        "episode_steps": float(steps_taken),
        "success_time_s": success_time_s if success else 0.0,
        "initial_distance_m": dists[0],
        "final_distance_m": dists[-1],
        "min_distance_m": float(min(dists)),
        "min_distance_fine_m": float(min_fine),
        "domega_rms": math.sqrt(domega_sq_sum / max(boundary_count, 1)),
        "domega_max": domega_max,
        "action_abs_mean": float(np.abs(actions_np).mean()) if len(actions_np) else 0.0,
        "action_slew_mean": float(np.abs(actions_np - prev).mean()) if len(actions_np) else 0.0,
        "action_saturated_frac": float((np.abs(raw_np) > 1.0).mean()) if len(raw_np) else 0.0,
        "tips": np.asarray(tips, dtype=np.float32),
        "actions": actions_np,
        "frames": frames,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _pct(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


def summarize(records: dict[str, np.ndarray], goals: np.ndarray, tolerance: float) -> dict[str, Any]:
    success = records["success"] > 0.5
    upper = goals[:, 1] > 0.0
    min_fine = records["min_distance_fine_m"]
    out = {
        "success_rate": float(success.mean()),
        "timeout_rate": float(records["time_out"].mean()),
        "spin_rate": float(records["spin"].mean()),
        "ood_rate": float(records["ood"].mean()),
        "success_rate_upper_goals": float(success[upper].mean()) if upper.any() else float("nan"),
        "success_rate_lower_goals": float(success[~upper].mean()) if (~upper).any() else float("nan"),
        "final_distance_m": {"median": _pct(records["final_distance_m"], 50), "p90": _pct(records["final_distance_m"], 90)},
        "min_distance_m": {"median": _pct(records["min_distance_m"], 50), "p90": _pct(records["min_distance_m"], 90)},
        "min_distance_fine_m": {"median": _pct(min_fine, 50), "p90": _pct(min_fine, 90)},
        "time_to_success_s": {
            "median": _pct(records["success_time_s"][success], 50),
            "p90": _pct(records["success_time_s"][success], 90),
        },
        "domega_rms_mean": float(records["domega_rms"].mean()),
        "domega_max_mean": float(records["domega_max"].mean()),
        "domega_max_max": float(records["domega_max"].max()),
        "action_abs_mean": float(records["action_abs_mean"].mean()),
        "action_slew_mean": float(records["action_slew_mean"].mean()),
        "action_saturated_frac": float(records["action_saturated_frac"].mean()),
        "success_vs_tolerance": {
            f"{tol * 1000:.0f}mm": float((records["min_distance_m"] <= tol).mean()) for tol in TOLERANCE_CURVE_M
        },
        "success_vs_tolerance_fine": {
            f"{tol * 1000:.0f}mm": float((min_fine <= tol).mean()) for tol in TOLERANCE_CURVE_M
        },
    }
    return out


def write_gif(path: Path, frames: list[np.ndarray], goal_xz: np.ndarray, tips: np.ndarray, tolerance: float, label: str, fps: float) -> None:
    from PIL import Image, ImageDraw

    scale = 2
    focal_px = (dp.IMAGE_WIDTH / 2.0) / math.tan(dp.CAMERA_HFOV_RAD / 2.0)
    gu, gv = dp.project_to_pixel(float(goal_xz[0]), float(goal_xz[1]))
    radius_px = max(3.0, focal_px * tolerance / dp.CAMERA_DISTANCE_M * scale)
    images = []
    for index, frame in enumerate(frames):
        image = Image.fromarray(frame).resize((dp.IMAGE_WIDTH * scale, dp.IMAGE_HEIGHT * scale), Image.NEAREST)
        draw = ImageDraw.Draw(image)
        cx, cy = gu * scale, gv * scale
        draw.ellipse([cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px], outline=(80, 230, 120), width=2)
        draw.line([cx - 8, cy, cx + 8, cy], fill=(80, 230, 120), width=1)
        draw.line([cx, cy - 8, cx, cy + 8], fill=(80, 230, 120), width=1)
        tip_index = min(index, len(tips) - 1)
        dist_mm = math.hypot(tips[tip_index, 0] - goal_xz[0], tips[tip_index, 1] - goal_xz[1]) * 1000.0
        draw.text((4, 4), f"{label}  t={index / fps:4.2f}s  d={dist_mm:5.1f} mm", fill=(240, 240, 240))
        images.append(image)
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(round(1000.0 / fps)), loop=0)


def write_figure(path: Path, nrd: dict[str, np.ndarray], chrono_rec: dict[str, np.ndarray] | None, tolerance: float, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"nrd": "#2a78d6", "chrono": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    tols = np.geomspace(0.002, 0.2, 60)
    ax = axes[0]
    ax.plot(tols * 1000, [(nrd["min_distance_m"] <= t).mean() for t in tols], color=colors["nrd"], lw=2, label="NRD (predicted z1)")
    if chrono_rec is not None:
        ax.plot(tols * 1000, [(chrono_rec["min_distance_m"] <= t).mean() for t in tols], color=colors["chrono"], lw=2, label="Chrono (true state)")
    ax.axvline(tolerance * 1000, color="#8a8a85", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("tolerance on closest approach [mm]")
    ax.set_ylabel("fraction of episodes within tolerance")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    if chrono_rec is not None:
        x = np.clip(nrd["min_distance_m"] * 1000, 0.5, None)
        y = np.clip(chrono_rec["min_distance_m"] * 1000, 0.5, None)
        ax.scatter(x, y, s=14, color="#4a4a46", alpha=0.7)
        lim = (0.5, max(float(x.max()), float(y.max())) * 1.2)
        ax.plot(lim, lim, color="#8a8a85", lw=1, ls=":")
        ax.axvline(tolerance * 1000, color=colors["nrd"], lw=1, ls="--")
        ax.axhline(tolerance * 1000, color=colors["chrono"], lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("NRD closest approach [mm]")
        ax.set_ylabel("Chrono closest approach [mm]")
        ax.grid(alpha=0.25, which="both")
    else:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, default=None, help="defaults to the latest model_*.pt")
    parser.add_argument("--context-bank", type=Path, default=DEFAULT_EVAL_CONTEXT_BANK, help="held-out (val) bank")
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--pairs-seed", type=int, default=20260826)
    parser.add_argument("--mode", choices=["nrd", "chrono", "both"], default="both")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--success-tolerance-m", type=float, default=None)
    parser.add_argument("--gif-count", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    device = args.device
    torch.set_float32_matmul_precision("high")
    env_cfg = json.loads((run_dir / "env_cfg.json").read_text())
    train_cfg = json.loads((run_dir / "train_cfg.json").read_text())
    env_cfg.update({"num_envs": int(args.num_pairs), "device": device, "auto_reset": False, "context_bank": str(args.context_bank)})
    if args.success_tolerance_m is not None:
        env_cfg["reward"]["success_tolerance_m"] = float(args.success_tolerance_m)
    max_steps = int(args.max_steps) if args.max_steps is not None else int(env_cfg["max_episode_steps"])

    env = DPendNRDReachEnv(env_cfg, device=device)
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=device)
    checkpoint_path = args.policy_checkpoint.resolve() if args.policy_checkpoint else latest_policy_checkpoint(run_dir)
    iteration = load_runner_checkpoint(runner, checkpoint_path, device)
    policy = runner.get_inference_policy(device=device)
    policy_obs = "z1z2" if env.observe_z2 else "z1"
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / f"nrd_chrono_transfer_eval_iter{iteration}"
    output_dir.mkdir(parents=True, exist_ok=True)

    bank = load_context_bank(args.context_bank)
    context_ids, goals = make_eval_pairs(
        bank, int(args.num_pairs), int(args.pairs_seed), env.cfg["goal"], env.link_lengths, env.success_tolerance
    )
    goals_np = goals.numpy()
    print(f"run={run_dir.name} policy_obs={policy_obs} checkpoint={checkpoint_path.name} pairs={args.num_pairs} "
          f"(upper-half goals {int((goals_np[:, 1] > 0).sum())}) tolerance={env.success_tolerance * 1000:.0f} mm")

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "policy_obs": policy_obs,
        "policy_checkpoint": str(checkpoint_path),
        "policy_iteration": iteration,
        "context_bank": str(args.context_bank),
        "num_pairs": int(args.num_pairs),
        "pairs_seed": int(args.pairs_seed),
        "success_tolerance_m": env.success_tolerance,
        "max_steps": max_steps,
        "action_repeat": env.action_repeat,
    }
    nrd_rec = None
    chrono_rec = None

    if args.mode in ("nrd", "both"):
        started = time.time()
        nrd_rec = run_nrd_rollouts(env, policy, context_ids, goals, max_steps)
        summary["nrd"] = summarize(nrd_rec, goals_np, env.success_tolerance)
        summary["nrd"]["wall_s"] = time.time() - started
        print(f"NRD: success {summary['nrd']['success_rate']:.2f} (upper {summary['nrd']['success_rate_upper_goals']:.2f} / "
              f"lower {summary['nrd']['success_rate_lower_goals']:.2f}) timeout {summary['nrd']['timeout_rate']:.2f} "
              f"spin {summary['nrd']['spin_rate']:.2f} ood {summary['nrd']['ood_rate']:.2f} "
              f"min-dist median {summary['nrd']['min_distance_m']['median'] * 1000:.1f} mm")

    if args.mode in ("chrono", "both"):
        started = time.time()
        scene = dp.build_scene(with_camera=True)
        tap = dp.FrameTap(scene.camera)
        chrono_list: list[dict[str, Any]] = []
        for pair_index in range(int(args.num_pairs)):
            context_state = bank["states"][int(context_ids[pair_index]), -1, :]
            record = run_chrono_episode(
                scene, tap, env, policy, context_state, goals_np[pair_index], max_steps,
                render_every_boundary=pair_index < int(args.gif_count),
            )
            if pair_index < int(args.gif_count):
                fps = 1.0 / env.dt_s
                write_gif(
                    output_dir / f"chrono_pair_{pair_index:03d}.gif", record["frames"], goals_np[pair_index],
                    record["tips"], env.success_tolerance, f"{policy_obs} #{pair_index}", fps,
                )
            record.pop("frames")
            chrono_list.append(record)
            if (pair_index + 1) % 10 == 0:
                done_so_far = np.mean([r["success"] for r in chrono_list])
                print(f"  chrono {pair_index + 1}/{args.num_pairs}: running success {done_so_far:.2f}")
        scalar_keys = [k for k in chrono_list[0] if k not in ("tips", "actions")]
        chrono_rec = {key: np.asarray([r[key] for r in chrono_list], dtype=np.float32) for key in scalar_keys}
        summary["chrono"] = summarize(chrono_rec, goals_np, env.success_tolerance)
        summary["chrono"]["wall_s"] = time.time() - started
        print(f"Chrono: success {summary['chrono']['success_rate']:.2f} (upper {summary['chrono']['success_rate_upper_goals']:.2f} / "
              f"lower {summary['chrono']['success_rate_lower_goals']:.2f}) timeout {summary['chrono']['timeout_rate']:.2f} "
              f"spin {summary['chrono']['spin_rate']:.2f} min-dist median {summary['chrono']['min_distance_m']['median'] * 1000:.1f} mm "
              f"(1 ms resolution {summary['chrono']['min_distance_fine_m']['median'] * 1000:.1f} mm)")
        np.savez_compressed(
            output_dir / "chrono_trajectories.npz",
            tips=np.array([r["tips"] for r in chrono_list], dtype=object),
            actions=np.array([r["actions"] for r in chrono_list], dtype=object),
            goals=goals_np,
            context_ids=context_ids.numpy(),
        )

    if nrd_rec is not None and chrono_rec is not None:
        nrd_ok = nrd_rec["success"] > 0.5
        chr_ok = chrono_rec["success"] > 0.5
        summary["paired"] = {
            "both_success": int((nrd_ok & chr_ok).sum()),
            "nrd_only_success": int((nrd_ok & ~chr_ok).sum()),
            "chrono_only_success": int((~nrd_ok & chr_ok).sum()),
            "neither": int((~nrd_ok & ~chr_ok).sum()),
            "transfer_gap_success_rate": float(nrd_ok.mean() - chr_ok.mean()),
            "nrd_success_chrono_fail_pairs": [int(i) for i in np.nonzero(nrd_ok & ~chr_ok)[0]],
            "min_distance_abs_diff_mm": {
                "median": _pct(np.abs(nrd_rec["min_distance_m"] - chrono_rec["min_distance_m"]) * 1000, 50),
                "p90": _pct(np.abs(nrd_rec["min_distance_m"] - chrono_rec["min_distance_m"]) * 1000, 90),
            },
        }
        print(f"paired: both {summary['paired']['both_success']} nrd-only {summary['paired']['nrd_only_success']} "
              f"chrono-only {summary['paired']['chrono_only_success']} neither {summary['paired']['neither']} "
              f"gap {summary['paired']['transfer_gap_success_rate']:+.2f}")

    per_pair = []
    for index in range(int(args.num_pairs)):
        row: dict[str, Any] = {
            "pair": index,
            "context_id": int(context_ids[index]),
            "episode_id": bank["meta"]["episode_ids"][int(context_ids[index])],
            "goal_x_m": float(goals_np[index, 0]),
            "goal_z_m": float(goals_np[index, 1]),
        }
        for name, rec in (("nrd", nrd_rec), ("chrono", chrono_rec)):
            if rec is None:
                continue
            row[name] = {
                key: float(rec[key][index])
                for key in ("success", "spin", "ood", "time_out", "episode_steps", "success_time_s",
                            "initial_distance_m", "final_distance_m", "min_distance_m", "min_distance_fine_m",
                            "domega_rms", "domega_max", "action_abs_mean", "action_slew_mean", "action_saturated_frac")
            }
        per_pair.append(row)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "per_pair.json").write_text(json.dumps(per_pair, indent=2))
    if nrd_rec is not None:
        np.savez_compressed(output_dir / "nrd_trajectories.npz", tips=nrd_rec["tips"], actions=nrd_rec["actions"],
                            goals=goals_np, context_ids=context_ids.numpy())
        write_figure(output_dir / "closest_approach.png", nrd_rec, chrono_rec, env.success_tolerance,
                     f"{run_dir.name} @ iter {iteration}: closest approach to goal, {args.num_pairs} held-out pairs")
    print(f"wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
