"""Evaluate a trained tracked-vehicle goal-reaching policy and plot the trajectories.

Rolls the deterministic policy out on N randomly sampled goals (one per env, no
auto-reset) and renders a small-multiples figure: one panel per goal, the driven
x-y path coloured by time (light -> dark = start -> end), the start pose, the goal
with its success tolerance, and a per-panel summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rsl_rl.runners import OnPolicyRunner

from nedm.rl.tracked_goal_env import TrackedGoalReachingEnv, merge_env_cfg

# palette (from the dataviz reference instance)
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
GOOD = "#0ca30c"
CRIT = "#d03b3b"
TIME_CMAP = LinearSegmentedColormap.from_list("progress_blue", ["#b7d3f6", "#5598e7", "#256abf", "#0d366b"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval + plot tracked goal-reaching policy.")
    p.add_argument("--checkpoint", type=Path,
                   default=REPO_ROOT / "artifacts/rl_runs/tracked_goal_v1/model_999.pt")
    p.add_argument("--num-goals", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=None, help="PNG path (default: <run_dir>/goal_reach_trajectories.png)")
    return p.parse_args(argv)


def load_policy(env: TrackedGoalReachingEnv, train_cfg: dict, checkpoint: Path, device: str):
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=device)
    loaded = torch.load(checkpoint, map_location=torch.device(device), weights_only=False)
    runner.alg.actor_critic.load_state_dict(loaded["model_state_dict"])
    if getattr(runner.alg, "rnd", None):
        runner.alg.rnd.load_state_dict(loaded["rnd_state_dict"])
    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded["critic_obs_norm_state_dict"])
    return runner.get_inference_policy(device=device)


def rollout(env: TrackedGoalReachingEnv, policy) -> dict:
    n = env.num_envs
    obs, _ = env.reset()
    goals = env.goal.clone().cpu().numpy()
    poses = [env.pose[:, :2].clone().cpu().numpy()]
    reached_step = np.full(n, -1, dtype=np.int64)
    min_dist = np.full(n, np.inf, dtype=np.float64)
    tol = env.success_tolerance
    max_steps = env.max_episode_length
    with torch.no_grad():
        for t in range(max_steps):
            action = policy(obs.to(env.device))
            obs, _, _, _ = env.step(action)
            poses.append(env.pose[:, :2].clone().cpu().numpy())
            dist = torch.linalg.norm(env.goal - env.pose[:, :2], dim=-1).cpu().numpy()
            min_dist = np.minimum(min_dist, dist)
            newly = (dist < tol) & (reached_step < 0)
            reached_step[newly] = t + 1
            if np.all(reached_step >= 0):
                break
    return {
        "poses": np.stack(poses, axis=0),          # (T+1, n, 2)
        "goals": goals,                            # (n, 2)
        "reached_step": reached_step,              # (n,)
        "min_dist": min_dist,                      # (n,)
        "tol": tol,
        "step_dt": env.step_dt,
    }


def plot_paths(paths, goals, reached_flags, times_s, min_dists, tol, output, suptitle) -> None:
    """Small-multiples trajectory figure. ``paths`` is a list of (Li, 2) arrays in the
    vehicle start frame (origin = start, +x = initial heading); one panel per goal."""
    n = len(paths)
    ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.5 * nrow), facecolor="#fcfcfb", squeeze=False)
    axes = axes.ravel()
    for i in range(n):
        ax = axes[i]
        ax.set_facecolor("#fcfcfb")
        path = np.asarray(paths[i])
        pts = path.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=TIME_CMAP, norm=plt.Normalize(0, max(len(segs), 1)))
        lc.set_array(np.arange(len(segs)))
        lc.set_linewidth(2.2)
        ax.add_collection(lc)

        gx, gy = goals[i]
        reached = bool(reached_flags[i])
        xs = np.concatenate([path[:, 0], [0, gx]])
        ys = np.concatenate([path[:, 1], [0, gy]])
        span = max(xs.max() - xs.min(), ys.max() - ys.min(), 1.0)
        ax.add_patch(plt.Circle((gx, gy), tol, fill=False, ls=(0, (4, 3)), lw=1.0, ec=MUTED, alpha=0.9))
        ax.scatter([gx], [gy], marker="*", s=200, color=(GOOD if reached else CRIT),
                   edgecolor="white", linewidth=0.8, zorder=5)
        ax.scatter([0], [0], marker="o", s=42, color=INK, zorder=5)
        ax.annotate("", xy=(0.10 * span, 0.0), xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))  # initial heading (+x)

        if reached:
            title = f"G{i}  ✓ {times_s[i]:.1f}s  ({min_dists[i] * 100:.0f} cm)"
            tc = "#006300"
        else:
            title = f"G{i}  ✕ timeout  (min {min_dists[i]:.2f} m)"
            tc = CRIT
        ax.set_title(title, fontsize=10, color=tc, pad=4)

        m = 0.06 * span + 0.5
        ax.set_xlim(xs.min() - m, xs.max() + m)
        ax.set_ylim(ys.min() - m, ys.max() + m)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color=GRID, lw=0.7)
        ax.tick_params(colors=MUTED, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#c3c2b7")
        if i % ncol == 0:
            ax.set_ylabel("y [m]", fontsize=8, color=SECONDARY)
        if i >= (nrow - 1) * ncol:
            ax.set_xlabel("x [m]", fontsize=8, color=SECONDARY)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(suptitle, fontsize=12.5, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=3.0, w_pad=1.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())
    print(f"wrote {output}")


def plot(result: dict, output: Path, checkpoint: Path) -> None:
    poses, goals = result["poses"], result["goals"]
    reached_step, min_dist = result["reached_step"], result["min_dist"]
    tol, step_dt = result["tol"], result["step_dt"]
    n = goals.shape[0]
    paths, reached_flags, times_s = [], [], []
    for i in range(n):
        reached = reached_step[i] >= 0
        end = reached_step[i] if reached else poses.shape[0] - 1
        paths.append(poses[: end + 1, i, :])
        reached_flags.append(reached)
        times_s.append(reached_step[i] * step_dt if reached else float("nan"))
    n_reached = int(np.sum(reached_step >= 0))
    times = reached_step[reached_step >= 0] * step_dt
    mean_time = float(np.mean(times)) if times.size else float("nan")
    suptitle = (
        f"Tracked-vehicle goal reaching · {checkpoint.parent.name}/{checkpoint.name}\n"
        f"{n_reached}/{n} reached (tol {tol:.2f} m) · mean time-to-goal {mean_time:.1f} s · "
        f"path colour = time (light→dark)"
    )
    plot_paths(paths, goals, reached_flags, times_s, min_dist, tol, output, suptitle)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    run_dir = checkpoint.parent
    env_cfg = json.loads((run_dir / "env_cfg.json").read_text())
    train_cfg = json.loads((run_dir / "train_cfg.json").read_text())
    env_cfg.update({"num_envs": int(args.num_goals), "auto_reset": False, "device": args.device})

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    env = TrackedGoalReachingEnv(merge_env_cfg(env_cfg), device=args.device)
    policy = load_policy(env, train_cfg, checkpoint, args.device)
    result = rollout(env, policy)

    # console summary
    print(f"\nGoal-reaching eval — {args.num_goals} goals, tol {result['tol']:.2f} m, dt {result['step_dt']:.2f} s")
    print(f"{'goal':>4} {'gx':>6} {'gy':>6} {'dist':>6} {'reached':>8} {'time_s':>7} {'min_err_m':>9}")
    for i in range(env.num_envs):
        gx, gy = result["goals"][i]
        rs = result["reached_step"][i]
        d0 = float(np.hypot(gx, gy))
        print(f"{i:>4} {gx:>6.2f} {gy:>6.2f} {d0:>6.2f} "
              f"{'yes' if rs >= 0 else 'no':>8} {rs * result['step_dt'] if rs >= 0 else float('nan'):>7.2f} "
              f"{result['min_dist'][i]:>9.3f}")
    n_reached = int(np.sum(result["reached_step"] >= 0))
    print(f"success: {n_reached}/{env.num_envs}")

    output = args.output.resolve() if args.output else (run_dir / "goal_reach_trajectories.png")
    plot(result, output, checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
