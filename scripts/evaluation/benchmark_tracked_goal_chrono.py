"""Seeded Chrono transfer benchmark for the tracked-base goal-reaching policy.

Runs a *battery* of N seeded goals through the single-goal transfer test
(``scripts/evaluation/eval_tracked_rl_goal_chrono.py``) -- one fresh process per goal, as
that script requires -- then aggregates a summary.json + trajectory plot. This
turns the previous 4-goal spot check into a reproducible benchmark mirroring the
arm reaching battery (success rate, final/closest error, time-to-success, path
efficiency).

Goals are sampled from the same region the policy trained on, read from the
run's ``env_cfg.json`` (radius_m, angle_rad), in the vehicle start frame.

Example:
    python scripts/evaluation/benchmark_tracked_goal_chrono.py \
        --checkpoint artifacts/rl_runs/tracked_goal_v2_far/model_1500.pt \
        --num-goals 30 --seed 12345 --max-steps 600
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_tracked_rl_goal_chrono.py"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Seeded Chrono transfer benchmark (tracked goal reaching).")
    p.add_argument("--checkpoint", type=Path, required=True, help="Policy checkpoint (e.g. model_1500.pt).")
    p.add_argument("--num-goals", type=int, default=30)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--radius-min", type=float, default=None, help="Override goal radius min (m); default from env_cfg.")
    p.add_argument("--radius-max", type=float, default=None, help="Override goal radius max (m); default from env_cfg.")
    p.add_argument("--max-steps", type=int, default=600, help="Max policy steps per goal (each = 0.1 s sim).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write per-goal npz + summary; default <run_dir>/chrono_benchmark_N{n}_seed{seed}.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--limit", type=int, default=None, help="Only run the first K sampled goals (smoke test).")
    return p.parse_args(argv)


def sample_goals(n: int, seed: int, r_lo: float, r_hi: float, a_lo: float, a_hi: float) -> np.ndarray:
    """Sample n goals uniformly in the (radius, angle) region, in the start frame."""
    rng = np.random.default_rng(seed)
    radius = rng.uniform(r_lo, r_hi, size=n)
    angle = rng.uniform(a_lo, a_hi, size=n)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1).astype(np.float32)


def path_length(poses: np.ndarray) -> float:
    if len(poses) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(poses, axis=0), axis=1)))


def aggregate(goal_files: list[Path], goals: np.ndarray) -> dict:
    per_goal = []
    for i, (gf, goal) in enumerate(zip(goal_files, goals)):
        if not gf.exists():
            per_goal.append({"index": i, "goal_local": goal.tolist(), "status": "MISSING"})
            continue
        d = np.load(gf)
        poses = d["poses"].astype(np.float64)
        g = d["goal"].astype(np.float64)
        reached_step = int(d["reached_step"])
        step_dt = float(d["step_dt"])
        tol = float(d["tol"])
        min_dist = float(d["min_dist"])
        final_dist = float(np.linalg.norm(g - poses[-1]))
        straight = float(np.linalg.norm(g))            # start (origin) -> goal
        driven = path_length(poses)
        reached = reached_step >= 0
        per_goal.append({
            "index": i,
            "goal_local": [float(g[0]), float(g[1])],
            "goal_range_m": straight,
            "reached": reached,
            "reached_step": reached_step,
            "time_to_success_s": (reached_step * step_dt) if reached else None,
            "final_dist_m": final_dist,
            "min_dist_m": min_dist,
            "tol_m": tol,
            "path_length_m": driven,
            "path_efficiency": (straight / driven) if driven > 1e-6 else None,
            "num_steps": int(len(poses) - 1),
        })

    reached = [g for g in per_goal if g.get("reached")]
    n_total = len(per_goal)
    n_reached = len(reached)

    def med(vals):
        vals = [v for v in vals if v is not None]
        return float(np.median(vals)) if vals else None

    def worst(vals):
        vals = [v for v in vals if v is not None]
        return float(np.max(vals)) if vals else None

    return {
        "benchmark": "tracked_goal_reaching_chrono",
        "num_goals": n_total,
        "num_reached": n_reached,
        "success_rate": (n_reached / n_total) if n_total else 0.0,
        "success_tolerance_m": per_goal[0].get("tol_m") if per_goal else None,
        # closest-approach (== success criterion) stats over all goals
        "median_min_dist_m": med([g.get("min_dist_m") for g in per_goal]),
        "worst_min_dist_m": worst([g.get("min_dist_m") for g in per_goal]),
        # final-pose error over all goals
        "median_final_dist_m": med([g.get("final_dist_m") for g in per_goal]),
        "worst_final_dist_m": worst([g.get("final_dist_m") for g in per_goal]),
        # timing / efficiency over successful goals only
        "median_time_to_success_s": med([g.get("time_to_success_s") for g in reached]),
        "worst_time_to_success_s": worst([g.get("time_to_success_s") for g in reached]),
        "median_path_efficiency": med([g.get("path_efficiency") for g in reached]),
        "goals": per_goal,
    }


def plot(goal_files: list[Path], out_png: Path, tol: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[plot] skipped ({exc})")
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    for gf in goal_files:
        if not gf.exists():
            continue
        d = np.load(gf)
        poses = d["poses"]
        g = d["goal"]
        reached = int(d["reached_step"]) >= 0
        color = "tab:green" if reached else "tab:red"
        ax.plot(poses[:, 0], poses[:, 1], "-", color=color, alpha=0.6, lw=1.0)
        ax.plot(g[0], g[1], "o", color=color, ms=5)
        ax.add_patch(plt.Circle((g[0], g[1]), tol, color=color, fill=False, alpha=0.4, lw=0.7))
    ax.plot(0, 0, "ks", ms=8, label="start")
    ax.set_aspect("equal")
    ax.set_xlabel("x (start frame, m)")
    ax.set_ylabel("y (start frame, m)")
    ax.set_title("Tracked goal-reaching: Chrono transfer battery")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[plot] -> {out_png}")


def main(argv=None) -> int:
    args = parse_args(argv)
    checkpoint = args.checkpoint.resolve()
    run_dir = checkpoint.parent
    env_cfg = json.loads((run_dir / "env_cfg.json").read_text())
    goal_cfg = env_cfg.get("goal", {})
    r_lo, r_hi = goal_cfg.get("radius_m", [20.0, 40.0])
    a_lo, a_hi = goal_cfg.get("angle_rad", [-math.pi, math.pi])
    if args.radius_min is not None:
        r_lo = args.radius_min
    if args.radius_max is not None:
        r_hi = args.radius_max

    goals = sample_goals(args.num_goals, args.seed, r_lo, r_hi, a_lo, a_hi)
    if args.limit is not None:
        goals = goals[: args.limit]

    out_dir = args.output_dir or (run_dir / f"chrono_benchmark_N{len(goals)}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "goals.json").write_text(json.dumps({
        "seed": args.seed, "num_goals": len(goals),
        "radius_m": [r_lo, r_hi], "angle_rad": [a_lo, a_hi],
        "goals_local": goals.tolist(),
    }, indent=2))

    print(f"[benchmark] {len(goals)} goals, seed={args.seed}, radius=[{r_lo},{r_hi}]m -> {out_dir}")
    goal_files = []
    t0 = time.time()
    for i, (gx, gy) in enumerate(goals):
        gf = out_dir / f"goal_{i:02d}.npz"
        goal_files.append(gf)
        if gf.exists():
            print(f"[{i+1}/{len(goals)}] exists, skipping {gf.name}")
            continue
        cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--checkpoint", str(checkpoint),
            "--goal-x", f"{gx:.4f}", "--goal-y", f"{gy:.4f}",
            "--max-steps", str(args.max_steps),
            "--output", str(gf),
            "--device", args.device,
        ]
        gt = time.time()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        print(f"[{i+1}/{len(goals)}] goal=({gx:.1f},{gy:.1f}) rc={proc.returncode} wall={time.time()-gt:.0f}s")

    summary = aggregate(goal_files, goals)
    summary["seed"] = args.seed
    summary["radius_m"] = [r_lo, r_hi]
    summary["total_wall_s"] = round(time.time() - t0, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    tol = summary.get("success_tolerance_m") or 0.75
    plot(goal_files, out_dir / "benchmark_trajectories.png", float(tol))

    print(f"\n[benchmark] success {summary['num_reached']}/{summary['num_goals']} "
          f"= {summary['success_rate']*100:.0f}%  "
          f"median_min_dist={summary['median_min_dist_m']:.3f}m  "
          f"worst_min_dist={summary['worst_min_dist_m']:.3f}m  "
          f"median_t2s={summary['median_time_to_success_s']}s  "
          f"median_path_eff={summary['median_path_efficiency']}")
    print(f"[benchmark] -> {out_dir/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
