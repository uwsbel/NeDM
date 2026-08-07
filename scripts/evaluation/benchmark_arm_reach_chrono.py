"""Seeded Chrono transfer benchmark for the arm-reaching goal policy.

The arm analog of ``scripts/evaluation/benchmark_tracked_goal_chrono.py``. Runs a *battery*
of N seeded reaching goals through the single-goal Chrono eval
(``scripts/evaluation/eval_arm_rl_chrono_reaching.py --goal X Y Z``) -- **one fresh process
per goal** -- then aggregates a summary.json + trajectory plot.

Why one process per goal: the arm Chrono env's ``reset_idx`` destroys and
rebuilds the whole M113+arm scene each goal, and repeated in-process sim
re-creation has stack-smashed in past eval loops (see the
``chrono-eval-multiref-stack-smash`` note). Isolating each goal in its own
process is the safe pattern (identical to the tracked benchmark).

Goals are precomputed here (before spawning any Chrono process) by sampling the
*same* joint-space region the policy trained on -- ``goal.q_lo``/``goal.q_hi``
read from the run's ``env_cfg.json`` -- via ``ArmReachingChronoEnv``'s own
``_sample_safe_goals`` (FK + safety filter, no Chrono sim needed thanks to
``defer_reset=True``). That yields a reproducible, inspectable EE goal set in the
arm base frame, written to ``goals.json``.

Example:
    PYTHONPATH=src python scripts/evaluation/benchmark_arm_reach_chrono.py \
        --run-dir artifacts/rl_runs/<run> \
        --num-goals 100 --seed 12345
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_arm_rl_chrono_reaching.py"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Seeded Chrono transfer benchmark (arm reaching).")
    p.add_argument("--run-dir", type=Path, required=True, help="RSL-RL arm run dir (env_cfg.json/train_cfg.json).")
    p.add_argument("--policy-checkpoint", type=Path, default=None, help="Defaults to latest model_*.pt in run-dir.")
    p.add_argument("--num-goals", type=int, default=100)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--max-steps", type=int, default=None, help="Override per-goal episode budget; default env_cfg.")
    p.add_argument("--pre-roll-time-s", type=float, default=None, help="Scene settle before each reach; default env (6s).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default <run_dir>/chrono_reach_benchmark_N{n}_seed{seed}.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--limit", type=int, default=None, help="Only run the first K sampled goals (smoke test).")
    return p.parse_args(argv)


def sample_goals(run_dir: Path, n: int, seed: int, device: str) -> tuple[np.ndarray, dict]:
    """Sample n safe EE goals from the run's trained joint-space region (base frame)."""
    import torch

    from nedm.rl.arm_reaching_chrono_env import ArmReachingChronoEnv

    env_cfg = json.loads((run_dir / "env_cfg.json").read_text())
    env_cfg.update({"num_envs": 1, "device": device, "defer_reset": True, "render": False})

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    env = ArmReachingChronoEnv(env_cfg, device=device)
    goals = env._sample_safe_goals(int(n)).detach().cpu().numpy().astype(np.float32)
    meta = {
        "q_lo": [float(v) for v in env.goal_q_lo.detach().cpu().numpy()],
        "q_hi": [float(v) for v in env.goal_q_hi.detach().cpu().numpy()],
        "success_tolerance_m": float(env.cfg["reward"]["success_tolerance_m"]),
        "max_episode_steps": int(env.max_episode_length),
        "control_dt_s": float(env.dt_s) * int(env.action_repeat),
    }
    return goals, meta


def path_length(ee: np.ndarray) -> float:
    if len(ee) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(ee, axis=0), axis=1)))


def aggregate(goal_dirs: list[Path], goals: np.ndarray, tol: float, control_dt: float) -> dict:
    per_goal = []
    for i, (gd, goal) in enumerate(zip(goal_dirs, goals)):
        npz = gd / "chrono_arm_reach_00.npz"
        summ = gd / "summary.json"
        if not npz.exists() or not summ.exists():
            per_goal.append({"index": i, "goal_base": goal.tolist(), "status": "MISSING"})
            continue
        d = np.load(npz)
        err = d["error_m"].astype(np.float64)
        ee = d["ee_base"].astype(np.float64)
        first_success_step = int(d["first_success_step"]) if "first_success_step" in d else -1
        row = json.loads(summ.read_text())["rollouts"][0]
        min_err = float(np.min(err))
        reached = bool(row.get("success")) or (first_success_step >= 0) or (min_err < tol)
        per_goal.append({
            "index": i,
            "goal_base": [float(v) for v in goal],
            "reached": reached,
            "success": bool(row.get("success")),
            "reached_tolerance": bool(row.get("reached_tolerance", min_err < tol)),
            "done_reason": row.get("done_reason"),
            "final_ee_error_m": float(err[-1]),
            "min_ee_error_m": min_err,
            "first_success_step": first_success_step,
            "time_to_success_s": (first_success_step * control_dt) if first_success_step >= 0 else None,
            "num_steps": int(len(err) - 1),
            "ee_path_len_m": path_length(ee),
            "contact_kind": row.get("contact_kind"),
            "joint_limit_labels": row.get("joint_limit_labels"),
        })

    valid = [g for g in per_goal if g.get("status") != "MISSING"]
    reached = [g for g in valid if g.get("reached")]
    n_total = len(per_goal)

    def med(vals):
        vals = [v for v in vals if v is not None]
        return float(np.median(vals)) if vals else None

    def worst(vals):
        vals = [v for v in vals if v is not None]
        return float(np.max(vals)) if vals else None

    # tally hard-failure reasons among non-reached goals
    reasons: dict[str, int] = {}
    for g in valid:
        if not g.get("reached"):
            reasons[str(g.get("done_reason"))] = reasons.get(str(g.get("done_reason")), 0) + 1

    return {
        "benchmark": "arm_reaching_chrono",
        "num_goals": n_total,
        "num_valid": len(valid),
        "num_reached": len(reached),
        "success_rate": (len(reached) / len(valid)) if valid else 0.0,
        "success_tolerance_m": tol,
        "median_min_ee_error_m": med([g.get("min_ee_error_m") for g in valid]),
        "worst_min_ee_error_m": worst([g.get("min_ee_error_m") for g in valid]),
        "median_final_ee_error_m": med([g.get("final_ee_error_m") for g in valid]),
        "worst_final_ee_error_m": worst([g.get("final_ee_error_m") for g in valid]),
        "median_time_to_success_s": med([g.get("time_to_success_s") for g in reached]),
        "worst_time_to_success_s": worst([g.get("time_to_success_s") for g in reached]),
        "failure_reasons": reasons,
        "goals": per_goal,
    }


def plot(goal_dirs: list[Path], goals: np.ndarray, out_png: Path, tol: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[plot] skipped ({exc})")
        return
    fig = plt.figure(figsize=(12, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_err = fig.add_subplot(1, 2, 2)
    for gd in goal_dirs:
        npz = gd / "chrono_arm_reach_00.npz"
        if not npz.exists():
            continue
        d = np.load(npz)
        ee = d["ee_base"]
        goal = d["goal_base"]
        err = d["error_m"]
        reached = bool(np.min(err) < tol)
        color = "tab:green" if reached else "tab:red"
        ax3d.plot(ee[:, 0], ee[:, 1], ee[:, 2], "-", color=color, alpha=0.5, lw=0.8)
        ax3d.scatter(goal[0], goal[1], goal[2], color=color, s=12)
        ax_err.plot(np.arange(len(err)), err, "-", color=color, alpha=0.4, lw=0.7)
    ax3d.set_xlabel("x base (m)")
    ax3d.set_ylabel("y base (m)")
    ax3d.set_zlabel("z base (m)")
    ax3d.set_title("Arm reaching: Chrono transfer battery")
    ax_err.axhline(tol, color="k", ls="--", lw=1.0, label=f"tol {tol} m")
    ax_err.set_xlabel("control step")
    ax_err.set_ylabel("EE error (m)")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[plot] -> {out_png}")


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    goals, meta = sample_goals(run_dir, args.num_goals, args.seed, args.device)
    if args.limit is not None:
        goals = goals[: args.limit]

    tol = meta["success_tolerance_m"]
    control_dt = meta["control_dt_s"]
    out_dir = args.output_dir or (run_dir / f"chrono_reach_benchmark_N{len(goals)}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "goals.json").write_text(json.dumps({
        "seed": args.seed, "num_goals": len(goals), **meta,
        "goal_bounds_base": {
            "min": [float(v) for v in goals.min(axis=0)],
            "max": [float(v) for v in goals.max(axis=0)],
        },
        "goals_base": goals.tolist(),
    }, indent=2))

    print(f"[benchmark] {len(goals)} arm goals, seed={args.seed}, tol={tol}m -> {out_dir}")
    goal_dirs = []
    t0 = time.time()
    for i, goal in enumerate(goals):
        gd = out_dir / f"goal_{i:03d}"
        goal_dirs.append(gd)
        if (gd / "summary.json").exists() and (gd / "chrono_arm_reach_00.npz").exists():
            print(f"[{i+1}/{len(goals)}] exists, skipping {gd.name}")
            continue
        gd.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--run-dir", str(run_dir),
            "--goal", f"{goal[0]:.5f}", f"{goal[1]:.5f}", f"{goal[2]:.5f}",
            "--output-dir", str(gd),
            "--no-plots",
            "--device", args.device,
            "--seed", str(args.seed),
        ]
        if args.policy_checkpoint is not None:
            cmd += ["--policy-checkpoint", str(args.policy_checkpoint.resolve())]
        if args.max_steps is not None:
            cmd += ["--max-steps", str(args.max_steps)]
        if args.pre_roll_time_s is not None:
            cmd += ["--pre-roll-time-s", str(args.pre_roll_time_s)]
        gt = time.time()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        print(f"[{i+1}/{len(goals)}] goal=({goal[0]:.2f},{goal[1]:.2f},{goal[2]:.2f}) "
              f"rc={proc.returncode} wall={time.time()-gt:.0f}s")

    summary = aggregate(goal_dirs, goals, tol, control_dt)
    summary["seed"] = args.seed
    summary["policy_checkpoint"] = str(args.policy_checkpoint) if args.policy_checkpoint else "latest"
    summary["total_wall_s"] = round(time.time() - t0, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(goal_dirs, goals, out_dir / "benchmark_trajectories.png", float(tol))

    print(f"\n[benchmark] success {summary['num_reached']}/{summary['num_valid']} "
          f"= {summary['success_rate']*100:.0f}%  "
          f"median_min_err={summary['median_min_ee_error_m']}  "
          f"worst_min_err={summary['worst_min_ee_error_m']}  "
          f"failures={summary['failure_reasons']}")
    print(f"[benchmark] -> {out_dir/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
