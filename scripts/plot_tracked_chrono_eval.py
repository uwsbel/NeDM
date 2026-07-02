"""Aggregate per-goal Chrono transfer-eval .npz files into one trajectory figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_tracked_rl_goal import plot_paths


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Plot Chrono goal-reach transfer trajectories.")
    p.add_argument("--eval-dir", type=Path, required=True, help="Dir with goal*.npz files.")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--label", type=str, default="Chrono transfer")
    args = p.parse_args(argv)

    files = sorted(args.eval_dir.glob("goal*.npz"))
    if not files:
        raise SystemExit(f"no goal*.npz in {args.eval_dir}")

    paths, goals, reached_flags, times_s, min_dists = [], [], [], [], []
    tol = 0.75
    for f in files:
        d = np.load(f)
        paths.append(d["poses"])
        goals.append(d["goal"])
        rs = int(d["reached_step"])
        reached_flags.append(rs >= 0)
        times_s.append(rs * float(d["step_dt"]) if rs >= 0 else float("nan"))
        min_dists.append(float(d["min_dist"]))
        tol = float(d["tol"])

    goals = np.asarray(goals)
    min_dists = np.asarray(min_dists)
    n = len(files)
    n_reached = int(sum(reached_flags))
    finite_times = [t for t in times_s if t == t]
    mean_time = float(np.mean(finite_times)) if finite_times else float("nan")
    suptitle = (
        f"Tracked-vehicle goal reaching in Chrono (sim-to-sim transfer) · {args.label}\n"
        f"{n_reached}/{n} reached (tol {tol:.2f} m) · mean time-to-goal {mean_time:.1f} s · "
        f"path colour = time (light→dark)"
    )
    output = args.output or (args.eval_dir / "chrono_goal_reach_trajectories.png")
    plot_paths(paths, goals, reached_flags, times_s, min_dists, tol, output, suptitle)
    print(f"{n_reached}/{n} reached; mean time {mean_time:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
