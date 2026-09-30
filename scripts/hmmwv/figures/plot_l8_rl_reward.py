#!/usr/bin/env python3
"""PPO training-convergence figure for the three L8-backbone RL policies.

Replaces the manuscript's hmmwv_rl_reward.pdf (built for the 6-layer/500-iter
run). Reads Train/mean_reward from each run's TensorBoard event file and marks
the iteration-1000 evaluation checkpoint (not 500).

Usage:
    python plot_l8_rl_reward.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = Path(__file__).resolve().parents[2]

RUNS = {
    "Mixture generalist": (
        "C0",
        "artifacts/rl_runs/hmmwv_rl_15d_crm2000mix25_onehot_ofatL8_bestval51_flat20crm20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010",
    ),
    "Rigid-only": (
        "C1",
        "artifacts/rl_runs/hmmwv_rl_15d_crm2000mix00_onehot_ofatL8_bestval61_rigid20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010_1000it",
    ),
    "CRM-only": (
        "C2",
        "artifacts/rl_runs/hmmwv_rl_15d_crm2000mix100_onehot_ofatL8_bestval54_crmonly20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010_1000it",
    ),
}

TAG = "Train/mean_reward"
EVAL_ITER = 1000


def smooth(values: np.ndarray, weight: float = 0.9) -> np.ndarray:
    out = np.empty_like(values)
    last = values[0]
    for i, v in enumerate(values):
        last = last * weight + (1 - weight) * v
        out[i] = last
    return out


def load_run(run_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matches = glob.glob(str(REPO_ROOT / run_dir / "**" / "events.out.tfevents.*"), recursive=True)
    ea = EventAccumulator(matches[0], size_guidance={"scalars": 0})
    ea.Reload()
    scalars = ea.Scalars(TAG)
    steps = np.array([s.step for s in scalars], dtype=float)
    vals = np.array([s.value for s in scalars], dtype=float)
    wall = np.array([s.wall_time for s in scalars], dtype=float)
    wall = (wall - wall[0]) / 60.0  # elapsed minutes
    return steps, vals, wall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts/analysis/manuscript_figs"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    min_per_iter = []
    for label, (color, run_dir) in RUNS.items():
        steps, vals, wall_min = load_run(run_dir)
        keep = steps <= EVAL_ITER
        steps, vals, wall_min = steps[keep], vals[keep], wall_min[keep]
        min_per_iter.append(wall_min[-1] / steps[-1])
        ax.plot(steps, vals, color=color, alpha=0.25, lw=0.8)
        ax.plot(steps, smooth(vals), color=color, lw=1.8, label=label)

    # All three policies train inside the same batched NN-ROM at the same PPO
    # config, so wall-clock throughput is essentially identical across runs
    # (~16 it/min on a single RTX 4090); one shared linear map is accurate for
    # all three curves, not just representative of one.
    rate = float(np.mean(min_per_iter))  # minutes per iteration

    ax.axvline(EVAL_ITER, color="black", ls="--", lw=1.2)
    ax.annotate(
        "evaluated\ncheckpoint\n(1000 it.)",
        xy=(EVAL_ITER, ax.get_ylim()[1]), xytext=(EVAL_ITER - 60, ax.get_ylim()[1] * 0.55),
        ha="right", va="center", fontsize=8.5,
        arrowprops=dict(arrowstyle="-", color="black", lw=0.8, shrinkA=0, shrinkB=2),
    )
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Mean episode reward")
    ax.set_xlim(0, EVAL_ITER * 1.02)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(True, which="major", axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    top_ax = ax.secondary_xaxis("top", functions=(lambda it: it * rate, lambda mn: mn / rate))
    top_ax.set_xlabel("Wall-clock time (min, RTX 4090)")

    fig.tight_layout()
    png_path = out_dir / "hmmwv_rl_reward_L8.png"
    pdf_path = out_dir / "hmmwv_rl_reward_L8.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}  (rate={rate:.5f} min/iter, {1/rate:.2f} it/min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
