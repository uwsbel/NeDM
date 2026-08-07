#!/usr/bin/env python3
"""PPO convergence figure for the two Study Case II goal-reaching policies.

Two stacked panels, one per task:
  (a) tracked-base planar goal reaching (run tracked_goal_v2_far_rollsel_rom_20260721)
  (b) arm end-effector reaching (run arm_reach_..._8d_rom_20260727)

Both panels read the run that was actually transferred to Chrono, so the reward
curve and the reported transfer result come from the same policy.

Each panel shows the mean episode reward vs PPO iteration, read from the rsl-rl
tensorboard logs, with a wall-clock top axis. Styled to match the Study Case I
RL-reward figure (Fig. 4, hmmwv_rl_reward.pdf): faint raw traces, EMA-smoothed
solid curves, a dashed transfer-checkpoint marker, and matching fonts/labels.
Writes to the manuscript image archive.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKED_EV = REPO_ROOT / ("artifacts/rl_runs/tracked_goal_v2_far_rollsel_rom_20260721/"
    "events.out.tfevents.1784672662.newton.2341918.0")
ARM_EV = REPO_ROOT / ("artifacts/rl_runs/"
    "arm_reach_adaptivekl005_lr1e4_tol005_ep150_bonus150_sigma015_8d_rom_20260727/"
    "events.out.tfevents.1785176587.newton.3199792.0")
DEFAULT_OUT = Path("/home/harry/Manuscripts/ImageArchive/journals/2026/neural-dynamics-model")

REWARD_C = "C0"


def series(ea: EventAccumulator, tag: str):
    ev = ea.Scalars(tag)
    return (np.array([e.step for e in ev], float),
            np.array([e.value for e in ev], float),
            np.array([e.wall_time for e in ev], float))


def smooth(values: np.ndarray, weight: float = 0.9) -> np.ndarray:
    """EMA smoothing matching the Study Case I RL-reward figure."""
    out = np.empty_like(values)
    last = values[0]
    for i, v in enumerate(values):
        last = last * weight + (1 - weight) * v
        out[i] = last
    return out


def panel(ax, ev_path: Path, tag: str, ckpt_iter: int) -> None:
    ea = EventAccumulator(str(ev_path), size_guidance={"scalars": 0})
    ea.Reload()
    rs, rv, rw = series(ea, "Train/mean_reward")
    wall_min = (rw - rw[0]) / 60.0
    rate = wall_min[-1] / rs[-1]  # minutes of wall-clock per PPO iteration

    ax.plot(rs, rv, color=REWARD_C, alpha=0.25, lw=0.8)
    ax.plot(rs, smooth(rv), color=REWARD_C, lw=1.8)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Mean episode reward")
    ax.set_xlim(0, rs.max() * 1.02)
    ax.grid(True, which="major", axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Transfer checkpoint: dashed marker + annotation, matching Fig. 4.
    ax.axvline(ckpt_iter, color="black", ls="--", lw=1.2)
    y0, y1 = ax.get_ylim()
    ax.annotate(
        f"transfer\ncheckpoint\n({ckpt_iter} it.)",
        xy=(ckpt_iter, y0 + 0.5 * (y1 - y0)),
        xytext=(ckpt_iter - 0.06 * rs.max(), y0 + 0.5 * (y1 - y0)),
        ha="right", va="center", fontsize=8.5,
        arrowprops=dict(arrowstyle="-", color="black", lw=0.8, shrinkA=0, shrinkB=2),
    )

    # Top axis: convert PPO iteration to wall-clock training time for this run.
    top_ax = ax.secondary_xaxis("top", functions=(lambda it: it * rate, lambda mn: mn / rate))
    top_ax.set_xlabel("Wall-clock time (min)")

    # Panel tag, inside the axes so it does not collide with the time axis.
    ax.text(0.015, 0.93, tag, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(4.8, 5.6))
    # Both runs' final checkpoint is model_1499.pt -- that is what transferred.
    panel(ax_a, TRACKED_EV, "(a)", 1499)
    panel(ax_b, ARM_EV, "(b)", 1499)

    fig.tight_layout(h_pad=2.2)
    png = out_dir / "tracked_arm_rl_reward.png"
    pdf = out_dir / "tracked_arm_rl_reward.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    print(f"wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
