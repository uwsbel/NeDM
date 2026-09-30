#!/usr/bin/env python3
"""Tracked-base goal-reaching: stress battery + consecutive-waypoint route.

Two-panel figure (side by side with --layout wide, the default, for the
single-column manuscript; stacked with --layout stacked) of the tracked drive policy under the
frozen NN-ROM, exercised in high-fidelity Chrono:

  (a) all 100 driven XY paths from the seeded 100-goal stress battery (origin to
      goal), overlaid in one colour;
  (b) a single figure-8 route in which the *same* policy is fed eight consecutive
      goals; the driven path is one continuous run and each goal is numbered by
      its order in the sequence.

Writes to the manuscript image archive.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "artifacts/rl_runs/tracked_goal_v2_far_rollsel_rom_20260721"
BENCH = RUN / "chrono_benchmark_N100_seed12345"
BOWTIE = RUN / "chrono_waypoints_fig8_bowtie" / "traj.npz"
DEFAULT_OUT = Path("/home/harry/Manuscripts/ImageArchive/journals/2026/"
                   "neural-dynamics-model/tracked_stress_trajectories.png")

PATH_C, GOAL_C, START_C = "#1f77b4", "#d62728", "black"

# Label offsets (data units) so each waypoint number sits clear of the path.
# Goals 4 and 8 both return to the origin, so their labels straddle the centre.
BOWTIE_LABEL_OFF = {
    1: (1.3, 1.2), 2: (1.5, 1.0), 3: (1.3, -2.0), 4: (-3.4, 1.4),
    5: (-2.8, 1.0), 6: (-3.8, 1.0), 7: (-2.8, -2.0), 8: (1.6, -2.8),
}


def panel_stress(ax) -> None:
    files = sorted(BENCH.glob("goal_*.npz"))
    for f in files:
        d = np.load(f)
        poses, goal = d["poses"], d["goal"]
        ax.plot(poses[:, 0], poses[:, 1], color=PATH_C, lw=0.7, alpha=0.4)
        ax.plot(goal[0], goal[1], marker="o", color=GOAL_C, ms=2.8, zorder=5)
    ax.plot(0, 0, marker="o", color=START_C, ms=6, zorder=6)

    handles = [
        Line2D([0], [0], color=PATH_C, lw=1.4, alpha=0.7, label="driven path"),
        Line2D([0], [0], marker="o", color=START_C, lw=0, ms=6, label="start (origin)"),
        Line2D([0], [0], marker="o", color=GOAL_C, lw=0, ms=5, label="goal"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, frameon=True,
              framealpha=0.9, borderpad=0.35)
    _style(ax, "(a)")


def panel_bowtie(ax) -> None:
    d = np.load(BOWTIE)
    poses = d["poses"]
    wps = d["waypoints"]           # (9,2): index 0 is the start, 1..8 the goals
    ax.plot(poses[:, 0], poses[:, 1], color=PATH_C, lw=1.4, alpha=0.9, zorder=4)

    for k in range(1, 9):
        gx, gy = wps[k]
        ax.plot(gx, gy, marker="o", color=GOAL_C, ms=6, zorder=5)
        dx, dy = BOWTIE_LABEL_OFF[k]
        ax.annotate(str(k), (gx, gy), (gx + dx, gy + dy), fontsize=9,
                    fontweight="bold", color="black", zorder=7)
    ax.plot(0, 0, marker="o", color=START_C, ms=7, zorder=6)
    _style(ax, "(b)")


def _style(ax, tag: str) -> None:
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.98, 0.96, tag, transform=ax.transAxes, ha="right", va="top",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--layout", choices=("wide", "stacked"), default="wide")
    args = ap.parse_args()

    if args.layout == "wide":
        fig, (ax0, ax1) = plt.subplots(
            1, 2, figsize=(6.8, 3.2),
            gridspec_kw={"width_ratios": [1.0, 1.25]})
    else:
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(3.5, 5.9),
            gridspec_kw={"height_ratios": [1.0, 0.6]})
    panel_stress(ax0)
    panel_bowtie(ax1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
