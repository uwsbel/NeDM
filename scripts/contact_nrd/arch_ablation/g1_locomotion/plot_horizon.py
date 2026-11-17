"""G1 error against horizon: the released 3T and the 1T (seed 61), README.

Reads ``horizon_errors.json`` (in ``artifacts/contact_nrd/arch_ablation/g1_locomotion/fair_1t/``, a stored result of the original runs): window rollouts of the test
episodes, the same windows at every horizon, with the robot's position, orientation and joint RMS
errors at 0.02-5 s. Median over the windows (thick) and p95 (thin). Writes ``horizon_error.png``.

Usage: python scripts/contact_nrd/arch_ablation/g1_locomotion/plot_horizon.py
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # layout.py is one folder up
import layout  # noqa: E402

RESULTS = layout.RESULTS / "g1_locomotion" / "fair_1t"
SCORES = json.loads((RESULTS / "horizon_errors.json").read_text())
H = SCORES["horizons_s"]

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TICKS = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
DEG = 180 / math.pi

SERIES = [  # (key, label, colour)
    ("3t_seed61", "3T, released seed 61", BLUE),
    ("1t_seed61", "1T, seed 61", ORANGE),
]
PANELS = [  # (metric, title, unit factor, unit)
    ("pos_m", "base position error", 1000.0, "mm"),
    ("angle_rad", "base orientation error", DEG, "deg"),
    ("joint_rms_rad", "joint angle error (RMS over joints)", DEG, "deg"),
]


def style(ax, title, unit):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(TICKS)
    ax.set_xticklabels([f"{t:g}" for t in TICKS])
    ax.set_xlim(0.018, 5.6)
    ax.set_xlabel("open-loop horizon (s, log scale)", color=MUTED)
    ax.set_ylabel(f"error ({unit}, log scale)", color=MUTED)
    ax.set_title(title, color=INK, loc="left", fontsize=10.5)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.tick_params(which="minor", length=0)


fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 4.6), dpi=150)
for ax, (metric, title, factor, unit) in zip(axes, PANELS):
    for key, label, colour in SERIES:
        row = SCORES["models"][key][metric]
        ax.plot(H, [factor * v for v in row["median"]], color=colour, linewidth=2.2,
                marker="o", markersize=4, markeredgecolor="white", markeredgewidth=1,
                label=f"{label}, median")
        ax.plot(H, [factor * v for v in row["p95"]], color=colour, linewidth=0.9,
                linestyle=(0, (4, 2)), label=f"{label}, p95")
    style(ax, title, unit)
axes[-1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left",
                bbox_to_anchor=(1.02, 1))
m = SCORES["models"]["3t_seed61"]
fig.suptitle(f"G1 test: window rollouts ({m['windows']} windows from {m['episodes']} episodes, "
             "the same at every horizon)", color=INK, x=0.01, ha="left", fontsize=11.5)
fig.tight_layout()
out = RESULTS / "horizon_error.png"
fig.savefig(out, facecolor="white")
print("wrote", out)
