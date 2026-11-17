"""G1 open-loop error against time, by family (3 x 4): 1T, 1M, 3T, 3M (seed 61).

Reads ``horizon_errors.npz`` (in ``artifacts/contact_nrd/arch_ablation/g1_locomotion/``, a stored result of the original runs): one open-loop trace per test
episode from its first state, with the robot's position, orientation and joint RMS errors. One row
per error, one column per family (nominal, difficult: 10 s; probe_drop: 1.2 s; probe_landing:
1.5 s), each on its episodes that reach that length, up to it. All four models in every panel. Line: the
median over the episodes; dashed edges with a light shade between: the minimum and maximum over
them. Writes ``horizon_error.png``.

Usage: python scripts/contact_nrd/arch_ablation/g1_locomotion/plot_horizon_families.py
"""
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # layout.py is one folder up
import layout  # noqa: E402

RESULTS = layout.RESULTS / "g1_locomotion"
D = np.load(RESULTS / "horizon_errors.npz")
H = D["horizons_s"]

BLUE, GREEN, RED, PURPLE = "#1f5fb4", "#12936b", "#d1352b", "#8a3fb0"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DEG = 180 / math.pi

SERIES = [  # (key, label, colour)
    ("3t_seed61", "3T: 3 networks, transformer 4 layers x 4 heads, width 128 (2.53 M)", BLUE),
    ("1t_seed61", "1T: 1 network, transformer 6 layers x 8 heads, width 256 (4.83 M)", GREEN),
    ("3m_seed61", "3M: 3 networks, residual MLP 2 blocks, width 256 (2.53 M)", RED),
    ("1m_seed61", "1M: 1 network, residual MLP 2 blocks, width 512, hidden 1986 (4.83 M)",
     PURPLE),
]
PANELS = [  # (metric, title, unit factor, unit)
    ("pos_m", "base position error", 1000.0, "mm"),
    ("angle_rad", "base orientation error", DEG, "deg"),
    ("joint_rms_rad", "joint angle error (RMS over joints)", DEG, "deg"),
]


TICKS_ALL = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 1.2, 1.5, 2, 5, 10]


def style(ax, title, unit, length):
    ticks = [t for t in TICKS_ALL if t <= length + 1e-9 and not (length > 2 and t in (1.2, 1.5))]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlim(0.018, length * 1.1)
    ax.set_ylabel(f"error ({unit}, log scale)", color=MUTED)
    ax.set_title(title, color=INK, loc="left", fontsize=10.5)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.tick_params(which="minor", length=0)


FAMILIES = [str(f) for f in D["families"]]
LENGTH = [float(x) for x in D["length_s"]]
FAMILY = D["3t_seed61/family"]
FULL = D["3t_seed61/full"]

# One row per error, one column per family, all four models in each panel: the median as a thick
# line, the min-max as a light shade with its edges drawn dashed in the model's colour.
fig, axes = plt.subplots(len(PANELS), len(FAMILIES), figsize=(22, 14), dpi=150)
for r, fname in enumerate(FAMILIES):
    sel = (FAMILY == r) & FULL
    n_h = int(sum(h <= LENGTH[r] + 1e-9 for h in H))
    t = H[:n_h]
    for c, (metric, title, factor, unit) in enumerate(PANELS):
        ax = axes[c, r]  # one row per error, one column per family
        for key, label, colour in SERIES:
            e = factor * D[f"{key}/{metric}"][sel][:, :n_h]
            ax.fill_between(t, e.min(0), e.max(0), color=colour, alpha=0.07, linewidth=0)
            for edge in (e.min(0), e.max(0)):
                ax.plot(t, edge, color=colour, linewidth=0.9, linestyle=(0, (3, 2)), alpha=0.9)
        for key, label, colour in SERIES:  # medians on top of every band
            ax.plot(t, np.median(factor * D[f"{key}/{metric}"][sel][:, :n_h], 0), color=colour,
                    linewidth=2.4, label=label)
        style(ax, f"{fname} ({int(sel.sum())} episodes, {LENGTH[r]:g} s)\n{title}", unit,
              LENGTH[r])
        ax.set_xlabel("open-loop time from the episode start (s, log scale)", color=MUTED,
                          fontsize=8.5)
handles, labels = axes[0, 0].get_legend_handles_labels()
handles += [plt.Line2D([], [], color=INK, linewidth=2.4),
            plt.Line2D([], [], color=INK, linewidth=0.9, linestyle=(0, (3, 2)))]
labels += ["median over the episodes", "min and max over the episodes (light shade between)"]
fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK, loc="lower center",
           ncol=3, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("G1 test by family: one open-loop trace per episode from its first state, "
             "to the family's episode length, seed 61", color=INK, x=0.01, ha="left",
             fontsize=12.5)
fig.tight_layout(rect=(0, 0.04, 1, 0.985))
out = RESULTS / "horizon_error.png"
fig.savefig(out, facecolor="white")
print("wrote", out)
