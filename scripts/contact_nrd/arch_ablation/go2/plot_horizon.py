"""Go2 open-loop error against horizon: the 3T, this 1T and the earlier 1T, all seed 8 (README §3.1).

Reads the 10 s scores of ``all_windows.json`` (in ``artifacts/contact_nrd/arch_ablation/go2/fair_1t/``, a stored result of the original runs): the same 4096
validation starts, with 128 rows of history and 1000 rows after. Writes
``horizon_error.png`` in the same folder.

Usage: python scripts/contact_nrd/arch_ablation/go2/plot_horizon.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # layout.py is one folder up
import layout  # noqa: E402

RESULTS = layout.RESULTS / "go2" / "fair_1t"
SCORES = json.loads((RESULTS / "all_windows.json").read_text())
DT = 0.01  # s per model step

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TICKS = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10]

SERIES = [  # (key, label, colour, line style)
    ("3t_best90k", "3T best.pt", BLUE, "-"),
    ("1t_best90k", "this 1T best.pt (6 Nov)", ORANGE, "-"),
    ("1t_nov3_best", "earlier 1T best.pt (3 Nov)", AQUA, "-"),
]


def curve(key):
    nrmse = SCORES[key]["1000"]["nrmse"]
    steps = sorted(int(k) for k in nrmse)
    return [s * DT for s in steps], [nrmse[str(s)] for s in steps]


def style(ax, title):
    ax.set_xscale("log")
    ax.set_xticks(TICKS)
    ax.set_xticklabels([f"{t:g}" for t in TICKS])
    ax.set_xlim(0.009, 11)
    ax.set_xlabel("open-loop horizon (s, log scale)", color=MUTED)
    ax.set_title(title, color=INK, loc="left", fontsize=10.5)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.tick_params(which="minor", length=0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")


def draw(ax, series):
    for key, label, colour, line in series:
        t, e = curve(key)
        ax.plot(t, e, color=colour, linestyle=line, linewidth=2.2, label=label, marker="o",
                markersize=4, markeredgecolor="white", markeredgewidth=1)


fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
draw(ax, SERIES)
ax.set_ylim(0, 0.45)
ax.set_ylabel("NRMSE at that step (lower is better)", color=MUTED)
style(ax, "Go2 rigid validation: error against horizon (4096 matched starts)")
fig.tight_layout()
out = RESULTS / "horizon_error.png"
fig.savefig(out, facecolor="white")
print("wrote", out)
