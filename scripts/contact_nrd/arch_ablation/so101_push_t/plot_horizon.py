"""SO101 push-T error against horizon: the released 3T and the ablation's 3M, 1T and 1M (seed 61).

Reads ``horizon_errors.json`` (in ``artifacts/contact_nrd/arch_ablation/so101_push_t/``, a stored result of the original runs): free rollouts of the 1 000
test shots from their first state, with the T's position and angle errors at every model step up to
the target step (4 s), on log axes. Median over the shots (line) and min to max (shaded); the grey
band is the window of the arm's first touch of the T. Writes ``horizon_error.png``.

Usage: python scripts/contact_nrd/arch_ablation/so101_push_t/plot_horizon.py
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

RESULTS = layout.RESULTS / "so101_push_t"
SCORES = json.loads((RESULTS / "horizon_errors.json").read_text())
DT = SCORES["dt"]

BLUE, GREEN, ORANGE, PURPLE = "#2a78d6", "#2e9e5b", "#eb6834", "#8a4fc7"
INK, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#d6d4ce"
FLOOR = 1e-6  # errors of exactly 0 (float round-off) are drawn at the floor of the log axis
TICKS = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4]

SERIES = [  # (key, label, colour)
    ("3t", "3T: 3 networks, transformer (released)", BLUE),
    ("3m", "3M: 3 networks, MLP", GREEN),
    ("1t", "1T: 1 network, transformer", ORANGE),
    ("1m", "1M: 1 network, MLP", PURPLE),
]
PANELS = [("pos_m", "T position error", "mm", 1000.0), ("angle_rad", "T angle error", "deg", 180 / math.pi)]


def style(ax, title, unit):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(TICKS)
    ax.set_xticklabels([f"{t:g}" for t in TICKS])
    ax.set_xlim(0.018, 4.4)
    ax.set_xlabel("open-loop horizon (s, log scale)", color=MUTED)
    ax.set_ylabel(f"error ({unit}, log scale)", color=MUTED)
    ax.set_title(title, color=INK, loc="left", fontsize=10.5, pad=28)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.tick_params(which="minor", length=0)


fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), dpi=150)
win = SCORES["contacts_s"]["arm_tshape"]
for ax, (field, title, unit, scale) in zip(axes, PANELS):
    ax.axvspan(win["min"], win["max"], color=BAND, alpha=0.6, linewidth=0, zorder=0)
    ax.text(win["median"], 1.01, f"arm first touches the T\n({win['shots']} of {SCORES['shots']} shots)",
            color=MUTED, fontsize=8, ha="center", va="bottom", transform=ax.get_xaxis_transform())
    for key, label, colour in SERIES:
        row = SCORES["models"][key][field]
        t = [DT * (i + 1) for i in range(len(row["median"]))]
        lo, hi, mid = ([max(scale * v, FLOOR) for v in row[q]] for q in ("min", "max", "median"))
        ax.fill_between(t, lo, hi,
                        color=colour, alpha=0.13, linewidth=0)
        ax.plot(t, mid, color=colour, linewidth=2.0, label=label)
    ax.set_ylim(bottom=FLOOR / 2)
    style(ax, f"{title}, free rollouts of {SCORES['shots']} shots", unit)
axes[1].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.01, 1),
               title="line: median, shade: min to max", title_fontsize=8)
fig.tight_layout()
out = RESULTS / "horizon_error.png"
fig.savefig(out, facecolor="white")
print("wrote", out)
