"""Ball error against horizon: the released 3T, the 1T of 6 layers x 8 heads, width 256, and the
MLP variants 3M and 1M (``mlp.py``), all seed 61, README.

Reads ``horizon_errors.json`` (in ``artifacts/contact_nrd/arch_ablation/bouncing_ball/``, a stored result of the original runs): free rollouts of the 1,800
test shots from their first state, as the study's test, with the ball's errors at every model step
up to 1.7 s, on a log time axis. Median over the shots (line) and min to max (shaded); the
grey bands are the windows of the first contact (ball-floor) and the second (ball-wall).
Writes ``horizon_error.png``.

Usage: python scripts/contact_nrd/arch_ablation/bouncing_ball/plot_horizon.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # layout.py is one folder up
import layout  # noqa: E402

RESULTS = layout.RESULTS / "bouncing_ball"
SCORES = json.loads((RESULTS / "horizon_errors.json").read_text())
DT = SCORES["dt"]

BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#2e9e5b", "#8a4fc7"
INK, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#d6d4ce"
TICKS = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 1.7]

SERIES = [  # (key, label, colour)
    ("3t_seed61", "3 networks, transformer 4 layers x 4 heads, width 128\n(released, seed 61)", BLUE),
    ("1t_6L8H256_seed61", "1 network, transformer 6 layers x 8 heads, width 256\n(4.80 M, seed 61)", ORANGE),
    ("3m_seed61", "3 networks, MLP trunks 4 blocks, width 128\n(3M, seed 61)", GREEN),
    ("1m_seed61", "1 network, MLP trunk 4 blocks, width 128\n(1M, core alone, seed 61)", PURPLE),
]
CONTACTS = [("ball_floor", "1st contact\n(floor)"), ("ball_wall", "2nd contact\n(wall)")]


def style(ax, title, unit):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(TICKS)
    ax.set_xticklabels([f"{t:g}" for t in TICKS])
    ax.set_xlim(0.018, 1.9)
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


fig, ax = plt.subplots(figsize=(12, 5.6), dpi=150)
for pair, text in CONTACTS:
    win = SCORES["contacts_s"][pair]
    ax.axvspan(win["min"], win["max"], color=BAND, alpha=0.6, linewidth=0, zorder=0)
    ax.text(win["median"], 1.01, text, color=MUTED, fontsize=8, ha="center", va="bottom",
            transform=ax.get_xaxis_transform())
for key, label, colour in SERIES:
    row = SCORES["models"][key]["pos_m"]
    t = [DT * (i + 1) for i in range(len(row["median"]))]
    ax.fill_between(t, [1000 * v for v in row["min"]], [1000 * v for v in row["max"]],
                    color=colour, alpha=0.18, linewidth=0, label=f"{label}, min to max")
    ax.plot(t, [1000 * v for v in row["median"]], color=colour, linewidth=2.0,
            label=f"{label}, median")
style(ax, f"Bouncing ball test: ball position error, free rollouts of {SCORES['shots']} shots",
      "mm")
ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout()
out = RESULTS / "horizon_error.png"
fig.savefig(out, facecolor="white")
print("wrote", out)
