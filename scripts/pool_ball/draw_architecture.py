"""Architecture figure of the selected pool NRD (per-cushion contact modules).

Numbered boxes are neural networks; everything else (threshold, multiply,
sum, feedback) is arithmetic around them."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "artifacts/pool_ball/architecture"
OUT.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(14, 8.4))
ax.set_xlim(0, 13.4)
ax.set_ylim(-0.2, 8.2)
ax.axis("off")


def box(x, y, w, h, text, color, fs=9.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1", fc=color, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, linespacing=1.35)


def arrow(p0, p1, color="#333", style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13, lw=1.3, color=color, linestyle=ls))


def line(xs, ys, color="#333", ls="-"):
    ax.plot(xs, ys, color=color, lw=1.3, ls=ls, solid_capstyle="round")


def node(x, y, label, fs=15):
    ax.add_patch(plt.Circle((x, y), 0.3, fc="white", ec="#333", lw=1.4, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=4)


# input and bus
box(0.1, 3.45, 1.75, 1.6, "state $s_t$  [14]\n\nA [7]: x, y, vx, vy,\n      wx, wy, wz\nB [7]: same", "#eeeeee", fs=9)
line([1.85, 2.3], [4.25, 4.25])
line([2.3, 2.3], [1.15, 7.55])
# networks
nets = [
    (6.3, 1.0, "1  Transformer backbone  (shared weights, run on each ball)\nin: one ball [7]   ->   out: smooth change [7]   (zero for a ball at rest)", "#cfe2ff", 6.8),
    (4.75, 0.85, "2  Ball-ball contact MLP\nin: s [14]  ->  p(A-B contact in next 10 ms) [1]", "#ffe0b2", 5.18),
    (3.55, 0.85, "3  Ball-ball bounce NN\nin: s [14]  ->  correction d_AB [14]", "#ffcc80", 3.98),
    (2.2, 0.85, "4-7  Cushion contact MLPs (xp, xm, yp, ym; each shared by A and B)\nin: one ball [7]  ->  p(it hits this cushion in next 10 ms) [1]", "#c8e6c9", 2.62),
    (1.0, 0.85, "8-11  Cushion bounce NNs (one per cushion; shared by A and B)\nin: one ball [7]  ->  correction for that ball [7]", "#a5d6a7", 1.42),
]
for y, h, text, color, yc in nets:
    box(2.7, y, 5.3, h, text, color, fs=8.8)
    arrow((2.3, yc), (2.7, yc))
# gates
box(8.45, 4.95, 1.25, 0.45, "1[p >= 0.5]", "#fff3e0", fs=8.5)
arrow((8.0, 5.18), (8.45, 5.18))
box(8.45, 2.4, 1.25, 0.45, "1[p >= 0.5]  x8", "#e8f5e9", fs=8.5)
arrow((8.0, 2.62), (8.45, 2.62))
node(10.3, 3.98, "x")
node(10.3, 1.42, "x")
arrow((8.0, 3.98), (10.0, 3.98))
arrow((8.0, 1.42), (10.0, 1.42))
line([9.7, 10.3], [5.18, 5.18]); arrow((10.3, 5.18), (10.3, 4.28))
line([9.7, 10.3], [2.62, 2.62]); arrow((10.3, 2.62), (10.3, 1.72))
ax.text(10.4, 4.7, "gate $g_{AB}$", fontsize=8.5, color="#e65100")
ax.text(10.4, 2.15, "gates $g_{ball,c}$", fontsize=8.5, color="#2e7d32")
# sum
node(11.6, 3.98, "+", fs=18)
line([8.0, 11.6], [6.8, 6.8]); arrow((11.6, 6.8), (11.6, 4.28))
ax.text(9.2, 6.9, "smooth change [14] (A ; B)", fontsize=8.5)
arrow((10.6, 3.98), (11.3, 3.98))
line([10.6, 11.6], [1.42, 1.42]); arrow((11.6, 1.42), (11.6, 3.68))
ax.text(10.65, 1.05, "cushion corrections [14]", fontsize=8.5)
ax.text(10.62, 4.08, "[14]", fontsize=8)
# skip connection s -> sum
line([2.3, 2.3, 11.85], [7.55, 7.7, 7.7], color="#777"); arrow((11.85, 7.7), (11.85, 4.25), color="#777")
ax.text(6.0, 7.8, "current state s (residual)", fontsize=8.5, color="#555")
# output and feedback
box(12.15, 3.55, 1.15, 0.85, "$s_{t+1}$ [14]\n10 ms later", "#eeeeee", fs=9)
arrow((11.9, 3.98), (12.15, 3.98))
line([12.72, 12.72, 0.97], [3.55, 0.1, 0.1], color="#1565c0", ls="--")
arrow((0.97, 0.1), (0.97, 3.45), color="#1565c0", ls="--")
ax.text(6.8, -0.12, "fed back as the next input, 200 times to t = 2.0 s; launch (vx, vy) enters only through $s_0$",
        ha="center", fontsize=9.5, color="#1565c0")
ax.text(6.7, 8.08, r"$s_{t+1} = s_t + d_{smooth} + g_{AB}\, d_{AB} + \sum_{ball}\,\sum_{c}\, g_{ball,c}\, d_c(ball)$"
        "      numbered boxes = neural networks (11); switches decide from the predicted state only",
        ha="center", fontsize=10)
fig.savefig(OUT / "pool_nrd_architecture.png", dpi=140, bbox_inches="tight")
fig.savefig(OUT / "pool_nrd_architecture.svg", bbox_inches="tight")
print(OUT / "pool_nrd_architecture.png")
