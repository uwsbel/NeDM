"""Architecture figure of the unified contact NRD (one Transformer core, one
collision network, one contact network; all shared). Numbered boxes are the
three neural networks; everything else is arithmetic around them."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "artifacts/contact_nrd/architecture"
OUT.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(15, 8.8))
ax.set_xlim(0, 14.6)
ax.set_ylim(-0.4, 8.6)
ax.axis("off")


def box(x, y, w, h, text, color, fs=9.3, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1", fc=color, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, linespacing=1.35)


def arrow(p0, p1, color="#333", ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, lw=1.3, color=color, linestyle=ls))


def line(xs, ys, color="#333", ls="-"):
    ax.plot(xs, ys, color=color, lw=1.3, ls=ls, solid_capstyle="round")


def node(x, y, label, fs=15):
    ax.add_patch(plt.Circle((x, y), 0.3, fc="white", ec="#333", lw=1.4, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=fs, zorder=4)


# inputs
box(0.1, 5.2, 2.2, 2.3, "moving bodies' state $s_t$\n9 numbers each:\nposition, velocity, spin\n(world frame)\n\nball: 1 body\npool: A, B", "#eeeeee", fs=8.8)
box(0.1, 1.6, 2.2, 2.9, "scene: one token per body\n[sphere?, plane?, radius,\npoint, normal,\nvelocity, spin]\n\nball: ball, floor, wall\npool: A, B, 4 cushions\n\npair list (i moving, j any)\nball: 2 pairs   pool: 9 pairs", "#f5f5f5", fs=8.3)
# core
box(3.0, 6.25, 5.6, 1.2, "1  Transformer core  (one set of weights, run on each moving body)\nin: that body's state [9]  ->  smooth change $d_i$ [9]\n(gravity, rolling, spin decay; no contacts)", "#cfe2ff", fs=8.8)
arrow((2.3, 6.85), (3.0, 6.85))
# pair frame
box(3.0, 2.6, 2.3, 2.6, "pair frame (arithmetic)\nfor each pair (i, j):\ne1 = toward the partner\n(-normal for a plane,\nline of centres for spheres)\ne3 from gravity, e2 = e3 x e1\n\nfeatures [20]: gap vector,\nboth velocities and spins,\ngravity, both radii", "#fff8e1", fs=8.0)
line([1.2, 1.2], [4.5, 5.2])
arrow((2.3, 3.9), (3.0, 3.9))
line([2.3, 2.65, 2.65], [6.0, 6.0, 4.6]); arrow((2.65, 4.6), (3.0, 4.6))
# collision + contact nets
box(6.0, 4.15, 3.1, 1.2, "2  Collision network\n(one, shared by every pair)\nfeatures [20] + pair type\n->  on / off for this step", "#ffe0b2", fs=8.6)
box(6.0, 2.0, 3.1, 1.55, "3  Contact network\n(one, shared by every pair;\nper-type scale and shift)\nfeatures [20]  ->  change of\nbody i due to j [9 channels]:\nposition, velocity, spin", "#c8e6c9", fs=8.4)
arrow((5.3, 4.75), (6.0, 4.75))
arrow((5.3, 2.75), (6.0, 2.75))
box(9.55, 4.5, 1.3, 0.5, "1[logit >= 0]", "#fff3e0", fs=8.5)
arrow((9.1, 4.75), (9.55, 4.75))
node(10.2, 2.75, "x")
arrow((9.1, 2.75), (9.9, 2.75))
arrow((10.2, 4.5), (10.2, 3.05))
ax.text(10.3, 3.7, "gate $g_{ij}$", fontsize=8.5, color="#e65100")
box(10.9, 2.3, 1.45, 0.9, "rotate back\nto world;\nsum over pairs", "#e8f5e9", fs=8.3)
arrow((10.5, 2.75), (10.9, 2.75))
ax.text(5.35, 1.55, "if j also moves (A-B): the same network gives j's change too;\nthe on/off uses the mean of both directions' logits",
        fontsize=8.0, color="#555")
# sum
node(13.0, 4.75, "+", fs=18)
line([8.6, 13.0], [6.85, 6.85]); arrow((13.0, 6.85), (13.0, 5.05))
ax.text(10.1, 6.95, "smooth change [9 per body]", fontsize=8.5)
line([12.35, 13.0], [2.75, 2.75]); arrow((13.0, 2.75), (13.0, 4.45))
ax.text(12.4, 2.4, "contact changes", fontsize=8.5)
line([1.2, 1.2, 13.3], [7.5, 8.05, 8.05], color="#777"); arrow((13.3, 8.05), (13.3, 5.0), color="#777")
ax.text(6.0, 8.15, "current state (residual)", fontsize=8.5, color="#555")
box(13.45, 4.3, 1.1, 0.9, "$s_{t+1}$\n10 ms later", "#eeeeee", fs=9)
arrow((13.3, 4.75), (13.45, 4.75))
line([14.0, 14.0, 0.6], [4.3, -0.15, -0.15], color="#1565c0", ls="--")
arrow((0.6, -0.15), (0.6, 5.2), color="#1565c0", ls="--")
ax.text(7.2, -0.38, "fed back as the next input; the launch enters only through $s_0$; fixed bodies never change",
        ha="center", fontsize=9.5, color="#1565c0")
ax.text(7.3, 8.48, r"$s_{i,t+1} = s_{i,t} + d_i + \sum_{j}\, g_{ij}\, c_{ij}$"
        "      numbered boxes = the only neural networks (3), whatever the number of bodies",
        ha="center", fontsize=10.5)
fig.savefig(OUT / "unified_contact_nrd_architecture.png", dpi=140, bbox_inches="tight")
print(OUT / "unified_contact_nrd_architecture.png")
