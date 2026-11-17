#!/usr/bin/env python3
"""The pool's architecture figures, from the stored rollouts.npz and results.json and the runs' logs.

    python scripts/contact_nrd/arch_ablation/pool/figures.py

writes, to ``artifacts/contact_nrd/arch_ablation/pool/``:

* ``horizon.png``   each ball's position error against time over the 4,800 test shots
* ``endpoints.png`` where each model puts ball B at the test's target time (2.0 s), relative to Chrono
* ``training.png``  the three ablations' validation score at every check of their training

and prints the table of ``results.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # layout.py is one folder up
import layout  # noqa: E402

RESULTS = layout.RESULTS / "pool"
RUNS = layout.NRDLAB_RUNS
MODELS = {  # key in rollouts.npz: (label, colour)
    "3T s61": ("3T: three transformers (released)", "#2a78d6"),
    "3Mlp": ("3Mlp: three MLPs", "#1baf7a"),
    "1Mlp": ("1Mlp: one MLP", "#eb6834"),
    "1Transformer": ("1Transformer: one transformer", "#6250d6"),
}
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "lines.linewidth": 2,
        "font.size": 9,
    }
)


def errors(d: dict, model: str) -> np.ndarray:
    """[shot, step, ball] position error (mm)."""
    return 1000 * np.linalg.norm(d[model] - d["GT"], axis=-1)


def horizon(d: dict) -> None:
    t = np.arange(d["GT"].shape[1]) * float(d["dt"])
    target_s = int(d["target_step"]) * float(d["dt"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ball, ax in enumerate(axes):
        for m, (label, color) in MODELS.items():
            q25, q50, q75 = np.percentile(errors(d, m)[:, 1:, ball], (25, 50, 75), axis=0)
            ax.fill_between(t[1:], q25, q75, color=color, alpha=0.15, lw=0)
            ax.plot(t[1:], q50, color=color, label=label)
        ax.axvline(target_s, color="#52514e", lw=1, ls="--")
        ax.text(target_s, ax.get_ylim()[0], " test's target time", color="#52514e", fontsize=7, va="bottom")
        ax.set(yscale="log", xlabel="time after the shot (s)", title=f"Ball {'AB'[ball]}" + (" (the cue ball)" if ball == 0 else " (the target ball)"))
    axes[0].set_ylabel("position error (mm)")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncols=4, loc="lower center", bbox_to_anchor=(0.5, 0.97))
    fig.text(0.5, -0.02, "4,800 test shots, each rolled out from its first state; median, band: 25th-75th percentile",
             ha="center", color="#52514e")
    fig.tight_layout()
    fig.savefig(RESULTS / "horizon.png", bbox_inches="tight")


def endpoints(d: dict) -> None:
    k = int(d["target_step"])
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharex=True, sharey=True)
    for ax, (m, (label, color)) in zip(axes, MODELS.items()):
        offset = 1000 * (d[m][:, k, 1, :2] - d["GT"][:, k, 1, :2])
        ax.scatter(offset[:, 0], offset[:, 1], s=2, color=color, alpha=0.35, lw=0)
        p95 = np.percentile(np.linalg.norm(offset, axis=1), 95)
        ax.add_patch(plt.Circle((0, 0), p95, fill=False, color="#52514e", lw=1, ls="--"))
        ax.set_title(f"{label}\n95 % within {p95:.1f} mm", fontsize=8)
        ax.set(xlabel="x error (mm)", aspect="equal")
    axes[0].set(xlim=(-30, 30), ylim=(-30, 30), ylabel="y error (mm)")
    fig.suptitle("Where each model puts ball B at 2.0 s, relative to Chrono (4,800 shots)", y=1.05)
    fig.tight_layout()
    fig.savefig(RESULTS / "endpoints.png", bbox_inches="tight")


def training() -> None:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for m in ("3Mlp", "1Mlp", "1Transformer"):
        run = next((RUNS / f"ContactNRD-Pool-{m}" / "train-model").glob("*_arch_s61"))
        checks = [r for r in map(json.loads, (run / "metrics.jsonl").open()) if r["kind"] == "check"]
        ax.plot([r["update"] for r in checks], [r["score"] for r in checks], "o-", ms=4,
                color=MODELS[m][1], label=MODELS[m][0])
    ax.set(yscale="log", xlabel="update", ylabel="validation score (lower is better)",
           title="The ablations' training: still falling at the last check")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "training.png", bbox_inches="tight")


def table() -> None:
    results = json.loads((RESULTS / "results.json").read_text())
    print("| model | parameters | ball B at 2 s, median / p95 (mm) | path RMS, median / p95 (mm) | events right | score |")
    print("| --- | --- | --- | --- | --- | --- |")
    for name, r in results.items():
        print(f"| {name} | {r['parameters'] / 1e6:.2f} M | {r['pos_median_mm']:.2f} / {r['pos_p95_mm']:.2f} | "
              f"{r['path_median_mm']:.2f} / {r['path_p95_mm']:.2f} | {r['events_right']} / {r['episodes']} | "
              f"{r['selection_score']:.4f} |")


def main() -> None:
    d = dict(np.load(RESULTS / "rollouts.npz"))
    horizon(d)
    endpoints(d)
    training()
    table()
    print("wrote", *(RESULTS / f for f in ("horizon.png", "endpoints.png", "training.png")))


if __name__ == "__main__":
    main()
