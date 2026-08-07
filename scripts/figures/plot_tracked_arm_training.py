#!/usr/bin/env python3
"""Training-curve figure for the Study Case II reduced-dynamics models.

Two stacked panels, overlaying BOTH task-specific models (tracked-base + arm):
  (a) train and validation one-step loss vs epoch (log scale)
  (b) open-loop rollout error vs Chrono ground truth (err/dist, %) vs epoch

The frozen checkpoint of each model is now selected by minimum open-loop
rollout error (the selection metric in each run's metrics.jsonl, `rollout_sel`),
not by one-step validation loss -- so the selected epoch is marked on panel (b).
Tracked-base selects on the 5 s horizon; arm selects on the 0.5 s horizon.
Writes to the manuscript image archive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKED = REPO_ROOT / "artifacts/training_runs/tracked_transformer_v1/metrics.jsonl"
ARM = REPO_ROOT / "artifacts/training_runs/arm_transformer_8d_v1/metrics.jsonl"
DEFAULT_OUT = Path(
    "/home/harry/Manuscripts/ImageArchive/journals/2026/neural-dynamics-model"
)

# One color per model; train dashed, validation solid.
TRACKED_C = "C0"
ARM_C = "C1"


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def best_epoch(rows: list[dict]) -> tuple[int, float]:
    """Selected checkpoint = minimum open-loop rollout error (rollout_sel)."""
    b = min(rows, key=lambda r: r["rollout_sel"])
    return b["epoch"], b["rollout_sel"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr, ar = load(TRACKED), load(ARM)
    tr_ep = [r["epoch"] for r in tr]
    ar_ep = [r["epoch"] for r in ar]
    tr_sel_ep, tr_sel = best_epoch(tr)
    ar_sel_ep, ar_sel = best_epoch(ar)
    xmax = max(max(tr_ep), max(ar_ep))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(4.8, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.1]}
    )

    # --- (a) train / validation loss, both models overlaid ---
    ax_top.plot(tr_ep, [r["train_loss"] for r in tr], color=TRACKED_C, lw=1.6, ls="--",
                label="tracked train")
    ax_top.plot(tr_ep, [r["val_loss"] for r in tr], color=TRACKED_C, lw=1.8,
                label="tracked val")
    ax_top.plot(ar_ep, [r["train_loss"] for r in ar], color=ARM_C, lw=1.6, ls="--",
                label="arm train")
    ax_top.plot(ar_ep, [r["val_loss"] for r in ar], color=ARM_C, lw=1.8,
                label="arm val")
    ax_top.set_yscale("log")
    ax_top.set_ylabel("one-step loss")
    ax_top.set_title("(a) Train / validation loss", fontsize=9.5, loc="left")
    leg = ax_top.legend(frameon=True, loc="center right", fontsize=8, ncol=1)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("none")
    leg.get_frame().set_alpha(0.9)
    ax_top.grid(True, which="major", axis="y", alpha=0.3)
    ax_top.spines[["top", "right"]].set_visible(False)

    # --- (b) open-loop rollout error vs Chrono, both models overlaid ---
    ax_bot.plot(tr_ep, [r["rollout_sel"] * 100 for r in tr], color=TRACKED_C, lw=1.6,
                label="tracked (5 s horizon)")
    ax_bot.plot(ar_ep, [r["rollout_sel"] * 100 for r in ar], color=ARM_C, lw=1.6,
                label="arm (0.5 s horizon)")
    for ep, val, c, ha, dy, va in (
        (tr_sel_ep, tr_sel, TRACKED_C, "center", -13, "top"),
        (ar_sel_ep, ar_sel, ARM_C, "right", 12, "bottom"),
    ):
        ax_bot.scatter([ep], [val * 100], color=c, edgecolor="black", zorder=5, s=32)
        ax_bot.annotate(
            f"selected (ep {ep})",
            xy=(ep, val * 100), xytext=(0, dy), textcoords="offset points",
            fontsize=8, ha=ha, va=va, color=c,
        )
    # Tie the selected epoch back to the loss panel above.
    for ax in (ax_top, ax_bot):
        ax.axvline(tr_sel_ep, color=TRACKED_C, ls=":", lw=0.9, alpha=0.55)
        ax.axvline(ar_sel_ep, color=ARM_C, ls=":", lw=0.9, alpha=0.55)
    ax_bot.set_yscale("log")
    ax_bot.set_ylabel("open-loop err/dist vs Chrono (%)")
    ax_bot.set_xlabel("epoch")
    ax_bot.set_title("(b) Open-loop rollout error", fontsize=9.5, loc="left")
    ax_bot.legend(frameon=False, loc="upper right", fontsize=8)
    ax_bot.grid(True, which="major", axis="y", alpha=0.3)
    ax_bot.spines[["top", "right"]].set_visible(False)
    ax_bot.set_xlim(1, xmax)

    fig.tight_layout()
    png = out_dir / "tracked_arm_training.png"
    pdf = out_dir / "tracked_arm_training.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    print(f"tracked selected ep={tr_sel_ep} (rollout_sel={tr_sel:.4f}); "
          f"arm selected ep={ar_sel_ep} (rollout_sel={ar_sel:.4f})")
    print(f"wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
