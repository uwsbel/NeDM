#!/usr/bin/env python3
"""Training-curve figure for the L8 terrain-conditioned generalist dynamics model.

Replaces the manuscript's hmmwv_cotrain_training.pdf, which was built for the
6-layer anchor (selected epoch 69). This reads metrics.jsonl for
ablation_ofat/L8_H8_E256_ctx128 and marks the actual best_val epoch (51).

Usage:
    python plot_l8_training_curves.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = REPO_ROOT / "artifacts/training_runs/ablation_ofat/L8_H8_E256_ctx128"


def load_metrics() -> list[dict]:
    rows = []
    for line in (RUN_DIR / "metrics.jsonl").read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts/analysis/manuscript_figs"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_metrics()
    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_flat_loss = [r["val_flat_loss"] for r in rows]
    val_crm_loss = [r["val_crm_loss"] for r in rows]
    flat10 = [r["rollout_flat_10.0s"]["errdist"] * 100 for r in rows]
    crm10 = [r["rollout_crm_10.0s"]["errdist"] * 100 for r in rows]
    sel = [r["rollout_sel"] * 100 for r in rows]

    valid = [(r["epoch"], r["rollout_sel"]) for r in rows if math.isfinite(r["rollout_sel"])]
    best_epoch, best_sel = min(valid, key=lambda t: t[1])

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(4.6, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.1]}
    )

    ax_top.plot(epochs, train_loss, color="black", lw=1.8, label="train")
    ax_top.plot(epochs, val_flat_loss, color="C0", lw=1.8, label="val (flat)")
    ax_top.plot(epochs, val_crm_loss, color="C1", lw=1.8, label="val (CRM)")
    ax_top.set_yscale("log")
    ax_top.set_ylabel("Huber loss")
    leg_top = ax_top.legend(frameon=True, loc="upper right", fontsize=9)
    leg_top.get_frame().set_facecolor("white")
    leg_top.get_frame().set_edgecolor("none")
    leg_top.get_frame().set_alpha(0.9)
    ax_top.grid(True, which="major", axis="y", alpha=0.3)
    ax_top.spines[["top", "right"]].set_visible(False)

    ax_bot.plot(epochs, flat10, color="C0", lw=1.3, label="flat")
    ax_bot.plot(epochs, crm10, color="C1", lw=1.3, label="CRM")
    ax_bot.plot(epochs, sel, color="gray", lw=1.5, ls="--", label="selection $S$")
    ax_bot.axvline(best_epoch, color="black", ls=":", lw=1.0)
    ax_bot.scatter([best_epoch], [best_sel * 100], color="black", zorder=5, s=28)
    ax_bot.annotate(
        f"selected (ep {best_epoch})",
        xy=(best_epoch, best_sel * 100), xytext=(best_epoch + 4, 38),
        fontsize=8.5, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
    )
    ax_bot.set_ylabel("10 s open-loop err/dist (%)")
    ax_bot.set_xlabel("epoch")
    ax_bot.legend(frameon=False, loc="upper right", fontsize=9, ncol=3)
    ax_bot.grid(True, which="major", axis="y", alpha=0.3)
    ax_bot.spines[["top", "right"]].set_visible(False)
    ax_bot.set_xlim(1, max(epochs))

    fig.tight_layout()
    png_path = out_dir / "hmmwv_cotrain_training_L8.png"
    pdf_path = out_dir / "hmmwv_cotrain_training_L8.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    print(f"best epoch = {best_epoch}, S = {best_sel:.4f}")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
