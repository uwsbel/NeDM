#!/usr/bin/env python3
"""Closed-loop Chrono policy-transfer bar chart for the L8 backbone.

Replaces the manuscript's hmmwv_policy_transfer_bars.pdf (6-layer/500-iter
run), reproducing the same visual style (value-labeled bars, black edges,
upper-left legend) with the new L8/iteration-1000 numbers.

Usage:
    python plot_l8_policy_transfer_bars.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
STATS_JSON = (
    REPO_ROOT
    / "artifacts/rl_runs/chrono_eval_comparisons/"
    "onehot_policy_3x3_chrono_xy_rmse_median_iqr_steerlim010_ofatL8_model1000.json"
)

TERRAINS = ["rigid_flat", "crm", "bumpy"]
TERRAIN_LABELS = ["Rigid flat", "CRM", "Rigid bumpy"]
POLICIES = ["mixture", "rigid_only", "crm_only"]
POLICY_LABELS = ["Mixture generalist", "Rigid-only", "CRM-only"]
COLORS = ["C0", "C1", "C2"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts/analysis/manuscript_figs"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(STATS_JSON.read_text())["stats"]

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    n_terrain = len(TERRAINS)
    n_policy = len(POLICIES)
    width = 0.25
    x = np.arange(n_terrain)

    for j, (pol, label, color) in enumerate(zip(POLICIES, POLICY_LABELS, COLORS)):
        medians = [data[t][pol]["median_xy_rmse_m"] for t in TERRAINS]
        q1 = [data[t][pol]["q1_xy_rmse_m"] for t in TERRAINS]
        q3 = [data[t][pol]["q3_xy_rmse_m"] for t in TERRAINS]
        lower_err = [m - a for m, a in zip(medians, q1)]
        upper_err = [b - m for m, b in zip(medians, q3)]
        offset = (j - (n_policy - 1) / 2) * width
        bars = ax.bar(
            x + offset, medians, width, color=color, edgecolor="black", linewidth=0.7,
            yerr=[lower_err, upper_err], capsize=2, error_kw=dict(lw=1.0, ecolor="black"),
            label=label,
        )
        for rect, m in zip(bars, medians):
            ax.text(
                rect.get_x() + rect.get_width() / 2, m + 0.012, f"{m:.2f}",
                ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(TERRAIN_LABELS, fontsize=10)
    ax.set_ylabel("Median XY RMSE [m] (IQR bars)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(True, which="major", axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(data[t][p]["q3_xy_rmse_m"] for t in TERRAINS for p in POLICIES) * 1.15)

    fig.tight_layout()
    png_path = out_dir / "hmmwv_policy_transfer_bars_L8.png"
    pdf_path = out_dir / "hmmwv_policy_transfer_bars_L8.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
