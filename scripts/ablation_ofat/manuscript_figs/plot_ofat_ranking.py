#!/usr/bin/env python3
"""Bar chart of all 14 Stage-A OFAT configs ranked by selection score S.

For the new manuscript appendix justifying the L8 dynamics backbone. Reuses
the same ranking logic as scripts/ablation_ofat/rank_stage_a.py (best-epoch
rollout_sel per config) but renders a bar chart grouped/colored by the swept
axis, with the anchor and the L8 winner called out.

Usage:
    python plot_ofat_ranking.py [--manifest configs/ablation_ofat/manifest.json] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]

AXIS_COLORS = {
    "anchor": "#555555",
    "depth": "C0",
    "width": "C1",
    "heads": "C2",
    "context": "C3",
}
AXIS_LABELS = {
    "anchor": "anchor (6L/8H/E256/ctx128)",
    "depth": "depth ($n_\\mathrm{layer}$)",
    "width": "width ($n_\\mathrm{embd}$)",
    "heads": "heads ($n_\\mathrm{head}$)",
    "context": "context (block_size)",
}


def load_metrics(run_dir: Path) -> list[dict]:
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def best_epoch(metrics: list[dict]) -> dict | None:
    valid = [
        m for m in metrics
        if isinstance(m.get("rollout_sel"), (int, float)) and math.isfinite(m["rollout_sel"])
    ]
    return min(valid, key=lambda m: m["rollout_sel"]) if valid else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/ablation_ofat/manifest.json")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts/analysis/manuscript_figs"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((REPO_ROOT / args.manifest).read_text())
    rows = []
    for e in manifest["entries"]:
        run_dir = REPO_ROOT / e["run_dir"]
        best = best_epoch(load_metrics(run_dir))
        if best is None:
            continue
        rows.append({"arm": e["arm"], "spec": e["spec"], "S": best["rollout_sel"]})

    rows.sort(key=lambda r: r["S"])

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    y = range(len(rows))
    colors = [AXIS_COLORS[r["arm"]] for r in rows]
    used_spec = "L8_H8_E256_ctx128"
    edge_widths = [1.6 if r["spec"] == used_spec else 0.6 for r in rows]
    ax.barh(y, [r["S"] * 100 for r in rows], color=colors, edgecolor="black", linewidth=edge_widths)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r["spec"] for r in rows], fontsize=8)
    for tick, r in zip(ax.get_yticklabels(), rows):
        if r["spec"] == used_spec:
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("Selection score $S$ (domain-balanced 10 s rollout err/dist, %)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=AXIS_COLORS[a], label=AXIS_LABELS[a])
        for a in ["anchor", "depth", "width", "heads", "context"]
    ]
    ax.legend(handles=handles, frameon=True, loc="upper right", fontsize=8)
    leg = ax.get_legend()
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("none")
    leg.get_frame().set_alpha(0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(r["S"] * 100 for r in rows) * 1.12)

    fig.tight_layout()
    png_path = out_dir / "hmmwv_ofat_ranking.png"
    pdf_path = out_dir / "hmmwv_ofat_ranking.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
