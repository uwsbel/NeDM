#!/usr/bin/env python3
"""Accuracy-vs-training-data-quantity curve for the L8 depth winner.

Reports the domain-balanced selection score S (== the trainer's `rollout_sel`,
the weight-averaged flat+CRM 10 s open-loop err/dist at each run's best epoch,
i.e. the score of the checkpoint kept as best_val.pt; lower is better) as a
function of the fraction of training TRAJECTORIES used. The L8_H8_E256_ctx128
run is the 100% anchor; the four L8_H8_E256_ctx128_data{80,60,40,20} runs hold
the L8 architecture / optimizer / loss / 75-25 mix / seed / compute (80x2000)
fixed and vary ONLY train_episode_fraction (nested seeded episode subsets).

Unlike rank_stage_a.py this does NOT sort competitively and does NOT read the
manifest (these runs are intentionally excluded from it); it scans the five
fixed run dirs in descending data order so the output reads as a curve, and adds
the delta-S vs the 100% anchor.

Usage:
    python scripts/ablations/rank_data_quantity.py [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "artifacts/training_runs/ablation_ofat"

# (data fraction, run dir name). 1.0 = the existing L8 depth winner (anchor).
POINTS = [
    (1.0, "L8_H8_E256_ctx128"),
    (0.8, "L8_H8_E256_ctx128_data80"),
    (0.6, "L8_H8_E256_ctx128_data60"),
    (0.4, "L8_H8_E256_ctx128_data40"),
    (0.2, "L8_H8_E256_ctx128_data20"),
]


def load_metrics(run_dir: Path) -> list[dict]:
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def best_epoch(metrics: list[dict]) -> dict | None:
    """Epoch minimizing a finite rollout_sel."""
    valid = [m for m in metrics if isinstance(m.get("rollout_sel"), (int, float))
             and math.isfinite(m["rollout_sel"])]
    return min(valid, key=lambda m: m["rollout_sel"]) if valid else None


def errdist(m: dict, key: str) -> float:
    v = m.get(key, {})
    return float(v.get("errdist", float("nan"))) if isinstance(v, dict) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Optional path to write the curve as CSV.")
    args = ap.parse_args()

    rows = []
    for frac, name in POINTS:
        run_dir = RUN_ROOT / name
        metrics = load_metrics(run_dir)
        best = best_epoch(metrics)
        last_ep = max((int(m.get("epoch", 0)) for m in metrics), default=0)
        rows.append({
            "frac": frac,
            "name": name,
            "S": best["rollout_sel"] if best else float("nan"),
            "best_ep": int(best["epoch"]) if best else None,
            "flat10": errdist(best, "rollout_flat_10.0s") if best else float("nan"),
            "crm10": errdist(best, "rollout_crm_10.0s") if best else float("nan"),
            "last_ep": last_ep,
            "status": "done" if last_ep >= 80 else (f"ep{last_ep}" if last_ep else "no-metrics"),
        })

    anchor_S = rows[0]["S"]  # the 100% run
    print(f"{'data%':>6s} {'spec':28s} {'S(sel)':>8s} {'dS_vs100':>9s} "
          f"{'flat10':>7s} {'crm10':>7s} {'best_ep':>7s} {'status':>10s}")
    print("-" * 90)
    for r in rows:
        s = f"{r['S']:.4f}" if not math.isnan(r["S"]) else "   -  "
        if not math.isnan(r["S"]) and not math.isnan(anchor_S):
            ds = f"{r['S'] - anchor_S:+.4f}"
        else:
            ds = "   -  "
        f10 = f"{r['flat10']:.4f}" if not math.isnan(r["flat10"]) else "  -  "
        c10 = f"{r['crm10']:.4f}" if not math.isnan(r["crm10"]) else "  -  "
        be = str(r["best_ep"]) if r["best_ep"] is not None else "-"
        print(f"{int(r['frac']*100):5d}% {r['name']:28s} {s:>8s} {ds:>9s} "
              f"{f10:>7s} {c10:>7s} {be:>7s} {r['status']:>10s}")

    finished = [r for r in rows if not math.isnan(r["S"])]
    print(f"\n{len(finished)}/{len(rows)} points have a finite S. Lower S = better; "
          f"dS_vs100 > 0 means worse than the full-data anchor.")

    if args.csv:
        out = Path(args.csv)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
