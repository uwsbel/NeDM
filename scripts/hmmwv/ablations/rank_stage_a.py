#!/usr/bin/env python3
"""Rank the OFAT ablation runs by the domain-balanced selection score S.

S == the trainer's `rollout_sel` (weight-averaged flat+CRM 10 s open-loop
err/dist), taken at each run's best (minimum) epoch — i.e. the score of the
checkpoint the trainer would keep as best_val.pt. Lower is better. This is the
same metric used to pick the current generalist, so ranking on it selects the
Stage B top-4 on a like-for-like basis.

Reads each run's metrics.jsonl (the anchor from its existing run dir via the
manifest). Also reports the per-domain 10 s err/dist at the best epoch and the
last epoch reached, so incomplete runs are visible.

Usage:
    python scripts/ablations/rank_stage_a.py [--manifest configs/ablation_ofat/manifest.json] [--top 4]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    ap.add_argument("--manifest", default="configs/ablation_ofat/manifest.json")
    ap.add_argument("--top", type=int, default=4, help="Mark this many best runs for Stage B.")
    args = ap.parse_args()

    manifest = json.loads((REPO_ROOT / args.manifest).read_text())
    rows = []
    for e in manifest["entries"]:
        run_dir = REPO_ROOT / e["run_dir"]
        metrics = load_metrics(run_dir)
        best = best_epoch(metrics)
        last_ep = max((int(m.get("epoch", 0)) for m in metrics), default=0)
        rows.append({
            "arm": e["arm"], "name": e["name"], "spec": e["spec"],
            "S": best["rollout_sel"] if best else float("nan"),
            "best_ep": int(best["epoch"]) if best else None,
            "flat10": errdist(best, "rollout_flat_10.0s") if best else float("nan"),
            "crm10": errdist(best, "rollout_crm_10.0s") if best else float("nan"),
            "last_ep": last_ep,
            "status": "done" if last_ep >= 80 else (f"ep{last_ep}" if last_ep else "no-metrics"),
        })

    ranked = sorted(rows, key=lambda r: (math.isnan(r["S"]), r["S"]))
    print(f"{'rank':>4s} {'arm':8s} {'spec':20s} {'S(sel)':>8s} {'flat10':>7s} "
          f"{'crm10':>7s} {'best_ep':>7s} {'status':>10s}")
    print("-" * 80)
    for i, r in enumerate(ranked, 1):
        mark = "  <-- Stage B" if (i <= args.top and not math.isnan(r["S"])) else ""
        s = f"{r['S']:.4f}" if not math.isnan(r["S"]) else "   -  "
        f10 = f"{r['flat10']:.4f}" if not math.isnan(r["flat10"]) else "  -  "
        c10 = f"{r['crm10']:.4f}" if not math.isnan(r["crm10"]) else "  -  "
        be = str(r["best_ep"]) if r["best_ep"] is not None else "-"
        print(f"{i:4d} {r['arm']:8s} {r['spec']:20s} {s:>8s} {f10:>7s} {c10:>7s} "
              f"{be:>7s} {r['status']:>10s}{mark}")

    finished = [r for r in ranked if not math.isnan(r["S"])]
    print(f"\n{len(finished)}/{len(rows)} runs have a finite selection score. "
          f"Lower S = better. Top-{args.top} marked for Stage B (2 extra seeds + full metrics).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
