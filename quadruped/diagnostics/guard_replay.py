#!/usr/bin/env python3
"""Replay the guard over every fine-tune this study has logged.

The guard in run_ppo stops a run when a window of iterations both leaves the corpus region
often and has stopped improving. Its thresholds were chosen from ONE collapse, so the first
question about them is not whether they catch that one -- they were drawn around it -- but
how close the fifty healthy runs come to tripping. Every run writes finetune.jsonl per
iteration, so this is answerable without spending a GPU minute.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def replay(rows, window, ood_thresh, spike_lim, drop_lim):
    """Returns (tripped_at, worst_spike_rate, worst_drop_when_spiking)."""
    buf, best, worst_s, worst_d = [], None, 0.0, 0.0
    for r in rows:
        buf.append(r)
        if len(buf) > window:
            buf.pop(0)
        if len(buf) < window:
            continue
        spike = sum(1 for x in buf if x["ood"] > ood_thresh) / window
        mean_r = sum(x["reward"] for x in buf) / window
        best = mean_r if best is None else max(best, mean_r)
        drop = (best - mean_r) / max(abs(best), 1e-9)
        worst_s = max(worst_s, spike)
        if spike > spike_lim:
            worst_d = max(worst_d, drop)
        if spike > spike_lim and drop > drop_lim:
            return r["iter"], worst_s, worst_d
    return None, worst_s, worst_d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="directories holding ft_* runs")
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--ood", type=float, default=0.005)
    ap.add_argument("--spike", type=float, default=0.6)
    ap.add_argument("--drop", type=float, default=0.25)
    a = ap.parse_args()
    runs = []
    for root in a.roots:
        runs += sorted(Path(root).glob("*/finetune.jsonl"))
    print(f"{len(runs)} runs; window {a.window}, spike > {a.spike}, reward drop > {a.drop}")
    tripped = []
    for f in runs:
        rows = []
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "ood" in r and "reward" in r:
                rows.append(r)
        if len(rows) < a.window:
            continue
        at, ws, wd = replay(rows, a.window, a.ood, a.spike, a.drop)
        flag = f"TRIPPED at {at}" if at else ""
        print(f"  {f.parent.name:44s} {len(rows):5d} iters  worst spike rate {ws:5.2f}  "
              f"worst drop while spiking {wd:6.2f}  {flag}")
        if at:
            tripped.append(f.parent.name)
    print(f"\ntripped: {len(tripped)} of {len(runs)}" + (": " + ", ".join(tripped) if tripped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
