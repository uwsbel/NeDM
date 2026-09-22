#!/usr/bin/env python3
"""Rollout error as a function of horizon, for one trained NN-ROM.

The selection metric is errdist at 10 s, and the 550-episode model scored 3.4x the
predict-no-motion floor there -- worse than assuming the robot stands still. But the
fine-tuner never rolls the model 10 s. Analytic policy gradient uses 15-step branches,
0.30 s. So "is this model usable" is a question about 0.30 s, and the 10 s number answers
a different one: it ranks checkpoints well (the previous study measured rho = +0.90
against transfer) without saying whether any of them is accurate at the horizon that is
actually used.

This measures errdist across horizons on the same held-out segments, so the crossing
point -- where the model stops beating "the robot does not move" -- is a number rather
than a guess. The previous study put its useful horizon at about 3 s.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]   # the quadruped/ directory; this script lives in diagnostics/
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--horizons", default="0.3,0.5,1,2,3,5,10")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--restamp", action="store_true",
                    help="write the measured profile into the checkpoint and recompute the "
                         "floor stamp at the 0.30 s use horizon. For checkpoints trained "
                         "before train.py recorded a profile, whose floor stamp was judged "
                         "at the selection horizon instead.")
    a = ap.parse_args()

    import torch
    import train as T
    import finetune as F

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    model, ck = F.load_nnrom(torch, a.model, dev, allow_smoke=True)
    ctx = ck["config"]["block_size"]
    corpus = T.Corpus(Path(a.corpus), ck["config"]["preset"], ctx)
    hs = [float(x) for x in a.horizons.split(",")]
    # The SAME episode set at every horizon, chosen so the longest horizon fits. Using
    # whatever fits each horizon would compare different segments at different lengths,
    # and a trend in the result would partly be a trend in which segments were eligible.
    need = ctx + int(round(max(hs) / 0.01))
    eps = [r for r in sorted(corpus.val, key=lambda r: -r["n"]) if r["n"] >= need][:a.episodes]
    print(f"model {Path(a.model).parent.name}  {len(eps)} held-out segments, each >= "
          f"{need} rows so every horizon uses the same set")
    print(f"{'horizon':>8} {'errdist':>9}   vs predict-no-motion (1.0)")
    prof = {}
    for h in hs:
        e, n = T.rollout_errdist(torch, model, corpus, eps, h, 0.01, ctx, dev)
        prof[f"{h:g}"] = e
        bar = "#" * min(40, int(round(10 * e)))
        verdict = "BETTER" if e < 1.0 else "worse "
        print(f"{h:>7.1f}s {e:>9.3f}   {verdict} {bar}")

    if a.restamp:
        # A CORRECTION, and recorded as one. The old stamp was computed at the selection
        # horizon, which is the wrong horizon for a usability floor; this recomputes it at
        # the use horizon from a measurement, and keeps the old value beside the new one so
        # nothing about the checkpoint's history is overwritten silently.
        crossing = next((float(k) for k in sorted(prof, key=float) if prof[k] >= 1.0), None)
        raw = torch.load(a.model, map_location="cpu", weights_only=False)
        old = raw.get("worse_than_no_motion")
        use = prof.get("0.3")
        raw["horizon_profile"] = prof
        raw["usable_to_s"] = crossing
        raw["worse_than_no_motion"] = bool(use is not None and use >= 1.0)
        raw["restamped"] = {"by": "horizon_sweep.py --restamp",
                            "previous_worse_than_no_motion": old,
                            "reason": "floor was judged at the selection horizon; "
                                      "recomputed at the 0.30 s use horizon",
                            "episodes": len(eps)}
        torch.save(raw, a.model)
        print(f"\nrestamped {a.model}: worse_than_no_motion {old} -> "
              f"{raw['worse_than_no_motion']} (errdist {use:.3f} at 0.30 s), usable to "
              f"~{crossing} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
