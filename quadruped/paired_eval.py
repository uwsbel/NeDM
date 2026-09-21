#!/usr/bin/env python3
"""Paired comparison of fine-tuned policies against the base, from evaluate.py records.

The study's claim is a GAIN -- fine-tuned minus base -- so the comparison is paired
episode by episode rather than a difference of two means. evaluate.py gives every arm the
identical case list (same command, same spawn offsets over +/-1 m), so episode k of one
arm and episode k of another started from the same place on the particle lattice. Pairing
cancels the part of the variance that comes from WHERE a run started, which on CRM is most
of it: tracking has a standard deviation of about 5.7 points between spawn positions and
0.1 on rigid ground.

WHAT THIS REFUSES

  Pairs from different Chrono builds. Two arms scored against different binaries are not
  a paired comparison however well the episode ids line up; the build md5 is on every
  record and is checked.

  Episodes either arm failed. A fall is not a slow walk, and averaging a fallen episode's
  tracking error into a mean would score the fall as a tracking result. Both arms must
  have completed AND stayed upright for a pair to count, and how many were dropped is
  printed rather than absorbed.

Lower tracking error is better, so a NEGATIVE paired difference is an improvement.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

METRICS = ("mae_vx", "mae_vy", "mae_wz")


def load(path):
    recs = json.loads(Path(path).read_text())
    return {r["episode_id"].rsplit("_", 1)[-1]: r for r in recs}


def usable(r):
    return r.get("completed") == 1 and r.get("upright") == 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="label=path")
    a = ap.parse_args()

    base = load(a.base)
    builds = {r.get("chrono_md5") for r in base.values()}
    ok_base = sum(usable(r) for r in base.values())
    print(f"base: {len(base)} episodes, {ok_base} completed upright, build {builds}")

    for spec in a.arms:
        label, path = spec.split("=", 1)
        arm = load(path)
        ab = {r.get("chrono_md5") for r in arm.values()}
        if ab != builds:
            print(f"\n{label}: REFUSED -- scored on build {ab}, base on {builds}. Not a "
                  f"paired comparison across binaries.")
            continue
        keys = sorted(set(base) & set(arm))
        pairs = [k for k in keys if usable(base[k]) and usable(arm[k])]
        dropped = len(keys) - len(pairs)
        falls = sum(1 for k in keys if not usable(arm[k]))
        print(f"\n{label}: {len(keys)} paired episodes, {len(pairs)} usable, "
              f"{dropped} dropped ({falls} where this arm fell or failed)")
        if len(pairs) < 2:
            print("  too few usable pairs to say anything")
            continue
        for m in METRICS:
            d = [arm[k][m] - base[k][m] for k in pairs]
            b = [base[k][m] for k in pairs]
            md = st.mean(d)
            sd = st.stdev(d)
            se = sd / math.sqrt(len(d))
            t = md / se if se > 0 else float("nan")
            pct = 100 * md / st.mean(b) if st.mean(b) else float("nan")
            wins = sum(1 for x in d if x < 0)
            sig = "  <-- clear of zero" if abs(t) >= 2 else ""
            print(f"  {m:7s} base {st.mean(b):.4f}  change {md:+.4f} ({pct:+.1f}%)  "
                  f"se {se:.4f}  t {t:+6.2f}  better in {wins}/{len(d)}{sig}")
        # The resolution this run could have detected, so a null reads as a bound.
        d_vx = [arm[k]["mae_vx"] - base[k]["mae_vx"] for k in pairs]
        se_vx = st.stdev(d_vx) / math.sqrt(len(d_vx))
        print(f"  -> resolution: a mae_vx change smaller than {2 * se_vx:.4f} m/s "
              f"would not show at 2 se with {len(pairs)} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
