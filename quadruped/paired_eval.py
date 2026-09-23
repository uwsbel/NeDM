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
# Push-mode records carry these instead; --metrics picks them (or any subset).
PUSH_METRICS = ("push_mae_vx", "push_mae_vy", "push_mae_wz", "push_peak_err_mps",
                "push_excursion_extra_m", "push_recover_s",
                "push_recover_fixed_s")


def load(path):
    recs = json.loads(Path(path).read_text())
    return {r["episode_id"].rsplit("_", 1)[-1]: r for r in recs}


def usable(r):
    return r.get("completed") == 1 and r.get("upright") == 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="label=path")
    ap.add_argument("--metrics", nargs="+", default=list(METRICS),
                    help=f"which per-episode numbers to compare; push runs want "
                         f"{' '.join(PUSH_METRICS)} (or 'push' for exactly those)")
    ap.add_argument("--by-family", action="store_true",
                    help="also break each arm down by command family (path-mode records)")
    a = ap.parse_args()
    metrics = list(PUSH_METRICS) if a.metrics == ["push"] else list(a.metrics)

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
        for m in metrics:
            # A pair is dropped for a metric only where that metric is missing or NaN:
            # push_recover_s is NaN for a robot that never recovers, and averaging
            # that away would score the failures as if they had not happened, so the
            # count is printed beside the mean.
            miss = [k for k in pairs if not (isinstance(arm[k].get(m), (int, float))
                                             and isinstance(base[k].get(m), (int, float))
                                             and arm[k][m] == arm[k][m]
                                             and base[k][m] == base[k][m])]
            use = [k for k in pairs if k not in miss]
            if len(use) < 2:
                print(f"  {m:18s} too few pairs with this number ({len(use)})")
                continue
            d = [arm[k][m] - base[k][m] for k in use]
            b = [base[k][m] for k in use]
            md = st.mean(d)
            sd = st.stdev(d)
            se = sd / math.sqrt(len(d))
            t = md / se if se > 0 else float("nan")
            pct = 100 * md / st.mean(b) if st.mean(b) else float("nan")
            wins = sum(1 for x in d if x < 0)
            sig = "  <-- clear of zero" if abs(t) >= 2 else ""
            drop = f"  ({len(miss)} pairs lack it)" if miss else ""
            print(f"  {m:18s} base {st.mean(b):.4f}  change {md:+.4f} ({pct:+.1f}%)  "
                  f"se {se:.4f}  t {t:+6.2f}  better in {wins}/{len(d)}{sig}{drop}")
        if a.by_family:
            fams = sorted({base[k].get("family", "?") for k in pairs})
            print(f"  {'family':<12s} {'n':>3s} " + " ".join(f"{'d ' + m:>9s}" for m in metrics))
            for fam in fams:
                ks = [k for k in pairs if base[k].get("family") == fam]
                cells = []
                for m in metrics:
                    kk = [k for k in ks if isinstance(arm[k].get(m), (int, float))
                          and isinstance(base[k].get(m), (int, float))
                          and arm[k][m] == arm[k][m] and base[k][m] == base[k][m]]
                    if not kk:
                        cells.append("      --")
                        continue
                    b_ = st.mean(base[k][m] for k in kk)
                    d_ = st.mean(arm[k][m] - base[k][m] for k in kk)
                    cells.append(f"{100 * d_ / b_:+8.1f}%" if b_ else "      --")
                print(f"  {fam:<12s} {len(ks):3d} " + " ".join(cells))
        # In push mode, whether the robot recovered at all is the headline, and it is not
        # an average: a pair where either arm never got back under the threshold has no
        # recovery time to compare.
        if any(m.startswith("push_") for m in metrics):
            rb = sum(1 for k in pairs if base[k].get("push_recovered") == 1)
            ra = sum(1 for k in pairs if arm[k].get("push_recovered") == 1)
            print(f"  recovered within the window: base {rb}/{len(pairs)}, "
                  f"this arm {ra}/{len(pairs)}")
        # The resolution this run could have detected, so a null reads as a bound.
        mr = metrics[0]
        d_vx = [arm[k][mr] - base[k][mr] for k in pairs
                if isinstance(arm[k].get(mr), (int, float))
                and isinstance(base[k].get(mr), (int, float))
                and arm[k][mr] == arm[k][mr] and base[k][mr] == base[k][mr]]
        if len(d_vx) < 2:
            print("  -> resolution: not enough pairs to say")
            continue
        se_vx = st.stdev(d_vx) / math.sqrt(len(d_vx))
        print(f"  -> resolution: a {mr} change smaller than {2 * se_vx:.4f} "
              f"would not show at 2 se with {len(d_vx)} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
