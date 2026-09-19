"""Pair two CRM tracking score files and report the delivered-policy verdict.

This exists because every ad-hoc version of it got something wrong, and the same three
mistakes kept coming back:

  PAIRING ACROSS MACHINES. Replay is not machine-invariant. The same policy on two boxes
  flips completion on a few percent of episodes, so a policy scored on one box against a
  base scored on another carries a machine effect inside the number. Refused by default.

  PAIRING ACROSS CHRONO BUILDS. Same argument, same refusal. The record already carries
  chrono_md5 for this reason.

  AVERAGING A FALL INTO A TRACKING NUMBER. An episode that ends early because the robot
  went down contributes the rows before it went down, and those rows are a measurement of
  when it fell, not of how well it tracked. Scouting the payload corpus produced exactly
  this inversion: 9 kg scored a BETTER mean|err_vx| than 8 kg while lying on its face at
  4 s of an 8 s episode. So tracking error is computed ONLY over episodes both arms
  completed, and when the arms disagree on completion that disagreement is reported as
  the finding rather than folded into the mean.

Primary metric is mean|err_vx| over the paired set, relative to base, which is the
quantity the whole study is stated in. vy and wz are reported alongside because a policy
can buy forward tracking with heading drift and the headline would not show it.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys


def load(path):
    try:
        recs = json.load(open(path))
    except FileNotFoundError:
        raise SystemExit(f"FATAL: no score file at {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"FATAL: {path} is not valid JSON ({exc})")
    if not isinstance(recs, list):
        raise SystemExit(f"FATAL: {path} is not a list of episode records")
    return [r for r in recs if isinstance(r, dict) and r.get("episode_id")]


def host_of(path, recs):
    # The host is in the filename by convention; the build is in the record.
    builds = {r.get("chrono_md5") for r in recs if r.get("chrono_md5")}
    return path.split("_")[-1].removesuffix(".json"), builds


def summarise(recs, keys):
    d = {r["episode_id"]: r for r in recs}
    out = {}
    for ax in ("vx", "vy", "wz"):
        vals = [d[k][f"mae_{ax}"] for k in keys if f"mae_{ax}" in d[k]]
        out[ax] = st.mean(vals) if vals else float("nan")
    z = [d[k]["min_z_m"] for k in keys if "min_z_m" in d[k]]
    if z:
        out["fell"] = sum(1 for v in z if v < 0.20)
        out["min_z"] = min(z)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base")
    ap.add_argument("arm")
    ap.add_argument("--label", default="arm")
    ap.add_argument("--allow-cross-host", action="store_true",
                    help="pair files from different boxes anyway; the machine effect is "
                         "then inside the number and the interval widens about 4x")
    a = ap.parse_args()

    b_recs, a_recs = load(a.base), load(a.arm)
    b_host, b_build = host_of(a.base, b_recs)
    a_host, a_build = host_of(a.arm, a_recs)
    if (b_host != a_host or b_build != a_build) and not a.allow_cross_host:
        print(f"REFUSED: base is {b_host} {sorted(b_build)}, arm is {a_host} {sorted(a_build)}.\n"
              f"  Replay is not machine-invariant and the difference would carry a machine\n"
              f"  effect. Score both on one box, or pass --allow-cross-host knowingly.",
              file=sys.stderr)
        return 2

    bd = {r["episode_id"]: r for r in b_recs}
    ad = {r["episode_id"]: r for r in a_recs}
    shared = sorted(set(bd) & set(ad))
    if not shared:
        print("FATAL: no shared episode ids; these files do not describe the same set",
              file=sys.stderr)
        return 2

    # Completion is a result, not a filter to apply quietly.
    b_done = {k for k in shared if bd[k].get("completed")}
    a_done = {k for k in shared if ad[k].get("completed")}
    both = sorted(b_done & a_done)
    only_b, only_a = sorted(b_done - a_done), sorted(a_done - b_done)

    print(f"  {len(shared)} shared episodes on {a_host}, build {sorted(a_build)[0][:8] if a_build else '?'}")
    print(f"  completed  base {len(b_done)}/{len(shared)}   {a.label} {len(a_done)}/{len(shared)}")
    if only_b or only_a:
        print(f"  COMPLETION DISAGREEMENT: {len(only_b)} base-only, {len(only_a)} {a.label}-only.")
        print(f"    This is a result in its own right. Tracking error below is over the "
              f"{len(both)} both arms finished.")
    if not both:
        print("  no episode completed under both arms; there is no tracking comparison to make")
        return 0

    B, A = summarise(b_recs, both), summarise(a_recs, both)
    print(f"\n  {'':<22s} {'vx':>9s} {'vy':>9s} {'wz':>9s}")
    print(f"  {'base':<22s} {B['vx']:9.4f} {B['vy']:9.4f} {B['wz']:9.4f}")
    print(f"  {a.label:<22s} {A['vx']:9.4f} {A['vy']:9.4f} {A['wz']:9.4f}")
    rel = {k: 100 * (A[k] - B[k]) / B[k] for k in ("vx", "vy", "wz")}
    print(f"  {'change':<22s} {rel['vx']:+8.1f}% {rel['vy']:+8.1f}% {rel['wz']:+8.1f}%")

    if "fell" in B or "fell" in A:
        print(f"\n  falls (min z < 0.20 m), over all {len(shared)} shared:")
        Bf, Af = summarise(b_recs, shared), summarise(a_recs, shared)
        print(f"    base {Bf.get('fell', 0)}/{len(shared)}   "
              f"{a.label} {Af.get('fell', 0)}/{len(shared)}")
        print(f"    For a loaded arm this is the primary quantity. Tracking error cannot\n"
              f"    distinguish a robot that tracked badly from one that stopped walking.")

    print(f"\n  HEADLINE  {rel['vx']:+.1f}% forward tracking, n={len(both)} paired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
