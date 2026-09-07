"""Closed-loop gain margin k*: the factor by which loop gain can be multiplied
before the plant-policy loop goes unstable.

USES NEDM_ACTION_MULT, THE CANONICAL PATH. An earlier version of this file scaled
the actor's final layer instead, which is affine and therefore an exact uniform
output gain -- validated to 5e-5 against a*k. Both paths were then compared
END TO END through the collector at k = 0.75:

    max |diff| in joint targets   3.8e-05 at row 0, 3.6e-05 at row 174, flat between

Constant and non-growing, so the two agree to float32 rounding order and neither is
semantically wrong. **But they are not bit-identical**, and bit-exact replay is the
property the paired verdict rests on. Two ways to scale one quantity, differing in
the last few digits, is the pipeline-duplication hazard this project has recorded
twice -- so the weight-scaling path was deleted in favour of sbel-pc's env var
rather than kept as an equivalent alternative.

k* is read off the divergence rate from standing_screen: the smallest k at which
the rate crosses 0.5.
"""
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
from standing_screen import screen




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ks", nargs="*", type=float, default=[0.7, 0.85, 1.0, 1.2, 1.5])
    ap.add_argument("--duration-s", type=float, default=20.0)
    a = ap.parse_args()
    print(f"  {'k':>6s} {'divergence rate':>16s}")
    kstar, prev_k, prev_r = None, None, None
    for k in sorted(a.ks):
        # screen() shells out to the collector, which inherits os.environ.
        os.environ["NEDM_ACTION_MULT"] = f"{k:.6f}"
        v, info = screen(a.ckpt, duration=a.duration_s)
        r = info["rate"]
        print(f"  {k:6.2f} {v:>10s} {r:5.2f}")
        if kstar is None and r >= 0.5:
            # linear interpolation between the last sub-0.5 rung and this one
            kstar = k if prev_k is None else prev_k + (0.5 - prev_r) * (k - prev_k) / (r - prev_r)
        prev_k, prev_r = k, r
    if kstar is None:
        print(f"\n  k* > {max(a.ks)}  (never crossed 0.5 in the swept range)")
    else:
        print(f"\n  k* = {kstar:.3f}   {'STABLE as deployed (k* > 1)' if kstar > 1 else 'UNSTABLE as deployed (k* < 1)'}")
    os.environ.pop("NEDM_ACTION_MULT", None)
