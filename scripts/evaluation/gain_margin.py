"""Uniform-gain stability margin k*.

OPERATIONAL DEFINITION: the factor by which the ENTIRE action vector is uniformly
scaled at which the closed plant-policy loop loses stability, read off as the
smallest k where the divergence rate crosses 0.5.

NOT the classical Nyquist gain margin, which is defined for a linear system at the
frequency where phase crosses -180 degrees. This is a legitimate generalisation to
a nonlinear closed loop and the interpretive force is the same -- k* < 1 means the
policy is unstable as deployed, with no margin at all -- but calling it "the gain
margin" claims a correspondence this does not establish. The classical connection
is an observation, not a claim.

USES NEDM_ACTION_MULT, THE CANONICAL PATH. An earlier version of this file scaled
the actor's final layer instead, which is affine and therefore an exact uniform
output gain -- validated to 5e-5 against a*k. Both paths were then compared
END TO END through the collector at k = 0.75:

    max |diff| in joint targets   3.8e-05 at row 0, 3.6e-05 at row 174, flat between

Constant and non-growing, so the two agree to float32 rounding order and neither is
semantically wrong.

SCOPE OF THAT TEST, because it will get reused: "the difference did not grow"
distinguishes rounding from a semantic difference **only in a contracting regime**,
and k = 0.75 is where these policies are stable. On a DIVERGING trajectory a
3.8e-05 discrepancy amplifies at roughly 1.4x per control step -- the measured
growth constant -- and reaches O(1) in about thirty steps. There, a genuine
rounding difference and a semantic one both grow, and non-growth is unavailable as
a discriminator. The conclusion here is "equivalent on a stable trajectory", not
"equivalent". **But they are not bit-identical**, and bit-exact replay is the
property the paired verdict rests on. Two ways to scale one quantity, differing in
the last few digits, is the pipeline-duplication hazard this project has recorded
twice -- so the weight-scaling path was deleted in favour of sbel-pc's env var
rather than kept as an equivalent alternative.

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
    # SEED PASS-THROUGH. Without it every sweep draws the SAME eight episodes per
    # rung, so two sweeps of one policy agree by construction and measure
    # determinism rather than repeatability. Two such sweeps were run on 2026-09-07
    # and reported as a repeatability check; they were one experiment run twice.
    ap.add_argument("--repeats", type=int, default=1,
                    help="pool this many x len(CONDITIONS) episodes per rung. n=8 gives a "
                         "binomial sd of 0.177 at p=0.5, 71%% of the rate span the crossing "
                         "is interpolated inside; resolving dk*=0.05 needs n~36.")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel episode collection; verified identical to serial")
    ap.add_argument("--seed", type=int, default=0,
                    help="episode seed; vary it to draw DIFFERENT episodes per rung")
    a = ap.parse_args()
    print(f"  episode seed {a.seed}")
    print(f"  {'k':>6s} {'divergence rate':>16s}")
    kstar, prev_k, prev_r = None, None, None
    for k in sorted(a.ks):
        # screen() shells out to the collector, which inherits os.environ.
        os.environ["NEDM_ACTION_MULT"] = f"{k:.6f}"
        v, info = screen(a.ckpt, duration=a.duration_s, seed=a.seed,
                         concurrency=a.concurrency, repeats=a.repeats)
        r = info["rate"]
        import math
        se = math.sqrt(max(r*(1-r), 1e-9)/max(info["n"], 1))
        print(f"  {k:6.2f} {v:>10s} {r:5.2f} +-{se:.3f}   (unbounded {info['unbounded']}, short {info['short']})")
        if kstar is None and r >= 0.5:
            # linear interpolation between the last sub-0.5 rung and this one
            kstar = k if prev_k is None else prev_k + (0.5 - prev_r) * (k - prev_k) / (r - prev_r)
        prev_k, prev_r = k, r
    if kstar is None:
        print(f"\n  k* > {max(a.ks)}  (never crossed 0.5 in the swept range)")
    else:
        print(f"\n  k* = {kstar:.3f}   {'STABLE as deployed (k* > 1)' if kstar > 1 else 'UNSTABLE as deployed (k* < 1)'}")
    os.environ.pop("NEDM_ACTION_MULT", None)
