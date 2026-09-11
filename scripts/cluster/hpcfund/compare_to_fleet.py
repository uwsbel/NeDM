"""Pair a cluster scoring run against the NVIDIA fleet's run of the same policy.

The question is not "did it crash". It is whether Chrono built with
CHRONO_GPU_BACKEND=HIP on an MI210 produces the same PHYSICS as the CUDA build
the desktop fleet runs, because a backend that quietly integrates differently
would give every arm a different score and silently invalidate the comparison
the scores exist to support.

Two failure modes, opposite in shape, and this reports both:

  TOO DIFFERENT   a systematic shift in mean mae_vx, or a per-episode spread far
                  wider than the metric's own resolution, means the backends do
                  not agree and cluster numbers cannot be compared with fleet
                  numbers.

  TOO IDENTICAL   bit-identical per-episode values would NOT be reassuring. SPH
                  runs thousands of floating-point reductions per step whose
                  order differs between CUDA and HIP, so a genuine re-simulation
                  must diverge at some rate. A max |delta| of ~0 means the run
                  did not actually re-simulate -- it read a cached result, or
                  scored the reference file, or every episode failed identically.

So the pass condition is an aggregate that lands in the fleet's range WITH a
nonzero per-episode spread, not an exact match.

Pairing is by episode_id, which the scorer derives from the last four components
of csv_path. Those are preserved verbatim when the index is restaged for the
cluster, so the ids are byte-identical across machines and the pairing is exact
rather than positional.
"""
import argparse
import json
import math

ap = argparse.ArgumentParser()
ap.add_argument("--cluster", required=True)
ap.add_argument("--reference", required=True)
ap.add_argument("--metric", default="mae_vx")
a = ap.parse_args()


def load(p):
    return {r["episode_id"]: r for r in json.load(open(p))}


C, R = load(a.cluster), load(a.reference)
m = a.metric

print("cluster   %s  (%d records)" % (a.cluster, len(C)))
print("reference %s  (%d records)" % (a.reference, len(R)))

only_c, only_r = set(C) - set(R), set(R) - set(C)
print("episode ids: %d shared, %d cluster-only, %d reference-only"
      % (len(set(C) & set(R)), len(only_c), len(only_r)))
for label, s in (("cluster-only", only_c), ("reference-only", only_r)):
    for e in sorted(s)[:5]:
        print("   %s: %s" % (label, e))

both = sorted(e for e in set(C) & set(R) if m in C[e] and m in R[e])
print("\nscored on BOTH machines: %d" % len(both))
print("  scored cluster-side only: %d" % len([e for e in set(C) & set(R)
                                              if m in C[e] and m not in R[e]]))
print("  scored reference-side only: %d" % len([e for e in set(C) & set(R)
                                                if m not in C[e] and m in R[e]]))
if not both:
    raise SystemExit("no episode scored on both machines: nothing to compare")

cv = [C[e][m] for e in both]
rv = [R[e][m] for e in both]
n = len(both)
mc, mr = sum(cv) / n, sum(rv) / n

# Aggregate over EVERY scored episode on each side, which is the number the
# sweep actually reports, alongside the paired means, which are like for like.
allc = [r[m] for r in C.values() if m in r]
allr = [r[m] for r in R.values() if m in r]
print("\n-- aggregate %s --" % m)
print("  cluster    all scored (n=%3d): %.4f" % (len(allc), sum(allc) / len(allc)))
print("  reference  all scored (n=%3d): %.4f" % (len(allr), sum(allr) / len(allr)))
print("  cluster    paired     (n=%3d): %.4f" % (n, mc))
print("  reference  paired     (n=%3d): %.4f" % (n, mr))
print("  paired mean delta            : %+.4f  (%+.2f%%)"
      % (mc - mr, 100 * (mc - mr) / mr if mr else float("nan")))

d = [c - r for c, r in zip(cv, rv)]
ad = sorted(abs(x) for x in d)
sd = math.sqrt(sum((x - (sum(d) / n)) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
print("\n-- per-episode delta (cluster - reference) --")
print("  mean %+.4f   sd %.4f" % (sum(d) / n, sd))
print("  |delta|  median %.4f   p90 %.4f   max %.4f"
      % (ad[n // 2], ad[min(n - 1, int(0.9 * n))], ad[-1]))

sc = math.sqrt(sum((x - mc) ** 2 for x in cv) / (n - 1)) if n > 1 else 0.0
sr = math.sqrt(sum((x - mr) ** 2 for x in rv) / (n - 1)) if n > 1 else 0.0
if sc > 0 and sr > 0:
    cov = sum((c - mc) * (r - mr) for c, r in zip(cv, rv)) / (n - 1)
    print("  pearson r across episodes: %.4f" % (cov / (sc * sr)))
print("  spread of the metric itself: cluster sd %.4f, reference sd %.4f" % (sc, sr))

# ---- systematic bias ----------------------------------------------------
# The mean delta alone cannot distinguish "both backends scatter symmetrically
# around the same answer" from "the cluster is consistently a little higher on
# every episode". Those have completely different consequences: the first is
# noise and arms stay comparable across machines, the second is a bias that
# shifts every arm in the same direction and silently changes which arm wins.
# So test the SIGN split as well as the magnitude. A 50/50 split with a wide
# spread is healthy; a 70/7 split is a one-directional offset even when the
# mean looks small next to the metric's own scale.
up = sum(1 for x in d if x > 0)
dn = sum(1 for x in d if x < 0)
eq = n - up - dn
print("\n-- systematic bias --")
print("  cluster higher on %d, lower on %d, identical on %d" % (up, dn, eq))
nz = up + dn
if nz:
    z = (up - nz / 2.0) / math.sqrt(nz / 4.0)
    p = math.erfc(abs(z) / math.sqrt(2))
    print("  sign test: z %+.2f  p %.3f%s" % (z, p, "  <-- ONE-DIRECTIONAL" if p < 0.05 else ""))
se = sd / math.sqrt(n) if n > 1 and sd > 0 else 0.0
if se:
    t = (sum(d) / n) / se
    pt = math.erfc(abs(t) / math.sqrt(2))
    print("  paired mean offset: %+.4f +/- %.4f (1 se)  t %+.2f  p %.3f%s"
          % (sum(d) / n, se, t, pt, "  <-- SIGNIFICANT OFFSET" if pt < 0.05 else ""))
# Scale the offset against the spread of the metric across episodes, which is
# what the arms are being separated by in the first place. An offset that is a
# few percent of that spread cannot reorder arms; one comparable to it can.
if sr:
    print("  offset as a fraction of the reference episode-to-episode sd (%.4f): %.1f%%"
          % (sr, 100 * abs(sum(d) / n) / sr))

# ---- episode flip rate --------------------------------------------------
# An episode "flips" when it scores on one machine and not the other. The
# desktop fleet measures 0.93% across five distinct pychrono builds, so that is
# the bar: a cluster flip rate near it is ordinary build-to-build variation, a
# much higher one means the AMD backend is failing episodes the CUDA one
# completes, which is a correctness problem and not a tolerance question.
shared = set(C) & set(R)
flips = [e for e in shared if (m in C[e]) != (m in R[e])]
print("\n-- episode flip rate --")
print("  %d of %d shared episodes scored on one machine only (%.2f%%)"
      % (len(flips), len(shared), 100 * len(flips) / max(len(shared), 1)))
print("  desktop fleet baseline across five pychrono builds: 0.93%")
for e in sorted(flips)[:6]:
    print("     %s: cluster %s, reference %s"
          % (e, "scored" if m in C[e] else "no", "scored" if m in R[e] else "no"))

print("\n-- worst 5 episodes by |delta| --")
for e in sorted(both, key=lambda e: -abs(C[e][m] - R[e][m]))[:5]:
    print("  %-52s cluster %.4f  ref %.4f  delta %+.4f"
          % (e, C[e][m], R[e][m], C[e][m] - R[e][m]))

print("\n-- reading --")
if ad[-1] < 1e-9:
    print("  SUSPICIOUS: max |delta| is ~0. CUDA and HIP reorder SPH reductions, so a")
    print("  real re-simulation cannot agree to machine precision. Check that the run")
    print("  actually simulated rather than reusing a cached or copied result.")
else:
    print("  per-episode deltas are nonzero, consistent with a genuine re-simulation")
    print("  on a different backend rather than a copied result.")
rel = abs(mc - mr) / mr if mr else float("nan")
print("  paired mean differs by %.2f%% -- judge this against the fleet's own"
      % (100 * rel))
print("  seed-to-seed spread, not against zero.")
