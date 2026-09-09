"""Compare arms with the SURROGATE TRAINING RUN as the experimental unit.

`banded_tilt_eval.py` treats the episode as the unit: it fixes one surrogate per
arm and asks whether 536 paired episodes discriminate the resulting policies. They
do, overwhelmingly -- and the answer reverses when you change the surrogate seed.
Arm A at dw6 completes 23.1% on seed 1 and 69.0% on seed 2, a swing wider than any
between-arm gap, and B-A comes back p<0.0001 in BOTH directions on the two seeds.

That is not a contradiction to resolve by choosing a seed. The policy is a
deterministic function of the surrogate, the surrogate is the random draw, and
conditioning on one draw as if it were the population is the wrong unit of analysis.

So: per (arm, seed) collapse the episodes to one rate, and let the SEED be the
replicate. The episode set is common to every cell, so the design stays paired
across arms; what changes is that the error term is now between-seed variance
instead of between-episode variance. The p-values get much larger. That is the
point -- the old ones were measuring how many episodes there are.

Significance is by exact permutation of seed labels between the two arms, which
needs no distributional assumption and is honest about how few seeds there are:
with n seeds per arm the smallest attainable two-sided p is 2 / C(2n, n), and that
floor is printed beside every contrast so a "p = 0.10" that could never have gone
lower is not mistaken for weak evidence of a null.

That floor is a hard design constraint, and it is why the seed counts are what
they are:

    seeds per arm    smallest possible two-sided p
        2                0.3333
        3                0.1000
        4                0.0286   <- first count that can reach 0.05
        5                0.0079
        6                0.0022

**Two seeds per arm cannot produce a p below 0.33 no matter how large the effect
or how many episodes are scored.** Every A/B/C comparison run so far sat at that
count, so at the seed level it had no power at all -- the p<0.0001 those runs
reported came entirely from treating 536 episodes as 536 independent replicates
of the arm. Four seeds per arm is the minimum at which this test can ever return
a significant result.
"""
import argparse, itertools, json, math, os, sys
from collections import defaultdict

BANDS = [(-3.1, -2.0), (-2.0, -1.0), (-1.0, 0.0), (0.0, 1.0), (1.0, 2.0), (2.0, 3.1)]


def key(r):
    """Same keying rule as the banded evaluator, and the same refusal."""
    return r.get("episode_id") or (r["seed"], r["pitch"], r.get("roll"), r.get("family"))


def load(path):
    recs = json.load(open(path))
    d = {key(r): r for r in recs}
    if len(d) != len(recs):
        raise SystemExit(f"FATAL: {path} has {len(recs)} records but {len(d)} distinct "
                         f"keys -- pairing would be silently wrong")
    return d


def perm_p(xa, xb):
    """Exact two-sided permutation test on the difference of means over seed labels."""
    obs = abs(sum(xb) / len(xb) - sum(xa) / len(xa))
    pool = list(xa) + list(xb)
    n = len(xa)
    hits = tot = 0
    for combo in itertools.combinations(range(len(pool)), n):
        s = set(combo)
        a = [pool[i] for i in s]
        b = [pool[i] for i in range(len(pool)) if i not in s]
        tot += 1
        if abs(sum(b) / len(b) - sum(a) / len(a)) >= obs - 1e-12:
            hits += 1
    return hits / tot, 2.0 / math.comb(len(pool), n)


ap = argparse.ArgumentParser()
ap.add_argument("--arm", action="append", required=True,
                help="NAME=file1.json,file2.json,... one file per surrogate seed")
ap.add_argument("--min-seeds", type=int, default=3,
                help="refuse below this; 2 seeds give a variance estimate worth nothing")
ap.add_argument("--metric", default="completed",
                help="'completed' scores a RATE (percent of episodes). Any other name is "
                     "averaged as a continuous per-episode field, e.g. mae_vx for CRM "
                     "velocity tracking error, where LOWER IS BETTER and the sign of a "
                     "contrast therefore reads the opposite way.")
ap.add_argument("--lower-is-better", action="store_true",
                help="flip the reading of contrast signs; set automatically for mae_*")
a = ap.parse_args()

arms = {}
for spec in a.arm:
    name, files = spec.split("=", 1)
    arms[name] = [load(f) for f in files.split(",")]

ns = {k: len(v) for k, v in arms.items()}
print("arm seeds: " + "  ".join(f"{k}={v}" for k, v in ns.items()))
thin = [k for k, v in ns.items() if v < a.min_seeds]
if thin:
    print(f"\n*** {', '.join(thin)} have fewer than {a.min_seeds} seeds. The contrast below "
          f"is reported but the between-seed variance is not estimable to any useful "
          f"precision, and a null here means nothing. ***")

# The episode set must be identical across every cell or the pairing is a fiction.
sets = [frozenset(d) for v in arms.values() for d in v]
if len(set(sets)) != 1:
    sizes = sorted({len(s) for s in sets})
    raise SystemExit(f"FATAL: cells do not share an episode set (sizes {sizes}); "
                     f"the comparison would not be paired")
episodes = sorted(next(iter(sets)), key=str)
print(f"episodes per cell: {len(episodes)}\n")


LOWER = a.lower_is_better or a.metric.startswith("mae_")


def rate(d, lo=None, hi=None):
    """Percent completed, or the mean of a continuous field.

    Episodes missing the metric are DROPPED, not counted as zero: on CRM an episode
    with no tracking score is one the collector failed to produce, and scoring it as
    a perfect zero error would reward exactly the failure it represents.
    """
    sel = [d[e] for e in episodes
           if lo is None or (d[e].get("pitch") is not None and lo <= d[e]["pitch"] < hi)]
    if a.metric == "completed":
        return (100.0 * sum(x["completed"] for x in sel) / len(sel), len(sel)) if sel else (float("nan"), 0)
    vals = [x[a.metric] for x in sel if a.metric in x and x[a.metric] == x[a.metric]]
    return (sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)


print(f"OVERALL, seed as the unit   metric={a.metric}"
      + ("   (LOWER IS BETTER)" if LOWER else ""))
_lbl = "mean%" if a.metric == "completed" else "mean"
print(f"  {'arm':6s} {'n':>3s}  {_lbl:>9s} {'sd':>8s}   per-seed values")
means = {}
for k, ds in arms.items():
    rs = [rate(d)[0] for d in ds]
    m = sum(rs) / len(rs)
    sd = math.sqrt(sum((x - m) ** 2 for x in rs) / (len(rs) - 1)) if len(rs) > 1 else float("nan")
    means[k] = rs
    _f = (lambda v: f"{v:9.1f}") if a.metric == "completed" else (lambda v: f"{v:9.4f}")
    _g = (lambda v: f"{v:8.1f}") if a.metric == "completed" else (lambda v: f"{v:8.4f}")
    _h = (lambda v: f"{v:.1f}") if a.metric == "completed" else (lambda v: f"{v:.4f}")
    print(f"  {k:6s} {len(rs):3d}  {_f(m)} {_g(sd)}   " + " ".join(_h(x) for x in rs))

print("\nCONTRASTS  (difference of arm means, exact permutation over seed labels)")
print(f"  {'contrast':12s} {'delta':>7s} {'p':>8s} {'floor':>8s}   reading")
for x, y in itertools.combinations(arms, 2):
    xa, xb = means[x], means[y]
    d = sum(xb) / len(xb) - sum(xa) / len(xa)
    p, floor = perm_p(xa, xb)
    # p == floor means the split is as extreme as this seed count PERMITS. Whether
    # that is a result depends entirely on where the floor sits:
    #   floor >= 0.05  -> the design could never have reached significance; the data
    #                     is maximally lopsided and still says nothing (the n=2 case,
    #                     floor 0.333).
    #   floor <  0.05  -> it IS a pass, and the strongest one available here.
    # Conflating the two reported a genuine separation as "cannot resolve".
    at_floor = abs(p - floor) < 1e-9
    if floor >= 0.05:
        note = ("cannot resolve: p is at its floor (%.3f), which is above 0.05 -- this "
                "seed count could not separate anything" % floor) if at_floor else \
               "not separated, and the floor %.3f is above 0.05 anyway" % floor
    elif p < 0.05:
        note = "SEPARATES" + (" (maximal for this seed count)" if at_floor else "")
    else:
        note = "not separated at this seed count"
    if LOWER and p < 0.05 and floor < 0.05:
        note += ("  -- %s is BETTER" % (y if d < 0 else x))
    _d = f"{d:+9.1f}" if a.metric == "completed" else f"{d:+9.4f}"
    print(f"  {y}-{x:9s} {_d} {p:8.4f} {floor:8.4f}   {note}")

print("\nBY PITCH BAND, seed as the unit")
for lo, hi in BANDS:
    row = []
    for k, ds in arms.items():
        rs = [rate(d, lo, hi)[0] for d in ds]
        m = sum(rs) / len(rs)
        sd = math.sqrt(sum((v - m) ** 2 for v in rs) / (len(rs) - 1)) if len(rs) > 1 else float("nan")
        row.append(f"{k} {m:5.1f}+-{sd:4.1f}")
    n = rate(next(iter(arms.values()))[0], lo, hi)[1]
    print(f"  [{lo:+.1f},{hi:+.1f})  n={n:4d}   " + "   ".join(row))

if LOWER:
    print("\nLOWER IS BETTER for this metric: a NEGATIVE delta means the second arm tracks")
    print("more accurately. Do not read these signs as if they were completion rates.")
print("\nThe +- column is the between-SEED sd, not a standard error over episodes.")
print("An arm difference smaller than that sd is not measurable at this seed count,")
print("however many episodes each cell contains.")
