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


def rate(d, lo=None, hi=None):
    sel = [d[e] for e in episodes if lo is None or lo <= d[e]["pitch"] < hi]
    return (100.0 * sum(x["completed"] for x in sel) / len(sel), len(sel)) if sel else (float("nan"), 0)


print("OVERALL, seed as the unit")
print(f"  {'arm':6s} {'n':>3s}  {'mean%':>7s} {'sd':>6s}   per-seed rates")
means = {}
for k, ds in arms.items():
    rs = [rate(d)[0] for d in ds]
    m = sum(rs) / len(rs)
    sd = math.sqrt(sum((x - m) ** 2 for x in rs) / (len(rs) - 1)) if len(rs) > 1 else float("nan")
    means[k] = rs
    print(f"  {k:6s} {len(rs):3d}  {m:7.1f} {sd:6.1f}   " + " ".join(f"{x:.1f}" for x in rs))

print("\nCONTRASTS  (difference of arm means, exact permutation over seed labels)")
print(f"  {'contrast':12s} {'delta':>7s} {'p':>8s} {'floor':>8s}   reading")
for x, y in itertools.combinations(arms, 2):
    xa, xb = means[x], means[y]
    d = sum(xb) / len(xb) - sum(xa) / len(xa)
    p, floor = perm_p(xa, xb)
    note = ("cannot resolve: p is at its floor" if abs(p - floor) < 1e-9
            else "separates" if p < 0.05 else "not separated at this seed count")
    print(f"  {y}-{x:9s} {d:+7.1f} {p:8.4f} {floor:8.4f}   {note}")

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

print("\nThe +- column is the between-SEED sd, not a standard error over episodes.")
print("An arm difference smaller than that sd is not measurable at this seed count,")
print("however many episodes each cell contains.")
