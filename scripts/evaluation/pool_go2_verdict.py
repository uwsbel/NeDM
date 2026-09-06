"""Pool per-machine verdict strata and score the criterion once.

STRATIFIED, NOT MERGED. Episodes replay bit-identically only on the machine that
collected them, so each box scores its own and the paired differences combine here.
Machine cancels within each pair, which makes this an ordinary stratified paired
design.

POOLS PER-EPISODE DIFFERENCES, NEVER SUMMARY STATISTICS. A median of medians is not
the median of the pool, and at these stratum sizes the difference is not negligible.

REPORTS PER-MACHINE FIRST. These two collections have already diverged on
inadmissibility (46x), stand-up failures (2x) and tilt fragility; a pooled number
that buries a fourth disagreement would be the same part-whole error this study keeps
finding.
"""
from __future__ import annotations
import argparse, json
from math import comb
import numpy as np

rng = np.random.default_rng(0)


def exact_median_ci(x, alpha=0.05):
    x = np.sort(np.asarray(x)); n = len(x)
    ks = [i for i in range(1, n // 2 + 1)
          if 2 * sum(comb(n, j) for j in range(i)) / 2 ** n <= alpha]
    if not ks:
        return float("nan"), float("nan"), 0.0
    k = max(ks)
    return float(x[k - 1]), float(x[n - k]), 1 - 2 * sum(comb(n, j) for j in range(k)) / 2 ** n


def mcnemar(b_bad, t_bad):
    n01 = int(np.sum(~b_bad & t_bad)); n10 = int(np.sum(b_bad & ~t_bad)); d = n01 + n10
    if d == 0:
        return n01, n10, 1.0, 1.0
    k = min(n01, n10)
    return n01, n10, min(1.0, 2 * sum(comb(d, i) for i in range(k + 1)) / 2 ** d), 2 / 2 ** d


def ratios(p):
    """ratio = achieved/cmd = 1 + err/cmd. Derived rather than required, so a stratum
    that did not emit ratios is still poolable. Verified against a stratum that did."""
    return (p.get("baseline_ratio", 1 + p["baseline_err"] / p["cmd"]),
            p.get("treated_ratio", 1 + p["treated_err"] / p["cmd"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    a = ap.parse_args()
    strata = []
    for f in a.summaries:
        d = json.load(open(f))
        strata.append((d.get("machine", f), d))
    # CONVENTION CHECK BEFORE POOLING. Both emitters must mean the same thing by
    # `difference`, or the pooled median is the average of two different questions.
    # Verified rather than assumed: difference == treated - baseline in each, and
    # every error is positive so signed and absolute coincide in this cell.
    for name, d in strata:
        P = d["pairs"]
        bad = [p for p in P
               if abs(p["difference"] - (p["treated_err"] - p["baseline_err"])) > 1e-12]
        if bad:
            raise SystemExit(f"{name}: `difference` is not treated - baseline; refusing to pool")
        if any(p["baseline_err"] < 0 or p["treated_err"] < 0 for p in P):
            raise SystemExit(f"{name}: signed errors present, so signed and absolute "
                             f"conventions differ; reconcile before pooling")

    print("=== PER-MACHINE, reported first and on purpose ===\n")
    print(f"  {'machine':<18}{'pairs':>7}{'dropped':>9}{'median diff':>14}{'wrong-way':>22}")
    allD = []; allB = []; allT = []; allM = []
    for name, d in strata:
        P = d["pairs"]
        D = np.array([p["difference"] for p in P])
        br = np.array([ratios(p)[0] < 0 for p in P]); tr = np.array([ratios(p)[1] < 0 for p in P])
        nd = d.get("n_dropped", d.get("survivorship", {}).get("n_dropped"))
        print(f"  {name:<18}{len(P):>7}{str(nd):>9}{np.median(D):>+13.4f}"
              f"{f'{br.mean():.0%} -> {tr.mean():.0%}':>22}")
        allD.append(D); allB.append(br); allT.append(tr); allM += [name] * len(P)
    D = np.concatenate(allD); B = np.concatenate(allB); T = np.concatenate(allT)
    M = np.array(allM)

    # Do the strata disagree beyond chance? Distribution-free permutation on the
    # difference of medians, since n is small and the differences are not normal.
    if len(strata) == 2:
        obs = abs(np.median(allD[0]) - np.median(allD[1]))
        null = np.array([abs(np.median(s[:len(allD[0])]) - np.median(s[len(allD[0]):]))
                         for s in (rng.permutation(D) for _ in range(20000))])
        p_int = float((null >= obs).mean())
        print(f"\n  machine-by-treatment interaction: |difference of medians| = {obs:.4f}")
        print(f"    permutation p = {p_int:.3f}  ->  "
              f"{'strata differ beyond chance' if p_int < 0.05 else 'not distinguishable from chance at this n'}")

    print(f"\n=== POOLED, n = {len(D)} ===\n")
    lo, hi, cov = exact_median_ci(D)
    med = float(np.median(D))
    n01, n10, p, pmin = mcnemar(B, T)
    print(f"  PRIMARY   median paired difference {med:+.4f} m/s")
    print(f"            exact 95% CI [{lo:+.4f}, {hi:+.4f}] (coverage {cov:.3f})")
    print(f"            threshold <= -0.020 with a CI excluding 0")
    print(f"  ANCHOR    wrong-way {B.mean():.0%} -> {T.mean():.0%}   discordant {n10}/{n01}")
    print(f"            McNemar p {p:.4f}, smallest attainable {pmin:.4f}"
          f"{'  VACUOUS' if pmin > 0.05 else ''}")
    sv = [d.get("survivorship", {}) for _, d in strata]
    print(f"\n  SURVIVORSHIP (pooled across strata)")
    for (name, _), s in zip(strata, sv):
        # The two emitters name these differently. Accept both rather than
        # silently printing one stratum, which is how a missing half looks like
        # a shorter table.
        bs = s.get("baseline_median_err_survivors",
                   s.get("survivors_median_baseline_abs_err"))
        bd = s.get("baseline_median_err_dropped",
                   s.get("dropped_median_baseline_abs_err"))
        if bs is not None and bd is not None:
            print(f"    {name:<18} survivors {bs:+.4f}   dropped {bd:+.4f}   gap {bd-bs:+.4f}")

    fail = []
    if not (med <= -0.020 and hi < 0):
        fail.append("primary: median paired difference did not reach -0.020 m/s with a CI excluding 0")
    if pmin > 0.05:
        fail.append(f"anchor NOT EVALUABLE: only {n01+n10} discordant pairs")
    elif p < 0.05 and n01 > n10:
        fail.append("anchor: wrong-way fraction increased SIGNIFICANTLY")
    print()
    for f in fail:
        print(f"  FAIL -- {f}")
    print("\nVERDICT: " + ("FAIL" if fail else "PASS") + " (rigid terrain only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
