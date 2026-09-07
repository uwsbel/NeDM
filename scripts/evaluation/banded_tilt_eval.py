"""The banded downstream endpoint: completion by ground-pitch band, on held-out
episodes, paired across arms.

REGISTERED READINGS, before any number exists:

  C - A  banded   MUST BE FLAT. Arm C's three extra channels carry grav_world
                  permuted across episodes, so they hold no tilt information. A
                  tilt-DEPENDENT C-A means the shuffle leaked, or the pipeline
                  distinguishes arms by something other than channel content.
                  This is a DIFFERENCE, not a level, so no absolute rate voids it.

  B - C  banded   should INCREASE with |pitch| if tilt observability is the
                  mechanism. Flat -> the gain is generic to 39-D. Decreasing ->
                  incoherent.

  arm A in [0,+1)  READ FIRST, before any comparison. If A completes 100% there,
                   the band is a ceiling and any B gain in it falsifies the
                   pipeline. If A completes less, the ceiling check is VOID and
                   the band is ordinary evidence. The precondition is measured,
                   not assumed.

Paired throughout: every arm runs the identical val episodes, so the unit is the
episode and the test is McNemar on discordant pairs. Unpaired tests on these are
anti-conservative -- measured tonight at 0.012 against a correct 0.023.
"""
import argparse, glob, json, math, os, sys
from math import comb

BANDS = [(-3.1, -2.0), (-2.0, -1.0), (-1.0, 0.0), (0.0, 1.0), (1.0, 2.0), (2.0, 3.1)]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return float("nan"), float("nan")
    k = min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return p, 2 / 2 ** n          # p, and the smallest p this discordance can reach


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True,
                    help="arm=path.json, each a list of {seed, pitch, completed}")
    ap.add_argument("--min-band", type=int, default=20)
    a = ap.parse_args()

    arms = {}
    for spec in a.results:
        tag, path = spec.split("=", 1)
        arms[tag] = {r["seed"]: r for r in json.load(open(path))}
    tags = list(arms)
    common = set.intersection(*(set(v) for v in arms.values()))
    print(f"arms: {tags}   episodes common to all: {len(common)}")
    if not common:
        print("FATAL: no episodes shared across arms", file=sys.stderr); return 2

    def band_of(p):
        for lo, hi in BANDS:
            if lo <= p < hi:
                return (lo, hi)
        return None

    # THE FLAT BAND HAS TWO FORMS AND NEITHER IS AN ABSENCE.
    #
    #   A completes 100%  -> LOGICAL impossibility. B > A is arithmetically
    #                        impossible, so a reported gain is a PIPELINE BUG.
    #                        Not statistical, so band size does not enter: equally
    #                        binding at n=40 and n=124.
    #
    #   A completes < 100% -> STATISTICAL null. B-A should be ~0 where tilt cannot
    #                         matter. This is the specificity test, and unlike the
    #                         impossibility it gains power as the band grows.
    #
    # An earlier version printed LIVE or VOID, which made the check disappear
    # exactly when A missed -- and created a perverse incentive, since a thin band
    # is easier to sweep and so more likely to read LIVE while carrying least
    # evidence. The two forms cover for each other: thin bands favour the logical
    # form, thick ones the statistical one, and no band size leaves you without a
    # check. There is therefore nothing to choose.
    if "A" in arms:
        g = [s_ for s_ in common if band_of(arms["A"][s_]["pitch"]) == (0.0, 1.0)]
        k = sum(arms["A"][s_]["completed"] for s_ in g)
        print(f"\nFLAT BAND [0.0,+1.0)   arm A completes {k} of {len(g)}")
        if g and k == len(g):
            print("  form: LOGICAL IMPOSSIBILITY -- A is at ceiling, so any B gain here")
            print("        is arithmetically impossible and indicates a pipeline bug.")
            print("        Binding regardless of band size.")
            if "B" in arms:
                gain = sum(arms["B"][s_]["completed"] for s_ in g) - k
                print(f"        B - A in this band: {gain:+d}  "
                      f"{'-> PIPELINE BUG' if gain > 0 else '-> consistent'}")
        else:
            print("  form: STATISTICAL NULL -- A is not at ceiling, so this band tests")
            print("        specificity: B-A should be ~0 where tilt cannot matter.")
            if "B" in arms:
                b = sum(1 for s_ in g if arms["A"][s_]["completed"] and not arms["B"][s_]["completed"])
                c = sum(1 for s_ in g if not arms["A"][s_]["completed"] and arms["B"][s_]["completed"])
                p_, fl = mcnemar(b, c)
                print(f"        B vs A discordance {c}:{b}   McNemar p={p_:.4f}   floor={fl:.2g}")

    print(f"\n{'band':14}{'n':>5}" + "".join(f"{t:>8}" for t in tags))
    counts = {}
    for lo, hi in BANDS:
        g = sorted(s for s in common if band_of(arms[tags[0]][s]["pitch"]) == (lo, hi))
        counts[(lo, hi)] = g
        if not g:
            continue
        rates = [100 * sum(arms[t][s]["completed"] for s in g) / len(g) for t in tags]
        flag = "   THIN" if len(g) < a.min_band else ""
        print(f"  [{lo:+.1f},{hi:+.1f})  {len(g):>4}" + "".join(f"{r:>7.1f}%" for r in rates) + flag)

    for x, y in (("C", "A"), ("B", "C"), ("B", "A")):
        if x not in arms or y not in arms:
            continue
        exp = {"CA": "MUST be flat", "BC": "should grow with |pitch|",
               "BA": "information + dimensionality"}[x + y]
        print(f"\n{x} - {y}   ({exp})")
        print(f"  {'band':14}{'n':>5}{'b':>5}{'c':>5}{'delta':>9}{'McNemar':>10}{'floor':>10}")
        for lo, hi in BANDS:
            g = counts[(lo, hi)]
            if not g:
                continue
            b = sum(1 for s in g if arms[y][s]["completed"] and not arms[x][s]["completed"])
            c = sum(1 for s in g if not arms[y][s]["completed"] and arms[x][s]["completed"])
            d = 100 * (sum(arms[x][s]["completed"] for s in g)
                       - sum(arms[y][s]["completed"] for s in g)) / len(g)
            p, fl = mcnemar(b, c)
            ps = f"{p:.4f}" if p == p else "   --"
            fs = f"{fl:.2g}" if fl == fl else "   --"
            print(f"  [{lo:+.1f},{hi:+.1f})  {len(g):>4}{b:>5}{c:>5}{d:>+8.1f}{ps:>10}{fs:>10}")
    print("\nfloor = the smallest two-sided p that discordance count can reach.")
    print("A band whose floor is near its p carries little evidence however lopsided.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
