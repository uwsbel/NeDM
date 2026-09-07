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

    # PRECONDITION FIRST, before any comparison is printed.
    if "A" in arms:
        g = [s for s in common if band_of(arms["A"][s]["pitch"]) == (0.0, 1.0)]
        k = sum(arms["A"][s]["completed"] for s in g)
        live = g and k == len(g)
        print(f"\nPRECONDITION  arm A in [0.0,+1.0): {k} of {len(g)} complete"
              f"   -> ceiling check {'LIVE' if live else 'VOID'}")
        if live:
            # A LIVE VERDICT IS NOT SELF-INTERPRETING. The precondition is "arm A
            # completes ALL of them", so a THIN band is easier to sweep than a thick
            # one -- the check is most likely to be live exactly where it has least
            # power. "All 43" and "all 124" are very different evidence for the same
            # word, so the count travels with the verdict.
            print(f"   LIVE on {len(g)} episodes -- a thin band is easier to sweep,")
            print(f"   so read this as evidence proportional to {len(g)}, not as a binary")
            print("   any B gain in that band falsifies the pipeline")
        else:
            print("   band is ordinary evidence; the registered impossibility does not apply")

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
