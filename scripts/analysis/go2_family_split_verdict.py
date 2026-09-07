#!/usr/bin/env python3
"""Registered endpoints E1/E2/E3 for the arm-A family split, plus the
unregistered five-way breakdown.

Reads a run_go2_finetune_verdict.py summary JSON (its `pairs` array) and the
baseline episode CSVs the pairs came from. Every threshold here is fixed by
docs/state/decisions/go2-family-split-prereg.md and must not be edited to suit
a result -- if a criterion is wrong, amend the prereg and say so.
"""
import argparse, csv, glob, json, os, re, sys
import numpy as np
from math import comb

STRAIGHT = ("constant", "vel_step")
TURNING  = ("arc", "weave", "yaw_step")
E2_CRITERION = -0.020            # m/s, the line this work was built against
E3_TIE_BAND  = 0.20              # within 20% of the smaller separation = tie
SCORED_ROWS  = 1000              # must match the harness


def exact_median_ci(x, alpha=0.05):
    x = np.sort(np.asarray(x, float)); n = len(x)
    ks = [i for i in range(1, n // 2 + 1)
          if 2 * sum(comb(n, j) for j in range(i)) / 2 ** n <= alpha]
    if not ks:
        return float("nan"), float("nan"), 0.0
    k = max(ks)
    cov = 1 - 2 * sum(comb(n, j) for j in range(k)) / 2 ** n
    return float(x[k - 1]), float(x[n - k]), cov


def report(name, d):
    d = np.asarray(d, float)
    if len(d) == 0:
        print(f"  {name:<12} n=0   (no surviving pairs)")
        return None
    lo, hi, cov = exact_median_ci(d)
    med = float(np.median(d))
    if not np.isfinite(lo):
        print(f"  {name:<12} n={len(d):<3} median {med:+.5f}   "
              f"CI UNDEFINED (n too small for a 95% order-statistic interval)")
        return dict(n=len(d), median=med, lo=None, hi=None, coverage=0.0)
    excl0 = "excludes 0" if (lo > 0 or hi < 0) else "INCLUDES 0"
    print(f"  {name:<12} n={len(d):<3} median {med:+.5f}   "
          f"95% CI [{lo:+.4f}, {hi:+.4f}] (cov {cov:.3f})  {excl0}")
    return dict(n=len(d), median=med, lo=lo, hi=hi, coverage=cov)


def body_motion(csv_path):
    """Per-episode median |v_body| over the scored window, or None."""
    try:
        rows = list(csv.DictReader(open(csv_path)))
    except Exception:
        return None
    if len(rows) < SCORED_ROWS:
        return None
    w = rows[-SCORED_ROWS:]
    try:
        V = np.array([[float(r["vel_body_x_mps"]), float(r["vel_body_y_mps"])] for r in w])
    except Exception:
        return None
    if not np.isfinite(V).all():
        return None
    return float(np.median(np.linalg.norm(V, axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--baseline-root", required=True)
    ap.add_argument("--out-json")
    a = ap.parse_args()

    S = json.load(open(a.summary_json))
    pairs = S["pairs"]
    print(f"summary {a.summary_json}")
    print(f"  {len(pairs)} surviving pairs, cell {S.get('cell')}, machine {S.get('machine')}")
    print(f"  aggregate median {S['median_paired_difference']:+.5f} "
          f"CI {S['exact_ci']}\n")

    fam = {}
    for p in pairs:
        fam.setdefault(p["family"], []).append(p["difference"])

    print("--- E1: does the family split replicate? (registered) ---")
    s = [p["difference"] for p in pairs if p["family"] in STRAIGHT]
    t = [p["difference"] for p in pairs if p["family"] in TURNING]
    rs = report("straight", s)
    rt = report("turning", t)

    e1 = bool(rs and rt and rs["lo"] is not None and rt["lo"] is not None
              and rs["hi"] < 0 and rt["lo"] > 0)
    print(f"\n  E1 {'MET' if e1 else 'NOT MET'} -- requires straight CI entirely "
          f"below 0 AND turning CI entirely above 0")
    if rs and rt and rs["lo"] is not None and rt["lo"] is not None and not e1:
        if rs["median"] < 0 and rt["median"] > 0:
            print("     signs are as predicted; at least one interval does not exclude zero")
        else:
            print("     A SIGN REVERSED -- this refutes the discovery, per the prereg")

    print(f"\n--- E2: is the straight-line effect powered past {E2_CRITERION:+.3f}? (registered) ---")
    if rs and rs["hi"] is not None:
        e2 = rs["hi"] < E2_CRITERION
        print(f"  straight upper bound {rs['hi']:+.4f} vs criterion {E2_CRITERION:+.3f}"
              f"  -> {'MET' if e2 else 'NOT MET'}")
        if not e2 and rs["median"] < E2_CRITERION:
            print("  point estimate beats the criterion, interval does not exclude it:")
            print("  the honest claim stays 'significantly non-zero, about "
                  f"{abs(rs['median'] / E2_CRITERION):.1f}x the criterion'.")
    else:
        e2 = False
        print("  NOT MEASURABLE -- no straight-line interval")

    print("\n--- E3: family split vs body-motion split (registered) ---")
    # Resolve (family, index) -> CSV through the SIDECAR, exactly as the harness does.
    # Globbing the directory name works only on this box's one-dir-per-episode layout
    # and matches nothing on dorm-pc's flat tree -- the same defect that once silently
    # dropped 3,003 episodes. command_family + the episode_id suffix travel with the
    # episode through consolidation; the directory name does not.
    index = {}
    for jp in glob.glob(os.path.join(a.baseline_root, "**", "episodes", "*.json"),
                        recursive=True):
        try:
            m = json.load(open(jp))
        except Exception:
            continue
        f = m.get("command_family")
        hit = re.search(r"(\d+)$", str(m.get("episode_id", "")))
        if not f or not hit:
            continue
        cp = os.path.splitext(jp)[0] + ".csv"
        if os.path.exists(cp):
            index[(f, int(hit.group(1)))] = cp

    mot, missing = {}, 0
    for p in pairs:
        cp = index.get((p["family"], p["idx"]))
        m = body_motion(cp) if cp else None
        if m is None:
            missing += 1
        else:
            mot[(p["family"], p["idx"])] = m

    if missing:
        print(f"  {missing}/{len(pairs)} pairs had no readable baseline CSV")
    if len(mot) < len(pairs):
        print("  E3 NOT EVALUATED -- the motion split needs every surviving pair;\n"
              "  a partial split is a different comparison than the registered one.")
        e3 = None
    else:
        vals = np.array([mot[(p["family"], p["idx"])] for p in pairs])
        d = np.array([p["difference"] for p in pairs])
        cut = float(np.median(vals))
        lo_h, hi_h = d[vals <= cut], d[vals > cut]
        sep_m = abs(np.median(lo_h) - np.median(hi_h))
        sep_f = abs(np.median(s) - np.median(t)) if s and t else float("nan")
        print(f"  motion split at |v_body| = {cut:.4f} m/s "
              f"(n={len(lo_h)} slow, {len(hi_h)} fast)")
        print(f"    slow  median {np.median(lo_h):+.5f}")
        print(f"    fast  median {np.median(hi_h):+.5f}")
        print(f"  separation  family {sep_f:.5f}   motion {sep_m:.5f}")
        smaller = min(sep_f, sep_m)
        if abs(sep_f - sep_m) <= E3_TIE_BAND * smaller:
            e3 = "tie"
            print(f"  E3 = TIE (within {E3_TIE_BAND:.0%} of the smaller). The two splits are\n"
                  "  collinear in this stratum and cannot be separated without conditions\n"
                  "  that break the correlation -- turning episodes ARE the fast ones here.")
        else:
            e3 = "motion" if sep_m > sep_f else "family"
            print(f"  E3 = {e3.upper()} split separates better by "
                  f"{abs(sep_f - sep_m) / smaller:.0%}")

    print("\n--- five-way family breakdown (NOT registered, descriptive) ---")
    five = {}
    for k in sorted(fam):
        five[k] = report(k, fam[k])
    print("  Not a registered endpoint: read as structure to follow up, not as a result.")

    out = dict(summary=a.summary_json, n_pairs=len(pairs),
               aggregate_median=S["median_paired_difference"],
               straight=rs, turning=rt, five_way=five,
               E1_met=e1, E2_met=e2, E3=e3,
               E2_criterion=E2_CRITERION, e3_tie_band=E3_TIE_BAND)
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=2)
        print(f"\nwrote {a.out_json}")


if __name__ == "__main__":
    main()
