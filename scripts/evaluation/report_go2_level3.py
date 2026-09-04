"""Emit the level-3 verdict mechanically, from the pre-registration.

WRITTEN BEFORE THE FINAL CHECKPOINT EXISTS, WHICH IS THE ENTIRE POINT. Every
threshold, every statistic and the order they are reported in come from
docs/state/decisions/go2-level3-preregistration.md and its three amendments. If
the verdict is computed by a script that predates the number, nobody gets to
choose how to compute it after seeing the number -- including the person who
wrote the pre-registration, which is the failure mode that amendment 1 exists to
document.

It takes the JSON written by eval_go2_rl_chrono_tracking.py (which already
contains both the replay floor and the policy, per reference) and produces:

  per reference first, pooled last          -- pooled is a statement about the mix
  the RATIO verdict                          -- registered primary
  the PAIRED ABSOLUTE DIFFERENCE verdict     -- co-reported, no denominator
  agreement, or SPLIT                        -- a disagreement is itself a finding
  slope of policy on floor, with leave-one-out range and Spearman with exact p
  cross-arm ratio comparison                 -- the only commensurable statistic

Refuses to do things the pre-registration forbids: it will not compare paired
absolute differences across arms, and it will not print a slope without its
leave-one-out range.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

# --- registered thresholds. Do not edit without amending the pre-registration. --
RATIO_BEAT = 0.90
RATIO_FAIL = 1.15
COUNT_MAJORITY = 6          # of 8
SLOPE_INDEPENDENT = 1.0 / 3  # policy error independent of reference difficulty
SLOPE_INHERITS = 2.0 / 3     # policy inherits reference difficulty
SPEARMAN_N8_P05 = 0.74       # |rho| needed for p < 0.05 at n = 8


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def exact_permutation_p(a: np.ndarray, b: np.ndarray) -> float | None:
    """Two-sided p by enumerating EVERY ordering. None above n=9.

    At n=8 there are 40320 orderings, so the exact value is cheap and a Monte
    Carlo estimate would be a needless approximation of something computable.
    """
    n = len(a)
    if n > 9:
        return None
    observed = abs(spearman(a, b))
    hits = total = 0
    for perm in itertools.permutations(range(n)):
        total += 1
        if abs(spearman(a, b[list(perm)])) >= observed - 1e-12:
            hits += 1
    return hits / total


def load_arm(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if "policy" not in data:
        raise SystemExit(f"{path} has no policy results -- it is a floor-only run")
    floor = {r["reference_id"]: r for r in data["replay_baseline"]["per_reference"]}
    rows = []
    for r in data["policy"]["per_reference"]:
        base = floor[r["reference_id"]]
        rows.append({
            "family": r["scenario_family"].removesuffix("_command").split("_", 2)[2],
            "episode_id": r["episode_id"],
            "floor": base["mean_position_error_m"],
            "policy": r["mean_position_error_m"],
            "ratio": r["mean_position_error_m"] / base["mean_position_error_m"],
            "diff": r["mean_position_error_m"] - base["mean_position_error_m"],
            "policy_steps": r["steps"],
            "floor_steps": base["steps"],
        })
    return {"path": str(path), "rows": rows,
            "reference_path": data.get("reference_path"),
            "chrono_config": data.get("chrono_config")}


def ratio_verdict(ratios: list[float]) -> tuple[str, dict[str, Any]]:
    median = float(np.median(ratios))
    beats = sum(1 for r in ratios if r < 1.0)
    worse = len(ratios) - beats
    if median <= RATIO_BEAT and beats >= COUNT_MAJORITY:
        v = "BEAT"
    elif median > RATIO_FAIL or worse >= COUNT_MAJORITY:
        v = "FAIL"
    else:
        v = "PARITY"
    return v, {"median": median, "beats": beats, "worse": worse, "n": len(ratios)}


def diff_verdict(diffs: list[float]) -> tuple[str, dict[str, Any]]:
    """The paired absolute difference has no denominator, so its verdict is a
    sign test on the median plus the same majority condition. There is no
    registered magnitude threshold in metres, deliberately -- one would have had
    to be invented after the fact, which is the thing being avoided."""
    median = float(np.median(diffs))
    better = sum(1 for d in diffs if d < 0)
    worse = len(diffs) - better
    if median < 0 and better >= COUNT_MAJORITY:
        v = "BEAT"
    elif median > 0 and worse >= COUNT_MAJORITY:
        v = "FAIL"
    else:
        v = "PARITY"
    return v, {"median_m": median, "better": better, "worse": worse, "n": len(diffs)}


def structure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    floor = np.array([r["floor"] for r in rows])
    policy = np.array([r["policy"] for r in rows])
    diff = policy - floor

    slope = float(np.polyfit(floor, policy, 1)[0])
    # LEAVE-ONE-OUT IS MANDATORY, not optional (amendment 3). With the predictor
    # spanning ~40x, one reference can own most of Sxx, and a leveraged estimate
    # and a robust verdict are different properties.
    loo = []
    for i in range(len(rows)):
        mask = np.ones(len(rows), bool)
        mask[i] = False
        loo.append(float(np.polyfit(floor[mask], policy[mask], 1)[0]))
    sxx = (floor - floor.mean()) ** 2
    leverage = (sxx / sxx.sum()).tolist()

    lo, hi = min(loo), max(loo)
    if hi < SLOPE_INDEPENDENT:
        reading = "policy error largely INDEPENDENT of reference difficulty"
    elif lo > SLOPE_INHERITS:
        reading = "policy INHERITS reference difficulty -- the beat is easier references"
    elif lo > SLOPE_INDEPENDENT and hi < SLOPE_INHERITS:
        reading = "partial"
    else:
        # The straddle IS the finding (amendment 3).
        reading = "LEAVE-ONE-OUT RANGE STRADDLES A THRESHOLD -- report the straddle, not a point estimate"

    r_pol = spearman(floor, policy)
    r_diff = spearman(floor, diff)
    return {
        "slope": slope, "loo_min": lo, "loo_max": hi, "loo": loo,
        "leverage": leverage, "reading": reading,
        "spearman_floor_policy": r_pol,
        "spearman_floor_policy_p": exact_permutation_p(floor, policy),
        "spearman_floor_diff": r_diff,
        "spearman_floor_diff_p": exact_permutation_p(floor, diff),
    }


def report_arm(label: str, arm: dict[str, Any]) -> dict[str, Any]:
    rows = sorted(arm["rows"], key=lambda r: -r["floor"])
    print(f"\n{'=' * 78}\n{label}\n  references: {arm['reference_path']}\n{'=' * 78}")
    print(f"  {'family':<14}{'floor':>9}{'policy':>9}{'ratio':>8}{'abs diff':>11}{'lev':>7}")
    st = structure(arm["rows"])
    lev_by_id = {r["episode_id"]: l for r, l in zip(arm["rows"], st["leverage"], strict=True)}
    for r in rows:
        print(f"  {r['family']:<14}{r['floor']:>9.4f}{r['policy']:>9.4f}"
              f"{r['ratio']:>8.2f}{r['diff']:>+11.4f}{100 * lev_by_id[r['episode_id']]:>6.1f}%")

    rv, rs = ratio_verdict([r["ratio"] for r in rows])
    dv, ds = diff_verdict([r["diff"] for r in rows])
    print(f"\n  RATIO (registered primary):  median {rs['median']:.3f}, "
          f"better on {rs['beats']}/{rs['n']}  ->  {rv}")
    print(f"    thresholds: BEAT if median <= {RATIO_BEAT} and >= {COUNT_MAJORITY} better; "
          f"FAIL if median > {RATIO_FAIL} or >= {COUNT_MAJORITY} worse")
    print(f"  PAIRED DIFFERENCE (co-reported): median {ds['median_m']:+.4f} m, "
          f"better on {ds['better']}/{ds['n']}  ->  {dv}")

    if rv == dv:
        verdict = rv
        print(f"\n  VERDICT: {verdict}   (both statistics agree)")
    else:
        verdict = "SPLIT"
        print(f"\n  VERDICT: SPLIT   ratio says {rv}, paired difference says {dv}")
        print("    A disagreement between a scale-free and a scale-dependent statistic is")
        print("    itself the finding: the policy improves the hard references and regresses")
        print("    the easy ones, or the reverse. Neither is 'the real one'.")

    print(f"\n  STRUCTURE -- does the policy inherit reference difficulty?")
    print(f"    OLS slope of policy on floor: {st['slope']:+.4f}")
    print(f"    leave-one-out range:          {st['loo_min']:+.4f} to {st['loo_max']:+.4f}   "
          f"(max leverage {100 * max(st['leverage']):.1f}%)")
    print(f"    reading: {st['reading']}")
    p_pol = st["spearman_floor_policy_p"]
    print(f"    spearman(floor, policy) = {st['spearman_floor_policy']:+.3f}"
          + (f", exact p = {p_pol:.4f}" if p_pol is not None else ""))
    print(f"      at n={len(rows)}, |rho| > {SPEARMAN_N8_P05:.2f} is needed for p < 0.05")
    p_d = st["spearman_floor_diff_p"]
    print(f"    spearman(floor, policy-floor) = {st['spearman_floor_diff']:+.3f}"
          + (f", exact p = {p_d:.4f}" if p_d is not None else "")
          + "   <- PART-WHOLE, near-mechanical; not evidence on its own")

    return {"label": label, "verdict": verdict, "ratio": {**rs, "verdict": rv},
            "paired_difference": {**ds, "verdict": dv}, "structure": st,
            "per_reference": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", type=Path, required=True,
                    help="TRAIN-reference arm JSON (registered primary)")
    ap.add_argument("--generalisation", type=Path, default=None,
                    help="VAL-reference arm JSON (supplementary)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    result: dict[str, Any] = {}
    result["primary"] = report_arm(
        "PRIMARY -- 6.0 s, random-start, TRAIN references (transfer)",
        load_arm(a.primary))

    if a.generalisation:
        result["generalisation"] = report_arm(
            "SUPPLEMENTARY -- 6.0 s, random-start, VAL references (generalisation)",
            load_arm(a.generalisation))

        # CROSS-ARM: RATIOS ONLY. The paired difference is in metres against
        # floors that differ ~43% with different command distributions, so
        # centimetres are not commensurable between arms. Refusing to print that
        # comparison is easier than remembering not to read it.
        rt = result["primary"]["ratio"]["median"]
        rv = result["generalisation"]["ratio"]["median"]
        print(f"\n{'=' * 78}\nCROSS-ARM (ratios only -- paired differences are NOT commensurable)\n{'=' * 78}")
        print(f"  median ratio   train {rt:.3f}   val {rv:.3f}")
        held = rv > rt
        print(f"  registered prediction: ratio_val > ratio_train  ->  {'HELD' if held else 'FALSIFIED'}")
        print("    mechanism: val references carry 2.5x more forward range, forward")
        print("    corrections land in the measured sub-0.35 m/s dead zone, so correction")
        print("    authority is weakest where the val arm spends more of its time.")
        if not held:
            print("    FALSIFIED means the beat is not dead-zone-limited and the mechanism is")
            print("    something else. It also means the ratio is robust across the plant's")
            print("    own worst nonlinearity, which is stronger than equality across two")
            print("    arbitrary draws.")
        result["cross_arm"] = {"median_ratio_train": rt, "median_ratio_val": rv,
                               "prediction_ratio_val_gt_train": bool(held)}

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2, default=float) + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
