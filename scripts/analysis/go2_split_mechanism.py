#!/usr/bin/env python3
"""Is the family split carried by yaw content, or by the family label itself?

E3 showed the family split separates better than a body-motion split, but not
WHAT about family does the work. The realisation pilot suggests an answer:
straight families command vx, which the robot realises at 4-5%, and turning
families command wz, which it realises at 0.8-0.9. If that is the mechanism,
REALISED |wz| should carry the effect and the family label should add nothing
once it is accounted for.

Reported as a nested comparison rather than two separate fits, because the
label and the yaw content are collinear by construction -- a turning family is
turning -- and the only honest question is what one adds over the other.
"""
import argparse, csv, glob, json, os, re
import numpy as np

STRAIGHT = ("constant", "vel_step")
SCORED_ROWS = 1000


def yaw_content(csv_path):
    try:
        rows = list(csv.DictReader(open(csv_path)))
    except Exception:
        return None
    if len(rows) < SCORED_ROWS:
        return None
    w = np.array([float(r["yaw_rate_radps"]) for r in rows[-SCORED_ROWS:]])
    return float(np.median(np.abs(w))) if np.isfinite(w).all() else None


def r2(y, X):
    """R^2 of an OLS fit of y on X (intercept added)."""
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--baseline-root", required=True)
    a = ap.parse_args()

    pairs = json.load(open(a.summary_json))["pairs"]
    index = {}
    for jp in glob.glob(os.path.join(a.baseline_root, "**", "episodes", "*.json"),
                        recursive=True):
        try:
            m = json.load(open(jp))
        except Exception:
            continue
        hit = re.search(r"(\d+)$", str(m.get("episode_id", "")))
        if m.get("command_family") and hit:
            cp = os.path.splitext(jp)[0] + ".csv"
            if os.path.exists(cp):
                index[(m["command_family"], int(hit.group(1)))] = cp

    d, lab, yaw = [], [], []
    for p in pairs:
        cp = index.get((p["family"], p["idx"]))
        y = yaw_content(cp) if cp else None
        if y is None:
            continue
        d.append(p["difference"]); yaw.append(y)
        lab.append(0.0 if p["family"] in STRAIGHT else 1.0)
    d, lab, yaw = np.array(d), np.array(lab), np.array(yaw)
    print(f"{len(d)} of {len(pairs)} pairs with readable yaw content")
    if len(d) < 20:
        raise SystemExit("too few pairs to compare nested fits")

    print(f"  realised |yaw| : straight median {np.median(yaw[lab == 0]):.4f}  "
          f"turning median {np.median(yaw[lab == 1]):.4f} rad/s")
    corr = float(np.corrcoef(lab, yaw)[0, 1])
    print(f"  label vs yaw content correlation r = {corr:.3f}"
          + ("   (collinear, as expected)" if abs(corr) > 0.7 else ""))

    r_lab, r_yaw = r2(d, [lab]), r2(d, [yaw])
    r_both = r2(d, [lab, yaw])
    print(f"\n  R^2  label only      {r_lab:.4f}")
    print(f"  R^2  yaw only        {r_yaw:.4f}")
    print(f"  R^2  both            {r_both:.4f}")
    print(f"  label adds over yaw  {r_both - r_yaw:+.4f}")
    print(f"  yaw adds over label  {r_both - r_lab:+.4f}")

    # THE FLOOR IS CHECKED FIRST. Ranking two explanations that both explain
    # nothing produces a confident sentence about noise: on the zero-disturbance
    # corpus every R^2 was below 0.006 and an earlier ordering of these branches
    # reported "yaw content accounts for the split" from a 0.0023 difference.
    if max(r_lab, r_yaw, r_both) < 0.02:
        print("\n  -> NEITHER explains much (all R^2 < 0.02). With effects this small "
              "relative to episode noise, this comparison cannot discriminate; "
              "the ranking above is not interpretable.")
    elif r_both - r_yaw < 0.005 and r_both - r_lab >= 0.005:
        print("\n  -> YAW CONTENT accounts for the split; the label adds nothing.")
    elif r_both - r_lab < 0.005 and r_both - r_yaw >= 0.005:
        print("\n  -> THE LABEL carries something yaw content does not.")
    else:
        print("\n  -> Each adds over the other; they are not interchangeable here.")


if __name__ == "__main__":
    main()
