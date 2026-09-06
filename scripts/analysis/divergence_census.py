"""Per-dataset census of numerically diverged rows.

WHY THIS EXISTS. The excitation collector rejects diverged episodes; the policy
collectors never did. So one corpus contains none and the other does -- not
because it is better behaved, but because a filter inside one collector threw
them away. That filter is recorded in NEITHER dataset, so the asymmetry is
invisible at analysis time and silently biases any cross-corpus comparison.

It already produced one: comparing the two corpora as they sit gave the policy
an effective rank of 1.00/12 against excitation's 12.00/12, because 0.78% of
policy rows carry joint targets up to 4e34 rad and those rows dominate the
covariance. That is the right conclusion by an entirely wrong mechanism, and it
overshot in the direction the conclusion already pointed, which is why it read
as confirmation rather than error.

So: measure it once, for every dataset, and let later comparisons consult the
census instead of rediscovering this.

BOUNDS ARE DELIBERATELY LOOSE. They are set well beyond anything physical, so a
trip means numerical divergence rather than an aggressive episode. A Go2 joint
moves within about +-3 rad; the bound is 10.

    "$NEDM_ANALYSIS_PY" scripts/analysis/divergence_census.py
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

# (family, substring test, absurd-value bound)
FAMILIES = [
    ("joint_pos",   lambda c: c.endswith("_pos_rad"),      10.0),
    ("joint_target", lambda c: c.endswith("_target_rad") or c.startswith("target_"), 10.0),
    ("joint_vel",   lambda c: c.endswith("_vel_radps") and "ang_" not in c, 200.0),
    ("lin_vel",     lambda c: c.startswith("vel_"),        100.0),
    ("ang_vel",     lambda c: c.startswith("ang_vel_"),    200.0),
    ("quat",        lambda c: c.startswith("quat_e"),        1.01),
    ("force",       lambda c: "_force_" in c or c.endswith("_n"), 1e5),
    ("torque",      lambda c: "torque" in c or c.startswith("tau"), 1e5),
]


def census_frame(d):
    out, bad_any = {}, None
    for fam, test, bound in FAMILIES:
        cols = [c for c in d.columns if test(c)]
        if not cols:
            continue
        v = d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        nonfinite = ~np.isfinite(v)
        over = (np.abs(np.nan_to_num(v, nan=0.0, posinf=np.inf, neginf=np.inf)) > bound)
        bad = (nonfinite | over).any(axis=1)
        finite = v[np.isfinite(v)]
        out[fam] = {"columns": len(cols), "bound": bound,
                    "rows_over_bound": int(bad.sum()),
                    "max_abs": float(np.abs(finite).max()) if finite.size else None,
                    "nonfinite_cells": int(nonfinite.sum())}
        bad_any = bad if bad_any is None else (bad_any | bad)
    return out, (int(bad_any.sum()) if bad_any is not None else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/kyle/sbel-artifacts/datasets")
    ap.add_argument("--out", default="docs/state/provenance/go2_divergence_census.json")
    ap.add_argument("--max-rows", type=int, default=400000)
    a = ap.parse_args()
    root = Path(a.root)
    res = {}
    print(f"{'dataset':28s} {'rows':>9s} {'diverged':>9s} {'pct':>7s}  worst family")
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(glob.glob(f"{d}/**/*.csv", recursive=True))
        if not files:
            continue
        frames, tot = [], 0
        for f in files:
            try:
                x = pd.read_csv(f)
            except Exception:
                continue
            frames.append(x); tot += len(x)
            if tot >= a.max_rows:
                break
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        fam, nbad = census_frame(df)
        worst = max((v["rows_over_bound"], k) for k, v in fam.items())[1] if fam else "-"
        res[d.name] = {"rows_scanned": len(df), "files_scanned": len(frames),
                       "diverged_rows": nbad, "diverged_pct": 100.0 * nbad / max(len(df), 1),
                       "truncated": tot >= a.max_rows, "families": fam}
        print(f"{d.name:28s} {len(df):9,d} {nbad:9,d} {100.0*nbad/max(len(df),1):6.3f}%  "
              f"{worst if nbad else '-'}")
    Path(a.out).write_text(json.dumps(
        {"note": ("Diverged-row census. Bounds are far beyond physical, so a trip means "
                  "numerical divergence rather than an aggressive episode. Consult this "
                  "before any cross-corpus comparison: the excitation collector filters "
                  "diverged episodes and the policy collectors do not, and that filter is "
                  "recorded in neither dataset."),
         "bounds": {f: b for f, _, b in FAMILIES}, "datasets": res}, indent=1))
    dirty = {k: v["diverged_pct"] for k, v in res.items() if v["diverged_rows"]}
    print(f"\n{len(dirty)}/{len(res)} datasets contain diverged rows -> {a.out}")
    for k, v in sorted(dirty.items(), key=lambda x: -x[1]):
        print(f"   {k:28s} {v:6.3f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
