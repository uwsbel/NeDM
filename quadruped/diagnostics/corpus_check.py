#!/usr/bin/env python3
"""Is a corpus good enough to fit a model that will be differentiated?

Gate 1, action identifiability. The previous corpus had only ~4% of action variance
surviving conditioning on the state, over an effectively rank-2 subspace of a
12-dimensional action. A model fit to that can ignore the action entirely and still fit
well -- and d s'/d a is the single quantity fine-tuning consumes. So this is not a
nice-to-have statistic; it is the property that decides whether the corpus can support
the method at all.

Also checks the mechanical things a corpus can get wrong silently: non-finite values,
push rows surviving into segments, and whether the segments are long enough to yield
training windows.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import sys

import numpy as np


def load(corpus: Path):
    rows, per_file = [], []
    for f in sorted(glob.glob(str(corpus / "episodes" / "*.csv"))):
        r = list(csv.DictReader(open(f)))
        per_file.append((Path(f).name, len(r)))
        rows.extend(r)
    return rows, per_file


def numeric(rows, cols):
    out = np.empty((len(rows), len(cols)), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = r.get(c, "")
            out[i, j] = float(v) if v not in ("", None) else np.nan
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--seq", type=int, default=128, help="training window length")
    a = ap.parse_args()
    corpus = Path(a.corpus)

    rows, per_file = load(corpus)
    if not rows:
        print("FATAL: no episodes"); return 1
    cols = list(rows[0].keys())
    act_cols = [c for c in cols if c.startswith("joint_") and c.endswith("_target_rad")]
    st_cols = ([c for c in cols if c.startswith("joint_") and c.endswith("_pos_rad")]
               + [c for c in cols if c.startswith("joint_") and c.endswith("_vel_radps")]
               + [c for c in cols if c in ("vel_body_x_mps", "vel_body_y_mps",
                                           "roll_rad", "pitch_rad", "yaw_rate_radps",
                                           "roll_rate_radps", "pos_z_m")])
    print(f"corpus   {corpus.name}")
    print(f"segments {len(per_file)}  rows {len(rows)}")
    print(f"state    {len(st_cols)} cols   action {len(act_cols)} cols")

    A = numeric(rows, act_cols)
    S = numeric(rows, st_cols)
    ok = np.isfinite(A).all(1) & np.isfinite(S).all(1)
    print(f"finite   {ok.sum()}/{len(rows)} rows ({100 * ok.mean():.1f}%)")
    if ok.sum() < 50:
        print("FATAL: too few finite rows"); return 1
    A, S = A[ok], S[ok]

    # GATE 1. Regress each action channel on the state; what is left is the part of the
    # action that the state does not already determine, and it is the only part that can
    # identify d s'/d a.
    X = np.hstack([S - S.mean(0), np.ones((len(S), 1))])
    Ac = A - A.mean(0)
    beta, *_ = np.linalg.lstsq(X, Ac, rcond=None)
    resid = Ac - X @ beta
    var_tot = float(np.sum(np.var(Ac, axis=0)))
    var_res = float(np.sum(np.var(resid, axis=0)))
    frac = var_res / var_tot if var_tot > 0 else float("nan")

    C = np.cov(resid.T)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    ev = np.clip(ev, 0, None)
    eff_rank = float(np.exp(-np.sum((ev / ev.sum()) * np.log(ev / ev.sum() + 1e-300))))

    print()
    print(f"GATE 1  action variance surviving conditioning : {100 * frac:.1f}%")
    print(f"        effective rank of the residual         : {eff_rank:.2f} of {len(act_cols)}")
    print(f"        reference: the previous corpus was ~4% over an effective rank of 2")
    g1 = frac > 0.15 and eff_rank > 6.0
    print(f"        -> {'PASS' if g1 else 'FAIL'}")

    # Segments must be long enough to produce windows at all.
    lens = np.array([n for _, n in per_file])
    wins = np.maximum(lens - a.seq + 1, 0)
    print()
    print(f"windows at seq={a.seq}: {int(wins.sum())} from {len(lens)} segments "
          f"(shortest {lens.min()}, longest {lens.max()})")
    g2 = int(wins.sum()) > 0 and (wins > 0).all()
    print(f"        -> {'PASS' if g2 else 'FAIL: a segment is shorter than the window'}")

    # No push may survive into a segment.
    pf = [c for c in cols if c.startswith("perturb_force")]
    if pf:
        P = numeric(rows, pf)
        live = int((np.abs(np.nan_to_num(P)).max(1) > 1e-9).sum())
        print()
        print(f"push-active rows inside segments: {live}  -> "
              f"{'PASS' if live == 0 else 'FAIL: the force window was not excised'}")
    # GATE 4. Held-out coverage: do the segments cover each other? A corpus whose own
    # halves do not cover each other cannot cover a policy that moves away from it. This
    # is the weak form; the strong form runs in evaluate.py against the states a
    # fine-tuned policy actually visits.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # quadruped/
    from lib.coverage import Reference, verdict as cov_verdict  # noqa: PLC0415
    XA = np.hstack([S, A])
    half = len(XA) // 2
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(XA))
    ref = Reference(XA[perm[:half]], rng=rng, names=st_cols + act_cols)
    sc = ref.score(XA[perm[half:]])
    g4, msg = cov_verdict(sc)
    print()
    print(f"GATE 4  held-out points outside the corpus region : "
          f"{100 * sc['ood_fraction']:.1f}%")
    print(f"        distance ratio to corpus self-distance    : {sc['dist_ratio']:.2f}")
    print(f"        channels extrapolated                     : "
          f"{sc['channels_extrapolated']} of {XA.shape[1]}")
    print(f"        -> {'PASS' if g4 else 'FAIL'}")

    man = corpus / "manifest.json"
    print()
    print(f"manifest: {'present' if man.exists() else 'MISSING'}")
    if man.exists():
        m = json.loads(man.read_text())
        print(f"  commit {m['git']['commit'][:8]} dirty={m['git']['dirty']} "
              f"chrono={m['chrono_build']['md5']}")
    return 0 if (g1 and g2 and g4) else 1


if __name__ == "__main__":
    raise SystemExit(main())
