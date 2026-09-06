"""Effective rank and conditional action variance for the excitation arms.

RUN THIS WITH THE ANALYSIS INTERPRETER, NOT THE COLLECTOR'S. The collector env
(envs/nedm-src, which carries pychrono) has neither pandas nor scipy, so this
script cannot run beside the thing that produced its input:

    /home/kyle/miniconda3/envs/ml/bin/python scripts/analysis/go2_excitation_confound.py

Stated here because it is not stated anywhere else -- two torch envs exist on
this box (`ml`, `entangle`) and no doc says which is canonical.

WHY BOTH NUMBERS AND WHY AT MATCHED n. The policy-generated data has a=pi(s),
so action is a deterministic function of state and a surrogate cannot tell which
of the two drives the dynamics. Two diagnostics measure how far a collection has
broken that:

  effective rank  -- participation ratio of the action covariance spectrum,
                     (sum s)^2 / sum s^2. 12/12 means all joint directions are
                     independently excited; the policy scores ~3.9.
  conditional var -- fraction of action variance surviving conditioning on the
                     state, via kNN in state space. The policy scores ~0.04.

CONDITIONAL VARIANCE FALLS WITH n FOR EVERY DATASET -- a kNN density artefact,
since more samples put neighbours closer and shrink the local action spread. It
was measured falling on BOTH excitation (0.776 -> 0.696) and policy data
(0.185 -> 0.085). It is therefore meaningless as an absolute number or as a
stopping rule, and only interpretable as a comparison AT MATCHED n. This script
refuses to compare unequal samples.
"""
import argparse, json, pathlib, sys
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors

ACTION = [f"target_{i}" for i in range(12)]
STATE = ([f"quat_e{i}" for i in range(4)]
         + [f"vel_body_{a}_mps" for a in "xyz"]
         + [f"ang_vel_body_{a}_radps" for a in "xyz"]
         + [f"joint_{l}_{j}_pos_rad" for l in ("rr", "rl", "fr", "fl")
            for j in ("hip", "thigh", "calf")])


def effective_rank(A):
    """Participation ratio of the covariance spectrum. 12 = all directions used."""
    s = np.linalg.svd(A - A.mean(0), compute_uv=False) ** 2
    return float(s.sum() ** 2 / (s ** 2).sum())


def conditional_variance(S, A, k=16, seed=0):
    """Fraction of action variance left after conditioning on state.

    For each row, the variance of its k state-neighbours' actions, pooled and
    divided by the unconditional action variance. Self is excluded: including it
    biases the local spread downward by 1/k, which matters at the small k that
    the kNN density artefact already makes delicate.
    """
    Ss = (S - S.mean(0)) / (S.std(0) + 1e-12)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Ss)
    idx = nn.kneighbors(Ss, return_distance=False)[:, 1:]
    local = A[idx].var(axis=1).mean(axis=0)
    return float(local.sum() / A.var(axis=0).sum())


def load(paths, phase=None):
    fr = []
    for p in paths:
        d = pd.read_csv(pathlib.Path(p) / "windows.csv", usecols=["phase"] + STATE + ACTION)
        fr.append(d[d.phase == phase] if phase else d)
    return pd.concat(fr, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a", nargs="+", required=True)
    ap.add_argument("--arm-c", nargs="*", default=[])
    ap.add_argument("--phase", default="perturb")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    A = load(a.arm_a, a.phase)
    print(f"arm A  {len(A):,} {a.phase} rows from {len(a.arm_a)} shard(s)")
    rng = np.random.default_rng(a.seed)

    def report(tag, df, n):
        s = rng.choice(len(df), n, replace=False) if n < len(df) else np.arange(len(df))
        S, Ac = df[STATE].to_numpy()[s], df[ACTION].to_numpy()[s]
        er, cv = effective_rank(Ac), conditional_variance(S, Ac, a.k, a.seed)
        print(f"  {tag:24s} n={n:>7,}  effective rank {er:5.2f}/12   conditional var {cv:.3f}")
        return er, cv

    if a.arm_c:
        C = load(a.arm_c, a.phase)
        print(f"arm C  {len(C):,} {a.phase} rows from {len(a.arm_c)} shard(s)")
        # MATCHED n IS THE WHOLE POINT: arm A has ~23x arm C's perturbed rows, and
        # comparing at native volumes confounds burst length with training volume --
        # the same error, one axis over, as holding L x N constant.
        n = min(len(A), len(C))
        print(f"\nmatched at n={n:,} (the smaller arm), burst length L=40 vs L=10 at N=4")
        report("arm A  L=40", A, n)
        report("arm C  L=10", C, n)
    print("\nfull volume (NOT comparable across arms -- n differs):")
    report("arm A  L=40", A, len(A))
    return 0


if __name__ == "__main__":
    sys.exit(main())
