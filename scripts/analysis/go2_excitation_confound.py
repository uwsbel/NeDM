"""Effective rank and conditional action variance for the excitation arms.

THE ARMS ARE EX-A AND EX-C, not "arm A" and "arm C". This repo already uses
"arm C" for a context-length arm on a different study line, so the excitation
arms carry the EX- prefix to keep the two unambiguous.

RUN THIS WITH $NEDM_ANALYSIS_PY, NOT $NEDM_PY. No env is named here on purpose:
the fleet convention (docs/state/machines/README.md) is that scripts read their
interpreter from the environment so that no script hardcodes an env name.

    "$NEDM_ANALYSIS_PY" scripts/analysis/go2_excitation_confound.py --ex-a ...

The variable exists separately from $NEDM_PY because on this box no env carries
both pychrono and the analysis stack, so simulation and analysis cannot share an
interpreter. Which env satisfies which is recorded on the machine page.

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
    ap.add_argument("--ex-a", nargs="+", required=True)
    ap.add_argument("--ex-c", nargs="*", default=[])
    ap.add_argument("--phase", default="perturb")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    A = load(a.ex_a, a.phase)
    print(f"EX-A  {len(A):,} {a.phase} rows from {len(a.ex_a)} shard(s)")
    rng = np.random.default_rng(a.seed)

    def report(tag, df, n):
        s = rng.choice(len(df), n, replace=False) if n < len(df) else np.arange(len(df))
        S, Ac = df[STATE].to_numpy()[s], df[ACTION].to_numpy()[s]
        er, cv = effective_rank(Ac), conditional_variance(S, Ac, a.k, a.seed)
        print(f"  {tag:24s} n={n:>7,}  effective rank {er:5.2f}/12   conditional var {cv:.3f}")
        return er, cv

    if a.ex_c:
        C = load(a.ex_c, a.phase)
        print(f"EX-C  {len(C):,} {a.phase} rows from {len(a.ex_c)} shard(s)")
        # MATCHED n IS THE WHOLE POINT: EX-A has ~23x EX-C's perturbed rows, and
        # comparing at native volumes confounds burst length with training volume --
        # the same error, one axis over, as holding L x N constant.
        n = min(len(A), len(C))
        # MATCHED n IS NOT ENOUGH. The kNN score also depends on the SAMPLING FRACTION
        # n/pool: drawing 100,000 rows from a 100,000-row pool takes every row of every
        # episode, so neighbours are consecutive timesteps and the local action spread
        # collapses. Drawing the same 100,000 from a million-row pool samples episodes
        # sparsely and the neighbours are further apart. Measured on one EX-A shard at
        # fixed n=100,000: 0.8351 at fraction 1.00 rising to 0.8538 at fraction 0.10.
        # So the smaller arm, which is always sampled at fraction 1.00, is scored on
        # harsher terms unless the larger arm is TRUNCATED to the same pool size rather
        # than subsampled from all of it.
        A = A.iloc[:n]
        C = C.iloc[:n]
        print(f"\nmatched at n={n:,} AND sampling fraction 1.00 for both arms")
        report("EX-A  L=40", A, n)
        report("EX-C  L=10", C, n)
    print("\nfull volume (NOT comparable across arms -- n differs):")
    report("EX-A  L=40", A, len(A))
    return 0


if __name__ == "__main__":
    sys.exit(main())
