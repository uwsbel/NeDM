"""Score the enforcement A/B on the clearance DISTRIBUTIONS, not the collision count.

A rare binary event is the least informative function of a continuous measurement
we already log. Every episode contributes to min_asset_clearance_m; only ~1 in 100
contributes to the collision count. Enforcement should shift the whole
distribution, and that is detectable at n=100 where the binary readout is not.

No scipy in this env, so Mann-Whitney U (normal approximation with tie
correction) and the bootstrap CI are implemented directly.
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

KEY = "min_asset_clearance_m"


def load(path: str) -> tuple[np.ndarray, list[dict]]:
    rows = [json.loads(l) for l in open(path)]
    if KEY not in rows[0]:
        raise SystemExit(f"{path}: no {KEY}; fields are {sorted(rows[0])}")
    return np.array([r[KEY] for r in rows], dtype=float), rows


def mannwhitney(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Returns (U for a, z, two-sided p) via the normal approximation."""
    n1, n2 = len(a), len(b)
    combined = np.concatenate([a, b])
    order = combined.argsort()
    ranks = np.empty(len(combined), dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1)
    # Average ranks within ties, and collect tie sizes for the variance correction.
    sortc = combined[order]
    i = 0
    ties = []
    while i < len(sortc):
        j = i
        while j + 1 < len(sortc) and sortc[j + 1] == sortc[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
            ties.append(j - i + 1)
        i = j + 1
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    n = n1 + n2
    tie_term = sum(t**3 - t for t in ties)
    var = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1)))
    if var <= 0:
        return u1, 0.0, 1.0
    z = (u1 - mu) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return u1, z, p


def bootstrap_diff(a: np.ndarray, b: np.ndarray, stat, n_boot: int = 20000, seed: int = 7):
    """Percentile CI on stat(a) - stat(b)."""
    rng = np.random.default_rng(seed)
    obs = stat(a) - stat(b)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = stat(rng.choice(a, len(a), replace=True)) - stat(rng.choice(b, len(b), replace=True))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return obs, lo, hi


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a>b) - P(a<b): effect size that does not assume a distribution shape."""
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: behaves at proportions near 0 where normal does not."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def describe(name: str, v: np.ndarray) -> None:
    q = np.percentile(v, [1, 5, 10, 25, 50])
    print(f"  {name:<12} n={len(v):3d}  min {v.min():.3f}  p05 {q[1]:.3f}  p10 {q[2]:.3f} "
          f" p25 {q[3]:.3f}  median {q[4]:.3f}  max {v.max():.3f}")


def main(enforced_path: str, unenforced_path: str) -> None:
    a, rows_a = load(enforced_path)
    b, rows_b = load(unenforced_path)

    print("=== min_asset_clearance_m, driven trajectory ===")
    describe("enforced", a)
    describe("unenforced", b)

    u, z, p = mannwhitney(a, b)
    print(f"\nMann-Whitney U (enforced vs unenforced): U={u:.0f}  z={z:.3f}  two-sided p={p:.3g}")
    print(f"Cliff's delta = {cliffs_delta(a, b):+.3f}   (+1 = enforced always larger)")

    for label, stat in (("median", np.median), ("p05", lambda v: np.percentile(v, 5)),
                        ("p10", lambda v: np.percentile(v, 10))):
        obs, lo, hi = bootstrap_diff(a, b, stat)
        excl = "excludes 0" if (lo > 0 or hi < 0) else "INCLUDES 0"
        print(f"bootstrap {label:>6} difference: {obs:+.3f} m   95% CI [{lo:+.3f}, {hi:+.3f}]  {excl}")

    # The headline: fraction below the bound the SEARCH was actually run under.
    # 2.60 m is not an arbitrary cut — it is inflation_m + tracker_p95_margin_m,
    # so below it means the planner's own guarantee was not delivered.
    print("\n=== FRACTION BELOW THE 2.60 m SEARCH BOUND (the planner's own guarantee) ===")
    ka, kb = int((a < 2.60).sum()), int((b < 2.60).sum())
    la, ha = wilson(ka, len(a))
    lb, hb = wilson(kb, len(b))
    print(f"  enforced    {ka:3d}/{len(a)} = {100*ka/len(a):5.1f}%   95% CI [{100*la:.1f}%, {100*ha:.1f}%]")
    print(f"  unenforced  {kb:3d}/{len(b)} = {100*kb/len(b):5.1f}%   95% CI [{100*lb:.1f}%, {100*hb:.1f}%]")
    obs, lo, hi = bootstrap_diff(a, b, lambda v: float((v < 2.60).mean()))
    print(f"  difference  {100*obs:+.1f} pp   95% CI [{100*lo:+.1f}, {100*hi:+.1f}] pp")

    # Descriptive footnote only. The binary readout is underpowered by design:
    # at a 1/100 baseline, 0/100 is the most likely outcome even under the null.
    ca = [r["episode"] for r in rows_a if r.get("max_asset_contact_n", 0) > 0]
    cb = [r["episode"] for r in rows_b if r.get("max_asset_contact_n", 0) > 0]
    print(f"\nFOOTNOTE (descriptive, underpowered): collisions enforced {len(ca)}/{len(a)} {ca}"
          f" | unenforced {len(cb)}/{len(b)} {cb}")
    print(f"episodes below 1.10 m hull half-width: enforced {(a < 1.10).sum()} | unenforced {(b < 1.10).sum()}")
    print(f"episodes below 2.60 m search bound:    enforced {(a < 2.60).sum()} | unenforced {(b < 2.60).sum()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
