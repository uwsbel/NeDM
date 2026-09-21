"""Gate 4: is a policy operating inside the data its model was fit on?

The risk this exists to catch. Fine-tuning changes the policy, so the fine-tuned policy
visits (state, action) pairs the corpus may not contain -- and a learned model asked to
predict there is extrapolating, which is precisely where an optimiser finds structure the
model invented. The previous study's whole exploitability thread is this failure.

Two distinct exposures, and they need separate answers:

  STATE coverage. If the corpus was collected only where the robot walks slowly, and
  fine-tuning pushes it faster, the states are new. This is fixed at COLLECTION time by
  commanding well beyond the target operating point -- measured on CRM, commanding 2.0 m/s
  achieves 1.5, so a 0.5 m/s target sits deep inside.

  ACTION coverage. Even at a state the corpus contains, a fine-tuned policy may choose a
  different action -- a different gait for the same pose. A corpus where the action is a
  function of the state (a = pi(s)) contains NO off-policy actions at all, so the model's
  gradient there is invented rather than fitted. This is what action injection buys, and
  what Gate 1 measures.

This module checks the result after the fact, which neither of those does. It asks: of the
points this policy actually visited, how many lie outside the region the corpus occupies?

METHOD. k-th nearest neighbour distance, which needs no density estimate and no
distributional assumption. The corpus's distance to its OWN k-th neighbour gives the scale
at which it is dense; a visited point further than the corpus's 99th percentile of that is
somewhere the corpus effectively does not reach. Reported as a fraction, plus a per-channel
range check that says WHICH channel is being extrapolated, because "3% outside" is
actionable only if you know along what axis.
"""
from __future__ import annotations

import numpy as np


class Reference:
    """The region a corpus occupies, in whitened (state, action) space."""

    def __init__(self, X: np.ndarray, k: int = 8, max_points: int = 4000,
                 rng: np.random.Generator | None = None, names: list[str] | None = None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or len(X) < k + 1:
            raise ValueError(f"need a 2-D array with more than k={k} rows, got {X.shape}")
        rng = rng or np.random.default_rng(0)
        self.names = names
        self.mu = X.mean(0)
        sd = X.std(0)
        # A constant channel carries no information and would divide by zero; it is kept
        # at unit scale so it contributes nothing to distance rather than infinity.
        self.sd = np.where(sd < 1e-9, 1.0, sd)
        self.lo, self.hi = X.min(0), X.max(0)
        Z = (X - self.mu) / self.sd
        idx = rng.choice(len(Z), size=min(max_points, len(Z)), replace=False)
        self.ref = Z[idx]
        self.k = k
        # Self-distance: for each reference point, distance to its k-th neighbour among
        # the others. This is the corpus's own notion of "close".
        d = _knn_dist(self.ref, self.ref, k + 1)   # +1: a point is its own 0-distance NN
        self.self_d = d
        self.thresh = float(np.quantile(d, 0.99))

    def score(self, X: np.ndarray) -> dict:
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self.mu) / self.sd
        d = _knn_dist(Z, self.ref, self.k)
        outside = d > self.thresh
        below = (X < self.lo).mean(0)
        above = (X > self.hi).mean(0)
        per_ch = below + above
        worst = int(np.argmax(per_ch))
        return {
            "n": int(len(X)),
            "ood_fraction": float(outside.mean()),
            "threshold": self.thresh,
            "median_dist": float(np.median(d)),
            "corpus_median_dist": float(np.median(self.self_d)),
            "dist_ratio": float(np.median(d) / max(np.median(self.self_d), 1e-12)),
            "worst_channel": (self.names[worst] if self.names else worst),
            "worst_channel_outside": float(per_ch[worst]),
            "channels_extrapolated": int((per_ch > 0.01).sum()),
        }


def _knn_dist(Q: np.ndarray, R: np.ndarray, k: int) -> np.ndarray:
    """Distance from each row of Q to its k-th nearest row of R. Chunked so a long
    rollout against a large reference does not allocate an n*m matrix."""
    out = np.empty(len(Q))
    step = max(1, int(2e7 // max(len(R), 1)))
    for i in range(0, len(Q), step):
        q = Q[i:i + step]
        d2 = ((q[:, None, :] - R[None, :, :]) ** 2).sum(-1)
        kk = min(k, d2.shape[1]) - 1
        out[i:i + step] = np.sqrt(np.partition(d2, kk, axis=1)[:, kk])
    return out


def verdict(sc: dict, max_ood: float = 0.05) -> tuple[bool, str]:
    """A policy operating outside its model's data is not a result, it is an
    extrapolation. The threshold is deliberately strict."""
    ok = sc["ood_fraction"] <= max_ood
    if ok:
        return True, (f"{100 * sc['ood_fraction']:.1f}% of visited points outside the "
                      f"corpus (limit {100 * max_ood:.0f}%)")
    return False, (f"{100 * sc['ood_fraction']:.1f}% of visited points lie outside the "
                   f"corpus region, above the {100 * max_ood:.0f}% limit. Worst channel "
                   f"{sc['worst_channel']} with {100 * sc['worst_channel_outside']:.1f}% "
                   f"beyond its recorded range. The model is extrapolating there and any "
                   f"gain it reports may be structure it invented.")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # A corpus occupying a slab, and three probes: inside, shifted, and widened.
    C = rng.normal(size=(3000, 6))
    ref = Reference(C, rng=rng, names=[f"ch{i}" for i in range(6)])

    inside = ref.score(rng.normal(size=(500, 6)))
    shifted = ref.score(rng.normal(size=(500, 6)) + 3.0)
    wider = ref.score(rng.normal(size=(500, 6)) * 3.0)

    print(f"in-distribution   ood {100 * inside['ood_fraction']:5.1f}%  "
          f"dist_ratio {inside['dist_ratio']:.2f}")
    print(f"shifted by 3 sd   ood {100 * shifted['ood_fraction']:5.1f}%  "
          f"dist_ratio {shifted['dist_ratio']:.2f}")
    print(f"3x wider          ood {100 * wider['ood_fraction']:5.1f}%  "
          f"dist_ratio {wider['dist_ratio']:.2f}")

    assert inside["ood_fraction"] < 0.05, inside
    assert shifted["ood_fraction"] > 0.9, shifted
    assert wider["ood_fraction"] > 0.3, wider
    ok, msg = verdict(inside); assert ok, msg
    ok, msg = verdict(shifted); assert not ok, msg
    print("\nverdict on the shifted probe:")
    print(" ", msg)
    print("\ncoverage self-test OK")
