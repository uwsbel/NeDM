"""Frame and ordering conversions. One definition, used by collection and by rollout.

These live here rather than inside a policy class because the dataset and the policy must
not be able to drift apart: a derived channel written at collection time and the same
quantity computed during a branch rollout have to be the identical function.
"""
from __future__ import annotations

import numpy as np

# rl_sar policy order -> our Chrono order. Self-inverse (it swaps front and rear pairs).
POLICY_TO_CHRONO = np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5], dtype=np.int64)
CHRONO_TO_POLICY = np.argsort(POLICY_TO_CHRONO)


def quat_to_rot(e0: float, e1: float, e2: float, e3: float) -> np.ndarray:
    """Body-to-world rotation from a Chrono quaternion (w, x, y, z)."""
    w, x, y, z = float(e0), float(e1), float(e2), float(e3)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError("degenerate quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def projected_gravity(e0, e1, e2, e3) -> np.ndarray:
    """Gravity direction in the body frame.

    Taken from the quaternion rather than rebuilt from logged roll/pitch, because the
    latter does not reproduce Chrono's value: measured error 0.0288 mean per component
    against components of order 1, and three different Euler compositions were all wrong
    by the same amount, which points at a Cardan convention difference rather than an
    ordering slip. From the quaternion it is exact by construction, and yaw-invariant by
    construction rather than by argument.
    """
    R = quat_to_rot(e0, e1, e2, e3)
    return R.T @ np.array([0.0, 0.0, -1.0])


def world_to_body(vec_world, e0, e1, e2, e3) -> np.ndarray:
    """Rotate a world-frame vector into the body frame.

    Used for the perturbation force, which is LOGGED in the world frame while every other
    channel in the state is body-frame. A world-frame force is uninterpretable to a policy
    whose state carries no yaw: the same physical shove would appear as different numbers
    depending on heading. Rotated, it is heading-invariant, exactly as projected gravity
    is and for the same reason.
    """
    R = quat_to_rot(e0, e1, e2, e3)
    return R.T @ np.asarray(vec_world, dtype=float)


def policy_to_chrono(v: np.ndarray) -> np.ndarray:
    """Reorder a 12-vector from rl_sar joint order into Chrono joint order."""
    v = np.asarray(v)
    if v.shape[-1] != 12:
        raise ValueError(f"expected a 12-vector, got {v.shape}")
    out = np.empty_like(v)
    out[..., POLICY_TO_CHRONO] = v
    return out


def chrono_to_policy(v: np.ndarray) -> np.ndarray:
    """Reorder a 12-vector from Chrono joint order into rl_sar joint order."""
    v = np.asarray(v)
    if v.shape[-1] != 12:
        raise ValueError(f"expected a 12-vector, got {v.shape}")
    return v[..., POLICY_TO_CHRONO]
