"""The 24 rotations of a cube, and the canonical relabelling of the cube's orientation for the grasp NRD.

A plain cube looks the same after each of the 24 rotations that map it onto itself (the rotation group of the cube).
The grasp NRD is not invariant to them: the same physical state with another body-frame labelling gives other
predictions (critic map, ``~/so101_policy/docs/map_critic.md`` sec. 3.4: 1 s cube p95 4.0 mm, 10.7 mm after a 90 deg
relabel about the cube z axis, 119 mm after 180 deg). The canonical relabelling below restores the accuracy.

Conventions: quaternions w x y z, unit, world <- body (``q`` turns cube-body vectors into the world frame), float64.
A relabelling by the group element r gives the orientation q' = q r: the same physical cube, with its body axes
renamed. Position, world-frame velocity and world-frame spin do not change.

Canonical rule (``~/so101_policy/scripts/critic/symmetry_canon.py``, tested there on 1,984 validation windows): choose
the r that maximises |<q r, Q_REF>|, with Q_REF a flat cube at yaw 45 deg (the middle of the data's 0-90 deg yaw band);
ties keep the lowest group index (index 0 is the identity). For a flat cube the rule maps any yaw into (0, 90) deg.

The group order is that of the critic's ``cube_group`` (closure of {90 deg about x, y, z} in the same loop order), so a
group index means the same rotation in both places. Entries are snapped to the exact values 0, +-1/2, +-1/sqrt(2), +-1.
"""
from __future__ import annotations

import math

import torch

ORDER = 24
Q_REF = (math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8))   # flat cube, yaw 45 deg


def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a b of quaternions [..., 4] (w x y z), broadcast."""
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2), -1)


def qconj(q: torch.Tensor) -> torch.Tensor:
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def axis_quat(axis: str, deg: float, dtype=torch.float64, device=None) -> torch.Tensor:
    """Rotation by deg about the world/body axis "x", "y" or "z" -> [4]."""
    t = math.radians(deg) / 2
    v = [0.0, 0.0, 0.0]
    v["xyz".index(axis)] = math.sin(t)
    return torch.tensor([math.cos(t)] + v, dtype=dtype, device=device)


def _snap(g: torch.Tensor) -> torch.Tensor:
    vals = torch.tensor([0.0, 0.5, math.sqrt(0.5), 1.0], dtype=torch.float64)
    a = g.abs()
    k = (a[..., None] - vals).abs().argmin(-1)
    if (a - vals[k]).abs().max() > 1e-9:
        raise AssertionError("cube group entries must be 0, 1/2, 1/sqrt(2) or 1")
    return torch.sign(g) * vals[k]


def _build_group() -> torch.Tensor:
    """The 24 cube rotations [24, 4] (float64, CPU), in the critic's order; one sign each (the closure's)."""
    gens = [axis_quat(a, 90) for a in "xyz"]
    group = [torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)]
    while True:
        new = []
        for g in group:
            for h in gens:
                q = qmul(g, h)
                if all(abs(float((q * x).sum())) < 1 - 1e-9 for x in group + new):
                    new.append(q)
        if not new:
            break
        group += new
    out = _snap(torch.stack(group))
    if len(out) != ORDER:
        raise AssertionError(f"cube group has {len(out)} elements")
    return out


_GROUP = _build_group()


def _tables(group):
    prod = qmul(group[:, None, :], group[None, :, :])                     # [24, 24, 4]
    dots = (prod[:, :, None, :] * group[None, None]).sum(-1)              # [24, 24, 24]
    mul = dots.abs().argmax(-1)
    if not torch.allclose(dots.abs().amax(-1), torch.ones(ORDER, ORDER, dtype=torch.float64), atol=1e-12):
        raise AssertionError("cube group is not closed")
    inv = (mul == 0).to(torch.long).argmax(-1)
    return mul, inv


_MUL, _INV = _tables(_GROUP)


def group(dtype=torch.float64, device=None) -> torch.Tensor:
    """The 24 rotations [24, 4] (w x y z); index 0 = identity."""
    return _GROUP.to(dtype=dtype, device=device)


def mul_table(device=None) -> torch.Tensor:
    """[24, 24] long: G[i] G[j] = +-G[mul[i, j]]."""
    return _MUL.to(device)


def inverse_table(device=None) -> torch.Tensor:
    """[24] long: G[i] G[inv[i]] = +-identity."""
    return _INV.to(device)


def score_matrix(q_ref=Q_REF, dtype=torch.float64, device=None) -> torch.Tensor:
    """[4, 24] M with (q @ M)[k] = <q G[k], q_ref> for any quaternion q (as <a b, c> = <a, c conj(b)>)."""
    ref = torch.as_tensor(q_ref, dtype=torch.float64)
    return qmul(ref.expand(ORDER, 4), qconj(_GROUP)).T.contiguous().to(dtype=dtype, device=device)


def canonical_index(q: torch.Tensor, scores: torch.Tensor | None = None, margin: float = 0.0) -> torch.Tensor:
    """q [..., 4] -> [...] long: the group index k of the canonical relabelling q G[k] (argmax |<q G[k], Q_REF>|, ties
    to the lowest index). margin > 0 adds hysteresis: keep the identity (0) unless another element scores more than
    margin above it. scores: score_matrix() on q's device (pass it to avoid a host-to-device copy per call)."""
    m = score_matrix(dtype=q.dtype, device=q.device) if scores is None else scores
    s = (q @ m).abs()
    k = s.argmax(-1)
    if margin > 0:
        k = torch.where(s.amax(-1) > s[..., 0] + margin, k, torch.zeros_like(k))
    return k


def relabel(q: torch.Tensor, k: torch.Tensor, g: torch.Tensor | None = None, newest_sign: bool = True) -> torch.Tensor:
    """q [B, ..., 4] relabelled by G[k] (k [B] long, one element per world for all its entries) -> q G[k]. With
    newest_sign, every entry of a world is multiplied by the sign that makes w >= 0 in its last entry along dim -2
    (q [B, K, 4]: one sign per history, so differences stay continuous); without, the sign is left as the product
    gives it."""
    gg = (group(q.dtype, q.device) if g is None else g)[k]
    gg = gg.view(gg.shape[0], *([1] * (q.dim() - 2)), 4)
    out = qmul(q, gg)
    if newest_sign:
        ref = out[:, -1:, :1] if q.dim() >= 3 else out[:, :1]
        out = out * torch.where(ref < 0, -1.0, 1.0).to(out.dtype)
    return out


def canonicalise(q: torch.Tensor, margin: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """q [B, 4] -> (canonical q G[k] with w >= 0 [B, 4], k [B])."""
    k = canonical_index(q, margin=margin)
    return relabel(q, k), k


def true_from_canonical(q_canon: torch.Tensor, c: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
    """The true orientation q_canon conj(G[c]) [B, 4] of a canonical one, with c [B] the cumulative relabelling
    (q_canon = q_true G[c] up to sign). The sign is the product's; align it with a previous value if needed."""
    gg = (group(q_canon.dtype, q_canon.device) if g is None else g)[c]
    return qmul(q_canon, qconj(gg))


def yaw_of_flat(q: torch.Tensor) -> torch.Tensor:
    """Yaw (rad) of the body x axis projected on the world xy plane, for orientations q [..., 4]."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
