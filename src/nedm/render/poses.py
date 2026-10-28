"""Reduced state -> poses the renderer can draw.

A transform is seven numbers, ``[x, y, z, qx, qy, qz, qw]``, as Warp and Newton store it.
Everything here is batched torch on whatever device the NRD model runs on, so the result
reaches the renderer without leaving the GPU.

The reduced states in this repository do not carry x, y or yaw. Every environment recovers
them by integrating the predicted body-frame velocities and yaw rate. :class:`PlanarPose` is
that integration, written once.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch


def quat_from_rpy(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Quaternion ``xyzw`` of ``R = Rz(yaw) Ry(pitch) Rx(roll)``, shape (..., 4)."""
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dim=-1,
    )


def tilt_from_gravity(gravity_body: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll and pitch from the gravity direction in the body frame.

    ``gravity_body = R^T (0, 0, -1)`` with ``R = Rz(yaw) Ry(pitch) Rx(roll)`` gives
    ``(sin pitch, -sin roll cos pitch, -cos roll cos pitch)``. The vector need not be unit
    length, which matters because a predicted state rarely is.
    """
    g = gravity_body
    pitch = torch.atan2(g[..., 0], torch.sqrt(g[..., 1] ** 2 + g[..., 2] ** 2))
    roll = torch.atan2(-g[..., 1], -g[..., 2])
    return roll, pitch


def base_transform(
    xy_yaw: torch.Tensor,
    z: torch.Tensor | float,
    roll: torch.Tensor | None = None,
    pitch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transform (N, 7) of a base at planar pose ``xy_yaw`` (N, 3), height ``z``, tilted."""
    n = xy_yaw.shape[0]
    zero = torch.zeros(n, dtype=torch.float32, device=xy_yaw.device)
    roll = zero if roll is None else roll.float()
    pitch = zero if pitch is None else pitch.float()
    out = torch.zeros(n, 7, dtype=torch.float32, device=xy_yaw.device)
    out[:, 0] = xy_yaw[:, 0].float()
    out[:, 1] = xy_yaw[:, 1].float()
    out[:, 2] = z
    out[:, 3:] = quat_from_rpy(roll, pitch, xy_yaw[:, 2].float())
    return out


def transform_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Homogeneous matrices (..., 4, 4) -> transforms (..., 7). Rotation must be proper."""
    m = matrix[..., :3, :3]
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    # Shepperd: divide by the LARGEST of the four components. Taking each from its own square
    # root loses half the precision whenever a component is near zero.
    four_sq = torch.stack([1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22,
                           1 + m00 + m11 + m22], dim=-1)                       # 4x^2, 4y^2, 4z^2, 4w^2
    rows = torch.stack([
        torch.stack([four_sq[..., 0], m01 + m10, m02 + m20, m21 - m12], dim=-1),   # 4x * (x, y, z, w)
        torch.stack([m01 + m10, four_sq[..., 1], m12 + m21, m02 - m20], dim=-1),   # 4y * ...
        torch.stack([m02 + m20, m12 + m21, four_sq[..., 2], m10 - m01], dim=-1),   # 4z * ...
        torch.stack([m21 - m12, m02 - m20, m10 - m01, four_sq[..., 3]], dim=-1),   # 4w * ...
    ], dim=-2)
    big = four_sq.argmax(dim=-1)
    quat = torch.gather(rows, -2, big[..., None, None].expand(*big.shape, 1, 4)).squeeze(-2)
    quat = quat / (2.0 * torch.sqrt(torch.gather(four_sq, -1, big[..., None])))
    quat = torch.where(quat[..., 3:4] < 0, -quat, quat)
    return torch.cat([matrix[..., :3, 3], quat], dim=-1).float()


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v`` (..., 3) by quaternions ``q`` (..., 4) in ``xyzw``."""
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def compose(parent: torch.Tensor, child: torch.Tensor) -> torch.Tensor:
    """``parent * child`` for transforms (..., 7): the child pose expressed in the parent's frame."""
    pq, cq = parent[..., 3:], child[..., 3:]
    pu, pw = pq[..., :3], pq[..., 3:4]
    cu, cw = cq[..., :3], cq[..., 3:4]
    quat = torch.cat([pw * cu + cw * pu + torch.cross(pu, cu, dim=-1),
                      pw * cw - (pu * cu).sum(-1, keepdim=True)], dim=-1)
    return torch.cat([parent[..., :3] + quat_rotate(pq, child[..., :3]), quat], dim=-1)


class PlanarPose:
    """x, y and yaw of every world, integrated from body velocities and yaw rate.

    The same update as ``HMMWVNeuralTrackingEnv._integrate_pose`` and the rollout metric in
    the trainers: yaw first, then the body velocity rotated by the new yaw. Kept in float64,
    as those do, because the sum runs for thousands of steps.
    """

    def __init__(self, num_worlds: int, dt_s: float, device: str | torch.device | None = None):
        self.dt_s = float(dt_s)
        self.xy_yaw = torch.zeros(num_worlds, 3, dtype=torch.float64, device=device)

    def reset(self, ids: torch.Tensor | None = None, xy_yaw: torch.Tensor | None = None) -> None:
        ids = slice(None) if ids is None else ids
        self.xy_yaw[ids] = 0.0 if xy_yaw is None else xy_yaw.double()

    def step(self, vx: torch.Tensor, vy: torch.Tensor, yaw_rate: torch.Tensor) -> torch.Tensor:
        vx, vy = vx.double(), vy.double()
        yaw = self.xy_yaw[:, 2] + yaw_rate.double() * self.dt_s
        x = self.xy_yaw[:, 0] + (vx * torch.cos(yaw) - vy * torch.sin(yaw)) * self.dt_s
        y = self.xy_yaw[:, 1] + (vx * torch.sin(yaw) + vy * torch.cos(yaw)) * self.dt_s
        self.xy_yaw = torch.stack([x, y, yaw], dim=-1)
        return self.xy_yaw


class JointMap:
    """State columns -> the joint coordinates of a URDF scene, matched by name.

    ``mapping`` gives, for each URDF joint, the state field that holds its angle, or a tuple
    ``(field, sign, offset)`` when the two conventions differ:
    ``q_urdf = sign * state[field] + offset``. Names rather than positions, because a
    reordered preset is exactly the kind of error that renders a plausible wrong robot.
    """

    def __init__(
        self,
        joint_index: Mapping[str, int],
        coord_count: int,
        state_fields: Sequence[str],
        mapping: Mapping[str, str | tuple[str, float] | tuple[str, float, float]],
        base_coords: int = 7,
    ):
        column = {name: i for i, name in enumerate(state_fields)}
        src, dst, sign, offset = [], [], [], []
        for joint, spec in mapping.items():
            field, s, o = (spec, 1.0, 0.0) if isinstance(spec, str) else (*spec, 0.0)[:3]
            if joint not in joint_index:
                raise KeyError(f"the scene has no joint {joint!r}, it has {sorted(joint_index)}")
            if field not in column:
                raise KeyError(f"the state has no field {field!r}")
            src.append(column[field])
            dst.append(int(joint_index[joint]))
            sign.append(float(s))
            offset.append(float(o))
        self.coord_count, self.base_coords = int(coord_count), int(base_coords)
        self.src, self.dst = src, dst
        self.sign = torch.tensor(sign, dtype=torch.float32)
        self.offset = torch.tensor(offset, dtype=torch.float32)

    def joint_q(self, state: torch.Tensor, base: torch.Tensor | None = None) -> torch.Tensor:
        """(N, coord_count). ``base`` (N, 7) fills a floating base. Omit it for a fixed one."""
        q = torch.zeros(state.shape[0], self.coord_count, dtype=torch.float32, device=state.device)
        if base is not None:
            q[:, : self.base_coords] = base
        elif self.base_coords == 7:
            q[:, 6] = 1.0
        sign, offset = self.sign.to(state.device), self.offset.to(state.device)
        q[:, self.dst] = sign * state[:, self.src] + offset
        return q
