"""Observation, action maps, reward and success rules of the SO-101 push-T goal-reaching task (PPO from scratch in the
frozen NRD), shared by the learned environment and the Chrono evaluation. Plan: PPO_GOAL_REACH_PLAN.md (10-06).

State order (23 values, as so101_push_common): arm q 0-4, averaged q_dot 5-9, T x y z 10-12, quaternion 13-16,
averaged velocity 17-19, averaged angular velocity 20-22. Goal = T planar pose (x, y, yaw).

Observation (fixed physical scales, frozen; history uses the RL-train statistics of the tracking study):
  [0:92)   four state-history entries (oldest first), normalised like the tracking study; [92:96) masks
  [96:101) previous applied q_cmd, (q_cmd - center) / scale
  [101:109) task geometry: goal xy error in the T frame / 0.02 m (2); wrapped goal yaw error / 0.1 rad, sin, cos (3);
            T planar speed in the T frame / 0.05 m/s (2); T yaw rate / 0.5 rad/s (1)
  [109:120) gripper: TCP minus T COM in the T frame / 0.05 m (2); TCP height above the table / 0.01 m (1);
            gripper heading minus T yaw: sin, cos (2); TCP velocity in the T frame / 0.1 m/s (3);
            fingertip height / 0.005 m (1); smallest link-T gap (base..wrist) / 0.02 m (1); time left / horizon (1)
  [120:124) decoder (variant B; zeros for A): target speed / 0.1 m/s, planned distance left / 0.02 m,
            push line heading minus T yaw: sin, cos
"""
from __future__ import annotations

import math

import torch

from nedm.so101_push_rl import so101_push_common as C

OBS_DIM = 124
N_HIST = C.HISTORY * C.N_STATE + C.HISTORY        # 96


def t_frame(xy, yaw):
    """Rotate world xy vectors [..., 2] into the frame of a body with yaw [...]."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c * xy[..., 0] + s * xy[..., 1], -s * xy[..., 0] + c * xy[..., 1]), -1)


def t_pose(state23):
    return state23[..., C.T_XY], C.yaw_of_q(state23[..., C.T_QUAT])


def goal_error(state23, goal):
    """-> (position error [m], wrapped yaw error goal - T [rad])."""
    xy, yaw = t_pose(state23)
    return (goal[..., :2] - xy).norm(dim=-1), C.wrap(goal[..., 2] - yaw)


def potential(state23, goal, yaw_weight=0.5, pos_scale=0.010, yaw_scale_deg=5.0):
    """D = |xy - goal| / 10 mm + yaw_weight |wrap(yaw - goal)| / 5 deg (the plan's progress potential)."""
    pe, ye = goal_error(state23, goal)
    return pe / pos_scale + yaw_weight * ye.abs() / math.radians(yaw_scale_deg)


def radians(deg):
    """math.radians for a number; the same product (deg x (pi / 180), bitwise equal) for a tensor."""
    return deg * (math.pi / 180.0) if torch.is_tensor(deg) else math.radians(deg)


def settled_inside(state23, goal, use_yaw=True, pos_tol=0.005, yaw_tol_deg=3.0, v_tol=0.005, w_tol_deg=5.0):
    """Inside the success tolerance and slow (T planar speed < 5 mm/s, yaw rate < 5 deg/s, averaged rates). The
    tolerances are numbers or tensors that broadcast with the batch shape of state23 (per-env tolerances)."""
    pe, ye = goal_error(state23, goal)
    ok = (pe < pos_tol) & (state23[..., 17:19].norm(dim=-1) < v_tol) & (state23[..., 22].abs() < radians(w_tol_deg))
    if use_yaw:
        ok = ok & (ye.abs() < radians(yaw_tol_deg))
    return ok


class GoalObs:
    """Builds the 124-value observation. hist_norm: the tracking study's ObsNorm (first 96 entries are used),
    amap: its ActionMap (center / scale of q_cmd)."""

    def __init__(self, kin, fk, hist_norm, amap):
        self.kin, self.fk = kin, fk
        self.hmean, self.hscale = hist_norm.mean[:N_HIST], hist_norm.scale[:N_HIST]
        self.cc, self.cs = amap.center, amap.scale

    def gripper(self, state23):
        """FK quantities of the current joint angles: TCP position, gripper yaw, TCP velocity (Jacobian x averaged rates)."""
        q, qd = state23[:, :5], state23[:, 5:10]
        p, R, J = self.kin.tcp_jacobian(q)
        v = torch.einsum("nij,nj->ni", J[:, :3], qd)
        return p, self.kin.gripper_yaw(R), v

    def __call__(self, hist23, ok, prev_cmd, goal, t_left_frac, diag, dec=None):
        """hist23 [N,4,23], ok [N,4], prev_cmd [N,5], goal [N,3], t_left_frac [N], diag: FK check diagnostics of the
        current state (tip_z, link_t_gap), dec: optional dict(v, rem, heading) of the decoder."""
        N = hist23.shape[0]
        dev = hist23.device
        cur = hist23[:, -1]
        h = torch.cat(((hist23 * ok[..., None]).reshape(N, -1), ok.to(hist23.dtype)), -1)
        h = (h - self.hmean.to(dev)) / self.hscale.to(dev)
        cmd = (prev_cmd - self.cc.to(dev)) / self.cs.to(dev)
        xy, yaw = t_pose(cur)
        ge = t_frame(goal[:, :2] - xy, yaw) / 0.02
        dy = C.wrap(goal[:, 2] - yaw)
        tv = t_frame(cur[:, 17:19], yaw) / 0.05
        task = torch.cat((ge, (dy / 0.1)[:, None], torch.sin(dy)[:, None], torch.cos(dy)[:, None], tv, (cur[:, 22] / 0.5)[:, None]), -1)
        p, gyaw, v = self.gripper(cur)
        rel = t_frame(p[:, :2] - xy, yaw) / 0.05
        dg = C.wrap(gyaw - yaw)
        vt = torch.cat((t_frame(v[:, :2], yaw), v[:, 2:3]), -1) / 0.1
        grip = torch.cat((rel, (p[:, 2] / 0.01)[:, None], torch.sin(dg)[:, None], torch.cos(dg)[:, None], vt,
                          (diag["tip_z"] / 0.005)[:, None], (diag["link_t_gap"].clamp(max=0.2) / 0.02)[:, None],
                          t_left_frac[:, None]), -1)
        if dec is None:
            d = torch.zeros(N, 4, dtype=hist23.dtype, device=dev)
        else:
            dh = C.wrap(dec["heading"] - yaw)
            d = torch.stack((dec["v"] / 0.1, dec["rem"] / 0.02, torch.sin(dh), torch.cos(dh)), -1)
        out = torch.cat((h, cmd, task, grip, d), -1)
        assert out.shape[1] == OBS_DIM, out.shape
        return out.to(torch.float32)


class IncrementMap:
    """Variant A: q_cmd = joint-limit projection of (previous q_cmd + delta_max tanh(z)). A geometric filter holds the
    command (zero increment) when the commanded configuration would put a fingertip / link near the table or a
    non-finger link near the T (the filter decision uses FK of the candidate command; it is logged)."""

    def __init__(self, delta_max, lo, hi):
        self.delta_max, self.lo, self.hi = delta_max, lo, hi

    def __call__(self, z, prev_cmd):
        cand = prev_cmd + self.delta_max.to(prev_cmd.device) * torch.tanh(z.to(prev_cmd.dtype))
        return torch.minimum(torch.maximum(cand, self.lo.to(prev_cmd.device)), self.hi.to(prev_cmd.device))


class StrokeMap:
    """Variant B: z [N,2] -> remaining stroke distance L in [0, L_max] and target speed v in [v_min, v_max]
    (tanh squashed); the decoder turns them into q_cmd."""

    def __init__(self, L_max=0.06, v_min=0.03, v_max=0.3):
        self.L_max, self.v_min, self.v_max = L_max, v_min, v_max

    def __call__(self, z):
        u = 0.5 * (torch.tanh(z.to(torch.float64)) + 1.0)
        return self.L_max * u[:, 0], self.v_min + (self.v_max - self.v_min) * u[:, 1]
