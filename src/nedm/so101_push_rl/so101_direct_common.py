"""Direct joint-command PPO for the SO-101 push-T (no push generator): observation, action maps and reward pieces shared
by the learned (NRD) environment and the Chrono evaluation.

Observation (129 values; physical scales, frozen; history normalised with the tracking study's RL-train statistics):
  [0:96)    four NRD state frames (23 values each, oldest first, missing entries zero) + validity masks
  [96:101)  previous applied q_cmd, (q_cmd - center) / scale
  [101:106) previous command change (q_cmd(t-1) - q_cmd(t-2)) / delta_max
  [106:114) object-goal: goal xy - T xy in the T frame / 0.02 m (2); wrapped goal yaw - T yaw / 0.1 rad, sin, cos (3);
            T planar velocity in the T frame / 0.05 m/s (2); T yaw rate / 0.5 rad/s (1)
  [114:127) fingertip-object geometry (forward kinematics of the NRD joint angles):
            TCP xy - T xy in the T frame / 0.05 m (2); TCP height / 0.01 m (1); gripper heading - T yaw: sin, cos (2);
            TCP velocity (Jacobian x averaged joint rates) in the T frame / 0.1 m/s (3);
            fingertip-T signed distance (gripper + jaw collision points) / 0.01 m (1);
            closest T boundary point - TCP, in the T frame / 0.02 m (2); lowest fingertip height / 0.005 m (1);
            smallest base..wrist link - T gap / 0.02 m (1)
  [127:129) time left / horizon, time left / 4 s
Actions (raw latent a, PPO keeps a):
  joint_inc  dq = delta_max tanh(a) [5]; optional low-pass inc = beta inc_prev + (1 - beta) dq; q_cmd = clip(prev + inc)
  ee_inc     (comparison only) a [4] -> TCP increment (dx, dy, dz) = dp_max tanh(a[:3]), gripper yaw increment
             dyaw_max tanh(a[3]); joint increment = weighted least squares on the TCP Jacobian (rows vx vy vz wx wy wz,
             tilt rows target 0, weights 1 1 1 0.05 0.05 0.2); q_cmd = clip(prev + inc)
  lead band  (optional, both maps) q_cmd is also clipped to q + [lead_lo, lead_hi], q = current joint angles: the command
             never runs further ahead of (or behind) the arm than in the collector data (p0.5 / p99.5 of q_cmd - q over
             all phases), where the NRD is accurate
"""
from __future__ import annotations

import math

import torch

from nedm.so101_push_rl import so101_push_common as C
from nedm.so101_push_rl.so101_goal_common import goal_error, t_frame, t_pose

OBS_DIM = 129
N_HIST = C.HISTORY * C.N_STATE + C.HISTORY


def closest_point_on_t(local_xy, centres, halves):
    """local_xy [N,2] in the T frame; boxes (centres [2,2], half sizes [2,2]) -> closest boundary / interior point [N,2]
    of the union (per box: clamp to the box; choose the nearer box)."""
    best, bestd = None, None
    for b in range(centres.shape[0]):
        lo, hi = centres[b] - halves[b], centres[b] + halves[b]
        p = torch.minimum(torch.maximum(local_xy, lo), hi)
        d = (p - local_xy).norm(dim=-1)
        if best is None:
            best, bestd = p, d
        else:
            m = d < bestd
            best = torch.where(m[:, None], p, best)
            bestd = torch.where(m, d, bestd)
    return best


class DirectGeometry:
    """FK geometry of the current state (shared by observation, reward shaping and validity). dense: the fingertip-T
    signed distance uses the dense fingertip surface points (fk.load_dense) instead of the shape vertices (vertex
    distances miss contacts on faces and edges: surface points are typically 4 mm, p95 12-16 mm, from a vertex)."""

    def __init__(self, kin, fk, dense=False, faces=False):
        self.kin, self.fk, self.dense, self.faces = kin, fk, dense, faces
        self.centres = fk.t_centres[:, :2]
        self.halves = fk.t_half[:, :2]

    def __call__(self, s23):
        q, qd = s23[:, :5], s23[:, 5:10]
        p, R, J = self.kin.tcp_jacobian(q)
        v = torch.einsum("nij,nj->ni", J[:, :3], qd)
        poses = self.fk.link_poses(q.to(self.fk.dtype))
        tstate = s23[:, 10:23].to(self.fk.dtype)
        finger = torch.cat([self.fk.world_points(poses, ln) for ln in ("gripper", "jaw")], 1)
        f_sd = self.fk.finger_sd_dense(poses, tstate).to(self.fk.dtype) if self.dense else self.fk.t_sdf(finger, tstate).min(-1).values
        f_z = finger[..., 2].min(-1).values
        others = [self.fk.world_points(poses, ln) for ln in ("base", "shoulder", "upper_arm", "lower_arm", "wrist")]
        link_gap = torch.stack([self.fk.t_sdf(P, tstate).min(-1).values for P in others]).min(0).values
        link_z = torch.stack([P[..., 2].min(-1).values for P in others[1:]]).min(0).values
        xy, yaw = t_pose(s23)
        local = t_frame(p[:, :2] - xy, yaw)
        cp = closest_point_on_t(local, self.centres.to(local.dtype), self.halves.to(local.dtype))
        tilt = torch.rad2deg(torch.acos(R[:, 2, 2].clamp(-1, 1)))           # gripper z axis vs world z (data: <= 1.2 deg)
        extra = {}
        if self.faces:                                     # contact with the side faces / the top face (dense points)
            d_side, d_top = self.fk.finger_face_dist_dense(poses, tstate)
            extra = dict(d_side=d_side.to(s23.dtype), d_top=d_top.to(s23.dtype), on_top=self.fk.finger_on_top_dense(poses, tstate),
                         low_inside=self.fk.finger_low_inside_dense(poses, tstate))
        return dict(**extra, q=q, tcp=p, R=R, J=J, tcp_v=v, finger_sd=f_sd.to(s23.dtype), finger_z=f_z.to(s23.dtype), tilt_deg=tilt,
                    link_gap=link_gap.to(s23.dtype), link_z=link_z.to(s23.dtype), tcp_local=local, closest_local=cp,
                    gyaw=self.kin.gripper_yaw(R), t_xy=xy, t_yaw=yaw)


class DirectObs:
    def __init__(self, hist_norm, amap, delta_max):
        self.hmean, self.hscale = hist_norm.mean[:N_HIST], hist_norm.scale[:N_HIST]
        self.cc, self.cs = amap.center, amap.scale
        self.dmax = delta_max

    def __call__(self, hist23, ok, prev_cmd, prev_inc, goal, t_left, horizon, geo):
        N, dev = hist23.shape[0], hist23.device
        cur = hist23[:, -1]
        h = torch.cat(((hist23 * ok[..., None]).reshape(N, -1), ok.to(hist23.dtype)), -1)
        h = (h - self.hmean.to(dev)) / self.hscale.to(dev)
        cmd = (prev_cmd - self.cc.to(dev)) / self.cs.to(dev)
        inc = prev_inc / self.dmax.to(dev)
        xy, yaw = geo["t_xy"], geo["t_yaw"]
        ge = t_frame(goal[:, :2] - xy, yaw) / 0.02
        dy = C.wrap(goal[:, 2] - yaw)
        task = torch.cat((ge, (dy / 0.1)[:, None], torch.sin(dy)[:, None], torch.cos(dy)[:, None],
                          t_frame(cur[:, 17:19], yaw) / 0.05, (cur[:, 22] / 0.5)[:, None]), -1)
        dg = C.wrap(geo["gyaw"] - yaw)
        vt = torch.cat((t_frame(geo["tcp_v"][:, :2], yaw), geo["tcp_v"][:, 2:3]), -1) / 0.1
        grip = torch.cat((geo["tcp_local"] / 0.05, (geo["tcp"][:, 2] / 0.01)[:, None], torch.sin(dg)[:, None], torch.cos(dg)[:, None],
                          vt, (geo["finger_sd"].clamp(max=0.1) / 0.01)[:, None], (geo["closest_local"] - geo["tcp_local"]) / 0.02,
                          (geo["finger_z"] / 0.005)[:, None], (geo["link_gap"].clamp(max=0.2) / 0.02)[:, None]), -1)
        tl = torch.stack((t_left / horizon.clamp(min=1), t_left * 0.02 / 4.0), -1)
        out = torch.cat((h, cmd, inc, task, grip, tl.to(h.dtype)), -1)
        assert out.shape[1] == OBS_DIM, out.shape
        return torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-50, 50).to(torch.float32)


def clip_cmd(cmd, lo, hi, lead, geo):
    """Joint limits, then the optional lead band around the current joint angles."""
    dev = cmd.device
    cmd = torch.minimum(torch.maximum(cmd, lo.to(dev)), hi.to(dev))
    if lead is not None:
        q = geo["q"].to(cmd.dtype)
        cmd = torch.minimum(torch.maximum(cmd, q + lead[0].to(dev)), q + lead[1].to(dev))
    return cmd


class JointIncMap:
    def __init__(self, delta_max, lo, hi, beta=0.0, lead=None):
        self.dmax, self.lo, self.hi, self.beta, self.lead = delta_max, lo, hi, float(beta), lead

    def __call__(self, a, prev_cmd, prev_inc, geo=None):
        dq = self.dmax.to(prev_cmd.device) * torch.tanh(a.to(prev_cmd.dtype))
        inc = self.beta * prev_inc + (1 - self.beta) * dq if self.beta > 0 else dq
        cmd = clip_cmd(prev_cmd + inc, self.lo, self.hi, self.lead, geo)
        return cmd, cmd - prev_cmd


class EEIncMap:
    """Comparison interface: gripper-space increments mapped to joint increments with the TCP Jacobian."""

    def __init__(self, lo, hi, dp_max=0.004, dyaw_max=0.05, beta=0.0, w=(1.0, 1.0, 1.0, 0.05, 0.05, 0.2), lead=None):
        self.lo, self.hi, self.dp, self.dyaw, self.beta, self.lead = lo, hi, dp_max, dyaw_max, float(beta), lead
        self.w = torch.tensor(w, dtype=torch.float64)

    def __call__(self, a, prev_cmd, prev_inc, geo):
        a = torch.tanh(a.to(prev_cmd.dtype))
        N = a.shape[0]
        target = torch.zeros(N, 6, dtype=prev_cmd.dtype, device=prev_cmd.device)
        target[:, :3] = self.dp * a[:, :3]
        target[:, 5] = self.dyaw * a[:, 3]
        w = self.w.to(prev_cmd.device)
        A = geo["J"] * w[None, :, None]                                   # [N, 6, 5]
        AtA = A.transpose(1, 2) @ A + 1e-8 * torch.eye(5, dtype=A.dtype, device=A.device)
        dq = torch.linalg.solve(AtA, (A.transpose(1, 2) @ (target * w)[..., None]))[..., 0]   # weighted least squares
        inc = self.beta * prev_inc + (1 - self.beta) * dq if self.beta > 0 else dq
        cmd = clip_cmd(prev_cmd + inc, self.lo, self.hi, self.lead, geo)
        return cmd, cmd - prev_cmd


def make_action_map(ac, lo, hi, dtype=torch.float64, device="cpu"):
    """Action map from the env config action block (shared by the NRD env and the Chrono controller)."""
    lead = None
    if ac.get("lead_lo") is not None:
        lead = (torch.tensor(ac["lead_lo"], dtype=dtype, device=device), torch.tensor(ac["lead_hi"], dtype=dtype, device=device))
    if ac["kind"] == "ee_inc":
        return EEIncMap(lo, hi, ac["dp_max"], ac["dyaw_max"], ac["beta"], lead=lead)
    return JointIncMap(torch.tensor(ac["delta_max"], dtype=dtype, device=device), lo, hi, ac["beta"], lead=lead)


def potential(s23, goal, yaw_weight, pos_scale=0.010, yaw_scale_deg=5.0):
    pe, ye = goal_error(s23, goal)
    return pe / pos_scale + yaw_weight * ye.abs() / math.radians(yaw_scale_deg)
