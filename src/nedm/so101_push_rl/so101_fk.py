"""Batched forward kinematics and contact-validity checks of the SO-101 + T scene in torch (for RL environments).

Port of nedm.so101_push.robot.ArmModel.link_poses (serial chain: pose[child] = pose[parent] @ F0[j] @ Rz(q_j) @ F1^-1[j],
the jaw locked at robot_cfg['jaw_lock_rad']) and of the collision-shape vertices of every arm link. Checks:
  * lowest point of the fingertips (gripper + jaw) and of every other link above the table top (z = 0),
  * signed distance of base .. wrist collision vertices to the two T boxes (intended gripper-T contact stays valid),
  * T height and tilt bounds, finite values.
A numpy cross-check against ArmModel is in check_against_numpy().
"""
from __future__ import annotations

import numpy as np
import torch

from nedm.so101_push.robot import ArmModel, link_collision_shapes, shape_points

OTHER = ("base", "shoulder", "upper_arm", "lower_arm", "wrist")
FINGER = ("gripper", "jaw")


def quat_to_rot(q):
    """[..., 4] wxyz -> [..., 3, 3]."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1).reshape(*q.shape[:-1], 3, 3)


class ArmFK(torch.nn.Module):
    def __init__(self, cfg: dict, tshape, device="cuda", dtype=torch.float32, max_points_per_link: int = 400):
        super().__init__()
        m = ArmModel(cfg["robot"], cfg["robot"]["description"])
        self.np_model = m
        self.joints = m.joints
        self.links = list(m.links)
        to = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
        self.register_buffer("T_base", to(m.T_base))
        self.register_buffer("F0", torch.stack([to(f) for f in m.F0]))
        self.register_buffer("F1inv", torch.stack([to(f) for f in m.F1inv]))
        self.parent = [m.parent[j] for j in range(len(m.joints))]
        self.child = [m.child[j] for j in range(len(m.joints))]
        self.jaw_lock = float(m.jaw_lock)
        self.register_buffer("lim", to(m.limits))
        pts = {}
        rng = np.random.default_rng(0)
        for ln in OTHER + FINGER:
            P = np.concatenate([shape_points(sh) for sh in link_collision_shapes(ln, cfg["robot"])])
            if len(P) > max_points_per_link:          # subsample the dense hull vertex sets (keep the extremes in z)
                keep = np.argsort(P[:, 2])[:20]
                rest = rng.choice(len(P), max_points_per_link - 20, replace=False)
                P = P[np.unique(np.r_[keep, rest])]
            pts[ln] = to(P)
        self.pts = pts
        self.ts = tshape
        self.dtype, self.device = dtype, device
        # T boxes in the T (COM) frame: centres and half sizes
        self.register_buffer("t_centres", to(np.stack([np.r_[tshape.bar_center, 0.0], np.r_[tshape.stem_center, 0.0]])))
        self.register_buffer("t_half", to(np.stack([np.r_[tshape.bar_l, tshape.bar_w, tshape.height] / 2,
                                                    np.r_[tshape.stem_w, tshape.stem_l, tshape.height] / 2])))
        self.t_height = float(tshape.height)

    def link_poses(self, q5):
        """q5 [B, 5] -> dict link -> [B, 4, 4]."""
        B = q5.shape[0]
        q6 = torch.cat((q5, torch.full((B, 1), self.jaw_lock, dtype=q5.dtype, device=q5.device)), -1)
        poses = {"base": self.T_base.expand(B, 4, 4)}
        for j in range(len(self.joints)):
            c, s = torch.cos(q6[:, j]), torch.sin(q6[:, j])
            Rz = torch.zeros(B, 4, 4, dtype=q5.dtype, device=q5.device)
            Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1] = c, -s, s, c
            Rz[:, 2, 2] = Rz[:, 3, 3] = 1.0
            poses[self.child[j]] = poses[self.parent[j]] @ self.F0[j] @ Rz @ self.F1inv[j]
        return poses

    def world_points(self, poses, ln):
        T = poses[ln]
        return self.pts[ln] @ T[:, :3, :3].transpose(1, 2) + T[:, None, :3, 3]          # [B, n, 3]

    def load_dense(self, path, links=("gripper_tip", "jaw_tip"), dtype=torch.float32):
        """Dense outer-surface points (scripts/preprocess/make_so101_dense_points.py) for contact distances; kept in
        float32 (4,096 environments x ~7,000 fingertip points per step)."""
        z = np.load(path)
        self.pts_dense = {ln: torch.as_tensor(z[ln], dtype=dtype, device=self.device) for ln in links}
        self.dense_dtype = dtype

    def finger_sd_dense(self, poses, t_state):
        """Smallest signed distance of the dense fingertip points (gripper_tip, jaw_tip) to the T [B]."""
        d = None
        for ln, link in (("gripper_tip", "gripper"), ("jaw_tip", "jaw")):
            T = poses[link].to(self.dense_dtype)
            P = self.pts_dense[ln] @ T[:, :3, :3].transpose(1, 2) + T[:, None, :3, 3]
            x = self.t_sdf(P, t_state.to(self.dense_dtype)).min(-1).values
            d = x if d is None else torch.minimum(d, x)
        return d

    def finger_face_dist_dense(self, poses, t_state):
        """Smallest signed distance of the dense fingertip points to the SIDE faces and to the TOP face of the T [B], [B].
        Each point is assigned the face of the nearer T box at its closest point (axis of the largest component of
        |p - centre| - half size in the T frame: z = top, x / y = side); as experiment path scripts/evaluation/so101_contact_faces.py."""
        t = t_state.to(self.dense_dtype)
        R = quat_to_rot(t[:, 3:7])
        inf = torch.full((len(t),), float("inf"), dtype=t.dtype, device=t.device)
        d_side, d_top = inf.clone(), inf.clone()
        for ln, link in (("gripper_tip", "gripper"), ("jaw_tip", "jaw")):
            T = poses[link].to(self.dense_dtype)
            P = self.pts_dense[ln] @ T[:, :3, :3].transpose(1, 2) + T[:, None, :3, 3]
            loc = (P - t[:, None, :3]) @ R
            dd, axs = None, None
            for b in range(2):
                q = (loc - self.t_centres[b].to(loc.dtype)).abs() - self.t_half[b].to(loc.dtype)
                d = q.clamp_min(0).norm(dim=-1) + q.max(-1).values.clamp_max(0)
                ax = q.argmax(-1)
                if dd is None:
                    dd, axs = d, ax
                else:
                    m = d < dd
                    dd, axs = torch.where(m, d, dd), torch.where(m, ax, axs)
            top = axs == 2
            big = torch.full_like(dd, float("inf"))
            d_top = torch.minimum(d_top, torch.where(top, dd, big).min(-1).values)
            d_side = torch.minimum(d_side, torch.where(~top, dd, big).min(-1).values)
        return d_side, d_top

    def finger_low_inside_dense(self, poses, t_state, inside_m=0.003, above_m=0.001):
        """Gripper came DOWN onto / into the T: the LOWEST dense fingertip point is at least inside_m inside the T outline
        (x-y, T frame, either box) and at most above_m above the top-face height. During a side push the lowest point is
        at the side face (low on the finger), so this stays false even when the NRD lets the finger go deep into a side
        face; in the NRD a finger that comes from above passes through the T (Chrono: rests on the top). Returns bool [B]."""
        t = t_state.to(self.dense_dtype)
        R = quat_to_rot(t[:, 3:7])
        P = torch.cat([self.pts_dense[ln] @ poses[link].to(self.dense_dtype)[:, :3, :3].transpose(1, 2) + poses[link].to(self.dense_dtype)[:, None, :3, 3]
                       for ln, link in (("gripper_tip", "gripper"), ("jaw_tip", "jaw"))], 1)
        low = P[torch.arange(len(P), device=P.device), P[..., 2].argmin(-1)]                 # [B, 3]
        loc = ((low - t[:, :3])[:, None, :] @ R)[:, 0]
        res = torch.zeros(len(t), dtype=torch.bool, device=t.device)
        for b in range(2):
            c, h = self.t_centres[b].to(loc.dtype), self.t_half[b].to(loc.dtype)
            q = (loc - c).abs() - h
            res = res | ((q[:, 0] <= -inside_m) & (q[:, 1] <= -inside_m) & (loc[:, 2] <= c[2] + h[2] + above_m))
        return res

    def finger_on_top_dense(self, poses, t_state, inside_m=0.002, height_m=0.001):
        """Gripper ON the T (robust test of the 10-07 review): some dense fingertip point is at least inside_m inside the T
        outline (x-y, in the T frame, either box) and within height_m of the top-face height. A finger beside a side face
        with a point near the top edge (the false 'top' of the face-axis rule) does not pass. Returns bool [B]."""
        t = t_state.to(self.dense_dtype)
        R = quat_to_rot(t[:, 3:7])
        on = torch.zeros(len(t), dtype=torch.bool, device=t.device)
        for ln, link in (("gripper_tip", "gripper"), ("jaw_tip", "jaw")):
            T = poses[link].to(self.dense_dtype)
            P = self.pts_dense[ln] @ T[:, :3, :3].transpose(1, 2) + T[:, None, :3, 3]
            loc = (P - t[:, None, :3]) @ R
            for b in range(2):
                c, h = self.t_centres[b].to(loc.dtype), self.t_half[b].to(loc.dtype)
                q = (loc - c).abs() - h
                inside = (q[..., 0] <= -inside_m) & (q[..., 1] <= -inside_m)
                at_top = (loc[..., 2] - (c[2] + h[2])).abs() <= height_m
                on = on | (inside & at_top).any(-1)
        return on

    def t_sdf(self, P, t_state):
        """Signed distance of points P [B, n, 3] to the T (union of two boxes) at t_state [B, 13]."""
        R = quat_to_rot(t_state[:, 3:7])                                                # [B, 3, 3]
        local = (P - t_state[:, None, :3]) @ R                                         # into the T frame
        d = None
        for b in range(2):
            q = (local - self.t_centres[b].to(P.dtype)).abs() - self.t_half[b].to(P.dtype)
            out = q.clamp_min(0).norm(dim=-1) + q.max(-1).values.clamp_max(0)
            d = out if d is None else torch.minimum(d, out)
        return d                                                                        # [B, n]

    @torch.no_grad()
    def check(self, q5, t_state, tip_floor=0.0005, link_floor=0.0005, link_gap=0.002, t_z_tol=(0.002, 0.01), max_tilt_deg=15.0):
        """Returns (valid [B] bool, reasons dict name -> [B] bool, diag dict)."""
        q5 = q5.to(self.dtype)
        t_state = t_state.to(self.dtype)
        poses = self.link_poses(q5)
        tip_z = torch.stack([self.world_points(poses, ln)[..., 2].min(-1).values for ln in FINGER]).min(0).values
        oth = [self.world_points(poses, ln) for ln in OTHER]
        link_z = torch.stack([P[..., 2].min(-1).values for P in oth[1:]]).min(0).values   # the base is fixed on the table
        gap = torch.stack([self.t_sdf(P, t_state).min(-1).values for P in oth]).min(0).values
        h = self.t_height / 2
        tilt = torch.rad2deg(torch.acos((1 - 2 * (t_state[:, 4] ** 2 + t_state[:, 5] ** 2)).clamp(-1, 1)))
        finite = torch.isfinite(q5).all(-1) & torch.isfinite(t_state).all(-1)
        reasons = dict(
            nonfinite=~finite,
            joint_limit=((q5 < self.lim[:, 0] - 1e-3) | (q5 > self.lim[:, 1] + 1e-3)).any(-1),
            tip_table=tip_z < tip_floor,
            link_table=link_z < link_floor,
            link_t_contact=gap < link_gap,
            t_height=(t_state[:, 2] < h - t_z_tol[0]) | (t_state[:, 2] > h + t_z_tol[1]),
            t_tilt=tilt > max_tilt_deg,
        )
        bad = torch.zeros_like(finite)
        for v in reasons.values():
            bad |= v
        return ~bad, reasons, dict(tip_z=tip_z, link_z=link_z, link_t_gap=gap, tilt_deg=tilt)


def check_against_numpy(cfg, tshape, n=64, device="cpu"):
    """Max difference of link poses vs ArmModel.link_poses over random joint angles."""
    fk = ArmFK(cfg, tshape, device=device, dtype=torch.float64)
    m = fk.np_model
    rng = np.random.default_rng(1)
    q = rng.uniform(m.limits[:, 0], m.limits[:, 1], size=(n, 5))
    tp = fk.link_poses(torch.as_tensor(q, dtype=torch.float64, device=device))
    err = 0.0
    for i in range(n):
        ref = m.link_poses(q[i])
        for ln in ref:
            err = max(err, float(np.abs(ref[ln] - tp[ln][i].cpu().numpy()).max()))
    return err
