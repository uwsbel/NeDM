"""Batched torch kinematics of the MuJoCo Menagerie SO101 arm of the grasp collector (no MuJoCo in a step).

Frames and units: world = arm base frame (base at the origin, z up, table top at z = 0), metres, radians, quaternions
w x y z. Joint vector q [..., 6] in the collector's order (``JOINTS``): shoulder_pan, shoulder_lift, elbow_flex,
wrist_flex, wrist_roll, gripper (= the moving jaw). Every joint is a hinge about its body's local z axis at the body
origin, so a body pose is R = R_parent R(quat) Rz(q), p = p_parent + R_parent pos (the rule of the NRD's
``model.KinematicChain`` and of MuJoCo for this model). The body constants below are the compiled ``body_pos`` /
``body_quat`` of ``collect.so101_grasp.build_spec`` (MuJoCo 3.15, Menagerie rev 0059d43); ``tests/so101_policy/
test_kin.py`` checks them, the FK, the lowest point and the IK against MuJoCo.

- ``fk(q)``: grasp point (``GRASP_POINT`` in the gripper frame, world) and the gripper rotation (columns = gripper axes
  in the world; top-down = gripper z up, fingertips down, closing axis x at the yaw, ``topdown``).
- ``body_poses(q)``: world pose of every arm body (``BODIES``), for rendering.
- ``lowest_point(q)``: lowest world z of the collision geoms of the gripper and the moving jaw (``Scene.lowest`` with
  ``which`` = those bodies: every vertex of boxes and meshes, capsule ends and sphere centres minus their radius). The
  points come from MuJoCo once (``collision_points``) or from a saved ``.npz`` (``save_points`` / ``points=``).
- ``ik`` / ``pose_ik`` / ``ik_topdown``: batched damped least squares with the collector's rules (``Scene.ik``,
  ``Scene.pose_ik``, ``Scene.solve``): the same iterations, stopping rules, tolerances, seeds and seed order, so a
  world gets the same solution branch as the collector would.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# body name, parent index in BODIES (-1 = world), compiled body_pos, compiled body_quat (w x y z), joint index or None
_TREE = (
    ("base", -1, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), None),
    ("shoulder", 0, (0.0388353, 0.0, 0.0624), (0.0, 0.0, -1.0, 0.0), 0),
    ("upper_arm", 1, (-0.0303992, -0.0182778, -0.0542), (0.5, -0.5, -0.5, -0.5), 1),
    ("lower_arm", 2, (-0.11257, -0.028, 0.0), (0.7071067811865475, 0.0, 0.0, 0.7071067811865475), 2),
    ("wrist", 3, (-0.1349, 0.0052, 0.0), (0.7071067811865475, 0.0, 0.0, -0.7071067811865475), 3),
    ("gripper", 4, (5.55112e-17, -0.0611, 0.0181),
     (0.017209108230571014, -0.017209108230571014, 0.7068973380865913, 0.7068973380865913), 4),
    ("camera_mount", 5, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), None),
    ("moving_jaw_so101_v1", 5, (0.0202, 0.0188, -0.0234), (0.7071067811865475, 0.7071067811865475, 0.0, 0.0), 5),
)
BODIES = tuple(t[0] for t in _TREE)
GRIPPER, JAW = BODIES.index("gripper"), BODIES.index("moving_jaw_so101_v1")
GRASP_POINT = (0.0105, 0.0, -0.092)            # grasp point in the gripper frame (collector grasp_point_m)
PAN_AXIS_XY = (0.0388, 0.0)                    # vertical pan axis in the world (collector workspace)
# applied-target range of the collector (Scene.tlo / thi: joint range 0.05 rad inside, within the ctrlrange)
TLO = (-1.8698599999999999, -1.6953293, -1.64, -1.608063, -2.6938473000000003, -0.12453299999999999)
THI = (1.8698599999999999, 1.6953293, 1.64, 1.608063, 2.7912063000000003, 1.6953292)
IK_TOL_M, IK_TOL_RAD = 0.0005, math.radians(2.0)   # collector cfg["ik"]
GUARD_BODIES = ("gripper", "moving_jaw_so101_v1")
_POINTS_CACHE: dict = {}


# ---- small rotation helpers (batched, any float dtype) ----------------------------------------------------------
def quat_mat(q):
    """Unit quaternions [..., 4] (w x y z) -> rotation matrices [..., 3, 3]."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1).reshape(*q.shape[:-1], 3, 3)


def mat_quat(r):
    """Rotation matrices [..., 3, 3] -> unit quaternions [..., 4] (w x y z, w >= 0), branch-free Shepperd form."""
    m00, m11, m22 = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
    t = torch.stack((1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1)
    k = t.argmax(-1, keepdim=True)
    s = torch.sqrt(t.gather(-1, k).clamp_min(1e-30))[..., 0] * 2          # 4 * |largest component|
    a = torch.stack((s / 4, (r[..., 2, 1] - r[..., 1, 2]) / s, (r[..., 0, 2] - r[..., 2, 0]) / s,
                     (r[..., 1, 0] - r[..., 0, 1]) / s), -1)
    b = torch.stack(((r[..., 2, 1] - r[..., 1, 2]) / s, s / 4, (r[..., 0, 1] + r[..., 1, 0]) / s,
                     (r[..., 0, 2] + r[..., 2, 0]) / s), -1)
    c = torch.stack(((r[..., 0, 2] - r[..., 2, 0]) / s, (r[..., 0, 1] + r[..., 1, 0]) / s, s / 4,
                     (r[..., 1, 2] + r[..., 2, 1]) / s), -1)
    d = torch.stack(((r[..., 1, 0] - r[..., 0, 1]) / s, (r[..., 0, 2] + r[..., 2, 0]) / s,
                     (r[..., 1, 2] + r[..., 2, 1]) / s, s / 4), -1)
    q = torch.where(k == 0, a, torch.where(k == 1, b, torch.where(k == 2, c, d)))
    q = q / q.norm(dim=-1, keepdim=True)
    return torch.where(q[..., :1] < 0, -q, q)


def quat_rotvec(q):
    """Unit quaternions [..., 4] -> rotation vectors [..., 3] (the shorter way round, as mju_quat2Vel of w >= 0)."""
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    n = v.norm(dim=-1, keepdim=True)
    ang = 2 * torch.atan2(n, q[..., :1])
    return v * torch.where(n > 1e-15, ang / n.clamp_min(1e-300), torch.full_like(n, 2.0))


def rotvec(r):
    """Rotation matrices [..., 3, 3] -> rotation vectors [..., 3] (angle in [0, pi])."""
    return quat_rotvec(mat_quat(r))


def rotvec_fast(r):
    """rotvec for rotations below about 170 deg with fewer operations (axis from the skew part, angle by atan2; the
    direction degrades near 180 deg). Used where the error angle stays small (ik_track)."""
    v = torch.stack((r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]), -1)
    n = v.norm(dim=-1, keepdim=True)
    ang = torch.atan2(0.5 * n, 0.5 * (r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2] - 1)[..., None])
    return v * torch.where(n > 1e-15, ang / n.clamp_min(1e-300), torch.full_like(n, 0.5))


def rotz(yaw):
    """Rotations about the world z axis [..., 3, 3]; topdown(yaw) of the collector (gripper z up, x at the yaw)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    o, z = torch.ones_like(c), torch.zeros_like(c)
    return torch.stack((c, -s, z, s, c, z, z, z, o), -1).reshape(*yaw.shape, 3, 3)


topdown = rotz


def yaw_of(r):
    """Yaw of the gripper closing axis (first column) as Scene.solve reads it: atan2(r10, r00)."""
    return torch.atan2(r[..., 1, 0], r[..., 0, 0])


def tilt_of(r):
    """Angle (rad) between the gripper z axis and the world z axis (0 = top-down)."""
    return torch.acos(r[..., 2, 2].clamp(-1.0, 1.0))


def wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def _turn_z(r, a):
    """r @ Rz(a) for rotations r [..., 3, 3] and angles a [...] (column mix, fewer ops than a matrix product)."""
    c, s = torch.cos(a)[..., None], torch.sin(a)[..., None]
    x, y = r[..., :, 0], r[..., :, 1]
    return torch.stack((x * c + y * s, y * c - x * s, r[..., :, 2]), -1)


# ---- collision points of the guard bodies ------------------------------------------------------------------------
def collision_points(bodies=GUARD_BODIES):
    """{body: (points [N, 3] in the body frame, radius [N])} of the arm collision geoms of the collector scene, exactly
    as ``Scene.lowest`` reads them (``so101_grasp.geom_points``). Needs mujoco and the vendor model; cached."""
    key = tuple(bodies)
    if key not in _POINTS_CACHE:
        import mujoco
        import numpy as np

        from nedm.contact_nrd.collect import so101_grasp as g
        s = g.Scene(g.load_config())
        m = s.m
        out = {}
        for name in bodies:
            b = m.body(name).id
            pts, rad = [], []
            for gid, (p, r) in zip(s.arm_geoms, s.points):
                if m.geom_bodyid[gid] != b:
                    continue
                rot = np.zeros(9)
                mujoco.mju_quat2Mat(rot, m.geom_quat[gid])
                pts.append(p @ rot.reshape(3, 3).T + m.geom_pos[gid])
                rad.append(np.full(len(p), r))
            out[name] = (np.concatenate(pts), np.concatenate(rad))
        _POINTS_CACHE[key] = out
    return _POINTS_CACHE[key]


def save_points(path, bodies=GUARD_BODIES):
    """Write the collision points to an .npz (for machines without mujoco)."""
    import numpy as np
    pts = collision_points(bodies)
    np.savez(path, bodies=np.array(list(bodies)), **{f"{b}_points": p for b, (p, _) in pts.items()},
             **{f"{b}_radius": r for b, (_, r) in pts.items()})


def load_points(path):
    import numpy as np
    f = np.load(path)
    return {str(b): (f[f"{b}_points"], f[f"{b}_radius"]) for b in f["bodies"]}


# ---- the kinematics ----------------------------------------------------------------------------------------------
class SO101Kin:
    """Batched FK, Jacobians, lowest point and IK of the SO101 on one device (float64 by default).

    points: None (collision points from MuJoCo on the first ``lowest_point`` call), a path to a ``save_points`` .npz,
    or the dict of ``collision_points``.
    """

    def __init__(self, device="cpu", dtype=torch.float64, points=None):
        self.device, self.dtype = torch.device(device), dtype
        f = dict(device=self.device, dtype=dtype)
        self.pos = torch.tensor([t[2] for t in _TREE], **f)
        self.rot = quat_mat(torch.tensor([t[3] for t in _TREE], **f))
        self.parent = [t[1] for t in _TREE]
        self.joint = [t[4] for t in _TREE]
        self.point = torch.tensor(GRASP_POINT, **f)
        self.tlo, self.thi = torch.tensor(TLO, **f), torch.tensor(THI, **f)
        self.pan_axis = torch.tensor(PAN_AXIS_XY, **f)
        self._points_src = points
        self._guard = None

    # -- forward kinematics --
    def body_poses(self, q):
        """q [..., >=5] (joint 5 = jaw needed for the moving jaw; missing -> 0) -> (pos [..., 8, 3], rot [..., 8, 3, 3])
        of BODIES in the world."""
        q = q.to(self.dtype)
        lead = q.shape[:-1]
        ps, rs = [], []
        for k in range(len(_TREE)):
            par, j = self.parent[k], self.joint[k]
            if par < 0:
                p = self.pos[k].expand(*lead, 3)
                r = self.rot[k].expand(*lead, 3, 3)
            else:
                p = ps[par] + (rs[par] @ self.pos[k])
                r = rs[par] @ self.rot[k]
            if j is not None:
                r = _turn_z(r, q[..., j] if j < q.shape[-1] else torch.zeros(lead, dtype=q.dtype, device=q.device))
            ps.append(p)
            rs.append(r)
        return torch.stack(ps, -2), torch.stack(rs, -3)

    def _chain(self, q):
        """Bodies up to the gripper only (5 arm joints): positions and rotations lists."""
        lead = q.shape[:-1]
        p = self.pos[0].expand(*lead, 3)
        r = self.rot[0].expand(*lead, 3, 3)
        ps, rs = [p], [r]
        for k in range(1, GRIPPER + 1):
            p = p + (r @ self.pos[k])
            r = _turn_z(r @ self.rot[k], q[..., self.joint[k]])
            ps.append(p)
            rs.append(r)
        return ps, rs

    def fk(self, q):
        """q [..., >=5] -> grasp point [..., 3] (world) and gripper rotation [..., 3, 3]."""
        ps, rs = self._chain(q.to(self.dtype))
        return ps[-1] + rs[-1] @ self.point, rs[-1]

    def jacobian(self, q):
        """q [..., >=5] -> (jp [..., 3, 5], jr [..., 3, 5], grasp point, gripper rotation): world-frame Jacobians of the
        grasp point and the gripper angular velocity in the 5 arm joints (mj_jac of the collector's Scene.ik)."""
        ps, rs = self._chain(q.to(self.dtype))
        p = ps[-1] + rs[-1] @ self.point
        axes = torch.stack([rs[k][..., :, 2] for k in range(1, GRIPPER + 1)], -1)          # [..., 3, 5]
        lever = p[..., :, None] - torch.stack([ps[k] for k in range(1, GRIPPER + 1)], -1)   # [..., 3, 5]
        return torch.cross(axes, lever, dim=-2), axes, p, rs[-1]

    # -- arm-table guard --
    def _guard_points(self):
        if self._guard is None:
            src = self._points_src
            pts = load_points(src) if isinstance(src, (str, Path)) else (src or collision_points())
            f = dict(device=self.device, dtype=self.dtype)
            self._guard = [(BODIES.index(b), torch.as_tensor(p, **f), torch.as_tensor(r, **f))
                           for b, (p, r) in pts.items()]
        return self._guard

    def lowest_point(self, q, chunk=4096, per_body=False):
        """Lowest world z (m) of the gripper and moving-jaw collision geoms at joint angles q [B, 6] -> [B]
        (per_body: [B, 2] in GUARD_BODIES order). Arm-table guard of the NRD: lowest_point < 0.5 mm (critic 3.7)."""
        q = q.to(self.dtype)
        out = []
        for s in range(0, max(1, q.shape[0]), chunk):
            pos, rot = self.body_poses(q[s: s + chunk])
            cols = []
            for b, pts, rad in self._guard_points():
                u = rot[:, b, 2, :]                                                   # world z row of the body rotation
                cols.append(pos[:, b, 2] + ((u @ pts.T) - rad).amin(-1))
            out.append(torch.stack(cols, -1))
        low = torch.cat(out)
        return low if per_body else low.amin(-1)

    # -- inverse kinematics --
    def ik(self, q0, pos, rot=None, w_rot=0.1, iters=150, damping=1e-4):
        """Scene.ik, batched: damped least squares on the 5 arm joints for the grasp point at ``pos`` [B, 3] and the
        gripper rotation ``rot`` [B, 3, 3] (None: position only), rotation weight w_rot (m/rad), joints clipped to
        TLO..THI. Per world the same stopping rules (error below 1e-6 m and 1e-5 rad; every 10 iterations, stop when
        10 iterations gained less than 10 %). q0 [B, >=5] -> (q [B, 5], position error [B] m, angle error [B] rad)."""
        q = q0[..., :5].to(self.dtype).clone()
        b = q.shape[0]
        lo, hi = self.tlo[:5], self.thi[:5]
        active = torch.ones(b, dtype=torch.bool, device=q.device)
        last = torch.full((b,), math.inf, dtype=self.dtype, device=q.device)
        eye = torch.eye(6, dtype=self.dtype, device=q.device)
        for it in range(iters):
            jp, jr, p, r = self.jacobian(q)
            ep = pos - p
            er = rotvec(rot @ r.transpose(-1, -2)) if rot is not None else torch.zeros_like(ep)
            nep, ner = ep.norm(dim=-1), er.norm(dim=-1)
            err = nep + w_rot * ner
            active = active & ~((nep < 1e-6) & (ner < 1e-5))
            if it % 10 == 0:
                active = active & ~(err > 0.9 * last)
                last = torch.where(active, err, last)
            if not bool(active.any()):
                break
            jac = torch.cat((jp, w_rot * jr), -2)                                       # [B, 6, 5]
            rhs = torch.cat((ep, w_rot * er), -1)[..., None]
            dq = (jac.transpose(-1, -2) @ torch.linalg.solve(jac @ jac.transpose(-1, -2) + damping * eye, rhs))[..., 0]
            q = torch.where(active[:, None], torch.minimum(torch.maximum(q + dq, lo), hi), q)
        p, r = self.fk(q)
        ea = rotvec(rot @ r.transpose(-1, -2)).norm(dim=-1) if rot is not None else torch.zeros_like(p[:, 0])
        return q, (pos - p).norm(dim=-1), ea

    def pose_ik(self, q0, pos, rot, max_tilt=None):
        """Scene.pose_ik, batched: within IK_TOL_M / IK_TOL_RAD, or else (max_tilt given, rad) the position within
        IK_TOL_M and the orientation within max_tilt (second IK from the first result, w_rot 0.001, 600 iterations).
        -> (q [B, 5], ok [B], position error, angle error)."""
        q, ep, ea = self.ik(q0, pos, rot)
        ok = (ep < IK_TOL_M) & (ea < IK_TOL_RAD)
        if max_tilt is not None and not bool(ok.all()):
            redo = ~ok
            q2, ep2, ea2 = self.ik(q[redo], pos[redo], rot[redo], w_rot=0.001, iters=600)
            ok2 = (ep2 < IK_TOL_M) & (ea2 < max_tilt)
            q, ep, ea, ok = q.clone(), ep.clone(), ea.clone(), ok.clone()
            q[redo], ep[redo], ea[redo], ok[redo] = q2, ep2, ea2, ok2
        return q, ok, ep, ea

    def seeds(self, pos, yaw):
        """Scene.solve's 3 starting guesses [3, B, 5]: [pan, lift in (0, -0.6, 0.6), 0.6, 1.1, wrap(yaw - pan)]."""
        pan = torch.atan2(pos[:, 1] - self.pan_axis[1], pos[:, 0] - self.pan_axis[0])
        roll = wrap(yaw - pan)
        return torch.stack([torch.stack((pan, torch.full_like(pan, lift), torch.full_like(pan, 0.6),
                                         torch.full_like(pan, 1.1), roll), -1) for lift in (0.0, -0.6, 0.6)])

    def ik_topdown(self, q0, pos, yaw, max_tilt=None, seeds=True):
        """Scene.solve for a top-down gripper at the yaw, batched: the hint q0 [B, >=5] (None: no hint) and then (seeds)
        the 3 standard guesses, all first within the tight tolerances, then (max_tilt) all with the tilt fallback; the
        first guess that succeeds wins, as in the collector. -> (q [B, 5], ok [B], position error [B] m, angle error
        [B] rad); worlds without a solution keep the last guess's result and ok False."""
        pos = pos.to(self.dtype)
        yaw = yaw.to(self.dtype)
        rot = rotz(yaw)
        guesses = ([] if q0 is None else [q0[..., :5].to(self.dtype)]) + (list(self.seeds(pos, yaw)) if seeds else [])
        b = pos.shape[0]
        out_q = torch.zeros(b, 5, dtype=self.dtype, device=pos.device)
        out_ep = torch.full((b,), math.inf, dtype=self.dtype, device=pos.device)
        out_ea = torch.full((b,), math.inf, dtype=self.dtype, device=pos.device)
        done = torch.zeros(b, dtype=torch.bool, device=pos.device)
        strict = []
        for g in guesses:
            q, ep, ea = self.ik(g, pos, rot)
            strict.append(q)
            ok = (ep < IK_TOL_M) & (ea < IK_TOL_RAD) & ~done
            out_q[ok], out_ep[ok], out_ea[ok] = q[ok], ep[ok], ea[ok]
            done |= ok
            out_q[~done], out_ep[~done], out_ea[~done] = q[~done], ep[~done], ea[~done]
        if max_tilt is not None:
            for qs in strict:
                todo = ~done
                if not bool(todo.any()):
                    break
                q, ep, ea = self.ik(qs[todo], pos[todo], rot[todo], w_rot=0.001, iters=600)
                ok = (ep < IK_TOL_M) & (ea < max_tilt)
                idx = torch.nonzero(todo).flatten()
                out_q[idx], out_ep[idx], out_ea[idx] = q, ep, ea
                done[idx[ok]] = True
        return out_q, done, out_ep, out_ea

    def ik_track(self, q0, pos, rot, iters=3, damping=1e-4, step_max=0.1):
        """Warm-started IK for a target that moves a little every step (the command decoder), with the collector's two
        weights (Scene.pose_ik): ``iters`` damped-least-squares iterations at w_rot 0.1 m/rad (top-down where it is in
        reach), then ``iters`` at w_rot 0.001 from that result (the position exact, the least orientation change: where
        top-down is out of reach, e.g. the wrist-flex limit above a grasp-point height of about 6-7 cm, the gripper
        tilts). Unlike Scene.ik, a joint at its limit (TLO..THI) that a step would push further leaves that step's
        Jacobian (no error build-up from clipping), and an iteration changes a joint by at most step_max (rad); fixed
        shapes, no early stop; orientation errors by rotvec_fast. -> (q [B, 5], position error [B] m, angle error [B])."""
        q = q0[..., :5].to(self.dtype)
        lo, hi = self.tlo[:5], self.thi[:5]
        eye = torch.eye(6, dtype=self.dtype, device=q.device)
        for w in (0.1,) * iters + (0.001,) * iters:
            jp, jr, p, r = self.jacobian(q)
            jac0 = torch.cat((jp, w * jr), -2)                                              # [B, 6, 5]
            rhs = torch.cat((pos - p, w * rotvec_fast(rot @ r.transpose(-1, -2))), -1)[..., None]
            free = torch.ones_like(q)
            for _ in range(2):   # a joint at its limit that the step would push further leaves the Jacobian
                jac = jac0 * free[:, None, :]
                jt = jac.transpose(-1, -2)
                dq = (jt @ torch.linalg.solve(jac @ jt + damping * eye, rhs))[..., 0]
                stuck = ((q >= hi - 1e-9) & (dq > 0)) | ((q <= lo + 1e-9) & (dq < 0))
                free = free * (~stuck).to(free)
            big = dq.abs().amax(-1, keepdim=True)
            dq = dq * torch.where(big > step_max, step_max / big.clamp_min(1e-12), torch.ones_like(big))
            q = torch.minimum(torch.maximum(q + dq, lo), hi)
        p, r = self.fk(q)
        return q, (pos - p).norm(dim=-1), rotvec_fast(rot @ r.transpose(-1, -2)).norm(dim=-1)
