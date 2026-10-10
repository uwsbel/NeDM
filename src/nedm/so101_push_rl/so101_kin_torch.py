"""Batched torch (float64) port of the SO-101 arm model of the push-T collector (spec M1, goal_ppo_spec.md).

Exact port of nedm.so101_push.robot.ArmModel: link_poses, tcp, tcp_jacobian, gripper_yaw, gravity_hold_torque and the
damped least squares IK, plus the collector command law (ScriptedPolicy.command / PushFamily.commands):

    q_cmd = clip(q_des + (g(q_des) + kd * qdot_des) / kp, raw joint limits),  qdot_des = (q_des(k+1) - q_des(k-1)) / 0.04

All constants are built by ArmModel in NumPy and copied (bit for bit) to torch buffers. The operation structure follows
the NumPy code (pose[child] = (pose[parent] @ F0[j]) @ Rz(q_j) [@ F1inv[j]], Jacobian columns cross(a, p - o), gravity
torque accumulated link by link in subtree order), so results agree with ArmModel to round-off (max abs differences
~1e-16; checks in tests/rl/check_so101_kin_torch.py). Inputs are batched: q5 [N, 5], p [N, 3], yaw [N]. Device-agnostic.

Batch invariance: every env gets bit-identical results whatever the batch size or its position in the batch (CPU):
the 4x4 / 6x5 products use torch.bmm (its small-matrix kernel is per batch element), cross products use
torch.linalg.cross, other sums are written out in a fixed order and atan2 goes through atan2_bi (see there). The
check (c) of tests/rl/check_so101_stroke_decoder.py (batch of 1,024 vs a per-env loop) tests this.

IK (ArmModel.ik): q_init is clipped to limits +- ik_margin; per iteration the weighted residual W e is computed and an
environment whose ||W e|| < tol is frozen (not updated again, exactly like the NumPy `break`); the others take
dq = A^T solve(A A^T + damping^2 I6, W e) with A = W J (torch.linalg.solve_ex, 6x6 LU) and are clipped again. Only the
still-active environments are evaluated in each iteration (compacted index set): a warm-started batch costs about two
Jacobians per env, but an env that does not converge (target out of reach / at a joint limit) runs all `iters`
iterations, and each iteration has a fixed cost of ~0.2-0.3 ms on CPU.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from nedm.so101_push.robot import ARM_JOINTS, ArmModel

_ATAN2_PAD = 64          # >= 2 x the widest CPU vector (AVX-512: 8 doubles)
_ATAN2_CHUNK = 16384     # below at::internal::GRAIN_SIZE: every chunk runs as one serial vectorized loop


def atan2_bi(y, x):
    """torch.atan2 with rounding that does not depend on the batch size or the position in the batch.

    On CPU, torch.atan2 uses the SLEEF vector atan2 for full vectors and std::atan2 for the tail elements of a loop;
    the two differ by 1 ulp for some inputs, so an env could get a different result in a batch of 1,024 than alone.
    Padding the inputs to a multiple of _ATAN2_PAD and running chunks of _ATAN2_CHUNK sends every element through
    the vector path. (Unary sin / cos / acos / sqrt use the same function for the tail and need no padding.)"""
    if y.device.type != "cpu":
        return torch.atan2(y, x)
    shape = torch.broadcast_shapes(y.shape, x.shape)
    n = math.prod(shape)
    m = -(-n // _ATAN2_PAD) * _ATAN2_PAD
    yx = torch.stack((y.expand(shape).reshape(n), x.expand(shape).reshape(n)))      # [2, n], contiguous rows
    if m != n:
        yx = torch.nn.functional.pad(yx, (0, m - n), value=1.0)
    if m > _ATAN2_CHUNK:
        out = torch.cat([torch.atan2(yx[0, i:i + _ATAN2_CHUNK], yx[1, i:i + _ATAN2_CHUNK]) for i in range(0, m, _ATAN2_CHUNK)])
    else:
        out = torch.atan2(yx[0], yx[1])
    return out[:n].reshape(shape)


class So101Kin(torch.nn.Module):
    """Batched kinematics, statics, IK and command law of the 5-DOF SO-101 (jaw locked)."""

    def __init__(self, cfg: dict, device="cpu", dtype=torch.float64):
        super().__init__()
        m = ArmModel(cfg["robot"], cfg["robot"]["description"])
        self.np_model = m
        self.dtype = dtype

        def to(a):
            return torch.as_tensor(np.array(a, dtype=np.float64), dtype=dtype, device=device)

        self.links = list(m.links)
        self.joints = list(m.joints)
        self.parent = list(m.parent)
        self.child = list(m.child)
        nj = len(self.joints)
        # serial chain base -> shoulder -> ... -> gripper -> jaw (the code below relies on it)
        assert self.parent[0] == "base" and all(self.parent[j] == self.child[j - 1] for j in range(1, nj))
        assert self.links == ["base"] + self.child and self.child[ARM_JOINTS - 1] == "gripper"
        assert all(m.subtree[j] == self.child[j:] for j in range(ARM_JOINTS))
        self.jaw_lock = float(m.jaw_lock)
        self.register_buffer("T_base", to(m.T_base))
        self.register_buffer("F0", torch.stack([to(f) for f in m.F0]))
        self.register_buffer("F1inv", torch.stack([to(f) for f in m.F1inv]))
        # F1inv is the identity for every SO-101 joint: multiplying by it is exact, so it is skipped (bit-identical)
        self.f1_identity = [bool(np.array_equal(f, np.eye(4))) for f in m.F1inv]
        # Rz(q) = E0 + cos(q) Ec + sin(q) Es (entries exact: c, -s, s, c, 1, 1, zeros)
        E0, Ec, Es = np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))
        E0[2, 2] = E0[3, 3] = 1.0
        Ec[0, 0] = Ec[1, 1] = 1.0
        Es[0, 1], Es[1, 0] = -1.0, 1.0
        self.register_buffer("_E0", to(E0))
        self.register_buffer("_Ec", to(Ec))
        self.register_buffer("_Es", to(Es))
        self.register_buffer("limits", to(m.limits))                                   # [5, 2] raw joint limits
        self.ik_margin = float(m.ik_margin)
        self.register_buffer("ik_lo", to(m.limits[:, 0] + m.ik_margin))                # computed in NumPy (parity)
        self.register_buffer("ik_hi", to(m.limits[:, 1] - m.ik_margin))
        self.register_buffer("tcp_local", to(m.tcp_local))
        self.register_buffer("_tcp4", to(np.r_[m.tcp_local, 1.0][:, None]))            # [4, 1]: R @ tcp + t
        self.register_buffer("_eye6", to(np.eye(6)))
        self.tcp_local_f = [float(v) for v in m.tcp_local]
        self.mass = {ln: float(m.mass[ln]) for ln in self.links}
        self.com = {ln: [float(v) for v in m.com[ln]] for ln in self.links}
        self.register_buffer("_com", to([m.com[ln] for ln in self.child]))               # [6, 3], chain children
        self.register_buffer("_mass", to([m.mass[ln] for ln in self.child]))             # [6]
        self.subtree = [list(s) for s in m.subtree]
        self.register_buffer("kp", to(cfg["controller"]["kp"]))
        self.register_buffer("kd", to(cfg["controller"]["kd"]))
        self.ctrl = float(cfg["simulation"]["control_step_s"])

    # ------------------------------------------------------------------ chain
    def _chain(self, q5, nj: int):
        """First nj joints of the serial chain. q5 [N, 5] (the jaw, joint 5, is at jaw_lock).
        Returns (joint frames poses[parent] @ F0[j] [N, nj, 4, 4], child link poses [N, nj, 4, 4])."""
        q5 = q5.to(self.dtype)
        N = q5.shape[0]
        q = q5[:, :min(nj, ARM_JOINTS)]
        if nj > ARM_JOINTS:
            q = torch.cat((q, torch.full((N, nj - ARM_JOINTS), self.jaw_lock, dtype=self.dtype, device=q5.device)), 1)
        c, s = torch.cos(q), torch.sin(q)
        Rz = self._E0 + c[..., None, None] * self._Ec + s[..., None, None] * self._Es                   # [N, nj, 4, 4]
        P = self.T_base.expand(N, 4, 4)
        frames, childs = [], []
        for j in range(nj):
            M = torch.bmm(P, self.F0[j].expand(N, 4, 4))
            P = torch.bmm(M, Rz[:, j])
            if not self.f1_identity[j]:
                P = torch.bmm(P, self.F1inv[j].expand(N, 4, 4))
            frames.append(M)
            childs.append(P)
        return torch.stack(frames, 1), torch.stack(childs, 1)

    @staticmethod
    def _xform_point(T, v):
        """T[..., :3, :3] @ v + T[..., :3, 3] for a constant 3-vector v (list of floats); NumPy order (R @ v) + t."""
        R = T[..., :3, :3]
        return (R[..., 0] * v[0] + R[..., 1] * v[1] + R[..., 2] * v[2]) + T[..., :3, 3]

    # ------------------------------------------------------------------ kinematics
    def link_poses(self, q5) -> dict:
        """{link: [N, 4, 4] world pose of the link frame} for all 7 links (jaw at jaw_lock)."""
        _, ch = self._chain(q5, len(self.joints))
        poses = {"base": self.T_base.expand(ch.shape[0], 4, 4).clone()}   # a copy: writes must not reach the buffer
        for j, ln in enumerate(self.child):
            poses[ln] = ch[:, j]
        return poses

    def joint_frames(self, q5) -> list:
        """World frames [N, 4, 4] of the five arm joints (parent side: z = joint axis, translation = axis origin)."""
        fr, _ = self._chain(q5, ARM_JOINTS)
        return [fr[:, j] for j in range(ARM_JOINTS)]

    def tcp(self, q5):
        """TCP position p [N, 3] and gripper link rotation R [N, 3, 3]."""
        _, ch = self._chain(q5, ARM_JOINTS)
        T = ch[:, ARM_JOINTS - 1]
        return self._tcp_point(T), T[:, :3, :3]

    def _tcp_point(self, T):
        """TCP position R_gripper @ tcp_local + t_gripper as one small bmm: sum_k T[i, k] [tcp, 1][k], k = 0..3."""
        return torch.bmm(T[:, :3, :], self._tcp4.expand(T.shape[0], 4, 1)).squeeze(-1)

    def tcp_jacobian(self, q5):
        """TCP position p [N, 3], gripper rotation R [N, 3, 3] and the geometric Jacobian J [N, 6, 5] ([v; w])."""
        fr, ch = self._chain(q5, ARM_JOINTS)
        T = ch[:, ARM_JOINTS - 1]
        p = self._tcp_point(T)
        a, o = fr[:, :, :3, 2], fr[:, :, :3, 3]                                    # [N, 5, 3]
        J = torch.cat((torch.linalg.cross(a, p[:, None, :] - o, dim=-1), a), dim=2).transpose(1, 2)
        return p, T[:, :3, :3], J

    @staticmethod
    def gripper_yaw(R):
        """Heading of the gripper x axis in the table plane: atan2(R[:, 1, 0], R[:, 0, 0]) [N]."""
        return atan2_bi(R[:, 1, 0], R[:, 0, 0])

    # ------------------------------------------------------------------ statics
    def gravity_hold_torque(self, q5, g: float = 9.81):
        """Joint torques [N, 5] that hold the arm against gravity (motor convention, + about the joint z).
        tau_j = - sum_{links l in subtree(j)} a_j . ((c_l - o_j) x m_l [0, 0, -g]), accumulated in subtree order."""
        fr, ch = self._chain(q5, len(self.joints))
        c = self._xform_point(ch, self._com[:, :, None].unbind(1))                      # [N, 6, 3] world COMs
        a, o = fr[:, :ARM_JOINTS, :3, 2], fr[:, :ARM_JOINTS, :3, 3]                     # [N, 5, 3]
        fz = self._mass * (-g)                                                          # (m [0, 0, -g])[2]
        tau = torch.zeros(q5.shape[0], ARM_JOINTS, dtype=self.dtype, device=q5.device)
        nl = c.shape[1]
        for li in range(nl):              # link li is in the subtree of joints 0..li (subtree(j) = children j..)
            nj = min(li + 1, ARM_JOINTS)
            r = c[:, li:li + 1, :] - o[:, :nj]                                          # [N, nj, 3]
            # np.cross(r, [0, 0, fz]) = [r1 fz - r2 0, r2 0 - r0 fz, r0 0 - r1 0] = [r1 fz, -(r0 fz), 0] exactly
            x0 = r[..., 1] * fz[li]
            x1 = -(r[..., 0] * fz[li])
            tau[:, :nj] = tau[:, :nj] - (a[:, :nj, 0] * x0 + a[:, :nj, 1] * x1)
        return tau

    # ------------------------------------------------------------------ inverse kinematics
    def ik(self, p_des, yaw_des, q_init, iters: int = 60, tol: float = 1e-6, damping: float = 1e-3,
           w_pos: float = 1.0, w_tilt: float = 0.05, w_yaw: float = 0.02):
        """Damped least squares IK (ArmModel.ik): TCP position, gripper z = world z, gripper yaw.

        p_des [N, 3], yaw_des [N], q_init [N, 5]. Returns (q [N, 5], info) with info = pos_err_m [N], tilt_rad [N],
        yaw_err_rad [N] (as ArmModel.ik) plus at_limit [N] bool and iters [N] (number of updates taken)."""
        dt, dev = self.dtype, self.T_base.device
        p_des = torch.as_tensor(p_des, dtype=dt, device=dev)
        yaw_des = torch.as_tensor(yaw_des, dtype=dt, device=dev)
        q_init = torch.as_tensor(q_init, dtype=dt, device=dev)
        lo, hi = self.ik_lo, self.ik_hi
        q = torch.clamp(q_init[:, :ARM_JOINTS], lo, hi).clone()
        N = q.shape[0]
        w = torch.tensor([w_pos] * 3 + [w_tilt] * 2 + [w_yaw], dtype=dt, device=dev)
        reg_eye = (damping ** 2) * self._eye6
        n_upd = torch.zeros(N, dtype=torch.long, device=dev)
        # active set (compacted): idx, qa, pa, ya, na; frozen envs are written back to q / n_upd when they stop
        idx, qa, pa, ya = torch.arange(N, device=dev), q.clone(), p_des, yaw_des
        na = torch.zeros(N, dtype=torch.long, device=dev)
        for _ in range(iters):
            p, R, J = self.tcp_jacobian(qa)
            yaw = atan2_bi(R[:, 1, 0], R[:, 0, 0])
            dd = ya - yaw
            dyaw = atan2_bi(torch.sin(dd), torch.cos(dd))
            # tilt = z x (0, 0, 1) = (z_y, -z_x, 0) exactly; residual e = [p_des - p, tilt_xy, dyaw]
            e = torch.cat((pa - p, R[:, 1:2, 2], -R[:, 0:1, 2], dyaw[:, None]), dim=1)
            We = w * e                                   # = W @ e (W diagonal; the zero products add exactly)
            sq = We * We
            err = torch.sqrt(((((sq[:, 0] + sq[:, 1]) + sq[:, 2]) + sq[:, 3]) + sq[:, 4]) + sq[:, 5])   # sqrt(dot)
            conv = err < tol
            if bool(conv.any()):                         # freeze the converged envs (the NumPy `break`)
                done = conv.nonzero().squeeze(1)
                q[idx[done]] = qa[done]
                n_upd[idx[done]] = na[done]
                keep = (~conv).nonzero().squeeze(1)
                if keep.numel() == 0:
                    idx = keep
                    break
                idx, qa, pa, ya, na, J, We = idx[keep], qa[keep], pa[keep], ya[keep], na[keep], J[keep], We[keep]
            A = w[:, None] * J                           # = W @ J
            M = torch.bmm(A, A.transpose(1, 2)) + reg_eye
            sol = torch.linalg.solve_ex(M, We.unsqueeze(-1))[0]    # = torch.linalg.solve (LU, LAPACK gesv) without
            #                                                        the singularity check (M is SPD: no sync needed)
            dq = torch.bmm(A.transpose(1, 2), sol).squeeze(-1)
            qa = torch.clamp(qa + dq, lo, hi)
            na = na + 1
        if idx.numel():                                  # envs still active after `iters` updates
            q[idx] = qa
            n_upd[idx] = na
        p, R = self.tcp(q)
        yaw = atan2_bi(R[:, 1, 0], R[:, 0, 0])
        dd = yaw_des - yaw
        dp = p_des - p
        info = {"pos_err_m": torch.sqrt((dp[:, 0] * dp[:, 0] + dp[:, 1] * dp[:, 1]) + dp[:, 2] * dp[:, 2]),
                "tilt_rad": torch.acos(torch.clamp(R[:, 2, 2], -1.0, 1.0)),
                "yaw_err_rad": torch.abs(atan2_bi(torch.sin(dd), torch.cos(dd))),
                "at_limit": ((q <= lo + 1e-9) | (q >= hi - 1e-9)).any(1),
                "iters": n_upd}
        return q, info

    # ------------------------------------------------------------------ command law
    def command_raw(self, q_des_prev, q_des, q_des_next, first=False):
        """Unclipped q_des + (g(q_des) + kd qdot) / kp. `first` (bool or [N] bool): ScriptedPolicy k = 0 rule
        qdot = (q_des_next - q_des) / ctrl * 0.5; else qdot = (q_des_next - q_des_prev) / (2 ctrl)."""
        dt = self.dtype                  # inputs in another dtype (e.g. float32) are converted first, like q5 in _chain
        q_des_prev, q_des, q_des_next = q_des_prev.to(dt), q_des.to(dt), q_des_next.to(dt)
        qdot = (q_des_next - q_des_prev) / (2 * self.ctrl)
        if isinstance(first, torch.Tensor):
            if bool(first.any()):
                qf = (q_des_next - q_des) / self.ctrl * 0.5
                qdot = torch.where(first.reshape(-1, 1), qf, qdot)
        elif first:
            qdot = (q_des_next - q_des) / self.ctrl * 0.5
        return q_des + (self.gravity_hold_torque(q_des) + self.kd * qdot) / self.kp

    def command(self, q_des_prev, q_des, q_des_next, first=False):
        """q_cmd [N, 5] = clip(q_des + (g(q_des) + kd qdot) / kp, raw joint limits) (PushFamily.commands)."""
        return torch.clamp(self.command_raw(q_des_prev, q_des, q_des_next, first), self.limits[:, 0], self.limits[:, 1])
