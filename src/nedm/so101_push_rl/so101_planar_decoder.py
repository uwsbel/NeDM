"""Batched planar pusher decoder for the SO-101 push-T rotation PPO (spec SPEC_lead_v1.md A1, A2).

One policy decision lasts T_d = decision_steps x ctrl (5 x 20 ms = 0.1 s). A decision gives a planar target
displacement d (world frame, |d| <= d_max_m = 30 mm) and a gripper yaw change dyaw (|dyaw| <= dyaw_max_rad). The
target moves in the table plane at the fixed push height z of the episode (the start context's p_line z). The fingers
point straight down (the IK tilt rows target 0). There is no lift, no descent and no tilt.

Target profile of one decision (closed form; tau = time since the decision):
  v_new = d / T_d (|v_new| <= v_max),  w_new = dyaw / T_d
  tau <= T_d:  v(tau) = v_k + (v_new - v_k)(1 - cos(pi tau / T_d)) / 2
               P(tau) = v_k tau + (v_new - v_k)(tau / 2 - T_d / (2 pi) sin(pi tau / T_d))
  tau >  T_d:  v(tau) = v_new,  P(tau) = (v_k + v_new) T_d / 2 + v_new (tau - T_d)
v_k is the target velocity of the previous profile at the decision, P(tau) the displacement since the decision. The yaw
uses the same law with the yaw rate (w_k -> w_new). After T_d the target keeps v_new until the next decision, so a
sequence of equal decisions gives a constant speed. From rest, a decision is the collector's cosine ramp-up of length
T_d (StrokeDecoder.plan_cruise with ramp T_d gives the same target).
  yaw_control false: w_new = 0 (the yaw stays at the start yaw).
  yaw_block True (the env passes: dense fingertip-T signed distance < yaw_free_gap_m): the yaw rate is 0 for the whole
  decision (w_k and w_new set to 0, no blend), so the gripper never turns near the T; flag yaw_blocked (counted only
  with yaw_control). A blend to 0 would still turn up to w_k T_d / 2 = 0.15 rad (8.6 deg) near the T, and a 15 deg
  turn in contact makes both frozen NRDs kick the T by 60-90 mm (gap probe, family D2).

Per control step k -> k + 1 (all environments):
  1. nominal increment dp = P(tau_{k+1}) - P(tau_k), dy = Y(tau_{k+1}) - Y(tau_k); raw target p(k) + dp, yaw(k) + dy.
     The target follows increments: a clipped, scaled or held step is not caught up later.
  2. workspace clip, polar about the pan axis a (joint 0 origin, So101Kin.joint_frames): r = |p - a| clipped to
     [min(ws_r_min, r_k), max(ws_r_max, r_k)], azimuth atan2(p - a) clipped to [min(-az, az_k), max(az, az_k)] with
     az = ws_az_deg and (r_k, az_k) of the current target; p = a + r (cos, sin)(azimuth). A start outside the sector
     never jumps; it only cannot move farther out. On the inner arc the radial projection can lengthen a step by the
     factor r_min / (r_min - |dp|) (<= 1.1 at 6 mm per step). Flag ws_clip.
  3. joint-speed guard: q_try = IK(target) (So101Kin.ik, q_init = q_des(k), ik_iters). If m = max_j |q_try_j - q_des_j(k)|
     / dq_max_j > 1 (dq_max_rad: one value for all joints or a list of 5), the planar displacement and the yaw change of
     this step are scaled by 1 / m and the IK is solved again from q_des(k) (only for these environments; the scaled
     target is clipped to the sector again, since a chord of the inner arc lies up to 0.1 mm inside it). The re-solve
     can still exceed dq_max_j slightly (nonlinear IK; <= 3 % in the checks). Flag dq_scaled.
  4. IK position error > ik_err_flag_m or a joint at its IK limit (limit +- ik_margin): hold the previous target,
     q_des(k+1) = q_des(k). Flags ik_err (both causes) and ik_at_limit (the limit part).
  5. q_cmd(k) = q_des(k) + (g(q_des(k)) + kd qdot_des) / kp with qdot_des = (q_des(k+1) - q_des(k-1)) / (2 ctrl)
     (So101Kin.command_raw, the collector law; k = 0: the ScriptedPolicy first-step rule), clipped to the raw joint
     limits (flag cmd_clipped), then to the lead band q_now + [lead_lo, lead_hi] (q_now = the current joint angles;
     the order of so101_direct_common.clip_cmd; flag lead_clipped). The band is off when lead_lo is None.

Batch invariance: every operation is elementwise, a fixed-order sum, a max over 5 joints, atan2_bi, or a So101Kin call
(bmm / solve_ex per batch element). The guard re-solve runs on a subset; the IK result of an environment does not depend
on the batch. tests/rl/check_so101_planar_decoder.py (b) compares a batch of 1,024 with a per-env loop (CPU).
Device: kin must live on the decoder device. All state is float64 (kin.dtype).
"""
from __future__ import annotations

import math

import torch

from nedm.so101_push_rl.so101_kin_torch import So101Kin, atan2_bi

# Defaults = configs/rl/so101_planar/env_v1.json (action + decoder blocks). The workspace sector is wider than the
# 0.11-0.25 m / 55 deg of the spec text: measured about the pan axis, the RL start contexts lie at 0.070-0.264 m and
# -60.6..+34.8 deg (416 of 2,048 train starts outside the spec sector) and the recorded push-height TCP positions at
# 0.0706-0.2640 m and -64.2..+57.6 deg (rl_train bank, all frames).
DEFAULT_CFG = dict(d_max_m=0.030, dyaw_max_rad=0.3, yaw_control=True, frame="t", decision_steps=5, ctrl=0.02, v_max=0.30,
                   ws_r_min=0.070, ws_r_max=0.265, ws_az_deg=65.0, dq_max_rad=0.09, ik_iters=60, ik_err_flag_m=0.002,
                   yaw_free_gap_m=0.010, lead_lo=None, lead_hi=None)
DECIDE_FLAGS = ("yaw_blocked",)
STEP_FLAGS = ("ws_clip", "dq_scaled", "ik_err", "ik_at_limit", "cmd_clipped", "lead_clipped")


def blend(v0, vn, tau, Td):
    """Cosine velocity blend from v0 to vn over Td (closed form). v0, vn [N] or [N, 2]; tau [N] (s since the decision).
    Returns (P(tau) displacement since the decision, v(tau)); after Td: v = vn exactly, P grows linearly."""
    if v0.dim() == 2:
        tau = tau[:, None]
    x = torch.clamp(tau, max=Td)
    P = v0 * x + (vn - v0) * (x / 2 - Td / (2 * math.pi) * torch.sin(math.pi * x / Td)) + vn * (tau - x)
    v = torch.where(tau >= Td, vn, v0 + (vn - v0) * (1 - torch.cos(math.pi * x / Td)) / 2)
    return P, v


def norm2(v):
    """|v| of [N, 2] vectors, fixed operation order (batch-invariant)."""
    return torch.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1])


def pan_axis_xy(kin: So101Kin):
    """xy of the shoulder pan axis (joint 0 origin; the base is fixed, so any q gives the same point) [2]."""
    return kin.joint_frames(torch.zeros(1, 5, dtype=kin.dtype, device=kin.T_base.device))[0][0, :2, 3]


class PlanarDecoder:
    def __init__(self, kin: So101Kin, n: int, device=None, cfg: dict | None = None):
        c = {**DEFAULT_CFG, **(cfg or {})}
        self.cfg = c
        self.kin = kin
        self.n = int(n)
        self.device = torch.device(device) if device is not None else kin.T_base.device
        self.dtype = kin.dtype
        self.ctrl = float(c["ctrl"])
        self.n_dec = int(c["decision_steps"])
        self.Td = self.n_dec * self.ctrl
        self.d_max = float(c["d_max_m"])
        self.v_max = float(c["v_max"])
        self.dyaw_max = float(c["dyaw_max_rad"])
        self.yaw_control = bool(c["yaw_control"])
        self.r_min, self.r_max = float(c["ws_r_min"]), float(c["ws_r_max"])
        self.az_max = math.radians(float(c["ws_az_deg"]))
        dqm = c["dq_max_rad"]
        self.dq_max = torch.as_tensor(dqm if isinstance(dqm, (list, tuple)) else [float(dqm)] * 5, dtype=kin.dtype,
                                      device=self.device)
        self.ik_iters = int(c["ik_iters"])
        self.ik_err_flag_m = float(c["ik_err_flag_m"])
        self.yaw_free_gap_m = float(c["yaw_free_gap_m"])
        dt, dev = self.dtype, self.device
        self.lead = None
        if c.get("lead_lo") is not None:
            self.lead = (torch.tensor(c["lead_lo"], dtype=dt, device=dev), torch.tensor(c["lead_hi"], dtype=dt, device=dev))
        self.axis = pan_axis_xy(kin).to(dev)
        N = self.n
        z = lambda *s: torch.zeros(*s, dtype=dt, device=dev)
        self.p, self.z, self.yaw = z(N, 2), z(N), z(N)                 # target at the current step k
        self.v0, self.vn, self.w0, self.wn = z(N, 2), z(N, 2), z(N), z(N)   # blend of the active decision
        self.q_des_prev, self.q_des_cur = z(N, 5), z(N, 5)
        self.k = torch.zeros(N, dtype=torch.long, device=dev)
        self.k_dec = torch.zeros(N, dtype=torch.long, device=dev)       # step of the active decision
        self.counts = {k: torch.zeros(N, dtype=torch.long, device=dev) for k in DECIDE_FLAGS + STEP_FLAGS}

    # ------------------------------------------------------------------ helpers
    def _ids(self, ids):
        if ids is None:
            return torch.arange(self.n, device=self.device)
        ids = torch.as_tensor(ids, device=self.device)
        if ids.dtype == torch.bool:
            return ids.nonzero().flatten()
        return ids.long().flatten()

    def _vals(self, x, ids, tail=(), dtype=None):
        """Per-env values for `ids`: [len(ids), *tail], a full [N, *tail] array (gathered) or a scalar / [*tail]."""
        x = torch.as_tensor(x, dtype=dtype or self.dtype, device=self.device)
        m = ids.numel()
        if x.dim() == len(tail):
            return x.expand(m, *tail).clone()
        if x.shape[0] == m:
            return x.clone()
        if x.shape[0] == self.n:
            return x[ids].clone()
        raise ValueError(f"expected {m} or {self.n} rows, got shape {tuple(x.shape)}")

    def _tau(self, k, k_dec):
        return (k - k_dec).to(self.dtype) * self.ctrl

    def _ws_clip(self, p_raw, p_cur):
        """Workspace sector clip (polar about the pan axis), relaxed to the current target. -> (p [N, 2], clip [N])."""
        rel, relk = p_raw - self.axis, p_cur - self.axis
        r, rk = norm2(rel), norm2(relk)
        az = atan2_bi(torch.stack((rel[:, 1], relk[:, 1])), torch.stack((rel[:, 0], relk[:, 0])))
        az, azk = az[0], az[1]
        rc = torch.minimum(torch.maximum(r, torch.clamp(rk, max=self.r_min)), torch.clamp(rk, min=self.r_max))
        azc = torch.minimum(torch.maximum(az, torch.clamp(azk, max=-self.az_max)), torch.clamp(azk, min=self.az_max))
        clip = (rc != r) | (azc != az)
        pc = self.axis + rc[:, None] * torch.stack((torch.cos(azc), torch.sin(azc)), -1)
        return torch.where(clip[:, None], pc, p_raw), clip

    # ------------------------------------------------------------------ API
    def reset(self, ids, p0, yaw0, q_des_prev, q_des_cur, k0):
        """Target at rest at p0 [m, 3] (x, y, push height z) with gripper yaw yaw0 [m] at control step k0 [m] (long);
        q_des_prev / q_des_cur [m, 5]: the desired joint angles of steps k0 - 1 and k0. Counts of ids are zeroed."""
        ids = self._ids(ids)
        p0 = self._vals(p0, ids, (3,))
        k0 = self._vals(k0, ids, (), dtype=torch.long)
        self.p[ids] = p0[:, :2]
        self.z[ids] = p0[:, 2]
        self.yaw[ids] = self._vals(yaw0, ids, ())
        for name in ("v0", "vn", "w0", "wn"):
            getattr(self, name)[ids] = 0.0
        self.q_des_prev[ids] = self._vals(q_des_prev, ids, (5,))
        self.q_des_cur[ids] = self._vals(q_des_cur, ids, (5,))
        self.k[ids] = k0
        self.k_dec[ids] = k0
        for c in self.counts.values():
            c[ids] = 0

    def decide(self, ids, d_world, dyaw, yaw_block=None):
        """Start the blend of a new decision for `ids`: d_world [m, 2] (m, world frame; |d| clipped to d_max_m),
        dyaw [m] (rad; clipped to +-dyaw_max_rad), yaw_block [m] bool or None. Returns dict(v_new, w_new, yaw_blocked)."""
        ids = self._ids(ids)
        m = ids.numel()
        d = self._vals(d_world, ids, (2,))
        dy = self._vals(dyaw, ids, ())
        nd = norm2(d)
        d = torch.where((nd > self.d_max)[:, None], d * (self.d_max / torch.where(nd > 0, nd, torch.ones_like(nd)))[:, None], d)
        vn = d / self.Td
        nv = norm2(vn)
        vn = torch.where((nv > self.v_max)[:, None], vn * (self.v_max / torch.where(nv > 0, nv, torch.ones_like(nv)))[:, None], vn)
        wn = torch.clamp(dy, -self.dyaw_max, self.dyaw_max) / self.Td
        tau = self._tau(self.k[ids], self.k_dec[ids])
        _, vk = blend(self.v0[ids], self.vn[ids], tau, self.Td)
        _, wk = blend(self.w0[ids], self.wn[ids], tau, self.Td)
        zero = torch.zeros_like(wn)
        if not self.yaw_control:
            wn, wk = zero, zero
        blocked = torch.zeros(m, dtype=torch.bool, device=self.device)
        if yaw_block is not None and self.yaw_control:
            blocked = self._vals(yaw_block, ids, (), dtype=torch.bool)
            wn = torch.where(blocked, zero, wn)
            wk = torch.where(blocked, zero, wk)
        self.counts["yaw_blocked"][ids] += blocked.long()
        self.v0[ids], self.vn[ids], self.w0[ids], self.wn[ids] = vk, vn, wk, wn
        self.k_dec[ids] = self.k[ids]
        return dict(v_new=vn, w_new=wn, yaw_blocked=blocked)

    def step(self, q_now):
        """q_cmd(k) [N, 5] (float64) for the current step k of every env; advances k. q_now [N, 5]: current joint angles
        (lead band). info: flags (bool [N]) ws_clip, dq_scaled, ik_err, ik_at_limit, cmd_clipped, lead_clipped;
        ik_pos_err_m (final IK solve), dq_step (max |q_des(k+1) - q_des(k)|), lead (q_cmd - q_now), target_xy [N, 2],
        target_yaw (the target of step k + 1 after clip / guard / hold), target_v [N, 2], yaw_rate (nominal profile at
        k + 1), k (before)."""
        kin, dt = self.kin, self.dtype
        q_now = torch.as_tensor(q_now, device=self.device).to(dt)
        k = self.k
        tau0, tau1 = self._tau(k, self.k_dec), self._tau(k + 1, self.k_dec)
        P0, _ = blend(self.v0, self.vn, tau0, self.Td)
        P1, v1 = blend(self.v0, self.vn, tau1, self.Td)
        Y0, _ = blend(self.w0, self.wn, tau0, self.Td)
        Y1, w1 = blend(self.w0, self.wn, tau1, self.Td)
        dyaw = Y1 - Y0
        p_t, ws_clip = self._ws_clip(self.p + (P1 - P0), self.p)
        yaw_t = self.yaw + dyaw
        q_cur = self.q_des_cur
        q_try, ik = kin.ik(torch.cat((p_t, self.z[:, None]), 1), yaw_t, q_cur, iters=self.ik_iters)
        pos_err, at_lim = ik["pos_err_m"], ik["at_limit"]
        ratio = ((q_try - q_cur).abs() / self.dq_max).max(1).values
        guard = ratio > 1.0
        if bool(guard.any()):
            gi = guard.nonzero().flatten()
            s = 1.0 / ratio[gi]
            p_g, c_g = self._ws_clip(self.p[gi] + s[:, None] * (p_t[gi] - self.p[gi]), self.p[gi])
            y_g = self.yaw[gi] + s * dyaw[gi]
            ws_clip = ws_clip.clone()
            ws_clip[gi] = ws_clip[gi] | c_g
            q_g, ik_g = kin.ik(torch.cat((p_g, self.z[gi, None]), 1), y_g, q_cur[gi], iters=self.ik_iters)
            p_t, yaw_t, q_try = p_t.clone(), yaw_t.clone(), q_try.clone()
            pos_err, at_lim = pos_err.clone(), at_lim.clone()
            p_t[gi], yaw_t[gi], q_try[gi] = p_g, y_g, q_g
            pos_err[gi], at_lim[gi] = ik_g["pos_err_m"], ik_g["at_limit"]
        ik_err = (pos_err > self.ik_err_flag_m) | at_lim
        q_next = torch.where(ik_err[:, None], q_cur, q_try)
        p_next = torch.where(ik_err[:, None], self.p, p_t)
        yaw_next = torch.where(ik_err, self.yaw, yaw_t)
        raw = kin.command_raw(self.q_des_prev, q_cur, q_next, first=(k == 0))
        lo, hi = kin.limits[:, 0], kin.limits[:, 1]
        cmd_clipped = ((raw < lo) | (raw > hi)).any(1)
        q_cmd = torch.minimum(torch.maximum(raw, lo), hi)
        lead_clipped = torch.zeros_like(cmd_clipped)
        if self.lead is not None:
            ql = torch.minimum(torch.maximum(q_cmd, q_now + self.lead[0]), q_now + self.lead[1])
            lead_clipped = (ql != q_cmd).any(1)
            q_cmd = ql
        flags = dict(ws_clip=ws_clip, dq_scaled=guard, ik_err=ik_err, ik_at_limit=at_lim, cmd_clipped=cmd_clipped,
                     lead_clipped=lead_clipped)
        for f, v in flags.items():
            self.counts[f] += v.long()
        info = dict(flags, ik_pos_err_m=pos_err, dq_step=(q_next - q_cur).abs().max(1).values, lead=q_cmd - q_now,
                    target_xy=p_next, target_v=v1, target_yaw=yaw_next, yaw_rate=w1, k=k.clone())
        self.q_des_prev = q_cur
        self.q_des_cur = q_next
        self.p, self.yaw = p_next, yaw_next
        self.k = k + 1
        return q_cmd, info

    def features(self):
        """Causal decoder state at the current step k: target_xy [N, 2], target_v [N, 2] (m/s, nominal profile),
        target_yaw [N], yaw_rate [N] (rad/s), z [N] (push height of the TCP), k [N]."""
        tau = self._tau(self.k, self.k_dec)
        _, v = blend(self.v0, self.vn, tau, self.Td)
        _, w = blend(self.w0, self.wn, tau, self.Td)
        return dict(target_xy=self.p.clone(), target_v=v, target_yaw=self.yaw.clone(), yaw_rate=w, z=self.z.clone(),
                    k=self.k.clone())
