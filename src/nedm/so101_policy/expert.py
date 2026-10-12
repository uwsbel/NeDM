"""Privileged closed-loop pick-and-place expert: a batched finite-state machine that acts through ``DecoderV2``.

Inputs once per decision (0.1 s): the true state (``WorldState``: joint angles, the true cube pose), the task goal
(cube centre x, y on the table, m; cube yaw, rad, mod 90 deg) and the decoder (its mode and Cartesian target). Output:
the decoder action a [B, 6] (``decoder`` module docstring). The expert re-plans from the CURRENT cube pose at every
decision, so the same expert serves both worlds, disturbed tasks and DAgger labels (``act`` on any state; ``reset``
infers the phase from the state and the decoder mode).

Phases (``PHASES``):

| Phase | Decoder mode | What the expert does | Ends when |
|---|---|---|---|
| approach | free | TCP to ``pre_height`` above the grasp height over the cube centre, yaw to the grasp yaw (``clear`` higher while the yaw is still off) | at the waypoint, slow, aligned, cube at rest -> GRASP trigger -> grasp |
| grasp | GRASP macro | holds the trigger; aborts (approach) if the cube leaves the descent axis by > ``abort_off`` before the close | decoder in hold -> carry if the cube is held, else recover |
| carry | hold | TCP so that the cube centre (its grasp-frame offset kept, plus ``release_shift``) is over the goal, gripper turned by the cube's yaw error (mod 90 deg) | at the waypoint, slow, aligned -> PLACE trigger -> place; cube lost -> recover |
| place | PLACE macro | nothing (macro) | decoder back in free -> done |
| recover | PLACE macro | a PLACE trigger re-opens the jaw and retreats (no cube: no lowering) | decoder in free -> approach (at most ``max_recover`` times, then done) |
| done | free | zero action | |

Grasp yaw: a cube face normal (cube yaw + k 90 deg), the one with the smallest worst wrist roll (|yaw - pan angle|)
over the pick and the place, i.e. the collector's ``Scene.grasp_yaw`` choice nearest the radial direction when the place
is near the pick; then tracked as the cube turns (nearest face normal to the last choice).

Motion per decision (free motion): the TCP velocity at the end of the decision that lands the target on the waypoint
when the next decision stops it, v = (w - p - v0 T_d / 2) / T_d (decoder blend "filter": the set point whose filter rest
point is the waypoint), its norm capped at ``v_cruise`` and at the speed from which ``a_plan`` can still stop before
the waypoint; the yaw likewise (``w_cruise``, ``alpha_plan``). The plan's accelerations stay below the decoder's
bounds, so the decoder's own caps rarely bind. ``carry_profile`` "minjerk" (ablation) carries along one minimum-jerk
segment instead. Decoder blend "hann" (option b1): the same landing rule with the decoder's ``rest_terms`` (its rest
point is linear in the set point). Decoder CARRY macro (option b2): in the carry the expert gives the carry waypoint as
the CARRY target (a straight line that would pass within ``carry_r_min`` of the pan axis goes to a polar midpoint
first), aborts a running line when the goal moved (``carry_retarget``) and triggers PLACE at rest within
``carry_place_tol`` / ``carry_place_yaw``, or after ``carry_fix_max`` correction lines. Triggers: GRASP when the target
is within ``pos_tol`` / ``z_tol`` / ``yaw_tol`` of the approach
waypoint, slower than ``still_v``, and the cube at rest; PLACE when the predicted cube xy (offset in the grasp frame +
``release_shift``) is within ``place_tol`` of the goal and the yaw within ``place_yaw_tol``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from nedm.so101_policy.decoder import CUBE_HALF, CUBE_REST_Z, GRASP_PRE_CLOSE, M, DecoderV2
from nedm.so101_policy.kin import PAN_AXIS_XY, quat_mat, wrap, yaw_of
from nedm.so101_policy.task import cube_yaw_tilt as _task_yaw_tilt

PHASES = ("approach", "grasp", "carry", "place", "recover", "done")
PH = {n: k for k, n in enumerate(PHASES)}
QUARTER = math.pi / 2


def wrap90(a):
    """Angle -> (-45, 45] deg equivalent under the cube's 90 deg symmetry."""
    return -(torch.remainder(-a + QUARTER / 2, QUARTER) - QUARTER / 2)


def cube_yaw_tilt(quat):
    """Cube orientation (w x y z) [B, 4] -> (yaw mod 90 deg in [0, pi/2), tilt rad): ``task.cube_yaw_tilt`` (the yaw of
    a side axis after the smallest rotation that makes the cube flat: the same for all 24 relabellings of the cube,
    also when tilted; review F7: the expert's own projection changed by up to 3.5 deg at 20 deg tilt)."""
    return _task_yaw_tilt(quat)


def cube_bottom(pos, quat):
    """Lowest cube corner height (m) [B]."""
    return pos[:, 2] - CUBE_HALF * quat_mat(quat)[:, 2, :].abs().sum(-1)


@dataclass
class ExpertConfig:
    pre_height: float = 0.05       # m, approach waypoint above the GRASP height (collector 4-8 cm)
    clear: float = 0.03            # m, extra height while the yaw is more than yaw_ok off
    lift: float = 0.05             # m, GRASP lift height (collector 3-12 cm)
    release_shift: float = 0.0025  # m, the cube moves this far along the closing axis when the jaw opens (MuJoCo,
                                   # 64 expert places: +2.52 mm, std 0.35; NRD s61 / s62: 2.34 / 2.22 mm)
    v_cruise: float = 0.30         # m/s
    a_plan: float = 0.6            # m/s^2, braking of the plan (decoder free-motion a_max 1.5)
    w_cruise: float = 1.5          # rad/s
    alpha_plan: float = 3.0        # rad/s^2
    pos_tol: float = 0.0015        # m, target xy on the waypoint for the GRASP trigger
    place_tol: float = 0.0015      # m, predicted cube xy on the goal for the PLACE trigger
    place_yaw_tol: float = math.radians(1.0)
    z_tol: float = 0.004           # m
    yaw_tol: float = math.radians(1.0)
    yaw_ok: float = math.radians(3.0)
    still_v: float = 0.01          # m/s, target speed for a trigger
    cube_still: float = 0.01       # m/s, cube speed for a GRASP trigger
    abort_off: float = 0.004       # m, cube off the descent axis before the close -> abort
    climb_r: float = 0.06          # m, below the approach height within this xy distance of the cube: climb first
    held_box: tuple = (0.015, 0.015, 0.02)   # m, |cube centre in the grasp frame| (x closing axis, y, z) when held
    held_lift: float = 0.01        # m, lowest cube corner above the table when held after the lift
    lost_decisions: int = 2
    max_recover: int = 3
    carry_profile: str = "plan"    # "plan": waypoint plan every decision; "minjerk": one minimum-jerk segment (the
    carry_v_peak: float = 0.30     #   collector's move: peak speed, yaw rate and shortest duration), then "plan"
    carry_w_peak: float = 1.5
    carry_t_min: float = 0.5
    carry_retarget: float = 0.005  # m, decoder CARRY macro: the goal moved this far since the line started -> abort
    carry_place_tol: float = 0.004  # m and rad: PLACE trigger tolerances with the decoder CARRY macro (one line usually
    carry_place_yaw: float = math.radians(2.0)   # lands within them; each correction line costs >= 0.5 s of hold time)
    carry_fix_max: int = 1         # decoder CARRY macro: at most this many correction lines, then PLACE where it is
    carry_r_min: float = 0.135     # m, decoder CARRY macro (option b2): a straight carry line that would pass closer to
                                   #   the pan axis goes to a polar midpoint first (gate: carries below 0.135 m touched
                                   #   the arm itself in MuJoCo, review F1)


class PickPlaceExpert:
    """Batched expert for B worlds on the decoder's device (module docstring)."""

    def __init__(self, decoder: DecoderV2, cfg: ExpertConfig | None = None):
        self.cfg = cfg or ExpertConfig()
        self.dec, self.kin = decoder, decoder.kin
        B, dev = decoder.B, decoder.dev
        f = dict(dtype=torch.float64, device=dev)
        self.B, self.dev, self.f = B, dev, f
        self.phase = torch.full((B,), PH["approach"], dtype=torch.long, device=dev)
        self.goal_xy = torch.zeros(B, 2, **f)
        self.goal_yaw = torch.zeros(B, **f)
        self.grasp_yaw = torch.zeros(B, **f)          # chosen gripper yaw for the grasp (absolute, tracked)
        self.has_yaw = torch.zeros(B, dtype=torch.bool, device=dev)
        self.recoveries = torch.zeros(B, dtype=torch.long, device=dev)
        self.lost = torch.zeros(B, dtype=torch.long, device=dev)
        self.c_k = torch.full((B,), -1.0, **f)       # minimum-jerk carry: decisions since its start, start, length (s)
        self.c_p0 = torch.zeros(B, 3, **f)
        self.c_y0 = torch.zeros(B, **f)
        self.c_T = torch.zeros(B, **f)
        self.c_goal = torch.full((B, 2), math.nan, **f)  # decoder CARRY macro: the goal the running carry aims at
        self.c_n = torch.zeros(B, dtype=torch.long, device=dev)    # ... lines to the waypoint since the carry start
        self.c_prev = torch.zeros(B, dtype=torch.long, device=dev)  # ... decoder carry count at the last decision
        self.c_split = torch.zeros(B, dtype=torch.bool, device=dev)  # ... the last carry action was a midpoint hop
        self.pan_axis = torch.tensor(PAN_AXIS_XY, **f)
        cl = decoder.cfg.lift_range
        self.a_lift = 2 * (self.cfg.lift - cl[0]) / (cl[1] - cl[0]) - 1

    # ---- task and reset -----------------------------------------------------------------------------------------
    def reset(self, idx, state, goal_xy, goal_yaw):
        """Worlds idx get the goal (cube centre x, y [n, 2] m; cube yaw [n] rad, mod 90 deg) and a phase inferred from
        the decoder mode and the state (a WorldState of those worlds): decoder in hold and the cube held -> carry;
        decoder in a macro -> grasp / place; else approach. Call after decoder.reset."""
        idx = self.dec._ids(idx)
        self.goal_xy[idx] = goal_xy.to(**self.f)
        self.goal_yaw[idx] = goal_yaw.to(**self.f)
        held, _ = self._held(state, self.dec.q_cmd[idx])
        mode = self.dec.mode[idx]
        ph = torch.full_like(mode, PH["approach"])
        ph = torch.where((mode >= M["g_brake"]) & (mode <= M["g_lift"]), PH["grasp"], ph)
        ph = torch.where(mode == M["hold"], torch.where(held, PH["carry"], PH["recover"]), ph)
        ph = torch.where(mode >= M["p_brake"], PH["place"], ph)
        self.phase[idx] = ph
        self.recoveries[idx], self.lost[idx] = 0, 0
        self.has_yaw[idx] = False
        self.c_n[idx], self.c_prev[idx], self.c_split[idx] = 0, self.dec.counts["carries"][idx], False
        self.c_goal[idx] = math.nan

    # ---- helpers --------------------------------------------------------------------------------------------------
    def _held(self, state, q_cmd=None):
        """(cube between the jaws and lifted [n], cube centre in the grasp frame [n, 3]) from the measured state."""
        c = self.cfg
        q = state.q.to(**self.f)
        p, r = self.kin.fk(q)
        rel = (r.transpose(-1, -2) @ (state.cube_pos.to(p) - p)[..., None])[..., 0]
        box = torch.tensor(c.held_box, **self.f)
        bottom = cube_bottom(state.cube_pos.to(p), state.cube_quat.to(p))
        held = (rel.abs() < box).all(-1) & (q[:, 5] > 0.08) & (q[:, 5] < 0.35) & (bottom > c.held_lift)
        return held, rel

    def _choose_grasp_yaw(self, cube_xy, cube_yaw):
        """Cube face normal (cube yaw + k 90 deg) with the smallest worst wrist roll over the pick and the place."""
        a_c = torch.atan2(cube_xy[:, 1] - self.pan_axis[1], cube_xy[:, 0] - self.pan_axis[0])
        a_g = torch.atan2(self.goal_xy[:, 1] - self.pan_axis[1], self.goal_xy[:, 0] - self.pan_axis[0])
        turn = wrap90(self.goal_yaw - cube_yaw)
        cand = cube_yaw[:, None] + QUARTER * torch.arange(4, **self.f)
        cost = torch.maximum(wrap(cand - a_c[:, None]).abs(), wrap(cand + turn[:, None] - a_g[:, None]).abs())
        return cand.gather(1, cost.argmin(1, keepdim=True))[:, 0]

    def _plan(self, wp, wy):
        """Decoder action rows 0:4 that drive the target to the waypoint (module docstring)."""
        c, d = self.cfg, self.dec
        td = d.T_d
        v0, w0 = d.vel, d.yrate
        v = (wp - d.pos - v0 * td / 2) / td
        if d.filt:                                        # filter blend: the set point whose rest point is wp
            v = self._land(wp - d.pos, d.fv, d.fa)
        if d.hann:                                        # Hann blends: rest point pos + alpha v + beta (linear)
            alpha, beta, beta_w = d.rest_terms()
            v = (wp - d.pos - beta) / alpha
            v0, w0 = d.setpoint()
        dist = (wp - d.pos).norm(dim=-1)
        u = (wp - d.pos) / dist.clamp_min(1e-9)[:, None]
        cap = torch.minimum(torch.full_like(dist, c.v_cruise),
                            d._allow(dist, (v0 * u).sum(-1), torch.full_like(dist, c.a_plan)).clamp_min(0))
        vn = v.norm(dim=-1, keepdim=True)
        v = v * torch.where(vn > cap[:, None], cap[:, None] / vn.clamp_min(1e-12), torch.ones_like(vn))
        ey = wy - d.yaw
        w = (ey - w0 * td / 2) / td
        if d.filt:
            w = self._land(ey[:, None], d.fw[:, None], d.fwa[:, None])[:, 0]
        if d.hann:
            w = (ey - beta_w) / alpha
        wcap = torch.minimum(torch.full_like(ey, c.w_cruise),
                             d._allow(ey.abs(), torch.sign(ey) * w0, torch.full_like(ey, c.alpha_plan)).clamp_min(0))
        w = torch.maximum(torch.minimum(w, wcap), -wcap)
        return torch.cat((v * td / d.cfg.d_max, (w * td / d.cfg.dyaw_max)[:, None]), -1).clamp(-1, 1)

    def _land(self, err, x, xd):
        """Filter blend (decoder blend "filter"): the set point u held for one decision after which the filter, set to
        0, comes to rest err (m or rad, [B, k]) away; the filter is linear: rest(u) = alpha u + beta(x, xd)."""
        from nedm.so101_policy.decoder import filter_step
        d = self.dec
        wn = d.cfg.wn

        def rest(x, xd, u):
            tot = torch.zeros_like(x)
            for _ in range(d.cfg.decision_steps):
                x, xd, disp = filter_step(x, xd, u, wn)
                tot = tot + disp
            return tot + 2 * x / wn + xd / (wn * wn)
        alpha = rest(torch.zeros_like(x), torch.zeros_like(x), torch.ones_like(x))
        beta = rest(x, xd, torch.zeros_like(x))
        return (err - beta) / alpha

    def _minjerk_carry(self, carry, wp, wy, a_plan):
        """Carry along one minimum-jerk segment from the carry start to the (re-planned) waypoint: each decision sets
        the target velocity halfway between the segment's velocity at the decision end and the velocity that lands
        the target on the segment there; after the segment the waypoint plan (a_plan) finishes."""
        c, d = self.cfg, self.dec
        td = d.T_d
        start = carry & (self.c_k < 0)
        if bool(start.any()):
            dist = (wp - d.pos).norm(dim=-1)
            T = torch.maximum(torch.full_like(dist, c.carry_t_min),
                              torch.maximum(1.875 * dist / c.carry_v_peak, 1.875 * (wy - d.yaw).abs() / c.carry_w_peak))
            self.c_T = torch.where(start, torch.ceil(T / td) * td, self.c_T)
            self.c_p0 = torch.where(start[:, None], d.pos, self.c_p0)
            self.c_y0 = torch.where(start, d.yaw, self.c_y0)
            self.c_k = torch.where(start, torch.zeros_like(self.c_k), self.c_k)
        s1 = ((self.c_k + 1) * td / self.c_T.clamp_min(1e-9)).clamp(0, 1)
        mj = s1 * s1 * s1 * (10 - 15 * s1 + 6 * s1 * s1)
        dmj = 30 * s1 * s1 * (1 - s1) ** 2 / self.c_T.clamp_min(1e-9)                    # d mj / dt (1/s)
        pd = self.c_p0 + (wp - self.c_p0) * mj[:, None]
        yd = self.c_y0 + (wy - self.c_y0) * mj
        v = 0.5 * (2 * (pd - d.pos) / td - d.vel) + 0.5 * (wp - self.c_p0) * dmj[:, None]
        w = 0.5 * (2 * (yd - d.yaw) / td - d.yrate) + 0.5 * (wy - self.c_y0) * dmj
        a = torch.cat((v * td / d.cfg.d_max, (w * td / d.cfg.dyaw_max)[:, None]), -1).clamp(-1, 1)
        on = carry & (self.c_k * td < self.c_T)
        self.c_k = torch.where(carry, self.c_k + 1, torch.full_like(self.c_k, -1.0))
        return torch.where(on[:, None], a, a_plan)

    def _carry_action(self, wp, wy):
        """Decoder CARRY macro (option b2) action rows 0:4 toward the carry waypoint wp [B, 3] and gripper yaw wy [B]:
        the straight line, or, if it would pass within ``carry_r_min`` of the pan axis, the polar midpoint (radius
        raised so that both halves clear it; half the yaw turn) first. -> (action [B, 4], midpoint hop [B] bool)."""
        c, d = self.cfg, self.dec
        p, w = d.pos[:, :2] - self.pan_axis, wp[:, :2] - self.pan_axis
        dd = w - p
        t = (-(p * dd).sum(-1) / (dd * dd).sum(-1).clamp_min(1e-18)).clamp(0, 1)
        split = (p + t[:, None] * dd).norm(dim=-1) < c.carry_r_min
        ap, aw = torch.atan2(p[:, 1], p[:, 0]), torch.atan2(w[:, 1], w[:, 0])
        da = wrap(aw - ap)
        am = ap + da / 2
        rm = torch.maximum((p.norm(dim=-1) + w.norm(dim=-1)) / 2, c.carry_r_min / torch.cos(da.abs() / 4))
        rm = rm.clamp(max=d.cfg.r_range[1])
        mid = self.pan_axis + torch.stack((torch.cos(am), torch.sin(am)), -1) * rm[:, None]
        tgt = torch.where(split[:, None], mid, wp[:, :2])
        ytgt = torch.where(split, d.yaw + (wy - d.yaw) / 2, wy)
        a = torch.zeros(self.B, 4, **self.f)
        a[:, 0:2] = (tgt - d.pos[:, :2]) / d.cfg.carry_reach
        a[:, 3] = (ytgt - d.yaw) / d.cfg.hold_turn_max
        return a.clamp(-1, 1), split

    def _yaw_target(self, target, ref):
        """The absolute yaw equal to target mod 2 pi nearest ref."""
        return ref + wrap(target - ref)

    # ---- one decision ---------------------------------------------------------------------------------------------
    @torch.no_grad()
    def act(self, state):
        """WorldState [B] -> decoder action [B, 6] for this decision; advances the phase machine."""
        c, d = self.cfg, self.dec
        cube = state.cube_pos.to(**self.f)
        quat = state.cube_quat.to(**self.f)
        cyaw, _ = cube_yaw_tilt(quat)
        held, rel = self._held(state)
        mode = d.mode
        ph = self.phase
        a = torch.zeros(self.B, 6, **self.f)
        a[:, 4] = -1.0
        a[:, 5] = self.a_lift

        # grasp yaw: chosen fresh after a reset or a recovery, then the face normal nearest the last choice
        fresh = ~self.has_yaw
        if bool(fresh.any()):
            self.grasp_yaw = torch.where(fresh, self._choose_grasp_yaw(cube[:, :2], cyaw), self.grasp_yaw)
            self.has_yaw = self.has_yaw | fresh
        self.grasp_yaw = torch.where(ph == PH["approach"], self.grasp_yaw + wrap90(cyaw - self.grasp_yaw),
                                     self.grasp_yaw)

        # phase transitions on the decoder mode
        free, hold = mode == M["free"], mode == M["hold"]
        in_grasp = (mode >= M["g_brake"]) & (mode <= M["g_lift"])
        self.lost = torch.where((ph == PH["carry"]) & ~held, self.lost + 1, torch.zeros_like(self.lost))
        new = ph.clone()
        new = torch.where((ph == PH["grasp"]) & free, PH["approach"], new)                 # aborted
        new = torch.where((ph == PH["grasp"]) & hold, torch.where(held, PH["carry"], PH["recover"]), new)
        new = torch.where((ph == PH["carry"]) & (self.lost >= c.lost_decisions), PH["recover"], new)
        new = torch.where((ph == PH["place"]) & free, PH["done"], new)
        back = (ph == PH["recover"]) & free
        self.recoveries = torch.where(back, self.recoveries + 1, self.recoveries)
        new = torch.where(back, torch.where(self.recoveries > c.max_recover, PH["done"], PH["approach"]), new)
        self.has_yaw = self.has_yaw & ~back
        if bool(back.any()):
            self.grasp_yaw = torch.where(back, self._choose_grasp_yaw(cube[:, :2], cyaw), self.grasp_yaw)
            self.has_yaw = self.has_yaw | back
        if d.cfg.carry_macro:   # CARRY lines to the waypoint itself (not a polar-midpoint hop) since the carry start
            started = d.counts["carries"] > self.c_prev
            self.c_n = torch.where(started & ~self.c_split, self.c_n + 1, self.c_n)
            self.c_n = torch.where((new == PH["carry"]) & (ph != PH["carry"]), torch.zeros_like(self.c_n), self.c_n)
            self.c_prev = d.counts["carries"].clone()
        self.phase = ph = new

        # approach: waypoint above the cube, then the GRASP trigger
        gz = torch.maximum(cube[:, 2], torch.full_like(cube[:, 2], CUBE_REST_Z)) + 0.001
        gy = self._yaw_target(self.grasp_yaw, d.yaw)
        yaw_off = (gy - d.yaw).abs() > c.yaw_ok
        wz = gz + c.pre_height + torch.where(yaw_off, c.clear, 0.0)
        # low near the cube (after an abort or a re-open): straight up first (keep-out rule: TCP z >= 4 cm within 5 cm)
        low = ((d.pos[:, :2] - cube[:, :2]).norm(dim=-1) < c.climb_r) & (d.pos[:, 2] < wz - c.z_tol - 0.005)
        wxy = torch.where(low[:, None] & ((d.pos[:, :2] - cube[:, :2]).norm(dim=-1) > c.pos_tol)[:, None],
                          d.pos[:, :2], cube[:, :2])
        wp = torch.cat((wxy, wz[:, None]), -1)
        appr = (ph == PH["approach"]) & free
        a_app = self._plan(wp, gy)
        err = d.pos - wp
        ready = (appr & (err[:, :2].norm(dim=-1) < c.pos_tol) & (err[:, 2].abs() < c.z_tol)
                 & ((gy - d.yaw).abs() < c.yaw_tol) & (d.vel.norm(dim=-1) < c.still_v)
                 & (state.cube_v.to(**self.f).norm(dim=-1) < c.cube_still))
        a[:, :4] = torch.where(appr[:, None], a_app, a[:, :4])
        a[:, 4] = torch.where(ready, 1.0, a[:, 4])
        self.phase = torch.where(ready, PH["grasp"], self.phase)

        # grasp macro: hold the trigger; abort before the close if the cube left the descent axis
        pre = torch.zeros_like(appr)
        for m in GRASP_PRE_CLOSE:
            pre |= mode == m
        off = (cube[:, :2] - d.macro_goal[:, :2]).norm(dim=-1) > c.abort_off
        keep = (ph == PH["grasp"]) & in_grasp
        a[:, 4] = torch.where(keep, torch.where(pre & off, -1.0, 1.0), a[:, 4])

        # carry: cube centre over the goal at the goal yaw, then the PLACE trigger
        carry = (ph == PH["carry"]) & hold
        _, r = self.kin.fk(state.q.to(**self.f))
        turn = wrap90(self.goal_yaw - cyaw)
        fy = d.yaw + turn                                                    # target gripper yaw
        cf, sf = torch.cos(fy), torch.sin(fy)
        rx = rel[:, 0] + c.release_shift
        offs = torch.stack((cf * rx - sf * rel[:, 1], sf * rx + cf * rel[:, 1]), -1)    # grasp-frame offset at fy
        wp_c = torch.cat((self.goal_xy - offs, d.pos[:, 2:3]), -1)
        a_car = self._plan(wp_c, fy)
        if c.carry_profile == "minjerk":
            a_car = torch.where(carry[:, None], self._minjerk_carry(carry, wp_c, fy, a_car), a_car)
        abort = torch.zeros_like(carry)
        if d.cfg.carry_macro:                           # option b2: one CARRY line per decision at rest
            a_car, split = self._carry_action(wp_c, fy)
            self.c_split = torch.where(carry, split, self.c_split)
            # a new goal while a line runs: abort it (brake to rest), then carry again from rest
            abort = carry & d.carry_on & ((self.goal_xy - self.c_goal).norm(dim=-1) > c.carry_retarget)
            self.c_n = torch.where(abort, torch.zeros_like(self.c_n), self.c_n)    # the new goal's line: first again
            self.c_goal = torch.where((carry & ~d.carry_on & ~d.carry_brk)[:, None], self.goal_xy, self.c_goal)
        err_c = (d.pos[:, :2] - wp_c[:, :2]).norm(dim=-1)
        ptol, pyaw = (c.carry_place_tol, c.carry_place_yaw) if d.cfg.carry_macro else (c.place_tol, c.place_yaw_tol)
        aim = (err_c < ptol) & (turn.abs() < pyaw)
        if d.cfg.carry_macro:                           # after carry_fix_max correction lines PLACE where it is
            aim = aim | ((self.c_n >= 1 + c.carry_fix_max) & ~self.c_split)
        place = carry & aim & (d.vel.norm(dim=-1) < c.still_v) & (d.yrate.abs() < 0.05) & ~d.carry_on & ~d.carry_brk
        a[:, :4] = torch.where(carry[:, None], a_car, a[:, :4])
        a[:, 4] = torch.where(place, 1.0, a[:, 4])
        a[:, 5] = torch.where(abort, 1.0, a[:, 5])
        self.phase = torch.where(place, PH["place"], self.phase)

        # recover: a PLACE trigger re-opens the jaw and retreats
        rec = (self.phase == PH["recover"]) & hold
        a[:, 4] = torch.where(rec & d.armed, 1.0, a[:, 4])            # (an unarmed trigger first arms with -1)
        a[:, :4] = torch.where(((self.phase == PH["done"]) | (self.phase == PH["place"]))[:, None],
                               torch.zeros_like(a[:, :4]), a[:, :4])
        return a.clamp(-1.0, 1.0)

    @property
    def phase_names(self):
        return [PHASES[k] for k in self.phase.tolist()]
