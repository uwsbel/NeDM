"""Privileged closed-loop pick-and-place expert (batched finite-state machine that outputs decoder actions).

Inputs every 20 ms step: the true state (``WorldState``: joint angles, the cube pose), the task goal (cube centre x, y
on the table and the cube yaw, mod 90 deg) and the ``CommandDecoder`` it drives (its Cartesian target). Output: the
decoder action a [B, 5]. It re-plans from the CURRENT cube pose at every step, so the same expert serves both worlds,
disturbed tasks and DAgger labels (``act`` can be called on any state; ``reset`` infers the phase from the state).

Phases (``PHASES``) and their waypoints (grasp point position and closing-axis yaw):

| Phase | Waypoint | Jaw | Ends when |
|---|---|---|---|
| approach | ``pre_height`` above the grasp point over the cube; grasp yaw | open | waypoint reached -> descend |
| descend | grasp point ``grasp_dz`` above the cube centre | open | reached and the arm there -> close; cube off the line by > ``off_line`` -> recover |
| close | hold | close | jaw move done -> grip |
| grip | hold ``grip_s`` | closed | cube between the jaws -> lift, else recover |
| lift | ``lift`` up | closed | reached -> carry; cube lost -> recover |
| carry | over the goal (cube offset in the grasp frame kept, plus ``release_shift``), goal yaw | closed | reached -> lower; cube lost -> recover |
| lower | cube bottom ``place_dz`` above the table | closed | reached or cube on the table -> release |
| release | hold | open | jaw open -> retreat |
| retreat | ``retreat`` up | open | reached -> done |
| done | hold | open | |
| recover | straight up to the approach height | open | jaw open and height reached -> approach (at most ``max_recover`` times, then done) |

Motion: every waypoint is reached by a straight grasp-point line with minimum-jerk timing (the collector's
``Plan.cart`` rule) of duration max(t_min, 1.875 distance / v_peak, 1.875 |yaw turn| / yaw_peak, 1.875 |pan or wrist
roll turn| / joint_peak), started from the
decoder's rest position (where its filter would stop: pos + 2 v / wn + a / wn^2). The commanded velocities sum to the
planned displacement, so the decoder target ends on the waypoint. The segment re-starts when the waypoint moves by more
than ``replan_tol`` (cube pushed, goal changed) or when a finished segment left the target more than ``settle_tol``
off. Grasp yaw: a cube face normal (cube yaw + k 90 deg), the one with the smallest wrist roll (|yaw - radial
direction|) over the pick and the place (with the goal turn), i.e. the collector's ``Scene.grasp_yaw`` choice nearest the
radial direction when the place is near the pick; then tracked continuously while the cube turns.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from nedm.so101_policy.decoder import CONTROL_DT, CommandDecoder, minjerk
from nedm.so101_policy.kin import PAN_AXIS_XY, quat_mat, wrap, yaw_of

PHASES = ("approach", "descend", "close", "grip", "lift", "carry", "lower", "release", "retreat", "done", "recover")
PH = {n: k for k, n in enumerate(PHASES)}
CUBE_HALF = 0.015
CUBE_REST_Z = 0.014993     # resting cube centre height in the data (m)
QUARTER = math.pi / 2


def wrap90(a):
    """Angle -> (-45, 45] deg equivalent under the cube's 90 deg symmetry."""
    return -(torch.remainder(-a + QUARTER / 2, QUARTER) - QUARTER / 2)


def cube_yaw_tilt(quat):
    """Cube orientation (w x y z) [B, 4] -> (yaw mod 90 deg in [0, pi/2), tilt rad): the axis nearest the world vertical
    is 'up'; the yaw is that of the next cube axis projected on the table; tilt = angle of 'up' from vertical."""
    r = quat_mat(quat)
    up = r[:, 2, :].abs().argmax(-1)                                           # cube axis nearest vertical
    tilt = torch.acos(r[:, 2, :].abs().amax(-1).clamp(max=1.0))
    side = torch.remainder(up + 1, 3)
    col = r.gather(-1, side[:, None, None].expand(-1, 3, 1))[..., 0]          # that axis in the world
    yaw = torch.remainder(torch.atan2(col[:, 1], col[:, 0]), QUARTER)
    return yaw, tilt


@dataclass
class ExpertConfig:
    pre_height: float = 0.05       # m, approach height above the grasp height
    grasp_dz: float = 0.002        # m, grasp point above the cube centre (data: 0-2 mm)
    lift: float = 0.05             # m (data 3-12 cm)
    place_dz: float = 0.0005       # m, cube bottom above the table at the release (data 0-3 mm)
    release_shift: float = 0.002   # m, the cube moves this far along the closing axis (+x) when the jaw opens
    retreat: float = 0.065         # m (data 4-10 cm)
    v_peak: dict = field(default_factory=lambda: dict(approach=0.33, descend=0.25, lift=0.25, carry=0.33, lower=0.2,
                                                      retreat=0.3, recover=0.2, hold=0.1))
    t_min: dict = field(default_factory=lambda: dict(approach=0.5, descend=0.3, lift=0.35, carry=0.5, lower=0.35,
                                                     retreat=0.25, recover=0.3, hold=0.2))
    yaw_peak: float = 1.5          # rad/s, closing-axis yaw
    joint_peak: float = 1.8        # rad/s, pan and wrist roll (the NRD's wrist roll oscillated at 2.3 rad/s)
    grip_s: float = 0.12           # s (data 0.1-0.25 s)
    settle_tol: float = 0.0015     # m, rest point of the decoder target on the waypoint
    chain_tol: float = 0.006       # m, decoder target near the waypoint: enough to chain approach, lift, carry, retreat
    yaw_tol: float = math.radians(2.0)
    arm_tol: float = 0.004         # m, measured grasp point at the grasp waypoint before closing
    replan_tol: float = 0.01       # m
    off_line: float = 0.008        # m, cube off the descent line -> recover
    held_box: tuple = (0.015, 0.015, 0.02)   # m, |cube centre in the grasp frame| limits (x closing axis, y, z)
    held_jaw: tuple = (0.10, 0.35)            # rad, jaw angle with the cube between the jaws (firm grip 0.167)
    lost_steps: int = 3
    open_jaw: float = 0.35         # rad, measured jaw angle that counts as open
    max_recover: int = 3
    r_safe: float = 0.13           # m, straight lines stay this far from the pan axis (else a via point)
    r_via: float = 0.17            # m, smallest via-point radius


class PickPlaceExpert:
    """Batched expert for B worlds on the decoder's device (see the module docstring)."""

    def __init__(self, decoder: CommandDecoder, cfg: ExpertConfig | None = None):
        self.cfg = cfg or ExpertConfig()
        self.dec, self.kin = decoder, decoder.kin
        B, dev = decoder.B, decoder.dev
        f = dict(dtype=torch.float64, device=dev)
        self.B, self.dev = B, dev
        self.phase = torch.full((B,), PH["approach"], dtype=torch.long, device=dev)
        self.timer = torch.zeros(B, dtype=torch.long, device=dev)
        self.goal_xy = torch.zeros(B, 2, **f)
        self.goal_yaw = torch.zeros(B, **f)
        self.grasp_yaw = torch.zeros(B, **f)          # chosen gripper yaw for the grasp (absolute, tracked)
        self.has_yaw = torch.zeros(B, dtype=torch.bool, device=dev)
        self.recoveries = torch.zeros(B, dtype=torch.long, device=dev)
        self.lost = torch.zeros(B, dtype=torch.long, device=dev)
        self.anchor = torch.zeros(B, 3, **f)          # phase anchor: position of the decoder target at phase entry
        # minimum-jerk segment
        self.p0 = torch.zeros(B, 3, **f)
        self.y0 = torch.zeros(B, **f)
        self.wp = torch.zeros(B, 3, **f)
        self.wy = torch.zeros(B, **f)
        self.k = torch.zeros(B, **f)
        self.n = torch.ones(B, **f)
        self.sent = torch.zeros(B, 3, **f)
        self.ysent = torch.zeros(B, **f)
        self.pan_axis = torch.tensor(PAN_AXIS_XY, **f)
        self.started = torch.zeros(B, dtype=torch.bool, device=dev)

    # ---- task and reset -----------------------------------------------------------------------------------
    def reset(self, idx, state, goal_xy, goal_yaw):
        """Worlds idx get the goal (cube centre x, y [n, 2] m; cube yaw [n] rad, mod 90 deg) and a phase inferred from
        the state (a WorldState of those worlds): cube held -> lift (it re-plans to the carry); grasp point less than
        3 cm above the grasp height -> recover (straight up first); else approach (the jaw opens on the way). Call
        after decoder.reset."""
        self.goal_xy[idx] = goal_xy.to(self.goal_xy)
        self.goal_yaw[idx] = goal_yaw.to(self.goal_yaw)
        held, _ = self._held_parts(state)
        gp, _ = self.kin.fk(state.q.to(self.goal_yaw))
        low = gp[:, 2] < state.cube_pos[:, 2].to(gp) + self.cfg.grasp_dz + 0.03
        ph = torch.where(held, PH["lift"], torch.where(low, PH["recover"], PH["approach"]))
        self.phase[idx] = ph.to(self.phase)
        for x in (self.timer, self.recoveries, self.lost):
            x[idx] = 0
        self.has_yaw[idx] = False
        self.started[idx] = False
        self.anchor[idx] = self.dec.pos[idx]
        if bool(held.any()):   # a held cube keeps its grasp yaw
            q = state.q.to(self.goal_yaw)
            _, r = self.kin.fk(q)
            gy = self.grasp_yaw[idx]
            gy = torch.where(held, yaw_of(r), gy)
            self.grasp_yaw[idx] = gy
            hy = self.has_yaw[idx]
            self.has_yaw[idx] = hy | held

    # ---- helpers ------------------------------------------------------------------------------------------
    def _held_parts(self, state):
        """(cube between the jaws [B], cube centre in the grasp frame [B, 3])."""
        c = self.cfg
        q = state.q.to(torch.float64)
        p, r = self.kin.fk(q)
        rel = (r.transpose(-1, -2) @ (state.cube_pos.to(p) - p)[..., None])[..., 0]
        box = torch.tensor(c.held_box, dtype=p.dtype, device=p.device)
        held = (rel.abs() < box).all(-1) & (q[:, 5] > c.held_jaw[0]) & (q[:, 5] < c.held_jaw[1])
        return held, rel

    def _rest(self):
        """Where the decoder target comes to rest if the commands stop (filter integral)."""
        d, w = self.dec, self.dec.cfg.wn
        return d.pos + 2 * d.vel / w + d.acc / (w * w), d.yaw + 2 * d.yrate / w + d.yacc / (w * w)

    def _choose_grasp_yaw(self, cube_xy, cube_yaw):
        """Cube face normal (cube yaw + k 90 deg) with the smallest worst wrist roll over the pick and the place."""
        a_c = torch.atan2(cube_xy[:, 1] - self.pan_axis[1], cube_xy[:, 0] - self.pan_axis[0])
        a_g = torch.atan2(self.goal_xy[:, 1] - self.pan_axis[1], self.goal_xy[:, 0] - self.pan_axis[0])
        turn = wrap90(self.goal_yaw - cube_yaw)
        cand = cube_yaw[:, None] + QUARTER * torch.arange(4, dtype=cube_yaw.dtype, device=cube_yaw.device)
        cost = torch.maximum(wrap(cand - a_c[:, None]).abs(), wrap(cand + turn[:, None] - a_g[:, None]).abs())
        return cand.gather(1, cost.argmin(1, keepdim=True))[:, 0]

    # ---- one step -----------------------------------------------------------------------------------------
    @torch.no_grad()
    def act(self, state):
        """WorldState [B] -> decoder action [B, 5]; advances the phase machine."""
        c, d = self.cfg, self.dec
        f64 = torch.float64
        q = state.q.to(f64)
        cube = state.cube_pos.to(f64)
        cyaw, _ = cube_yaw_tilt(state.cube_quat.to(f64))
        gp, r = self.kin.fk(q)
        held, rel = self._held_parts(state)
        rest, ryaw = self._rest()
        ph = self.phase.clone()
        prev = ph.clone()
        jaw = q[:, 5]

        # grasp yaw: chosen fresh, then the face normal nearest the last choice (the cube may turn)
        fresh = ~self.has_yaw
        if bool(fresh.any()):
            self.grasp_yaw = torch.where(fresh, self._choose_grasp_yaw(cube[:, :2], cyaw), self.grasp_yaw)
            self.has_yaw = self.has_yaw | fresh
        tracking = (ph == PH["approach"]) | (ph == PH["descend"])
        follow = self.grasp_yaw + wrap90(cyaw - self.grasp_yaw)
        self.grasp_yaw = torch.where(tracking, follow, self.grasp_yaw)

        # held bookkeeping
        carrying = (ph == PH["lift"]) | (ph == PH["carry"]) | (ph == PH["lower"])
        self.lost = torch.where(carrying & ~held, self.lost + 1, torch.zeros_like(self.lost))
        arrived = self._arrived(rest, ryaw)

        # transitions (evaluated on the phase at the start of the step)
        grip_steps = max(1, round(c.grip_s / CONTROL_DT))
        line_off = (cube[:, :2] - d.pos[:, :2]).norm(dim=-1) > c.off_line
        at_grasp = arrived & ((gp - self.wp).norm(dim=-1) < c.arm_tol)
        on_table = cube[:, 2] < CUBE_REST_Z + c.place_dz + 0.0005
        new = ph.clone()
        new = torch.where((ph == PH["approach"]) & arrived & self.started, PH["descend"], new)
        new = torch.where((ph == PH["descend"]) & line_off, PH["recover"], new)
        new = torch.where((ph == PH["descend"]) & ~line_off & at_grasp & self.started, PH["close"], new)
        new = torch.where((ph == PH["close"]) & d.jaw_done & (d.jaw_mode == 1) & (self.timer > 0), PH["grip"], new)
        new = torch.where((ph == PH["grip"]) & (self.timer >= grip_steps), torch.where(held, PH["lift"], PH["recover"]), new)
        new = torch.where(carrying & (self.lost >= c.lost_steps), PH["recover"], new)
        new = torch.where((ph == PH["lift"]) & arrived & self.started & (self.lost < c.lost_steps), PH["carry"], new)
        new = torch.where((ph == PH["carry"]) & arrived & self.started & (self.lost < c.lost_steps), PH["lower"], new)
        new = torch.where((ph == PH["lower"]) & ((arrived & self.started) | on_table) & (self.lost < c.lost_steps),
                          PH["release"], new)
        new = torch.where((ph == PH["release"]) & d.jaw_done & (d.jaw_mode == 0) & (jaw > c.open_jaw), PH["retreat"], new)
        new = torch.where((ph == PH["retreat"]) & arrived & self.started, PH["done"], new)
        back = (ph == PH["recover"]) & arrived & self.started & (jaw > c.open_jaw)
        self.recoveries = torch.where((new == PH["recover"]) & (ph != PH["recover"]), self.recoveries + 1, self.recoveries)
        new = torch.where(back, PH["approach"], new)
        new = torch.where((new == PH["recover"]) & (self.recoveries > c.max_recover), PH["done"], new)
        self.has_yaw = self.has_yaw & ~back                      # a fresh grasp yaw after a recovery
        if bool(back.any()):
            self.grasp_yaw = torch.where(back, self._choose_grasp_yaw(cube[:, :2], cyaw), self.grasp_yaw)
            self.has_yaw = self.has_yaw | back
        changed = new != prev
        self.phase = new
        self.timer = torch.where(changed, torch.zeros_like(self.timer), self.timer + 1)
        self.anchor = torch.where(changed[:, None], rest, self.anchor)
        ph = new

        # waypoints of the (new) phase
        wp, wy = self._waypoints(ph, cube, cyaw, rel, r, rest, ryaw)

        # minimum-jerk segment from the rest position to the waypoint
        restart = changed | ~self.started | ((wp - self.wp).norm(dim=-1) > c.replan_tol)
        restart |= (self.k >= self.n) & (((wp - rest).norm(dim=-1) > c.settle_tol) | ((wy - ryaw).abs() > c.yaw_tol))
        name_v = torch.zeros_like(rest[:, 0])
        name_t = torch.zeros_like(rest[:, 0])
        for name, k in PH.items():
            key = name if name in c.v_peak else "hold"
            name_v = torch.where(ph == k, c.v_peak[key], name_v)
            name_t = torch.where(ph == k, c.t_min[key], name_t)
        dist = (wp - rest).norm(dim=-1)
        dyaw = (wy - ryaw).abs()
        a0 = torch.atan2(rest[:, 1] - self.pan_axis[1], rest[:, 0] - self.pan_axis[0])
        a1 = torch.atan2(wp[:, 1] - self.pan_axis[1], wp[:, 0] - self.pan_axis[0])
        dpan = wrap(a1 - a0)
        droll = (wy - ryaw - dpan).abs()                         # wrist roll follows yaw minus pan (top-down)
        joint = torch.maximum(dpan.abs(), droll) / c.joint_peak
        steps = torch.ceil(torch.stack((name_t, 1.875 * dist / name_v, 1.875 * dyaw / c.yaw_peak, 1.875 * joint), -1)
                           .amax(-1) / CONTROL_DT)
        self.p0 = torch.where(restart[:, None], rest, self.p0)
        self.y0 = torch.where(restart, ryaw, self.y0)
        self.n = torch.where(restart, steps.clamp_min(1), self.n)
        self.k = torch.where(restart, torch.zeros_like(self.k), self.k)
        self.sent = torch.where(restart[:, None], torch.zeros_like(self.sent), self.sent)
        self.ysent = torch.where(restart, torch.zeros_like(self.ysent), self.ysent)
        self.started = self.started | restart
        self.wp, self.wy = wp, wy
        self.k = torch.minimum(self.k + 1, self.n)
        s = minjerk(self.k / self.n)
        u = ((wp - self.p0) * s[:, None] - self.sent) / CONTROL_DT
        uy = ((wy - self.y0) * s - self.ysent) / CONTROL_DT
        vmax, ymax = d.cfg.v_max, d.cfg.yaw_rate_max
        un = u.norm(dim=-1, keepdim=True)
        u = u * torch.where(un > vmax, vmax / un.clamp_min(1e-12), torch.ones_like(un))
        uy = uy.clamp(-ymax, ymax)
        self.sent = self.sent + u * CONTROL_DT
        self.ysent = self.ysent + uy * CONTROL_DT
        close = (ph == PH["close"]) | (ph == PH["grip"]) | (ph == PH["lift"]) | (ph == PH["carry"]) | (ph == PH["lower"])
        a = torch.cat((u / vmax, (uy / ymax)[:, None], torch.where(close, 1.0, -1.0).to(u)[:, None]), -1)
        return a.clamp(-1.0, 1.0)

    def _arrived(self, rest, ryaw):
        """Segment finished, its rest point on the waypoint and the yaw turned; the decoder target itself within
        settle_tol before closing or releasing (descend, lower, recover), within chain_tol elsewhere (the next segment
        starts from the rest point, so the filter blends the corner)."""
        c, d = self.cfg, self.dec
        ph = self.phase
        tight = (ph == PH["descend"]) | (ph == PH["lower"]) | (ph == PH["recover"])
        tol = torch.where(tight, c.settle_tol, c.chain_tol)
        return ((self.k >= self.n) & ((self.wp - rest).norm(dim=-1) < c.settle_tol)
                & ((self.wp - d.pos).norm(dim=-1) < tol) & ((self.wy - ryaw).abs() < c.yaw_tol))

    def _waypoints(self, ph, cube, cyaw, rel, r, rest, ryaw):
        c, d = self.cfg, self.dec
        grasp_z = cube[:, 2] + c.grasp_dz
        a_rest = torch.atan2(rest[:, 1] - self.pan_axis[1], rest[:, 0] - self.pan_axis[0])

        def yaw_in_roll(target_yaw, at_xy):
            """Waypoint yaw: the target yaw reached through the wrist roll range (no turn through roll +-pi)."""
            a_t = torch.atan2(at_xy[:, 1] - self.pan_axis[1], at_xy[:, 0] - self.pan_axis[0])
            rho_t, rho = wrap(target_yaw - a_t), wrap(ryaw - a_rest)
            return ryaw + (rho_t - rho) + wrap(a_t - a_rest)

        wp, wy = rest.clone(), ryaw.clone()   # waypoint yaws are absolute (unwrapped), near the rest yaw
        # approach / descend / recover
        above = torch.cat((cube[:, :2], (grasp_z + c.pre_height)[:, None]), -1)
        down = torch.cat((cube[:, :2], grasp_z[:, None]), -1)
        gy = yaw_in_roll(self.grasp_yaw, cube[:, :2])
        sel = ph == PH["approach"]
        wp, wy = torch.where(sel[:, None], above, wp), torch.where(sel, gy, wy)
        sel = ph == PH["descend"]
        wp, wy = torch.where(sel[:, None], down, wp), torch.where(sel, gy, wy)
        up = torch.cat((self.anchor[:, :2], torch.maximum(self.anchor[:, 2], grasp_z + c.pre_height)[:, None]), -1)
        sel = ph == PH["recover"]
        wp = torch.where(sel[:, None], up, wp)
        # close / grip / release / done: hold the phase anchor
        for name in ("close", "grip", "release", "done"):
            sel = ph == PH[name]
            wp = torch.where(sel[:, None], self.anchor, wp)
        # lift
        sel = ph == PH["lift"]
        lift = self.anchor + torch.tensor([0.0, 0.0, c.lift], dtype=rest.dtype, device=rest.device)
        wp = torch.where(sel[:, None], lift, wp)
        # carry / lower: the cube centre over the goal at the goal yaw, its offset in the grasp frame kept
        dgrip = wrap90(cyaw - yaw_of(r))                         # cube yaw relative to the gripper
        fy = self.goal_yaw - dgrip
        fy = ryaw + wrap90(fy - ryaw)
        cf, sf = torch.cos(fy), torch.sin(fy)
        rx = rel[:, 0] + c.release_shift                         # where the cube will be after the release
        off = torch.stack((cf * rx - sf * rel[:, 1], sf * rx + cf * rel[:, 1]), -1)   # top-down at fy
        place_xy = self.goal_xy - off
        sel = ph == PH["carry"]
        carry = torch.cat((place_xy, self.anchor[:, 2:3]), -1)
        wp, wy = torch.where(sel[:, None], carry, wp), torch.where(sel, fy, wy)
        sel = ph == PH["lower"]
        low_z = CUBE_REST_Z + c.place_dz + (d.pos[:, 2] - cube[:, 2])
        lower = torch.cat((place_xy, low_z[:, None]), -1)
        wp, wy = torch.where(sel[:, None], lower, wp), torch.where(sel, fy, wy)
        # retreat
        sel = ph == PH["retreat"]
        wp = torch.where(sel[:, None], self.anchor + torch.tensor([0.0, 0.0, c.retreat], dtype=rest.dtype,
                                                                   device=rest.device), wp)
        # a straight line that would pass near the pan axis goes through a via point (same height) instead
        wp = self._via(rest, wp)
        # keep the waypoints inside the decoder workspace
        z_lo, z_hi = d.cfg.z_range
        wp = torch.cat((wp[:, :2], wp[:, 2:3].clamp(z_lo, z_hi)), -1)
        return wp, wy

    def _via(self, rest, wp):
        """Waypoint, or a via point if the straight grasp-point line from the rest point passes within ``r_safe`` of the
        pan axis (the decoder keeps the target 0.10 m out): at the mean pan angle, radius max(r_via, mean radius)."""
        c = self.cfg
        a, b = rest[:, :2] - self.pan_axis, wp[:, :2] - self.pan_axis
        ab = b - a
        t = (-(a * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)).clamp(0.0, 1.0)
        near = (a + t[:, None] * ab).norm(dim=-1) < c.r_safe
        ang_a, ang_b = torch.atan2(a[:, 1], a[:, 0]), torch.atan2(b[:, 1], b[:, 0])
        mid = ang_a + 0.5 * wrap(ang_b - ang_a)
        r = torch.maximum(torch.full_like(mid, c.r_via), 0.5 * (a.norm(dim=-1) + b.norm(dim=-1)))
        via = torch.cat((self.pan_axis + r[:, None] * torch.stack((torch.cos(mid), torch.sin(mid)), -1),
                         torch.maximum(rest[:, 2:3], wp[:, 2:3])), -1)
        return torch.where(near[:, None], via, wp)

    @property
    def phase_names(self):
        return [PHASES[k] for k in self.phase.tolist()]
