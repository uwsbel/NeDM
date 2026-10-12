"""Policy action -> 6 absolute joint targets inside the scripted command family of the grasp NRD data.

The grasp NRD is accurate only on the collector's commands (critic 3.5; review_spec_feasibility sec. 3): top-down
gripper, straight grasp-point lines with minimum-jerk timing, the arm still while the jaw moves, a firm jaw command
to grip and an open command, per-joint command steps and accelerations at the data's p99. ``DecoderV2`` (spec_v2 B1,
review_spec_feasibility P0-1..3) is the study's action design; ``DecoderV1`` (spec_v1 2.4: per-step velocity, binary
jaw) stays for the ablation. Frames: world = arm base (z up, table top z = 0), metres, radians; TCP = the collector's
grasp point (``kin.GRASP_POINT``); yaw = direction of the gripper closing axis (``kin.yaw_of``).

DecoderV2
---------
Action a [B, 6] in [-1, 1], read by ``decide`` once per decision (``decision_steps`` = 5 control steps = 0.1 s):

    a[0:3]  TCP target displacement over the decision, world frame, x d_max = 35 mm (norm clipped to d_max)
    a[3]    yaw change of the closing axis over the decision, x dyaw_max = 0.2 rad
    a[4]    trigger: > 0 starts GRASP (mode free, jaw open) or PLACE (mode hold) if the trigger is armed; while GRASP
            has not started to close the jaw, a[4] <= 0 at a decision aborts it (the policy holds the trigger through
            the descent). A macro start disarms the trigger; a decision with a[4] <= 0 arms it again, so a trigger
            held through a GRASP cannot start a PLACE right after the lift (``armed``, in the observation)
    a[5]    GRASP lift height, read at the trigger: h = 3 + 9 (a[5] + 1) / 2 cm

Modes (``MODES``): ``free`` (jaw open, free motion) -GRASP-> g_brake, g_descend, g_settle, g_close, g_grip, g_lift ->
``hold`` (jaw closed, free motion with the held-cube rules) -PLACE-> p_brake, p_settle, p_lower, p_open, p_retreat ->
``free``. Decisions inside a macro are ignored except the GRASP abort (no exploration noise inside a macro).

Free motion (free, hold), per decision: v_new = d / T_d (T_d = 0.1 s) and w_new = dyaw / T_d. The change from the
current target velocity is bounded, |v_new - v| <= dv_max = a_max T_d 2 / pi (a_max = 1.5 m/s^2 TCP acceleration:
the cosine blend's peak acceleration is pi |dv| / (2 T_d)) and |w_new - w| <= alpha_max T_d 2 / pi (alpha_max = 6
rad/s^2); then the change is scaled down (one factor) so that the joint rates of the blend (Jacobian map at the last
IK solution) keep every joint's peak step <= dq_max and step change <= ddq_max. Then, wall by wall, the velocity
toward a wall at distance D is capped so that the target can still stop before it (D = (v_wall + v) T_d / 2 +
v^2 / (2 a), a = ``wall_margin`` x the joint-limited deceleration along the wall normal; walls: TCP z floor
(``low_min`` + the lowest-point offset of the last IK solution) and ceiling (``z_max`` = 0.15 m; in mode hold also
cube bottom <= 12 cm), radius about the pan axis ``r_range``, pan angle ``angle_max``; yaw: the hold turn cap and the
wrist roll range); walls win. A target outside a radius / angle / ceiling bound (a start pose) only cannot move
farther out. The target then follows the push-T cosine velocity blend (``blend``; so101_planar_decoder): from
the current velocity to v_new over T_d, then constant until the next decision. Yaw rules (P0-3): in mode free, with the
jaw open and the cube on the table, the yaw rate goes to 0 (blend to 0 in the decision) when the jaw collision points
are within ``yaw_free_gap`` = 2 cm (+ the target's travel in one decision) of the cube; in mode hold the gripper yaw
stays within ``hold_turn_max`` = 0.6 rad of its yaw at the end of the GRASP lift.

Macros (copied from the collector's ``Plan.cart`` / ``Plan.jaw`` / ``grasp_approach`` / ``carry_and_place``; every
duration drawn uniformly from the collector's range by a counter-based hash of (world seed, draw counter), so a world
is reproducible on any device and in any batch; a line's duration is then raised, if needed, to keep its minimum-jerk
peak speed <= v_macro = 0.35 m/s and peak acceleration (5.77 D / T^2) <= a_macro = 2.5 m/s^2, as Plan.cart slows a path
that a joint cannot follow):

    GRASP: g_brake    cosine blend of the target velocity and yaw rate to 0 in max(T_d, pi |v| / (2 a_max),
                      pi |w| / (2 alpha_max)) (skipped at rest)
           g_descend  vertical TCP line to the cube centre height (cube state at the trigger) + U(0, 2) mm, U(0.25, 0.5) s
           g_settle   hold U(0, 0.15) s (nominal's "descend" hold)
           g_close    jaw to 0.0 rad (firm) in max(U(0.2, 0.4) s, 1.875 |dtheta| / 4.5 rad/s), arm still
           g_grip     hold U(0.1, 0.25) s
           g_lift     vertical line up by h (policy; capped: TCP z <= z_max, cube bottom <= 12 cm), U(0.35, 0.8) s
    PLACE: p_brake    as g_brake
           p_settle   hold U(0.1, 0.3) s (carry_and_place hold2)
           p_lower    vertical line to the TCP height that puts the lowest cube corner (cube state at the trigger)
                      U(0, 3) mm above the table, U(0.35, 0.7) s
           p_open     jaw to 0.7 rad in max(U(0.15, 0.35) s, 1.875 |dtheta| / 4.5 rad/s), arm still
           p_retreat  vertical line up by U(5, 10) cm (TCP z <= z_max), U(0.25, 0.5) s (collector 4-10 cm: a
                      4 cm retreat leaves the TCP 25-30 mm above the cube top, below the success rule's 3 cm)

A PLACE without a cube in the jaws lowers by 0-3 mm at most, opens and retreats: the re-open after a failed grasp.
The GRASP trigger is ignored while the jaw is still opening (after a reset); the descent's (x, y, z, yaw) end is
``macro_goal`` (for the task's grasp-box rule at the trigger).

Per control step (``step``): the target increment of the mode (blend, minimum-jerk line, or none), the workspace clip
(free motion), the start tilt blended out over ``tilt_s`` = 0.6 s (data start poses tilt up to 60 deg), IK
(``SO101Kin.ik_track``, warm-started, top-down where reachable, the collector's tilt fallback above it), then:

1. Speed guard (push-T): if a joint's IK step exceeds dq_max (data p99 per 20 ms: pan 0.0347, lift 0.0347, elbow
   0.0423, wrist flex 0.0475, wrist roll 0.0409 rad), the TCP and yaw step are scaled down to fit (free motion: the
   lost part is not caught up; macros: the line's clock slows, so the line keeps its shape and end).
2. Floor: if the lowest gripper / moving-jaw collision point of the IK solution (``SO101Kin.lowest_point``, fast set)
   would be below ``low_min`` = 1.5 mm, the target is raised and re-solved. The jaw angle of the check is the one in
   [measured, commanded, scheduled end] with the deepest moving jaw (about 0.17 rad, 14.2 mm below the TCP top-down).
3. Command limiter per arm joint: the command c follows the IK solution with |dc| <= dq_max and |d2c| <= ddq_max
   (data p99: 0.0028, 0.0069, 0.0087, 0.0095, 0.0034 rad; inside a macro line the larger of these and the data's p99
   of the collector phase it copies, ``ddq_line``: the collector's lifts and retreats reach 0.028-0.046 rad), braking-
   aware (the lag behind a target that moves at its own speed closes without overshoot), stopping before the joint
   limits; inactive when the IK solution itself is within the bounds.
4. Lead band (if the measured joint angles are given): |c - q| <= data p99.9 (0.0445, 0.1235, 0.0552, 0.0559, 0.0476
   rad); the band wins over the acceleration bound. Then the collector's target range (``kin.TLO`` .. ``kin.THI``).
5. Jaw: minimum-jerk schedule (Plan.jaw rule), steps clipped to 0.09 rad.

Same code on CPU and GPU (all state per world, [B, ...] float64 tensors; a few ``.any()`` checks skip work). Flags
per step in ``info`` and summed per world in ``counts``. ``macro()`` gives the task's macro phase codes (``MACROS``).

Ablation option ``blend="filter"``: the decision's velocity is the set point of a critically damped 2nd-order filter
(natural frequency ``wn`` = 12.5 rad/s, the v1 filter) instead of the cosine blend: the acceleration is continuous
across decisions (a cosine blend returns the acceleration to 0 at every decision boundary, a 10 Hz ripple while the
target accelerates); the brake ends when the filter has stopped. Measured 2026-10-11 (64 tasks, FSM expert): MuJoCo
63/64, NRD s61 48/64 against 64/64 and 46/64 with the cosine blend: no clear gain (build2_kin_decoder_expert.md).

Option b1, ``blend="hann"`` (build3_fixes.md): every decision starts one more blend of the target velocity, the change
of the set point (u_k - u_{k-1}) spread over 2 T_d with a Hann (raised-cosine) acceleration window; the blends of
consecutive decisions overlap by half and their windows sum to a constant, so equal set-point changes give a constant
acceleration (no ripple at the decision rate) and the acceleration is continuous. Peak acceleration of a decision's
change dv is |dv| / T_d (bound: a_max), the set point is bounded like the cosine blend's end velocity; walls use 2 T_d
(the response lags the set point). The GRASP / PLACE brake is one more Hann blend to 0 of length max(2 T_d,
2 |u| / a_max, 2 |w| / alpha_max, joint bound); ``macro_goal`` is its exact rest point. ``rest_terms()`` gives the
linear rest-point model the expert plans with (cosine: pos + T_d v + v0 T_d / 2).

Option b2, ``carry_macro=True`` (build3_fixes.md): in mode hold there is no free motion. A decision at rest gives a
CARRY target: TCP xy + a[0:2] ``carry_reach`` (norm clipped; relative to the current target), the same height, yaw +
a[3] ``hold_turn_max`` (within the hold turn cap and the wrist roll range); radius / pan-angle bounds as free motion,
and a straight line that would pass inside the inner radius ends where it would cross it (``carry_target``). A target
within ``carry_min_d`` / ``carry_min_yaw`` of the current one is no move. The line copies the collector's
carry_and_place 'move' (Plan.cart): straight TCP line, minimum-jerk timing, the yaw turning with the same timing,
duration U(0.5, 0.9) s x 1.3 (up to 6 times) while the peak joint step exceeds ``carry_arm_step`` (0.06 rad per step =
the collector's 3 rad/s); the command limiter uses ``dq_carry`` / ``ddq_carry`` on the line. Decisions during the line
are ignored (also PLACE), except a[5] > ``carry_abort_thr``: a cosine brake of the target to rest, after which the next
decision may carry again (re-targeting only from rest). PLACE starts from rest as before. Counts: ``carries``,
``carry_aborts``.

Macro-duration seeds (review F5): ``reset`` without ``seed`` gives each world a fresh stream per episode (the first
episode of a world keeps the earlier default base_seed * 1,000,003 + world index); a trainer may pass its own episode
seeds, a paired evaluation passes the task seed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from nedm.so101_policy.kin import PAN_AXIS_XY, SO101Kin, quat_mat, rotvec, rotz, wrap, yaw_of

CONTROL_DT = 0.02
CUBE_HALF = 0.015
CUBE_REST_Z = 0.014993     # resting cube centre height in the data (m)
MODES = ("free", "g_brake", "g_descend", "g_settle", "g_close", "g_grip", "g_lift", "hold", "p_brake", "p_settle",
         "p_lower", "p_open", "p_retreat")
M = {n: k for k, n in enumerate(MODES)}
GRASP_PRE_CLOSE = (M["g_brake"], M["g_descend"], M["g_settle"])
# macro phases of the task rules (task.MACROS, same names and order) for each decoder mode
MACROS = ("free", "grasp_descend", "grasp_close", "grasp_grip", "grasp_lift", "place_lower", "place_open",
          "place_retreat")
MACRO_OF_MODE = ("free", "grasp_descend", "grasp_descend", "grasp_descend", "grasp_close", "grasp_grip", "grasp_lift",
                 "free", "place_lower", "place_lower", "place_lower", "place_open", "place_retreat")
ACTION_DIM = 6
FLAGS = ("guard_scaled", "limited", "lead_clipped", "floor_raised", "floor_override", "range_clipped", "ws_clipped",
         "yaw_blocked", "yaw_capped", "trigger_ignored", "aborted", "carries", "carry_aborts")


def minjerk(s):
    s = s.clamp(0.0, 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def expmap(v):
    """Rotation vectors [..., 3] -> rotation matrices (Rodrigues)."""
    th = v.norm(dim=-1, keepdim=True)[..., None]
    k = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-300)
    z = torch.zeros_like(k[..., 0])
    kx = torch.stack((z, -k[..., 2], k[..., 1], k[..., 2], z, -k[..., 0], -k[..., 1], k[..., 0], z), -1).reshape(*v.shape, 3)
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand(*v.shape[:-1], 3, 3)
    return eye + torch.sin(th) * kx + (1 - torch.cos(th)) * (kx @ kx)


def blend(v0, vn, tau, td):
    """Cosine velocity blend from v0 to vn over td (s), closed form (push-T so101_planar_decoder.blend, per-world td).
    v0, vn [B] or [B, 3]; tau, td [B] (s). -> (displacement since the blend start, velocity at tau)."""
    if v0.dim() == 2:
        tau, td = tau[:, None], td[:, None]
    tdp = td.clamp_min(1e-12)
    x = torch.minimum(tau, td)
    p = v0 * x + (vn - v0) * (x / 2 - tdp / (2 * math.pi) * torch.sin(math.pi * x / tdp)) + vn * (tau - x)
    v = torch.where(tau >= td, vn, v0 + (vn - v0) * (1 - torch.cos(math.pi * x / tdp)) / 2)
    return p, v


def hann_P(x):
    """Integral from 0 to x (in units of the blend length, x >= 0) of the Hann step S(x) = x - sin(2 pi x) / (2 pi)
    on [0, 1] (S = 1 after): the displacement of a unit Hann velocity increment of length 1 (option b1)."""
    xc = x.clamp(0.0, 1.0)
    return xc * xc / 2 + (torch.cos(2 * math.pi * xc) - 1) / (4 * math.pi * math.pi) + (x - 1).clamp_min(0)


def hann_R(x):
    """Integral from x to 1 of (1 - S): the displacement still missing (against a velocity step) of a unit Hann
    increment at clock x (units of its length; 0 for x >= 1)."""
    xc = x.clamp(0.0, 1.0)
    return 0.5 - xc + hann_P(xc)


def filter_step(x, xd, u, wn, dt=CONTROL_DT):
    """Critically damped 2nd-order filter x'' = wn^2 (u - x) - 2 wn x', input u held over one step of dt (exact).
    -> (x, x', integral of x over the step)."""
    E = math.exp(-wn * dt)
    e0 = x - u
    x1 = u + E * (e0 * (1 + wn * dt) + xd * dt)
    xd1 = E * (xd * (1 - wn * dt) - e0 * wn * wn * dt)
    disp = u * dt + e0 * (2 - E * (2 + wn * dt)) / wn + xd * (1 - E * (1 + wn * dt)) / (wn * wn)
    return x1, xd1, disp


_M32 = 0xFFFFFFFF
EPISODE_MIX = 0x5851F42D      # default macro-duration seed step per reset of a world (DecoderV2.reset, review F5)


def _mul32(x, c):
    """(x * c) mod 2^32 for int64 tensors x in [0, 2^32) and a constant c < 2^32, without int64 overflow."""
    return ((((x >> 16) * c) & 0xFFFF) * 65536 + (x & 0xFFFF) * c) & _M32


def hash_uniform(seed, counter, stream=0):
    """Counter-based uniform [0, 1) numbers (lowbias32 hash of seed, counter, stream; int64 tensors): the same values on
    every device and batch layout."""
    x = (seed & _M32) ^ ((counter * 0x9E3779B1 + stream * 0x85EBCA77 + 0x165667B1) & _M32)
    for _ in range(2):
        x = x ^ (x >> 16)
        x = _mul32(x, 0x7FEB352D)
        x = x ^ (x >> 15)
        x = _mul32(x, 0x846CA68B)
        x = x ^ (x >> 16)
    return x.to(torch.float64) / 4294967296.0


@dataclass
class DecoderV2Config:
    decision_steps: int = 5
    d_max: float = 0.035                  # m per decision (0.35 m/s)
    dyaw_max: float = 0.2                 # rad per decision (2 rad/s)
    a_max: float = 1.5                    # m/s^2, TCP target acceleration (free motion)
    alpha_max: float = 6.0                # rad/s^2, yaw acceleration (free motion)
    wall_margin: float = 0.5              # walls brake at this fraction of the estimated joint-limited deceleration
    dq_max: tuple = (0.0347, 0.0347, 0.0423, 0.0475, 0.0409)      # rad per 20 ms, data p99 (RF P0-2)
    ddq_max: tuple = (0.0028, 0.0069, 0.0087, 0.0095, 0.0034)     # rad per 20 ms^2, data p99
    # macro lines: the larger of ddq_max and the data's p99 in the collector phase the line copies (val_only task
    # families: descend, lift, lower, retreat; runs/kin_decoder_expert/v2/data_cmd_stats.json)
    ddq_line: dict = field(default_factory=lambda: dict(
        g_descend=(0.0028, 0.0071, 0.0101, 0.0112, 0.0034), g_lift=(0.0028, 0.0278, 0.0392, 0.0302, 0.0040),
        p_lower=(0.0028, 0.0070, 0.0087, 0.0095, 0.0034), p_retreat=(0.0028, 0.0311, 0.0460, 0.0351, 0.0053)))
    lead: tuple = (0.0445, 0.1235, 0.0552, 0.0559, 0.0476)        # rad, |cmd - q| data p99.9
    r_range: tuple = (0.12, 0.30)         # m, TCP radius about the pan axis (free motion)
    angle_max: float = math.radians(85)   # rad, |pan angle| of the TCP
    z_max: float = 0.15                   # m, TCP height in free motion (spec <= 0.20; above about 0.15 m the IK
                                          # needs tilts of 30-50 deg and tracks badly; starts above stay there)
    low_min: float = 0.0015               # m, lowest collision point of the IK solution
    roll_max: float = 2.5                 # rad, |yaw - pan angle| (wrist roll range)
    cube_bottom_max: float = 0.12         # m, lowest cube corner while held
    yaw_free_gap: float = 0.02            # m, yaw lock distance jaws-cube (free mode)
    cube_on_table: float = 0.005          # m, lowest cube corner below this = on the table (yaw lock)
    hold_turn_max: float = 0.6            # rad, gripper yaw change while holding
    jaw_open: float = 0.7
    jaw_closed: float = 0.0
    jaw_speed: float = 4.5                # rad/s, Plan.jaw
    jaw_min_s: float = 0.2                # s, jaw move after a reset
    jaw_step: float = 0.09                # rad per step
    lift_range: tuple = (0.03, 0.12)      # m, GRASP lift (a[5])
    v_macro: float = 0.35                 # m/s, peak speed of a macro line
    a_macro: float = 2.5                  # m/s^2, peak acceleration of a macro line
    line_margin: float = 1.1              # joint-feasible macro line duration x this
    t_descend: tuple = (0.25, 0.5)        # s, grasp_approach
    grasp_dz: tuple = (0.0, 0.002)        # m above the cube centre, fam_nominal
    t_settle: tuple = (0.0, 0.15)         # s, fam_nominal
    t_close: tuple = (0.2, 0.4)
    t_grip: tuple = (0.1, 0.25)
    t_lift: tuple = (0.35, 0.8)
    t_place_settle: tuple = (0.1, 0.3)
    t_lower: tuple = (0.35, 0.7)
    place_dz: tuple = (0.0, 0.003)        # m, lowest cube corner above the table
    t_open: tuple = (0.15, 0.35)
    retreat: tuple = (0.05, 0.10)         # m (collector 4-10 cm; >= 5 cm so the TCP ends >= 3 cm above the cube top)
    t_retreat: tuple = (0.25, 0.5)
    trigger_thr: float = 0.0
    abort_thr: float = 0.0
    blend: str = "cosine"                 # "cosine" (spec_v2 B1, push-T) | "filter" (ablation: the decision's velocity
    wn: float = 12.5                      #   is the set point of a critically damped 2nd-order filter, natural freq.)
                                          # | "hann" (option b1: overlap-added Hann velocity blends, 2 decisions long)
    tilt_s: float = 0.6
    ik_iters: int = 3
    # option b2: CARRY macro in mode hold (module docstring, "CARRY macro"); False = free motion while holding
    carry_macro: bool = False
    carry_reach: float = 0.32             # m, |a[0:2]| = 1 -> this horizontal displacement of the TCP target
    carry_t: tuple = (0.5, 0.9)           # s, collector carry_and_place move_s, then x 1.3 until the peak joint step
    carry_arm_step: float = 0.06          #   <= this (rad per 20 ms; the collector's Plan.cart, arm_max 3 rad/s)
    carry_min_d: float = 0.001            # m, a smaller carry displacement (and yaw change below carry_min_yaw) is no
    carry_min_yaw: float = math.radians(0.5)   # move (dead band; below the expert's 1.5 mm / 1 deg PLACE tolerance)
    carry_abort_thr: float = 0.5          # a[5] above this during a CARRY line brakes it to rest (cosine brake, as a
                                          # macro start); the next carry starts from there (re-target from rest)
    # command limiter on a carry line: steps <= the collector's arm_max step, step changes <= the larger of ddq_max and
    # the data's p99.9 in the collector's 'move' phase (val_only task families, 20 ms grid: d1 p99.9 0.058 / 0.047 /
    # 0.057 / 0.018 / 0.059 rad, max 0.060; d2 p99.9 below; ~/so101_policy/docs/build3_fixes.md)
    dq_carry: tuple = (0.06, 0.06, 0.06, 0.06, 0.06)
    ddq_carry: tuple = (0.0061, 0.0109, 0.0148, 0.0095, 0.0301)


class DecoderV2:
    """Batched decoder v2 (module docstring) for B worlds on one device. Per decision ``decide(a, state)``, then
    ``decision_steps`` x ``step(q_measured)`` -> (joint targets [B, 6] rad, info)."""

    def __init__(self, num_envs, device="cpu", kin: SO101Kin | None = None, cfg: DecoderV2Config | None = None,
                 seed: int = 0):
        self.cfg = c = cfg or DecoderV2Config()
        self.dev = torch.device(device)
        self.kin = kin or SO101Kin(self.dev)
        self.B = B = int(num_envs)
        self.base_seed = int(seed)
        f = dict(dtype=torch.float64, device=self.dev)
        z = lambda *s: torch.zeros(*s, **f)
        self.f = f
        self.T_d = c.decision_steps * CONTROL_DT
        self.dq_max, self.ddq_max = torch.tensor(c.dq_max, **f), torch.tensor(c.ddq_max, **f)
        self.ddq_mode = self.ddq_max.expand(len(MODES), 5).clone()                     # step-change bound per mode
        for name, v in c.ddq_line.items():
            self.ddq_mode[M[name]] = torch.tensor(v, **f)
        self.lead_band = torch.tensor(c.lead, **f)
        self.pan_axis = torch.tensor(PAN_AXIS_XY, **f)
        self.mode = torch.zeros(B, dtype=torch.long, device=self.dev)
        self.pos, self.yaw = z(B, 3), z(B)                    # Cartesian target: TCP (world, m), closing-axis yaw
        self.vel, self.yrate = z(B, 3), z(B)                  # last target step / dt (m/s, rad/s)
        self.v0, self.vn, self.w0, self.wn = z(B, 3), z(B, 3), z(B), z(B)   # active cosine blend
        self.tau, self.tb = z(B), torch.full((B,), self.T_d, **f)          # blend clock and length (s)
        self.fv, self.fa, self.fw, self.fwa = z(B, 3), z(B, 3), z(B), z(B)  # blend "filter": velocity, its rate
        if c.blend not in ("cosine", "filter", "hann"):
            raise ValueError(f"blend {c.blend!r}: 'cosine', 'filter' or 'hann'")
        self.filt, self.hann = c.blend == "filter", c.blend == "hann"
        # peak acceleration of a blend per unit velocity change (1/s): cosine pi / (2 T_d), filter wn / e, hann 1 / T_d
        # (a Hann blend of length 2 T_d peaks at 2 dv / (2 T_d); two overlapping ones peak at the larger dv / T_d)
        self.k_peak = c.wn / math.e if self.filt else (1 / self.T_d if self.hann else math.pi / (2 * self.T_d))
        self.dv_max, self.dw_max = c.a_max / self.k_peak, c.alpha_max / self.k_peak
        self.lp0, self.lp1, self.lk, self.ln = z(B, 3), z(B, 3), z(B), z(B)  # macro line / hold: ends, clock, length
        self.jaw_from, self.jaw_to, self.jaw_k, self.jaw_n = z(B), z(B), z(B), torch.ones(B, **f)
        self.q_des, self.q_cmd, self.dq = z(B, 5), z(B, 6), z(B, 5)      # IK solution, command, last arm step
        self.low_off = torch.full((B,), 0.0143, **f)          # TCP z minus lowest collision point (m), last solution
        self.z_goal, self.h = z(B), z(B)                      # macro line end height (m), GRASP lift (m)
        self.z_ceil = torch.full((B,), c.z_max, **f)          # TCP ceiling while holding (m)
        self.yaw_hold0 = z(B)
        self.goal = z(B, 4)                                   # GRASP descent end (x, y, z, yaw)
        self.tilt0, self.t = z(B, 3), z(B)
        self.seed = torch.arange(B, dtype=torch.long, device=self.dev)
        self.draws = torch.zeros(B, dtype=torch.long, device=self.dev)
        self.episode = torch.zeros(B, dtype=torch.long, device=self.dev)   # resets per world (default seed stream)
        # "hann" blend state: velocity of the finished blends, then two running blends A (older) and B (newest):
        # increment (m/s, rad/s), length T (s) and clock tau (s)
        self.hv, self.hw = z(B, 3), z(B)
        self.hA_d, self.hA_w, self.hA_T, self.hA_tau = z(B, 3), z(B), torch.ones(B, **f), torch.ones(B, **f)
        self.hB_d, self.hB_w, self.hB_T, self.hB_tau = z(B, 3), z(B), torch.ones(B, **f), torch.ones(B, **f)
        # CARRY macro (option b2): active flag and the line's yaw ends
        self.carry_on = torch.zeros(B, dtype=torch.bool, device=self.dev)
        self.carry_brk = torch.zeros(B, dtype=torch.bool, device=self.dev)   # an aborted CARRY brakes to rest
        self.ly0, self.ly1 = z(B), z(B)
        self.dq_carry, self.ddq_carry = torch.tensor(c.dq_carry, **f), torch.tensor(c.ddq_carry, **f)
        self.ik_err = z(B, 2)
        self.armed = torch.ones(B, dtype=torch.bool, device=self.dev)   # a trigger can start a macro
        self.counts = {k: torch.zeros(B, dtype=torch.long, device=self.dev) for k in FLAGS}
        self.theta_worst = self._worst_jaw()

    # ---- helpers ------------------------------------------------------------------------------------------------
    def _ids(self, idx):
        if idx is None:
            return torch.arange(self.B, device=self.dev)
        idx = torch.as_tensor(idx, device=self.dev)
        return idx.nonzero().flatten() if idx.dtype == torch.bool else idx.long().flatten()

    def _worst_jaw(self):
        """Jaw angle (rad, within the target range up to 0.6) with the deepest moving jaw below the TCP, top-down."""
        th = torch.linspace(float(self.kin.tlo[5]), 0.6, 241, **self.f)
        q, _, _, _ = self.kin.ik_topdown(None, torch.tensor([[0.25, 0.0, 0.05]], **self.f).expand(len(th), 3),
                                         torch.zeros(len(th), **self.f))
        low = self.kin.lowest_point(torch.cat((q, th[:, None]), -1), per_body=True)[:, 1]
        return float(th[low.argmin()])

    def _draw(self, mask, rng):
        u = hash_uniform(self.seed, self.draws)
        self.draws = self.draws + mask.long()
        return rng[0] + (rng[1] - rng[0]) * u

    def _line_steps(self, t_draw, end, mode, samples=8, yaw_end=None, dq=None, ddq=None):
        """Steps of a minimum-jerk TCP line from the target to end [B, 3]: the drawn duration, raised to keep the peak
        TCP speed <= v_macro and acceleration <= a_macro, and every joint's peak step <= dq_max and peak step change
        <= ddq_max. The joint path q(u) comes from warm-started IK at ``samples`` points of the line; with u(t) minimum
        jerk (|u'| <= 1.875 / T, |u''| <= 5.77 / T^2): T >= 1.875 max|q'| / dq_max and
        T^2 >= (3.52 max|q''| + 5.77 max|q'|) / ddq_max (T in steps), then x line_margin. yaw_end [B]: the yaw turns
        with the same timing (CARRY line); dq, ddq [5]: other joint bounds (default dq_max and the mode's bound)."""
        c = self.cfg
        dist = (end - self.pos).norm(dim=-1)
        t = torch.maximum(t_draw, torch.maximum(1.875 * dist / c.v_macro, torch.sqrt(5.7735 * dist / c.a_macro)))
        rot = self._rot(self.yaw)
        qs, q = [self.q_des], self.q_des
        for k in range(1, samples + 1):
            if yaw_end is not None:
                rot = self._rot(self.yaw + (yaw_end - self.yaw) * (k / samples))
            q, _, _ = self.kin.ik_track(q, self.pos + (end - self.pos) * (k / samples), rot, 2 * c.ik_iters)
            qs.append(q)
        qs = torch.stack(qs, 1)                                                           # [B, K + 1, 5]
        d1 = qs.diff(dim=1).abs().amax(1) * samples                                       # max |dq/du|
        d2 = qs.diff(n=2, dim=1).abs().amax(1) * samples * samples                        # max |d2q/du2|
        A = self.ddq_mode[M[mode]] if ddq is None else ddq
        dqm = self.dq_max if dq is None else dq
        steps = torch.maximum(1.875 * d1 / dqm, torch.sqrt((3.516 * d2 + 5.7735 * d1) / A)).amax(-1)
        t = torch.maximum(t, c.line_margin * steps * CONTROL_DT)
        return torch.round(t / CONTROL_DT).clamp_min(1)

    def _jaw_move(self, mask, goal, t_draw):
        jaw = self.q_cmd[:, 5]
        n = torch.round(torch.maximum(t_draw, 1.875 * (goal - jaw).abs() / self.cfg.jaw_speed) / CONTROL_DT).clamp_min(1)
        self.jaw_from = torch.where(mask, jaw, self.jaw_from)
        self.jaw_to = torch.where(mask, goal, self.jaw_to)
        self.jaw_k = torch.where(mask, torch.zeros_like(jaw), self.jaw_k)
        self.jaw_n = torch.where(mask, n, self.jaw_n)

    def _rot(self, yaw):
        lam = 1 - minjerk((self.t + 1) * CONTROL_DT / self.cfg.tilt_s)
        return rotz(yaw) @ expmap(self.tilt0 * lam[:, None])

    def _jaw_check(self, q_meas):
        """Jaw angle for the floor: the angle in the jaw's range of this step nearest to theta_worst."""
        j = [self.q_cmd[:, 5], self.jaw_to]
        if q_meas is not None:
            j.append(q_meas[:, 5].to(self.pos))
        j = torch.stack(j, -1)
        return torch.minimum(torch.maximum(torch.full_like(j[:, 0], self.theta_worst), j.amin(-1)), j.amax(-1))

    def _low_offset(self, q5, jaw):
        """TCP z minus the lowest collision point (m) at arm angles q5 [B, 5] and jaw [B]."""
        q = torch.cat((q5, jaw[:, None]), -1)
        return self.kin.fk(q)[0][:, 2] - self.kin.lowest_point(q, exact=False)

    def _polar(self, p):
        rel = p[:, :2] - self.pan_axis
        r = rel.norm(dim=-1)
        return r, torch.atan2(rel[:, 1], rel[:, 0]), rel / r.clamp_min(1e-9)[:, None]

    def _allow(self, dist, v_wall, acc):
        """Largest speed toward a wall at distance dist (>= 0) that a decision may end with and still stop before it
        (cosine blend from v_wall over T_d, then braking at the average deceleration acc [B]). Blend "hann": v_wall =
        the current set point, and the blend counts 2 T_d (its response lags the cosine blend by about T_d / 2 on
        the way up and on the way down)."""
        td = 2 * self.T_d if self.hann else self.T_d
        disc = td * td / 4 - 2 * (v_wall * td / 2 - dist.clamp_min(0)) / acc
        return acc * (-td / 2 + torch.sqrt(disc.clamp_min(0)))

    # ---- "hann" blend (option b1) ----------------------------------------------------------------------------------
    def setpoint(self):
        """(velocity [B, 3], yaw rate [B]) the free-motion target is heading to: the last decision's set point (blend
        "hann": the sum of all its increments; "cosine": the blend's end velocity; "filter": its set point)."""
        if self.hann:
            return self.hv + self.hA_d + self.hB_d, self.hw + self.hA_w + self.hB_w
        return self.vn.clone(), self.wn.clone()

    def _hann_missing(self):
        """Displacement (m [B, 3], rad [B]) the running Hann blends still owe against a velocity step to their
        increments: sum of d T R(tau / T)."""
        rA = self.hA_T * hann_R(self.hA_tau / self.hA_T)
        rB = self.hB_T * hann_R(self.hB_tau / self.hB_T)
        return self.hA_d * rA[:, None] + self.hB_d * rB[:, None], self.hA_w * rA + self.hB_w * rB

    def rest_terms(self):
        """Free motion: where the target comes to rest if this decision sets the velocity v (and yaw rate w) and the
        next decision sets 0: pos + alpha v + beta, yaw + alpha w + beta_yaw -> (alpha (s), beta [B, 3], beta_yaw [B]).
        Cosine: alpha = T_d, beta = v0 T_d / 2 (v0 = the last step's velocity); hann: alpha = T_d, beta = u T_d - the
        running blends' missing displacement (u = the current set point). Not for blend "filter" (the expert solves
        it numerically)."""
        if self.hann:
            u, w = self.setpoint()
            mp, mw = self._hann_missing()
            return self.T_d, u * self.T_d - mp, w * self.T_d - mw
        return self.T_d, self.vel * self.T_d / 2, self.yrate * self.T_d / 2

    def _hann_zero(self, mask, v=None, w=None):
        """Hann blend state of the worlds in mask at a constant velocity v [B, 3] / yaw rate w [B] (default rest)."""
        m3 = mask[:, None]
        self.hv = torch.where(m3, torch.zeros_like(self.hv) if v is None else v, self.hv)
        self.hw = torch.where(mask, torch.zeros_like(self.hw) if w is None else w, self.hw)
        for x in (self.hA_d, self.hB_d):
            x.copy_(torch.where(m3, 0.0, x))
        for x in (self.hA_w, self.hB_w):
            x.copy_(torch.where(mask, 0.0, x))
        for x in (self.hA_tau, self.hA_T, self.hB_tau, self.hB_T):
            x.copy_(torch.where(mask, 1.0, x))

    def _hann_done(self):
        """[B] bool: no Hann blend is running."""
        return (self.hA_tau >= self.hA_T - 1e-9) & (self.hB_tau >= self.hB_T - 1e-9)

    def _hann_shift(self, mask, d, w, T):
        """Decision of the worlds in mask: the finished blend A folds into the base velocity, B becomes A, and a new
        blend B starts with increment d [B, 3] / w [B] and length T [B] (s)."""
        m3 = mask[:, None]
        # with decisions every T_d (the API) blend A started 2 T_d ago and has finished; were it not (decide called at
        # other times), folding it in whole would step the velocity by its missing part
        self.hv = torch.where(m3, self.hv + self.hA_d, self.hv)
        self.hw = torch.where(mask, self.hw + self.hA_w, self.hw)
        self.hA_d, self.hA_w = torch.where(m3, self.hB_d, self.hA_d), torch.where(mask, self.hB_w, self.hA_w)
        self.hA_T, self.hA_tau = torch.where(mask, self.hB_T, self.hA_T), torch.where(mask, self.hB_tau, self.hA_tau)
        self.hB_d, self.hB_w = torch.where(m3, d, self.hB_d), torch.where(mask, w, self.hB_w)
        self.hB_T = torch.where(mask, T, self.hB_T)
        self.hB_tau = torch.where(mask, torch.zeros_like(self.hB_tau), self.hB_tau)

    def _hann_step(self):
        """Target displacement (m [B, 3], rad [B]) of the Hann blends over one control step (closed form)."""
        dt = CONTROL_DT
        kA = self.hA_T * (hann_P((self.hA_tau + dt) / self.hA_T) - hann_P(self.hA_tau / self.hA_T))
        kB = self.hB_T * (hann_P((self.hB_tau + dt) / self.hB_T) - hann_P(self.hB_tau / self.hB_T))
        return (self.hv * dt + self.hA_d * kA[:, None] + self.hB_d * kB[:, None],
                self.hw * dt + self.hA_w * kA + self.hB_w * kB)

    # ---- CARRY macro (option b2) ---------------------------------------------------------------------------------
    def carry_target(self, a):
        """CARRY end of action a [B, 6] from the current target (module docstring): TCP xy + a[0:2] carry_reach (norm
        clipped), same height; radius and pan-angle bounds as free motion; the straight line shortened where it would
        pass inside the inner radius; yaw + a[3] hold_turn_max within the hold turn cap and the wrist roll range.
        -> (end [B, 3], yaw [B])."""
        c = self.cfg
        dxy = a[:, 0:2] * c.carry_reach
        n = dxy.norm(dim=-1, keepdim=True)
        dxy = dxy * torch.where(n > c.carry_reach, c.carry_reach / n.clamp_min(1e-12), torch.ones_like(n))
        r_lo, r_hi, a_lo, a_hi = self._bounds()
        r, ang, _ = self._polar(torch.cat((self.pos[:, :2] + dxy, self.pos[:, 2:]), -1))
        r = torch.minimum(torch.maximum(r, r_lo), r_hi)
        ang = torch.minimum(torch.maximum(ang, a_lo), a_hi)
        p = self.pos[:, :2] - self.pan_axis                                           # line start about the pan axis
        d = torch.stack((torch.cos(ang), torch.sin(ang)), -1) * r[:, None] - p
        dd, pd = (d * d).sum(-1), (p * d).sum(-1)
        disc = pd * pd - dd * ((p * p).sum(-1) - r_lo * r_lo)
        t1 = (-pd - torch.sqrt(disc.clamp_min(0))) / dd.clamp_min(1e-18)             # first crossing of the r_lo circle
        cut = (disc > 0) & (t1 >= 0) & (t1 < 1) & (dd > 1e-18)
        d = d * torch.where(cut, t1, torch.ones_like(t1))[:, None]
        end = torch.cat((self.pos[:, :2] + d, self.pos[:, 2:]), -1)
        _, ang_e, _ = self._polar(end)
        y1 = self.yaw + a[:, 3] * c.hold_turn_max
        y1 = torch.minimum(torch.maximum(y1, self.yaw_hold0 - c.hold_turn_max), self.yaw_hold0 + c.hold_turn_max)
        rho = wrap(y1 - ang_e)
        y1 = y1 - (rho - rho.clamp(-c.roll_max, c.roll_max))
        return end, y1

    def _carry_steps(self, t_draw, end, yaw_end, samples=16):
        """Steps of a CARRY line (the collector's Plan.cart timing): the drawn duration, x 1.3 (up to 6 times) while the
        largest joint step of the minimum-jerk line exceeds ``carry_arm_step``. The joint path comes from warm-started
        IK at ``samples`` points (the yaw turning along); the largest step of a line of n steps is
        max over the time fraction s of |dq/du|(u(s)) 30 s^2 (1 - s)^2 / n (chord slopes, the collector's actual
        largest step rather than the bound 1.875 max|dq/du| / n)."""
        c = self.cfg
        qs, q = [self.q_des], self.q_des
        for k in range(1, samples + 1):
            rot = self._rot(self.yaw + (yaw_end - self.yaw) * (k / samples))
            q, _, _ = self.kin.ik_track(q, self.pos + (end - self.pos) * (k / samples), rot, 2 * c.ik_iters)
            qs.append(q)
        slope = (torch.stack(qs, 1).diff(dim=1).abs() * samples).amax(-1)            # [B, K] max over joints, per chord
        sg = torch.linspace(0, 1, 201, **self.f)
        chord = (minjerk(sg) * samples).long().clamp(0, samples - 1)                  # chord of u(s)
        peak = (slope[:, chord] * (30 * sg * sg * (1 - sg) ** 2)).amax(-1)             # max |dq/ds| [B]
        n = torch.round(t_draw / CONTROL_DT).clamp_min(1)
        for _ in range(6):
            n = torch.where(peak / n > c.carry_arm_step, torch.round(n * 1.3), n)
        return n

    def _carry_abort(self, mask):
        """Stop the CARRY lines of the worlds in mask: a cosine brake of the target velocity of the last step to 0 (the
        free-motion brake rule: max(T_d, pi |v| / (2 a_max), pi |w| / (2 alpha_max)), rounded up to the step), then
        rest in mode hold. -> mask."""
        if not bool(mask.any()):
            return mask
        c = self.cfg
        sp, wr = self.vel.norm(dim=-1), self.yrate.abs()
        tbk = torch.maximum(torch.maximum(torch.full_like(sp, self.T_d), math.pi * sp / (2 * c.a_max)),
                            math.pi * wr / (2 * c.alpha_max))
        tbk = torch.ceil(tbk / CONTROL_DT - 1e-9) * CONTROL_DT
        m3 = mask[:, None]
        self.v0, self.vn = torch.where(m3, self.vel, self.v0), torch.where(m3, 0.0, self.vn)
        self.w0, self.wn = torch.where(mask, self.yrate, self.w0), torch.where(mask, 0.0, self.wn)
        self.tau, self.tb = torch.where(mask, 0.0, self.tau), torch.where(mask, tbk, self.tb)
        self._hann_zero(mask)                             # (a Hann world brakes with the cosine rule here)
        self.carry_on = self.carry_on & ~mask
        self.carry_brk = self.carry_brk | mask
        self.counts["carry_aborts"] += mask.long()
        return mask

    def _carry_start(self, a, mask):
        """Start CARRY lines in the worlds of mask whose target differs from the current one by more than the dead
        band. -> started [B] bool."""
        c = self.cfg
        end, y1 = self.carry_target(a)
        start = mask & (((end - self.pos)[:, :2].norm(dim=-1) >= c.carry_min_d)
                        | ((y1 - self.yaw).abs() >= c.carry_min_yaw))
        if not bool(start.any()):
            return start
        ln = self._carry_steps(self._draw(start, c.carry_t), end, y1)
        m3 = start[:, None]
        self.lp0, self.lp1 = torch.where(m3, self.pos, self.lp0), torch.where(m3, end, self.lp1)
        self.ly0, self.ly1 = torch.where(start, self.yaw, self.ly0), torch.where(start, y1, self.ly1)
        self.ln = torch.where(start, ln, self.ln)
        self.lk = torch.where(start, torch.zeros_like(self.lk), self.lk)
        self.carry_on = self.carry_on | start
        self.counts["carries"] += start.long()
        return start

    def _ceiling(self):
        """TCP ceiling (m) [B]: z_max, in mode hold also the held-cube ceiling; never below the current target (a
        start above it can only move down: no jump)."""
        top = torch.where(self.mode == M["hold"], torch.minimum(self.z_ceil, torch.full_like(self.z_ceil, self.cfg.z_max)),
                          torch.full_like(self.z_ceil, self.cfg.z_max))
        return torch.maximum(top, self.pos[:, 2])

    def _bounds(self):
        """Free-motion radius and pan-angle bounds [B], relaxed to the current target (push-T rule: a target outside
        never jumps, it only cannot move farther out)."""
        c = self.cfg
        r, ang, _ = self._polar(self.pos)
        return (torch.clamp_max(r, c.r_range[0]), torch.clamp_min(r, c.r_range[1]),
                torch.clamp_max(ang, -c.angle_max), torch.clamp_min(ang, c.angle_max))

    def _rate_map(self):
        """Damped least-squares map of a target velocity (v [3] m/s, yaw rate w rad/s) to joint rates at the last IK
        solution, with the IK's weight 0.1 m/rad on the rotation: [B, 5, 4] (columns vx, vy, vz, w)."""
        jp, jr, _, _ = self.kin.jacobian(self.q_des)
        jac = torch.cat((jp, 0.1 * jr), -2)                                              # [B, 6, 5]
        jt = jac.transpose(-1, -2)
        eye = torch.eye(5, **self.f)
        pinv = torch.linalg.solve(jt @ jac + 1e-4 * eye, jt)                           # [B, 5, 6]
        return torch.cat((pinv[:, :, :3], 0.1 * pinv[:, :, 5:6]), -1)

    def _acc_along(self, rmap, n):
        """Average deceleration (m/s^2 or rad/s^2) of a cosine blend along task direction n [B, 4] that keeps every
        joint's peak step change <= ddq_max (and <= the Cartesian a_max / alpha_max)."""
        qn = (rmap @ n[..., None])[..., 0].abs()                                         # joint rate per unit rate
        k = 1 / (self.k_peak * CONTROL_DT * CONTROL_DT)                                  # peak accel -> blend dv
        dv = (k * self.ddq_max / qn.clamp_min(1e-9)).amin(-1)
        lin = n[:, :3].norm(dim=-1) > 0.5
        cap = torch.where(lin, self.dv_max, self.dw_max)
        return self.cfg.wall_margin * torch.minimum(dv, cap) / self.T_d

    def _walls(self, vn, v0, rmap):
        """Cap the velocity toward each workspace wall (module docstring). -> (vn, capped [B])."""
        c = self.cfg
        p = self.pos
        r, ang, er = self._polar(p)
        zero = torch.zeros_like(r)
        et = torch.stack((-er[:, 1], er[:, 0], zero), -1)
        er3 = torch.cat((er, zero[:, None]), -1)
        ez = torch.zeros_like(p)
        ez[:, 2] = 1.0
        floor = c.low_min + self.low_off
        r_lo, r_hi, a_lo, a_hi = self._bounds()
        walls = ((-ez, p[:, 2] - floor), (ez, self._ceiling() - p[:, 2]), (-er3, r - r_lo),
                 (er3, r_hi - r), (et, r * (a_hi - ang)), (-et, r * (ang - a_lo)))
        capped = torch.zeros_like(r, dtype=torch.bool)
        for n, dist in walls:
            acc = self._acc_along(rmap, torch.cat((n, zero[:, None]), -1))
            allow = self._allow(dist, (v0 * n).sum(-1), acc)
            over = ((vn * n).sum(-1) - allow).clamp_min(0)
            capped |= over > 1e-12
            vn = vn - over[:, None] * n
        return vn, capped

    def _yaw_walls(self, wn, w0, rmap):
        c = self.cfg
        hold = self.mode == M["hold"]
        _, ang, _ = self._polar(self.pos)
        rho = wrap(self.yaw - ang)
        big = torch.full_like(wn, 1e9)
        ew = torch.zeros(self.B, 4, **self.f)
        ew[:, 3] = 1.0
        acc = self._acc_along(rmap, ew)
        walls = ((1.0, torch.where(hold, self.yaw_hold0 + c.hold_turn_max - self.yaw, big)),
                 (-1.0, torch.where(hold, self.yaw - (self.yaw_hold0 - c.hold_turn_max), big)),
                 (1.0, c.roll_max - rho), (-1.0, c.roll_max + rho))
        capped = torch.zeros_like(wn, dtype=torch.bool)
        for sgn, dist in walls:
            allow = self._allow(dist, sgn * w0, acc)
            over = (sgn * wn - allow).clamp_min(0)
            capped |= over > 1e-12
            wn = wn - sgn * over
        return wn, capped & hold

    def _joint_scale(self, rmap, v0, vn, w0, wn):
        """Largest lam in [0, 1] for the velocity change (v0, w0) -> (vn, wn) such that the blend keeps every joint's
        step change <= ddq_max and its step <= dq_max (Jacobian estimate at the decision)."""
        qd0 = (rmap @ torch.cat((v0, w0[:, None]), -1)[..., None])[..., 0]               # rad/s
        dqd = (rmap @ torch.cat((vn - v0, (wn - w0)[:, None]), -1)[..., None])[..., 0]
        dt = CONTROL_DT
        lam_acc = (self.ddq_max / (self.k_peak * dt * dt) / dqd.abs().clamp_min(1e-12)).amin(-1)
        room = self.dq_max / dt - torch.sign(dqd) * qd0                                   # rate left in the move direction
        lam_spd = torch.where(dqd.abs() > 1e-12, room.clamp_min(0) / dqd.abs().clamp_min(1e-12),
                              torch.full_like(room, 1e9)).amin(-1)
        return torch.minimum(torch.minimum(lam_acc, lam_spd), torch.ones_like(lam_acc)).clamp_min(0)

    def _clip(self, pos, yaw, apply):
        """Hard workspace clip of a free-motion target (the walls of ``_walls``). -> (pos, yaw, clipped [B])."""
        c = self.cfg
        floor = c.low_min + self.low_off
        z = torch.minimum(torch.maximum(pos[:, 2], floor), self._ceiling())
        r, ang, _ = self._polar(pos)
        r_lo, r_hi, a_lo, a_hi = self._bounds()
        r2, ang2 = torch.minimum(torch.maximum(r, r_lo), r_hi), torch.minimum(torch.maximum(ang, a_lo), a_hi)
        xy = self.pan_axis + torch.stack((torch.cos(ang2), torch.sin(ang2)), -1) * r2[:, None]
        p2 = torch.cat((xy, z[:, None]), -1)
        rho = wrap(yaw - ang2)
        y2 = yaw - (rho - rho.clamp(-c.roll_max, c.roll_max))
        hold = self.mode == M["hold"]
        y2 = torch.where(hold, torch.minimum(torch.maximum(y2, self.yaw_hold0 - c.hold_turn_max),
                                             self.yaw_hold0 + c.hold_turn_max), y2)
        clipped = apply & (((p2 - pos).abs().amax(-1) > 1e-12) | ((y2 - yaw).abs() > 1e-12))
        return torch.where(apply[:, None], p2, pos), torch.where(apply, y2, yaw), clipped

    # ---- reset --------------------------------------------------------------------------------------------------
    def reset(self, idx, q, hold=None, seed=None, z_ceil=None):
        """Worlds idx (index tensor or bool mask) start at rest at the measured joint angles q [n, 6]. hold [n] bool:
        start in mode hold (cube in the jaws; z_ceil [n] = TCP ceiling, default z_max), else free (the jaw opens to
        jaw_open). seed [n] int64: macro-duration seeds; the same seed gives the same durations (paired evaluations:
        pass the task seed). Default (review F5): a fresh stream per episode, base_seed * 1,000,003 + world index +
        EPISODE_MIX * (number of earlier resets of that world), so the first episode of a world keeps the old default;
        a trainer may also pass its own episode seed (e.g. episode counter * B + slot)."""
        c = self.cfg
        idx = self._ids(idx)
        n = len(idx)
        q = torch.minimum(torch.maximum(q.to(**self.f), self.kin.tlo), self.kin.thi)
        hold = torch.zeros(n, dtype=torch.bool, device=self.dev) if hold is None else torch.as_tensor(hold, device=self.dev)
        p, r = self.kin.fk(q)
        yaw = yaw_of(r)
        self.mode[idx] = torch.where(hold, M["hold"], M["free"])
        self.pos[idx], self.yaw[idx] = p, yaw
        for x in (self.vel, self.yrate, self.v0, self.vn, self.w0, self.wn, self.tau, self.lk, self.ln, self.dq,
                  self.fv, self.fa, self.fw, self.fwa,
                  self.t, self.ik_err, self.h, self.z_goal, self.goal):
            x[idx] = 0.0
        self.tb[idx] = self.T_d
        self.lp0[idx], self.lp1[idx] = p, p
        self.q_des[idx], self.q_cmd[idx] = q[:, :5], q
        self.tilt0[idx] = rotvec(rotz(yaw).transpose(-1, -2) @ r)
        self.yaw_hold0[idx] = yaw
        self.z_ceil[idx] = c.z_max if z_ceil is None else torch.as_tensor(z_ceil, **self.f)
        goal = torch.where(hold, torch.full((n,), c.jaw_closed, **self.f), torch.full((n,), c.jaw_open, **self.f))
        jaw = q[:, 5]
        self.jaw_from[idx], self.jaw_to[idx], self.jaw_k[idx] = jaw, goal, 0.0
        self.jaw_n[idx] = torch.round(torch.clamp_min(1.875 * (goal - jaw).abs() / c.jaw_speed, c.jaw_min_s)
                                      / CONTROL_DT).clamp_min(1)
        jc = torch.minimum(torch.maximum(torch.full_like(jaw, self.theta_worst), torch.minimum(jaw, goal)),
                           torch.maximum(jaw, goal))
        self.low_off[idx] = self._low_offset(q[:, :5], jc)
        self.seed[idx] = (torch.as_tensor(seed, dtype=torch.long, device=self.dev).clone() if seed is not None
                          else self.base_seed * 1000003 + idx + EPISODE_MIX * self.episode[idx])
        self.episode[idx] += 1
        self.draws[idx] = 0
        self.armed[idx] = True
        sel = torch.zeros(self.B, dtype=torch.bool, device=self.dev)
        sel[idx] = True
        self._hann_zero(sel)
        self.carry_on[idx] = False
        self.carry_brk[idx] = False
        self.ly0[idx], self.ly1[idx] = yaw, yaw
        for v in self.counts.values():
            v[idx] = 0

    # ---- one decision -------------------------------------------------------------------------------------------
    @torch.no_grad()
    def decide(self, action, state=None):
        """action [B, 6] in [-1, 1] (module docstring); state: the WorldState of the decision (q, cube_pos, cube_quat
        used; None: no yaw lock, GRASP height from the resting cube height). -> info dict of [B] bools."""
        c = self.cfg
        a = action.to(**self.f).clamp(-1.0, 1.0)
        mode = self.mode
        free, hold = mode == M["free"], mode == M["hold"]
        pre = (mode == M["g_brake"]) | (mode == M["g_descend"]) | (mode == M["g_settle"])
        abort = pre & (a[:, 4] <= c.abort_thr)
        free = free | abort
        self.mode = torch.where(abort, M["free"], mode)
        if self.filt:
            self.fv = torch.where(abort[:, None], self.vel, self.fv)
            self.fa = torch.where(abort[:, None], torch.zeros_like(self.fa), self.fa)
            self.fw = torch.where(abort, self.yrate, self.fw)
            self.fwa = torch.where(abort, torch.zeros_like(self.fwa), self.fwa)
        if self.hann:
            self._hann_zero(abort, self.vel, self.yrate)
        moving = hold & (self.carry_on | self.carry_brk)  # a CARRY line or its abort brake runs: decision ignored
        take = free | (hold & ~moving) if not c.carry_macro else free   # free-motion decisions
        v0, w0 = self.setpoint() if self.hann else (self.vel, self.yrate)
        d = a[:, :3] * c.d_max
        dn = d.norm(dim=-1, keepdim=True)
        d = d * torch.where(dn > c.d_max, c.d_max / dn.clamp_min(1e-12), torch.ones_like(dn))
        dv = d / self.T_d - v0
        dvn = dv.norm(dim=-1, keepdim=True)
        vn = v0 + dv * torch.where(dvn > self.dv_max, self.dv_max / dvn.clamp_min(1e-12), torch.ones_like(dvn))
        wn = w0 + (a[:, 3] * c.dyaw_max / self.T_d - w0).clamp(-self.dw_max, self.dw_max)
        rmap = self._rate_map()
        lam = self._joint_scale(rmap, v0, vn, w0, wn)
        vn = v0 + lam[:, None] * (vn - v0)
        wn = w0 + lam * (wn - w0)
        vn, ws_capped = self._walls(vn, v0, rmap)
        blocked = torch.zeros_like(free)
        cube_z = torch.full_like(self.yaw, CUBE_REST_Z)
        bottom = torch.zeros_like(self.yaw)
        if state is not None:
            cube_p = state.cube_pos.to(**self.f)
            rc = quat_mat(state.cube_quat.to(**self.f))
            cube_z = cube_p[:, 2]
            bottom = cube_z - CUBE_HALF * rc[:, 2, :].abs().sum(-1)                   # lowest cube corner (m)
            near = free & (self.q_cmd[:, 5] > 0.3) & (bottom < c.cube_on_table)
            if bool(near.any()):
                dist = self.kin.box_distance(self.q_cmd, cube_p, state.cube_quat, CUBE_HALF)
                blocked = near & (dist < c.yaw_free_gap + v0.norm(dim=-1) * self.T_d)
        wn = torch.where(blocked, torch.zeros_like(wn), wn)
        wn, yaw_capped = self._yaw_walls(wn, w0, rmap)
        trig = (a[:, 4] > c.trigger_thr) & self.armed
        self.armed = self.armed | (a[:, 4] <= c.trigger_thr)
        g_trig = free & ~abort & trig
        jaw_ready = (self.jaw_k >= self.jaw_n) & ((self.jaw_to - c.jaw_open).abs() < 1e-9)
        ignored = g_trig & ~jaw_ready
        g_trig = g_trig & jaw_ready
        p_trig = hold & ~moving & trig
        self.armed = self.armed & ~(g_trig | p_trig)
        go = take & ~g_trig & ~p_trig
        # free-motion decisions
        self.v0 = torch.where(take[:, None], v0, self.v0)
        self.w0 = torch.where(take, w0, self.w0)
        self.vn = torch.where(go[:, None], vn, torch.where((g_trig | p_trig)[:, None], torch.zeros_like(vn), self.vn))
        self.wn = torch.where(go, wn, torch.where(g_trig | p_trig, torch.zeros_like(wn), self.wn))
        self.tau = torch.where(take, torch.zeros_like(self.tau), self.tau)
        # macro starts: brake to rest first (skipped at rest)
        brake = g_trig | p_trig
        sp, wr = v0.norm(dim=-1), w0.abs()
        qd0 = (rmap @ torch.cat((v0, w0[:, None]), -1)[..., None])[..., 0].abs()
        tj = (math.pi * qd0 * CONTROL_DT * CONTROL_DT / (2 * self.ddq_max)).amax(-1)       # joint step-change bound
        tbk = torch.maximum(torch.maximum(torch.full_like(sp, self.T_d), math.pi * sp / (2 * c.a_max)),
                            torch.maximum(math.pi * wr / (2 * c.alpha_max), tj))
        tbk = torch.ceil(tbk / CONTROL_DT - 1e-9) * CONTROL_DT
        rest = (sp < 1e-9) & (wr < 1e-9)
        self.tb = torch.where(brake, torch.where(rest, torch.zeros_like(tbk), tbk), torch.where(take, self.T_d, self.tb))
        stop = self.pos + v0 * (self.tb / 2)[:, None]
        ystop = self.yaw + w0 * self.tb / 2
        if self.filt:                                     # where the filter comes to rest with the set point 0
            stop = self.pos + 2 * self.fv / c.wn + self.fa / (c.wn * c.wn)
            ystop = self.yaw + 2 * self.fw / c.wn + self.fwa / (c.wn * c.wn)
        if self.hann:   # new blend B per decision: the set point change over 2 T_d; a brake: to 0 over T_bk (2 dv / T)
            tjh = (2 * qd0 * CONTROL_DT * CONTROL_DT / self.ddq_max).amax(-1)
            tbh = torch.maximum(torch.maximum(torch.full_like(sp, 2 * self.T_d), 2 * sp / c.a_max),
                                torch.maximum(2 * wr / c.alpha_max, tjh))
            tbh = torch.ceil(tbh / CONTROL_DT - 1e-9) * CONTROL_DT
            tbh = torch.where(brake & rest & self._hann_done(), torch.full_like(tbh, 1e-9), tbh)
            vgo = torch.where(go[:, None], vn, torch.zeros_like(vn))
            wgo = torch.where(go, wn, torch.zeros_like(wn))
            self._hann_shift(take, vgo - v0, wgo - w0, torch.where(brake, tbh, torch.full_like(tbh, 2 * self.T_d)))
            mp, mw = self._hann_missing()
            stop, ystop = self.pos - mp, self.yaw - mw    # rest point of the brake: sum of d T R(tau / T)
        # GRASP: descent end and the hold ceiling
        if bool(g_trig.any()):
            q_w = torch.cat((self.q_des, torch.full_like(self.yaw[:, None], self.theta_worst)), -1)
            off = self.kin.fk(q_w)[0][:, 2] - self.kin.lowest_point(q_w, exact=False)
            zg = torch.maximum(cube_z + self._draw(g_trig, c.grasp_dz), c.low_min + off)
            self.z_goal = torch.where(g_trig, zg, self.z_goal)
            self.h = torch.where(g_trig, c.lift_range[0] + (a[:, 5] + 1) / 2 * (c.lift_range[1] - c.lift_range[0]), self.h)
            ceil = torch.clamp_max(zg + c.cube_bottom_max - bottom.clamp_min(0), c.z_max)
            self.z_ceil = torch.where(g_trig, ceil, self.z_ceil)
            self.goal = torch.where(g_trig[:, None], torch.stack((stop[:, 0], stop[:, 1], zg, ystop), -1), self.goal)
        if bool(p_trig.any()):
            zp = self.pos[:, 2] - bottom + self._draw(p_trig, c.place_dz)
            self.z_goal = torch.where(p_trig, zp, self.z_goal)
        self.mode = torch.where(g_trig, M["g_brake"], torch.where(p_trig, M["p_brake"], self.mode))
        carry_started = torch.zeros_like(p_trig)
        carry_aborted = torch.zeros_like(p_trig)
        if c.carry_macro:   # option b2: a hold decision at rest gives the CARRY target; a[5] aborts a running line
            carry_aborted = self._carry_abort(hold & self.carry_on & (a[:, 5] > c.carry_abort_thr))
            carry_started = self._carry_start(a, hold & ~moving & ~p_trig)
        info = dict(grasp_started=g_trig, place_started=p_trig, aborted=abort, trigger_ignored=ignored,
                    yaw_blocked=blocked & take, yaw_capped=yaw_capped & take, ws_capped=ws_capped & go,
                    carry_started=carry_started, carry_aborted=carry_aborted)
        for k in ("aborted", "trigger_ignored", "yaw_blocked", "yaw_capped"):
            self.counts[k] += info[k].long()
        return info

    # ---- one control step ---------------------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, q_meas=None):
        """One 20 ms step. q_meas [B, 6]: measured joint angles (lead band; None: no band). -> (joint targets [B, 6]
        rad, info dict)."""
        c, kin = self.cfg, self.kin
        mode = self.mode
        carry = self.carry_on & (mode == M["hold"])            # a CARRY line (option b2) runs or waits for rest
        blendm = (mode == M["free"]) | ((mode == M["hold"]) & ~carry) | (mode == M["g_brake"]) | (mode == M["p_brake"])
        linem = ((mode == M["g_descend"]) | (mode == M["g_lift"]) | (mode == M["p_lower"]) | (mode == M["p_retreat"])
                 | carry)
        linem = linem & (self.lk < self.ln - 1e-9)           # a finished line waits for rest without IK
        still = ~blendm & ~linem
        if q_meas is not None:
            q_meas = q_meas.to(**self.f)
        # nominal target step of the mode
        tau1 = self.tau + CONTROL_DT
        p0, _ = blend(self.v0, self.vn, self.tau, self.tb)
        p1, _ = blend(self.v0, self.vn, tau1, self.tb)
        y0, _ = blend(self.w0, self.wn, self.tau, self.tb)
        y1, _ = blend(self.w0, self.wn, tau1, self.tb)
        n = self.ln.clamp_min(1.0)
        u0, u1 = minjerk(self.lk / n), minjerk((self.lk + 1) / n)
        dline = (self.lp1 - self.lp0) * (u1 - u0)[:, None]
        if self.filt:                                     # (a CARRY abort brake keeps the cosine rule)
            fv1, fa1, fdisp = filter_step(self.fv, self.fa, self.vn, c.wn)
            fw1, fwa1, fydisp = filter_step(self.fw, self.fwa, self.wn, c.wn)
            cb = self.carry_brk
            p1 = torch.where(cb[:, None], p1 - p0, fdisp)
            y1 = torch.where(cb, y1 - y0, fydisp)
            p0, y0 = torch.zeros_like(p1), torch.zeros_like(y1)
        if self.hann:                                     # (a CARRY abort brake keeps the cosine rule)
            hdisp, hydisp = self._hann_step()
            cb = self.carry_brk
            p1 = torch.where(cb[:, None], p1 - p0, hdisp)
            y1 = torch.where(cb, y1 - y0, hydisp)
            p0, y0 = torch.zeros_like(p1), torch.zeros_like(y1)
        dp = torch.where(blendm[:, None], p1 - p0, torch.where(linem[:, None], dline, torch.zeros_like(dline)))
        cline = linem & carry
        dy = torch.where(blendm, y1 - y0, torch.where(cline, (self.ly1 - self.ly0) * (u1 - u0), torch.zeros_like(y1)))
        pos1, yaw1, ws_clip = self._clip(self.pos + dp, self.yaw + dy, blendm)
        # IK and the speed guard
        rot = self._rot(yaw1)
        q_try, ep, ea = kin.ik_track(self.q_des, pos1, rot, c.ik_iters)
        q_try = torch.where(still[:, None], self.q_des, q_try)
        dqb = torch.where(carry[:, None], self.dq_carry, self.dq_max)            # step bound (CARRY: data 'move')
        m = ((q_try - self.q_des).abs() / dqb).amax(-1)
        s = torch.where(m > 1, 1 / m.clamp_min(1e-12), torch.ones_like(m))
        guard = m > 1
        k1 = torch.where(linem, torch.minimum(self.lk + s, n), self.lk + 1)
        u_s = minjerk(k1 / n)
        ratio = torch.where(linem, (u_s - u0) / (u1 - u0).clamp_min(1e-15), s)
        ratio = torch.where(linem & ((u1 - u0) <= 1e-15), s, ratio)
        pos_new = torch.where(linem[:, None], self.lp0 + (self.lp1 - self.lp0) * u_s[:, None],
                              self.pos + s[:, None] * (pos1 - self.pos))
        yaw_new = torch.where(blendm, self.yaw + s * (yaw1 - self.yaw),
                              torch.where(cline, self.ly0 + (self.ly1 - self.ly0) * u_s, self.yaw))
        q_new = self.q_des + ratio[:, None] * (q_try - self.q_des)
        if bool((guard & ~still).any()):    # a scaled step: solve again for the scaled target (push-T guard)
            q_re, ep2, ea2 = kin.ik_track(self.q_des, pos_new, self._rot(yaw_new), c.ik_iters)
            redo = guard & ~still
            q_new = torch.where(redo[:, None], q_re, q_new)
            ep, ea = torch.where(redo, ep2, ep), torch.where(redo, ea2, ea)
        # floor: lowest collision point of the solution >= low_min
        jc = self._jaw_check(q_meas)
        qj = torch.cat((q_new, jc[:, None]), -1)
        low = kin.lowest_point(qj, exact=False)
        need = ~still & (low < c.low_min - 1e-9)
        if bool(need.any()):
            pos_new = pos_new + torch.where(need, c.low_min - low, torch.zeros_like(low))[:, None] * \
                torch.tensor([0.0, 0.0, 1.0], **self.f)
            q_fix, _, _ = kin.ik_track(q_new, pos_new, self._rot(yaw_new), c.ik_iters)
            q_new = torch.where(need[:, None], q_fix, q_new)
            qj = torch.cat((q_new, jc[:, None]), -1)
            low = kin.lowest_point(qj, exact=False)
        self.low_off = torch.where(still, self.low_off, kin.fk(qj)[0][:, 2] - low)
        # command limiter: steps <= dq_max, step changes <= ddq_max, braking-aware
        cprev = self.q_cmd[:, :5]
        tv = q_new - self.q_des
        lag = self.q_des - cprev
        A = self.ddq_mode[mode]                                   # [B, 5]: per-mode bound (module docstring)
        A = torch.where(carry[:, None], self.ddq_carry, A)
        sb = A * (torch.sqrt(0.25 + 2 * lag.abs() / A) - 0.5)
        want = tv + torch.sign(lag) * torch.minimum(lag.abs(), sb)
        inc = torch.minimum(torch.maximum(want, self.dq - A), self.dq + A)
        Ab = 0.9 * A                                     # braking margin for the discrete form of the stop law
        lim_hi = Ab * (torch.sqrt(0.25 + 2 * (kin.thi[:5] - cprev).clamp_min(0) / Ab) - 0.5)   # stop before a limit
        lim_lo = Ab * (torch.sqrt(0.25 + 2 * (cprev - kin.tlo[:5]).clamp_min(0) / Ab) - 0.5)
        inc = torch.minimum(torch.maximum(inc, -torch.minimum(dqb, lim_lo)), torch.minimum(dqb, lim_hi))
        inc = torch.minimum(torch.maximum(inc, self.dq - A), self.dq + A)                # the step-change bound wins
        limited = ((inc - want).abs() > 1e-12).any(-1)
        cnew = cprev + inc
        # floor on the command itself: a lagging command below the floor takes the IK solution (step bounds broken)
        low_c = kin.lowest_point(torch.cat((cnew, jc[:, None]), -1), exact=False)
        over = (low_c < torch.minimum(low, torch.full_like(low, c.low_min)) - 1e-6)
        cnew = torch.where(over[:, None], q_new, cnew)
        lead_clip = torch.zeros_like(limited)
        if q_meas is not None:
            band = torch.minimum(torch.maximum(cnew, q_meas[:, :5] - self.lead_band), q_meas[:, :5] + self.lead_band)
            lead_clip = ((band - cnew).abs() > 1e-12).any(-1)
            cnew = band
        rng = torch.minimum(torch.maximum(cnew, kin.tlo[:5]), kin.thi[:5])
        range_clip = ((rng - cnew).abs() > 1e-12).any(-1)
        cnew = rng
        # jaw schedule
        self.jaw_k = torch.minimum(self.jaw_k + 1, self.jaw_n)
        jaw = self.jaw_from + (self.jaw_to - self.jaw_from) * minjerk(self.jaw_k / self.jaw_n)
        jprev = self.q_cmd[:, 5]
        jaw = (jprev + (jaw - jprev).clamp(-c.jaw_step, c.jaw_step)).clamp(float(kin.tlo[5]), float(kin.thi[5]))
        # commit
        self.dq = cnew - cprev
        self.vel = (pos_new - self.pos) / CONTROL_DT
        self.yrate = (yaw_new - self.yaw) / CONTROL_DT
        self.pos, self.yaw, self.q_des = pos_new, yaw_new, q_new
        self.tau = torch.where(blendm, tau1, self.tau)
        if self.filt:                                     # a clipped target restarts its filter at rest (anti-windup)
            keep = (blendm & ~ws_clip)[:, None]
            self.fv = torch.where(keep, fv1, torch.where(blendm[:, None], torch.zeros_like(fv1), self.fv))
            self.fa = torch.where(keep, fa1, torch.where(blendm[:, None], torch.zeros_like(fa1), self.fa))
            self.fw = torch.where(keep[:, 0], fw1, torch.where(blendm, torch.zeros_like(fw1), self.fw))
            self.fwa = torch.where(keep[:, 0], fwa1, torch.where(blendm, torch.zeros_like(fwa1), self.fwa))
        if self.hann:                                     # blend clocks; a clipped target restarts at rest
            self.hA_tau = torch.where(blendm, self.hA_tau + CONTROL_DT, self.hA_tau)
            self.hB_tau = torch.where(blendm, self.hB_tau + CONTROL_DT, self.hB_tau)
            self._hann_zero(blendm & ws_clip)
        self.lk = torch.where(linem, k1, torch.where(still, self.lk + 1, self.lk))
        self.t = self.t + 1
        self.ik_err = torch.where(still[:, None], self.ik_err, torch.stack((ep, ea), -1))
        self.q_cmd = torch.cat((cnew, jaw[:, None]), -1)
        flags = dict(guard_scaled=guard & ~still, limited=limited, lead_clipped=lead_clip, floor_raised=need,
                     floor_override=over, range_clipped=range_clip, ws_clipped=ws_clip)
        for k, v in flags.items():
            self.counts[k] += v.long()
        events = self._advance()
        info = dict(flags, mode=mode, **events)
        return self.q_cmd.clone(), info

    def _advance(self):
        """Mode transitions at the end of a step (module docstring)."""
        mode = self.mode
        seg = self.lk >= self.ln - 1e-9
        brk = self.tau >= self.tb - 1e-9
        if self.filt:                                     # the filter has (nearly) stopped, or 1 s has passed
            brk = (((self.fv.norm(dim=-1) < 1e-3) & (self.fw.abs() < 1e-3) & (self.fa.norm(dim=-1) < 0.02)
                    & (self.fwa.abs() < 0.05)) | (self.tau >= 1.0))
        if self.hann:                                     # every Hann blend has finished
            brk = self._hann_done()
        jaw = self.jaw_k >= self.jaw_n
        rest = (self.dq.abs().amax(-1) < 1e-9) & ((self.q_des - self.q_cmd[:, :5]).abs().amax(-1) < 1e-9)
        rules = (("g_brake", brk, "g_descend"), ("g_descend", seg, "g_settle"), ("g_settle", seg & rest, "g_close"),
                 ("g_close", jaw, "g_grip"), ("g_grip", seg, "g_lift"), ("g_lift", seg, "hold"),
                 ("p_brake", brk, "p_settle"), ("p_settle", seg, "p_lower"), ("p_lower", seg & rest, "p_open"),
                 ("p_open", jaw, "p_retreat"), ("p_retreat", seg, "free"))
        moves = [((mode == M[a]) & ok, b) for a, ok, b in rules]
        cdone = self.carry_on & (mode == M["hold"]) & seg & rest      # a CARRY line ends at rest: hold again
        bdone = self.carry_brk & (mode == M["hold"]) & (self.tau >= self.tb - 1e-9) & rest   # an abort brake too
        if bool((cdone | bdone).any()):
            self.carry_on = self.carry_on & ~cdone
            self.carry_brk = self.carry_brk & ~bdone
            self._rest_motion(cdone | bdone)
        for mask, b in moves:
            self.mode = torch.where(mask, M[b], self.mode)
        for mask, b in moves:
            if bool(mask.any()):
                self._enter(mask, b)
        return dict(close_started=moves[2][0], grasp_done=moves[5][0], open_started=moves[8][0],
                    place_done=moves[10][0])

    def _enter(self, mask, mode):
        """Set up the segment of ``mode`` for the worlds in mask (all at rest except g_descend / p_settle after a
        brake, which also ends at rest)."""
        c = self.cfg
        zero = torch.zeros_like(self.lk)
        up = torch.tensor([0.0, 0.0, 1.0], **self.f)

        def line(end, t_rng):
            self.lp0 = torch.where(mask[:, None], self.pos, self.lp0)
            self.lp1 = torch.where(mask[:, None], end, self.lp1)
            self.ln = torch.where(mask, self._line_steps(self._draw(mask, t_rng), end, mode), self.ln)
            self.lk = torch.where(mask, zero, self.lk)

        def hold(t_rng):
            self.ln = torch.where(mask, torch.round(self._draw(mask, t_rng) / CONTROL_DT), self.ln)
            self.lk = torch.where(mask, zero, self.lk)

        if mode == "g_descend":
            line(torch.cat((self.pos[:, :2], self.z_goal[:, None]), -1), c.t_descend)
        elif mode == "g_settle":
            hold(c.t_settle)
        elif mode == "g_close":
            self._jaw_move(mask, torch.full_like(self.lk, c.jaw_closed), self._draw(mask, c.t_close))
        elif mode == "g_grip":
            hold(c.t_grip)
        elif mode == "g_lift":
            rise = torch.minimum(self.h, (self.z_ceil - self.pos[:, 2]).clamp_min(0))
            line(self.pos + rise[:, None] * up, c.t_lift)
        elif mode == "p_settle":
            hold(c.t_place_settle)
        elif mode == "p_lower":
            line(torch.cat((self.pos[:, :2], self.z_goal[:, None]), -1), c.t_lower)
        elif mode == "p_open":
            self._jaw_move(mask, torch.full_like(self.lk, c.jaw_open), self._draw(mask, c.t_open))
        elif mode == "p_retreat":
            rise = torch.minimum(self._draw(mask, c.retreat), (c.z_max - self.pos[:, 2]).clamp_min(0))
            line(self.pos + rise[:, None] * up, c.t_retreat)
        if mode in ("hold", "free", "g_descend", "p_settle"):     # free motion (or the macro) starts from rest
            self._rest_motion(mask)
        if mode == "hold":
            self.yaw_hold0 = torch.where(mask, self.yaw, self.yaw_hold0)

    def _rest_motion(self, mask):
        """Free-motion state of the worlds in mask at rest (blends, filter, Hann blends, target velocity)."""
        m3 = mask[:, None]
        self.v0 = torch.where(m3, 0.0, self.v0)
        self.vn = torch.where(m3, 0.0, self.vn)
        self.w0 = torch.where(mask, 0.0, self.w0)
        self.wn = torch.where(mask, 0.0, self.wn)
        self.tau = torch.where(mask, self.T_d, self.tau)
        self.tb = torch.where(mask, self.T_d, self.tb)
        self.vel = torch.where(m3, 0.0, self.vel)     # a macro line ends at rest (its last discrete step is not
        self.yrate = torch.where(mask, 0.0, self.yrate)   # exactly 0): the next decision blends from rest
        for x in (self.fv, self.fa):
            x.copy_(torch.where(m3, 0.0, x))
        for x in (self.fw, self.fwa):
            x.copy_(torch.where(mask, 0.0, x))
        self._hann_zero(mask)

    # ---- views --------------------------------------------------------------------------------------------------
    @property
    def macro_goal(self):
        """[B, 4] end of the last GRASP descent: TCP x, y, z (m) and yaw (rad)."""
        return self.goal

    @property
    def holding(self):
        """[B] bool: modes in which the decoder holds the jaw closed on a cube (g_grip .. p_lower)."""
        m = self.mode
        return (m == M["g_grip"]) | (m == M["g_lift"]) | (m == M["hold"]) | (m == M["p_brake"]) | \
            (m == M["p_settle"]) | (m == M["p_lower"])

    def progress(self):
        """[B] in [0, 1]: fraction of the current macro segment done (0 in free and hold)."""
        m = self.mode
        brk = (m == M["g_brake"]) | (m == M["p_brake"])
        jawm = (m == M["g_close"]) | (m == M["p_open"])
        seg = (self.lk / self.ln.clamp_min(1)).clamp(0, 1)
        out = torch.where(brk, (self.tau / self.tb.clamp_min(1e-9)).clamp(0, 1), seg)
        out = torch.where(jawm, self.jaw_k / self.jaw_n, out)
        return torch.where((m == M["free"]) | ((m == M["hold"]) & ~self.carry_on), torch.zeros_like(out), out)

    def macro(self):
        """[B] long: the task's macro phase code (``MACROS`` = task.MACROS) of the current mode (``MACRO_OF_MODE``:
        g_brake, g_descend, g_settle -> grasp_descend; hold -> free; p_brake, p_settle, p_lower -> place_lower)."""
        if not hasattr(self, "_macro_lut"):
            self._macro_lut = torch.tensor([MACROS.index(n) for n in MACRO_OF_MODE], dtype=torch.long, device=self.dev)
        return self._macro_lut[self.mode]

    OBS_DIM = 24

    def observation(self):
        """Decoder state for a policy [B, 24]: target TCP (3), sin and cos of the target yaw (2), target velocity (3),
        yaw rate (1), jaw command (1), trigger armed (1), mode one-hot (13)."""
        oh = torch.nn.functional.one_hot(self.mode, len(MODES)).to(self.pos)
        return torch.cat((self.pos, torch.sin(self.yaw)[:, None], torch.cos(self.yaw)[:, None], self.vel,
                          self.yrate[:, None], self.q_cmd[:, 5:6], self.armed[:, None].to(self.pos), oh), -1)


# ---- DecoderV1 (ablation) -------------------------------------------------------------------------------------
# spec_v1 2.4 as built before the reviews: action a [B, 5] in [-1, 1]: a[0:3] TCP velocity (x 0.35 m/s, norm clipped),
# a[3] yaw rate (x 2 rad/s), a[4] jaw (> 0: close to 0.0 rad, else open to 0.7 rad), every 20 ms. Per step: the
# velocity and yaw-rate commands pass a critically damped 2nd-order low pass (wn 12.5 rad/s, exact for a command held
# over the step, the integral kept); the Cartesian target integrates it, clamped to the workspace (radius 0.10-0.32 m,
# |pan angle| <= 1.75 rad, TCP z 0.015-0.22 m, |roll| <= 2.5 rad; a clamped axis loses its filter state); the start
# tilt is blended out over 0.6 s; IK (SO101Kin.ik_track) -> arm targets with steps <= 0.06 rad and the target range;
# the Cartesian target is pulled back to within 1 cm of the commanded pose (anti-windup); the jaw makes a minimum-jerk
# move to the commanded end (Plan.jaw duration rule), restarted when the command flips, steps <= 0.09 rad.


@dataclass
class DecoderV1Config:
    v_max: float = 0.35            # m/s, grasp-point speed (norm)
    yaw_rate_max: float = 2.0      # rad/s
    wn: float = 12.5               # rad/s, natural frequency of the critically damped velocity filter
    jaw_closed: float = 0.0        # rad, firm command (the data close to -0.1245..0.05)
    jaw_open: float = 0.7          # rad (the data open to 0.55-0.9)
    jaw_speed: float = 4.5         # rad/s, Plan.jaw
    jaw_min_s: float = 0.2         # s, shortest jaw move
    arm_step: float = 0.06         # rad per step
    jaw_step: float = 0.09         # rad per step
    r_range: tuple = (0.10, 0.32)  # m, grasp point radius about the pan axis
    angle_max: float = 1.75        # rad, |pan angle| of the grasp point
    z_range: tuple = (0.015, 0.22)  # m, grasp point height
    roll_max: float = 2.5          # rad, |yaw - radial direction|
    tilt_s: float = 0.6            # s, removal of the start tilt
    max_lead: float = 0.01         # m, Cartesian target ahead of the commanded arm pose at most
    ik_iters: int = 3              # IK iterations per weight per step (SO101Kin.ik_track)


class DecoderV1:
    """Decoder v1 (ablation arm; module docstring, section DecoderV1). ``step(a) -> targets [B, 6]`` (rad, jaw last)."""

    def __init__(self, num_envs, device="cpu", kin: SO101Kin | None = None, cfg: DecoderV1Config | None = None):
        self.cfg = cfg or DecoderV1Config()
        self.dev = torch.device(device)
        self.kin = kin or SO101Kin(self.dev)
        self.B = num_envs
        f = dict(dtype=torch.float64, device=self.dev)
        self.pos = torch.zeros(num_envs, 3, **f)          # Cartesian target: grasp point (world, m)
        self.yaw = torch.zeros(num_envs, **f)             # target yaw of the closing axis (rad, not wrapped)
        self.vel = torch.zeros(num_envs, 3, **f)          # filtered velocity (m/s) and its derivative
        self.acc = torch.zeros(num_envs, 3, **f)
        self.yrate = torch.zeros(num_envs, **f)
        self.yacc = torch.zeros(num_envs, **f)
        self.tilt0 = torch.zeros(num_envs, 3, **f)        # start tilt (rotation vector in the top-down frame)
        self.t = torch.zeros(num_envs, **f)               # steps since reset
        self.q_cmd = torch.zeros(num_envs, 6, **f)        # last joint targets
        self.jaw_from = torch.zeros(num_envs, **f)        # jaw minimum-jerk segment: start, end, step, length
        self.jaw_to = torch.zeros(num_envs, **f)
        self.jaw_k = torch.ones(num_envs, **f)
        self.jaw_n = torch.ones(num_envs, **f)
        self.jaw_mode = torch.full((num_envs,), -1, dtype=torch.long, device=self.dev)   # -1 none, 0 open, 1 closed
        self.ik_err = torch.zeros(num_envs, 2, **f)       # position (m) and angle (rad) error of the last IK
        self.lead = torch.zeros(num_envs, **f)            # Cartesian target minus commanded arm pose (m)
        self.pan_axis = torch.tensor(PAN_AXIS_XY, **f)

    # ---- reset ---------------------------------------------------------------------------------------------
    def reset(self, idx, q):
        """Worlds idx (index tensor or bool mask) start holding the measured joint angles q [n, 6]."""
        q = torch.minimum(torch.maximum(q.to(self.q_cmd), self.kin.tlo), self.kin.thi)
        p, r = self.kin.fk(q)
        yaw = yaw_of(r)
        self.q_cmd[idx] = q
        self.pos[idx], self.yaw[idx] = p, yaw
        self.tilt0[idx] = rotvec(rotz(yaw).transpose(-1, -2) @ r)
        for x in (self.vel, self.acc, self.yrate, self.yacc, self.t, self.ik_err, self.lead):
            x[idx] = 0.0
        self.jaw_from[idx], self.jaw_to[idx] = q[:, 5], q[:, 5]
        self.jaw_k[idx], self.jaw_n[idx] = 1.0, 1.0
        self.jaw_mode[idx] = -1

    # ---- one step ------------------------------------------------------------------------------------------
    def _filter(self, x, xd, u):
        """Critically damped 2nd-order filter x'' = wn^2 (u - x) - 2 wn x', input held over one step ->
        (x, x', integral of x over the step)."""
        w, T = self.cfg.wn, CONTROL_DT
        E = math.exp(-w * T)
        e0 = x - u
        x1 = u + E * (e0 * (1 + w * T) + xd * T)
        xd1 = E * (xd * (1 - w * T) - e0 * w * w * T)
        disp = u * T + e0 * (2 - E * (2 + w * T)) / w + xd * (1 - E * (1 + w * T)) / (w * w)
        return x1, xd1, disp

    def target_rot(self):
        """Target gripper rotation [B, 3, 3]: top-down at the target yaw, with the start tilt blended out."""
        lam = 1 - minjerk(self.t * CONTROL_DT / self.cfg.tilt_s)
        return rotz(self.yaw) @ expmap(self.tilt0 * lam[:, None])

    @torch.no_grad()
    def step(self, action):
        """action [B, 5] in [-1, 1] -> joint targets [B, 6] (rad)."""
        c = self.cfg
        a = action.to(self.pos).clamp(-1.0, 1.0)
        u = a[:, :3] * c.v_max
        n = u.norm(dim=-1, keepdim=True)
        u = u * torch.where(n > c.v_max, c.v_max / n.clamp_min(1e-12), torch.ones_like(n))
        self.vel, self.acc, disp = self._filter(self.vel, self.acc, u)
        self.yrate, self.yacc, ydisp = self._filter(self.yrate, self.yacc, a[:, 3] * c.yaw_rate_max)
        pos = self.pos + disp
        yaw = self.yaw + ydisp
        # workspace clamp (anti-windup: a clamped axis loses its filter state)
        z = pos[:, 2].clamp(*c.z_range)
        zc = z != pos[:, 2]
        rel = pos[:, :2] - self.pan_axis
        r = rel.norm(dim=-1)
        ang = torch.atan2(rel[:, 1], rel[:, 0])
        r2, ang2 = r.clamp(*c.r_range), ang.clamp(-c.angle_max, c.angle_max)
        xyc = (r2 != r) | (ang2 != ang)
        xy = self.pan_axis + torch.stack((torch.cos(ang2), torch.sin(ang2)), -1) * r2[:, None]
        pos = torch.cat((xy, z[:, None]), -1)
        roll = wrap(yaw - ang2)
        yc = roll.abs() > c.roll_max
        yaw = yaw - (roll - roll.clamp(-c.roll_max, c.roll_max))
        self.vel[:, 2] = torch.where(zc, 0.0, self.vel[:, 2])
        self.acc[:, 2] = torch.where(zc, 0.0, self.acc[:, 2])
        self.vel[:, :2] = torch.where(xyc[:, None], 0.0, self.vel[:, :2])
        self.acc[:, :2] = torch.where(xyc[:, None], 0.0, self.acc[:, :2])
        self.yrate = torch.where(yc, 0.0, self.yrate)
        self.yacc = torch.where(yc, 0.0, self.yacc)
        self.pos, self.yaw = pos, yaw
        self.t = self.t + 1
        # IK, rate limit, joint range
        q, ep, ea = self.kin.ik_track(self.q_cmd[:, :5], self.pos, self.target_rot(), c.ik_iters)
        q = self.q_cmd[:, :5] + (q - self.q_cmd[:, :5]).clamp(-c.arm_step, c.arm_step)
        q = torch.minimum(torch.maximum(q, self.kin.tlo[:5]), self.kin.thi[:5])
        self.ik_err = torch.stack((ep, ea), -1)
        p_cmd, _ = self.kin.fk(q)
        lead = self.pos - p_cmd
        ln = lead.norm(dim=-1, keepdim=True)
        self.pos = p_cmd + lead * torch.where(ln > c.max_lead, c.max_lead / ln.clamp_min(1e-12), torch.ones_like(ln))
        self.lead = ln[:, 0].clamp_max(c.max_lead)
        # jaw: minimum-jerk move to the commanded end, restarted when the command flips
        mode = (a[:, 4] > 0).long()
        flip = mode != self.jaw_mode
        goal = torch.where(mode == 1, c.jaw_closed, c.jaw_open).to(self.pos)
        jaw = self.q_cmd[:, 5]
        steps = torch.maximum(torch.full_like(jaw, round(c.jaw_min_s / CONTROL_DT)),
                              torch.ceil(1.875 * (goal - jaw).abs() / (c.jaw_speed * CONTROL_DT)))
        self.jaw_from = torch.where(flip, jaw, self.jaw_from)
        self.jaw_to = torch.where(flip, goal, self.jaw_to)
        self.jaw_n = torch.where(flip, steps, self.jaw_n)
        self.jaw_k = torch.where(flip, torch.zeros_like(jaw), self.jaw_k)
        self.jaw_mode = mode
        self.jaw_k = torch.minimum(self.jaw_k + 1, self.jaw_n)
        j = self.jaw_from + (self.jaw_to - self.jaw_from) * minjerk(self.jaw_k / self.jaw_n)
        j = jaw + (j - jaw).clamp(-c.jaw_step, c.jaw_step)
        j = j.clamp(float(self.kin.tlo[5]), float(self.kin.thi[5]))
        self.q_cmd = torch.cat((q, j[:, None]), -1)
        return self.q_cmd.clone()

    # ---- views ---------------------------------------------------------------------------------------------
    @property
    def jaw_done(self):
        """[B] bool: the jaw target has reached the end of its move."""
        return self.jaw_k >= self.jaw_n

    def macro(self):
        """None: decoder v1 has no macros (review F6). ``task.PickPlaceTask.step(state, macro=None)`` then treats every
        step as free motion and finds the close start geometrically (the jaw closing through ``close_jaw``)."""
        return None

    @property
    def macro_goal(self):
        """None: no GRASP descent is planned (the task's trigger-box rule then uses the measured TCP)."""
        return None

    def observation(self):
        """Decoder target state for a policy [B, 11]: target position (3), sin and cos of the target yaw, filtered
        velocity (3), filtered yaw rate, jaw target (rad), jaw command (1 closed, 0 open, -1 none)."""
        return torch.cat((self.pos, torch.sin(self.yaw)[:, None], torch.cos(self.yaw)[:, None], self.vel,
                          self.yrate[:, None], self.q_cmd[:, 5:6], self.jaw_mode[:, None].to(self.pos)), -1)


CommandDecoder = DecoderV1     # name of the draft (01:20)
