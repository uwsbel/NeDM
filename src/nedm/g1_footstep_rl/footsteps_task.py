"""The footstep-target task for the G1 (numpy only): courses, their generators and banks, the 1-D planner for the
old plank layouts, the showcase courses, and FootstepProgress, the numpy reference scorer of both worlds.

The task follows the internal design spec of the study (cited as "spec N" below; not released; see the package
docstring). The choices made where that spec is silent or contradicts itself are listed under "Decisions" below.

Frames (spec 4.1)
- World: MuJoCo / NRD world, z up, floor or plank tops at z = 0.
- Course frame: origin = midpoint of the two foot centres of the start state, x = the start pelvis heading. Banks
  store courses in this frame (origin (0, 0), heading 0). A placed course stores WORLD values plus its origin and
  heading; Course.local() gives the course-frame values back. For the default standing pose the foot centres are at
  (0, +-FOOT0_L) of the course frame and the pelvis at (-FOOT0_S, 0).
- Local path frame of target i: origin C_i = centre[i, :2], tangent t_i = (cos psi_i, sin psi_i), left normal n_i.
- Foot centre = mean of the 4 sphere centres; foot yaw = atan2 of (mean toe spheres - mean heel spheres)
  (foot_pose). Sphere order = planks_task.FOOT_SPHERES (heel left, heel right, toe left, toe right).
- Step vector of target i: target i minus the previous footprint (target i - 1; for target 0 the other foot's start
  centre), L along t_i, W = side_i * (component along n_i) (positive when the feet are apart).

Arrays of a Course (float64): targets [N, 5] = (x, y, yaw, side +1 left / -1 right, stop 0/1), world;
centre [N, 3] = (x, y, psi) world; intervals [P, 2] (s_start, s_end) of the start platform, the planks and the final
platform in the COURSE frame (None on flat courses); half_width = |l| limit of every plank. Foot index f = 0 for
side +1 (left), 1 for side -1 (right), as the NRD state and fk.foot_points.

Course banks (build_course_bank.py; save_bank / load_bank): float32 course-frame arrays level [M] int8 (index in
LEVEL_CODES), seed [M] int64, n [M] int16, targets [M, 40, 5], centre [M, 40, 3], f0 [M] int8, d0 [M], end [M] int8
(0 through, 1 stop), cliff [M] bool, n_planks [M] int8 (interior planks, 0 flat), n_intervals [M] int8 (rows of
intervals in use, 0 flat), intervals [M, 16, 2], half_width [M]; rows past n repeat the last target / centre row,
interval rows past n_intervals are (1e9, -1e9) (empty: they never contain a point). load_bank returns float64.

Decisions (the spec is silent or self-contradictory; the simplest option was taken)
 D1  A step-length draw that must obey |L_i - L_(i-1)| <= dL_max is drawn uniformly on the intersection of the
     level range with [L_prev - dL_max, L_prev + dL_max]: the same distribution as rejecting that single draw. The
     course-level rejection loop handles the remaining checks: at most 50 tries on flat levels (4.3; the largest
     count seen in 500 courses per level was 4), at most 1,000 on plank levels (4.4 gives no cap and P1 accepts
     only ~18 % of its draws: 50 tries would fail about once per 25,000 courses); then RuntimeError. The discrete
     course type (N, first foot, stop / through, turns, lanes, number of planks, half width) is drawn once per
     course, so the level shares are exact.
 D2  The dL rule applies from step 2 on (L_1 is the first step from standing, capped at 0.20 m, as the planner of
     4.5). The closing step (L = 0) is a step: the step before it is <= min(0.18, dL_max).
 D3  Stop endings on plank courses: the final platform holds 4 footprints (the crossing footprint, a step a <= 0.28,
     a step b <= min(0.18, dL_max), the closing step), because a stop pair right after a crossing (>= 0.26 m) breaks
     both the 4.2 caps and the dL rule. Through endings: 2 footprints as specified.
 D4  F3 widths: h_i ~ U(0.07, 0.14) (width in [0.14, 0.28]); turns and lanes on every F3 course (the spec gives no
     share; F2: 50 % turns, 30 % lanes). Turn segments are 4-10 steps on F2 and F3.
 D5  Lanes: a random walk of the centre line from target 3 on, lane_i = clip(lane_(i-1) + U(-step, step), +-total).
     The centre points include the lane, target_i = centre_i + side_i h_i n_i.
 D6  Curves: dpsi_i = clip(kappa L_i, +-dpsi_max), kappa ~ U(-k_max, k_max) per segment; the path point advances by
     L_i along the mid heading psi_(i-1) + dpsi_i / 2.
 D7  h_i is clipped so that the exact width W_i lies in [0.14, 0.28]. Stop targets use h = 0.12 (width 0.24); a
     course whose step into the stop pair leaves [0.14, 0.28] is redrawn.
 D8  Plank levels use the F2 width jitter 0.06 (the plank table gives none).
 D9  First foot: +1 / -1 with probability 1/2 (generators); +1 (left) for the planner and the showcases.
 D10 ES cliff: the stop pair's front sphere clearance (0.025-0.045 m incl. the 5 mm radius) replaces the true-margin
     check at the cliff edge.
 D11 Planner cost: the margin term is 2 (0.05 - min(margin, 0.05)) >= 0, i.e. the spec's -2 min(margin, 0.05) plus
     a constant 0.10 per step (the spec's form is unbounded below in the number of steps: 1 cm steps would be
     "cheaper"). Grid and start set as S/scripts/spec_v4_feasibility.py (start = the 7 grid points from the point
     nearest FOOT0_S, i.e. 0.01-0.07 m in the layout frame), so the section 2.2 counts are reproduced exactly.
 D12 Planner stop end: plan to the first footprint on the final platform, then steps a = min(0.28, L_last),
     b = min(0.15, a), and the closing pair; no plan if a footprint leaves the final platform's safe window.
 D13 Course.to_layout() is the pelvis-origin layout of the default standing pose (intervals + FOOT0_S, origin
     FOOT0_S behind the course origin), so make_planks_env(course.to_layout()) and run_planks_episode put the robot
     on the course. Planner courses return their source layout exactly (no float round trip).
 D14 Strict-only failures: wrong_foot, far_miss, restep_budget, overrun, no_stop (spec 3.4, 5.3 and 8 together);
     stall is always a failure. With strict false an overrun is only logged and no_stop is flagged once; the stop
     phase continues and a later stop success still completes the course.
 D15 One swing counter per foot with the PlanksProgress rule (reset by any gate-on step).
 D15b (2026-10-11, review finding) A dropped flicker sets the foot's swing counter to min_swing_steps, so a landing
     right after it (on, off, on) is a touchdown at the second on-step. Without this the NRD's rare one-step contact
     dropouts (about 1 % of touchdowns under exploration noise) lost the landing and the next step failed as wrong
     foot. MuJoCo labels cannot drop for one step after a landing (OR of two steps): MuJoCo scoring is unchanged.
 D16 Timing uses the touchdown step k (processed at k + 1 with confirm on, at k without): start window (k - t0_step < 50 and
     pointer < 2), stall (step - last scored touchdown step >= 75, the start counts as t0_step), stop limit
     (step - stop touchdown step >= 125). Stop stillness counts update steps from the processing step on.
 D17 A through course whose last target is scored as a (non-strict) far miss is finished but not completed.
 D18 The plank counters (touchdowns, touchdowns_over_gap, unsupported_steps; the PlanksProgress rule on the true
     planks, no edge margin) are updated at every update() call, also before t0_step and after the scoring ended;
     the drift statistics from t0_step on, also after the scoring ended (so controllers that ignore the footprints
     get them too);
     scoring runs from t0_step until the first terminal event.
 D19 since_td starts saturated (1.0 s); last placements start at the given start feet and yaws (default: the
     default-pose feet, yaw = course heading).
 D20 Seed level offsets = index in LEVEL_CODES. Old-layout courses (old_course): first feasible of V1, V2, V3 for
     every old level; the variant is returned and the evaluation decides what it claims on.
 D21 Showcases: f0 left, d0 0.03 m, widths W_nat(L) / 2; mixed_*: fixed gap list SHOW_GAPS, start platform to
     0.50 m, footprints by the planner with margin 0.03 m first (variant "VS", the training margin), then V1-V3,
     stop tail of D12, mixed_narrow_edge cliff 0.03 m in front of the toe spheres; lane_change shifts 0.05 m per
     step at L = 0.20 (at 0.25 m the right-foot width would fall below 0.14 m); step_change ends "through" (the spec
     names no stop). No showcase uses a course seed of SEEDS["test"] or SEEDS["test_old"] (sealed final test).
"""
from __future__ import annotations

import copy
import json
import math
import zlib
from dataclasses import dataclass, field

import numpy as np

from nedm.g1_footstep_rl import planks_task as P

# ---- constants (m, s, rad) ---------------------------------------------------------------------------------------
CONTROL_S = 0.02
HALF_SOLE = 0.085                 # sphere centres +-0.085 m along the foot from the foot centre
SOLE_HALF_WIDTH = 0.03            # toe sphere centres +-0.03 m across (heel +-0.025)
FOOT0_S = 0.009                   # default standing pose: foot centres 0.009 m ahead of the pelvis
FOOT0_L = 0.1185                  # ... and +-0.1185 m to the side
FOOT_RADIUS = P.FOOT_RADIUS
SOLE_OFFSETS = P.FOOT_SPHERES[:, :2] - P.FOOT_SPHERES[:, :2].mean(0)     # [4, 2] sphere centres in the foot frame
W_MIN, W_MAX = 0.14, 0.28
STOP_WIDTH = 0.24
STOP_CAPS = (0.28, 0.18)          # the two steps before the closing step (spec 4.2)
RAMP_D0 = (0.0, 0.06)
RAMP_L1 = (0.08, 0.20)
RAMP_L2 = (0.10, 0.05, 0.28)      # L_2 ~ U(max(0.10, L_1 - 0.05), min(0.28, L_1 + dL_max))
START_BACK = 0.80                 # start platform from s_feet - 0.80
FINAL_LEN = 1.0
CLIFF = (0.02, 0.04)
GAP_MIN = 0.03
EDGE_MARGIN = 0.02
NMAX, PMAX = 40, 16
MAX_TRIES = 50                    # flat step programs (spec 4.3)
MAX_TRIES_PLANKS = 1000           # plank courses: 4.4 says "retry the course" with no cap; P1 accepts ~18 % of draws
PAD_INTERVAL = (1e9, -1e9)

LEVEL_CODES = ("F0", "F1", "F2", "F3", "P1", "P2", "P3", "ES", "MP", "MN")
TRAIN_LEVELS = LEVEL_CODES[:8]
VAL_LEVELS = LEVEL_CODES
FLAT_LEVELS = LEVEL_CODES[:4]
PLANK_LEVELS = LEVEL_CODES[4:]
OLD_LEVELS = ("L1", "L2", "L3", "L4")
END_CODES = {"through": 0, "stop": 1}
END_NAMES = {v: k for k, v in END_CODES.items()}

# Levels (spec 4.4). Flat: n targets, program kind, L range after the ramp, dL max, width jitter (or a width range),
# turns (share, k_max 1/m, dpsi_max rad), lanes (share, per step, total), stop share.
LEVELS_FS = {
    "F0": dict(kind="flat", n=(12, 16), prog="hold1", L=(0.10, 0.28), dl=0.10, w_jit=0.02, turn_p=0.0, kmax=0.0,
               dpsi=0.0, lane_p=0.0, lane_step=0.0, lane_max=0.0, stop=0.0),
    "F1": dict(kind="flat", n=(12, 20), prog="piecewise", seg=(2, 5), L=(0.05, 0.30), dl=0.12, w_jit=0.04,
               turn_p=0.0, kmax=0.0, dpsi=0.0, lane_p=0.0, lane_step=0.0, lane_max=0.0, stop=0.5),
    "F2": dict(kind="flat", n=(16, 24), prog="markov", hold=0.3, L=(0.0, 0.34), dl=0.15, w_jit=0.06, turn_p=0.5,
               kmax=0.5, dpsi=0.08, lane_p=0.3, lane_step=0.03, lane_max=0.15, stop=0.5),
    "F3": dict(kind="flat", n=(16, 24), prog="markov", hold=0.2, L=(-0.05, 0.36), dl=0.20, width=(W_MIN, W_MAX),
               turn_p=1.0, kmax=0.83, dpsi=0.15, lane_p=1.0, lane_step=0.05, lane_max=0.25, stop=0.5),
    "P1": dict(kind="planks", planks=(6, 9), k=(0.20, 0.50, 0.30), G=(0.03, 0.07), m=(0.040, 0.060),
               Lin=(0.10, 0.28), Lmax=0.34, dl=0.15, hw=(1.0,), stop=0.5, cliff=False, w_jit=0.06),
    "P2": dict(kind="planks", planks=(6, 9), k=(0.35, 0.45, 0.20), G=(0.03, 0.10), m=(0.035, 0.050),
               Lin=(0.10, 0.30), Lmax=0.34, dl=0.20, hw=(1.0,), stop=0.5, cliff=False, w_jit=0.06),
    "P3": dict(kind="planks", planks=(6, 9), k=(0.50, 0.35, 0.15), G=(0.05, 0.13), m=(0.030, 0.045),
               Lin=(0.08, 0.30), Lmax=0.36, dl=0.20, hw=(1.0, 0.30, 0.20), stop=0.5, cliff=False, w_jit=0.06),
    "ES": dict(kind="planks", planks=(4, 7), k=(0.40, 0.40, 0.20), G=(0.03, 0.13), m=(0.030, 0.045),
               Lin=(0.08, 0.30), Lmax=0.36, dl=0.20, hw=(0.25,), stop=1.0, cliff=True, w_jit=0.06),
    "MP": dict(kind="planks", planks=(8, 8), k=(0.40, 0.40, 0.20), G=(0.03, 0.13), m=(0.030, 0.045),
               Lin=(0.08, 0.30), Lmax=0.36, dl=0.20, hw=(1.0,), stop=0.5, cliff=False, w_jit=0.06),
    "MN": dict(kind="planks", planks=(8, 8), k=(0.40, 0.40, 0.20), G=(0.03, 0.13), m=(0.030, 0.045),
               Lin=(0.08, 0.30), Lmax=0.36, dl=0.20, hw=(0.25,), stop=0.5, cliff=False, w_jit=0.06),
}

# Planner variants (spec 4.5): (name, margin m, L max, dL max); first feasible wins.
PLANNER_VARIANTS = (("V1", 0.02, 0.36, 0.20), ("V2", 0.015, 0.36, 0.25), ("V3", 0.01, 0.42, 0.30))
GRID = 0.01

# Seeds (spec 4.7, 12.3).
SEEDS = dict(train_base=10_000, train_stride=100_000, train_per_level=25_000, val=(9000, 9128), hfs_tune=(0, 16),
             dev=(1000, 1064), dev_old=(100, 164), test=(6000, 6064), test_old=(5000, 5064), rounds=(20000, 22000))

# FootstepProgress config (= the env config's "score" block + rules.edge_margin + the control step).
SCORE_DEFAULTS = dict(tol=0.05, tol_fail=0.15, r_adj=0.05, r_start=0.15, start_window_s=1.0, restep_budget=3,
                      stall_s=1.5, confirm=True, min_swing_steps=2, stop_still_s=0.5, stop_v=0.10, stop_wz=0.30,
                      stop_limit_s=2.5, strict=True, edge_margin=EDGE_MARGIN, dt=CONTROL_S)
FAIL_REASONS = ("stall", "wrong_foot", "far_miss", "restep_budget", "overrun", "no_stop")
STRICT_ONLY = ("wrong_foot", "far_miss", "restep_budget", "overrun", "no_stop")


def W_nat(L):
    """HOMIE's natural step width for step length L (m): clip(0.28 - 0.40 L, 0.15, 0.25)."""
    return np.clip(0.28 - 0.40 * np.asarray(L, np.float64), 0.15, 0.25)


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def train_seed(level, i):
    """Seed of training-bank course i of a level: 10,000 + level index x 100,000 + i."""
    return SEEDS["train_base"] + LEVEL_CODES.index(level) * SEEDS["train_stride"] + int(i)


def _rot(h):
    c, s = math.cos(h), math.sin(h)
    return np.array([[c, -s], [s, c]])


def to_world(xy_local, origin, heading):
    """Course frame [..., 2] -> world [..., 2]."""
    return np.asarray(origin, np.float64) + np.asarray(xy_local, np.float64) @ _rot(heading).T


def to_local(xy, origin, heading):
    """World [..., 2] -> course frame (s, l) [..., 2]."""
    return (np.asarray(xy, np.float64)[..., :2] - np.asarray(origin, np.float64)) @ _rot(heading)


def quat_yaw(q):
    """Yaw of wxyz quaternions [..., 4]."""
    q = np.asarray(q, np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def foot_pose(spheres):
    """Sphere centres [..., 2, 4, >=2] -> (foot centres [..., 2, k] (mean of the 4), foot yaws [..., 2])."""
    sp = np.asarray(spheres, np.float64)
    d = sp[..., 2:4, :2].mean(-2) - sp[..., 0:2, :2].mean(-2)
    return sp.mean(-2), np.arctan2(d[..., 1], d[..., 0])


def course_frame(foot_centres, pelvis_yaw):
    """(origin (2,), heading) of the course frame of a start state: midpoint of the two foot centres [2, >=2] and
    the pelvis yaw."""
    fc = np.asarray(foot_centres, np.float64)[:, :2]
    return fc.mean(0), float(pelvis_yaw)


def sole_points(xy, yaw):
    """Sphere centres [..., 4, 2] of feet with centres xy [..., 2] and yaws [...] (flat foot)."""
    xy, yaw = np.asarray(xy, np.float64), np.asarray(yaw, np.float64)
    c, s = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
    ox, oy = SOLE_OFFSETS[:, 0], SOLE_OFFSETS[:, 1]
    return np.stack((xy[..., None, 0] + c * ox - s * oy, xy[..., None, 1] + s * ox + c * oy), -1)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return o


# ---- the course ------------------------------------------------------------------------------------------------
@dataclass
class Course:
    """One footstep course (see the module docstring for the frames and array layouts)."""
    origin: np.ndarray
    heading: float
    targets: np.ndarray
    centre: np.ndarray
    f0: int = 1
    d0: float = 0.0
    end: str = "through"
    cliff: bool = False
    intervals: np.ndarray | None = None
    half_width: float = P.HALF_WIDTH
    level: str = ""
    seed: int = -1
    program: dict = field(default_factory=dict)

    def __post_init__(self):
        self.origin = np.asarray(self.origin, np.float64).reshape(2)
        self.heading = float(self.heading)
        self.targets = np.asarray(self.targets, np.float64).reshape(-1, 5)
        self.centre = np.asarray(self.centre, np.float64).reshape(-1, 3)
        self.f0, self.d0, self.cliff = int(self.f0), float(self.d0), bool(self.cliff)
        self.half_width, self.seed = float(self.half_width), int(self.seed)
        if self.intervals is not None:
            self.intervals = np.asarray(self.intervals, np.float64).reshape(-1, 2)
        if self.end not in END_CODES:
            raise ValueError(f"end must be one of {list(END_CODES)}")
        if len(self.targets) != len(self.centre) or not len(self.targets):
            raise ValueError("targets and centre must have the same, non-zero length")

    # -- basic properties ------------------------------------------------------------------------------------
    @property
    def n(self):
        return len(self.targets)

    @property
    def is_planks(self):
        return self.intervals is not None

    @property
    def n_planks(self):
        """Interior planks (planks_task.Layout.n_planks); 0 on flat courses."""
        return 0 if self.intervals is None else len(self.intervals) - 2

    def foot_of(self, i=None):
        """Foot index (0 left, 1 right) of target i (or of all targets)."""
        side = self.targets[:, 3] if i is None else self.targets[i, 3]
        return np.where(np.asarray(side) > 0, 0, 1)

    # -- frames ---------------------------------------------------------------------------------------------
    def path_coords(self, xy):
        """World [..., 2] -> course frame (s, l) [..., 2]."""
        return to_local(xy, self.origin, self.heading)

    def world_xy(self, s, l=0.0):
        s, l = np.broadcast_arrays(np.asarray(s, np.float64), np.asarray(l, np.float64))
        return to_world(np.stack((s, l), -1), self.origin, self.heading)

    def local(self):
        """(targets [N, 5], centre [N, 3]) in the course frame."""
        t, c = self.targets.copy(), self.centre.copy()
        t[:, :2], c[:, :2] = self.path_coords(t[:, :2]), self.path_coords(c[:, :2])
        t[:, 2], c[:, 2] = wrap(t[:, 2] - self.heading), wrap(c[:, 2] - self.heading)
        return t, c

    def placed(self, origin=(0.0, 0.0), heading=0.0):
        """The same course placed rigidly at a new origin and heading (world = origin + Rot(heading) x local)."""
        t, c = self.local()
        out = copy.deepcopy(self)
        out.origin, out.heading = np.asarray(origin, np.float64).reshape(2), float(heading)
        out.targets[:, :2], out.centre[:, :2] = to_world(t[:, :2], origin, heading), to_world(c[:, :2], origin, heading)
        out.targets[:, 2], out.centre[:, 2] = wrap(t[:, 2] + heading), wrap(c[:, 2] + heading)
        return out

    def with_target0(self, foot_centre_xy):
        """A copy with target 0 = that foot centre (world x, y of foot f0 at the start) + d0 along t_0."""
        out = copy.deepcopy(self)
        psi = self.centre[0, 2]
        out.targets[0, :2] = np.asarray(foot_centre_xy, np.float64)[:2] + self.d0 * np.array([math.cos(psi),
                                                                                             math.sin(psi)])
        return out

    def at_start(self, foot_centres, pelvis_yaw):
        """This course (any placement) placed in the course frame of a start state (course_frame of its foot
        centres [2, >=2] (left, right) and pelvis yaw), with target 0 set from the start centre of foot f0
        (with_target0): the NRD reset of spec 4.1."""
        fc = np.asarray(foot_centres, np.float64)[:, :2]
        origin, heading = course_frame(fc, pelvis_yaw)
        return self.placed(origin, heading).with_target0(fc[int(self.foot_of(0))])

    def start_feet(self):
        """World foot centres [2, 2] (left, right) of the default standing pose on this course."""
        return to_world(np.array([[0.0, FOOT0_L], [0.0, -FOOT0_L]]), self.origin, self.heading)

    def start_pose(self):
        """World pelvis pose (x, y, yaw) of the default standing pose (= to_layout().origin, heading)."""
        x, y = to_world(np.array([-FOOT0_S, 0.0]), self.origin, self.heading)
        return float(x), float(y), self.heading

    # -- supports -------------------------------------------------------------------------------------------
    def support_index(self, xy, margin=0.0):
        """Interval row under world points [..., 2] (planks shrunk by margin along and across), -1 over a gap or
        beside the planks; flat courses: 0 everywhere. margin 0 = planks_task.Layout.plank_of."""
        xy = np.asarray(xy, np.float64)
        if self.intervals is None:
            return np.zeros(xy.shape[:-1], np.int64)
        sl = self.path_coords(xy)
        s, l = sl[..., 0, None], sl[..., 1, None]
        iv = self.intervals
        inside = (s >= iv[:, 0] + margin) & (s <= iv[:, 1] - margin) & (np.abs(l) <= self.half_width - margin)
        return np.where(inside.any(-1), inside.argmax(-1), -1)

    def target_planks(self):
        """[N] interval row under each target's foot centre (0 on flat courses)."""
        return self.support_index(self.targets[:, :2])

    def target_slack(self, edge_margin=EDGE_MARGIN, clip=(-0.10, 0.30)):
        """[N, 4] slack (front, back, left, right) of every target (spec 6): how far the foot centre may move before
        a sphere centre of the foot at the target pose leaves the target's support shrunk by edge_margin (bounding
        box of the 4 sphere centres, support frame = course frame), clipped; flat courses: clip[1]."""
        out = np.full((self.n, 4), float(clip[1]))
        if self.intervals is None:
            return out
        t, _ = self.local()
        pts = sole_points(t[:, :2], t[:, 2])                         # [N, 4, 2] course frame
        k = np.maximum(self.target_planks(), 0)
        a, b, hw = self.intervals[k, 0] + edge_margin, self.intervals[k, 1] - edge_margin, self.half_width - edge_margin
        out[:, 0] = b - pts[..., 0].max(-1)
        out[:, 1] = pts[..., 0].min(-1) - a
        out[:, 2] = hw - pts[..., 1].max(-1)
        out[:, 3] = pts[..., 1].min(-1) + hw
        return np.clip(out, *clip)

    def to_layout(self):
        """planks_task.Layout of a plank course (pelvis-origin path frame of the default standing pose, D13); None
        on flat courses. MuJoCo: make_planks_env(course.to_layout(), ...) and reset at start_pose()."""
        if self.intervals is None:
            return None
        ex = self.program.get("layout")
        if ex is not None:
            return P.Layout(**ex)
        ox, oy, _ = self.start_pose()
        return P.Layout(self.intervals + FOOT0_S, origin=(ox, oy), heading=self.heading, level=self.level,
                        seed=self.seed, half_width=self.half_width)

    # -- serialisation --------------------------------------------------------------------------------------
    def to_dict(self):
        return _jsonable(dict(origin=self.origin, heading=self.heading, targets=self.targets, centre=self.centre,
                              f0=self.f0, d0=self.d0, end=self.end, cliff=self.cliff, intervals=self.intervals,
                              half_width=self.half_width, level=self.level, seed=self.seed, program=self.program))

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def step_vectors(course, start_feet=None):
    """([N] L, [N] W): each target minus the previous footprint (target i - 1; target 0: the other foot's start
    centre, default pose unless start_feet [2, 2] world is given) in the local path frame of the target."""
    t, c = course.local()
    if start_feet is None:
        other0 = np.array([0.0, -course.f0 * FOOT0_L])
    else:
        other0 = course.path_coords(np.asarray(start_feet, np.float64)[int(course.foot_of(0)) ^ 1, :2])
    prev = np.vstack((other0[None], t[:-1, :2]))
    d = t[:, :2] - prev
    cs, sn = np.cos(c[:, 2]), np.sin(c[:, 2])
    return d[:, 0] * cs + d[:, 1] * sn, t[:, 3] * (-d[:, 0] * sn + d[:, 1] * cs)


# ---- placement of a step program (canonical course frame) -----------------------------------------------------
def _place(L, dpsi, lane, h_nom, f0, stop):
    """Targets [n, 5] and centre [n, 3] (course frame) of a step program; h clipped so that W in [W_MIN, W_MAX]
    (D7). Returns (targets, centre, h, ok); ok False if the step into the stop pair leaves the width range."""
    n = len(L)
    side = f0 * (-1.0) ** np.arange(n)
    tg, ce, h = np.zeros((n, 5)), np.zeros((n, 3)), np.zeros(n)
    q, psi = np.zeros(2), 0.0
    prev = np.array([0.0, -f0 * FOOT0_L])
    ok = True
    for i in range(n):
        mid = psi + dpsi[i] / 2
        psi = psi + dpsi[i]
        q = q + L[i] * np.array([math.cos(mid), math.sin(mid)])
        nv = np.array([-math.sin(psi), math.cos(psi)])
        c = q + lane[i] * nv
        is_stop = stop and i >= n - 2
        base = side[i] * float((c - prev) @ nv)
        if i == 0:
            hi = FOOT0_L
        elif is_stop:
            hi = STOP_WIDTH / 2
            ok &= W_MIN - 1e-9 <= base + hi <= W_MAX + 1e-9
        else:
            hi = float(np.clip(h_nom[i], W_MIN - base, W_MAX - base))
        t = c + side[i] * hi * nv
        tg[i] = (t[0], t[1], wrap(psi), side[i], 1.0 if is_stop else 0.0)
        ce[i] = (c[0], c[1], wrap(psi))
        h[i], prev = hi, t
    return tg, ce, h, bool(ok)


def _uniform(rng, lo, hi):
    return float(rng.uniform(lo, hi)) if hi > lo else float(lo)


def _ramp(rng, dl):
    d0 = _uniform(rng, *RAMP_D0)
    l1 = _uniform(rng, *RAMP_L1)
    l2 = _uniform(rng, max(RAMP_L2[0], l1 - RAMP_L2[1]), min(RAMP_L2[2], l1 + dl))
    return [d0, l1, l2]


def _flat_program(lv, rng, n, stop):
    """Step lengths [n] (L_0 = d0) of one flat course, or None if a capped draw has an empty range."""
    dl, (lo, hi) = lv["dl"], lv["L"]
    L = np.zeros(n)
    L[:3] = _ramp(rng, dl)
    cap = np.full(n, hi)
    last = n - 2 if stop else n - 1
    if stop:
        cap[n - 2] = min(hi, STOP_CAPS[1], dl)
        cap[n - 3] = min(cap[n - 3], STOP_CAPS[0])
    prev, kind = L[2], lv["prog"]
    val = _uniform(rng, max(lo, prev - dl), min(hi, prev + dl)) if kind == "hold1" else None
    seg_left = 0
    for i in range(3, last + 1):
        if kind == "hold1":
            x = val
        elif kind == "piecewise":
            if seg_left == 0:
                val = _uniform(rng, max(lo, prev - dl), min(hi, prev + dl))
                seg_left = int(rng.integers(lv["seg"][0], lv["seg"][1] + 1))
            x = val
            seg_left -= 1
        else:
            x = prev if rng.random() < lv["hold"] else _uniform(rng, max(lo, prev - dl), min(hi, prev + dl))
        if x > cap[i] + 1e-12 or abs(x - prev) > dl + 1e-12:
            a, b = max(lo, prev - dl), min(cap[i], prev + dl)
            if a > b:
                return None
            x = _uniform(rng, a, b)
        L[i] = prev = x
    return L


def _gen_flat(code, lv, rng, seed):
    n = int(rng.integers(lv["n"][0], lv["n"][1] + 1))
    stop = bool(rng.random() < lv["stop"])
    f0 = 1 if rng.random() < 0.5 else -1
    turn = bool(rng.random() < lv["turn_p"])
    lanes = bool(rng.random() < lv["lane_p"])
    for tries in range(1, MAX_TRIES + 1):
        L = _flat_program(lv, rng, n, stop)
        if L is None:
            continue
        kap = np.zeros(n)
        if turn:
            i = 3
            while i < n:
                seg = int(rng.integers(4, 11))
                kap[i:i + seg] = rng.uniform(-lv["kmax"], lv["kmax"])
                i += seg
        dpsi = np.clip(kap * L, -lv["dpsi"], lv["dpsi"])
        lane = np.zeros(n)
        if lanes:
            for i in range(3, n - 1 if stop else n):
                lane[i] = np.clip(lane[i - 1] + rng.uniform(-lv["lane_step"], lv["lane_step"]), -lv["lane_max"],
                                  lv["lane_max"])
            if stop:
                lane[n - 1] = lane[n - 2]
        if "width" in lv:
            h_nom = rng.uniform(lv["width"][0] / 2, lv["width"][1] / 2, n)
        else:
            h_nom = W_nat(L) / 2 + rng.uniform(-lv["w_jit"] / 2, lv["w_jit"] / 2, n)
        tg, ce, h, ok = _place(L, dpsi, lane, h_nom, f0, stop)
        if not ok:
            continue
        prog = dict(L=L, h=h, lane=lane, dpsi=dpsi, kappa=kap, turn=turn, lanes=lanes, tries=tries)
        return Course((0.0, 0.0), 0.0, tg, ce, f0=f0, d0=L[0], end="stop" if stop else "through", level=code,
                      seed=seed, program=prog)
    raise RuntimeError(f"{code} seed {seed}: no valid course in {MAX_TRIES} tries")


def _gen_planks(code, lv, rng, seed):
    """Plank-by-plank generator (spec 4.4 algorithm, D3 stop ending)."""
    n_int = int(rng.integers(lv["planks"][0], lv["planks"][1] + 1))
    stop = bool(rng.random() < lv["stop"])
    f0 = 1 if rng.random() < 0.5 else -1
    hw = float(lv["hw"][int(rng.integers(len(lv["hw"])))])
    dl, lmax = lv["dl"], lv["Lmax"]
    (glo, ghi), (mlo, mhi), (ilo, ihi) = lv["G"], lv["m"], lv["Lin"]
    kp = np.asarray(lv["k"], np.float64)
    for tries in range(1, MAX_TRIES_PLANKS + 1):
        L = _ramp(rng, dl)
        s = [L[0], L[0] + L[1], L[0] + L[1] + L[2]]
        m_f = _uniform(rng, mlo, mhi)
        iv = [[-START_BACK, s[2] + HALF_SOLE + m_f]]
        ks, gaps, m_back, m_front, plank_of = [], [], [], [m_f], [0, 0, 0]
        bad = False
        for j in range(n_int + 1):
            final = j == n_int
            k = (4 if stop else 2) if final else int(rng.choice(3, p=kp)) + 1
            m_b, g = _uniform(rng, mlo, mhi), _uniform(rng, glo, ghi)
            lx = 2 * HALF_SOLE + m_f + m_b + g
            if lx > lmax:                                   # shrink the gap to the longest allowed step
                g -= lx - lmax
                lx = 2 * HALF_SOLE + m_f + m_b + g
                if g < glo - 1e-12:
                    bad = True
                    break
            if abs(lx - L[-1]) > dl + 1e-12:
                bad = True
                break
            L.append(lx)
            s.append(s[-1] + lx)
            start = s[-1] - HALF_SOLE - m_b
            for t in range(k - 1):
                if final and stop and t == 2:
                    x = 0.0                                 # closing step: the stop pair shares one centre
                else:
                    cap = ihi if not (final and stop) else min(ihi, STOP_CAPS[0] if t == 0 else min(STOP_CAPS[1], dl))
                    a, b = max(ilo, L[-1] - dl), min(cap, L[-1] + dl)
                    if a > b:
                        bad = True
                        break
                    x = _uniform(rng, a, b)
                L.append(x)
                s.append(s[-1] + x)
            if bad:
                break
            if final:
                end = (s[-1] + HALF_SOLE + FOOT_RADIUS + _uniform(rng, *CLIFF)) if lv["cliff"] else start + FINAL_LEN
            else:
                m_f = _uniform(rng, mlo, mhi)
                end = s[-1] + HALF_SOLE + m_f
                ks.append(k)
                m_front.append(m_f)
            iv.append([start, end])
            gaps.append(g)
            m_back.append(m_b)
            plank_of += [j + 1] * k
        if bad or len(L) > NMAX:
            continue
        n = len(L)
        L = np.asarray(L)
        h_nom = W_nat(L) / 2 + rng.uniform(-lv["w_jit"] / 2, lv["w_jit"] / 2, n)
        tg, ce, h, ok = _place(L, np.zeros(n), np.zeros(n), h_nom, f0, stop)
        if not ok or (np.abs(tg[:, 1]) + SOLE_HALF_WIDTH + EDGE_MARGIN > hw + 1e-12).any():
            continue
        prog = dict(L=L, h=h, k=ks, gaps=gaps, m_back=m_back, m_front=m_front, plank_of_target=plank_of,
                    n_int=n_int, tries=tries)
        return Course((0.0, 0.0), 0.0, tg, ce, f0=f0, d0=L[0], end="stop" if stop else "through",
                      cliff=lv["cliff"], intervals=np.asarray(iv), half_width=hw, level=code, seed=seed, program=prog)
    raise RuntimeError(f"{code} seed {seed}: no valid course in {MAX_TRIES_PLANKS} tries")


def sample_canonical(level, rng, seed=-1):
    """One course of a new level (LEVEL_CODES) in the course frame (origin (0, 0), heading 0) from a numpy
    Generator; the bank builder calls it with rng = default_rng([seed, crc32(level)])."""
    lv = LEVELS_FS[level]
    return (_gen_flat if lv["kind"] == "flat" else _gen_planks)(level, lv, rng, int(seed))


def level_rng(level, seed):
    return np.random.default_rng([int(seed), zlib.crc32(level.encode())])


def sample_course(level, seed, origin=(0.0, 0.0), heading=None):
    """The course of (level, seed) of a new level, placed at origin (the start foot midpoint) with heading
    U[-pi, pi) drawn after the course (so heading 0 gives the bank's course of that seed)."""
    if level not in LEVELS_FS:
        raise ValueError(f"{level}: not a new level (old layouts: old_course)")
    rng = level_rng(level, seed)
    c = sample_canonical(level, rng, seed)
    h = rng.uniform(-np.pi, np.pi) if heading is None else float(heading)
    return c.placed(origin, h)


# ---- validation of a course ------------------------------------------------------------------------------------
def check_course(course, tol=1e-6):
    """List of rule violations (empty = valid): generic rules for every course, the level's rules of spec 4.2-4.4
    for new levels (program values when the course carries its program, else the geometry; a curved course without
    its program skips the step-length rules, whose program values differ from the geometry by up to ~2 cm)."""
    v = []
    t, c = course.local()
    n = len(t)
    lv = LEVELS_FS.get(course.level)
    if n < 3 or n > NMAX:
        v.append(f"n {n}")
    if course.f0 not in (1, -1) or not np.array_equal(t[:, 3], course.f0 * (-1.0) ** np.arange(n)):
        v.append("feet do not alternate from f0")
    want_stop = np.zeros(n)
    if course.end == "stop":
        want_stop[-2:] = 1
    if not np.array_equal(t[:, 4], want_stop):
        v.append("stop flags")
    if (np.abs(wrap(t[:, 2] - c[:, 2])) > tol).any():
        v.append("target yaw != path heading")
    L, W = step_vectors(course)
    prog = course.program if isinstance(course.program, dict) else {}
    Lp = np.asarray(prog["L"], np.float64) if len(prog.get("L", ())) == n else L
    dpsi = wrap(np.diff(np.concatenate(([0.0], c[:, 2]))))
    curved = bool((np.abs(dpsi) > 1e-9).any())
    if not curved and (np.abs(Lp - L) > tol).any():
        v.append("program L != geometry")
    exact_L = not curved or len(prog.get("L", ())) == n     # curved courses need their program for the L rules
    if (W < W_MIN - tol).any() or (W > W_MAX + tol).any():
        v.append(f"width out of range {W.min():.3f}..{W.max():.3f}")
    d0_hi = RAMP_D0[1] if lv is not None else RAMP_D0[1] + GRID     # planner start set: 0.001-0.061 m
    if not (RAMP_D0[0] - tol <= Lp[0] <= d0_hi + tol) or abs(Lp[0] - course.d0) > tol:
        v.append("d0")
    if Lp[1] > RAMP_L1[1] + tol or Lp[2] > RAMP_L2[2] + tol:
        v.append("start steps above 0.20 / 0.28")
    if course.end == "stop":
        if np.abs(c[-1] - c[-2]).max() > tol or abs(Lp[-1]) > tol or abs(W[-1] - STOP_WIDTH) > tol:
            v.append("stop pair")
        if Lp[-2] > STOP_CAPS[1] + tol or Lp[-3] > STOP_CAPS[0] + tol:
            v.append("stop approach steps")
    if lv is not None:
        dl = lv["dl"]
        if exact_L:
            if not (RAMP_L1[0] - tol <= Lp[1]) or not (max(RAMP_L2[0], Lp[1] - RAMP_L2[1]) - tol <= Lp[2]
                                                       <= min(RAMP_L2[2], Lp[1] + dl) + tol):
                v.append("start ramp")
            if (np.abs(np.diff(Lp)[1:]) > dl + tol).any():
                v.append("dL")
            if course.end == "stop" and Lp[-2] > dl + tol:
                v.append("closing step dL")
        if lv["kind"] == "flat":
            v += _check_flat(course, lv, Lp, dpsi, n, tol, exact_L)
        else:
            v += _check_planks(course, lv, t, Lp, tol)
    elif course.intervals is not None:
        v += _check_supports(course, t, float(prog.get("m", 0.0)), tol)
    return v


def _check_flat(course, lv, Lp, dpsi, n, tol, exact_L=True):
    v = []
    if course.intervals is not None or course.cliff:
        v.append("flat level with planks")
    if not lv["n"][0] <= n <= lv["n"][1]:
        v.append("n range")
    if lv["stop"] == 0 and course.end == "stop":
        v.append("stop on a level without stops")
    prog_end = n - 2 if course.end == "stop" else n - 1
    lo, hi = lv["L"]
    x = Lp[3:prog_end + 1]
    if exact_L and ((x < lo - tol).any() or (x > hi + tol).any()):
        v.append("program L range")
    if (np.abs(dpsi) > lv["dpsi"] + tol).any():
        v.append("turn per step")
    prog = course.program if isinstance(course.program, dict) else {}
    if "lane" in prog:
        lane = np.asarray(prog["lane"])
        if (np.abs(np.diff(lane)) > lv["lane_step"] + tol).any() or (np.abs(lane) > lv["lane_max"] + tol).any():
            v.append("lanes")
    return v


def _sphere_margins(course, t):
    """[N] (k, back margin, front margin, |y| + 0.03) of each target's spheres on its own plank (course frame)."""
    pts = sole_points(t[:, :2], t[:, 2])
    k = course.support_index(course.targets[:, :2])
    kk = np.maximum(k, 0)
    back = pts[..., 0].min(-1) - course.intervals[kk, 0]
    front = course.intervals[kk, 1] - pts[..., 0].max(-1)
    side = np.abs(pts[..., 1]).max(-1)
    return k, back, front, side


def _check_supports(course, t, m, tol):
    v = []
    iv = course.intervals
    if (iv[:, 1] <= iv[:, 0]).any() or (iv[1:, 0] - iv[:-1, 1] < GAP_MIN - tol).any():
        v.append("intervals not sorted or gap < 0.03")
    k, back, front, side = _sphere_margins(course, t)
    if (k < 0).any():
        v.append("target over a gap")
    if (back < m - 1e-3).any() or (front < m - 1e-3).any():
        v.append("sphere margin")
    if (side + EDGE_MARGIN > course.half_width + tol).any():
        v.append("lateral")
    return v


def _check_planks(course, lv, t, Lp, tol):
    v = []
    iv = course.intervals
    if iv is None:
        return ["plank level without planks"]
    if len(iv) > PMAX or not lv["planks"][0] <= len(iv) - 2 <= lv["planks"][1]:
        v.append("number of planks")
    if not any(abs(course.half_width - h) < 1e-6 for h in lv["hw"]):
        v.append("half width")
    if course.cliff != lv["cliff"]:
        v.append("cliff flag")
    if (iv[:, 1] <= iv[:, 0]).any():
        v.append("plank length")
    g = iv[1:, 0] - iv[:-1, 1]
    if (g < lv["G"][0] - tol).any() or (g > lv["G"][1] + tol).any():
        v.append(f"gap range {g.min():.3f}..{g.max():.3f}")
    if abs(iv[0, 0] + START_BACK) > tol:
        v.append("start platform rear")
    k, back, front, side = _sphere_margins(course, t)
    n = len(t)
    if (k < 0).any():
        return v + ["target over a gap"]
    counts = np.bincount(k, minlength=len(iv))
    if (counts[1:-1] < 1).any() or (counts[1:-1] > 3).any():
        v.append("footprints per plank")
    if counts[0] != 3 or counts[-1] != (4 if course.end == "stop" else 2) or (np.diff(k) < 0).any():
        v.append("footprints on the platforms")
    mlo = lv["m"][0]
    cliff_front = np.zeros(n, bool)
    if course.cliff:
        cliff_front[k == len(iv) - 1] = True
        clear = iv[-1, 1] - (t[-2:, 0].max() + HALF_SOLE + FOOT_RADIUS)
        if not CLIFF[0] - tol <= clear <= CLIFF[1] + tol:
            v.append("cliff clearance")
        if (front[cliff_front] < CLIFF[0] + FOOT_RADIUS - 1e-3).any():
            v.append("cliff front margin")
    elif abs(iv[-1, 1] - iv[-1, 0] - FINAL_LEN) > tol:
        v.append("final platform length")
    if (back < mlo - 1e-3).any() or (front[~cliff_front] < mlo - 1e-3).any():
        v.append("sphere margin")
    # true margins at the edges next to every gap: last footprint of plank j (front), first of plank j + 1 (back)
    for j in range(len(iv) - 1):
        mf = front[k == j].min()
        mb = back[k == j + 1].min()
        if not (mlo - 1e-3 <= mf <= lv["m"][1] + tol and mlo - 1e-3 <= mb <= lv["m"][1] + tol):
            v.append(f"true margin at gap {j}")
            break
    if (side + EDGE_MARGIN > course.half_width + tol).any():
        v.append("lateral")
    cross = np.diff(k) > 0
    steps = Lp[1:]
    if (steps[cross] > lv["Lmax"] + tol).any():
        v.append("crossing step above L max")
    within = ~cross
    within[:2] = False                                   # ramp steps 1, 2
    if course.end == "stop":
        within[-1] = False                               # closing step
    x = steps[within]
    if (x < lv["Lin"][0] - tol).any() or (x > lv["Lin"][1] + tol).any():
        v.append("within-plank L range")
    return v


# ---- planner for the old layouts (spec 4.5) -------------------------------------------------------------------
def plan_dp(intervals, m, l_max, dl_max, end="through", l_first=(0.20, 0.28)):
    """Min-cost forward-facing footstep plan on plank intervals [P, 2] (pelvis-origin path frame of a
    planks_task.Layout): footprint s positions (1 cm grid) or None. Feasibility = the reachability DP of
    S/scripts/spec_v4_feasibility.py exactly; cost D11. end "through": the first footprint >= 0.30 m into the final
    platform; "stop": the first footprint on the final platform (then planned_stop_tail)."""
    iv = np.asarray(intervals, np.float64)
    lo, hi = iv[:, 0] + HALF_SOLE + m, iv[:, 1] - HALF_SOLE - m
    xs = np.arange(-0.2, iv[-1, 0] + 1.0, GRID)
    npos = len(xs)
    ok = np.zeros(npos, bool)
    marg = np.full(npos, -np.inf)
    for a, b in zip(lo, hi):
        inside = (xs >= a - 1e-9) & (xs <= b + 1e-9)
        ok |= inside
        marg = np.where(inside, np.clip(np.minimum(xs - a, b - xs), 0.0, None), marg)
    pen = 2.0 * (0.05 - np.minimum(marg, 0.05))
    nL = int(round(l_max / GRID)) + 1
    Ls = np.arange(nL) * GRID
    dmax = int(round(dl_max / GRID))
    caps = [int(round(l_first[0] / GRID)), int(round(l_first[1] / GRID)), nL - 1]
    j, jn = np.arange(nL)[:, None], np.arange(nL)[None, :]
    base = (Ls[jn] - 0.25) ** 2 + 4.0 * (Ls[jn] - Ls[j]) ** 2
    T = [np.where((jn >= 1) & (np.abs(jn - j) <= dmax) & (jn <= min(nL - 1, caps[k])), base, np.inf) for k in range(3)]
    i0 = int(round((FOOT0_S - xs[0]) / GRID))
    start = np.zeros(npos, bool)
    start[i0:i0 + 7] = True
    start &= ok
    done = ok & (xs >= (iv[-1, 0] + 0.30 if end == "through" else iv[-1, 0]))
    C = np.full((3, npos, nL), np.inf)
    BK = np.full((3, npos, nL), -1, np.int8)
    BJ = np.full((3, npos, nL), -1, np.int16)
    C[0, start, 0] = pen[start]
    steps = np.arange(nL)
    for i in range(npos):
        if done[i]:
            continue
        for k in range(3):
            c = C[k, i]
            fin = np.flatnonzero(np.isfinite(c))
            if not len(fin):
                continue
            M = c[fin, None] + T[k][fin]
            arg = M.argmin(0)
            best = M[arg, steps]
            ii = i + steps
            sel = np.isfinite(best) & (ii < npos)
            sel[sel] &= ok[ii[sel]]
            jj = np.flatnonzero(sel)
            if not len(jj):
                continue
            kn = min(k + 1, 2)
            new = best[jj] + pen[ii[jj]]
            better = new < C[kn, ii[jj], jj]
            jj = jj[better]
            C[kn, i + jj, jj] = new[better]
            BK[kn, i + jj, jj] = k
            BJ[kn, i + jj, jj] = fin[arg[jj]]
    cand = np.where(done[None, :, None], C, np.inf)
    if not np.isfinite(cand).any():
        return None
    k, i, jl = np.unravel_index(int(cand.argmin()), cand.shape)
    path = [i]
    while k != 0:
        k, i, jl = int(BK[k, i, jl]), i - jl, int(BJ[k, i, jl])
        path.append(i)
    return xs[np.asarray(path[::-1])]


def planned_stop_tail(s, intervals, m):
    """Append the stop tail of D12 to footprints s (layout frame): a = min(0.28, L_last), b = min(0.15, a), closing
    pair. None if a new footprint leaves the final platform's safe window."""
    s = list(s)
    a = min(STOP_CAPS[0], s[-1] - s[-2])
    b = min(0.15, a)
    s += [s[-1] + a, s[-1] + a + b, s[-1] + a + b]
    iv = np.asarray(intervals)
    lo, hi = iv[-1, 0] + HALF_SOLE + m, iv[-1, 1] - HALF_SOLE - m
    return s if all(lo - 1e-9 <= x <= hi + 1e-9 for x in s[-3:]) else None


def course_from_layout(layout, s_targets, f0=1, end="through", level=None, seed=None, program=None):
    """Course on a planks_task.Layout from footprint positions s_targets along its path (pelvis-origin frame).
    Straight; widths W_nat(L) / 2 (target 0: FOOT0_L; stop pair: 0.12); d0 = s_0 - FOOT0_S. The course keeps the
    exact layout (to_layout() returns it)."""
    s = np.asarray(s_targets, np.float64) - FOOT0_S
    L = np.diff(np.concatenate(([0.0], s)))
    stop = end == "stop"
    tg, ce, h, _ = _place(L, np.zeros(len(L)), np.zeros(len(L)), W_nat(L) / 2, f0, stop)
    lay = dict(intervals=layout.intervals.tolist(), origin=list(layout.origin), heading=layout.heading,
               level=layout.level, seed=int(layout.seed), half_width=layout.half_width, pit_depth=layout.pit_depth)
    prog = dict(program or {}, L=L, h=h, layout=lay)
    c = Course((0.0, 0.0), 0.0, tg, ce, f0=f0, d0=L[0], end=end, cliff=False,
               intervals=layout.intervals - FOOT0_S, half_width=layout.half_width,
               level=layout.level if level is None else level, seed=layout.seed if seed is None else seed,
               program=prog)
    return c.placed(layout.world_xy(FOOT0_S, 0.0), layout.heading)


def plan_on_layout(layout, variants=PLANNER_VARIANTS, end="through"):
    """(Course | None, variant name | None): the first variant (name, m, L max, dL max) with a plan (spec 4.5).
    f0 = left; the course keeps the layout's origin, heading, level and seed."""
    for name, m, lmax, dlmax in variants:
        s = plan_dp(layout.intervals, m, lmax, dlmax, end=end)
        if s is not None and end == "stop":
            s = planned_stop_tail(s, layout.intervals, m)
        if s is not None:
            return course_from_layout(layout, s, f0=1, end=end,
                                      program=dict(variant=name, m=m, L_max=lmax, dL_max=dlmax)), name
    return None, None


def old_course(level, seed, variants=PLANNER_VARIANTS):
    """(Course | None, variant): planks_task.sample_layout(level, seed) of an old level + plan_on_layout (D20)."""
    return plan_on_layout(P.sample_layout(level, seed), variants)


def eval_course(level, seed):
    """(Course | None, variant | None) of any evaluation level: new levels sample_course(level, seed) (random
    heading, origin 0), old levels old_course."""
    if level in OLD_LEVELS:
        return old_course(level, seed)
    return sample_course(level, seed), None


# ---- showcases (spec 4.6) --------------------------------------------------------------------------------------
SHOWCASES = ("rhythm", "ramp", "lane_change", "mixed_wide", "mixed_narrow_edge", "step_change")
SHOW_PLANKS = (0.25, 0.48, 0.24, 0.70, 0.26, 0.45, 0.30, 0.62)
SHOW_GAPS = (0.06, 0.09, 0.05, 0.08, 0.12, 0.07, 0.10, 0.05, 0.11)    # start | 8 planks | final
SHOW_START_EDGE = 0.50                                                # start platform end (course frame)
SHOW_VARIANTS = (("VS", 0.03, 0.36, 0.20),) + PLANNER_VARIANTS


def flat_course(steps, d0=0.03, f0=1, end="stop", lanes=None, level="", seed=-1, program=None):
    """Straight flat course from step lengths L_1.. (after target 0 = f0 start foot + d0); end "stop" appends the
    closing step (the last given step lands the first stop target). lanes: per target (len = len(steps) + 1)."""
    L = np.concatenate(([d0], np.asarray(steps, np.float64), [0.0] if end == "stop" else []))
    n = len(L)
    lane = np.zeros(n)
    if lanes is not None:
        lane[:len(lanes)] = lanes
        lane[len(lanes):] = lanes[-1]
    tg, ce, h, ok = _place(L, np.zeros(n), lane, W_nat(L) / 2, f0, end == "stop")
    if not ok:
        raise ValueError("the step into the stop pair leaves the width range")
    return Course((0.0, 0.0), 0.0, tg, ce, f0=f0, d0=d0, end=end, level=level, seed=seed,
                  program=dict(program or {}, L=L, h=h, lane=lane))


def _show_layout(half_width=P.HALF_WIDTH):
    rows = [(-START_BACK, SHOW_START_EDGE)]
    s = SHOW_START_EDGE
    for plank, gap in zip(SHOW_PLANKS + (FINAL_LEN,), SHOW_GAPS):
        rows.append((s + gap, s + gap + plank))
        s += gap + plank
    return P.Layout(np.asarray(rows) + FOOT0_S, origin=(-FOOT0_S, 0.0), heading=0.0, level="showcase",
                    half_width=half_width)


def showcase(name):
    """One fixed showcase course (spec 4.6), course origin (0, 0), heading 0."""
    if name == "rhythm":
        return flat_course([0.10, 0.10, 0.20, 0.30, 0.30, 0.20, 0.10, 0.10] * 2, level=name)
    if name == "ramp":
        up = [0.06 + 0.04 * i for i in range(8)]
        return flat_course(up + up[-2::-1], level=name)
    if name == "lane_change":
        steps = [0.12, 0.20, 0.20, 0.20, 0.20, 0.20] + [0.20] * 5 + [0.20] * 5 + [0.15]
        lanes = [0.0] * 7 + [0.05 * (i + 1) for i in range(5)] + [0.25] * 6
        return flat_course(steps, lanes=lanes, level=name)
    if name == "step_change":
        return flat_course([0.15] * 6 + [0.30] * 6 + [0.15] * 6, end="through", level=name)
    if name in ("mixed_wide", "mixed_narrow_edge"):
        lay = _show_layout()
        c, var = plan_on_layout(lay, SHOW_VARIANTS, end="stop")
        if c is None:
            raise RuntimeError("no plan for the showcase planks")
        if name == "mixed_wide":
            c.level = name
            return c
        s = np.asarray(c.program["L"]).cumsum() + FOOT0_S                      # footprints, layout frame
        iv = lay.intervals.copy()
        iv[-1, 1] = s[-1] + HALF_SOLE + FOOT_RADIUS + 0.03                     # cliff 3 cm in front of the toes
        cut = P.Layout(iv, origin=lay.origin, heading=0.0, level="showcase", half_width=0.25)
        out = course_from_layout(cut, s, f0=1, end="stop", level=name, program=dict(variant=var, m=c.program["m"]))
        out.cliff = True
        return out
    raise ValueError(f"unknown showcase {name}; choose from {SHOWCASES}")


# ---- banks -----------------------------------------------------------------------------------------------------
BANK_KEYS = ("level", "seed", "n", "targets", "centre", "f0", "d0", "end", "cliff", "n_planks", "n_intervals",
             "intervals", "half_width")


def courses_to_arrays(courses):
    """Bank arrays (float32 course-frame values) of a list of courses (module docstring)."""
    M = len(courses)
    out = dict(level=np.zeros(M, np.int8), seed=np.zeros(M, np.int64), n=np.zeros(M, np.int16),
               targets=np.zeros((M, NMAX, 5), np.float32), centre=np.zeros((M, NMAX, 3), np.float32),
               f0=np.zeros(M, np.int8), d0=np.zeros(M, np.float32), end=np.zeros(M, np.int8),
               cliff=np.zeros(M, bool), n_planks=np.zeros(M, np.int8), n_intervals=np.zeros(M, np.int8),
               intervals=np.tile(np.asarray(PAD_INTERVAL, np.float32), (M, PMAX, 1)),
               half_width=np.zeros(M, np.float32))
    for i, c in enumerate(courses):
        t, ce = c.local()
        n = len(t)
        if n > NMAX:
            raise ValueError(f"course {i}: {n} targets > {NMAX}")
        out["level"][i] = LEVEL_CODES.index(c.level) if c.level in LEVEL_CODES else -1
        out["seed"][i], out["n"][i], out["f0"][i], out["d0"][i] = c.seed, n, c.f0, c.d0
        out["targets"][i, :n], out["targets"][i, n:] = t, t[-1]
        out["centre"][i, :n], out["centre"][i, n:] = ce, ce[-1]
        out["end"][i], out["cliff"][i], out["half_width"][i] = END_CODES[c.end], c.cliff, c.half_width
        if c.intervals is not None:
            p = len(c.intervals)
            out["intervals"][i, :p] = c.intervals
            out["n_intervals"][i], out["n_planks"][i] = p, p - 2
    return out


def course_from_arrays(arrays, i, origin=(0.0, 0.0), heading=0.0):
    """Course i of bank arrays (float64), placed at origin / heading (default: the course frame)."""
    a = arrays
    n, p = int(a["n"][i]), int(a["n_intervals"][i])
    lv = int(a["level"][i])
    c = Course((0.0, 0.0), 0.0, np.asarray(a["targets"][i, :n], np.float64),
               np.asarray(a["centre"][i, :n], np.float64), f0=int(a["f0"][i]), d0=float(a["d0"][i]),
               end=END_NAMES[int(a["end"][i])], cliff=bool(a["cliff"][i]),
               intervals=np.asarray(a["intervals"][i, :p], np.float64) if p else None,
               half_width=float(a["half_width"][i]), level=LEVEL_CODES[lv] if 0 <= lv < len(LEVEL_CODES) else "",
               seed=int(a["seed"][i]))
    return c if (origin is None or (np.allclose(origin, 0) and heading == 0)) else c.placed(origin, heading)


def save_bank(path, arrays, meta=None):
    np.savez_compressed(path, **arrays, meta=np.array(json.dumps(_jsonable(meta or {}))))


def load_bank(path):
    """Bank arrays from save_bank: float arrays as float64; 'meta' as a dict."""
    with np.load(path) as z:
        out = {k: (z[k].astype(np.float64) if z[k].dtype == np.float32 else z[k]) for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return out


def place_arrays(targets, centre, origin, heading):
    """Rigid placement of bank arrays: targets [M, N, 5], centre [M, N, 3] (course frame), origin [M, 2],
    heading [M] -> world (targets, centre) float64 (the reference of the torch placement)."""
    t, c = np.array(targets, np.float64), np.array(centre, np.float64)
    o, h = np.asarray(origin, np.float64), np.asarray(heading, np.float64)
    cs, sn = np.cos(h)[:, None], np.sin(h)[:, None]
    for a in (t, c):
        x, y = a[..., 0].copy(), a[..., 1].copy()
        a[..., 0] = o[:, 0, None] + cs * x - sn * y
        a[..., 1] = o[:, 1, None] + sn * x + cs * y
        a[..., 2] = wrap(a[..., 2] + h[:, None])
    return t, c


# ---- the scorer (spec 5) ---------------------------------------------------------------------------------------
def score_cfg(cfg=None):
    """SCORE_DEFAULTS updated by a score dict, or by an env config (its 'score' block and rules.edge_margin)."""
    out = dict(SCORE_DEFAULTS)
    if cfg is None:
        return out
    if "score" in cfg:
        out.update(cfg["score"])
        em = (cfg.get("rules") or {}).get("edge_margin")
        if em is not None:
            out["edge_margin"] = em
    else:
        out.update(cfg)
    unknown = set(out) - set(SCORE_DEFAULTS)
    if unknown:
        raise KeyError(f"unknown score keys {sorted(unknown)}")
    return out


class FootstepProgress:
    """Numpy reference scorer of one episode (spec 5); the MuJoCo runner and the torch FootstepTracker follow it.

    FootstepProgress(course, cfg, start_feet [2, 2], start_yaw [2], t0_step, start_spheres=None)
      course: the PLACED course (target 0 already set by with_target0). cfg: score dict or env config
      (score_cfg). start_feet / start_yaw: world foot centres and yaws at the start (default: the course's default
      pose); they are the first "last placements". t0_step: the step index of the first update() that belongs to
      the policy (NRD: 0; MuJoCo: the hand-over control step). start_spheres [2, 4, >=2]: the sphere centres before
      the first update (the previous state of the plank counters; default: the first update's own).

    update(step, pelvis_xy, pelvis_yaw, foot_centres [2, >=2], foot_yaws [2], spheres [2, 4, >=2], gate [2],
           pelvis_speed_xy, wz) after control step `step`, with the state at the END of the step (world; NRD: the
      newest state; MuJoCo: avg_state) and the gate of the step (NRD: switch; MuJoCo: contact in this step OR the
      previous one). pelvis_speed_xy: planar pelvis velocity [2] (or its norm); wz: pelvis yaw rate (rad/s).

    Per update, in this order:
     1. plank counters (PlanksProgress rule, true planks): a raw touchdown = gate on after >= min_swing_steps
        steps off; counted, over a gap if any sphere centre is off the planks; a later contact step with a sphere
        off the planks in this AND the previous state = unsupported step. Swing counters reset on any gate-on step.
     2. slip of scored footprints: foot centre 2 steps after the touchdown step minus the placement.
     3. confirmation (confirm true): a raw touchdown of step k is pending; at step k + 1 it is confirmed if the
        gate is still on, else dropped (flicker). Placement = foot centre and yaw of the state of step k.
     4. confirmed touchdowns are processed, the expected foot first (spec 5.3 event table, D14, D16):
        stop phase: |c - P_f| > r_adj -> overrun, else a stop adjustment; P_f = c.
        else, with p = pointer, e = |c - T_p|, start window sw = (k - t0_step < start_window_s / dt) and p < 2,
        r = r_start if sw else r_adj: if |c - P_f| <= r and (f != f_p or e > tol) -> re-step (free in the start
        window, else counted, > restep_budget -> restep_budget); elif f == f_p -> scored (p += 1; e > tol_fail ->
        far_miss; last target: through -> finished, completed if e <= tol_fail; stop -> stop phase); else wrong
        foot. P_f = c in every case.
     5. stop phase: still counter += 1 if both gates on, |v_xy| < stop_v and |wz| < stop_wz, else 0; >= stop_still_s
        / dt -> stop success (completed); step - stop touchdown step >= stop_limit_s / dt -> no_stop.
     6. stall (not in the stop phase): step - last scored touchdown step (start: t0_step) >= stall_s / dt.
     7. drift: l and heading error of the pelvis against the local line of target min(p, N - 1).
    Failures in STRICT_ONLY need cfg strict; the first failure or completion ends the scoring (done).

    update() returns an events dict: step, raw_touchdown [2], touchdowns (list of per-touchdown dicts: foot, step,
    kind in scored / restep / restep_free / wrong_foot / overrun / stop_adjust, target, err, err_along, err_across,
    yaw_err, hit, far, window), counts scored, hits, resteps, resteps_free, wrong_foot, far_miss, flicker,
    overrun; flags stop_phase (entered now), stop_success, no_stop, finished, completed; fail (reason or None);
    done; pointer. state() and metrics() give the rest."""

    def __init__(self, course, cfg=None, start_feet=None, start_yaw=None, t0_step=0, start_spheres=None):
        self.course = course
        self.cfg = cfg = score_cfg(cfg)
        dt = float(cfg["dt"])
        self.dt = dt
        st = lambda x: int(round(float(x) / dt))
        self.sw_steps, self.stall_steps = st(cfg["start_window_s"]), st(cfg["stall_s"])
        self.still_steps, self.stop_limit_steps = st(cfg["stop_still_s"]), st(cfg["stop_limit_s"])
        self.min_swing, self.strict, self.confirm = int(cfg["min_swing_steps"]), bool(cfg["strict"]), bool(cfg["confirm"])
        self.N = course.n
        self.T, self.C = course.targets, course.centre
        self.foot = course.foot_of()
        self.t0 = int(t0_step)
        sf = course.start_feet() if start_feet is None else np.asarray(start_feet, np.float64)[:, :2]
        sy = np.full(2, course.heading) if start_yaw is None else np.asarray(start_yaw, np.float64)
        self.start_place = np.column_stack((sf, sy))
        self.last_place = self.start_place.copy()
        self.plank_of_target = course.target_planks()
        self.L_plan, self.W_plan = step_vectors(course, start_feet=sf)
        # scoring state
        self.ptr, self.restep, self.step, self._now = 0, 0, self.t0 - 1, self.t0 - 1
        self.last_score_step = self.t0
        self.last_td_step = np.full(2, self.t0 - 10 ** 6)
        self.last_err = np.zeros(2)
        self.pending = [None, None]
        self.in_stop, self.stop_t0, self.still, self.no_stop_flag = False, None, 0, False
        self.done, self.completed, self.finished, self.fail, self.end_step, self.stop_step = (False, False, False,
                                                                                              None, None, None)
        self.counts = dict(scored=0, resteps=0, resteps_free=0, wrong_foot=0, far_miss=0, flickers=0, overruns=0)
        self.rows = []
        # plank counters (PlanksProgress rule) and drift
        self.off = np.zeros(2, np.int64)
        self.touchdowns = self.touchdowns_over_gap = self.unsupported_steps = 0
        self.prev_under = None if start_spheres is None else self._under(start_spheres)
        self.max_abs_l, self.max_abs_dpsi, self.sum_abs_dpsi, self.n_drift = 0.0, 0.0, 0.0, 0

    # -- helpers ----------------------------------------------------------------------------------------------
    def _under(self, spheres):
        return self.course.support_index(np.asarray(spheres, np.float64)[..., :2])

    def expected_foot(self):
        return int(self.foot[min(self.ptr, self.N - 1)])

    def start_window(self, step=None):
        k = self.step if step is None else step
        return (k - self.t0) < self.sw_steps and self.ptr < 2

    def drift(self, pelvis_xy, pelvis_yaw):
        """(l, dpsi): pelvis offset from the line through C_p along t_p (left positive) and wrap(yaw - psi_p)."""
        c = self.C[min(self.ptr, self.N - 1)]
        d = np.asarray(pelvis_xy, np.float64)[:2] - c[:2]
        return float(-d[0] * math.sin(c[2]) + d[1] * math.cos(c[2])), float(wrap(pelvis_yaw - c[2]))

    def _local_err(self, p, xy):
        c = self.C[p]
        d = np.asarray(xy) - self.T[p, :2]
        cs, sn = math.cos(c[2]), math.sin(c[2])
        return float(np.hypot(*d)), float(d[0] * cs + d[1] * sn), float(-d[0] * sn + d[1] * cs)

    def _fail(self, reason, ev, step):
        if self.done:
            return
        self.fail, self.done, self.end_step = reason, True, step
        ev["fail"] = reason

    # -- the update -------------------------------------------------------------------------------------------
    def update(self, step, pelvis_xy, pelvis_yaw, foot_centres, foot_yaws, spheres, gate, pelvis_speed_xy=0.0,
               wz=0.0):
        step = int(step)
        gate = np.asarray(gate).astype(bool).reshape(2)
        fc = np.asarray(foot_centres, np.float64)[:, :2]
        fy = np.asarray(foot_yaws, np.float64).reshape(2)
        sp = np.asarray(spheres, np.float64)[..., :2]
        ev = dict(step=step, raw_touchdown=np.zeros(2, bool), touchdowns=[], scored=0, hits=0, resteps=0,
                  resteps_free=0, wrong_foot=0, far_miss=0, flicker=0, overrun=0, stop_phase=False,
                  stop_success=False, no_stop=False, finished=False, completed=False, fail=None)
        # 1. plank counters
        under = self._under(sp)
        prev = under if self.prev_under is None else self.prev_under
        for f in range(2):
            if gate[f]:
                bad = bool((under[f] < 0).any())
                if self.off[f] >= self.min_swing:
                    ev["raw_touchdown"][f] = True
                    self.touchdowns += 1
                    self.touchdowns_over_gap += bad
                elif bad and (prev[f] < 0).any():
                    self.unsupported_steps += 1
                self.off[f] = 0
            else:
                self.off[f] += 1
        self.prev_under = under
        # 2. slip
        for r in self.rows:
            if r["settle_step"] == step:
                r["slip"] = float(np.hypot(*(fc[r["foot"]] - r["place"])))
        # 3. confirmation
        active = step >= self.t0
        todo = []
        if self.confirm:
            for f in range(2):
                if self.pending[f] is not None:
                    if gate[f]:
                        todo.append(self.pending[f])
                    else:
                        ev["flicker"] += 1
                        self.counts["flickers"] += 1
                        # D15b: a one-step contact dropout right after a landing (on, off, on: seen in the NRD only;
                        # the MuJoCo label is an OR of two steps) must not lose the landing: the next on-step counts
                        # as a new raw touchdown
                        self.off[f] = max(self.off[f], self.min_swing)
                    self.pending[f] = None
        new = [(f, step, fc[f].copy(), float(fy[f]), sp[f].copy()) for f in range(2)
               if ev["raw_touchdown"][f] and active]
        if self.confirm:
            for td in new:
                self.pending[td[0]] = td
        else:
            todo = new
        # 4. processing, the expected foot first
        self._now = step
        if active and not self.done and todo:
            fp = self.expected_foot()
            todo.sort(key=lambda td: td[0] != fp)
            for td in todo:
                if self.done:
                    break
                self._process(td, ev)
        # 5. stop phase
        if active and not self.done and self.in_stop:
            v = np.atleast_1d(np.asarray(pelvis_speed_xy, np.float64))
            speed = float(np.hypot(v[0], v[1])) if v.size >= 2 else float(abs(v[0]))
            self.still = self.still + 1 if (gate.all() and speed < self.cfg["stop_v"]
                                            and abs(float(wz)) < self.cfg["stop_wz"]) else 0
            if self.still >= self.still_steps:
                self.completed = self.done = True
                self.stop_step = self.end_step = step
                ev["stop_success"] = ev["completed"] = True
            elif step - self.stop_t0 >= self.stop_limit_steps and not self.no_stop_flag:
                self.no_stop_flag = ev["no_stop"] = True
                if self.strict:
                    self._fail("no_stop", ev, step)
        # 6. stall
        if active and not self.done and not self.in_stop and step - self.last_score_step >= self.stall_steps:
            self._fail("stall", ev, step)
        # 7. drift
        if active:
            l, dpsi = self.drift(pelvis_xy, pelvis_yaw)
            self.max_abs_l = max(self.max_abs_l, abs(l))
            self.max_abs_dpsi = max(self.max_abs_dpsi, abs(dpsi))
            self.sum_abs_dpsi += abs(dpsi)
            self.n_drift += 1
        self.step = step
        ev["done"], ev["pointer"] = self.done, self.ptr
        return ev

    def _process(self, td, ev):
        f, k, c, yaw, sph = td
        cfg = self.cfg
        rec = dict(foot=f, step=k, kind="", target=-1, err=float("nan"), err_along=float("nan"),
                   err_across=float("nan"), yaw_err=float("nan"), hit=False, far=False, window=False)
        self.last_td_step[f] = k
        if self.in_stop:
            if np.hypot(*(c - self.last_place[f, :2])) > cfg["r_adj"]:
                rec["kind"] = "overrun"
                ev["overrun"] += 1
                self.counts["overruns"] += 1
                if self.strict:
                    self._fail("overrun", ev, self._now)
            else:
                rec["kind"] = "stop_adjust"
            self.last_place[f] = (c[0], c[1], yaw)
            ev["touchdowns"].append(rec)
            return
        p = self.ptr
        fp = int(self.foot[p])
        e, along, across = self._local_err(p, c)
        sw = self.start_window(k)
        near = np.hypot(*(c - self.last_place[f, :2])) <= (cfg["r_start"] if sw else cfg["r_adj"])
        if near and (f != fp or e > cfg["tol"]):
            if sw:
                rec["kind"] = "restep_free"
                ev["resteps_free"] += 1
                self.counts["resteps_free"] += 1
            else:
                rec["kind"] = "restep"
                ev["resteps"] += 1
                self.counts["resteps"] += 1
                self.restep += 1
                if self.restep > cfg["restep_budget"] and self.strict:
                    self._fail("restep_budget", ev, self._now)
        elif f == fp:
            other = self.last_place[1 - f, :2]
            cs, sn = math.cos(self.C[p, 2]), math.sin(self.C[p, 2])
            d = c - other
            win = False
            if self.course.is_planks:
                pts = np.asarray(sph)[:, :2]
                win = bool((self.course.support_index(pts, margin=cfg["edge_margin"]) == self.plank_of_target[p]).all())
            rec.update(kind="scored", target=p, err=e, err_along=along, err_across=across,
                       yaw_err=float(wrap(yaw - self.T[p, 2])), hit=e <= cfg["tol"], far=e > cfg["tol_fail"],
                       window=win)
            pk = self.plank_of_target
            self.rows.append(dict(target=p, foot=f, side=int(self.T[p, 3]), step=k, e=e, along=along,
                                  across=across, yaw_err=rec["yaw_err"], L_plan=float(self.L_plan[p]),
                                  W_plan=float(self.W_plan[p]), L_act=float(d[0] * cs + d[1] * sn),
                                  W_act=float(self.T[p, 3] * (-d[0] * sn + d[1] * cs)),
                                  dL_plan=float(self.L_plan[p] - self.L_plan[p - 1]) if p else float("nan"),
                                  crossing=bool(p > 0 and pk[p] != pk[p - 1]), plank=int(pk[p]), window=win,
                                  slip=float("nan"), settle_step=k + 2, place=c.copy()))
            ev["scored"] += 1
            ev["hits"] += int(rec["hit"])
            self.counts["scored"] += 1
            self.ptr += 1
            self.last_score_step = k
            self.last_err[:] = (along, across)
            if rec["far"]:
                ev["far_miss"] += 1
                self.counts["far_miss"] += 1
                if self.strict:
                    self.last_place[f] = (c[0], c[1], yaw)
                    ev["touchdowns"].append(rec)
                    self._fail("far_miss", ev, self._now)
                    return
            if self.ptr == self.N:
                if self.course.end == "stop":
                    self.in_stop, self.stop_t0, self.still = True, k, 0
                    ev["stop_phase"] = True
                else:
                    self.finished = self.done = True
                    self.completed = not rec["far"]
                    self.end_step = self._now
                    ev["finished"], ev["completed"] = True, self.completed
        else:
            rec["kind"] = "wrong_foot"
            ev["wrong_foot"] += 1
            self.counts["wrong_foot"] += 1
            if self.strict:
                self._fail("wrong_foot", ev, self._now)
        self.last_place[f] = (c[0], c[1], yaw)
        ev["touchdowns"].append(rec)

    def update_state(self, step, state, spheres, gate):
        """update() from a [67] state (pelvis x, y, quaternion, velocity channels 34-35, yaw rate 39) and its sphere
        centres [2, 4, 3]."""
        x = np.asarray(state, np.float64)
        fc, fy = foot_pose(spheres)
        return self.update(step, x[:2], quat_yaw(x[3:7]), fc, fy, spheres, gate, x[34:36], x[39])

    # -- read-out ---------------------------------------------------------------------------------------------
    def state(self):
        """Pointer and the observation's status values at the last update."""
        return dict(step=self.step, pointer=self.ptr, expected_foot=self.expected_foot(),
                    last_place=self.last_place.copy(),
                    since_td=np.minimum((self.step - self.last_td_step) * self.dt, 1.0),
                    steps_since_score=self.step - self.last_score_step, last_err=self.last_err.copy(),
                    restep=self.restep, start_window=self.start_window(), stop_phase=self.in_stop, still=self.still,
                    targets_left=self.N - self.ptr, done=self.done, completed=self.completed, fail=self.fail)

    def metrics(self):
        """dict(rows=[per scored target], summary={...}) (spec 5.4, 12.4)."""
        rows = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items() if k != "settle_step"}
                for r in self.rows]
        e = np.array([r["e"] for r in self.rows])
        planks = self.course.is_planks
        clean = (self.touchdowns_over_gap == 0 and self.unsupported_steps == 0) if planks else True
        q = lambda a, x: float(np.percentile(a, x)) if len(a) else float("nan")
        summ = dict(level=self.course.level, seed=self.course.seed, n_targets=self.N,
                    completed=self.completed, finished=self.finished or self.completed, fail=self.fail,
                    end_step=self.end_step, stop_success=self.stop_step is not None, stop_step=self.stop_step,
                    clean=bool(clean), clean_completion=bool(clean and self.completed),
                    err_p50=q(e, 50), err_p95=q(e, 95), err_max=float(e.max()) if len(e) else float("nan"),
                    along_abs_p95=q(np.abs([r["along"] for r in self.rows]), 95),
                    across_abs_p95=q(np.abs([r["across"] for r in self.rows]), 95),
                    touchdowns=self.touchdowns, touchdowns_over_gap=self.touchdowns_over_gap,
                    unsupported_steps=self.unsupported_steps, max_abs_l=self.max_abs_l,
                    max_abs_dpsi=self.max_abs_dpsi,
                    mean_abs_dpsi=self.sum_abs_dpsi / self.n_drift if self.n_drift else float("nan"),
                    steps=self.step - self.t0 + 1, **self.counts)
        for x in (3, 5, 8):
            summ[f"hit{x}"] = float((e <= x / 100).mean()) if len(e) else float("nan")
            summ[f"hits{x}"] = int((e <= x / 100).sum())
            summ[f"success{x}"] = bool(self.completed and (e <= x / 100).all())
        return dict(rows=rows, summary=summ)
