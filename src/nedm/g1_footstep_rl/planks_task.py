"""The "stepping planks" task for the G1: layouts, the foot-support rule, success and metrics (numpy only).

The same definition serves both worlds:
- MuJoCo (ground truth, nedm.g1_footstep_rl.mujoco_env): the planks are box geoms with their tops at z = 0 and pits
  0.6 m deep between them.
- The contact NRD (flat floor at z = 0): the planks are a constraint on the foot positions (forward kinematics of
  the 4 foot spheres of each foot).

Path frame: origin = the robot's start pelvis position (world x, y), heading = its start yaw. s = distance along
the heading, l = distance to the left. The robot stands at s = l = 0, facing +s.

Layout (one per (level, seed)): plank 0 = the start platform [START_EDGE - PLATFORM_LEN, START_EDGE], then
N_PLANKS planks of length L_i separated by gaps G_1 .. G_{N+1}, then the final platform (PLATFORM_LEN long).
Every plank spans |l| <= HALF_WIDTH across the path.

Support rule: a foot sphere is "over a plank" if its centre (s, l) lies in [s_start, s_end] x [-HALF_WIDTH,
HALF_WIDTH] of a plank. A foot in contact with any sphere centre not over a plank is "unsupported" (it stands
partly over a gap). A touchdown is the first control step with contact after at least MIN_SWING_STEPS steps
without contact.

Success: at a control step the pelvis is at s >= final start + SUCCESS_MARGIN and all 8 sphere centres are over
the final platform, within EPISODE_S seconds (including the STAND_S standing start), and before it no end event
(fall, non-foot or self contact, low pelvis, a foot in a pit). "Clean" success: success with no touchdown over a gap
and no unsupported contact step during the whole episode (the strict rule that the NRD constraint enforces).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

# ---- task constants (m, s) ------------------------------------------------------------------------------------
N_PLANKS = 6               # planks between the start and the final platform
START_EDGE = 0.20          # far edge of the start platform (the standing robot's toe spheres end at s = 0.099)
PLATFORM_LEN = 1.0         # length of the start and the final platform
HALF_WIDTH = 1.0           # every plank spans |l| <= 1.0 m across the path
PIT_DEPTH = 0.6            # MuJoCo only: the pit floor is at z = -0.6 m
STAND_S = 1.0              # the robot stands (zero command) for 1 s before it may walk
EPISODE_S = 15.0           # time limit T, including the standing second
SUCCESS_MARGIN = 0.3       # the pelvis must be 0.3 m past the start of the final platform
MIN_SWING_STEPS = 2        # a touchdown needs >= 2 control steps (40 ms) without contact before it
CONTROL_S = 0.02

# Foot sole: the 4 collision spheres of each ankle_roll_link (MJCF order), radius 5 mm -> sole 0.18 m long.
FOOT_SPHERES = np.array([[-0.05, 0.025, -0.03], [-0.05, -0.025, -0.03], [0.12, 0.03, -0.03], [0.12, -0.03, -0.03]])
FOOT_RADIUS = 0.005

# Difficulty levels: plank length L and gap G drawn uniformly (m) per plank and per gap.
# A step that crosses a gap from a fully supported stance foot needs a step length (foot centre to foot centre,
# along the path) of at least G + 0.17 m (sphere centres +-0.085 m from the foot centre); the teacher's step length at touchdown is -0.18 .. 0.36 m (p1 - p99,
# nominal data; max 0.42 m). Reasoning and measured HOMIE baselines: the internal design notes (not released).
LEVELS = {
    "L0": dict(plank=(0.60, 0.80), gap=(0.03, 0.05)),     # calibration only: long planks, narrow gaps
    "L1": dict(plank=(0.40, 0.50), gap=(0.05, 0.10)),
    "L2": dict(plank=(0.30, 0.40), gap=(0.10, 0.15)),
    "L3": dict(plank=(0.25, 0.35), gap=(0.15, 0.20)),
    "L4": dict(plank=(0.22, 0.28), gap=(0.15, 0.20)),
}


@dataclass
class Layout:
    """One plank layout. intervals [N_PLANKS + 2, 2]: (s_start, s_end) of the start platform, the planks and the
    final platform along the path (m); origin (world x, y) and heading (rad) of the path frame."""
    intervals: np.ndarray
    origin: tuple = (0.0, 0.0)
    heading: float = 0.0
    level: str = ""
    seed: int = -1
    half_width: float = HALF_WIDTH
    pit_depth: float = PIT_DEPTH
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.intervals = np.asarray(self.intervals, np.float64).reshape(-1, 2)
        if len(self.intervals) < 2 or (self.intervals[:, 1] <= self.intervals[:, 0]).any() \
                or (self.intervals[1:, 0] <= self.intervals[:-1, 1]).any():
            raise ValueError("plank intervals must be sorted, of positive length and separated by gaps")
        self.origin = (float(self.origin[0]), float(self.origin[1]))
        self.heading = float(self.heading)

    # -- geometry ---------------------------------------------------------------------------------------------
    @property
    def n_planks(self):
        """Planks between the platforms."""
        return len(self.intervals) - 2

    @property
    def final_start(self):
        return float(self.intervals[-1, 0])

    @property
    def gaps(self):
        """[n_planks + 1, 2] (s_start, s_end) of the gaps."""
        return np.stack((self.intervals[:-1, 1], self.intervals[1:, 0]), 1)

    def path_coords(self, xy):
        """World x, y [..., 2] -> path (s, l) [..., 2]."""
        d = np.asarray(xy, np.float64)[..., :2] - np.asarray(self.origin)
        c, s = np.cos(self.heading), np.sin(self.heading)
        return np.stack((c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]), -1)

    def world_xy(self, s, l=0.0):
        """Path (s, l) -> world (x, y) [..., 2]."""
        s, l = np.broadcast_arrays(np.asarray(s, np.float64), np.asarray(l, np.float64))
        c, sn = np.cos(self.heading), np.sin(self.heading)
        return np.stack((self.origin[0] + c * s - sn * l, self.origin[1] + sn * s + c * l), -1)

    def plank_of(self, s, l=None):
        """Index of the plank under path points s (and l) [...]: 0 start platform, 1..n planks, n + 1 final
        platform, -1 over a gap or a pit (|l| > half_width or before the start / after the end)."""
        s = np.asarray(s, np.float64)
        k = np.searchsorted(self.intervals[:, 0], s, side="right") - 1
        kk = np.clip(k, 0, len(self.intervals) - 1)
        inside = (k >= 0) & (s <= self.intervals[kk, 1])
        if l is not None:
            inside &= np.abs(np.asarray(l, np.float64)) <= self.half_width
        return np.where(inside, kk, -1)

    # -- serialisation ----------------------------------------------------------------------------------------
    def to_dict(self):
        return dict(intervals=self.intervals.round(6).tolist(), origin=list(self.origin), heading=self.heading,
                    level=self.level, seed=int(self.seed), half_width=self.half_width, pit_depth=self.pit_depth,
                    extra=self.extra)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def to_json(self):
        return json.dumps(self.to_dict())


def sample_layout(level, seed, origin=(0.0, 0.0), heading=None, n_planks=N_PLANKS):
    """The layout of one (level, seed). Plank lengths and gaps are uniform in the level's ranges; the heading is
    uniform in [-pi, pi) unless given (the NRD is not heading invariant; the training data cover all headings)."""
    if level not in LEVELS:
        raise ValueError(f"unknown level {level}; choose from {sorted(LEVELS)}")
    lv = LEVELS[level]
    rng = np.random.default_rng([int(seed), 100 + int(level[1:]), 20261009])   # one fixed stream per level name
    plank = rng.uniform(*lv["plank"], n_planks)
    gap = rng.uniform(*lv["gap"], n_planks + 1)
    yaw = rng.uniform(-np.pi, np.pi) if heading is None else float(heading)
    rows = [(START_EDGE - PLATFORM_LEN, START_EDGE)]
    s = START_EDGE
    for k in range(n_planks):
        s += gap[k]
        rows.append((s, s + plank[k]))
        s += plank[k]
    s += gap[n_planks]
    rows.append((s, s + PLATFORM_LEN))
    return Layout(np.array(rows), origin=origin, heading=yaw, level=level, seed=int(seed))


def foot_spheres_from_ankles(ankle_pos, ankle_rot):
    """World sphere centres [..., 2, 4, 3] from the ankle_roll_link origins [..., 2, 3] and rotations
    [..., 2, 3, 3] (body -> world)."""
    return np.asarray(ankle_pos)[..., None, :] + np.einsum("...ij,kj->...ki", np.asarray(ankle_rot), FOOT_SPHERES)


# ---- online progress and metrics ------------------------------------------------------------------------------
class PlanksProgress:
    """Score one episode step by step on the 20 ms control grid (the same code for MuJoCo and the NRD).

    update(pelvis_xy, spheres, contact) after each control step k with the state at the END of the step: pelvis
    world x, y [2]; sphere centres [2, 4, >=2] (world, left foot then right foot, FOOT_SPHERES order); contact [2]
    bool = the model label of the step: the foot touched the ground in step k or in step k - 1 (NRD: the switch,
    collision logit >= 0, which predicts exactly this label; MuJoCo: the OR of the 10 ms record labels of steps k and
    k - 1, mujoco_env.run_planks_episode). Returns True at the first step that meets the success rule.

    Counting:
    - touchdown: contact after >= min_swing_steps steps without contact; "over a gap" if any sphere centre of that
      foot is not over a plank in the state at the end of the touchdown step (the first state after the contact
      began).
    - unsupported step: a later contact step of a foot with a sphere centre not over a plank in the states at both
      the start and the end of the step (a step with the label on only because of the one-step label tail after a
      clean liftoff is not counted)."""

    def __init__(self, layout, min_swing_steps=MIN_SWING_STEPS, margin=SUCCESS_MARGIN, spheres0=None):
        self.layout, self.min_swing, self.margin = layout, int(min_swing_steps), float(margin)
        self.steps = 0
        self.off = np.zeros(2, np.int64)         # control steps since the foot last had contact (start: standing)
        self.touchdowns = 0
        self.touchdowns_over_gap = 0
        self.unsupported_steps = 0
        self.reached = 0                         # furthest plank index a foot in contact stood on
        self.success_step = None
        self.max_pelvis_s = -np.inf
        self.max_abs_l = 0.0
        self.touchdown_log = []                  # (step, foot, plank index under each sphere [4])
        self.prev_under = None if spheres0 is None else self.under(spheres0)

    def under(self, spheres):
        """[2, 4] plank index under each sphere centre (-1 = over a gap or beside the path)."""
        sl = self.layout.path_coords(np.asarray(spheres)[..., :2])
        return self.layout.plank_of(sl[..., 0], sl[..., 1])

    def update(self, pelvis_xy, spheres, contact):
        lay = self.layout
        under = self.under(spheres)
        prev = under if self.prev_under is None else self.prev_under
        p = lay.path_coords(np.asarray(pelvis_xy)[:2])
        self.max_pelvis_s = max(self.max_pelvis_s, float(p[0]))
        self.max_abs_l = max(self.max_abs_l, float(abs(p[1])))
        for f in range(2):
            if contact[f]:
                bad = bool((under[f] < 0).any())
                if self.off[f] >= self.min_swing:
                    self.touchdowns += 1
                    self.touchdowns_over_gap += bad
                    self.touchdown_log.append((self.steps, f, under[f].tolist()))
                elif bad and (prev[f] < 0).any():
                    self.unsupported_steps += 1
                if (under[f] >= 0).any():
                    self.reached = max(self.reached, int(under[f].max()))
                self.off[f] = 0
            else:
                self.off[f] += 1
        self.prev_under = under
        self.steps += 1
        final = lay.n_planks + 1
        done = p[0] >= lay.final_start + self.margin and bool((under == final).all())
        if done and self.success_step is None:
            self.success_step = self.steps
        return done

    def metrics(self):
        """Episode metrics (success itself is decided by the caller together with the end events)."""
        return dict(steps=self.steps, reached=self.reached,
                    planks_crossed=int(min(self.reached, self.layout.n_planks)),
                    touchdowns=self.touchdowns, touchdowns_over_gap=self.touchdowns_over_gap,
                    unsupported_steps=self.unsupported_steps, success_step=self.success_step,
                    max_pelvis_s=round(self.max_pelvis_s, 4), max_abs_l=round(self.max_abs_l, 4))
