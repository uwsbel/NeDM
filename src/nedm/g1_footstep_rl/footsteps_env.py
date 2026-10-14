"""The footstep-target task inside the G1 contact NRD as a batched RSL-RL 2.2.4 VecEnv (GPU or CPU).

It follows the internal design spec of the study (cited as "spec N"; not released; see the package docstring).
Courses, frames, banks and the numpy reference scorer FootstepProgress live in nedm.g1_footstep_rl.footsteps_task
(FT). This module places bank courses in torch, scores them with FootstepTracker (the torch twin of
FT.FootstepProgress: the same rules, order and timing, checked state by state against it), builds the 163
observation values with the pure function footsteps_observation (the MuJoCo runner calls the same function, with
views_from_progress), and applies the rules and the reward of the spec. Helpers come from planks_env, fk, sim, homie.

One env step = one 20 ms control step = one NRD step (the action pipeline of the planks env, spec 7):

    a [B, 15] (3 if action.alpha = 0) -> clip +-action.clip -> u = tanh(a)
    command = lo + (hi - lo) (u[:3] + 1) / 2, height 0.74;  raw, legs = HomieTeacher.step(command, newest state)
    target  = cat(legs + alpha u[3:15], upper body default);  next, switch = NrdSim.step(history, target, member)
    -> FootstepTracker.update(next state, switch) -> rules -> reward -> history advanced -> finished envs reset

Reset (stand starts only; spec 4.1, 4.7): a start of the training-split start bank (start_filter: nominal stand
starts) and a course of the course bank (course_bank, float32 course-frame rows; a level drawn from level_weights, then
a course of that level uniformly). Course frame of the start: origin = midpoint of the two start foot centres, x =
the start pelvis yaw; the course is placed rigidly (place_courses = FT.place_arrays), then target 0 = start centre of
foot f0 + d0 t_0 (FT.Course.at_start). The tracker starts with the start foot centres and foot yaws as the last
placements. reset_from_snapshots: MuJoCo hand-over snapshots (spec 12.2 rule 13) with their own placed courses.

FootstepTracker (spec 5; FT.FootstepProgress docstring for the full rule list, decisions D14-D19 there): per update
1. plank counters (true planks), 2. -, 3. confirmation of the raw touchdowns of the previous step (flickers dropped),
4. confirmed touchdowns processed, the expected foot first (scored / re-step / wrong foot / overrun / stop adjust),
5. stop phase (stillness, no_stop), 6. stall, 7. drift. Step k = the k-th update after the reset (t0_step = 0).

Termination (failure reasons, step_info["reasons"], logs per level; spec 8):
    support      plank courses: a foot with its switch on has >= rules.support_min_off sphere centres outside every
                 plank shrunk by rules.edge_margin (along and across), in this state and (two_state) the previous one
                 (rules.support "centre": its centre outside instead); flat courses: off
    unstable     pelvis z < min_z or tilt > max_tilt_deg ("fall" = z < fall_z or tilt > fall_tilt_deg, logged inside it)
    legs         fk.swept_leg_distances(previous state, new state, *rules.legs_sweep): legs < legs_min_dist or pelvis <
                 legs_pelvis_min_dist (None: off); replaces the planks env's feet_min_dist
    joint_range  a joint angle outside rules.joint_range (min - margin, max + margin)
    nonfinite    the NRD returned a non-finite state (the previous state is kept until the reset)
    stall, wrong_foot, far_miss, restep_budget, overrun, no_stop   the tracker (all but stall only with score.strict)
Course end: completion (through: last target scored within tol_fail; stop: stop success) or "finished" (a non-strict
through course whose last target is a far miss: ends without completion, E7). Time-out at episode_steps (value
bootstrap: extras["time_outs"], extras["final_observations"]).

Reward per step (reward.*; spec 9; no alive bonus, no remaining-step charge):
    per scored touchdown   hit [0.5 exp(-(e / hit_wide)^2) + 0.5 exp(-(e / hit_narrow)^2)] + tol_bonus [e <= score.tol]
                           + window [plank course, all 4 sphere centres on the target's plank shrunk by edge_margin]
                           - miss [e > miss_dist] + yaw exp(-(foot yaw - target yaw)^2 / yaw_scale^2)
    progress               progress clip((pelvis_xy(t+1) - pelvis_xy(t)) . t_p, +-progress_clip) while p < N (E5)
    drift                  - lateral min((l / lateral_scale)^2, 1) - heading min((1 - cos dpsi) / (1 - cos
                           heading_scale_deg), 1); l, dpsi of the pelvis against the line of target min(p, N - 1)
    regularisers           - residual mean(u_res^2) - action_rate mean((u - u_prev)^2) / 4 - tilt term - slip term
                           (as the planks env)
    events                 - restep per counted re-step (after the start window) - wrong_foot per wrong-foot step
                           + stop at stop success + complete at completion + success more if every scored e <= tol
    failure                - fail; the failing step's own terms are not paid
reward.legs_clear (arm B) must be 0: arm B was cut. reward.gamma is unused (kept for the trainer and the spec keys).

Observation (obs_names(), 99 + 2 + 14 K + 6 = 163 for K = obs.n_targets = 4, float32, clipped to +-obs.clip; spec 6):
    proprio_block (99) = planks_env OBS_NAMES[:99] (bit-identical to planks_env.planks_observation(...)[:, :99]);
    path_l (clip +-1), path_yaw: the pelvis against the line through C_p along t_p, p = min(pointer, N - 1);
    per target k = 0..K-1 (target j = min(p + k, N - 1)): px, py (target - pelvis), fx, fy (target - current centre of
    its foot), both in the pelvis heading frame clipped to +-obs.pos_clip; L, W (step vector in the target's local path
    frame: target minus the other foot's last placement for k = 0, minus target j(k-1) for k >= 1; W positive when the
    feet are apart); yaw (wrap(target yaw - pelvis yaw)); side (+1 left, -1 right); slack front, back, left, right
    (support_slack at reset, clipped to obs.slack_clip; flat 0.30); stop; valid (p + k < N);
    status: since_td_l, since_td_r (s, clip 1), targets_left ((N - p) / 10, clip 2.5), last_err_along, last_err_across
    (clip +-0.2), start_window (0 / 1). The critic sees the same values.

extras["step_info"] (per env, this step): outcome flags completed, success (completed and every scored e <=
score.tol), clean_completion, finished, fail, time_out; reasons {name: bool}; level, steps, member, course_row,
course_seed; episode = FootstepTracker.results() (counts, errors, pointer; read them at the done step); returns
(episode reward sums); command, state (new NRD state), gates, touchdown (raw), target, pointer, legs_dist (swept leg
distances or None) and tracker (the tracker's events of the step).

Logs (extras["log"], RSL-RL "/..." keys): /step/<reward term> means and command means; EMAs (0.95) over finished
episodes of EPISODE_KEYS (/episode/...); per active level /level/<name>/{completion, clean_completion, fail, hit5} and
/level/<name>/fail_<reason> for every reason of LOG_REASONS. Every key exists from the first step (NaN until set).

Decisions where the spec is silent (simplest option; E = env):
 E1  Env level list LEVELS = FT.LEVEL_CODES + FT.OLD_LEVELS + ("other",), fixed indices (bank level codes = the first
     ten). Old-layout and showcase courses come only through reset_from_snapshots.
 E2  Course banks stay float32 on the device (the float64 conversion at reset is exact, so placement equals
     FT.place_arrays of FT.load_bank's float64 values); rows sorted by level; course_bank_max > 0 keeps the first K
     courses of each level (tests). Banks are cached per (path, K, device).
 E3  Starts are drawn uniformly from the filtered start bank (the spec filter keeps stand starts only; no stand_share).
 E4  The tracker's swing counters start at 0 for both feet at every reset, as FootstepProgress (88 of the 22,674
     filtered stand starts have a foot in the air: its landing within 2 steps is then not a touchdown).
 E5  Progress uses the pointer at the start of the step (the target the policy saw), paid while it is < N; the drift
     terms use the pointer after the step's touchdowns (the tracker's drift and the next observation's path_l).
 E6  A physical failure in the step of a completion: the failure wins. Tracker and physical reasons can both be set.
 E7  A non-strict through course whose last target is a far miss ends as "finished" (no completion, no bonus, no
     failure, no bootstrap).
 E8  Touchdown terms are paid for every scored touchdown (also non-strict far misses); window only on plank courses.
 E9  rules.support "centre" is kept as an option (planks env parity); the v4 configs use "spheres".
 E10 reset_from_snapshots takes hand-over snapshots only (fresh progress: pointer 0, no stop phase). Last placements:
     snap["progress"]["last_place"] if given, else the foot pose of the newest state.
 E11 Slack uses rules.edge_margin and obs.slack_clip at reset (the trainer resets after every stage change).
 E12 L, W and the target yaw have no own clip (only obs.clip).
 E13 Log hit rates are pooled over the scored targets of the episodes that end in a step (NaN-free EMA).
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from rsl_rl.env import VecEnv

from nedm.g1_footstep_rl.homie_policy import ACTION_SCALE, DEFAULT_ANGLES, N_LEG
from nedm.g1_footstep_rl import REPO_ROOT
from nedm.g1_footstep_rl import footsteps_task as FT
from nedm.g1_footstep_rl.fk import foot_points, swept_leg_distances
from nedm.g1_footstep_rl.homie import HomieNet, HomieTeacher, projected_gravity, world_to_body
from nedm.g1_footstep_rl.planks_env import BANK, NUM_ACTIONS, OBS_NAMES as PLANKS_OBS_NAMES, StartBank, homie_preview, \
    path_coords, wrap
from nedm.g1_footstep_rl.sim import HEADLINE, NrdSim, tilt_cos, yaw

COURSE_BANK = REPO_ROOT / "artifacts/g1_footstep_rl/rl_inputs/courses/course_bank_v4_train.npz"
VAL_COURSE_BANK = REPO_ROOT / "artifacts/g1_footstep_rl/rl_inputs/courses/course_bank_v4_val.npz"
LEVELS = FT.LEVEL_CODES + FT.OLD_LEVELS + ("other",)
N_PROPRIO = 99
PER_TARGET = ("px", "py", "fx", "fy", "L", "W", "yaw", "side", "sf", "sb", "sl", "sr", "stop", "valid")
STATUS = ("since_td_l", "since_td_r", "targets_left", "last_err_along", "last_err_across", "start_window")
PHYS_REASONS = ("support", "unstable", "legs", "joint_range", "nonfinite")
REASONS = PHYS_REASONS + FT.FAIL_REASONS
LOG_REASONS = REASONS + ("fall",)
EPISODE_KEYS = (("completion", "clean_completion", "success", "finished", "fail", "time_out", "length", "ret",
                 "targets_frac", "scored", "hit3", "hit5", "hit8", "err_cm", "resteps", "resteps_free", "wrong_foot",
                 "far_miss", "flickers", "overruns", "touchdowns", "td_over_gap", "unsupported", "max_abs_l",
                 "max_abs_dpsi_deg") + tuple(f"fail_{k}" for k in LOG_REASONS))
LEVEL_KEYS = ("completion", "clean_completion", "fail", "hit5") + tuple(f"fail_{k}" for k in LOG_REASONS)
STEP_TERMS = ("touchdown", "tol_bonus", "window", "miss", "yaw", "progress", "lateral", "heading", "residual",
              "action_rate", "tilt", "slip", "restep", "wrong_foot", "stop", "complete", "reward")
_SCORE_KEYS = tuple(k for k in FT.SCORE_DEFAULTS if k not in ("edge_margin", "dt"))


def default_cfg():
    """All env options (spec 3.4; the json configs override any subset, merge_cfg)."""
    return dict(
        device="cpu", num_envs=64, seed=0,
        checkpoints=[str(p) for p in HEADLINE], members=[0, 1], homie_weights=None,
        bank_path=str(BANK), bank_max=0,
        start_filter=dict(families=["nominal"], kinds=["stand"], vx=[-0.1, 0.1], vy_abs_max=0.1, wz_abs_max=0.2,
                          height_min=0.735),
        course_bank=str(COURSE_BANK), val_course_bank=str(VAL_COURSE_BANK), course_bank_max=0,
        episode_steps=600, auto_reset=True, level_weights={"F0": 1.0},
        action=dict(vx=[-0.5, 1.2], vy=[-0.5, 0.5], wz=[-0.8, 0.8], height=0.74, alpha=0.03, clip=5.0),
        obs=dict(n_targets=4, clip=5.0, pos_clip=3.0, slack_clip=[-0.10, 0.30], homie_preview=True),
        rules=dict(support="spheres", support_min_off=1, two_state=True, edge_margin=0.02, fall_z=0.35,
                   fall_tilt_deg=45.0, min_z=0.55, max_tilt_deg=30.0, legs_min_dist=0.004, legs_pelvis_min_dist=-0.001,
                   legs_sweep=[2, 1.0, 2], joint_range=None),
        score={k: FT.SCORE_DEFAULTS[k] for k in _SCORE_KEYS},
        reward=dict(hit=2.0, hit_wide=0.06, hit_narrow=0.02, tol_bonus=1.0, window=0.5, miss=1.0, miss_dist=0.10,
                    yaw=0.3, yaw_scale=0.15, progress=5.0, progress_clip=0.03, lateral=0.10, lateral_scale=0.10,
                    heading=0.05, heading_scale_deg=20.0, residual=0.02, action_rate=0.02, tilt=0.05, slip=0.02,
                    slip_cap=1.0, restep=0.2, wrong_foot=1.0, legs_clear=0.0, stop=2.0, complete=5.0, success=5.0,
                    fail=2.0, gamma=0.99),
    )


FREE_KEYS = ("level_weights",)


def merge_cfg(cfg):
    """default_cfg() with the given (nested, one level deep) overrides. Unknown option names raise (keys starting
    with "_" and "note" / "notes" are comments)."""
    out = default_cfg()
    for k, v in (cfg or {}).items():
        if k.startswith("_") or k in ("note", "notes"):
            continue
        if k not in out:
            raise KeyError(f"unknown footsteps env option {k!r}")
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in FREE_KEYS:
            bad = sorted(set(v) - set(out[k]))
            if bad:
                raise KeyError(f"unknown footsteps env option {k}.{bad}")
            out[k] = {**out[k], **copy.deepcopy(v)}
        else:
            out[k] = copy.deepcopy(v)
    return out


def obs_names(n_targets=4):
    """Names of the observation values (163 for n_targets 4)."""
    return (list(PLANKS_OBS_NAMES[:N_PROPRIO]) + ["path_l", "path_yaw"]
            + [f"t{k}_{x}" for k in range(int(n_targets)) for x in PER_TARGET] + list(STATUS))


OBS_NAMES = obs_names()
OBS_DIM = len(OBS_NAMES)


# ---- geometry (torch twins of footsteps_task) ----------------------------------------------------------------------
def place_courses(targets, centre, origin, heading):
    """Rigid placement (torch twin of FT.place_arrays): targets [B, N, 5], centre [B, N, 3] in the course frame,
    origin [B, 2], heading [B] -> world (targets, centre) float64."""
    t, c = targets.to(torch.float64).clone(), centre.to(torch.float64).clone()
    o, h = origin.to(torch.float64), heading.to(torch.float64)
    cs, sn = torch.cos(h)[:, None], torch.sin(h)[:, None]
    for a in (t, c):
        x, y = a[..., 0].clone(), a[..., 1].clone()
        a[..., 0] = o[:, 0, None] + cs * x - sn * y
        a[..., 1] = o[:, 1, None] + sn * x + cs * y
        a[..., 2] = wrap(a[..., 2] + h[:, None])
    return t, c


def foot_pose(spheres):
    """Sphere centres [..., 2, 4, >=2] -> (foot centres [..., 2, k], foot yaws [..., 2]) (FT.foot_pose)."""
    d = spheres[..., 2:4, :2].mean(-2) - spheres[..., 0:2, :2].mean(-2)
    return spheres.mean(-2), torch.atan2(d[..., 1], d[..., 0])


def sole_points(xy, yaw_):
    """Sphere centres [..., 4, 2] of flat feet with centres xy [..., 2] and yaws [...] (FT.sole_points)."""
    off = torch.as_tensor(FT.SOLE_OFFSETS, dtype=xy.dtype, device=xy.device)
    c, s = torch.cos(yaw_)[..., None], torch.sin(yaw_)[..., None]
    ox, oy = off[:, 0], off[:, 1]
    return torch.stack((xy[..., None, 0] + c * ox - s * oy, xy[..., None, 1] + s * ox + c * oy), -1)


def support_index(sl, intervals, half_width, margin=0.0, planks=None):
    """Interval row under course-frame points sl [B, ..., 2] for per-env intervals [B, P, 2] (course frame) and half
    widths [B], every plank shrunk by margin along and across -> [B, ...] long, -1 over a gap or beside the planks;
    envs with planks False (flat courses) give 0 everywhere (FT.Course.support_index)."""
    B = sl.shape[0]
    s, l = sl[..., 0].reshape(B, -1, 1), sl[..., 1].reshape(B, -1, 1)
    a, b = intervals[:, None, :, 0], intervals[:, None, :, 1]
    inside = (s >= a + margin) & (s <= b - margin) & (l.abs() <= half_width.view(B, 1, 1) - margin)
    idx = torch.where(inside.any(-1), inside.long().argmax(-1), torch.full_like(s[..., 0], -1, dtype=torch.long))
    if planks is not None:
        idx = torch.where(planks.view(B, 1), idx, torch.zeros_like(idx))
    return idx.view(sl.shape[:-1])


def support_slack(t_local, intervals, half_width, planks, edge_margin, clip=(-0.10, 0.30)):
    """Slack [B, N, 4] (front, back, left, right) of targets t_local [B, N, >=3] (course-frame x, y, yaw): how far
    the foot centre may move before a sphere centre of the foot at the target pose leaves the target's plank (the
    interval under the target centre) shrunk by edge_margin; clipped; flat courses clip[1] (FT.Course.target_slack)."""
    pts = sole_points(t_local[..., :2], t_local[..., 2])                       # [B, N, 4, 2]
    k = support_index(t_local[..., :2], intervals, half_width, 0.0).clamp(min=0)
    em = float(edge_margin)
    a = torch.gather(intervals[..., 0], 1, k) + em
    b = torch.gather(intervals[..., 1], 1, k) - em
    hw = (half_width - em)[:, None]
    out = torch.stack((b - pts[..., 0].amax(-1), pts[..., 0].amin(-1) - a, hw - pts[..., 1].amax(-1),
                       pts[..., 1].amin(-1) + hw), -1)
    out = torch.where(planks.view(-1, 1, 1), out, torch.full_like(out, float(clip[1])))
    return out.clamp(float(clip[0]), float(clip[1]))


def course_arrays(courses, nmax=FT.NMAX, pmax=FT.PMAX):
    """Padded float64 WORLD arrays of placed numpy courses (list of FT.Course) for the env and the observation:
    targets [M, nmax, 5], centre [M, nmax, 3] (rows past n repeat the last row), n, f0, d0, stop (bool), cliff,
    intervals [M, pmax, 2] (course frame; unused rows FT.PAD_INTERVAL), n_int, half_width, origin [M, 2], heading,
    level (index in LEVELS), seed."""
    M = len(courses)
    out = dict(targets=np.zeros((M, nmax, 5)), centre=np.zeros((M, nmax, 3)), n=np.zeros(M, np.int64),
               f0=np.zeros(M, np.int64), d0=np.zeros(M), stop=np.zeros(M, bool), cliff=np.zeros(M, bool),
               intervals=np.tile(np.asarray(FT.PAD_INTERVAL, np.float64), (M, pmax, 1)), n_int=np.zeros(M, np.int64),
               half_width=np.zeros(M), origin=np.zeros((M, 2)), heading=np.zeros(M), level=np.zeros(M, np.int64),
               seed=np.zeros(M, np.int64))
    for i, c in enumerate(courses):
        n = c.n
        if n > nmax:
            raise ValueError(f"course {i}: {n} targets > {nmax}")
        out["targets"][i, :n], out["targets"][i, n:] = c.targets, c.targets[-1]
        out["centre"][i, :n], out["centre"][i, n:] = c.centre, c.centre[-1]
        out["n"][i], out["f0"][i], out["d0"][i] = n, c.f0, c.d0
        out["stop"][i], out["cliff"][i], out["half_width"][i] = c.end == "stop", c.cliff, c.half_width
        out["origin"][i], out["heading"][i] = c.origin, c.heading
        out["level"][i] = LEVELS.index(c.level) if c.level in LEVELS else LEVELS.index("other")
        out["seed"][i] = c.seed
        if c.intervals is not None:
            p = len(c.intervals)
            if p > pmax:
                raise ValueError(f"course {i}: {p} intervals > {pmax}")
            out["intervals"][i, :p], out["n_int"][i] = c.intervals, p
    return out


# ---- course bank ---------------------------------------------------------------------------------------------------
class CourseBank:
    """A course bank file (FT.save_bank; scripts/g1_footstep_rl/build_course_bank.py) on the device. Rows sorted by
    level (stable); max_per_level > 0 keeps the first that many courses of each level. Positions stay float32 (E2):
    targets [M, 40, 5], centre [M, 40, 3], intervals [M, 16, 2] (course frame); level [M] long (index in
    FT.LEVEL_CODES = LEVELS), seed, n, f0, n_int (rows of intervals in use, 0 flat), row (row in the file) long; d0,
    half_width float64; stop, cliff bool."""

    def __init__(self, path, device="cpu", max_per_level=0):
        path = Path(path)
        with np.load(path) as z:
            meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
            if meta.get("level_codes", list(FT.LEVEL_CODES)) != list(FT.LEVEL_CODES):
                raise ValueError(f"{path}: level codes {meta.get('level_codes')} differ from {FT.LEVEL_CODES}")
            a = {k: z[k] for k in FT.BANK_KEYS}
        level = a["level"].astype(np.int64)
        order = np.argsort(level, kind="stable")
        if max_per_level:
            keep = np.zeros(len(order), bool)
            lv = level[order]
            for j in np.unique(lv):
                keep[np.flatnonzero(lv == j)[:int(max_per_level)]] = True
            order = order[keep]
        t = lambda x, dt=None: torch.as_tensor(np.ascontiguousarray(x[order]), device=device).to(dt) if dt else \
            torch.as_tensor(np.ascontiguousarray(x[order]), device=device)
        self.path, self.meta, self.device = str(path), meta, torch.device(device)
        self.row = torch.as_tensor(order, dtype=torch.long, device=device)
        self.level, self.seed = t(a["level"], torch.long), t(a["seed"], torch.long)
        self.n, self.f0, self.n_int = t(a["n"], torch.long), t(a["f0"], torch.long), t(a["n_intervals"], torch.long)
        self.d0, self.half_width = t(a["d0"], torch.float64), t(a["half_width"], torch.float64)
        self.stop, self.cliff = t(a["end"], torch.long) == FT.END_CODES["stop"], t(a["cliff"], torch.bool)
        self.targets, self.centre = t(a["targets"], torch.float32), t(a["centre"], torch.float32)
        self.intervals = t(a["intervals"], torch.float32)
        self.M = len(order)
        cnt = np.bincount(level[order], minlength=len(LEVELS))
        self.count = torch.as_tensor(cnt, dtype=torch.long, device=device)
        self.start = torch.as_tensor(np.concatenate(([0], np.cumsum(cnt)[:-1])), dtype=torch.long, device=device)
        self.levels = [LEVELS[j] for j in range(len(LEVELS)) if cnt[j] > 0]
        self.summary = dict(path=self.path, courses=self.M, per_level={LEVELS[j]: int(cnt[j]) for j in range(len(cnt))
                                                                       if cnt[j]})

    def draw(self, levels, gen):
        """One random course row of each level index in levels [n] (uniform within the level)."""
        cnt = self.count[levels]
        if bool((cnt == 0).any()):
            raise ValueError(f"no course of level(s) {sorted({LEVELS[int(j)] for j in levels[cnt == 0]})} in {self.path}")
        u = torch.rand(len(levels), generator=gen, dtype=torch.float64, device=levels.device)
        return self.start[levels] + (u * cnt).long().clamp(max=cnt - 1)

    def rows_of(self, level):
        """All rows of one level name (bank order = seed order)."""
        j = LEVELS.index(level)
        s, c = int(self.start[j]), int(self.count[j])
        return torch.arange(s, s + c, device=self.device)


_BANKS = {}


def course_bank(path, device="cpu", max_per_level=0):
    """CourseBank cache per (path, max_per_level, device)."""
    key = (str(Path(path).resolve()), int(max_per_level), str(torch.device(device)))
    if key not in _BANKS:
        _BANKS[key] = CourseBank(path, device, max_per_level)
    return _BANKS[key]


# ---- the scorer ----------------------------------------------------------------------------------------------------
class FootstepTracker:
    """Torch twin of FT.FootstepProgress for B episodes (spec 5): the same rules, processing order (expected foot
    first) and timing, with t0_step = 0 (step k = the k-th update() after the env's reset). FT.FootstepProgress's
    docstring lists the per-update rules; the tracker additionally pools per-episode hit counts and errors.

    reset(ids, course, start_place [n, 2, 3] (x, y, foot yaw of each foot), start_spheres [n, 2, 4, >=2]) with
    course = dict(targets [n, NMAX, 5], centre [n, NMAX, 3] world (placed, target 0 set), n [n], stop [n] bool,
    planks [n] bool, intervals [n, PMAX, 2] course frame, half_width [n], origin [n, 2], heading [n]).
    update(pelvis_xy [B, 2], pelvis_yaw [B], foot_xy [B, 2, 2], foot_yaw [B, 2], spheres_xy [B, 2, 4, 2], gate [B, 2],
    speed_xy [B, 2], wz [B]) -> events dict (module docstring of the env: per-pass touchdown tensors).
    update_state(state [B, 67], spheres [B, 2, 4, >=2], gate [B, 2]) derives those inputs as FootstepProgress.
    view() -> the observation's tracker values (names of FootstepProgress.state()); results() -> episode values.
    All updates are functional (new tensors), so the tracker runs inside and outside torch.inference_mode."""

    COUNTS = ("scored", "resteps", "resteps_free", "wrong_foot", "far_miss", "flickers", "overruns", "hits3", "hits5",
              "hits8", "touchdowns", "touchdowns_over_gap", "unsupported_steps")

    def __init__(self, num_envs, device="cpu", cfg=None, nmax=FT.NMAX, pmax=FT.PMAX):
        self.B, self.device, self.nmax, self.pmax = int(num_envs), torch.device(device), int(nmax), int(pmax)
        B, dev = self.B, self.device
        L, f64 = torch.long, dict(dtype=torch.float64, device=dev)
        z = lambda *s, dt=torch.float64: torch.zeros(*s, dtype=dt, device=dev)
        self.ar = torch.arange(B, device=dev)
        self.T, self.C = z(B, nmax, 5), z(B, nmax, 3)
        self.N = torch.ones(B, dtype=L, device=dev)
        self.stop_end, self.planks = z(B, dt=torch.bool), z(B, dt=torch.bool)
        self.iv = torch.tensor(FT.PAD_INTERVAL, **f64).repeat(B, pmax, 1)
        self.hw, self.origin, self.heading = torch.ones(B, **f64), z(B, 2), z(B)
        self.foot, self.plank_of_target = z(B, nmax, dt=L), z(B, nmax, dt=L)
        self.k, self.ptr, self.last_score_step = z(B, dt=L), z(B, dt=L), z(B, dt=L)
        self.last_place, self.last_td_step, self.last_err = z(B, 2, 3), z(B, 2, dt=L), z(B, 2)
        self.pend, self.pend_step = z(B, 2, dt=torch.bool), z(B, 2, dt=L)
        self.pend_xy, self.pend_yaw, self.pend_under = z(B, 2, 2), z(B, 2), z(B, 2, 4, dt=L)
        self.in_stop, self.no_stop_flag = z(B, dt=torch.bool), z(B, dt=torch.bool)
        self.stop_t0, self.still = z(B, dt=L), z(B, dt=L)
        self.done, self.completed, self.finished, self.stop_success = (z(B, dt=torch.bool) for _ in range(4))
        self.fail = torch.full((B,), -1, dtype=L, device=dev)
        self.off, self.prev_bad = z(B, 2, dt=L), z(B, 2, dt=torch.bool)
        for k in self.COUNTS:
            setattr(self, k, z(B, dt=L))
        self.err_sum, self.err_max = z(B), z(B)
        self.max_abs_l, self.max_abs_dpsi, self.sum_abs_dpsi, self.n_drift = z(B), z(B), z(B), z(B, dt=L)
        self.set_cfg(cfg)

    def set_cfg(self, cfg=None):
        """Score options: FT.score_cfg of a score dict or an env config (score block + rules.edge_margin)."""
        c = FT.score_cfg(cfg)
        self.cfg = c
        self.dt = dt = float(c["dt"])
        st = lambda x: int(round(float(x) / dt))
        self.sw_steps, self.stall_steps = st(c["start_window_s"]), st(c["stall_s"])
        self.still_steps, self.stop_limit_steps = st(c["stop_still_s"]), st(c["stop_limit_s"])
        self.min_swing, self.strict, self.confirm = int(c["min_swing_steps"]), bool(c["strict"]), bool(c["confirm"])
        self.tol, self.tol_fail = float(c["tol"]), float(c["tol_fail"])
        self.r_adj, self.r_start, self.budget = float(c["r_adj"]), float(c["r_start"]), float(c["restep_budget"])
        self.stop_v, self.stop_wz, self.em = float(c["stop_v"]), float(c["stop_wz"]), float(c["edge_margin"])

    # -- reset ----------------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def reset(self, ids, course, start_place, start_spheres):
        ids = torch.as_tensor(ids, device=self.device).long()
        n = len(ids)
        if n == 0:
            return
        f64 = lambda x: torch.as_tensor(x, device=self.device).to(torch.float64)
        T, C = f64(course["targets"]), f64(course["centre"])
        iv, hw = f64(course["intervals"]), f64(course["half_width"])
        o, h = f64(course["origin"]), f64(course["heading"])
        planks = torch.as_tensor(course["planks"], device=self.device).bool()
        self.T[ids], self.C[ids], self.iv[ids], self.hw[ids], self.origin[ids], self.heading[ids] = T, C, iv, hw, o, h
        self.N[ids] = torch.as_tensor(course["n"], device=self.device).long()
        self.stop_end[ids] = torch.as_tensor(course["stop"], device=self.device).bool()
        self.planks[ids] = planks
        self.foot[ids] = (T[..., 3] <= 0).long()                       # side +1 -> foot 0 (left)
        self.plank_of_target[ids] = support_index(path_coords(T[..., :2], o, h), iv, hw, 0.0, planks)
        for name in ("k", "ptr", "last_score_step", "stop_t0", "still", "off", "pend_step"):
            getattr(self, name)[ids] = 0
        for name in self.COUNTS:
            getattr(self, name)[ids] = 0
        for name in ("in_stop", "no_stop_flag", "done", "completed", "finished", "stop_success", "pend"):
            getattr(self, name)[ids] = False
        for name in ("last_err", "pend_xy", "pend_yaw", "err_sum", "err_max", "max_abs_l", "max_abs_dpsi",
                     "sum_abs_dpsi"):
            getattr(self, name)[ids] = 0.0
        self.n_drift[ids] = 0
        self.pend_under[ids] = 0
        self.fail[ids] = -1
        self.last_place[ids] = f64(start_place)
        self.last_td_step[ids] = -10 ** 6
        sp = f64(start_spheres)[..., :2]
        sl = path_coords(sp.reshape(n, 8, 2), o, h).view(n, 2, 4, 2)
        self.prev_bad[ids] = (support_index(sl, iv, hw, 0.0, planks) < 0).any(-1)

    # -- helpers --------------------------------------------------------------------------------------------------
    def _put(self, x, f, m, val):
        """x [B, 2, ...] with x[b, f[b]] = val[b] where m[b] (functional)."""
        sel = (torch.arange(2, device=self.device)[None] == f[:, None]) & m[:, None]
        sel = sel.view(self.B, 2, *([1] * (x.dim() - 2)))
        return torch.where(sel, val[:, None], x)

    def _fail(self, m, reason):
        new = m & ~self.done
        self.fail = torch.where(new, FT.FAIL_REASONS.index(reason), self.fail)
        self.done = self.done | new

    def drift(self, pelvis_xy, pelvis_yaw):
        """(l, dpsi) [B]: pelvis offset from the line through C_p along t_p (left positive) and wrap(yaw - psi_p),
        p = min(pointer, N - 1)."""
        c = self.C[self.ar, torch.minimum(self.ptr, self.N - 1)]
        d = pelvis_xy[:, :2] - c[:, :2]
        return -d[:, 0] * torch.sin(c[:, 2]) + d[:, 1] * torch.cos(c[:, 2]), wrap(pelvis_yaw - c[:, 2])

    def _process(self, m, f, k_td, c, yaw_, under):
        """One confirmed touchdown per env (mask m) of foot f [B] at touchdown step k_td [B], placement c [B, 2],
        yaw [B], support rows of its spheres with the edge margin under [B, 4] (FootstepProgress._process)."""
        ar = self.ar
        m = m & ~self.done
        self.last_td_step = self._put(self.last_td_step, f, m, k_td)
        lp = self.last_place[ar, f]
        dist = torch.hypot(c[:, 0] - lp[:, 0], c[:, 1] - lp[:, 1])
        ms, mn = m & self.in_stop, m & ~self.in_stop
        over = ms & (dist > self.r_adj)
        self.overruns = self.overruns + over
        if self.strict:
            self._fail(over, "overrun")
        p = torch.minimum(self.ptr, self.N - 1)
        Tp, Cp = self.T[ar, p], self.C[ar, p]
        fpp = self.foot[ar, p]
        d = c - Tp[:, :2]
        e = torch.hypot(d[:, 0], d[:, 1])
        cs, sn = torch.cos(Cp[:, 2]), torch.sin(Cp[:, 2])
        along, across = d[:, 0] * cs + d[:, 1] * sn, -d[:, 0] * sn + d[:, 1] * cs
        sw = (k_td < self.sw_steps) & (self.ptr < 2)
        near = dist <= torch.where(sw, torch.full_like(dist, self.r_start), torch.full_like(dist, self.r_adj))
        rst = mn & near & ((f != fpp) | (e > self.tol))
        rfree, rcnt = rst & sw, rst & ~sw
        self.resteps_free = self.resteps_free + rfree
        self.resteps = self.resteps + rcnt
        if self.strict:
            self._fail(rcnt & (self.resteps > self.budget), "restep_budget")
        sc = mn & ~rst & (f == fpp)
        wf = mn & ~rst & (f != fpp)
        far = e > self.tol_fail
        win = self.planks & (under == self.plank_of_target[ar, p][:, None]).all(-1)
        yaw_err = wrap(yaw_ - Tp[:, 2])
        self.scored = self.scored + sc
        self.ptr = self.ptr + sc
        self.last_score_step = torch.where(sc, k_td, self.last_score_step)
        self.last_err = torch.where(sc[:, None], torch.stack((along, across), -1), self.last_err)
        for x in (3, 5, 8):
            setattr(self, f"hits{x}", getattr(self, f"hits{x}") + (sc & (e <= x / 100)))
        self.err_sum = self.err_sum + torch.where(sc, e, torch.zeros_like(e))
        self.err_max = torch.where(sc, torch.maximum(self.err_max, e), self.err_max)
        self.far_miss = self.far_miss + (sc & far)
        ffail = sc & far & self.strict
        self._fail(ffail, "far_miss")
        fin = sc & ~ffail & (self.ptr == self.N)
        st = fin & self.stop_end
        self.in_stop = self.in_stop | st
        self.stop_t0 = torch.where(st, k_td, self.stop_t0)
        self.still = torch.where(st, torch.zeros_like(self.still), self.still)
        thr = fin & ~self.stop_end
        self.finished, self.done = self.finished | thr, self.done | thr
        self.completed = self.completed | (thr & ~far)
        self.wrong_foot = self.wrong_foot + wf
        if self.strict:
            self._fail(wf, "wrong_foot")
        self.last_place = self._put(self.last_place, f, m, torch.cat((c, yaw_[:, None]), -1))
        return dict(mask=m, foot=f, step=k_td, scored=sc, e=e, along=along, across=across, yaw_err=yaw_err,
                    window=sc & win, hit=sc & (e <= self.tol), far=sc & far, restep=rcnt, restep_free=rfree,
                    wrong_foot=wf, overrun=over, stop_adjust=ms & ~over, stop_phase=st, finished=thr,
                    completed=thr & ~far)

    # -- the update -----------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def update(self, pelvis_xy, pelvis_yaw, foot_xy, foot_yaw, spheres_xy, gate, speed_xy, wz):
        B, ar = self.B, self.ar
        f64 = lambda x: torch.as_tensor(x, device=self.device).to(torch.float64)
        pelvis_xy, pelvis_yaw, foot_xy, foot_yaw = f64(pelvis_xy), f64(pelvis_yaw), f64(foot_xy)[..., :2], f64(foot_yaw)
        sp, speed_xy, wz = f64(spheres_xy)[..., :2], f64(speed_xy), f64(wz)
        gate = torch.as_tensor(gate, device=self.device).bool()
        step = self.k
        was_done = self.done
        sl = path_coords(sp.reshape(B, 8, 2), self.origin, self.heading).view(B, 2, 4, 2)
        under0 = support_index(sl, self.iv, self.hw, 0.0, self.planks)           # true planks
        underm = support_index(sl, self.iv, self.hw, self.em, self.planks)       # shrunk by the edge margin
        bad0 = (under0 < 0).any(-1)
        # 1. plank counters (PlanksProgress rule)
        raw = gate & (self.off >= self.min_swing)
        self.touchdowns = self.touchdowns + raw.sum(-1)
        self.touchdowns_over_gap = self.touchdowns_over_gap + (raw & bad0).sum(-1)
        self.unsupported_steps = self.unsupported_steps + (gate & ~raw & bad0 & self.prev_bad).sum(-1)
        self.off = torch.where(gate, torch.zeros_like(self.off), self.off + 1)
        self.prev_bad = bad0
        # 3. confirmation (a raw touchdown of step k is processed at step k + 1 if the gate is still on)
        now2 = step[:, None].expand(B, 2)
        if self.confirm:
            todo, flick = self.pend & gate, self.pend & ~gate
            t_step, t_xy, t_yaw, t_under = self.pend_step, self.pend_xy, self.pend_yaw, self.pend_under
            self.flickers = self.flickers + flick.sum(-1)
            # footsteps_task D15b: after a dropped flicker the next on-step is a new raw touchdown
            self.off = torch.where(flick, self.off.clamp(min=self.min_swing), self.off)
            self.pend = raw
            self.pend_step = torch.where(raw, now2, self.pend_step)
            self.pend_xy = torch.where(raw[..., None], foot_xy, self.pend_xy)
            self.pend_yaw = torch.where(raw, foot_yaw, self.pend_yaw)
            self.pend_under = torch.where(raw[..., None], underm, self.pend_under)
        else:
            todo, flick = raw, torch.zeros_like(raw)
            t_step, t_xy, t_yaw, t_under = now2, foot_xy, foot_yaw, underm
        # 4. processing, the expected foot first
        fp = self.foot[ar, torch.minimum(self.ptr, self.N - 1)]
        passes = []
        for f in (fp, 1 - fp):
            passes.append(self._process(todo[ar, f], f, t_step[ar, f], t_xy[ar, f], t_yaw[ar, f], t_under[ar, f]))
        # 5. stop phase
        sp_on = ~self.done & self.in_stop
        cond = gate.all(-1) & (torch.hypot(speed_xy[:, 0], speed_xy[:, 1]) < self.stop_v) & (wz.abs() < self.stop_wz)
        self.still = torch.where(sp_on, torch.where(cond, self.still + 1, torch.zeros_like(self.still)), self.still)
        succ = sp_on & (self.still >= self.still_steps)
        self.completed, self.done, self.stop_success = self.completed | succ, self.done | succ, self.stop_success | succ
        no_stop = sp_on & ~succ & (step - self.stop_t0 >= self.stop_limit_steps) & ~self.no_stop_flag
        self.no_stop_flag = self.no_stop_flag | no_stop
        if self.strict:
            self._fail(no_stop, "no_stop")
        # 6. stall
        self._fail(~self.done & ~self.in_stop & (step - self.last_score_step >= self.stall_steps), "stall")
        # 7. drift
        l, dpsi = self.drift(pelvis_xy, pelvis_yaw)
        self.max_abs_l = torch.maximum(self.max_abs_l, l.abs())
        self.max_abs_dpsi = torch.maximum(self.max_abs_dpsi, dpsi.abs())
        self.sum_abs_dpsi = self.sum_abs_dpsi + dpsi.abs()
        self.n_drift = self.n_drift + 1
        self.k = self.k + 1
        ended = self.done & ~was_done
        return dict(step=step, raw=raw, flicker=flick, passes=passes, stop_success=succ, no_stop=no_stop,
                    ended=ended, failed=ended & (self.fail >= 0), completed=ended & self.completed,
                    finished=ended & self.finished & ~self.completed, fail=self.fail.clone(), l=l, dpsi=dpsi,
                    under0=under0, underm=underm)

    def update_state(self, state, spheres, gate):
        """update() from states [B, 67] (pelvis x, y, quaternion yaw, velocity channels 34-35, yaw rate 39) and their
        sphere centres [B, 2, 4, >=2] (FootstepProgress.update_state)."""
        x = torch.as_tensor(state, device=self.device).to(torch.float64)
        sp = torch.as_tensor(spheres, device=self.device).to(torch.float64)
        fc, fy = foot_pose(sp)
        return self.update(x[:, :2], yaw(x), fc[..., :2], fy, sp[..., :2], gate, x[:, 34:36], x[:, 39])

    # -- read-out -------------------------------------------------------------------------------------------------
    def view(self):
        """The observation's tracker values at the last update (FootstepProgress.state() names): pointer [B],
        last_place [B, 2, 3], since_td [B, 2] (s, clipped to 1), last_err [B, 2], start_window [B] bool."""
        last = self.k - 1
        since = ((last[:, None] - self.last_td_step).to(torch.float64) * self.dt).clamp(max=1.0)
        return dict(pointer=self.ptr, last_place=self.last_place, since_td=since, last_err=self.last_err,
                    start_window=(last < self.sw_steps) & (self.ptr < 2))

    def results(self):
        """Per-env episode values (spec 5.4): completed, clean (plank courses: no touchdown over a gap and no
        unsupported step; flat: True), clean_completion, success (completed and every scored e <= tol),
        success3/5/8, finished, stop_success, fail (index in FT.FAIL_REASONS or -1), pointer, n, the counts, err_sum,
        err_max, max_abs_l, max_abs_dpsi, mean_abs_dpsi."""
        clean = ~self.planks | ((self.touchdowns_over_gap == 0) & (self.unsupported_steps == 0))
        out = dict(completed=self.completed.clone(), clean=clean, clean_completion=clean & self.completed,
                   success=self.completed & (self.err_max <= self.tol), finished=self.finished | self.completed,
                   stop_success=self.stop_success.clone(), fail=self.fail.clone(), pointer=self.ptr.clone(),
                   n=self.N.clone(), err_sum=self.err_sum.clone(), err_max=self.err_max.clone(),
                   max_abs_l=self.max_abs_l.clone(), max_abs_dpsi=self.max_abs_dpsi.clone(),
                   mean_abs_dpsi=self.sum_abs_dpsi / self.n_drift.clamp(min=1))
        for x in (3, 5, 8):
            out[f"success{x}"] = self.completed & (self.err_max <= x / 100)
        for k in self.COUNTS:
            out[k] = getattr(self, k).clone()
        return out


# ---- observation (pure functions: the env and the MuJoCo runner call the same code) ---------------------------------
def _proprio(state, preview, prev_u, gate):
    st = state
    f64 = dict(dtype=torch.float64, device=st.device)
    quat = st[:, 3:7]
    default = torch.as_tensor(DEFAULT_ANGLES, **f64)
    frame = [projected_gravity(quat), 0.5 * world_to_body(quat, st[:, 37:40]), st[:, 7:34] - default, 0.05 * st[:, 40:67]]
    prev = [ACTION_SCALE * preview.to(torch.float64), prev_u, gate]
    th = yaw(st)
    c, s = torch.cos(th)[:, None], torch.sin(th)[:, None]
    fp = foot_points(st)
    d = fp["center"][..., :2] - st[:, None, 0:2]
    feet = torch.stack((c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1], fp["center"][..., 2]), -1)
    v = st[:, 34:37]
    vel = torch.stack((c[:, 0] * v[:, 0] + s[:, 0] * v[:, 1], -s[:, 0] * v[:, 0] + c[:, 0] * v[:, 1], v[:, 2]), -1)
    return torch.cat(frame + prev + [feet.flatten(1), vel, st[:, 2:3]], 1), fp, th


def proprio_block(state, preview, prev_u, gate):
    """The proprioception block [B, 99] float64 (unclipped) = the first 99 values of planks_env.planks_observation
    (gravity, angular velocity x 0.5, joints - default, joint rates x 0.05, HOMIE preview, previous action u,
    switches, foot centres - pelvis, pelvis velocity, pelvis z; pelvis heading frame)."""
    return _proprio(state, preview, prev_u, gate)[0]


def footsteps_observation(state, preview, prev_u, gate, tracker_view, course_view, obs_cfg):
    """The footsteps observation [B, 163] float32 (module docstring, obs_names) of the newest states [B, 67] float64
    (20 ms averaged rates). preview [B, 12] = planks_env.homie_preview(...); prev_u [B, 15] the previous
    tanh-squashed action; gate [B, 2] float64 the switches of the last step; tracker_view = FootstepTracker.view()
    (or views_from_progress); course_view = dict(targets [B, NMAX, 5], centre [B, NMAX, 3] world, n [B], slack
    [B, NMAX, 4]); obs_cfg = cfg["obs"]. Pure: G1FootstepsEnv.observe and the MuJoCo runner both call it."""
    prop, fp, th = _proprio(state, preview, prev_u, gate)
    B, dev = state.shape[0], state.device
    K = int(obs_cfg["n_targets"])
    T = course_view["targets"].to(torch.float64)
    C = course_view["centre"].to(torch.float64)
    N = torch.as_tensor(course_view["n"], device=dev).long()
    slack = course_view["slack"].to(torch.float64)
    p = torch.as_tensor(tracker_view["pointer"], device=dev).long()
    ar = torch.arange(B, device=dev)
    cp = C[ar, torch.minimum(p, N - 1)]
    d = state[:, :2] - cp[:, :2]
    path_l = (-d[:, 0] * torch.sin(cp[:, 2]) + d[:, 1] * torch.cos(cp[:, 2])).clamp(-1.0, 1.0)
    path_yaw = wrap(th - cp[:, 2])
    j = p[:, None] + torch.arange(K, device=dev)[None]
    valid = (j < N[:, None]).to(torch.float64)
    jc = torch.minimum(j, (N - 1)[:, None])
    g = lambda x: torch.gather(x, 1, jc[..., None].expand(-1, -1, x.shape[-1]))
    Tk, Ck, Sk = g(T), g(C), g(slack)
    c, s = torch.cos(th)[:, None], torch.sin(th)[:, None]
    pc = float(obs_cfg["pos_clip"])
    dp = Tk[..., :2] - state[:, None, :2]
    px, py = (c * dp[..., 0] + s * dp[..., 1]).clamp(-pc, pc), (-s * dp[..., 0] + c * dp[..., 1]).clamp(-pc, pc)
    foot_k = (Tk[..., 3] <= 0).long()
    fck = torch.gather(fp["center"][..., :2], 1, foot_k[..., None].expand(-1, -1, 2))
    df = Tk[..., :2] - fck
    fx, fy = (c * df[..., 0] + s * df[..., 1]).clamp(-pc, pc), (-s * df[..., 0] + c * df[..., 1]).clamp(-pc, pc)
    other0 = tracker_view["last_place"].to(torch.float64)[ar, 1 - foot_k[:, 0], :2]
    prev_xy = torch.cat((other0[:, None], Tk[:, :-1, :2]), 1)
    dv = Tk[..., :2] - prev_xy
    cps, sps = torch.cos(Ck[..., 2]), torch.sin(Ck[..., 2])
    L = dv[..., 0] * cps + dv[..., 1] * sps
    W = Tk[..., 3] * (-dv[..., 0] * sps + dv[..., 1] * cps)
    lo, hi = obs_cfg["slack_clip"]
    Sk = Sk.clamp(float(lo), float(hi))
    per = torch.stack((px, py, fx, fy, L, W, wrap(Tk[..., 2] - th[:, None]), Tk[..., 3], Sk[..., 0], Sk[..., 1],
                       Sk[..., 2], Sk[..., 3], Tk[..., 4], valid), -1)
    since = tracker_view["since_td"].to(torch.float64).clamp(max=1.0)
    left = ((N - p).to(torch.float64) / 10).clamp(max=2.5)
    err = tracker_view["last_err"].to(torch.float64).clamp(-0.2, 0.2)
    sw = torch.as_tensor(tracker_view["start_window"], device=dev).to(torch.float64).view(B, 1)
    out = torch.cat((prop, path_l[:, None], path_yaw[:, None], per.flatten(1), since, left[:, None], err, sw), 1)
    clip = float(obs_cfg["clip"])
    return out.clamp(-clip, clip).to(torch.float32)


def views_from_progress(course, progress=None, edge_margin=FT.EDGE_MARGIN, slack_clip=(-0.10, 0.30), device="cpu"):
    """(tracker_view, course_view) of ONE placed numpy course (target 0 already set) and its FT.FootstepProgress
    (None: the fresh state at the default pose) as batch-1 tensors for footsteps_observation (the MuJoCo runner)."""
    a = course_arrays([course])
    sl = course.target_slack(edge_margin, tuple(slack_clip))
    slack = np.concatenate((sl, np.repeat(sl[-1:], FT.NMAX - course.n, 0)), 0)[None]
    t = lambda x, dt=torch.float64: torch.as_tensor(np.asarray(x), device=device).to(dt)
    cv = dict(targets=t(a["targets"]), centre=t(a["centre"]), n=t(a["n"], torch.long), slack=t(slack))
    if progress is None:
        progress = FT.FootstepProgress(course)
    st = progress.state()
    tv = dict(pointer=t([st["pointer"]], torch.long), last_place=t(st["last_place"][None]),
              since_td=t(st["since_td"][None]), last_err=t(st["last_err"][None]),
              start_window=t([bool(st["start_window"])], torch.bool))
    return tv, cv


# ---- the env -------------------------------------------------------------------------------------------------------
class G1FootstepsEnv(VecEnv):
    """See the module docstring. cfg: merge_cfg overrides. share: another G1FootstepsEnv (or any object with .sim,
    .net, .bank) whose NRD, HOMIE network and start bank are reused (the validation env of the trainer)."""

    def __init__(self, cfg=None, device=None, share=None):
        self.cfg = cfg = merge_cfg(cfg)
        self.device = dev = torch.device(device or cfg["device"])
        self.num_envs = B = int(cfg["num_envs"])
        self.num_actions = 3 if float(cfg["action"]["alpha"]) == 0 else NUM_ACTIONS     # fixed for the env's life
        self.n_targets = int(cfg["obs"]["n_targets"])
        self.obs_names = obs_names(self.n_targets)
        self.num_obs = len(self.obs_names)
        self.num_privileged_obs = None
        self.max_episode_length = int(cfg["episode_steps"])
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(int(cfg["seed"]))
        if share is not None:
            self.sim, self.net, self.bank = share.sim, share.net, share.bank
        else:
            self.sim = NrdSim(cfg["checkpoints"], device=dev)
            self.net = HomieNet.load(cfg["homie_weights"], dev)
            self.bank = StartBank(cfg["bank_path"], cfg["start_filter"], cfg["bank_max"], cfg["seed"], dev)
        self.courses = course_bank(cfg["course_bank"], dev, cfg["course_bank_max"])
        self.teacher = HomieTeacher(B, dev, net=self.net)
        self.tracker = FootstepTracker(B, dev, cfg)
        self.f64 = dict(dtype=torch.float64, device=dev)
        z = lambda *s, dt=torch.float64: torch.zeros(*s, dtype=dt, device=dev)
        self.hist, self.ok = z(B, 4, 67), z(B, 4, dt=torch.bool)
        self.member, self.level = z(B, dt=torch.long), z(B, dt=torch.long)
        self.prev_u, self.prev_cmd, self.gate = z(B, NUM_ACTIONS), z(B, 4), z(B, 2)
        self.bad_prev = z(B, 2, dt=torch.bool)                # support rule (edge margin, min_off) in the newest state
        self.centre_prev = z(B, 2, 3)
        self.slack = z(B, FT.NMAX, 4)
        self.f0, self.n_int = z(B, dt=torch.long), z(B, dt=torch.long)
        self.d0, self.cliff = z(B), z(B, dt=torch.bool)
        self.course_row, self.course_seed = torch.full((B,), -1, dtype=torch.long, device=dev), z(B, dt=torch.long)
        self.t = z(B, dt=torch.long)
        self.episode_length_buf = self.t
        self.ep = {k: z(B) for k in ("reward", "touchdown", "progress", "drift", "reg", "events")}
        self.default_upper = torch.as_tensor(DEFAULT_ANGLES[N_LEG:], **self.f64)
        self.done_stats = {k: torch.tensor(float("nan"), **self.f64) for k in EPISODE_KEYS}
        self.level_names = list(LEVELS)
        self.level_stats = torch.full((len(LEVEL_KEYS), len(LEVELS)), float("nan"), **self.f64)
        self.obs_buf = torch.zeros(B, self.num_obs, device=dev)
        self.extras = {}
        self.set_cfg(cfg)
        self.set_level_weights(cfg["level_weights"])
        self.reset()

    # ------------------------------------------------------------------ configuration
    def set_cfg(self, cfg):
        """(Re)read the options that may change between curriculum stages (rules, score, reward, action, members).
        A reset should follow (slack and the support state are set at reset)."""
        cfg = merge_cfg(cfg)
        dev = self.device
        if int(cfg["obs"]["n_targets"]) != self.n_targets:
            raise ValueError("obs.n_targets cannot change in one env (the observation size is fixed)")
        self.cfg = cfg
        ac = cfg["action"]
        self.cmd_lo = torch.tensor([ac["vx"][0], ac["vy"][0], ac["wz"][0]], **self.f64)
        self.cmd_hi = torch.tensor([ac["vx"][1], ac["vy"][1], ac["wz"][1]], **self.f64)
        self.alpha = float(ac["alpha"])
        if (self.alpha == 0) != (self.num_actions == 3):
            raise ValueError("action.alpha cannot switch between 0 (3 actions) and > 0 (15 actions) in one env")
        rl, rw = cfg["rules"], cfg["reward"]
        if rl["support"] not in ("spheres", "centre"):
            raise ValueError(f"rules.support {rl['support']!r}: 'spheres' or 'centre'")
        if float(rw["legs_clear"]) != 0.0:
            raise NotImplementedError("reward.legs_clear (arm B) was cut from v4; it must be 0")
        jr = rl.get("joint_range")
        self.q_lo = self.q_hi = None
        if jr:
            m = float(jr.get("margin", 0.0))
            self.q_lo = torch.tensor(jr["min"], **self.f64) - m
            self.q_hi = torch.tensor(jr["max"], **self.f64) + m
            if self.q_lo.shape != (27,) or self.q_hi.shape != (27,):
                raise ValueError("rules.joint_range min / max need 27 values")
        self.legs_on = rl.get("legs_min_dist") is not None or rl.get("legs_pelvis_min_dist") is not None
        members = [int(m) for m in cfg["members"]]
        if not members or max(members) >= self.sim.n_members:
            raise ValueError(f"members {members}: the NRD has {self.sim.n_members} members")
        self.members = torch.tensor(members, dtype=torch.long, device=dev)
        self.auto_reset = bool(cfg["auto_reset"])
        self.max_episode_length = int(cfg["episode_steps"])
        self.tracker.set_cfg(cfg)

    def set_level_weights(self, weights):
        """weights {level name: weight}; a level with a positive weight needs courses in the course bank."""
        w = torch.zeros(len(LEVELS), **self.f64)
        for k, v in (weights or {n: 1.0 for n in self.courses.levels}).items():
            if k not in LEVELS:
                raise KeyError(f"level {k!r} not in {LEVELS}")
            if float(v) > 0 and int(self.courses.count[LEVELS.index(k)]) == 0:
                raise ValueError(f"level {k!r} has no course in {self.courses.path}")
            w[LEVELS.index(k)] = float(v)
        if not bool((w > 0).any()):
            raise ValueError("level_weights: no positive weight")
        self.level_w = w
        self.active_levels = [j for j in range(len(LEVELS)) if float(w[j]) > 0]
        self.level_stats = torch.where((w > 0)[None], self.level_stats, torch.full_like(self.level_stats, float("nan")))

    # ------------------------------------------------------------------ VecEnv
    @property
    def unwrapped(self):
        return self

    def get_observations(self):
        return self.obs_buf, {"observations": {"critic": self.obs_buf}}

    def get_privileged_observations(self):
        return None

    @torch.inference_mode()
    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._compute_obs()
        return self.obs_buf, {"observations": {"critic": self.obs_buf}}

    def _rule_bad(self, underm, cidx):
        """Training support rule per foot [n, 2] from the sphere rows (edge margin) [n, 2, 4] and the centre rows
        [n, 2] (rules.support); plank courses only (flat rows are 0, never bad)."""
        rl = self.cfg["rules"]
        if rl["support"] == "centre":
            return cidx < 0
        return (underm < 0).sum(-1) >= int(rl["support_min_off"])

    def _install(self, ids, course, start_state, extra):
        """Set the course, the tracker and the support state of envs ids from course tensors (placed, target 0 set)
        and the start states [n, 67]."""
        fp = foot_points(start_state)
        sph = fp["spheres"]
        fc, fy = foot_pose(sph)
        start_place = extra.get("start_place")
        if start_place is None:
            start_place = torch.cat((fc[..., :2], fy[..., None]), -1)
        self.tracker.reset(ids, course, start_place, sph[..., :2])
        o, h, iv, hw = course["origin"], course["heading"], course["intervals"], course["half_width"]
        planks = course["planks"]
        T = course["targets"]
        t_local = torch.cat((path_coords(T[..., :2], o, h), wrap(T[..., 2] - h[:, None])[..., None]), -1)
        em = float(self.cfg["rules"]["edge_margin"])
        self.slack[ids] = support_slack(t_local, iv, hw, planks, em, self.cfg["obs"]["slack_clip"])
        n = len(ids)
        sl = path_coords(sph[..., :2].reshape(n, 8, 2), o, h).view(n, 2, 4, 2)
        underm = support_index(sl, iv, hw, em, planks)
        cidx = support_index(path_coords(fc[..., :2], o, h), iv, hw, em, planks)
        self.bad_prev[ids] = self._rule_bad(underm, cidx) & planks[:, None]
        self.centre_prev[ids] = fp["center"]
        for k in ("f0", "d0", "cliff", "n_int", "course_row", "course_seed"):
            getattr(self, k)[ids] = extra[k]

    @torch.inference_mode()
    def reset_idx(self, ids, starts=None, levels=None, members=None, courses=None):
        """Reset envs ids (long [n]). starts (start bank indices [n]), levels (indices in LEVELS [n]), members ([n])
        and courses (course bank rows [n]) are drawn when None; given courses fix the levels."""
        ids = torch.as_tensor(ids, device=self.device).long()
        n = len(ids)
        if n == 0:
            return
        dev, b, cb, cfg = self.device, self.bank, self.courses, self.cfg
        si = (torch.randint(0, b.n, (n,), generator=self.gen, device=dev) if starts is None
              else torch.as_tensor(starts, device=dev).long())
        if courses is not None:
            rows = torch.as_tensor(courses, device=dev).long()
            lv = cb.level[rows]
            if levels is not None and not torch.equal(torch.as_tensor(levels, device=dev).long(), lv):
                raise ValueError("reset_idx: the given levels differ from the levels of the given courses")
        else:
            lv = (torch.multinomial(self.level_w, n, replacement=True, generator=self.gen) if levels is None
                  else torch.as_tensor(levels, device=dev).long())
            rows = cb.draw(lv, self.gen)
        mb = (self.members[torch.randint(0, len(self.members), (n,), generator=self.gen, device=dev)] if members is None
              else torch.as_tensor(members, device=dev).long())
        hist = b.hist[si]
        cur = hist[:, -1]
        self.hist[ids], self.ok[ids] = hist, b.ok[si]
        self.member[ids], self.level[ids] = mb, lv
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        mask[ids] = True
        order = torch.argsort(ids)                            # HomieTeacher.reset takes rows in env index order
        self.teacher.reset(mask, b.teacher_history[si][order], b.teacher_last_action[si][order])
        # course frame of the start (FT.course_frame), rigid placement, target 0 from the start foot (FT.at_start)
        fc = foot_points(cur)["center"]
        origin, heading = fc[..., :2].mean(1), yaw(cur)
        T, C = place_courses(cb.targets[rows], cb.centre[rows], origin, heading)
        f = (T[:, 0, 3] <= 0).long()
        psi0 = C[:, 0, 2]
        d0 = cb.d0[rows]
        T[:, 0, :2] = fc[torch.arange(n, device=dev), f, :2] + d0[:, None] * torch.stack((torch.cos(psi0),
                                                                                            torch.sin(psi0)), -1)
        n_int = cb.n_int[rows]
        course = dict(targets=T, centre=C, n=cb.n[rows], stop=cb.stop[rows], planks=n_int > 0,
                      intervals=cb.intervals[rows].to(torch.float64), half_width=cb.half_width[rows],
                      origin=origin, heading=heading)
        self._install(ids, course, cur, dict(f0=cb.f0[rows], d0=d0, cliff=cb.cliff[rows], n_int=n_int,
                                             course_row=rows, course_seed=cb.seed[rows]))
        # previous command = the start's recorded command (clipped into the action ranges), previous residual 0
        c3 = torch.minimum(torch.maximum(b.command[si, :3], self.cmd_lo), self.cmd_hi)
        u3 = (2 * (c3 - self.cmd_lo) / (self.cmd_hi - self.cmd_lo) - 1).clamp(-0.999, 0.999)
        self.prev_u[ids] = torch.cat((u3, torch.zeros(n, NUM_ACTIONS - 3, **self.f64)), 1)
        self.prev_cmd[ids] = torch.cat((c3, torch.full((n, 1), float(cfg["action"]["height"]), **self.f64)), 1)
        self.gate[ids] = b.contact_before[si].to(torch.float64)
        self.t[ids] = 0
        for v in self.ep.values():
            v[ids] = 0.0

    @torch.inference_mode()
    def reset_from_snapshots(self, snaps, members):
        """Reset envs 0..n-1 from MuJoCo hand-over snapshots (spec 12.2 rule 13; E10): each snap has hist [4, 67]
        (the 4 last averaged states), teacher_history [6, 76], teacher_last_action [12], prev_cmd [4], prev_u [15],
        gate [2], course (FT.Course or its to_dict(): the placed course with target 0 set at the hand-over) and
        optionally progress (FootstepProgress.state() at the hand-over). members [n]: NRD member per snapshot."""
        n = len(snaps)
        if n > self.num_envs:
            raise ValueError(f"{n} snapshots > {self.num_envs} envs")
        progs = [s.get("progress") for s in snaps]
        for pr in progs:
            if pr is not None and (int(pr["pointer"]) != 0 or bool(pr.get("stop_phase")) or int(pr.get("restep", 0))):
                raise NotImplementedError("reset_from_snapshots takes hand-over snapshots only (fresh progress)")
        dev = self.device
        ids = torch.arange(n, device=dev)
        t = lambda k, dt=torch.float64: torch.as_tensor(np.stack([np.asarray(s[k]) for s in snaps]), device=dev).to(dt)
        hist = t("hist")
        self.hist[ids] = hist
        self.ok[ids] = True
        self.member[ids] = torch.as_tensor(list(members), device=dev).long()
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        mask[ids] = True
        self.teacher.reset(mask, t("teacher_history", torch.float32), t("teacher_last_action", torch.float32))
        courses = [s["course"] if isinstance(s["course"], FT.Course) else FT.Course.from_dict(s["course"]) for s in snaps]
        a = course_arrays(courses)
        ta = lambda k, dt=torch.float64: torch.as_tensor(a[k], device=dev).to(dt)
        course = dict(targets=ta("targets"), centre=ta("centre"), n=ta("n", torch.long), stop=ta("stop", torch.bool),
                      planks=ta("n_int", torch.long) > 0, intervals=ta("intervals"), half_width=ta("half_width"),
                      origin=ta("origin"), heading=ta("heading"))
        start_place = None
        if all(pr is not None for pr in progs):
            start_place = torch.as_tensor(np.stack([np.asarray(pr["last_place"], np.float64) for pr in progs]), **self.f64)
        self.level[ids] = ta("level", torch.long)
        self._install(ids, course, hist[:, -1], dict(f0=ta("f0", torch.long), d0=ta("d0"), cliff=ta("cliff", torch.bool),
                                                     n_int=ta("n_int", torch.long), course_row=torch.full((n,), -1, device=dev),
                                                     course_seed=ta("seed", torch.long), start_place=start_place))
        self.prev_u[ids], self.prev_cmd[ids], self.gate[ids] = t("prev_u"), t("prev_cmd"), t("gate")
        self.t[ids] = 0
        for v in self.ep.values():
            v[ids] = 0.0
        self._compute_obs()
        return self.obs_buf

    # ------------------------------------------------------------------ step
    @torch.inference_mode()
    def step(self, actions):
        cfg, dev, B = self.cfg, self.device, self.num_envs
        rw, rl, tr = cfg["reward"], cfg["rules"], self.tracker
        a = actions.to(dev).to(torch.float64).clamp(-float(cfg["action"]["clip"]), float(cfg["action"]["clip"]))
        if a.shape[1] != self.num_actions:
            raise ValueError(f"actions [B, {a.shape[1]}], the env has {self.num_actions}")
        if self.num_actions < NUM_ACTIONS:                    # command-only ablation: residual 0
            a = torch.cat((a, torch.zeros(B, NUM_ACTIONS - self.num_actions, **self.f64)), 1)
        u = torch.tanh(a)
        cmd3 = self.cmd_lo + (self.cmd_hi - self.cmd_lo) * (u[:, :3] + 1) / 2
        cmd = torch.cat((cmd3, torch.full((B, 1), float(cfg["action"]["height"]), **self.f64)), 1)
        cur = self.hist[:, -1]
        _, legs = self.teacher.step(cmd, cur)
        target = torch.cat((legs + self.alpha * u[:, 3:], self.default_upper.expand(B, -1)), 1)
        member = None if (len(self.members) == 1 and int(self.members[0]) == 0) else self.member
        nxt, g, _ = self.sim.step(self.hist, self.ok, target, member=member)
        nonfinite = ~torch.isfinite(nxt).all(-1)
        nxt = torch.where(nonfinite[:, None], cur, nxt)
        g = torch.where(nonfinite[:, None], torch.zeros_like(g), g)
        self.hist, self.ok = NrdSim.advance(self.hist, self.ok, nxt)
        gate = g > 0.5
        zero = torch.zeros(B, **self.f64)
        # ---- geometry and the scorer
        fp = foot_points(nxt)
        sph, cen = fp["spheres"], fp["center"]
        fc, fy = foot_pose(sph)
        th = yaw(nxt)
        p_old = tr.ptr
        c_old = tr.C[tr.ar, torch.minimum(p_old, tr.N - 1)]
        ev = tr.update(nxt[:, :2], th, fc[..., :2], fy, sph[..., :2], gate, nxt[:, 34:36], nxt[:, 39])
        # ---- rules
        planks = tr.planks
        if rl["support"] == "centre":
            em = float(rl["edge_margin"])
            cidx = support_index(path_coords(cen[..., :2], tr.origin, tr.heading), tr.iv, tr.hw, em, planks)
        else:
            cidx = None
        bad = self._rule_bad(ev["underm"], cidx) & planks[:, None]
        support = (gate & bad & (self.bad_prev if rl["two_state"] else True)).any(-1)
        tc = tilt_cos(nxt)
        fall = (nxt[:, 2] < rl["fall_z"]) | (tc < math.cos(math.radians(rl["fall_tilt_deg"])))
        unstable = (nxt[:, 2] < rl["min_z"]) | (tc < math.cos(math.radians(rl["max_tilt_deg"])))
        legs_hit = torch.zeros(B, dtype=torch.bool, device=dev)
        legs_d = None
        if self.legs_on:
            nb, ahead, na = rl["legs_sweep"]
            legs_d = swept_leg_distances(cur, nxt, int(nb), float(ahead), int(na))
            if rl.get("legs_min_dist") is not None:
                legs_hit = legs_hit | (legs_d["legs"] < float(rl["legs_min_dist"]))
            if rl.get("legs_pelvis_min_dist") is not None:
                legs_hit = legs_hit | (legs_d["pelvis"] < float(rl["legs_pelvis_min_dist"]))
        jr = torch.zeros(B, dtype=torch.bool, device=dev)
        if self.q_lo is not None:
            q = nxt[:, 7:34]
            jr = ((q < self.q_lo) | (q > self.q_hi)).any(-1)
        reasons = dict(support=support & ~nonfinite, unstable=unstable & ~nonfinite, legs=legs_hit & ~nonfinite,
                       joint_range=jr & ~nonfinite, nonfinite=nonfinite)
        phys = nonfinite | support | unstable | legs_hit | jr
        for i, k in enumerate(FT.FAIL_REASONS):
            reasons[k] = ev["failed"] & (ev["fail"] == i)
        fail = phys | ev["failed"]
        completed = ev["completed"] & ~fail
        finished = ev["finished"] & ~fail
        # ---- reward
        r_td, r_tol, r_win, r_miss, r_yaw = zero.clone(), zero.clone(), zero.clone(), zero.clone(), zero.clone()
        n_restep, n_wrong = zero.clone(), zero.clone()
        tol = float(cfg["score"]["tol"])
        td_err, td_n = zero.clone(), zero.clone()
        for ps in ev["passes"]:
            sc = ps["scored"].to(torch.float64)
            e = ps["e"]
            r_td = r_td + sc * rw["hit"] * (0.5 * torch.exp(-(e / rw["hit_wide"]) ** 2)
                                            + 0.5 * torch.exp(-(e / rw["hit_narrow"]) ** 2))
            r_tol = r_tol + sc * rw["tol_bonus"] * (e <= tol).to(torch.float64)
            r_win = r_win + rw["window"] * ps["window"].to(torch.float64)
            r_miss = r_miss - sc * rw["miss"] * (e > rw["miss_dist"]).to(torch.float64)
            r_yaw = r_yaw + sc * rw["yaw"] * torch.exp(-(ps["yaw_err"] / rw["yaw_scale"]) ** 2)
            n_restep = n_restep + ps["restep"].to(torch.float64)
            n_wrong = n_wrong + ps["wrong_foot"].to(torch.float64)
            td_err, td_n = td_err + sc * e, td_n + sc
        r_touch = r_td + r_tol + r_win + r_miss + r_yaw
        r_restep = -rw["restep"] * n_restep
        r_wrong = -rw["wrong_foot"] * n_wrong
        r_stop = rw["stop"] * ev["stop_success"].to(torch.float64)
        succ_all = completed & (tr.err_max <= tol)
        r_comp = rw["complete"] * completed.to(torch.float64) + rw["success"] * succ_all.to(torch.float64)
        dxy = nxt[:, :2] - cur[:, :2]
        ds = dxy[:, 0] * torch.cos(c_old[:, 2]) + dxy[:, 1] * torch.sin(c_old[:, 2])
        r_prog = rw["progress"] * ds.clamp(-rw["progress_clip"], rw["progress_clip"]) * (p_old < tr.N).to(torch.float64)
        l, dpsi = ev["l"], ev["dpsi"]
        r_lat = -rw["lateral"] * (l / rw["lateral_scale"]).square().clamp(max=1.0)
        cos_h = math.cos(math.radians(rw["heading_scale_deg"]))
        r_head = -rw["heading"] * ((1 - torch.cos(dpsi)) / (1 - cos_h)).clamp(max=1.0)
        r_res = -rw["residual"] * u[:, 3:].square().mean(-1) if self.alpha > 0 else zero
        r_rate = -rw["action_rate"] * (u - self.prev_u).square().mean(-1) / 4
        cos_max = math.cos(math.radians(rl["max_tilt_deg"]))
        r_tilt = -rw["tilt"] * ((1 - tc) / (1 - cos_max)).clamp(0.0, 1.0)
        both = gate & (self.gate > 0.5)
        slip = ((cen[..., :2] - self.centre_prev[..., :2]).norm(dim=-1) / 0.02 * both).sum(-1)
        r_slip = -rw["slip"] * slip.clamp(max=float(rw["slip_cap"])) / float(rw["slip_cap"])
        r_reg = r_res + r_rate + r_tilt + r_slip
        r_events = r_restep + r_wrong + r_stop + r_comp
        r = r_touch + r_prog + r_lat + r_head + r_reg + r_events
        r = torch.where(fail, torch.full_like(r, -float(rw["fail"])), r)
        self.t = self.t + 1
        time_out = ~fail & ~completed & ~finished & (self.t >= self.max_episode_length)
        dones = fail | completed | finished | time_out
        # ---- bookkeeping
        keep = lambda x: torch.where(fail, zero, x)
        self.ep["reward"] += r
        self.ep["touchdown"] += keep(r_touch)
        self.ep["progress"] += keep(r_prog)
        self.ep["drift"] += keep(r_lat + r_head)
        self.ep["reg"] += keep(r_reg)
        self.ep["events"] += keep(r_events)
        self.gate, self.prev_u, self.prev_cmd = g.to(torch.float64), u, cmd
        self.bad_prev, self.centre_prev = bad, cen
        self.episode_length_buf = self.t
        reasons["fall"] = fall & ~nonfinite
        res = tr.results()            # the tracker's episode values; the outcome flags below decide (E6)
        step_info = dict(completed=completed, success=completed & (tr.err_max <= tol),
                         clean_completion=completed & res["clean"], finished=finished, fail=fail, time_out=time_out,
                         reasons={k: v & fail for k, v in reasons.items()}, level=self.level.clone(),
                         steps=self.t.clone(), member=self.member.clone(), course_row=self.course_row.clone(),
                         course_seed=self.course_seed.clone(), episode=res,
                         returns={k: v.clone() for k, v in self.ep.items()}, command=cmd, state=nxt, gates=gate,
                         touchdown=ev["raw"], target=target, pointer=tr.ptr.clone(), legs_dist=legs_d, tracker=ev)
        parts = dict(touchdown=r_td, tol_bonus=r_tol, window=r_win, miss=r_miss, yaw=r_yaw, progress=r_prog,
                     lateral=r_lat, heading=r_head, residual=r_res, action_rate=r_rate, tilt=r_tilt, slip=r_slip,
                     restep=r_restep, wrong_foot=r_wrong, stop=r_stop, complete=r_comp, reward=r)
        log = self._log(parts, step_info, dones, u, cmd3, td_err, td_n)
        self._compute_obs()
        extras = {"time_outs": time_out, "final_observations": self.obs_buf.clone(), "log": log,
                  "step_info": step_info, "observations": {}}
        if self.auto_reset:
            ids = torch.nonzero(dones).flatten()
            if len(ids):
                self.reset_idx(ids)
                self._compute_obs()
        extras["observations"]["critic"] = self.obs_buf
        self.extras = extras
        return self.obs_buf, r.to(torch.float32), dones.long(), extras

    # ------------------------------------------------------------------ logs
    def _log(self, parts, si, dones, u, cmd3, td_err, td_n):
        f = lambda x: x.to(torch.float64)
        nan = torch.tensor(float("nan"), **self.f64)
        lg = {f"/step/{k}": v.mean() for k, v in parts.items()}
        lg["/step/vx_cmd"], lg["/step/vy_cmd"], lg["/step/wz_cmd"] = cmd3.mean(0)
        lg["/step/res_sat"] = (u[:, 3:].abs() > 0.95).to(torch.float64).mean() if self.alpha > 0 else nan
        n_td = td_n.sum()
        lg["/step/td_err_cm"] = 100 * td_err.sum() / n_td if float(n_td) > 0 else nan
        lg["/step/scored"] = td_n.mean()
        ids = torch.nonzero(dones).flatten()
        if len(ids):
            ep = si["episode"]
            sc = f(ep["scored"][ids])
            pooled = lambda x: f(x[ids]).sum() / sc.sum() if float(sc.sum()) > 0 else nan
            cur = dict(completion=f(si["completed"][ids]).mean(), clean_completion=f(si["clean_completion"][ids]).mean(),
                       success=f(si["success"][ids]).mean(), finished=f(si["finished"][ids]).mean(),
                       fail=f(si["fail"][ids]).mean(), time_out=f(si["time_out"][ids]).mean(),
                       length=f(self.t[ids]).mean(), ret=self.ep["reward"][ids].mean(),
                       targets_frac=(f(ep["pointer"][ids]) / f(ep["n"][ids])).mean(), scored=sc.mean(),
                       hit3=pooled(ep["hits3"]), hit5=pooled(ep["hits5"]), hit8=pooled(ep["hits8"]),
                       err_cm=100 * ep["err_sum"][ids].sum() / sc.sum() if float(sc.sum()) > 0 else nan,
                       resteps=f(ep["resteps"][ids]).mean(), resteps_free=f(ep["resteps_free"][ids]).mean(),
                       wrong_foot=f(ep["wrong_foot"][ids]).mean(), far_miss=f(ep["far_miss"][ids]).mean(),
                       flickers=f(ep["flickers"][ids]).mean(), overruns=f(ep["overruns"][ids]).mean(),
                       touchdowns=f(ep["touchdowns"][ids]).mean(), td_over_gap=f(ep["touchdowns_over_gap"][ids]).mean(),
                       unsupported=f(ep["unsupported_steps"][ids]).mean(), max_abs_l=ep["max_abs_l"][ids].mean(),
                       max_abs_dpsi_deg=torch.rad2deg(ep["max_abs_dpsi"][ids]).mean())
            for k in LOG_REASONS:
                cur[f"fail_{k}"] = f(si["reasons"][k][ids]).mean()
            for k, v in cur.items():
                o = self.done_stats[k]
                self.done_stats[k] = torch.where(torch.isnan(v), o, torch.where(torch.isnan(o), v, 0.95 * o + 0.05 * v))
            # per level: shares among the episodes of each level that ended now
            hit = self.level[ids][:, None] == torch.arange(len(LEVELS), device=self.device)[None]     # [n, L]
            cnt = hit.sum(0)
            sc_l = (hit * sc[:, None]).sum(0)
            vals = [f((hit & si["completed"][ids][:, None]).sum(0)) / cnt.clamp(min=1),
                    f((hit & si["clean_completion"][ids][:, None]).sum(0)) / cnt.clamp(min=1),
                    f((hit & si["fail"][ids][:, None]).sum(0)) / cnt.clamp(min=1),
                    torch.where(sc_l > 0, (hit * f(ep["hits5"][ids])[:, None]).sum(0) / sc_l.clamp(min=1), nan)]
            vals += [f((hit & si["reasons"][k][ids][:, None]).sum(0)) / cnt.clamp(min=1) for k in LOG_REASONS]
            share = torch.stack(vals)
            ok = (cnt > 0)[None] & ~torch.isnan(share)
            old = self.level_stats
            self.level_stats = torch.where(ok, torch.where(torch.isnan(old), share, 0.95 * old + 0.05 * share), old)
        for k, v in self.done_stats.items():
            lg[f"/episode/{k}"] = v
        for j in self.active_levels:
            for r, k in enumerate(LEVEL_KEYS):
                lg[f"/level/{LEVELS[j]}/{k}"] = self.level_stats[r, j]
        return lg

    # ------------------------------------------------------------------ observation
    def _compute_obs(self):
        self.obs_buf = self.observe()
        return self.obs_buf

    def course_view(self):
        tr = self.tracker
        return dict(targets=tr.T, centre=tr.C, n=tr.N, slack=self.slack)

    def observe(self):
        """The observation of the newest state of every env [B, num_obs] float32: footsteps_observation (shared with
        the MuJoCo runner)."""
        st = self.hist[:, -1]
        raw = homie_preview(self.teacher, self.net, self.prev_cmd, st, self.cfg["obs"]["homie_preview"])
        return footsteps_observation(st, raw, self.prev_u, self.gate, self.tracker.view(), self.course_view(),
                                     self.cfg["obs"])

    # ------------------------------------------------------------------ helpers for checks and MuJoCo
    def course_of(self, i):
        """Env i's current course as a placed FT.Course (world targets with target 0 as set at the reset, course-frame
        intervals, origin = the start foot midpoint, heading = the start pelvis yaw)."""
        tr = self.tracker
        n, p = int(tr.N[i]), int(self.n_int[i])
        np_ = lambda x: x.detach().cpu().numpy().astype(np.float64)
        return FT.Course(origin=np_(tr.origin[i]), heading=float(tr.heading[i]), targets=np_(tr.T[i, :n]),
                         centre=np_(tr.C[i, :n]), f0=int(self.f0[i]), d0=float(self.d0[i]),
                         end="stop" if bool(tr.stop_end[i]) else "through", cliff=bool(self.cliff[i]),
                         intervals=np_(tr.iv[i, :p]) if p else None, half_width=float(tr.hw[i]),
                         level=LEVELS[int(self.level[i])], seed=int(self.course_seed[i]),
                         program=dict(bank_row=int(self.course_row[i])))

    def command_of(self, actions):
        """The HOMIE command (vx, vy, wz) [B, 3] of actions [B, num_actions]."""
        u = torch.tanh(actions[:, :3].to(self.device).to(torch.float64).clamp(-float(self.cfg["action"]["clip"]),
                                                                              float(self.cfg["action"]["clip"])))
        return self.cmd_lo + (self.cmd_hi - self.cmd_lo) * (u + 1) / 2

    def action_of_command(self, cmd3):
        """Inverse of command_of for the command part (residual 0): pre-tanh actions [B, num_actions]."""
        c = torch.as_tensor(cmd3, **self.f64)
        u = (2 * (c - self.cmd_lo) / (self.cmd_hi - self.cmd_lo) - 1).clamp(-0.999999, 0.999999)
        return torch.cat((torch.atanh(u), torch.zeros(c.shape[0], self.num_actions - 3, **self.f64)), 1)

    @property
    def action_groups(self):
        """Action counts per group: [3] (command only) or [3, 12] (command, leg residual)."""
        return [3] if self.num_actions == 3 else [3, NUM_ACTIONS - 3]
