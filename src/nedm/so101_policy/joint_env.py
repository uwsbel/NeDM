"""Joint-target task env and the Markov joint teacher of build9 (``~/so101_policy/docs/build9_viral_student.md``).

Why: the study's PPO teacher acts through ``decoder.DecoderV2`` (hidden macro phases, clocks, line plans that read the
true cube pose), so its label at a state that a camera student reached is ill-defined (build8: DAgger 36 -> 1 %). The
joint teacher ("JT") is Markov: its action at ANY state is a function of the privileged observation of that state.

Observation (``OBS_DIM`` = 47): ``task.observation`` without the decoder state and macro phase (q, qd, TCP, gripper axes,
cube position / yaw / tilt / v / w, cube in the grasp frame, relative yaw, goal, goal - cube, yaw error, the "lifted"
flag and the held-in-air time: the last two are task-tracker values owned by the env, as VIRAL's task stage).

Action (``ACT_DIM`` = 18): one decision every 0.1 s = the 5 joint targets of the next 5 control steps (20 ms) relative
to q at the decision start, D[j] = PHI[j] @ c, c = 3 coefficients per joint (``PHI``: 1, x, (3 x^2 - 1) / 2 over
x = -1 .. 1), normalised: c = mean + std * a (``Codec``; a in data units, no tanh). The env executes the targets
through the ACTION LAYER (``safety``, the same for the student): each arm target clamped to the decoder's lead band
around the MEASURED q of that step (``LEAD``, data p99.9), every target to the collector's target range, and the table
floor (``floor_guard``: the decoder's floor rule in joint space, lowest gripper / moving-jaw point >= 1.5 mm unless the
measured pose is already lower). Stateless: it needs only the measured q and the robot's kinematics, so the label stays
a function of the state.

``JointVecEnv``: B worlds (``vec_env.make_world``: nrd, kin, mjwarp, mujoco) + ``task.PickPlaceTask`` with build8's
macro-free rules (``tray.LOG_ONLY`` log only) + the lenient success tracker (``attach_lenient``). PPO interface of
``ppo.PPO`` (num_envs, num_obs, num_actions, steps_per_decision, reset, step -> obs, reward, done, extras with
action_mask / time_outs / final_obs / episode, gen, ep_count, resolved). ``step(a, targets=...)`` takes absolute
targets [B, 5, 6] instead (student-driven rows). Auto reset from the start bank + ``task.sample_goals`` (training), or
``reset_tasks`` (fixed tasks: evaluation, DAgger collection).

Lenient success (build9 sec. 1.3): released (not held, jaw >= open_jaw, TCP >= tcp_clear above the cube top), at rest
(speed < rest_speed, tilt <= flat_tilt, on the table), after a firm lift, the cube centre inside the goal block's square
(|x|, |y| <= ``BLOCK_HALF`` in the block frame, any yaw), no failure rule in that step; latched per episode. The task
still ends at strict success (15 mm / 15 deg), a failure or the time-out.

Robot dynamics (real2sim, OFF by default: ``JointEnvConfig.dyn`` None): ``RobotDyn`` sits between the action layer and
the world step of every 20 ms step and emulates the real SO101's timing and servo per world (command delay with jitter,
weaker arm servo, first-order command lag, gravity sag of shoulder_lift / elbow with the robot guard's integral
correction, command noise; ``RobotDynConfig``), drawn at every episode reset. ``joint_teacher.make_env`` reads a fixed
evaluation setting from ``$NEDM_SO101_DYN``.
"""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy import task as T
from nedm.so101_policy.world import CONTROL_DT, WorldState

OBS_DIM = 47
STEPS = 5
NB = 3
ACT_DIM = NB * 6
ACTION_GROUPS = (("arm", NB * 5), ("jaw", NB))      # a = c.view(3, 6) flattened: [k0 j0..j5, k1 ..., k2 ...]
LEAD = (0.0445, 0.1235, 0.0552, 0.0559, 0.0476)      # decoder lead band (rad), |cmd - q| data p99.9
BLOCK_HALF = 0.025                                   # goal block half side (m): lenient success square
LOG_ONLY = ("grasp_box", "keep_out", "cube_slide")   # build8's macro-defined rules: log only
PP_DATA = Path.home() / "so101_nrd_fix" / "data" / "production_plus"
_X = (torch.arange(STEPS, dtype=torch.float64) - (STEPS - 1) / 2) / ((STEPS - 1) / 2)
PHI = torch.stack((torch.ones_like(_X), _X, (3 * _X ** 2 - 1) / 2), -1)        # [5, 3]


# ---- action codec ------------------------------------------------------------------------------------------------
class Codec:
    """Normalised coefficients a [B, 18] <-> relative targets D [B, 5, 6] (module docstring)."""

    def __init__(self, mean, std, device="cpu"):
        self.dev = torch.device(device)
        self.mean = torch.as_tensor(mean, dtype=torch.float64).reshape(NB, 6).to(self.dev)
        self.std = torch.as_tensor(std, dtype=torch.float64).reshape(NB, 6).to(self.dev)
        self.phi = PHI.to(self.dev)
        self.pinv = torch.linalg.pinv(PHI).to(self.dev)                          # [3, 5]

    @staticmethod
    def fit(D, floor=(1e-3, 1e-3, 1e-3)):
        """Codec from relative targets D [N, 5, 6] (float): mean / std of the least-squares coefficients."""
        D = torch.as_tensor(D, dtype=torch.float64)
        c = torch.einsum("kj,njd->nkd", torch.linalg.pinv(PHI), D)
        fl = torch.as_tensor(floor, dtype=torch.float64)[:, None]
        return Codec(c.mean(0), torch.maximum(c.std(0), fl))

    def to(self, device):
        return Codec(self.mean, self.std, device)

    def state(self):
        return dict(mean=self.mean.cpu().numpy(), std=self.std.cpu().numpy())

    def coefs(self, D):
        return torch.einsum("kj,njd->nkd", self.pinv.to(D), D)

    def encode(self, D):
        """[N, 5, 6] -> normalised a [N, 18]."""
        return ((self.coefs(D) - self.mean.to(D)) / self.std.to(D)).reshape(D.shape[0], ACT_DIM)

    def decode(self, a):
        """normalised a [N, 18] -> relative targets [N, 5, 6]."""
        c = self.mean.to(a) + self.std.to(a) * a.reshape(a.shape[0], NB, 6)
        return torch.einsum("jk,nkd->njd", self.phi.to(a), c)

    def targets(self, a, q):
        """normalised a [B, 18], q [B, 6] at the decision start -> absolute targets [B, 5, 6]."""
        return q[:, None, :].to(a) + self.decode(a)


FLOOR = 0.0015                                       # m, the decoder's low_min (lowest gripper / moving-jaw point)
# evaluation only (real2sim debugging): a weaker arm servo, as the real STS3215 (P 16): the executed arm target is moved
# to q + s * (target - q), i.e. the position gain times s; 1 = off. Not used in training or data collection. Applied
# after the robot dynamics stage (``RobotDyn``; the two compound); per world and recorded: ``RobotDynConfig.s_*``.
EVAL_KP_SCALE = float(os.environ.get("NEDM_SO101_EVAL_KP_SCALE", "1"))


def _low(kin, u, q, plane=None):
    """Lowest gripper / moving-jaw collision point (m) of the arm pose u[:, :5] with the commanded and the measured jaw
    angle (the smaller). ``plane``: height above that plane instead of world z (``table.py``)."""
    kw = {} if plane is None else dict(plane=plane)
    a = kin.lowest_point(u, exact=False, **kw)
    b = kin.lowest_point(torch.cat((u[:, :5], q[:, 5:6]), -1), exact=False, **kw)
    return torch.minimum(a, b)


def floor_guard(u, q, kin, floor=FLOOR, iters=8, plane=None):
    """Table floor of the action layer (the decoder's floor rule in joint space): a target whose lowest gripper /
    moving-jaw point is below ``floor`` AND below that of the measured pose is pulled back toward the measured q along
    the straight joint line, u' = q + lam (u - q) with the largest lam (bisection) that keeps it at or above
    min(floor, the measured pose's lowest point). Stateless (robot kinematics + measured q). u, q [B, 6] -> [B, 6].
    ``plane`` (c0, c1, c2): heights above the plane z = c0 + c1 x + c2 y instead of world z (a floor that follows a
    tilted table; a flat floor at height h is ``floor`` = FLOOR + h, as the robot's ``--table-z``)."""
    lu = _low(kin, u, q, plane)
    bad = lu < floor - 1e-9
    if not bool(bad.any()):
        return u
    lq = _low(kin, q, q, plane)
    ref = torch.minimum(lq, torch.full_like(lq, floor))
    bad = bad & (lu < ref - 1e-9)
    if not bool(bad.any()):
        return u
    idx = bad.nonzero().flatten()
    ub, qb, rb = u[idx], q[idx], ref[idx]
    lo = torch.zeros(len(idx), dtype=u.dtype, device=u.device)
    hi = torch.ones_like(lo)
    d = torch.cat((ub[:, :5] - qb[:, :5], torch.zeros_like(ub[:, 5:])), -1)
    base = torch.cat((qb[:, :5], ub[:, 5:]), -1)
    for _ in range(iters):
        mid = (lo + hi) / 2
        ok = _low(kin, base + mid[:, None] * d, qb, plane) >= rb - 1e-9
        lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid)
    out = u.clone()
    out[idx] = base + lo[:, None] * d
    return out


def safety(u, q, tlo, thi, lead=None, kin=None, floor=None, floor_plane=None):
    """The action layer of teacher AND student: arm targets within ``lead`` of the measured q, the table floor
    (``floor_guard`` with ``kin``, if ``floor`` is given; ``floor_plane``: heights above that plane), all targets
    within the collector's range. u, q [B, 6] -> [B, 6]."""
    if lead is not None:
        arm = torch.minimum(torch.maximum(u[:, :5], q[:, :5] - lead), q[:, :5] + lead)
        u = torch.cat((arm, u[:, 5:]), -1)
    u = torch.minimum(torch.maximum(u, tlo), thi)
    if kin is not None and floor is not None:
        u = floor_guard(u, q, kin, floor) if floor_plane is None else floor_guard(u, q, kin, floor, plane=floor_plane)
    return u


# ---- robot dynamics emulation (real2sim) -------------------------------------------------------------------------
DYN_ENV = "NEDM_SO101_DYN"              # JSON ``RobotDynConfig`` of the evaluation envs (``joint_teacher.make_env``)
SAG_JOINTS = (1, 2)                     # shoulder_lift, elbow: the joints that settle below their command on the robot
_DYN_SHORT = dict(d=("d_lo", "d_hi"), s=("s_lo", "s_hi"), tau=("tau_lo", "tau_hi"),
                  sag_deg=("sag_lo_deg", "sag_max_deg"), noise=("noise_lo", "noise_max"))


@dataclass
class RobotDynConfig:
    """The real SO101's timing and servo, emulated per world between the action layer and the world step (``RobotDyn``;
    robot values: ~/so101_r2s_claude/sim_debug/scout/timing.md: q-read-to-first-target 46 ms = 2.3 steps, the robot
    holds the previous decision's last target until the new ones are written, servo delay 80-132 ms, achieved /
    commanded motion per decision 0.3-0.5, measured - commanded SENT +1.5 deg shoulder_lift / +1.9 deg elbow under
    load, with the guard's 1-2.8 deg integral correction already in the sent command). Every value is drawn per world
    at its episode reset; a world is perturbed with probability ``p`` (the others take the old path bit for bit).
    Ranges are inclusive, lo == hi is a fixed value (evaluation). Short forms in ``from_dict`` set both ends of a
    range: d, s, tau, sag_deg, noise. Stage order per 20 ms step: delay -> integral correction -> lag -> sag -> gain
    -> noise -> actuator range (the lag filters the COMMAND; the gain acts on the error to the measured q of this
    step). The sag is added before the gain, so the stage's own offset is the sag for any gain; a world with gravity
    adds its own error at a low gain on top (MuJoCo, measured - command: s 0.1 alone +0.14 / +0.21 deg, s 0.1 with a
    1.7 deg sag +1.84 / +1.91 deg). The integral correction (``i_time_s`` > 0) is the robot guard's
    (``hw_lib.GuardConfig.i_*``, robot runs: i_time_s 0.2, i_max_deg 3): without it a constant sag moves a policy
    that commands relative to the measured q by the sag at every decision (jt_h110: 0.8 % lenient at 0.5 deg)."""
    p: float = 1.0                      # share of perturbed worlds (per episode)
    d_lo: int = 0                       # command delay (20 ms steps), per episode U{d_lo .. d_hi}: the first d steps
    d_hi: int = 0                       # of a decision run the previous decision's last command, then u5[0 .. 4 - d]
    jitter: float = 0.0                 # per decision d + (-1, 0, +1) with probability j / 2, 1 - j, j / 2 (j = jitter)
    s_lo: float = 1.0                   # arm servo gain scale, log-uniform: u_arm = q + s (u_arm - q) (jaw: unchanged)
    s_hi: float = 1.0
    tau_lo: float = 0.0                 # first-order lag of the arm + jaw command (s), log-uniform, 0 = off:
    tau_hi: float = 0.0                 # f <- f + min(1, DT / tau) (u - f), f = the measured q at the reset
    sag_lo_deg: float = 0.0             # static error (deg), uniform: + sag on the shoulder_lift and elbow commands
    sag_max_deg: float = 0.0            # (measured q = command + sag, the robot's sign)
    noise_lo: float = 0.0               # Gaussian noise on the arm commands, every step; sigma (rad) uniform per
    noise_max: float = 0.0              # episode
    i_time_s: float = 0.0               # integral correction of the robot guard (0 = off; arm joints, every step):
    i_max_deg: float = 3.0              #   integ += (command - q) DT / i_time_s, |integ| <= i_max_deg, only while
    i_e_max_deg: float = 1.0            #   |command - q| < i_e_max_deg and the joint does not move to its command
    i_v_min: float = 0.02               #   faster than i_v_min (rad/s); the sent command = command + integ

    def __post_init__(self):
        if not 0.0 <= self.p <= 1.0 or not 0.0 <= self.jitter <= 1.0:
            raise ValueError(f"robot dyn: p {self.p}, jitter {self.jitter} must be in [0, 1]")
        if not 0 <= int(self.d_lo) <= int(self.d_hi) <= STEPS - 1:
            raise ValueError(f"robot dyn: delay {self.d_lo} .. {self.d_hi} must be within 0 .. {STEPS - 1} steps")
        if not 0.0 < self.s_lo <= self.s_hi:
            raise ValueError(f"robot dyn: gain {self.s_lo} .. {self.s_hi} must be > 0, lo <= hi")
        if not 0.0 <= self.tau_lo <= self.tau_hi or (self.tau_lo == 0.0 < self.tau_hi):
            raise ValueError(f"robot dyn: tau {self.tau_lo} .. {self.tau_hi}: log-uniform, both > 0 (or both 0 = off)")
        if not self.sag_lo_deg <= self.sag_max_deg or not 0.0 <= self.noise_lo <= self.noise_max:
            raise ValueError("robot dyn: sag / noise ranges need lo <= hi (noise >= 0)")
        if self.i_time_s < 0.0 or self.i_max_deg < 0.0 or self.i_e_max_deg < 0.0:
            raise ValueError("robot dyn: integral correction i_time_s / i_max_deg / i_e_max_deg must be >= 0")
        self.d_lo, self.d_hi = int(self.d_lo), int(self.d_hi)

    @staticmethod
    def from_dict(d):
        d = dict(d or {})
        known = {f.name for f in fields(RobotDynConfig)}
        bad = sorted(set(d) - known - set(_DYN_SHORT) - {"note"})
        if bad:
            raise ValueError(f"unknown robot dyn keys {bad}")
        for k, (lo, hi) in _DYN_SHORT.items():
            if k in d:
                if lo in d or hi in d:
                    raise ValueError(f"robot dyn: {k!r} sets {lo} and {hi}; give one form")
                d[lo] = d[hi] = d.pop(k)
        return RobotDynConfig(**{k: v for k, v in d.items() if k in known})

    @property
    def neutral(self):
        """True: every perturbed world equals the old path (no delay, gain 1, no lag, sag, noise or integral
        correction); ``RobotDyn`` then skips the stage (bit for bit the old step)."""
        return (self.d_hi == 0 and self.s_lo == self.s_hi == 1.0 and self.tau_hi == 0.0 and self.sag_lo_deg ==
                self.sag_max_deg == 0.0 and self.noise_max == 0.0 and self.i_time_s == 0.0)


def dyn_config(v) -> RobotDynConfig | None:
    """None / False / {} / "" / "{}" / "null" -> None (off); a dict or a JSON string -> ``RobotDynConfig``."""
    if isinstance(v, RobotDynConfig):
        return v
    if isinstance(v, str):
        v = json.loads(v) if v.strip() else None
    if v is None or v is False or (isinstance(v, dict) and not v):
        return None
    if not isinstance(v, dict):
        raise ValueError(f"robot dyn: a dict / JSON object, got {v!r}")
    return RobotDynConfig.from_dict(v)


@dataclass
class PauseConfig:
    """The real robot's hold state (real2sim, 2026-10-22): on the robot the student carries the cube over the plate,
    then the arm stands still 1-5 cm above it (qdot exactly 0) and the student never releases; in the sim the arm
    never stands still there, so the student has no data of that state. A pause freezes the 5 arm joints at their
    measured q for ``s_lo``-``s_hi`` s (the jaw follows its command), started at a decision where the cube is held,
    lifted, within ``near_m`` (xy) of the goal and its bottom ``z_lo``-``z_hi`` above z = 0, with probability ``rate``
    per decision, at most ``max_n`` times per episode, in a share ``p`` of the episodes. The JT labels the still
    states ("descend"); after the pause the normal physics continue. Training only (``JointEnvConfig.pause``)."""
    p: float = 0.5
    s_lo: float = 0.5
    s_hi: float = 3.0
    near_m: float = 0.04
    z_lo: float = 0.008
    z_hi: float = 0.06
    rate: float = 0.25
    max_n: int = 2

    def __post_init__(self):
        if not (0.0 <= self.p <= 1.0 and 0.0 <= self.rate <= 1.0 and 0.0 < self.s_lo <= self.s_hi
                and self.z_lo <= self.z_hi and self.near_m > 0 and int(self.max_n) >= 1):
            raise ValueError(f"pause config out of range: {self}")
        self.max_n = int(self.max_n)


def pause_config(v) -> PauseConfig | None:
    """None / False / {} / "" -> None (off); a dict or a JSON string -> ``PauseConfig``."""
    if isinstance(v, PauseConfig):
        return v
    if isinstance(v, str):
        v = json.loads(v) if v.strip() else None
    if v is None or v is False or (isinstance(v, dict) and not v):
        return None
    d = {k: x for k, x in dict(v).items() if k != "note"}
    bad = sorted(set(d) - {f.name for f in fields(PauseConfig)})
    if bad:
        raise ValueError(f"unknown pause keys {bad}")
    return PauseConfig(**d)


class Pause:
    """Per-world state of ``PauseConfig`` (``JointVecEnv.step``), own generator."""

    def __init__(self, cfg: PauseConfig, B, device, seed):
        self.cfg, self.B, self.dev = cfg, int(B), torch.device(device)
        self.gen = torch.Generator(self.dev).manual_seed(int(seed))
        self.on = torch.zeros(B, dtype=torch.bool, device=self.dev)
        self.n = torch.zeros(B, dtype=torch.long, device=self.dev)        # pauses so far in this episode
        self.left = torch.zeros(B, dtype=torch.long, device=self.dev)     # steps left in the current pause
        self.qh = torch.zeros(B, 6, dtype=torch.float64, device=self.dev)  # the frozen arm q
        self.started = 0                                                  # pauses started (log)

    def reset(self, idx):
        u = torch.rand(len(idx), generator=self.gen, dtype=torch.float64, device=self.dev)
        self.on[idx] = u < self.cfg.p
        self.n[idx], self.left[idx] = 0, 0

    def begin(self, task, q):
        """Decision start: start a pause where the condition holds (task.F of the last step, q [B, 6] measured)."""
        c, F = self.cfg, task.F
        if F is None:
            return
        r = torch.rand(self.B, 2, generator=self.gen, dtype=torch.float64, device=self.dev)
        dxy = (F["c"][:, :2] - task.goal_xy).norm(dim=-1)
        zb = F["c"][:, 2] - T.CUBE_HALF
        go = (self.on & (self.left == 0) & (self.n < c.max_n) & F["held"] & task.tr.lifted & (dxy < c.near_m)
              & (zb >= c.z_lo) & (zb <= c.z_hi) & (r[:, 0] < c.rate))
        if bool(go.any()):
            steps = torch.round((c.s_lo + r[:, 1] * (c.s_hi - c.s_lo)) / CONTROL_DT).long().clamp(min=1)
            self.left = torch.where(go, steps, self.left)
            self.qh = torch.where(go[:, None], q.to(torch.float64), self.qh)
            self.n = self.n + go.long()
            self.started += int(go.sum())

    def apply(self, u):
        """One 20 ms step: the paused worlds' arm targets = the frozen q (the jaw keeps its command)."""
        m = self.left > 0
        if not bool(m.any()):
            return u
        out = torch.where(m[:, None], torch.cat((self.qh[:, :5], u[:, 5:]), -1), u)
        self.left = (self.left - 1).clamp(min=0)
        return out

    def summary(self):
        return dict(pause_on=round(float(self.on.double().mean()), 3),
                    pause_now=round(float((self.left > 0).double().mean()), 3), pause_started=int(self.started))


def dyn_from_env() -> RobotDynConfig | None:
    """The evaluation setting of ``$NEDM_SO101_DYN`` (JSON ``RobotDynConfig``, e.g. '{"d": 2, "s": 0.1, "tau": 0.1,
    "sag_deg": 1.7}'); unset / empty = off."""
    return dyn_config(os.environ.get(DYN_ENV, "").strip())


def _log_uniform(lo, hi, u):
    if lo == hi:
        return torch.full_like(u, float(lo))
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


class RobotDyn:
    """Per-world state of ``RobotDynConfig`` for B worlds on one device (``JointVecEnv.step``). Own generator (seeded
    from the env seed; ``reseed`` at ``reset_tasks``), so the env's other draws never change. Six uniforms per world
    and reset (perturbed, delay, gain, tau, sag, sigma), B uniforms per decision (jitter) and B x 5 normals per step
    (noise), all drawn whatever the ranges are, so paired settings share their draws. A neutral config
    (``RobotDynConfig.neutral``) never activates the stage."""
    COLS = ("on", "d0", "s", "tau", "sag_deg", "sigma")
    SUMMARY = ("dyn_on", "dyn_d0", "dyn_d", "dyn_s_p50", "dyn_tau_p50", "dyn_sag_deg", "dyn_sigma", "dyn_i_deg")

    def __init__(self, cfg: RobotDynConfig, B, device, seed):
        self.cfg, self.B, self.dev = cfg, int(B), torch.device(device)
        self.gen = torch.Generator(self.dev).manual_seed(int(seed))
        f, li = dict(dtype=torch.float64, device=self.dev), dict(dtype=torch.long, device=self.dev)
        self.on = torch.zeros(B, dtype=torch.bool, device=self.dev)
        self.d0, self.d = torch.zeros(B, **li), torch.zeros(B, **li)
        self.s, self.tau, self.a = torch.ones(B, **f), torch.zeros(B, **f), torch.ones(B, **f)
        self.sag, self.sigma = torch.zeros(B, **f), torch.zeros(B, **f)
        self.hold = torch.zeros(B, 6, **f)        # the previous decision's last command (held during the delay)
        self.f = torch.zeros(B, 6, **f)           # the lag filter's state
        self.integ = torch.zeros(B, 5, **f)       # the integral correction (rad, arm joints; i_time_s > 0)
        self.active = False                       # any world perturbed (else ``JointVecEnv.step`` skips the stage)
        self.eligible = None                      # [B] bool: worlds that may be perturbed (None = all; ``set_eligible``)
        self.sag_idx = torch.tensor(SAG_JOINTS, dtype=torch.long, device=self.dev)

    def reseed(self, seed):
        self.gen.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)

    def set_eligible(self, mask):
        """Only these worlds may be perturbed (e.g. the online loop's student slots); the draws stay the same."""
        self.eligible = None if mask is None else torch.as_tensor(mask, dtype=torch.bool, device=self.dev).clone()
        if self.eligible is not None:
            self.on &= self.eligible
            self.active = bool(self.on.any()) and not self.cfg.neutral

    def reset(self, idx, q):
        """Episode start of worlds ``idx`` (q [n, 6]: their measured start q): new draws; hold and lag state = q,
        integral correction = 0 (``Guard.reset``)."""
        c, n = self.cfg, len(idx)
        u = torch.rand(n, 6, generator=self.gen, dtype=torch.float64, device=self.dev)
        self.on[idx] = (u[:, 0] < c.p) if self.eligible is None else (u[:, 0] < c.p) & self.eligible[idx]
        self.d0[idx] = (c.d_lo + (u[:, 1] * (c.d_hi - c.d_lo + 1)).long()).clamp(max=c.d_hi)
        self.s[idx] = _log_uniform(c.s_lo, c.s_hi, u[:, 2])
        tau = _log_uniform(c.tau_lo, c.tau_hi, u[:, 3]) if c.tau_hi > 0 else torch.zeros_like(u[:, 3])
        self.tau[idx] = tau
        self.a[idx] = torch.where(tau > 0, (CONTROL_DT / tau.clamp_min(1e-12)).clamp(max=1.0), torch.ones_like(tau))
        self.sag[idx] = torch.deg2rad(c.sag_lo_deg + u[:, 4] * (c.sag_max_deg - c.sag_lo_deg))
        self.sigma[idx] = c.noise_lo + u[:, 5] * (c.noise_max - c.noise_lo)
        q = q.to(torch.float64)
        self.hold[idx], self.f[idx] = q, q
        self.integ[idx] = 0.0
        self.active = bool(self.on.any()) and not c.neutral

    def begin(self):
        """Decision start: this decision's delay d = clip(d0 + jitter, 0, STEPS - 1)."""
        r = torch.rand(self.B, generator=self.gen, dtype=torch.float64, device=self.dev)
        j = self.cfg.jitter
        self.d = (self.d0 + (r >= 1.0 - j / 2).long() - (r < j / 2).long()).clamp(0, STEPS - 1)

    def apply(self, j, ex, u, q, tlo, thi, qd=None):
        """Step j of the decision: ``ex`` = the action layer's targets of steps 0 .. j (list of [B, 6]), ``u`` = ex[j],
        q [B, 6] the measured q now, ``qd`` [B, 6] its speed (integral correction; None = not moving) -> the target
        sent to the world [B, 6] (unperturbed rows: ``u`` itself)."""
        B, d, c = self.B, self.d, self.cfg
        ar = torch.arange(B, device=self.dev)
        new = torch.stack(ex, 1)[ar, (j - d).clamp(min=0)]                     # u5[j - d] after the action layer
        cmd = torch.where((d > j)[:, None], self.hold, new)                     # the first d steps: the held command
        if j == STEPS - 1:
            self.hold = cmd                                                     # held until the next decision's d
        q = q.to(torch.float64)
        if c.i_time_s > 0:                # the robot guard's integral correction (hw_lib.Guard.apply, every step)
            e = cmd[:, :5] - q[:, :5]
            if qd is not None:            # a joint that follows its command needs no correction
                v_ = torch.nan_to_num(qd.to(torch.float64)[:, :5], 0.0, 0.0, 0.0)
                e = torch.where(v_ * torch.sign(e) > c.i_v_min, torch.zeros_like(e), e)
            e = torch.where(e.abs() < math.radians(c.i_e_max_deg), e, torch.zeros_like(e))    # blocked / new target
            im = math.radians(c.i_max_deg)
            self.integ = torch.where(self.on[:, None], (self.integ + e * (CONTROL_DT / c.i_time_s)).clamp(-im, im),
                                     self.integ)
            cmd = torch.cat((cmd[:, :5] + self.integ, cmd[:, 5:]), -1)
        a = self.a[:, None]
        self.f = torch.where(a >= 1.0, cmd, self.f + a * (cmd - self.f))       # first-order lag of the command
        v = self.f.clone()
        v[:, self.sag_idx] += self.sag[:, None]                                 # static error (before the gain)
        s = self.s[:, None]
        arm = torch.where(s == 1.0, v[:, :5], q[:, :5] + s * (v[:, :5] - q[:, :5]))
        arm = arm + self.sigma[:, None] * torch.randn(B, 5, generator=self.gen, dtype=torch.float64, device=self.dev)
        out = torch.minimum(torch.maximum(torch.cat((arm, v[:, 5:]), -1), tlo), thi)    # actuator range
        return torch.where(self.on[:, None], out, u)

    def params(self):
        """[B, 6] float64 per-world draws (``COLS``; sag in deg)."""
        return torch.stack((self.on.double(), self.d0.double(), self.s, self.tau, torch.rad2deg(self.sag), self.sigma),
                           -1)

    def summary(self):
        """Compact draw statistics (perturbed share; means / medians over the perturbed worlds; the mean |integral
        correction| in deg) for logs. Always the keys ``SUMMARY`` (NaN while no world is perturbed), so the log's
        csv header (taken from its first row) has every column."""
        on = self.on
        out = dict.fromkeys(self.SUMMARY, float("nan"))
        out["dyn_on"] = round(float(on.double().mean()), 3)
        if bool(on.any()):
            r = lambda x, k=3: round(float(x), k)                                     # noqa: E731
            out.update(dyn_d0=r(self.d0[on].double().mean()), dyn_d=r(self.d[on].double().mean()),
                       dyn_s_p50=r(self.s[on].median(), 4), dyn_tau_p50=r(self.tau[on].median(), 4),
                       dyn_sag_deg=r(torch.rad2deg(self.sag[on]).mean()), dyn_sigma=r(self.sigma[on].mean(), 5),
                       dyn_i_deg=r(torch.rad2deg(self.integ[on].abs()).mean()))
        return out


# ---- lenient success ---------------------------------------------------------------------------------------------
def lenient_now(F, cfg: T.TaskConfig, goal_xy, goal_yaw, lifted, half=BLOCK_HALF):
    """[B] bool: the strict success conditions with the position test replaced by 'cube centre inside the goal block's
    square of half side ``half``' and no yaw test."""
    d = F["c"][:, :2] - goal_xy
    c, s = torch.cos(goal_yaw), torch.sin(goal_yaw)
    x, y = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    return (lifted & ~F["held"] & F["on_table"] & (F["tilt"] <= cfg.flat_tilt) & (F["speed"] < cfg.rest_speed)
            & (x.abs() <= half) & (y.abs() <= half) & (F["jaw"] >= cfg.open_jaw)
            & (F["tcp"][:, 2] >= F["c"][:, 2] + T.CUBE_HALF + cfg.tcp_clear))


def attach_lenient(task: T.PickPlaceTask, half=BLOCK_HALF):
    """Wrap ``task.step`` / ``task.reset`` so that ``task.lenient`` [B] bool latches the lenient success of each episode
    (module docstring) and ``task.lenient_t`` [B] holds the step it first held (-1 never). Idempotent."""
    if getattr(task, "_lenient_attached", False):
        return task
    B, dev = task.B, task.dev
    task.lenient = torch.zeros(B, dtype=torch.bool, device=dev)
    task.lenient_t = torch.full((B,), -1, dtype=torch.long, device=dev)
    step0, reset0 = task.step, task.reset

    def step(state, macro=None, macro_goal=None):
        live = task.tr.end == 0
        out = step0(state, macro, macro_goal)
        info = out[3]
        ok = lenient_now(task.F, task.cfg, task.goal_xy, task.goal_yaw, task.tr.lifted, half) & live & ~info["fail"]
        new = ok & ~task.lenient
        task.lenient_t = torch.where(new, task.tr.t, task.lenient_t)
        task.lenient = task.lenient | ok
        return out

    def reset(idx, state, goal_xy, goal_yaw):
        reset0(idx, state, goal_xy, goal_yaw)
        task.lenient[idx] = False
        task.lenient_t[idx] = -1

    task.step, task.reset = step, reset
    task._lenient_attached = True
    return task


# ---- observation ----------------------------------------------------------------------------------------------------
def markov_obs(F, task: T.PickPlaceTask, idx=None, cfg: T.TaskConfig | None = None):
    """[B, 47] float32 (module docstring) from task features F of the current state. ``cfg``: the task config of the
    observation (default the task's; ``JointEnvConfig.hold_obs_s``)."""
    sel = (lambda x: x) if idx is None else (lambda x: x[idx])
    o = T.observation(F, sel(task.goal_xy), sel(task.goal_yaw), sel(task.tr.lifted), sel(task.tr.air_run),
                      task.cfg if cfg is None else cfg, None, None)
    return o[:, :OBS_DIM]


def hold_cap_kw(hold_cap_s=None, hold_obs_s=0.0) -> dict:
    """``JointEnvConfig`` keywords of another hold_cap rule (``TaskConfig.hold_cap_s``: held in the air for longer
    than this ends the episode; slow robot worlds need more time) and of the JT observation's held-time scale
    (``hold_obs_s``; 0 with a new cap = keep the old envs' scale, the task default 3.5 s, which every evaluation and
    online run so far gave the JT; jt_h110 itself was trained with hold_cap_s 2.5 (its env record). With the scale
    moved to a 20 s cap jt_h110 drops from 98 to 36 % lenient in nominal NRD worlds, mostly mid-air releases).
    None / the default and 0 -> {} (the old env)."""
    out = {}
    if hold_cap_s is not None and float(hold_cap_s) != T.TaskConfig.hold_cap_s:
        out["task"] = dict(hold_cap_s=float(hold_cap_s))
        out["hold_obs_s"] = float(hold_obs_s or T.TaskConfig.hold_cap_s)
    elif hold_obs_s:
        out["hold_obs_s"] = float(hold_obs_s)
    return out


# ---- env ------------------------------------------------------------------------------------------------------------
@dataclass
class JointEnvConfig:
    world: str = "nrd"
    num_envs: int = 4096
    device: str = "cuda"
    seed: int = 0
    ckpts: list = field(default_factory=list)
    members: object = "split"
    nrd_cuda_graph: bool = True
    starts: str = str(PP_DATA)
    start_split: int = 0
    pool_size: int = 65536
    gamma: float = 0.99
    horizon: int = 400
    task: dict = field(default_factory=dict)          # TaskConfig overrides
    log_only: tuple = LOG_ONLY
    lead_clamp: bool = True
    floor_guard: bool = True                          # the table floor of the action layer (``floor_guard``)
    codec: str = ""                                   # .npz with mean / std (Codec)
    auto_reset: bool = True
    stagger: bool = True
    block_half: float = BLOCK_HALF
    geom_path: str = str(Path.home() / "so101_policy" / "ckpt" / "task_geom_v2.npz")
    kin_points: str = str(Path.home() / "so101_policy" / "ckpt" / "kin_points_v1.npz")
    mj_workers: int = 0
    mj_python: object = None
    mjwarp: dict = field(default_factory=dict)
    dyn: object = None                                # RobotDynConfig (or its dict): robot dynamics stage; None = off
    pause: object = None                              # PauseConfig (or its dict): the robot's hold state; None = off
    hold_obs_s: float = 0.0                           # held-time scale (s) of the JT observation; 0 = task hold_cap_s
                                                      # (``hold_cap_kw`` keeps the old envs' 3.5 s with a new cap)

    def __post_init__(self):
        self.dyn = dyn_config(self.dyn)
        self.pause = pause_config(self.pause)

    @staticmethod
    def from_dict(d):
        d = dict(d or {})
        known = {f.name for f in fields(JointEnvConfig)}
        bad = sorted(set(d) - known - {"note", "notes"})
        if bad:
            raise ValueError(f"unknown joint env config keys {bad}")
        if "log_only" in d:
            d["log_only"] = tuple(d["log_only"])
        return JointEnvConfig(**{k: v for k, v in d.items() if k in known})

    def to_dict(self):
        return copy.deepcopy(asdict(self))


def load_codec(path, device="cpu"):
    z = np.load(Path(path).expanduser())
    return Codec(z["mean"], z["std"], device)


class JointVecEnv:
    """B task worlds with joint-target actions (module docstring)."""

    def __init__(self, cfg: JointEnvConfig | dict, codec: Codec | None = None, share=None):
        from nedm.so101_policy import vec_env as VE
        self.cfg = cfg = JointEnvConfig.from_dict(cfg) if isinstance(cfg, dict) else cfg
        self.dev = dev = torch.device("cpu" if cfg.world == "mujoco" else cfg.device)
        self.B = B = int(cfg.num_envs)
        self.num_envs, self.num_actions, self.num_obs = B, ACT_DIM, OBS_DIM
        self.action_groups = [n for _, n in ACTION_GROUPS]
        self.action_group_names = [g for g, _ in ACTION_GROUPS]
        self.steps_per_decision = STEPS
        self.kin = share.kin if share is not None else VE.kinematics(cfg.kin_points, dev)
        self.geom = share.geom if share is not None else VE.task_geometry(cfg.geom_path, dev)
        self.tcfg = T.TaskConfig(**{"horizon": int(cfg.horizon), "obs_rates": True, "log_only": tuple(cfg.log_only),
                                    "gamma": float(cfg.gamma) ** (1.0 / STEPS), **dict(cfg.task)})
        self.max_episode_length = math.ceil(self.tcfg.horizon / STEPS)
        wcfg = VE.EnvConfig(world=cfg.world, num_envs=B, device=cfg.device, ckpts=list(cfg.ckpts), members=cfg.members,
                            nrd_cuda_graph=cfg.nrd_cuda_graph, starts="", mj_workers=cfg.mj_workers,
                            mj_python=cfg.mj_python, mjwarp=dict(cfg.mjwarp))
        self.world = VE.make_world(wcfg, B, dev, self.kin, self.geom, share)
        self.task = attach_lenient(T.PickPlaceTask(B, dev, kin=self.kin, geom=self.geom, cfg=self.tcfg,
                                                   overlap_rules=cfg.world != "mujoco"), cfg.block_half)
        if codec is None:
            if not cfg.codec:
                raise ValueError("JointVecEnv needs a codec (cfg.codec or the argument)")
            codec = load_codec(cfg.codec)
        self.codec = codec.to(dev)
        self.tlo, self.thi = self.kin.tlo.to(dev), self.kin.thi.to(dev)
        self.lead = torch.tensor(LEAD, dtype=torch.float64, device=dev) if cfg.lead_clamp else None
        self.gen = torch.Generator(dev).manual_seed(int(cfg.seed))
        if share is not None:
            self.bank = share.bank
        else:
            self.bank = VE.load_starts(cfg.starts, cfg.start_split, dev) if cfg.starts and cfg.world != "mujoco" else None
        self.pool = None
        f, li = dict(dtype=torch.float64, device=dev), dict(dtype=torch.long, device=dev)
        self.ep_count = torch.zeros(B, **li)
        self.ep_ret = torch.zeros(B, **f)
        self.ep_len = torch.zeros(B, **li)
        self.task_seed = torch.zeros(B, **li)
        self.ws = None
        self.obs = None
        self.world_steps = 0
        self.decisions = 0
        self.u_exe = None                 # [B, 5, 6] targets executed in the last decision (after the action layer)
        self.table_plane = None           # ``set_table`` (evaluation only): the table top plane, None = z = 0
        self.floor_plane = None           # the action layer's floor plane (tilted); None = flat floor at floor_z
        self.floor_z = 0.0                # flat floor height (m): FLOOR + floor_z (the robot script's --table-z)
        self.q_offset = None              # [6] joint offset of the policy frame (sim emulation of a joint-map offset)
        # robot dynamics stage (real2sim; None = off) with its own generator, and the JT observation's held-time scale
        self.dyn = None if cfg.dyn is None else RobotDyn(cfg.dyn, B, dev, (int(cfg.seed) * 0x9E3779B1 + 0xD1A) &
                                                         0x7FFFFFFFFFFF)
        self.pause = None if cfg.pause is None else Pause(cfg.pause, B, dev, (int(cfg.seed) * 0x2545F491 + 0x9A5) &
                                                          0x7FFFFFFFFFFF)
        self.obs_tcfg = None
        if cfg.hold_obs_s and float(cfg.hold_obs_s) != self.tcfg.hold_cap_s:
            from dataclasses import replace
            self.obs_tcfg = replace(self.tcfg, hold_cap_s=float(cfg.hold_obs_s))

    def resolved(self) -> dict:
        return dict(env=self.cfg.to_dict(), task=asdict(self.tcfg), codec={k: v.tolist() for k, v in
                                                                            self.codec.state().items()},
                    num_obs=self.num_obs, num_actions=self.num_actions,
                    action_groups=list(zip(self.action_group_names, self.action_groups)), kind="joint_env")

    # ---- a table that is not at z = 0 (evaluation only; ``table.py``) --------------------------------------------
    def set_table(self, plane=None, floor="zero", q_offset=None):
        """The table top on the plane z = c0 + c1 x + c2 y (None: z = 0, the old env): the CPU MuJoCo table geom, the
        task starts (``reset_tasks``: the cube moved onto the plane before the warm-up, ``table.map_cube``), and the
        task rules (heights above the plane: on the table, lift, release, lowest arm point; ``task.features``).
        ``floor``: the action layer's table floor (lowest gripper / moving-jaw point >= FLOOR above it): "zero" (z = 0,
        the default and the robot script's default --table-z 0), "table" (follows the table: a flat table -> FLOOR +
        dz, as --table-z dz; a tilted plane -> heights above the plane, sim only), or a float = a flat floor at that
        z (--table-z). ``q_offset`` [6] (rad): the policy and the action layer work in a joint frame q + q_offset
        (the targets are mapped back by - q_offset; the world and the task stay true): a sim emulation of an extra
        joint-map offset on the robot. World mujoco only."""
        from nedm.so101_policy import table as TB
        if self.cfg.world != "mujoco":
            raise ValueError("set_table: CPU MuJoCo worlds only")
        plane = None if plane is None or not any(plane) else tuple(float(c) for c in plane)
        self.table_plane = plane
        self.task.table_plane = plane
        self.world.set_table(plane)
        self.floor_plane, self.floor_z = None, 0.0
        if isinstance(floor, (int, float)) and not isinstance(floor, bool):
            self.floor_z = float(floor)
        elif floor == "table":
            if TB.is_flat(plane):
                self.floor_z = 0.0 if plane is None else plane[0]
            else:
                self.floor_plane = plane
        elif floor != "zero":
            raise ValueError(f"floor {floor!r}: 'zero', 'table' or a height (m)")
        self.q_offset = None if q_offset is None or not any(q_offset) else \
            torch.as_tensor(q_offset, dtype=torch.float64, device=self.dev)
        return dict(plane=plane, floor_plane=self.floor_plane, floor_z=self.floor_z,
                    q_offset=None if self.q_offset is None else self.q_offset.tolist(), table=self.world.table)

    def table_info(self) -> dict:
        return dict(plane=self.table_plane, floor_plane=self.floor_plane, floor_z=self.floor_z,
                    q_offset=None if self.q_offset is None else self.q_offset.tolist())

    def dyn_info(self) -> dict | None:
        """The robot dynamics setting (``RobotDynConfig`` as a dict) + the per-world draws of the current episodes
        (``RobotDyn.COLS``) + the hold cap / observation scale; None = off (evaluation files: the "dyn" key)."""
        if self.dyn is None:
            return None
        return dict(cfg=asdict(self.dyn.cfg), cols=list(RobotDyn.COLS), params=self.dyn.params().cpu().tolist(),
                    hold_cap_s=self.tcfg.hold_cap_s, hold_obs_s=float(self.cfg.hold_obs_s), kp_scale=EVAL_KP_SCALE)

    # ---- tasks -------------------------------------------------------------------------------------------------
    def refresh_pool(self, n=None):
        if self.bank is None:
            raise RuntimeError("no start bank (world mujoco: use reset_tasks)")
        n = int(n or self.cfg.pool_size)
        N = self.bank["q"].shape[0]
        i = torch.randint(N, (n,), generator=self.gen, device=self.dev)
        yaw, _ = T.cube_yaw_tilt(self.bank["cube_quat"][i])
        gxy, gyaw = T.sample_goals(self.bank["cube_pos"][i, :2], yaw, self.gen)
        self.pool = dict(i=i, goal_xy=gxy, goal_yaw=gyaw)

    def _reset_rows(self, idx, start: WorldState, goal_xy, goal_yaw, seeds, mj_start=None):
        if self.cfg.world == "mujoco":
            ws = self.world.reset(idx.cpu(), mj_start, warmup=True)
        elif self.cfg.world == "mjwarp":
            ws = self.world.reset(idx, start, warmup=False)
        else:
            ws = self.world.reset(idx, start)
        self.ws = ws.to(self.dev)
        self.task.reset(idx, self.ws.index(idx), goal_xy, goal_yaw)
        self.task_seed[idx] = seeds
        self.ep_ret[idx], self.ep_len[idx] = 0.0, 0
        if self.dyn is not None:          # robot dynamics: new draws, hold / lag state = the measured start q
            self.dyn.reset(idx, self.ws.q[idx])
        if self.pause is not None:
            self.pause.reset(idx)

    def _pool_reset(self, idx):
        from nedm.so101_policy.vec_env import start_rows
        if self.pool is None:
            self.refresh_pool()
        k = torch.randint(len(self.pool["i"]), (len(idx),), generator=self.gen, device=self.dev)
        i = self.pool["i"][k]
        s = (int(self.cfg.seed) * 0x9E3779B1 + self.ep_count[idx] * self.B + idx) & 0xFFFFFFFF
        self.ep_count[idx] += 1
        self._reset_rows(idx, start_rows(self.bank, i), self.pool["goal_xy"][k], self.pool["goal_yaw"][k], s)

    def reset(self):
        idx = torch.arange(self.B, device=self.dev)
        self._pool_reset(idx)
        if self.cfg.stagger and self.cfg.auto_reset:
            k = torch.randint(self.max_episode_length, (self.B,), generator=self.gen, device=self.dev)
            self.task.tr.t.copy_(k * STEPS)
        self.obs = self._observe()
        return self.obs

    def reset_tasks(self, idx, starts: WorldState, goal_xy, goal_yaw, seeds, mj_start=None, check_tol=1e-6):
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.dev)
        st = starts.to(self.dev)
        if self.table_plane is not None and self.cfg.world == "mujoco":
            # the task's cube (start and stored settled pose) moved onto the plane; the warm-up settles it there
            from nedm.so101_policy import table as TB
            mj_start = dict(mj_start)
            mj_start["cube_pos"], mj_start["cube_quat"] = TB.map_cube(self.table_plane, mj_start["cube_pos"],
                                                                       mj_start["cube_quat"])
            cp, cq = TB.map_cube(self.table_plane, st.cube_pos.cpu().numpy(), st.cube_quat.cpu().numpy())
            st = WorldState(**{**{f: getattr(st, f) for f in st.__dataclass_fields__},
                               "cube_pos": torch.as_tensor(cp, dtype=st.cube_pos.dtype, device=self.dev),
                               "cube_quat": torch.as_tensor(cq, dtype=st.cube_quat.dtype, device=self.dev)})
            check_tol = None if check_tol is None else max(check_tol, 1e-4)   # a new settle on the plane
        if self.dyn is not None:          # fixed tasks: robot dynamics draws from the env seed and the task seeds
            sd = torch.as_tensor(seeds).reshape(-1)
            self.dyn.reseed(int(self.cfg.seed) * 1_000_003 + 7919 * (int(sd[0]) if len(sd) else 0) + len(sd) + 0xD1A)
        self._reset_rows(idx, st, torch.as_tensor(goal_xy, dtype=torch.float64, device=self.dev),
                         torch.as_tensor(goal_yaw, dtype=torch.float64, device=self.dev),
                         torch.as_tensor(seeds, dtype=torch.long, device=self.dev), mj_start)
        if self.cfg.world == "mujoco" and check_tol is not None:
            got = self.ws.index(idx)
            dq = float((got.q - st.q).abs().max())
            dc = float((got.cube_pos - st.cube_pos).abs().max())
            if dq > check_tol or dc > check_tol:
                raise RuntimeError(f"MuJoCo warm-up start differs from the stored settled state ({dq:.2e}, {dc:.2e})")
        self.obs = self._observe()
        return self.obs

    # ---- observation ---------------------------------------------------------------------------------------------
    def _observe(self, F=None, idx=None):
        ws = self.ws if idx is None else self.ws.index(idx)
        if F is None:
            F = T.features(ws, self.kin, None, self.tcfg)
        return markov_obs(F, self.task, idx, self.obs_tcfg)

    def get_observations(self):
        return self.obs

    # ---- one decision --------------------------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, actions=None, targets=None, use_targets=None, step_hook=None):
        """actions [B, 18] normalised coefficients (``Codec``) and/or targets [B, 5, 6] absolute (``use_targets`` [B]
        bool: rows that take ``targets``; default all rows if only targets are given) -> (obs [B, 47], reward [B],
        done [B], extras). step_hook(env, j): called after each of the 5 world steps (records)."""
        task, B = self.task, self.B
        live0 = task.tr.end == 0
        q0 = self.ws.q.to(torch.float64)
        if actions is not None:
            a = actions.to(device=self.dev, dtype=torch.float64).clamp(-8.0, 8.0)
            u5 = self.codec.targets(a, q0)
        else:
            u5 = None
        if targets is not None:
            tg = targets.to(device=self.dev, dtype=torch.float64)
            if u5 is None:
                u5 = tg
            else:
                m = torch.ones(B, dtype=torch.bool, device=self.dev) if use_targets is None else use_targets.to(self.dev)
                u5 = torch.where(m[:, None, None], tg, u5)
        rew = torch.zeros(B, dtype=torch.float64, device=self.dev)
        info = None
        ex = []
        fl = (FLOOR if self.floor_z == 0.0 else FLOOR + self.floor_z) if self.cfg.floor_guard else None
        dyn = self.dyn if self.dyn is not None and self.dyn.active else None      # None: the old path, bit for bit
        if dyn is not None:
            dyn.begin()
        if self.pause is not None:
            self.pause.begin(task, self.ws.q)
        for j in range(STEPS):
            if self.q_offset is None:
                u = safety(u5[:, j], self.ws.q.to(torch.float64), self.tlo, self.thi, self.lead, self.kin, fl,
                           self.floor_plane)
            else:   # the action layer in the policy's joint frame (targets u5 are in it), mapped back to the world
                qo = self.q_offset
                u = safety(u5[:, j], self.ws.q.to(torch.float64) + qo, self.tlo, self.thi, self.lead, self.kin, fl,
                           self.floor_plane) - qo
            ex.append(u)                  # u_exe: the action layer's targets (before the robot dynamics / servo scale)
            if dyn is not None:           # the robot's timing and servo (RobotDyn), perturbed worlds only
                u = dyn.apply(j, ex, u, self.ws.q, self.tlo, self.thi, self.ws.qd)
            if EVAL_KP_SCALE != 1.0:
                qn = self.ws.q.to(device=u.device, dtype=u.dtype)
                u = u.clone()
                u[:, :5] = qn[:, :5] + EVAL_KP_SCALE * (u[:, :5] - qn[:, :5])
            if self.pause is not None:    # the robot's hold state (training only): frozen arm in paused worlds
                u = self.pause.apply(u)
            ws = self.world.step(u)
            self.ws = ws if self.cfg.world != "mujoco" else ws.to(self.dev)
            r, _, _, info = task.step(self.ws, None, None)
            rew += r
            if step_hook is not None:
                step_hook(self, j)
        self.u_exe = torch.stack(ex, 1)
        self.world_steps += B * STEPS
        self.decisions += B
        end = info["end"]
        done = live0 & (end != 0)
        time_out = done & (end == T.EC["timeout"])
        self.ep_ret += rew
        self.ep_len += live0.long()
        ep = dict(end=end.clone(), success=end == T.EC["success"], success10=info["success10"].clone(),
                  lifted=info["lifted"].clone(), end_xy=info["end_xy_err"].clone(), end_yaw=info["end_yaw_err"].clone(),
                  ret=self.ep_ret.clone(), length=self.ep_len.clone(), steps=info["t"].clone(),
                  lenient=task.lenient.clone(), lenient_t=task.lenient_t.clone())
        obs = self._observe(task.F)
        final_obs = obs
        if self.cfg.auto_reset:
            idx = done.nonzero().flatten()
            if len(idx):
                obs = obs.clone()
                self._pool_reset(idx)
                obs[idx] = self._observe(idx=idx)
        self.obs = obs
        mask = torch.ones(B, ACT_DIM, dtype=torch.bool, device=self.dev)
        extras = dict(time_outs=time_out, final_obs=final_obs, action_mask=mask, episode=ep, counts={},
                      world_steps=B * STEPS)
        return obs, rew.float(), done, extras

    def close(self):
        if hasattr(self.world, "close"):
            self.world.close()


# ---- the joint teacher ----------------------------------------------------------------------------------------------
class JointTeacher:
    """A JT checkpoint (``ppo.PPO.save`` / ``save_bc``): ``act(obs)`` -> deterministic normalised action [B, 18];
    ``targets(obs, q)`` -> absolute targets [B, 5, 6]; ``std`` the PPO action std."""

    def __init__(self, path, device="cuda"):
        from nedm.so101_policy.ppo import Policy, sha256
        self.path = str(Path(path).expanduser())
        d = torch.load(self.path, map_location=device, weights_only=False)
        self.pol = Policy(**d["policy_kwargs"]).to(device)
        self.pol.load_state_dict(d["policy"])
        self.pol.eval()
        rec = d.get("env") or {}
        self.codec = Codec(np.asarray(rec["codec"]["mean"]), np.asarray(rec["codec"]["std"]), device)
        self.env_rec = rec
        self.dev = torch.device(device)
        self.std = self.pol.std.detach().float()
        self.sha256 = sha256(self.path)
        self.iteration = int(d.get("iteration", 0))

    @torch.no_grad()
    def act(self, obs):
        return self.pol.act_inference(obs.to(self.dev, torch.float32)).to(obs.device)

    @torch.no_grad()
    def targets(self, obs, q):
        a = self.act(obs).to(q.device, torch.float64)
        return self.codec.to(q.device).targets(a, q.to(torch.float64))

    def info(self):
        return dict(path=self.path, sha256=self.sha256, iteration=self.iteration)


def save_bc(path, policy, codec: Codec, env_cfg: dict, extra=None):
    """A BC-trained ``ppo.Policy`` in the ``PPO.save`` layout (so ``JointTeacher`` and PPO resume read it)."""
    rec = dict(env=env_cfg, codec={k: v.tolist() for k, v in codec.state().items()}, kind="joint_env",
               num_obs=OBS_DIM, num_actions=ACT_DIM, action_groups=list(ACTION_GROUPS))
    torch.save(dict(policy=policy.state_dict(), policy_kwargs=policy.kwargs, iteration=0, env=rec,
                    extra=extra or {}), path)
