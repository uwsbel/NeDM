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
evaluation setting from ``$NEDM_SO101_DYN``. ``RobotDynConfig.servo`` (off: {}) replaces the crude gain / lag / sag by
the servo model fitted on the robot logs (``RealServo``: delay, lag, dead band, gravity compliance, speed limit).

Goal jumps (studentlab, OFF by default: ``JointEnvConfig.goal_jump`` None): ``GoalJump`` moves the goal at the end of a
step while the cube is held and lifted near it, so the cube ends past (or before) the new goal (``GoalJumpConfig``);
the observation rows of those worlds are recomputed, so the next decision's teacher input, task rules and renders all
use the moved goal.

Wrist-roll cap (real2sim 2026-10-23, OFF by default: ``JointEnvConfig.roll_hi_deg`` None): the real wrist roll cannot go
above +101.2 sim deg (the motor's calibrated range ends at +-180 real deg, real = sim + 75.8 deg; the robot guard
clamps the roll command there), the sim target range goes to ``kin.THI[4]`` = 159.9 deg. With ``roll_hi_deg`` set
(sim deg) the action layer's range is min(THI, cap) for the wrist roll (``safety``: the executed targets ``u_exe``,
as the guard's ``prev``; the robot dynamics stage's actuator range), and the target sent to the world is capped once
more after every stage (``cap_roll``; the world's own clamps ``kin.TLO`` .. ``THI`` in the NRD / MuJoCo worlds are
wider). With a joint offset (``set_table(q_offset=...)``, evaluation only) the cap is in the policy's joint frame, as
the guard's. The teacher's observation does not change (it reads the measured q). Start-bank rows whose wrist roll is
above the cap are dropped (``drop_high_roll``); fixed tasks with such a start warn. Per episode the env records
``roll_cmd_max`` (the largest commanded wrist roll, before the action layer, policy frame) and ``roll_q_max`` (the
largest measured one) in ``extras["episode"]`` (rad).

Held-cube rest starts (real2sim round v9, 2026-10-25, OFF by default: ``JointEnvConfig.rest`` None): a share of the
automatic resets starts from a rest bank (``RestConfig``: the cube held at rest, low and far from the goal, the task's
lifted flag and held time preset), the state where the robot student stopped on 2026-10-24. With the arm pause
(``PauseConfig``) widened to the whole carry (``near_m`` up to 0.30 m, ``z_lo`` 0.002 m; a config change only) the
PPO teacher learns to start again from rest.

Kinematic arm-table end (real2sim round v9c, 2026-10-25, OFF by default: ``JointEnvConfig.arm_table`` None): the NRD has
no arm-table contact; with ``ArmTableConfig`` the env raises the world's arm_table flag (the task's world_arm_table
end) when the lowest gripper / moving-jaw collision point at q + tau x qd is below the table (before the first firm
lift), the MuJoCo contact end of a fast grasp descent.

Grasp target toward the fixed jaw (real2sim round v11, 2026-10-25, OFF by default: ``JointEnvConfig.grasp_shift``
None): ``GraspShiftConfig`` charges the offset of the cube from a grasp target moved ``d_m`` along the closing axis
toward the fixed jaw (cube centre wanted at hx = +d_m: nearer the middle of the open jaws, more fixed-jaw margin) at
the moment the fingertips reach the cube-top level and at the close start (training reward only).

Floor raise (real2sim round v12, 2026-10-26, OFF by default: ``JointEnvConfig.floor_raise`` None): the action layer's
table floor is ``FLOOR`` above z = raise(xy) = c0 + c1 x + c2 y instead of above z = 0; the table, the cube, the task
rules, the observation and the kinematic arm-table end stay at the sim table (z = 0). Why: the real table is 8-17 mm
below the sim table relative to the arm and the robot keeps its floor flat at 1.5 mm above z = 0 of the ARM frame; a
student distilled with the table frame map (``table_bias``: the student reads Phi(q), the joints of an arm whose world
lies on the table plane) sees that robot floor at z = -plane(xy) in the TRUE (physics) frame, i.e. raised by the drop
(``raise_of_table``). A flat raise (c1 = c2 = 0) is the robot's ``--table-z c0`` rule (``floor`` = FLOOR + c0); a tilted
one measures each pose's lowest gripper / moving-jaw point from raise(x_g, y_g) at THAT pose's grasp point
(``floor_guard`` lift): the frame map moves the whole gripper by the plane height at the grasp point, so this is the
robot's flat arm-frame floor exactly (the guard's pull-back line is straight in the true joints, the robot's in the
read joints: the clamped target can differ slightly where the floor acts). The PPO teacher env, the fixed-task
envs (``joint_teacher.make_env``: teacher / student evaluations, PPO dev evaluations) and the online env
(``viral.OnlineConfig.student_floor`` "observed") take it. ``set_table`` (MuJoCo evaluations) allows only its floor
"zero" with a raise.
"""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, dataclass, field, fields, replace
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
# after the robot dynamics stage (``RobotDyn``; the two compound; with the fitted servo ``RobotDynConfig.servo`` an
# error: counted twice); per world and recorded: ``RobotDynConfig.s_*``.
EVAL_KP_SCALE = float(os.environ.get("NEDM_SO101_EVAL_KP_SCALE", "1"))


def lift_at(kin, u, lift):
    """The floor raise (``JointEnvConfig.floor_raise``) of poses u [B, >= 5] (m): c0 + c1 x + c2 y at the pose's grasp
    point (``kin.fk``; the arm joints only), ``lift`` = (c0, c1, c2). float64 [B]."""
    p, _ = kin.fk(u[:, :5])
    c0, c1, c2 = (float(c) for c in lift)
    return c0 + c1 * p[:, 0] + c2 * p[:, 1]


def _low(kin, u, q, plane=None, lift=None):
    """Lowest gripper / moving-jaw collision point (m) of the arm pose u[:, :5] with the commanded and the measured jaw
    angle (the smaller). ``plane``: height above that plane instead of world z (``table.py``). ``lift`` (c0, c1, c2):
    minus the floor raise at the pose's grasp point (``lift_at``; None: the old code)."""
    kw = {} if plane is None else dict(plane=plane)
    a = kin.lowest_point(u, exact=False, **kw)
    b = kin.lowest_point(torch.cat((u[:, :5], q[:, 5:6]), -1), exact=False, **kw)
    if lift is None:
        return torch.minimum(a, b)
    return torch.minimum(a, b) - lift_at(kin, u, lift)


def floor_guard(u, q, kin, floor=FLOOR, iters=8, plane=None, lift=None):
    """Table floor of the action layer (the decoder's floor rule in joint space): a target whose lowest gripper /
    moving-jaw point is below ``floor`` AND below that of the measured pose is pulled back toward the measured q along
    the straight joint line, u' = q + lam (u - q) with the largest lam (bisection) that keeps it at or above
    min(floor, the measured pose's lowest point). Stateless (robot kinematics + measured q). u, q [B, 6] -> [B, 6].
    ``plane`` (c0, c1, c2): heights above the plane z = c0 + c1 x + c2 y instead of world z (a floor that follows a
    tilted table; a flat floor at height h is ``floor`` = FLOOR + h, as the robot's ``--table-z``). ``lift`` (c0, c1,
    c2): a tilted floor raise (``JointEnvConfig.floor_raise``): every pose's lowest point is measured from c0 + c1 x +
    c2 y at ITS OWN grasp point (vertical), the image of the robot's flat arm-frame floor under the student's table
    frame map (``table_bias``: the map moves the whole gripper by the plane height at the grasp point)."""
    lu = _low(kin, u, q, plane, lift)
    bad = lu < floor - 1e-9
    if not bool(bad.any()):
        return u
    lq = _low(kin, q, q, plane, lift)
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
        ok = _low(kin, base + mid[:, None] * d, qb, plane, lift) >= rb - 1e-9
        lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid)
    out = u.clone()
    out[idx] = base + lo[:, None] * d
    return out


def safety(u, q, tlo, thi, lead=None, kin=None, floor=None, floor_plane=None, floor_lift=None):
    """The action layer of teacher AND student: arm targets within ``lead`` of the measured q, the table floor
    (``floor_guard`` with ``kin``, if ``floor`` is given; ``floor_plane``: heights above that plane; ``floor_lift``: a
    tilted floor raise measured at each pose's grasp point, ``floor_guard`` lift), all targets within the collector's
    range. u, q [B, 6] -> [B, 6]."""
    if lead is not None:
        arm = torch.minimum(torch.maximum(u[:, :5], q[:, :5] - lead), q[:, :5] + lead)
        u = torch.cat((arm, u[:, 5:]), -1)
    u = torch.minimum(torch.maximum(u, tlo), thi)
    if kin is not None and floor is not None:
        if floor_lift is not None:
            return floor_guard(u, q, kin, floor, plane=floor_plane, lift=floor_lift)
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
    that commands relative to the measured q by the sag at every decision (jt_h110: 0.8 % lenient at 0.5 deg).
    ``i_antiwindup`` (off by default; the robot guard has it on since 2026-10-24, ``hw_lib.Guard``): the integral does
    not grow on a joint held at its actuator range (the robot: a wrist roll at its real stop wound up to +3 deg).
    ``servo`` ({} = off): the real servo fitted on the robot logs (``RealServo``, ``servo_spec``; report
    ~/so101_r2s_claude/sim_debug/scout/tr_servo.md), per joint fixed values or [lo, hi] ranges drawn per world and
    episode. It REPLACES the lag / sag / gain (they must stay neutral: s 1, tau 0, sag 0, else counted twice); stage
    order then: delay -> integral correction -> servo -> noise -> actuator range (its input is the command SENT)."""
    p: float = 1.0                      # share of perturbed worlds (per episode)
    d_lo: int = 0                       # command delay (20 ms steps), per episode U{d_lo .. d_hi}: the first d steps
    d_hi: int = 0                       # of a decision run the previous decision's last command, then u5[0 .. 4 - d]
    jitter: float = 0.0                 # per decision d + (-1, 0, +1) with probability j / 2, 1 - j, j / 2 (j = jitter)
    delay_mode: str = "hold"            # "hold": the first d steps run the previous decision's last command, then
                                        # u5[0 .. 4 - d] (the last d targets are dropped); "queue": every target runs d
                                        # steps late, none is dropped (the first d steps run the previous decision's
                                        # targets 5 - d .. 4; VIRAL's action-delay rule; the robot loop runs all 5)
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
    i_antiwindup: bool = False          #   the robot guard's anti-windup (hw_lib.Guard, 2026-10-24): no integration on
                                        #   a joint whose command is at its actuator range (tlo .. thi, the roll cap
                                        #   included) or whose corrected command would pass it; the correction is 0
                                        #   for a command at / beyond the range (either sign), else kept within
                                        #   [tlo - cmd, thi - cmd]; False = the old rule
    servo: dict = field(default_factory=dict)   # the fitted real servo (``servo_spec`` keys); {} = off

    def __post_init__(self):
        if not 0.0 <= self.p <= 1.0 or not 0.0 <= self.jitter <= 1.0:
            raise ValueError(f"robot dyn: p {self.p}, jitter {self.jitter} must be in [0, 1]")
        if self.delay_mode not in ("hold", "queue"):
            raise ValueError(f"robot dyn: delay_mode {self.delay_mode!r} must be 'hold' or 'queue'")
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
        if self.servo is None or self.servo is False:
            self.servo = {}
        if not isinstance(self.servo, dict):
            raise ValueError(f"robot dyn: servo must be a dict ({{}} = off), got {self.servo!r}")
        if self.servo:
            self.servo = copy.deepcopy(self.servo)
            servo_spec(self.servo)                                          # raises on bad keys / values
            if not (self.s_lo == self.s_hi == 1.0 and self.tau_hi == 0.0 and self.sag_lo_deg == self.sag_max_deg
                    == 0.0):
                raise ValueError("robot dyn: the servo model replaces the gain s, lag tau and sag; keep them neutral "
                                 "(s 1, tau 0, sag 0), else they are counted twice")

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

    def to_dict(self):
        """The record (``asdict``) without the servo key while it is off and without ``i_antiwindup`` while it is
        False: records of the old settings stay as they were (older code copies reject unknown keys in
        ``from_dict``)."""
        d = copy.deepcopy(asdict(self))
        if not d["servo"]:
            del d["servo"]
        if not d["i_antiwindup"]:
            del d["i_antiwindup"]
        return d

    @property
    def neutral(self):
        """True: every perturbed world equals the old path (no delay, gain 1, no lag, sag, noise or integral
        correction, no servo model); ``RobotDyn`` then skips the stage (bit for bit the old step)."""
        return (self.d_hi == 0 and self.s_lo == self.s_hi == 1.0 and self.tau_hi == 0.0 and self.sag_lo_deg ==
                self.sag_max_deg == 0.0 and self.noise_max == 0.0 and self.i_time_s == 0.0 and not self.servo)


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
    states ("descend"); after the pause the normal physics continue. Training only (``JointEnvConfig.pause``).
    Round v9 (2026-10-25): ``near_m`` up to 0.30 and ``z_lo`` 0.002 put the pauses anywhere in the carry (the robot's
    carry stops: 50-110 mm from the plate, cube bottom a few mm above the sim table); the defaults are unchanged."""
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


# ---- held-cube rest starts (real2sim round v9, 2026-10-25; off by default) -----------------------------------------
REST_STATE = ("q", "qd", "cube_pos", "cube_quat", "cube_v", "cube_w")
REST_KEYS = ("rest_goal_xy", "rest_goal_yaw", "rest_lifted", "rest_air_s")


@dataclass
class RestConfig:
    """Held-cube rest starts (real2sim round v9): on the robot of 2026-10-24 the student stopped with the cube in the
    jaw, low (TCP 19-29 mm sim z) and 50-110 mm from the plate, and the teacher labels "stay" / "open the jaw" there:
    with the lifted flag 1, a long held time and no arm speed a low held cube is a place state for it (step 0,
    ~/so101_r2s_claude/v9/restart/STEP0.md: 86-91 % far drops from such starts). With this option every automatic
    reset (``JointVecEnv._pool_reset``) is, with probability ``share``, a rest start: a row of the rest bank ``bank``
    (.npz: the start-bank keys ``REST_STATE`` at rest + ``rest_goal_xy`` [N, 2], ``rest_goal_yaw`` [N],
    ``rest_lifted`` [N], ``rest_air_s`` [N] (s held so far), optional ``rest_ctrl`` [N, 6]: the held targets, whose jaw
    column becomes the robot dynamics stage's past jaw command), with the row's goal, the task's lifted flag =
    ``rest_lifted`` and held time = ``rest_air_s`` (``JointVecEnv.set_rest``). The other resets draw their rows and goals
    as before (the env's generator makes the same draws); the rest draws use their own generator. Rows above the
    wrist-roll cap are dropped. Training only (start-bank worlds); fixed tasks with rest_* keys:
    ``joint_teacher.run_tasks``."""
    bank: str = ""
    share: float = 0.25

    def __post_init__(self):
        if not self.bank or isinstance(self.share, bool) or not 0.0 <= float(self.share) <= 1.0:
            raise ValueError(f"rest config: a bank path and a share in [0, 1], got {self}")
        self.bank, self.share = str(self.bank), float(self.share)


def rest_config(v) -> RestConfig | None:
    """None / False / {} / "" -> None (off); a dict or a JSON string -> ``RestConfig``."""
    if isinstance(v, RestConfig):
        return v
    if isinstance(v, str):
        v = json.loads(v) if v.strip() else None
    if v is None or v is False or (isinstance(v, dict) and not v):
        return None
    d = {k: x for k, x in dict(v).items() if k != "note"}
    bad = sorted(set(d) - {f.name for f in fields(RestConfig)})
    if bad:
        raise ValueError(f"unknown rest keys {bad}")
    return RestConfig(**d)


@dataclass
class FarOpenConfig:
    """A dense cost on jaw-open commands while the held cube is far from the goal (real2sim round v9, 2026-10-25;
    training reward only, off by default: ``JointEnvConfig.far_open`` None). Every 20 ms step, in live worlds whose cube
    is held (``task.F["held"]`` of the previous step), lifted (``task.tr.lifted``) and more than ``far_m`` (xy) from the
    goal: reward -= ``cost`` x max(0, jaw target - measured jaw) (rad; the action layer's executed jaw target ``u_exe``).
    Why: from the held-cube rest starts (``RestConfig``) D's teacher opens the jaw far from the goal; the closed jaw is
    7-19 policy std away and a successful restart needs it closed for several decisions in a row, so the success reward
    alone gave PPO no signal (tppo_restart_v9a_s31: rest dev set 10 -> 12 % in 140 iterations). The cost is a smooth
    function of the action at every such step. It never acts at the place (within ``far_m``) or before the first lift."""
    cost: float = 0.5
    far_m: float = 0.04

    def __post_init__(self):
        if isinstance(self.cost, bool) or not (float(self.cost) >= 0.0 and float(self.far_m) > 0.0):
            raise ValueError(f"far_open config: cost >= 0 and far_m > 0, got {self}")
        self.cost, self.far_m = float(self.cost), float(self.far_m)


def far_open_config(v) -> FarOpenConfig | None:
    """None / False / {} / "" -> None (off); a dict or a JSON string -> ``FarOpenConfig``."""
    if isinstance(v, FarOpenConfig):
        return v
    if isinstance(v, str):
        v = json.loads(v) if v.strip() else None
    if v is None or v is False or (isinstance(v, dict) and not v):
        return None
    d = {k: x for k, x in dict(v).items() if k != "note"}
    bad = sorted(set(d) - {f.name for f in fields(FarOpenConfig)})
    if bad:
        raise ValueError(f"unknown far_open keys {bad}")
    return FarOpenConfig(**d)


@dataclass
class ArmTableConfig:
    """A kinematic arm-table end for worlds without an arm-table contact (the NRD; real2sim round v9c, 2026-10-25; off by
    default: ``JointEnvConfig.arm_table`` None). After every 20 ms world step: the lowest collision point of the gripper
    and the moving jaw (``kin.lowest_point``, the deploy guard's points; heights above the table plane if one is set) at
    the pose extrapolated ``tau_s`` ahead with the measured joint rates, q + ``tau_s`` x qd, below ``margin_m`` -> the
    world's ``arm_table`` flag -> the task's ``world_arm_table`` end (the failure cost of every rule), as MuJoCo's
    contact end. ``tau_s`` 0: the pose itself. ``grasp_only``: only in worlds before their first firm lift
    (``task.tr.lifted`` False before the step): the place descent and the held-cube rest starts are not touched.
    Why: the restart teachers (s36_cont) lost 13-14 points of MuJoCo HOME roll 0 to arm-table ends at the grasp (22 of
    200 episodes, D's teacher 0): the jaw closes while the arm still descends at about 0.2 m/s, the targets sit on the
    action layer's 1.5 mm floor and the MuJoCo arm overshoots it; the moving jaw's fingertip mesh touches the table
    (all 22). The NRD has no arm-table pair (``nrd_sim``: arm_table all False), so PPO in the NRD never paid for it.
    Fit on the gate rollouts (``~/so101_r2s_claude/v9/fix/teacher/``): all 22 MuJoCo ends were before the first lift.
    margin 0, tau 0.02 s, grasp only fires on all 22 (MuJoCo states), in 87 of 200 of s36_cont's NRD HOME roll 0
    episodes and in 3 of 200 of D's teacher's (6 of D's 193 MuJoCo successes)."""
    margin_m: float = 0.0
    tau_s: float = 0.02
    grasp_only: bool = True

    def __post_init__(self):
        if not isinstance(self.grasp_only, bool):
            raise ValueError(f"arm_table config: grasp_only must be a bool, got {self.grasp_only!r}")
        for k in ("margin_m", "tau_s"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ValueError(f"arm_table config: {k} must be a finite number, got {v!r}")
        if not (0.0 <= float(self.tau_s) <= 0.1 and -0.005 <= float(self.margin_m) <= 0.02):
            raise ValueError(f"arm_table config: tau_s in [0, 0.1] s and margin_m in [-0.005, 0.02] m, got {self}")
        self.margin_m, self.tau_s = float(self.margin_m), float(self.tau_s)


def arm_table_config(v) -> ArmTableConfig | None:
    """None / False / {} / "" -> None (off); a dict or a JSON string -> ``ArmTableConfig``."""
    if isinstance(v, ArmTableConfig):
        return v
    if isinstance(v, str):
        v = json.loads(v) if v.strip() else None
    if v is None or v is False or (isinstance(v, dict) and not v):
        return None
    d = {k: x for k, x in dict(v).items() if k != "note"}
    bad = sorted(set(d) - {f.name for f in fields(ArmTableConfig)})
    if bad:
        raise ValueError(f"unknown arm_table keys {bad}")
    return ArmTableConfig(**d)


def arm_table_rule(q, qd, kin, at: ArmTableConfig, plane=None):
    """[B] bool: ``ArmTableConfig``'s kinematic arm-table end at the measured state (q, qd [B, 6])."""
    q = q.to(torch.float64)
    qx = q if at.tau_s == 0.0 else q + at.tau_s * qd.to(torch.float64)
    low = kin.lowest_point(qx) if plane is None else kin.lowest_point(qx, plane=plane)
    return low < at.margin_m


@dataclass
class GraspShiftConfig:
    """A grasp target moved ``d_m`` along the jaw axis toward the FIXED jaw (real2sim round v11, 2026-10-25; training
    reward only, off by default: ``JointEnvConfig.grasp_shift`` None). The cube then sits nearer the middle of the open
    jaws: wanted offset of the cube centre from the grasp point (TCP) along the gripper's closing axis (gripper x,
    fixed jaw -> moving jaw; ``task.features`` ``hx``: horizontal, in the gripper yaw frame) = +``d_m``.

    The cost acts at two events of a live world before its first firm lift, with the cube free (not held, on the
    table) and the TCP within ``r_m`` (xy) of the cube centre: (1) the TCP goes below ``h_m`` above the cube centre
    (a downward crossing between two 20 ms steps; with a top-down gripper the fingertips are then at the cube-top level:
    the moment the fixed jaw clears the cube or lands on it), (2) the measured jaw goes below ``jaw_rad`` (the close
    start; the moving jaw has not reached the cube yet) with the TCP below ``h_m``. At each event: reward -= ``cost`` x
    min(max(0, |hx - d_m| - ``tol_m``), ``cap_m``) (m). Event (1) does not depend on the jaw (a policy cannot avoid it by
    a half-closed descent); event (2) measures the cube position in the open jaws at the close.

    Why: on the robot the grasp fails when the fixed-jaw tip ends > 3 mm over the cube (0 / 38 held); a 30 mm cube on
    the grasp point clears the fixed jaw by only 2.8 mm (arm model), the open moving jaw by 33 mm. In sim, a cube 3-5 mm
    toward the moving jaw at the descent start gave +31 to +41 of 224 grasps for all 4 robot students
    (~/so101_r2s_claude/grasp_why/). The teacher reward has no placement term: the reach potential (``task.potential``,
    TCP xy to the cube xy) is cashed in at the first firm lift, so it does not set the grasp point, and the NRD lets
    the fixed jaw pass through the cube top (the task's jaw_overlap rule ends it; ~/so101_r2s_claude/v11/teacher/).
    Check before use (v11 step 2): with the cube 0-12 mm toward the moving jaw, a scripted close pushes it onto the
    fixed jaw and holds it in MuJoCo and in the NRD (144 / 144 lifted in both; NRD in-hand x error about -2 mm)."""
    d_m: float = 0.004
    cost: float = 50.0
    tol_m: float = 0.001
    cap_m: float = 0.015
    h_m: float = 0.028
    r_m: float = 0.03
    jaw_rad: float = 0.45

    def __post_init__(self):
        for k in ("d_m", "cost", "tol_m", "cap_m", "h_m", "r_m", "jaw_rad"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ValueError(f"grasp_shift config: {k} must be a finite number, got {v!r}")
            setattr(self, k, float(v))
        if not (0.0 <= self.d_m <= 0.015 and self.cost >= 0.0 and 0.0 <= self.tol_m <= 0.005
                and 0.0 < self.cap_m <= 0.03 and 0.005 <= self.h_m <= 0.06 and 0.005 <= self.r_m <= 0.06
                and 0.2 <= self.jaw_rad <= 0.7):
            raise ValueError(f"grasp_shift config: d_m in [0, 0.015] m, cost >= 0, tol_m in [0, 0.005] m, cap_m in "
                             f"(0, 0.03] m, h_m and r_m in [0.005, 0.06] m, jaw_rad in [0.2, 0.7] rad, got {self}")


def grasp_shift_config(v) -> GraspShiftConfig | None:
    """None / False / {} / "" -> None (off); a dict or a JSON string -> ``GraspShiftConfig``."""
    if isinstance(v, GraspShiftConfig):
        return v
    if isinstance(v, str):
        v = json.loads(v) if v.strip() else None
    if v is None or v is False or (isinstance(v, dict) and not v):
        return None
    d = {k: x for k, x in dict(v).items() if k != "note"}
    bad = sorted(set(d) - {f.name for f in fields(GraspShiftConfig)})
    if bad:
        raise ValueError(f"unknown grasp_shift keys {bad}")
    return GraspShiftConfig(**d)


def grasp_height(F):
    """[B] TCP height above the cube centre (m) of task features F."""
    return F["tcp"][:, 2] - F["c"][:, 2]


def grasp_shift_events(F, h_prev, jaw_prev, lifted, gs: GraspShiftConfig):
    """``GraspShiftConfig``'s events at the state of features F (after a 20 ms step), with the previous step's TCP
    height above the cube centre ``h_prev`` [B] and measured jaw ``jaw_prev`` [B] and the task's lifted flag before
    the step [B] -> (descent event [B] bool, close event [B] bool, hx - d_m [B] m, TCP height now [B] m)."""
    h = grasp_height(F)
    free = ~F["held"] & F["on_table"] & ~lifted & (F["dxy"] < gs.r_m)
    low = h < gs.h_m
    e1 = free & low & (h_prev >= gs.h_m)
    e2 = free & low & (jaw_prev >= gs.jaw_rad) & (F["jaw"] < gs.jaw_rad)
    return e1, e2, F["hx"] - gs.d_m, h


def grasp_shift_cost(err, gs: GraspShiftConfig):
    """[B] cost of one event at the offset error err = hx - d_m [B] (m)."""
    return gs.cost * (err.abs() - gs.tol_m).clamp(min=0.0, max=gs.cap_m)


GS_STATS = ("n_descent", "hx_descent", "abs_err_descent", "n_close", "hx_close", "abs_err_close", "cost")


def load_rest_bank(path, device) -> dict:
    """Rest bank (``RestConfig``) -> dict of [N, ...] tensors: ``REST_STATE`` (float64), goal_xy, goal_yaw, lifted
    (bool), air (held time in 20 ms steps, long), jaw_cmd (float64; only with ``rest_ctrl``). Refuses the sealed test
    and a task file (``goal_xy`` key: development / fresh sets are not training starts)."""
    p = Path(str(path)).expanduser()
    if "test_sealed" in str(p):
        raise ValueError("the sealed test data must not be used for starts")
    z = np.load(p)
    if "goal_xy" in z.files:
        raise ValueError(f"{p}: a task file (goal_xy), not a rest bank")
    miss = [k for k in REST_STATE + REST_KEYS if k not in z.files]
    if miss:
        raise ValueError(f"{p}: rest bank without {miss}")
    f = lambda k: torch.as_tensor(np.asarray(z[k], np.float64), device=device)               # noqa: E731
    b = {k: f(k) for k in REST_STATE}
    b.update(goal_xy=f("rest_goal_xy"), goal_yaw=f("rest_goal_yaw"),
             lifted=torch.as_tensor(np.asarray(z["rest_lifted"], bool), device=device),
             air=torch.round(f("rest_air_s") / CONTROL_DT).long())
    if "rest_ctrl" in z.files:
        b["jaw_cmd"] = f("rest_ctrl")[:, 5]
    return b


# ---- goal jumps (studentlab trials A2-A4, 2026-10-22; off by default) ---------------------------------------------
@dataclass
class GoalJumpConfig:
    """Goal-jump coverage: states with the cube held PAST (or before) the goal, which the carry never makes on its own
    (the real robot hovers ~19 mm past the mat centre). In a share ``p`` of the episodes, at most ``max_n`` times per
    episode, at a decision where the cube is held, lifted (``task.tr.lifted``), its bottom ``z_lo``-``z_hi`` above the
    table and its centre within ``near_m`` (xy) of the goal, with probability ``rate`` per such decision and at least
    ``gap_s`` after the previous jump, the goal moves so that the cube is PAST it along the carry direction by
    d ~ U(d_lo, d_hi) (share 1 - ``before_p``) or BEFORE it by d (share ``before_p``), plus a lateral offset
    U(-lat_m, lat_m). Carry direction = unit vector from the cube's start xy to the goal at the episode start (radial
    from the pan axis if they are within 1 mm). The goal yaw stays. A new goal outside the workspace annulus
    (``task.ANNULUS_R`` / ``ANNULUS_DEG`` widened by ``margin_m``) is tried with the lateral offset mirrored, else that
    world does not jump at this decision. ``JointVecEnv.step`` applies the jump at the END of a step (after the task
    rules and the outcome record of that step), so from the next decision on the teacher observation, the task rules
    (success, errors), the renderer's goal and the student's aux targets all read the moved goal (``JointVecEnv``
    recomputes the observation rows of the worlds that jumped). Own generator: the env's other draws never change."""
    p: float = 0.4
    max_n: int = 2
    rate: float = 0.35
    gap_s: float = 1.0
    near_m: float = 0.03
    z_lo: float = 0.005
    z_hi: float = 0.045
    d_lo: float = 0.008
    d_hi: float = 0.028
    before_p: float = 0.2
    lat_m: float = 0.008
    margin_m: float = 0.01

    def __post_init__(self):
        self.max_n = int(self.max_n)
        if not (0.0 <= self.p <= 1.0 and 0.0 <= self.rate <= 1.0 and 0.0 <= self.before_p <= 1.0 and self.max_n >= 1
                and 0.0 <= self.d_lo <= self.d_hi and self.lat_m >= 0.0 and self.near_m > 0.0
                and self.z_lo <= self.z_hi and self.gap_s >= 0.0 and self.margin_m >= 0.0):
            raise ValueError(f"goal jump config out of range: {self}")

    @staticmethod
    def from_spec(spec: str, **base) -> "GoalJumpConfig":
        """Short text form (evaluation CLI), lengths in mm: "p=1,d=10:25,lat=8[,n=1,before=0,rate=0.35,near=30,
        z=5:45,gap=1.0,margin=10]". ``base``: defaults under the text (e.g. max_n=1, before_p=0)."""
        kw = dict(base)
        for part in [s.strip() for s in str(spec).split(",") if s.strip()]:
            k, v = part.split("=", 1)
            k = k.strip()
            lo_hi = [float(x) for x in v.split(":")]
            if k in ("d", "z"):
                if len(lo_hi) != 2:
                    raise ValueError(f"goal jump spec {k}={v}: lo:hi (mm)")
                kw[f"{k}_lo"], kw[f"{k}_hi"] = lo_hi[0] * 1e-3, lo_hi[1] * 1e-3
            elif k in ("lat", "near", "margin"):
                kw[f"{k}_m"] = lo_hi[0] * 1e-3
            elif k == "n":
                kw["max_n"] = int(lo_hi[0])
            elif k == "before":
                kw["before_p"] = lo_hi[0]
            elif k in ("p", "rate", "gap"):
                kw[{"gap": "gap_s"}.get(k, k)] = lo_hi[0]
            else:
                raise ValueError(f"goal jump spec: unknown key {k!r} (p, d, lat, n, before, rate, near, z, gap, margin)")
        return GoalJumpConfig(**kw)


def goal_jump_config(v) -> GoalJumpConfig | None:
    """None / False / {} / "" -> None (off); a dict / JSON string -> ``GoalJumpConfig``."""
    if isinstance(v, GoalJumpConfig):
        return v
    if isinstance(v, str):
        v = json.loads(v) if v.strip() else None
    if v is None or v is False or (isinstance(v, dict) and not v):
        return None
    d = {k: x for k, x in dict(v).items() if k not in ("note", "notes")}
    bad = sorted(set(d) - {f.name for f in fields(GoalJumpConfig)})
    if bad:
        raise ValueError(f"unknown goal jump keys {bad}")
    return GoalJumpConfig(**d)


# ---- the wrist-roll cap (real2sim; off by default) -----------------------------------------------------------------
ROLL = 4                                   # wrist_roll joint index
ROLL_NEAR_DEG = 0.5                        # summaries: a measured roll this close to the cap counts as "at the cap"


def roll_hi_config(v) -> float | None:
    """None -> None (off); a number (sim deg) -> float. Raises ValueError if not finite or at / below the target
    range's lower end (``kin.TLO[4]``)."""
    import numbers
    from nedm.so101_policy.kin import TLO
    if v is None:
        return None
    if not isinstance(v, numbers.Real) or isinstance(v, bool) or not math.isfinite(float(v)):
        raise ValueError(f"roll_hi_deg {v!r}: a finite number (sim deg) or None")
    if math.radians(float(v)) <= TLO[ROLL]:
        raise ValueError(f"roll_hi_deg {v}: must be above the wrist roll's lower target limit "
                         f"{math.degrees(TLO[ROLL]):.1f} deg")
    return float(v)


FLOOR_RAISE_MAX = (0.03, 0.2, 0.2)          # floor_raise limits: |c0| m, |c1|, |c2| m per m (a units guard)


def floor_raise_config(v) -> tuple | None:
    """``JointEnvConfig.floor_raise`` (module docstring "Floor raise"): None / False / 0 / [] / "" / all zeros -> None
    (off); a number (m) or the string "h" -> the flat raise (h, 0, 0); a sequence (c0, c1, c2) (m, m per m) or the
    string "c0,c1,c2" -> a tuple of floats. Raises ValueError for another length, a bool / non-finite value, or values
    beyond ``FLOOR_RAISE_MAX`` (mm given as m)."""
    import numbers
    if isinstance(v, str):
        v = v.strip()
        v = [float(x) for x in v.split(",")] if v else None
        v = v[0] if v is not None and len(v) == 1 else v
    if v is None or v is False:
        return None
    if isinstance(v, numbers.Real) and not isinstance(v, bool):
        v = (v, 0.0, 0.0)
    try:
        c = tuple(v)
    except TypeError:
        raise ValueError(f"floor_raise {v!r}: a height (m) or (c0, c1, c2)") from None
    if len(c) == 0:
        return None
    if len(c) != 3 or any(isinstance(x, bool) or not isinstance(x, numbers.Real) or not math.isfinite(float(x))
                          for x in c):
        raise ValueError(f"floor_raise {v!r}: a height (m) or 3 finite numbers (c0 m, c1, c2 m per m)")
    c = tuple(float(x) + 0.0 for x in c)                 # (+ 0.0: no -0.0 in the record)
    if any(abs(x) > m for x, m in zip(c, FLOOR_RAISE_MAX)):
        raise ValueError(f"floor_raise {list(c)}: beyond |c0| {FLOOR_RAISE_MAX[0]} m, |c1|, |c2| {FLOOR_RAISE_MAX[1]} "
                         "(units: m, m per m)")
    return None if not any(c) else c


def raise_of_table(plane) -> tuple | None:
    """The floor raise that puts the action-layer floor flat in the frame of a world whose table lies on ``plane``
    (c0, c1, c2; ``table.py``, c0 < 0 = lower than the sim table): -plane (``table_bias`` frame map: the reading is
    the true pose moved by plane(xy) at the grasp point, so the robot's flat floor z = 0 of the reading is z =
    -plane(xy) in the true frame). None / zeros -> None."""
    if plane is None:
        return None
    return floor_raise_config([-float(c) for c in plane])


def drop_high_roll(bank: dict, hi: float):
    """Start bank (dict of [N, ...] tensors with "q") -> (the rows whose wrist roll q[:, 4] <= hi, rows dropped). No row
    above: the same dict (its draws do not change)."""
    keep = ~(bank["q"][:, ROLL] > hi)
    n = int((~keep).sum())
    if n == 0:
        return bank, 0
    idx = keep.nonzero().flatten()
    return {k: v[idx] for k, v in bank.items()}, n


def in_workspace(xy, margin=0.0):
    """[n, 2] goal xy -> [n] bool: on the workspace annulus of ``task.sample_annulus`` widened by ``margin`` (m)."""
    from nedm.so101_policy.kin import PAN_AXIS_XY
    dx, dy = xy[:, 0] - PAN_AXIS_XY[0], xy[:, 1] - PAN_AXIS_XY[1]
    r = torch.sqrt(dx ** 2 + dy ** 2)
    a = torch.atan2(dy, dx).abs()
    amax = math.radians(max(abs(T.ANNULUS_DEG[0]), abs(T.ANNULUS_DEG[1]))) + margin / r.clamp_min(1e-3)
    return (r >= T.ANNULUS_R[0] - margin) & (r <= T.ANNULUS_R[1] + margin) & (a <= amax)


class GoalJump:
    """Per-world state of ``GoalJumpConfig`` (``JointVecEnv``); own generator. ``just`` [B]: jumped at the end of the
    last step; ``n`` [B]: jumps in the current episode; ``goal0`` [B, 2]: the episode's first goal; ``d_last`` /
    ``lat_last`` [B] (m): the last jump's offset of the cube from the new goal along the carry (+ = past) / across;
    ``z_last`` [B] (m): the cube bottom at the last jump."""

    def __init__(self, cfg: GoalJumpConfig, B, device, seed):
        self.cfg, self.B, self.dev = cfg, int(B), torch.device(device)
        self.gen = torch.Generator(self.dev).manual_seed(int(seed))
        f = dict(dtype=torch.float64, device=self.dev)
        self.on = torch.zeros(B, dtype=torch.bool, device=self.dev)
        self.n = torch.zeros(B, dtype=torch.long, device=self.dev)
        self.last_t = torch.full((B,), -10 ** 9, dtype=torch.long, device=self.dev)
        self.just = torch.zeros(B, dtype=torch.bool, device=self.dev)
        self.dir = torch.zeros(B, 2, **f)
        self.goal0 = torch.zeros(B, 2, **f)
        self.d_last = torch.full((B,), math.nan, **f)
        self.lat_last = torch.full((B,), math.nan, **f)
        self.z_last = torch.full((B,), math.nan, **f)    # cube bottom (m) at the last jump
        self.jumps = 0                                  # jumps so far (log)

    def reseed(self, seed):
        self.gen.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)

    def reset(self, idx, cube_xy, goal_xy):
        """Episode start of worlds ``idx``: cube_xy, goal_xy [n, 2] of the start."""
        from nedm.so101_policy.kin import PAN_AXIS_XY
        u = torch.rand(len(idx), generator=self.gen, dtype=torch.float64, device=self.dev)
        self.on[idx] = u < self.cfg.p
        self.n[idx] = 0
        self.last_t[idx] = -10 ** 9
        self.just[idx] = False
        g = torch.as_tensor(goal_xy, dtype=torch.float64, device=self.dev).reshape(-1, 2)
        c = torch.as_tensor(cube_xy, dtype=torch.float64, device=self.dev).reshape(-1, 2)
        v = g - c
        nv = v.norm(dim=-1, keepdim=True)
        rad = g - torch.tensor(PAN_AXIS_XY, dtype=torch.float64, device=self.dev)
        rad = rad / rad.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        self.dir[idx] = torch.where(nv > 1e-3, v / nv.clamp_min(1e-9), rad)
        self.goal0[idx] = g
        self.d_last[idx], self.lat_last[idx], self.z_last[idx] = math.nan, math.nan, math.nan

    def apply(self, task, live):
        """End of a step (``task.F`` = this step's features): move the goal of the eligible worlds (``task.set_goal``)
        -> long [k] indices of the worlds that jumped (empty: none). Draws B x 4 uniforms at every call."""
        c = self.cfg
        F, tr = task.F, task.tr
        self.just = torch.zeros_like(self.just)
        r = torch.rand(self.B, 4, generator=self.gen, dtype=torch.float64, device=self.dev)
        cxy = F["c"][:, :2].to(torch.float64)
        dxy = (cxy - task.goal_xy).norm(dim=-1)
        zb = F["bottom"]
        gap_ok = (tr.t - self.last_t).to(torch.float64) * CONTROL_DT >= c.gap_s - 1e-9
        go = (self.on & live & (self.n < c.max_n) & F["held"] & tr.lifted & (dxy < c.near_m) & (zb >= c.z_lo)
              & (zb <= c.z_hi) & gap_ok & (r[:, 0] < c.rate))
        if not bool(go.any()):
            return go.nonzero().flatten()
        d = c.d_lo + r[:, 1] * (c.d_hi - c.d_lo)
        sgn = torch.where(r[:, 2] < c.before_p, -1.0, 1.0).to(d)              # +1: the cube is PAST the new goal
        lat = (2.0 * r[:, 3] - 1.0) * c.lat_m
        u = self.dir
        perp = torch.stack((-u[:, 1], u[:, 0]), -1)
        base = cxy - (sgn * d)[:, None] * u                                     # cube - goal = sgn d u + lat perp
        g1, g2 = base - lat[:, None] * perp, base + lat[:, None] * perp
        ok1, ok2 = in_workspace(g1, c.margin_m), in_workspace(g2, c.margin_m)
        go = go & (ok1 | ok2)
        idx = go.nonzero().flatten()
        if len(idx):
            new = torch.where(ok1[:, None], g1, g2)[idx]
            task.set_goal(idx, new, task.goal_yaw[idx])
            self.n[idx] += 1
            self.last_t[idx] = tr.t[idx]
            self.just[idx] = True
            self.d_last[idx] = (sgn * d)[idx]
            self.lat_last[idx] = torch.where(ok1, lat, -lat)[idx]
            self.z_last[idx] = zb.to(torch.float64)[idx]
            self.jumps += len(idx)
        return idx

    def summary(self):
        return dict(gj_on=round(float(self.on.double().mean()), 3), gj_jumps=int(self.jumps))


def dyn_from_env() -> RobotDynConfig | None:
    """The evaluation setting of ``$NEDM_SO101_DYN`` (JSON ``RobotDynConfig``, e.g. '{"d": 2, "s": 0.1, "tau": 0.1,
    "sag_deg": 1.7}'); unset / empty = off."""
    return dyn_config(os.environ.get(DYN_ENV, "").strip())


def _log_uniform(lo, hi, u):
    if lo == hi:
        return torch.full_like(u, float(lo))
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


# ---- the fitted real servo (``RobotDynConfig.servo``) ----------------------------------------------------------------
# Per joint, continuous time, exact per sub-step for the piecewise-constant command c = the command SENT (after the
# stage's delay d and the integral correction); fitted on 12 robot + 15 teleop episodes (report
# ~/so101_r2s_claude/sim_debug/scout/tr_servo.md; reference /tmp/claude_servo_fit/servo_emu.py plus the load-dependent
# dead band of its sim.py):
#     r = c(t - delay) + o,   o = bias - compliance (G(q) + m_cube Gc(q) held)         (where the servo settles)
#     e = r - y,   db = deadband + deadband_per_Nm |G(q)|
#     a still joint starts only if |e| > db + breakaway_extra, a moving joint stops once |e| <= db (+ SERVO_EPS)
#     while moving: dy/dt = clip((e - db sign e) / tau, +-vmax)
# G, Gc: closed-form gravity torque (N m) of shoulder_lift / elbow / wrist_flex and the held-cube torque per kg at the
# grasp point (least squares on the Menagerie MuJoCo model, residual 0.004 N m) at the MEASURED q; ``held`` = the
# task's held flag. The output y (the emulated real joint position at the end of the step) is the target sent to the
# world: the sim servo is stiff (MuJoCo kp 998: lag 3 ms, sag < 0.03 deg), so the world follows y within a few ms.
# y is re-anchored per joint to the world q where they differ by more than ``reanchor_deg`` (contact, blocked jaw).
# Held cube (``held_jaw``, default on; off = the reference): while the task's held flag is set the cube blocks the
# jaw, so the emulated jaw cannot close past the world's jaw angle (y = max(y, q)), and while the servo squeezes
# (its settle point below q - the dead band) the world's jaw target is that settle point (the delayed command), not y:
# with the reference's y the jaw target is only 2.5 / 4.9 deg (alternating) past the held jaw instead of the commanded
# ~9.6 deg; MuJoCo's grip does not change (force-saturated beyond 0.2 deg), the NRD world's does (jt_h110 on dev100:
# 99 % lenient nominal, 97 % at a 4.9 deg gap, 35 % alternating, 0 % at 2.5 deg; the fitted jaw servo alone: 10 %
# with the reference rule, 97 % with this one, MuJoCo 78 / 81 %; ~/so101_r2s_claude/sim_debug/servo/diag), and on
# release the jaw opens from the held angle (the reference first runs back up from up to 4.9 deg: 0-1 step later).
SERVO_KEYS = ("delay_ms", "tau_ms", "deadband_deg", "deadband_per_Nm_deg", "breakaway_extra_deg",
              "compliance_rad_per_Nm", "cube_mass_kg", "bias_deg", "vmax_rad_s")   # = servo_fit_params.json "proposed"
SERVO_NEED = ("delay_ms", "tau_ms", "deadband_deg")              # required keys; the others default to:
SERVO_DEFAULT = dict(deadband_per_Nm_deg=0.0, breakaway_extra_deg=0.0, compliance_rad_per_Nm=0.0, cube_mass_kg=0.0,
                     bias_deg=0.0, vmax_rad_s=1e3)               # 1e3 rad/s = no speed limit
SERVO_LOG = ("tau_ms", "vmax_rad_s")                             # [lo, hi] log-uniform (the fit's randomisation)
SERVO_OPTS = dict(n_sub=10, reanchor_deg=3.0, held_jaw=True)     # sub-steps per 20 ms step, re-anchor threshold,
                                                                 # the held-cube jaw rule (False = the reference)
SERVO_SI = dict(delay_ms=1e-3, tau_ms=1e-3, deadband_deg=math.pi / 180, deadband_per_Nm_deg=math.pi / 180,
                breakaway_extra_deg=math.pi / 180, compliance_rad_per_Nm=1.0, cube_mass_kg=1.0,
                bias_deg=math.pi / 180, vmax_rad_s=1.0)          # config units -> s, rad, rad / N m, kg, rad / s
SERVO_MAX_DELAY_S = 0.25
SERVO_EPS = 1.7e-4                                               # rad: a moving joint stops this far past the edge
SERVO_SUMMARY = ("dyn_servo_still", "dyn_servo_db_deg")          # ``RobotDyn.summary`` keys with the servo on
# gravity torque (N m) of joints 1..3 and the held-cube torque (N m per kg): w . [c1 s1 c12 s12 c123 s123]
SERVO_W_BIAS = ((-0.08956, -0.41704, -0.35361, -0.01302, -0.12852, -0.00093),
                (0.0, 0.0, -0.35361, -0.01302, -0.12852, -0.00093),
                (0.0, 0.0, 0.0, 0.0, -0.12852, -0.00093))
SERVO_W_CUBE = ((-0.27491, -1.10161, -1.32276, -0.05156, -1.50263, -0.01843),
                (0.0, 0.0, -1.32276, -0.05156, -1.50263, -0.01843),
                (0.0, 0.0, 0.0, 0.0, -1.50263, -0.01843))


def servo_spec(d):
    """``RobotDynConfig.servo`` dict -> (lo, hi) float64 [len(SERVO_KEYS), 6] in SI units (``SERVO_SI``) and the options
    (``SERVO_OPTS``). Per key: a number (every joint) or 6 entries (shoulder_pan .. gripper), each a number (fixed) or
    [lo, hi] (drawn per world and episode: log-uniform for ``SERVO_LOG``, else uniform); cube_mass_kg: a number or
    [lo, hi] (one draw per world). Missing keys: ``SERVO_DEFAULT`` (``SERVO_NEED`` must be given); "note" ignored.
    Raises ValueError on unknown keys, bad shapes or ranges."""
    import numbers
    d = {k: v for k, v in dict(d).items() if k != "note"}
    bad = sorted(set(d) - set(SERVO_KEYS) - set(SERVO_OPTS))
    if bad:
        raise ValueError(f"robot dyn servo: unknown keys {bad} (known: {list(SERVO_KEYS) + list(SERVO_OPTS)})")
    miss = [k for k in SERVO_NEED if k not in d]
    if miss:
        raise ValueError(f"robot dyn servo: missing keys {miss}")
    num = lambda x: isinstance(x, numbers.Real) and not isinstance(x, bool)          # noqa: E731
    seq = lambda x, n: isinstance(x, (list, tuple)) and len(x) == n                  # noqa: E731
    lo, hi = np.zeros((len(SERVO_KEYS), 6)), np.zeros((len(SERVO_KEYS), 6))
    for i, k in enumerate(SERVO_KEYS):
        v = d.get(k, SERVO_DEFAULT.get(k))
        if num(v) or (k == "cube_mass_kg" and seq(v, 2)):
            ent = [v] * 6
        elif seq(v, 6) and k != "cube_mass_kg":
            ent = list(v)
        else:
            raise ValueError(f"robot dyn servo: {k} = {v!r}: a number, " +
                             ("or [lo, hi]" if k == "cube_mass_kg" else "or 6 entries (numbers or [lo, hi])"))
        for j, e in enumerate(ent):
            if num(e):
                a = b = float(e)
            elif seq(e, 2) and num(e[0]) and num(e[1]):
                a, b = float(e[0]), float(e[1])
            else:
                raise ValueError(f"robot dyn servo: {k}[{j}] = {e!r}: a number or [lo, hi]")
            if not (math.isfinite(a) and math.isfinite(b) and a <= b):
                raise ValueError(f"robot dyn servo: {k}[{j}] = {e!r}: finite, lo <= hi")
            lo[i, j], hi[i, j] = a * SERVO_SI[k], b * SERVO_SI[k]
    K = {k: i for i, k in enumerate(SERVO_KEYS)}
    if lo[K["delay_ms"]].min() < 0.0 or hi[K["delay_ms"]].max() > SERVO_MAX_DELAY_S:
        raise ValueError(f"robot dyn servo: delay_ms must be within 0 .. {SERVO_MAX_DELAY_S * 1e3:g}")
    if min(lo[K["tau_ms"]].min(), lo[K["vmax_rad_s"]].min()) <= 0.0:
        raise ValueError("robot dyn servo: tau_ms and vmax_rad_s must be > 0")
    for k in ("deadband_deg", "deadband_per_Nm_deg", "breakaway_extra_deg", "compliance_rad_per_Nm", "cube_mass_kg"):
        if lo[K[k]].min() < 0.0:
            raise ValueError(f"robot dyn servo: {k} must be >= 0")
    opts = dict(SERVO_OPTS, **{k: d[k] for k in SERVO_OPTS if k in d})
    if not (num(opts["n_sub"]) and int(opts["n_sub"]) == opts["n_sub"] and int(opts["n_sub"]) >= 1
            and num(opts["reanchor_deg"]) and opts["reanchor_deg"] > 0 and isinstance(opts["held_jaw"], bool)):
        raise ValueError(f"robot dyn servo: n_sub (int >= 1) / reanchor_deg (> 0) / held_jaw (bool): {opts}")
    return lo, hi, dict(n_sub=int(opts["n_sub"]), reanchor_deg=float(opts["reanchor_deg"]),
                        held_jaw=bool(opts["held_jaw"]))


def servo_gravity(q, w):
    """q [B, 6] (sim rad) -> (G, Gc) [B, 6] float64: the gravity torque (N m) and the held-cube torque per kg (N m / kg)
    of shoulder_lift / elbow / wrist_flex (zero for pan, wrist_roll, jaw); w = (``SERVO_W_BIAS``, ``SERVO_W_CUBE``) as
    [3, 6] float64 tensors on q's device."""
    a1 = q[:, 1]
    a2 = a1 + q[:, 2]
    a3 = a2 + q[:, 3]
    X = torch.stack((a1.cos(), a1.sin(), a2.cos(), a2.sin(), a3.cos(), a3.sin()), -1).to(torch.float64)
    G = torch.zeros(q.shape[0], 6, dtype=torch.float64, device=q.device)
    Gc = torch.zeros_like(G)
    G[:, 1:4] = X @ w[0].T
    Gc[:, 1:4] = X @ w[1].T
    return G, Gc


def _servo_seed(seed):
    return (int(seed) * 0x5851F42D + 0x5E7) & 0x7FFFFFFFFFFF


class RealServo:
    """Per-world state of the fitted real servo (``RobotDynConfig.servo``; the model above) for B worlds x 6 joints,
    float64, vectorised over worlds and joints, own generator (``RobotDyn`` seeds it from its own seed, so the stage's
    other draws do not change with the servo on). State: the emulated position y, the moving / still flag, the last
    ``nh`` sent commands. At every reset one uniform per world, key and joint (``SERVO_KEYS`` x 6, fixed keys too), so
    paired settings share their draws. ``step``: one 20 ms step in ``n_sub`` sub-steps, each in closed form (lag +
    speed limit) for the command acting in that sub-step (the command of m steps ago, m = ceil((delay / h - i) / n))."""

    def __init__(self, spec: dict, B, device, seed):
        lo, hi, opts = servo_spec(spec)
        self.B, self.dev = int(B), torch.device(device)
        f = dict(dtype=torch.float64, device=self.dev)
        self.lo, self.hi = torch.as_tensor(lo, **f), torch.as_tensor(hi, **f)
        self.log = torch.tensor([k in SERVO_LOG for k in SERVO_KEYS], dtype=torch.bool, device=self.dev)[:, None]
        self.n = int(opts["n_sub"])
        self.h = CONTROL_DT / self.n
        self.reanchor = math.radians(opts["reanchor_deg"])
        self.held_jaw = bool(opts["held_jaw"])
        self.nh = int(math.ceil(float(hi[SERVO_KEYS.index("delay_ms")].max()) / CONTROL_DT)) + 2   # history (steps)
        self.gen = torch.Generator(self.dev).manual_seed(int(seed))
        self.w = (torch.tensor(SERVO_W_BIAS, **f), torch.tensor(SERVO_W_CUBE, **f))
        self.draw = torch.zeros(B, len(SERVO_KEYS), 6, **f)          # per-world values (SI units, ``SERVO_KEYS``)
        self.p = {k: torch.zeros(B, 6, **f) for k in ("delay", "tau", "db", "db1", "dbs", "kg", "kh", "b", "vmax")}
        self.p["tau"][:], self.p["vmax"][:] = 0.04, 1e3
        self.m = torch.zeros(B, self.n, 6, dtype=torch.long, device=self.dev)    # command age per sub-step (steps)
        self.y = torch.zeros(B, 6, **f)                                 # the emulated real joint position
        self.moving = torch.zeros(B, 6, dtype=torch.bool, device=self.dev)
        self.hist = torch.zeros(B, self.nh, 6, **f)                     # hist[:, m] = the command sent m steps ago
        self.fresh = torch.ones(B, dtype=torch.bool, device=self.dev)   # reset since the last step: held flag stale

    def reseed(self, seed):
        self.gen.manual_seed(int(seed))

    def reset(self, idx, q):
        """Episode start of worlds ``idx`` (q [n, 6] measured): new draws; y = q, still, command history = q."""
        n = len(idx)
        u = torch.rand(n, len(SERVO_KEYS), 6, generator=self.gen, dtype=torch.float64, device=self.dev)
        lo, hi = self.lo, self.hi
        lin = lo + u * (hi - lo)                                        # lo == hi: exactly lo
        llo, lhi = torch.log(lo.clamp_min(1e-300)), torch.log(hi.clamp_min(1e-300))
        v = torch.where(self.log & (hi > lo), torch.exp(llo + u * (lhi - llo)), lin)
        km = SERVO_KEYS.index("cube_mass_kg")
        v[:, km] = v[:, km, :1].expand(n, 6)                            # one cube mass per world
        self.draw[idx] = v
        K = {k: v[:, i] for i, k in enumerate(SERVO_KEYS)}
        p = self.p
        for a, k in (("delay", "delay_ms"), ("tau", "tau_ms"), ("db", "deadband_deg"), ("db1", "deadband_per_Nm_deg"),
                     ("dbs", "breakaway_extra_deg"), ("kg", "compliance_rad_per_Nm"), ("b", "bias_deg"),
                     ("vmax", "vmax_rad_s")):
            p[a][idx] = K[k]
        p["kh"][idx] = K["compliance_rad_per_Nm"] * K["cube_mass_kg"]
        dsub = torch.round(K["delay_ms"] / self.h).long()              # delay in sub-steps
        i = torch.arange(self.n, device=self.dev)[None, :, None]
        self.m[idx] = torch.div(dsub[:, None, :] - i + self.n - 1, self.n, rounding_mode="floor").clamp(0, self.nh - 1)
        q = q.to(torch.float64)
        self.y[idx] = q
        self.moving[idx] = False
        self.hist[idx] = q[:, None, :]
        self.fresh[idx] = True

    def step(self, cmd, q, held=None):
        """One 20 ms step: cmd [B, 6] the command sent (after delay + integral correction), q [B, 6] the measured q
        (world), held [B] bool the task's held flag (None: not held) -> the target for the world [B, 6] (y at the end
        of the step; ``held_jaw``: a held jaw that squeezes gets the settle point instead)."""
        p, h = self.p, self.h
        cmd, qw = cmd.to(torch.float64), q.to(torch.float64)
        hb = torch.zeros(self.B, dtype=torch.bool, device=self.dev) if held is None else \
            held.to(self.dev).bool() & ~self.fresh                      # a reset world's flag is the old episode's
        hf = hb.to(torch.float64)
        self.fresh = torch.zeros_like(self.fresh)
        self.hist = torch.cat((cmd[:, None], self.hist[:, :-1]), 1)
        y = torch.where((self.y - qw).abs() > self.reanchor, qw, self.y)
        if self.held_jaw:                 # the held cube blocks the jaw: it cannot close past the world's jaw angle
            y = torch.cat((y[:, :5], torch.where(hb, torch.maximum(y[:, 5], qw[:, 5]), y[:, 5])[:, None]), -1)
        G, Gc = servo_gravity(qw, self.w)
        o = p["b"] - p["kg"] * G - p["kh"] * Gc * hf[:, None]
        tau, vmax, dbs = p["tau"].clamp_min(1e-4), p["vmax"], p["dbs"]
        db = p["db"] + p["db1"] * G.abs()                               # the dead band grows with the load
        vt = vmax * tau
        th_mov, th_still = db + SERVO_EPS, db + dbs                     # stiction: a still joint needs db + dbs
        R = torch.gather(self.hist, 1, self.m) + o[:, None]             # [B, n_sub, 6] the settle point per sub-step
        mov = self.moving
        for i in range(self.n):
            r = R[:, i]
            e = r - y
            ae, sg = e.abs(), torch.sign(e)
            mv = torch.where(mov, ae > th_mov, ae > th_still)
            rp = r - db * sg                                            # it settles at the dead band's edge
            ep = (rp - y).abs()
            t1 = (ep - vt) / vmax                                       # length of the speed-limited phase
            q_lin = y + sg * vmax * h
            tb = t1.clamp(0.0, h)
            qb = torch.where(ep > vt, rp - sg * vt, y)                  # the state after the speed-limited phase
            q_exp = rp - (rp - qb) * torch.exp(-(h - tb) / tau)
            y = torch.where(mv, torch.where(t1 >= h, q_lin, q_exp), y)
            mov = mv
        self.y, self.moving = y, mov
        if not self.held_jaw:
            return y
        rj = R[:, -1, 5]                  # held and squeezing (settle point below the jaw angle - the dead band): the
        sq = hb & (rj < qw[:, 5] - db[:, 5])                            # world's jaw target is the settle point
        return torch.cat((y[:, :5], torch.where(sq, rj, y[:, 5])[:, None]), -1)

    def summary(self, on):
        """``SERVO_SUMMARY`` over the worlds ``on`` [B] bool: share of still arm joints in the last step, mean arm dead
        band (deg, without the load term)."""
        if not bool(on.any()):
            return dict.fromkeys(SERVO_SUMMARY, float("nan"))
        return dict(dyn_servo_still=round(float((~self.moving[on, :5]).double().mean()), 3),
                    dyn_servo_db_deg=round(float(torch.rad2deg(self.p["db"][on, :5]).mean()), 3))


class RobotDyn:
    """Per-world state of ``RobotDynConfig`` for B worlds on one device (``JointVecEnv.step``). Own generator (seeded
    from the env seed; ``reseed`` at ``reset_tasks``), so the env's other draws never change. Six uniforms per world
    and reset (perturbed, delay, gain, tau, sag, sigma), B uniforms per decision (jitter) and B x 5 normals per step
    (noise), all drawn whatever the ranges are, so paired settings share their draws. A neutral config
    (``RobotDynConfig.neutral``) never activates the stage. ``servo`` (``RealServo``, None = off) has its own
    generator: the draws above are the same with and without it."""
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
        self.prev5 = torch.zeros(B, STEPS, 6, **f)  # the previous decision's action-layer targets (delay_mode queue)
        self.f = torch.zeros(B, 6, **f)           # the lag filter's state
        self.integ = torch.zeros(B, 5, **f)       # the integral correction (rad, arm joints; i_time_s > 0)
        self.active = False                       # any world perturbed (else ``JointVecEnv.step`` skips the stage)
        self.eligible = None                      # [B] bool: worlds that may be perturbed (None = all; ``set_eligible``)
        self.sag_idx = torch.tensor(SAG_JOINTS, dtype=torch.long, device=self.dev)
        self.servo = RealServo(cfg.servo, B, self.dev, _servo_seed(seed)) if cfg.servo else None

    def reseed(self, seed):
        self.gen.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
        if self.servo is not None:
            self.servo.reseed(_servo_seed(int(seed) & 0x7FFFFFFFFFFFFFFF))

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
        self.prev5[idx] = q[:, None, :].expand(-1, STEPS, -1)
        self.integ[idx] = 0.0
        if self.servo is not None:        # the fitted servo: its own draws, emulated position = q
            self.servo.reset(idx, q)
        self.active = bool(self.on.any()) and not c.neutral

    def begin(self):
        """Decision start: this decision's delay d = clip(d0 + jitter, 0, STEPS - 1)."""
        r = torch.rand(self.B, generator=self.gen, dtype=torch.float64, device=self.dev)
        j = self.cfg.jitter
        self.d = (self.d0 + (r >= 1.0 - j / 2).long() - (r < j / 2).long()).clamp(0, STEPS - 1)

    def apply(self, j, ex, u, q, tlo, thi, qd=None, held=None):
        """Step j of the decision: ``ex`` = the action layer's targets of steps 0 .. j (list of [B, 6]), ``u`` = ex[j],
        q [B, 6] the measured q now, ``qd`` [B, 6] its speed (integral correction; None = not moving), ``held`` [B]
        the task's held flag (the servo's cube term and held-cube jaw rule; None = not held; unused without the servo)
        -> the target sent to the world [B, 6] (unperturbed rows: ``u`` itself)."""
        B, d, c = self.B, self.d, self.cfg
        ar = torch.arange(B, device=self.dev)
        new = torch.stack(ex, 1)[ar, (j - d).clamp(min=0)]                     # u5[j - d] after the action layer
        if c.delay_mode == "queue":       # every target d steps late: the first d steps run prev5[5 - d + j]
            old = self.prev5[ar, (STEPS - d + j).clamp(max=STEPS - 1)]
            cmd = torch.where((d > j)[:, None], old, new)
            if j == STEPS - 1:
                self.prev5 = torch.stack(ex, 1).to(torch.float64)
        else:
            cmd = torch.where((d > j)[:, None], self.hold, new)                 # the first d steps: the held command
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
            if c.i_antiwindup:            # the robot guard's anti-windup at the actuator range (RobotDynConfig)
                lo5 = tlo[..., :5].to(torch.float64).expand(B, 5)
                hi5 = thi[..., :5].to(torch.float64).expand(B, 5)
                c5, w5 = cmd[:, :5], cmd[:, :5] + self.integ
                tol = 1e-9
                sat = (c5 >= hi5 - tol) | (c5 <= lo5 + tol) | (w5 >= hi5 - tol) | (w5 <= lo5 + tol)
                e = torch.where(sat, torch.zeros_like(e), e)
                new = (self.integ + e * (CONTROL_DT / c.i_time_s)).clamp(-im, im)
                at = (c5 >= hi5 - tol) | (c5 <= lo5 + tol)      # a command at / beyond the range: no correction
                new = torch.where(at, torch.zeros_like(new), torch.minimum(torch.maximum(new, lo5 - c5), hi5 - c5))
                self.integ = torch.where(self.on[:, None], new, self.integ)
            else:
                self.integ = torch.where(self.on[:, None], (self.integ + e * (CONTROL_DT / c.i_time_s)).clamp(-im, im),
                                         self.integ)
            cmd = torch.cat((cmd[:, :5] + self.integ, cmd[:, 5:]), -1)
        if self.servo is not None:        # the fitted real servo (arm + jaw) replaces the lag / sag / gain (neutral)
            v = self.servo.step(cmd, q, held)
            arm = v[:, :5]
        else:
            a = self.a[:, None]
            self.f = torch.where(a >= 1.0, cmd, self.f + a * (cmd - self.f))   # first-order lag of the command
            v = self.f.clone()
            v[:, self.sag_idx] += self.sag[:, None]                             # static error (before the gain)
            s = self.s[:, None]
            arm = torch.where(s == 1.0, v[:, :5], q[:, :5] + s * (v[:, :5] - q[:, :5]))
        arm = arm + self.sigma[:, None] * torch.randn(B, 5, generator=self.gen, dtype=torch.float64, device=self.dev)
        out = torch.minimum(torch.maximum(torch.cat((arm, v[:, 5:]), -1), tlo), thi)    # actuator range
        return torch.where(self.on[:, None], out, u)

    def params(self):
        """[B, 6] float64 per-world draws (``COLS``; sag in deg)."""
        return torch.stack((self.on.double(), self.d0.double(), self.s, self.tau, torch.rad2deg(self.sag), self.sigma),
                           -1)

    def servo_params(self):
        """[B, len(SERVO_KEYS), 6] float64 per-world servo draws in SI units (``SERVO_SI``); None without the servo."""
        return None if self.servo is None else self.servo.draw

    def summary(self):
        """Compact draw statistics (perturbed share; means / medians over the perturbed worlds; the mean |integral
        correction| in deg) for logs. Always the keys ``SUMMARY`` (NaN while no world is perturbed), so the log's
        csv header (taken from its first row) has every column; with the servo also ``SERVO_SUMMARY``."""
        on = self.on
        out = dict.fromkeys(self.SUMMARY, float("nan"))
        out["dyn_on"] = round(float(on.double().mean()), 3)
        if bool(on.any()):
            r = lambda x, k=3: round(float(x), k)                                     # noqa: E731
            out.update(dyn_d0=r(self.d0[on].double().mean()), dyn_d=r(self.d[on].double().mean()),
                       dyn_s_p50=r(self.s[on].median(), 4), dyn_tau_p50=r(self.tau[on].median(), 4),
                       dyn_sag_deg=r(torch.rad2deg(self.sag[on]).mean()), dyn_sigma=r(self.sigma[on].mean(), 5),
                       dyn_i_deg=r(torch.rad2deg(self.integ[on].abs()).mean()))
        if self.servo is not None:
            out.update(self.servo.summary(on))
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
    goal_jump: object = None                          # GoalJumpConfig (or its dict): goal jumps; None = off
    roll_hi_deg: object = None                        # wrist-roll target cap (sim deg; module docstring); None = off
    rest: object = None                               # RestConfig (or its dict): held-cube rest starts; None = off
    far_open: object = None                           # FarOpenConfig (or its dict): jaw-open cost far from the goal
    arm_table: object = None                          # ArmTableConfig (or its dict): kinematic arm-table end
    grasp_shift: object = None                        # GraspShiftConfig (or its dict): grasp target toward the
                                                      # fixed jaw (placement cost at the descent and the close)
    floor_raise: object = None                        # (c0, c1, c2) m or a flat height: the action-layer floor
                                                      # 1.5 mm above z = raise(xy) (module docstring); None = off

    def __post_init__(self):
        self.dyn = dyn_config(self.dyn)
        self.pause = pause_config(self.pause)
        self.goal_jump = goal_jump_config(self.goal_jump)
        self.roll_hi_deg = roll_hi_config(self.roll_hi_deg)
        self.rest = rest_config(self.rest)
        self.far_open = far_open_config(self.far_open)
        self.arm_table = arm_table_config(self.arm_table)
        self.grasp_shift = grasp_shift_config(self.grasp_shift)
        self.floor_raise = floor_raise_config(self.floor_raise)

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
        d = copy.deepcopy(asdict(self))
        if isinstance(self.dyn, RobotDynConfig):
            d["dyn"] = self.dyn.to_dict()     # no "servo" key while it is off (``RobotDynConfig.to_dict``)
        if d.get("goal_jump") is None:    # off: the old record (readable by code without goal jumps)
            d.pop("goal_jump", None)
        if d.get("roll_hi_deg") is None:  # off: the old record (readable by code without the roll cap)
            d.pop("roll_hi_deg", None)
        if d.get("rest") is None:         # off: the old record (readable by code without rest starts)
            d.pop("rest", None)
        if d.get("far_open") is None:     # off: the old record
            d.pop("far_open", None)
        if d.get("arm_table") is None:    # off: the old record
            d.pop("arm_table", None)
        if d.get("grasp_shift") is None:  # off: the old record
            d.pop("grasp_shift", None)
        if d.get("floor_raise") is None:  # off: the old record
            d.pop("floor_raise", None)
        else:
            d["floor_raise"] = list(d["floor_raise"])
        return d


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
        # wrist-roll cap (module docstring; None = off): the action layer's range min(THI, cap) for the wrist roll
        self.roll_hi = None if cfg.roll_hi_deg is None else math.radians(float(cfg.roll_hi_deg))
        if self.roll_hi is not None:
            self.thi = self.thi.clone()                                  # kin.thi may be shared: never in place
            self.thi[ROLL] = min(float(self.thi[ROLL]), self.roll_hi)
        self.lead = torch.tensor(LEAD, dtype=torch.float64, device=dev) if cfg.lead_clamp else None
        self.gen = torch.Generator(dev).manual_seed(int(cfg.seed))
        if share is not None:
            self.bank = share.bank
        else:
            self.bank = VE.load_starts(cfg.starts, cfg.start_split, dev) if cfg.starts and cfg.world != "mujoco" else None
        self.roll_dropped = 0             # roll cap: start-bank rows dropped (wrist roll above the cap)
        if self.roll_hi is not None and self.bank is not None:
            self.bank, self.roll_dropped = drop_high_roll(self.bank, float(self.thi[ROLL]))
        self.far_open_sum = torch.zeros((), dtype=torch.float64, device=dev)   # FarOpenConfig: summed cost (log)
        self.arm_table_n = torch.zeros((), dtype=torch.long, device=dev)       # ArmTableConfig: ends by the rule (log)
        # GraspShiftConfig (None = off): the previous step's TCP height above the cube centre and measured jaw (set at
        # every reset), and the summed event statistics (log; GS_STATS order)
        self.gs_h = self.gs_jaw = None
        self.gs_stats = torch.zeros(len(GS_STATS), dtype=torch.float64, device=dev)
        if cfg.grasp_shift is not None:
            self.gs_h = torch.zeros(B, dtype=torch.float64, device=dev)
            self.gs_jaw = torch.zeros(B, dtype=torch.float64, device=dev)
        self.rest = None                 # held-cube rest starts (RestConfig; None = off): the bank, own generator
        if cfg.rest is not None:
            if self.bank is None:
                raise ValueError("rest starts: automatic resets from a start bank only (cfg.starts, world not mujoco)")
            self.rest, self.rest_dropped = load_rest_bank(cfg.rest.bank, dev), 0
            if self.roll_hi is not None:
                self.rest, self.rest_dropped = drop_high_roll(self.rest, float(self.thi[ROLL]))
            self.rest_gen = torch.Generator(dev).manual_seed((int(cfg.seed) * 0x27D4EB2F + 0x7E5) & 0x7FFFFFFFFFFF)
            self.rest_n, self.rest_resets = 0, 0                 # rest starts / all automatic resets so far (log)
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
        # floor raise (JointEnvConfig.floor_raise; None = off): flat -> raise_z (added to floor_z), tilted -> raise_lift
        # (floor_guard lift: measured at each pose's grasp point)
        fr = cfg.floor_raise
        self.raise_z = None if fr is None or fr[1] != 0.0 or fr[2] != 0.0 else fr[0]
        self.raise_lift = None if fr is None or self.raise_z is not None else fr
        # robot dynamics stage (real2sim; None = off) with its own generator, and the JT observation's held-time scale
        self.dyn = None if cfg.dyn is None else RobotDyn(cfg.dyn, B, dev, (int(cfg.seed) * 0x9E3779B1 + 0xD1A) &
                                                         0x7FFFFFFFFFFF)
        if self.dyn is not None and self.dyn.servo is not None and EVAL_KP_SCALE != 1.0:
            raise ValueError(f"robot dyn servo with $NEDM_SO101_EVAL_KP_SCALE = {EVAL_KP_SCALE:g}: the fitted servo "
                             "already has the real servo's softness (counted twice); unset it (1)")
        self.pause = None if cfg.pause is None else Pause(cfg.pause, B, dev, (int(cfg.seed) * 0x2545F491 + 0x9A5) &
                                                          0x7FFFFFFFFFFF)
        self.goal_jump = None if cfg.goal_jump is None else GoalJump(cfg.goal_jump, B, dev, (int(cfg.seed) * 0x5851F42D
                                                                                            + 0x6A1) & 0x7FFFFFFFFFFF)
        self.obs_tcfg = None
        if cfg.hold_obs_s and float(cfg.hold_obs_s) != self.tcfg.hold_cap_s:
            from dataclasses import replace
            self.obs_tcfg = replace(self.tcfg, hold_cap_s=float(cfg.hold_obs_s))
        if self.roll_hi is not None:      # per episode: the largest commanded / measured wrist roll (policy frame)
            self.roll_cmd_max = torch.full((B,), -math.inf, **f)
            self.roll_q_max = torch.full((B,), -math.inf, **f)

    # ---- the wrist-roll cap (module docstring) ------------------------------------------------------------------
    def roll_cap(self) -> float | None:
        """The wrist-roll cap of the target sent to the world (rad, world frame: the action layer's cap minus the joint
        offset of ``set_table``), None = off."""
        if self.roll_hi is None:
            return None
        hi = float(self.thi[ROLL])
        return hi if self.q_offset is None else hi - float(self.q_offset[ROLL])

    def cap_roll(self, u):
        """Targets [..., 6] (world frame) -> a new tensor with the wrist roll <= ``roll_cap()``; off: ``u`` itself."""
        c = self.roll_cap()
        if c is None:
            return u
        return torch.cat((u[..., :ROLL], u[..., ROLL:ROLL + 1].clamp(max=c), u[..., ROLL + 1:]), -1)

    def roll_info(self) -> dict | None:
        """The cap for output files (None = off): sim deg, the action layer's cap (rad), start-bank rows dropped."""
        if self.roll_hi is None:
            return None
        return dict(roll_hi_deg=float(self.cfg.roll_hi_deg), cap_rad=float(self.thi[ROLL]),
                    bank_dropped=int(self.roll_dropped))

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
        if self.cfg.floor_raise is not None and floor != "zero":
            raise ValueError(f"set_table floor {floor!r} with floor_raise {list(self.cfg.floor_raise)}: one floor "
                             "rule at a time (floor 'zero' + the raise, or no raise)")
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
        out = dict(plane=self.table_plane, floor_plane=self.floor_plane, floor_z=self.floor_z,
                   q_offset=None if self.q_offset is None else self.q_offset.tolist())
        if self.cfg.floor_raise is not None:      # (off: the old keys)
            out["floor_raise"] = list(self.cfg.floor_raise)
        return out

    def dyn_info(self) -> dict | None:
        """The robot dynamics setting (``RobotDynConfig`` as a dict) + the per-world draws of the current episodes
        (``RobotDyn.COLS``) + the hold cap / observation scale; None = off (evaluation files: the "dyn" key). With the
        fitted servo also its per-world draws ``servo_params`` [B, len(SERVO_KEYS), 6] (SI units, ``servo_keys``)."""
        if self.dyn is None:
            return None
        out = dict(cfg=self.dyn.cfg.to_dict(), cols=list(RobotDyn.COLS), params=self.dyn.params().cpu().tolist(),
                   hold_cap_s=self.tcfg.hold_cap_s, hold_obs_s=float(self.cfg.hold_obs_s), kp_scale=EVAL_KP_SCALE)
        if self.dyn.servo is not None:
            out.update(servo_keys=list(SERVO_KEYS), servo_params=self.dyn.servo_params().cpu().tolist())
        return out

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
        if self.gs_h is not None:         # GraspShiftConfig: the start state is the first "previous step"
            F0 = T.features(self.ws.index(idx), self.kin, None, self.tcfg)
            self.gs_h[idx], self.gs_jaw[idx] = grasp_height(F0), F0["jaw"]
        self.task_seed[idx] = seeds
        self.ep_ret[idx], self.ep_len[idx] = 0.0, 0
        if self.dyn is not None:          # robot dynamics: new draws, hold / lag state = the measured start q
            self.dyn.reset(idx, self.ws.q[idx])
        if self.pause is not None:
            self.pause.reset(idx)
        if self.goal_jump is not None:    # goal jumps: per-episode draw, carry direction start cube -> goal
            self.goal_jump.reset(idx, self.ws.cube_pos[idx, :2], goal_xy)
        if self.roll_hi is not None:      # roll cap records: the measured start roll (policy frame), no command yet
            r0 = self.ws.q[idx, ROLL].to(torch.float64) + (0.0 if self.q_offset is None else self.q_offset[ROLL])
            self.roll_q_max[idx] = r0.nan_to_num(-math.inf)
            self.roll_cmd_max[idx] = -math.inf

    def _pool_reset(self, idx):
        from nedm.so101_policy.vec_env import start_rows
        if self.pool is None:
            self.refresh_pool()
        k = torch.randint(len(self.pool["i"]), (len(idx),), generator=self.gen, device=self.dev)
        i = self.pool["i"][k]
        s = (int(self.cfg.seed) * 0x9E3779B1 + self.ep_count[idx] * self.B + idx) & 0xFFFFFFFF
        self.ep_count[idx] += 1
        if self.rest is None:
            self._reset_rows(idx, start_rows(self.bank, i), self.pool["goal_xy"][k], self.pool["goal_yaw"][k], s)
            return
        # held-cube rest starts (RestConfig): a share of the rows from the rest bank; the others as above (same draws)
        r = torch.rand(len(idx), generator=self.rest_gen, dtype=torch.float64, device=self.dev) < self.cfg.rest.share
        n = ~r
        if bool(n.any()):
            self._reset_rows(idx[n], start_rows(self.bank, i[n]), self.pool["goal_xy"][k[n]],
                             self.pool["goal_yaw"][k[n]], s[n])
        if bool(r.any()):
            ri, R = idx[r], self.rest
            j = torch.randint(len(R["q"]), (len(ri),), generator=self.rest_gen, device=self.dev)
            self._reset_rows(ri, start_rows(R, j), R["goal_xy"][j], R["goal_yaw"][j], s[r])
            self.set_rest(ri, R["lifted"][j], R["air"][j], R["jaw_cmd"][j] if "jaw_cmd" in R else None)
        self.rest_n += int(r.sum())
        self.rest_resets += len(idx)

    def set_rest(self, idx, lifted, air_steps, jaw_cmd=None):
        """Worlds ``idx`` (just reset) start at rest with the cube in the jaw (``RestConfig``; fixed tasks:
        ``joint_teacher.run_tasks``): the task's lifted flag ``lifted`` [n] bool, the held time ``air_steps`` [n] (20 ms
        steps, the tracker's ``air_run``: the JT observation's held-time input), held now (``since_release`` 0); with
        the robot dynamics stage its past jaw commands = ``jaw_cmd`` [n] (rad: the closed command of the carry, not
        the measured jaw; None = keep the reset's measured jaw). The caller computes the observation afterwards."""
        tr = self.task.tr
        tr.lifted[idx] = torch.as_tensor(lifted, device=self.dev).bool()
        tr.air_run[idx] = torch.as_tensor(air_steps, device=self.dev).long()
        tr.prev_held[idx] = True
        tr.since_release[idx] = 0
        if jaw_cmd is not None and self.dyn is not None:
            cj = torch.as_tensor(jaw_cmd, dtype=torch.float64, device=self.dev)
            self.dyn.hold[idx, 5] = cj
            self.dyn.prev5[idx, :, 5] = cj[:, None]
            self.dyn.f[idx, 5] = cj
            if self.dyn.servo is not None:
                self.dyn.servo.hist[idx, :, 5] = cj[:, None]

    def rest_info(self) -> dict | None:
        """The rest-start option for logs (None = off): bank, share, rows (after the roll cap), rows dropped, rest
        starts / all automatic resets so far."""
        if self.rest is None:
            return None
        return dict(bank=self.cfg.rest.bank, share=self.cfg.rest.share, rows=int(len(self.rest["q"])),
                    dropped=int(self.rest_dropped), rest_starts=int(self.rest_n), resets=int(self.rest_resets))

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
        if self.goal_jump is not None:    # fixed tasks: goal-jump draws from the env seed and the task seeds
            sd = torch.as_tensor(seeds).reshape(-1)
            self.goal_jump.reseed(int(self.cfg.seed) * 1_000_033 + 7927 * (int(sd[0]) if len(sd) else 0) + len(sd)
                                  + 0x6A1)
        if self.roll_hi is not None:      # a start above the cap: the first targets pull the roll down to it
            r0 = st.q[:, ROLL].to(torch.float64) + (0.0 if self.q_offset is None else self.q_offset[ROLL])
            n_hi = int((r0 > float(self.thi[ROLL]) + 1e-9).sum())
            if n_hi:
                import warnings
                warnings.warn(f"roll cap {self.cfg.roll_hi_deg:g} deg: {n_hi} of {len(r0)} task starts have the wrist "
                              f"roll above it (max {math.degrees(float(r0.max())):.1f} deg)", stacklevel=2)
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
        fz = self.floor_z if self.raise_z is None else self.floor_z + self.raise_z     # (floor raise: flat part)
        fl = (FLOOR if fz == 0.0 else FLOOR + fz) if self.cfg.floor_guard else None
        lk = {} if self.raise_lift is None else dict(floor_lift=self.raise_lift)   # (floor raise: tilted; off: {})
        dyn = self.dyn if self.dyn is not None and self.dyn.active else None      # None: the old path, bit for bit
        if dyn is not None:
            dyn.begin()
        if self.pause is not None:
            self.pause.begin(task, self.ws.q)
        rc = self.roll_hi is not None     # the wrist-roll cap (off: the old path, bit for bit)
        fo = self.cfg.far_open if task.F is not None else None   # FarOpenConfig (None: the old reward, bit for bit)
        atc = self.cfg.arm_table          # ArmTableConfig (None: the old world flags, bit for bit)
        gs = self.cfg.grasp_shift         # GraspShiftConfig (None: the old reward, bit for bit)
        for j in range(STEPS):
            if rc:                        # the commanded roll before the action layer (policy frame), live worlds
                self.roll_cmd_max = torch.where(task.tr.end == 0, torch.fmax(self.roll_cmd_max, u5[:, j, ROLL]),
                                                self.roll_cmd_max)
            if self.q_offset is None:
                u = safety(u5[:, j], self.ws.q.to(torch.float64), self.tlo, self.thi, self.lead, self.kin, fl,
                           self.floor_plane, **lk)
            else:   # the action layer in the policy's joint frame (targets u5 are in it), mapped back to the world
                qo = self.q_offset
                u = safety(u5[:, j], self.ws.q.to(torch.float64) + qo, self.tlo, self.thi, self.lead, self.kin, fl,
                           self.floor_plane, **lk) - qo
            if rc:                        # the cap after the floor guard too (it pulls a target toward a measured roll
                u = self.cap_roll(u)      # above the cap), as the guard's range clamp after its action layer
            ex.append(u)                # u_exe: the action layer's targets (before the robot dynamics / servo scale)
            if dyn is not None:           # the robot's timing and servo (RobotDyn), perturbed worlds only; the held
                u = dyn.apply(j, ex, u, self.ws.q, self.tlo, self.thi, self.ws.qd,       # flag: the servo's cube / jaw
                              None if task.F is None else task.F["held"])
            if EVAL_KP_SCALE != 1.0:
                qn = self.ws.q.to(device=u.device, dtype=u.dtype)
                u = u.clone()
                u[:, :5] = qn[:, :5] + EVAL_KP_SCALE * (u[:, :5] - qn[:, :5])
            if self.pause is not None:    # the robot's hold state (training only): frozen arm in paused worlds
                u = self.pause.apply(u)
            if rc:                        # the cap on the target sent to the world, after every stage
                u = self.cap_roll(u)
                lv = task.tr.end == 0
            if fo is not None:            # FarOpenConfig: held, lifted, far from the goal (task.F of the last step);
                Fp = task.F               # the executed jaw target above the measured jaw before this step
                far = (task.tr.end == 0) & Fp["held"] & task.tr.lifted & \
                    ((Fp["c"][:, :2] - task.goal_xy).norm(dim=-1) > fo.far_m)
                jo = (ex[-1][:, 5].to(torch.float64) - self.ws.q[:, 5].to(torch.float64)).clamp(min=0.0)
                fo_cost = torch.where(far, fo.cost * jo, torch.zeros_like(jo))
            ws = self.world.step(u)
            self.ws = ws if self.cfg.world != "mujoco" else ws.to(self.dev)
            if atc is not None:           # ArmTableConfig: the kinematic arm-table end (a new WorldState, the world's
                hit = arm_table_rule(self.ws.q, self.ws.qd, self.kin, atc, self.table_plane)   # own record unchanged)
                if atc.grasp_only:        # before the first firm lift only (the tracker before this step)
                    hit = hit & ~task.tr.lifted
                self.arm_table_n = self.arm_table_n + (hit & (task.tr.end == 0)).sum()
                self.ws = replace(self.ws, arm_table=self.ws.arm_table.to(device=hit.device, dtype=torch.bool) | hit)
            if rc:                        # the measured roll after the step (policy frame)
                rq = self.ws.q[:, ROLL].to(torch.float64) + (0.0 if self.q_offset is None else self.q_offset[ROLL])
                self.roll_q_max = torch.where(lv, torch.fmax(self.roll_q_max, rq), self.roll_q_max)
            if gs is not None:            # the tracker before this step: live worlds, lifted flag
                gs_live, gs_lifted = task.tr.end == 0, task.tr.lifted.clone()
            r, _, _, info = task.step(self.ws, None, None)
            if fo is not None:
                r = r - fo_cost
                self.far_open_sum = self.far_open_sum + fo_cost.sum()
            if gs is not None:            # GraspShiftConfig: placement cost at the descent / close events
                e1, e2, err, h = grasp_shift_events(task.F, self.gs_h, self.gs_jaw, gs_lifted, gs)
                e1, e2 = e1 & gs_live, e2 & gs_live
                gc = (e1.to(torch.float64) + e2.to(torch.float64)) * grasp_shift_cost(err, gs)
                r = r - gc
                hx = task.F["hx"]
                self.gs_stats = self.gs_stats + torch.stack((
                    e1.sum().to(torch.float64), (hx * e1).sum(), (err.abs() * e1).sum(),
                    e2.sum().to(torch.float64), (hx * e2).sum(), (err.abs() * e2).sum(), gc.sum()))
                self.gs_h, self.gs_jaw = h, task.F["jaw"].clone()   # own tensors (F["jaw"] can view ws.q)
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
        if self.goal_jump is not None:    # jumps of the episode (final for the worlds that ended in this step)
            ep["goal_jumps"] = self.goal_jump.n.clone()
        if rc:                            # the largest commanded / measured wrist roll of the episode so far (rad)
            ep["roll_cmd_max"], ep["roll_q_max"] = self.roll_cmd_max.clone(), self.roll_q_max.clone()
        obs = self._observe(task.F)
        if self.goal_jump is not None:    # after the outcome record: the next decision sees the moved goals
            gj = self.goal_jump.apply(task, task.tr.end == 0)
            if len(gj):
                obs = obs.clone()
                obs[gj] = markov_obs({k: v[gj] for k, v in task.F.items() if torch.is_tensor(v)}, task, gj,
                                     self.obs_tcfg)
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
