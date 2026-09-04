"""Adapter for an imported legged_gym-family Go2 policy (wty-yy/go2_rl_gym).

DELIBERATELY SEPARATE FROM PolicyController. That class's entire guarantee is
that it builds observations by calling the training harness's own
_compute_observations unmodified, so the conventions are inherited rather than
reimplemented. That guarantee does not extend to a policy the harness never saw,
and sharing code would weaken it to no benefit. This class reimplements the
observation ON PURPOSE, against the imported policy's OWN published config.

EVERY CONVENTION HERE COMES FROM THEIR deploy config, NOT FROM OURS. The two
differ in three ways that would each silently corrupt the observation:

  1. JOINT ORDER is FL, FR, RL, RR. Genesis -- and therefore our existing
     CHRONO_TO_POLICY -- is FR, FL, RR, RL. Reusing ours would swap left and
     right legs while producing a perfectly plausible-looking 45-vector.
  2. DEFAULT ANGLES have non-zero hips: +0.1 on left legs, -0.1 on right.
     Genesis uses 0.0. A 0.1 rad offset on four channels is small enough to look
     like noise and large enough to matter.
  3. COMMAND SCALE is [2.0, 2.0, 0.25] -- yaw by ang_vel_scale, not by
     lin_vel_scale. Our harness multiplies the whole vector by lin_vel_scale,
     which is unobservable there only because its yaw command is identically
     zero, and 0*2.0 == 0*0.25.

THE POLICY IS STATEFUL. It is a concurrent teacher-student model: a 5-step
observation history feeds a student_encoder to a 32-dim latent, and the actor
consumes [obs 45, latent 32] = 77. Measured, it stabilises after exactly 5 calls
on a repeated input. So it must be called once per control step in order, and
RELOADED between episodes -- otherwise episode N inherits episode N-1's history.
`reset()` reloads.

THE SIGN NEGATION IS INHERITED ON FAITH. The Chrono harness negates joint
positions, velocities and targets, and no source we have records why. It is
almost certainly a Chrono URDF joint-sense convention and so should apply to any
policy trained against the standard Unitree URDF, but that cannot be verified
from the imported side, since it never ran in Chrono. If the robot walks
backwards or sideways, invert SIGN and retest before suspecting anything else.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# Their joint order, FL/FR/RL/RR, expressed as indices into our Chrono order
# (MOTOR_NAMES = RR, RL, FR, FL, each hip/thigh/calf).
CHRONO_TO_IMPORTED = [9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2]

# Their default_angles, in their own order. Hips are +/-0.1, not 0.0.
IMPORTED_DEFAULTS = np.array(
    [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    dtype=np.float32)

CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
ACTION_SCALE = 0.25

SIGN = -1.0   # see module docstring


# Structured excitation families, mirroring the paper's approach: keep the SHAPE
# and draw the AMPLITUDE per episode ("amplitudes, frequencies and durations are
# drawn per episode to span the vehicle's operating envelope").
#
# EIGHT SHAPES, NOT ELEVEN. The previous march_in_place_015/030, constant_high and
# reverse were four fixed points on one shape -- constant vx -- and collapse into
# a single family the moment amplitude is randomised. Keeping them separate would
# have been four names for one experiment.
#
# BOUNDED BY THE POLICY'S TRAINED RANGES, not its deployment clips: vx and vy in
# +/-0.5, yaw in +/-1.0. The deploy config's max_cmd [2.0, 1.0, 2.5] is wider and
# driving to it would ask for behaviour never trained.
#
# THE DEAD ZONE IS NOT SAMPLED AROUND. Drawing vx uniformly over [-0.5, 0.5] puts
# most draws below the ~0.35 m/s translation threshold, so most episodes barely
# move. That is the plant's actual behaviour and the model has to learn it;
# biasing the draw away from it would be hiding the most interesting nonlinearity
# we have measured.
PARAM_RANGES = {
    "vx": (-0.5, 0.5), "vy": (-0.5, 0.5), "wz": (-1.0, 1.0),
    "vx0": (-0.5, 0.5), "vx1": (-0.5, 0.5),
    "wz1": (-1.0, 1.0), "wz_amp": (0.2, 1.0),
    "t_switch": (4.0, 12.0), "freq": (0.05, 0.30), "period": (2.0, 6.0),
}

# WIDE RANGES, MEASURED NOT GUESSED. No training config ships with the imported
# checkpoint, so the envelope was measured directly on rigid: the policy tracks at
# 0.90-0.99 from 0.8 m/s up to 2.0 m/s forward, 0.87 at -1.4 reverse, 0.94 at
# 3.0 rad/s yaw, 0.89 at 1.0 m/s lateral, and falls nowhere in that range.
#
# The narrow ranges above sample ONLY the band where it works worst -- 0.3 m/s
# achieves 0.01 of command. Every episode collected before 2026-09-04 lives in
# that band, which is why the plant looked like it tracked at 63% of command.
#
# Capped at what was actually probed. No claim is made past 2.0 m/s or 3.0 rad/s.
PARAM_RANGES_WIDE = {
    "vx": (-1.5, 2.0), "vy": (-1.5, 1.5), "wz": (-3.0, 3.0),
    "vx0": (-1.5, 2.0), "vx1": (-1.5, 2.0),
    "wz1": (-3.0, 3.0), "wz_amp": (0.2, 3.0),
    "t_switch": (4.0, 12.0), "freq": (0.05, 0.30), "period": (2.0, 6.0),
}

FAMILY_PARAMS = {
    "constant":    ["vx"],
    "lateral":     ["vy"],
    "pivot":       ["wz"],
    "arc":         ["vx", "wz"],
    "vel_step":    ["vx0", "vx1", "t_switch"],
    "yaw_step":    ["vx", "wz1", "t_switch"],
    "weave":       ["vx", "wz_amp", "freq"],
    "stop_and_go": ["vx", "period"],
}


def _sched(family, p):
    """(t, T) -> (vx, vy, wz) for one family at one parameter draw."""
    if family == "constant":
        return lambda t, T: (p["vx"], 0.0, 0.0)
    if family == "lateral":
        return lambda t, T: (0.0, p["vy"], 0.0)
    if family == "pivot":
        return lambda t, T: (0.0, 0.0, p["wz"])
    if family == "arc":
        return lambda t, T: (p["vx"], 0.0, p["wz"])
    if family == "vel_step":
        return lambda t, T: (p["vx0"] if t < p["t_switch"] else p["vx1"], 0.0, 0.0)
    if family == "yaw_step":
        return lambda t, T: (p["vx"], 0.0, 0.0 if t < p["t_switch"] else p["wz1"])
    if family == "weave":
        return lambda t, T: (p["vx"], 0.0,
                             p["wz_amp"] * math.sin(2 * math.pi * p["freq"] * t))
    if family == "stop_and_go":
        return lambda t, T: ((p["vx"] if (t % (2 * p["period"])) < p["period"] else 0.0),
                             0.0, 0.0)
    raise KeyError(family)


def family_seed(family: str, offset: int = 0) -> int:
    """Stable per-family seed. NOT hash().

    Python salts str hashing per process (PYTHONHASHSEED), so `hash(family)`
    returns a different value every run. A collection seeded that way is not
    reproducible, cannot be resumed exactly, and cannot be coordinated with
    another machine -- all three of which we assumed it could. crc32 is stable
    across processes, machines and Python versions.

    `offset` partitions the draw space so two machines collecting in parallel
    produce DISJOINT parameters. Without it, identical family names and counts
    give byte-identical episodes, which look like data rather than duplicates.
    """
    import zlib

    return (zlib.crc32(family.encode()) + offset) & 0xFFFFFFFF


def stratified_params(family: str, n: int, seed: int, wide: bool = False) -> list[dict]:
    """n parameter draws per family, STRATIFIED rather than uniform.

    Each varying parameter's range is cut into n equal bins and one value is
    drawn from each, then the bin order is shuffled INDEPENDENTLY per parameter.
    That guarantees the range is spanned rather than merely covered in
    expectation -- which matters enormously at n=19 and not at all at n=1000.
    Uniform draws at this size leave gaps and cluster by luck.
    """
    import random

    rng = random.Random(seed)
    out = [dict() for _ in range(n)]
    table = PARAM_RANGES_WIDE if wide else PARAM_RANGES
    for name in FAMILY_PARAMS[family]:
        lo, hi = table[name]
        order = list(range(n))
        rng.shuffle(order)
        for i, b in enumerate(order):
            out[i][name] = lo + (b + rng.random()) / n * (hi - lo)
    return out


COMMAND_FAMILIES = list(FAMILY_PARAMS)


class ImportedGo2Policy:
    """go2_cts_150k.pt driven on our Chrono Go2."""

    def __init__(self, ckpt: Path, command=(0.5, 0.0, 0.0), family=None, duration=8.0,
                 params=None):
        import torch

        self.torch = torch
        self.ckpt = Path(ckpt)
        self.command = np.asarray(command, dtype=np.float32)
        self.family = family
        self.params = dict(params) if params else None
        self._sched = _sched(family, self.params) if (family and self.params) else None
        self.duration = float(duration)
        # Every command actually issued, so an episode records what it was ASKED
        # to do and not merely which family it belonged to.
        self.command_log: list[tuple[float, float, float, float]] = []
        self.last_actions = np.zeros(12, dtype=np.float32)
        self.model = None
        self.reset()

    def reset(self) -> None:
        """Reload, clearing the 5-step history. Call once per episode."""
        self.model = self.torch.jit.load(str(self.ckpt), map_location="cpu")
        self.model.eval()
        self.last_actions = np.zeros(12, dtype=np.float32)

    @staticmethod
    def _projected_gravity(q) -> np.ndarray:
        qw, qx, qy, qz = q.e0, q.e1, q.e2, q.e3
        return np.array([-2 * (qx * qz - qw * qy),
                         -2 * (qy * qz + qw * qx),
                         -(1 - 2 * (qx * qx + qy * qy))], dtype=np.float32)

    def set_time(self, t: float) -> None:
        """Advance the command schedule. No-op for a fixed command."""
        if self._sched is not None:
            vx, vy, wz = self._sched(t, self.duration)
            self.command = np.array([vx, vy, wz], dtype=np.float32)
        self.command_log.append((t, *map(float, self.command)))

    def observe(self, robot) -> np.ndarray:
        """The 45-vector in THEIR block order and THEIR scales.

        ang_vel(3) | projected_gravity(3) | command(3) | dof_pos(12) |
        dof_vel(12) | prev_actions(12)

        Note there is no base linear velocity block. That is what makes this a 45
        and not the legged_gym base config's 48, and it is why this checkpoint is
        usable at all -- we could supply base lin vel from the simulator, but a
        policy that never had it is the one that ports cleanly.
        """
        base = robot.base()
        w = base.GetAngVelLocal()
        ang = np.array([w.x, w.y, w.z], dtype=np.float32) * ANG_VEL_SCALE
        grav = self._projected_gravity(base.GetRot())
        cmd = self.command * CMD_SCALE
        q = SIGN * robot.joint_pos().astype(np.float32)[CHRONO_TO_IMPORTED]
        qd = SIGN * robot.joint_vel().astype(np.float32)[CHRONO_TO_IMPORTED]
        dof_pos = (q - IMPORTED_DEFAULTS) * DOF_POS_SCALE
        dof_vel = qd * DOF_VEL_SCALE
        return np.concatenate([ang, grav, cmd, dof_pos, dof_vel,
                               self.last_actions]).astype(np.float32)

    def act(self, robot) -> np.ndarray:
        torch = self.torch
        obs = torch.from_numpy(self.observe(robot)).unsqueeze(0)
        with torch.no_grad():
            action = self.model(obs).squeeze(0).numpy().astype(np.float32)
        self.last_actions = action
        targets = action * ACTION_SCALE + IMPORTED_DEFAULTS
        # back to Chrono order, and back through the sign convention
        out = np.zeros(12, dtype=np.float64)
        out[CHRONO_TO_IMPORTED] = targets
        return SIGN * out
