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


class ImportedGo2Policy:
    """go2_cts_150k.pt driven on our Chrono Go2."""

    def __init__(self, ckpt: Path, command=(0.5, 0.0, 0.0)):
        import torch

        self.torch = torch
        self.ckpt = Path(ckpt)
        self.command = np.asarray(command, dtype=np.float32)
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
