"""Go2 URDF loading and joint access. Moved verbatim."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import (CALF_BODIES, FOOT_BODIES, JOINT_EFFORT_NM,
                          MOTOR_NAMES, PD_KP, PD_KD)


class Go2Robot:
    """The Go2, on either of two actuator plants.

    "torque" (default) is a PD law driving ChLinkMotorRotationTorque, clamped to
    the URDF effort limits -- what every legged_gym-family policy assumes.

    "position" is the historical plant: ChLinkMotorRotationAngle, a KINEMATIC
    CONSTRAINT. Measured tracking error there is ~1e-8 rad in steady state, which
    means infinite joint stiffness, zero compliance and UNBOUNDED joint torque --
    the robot could produce motions no real Go2 could. model_2999 was trained
    against it, and every gate and dataset before 2026-09-03 was measured on it,
    so it is kept selectable to reproduce those numbers rather than deleted.
    """

    def __init__(self, chsystem, urdf_path: Path, init_frame, actuation: str = "torque"):
        import pychrono as chrono
        import pychrono.parsers as parsers

        if actuation not in ("torque", "position"):
            raise ValueError(f"actuation must be 'torque' or 'position', got {actuation!r}")
        self.chrono = chrono
        self.actuation = actuation
        self.parser = parsers.ChParserURDF(str(urdf_path))
        self.parser.SetRootInitPose(init_frame)
        # MUST precede PopulateSystem. See docstring note 1.
        # NB the enum member is ActuationType_FORCE, not _TORQUE.
        self.parser.SetAllJointsActuationType(
            parsers.ChParserURDF.ActuationType_FORCE if actuation == "torque"
            else parsers.ChParserURDF.ActuationType_POSITION)
        for name in FOOT_BODIES:
            self.parser.SetBodyMeshCollisionType(
                name, parsers.ChParserURDF.MeshCollisionType_CONVEX_HULL)
        # Deliberately NOT "base": the URDF already declares a tight box there,
        # and a hull from trunk.obj engulfs the legs and makes the dog sprawl.
        self.parser.PopulateSystem(chsystem)
        self.parser.GetRootChBody().SetFixed(False)
        self._configure_collision()
        self.motors = [self.parser.GetChMotor(n) for n in MOTOR_NAMES]
        self.target = np.zeros(len(MOTOR_NAMES))

    def _configure_collision(self):
        c = self.chrono
        mat = c.ChContactMaterialSMC()
        mat.SetFriction(0.9)
        mat.SetRestitution(0.01)
        mat.SetGn(60.0)
        mat.SetKn(2e5)
        for name in FOOT_BODIES + ["base"]:
            body = self.parser.GetChBody(name)
            if body is None:
                continue
            body.EnableCollision(True)
            if body.GetCollisionModel() is not None:
                body.GetCollisionModel().SetAllShapesMaterial(mat)
        # Calves collide with SOIL through FSI, not through the contact system;
        # leaving rigid collision on invites self-collision artifacts in gait.
        for name in CALF_BODIES:
            body = self.parser.GetChBody(name)
            if body is not None:
                body.EnableCollision(False)

    def body(self, name):
        return self.parser.GetChBody(name)

    def base(self):
        return self.parser.GetChBody("base")

    def joint_pos(self) -> np.ndarray:
        c = self.chrono
        return np.array([c.CastToChLinkMotorRotation(m).GetMotorAngle()
                         for m in self.motors], dtype=np.float32)

    def joint_vel(self) -> np.ndarray:
        c = self.chrono
        return np.array([c.CastToChLinkMotorRotation(m).GetMotorAngleDt()
                         for m in self.motors], dtype=np.float32)

    def actuate(self, chrono_order_angles: np.ndarray) -> None:
        """Set the joint target. On the position plant this IS the command; on
        the torque plant it is the PD setpoint, and apply_pd does the work."""
        self.target = np.asarray(chrono_order_angles, dtype=np.float64).copy()
        if self.actuation == "position":
            c = self.chrono
            for motor, angle in zip(self.motors, self.target):
                motor.SetMotorFunction(c.ChFunctionConst(float(angle)))

    def apply_pd(self) -> np.ndarray:
        """One PD update: tau = kp*(q_target - q) - kd*qd, clamped to effort.

        MUST BE CALLED EVERY PHYSICS STEP, not every control step. legged_gym
        runs its PD at 200 Hz under a 50 Hz policy (decimation 4); a PD law
        evaluated only at the policy rate is a different controller and would not
        match what any policy from that family expects. Our physics step is
        2.5e-3 s, so this runs at 400 Hz.

        No-op on the position plant, so the sim loop can call it unconditionally.
        """
        if self.actuation != "torque":
            return np.zeros(len(self.motors))
        c = self.chrono
        tau = PD_KP * (self.target - self.joint_pos()) - PD_KD * self.joint_vel()
        tau = np.clip(tau, -JOINT_EFFORT_NM, JOINT_EFFORT_NM)
        for motor, t in zip(self.motors, tau):
            motor.SetMotorFunction(c.ChFunctionConst(float(t)))
        return tau


