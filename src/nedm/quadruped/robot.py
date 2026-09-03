"""Go2 URDF loading and joint access. Moved verbatim."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import CALF_BODIES, FOOT_BODIES, MOTOR_NAMES


class Go2Robot:
    def __init__(self, chsystem, urdf_path: Path, init_frame):
        import pychrono as chrono
        import pychrono.parsers as parsers

        self.chrono = chrono
        self.parser = parsers.ChParserURDF(str(urdf_path))
        self.parser.SetRootInitPose(init_frame)
        # MUST precede PopulateSystem. See docstring note 1.
        self.parser.SetAllJointsActuationType(parsers.ChParserURDF.ActuationType_POSITION)
        for name in FOOT_BODIES:
            self.parser.SetBodyMeshCollisionType(
                name, parsers.ChParserURDF.MeshCollisionType_CONVEX_HULL)
        # Deliberately NOT "base": the URDF already declares a tight box there,
        # and a hull from trunk.obj engulfs the legs and makes the dog sprawl.
        self.parser.PopulateSystem(chsystem)
        self.parser.GetRootChBody().SetFixed(False)
        self._configure_collision()
        self.motors = [self.parser.GetChMotor(n) for n in MOTOR_NAMES]

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
        c = self.chrono
        for motor, angle in zip(self.motors, chrono_order_angles):
            motor.SetMotorFunction(c.ChFunctionConst(float(angle)))


