"""A parameterised scripted trot, as a drop-in alternative to the imported policy.

WHY. Every one of the 341 collected episodes was driven by go2_cts_150k.pt, which
has exactly one gait. Changing the velocity command changes that gait's speed and
heading, not its structure -- step frequency, duty factor, foot clearance and
phase pattern are whatever the policy learned and are effectively constant across
the whole dataset. So the joint-space region we cover is a ONE-PARAMETER TUBE, and
the joint-level surrogate leaving its training distribution after 1.7 s with 16.9%
of channel-steps unsupported is a direct consequence of that, not a fitting
failure.

Sweeping gait parameters visits the part of joint space the policy never enters.
The intent is to MIX this with policy-driven data, not to replace it.

SAME INTERFACE AS THE POLICY: consumes a body velocity command, emits twelve joint
targets at the control rate, feeds the existing PD in Go2Robot.actuate. Everything
below the joint targets is unchanged, so it is a fair swap.

TWO THINGS THAT ARE MEASURED HERE RATHER THAN ASSUMED, both because the first
attempt at each was wrong:

  GEOMETRY comes from the URDF's JOINT frames, not from distances between body
  origins. Body origins sit at COMs: measuring between them gave thigh 0.2962 and
  calf 0.0982, plausible numbers that cannot reach a foot 0.426 m below the hip.
  The joint frames give 0.2130 and 0.2130, which extends to exactly 0.426.

  JOINT ORDER is MOTOR_NAMES -- RR, RL, FR, FL. That is Chrono order, and it is
  what robot.actuate() consumes. It is NOT FOOT_BODIES (FR FL RR RL), NOT
  LEG_ORDER (fl fr rl rr), and NOT the imported policy's order (FL FR RL RR).
  dataset.py:105 enumerates all four.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .constants import MOTOR_NAMES

# Legs in the order the URDF names them. Independent of every column ordering.
LEGS = ("FR", "FL", "RR", "RL")
# +1 for the left legs, whose abduction offset is +y.
SIDE = {"FR": -1.0, "FL": +1.0, "RR": -1.0, "RL": +1.0}


@dataclass
class LegGeometry:
    """Per-leg kinematics, read from the URDF rather than hardcoded."""

    hip_xy: dict[str, tuple[float, float]]     # hip joint offset from the base LINK frame
    abduction: float                            # hip -> thigh lateral offset, magnitude
    thigh: float
    calf: float
    base_com_offset: np.ndarray                 # base BODY origin - base LINK frame

    @property
    def max_reach(self) -> float:
        return self.thigh + self.calf


def extract_geometry(urdf_path: str | Path) -> LegGeometry:
    """Parse the URDF joint frames. No dynamics, no Chrono system needed.

    The base COM offset DOES need Chrono, because it is a property of how the
    inertial data resolves rather than of the joint tree, so it is measured by
    spawning once at a known frame and differencing. See the module docstring.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(str(urdf_path)).getroot()
    joints = {}
    for j in root.findall("joint"):
        origin = j.find("origin")
        xyz = [float(v) for v in (origin.get("xyz", "0 0 0").split() if origin is not None
                                  else ("0", "0", "0"))]
        joints[j.get("name")] = np.asarray(xyz, dtype=float)

    hip_xy, abduction, thigh, calf = {}, [], [], []
    for leg in LEGS:
        h = joints[f"{leg}_hip_joint"]
        hip_xy[leg] = (float(h[0]), float(h[1]))
        abduction.append(abs(float(joints[f"{leg}_thigh_joint"][1])))
        thigh.append(float(np.linalg.norm(joints[f"{leg}_calf_joint"])))
        calf.append(float(np.linalg.norm(joints[f"{leg}_foot_joint"])))

    # All four legs must agree, or the single-length IK below is wrong for some.
    for name, values in (("abduction", abduction), ("thigh", thigh), ("calf", calf)):
        if max(values) - min(values) > 1e-9:
            raise ValueError(f"{name} differs between legs: {values}. "
                             "The IK assumes one length per segment.")

    return LegGeometry(hip_xy=hip_xy, abduction=abduction[0], thigh=thigh[0],
                       calf=calf[0], base_com_offset=np.zeros(3))


def measure_base_com_offset(chrono: Any, robot: Any, spawn_xyz: tuple[float, float, float]) -> np.ndarray:
    """base BODY origin minus base LINK frame, read at spawn.

    MUST be called before any dynamics. At spawn the link frame IS the spawn
    frame, so the difference is exactly the inertial displacement. Getting this
    wrong put the foot 0.0218 m off on every leg -- identical across all four,
    which is what identified it as a shared constant rather than a per-leg sign
    error.
    """
    p = robot.base().GetPos()
    return np.asarray([p.x - spawn_xyz[0], p.y - spawn_xyz[1], p.z - spawn_xyz[2]], dtype=float)


def forward_kinematics(q_hip: float, q_thigh: float, q_calf: float,
                       side: float, geo: LegGeometry) -> np.ndarray:
    """Foot position in the HIP-JOINT frame. Validated against Chrono to 0.00000 m.

    Hip rotates about +x, thigh and calf about +y, composed in the URDF's own
    parent-child order.
    """
    d = side * geo.abduction
    x = geo.thigh * math.sin(q_thigh) + geo.calf * math.sin(q_thigh + q_calf)
    z_sag = -(geo.thigh * math.cos(q_thigh) + geo.calf * math.cos(q_thigh + q_calf))
    return np.asarray([
        x,
        d * math.cos(q_hip) - z_sag * math.sin(q_hip),
        d * math.sin(q_hip) + z_sag * math.cos(q_hip),
    ])


class Unreachable(ValueError):
    """Raised rather than clamped. A silent clamp looks like a gait bug later."""


def inverse_kinematics(target: np.ndarray, side: float, geo: LegGeometry) -> tuple[float, float, float]:
    """Joint angles putting the foot at `target` in the HIP-JOINT frame.

    Abduction is resolved first, then planar 2-link IK in the resulting sagittal
    plane. The knee bends in the direction STAND_ACTION uses (calf positive).

    RAISES on an unreachable target instead of clamping the acos domain, because
    a clamp produces a plausible pose for an impossible request and the resulting
    gait defect is attributed to tuning.
    """
    px, py, pz = (float(v) for v in target)
    d = side * geo.abduction

    lateral_sq = py * py + pz * pz - d * d
    if lateral_sq < 0.0:
        raise Unreachable(
            f"target {np.round(target, 4).tolist()} is inside the abduction offset: "
            f"y^2+z^2 = {py*py+pz*pz:.6f} < d^2 = {d*d:.6f}")
    z_sag = -math.sqrt(lateral_sq)                       # foot below the hip
    q_hip = math.atan2(pz, py) - math.atan2(z_sag, d)
    q_hip = math.atan2(math.sin(q_hip), math.cos(q_hip))  # wrap

    reach = math.hypot(px, z_sag)
    if reach > geo.max_reach:
        raise Unreachable(
            f"target {np.round(target, 4).tolist()} needs reach {reach:.4f} m, "
            f"limit {geo.max_reach:.4f} m")
    cos_calf = (reach * reach - geo.thigh ** 2 - geo.calf ** 2) / (2.0 * geo.thigh * geo.calf)
    if not -1.0 <= cos_calf <= 1.0:
        # Only reachable by floating-point drift once `reach` has been checked;
        # if it fires for any other reason the geometry is inconsistent.
        if abs(cos_calf) - 1.0 > 1e-9:
            raise Unreachable(f"acos domain violated: cos_calf = {cos_calf:.9f}")
        cos_calf = max(-1.0, min(1.0, cos_calf))
    q_calf = math.acos(cos_calf)                          # positive: knee bends as STAND_ACTION does
    q_thigh = math.atan2(px, -z_sag) - math.atan2(
        geo.calf * math.sin(q_calf), geo.thigh + geo.calf * math.cos(q_calf))
    return q_hip, q_thigh, q_calf


# --- gait ------------------------------------------------------------------

# Phase offsets as a table so other patterns drop in. Trot = diagonal pairs.
GAIT_OFFSETS: dict[str, dict[str, float]] = {
    "trot":  {"FR": 0.0, "FL": 0.5, "RR": 0.5, "RL": 0.0},
    "pace":  {"FR": 0.0, "FL": 0.5, "RR": 0.0, "RL": 0.5},
    "bound": {"FR": 0.0, "FL": 0.0, "RR": 0.5, "RL": 0.5},
    "walk":  {"FR": 0.0, "FL": 0.5, "RR": 0.25, "RL": 0.75},
}


@dataclass
class GaitParams:
    frequency_hz: float = 2.0
    duty: float = 0.6                 # fraction of the cycle in stance
    step_height_m: float = 0.08
    stand_height_m: float = 0.30      # nominal hip-to-foot drop
    pattern: str = "trot"
    # Body-attitude PD modulating leg length. Zero gains = pure open loop.
    kp_roll: float = 0.0
    kd_roll: float = 0.0
    kp_pitch: float = 0.0
    kd_pitch: float = 0.0


def foot_trajectory(phase: float, params: GaitParams,
                    stance_disp: np.ndarray) -> np.ndarray:
    """Foot offset from its nominal stance centre, in the hip frame.

    Stance is a straight line moving backward at the commanded speed. Swing is a
    CYCLOID, chosen because its velocity is continuous at liftoff and touchdown --
    a triangle wave has a velocity step at touchdown, and that impact transient
    would contaminate exactly the contact channels this data is being collected
    for.
    """
    phase = phase % 1.0
    d = params.duty
    if phase < d:
        # stance: sweep from +half the displacement to -half, linearly
        s = phase / d
        return np.asarray([*(stance_disp * (0.5 - s)), 0.0])
    # swing: cycloid back to the front, lifting to step_height
    u = (phase - d) / (1.0 - d)
    horiz = stance_disp * (u - math.sin(2.0 * math.pi * u) / (2.0 * math.pi) - 0.5)
    vert = params.step_height_m * (1.0 - math.cos(2.0 * math.pi * u)) / 2.0
    return np.asarray([*horiz, vert])


class TrotController:
    """Body velocity command -> twelve joint targets in MOTOR_NAMES order."""

    def __init__(self, geo: LegGeometry, params: GaitParams | None = None):
        self.geo = geo
        self.params = params or GaitParams()
        self.phase = 0.0
        self.command = np.zeros(3)          # vx, vy, wz
        self._motor_index = {name.removesuffix("_joint"): i for i, name in enumerate(MOTOR_NAMES)}
        self.unreachable_count = 0

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        self.command = np.asarray([vx, vy, wz], dtype=float)

    def advance(self, dt: float) -> None:
        self.phase = (self.phase + self.params.frequency_hz * dt) % 1.0

    def _stance_displacement(self, leg: str) -> np.ndarray:
        """How far this foot travels backward during one stance, in the hip frame.

        The yaw term adds a tangential component w x r taken from the foot's OWN
        hip position, so turning sweeps the outer feet further than the inner.
        """
        vx, vy, wz = self.command
        hx, hy = self.geo.hip_xy[leg]
        vx_leg = vx - wz * hy
        vy_leg = vy + wz * hx
        stance_time = self.params.duty / self.params.frequency_hz
        return np.asarray([vx_leg, vy_leg]) * stance_time

    def joint_targets(self, roll: float = 0.0, pitch: float = 0.0,
                      roll_rate: float = 0.0, pitch_rate: float = 0.0) -> np.ndarray:
        """Twelve targets in MOTOR_NAMES order (RR, RL, FR, FL)."""
        out = np.zeros(12, dtype=float)
        p = self.params
        for leg in LEGS:
            offsets = GAIT_OFFSETS[p.pattern]
            phase = (self.phase + offsets[leg]) % 1.0
            disp = self._stance_displacement(leg)
            delta = foot_trajectory(phase, p, disp)

            # Virtual-model balance: lengthen the legs on the side the body is
            # falling toward. Zero gains leave this exactly open-loop.
            hx, hy = self.geo.hip_xy[leg]
            correction = (p.kp_roll * roll + p.kd_roll * roll_rate) * np.sign(hy or 1.0) \
                + (p.kp_pitch * pitch + p.kd_pitch * pitch_rate) * np.sign(hx or 1.0)

            target = np.asarray([
                delta[0],
                SIDE[leg] * self.geo.abduction + delta[1],
                -(p.stand_height_m - delta[2]) + correction,
            ])
            try:
                qh, qt, qc = inverse_kinematics(target, SIDE[leg], self.geo)
            except Unreachable:
                self.unreachable_count += 1
                qh, qt, qc = inverse_kinematics(
                    np.asarray([target[0], target[1],
                                -min(self.geo.max_reach * 0.98,
                                     abs(target[2]))]), SIDE[leg], self.geo)
            out[self._motor_index[f"{leg}_hip"]] = qh
            out[self._motor_index[f"{leg}_thigh"]] = qt
            out[self._motor_index[f"{leg}_calf"]] = qc
        return out
