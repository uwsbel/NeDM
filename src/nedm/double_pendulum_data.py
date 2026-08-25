"""Chrono double-pendulum + camera data collection for NRD Study 1.

First NRD (vision-in-the-loop) study case, per ``docs/vision/double_pen/
NRD_double_pendulum_study_plan.md``: an actuated planar double pendulum whose
physical state z1 and synchronized Chrono::Sensor camera frame x are recorded
together, so a joint dynamics model can learn to predict both.

Mechanism (plan section 3.2): fixed ground, two slender links in the world X-Z
plane, a passive revolute shoulder (ground -> link1), a torque-motor elbow
(link1 -> link2), light viscous damping at both joints via ChLinkRSDA. Contact
is disabled. The single normalized action a in [-1, 1] maps to elbow torque
a * TAU_MAX held over one control period.

State (plan section 4): z1 = [cos q1, sin q1, cos q2, sin q2, omega1, omega2],
with q1 the link-1 angle from the downward (-Z) direction, q2 the link-2 angle
relative to link 1, and omega1/omega2 their generalized rates.

Camera (plan section 5.1): fixed world camera on the -Y axis looking at the
mechanism plane, RGB 128x128 at 50 Hz -- one frame per control boundary. Frames
are associated to control rows by the sensor buffer's TimeStamp, never by
"most recent buffer" luck, and the collector hard-fails on any mismatch. The
tip of link 2 carries a yellow marker sphere so an automated pixel-space
alignment test (validate_dataset) can project the recorded state through the
pinhole model and compare against the rendered blob.

Episode scheduling: ONE Chrono system + sensor manager serve all episodes in a
process (repeated scene/OptiX re-creation is the known stack-smash footgun, see
memory chrono-eval-multiref-stack-smash). Between episodes the body states are
reset analytically exactly on a control boundary, immediately before that
boundary's sensor update, so row 0 of every episode is the sampled initial
condition rendered exactly. One control period between episodes is discarded.

Run in the NeDM conda env:

    conda run -n nedm python -m nedm.double_pendulum_data \
        --episodes 10 --max-steps 100 --output-root artifacts/datasets/dpend_smoke

Outputs the standard NeDM raw-dataset layout: ``episodes/{id}.csv`` (one row
per control boundary), ``episodes/{id}_frames.npy`` (uint8, rows x 128 x 128 x 3,
frame i belongs to CSV row i), ``episodes/{id}.json`` and ``dataset_index.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pychrono as chrono
import pychrono.sensor as sens

# ---------------------------------------------------------------------------
# Versioned mechanism parameters (plan section 3.2)
# ---------------------------------------------------------------------------
LINK1_LENGTH_M = 0.3
LINK2_LENGTH_M = 0.3
LINK1_MASS_KG = 0.5
LINK2_MASS_KG = 0.5
LINK_SIDE_M = 0.04           # square cross-section of the slender box links
# Viscous damping at BOTH joints (ChLinkRSDA). Pilot-tuned: at 0.02 the elbow
# motor pumps most 10 s episodes past the spin guard; at 0.06 ~75% run full
# length while the motion stays chaotic and lively.
JOINT_DAMPING_NMS = 0.06
TAU_MAX_NM = 1.5             # elbow torque at |a| = 1
GRAVITY_MPS2 = 9.81

# Timing (plan section 3.2): physics substep, control/record period, camera rate.
DT_SIM_S = 1e-3
CONTROL_DT_S = 0.02
CAMERA_HZ = 1.0 / CONTROL_DT_S
SUBSTEPS_PER_CONTROL = int(round(CONTROL_DT_S / DT_SIM_S))

# Camera (plan section 5.1): fixed pose on -Y looking +Y, image right = +X,
# image up = +Z, pinhole with horizontal FOV below.
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
CAMERA_DISTANCE_M = 2.0
CAMERA_HFOV_RAD = 0.72
BACKGROUND_RGB = (0.08, 0.08, 0.10)
LINK1_RGB = (0.85, 0.15, 0.10)
LINK2_RGB = (0.10, 0.35, 0.90)
TIP_RGB = (0.95, 0.85, 0.05)
ELBOW_RGB = (0.90, 0.90, 0.90)
TIP_MARKER_RADIUS_M = 0.035
ELBOW_MARKER_RADIUS_M = 0.025

# Episode safety guard: runaway spin ends the episode (recorded, then truncated).
# 35 rad/s = 0.7 rad of motion between 50 Hz frames; beyond that the camera
# aliases badly and the regime is outside the study's interest anyway.
OMEGA_LIMIT_RADPS = 35.0

STATE_FIELDS = ["cos_q1", "sin_q1", "cos_q2", "sin_q2", "omega1_radps", "omega2_radps"]
ACTION_FIELDS = ["action_elbow"]
ROLLOUT_FIELDS = ["tip_x_m", "tip_z_m"]

CSV_HEADER = [
    "episode_id", "split", "sample_index", "time_s",
    "cos_q1", "sin_q1", "cos_q2", "sin_q2", "omega1_radps", "omega2_radps",
    "action_elbow",
    "tip_x_m", "tip_z_m", "elbow_x_m", "elbow_z_m",
    "q1_rad", "q2_rad", "applied_torque_nm", "cam_time_s", "out_of_plane_m",
]


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@dataclass
class PendulumScene:
    system: chrono.ChSystemNSC
    ground: chrono.ChBody
    link1: chrono.ChBody
    link2: chrono.ChBody
    elbow_torque: chrono.ChFunctionSetpoint
    manager: "sens.ChSensorManager | None"
    camera: "sens.ChCameraSensor | None"


def _link_body(length_m: float, mass_kg: float, rgb: tuple[float, float, float]) -> chrono.ChBody:
    density = mass_kg / (LINK_SIDE_M * LINK_SIDE_M * length_m)
    body = chrono.ChBodyEasyBox(LINK_SIDE_M, LINK_SIDE_M, length_m, density, True, False)
    material = chrono.ChVisualMaterial()
    material.SetDiffuseColor(chrono.ChColor(*rgb))
    body.GetVisualShape(0).SetMaterial(0, material)
    return body


def _add_marker_sphere(body: chrono.ChBody, local_z: float, radius: float,
                       rgb: tuple[float, float, float]) -> None:
    sphere = chrono.ChVisualShapeSphere(radius)
    material = chrono.ChVisualMaterial()
    material.SetDiffuseColor(chrono.ChColor(*rgb))
    sphere.SetMaterial(0, material)
    body.AddVisualShape(sphere, chrono.ChFramed(chrono.ChVector3d(0, 0, local_z), chrono.QUNIT))


def build_scene(with_camera: bool = True) -> PendulumScene:
    system = chrono.ChSystemNSC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -GRAVITY_MPS2))
    system.SetNumThreads(1)

    ground = chrono.ChBody()
    ground.SetFixed(True)
    ground.EnableCollision(False)
    system.Add(ground)
    # Small gray sphere marks the fixed pivot in the image.
    _add_marker_sphere(ground, 0.0, 0.02, (0.6, 0.6, 0.6))

    link1 = _link_body(LINK1_LENGTH_M, LINK1_MASS_KG, LINK1_RGB)
    link2 = _link_body(LINK2_LENGTH_M, LINK2_MASS_KG, LINK2_RGB)
    system.Add(link1)
    system.Add(link2)
    # Elbow marker on link1's lower end, tip marker on link2's lower end.
    _add_marker_sphere(link1, -LINK1_LENGTH_M / 2.0, ELBOW_MARKER_RADIUS_M, ELBOW_RGB)
    _add_marker_sphere(link2, -LINK2_LENGTH_M / 2.0, TIP_MARKER_RADIUS_M, TIP_RGB)

    # Place at q1 = q2 = 0 (hanging straight down) so the joints can be
    # initialized at their world locations; reset_state() moves things after.
    link1.SetPos(chrono.ChVector3d(0, 0, -LINK1_LENGTH_M / 2.0))
    link2.SetPos(chrono.ChVector3d(0, 0, -LINK1_LENGTH_M - LINK2_LENGTH_M / 2.0))

    # Revolute axes along world Y: joint frame Z must map to world Y.
    hinge_rot = chrono.QuatFromAngleX(-math.pi / 2.0)
    shoulder_frame = chrono.ChFramed(chrono.ChVector3d(0, 0, 0), hinge_rot)
    elbow_frame = chrono.ChFramed(chrono.ChVector3d(0, 0, -LINK1_LENGTH_M), hinge_rot)

    shoulder = chrono.ChLinkLockRevolute()
    shoulder.Initialize(link1, ground, shoulder_frame)
    system.Add(shoulder)

    elbow = chrono.ChLinkMotorRotationTorque()
    elbow.Initialize(link2, link1, elbow_frame)
    elbow_torque = chrono.ChFunctionSetpoint()
    elbow.SetTorqueFunction(elbow_torque)
    system.Add(elbow)

    # Viscous joint damping at both joints (spring coefficient 0).
    for body_a, body_b, frame in ((link1, ground, shoulder_frame), (link2, link1, elbow_frame)):
        rsda = chrono.ChLinkRSDA()
        rsda.Initialize(body_a, body_b, frame)
        rsda.SetSpringCoefficient(0.0)
        rsda.SetDampingCoefficient(JOINT_DAMPING_NMS)
        system.Add(rsda)

    manager = None
    camera = None
    if with_camera:
        manager = sens.ChSensorManager(system)
        manager.scene.AddPointLight(chrono.ChVector3f(-1.0, -2.5, 1.5), chrono.ChColor(1.0, 1.0, 1.0), 6.0)
        manager.scene.AddPointLight(chrono.ChVector3f(1.0, -2.5, -1.0), chrono.ChColor(0.7, 0.7, 0.7), 6.0)
        try:
            background = sens.Background()
            background.mode = sens.BackgroundMode_SOLID_COLOR
            background.color_zenith = chrono.ChVector3f(*BACKGROUND_RGB)
            manager.scene.SetBackground(background)
        except (AttributeError, TypeError):
            pass  # keep the default background; it is versioned by chrono version
        camera_pose = chrono.ChFramed(
            chrono.ChVector3d(0, -CAMERA_DISTANCE_M, 0), chrono.QuatFromAngleZ(math.pi / 2.0)
        )
        # Nominal rate = one substep so the internal schedule is always behind the
        # sim clock and every manager.Update() fires exactly one render (see the
        # FrameTap comment). The EFFECTIVE frame rate is CAMERA_HZ because Update
        # is only called at control boundaries.
        trigger_rate_hz = 1.0 / DT_SIM_S
        camera = sens.ChCameraSensor(ground, trigger_rate_hz, camera_pose, IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_HFOV_RAD)
        camera.SetName("dpend_camera")
        # Instantaneous shutter, no lag: the default lag is one full period, which
        # would hold frame k's data back until the next boundary and deadlock the
        # blocking reader. Collection window 0 also disables motion blur so a frame
        # is an exact snapshot of the boundary state.
        camera.SetLag(0.0)
        camera.SetCollectionWindow(0.0)
        camera.PushFilter(sens.ChFilterRGBA8Access())
        manager.AddSensor(camera)

    return PendulumScene(
        system=system, ground=ground, link1=link1, link2=link2,
        elbow_torque=elbow_torque, manager=manager, camera=camera,
    )


# ---------------------------------------------------------------------------
# State reset / readout
# ---------------------------------------------------------------------------
def reset_state(scene: PendulumScene, q1: float, q2: float, w1: float, w2: float) -> None:
    """Set link poses/velocities exactly consistent with (q1, q2, w1, w2).

    q1: link-1 angle from the downward -Z direction (positive tips toward +X).
    q2: link-2 angle relative to link 1. w1 = dq1/dt, w2 = dq2/dt.
    Body rotation about +Y is -q for this convention (see angle readout).
    """
    q2_abs = q1 + q2
    w2_abs = w1 + w2
    dir1 = np.array([math.sin(q1), 0.0, -math.cos(q1)])
    dir2 = np.array([math.sin(q2_abs), 0.0, -math.cos(q2_abs)])
    tan1 = np.array([math.cos(q1), 0.0, math.sin(q1)])
    tan2 = np.array([math.cos(q2_abs), 0.0, math.sin(q2_abs)])

    com1 = dir1 * (LINK1_LENGTH_M / 2.0)
    elbow = dir1 * LINK1_LENGTH_M
    com2 = elbow + dir2 * (LINK2_LENGTH_M / 2.0)
    vel_com1 = tan1 * (w1 * LINK1_LENGTH_M / 2.0)
    vel_elbow = tan1 * (w1 * LINK1_LENGTH_M)
    vel_com2 = vel_elbow + tan2 * (w2_abs * LINK2_LENGTH_M / 2.0)

    scene.link1.SetPos(chrono.ChVector3d(*com1))
    scene.link1.SetRot(chrono.QuatFromAngleY(-q1))
    scene.link1.SetPosDt(chrono.ChVector3d(*vel_com1))
    scene.link1.SetAngVelParent(chrono.ChVector3d(0.0, -w1, 0.0))

    scene.link2.SetPos(chrono.ChVector3d(*com2))
    scene.link2.SetRot(chrono.QuatFromAngleY(-q2_abs))
    scene.link2.SetPosDt(chrono.ChVector3d(*vel_com2))
    scene.link2.SetAngVelParent(chrono.ChVector3d(0.0, -w2_abs, 0.0))

    # Full refresh after teleporting the bodies. Without this, stale cached
    # assembly state makes the next step depend on the PREVIOUS episode's motion
    # (measured: up to 0.47 rad/s one-step deviation); with it, one-step replay
    # is bitwise deterministic.
    scene.system.Setup()
    scene.system.Update()


def read_state(scene: PendulumScene) -> dict[str, float]:
    """Extract z1, marker world positions, and raw angles from the bodies."""
    down1 = scene.link1.GetRot().Rotate(chrono.ChVector3d(0, 0, -1))
    down2 = scene.link2.GetRot().Rotate(chrono.ChVector3d(0, 0, -1))
    q1 = math.atan2(down1.x, -down1.z)
    q2_abs = math.atan2(down2.x, -down2.z)
    q2 = wrap_angle(q2_abs - q1)
    w1 = -scene.link1.GetAngVelParent().y
    w2_abs = -scene.link2.GetAngVelParent().y
    w2 = w2_abs - w1

    elbow_world = scene.link1.TransformPointLocalToParent(
        chrono.ChVector3d(0, 0, -LINK1_LENGTH_M / 2.0)
    )
    tip_world = scene.link2.TransformPointLocalToParent(
        chrono.ChVector3d(0, 0, -LINK2_LENGTH_M / 2.0)
    )
    return {
        "q1_rad": q1,
        "q2_rad": q2,
        "cos_q1": math.cos(q1),
        "sin_q1": math.sin(q1),
        "cos_q2": math.cos(q2),
        "sin_q2": math.sin(q2),
        "omega1_radps": w1,
        "omega2_radps": w2,
        "elbow_x_m": elbow_world.x,
        "elbow_z_m": elbow_world.z,
        "tip_x_m": tip_world.x,
        "tip_z_m": tip_world.z,
        "out_of_plane_m": max(abs(scene.link1.GetPos().y), abs(scene.link2.GetPos().y)),
    }


def forward_kinematics(q1: float, q2: float) -> tuple[float, float, float, float]:
    """(elbow_x, elbow_z, tip_x, tip_z) from joint angles and fixed geometry."""
    q2_abs = q1 + q2
    elbow_x = math.sin(q1) * LINK1_LENGTH_M
    elbow_z = -math.cos(q1) * LINK1_LENGTH_M
    tip_x = elbow_x + math.sin(q2_abs) * LINK2_LENGTH_M
    tip_z = elbow_z - math.cos(q2_abs) * LINK2_LENGTH_M
    return elbow_x, elbow_z, tip_x, tip_z


def find_tip_pixel(frame: np.ndarray) -> tuple[float, float] | None:
    """Centroid of the yellow tip-marker blob in an RGB frame; None if absent.

    The mask must exclude brightly lit pixels of the red link (whose green
    channel rises under white light) -- hence the G > 140 requirement.
    """
    red = frame[:, :, 0].astype(np.int32)
    green = frame[:, :, 1].astype(np.int32)
    blue = frame[:, :, 2].astype(np.int32)
    mask = (red > 170) & (green > 140) & (blue < 110)
    if mask.sum() < 3:
        return None
    vs, us = np.nonzero(mask)
    return float(us.mean()), float(vs.mean())


def project_to_pixel(x_m: float, z_m: float) -> tuple[float, float]:
    """Pinhole projection of a point in the pendulum plane (y=0) to pixels.

    Camera at (0, -CAMERA_DISTANCE_M, 0) looking +Y; image right = +X world,
    image up = +Z world.
    """
    focal_px = (IMAGE_WIDTH / 2.0) / math.tan(CAMERA_HFOV_RAD / 2.0)
    u = IMAGE_WIDTH / 2.0 + focal_px * (x_m / CAMERA_DISTANCE_M)
    v = IMAGE_HEIGHT / 2.0 - focal_px * (z_m / CAMERA_DISTANCE_M)
    return u, v


# ---------------------------------------------------------------------------
# Camera frame collection: manual-trigger pattern.
#
# Chrono::Sensor's own launch scheduler compares float32 accumulations of
# k / update_rate against the sim clock, and around some boundaries the launch
# slips one substep late -- a 1 ms content error and (for a blocking reader) a
# deadlock. We bypass the scheduler entirely: the camera's nominal period is one
# physics substep, so its schedule is always behind the clock and EVERY
# ChSensorManager.Update() call fires exactly one render of the current state
# (verified: no catch-up loop). Update() is called only at control boundaries,
# and frames are associated to rows by LaunchedCount, not timestamps.
# ---------------------------------------------------------------------------
class FrameTap:
    """Consumes exactly one rendered frame per manager.Update() trigger."""

    def __init__(self, camera: "sens.ChCameraSensor") -> None:
        self.camera = camera
        self.taken_count = 0

    def skip(self) -> None:
        """Mark the next completed launch as consumed without reading it."""
        self.taken_count += 1

    def take(self, timeout_s: float = 10.0) -> np.ndarray:
        """The frame from the launch triggered by the latest manager.Update()."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            buffer = self.camera.GetMostRecentRGBA8Buffer()
            if buffer.HasData() and buffer.LaunchedCount > self.taken_count:
                if buffer.LaunchedCount != self.taken_count + 1:
                    raise RuntimeError(
                        f"camera frames were skipped: expected launch {self.taken_count + 1}, "
                        f"got {buffer.LaunchedCount}"
                    )
                self.taken_count = buffer.LaunchedCount
                rgba = buffer.GetRGBA8Data()  # (H, W, 4) uint8, bottom-up per OptiX
                return np.ascontiguousarray(rgba[::-1, :, :3])
            time.sleep(0.0005)
        raise RuntimeError(f"camera frame for launch {self.taken_count + 1} never arrived")


# ---------------------------------------------------------------------------
# Action scenarios (plan section 6.2)
# ---------------------------------------------------------------------------
SCENARIO_FAMILIES = ("unforced", "piecewise", "smooth")
SCENARIO_WEIGHTS = (0.30, 0.50, 0.20)


def sample_action_sequence(rng: np.random.Generator, family: str, num_steps: int) -> np.ndarray:
    if family == "unforced":
        return np.zeros(num_steps, dtype=np.float64)
    if family == "piecewise":
        actions = np.empty(num_steps, dtype=np.float64)
        index = 0
        while index < num_steps:
            dwell_steps = int(round(rng.uniform(0.1, 0.5) / CONTROL_DT_S))
            actions[index : index + dwell_steps] = rng.uniform(-1.0, 1.0)
            index += dwell_steps
        return actions
    if family == "smooth":
        t = np.arange(num_steps) * CONTROL_DT_S
        amplitude = rng.uniform(0.3, 1.0)
        if rng.random() < 0.5:  # sinusoid
            freq_hz = rng.uniform(0.2, 1.5)
            phase = rng.uniform(0.0, 2.0 * math.pi)
            actions = amplitude * np.sin(2.0 * math.pi * freq_hz * t + phase)
        else:  # linear chirp
            f0 = rng.uniform(0.1, 0.6)
            f1 = rng.uniform(0.8, 2.0)
            duration = max(t[-1], CONTROL_DT_S)
            actions = amplitude * np.sin(2.0 * math.pi * (f0 * t + (f1 - f0) * t * t / (2.0 * duration)))
        return np.clip(actions, -1.0, 1.0)
    raise ValueError(f"unknown scenario family {family!r}")


def sample_initial_condition(rng: np.random.Generator) -> tuple[float, float, float, float]:
    q1 = rng.uniform(-math.pi, math.pi)
    q2 = rng.uniform(-math.pi, math.pi)
    # Pilot-validated bounds (plan 6.2): with the earlier +-6/+-8 range most
    # episodes reached the spin guard within 1-2 s from gravity alone.
    w1 = rng.uniform(-4.0, 4.0)
    w2 = rng.uniform(-6.0, 6.0)
    return q1, q2, w1, w2


def assign_split(episode_id: str, validation_ratio: float) -> str:
    digest = hashlib.sha1(episode_id.encode("utf-8")).hexdigest()
    return "val" if (int(digest[:8], 16) / 0xFFFFFFFF) < validation_ratio else "train"


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    episode_id: str
    split: str
    scenario_family: str
    rows: int
    truncated_spin: bool
    csv_path: Path
    frames_path: Path | None


def _advance_to_next_boundary(
    scene: PendulumScene, tap: FrameTap | None = None, dt_sim_s: float = DT_SIM_S
) -> None:
    """Step exactly one control period. No sensor updates: renders are triggered
    only by explicit manager.Update() calls at boundaries (manual-trigger pattern).
    The ``tap`` argument is accepted for call-site compatibility and unused."""
    for _ in range(int(round(CONTROL_DT_S / dt_sim_s))):
        scene.system.DoStepDynamics(dt_sim_s)


def run_episode(
    scene: PendulumScene,
    tap: FrameTap | None,
    episode_id: str,
    split: str,
    family: str,
    actions: np.ndarray,
    initial_condition: tuple[float, float, float, float],
    output_root: Path | None,
) -> tuple[EpisodeResult, np.ndarray, np.ndarray]:
    """Run one episode; the system clock is already ON a control boundary.

    Sequence: reset state exactly at the current boundary (before its sensor
    update), record row 0, then for each action step one control period and
    record the next row. Returns (result, state_rows, frames).
    """
    num_steps = actions.shape[0]
    scene.elbow_torque.SetSetpoint(0.0, scene.system.GetChTime())
    reset_state(scene, *initial_condition)

    rows: list[dict[str, float]] = []
    frames: list[np.ndarray] = []
    truncated_spin = False

    def record_row(sample_index: int, action_value: float, applied_torque: float) -> dict[str, float]:
        now = scene.system.GetChTime()
        state = read_state(scene)
        if tap is not None:
            frame = tap.take()
            frames.append(frame)
            state["cam_time_s"] = now
        else:
            state["cam_time_s"] = float("nan")
        state.update(
            {
                "episode_id": episode_id,
                "split": split,
                "sample_index": sample_index,
                "time_s": now,
                "action_elbow": action_value,
                "applied_torque_nm": applied_torque,
            }
        )
        rows.append(state)
        return state

    for sample_index in range(num_steps + 1):
        if sample_index < num_steps:
            action_value = float(actions[sample_index])
        else:
            action_value = float(actions[-1])  # final row repeats last action (never used as input)
        torque = action_value * TAU_MAX_NM
        scene.elbow_torque.SetSetpoint(torque, scene.system.GetChTime())

        if tap is not None:
            # Manual trigger: this Update fires exactly one render of the CURRENT
            # state -- for row 0 that is the freshly reset initial condition.
            scene.manager.Update()
        row = record_row(sample_index, action_value, torque)
        if abs(row["omega1_radps"]) > OMEGA_LIMIT_RADPS or abs(row["omega2_radps"]) > OMEGA_LIMIT_RADPS:
            truncated_spin = True
            break
        if sample_index == num_steps:
            break
        _advance_to_next_boundary(scene)

    # One quiet control period between episodes (no renders are triggered).
    scene.elbow_torque.SetSetpoint(0.0, scene.system.GetChTime())
    _advance_to_next_boundary(scene)

    state_rows = np.array(
        [[row[field] for field in CSV_HEADER[3:]] for row in rows], dtype=np.float64
    )
    frame_array = np.stack(frames, axis=0) if frames else np.zeros((0, IMAGE_HEIGHT, IMAGE_WIDTH, 3), np.uint8)

    csv_path = None
    frames_path = None
    if output_root is not None:
        episodes_dir = output_root / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        csv_path = episodes_dir / f"{episode_id}.csv"
        with csv_path.open("w", encoding="utf-8") as handle:
            handle.write(",".join(CSV_HEADER) + "\n")
            for row in rows:
                values = [str(row["episode_id"]), str(row["split"]), str(int(row["sample_index"]))]
                values += [f"{float(row[field]):.9g}" for field in CSV_HEADER[3:]]
                handle.write(",".join(values) + "\n")
        if tap is not None:
            frames_path = episodes_dir / f"{episode_id}_frames.npy"
            np.save(frames_path, frame_array)

    result = EpisodeResult(
        episode_id=episode_id,
        split=split,
        scenario_family=family,
        rows=len(rows),
        truncated_spin=truncated_spin,
        csv_path=csv_path if csv_path is not None else Path(""),
        frames_path=frames_path,
    )
    return result, state_rows, frame_array


# ---------------------------------------------------------------------------
# Dataset collection entry point
# ---------------------------------------------------------------------------
def dataset_config() -> dict[str, object]:
    return {
        "control_dt_s": CONTROL_DT_S,
        "dt_sim_s": DT_SIM_S,
        "camera_hz": CAMERA_HZ,
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "camera_distance_m": CAMERA_DISTANCE_M,
        "camera_hfov_rad": CAMERA_HFOV_RAD,
        "background_rgb": list(BACKGROUND_RGB),
        "link1_length_m": LINK1_LENGTH_M,
        "link2_length_m": LINK2_LENGTH_M,
        "link1_mass_kg": LINK1_MASS_KG,
        "link2_mass_kg": LINK2_MASS_KG,
        "link_side_m": LINK_SIDE_M,
        "joint_damping_nms": JOINT_DAMPING_NMS,
        "tau_max_nm": TAU_MAX_NM,
        "gravity_mps2": GRAVITY_MPS2,
        "omega_limit_radps": OMEGA_LIMIT_RADPS,
        "state_fields": STATE_FIELDS,
        "action_fields": ACTION_FIELDS,
        "rollout_fields": ROLLOUT_FIELDS,
        "chrono_version": getattr(chrono, "CHRONO_VERSION", "unknown"),
    }


def collect(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    scene = build_scene(with_camera=not args.no_camera)
    tap = FrameTap(scene.camera) if scene.camera is not None else None
    # One quiet period so every episode (including the first) starts the same way.
    _advance_to_next_boundary(scene)

    results: list[EpisodeResult] = []
    started = time.monotonic()
    for episode_index in range(args.episodes):
        episode_id = f"{args.episode_prefix}_{episode_index:04d}"
        split = assign_split(episode_id, args.validation_ratio)
        family = str(rng.choice(SCENARIO_FAMILIES, p=SCENARIO_WEIGHTS))
        actions = sample_action_sequence(rng, family, args.max_steps)
        initial_condition = sample_initial_condition(rng)
        result, state_rows, _ = run_episode(
            scene, tap, episode_id, split, family, actions, initial_condition, output_root
        )
        results.append(result)
        (output_root / "episodes" / f"{episode_id}.json").write_text(
            json.dumps(
                {
                    "episode_id": episode_id,
                    "split": split,
                    "scenario_family": family,
                    "rows": result.rows,
                    "truncated_spin": result.truncated_spin,
                    "initial_condition": list(initial_condition),
                    "seed": args.seed,
                },
                indent=2,
            )
        )
        flag = "SPIN-TRUNCATED" if result.truncated_spin else "ok"
        print(
            f"  {episode_id}: {result.rows} rows, {flag}, split={result.split}, family={family}",
            flush=True,
        )

    elapsed = time.monotonic() - started
    sim_time = sum(r.rows - 1 for r in results) * CONTROL_DT_S
    index = {
        "dataset_name": args.dataset_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": dataset_config(),
        "collection": {
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "validation_ratio": args.validation_ratio,
            "wall_time_s": elapsed,
            "realtime_factor": sim_time / elapsed if elapsed > 0 else float("nan"),
        },
        "episodes": [
            {
                "episode_id": r.episode_id,
                "split": r.split,
                "scenario_family": r.scenario_family,
                "rows": r.rows,
                "truncated_spin": r.truncated_spin,
                "csv_path": str(r.csv_path.relative_to(output_root)),
                "frames_path": str(r.frames_path.relative_to(output_root)) if r.frames_path else None,
            }
            for r in results
        ],
    }
    (output_root / "dataset_index.json").write_text(json.dumps(index, indent=2))
    print(
        f"collected {args.episodes} episodes ({sum(r.rows for r in results)} rows) "
        f"in {elapsed:.1f}s (RTF {index['collection']['realtime_factor']:.2f}x) -> {output_root}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Chrono double-pendulum + camera transitions.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100, help="Control steps per episode (rows = steps + 1).")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--episode-prefix", type=str, default="dpend")
    parser.add_argument("--dataset-name", type=str, default="dpend_v1")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/datasets/dpend_smoke"))
    parser.add_argument("--no-camera", action="store_true", help="Physics-only collection (no frames).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    collect(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
