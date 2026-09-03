#!/usr/bin/env python
"""Unitree Go2 walking on CRM deformable terrain, headless, with video.

The second half of the WP0 gate for the proposed quadruped case study
(docs/state/progress/future-case-studies.md), which prescribes "a privileged
scripted gait walking on rigid ground, THEN CRM, with zero learning, before any
model work". `quadruped_wp0_gait.py` did the rigid half on RoboSimian. This does
the CRM half on the robot the study actually targets.

WHY GO2 RATHER THAN THE RoboSimian PROTOTYPE. The plan ranks the bootstrapping
problem as the study's real risk, and ranks "import a pretrained Go2 policy" as
option 3, "highest risk, keep off the critical path". That ranking is stale:
`uwsbel/sbel-reproducibility` 2025/multi-terrain-RL already trained a Go2
locomotion policy in Chrono on rigid ground and finetuned it on CRM granular
terrain. `model_2999.pt` is the CRM-finetuned checkpoint. So the seed-controller
problem is solved in-house, and RoboSimian's only stated purpose, shaking out
CRM foot-contact machinery, is served better by the target robot itself.

This is a PORT, not a reuse. That work runs `bochengzou::pychrono`; everything
here runs the `nedm` environment this repo specifies (`projectchrono` 10.0.0).
The observation convention is taken verbatim from `chrono_crmenv.py`, which is
authoritative because it ships with the checkpoint.

FOUR CONVENTIONS THAT SILENTLY BREAK THIS IF MISSED, all verified on kyle-sbel:

1. Joints are NOT actuated by default. `SetAllJointsActuationType` must be
   called BEFORE `PopulateSystem`. Without it `GetChMotor` returns a wrapped
   null pointer that is not None, reports a plausible type, and kills the
   interpreter with no traceback when touched. `if motor is not None` is not a
   valid success test here.
2. Joint positions and velocities are NEGATED into the policy's frame. This is
   a real sign-convention difference, not a reordering artifact.
3. Chrono orders joints [RR, RL, FR, FL]; the policy expects [FR, FL, RR, RL].
   The map is an involution, so the same array converts both ways.
4. The 3-wide command slot is a HARDCODED [0.5, 0, 0] * lin_vel_scale. It is
   NOT `env_cfg['target_lin_vel']`, which has two elements and is used only for
   reward. Wiring the config value in here produces a subtly wrong observation.

Requires the `nedm` env AND an OptiX-capable driver (R590+) for --video; see
docs/state/lessons/chrono-versions.md. CRM alone needs no OptiX.

Usage:
  "$NEDM_PY" scripts/quadruped_go2_crm.py --sim-seconds 3 --out artifacts/go2_crm
  "$NEDM_PY" scripts/quadruped_go2_crm.py --sim-seconds 6 --video --out artifacts/go2_crm_vid
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nedm import chrono_crm_compat as crm_compat  # noqa: E402

GRAVITY = 9.81

# Chrono joint order. The policy does not use this order; see CHRONO_TO_GENESIS.
MOTOR_NAMES = [
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
]
FOOT_BODIES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
CALF_BODIES = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]

# Chrono [RR,RL,FR,FL] -> policy [FR,FL,RR,RL]. Swapping two halves of six is
# its own inverse, so this converts observations one way and actions the other.
CHRONO_TO_GENESIS = np.array([6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5], dtype=np.int64)

# Policy-frame rest pose, in policy order [FR, FL, RR, RL]. Front and rear thigh
# defaults differ (0.8 vs 1.0); normalising that away breaks the stance.
GENESIS_DEFAULTS = np.array([0.0, 0.8, -1.5, 0.0, 0.8, -1.5,
                             0.0, 1.0, -1.5, 0.0, 1.0, -1.5], dtype=np.float32)

# Chrono-order standing pose, held while the robot settles onto the soil.
STAND_ACTION = np.array([0.0, -1.0, 1.5, 0.0, -1.0, 1.5,
                         0.0, -0.8, 1.5, 0.0, -0.8, 1.5], dtype=np.float64)

# ~63 deg from upright. A walking Go2 stays well inside this; a tumble
# crosses it decisively, so it separates gait roll from falling over.
FALL_TILT_RAD = math.radians(63.0)

LIN_VEL_SCALE, ANG_VEL_SCALE, DOF_POS_SCALE, DOF_VEL_SCALE = 2.0, 0.25, 1.0, 0.05

# Two soil presets, and which one you pick is a research decision.
#
# "eval" matches configs/hmmwv_crm_eval.json and demo_ROBOT_Viper_CRM.py. It is
# the soil the HMMWV work uses, and it is what this script ran until it was found
# to put a landing body into an undamped limit cycle.
#
# "training" is what chrono_crmenv.py actually used for the CRM policy finetune,
# with Young's modulus HALVED and cohesion cut to 40%, both commented "Reduced"
# in the source. Whoever wrote it evidently hit the same wall and solved it by
# softening the soil rather than by raising artificial dissipation. That is the
# physically honest fix: Young's modulus and cohesion are soil properties, and
# artificial_viscosity is a numerical damping term that changes the foot-soil
# interaction Case Study IV exists to measure.
# WORKING COMBINATION, measured: soil "training" AND artificial_viscosity 2.0.
# Neither alone is sufficient and the pair is better than either, because they
# fix different halves of the problem. Standing, 8 s, depth 0.2:
#   eval soil     + av 0.5 -> flips at 1.4 s, 178 deg
#   training soil + av 0.5 -> falls at 2.4 s, 103 deg
#   eval soil     + av 2.0 -> PASS but drifts to 13.7 deg, 11 cm front-rear split
#   training soil + av 2.0 -> PASS, 6.8 deg peak, 4 cm split, tilt 0.7 deg at t=1
# Soft soil fixes the IMPACT: spike falls 1168 N to 138 N, about one robot weight.
# Viscosity fixes the RINGING: on soft soil at av 0.5 the box force swing halves
# but its vertical excursion nearly TRIPLES, 0.024 m to 0.069 m, and it is the
# movement rather than the force that topples a quadruped.
SOIL_PRESETS = {
    "eval": dict(density=1700.0, young=1.0e6, poisson=0.3, mu_I0=0.04,
                 friction=0.8, diam=0.005, cohesion=5000.0),
    "training": dict(density=1700.0, young=5.0e5, poisson=0.3, mu_I0=0.04,
                     friction=0.8, diam=0.005, cohesion=2000.0),
}


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


class PolicyController:
    """model_2999.pt, the CRM-finetuned checkpoint. 45 obs in, 12 actions out."""

    def __init__(self, ckpt: Path, cfgs: Path):
        import torch
        import torch.nn as nn

        self.torch = torch
        with cfgs.open("rb") as fh:
            self.env_cfg, self.train_cfg = pickle.load(fh)
        hidden = self.train_cfg["policy"]["actor_hidden_dims"]

        layers, in_dim = [], 45
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 12))
        self.actor = nn.Sequential(*layers)

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.actor.load_state_dict({
            k.removeprefix("actor."): v
            for k, v in state["model_state_dict"].items() if k.startswith("actor.")
        })
        self.actor.eval()
        self.last_actions = np.zeros(12, dtype=np.float32)
        # Hardcoded, NOT env_cfg['target_lin_vel']. See docstring note 4.
        self.command = np.array([0.5, 0.0, 0.0], dtype=np.float32) * LIN_VEL_SCALE

    @staticmethod
    def _projected_gravity(q) -> np.ndarray:
        qw, qx, qy, qz = q.e0, q.e1, q.e2, q.e3
        return np.array([-2 * (qx * qz - qw * qy),
                         -2 * (qy * qz + qw * qx),
                         -(1 - 2 * (qx * qx + qy * qy))], dtype=np.float32)

    def observe(self, robot: Go2Robot) -> np.ndarray:
        base = robot.base()
        w = base.GetAngVelLocal()
        # Negated AND reordered. See docstring notes 2 and 3.
        pos = -robot.joint_pos()[CHRONO_TO_GENESIS]
        vel = -robot.joint_vel()[CHRONO_TO_GENESIS]
        return np.concatenate([
            np.array([w.x, w.y, w.z], dtype=np.float32) * ANG_VEL_SCALE,
            self._projected_gravity(base.GetRot()),
            self.command,
            (pos - GENESIS_DEFAULTS) * DOF_POS_SCALE,
            vel * DOF_VEL_SCALE,
            self.last_actions,
        ]).astype(np.float32)

    def act(self, robot: Go2Robot) -> np.ndarray:
        obs = self.torch.from_numpy(self.observe(robot)).unsqueeze(0)
        with self.torch.no_grad():
            action = self.actor(obs).squeeze(0).numpy().astype(np.float32)
        self.last_actions = action
        targets_policy_frame = action * 0.25 + GENESIS_DEFAULTS
        return -targets_policy_frame[CHRONO_TO_GENESIS].astype(np.float64)


def build_crm(chrono, fsi, veh, system, robot, args):
    terrain = veh.CRMTerrain(system, args.spacing)
    terrain.SetVerbose(False)
    terrain.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -GRAVITY))
    terrain.SetStepSizeCFD(args.step)

    soil = SOIL_PRESETS[args.soil]
    mat = crm_compat.soil_properties()
    mat.density, mat.Young_modulus, mat.Poisson_ratio = soil["density"], soil["young"], soil["poisson"]
    mat.mu_I0, mat.mu_fric_s, mat.mu_fric_2 = soil["mu_I0"], soil["friction"], soil["friction"]
    mat.average_diam, mat.cohesion_coeff = soil["diam"], soil["cohesion"]
    crm_compat.set_crm_soil(terrain, mat)

    p = fsi.SPHParameters()

    def sph_set(name, value):
        """Assign an SPHParameters field, or fail loudly.

        SPHParameters accepts ANY attribute name silently: `p.kernel_threshold
        = 0.8` is accepted and reads back as 0.8, while the C++ never sees it.
        So a config copied from a different Chrono version fails OPEN, running
        solver defaults while appearing configured. `playground_crm.py` uses
        kernel_threshold, viscosity_type, boundary_type and
        consistent_gradient_discretization; none of those exist in pychrono
        10.0.0, and adopting them would have silently disabled four settings.
        This class of bug is invisible by construction, so assert.
        """
        if not hasattr(p, name):
            raise AttributeError(
                f"SPHParameters has no field {name!r} in this Chrono build. "
                f"Available: {sorted(a for a in dir(p) if not a.startswith('_'))}"
            )
        setattr(p, name, value)

    sph_set("integration_scheme", fsi.IntegrationScheme_RK2)
    sph_set("initial_spacing", args.spacing)
    sph_set("d0_multiplier", 1.0)
    sph_set("free_surface_threshold", 0.8)
    # 0.5 comes from demo_ROBOT_Viper_CRM.py, and Viper is a wheeled rover
    # ROLLING, not a legged robot LANDING 15 kg on four small contact patches.
    # At 0.5 a dropped body enters an undamped limit cycle: force swinging 161 N
    # peak-to-peak on a 36.8 N body, still going after two seconds. Monotonic
    # dose-response, measured on a single sphere:
    #   av 0.5 -> Fz p2p 161.1 N, 6.3 Hz     av 2.0 -> 69.7 N
    #   av 1.0 -> 111.6 N                    av 5.0 -> 1.0 N, no oscillation
    #
    # CAVEAT, and it matters more for the study than for the demo: artificial
    # viscosity is a NUMERICAL dissipation term, not a soil property. Over-
    # damping changes the foot-soil interaction Case Study IV exists to measure,
    # so a sinkage or drawbar number is not version-independent across this
    # value. Use the LOWEST value that removes the limit cycle, not the highest
    # that works.
    sph_set("artificial_viscosity", args.artificial_viscosity)
    sph_set("shifting_method", fsi.ShiftingMethod_NONE)
    # Chrono warns that ARTIFICIAL_UNILATERAL, the default, is less stable for
    # CRM granular; demo_ROBOT_Viper_CRM.py sets these explicitly.
    sph_set("viscosity_method", fsi.ViscosityMethod_ARTIFICIAL_BILATERAL)
    sph_set("boundary_method", fsi.BoundaryMethod_ADAMI)
    terrain.SetSPHParameters(p)

    # FSI bodies MUST be registered before Construct/Initialize: BCE markers are
    # generated at initialisation, and a body added later is silently uncoupled.
    foot_geom = chrono.ChBodyGeometry()
    foot_geom.coll_spheres.append(chrono.SphereShape(chrono.VNULL, 0.025))
    calf_geom = chrono.ChBodyGeometry()
    calf_geom.coll_cylinders.append(
        chrono.CylinderShape(chrono.ChVector3d(0, 0, 0), chrono.QUNIT, 0.02, 0.2))

    # check_embedded=True. With it False, Chrono does NOT remove SPH particles
    # that overlap a body's BCE markers at initialisation, and those overlaps
    # resolve as an enormous repulsive impulse on the first steps. The Go2
    # spawns with legs extended in the URDF rest pose, so its feet start below
    # the soil surface and get ejected.
    #
    # The evidence is an inverted correlation: lowering the spawn from 0.42 m
    # to 0.34 m made the launch BIGGER, 9.2 cm of rise to 15.9 cm, and the fall
    # EARLIER, 1.20 s to 1.04 s. Less initial height cannot mean more energy
    # unless the energy is coming from depth of embedding. A pose-snap
    # explanation predicts the opposite, and a 0.75 s joint ramp made it worse
    # rather than better, which rules that out.
    #
    # demo_ROBOT_Viper_CRM.py passes False, but it spawns the rover clear of the
    # soil, so it never exercises the embedded case.
    # The calves are coupled on CRM but have rigid collision DISABLED on the
    # rigid control, so the two cases are not comparable in what touches the
    # ground. playground_crm.py couples them; --no-calf-fsi removes them so the
    # difference can be tested rather than assumed.
    pairs = [(n, foot_geom) for n in FOOT_BODIES]
    if not args.no_calf_fsi:
        pairs += [(n, calf_geom) for n in CALF_BODIES]
    coupled = []
    for name, geom in pairs:
        body = robot.body(name)
        if body is None:
            continue
        try:
            terrain.AddRigidBody(body, geom, args.check_embedded)
            coupled.append(name)
        except Exception as exc:  # noqa: BLE001
            print(f"  FSI registration failed for {name}: {type(exc).__name__}: {exc}")

    terrain.SetActiveDomain(chrono.ChVector3d(1.0, 1.0, 1.0))
    crm_compat.set_free_flow_duration(terrain, 0.1)
    terrain.Construct(
        chrono.ChVector3d(args.patch_x, args.patch_y, args.depth),
        chrono.ChVector3d(args.patch_x / 2 - 0.6, 0, args.soil_bottom),
        fsi.BoxSide_ALL & ~fsi.BoxSide_Z_POS,   # bitwise, not `and`
    )
    terrain.Initialize()
    return terrain, coupled


def add_soil_visual_proxy(chrono, system, args, soil_top: float):
    """A static box at the soil surface, purely so the camera has a floor.

    SPH markers are not visual shapes, so Chrono::Sensor has nothing to
    ray-trace and renders the robot in an empty void. Chrono's own Viper CRM
    demo shows the soil through ChSphVisualizationVSG, a VSG plugin with no
    equivalent on the sensor path.

    This is a PROXY and does not deform: it shows where the surface is, not what
    the soil is doing. It is visual only, no collision, no FSI coupling, and it
    cannot be used to judge sinkage or soil deformation from a frame.
    """
    body = chrono.ChBody()
    body.SetFixed(True)
    body.EnableCollision(False)
    box = chrono.ChVisualShapeBox(args.patch_x, args.patch_y, 0.02)
    texture = chrono.GetChronoDataFile("textures/pinkwhite.png")
    if Path(texture).is_file():
        box.SetTexture(texture, 8 * args.patch_x, 8 * args.patch_y)
    body.AddVisualShape(box, chrono.ChFramed(
        chrono.ChVector3d(args.patch_x / 2 - 0.6, 0, soil_top - 0.01), chrono.QUNIT))
    system.AddBody(body)
    return body


def measure_leg_reach(chrono, urdf: Path) -> float:
    """Base-to-lowest-foot distance in the URDF rest pose.

    A throwaway system is built and discarded. Nothing has stepped, so the pose
    is purely the URDF's own rest configuration and the offset is a fixed
    geometric property of the model. Cheap: a URDF parse, no dynamics.

    This exists because a constant spawn height is guesswork. The Go2 spawns with
    its legs extended, so a clearance chosen for the BASE says nothing about
    where the FEET are, and the feet are what meets the soil. Watching the first
    render, the failure is plainly visible: the robot starts with its feet below
    the surface.
    """
    probe_system = chrono.ChSystemSMC()
    probe = Go2Robot(probe_system, urdf,
                     chrono.ChFramed(chrono.ChVector3d(0, 0, 1.0), chrono.QuatFromAngleZ(0.0)))
    base_z = probe.base().GetPos().z
    foot_z = min(probe.body(n).GetPos().z for n in FOOT_BODIES if probe.body(n) is not None)
    return float(base_z - foot_z)


def build_rigid_ground(chrono, system):
    """The ground the policy was actually trained on.

    The Go2 skill is explicit: the RL policy was trained with a ChBodyEasyBox at
    z=0, so its top surface sits at z=+0.05. Reproducing that exactly is the
    control case. If the Go2 does not walk here, nothing about CRM is the
    problem.
    """
    mat = chrono.ChContactMaterialSMC()
    mat.SetFriction(0.9)
    mat.SetRestitution(0.01)
    mat.SetGn(60.0)
    mat.SetKn(2e5)
    ground = chrono.ChBodyEasyBox(10, 10, 0.1, 1000, True, True, mat)
    ground.SetName("ground")
    ground.SetPos(chrono.ChVector3d(0, 0, 0.0))
    ground.SetFixed(True)
    texture = chrono.GetChronoDataFile("textures/concrete.jpg")
    if Path(texture).is_file() and ground.GetVisualShape(0) is not None:
        ground.GetVisualShape(0).SetTexture(texture, 10, 10)
    system.AddBody(ground)
    return ground, 0.05   # top surface


def attach_sph_rendering(sens, manager, terrain, args):
    """Make the CRM soil visible to Chrono::Sensor.

    SPH markers are not visual shapes, so a camera sees nothing by default and
    the robot appears to tumble in a void. The sensor manager renders particles
    as sprites instead, via AttachFsiSphSystem; see Chrono's own
    src/demos/sensor/demo_SEN_CRM_Rendering.cpp. This is the real mechanism and
    it supersedes the static proxy box, which showed where the surface was but
    could not show the soil deform.
    """
    attach = getattr(manager, "AttachFsiSphSystem", None)
    if attach is None:
        return "AttachFsiSphSystem unavailable in this build"
    getter = getattr(terrain, "GetFluidSystemSPH", None) or getattr(terrain, "GetFsiSystemSPH", None)
    if getter is None:
        return "no SPH system accessor on CRMTerrain"
    try:
        sph = getter()
        opts_cls = getattr(sens, "ChFsiSphRenderOptions", None)
        if opts_cls is None:
            attach(sph)
            return "attached (no render options class; defaults)"
        opts = opts_cls()
        # Sprite spacing drives how densely particles are drawn. Tie it to the
        # actual particle spacing rather than the demo's hardcoded 0.01.
        if hasattr(opts, "render_particle_spacing"):
            opts.render_particle_spacing = float(args.sph_render_spacing or args.spacing)
        attach(sph, opts)
        return f"attached, render_particle_spacing={args.sph_render_spacing or args.spacing}"
    except Exception as exc:  # noqa: BLE001
        return f"FAILED: {type(exc).__name__}: {exc}"


def attach_camera(chrono, system, args, soil_top: float, terrain=None):
    try:
        import pychrono.sensor as sens
    except Exception as exc:  # noqa: BLE001
        return None, f"pychrono.sensor unavailable ({type(exc).__name__})", None
    mount = chrono.ChBody()
    mount.SetFixed(True)
    mount.EnableCollision(False)
    system.AddBody(mount)

    manager = sens.ChSensorManager(system)
    manager.scene.SetAmbientLight(chrono.ChVector3f(0.35, 0.35, 0.40))
    if hasattr(manager.scene, "AddDirectionalLight"):
        manager.scene.AddDirectionalLight(chrono.ChColor(1.0, 0.95, 0.85),
                                          math.radians(55.0), math.radians(120.0))
    else:
        manager.scene.AddPointLight(chrono.ChVector3f(2, -2.5, 3),
                                    chrono.ChColor(1.0, 0.95, 0.85), 25.0)
    crm_compat.set_solid_background(manager.scene, chrono.ChVector3f(0.55, 0.68, 0.85))

    # Attach the SPH system BEFORE the camera exists, matching the C++ demo
    # (AttachFsiSphSystem at demo line 295, camera built after). The attach calls
    # ReconstructScenes() internally, so a camera added first is built against a
    # scene that has no FSI source in it, and the later rebuild does not retrofit
    # the pipeline that camera already holds. Attaching after AddSensor produced
    # frames byte-identical to attaching not at all -- options populated, handle
    # 0, no error, and nothing drawn.
    sph_note = attach_sph_rendering(sens, manager, terrain, args) if terrain is not None else "n/a"
    print(f"sph rendering: {sph_note}")

    eye = np.array([-1.15, -1.45, soil_top + 0.62])
    target = np.array([0.30, 0.0, soil_top + 0.08])
    pose = chrono.ChFramed(chrono.ChVector3d(*eye), _look_at(chrono, eye, target))
    cam = sens.ChCameraSensor(mount, args.video_fps, pose,
                              args.video_width, args.video_height, math.radians(58.0))
    cam.SetLag(0.0)
    cam.SetCollectionWindow(0.0)
    frames = Path(args.out) / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    save = getattr(sens, "ChFilterSave", None)
    if save is None:
        return None, "ChFilterSave unavailable", None
    cam.PushFilter(save(str(frames) + "/"))
    manager.AddSensor(cam)
    return manager, f"frames -> {frames} | sph: {sph_note}", mount


def _look_at(chrono, eye, target):
    """Chrono::Sensor cameras look down +X with +Z up."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    up_hint = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.999 else np.array([0.0, 1.0, 0.0])
    left = np.cross(up_hint, f); left /= np.linalg.norm(left)
    up = np.cross(f, left)
    r = np.column_stack([f, left, up])
    tr = float(np.trace(r))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r))); j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
        q = [0.0] * 4
        q[0] = (r[k, j] - r[j, k]) / s
        q[i + 1], q[j + 1], q[k + 1] = 0.25 * s, (r[j, i] + r[i, j]) / s, (r[k, i] + r[i, k]) / s
    return chrono.ChQuaterniond(*q)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", default="/home/kyle/Documents/sbel/sbel-reproducibility/2025/multi-terrain-RL")
    ap.add_argument("--sim-seconds", type=float, default=3.0)
    ap.add_argument("--step", type=float, default=5e-4, help="CFD step")
    ap.add_argument("--exchange-mult", type=int, default=5, help="MBS/CFD exchange = mult * step")
    ap.add_argument("--control-hz", type=float, default=50.0)
    ap.add_argument("--pose-ramp-seconds", type=float, default=0.75,
                    help="ease from the URDF spawn pose to the stand pose")
    ap.add_argument("--settle-seconds", type=float, default=0.5,
                    help="hold the stand pose after the ramp, before the policy")
    # playground_crm.py values. The foot sphere is r=0.025, so 0.02 gives a
    # foot spanning 2.5 spacings against 1.7 at 0.03.
    ap.add_argument("--spacing", type=float, default=0.02)
    ap.add_argument("--patch-x", type=float, default=8.0)
    ap.add_argument("--patch-y", type=float, default=4.0)
    # 0.2, matching chrono_crmenv.py and sitting just under the ~0.22 m depth
    # at which the bed starts heaving.
    ap.add_argument("--depth", type=float, default=0.20)
    ap.add_argument("--soil-bottom", type=float, default=0.0)
    # Swept, 3 s each, standing: 0.34 -> 0.13-0.16 m of launch and a fall at
    # 0.64-0.85 s; 0.42 -> 0.07 m and 1.08-1.21 s; 0.60 -> ZERO launch and 2.97 s.
    # Monotonic and it disappears entirely by 0.60. The Go2's URDF rest pose has
    # the legs extended, so at low clearance the foot BCE markers spawn within
    # interaction range of a dense particle bed and take a contact impulse.
    # Note this is NOT particles trapped inside the feet: check_embedded=True
    # removes only 40 of 43,632 particles and changes nothing.
    # Default None means derive it: place the base so the lowest foot sits
    # foot_margin above the soil. A constant cannot do this, because it fixes
    # the BASE height while the FEET are what meets the soil, and the Go2's rest
    # pose has the legs extended. Passing a value overrides and reproduces the
    # old behaviour. Swept values for reference, standing, 3 s: base clearance
    # 0.34 gave 0.13-0.16 m of launch and a fall at 0.64-0.85 s; 0.42 gave 0.07 m
    # and 1.08-1.21 s; 0.60 gave zero launch and 2.97 s.
    ap.add_argument("--spawn-clearance", type=float, default=None,
                    help="base height above soil; omit to derive from leg reach")
    # Expressed in SPH spacings, not metres, because what the foot has to clear
    # is the kernel support radius and that scales with spacing. Measured against
    # leg reach 0.421 m: feet 0.081 m BELOW the surface gave 0.132 m of launch,
    # 0.001 m below gave 0.074 m, and 0.179 m above (about 6 spacings at 0.03)
    # gave none. The cost of too much margin is a slightly longer drop; the cost
    # of too little is a launch, so this errs high.
    ap.add_argument("--foot-margin-spacings", type=float, default=5.0,
                    help="foot gap above the soil at spawn, in SPH spacings")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--video-fps", type=float, default=30.0)
    ap.add_argument("--video-width", type=int, default=960)
    ap.add_argument("--video-height", type=int, default=540)
    # Default ON: a fixed frame loses a walking robot (the first CRM walk
    # left shot at t=7.40 of 8 s). --no-cam-follow restores the fixed frame.
    ap.add_argument("--cam-follow", dest="cam_follow", action="store_true", default=True)
    ap.add_argument("--no-cam-follow", dest="cam_follow", action="store_false")
    ap.add_argument("--no-check-embedded", dest="check_embedded",
                    action="store_false", default=True,
                    help="keep SPH particles that overlap the feet at init; "
                         "reproduces the launch, for comparison only")
    # The static proxy is now a fallback: AttachFsiSphSystem renders the actual
    # particles, which the proxy cannot, so the proxy is off unless asked for.
    ap.add_argument("--soil-proxy", action="store_true",
                    help="add a static non-deforming floor box as well as the SPH sprites")
    ap.add_argument("--sph-render-spacing", type=float, default=None,
                    help="sprite spacing; defaults to the SPH initial spacing")
    ap.add_argument("--terrain", choices=["crm", "rigid"], default="crm",
                    help="rigid reproduces the ground the policy was trained on")
    ap.add_argument("--solver-iters", type=int, default=150)
    ap.add_argument("--soil", choices=["eval", "training"], default="training",
                    help="training is the softer soil the CRM finetune actually used")
    ap.add_argument("--artificial-viscosity", type=float, default=2.0,
                    help="0.5 is the Viper demo value and leaves an undamped "
                         "limit cycle under impact; see the note in build_crm")
    ap.add_argument("--no-calf-fsi", action="store_true",
                    help="couple only the feet to the SPH, not the calves")
    ap.add_argument("--no-policy", action="store_true", help="hold the stand pose throughout")
    ap.add_argument("--out", default="artifacts/go2_crm")
    args = ap.parse_args()

    import os
    cwd_at_start = os.getcwd()

    import pychrono as chrono
    import pychrono.vehicle as veh
    fsi = None
    if args.terrain == "crm":
        try:
            import pychrono.fsi as fsi
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: pychrono.fsi unavailable ({type(exc).__name__}). "
                  "CRM needs the nedm env; --terrain rigid does not.")
            return 1

    assets = Path(args.assets)
    urdf = assets / "data/robot/go2_irrvis/urdf/go2_description.urdf"
    ckpt = assets / "data/rl_models/rslrl/model_2999.pt"
    cfgs = assets / "data/rl_models/rslrl/cfgs.pkl"
    for f in (urdf, ckpt, cfgs):
        if not f.is_file():
            print(f"FAIL: missing {f}")
            return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    system = chrono.ChSystemSMC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -GRAVITY))
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    # 150, from playground_crm.py. The Go2 skill says 60, but that is for
    # rigid ground; a 42-body articulated robot on compliant terrain is
    # exactly where an under-converged contact solve shows up as excess bounce.
    system.GetSolver().AsIterative().SetMaxIterations(args.solver_iters)
    chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0025)
    chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0025)

    rigid = args.terrain == "rigid"
    # For CRM, Construct's pos is the BOTTOM of the soil box, and GetHeight does
    # NOT report the free surface (it is the height-functor hook, flat zero by
    # default). The surface is soil_bottom + depth. For rigid, the Go2 skill's
    # ChBodyEasyBox at z=0 puts its top at +0.05.
    soil_top = 0.05 if rigid else args.soil_bottom + args.depth
    if args.spawn_clearance is None:
        os.chdir(urdf.parent)
        try:
            leg_reach = measure_leg_reach(chrono, urdf)
        finally:
            os.chdir(cwd_at_start)
        foot_margin = args.foot_margin_spacings * args.spacing
        spawn_z = soil_top + foot_margin + leg_reach
        print(f"auto-spawn: leg reach {leg_reach:.3f} m, foot margin {foot_margin:.3f} m "
              f"({args.foot_margin_spacings:g} spacings) -> base at {spawn_z:.3f}")
    else:
        leg_reach = None
        spawn_z = soil_top + args.spawn_clearance
    init = chrono.ChFramed(chrono.ChVector3d(0.0, 0.0, spawn_z), chrono.QuatFromAngleZ(0.0))

    # URDF meshes are referenced relatively; resolve from the urdf directory.
    cwd = cwd_at_start
    os.chdir(urdf.parent)
    try:
        robot = Go2Robot(system, urdf, init)
    finally:
        os.chdir(cwd)

    foot_z = [robot.body(n).GetPos().z for n in FOOT_BODIES if robot.body(n) is not None]
    foot_clearance = min(foot_z) - soil_top if foot_z else float("nan")
    if foot_clearance < 0.05:
        print(f"WARNING: lowest foot is {foot_clearance:.3f} m above the soil surface. "
              "Below ~0.05 m the foot BCE markers take a launch impulse from the "
              "particle bed; raise --spawn-clearance.")

    if rigid:
        build_rigid_ground(chrono, system)
        terrain, coupled = None, []
    else:
        terrain, coupled = build_crm(chrono, fsi, veh, system, robot, args)
    print(f"soil top {soil_top:.3f}  spawn z {spawn_z:.3f}  "
          f"lowest foot {min(foot_z):.3f} (clearance {foot_clearance:.3f})  "
          f"FSI-coupled bodies: {len(coupled)}")
    if terrain is not None:
        print(f"SPH particles {terrain.GetNumSPHParticles()}  "
              f"boundary BCE {terrain.GetNumBoundaryBCEMarkers()}")

    manager, video_note, cam_mount = (None, "disabled", None)
    if args.video:
        if args.soil_proxy:
            add_soil_visual_proxy(chrono, system, args, soil_top)
        manager, video_note, cam_mount = attach_camera(chrono, system, args, soil_top, terrain)
        print(f"video: {video_note}")

    policy = None if args.no_policy else PolicyController(ckpt, cfgs)

    exchange = args.exchange_mult * args.step
    control_every = max(1, int(round((1.0 / args.control_hz) / exchange)))
    n_steps = int(args.sim_seconds / exchange)
    base = robot.base()
    z0 = base.GetPos().z
    x0 = base.GetPos().x
    foot_bodies = {n: robot.body(n) for n in FOOT_BODIES}
    try:
        total_mass = sum(b.GetMass() for b in system.GetBodies())
    except Exception:  # noqa: BLE001 - a diagnostic must not break the run
        total_mass = float("nan")
    log, tilts, wall0, fell_at = [], [], time.perf_counter(), None

    # The URDF spawns at its own rest configuration, which is NOT the stand
    # pose. Commanding the stand pose directly gives ChParserURDF's position
    # motors a large step error to close in one control tick, and they close it
    # by launching the robot: measured 9.2 cm of RISE in the first 200 ms on a
    # robot commanded only to hold still, followed by a topple at 1.2 s. Nothing
    # about soil compliance makes a stationary robot go up. Ramp instead.
    q0 = robot.joint_pos().astype(np.float64)
    initial_error = np.abs(q0 - STAND_ACTION)
    print(f"initial joint error vs stand pose: max {initial_error.max():.3f} rad, "
          f"mean {initial_error.mean():.3f} rad")

    robot.actuate(q0)
    for i in range(n_steps):
        t = i * exchange
        if i % control_every == 0:
            if t < args.pose_ramp_seconds:
                a = t / max(args.pose_ramp_seconds, 1e-9)
                robot.actuate(q0 + a * (STAND_ACTION - q0))
            elif policy is None or t < args.pose_ramp_seconds + args.settle_seconds:
                robot.actuate(STAND_ACTION)
            else:
                robot.actuate(policy.act(robot))
        if terrain is not None:
            terrain.DoStepDynamics(exchange)   # advances BOTH fluid and multibody
        else:
            system.DoStepDynamics(exchange)
        if cam_mount is not None and args.cam_follow:
            # Translate the camera mount with the robot, so the pose the camera
            # holds relative to it is preserved. A fixed frame loses a walking
            # robot: the first CRM walk left shot at t=7.40 of 8 s, and the
            # RoboSimian framing only survived because it barely moved.
            cam_mount.SetPos(chrono.ChVector3d(base.GetPos().x - x0, 0.0, 0.0))
        if manager is not None:
            manager.Update()
        p, q = base.GetPos(), base.GetRot()
        fz, ffz = [], []
        for n in FOOT_BODIES:
            b = foot_bodies.get(n)
            fz.append(b.GetPos().z if b is not None else float("nan"))
            if b is not None and terrain is not None:
                try:
                    ffz.append(float(terrain.GetFsiBodyForce(b).z))
                except Exception:  # noqa: BLE001
                    ffz.append(float("nan"))
            else:
                ffz.append(float("nan"))
        log.append([t, p.x, p.y, p.z, q.e0, q.e1, q.e2, q.e3, *fz, *ffz])
        # Tilt from upright, NOT base height. A 6 s run reported PASS on a robot
        # lying inverted on the soil: base z was 0.0074, still nominally above a
        # soil top of 0.0, because 7 mm off the ground is what lying down looks
        # like. Height measures something adjacent to falling; angle measures
        # falling. Same failure the RoboSimian roll metric had.
        up_z = 1.0 - 2.0 * (q.e1 * q.e1 + q.e2 * q.e2)
        tilt = math.acos(max(-1.0, min(1.0, up_z)))
        tilts.append(tilt)
        if fell_at is None and (tilt > FALL_TILT_RAD or p.z < soil_top - 0.05):
            fell_at = t

    wall = time.perf_counter() - wall0
    arr = np.asarray(log)
    np.savez_compressed(out / "trajectory.npz", log=arr,
                        columns=np.array(["t", "x", "y", "z", "e0", "e1", "e2", "e3",
                                          *[f"footz_{n}" for n in FOOT_BODIES],
                                          *[f"fsiFz_{n}" for n in FOOT_BODIES]]))
    n_frames = len(list((out / "frames").glob("*"))) if (out / "frames").is_dir() else 0
    summary = {
        "sim_seconds": args.sim_seconds, "wall_seconds": round(wall, 1),
        "realtime_factor": round(args.sim_seconds / wall, 5) if wall else None,
        "rtf_cfd": round(float(terrain.GetRtfCFD()), 5) if terrain else None,
        "rtf_mbd": round(float(terrain.GetRtfMBD()), 5) if terrain else None,
        "fsi_coupled_bodies": len(coupled), "coupled_names": coupled,
        "terrain": args.terrain,
        "system_total_mass_kg": round(float(total_mass), 3),
        "weight_n": round(float(total_mass * GRAVITY), 1),
        "solver_iters": args.solver_iters,
        "artificial_viscosity": args.artificial_viscosity,
        "soil_preset": args.soil,
        "soil": SOIL_PRESETS[args.soil],
        "sph_particles": int(terrain.GetNumSPHParticles()) if terrain else 0,
        "soil_top_m": soil_top, "spawn_z_m": spawn_z,
        "forward_travel_m": round(float(arr[-1, 1] - arr[0, 1]), 4),
        "lateral_travel_m": round(float(arr[-1, 2] - arr[0, 2]), 4),
        "base_z_start_end_m": [round(z0, 4), round(float(arr[-1, 3]), 4)],
        "base_z_min_m": round(float(arr[:, 3].min()), 4),
        "fell": bool(fell_at is not None), "fell_at_s": fell_at,
        "max_tilt_deg": round(math.degrees(max(tilts)), 1) if tilts else None,
        "final_tilt_deg": round(math.degrees(tilts[-1]), 1) if tilts else None,
        "policy": "none (stand pose)" if policy is None else "model_2999.pt",
        "pose_ramp_s": args.pose_ramp_seconds,
        "check_embedded": args.check_embedded,
        "foot_clearance_above_soil_m": round(float(foot_clearance), 4),
        "leg_reach_m": round(leg_reach, 4) if leg_reach is not None else None,
        "initial_joint_error_max_rad": round(float(initial_error.max()), 4),
        "initial_joint_error_mean_rad": round(float(initial_error.mean()), 4),
        "video": video_note, "frames_written": n_frames,
    }
    if args.video and n_frames:
        try:
            from PIL import Image
            dst = out / "jpg"
            dst.mkdir(exist_ok=True)
            # NUMERIC sort. Chrono names frames frame_0.png, frame_1.png,
            # frame_10.png, so lexicographic order is not chronological and the
            # clip plays the robot flipping back and forth at random. Caught
            # only because the frames contradicted the trajectory.
            def _frame_index(path):
                digits = "".join(c for c in path.stem if c.isdigit())
                return int(digits) if digits else -1
            for i, png in enumerate(sorted((out / "frames").glob("*.png"), key=_frame_index)):
                Image.open(png).convert("RGB").save(dst / f"f{i:05d}.jpg", quality=85)
            summary["frames_jpeg"] = str(dst)
        except Exception as exc:  # noqa: BLE001
            summary["frames_jpeg"] = f"transcode skipped ({type(exc).__name__})"

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nGATE: " + (f"FAIL, fell at {fell_at:.2f}s (max tilt {summary['max_tilt_deg']} deg)"
                    if summary["fell"] else
                    f"PASS, upright for the full window (max tilt {summary['max_tilt_deg']} deg)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
