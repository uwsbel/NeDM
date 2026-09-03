"""CRM and rigid terrain construction, and the spawn-height derivation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nedm import chrono_crm_compat as crm_compat

from .constants import (CALF_BODIES, FOOT_BODIES, GRAVITY, MOTOR_NAMES,
                          SOIL_PRESETS)
from .robot import Go2Robot


def build_crm(chrono, fsi, veh, system, robot, args):
    terrain = veh.CRMTerrain(system, args.spacing)
    terrain.SetVerbose(False)
    terrain.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -GRAVITY))
    terrain.SetStepSizeCFD(args.step)

    # Preset, then explicit overrides. The presets are the two we have evidence
    # for; --soil-young / --soil-cohesion exist to sweep BELOW the softer of them
    # in search of visible sinkage, which neither preset delivers.
    soil = dict(SOIL_PRESETS[args.soil])
    if getattr(args, "soil_young", None) is not None:
        soil["young"] = float(args.soil_young)
    if getattr(args, "soil_cohesion", None) is not None:
        soil["cohesion"] = float(args.soil_cohesion)
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

