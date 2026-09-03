#!/usr/bin/env python
"""Prove CRM terrain and a Chrono::Sensor camera can run in one process.

Nothing in this repository has ever done both at once. `hmmwv_crm.py` uses FSI
with no camera; `traverse/scene.py` and `double_pendulum_data.py` use a camera
on rigid or heightmap terrain. Case Study IV (excavation) needs both, and Case
Study III wants CRM with the option of vision, so this is the combination that
has to be established before either is planned around.

It also measures the **CRM realtime factor**, which
`docs/state/progress/future-case-studies.md` argues about ("CRM runs below
realtime") and which nothing has ever measured. `--no-camera` gives that number
without rendering cost.

Requirements, verified against Chrono's own changelog for 10.0.0:

    Ray-tracing sensor models in Chrono::Sensor now require OptiX 9.0 or 9.1
    (and corresponding NVIDIA driver versions).

OptiX 9.1 needs an **R590 or newer** NVIDIA driver. On an older driver the
camera half fails with `OPTIX_ERROR_UNSUPPORTED_ABI_VERSION` at
`ChOptixEngine.cpp:86` while CRM works fine, which is exactly the split seen on
`kyle-sbel` (driver 580.173.02). Run `--no-camera` there until the driver moves.

Usage, under the `nedm` environment (pychrono 10; 9.0.x has no FSI at all):

    "$NEDM_PY" scripts/crm_sensor_smoke.py                 # both halves
    "$NEDM_PY" scripts/crm_sensor_smoke.py --no-camera     # CRM throughput only
    "$NEDM_PY" scripts/crm_sensor_smoke.py --no-crm        # camera only

Soil parameters are taken from `configs/hmmwv_crm_eval.json` so the throughput
number is comparable to the collection the project actually runs: spacing 0.08 m
and step 5e-4 s match exactly, and the 2 m patch is close to that config's
`active_domain_m` of [2, 2, 1], which is what actually gets simulated. Real
collection still carries roughly 2.5x the particles plus BCE markers for four
tires plus the vehicle's own multibody dynamics, so treat any number here as
optimistic against a full collection run.

**Ordering rule:** rigid bodies must be registered with the terrain *before*
`Construct`/`Initialize`. BCE markers are generated at initialisation, so a body
added afterwards raises `Expression '!m_is_initialized' returned false` and stays
silently uncoupled, making the SPH look fast rather than broken.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nedm import chrono_crm_compat as crm_compat  # noqa: E402

GRAVITY = 9.81

# From configs/hmmwv_crm_eval.json, so timings are comparable to real collection.
SOIL = dict(density=1700.0, young_modulus_pa=1.0e6, poisson_ratio=0.3,
            mu_I0=0.04, friction=0.8, average_diam_m=0.005, cohesion=5000.0)
SPH = dict(d0_multiplier=1.0, free_surface_threshold=2.0, artificial_viscosity=0.5,
           shifting_ppst_push=1.0, shifting_ppst_pull=1.0, num_proximity_search_steps=4)


def build_crm(chrono, fsi, veh, system, args):
    terrain = veh.CRMTerrain(system, args.spacing)
    terrain.SetVerbose(False)
    terrain.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -GRAVITY))
    terrain.SetStepSizeCFD(args.step)

    mat = crm_compat.soil_properties()
    mat.density = SOIL["density"]
    mat.Young_modulus = SOIL["young_modulus_pa"]
    mat.Poisson_ratio = SOIL["poisson_ratio"]
    mat.mu_I0 = SOIL["mu_I0"]
    mat.mu_fric_s = SOIL["friction"]
    mat.mu_fric_2 = SOIL["friction"]
    mat.average_diam = SOIL["average_diam_m"]
    mat.cohesion_coeff = SOIL["cohesion"]
    crm_compat.set_crm_soil(terrain, mat)

    p = fsi.SPHParameters()
    p.integration_scheme = fsi.IntegrationScheme_RK2
    p.initial_spacing = args.spacing
    p.d0_multiplier = SPH["d0_multiplier"]
    p.free_surface_threshold = SPH["free_surface_threshold"]
    p.artificial_viscosity = SPH["artificial_viscosity"]
    p.shifting_method = fsi.ShiftingMethod_NONE
    p.shifting_ppst_push = SPH["shifting_ppst_push"]
    p.shifting_ppst_pull = SPH["shifting_ppst_pull"]
    p.use_consistent_gradient_discretization = False
    p.use_consistent_laplacian_discretization = False
    p.viscosity_method = fsi.ViscosityMethod_ARTIFICIAL_BILATERAL
    p.boundary_method = fsi.BoundaryMethod_ADAMI
    if hasattr(p, "num_proximity_search_steps"):
        p.num_proximity_search_steps = SPH["num_proximity_search_steps"]
    terrain.SetSPHParameters(p)

    # Rigid bodies MUST be registered before Construct/Initialize. BCE markers
    # are generated at initialisation, so adding afterwards raises
    # "Expression '!m_is_initialized' returned false" from
    # ChFsiFluidSystemSPH.cpp and leaves the body silently uncoupled: the SPH
    # then advances with nothing in it, which looks like a fast CRM run rather
    # than a broken one. hmmwv_crm.py registers its spindles here for the same
    # reason.
    probe, coupling = add_probe_body(chrono, system, terrain, args)

    terrain.SetActiveDomain(chrono.ChVector3d(2.0, 2.0, 1.0))
    crm_compat.set_free_flow_duration(terrain, 0.1)
    # Open top: BoxSide_ALL minus Z_POS, same as the HMMWV CRM path.
    terrain.Construct(
        chrono.ChVector3d(args.patch, args.patch, args.depth),
        chrono.ChVector3d(0, 0, 0),
        fsi.BoxSide_ALL & ~fsi.BoxSide_Z_POS,
    )
    terrain.Initialize()
    return terrain, probe, coupling


def add_probe_body(chrono, system, terrain, args):
    """A rigid box dropped on the soil, so the SPH has something to react to."""
    body = chrono.ChBody()
    body.SetPos(chrono.ChVector3d(0, 0, args.drop_height))
    body.SetMass(args.probe_mass)
    body.SetFixed(False)
    body.EnableCollision(False)  # CRM couples through FSI, not the contact system
    d = args.probe_size
    vis = chrono.ChVisualShapeBox(d, d, d)
    body.AddVisualShape(vis, chrono.ChFramed(chrono.VNULL, chrono.QUNIT))
    system.AddBody(body)
    try:
        geometry = chrono.ChBodyGeometry()
        geometry.coll_boxes.append(
            chrono.BoxShape(chrono.VNULL, chrono.QUNIT, chrono.ChVector3d(d, d, d), 0)
        )
        terrain.AddRigidBody(body, geometry, False)
        return body, "coupled via coll_boxes"
    except Exception as exc:  # noqa: BLE001
        return body, f"NOT coupled to SPH ({type(exc).__name__}: {exc})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-seconds", type=float, default=1.0)
    ap.add_argument("--step", type=float, default=5e-4)
    ap.add_argument("--spacing", type=float, default=0.08)
    ap.add_argument("--patch", type=float, default=2.0)
    ap.add_argument("--depth", type=float, default=0.4)
    ap.add_argument("--probe-mass", type=float, default=50.0)
    ap.add_argument("--probe-size", type=float, default=0.25)
    ap.add_argument("--drop-height", type=float, default=0.45)
    ap.add_argument("--no-crm", action="store_true")
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--out", default="artifacts/crm_sensor_smoke")
    ap.add_argument("--video-fps", type=float, default=20.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"crm": "skipped", "camera": "skipped"}

    import pychrono as chrono
    import pychrono.vehicle as veh

    system = chrono.ChSystemNSC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -GRAVITY))
    system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

    terrain = None
    if not args.no_crm:
        try:
            import pychrono.fsi as fsi
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: pychrono.fsi unavailable ({type(exc).__name__}: {exc})")
            print("      CRM needs pychrono 10; the 9.0.x builds ship no fsi module.")
            return 1
        t0 = time.perf_counter()
        terrain, body, coupling = build_crm(chrono, fsi, veh, system, args)
        report["crm"] = "built"
        report["crm_build_seconds"] = round(time.perf_counter() - t0, 2)
        report["probe_coupling"] = coupling
        print(f"CRM built in {report['crm_build_seconds']}s; probe {coupling}")

    manager = None
    if not args.no_camera:
        try:
            import pychrono.sensor as sens
            manager = sens.ChSensorManager(system)
            manager.scene.SetAmbientLight(chrono.ChVector3f(0.4, 0.4, 0.45))
            manager.scene.AddDirectionalLight(
                chrono.ChColor(1.0, 0.95, 0.85), math.radians(55.0), math.radians(120.0))
            mount = chrono.ChBody()
            mount.SetFixed(True)
            mount.EnableCollision(False)
            system.AddBody(mount)
            pose = chrono.ChFramed(chrono.ChVector3d(-2.2, -2.2, 1.6),
                                   chrono.QuatFromAngleZ(math.radians(45.0)))
            cam = sens.ChCameraSensor(mount, args.video_fps, pose, 640, 360, math.radians(60.0))
            cam.SetLag(0.0)
            cam.SetCollectionWindow(0.0)
            frames = out / "frames"
            frames.mkdir(exist_ok=True)
            cam.PushFilter(sens.ChFilterSave(str(frames) + "/"))
            manager.AddSensor(cam)
            report["camera"] = "attached"
        except Exception as exc:  # noqa: BLE001
            report["camera"] = f"FAILED: {type(exc).__name__}: {exc}"
            print(f"camera: {report['camera']}")
            manager = None

    # Step. With CRM present the terrain owns the coupled FSI + MBS advance:
    # call terrain.Advance only, never system.DoStepDynamics as well.
    steps = int(args.sim_seconds / args.step)
    wall0 = time.perf_counter()
    optix_error = None
    for _ in range(steps):
        if terrain is not None:
            terrain.Advance(args.step)
        else:
            system.DoStepDynamics(args.step)
        if manager is not None:
            try:
                manager.Update()
            except Exception as exc:  # noqa: BLE001
                optix_error = f"{type(exc).__name__}: {exc}"
                manager = None
    wall = time.perf_counter() - wall0

    if optix_error:
        report["camera"] = f"FAILED AT RENDER: {optix_error}"
    n_frames = len(list((out / "frames").glob("*"))) if (out / "frames").is_dir() else 0
    report.update({
        "sim_seconds": args.sim_seconds,
        "steps": steps,
        "wall_seconds": round(wall, 2),
        "realtime_factor": round(args.sim_seconds / wall, 5) if wall else None,
        "steps_per_wall_second": round(steps / wall, 1) if wall else None,
        "frames_written": n_frames,
        "spacing_m": args.spacing,
        "patch_m": args.patch,
        # An uncoupled run is the cheapest possible CRM step: no BCE markers and
        # no fluid-solid force computation. Its realtime factor is an upper
        # bound, not a measurement, so say which one this is.
        "timing_is_coupled": bool(terrain is not None
                                  and str(report.get("probe_coupling", "")).startswith("coupled")),
    })
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    both = report["crm"] == "built" and report["camera"] == "attached"
    print("\nRECIPE: " + ("CRM + Chrono::Sensor coexist in one process"
                          if both else "NOT both; see the fields above"))
    if terrain is not None and not report["timing_is_coupled"]:
        print("WARNING: probe is not coupled to the SPH, so realtime_factor is an "
              "UPPER BOUND on CRM cost, not a measurement of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
