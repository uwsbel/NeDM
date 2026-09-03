"""Python port of tools/crm_render_gate.cpp — equivalence check, not diagnosis.

Ports the C++ gate's scene 1:1 so the Python and C++ Chrono::Sensor paths can be
compared on the SAME scene. Nobody has demonstrated that equivalence yet; the
C++ path renders and the Python path now renders, but on different scenes.

Uses TRIANGLE MESH sprites, not primitives: sprite_shapes accepts a
ChVisualShapeSphere (the declared element type) and silently draws nothing.

Arms: default attaches; --no-attach is the negative control from the same file.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import pychrono as ch
import pychrono.fsi as fsi
import pychrono.sensor as sens
import pychrono.vehicle as veh
import pychrono.robot as robot

ap = argparse.ArgumentParser()
ap.add_argument("--no-attach", action="store_true", help="negative control")
ap.add_argument("--seconds", type=float, default=2.0)
ap.add_argument("--out", default="/tmp/gate_run/run/DEMO_OUTPUT")
args = ap.parse_args()

do_attach = not args.no_attach
gate_tag = "py_attached" if do_attach else "py_noattach"

# Scene constants, copied from crm_render_gate.cpp so the two are comparable.
terrain_length, terrain_width = 8.0, 3.0
init_loc = ch.ChVector3d(1.25, 0.0, 0.55)
num_meshes = 3
density, cohesion, friction = 100.0, 5e3, 0.7
youngs_modulus, poisson_ratio = 0.5e5, 0.3
tend, step_size = args.seconds, 5e-4
active_box_dim = ch.ChVector3d(5.0, 5.0, 5.0)
render_fps = 200.0
initial_spacing = 0.02

out_dir = os.path.join(args.out, f"CRM_RENDER_GATE_{gate_tag}") + "/"
os.makedirs(out_dir + "img_SEN", exist_ok=True)

ch.SetChronoDataPath("/home/kyle/chrono-src/data/")
veh.SetVehicleDataPath("/home/kyle/chrono-src/data/vehicle/")

sysMBS = ch.ChSystemNSC()
sysMBS.SetCollisionSystemType(ch.ChCollisionSystem.Type_BULLET)

# Rover
wheel_mat = ch.ChContactMaterialData()
wheel_mat.mu, wheel_mat.cr, wheel_mat.Y, wheel_mat.nu = 0.4, 0.2, 2e7, 0.3
wheel_mat.kn, wheel_mat.gn, wheel_mat.kt, wheel_mat.gt = 2e5, 40.0, 2e5, 20.0

driver = robot.ViperDCMotorControl()
rover = robot.Viper(sysMBS, robot.ViperWheelType_RealWheel)
rover.SetDriver(driver)
rover.SetWheelContactMaterial(wheel_mat.CreateMaterial(sysMBS.GetContactMethod()))
rover.Initialize(ch.ChFramed(init_loc, ch.QUNIT))

# CRM terrain
terrain = veh.CRMTerrain(sysMBS, initial_spacing)
sysFSI = terrain.GetFsiSystemSPH()
sysSPH = terrain.GetFluidSystemSPH()
sysSPH.EnableGPUErrorCheck(False)
terrain.SetVerbose(True)
terrain.SetGravitationalAcceleration(ch.ChVector3d(0, 0, -9.81))
terrain.SetStepSizeCFD(step_size)

mat_props = fsi.SoilProperties()
mat_props.Young_modulus = youngs_modulus
mat_props.Poisson_ratio = poisson_ratio
mat_props.mu_I0 = 0.04
mat_props.mu_fric_s = friction
mat_props.mu_fric_2 = friction
mat_props.average_diam = 0.005
mat_props.cohesion_coeff = cohesion
terrain.SetCrmSPH(mat_props)

sph_params = fsi.SPHParameters()
sph_params.integration_scheme = fsi.IntegrationScheme_RK2
sph_params.initial_spacing = initial_spacing
sph_params.d0_multiplier = 1.3
sph_params.free_surface_threshold = 2.0
sph_params.artificial_viscosity = 0.5
sph_params.use_consistent_gradient_discretization = False
sph_params.use_consistent_laplacian_discretization = False
sph_params.viscosity_method = fsi.ViscosityMethod_ARTIFICIAL_BILATERAL
sph_params.boundary_method = fsi.BoundaryMethod_ADAMI
sph_params.use_variable_time_step = True
terrain.SetSPHParameters(sph_params)
sysSPH.SetDensity(density)
terrain.SetOutputLevel(fsi.OutputLevel_STATE)

# Wheel BCE markers
mesh_filename = ch.GetChronoDataFile("robot/viper/obj/viper_cylwheel.obj")
geometry = ch.ChBodyGeometry()
geometry.materials.push_back(ch.ChContactMaterialData())
geometry.coll_meshes.push_back(
    ch.TrimeshShape(ch.VNULL, ch.QUNIT, mesh_filename, ch.VNULL)
)
# GetWheels() returns a std::array SWIG cannot subscript; use the ID accessor.
for wid in (robot.V_LF, robot.V_RF, robot.V_LB, robot.V_RB):
    terrain.AddRigidBody(rover.GetWheel(wid).GetBody(), geometry, False)

terrain.SetActiveDomain(active_box_dim)
terrain.Construct(
    ch.ChVector3d(terrain_length, terrain_width, 0.25),
    ch.ChVector3d(terrain_length / 2, 0, 0),
    fsi.BoxSide_ALL & ~fsi.BoxSide_Z_POS,
)
terrain.Initialize()

# Regolith sprite meshes — TRIANGLE MESHES, not primitives.
regolith_meshes = []
for i in range(1, num_meshes + 1):
    mmesh = ch.ChTriangleMeshConnected.CreateFromWavefrontFile(
        ch.GetChronoDataFile(f"models/regolith/particle_{i}.obj"), False, True
    )
    shape = ch.ChVisualShapeTriangleMesh()
    shape.SetMesh(mmesh)
    shape.SetName(f"RegolithMesh{i}")
    shape.SetMutable(False)
    regolith_meshes.append(shape)

regolith_material = ch.ChVisualMaterial()
regolith_material.SetAmbientColor(ch.ChColor(1, 1, 1))
regolith_material.SetDiffuseColor(ch.ChColor(1, 1, 1))
regolith_material.SetSpecularColor(ch.ChColor(1, 1, 1))
regolith_material.SetUseSpecularWorkflow(True)
regolith_material.SetRoughness(1.0)
regolith_material.SetHapkeParameters(
    0.32357, 0.23955, 0.30452, 1.80238, 0.07145, 0.3, 23.4 * ch.CH_DEG_TO_RAD
)
regolith_material.SetClassID(30000)
regolith_material.SetInstanceID(20000)
for mesh in regolith_meshes:
    if mesh.GetNumMaterials() == 0:
        mesh.AddMaterial(regolith_material)
    else:
        mesh.GetMaterials()[0] = regolith_material

manager = sens.ChSensorManager(sysMBS)
manager.scene.AddPointLight(
    ch.ChVector3f(0.5, 1, 1), ch.ChColor(1.0, 1.0, 1.0), 500
)
manager.scene.SetAmbientLight(ch.ChVector3f(0.1, 0.1, 0.1))
manager.SetVerbose(False)
manager.SetRayRecursions(4)

opts = sens.ChFsiSphRenderOptions()
for mesh in regolith_meshes:
    opts.sprite_shapes.append(mesh)
opts.sprite_position_jitter = ch.ChVector3f(0.005, 0.005, 0.0)
opts.render_particle_spacing = 0.01
# Read back rather than trusting assignment: a SWIG struct can accept a value
# that never reaches the C++ member.
print(
    f"GATE: options set  spacing={opts.render_particle_spacing:.4f}"
    f"  n_sprites={len(opts.sprite_shapes)}",
    flush=True,
)

fsi_render_handle = -1
if do_attach:
    fsi_render_handle = manager.AttachFsiSphSystem(sysSPH, opts)
    print(f"GATE: AttachFsiSphSystem handle = {fsi_render_handle}", flush=True)
    if fsi_render_handle < 0:
        print("GATE FAIL: attach returned a negative handle", file=sys.stderr)
        raise SystemExit(2)
else:
    print("GATE: control arm, AttachFsiSphSystem NOT called", flush=True)

floor = ch.ChBodyEasyBox(1, 1, 1, 1000, False, False)
floor.SetPos(ch.ChVector3d(0, 0, 0))
floor.SetFixed(True)
sysMBS.Add(floor)

offset_pose1 = ch.ChFramed(
    ch.ChVector3d(0.5, 1, 1), ch.QuatFromAngleAxis(-ch.CH_PI_2, ch.ChVector3d(0, 0, 1))
)
rot = ch.ChFramed(
    ch.ChVector3d(0, 0, 0),
    ch.QuatFromAngleAxis(60.0 / 180.0 * ch.CH_PI, ch.ChVector3d(0, 1, 0)),
)

cam = sens.ChCameraSensor(
    floor, render_fps, offset_pose1 * rot, 1280, 720, 1.5707963267948966, 1,
    sens.CameraLensModelType_PINHOLE, False,
)
cam.PushFilter(sens.ChFilterSave(out_dir + "img_SEN/"))
manager.AddSensor(cam)

time = 0.0
sim_frame = 0
render_frame = 0
exchange_info = 5 * step_size
while time < tend:
    rover.Update()
    if time >= render_frame / render_fps:
        render_frame += 1
    print(
        f"GATE_STEP t={time} sim_frame={sim_frame} render_frame={render_frame}"
        f" rtf_cfd={terrain.GetRtfCFD()} rtf_mbd={terrain.GetRtfMBD()}",
        flush=True,
    )
    terrain.DoStepDynamics(exchange_info)
    if time > 0:
        manager.Update()
    time += exchange_info
    sim_frame += 1

n_frames = len(
    [f for f in os.listdir(out_dir + "img_SEN") if os.path.isfile(out_dir + "img_SEN/" + f)]
)
print(
    f"GATE: arm={gate_tag} handle={fsi_render_handle} frames_written={n_frames}"
    f" sim_frames={sim_frame} final_time={time}",
    flush=True,
)
if n_frames == 0:
    print("GATE VOID: no frames written", file=sys.stderr)
    raise SystemExit(3)
