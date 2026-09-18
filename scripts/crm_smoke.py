"""CRM smoke test on the f104 arena heightmap: particle count, heightmap orientation, throughput.

Builds a CRMTerrain from the arena BMP (optionally y-flipped), drops an HMMWV on it, drives a straight path with
Chrono's path follower and reports the real-time factor. With --check-surface it dumps the initial SPH markers and
compares the top surface with the repo's TerrainMap for the BMP as-is and y-flipped.

  luffy:   PYTHONPATH=/home/harry/chrono/build/bin:src /usr/bin/python3.12 scripts/crm_smoke.py ...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pychrono as chrono  # noqa: E402
import pychrono.fsi as fsi  # noqa: E402
import pychrono.vehicle as veh  # noqa: E402

from nedm.hmmwv_data import configure_chrono_data_paths, create_hmmwv  # noqa: E402
from nedm.traverse.scene import build_config  # noqa: E402
from nedm.traverse.terrain import TerrainMap  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--arena', default=str(ROOT / 'assets/traverse/arena_f104_50h_v1'))
    ap.add_argument('--spacing', type=float, default=0.08)
    ap.add_argument('--depth', type=float, default=0.24)
    ap.add_argument('--step', type=float, default=5e-4)
    ap.add_argument('--active', type=float, nargs=3, default=[2.0, 2.0, 1.0])
    ap.add_argument('--start', type=float, nargs=2, default=[-30.0, -30.0])
    ap.add_argument('--goal', type=float, nargs=2, default=[0.0, -30.0])
    ap.add_argument('--speed', type=float, default=4.0)
    ap.add_argument('--duration', type=float, default=4.0)
    ap.add_argument('--flip-bmp', action='store_true', help='feed CRM a y-flipped copy of the BMP')
    ap.add_argument('--check-surface', action='store_true')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--d0', type=float, default=1.2)
    ap.add_argument('--out', default='')
    ap.add_argument('--chrono-data', default='')
    ap.add_argument('--crop', type=float, nargs=4, default=None, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'))
    args = ap.parse_args()

    arena = Path(args.arena)
    tmap = TerrainMap.from_dir(arena)
    meta = tmap.meta
    sx, sy = args.start
    gx, gy = args.goal
    yaw = math.atan2(gy - sy, gx - sx)
    z0 = float(tmap.height(sx, sy)) + 0.9
    config = build_config(arena, (sx, sy, z0), yaw, step_size_s=args.step, tire_step_size_s=args.step)
    config['vehicle']['tire_model'] = 'RIGID_MESH'
    config['vehicle']['chassis_collision'] = 'NONE'
    if args.chrono_data:
        config['chrono_data_root'] = str(Path(args.chrono_data).resolve())
        config['vehicle_data_root'] = str(Path(args.chrono_data).resolve() / 'vehicle')
    configure_chrono_data_paths(ROOT, config)
    hmmwv = create_hmmwv(config)
    vehicle = hmmwv.GetVehicle()
    system = hmmwv.GetSystem()
    system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
    system.SetNumThreads(args.threads, 1, 1)

    bmp = arena / meta['bmp']
    if args.flip_bmp:
        img = Image.open(bmp)
        flipped = Path(tempfile.mkdtemp()) / 'arena_flipped.bmp'
        img.transpose(Image.FLIP_TOP_BOTTOM).save(flipped)
        bmp = flipped

    t_build = time.time()
    terrain = veh.CRMTerrain(system, args.spacing)
    terrain.SetVerbose(True)
    terrain.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    terrain.SetStepSizeCFD(args.step)
    terrain.RegisterVehicle(vehicle)

    soil = fsi.SoilProperties()
    soil.density = 1700.0
    soil.Young_modulus = 1e6
    soil.Poisson_ratio = 0.3
    soil.mu_I0 = 0.04
    soil.mu_fric_s = 0.8
    soil.mu_fric_2 = 0.8
    soil.average_diam = 0.005
    soil.cohesion_coeff = 5e3
    terrain.SetCrmSPH(soil)

    sph = fsi.SPHParameters()
    sph.integration_scheme = fsi.IntegrationScheme_RK2
    sph.initial_spacing = args.spacing
    sph.d0_multiplier = args.d0
    sph.free_surface_threshold = 0.8
    sph.artificial_viscosity = 0.5
    sph.shifting_method = fsi.ShiftingMethod_PPST
    sph.shifting_ppst_push = 3.0
    sph.shifting_ppst_pull = 1.0
    sph.use_consistent_gradient_discretization = False
    sph.use_consistent_laplacian_discretization = False
    sph.viscosity_method = fsi.ViscosityMethod_ARTIFICIAL_BILATERAL
    sph.boundary_method = fsi.BoundaryMethod_ADAMI
    sph.num_proximity_search_steps = 4
    terrain.SetSPHParameters(sph)

    mesh = veh.GetVehicleDataFile('hmmwv/hmmwv_tire_coarse_closed.obj')
    geometry = chrono.ChBodyGeometry()
    geometry.coll_meshes.append(chrono.TrimeshShape(chrono.VNULL, chrono.QUNIT, mesh, chrono.VNULL))
    for axle in vehicle.GetAxles():
        for wheel in axle.GetWheels():
            terrain.AddRigidBody(wheel.GetSpindle(), geometry, False)

    terrain.SetActiveDomain(chrono.ChVector3d(*args.active))
    terrain.SetActiveDomainDelay(0.1) if hasattr(terrain, 'SetActiveDomainDelay') else None
    size = float(meta['size_m'])
    length, width, centre = size, size, (0.0, 0.0)
    if args.crop:
        raw = Image.open(bmp)
        n = raw.size[0]
        dx = size / (n - 1)  # Chrono places pixel nodes on the patch edges
        xa, xb, ya, yb = args.crop
        c0, c1 = int(round((xa + size / 2) / dx)), int(round((xb + size / 2) / dx))
        r0, r1 = int(round((size / 2 - yb) / dx)), int(round((size / 2 - ya) / dx))  # image row 0 = +y
        cropped = Path(tempfile.mkdtemp()) / 'arena_crop.bmp'
        raw.crop((c0, r0, c1 + 1, r1 + 1)).save(cropped)
        bmp = cropped
        length, width = (c1 - c0) * dx, (r1 - r0) * dx
        centre = (-size / 2 + 0.5 * (c0 + c1) * dx, size / 2 - 0.5 * (r0 + r1) * dx)
        print('CROP', c0, c1, r0, r1, length, width, centre)
    terrain.Construct(
        str(bmp), length, width,
        chrono.ChVector2d(float(meta['height_min_m']), float(meta['height_max_m'])),
        args.depth, True, chrono.ChVector3d(centre[0], centre[1], 0),
        fsi.BoxSide_Z_NEG,
    )
    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix='crm_smoke_'))
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.check_surface:
        terrain.SaveInitialMarkers(str(out_dir))
    terrain.Initialize()
    t_build = time.time() - t_build
    aabb = terrain.GetSPHBoundingBox()
    report = {
        'spacing': args.spacing, 'depth': args.depth, 'step': args.step, 'active': args.active,
        'n_sph': int(terrain.GetNumSPHParticles()), 'n_bce_boundary': int(terrain.GetNumBoundaryBCEMarkers()),
        'build_s': t_build, 'aabb_min': [aabb.min.x, aabb.min.y, aabb.min.z],
        'aabb_max': [aabb.max.x, aabb.max.y, aabb.max.z], 'flip_bmp': bool(args.flip_bmp),
    }
    print('REPORT_BUILD', json.dumps(report), flush=True)

    if args.check_surface:
        files = sorted(out_dir.glob('*'))
        print('marker files', [f.name for f in files])
        for f in files:
            if True:
                if f.name != 'sph_grid.txt':
                    continue
                grid = np.loadtxt(f, dtype=np.int64)  # integer lattice coordinates (Ix, Iy, Iz)
                n_xy = int(round(size / args.spacing))
                data = np.column_stack([-size / 2 + grid[:, 0] * (size / n_xy), -size / 2 + grid[:, 1] * (size / n_xy),
                                        grid[:, 2] * args.spacing])
                key = np.round(data[:, :2] / args.spacing).astype(np.int64)
                order = np.lexsort((data[:, 2], key[:, 1], key[:, 0]))
                d = data[order]
                k = key[order]
                last = np.r_[(k[1:] != k[:-1]).any(axis=1), True]
                top = d[last]
                inner = (np.abs(top[:, 0]) < 36) & (np.abs(top[:, 1]) < 36)
                top = top[inner]
                for name, yy in (('as_is', top[:, 1]), ('y_mirrored', -top[:, 1])):
                    err = top[:, 2] - tmap.height(top[:, 0], yy)
                    print(f'SURFACE {f.name} {name}: rmse {np.sqrt(np.mean(err**2)):.4f} m, bias {err.mean():+.4f} m, n={len(err)}')

    path = veh.StraightLinePath(chrono.ChVector3d(sx, sy, z0), chrono.ChVector3d(gx, gy, z0), 1)
    driver = veh.ChPathFollowerDriver(vehicle, path, 'smoke_path', args.speed)
    driver.GetSteeringController().SetLookAheadDistance(5.0)
    driver.GetSteeringController().SetGains(0.8, 0.0, 0.0)
    driver.GetSpeedController().SetGains(0.4, 0.0, 0.0)
    driver.Initialize()

    sim_t = 0.0
    n_steps = int(round(args.duration / args.step))
    t0 = time.time()
    t_mark = t0
    log = []
    for i in range(n_steps):
        inputs = driver.GetInputs()
        if sim_t < 0.8:
            inputs.m_throttle = 0.0
            inputs.m_braking = 1.0
        driver.Synchronize(sim_t)
        terrain.Synchronize(sim_t)
        vehicle.Synchronize(sim_t, inputs, terrain)
        driver.Advance(args.step)
        terrain.Advance(args.step)
        sim_t += args.step
        if (i + 1) % int(round(0.5 / args.step)) == 0:
            now = time.time()
            p = vehicle.GetPos()
            rec = {'t': round(sim_t, 3), 'x': p.x, 'y': p.y, 'z': p.z, 'ground': float(tmap.height(p.x, p.y)),
                   'speed': vehicle.GetSpeed(), 'rtf_window': 0.5 / (now - t_mark), 'throttle': inputs.m_throttle}
            t_mark = now
            log.append(rec)
            print('STEP', json.dumps(rec), flush=True)
    wall = time.time() - t0
    report.update({'sim_s': sim_t, 'wall_s': wall, 'rtf': sim_t / wall, 'rtf_cfd': terrain.GetRtfCFD(),
                   'rtf_mbd': terrain.GetRtfMBD(), 'log': log})
    print('REPORT_FINAL', json.dumps({k: v for k, v in report.items() if k != 'log'}), flush=True)
    (out_dir / 'smoke_report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
