"""The Chrono world of one drive: terrain, vehicle, path follower, clock and measurements.

Ports of the frozen collectors at 901d6c9 (rigid: traverse_fdm_rgbd_diverse_chrono.run_chrono as gen_collect(_ext) ran
it; soil: crm_collect(_ext).py), statement order kept, which is what makes recorded drives replay bit for bit.
Clock: rigid GetChTime(), 25 x 2 ms (tyres 1 ms), terrain then vehicle Advance; soil a Python sum of the config step
(crm_main 1 ms: 50 substeps; tyre = MBS = CFD step), terrain.Advance only (crm_collect.py:225-271).
TerrainMap: pixel-centre bilinear BMP heights (terrain.py:260-329), 511/512 off Chrono's node-on-edge grid, used as is.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from nedm.traversing.training.state import STATE_FIELDS

from .config import DT, REPO_ROOT, ConfigError, sha256_file
from .vehicles import Belly, hmmwv_data, show

RIGID_STEP_S, RIGID_TIRE_STEP_S = 0.002, 0.001
TEXTURE = REPO_ROOT / 'chrono/data/sensor/textures/grass_texture.jpg'      # scene.py:399-403 (none on the cluster)
# B8 (n_sph, n_bce) of the 0.08 m / 0.24 m walled CRM per arena: class on chrono-build-fsi == every released outcome.json
CRM_COUNTS = {f'arena_{a}': (4008004, b) for a, b in dict(
    f104_50h_v1=3981891, g203=4054179, g228=4078275, g241=3993939, g247=4078275, g251=3993939, g258=4090323,
    g260=4030083, g263=3957795, g268=4150563, g271=3993939).items()}


class LaunchError(RuntimeError):
    """The settled vehicle failed the frame-0 launch check: no label, not retried (deterministic)."""


class FellThrough(RuntimeError):
    """The chassis dropped more than 1 m below the BMP on soil: invalid physics, never a label; the runner retries."""


def build_crm(chrono, veh, fsi, system, vehicle, bmp, meta, c, wheels=None):
    """crm_collect.build_crm (getattr fallbacks: the pychrono 10.0.0 spellings, smoke tests only); `wheels`: one FSI
    geometry per axle in place of the tyre mesh (ag_vehicle.make_build_crm), the mesh geometry still built."""
    system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
    system.SetNumThreads(int(c['mbs_threads']), 1, 1)
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    terrain = veh.CRMTerrain(system, float(c['spacing_m']))
    terrain.SetVerbose(False)
    terrain.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    terrain.SetStepSizeCFD(float(c['step_s']))
    terrain.RegisterVehicle(vehicle)
    s, soil = c['soil'], (getattr(fsi, 'SoilProperties', None) or fsi.ElasticMaterialProperties)()
    soil.density, soil.Young_modulus = float(s['density']), float(s['young_modulus_pa'])
    soil.Poisson_ratio, soil.mu_I0 = float(s['poisson_ratio']), float(s['mu_I0'])
    soil.mu_fric_s = soil.mu_fric_2 = float(s['friction'])
    soil.average_diam, soil.cohesion_coeff = float(s['average_diam_m']), float(s['cohesion_pa'])
    (getattr(terrain, 'SetCrmSPH', None) or terrain.SetElasticSPH)(soil)
    p, sph = c['sph'], fsi.SPHParameters()
    sph.integration_scheme = fsi.IntegrationScheme_RK2
    sph.initial_spacing = float(c['spacing_m'])
    sph.d0_multiplier, sph.free_surface_threshold = float(p['d0_multiplier']), float(p['free_surface_threshold'])
    sph.artificial_viscosity = float(p['artificial_viscosity'])
    sph.shifting_method = getattr(fsi, 'ShiftingMethod_' + str(p['shifting_method']).upper())
    sph.shifting_ppst_push, sph.shifting_ppst_pull = float(p['shifting_ppst_push']), float(p['shifting_ppst_pull'])
    sph.use_consistent_gradient_discretization = sph.use_consistent_laplacian_discretization = False
    sph.viscosity_method = fsi.ViscosityMethod_ARTIFICIAL_BILATERAL
    sph.boundary_method = fsi.BoundaryMethod_ADAMI
    sph.num_proximity_search_steps = int(p['num_proximity_search_steps'])
    terrain.SetSPHParameters(sph)
    geometry = chrono.ChBodyGeometry()
    geometry.coll_meshes.append(chrono.TrimeshShape(chrono.VNULL, chrono.QUNIT, veh.GetVehicleDataFile(c['tire_mesh']),
                                                    chrono.VNULL))
    for i, axle in enumerate(vehicle.GetAxles()):
        for wheel in axle.GetWheels():
            terrain.AddRigidBody(wheel.GetSpindle(), wheels[i] if wheels else geometry, False)
    terrain.SetActiveDomain(chrono.ChVector3d(*[float(v) for v in c['active_domain_m']]))
    (getattr(terrain, 'SetFreeFlowDuration', None) or terrain.SetActiveDomainDelay)(float(c['active_domain_delay_s']))
    size = float(meta['size_m'])
    sides = (fsi.BoxSide_ALL & ~fsi.BoxSide_Z_POS) if c['side_walls'] else fsi.BoxSide_Z_NEG
    terrain.Construct(str(bmp), size, size, chrono.ChVector2d(float(meta['height_min_m']), float(meta['height_max_m'])),
                      float(c['depth_m']), True, chrono.ChVector3d(0, 0, 0), sides)
    terrain.Initialize()
    return terrain


class TerrainMap:
    """The arena BMP as heights: h = hmin + gray / 255 (hmax - hmin) after the calibrated orientation; bilinear in
    world (x, y) at pixel centres, clamped to the grid (terrain.TerrainMap)."""

    def __init__(self, arena_dir):
        self.meta = json.loads((Path(arena_dir) / 'arena_meta.json').read_text())
        raw = np.asarray(Image.open(Path(arena_dir) / self.meta['bmp']).convert('L'), dtype=np.float64)
        o = self.meta.get('orientation', {})
        g = np.rot90(raw, int(o.get('rot90', 0)))
        g = np.flipud(g) if o.get('flipud', False) else g
        self.grid = self.meta['height_min_m'] + g / 255.0 * (self.meta['height_max_m'] - self.meta['height_min_m'])
        if self.grid.shape[0] != self.grid.shape[1]:
            raise ValueError(f'{arena_dir}: the BMP is not square')
        self.n = self.grid.shape[0]
        self.res, self.half = float(self.meta['size_m']) / self.n, float(self.meta['size_m']) / 2.0

    def height(self, x, y):
        fx = np.clip((np.asarray(x) + self.half) / self.res - 0.5, 0.0, self.n - 1.001)
        fy = np.clip((np.asarray(y) + self.half) / self.res - 0.5, 0.0, self.n - 1.001)
        ix0, iy0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        ix1, iy1 = np.minimum(ix0 + 1, self.n - 1), np.minimum(iy0 + 1, self.n - 1)
        ax, ay, h = fx - ix0, fy - iy0, self.grid
        top = h[iy0, ix0] * (1 - ax) + h[iy0, ix1] * ax
        bot = h[iy1, ix0] * (1 - ax) + h[iy1, ix1] * ax
        return top * (1 - ay) + bot * ay


@dataclass(frozen=True)
class After:
    """The chassis reference after a frame's physics: what goal, rollover and the stop rules read."""
    pose: np.ndarray                                    # (x, y, CardanZYX.z) f64
    roll: float
    pitch: float
    sinkage: float | None = None                        # soil: deepest wheel below the BMP (m, stock tyre radius)


@dataclass(eq=False)
class Sim:
    """One Chrono world on rigid ground (soil None) or on CRM soil (soil = the loaded soil config)."""
    case: dict
    tmap: TerrainMap
    model: object                                       # the vehicle wrapper (vehicles.Vehicle.create)
    terrain: object
    dt: float
    soil: dict | None = None
    launch: dict = field(default_factory=dict)          # the frame-0 launch report (provenance; B8 fingerprint)
    info: dict = field(default_factory=dict)            # scene provenance: BMP sha256, patch look | CRM counts
    vehicle_info: dict = field(default_factory=dict)    # vehicles.Vehicle.info (provenance)
    belly: object = None                                # vehicles.Belly of a vehicle with belly points

    @classmethod
    def build(cls, case, arena_dir, vehicle, chrono_data, soil=None):
        """The vehicle at the layout pose at its spawn height, then rigid: MESH visuals, the roof marker and a
        RigidTerrain BMP patch (textured if the repo has chrono/'s grass texture; visual only, scene.py:345-403); soil:
        build_crm with the vehicle's soil wheels, the CRM counts of a pinned arena checked."""
        import pychrono as chrono
        if soil is not None:
            import pychrono.fsi as fsi
        import pychrono.vehicle as veh
        import pychrono.sensor  # noqa: F401  (scene.py imported it at module level)
        hmmwv_data()                                    # the source pins first: a ConfigError before any Chrono object
        tmap, (x, y), v = TerrainMap(arena_dir), case['layout']['start_xy'], vehicle
        z, bmp, m = v.spawn(float(tmap.height(x, y))), Path(arena_dir) / tmap.meta['bmp'], tmap.meta
        root = Path(chrono_data).resolve()              # configure_chrono_data_paths with absolute paths
        hmmwv_data().configure_chrono_data_paths(root, dict(chrono_data_root=str(root),
                                                            vehicle_data_root=str(root / 'vehicle')))
        dt = RIGID_STEP_S if soil is None else float(soil['step_s'])
        model = v.create((x, y, z), case['layout']['start_yaw'], tire_step=RIGID_TIRE_STEP_S if soil is None else dt,
                         soil=soil is not None)
        info = dict(arena_bmp_sha256=sha256_file(bmp))
        kw = dict(vehicle_info=v.info(model), belly=v.belly and Belly(v.files['belly_points'], tmap), info=info)
        if soil is not None:
            terrain = build_crm(chrono, veh, fsi, model.GetSystem(), model.GetVehicle(), bmp, m, soil, v.soil_wheels(chrono))
            n, pin = (terrain.GetNumSPHParticles(), terrain.GetNumBoundaryBCEMarkers()), CRM_COUNTS.get(bmp.parent.name)
            pin = pin if (soil['spacing_m'], soil['depth_m'], soil['side_walls']) == (0.08, 0.24, True) else None
            if pin and n != pin:                        # deterministic: a refusal, never a crash record
                raise ConfigError([f'CRM of {arena_dir}: (n_sph, n_bce) {n}, recorded {pin}: another build or BMP'])
            info.update(n_sph=n[0], n_bce=n[1], crm_counts_pinned=pin is not None)
            return cls(case, tmap, model, terrain, dt, soil, **kw)
        show(chrono, model)
        terrain = veh.RigidTerrain(model.GetSystem())
        pm = chrono.ChContactMaterialSMC()
        pm.SetFriction(0.9)
        pm.SetRestitution(0.01)
        pm.SetYoungModulus(2.0e7)
        patch = terrain.AddPatch(pm, chrono.CSYSNORM, str(bmp), float(m['size_m']), float(m['size_m']),
                                 float(m['height_min_m']), float(m['height_max_m']))
        if TEXTURE.is_file():
            patch.SetTexture(str(TEXTURE), 40.0, 40.0)
        else:
            patch.SetColor(chrono.ChColor(0.42, 0.5, 0.32))
        info['patch'] = str(TEXTURE) if TEXTURE.is_file() else 'solid_color'
        terrain.Initialize()
        return cls(case, tmap, model, terrain, dt, **kw)

    def __post_init__(self):
        self.ground, self.substeps, self.t = 'rigid' if self.soil is None else 'soil', int(round(DT / self.dt)), 0.
        self.breakthrough_m = self.soil and (       # soil depth + margin (default 0.06 m, crm_collect.py:289)
            float(self.soil['depth_m']) + float(self.soil.get('breakthrough_margin_m', .06)))
        self.system, self.vehicle = self.model.GetSystem(), self.model.GetVehicle()
        self.chassis = self.model.GetChassis().GetBody()
        self.engine, self.transmission = self.vehicle.GetEngine(), self.vehicle.GetTransmission()
        self.wheels = hmmwv_data().WHEEL_SPECS
        self.tire_radii = {n: float(self.vehicle.GetTire(axle, side).GetRadius()) for n, axle, side in self.wheels}

    def follower(self, route, *, initialize, z=None):
        """The path follower on `route`: points >= 2 m apart (the last kept) at the TerrainMap height (M1: the sensed
        heights `z`, one per waypoint) + 0.5 m."""
        import pychrono as chrono
        import pychrono.vehicle as veh
        pts, last, st = chrono.vector_ChVector3d(), -10.0, route['stations']
        for i, ((x, y), s) in enumerate(zip(route['waypoints'], st)):
            if s - last < 2.0 and s != st[-1]:
                continue
            last = s
            h = float(self.tmap.height(x, y)) if z is None else float(z[i])
            pts.append(chrono.ChVector3d(float(x), float(y), h + 0.5))
        fol = veh.ChPathFollowerDriver(self.vehicle, chrono.ChBezierCurve(pts), 'route', float(route['speeds'][0]))
        fol.GetSteeringController().SetLookAheadDistance(5.0)
        fol.GetSteeringController().SetGains(0.8, 0.0, 0.0)
        fol.GetSpeedController().SetGains(0.6, 0.05, 0.0)
        if initialize:
            fol.Initialize()
        return fol

    def now(self):
        return float(self.system.GetChTime()) if self.soil is None else self.t

    def ref_xy(self):
        p = self.chassis.GetFrameRefToAbs().GetPos()
        return np.array([p.x, p.y])

    def sync(self, t, u):
        self.terrain.Synchronize(t)
        self.model.Synchronize(t, u, self.terrain)

    def advance(self):
        self.terrain.Advance(self.dt)
        if self.soil is None:
            self.model.Advance(self.dt)
        self.t += self.dt                               # the soil clock (rigid reads GetChTime)

    def power_kw(self):
        return float(self.engine.GetOutputMotorshaftTorque()) * float(self.transmission.GetOutputMotorshaftSpeed()) / 1000.

    def crm_wheels(self):
        """The crm_collect.crm_tire_fields that are read (FSI spindle force in the wheel frame, x = spin axis x up)."""
        import pychrono as chrono
        up, row = chrono.ChVector3d(0, 0, 1), {}
        for n, axle, side in self.wheels:
            force = self.terrain.GetFsiBodyForce(self.vehicle.GetWheel(axle, side).GetSpindle())
            spin = self.vehicle.GetSpindleRot(axle, side).GetAxisY()
            heading = spin.Cross(up).GetNormalized()
            w = float(self.vehicle.GetSpindleAngVel(axle, side).Dot(spin))
            vx = float(self.vehicle.GetSpindleLinVel(axle, side).Dot(heading))
            row.update({f'{n}_force_wheel_fx_n': float(force.Dot(heading)), f'{n}_force_wheel_fz_n': float(force.Dot(up)),
                        f'{n}_spindle_omega_radps': w, f'{n}_slip_ratio': (w * self.tire_radii[n] - vx) / max(abs(vx), .1),
                        f'{n}_spindle_z_m': float(self.vehicle.GetSpindlePos(axle, side).z)})
        return row

    def measure(self, k, t, u):
        """(state f32, action f32, pose f64, power kW[, crm_extra row]) after vehicle.Synchronize."""
        soil = self.soil is not None
        row = hmmwv_data().capture_row(self.model, self.terrain, 'crm_f104' if soil else 'fdm_rgbd_eval', 'follower',
                                       self.case['id'], 'development', k, t, u, include_tires=not soil,
                                       tire_radii=None if soil else self.tire_radii)
        if soil:
            row.update(self.crm_wheels())
        row['engine_motor_speed_radps'] = float(self.engine.GetMotorSpeed())
        row['engine_motorshaft_torque_nm'] = float(self.engine.GetOutputMotorshaftTorque())
        if self.belly is not None and k >= 0:          # the wrappers' frame hook (ag_crm_collect.py:83-84)
            self.belly.on_frame(self.chassis)
        state = np.array([float(row[f]) for f in STATE_FIELDS], np.float32)
        pose = np.array([row['pos_x_m'], row['pos_y_m'], row['yaw_rad']], np.float64)
        m = state, np.array([u.m_steering, u.m_throttle, u.m_braking], np.float32), pose, self.power_kw()
        if k >= 0 and not (np.isfinite(state).all() and np.isfinite(pose).all()):
            raise FloatingPointError(f'non-finite vehicle state at frame {k}')
        if not soil:
            return m
        return *m, [row['pos_z_m'], float(self.tmap.height(row['pos_x_m'], row['pos_y_m'])),
                    *(row[f'quat_e{i}'] for i in range(4)), *(row[f'{n}_{f}'] for f in (
                        'spindle_z_m', 'slip_ratio', 'force_wheel_fx_n') for n, _, _ in self.wheels)]

    def crm_extra(self, rows) -> dict:
        """crm_extra.npz of the recorded rows (crm_collect.py:339-342)."""
        x = np.asarray(rows, np.float32).reshape(-1, 18)
        return dict(pos_z_m=x[:, 0], bmp_ground_z_m=x[:, 1], quat=x[:, 2:6], spindle_z_m=x[:, 6:10],
                    slip_ratio=x[:, 10:14], fsi_force_wheel_fx_n=x[:, 14:18],
                    wheel_order=np.asarray(list(self.tire_radii)), tire_radius_m=np.asarray(list(self.tire_radii.values())))

    def chassis_pose(self):
        """(pose (x, y, CardanZYX.z), z) of the chassis reference frame now."""
        ref = self.chassis.GetFrameRefToAbs()
        p = ref.GetPos()
        return np.array([p.x, p.y, ref.GetRot().GetCardanAnglesZYX().z]), float(p.z)

    def after_frame(self):
        pose, z = self.chassis_pose()
        if not np.isfinite(pose).all():
            raise FloatingPointError(f'non-finite chassis pose {pose}')
        sink = None
        if self.soil is not None:
            if z < float(self.tmap.height(pose[0], pose[1])) - 1.0:
                raise FellThrough(f'the vehicle fell through the soil at {pose}')
            sink = max(self.tire_radii[n] - (float((p := self.vehicle.GetSpindlePos(a, s)).z)
                                             - float(self.tmap.height(p.x, p.y))) for n, a, s in self.wheels)
        return After(pose, float(self.vehicle.GetRoll()), float(self.vehicle.GetPitch()), sink)

    def launch_check(self, ep):
        """The settled state at frame 0 (rigid: gen_collect.StopPolicy.on_anchor; soil: CrmPolicy.on_anchor_crm)."""
        lay, state, pose, soil = self.case['layout'], ep.state[0], ep.pose[0], self.soil is not None
        dyaw = pose[2] - lay['start_yaw']
        rep = dict(speed_mps=float(np.linalg.norm(np.asarray(state)[:2])), roll_rad=float(state[2]),
                   pitch_rad=float(state[3]), yaw_error_rad=float(math.atan2(math.sin(dyaw), math.cos(dyaw))),
                   start_xy_error_m=float(np.linalg.norm(np.asarray(pose[:2]) - lay['start_xy'])))
        ok = dict(finite=bool(np.isfinite(state).all() and np.isfinite(pose).all()), speed=rep['speed_mps'] <= 1.,
                  tilt=max(abs(rep['roll_rad']), abs(rep['pitch_rad'])) <= math.radians(25. if soil else 20.),
                  yaw=abs(rep['yaw_error_rad']) <= math.radians(10.), start=rep['start_xy_error_m'] <= 1.)
        if soil:
            rep['clearance_m'] = float(ep.extra[0][0] - ep.extra[0][1])
            ok['clearance'] = 0. <= rep['clearance_m'] <= 1.2
        else:
            import pychrono as chrono
            grid = np.linspace(-36., 36., 7)
            pts = [(float(x), float(y)) for x in grid for y in grid] + [tuple(lay['start_xy'])]
            native = np.asarray([self.terrain.GetHeight(chrono.ChVector3d(x, y, 20.)) for x, y in pts])
            err = np.abs(native - np.asarray([self.tmap.height(x, y) for x, y in pts]))
            rep.update(native_height_m=native.tolist(), p95_abs_error_m=float(np.quantile(err, .95)),
                       max_abs_error_m=float(err.max()))
            ok['native_height'] = bool(np.isfinite(native).all() and rep['p95_abs_error_m'] <= .08
                                       and rep['max_abs_error_m'] <= .15)
        fails = [k for k, v in ok.items() if not v]
        self.launch = dict(rep, passed=not fails, failed=fails)
        if fails:
            raise LaunchError(f'launch check failed ({fails}): {rep}')

    def terminal(self, u):
        """The state at the end of the last frame: Synchronize(now, last inputs), no Advance."""
        t = self.now()
        self.sync(t, u)
        return self.measure(-1, t, u)[0]
