"""Vehicles of the traversing evaluation: the HMMWV, the Gator (ag_vehicle.py at 901d6c9) and the re-framed JSON
Polaris with its driveline and soil-wheel variants (ov_vehicle.py; soil only; its JSON puts the chassis reference at
mid-wheelbase); config refuses the M113. ``load`` checks a vehicle's files (ValueError or OSError: runner.drive
refuses). ``capture_row``, ``WHEEL_SPECS`` and ``configure_chrono_data_paths`` come from the published HMMWV module
behind a source pin (``hmmwv_data()``); main's create_hmmwv would silently drop the HULLS chassis of every rigid drive.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import tempfile
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path

import numpy as np

from .config import ConfigError, sha256_file

SPAWN_DZ_M = 0.75                   # traverse_fdm_rgbd_diverse_chrono.py:162, crm_collect.py:177
MARKER_RGB = (0.05, 0.2, 1.0)       # rigid roof marker (scene.py:48), visual only
VISUALS = ('Chassis', 'Suspension', 'Steering', 'Wheel', 'Tire')
OV = 'data:assets/traverse/vehicles/ov_polaris/'
ENGINE, TIRE = 'Polaris/Polaris_EngineSimpleMap.json', 'Polaris/Polaris_RigidTire.json'   # soil RIGID_MESH -> rigid
STOCK_TX = 'Polaris/Polaris_AutomaticTransmissionSimpleMap.json'
GATOR_BELLY = ('configs/traversing/evaluation/ag_gator_belly.json',       # not released: scripts/ag_gator_belly.json
               '0c25a57cc77011b71eb7bd31dd6a1b926581f52efacccba3bfd4d20f54d0ded0')
# sha256 of inspect.getsource (functions) / repr (WHEEL_SPECS) of the reused names; equal to the 901d6c9 branch copies
PINS = dict(capture_row='7b6a9b096652f55200111f3057f045d467de50211f48a5ba82997630f33e9d2b',
            configure_chrono_data_paths='1add872fe0be03079c03176c2703e9372bfb360228017287ca8bbaf8b1489304',
            WHEEL_SPECS='4714f2f8aabe9027bbf93b93e791fcd50066adba81598282ae29169697a87f37')


@cache
def hmmwv_data():
    """The published HMMWV module, the source of every reused name checked against PINS (a mismatch: no drive)."""
    from nedm.hmmwv import hmmwv_data as m
    got = {n: hashlib.sha256((inspect.getsource(v) if callable(v := getattr(m, n)) else repr(v)).encode()).hexdigest()
           for n in PINS}
    if bad := sorted(n for n in PINS if got[n] != PINS[n]):
        raise ConfigError([f'nedm.hmmwv.hmmwv_data: {bad} changed since the drives were recorded (sha256 {got}); '
                           'a changed capture_row or data-path rule changes the recorded state'])
    return m


@dataclass(frozen=True)
class Vehicle:
    name: str
    dz: float = SPAWN_DZ_M              # spawn height above the BMP
    wheels: tuple = ()                  # soil cylinders (radius, width) per axle, front first; () = the tyre mesh
    belly: tuple | None = None          # belly points (chassis frame): (reference, pinned sha256)
    json_files: tuple = ()              # JSON Polaris: (vehicle, gearbox); Polaris_ov/ released, Polaris/ the build's
    files: dict = field(default_factory=dict, compare=False)     # set by load()

    def spawn(self, h):
        """Spawn height over the BMP height h: HMMWV h + 0.75, the others ((h + 0.75) - 0.75) + dz (ag_vehicle.py:165)."""
        z = h + SPAWN_DZ_M
        return z if self.name == 'hmmwv' else z - SPAWN_DZ_M + self.dz

    def create(self, xyz, yaw, *, tire_step, soil=False):
        """The wrapper the collectors called `hmmwv` (GetVehicle, GetSystem, GetChassis, Synchronize, Advance) at xyz
        heading yaw, Initialize()d, visuals NONE, Bullet collision: HMMWV_Full / Gator in create_hmmwv's setter order
        (rigid TMEASY + HULLS chassis, soil RIGID_MESH + no chassis collision, crm_collect.py:176-180); the Polaris from
        its JSON files (ov_vehicle.py:231-336) in a temporary vehicle-data root of Polaris_ov/ + the build's Polaris/."""
        import pychrono as chrono
        import pychrono.vehicle as veh
        if self.json_files:
            v = self._polaris(chrono, veh, xyz, yaw, tire_step)
        else:
            v = veh.HMMWV_Full() if self.name == 'hmmwv' else veh.Gator()
            v.SetContactMethod(chrono.ChContactMethod_SMC)
            v.SetChassisFixed(False)
            v.SetInitPosition(chrono.ChCoordsysd(chrono.ChVector3d(*xyz), chrono.QuatFromAngleZ(float(yaw))))
            if self.name == 'hmmwv':
                v.SetEngineType(veh.EngineModelType_SHAFTS)
                v.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SHAFTS)
                v.SetDriveType(veh.DrivelineTypeWV_AWD)
                v.SetSteeringType(veh.SteeringTypeWV_PITMAN_ARM)
            else:
                v.SetDrivelineType(veh.DrivelineTypeWV_SIMPLE)
                v.SetBrakeType(veh.BrakeType_SIMPLE)
                v.EnableBrakeLocking(False)
            v.SetTireType(veh.TireModelType_RIGID_MESH if soil else veh.TireModelType_TMEASY)
            v.SetTireStepSize(tire_step)
            v.SetChassisCollisionType(veh.CollisionType_NONE if soil else veh.CollisionType_HULLS)
            v.Initialize()
        for part in VISUALS:
            getattr(v, f'Set{part}VisualizationType')(chrono.VisualizationType_NONE)
        v.GetSystem().SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        return v

    def _polaris(self, chrono, veh, xyz, yaw, tire_step):
        build = veh.GetVehicleDataFile('')
        with tempfile.TemporaryDirectory(prefix='polaris_vehicle_data_') as d:
            root = Path(d)
            (root / 'Polaris_ov').symlink_to(self.files['assets'] / 'Polaris_ov')
            (root / 'Polaris').symlink_to(Path(build) / 'Polaris')
            veh.SetVehicleDataPath(f'{root}/')
            try:
                v = veh.WheeledVehicle(str(root / self.json_files[0]), chrono.ChContactMethod_SMC)
                v.Initialize(chrono.ChCoordsysd(chrono.ChVector3d(*xyz), chrono.QuatFromAngleZ(float(yaw))), 0.0)
                v.GetChassis().SetFixed(False)
                parts = veh.ReadEngineJSON(str(root / ENGINE)), veh.ReadTransmissionJSON(str(root / self.json_files[1]))
                v.InitializePowertrain(veh.ChPowertrainAssembly(*parts))
                tires = []
                for wheel in (w for axle in v.GetAxles() for w in axle.GetWheels()):
                    tires.append(t := veh.ReadTireJSON(str(root / TIRE)))
                    t.SetStepsize(float(tire_step))
                    v.InitializeTire(t, wheel, chrono.VisualizationType_NONE)
            finally:
                veh.SetVehicleDataPath(build)
        return JsonVehicle(v, tires)

    def soil_wheels(self, chrono):
        """One ChBodyGeometry per axle (front first), or None for the HMMWV's tyre mesh."""
        out = []
        for r, w in self.wheels:
            out.append(g := chrono.ChBodyGeometry())
            g.coll_cylinders.append(chrono.CylinderShape(chrono.VNULL, chrono.ChVector3d(0, 1, 0), float(r), float(w)))
        return out or None

    def info(self, model) -> dict:
        """Provenance: spec, file shas and the build as measured (ov_vehicle.build_info)."""
        import pychrono.vehicle as veh
        v, sp = model.GetVehicle(), {}
        for n, a, s in hmmwv_data().WHEEL_SPECS:
            p = v.GetChassisBody().GetFrameRefToAbs().TransformPointParentToLocal(v.GetSpindlePos(a, s))
            sp[n.removeprefix('tire_')] = [round(float(p.x), 5), round(float(p.y), 5), round(float(p.z), 5)]
        return dict(name=self.name, spawn_dz_m=self.dz, soil_wheels=self.wheels, json=self.json_files,
                    **{k: x for k, x in self.files.items() if k.endswith('sha256')},
                    build=dict(mass_kg=round(float(v.GetMass()), 3), spindles_rel_reference_m=sp,
                               tire_radius_m=round(float(v.GetTire(0, veh.LEFT).GetRadius()), 6)))


class JsonVehicle:
    """A JSON WheeledVehicle behind the HMMWV wrapper's interface (ov_vehicle.PolarisModel); tyres kept referenced."""

    def __init__(self, vehicle, tires):
        self.vehicle, self.tires = vehicle, tires

    def GetVehicle(self):  # noqa: N802 (Chrono name)
        return self.vehicle

    def __getattr__(self, name):
        return getattr(self.vehicle, name)


class Belly:
    """Lowest belly point minus the undisturbed surface under it, every recorded frame (ag_vehicle.BellyClearance):
    the BMP read at xy * (N-1)/N, Chrono's node-on-edge grid, not the pixel-centre TerrainMap convention."""

    def __init__(self, points, tmap):
        self.points, self.tmap, self.scale, self.rows = points, tmap, (tmap.n - 1) / tmap.n, []

    def clearance(self, xyz, quat):
        """Every point's height above the surface, chassis reference at xyz with rotation quat (e0..e3)."""
        w, x, y, z = map(float, quat)
        rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        world = self.points @ rot.T + np.array(xyz, np.float64)
        return world[:, 2] - self.tmap.height(world[:, 0] * self.scale, world[:, 1] * self.scale)

    def on_frame(self, body):
        ref = body.GetFrameRefToAbs()
        p, q = ref.GetPos(), ref.GetRot()
        c = self.clearance((p.x, p.y, p.z), (q.e0, q.e1, q.e2, q.e3))
        i = int(np.argmin(c))
        self.rows.append((float(c[i]), i, int((c < 0.).sum()), float(ref.GetPosDt().z)))

    def arrays(self) -> dict:
        """vehicle_extra.npz (ag_vehicle.py:291-297)."""
        c, i, below, vz = np.array(self.rows, np.float64).reshape(-1, 4).T      # exact for these ints
        return dict(frame=np.arange(len(c), dtype=np.int64), belly_clearance_min_m=c.astype(np.float32),
                    belly_argmin_point=i.astype(np.int32), belly_points_below_surface=below.astype(np.int32),
                    chassis_vz_mps=vz.astype(np.float32), belly_points_chassis_m=self.points.astype(np.float32))


def show(chrono, model):
    """Rigid scenes (scene.py:357-372, visual only): chassis, wheel and tyre visuals back to MESH, the roof marker."""
    for part in ('Chassis', 'Wheel', 'Tire'):
        getattr(model, f'Set{part}VisualizationType')(chrono.VisualizationType_MESH)
    marker, mat = chrono.ChVisualShapeBox(1.6, 1.0, 0.12), chrono.ChVisualMaterial()
    mat.SetDiffuseColor(chrono.ChColor(*MARKER_RGB))
    mat.SetEmissiveColor(chrono.ChColor(*[0.1 * c for c in MARKER_RGB]))
    marker.SetMaterial(0, mat)
    model.GetChassisBody().AddVisualShape(marker, chrono.ChFramed(chrono.ChVector3d(0.1, 0.0, 0.95), chrono.QUNIT))


def load(name, env) -> Vehicle:
    """VEHICLES[name] with its files checked and hashed; the Polaris JSONs are walked from the vehicle file through
    their 'Input File' keys, then engine, gearbox and tyre."""
    v, files, vdata = VEHICLES[name], {}, Path(env.chrono_data) / 'vehicle'
    if v.belly:
        p = env.file(v.belly[0])
        spec = json.loads(p.read_text())
        obj = vdata / spec['source_obj']
        if sha256_file(p) != v.belly[1] or sha256_file(obj) != spec['source_obj_sha256']:
            raise ValueError(f'belly points {p} (pinned {v.belly[1][:12]}...) or their source mesh {obj} differ')
        files.update(belly_points=np.asarray(spec['points'], np.float64), belly_sha256=v.belly[1])
    if v.json_files:
        def path(r):                        # a missing JSON would abort Chrono without a Python error
            if not (q := env.file(OV + r) if r.startswith('Polaris_ov/') else vdata / r).is_file():
                raise FileNotFoundError(f'Polaris JSON {r}: {q} is missing')
            return q
        todo, seen = [v.json_files[0]], {}
        while todo:
            if (r := todo.pop()) not in seen:
                seen[r] = path(r)
                todo += re.findall(r'"[^"]*Input File"\s*:\s*"([^"]+)"', seen[r].read_text())
        seen.update((r, path(r)) for r in (ENGINE, v.json_files[1], TIRE))
        files.update(assets=env.path(OV), json_sha256={r: sha256_file(q) for r, q in seen.items()})
    return replace(v, files=files)


POLARIS = dict(dz=0.40, wheels=((0.25, 0.2121),) * 2,
               belly=(OV + 'ov_polaris_belly.json', '254eacd703899eb340f89c80d5d89be0ac54fbf319f0e588590ccb2d2ffbb1f0'))
VEHICLES = {v.name: v for v in (
    Vehicle('hmmwv'),
    Vehicle('gator', 0.35, ((0.19575, 0.254), (0.2275, 0.3048)), GATOR_BELLY),
    Vehicle('polaris', json_files=('Polaris_ov/Polaris_ovc_stock.json', STOCK_TX), **POLARIS),
    Vehicle('polaris_pc', json_files=('Polaris_ov/Polaris_ovc_pc.json',
                                      'Polaris_ov/Polaris_ovc_AutomaticTransmissionSimpleMap_pc.json'), **POLARIS),
    Vehicle('polaris_4wd', json_files=('Polaris_ov/Polaris_ovc_4wd.json', STOCK_TX), **POLARIS),
    Vehicle('polaris_w08', json_files=('Polaris_ov/Polaris_ovc_stock.json', STOCK_TX),
            **{**POLARIS, 'wheels': ((0.33, 0.2121),) * 2}))}
