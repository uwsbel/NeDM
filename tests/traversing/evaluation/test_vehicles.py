"""Vehicles without Chrono (FINAL_DESIGN 4.2, 5 #4/#14/#15, 8.2 A4). With a mock pychrono: the construction calls of
the HMMWV, the Gator and the JSON Polaris against hand-written sequences (hmmwv_data.create_hmmwv,
ag_vehicle.create_gator, ov_vehicle.PolarisModel at 901d6c9), the Polaris data root restored; the spawn arithmetic; one
soil geometry per axle in build_crm with the tyre mesh still built; the pre-flight refusals and JSON walk; the belly
geometry, the vehicle_extra schema and the belly recorded at recorded frames only.

With the release data (A4): the belly flag recomputed from the stored pose + crm_extra z/quat + the belly points equals
the recorded and the indexed flag, on every flagged drive and the 10 lowest-md5 drives per arm. One-off over all 3,400
drives of the four arms: Polaris 0/800 (f104) and 0/1,000 (unseen), polaris_pc 1/800, Gator 10/800, all equal to the
index; |recomputed - recorded clearance| <= 2.7e-7 m. With NEDM_CHRONO_DATA too: the Polaris pre-flight hashes the
JSON files the recorded drives hashed.

    NEDM_DATA=... NEDM_RELEASE_CACHE=... [NEDM_CHRONO_DATA=...] PYTHONPATH=src python -m unittest discover \\
        -s tests/traversing/evaluation -p test_vehicles.py -v
"""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock, call

import numpy as np
from PIL import Image

from nedm.traversing.evaluation import config as C
from nedm.traversing.evaluation import sim as S
from nedm.traversing.evaluation import vehicles as V
from nedm.traversing.evaluation.labels import belly_flag
from nedm.traversing.training.state import STATE_FIELDS

from .common import CACHE, DATA
from .test_sim_soil import CRM_MAIN

NONE5 = [getattr(call, f'Set{p}VisualizationType') for p in V.VISUALS]


def pychrono():
    """A mock pychrono + pychrono.vehicle installed in sys.modules; every call lands in mgr.mock_calls, in order."""
    mgr = MagicMock()
    mgr.chrono.vehicle = mgr.veh
    return mgr, mock.patch.dict(sys.modules, {'pychrono': mgr.chrono, 'pychrono.vehicle': mgr.veh})


def arena(d, n=8):
    """A n x n arena of 8 m whose BMP rises along x (a TerrainMap; the belly lookup differs from the pixel centres)."""
    Image.fromarray(np.tile(np.arange(n, dtype=np.uint8) * 30, (n, 1))).save(Path(d) / 'a.bmp')
    (Path(d) / 'arena_meta.json').write_text(json.dumps(dict(bmp='a.bmp', size_m=8.0, height_min_m=0.0,
                                                            height_max_m=2.55)))
    return S.TerrainMap(d)


class TestCreate(unittest.TestCase):
    def test_hmmwv_and_gator_call_order(self):
        """create_hmmwv's setters in its order (rigid: TMEASY + HULLS; soil: RIGID_MESH + no chassis collision); the
        Gator swaps engine/gearbox/drive/steering for SIMPLE driveline and brakes, no brake locking."""
        for name in ('hmmwv', 'gator'):
            for soil in (False, True):
                mgr, mods = pychrono()
                ch, veh = mgr.chrono, mgr.veh
                with mods:
                    v = V.VEHICLES[name].create((1., 2., 3.), 0.5, tire_step=0.001, soil=soil)
                mid = [call.SetEngineType(veh.EngineModelType_SHAFTS),
                       call.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SHAFTS),
                       call.SetDriveType(veh.DrivelineTypeWV_AWD), call.SetSteeringType(veh.SteeringTypeWV_PITMAN_ARM)] \
                    if name == 'hmmwv' else [call.SetDrivelineType(veh.DrivelineTypeWV_SIMPLE),
                                             call.SetBrakeType(veh.BrakeType_SIMPLE), call.EnableBrakeLocking(False)]
                want = [call.SetContactMethod(ch.ChContactMethod_SMC), call.SetChassisFixed(False),
                        call.SetInitPosition(ch.ChCoordsysd.return_value), *mid,
                        call.SetTireType(veh.TireModelType_RIGID_MESH if soil else veh.TireModelType_TMEASY),
                        call.SetTireStepSize(0.001),
                        call.SetChassisCollisionType(veh.CollisionType_NONE if soil else veh.CollisionType_HULLS),
                        call.Initialize(), *[f(ch.VisualizationType_NONE) for f in NONE5], call.GetSystem(),
                        call.GetSystem().SetCollisionSystemType(ch.ChCollisionSystem.Type_BULLET)]
                with self.subTest(name=name, soil=soil):
                    self.assertIs(v, (veh.HMMWV_Full if name == 'hmmwv' else veh.Gator).return_value)
                    self.assertEqual(v.mock_calls, want)
                    ch.ChVector3d.assert_called_once_with(1., 2., 3.)
                    ch.QuatFromAngleZ.assert_called_once_with(0.5)

    def test_polaris_call_order_and_data_root(self):
        """ov_vehicle.PolarisModel's order inside a temporary data root (Polaris_ov -> released assets, Polaris -> the
        build's folder), the build's data path restored after; the visuals and Bullet follow the restore (no file is
        read there). A failing build restores the path too."""
        mgr, mods = pychrono()
        ch, veh, v = mgr.chrono, mgr.veh, mgr.v
        veh.GetVehicleDataFile.return_value = '/build/vehicle/'
        axles, seen = [MagicMock(), MagicMock()], {}
        for i, a in enumerate(axles):
            a.GetWheels.return_value = [f'w{i}L', f'w{i}R']
        v.GetAxles.return_value = axles
        tires = [getattr(mgr, f't{i}') for i in range(4)]

        def wheeled(path, method):
            root = Path(path).parents[1]
            seen.update(root=root, ov=os.readlink(root / 'Polaris_ov'), stock=os.readlink(root / 'Polaris'))
            return v
        veh.WheeledVehicle.side_effect, veh.ReadTireJSON.side_effect = wheeled, tires
        with tempfile.TemporaryDirectory() as d, mods:
            m = replace(V.VEHICLES['polaris_pc'], files=dict(assets=Path(d))).create((1., 2., 3.), 0.5, tire_step=0.001,
                                                                                       soil=True)
            root = seen['root']
            self.assertEqual((seen['ov'], seen['stock']), (f'{d}/Polaris_ov', '/build/vehicle/Polaris'))
        self.assertFalse(root.exists())
        self.assertEqual((m.GetVehicle(), m.tires), (v, tires))
        r = str(root)
        tire = [c for i, t in enumerate(tires) for c in (
            call.veh.ReadTireJSON(f'{r}/{V.TIRE}'), getattr(call, f't{i}').SetStepsize(0.001),
            call.v.InitializeTire(t, f'w{i // 2}{"LR"[i % 2]}', ch.VisualizationType_NONE))]
        want = [call.veh.GetVehicleDataFile(''), call.veh.SetVehicleDataPath(f'{r}/'),
                call.veh.WheeledVehicle(f'{r}/Polaris_ov/Polaris_ovc_pc.json', ch.ChContactMethod_SMC),
                call.v.Initialize(ch.ChCoordsysd.return_value, 0.0), call.v.GetChassis(),
                call.v.GetChassis().SetFixed(False), call.veh.ReadEngineJSON(f'{r}/{V.ENGINE}'),
                call.veh.ReadTransmissionJSON(f'{r}/Polaris_ov/Polaris_ovc_AutomaticTransmissionSimpleMap_pc.json'),
                call.veh.ChPowertrainAssembly(veh.ReadEngineJSON.return_value, veh.ReadTransmissionJSON.return_value),
                call.v.InitializePowertrain(veh.ChPowertrainAssembly.return_value), call.v.GetAxles(), *tire,
                call.veh.SetVehicleDataPath('/build/vehicle/'),
                *[getattr(call.v, f'Set{p}VisualizationType')(ch.VisualizationType_NONE) for p in V.VISUALS],
                call.v.GetSystem(), call.v.GetSystem().SetCollisionSystemType(ch.ChCollisionSystem.Type_BULLET)]
        self.assertEqual([c for c in mgr.mock_calls if c[0].split('.')[0] in ('veh', 'v', 't0', 't1', 't2', 't3')], want)
        mgr, mods = pychrono()
        mgr.veh.GetVehicleDataFile.return_value = '/build/vehicle/'
        mgr.veh.WheeledVehicle.side_effect = RuntimeError('JSON')
        with tempfile.TemporaryDirectory() as d, mods, self.assertRaisesRegex(RuntimeError, 'JSON'):
            replace(V.VEHICLES['polaris'], files=dict(assets=Path(d))).create((0, 0, 0), 0, tire_step=0.001, soil=True)
        self.assertEqual(mgr.veh.SetVehicleDataPath.call_args, call('/build/vehicle/'))


class TestGeometry(unittest.TestCase):
    def test_table(self):
        """The vehicles config validates, with belly points exactly where the labels need them; Polaris variants
        differ only in their JSON files or soil wheels."""
        self.assertEqual(tuple(n for n, v in V.VEHICLES.items() if v.belly), ('gator', 'polaris', 'polaris_pc',
                                                                             'polaris_4wd', 'polaris_w08'))
        self.assertIn("vehicle 'tank' is not one of ('hmmwv', 'gator', 'polaris'",          # config reads this table
                      ' '.join(C.EvalConfig(name='a', vehicle='tank').validate()))
        p = V.VEHICLES['polaris']
        for n in ('polaris_pc', 'polaris_4wd', 'polaris_w08'):
            q = V.VEHICLES[n]
            self.assertEqual((q.dz, q.belly, q.json_files[1] == V.STOCK_TX), (p.dz, p.belly, n != 'polaris_pc'))
            self.assertEqual(q.wheels, ((0.33, 0.2121),) * 2 if n == 'polaris_w08' else p.wheels)
        self.assertEqual((V.VEHICLES['hmmwv'].wheels, V.VEHICLES['hmmwv'].belly, V.VEHICLES['hmmwv'].json_files), ((), None, ()))

    def test_spawn_arithmetic(self):
        """HMMWV h + 0.75; the others ((h + 0.75) - 0.75) + dz (ag_vehicle.py:165), which is not h + dz bitwise."""
        hs = np.random.default_rng(0).random(2000) * 2.9 - 0.4         # full-precision heights, as interpolated
        for h in hs:
            self.assertEqual(V.VEHICLES['hmmwv'].spawn(h), h + 0.75)
            for n, dz in (('gator', 0.35), ('polaris', 0.40), ('polaris_w08', 0.40)):
                self.assertEqual(V.VEHICLES[n].spawn(h), ((h + 0.75) - 0.75) + dz)
        self.assertTrue(any(V.VEHICLES['gator'].spawn(h) != h + 0.35 for h in hs))

    def test_soil_wheels_in_build_crm(self):
        """One cylinder geometry per axle (axis = spindle y), front first; build_crm couples each spindle with its
        axle's geometry and still builds the tyre-mesh geometry; the HMMWV keeps the mesh on every wheel."""
        for name in ('gator', 'polaris_w08', 'hmmwv'):
            mgr, _ = pychrono()
            ch, veh, fsi = mgr.chrono, mgr.veh, MagicMock()
            ch.ChBodyGeometry.side_effect = geoms = [MagicMock(name=f'g{i}') for i in range(3)]
            wheels = V.VEHICLES[name].soil_wheels(ch)
            vehicle = MagicMock()
            vehicle.GetAxles.return_value = axles = [MagicMock(), MagicMock()]
            for a in axles:
                a.GetWheels.return_value = [MagicMock(), MagicMock()]
            S.build_crm(ch, veh, fsi, MagicMock(), vehicle, Path('/a.bmp'),
                        dict(size_m=80., height_min_m=0., height_max_m=1.), CRM_MAIN, wheels)
            spindles = [w.GetSpindle.return_value for a in axles for w in a.GetWheels.return_value]
            n = len(V.VEHICLES[name].wheels)
            per_axle = [geoms[0], geoms[0], geoms[1], geoms[1]] if n else [geoms[0]] * 4
            with self.subTest(name):
                self.assertEqual(veh.CRMTerrain.return_value.AddRigidBody.call_args_list,
                                 [call(s, g, False) for s, g in zip(spindles, per_axle)])
                ch.TrimeshShape.assert_called_once()
                self.assertEqual(ch.CylinderShape.call_args_list, [call(ch.VNULL, ch.ChVector3d.return_value, r, w)
                                                                   for r, w in V.VEHICLES[name].wheels])
                for g, cyl in zip(geoms, ch.CylinderShape.call_args_list):
                    g.coll_cylinders.append.assert_called_once_with(ch.CylinderShape.return_value)

    def test_belly(self):
        """clearance = point z - BMP at xy * 7/8 (an 8-pixel arena; not the pixel-centre lookup), rotated by the chassis
        quaternion; on_frame rows -> vehicle_extra.npz in the recorded schema."""
        pts = np.array([[1., 0., -0.1], [-1., 0.5, -0.2], [0., 0., 0.3], [2., -1., -0.6]])
        with tempfile.TemporaryDirectory() as d:
            tm = arena(d)
        b, xyz = V.Belly(pts, tm), (0.3, -0.2, 1.0)
        c = b.clearance(xyz, (1., 0., 0., 0.))
        np.testing.assert_array_equal(c, pts[:, 2] + 1.0 - tm.height((pts[:, 0] + 0.3) * 7 / 8, (pts[:, 1] - 0.2) * 7 / 8))
        self.assertFalse(np.allclose(c, pts[:, 2] + 1.0 - tm.height(pts[:, 0] + 0.3, pts[:, 1] - 0.2)))
        q = (np.cos(np.pi / 4), 0., 0., np.sin(np.pi / 4))         # 90 deg yaw: (x, y) -> (-y, x)
        np.testing.assert_allclose(b.clearance(xyz, q), pts[:, 2] + 1.0 - tm.height((0.3 - pts[:, 1]) * 7 / 8,
                                                                                    (pts[:, 0] - 0.2) * 7 / 8), atol=1e-12)
        body = MagicMock()
        ref = body.GetFrameRefToAbs.return_value
        ref.GetPos.return_value, ref.GetRot.return_value = SimpleNamespace(x=.3, y=-.2, z=1.), SimpleNamespace(
            e0=1., e1=0., e2=0., e3=0.)
        ref.GetPosDt.return_value = SimpleNamespace(z=-0.01)
        b.on_frame(body)
        b.on_frame(body)
        a = b.arrays()
        self.assertEqual({k: (x.dtype.str, x.shape) for k, x in a.items()}, dict(
            frame=('<i8', (2,)), belly_clearance_min_m=('<f4', (2,)), belly_argmin_point=('<i4', (2,)),
            belly_points_below_surface=('<i4', (2,)), chassis_vz_mps=('<f4', (2,)), belly_points_chassis_m=('<f4', (4, 3))))
        self.assertEqual((a['frame'].tolist(), a['belly_clearance_min_m'][1], a['belly_argmin_point'][0],
                          a['belly_points_below_surface'][0], a['chassis_vz_mps'][0]),
                         ([0, 1], np.float32(c.min()), c.argmin(), (c < 0).sum(), np.float32(-0.01)))

    def test_measure_records_belly_at_recorded_frames(self):
        """The wrappers' frame hook: once per recorded frame (k >= 0), not for the terminal state (k = -1)."""
        s = S.Sim.__new__(S.Sim)
        s.soil, s.case, s.tire_radii, s.terrain, s.model, s.belly, s.chassis = None, {'id': 'x'}, {}, None, None, \
            MagicMock(), object()
        s.engine = MagicMock(**{'GetMotorSpeed.return_value': 1., 'GetOutputMotorshaftTorque.return_value': 2.})
        s.transmission = MagicMock(**{'GetOutputMotorshaftSpeed.return_value': 3.})
        row = {**{f: 0.5 for f in STATE_FIELDS}, 'pos_x_m': 1., 'pos_y_m': 2., 'yaw_rad': 0.3}
        u = SimpleNamespace(m_steering=0., m_throttle=1., m_braking=0.)
        with mock.patch.object(S, 'hmmwv_data', return_value=SimpleNamespace(capture_row=lambda *a, **k: dict(row))):
            for k in (0, 1, -1):
                s.measure(k, 0., u)
        self.assertEqual(s.belly.on_frame.call_args_list, [call(s.chassis)] * 2)


class TestLoad(unittest.TestCase):
    def env(self, d):
        def file(ref):
            p = d / 'assets' / ref[len(V.OV):] if ref.startswith(V.OV) else C.REPO_ROOT / ref
            if not p.is_file():
                raise C.ReleaseError(f'{p} not found')
            return p
        return SimpleNamespace(chrono_data=d / 'chrono', file=file, path=lambda ref: d / 'assets')

    def put(self, p, text='{}'):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def test_refusals_and_json_walk(self):
        """A belly mesh other than the sampled one and a missing Polaris JSON (found through the 'Input File' keys)
        raise naming the file (runner.drive refuses them); a complete set is hashed: walked files, engine, gearbox,
        tyre. A source-pin mismatch of the reused HMMWV module refuses (ConfigError)."""
        with tempfile.TemporaryDirectory() as d:
            d, env = Path(d), self.env(Path(d))
            self.put(d / 'chrono/vehicle/gator/gator_chassis_col.obj', 'not the hull')
            with self.assertRaisesRegex(ValueError, 'source mesh .*gator_chassis_col.obj'):
                V.load('gator', env)
            self.put(d / 'assets/Polaris_ov/top.json', json.dumps({'Chassis': {'Input File': 'Polaris_ov/ch.json'},
                                                                   'Axles': [{'Suspension Input File': 'Polaris/s.json'}]}))
            self.put(d / 'assets/Polaris_ov/ch.json')
            self.put(d / 'chrono/vehicle/Polaris/s.json', '{"Spring Input File" : "Polaris/spring.json"}')
            for f in (V.ENGINE, V.TIRE, 'Polaris/tx.json'):
                self.put(d / 'chrono/vehicle' / f)
            fake = V.Vehicle('fake', 0.4, json_files=('Polaris_ov/top.json', 'Polaris/tx.json'))
            with mock.patch.dict(V.VEHICLES, fake=fake):
                with self.assertRaisesRegex(FileNotFoundError, 'Polaris JSON Polaris/spring.json'):
                    V.load('fake', env)
                self.put(d / 'chrono/vehicle/Polaris/spring.json')
                got = V.load('fake', env)
        self.assertEqual(sorted(got.files['json_sha256']), sorted(['Polaris_ov/top.json', 'Polaris_ov/ch.json',
                         'Polaris/s.json', 'Polaris/spring.json', V.ENGINE, V.TIRE, 'Polaris/tx.json']))
        self.assertEqual(got.files['assets'], d / 'assets')

    @unittest.skipUnless(importlib.util.find_spec('pychrono'), 'the published HMMWV module imports pychrono')
    def test_source_pins(self):
        """The reused names of the published HMMWV module equal their pins; a changed one refuses every drive."""
        V.hmmwv_data.cache_clear()
        try:
            self.assertEqual(V.hmmwv_data().WHEEL_SPECS[0][0], 'tire_fl')
            V.hmmwv_data.cache_clear()
            with mock.patch.dict(V.PINS, capture_row='0' * 64), self.assertRaisesRegex(C.ConfigError, 'capture_row'):
                V.hmmwv_data()
        finally:
            V.hmmwv_data.cache_clear()

    @unittest.skipUnless(DATA and CACHE and os.environ.get('NEDM_CHRONO_DATA'), 'needs the release and a Chrono build')
    def test_polaris_files_as_recorded(self):
        """The pre-flight hashes the JSON files the recorded Polaris drives hashed (their vehicle block; it also listed
        the TMEASY tyre, which a soil drive never opens), and the belly files are the pinned ones."""
        env = C.Env.from_environ()
        runs = 'data:artifacts/traverse/offroad_vehicles_20260927/e6/sync/runs/'
        for name, run in (('polaris', 'polaris__f104_crm_eval_group_0000__polaris_cem'),
                          ('polaris_pc', 'polaris_pc__f104_crm_eval_group_0000__polaris_grad_pc')):
            rec = json.loads(env.file(f'{runs}{run}/outcome.json').read_text())['vehicle']
            got = V.load(name, env).files
            with self.subTest(name):
                self.assertEqual(got['json_sha256'], {k: v for k, v in rec['json_sha256'].items() if 'TMeasy' not in k})
                self.assertEqual(got['belly_sha256'], rec['belly']['belly_points_sha256'])
        self.assertEqual(V.load('gator', env).files['belly_sha256'], V.GATOR_BELLY[1])


@unittest.skipUnless(DATA and CACHE, 'A4 needs NEDM_DATA (the release restore base) and the release cache')
class TestReleasedBelly(unittest.TestCase):
    def test_a4_belly_flags(self):
        """Index counts (Polaris 0/800 and 0/1,000, polaris_pc 1/800, Gator 10/800) and, on every flagged drive plus the
        10 lowest md5(run) per arm, the flag recomputed from pose + crm_extra z/quat (float32) + belly points."""
        from nedm.traversing.evaluation.suites import arena_dir
        env = C.Env.from_environ()
        ix = 'data:artifacts/traverse/offroad_vehicles_20260927/e6/index/'
        arms = dict(polaris_grad=0, polaris_grad_pc=1, Gfull_grad_gator=10, polaris_u_grad=0)
        rows = [r for f in ('soil_eval_ov_final.json', 'unseen_polaris_v1.json')
                for r in json.loads(env.file(ix + f).read_text())['rows'] if r['arm'] in arms]
        self.assertEqual(Counter(r['arm'] for r in rows), dict(polaris_grad=800, polaris_grad_pc=800, Gfull_grad_gator=800,
                                                               polaris_u_grad=1000))
        self.assertEqual({a: sum(r['belly_flag'] for r in rows if r['arm'] == a) for a in arms}, arms)
        pick = sorted(rows, key=lambda r: (r['arm'], hashlib.md5(r['run_dir'].encode()).hexdigest()))
        pick = [r for r in pick if r['belly_flag']] + [r for a in arms for r in [x for x in pick if x['arm'] == a][:10]]
        pts = {n: np.asarray(json.loads(env.file(V.VEHICLES[n].belly[0]).read_text())['points'], float)
               for n in ('gator', 'polaris')}
        tmaps = {}
        for r in pick:
            tm = tmaps.setdefault(r['arena'], S.TerrainMap(arena_dir(r['arena'], env)))
            b, d = V.Belly(pts['gator' if r['vehicle'] == 'gator' else 'polaris'], tm), DATA / r['run_dir']
            with np.load(d / 'trajectory.npz') as t, np.load(d / 'crm_extra.npz') as x, np.load(d / 'vehicle_extra.npz') as v:
                c = np.array([b.clearance((t['pose'][k, 0], t['pose'][k, 1], x['pos_z_m'][k]), x['quat'][k]).min()
                              for k in range(len(t['pose']))])
                rec = v['belly_clearance_min_m']
                self.assertEqual(sorted(v.files), sorted(V.Belly(pts['polaris'], tm).arrays()))
            with self.subTest(run=r['run_dir']):
                self.assertLess(np.abs(c - rec).max(), 1e-6)
                self.assertEqual((belly_flag(c), belly_flag(rec)), (r['belly_flag'], r['belly_flag']))
        self.assertEqual(len(pick), 51)


if __name__ == '__main__':
    unittest.main()
