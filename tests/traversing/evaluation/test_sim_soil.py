"""Soil drives without Chrono (FINAL_DESIGN 2.2, 4.3-4.4, 5 #2/#3/#10/#13): the soil config, the Python clock and the
terrain-only advance, fell-through and the stock-radius sinkage, breakthrough in the stop order, the soil branch goal,
the crm_extra layout, the process environment and the runner's retries. With the release data, released soil drives
are replayed through Stops (the soil A3): every non-breakthrough drive gets its recorded status and frame count, and a
breakthrough drive has no earlier stop and a recorded maximum sinkage above the threshold (its per-frame sinkage is not
stored). Over all 28,042 released soil drives (shared model 18,027, tracker 1,269, unseen arenas 8,746) this replay
agreed on every one: goal 24,164, blockage 683 (near-stop included), breakthrough 3,158, timeout 36, rollover 1.

    NEDM_DATA=... NEDM_RELEASE_CACHE=... PYTHONPATH=src python -m unittest discover -s tests/traversing/evaluation \\
        -p test_sim_soil.py -v
"""
import csv
import gzip
import hashlib
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from nedm.traversing.evaluation import episode as E
from nedm.traversing.evaluation import runner as R
from nedm.traversing.evaluation.config import Env, EvalConfig, soil_config
from nedm.traversing.evaluation.sim import After, FellThrough, Sim

from .common import CACHE, CACHE_DIR, DATA
from .test_episode import FakeSim, route

CRM_MAIN = dict(schema='crm_f104_config_v1', spacing_m=0.08, depth_m=0.24, step_s=0.001, active_domain_m=[2.0, 2.0, 1.0],
                active_domain_delay_s=0.0, side_walls=True, mbs_threads=4, tire_mesh='hmmwv/hmmwv_tire_coarse_closed.obj',
                soil=dict(density=1700.0, young_modulus_pa=1e6, poisson_ratio=0.3, mu_I0=0.04, friction=0.8,
                          average_diam_m=0.005, cohesion_pa=5e3),
                sph=dict(d0_multiplier=1.2, free_surface_threshold=0.8, artificial_viscosity=0.5, shifting_method='PPST',
                         shifting_ppst_push=3.0, shifting_ppst_pull=1.0, num_proximity_search_steps=4))
SOIL_ARM = EvalConfig(name='pid_native', ground='soil', soil_config='crm_main', planner='given', label='tracker')


class V:                                               # a pychrono ChVector3d stand-in
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def bare_sim(soil, **kw):
    """A Sim without Chrono: the attributes __post_init__ would set from the vehicle, plus mocks."""
    s = Sim.__new__(Sim)
    s.soil, s.dt, s.t, s.terrain, s.model = soil, 0.001 if soil else 0.002, 0., mock.Mock(), mock.Mock()
    s.wheels = tuple((n, i // 2, i % 2) for i, n in enumerate(('tire_fl', 'tire_fr', 'tire_rl', 'tire_rr')))
    s.tire_radii = dict(tire_fl=.47, tire_fr=.47, tire_rl=.47, tire_rr=.47)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


class TestSoilConfig(unittest.TestCase):
    def load(self, cfg):
        with tempfile.TemporaryDirectory() as d:
            (p := Path(d) / 'c.json').write_text(json.dumps(cfg))
            return soil_config(str(p), Env(Path(d)))[1]

    def test_crm_main(self):
        """crm_main: 1 ms, 50 substeps; the released file is this config (plus its name)."""
        self.assertEqual(round(0.05 / self.load(CRM_MAIN)['step_s']), 50)
        if DATA:
            got = soil_config('crm_main', Env.from_environ())[1]
            self.assertEqual(got, {**CRM_MAIN, 'name': 'crm_main_step1ms_spacing008'})

    def test_refusals(self):
        """No silent default: a missing step or key, or a step that does not divide 50 ms, is an error naming it."""
        drop = lambda d, k: {x: v for x, v in d.items() if x != k}       # noqa: E731
        for bad, word in ((dict(CRM_MAIN, step_s=0.003), 'step_s'), (drop(CRM_MAIN, 'step_s'), 'step_s'),
                          (dict(CRM_MAIN, step_s=1), 'step_s'), (drop(CRM_MAIN, 'depth_m'), 'depth_m'),
                          (dict(CRM_MAIN, sph=drop(CRM_MAIN['sph'], 'shifting_method')), 'sph.shifting_method'),
                          (dict(CRM_MAIN, soil=drop(CRM_MAIN['soil'], 'cohesion_pa')), 'soil.cohesion_pa')):
            with self.subTest(word=word), self.assertRaisesRegex(ValueError, word.replace('.', r'\.')):
                self.load(bad)
        self.assertEqual(self.load(dict(CRM_MAIN, step_s=0.0005))['step_s'], 0.0005)     # the M113 step: 100 substeps


class TestSoilSim(unittest.TestCase):
    def test_python_clock_and_terrain_advance(self):
        """Soil: t is the running Python sum of the step (not k * dt) and only the terrain advances; rigid: both."""
        s, want = bare_sim(CRM_MAIN), 0.
        for _ in range(150):                           # three frames
            self.assertEqual(s.now(), want)
            s.advance()
            want += 0.001
        self.assertNotEqual(s.now(), 150 * 0.001)      # the accumulated clock differs from the product in the last bit
        self.assertEqual(s.terrain.Advance.call_count, 150)
        s.model.Advance.assert_not_called()
        r = bare_sim(None, system=mock.Mock(**{'GetChTime.return_value': 1.5}))
        r.advance()
        r.model.Advance.assert_called_once_with(0.002)
        self.assertEqual(r.now(), 1.5)

    def after(self, z_chassis, spindle_z, h=1.0):
        ref = mock.Mock()
        ref.GetPos.return_value = V(3., 4., z_chassis)
        ref.GetRot.return_value.GetCardanAnglesZYX.return_value = V(0., 0., .2)
        veh = mock.Mock(**{'GetRoll.return_value': .1, 'GetPitch.return_value': -.2})
        veh.GetSpindlePos.side_effect = lambda a, s: V(1., 2., spindle_z[2 * a + s])
        s = bare_sim(CRM_MAIN, chassis=mock.Mock(**{'GetFrameRefToAbs.return_value': ref}), vehicle=veh,
                     tmap=SimpleNamespace(height=lambda x, y: np.float64(h)))
        return s.after_frame()

    def test_after_frame_sinkage_and_fell_through(self):
        """sinkage = max over wheels of stock radius - (spindle z - TerrainMap z); 1 m below the BMP raises."""
        a = self.after(1.6, (1.2, 1.3, 1.0, 1.4))
        self.assertEqual((a.pose.tolist(), a.roll, a.pitch), ([3., 4., .2], .1, -.2))
        self.assertEqual(a.sinkage, .47 - (1.0 - 1.0))
        with self.assertRaises(FellThrough):
            self.after(-0.0001, (1.2, 1.3, 1.0, 1.4))
        self.assertIsNotNone(self.after(0.0, (1.2, 1.3, 1.0, 1.4)))    # exactly 1 m below: not fallen through

    def test_crm_extra_layout(self):
        """crm_extra.npz columns as crm_collect.py:339-342: z, ground, quaternion, then per wheel spindle z, slip, fx."""
        row = [10., 9.5, 1., 0., 0., 0., *range(4), *range(10, 14), *range(20, 24)]
        x = Sim.crm_extra(bare_sim(CRM_MAIN), [row, row])
        self.assertEqual((x['pos_z_m'].tolist(), x['bmp_ground_z_m'].dtype), ([10., 10.], np.float32))
        self.assertEqual((x['quat'].shape, x['spindle_z_m'][0].tolist(), x['slip_ratio'][0].tolist(),
                          x['fsi_force_wheel_fx_n'][0].tolist()), ((2, 4), [0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]))
        self.assertEqual(x['wheel_order'].tolist(), ['tire_fl', 'tire_fr', 'tire_rl', 'tire_rr'])


class TestBreakthrough(unittest.TestCase):
    def run_stops(self, sinks, roll=0., x=0., limit=.3):
        """(status, frame) of a standing vehicle with the given post-frame sinkages."""
        st = E.Stops((30., 0.), 2.5, breakthrough_m=limit)
        ep = SimpleNamespace(state=[], action=[], pose=[], parked=[])
        for k, s in enumerate(sinks):
            ep.k, ep.after = k, After(np.array([x, 0., 0.]), roll, 0., s)
            for rows, v in ((ep.state, np.zeros(17, np.float32)), (ep.action, np.zeros(3, np.float32)),
                            (ep.pose, np.array([x, 0., 0.])), (ep.parked, False)):
                rows.append(v)
            if status := st.check(ep):
                return status, k + 1, st.max_sinkage
        return None, len(sinks), st.max_sinkage

    def test_five_frames_in_a_row(self):
        self.assertEqual(self.run_stops([.31] * 5), ('soil_breakthrough_terminated', 5, .31))
        self.assertEqual(self.run_stops([.31] * 4 + [.2] + [.31] * 4)[:2], (None, 9))      # a shallower frame resets
        self.assertEqual(self.run_stops([.3] * 9)[:2], (None, 9))                          # == threshold: not deeper
        self.assertEqual(self.run_stops([.1, .5, .2])[2], .5)                              # max sinkage recorded

    def test_order(self):
        """Breakthrough first: a 5th deep frame that also rolls over or reaches the goal is a breakthrough; earlier
        frames stop on their own rules. Rigid (no threshold) never reads the sinkage."""
        self.assertEqual(self.run_stops([.31] * 5, roll=1.1)[:2], ('rollover', 1))
        self.assertEqual(self.run_stops([.31] * 5, x=30.)[:2], ('goal_reached', 1))
        st = E.Stops((30., 0.), 2.5, breakthrough_m=.3)
        st.deep = 4                                    # four deep frames behind: the 5th wins over rollover and goal
        ep = SimpleNamespace(k=4, after=After(np.array([30., 0., 0.]), 1.1, 0., .31))
        self.assertEqual(st.check(ep), 'soil_breakthrough_terminated')
        ep = SimpleNamespace(k=0, after=After(np.zeros(3), 0., 0., 9.), state=[np.zeros(17)], action=[np.zeros(3)],
                             pose=[np.zeros(3)], parked=[False])
        self.assertIsNone(E.Stops((30., 0.), 2.5).check(ep))


class TestSoilBranch(unittest.TestCase):
    def setUp(self):
        self.settle, E.SETTLE_FRAMES = E.SETTLE_FRAMES, 2

    def tearDown(self):
        E.SETTLE_FRAMES = self.settle

    def test_soil_branch_retargets(self):
        """Soil: a route ending 0.4 m from the case goal is accepted and becomes the goal after the switch (the rigid
        tolerance is 0.25 m and the case goal stays); the crm_extra rows of a 5-tuple measure are kept."""
        b = route(12, 10., y=.5)
        b['waypoints'][-1] = [20., .4]
        with self.assertRaises(ValueError):
            E.Branch(2, b, np.array([20., 0.]))
        sim = FakeSim([.3, -.4])
        sim.measure = lambda k, t, u: (np.zeros(17, np.float32), np.zeros(3, np.float32), np.array([sim.x, 0., 0.]), 1.,
                                       [float(k)] * 18)
        stops = E.Stops((20., 0.), 2.5, breakthrough_m=.3)
        sim.after_frame = lambda: (setattr(sim, 'x', sim.x + 6.), After(np.array([sim.x, 0., 0.]), 0., 0., 0.))[1]
        ep = E.drive_episode(sim, route(0, 1.), R.make_controller(SOIL_ARM), stops, E.Branch(2, b, np.array([20., 0.]),
                                                                                             soil=True))
        self.assertEqual(stops.goal.tolist(), [20., .4])
        self.assertEqual((ep.status, len(ep.state), [r[0] for r in ep.extra]), ('goal_reached', 3, [0., 1., 2.]))


class TestProcess(unittest.TestCase):
    def test_env(self):
        """Soil: OMP 4 (CRM_OMP), BLAS 1 and one GPU per process; rigid single-threaded (FINAL_DESIGN 4.3)."""
        e = R.process_env('soil', 3)
        self.assertEqual(e, dict(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                 ROCR_VISIBLE_DEVICES='3', HIP_VISIBLE_DEVICES='0'))
        self.assertEqual((R.env_problems('soil', e), R.env_problems('rigid', R.process_env('rigid'))), ([], []))
        self.assertEqual(len(R.env_problems('soil', {**e, 'OMP_NUM_THREADS': '1'})), 1)
        self.assertEqual(len(R.env_problems('soil', {k: v for k, v in e.items() if 'VISIBLE' not in k})), 1)
        self.assertEqual(len(R.env_problems('soil', {**e, 'ROCR_VISIBLE_DEVICES': '0,1'})), 1)
        self.assertEqual(len(R.env_problems('rigid', e)), 2)                       # OMP 4 and no LP_NUM_THREADS

    def spawn(self, fake, out=None):
        task = SimpleNamespace(id='t1', to_dict=lambda env: dict(id='t1'))
        env = SimpleNamespace(data=Path('/data'), chrono_data=Path('/chrono'), release_cache=None)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(R.subprocess, 'run', side_effect=fake) as run:
            out = out or Path(d) / 'run'
            try:
                rec = R.spawn(SOIL_ARM, task, route(0, 1.), out, env=env, gpu=2)
            finally:
                att = [json.loads(x) for x in (out / 'attempts.jsonl').read_text().splitlines()] if \
                    (out / 'attempts.jsonl').exists() else []
            return rec, att, run.call_args_list

    def test_retry_then_crash(self):
        """A failing drive process (fell-through, abort, wall timeout) is retried once; then a 'crash' record."""
        def fake(cmd, **kw):
            if len(fake.calls) == 0:
                fake.calls.append(1)
                raise R.subprocess.TimeoutExpired(cmd, 1)
            return SimpleNamespace(returncode=1)
        fake.calls = []
        rec, att, calls = self.spawn(fake)
        self.assertEqual((rec.status, rec.frames, [a['rc'] for a in att]), ('crash', 0, ['wall_timeout', 1]))
        env = calls[0].kwargs['env']
        self.assertEqual((env['ROCR_VISIBLE_DEVICES'], env['HIP_VISIBLE_DEVICES'], env['OMP_NUM_THREADS'],
                          env['NEDM_CHRONO_DATA'], calls[0].kwargs['timeout']), ('2', '0', '4', '/chrono', 2400.))

    def test_refusal_and_success(self):
        with self.assertRaises(R.ConfigError):
            self.spawn(lambda cmd, **kw: SimpleNamespace(returncode=R.REFUSED))

        def ok(cmd, **kw):
            out = Path(cmd[cmd.index('--out') + 1])
            R.Record('t1', 'pid_native', 'goal_reached', True, frames=3, arrays=dict(state=np.zeros((3, 17), np.float32)),
                     extras=dict(crm_extra=dict(pos_z_m=np.ones(3, np.float32)))).save(out)
            return SimpleNamespace(returncode=0)
        rec, att, calls = self.spawn(ok)
        self.assertEqual((rec.status, len(att), rec.extras['crm_extra']['pos_z_m'].tolist()), ('goal_reached', 1, [1.] * 3))
        self.assertEqual(calls[0].args[0][-2:], ['--attempt', '1'])

    def test_attempts_survive_job_kills(self):
        """drive.log counts the attempts across job kills (a kill leaves no attempts.jsonl line), refusals not: a
        kill, a refusal, then a failure end in 'crash' with the log's last line; a finished folder is then loaded."""
        def fake(cmd, **kw):
            fake.n += 1
            if fake.n == 1:
                raise KeyboardInterrupt('the job was killed')
            kw['stdout'].write('# refused: x\n' if fake.n == 2 else 'FellThrough: boom\n')
            return SimpleNamespace(returncode=R.REFUSED if fake.n == 2 else 1)
        fake.n = 0
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'run'
            with self.assertRaises(KeyboardInterrupt):
                self.spawn(fake, out)
            with self.assertRaises(R.ConfigError):
                self.spawn(fake, out)
            rec, att, _ = self.spawn(fake, out)
            self.assertEqual((rec.status, rec.provenance['last_log_line'], [(a['attempt'], a['rc']) for a in att]),
                             ('crash', 'FellThrough: boom', [(2, R.REFUSED), (2, 1)]))
            self.assertEqual((self.spawn(fake, out)[0].status, fake.n), ('crash', 3))


ITEMS = ('shared_model_drive_folders_soil', 'tracker_drive_folders', 'unseen_arena_drive_folders_soil')
NEAR_STOP = ('pid_held', 'pid_perturbed', 'policy')


def replay_soil(z, o, goal):
    """(status, frames) of a stored soil drive through Stops without the breakthrough rule (no per-frame sinkage is
    stored); the goal switches to the branch route's end at the branch frame (soil re-targets)."""
    br = o.get('branch') if (o.get('branch') or {}).get('reached') else None
    st, ac, po, pk, n = z['state'], z['action'], z['pose'], z['parked'], len(z['state'])
    stops = E.Stops(goal, o['goal_radius_m'], near_stop=o.get('mode') in NEAR_STOP)
    ep = SimpleNamespace(state=[], action=[], pose=[], parked=[])
    for k in range(n):
        ep.k = k
        for rows, v in ((ep.state, st[k]), (ep.action, ac[k]), (ep.pose, po[k]), (ep.parked, bool(pk[k]))):
            rows.append(v)
        p, s = (po[k + 1], st[k + 1]) if k + 1 < n else (z['terminal_pose'], z['terminal_state'])
        ep.after = After(np.asarray(p, float), float(s[2]), float(s[3]))
        if status := stops.check(ep):
            return status, k + 1
        if br and k + 1 == br['branch_frame']:
            stops.goal = np.asarray(br['branch_goal_xy'], float)
    return None, n


@unittest.skipUnless(DATA and CACHE, 'the soil A3 needs NEDM_DATA (the release restore base) and the release cache')
class TestReleasedSoilDrives(unittest.TestCase):
    def test_soil_a3_replay(self):
        """The 10 lowest md5(run folder) drives of every (run root, status, mode) class of the released HMMWV soil
        drives."""
        from nedm.traversing.evaluation.suites import load_suite
        cases = {t.id: t for s in ('f104_800', 'tracker423') for t in load_suite(s)}
        classes = defaultdict(list)
        for item in ITEMS:
            with gzip.open(CACHE_DIR / 'traversing/evaluation' / item / 'index.csv.gz', 'rt') as f:
                runs = [r['path'][:-13] for r in csv.DictReader(f) if r['path'].endswith('/outcome.json')]
            for run in runs:
                o = json.loads((DATA / run / 'outcome.json').read_text())
                if o.get('terrain') == 'crm' and '__' in run.rsplit('/', 1)[1]:
                    classes[(run.rsplit('/', 1)[0], o['status'], o.get('mode'))].append((run, o))
        n = bt = 0
        for key, rows in sorted(classes.items(), key=str):
            rows = sorted(rows, key=lambda r: hashlib.md5(r[0].encode()).hexdigest())
            for run, o in rows[:10]:
                c = DATA / run / 'case.json'
                goal = o['branch']['case_goal_xy'] if o.get('branch') else json.loads(c.read_text())['goal_xy'] if \
                    c.exists() else cases[run.rsplit('/', 1)[1].split('__')[0]].read_case()['goal_xy']
                with np.load(DATA / run / 'trajectory.npz') as z, self.subTest(run=run):
                    got = replay_soil(z, o, np.asarray(goal, float))
                    if o['status'] == 'soil_breakthrough_terminated':
                        self.assertTrue(got[0] is None or got[1] == o['frames'], got)
                        self.assertGreater(o['crm']['max_wheel_sinkage_below_bmp_m'], .24 + .06)
                        bt += 1
                    else:
                        self.assertEqual(got, (o['status'], o['frames']))
                n += 1
        self.assertGreaterEqual(n, 100)
        self.assertGreater(bt, 10)
        self.assertTrue({'goal_reached', 'prolonged_blockage_terminated', 'timeout', 'soil_breakthrough_terminated'}
                        <= {k[1] for k in classes})


if __name__ == '__main__':
    unittest.main()
