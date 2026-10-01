"""M1 live navigation (nav.py, NavPID), stdlib unittest only.

CI (goldens/nav, written by the goldens generator (goldens/README.md) with the ORIGINAL 901d6c9 code): SpeedPI == nav_online.SpeedPI in a
closed loop; NavPID's inputs; the rescue pool rules; a latency-replay miss raises.
NEDM_DATA (+ the release cache for the mission suite and the R1L latency tables):
  grids      grid_from_depth on the 20 sensor_v2 vehicle_frames == sensor_map_v2.grid_from_arrays (sha256 per array)
  decisions  on 6 of them: the pool, the footprint mask, corridors12 (X, lengths, invalid fractions), path heights ==
             nav_online; the direct-depth scorer == GridRiskModel (the full logit vector; cuda on the record GPU, cpu)
  replay     rng / candidate replay of every recorded decision of the 30 W runs (194) and of 3 periodic runs with the
             rescue rungs and no-route decisions: nav.ladder with the recorded index == the recorded route (waypoints,
             speeds, stations, meta), candidate count, tries, planning bound, aim, rung
  schedule   the episode loop + NavHook driven through the recorded trajectories of all 120 runs (no physics): every
             decision at the recorded frame with the recorded trigger, waypoint, route and latency; the per-frame
             waypoint index; no stop before the last frame (the runs store no post-frame state after it)
  nav_analyze  labels.mission_labels over the released outcomes: complete 27/25/25/22, waypoints 191/186/179/173,
             slides, unsafe, median time and the paired bootstrap intervals (seed 0, 20,000) of summary.json
All 6,035 recorded decisions of the 120 runs replay the same way (0 mismatches, a one-off with replay_run below).

    NEDM_DATA=<release restore root> NEDM_RELEASE_CACHE=<release download cache> \\
        PYTHONPATH=src python -m unittest discover -s tests/traversing/evaluation -p test_nav.py -v
"""
import dataclasses
import hashlib
import json
import os
import unittest
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from nedm.traversing.evaluation import episode as E
from nedm.traversing.evaluation import nav
from nedm.traversing.evaluation.config import DT, Env, EvalConfig
from nedm.traversing.evaluation.controllers import NavPID, SpeedPI
from nedm.traversing.evaluation.labels import drive_labels
from nedm.traversing.evaluation.planner import RECORD_ENV
from nedm.traversing.evaluation.routes import geom5
from nedm.traversing.evaluation.sim import After

try:
    from .common import CACHE, DATA
    from .tables import released
except ImportError:                     # discover -s tests/traversing/evaluation
    from common import CACHE, DATA
    from tables import released

HERE = Path(__file__).resolve().parent
GOLD = json.loads((HERE / 'goldens/nav/nav.json').read_text())
NAV = 'artifacts/traverse/fdm_f104_50h_20260909/nav_v1'
ARMS = dict(W=('plan_once_per_waypoint', 'waypoint'), R2=('replan_every_2s', 2.0), R1=('replan_every_1s', 1.0),
            R1L=('replan_every_1s_delay_charged', 1.0))
R1L_REPLAY = f'data:{NAV}/local_luffy/runs/{{task}}__R1L/decisions.json'
MODELS = 'data:artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/matched/matched_Dabs_s*.pt'
GPU = torch.cuda.is_available() and (torch.cuda.get_device_name(0), torch.__version__) == RECORD_ENV


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def arm_cfg(arm):
    name, replan = ARMS[arm]
    return EvalConfig(name=name, planner='live', models=MODELS, replan=replan, controller='nav_pid',
                      label='mission', latency_replay=R1L_REPLAY if arm == 'R1L' else None, build_lock='x')


def speed_pi_loop(pi, steps=20000, dt=0.002):         # as in the goldens generator
    v, out = 0.0, []
    for k in range(steps):
        th, br = pi.advance(v, (6.0, 3.0, 0.5, 4.0, 0.0)[k // 2000 % 5], dt)
        out.append((th, br))
        v = max(v + dt * (3.0 * th - 8.0 * br - 0.05 * v), 0.0)
    return np.asarray(out, np.float64)


def replay_run(run):
    """(decisions, mismatches) of nav.ladder over one recorded run's decisions, rng drawn across them, pick = the
    recorded index; a mismatch is (frame, got, expected)."""
    mid, arm = run.name.split('__')
    goals = [np.asarray(g, float) for g in json.loads((DATA / NAV / f'missions/{mid}.json').read_text())['goals']]
    routes = {r['route_id']: r for r in json.loads((run / 'routes.json').read_text())}
    rng, bad = np.random.default_rng(arm_cfg(arm).seed(mid)), []
    decisions = json.loads((run / 'decisions.json').read_text())
    for d in decisions:
        r, rec = nav.ladder(np.asarray(d['pose'], float), goals[d['goal_index']], rng, lambda c, p, g: (d['index'], {}))
        want = dict(n=d['n_candidates'], tries=d['tries'], half=d['plan_half_m'], aim=d['aim_xy'], route=d.get('index'),
                    rung=next((k for k in ('used_fallback_base', 'surrogate_goal', 'straight_ahead') if k in d), None))
        got = dict(n=rec['n_candidates'], tries=rec['tries'], half=rec['plan_half_m'], aim=rec['aim_xy'], route=None,
                   rung={'fallback_base': 'used_fallback_base'}.get(rec.get('rung'), rec.get('rung')))
        if r is not None:
            ro = routes[d['route_id']]
            same = r['meta'] == ro['meta'] and all(np.array_equal(r[k], ro[k]) for k in ('waypoints', 'speeds',
                                                                                            'stations'))
            got['route'] = d['index'] if same else 'differs'
        if got != want:
            bad.append((d['frame'], got, want))
    return len(decisions), bad


class Follower:
    SetDesiredSpeed = Synchronize = Advance = lambda self, *a: None

    def GetInputs(self):
        return SimpleNamespace(m_steering=0., m_throttle=0., m_braking=0.)


class End(Exception):
    pass


class Replay:
    """A Sim that replays a recorded mission (no physics): rows k from the trajectory, the post-frame chassis pose of
    frame k = pose[k + 1] (roll, pitch = state[k + 1] columns 2-3); stops before the last frame's end."""
    substeps, dt, ground = 1, DT, 'rigid'

    def __init__(self, z, start):
        self.st, self.ac, self.po, self.start = z['state'], z['action'], z['pose'], np.asarray(start, float)
        self.top, self.k, self.cur = -E.SETTLE_FRAMES - 1, -1, z['pose'][0]
        self.vehicle = SimpleNamespace(GetSpeed=lambda: 0.)

    def chassis_pose(self):                             # the chassis now (pose, z)
        return np.array([float(v) for v in self.cur]), 0.

    def follower(self, route, *, initialize, z=None):
        return Follower()

    def now(self):
        return 0.

    def ref_xy(self):
        self.top += 1
        return self.po[self.top][:2] if self.top >= 0 else self.start

    def measure(self, k, t, u):
        return self.st[k], self.ac[k], self.po[k], 0.

    def after_frame(self):
        self.k += 1
        if self.k + 1 >= len(self.st):
            raise End
        self.cur, s = self.po[self.k + 1], self.st[self.k + 1]
        return After(np.asarray(self.cur, float), float(s[2]), float(s[3]))

    sync = advance = lambda self, *a: None
    power_kw = lambda self: 0.                          # noqa: E731


class ReplayHook(nav.NavHook):
    """NavHook with the recorded decisions in place of rendering and planning."""
    EMPTY = dict(z=np.full((nav.N, nav.N), np.nan, np.float32), cover=np.zeros((nav.N, nav.N), np.float32))

    def __init__(self, cfg, task, case, env, run):
        self.rec, self.used = json.loads((run / 'decisions.json').read_text()), 0
        self.rroutes = {r['route_id']: r for r in json.loads((run / 'routes.json').read_text())}
        super().__init__(cfg, task, case, env)

    def load_models(self, cfg, env):
        return {}

    def route_at(self, pose, goal, t):
        d, self.grid = self.rec[self.used], self.EMPTY
        self.used += 1
        if (d['frame'], d['pose']) != (t, pose.tolist()):
            raise AssertionError(f'decision {self.used - 1}: frame {t} pose {pose.tolist()}, recorded {d["frame"]}')
        r = self.rroutes.get(d.get('route_id'))
        return r and {**{k: np.asarray(r[k], float) for k in ('waypoints', 'speeds', 'stations')},
                      'meta': r['meta']}, {}


def schedule_run(args):
    """(recorded, replayed) decision schedules and per-frame waypoint indexes of one run (frames before the last)."""
    task, run, arm = args
    env = Env.from_environ()
    case = task.read_case()
    hook = ReplayHook(arm_cfg(arm), task, case, env, run)
    with np.load(run / 'trajectory.npz') as z:
        z = {k: z[k] for k in ('state', 'action', 'pose', 'leg')}
    stops = E.Stops(hook.goals[0], hook.radius, mission_s=nav.MISSION_S)
    try:
        status = E.drive_episode(Replay(z, case['layout']['start_xy']), hook.route0, NavPID(), stops, hook,
                                 launch_check=False).status
    except End:
        status = None
    n = len(z['state'])
    key = ('frame', 'trigger', 'goal_index', 'route_id', 'latency_charged_s')
    rec = [tuple(d.get(k) for k in key) for d in hook.rec if d['frame'] < n]
    got = [tuple(d.get(k) for k in key) for d in hook.decisions]
    return dict(run=run.name, status=status, n=n, rec=rec, got=got, leg=hook.frame_leg == z['leg'][:n - 1].tolist())


class TestControllers(unittest.TestCase):
    def test_speed_pi_golden(self):
        """controllers.SpeedPI == nav_online.SpeedPI over 20,000 closed-loop steps (saturation, both brake branches)."""
        tb = speed_pi_loop(SpeedPI())
        g = GOLD['speed_pi']
        self.assertEqual((sha(tb), int((tb[:, 1] > 0).sum())), (g['out'], g['braking_steps']))

    def test_nav_pid(self):
        """Steering clamped as the stock PID (0 while settling); throttle and brake from SpeedPI from frame 0 on; the
        PI advances on the vehicle speed and the frame's desired speed; no SetDesiredSpeed after the settle."""
        c = NavPID()
        ep = SimpleNamespace(k=-1, prev_steer=0., desired=4., sim=SimpleNamespace(dt=.002, vehicle=SimpleNamespace(
            GetSpeed=lambda: 1.)))
        c.reset(ep)
        u = c.inputs(ep, SimpleNamespace(m_steering=.5, m_throttle=.7, m_braking=.1))
        self.assertEqual((u.m_steering, u.m_throttle, u.m_braking), (0., .7, .1))
        ep.k = 0
        c.after_sync(ep)
        pi = SpeedPI()
        pi.advance(1., 4., .002)
        u = c.inputs(ep, SimpleNamespace(m_steering=.5, m_throttle=.7, m_braking=.1))
        self.assertEqual((u.m_steering, u.m_throttle, u.m_braking), (.004, pi.throttle, pi.braking))
        self.assertTrue(c.owns_speed)


class TestPools(unittest.TestCase):
    def test_fixed_pool_and_bound(self):
        """A rescue pool: the base at 2 m/s first (candidate fallback_base, never validated), then fixed-speed draws
        validated with the bound, tries reported as 6n; the planning bound widens with the pose to at most 45 m."""
        pose, goal = np.array([0., 0., 0.]), np.array([20., 0.])
        base = nav.base_route(pose, goal, nav.m1_valid(37.))
        never = lambda r, p: False                      # noqa: E731
        out, tries = nav.pool(base, pose, np.random.default_rng(0), never, fixed=2.)
        self.assertEqual((len(out), tries, out[0]['meta']['candidate']), (1, 6 * 256, 'fallback_base'))
        np.testing.assert_array_equal(out[0]['speeds'], np.minimum(2., np.sqrt(4 * (base['stations'][-1]
                                                                                  - base['stations']))))
        out, tries = nav.pool(base, pose, np.random.default_rng(0), nav.m1_valid(37.))
        k = sum(map(nav.m1_valid(37.), nav.family_anchors(base), [pose] * 9))     # the +-4 m anchors bend too much
        self.assertEqual((k, len(out), [r['meta']['candidate'] for r in out]),
                         (3, 256, ['n2_anchor'] * 3 + ['n2_wide'] * 253))
        self.assertEqual([nav.widened_bound(np.array([x, 0., 0.])) for x in (10., 36., 41., 44.)],
                         [37., 38.5, 43.5, 45.])
        self.assertEqual(nav.passes(np.array([10., 0., 0.])), ((37., 36.5), (39.5, 39.)))

    def test_replay_miss_raises(self):
        """A periodic decision at a frame the recorded run has no latency for raises (the original measured anew)."""
        hook = object.__new__(nav.NavHook)
        hook.__dict__.update(periodic=True, pending=None, next=0, period=20, replay={5: .1}, goal_i=0, legs=[dict(
            start_frame=0)], frame_leg=[], goals=[np.array([30., 0.])], last=0, decisions=[{}])
        hook.decide = lambda ep, trigger, t: dict(waypoints=np.zeros((3, 2)))
        ep = SimpleNamespace(k=6, stops=SimpleNamespace(check=lambda ep: None), wp=0, xy=np.zeros((9, 2)),
                             after=SimpleNamespace(pose=np.array([10., 10., 0.])))
        with self.assertRaisesRegex(RuntimeError, 'latency replay: no recorded decision at frame 7'):
            hook.end_frame(ep)
        hook.next, ep.k = 0, 4                          # frame 5: charged 0.1 s = 2 frames, pending until frame 7
        self.assertIsNone(hook.end_frame(ep))
        self.assertEqual((hook.pending[1], hook.decisions[-1]['latency_charged_s']), (7, .1))


@unittest.skipUnless(DATA, 'needs NEDM_DATA (sensor_v2 vehicle frames, matched_Dabs checkpoints)')
class TestSensing(unittest.TestCase):
    FRAMES = 'artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/vehicle_frames'

    def frame(self, name):
        with np.load(DATA / self.FRAMES / name / 'observation.npz') as z:
            return z['depth_m']

    def test_grids(self):
        """grid_from_depth == sensor_map_v2.grid_from_arrays (blank RGB) on the 20 frames, whose camera is M1's."""
        for name, g in GOLD['grids'].items():
            with self.subTest(name):
                self.assertEqual({k: g['camera'][k] for k in nav.CAMERA}, nav.CAMERA)
                got = nav.grid_from_depth(self.frame(name))
                self.assertEqual({k: sha(got[k]) for k in ('z', 'range_m', 'sec', 'rgb', 'cover')},
                                 {k: g[k] for k in ('z', 'range_m', 'sec', 'rgb', 'cover')})
        self.assertEqual(len(GOLD['grids']), 20)

    def test_decisions(self):
        """Pool, footprint, corridors12, path heights == nav_online on 6 frames; the scorer == GridRiskModel (cpu with
        torch threads 1; cuda on the record GPU)."""
        hook = SimpleNamespace()
        nav.NavHook.load_models(hook, SimpleNamespace(models=MODELS), dataclasses.replace(Env.from_environ(), device='cpu'))
        cpu = hook.ens
        with np.load(HERE / 'goldens/nav/nav.npz') as z:
            arrays = {k: z[k] for k in z.files}
        for name, g in GOLD['decisions'].items():
            with self.subTest(name):
                grid = nav.grid_from_depth(self.frame(name))
                pose, goal = np.asarray(g['pose']), np.asarray(g['goal'])
                rng = np.random.default_rng(int(hashlib.md5(name.encode()).hexdigest()[:8], 16))
                valid = nav.m1_valid(nav.widened_bound(pose))
                cands, tries = nav.pool(nav.base_route(pose, goal, valid), pose, rng, valid)
                self.assertEqual((len(cands), tries, nav.widened_bound(pose)), (g['n'], g['tries'], g['half']))
                self.assertEqual(sha(np.concatenate([np.r_[r['waypoints'].ravel(), r['speeds']] for r in cands])),
                                 g['routes'])
                excl = nav.footprint(pose)
                X, L, inval = nav.corridors12(cands, grid, excl)
                self.assertEqual((sha(excl), sha(X)), (g['footprint'], g['X']))
                np.testing.assert_array_equal(L, arrays[f'{name}/L'])
                np.testing.assert_array_equal(inval, arrays[f'{name}/inval'])
                obs = json.loads((DATA / self.FRAMES / name / 'observation.json').read_text())
                np.testing.assert_array_equal(nav.path_heights(cands[g['index']]['waypoints'], grid,
                                                               obs['chassis_ref_z_m'] - .75), arrays[f'{name}/heights'])
                torch.set_num_threads(1)
                for dev, ens in [('cpu', cpu)] + ([('cuda', nav.Ensemble.load(cpu.paths, 'cuda'))] if GPU else []):
                    z = ens.logits(X[:, hook.sel], geom5(pose, goal, L), hook.zs).mean(0)
                    np.testing.assert_array_equal(z, arrays[f'{name}/z_{dev}'])
                self.assertEqual(int(np.argmin(arrays[f'{name}/z_cuda'])), g['index'])


@unittest.skipUnless(DATA and CACHE, 'needs NEDM_DATA and NEDM_RELEASE_CACHE (released mission runs)')
class TestRecordedRuns(unittest.TestCase):
    RUNS = 'local_luffy/runs'

    def runs(self, pattern):
        return sorted((DATA / NAV / self.RUNS).glob(pattern))

    def test_replay_decisions(self):
        """nav.ladder replays the 194 W decisions and 3 periodic runs (208 decisions: 5 fallback, 14 surrogate, 5 no
        route, 1 reroute) exactly."""
        runs = self.runs('*__W') + [DATA / NAV / self.RUNS / r for r in (                # surrogate, fallback,
            'g204_nav_001__R1', 'f104_nav_001__R2', 'g223_nav_001__R1L')]                 # reroute, no route
        with get_context('spawn').Pool(min(16, os.cpu_count())) as p:   # torch is loaded
            res = p.map(replay_run, runs, chunksize=1)
        self.assertEqual([b for _, b in res if b], [])
        self.assertEqual(sum(n for n, _ in res[:30]), 194)

    def test_schedule(self):
        """The episode loop and NavHook reproduce every recorded decision (frame, trigger, waypoint, route, latency)
        and the per-frame waypoint index of all 120 runs, with no stop before the last frame."""
        from nedm.traversing.evaluation.suites import load_suite
        tasks = {t.id: t for t in load_suite('missions30')}
        jobs = [(tasks[r.name.split('__')[0]], r, r.name.split('__')[1]) for r in self.runs('*__*')]
        with get_context('spawn').Pool(min(16, os.cpu_count())) as p:   # torch is loaded
            res = p.map(schedule_run, jobs, chunksize=1)
        self.assertEqual(len(res), 120)
        for r in res:
            with self.subTest(r['run']):
                self.assertEqual((r['status'], r['got'], r['leg']), (None, r['rec'], True))
        self.assertEqual(sum(len(r['rec']) for r in res), 6032)         # 3 decisions at a final frame are not replayed

    def test_nav_analyze(self):
        """mission labels of the released outcomes reproduce nav_analyze's summary.json per arm, with the paired
        bootstrap (seed 0, 20,000 resamples) of the time and waypoints differences against W."""
        summ = json.loads((DATA / NAV / 'local_luffy/summary.json').read_text())
        rows = {}
        for run in self.runs('*__*'):
            mid, arm = run.name.split('__')
            o = json.loads((run / 'mission_outcome.json').read_text())
            lab = drive_labels(released(run, 'mission'), 'mission')
            rows[arm, mid] = dict(lab, success=lab['status'] == 'mission_complete', time=o['total_time_s'])
        want = dict(W=(27, 191, 4, 5), R2=(25, 186, 9, 12), R1=(25, 179, 10, 11), R1L=(22, 173, 9, 11))
        for arm, (complete, wp, slides, unsafe) in want.items():
            sel = [r for (a, _), r in rows.items() if a == arm]
            got = (sum(r['success'] for r in sel), sum(r['waypoints_reached'] for r in sel),
                   sum(r['backward_slide'] for r in sel), sum(r['code'] != 'S' for r in sel))
            s = summ['by_arm'][arm]
            self.assertEqual(got, (complete, wp, slides, unsafe))
            self.assertEqual(got, (s['complete'], round(s['waypoint_rate'] * 199), s['any_slide'], s['unsafe']))
            self.assertEqual(float(np.median([r['elapsed_s'] for r in sel])), s['median_time_s'])
            if arm != 'W':
                for k, key in (('time', 'time'), ('waypoints_reached', 'waypoints')):
                    d = np.array([rows[arm, m][k] - rows['W', m][k] for m in sorted(m for a, m in rows if a == arm)],
                                 float)
                    boot = d[np.random.default_rng(0).integers(0, len(d), (20000, len(d)))].mean(1)
                    pv = summ['paired_vs_reference'][arm]
                    self.assertEqual((float(d.mean()), [float(np.percentile(boot, q)) for q in (2.5, 97.5)]),
                                     (pv[f'{key}_mean_diff' + ('_s' if key == 'time' else '')], pv[f'{key}_ci']))


if __name__ == '__main__':
    unittest.main()
