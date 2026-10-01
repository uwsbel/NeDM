"""Held PID and tracker (controllers.HeldPID, Tracker).

goldens/controllers was written by the ORIGINAL code at 901d6c9 (gc_control.py, and the per-frame controller logic of
gen_collect_ext.GenExt.command and crm_collect_ext.run replayed verbatim; generator
the goldens generator (goldens/README.md), which rewrites them byte for byte): a 2,000-command hold_clip chain, the 64 check_actors observations
through gc_control.NumpyActor, and a synthetic 120-frame drive along a released route (poses, the state captured
before Synchronize, the recorded state, the shadow follower's commands) with the held triples and observations of
pid_held and policy on both grounds.
  A10 (CI)       hold_clip chain; held PID flows (rigid float64 vs soil float32 feedback); tracker frame-0 padding,
                 history order and layout refusal (stub actor); the loop's call order with held controllers.
  A9 (release)   main's NumpyActor on the 64 observations and the tracker flows: bit-exact on the goldens' machine (CPU,
                 numpy, BLAS), else within 4e-16 (actor) or 1e-9 (closed-loop flows). Released drives of the 10 lowest
                 md5 feasible routes replayed frame by frame: rigid held PID (ext_control shadow -> held float64, exact
                 anywhere) and the tracker, float32 actions equal to the recording (soil: recorded float32 feedback;
                 rigid: its own float64 chain, exact on the recording machine, else within 1e-6).
The one-off over every recorded drive (2026-09-30): rigid tracker with the recorded float64 feedback 220,446 / 220,446
frames bit-exact on the cluster login node (EPYC 7V13, the recording CPU and venv) and 214,691 / 220,446 on luffy
(max 4.4e-16); soil tracker 146,456 / 146,456 float32 actions; rigid held PID 185,992 / 185,992.

    NEDM_DATA=... NEDM_RELEASE_CACHE=... PYTHONPATH=src python -m unittest discover -s tests/traversing/evaluation \\
        -p test_controllers.py -v
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from nedm.traversing.evaluation import controllers as C
from nedm.traversing.evaluation import episode as E

from .common import CACHE, DATA
from .test_episode import FakeSim, frame, route, settle_frame

GOLD = Path(__file__).parent / 'goldens/controllers'
Z = dict(np.load(GOLD / 'controllers.npz'))
META = json.loads((GOLD / 'controllers.json').read_text())
ROUTE = {k: Z[f'route_{k}'] for k in ('waypoints', 'speeds', 'stations', 'headings')}
ACTOR = 'data:artifacts/traverse/generalist_20260921/B_tracker/ppo_v2/actor.npz'
SETTLE = np.array([0., 0., 1.])


def machine():
    """What float64 BLAS results depend on (the goldens' 'machine')."""
    import platform
    model = next((ln.split(':', 1)[1].strip() for ln in open('/proc/cpuinfo') if ln.startswith('model name')), '?')
    blas = np.show_config(mode='dicts')['Build Dependencies']['blas']
    return dict(cpu=model, numpy=np.__version__, blas=f"{blas.get('name')} {blas.get('version')}",
                python=platform.python_version())


SAME = machine() == META['machine']
# the rigid tracker drives' machine (mi2104x EPYC 7V13, the cluster venv): replays there are bit-exact (A9 one-off)
RECORDING = dict(cpu='AMD EPYC 7V13 64-Core Processor', numpy='2.4.3', blas='scipy-openblas 0.3.31.dev')


def inputs(cmd):
    return SimpleNamespace(m_steering=float(cmd[0]), m_throttle=float(cmd[1]), m_braking=float(cmd[2]))


class Capture:
    """Sim stand-in at the frame top: measure() = (state before Synchronize, -, pose, -), transmission column 15."""

    def __init__(self, ground, pre, pose):
        self.ground, self.pre, self.pose, self.transmission, self.k = ground, pre, pose, self, None

    def now(self):
        return 0.

    def measure(self, k, t, u):
        self.k = k
        return self.pre[k].copy(), None, self.pose[k].copy(), 0.

    def GetOutputMotorshaftSpeed(self):
        return float(self.pre[self.k][15])


def run(ctrl, ground, rte, pose, pre, rec, shadow=None, recorded=None):
    """`ctrl` frame by frame as drive_episode calls it (frame_top, the frame's record, frame_end). The record is
    (rec[k], float32(held)) or the `recorded` actions of a released drive. Returns (held (T, 3), observations)."""
    ep = SimpleNamespace(sim=Capture(ground, pre, pose), route=rte, k=0, state=[], action=[])
    ep.fol = SimpleNamespace(GetInputs=lambda: inputs(shadow[ep.k]))
    held, obs = [], []
    if isinstance(ctrl, C.Tracker):
        observe = ctrl.observe
        ctrl.observe = lambda p, s: obs.append(observe(p, s)) or obs[-1]
    with mock.patch.object(C, 'driver_inputs', inputs):
        ctrl.reset(ep)
        for k in range(len(pre)):
            ep.k = k
            ctrl.frame_top(ep)
            held.append(ctrl.held)
            ep.state.append(rec[k])
            ep.action.append(np.array(ctrl.held, np.float32) if recorded is None else recorded[k])
            ctrl.frame_end(ep)
    return np.asarray(held), np.asarray(obs)


def stub_actor(d, **layout):
    """A one-layer zero-weight actor: act = (0, 0.5, 0.5) whatever the observation."""
    p = Path(d) / 'stub.npz'
    np.savez(p, format=np.array('gc_actor_v1'), num_obs=np.int64(158), num_actions=np.int64(3), obs_mean=np.zeros(158),
             obs_var=np.ones(158), obs_eps=np.float64(.01), n_layers=np.int64(1), W0=np.zeros((3, 158)), b0=np.zeros(3),
             activation=np.array('elu'), action_center=np.array([0., .5, .5]), action_scale=np.array([1., .5, .5]),
             action_low=np.array([-1., 0., 0.]), action_high=np.ones(3),
             meta_json=np.array(json.dumps(dict(obs_layout={**C.OBS_LAYOUT, **layout}))))
    return p


def close(test, a, b, tol):
    """Bit-exact on the goldens' machine, else within tol."""
    if SAME:
        np.testing.assert_array_equal(a, b)
    else:
        test.assertLessEqual(float(np.abs(np.asarray(a) - b).max()), tol)


class TestUnits(unittest.TestCase):
    def test_hold_chain(self):
        prev, got = 0., []
        for c in Z['hold_cmds']:
            got.append(C.hold_clip(c, prev))
            prev = got[-1][0]
        np.testing.assert_array_equal(got, Z['hold_held'])

    def test_held_pid_flows(self):
        """The shadow command held per frame: rigid feeds the float64 command back, soil the float32 record."""
        for g in ('rigid', 'soil'):
            held, _ = run(C.HeldPID(), g, ROUTE, Z['pose'], Z['pre'], Z['rec'], Z['shadow'])
            np.testing.assert_array_equal(held, Z[f'{g}_pid_held_held'])
        self.assertFalse(np.array_equal(Z['rigid_pid_held_held'], Z['soil_pid_held_held']))

    def test_tracker_padding_history_layout(self):
        """Frame 0 equals the original's (no actor involved): settle action and rest state fill the history; later
        frames hold the previous 8 commands (rigid float64, soil float32) and states k-7 .. k, oldest first."""
        pre, cols = Z['pre'], C.COLS
        with tempfile.TemporaryDirectory() as d:
            for g in ('rigid', 'soil'):
                held, obs = run(C.Tracker(stub_actor(d)), g, ROUTE, Z['pose'], pre, Z['rec'])
                self.assertEqual(obs.shape, (120, 158))
                np.testing.assert_array_equal(obs[0], Z[f'{g}_policy_obs'][0])
                fed = held if g == 'rigid' else held.astype(np.float32).astype(float)
                for k in range(12):
                    S = pre[np.clip(np.arange(k - 7, k + 1), 0, None)][:, cols]
                    A = np.array([SETTLE if j < 0 else fed[j] for j in range(k - 8, k)])
                    np.testing.assert_array_equal(obs[k][62:].reshape(8, 12), S)
                    np.testing.assert_array_equal(obs[k][38:62].reshape(8, 3), A)
                    np.testing.assert_array_equal(obs[k][35:38], A[-1])
            with self.assertRaises(ValueError):
                C.Tracker(stub_actor(d, search_window=30))
        self.assertEqual((META['layout']['num_obs'], META['layout']['past_states']), (158, [62, 158]))

    def test_loop_sequence(self):
        """drive_episode with a held controller: the shadow is read (pid_held) or the state captured (tracker) at the
        frame top after SetDesiredSpeed, before the first Synchronize; the held steering is written unclamped at
        every substep (0.3 shadow: 0.1, 0.2, 0.3); the settle keeps the native zero steering."""
        class Sim(FakeSim):
            ground, transmission = 'rigid', SimpleNamespace(GetOutputMotorshaftSpeed=lambda: 0.)

        def held_frame(top, k, t0, desired, s, launch=False):
            f = frame('fol0', k, t0, desired, (s, s), launch)
            return f[:2] + [top] + f[2:]
        settle, E.SETTLE_FRAMES = E.SETTLE_FRAMES, 2
        try:
            with tempfile.TemporaryDirectory() as d, mock.patch.object(C, 'driver_inputs', inputs):
                for ctrl, tops, steers in ((C.HeldPID(), ['fol0.GetInputs'] * 3, (.1, .2, .3)),
                                           (C.Tracker(stub_actor(d)), [f'measure({k})' for k in range(3)], (0,) * 3)):
                    sim = Sim([.3])
                    ep = E.drive_episode(sim, route(0, 1.), ctrl, E.Stops((20., 0.), 2.5))
                    want = (['follower(start=0, init=True)'] + settle_frame(0.) + settle_frame(.05)
                            + held_frame(tops[0], 0, .1, 1, steers[0], launch=True)
                            + held_frame(tops[1], 1, .15, 7, steers[1]) + held_frame(tops[2], 2, .2, 13, steers[2])
                            + ['terminal'])
                    self.assertEqual(sim.log, want)
                    np.testing.assert_array_equal(ep.arrays()['action'][:, 0], np.float32(steers))
        finally:
            E.SETTLE_FRAMES = settle

    def test_make_controller(self):
        from nedm.traversing.evaluation.config import EvalConfig
        env = SimpleNamespace(file=lambda ref: ref)
        self.assertIs(type(C.make_controller(EvalConfig(name='a', controller='pid_held'), env)), C.HeldPID)
        self.assertIs(type(C.make_controller(EvalConfig(name='a', controller='nav_pid'), env)), C.NavPID)


@unittest.skipUnless(DATA, 'A9 needs NEDM_DATA (the released actor)')
class TestActor(unittest.TestCase):
    def setUp(self):
        from nedm.traversing.evaluation.config import Env
        self.env = Env.from_environ()
        self.npz = self.env.file(ACTOR)
        self.assertEqual(hashlib.sha256(self.npz.read_bytes()).hexdigest(), META['actor_sha256'])

    def test_actor64(self):
        """main's NumpyActor == gc_control.NumpyActor, one observation per call and as a batch."""
        actor = C.Tracker(self.npz).actor
        close(self, np.stack([actor.act(o) for o in Z['obs64']]), Z['act64'], 4e-16)
        close(self, actor.act(Z['obs64']), Z['act64_batch'], 4e-16)

    def test_tracker_flows(self):
        for g in ('rigid', 'soil'):
            held, obs = run(C.Tracker(self.npz), g, ROUTE, Z['pose'], Z['pre'], Z['rec'])
            close(self, held, Z[f'{g}_policy_held'], 1e-9)
            close(self, obs, Z[f'{g}_policy_obs'], 1e-9)

    @unittest.skipUnless(CACHE, 'released drives need the release cache (NEDM_RELEASE_CACHE)')
    def test_released_drives(self):
        """The 10 lowest md5 feasible routes (B2): rigid held PID from ext_control exactly; the tracker's float32
        actions equal the recorded ones on every frame (soil and rigid)."""
        from nedm.traversing.evaluation.suites import load_suite
        tasks = sorted((t for t in load_suite('tracker423', env=self.env) if t.meta['stratum'] == 'feasible'),
                       key=lambda t: hashlib.md5(t.id.encode()).hexdigest())[:10]
        b0 = 'data:artifacts/traverse/generalist_20260921/B_tracker/b0/'
        for t in tasks:
            rte = t.read_route('route')
            with self.subTest(t.id):
                d = b0 + f'rigid_runs/{t.id}__held_pid/'
                z, x = (dict(np.load(self.env.file(d + f))) for f in ('trajectory.npz', 'ext_control.npz'))
                held, _ = run(C.HeldPID(), 'rigid', rte, z['pose'], z['state'], z['state'], x['shadow_action'])
                np.testing.assert_array_equal(held, x['held_action'])
                np.testing.assert_array_equal(held.astype(np.float32), z['action'])
                for g, runs in (('soil', 'crm_v2_runs'), ('rigid', 'rigid_v2_runs')):
                    z = dict(np.load(self.env.file(b0 + f'{runs}/{t.id}__policy_v2/trajectory.npz')))
                    held, _ = run(C.Tracker(self.npz), g, rte, z['pose'], z['state'], z['state'],
                                  recorded=z['action'])
                    if g == 'soil' or {k: machine()[k] for k in RECORDING} == RECORDING:
                        np.testing.assert_array_equal(held.astype(np.float32), z['action'])
                    else:               # its own float64 chain: float32 ulps after BLAS rounding differences
                        self.assertLessEqual(float(np.abs(held.astype(np.float32) - z['action']).max()), 1e-6)


if __name__ == '__main__':
    unittest.main()
