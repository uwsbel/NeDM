"""A10 (CI, no Chrono): the episode loop's call sequence against a mock Sim and follower, compared with hand-written
expected sequences for a native rigid drive and a branch switch (FINAL_DESIGN 2.2, 8.2), plus hold_clip
(gc_control.py:897-913 self-test cases).

    PYTHONPATH=src python -m unittest discover -s tests/traversing/evaluation -p test_episode.py -v
"""
import unittest

import numpy as np

from nedm.traversing.evaluation import episode as E
from nedm.traversing.evaluation.controllers import StockPID, hold_clip
from nedm.traversing.evaluation.routes import route_sha256
from nedm.traversing.evaluation.sim import After


def route(start, speed0, n=21, y=0.):
    xy = np.array([[start + i, y if i == 0 else 0.] for i in range(n - start)], float)
    st = np.r_[0., np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
    return dict(waypoints=xy, speeds=speed0 + np.arange(len(xy), dtype=float), stations=st, headings=np.zeros(len(xy)))


class Inputs:                                           # pychrono DriverInputs stand-in
    def __init__(self, s):
        self.m_steering, self.m_throttle, self.m_braking = s, .5, 0.


class Follower:
    def __init__(self, name, log, steer):
        self.name, self.log, self.steer = name, log, steer

    def SetDesiredSpeed(self, v):
        self.log.append(f'{self.name}.SetDesiredSpeed({v:g})')

    def Synchronize(self, t):
        self.log.append(f'{self.name}.Synchronize({t:g})')

    def GetInputs(self):
        self.log.append(f'{self.name}.GetInputs')
        return Inputs(self.steer)

    def Advance(self, dt):
        self.log.append(f'{self.name}.Advance')


class FakeSim:
    """2 substeps of 25 ms; the vehicle moves +6 m in x per recorded frame (none while settling)."""
    substeps, dt = 2, .025

    def __init__(self, steers):
        self.log, self.t, self.x, self.k, self.steers, self.fols = [], 0., 0., None, list(steers), []

    def follower(self, r, *, initialize):
        self.fols.append(Follower(f'fol{len(self.fols)}', self.log, self.steers.pop(0)))
        self.log.append(f'follower(start={r["waypoints"][0][0]:g}, init={initialize})')
        return self.fols[-1]

    def now(self):
        return self.t

    def ref_xy(self):
        self.log.append('ref_xy')
        return np.array([self.x, 0.])

    def sync(self, t, u):
        self.log.append(f'sync(steer={u.m_steering:.4g})')

    def measure(self, k, t, u):
        self.log.append(f'measure({k})')
        self.k = k
        return np.zeros(17, np.float32), np.array([u.m_steering, u.m_throttle, u.m_braking], np.float32), \
            np.array([self.x, 0., 0.]), 1.

    def launch_check(self, ep):
        self.log.append('launch_check')

    def power_kw(self):
        self.log.append('power_kw')
        return 1.

    def advance(self):
        self.log.append('advance')
        self.t += self.dt

    def after_frame(self):
        self.log.append('after_frame')
        self.x += 6.
        return After(np.array([self.x, 0., 0.]), 0., 0.)

    def terminal(self, u):
        self.log.append('terminal')
        return np.ones(17, np.float32)


def settle_frame(t0):                                   # hand-written: k < 0, steering forced to 0, nothing recorded
    return ['ref_xy', 'fol0.SetDesiredSpeed(0)',
            f'fol0.Synchronize({t0:g})', 'fol0.GetInputs', 'sync(steer=0)', 'fol0.Advance', 'advance',
            f'fol0.Synchronize({t0 + .025:g})', 'fol0.GetInputs', 'sync(steer=0)', 'fol0.Advance', 'advance']


def frame(fol, k, t0, desired, steers, launch=False):  # hand-written: k >= 0, record at substep 0
    return (['ref_xy', f'{fol}.SetDesiredSpeed({desired:g})',
             f'{fol}.Synchronize({t0:g})', f'{fol}.GetInputs', f'sync(steer={steers[0]:.4g})', f'measure({k})']
            + (['launch_check'] if launch else [])
            + ['power_kw', f'{fol}.Advance', 'advance',
               f'{fol}.Synchronize({t0 + .025:g})', f'{fol}.GetInputs', f'sync(steer={steers[1]:.4g})', 'power_kw',
               f'{fol}.Advance', 'advance', 'after_frame'])


class TestLoop(unittest.TestCase):
    def setUp(self):
        self.settle, E.SETTLE_FRAMES = E.SETTLE_FRAMES, 2   # two settle frames keep the sequences short

    def tearDown(self):
        E.SETTLE_FRAMES = self.settle

    def drive(self, steers, hook=None):
        sim = FakeSim(steers)
        ep = E.drive_episode(sim, route(0, 1.), StockPID(), E.Stops((20., 0.), 2.5), hook)
        return sim, ep

    def test_native_rigid(self):
        """Settle with steering 0 and desired speed 0, record at substep 0, launch check at frame 0, the steering
        clamp +-2 dt per substep from the previous substep, the goal after frame 2 (x = 18), terminal Synchronize."""
        sim, ep = self.drive([.3])
        want = (['follower(start=0, init=True)'] + settle_frame(0.) + settle_frame(.05)
                + frame('fol0', 0, .1, 1, (.05, .1), launch=True)        # x 0 -> waypoint 0, speed 1
                + frame('fol0', 1, .15, 7, (.15, .2))                    # x 6 -> waypoint 6, speed 7
                + frame('fol0', 2, .2, 13, (.25, .3)) + ['terminal'])    # x 12 -> waypoint 12; then x 18: goal
        self.assertEqual(sim.log, want)
        a = ep.arrays()
        self.assertEqual((ep.status, len(a['state']), ep.branch), ('goal_reached', 3, None))
        self.assertEqual(a['desired_speed_mps'].tolist(), [1., 7., 13.])
        np.testing.assert_array_equal(a['action'][:, 0], np.float32([.05, .15, .25]))
        self.assertEqual(a['positive_work_kj_per_interval'].tolist(), [.025 + .025] * 3)
        self.assertEqual(ep.total_work, (.05 + .05) + .05)          # frame by frame, sequential
        self.assertEqual(a['terminal_pose'].tolist(), [18., 0., 0.])
        self.assertEqual(a['terminal_state'].tolist(), [1.] * 17)

    def test_branch_switch(self):
        """F = 2: frames 0-1 on the approach follower; after frame 1's stop check a fresh follower without
        Initialize() on the branch route; frame 2 searches the new route from waypoint 0 and its steering continues
        from the clamped value (0.2 -> 0.15 towards the new follower's -0.4); the old follower is never called again."""
        b = route(12, 10., y=.5)
        sim, ep = self.drive([.3, -.4], E.Branch(2, b, np.array([20., 0.])))
        want = (['follower(start=0, init=True)'] + settle_frame(0.) + settle_frame(.05)
                + frame('fol0', 0, .1, 1, (.05, .1), launch=True)
                + frame('fol0', 1, .15, 7, (.15, .2))
                + ['follower(start=12, init=False)']
                + frame('fol1', 2, .2, 10, (.15, .1)) + ['terminal'])
        self.assertEqual(sim.log, want)
        self.assertEqual(ep.status, 'goal_reached')
        self.assertEqual(ep.branch, dict(frame=2, pose=[12., 0., 0.], start_error_m=.5, route_sha256=route_sha256(b)))
        self.assertEqual(ep.old_followers, sim.fols[:1])
        self.assertIs(ep.route, b)

    def test_branch_refusals(self):
        far = route(12, 10.)
        far['waypoints'][-1] = [20.5, 0.]
        with self.assertRaises(ValueError):
            E.Branch(2, far, np.array([20., 0.]))                   # route end 0.5 m from the case goal
        with self.assertRaises(RuntimeError):                       # start 3 m from the vehicle at F
            self.drive([.3, -.4], E.Branch(2, route(15, 10.), np.array([20., 0.])))

    def test_branch_after_the_end(self):
        """A drive that stops before F is never branched (its own status is the label)."""
        sim, ep = self.drive([.3, -.4], E.Branch(5, route(12, 10.), np.array([20., 0.])))
        self.assertEqual((ep.status, ep.branch, len(sim.fols)), ('goal_reached', None, 1))


class TestRecord(unittest.TestCase):
    def test_roundtrip_and_labels(self):
        """record.json + trajectory.npz + route.json round-trip, are read by labels (rollback and tracker), and a
        finished drive is never overwritten."""
        import tempfile
        from .tables import code
        from nedm.traversing.evaluation.runner import Record
        n = 30
        a = dict(state=np.zeros((n, 17), np.float32), action=np.zeros((n, 3), np.float32),
                 pose=np.c_[np.linspace(0., 20., n), np.zeros((n, 2))], parked=np.zeros(n, bool),
                 desired_speed_mps=np.full(n, 2.), terminal_pose=np.array([20., 0., 0.]))
        a['state'][:, 0] = 2.
        rec = Record('t', 'arm', 'goal_reached', True, frames=n, elapsed_s=n * .05, goal_time_s=n * .05,
                     positive_work_kj=1.5, route_sha256='sha', provenance=dict(h='x'), arrays=a)
        with tempfile.TemporaryDirectory() as d:
            rec.save(d, [('route.json', route(0, 1.))])
            back = Record.load(d)
            self.assertEqual({k: v for k, v in vars(back).items() if k != 'arrays'},
                             {k: v for k, v in vars(rec).items() if k != 'arrays'})
            self.assertTrue(all(np.array_equal(back.arrays[k], v) and back.arrays[k].dtype == v.dtype
                                for k, v in a.items()))
            self.assertEqual((code(d, 'rollback'), code(d, 'tracker')), ('S', 'S'))
            with self.assertRaises(FileExistsError):
                rec.save(d)


class TestHoldClip(unittest.TestCase):
    def test_selftest_cases(self):
        for cmd, prev, kw, want in (((.5, 1.3, -.2), 0., {}, (.1, 1., 0.)), ((-.5, .4, .2), -.35, {}, (-.45, .4, .2)),
                                    ((0., 0., 1.), .95, {}, (.85, 0., 1.)), ((1.5, .5, 0.), .95, {}, (1., .5, 0.)),
                                    ((.3, .2, 0.), .25, dict(rate=0.), (.25, .2, 0.))):
            got = hold_clip(cmd, prev, **kw)
            self.assertTrue(np.allclose(got, want, atol=1e-12, rtol=0) and all(type(v) is float for v in got))
        for bad in ((float('nan'), 0, 0), (0, float('inf'), 0)):
            with self.assertRaises(ValueError):
                hold_clip(bad, 0.)


if __name__ == '__main__':
    unittest.main()
