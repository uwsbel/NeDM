"""Stop rules (episode.Stops, Blockage): CI synthetic drives for the rule timing and order; A3 (release) replays
released rigid drives through Stops and requires the recorded status and frame count.

A3 replays the stored rows: frame k's post-frame pose is pose[k + 1] (terminal_pose after the last frame; the chassis
does not move between a frame's end and the next frame's substep-0 capture) and its roll and pitch are state[k + 1]
columns 2-3 (terminal_state). It takes the 10 lowest md5(run folder) drives of every (run root, status, near-stop)
class of the released rigid drive folders; over all 45,791 released rigid drives this replay agreed 45,791 / 45,791
(goal 44,377, blockage 1,141 incl. 21 near-stop, timeout 211, bounds 42, rollover 20).

    NEDM_DATA=... NEDM_RELEASE_CACHE=... PYTHONPATH=src python -m unittest discover -s tests/traversing/evaluation \\
        -p test_stops.py -v
"""
import csv
import gzip
import hashlib
import json
import unittest
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from nedm.traversing.evaluation.episode import Stops
from nedm.traversing.evaluation.sim import After

from .common import CACHE, CACHE_DIR, DATA

GOAL = np.array([30., 0.])
ITEMS = ('unseen_arena_drive_folders_rigid', 'shared_model_drive_folders_rigid', 'tracker_drive_folders')


def replay(z, goal, radius, near_stop=False):
    """(status, frames, near_stop_fired) of stored rows replayed through Stops."""
    st, ac, po, pk = z['state'], z['action'], z['pose'], z['parked']
    n, stops = len(st), Stops(goal, radius, near_stop=near_stop)
    ep = SimpleNamespace(state=[], action=[], pose=[], parked=[], k=0, after=None)
    for k in range(n):
        ep.k = k
        for rows, v in ((ep.state, st[k]), (ep.action, ac[k]), (ep.pose, po[k]), (ep.parked, bool(pk[k]))):
            rows.append(v)
        p, s = (po[k + 1], st[k + 1]) if k + 1 < n else (z['terminal_pose'], z['terminal_state'])
        ep.after = After(np.asarray(p, float), float(s[2]), float(s[3]))
        if status := stops.check(ep):
            return status, k + 1, stops.near_stop_fired
    return None, n, stops.near_stop_fired


def synthetic(n, xy=(0., 0.), thr=.5, vx=0., roll=0., parked=False):
    """A drive standing at xy for n frames (+ a terminal row) with the given throttle, vx and roll."""
    st, ac = np.zeros((n + 1, 17), np.float32), np.zeros((n, 3), np.float32)
    st[:, 0], st[:, 2], ac[:, 1] = vx, roll, thr
    po = np.tile(np.array([*xy, 0.]), (n + 1, 1))
    return dict(state=st[:n], action=ac, pose=po[:n], parked=np.full(n, parked), terminal_pose=po[n],
                terminal_state=st[n])


class TestRules(unittest.TestCase):
    def test_blockage_timing(self):
        """Stalled under throttle from the start: first qualifying window counted at 24 s, confirmed 2 s later, then
        the 8 s tail: the stop is at 34 s = 680 frames (the earliest possible)."""
        self.assertEqual(replay(synthetic(2400), GOAL, 2.5), ('prolonged_blockage_terminated', 680, False))

    def test_blockage_float32_throttle_and_parking(self):
        """throttle == f32(0.3) is not > 0.3; parked frames never qualify: both run to the 120 s timeout."""
        self.assertEqual(replay(synthetic(2400, thr=np.float32(.3)), GOAL, 2.5)[:2], ('timeout', 2400))
        self.assertEqual(replay(synthetic(2400, parked=True), GOAL, 2.5)[:2], ('timeout', 2400))

    def test_blockage_cancel(self):
        """A throttle dip at frame 600 (30 s) cancels the pending 34 s stop; the windows qualify again at frame 640
        (32.05 s), so the stop comes 2 + 8 s later at 42.05 s = 841 frames."""
        z = synthetic(2400)
        z['action'][600, 1] = 0.
        self.assertEqual(replay(z, GOAL, 2.5)[:2], ('prolonged_blockage_terminated', 841))

    def test_near_stop_held_and_tracker_only(self):
        """|vx| < 0.3 with zero throttle: the native rules never fire; the held PID and the tracker stop after 800 frames."""
        z = synthetic(2400, thr=0.)
        self.assertEqual(replay(z, GOAL, 2.5, near_stop=True), ('prolonged_blockage_terminated', 800, True))
        self.assertEqual(replay(z, GOAL, 2.5, near_stop=False), ('timeout', 2400, False))

    def test_order(self):
        """rollover before goal before bounds; bounds on the post-frame pose, before 40 frames are recorded."""
        self.assertEqual(replay(synthetic(5, xy=(30., 0.), roll=1.1), GOAL, 2.5)[:2], ('rollover', 1))
        self.assertEqual(replay(synthetic(5, xy=(30., 0.), roll=1.0), GOAL, 2.5)[:2], ('goal_reached', 1))
        self.assertEqual(replay(synthetic(5, xy=(40.5, 0.)), np.array([39., 0.]), 2.5)[:2], ('goal_reached', 1))
        self.assertEqual(replay(synthetic(5, xy=(40.5, 0.)), GOAL, 2.5)[:2], ('terrain_bounds_exit', 1))
        self.assertEqual(replay(synthetic(10), GOAL, 2.5, near_stop=False)[:2], (None, 10))


@unittest.skipUnless(DATA and CACHE, 'A3 needs NEDM_DATA (the release restore base) and the release cache')
class TestReleasedDrives(unittest.TestCase):
    def test_a3_replay(self):
        from nedm.traversing.evaluation.suites import load_suite
        cases = {t.id: t for s in ('f104_800', 'tracker423') for t in load_suite(s)}
        classes = defaultdict(list)
        for item in ITEMS:
            with gzip.open(CACHE_DIR / 'traversing/evaluation' / item / 'index.csv.gz', 'rt') as f:
                runs = [r['path'][:-13] for r in csv.DictReader(f) if r['path'].endswith('/outcome.json')
                        and 'crm' not in r['path'].rsplit('/', 2)[0]]               # rigid run roots only
            for run in runs:
                o = json.loads((DATA / run / 'outcome.json').read_text())
                ext = (o.get('ext') or {})
                fired = (ext.get('near_stop_rule') or {}).get('fired')
                classes[(run.rsplit('/', 1)[0], o['status'], fired)].append((run, o, ext.get('mode')))
        n = 0
        for key, rows in sorted(classes.items(), key=str):
            for run, o, mode in sorted(rows, key=lambda r: hashlib.md5(r[0].encode()).hexdigest())[:10]:
                c = json.loads((DATA / run / 'case.json').read_text()) if (DATA / run / 'case.json').exists() else \
                    cases[run.rsplit('/', 1)[1].split('__')[0]].read_case()
                with np.load(DATA / run / 'trajectory.npz') as z, self.subTest(run=run):
                    got = replay(z, np.asarray(c['goal_xy'], float), c.get('goal_radius_m', 2.5),
                                 near_stop=mode in ('pid_held', 'policy'))
                    self.assertEqual(got[:2], (o['status'], o['frames']))
                    self.assertEqual(got[2], bool(key[2]))
                n += 1
        self.assertGreaterEqual(n, 150)
        self.assertTrue({'goal_reached', 'rollover', 'terrain_bounds_exit', 'prolonged_blockage_terminated',
                         'timeout'} <= {k[1] for k in classes} and any(k[2] for k in classes))


if __name__ == '__main__':
    unittest.main()
