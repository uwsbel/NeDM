"""Tests of nedm.traversing.evaluation.labels (stdlib unittest only).

CI (no data needed): float32 edge cases (spec 0.11), 45 real drives and 20 synthetic edge drives with the ORIGINAL
901d6c9 label outputs (goldens/labels, made by the goldens generator (goldens/README.md)), the real drives' cells of
the released traversing/results CSVs, table schemas and the per-table missing-drive policies.
Release (NEDM_DATA = the release restore base): A2 tracker metrics vs results_{rigid,crm}_b0v2.json per route; A1 every
present drive folder relabelled vs the study's indexes and per-group records, the 9 CSVs rebuilt from the relabelled
drives and compared byte for byte, then recount_milestones.py --check-only on the rebuilt folder.

    NEDM_DATA=<release restore root> PYTHONPATH=src python -m unittest discover -s tests/traversing/evaluation \\
        -p test_labels.py -v
"""
import concurrent.futures as cf
import csv
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nedm.traversing.evaluation import labels as L

from . import tables as TB

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESULTS = REPO / 'traversing' / 'results'
GOLD = HERE / 'goldens' / 'labels'
DATA = Path(os.environ['NEDM_DATA']) if os.environ.get('NEDM_DATA') else None
F32 = np.float32


def released(name):
    with open(RESULTS / f'{name}.csv', newline='') as f:
        return list(csv.reader(f))


def find(name, **key):
    """The released CSV row (as a dict) whose columns equal `key`."""
    head, *rows = released(name)
    hit = [dict(zip(head, r)) for r in rows if all(r[head.index(k)] == v for k, v in key.items())]
    assert len(hit) == 1, (name, key, len(hit))
    return hit[0]


def drive(n, vx=1.0, thr=0.5):
    s, a = np.zeros((n, 17), F32), np.zeros((n, 3), F32)
    s[:, 0], a[:, 1] = vx, thr
    return s, a


def pad(state4):
    s = np.zeros((len(state4), 17), F32)
    s[:, :4] = state4
    return s


class Float32Edges(unittest.TestCase):
    def test_vx_minus_0p1_is_not_backwards(self):
        s, a = drive(40, thr=0.9)
        s[30, 0] = F32(-0.1)
        self.assertTrue(float(s[30, 0]) < -0.1)            # a float64 compare would call this frame backwards
        self.assertEqual(L.point_labels(s, a, 'goal_reached'), dict(fail=0, unsafe=0, back_s=0.0, min_vx=float(F32(-0.1)), max_tilt=0.0))
        s[30, 0] = np.nextafter(F32(-0.1), F32(-1))
        self.assertEqual(L.point_labels(s, a, 'goal_reached')['back_s'], 0.05)
        self.assertEqual(TB.code(dict(status='goal_reached', state=s, action=a), 'rollback'), 's')

    def test_throttle_0p3_is_not_effortful(self):
        s, a = drive(40)
        s[30, 0], a[30, 1] = -0.2, F32(0.3)
        self.assertTrue(float(a[30, 1]) > 0.3)
        self.assertEqual(L.point_labels(s, a, 'goal_reached')['unsafe'], 0)
        a[30, 1] = np.nextafter(F32(0.3), F32(1))
        self.assertEqual(L.point_labels(s, a, 'goal_reached')['unsafe'], 1)

    def test_short_drive_guard(self):
        s, a = drive(21)
        self.assertEqual(L.point_labels(s, a, 'goal_reached'), dict(fail=0, unsafe=1, back_s=0.0, min_vx=0.0, max_tilt=0.0))
        self.assertEqual(TB.code(dict(status='timeout', state=s, action=a), 'rollback'), 'U')
        s, a = drive(22)
        self.assertEqual(TB.code(dict(status='goal_reached', state=s, action=a), 'rollback'), 'S')

    def test_min_vx_minus_0p3(self):
        s, a = drive(40, thr=0.0)
        s[30, 0] = F32(-0.3)                                # below -0.3 in float64 too: the compare is dtype-safe
        self.assertEqual((float(s[30, 0]) > -0.30, s[30, 0] > F32(-0.30)), (False, False))
        self.assertEqual(L.point_labels(s, a, 'goal_reached')['unsafe'], 1)
        m = L.mission_labels(s, a, 'mission_complete', 3, 3)   # the slide rule is strict: the two rules disagree here
        self.assertEqual((m['backward_slide'], m['code']), (0, 'S'))
        s[30, 0] = np.nextafter(F32(-0.3), F32(0))
        self.assertEqual(L.point_labels(s, a, 'goal_reached')['unsafe'], 0)
        s[30, 0] = np.nextafter(F32(-0.3), F32(-1))
        self.assertEqual(L.mission_labels(s, a, 'mission_complete', 3, 3)['code'], 's')
        s, a = drive(40, thr=0.9)
        s[19, 0] = -5.0                                     # frames before 20 never count
        self.assertEqual(L.mission_labels(s, a, 'mission_complete', 3, 3)['backward_slide'], 0)
        self.assertEqual(L.point_labels(s, a, 'goal_reached')['unsafe'], 0)

    def test_belly_20_vs_21_frames(self):
        c = np.full(80, 0.1, F32)
        c[10:30] = -0.06
        self.assertEqual(L.belly_flag(c), 0)                # 20 x 0.05 = 1.0 s is not > 1.0 s
        c[40:60] = -0.06
        self.assertEqual(L.belly_flag(c), 0)                # two runs of 20 do not add up
        c[30] = F32(-0.05)                                  # float64 compare: f32(-0.05) < -0.05, so it is deep
        self.assertEqual(L.belly_flag(c), 1)
        c[30] = np.nan
        self.assertEqual(L.belly_flag(c), 0)
        self.assertEqual(L.belly_flag(np.zeros(0, F32)), 0)

    def test_refusals(self):
        s, a = drive(40)
        with self.assertRaises(TypeError):
            L.point_labels(s.astype(float), a, 'goal_reached')      # upcasting would move the thresholds
        with self.assertRaises(ValueError):
            L.point_labels(s, a[:-1], 'goal_reached')
        with self.assertRaises(ValueError):
            L.point_labels(s, a, 'goal')
        for st, lab in (('crash', 'rollback'), ('launch_failed', 'rollback'), ('no_route', 'tracker')):
            with self.assertRaises(ValueError):
                L.drive_labels(dict(status=st), lab)
        with self.assertRaises(ValueError):
            L.drive_labels(dict(status='goal_reached', state=s, action=a), 'unsafe')
        with self.assertRaises(KeyError):                   # a belly label needs the belly record
            L.drive_labels(dict(status='goal_reached', state=s, action=a), 'rollback_belly')
        with self.assertRaises(ValueError):
            L.mission_labels(s, a, 'mission_complete', 2, 3)
        self.assertEqual(L.drive_labels(dict(status='no_route'), 'rollback_belly')['code'], 'U')

    def test_tracker_small(self):
        pose = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
        s, a = drive(3)
        m = L.tracker_metrics(pose, s, a, [[0.5, 0.0], [1.5, 1.0], [9.0, 0.0]], [1.0, 1.0, 1.0], 'timeout')
        self.assertEqual((m['code'], m['completed'], m['unsafe']), ('F', 0, 0))
        self.assertAlmostEqual(m['xtrack_station_winsor_mean_m'], np.clip([0, 1, 6], 0.1, 5.5).mean())
        self.assertEqual((m['speed_abs_err_mean_mps'], m['mean_abs_action_change']), (0.0, 0.0))
        one = L.tracker_metrics(pose[:1], s[:1], a[:1], [[3.0, 4.0]], [2.0], 'soil_breakthrough_terminated')
        self.assertEqual((one['code'], one['xtrack_station_winsor_mean_m'], one['speed_abs_err_mean_mps']), ('U', 5.0, 1.0))
        self.assertTrue(np.isnan(one['mean_abs_action_change']))
        with self.assertRaises(IndexError):
            L.tracker_metrics(pose, s, a, np.zeros((0, 2)), [1.0] * 3, 'timeout')     # no reference: raises, not nan

    def test_missions(self):
        s, a = drive(0)
        m = L.mission_labels(s, a, 'no_route_leg0', 0, 5)              # no frames: never driven, U
        self.assertEqual((m['code'], m['elapsed_s'], m['backward_slide']), ('U', 0.0, 0))
        s, a = drive(40)
        self.assertEqual(L.mission_labels(s, a, 'timeout', 2, 5)['code'], 'U')
        for bad in (('mission_complete', 4, 5), ('timeout', 5, 5), ('timeout', 0, 0), ('goal_reached', 2, 5)):
            with self.assertRaises(ValueError):
                L.mission_labels(s, a, *bad)


class GoldenDrives(unittest.TestCase):
    """45 real drives: the original functions' outputs, the study's own records and the released CSV cells; 20
    synthetic float32 edge drives through the original functions."""

    @classmethod
    def setUpClass(cls):
        g = json.loads((GOLD / 'drives.json').read_text())
        cls.drives, cls.edges = g['drives'], g['edges']
        with np.load(GOLD / 'drives.npz') as z:
            cls.arr = {k: z[k] for k in z.files}

    def mapping(self, x):
        g = self.arr
        k, e = x['key'], x['expected']
        m = dict(status=e['status'], state=pad(g[k + '_state']), action=g[k + '_action'])
        if k + '_belly' in g:
            m['belly_clearance_min_m'] = g[k + '_belly']
        if x['kind'] == 'tracker':
            m.update(pose=g[k + '_pose'], desired_speed_mps=g[k + '_desired'], reference_waypoints=g[k + '_ref'],
                     positive_work_kj=e['positive_work_kj'], near_stop_fired=e['near_stop_fired'])
        if x['kind'] == 'mission':
            m.update(goals_reached=e['goals_reached'], n_goals=e['n_goals'], mission_outcome_sha256=e['sha256'])
        return m

    def test_edge_drives_match_original(self):
        for x in self.edges:
            with self.subTest(x['name']):
                g, k, st = self.arr, x['key'], x['status']
                s, a = pad(g[k + '_state']), g[k + '_action']
                if x['kind'] == 'point':
                    got = dict(status=st, **L.point_labels(s, a, st))
                elif x['kind'] == 'belly':
                    got = dict(x['expected'], belly_flag=L.belly_flag(g[k + '_belly']))   # the run length is not ported
                else:
                    m = L.tracker_metrics(g[k + '_pose'], s, a, g[k + '_ref'], g[k + '_desired'], st)
                    got = dict(zip(('status', 'completed', 'unsafe', 'xtrack', 'speed_err', 'l1'), [m[c] for c in (
                        'status', 'completed', 'unsafe', 'xtrack_station_winsor_mean_m', 'speed_abs_err_mean_mps',
                        'mean_abs_action_change')]))
                np.testing.assert_equal(got, x['expected'])          # exact; nan == nan
        self.assertEqual(len(self.edges), 20)

    def test_point_drives_match_original_and_tables(self):
        n = 0
        for x in (x for x in self.drives if x['kind'] == 'point'):
            with self.subTest(x['run_dir']):
                e, got = x['expected'], L.drive_labels(self.mapping(x), x['label'])
                self.assertEqual({k: got[k] for k in ('fail', 'unsafe', 'back_s', 'min_vx', 'max_tilt', 'status')},
                                 {k: e[k] for k in ('fail', 'unsafe', 'back_s', 'min_vx', 'max_tilt', 'status')})
                if x['label'] == 'rollback_belly':
                    self.assertEqual(got['belly_flag'], e['belly_flag'])
                    self.assertEqual(int(got['code'] != 'S'), e['unsafe_belly'])
                    gb = L.drive_labels(self.mapping(x), 'goal_belly')
                    self.assertEqual((gb['fail'], gb['belly_flag']), (e['fail'], e['belly_flag']))
                self.assertEqual(TB.TABLES[x['table']].arms[x['column']], x['label'])
                self.assertEqual(got['code'], find(x['table'], **x['row'])[x['column']])
                n += 1
        self.assertEqual(n, 36)

    def test_tracker_drives_match_original_and_rows(self):
        for x in (x for x in self.drives if x['kind'] == 'tracker'):
            with self.subTest(x['run_dir']):
                e, got = x['expected'], L.drive_labels(self.mapping(x), 'tracker')
                self.assertEqual([got[k] for k in ('status', 'completed', 'unsafe', 'xtrack_station_winsor_mean_m',
                                                   'speed_abs_err_mean_mps', 'mean_abs_action_change', 'positive_work_kj')],
                                 [e[k] for k in ('status', 'completed', 'unsafe', 'xtrack', 'speed_err', 'l1', 'positive_work_kj')])
                row = find('m3_tracker_routes', arm=x['arm'], **x['row'])
                meta = {k: row[k] for k in TB.TABLES['m3_tracker_routes'].ids}
                self.assertEqual(TB.table_rows('m3_tracker_routes', [(meta, {x['arm']: got})])[1], list(row.values()))

    def test_missions_match_runner_and_rows(self):
        for x in (x for x in self.drives if x['kind'] == 'mission'):
            with self.subTest(x['run_dir']):
                e, got = x['expected'], L.drive_labels(self.mapping(x), 'mission')
                self.assertEqual([got['backward_slide'], got['back_s'], got['elapsed_s'], got['waypoints_reached']],
                                 [int(e['any_slide']), e['back_s'], e['total_time_s'], e['goals_reached']])
                row = find('m1_navigation_missions', arm=x['arm'], **x['row'])
                meta = {k: row[k] for k in TB.TABLES['m1_navigation_missions'].ids}
                self.assertEqual(TB.table_rows('m1_navigation_missions', [(meta, {x['arm']: got})])[1], list(row.values()))


class FolderLayouts(unittest.TestCase):
    """load_drive on a synthetic class run (record.json + route.json) and tables.released on a released drive
    (outcome.json + command_reference.npz) holding the same arrays give the same labels; a Record object gives them
    too (a mission Record in memory included)."""

    def test_class_and_released_folders_agree(self):
        rng = np.random.default_rng(0)
        s, a = drive(60)
        s[:, 1:4] = rng.normal(0, 0.3, (60, 3)).astype(F32)
        pose = np.c_[np.linspace(0, 20, 60), rng.normal(0, 0.2, 60), np.zeros(60)]
        wp, des = np.c_[np.linspace(0, 20, 11), np.zeros(11)], np.full(60, 4.0)
        belly = np.full(60, -0.1, F32)
        with tempfile.TemporaryDirectory() as tmp:
            ours, rel = Path(tmp, 'ours'), Path(tmp, 'rel')
            for d in (ours, rel):
                d.mkdir()
                np.savez(d / 'vehicle_extra.npz', belly_clearance_min_m=belly)
            np.savez(ours / 'trajectory.npz', state=s, action=a, pose=pose, desired_speed_mps=des)
            (ours / 'record.json').write_text(json.dumps(dict(status='goal_reached', positive_work_kj=12.5, near_stop_fired=None)))
            (ours / 'route.json').write_text(json.dumps(dict(waypoints=wp.tolist())))
            np.savez(rel / 'trajectory.npz', state=s, action=a, pose=pose)
            np.savez(rel / 'command_reference.npz', desired_speed_mps=des, reference_waypoints=wp)
            (rel / 'outcome.json').write_text(json.dumps(dict(status='goal_reached', positive_work_kj=12.5)))
            for lab in ('rollback', 'rollback_belly', 'goal_belly', 'tracker'):
                self.assertEqual(L.drive_labels(ours, lab), L.drive_labels(TB.released(str(rel), lab), lab), lab)
            self.assertEqual(TB.code(ours, 'rollback_belly'), 's')           # belly flag on a clean drive
            self.assertEqual(L.drive_labels(TB.released(rel, 'tracker'), 'tracker')['near_stop_40s_fired'], 'na')
            from nedm.traversing.evaluation.runner import Record          # in memory: the belly in its extras
            rec = Record('t', 'a', 'goal_reached', True, positive_work_kj=12.5, arrays=dict(state=s, action=a, pose=pose),
                         extras=dict(vehicle_extra=dict(belly_clearance_min_m=belly)))
            for lab in ('rollback_belly', 'goal_belly'):
                self.assertEqual(L.drive_labels(rec, lab), L.drive_labels(TB.released(rel, lab), lab), lab)
            mission = Record('m', 'a', 'mission_complete', True, goals_reached=3, n_goals=3, arrays=dict(state=s, action=a))
            self.assertEqual(L.drive_labels(mission, 'mission')['mission_outcome_sha256'], None)    # in memory: no file
            (ours / 'vehicle_extra.npz').unlink()
            with self.assertRaises(FileNotFoundError):                       # a belly label without the record
                L.drive_labels(ours, 'rollback_belly')
            (ours / 'record.json').write_text(json.dumps(dict(status='crash', near_stop_fired=None)))
            with self.assertRaises(ValueError):
                TB.code(ours, 'rollback')


class Tables(unittest.TestCase):
    def test_tables_on_missing_is_the_library_policy(self):
        self.assertEqual({n: t.on_missing for n, t in TB.TABLES.items() if t.on_missing != '-'}, L.ON_MISSING)

    def test_schemas_equal_released_headers(self):
        self.assertEqual(sorted(TB.TABLES), sorted(p.stem for p in RESULTS.glob('*.csv')))
        for name, t in TB.TABLES.items():
            self.assertEqual(list(t.columns), released(name)[0], name)

    def test_missing_drive_policies(self):
        ok = dict(label='rollback', code='S')
        meta = lambda g: dict(group_id=g, arena='f104', world='rigid', suite_stratum='fresh', terrain_stratum='x')
        arms = list(TB.TABLES['m2_shared_risk_rigid'].arms)
        rows = TB.table_rows('m2_shared_risk_rigid', [(meta('a'), {arms[0]: ok, arms[1]: 'crash'}), (meta('b'), {arms[0]: ok}),
                                                     (meta('c'), {arms[0]: ok, arms[2]: 'launch_failed'})])
        self.assertEqual(rows[1][5:], ['-'] * len(arms))                # drop_pair: the whole pair leaves
        self.assertEqual(rows[2][5:], ['S'] + ['-'] * (len(arms) - 1))  # not run is not a crash
        self.assertEqual(rows[3][5:], ['-'] * len(arms))                # a launch failure is a failed drive too
        meta = dict(task_id='t', arena='g241', arena_kind='near', world='crm_soil', terrain_cluster='c', task_type='x')
        rows = TB.table_rows('m4_polaris_unseen_soil', [(meta, {'polaris_own_model_sampling_grad': 'crash'})])
        self.assertEqual(rows[1][6:], ['U', '-', '-'])                  # ov_unseen: failed twice = not reached safely
        meta = dict(route_id='r', group_id='g', route_kind='designed', speed_profile='constant_2', task_type='x')
        self.assertEqual(TB.table_rows('m4_vehicle_smoke', [(meta, {'m113_regeared_4x': 'crash'})])[1][-1], '-')
        meta = dict(world='crm', route_id='r', stratum='feasible')
        rows = TB.table_rows('m3_tracker_routes', [(meta, {'pid_native': 'crash'})])
        self.assertEqual(rows[1:], [['crm', 'r', 'feasible', 'pid_native', 'crash', '', '', '-'] + [''] * 5])

    def test_refusals(self):
        meta = dict(task_id='t', arena='g241', arena_kind='near', world='crm_soil', terrain_cluster='c', task_type='x')
        with self.assertRaises(ValueError):                             # a Polaris column reads the belly label
            TB.table_rows('m4_polaris_unseen_soil', [(meta, {'polaris_straight_6mps': dict(label='rollback', code='S')})])
        with self.assertRaises(KeyError):
            TB.table_rows('m4_polaris_unseen_soil', [(meta, {'hmmwv': 'crash'})])
        with self.assertRaises(KeyError):
            TB.table_rows('m4_polaris_unseen_soil', [({'task_id': 't'}, {})])
        with self.assertRaises(KeyError):                               # a task twice
            TB.table_rows('m4_polaris_unseen_soil', [(meta, {}), (meta, {})])
        with self.assertRaises(ValueError):                             # a status string that is not a failed drive
            TB.table_rows('m4_polaris_unseen_soil', [(meta, {'polaris_straight_6mps': 'timeout'})])


# ---------------------------------------------------------------------------------------------------- release (A1, A2)
T = 'artifacts/traverse'
K1, K2 = f'{T}/generalist_20260921/A_adapt', f'{T}/crm_improve_20260922'
M2_FILES = {'a3_crm': f'{K1}/a3/results_crm_A0A3.json', 'a3_rigid': f'{K1}/a3/results_rigid_A0A3.json',
            'a5_crm': f'{K1}/a5/results_crm_A5.json', 'a5_rigid': f'{K1}/a5/results_rigid_A5.json',
            's2_crm': f'{K2}/s2/results_s2_crm_vs3s.json', 's4_crm': f'{K2}/s4/results_s4.json',
            's4_rigid': f'{K2}/s4/results_s4_rigid.json'}
# CSV column -> (per-group file, study arm): pr1_staging/m2_build_tables.py (the builder of the released tables)
M2_COLS = {'m2_shared_risk_soil': dict(zip(TB.TABLES['m2_shared_risk_soil'].arms, [
    ('a3_crm', a) for a in ('Scrm', 'Srigid', 'T', 'H')] + [('a5_crm', a) for a in ('Spcrm', 'Sprigid', 'T', 'P', 'Hmask', 'H')] + [
    ('s2_crm', a) for a in ('L1_Spcrm', 'L1_T', 'L1_P', 'L1_H', 'L1_Hn', 'L1_X', 'L0p5_T', 'L0p5_P', 'L0p5_H', 'L0p5_Hn', 'L0p5_X')] + [
    ('s4_crm', 'HnG05'), ('s4_crm', 'XG05')])),
    'm2_shared_risk_rigid': dict(zip(TB.TABLES['m2_shared_risk_rigid'].arms, [
        ('a3_rigid', a) for a in ('Srigid', 'Scrm', 'T', 'H')] + [('a5_rigid', a) for a in ('Sprigid', 'Spcrm', 'T', 'P', 'Hmask', 'H')] + [
        ('s4_rigid', 'Hn05'), ('s4_rigid', 'HnG05')]))}
INDEXES = {'soil_v1': f'{T}/arena_gator_20260925/e6/index/soil_eval_v1.json',
           'rigid_v1': f'{T}/arena_gator_20260925/e6/index/rigid_eval_v1.json',
           'bfull': f'{T}/arena_gator_20260925/e6/index/soil_eval_bfull.json',
           'ovf': f'{T}/offroad_vehicles_20260927/e6/index/soil_eval_ov_final.json',
           'unseen': f'{T}/offroad_vehicles_20260927/e6/index/unseen_polaris_v1.json'}
# CSV column -> (index, study arm): pr1_staging/m4a/build_m4a.py, pr1_staging/m4b/build_m4b_tables.py
_M4A = ('M1a', 'M1b', 'M2', 'M3a', 'M3b', 'A3')
M4_COLS = {
    'm4_unseen_arenas_hmmwv_soil': dict(zip(TB.TABLES['m4_unseen_arenas_hmmwv_soil'].arms,
                                            [('soil_v1', f'{m}_free') for m in _M4A] + [('soil_v1', 'straight6')])),
    'm4_unseen_arenas_hmmwv_rigid': dict(zip(TB.TABLES['m4_unseen_arenas_hmmwv_rigid'].arms, [('rigid_v1', f'{m}_fx2') for m in _M4A] + [
        ('rigid_v1', 'straight2')] + [('rigid_v1', f'{m}_free') for m in _M4A] + [('rigid_v1', 'straight6')])),
    'm4_vehicles_f104_soil': dict(zip(TB.TABLES['m4_vehicles_f104_soil'].arms, [
        ('bfull', 'Gfull_free_gator'), ('ovf', 'Gfull_grad_gator'), ('bfull', 'G_free_gator'), ('bfull', 'Hfull_free_gator'),
        ('bfull', 'straight6_gator'), ('bfull', 'Hfull_free'), ('ovf', 'Hfull_grad_hmmwv'), ('bfull', 'straight6'),
        ('ovf', 'polaris_grad'), ('ovf', 'polaris_cem'), ('ovf', 'polaris_grad_pc'), ('ovf', 'straight6_polaris')])),
    'm4_polaris_unseen_soil': dict(zip(TB.TABLES['m4_polaris_unseen_soil'].arms,
                                       [('unseen', 'polaris_u_grad'), ('unseen', 'polaris_u_cem'), ('unseen', 'straight6_polaris_u')]))}
M4A_SETS = ('unseen', 'indist_f104', 'heldout', 'dev')
G = f'{T}/generalist_20260921'
# released targets of the by-arm views (manifest item tracker_drive_folders)
M3_RUN = {('rigid', 'native_pid'): f'{G}/B_tracker/b0/rigid_runs/{{}}__native_pid', ('rigid', 'held_pid'): f'{G}/B_tracker/b0/rigid_runs/{{}}__held_pid',
          ('rigid', 'policy_v2'): f'{G}/B_tracker/b0/rigid_v2_runs/{{}}__policy_v2', ('crm', 'native_pid'): f'{G}/A_adapt/mixed_out_crm/runs/{{}}__native_pid',
          ('crm', 'held_pid'): f'{G}/A_adapt/mixed_out_crm/runs/{{}}__held_pid', ('crm', 'policy_v2'): f'{G}/B_tracker/b0/crm_v2_runs/{{}}__policy_v2'}
M3_ARM = dict(zip(TB.TABLES['m3_tracker_routes'].arms, ('native_pid', 'held_pid', 'policy_v2')))
M1_ARM = dict(zip(TB.TABLES['m1_navigation_missions'].arms, ('W', 'R2', 'R1', 'R1L')))
NAV = f'{T}/fdm_f104_50h_20260909/nav_v1/local_luffy/runs'
SMOKE = dict(zip(TB.TABLES['m4_vehicle_smoke'].arms, ('gatorctl', 'gatorh', 'hmmwv_stored', 'polaris', 'polaris_pc', 'polaris_4wd',
                                                     'polaris_w08', 'm113', 'm113_g4')))


def _relabel(job):
    try:
        return job, L.drive_labels(TB.released(DATA / job[0], job[1]), job[1])
    except Exception as e:  # noqa: BLE001  (reported per drive by the test)
        return job, f'{type(e).__name__}: {e}'


def _load(files, key):
    """{name: json[key]} of the files present under NEDM_DATA (a partial release restore skips the rest)."""
    return {k: json.loads((DATA / p).read_text())[key] for k, p in files.items() if (DATA / p).exists()}


@unittest.skipUnless(DATA, 'NEDM_DATA is not set')
class Release(unittest.TestCase):
    """Every table cell and index row whose drive folder is present under NEDM_DATA, relabelled once (process pool)."""

    @classmethod
    def setUpClass(cls):
        cls.m2, cls.idx = _load(M2_FILES, 'per_group'), _load(INDEXES, 'rows')
        cls.b0 = _load({w: f'{G}/B_tracker/b0/results_{w}_b0v2.json' for w in ('rigid', 'crm')}, 'per_route')
        jobs = {(r['run_dir'], 'rollback') for g in cls.m2.values() for a in g.values() for r in a.values()}
        for k, rows in cls.idx.items():
            jobs |= {(r['run_dir'], 'rollback_belly' if r.get('belly_available') else 'rollback') for r in rows}
        for name, cols in M4_COLS.items():          # the table's own label per column
            jobs |= {(r['run_dir'], TB.TABLES[name].arms[c]) for c, (k, arm) in cols.items() for r in cls.idx.get(k, ())
                     if r['arm'] == arm}
        jobs |= {(M3_RUN[w, a].format(rid), 'tracker') for w in cls.b0 for a in M3_ARM.values() for rid in cls.b0[w][a]}
        jobs |= {(f'{NAV}/{m}__{a}', 'mission') for m in {r[1] for r in released('m1_navigation_missions')[1:]} for a in M1_ARM.values()}
        present = [j for j in sorted(jobs) if (DATA / j[0] / ('mission_outcome.json' if j[1] == 'mission' else 'trajectory.npz')).exists()]
        with cf.ProcessPoolExecutor(min(16, os.cpu_count() or 1)) as pool:
            cls.lab = dict(pool.map(_relabel, present, chunksize=256))
        cls.absent = len(jobs) - len(present)
        print(f'\n[labels release] relabelled {len(present)} (folder, label) pairs, {cls.absent} absent; records absent: '
              f'{sorted(set(M2_FILES) - set(cls.m2)) + sorted(set(INDEXES) - set(cls.idx)) + sorted({"rigid", "crm"} - set(cls.b0))}',
              file=sys.stderr)

    def get(self, run_dir, label):
        return self.lab.get((run_dir, label))

    def test_a1_drives_equal_study_records(self):
        n, bad = 0, []
        for key, groups in self.m2.items():
            for g, arms in groups.items():
                for arm, r in arms.items():
                    x = self.get(r['run_dir'], 'rollback')
                    if x is None:
                        continue
                    n += 1
                    if not isinstance(x, dict) or [x[k] for k in ('status', 'fail', 'unsafe', 'back_s', 'min_vx', 'max_tilt')] != [
                            r[k] for k in ('status', 'fail', 'unsafe', 'back_s', 'min_vx', 'max_tilt')]:
                        bad.append((key, g, arm, x))
        for key, rows in self.idx.items():
            for r in rows:
                x = self.get(r['run_dir'], 'rollback_belly' if r.get('belly_available') else 'rollback')
                if x is None:
                    continue
                n += 1
                want = [r[k] for k in ('status', 'fail', 'unsafe', 'back_s', 'min_vx', 'max_tilt_deg')]
                got = [x[k] for k in ('status', 'fail', 'unsafe', 'back_s', 'min_vx', 'max_tilt')] if isinstance(x, dict) else x
                if r.get('belly_available'):
                    want += [r['belly_flag'], r['unsafe_belly']]
                    got = got + [x['belly_flag'], int(x['code'] != 'S')] if isinstance(x, dict) else got
                if got != want:
                    bad.append((key, r['run_id'], got, want))
        print(f'[labels release] A1: {n} drive records compared, {len(bad)} mismatches', file=sys.stderr)
        self.assertEqual(bad[:5], [])
        if not n:
            self.skipTest('no released single-goal drive folder with its study record is present')

    def test_a2_tracker_metrics_equal_b0v2(self):
        n, bad = 0, []
        for w, arms in self.b0.items():
            for arm in M3_ARM.values():
                for rid, r in arms[arm].items():
                    x = self.get(M3_RUN[w, arm].format(rid), 'tracker')
                    if x is None:
                        continue
                    n += 1
                    want = [r['status'], r['goal_reached'], r['unsafe'], r['xtrack']['winsor_mean'], r['speed_err_mps']['mean'],
                            r['mean_abs_da']['l1'], r['positive_work_kj']]
                    if not isinstance(x, dict) or [x[k] for k in ('status', 'completed', 'unsafe', 'xtrack_station_winsor_mean_m',
                                                                   'speed_abs_err_mean_mps', 'mean_abs_action_change', 'positive_work_kj')] != want:
                        bad.append((w, arm, rid, x))
        print(f'[labels release] A2: {n} tracker routes compared (float-equal), {len(bad)} mismatches', file=sys.stderr)
        self.assertEqual(bad[:5], [])
        if not n:
            self.skipTest('no released tracker drive folder with its per-route record is present')

    def entries(self, name):
        """(meta, cells) per released row: task columns copied, every arm cell re-derived from the drive folders."""
        t, (head, *rows) = TB.TABLES[name], released(name)
        rows = [dict(zip(head, r)) for r in rows]
        if name.startswith('m2_'):
            src = M2_COLS[name]
            if {f for f, _ in src.values()} - set(self.m2):
                return None
            key = lambda r, c: self.m2[src[c][0]].get(r['group_id'], {}).get(src[c][1], {}).get('run_dir')
        elif name.startswith('m4_') and name != 'm4_vehicle_smoke':
            where = {}
            if {k for k, _ in M4_COLS[name].values()} - set(self.idx):
                return None
            for c, (k, arm) in M4_COLS[name].items():
                for r in self.idx[k]:
                    if r['arm'] == arm and (k != 'soil_v1' and k != 'rigid_v1' or r['vehicle'] == 'hmmwv' and r['set'] in M4A_SETS):
                        where[r['group'], c] = r['run_dir']
            key = lambda r, c: where.get((r[head[0]], c))
        out, seen = [], set()
        for r in rows:
            meta = {k: r[k] for k in t.ids}
            if name == 'm3_tracker_routes':
                ident = (r['world'], r['route_id'])
                cells = {a: self.get(M3_RUN[r['world'], M3_ARM[a]].format(r['route_id']), 'tracker') for a in t.arms}
            elif name == 'm1_navigation_missions':
                ident = r['mission_id']
                cells = {a: self.get(f'{NAV}/{r["mission_id"]}__{M1_ARM[a]}', 'mission') for a in t.arms}
            else:
                ident = id(r)
                cells = {c: self.get(key(r, c), t.arms[c]) for c in t.arms if key(r, c)}
            if ident not in seen:
                seen.add(ident)
                self.assertEqual([v for v in cells.values() if isinstance(v, str)], [], name)   # labelling errors
                if None in cells.values():
                    return None                              # a drive folder is absent: no rebuild
                out.append((meta, cells))
        return out

    def smoke_entries(self):
        runs, src = {}, DATA / f'{T}/offroad_vehicles_20260927/analysis/k4_extract.jsonl'
        if not src.exists():
            return None
        for line in src.read_text().splitlines():
            r = json.loads(line)
            if r.get('sample') == 'A':
                runs.setdefault(r['pair_id'], {})[r['arm']] = r['run']
                if r['arm'] == 'polaris':
                    runs[r['pair_id']]['hmmwv_stored'] = r['hmmwv']
        out, t = [], TB.TABLES['m4_vehicle_smoke']
        for r in (dict(zip(released('m4_vehicle_smoke')[0], x)) for x in released('m4_vehicle_smoke')[1:]):
            cells = {}
            for c, arm in SMOKE.items():                     # no folders released: the extract's status and belly flag
                x = runs[r['route_id']][arm]
                fail = int(x['status'] != 'goal_reached')
                cells[c] = (dict(label='goal_belly', code=L._code(fail, fail or int(bool(x.get('belly_flag')))))
                            if x.get('complete') and x.get('qa_ok') else 'crash')
            out.append(({k: r[k] for k in t.ids}, cells))
        return out

    def test_a1_rebuild_tables_and_recount(self):
        tmp = Path(tempfile.mkdtemp(prefix='labels_tables_'))
        try:
            rebuilt = []
            for name in TB.TABLES:
                e = self.smoke_entries() if name == 'm4_vehicle_smoke' else self.entries(name)
                if e is None:
                    print(f'[labels release] {name}: drive folders or records absent, not rebuilt', file=sys.stderr)
                    continue
                text = TB.table_csv(name, e)
                self.assertEqual(text, (RESULTS / f'{name}.csv').read_text(), name)
                (tmp / f'{name}.csv').write_text(text)
                rebuilt.append(name)
            print(f'[labels release] rebuilt byte-identical: {len(rebuilt)}/9 tables ({", ".join(rebuilt)})', file=sys.stderr)
            if len(rebuilt) == len(TB.TABLES):
                shutil.copy(RESULTS / 'expected_counts.json', tmp)
                spec = importlib.util.spec_from_file_location('recount', REPO / 'scripts/traversing/analysis/recount_milestones.py')
                rc = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(rc)
                self.assertEqual(rc.main(['--check-only', '--results', str(tmp)]), 0)
                n = sum(len(f['expectations']) for f in json.loads((tmp / 'expected_counts.json').read_text())['files'].values())
                print(f'[labels release] recount_milestones --check-only: {n}/{n} expectations', file=sys.stderr)
        finally:
            shutil.rmtree(tmp)


if __name__ == '__main__':
    unittest.main()
