"""Tests of nedm.traversing.evaluation.routes (stdlib unittest only).

CI (goldens/routes, written by eval_class_staging/goldens/gen_routes_goldens.py with the ORIGINAL 901d6c9 code): the
one-shot pools of the four seed tags on 3 cases route by route (spec A5, f104_n2_iter.selftest), CEM draws from N(mu, sd)
with the rng state after them, anchors, straight routes, base_route on the cases, on all 13 fallback branches and at
released moving decisions, M1 base routes and a pool under the widened bound and the reversal check, history windows
(terminal and row paths) and input refusals (import boundaries: test_imports.py).
Release (NEDM_DATA = the release restore base): corridors, float16 corridors and geom5 on the 12 planner maps;
base_route(layout pose) == route_00 on every released case (3,250); every released straight6 / straight2 route file
rebuilt byte for byte (A8); history_window == every released decision-state history npz (800 groups x 2 worlds x
F in 10, 20, 60).

    NEDM_DATA=/home/harry/NeDM-traverse_mppi PYTHONPATH=src python -m unittest discover -s tests/traversing/evaluation \\
        -p test_routes.py -v
"""
import functools
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nedm.traversing.evaluation import routes as R

HERE = Path(__file__).resolve().parent
GOLD = HERE / 'goldens' / 'routes'
DATA = Path(os.environ['NEDM_DATA']) if os.environ.get('NEDM_DATA') else None
T = DATA / 'artifacts' / 'traverse' if DATA else None
MAP_RESTORE = {'f104': 'crm_f104_v1/maps/arena_f104_50h_v1'}           # release restore roots (no map_root links)
HIST_SETS = [('L0p5', 10, 'crm_improve_20260922/a5data/decision_suite_L0p5/hist_{w}_T40',
              'crm_improve_20260922/a5data/pass1_suite_{w}/runs/{g}__p1_0p5'),
             ('L1', 20, 'crm_improve_20260922/a5data/decision_suite_L1/hist_{w}_T40',
              'crm_improve_20260922/a5data/pass1_suite_{w}/runs/{g}__p1_1'),
             ('a5', 60, 'generalist_20260921/A_adapt/a5/hist_{w}', 'generalist_20260921/A_adapt/a5/pass1_out_{w}/runs/{g}__pass1')]


def jsha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def seed(g, tag):
    return int(hashlib.md5((g + tag).encode()).hexdigest()[:8], 16)


def pool(base, pose, rng, fixed_speed=None, n=256, valid=R.validate):
    """The one-shot pool = round 0 of the search at rounds=1: valid anchors first, then prior draws until n are valid
    (8n tries); the fixed2 pool has no anchors and 6n tries (gen_planner.proposal_pool / fixed2_pool)."""
    out = [] if fixed_speed else [r for r in R.family_anchors(base) if valid(r, pose)]
    L, tries = float(base['stations'][-1] - base['stations'][0]), 0
    while len(out) < n and tries < (6 if fixed_speed else 8) * n:
        tries += 1
        r = R.from_params(base, R.draw(rng, L, fixed_speed), fixed_speed)
        if valid(r, pose):
            out.append(r)
    return out, tries


def edge_route(x0):
    """The generator's synthetic north-going line at x = x0."""
    wp = np.stack([np.full(81, float(x0)), np.linspace(-20.0, 20.0, 81)], 1)
    st = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
    return {'waypoints': wp, 'speeds': np.full(81, 3.0), 'stations': st, 'headings': np.full(81, np.pi / 2), 'meta': {}}


def reversal_route():
    """Straight out 10 m and back on the same line (built identically by the generator): collinear at the cusp."""
    wp = np.r_[np.stack([np.linspace(0, 10, 21), np.zeros(21)], 1), np.stack([np.linspace(9.5, 0, 20), np.zeros(20)], 1)]
    st = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
    return {'waypoints': wp, 'speeds': np.full(len(wp), 1.0), 'stations': st, 'headings': np.zeros(len(wp)), 'meta': {}}


def same(a, b):
    return all(np.shape(a[k]) == np.shape(b[k]) and np.array_equal(a[k], b[k]) for k in R.KEYS)


class Goldens(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G = json.loads((GOLD / 'routes_goldens.json').read_text())
        with np.load(GOLD / 'routes_arrays.npz') as z:
            cls.A = {k: z[k] for k in z.files}

    def route(self, prefix):
        return {**{k: self.A[f'{prefix}_{k}'] for k in R.KEYS}, 'meta': {}}

    def case(self, ci):
        return self.route(f'case{ci}_base'), self.A[f'case{ci}_pose'], self.A[f'case{ci}_goal']


class TestFamily(Goldens):
    def test_pool_equals_selftest(self):
        """Pool route shas, counts, tries and the rng state after, for crm_proposal | gen_night2 | crm_fixed2 | gen_fixed2."""
        n = 0
        for ci, c in enumerate(self.G['cases']):
            base, pose, _ = self.case(ci)
            self.assertEqual(R.caps(float(base['stations'][-1] - base['stations'][0])).tolist(), c['caps'])
            for tag, p in c['pools'].items():
                with self.subTest(case=c['id'], tag=tag):
                    self.assertEqual(seed(c['id'], tag), p['seed'])
                    rng = np.random.default_rng(p['seed'])
                    got, tries = pool(base, pose, rng, 2.0 if tag.endswith('fixed2') else None)
                    self.assertEqual((len(got), tries), (p['n'], p['tries']))
                    self.assertEqual(sum(r['meta']['candidate'] == 'n2_anchor' for r in got), p['n_anchor'])
                    self.assertEqual([R.route_sha256(r) for r in got], p['shas'])
                    self.assertEqual(jsha([[r['meta']['max_lateral_m'], r['meta']['mean_speed_mps']] for r in got]), p['meta_sha'])
                    self.assertEqual(jsha(rng.bit_generator.state), p['rng_after'])
                    n += len(got)
        self.assertEqual(n, 12 * 256)

    def test_cem_draws(self):
        """24 draws from N(mu, sd) per case and family (the later search rounds), projected; mu beyond the caps; validity
        from the pose and from meta.fdm_station (an unstripped route_00 is validated from that station)."""
        for ci, c in enumerate(self.G['cases']):
            base, pose, _ = self.case(ci)
            L = float(base['stations'][-1] - base['stations'][0])
            for fam, fs in (('free', None), ('fixed2', 2.0)):
                g, mu, sd = c['cem'][fam], self.A[f'case{ci}_{fam}_mu'], self.A[f'case{ci}_{fam}_sd']
                with self.subTest(case=c['id'], family=fam):
                    rng = np.random.default_rng(seed(c['id'], 'goldens_cem_' + fam))
                    th = [R.draw(rng, L, fs, mu, sd) for _ in range(24)]
                    self.assertTrue(np.array_equal(np.stack(th), self.A[f'case{ci}_{fam}_theta']))
                    rs = [R.from_params(base, t, fs) for t in th]
                    self.assertEqual([R.route_sha256(r) for r in rs], g['shas'])
                    self.assertEqual([bool(R.validate(r, pose)) for r in rs], g['valid'])
                    self.assertEqual([bool(R.validate({**r, 'meta': {'fdm_station': 0.5 * L}}, pose)) for r in rs], g['fdm_valid'])
                    self.assertEqual(jsha([r['meta'] for r in rs]), g['meta_sha'])
                    self.assertEqual(jsha(rng.bit_generator.state), g['rng_after'])
                    self.assertEqual(R.project(mu, L, fs).tolist(), g['project_mu'])
                    rm = R.from_params(base, mu, fs)
                    self.assertEqual((R.route_sha256(rm), rm['meta'], bool(R.validate(rm, pose))),
                                     (g['raw_mu_sha'], g['raw_mu_meta'], g['raw_mu_valid']))

    def test_anchors_equal_original(self):
        for ci, c in enumerate(self.G['cases']):
            base, pose, _ = self.case(ci)
            for fam, sp in (('free', R.ANCHOR_SPEEDS), ('fixed2', (2.0,))):
                anc, g = R.family_anchors(base, sp), c['anchors'][fam]
                self.assertEqual([R.route_sha256(r) for r in anc], g['shas'])
                self.assertEqual([bool(R.validate(r, pose)) for r in anc], g['valid'])
                self.assertEqual(jsha([r['meta'] for r in anc]), g['meta_sha'])

    def test_straight_equals_selfcheck(self):
        """ag_picks.straight_route: the offset-0 anchor and its index in the crm_proposal pool."""
        for ci, c in enumerate(self.G['cases']):
            base, pose, _ = self.case(ci)
            for cruise in (6.0, 2.0):
                r, i = R.straight(base, pose, cruise)
                g = c['straight'][f'{cruise:g}']
                self.assertEqual((R.route_sha256(r), i, R.route_time(r)), (g['sha'], g['index'], g['T']))


class TestBaseRoute(Goldens):
    def check(self, rows, poses, goals, valid=R.validate):
        for p, q, g in zip(poses, goals, rows):
            r = R.base_route(p, q, valid)
            self.assertEqual((R.route_sha256(r), r['meta'], bool(valid(r, p))), (g['sha'], g['meta'], g['valid']), g)

    def test_base_route_vs_route00(self):
        """Informational (spec Q12): the standing-start base route equals the case's route_00 on the golden cases."""
        for ci, c in enumerate(self.G['base_route']['cases']):
            base, pose, goal = self.case(ci)
            r = R.base_route(pose, goal)
            self.assertEqual((R.route_sha256(r), r['meta']), (c['sha'], c['meta']))
            self.assertEqual(same(r, base), c['equals_route00'])
            self.assertTrue(c['equals_route00'])

    def test_every_fallback_branch(self):
        B = self.G['base_route']
        self.assertEqual(len(B['branch_counts']), 13)          # first (valid / invalid), 4 Hermite scales, 7 arcs
        self.check(B['sampled'], self.A['sampled_pose'], self.A['sampled_goal'])

    def test_released_moving_decisions(self):
        self.check(self.G['base_route']['decisions'], self.A['decision_pose'], self.A['decision_goal'])

    def test_m1_bound_and_reversal(self):
        """nav_online.plan_bound + safe_validate_no_reversal, as parameters: base routes, their anchors, one pool."""
        M = self.G['m1']
        for g in M['rows']:
            p, q = np.asarray(g['pose']), np.asarray(g['goal'])
            half = R.widened_bound(p)
            valid = functools.partial(R.validate, bound=half, reversal_deg=45.0)
            r = R.base_route(p, q, valid)
            self.assertEqual((half, R.route_sha256(r), r['meta'], bool(valid(r, p)), R.max_step_turn_deg(r)),
                             (g['half'], g['sha'], g['meta'], g['valid'], g['turn_deg']))
            self.assertEqual([bool(valid(a, p)) for a in R.family_anchors(r)], g['anchors_valid'])
        P = M['pool']
        p, q = np.asarray(P['pose']), np.asarray(P['goal'])
        valid = functools.partial(R.validate, bound=R.widened_bound(p), reversal_deg=45.0)
        base = R.base_route(p, q, valid)
        rng = np.random.default_rng(seed('m1_goldens', 'W'))
        got, tries = pool(base, p, rng, valid=valid)
        self.assertEqual((R.route_sha256(base), len(got), tries), (P['base_sha'], P['n'], P['tries']))
        self.assertEqual([R.route_sha256(r) for r in got], P['shas'])
        self.assertEqual(jsha(rng.bit_generator.state), P['rng_after'])
        rev, V = reversal_route(), M['reversal']
        self.assertEqual((R.max_step_turn_deg(rev), bool(R.validate(rev, np.zeros(3), bound=37.0)),
                          bool(R.validate(rev, np.zeros(3), bound=37.0, reversal_deg=45.0))),
                         (V['turn_deg'], V['frozen_valid'], V['m1_valid']))
        self.assertEqual((V['frozen_valid'], V['m1_valid']), (True, False))


class TestHistory(Goldens):
    def test_windows_equal_original(self):
        """Terminal path (row k = terminal_state, passed or pre-stacked) and row path (k < rows) vs ga_approach."""
        for i, h in enumerate(self.G['history']):
            st, ac, term = (self.A[f'hist{i}_{k}'] for k in ('state', 'action', 'terminal'))
            with self.subTest(**{k: h[k] for k in ('set', 'world', 'id')}):
                for got in (R.history_window(st, ac, h['F'], terminal_state=term),
                            R.history_window(np.vstack([st, term[None]]), ac, h['F'])):
                    self.assertTrue(np.array_equal(got[0], self.A[f'hist{i}_hist']))
                    self.assertTrue(np.array_equal(got[1], self.A[f'hist{i}_hmask']))
                    self.assertEqual(int(got[1].sum()), min(h['F'], R.HIST_T))
                for k in h['cuts']:
                    hh, mm = R.history_window(st, ac, k)
                    self.assertTrue(np.array_equal(hh, self.A[f'hist{i}_k{k}_hist']) and np.array_equal(mm, self.A[f'hist{i}_k{k}_hmask']))
        self.assertEqual(len(self.G['history']), 6)

    def test_refusals(self):
        st, ac = np.zeros((10, 17), np.float32), np.zeros((10, 3), np.float32)
        with self.assertRaisesRegex(ValueError, 'terminal_state'):
            R.history_window(st, ac, 10)                        # the original silently masked row 10
        with self.assertRaisesRegex(ValueError, 'no decision window'):
            R.history_window(st, ac, 11)
        with self.assertRaises(ValueError):
            R.history_window(st[:, :15], ac, 5)
        h, m = R.history_window(np.zeros((0, 17)), np.zeros((0, 3)), 0)
        self.assertEqual((h.shape, h.dtype, m.shape, int(m.sum())), ((40, 15), np.float32, (40,), 0))
        self.assertEqual(R.HIST_DIM, 15)

    def test_load_route(self):
        d = dict(waypoints=[[0, 0], [1, 0], [2, 0]], speeds=[1, 1, 0], stations=[0, 1, 2], headings=[0, 0, 0],
                 meta={'fdm_station': 1.0})
        r = R.load_route(d)
        self.assertEqual((r['meta'], r['speeds'].dtype, r['waypoints'].shape), ({}, np.float64, (3, 2)))
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / 'r.json').write_text(json.dumps(d))
            self.assertEqual(R.route_sha256(R.load_route(Path(t) / 'r.json')), R.route_sha256(r))
        with self.assertRaisesRegex(ValueError, "route lacks \\['headings'\\]"):
            R.load_route({k: v for k, v in d.items() if k != 'headings'})


class TestRefusals(Goldens):
    def test_family_and_validator(self):
        base, pose, goal = self.case(0)
        with self.assertRaisesRegex(ValueError, 'theta shape'):
            R.from_params(base, np.zeros(7), 2.0)
        with self.assertRaisesRegex(ValueError, 'both mu and sd'):
            R.draw(np.random.default_rng(0), 30.0, None, mu=np.zeros(7))
        with self.assertRaisesRegex(ValueError, 'coincides'):
            R.base_route(pose, pose[:2])
        with self.assertRaisesRegex(ValueError, 'pose'):
            R.validate(base, pose[:2])                          # a bad pose is an error, not an invalid route
        with self.assertRaisesRegex(ValueError, 'pose'):
            R.base_route(pose[:2], goal)
        with self.assertRaisesRegex(ValueError, 'anchor speed'):
            R.straight(base, pose, 3.0)
        dup = {k: np.r_[v[:1], v] for k, v in base.items() if k in R.KEYS}
        self.assertFalse(R.validate(dup, pose))                 # degenerate: rejected, not raised
        self.assertFalse(R.validate({**base, 'speeds': base['speeds'][:-1]}, pose))
        edge = R.base_route([39.0, -10.0, np.pi / 2], [39.0, 10.0])    # the swept box leaves the +-40 m arena
        self.assertFalse(R.validate(edge, [39.0, -10.0, np.pi / 2]))
        self.assertEqual(R.straight(edge, [39.0, -10.0, np.pi / 2]), (None, None))

    def test_static_map_checks(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                R.StaticMap.load(d)
            np.savez(Path(d) / 'observation.npz', rgbd=np.zeros((4, 8, 8), np.float32))
            cam = dict(elevation_scale_m=10.0, cam_height_m=110.0, hfov_rad=0.82)
            (Path(d) / 'observation.json').write_text(json.dumps(dict(camera=cam, observation_sha256='0' * 64)))
            with self.assertRaisesRegex(ValueError, 'observation_sha256'):
                R.StaticMap.load(d)
            sha = hashlib.sha256((Path(d) / 'observation.npz').read_bytes()).hexdigest()
            (Path(d) / 'observation.json').write_text(json.dumps(dict(camera=cam, observation_sha256=sha)))
            m = R.StaticMap.load(d)
            self.assertEqual((m.ctr, m.rgbd.shape, m.rgbd.dtype, m.rgbd.flags.writeable), (3.5, (4, 8, 8), np.float32, False))
            with self.assertRaisesRegex(ValueError, 'empty'):
                m.corridors([])



@unittest.skipUnless(DATA, 'NEDM_DATA (the release restore base) is not set')
class TestRelease(Goldens):
    def test_corridors_12_maps(self):
        """Corridors (f32 and float16-rounded), L and geom5 on 26 routes per planner map vs corridors_batch."""
        C = self.G['corridors']
        self.assertEqual(len(C), 12)
        for mi, c in enumerate(C):
            with self.subTest(arena=c['arena']):
                m = R.StaticMap.load(T / MAP_RESTORE.get(c['arena'], f'arena_gator_20260925/maps/arena_{c["arena"]}'))
                self.assertEqual((m.sha256, m.mpp, m.ctr, m.elev_scale), (c['map_sha256'], c['mpp'], c['ctr'], c['elev_scale']))
                base, pose, goal = self.route(f'map{mi}_base'), self.A[f'map{mi}_pose'], self.A[f'map{mi}_goal']
                routes = R.family_anchors(base) + [R.from_params(base, t) for t in self.A[f'map{mi}_theta']] + [edge_route(42.0)]
                X, L = m.corridors(routes)
                self.assertEqual(len(routes), c['n_routes'])
                self.assertEqual([hashlib.sha256(x.tobytes()).hexdigest() for x in X], c['X_route_shas'])
                self.assertEqual((hashlib.sha256(X.tobytes()).hexdigest(), hashlib.sha256(L.tobytes()).hexdigest(),
                                  hashlib.sha256(X.astype(np.float16).tobytes()).hexdigest(), L.tolist()),
                                 (c['X_sha'], c['L_sha'], c['X16_sha'], c['L']))
                self.assertEqual(hashlib.sha256(R.geom5(pose, goal, L).tobytes()).hexdigest(), c['geom5_sha'])
                self.assertEqual((int((X[:, 4] == 0).sum()), int((X[:, 4, 0, R.N_LATERAL // 2] == 0).sum())),
                                 (c['invalid_cells'], c['e0_fallbacks']))
                self.assertTrue(c['offmap_route_gives_nan'])
                with self.assertRaisesRegex(ValueError, 'off the map'):
                    m.corridors([edge_route(47.0)])

    def test_base_route_equals_every_route00(self):
        """spec Q12 / A5: base_route(layout pose, goal) == route_00 bit for bit on every released case."""
        total = 0
        for arena, g in self.G['base_route']['route00_equality'].items():
            n, bad = 0, []
            for cp in sorted((DATA / g['cases_dir']).glob('*.json')):
                if cp.name == 'cases.json':
                    continue
                case = json.loads(cp.read_text()); lay = case['layout']
                pose = np.array([lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']], float)
                r00 = R.load_route(cp.parent / 'routes' / case['id'] / 'route_00.json')
                n += 1
                if not same(R.base_route(pose, np.asarray(case['goal_xy'], float)), r00):
                    bad.append(case['id'])
            self.assertEqual((n, bad), (g['n'], []), arena)
            total += n
        self.assertEqual(total, 3250)             # f104 800, 8 test arenas x 250, g203/g228/g217 x 150

    def test_straight_equals_released_routes(self):
        """A8: every released straight6 / straight2 route file (unseen_arena_route_picks) rebuilt byte for byte from its
        case (ag_picks.straight_dump): route, pool index and meta."""
        built, n = {}, 0
        files = [fp for w in ('crm', 'rigid') for fp in sorted((T / 'arena_gator_20260925/e6/picks' / w).glob('*/straight[26]/routes/*.json'))]
        for fp in files:
            g, cruise = fp.stem.split('__')[0], float(fp.stem[-1])
            if (g, cruise) not in built:
                a, role = g.split('_')[:2]
                cdir = T / ('generalist_20260921/A_adapt/suite/cases' if a == 'f104' else f'arena_gator_20260925/cases/{role}_{a}/cases')
                lay = json.loads((cdir / f'{g}.json').read_text())['layout']
                r, i = R.straight(R.load_route(cdir / 'routes' / g / 'route_00.json'),
                                  [lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']], cruise)
                meta = {'candidate': f'crm_eval_straight{int(cruise)}', 'scene_id': g, 'pool': 'proposal', 'cand_index': i}
                built[g, cruise] = json.dumps({**{k: np.asarray(r[k]).tolist() for k in R.KEYS}, 'meta': meta}).encode()
            self.assertEqual(built[g, cruise], fp.read_bytes(), fp)
            n += 1
        self.assertEqual((n, len(built)), (8000, 5900))      # soil straight6 share the rigid ones

    def test_history_equals_released_npz(self):
        """A5: every released decision-state window (40 x 15 + mask) recomputed from its pass-1 recording."""
        n = 0
        for name, F, hdir, rdir in HIST_SETS:
            for w in ('crm', 'rigid'):
                files = sorted((T / hdir.format(w=w)).glob('*.npz'))
                self.assertEqual(len(files), 800, (name, w))
                for fp in files:
                    with np.load(T / rdir.format(w=w, g=fp.stem) / 'trajectory.npz') as tr, np.load(fp) as z:
                        h, m = R.history_window(tr['state'], tr['action'], F, terminal_state=tr['terminal_state'])
                        self.assertTrue(np.array_equal(h, z['hist']) and np.array_equal(m, z['hmask']), fp)
                    n += 1
        self.assertEqual(n, 4800)


if __name__ == '__main__':
    unittest.main()
