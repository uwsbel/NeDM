"""Tests of nedm.traversing.evaluation.planner (stdlib unittest only).

CI (goldens/planner, written by eval_class_staging/goldens/planner_goldens.py with the ORIGINAL 901d6c9 code):
optimize() with the stored float64 stub scorer vs f104_n2_iter.plan_iter / oneshot on 3 decisions x 5 modes (cem,
mppi, fixed2, oneshot, oneshot_fixed2), the scores raw and rounded to 0 and 1 decimals (ties: argmin and the stable
elite sort): every log entry, the pick, n_evaluated (257 / 256), tries, final mu / sd, every candidate's route sha and
z, and the rng state after the search. Also: refusals, the ESS weights, the model adapters (key remap, strict loads,
refused kinds, history length) on synthetic checkpoints, decisions, the pick round trip and the numerics assertion.
NEDM_DATA + the release index cache: Decision.after_approach on 80 released pass-1 runs == the released decision
states (pose, history, mask, base route, frame); A8 locked picks (below).
GPU + NEDM_DATA (the record environment, RTX 5090 + torch 2.12.0+cu130):
  A5  per-member Z of every model kind (ci_train none / hist_aux, ga_train none / hist_aux, legacy CRM_N2 / N2) on
      released decisions == the original CIScorer / ga_planner.Scorer (array_equal, 300-route lists);
  A6  10 groups per released pick folder re-planned with plan() == the released picks (route sha, z_mean, z_pess,
      index, kind, round, theta, n_evaluated, tries, every log entry, final mu / sd, kinds count): the 0.5 s shared
      model (soil, rigid), the 3 s ga_train baseline (soil, rigid), the 0.5 s ga_train model (soil, 753), two unseen
      arenas free (soil) and fixed2 (rigid), the Polaris on f104 and on g260, the Gator on f104 (536).

    NEDM_DATA=/home/harry/NeDM-traverse_mppi NEDM_RELEASE_CACHE=/home/harry/hf_staging/traversing_v1 \\
        PYTHONPATH=src python -m unittest discover -s tests/traversing/evaluation -p test_planner.py -v
"""
import functools
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from nedm.traversing.evaluation import planner as P
from nedm.traversing.evaluation import routes as R
from nedm.traversing.evaluation.config import Env, EvalConfig
from nedm.traversing.evaluation.suites import Task, load_suite, md5hex, select
from nedm.traversing.training.risk_model import RiskModel

try:
    from .common import CACHE, DATA
except ImportError:                     # discover -s tests/traversing/evaluation
    from common import CACHE, DATA

HERE = Path(__file__).resolve().parent
GOLD = HERE / 'goldens' / 'planner'
GPU = torch.cuda.is_available() and (torch.cuda.get_device_name(0), torch.__version__) == P.RECORD_ENV
T = 'artifacts/traverse/'


def jsha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def seed(g, tag):
    return int(hashlib.md5((g + tag).encode()).hexdigest()[:8], 16)


def gold(stem):
    with np.load(GOLD / f'{stem}.npz') as z:
        return json.loads((GOLD / f'{stem}.json').read_text()), {k: z[k] for k in z.files}


def env_with_cache():
    """The Env of the test run and whether tar members (decision states, pick folders) can be release-checked."""
    return Env.from_environ(), CACHE


@functools.lru_cache(maxsize=None)
def suite(name):
    return {t.id: t for t in load_suite(name, env=Env.from_environ())}


def hashed(id, arena, case, **paths):
    """A hand-built Task pinning its files as load_suite would."""
    sha = {k: hashlib.sha256(Path(p).read_bytes()).hexdigest() for k, p in dict(case=case, **paths).items()}
    return Task(id, arena, Path(case), **{k: Path(p) for k, p in paths.items()}, sha=sha)


# ----------------------------------------------------------------------------------------------------------- CI
class TestOptimizeStub(unittest.TestCase):
    """optimize() == f104_n2_iter.plan_iter / oneshot with the same stub scorer, bit for bit."""

    @classmethod
    def setUpClass(cls):
        cls.G, cls.A = gold('optimize_stub')
        ns = {'np': np}
        exec(cls.G['stub_source'], ns)
        cls.stub = staticmethod(ns['stub_score'])

    def run_mode(self, name, mode, q=None):
        base = {**{k: self.A[f'{name}__base_{k}'] for k in R.KEYS}, 'meta': {}}
        pose, m = self.A[f'{name}__pose'], self.G['modes'][mode]
        g = next(d['group'] for d in self.G['decisions'] if d['name'] == name)
        rng = np.random.default_rng(seed(g, m['tag']))
        fs, one = m.get('fixed_speed'), mode.startswith('oneshot')
        res = P.optimize(base, pose, lambda cs: self.stub(cs, q), rng, rounds=1 if one else 4, n=256 if one else 64,
                         update=m.get('weighting', 'cem'), fixed_speed=fs, anchors=not (one and fs),
                         tries_factor=6 if one and fs else 8)
        return res, rng

    def test_every_decision_and_mode(self):
        n = ties = 0
        for d in self.G['decisions']:
            for mode, q in [(m, None) for m in self.G['modes']] + [(m, q) for q in self.G['rounded'] for m in self.G['modes']]:
                key = f'{d["name"]}__{mode}' + ('' if q is None else f'__q{q}')
                with self.subTest(key):
                    res, rng = self.run_mode(d['name'], mode, q)
                    g, i = self.G['runs'][key], res.index
                    self.assertEqual(len(res.cands), 256 if mode.startswith('oneshot') else 257)
                    self.assertEqual(len(res.cands), g['n_evaluated'])
                    self.assertEqual(res.log, g['log'])                       # every log entry, floats exact
                    self.assertEqual((i, res.kinds[i], res.rounds[i], res.tries),
                                     (g['index'], g['kind'], g['round'], g['tries']))
                    self.assertEqual((float(res.z_mean[i]), float(res.z_pess[i])), (g['z_mean'], g['z_pess']))
                    self.assertEqual((res.mu.tolist(), res.sd.tolist()), (g['mu'], g['sd']))
                    self.assertEqual(res.kinds, g['kinds'])
                    self.assertTrue(np.array_equal(res.z_mean, self.A[f'{key}__z_mean']))
                    self.assertTrue(np.array_equal(res.z_pess, self.A[f'{key}__z_pess']))
                    r = res.route
                    self.assertEqual(R.route_sha256(r), g['route_sha256'])
                    self.assertEqual({k: r['meta'][k] for k in g['route_meta']}, g['route_meta'])
                    self.assertEqual(r['meta']['theta'], g['theta'])
                    self.assertEqual(jsha(rng.bit_generator.state), g['rng_state'])
                    ties += int((res.z_mean == res.z_mean.min()).sum()) > 1
                    n += 1
        self.assertEqual(n, 45)
        self.assertGreater(ties, 20)                            # the rounded scores tie at the argmin

    def test_pick_record_fields(self):
        res, _ = self.run_mode('f104_standing', 'cem')
        pk = P.Pick.of_result(res, tag='n2iter_cem4x64')
        g = self.G['runs']['f104_standing__cem']
        self.assertEqual((pk.arm, pk.route_sha256, pk.record['n_evaluated'], pk.record['index']),
                         ('B', g['route_sha256'], 257, g['index']))
        self.assertEqual(sum(pk.record['kinds_count'].values()), 257)
        back = P.Pick.from_dict(json.loads(json.dumps(pk.to_dict())))
        self.assertEqual((back.route_sha256, back.record['log']), (pk.route_sha256, pk.record['log']))


class TestOptimizeRefusals(unittest.TestCase):
    def setUp(self):
        G, A = gold('optimize_stub')
        self.base = {**{k: A[f'f104_standing__base_{k}'] for k in R.KEYS}, 'meta': {}}
        self.pose = A['f104_standing__pose']

    def test_bad_arguments(self):
        rng = np.random.default_rng(0)
        for kw in (dict(rounds=0), dict(n=0), dict(update='softmax'), dict(tries_factor=0)):
            with self.subTest(kw), self.assertRaises(ValueError):
                P.optimize(self.base, self.pose, lambda cs: np.zeros((1, len(cs))), rng, **kw)

    def test_broken_scorer(self):
        for bad in (lambda cs: np.zeros((1, len(cs)), np.float32), lambda cs: np.zeros(len(cs)),
                    lambda cs: np.zeros((2, len(cs) + 1))):
            with self.assertRaises(ValueError):
                P.optimize(self.base, self.pose, bad, np.random.default_rng(0), rounds=2, n=8)

    def test_nothing_valid(self):
        res = P.optimize(self.base, self.pose, lambda cs: np.zeros((1, len(cs))), np.random.default_rng(0), rounds=4,
                         n=8, valid=lambda r, p: False)
        self.assertIsNone(res)

    def test_ess_weights(self):
        J = np.random.default_rng(3).normal(size=64)
        w, ess, T = P.ess_weights(J, 0.25)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
        self.assertTrue(1e-3 <= T <= 1e3 and abs(ess - 16.0) < 0.05)
        self.assertEqual(int(np.argmax(w)), int(np.argmin(J)))


class TestModelAdapters(unittest.TestCase):
    """Synthetic checkpoints of each kind: the key layouts load strictly into main's RiskModel; refused kinds raise."""

    def ck(self, kind, cond='none', **over):
        torch.manual_seed(0)
        m = RiskModel(cond, 6, 5)
        st = {k: v.clone() for k, v in m.state_dict().items()}
        norm = dict(mu=np.zeros(4, np.float32), sd=np.ones(4, np.float32), cont_index=[0, 1, 2, 3])
        ck = dict(state=st, cin=6, nctx=5, norm=norm, ctx_mu=np.zeros(5, np.float32), ctx_sd=np.ones(5, np.float32))
        if kind == 'legacy':
            ck.update(state={k.replace('front.cnn.', 'cnn.'): v for k, v in st.items()}, arch='gru', layers=2)
        elif kind == 'ga_train':
            ck.update(model_kind='ga_train', cond=cond, zdim=16 if cond == 'hist_aux' else 0, hist_T=40,
                      hist_mu=np.zeros(15, np.float32), hist_sd=np.ones(15, np.float32))
        else:
            ck.update(model_kind='ci_train', arch='gru', hist_enc='gru', ctx_mode='geom', cond=cond, zdim=0, hist_T=40,
                      hist_cols=list(range(15)), geom_cols=[17, 18, 19, 20, 21], width=64)
        ck.update(over)
        return ck, m

    def save(self, d, name, ck):
        p = Path(d) / name
        torch.save(ck, p)
        return p

    def test_kinds_load_and_agree(self):
        with tempfile.TemporaryDirectory() as d:
            X = np.random.default_rng(1).normal(size=(3, 5, 96, 32)).astype(np.float32)
            g5 = np.random.default_rng(2).normal(size=(3, 5)).astype(np.float32)
            out = {}
            for kind in ('legacy', 'ga_train', 'ci_train'):
                ck, _ = self.ck(kind, **({} if kind == 'legacy' else dict(domain_filter='crm')))
                ens = P.Ensemble([self.save(d, f'{kind}.pt', ck)], 'cpu')
                self.assertEqual((ens.kind, P.model_info(ens.paths[0])), (kind, (kind, None if kind == 'legacy' else 'crm')))
                out[kind] = ens.logits(X, g5, ens.encode(None, None))
                self.assertEqual((out[kind].shape, out[kind].dtype), ((1, 3), np.float64))
            np.testing.assert_allclose(out['legacy'], out['ci_train'], rtol=0, atol=1e-5)   # same weights, other paths
            self.assertTrue(np.array_equal(out['legacy'], out['ga_train']))

    def test_history_member(self):
        with tempfile.TemporaryDirectory() as d:
            ck, _ = self.ck('ga_train', 'hist_aux')
            ens = P.Ensemble([self.save(d, 'h.pt', ck)], 'cpu')
            z0 = ens.encode(None, None)[0]
            h = np.random.default_rng(0).normal(size=(40, 15)).astype(np.float32)
            z1 = ens.encode(h, np.r_[np.zeros(30, bool), np.ones(10, bool)])[0]
            self.assertEqual(tuple(z0.shape), (1, 16))
            self.assertFalse(torch.equal(z0, z1))

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            ck, _ = self.ck('ga_train', 'hist_aux')
            bad = [('tag.pt', {**self.ck('ga_train')[0], 'cond': 'tag'}),
                   ('arch.pt', self.ck('legacy', arch='tx')[0]),
                   ('tx.pt', self.ck('ci_train', arch='txjoint')[0]),
                   ('stack.pt', {**self.ck('legacy')[0], 'state': {**self.ck('legacy')[0]['state'],
                                                                   'mix.weight_hh_l1': torch.zeros(192, 64)}}),
                   ('nodom.pt', {**ck, 'state': {k: v for k, v in ck['state'].items() if not k.startswith('dom.')}}),
                   ('long.pt', self.ck('ci_train', 'hist_aux', zdim=16, hist_T=60, hist_mu=np.zeros(15, np.float32),
                                       hist_sd=np.ones(15, np.float32))[0]),        # would be zero-padded to 60
                   ('raw.pt', {'a': 1})]
            for name, c in bad:
                with self.subTest(name), self.assertRaises((ValueError, RuntimeError, AssertionError, KeyError)):
                    P.Ensemble([self.save(d, name, c)], 'cpu')
            with self.assertRaises(ValueError):             # mixed kinds
                P.Ensemble([self.save(d, 'a.pt', self.ck('legacy')[0]), self.save(d, 'b.pt', self.ck('ci_train')[0])],
                           'cpu')
            self.assertEqual(P.model_info(self.ck('ci_train', arch='txjoint')[0]), ('txjoint', None))
            ok = self.ck('ci_train', 'hist_aux', zdim=16, hist_T=20, hist_mu=np.zeros(15, np.float32),
                         hist_sd=np.ones(15, np.float32))[0]
            self.assertEqual(P.Ensemble([self.save(d, 'short.pt', ok)], 'cpu').kind, 'ci_train')  # newest 20 rows


class TestDecisionPickNumerics(unittest.TestCase):
    def test_decision_refusals(self):
        base = {'waypoints': np.zeros((3, 2)), 'speeds': np.zeros(3), 'stations': np.arange(3.0),
                'headings': np.zeros(3), 'meta': {}}
        for kw in (dict(pose=np.zeros(2), goal=np.zeros(2)), dict(pose=np.r_[0, 0, np.nan], goal=np.zeros(2)),
                   dict(pose=np.zeros(3), goal=np.zeros(2), hist=np.zeros((40, 15), np.float32)),
                   dict(pose=np.zeros(3), goal=np.zeros(2), hist=np.zeros((40, 15)), hmask=np.zeros(40, bool)),
                   dict(pose=np.zeros(3), goal=np.zeros(2), hist=np.zeros((20, 15), np.float32), hmask=np.zeros(20, bool))):
            with self.subTest(kw.keys()), self.assertRaises(ValueError):
                P.Decision(base=base, **kw)

    def test_after_approach(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            case = Path(d) / 'c.json'
            case.write_text(json.dumps(dict(id='c', goal_xy=[20.0, 0.0], layout=dict(start_xy=[0.0, 0.0], start_yaw=0.0))))
            task = hashed('c', 'f104', case)
            st, ac = rng.normal(size=(10, 17)).astype(np.float32), rng.normal(size=(10, 3)).astype(np.float32)
            po = np.c_[np.linspace(0, 1, 10), np.zeros(10), np.zeros(10)]
            arr = dict(state=st, action=ac, pose=po, terminal_pose=np.array([1.2, 0.0, 0.0]), terminal_state=st[-1] * 2)
            dec = P.Decision.after_approach(task, type('Rec', (), {'arrays': arr})(), 10)
            self.assertEqual(dec.pose.tolist(), [1.2, 0.0, 0.0])
            self.assertEqual((int(dec.hmask.sum()), dec.frame), (10, 10))
            self.assertTrue(np.array_equal(dec.hist[-1], np.r_[(st[-1] * 2)[list(R.OBSERVABLE_COLS)], ac[9]]))
            dec5 = P.Decision.after_approach(task, type('Rec', (), {'arrays': arr})(), 5)
            self.assertEqual((dec5.pose.tolist(), int(dec5.hmask.sum())), (po[5].tolist(), 5))
            with self.assertRaises(ValueError):
                P.Decision.after_approach(task, type('Rec', (), {'arrays': arr})(), 11)
            with self.assertRaises(AttributeError):              # a Record only (its arrays), not a run folder
                P.Decision.after_approach(task, d, 5)
            self.assertEqual(P.Decision.standing(task).source, 'layout')      # no route_00: a valid base_route
            case.write_text(json.dumps(dict(id='c', goal_xy=[-20.0, 0.0], layout=dict(start_xy=[39.0, 0.0],
                                                                                      start_yaw=0.0))))
            with self.assertRaisesRegex(ValueError, 'is not the locked'):     # changed after loading
                P.Decision.standing(task)
            with self.assertRaisesRegex(ValueError, 'no valid base route'):   # facing out at the arena edge
                P.Decision.standing(hashed('c', 'f104', case))

    def test_pick_tamper(self):
        G, A = gold('optimize_stub')
        r = {**{k: A[f'f104_standing__base_{k}'] for k in R.KEYS}, 'meta': {}}
        d = json.loads(json.dumps(P.Pick.of(r, 'given').to_dict()))
        self.assertEqual(P.Pick.from_dict(d).route_sha256, R.route_sha256(r))
        d['route']['speeds'][3] += 1e-9
        with self.assertRaises(ValueError):
            P.Pick.from_dict(d)
        self.assertIsNone(P.Pick.from_dict(P.Pick.of(None, 'B', reason='x').to_dict()).route)

    def test_numerics_asserted_not_set(self):
        rec = P.record_numerics('cpu')
        self.assertEqual((rec['gpu'], rec['env']), ('cpu', 'comparable'))
        old = torch.backends.cudnn.benchmark
        try:
            torch.backends.cudnn.benchmark = True
            with self.assertRaisesRegex(RuntimeError, 'differ from the torch defaults'):
                P.record_numerics('cpu')
        finally:
            torch.backends.cudnn.benchmark = old
        self.assertEqual(P.record_numerics('cpu')['cudnn_benchmark'], False)
        with mock.patch.object(torch.cuda, 'is_available', return_value=False), \
                self.assertRaisesRegex(RuntimeError, 'no CUDA GPU here'):
            P.record_numerics('cuda')


# ------------------------------------------------------------------------------------------- GPU + release data
def f104_groups():
    """The 10 lowest-md5 fresh pairs of the f104 suite (the PR3 risk check's groups)."""
    return [t.id for t in select(suite('f104_800').values(), 'stratum=fresh,lowest_md5=10')]


@unittest.skipUnless(DATA and GPU, 'needs NEDM_DATA and the record GPU (RTX 5090, torch 2.12.0+cu130)')
class TestScorerMembers(unittest.TestCase):
    """A5: per-member Z == the original scorers (CIScorer / ga_planner.Scorer) on released decisions."""

    def test_every_kind(self):
        G, A = gold('scorer_members')
        env, cache = env_with_cache()
        n = 0
        for name, c in G['cases'].items():
            with self.subTest(name):
                if c['poses'] and not cache:
                    self.skipTest('moving decisions need the release index cache (NEDM_RELEASE_CACHE)')
                cdir = DATA / Path(c['models']).parent
                task = suite('f104_800' if c['group'].startswith('f104') else 'unseen_rigid2000')[c['group']]
                dec = (P.Decision.from_release(task, 'data:' + c['poses'], env) if c['poses'] else P.Decision.standing(task))
                self.assertEqual(dec.pose.tolist(), c['pose'])
                cands = [r for r in R.family_anchors(dec.base) if R.validate(r, dec.pose)]
                rng, L = np.random.default_rng(seed(c['group'], 'scorer_golden')), R._base_arrays(dec.base)[2]
                while len(cands) < G['n_score']:
                    r = R.from_params(dec.base, R.draw(rng, L))
                    if R.validate(r, dec.pose):
                        cands.append(r)
                self.assertEqual(jsha([R.route_sha256(r) for r in cands]), c['cand_sha256'])
                ens = P.Ensemble.load(sorted(cdir.glob(Path(c['models']).name)), 'cuda')
                self.assertEqual([ens.kind], c['kinds'])
                sc = P.Scorer(ens, R.StaticMap.load(task.map), dec)
                self.assertEqual(sum(z is not None for z in sc.z), c['n_encode'])
                Z = sc(cands)
                self.assertTrue(np.array_equal(Z, A[f'{name}__Z']), f'{name}: max |dZ| {np.abs(Z - A[name + "__Z"]).max()}')
                n += 1
        scores = sum(c['members'] * c['n'] for c in G['cases'].values())
        print(f'\nA5: {n}/{len(G["cases"])} member-score cases bitwise ({scores} member scores)')


# folder, suite, subset, ground, models, speed, approach_s, decisions
A6 = [('crm_improve_20260922/s2/L0p5/picks_crm_Hn', 'f104_800', None, 'soil',
       'crm_improve_20260922/deploy_v1/deploy_a1_haux_gru_s*.pt', None, 0.5,
       'crm_improve_20260922/a5data/decision_suite_L0p5/poses_crm_all.json'),
      ('crm_improve_20260922/s2/L0p5/picks_rigid_Hn', 'f104_800', None, 'rigid',
       'crm_improve_20260922/deploy_v1/deploy_a1_haux_gru_s*.pt', None, 0.5,
       'crm_improve_20260922/a5data/decision_suite_L0p5/poses_rigid_all.json'),
      ('generalist_20260921/A_adapt/a5/picks_crm_H', 'f104_800', None, 'soil',
       'generalist_20260921/A_adapt/train/deploy_v1/H_deploy_s*.pt', None, 3.0, 'generalist_20260921/A_adapt/a5/poses_crm.json'),
      ('generalist_20260921/A_adapt/a5/picks_rigid_H', 'f104_800', None, 'rigid',
       'generalist_20260921/A_adapt/train/deploy_v1/H_deploy_s*.pt', None, 3.0,
       'generalist_20260921/A_adapt/a5/poses_rigid.json'),
      ('crm_improve_20260922/s2/L0p5/picks_crm_H', 'f104_800', None, 'soil',                      # 753 (M2 decomposition)
       'generalist_20260921/A_adapt/train/deploy_v1/H_deploy_s*.pt', None, 0.5,
       'crm_improve_20260922/a5data/decision_suite_L0p5/poses_crm_all.json'),
      ('arena_gator_20260925/e6/picks/crm/g260/A3_free', 'unseen_soil1000', 'arena=g260', 'soil',
       'arena_gator_20260925/e5/deploy/A3_soil/A3_soil_deploy_s*.pt', None, 0.0, None),
      ('arena_gator_20260925/e6/picks/crm/g241/A3_free', 'unseen_soil1000', 'arena=g241', 'soil',
       'arena_gator_20260925/e5/deploy/A3_soil/A3_soil_deploy_s*.pt', None, 0.0, None),
      ('arena_gator_20260925/e6/picks/rigid/g260/M3a_fixed2', 'unseen_rigid2000', 'arena=g260', 'rigid',
       'arena_gator_20260925/e5/deploy/M3a/M3a_rigid_deploy_s*.pt', 2.0, 0.0, None),
      ('arena_gator_20260925/e6/picks/rigid/g241/M3a_fixed2', 'unseen_rigid2000', 'arena=g241', 'rigid',
       'arena_gator_20260925/e5/deploy/M3a/M3a_rigid_deploy_s*.pt', 2.0, 0.0, None),
      ('offroad_vehicles_20260927/e6/picks/f104/polaris_full_free', 'f104_800', None, 'soil',
       'offroad_vehicles_20260927/e5/deploy/polaris_full_soil/polaris_full_soil_deploy_s*.pt', None, 0.0, None),
      ('offroad_vehicles_20260927/e6/picks/g260/polaris_u_free', 'unseen_soil1000', 'arena=g260', 'soil',
       'offroad_vehicles_20260927/e5/deploy/polaris_full_soil/polaris_full_soil_deploy_s*.pt', None, 0.0, None),
      ('arena_gator_20260925/e6/picks/crm_bfull/f104/G_full_free', 'f104_800', None, 'soil',      # Gator 536
       'arena_gator_20260925/e5/deploy/G_full_soil/G_full_soil_deploy_s*.pt', None, 0.0, None)]
B_FIELDS = ('route_sha256', 'z_mean', 'z_pess', 'index', 'kind', 'round', 'theta', 'n_evaluated', 'tries', 'log',
            'mu_final', 'sd_final', 'kinds_count', 'P', 'T')


def replan(folder, name, subset, ground, models, speed, approach_s, decisions, env, planner='cem'):
    """(task id, new pick, released picks/<g>.json) for 10 groups of a released pick folder: the f104 check groups,
    or the 10 lowest md5 of an arena's folder."""
    by = {t.id: t for t in select(suite(name).values(), subset)}
    have = {p.stem for p in (DATA / T / folder / 'picks').glob('*.json')}
    ids = f104_groups() if name == 'f104_800' else sorted(have, key=md5hex)[:10]
    cfg = EvalConfig(name='check', ground=ground, soil_config='crm_main' if ground == 'soil' else None,
                     approach_s=approach_s, planner=planner, models='data:' + T + models, speed=speed,
                     decisions=None if decisions is None else 'data:' + T + decisions)
    for g in ids:
        t = by[g]
        dec = P.Decision.from_release(t, cfg.decisions, env) if decisions else P.Decision.standing(t)
        yield g, P.plan(cfg, t, dec, env), json.loads((DATA / T / folder / 'picks' / f'{g}.json').read_text())


@unittest.skipUnless(DATA and GPU, 'needs NEDM_DATA and the record GPU (RTX 5090, torch 2.12.0+cu130)')
class TestReplanReleased(unittest.TestCase):
    """A6: sampling-search picks re-planned from the released inputs == the released picks, bit for bit."""

    def test_folders(self):
        env, cache = env_with_cache()
        counts = {}
        for spec in A6:
            if spec[7] and not cache:
                print(f'\nA6: {spec[0]} skipped (moving decisions need NEDM_RELEASE_CACHE)')
                continue
            ok = bad = 0
            for g, pk, rel in replan(*spec, env):
                b, mine = rel['arms']['B'], {'route_sha256': pk.route_sha256, 'z_mean': pk.z_mean, 'z_pess': pk.z_pess,
                                             **pk.record}
                diff = [k for k in B_FIELDS if mine[k] != b[k]]
                with self.subTest(folder=spec[0], group=g):
                    self.assertEqual(diff, [])
                    self.assertEqual(pk.record['numerics']['env'], 'bitwise_env')
                ok, bad = ok + (not diff), bad + bool(diff)
            counts[spec[0]] = (ok, ok + bad)
        print('\nA6 picks bitwise equal (all of ' + ', '.join(B_FIELDS) + '):')
        for k, (ok, n) in counts.items():
            print(f'  {ok}/{n}  {k}')


SUITE_OF = {('m2_shared_risk', 'soil'): 'f104_800', ('m2_shared_risk', 'rigid'): 'f104_800',
            ('m4a_unseen', 'soil'): 'unseen_soil1000', ('m4a_unseen', 'rigid'): 'unseen_rigid2000',
            ('m4b_vehicles', 'soil'): 'f104_800'}


@unittest.skipUnless(DATA and CACHE, 'locked pick folders are tar members: needs NEDM_DATA and NEDM_RELEASE_CACHE')
class TestLockedPicks(unittest.TestCase):
    """A8: the locked pick folder of every headline arm serves its tasks (PICKS_LOCKED recomputes, each route's content
    hash equals its tasks.json row) and each served pick equals picks/<task>.json (route sha, z); a folder of another
    planner or decision timing is refused; straight picks re-planned (rng-free) equal the locked straight routes; a
    tampered route file is refused."""

    @classmethod
    def setUpClass(cls):
        from nedm.traversing.evaluation.config import REPO_ROOT, load_arms
        cls.env = Env.from_environ()
        cls.arms = {(stem, g): load_arms(REPO_ROOT / f'configs/traversing/evaluation/{stem}.toml', ground=g)
                    for stem, g in SUITE_OF}

    def test_headline_arms(self):
        n = arms = 0
        for (stem, ground), cfgs in self.arms.items():
            for cfg in cfgs:
                arms += 1
                name = 'unseen_soil1000' if cfg.name.endswith('_unseen') else SUITE_OF[(stem, cfg.ground)]
                letter = {'cem': 'B', 'cem_grad': 'G', 'straight': 'S'}[cfg.planner]
                for t in select(suite(name).values(), 'lowest_md5_per_arena=2' if 'unseen' in name
                                else 'stratum=fresh,lowest_md5=3'):
                    with self.subTest(arm=cfg.name, ground=cfg.ground, task=t.id):
                        pk = P.plan(cfg, t, None, self.env)
                        rel = json.loads(self.env.file(f'{pk.record["picks"]}/picks/{t.id}.json').read_text())
                        e = rel['arms'][letter]
                        self.assertEqual(pk.arm, {'S': 'straight'}.get(letter, letter))
                        self.assertEqual((pk.route_sha256, pk.z_mean, pk.z_pess),
                                         (e['route_sha256'], e['z_mean'], e.get('z_pess')))
                        n += 1
        print(f'\nA8: {n} locked picks served == picks/<task>.json ({arms} headline arms)')

    def test_other_planner_or_timing_refused(self):
        m2 = {c.name: c for c in self.arms['m2_shared_risk', 'soil']}
        t = suite('f104_800')[f104_groups()[0]]
        for cfg, msg in ((replace(m2['shared_hist_early_rows_0p5s_grad'], planner='cem'), 'has no B pick'),
                         (replace(m2['shared_hist_early_rows_0p5s'], planner='cem_grad'), 'has no G pick'),
                         (replace(m2['shared_hist_3s'], picks=m2['shared_hist_0p5s'].picks), 'was planned at'),
                         (replace(m2['shared_hist_3s'], decisions=None, approach_s=0.0), 'was planned at')):
            with self.subTest(msg=msg), self.assertRaisesRegex(ValueError, msg):
                P.plan(cfg, t, None, self.env)

    def test_straight_replanned(self):
        n = 0
        for name, subset, ground, speed, folder in (
                ('unseen_soil1000', 'arena=g260', 'soil', 6.0, 'arena_gator_20260925/e6/picks/crm/{arena}/straight6'),
                ('unseen_rigid2000', 'arena=g241', 'rigid', 2.0, 'arena_gator_20260925/e6/picks/rigid/{arena}/straight2'),
                ('f104_800', 'stratum=fresh', 'soil', 6.0, 'arena_gator_20260925/e6/picks/crm/{arena}/straight6')):
            cfg = EvalConfig(name='s', ground=ground, soil_config='crm_main' if ground == 'soil' else None,
                             planner='straight', speed=speed)
            for t in select(suite(name).values(), subset + ',lowest_md5=10'):
                mine = P.plan(cfg, t, P.Decision.standing(t), self.env)
                rec = P.Pick.locked(replace(cfg, picks='data:' + T + folder), t, self.env)
                rel = json.loads((DATA / T / self.env.expand(folder, t) / 'picks' / f'{t.id}.json').read_text())
                self.assertEqual((mine.route_sha256, mine.record['index']), (rec.route_sha256, rel['arms']['S']['index']))
                n += 1
        print(f'\nA8: {n}/30 straight routes re-planned == locked')

    def test_tampered_folder(self):
        src = DATA / T / 'arena_gator_20260925/e6/picks/crm/g260/straight6'
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'routes').mkdir()
            for q in (src / 'routes').glob('*.json'):
                (d / 'routes' / q.name).write_bytes(q.read_bytes())
            args = (str(d / 'x'), str(src / 'PICKS_LOCKED.sha256'), str(src / 'tasks.json'), str(d / 'routes'))
            self.assertEqual(len(P._locked_rows(*args)), 125)
            q = sorted((d / 'routes').glob('*.json'))[7]
            q.write_bytes(q.read_bytes().replace(b'0', b'1', 1))
            with self.assertRaises(ValueError):
                P._locked_rows(str(d / 'y'), *args[1:])


# set, F, poses file, pass-1 run folder ({w} world, {g} group)
PASS1 = [('L0p5', 10, 'crm_improve_20260922/a5data/decision_suite_L0p5/poses_{w}_all.json',
          'crm_improve_20260922/a5data/pass1_suite_{w}/runs/{g}__p1_0p5'),
         ('a5', 60, 'generalist_20260921/A_adapt/a5/poses_{w}.json', 'generalist_20260921/A_adapt/a5/pass1_out_{w}/runs/{g}__pass1')]


@unittest.skipUnless(DATA and CACHE, 'decision states are tar members: needs NEDM_DATA and NEDM_RELEASE_CACHE')
class TestAfterApproachReleased(unittest.TestCase):
    def test_equals_released_decision_states(self):
        """Decision.after_approach on a released pass-1 run (its arrays, as a Record carries them) == the released
        decision state of that run (from_release: pose, history window, mask, base route, frame): the 20 lowest-md5
        groups x 2 worlds x 2 decision sets (F = 10 from the terminal state, F = 60)."""
        env, n = Env.from_environ(), 0
        groups = sorted(suite('f104_800'), key=md5hex)[:20]
        for name, F, poses, run in PASS1:
            for w in ('crm', 'rigid'):
                for g in groups:
                    with self.subTest(set=name, world=w, group=g), \
                            np.load(DATA / T / run.format(w=w, g=g) / 'trajectory.npz') as z:
                        t = suite('f104_800')[g]
                        rec = type('Record', (), {'arrays': {k: z[k] for k in z.files}})()
                        mine, rel = P.Decision.after_approach(t, rec, F), P.Decision.from_release(
                            t, 'data:' + T + poses.format(w=w), env)
                        self.assertEqual((mine.pose.tolist(), mine.goal.tolist(), mine.frame),
                                         (rel.pose.tolist(), rel.goal.tolist(), rel.frame))
                        self.assertTrue(np.array_equal(mine.hist, rel.hist) and np.array_equal(mine.hmask, rel.hmask))
                        self.assertEqual(R.route_sha256(mine.base), R.route_sha256(rel.base))
                        n += 1
        self.assertEqual(n, 80)
        print(f'\nafter_approach: {n}/80 released decision states reproduced from their pass-1 runs')


if __name__ == '__main__':
    unittest.main()
