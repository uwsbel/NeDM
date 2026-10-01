"""Tests of nedm.traversing.evaluation.refine (stdlib unittest only).

CI: every start of a search (anchors, prior draws, CEM means, the pick) is reproduced bit for bit by its float64
re-shape; the float32 torch chain follows the numpy route family and corridors (synthetic map); the min-plus speed
passes equal the sequential ones; an end-to-end refinement on the CPU with a synthetic ci_train ensemble is
deterministic and obeys the start, re-score, keep and abstain rules; refusals.
GPU + NEDM_DATA + the release index cache (A7, the record environment): gradient picks re-planned from the released
inputs == the released picks for 10 groups of each folder: the final shared model after the 0.5 s approach (soil,
rigid), the Polaris, Gator-trained and HMMWV-trained ensembles on f104 and the Polaris on g260: G and B route sha,
abstained, gain, J_B, J_B_cem, G z, per-row best_step, J_torch, validity, final route sha and re-scored z, the
best-J trace, and B's search record.

    NEDM_DATA=/home/harry/NeDM-traverse_mppi NEDM_RELEASE_CACHE=/home/harry/hf_staging/traversing_v1 \\
        PYTHONPATH=src python -m unittest discover -s tests/traversing/evaluation -p test_refine.py -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from nedm.traversing.evaluation import planner as P
from nedm.traversing.evaluation import refine as G
from nedm.traversing.evaluation import routes as R
from nedm.traversing.training.risk_model import RiskModel

try:                                    # the package from the repo root, or discover -s tests/traversing/evaluation
    from .test_planner import B_FIELDS, DATA, GPU, env_with_cache, gold, replan, seed
except ImportError:
    from test_planner import B_FIELDS, DATA, GPU, env_with_cache, gold, replan, seed


def stub_search(name='f104_standing', rounds=4, n=64):
    Gd, A = gold('optimize_stub')
    ns = {'np': np}
    exec(Gd['stub_source'], ns)
    base = {**{k: A[f'{name}__base_{k}'] for k in R.KEYS}, 'meta': {}}
    pose, goal = A[f'{name}__pose'], A[f'{name}__goal']
    g = next(d['group'] for d in Gd['decisions'] if d['name'] == name)
    res = P.optimize(base, pose, ns['stub_score'], np.random.default_rng(seed(g, 'n2iter_cem4x64')), rounds=rounds, n=n)
    return base, pose, goal, res


def synthetic_map(n=512, mpp=0.18683):
    """A smooth, fully rendered elevation channel (rolling hills) as a StaticMap."""
    c = (n - 1) / 2.0
    x = (np.arange(n) - c) * mpp
    X, Y = np.meshgrid(x, -x)
    rgbd = np.zeros((4, n, n), np.float32)
    rgbd[3] = (0.15 * np.sin(X / 7.0) * np.cos(Y / 9.0) + 0.05 * np.sin((X + Y) / 4.0)).astype(np.float32)
    return R.StaticMap(Path('/synthetic'), {}, 'synthetic', rgbd, 10.0, mpp, c)


class TestStarts(unittest.TestCase):
    def test_every_candidate_reproduced(self):
        for name in ('f104_standing', 'g260_standing', 'f104_moving_L0p5'):
            base, pose, goal, res = stub_search(name)
            n = 0
            for i, (r, kind) in enumerate([(res.route, res.kinds[res.index])] + list(zip(res.cands, res.kinds))):
                a, dv, anc = G.start_params(r, kind)
                self.assertEqual(R.route_sha256(G.np_route(base, a, dv, anc)), R.route_sha256(r), f'{name} {i} {kind}')
                n += 1
            self.assertEqual(n, 258)

    def test_free_family_only(self):
        with self.assertRaises(ValueError):
            G.start_params({'meta': {'theta': [1.0, 2.0, 3.0]}}, 'sample')


class TestChain(unittest.TestCase):
    """The float32 torch chain against the float64 numpy family and corridors (CPU)."""

    def setUp(self):
        self.base, self.pose, self.goal, self.res = stub_search('f104_standing')
        self.cands = [c for c, k in zip(self.res.cands, self.res.kinds) if k != 'anchor'][:24]

    def test_shape_and_corridor(self):
        xy, f, L = R._base_arrays(self.base)
        a = np.stack([np.asarray(c['meta']['theta'])[:3] for c in self.cands])
        dv = np.stack([np.asarray(c['meta']['theta'])[3:] for c in self.cands])
        t = lambda v: torch.as_tensor(np.asarray(v, np.float64)).to(torch.float32)     # noqa: E731
        pts, v, st = G._shape_rows(t(self.base['waypoints']), t(self.base['stations']),
                                   t(np.tile(self.base['speeds'], (len(a), 1))), t(a), t(dv), t(np.zeros((len(a), len(xy)))))
        np.testing.assert_allclose(pts.numpy(), np.stack([c['waypoints'] for c in self.cands]), atol=2e-4)
        np.testing.assert_allclose(v.numpy(), np.stack([c['speeds'] for c in self.cands]), atol=2e-4)
        smap = synthetic_map()
        X, Lt = G._corridor(t(np.stack([c['waypoints'] for c in self.cands])),
                            t(np.stack([c['speeds'] for c in self.cands])), G.DetMap(smap, 'cpu'))
        Xn, Ln = smap.corridors(self.cands)
        np.testing.assert_allclose(Lt.numpy(), Ln, rtol=1e-5)
        for ch, tol in ((0, 2e-3), (1, 2e-3), (2, 2e-3), (3, 2e-4), (4, 0)):
            np.testing.assert_allclose(X[:, ch].numpy(), Xn[:, ch], atol=tol, err_msg=f'channel {ch}')

    def test_minplus_equals_sequential(self):
        rng = np.random.default_rng(0)
        st = np.cumsum(rng.uniform(0.2, 0.6, (3, 40)), 1)
        u = rng.uniform(0.5, 6.0, (3, 40))
        ref = u.copy()
        for r in range(3):
            for j in range(1, 40):
                ref[r, j] = min(ref[r, j], np.sqrt(ref[r, j - 1] ** 2 + 2 * R.A_ACC * (st[r, j] - st[r, j - 1])))
            for j in range(38, -1, -1):
                ref[r, j] = min(ref[r, j], np.sqrt(ref[r, j + 1] ** 2 + 2 * R.A_DEC * (st[r, j + 1] - st[r, j])))
        v2 = G._minplus(G._minplus(torch.tensor(u ** 2), torch.tensor(st), R.A_ACC, True), torch.tensor(st), R.A_DEC, False)
        np.testing.assert_allclose(np.sqrt(v2.numpy()), ref, rtol=1e-12)


class TestRefineCPU(unittest.TestCase):
    """End to end on the CPU: a synthetic ci_train ensemble on a synthetic map (not a parity check: the rules)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        paths = []
        for s in range(2):
            torch.manual_seed(s)
            m = RiskModel('none', 6, 5)
            ck = dict(model_kind='ci_train', arch='gru', hist_enc='gru', ctx_mode='geom', cond='none', cin=6, nctx=5,
                      zdim=0, hist_T=40, hist_cols=list(range(15)), geom_cols=[17, 18, 19, 20, 21], width=64,
                      state=m.state_dict(), norm=dict(mu=np.zeros(4, np.float32), sd=np.ones(4, np.float32),
                                                      cont_index=[0, 1, 2, 3]),
                      ctx_mu=np.zeros(5, np.float32), ctx_sd=np.full(5, 10.0, np.float32))
            paths.append(Path(cls.tmp.name) / f'ci_s{s}.pt')
            torch.save(ck, paths[-1])
        cls.ens = P.Ensemble(paths, 'cpu')
        cls.base, pose, goal, _ = stub_search('f104_standing')
        cls.dec = P.Decision(pose, goal, cls.base)
        cls.smap = synthetic_map()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_refine(self, **grad):
        score = P.Scorer(self.ens, self.smap, self.dec)
        res = P.optimize(self.base, self.dec.pose, score, np.random.default_rng(7), rounds=2, n=12)
        return res, G.refine_pick(res, score, dict(tag='t'), {**G.GRAD, 'starts': 5, 'steps': 4, **grad})

    def test_rules_and_determinism(self):
        res, pk = self.run_refine()
        _, again = self.run_refine()
        self.assertEqual(json.dumps(pk.to_dict(), sort_keys=True), json.dumps(again.to_dict(), sort_keys=True))
        gr = pk.record['grad']
        rows = gr['rows']
        self.assertEqual(len(rows), 5)
        self.assertEqual((rows[0]['src'], rows[0]['pool_index']), ('B', res.index))
        n0 = res.log[0]['n']
        want = [int(i) for i in np.argsort(res.z_mean[:n0], kind='stable') if int(i) != res.index][:4]
        self.assertEqual([r['pool_index'] for r in rows[1:]], want)
        self.assertEqual(gr['J_B'], rows[0]['z0_rescored_pess'])
        valid = [r for r in rows if r['valid']]
        self.assertEqual(gr['n_valid_finals'], len(valid))
        best = min(valid, key=lambda r: r['z_pess'])
        self.assertEqual(gr['gain'], gr['J_B'] - best['z_pess'])
        self.assertEqual(gr['abstained'], not gr['gain'] >= 0.3)
        self.assertEqual(pk.record['B']['route_sha256'], R.route_sha256(res.route))
        if gr['abstained']:
            self.assertEqual((pk.route_sha256, pk.z_pess), (pk.record['B']['route_sha256'], pk.record['B']['z_pess']))
        for r in rows:
            if r['best_step'] == 0:                      # an unrefined row keeps its start route
                start = res.route if r['src'] == 'B' else res.cands[r['pool_index']]
                self.assertEqual(r['route_sha256'], R.route_sha256(start))

    def test_forced_outcomes(self):
        _, pk = self.run_refine(abstain=1e9)
        self.assertTrue(pk.record['grad']['abstained'])
        self.assertEqual((pk.arm, pk.route_sha256), ('G', pk.record['B']['route_sha256']))
        _, pk = self.run_refine(abstain=-1e9)
        gr = pk.record['grad']
        self.assertFalse(gr['abstained'])
        self.assertEqual(pk.route_sha256, gr['rows'][gr['start_row']]['route_sha256'])

    def test_refusals(self):
        score = P.Scorer(self.ens, self.smap, self.dec)
        res = P.optimize(self.base, self.dec.pose, score, np.random.default_rng(7), rounds=2, n=12)
        with tempfile.TemporaryDirectory() as d:
            torch.manual_seed(0)
            m = RiskModel('none', 6, 5)
            st = {k.replace('front.cnn.', 'cnn.'): v for k, v in m.state_dict().items()}
            p = Path(d) / 'legacy.pt'
            torch.save(dict(state=st, arch='gru', cin=6, nctx=5, norm=dict(mu=np.zeros(4), sd=np.ones(4)),
                            ctx_mu=np.zeros(5), ctx_sd=np.ones(5)), p)
            ens = P.Ensemble([p], 'cpu')
            sc = P.Scorer(ens, self.smap, self.dec)
            with self.assertRaises(ValueError):
                G.refine_pick(res, sc, {})


# folder, suite, subset, ground, models, speed, approach_s, decisions
A7 = [('crm_improve_20260922/s4/L0p5/picks_crm_HnG', 'f104_800', None, 'soil',
       'crm_improve_20260922/deploy_v1/deploy_a1_haux_gru_s*.pt', None, 0.5,
       'crm_improve_20260922/a5data/decision_suite_L0p5/poses_crm_all.json'),
      ('crm_improve_20260922/s4/L0p5/picks_rigid_HnG', 'f104_800', None, 'rigid',
       'crm_improve_20260922/deploy_v1/deploy_a1_haux_gru_s*.pt', None, 0.5,
       'crm_improve_20260922/a5data/decision_suite_L0p5/poses_rigid_all.json'),
      ('offroad_vehicles_20260927/e6/picks/f104/polaris_full_grad', 'f104_800', None, 'soil',
       'offroad_vehicles_20260927/e5/deploy/polaris_full_soil/polaris_full_soil_deploy_s*.pt', None, 0.0, None),
      ('offroad_vehicles_20260927/e6/picks/f104/G_full_grad', 'f104_800', None, 'soil',
       'arena_gator_20260925/e5/deploy/G_full_soil/G_full_soil_deploy_s*.pt', None, 0.0, None),
      ('offroad_vehicles_20260927/e6/picks/f104/H_full_grad', 'f104_800', None, 'soil',
       'arena_gator_20260925/e5/deploy/H_full_soil/H_full_soil_deploy_s*.pt', None, 0.0, None),
      ('offroad_vehicles_20260927/e6/picks/g260/polaris_u_grad', 'unseen_soil1000', 'arena=g260', 'soil',
       'offroad_vehicles_20260927/e5/deploy/polaris_full_soil/polaris_full_soil_deploy_s*.pt', None, 0.0, None)]
G_FIELDS = ('route_sha256', 'z_mean', 'z_pess', 'abstained', 'gain', 'J_B', 'J_B_cem', 'start_row', 'n_valid_finals',
            'steps_run')
ROW_FIELDS = ('row', 'src', 'pool_index', 'kind', 'best_step', 'J_torch', 'valid', 'route_sha256', 'z0_rescored_mean',
              'z0_rescored_pess', 'z_mean', 'z_pess')


@unittest.skipUnless(DATA and GPU, 'needs NEDM_DATA and the record GPU (RTX 5090, torch 2.12.0+cu130)')
class TestGradientReleased(unittest.TestCase):
    """A7: gradient picks re-planned from the released inputs == the released picks, bit for bit."""

    def test_folders(self):
        env, cache = env_with_cache()
        counts, rows_checked = {}, 0
        for spec in A7:
            if spec[7] and not cache:
                print(f'\nA7: {spec[0]} skipped (moving decisions need NEDM_RELEASE_CACHE)')
                continue
            ok = n = 0
            for g, pk, rel in replan(*spec, env, planner='cem_grad'):
                eG, eB, grad = rel['arms']['G'], rel['arms']['B'], rel['grad']
                gr, bB = pk.record['grad'], pk.record['B']
                mineG = {'route_sha256': pk.route_sha256, 'z_mean': pk.z_mean, 'z_pess': pk.z_pess, **gr}
                diff = [f'G.{k}' for k in G_FIELDS if mineG[k] != eG.get(k, grad.get(k))]
                diff += [f'B.{k}' for k in B_FIELDS if bB[k] != eB[k]]
                diff += ['trace'] if gr['trace_best_J_min'] != grad['trace_best_J_min'] else []
                diff += ['rows'] if len(gr['rows']) != len(grad['rows']) else [
                    f'row{r["row"]}.{k}' for r, q in zip(gr['rows'], grad['rows']) for k in ROW_FIELDS if r.get(k) != q.get(k)]
                rows_checked += len(gr['rows'])
                with self.subTest(folder=spec[0], group=g):
                    self.assertEqual(diff, [])
                ok, n = ok + (not diff), n + 1
            counts[spec[0]] = (ok, n)
        print(f'\nA7 gradient picks bitwise equal ({rows_checked} start rows; G: {", ".join(G_FIELDS)}; B search record; '
              f'rows: {", ".join(ROW_FIELDS)}; best-J trace):')
        for k, (ok, n) in counts.items():
            print(f'  {ok}/{n}  {k}')


if __name__ == '__main__':
    unittest.main()
