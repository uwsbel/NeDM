"""D2 cross-scoring (diagnosis only, plan S0): every existing risk model scores every distinct route that was driven
from the identical frame-60 state of each of the 800 CRM suite groups (K1 A5 pass 2), then ranking, calibration and
pick-regret against the realised outcome (fail = status != goal_reached), and a conditional (within-group) logistic
test of whether the branch-point speed step explains the misses. Nothing here is used to select anything.

Loading and scoring are ga_planner.py's own code: ga_planner.Ensemble on the checkpoint glob, ga_planner.decision_for
on the A5 poses files (pose + frame-60 history; the masked file for 'H_masked'), ga_planner.Scorer (depth-map corridors
via f104_n2_dataset, float16 rounding, history z once per decision, crm tag for T).

  OMP_NUM_THREADS=6 PYTHONPATH=src:scripts /home/harry/miniconda3/envs/nedm/bin/python \
      artifacts/traverse/crm_improve_20260922/scout/D2_crossscore.py
"""
import hashlib, itertools, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
import ga_planner as G
IT, DS = G.IT, G.DS

K1 = ROOT / 'artifacts/traverse/generalist_20260921/A_adapt'
A5 = K1 / 'a5'
OUT = Path(__file__).resolve().parent
MAP_ROOT = ROOT / 'artifacts/traverse/crm_f104_v1/map_root'
D = K1 / 'train/deploy_v1'
MODELS = [  # name, glob, poses file (history source), description
    ('H', D / 'H_deploy_s*.pt', 'poses_crm.json', 'shared history model (hist_aux), frame-60 history'),
    ('H_masked', D / 'H_deploy_s*.pt', 'poses_crm_masked.json', 'same H checkpoints, history all masked (the A5 Hmask arm)'),
    ('T', D / 'T_deploy_s*.pt', 'poses_crm.json', 'oracle domain tag model, tag = crm'),
    ('P', D / 'P_deploy_s*.pt', 'poses_crm.json', 'pooled model, no label, no history'),
    ('Sp_crm', D / 'Sp_crm_deploy_s*.pt', 'poses_crm.json', 'same-row CRM specialist'),
    ('Sp_rigid', D / 'Sp_rigid_deploy_s*.pt', 'poses_crm.json', 'same-row rigid specialist'),
    ('CRM_N2_legacy', ROOT / 'artifacts/traverse/crm_f104_v1/train_v1/deploy/CRM_N2_s*.pt', 'poses_crm.json',
     'deployed legacy CRM ensemble (crm_f104_v1, standing-start rows only)'),
]
ARMS = ['Spcrm', 'Sprigid', 'H', 'Hmask', 'P', 'T']
ARM_MODEL = {'Spcrm': 'Sp_crm', 'Sprigid': 'Sp_rigid', 'H': 'H', 'Hmask': 'H_masked', 'P': 'P', 'T': 'T'}
NBOOT, NFOLD, L2 = 2000, 5, 1e-3
NBOOT_POOLED = 500
PAD = 64   # score every group in one batch of 64 (routes repeated): cuDNN picks its TF32 conv algorithm by batch shape


# ---------------------------------------------------------------------------------------------------------------- data
def load_routes(limit=0):
    ri = json.load(open(A5 / 'run_index_crm_pass2.json'))
    poses = json.load(open(A5 / 'poses_crm.json'))
    res = json.load(open(A5 / 'results_crm_A5.json'))['per_group']
    groups = sorted(poses)
    assert len(groups) == 800
    if limit:
        groups = groups[:limit] + groups[-limit:]
    rows, checks = [], dict(sha_mismatch=0, dup_sha_mismatch=0, frame60_state_maxdiff=0.0, frame60_pose_maxdiff=0.0,
                            vx60_hist_vs_pass1_maxdiff=0.0, fail_vs_results_mismatch=0)
    for g in groups:
        h = np.load(poses[g]['history'], allow_pickle=True)
        assert int(h['frame']) == 60 and bool(h['hmask'][-1])
        vx60 = float(h['hist'][-1, 0])                       # state col 0 = vel_body_x_mps at frame 60
        p1 = np.load(Path(poses[g]['pass1_run']) / 'trajectory.npz', allow_pickle=True)
        checks['vx60_hist_vs_pass1_maxdiff'] = max(checks['vx60_hist_vs_pass1_maxdiff'], abs(vx60 - float(p1['terminal_state'][0])))
        pose = np.asarray(poses[g]['pose'], float)
        by_drive = {}
        for arm in ARMS:
            rid = f'{g}__{arm}_B'; e = ri[rid]; drv = e['driven_as']
            r = json.load(open(A5 / 'routes_pass2_crm' / f'{rid}.json'))
            sha = IT.route_sha256(r)
            checks['sha_mismatch'] += int(sha != e['sha256'])
            if drv in by_drive:
                checks['dup_sha_mismatch'] += int(sha != by_drive[drv]['sha256']); by_drive[drv]['arms'].append(arm); continue
            o = json.load(open(A5 / 'crm_pass2_runs' / drv / 'outcome.json'))
            tr = np.load(A5 / 'crm_pass2_runs' / drv / 'trajectory.npz', allow_pickle=True)
            checks['frame60_state_maxdiff'] = max(checks['frame60_state_maxdiff'], float(np.abs(tr['state'][60] - p1['terminal_state']).max()))
            checks['frame60_pose_maxdiff'] = max(checks['frame60_pose_maxdiff'], float(np.abs(tr['pose'][60] - pose).max()))
            sp = np.asarray(r['speeds'], float); hd = np.asarray(r['headings'], float)
            dh = (hd[0] - pose[2] + np.pi) % (2 * np.pi) - np.pi
            by_drive[drv] = dict(group=g, driven_as=drv, arms=[arm], sha256=sha, route=r, status=o['status'],
                                 fail=int(o['status'] != 'goal_reached'), elapsed_s=float(o['elapsed_s']), vx60=vx60,
                                 v0=float(sp[0]), step=float(sp[0] - vx60), step_pos=float(max(sp[0] - vx60, 0.0)),
                                 mean_speed=float(sp[1:-1].mean()), length_m=float(np.asarray(r['stations'])[-1]),
                                 head_step_deg=float(np.degrees(abs(dh))),
                                 stratum='reused' if g.startswith('f104_crm_eval_group') else 'fresh')
        for d_ in by_drive.values():
            for arm in d_['arms']:
                checks['fail_vs_results_mismatch'] += int(int(res[g][arm]['fail']) != d_['fail'])
        rows.extend(by_drive.values())
    return rows, res, checks


def score_all(rows):
    """{model: {'mean': (n,), 'members': (n, m)}} with the planner's own Ensemble / decision_for / Scorer."""
    DS.init_map(str(MAP_ROOT))
    cases = K1 / 'suite/cases'
    a = SimpleNamespace(from_run=None, frame=60, pose_along_s=None, pose_override=None, goal=None, history=None)
    by_group = {}
    for i, r in enumerate(rows):
        by_group.setdefault(r['group'], []).append(i)
    out, info = {}, {}
    for name, pat, pfile, desc in MODELS:
        t0 = time.time()
        ens = G.Ensemble(str(pat))
        poses = json.load(open(A5 / pfile))
        Zm = np.full(len(rows), np.nan); Zmem = np.full((len(rows), len(ens.members)), np.nan); nvalid = []
        for g, idx in by_group.items():
            case, lay, pose, goal, base, hist, hmask, src = G.decision_for(g, str(cases / f'{g}.json'), a, poses, ens.hist_T)
            sc = G.Scorer(ens, pose[:2], goal, float(pose[2]), 'crm', hist, hmask, float16=True)
            nvalid.append(sc.history['n_valid'])
            cands = [{k: np.asarray(rows[i]['route'][k], float) for k in ('waypoints', 'speeds', 'stations', 'headings')} for i in idx]
            for c in cands:
                c['meta'] = {}
            n = len(cands)
            Z, zm, _ = sc([cands[j % n] for j in range(PAD)])      # the planner's batch shape (CEM rounds of 64)
            Zm[idx] = zm[:n]; Zmem[idx] = Z[:, :n].T
        enc = int(sum(m['model'].encode_calls for m in ens.members if m['kind'] == 'ga_train'))
        info[name] = dict(glob=str(pat.relative_to(ROOT)), poses=pfile, description=desc, members=ens.describe(),
                          conds=ens.conds, kinds=ens.kinds, needs_tag=ens.needs_tag, needs_hist=ens.needs_hist,
                          history_valid_steps=dict(min=int(min(nvalid)), max=int(max(nvalid))), encode_calls=enc,
                          wall_s=round(time.time() - t0, 1))
        print(f'{name}: {len(ens.members)} members {ens.conds} history valid {min(nvalid)}..{max(nvalid)} encodes {enc} '
              f'{time.time() - t0:.0f}s', flush=True)
        out[name] = dict(mean=Zm, members=Zmem)
        del ens; torch.cuda.empty_cache()
    return out, info


# ------------------------------------------------------------------------------------------------------------- metrics
def rank_avg(s):
    _, inv, cnt = np.unique(np.asarray(s, float), return_inverse=True, return_counts=True)
    first = np.cumsum(cnt) - cnt + 1.0
    return first[inv] + (cnt[inv] - 1) / 2.0


def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    if len(y) == 0 or y.min() == y.max():
        return float('nan')
    r = rank_avg(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def pair_counts(y, s, gidx):
    """Per group (concordant + 0.5 ties, pairs) over (fail, success) pairs of the same group: arrays (G,), (G,)."""
    ok = np.zeros(len(gidx)); tot = np.zeros(len(gidx))
    for k, idx in enumerate(gidx):
        yy, ss = y[idx], s[idx]
        if len(yy) < 2 or yy.min() == yy.max():
            continue
        d = ss[yy == 1][:, None] - ss[yy == 0][None, :]
        ok[k] = (d > 0).sum() + 0.5 * (d == 0).sum(); tot[k] = d.size
    return ok, tot


def boot_weights(G_, seed=0):
    rng = np.random.default_rng(seed)
    return rng.multinomial(G_, np.full(G_, 1.0 / G_), size=NBOOT).astype(float)      # (B, G) group resampling counts


def wg_auc(y, s, gidx, W=None):
    ok, tot = pair_counts(y, s, gidx)
    out = dict(auc=float(ok.sum() / tot.sum()) if tot.sum() else float('nan'), pairs=int(tot.sum()),
               groups=int((tot > 0).sum()),
               auc_group_mean=float(np.mean(ok[tot > 0] / tot[tot > 0])) if (tot > 0).any() else float('nan'))
    if W is not None and tot.sum():
        b = (W @ ok) / np.maximum(W @ tot, 1e-12)
        out['ci95'] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    return out, ok, tot


def calib(z, y, nbin=10):
    p = 1.0 - np.exp(-np.exp(np.asarray(z, float)))
    o = np.argsort(p, kind='stable'); bins = []
    for b in np.array_split(o, nbin):
        bins.append(dict(n=int(len(b)), p_lo=float(p[b].min()), p_hi=float(p[b].max()), mean_p=float(p[b].mean()),
                         fail_rate=float(y[b].mean())))
    return dict(mean_p=float(p.mean()), fail_rate=float(y.mean()), brier=float(np.mean((p - y) ** 2)),
                ece_decile=float(sum(bb['n'] * abs(bb['mean_p'] - bb['fail_rate']) for bb in bins) / len(p)), deciles=bins)


def pick_stats(y, s, gidx, W=None):
    """Argmin of s among each group's routes (ties: mean outcome of the tied routes) vs the oracle (best driven route)."""
    pk, orc, rnd, mixed = [], [], [], []
    for idx in gidx:
        yy, ss = y[idx], s[idx]
        t = ss <= ss.min() + 1e-12
        pk.append(yy[t].mean()); orc.append(yy.min()); rnd.append(yy.mean()); mixed.append(yy.min() != yy.max())
    pk, orc, rnd, mixed = map(np.asarray, (pk, orc, rnd, mixed))
    reg = pk - orc
    out = dict(groups=int(len(gidx)), pick_fail_pct=100 * float(pk.mean()), oracle_fail_pct=100 * float(orc.mean()),
               random_fail_pct=100 * float(rnd.mean()), regret_pts=100 * float(reg.mean()),
               groups_with_regret=int((reg > 1e-9).sum()), mixed_groups=int(mixed.sum()),
               regret_pts_mixed_only=100 * float(reg[mixed].mean()) if mixed.any() else float('nan'),
               share_of_random_regret_removed=float(1 - reg.mean() / (rnd - orc).mean()) if (rnd - orc).mean() > 0 else float('nan'))
    if W is not None:
        b = 100 * (W @ reg) / W.sum(1)
        out['regret_ci95'] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    return out, reg


# ------------------------------------------------------------------------------------------ conditional logistic (CV)
def clogit_items(gidx, y):
    it = []
    for idx in gidx:
        yy = y[idx]; k = int(yy.sum()); n = len(idx)
        if k == 0 or k == n:
            continue
        it.append((idx, np.array(list(itertools.combinations(range(n), k))), np.flatnonzero(yy)))
    return it


def clogit_fit(items, X):
    """Exact conditional logistic likelihood per group (P(this failure set | number of failures)), ridge L2 on
    standardised features for numerical stability only (not tuned)."""
    d = X.shape[1]
    if not items:
        return np.zeros(d)
    M = max(len(s) for _, s, _ in items)
    S = np.zeros((len(items), M, d)); msk = np.zeros((len(items), M), bool); O = np.zeros((len(items), d))
    for gi, (idx, subs, fi) in enumerate(items):
        Xi = X[idx]; S[gi, :len(subs)] = Xi[subs].sum(1); msk[gi, :len(subs)] = True; O[gi] = Xi[fi].sum(0)
    S, msk, O = torch.tensor(S), torch.tensor(msk), torch.tensor(O)
    b = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([b], lr=1.0, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-13, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        eta = (S @ b).masked_fill(~msk, -np.inf)
        loss = -(O @ b - torch.logsumexp(eta, 1)).mean() + L2 * (b ** 2).sum()
        loss.backward(); return loss
    for _ in range(3):
        opt.step(closure)
    return b.detach().numpy()


def fold_of(g):
    return int(hashlib.md5(g.encode()).hexdigest(), 16) % NFOLD


def wauc(y, s, w):
    """AUC with non-negative row weights (ties count one half); used for the group bootstrap of the pooled AUC."""
    _, inv = np.unique(s, return_inverse=True)
    W1 = np.bincount(inv, weights=w * (y == 1)); W0 = np.bincount(inv, weights=w * (y == 0))
    below = np.cumsum(W0) - W0
    den = W1.sum() * W0.sum()
    return float((W1 * (below + 0.5 * W0)).sum() / den) if den > 0 else float('nan')


def logit_fit(X, y):
    """Ordinary (pooled) logistic regression with intercept, L2 1e-3 on the slopes (stability only)."""
    Xt = torch.tensor(X); yt = torch.tensor(y, dtype=torch.float64)
    b = torch.zeros(X.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([b], lr=1.0, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-13, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        eta = Xt @ b[1:] + b[0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(eta, yt) + L2 * (b[1:] ** 2).sum()
        loss.backward(); return loss
    for _ in range(3):
        opt.step(closure)
    return b.detach().numpy()


def cv_pooled(F, y, gidx, gnames):
    """Out-of-fold probability of the pooled logistic on F (group-level 5-fold CV, same folds as the conditional fit)."""
    folds = np.array([fold_of(g) for g in gnames]); oof = np.full(len(y), np.nan); coefs = []
    for f in range(NFOLD):
        rtr = np.concatenate([gidx[k] for k in np.flatnonzero(folds != f)]); rte = np.concatenate([gidx[k] for k in np.flatnonzero(folds == f)])
        mu = F[rtr].mean(0); sd = F[rtr].std(0); sd[sd < 1e-12] = 1.0
        b = logit_fit((F[rtr] - mu) / sd, y[rtr]); coefs.append(b.tolist())
        oof[rte] = 1 / (1 + np.exp(-(((F[rte] - mu) / sd) @ b[1:] + b[0])))
    return oof, np.asarray(coefs)


def cv_score(F, y, gidx, gnames):
    """Out-of-fold linear predictor of the conditional logistic on features F (group-level 5-fold CV)."""
    folds = np.array([fold_of(g) for g in gnames]); oof = np.full(len(y), np.nan); coefs = []
    for f in range(NFOLD):
        tr = [gidx[k] for k in np.flatnonzero(folds != f)]; te = [gidx[k] for k in np.flatnonzero(folds == f)]
        rtr = np.concatenate(tr); mu = F[rtr].mean(0); sd = F[rtr].std(0); sd[sd < 1e-12] = 1.0
        Fs = (F - mu) / sd
        b = clogit_fit(clogit_items(tr, y), Fs)
        coefs.append(b.tolist())
        for idx in te:
            oof[idx] = Fs[idx] @ b
    return oof, np.asarray(coefs)


# ------------------------------------------------------------------------------------------------------------ analysis
def analyse(rows, Z, sel, label):
    """All metrics on the routes in boolean mask sel (groups keep only their selected routes)."""
    y_all = np.array([r['fail'] for r in rows], float)
    gnames = sorted({rows[i]['group'] for i in np.flatnonzero(sel)})
    gpos = {g: k for k, g in enumerate(gnames)}
    gl = [[] for _ in gnames]
    for i in np.flatnonzero(sel):
        gl[gpos[rows[i]['group']]].append(i)
    gidx = [np.asarray(v) for v in gl]
    idx_all = np.concatenate(gidx)
    W = boot_weights(len(gidx))
    step = np.array([r['step'] for r in rows]); step_pos = np.array([r['step_pos'] for r in rows])
    msp = np.array([r['mean_speed'] for r in rows])
    vx = np.array([r['vx60'] for r in rows]); strat = np.array([r['stratum'] for r in rows])
    y = y_all
    res = dict(label=label, routes=int(sel.sum()), groups=len(gidx), fail_rate_pct=100 * float(y[idx_all].mean()),
               mixed_groups=int(sum(1 for g in gidx if y[g].min() != y[g].max())), models={}, speed_step_only={})
    # speed-step alone (no model), as rankers
    for nm, s in (('step', step), ('step_pos', step_pos), ('abs_step', np.abs(step)), ('mean_speed', msp)):
        res['speed_step_only'][nm] = wg_auc(y, s, gidx, W)[0]
    for nm, F in (('step+step_pos', np.c_[step, step_pos]), ('step+step_pos+vx_interactions', np.c_[step, step_pos, vx * step, vx * step_pos])):
        oof, co = cv_score(F, y, gidx, gnames)
        res['speed_step_only'][f'clogit[{nm}]'] = dict(wg_auc(y, oof, gidx, W)[0], coef_mean=co.mean(0).tolist(), coef_sd=co.std(0).tolist())
    for name in Z:
        z = Z[name]['mean']
        m = {}
        base, ok0, tot0 = wg_auc(y, z, gidx, W)
        m['within_group_auc'] = base
        okH, totH = pair_counts(y, Z['H']['mean'], gidx)
        dd = W @ ok0 / np.maximum(W @ tot0, 1e-12) - W @ okH / np.maximum(W @ totH, 1e-12)
        m['within_group_auc_minus_H'] = dict(diff=float(ok0.sum() / tot0.sum() - okH.sum() / totH.sum()),
                                             ci95=[float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))])
        for st_ in ('fresh', 'reused'):
            gs = [g for g in gidx if strat[g[0]] == st_]
            m[f'within_group_auc_{st_}'] = wg_auc(y, z, gs)[0] if gs else None
        m['within_group_auc_members'] = [wg_auc(y, Z[name]['members'][:, j], gidx)[0]['auc'] for j in range(Z[name]['members'].shape[1])]
        m['within_group_auc_ensemble_max'] = wg_auc(y, Z[name]['members'].max(1), gidx)[0]['auc']
        m['pooled_auc'] = auc(y[idx_all], z[idx_all])
        m['calibration'] = calib(z[idx_all], y[idx_all])
        m['pick'], reg = pick_stats(y, z, gidx, W)
        # misses: misranked (fail, success) pairs vs correctly ranked ones, by speed step
        mis_d, cor_d, mis_n, cor_n, tie_n = [], [], 0, 0, 0
        for g in gidx:
            yy, zz, ss = y[g], z[g], step[g]
            fi, si = np.flatnonzero(yy == 1), np.flatnonzero(yy == 0)
            for i in fi:
                for j in si:
                    dz = zz[i] - zz[j]; ds = ss[i] - ss[j]
                    if dz < 0: mis_d.append(ds); mis_n += 1
                    elif dz > 0: cor_d.append(ds); cor_n += 1
                    else: tie_n += 1
        mis_d, cor_d = np.asarray(mis_d), np.asarray(cor_d)
        m['misses'] = dict(misranked_pairs=mis_n, correct_pairs=cor_n, tied_pairs=tie_n,
                           misranked_frac_fail_route_larger_step=float((mis_d > 0).mean()) if mis_n else None,
                           correct_frac_fail_route_larger_step=float((cor_d > 0).mean()) if cor_n else None,
                           misranked_median_step_diff=float(np.median(mis_d)) if mis_n else None,
                           correct_median_step_diff=float(np.median(cor_d)) if cor_n else None)
        # groups where the argmin failed although some driven route succeeded: picked vs best routes
        pst, bst = [], []
        for g in gidx:
            yy, zz = y[g], z[g]
            k = g[np.argmin(zz)]
            if y[k] == 1 and yy.min() == 0:
                pst.append(step[k]); bst.append(step[g[yy == 0]].mean())
        m['regret_groups'] = dict(n=len(pst), picked_step_median=float(np.median(pst)) if pst else None,
                                  succeeding_routes_step_median=float(np.median(bst)) if bst else None,
                                  frac_picked_step_gt_succeeding_mean=float(np.mean(np.asarray(pst) > np.asarray(bst))) if pst else None)
        # conditional logistic on top of the logit (group-level 5-fold CV)
        feats = {'logit': np.c_[z], 'logit+step+step_pos': np.c_[z, step, step_pos],
                 'logit+step+step_pos+vx_interactions': np.c_[z, step, step_pos, vx * step, vx * step_pos, vx * z],
                 'logit+mean_speed': np.c_[z, msp],                                   # control: overall route speed
                 'logit+mean_speed+step+step_pos': np.c_[z, msp, step, step_pos]}
        m['clogit'] = {}
        oof0 = None
        for fn, F in feats.items():
            oof, co = cv_score(F, y, gidx, gnames)
            r_, ok, tot = wg_auc(y, oof, gidx, W)
            r_.update(coef_mean=co.mean(0).tolist(), coef_sd=co.std(0).tolist())
            if fn == 'logit+mean_speed':
                okms = ok
            if fn == 'logit+mean_speed+step+step_pos':
                am = W @ okms / np.maximum(W @ tot, 1e-12); a1 = W @ ok / np.maximum(W @ tot, 1e-12)
                r_['rise_vs_logit+mean_speed'] = float(ok.sum() / tot.sum() - okms.sum() / tot.sum())
                r_['rise_vs_logit+mean_speed_ci95'] = [float(np.percentile(a1 - am, 2.5)), float(np.percentile(a1 - am, 97.5))]
            if fn == 'logit':
                oof0, ok0c = oof, ok
            else:
                a0 = W @ ok0c / np.maximum(W @ tot, 1e-12); a1 = W @ ok / np.maximum(W @ tot, 1e-12)
                r_['rise_vs_logit'] = float(ok.sum() / tot.sum() - ok0c.sum() / tot.sum())
                r_['rise_vs_raw_logit'] = float(ok.sum() / tot.sum() - ok0.sum() / tot0.sum())
                r_['rise_ci95'] = [float(np.percentile(a1 - a0, 2.5)), float(np.percentile(a1 - a0, 97.5))]
            r_['pick'] = pick_stats(y, oof, gidx)[0]
            m['clogit'][fn] = r_
        # pooled (between + within group) logistic: does the step / vx explain the absolute level the logit misses?
        idx_c = idx_all; gid = np.zeros(len(y), int)
        for k, g in enumerate(gidx):
            gid[g] = k
        Wr = W[:NBOOT_POOLED][:, gid[idx_c]]
        pf = {'logit': np.c_[z], 'logit+step+step_pos': np.c_[z, step, step_pos], 'logit+vx': np.c_[z, vx],
              'logit+step+step_pos+vx+vx_interactions': np.c_[z, step, step_pos, vx, vx * step, vx * step_pos]}
        m['pooled_logistic'] = {}
        for fn, F in pf.items():
            oof, co = cv_pooled(F, y, gidx, gnames)
            po = oof[idx_c]; yy = y[idx_c]
            r_ = dict(auc=auc(yy, po), brier=float(np.mean((po - yy) ** 2)),
                      logloss=float(-np.mean(yy * np.log(np.clip(po, 1e-12, 1)) + (1 - yy) * np.log(np.clip(1 - po, 1e-12, 1)))),
                      coef_mean=co.mean(0).tolist())
            bo = np.array([wauc(yy, po, w) for w in Wr])
            if fn == 'logit':
                po0, b0 = po, bo
            else:
                r_['auc_rise'] = r_['auc'] - m['pooled_logistic']['logit']['auc']
                r_['auc_rise_ci95'] = [float(np.percentile(bo - b0, 2.5)), float(np.percentile(bo - b0, 97.5))]
                r_['brier_change'] = r_['brier'] - m['pooled_logistic']['logit']['brier']
            m['pooled_logistic'][fn] = r_
        res['models'][name] = m
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--limit', type=int, default=0, help='smoke test: first + last N groups, output to /tmp')
    a = ap.parse_args()
    torch.set_num_threads(6)
    t0 = time.time()
    rows, res_a5, checks = load_routes(a.limit)
    print(f'{len(rows)} distinct driven routes in {len({r["group"] for r in rows})} groups; checks {checks}', flush=True)
    Z, info = score_all(rows)
    # self-check: each arm's own model re-scores its own pick; must equal the logit recorded at planning time
    self_check = {}
    rid_row = {(r['group'], a): i for i, r in enumerate(rows) for a in r['arms']}
    grows = {}
    for i, r in enumerate(rows):
        grows.setdefault(r['group'], []).append(i)
    for arm, mn in ARM_MODEL.items():
        d = [abs(Z[mn]['mean'][rid_row[(g, arm)]] - float(res_a5[g][arm]['z'])) for g in grows]
        own_argmin = []
        for g, ii in grows.items():
            own_argmin.append(int(ii[int(np.argmin(Z[mn]['mean'][ii]))] == rid_row[(g, arm)]))
        gap, own_fail, amin_fail = [], [], []
        for g, ii in grows.items():
            zz = Z[mn]['mean'][ii]; k = ii[int(np.argmin(zz))]; o = rid_row[(g, arm)]
            if k != o:
                gap.append(float(Z[mn]['mean'][o] - zz.min())); own_fail.append(rows[o]['fail']); amin_fail.append(rows[k]['fail'])
        self_check[arm] = dict(model=mn, max_abs_diff=float(np.max(d)), median_abs_diff=float(np.median(d)),
                               own_pick_is_argmin_among_driven=float(np.mean(own_argmin)),
                               when_not=dict(n=len(gap), logit_gap_median=float(np.median(gap)) if gap else None,
                                             logit_gap_p90=float(np.percentile(gap, 90)) if gap else None,
                                             own_pick_fail_pct=100 * float(np.mean(own_fail)) if gap else None,
                                             lower_scored_route_fail_pct=100 * float(np.mean(amin_fail)) if gap else None))
        print(f'self-check {arm}->{mn}: max |z - recorded| {np.max(d):.2e}  own pick is argmin among driven {np.mean(own_argmin):.3f}', flush=True)
    step = np.array([r['step'] for r in rows])
    soil_arm = np.array([any(a_ != 'Sprigid' for a_ in r['arms']) for r in rows])
    subsets = [('all routes', np.ones(len(rows), bool)), ('speed step < 0.5 m/s', step < 0.5),
               ('|speed step| < 0.5 m/s', np.abs(step) < 0.5),
               ('soil-competent arms only (routes picked by H, Hmask, T, P or Spcrm)', soil_arm)]
    analyses = {}
    for lab, sel in subsets:
        analyses[lab] = analyse(rows, Z, sel, lab)
        print(f'[{lab}] routes {analyses[lab]["routes"]} groups {analyses[lab]["groups"]} mixed {analyses[lab]["mixed_groups"]}', flush=True)
        for name, m in analyses[lab]['models'].items():
            c = m['clogit']
            print(f'  {name:14s} WG {m["within_group_auc"]["auc"]:.3f} pooled {m["pooled_auc"]:.3f} meanP {m["calibration"]["mean_p"]:.3f} '
                  f'fail {m["calibration"]["fail_rate"]:.3f} regret {m["pick"]["regret_pts"]:.2f} | clogit {c["logit"]["auc"]:.3f} '
                  f'+step {c["logit+step+step_pos"]["auc"]:.3f} +vx {c["logit+step+step_pos+vx_interactions"]["auc"]:.3f}', flush=True)
    # descriptive fail rate by speed-step bin over the distinct routes (plan bins)
    y = np.array([r['fail'] for r in rows]); vx = np.array([r['vx60'] for r in rows])
    edges = [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf]; bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (step >= lo) & (step < hi)
        bins.append(dict(lo=lo if np.isfinite(lo) else None, hi=hi if np.isfinite(hi) else None, n=int(k.sum()),
                         fail_pct=100 * float(y[k].mean()) if k.any() else None))
    route_table = [dict({k: v for k, v in r.items() if k != 'route'}, z={n: float(Z[n]['mean'][i]) for n in Z})
                   for i, r in enumerate(rows)]
    out = dict(task='D2 cross-scoring of existing models on driven A5 pass-2 CRM routes (diagnosis only)',
               written=time.strftime('%Y-%m-%d %H:%M'), command='OMP_NUM_THREADS=6 PYTHONPATH=src:scripts '
               '/home/harry/miniconda3/envs/nedm/bin/python artifacts/traverse/crm_improve_20260922/scout/D2_crossscore.py',
               inputs=dict(run_index=str((A5 / 'run_index_crm_pass2.json').relative_to(ROOT)), routes=str((A5 / 'routes_pass2_crm').relative_to(ROOT)),
                           runs=str((A5 / 'crm_pass2_runs').relative_to(ROOT)), poses=str((A5 / 'poses_crm.json').relative_to(ROOT)),
                           cases=str((K1 / 'suite/cases').relative_to(ROOT)), map_root=str(MAP_ROOT.relative_to(ROOT))),
               definitions=dict(fail='outcome.json status != goal_reached', step='route speeds[0] - vx at frame 60 (hist[-1, 0])',
                                step_pos='max(step, 0)', logit='ensemble mean of member route logits (log sum softplus hazard)',
                                P='1 - exp(-exp(logit))',
                                within_group_auc='pairs (failed, succeeded) of distinct routes of the same group, concordant + 0.5 ties, pooled over pairs; CI = 2000 group bootstrap',
                                regret='fail(argmin logit among the group\'s driven routes) - min fail over them, mean over groups, in points',
                                clogit='exact conditional logistic likelihood per group (P(failure set | number of failures)), features standardised on the training folds, L2 1e-3, group-level 5-fold CV (fold = md5(group) % 5); vx is constant within a group, so it enters only through interactions'),
               checks=dict(checks, self_score=self_check), models=info, fail_by_step_bin=bins,
               n_routes=len(rows), n_groups=len(grows), analyses=analyses, routes=route_table, wall_s=round(time.time() - t0, 1))
    dst = Path('/tmp/D2_crossscore_smoke.json') if a.limit else OUT / 'D2_crossscore.json'
    json.dump(out, open(dst, 'w'), indent=1, default=float)
    print(f'wrote {dst} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
