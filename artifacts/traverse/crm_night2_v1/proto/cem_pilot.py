"""Offline pilot: one-shot 256 vs iterated CEM (4x64 and 4x256) on CRM eval cases, scoring only (no simulation)."""
import sys, time, json, glob, os, math
import numpy as np
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
import gen_planner as GP, f104_n2_dataset as DS, f104_n2_sampler as S
DS.init_map('artifacts/traverse/crm_f104_v1/map_root')
model = GP.RiskModel(pattern='artifacts/traverse/crm_f104_v1/train_v1/deploy/CRM_N2_s*.pt')
PRIOR_SD = np.array([5.0, 2.5, 5/3, 1.5, 1.5, 1.5, 1.5])
LAT_CLIP, SP_CLIP = 10.0, 4.0

def from_params(base, th):
    xy = np.asarray(base['waypoints'], float); station = np.asarray(base['stations'], float); speed = np.asarray(base['speeds'], float)
    f = np.clip((station - station[0]) / max(station[-1] - station[0], 1e-6), 0, 1); L = float(station[-1] - station[0])
    cap = 0.55 * 0.125 * L ** 2 / (np.arange(1, 4) * np.pi) ** 2
    a = np.clip(th[:3], -cap, cap)
    lat = np.clip(sum(a[j] * np.sin((j + 1) * np.pi * f) for j in range(3)), -LAT_CLIP, LAT_CLIP)
    dv = S.speed_knots(f, 4, np.clip(th[3:], -SP_CLIP, SP_CLIP))
    r = S.shape(xy, station, speed, lat, dv); r['meta'] = {'candidate': 'cem', 'theta': th.tolist()}
    return r

def route_time(r):
    st = np.asarray(r['stations']); v = np.asarray(r['speeds']); vm = np.maximum(0.5 * (v[1:] + v[:-1]), 0.3)
    return float((np.diff(st) / vm).sum())

def member_logits(X, ctx):
    out = []
    for m, ck in model.members:
        one = GP.RiskModel.__new__(GP.RiskModel); one.torch, one.route_logit, one.dev, one.members = model.torch, model.route_logit, model.dev, [(m, ck)]
        out.append(one.score(X, ctx)[0])
    return np.stack(out)

def score(cands, pose, goal):
    X, L = GP.corridors(cands); Z = member_logits(X, GP.geom_ctx(pose[:2], goal, pose[2], L)); return Z.mean(0), Z.max(0)

def sample_valid(base, pose, rng, mu, sd, n, max_tries):
    out, th, tries = [], [], 0
    while len(out) < n and tries < max_tries:
        tries += 1; t = rng.normal(mu, sd); r = from_params(base, t)
        if GP.safe_validate(r, [], GP.CFG, pose)['valid']: out.append(r); th.append(t)
    return out, np.array(th).reshape(-1, 7), tries

def cem(base, pose, goal, rng, rounds, n, elite_frac=0.15, sd_floor=0.15, weighting='cem'):
    mu, sd = np.zeros(7), PRIOR_SD.copy(); allc, allth, allz, allzp, hist = [], [], [], [], []
    anchors = [r for r in S.anchors(base) if GP.safe_validate(r, [], GP.CFG, pose)['valid']]
    for k in range(rounds):
        need = n - (len(anchors) if k == 0 else 0)
        cs, th, tries = sample_valid(base, pose, rng, mu, sd, need, 8 * need)
        if k == 0: cs = anchors + cs; th = np.concatenate([np.full((len(anchors), 7), np.nan), th])
        if not cs: break
        z, zp = score(cs, pose, goal)
        allc += cs; allth.append(th); allz.append(z); allzp.append(zp)
        Zall = np.concatenate(allz); Th = np.concatenate(allth); ok = np.isfinite(Th).all(1)
        ne = max(4, int(elite_frac * n)); order = np.argsort(np.where(ok, Zall, np.inf))[:ne]
        E = Th[order]; Ez = Zall[order]
        if weighting == 'cem':
            mu = E.mean(0); sd = np.maximum(E.std(0), sd_floor * PRIOR_SD)
        else:  # MPPI softmax with ESS-targeted temperature over all valid samples so far
            J = np.where(ok, Zall, np.inf); Jv = J[ok] - J[ok].min(); lo, hi = 1e-3, 1e3; target = max(2.0, 0.25 * ok.sum())
            for _ in range(40):
                T = math.sqrt(lo * hi); w = np.exp(-Jv / T); w /= w.sum(); ess = 1 / (w ** 2).sum()
                lo, hi = (T, hi) if ess < target else (lo, T)
            w = np.exp(-Jv / math.sqrt(lo * hi)); w /= w.sum(); Tv = Th[ok]
            mu = (w[:, None] * Tv).sum(0); sd = np.maximum(np.sqrt((w[:, None] * (Tv - mu) ** 2).sum(0)), sd_floor * PRIOR_SD)
        hist.append(dict(round=k, n=len(cs), tries=tries, acc=len(cs) / max(tries, 1), zmin=float(Zall.min()), sd=sd.round(2).tolist()))
    Zall = np.concatenate(allz); Zp = np.concatenate(allzp); i = int(np.argmin(Zall))
    return allc[i], float(Zall[i]), float(Zp[i]), len(allc), hist, Zall, Zp

cases = sorted(p for p in glob.glob('artifacts/traverse/crm_f104_v1/cases_eval/cases/*.json') if not p.endswith('/cases.json'))[:24]
rows = []
for cp in cases:
    c = json.load(open(cp)); pose = np.array([c['layout']['start_xy'][0], c['layout']['start_xy'][1], c['layout']['start_yaw']], float); goal = np.asarray(c['goal_xy'], float)
    base = GP.base_route(pose, goal)
    t0 = time.perf_counter(); cands, tries = GP.proposal_pool(base, pose, np.random.default_rng(0)); z1, zp1 = score(cands, pose, goal); i1 = int(np.argmin(z1)); t1 = time.perf_counter()
    r1 = cands[i1]
    res = dict(case=os.path.basename(cp), L=float(base['stations'][-1]), oneshot=dict(zmin=float(z1[i1]), zpess=float(zp1[i1]), z_2nd=float(np.sort(z1)[1]), t=route_time(r1), v=float(np.asarray(r1['speeds'])[1:-1].mean()), n=len(cands), sec=t1 - t0, n_below_1pct=int((1 - np.exp(-np.exp(z1)) < 0.01).sum())))
    for name, (rounds, n, w) in dict(cem4x64=(4, 64, 'cem'), mppi4x64=(4, 64, 'mppi'), cem4x256=(4, 256, 'cem'), cem8x64=(8, 64, 'cem')).items():
        t0 = time.perf_counter(); r, zmin, zp, nev, hist, Zall, Zp = cem(base, pose, goal, np.random.default_rng(0), rounds, n, weighting=w); t1 = time.perf_counter()
        res[name] = dict(zmin=zmin, zpess=zp, t=route_time(r), v=float(np.asarray(r['speeds'])[1:-1].mean()), n_eval=nev, sec=t1 - t0, hist=hist, max_lat=float(r['meta'].get('max_lateral_m', np.abs(np.asarray(r['waypoints'])).max()) if 'max_lateral_m' in r['meta'] else -1))
    rows.append(res)
    print(res['case'], 'L=%.0f' % res['L'], ' '.join('%s z=%.2f zp=%.2f t=%.1f v=%.2f n=%d %.1fs' % (k, res[k]['zmin'], res[k]['zpess'], res[k]['t'], res[k]['v'], res[k].get('n', res[k].get('n_eval')), res[k]['sec']) for k in ('oneshot', 'cem4x64', 'mppi4x64', 'cem4x256', 'cem8x64')), flush=True)
json.dump(rows, open('/tmp/cem_pilot.json', 'w'), indent=1)
A = lambda k, f: np.array([r[k][f] for r in rows])
print('SUMMARY over %d cases (mean; z = ensemble-mean route logit, lower is safer)' % len(rows))
for k in ('oneshot', 'cem4x64', 'mppi4x64', 'cem4x256', 'cem8x64'):
    d = A(k, 'zmin') - A('oneshot', 'zmin')
    print('%-9s zmin %.3f  d_vs_oneshot %.3f (median %.3f, wins %d/%d)  zpess %.3f  gap(pess-mean) %.3f  time %.1f s  v %.2f  evals %.0f  wall %.2f s' % (
        k, A(k, 'zmin').mean(), d.mean(), np.median(d), (d < -1e-6).sum(), len(rows), A(k, 'zpess').mean(), (A(k, 'zpess') - A(k, 'zmin')).mean(), A(k, 't').mean(), A(k, 'v').mean(), A(k, 'n' if k == 'oneshot' else 'n_eval').mean(), A(k, 'sec').mean()))
print('one-shot: gap to 2nd best %.3f; candidates below 1%% risk %.1f/256' % ((A('oneshot', 'z_2nd') - A('oneshot', 'zmin')).mean(), A('oneshot', 'n_below_1pct').mean()))
print('cem4x64 acceptance by round', np.mean([[h['acc'] for h in r['cem4x64']['hist']] for r in rows], 0).round(2), 'sd by round (last case)', [h['sd'] for h in rows[-1]['cem4x64']['hist']])
