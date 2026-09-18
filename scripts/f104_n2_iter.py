"""Iterated resampling (CEM / MPPI-weighted) over the night-2 route parameterisation, plus the exact one-shot planner.

theta = (a_1, a_2, a_3, dv_1..dv_4) in R^7 (R^3 with fixed_speed): lateral offset sum_j a_j sin(j pi f) (a_j projected
to the validator-derived caps cap_j = 0.55 * 0.125 * L^2 / (j pi)^2, profile clipped +-10 m) and 4 free-end speed knots
dv_k (clipped +-4 m/s) on top of the base speeds, then f104_n2_sampler.shape (v in [0.5, 6], terminal and accel cones).
`from_params` is bit-identical to `f104_n2_sampler.sample_one` when theta is what sample_one drew (checked by
`selftest`), so `plan_iter(rounds=1, n=256)` with the deployed rng tag reproduces the deployed 256-candidate pool and
pick (crm_pools.py / gen_pools.py) exactly, including the float16 rounding of the corridor tensor before scoring.

Spec: artifacts/traverse/crm_night2_v1/scout/planner_sampling.md section 5; pilot: crm_night2_v1/proto/cem_pilot.py.
"""
import hashlib, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_planner as GP
import f104_n2_dataset as DS
import f104_n2_sampler as S

LAT_SIGMA, LAT_CLIP, SP_SIGMA, SP_CLIP = 5.0, 10.0, 1.5, 4.0
MODES, KNOTS, KAPPA_MAX, BUDGET = 3, 4, 0.125, 0.55
PRIOR_SD = np.array([LAT_SIGMA / j for j in range(1, MODES + 1)] + [SP_SIGMA] * KNOTS)   # (5, 2.5, 5/3, 1.5 x4)


def seed(group, tag):
    return int(hashlib.md5((group + tag).encode()).hexdigest()[:8], 16)


def _base_arrays(base):
    xy = np.asarray(base['waypoints'], float); station = np.asarray(base['stations'], float)
    f = np.clip((station - station[0]) / max(station[-1] - station[0], 1e-6), 0, 1)
    L = float(station[-1] - station[0])
    return xy, station, f, L


def caps(L, modes=MODES, kappa_max=KAPPA_MAX, budget=BUDGET):
    """Per-mode amplitude caps, the same expression as f104_n2_sampler.lateral_profile."""
    return budget * kappa_max * L ** 2 / (np.arange(1, modes + 1) * np.pi) ** 2


def project(theta, L, fixed_speed=None):
    """Projection of a raw theta onto the caps (a_j) and the +-4 m/s clip (dv_k); returns the clipped theta."""
    th = np.asarray(theta, float).copy()
    th[:MODES] = np.clip(th[:MODES], -caps(L), caps(L))
    if fixed_speed is None:
        th[MODES:] = np.clip(th[MODES:], -SP_CLIP, SP_CLIP)
    return th


def from_params(base, theta, fixed_speed=None, lat_clip=LAT_CLIP, sp_clip=SP_CLIP, kappa_max=KAPPA_MAX, budget=BUDGET):
    """Deterministic theta -> route. theta is (a_1..a_3, dv_1..dv_4), or (a_1..a_3) with fixed_speed (the fixed-2 pool
    family: sp_sigma 0, base_speed = fixed_speed). The clipped theta is stored in meta['theta']."""
    xy, station, f, L = _base_arrays(base)
    theta = np.asarray(theta, float)
    cap = budget * kappa_max * L ** 2 / (np.arange(1, MODES + 1) * np.pi) ** 2
    a = np.clip(theta[:MODES], -cap, cap)
    lat = sum(a[j] * np.sin((j + 1) * np.pi * f) for j in range(MODES))
    lat = np.clip(lat, -lat_clip, lat_clip)
    if fixed_speed is None:
        speed = np.asarray(base['speeds'], float)
        dvk = np.clip(theta[MODES:MODES + KNOTS], -sp_clip, sp_clip)
        dv = S.speed_knots(f, KNOTS, dvk)
        th = np.r_[a, dvk]
    else:
        speed = np.full(len(xy), float(fixed_speed))
        dv = S.speed_knots(f, KNOTS, np.zeros(KNOTS))      # == zeros, kept for symmetry with sample_one(sp_sigma=0)
        th = a
    r = S.shape(xy, station, speed, lat, dv)
    r['meta'] = {'candidate': 'n2_iter', 'theta': th.tolist(), 'max_lateral_m': float(np.abs(lat).max()),
                 'mean_speed_mps': float(r['speeds'][1:-1].mean())}
    return r


def draw_prior(rng, L, fixed_speed=None, mu=None, sd=None):
    """One theta from the sampling Gaussian; with mu/sd None this consumes the rng stream exactly as sample_one does
    (3 lateral normals scaled by 1/j, then 4 speed normals - drawn even at sp_sigma 0), so the deployed pools are
    reproduced draw by draw."""
    if mu is None:
        a = rng.normal(0, LAT_SIGMA, MODES) / np.arange(1, MODES + 1)
        dv = rng.normal(0, 0.0 if fixed_speed is not None else SP_SIGMA, KNOTS)
        th = a if fixed_speed is not None else np.r_[a, dv]
    else:
        th = rng.normal(mu, sd)
    return project(th, L, fixed_speed)


def route_time(r):
    """Commanded route time: sum ds / v_mid (v_mid floored at 0.3 m/s)."""
    st = np.asarray(r['stations'], float); v = np.asarray(r['speeds'], float)
    vm = np.maximum(0.5 * (v[1:] + v[:-1]), 0.3)
    return float((np.diff(st) / vm).sum())


def route_sha256(r):
    """Content hash of the route as it will be written (waypoints, speeds, stations, headings)."""
    import json
    s = json.dumps({k: np.asarray(r[k], float).tolist() for k in ('waypoints', 'speeds', 'stations', 'headings')})
    return hashlib.sha256(s.encode()).hexdigest()


def valid(r, pose):
    return GP.safe_validate(r, [], GP.CFG, np.asarray(pose, float))['valid']


# ------------------------------------------------------------------------------------------------------------------
# corridors in one batch (bit-identical to gen_planner.corridors, checked at import by corridors_check)
# ------------------------------------------------------------------------------------------------------------------
def corridors_batch(cands):
    ns, nl = DS.N_STATION, DS.N_LATERAL
    n = len(cands)
    GX = np.empty((n, ns, nl)); GY = np.empty((n, ns, nl)); V = np.empty((n, ns)); LEN = np.empty(n); DSS = np.empty(n)
    off = np.linspace(-DS.HALF_WIDTH_M, DS.HALF_WIDTH_M, nl)
    for i, r in enumerate(cands):
        wp = np.asarray(r['waypoints']); sp = np.asarray(r['speeds']); st = np.asarray(r['stations'])
        pts, grid = DS.resample_route(wp, st, ns)
        d = np.gradient(pts, axis=0); tn = np.linalg.norm(d, axis=1, keepdims=True); tn[tn < 1e-9] = 1e-9
        tang = d / tn; norm = np.stack([-tang[:, 1], tang[:, 0]], 1)
        GX[i] = pts[:, 0:1] + norm[:, 0:1] * off[None, :]; GY[i] = pts[:, 1:2] + norm[:, 1:2] * off[None, :]
        s_ref = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
        V[i] = np.interp(np.linspace(0, s_ref[-1], ns), s_ref, sp)
        LEN[i] = float(grid[-1] - grid[0]); DSS[i] = max(LEN[i] / (ns - 1), 1e-3)
    patch, vld = DS.sample_map(GX, GY)
    elev = np.where(vld, patch[3] * DS.G['elev_scale'], np.nan)
    e0 = elev[:, 0, nl // 2].copy()
    bad = ~np.isfinite(e0)
    if bad.any():
        e0[bad] = np.array([np.nanmean(elev[i, 0]) for i in np.flatnonzero(bad)])
    fill = np.where(np.isfinite(elev), elev, e0[:, None, None])
    dl = 2 * DS.HALF_WIDTH_M / (nl - 1)
    ga = np.clip(np.gradient(fill, axis=1) / DSS[:, None, None], -2, 2)   # (x/2)/ds == x/(2 ds) bitwise
    gc = np.clip(np.gradient(fill, dl, axis=2), -2, 2)
    X = np.stack([fill - e0[:, None, None], ga, gc, np.repeat(V[:, :, None], nl, 2), vld.astype(float)], 1)
    return X.astype(np.float32), LEN.astype(np.float32)


_CORR = {'fn': None}


def corridors(cands):
    """Batched corridors; on the first call the batched output is checked bit-for-bit against gen_planner.corridors
    and the per-candidate loop is used instead if they ever differ."""
    if _CORR['fn'] is None:
        Xb, Lb = corridors_batch(cands); Xl, Ll = GP.corridors(cands)
        ok = Xb.shape == Xl.shape and np.array_equal(Xb, Xl) and np.array_equal(Lb, Ll)
        _CORR['fn'] = corridors_batch if ok else GP.corridors; _CORR['identical'] = bool(ok)
        return (Xb, Lb) if ok else (Xl, Ll)
    return _CORR['fn'](cands)


# ------------------------------------------------------------------------------------------------------------------
# scoring: per-member route logits, ensemble mean == RiskModel.score's z
# ------------------------------------------------------------------------------------------------------------------
def member_logits(model, X, ctx5, bs=256):
    """(members, n) route logits, the same arithmetic as gen_planner.RiskModel.score member by member."""
    torch = model.torch; zs = []
    for m, ck in model.members:
        nm = ck['norm']; out = []
        for i in range(0, len(X), bs):
            x = X[i:i + bs].astype(np.float32).copy()
            x[:, :4] = (x[:, :4] - nm['mu'][None, :, None, None]) / nm['sd'][None, :, None, None]
            x = np.concatenate([x, np.ones((len(x), 1, x.shape[2], x.shape[3]), np.float32)], 1)
            c = (ctx5[i:i + bs] - ck['ctx_mu']) / ck['ctx_sd']
            with torch.no_grad():
                out.append(model.route_logit(m(torch.tensor(x, device=model.dev),
                                               torch.tensor(c, dtype=torch.float32, device=model.dev))).double().cpu().numpy())
        zs.append(np.concatenate(out))
    return np.stack(zs)


class Scorer:
    """cands -> (Z (members, n), z_mean, z_pess). X is rounded to float16 first (crm_pools.py:41-43 stored the pools
    as float16 and scored the float32 cast of that), so the one-shot arm reproduces eval_v1 / gen_v1 bit-for-bit."""
    def __init__(self, model, start_xy, goal_xy, start_yaw, float16=True):
        self.model, self.start_xy, self.goal_xy, self.yaw, self.f16 = model, start_xy, goal_xy, start_yaw, float16

    def __call__(self, cands):
        X, L = corridors(cands)
        if self.f16:
            X = X.astype(np.float16).astype(np.float32)
        ctx = GP.geom_ctx(self.start_xy, self.goal_xy, self.yaw, L)
        Z = member_logits(self.model, X, ctx)
        return Z, Z.mean(0), Z.max(0)


def objective_values(z_mean, z_pess, T, objective='mean', c_fail=60.0):
    if objective == 'mean':
        return np.asarray(z_mean, float)
    if objective == 'pess':
        return np.asarray(z_pess, float)
    if objective == 'expected_cost':
        P = 1 - np.exp(-np.exp(np.asarray(z_mean, float)))
        return P * c_fail + np.asarray(T, float)
    raise ValueError(objective)


def ess_weights(J, ess_frac):
    """rc_planner.RouteMPPI._weights: softmax with the temperature bisected to hit ESS = ess_frac * len(J)."""
    J = np.asarray(J, float); J = J - J.min()
    target = max(2.0, ess_frac * len(J))
    lo, hi = 1e-3, 1e3
    for _ in range(40):
        T = math.sqrt(lo * hi)
        w = np.exp(-J / T); w /= w.sum()
        ess = 1.0 / np.sum(w ** 2)
        if ess < target:
            lo = T
        else:
            hi = T
    w = np.exp(-J / math.sqrt(lo * hi)); w /= w.sum()
    return w, float(1.0 / np.sum(w ** 2)), math.sqrt(lo * hi)


# ------------------------------------------------------------------------------------------------------------------
# the planner
# ------------------------------------------------------------------------------------------------------------------
def sample_valid(base, pose, rng, need, max_tries, L, fixed_speed, mu=None, sd=None):
    out, th, tries = [], [], 0
    while len(out) < need and tries < max_tries:
        tries += 1
        t = draw_prior(rng, L, fixed_speed, mu, sd)
        r = from_params(base, t, fixed_speed)
        if valid(r, pose):
            out.append(r); th.append(t)
    return out, th, tries


def plan_iter(base, pose, goal, scorer, rounds=4, n=64, elite_frac=0.15, sd_floor=0.15, weighting='cem', ess_frac=0.25,
              objective='mean', c_fail=60.0, rng=None, anchors=True, fixed_speed=None, tries_factor=8, mean_in_pop=True,
              score_mean=None):
    """Iterated resampling over theta. rounds=1 is the deployed one-shot planner (anchors + prior draws, argmin J).

    Round 0: the designed anchors (all 9; with fixed_speed only the 3 at that cruise speed) that pass the validator,
    then prior draws until n valid routes (anchors count towards n, at most tries_factor*n tries - as
    f104_n2_sampler.propose). Later rounds: the current refit mean first (mean_in_pop), then draws from N(mu, sd)
    until n valid. Every round: one batched corridor build + one score call (member logits -> z_mean, z_pess),
    J = objective. Refit on the CLIPPED thetas of everything evaluated so far that has a theta (anchors do not):
    CEM = best ceil(elite_frac*n) (min 4) by J, mu = mean, sd = max(std, sd_floor*prior_sd); MPPI = ESS-tempered
    softmax weights over all of them. After the last round the final mean is built and scored too (score_mean,
    default rounds > 1). Pick = argmin J over everything evaluated.
    """
    rng = np.random.default_rng() if rng is None else rng
    if score_mean is None:
        score_mean = rounds > 1
    dim = MODES if fixed_speed is not None else MODES + KNOTS
    prior_sd = PRIOR_SD[:dim]
    mu, sd = np.zeros(dim), prior_sd.copy()
    xy, station, f, L = _base_arrays(base)
    allc, allth, allz, allzp, allT, kinds, rnd, log = [], [], [], [], [], [], [], []
    tries_total = 0
    anc = []
    if anchors:
        sp = (2.0, 4.0, 6.0) if fixed_speed is None else (float(fixed_speed),)
        anc = [r for r in S.anchors(base, speeds=sp) if valid(r, pose)]

    def evaluate(cs, ths, ks, k):
        Z, zm, zp = scorer(cs)
        T = np.array([route_time(r) for r in cs])
        allc.extend(cs); allth.extend(ths); kinds.extend(ks); rnd.extend([k] * len(cs))
        allz.append(zm); allzp.append(zp); allT.append(T)
        return zm, zp, T

    for k in range(rounds):
        cs, ths, ks = [], [], []
        if k == 0:
            cs += anc; ths += [None] * len(anc); ks += ['anchor'] * len(anc)
        elif mean_in_pop:
            rm = from_params(base, mu, fixed_speed)
            if valid(rm, pose):
                cs.append(rm); ths.append(np.asarray(rm['meta']['theta'])); ks.append('mean')
        need = n - len(cs)
        drawn, dth, tries = sample_valid(base, pose, rng, need, tries_factor * n, L, fixed_speed,
                                         None if k == 0 else mu, None if k == 0 else sd)
        tries_total += tries
        cs += drawn; ths += dth; ks += ['sample'] * len(drawn)
        if not cs:
            log.append(dict(round=k, n=0, tries=tries, acceptance=0.0)); break
        zm, zp, T = evaluate(cs, ths, ks, k)
        Jall = objective_values(np.concatenate(allz), np.concatenate(allzp), np.concatenate(allT), objective, c_fail)
        has = np.array([t is not None for t in allth])
        entry = dict(round=k, n=len(cs), tries=tries, acceptance=len(drawn) / max(tries, 1),
                     zmin_round=float(zm.min()), zmin=float(np.concatenate(allz).min()),
                     jmin=float(Jall.min()), mu=mu.round(3).tolist(), sd=sd.round(3).tolist())
        if k < rounds - 1 or score_mean:
            Th = np.stack([t for t in allth if t is not None]) if has.any() else np.zeros((0, dim))
            Jh = Jall[has]
            if len(Th):
                if weighting == 'cem':
                    ne = min(len(Th), max(4, math.ceil(elite_frac * n)))
                    order = np.argsort(Jh, kind='stable')[:ne]
                    E = Th[order]
                    mu = E.mean(0); sd = np.maximum(E.std(0), sd_floor * prior_sd)
                    entry.update(n_elite=int(ne), ess=float(ne), j_elite_max=float(Jh[order[-1]]))
                elif weighting == 'mppi':
                    w, ess, temp = ess_weights(Jh, ess_frac)
                    mu = (w[:, None] * Th).sum(0)
                    sd = np.maximum(np.sqrt((w[:, None] * (Th - mu) ** 2).sum(0)), sd_floor * prior_sd)
                    entry.update(ess=ess, temperature=temp)
                else:
                    raise ValueError(weighting)
                mu = project(mu, L, fixed_speed)
            entry.update(mu_next=mu.round(3).tolist(), sd_next=sd.round(3).tolist())
        log.append(entry)
    if score_mean and allc:
        rm = from_params(base, mu, fixed_speed)
        if valid(rm, pose):
            evaluate([rm], [np.asarray(rm['meta']['theta'])], ['mean'], rounds)
            log.append(dict(round=rounds, n=1, tries=0, acceptance=1.0, kind='final_mean', zmin_round=float(allz[-1][0])))
        else:
            log.append(dict(round=rounds, n=0, tries=0, acceptance=0.0, kind='final_mean_invalid'))
    if not allc:
        return None
    Zm = np.concatenate(allz); Zp = np.concatenate(allzp); T = np.concatenate(allT)
    J = objective_values(Zm, Zp, T, objective, c_fail)
    i = int(np.argmin(J))          # ties -> first index (anchors first), as np.argmin in the deployed planner
    r = allc[i]
    r = {'waypoints': np.asarray(r['waypoints']), 'speeds': np.asarray(r['speeds']), 'stations': np.asarray(r['stations']),
         'headings': np.asarray(r['headings']),
         'meta': {**r.get('meta', {}), 'candidate': 'n2_iter' if rounds > 1 else r.get('meta', {}).get('candidate', 'n2_wide'),
                  'kind': kinds[i], 'round': int(rnd[i]), 'rank': int(i - sum(1 for q in rnd[:i] if q != rnd[i])),
                  'theta': None if allth[i] is None else np.asarray(allth[i]).tolist()}}
    return dict(route=r, index=i, kind=kinds[i], round=int(rnd[i]), theta=r['meta']['theta'],
                z_mean=float(Zm[i]), z_pess=float(Zp[i]), P=float(1 - np.exp(-np.exp(Zm[i]))), T=float(T[i]), J=float(J[i]),
                n_evaluated=len(allc), tries=tries_total, log=log, Z_mean=Zm, Z_pess=Zp, T_all=T, kinds=kinds,
                mu=mu.tolist(), sd=sd.tolist(), objective=objective)


def oneshot(base, pose, goal, scorer, rng, n=256, fixed_speed=None, objective='mean', c_fail=60.0):
    """The deployed planner: gen_planner.proposal_pool (anchors + 8n tries) or fixed2_pool (no anchors, 6n tries)."""
    if fixed_speed is None:
        return plan_iter(base, pose, goal, scorer, rounds=1, n=n, rng=rng, anchors=True, tries_factor=8,
                         objective=objective, c_fail=c_fail, score_mean=False)
    return plan_iter(base, pose, goal, scorer, rounds=1, n=n, rng=rng, anchors=False, fixed_speed=fixed_speed,
                     tries_factor=6, objective=objective, c_fail=c_fail, score_mean=False)


# ------------------------------------------------------------------------------------------------------------------
# checks
# ------------------------------------------------------------------------------------------------------------------
def _same_route(a, b):
    return all(np.array_equal(np.asarray(a[k], float), np.asarray(b[k], float)) for k in ('waypoints', 'speeds', 'stations', 'headings'))


def selftest(base, pose, seed_int, n=256, fixed_speed=None):
    """The one-shot pool rebuilt through from_params must equal gen_planner.proposal_pool / fixed2_pool route by route."""
    if fixed_speed is None:
        ref, _ = GP.proposal_pool(base, pose, np.random.default_rng(seed_int), n=n)
    else:
        ref = GP.fixed2_pool(base, pose, np.random.default_rng(seed_int), n=n)
    rng = np.random.default_rng(seed_int)
    xy, station, f, L = _base_arrays(base)
    if fixed_speed is None:
        mine = [r for r in S.anchors(base) if valid(r, pose)]
        drawn, _, _ = sample_valid(base, pose, rng, n - len(mine), 8 * n, L, None)
        mine += drawn
    else:
        mine, _, _ = sample_valid(base, pose, rng, n, 6 * n, L, fixed_speed)
    assert len(mine) == len(ref), (len(mine), len(ref))
    assert all(_same_route(a, b) for a, b in zip(mine, ref)), 'from_params does not reproduce sample_one'
    return len(ref)
