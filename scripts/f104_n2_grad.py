"""Gradient-based route refinement through the differentiable risk network (night-2 planner study, arms G/H).

Torch re-implementation of the deployed corridor chain, verified against the numpy originals (selfcheck()):
  (a (B,3), dv (B,4)) --t_shape--> raw route (pts, v, st) --t_station_tensor--> corridor X (B,5,96,32), L
  --Ensemble--> per-member route logits (M,B) --> objective --> Adam on (a, dv) with box projection.

Parameterisation = the night-2 sampler's own family (f104_n2_sampler): lateral = lat0 + sum_j a_j sin(j pi f) clipped
to +-10 m (lat0 is a constant per row: 0 for the wide samples, off*sin^2(pi f) for the designed anchors, so every pool
candidate is reproduced exactly), speed = clip(base + smoothstep knots(dv), 0.5, 6) under the terminal cone and the
forward/backward acceleration passes (vectorised min-plus form, exact). The final parameters are re-shaped in float64 by
numpy S.shape and checked by gen_planner.safe_validate (the validator's 1e-6 tolerances trip float32 routes).

Objectives: 'logit' (ensemble-mean z), 'expected_cost' (C_fail*P + T), 'expected_cost_energy' (+ lambda_E*E*KJ_TO_S)
with E from the analytic positive-work model (traverse_wp9_analytic constants, coefficients refit on f104 CRM W+ by
fit_energy_model). Keep-best and the final choice use the PESSIMISTIC score (max over the members the row was
optimised on); leave-one-member-out rows (optimise on 4, read the 5th) run in the same batch as guards.
"""
import hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / 'src'))
import f104_n2_dataset as DS          # noqa: E402
import f104_n2_sampler as S           # noqa: E402
import gen_planner as P               # noqa: E402
from gen_riskmodel import route_logit  # noqa: E402
import traverse_wp9_analytic as W     # noqa: E402  (physical constants + NNLS)

NS, NL, HW = DS.N_STATION, DS.N_LATERAL, DS.HALF_WIDTH_M
KAPPA_MAX = P.CFG.max_curvature_inv_m
MODES, KNOTS = 3, 4
LAT_CLIP, DV_CLIP = 10.0, 4.0
ARENA_SOFT = 37.0                     # 40 m minus footprint half-length 2.7 m and margin
KJ_TO_S = 0.2                         # 1 kJ = 0.2 s in the expected-cost objective
Z_SWITCH = math.log(-math.log(0.1))   # cloglog^-1(0.9): P is linearised beyond this (gradient of P vanishes at P->1)
RIGID_MODELS = str(ROOT / 'artifacts/traverse/fdm_f104_50h_20260909/night2_v1/final/N2_s*.pt')
ENERGY_TERMS = ['grade', 'desc', 'roll', 'accel', 'brake', 'aero', 'time', 'cross', 'tract', 'tract2',
                'tract_lowv', 'tract_hiv', 'grade_lowv', 'kemax', 'undul', 'one']
ENERGY_POOLS = {'tract': ['tract', 'tract2', 'tract_lowv', 'tract_hiv', 'time', 'cross', 'one'],
                'full': ENERGY_TERMS}


# ------------------------------------------------------------------ torch mirrors of the numpy semantics
def npgrad(f, h, dim):
    """np.gradient along dim (central interior, one-sided ends); h may be a broadcastable tensor."""
    f = f.movedim(dim, 0)
    g = torch.cat([f[1:2] - f[0:1], (f[2:] - f[:-2]) / 2, f[-1:] - f[-2:-1]], 0)
    return g.movedim(0, dim) / h


def tinterp(xq, xp, fp):
    """np.interp, batched: xq (B,Q), xp (B,N) increasing, fp (B,N). Gradient flows through values and breakpoints."""
    idx = (torch.searchsorted(xp.detach().contiguous(), xq.detach().contiguous(), right=True) - 1).clamp(0, xp.shape[1] - 2)
    x0 = xp.gather(1, idx); x1 = xp.gather(1, idx + 1)
    w = ((xq - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0, 1)
    return fp.gather(1, idx) * (1 - w) + fp.gather(1, idx + 1) * w


class TMap:
    """The static overhead map of f104_n2_dataset.G as a grid_sample-able tensor (border padding == numpy clipping)."""
    def __init__(self, G, device, dtype):
        z = G['rgbd'][3]
        self.R = torch.tensor(z, dtype=dtype, device=device)[None, None]
        self.vpx = torch.tensor(z > -1.999, device=device)
        self.n, self.mpp, self.ctr, self.es = G['npx'], float(G['mpp']), float(G['ctr']), float(G['elev_scale'])

    def sample(self, x, y):
        row = self.ctr - y / self.mpp; col = self.ctr + x / self.mpp
        grid = torch.stack([col / (self.n - 1) * 2 - 1, row / (self.n - 1) * 2 - 1], -1).reshape(1, -1, 1, 2)
        z = F.grid_sample(self.R, grid, mode='bilinear', padding_mode='border', align_corners=True).reshape(x.shape)
        r0 = torch.floor(row.detach()).long().clamp(0, self.n - 2); c0 = torch.floor(col.detach()).long().clamp(0, self.n - 2)
        valid = self.vpx[r0, c0] & self.vpx[r0, c0 + 1] & self.vpx[r0 + 1, c0] & self.vpx[r0 + 1, c0 + 1]
        return z, valid


def t_station_tensor(wp, sp, M):
    """Batched DS.station_tensor: wp (B,N,2), sp (B,N) -> X (B,5,96,32), route_len (B,)."""
    B = wp.shape[0]
    seg = torch.linalg.norm(wp[:, 1:] - wp[:, :-1], dim=-1); s = torch.cat([seg.new_zeros(B, 1), seg.cumsum(1)], 1)
    grid = torch.linspace(0, 1, NS, device=wp.device, dtype=wp.dtype)[None] * s[:, -1:]
    pts = torch.stack([tinterp(grid, s, wp[..., 0]), tinterp(grid, s, wp[..., 1])], -1)
    d = npgrad(pts, 1.0, 1); tn = torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(1e-9)
    tang = d / tn; nrm = torch.stack([-tang[..., 1], tang[..., 0]], -1)
    off = torch.linspace(-HW, HW, NL, device=wp.device, dtype=wp.dtype)
    gx = pts[..., 0:1] + nrm[..., 0:1] * off; gy = pts[..., 1:2] + nrm[..., 1:2] * off
    z, valid = M.sample(gx, gy); elev = z * M.es
    row0v = valid[:, 0]; e_mid = elev[:, 0, NL // 2]
    e_mean = (elev[:, 0] * row0v).sum(1) / row0v.sum(1).clamp_min(1)
    e0 = torch.where(row0v[:, NL // 2], e_mid, e_mean)
    fill = torch.where(valid, elev, e0[:, None, None])
    ds = (s[:, -1] / (NS - 1)).clamp_min(1e-3); dl = 2 * HW / (NL - 1)
    ga = npgrad(fill, ds[:, None, None], 1).clamp(-2, 2); gc = npgrad(fill, dl, 2).clamp(-2, 2)
    v = tinterp(grid, s, sp)
    X = torch.stack([fill - e0[:, None, None], ga, gc, v[..., None].expand(B, NS, NL), valid.to(wp.dtype)], 1)
    return X, s[:, -1]


def t_speed_knots(f, vals):
    """S.speed_knots batched: f (N,), vals (B,k) -> (B,N) smoothstep with free ends."""
    k = vals.shape[1]; kx = torch.linspace(0, 1, k, device=f.device, dtype=f.dtype)
    seg = (torch.searchsorted(kx, f.contiguous(), right=True) - 1).clamp(0, k - 2)
    u = (f - kx[seg]) / (kx[seg + 1] - kx[seg]); w = u * u * (3 - 2 * u)
    v0 = vals[:, seg]; v1 = vals[:, seg + 1]
    return v0 + (v1 - v0) * w[None]


def _minplus(u2, s, acc, forward):
    """v_j^2 = min_{k<=j} (u_k^2 + 2 acc (s_j - s_k)) (forward) or min_{k>=j} (u_k^2 + 2 acc (s_k - s_j)) (backward).
    Exactly the sequential torch.minimum/sqrt recurrence of S.shape, without the 105-step python loop."""
    B, N = u2.shape
    ds = s[:, :, None] - s[:, None, :]                       # (B, j, k) = s_j - s_k
    if forward:
        cand = u2[:, None, :] + 2 * acc * ds
        mask = torch.ones(N, N, dtype=torch.bool, device=u2.device).tril()
    else:
        cand = u2[:, None, :] - 2 * acc * ds
        mask = torch.ones(N, N, dtype=torch.bool, device=u2.device).triu()
    cand = torch.where(mask[None], cand, torch.full_like(cand, float('inf')))
    return cand.amin(2)


def t_shape(bxy, bst, bsp, a, dvk, lat0=None, lat_clip=LAT_CLIP):
    """Mirror of S.shape with lat = lat0 + sum_j a_j sin(j pi f) (clipped +-lat_clip) and dv = speed knots.
    bxy (N,2), bst (N,), bsp (N,) base route; a (B,3), dvk (B,4), lat0 (B,N) or None -> pts (B,N,2), v (B,N), st (B,N)."""
    B = a.shape[0]; f = ((bst - bst[0]) / (bst[-1] - bst[0])).clamp(0, 1)
    lat = sum(a[:, j:j + 1] * torch.sin((j + 1) * math.pi * f)[None] for j in range(a.shape[1]))
    if lat0 is not None:
        lat = lat + lat0
    lat = lat.clamp(-lat_clip, lat_clip)
    t = npgrad(bxy, 1.0, 0); t = t / torch.linalg.norm(t, dim=1, keepdim=True).clamp_min(1e-9)
    nrm = torch.stack([-t[:, 1], t[:, 0]], 1)
    pts = bxy[None] + lat[..., None] * nrm[None]
    seg = torch.linalg.norm(pts[:, 1:] - pts[:, :-1], dim=-1); st = torch.cat([seg.new_zeros(B, 1), seg.cumsum(1)], 1)
    u = (bsp[None] + t_speed_knots(f, dvk)).clamp(S.V_MIN, S.V_MAX)
    u = torch.minimum(u, torch.sqrt((2 * S.A_DEC * (st[:, -1:] - st)).clamp_min(1e-12)))    # terminal cone
    v2 = _minplus(u ** 2, st, S.A_ACC, forward=True)
    v2 = _minplus(v2, st, S.A_DEC, forward=False)
    return pts, torch.sqrt(v2.clamp_min(1e-12)), st


def t_curv(pts):
    """Validator's three-point curvature on the raw waypoints, per interior point: (B, N-2)."""
    a, b, c = pts[:, :-2], pts[:, 1:-1], pts[:, 2:]
    ab, bc, ac = b - a, c - b, c - a
    cross = (ab[..., 0] * bc[..., 1] - ab[..., 1] * bc[..., 0]).abs()
    den = torch.linalg.norm(ab, dim=-1) * torch.linalg.norm(bc, dim=-1) * torch.linalg.norm(ac, dim=-1)
    return 2 * cross / den.clamp_min(1e-9)


def t_ctx(start_xy, goal_xy, yaw, L):
    rel = np.asarray(goal_xy, float) - np.asarray(start_xy, float)
    base = torch.tensor([rel[0], rel[1], float(np.linalg.norm(rel)), float(yaw)], dtype=L.dtype, device=L.device)
    return torch.cat([base[None].expand(len(L), 4), L[:, None]], 1)


def commanded_time(v, st):
    """T = sum ds / max(mean segment speed, 0.25) on the raw route (s)."""
    vm = 0.5 * (v[:, 1:] + v[:, :-1]); ds = st[:, 1:] - st[:, :-1]
    return (ds / vm.clamp_min(0.25)).sum(1)


# ------------------------------------------------------------------ analytic positive work on the 96 stations
def energy_terms(X, L):
    """traverse_wp9_analytic.route_terms evaluated on the corridor's centre line (differentiable).
    X (B,5,96,32): elev_rel / grade / cross / speed / valid; L (B,) route length. Returns (B, len(ENERGY_TERMS))."""
    m, g, kj = W.MASS_KG, W.G, 1e-3
    c = NL // 2
    h = X[:, 0, :, c]; cr = X[:, 2, :, c].abs(); vc = X[:, 3, :, c]
    ds = (L / (NS - 1)).clamp_min(1e-3)[:, None]
    v = vc.clamp_min(0.3); vm = 0.5 * (v[:, 1:] + v[:, :-1])
    dh = h[:, 1:] - h[:, :-1]
    hyp = torch.sqrt((ds ** 2 + dh ** 2).clamp_min(1e-12)); sin_t, cos_t = dh / hyp, ds / hyp
    dke = 0.5 * m * (v[:, 1:] ** 2 - v[:, :-1] ** 2)
    a_seg = (v[:, 1:] ** 2 - v[:, :-1] ** 2) / (2 * ds)
    force = m * g * sin_t + W.CRR_NOM * m * g * cos_t + 0.5 * W.RHO_AIR * W.CDA_NOM * vm ** 2 + m * a_seg
    fp = F.relu(force); dt = ds / vm; crm = 0.5 * (cr[:, 1:] + cr[:, :-1]); r5 = vm / W.V_SCALE
    Lr = ds[:, 0] * (NS - 1)
    t = {'grade': kj * m * g * F.relu(dh).sum(1), 'desc': kj * m * g * F.relu(-dh).sum(1),
         'roll': kj * W.CRR_NOM * m * g * Lr,
         'accel': kj * (F.relu(dke).sum(1) + 0.5 * m * v[:, 0] ** 2), 'brake': kj * F.relu(-dke).sum(1),
         'aero': kj * 0.5 * W.RHO_AIR * W.CDA_NOM * (vm ** 2 * ds).sum(1), 'time': dt.sum(1),
         'cross': kj * m * g * (crm * ds).sum(1), 'tract': kj * (fp * ds).sum(1),
         'tract2': kj * (fp ** 2 * ds).sum(1) / (m * g), 'tract_lowv': kj * (fp * ds / r5).sum(1),
         'tract_hiv': kj * (fp * ds * r5).sum(1), 'grade_lowv': kj * m * g * (F.relu(dh) / r5).sum(1),
         'kemax': kj * 0.5 * m * v.amax(1) ** 2, 'undul': kj * m * g * dh.abs().sum(1),
         'one': torch.ones_like(Lr)}
    return torch.stack([t[n] for n in ENERGY_TERMS], 1)


def fit_energy_model(twin_npz, device='cuda', bs=2048):
    """Refit the analytic work model on f104 CRM: NNLS of total W+ (kJ, clean goal-reached routes) on the terms above,
    pool chosen by held-out (val+test groups) MAE. Returns {'names', 'w', 'pool', 'stats'}."""
    z = np.load(twin_npz, allow_pickle=True)
    ok = (z['fail'] == 0) & np.isfinite(z['total_wplus'])
    idx = np.flatnonzero(ok); A = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            sel = idx[i:i + bs]
            X = torch.tensor(z['X'][sel].astype(np.float32), device=device); L = torch.tensor(z['route_len'][sel], device=device)
            A.append(energy_terms(X, L).double().cpu().numpy())
    A = np.concatenate(A); y = z['total_wplus'][idx].astype(np.float64); split = z['split'][idx]
    tr = split == 'train'; ho = ~tr
    best = None
    for pool, names in ENERGY_POOLS.items():
        cols = [ENERGY_TERMS.index(n) for n in names]
        w = W.fit_nnls(A[tr][:, cols], y[tr])
        p = np.maximum(A[:, cols] @ w, 0)
        st = {'n_train': int(tr.sum()), 'n_heldout': int(ho.sum()),
              'train_mae_kj': float(np.abs(p[tr] - y[tr]).mean()), 'heldout_mae_kj': float(np.abs(p[ho] - y[ho]).mean()),
              'heldout_spearman': W.spearman(p[ho], y[ho]), 'heldout_mape_pct': float(100 * np.mean(np.abs(p[ho] - y[ho]) / y[ho])),
              'true_mean_kj': float(y.mean())}
        cand = {'pool': pool, 'names': names, 'w': [float(v) for v in w], 'stats': st,
                'coefficients_nonzero': {n: float(c) for n, c in zip(names, w) if c > 0}}
        if best is None or st['heldout_mae_kj'] < best['stats']['heldout_mae_kj']:
            best = cand
    best['source'] = str(twin_npz); best['target'] = 'total_wplus (kJ) of clean goal-reached CRM routes'
    best['nominal_constants'] = {'MASS_KG': W.MASS_KG, 'CRR_NOM': W.CRR_NOM, 'CDA_NOM': W.CDA_NOM, 'RHO_AIR': W.RHO_AIR}
    return best


def energy_kj(X, L, model):
    """E (B,) in kJ from a fitted model dict (or the raw positive tractive work when model is None)."""
    T = energy_terms(X, L)
    if model is None:
        return T[:, ENERGY_TERMS.index('tract')]
    w = torch.zeros(len(ENERGY_TERMS), dtype=X.dtype, device=X.device)
    for n, c in zip(model['names'], model['w']):
        w[ENERGY_TERMS.index(n)] = c
    return T @ w


# ------------------------------------------------------------------ ensemble
class Ensemble:
    """Differentiable wrapper around gen_planner.RiskModel members. logits(X, ctx) -> (M, B) route logits.
    A cuDNN GRU in eval mode cannot run backward; the GRU alone is put in train mode (it has no dropout, so its
    numerics are unchanged and cuDNN's fused kernels stay in use: 66 ms vs 429 ms per step for 102 rows with cuDNN
    disabled). BatchNorm and Dropout stay in eval mode. A copy of the members is made when the wrapper shares a
    RiskModel, so the numpy-side RiskModel.score is untouched."""
    def __init__(self, pattern, device='cuda', dtype=torch.float32, risk_model=None):
        import copy
        self.device, self.dtype = device, dtype
        rm = risk_model or P.RiskModel(pattern, device=device)
        self.risk_model = rm
        self.members = []
        for m, ck in rm.members:
            mm = copy.deepcopy(m).to(device, dtype).eval()
            if isinstance(mm.mix, torch.nn.GRU):
                mm.mix.train()
            t = lambda v: torch.tensor(np.asarray(v, np.float64), dtype=dtype, device=device)
            self.members.append((mm, t(ck['norm']['mu']), t(ck['norm']['sd']), t(ck['ctx_mu']), t(ck['ctx_sd'])))
        self.M = len(self.members)

    def logits(self, X, ctx, members=None):
        out = []
        with torch.backends.cudnn.flags(enabled=self.dtype == torch.float32):
            for i, (m, mu, sd, cmu, csd) in enumerate(self.members):
                if members is not None and i not in members:
                    continue
                x = torch.cat([(X[:, :4] - mu[None, :, None, None]) / sd[None, :, None, None], X[:, 4:5], torch.ones_like(X[:, :1])], 1)
                out.append(route_logit(m(x.to(self.dtype), ((ctx - cmu) / csd).to(self.dtype))))
        return torch.stack(out)


def p_unsafe(z):
    return 1 - torch.exp(-torch.exp(z))


def p_surrogate(z):
    """P(z) with a linear extension beyond P = 0.9 so that bad starts keep a gradient (scout section 7)."""
    slope = math.exp(Z_SWITCH) * math.exp(-math.exp(Z_SWITCH)); p_sw = 1 - math.exp(-math.exp(Z_SWITCH))
    return torch.where(z <= Z_SWITCH, p_unsafe(z), p_sw + slope * (z - Z_SWITCH))


OBJ_ID = {'logit': 0, 'expected_cost': 1, 'expected_cost_energy': 2}


def objective_value(obj, z, T, E, C_fail, lam_E, surrogate=False):
    """Per-row objective; obj (B,) int ids, z/T/E (B,), C_fail/lam_E (B,)."""
    Pz = p_surrogate(z) if surrogate else p_unsafe(z)
    cost = C_fail * Pz + T + lam_E * KJ_TO_S * E
    cost = torch.where(obj == OBJ_ID['expected_cost'], C_fail * Pz + T, cost)
    return torch.where(obj == 0, z, cost)


# ------------------------------------------------------------------ route family helpers (numpy side)
def base_frame(base):
    xy = np.asarray(base['waypoints'], float); st = np.asarray(base['stations'], float)
    f = np.clip((st - st[0]) / max(st[-1] - st[0], 1e-6), 0, 1); L = float(st[-1] - st[0])
    cap = 0.55 * KAPPA_MAX * L ** 2 / (np.arange(1, MODES + 1) * np.pi) ** 2
    return f, L, cap


def np_route(base, a, dv, lat0=None):
    """Float64 re-shape of (a, dv[, lat0]) with the sampler's own S.shape (what the pool candidates were made with)."""
    f, L, cap = base_frame(base)
    lat = sum(float(a[j]) * np.sin((j + 1) * np.pi * f) for j in range(MODES)) + (0.0 if lat0 is None else np.asarray(lat0, float))
    lat = np.clip(lat, -LAT_CLIP, LAT_CLIP)
    r = S.shape(np.asarray(base['waypoints'], float), np.asarray(base['stations'], float), np.asarray(base['speeds'], float),
                lat, S.speed_knots(f, KNOTS, np.asarray(dv, float)))
    r['meta'] = {'candidate': 'n2_grad', 'max_lateral_m': float(np.abs(lat).max()), 'mean_speed_mps': float(r['speeds'][1:-1].mean()),
                 'a': [float(x) for x in a], 'dv': [float(x) for x in dv]}
    return r


def propose_with_params(base, pose, rng, n=P.N_CAND, validate=None, cfg=None):
    """Mirror of S.propose (same rng call order, same functions) that also returns the parameters of every accepted
    candidate: (a (3,), dv (4,), lat0 (N,)) with lat0 = off*sin^2(pi f) for the 9 anchors (a = 0, dv = v-2) and 0 for
    the wide samples. The candidates are identical to gen_planner.proposal_pool's (asserted by the caller)."""
    validate = validate or P.safe_validate; cfg = cfg or P.CFG
    xy = np.asarray(base['waypoints'], float); station = np.asarray(base['stations'], float); speed = np.asarray(base['speeds'], float)
    f, L, cap = base_frame(base)
    out, params = [], []
    for r in S.anchors(base):
        if validate(r, [], cfg, pose)['valid']:
            out.append(r); mt = r['meta']
            params.append((np.zeros(MODES), np.full(KNOTS, mt['cruise_speed_mps'] - 2.0), mt['lateral_offset_m'] * np.sin(np.pi * f) ** 2))
    tries = 0
    while len(out) < n and tries < 8 * n:
        tries += 1
        lat, a = S.lateral_profile(f, L, rng, sigma=5.0, modes=MODES, kappa_max=KAPPA_MAX)
        lat = np.clip(lat, -LAT_CLIP, LAT_CLIP)
        dvr = np.clip(rng.normal(0, 1.5, KNOTS), -DV_CLIP, DV_CLIP)
        r = S.shape(xy, station, speed, lat, S.speed_knots(f, KNOTS, dvr))
        r['meta'] = {'candidate': 'n2_wide', 'max_lateral_m': float(np.abs(lat).max()), 'mean_speed_mps': float(r['speeds'][1:-1].mean())}
        if validate(r, [], cfg, pose)['valid']:
            out.append(r); params.append((a.copy(), dvr, np.zeros(len(f))))
    return out, tries, params


def route_hash(route):
    wp = np.round(np.asarray(route['waypoints'], float), 6); sp = np.round(np.asarray(route['speeds'], float), 6)
    return hashlib.md5(np.ascontiguousarray(wp).tobytes() + np.ascontiguousarray(sp).tobytes()).hexdigest()


# ------------------------------------------------------------------ the refinement
class Problem:
    """One start/goal case: base route, map, pose; caches the torch base tensors."""
    def __init__(self, base, pose, goal_xy, tmap, device='cuda', dtype=torch.float32):
        self.base, self.pose, self.goal = base, np.asarray(pose, float), np.asarray(goal_xy, float)
        self.M, self.device, self.dtype = tmap, device, dtype
        t = lambda v: torch.tensor(np.asarray(v, float), dtype=dtype, device=device)
        self.bxy, self.bst, self.bsp = t(base['waypoints']), t(base['stations']), t(base['speeds'])
        self.f, self.L0, self.cap = base_frame(base)
        self.cap_t = t(self.cap)
        tb = npgrad(self.bxy, 1.0, 0); tb = tb / torch.linalg.norm(tb, dim=1, keepdim=True).clamp_min(1e-9)
        self.nrm = torch.stack([-tb[:, 1], tb[:, 0]], 1)          # left normal of the base route (N,2)

    def forward(self, a, dv, lat0):
        pts, v, st = t_shape(self.bxy, self.bst, self.bsp, a, dv, lat0)
        X, L = t_station_tensor(pts, v, self.M)
        return pts, v, st, X, L, t_ctx(self.pose[:2], self.goal, self.pose[2], L)


def refine(prob, ens, rows, steps=100, lr_a=0.02, lr_dv=0.10, betas=(0.9, 0.99), clip_grad=10.0, patience=15,
           trust=0.0, energy_model=None, rigid=None, verbose=False):
    """Batched Adam multi-start descent.
    rows: dict with a0 (B,3), dv0 (B,4), lat0 (B,N), w (B,M) member weights (1 = optimise on this member),
          obj (B,) objective ids, C_fail (B,), lam_E (B,).
    Keep-best per row by the pessimistic objective (max over the row's fit members) + penalties; early stop when no row
    improved for `patience` steps. Returns per-row best parameters and diagnostics (all numpy)."""
    dev, dt = prob.device, prob.dtype
    t = lambda v: torch.tensor(np.asarray(v), dtype=dt, device=dev)
    a = t(rows['a0']).clone().requires_grad_(True); dv = t(rows['dv0']).clone().requires_grad_(True)
    lat0 = t(rows['lat0']); w = t(rows['w']); obj = torch.tensor(rows['obj'], device=dev)
    C_fail, lam_E = t(rows['C_fail']), t(rows['lam_E']); a_start, dv_start = a.detach().clone(), dv.detach().clone()
    B = a.shape[0]; wsum = w.sum(1); neg = torch.full((ens.M, B), -float('inf'), dtype=dt, device=dev)
    opt = torch.optim.Adam([{'params': [a], 'lr': lr_a}, {'params': [dv], 'lr': lr_dv}], betas=betas)
    best = {'J': torch.full((B,), float('inf'), dtype=dt, device=dev), 'a': a.detach().clone(), 'dv': dv.detach().clone(), 'step': torch.zeros(B, dtype=torch.long, device=dev)}
    since = torch.zeros(B, dtype=torch.long, device=dev)
    prevX = prevz = None; path_dz = torch.zeros(B, dtype=dt, device=dev); path_dX = torch.zeros(B, dtype=dt, device=dev)
    sd = torch.stack([m[2] for m in ens.members]).mean(0)[None, :, None, None]
    zero_frac = []; hist = []
    t0 = time.time(); n_steps = 0
    for it in range(steps + 1):
        pts, v, st, X, L, ctx = prob.forward(a, dv, lat0)
        Z = ens.logits(X, ctx)                                       # (M, B)
        z_fit = (Z * w.T).sum(0) / wsum
        z_pess = torch.where(w.T > 0, Z, neg).amax(0)
        T = commanded_time(v, st); E = energy_kj(X, L, energy_model)
        kap = t_curv(pts)
        pen = 1e5 * F.relu(kap - 0.95 * KAPPA_MAX).pow(2).sum(1) + 10.0 * F.relu(pts.abs() - ARENA_SOFT).pow(2).sum((1, 2))
        if trust > 0:
            pen = pen + trust * ((a - a_start) ** 2).sum(1) + trust * ((dv - dv_start) ** 2).sum(1)
        J_fit = objective_value(obj, z_fit, T, E, C_fail, lam_E, surrogate=True) + pen
        with torch.no_grad():
            J_keep = objective_value(obj, z_pess, T, E, C_fail, lam_E) + pen
            better = J_keep < best['J'] - 1e-3
            best['J'] = torch.where(better, J_keep, best['J']); best['step'] = torch.where(better, torch.full_like(since, it), best['step'])
            best['a'][better] = a.detach()[better]; best['dv'][better] = dv.detach()[better]
            since = torch.where(better, torch.zeros_like(since), since + 1)
            Xn = X[:, :4] / sd
            if prevX is not None:
                path_dz += (z_fit - prevz).abs(); path_dX += torch.sqrt((Xn - prevX).pow(2).mean((1, 2, 3)))
            prevX, prevz = Xn, z_fit.detach().clone()
            hist.append(J_keep.min())                                   # device-side; copied once at the end
        if it == steps or (it > 0 and bool((since >= patience).all())):
            break
        opt.zero_grad(set_to_none=True)
        J_fit.sum().backward()
        with torch.no_grad():
            ga, gd = a.grad, dv.grad
            zero_frac.append(float(((ga == 0).float().mean() * 3 + (gd == 0).float().mean() * 4) / 7))
            gn = torch.sqrt((ga ** 2).sum(1) + (gd ** 2).sum(1)).clamp_min(1e-12)
            sc = (clip_grad / gn).clamp_max(1.0)
            ga.mul_(sc[:, None]); gd.mul_(sc[:, None])
        opt.step(); n_steps += 1
        with torch.no_grad():
            a.clamp_(-prob.cap_t, prob.cap_t); dv.clamp_(-DV_CLIP, DV_CLIP)
        if verbose and it % 20 == 0:
            print(f'  step {it:3d}  best J min {best["J"].min():+.3f}  mean {best["J"].mean():+.3f}  {time.time() - t0:.1f}s', flush=True)
    # final evaluation of the kept iterates (all members, torch chain), plus the rigid second opinion on the same tensors
    with torch.no_grad():
        pts, v, st, X, L, ctx = prob.forward(best['a'], best['dv'], lat0)
        Z = ens.logits(X, ctx); z_fit = (Z * w.T).sum(0) / wsum; z_pess = torch.where(w.T > 0, Z, neg).amax(0)
        T = commanded_time(v, st); E = energy_kj(X, L, energy_model); kap = t_curv(pts).amax(1)
        lat = ((pts - prob.bxy[None]) * prob.nrm[None]).sum(-1)
        z_rig = rigid.logits(X, ctx) if rigid is not None else None
        pts0, v0, st0, X0, L0, ctx0 = prob.forward(a_start, dv_start, lat0)
        Z0 = ens.logits(X0, ctx0); z_rig0 = rigid.logits(X0, ctx0) if rigid is not None else None
        T0 = commanded_time(v0, st0); E0 = energy_kj(X0, L0, energy_model)
    cpu = lambda x: None if x is None else x.detach().cpu().numpy()
    return dict(a=cpu(best['a']), dv=cpu(best['dv']), best_step=cpu(best['step']), J_keep=cpu(best['J']),
                Z=cpu(Z), z_fit=cpu(z_fit), z_pess=cpu(z_pess), T=cpu(T), E=cpu(E), kappa_max=cpu(kap),
                mean_v=cpu(v[:, 1:-1].mean(1)), max_lat=cpu(lat.abs().amax(1)), max_abs_pt=cpu(pts.abs().amax((1, 2))),
                length=cpu(st[:, -1]), z_rigid=cpu(z_rig.mean(0)) if z_rig is not None else None,
                Z0=cpu(Z0), T0=cpu(T0), E0=cpu(E0), z_rigid0=cpu(z_rig0.mean(0)) if z_rig0 is not None else None,
                path_dz=cpu(path_dz), path_dX=cpu(path_dX), zero_grad_frac=float(np.mean(zero_frac)) if zero_frac else 0.0,
                steps_run=n_steps, seconds=time.time() - t0, best_J_trace=cpu(torch.stack(hist)))


# ------------------------------------------------------------------ selfcheck against the numpy originals
def selfcheck(device='cuda', verbose=True):
    """Torch chain vs numpy on the CRM demo pool (group 0013): corridor, logits/argmin, S.shape, min-plus passes, FD."""
    K = ROOT / 'artifacts/traverse/crm_f104_v1'
    DS.init_map(str(K / 'map_root'))
    pool = np.load(K / 'demo_v1/pool.npz')
    case = json.load(open(K / 'cases_eval/cases/f104_crm_eval_group_0013.json'))
    lay = case['layout']; pose = [lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']]
    base = {k: np.asarray(v, float) for k, v in json.load(open(K / 'cases_eval/cases/routes/f104_crm_eval_group_0013/route_00.json')).items()
            if k in ('waypoints', 'speeds', 'stations', 'headings')}
    dt = torch.float64; M = TMap(DS.G, device, dt); out = {}
    wp = torch.tensor(pool['wp'], dtype=dt, device=device); sp = torch.tensor(pool['sp'], dtype=dt, device=device)
    Xn, Ln = P.corridors([dict(waypoints=pool['wp'][i], speeds=pool['sp'][i], stations=pool['st'][i]) for i in range(len(wp))])
    Xt, Lt = t_station_tensor(wp, sp, M); Xt_np = Xt.cpu().numpy()
    out['corridor_max_abs_diff'] = {n: float(np.abs(Xt_np[:, c] - Xn[:, c]).max()) for c, n in enumerate(['elev_rel', 'grade', 'cross', 'speed', 'valid'])}
    out['route_len_max_abs_diff'] = float(np.abs(Lt.cpu().numpy() - Ln).max())
    ens = Ensemble(str(K / 'train_v1/deploy/CRM_N2_s*.pt'), device, dt)
    with torch.no_grad():
        zt = ens.logits(Xt, t_ctx(lay['start_xy'], case['goal_xy'], lay['start_yaw'], Lt)).cpu().numpy()
    out['logit_mean_max_abs_diff'] = float(np.abs(zt.mean(0) - pool['logit_crm']).max())
    out['logit_member_max_abs_diff'] = float(np.abs(zt - pool['logit_crm_members']).max())
    out['argmin_torch'] = int(zt.mean(0).argmin()); out['argmin_pool'] = int(np.argmin(pool['logit_crm']))
    # mirrored sampler reproduces the pool bit-for-bit and its parameters reproduce every candidate through t_shape
    rng = np.random.default_rng(int(hashlib.md5(('f104_crm_eval_group_0013' + 'crm_proposal').encode()).hexdigest()[:8], 16))
    cands, tries, params = propose_with_params(base, np.asarray(pose), rng)
    out['pool_mirror_max_abs_diff'] = float(max(max(np.abs(c['waypoints'] - pool['wp'][i]).max(), np.abs(c['speeds'] - pool['sp'][i]).max()) for i, c in enumerate(cands)))
    prob = Problem(base, pose, case['goal_xy'], M, device, dt)
    A = torch.tensor(np.stack([p[0] for p in params]), dtype=dt, device=device); D = torch.tensor(np.stack([p[1] for p in params]), dtype=dt, device=device)
    L0 = torch.tensor(np.stack([p[2] for p in params]), dtype=dt, device=device)
    pts, v, st = t_shape(prob.bxy, prob.bst, prob.bsp, A, D, L0)
    out['t_shape_vs_pool_waypoints'] = float(np.abs(pts.cpu().numpy() - pool['wp']).max()); out['t_shape_vs_pool_speeds'] = float(np.abs(v.cpu().numpy() - pool['sp']).max())
    # random parameters: t_shape (min-plus passes) vs S.shape (sequential passes)
    r = np.random.default_rng(1); f, L, cap = base_frame(base)
    Ar = np.clip(r.normal(0, 5.0, (16, 3)) / np.arange(1, 4), -cap, cap); Dr = np.clip(r.normal(0, 1.5, (16, 4)), -4, 4)
    pts, v, st = t_shape(prob.bxy, prob.bst, prob.bsp, torch.tensor(Ar, dtype=dt, device=device), torch.tensor(Dr, dtype=dt, device=device))
    ep = ev = 0.0
    for i in range(16):
        rr = np_route(base, Ar[i], Dr[i]); ep = max(ep, np.abs(rr['waypoints'] - pts[i].cpu().numpy()).max()); ev = max(ev, np.abs(rr['speeds'] - v[i].cpu().numpy()).max())
    out['t_shape_random_waypoints'] = float(ep); out['t_shape_random_speeds'] = float(ev)
    # autograd vs central finite differences (float64) on the top-4 pool candidates
    top = np.argsort(pool['logit_crm'])[:4]
    a0 = A[top].clone().requires_grad_(True); d0 = D[top].clone().requires_grad_(True); l0 = L0[top]
    def chain(a_, d_):
        _, _, _, X, L, ctx = prob.forward(a_, d_, l0); return ens.logits(X, ctx).mean(0)
    z = chain(a0, d0); z.sum().backward(); ga, gd = a0.grad.clone(), d0.grad.clone()
    h = 1e-4; fa = torch.zeros_like(ga); fd = torch.zeros_like(gd)
    with torch.no_grad():
        for j in range(3):
            e = torch.zeros_like(a0); e[:, j] = h; fa[:, j] = (chain(a0 + e, d0) - chain(a0 - e, d0)) / (2 * h)
        for j in range(4):
            e = torch.zeros_like(d0); e[:, j] = h; fd[:, j] = (chain(a0, d0 + e) - chain(a0, d0 - e)) / (2 * h)
    rel = lambda g, f_: ((g - f_).abs() / (f_.abs() + 1e-6)).cpu().numpy()
    out['fd_rel_err_a_median'] = float(np.median(rel(ga, fa))); out['fd_rel_err_a_max'] = float(rel(ga, fa).max())
    out['fd_rel_err_dv_median'] = float(np.median(rel(gd, fd))); out['fd_rel_err_dv_max'] = float(rel(gd, fd).max())
    out['grad_finite'] = bool(torch.isfinite(ga).all() and torch.isfinite(gd).all())
    if verbose:
        for k, v_ in out.items():
            print(f'  {k}: {v_}')
    return out


if __name__ == '__main__':
    selfcheck()
