"""Gradient route refinement (night-2 arm G) for the history-conditioned ensembles, planned from a recorded moving
decision state (PLAN crm_improve; module note artifacts/traverse/crm_improve_20260922/NOTES_ci_grad.md).

Per group (one decision state):
  1. Decision state exactly as ci_planner / ga_planner: ga_planner.decision_for with the --poses entry (pose, history
     window npz; base route = gen_planner.base_route(pose, goal)); v0 = ci_planner.decision_v0 (recorded only: the
     candidate family is the free night-2 family, which does not tie the route's start speed to the vehicle speed).
  2. Scorer = ci_planner.CIScorer (ga_planner.Scorer for ga_train / legacy ensembles): the history context of every
     member (16-d code, or the joint transformer's history tokens, plus the velocity context) is computed ONCE here.
  3. Arm B = the CEM 4x64 pick of the same ensemble, ci_planner's own loop in process (plan_iter_family, family free,
     rng default_rng(IT.seed(group, 'n2iter_cem4x64')), anchors, objective = ensemble-mean logit): the same pick as
     `ci_planner.py --family free --arms B` for the same decision (self-test T2).
  4. Starts (--starts, default 17): the B pick + the best (starts - 1) routes of the CEM's round-0 pool (the valid designed
     anchors and the prior draws, ranked by ensemble-mean logit; B itself is not repeated). Night 2 used the top 16 of
     the one-shot 256 pool plus the argmin's lateral mirror. Every start carries its exact parameters: samples and CEM
     means theta = (a1..a3 lateral sine amplitudes, dv1..dv4 speed knots) on the base route's 2 m/s profile; an anchor
     a = dv = 0, lateral lat0 = offset * sin^2(pi f) and a constant cruise-speed base profile. The float64 re-shape of
     every start's parameters reproduces its route bit for bit (checked per group, 'start_reproduced').
  5. Adam on theta for all starts in ONE GPU batch (--steps 60, --lr 0.02,0.10 = lateral / speed, betas (0.9, 0.99),
     per-row gradient-norm clip 10, projection onto the amplitude caps and +-4 m/s knots after every step, early stop when
     no row improved for --patience 15 steps). Objective = ensemble-mean route logit + curvature penalty
     1e5 * relu(kappa - 0.95 kappa_max)^2 + arena penalty 10 * relu(|xy| - 37 m)^2, through the torch corridor chain of
     f104_n2_grad (t_shape mirror of f104_n2_sampler.shape with per-row base profiles, t_station_tensor on the static
     depth map) and every member network, with the decision's history context held fixed (no re-encoding).
     The chain is written deterministically here (bilinear map sampling = f104_n2_dataset.sample_map with the map values
     as constants instead of grid_sample, one-hot takes instead of gather / indexing, deterministic cuDNN and the math
     attention kernel over forward and backward): in float64 it equals f104_n2_grad's chain to 1e-13 (self-test T1), and
     two runs give identical picks (T6); with grid_sample / gather the atomic adds of the backward pass made two runs
     end 0.1-0.2 apart in theta.
     Keep-best per row by --keep pessimistic (max over members, night 2; default) or mean, 1e-3 hysteresis.
  6. Finals: each row's kept parameters are re-shaped in float64 with the family's numpy functions (a row whose best
     iterate is its start keeps its start route object), checked with the planner validator (gen_planner.safe_validate
     anchored at the pose: curvature, speed limits, accel / decel limits, arena extent) and the route contract (start
     within 0.25 m of the pose, end within 0.25 m of the goal, finite, speeds in [0, 6]), and re-scored with the deployed
     scorer (numpy corridors, float16 rounding, the same ensemble and history context). G = the best valid final by the
     keep criterion if it beats B by at least --abstain-logit (0.3) in that criterion; otherwise G = B (abstained).

Outputs (--out): picks/<g>.json (arms B = the CEM pick entry in ga_planner's format, G = the refined entry with z_mean,
z_pess, P, abstained, gain; plus the per-start rows and timings), routes/<g>__G.json (always; the B route when G
abstained) and routes/<g>__B.json (only when B differs from G), tasks.json (one row per group for arm G, ga_planner row
format: id, group, case, route, run, tier, episode_seed, arms, sha256), tasks_new_only.json (G rows equal to B get
run=false, ref_arm B), summary.json, PICKS_LOCKED.sha256 (routes/*.json name + content, sorted, as ga_planner).
tasks.json plugs into ga_a5_pass2_tasks.py when the out dir is named picks_<world>_<ARM>.

  PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts OMP_NUM_THREADS=6
  K1=artifacts/traverse/generalist_20260921/A_adapt; K2=artifacts/traverse/crm_improve_20260922
  $PY scripts/ci_grad.py --cases $K1/suite/cases --map-root artifacts/traverse/crm_f104_v1/map_root \
      --models "$K2/deploy_v1/deploy_a3_haux_txjoint_s*.pt" --world crm --domain crm \
      --poses $K2/a5data/decision_suite_L1/poses_crm_all.json --task-root $K2 --out $K2/<dir>/picks_crm_XG
  $PY scripts/ci_grad.py --selftest $K2/ci_grad_selftest       # T1-T6 on 10 suite groups, 3 ensembles
"""
import argparse, copy, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_planner as CP
import f104_n2_grad as NG
GA, IT, GP, DS, S, PA = CP.GA, CP.IT, CP.GP, CP.DS, CP.S, CP.PA
torch = CP.torch
nn = torch.nn
F = torch.nn.functional
from gen_riskmodel import route_logit     # noqa: E402

ROOT = CP.ROOT
K1 = ROOT / 'artifacts/traverse/generalist_20260921/A_adapt'
K2 = ROOT / 'artifacts/traverse/crm_improve_20260922'
KAPPA_MAX = float(GP.CFG.max_curvature_inv_m)
ARENA_SOFT = NG.ARENA_SOFT
START_TOL_M = 0.25
assert (IT.MODES, IT.KNOTS, IT.LAT_CLIP, IT.SP_CLIP) == (NG.MODES, NG.KNOTS, NG.LAT_CLIP, NG.DV_CLIP)
assert np.allclose(IT.caps(40.0), NG.base_frame({'waypoints': np.zeros((2, 2)), 'stations': np.array([0.0, 40.0]), 'speeds': np.zeros(2)})[2])


# ------------------------------------------------------------------------------------------------------------------
# differentiable ensemble: the members' own networks, the decision's history context held fixed
# ------------------------------------------------------------------------------------------------------------------
def det_ctx(dtype):
    """Scope of every forward AND backward pass of the chain: deterministic cuDNN algorithms (TF32 allowed, as the
    deployed scorer), cuDNN off for float64 (the GRU's float64 path), and the math attention kernel (the fused attention
    kernels' backward passes are not deterministic). The global flags (used by the deployed scorer and the CEM) are
    untouched outside this scope."""
    import contextlib
    st = contextlib.ExitStack()
    st.enter_context(torch.backends.cudnn.flags(enabled=dtype == torch.float32, benchmark=False, deterministic=True, allow_tf32=True))
    st.enter_context(torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]))
    return st


class DiffEnsemble:
    """Differentiable copies of the members of a ci_planner.CIEnsemble / ga_planner.Ensemble. Deep copies (the scorer's
    models are untouched), parameters frozen, eval mode everywhere except the recurrent modules: cuDNN cannot run a
    backward pass through an RNN in eval mode, so they run in train mode (single-layer or dropout-free GRUs only, so the
    numerics are those of eval mode; the history-encoder GRUs are never called here). set_context(scorer) copies the
    decision's history context from the scorer (z / history tokens, velocity [vx, yaw rate], domain tag)."""
    def __init__(self, ens, dtype=torch.float32):
        self.ens, self.dev, self.dtype = ens, ens.dev, dtype
        self.ci = bool(getattr(ens, 'has_ci', False))
        self.members = []
        for mem in ens.members:
            m = copy.deepcopy(mem['model']).to(self.dev, dtype).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            for mod in m.modules():
                if isinstance(mod, nn.RNNBase):
                    assert mod.num_layers == 1 or mod.dropout == 0.0, 'multi-layer RNN with dropout: train mode would change it'
                    mod.train()
            ck = mem['ck']; nm = ck['norm']
            cont = list(nm.get('cont_index', [0, 1, 2, 3]))
            assert cont == [0, 1, 2, 3], f"{mem['name']}: corridor standardisation of channels {cont} not supported"
            t = lambda v: torch.as_tensor(np.asarray(v, np.float32).reshape(-1), device=self.dev).to(dtype)
            self.members.append(dict(name=mem['name'], model=m, kind=mem['kind'], cond=mem['cond'], mu=t(nm['mu']), sd=t(nm['sd']),
                                     cmu=t(ck['ctx_mu']), csd=t(ck['ctx_sd']), ctx_mode=mem.get('ctx_mode', 'geom'),
                                     arch_kind=getattr(m, 'kind', None), z=None, vel=None, tag=None))
        self.M = len(self.members)

    def set_context(self, scorer):
        dt, dev = self.dtype, self.dev
        for i, mem in enumerate(self.members):
            z = scorer.z.get(i) if hasattr(scorer, 'z') else None
            if z is None:
                mem['z'] = None
            elif isinstance(z, dict):                                   # txjoint: history tokens + mask
                mem['z'] = (torch.as_tensor(np.asarray(z['tok'], np.float32), device=dev).to(dt), torch.as_tensor(np.asarray(z['mask'], bool), device=dev))
            elif torch.is_tensor(z):                                    # ga_train: (1, zdim) on the device
                mem['z'] = z.detach().to(dev, dt)
            else:                                                       # ci_train gru / tx: (1, zdim) numpy
                mem['z'] = torch.as_tensor(np.asarray(z, np.float32), device=dev).to(dt)
            mem['vel'] = torch.as_tensor(np.asarray(scorer.vel, np.float32).reshape(1, 2), device=dev).to(dt) if (self.ci and mem['ctx_mode'] == 'geom_vel') else None
            tag = getattr(scorer, 'tag', None)
            mem['tag'] = torch.as_tensor(np.asarray(tag, np.float32).reshape(1, 2), device=dev).to(dt) if (tag is not None and mem['cond'] == 'tag') else None

    def logits(self, X, geom5):
        """X (B,5,96,32) raw corridor, geom5 (B,5) raw [goal dx, dy, |goal|, start yaw, route length] -> (M,B) logits."""
        B = X.shape[0]; out = []
        ones = torch.ones_like(X[:, :1])
        with det_ctx(self.dtype):
            for mem in self.members:
                x = torch.cat([(X[:, :4] - mem['mu'][None, :, None, None]) / mem['sd'][None, :, None, None], X[:, 4:5], ones], 1)
                c = geom5
                if mem['vel'] is not None:
                    c = torch.cat([c, mem['vel'].expand(B, 2)], 1)
                c = (c - mem['cmu']) / mem['csd']
                if mem['tag'] is not None:
                    c = torch.cat([c, mem['tag'].expand(B, 2)], 1)
                m, z = mem['model'], mem['z']
                if mem['kind'] == 'legacy':
                    haz = m(x, c)
                elif mem['kind'] == 'ga_train':
                    haz = m(x, c, z=None if z is None else z.expand(B, -1))['haz']
                else:
                    if z is None:
                        zb = None
                    elif isinstance(z, tuple):
                        zb = (z[0].expand(B, -1, -1), z[1].expand(B, -1))
                    else:
                        zb = z.expand(B, -1)
                    haz = m(x, c, z=zb)['haz']
                out.append(route_logit(haz))
        return torch.stack(out)


# ------------------------------------------------------------------------------------------------------------------
# deterministic corridor chain: f104_n2_grad's torch mirrors with every backward pass free of atomic adds
# (grid_sample's backward and the backward of gather / advanced indexing accumulate with atomics on CUDA, which made
# two identical runs end 0.1-0.2 apart in theta after 60 Adam steps)
# ------------------------------------------------------------------------------------------------------------------
class DetMap:
    """Bilinear sampling of the static depth map's elevation channel with the arithmetic of f104_n2_dataset.sample_map
    (cell index floor(.) clipped to [0, n-2], fractions clipped to [0, 1], valid = all 4 corners > -1.999). The map values
    are constants gathered at detached integer indices, so the gradient reaches the coordinates through the fractions
    only (no atomics). Same interface as f104_n2_grad.TMap."""
    def __init__(self, G, device, dtype):
        z = torch.as_tensor(np.asarray(G['rgbd'][3], np.float32), device=device).to(dtype)
        self.n, self.mpp, self.ctr, self.es = int(G['npx']), float(G['mpp']), float(G['ctr']), float(G['elev_scale'])
        self.Z = z.reshape(-1).contiguous(); self.V = (z > -1.999).reshape(-1).contiguous()

    def sample(self, x, y):
        n = self.n
        row = self.ctr - y / self.mpp; col = self.ctr + x / self.mpp
        r0 = torch.floor(row.detach()).long().clamp(0, n - 2); c0 = torch.floor(col.detach()).long().clamp(0, n - 2)
        fr = (row - r0.to(row.dtype)).clamp(0, 1); fc = (col - c0.to(col.dtype)).clamp(0, 1)
        i = r0 * n + c0
        p00, p01, p10, p11 = self.Z[i], self.Z[i + 1], self.Z[i + n], self.Z[i + n + 1]
        valid = self.V[i] & self.V[i + 1] & self.V[i + n] & self.V[i + n + 1]
        return p00 * (1 - fr) * (1 - fc) + p01 * (1 - fr) * fc + p10 * fr * (1 - fc) + p11 * fr * fc, valid


def _onehot_take(v, idx):
    """v (B,N) values at idx (B,Q) as a one-hot product: the backward is a plain reduction (no scatter_add)."""
    W = F.one_hot(idx, v.shape[1]).to(v.dtype)                      # (B,Q,N)
    return (W * v[:, None, :]).sum(-1)


def tinterp_det(xq, xp, fp):
    """f104_n2_grad.tinterp (np.interp, batched) with one-hot takes instead of gather."""
    idx = (torch.searchsorted(xp.detach().contiguous(), xq.detach().contiguous(), right=True) - 1).clamp(0, xp.shape[1] - 2)
    x0 = _onehot_take(xp, idx); x1 = _onehot_take(xp, idx + 1)
    w = ((xq - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0, 1)
    return _onehot_take(fp, idx) * (1 - w) + _onehot_take(fp, idx + 1) * w


def t_speed_knots_det(f, vals):
    """f104_n2_grad.t_speed_knots (smoothstep knots with free ends) with the knot lookup as a matrix product."""
    k = vals.shape[1]; kx = torch.linspace(0, 1, k, device=f.device, dtype=f.dtype)
    seg = (torch.searchsorted(kx, f.contiguous(), right=True) - 1).clamp(0, k - 2)
    u = (f - kx[seg]) / (kx[seg + 1] - kx[seg]); w = u * u * (3 - 2 * u)
    W0 = F.one_hot(seg, k).to(vals.dtype); W1 = F.one_hot(seg + 1, k).to(vals.dtype)   # (N,k)
    v0 = vals @ W0.T; v1 = vals @ W1.T
    return v0 + (v1 - v0) * w[None]


def t_station_tensor_det(wp, sp, M):
    """f104_n2_grad.t_station_tensor (batched f104_n2_dataset.station_tensor) with tinterp_det; M = DetMap."""
    NS, NL, HW = DS.N_STATION, DS.N_LATERAL, DS.HALF_WIDTH_M
    B = wp.shape[0]
    seg = torch.linalg.norm(wp[:, 1:] - wp[:, :-1], dim=-1); s = torch.cat([seg.new_zeros(B, 1), seg.cumsum(1)], 1)
    grid = torch.linspace(0, 1, NS, device=wp.device, dtype=wp.dtype)[None] * s[:, -1:]
    pts = torch.stack([tinterp_det(grid, s, wp[..., 0]), tinterp_det(grid, s, wp[..., 1])], -1)
    d = NG.npgrad(pts, 1.0, 1); tn = torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(1e-9)
    tang = d / tn; nrm = torch.stack([-tang[..., 1], tang[..., 0]], -1)
    off = torch.linspace(-HW, HW, NL, device=wp.device, dtype=wp.dtype)
    gx = pts[..., 0:1] + nrm[..., 0:1] * off; gy = pts[..., 1:2] + nrm[..., 1:2] * off
    z, valid = M.sample(gx, gy); elev = z * M.es
    row0v = valid[:, 0]; e_mid = elev[:, 0, NL // 2]
    e_mean = (elev[:, 0] * row0v).sum(1) / row0v.sum(1).clamp_min(1)
    e0 = torch.where(row0v[:, NL // 2], e_mid, e_mean)
    fill = torch.where(valid, elev, e0[:, None, None])
    ds = (s[:, -1] / (NS - 1)).clamp_min(1e-3); dl = 2 * HW / (NL - 1)
    ga = NG.npgrad(fill, ds[:, None, None], 1).clamp(-2, 2); gc = NG.npgrad(fill, dl, 2).clamp(-2, 2)
    v = tinterp_det(grid, s, sp)
    X = torch.stack([fill - e0[:, None, None], ga, gc, v[..., None].expand(B, NS, NL), valid.to(wp.dtype)], 1)
    return X, s[:, -1]


# ------------------------------------------------------------------------------------------------------------------
# the route family with per-row base profiles (torch and float64 numpy)
# ------------------------------------------------------------------------------------------------------------------
def t_shape_rows(bxy, bst, bsp, a, dvk, lat0, lat_clip=IT.LAT_CLIP, knots=t_speed_knots_det):
    """f104_n2_grad.t_shape with a per-row base speed profile: bxy (N,2), bst (N,), bsp (B,N), a (B,3), dvk (B,4),
    lat0 (B,N) -> pts (B,N,2), v (B,N), st (B,N). lat = lat0 + sum_j a_j sin(j pi f) clipped +-10 m; speed =
    clip(bsp + smoothstep knots(dv), 0.5, 6) under the terminal cone and the forward / backward acceleration passes."""
    B = a.shape[0]; f = ((bst - bst[0]) / (bst[-1] - bst[0])).clamp(0, 1)
    lat = sum(a[:, j:j + 1] * torch.sin((j + 1) * math.pi * f)[None] for j in range(a.shape[1]))
    lat = (lat + lat0).clamp(-lat_clip, lat_clip)
    t = NG.npgrad(bxy, 1.0, 0); t = t / torch.linalg.norm(t, dim=1, keepdim=True).clamp_min(1e-9)
    nrm = torch.stack([-t[:, 1], t[:, 0]], 1)
    pts = bxy[None] + lat[..., None] * nrm[None]
    seg = torch.linalg.norm(pts[:, 1:] - pts[:, :-1], dim=-1); st = torch.cat([seg.new_zeros(B, 1), seg.cumsum(1)], 1)
    u = (bsp + knots(f, dvk)).clamp(S.V_MIN, S.V_MAX)
    u = torch.minimum(u, torch.sqrt((2 * S.A_DEC * (st[:, -1:] - st)).clamp_min(1e-12)))
    v2 = NG._minplus(u ** 2, st, S.A_ACC, forward=True)
    v2 = NG._minplus(v2, st, S.A_DEC, forward=False)
    return pts, torch.sqrt(v2.clamp_min(1e-12)), st


def np_route(base, a, dv, anchor=None):
    """Float64 route of the family. anchor None: IT.from_params(base, [a, dv]) (samples, CEM means). anchor =
    (offset_m, cruise_mps): lat0 = offset * sin^2(pi f), constant cruise base profile (a = dv = 0 gives S.anchors')."""
    a = np.asarray(a, float); dv = np.asarray(dv, float)
    if anchor is None:
        r = IT.from_params(base, np.r_[a, dv])
        r['meta'].update(a=[float(x) for x in r['meta']['theta'][:IT.MODES]], dv=[float(x) for x in r['meta']['theta'][IT.MODES:]], anchor=None)
        return r
    xy, station, f, L = IT._base_arrays(base)
    cap = IT.caps(L); aa = np.clip(a, -cap, cap)
    lat = sum(aa[j] * np.sin((j + 1) * np.pi * f) for j in range(IT.MODES)) + float(anchor[0]) * np.sin(np.pi * f) ** 2
    lat = np.clip(lat, -IT.LAT_CLIP, IT.LAT_CLIP)
    dvk = np.clip(dv, -IT.SP_CLIP, IT.SP_CLIP)
    r = S.shape(xy, station, np.full(len(xy), float(anchor[1])), lat, S.speed_knots(f, IT.KNOTS, dvk))
    r['meta'] = {'candidate': 'n2_grad_anchor', 'theta': None, 'a': aa.tolist(), 'dv': dvk.tolist(), 'anchor': [float(anchor[0]), float(anchor[1])],
                 'max_lateral_m': float(np.abs(lat).max()), 'mean_speed_mps': float(r['speeds'][1:-1].mean())}
    return r


def start_params(route, kind):
    """(a (3,), dv (4,), anchor or None) of a planner candidate: theta for samples / means, the designed offset and
    cruise speed for anchors."""
    m = route.get('meta', {})
    if kind == 'anchor' or m.get('candidate') == 'n2_anchor':
        return np.zeros(IT.MODES), np.zeros(IT.KNOTS), (float(m['lateral_offset_m']), float(m['cruise_speed_mps']))
    th = np.asarray(m['theta'], float)
    assert th.shape == (IT.MODES + IT.KNOTS,), th.shape
    return th[:IT.MODES].copy(), th[IT.MODES:].copy(), None


def route_contract(route, pose, goal):
    """Planner validator (anchored at the pose) + collector contract: start within 0.25 m of the pose, end within
    0.25 m of the goal, finite arrays, speeds in [0, 6]. Returns (ok, info)."""
    chk = GP.safe_validate(route, [], GP.CFG, np.asarray(pose, float))
    wp = np.asarray(route['waypoints'], float); v = np.asarray(route['speeds'], float)
    d0 = float(np.linalg.norm(wp[0] - np.asarray(pose, float)[:2])); d1 = float(np.linalg.norm(wp[-1] - np.asarray(goal, float)))
    finite = bool(np.isfinite(wp).all() and np.isfinite(v).all() and np.isfinite(np.asarray(route['stations'], float)).all())
    reasons = list(chk.get('reasons', []))
    if d0 > START_TOL_M: reasons.append('start_off_pose')
    if d1 > START_TOL_M: reasons.append('end_off_goal')
    if not finite: reasons.append('non_finite')
    if finite and (v.min() < -1e-9 or v.max() > S.V_MAX + 1e-6): reasons.append('speed_range')
    acc = CP.accel_profile(route)
    return (not reasons), dict(valid_validator=bool(chk['valid']), reasons=reasons, start_dist_m=d0, end_dist_m=d1,
                               max_curvature=float(chk.get('max_curvature', float('nan'))), v_min=float(v.min()), v_max=float(v.max()),
                               accel_max=float(acc.max(initial=0.0)), accel_min=float(acc.min(initial=0.0)))


# ------------------------------------------------------------------------------------------------------------------
# the chain and the refinement
# ------------------------------------------------------------------------------------------------------------------
class Chain:
    """theta rows -> route -> corridor -> member logits for one decision (base route, pose, goal fixed).
    det=True (default): the deterministic mirrors above (tmap = DetMap); det=False: f104_n2_grad's own t_speed_knots /
    t_station_tensor (tmap = f104_n2_grad.TMap, grid_sample), kept for the equivalence check."""
    def __init__(self, base, pose, goal, tmap, dens, bsp_rows, lat0_rows, det=True):
        dt, dev = dens.dtype, dens.dev
        self.det = det
        self.knots, self.station = (t_speed_knots_det, t_station_tensor_det) if det else (NG.t_speed_knots, NG.t_station_tensor)
        t = lambda v: torch.as_tensor(np.asarray(v, np.float64), device=dev).to(dt)
        self.pose, self.goal, self.tmap, self.dens = np.asarray(pose, float), np.asarray(goal, float), tmap, dens
        self.bxy, self.bst = t(base['waypoints']), t(base['stations'])
        self.bsp, self.lat0 = t(bsp_rows), t(lat0_rows)
        xy, station, f, L = IT._base_arrays(base)
        self.cap = t(IT.caps(L))

    def __call__(self, a, dv, rows=None):
        bsp, lat0 = (self.bsp, self.lat0) if rows is None else (self.bsp[rows], self.lat0[rows])
        pts, v, st = t_shape_rows(self.bxy, self.bst, bsp, a, dv, lat0, knots=self.knots)
        X, L = self.station(pts, v, self.tmap)
        geom = NG.t_ctx(self.pose[:2], self.goal, self.pose[2], L)
        return pts, v, st, X, L, self.dens.logits(X, geom)


def penalty(pts):
    kap = NG.t_curv(pts)
    return 1e5 * F.relu(kap - 0.95 * KAPPA_MAX).pow(2).sum(1) + 10.0 * F.relu(pts.abs() - ARENA_SOFT).pow(2).sum((1, 2))


def refine(chain, a0, dv0, steps=60, lr_a=0.02, lr_dv=0.10, keep='pessimistic', patience=15, clip_grad=10.0, betas=(0.9, 0.99)):
    """Batched Adam multi-start descent (f104_n2_grad.refine, objective 'logit'): optimise the ensemble-mean logit +
    penalties, keep-best per row by the pessimistic (max member) or mean logit + penalties (1e-3 hysteresis), project
    onto the caps after every step, stop early when no row improved for `patience` steps."""
    dev, dt = chain.dens.dev, chain.dens.dtype
    a = torch.as_tensor(a0, device=dev).to(dt).clone().requires_grad_(True)
    dv = torch.as_tensor(dv0, device=dev).to(dt).clone().requires_grad_(True)
    B = a.shape[0]
    opt = torch.optim.Adam([{'params': [a], 'lr': lr_a}, {'params': [dv], 'lr': lr_dv}], betas=betas)
    best_J = torch.full((B,), float('inf'), dtype=dt, device=dev); best_a = a.detach().clone(); best_dv = dv.detach().clone()
    best_step = torch.zeros(B, dtype=torch.long, device=dev); since = torch.zeros(B, dtype=torch.long, device=dev)
    J0 = Z0 = None; n_steps = 0; trace = []
    for it in range(steps + 1):
        pts, v, st, X, L, Z = chain(a, dv)
        z_mean = Z.mean(0); pen = penalty(pts)
        J_fit = z_mean + pen
        with torch.no_grad():
            J_keep = (Z.amax(0) if keep == 'pessimistic' else z_mean) + pen
            if it == 0:
                J0, Z0, pen0 = J_keep.clone(), Z.detach().clone(), pen.detach().clone()
            better = J_keep < best_J - 1e-3
            best_J = torch.where(better, J_keep, best_J); best_step = torch.where(better, torch.full_like(best_step, it), best_step)
            best_a[better] = a.detach()[better]; best_dv[better] = dv.detach()[better]
            since = torch.where(better, torch.zeros_like(since), since + 1)
            trace.append(best_J.min())
        if it == steps or (it > 0 and bool((since >= patience).all())):
            break
        opt.zero_grad(set_to_none=True)
        with det_ctx(dt):
            J_fit.sum().backward()
        with torch.no_grad():
            ga, gd = a.grad, dv.grad
            gn = torch.sqrt((ga ** 2).sum(1) + (gd ** 2).sum(1)).clamp_min(1e-12)
            sc = (clip_grad / gn).clamp_max(1.0); ga.mul_(sc[:, None]); gd.mul_(sc[:, None])
        opt.step(); n_steps += 1
        with torch.no_grad():
            a.clamp_(-chain.cap, chain.cap); dv.clamp_(-IT.SP_CLIP, IT.SP_CLIP)
    with torch.no_grad():
        pts, v, st, X, L, Z = chain(best_a, best_dv)
    cpu = lambda x: x.detach().double().cpu().numpy()
    return dict(a=cpu(best_a), dv=cpu(best_dv), best_step=best_step.cpu().numpy(), J_keep=cpu(best_J), J0=cpu(J0), Z0=cpu(Z0), pen0=cpu(pen0),
                Z=cpu(Z), pen=cpu(penalty(pts)), steps_run=n_steps, trace=cpu(torch.stack(trace)))


# ------------------------------------------------------------------------------------------------------------------
# one decision
# ------------------------------------------------------------------------------------------------------------------
def keep_value(keep, z_mean, z_pess):
    return float(z_pess if keep == 'pessimistic' else z_mean)


def b_entry(res, rid, sp, pose):
    """arm B entry, the fields of ga_planner.main."""
    r = res['route']
    e = dict(route_id=rid, arm='B', label=sp['label'], tag=sp['tag'], objective=sp['objective'], index=res['index'], kind=res['kind'],
             round=res['round'], theta=res['theta'], z_mean=res['z_mean'], z_pess=res['z_pess'], P=res['P'], T=res['T'], J=res['J'],
             mean_speed=float(np.asarray(r['speeds'])[1:-1].mean()), max_lateral_m=float(r['meta'].get('max_lateral_m', 0.0)),
             length_m=float(np.asarray(r['stations'])[-1]), route_sha256=IT.route_sha256(r), n_evaluated=res['n_evaluated'], tries=res['tries'],
             n_below_1pct=int(((1 - np.exp(-np.exp(res['Z_mean']))) < 0.01).sum()), z_mean_min=float(res['Z_mean'].min()),
             z_pess_min=float(res['Z_pess'].min()), wall_s=None, start_xy=[float(v) for v in np.asarray(r['waypoints'])[0]],
             start_dist_to_pose_m=float(np.linalg.norm(np.asarray(r['waypoints'])[0] - pose[:2])),
             log=res['log'], mu_final=res['mu'], sd_final=res['sd'], kinds_count={k: int(sum(1 for q in res['kinds'] if q == k)) for k in ('anchor', 'sample', 'mean')})
    return e


def plan_group(g, cp, poses, ens, dens, tmap, opts, specB, domain):
    """B (CEM 4x64) and G (gradient refinement) for one group. Returns (summ dict, B route, G route, timings)."""
    t0 = time.perf_counter()
    a = argparse.Namespace(from_run=None, frame=opts.frame, pose_along_s=None, pose_override=None, goal=None, history=None)
    case, lay, pose, goal, base, hist, hmask, src = GA.decision_for(g, cp, a, poses, ens.hist_T)
    entry = (poses or {}).get(g, {})
    v0, v0_how, v0_check = CP.decision_v0(entry, hist, hmask, src, None, opts.frame)
    scorer = CP.CIScorer(ens, pose[:2], goal, float(pose[2]), domain, hist, hmask, float16=True)
    fam = CP.Family('free', v0, float(pose[2]))
    rng = np.random.default_rng(IT.seed(g, specB['tag']))
    res = CP.plan_iter_family(base, pose, goal, scorer, fam, rounds=specB['rounds'], n=specB['n'], objective=specB['objective'],
                              c_fail=specB.get('c_fail', 60.0), rng=rng, anchors=True, fixed_speed=None, keep_candidates=True)
    assert res is not None, f'{g}: no valid CEM candidate'
    t_cem = time.perf_counter() - t0
    rB = res['route']; hB = IT.route_sha256(rB)
    # ---- starts: B + the best of the round-0 pool (anchors + prior draws) by ensemble-mean logit
    n0 = int(res['log'][0]['n']); cands = res['candidates']; kinds = res['kinds']
    order0 = [int(i) for i in np.argsort(res['Z_mean'][:n0], kind='stable') if int(i) != res['index']]
    pool_idx = order0[:max(opts.starts - 1, 0)]
    starts = [dict(src='B', pool_index=int(res['index']), kind=res['kind'], route=rB, z_mean=res['z_mean'], z_pess=res['z_pess'])]
    starts += [dict(src='round0', pool_index=i, kind=kinds[i], route=cands[i], z_mean=float(res['Z_mean'][i]), z_pess=float(res['Z_pess'][i])) for i in pool_idx]
    xy, station, f, L = IT._base_arrays(base)
    A0, D0, BSP, LAT0 = [], [], [], []
    for s in starts:
        aa, dd, anc = start_params(s['route'], s['kind'])
        s['a0'], s['dv0'], s['anchor'] = aa, dd, anc
        rr = np_route(base, aa, dd, anc)
        s['start_reproduced'] = IT.route_sha256(rr) == IT.route_sha256(s['route'])
        A0.append(aa); D0.append(dd)
        BSP.append(np.asarray(base['speeds'], float) if anc is None else np.full(len(xy), anc[1]))
        LAT0.append(np.zeros(len(xy)) if anc is None else anc[0] * np.sin(np.pi * f) ** 2)
    # ---- refinement (all starts in one batch)
    dens.set_context(scorer)
    chain = Chain(base, pose, goal, tmap, dens, np.stack(BSP), np.stack(LAT0))
    if dens.dev != 'cpu' and str(dens.dev).startswith('cuda'): torch.cuda.synchronize()
    t1 = time.perf_counter()
    R = refine(chain, np.stack(A0), np.stack(D0), steps=opts.steps, lr_a=opts.lr_a, lr_dv=opts.lr_dv, keep=opts.keep,
               patience=opts.patience, clip_grad=opts.clip_grad)
    if str(dens.dev).startswith('cuda'): torch.cuda.synchronize()
    t_ref = time.perf_counter() - t1
    # ---- finals: float64 re-shape, validator + contract, deployed re-score
    t2 = time.perf_counter()
    finals, fin_routes = [], []
    for k, s in enumerate(starts):
        refined = int(R['best_step'][k]) > 0
        route = np_route(base, R['a'][k], R['dv'][k], s['anchor']) if refined else s['route']
        ok, info = route_contract(route, pose, goal)
        finals.append(dict(row=k, src=s['src'], pool_index=s['pool_index'], kind=s['kind'], anchor=s['anchor'], start_reproduced=bool(s['start_reproduced']),
                           refined=refined, best_step=int(R['best_step'][k]), a=[float(x) for x in R['a'][k]], dv=[float(x) for x in R['dv'][k]],
                           z0_mean=s['z_mean'], z0_pess=s['z_pess'], J0_torch=float(R['J0'][k]), J_torch=float(R['J_keep'][k]),
                           z0_torch_mean=float(R['Z0'][:, k].mean()), z0_torch_pess=float(R['Z0'][:, k].max()),
                           z_torch_mean=float(R['Z'][:, k].mean()), z_torch_pess=float(R['Z'][:, k].max()), pen_torch=float(R['pen'][k]),
                           valid=bool(ok), **info, route_sha256=IT.route_sha256(route)))
        fin_routes.append(route)
    vidx = [k for k, fr in enumerate(finals) if fr['valid']]
    # one deployed re-score batch: every start route (row 0 = the B pick), then every valid final. The deployed scores
    # move by ~1e-2 with the batch shape (cuDNN / TF32 algorithm choice), so B, the starts and the finals are compared
    # inside one batch.
    ns = len(starts)
    Zf, zmf, zpf = scorer([s['route'] for s in starts] + [fin_routes[k] for k in vidx])
    for k in range(ns):
        finals[k].update(z0_rescored_mean=float(zmf[k]), z0_rescored_pess=float(zpf[k]))
    for j, k in enumerate(vidx, start=ns):
        fr = finals[k]; fr.update(z_mean=float(zmf[j]), z_pess=float(zpf[j]), z_members=[float(x) for x in Zf[:, j]],
                                  J_keep=keep_value(opts.keep, zmf[j], zpf[j]), J0_keep=keep_value(opts.keep, zmf[k], zpf[k]),
                                  T=IT.route_time(fin_routes[k]))
        fr['dJ_vs_start'] = fr['J_keep'] - fr['J0_keep']
    J_B_cem = keep_value(opts.keep, res['z_mean'], res['z_pess'])
    J_B = keep_value(opts.keep, zmf[0], zpf[0]); zb = np.asarray(Zf[:, 0])
    best = min(vidx, key=lambda k: finals[k]['J_keep']) if vidx else None
    gain = (J_B - finals[best]['J_keep']) if best is not None else float('nan')
    abstain = best is None or not (gain >= opts.abstain_logit)
    t_fin = time.perf_counter() - t2
    # ---- G entry
    if abstain:
        rG = rB; zG_mean, zG_pess, zG_members = res['z_mean'], res['z_pess'], None; fG = None
    else:
        fG = finals[best]; rG = fin_routes[best]; zG_mean, zG_pess, zG_members = fG['z_mean'], fG['z_pess'], fG['z_members']
    hG = IT.route_sha256(rG)
    okG, infoG = route_contract(rG, pose, goal)
    eG = dict(route_id=f'{g}__G', arm='G', label=f'grad{opts.steps}_{opts.keep}', objective='logit', keep=opts.keep, abstain_logit=opts.abstain_logit,
              abstained=bool(abstain), gain=(float(gain) if np.isfinite(gain) else None), gain_unit='logit (keep criterion, B minus G)',
              J_B=J_B, J_B_cem=J_B_cem, B_rescored=dict(z_mean=float(zmf[0]), z_pess=float(zpf[0])), J_G=(J_B if abstain else fG['J_keep']), z_mean=float(zG_mean), z_pess=float(zG_pess), P=float(1 - np.exp(-np.exp(zG_mean))),
              P_pess=float(1 - np.exp(-np.exp(zG_pess))), z_members=zG_members, T=IT.route_time(rG),
              mean_speed=float(np.asarray(rG['speeds'])[1:-1].mean()), max_lateral_m=float(rG['meta'].get('max_lateral_m', 0.0)),
              length_m=float(np.asarray(rG['stations'])[-1]), route_sha256=hG, same_route_as_B=(hG == hB),
              start_row=(None if abstain else int(best)), start_src=(None if abstain else fG['src']), start_kind=(None if abstain else fG['kind']),
              start_pool_index=(None if abstain else fG['pool_index']), best_step=(None if abstain else fG['best_step']), refined=(False if abstain else fG['refined']),
              theta=(None if abstain else dict(a=fG['a'], dv=fG['dv'], anchor=fG['anchor'])),
              n_finals=len(finals), n_valid_finals=len(vidx), n_refined_rows=int(sum(fr['refined'] for fr in finals)),
              validator_pass_rate=float(len(vidx) / max(len(finals), 1)), reasons=sorted({r_ for fr in finals for r_ in fr['reasons']}),
              contract=dict(valid=bool(okG), **infoG), start_xy=[float(v) for v in np.asarray(rG['waypoints'])[0]],
              start_dist_to_pose_m=infoG['start_dist_m'], end_dist_to_goal_m=infoG['end_dist_m'], steps_run=int(R['steps_run']))
    if not abstain:
        eG['frac_members_improved_vs_B'] = float(np.mean(np.asarray(zG_members) < zb))
    # night-2 exploitation flags (diagnostics only; nothing is filtered on them)
    wpG = np.asarray(rG['waypoints'], float)
    eG['flags'] = dict(mean_worse=bool(not abstain and zG_mean > float(zmf[0])), minority_members=bool(not abstain and eG['frac_members_improved_vs_B'] < 0.5),
                       lateral_clip=bool(not abstain and eG['max_lateral_m'] >= IT.LAT_CLIP - 0.1),
                       dv_clip=bool(not abstain and int(np.sum(np.abs(fG['dv']) >= IT.SP_CLIP - 1e-6)) >= 3),
                       fast=bool(not abstain and eG['mean_speed'] > 5.0), edge=bool(not abstain and float(np.abs(wpG).max()) > 34.0))
    eG['n_flags'] = int(sum(eG['flags'].values()))
    eB = b_entry(res, f'{g}__B' if hG != hB else f'{g}__G', specB, pose)
    eB['z_members_rescored'] = [float(x) for x in zb]
    summ = dict(group=g, world=opts.world, domain=domain, fixed2=False, stratum=case.get('evaluation_stratum'), base_length_m=float(base['stations'][-1]),
                corridors_batched=IT._CORR.get('identical'), arms=dict(B=eB, G=eG),
                pose=[float(v) for v in pose], goal=[float(v) for v in goal], source=src, history=scorer.history, score_calls=scorer.calls,
                family=dict(v0_mps=v0, v0_source=v0_how, v0_check=v0_check, yaw=float(pose[2]), family=fam.describe(),
                            base_start_heading_err_deg=CP.start_heading_err_deg(base, pose[2]),
                            arms=dict(B=CP.pick_stats(rB, v0, pose[2]), G=CP.pick_stats(rG, v0, pose[2]))),
                grad=dict(n_starts=len(starts), steps=opts.steps, steps_run=int(R['steps_run']), lr_a=opts.lr_a, lr_dv=opts.lr_dv, keep=opts.keep,
                          patience=opts.patience, clip_grad=opts.clip_grad, starts_reproduced=bool(all(s['start_reproduced'] for s in starts)),
                          keep_best_ok=bool((R['J_keep'] <= R['J0'] + 1e-6).all()), trace_best_J_min=[float(x) for x in R['trace']], rows=finals),
                seconds=dict(cem=t_cem, refine=t_ref, finals=t_fin, total=time.perf_counter() - t0))
    return summ, rB, rG


# ------------------------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------------------------
def route_file(r, rid, g, arm, label, world, extra=None):
    out = PA.route_json(r, rid, g, arm, label, world)
    if extra:
        out['meta'].update(extra)
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', help='dir with <group>.json and routes/<group>/route_00.json')
    ap.add_argument('--map-root', help='root with static_map_v1 (the OptiX static depth map; both worlds)')
    ap.add_argument('--models', help='glob of ga_train or ci_train (or legacy N2) checkpoints')
    ap.add_argument('--out'); ap.add_argument('--world', choices=['crm', 'rigid'])
    ap.add_argument('--domain', choices=['crm', 'rigid'], help='tag one-hot for cond=tag members (default: --world)')
    ap.add_argument('--poses', help="ga_planner --poses json {group: {pose, history, frame, ...}}; plans those groups only")
    ap.add_argument('--task-root', help='tasks.json paths relative to this (default: two levels above --cases)')
    ap.add_argument('--groups', help='comma list of group ids or @file'); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--frame', type=int, default=60, help='as ga_planner (only for run/trajectory pose entries)')
    ap.add_argument('--steps', type=int, default=60); ap.add_argument('--starts', type=int, default=17)
    ap.add_argument('--lr', default='0.02,0.10', help='Adam learning rates: lateral amplitudes,speed knots')
    ap.add_argument('--keep', choices=['pessimistic', 'mean'], default='pessimistic', help='keep-best and final-choice criterion')
    ap.add_argument('--abstain-logit', type=float, default=0.3, help='G = B unless the refined route is better by this much (keep criterion)')
    ap.add_argument('--patience', type=int, default=15); ap.add_argument('--clip-grad', type=float, default=10.0)
    ap.add_argument('--ref-b-picks', help='ci_planner / ga_planner picks dir: check that arm B equals its arm B (route sha256)')
    ap.add_argument('--arena-tag', default='f104'); ap.add_argument('--shards', type=int, default=6)
    ap.add_argument('--device'); ap.add_argument('--selftest', help='run self-tests T1-T6 into this directory and exit')
    return ap


def main(argv=None):
    opts = build_parser().parse_args(argv)
    if opts.selftest:
        return selftest(Path(opts.selftest))
    for k in ('cases', 'map_root', 'models', 'out', 'world'):
        assert getattr(opts, k), f'--{k.replace("_", "-")} is required'
    opts.lr_a, opts.lr_dv = [float(x) for x in opts.lr.split(',')]
    t_start = time.time()
    DS.init_map(opts.map_root)
    domain = opts.domain or opts.world
    ens = CP.CIEnsemble(opts.models, opts.device)
    print(f'ci_grad: {len(ens.members)} members from {opts.models} on {ens.dev}: kinds {ens.kinds} conds {ens.conds} hist_T {ens.hist_T}; '
          f'steps {opts.steps} starts {opts.starts} lr {opts.lr_a}/{opts.lr_dv} keep {opts.keep} abstain {opts.abstain_logit}', flush=True)
    dens = DiffEnsemble(ens); tmap = DetMap(DS.G, ens.dev, torch.float32)
    deployed_tag = 'crm_proposal' if opts.world == 'crm' else 'gen_night2'
    specB = PA.arm_specs(deployed_tag, '')['B']
    out = Path(opts.out); (out / 'routes').mkdir(parents=True, exist_ok=True); (out / 'picks').mkdir(exist_ok=True)
    cases = sorted(str(p) for p in Path(opts.cases).glob('*.json') if p.name != 'cases.json')
    poses = json.load(open(opts.poses)) if opts.poses else None
    want = GA.parse_groups(opts.groups)
    if poses:
        want = set(poses) if want is None else want & set(poses)
    if want is not None:
        cases = [c for c in cases if Path(c).stem in want]
    if opts.limit:
        cases = cases[:opts.limit]
    task_root = Path(opts.task_root).resolve() if opts.task_root else Path(opts.cases).resolve().parents[1]
    ref_dir = Path(opts.ref_b_picks) if opts.ref_b_picks else None
    if str(ens.dev).startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    tasks, rows, ref = [], [], dict(checked=0, match=0, z_equal=0, mismatch=[])
    for gi, cp in enumerate(cases):
        g = Path(cp).stem
        summ, rB, rG = plan_group(g, cp, poses, ens, dens, tmap, opts, specB, domain)
        eB, eG = summ['arms']['B'], summ['arms']['G']
        label = eG['label']
        gx = dict(abstained=eG['abstained'], gain_logit=eG['gain'], keep=opts.keep, start_src=eG['start_src'], start_kind=eG['start_kind'],
                  best_step=eG['best_step'], grad_theta=eG['theta'], z_mean=eG['z_mean'], z_pess=eG['z_pess'])
        if eG['same_route_as_B']:
            json.dump(route_file(rB, f'{g}__G', g, 'G', label, opts.world, dict(gx, same_as='B', b_label=specB['label'])), open(out / 'routes' / f'{g}__G.json', 'w'))
        else:
            json.dump(route_file(rG, f'{g}__G', g, 'G', label, opts.world, dict(gx, kind='grad')), open(out / 'routes' / f'{g}__G.json', 'w'))
            json.dump(PA.route_json(rB, f'{g}__B', g, 'B', specB['label'], opts.world), open(out / 'routes' / f'{g}__B.json', 'w'))
        rid = f'{g}__G'
        row = dict(id=rid, group=g, case=os.path.relpath(Path(cp).resolve(), task_root), route=os.path.relpath((out / 'routes' / f'{rid}.json').resolve(), task_root),
                   run=True, tier=gi, episode_seed=int(hashlib.md5(rid.encode()).hexdigest()[:8], 16), arms=['G'] + (['B'] if eG['same_route_as_B'] else []),
                   sha256=eG['route_sha256'], abstained=eG['abstained'], same_as_B=eG['same_route_as_B'], b_sha256=eB['route_sha256'])
        if opts.world == 'rigid':
            row.update(arena=opts.arena_tag, shard=int(hashlib.md5(g.encode()).hexdigest(), 16) % opts.shards)
        tasks.append(row)
        if ref_dir is not None and (ref_dir / f'{g}.json').exists():
            rb = json.load(open(ref_dir / f'{g}.json'))['arms'].get('B')
            if rb:
                ok = rb['route_sha256'] == eB['route_sha256']; ref['checked'] += 1; ref['match'] += int(ok); ref['z_equal'] += int(rb['z_mean'] == eB['z_mean'])
                eB.update(ref_match=ok, ref_z_mean=rb['z_mean'])
                if not ok: ref['mismatch'].append(dict(group=g, ref_index=rb['index'], index=eB['index'], ref_z=rb['z_mean'], z=eB['z_mean']))
        json.dump(summ, open(out / 'picks' / f'{g}.json', 'w'), indent=1, default=float)
        rows.append(summ)
        s = summ['seconds']
        print(f"  [{gi + 1}/{len(cases)}] {g} B z {eB['z_mean']:.3f} (pess {eB['z_pess']:.3f}) -> G " +
              ('abstain' if eG['abstained'] else f"z {eG['z_mean']:.3f} (pess {eG['z_pess']:.3f}) gain {eG['gain']:.3f} from {eG['start_src']}/{eG['start_kind']} step {eG['best_step']}")
              + f" | valid {eG['n_valid_finals']}/{eG['n_finals']} | {s['total']:.2f}s (cem {s['cem']:.2f} refine {s['refine']:.2f} [{summ['grad']['steps_run']} steps] finals {s['finals']:.2f})", flush=True)
    json.dump(tasks, open(out / 'tasks.json', 'w'), indent=1)
    json.dump([dict(t, run=not t['same_as_B'], **({'ref_arm': 'B', 'ref_id': f"{t['group']}__B"} if t['same_as_B'] else {})) for t in tasks], open(out / 'tasks_new_only.json', 'w'), indent=1)
    lock = hashlib.sha256()
    for p in sorted((out / 'routes').glob('*.json')):
        lock.update(p.name.encode()); lock.update(hashlib.sha256(p.read_bytes()).digest())
    (out / 'PICKS_LOCKED.sha256').write_text(lock.hexdigest() + '  routes/*.json (name + content, sorted)\n')
    summary = summarise(rows, opts, ens, ref, ref_dir, t_start, lock.hexdigest())
    h = summary['history']
    assert h['encode_calls_ga_train'] == h['expected_encode_calls_ga_train'], f'history encoded {h["encode_calls_ga_train"]} times, expected once per member per decision'
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1, default=float)
    pa = summary['per_arm']
    print(f"B z_mean {pa['B']['z_mean_mean']:.3f} P {pa['B']['P_mean']:.4f} | G z_mean {pa['G']['z_mean_mean']:.3f} P {pa['G']['P_mean']:.4f} "
          f"changed {pa['G']['changed']}/{pa['G']['n']} abstained {pa['G']['abstained']} | s/group {summary['seconds_per_group']['total']['mean']:.2f} "
          f"(median {summary['seconds_per_group']['total']['median']:.2f}) | GPU peak {summary.get('gpu_peak_gb')} GB | ref B {ref['match']}/{ref['checked']} | lock {lock.hexdigest()[:16]}", flush=True)
    return summary


def summarise(rows, opts, ens, ref, ref_dir, t_start, lock):
    def col(arm, key):
        return np.array([r['arms'][arm][key] for r in rows], float)
    q = lambda x: [float(v) for v in np.quantile(np.asarray(x, float), [0, .1, .25, .5, .75, .9, 1])] if len(x) else None
    per = {}
    for arm in ('B', 'G'):
        z, zp, P = col(arm, 'z_mean'), col(arm, 'z_pess'), col(arm, 'P')
        per[arm] = dict(n=len(rows), z_mean_mean=float(z.mean()), z_mean_median=float(np.median(z)), z_pess_mean=float(zp.mean()), z_pess_median=float(np.median(zp)),
                        P_mean=float(P.mean()), P_median=float(np.median(P)), P_gt_0p1=int((P > 0.1).sum()), P_gt_0p5=int((P > 0.5).sum()),
                        T_mean=float(col(arm, 'T').mean()), mean_speed=float(col(arm, 'mean_speed').mean()),
                        start_dist_to_pose_max_m=float(col(arm, 'start_dist_to_pose_m').max()))
    G = [r['arms']['G'] for r in rows]
    per['G'].update(changed=int(sum(not e['same_route_as_B'] for e in G)), abstained=int(sum(e['abstained'] for e in G)),
                    gain_quantiles=q([e['gain'] for e in G if e['gain'] is not None]),
                    dz_mean_vs_B=q(col('G', 'z_mean') - col('B', 'z_mean')), dz_pess_vs_B=q(col('G', 'z_pess') - col('B', 'z_pess')),
                    start_src={k: int(sum(e['start_src'] == k for e in G)) for k in ('B', 'round0')},
                    refined_pick=int(sum(bool(e['refined']) for e in G)), validator_pass_rate_mean=float(np.mean([e['validator_pass_rate'] for e in G])),
                    reasons=sorted({x for e in G for x in e['reasons']}), contract_all_valid=bool(all(e['contract']['valid'] for e in G)),
                    end_dist_to_goal_max_m=float(max(e['end_dist_to_goal_m'] for e in G)),
                    frac_members_improved_vs_B_mean=(float(np.mean([e['frac_members_improved_vs_B'] for e in G if 'frac_members_improved_vs_B' in e])) if any('frac_members_improved_vs_B' in e for e in G) else None))
    secs = {k: [r['seconds'][k] for r in rows] for k in ('cem', 'refine', 'finals', 'total')}
    rows_all = [fr for r in rows for fr in r['grad']['rows'] if 'J_keep' in fr]
    out = dict(world=opts.world, domain=opts.domain or opts.world, n_groups=len(rows), arms=['B', 'G'], models=opts.models, members=ens.describe(),
               model_kinds=ens.kinds, conds=ens.conds, map_root=opts.map_root, poses=opts.poses,
               config=dict(steps=opts.steps, starts=opts.starts, lr_a=opts.lr_a, lr_dv=opts.lr_dv, keep=opts.keep, abstain_logit=opts.abstain_logit,
                           patience=opts.patience, clip_grad=opts.clip_grad, family='free', B_tag='n2iter_cem4x64',
                           penalties='1e5*relu(kappa-0.95*0.125)^2 + 10*relu(|xy|-37)^2'),
               per_arm=per, lock=lock,
               checks=dict(starts_reproduced_all=bool(all(r['grad']['starts_reproduced'] for r in rows)),
                           keep_best_ok_all=bool(all(r['grad']['keep_best_ok'] for r in rows)),
                           rows_with_deployed_rescore=len(rows_all),
                           dJ_final_vs_start_deployed=q([fr['dJ_vs_start'] for fr in rows_all]),
                           rows_deployed_worse_than_start_by_0p01=int(sum(fr['dJ_vs_start'] > 0.01 for fr in rows_all)),
                           G_not_worse_than_B=bool(all((e['J_G'] <= e['J_B'] + 1e-12) for e in G))),
               ref_b=dict(dir=str(ref_dir) if ref_dir else None, **ref),
               history=dict(n_encode_per_group=sorted({r['history']['n_encode'] for r in rows}), n_valid_per_group=sorted({r['history']['n_valid'] for r in rows}),
                            encode_calls_ga_train=int(sum(m['model'].encode_calls for m in ens.members if m['kind'] == 'ga_train')),
                            expected_encode_calls_ga_train=int(sum(1 for m in ens.members if m['kind'] == 'ga_train' and m.get('zdim'))) * len(rows)),
               flags={k: int(sum(r['arms']['G']['flags'][k] for r in rows)) for k in rows[0]['arms']['G']['flags']} if rows else None,
               seconds_per_group={k: dict(mean=float(np.mean(v)), median=float(np.median(v)), max=float(np.max(v)),
                                          mean_excl_first=float(np.mean(v[1:])) if len(v) > 1 else float(np.mean(v))) for k, v in secs.items()},
               steps_run=q([r['grad']['steps_run'] for r in rows]), wall_s_total=time.time() - t_start)
    if str(ens.dev).startswith('cuda'):
        out['gpu_peak_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3); out['gpu_reserved_peak_gb'] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)
    return out


# ------------------------------------------------------------------------------------------------------------------
# self-tests (<= 10 suite groups; diagnostics only, nothing is tuned on them)
# ------------------------------------------------------------------------------------------------------------------
ENSEMBLES = {'X': (K2 / 'deploy_v1/deploy_a3_haux_txjoint_s*.pt', K2 / 's2/L1/picks_crm_X/picks'),
             'Hn': (K2 / 'deploy_v1/deploy_a1_haux_gru_s*.pt', K2 / 's2/L1/picks_crm_Hn/picks'),
             'H': (K1 / 'train/deploy_v1/H_deploy_s*.pt', K2 / 's2/L1/picks_crm_H/picks')}
POSES = K2 / 'a5data/decision_suite_L1/poses_crm_all.json'
MAP_ROOT = 'artifacts/traverse/crm_f104_v1/map_root'


def _run_cli(args, log):
    cmd = [sys.executable, str(Path(__file__).resolve())] + [str(x) for x in args]
    env = dict(os.environ, PYTHONPATH='src:scripts', OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '6'))
    with open(log, 'w') as fh:
        fh.write(' '.join(cmd) + '\n'); fh.flush()
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    assert rc == 0, f'{cmd} failed, see {log}'
    return ' '.join(cmd)


def selftest_groups():
    return sorted(json.load(open(POSES)))[::80][:10]


def fd_check(ch, A, D, h):
    """Autograd vs central differences of the ensemble-mean logit w.r.t. (a, dv) for every row (float64 chain)."""
    dev = ch.dens.dev
    At = torch.as_tensor(A, device=dev, dtype=torch.float64).clone().requires_grad_(True); Dt = torch.as_tensor(D, device=dev, dtype=torch.float64).clone().requires_grad_(True)
    z = ch(At, Dt)[5].mean(0)
    with det_ctx(torch.float64):
        z.sum().backward()
    ga, gd = At.grad.clone(), Dt.grad.clone()
    fa, fd = torch.zeros_like(ga), torch.zeros_like(gd)
    with torch.no_grad():
        for j in range(IT.MODES):
            E = torch.zeros_like(At); E[:, j] = h; fa[:, j] = (ch(At + E, Dt)[5].mean(0) - ch(At - E, Dt)[5].mean(0)) / (2 * h)
        for j in range(IT.KNOTS):
            E = torch.zeros_like(Dt); E[:, j] = h; fd[:, j] = (ch(At, Dt + E)[5].mean(0) - ch(At, Dt - E)[5].mean(0)) / (2 * h)
    G_ = torch.cat([ga, gd], 1).cpu().numpy(); FD = torch.cat([fa, fd], 1).cpu().numpy()
    rel = np.abs(G_ - FD) / (np.abs(FD) + 1e-6)
    cos = [float(np.dot(G_[i], FD[i]) / max(np.linalg.norm(G_[i]) * np.linalg.norm(FD[i]), 1e-30)) for i in range(len(G_))]
    return dict(rel_err_median=float(np.median(rel)), rel_err_p90=float(np.quantile(rel, 0.9)), rel_err_max=float(rel.max()),
                frac_rel_below_1e3=float((rel < 1e-3).mean()), frac_rel_below_1e2=float((rel < 1e-2).mean()), cos_per_row=cos,
                rel_norm_err_per_row=[float(np.linalg.norm(G_[i] - FD[i]) / max(np.linalg.norm(FD[i]), 1e-30)) for i in range(len(G_))],
                grad_norm_per_row=[float(np.linalg.norm(x)) for x in G_], grad_finite=bool(np.isfinite(G_).all())), G_


def t1_chain(out, ens_key, groups, n_rows=4, h=1e-5):
    """T1: corridor chain vs numpy, torch logits vs the deployed scorer, autograd vs central differences (float64) at the
    start parameters (anchors sit on kinks: speed clamp at 6 m/s, zero amplitudes) and at a generic nearby point."""
    pattern, _ = ENSEMBLES[ens_key]
    DS.init_map(str(ROOT / MAP_ROOT))
    ens = CP.CIEnsemble(str(pattern)); poses = json.load(open(POSES))
    d32 = DiffEnsemble(ens, torch.float32); d64 = DiffEnsemble(ens, torch.float64)
    tm32 = DetMap(DS.G, ens.dev, torch.float32); tm64 = DetMap(DS.G, ens.dev, torch.float64)
    ng32 = NG.TMap(DS.G, ens.dev, torch.float32); ng64 = NG.TMap(DS.G, ens.dev, torch.float64)
    specB = PA.arm_specs('crm_proposal', '')['B']; R = []
    for g in groups:
        a = argparse.Namespace(from_run=None, frame=60, pose_along_s=None, pose_override=None, goal=None, history=None)
        case, lay, pose, goal, base, hist, hmask, src = GA.decision_for(g, str(K1 / 'suite/cases' / f'{g}.json'), a, poses, ens.hist_T)
        sc = CP.CIScorer(ens, pose[:2], goal, float(pose[2]), 'crm', hist, hmask)
        res = CP.plan_iter_family(base, pose, goal, sc, CP.Family('free'), rounds=4, n=64, rng=np.random.default_rng(IT.seed(g, specB['tag'])), keep_candidates=True)
        n0 = int(res['log'][0]['n'])
        # rows: the B pick + the best round-0 anchor + the best round-0 samples
        order0 = [int(i) for i in np.argsort(res['Z_mean'][:n0], kind='stable') if int(i) != res['index']]
        anc = [i for i in order0 if res['kinds'][i] == 'anchor'][:1]; smp = [i for i in order0 if res['kinds'][i] == 'sample'][:n_rows - 2]
        rows = [(res['route'], res['kind'])] + [(res['candidates'][i], res['kinds'][i]) for i in anc + smp]
        xy, station, f, L = IT._base_arrays(base)
        P_ = [start_params(r, k) for r, k in rows]
        bsp = np.stack([np.asarray(base['speeds'], float) if p[2] is None else np.full(len(xy), p[2][1]) for p in P_])
        lat0 = np.stack([np.zeros(len(xy)) if p[2] is None else p[2][0] * np.sin(np.pi * f) ** 2 for p in P_])
        A0 = np.stack([p[0] for p in P_]); D0 = np.stack([p[1] for p in P_])
        e = dict(group=g, kinds=[k for _, k in rows])
        for dens, tm, tng, nm in ((d32, tm32, ng32, 'f32'), (d64, tm64, ng64, 'f64')):
            dens.set_context(sc)
            ch = Chain(base, pose, goal, tm, dens, bsp, lat0)
            chn = Chain(base, pose, goal, tng, dens, bsp, lat0, det=False)      # f104_n2_grad's own chain (grid_sample)
            with torch.no_grad():
                pts, v, st, X, Lt, Z = ch(torch.as_tensor(A0, device=ens.dev).to(dens.dtype), torch.as_tensor(D0, device=ens.dev).to(dens.dtype))
            wp_err = max(float(np.abs(pts[i].double().cpu().numpy() - np.asarray(r['waypoints'])).max()) for i, (r, _) in enumerate(rows))
            sp_err = max(float(np.abs(v[i].double().cpu().numpy() - np.asarray(r['speeds'])).max()) for i, (r, _) in enumerate(rows))
            Xn, Ln = IT.corridors([r for r, _ in rows])
            Xt = X.double().cpu().numpy()
            e[f'{nm}_shape_waypoints_max_abs'] = wp_err; e[f'{nm}_shape_speeds_max_abs'] = sp_err
            e[f'{nm}_corridor_max_abs'] = {c: float(np.abs(Xt[:, j] - Xn[:, j]).max()) for j, c in enumerate(['elev_rel', 'grade', 'cross', 'speed', 'valid'])}
            e[f'{nm}_route_len_max_abs'] = float(np.abs(Lt.double().cpu().numpy() - Ln).max())
            # torch member logits on the SAME float16-rounded numpy corridors (the whole 64-route round-0 pool, one batch)
            # vs the deployed scorer; and the chain's own corridor (no float16 rounding) vs the scorer on the 4 rows
            Xp, Lp = IT.corridors(res['candidates'][:n0]); X16 = Xp.astype(np.float16).astype(np.float32)
            geom = GP.geom_ctx(pose[:2], goal, float(pose[2]), Lp)
            Zs = sc.member_logits(X16, geom)
            with torch.no_grad():
                Zt = dens.logits(torch.as_tensor(X16, device=ens.dev).to(dens.dtype), torch.as_tensor(geom, device=ens.dev).to(dens.dtype)).double().cpu().numpy()
            e[f'{nm}_logits_vs_scorer_same_input_max_abs'] = float(np.abs(Zt - Zs).max()); e['n_pool_rows'] = int(n0)
            Zs4 = sc.member_logits(Xn.astype(np.float16).astype(np.float32), GP.geom_ctx(pose[:2], goal, float(pose[2]), Ln))
            e[f'{nm}_logits_chain_vs_scorer_max_abs'] = float(np.abs(Z.double().cpu().numpy().mean(0) - Zs4.mean(0)).max())
            if nm == 'f32':      # the deployed scorer's own batch-shape noise (cuDNN / TF32 algorithm choice): 4-row batches vs one 64-row batch
                Zb = np.concatenate([sc.member_logits(X16[i:i + 4], geom[i:i + 4]) for i in range(0, n0, 4)], 1)
                e['scorer_batch4_vs_batch64_max_abs'] = float(np.abs(Zb - Zs).max()); e['scorer_batch4_vs_batch64_mean_max_abs'] = float(np.abs(Zb.mean(0) - Zs.mean(0)).max())
            rng = np.random.default_rng(0)       # a generic point near the starts (off the kinks), inside the caps
            cap = IT.caps(L)
            Ap = np.clip(A0 + 0.1 * rng.standard_normal(A0.shape), -cap, cap); Dp = np.clip(D0 - 0.3 + 0.1 * rng.standard_normal(D0.shape), -IT.SP_CLIP, IT.SP_CLIP)
            # the deterministic chain vs f104_n2_grad's chain: corridor and gradient at the generic point
            At = torch.as_tensor(Ap, device=ens.dev).to(dens.dtype).requires_grad_(True); Dt = torch.as_tensor(Dp, device=ens.dev).to(dens.dtype).requires_grad_(True)
            o1 = ch(At, Dt)
            with det_ctx(dens.dtype):
                o1[5].mean(0).sum().backward()
            gd1 = torch.cat([At.grad, Dt.grad], 1).double().cpu().numpy()
            At2 = At.detach().clone().requires_grad_(True); Dt2 = Dt.detach().clone().requires_grad_(True)
            o2 = chn(At2, Dt2)
            with det_ctx(dens.dtype):
                o2[5].mean(0).sum().backward()
            gd2 = torch.cat([At2.grad, Dt2.grad], 1).double().cpu().numpy()
            e[f'{nm}_det_vs_n2_corridor_max_abs'] = float((o1[3] - o2[3]).detach().abs().max()); e[f'{nm}_det_vs_n2_logit_max_abs'] = float((o1[5] - o2[5]).detach().abs().max())
            e[f'{nm}_det_vs_n2_grad_rel_norm_err_max'] = float(max(np.linalg.norm(gd1[i] - gd2[i]) / max(np.linalg.norm(gd2[i]), 1e-30) for i in range(len(gd1))))
            if nm == 'f64':      # autograd vs central differences of the ensemble-mean logit
                e['fd_start'], g64s = fd_check(ch, A0, D0, h)
                e['fd_perturbed'], g64p = fd_check(ch, Ap, Dp, h)
            else:
                grads = []
                for A_, D_ in ((A0, D0), (Ap, Dp)):
                    At = torch.as_tensor(A_, device=ens.dev, dtype=torch.float32).clone().requires_grad_(True); Dt = torch.as_tensor(D_, device=ens.dev, dtype=torch.float32).clone().requires_grad_(True)
                    z = ch(At, Dt)[5].mean(0)
                    with det_ctx(torch.float32):
                        z.sum().backward()
                    grads.append(torch.cat([At.grad, Dt.grad], 1).double().cpu().numpy())
                g32s, g32p = grads
        cosr = lambda x, y: [float(np.dot(x[i], y[i]) / max(np.linalg.norm(x[i]) * np.linalg.norm(y[i]), 1e-30)) for i in range(len(y))]
        e['grad_f32_vs_f64_cos_start'] = cosr(g32s, g64s); e['grad_f32_vs_f64_cos_perturbed'] = cosr(g32p, g64p)
        R.append(e)
    return R


def selftest(out):
    out.mkdir(parents=True, exist_ok=True)
    groups = selftest_groups()
    Rs = dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), groups=groups, poses=str(POSES), note='<= 10 suite groups, diagnostics only (no tuning)')
    common = ['--cases', K1 / 'suite/cases', '--map-root', MAP_ROOT, '--world', 'crm', '--domain', 'crm', '--poses', POSES, '--task-root', K2, '--groups', ','.join(groups)]
    # T1 (in process): chain and gradients, 2 groups x 4 rows per ensemble
    Rs['T1'] = {}
    for k in ENSEMBLES:
        r = t1_chain(out, k, groups[:2])
        Rs['T1'][k] = r
        print('T1', k, json.dumps([dict(group=x['group'], kinds=x['kinds'], fd_start_cos=x['fd_start']['cos_per_row'], fd_pert_cos=x['fd_perturbed']['cos_per_row'],
                                        fd_pert_rel_median=x['fd_perturbed']['rel_err_median'], f32_logits_same_input=x['f32_logits_vs_scorer_same_input_max_abs'],
                                        f64_logits_same_input=x['f64_logits_vs_scorer_same_input_max_abs'], f64_corridor=x['f64_corridor_max_abs'],
                                        f64_shape_wp=x['f64_shape_waypoints_max_abs'], cos32_64=x['grad_f32_vs_f64_cos_perturbed']) for x in r], default=float), flush=True)
    T1 = [x for k in ENSEMBLES for x in Rs['T1'][k]]
    # pass: gradients match central differences at the generic points (cosine > 0.999 per row, median component error < 1e-3);
    # the torch networks reproduce the deployed scorer on identical corridors (float32 < 1e-3; float64 differs by the
    # scorer's TF32 convolutions, < 0.05); float64 route / corridor mirrors exact to 1e-5
    Rs['T1_passed'] = bool(all(min(x['fd_perturbed']['cos_per_row']) > 0.999 and x['fd_perturbed']['rel_err_median'] < 1e-3 and x['fd_perturbed']['grad_finite']
                               and x['f32_logits_vs_scorer_same_input_max_abs'] < 1e-3 and x['f64_logits_vs_scorer_same_input_max_abs'] < 0.05
                               and x['f64_shape_waypoints_max_abs'] < 1e-9 and x['f64_shape_speeds_max_abs'] < 1e-5
                               and max(x['f64_corridor_max_abs'][c] for c in ('elev_rel', 'grade', 'cross', 'speed')) < 1e-5
                               and min(x['grad_f32_vs_f64_cos_perturbed']) > 0.99 for x in T1))
    # T2: --steps 0 -> no row moves; with --keep mean G == B (B is the argmin of the ensemble-mean logit over every CEM
    # candidate, the starts included), B == ci_planner's own B (existing S2 L1 pick sets; plan_decision in process in T2b).
    # T2p: the same with --keep pessimistic (information: at 0 steps G can still differ from B when a round-0 start has a
    # pessimistic logit lower than B's by >= 0.3, because B was chosen by the mean)
    Rs['T2'] = {}; Rs['T2p'] = {}; Rs['T5'] = {}; Rs['T3'] = {}; Rs['T4'] = {}
    for k, (pattern, refdir) in ENSEMBLES.items():
        for tag, keep in (('T2', 'mean'), ('T2p', 'pessimistic')):
            d0 = out / f"{tag.lower()}_steps0_{k}"
            cmd = _run_cli(['--models', pattern, '--steps', '0', '--keep', keep, '--out', d0, '--ref-b-picks', refdir] + common, out / f"{tag.lower()}_steps0_{k}.log")
            rows = []
            for g in groups:
                p = json.load(open(d0 / 'picks' / f'{g}.json')); ref = json.load(open(refdir / f'{g}.json'))['arms']['B']
                rB = json.load(open(d0 / 'routes' / f"{g}__{'G' if p['arms']['G']['same_route_as_B'] else 'B'}.json"))
                rows.append(dict(group=g, sha_B=p['arms']['B']['route_sha256'], sha_ref=ref['route_sha256'], sha_G=p['arms']['G']['route_sha256'],
                                 z_B=p['arms']['B']['z_mean'], z_ref=ref['z_mean'], G_abstained=p['arms']['G']['abstained'], starts_reproduced=p['grad']['starts_reproduced'],
                                 n_starts=p['grad']['n_starts'], gain=p['arms']['G']['gain'], refined_rows=p['arms']['G']['n_refined_rows'],
                                 B_route_file_arrays_equal_ref=all(rB[kk] == json.load(open(refdir.parent / 'routes' / f'{g}__B.json'))[kk] for kk in ('waypoints', 'speeds', 'stations', 'headings'))))
            ok_b = all(r['sha_B'] == r['sha_ref'] and r['z_B'] == r['z_ref'] and r['starts_reproduced'] and r['refined_rows'] == 0 and r['B_route_file_arrays_equal_ref'] for r in rows)
            Rs[tag][k] = dict(cmd=cmd, rows=rows, n_G_equal_B=int(sum(r['sha_G'] == r['sha_B'] for r in rows)), B_equal_ref=ok_b,
                              passed=bool(ok_b and (tag == 'T2p' or all(r['sha_G'] == r['sha_B'] and r['G_abstained'] for r in rows))))
            print(tag, k, Rs[tag][k]['passed'], 'G == B in', Rs[tag][k]['n_G_equal_B'], 'of', len(rows), flush=True)
    # T2b: ci_planner.plan_decision in process (the planner's own API) for the same seeds and decision states
    DS.init_map(str(ROOT / MAP_ROOT)); poses = json.load(open(POSES)); Rs['T2b'] = {}
    for k, (pattern, refdir) in ENSEMBLES.items():
        ens = CP.CIEnsemble(str(pattern)); rows = []
        for g in groups:
            a = argparse.Namespace(from_run=None, frame=60, pose_along_s=None, pose_override=None, goal=None, history=None)
            case, lay, pose, goal, base, hist, hmask, src = GA.decision_for(g, str(K1 / 'suite/cases' / f'{g}.json'), a, poses, ens.hist_T)
            v0, _, _ = CP.decision_v0(poses[g], hist, hmask, src)
            r = CP.plan_decision(ens, pose, goal, v0, hist, hmask, group=g, arm='B', family='free', world='crm', domain='crm')
            mine = json.load(open(out / f't2_steps0_{k}' / 'picks' / f'{g}.json'))['arms']['B']
            rows.append(dict(group=g, equal_sha=r['route_sha256'] == mine['route_sha256'], equal_z=r['z_mean'] == mine['z_mean']))
        Rs['T2b'][k] = dict(rows=rows, passed=all(x['equal_sha'] and x['equal_z'] for x in rows))
        print('T2b', k, Rs['T2b'][k]['passed'], flush=True)
        del ens; torch.cuda.empty_cache()
    # T3-T5: --steps 60 (defaults): keep-best, validity / contract, seconds per group
    for k, (pattern, refdir) in ENSEMBLES.items():
        d = out / f't5_steps60_{k}'
        cmd = _run_cli(['--models', pattern, '--out', d, '--ref-b-picks', refdir] + common, out / f't5_steps60_{k}.log')
        sm = json.load(open(d / 'summary.json')); picks = [json.load(open(d / 'picks' / f'{g}.json')) for g in groups]
        rows_all = [fr for p in picks for fr in p['grad']['rows']]
        t3 = dict(keep_best_torch_all=all(p['grad']['keep_best_ok'] for p in picks),
                  torch_J_kept_minus_start_max=float(max(fr['J_torch'] - fr['J0_torch'] for fr in rows_all)),
                  deployed_dJ_final_vs_start=sm['checks']['dJ_final_vs_start_deployed'], rows_deployed_worse_by_0p01=sm['checks']['rows_deployed_worse_than_start_by_0p01'],
                  G_vs_B=[dict(group=p['group'], J_B=p['arms']['G']['J_B'], J_G=p['arms']['G']['J_G'], gain=p['arms']['G']['gain'], abstained=p['arms']['G']['abstained'],
                               z_mean_B=p['arms']['B']['z_mean'], z_mean_G=p['arms']['G']['z_mean'], z_pess_B=p['arms']['B']['z_pess'], z_pess_G=p['arms']['G']['z_pess'],
                               P_B=p['arms']['B']['P'], P_G=p['arms']['G']['P'], start=p['arms']['G']['start_src'], kind=p['arms']['G']['start_kind'],
                               best_step=p['arms']['G']['best_step'], steps_run=p['grad']['steps_run'], frac_members_improved=p['arms']['G'].get('frac_members_improved_vs_B'))
                          for p in picks])
        t3['deployed_dJ_max'] = float(max(fr['dJ_vs_start'] for fr in rows_all if 'dJ_vs_start' in fr))
        # pass: the torch keep-best never ends above its start; the deployed re-score (float16 corridor, TF32 convolutions)
        # of every kept final is within 0.02 of its start's re-score in the same batch; G is never worse than B
        t3['passed'] = bool(t3['keep_best_torch_all'] and t3['torch_J_kept_minus_start_max'] <= 1e-6 and t3['deployed_dJ_max'] <= 0.02
                            and all(x['J_G'] <= x['J_B'] for x in t3['G_vs_B']))
        t4rows = []
        for p in picks:
            g = p['group']; rG = json.load(open(d / 'routes' / f'{g}__G.json'))
            ok, info = route_contract(rG, np.asarray(p['pose']), np.asarray(p['goal']))
            fa = p['family']['arms']['G']
            t4rows.append(dict(group=g, valid=ok, reasons=info['reasons'], start_dist_m=info['start_dist_m'], end_dist_m=info['end_dist_m'], v_max=info['v_max'],
                               accel_max=info['accel_max'], accel_min=info['accel_min'], max_curvature=info['max_curvature'], v0=p['family']['v0_mps'],
                               v0_source=p['family']['v0_source'], v0_check=p['family']['v0_check'], start_speed=fa['start_speed_mps'], speed_step=fa['speed_step_mps'],
                               start_heading_err_deg=fa['start_heading_err_deg'], sha_file_eq_pick=IT.route_sha256({kk: np.asarray(rG[kk]) for kk in ('waypoints', 'speeds', 'stations', 'headings')}) == p['arms']['G']['route_sha256']))
        t4 = dict(rows=t4rows, passed=all(r['valid'] and r['start_dist_m'] <= START_TOL_M and r['sha_file_eq_pick'] and r['v0_source'] == 'history_window'
                                          and r['v0_check'].get('abs_diff', 0.0) == 0.0 for r in t4rows),
                  validator_pass_rate_all_finals=float(np.mean([fr['valid'] for fr in rows_all])), reasons_finals=sorted({x for fr in rows_all for x in fr['reasons']}))
        t5 = dict(cmd=cmd, seconds_per_group=sm['seconds_per_group'], steps_run=sm['steps_run'], gpu_peak_gb=sm.get('gpu_peak_gb'), gpu_reserved_peak_gb=sm.get('gpu_reserved_peak_gb'),
                  ref_b=sm['ref_b'], per_arm=sm['per_arm'], passed=bool(sm['seconds_per_group']['total']['mean'] < 5.0 and (sm.get('gpu_reserved_peak_gb') or 0) < 8.0))
        Rs['T3'][k], Rs['T4'][k], Rs['T5'][k] = t3, t4, t5
        print('T3', k, t3['passed'], 'T4', k, t4['passed'], 'T5', k, json.dumps(dict(s=sm['seconds_per_group']['total'], gpu=sm.get('gpu_reserved_peak_gb'),
                                                                                          changed=sm['per_arm']['G']['changed'], abstained=sm['per_arm']['G']['abstained'])), flush=True)
    # T6: determinism - the first 2 groups replanned with the same command give the same G (route sha256)
    Rs['T6'] = {}
    for k, (pattern, refdir) in ENSEMBLES.items():
        d = out / f't6_repeat_{k}'
        _run_cli(['--models', pattern, '--out', d] + common[:-1] + [','.join(groups[:3])], out / f't6_repeat_{k}.log')
        rows = [dict(group=g, same=json.load(open(d / 'picks' / f'{g}.json'))['arms']['G']['route_sha256'] == json.load(open(out / f't5_steps60_{k}' / 'picks' / f'{g}.json'))['arms']['G']['route_sha256'])
                for g in groups[:3]]
        Rs['T6'][k] = dict(rows=rows, passed=all(r['same'] for r in rows))
        print('T6', k, Rs['T6'][k]['passed'], flush=True)
    Rs['passed'] = dict(T1=Rs['T1_passed'], **{f'{t}_{k}': Rs[t][k]['passed'] for t in ('T2', 'T2p', 'T2b', 'T3', 'T4', 'T5', 'T6') for k in ENSEMBLES})
    Rs['all_passed'] = bool(all(Rs['passed'].values()))
    Rs['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(Rs, open(out / 'RESULTS.json', 'w'), indent=1, default=float)
    print('ALL PASSED' if Rs['all_passed'] else f"SOME FAILED {Rs['passed']}", flush=True)
    return Rs


if __name__ == '__main__':
    main()
