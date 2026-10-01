"""Gradient refinement of the sampling-search pick: the 'cem_grad' planner (ci_grad.plan_group, ci_grad.py:400-514).

Starts = the search pick B + the 16 lowest-mean round-0 routes (anchors and prior draws; B's own pool index excluded,
ci_grad.py:417). Each start's parameters (theta rows; an anchor is its offset * sin^2(pi f) with a constant cruise
profile and a = dv = 0) are re-shaped in float64 and must reproduce its route. All starts run ONE batch of Adam steps
through a float32 torch copy of the corridor chain (DetMap sampling, one-hot takes, min-plus speed passes: no atomic
adds in the backward pass) and of the ensemble, the decision's history code held fixed, under deterministic cuDNN
(TF32 allowed) scoped to this step. Fit = ensemble-mean logit + 1e5 sum relu(kappa - 0.95 kappa_max)^2 +
10 sum relu(|x|, |y| - 37)^2; keep-best per row by the WORST member + penalty (1e-3 hysteresis); per-row gradient-norm
clip 10, a clamped to its caps and dv to +-4 after every step; early stop when no row improved for 15 steps.
Finals are re-shaped in float64 (a row whose best iterate is its start keeps its start route), validated and
contract-checked, then re-scored by the deployed scorer in ONE batch [starts (row 0 = B) + valid finals]. G = the best
valid final by the worst member if it beats the re-scored B by >= 0.3 logit, else B (abstained; z from the search).
Only ci_train ensembles and the free family (every headline gradient arm). PARITY: the torch code keeps ci_grad's
op creation order statement for statement; autograd sums the gradients of a shared tensor in node order, so reordering
(e.g. the curvature slices or e_mid / e_mean) changes bits (a reordered port matched only 22 of 60 released picks).
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from nedm.traversing.training.risk_model import route_logit

from .planner import Pick, Result
from .routes import (A_ACC, A_DEC, KAPPA_MAX, KNOTS, LAT_CLIP, MODES, N_LATERAL, N_STATION, SP_CLIP, V_MAX, V_MIN,
                     CORRIDOR_HALF_M, _base_arrays, _shape, _speed_knots, caps, ends_within, from_params, route_sha256,
                     validate)

GRAD = dict(starts=17, steps=60, lr_a=0.02, lr_dv=0.10, betas=(0.9, 0.99), clip=10.0, patience=15, keep='pessimistic',
            abstain=0.3, arena_soft=37.0)               # ci_grad.py:400-514; recorded in every gradient pick


def _det():
    """cuDNN flags of every forward and backward pass of the chain (ci_grad.det_ctx, ci_grad.py:77-86; its
    math-attention scope is omitted: the CNN-GRU has no attention). Global flags stay the defaults the search
    asserts."""
    return torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=True)


# -------------------------------------------------------------------------------------------- differentiable chain
class DetMap:
    """Bilinear elevation lookup with f104_n2_dataset.sample_map's arithmetic; map values are constants taken at
    detached integer indices, so gradients reach the coordinates through the fractions only (ci_grad.DetMap)."""

    def __init__(self, smap, device):
        z = torch.as_tensor(np.array(smap.rgbd[3], np.float32), device=device)
        self.n, self.mpp, self.ctr, self.es = smap.rgbd.shape[1], float(smap.mpp), float(smap.ctr), smap.elev_scale
        self.Z, self.V = z.reshape(-1).contiguous(), (z > -1.999).reshape(-1).contiguous()

    def sample(self, x, y):
        n = self.n
        row, col = self.ctr - y / self.mpp, self.ctr + x / self.mpp
        r0, c0 = torch.floor(row.detach()).long().clamp(0, n - 2), torch.floor(col.detach()).long().clamp(0, n - 2)
        fr, fc = (row - r0.to(row.dtype)).clamp(0, 1), (col - c0.to(col.dtype)).clamp(0, 1)
        i = r0 * n + c0
        p00, p01, p10, p11 = self.Z[i], self.Z[i + 1], self.Z[i + n], self.Z[i + n + 1]
        valid = self.V[i] & self.V[i + 1] & self.V[i + n] & self.V[i + n + 1]
        return p00 * (1 - fr) * (1 - fc) + p01 * (1 - fr) * fc + p10 * fr * (1 - fc) + p11 * fr * fc, valid


def npgrad(f, h, dim):
    """np.gradient along dim (f104_n2_grad.npgrad)."""
    f = f.movedim(dim, 0)
    return torch.cat([f[1:2] - f[0:1], (f[2:] - f[:-2]) / 2, f[-1:] - f[-2:-1]], 0).movedim(0, dim) / h


def _take(v, idx):
    """v (B, N) at idx (B, Q) as a one-hot product (no scatter_add in the backward)."""
    return (F.one_hot(idx, v.shape[1]).to(v.dtype) * v[:, None, :]).sum(-1)


def _interp(xq, xp, fp):
    """np.interp batched (ci_grad.tinterp_det)."""
    idx = torch.searchsorted(xp.detach().contiguous(), xq.detach().contiguous(), right=True) - 1
    idx = idx.clamp(0, xp.shape[1] - 2)
    x0, x1 = _take(xp, idx), _take(xp, idx + 1)
    w = ((xq - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0, 1)
    return _take(fp, idx) * (1 - w) + _take(fp, idx + 1) * w


def _knots(f, vals):
    """Free-end smoothstep speed knots (ci_grad.t_speed_knots_det)."""
    k = vals.shape[1]; kx = torch.linspace(0, 1, k, device=f.device, dtype=f.dtype)
    seg = (torch.searchsorted(kx, f.contiguous(), right=True) - 1).clamp(0, k - 2)
    u = (f - kx[seg]) / (kx[seg + 1] - kx[seg]); w = u * u * (3 - 2 * u)
    v0, v1 = vals @ F.one_hot(seg, k).to(vals.dtype).T, vals @ F.one_hot(seg + 1, k).to(vals.dtype).T
    return v0 + (v1 - v0) * w[None]


def _minplus(u2, s, acc, forward):
    """The sequential acceleration pass in min-plus form (f104_n2_grad._minplus)."""
    N, ds = u2.shape[1], s[:, :, None] - s[:, None, :]
    if forward:
        cand, mask = u2[:, None, :] + 2 * acc * ds, torch.ones(N, N, dtype=torch.bool, device=u2.device).tril()
    else:
        cand, mask = u2[:, None, :] - 2 * acc * ds, torch.ones(N, N, dtype=torch.bool, device=u2.device).triu()
    return torch.where(mask[None], cand, torch.full_like(cand, float('inf'))).amin(2)


def _shape_rows(bxy, bst, bsp, a, dvk, lat0):
    """routes._shape per row (ci_grad.t_shape_rows): lat = lat0 + sum_j a_j sin(j pi f) clipped +-10 m, speeds
    clip(bsp + knots(dv), 0.5, 6) under the terminal cone and the forward / backward passes."""
    B = a.shape[0]; f = ((bst - bst[0]) / (bst[-1] - bst[0])).clamp(0, 1)
    lat = sum(a[:, j:j + 1] * torch.sin((j + 1) * math.pi * f)[None] for j in range(a.shape[1]))
    lat = (lat + lat0).clamp(-LAT_CLIP, LAT_CLIP)
    t = npgrad(bxy, 1.0, 0); t = t / torch.linalg.norm(t, dim=1, keepdim=True).clamp_min(1e-9)
    pts = bxy[None] + lat[..., None] * torch.stack([-t[:, 1], t[:, 0]], 1)[None]
    seg = torch.linalg.norm(pts[:, 1:] - pts[:, :-1], dim=-1); st = torch.cat([seg.new_zeros(B, 1), seg.cumsum(1)], 1)
    u = (bsp + _knots(f, dvk)).clamp(V_MIN, V_MAX)
    u = torch.minimum(u, torch.sqrt((2 * A_DEC * (st[:, -1:] - st)).clamp_min(1e-12)))
    v2 = _minplus(_minplus(u ** 2, st, A_ACC, True), st, A_DEC, False)
    return pts, torch.sqrt(v2.clamp_min(1e-12)), st


def _corridor(wp, sp, M):
    """routes.StaticMap.corridors in torch (ci_grad.t_station_tensor_det): X (B, 5, 96, 32), route length (B,)."""
    NS, NL, HW, B = N_STATION, N_LATERAL, CORRIDOR_HALF_M, wp.shape[0]
    seg = torch.linalg.norm(wp[:, 1:] - wp[:, :-1], dim=-1); s = torch.cat([seg.new_zeros(B, 1), seg.cumsum(1)], 1)
    grid = torch.linspace(0, 1, NS, device=wp.device, dtype=wp.dtype)[None] * s[:, -1:]
    pts = torch.stack([_interp(grid, s, wp[..., 0]), _interp(grid, s, wp[..., 1])], -1)
    d = npgrad(pts, 1.0, 1); tang = d / torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(1e-9)
    nrm = torch.stack([-tang[..., 1], tang[..., 0]], -1)
    off = torch.linspace(-HW, HW, NL, device=wp.device, dtype=wp.dtype)
    z, valid = M.sample(pts[..., 0:1] + nrm[..., 0:1] * off, pts[..., 1:2] + nrm[..., 1:2] * off)
    elev = z * M.es
    row0v, e_mid = valid[:, 0], elev[:, 0, NL // 2]
    e_mean = (elev[:, 0] * row0v).sum(1) / row0v.sum(1).clamp_min(1)
    e0 = torch.where(row0v[:, NL // 2], e_mid, e_mean)
    fill = torch.where(valid, elev, e0[:, None, None])
    ds = (s[:, -1] / (NS - 1)).clamp_min(1e-3)
    ga, gc = npgrad(fill, ds[:, None, None], 1).clamp(-2, 2), npgrad(fill, 2 * HW / (NL - 1), 2).clamp(-2, 2)
    v = _interp(grid, s, sp)
    X = torch.stack([fill - e0[:, None, None], ga, gc, v[..., None].expand(B, NS, NL), valid.to(wp.dtype)], 1)
    return X, s[:, -1]


def _curv(pts):
    """Three-point curvature per interior waypoint (f104_n2_grad.t_curv)."""
    a, b, c = pts[:, :-2], pts[:, 1:-1], pts[:, 2:]
    ab, bc, ac = b - a, c - b, c - a
    cross = (ab[..., 0] * bc[..., 1] - ab[..., 1] * bc[..., 0]).abs()
    den = torch.linalg.norm(ab, dim=-1) * torch.linalg.norm(bc, dim=-1) * torch.linalg.norm(ac, dim=-1)
    return 2 * cross / den.clamp_min(1e-9)


class DiffEnsemble:
    """Frozen float32 deep copies of the ci_train members; recurrent modules in train mode (cuDNN has no RNN backward
    in eval mode; single-layer, dropout-free, so eval numerics), everything else in eval mode (ci_grad.DiffEnsemble)."""

    def __init__(self, ens, z):
        if ens.kind != 'ci_train':
            raise ValueError(f'gradient refinement needs a ci_train ensemble, got {ens.kind}')
        self.dev, self.members = ens.device, []
        t = lambda v: torch.as_tensor(np.asarray(v, np.float32).reshape(-1), device=self.dev)      # noqa: E731
        for (model, ck, _), zi in zip(ens.members, z):
            m = copy.deepcopy(model).to(self.dev, torch.float32).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            for mod in m.modules():
                if isinstance(mod, torch.nn.RNNBase):
                    assert mod.num_layers == 1 or mod.dropout == 0.0, 'train mode would change this RNN'
                    mod.train()
            if list(ck['norm']['cont_index']) != [0, 1, 2, 3]:
                raise ValueError(f'corridor standardisation of channels {ck["norm"]["cont_index"]} not supported')
            self.members.append((m, t(ck['norm']['mu']), t(ck['norm']['sd']), t(ck['ctx_mu']), t(ck['ctx_sd']),
                                 None if zi is None else torch.as_tensor(np.asarray(zi, np.float32), device=self.dev)))

    def logits(self, X, g5):
        """(M, B) route logits of raw corridors X (B, 5, 96, 32) at raw contexts g5 (B, 5)."""
        out, ones = [], torch.ones_like(X[:, :1])
        with _det():
            for m, mu, sd, cmu, csd, z in self.members:
                x = torch.cat([(X[:, :4] - mu[None, :, None, None]) / sd[None, :, None, None], X[:, 4:5], ones], 1)
                out.append(route_logit(m(x, (g5 - cmu) / csd, z=None if z is None else z.expand(len(X), -1))['haz']))
        return torch.stack(out)


class Chain:
    """Parameter rows (a (B, 3), dv (B, 4)) -> routes -> corridors -> member logits for one decision."""

    def __init__(self, base, dec, dmap, dens, bsp_rows, lat0_rows, arena_soft):
        t = lambda v: torch.as_tensor(np.asarray(v, np.float64), device=dens.dev).to(torch.float32)     # noqa: E731
        self.dec, self.dmap, self.dens, self.arena_soft = dec, dmap, dens, arena_soft
        self.bxy, self.bst, self.bsp, self.lat0 = t(base['waypoints']), t(base['stations']), t(bsp_rows), t(lat0_rows)
        self.cap = t(caps(_base_arrays(base)[2]))
        rel = dec.goal - dec.pose[:2]
        self.g4 = [rel[0], rel[1], float(np.linalg.norm(rel)), float(dec.pose[2])]              # f104_n2_grad.t_ctx

    def __call__(self, a, dv):
        pts, v, st = _shape_rows(self.bxy, self.bst, self.bsp, a, dv, self.lat0)
        X, L = _corridor(pts, v, self.dmap)
        g = torch.tensor(self.g4, dtype=L.dtype, device=L.device)
        return pts, self.dens.logits(X, torch.cat([g[None].expand(len(L), 4), L[:, None]], 1))

    def penalty(self, pts):
        return 1e5 * F.relu(_curv(pts) - 0.95 * KAPPA_MAX).pow(2).sum(1) + \
            10.0 * F.relu(pts.abs() - self.arena_soft).pow(2).sum((1, 2))


def descend(chain, a0, dv0, g):
    """Batched Adam (ci_grad.refine, ci_grad.py:335-376) -> best a, dv (float64), best_step, J_keep per row."""
    dev = chain.dens.dev
    a = torch.as_tensor(a0, device=dev).to(torch.float32).clone().requires_grad_(True)
    dv = torch.as_tensor(dv0, device=dev).to(torch.float32).clone().requires_grad_(True)
    B = a.shape[0]
    opt = torch.optim.Adam([{'params': [a], 'lr': g['lr_a']}, {'params': [dv], 'lr': g['lr_dv']}],
                           betas=tuple(g['betas']))
    best_J = torch.full((B,), float('inf'), dtype=torch.float32, device=dev)
    best_a, best_dv = a.detach().clone(), dv.detach().clone()
    best_step, since = torch.zeros(B, dtype=torch.long, device=dev), torch.zeros(B, dtype=torch.long, device=dev)
    steps, trace = 0, []
    for it in range(g['steps'] + 1):
        pts, Z = chain(a, dv)
        z_mean = Z.mean(0); pen = chain.penalty(pts)                      # autograd node order as ci_grad.py:348-350
        J_fit = z_mean + pen
        with torch.no_grad():
            J_keep = Z.amax(0) + pen                                       # keep 'pessimistic'
            better = J_keep < best_J - 1e-3
            best_J = torch.where(better, J_keep, best_J)
            best_step = torch.where(better, torch.full_like(best_step, it), best_step)
            best_a[better], best_dv[better] = a.detach()[better], dv.detach()[better]
            since = torch.where(better, torch.zeros_like(since), since + 1)
            trace.append(best_J.min())
        if it == g['steps'] or (it > 0 and bool((since >= g['patience']).all())):
            break
        opt.zero_grad(set_to_none=True)
        with _det():
            J_fit.sum().backward()
        with torch.no_grad():
            gn = torch.sqrt((a.grad ** 2).sum(1) + (dv.grad ** 2).sum(1)).clamp_min(1e-12)
            sc = (g['clip'] / gn).clamp_max(1.0); a.grad.mul_(sc[:, None]); dv.grad.mul_(sc[:, None])
        opt.step(); steps += 1
        with torch.no_grad():
            a.clamp_(-chain.cap, chain.cap); dv.clamp_(-SP_CLIP, SP_CLIP)
    cpu = lambda x: x.detach().double().cpu().numpy()                       # noqa: E731
    return dict(a=cpu(best_a), dv=cpu(best_dv), best_step=best_step.cpu().numpy(), J_keep=cpu(best_J), steps_run=steps,
                trace=cpu(torch.stack(trace)))


# ------------------------------------------------------------------------------------------------- float64 finals
def np_route(base, a, dv, anchor=None):
    """Float64 route of the family (ci_grad.np_route): from_params(theta) or, for an anchor (offset, cruise),
    lat0 = offset sin^2(pi f) on a constant cruise profile."""
    a, dv = np.asarray(a, float), np.asarray(dv, float)
    if anchor is None:
        return from_params(base, np.r_[a, dv])
    xy, f, L = _base_arrays(base)
    aa = np.clip(a, -caps(L), caps(L))
    lat = sum(aa[j] * np.sin((j + 1) * np.pi * f) for j in range(MODES)) + float(anchor[0]) * np.sin(np.pi * f) ** 2
    r = _shape(xy, np.full(len(xy), float(anchor[1])), np.clip(lat, -LAT_CLIP, LAT_CLIP),
               _speed_knots(f, np.clip(dv, -SP_CLIP, SP_CLIP)))
    r['meta'] = {'candidate': 'n2_grad_anchor', 'anchor': [float(anchor[0]), float(anchor[1])]}
    return r


def start_params(route, kind):
    """(a, dv, anchor) of a scored candidate: its theta, or (0, 0, (offset, cruise)) for an anchor (ci_grad.py:275)."""
    m = route.get('meta', {})
    if kind == 'anchor' or m.get('candidate') == 'n2_anchor':
        return np.zeros(MODES), np.zeros(KNOTS), (float(m['lateral_offset_m']), float(m['cruise_speed_mps']))
    th = np.asarray(m['theta'], float)
    if th.shape != (MODES + KNOTS,):
        raise ValueError(f'start theta {th.shape}: the gradient chain drives the free family only')
    return th[:MODES].copy(), th[MODES:].copy(), None


def contract(route, dec):
    """Validator at the decision pose + the collector contract: start / end within 0.25 m, finite, speeds in [0, 6]."""
    wp, v = np.asarray(route['waypoints'], float), np.asarray(route['speeds'], float)
    finite = bool(np.isfinite(wp).all() and np.isfinite(v).all() and np.isfinite(np.asarray(route['stations'])).all())
    return bool(validate(route, dec.pose) and finite and ends_within(route, dec.pose[:2], dec.goal)
                and v.min() >= -1e-9 and v.max() <= V_MAX + 1e-6)


def refine_pick(res: Result, score, record, g=GRAD) -> Pick:
    """B (the search result of planner.Scorer `score`) -> the gradient pick G (ci_grad.plan_group) with settings `g`
    (GRAD in every arm; the tests shorten it)."""
    dec, ens = score.dec, score.ens
    B = Pick.of_result(res)
    n0 = int(res.log[0]['n'])
    pool = [int(i) for i in np.argsort(res.z_mean[:n0], kind='stable') if int(i) != res.index][:int(g['starts']) - 1]
    starts = [('B', res.index, res.kinds[res.index], B.route)]
    starts += [('round0', i, res.kinds[i], res.cands[i]) for i in pool]
    xy, f, _ = _base_arrays(dec.base)
    A0, D0, BSP, LAT0, ANC = [], [], [], [], []
    for src, i, kind, r in starts:
        a0, d0, anc = start_params(r, kind)
        if route_sha256(np_route(dec.base, a0, d0, anc)) != route_sha256(r):
            raise RuntimeError(f'start {src}/{i} ({kind}) is not reproduced by its parameters')
        A0.append(a0); D0.append(d0); ANC.append(anc)
        BSP.append(np.asarray(dec.base['speeds'], float) if anc is None else np.full(len(xy), anc[1]))
        LAT0.append(np.zeros(len(xy)) if anc is None else anc[0] * np.sin(np.pi * f) ** 2)
    chain = Chain(dec.base, dec, DetMap(score.smap, ens.device), DiffEnsemble(ens, score.z), np.stack(BSP),
                  np.stack(LAT0), float(g['arena_soft']))
    R = descend(chain, np.stack(A0), np.stack(D0), g)
    finals = [np_route(dec.base, R['a'][k], R['dv'][k], ANC[k]) if R['best_step'][k] > 0 else s[3]
              for k, s in enumerate(starts)]
    ok = [contract(r, dec) for r in finals]
    vidx = [k for k in range(len(starts)) if ok[k]]
    Z = score([s[3] for s in starts] + [finals[k] for k in vidx])           # ONE deployed re-score batch
    zm, zp, ns = Z.mean(0), Z.max(0), len(starts)
    rows = [dict(row=k, src=s[0], pool_index=s[1], kind=s[2], best_step=int(R['best_step'][k]),
                 J_torch=float(R['J_keep'][k]), valid=ok[k], route_sha256=route_sha256(finals[k]),
                 z0_rescored_mean=float(zm[k]), z0_rescored_pess=float(zp[k])) for k, s in enumerate(starts)]
    for j, k in enumerate(vidx, start=ns):
        rows[k].update(z_mean=float(zm[j]), z_pess=float(zp[j]))
    J_B = float(zp[0])
    best = min(vidx, key=lambda k: rows[k]['z_pess']) if vidx else None
    gain = J_B - rows[best]['z_pess'] if best is not None else float('nan')
    abstained = best is None or not gain >= g['abstain']
    rG, zG = (B.route, (B.z_mean, B.z_pess)) if abstained else (finals[best], (rows[best]['z_mean'],
                                                                                 rows[best]['z_pess']))
    grad_rec = dict(settings=g, abstained=bool(abstained), gain=None if math.isnan(gain) else float(gain), J_B=J_B,
                    J_B_cem=B.z_pess, start_row=None if abstained else best, n_valid_finals=len(vidx),
                    steps_run=int(R['steps_run']), trace_best_J_min=[float(x) for x in R['trace']], rows=rows)
    return Pick.of(rG, 'G', float(zG[0]), float(zG[1]), **record, score_calls=score.calls,
                   B=dict(route_sha256=B.route_sha256, z_mean=B.z_mean, z_pess=B.z_pess, **B.record), grad=grad_rec)
