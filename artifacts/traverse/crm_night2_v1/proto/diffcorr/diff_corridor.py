"""Differentiable (torch) re-implementation of f104_n2_dataset.station_tensor + f104_n2_sampler.shape, checked
against the numpy originals on the CRM demo pool, plus gradient-landscape measurements of the route logit."""
import sys, json, time, math, numpy as np, torch, torch.nn.functional as F
ROOT = '/home/harry/NeDM-traverse_mppi'
sys.path.insert(0, ROOT + '/scripts'); sys.path.insert(0, ROOT + '/src')
import f104_n2_dataset as DS, f104_n2_sampler as S, gen_planner as P
from nedm.traverse.fdm_mppi import validate_reference
K = ROOT + '/artifacts/traverse/crm_f104_v1'
DS.init_map(K + '/map_root')
pool = np.load(K + '/demo_v1/pool.npz')
case = json.load(open(K + '/cases_eval/cases/f104_crm_eval_group_0013.json'))
lay = case['layout']; start_xy = np.asarray(lay['start_xy']); goal_xy = np.asarray(case['goal_xy']); yaw = lay['start_yaw']
torch.manual_seed(0); np.random.seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NS, NL, HW = 96, 32, 6.0

# ------------------------------------------------------------------ torch helpers (mirror numpy semantics exactly)
def npgrad(f, h, dim):
    f = f.movedim(dim, 0)
    g = torch.cat([f[1:2] - f[0:1], (f[2:] - f[:-2]) / 2, f[-1:] - f[-2:-1]], 0)
    return g.movedim(0, dim) / h

def tinterp(xq, xp, fp):
    """np.interp, batched: xq (B,Q), xp (B,N) increasing, fp (B,N)."""
    idx = (torch.searchsorted(xp.detach().contiguous(), xq.detach().contiguous(), right=True) - 1).clamp(0, xp.shape[1] - 2)
    x0 = xp.gather(1, idx); x1 = xp.gather(1, idx + 1)
    w = ((xq - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0, 1)
    return fp.gather(1, idx) * (1 - w) + fp.gather(1, idx + 1) * w

class TMap:
    def __init__(self, G, dev, dtype):
        z = G['rgbd'][3]
        self.R = torch.tensor(z, dtype=dtype, device=dev)[None, None]
        self.vpx = torch.tensor(z > -1.999, device=dev)
        self.n, self.mpp, self.ctr, self.es = G['npx'], float(G['mpp']), float(G['ctr']), float(G['elev_scale'])
    def sample(self, x, y):
        row = self.ctr - y / self.mpp; col = self.ctr + x / self.mpp
        grid = torch.stack([col / (self.n - 1) * 2 - 1, row / (self.n - 1) * 2 - 1], -1).reshape(1, -1, 1, 2)
        z = F.grid_sample(self.R, grid, mode='bilinear', padding_mode='border', align_corners=True).reshape(x.shape)
        r0 = torch.floor(row.detach()).long().clamp(0, self.n - 2); c0 = torch.floor(col.detach()).long().clamp(0, self.n - 2)
        valid = self.vpx[r0, c0] & self.vpx[r0, c0 + 1] & self.vpx[r0 + 1, c0] & self.vpx[r0 + 1, c0+ 1]
        return z, valid

def t_station_tensor(wp, sp, M):
    """wp (B,N,2), sp (B,N) -> X (B,5,96,32), route_len (B,)."""
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
    """f (N,), vals (B,k) -> (B,N) smoothstep, free ends (mirror of S.speed_knots)."""
    k = vals.shape[1]; kx = torch.linspace(0, 1, k, device=f.device, dtype=f.dtype)
    seg = (torch.searchsorted(kx, f.contiguous(), right=True) - 1).clamp(0, k - 2)
    u = (f - kx[seg]) / (kx[seg + 1] - kx[seg]); w = u * u * (3 - 2 * u)
    v0 = vals[:, seg]; v1 = vals[:, seg + 1]
    return v0 + (v1 - v0) * w[None]

def t_shape(bxy, bst, bsp, a, dvk, lat_clip=10.0):
    """Mirror of S.shape with lat = sum_j a_j sin(j pi f), dv = speed_knots. a (B,3), dvk (B,4)."""
    B = a.shape[0]; f = ((bst - bst[0]) / (bst[-1] - bst[0])).clamp(0, 1)
    lat = sum(a[:, j:j + 1] * torch.sin((j + 1) * math.pi * f)[None] for j in range(a.shape[1])).clamp(-lat_clip, lat_clip)
    t = npgrad(bxy, 1.0, 0); t = t / torch.linalg.norm(t, dim=1, keepdim=True).clamp_min(1e-9)
    nrm = torch.stack([-t[:, 1], t[:, 0]], 1)
    pts = bxy[None] + lat[..., None] * nrm[None]
    seg = torch.linalg.norm(pts[:, 1:] - pts[:, :-1], dim=-1); st = torch.cat([seg.new_zeros(B, 1), seg.cumsum(1)], 1)
    v = (bsp[None] + t_speed_knots(f, dvk)).clamp(S.V_MIN, S.V_MAX)
    v = torch.minimum(v, torch.sqrt((2 * S.A_DEC * (st[:, -1:] - st)).clamp_min(1e-12)))
    vl = list(v.unbind(1)); N = len(vl)
    for j in range(1, N):
        vl[j] = torch.minimum(vl[j], torch.sqrt((vl[j - 1] ** 2 + 2 * S.A_ACC * (st[:, j] - st[:, j - 1])).clamp_min(1e-12)))
    for j in range(N - 2, -1, -1):
        vl[j] = torch.minimum(vl[j], torch.sqrt((vl[j + 1] ** 2 + 2 * S.A_DEC * (st[:, j + 1] - st[:, j])).clamp_min(1e-12)))
    return pts, torch.stack(vl, 1), st

def t_curv_max(pts):
    a, b, c = pts[:, :-2], pts[:, 1:-1], pts[:, 2:]
    ab, bc, ac = b - a, c - b, c - a
    cross = (ab[..., 0] * bc[..., 1] - ab[..., 1] * bc[..., 0]).abs()
    den = torch.linalg.norm(ab, dim=-1) * torch.linalg.norm(bc, dim=-1) * torch.linalg.norm(ac, dim=-1)
    return (2 * cross / den.clamp_min(1e-9)), (2 * cross / den.clamp_min(1e-9)).amax(1)

class TEnsemble:
    def __init__(self, pattern, dev, dtype):
        self.members = []
        for m, ck in P.RiskModel(pattern, device='cpu').members:
            m = m.to(dev, dtype).eval()
            self.members.append((m, torch.tensor(ck['norm']['mu'], dtype=dtype, device=dev), torch.tensor(ck['norm']['sd'], dtype=dtype, device=dev),
                                 torch.tensor(ck['ctx_mu'], dtype=dtype, device=dev), torch.tensor(ck['ctx_sd'], dtype=dtype, device=dev)))
    def logits(self, X, ctx, members=None):
        out = []
        for i, (m, mu, sd, cmu, csd) in enumerate(self.members):
            if members is not None and i not in members: continue
            x = torch.cat([(X[:, :4] - mu[None, :, None, None]) / sd[None, :, None, None], X[:, 4:5], torch.ones_like(X[:, :1])], 1)
            with torch.backends.cudnn.flags(enabled=False):
                out.append(P.RiskModel.__dict__ and __import__('gen_riskmodel').route_logit(m(x, (ctx - cmu) / csd)))
        return torch.stack(out)  # (members, B)

def t_ctx(L, dtype, dev):
    rel = goal_xy - start_xy
    base = torch.tensor([rel[0], rel[1], np.linalg.norm(rel), yaw], dtype=dtype, device=dev)
    return torch.cat([base[None].expand(len(L), 4), L[:, None]], 1)

# ------------------------------------------------------------------ 1. corridor: torch vs numpy on the 256 pool routes
dt = torch.float64
M = TMap(DS.G, DEV, dt)
wp = torch.tensor(pool['wp'], dtype=dt, device=DEV); sp = torch.tensor(pool['sp'], dtype=dt, device=DEV)
Xn, Ln = P.corridors([dict(waypoints=pool['wp'][i], speeds=pool['sp'][i], stations=pool['st'][i]) for i in range(256)])
Xt, Lt = t_station_tensor(wp, sp, M)
Xt_np = Xt.cpu().numpy()
print('== 1. station_tensor torch vs numpy (256 pool routes, float64 torch vs float32 numpy)')
for c, name in enumerate(['elev_rel', 'grade', 'cross', 'speed', 'valid']):
    print(f'   {name:9s} max|diff| {np.abs(Xt_np[:, c] - Xn[:, c]).max():.2e}   numpy range [{Xn[:, c].min():+.3f}, {Xn[:, c].max():+.3f}]')
print(f'   route_len max|diff| {np.abs(Lt.cpu().numpy() - Ln).max():.2e};  valid fraction {Xn[:, 4].mean():.3f};  clip active grade {np.mean(np.abs(Xn[:,1])>=2):.2e} cross {np.mean(np.abs(Xn[:,2])>=2):.2e}')

# 2. network logits: torch pipeline vs RiskModel.score
ens = TEnsemble(K + '/train_v1/deploy/CRM_N2_s*.pt', DEV, dt)
rig = TEnsemble(ROOT + '/artifacts/traverse/fdm_f104_50h_20260909/night2_v1/final/N2_s*.pt', DEV, dt)
ctx = t_ctx(Lt, dt, DEV)
with torch.no_grad():
    zt = ens.logits(Xt, ctx)
print('== 2. route logit torch pipeline vs pool.npz (RiskModel.score, float32)')
print(f'   ensemble mean max|diff| {np.abs(zt.mean(0).cpu().numpy() - pool["logit_crm"]).max():.2e}; per-member max|diff| {np.abs(zt.cpu().numpy() - pool["logit_crm_members"]).max():.2e}')
print(f'   argmin torch {int(zt.mean(0).argmin())} vs pool {int(np.argmin(pool["logit_crm"]))}')

# ------------------------------------------------------------------ 3. route parameterisation torch vs numpy S.shape
base = dict(waypoints=pool['wp'][0], speeds=np.full(106, 2.0), stations=pool['st'][0])   # cand 0 = anchor offset 0 @ 2 m/s
bxy = torch.tensor(base['waypoints'], dtype=dt, device=DEV); bst = torch.tensor(base['stations'], dtype=dt, device=DEV)
bsp = torch.full((106,), 2.0, dtype=dt, device=DEV)
f_np = np.clip((base['stations'] - base['stations'][0]) / (base['stations'][-1] - base['stations'][0]), 0, 1)
L0 = float(base['stations'][-1]); cap = 0.55 * 0.125 * L0 ** 2 / (np.arange(1, 4) * np.pi) ** 2
print(f'== 3. parameterisation: base route length {L0:.2f} m, sine-amplitude caps {np.round(cap, 2)} m, dv clip +-4 m/s')
rng = np.random.default_rng(1)
A = np.clip(rng.normal(0, 5.0, (16, 3)) / np.arange(1, 4), -cap, cap); DV = np.clip(rng.normal(0, 1.5, (16, 4)), -4, 4)
pts_t, v_t, st_t = t_shape(bxy, bst, bsp, torch.tensor(A, dtype=dt, device=DEV), torch.tensor(DV, dtype=dt, device=DEV))
err_p = err_v = 0
for i in range(16):
    lat = sum(A[i, j] * np.sin((j + 1) * np.pi * f_np) for j in range(3)); lat = np.clip(lat, -10, 10)
    dv = S.speed_knots(f_np, 4, DV[i]); r = S.shape(base['waypoints'], base['stations'], base['speeds'], lat, dv)
    err_p = max(err_p, np.abs(r['waypoints'] - pts_t[i].cpu().numpy()).max()); err_v = max(err_v, np.abs(r['speeds'] - v_t[i].cpu().numpy()).max())
print(f'   16 random (a, dv): waypoints max|diff| {err_p:.2e} m, speeds max|diff| {err_v:.2e} m/s')

# recover (a, dv) of every pool candidate by least squares (lat exact on the sine basis; dv only where no cap is active)
nrm_np = np.gradient(base['waypoints'], axis=0); nrm_np /= np.linalg.norm(nrm_np, axis=1, keepdims=True); nrm_np = np.stack([-nrm_np[:, 1], nrm_np[:, 0]], 1)
Bs = np.stack([np.sin((j + 1) * np.pi * f_np) for j in range(3)], 1)
lat_all = ((pool['wp'] - base['waypoints'][None]) * nrm_np[None]).sum(-1)
A_pool = np.linalg.lstsq(Bs, lat_all.T, rcond=None)[0].T
kx = np.linspace(0, 1, 4); segk = np.clip(np.searchsorted(kx, f_np, side='right') - 1, 0, 2); u = (f_np - kx[segk]) / (kx[segk + 1] - kx[segk]); w = u * u * (3 - 2 * u)
Bk = np.zeros((106, 4)); Bk[np.arange(106), segk] += 1 - w; Bk[np.arange(106), segk + 1] += w
DV_pool = np.zeros((256, 4))
for i in range(256):
    free = (pool['sp'][i] > 0.5 + 1e-6) & (pool['sp'][i] < 6 - 1e-6) & (np.arange(106) < 90)   # away from the terminal cone
    DV_pool[i] = np.linalg.lstsq(Bk[free], pool['sp'][i][free] - 2.0, rcond=None)[0] if free.sum() > 8 else 0
pts_r, v_r, _ = t_shape(bxy, bst, bsp, torch.tensor(A_pool, dtype=dt, device=DEV), torch.tensor(DV_pool, dtype=dt, device=DEV))
rep_p = np.abs(pts_r.cpu().numpy() - pool['wp']).max(axis=(1, 2)); rep_v = np.abs(v_r.cpu().numpy() - pool['sp']).max(1)
print(f'   pool recovery: waypoint error median {np.median(rep_p):.2e} max {rep_p.max():.2e} m; speed error median {np.median(rep_v):.3f} max {rep_v.max():.2f} m/s (dv is not identifiable where clips/cone bind)')
order = np.argsort(pool['logit_crm']); top = order[:8]
print('   top-8 pool candidates: idx', top.tolist(), 'logit', np.round(pool['logit_crm'][top], 2).tolist(), 'a_1', np.round(A_pool[top, 0], 2).tolist())

# ------------------------------------------------------------------ 4. full differentiable chain + finite-difference check
def chain(a, dvk, members=None, E=ens):
    pts, v, st = t_shape(bxy, bst, bsp, a, dvk)
    X, L = t_station_tensor(pts, v, M)
    return E.logits(X, t_ctx(L, a.dtype, a.device), members).mean(0), pts, v, st, X

a0 = torch.tensor(A_pool[top], dtype=dt, device=DEV, requires_grad=True); d0 = torch.tensor(DV_pool[top], dtype=dt, device=DEV, requires_grad=True)
z, *_ = chain(a0, d0); z.sum().backward()
ga, gd = a0.grad.clone(), d0.grad.clone()
print('== 4. autograd vs central finite differences (float64, top-8 candidates, h=1e-4)')
h = 1e-4; fd_a = torch.zeros_like(ga); fd_d = torch.zeros_like(gd)
with torch.no_grad():
    for j in range(3):
        e = torch.zeros_like(a0); e[:, j] = h
        fd_a[:, j] = (chain(a0 + e, d0)[0] - chain(a0 - e, d0)[0]) / (2 * h)
    for j in range(4):
        e = torch.zeros_like(d0); e[:, j] = h
        fd_d[:, j] = (chain(a0, d0 + e)[0] - chain(a0, d0 - e)[0]) / (2 * h)
rel = lambda g, f: ((g - f).abs() / (f.abs() + 1e-6))
print(f'   |dz/da| median {ga.abs().median():.3f} max {ga.abs().max():.3f} (logit per metre);  rel err vs FD median {rel(ga, fd_a).median():.1e} max {rel(ga, fd_a).max():.1e}')
print(f'   |dz/ddv| median {gd.abs().median():.3f} max {gd.abs().max():.3f} (logit per m/s); rel err vs FD median {rel(gd, fd_d).median():.1e} max {rel(gd, fd_d).max():.1e}; exactly-zero dv grads {(gd == 0).float().mean():.2f}')
bad = (rel(ga, fd_a) > 1e-2).sum().item() + (rel(gd, fd_d) > 1e-2).sum().item()
print(f'   entries with rel err > 1e-2: {bad}/{ga.numel() + gd.numel()}  (FD values there: a {fd_a[rel(ga, fd_a) > 1e-2].tolist()[:4]}, dv {fd_d[rel(gd, fd_d) > 1e-2].tolist()[:4]})')

# ------------------------------------------------------------------ 5. 1-D landscape sweeps around the top-1 candidate
print('== 5. 1-D sweeps around the pool argmin (ensemble-mean logit; 401 points per axis)')
i0 = top[0]; a_c = torch.tensor(A_pool[i0], dtype=dt, device=DEV); d_c = torch.tensor(DV_pool[i0], dtype=dt, device=DEV)
def sweep(axis, lo, hi, n=401):
    t = torch.linspace(lo, hi, n, dtype=dt, device=DEV)
    a = a_c[None].repeat(n, 1); d = d_c[None].repeat(n, 1)
    if axis < 3: a[:, axis] = t
    else: d[:, axis - 3] = t
    with torch.no_grad():
        z, pts, v, st, X = chain(a, d)
        curv = t_curv_max(pts)[1]
    return t.cpu().numpy(), z.cpu().numpy(), curv.cpu().numpy(), X[:, 4].sum((1, 2)).cpu().numpy(), pts, v
rows = []
for axis, (lo, hi, name) in enumerate([(-cap[0], cap[0], 'a_1'), (-cap[1], cap[1], 'a_2'), (-cap[2], cap[2], 'a_3'), (-4, 4, 'dv_1'), (-4, 4, 'dv_2'), (-4, 4, 'dv_3'), (-4, 4, 'dv_4')]):
    t, z, curv, nvalid, pts, v = sweep(axis, lo, hi)
    feas = curv <= 0.125 + 1e-6; sel = feas[1:] & feas[:-1]
    dz = (np.diff(z) / np.diff(t))[sel]; d2 = (np.abs(np.diff(z, 2)) / np.diff(t)[0] ** 2)[sel[1:] & sel[:-1]]
    jumps = np.abs(np.diff(z))[sel]; nvchg = (np.diff(nvalid) != 0)[sel]
    flips = int((np.sign(dz[1:]) * np.sign(dz[:-1]) < 0).sum())
    valid_frac = float(feas.mean())
    print(f'   {name:4s} [{lo:+.1f},{hi:+.1f}]  z range [{z.min():+.2f},{z.max():+.2f}]  |dz/dt| median {np.median(np.abs(dz)):.3f} max {np.abs(dz).max():.3f}  '
          f'slope-sign flips {flips:3d}/{len(dz)-1}  |d2z| p99 {np.quantile(d2, .99):.1f}  largest step jump {jumps.max():.3f} (valid-count changed at that step: {bool(nvchg[jumps.argmax()])}, #valid changes {int(nvchg.sum())})  curvature-feasible {valid_frac:.2f}')
    rows.append((name, t, z, curv))
np.savez('/tmp/diffcorr/sweeps.npz', **{n: np.stack([t, z, c]) for n, t, z, c in rows})

# global Lipschitz-ish estimate on random pairs inside the caps
with torch.no_grad():
    n = 512; a = torch.tensor(np.clip(rng.normal(0, 5.0, (n, 3)) / np.arange(1, 4), -cap, cap), dtype=dt, device=DEV)
    d = torch.tensor(np.clip(rng.normal(0, 1.5, (n, 4)), -4, 4), dtype=dt, device=DEV)
    z, pts, v, st, X = chain(a, d); curv = t_curv_max(pts)[1]
    th = torch.cat([a, d], 1); D = torch.cdist(th, th) + torch.eye(n, dtype=dt, device=DEV) * 1e9
    Lip = ((z[:, None] - z[None]).abs() / D)
    print(f'   512 random params: z quantiles {np.round(np.quantile(z.cpu().numpy(), [0, .1, .5, .9, 1]), 2).tolist()}, curvature-feasible {(curv <= .125).float().mean():.2f}, '
          f'pairwise |dz|/|dtheta| median {Lip.median():.3f} p99 {Lip.quantile(.99):.3f} max {Lip.max():.3f} (theta in m and m/s)')

# ------------------------------------------------------------------ 6. sensitivity to the raw corridor (the "10x anomaly" probe)
Xv = Xt[top].clone().requires_grad_(True)
zz = ens.logits(Xv, t_ctx(Lt[top], dt, DEV)).mean(0); zz.sum().backward()
gX = Xv.grad
print('== 6. d logit / d corridor at the top-8 (per-channel L2 norm of the 96x32 gradient, logit per unit of raw channel)')
for c, name in enumerate(['elev_rel', 'grade', 'cross', 'speed', 'valid']):
    print(f'   {name:9s} median {gX[:, c].flatten(1).norm(dim=1).median():.3f} max {gX[:, c].flatten(1).norm(dim=1).max():.3f}')

# ------------------------------------------------------------------ 7. timing (float32, cuda) forward+backward for a batch of 256
ens32 = TEnsemble(K + '/train_v1/deploy/CRM_N2_s*.pt', DEV, torch.float32); M32 = TMap(DS.G, DEV, torch.float32)
bxy32, bst32, bsp32 = bxy.float(), bst.float(), bsp.float()
def chain32(a, d, members=None):
    pts, v, st = t_shape(bxy32, bst32, bsp32, a, d); X, L = t_station_tensor(pts, v, M32)
    return ens32.logits(X, t_ctx(L, torch.float32, DEV), members).mean(0), pts, v, st
a = torch.tensor(A_pool, dtype=torch.float32, device=DEV, requires_grad=True); d = torch.tensor(DV_pool, dtype=torch.float32, device=DEV, requires_grad=True)
for _ in range(2): chain32(a, d)[0].sum().backward()
torch.cuda.synchronize(); t0 = time.time()
for _ in range(5): chain32(a, d)[0].sum().backward()
torch.cuda.synchronize(); print(f'== 7. float32 {DEV}: forward+backward of 256 routes through the 5-member ensemble: {(time.time() - t0) / 5 * 1e3:.0f} ms per step')


# ------------------------------------------------------------------ 8. cuDNN gotcha + B=16 timing
try:
    m0 = ens32.members[0][0]; xx = torch.randn(4, 6, 96, 32, device=DEV, requires_grad=True); cc = torch.randn(4, 5, device=DEV)
    m0(xx, cc).sum().backward(); print('== 8. cudnn eval-mode GRU backward: OK (no error)')
except RuntimeError as e:
    print('== 8. cudnn eval-mode GRU backward on CUDA raises:', str(e).splitlines()[0][:90])
    m0.mix.train(); xx.grad = None; m0(xx, cc).sum().backward(); print('   workaround m.mix.train() (GRU only; BN/dropout stay eval) works; or torch.backends.cudnn.flags(enabled=False)')
a16 = torch.tensor(A_pool[:16], dtype=torch.float32, device=DEV, requires_grad=True); d16 = torch.tensor(DV_pool[:16], dtype=torch.float32, device=DEV, requires_grad=True)
for _ in range(2): chain32(a16, d16)[0].sum().backward()
torch.cuda.synchronize(); t0 = time.time()
for _ in range(5): chain32(a16, d16)[0].sum().backward()
torch.cuda.synchronize(); print(f'   B=16 forward+backward (5 members, cudnn off): {(time.time() - t0) / 5 * 1e3:.0f} ms per step')

# ------------------------------------------------------------------ 9. Adam descent: leave-one-member-out + random starts
def penalties(pts):
    kap = t_curv_max(pts)[0]
    return F.relu(kap - 0.125).pow(2).sum(1) * 1e3 + F.relu(pts.abs() - 37.0).pow(2).sum((1, 2)) * 10.0
cap_t = torch.tensor(cap, dtype=torch.float32, device=DEV)
def descend(A0, D0, fit, steps=60, lr_a=0.05, lr_d=0.1):
    a = torch.tensor(A0, dtype=torch.float32, device=DEV, requires_grad=True); d = torch.tensor(D0, dtype=torch.float32, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([{'params': [a], 'lr': lr_a}, {'params': [d], 'lr': lr_d}])
    best = None
    for it in range(steps):
        z, pts, v, st = chain32(a, d, members=fit); pen = penalties(pts)
        (z + pen).sum().backward(); opt.step(); opt.zero_grad()
        with torch.no_grad(): a.clamp_(-cap_t, cap_t); d.clamp_(-4, 4)
    return a.detach(), d.detach()
def score_all(a, d):
    with torch.no_grad():
        pts, v, st = t_shape(bxy32, bst32, bsp32, a, d); X, L = t_station_tensor(pts, v, M32); ctx = t_ctx(L, torch.float32, DEV)
        zm = ens32.logits(X, ctx); Xd, Ld = t_station_tensor(pts.double(), v.double(), M); zr = rig.logits(Xd, t_ctx(Ld, dt, DEV)).mean(0)
        kmax = t_curv_max(pts)[1]
    return zm, zr, pts, v, st, kmax
print('== 9a. leave-one-member-out descent from the top-8 (60 Adam steps, lr 0.05 m / 0.1 m/s, fit = 4 members, held-out = 5th)')
zm0, zr0, *_ = score_all(torch.tensor(A_pool[top], dtype=torch.float32, device=DEV), torch.tensor(DV_pool[top], dtype=torch.float32, device=DEV))
fit_gain, ho_gain, pess_gain, rig_gain = [], [], [], []
for ho in range(5):
    fit = [k for k in range(5) if k != ho]
    a1, d1 = descend(A_pool[top], DV_pool[top], fit)
    zm1, zr1, pts1, v1, st1, kmax = score_all(a1, d1)
    fit_gain.append((zm1[fit].mean(0) - zm0[fit].mean(0)).cpu().numpy()); ho_gain.append((zm1[ho] - zm0[ho]).cpu().numpy())
    pess_gain.append((zm1.amax(0) - zm0.amax(0)).cpu().numpy()); rig_gain.append((zr1 - zr0).cpu().numpy())
    print(f'   held-out {ho}: fit-mean logit change per start {np.round(fit_gain[-1], 2).tolist()}')
    print(f'              held-out member change     {np.round(ho_gain[-1], 2).tolist()}   curvature ok {(kmax <= .125).cpu().numpy().tolist()}')
fg, hg, pg, rg = (np.array(x) for x in (fit_gain, ho_gain, pess_gain, rig_gain))
print(f'   MEAN over 5 folds x 8 starts: fit-members {fg.mean():+.2f} logit, held-out member {hg.mean():+.2f}, pessimistic(max of 5) {pg.mean():+.2f}, rigid ensemble {rg.mean():+.2f}')
print(f'   fraction of (fold, start) where the held-out member also improved: {(hg < 0).mean():.2f}; where it got worse by > 0.5: {(hg > 0.5).mean():.2f}')
print(f'   start-wise (pool idx {top.tolist()}): fit {np.round(fg.mean(0), 2).tolist()} held-out {np.round(hg.mean(0), 2).tolist()}')

print('== 9b. descent with ALL 5 members from the top-8 and from 8 random valid draws; final routes re-validated')
for label, A0, D0 in (('top-8', A_pool[top], DV_pool[top]), ('random-8', np.clip(rng.normal(0, 5.0, (8, 3)) / np.arange(1, 4), -cap, cap), np.clip(rng.normal(0, 1.5, (8, 4)), -4, 4))):
    zm0, zr0, *_ = score_all(torch.tensor(A0, dtype=torch.float32, device=DEV), torch.tensor(D0, dtype=torch.float32, device=DEV))
    a1, d1 = descend(A0, D0, [0, 1, 2, 3, 4], steps=100)
    zm1, zr1, pts1, v1, st1, kmax = score_all(a1, d1)
    print(f'   {label}: mean-logit before {np.round(zm0.mean(0).cpu().numpy(), 2).tolist()}')
    print(f'   {label}: mean-logit after  {np.round(zm1.mean(0).cpu().numpy(), 2).tolist()}  (pool argmin -5.59)')
    print(f'   {label}: pessimistic after {np.round(zm1.amax(0).cpu().numpy(), 2).tolist()}  rigid-ens before->after {np.round(zr0.cpu().numpy(), 1).tolist()} -> {np.round(zr1.cpu().numpy(), 1).tolist()}')
    for i in range(8):
        r = dict(waypoints=pts1[i].cpu().numpy().astype(float), speeds=v1[i].cpu().numpy().astype(float), stations=st1[i].cpu().numpy().astype(float))
        r['headings'] = np.arctan2(np.gradient(r['waypoints'][:, 1]), np.gradient(r['waypoints'][:, 0]))
        chk = P.safe_validate(r, [], P.CFG, np.array([start_xy[0], start_xy[1], yaw]))
        lat = ((pts1[i].cpu().numpy() - base['waypoints']) * nrm_np).sum(-1)
        print(f'      start {i}: validator {"ok" if chk["valid"] else chk["reasons"]} kmax {chk.get("max_curvature", float("nan")):.3f}  a {np.round(a1[i].cpu().numpy(), 2).tolist()} dv {np.round(d1[i].cpu().numpy(), 2).tolist()}  max|lat| {np.abs(lat).max():.1f} m  mean v {r["speeds"][1:-1].mean():.2f}  L {r["stations"][-1]:.1f}')
    np.savez(f'/tmp/diffcorr/descent_{label}.npz', a=a1.cpu().numpy(), d=d1.cpu().numpy(), zm0=zm0.cpu().numpy(), zm1=zm1.cpu().numpy(), pts=pts1.cpu().numpy(), v=v1.cpu().numpy())
