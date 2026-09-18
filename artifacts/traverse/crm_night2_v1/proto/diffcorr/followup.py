import sys, glob, json, numpy as np, torch, torch.nn.functional as F
src = open('/tmp/diffcorr/diff_corridor.py').read().split('# ------------------------------------------------------------------ 1.')[0]
exec(src)
dt = torch.float64; M = TMap(DS.G, DEV, dt); M32 = TMap(DS.G, DEV, torch.float32)
base = dict(waypoints=pool['wp'][0], speeds=np.full(106, 2.0), stations=pool['st'][0])
bxy = torch.tensor(base['waypoints'], dtype=dt, device=DEV); bst = torch.tensor(base['stations'], dtype=dt, device=DEV); bsp = torch.full((106,), 2.0, dtype=dt, device=DEV)
f_np = np.clip((base['stations'] - base['stations'][0]) / (base['stations'][-1] - base['stations'][0]), 0, 1)
L0 = float(base['stations'][-1]); cap = 0.55 * 0.125 * L0 ** 2 / (np.arange(1, 4) * np.pi) ** 2
ens = TEnsemble(K + '/train_v1/deploy/CRM_N2_s*.pt', DEV, dt)
anchor = np.array([start_xy[0], start_xy[1], yaw])
def np_route(a, dv):
    lat = np.clip(sum(a[j] * np.sin((j + 1) * np.pi * f_np) for j in range(3)), -10, 10)
    r = S.shape(base['waypoints'], base['stations'], base['speeds'], lat, S.speed_knots(f_np, 4, np.asarray(dv, float)))
    return r
print('== A. final descent parameters re-shaped in float64 numpy (S.shape) and validated')
for label in ('top-8', 'random-8'):
    z = np.load(f'/tmp/diffcorr/descent_{label}.npz')
    out = []
    for i in range(8):
        r = np_route(z['a'][i].astype(float), z['d'][i].astype(float)); chk = P.safe_validate(r, [], P.CFG, anchor)
        acc = np.diff(r['speeds'] ** 2) / (2 * np.maximum(np.diff(r['stations']), 1e-8))
        out.append(f'{"ok" if chk["valid"] else "/".join(chk["reasons"])}(k{chk.get("max_curvature", np.nan):.3f},acc[{acc.min():+.2f},{acc.max():+.2f}])')
    print(f'   {label}: ', ' '.join(out))
# float32 torch route -> validator, to confirm the round-off explanation
z = np.load('/tmp/diffcorr/descent_top-8.npz')
r32 = dict(waypoints=z['pts'][0].astype(float), speeds=z['v'][0].astype(float), stations=np.r_[0, np.cumsum(np.linalg.norm(np.diff(z['pts'][0].astype(float), axis=0), axis=1))])
acc = np.diff(r32['speeds'] ** 2) / (2 * np.maximum(np.diff(r32['stations']), 1e-8))
print(f'   float32 torch route (start 0): accel range [{acc.min():+.5f}, {acc.max():+.5f}] vs limits [-2, +1.5] +-1e-6  -> float32 round-off trips the validator when the limits are active')

print('== B. descent variants from the top-8 (all 5 members, 100 Adam steps): geometry-only (dv frozen) vs speed-only (a frozen) vs both')
def chain64(a, d):
    pts, v, st = t_shape(bxy, bst, bsp, a, d); X, L = t_station_tensor(pts, v, M)
    return ens.logits(X, t_ctx(L, dt, DEV)), pts, v, st
def penalties(pts):
    kap = t_curv_max(pts)[0]
    return F.relu(kap - 0.125).pow(2).sum(1) * 1e3 + F.relu(pts.abs() - 37.0).pow(2).sum((1, 2)) * 10.0
cap_t = torch.tensor(cap, dtype=dt, device=DEV)
nrm_np = np.gradient(base['waypoints'], axis=0); nrm_np /= np.linalg.norm(nrm_np, axis=1, keepdims=True); nrm_np = np.stack([-nrm_np[:, 1], nrm_np[:, 0]], 1)
Bs = np.stack([np.sin((j + 1) * np.pi * f_np) for j in range(3)], 1)
A_pool = np.linalg.lstsq(Bs, ((pool['wp'] - base['waypoints'][None]) * nrm_np[None]).sum(-1).T, rcond=None)[0].T
kx = np.linspace(0, 1, 4); segk = np.clip(np.searchsorted(kx, f_np, side='right') - 1, 0, 2); u = (f_np - kx[segk]) / (kx[segk + 1] - kx[segk]); w = u * u * (3 - 2 * u)
Bk = np.zeros((106, 4)); Bk[np.arange(106), segk] += 1 - w; Bk[np.arange(106), segk + 1] += w
DV_pool = np.zeros((256, 4))
for i in range(256):
    free = (pool['sp'][i] > 0.5 + 1e-6) & (pool['sp'][i] < 6 - 1e-6) & (np.arange(106) < 90)
    DV_pool[i] = np.linalg.lstsq(Bk[free], pool['sp'][i][free] - 2.0, rcond=None)[0] if free.sum() > 8 else 0
top = np.argsort(pool['logit_crm'])[:8]
def descend(A0, D0, opt_a=True, opt_d=True, steps=100, lr_a=0.05, lr_d=0.1, curv_w=1e3):
    a = torch.tensor(A0, dtype=dt, device=DEV, requires_grad=opt_a); d = torch.tensor(D0, dtype=dt, device=DEV, requires_grad=opt_d)
    groups = ([{'params': [a], 'lr': lr_a}] if opt_a else []) + ([{'params': [d], 'lr': lr_d}] if opt_d else [])
    opt = torch.optim.Adam(groups)
    for it in range(steps):
        zm, pts, v, st = chain64(a, d); kap = t_curv_max(pts)[0]
        pen = F.relu(kap - 0.125).pow(2).sum(1) * curv_w + F.relu(pts.abs() - 37.0).pow(2).sum((1, 2)) * 10.0
        (zm.mean(0) + pen).sum().backward(); opt.step(); opt.zero_grad()
        with torch.no_grad():
            if opt_a: a.clamp_(-cap_t, cap_t)
            if opt_d: d.clamp_(-4, 4)
    with torch.no_grad():
        zm, pts, v, st = chain64(a, d); kmax = t_curv_max(pts)[1]
    return a.detach().cpu().numpy(), d.detach().cpu().numpy(), zm.mean(0).cpu().numpy(), zm.amax(0).cpu().numpy(), kmax.cpu().numpy(), v[:, 1:-1].mean(1).cpu().numpy()
with torch.no_grad():
    z0 = chain64(torch.tensor(A_pool[top], dtype=dt, device=DEV), torch.tensor(DV_pool[top], dtype=dt, device=DEV))[0].mean(0).cpu().numpy()
print('   before            ', np.round(z0, 2).tolist())
for name, oa, od in (('geometry-only', True, False), ('speed-only', False, True), ('both', True, True)):
    a1, d1, zm1, zp1, kmax, mv = descend(A_pool[top], DV_pool[top], oa, od)
    val = [P.safe_validate(np_route(a1[i], d1[i]), [], P.CFG, anchor) for i in range(8)]
    print(f'   {name:14s} mean-logit after {np.round(zm1, 2).tolist()}  mean v {np.round(mv, 2).tolist()}  validator {[("ok" if c["valid"] else "/".join(c["reasons"])) for c in val]}')
# stronger curvature handling: log-barrier-ish weight 1e5 + smaller lr on a
a1, d1, zm1, zp1, kmax, mv = descend(A_pool[top], DV_pool[top], True, True, curv_w=1e5, lr_a=0.02)
val = [P.safe_validate(np_route(a1[i], d1[i]), [], P.CFG, anchor) for i in range(8)]
print(f'   both, curv_w=1e5, lr_a=0.02: mean-logit after {np.round(zm1, 2).tolist()}  kmax {np.round(kmax, 3).tolist()}  validator {[("ok" if c["valid"] else "/".join(c["reasons"])) for c in val]}')

print('== C. locked evaluation picks (eval_v1, 200 pairs): risk of the CRM-trained pick')
pk = sorted(glob.glob(K + '/eval_v1/picks/*.json')); one = json.load(open(pk[0]))
print('   pick file keys:', list(one.keys()), '| arms:', list(one['arms'].keys()), '| crm arm:', {k: v for k, v in one['arms']['crm'].items() if k != 'route'})
risks = []
for p in pk:
    d = json.load(open(p))['arms']['crm']; risks.append(d.get('risk', d.get('p', np.nan)))
risks = np.array(risks, float)
print(f'   n {len(risks)}  P(unsafe) of pick: quantiles {np.round(np.nanquantile(risks, [0, .25, .5, .75, .9, 1]), 3).tolist()};  #picks with P>5% {int((risks > .05).sum())}, >20% {int((risks > .2).sum())}, >50% {int((risks > .5).sum())}')
res = K + '/eval_v1/results.json'
try:
    R = json.load(open(res)); print('   results.json keys:', list(R.keys())[:12])
except Exception as e: print('   results.json:', e)
print('== D. demo pool P(unsafe) by rank:', np.round(np.sort(pool['p_crm'])[:12], 3).tolist(), '... #P<5%:', int((pool['p_crm'] < .05).sum()), '#P<50%:', int((pool['p_crm'] < .5).sum()))
