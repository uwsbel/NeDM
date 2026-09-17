"""Night session step 3: train risk-model variants and score them on a group-held-out dev fold.

Architectures
  old    : current production MultiHead (pooled 2-D CNN + 39-d scalar MLP incl. mean speed)
  hazard : station-preserving CNN -> BiGRU -> per-station hazard (discrete-time survival)
Labels
  fail | unsafe (fail or rollback) | hazard (unsafe, with the event pinned to its station)
Regularisers
  speed_drop p : hide speed (old: the two speed scalars; hazard: the speed channel + a flag)
  pair_w       : logistic ranking loss on same-group same-speed pairs (terrain-only supervision)
  jitter       : random +-1 lateral-cell shift of the corridor each step
  gradpen      : input-gradient penalty on the corridor (smoothness vs MPPI perturbations)
  dq           : down-weight runs that drifted > 3 m off the planned route before the event
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, 'scripts')
from train_f104_multihead import MultiHead

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
DS = ROOT + '/night_v1/station_ds.npz'
DEV = 'cuda'
SPEED_SCAL = (15, 16)            # mean_ref_speed, min_ref_speed in the 18 old scalars


def dev_group(g):
    return int(hashlib.md5(g.encode()).hexdigest(), 16) % 5 == 0


class HazardNet(nn.Module):
    def __init__(self, cin, nctx, width=64):
        super().__init__()
        c = [cin, 32, 64, 64, 96]
        L = []
        for i in range(4):
            L += [nn.Conv2d(c[i], c[i+1], 3, stride=(1, 1 if i == 0 else 2), padding=1),
                  nn.BatchNorm2d(c[i+1]), nn.GELU()]
        self.cnn = nn.Sequential(*L)
        self.lat = nn.Linear(2 * c[-1], 96)
        self.ctx = nn.Sequential(nn.Linear(nctx, 32), nn.GELU())
        self.tconv = nn.Sequential(nn.Conv1d(96 + 32 + 1, 96, 5, padding=2), nn.GELU(), nn.Dropout(0.1))
        self.gru = nn.GRU(96, width, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * width, 1)

    def forward(self, x, ctx):
        f = self.cnn(x)                                          # B,C,96,4
        f = torch.cat([f.mean(-1), f.amax(-1)], 1).transpose(1, 2)   # B,96,2C
        f = F.gelu(self.lat(f))
        B, S, _ = f.shape
        pos = torch.linspace(0, 1, S, device=x.device)[None, :, None].expand(B, S, 1)
        c = self.ctx(ctx)[:, None, :].expand(B, S, 32)
        h = torch.cat([f, c, pos], -1).transpose(1, 2)
        h = self.tconv(h).transpose(1, 2)
        h, _ = self.gru(h)
        return self.head(h).squeeze(-1)                          # B,96 hazard logits


def route_logit(haz):
    """log cumulative hazard == cloglog(P(event anywhere on the route))."""
    return torch.log(F.softplus(haz).sum(1) + 1e-6)


def survival_nll(haz, ev):
    """ev = event station (>=0) or -1 for a censored (clean) route."""
    S = haz.shape[1]
    idx = torch.arange(S, device=haz.device)[None, :]
    evc = torch.where(ev >= 0, ev, torch.full_like(ev, S - 1))[:, None]
    surv_mask = (idx < evc) | ((ev < 0)[:, None] & (idx <= evc))
    nll = (F.softplus(haz) * surv_mask).sum(1)
    hit = (ev >= 0)
    nll = nll + torch.where(hit, F.softplus(-haz.gather(1, evc).squeeze(1)), torch.zeros_like(nll))
    return nll


def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    if y.min() == y.max(): return float('nan')
    r = np.argsort(np.argsort(s)) + 1.0; n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def cell_auc(y, s, cells):
    ok = tot = 0.0
    for c in np.unique(cells):
        m = cells == c; yy, ss = y[m], s[m]
        if yy.min() == yy.max(): continue
        p, n = ss[yy == 1], ss[yy == 0]
        cmp = p[:, None] - n[None, :]
        ok += (cmp > 0).sum() + 0.5 * (cmp == 0).sum(); tot += cmp.size
    return (ok / tot if tot else float('nan')), int(tot)


def metrics(s, d, m):
    out = {}
    grp = d['group'][m].astype(str); prof = d['profile'][m].astype(int)
    gcell = np.array([f'{g}|{p}' for g, p in zip(grp, prof)])
    for lab in ('unsafe', 'fail'):
        y = d[lab][m].astype(float)
        out[f'P_{lab}'] = auc(y, s)
        out[f'W_{lab}'], out[f'Wn_{lab}'] = cell_auc(y, s, grp)
        out[f'G_{lab}'], out[f'Gn_{lab}'] = cell_auc(y, s, gcell)
    return out


class Data:
    def __init__(self):
        d = np.load(DS, allow_pickle=True)
        self.d = {k: d[k] for k in d.files}
        self.n = len(self.d['id'])
        src, sp = self.d['source'].astype(str), self.d['split'].astype(str)
        grp = self.d['group'].astype(str)
        isdev = np.array([dev_group(g) for g in grp])
        self.fit = (src == 'designed') & (sp == 'train') & ~isdev
        self.dev = (src == 'designed') & (sp == 'train') & isdev
        ctx = self.d['ctx'].astype(np.float32)
        self.ctx_mu, self.ctx_sd = ctx[self.fit].mean(0), ctx[self.fit].std(0) + 1e-6
        self.ctx = (ctx - self.ctx_mu) / self.ctx_sd


def build_inputs(D, arch):
    d = D.d
    if arch == 'hazard':
        X = d['X'].astype(np.float32)
        mu = X[D.fit][:, :4].mean((0, 2, 3)); sd = X[D.fit][:, :4].std((0, 2, 3)) + 1e-6
        Xn = X.copy(); Xn[:, :4] = (X[:, :4] - mu[None, :, None, None]) / sd[None, :, None, None]
        known = np.ones((len(X), 1, X.shape[2], X.shape[3]), np.float32)
        return np.concatenate([Xn, known], 1), dict(mu=mu, sd=sd)
    P = d['old_patch'].astype(np.float32)
    vec = np.concatenate([d['old_scal'].astype(np.float32), d['ctx'][:, :21].astype(np.float32)], 1)
    pmu = P[D.fit].mean((0, 2, 3), keepdims=True); psd = P[D.fit].std((0, 2, 3), keepdims=True) + 1e-6
    vmu, vsd = vec[D.fit].mean(0), vec[D.fit].std(0) + 1e-6
    return ((P - pmu) / psd, (vec - vmu) / vsd), dict(pmu=pmu, psd=psd, vmu=vmu, vsd=vsd)


def geometry_pairs(D, lab, mask):
    d = D.d; idx = np.where(mask)[0]; cells = {}
    for i in idx:
        cells.setdefault((d['group'][i], int(d['profile'][i])), []).append(i)
    pairs = []
    for c, ii in cells.items():
        ii = np.array(ii); y = d[lab][ii]
        for p in ii[y == 1]:
            for n in ii[y == 0]:
                pairs.append((p, n))
    return np.array(pairs, np.int64)


def score(model, arch, inp, ctx, idx, bs=1024):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i+bs]
            if arch == 'hazard':
                z = route_logit(model(torch.tensor(inp[j], device=DEV), torch.tensor(ctx[j], device=DEV)))
            else:
                p, v = inp
                z = model(torch.tensor(p[j], device=DEV), torch.tensor(v[j], device=DEV))[0]
            out.append(z.float().cpu().numpy())
    return np.concatenate(out)


def train(cfg, D=None, fit_mask=None, save=None, log=print):
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    D = D or Data(); d = D.d; arch = cfg['arch']; lab = cfg['label']
    fit = D.fit if fit_mask is None else fit_mask
    inp, norm = build_inputs(D, arch)
    ctx = D.ctx
    y_route = d['fail' if lab == 'fail' else 'unsafe'].astype(np.float32)
    ev = d['event_idx'].astype(np.int64)
    w = np.ones(D.n, np.float32)
    if cfg.get('dq'):
        w[(d['max_dev_pre'] > 3.0)] = 0.3
    fi = np.where(fit)[0]
    if arch == 'hazard':
        model = HazardNet(inp.shape[1], ctx.shape[1]).to(DEV)
        Xg = torch.tensor(inp[fi], device=DEV); Cg = torch.tensor(ctx[fi], device=DEV)
    else:
        model = MultiHead(inp[1].shape[1]).to(DEV)
        Pg = torch.tensor(inp[0][fi], device=DEV); Vg = torch.tensor(inp[1][fi], device=DEV)
    Yg = torch.tensor(y_route[fi], device=DEV); Eg = torch.tensor(ev[fi], device=DEV)
    Wg = torch.tensor(w[fi], device=DEV)
    pos = {i: k for k, i in enumerate(fi)}
    pairs = geometry_pairs(D, 'fail' if lab == 'fail' else 'unsafe', fit) if cfg.get('pair_w', 0) > 0 else None
    if pairs is not None:
        pairs = np.array([(pos[a], pos[b]) for a, b in pairs], np.int64)
        log(f'  geometry pairs for ranking loss: {len(pairs)}')
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr', 2e-3), weight_decay=cfg.get('wd', 1e-4))
    n = len(fi); bs = 256; E = cfg.get('epochs', 30)
    steps = E * (n // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.get('lr', 2e-3), total_steps=steps)
    pw = torch.tensor(float((y_route[fi] == 0).sum() / max(y_route[fi].sum(), 1)), device=DEV)
    pdrop = cfg.get('speed_drop', 0.0)

    def prep_hazard(k):
        x = Xg[k]
        if pdrop > 0:
            drop = (torch.rand(len(k), device=DEV) < pdrop)
            if drop.any():
                x = x.clone(); x[drop, 3] = 0.0; x[drop, 5] = 0.0
        if cfg.get('jitter'):
            sh = int(np.random.choice([-1, 0, 1]))
            if sh: x = torch.roll(x, sh, dims=3)
        return x

    def prep_old(k):
        p, v = Pg[k], Vg[k]
        if pdrop > 0:
            drop = (torch.rand(len(k), device=DEV) < pdrop)
            if drop.any():
                v = v.clone()
                rows = drop.nonzero(as_tuple=True)[0][:, None]
                v[rows, torch.tensor(SPEED_SCAL, device=DEV)[None, :]] = 0.0
        return p, v

    t0 = time.time(); step = 0
    for e in range(E):
        model.train(); perm = torch.randperm(n, device=DEV)
        for i in range(0, n - bs + 1, bs):
            k = perm[i:i+bs]
            gp = arch == 'hazard' and cfg.get('gradpen', 0) > 0
            if arch == 'hazard':
                x = prep_hazard(k)
                if gp:
                    x = x.detach().requires_grad_(True)
                    with torch.backends.cudnn.flags(enabled=False):
                        haz = model(x, Cg[k])
                else:
                    haz = model(x, Cg[k])
                if lab == 'hazard':
                    loss = (survival_nll(haz, Eg[k]) * Wg[k]).mean()
                else:
                    loss = (F.binary_cross_entropy_with_logits(route_logit(haz), Yg[k], pos_weight=pw, reduction='none') * Wg[k]).mean()
                if gp:
                    g, = torch.autograd.grad(route_logit(haz).sum(), x, create_graph=True)
                    loss = loss + cfg['gradpen'] * (g[:, :3] ** 2).mean() * 1e3
            else:
                p_, v_ = prep_old(k)
                lf = model(p_, v_)[0]
                loss = (F.binary_cross_entropy_with_logits(lf, Yg[k], pos_weight=pw, reduction='none') * Wg[k]).mean()
            if pairs is not None:
                pk = pairs[np.random.randint(0, len(pairs), 128)]
                a = torch.tensor(pk[:, 0], device=DEV); b = torch.tensor(pk[:, 1], device=DEV)
                if arch == 'hazard':
                    za = route_logit(model(Xg[a], Cg[a])); zb = route_logit(model(Xg[b], Cg[b]))
                else:
                    za = model(Pg[a], Vg[a])[0]; zb = model(Pg[b], Vg[b])[0]
                loss = loss + cfg['pair_w'] * F.softplus(zb - za).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if sch.last_epoch < sch.total_steps - 1: sch.step()
            step += 1
    dev_idx = np.where(D.dev)[0]
    s_dev = score(model, arch, inp, ctx, dev_idx)
    m = metrics(s_dev, d, D.dev)
    m['train_s'] = round(time.time() - t0, 1)
    if save:
        torch.save({'state': model.state_dict(), 'cfg': cfg, 'norm': norm,
                    'ctx_mu': D.ctx_mu, 'ctx_sd': D.ctx_sd}, save)
    return model, m, (inp, ctx)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(a.cfg)
    _, m, _ = train(cfg, save=a.out.replace('.json', '.pt'))
    json.dump({'cfg': cfg, 'dev': m}, open(a.out, 'w'), indent=1)
    print(json.dumps({'name': cfg.get('name'), 'seed': cfg['seed'], **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}}))
