"""Night-2 architecture / input-output study trainer (single file, one JSON row per run, RiskModel-compatible checkpoints).

Arms (--arch): gru (deployed CNN-GRU, --width 64), gru120 (width 120, parameter-matched to tx96_2), gru2 (two stacked
GRU layers width 96), tx_conv (existing conv-transformer + final LayerNorm), tx96_2 / tx96_4 / tx128_4 / tx192_4 (tokenised
transformer: per-station CNN column -> token + sinusoidal position, [CTX] token, pre-LN, final LN), patch4_128_4 /
patch2_128_6 (patch transformer without the CNN), mlp (no cross-station mixing).
Inputs: corridor X (5 channels + ones) [+ vx plane with --vplane]; context --ctx geom|chassis|vel (5 / 12 / 8 columns of the
stored 22-d ctx). Outputs: per-station hazard (survival loss) [+ energy head kJ/m and time head s/m with --energy LAMBDA:
first-arrival increments, censored at the event station, Huber on log1p, plus a route-level term].
Data: npz with X (n,5,96,32) f16, ctx (n,22), group, split, source, profile, fail, unsafe, event_idx, route_len [, E, T (n,96)].
Rows: --mode holdout fits split==train & ~dev(md5 % 5), --mode deploy fits every split==train row; held-out = split != train.
"""
import argparse, hashlib, json, math, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_riskmodel import Net as LegacyNet, route_logit

DEV = os.environ.get('N2_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
CTX_COLS = {'geom': [17, 18, 19, 20, 21], 'chassis': [0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 21], 'vel': [0, 1, 6, 17, 18, 19, 20, 21]}


# ----------------------------------------------------------------------------- models
class CNNFront(nn.Module):
    """Station-preserving CNN of the deployed net: (B,cin,96,32) -> (B,96,192) [mean,max over the 4 lateral cells]."""
    def __init__(self, cin):
        super().__init__()
        c = [cin, 32, 64, 64, 96]; L = []
        for i in range(4):
            L += [nn.Conv2d(c[i], c[i + 1], 3, stride=(1, 1 if i == 0 else 2), padding=1), nn.BatchNorm2d(c[i + 1]), nn.GELU()]
        self.cnn = nn.Sequential(*L)

    def forward(self, x):
        f = self.cnn(x)
        return torch.cat([f.mean(-1), f.amax(-1)], 1).transpose(1, 2)


def sinusoid(n, d):
    pos = torch.arange(n)[:, None].float(); i = torch.arange(0, d, 2).float()
    pe = torch.zeros(n, d); pe[:, 0::2] = torch.sin(pos / 10000 ** (i / d)); pe[:, 1::2] = torch.cos(pos / 10000 ** (i / d))
    return pe


class Heads(nn.Module):
    def __init__(self, d, energy):
        super().__init__()
        self.h = nn.Linear(d, 1); self.energy = energy
        if energy:
            self.e = nn.Linear(d, 1); self.t = nn.Linear(d, 1)

    def forward(self, h):
        out = {'haz': self.h(h).squeeze(-1)}
        if self.energy:
            out['e'] = F.softplus(self.e(h).squeeze(-1)); out['t'] = F.softplus(self.t(h).squeeze(-1))   # kJ/m, s/m
        return out


class GRUNet(nn.Module):
    def __init__(self, cin, nctx, width=64, layers=1, energy=False, mlp=False):
        super().__init__()
        self.front = CNNFront(cin); self.lat = nn.Linear(192, 96); self.ctx = nn.Sequential(nn.Linear(nctx, 32), nn.GELU())
        k = 1 if mlp else 5
        self.tconv = nn.Sequential(nn.Conv1d(129, 96, k, padding=k // 2), nn.GELU(), nn.Dropout(0.1)); self.mlp = mlp
        if mlp:
            self.mix = nn.Sequential(nn.Linear(96, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU()); d = 128
        else:
            self.mix = nn.GRU(96, width, num_layers=layers, batch_first=True, bidirectional=True); d = 2 * width
        self.heads = Heads(d, energy)

    def forward(self, x, ctx):
        f = F.gelu(self.lat(self.front(x))); B, S, _ = f.shape
        pos = torch.linspace(0, 1, S, device=x.device)[None, :, None].expand(B, S, 1)
        c = self.ctx(ctx)[:, None, :].expand(B, S, 32)
        h = self.tconv(torch.cat([f, c, pos], -1).transpose(1, 2)).transpose(1, 2)
        h = self.mix(h) if self.mlp else self.mix(h)[0]
        return self.heads(h)


class TokenTx(nn.Module):
    """Tokenised transformer: per-station CNN column -> token (+ sinusoidal PE), [CTX] token, pre-LN encoder, final LN."""
    def __init__(self, cin, nctx, d=96, layers=2, energy=False, patch=0):
        super().__init__()
        self.patch = patch
        if patch:
            self.embed = nn.Linear(patch * 32 * cin, d); n_tok = 96 // patch
        else:
            self.front = CNNFront(cin); self.embed = nn.Linear(192, d); n_tok = 96
        self.register_buffer('pe', sinusoid(n_tok, d)); self.ctx = nn.Linear(nctx, d)
        enc = nn.TransformerEncoderLayer(d, max(1, d // 32), 4 * d, dropout=0.1, batch_first=True, norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False); self.ln = nn.LayerNorm(d)
        self.heads = Heads(d, energy); self.up = nn.Linear(d, patch * d) if patch else None; self.d = d

    def forward(self, x, ctx):
        if self.patch:
            B, C, S, W = x.shape; p = self.patch
            tok = self.embed(x.permute(0, 2, 1, 3).reshape(B, S // p, p * C * W))
        else:
            tok = self.embed(self.front(x))
        h = torch.cat([self.ctx(ctx)[:, None, :], tok + self.pe[None]], 1)
        h = self.ln(self.enc(h))[:, 1:]
        if self.patch:
            B, T, d = h.shape; h = self.up(h).reshape(B, T * self.patch, d)
        return self.heads(h)


class LegacyTxLN(nn.Module):
    """Existing conv-transformer arm with a final LayerNorm added (A3); heads on its 96-d tokens."""
    def __init__(self, cin, nctx, energy=False):
        super().__init__()
        self.net = LegacyNet(cin, nctx, arch='tx', layers=2); self.ln = nn.LayerNorm(96); self.heads = Heads(96, energy)

    def forward(self, x, ctx):
        n = self.net; f = n.cnn(x); f = torch.cat([f.mean(-1), f.amax(-1)], 1).transpose(1, 2); f = F.gelu(n.lat(f))
        B, S, _ = f.shape; pos = torch.linspace(0, 1, S, device=x.device)[None, :, None].expand(B, S, 1)
        c = n.ctx(ctx)[:, None, :].expand(B, S, 32); h = n.tconv(torch.cat([f, c, pos], -1).transpose(1, 2)).transpose(1, 2)
        return self.heads(self.ln(n.mix(h + n.pos)))


def build(arch, cin, nctx, energy):
    if arch == 'gru': return GRUNet(cin, nctx, 64, 1, energy)
    if arch == 'gru120': return GRUNet(cin, nctx, 120, 1, energy)
    if arch == 'gru2': return GRUNet(cin, nctx, 96, 2, energy)
    if arch == 'mlp': return GRUNet(cin, nctx, 64, 1, energy, mlp=True)
    if arch == 'tx_conv': return LegacyTxLN(cin, nctx, energy)
    if arch.startswith('tx'):
        d, L = arch[2:].split('_'); return TokenTx(cin, nctx, int(d), int(L), energy)
    if arch.startswith('patch'):
        p, d, L = arch[5:].split('_'); return TokenTx(cin, nctx, int(d), int(L), energy, patch=int(p))
    raise ValueError(arch)


# ----------------------------------------------------------------------------- data + metrics
def dev_group(g): return int(hashlib.md5(g.encode()).hexdigest(), 16) % 5 == 0


def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    if len(y) == 0 or y.min() == y.max(): return float('nan')
    r = np.argsort(np.argsort(s)) + 1.0; n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def cell_auc(y, s, cells):
    ok = tot = 0.0
    for c in np.unique(cells):
        m = cells == c; yy, ss = y[m], s[m]
        if len(yy) == 0 or yy.min() == yy.max(): continue
        d = ss[yy == 1][:, None] - ss[yy == 0][None, :]; ok += (d > 0).sum() + 0.5 * (d == 0).sum(); tot += d.size
    return (ok / tot if tot else float('nan')), int(tot)


def metrics(s, d, m):
    out = {}; grp = d['group'][m].astype(str); prof = d['profile'][m].astype(int); des = prof >= 0
    gcell = np.array([f'{g}|{p}' for g, p in zip(grp, prof)])
    for lab in ('unsafe', 'fail'):
        y = d[lab][m].astype(float)
        out[f'P_{lab}'] = auc(y, s); out[f'W_{lab}'], out[f'Wn_{lab}'] = cell_auc(y, s, grp)
        out[f'G_{lab}'], out[f'Gn_{lab}'] = cell_auc(y[des], s[des], gcell[des]) if des.any() else (float('nan'), 0)
    # route choice: lowest-risk route per group vs random / oracle (fail)
    y = d['fail'][m].astype(float); picks, rand, orac = [], [], []
    for g in np.unique(grp):
        k = np.where(grp == g)[0]
        if len(k) < 2: continue
        picks.append(y[k[np.argmin(s[k])]]); rand.append(y[k].mean()); orac.append(y[k].min())
    out.update(pick_fail=float(np.mean(picks)) if picks else float('nan'), random_fail=float(np.mean(rand)) if rand else float('nan'),
               oracle_fail=float(np.mean(orac)) if orac else float('nan'), n_groups=len(picks))
    return out


def energy_metrics(pred_e, pred_t, d, m, ds):
    """Goal-reached held-out routes: log-RMSE of route energy/time at the last observed station, within-group Spearman,
    and the curse metric (pred/true at the lowest-predicted route per group)."""
    def spearman(x, y):
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 and rx.std() > 0 and ry.std() > 0 else float('nan')
    E, T = d['E'][m], d['T'][m]; ok = (d['fail'][m] == 0) & (d['unsafe'][m] == 0); grp = d['group'][m].astype(str)   # clean arrivals only (like-for-like with the clean-only analytic fit)
    out = {}
    for name, P, Y in (('E', pred_e, E), ('T', pred_t, T)):
        jstar = np.array([np.where(np.isfinite(r))[0].max() if np.isfinite(r).any() else -1 for r in Y])
        sel = ok & (jstar >= 20)
        yt = np.array([Y[i, jstar[i]] for i in np.where(sel)[0]]); yp = np.array([P[i, :jstar[i] + 1].sum() * ds[m][i] for i in np.where(sel)[0]])
        out[f'{name}_n'] = int(sel.sum())
        if sel.sum() < 5: continue
        out[f'{name}_log_rmse'] = float(np.sqrt(np.mean((np.log(yp + 1) - np.log(yt + 1)) ** 2)))
        out[f'{name}_mape'] = float(np.median(np.abs(yp - yt) / np.maximum(yt, 1e-6)))
        g = grp[sel]; rho = []; curse = []
        for gg in np.unique(g):
            k = np.where(g == gg)[0]
            if len(k) >= 3:
                r = spearman(yp[k], yt[k])
                if np.isfinite(r): rho.append(r)
                j = k[np.argmin(yp[k])]; curse.append((yp[j] / yt[j]) / np.mean(yp[k] / yt[k]))
        out[f'{name}_spearman_within'] = float(np.mean(rho)) if rho else float('nan')
        out[f'{name}_curse_ratio_at_argmin'] = float(np.mean(curse)) if curse else float('nan')
    return out


class Data:
    def __init__(self, path, mode, ctx_variant, vplane, energy, subsample=None, seed=0):
        d = np.load(path, allow_pickle=True); self.d = {k: d[k] for k in d.files if k != 'X'}
        X = d['X'].astype(np.float32); sp = self.d['split'].astype(str); grp = self.d['group'].astype(str)
        isdev = np.array([dev_group(g) for g in grp])
        self.fit = (sp == 'train') & (~isdev if mode == 'holdout' else True); self.dev = (sp == 'train') & isdev; self.test = sp != 'train'
        if subsample:  # equal-data control: keep a fraction of the TRAINING GROUPS
            rng = np.random.default_rng(seed); gs = np.unique(grp[self.fit]); keep = set(rng.choice(gs, int(len(gs) * subsample), replace=False))
            self.fit = self.fit & np.array([g in keep for g in grp])
        cols = CTX_COLS[ctx_variant]; ctx = self.d['ctx'].astype(np.float32)[:, cols]
        cont = [0, 1, 2, 3]
        if vplane:  # vx as a constant input plane (before the ones channel)
            X = np.concatenate([X, np.broadcast_to((self.d['ctx'][:, 0] / 6.0).astype(np.float32)[:, None, None, None], (len(X), 1, 96, 32))], 1)
            X[:, [4, 5]] = X[:, [5, 4]]; cont = [0, 1, 2, 3, 4]      # valid is now channel 5, vx plane channel 4
        mu = X[self.fit][:, cont].mean((0, 2, 3)); sd = X[self.fit][:, cont].std((0, 2, 3)) + 1e-6
        X[:, cont] = (X[:, cont] - mu[None, :, None, None]) / sd[None, :, None, None]
        self.norm = dict(mu=mu, sd=sd, cont_index=cont, channels=['elev_rel', 'grade', 'cross', 'speed'] + (['vx_plane'] if vplane else []) + ['valid'])
        self.X = torch.from_numpy(np.concatenate([X, np.ones((len(X), 1, 96, 32), np.float32)], 1)).to(DEV)
        self.ctx_mu, self.ctx_sd = ctx[self.fit].mean(0), ctx[self.fit].std(0) + 1e-6
        self.ctx = torch.from_numpy((ctx - self.ctx_mu) / self.ctx_sd).to(DEV); self.ctx_cols = cols
        self.ev = torch.from_numpy(self.d['event_idx'].astype(np.int64)).to(DEV); self.n = len(X)
        self.ds = (self.d['route_len'].astype(np.float32) / 95.0)
        self.energy = energy and 'E' in self.d
        if self.energy:
            E, T = self.d['E'].astype(np.float32), self.d['T'].astype(np.float32); ev = self.d['event_idx'].astype(int)
            dE = np.diff(np.concatenate([np.zeros((len(E), 1), np.float32), E], 1), axis=1); dT = np.diff(np.concatenate([np.zeros((len(T), 1), np.float32), T], 1), axis=1)
            j = np.arange(96)[None, :]; mask = np.isfinite(dE) & np.isfinite(dT) & ((ev[:, None] < 0) | (j < ev[:, None]))   # clean-driving prefix only
            self.dE = torch.from_numpy(np.nan_to_num(dE)).to(DEV); self.dT = torch.from_numpy(np.nan_to_num(dT)).to(DEV)
            self.emask = torch.from_numpy(mask).to(DEV); self.dsg = torch.from_numpy(self.ds).to(DEV)


def survival_nll(haz, ev):
    S = haz.shape[1]; idx = torch.arange(S, device=haz.device)[None, :]
    evc = torch.where(ev >= 0, ev, torch.full_like(ev, S - 1))[:, None]
    surv = (idx < evc) | ((ev < 0)[:, None] & (idx <= evc))
    nll = (F.softplus(haz) * surv).sum(1)
    return nll + torch.where(ev >= 0, F.softplus(-haz.gather(1, evc).squeeze(1)), torch.zeros_like(nll))


def energy_loss(out, D, k):
    dE, dT, m, ds = D.dE[k], D.dT[k], D.emask[k], D.dsg[k][:, None]
    pe, pt = out['e'] * ds, out['t'] * ds
    per = (F.huber_loss(torch.log1p(pe), torch.log1p(dE), reduction='none') + F.huber_loss(torch.log1p(pt), torch.log1p(dT), reduction='none')) * m
    per = per.sum(1) / m.sum(1).clamp_min(1)
    # route-level term at the last observed station
    last = (m.cumsum(1) * m).argmax(1); has = m.any(1)
    cum_pe, cum_pt = (pe * m).cumsum(1).gather(1, last[:, None]).squeeze(1), (pt * m).cumsum(1).gather(1, last[:, None]).squeeze(1)
    cum_e, cum_t = (dE * m).cumsum(1).gather(1, last[:, None]).squeeze(1), (dT * m).cumsum(1).gather(1, last[:, None]).squeeze(1)
    route = (F.huber_loss(torch.log1p(cum_pe), torch.log1p(cum_e), reduction='none') + F.huber_loss(torch.log1p(cum_pt), torch.log1p(cum_t), reduction='none')) * has
    return (per + route).mean()


def predict(model, D, bs=1024):
    model.eval(); z, e, t = [], [], []
    with torch.no_grad():
        for i in range(0, D.n, bs):
            o = model(D.X[i:i + bs], D.ctx[i:i + bs]); z.append(route_logit(o['haz']).float().cpu().numpy())
            if 'e' in o: e.append(o['e'].float().cpu().numpy()); t.append(o['t'].float().cpu().numpy())
    return np.concatenate(z), (np.concatenate(e) if e else None), (np.concatenate(t) if t else None)


def train_one(D, a, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    fi = np.where(D.fit)[0]; n = len(fi); bs = min(256, n)
    model = build(a.arch, D.X.shape[1], D.ctx.shape[1], D.energy).to(DEV)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wd = a.wd if a.wd is not None else (1e-4 if a.arch.startswith(('gru', 'mlp')) else 0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=wd)
    steps = max(a.epochs * (n // bs), 1); sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)
    fi_t = torch.from_numpy(fi).to(DEV); t0 = time.time(); step = 0
    while step < steps:
        model.train(); perm = fi_t[torch.randperm(n, device=DEV)]
        for i in range(0, n - bs + 1, bs):
            if step >= steps: break
            k = perm[i:i + bs]; out = model(D.X[k], D.ctx[k])
            loss = survival_nll(out['haz'], D.ev[k]).mean()
            if D.energy: loss = loss + a.energy * energy_loss(out, D, k)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            if step < steps - 1: sch.step()
            step += 1
    took = time.time() - t0
    z, e, t = predict(model, D)
    row = dict(arch=a.arch, world=a.world, ctx=a.ctx, vplane=a.vplane, energy=a.energy, lr=a.lr, wd=wd, epochs=a.epochs, seed=seed, params=nparam,
               secs=round(took, 1), n_fit=n, steps=step, final_loss=float(loss), dev=metrics(z[D.dev], D.d, D.dev), heldout=metrics(z[D.test], D.d, D.test))
    if D.energy:
        row['energy_heldout'] = energy_metrics(e[D.test], t[D.test], D.d, D.test, D.ds)
    return model, row, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True); ap.add_argument('--out', required=True); ap.add_argument('--world', default='?')
    ap.add_argument('--arch', default='gru'); ap.add_argument('--ctx', default='geom'); ap.add_argument('--vplane', action='store_true')
    ap.add_argument('--energy', type=float, default=0.0, help='lambda for the energy+time heads (0 = hazard only)')
    ap.add_argument('--mode', choices=['holdout', 'deploy'], default='holdout'); ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--seed0', type=int, default=0); ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--wd', type=float, default=None)
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--subsample', type=float, default=None); ap.add_argument('--tag', default=None)
    ap.add_argument('--save', action='store_true')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    tag = a.tag or f"{a.world}_{a.arch}_{a.ctx}{'_vp' if a.vplane else ''}{'_E%g' % a.energy if a.energy else ''}_lr{a.lr:g}_{a.mode}{'_sub%g' % a.subsample if a.subsample else ''}"
    D = Data(a.ds, a.mode, a.ctx, a.vplane, a.energy > 0, a.subsample, a.seed0)
    print(f'=== {tag}: fit {int(D.fit.sum())} dev {int(D.dev.sum())} heldout {int(D.test.sum())} energy {D.energy} cin {D.X.shape[1]} nctx {D.ctx.shape[1]}', flush=True)
    rows, logits = [], []
    for s in range(a.seed0, a.seed0 + a.seeds):
        model, row, z = train_one(D, a, s); rows.append(row); logits.append(z)
        h = row['heldout']; print(f"  s{s} W_fail {h['W_fail']:.3f} P_fail {h['P_fail']:.3f} G_fail {h['G_fail']:.3f} pick {h['pick_fail']:.3f} (rand {h['random_fail']:.3f}, oracle {h['oracle_fail']:.3f})"
              + (f"  E_rmse {row['energy_heldout'].get('E_log_rmse', float('nan')):.3f} E_rho {row['energy_heldout'].get('E_spearman_within', float('nan')):.3f}" if D.energy else '') + f"  {row['secs']}s {row['params']}p", flush=True)
        if a.save:
            torch.save(dict(state=model.state_dict(), arch=a.arch, model_kind='n2_arch_train', cin=D.X.shape[1], nctx=D.ctx.shape[1], ctx_cols=D.ctx_cols,
                            ctx_variant=a.ctx, vplane=a.vplane, energy=D.energy, norm=D.norm, ctx_mu=D.ctx_mu, ctx_sd=D.ctx_sd, layers=2), f'{a.out}/{tag}_s{s}.pt')
    ens = np.mean(logits, 0)
    np.savez_compressed(f'{a.out}/{tag}_logits.npz', id=D.d['id'], ensemble_logit=ens, member_logits=np.stack(logits), fit=D.fit, dev=D.dev, heldout=D.test,
                        **({'vx_anchor': D.d['vx_anchor'], 'anchor_frame': D.d['anchor_frame'], 'time_to_event_s': D.d['time_to_event_s'], 'rem_m': D.d['rem_m']} if 'vx_anchor' in D.d else {}))
    summary = dict(tag=tag, args=vars(a), members=rows, ensemble=dict(dev=metrics(ens[D.dev], D.d, D.dev), heldout=metrics(ens[D.test], D.d, D.test)))
    json.dump(summary, open(f'{a.out}/{tag}.json', 'w'), indent=1, default=float)
    h = summary['ensemble']['heldout']; print(f"ENSEMBLE {tag}: heldout W_fail {h['W_fail']:.4f} P_fail {h['P_fail']:.4f} G_fail {h['G_fail']:.4f} pick {h['pick_fail']:.3f}", flush=True)


if __name__ == '__main__':
    main()
