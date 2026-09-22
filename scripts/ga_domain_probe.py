"""Domain identifiability probes on the mixed re-anchored dataset (PLAN A1b).

(a) established probe: rows with anchor_frame > 0 whose (episode, anchor_frame) exists in BOTH domains (same anchors, so no
    selection confound between the two worlds). A causal GRU (15 -> 32) on the 2 s history window -> domain, trained on
    train-split groups; AUC on val and test groups, overall and by anchor-speed bin (vx < 1, 1-3, > 3 m/s). Baseline: a
    ridge-logistic on 5 hand features of the window (mean vx, mean throttle, throttle-to-vx ratio = mean throttle /
    max(mean vx, 0.1), mean spindle slip speed = omega - vx / 0.467 over wheels and steps, yaw-rate variance).
(b) startup probe: rows with anchor_frame == 0; per-column and joint ridge-logistic AUC of the frame-0 12-column state
    (ctx columns 0-6, 11-15), fitted on train groups and evaluated on val / test groups. The all-masked window itself is
    0.5 by definition and is not a finding.
Numpy-only metrics (tie-aware rank AUC, IRLS ridge logistic); torch only for the GRU. Works on any npz with the
mixed_reanchor keys (hist, hmask, domain, episode, anchor_frame, split, vx_anchor, ctx), e.g. the A4 branch rows later.

  PYTHONPATH=src:scripts python scripts/ga_domain_probe.py --dataset <mixed_reanchor.npz> --out <probe.json>
"""
import argparse, json, os, time
import numpy as np

STATE12 = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]
STATE12_NAMES = ['vx', 'vy', 'roll', 'pitch', 'roll_rate', 'pitch_rate', 'yaw_rate', 'omega_fl', 'omega_fr', 'omega_rl', 'omega_rr', 'engine_speed']
HAND_NAMES = ['mean_vx', 'mean_throttle', 'throttle_to_vx', 'mean_slip_speed', 'yaw_rate_var']
BINS = [('vx<1', -np.inf, 1.0), ('1-3', 1.0, 3.0), ('vx>3', 3.0, np.inf)]
TYRE_R = 0.467
HIST_NAMES = STATE12_NAMES + ['steer', 'throttle', 'brake']
ABLATION_GROUPS = {'actions': [12, 13, 14], 'body_vel': [0, 1], 'attitude': [2, 3], 'ang_rates': [4, 5, 6], 'omegas': [7, 8, 9, 10], 'engine': [11],
                   'state12': list(range(12)), 'no_engine': [c for c in range(15) if c != 11], 'no_omegas': [c for c in range(15) if c not in (7, 8, 9, 10)], 'vx_only': [0]}


def auc(score, y):
    """Mann-Whitney AUC with average ranks for ties; NaN if one class is absent."""
    y = np.asarray(y, bool); score = np.asarray(score, np.float64)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = np.argsort(score, kind='mergesort'); s = score[order]
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt); avg = (cum - cnt + 1 + cum) / 2.0
    ranks = avg[inv]
    return float((ranks[y[order]].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def logistic_fit(X, y, l2=1.0, iters=100):
    """Ridge logistic regression by IRLS on standardised features (bias unpenalised). Returns w (d+1,)."""
    Xb = np.c_[X, np.ones(len(X))]; w = np.zeros(Xb.shape[1]); y = y.astype(np.float64)
    pen = np.r_[np.ones(Xb.shape[1] - 1), 0.0] * l2
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xb @ w)); W = p * (1 - p) + 1e-9
        g = Xb.T @ (p - y) + pen * w
        H = (Xb * W[:, None]).T @ Xb + np.diag(pen)
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w


def logistic_score(X, w):
    return np.c_[X, np.ones(len(X))] @ w


def standardise(train, *others):
    mu = train.mean(0); sd = train.std(0) + 1e-6
    return [(x - mu) / sd for x in (train,) + others]


def by_bin(score, y, vx):
    out = {}
    for name, lo, hi in BINS:
        m = (vx >= lo) & (vx < hi)
        out[name] = dict(n=int(m.sum()), frac_crm=float(y[m].mean()) if m.any() else float('nan'), auc=auc(score[m], y[m]) if m.any() else float('nan'))
    return out


def hand_features(H, M):
    """H (n,T,15) float32, M (n,T) bool -> (n,5) features over valid steps."""
    w = M.astype(np.float32); cnt = np.maximum(w.sum(1), 1.0)
    mean = lambda x: (x * w).sum(1) / cnt
    vx = mean(H[:, :, 0]); thr = mean(H[:, :, 13])
    ratio = thr / np.maximum(vx, 0.1)
    slip = mean((H[:, :, 7:11] - H[:, :, 0:1] / TYRE_R).mean(2))
    yaw = H[:, :, 6]; yv = mean((yaw - mean(yaw)[:, None]) ** 2)
    return np.stack([vx, thr, ratio, slip, yv], 1)


def gru_probe(H, M, y, vx, split, seeds, epochs, device, hidden=32, lr=2e-3, bs=512):
    import torch, torch.nn as nn
    tr, va, te = (split == 'train'), (split == 'val'), (split == 'test')
    mu = (H[tr][M[tr]]).mean(0); sd = (H[tr][M[tr]]).std(0) + 1e-6
    Hn = ((H - mu) / sd * M[..., None]).astype(np.float32)
    Ht = torch.tensor(Hn, device=device); Mt = torch.tensor(M, device=device, dtype=torch.float32); yt = torch.tensor(y, device=device, dtype=torch.float32)

    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.gru = nn.GRU(15, hidden, batch_first=True); s.head = nn.Linear(hidden, 1)

        def forward(s, h, m):
            o, _ = s.gru(h * m[..., None]); return s.head(o[:, -1]).squeeze(-1)

    def predict(net, idx):
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(idx), 4096):
                j = idx[i:i + 4096]; out.append(net(Ht[j], Mt[j]).float().cpu().numpy())
        net.train(); return np.concatenate(out) if out else np.zeros(0)

    tri = np.flatnonzero(tr); vai = np.flatnonzero(va); tei = np.flatnonzero(te); res = []
    for seed in range(seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        net = Net().to(device); opt = torch.optim.Adam(net.parameters(), lr=lr); lossf = nn.BCEWithLogitsLoss()
        curve = []; best = (-1.0, None, None)
        for ep in range(epochs):
            perm = np.random.permutation(tri)
            for i in range(0, len(perm), bs):
                j = perm[i:i + bs]; opt.zero_grad(); loss = lossf(net(Ht[j], Mt[j]), yt[j]); loss.backward(); opt.step()
            sv = predict(net, vai); av = auc(sv, y[vai]); curve.append(round(av, 4))
            if av > best[0]:
                best = (av, ep, auc(predict(net, tei), y[tei]))
        s_tr, s_va, s_te = predict(net, tri), predict(net, vai), predict(net, tei)
        res.append(dict(seed=seed, train_auc=auc(s_tr, y[tri]), val_auc=auc(s_va, y[vai]), test_auc=auc(s_te, y[tei]),
                        val_by_bin=by_bin(s_va, y[vai], vx[vai]), test_by_bin=by_bin(s_te, y[tei], vx[tei]),
                        val_auc_curve=curve, best_val_epoch=best[1], best_val_auc=best[0], test_auc_at_best_val=best[2]))
        print(f'  GRU seed {seed}: train {res[-1]["train_auc"]:.4f} val {res[-1]["val_auc"]:.4f} test {res[-1]["test_auc"]:.4f}', flush=True)
    agg = {k: float(np.mean([r[k] for r in res])) for k in ['train_auc', 'val_auc', 'test_auc']}
    for sp in ['val_by_bin', 'test_by_bin']:
        agg[sp] = {b: dict(n=res[0][sp][b]['n'], frac_crm=res[0][sp][b]['frac_crm'], auc=float(np.nanmean([r[sp][b]['auc'] for r in res]))) for b in res[0][sp]}
    return dict(per_seed=res, mean=agg, hidden=hidden, lr=lr, epochs=epochs, batch=bs, n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(te.sum()))


def logistic_probe(F, y, vx, split, names, l2=1.0, per_column=True):
    tr, va, te = (split == 'train'), (split == 'val'), (split == 'test')
    Ftr, Fva, Fte = standardise(F[tr], F[va], F[te])
    w = logistic_fit(Ftr, y[tr], l2)
    s_va, s_te, s_tr = logistic_score(Fva, w), logistic_score(Fte, w), logistic_score(Ftr, w)
    out = dict(joint=dict(train_auc=auc(s_tr, y[tr]), val_auc=auc(s_va, y[va]), test_auc=auc(s_te, y[te]), coef=dict(zip(names, np.round(w[:-1], 4).tolist())),
                          val_by_bin=by_bin(s_va, y[va], vx[va]) if vx is not None else None, test_by_bin=by_bin(s_te, y[te], vx[te]) if vx is not None else None),
               n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(te.sum()), l2=l2)
    if per_column:
        cols = {}
        for j, nm in enumerate(names):
            wj = logistic_fit(Ftr[:, j:j + 1], y[tr], l2)
            cols[nm] = dict(val_auc=auc(logistic_score(Fva[:, j:j + 1], wj), y[va]), test_auc=auc(logistic_score(Fte[:, j:j + 1], wj), y[te]),
                            coef=float(wj[0]), mean_rigid=float(F[tr][y[tr] == 0, j].mean()), mean_crm=float(F[tr][y[tr] == 1, j].mean()))
        out['per_column'] = cols
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, default=3); ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--all-established', action='store_true', help='use every established row, not only the anchors present in both worlds (selection confound)')
    ap.add_argument('--device', default=None)
    ap.add_argument('--ablate', action='store_true', help='also train the GRU on channel subsets (1 seed each) and a last-step logistic to attribute the signal')
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.dataset, allow_pickle=True)   # X is never touched
    dom = z['domain'].astype(np.int8); ep = z['episode'].astype(str); k = z['anchor_frame'].astype(int); split = z['split'].astype(str)
    vx = z['vx_anchor'].astype(np.float32); ctx = z['ctx'].astype(np.float32); H = z['hist'].astype(np.float32); M = z['hmask'].astype(bool)
    y = (dom == 1).astype(np.int8)
    print(f'{len(y)} rows loaded in {time.time() - t0:.0f} s', flush=True)
    res = dict(dataset=os.path.abspath(a.dataset), rows=int(len(y)), pairs_only=not a.all_established)

    # (a) established
    est = k > 0
    if a.all_established:
        sel = est
    else:
        have = {}
        for e, kk, d in zip(ep[est], k[est], dom[est]):
            have.setdefault((e, int(kk)), set()).add(int(d))
        both = {p for p, s in have.items() if len(s) == 2}
        sel = est & np.array([(e, int(kk)) in both for e, kk in zip(ep, k)])
    idx = np.flatnonzero(sel)
    print(f'established probe on {len(idx)} rows ({int(y[idx].sum())} crm); splits train/val/test = '
          f'{(split[idx] == "train").sum()}/{(split[idx] == "val").sum()}/{(split[idx] == "test").sum()}', flush=True)
    import torch
    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    res['established'] = dict(n_rows=int(len(idx)), n_pairs=int(len(idx) // 2), frac_crm=float(y[idx].mean()),
                              speed_bins={b: dict(n=int(((vx[idx] >= lo) & (vx[idx] < hi)).sum()), frac_crm=float(y[idx][(vx[idx] >= lo) & (vx[idx] < hi)].mean()) if ((vx[idx] >= lo) & (vx[idx] < hi)).any() else float('nan')) for b, lo, hi in BINS},
                              gru=gru_probe(H[idx], M[idx], y[idx], vx[idx], split[idx], a.seeds, a.epochs, device),
                              hand_logistic=logistic_probe(hand_features(H[idx], M[idx]), y[idx], vx[idx], split[idx], HAND_NAMES))
    if a.ablate:
        # attribution: which channel groups carry the domain signal (1 seed, fewer epochs), plus a one-frame logistic on the
        # 15 channels of the last window step (instantaneous state vs temporal structure)
        abl = {}
        for name, ch in ABLATION_GROUPS.items():
            Hs = np.zeros_like(H[idx]); Hs[:, :, ch] = H[idx][:, :, ch]
            r = gru_probe(Hs, M[idx], y[idx], vx[idx], split[idx], 1, max(5, a.epochs // 2), device)['mean']
            abl[name] = dict(channels=ch, val_auc=r['val_auc'], test_auc=r['test_auc']); print(f'  ablation {name:12s} val {r["val_auc"]:.4f} test {r["test_auc"]:.4f}', flush=True)
        last = H[idx][:, -1, :]
        abl['last_step_logistic_15ch'] = {k: v for k, v in logistic_probe(last, y[idx], vx[idx], split[idx], HIST_NAMES)['joint'].items() if k in ('train_auc', 'val_auc', 'test_auc', 'coef')}
        w = M[idx].astype(np.float32)[..., None]; cnt = np.maximum(w.sum(1), 1.0)
        mean = (H[idx] * w).sum(1) / cnt; std = np.sqrt(((H[idx] - mean[:, None]) ** 2 * w).sum(1) / cnt)
        abl['window_mean_std_logistic_30'] = {k: v for k, v in logistic_probe(np.concatenate([mean, std], 1), y[idx], vx[idx], split[idx], [f'mean_{n}' for n in HIST_NAMES] + [f'std_{n}' for n in HIST_NAMES])['joint'].items() if k in ('train_auc', 'val_auc', 'test_auc', 'coef')}
        abl['window_std_logistic_15'] = {k: v for k, v in logistic_probe(std, y[idx], vx[idx], split[idx], [f'std_{n}' for n in HIST_NAMES])['joint'].items() if k in ('train_auc', 'val_auc', 'test_auc', 'coef')}
        res['established']['ablation'] = abl
    # (b) startup
    s0 = np.flatnonzero(k == 0)
    print(f'startup probe on {len(s0)} rows ({int(y[s0].sum())} crm)', flush=True)
    res['startup'] = dict(n_rows=int(len(s0)), frac_crm=float(y[s0].mean()),
                          logistic=logistic_probe(ctx[s0][:, STATE12], y[s0], None, split[s0], STATE12_NAMES))
    res['elapsed_s'] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)

    # short table
    g = res['established']['gru']['mean']; h = res['established']['hand_logistic']['joint']; st = res['startup']['logistic']
    print('\n| probe | val AUC | test AUC |'); print('|---|---|---|')
    print(f'| established GRU(15->32) on history, {a.seeds}-seed mean | {g["val_auc"]:.3f} | {g["test_auc"]:.3f} |')
    for b in g['val_by_bin']:
        print(f'|   GRU, anchor {b} (n val/test {g["val_by_bin"][b]["n"]}/{g["test_by_bin"][b]["n"]}, crm share {g["val_by_bin"][b]["frac_crm"]:.2f}/{g["test_by_bin"][b]["frac_crm"]:.2f}) | {g["val_by_bin"][b]["auc"]:.3f} | {g["test_by_bin"][b]["auc"]:.3f} |')
    print(f'| established hand-feature logistic (5 features) | {h["val_auc"]:.3f} | {h["test_auc"]:.3f} |')
    for b in h['val_by_bin']:
        print(f'|   logistic, anchor {b} | {h["val_by_bin"][b]["auc"]:.3f} | {h["test_by_bin"][b]["auc"]:.3f} |')
    for nm, r in res['established'].get('ablation', {}).items():
        lab = f'logistic ablation: {nm}' if 'logistic' in nm else f'GRU ablation: {nm} channels only (1 seed)'
        print(f'|   {lab} | {r["val_auc"]:.3f} | {r["test_auc"]:.3f} |')
    print(f'| startup joint logistic on frame-0 state (12 cols) | {st["joint"]["val_auc"]:.3f} | {st["joint"]["test_auc"]:.3f} |')
    for nm, c in st['per_column'].items():
        print(f'|   startup {nm} alone (train mean rigid/crm {c["mean_rigid"]:.3g}/{c["mean_crm"]:.3g}) | {c["val_auc"]:.3f} | {c["test_auc"]:.3f} |')
    print(f'\nwrote {a.out} in {res["elapsed_s"]} s', flush=True)


if __name__ == '__main__':
    main()
