"""How fast does the measured history identify the world (rigid vs CRM soil)? Window-length x decision-time probe (PLAN S0-W).

Rows: (episode, decision frame k) pairs that exist in BOTH worlds (twin episodes: identical commanded route, so the only
difference in the prefix is the physics response; the applied actions differ because the follower is closed loop).
Sources per k (hist (40, 15) cut at k, hmask = action frame >= 0, built by ga_build_mixed.cut_episode):
  0 < k < 40        --short file, default datasets/short_anchor.npz (scripts/ci_short_anchors.py; k = 10 / 20 / 30)
  k in {40, 80}      the k > 0 rows of generalist_20260921/A_adapt/datasets/mixed_reanchor.npz (primary, as briefed); note that
                     n2's speed-stratified anchor choice keeps k = 80 mostly for episodes with few admissible anchors
  k = 60             datasets/anchor_k40_60_80.npz (same builder; mixed_reanchor has only multiples of 40)
  --full-coverage    k in {40, 60, 80} all from anchor_k40_60_80.npz (every admitted anchor, no selection), as a check.
  --common           additionally restrict every k to the episodes present in both worlds at ALL ks (fixed population).
Window: the last L frames of the stored window (L in {5, 10, 20, 40} = 0.25 / 0.5 / 1 / 2 s); for L > k only k frames are
valid (standing start), so the effective window is min(L, k).
Models per (k, L) cell: causal GRU (15 -> 32, as ga_domain_probe.py) on the normalised window, read-out at the last step,
3 seeds, fixed 20 epochs (no early stopping, no selection on val); trained on train-split groups only, AUC read on the
val groups. Baselines: ridge logistic on the window means and standard deviations of the 15 channels over the valid steps
(30 features) and on the last step alone (15 channels at frame k). Test groups are dropped at load time unless
--report-test. Val AUC: mean over seeds (and min), seed-ensemble (mean logit) AUC with a 95 % bootstrap interval over val
groups (2,000 resamples; a group carries both worlds' rows).
Optional --ablate k:L [k:L ...]: GRU on channel subsets (ga_domain_probe.ABLATION_GROUPS + longitudinal [vx, pitch, throttle,
brake], no_rates, no_engine_brake, motion_no_actions; 1 seed) at those cells.

  PYTHONPATH=src:scripts OMP_NUM_THREADS=6 python scripts/ci_window_probe.py --out artifacts/traverse/crm_improve_20260922/scout/W_window
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ga_domain_probe import auc, logistic_fit, logistic_score, standardise, ABLATION_GROUPS, HIST_NAMES

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'artifacts', 'traverse')
K2 = ART + '/crm_improve_20260922'
DEFAULTS = dict(short=K2 + '/datasets/short_anchor.npz', mixed=ART + '/generalist_20260921/A_adapt/datasets/mixed_reanchor.npz',
                extra=K2 + '/datasets/anchor_k40_60_80.npz')
KEYS = ['hist', 'hmask', 'domain', 'episode', 'group', 'split', 'anchor_frame', 'vx_anchor']
BINS = [('vx<1', -np.inf, 1.0), ('1-3', 1.0, 3.0), ('vx>3', 3.0, np.inf)]
DT = 0.05
# channel-group ablations: the K1 groups plus physically motivated subsets (hist channels: 0 vx, 1 vy, 2 roll, 3 pitch,
# 4-6 roll/pitch/yaw rate, 7-10 wheel spin rates, 11 engine speed, 12 steer, 13 throttle, 14 brake)
ABL_GROUPS = dict(ABLATION_GROUPS, longitudinal=[0, 3, 13, 14], no_rates=[c for c in range(15) if c not in (4, 5, 6)],
                  no_engine_brake=[c for c in range(15) if c not in (11, 14)], motion_no_actions=list(range(11)))


def load(path, ks, keep_splits):
    z = np.load(path, allow_pickle=True)
    af = z['anchor_frame'].astype(int); sp = z['split'].astype(str)
    sel = np.isin(af, ks) & np.isin(sp, keep_splits)
    out = {}
    for k in KEYS:
        v = z[k]
        out[k] = v[sel]
        del v
    out['hist'] = out['hist'].astype(np.float32)
    for k in ('episode', 'group', 'split'): out[k] = out[k].astype(str)
    out['anchor_frame'] = out['anchor_frame'].astype(int); out['domain'] = out['domain'].astype(int)
    return out


def cat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in KEYS}


def paired(D, k, allowed=None):
    """Row indices at frame k whose episode has a row in both worlds (and is in `allowed` if given)."""
    m = D['anchor_frame'] == k
    ep = D['episode']; dm = D['domain']
    r = set(ep[m & (dm == 0)]); c = set(ep[m & (dm == 1)]); both = r & c
    if allowed is not None: both &= allowed
    idx = np.flatnonzero(m & np.isin(ep, sorted(both)))
    assert len(idx) == 2 * len(both), (k, len(idx), len(both))
    return idx, both


def group_boot(score, y, groups, n=2000, seed=0):
    """95 % percentile interval of the AUC under resampling of groups with replacement."""
    rng = np.random.default_rng(seed); ug, inv = np.unique(groups, return_inverse=True)
    rows = [np.flatnonzero(inv == g) for g in range(len(ug))]; vals = []
    for _ in range(n):
        pick = rng.integers(0, len(ug), len(ug)); j = np.concatenate([rows[g] for g in pick])
        a = auc(score[j], y[j])
        if np.isfinite(a): vals.append(a)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def by_bin(score, y, vx):
    out = {}
    for name, lo, hi in BINS:
        m = (vx >= lo) & (vx < hi)
        v = auc(score[m], y[m]) if m.any() else float('nan')
        out[name] = dict(n=int(m.sum()), frac_crm=round(float(y[m].mean()), 3) if m.any() else None, auc=v if np.isfinite(v) else None)
    return out


def gru_cell(H, M, y, split, seeds, epochs, device, report_test, hidden=32, lr=2e-3, bs=512):
    """Train `seeds` GRUs on train rows, score val (and test if asked). H (n, L, 15) float32, M (n, L) bool."""
    import torch, torch.nn as nn
    tr = split == 'train'
    v = H[tr][M[tr]]; mu = v.mean(0); sd = v.std(0) + 1e-6
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
            for i in range(0, len(idx), 8192):
                j = idx[i:i + 8192]; out.append(net(Ht[j], Mt[j]).float().cpu().numpy())
        net.train(); return np.concatenate(out) if out else np.zeros(0)

    tri = np.flatnonzero(tr); vai = np.flatnonzero(split == 'val'); tei = np.flatnonzero(split == 'test')
    per, s_val, s_test = [], [], []
    for seed in range(seeds):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        net = Net().to(device); opt = torch.optim.Adam(net.parameters(), lr=lr); lossf = nn.BCEWithLogitsLoss()
        for ep in range(epochs):
            perm = rng.permutation(tri)
            for i in range(0, len(perm), bs):
                j = perm[i:i + bs]; opt.zero_grad(); loss = lossf(net(Ht[j], Mt[j]), yt[j]); loss.backward(); opt.step()
        sv = predict(net, vai); s_val.append(sv)
        r = dict(seed=seed, train_auc=auc(predict(net, tri), y[tri]), val_auc=auc(sv, y[vai]))
        if report_test and len(tei):
            st = predict(net, tei); s_test.append(st); r['test_auc'] = auc(st, y[tei])
        per.append(r)
    del Ht, Mt, yt
    return per, np.mean(s_val, 0), (np.mean(s_test, 0) if s_test else None), vai, tei


def window_stats(H, M):
    w = M.astype(np.float32)[..., None]; cnt = np.maximum(w.sum(1), 1.0)
    mean = (H * w).sum(1) / cnt; std = np.sqrt(((H - mean[:, None]) ** 2 * w).sum(1) / cnt)
    return np.concatenate([mean, std], 1)


def logistic_cell(F, y, split, report_test):
    tr, va, te = split == 'train', split == 'val', split == 'test'
    Ftr, Fva, Fte = standardise(F[tr], F[va], F[te])
    w = logistic_fit(Ftr, y[tr], 1.0)
    out = dict(train_auc=auc(logistic_score(Ftr, w), y[tr]), val_auc=auc(logistic_score(Fva, w), y[va]))
    if report_test and te.any(): out['test_auc'] = auc(logistic_score(Fte, w), y[te])
    return out, logistic_score(Fva, w)


def first_cell(grid, ks, Ls, thr, key):
    """Earliest decision frame (then shortest window) whose value >= thr; None if never. grid[(k, L)] -> dict."""
    for k in ks:
        for L in Ls:
            v = grid.get((k, L), {}).get(key)
            if v is not None and v >= thr:
                return dict(k=k, decision_s=k * DT, L=L, window_s=L * DT, effective_window_s=min(L, k) * DT, value=round(v, 4))
    return None


def run_grid(D, ks, Ls, a, device, allowed=None, label='primary'):
    res = {}; pops = {}
    for k in ks:
        idx, both = paired(D, k, allowed)
        H = D['hist'][idx]; M = D['hmask'][idx]; y = (D['domain'][idx] == 1).astype(np.int8); split = D['split'][idx]
        grp = D['group'][idx]; vx = D['vx_anchor'][idx]
        pops[k] = dict(pairs=len(both), rows=int(len(idx)), train=int((split == 'train').sum()), val=int((split == 'val').sum()),
                       test=int((split == 'test').sum()), val_groups=int(len(set(grp[split == 'val']))),
                       vx_p50_rigid=round(float(np.median(vx[y == 0])), 3), vx_p50_crm=round(float(np.median(vx[y == 1])), 3),
                       valid_steps=int(M[0].sum()))
        va = split == 'val'
        # last-step logistic (frame k only): independent of L
        last_lr, _ = logistic_cell(H[:, -1, :], y, split, a.report_test)
        for L in Ls:
            t1 = time.time()
            Hl, Ml = np.ascontiguousarray(H[:, -L:]), np.ascontiguousarray(M[:, -L:])
            per, sv, st, vai, tei = gru_cell(Hl, Ml, y, split, a.seeds, a.epochs, device, a.report_test)
            assert np.array_equal(vai, np.flatnonzero(va))
            ens = auc(sv, y[va]); ci = group_boot(sv, y[va], grp[va], n=a.boot)
            lr, _ = logistic_cell(window_stats(Hl, Ml), y, split, a.report_test)
            cell = dict(k=k, L=L, eff_steps=int(Ml[0].sum()), gru_val_auc_mean=float(np.mean([r['val_auc'] for r in per])),
                        gru_val_auc_min=float(np.min([r['val_auc'] for r in per])), gru_val_auc_ens=ens, gru_val_auc_ens_ci95=ci,
                        gru_train_auc_mean=float(np.mean([r['train_auc'] for r in per])), gru_per_seed=per,
                        gru_val_by_speed=by_bin(sv, y[va], vx[va]),
                        logistic_meanstd_val_auc=lr['val_auc'], logistic_meanstd_train_auc=lr['train_auc'],
                        logistic_last_step_val_auc=last_lr['val_auc'], seconds=round(time.time() - t1, 1))
            if a.report_test:
                cell['gru_test_auc_mean'] = float(np.mean([r['test_auc'] for r in per])); cell['gru_test_auc_ens'] = auc(st, y[tei])
                cell['logistic_meanstd_test_auc'] = lr.get('test_auc'); cell['logistic_last_step_test_auc'] = last_lr.get('test_auc')
            res[(k, L)] = cell
            print(f'  [{label}] k {k:3d} ({k * DT:.2f} s) L {L:2d} (eff {cell["eff_steps"]:2d}) pairs {len(both):5d} | GRU val mean {cell["gru_val_auc_mean"]:.4f} '
                  f'min {cell["gru_val_auc_min"]:.4f} ens {ens:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | logistic mean/std {lr["val_auc"]:.4f} last-step {last_lr["val_auc"]:.4f} '
                  f'| {cell["seconds"]} s', flush=True)
    return res, pops


def summarise(grid, ks, Ls):
    s = {}
    for thr in (0.95, 0.99):
        s[f'{thr}'] = dict(gru_seed_mean=first_cell(grid, ks, Ls, thr, 'gru_val_auc_mean'), gru_every_seed=first_cell(grid, ks, Ls, thr, 'gru_val_auc_min'),
                           gru_ensemble=first_cell(grid, ks, Ls, thr, 'gru_val_auc_ens'),
                           gru_ensemble_ci_lower=first_cell({kk: dict(v=v['gru_val_auc_ens_ci95'][0]) for kk, v in grid.items()}, ks, Ls, thr, 'v'),
                           logistic_meanstd=first_cell(grid, ks, Ls, thr, 'logistic_meanstd_val_auc'),
                           per_k_shortest_L_gru_mean={int(k): next((L for L in Ls if grid[(k, L)]['gru_val_auc_mean'] >= thr), None) for k in ks})
    return s


def md_table(grid, pops, ks, Ls, key, fmt='{:.3f}'):
    lines = ['| decision frame k (time) | pairs (val) | ' + ' | '.join(f'L = {L} ({L * DT:.2f} s)' for L in Ls) + ' |', '|---|---|' + '---|' * len(Ls)]
    for k in ks:
        row = []
        for L in Ls:
            v = grid[(k, L)][key]
            row.append(fmt.format(v) + ('*' if L > k else ''))
        lines.append(f'| {k} ({k * DT:.2f} s) | {pops[k]["pairs"]} ({pops[k]["val"] // 2}) | ' + ' | '.join(row) + ' |')
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--short', default=DEFAULTS['short']); ap.add_argument('--mixed', default=DEFAULTS['mixed']); ap.add_argument('--extra', default=DEFAULTS['extra'])
    ap.add_argument('--ks', type=int, nargs='+', default=[10, 20, 30, 40, 60, 80]); ap.add_argument('--Ls', type=int, nargs='+', default=[5, 10, 20, 40])
    ap.add_argument('--seeds', type=int, default=3); ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--full-coverage', action='store_true', help='also run k in {40, 60, 80} from anchor_k40_60_80.npz (all admitted anchors)')
    ap.add_argument('--common', action='store_true', help='also run on the episodes paired at every k (fixed population; short + extra sources)')
    ap.add_argument('--ablate', nargs='*', default=[], help='k:L cells for channel-group ablations (1 seed)')
    ap.add_argument('--report-test', action='store_true', help='also score the sealed test groups (default: test rows are dropped at load)')
    ap.add_argument('--out', required=True, help='output stem: writes <out>.json and <out>.md')
    ap.add_argument('--device', default=None)
    a = ap.parse_args()
    t0 = time.time()
    import torch
    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(min(1.0, 5.5e9 / torch.cuda.get_device_properties(0).total_memory))
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '6')))
    splits = ['train', 'val', 'test'] if a.report_test else ['train', 'val']
    ks = sorted(a.ks); Ls = sorted(a.Ls)
    k_short = [k for k in ks if 0 < k < 40]; k_mixed = [k for k in ks if k > 0 and k % 40 == 0]; k_extra = [k for k in ks if k >= 40 and k % 40 != 0]
    parts = [load(a.short, k_short, splits), load(a.mixed, k_mixed, splits)]
    if k_extra: parts.append(load(a.extra, k_extra, splits))
    D = cat(parts)
    src = {int(k): os.path.basename(a.short if k in k_short else a.mixed if k in k_mixed else a.extra) for k in ks}
    assert set(np.unique(D['anchor_frame'])) == set(ks), (np.unique(D['anchor_frame']), ks)
    if not a.report_test: assert not np.any(D['split'] == 'test')
    print(f'{len(D["domain"])} rows loaded ({", ".join(f"k {k}: {src[k]}" for k in ks)}) in {time.time() - t0:.0f} s; splits {splits}', flush=True)

    out = dict(sources=dict(short=a.short, mixed=a.mixed, extra=a.extra, per_k=src), ks=ks, Ls=Ls, seeds=a.seeds, epochs=a.epochs, boot=a.boot,
               report_test=a.report_test, model='GRU(15->32) + linear head at the last step, Adam 2e-3, batch 512, fixed epochs, train-split normalisation',
               baselines=['ridge logistic (l2 1) on window means + stds of 15 channels over valid steps (30 features)', 'ridge logistic on the last step (15 channels)'])
    grid, pops = run_grid(D, ks, Ls, a, device, label='primary')
    out['primary'] = dict(populations={int(k): v for k, v in pops.items()}, cells=[grid[(k, L)] for k in ks for L in Ls], thresholds=summarise(grid, ks, Ls))
    variants = {}
    if a.full_coverage or a.common:
        kx = [k for k in ks if k >= 40]
        Dx = cat([load(a.short, k_short, splits), load(a.extra, kx, splits)]) if kx else None
    if a.full_coverage and kx:
        g2, p2 = run_grid(Dx, kx, Ls, a, device, label='full-coverage')
        variants['full_coverage'] = dict(note='k >= 40 from anchor_k40_60_80.npz: every admitted anchor, no speed-stratified selection',
                                         populations={int(k): v for k, v in p2.items()}, cells=[g2[(k, L)] for k in kx for L in Ls])
        variants['full_coverage']['grid'] = g2; variants['full_coverage']['pops'] = p2
    if a.common:
        sets = [paired(Dx, k)[1] for k in ks]
        common = set.intersection(*sets)
        g3, p3 = run_grid(Dx, ks, Ls, a, device, allowed=common, label='common')
        variants['common'] = dict(note=f'episodes paired at every k in {ks} ({len(common)} episodes); k >= 40 from anchor_k40_60_80.npz',
                                  n_episodes=len(common), populations={int(k): v for k, v in p3.items()}, cells=[g3[(k, L)] for k in ks for L in Ls],
                                  thresholds=summarise(g3, ks, Ls))
        variants['common']['grid'] = g3; variants['common']['pops'] = p3
    abl = {}
    for spec in a.ablate:
        k, L = map(int, spec.split(':'))
        idx, _ = paired(D, k)
        H = np.ascontiguousarray(D['hist'][idx][:, -L:]); M = np.ascontiguousarray(D['hmask'][idx][:, -L:]); y = (D['domain'][idx] == 1).astype(np.int8); split = D['split'][idx]
        r = {}
        for name, ch in ABL_GROUPS.items():
            Hs = np.zeros_like(H); Hs[:, :, ch] = H[:, :, ch]
            per, sv, _, vai, _ = gru_cell(Hs, M, y, split, 1, a.epochs, device, False)
            r[name] = dict(channels=[HIST_NAMES[c] for c in ch], val_auc=per[0]['val_auc'])
            print(f'  ablation k {k} L {L} {name:10s} val {per[0]["val_auc"]:.4f}', flush=True)
        abl[f'{k}:{L}'] = r
    if abl: out['ablation'] = abl
    out['elapsed_s'] = round(time.time() - t0, 1)
    if device == 'cuda': out['gpu_peak_mem_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    # ---- markdown tables (the prose note is written separately)
    lines = [f'Command: `PYTHONPATH=src:scripts OMP_NUM_THREADS=6 python scripts/ci_window_probe.py {" ".join(sys.argv[1:])}`', '',
             f'Sources per k: ' + ', '.join(f'k {k}: {v}' for k, v in src.items()) + '. * = window longer than the time since the start (only k frames valid).', '',
             '### GRU val AUC, mean of 3 seeds', ''] + md_table(grid, pops, ks, Ls, 'gru_val_auc_mean', '{:.4f}') + \
            ['', '### GRU val AUC, worst seed', ''] + md_table(grid, pops, ks, Ls, 'gru_val_auc_min', '{:.4f}') + \
            ['', '### GRU seed-ensemble val AUC, 95 % group-bootstrap lower bound', ''] + \
            md_table({kk: dict(v=v['gru_val_auc_ens_ci95'][0]) for kk, v in grid.items() if isinstance(kk[0], int)}, pops, ks, Ls, 'v', '{:.4f}') + \
            ['', '### Logistic on window means + stds (30 features), val AUC', ''] + md_table(grid, pops, ks, Ls, 'logistic_meanstd_val_auc') + \
            ['', '### Last-step logistic (state + action at frame k only), val AUC', '', '| k | ' + ' | '.join(str(k) for k in ks) + ' |', '|---|' + '---|' * len(ks),
             '| val AUC | ' + ' | '.join(f'{grid[(k, Ls[0])]["logistic_last_step_val_auc"]:.3f}' for k in ks) + ' |']
    if a.report_test:
        lines += ['', '### GRU TEST AUC, mean of 3 seeds (sealed groups, read once)', ''] + md_table(grid, pops, ks, Ls, 'gru_test_auc_mean', '{:.4f}')
    for name in ('full_coverage', 'common'):
        if name in variants:
            v = variants[name]; kk = sorted(v['pops'])
            lines += ['', f'### Variant {name}: GRU val AUC mean of 3 seeds ({v["note"]})', ''] + md_table(v['grid'], v['pops'], kk, Ls, 'gru_val_auc_mean', '{:.4f}')
            lines += ['', f'### Variant {name}: logistic on window means + stds, val AUC', ''] + md_table(v['grid'], v['pops'], kk, Ls, 'logistic_meanstd_val_auc')
    for spec, r in abl.items():
        lines += ['', f'### Channel-group ablation at k:L = {spec} (GRU, 1 seed, val AUC)', '', '| channels kept | val AUC |', '|---|---|'] + \
                 [f'| {n} ({", ".join(v["channels"])}) | {v["val_auc"]:.4f} |' for n, v in r.items()]
    for name in list(variants):
        variants[name].pop('grid', None); variants[name].pop('pops', None)
    out['variants'] = variants
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + '.json', 'w') as f: json.dump(out, f, indent=1)
    with open(a.out + '_tables.md', 'w') as f: f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(json.dumps(out['primary']['thresholds'], indent=1))
    print(f'wrote {a.out}.json and {a.out}_tables.md in {out["elapsed_s"]} s', flush=True)


if __name__ == '__main__':
    main()
