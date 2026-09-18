"""Aggregate stage A/B/C JSON rows into tables (per world): mean +- sd over seeds and the 5-seed ensemble, held-out.
  python scripts/n2_arch_report.py <dir> [--vx]   (--vx: stratified read-out from *_logits.npz with vx_anchor)"""
import glob, json, os, sys
import numpy as np

d = sys.argv[1]; vx_mode = '--vx' in sys.argv
rows = []
for f in sorted(p for p in glob.glob(d + '/*.json') if not p.endswith('report_rows.json')):
    s = json.load(open(f))
    if s.get('skipped'): continue
    m = s['members']; a = s['args']
    h = lambda k: np.array([r['heldout'][k] for r in m], float)
    row = dict(world=a['world'], arch=a['arch'], ctx=a['ctx'], vplane=a.get('vplane', False), energy=a.get('energy', 0), lr=a['lr'], params=m[0]['params'],
               secs=np.mean([r['secs'] for r in m]), n=len(m), W=h('W_fail').mean(), Wsd=h('W_fail').std(), P=h('P_fail').mean(), G=h('G_fail').mean(),
               Wens=s['ensemble']['heldout']['W_fail'], Pens=s['ensemble']['heldout']['P_fail'], Gens=s['ensemble']['heldout']['G_fail'],
               pick=s['ensemble']['heldout']['pick_fail'], rand=s['ensemble']['heldout']['random_fail'], oracle=s['ensemble']['heldout']['oracle_fail'], tag=s['tag'])
    if 'energy_heldout' in m[0]:
        e = lambda k: np.nanmean([r['energy_heldout'].get(k, np.nan) for r in m])
        row.update(E_rmse=e('E_log_rmse'), E_rho=e('E_spearman_within'), E_curse=e('E_curse_ratio_at_argmin'), T_rmse=e('T_log_rmse'), T_rho=e('T_spearman_within'))
    if vx_mode and os.path.exists(f[:-5] + '_logits.npz'):
        L = np.load(f[:-5] + '_logits.npz', allow_pickle=True)
        if 'vx_anchor' in L:
            dsp = a['ds'] if os.path.exists(a['ds']) else os.path.join(os.path.dirname(d.rstrip('/')), 'datasets', os.path.basename(a['ds']))   # cluster-trained arms store the cluster path
            ds = np.load(dsp, allow_pickle=True); y = ds['fail'].astype(float); grp = ds['group'].astype(str); ho = L['heldout']; vx = L['vx_anchor']; z = L['ensemble_logit']
            tte = L['time_to_event_s']; k0 = L['anchor_frame'] == 0
            def wauc(mask):
                ok = tot = 0.0
                for g in np.unique(grp[mask]):
                    mm = mask & (grp == g); yy, ss = y[mm], z[mm]
                    if len(yy) < 2 or yy.min() == yy.max(): continue
                    dd = ss[yy == 1][:, None] - ss[yy == 0][None, :]; ok += (dd > 0).sum() + 0.5 * (dd == 0).sum(); tot += dd.size
                return ok / tot if tot else np.nan
            moving = ho & ~k0 & ((tte < 0) | (tte > 2.0))   # moving anchors, event not within 2 s (drop the trivially-stuck mass)
            row.update(W_start=wauc(ho & k0), W_mov=wauc(moving), W_v01=wauc(moving & (vx < 1)), W_v13=wauc(moving & (vx >= 1) & (vx < 3)), W_v36=wauc(moving & (vx >= 3)),
                       n_mov=int(moving.sum()))
    rows.append(row)
for w in sorted({r['world'] for r in rows}):
    print(f'\n=== world {w}  (held-out groups; W = within-group AUC for goal-not-reached, P pooled, G same-speed; pick = failure rate of the lowest-risk route per group)')
    keys = ['arch', 'ctx', 'vplane', 'energy', 'lr', 'params', 'secs', 'n', 'W', 'Wsd', 'Wens', 'Pens', 'Gens', 'pick', 'rand', 'oracle'] + (['E_rmse', 'E_rho', 'E_curse', 'T_rmse', 'T_rho'] if any('E_rmse' in r for r in rows) else []) + (['W_start', 'W_mov', 'W_v01', 'W_v13', 'W_v36', 'n_mov'] if vx_mode else [])
    print(' | '.join(f'{k:>7s}' for k in keys))
    for r in sorted([r for r in rows if r['world'] == w], key=lambda r: -r['Wens']):
        print(' | '.join((f'{r.get(k, float("nan")):7.3f}' if isinstance(r.get(k), (float, np.floating)) else f'{str(r.get(k, "")):>7s}') for k in keys))

if '--json' in sys.argv:   # machine-readable copy of the table rows (used by scripts/n2_figures.py)
    jp = sys.argv[sys.argv.index('--json') + 1]
    json.dump([{k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()} for r in rows], open(jp, 'w'), indent=1)
