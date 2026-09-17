"""Task B part 1 (corrected) + the corridor-level identifiability the per-cell view misses.

Two points of the SAME route share d0 and s0, so within a route (dd, s) determines z - z0 exactly. Pooling
same-route pairs therefore understates the ambiguity. Here every within-bin spread is computed ACROSS GROUPS
(distinct start/goal cases): one value per group per bin, then the spread of those.

Also tested: the corridor is a field, not one cell. If the terrain under the corridor were flat at z0,
    dd_i = (H - z0) (s_i - s0),
so the least-squares slope of dd against (s - s0) over the 96x32 cells estimates Q = H - z0, i.e. recovers the
omitted d0 = Q*s0 from the D arm's own channels. We run that estimator and report the height error it leaves.
"""
import glob, json, os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
V1 = ROOT / 'artifacts/traverse/fdm_f104_50h_20260909/sensor_v1'
TAGS = ['f104', 'g203', 'g216', 'g217', 'g228', 'g231']
H = 110.0
DD_BIN, S_BIN = 0.02, 2e-4
ROUTE_STRIDE = 2


def route_group_index(tag):
    files = []
    for wave in ('test', 'test2'):
        files += sorted(glob.glob(str(V1 / f'{wave}_{tag}' / 'routes' / '*.json')))
    files = files[::ROUTE_STRIDE]
    scenes = [os.path.basename(f).split('__')[0] for f in files]
    uniq = {s: i for i, s in enumerate(sorted(set(scenes)))}
    return np.array([uniq[s] for s in scenes], np.int64), len(uniq)


def load(tag):
    z = np.load(HERE / 'points' / f'{tag}.npz', allow_pickle=True)
    d = {k: z[k].astype(np.float64) for k in ('dep', 'sec', 'd0', 's0', 'zpix', 'gx', 'gy', 'route')}
    keep = (np.abs(d['gx']) < 39.5) & (np.abs(d['gy']) < 39.5)
    d = {k: v[keep] for k, v in d.items()}
    r2g, ngroups = route_group_index(tag)
    d['group'] = r2g[d['route'].astype(int)]
    d['ngroups'] = ngroups
    d['dd'] = d['dep'] - d['d0']
    d['z0'] = H - d['d0'] / d['s0']
    d['zrel'] = d['zpix'] - d['z0']
    return d


def crossgroup_bin_stats(keys, group, y, min_groups=2):
    """One value per (bin, group) [the group's mean y in that bin], then the spread across groups inside a bin."""
    kk = list(keys) + [group]
    order = np.lexsort(tuple(kk[::-1]))
    k = np.stack(kk)[:, order]; ys = y[order]
    new = np.r_[True, np.any(k[:, 1:] != k[:, :-1], axis=0)]
    st = np.flatnonzero(new); en = np.r_[st[1:], len(ys)]
    cs = np.r_[0.0, np.cumsum(ys)]
    gmean = (cs[en] - cs[st]) / (en - st)                      # per (bin, group)
    kb = k[:-1, st]                                            # bin key of each (bin, group) cell
    new2 = np.r_[True, np.any(kb[:, 1:] != kb[:, :-1], axis=0)]
    st2 = np.flatnonzero(new2); en2 = np.r_[st2[1:], len(gmean)]
    n = en2 - st2
    c1 = np.r_[0.0, np.cumsum(gmean)]; c2 = np.r_[0.0, np.cumsum(gmean ** 2)]
    m = (c1[en2] - c1[st2]) / n
    var = np.maximum((c2[en2] - c2[st2]) / n - m ** 2, 0.0)
    rng = np.maximum.reduceat(gmean, st2) - np.minimum.reduceat(gmean, st2)
    sel = n >= min_groups
    if not sel.any():
        return dict(n_bins=int(len(n)), n_bins_multigroup=0)
    w = n[sel].astype(float)
    pooled = float(np.sqrt(np.sum(var[sel] * w) / w.sum()))
    o = np.argsort(rng[sel]); rr = rng[sel][o]; ww = w[o]
    c = np.cumsum(ww) - .5 * ww
    q = lambda p: float(np.interp(p * ww.sum(), c, rr))
    return dict(n_bins=int(len(n)), n_bins_multigroup=int(sel.sum()),
                groups_per_bin_mean=float(w.mean()), groups_per_bin_max=int(n.max()),
                pooled_crossgroup_std_m=pooled,
                crossgroup_range_p50_m=q(.5), crossgroup_range_p90_m=q(.9),
                crossgroup_range_p99_m=q(.99), crossgroup_range_max_m=float(rr.max()))


def worst_pairs(d, keys, n=5):
    order = np.lexsort(tuple(list(keys)[::-1]))
    k = np.stack(keys)[:, order]; y = d['zrel'][order]; g = d['group'][order]
    new = np.r_[True, np.any(k[:, 1:] != k[:, :-1], axis=0)]
    st = np.flatnonzero(new); en = np.r_[st[1:], len(y)]
    out = []
    gaps = []
    for a, b in zip(st, en):
        if b - a < 2:
            gaps.append(0.0); continue
        gg = g[a:b]
        if gg.min() == gg.max():
            gaps.append(0.0); continue
        gaps.append(float(y[a:b].max() - y[a:b].min()))
    gaps = np.asarray(gaps)
    for b in np.argsort(-gaps)[:n]:
        a, e = st[b], en[b]
        i = order[a + int(np.argmax(y[a:e]))]; j = order[a + int(np.argmin(y[a:e]))]
        out.append({kk: [round(float(d[kk][i]), 4), round(float(d[kk][j]), 4)] for kk in
                    ('dd', 'sec', 's0', 'd0', 'z0', 'zpix', 'zrel', 'gx', 'gy')} |
                   {'group': [int(d['group'][i]), int(d['group'][j])],
                    'true_height_gap_m': round(float(abs(d['zrel'][i] - d['zrel'][j])), 4)})
    return out


def corridor_estimator(d):
    """Per route, estimate Q = H - z0 by least squares of dd on (s - s0), then rebuild z - z0."""
    r = d['route'].astype(int)
    o = np.argsort(r, kind='stable')
    rs = r[o]
    st = np.flatnonzero(np.r_[True, rs[1:] != rs[:-1]]); en = np.r_[st[1:], len(rs)]
    x = (d['sec'] - d['s0'])[o]; yv = d['dd'][o]
    sx = np.add.reduceat(x, st); sy = np.add.reduceat(yv, st)
    sxx = np.add.reduceat(x * x, st); sxy = np.add.reduceat(x * yv, st)
    n = (en - st).astype(float)
    den = sxx - sx * sx / n
    slope = np.where(den > 1e-12, (sxy - sx * sy / n) / np.maximum(den, 1e-12), np.nan)
    Qtrue = np.add.reduceat((H - d['z0'])[o], st) / n
    Qhat = np.repeat(slope, en - st)
    zrel_hat = Qhat * (1 - d['s0'][o] / d['sec'][o]) - d['dd'][o] / d['sec'][o]
    err = zrel_hat - d['zrel'][o]
    fin = np.isfinite(err)
    return dict(n_routes=int(len(st)),
                Q_true_mean_m=float(Qtrue.mean()), Q_hat_bias_m=float(np.nanmean(slope - Qtrue)),
                Q_hat_rmse_m=float(np.sqrt(np.nanmean((slope - Qtrue) ** 2))),
                Q_hat_p95_abs_err_m=float(np.nanquantile(np.abs(slope - Qtrue), .95)),
                sec_std_within_corridor_mean=float(np.mean([x[a:b].std() for a, b in zip(st, en)])),
                height_rmse_m=float(np.sqrt(np.mean(err[fin] ** 2))),
                height_mae_m=float(np.mean(np.abs(err[fin]))),
                height_p95_abs_m=float(np.quantile(np.abs(err[fin]), .95)),
                height_max_abs_m=float(np.abs(err[fin]).max()),
                r2_vs_true=float(1 - np.var(err[fin]) / np.var(d['zrel'][o][fin])))


def main():
    res = {'method': 'within-bin spread measured across GROUPS (one mean per group per bin); same-route pairs excluded',
           'bins': {'delta_d_m': DD_BIN, 'sec': S_BIN}, 'per_arena': {}}
    per = {}
    for t in TAGS:
        d = load(t); per[t] = d
        kd = np.round(d['dd'] / DD_BIN).astype(np.int64)
        ks = np.round(d['sec'] / S_BIN).astype(np.int64)
        k0 = np.round(d['s0'] / S_BIN).astype(np.int64)
        res['per_arena'][t] = {
            'n_points': int(len(d['dd'])), 'n_groups': int(d['ngroups']),
            'true_relative_height_std_m': float(d['zrel'].std()),
            'crossgroup_match_dd_sec': crossgroup_bin_stats([kd, ks], d['group'], d['zrel']),
            'crossgroup_match_dd_sec_s0': crossgroup_bin_stats([kd, ks, k0], d['group'], d['zrel']),
            'worst_pairs_same_dd_sec_different_group': worst_pairs(d, [kd, ks]),
            'corridor_slope_estimator_of_d0': corridor_estimator(d),
        }
        r = res['per_arena'][t]
        print(f"{t}: zrel std {r['true_relative_height_std_m']:.3f} | cross-group (dd,sec) std "
              f"{r['crossgroup_match_dd_sec']['pooled_crossgroup_std_m']:.3f} m, max gap "
              f"{r['crossgroup_match_dd_sec']['crossgroup_range_max_m']:.2f} m | +s0 std "
              f"{r['crossgroup_match_dd_sec_s0']['pooled_crossgroup_std_m']:.3f} m "
              f"(bins {r['crossgroup_match_dd_sec_s0']['n_bins_multigroup']}) | corridor-slope estimator "
              f"Q rmse {r['corridor_slope_estimator_of_d0']['Q_hat_rmse_m']:.3f} m -> height rmse "
              f"{r['corridor_slope_estimator_of_d0']['height_rmse_m']:.3f} m", flush=True)

    allk = {k: np.concatenate([per[t][k] for t in TAGS]) for k in ('dd', 'sec', 's0', 'zrel')}
    gshift = np.concatenate([per[t]['group'] + 1000 * i for i, t in enumerate(TAGS)])
    kd = np.round(allk['dd'] / DD_BIN).astype(np.int64)
    ks = np.round(allk['sec'] / S_BIN).astype(np.int64)
    k0 = np.round(allk['s0'] / S_BIN).astype(np.int64)
    res['pooled_six_arenas'] = {
        'n_points': int(len(kd)), 'true_relative_height_std_m': float(allk['zrel'].std()),
        'crossgroup_match_dd_sec': crossgroup_bin_stats([kd, ks], gshift, allk['zrel']),
        'crossgroup_match_dd_sec_s0': crossgroup_bin_stats([kd, ks, k0], gshift, allk['zrel'])}
    json.dump(res, open(HERE / 'ambiguity_crossgroup.json', 'w'), indent=1)
    print(json.dumps(res['pooled_six_arenas'], indent=1))
    print('wrote', HERE / 'ambiguity_crossgroup.json')


if __name__ == '__main__':
    main()
