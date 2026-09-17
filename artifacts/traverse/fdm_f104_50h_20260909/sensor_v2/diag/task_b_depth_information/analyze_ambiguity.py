"""Task B parts 1 and 2: is (delta_d, sec) enough for relative height, and how big is the omitted d0?

Algebra under test (review finding 2):
    z_i = H - d_i / s_i,   z_0 = H - d_0 / s_0
    z_i - z_0 = d_0 / s_0 - (d_0 + dd_i) / s_i = d_0 (1/s_0 - 1/s_i) - dd_i / s_i
              = (H - z_0) (1 - s_0 / s_i) - dd_i / s_i
    d(z_i - z_0) / d(d_0) = 1/s_0 - 1/s_i  -> zero only where s_i = s_0.

So we quantise (dd, s) [and optionally s_0] into bins much finer than the sensor, and measure the spread of the TRUE
relative height inside a bin. Any spread above the bin's own width is information the D arm cannot have.

Read-only; inputs are points/*.npz from extract_corridor_points.py.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TAGS = ['f104', 'g203', 'g216', 'g217', 'g228', 'g231']
H = 110.0
DD_BIN, S_BIN = 0.02, 2e-4          # 2 cm of range; 2e-4 of sec  (ray-angle bin ~ 0.012 deg)


def load(tag):
    z = np.load(HERE / 'points' / f'{tag}.npz', allow_pickle=True)
    d = {k: z[k].astype(np.float64) for k in ('dep', 'sec', 'd0', 's0', 'zpix', 'gx', 'gy', 'route', 'station', 'lateral')}
    keep = (np.abs(d['gx']) < 39.5) & (np.abs(d['gy']) < 39.5)          # drop arena-edge depth mixing
    d = {k: v[keep] for k, v in d.items()}
    d['dd'] = d['dep'] - d['d0']
    d['z0'] = H - d['d0'] / d['s0']
    d['zrel'] = d['zpix'] - d['z0']
    d['tag'] = np.full(len(d['dep']), TAGS.index(tag))
    return d


def within_bin_stats(keys, y, min_count=2):
    """Group y by the integer key rows and return per-bin count/std/range plus pooled numbers."""
    order = np.lexsort(tuple(keys[::-1]))
    k = np.stack(keys)[:, order]
    ys = y[order]
    new = np.r_[True, np.any(k[:, 1:] != k[:, :-1], axis=0)]
    start = np.flatnonzero(new)
    cnt = np.diff(np.r_[start, len(ys)])
    csum = np.r_[0.0, np.cumsum(ys)]
    csq = np.r_[0.0, np.cumsum(ys * ys)]
    ends = np.r_[start[1:], len(ys)]
    s1 = csum[ends] - csum[start]
    s2 = csq[ends] - csq[start]
    mean = s1 / cnt
    var = np.maximum(s2 / cnt - mean ** 2, 0.0)
    mx = np.maximum.reduceat(ys, start)
    mn = np.minimum.reduceat(ys, start)
    sel = cnt >= min_count
    # pooled within-bin variance, weighted by points in multi-point bins
    pooled = float(np.sum(var[sel] * cnt[sel]) / cnt[sel].sum()) if sel.any() else float('nan')
    rng = (mx - mn)[sel]
    w = cnt[sel]
    q = lambda p: float(weighted_quantile(rng, w, p)) if sel.any() else float('nan')
    return dict(n_bins=int(len(cnt)), n_bins_multi=int(sel.sum()),
                points_in_multi_bins=int(w.sum()), frac_points_in_multi_bins=float(w.sum() / len(ys)),
                pooled_within_bin_std_m=float(np.sqrt(pooled)),
                bin_range_p50_m=q(.50), bin_range_p90_m=q(.90), bin_range_p99_m=q(.99),
                bin_range_max_m=float(rng.max()) if sel.any() else float('nan')), (order, start, ends, cnt, mn, mx)


def weighted_quantile(v, w, p):
    o = np.argsort(v); v = v[o]; w = w[o].astype(float)
    c = np.cumsum(w) - 0.5 * w
    return np.interp(p * w.sum(), c, v)


def example_pairs(d, n=6, min_gap=0.5):
    """Concrete corridor-point pairs with the same (dd, sec) to 2 cm / 2e-4 but different true height."""
    kd = np.round(d['dd'] / DD_BIN).astype(np.int64)
    ks = np.round(d['sec'] / S_BIN).astype(np.int64)
    order = np.lexsort((ks, kd))
    kd, ks, y = kd[order], ks[order], d['zrel'][order]
    new = np.r_[True, (kd[1:] != kd[:-1]) | (ks[1:] != ks[:-1])]
    start = np.flatnonzero(new); ends = np.r_[start[1:], len(y)]
    out = []
    mx = np.maximum.reduceat(y, start); mn = np.minimum.reduceat(y, start)
    big = np.argsort(-(mx - mn))
    for b in big[:n]:
        sl = slice(start[b], ends[b])
        i = order[sl][np.argmax(y[sl])]; j = order[sl][np.argmin(y[sl])]
        if abs(d['zrel'][i] - d['zrel'][j]) < min_gap:
            break
        out.append({k: [round(float(d[k][i]), 4), round(float(d[k][j]), 4)] for k in
                    ('dd', 'sec', 'd0', 's0', 'z0', 'dep', 'zpix', 'zrel', 'gx', 'gy')} |
                   {'route_idx': [int(d['route'][i]), int(d['route'][j])],
                    'station': [int(d['station'][i]), int(d['station'][j])],
                    'true_height_gap_m': round(float(abs(d['zrel'][i] - d['zrel'][j])), 4)})
    return out


def main():
    res = {'algebra': {
        'identity': 'z_i - z_0 = d_0*(1/s_0 - 1/s_i) - dd_i/s_i = (H - z_0)*(1 - s_0/s_i) - dd_i/s_i',
        'sensitivity_to_omitted_d0': 'd(z_i - z_0)/d(d_0) = 1/s_0 - 1/s_i',
        'determined_iff': 's_i == s_0 (corridor point on the same camera ray cone as the route start)'},
        'bins': {'delta_d_m': DD_BIN, 'sec': S_BIN}, 'per_arena': {}}
    per = {t: load(t) for t in TAGS}

    for t in TAGS:
        d = per[t]
        kd = np.round(d['dd'] / DD_BIN).astype(np.int64)
        ks = np.round(d['sec'] / S_BIN).astype(np.int64)
        k0 = np.round(d['s0'] / S_BIN).astype(np.int64)
        kD0 = np.round(d['d0'] / DD_BIN).astype(np.int64)
        a, _ = within_bin_stats([kd, ks], d['zrel'])
        b, _ = within_bin_stats([kd, ks, k0], d['zrel'])
        c, _ = within_bin_stats([kd, ks, k0, kD0], d['zrel'])
        sens = np.abs(1.0 / d['s0'] - 1.0 / d['sec'])
        res['per_arena'][t] = {
            'n_points': int(len(d['dep'])), 'n_routes': int(d['route'].max() + 1),
            'true_relative_height_std_m': float(d['zrel'].std()),
            'true_relative_height_p1_p99_m': [float(np.quantile(d['zrel'], .01)), float(np.quantile(d['zrel'], .99))],
            'match_dd_sec': a,
            'match_dd_sec_s0': b,
            'match_dd_sec_s0_d0': c,
            'sensitivity_abs_1_over_s0_minus_1_over_s': {
                'mean': float(sens.mean()), 'p50': float(np.median(sens)),
                'p95': float(np.quantile(sens, .95)), 'max': float(sens.max())},
            'example_pairs_same_dd_and_sec': example_pairs(d),
        }

    # pooled over all six arenas (the deployment population)
    allk = {k: np.concatenate([per[t][k] for t in TAGS]) for k in ('dd', 'sec', 's0', 'd0', 'zrel', 'zpix', 'dep')}
    kd = np.round(allk['dd'] / DD_BIN).astype(np.int64)
    ks = np.round(allk['sec'] / S_BIN).astype(np.int64)
    k0 = np.round(allk['s0'] / S_BIN).astype(np.int64)
    p_a, _ = within_bin_stats([kd, ks], allk['zrel'])
    p_b, _ = within_bin_stats([kd, ks, k0], allk['zrel'])
    res['pooled_six_arenas'] = {'n_points': int(len(kd)),
                                'true_relative_height_std_m': float(allk['zrel'].std()),
                                'match_dd_sec': p_a, 'match_dd_sec_s0': p_b}

    # ---- part 2: the omitted d0 across groups and arenas -------------------------------------------------
    d0tab = {}
    for t in TAGS:
        z = np.load(HERE / 'points' / f'{t}.npz', allow_pickle=True)
        d0 = z['d0_all'].astype(np.float64); s0 = z['s0_all'].astype(np.float64)
        grp = z['group_all'].astype(str)
        _, first = np.unique(grp, return_index=True)
        z0 = H - d0 / s0
        q = lambda v, p: float(np.quantile(v, p))
        d0tab[t] = {
            'n_routes': int(len(d0)), 'n_groups': int(len(first)),
            'd0_m': {'mean': float(d0.mean()), 'std': float(d0.std()), 'min': float(d0.min()),
                     'p5': q(d0, .05), 'p50': q(d0, .5), 'p95': q(d0, .95), 'max': float(d0.max()),
                     'range': float(d0.max() - d0.min())},
            'd0_per_group_m': {'std_of_group_means': float(np.array([d0[grp == g].mean() for g in np.unique(grp)]).std()),
                               'mean_within_group_std': float(np.array([d0[grp == g].std() for g in np.unique(grp)]).mean())},
            's0': {'mean': float(s0.mean()), 'std': float(s0.std()), 'min': float(s0.min()), 'max': float(s0.max())},
            'start_height_z0_m': {'mean': float(z0.mean()), 'std': float(z0.std()),
                                  'min': float(z0.min()), 'max': float(z0.max())},
            'variance_split': {
                'note': 'd0 = (H - z0) * s0; log-variance split of the two factors',
                'std_from_start_height_if_s0_fixed_m': float((s0.mean() * z0).std()),
                'std_from_viewing_geometry_if_z0_fixed_m': float(((H - z0.mean()) * s0).std())},
        }
    alld0 = np.concatenate([np.load(HERE / 'points' / f'{t}.npz', allow_pickle=True)['d0_all'].astype(np.float64) for t in TAGS])
    d0tab['pooled'] = {'n_routes': int(len(alld0)), 'mean': float(alld0.mean()), 'std': float(alld0.std()),
                       'min': float(alld0.min()), 'max': float(alld0.max()),
                       'range': float(alld0.max() - alld0.min()),
                       'between_arena_std_of_means': float(np.array(
                           [d0tab[t]['d0_m']['mean'] for t in TAGS]).std())}
    res['d0_spread'] = d0tab

    # how much of the measured ambiguity the d0 spread explains:
    # predicted |dz| for a pair matched on (dd, sec, s0) is |d0 - d0'| * |1/s0 - 1/s|
    expl = {}
    for t in TAGS:
        d = per[t]
        sens = np.abs(1.0 / d['s0'] - 1.0 / d['sec'])
        sd0 = d0tab[t]['d0_m']['std']
        pred = float(np.sqrt(np.mean((sens * sd0 * np.sqrt(2)) ** 2)) / np.sqrt(2))   # rms of sens*std(d0)
        expl[t] = {'predicted_irreducible_rms_m': float(np.sqrt(np.mean((sens * sd0) ** 2))),
                   'measured_pooled_within_bin_std_m': res['per_arena'][t]['match_dd_sec_s0']['pooled_within_bin_std_m'],
                   'measured_over_true_height_std': res['per_arena'][t]['match_dd_sec_s0']['pooled_within_bin_std_m'] /
                                                    res['per_arena'][t]['true_relative_height_std_m']}
        del pred
    res['d0_explains_ambiguity'] = expl

    out = HERE / 'ambiguity.json'
    json.dump(res, open(out, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('algebra', 'pooled_six_arenas')}, indent=1))
    for t in TAGS:
        r = res['per_arena'][t]
        print(f"{t}: true zrel std {r['true_relative_height_std_m']:.3f} m | "
              f"within (dd,sec) bin std {r['match_dd_sec']['pooled_within_bin_std_m']:.3f} m "
              f"(max bin range {r['match_dd_sec']['bin_range_max_m']:.2f} m) | "
              f"within (dd,sec,s0) bin std {r['match_dd_sec_s0']['pooled_within_bin_std_m']:.3f} m | "
              f"within (dd,sec,s0,d0) {r['match_dd_sec_s0_d0']['pooled_within_bin_std_m']:.4f} m")
    print('wrote', out)


if __name__ == '__main__':
    main()
