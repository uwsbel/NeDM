"""Task B part 3, mechanism: is the raw-depth channel decodable LOCALLY, at the scale the CNN front end works at?

scripts/gen_riskmodel.Net convolves the 96x32 corridor with four 3x3 kernels and then collapses the lateral axis
with mean+max (f.mean(-1), f.amax(-1)) before the BiGRU ever runs along stations. So whatever terrain shape the
network uses laterally must be formed by LOCAL operations on the raw channels. Exactly:

    dd_i = d_i - d_0 = Q (s_i - s_0) - z_rel_i * s_i,          Q = H - z_0 ~= 110 m

The first term is pure perspective and depends on the route start, which no local kernel can see. Here we measure,
per route and station, the lateral least-squares slope of the raw relative-depth channel and of the true height,
and the same along the route; the gap between them is the spurious slope a local operator reads off the D channel.
Also reported: the linear readout of relative height from the delivered D channels and its channel weights.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TAGS = ['f104', 'g203', 'g216', 'g217', 'g228', 'g231']
H = 110.0
N_LATERAL, HALF_WIDTH_M = 32, 6.0


def load(tag):
    z = np.load(HERE / 'points' / f'{tag}.npz', allow_pickle=True)
    d = {k: z[k].astype(np.float64) for k in
         ('dep', 'sec', 'd0', 's0', 'zpix', 'gx', 'gy', 'xbp', 'ybp', 'route', 'station', 'lateral')}
    keep = (np.abs(d['gx']) < 39.5) & (np.abs(d['gy']) < 39.5)
    d = {k: v[keep] for k, v in d.items()}
    d['dd'] = d['dep'] - d['d0']
    d['z0'] = H - d['d0'] / d['s0']
    d['zrel'] = d['zpix'] - d['z0']
    d['off'] = -HALF_WIDTH_M + d['lateral'] * (2 * HALF_WIDTH_M) / (N_LATERAL - 1)
    return d


def group_slopes(key, x, ys, min_n=6):
    """Least-squares slope of each y in ys against x, within groups defined by the integer key rows."""
    order = np.lexsort(tuple(np.asarray(key)[::-1]))
    k = np.stack(key)[:, order]
    new = np.r_[True, np.any(k[:, 1:] != k[:, :-1], axis=0)]
    st = np.flatnonzero(new); en = np.r_[st[1:], len(order)]
    n = (en - st).astype(float)
    xs = x[order]
    sx = np.add.reduceat(xs, st); sxx = np.add.reduceat(xs * xs, st)
    den = sxx - sx * sx / n
    ok = (n >= min_n) & (den > 1e-9)
    out = []
    for y in ys:
        yo = y[order]
        sy = np.add.reduceat(yo, st); sxy = np.add.reduceat(xs * yo, st)
        out.append(np.where(ok, (sxy - sx * sy / n) / np.where(den > 1e-9, den, 1.0), np.nan))
    return out, ok


def stats(v):
    v = v[np.isfinite(v)]
    return dict(std=float(v.std()), mean=float(v.mean()), p50_abs=float(np.median(np.abs(v))),
                p95_abs=float(np.quantile(np.abs(v), .95)), max_abs=float(np.abs(v).max()))


def main():
    res = {'note': 'slopes in m/m; dd is the v1 depth_rel channel, z the back-projected height',
           'per_arena': {}}
    for t in TAGS:
        d = load(t)
        rt = d['route'].astype(np.int64); sta = d['station'].astype(np.int64); lat = d['lateral'].astype(np.int64)
        Q = H - d['z0']
        geom = Q * (d['sec'] - d['s0'])                     # pure perspective part of dd
        terr = -d['zrel'] * d['sec']                        # terrain part of dd
        chk = float(np.abs(geom + terr - d['dd']).max())

        # --- lateral (cross-slope) direction, within one station of one route
        (mz, mdd, mg), ok = group_slopes([rt, sta], d['off'], [d['zpix'], d['dd'], geom])
        # --- along-route direction, within one lateral lane of one route
        s_along = np.hypot(d['gx'], d['gy']) * 0  # placeholder, replaced by arc index below
        st_m = sta * 0.0
        # station spacing differs per route; use the station index scaled by the route's mean spacing
        (az, add, ag), ok2 = group_slopes([rt, lat], sta.astype(float), [d['zpix'], d['dd'], geom])

        res['per_arena'][t] = {
            'identity_check_max_abs_m': chk,
            'dd_variance_split': {
                'dd_std_m': float(d['dd'].std()), 'perspective_term_std_m': float(geom.std()),
                'terrain_term_std_m': float(terr.std()),
                'perspective_over_terrain_std_ratio': float(geom.std() / terr.std()),
                'corr_perspective_terrain': float(np.corrcoef(geom, terr)[0, 1])},
            'cross_slope_m_per_m': {
                'true_height_slope': stats(mz), 'raw_depth_channel_slope': stats(mdd),
                'perspective_only_slope': stats(mg),
                'rms_gap_depth_minus_true': float(np.sqrt(np.nanmean((mdd + mz) ** 2))),
                'corr_depth_vs_true': float(np.corrcoef(*[v[np.isfinite(mdd) & np.isfinite(mz)] for v in (mdd, mz)])[0, 1]),
                'note': 'dd ~ -z*s + perspective, so a faithful channel would give slope(dd) ~ -slope(z)'},
            'along_route_slope_per_station': {
                'true_height_slope': stats(az), 'raw_depth_channel_slope': stats(add),
                'perspective_only_slope': stats(ag),
                'corr_depth_vs_true': float(np.corrcoef(*[v[np.isfinite(add) & np.isfinite(az)] for v in (add, az)])[0, 1])},
        }
        r = res['per_arena'][t]
        print(f"{t}: dd std {r['dd_variance_split']['dd_std_m']:.3f} = perspective {r['dd_variance_split']['perspective_term_std_m']:.3f} "
              f"+ terrain {r['dd_variance_split']['terrain_term_std_m']:.3f} (ratio {r['dd_variance_split']['perspective_over_terrain_std_ratio']:.2f}) | "
              f"cross-slope: true std {r['cross_slope_m_per_m']['true_height_slope']['std']:.3f}, "
              f"depth-channel std {r['cross_slope_m_per_m']['raw_depth_channel_slope']['std']:.3f}, "
              f"perspective-only std {r['cross_slope_m_per_m']['perspective_only_slope']['std']:.3f}, "
              f"corr {r['cross_slope_m_per_m']['corr_depth_vs_true']:+.3f}", flush=True)
        del s_along, st_m

    # linear readout of z - z0 from the delivered D channels, fitted on f104
    d = load('f104')
    X = np.c_[d['dd'], d['sec'] - 1, d['s0'] - 1, np.ones(len(d['dd']))]
    w = np.linalg.lstsq(X, d['zrel'], rcond=None)[0]
    res['linear_readout_fitted_on_f104'] = {
        'form': 'z - z0 ~= w0*depth_rel + w1*ray_sec1 + w2*ray_sec1_at_start + b',
        'weights': {'depth_rel': float(w[0]), 'ray_sec1': float(w[1]), 'ray_sec1_start': float(w[2]),
                    'bias': float(w[3])},
        'channel_std': {'depth_rel': float(d['dd'].std()), 'ray_sec1': float((d['sec'] - 1).std()),
                        'ray_sec1_start': float((d['s0'] - 1).std())},
        'weight_ratio_sec_over_depth': float(abs(w[1] / w[0])),
        'rmse_on_f104_m': float(np.sqrt(np.mean((X @ w - d['zrel']) ** 2))),
        'naive_no_sec_rmse_m': float(np.sqrt(np.mean((-d['dd'] - d['zrel']) ** 2))),
    }
    print(json.dumps(res['linear_readout_fitted_on_f104'], indent=1))
    json.dump(res, open(HERE / 'local_structure.json', 'w'), indent=1)
    print('wrote', HERE / 'local_structure.json')


if __name__ == '__main__':
    main()
