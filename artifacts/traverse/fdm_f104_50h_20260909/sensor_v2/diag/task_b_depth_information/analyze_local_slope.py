"""Task B part 3, follow-up: local slopes are recoverable from (depth_rel, sec) with one plug-in constant;
the relative-height OFFSET is not. Differentiating z = H - (d0 + dd)/s along the corridor,

    dz/dl = -(1/s) d(dd)/dl + ((d0 + dd)/s^2) d(s)/dl,

so a local kernel needs d0 only as a MULTIPLIER on d(s)/dl, and d0 varies by only +-2.5 m about 113 m. Substituting a
single constant therefore costs almost nothing in slope, whereas the height offset needs Q*(1 - s0/s), whose spread is
2.1-2.5 m. Fit the constant on f104, evaluate the slope error on all six arenas.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TAGS = ['f104', 'g203', 'g216', 'g217', 'g228', 'g231']
H, N_LATERAL, HALF = 110.0, 32, 6.0


def load(tag):
    z = np.load(HERE / 'points' / f'{tag}.npz', allow_pickle=True)
    d = {k: z[k].astype(np.float64) for k in ('dep', 'sec', 'd0', 's0', 'zpix', 'gx', 'gy', 'route', 'station', 'lateral')}
    keep = (np.abs(d['gx']) < 39.5) & (np.abs(d['gy']) < 39.5)
    d = {k: v[keep] for k, v in d.items()}
    d['dd'] = d['dep'] - d['d0']
    d['off'] = -HALF + d['lateral'] * (2 * HALF) / (N_LATERAL - 1)
    return d


def per_station(d, min_n=8):
    key = [d['route'].astype(np.int64), d['station'].astype(np.int64)]
    order = np.lexsort(tuple(key[::-1]))
    k = np.stack(key)[:, order]
    new = np.r_[True, np.any(k[:, 1:] != k[:, :-1], axis=0)]
    st = np.flatnonzero(new); en = np.r_[st[1:], len(order)]
    n = (en - st).astype(float)
    x = d['off'][order]
    sx = np.add.reduceat(x, st); sxx = np.add.reduceat(x * x, st)
    den = sxx - sx * sx / n
    ok = (n >= min_n) & (den > 1e-9)
    slope = lambda y: np.where(ok, (np.add.reduceat(x * y[order], st) - sx * np.add.reduceat(y[order], st) / n)
                               / np.where(den > 1e-9, den, 1.0), np.nan)
    mean = lambda y: np.add.reduceat(y[order], st) / n
    return dict(sl_dd=slope(d['dd']), sl_sec=slope(d['sec']), sl_z=slope(d['zpix']),
                m_sec=mean(d['sec']), m_dd=mean(d['dd']), ok=ok)


def main():
    P = {t: per_station(load(t)) for t in TAGS}
    f = P['f104']
    m = f['ok'] & np.isfinite(f['sl_z'])
    # one constant D fitted on f104:  sl_z = -(1/s) sl_dd + ((D + dd)/s^2) sl_sec
    a = -f['sl_dd'][m] / f['m_sec'][m]
    b = f['sl_sec'][m] / f['m_sec'][m] ** 2
    resid = f['sl_z'][m] - a - (f['m_dd'][m] / f['m_sec'][m] ** 2) * f['sl_sec'][m]
    D = float(np.sum(b * resid) / np.sum(b * b))
    out = {'plugin_absolute_range_constant_m': D, 'per_arena': {}}
    for t in TAGS:
        p = P[t]
        mm = p['ok'] & np.isfinite(p['sl_z'])
        pred = -p['sl_dd'][mm] / p['m_sec'][mm] + ((D + p['m_dd'][mm]) / p['m_sec'][mm] ** 2) * p['sl_sec'][mm]
        naive = -p['sl_dd'][mm] / p['m_sec'][mm]
        e = pred - p['sl_z'][mm]
        out['per_arena'][t] = {
            'n_stations': int(mm.sum()),
            'true_cross_slope_std_m_per_m': float(p['sl_z'][mm].std()),
            'plugin_local_decode_rmse_m_per_m': float(np.sqrt(np.mean(e ** 2))),
            'plugin_local_decode_r2': float(1 - np.mean(e ** 2) / np.var(p['sl_z'][mm])),
            'plugin_local_decode_p95_abs': float(np.quantile(np.abs(e), .95)),
            'no_sec_correction_rmse_m_per_m': float(np.sqrt(np.mean((naive - p['sl_z'][mm]) ** 2))),
            'no_sec_correction_r2': float(1 - np.mean((naive - p['sl_z'][mm]) ** 2) / np.var(p['sl_z'][mm])),
        }
        r = out['per_arena'][t]
        print(f"{t}: true cross-slope std {r['true_cross_slope_std_m_per_m']:.4f} | "
              f"local decode with one constant: rmse {r['plugin_local_decode_rmse_m_per_m']:.4f} "
              f"(R2 {r['plugin_local_decode_r2']:.4f}) | without the sec term: rmse "
              f"{r['no_sec_correction_rmse_m_per_m']:.4f} (R2 {r['no_sec_correction_r2']:.3f})", flush=True)
    json.dump(out, open(HERE / 'local_slope.json', 'w'), indent=1)
    print('plug-in constant absolute range =', round(D, 3), 'm; wrote', HERE / 'local_slope.json')


if __name__ == '__main__':
    main()
