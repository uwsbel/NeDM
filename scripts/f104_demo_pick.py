"""Demo scenarios: for each candidate hazard start/goal, take THREE routes out of the planner's own
256-candidate pool, ranked by the deployed risk model, and stage them for Chrono.

  optimal    = argmin risk                    (what the planner actually drives)
  suboptimal = nearest candidate to a target risk, geometrically distinct from the optimal
  risky      = argmax risk                    (what the planner refuses)

One model, one candidate pool, three ranks -- so the video set is a direct read on whether the
predicted ordering matches Chrono.
"""
import hashlib, json, os, sys
import numpy as np
sys.path.insert(0, 'scripts')
from f104_n2_pick import load_night2, risk

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
N2 = ROOT + '/night2_v1'
CAND = N2 + '/testcand_haz'
OUT = N2 + '/demo_v1'
TARGET = 0.10          # suboptimal aim point in predicted risk


def elevation():
    o = json.load(open(f'{ROOT}/static_map_v1/observation.json'))['camera']
    z = np.load(f'{ROOT}/static_map_v1/observation.npz')
    e = np.where(z['rgbd'][3] > -1.999, z['rgbd'][3].astype(float) * o['elevation_scale_m'], np.nan)
    n = e.shape[0]; mpp = (2 * o['cam_height_m'] * np.tan(o['hfov_rad'] / 2)) / n; h = n / 2 * mpp
    return np.flipud(e), [-h, h, -h, h]


E, EXT = elevation()
GROUND = float(np.nanmedian(E))


def height(xy):
    n = E.shape[0]; mpp = (EXT[1] - EXT[0]) / n
    j = np.clip(((xy[:, 0] - EXT[0]) / mpp).astype(int), 0, n - 1)
    i = np.clip(((xy[:, 1] - EXT[2]) / mpp).astype(int), 0, n - 1)
    return E[i, j]


def chord_offset(wp):
    a, b = wp[0], wp[-1]; d = b - a; L = max(np.linalg.norm(d), 1e-9); u = d / L
    rel = wp - a; along = rel @ u
    return float(np.abs(np.linalg.norm(rel - along[:, None] * u[None], axis=1)).max())


def describe(wp, sp):
    h = height(wp)
    return dict(detour_m=round(chord_offset(wp), 2), mean_speed=round(float(sp[1:-1].mean()), 2),
                max_speed=round(float(sp.max()), 2), length_m=round(float(np.linalg.norm(np.diff(wp, axis=0), axis=1).sum()), 1),
                rise_m=round(float(np.nanmax(h) - GROUND), 2), dip_m=round(float(GROUND - np.nanmin(h)), 2))


def main():
    groups = json.load(open(OUT + '/groups.json'))
    os.makedirs(OUT + '/routes', exist_ok=True)
    m2 = load_night2(N2 + '/final/N2_s*.pt')
    med = np.median(np.load(N2 + '/station_ds_fix.npz', allow_pickle=True)['ctx'][:, :17], 0)
    tasks, rows = [], []
    for g in groups:
        z = np.load(f'{CAND}/{g}__night2.npz')
        ctx = np.concatenate([np.repeat(med[None], len(z['geom_ctx']), 0), z['geom_ctx']], 1).astype(np.float32)
        p = risk(m2, z, ctx, 'n2')
        i_opt = int(np.argmin(p)); i_bad = int(np.argmax(p))
        wp = z['wp']
        sep = np.abs(wp - wp[i_opt][None]).sum(-1).max(-1)          # how far each candidate departs from the pick
        ok = np.where((p > p[i_opt]) & (sep > 3.0) & (np.arange(len(p)) != i_bad))[0]
        i_sub = int(ok[np.argmin(np.abs(p[ok] - TARGET))]) if len(ok) else int(np.argsort(p)[len(p) // 2])
        shard = int(hashlib.md5(g.encode()).hexdigest(), 16) % 28
        row = dict(group=g, shard=shard, roles={})
        for role, i in (('optimal', i_opt), ('suboptimal', i_sub), ('risky', i_bad)):
            rid = f'{g}__{role}'
            json.dump({'waypoints': z['wp'][i].tolist(), 'speeds': z['sp'][i].tolist(),
                       'stations': z['st'][i].tolist(), 'headings': z['hd'][i].tolist(),
                       'meta': {'candidate': f'demo_{role}', 'scene_id': g, 'cand_index': i}},
                      open(f'{OUT}/routes/{rid}.json', 'w'))
            d = describe(z['wp'][i], z['sp'][i])
            d.update(role=role, risk=float(p[i]), cand_index=i,
                     anchor=bool(z['anchor'][i]), rank=int((p < p[i]).sum()))
            row['roles'][role] = d
            tasks.append(dict(id=rid, group_id=g, role=role, risk=float(p[i]), shard=shard, run=True))
        row['p_min'], row['p_max'], row['p_median'] = float(p.min()), float(p.max()), float(np.median(p))
        rows.append(row)
        print(f"{g}  risk opt {p[i_opt]*100:7.4f}%  sub {p[i_sub]*100:7.3f}%  bad {p[i_bad]*100:6.2f}%  "
              f"detour {row['roles']['optimal']['detour_m']:4.1f}/{row['roles']['suboptimal']['detour_m']:4.1f}/"
              f"{row['roles']['risky']['detour_m']:4.1f} m  "
              f"speed {row['roles']['optimal']['mean_speed']:4.1f}/{row['roles']['suboptimal']['mean_speed']:4.1f}/"
              f"{row['roles']['risky']['mean_speed']:4.1f} m/s", flush=True)
    json.dump(tasks, open(OUT + '/tasks_cluster.json', 'w'), indent=1)
    json.dump(rows, open(OUT + '/picks.json', 'w'), indent=1)
    print(f'\n{len(rows)} groups, {len(tasks)} episodes staged in {OUT}')


if __name__ == '__main__':
    main()
