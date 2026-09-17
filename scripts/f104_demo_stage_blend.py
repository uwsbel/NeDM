"""Build episodes.json for the Blender re-run of the 15 chosen demo episodes, and the route ribbons."""
import json, os, sys
import numpy as np
sys.path.insert(0, 'scripts')
from f104_n2_analyze import labels

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
OUT = ROOT + '/night2_v1/demo_v1'
ROLES = ('optimal', 'suboptimal', 'risky')


def height_fn():
    o = json.load(open(f'{ROOT}/static_map_v1/observation.json'))['camera']
    z = np.load(f'{ROOT}/static_map_v1/observation.npz')
    e = np.where(z['rgbd'][3] > -1.999, z['rgbd'][3].astype(float) * o['elevation_scale_m'], np.nan)
    n = e.shape[0]; mpp = (2 * o['cam_height_m'] * np.tan(o['hfov_rad'] / 2)) / n; ctr = (n - 1) / 2.0

    def h(xy):
        row = np.clip(ctr - xy[:, 1] / mpp, 0, n - 1).astype(int)
        col = np.clip(ctr + xy[:, 0] / mpp, 0, n - 1).astype(int)
        return e[row, col]
    return h


def main():
    sel = json.load(open(OUT + '/selection.json'))
    picks = {p['group']: p for p in json.load(open(OUT + '/picks.json'))}
    H = height_fn()
    os.makedirs(OUT + '/ribbons', exist_ok=True)
    eps, meta = [], []
    for k, g in enumerate(sel['chosen']):
        case = json.load(open(f'{ROOT}/cases_haz_final/{g}.json'))
        for r in ROLES:
            rid = f'{g}__{r}'
            lab = labels(f'{OUT}/runs/{rid}')
            eps.append(dict(id=rid, group=g, role=r, shard=len(eps),
                            status=lab['status'], elapsed_s=lab['elapsed']))
            rt = json.load(open(f'{OUT}/routes/{rid}.json'))
            wp = np.asarray(rt['waypoints'], float)
            s = np.r_[0.0, np.linalg.norm(np.diff(wp, axis=0), axis=1).cumsum()]
            gd = np.linspace(0, s[-1], max(int(s[-1] / 0.3), 8))
            d = np.stack([np.interp(gd, s, wp[:, 0]), np.interp(gd, s, wp[:, 1])], 1)
            z = H(d) + 0.18
            np.save(f'{OUT}/ribbons/{rid}.npy', np.c_[d, z])
            meta.append(dict(id=rid, scenario=k + 1, group=g, role=r,
                             goal=[float(case['goal_xy'][0]), float(case['goal_xy'][1]),
                                   float(case.get('goal_radius_m', 2.5))],
                             risk=picks[g]['roles'][r]['risk'], rank=picks[g]['roles'][r]['rank'],
                             detour_m=picks[g]['roles'][r]['detour_m'],
                             mean_speed=picks[g]['roles'][r]['mean_speed'], **lab))
    json.dump(eps, open(OUT + '/episodes.json', 'w'), indent=1)
    json.dump(meta, open(OUT + '/video_meta.json', 'w'), indent=1)
    print(f'{len(eps)} episodes staged for the Blender re-run')


if __name__ == '__main__':
    main()
