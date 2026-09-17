"""Night-2 wave A2: training routes drawn from the PLANNER's own proposal distribution (scripts/f104_n2_sampler.py).

The model is queried on these routes at plan time but was never trained on anything like them: night-1 training
data is 12 designed routes per group (3 offsets x 4 speed profiles), while the planner asks about smooth
multi-mode detours with freely varying speed. This wave drives 8 such routes per group so the training
distribution covers the proposal distribution, including the wide/fast region the new sampler can now reach.
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from nedm.traverse.fdm_mppi import MPPIConfig, validate_reference
from nedm.traverse.fdm_diverse_planner import check_reference_contract
import f104_n2_sampler as S

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
CFG = MPPIConfig(max_speed_mps=6., min_speed_mps=0.0, max_curvature_inv_m=.125, arena_half_extent_m=40.)


def one(item):
    gi, g, cases_dir, out_dir, n, seed = item
    base_path = f'{cases_dir}/routes/{g}/route_00.json'
    base = {k: np.asarray(v, float) for k, v in json.load(open(base_path)).items()
            if k in ('waypoints', 'speeds', 'stations', 'headings')}
    lay = json.load(open(f'{cases_dir}/{g}.json'))['layout']
    anchor = np.array([lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']])
    rng = np.random.default_rng(seed + gi)
    kept = []
    tries = 0
    while len(kept) < n and tries < 40 * n:
        tries += 1
        sig = [7.0, 7.0, 5.0, 5.0, 5.0, 5.0, 3.0, 3.0][len(kept)]      # a couple of very wide, then typical
        spv = [2.5, 1.0, 2.5, 1.5, 1.5, 1.0, 2.5, 1.5][len(kept)]
        r = S.sample_one(base, rng, lat_sigma=sig, sp_sigma=spv, base_speed=2.0)
        if not validate_reference(r, [], CFG, anchor)['valid']:
            continue
        try:
            check_reference_contract(r)
        except Exception:
            continue
        kept.append(r)
    os.makedirs(f'{out_dir}/routes/{g}', exist_ok=True)
    rows = []
    for i, r in enumerate(kept):
        r['meta'] = {**r['meta'], 'scene_id': g, 'route_index': i, 'wave': 'night2_A2_on_policy',
                     'proposal': 'f104_n2_sampler.sample_one', 'route_seed': int(seed + gi)}
        p = f'{out_dir}/routes/{g}/op_{i:02d}.json'
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}, open(p, 'w'))
        rows.append(dict(group=g, route=p, index=i, max_lateral_m=r['meta']['max_lateral_m'],
                         mean_speed_mps=r['meta']['mean_speed_mps'],
                         speed_at_85pct=float(r['speeds'][int(.85 * len(r['speeds']))])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default=ROOT + '/cases_night2/cases')
    ap.add_argument('--out', default=ROOT + '/cases_night2_onpolicy')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--seed', type=int, default=4242)
    a = ap.parse_args()
    groups = [r['scene_id'] for r in json.load(open(a.cases + '/cases.json'))['records']]
    os.makedirs(a.out, exist_ok=True)
    items = [(i, g, a.cases, a.out, a.n, a.seed) for i, g in enumerate(groups)]
    rows = []
    with ProcessPoolExecutor(14) as ex:
        for k, rr in enumerate(ex.map(one, items, chunksize=8)):
            rows += rr
            if (k + 1) % 300 == 0:
                print(f'  {k+1}/{len(items)} groups, {len(rows)} routes', flush=True)
    json.dump(rows, open(a.out + '/routes.json', 'w'))
    lat = np.array([r['max_lateral_m'] for r in rows]); sp = np.array([r['mean_speed_mps'] for r in rows])
    e85 = np.array([r['speed_at_85pct'] for r in rows])
    print(json.dumps({'groups': len(groups), 'routes': len(rows),
                      'lateral_m': [round(float(x), 2) for x in np.percentile(lat, [50, 90, 100])],
                      'mean_speed': [round(float(x), 2) for x in np.percentile(sp, [5, 50, 95])],
                      'speed_at_85pct': [round(float(x), 2) for x in np.percentile(e85, [5, 50, 95])]}))


if __name__ == '__main__':
    main()
