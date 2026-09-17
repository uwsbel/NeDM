"""Tilt experiment picks: current night-2 model vs tilt-aware model on the SAME hazard candidate sets."""
import hashlib, json, os, sys
import numpy as np
sys.path.insert(0, 'scripts')
from f104_n2_pick import load_night2, risk

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
N2 = ROOT + '/night2_v1'
CAND = N2 + '/testcand_haz'
OUT = N2 + '/closed_tilt'

if __name__ == '__main__':
    os.makedirs(OUT + '/routes', exist_ok=True)
    base = load_night2(N2 + '/final/N2_s*.pt'); tilt = load_night2(N2 + '/final/N2T_s*.pt')
    med = np.median(np.load(N2 + '/station_ds_fix.npz', allow_pickle=True)['ctx'][:, :17], 0)
    groups = [g['group'] for g in json.load(open(N2 + '/haz_test_groups.json'))['groups']]
    tasks, rows = [], []
    for g in groups:
        z = np.load(f'{CAND}/{g}__night2.npz')
        full = np.concatenate([np.repeat(med[None], len(z['geom_ctx']), 0), z['geom_ctx']], 1).astype(np.float32)
        pb = risk(base, z, full, 'n2'); pt = risk(tilt, z, full, 'n2')
        pick = {'base': int(np.argmin(pb)), 'tilt': int(np.argmin(pt))}
        shard = int(hashlib.md5(g.encode()).hexdigest(), 16) % 40
        seen = {}
        for arm, idx in pick.items():
            first = idx not in seen
            seen.setdefault(idx, arm)
            rid = f'{g}__{seen[idx]}'
            if first:
                json.dump({'waypoints': z['wp'][idx].tolist(), 'speeds': z['sp'][idx].tolist(),
                           'stations': z['st'][idx].tolist(), 'headings': z['hd'][idx].tolist(),
                           'meta': {'candidate': f'tiltexp_{arm}', 'scene_id': g, 'cand_index': idx}},
                          open(f'{OUT}/routes/{rid}.json', 'w'))
            tasks.append(dict(id=rid, group_id=g, arm=arm, cand_index=idx,
                              risk=float((pb if arm == 'base' else pt)[idx]), shard=shard, run=first))
        rows.append(dict(group=g, same=pick['base'] == pick['tilt'],
                         speed_base=float(z['sp'][pick['base']][10:-10].mean()),
                         speed_tilt=float(z['sp'][pick['tilt']][10:-10].mean())))
    json.dump(tasks, open(OUT + '/tasks_cluster.json', 'w'), indent=1)
    json.dump(rows, open(OUT + '/picks.json', 'w'), indent=1)
    print(f"{len(groups)} groups; {sum(t['run'] for t in tasks)} episodes; identical picks in {sum(r['same'] for r in rows)}; "
          f"mean commanded speed base {np.mean([r['speed_base'] for r in rows]):.2f} vs tilt-aware "
          f"{np.mean([r['speed_tilt'] for r in rows]):.2f} m/s")
