"""Extension picks: night-1 model vs night-2 model on identical fixed-2 m/s candidate sets."""
import glob, hashlib, json, os, sys
import numpy as np, torch
sys.path.insert(0, 'scripts')
from f104_night_train import HazardNet, route_logit, DEV
from f104_n2_train import Net
from f104_n2_pick import load_night1, load_night2, risk

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
N2 = ROOT + '/night2_v1'
CAND = N2 + '/testcand_ext'
OUT = N2 + '/closed_ext'

if __name__ == '__main__':
    os.makedirs(OUT + '/routes', exist_ok=True)
    m1 = load_night1(); m2 = load_night2(N2 + '/final/N2_s*.pt')
    med = np.median(np.load(N2 + '/station_ds_fix.npz', allow_pickle=True)['ctx'][:, :17], 0)
    groups = [g['group'] for g in json.load(open(N2 + '/ext_test_groups.json'))['groups']]
    tasks, rows = [], []
    for g in groups:
        z = np.load(f'{CAND}/{g}.npz')
        full = np.concatenate([np.repeat(med[None], len(z['geom_ctx']), 0), z['geom_ctx']], 1).astype(np.float32)
        p1 = risk(m1, z, full, 'n1'); p2 = risk(m2, z, full, 'n2')
        pick = {'old': int(np.argmin(p1)), 'new': int(np.argmin(p2))}
        shard = int(hashlib.md5(g.encode()).hexdigest(), 16) % 40
        seen = {}
        for arm, idx in pick.items():
            first = idx not in seen
            seen.setdefault(idx, arm)
            rid = f'{g}__{seen[idx]}'
            if first:
                json.dump({'waypoints': z['wp'][idx].tolist(), 'speeds': z['sp'][idx].tolist(),
                           'stations': z['st'][idx].tolist(), 'headings': z['hd'][idx].tolist(),
                           'meta': {'candidate': f'ext_{arm}', 'scene_id': g, 'cand_index': idx}},
                          open(f'{OUT}/routes/{rid}.json', 'w'))
            tasks.append(dict(id=rid, group_id=g, arm=arm, cand_index=idx,
                              risk=float((p1 if arm == 'old' else p2)[idx]), shard=shard, run=first))
        rows.append(dict(group=g, same=pick['old'] == pick['new'], p_old=float(p1.min()), p_new=float(p2.min())))
    json.dump(tasks, open(OUT + '/tasks_cluster.json', 'w'), indent=1)
    json.dump(rows, open(OUT + '/picks.json', 'w'), indent=1)
    print(f"{len(groups)} groups; {sum(t['run'] for t in tasks)} episodes to drive; "
          f"identical picks in {sum(r['same'] for r in rows)} groups")
