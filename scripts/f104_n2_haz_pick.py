"""Hazard-test picks: six arms, identical groups, one node per group.

  control          night-1 model, night-1 proposal
  sampler          night-2 model, night-2 proposal (with the 9 designed-route anchors)
  sampler_noanchor night-2 model, night-2 proposal with anchors REMOVED  (separates "wider sampler" from
                   "added a designed-route fallback", which the audit showed were bundled)
  anchor6          always the 6 m/s straight line
  fixed2_old/new   2 m/s fixed, geometry-only candidates, night-1 vs night-2 model
"""
import hashlib, json, os, sys
import numpy as np
sys.path.insert(0, 'scripts')
from f104_n2_pick import load_night1, load_night2, risk

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
N2 = ROOT + '/night2_v1'
CAND = N2 + '/testcand_haz'
OUT = N2 + '/closed_haz'

if __name__ == '__main__':
    os.makedirs(OUT + '/routes', exist_ok=True)
    m1 = load_night1(); m2 = load_night2(N2 + '/final/N2_s*.pt')
    med = np.median(np.load(N2 + '/station_ds_fix.npz', allow_pickle=True)['ctx'][:, :17], 0)
    groups = [g['group'] for g in json.load(open(N2 + '/haz_test_groups.json'))['groups']]
    tasks, rows = [], []
    for g in groups:
        z1 = np.load(f'{CAND}/{g}__night1.npz'); z2 = np.load(f'{CAND}/{g}__night2.npz')
        z3 = np.load(f'{CAND}/{g}__fixed2.npz')
        full = lambda z: np.concatenate([np.repeat(med[None], len(z['geom_ctx']), 0), z['geom_ctx']], 1).astype(np.float32)
        p_c = risk(m1, z1, full(z1), 'n1')
        p_s = risk(m2, z2, full(z2), 'n2')
        p_f_old = risk(m1, z3, full(z3), 'n1'); p_f_new = risk(m2, z3, full(z3), 'n2')
        anch = z2['anchor'].astype(bool)
        free = np.where(~anch)[0]
        six = np.where(anch & (z2['cruise'] == 6.0) & (np.abs(z2['offset']) < 1e-6))[0]
        pick = {'control': ('night1', int(np.argmin(p_c))),
                'sampler': ('night2', int(np.argmin(p_s))),
                'sampler_noanchor': ('night2', int(free[np.argmin(p_s[free])])),
                'anchor6': ('night2', int(six[0])),
                'fixed2_old': ('fixed2', int(np.argmin(p_f_old))),
                'fixed2_new': ('fixed2', int(np.argmin(p_f_new)))}
        shard = int(hashlib.md5(g.encode()).hexdigest(), 16) % 40
        seen = {}
        for arm, (cs, idx) in pick.items():
            first = (cs, idx) not in seen
            seen.setdefault((cs, idx), arm)
            rid = f'{g}__{seen[(cs, idx)]}'
            if first:
                z = {'night1': z1, 'night2': z2, 'fixed2': z3}[cs]
                json.dump({'waypoints': z['wp'][idx].tolist(), 'speeds': z['sp'][idx].tolist(),
                           'stations': z['st'][idx].tolist(), 'headings': z['hd'][idx].tolist(),
                           'meta': {'candidate': f'haz_{arm}', 'scene_id': g, 'cand_set': cs, 'cand_index': idx}},
                          open(f'{OUT}/routes/{rid}.json', 'w'))
            p = {'control': p_c, 'sampler': p_s, 'sampler_noanchor': p_s, 'anchor6': p_s,
                 'fixed2_old': p_f_old, 'fixed2_new': p_f_new}[arm]
            tasks.append(dict(id=rid, group_id=g, arm=arm, cand_set=cs, cand_index=idx,
                              risk=float(p[idx]), shard=shard, run=first))
        rows.append(dict(group=g, pick_is_anchor=bool(anch[pick['sampler'][1]]),
                         p_control=float(p_c.min()), p_sampler=float(p_s.min()),
                         p_sampler_noanchor=float(p_s[free].min())))
    json.dump(tasks, open(OUT + '/tasks_cluster.json', 'w'), indent=1)
    json.dump(rows, open(OUT + '/picks.json', 'w'), indent=1)
    print(f"{len(groups)} groups, {sum(t['run'] for t in tasks)} episodes to drive; "
          f"night-2 pick was a designed anchor in {sum(r['pick_is_anchor'] for r in rows)} groups")
