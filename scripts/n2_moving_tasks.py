"""Moving-start ground-truth tasks (rigid): held-out re-anchored routes driven from the anchor pose at v0 in {0, 2, 4} m/s.
Rebuilds the remaining route exactly as n2_reanchor_dataset.py does, writes case + route JSONs and a task list.
  python scripts/n2_moving_tasks.py --reanchor reanchor_rigid.npz --runs DIR... --out DIR --n 350
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f104_n2_dataset as DS

ap = argparse.ArgumentParser()
ap.add_argument('--reanchor', required=True); ap.add_argument('--runs', nargs='+', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=350); ap.add_argument('--v0', default='0,2,4'); ap.add_argument('--seed', type=int, default=20260918)
a = ap.parse_args()
d = np.load(a.reanchor, allow_pickle=True)
sp = d['split'].astype(str); k = d['anchor_frame']; vx = d['vx_anchor']; rem = d['rem_m']
cand = np.where((sp != 'train') & (k > 0) & (rem >= 15.0) & (d['time_to_event_s'] != 0))[0]
rng = np.random.default_rng(a.seed)
# stratify over anchor speed: equal numbers from three speed bins
bins = [(0, 1), (1, 3), (3, 9)]; pick = []
for lo, hi in bins:
    c = cand[(vx[cand] >= lo) & (vx[cand] < hi)]; pick += list(rng.choice(c, min(len(c), a.n // 3), replace=False))
pick = sorted(pick); print('anchors', len(pick), 'by bin', [int(((vx[pick] >= lo) & (vx[pick] < hi)).sum()) for lo, hi in bins])
runs = {os.path.basename(p): p for pat in a.runs for p in glob.glob(pat + '/*')}
os.makedirs(a.out + '/cases', exist_ok=True); os.makedirs(a.out + '/routes', exist_ok=True)
tasks, index = [], []
for i in pick:
    ep, kk = str(d['episode'][i]), int(k[i]); rd = runs[ep]
    z = np.load(rd + '/trajectory.npz'); r = np.load(rd + '/command_reference.npz'); c = json.load(open(rd + '/case.json'))
    wp, spd, st = np.asarray(r['reference_waypoints'], float), np.asarray(r['reference_speeds'], float), np.asarray(r['reference_stations'], float)
    pose = z['pose'].astype(float); s, dev, L = DS.project(pose[:, :2], wp); smax = np.maximum.accumulate(s); sk = float(smax[kk])
    j = int(np.searchsorted(st, sk, side='right')); p0 = np.array([np.interp(sk, st, wp[:, 0]), np.interp(sk, st, wp[:, 1])])
    wp2 = np.vstack([p0[None], wp[j:]]); sp2 = np.r_[float(np.interp(sk, st, spd)), spd[j:]]
    st2 = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp2, axis=0), axis=1))]
    hd2 = np.arctan2(np.gradient(wp2[:, 1]), np.gradient(wp2[:, 0]))
    # the vehicle is spawned at the ROUTE point (not the recorded pose) with the route heading: identical for every v0
    aid = f'{ep}@{kk}'; case_id = f'mv_{hashlib.md5(aid.encode()).hexdigest()[:10]}'
    case = dict(c); case['id'] = case_id; case['split'] = 'test'
    case['layout'] = dict(c['layout'], episode_id=case_id, start_xy=p0.tolist(), start_yaw=float(hd2[0]), assets=[])
    case['moving_start'] = dict(episode=ep, anchor_frame=kk, vx_anchor=float(vx[i]), remaining_m=float(st2[-1]), profile_speed_at_anchor=float(sp2[0]))
    json.dump(case, open(f'{a.out}/cases/{case_id}.json', 'w'))
    json.dump({'waypoints': wp2.tolist(), 'speeds': sp2.tolist(), 'stations': st2.tolist(), 'headings': hd2.tolist(), 'meta': {'candidate': 'moving_start_remaining_route', 'source_episode': ep, 'anchor_frame': kk}},
              open(f'{a.out}/routes/{case_id}.json', 'w'))
    for v0 in [float(x) for x in a.v0.split(',')]:
        rid = f'{case_id}__v{v0:g}'
        tasks.append(dict(id=rid, group=case_id, case=f'moving_v1/cases/{case_id}.json', route=f'moving_v1/routes/{case_id}.json', run=True, tier=len(index),
                          episode_seed=int(hashlib.md5(rid.encode()).hexdigest()[:8], 16), extra=['--v0', f'{v0:g}']))
    index.append(dict(case=case_id, anchor=aid, vx_anchor=float(vx[i]), rem_m=float(st2[-1]), recorded_unsafe=int(d['unsafe'][i]), recorded_fail=int(d['fail'][i]),
                      group=str(d['group'][i]), row=int(i)))
json.dump(tasks, open(f'{a.out}/tasks.json', 'w')); json.dump(index, open(f'{a.out}/index.json', 'w'), indent=1)
print(len(tasks), 'tasks written to', a.out)
