#!/usr/bin/env python3
"""VERIFY_S1: one record per run directory (vehicle block, status, completion marker, launch check, case arena/split)
for every run in the given run roots. Reads only.  vs1_runs.py <out.jsonl> <runs dir> [<runs dir> ...]"""
import sys, os, json
from concurrent.futures import ProcessPoolExecutor
def one(d):
    rid = os.path.basename(d); r = dict(id=rid, root=os.path.dirname(d))
    r['complete'] = os.path.isfile(os.path.join(d, 'episode_complete.json'))
    try:
        o = json.load(open(os.path.join(d, 'outcome.json')))
        r['has_outcome'] = True; r['has_vehicle_key'] = 'vehicle' in o
        v = o.get('vehicle'); r['vehicle_name'] = v.get('name') if isinstance(v, dict) else v
        r['wheel'] = (v or {}).get('soil_wheel_geometry', {}).get('rear', {}).get('radius_m') if isinstance(v, dict) else None
        r['status'] = o.get('status'); r['goal_reached'] = o.get('goal_reached'); r['elapsed_s'] = o.get('elapsed_s')
        b = (v or {}).get('belly') if isinstance(v, dict) else None
        r['belly_frames_below'] = None if not b else b.get('frames_below_surface')
    except Exception as e:
        r['has_outcome'] = False; r['err'] = type(e).__name__
    try:
        c = json.load(open(os.path.join(d, 'case.json'))); r['case_arena'] = c.get('arena'); r['case_id'] = c.get('id'); r['case_split'] = c.get('split')
    except Exception:
        r['case_arena'] = None
    p = os.path.join(d, 'initial_state_validation.json')
    try:
        r['launch_passed'] = json.load(open(p)).get('passed') if os.path.isfile(p) else None
    except Exception:
        r['launch_passed'] = 'unreadable'
    return r
if __name__ == '__main__':
    out = sys.argv[1]; dirs = []
    for root in sys.argv[2:]:
        dirs += sorted(os.path.join(root, x) for x in os.listdir(root))
    with ProcessPoolExecutor(8) as ex, open(out, 'w') as f:
        for r in ex.map(one, dirs, chunksize=64):
            f.write(json.dumps(r) + '\n')
    print('runs', len(dirs))
