#!/usr/bin/env python3
"""VERIFY_S1: belly-in-soil flag of every Gator soil run in the training file (own code): lowest hull point more than
0.05 m under the undisturbed surface for more than 1 s in a row; frame spacing from outcome.json (elapsed_s / frames),
not assumed. Reads only.  vs1_belly.py <ids txt> <runs dir> <out json>"""
import sys, os, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
RUNS = sys.argv[2]
def one(i):
    d = os.path.join(RUNS, i)
    o = json.load(open(os.path.join(d, 'outcome.json')))
    b = np.load(os.path.join(d, 'vehicle_extra.npz'))['belly_clearance_min_m'].astype(float)
    dt = o['elapsed_s'] / max(o['frames'] - 1, 1) if o.get('frames') else None
    deep = b < -0.05
    best = cur = 0
    for v in deep:
        cur = cur + 1 if v else 0; best = max(best, cur)
    return dict(id=i, n=len(b), frames=o.get('frames'), dt=dt, run_frames=best, total_frames=int(deep.sum()), goal=bool(o.get('goal_reached')),
                nan=int(np.isnan(b).sum()))
if __name__ == '__main__':
    ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    with ProcessPoolExecutor(8) as ex:
        recs = list(ex.map(one, ids, chunksize=64))
    json.dump(recs, open(sys.argv[3], 'w'))
    print('done', len(recs))
