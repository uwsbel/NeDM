"""Per-route max body tilt for the night-1 training routes (the night-2 waves get it from the cluster builder)."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
S0 = 20


def one(r):
    try:
        z = np.load(r['traj'])
        return r['id'], float(np.degrees(np.abs(z['state'][S0:, 2:4])).max())
    except Exception:
        return r['id'], float('nan')


if __name__ == '__main__':
    R = json.load(open(ROOT + '/night_v1/episodes.json'))
    out = {}
    with ProcessPoolExecutor(14) as ex:
        for i, (k, v) in enumerate(ex.map(one, R, chunksize=32)):
            out[k] = v
            if (i + 1) % 4000 == 0: print(f'  {i+1}/{len(R)}', flush=True)
    json.dump(out, open(ROOT + '/night2_v1/tilt_night1.json', 'w'))
    t = np.array([v for v in out.values() if np.isfinite(v)])
    print(f'{len(t)} routes; max tilt median {np.median(t):.1f} deg, p90 {np.percentile(t,90):.1f}, '
          f'>30 deg {100*(t>30).mean():.1f}%, >35 deg {100*(t>35).mean():.1f}%')
