"""TASK C step 3 input: for every sensor_v1 test-2 group, the route each arm drove and how that run turned out.

picks/<group>.json gives (arm -> route_id, pool, candidate index); test2/runs/<route_id>/ gives the outcome.
Labels are scripts/f104_n2_analyze.labels verbatim (unsafe = not goal_reached, or >=0.05 s rolling backwards under
throttle, or min vx <= -0.30 m/s, after a 1 s settle).
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path('/home/harry/NeDM-traverse_mppi')
EXP = ROOT / 'artifacts/traverse/fdm_f104_50h_20260909'
sys.path.insert(0, str(ROOT / 'scripts'))
from f104_n2_analyze import labels

ARENAS = ['f104', 'g203', 'g216', 'g217', 'g228', 'g231']
RUNS = EXP / 'sensor_v1/test2/runs'


def main():
    out = {}
    miss = 0
    for arena in ARENAS:
        for p in sorted((EXP / f'sensor_v1/test2_{arena}/picks').glob('*.json')):
            s = json.load(open(p)); g = s['group']
            rec = {'arena': arena, 'arms': {}}
            for arm, v in s['arms'].items():
                if v is None:
                    continue
                d = RUNS / v['route_id']
                if not (d / 'outcome.json').exists() or not (d / 'trajectory.npz').exists():
                    miss += 1
                    continue
                rec['arms'][arm] = dict(route_id=v['route_id'], pool=v['pool'], index=v['index'],
                                        mean_speed=v['mean_speed'],
                                        **{k: v[k] for k in v if k.startswith(('risk_', 'rank_'))},
                                        **labels(str(d)))
            out[g] = rec
    json.dump(out, open(Path(__file__).resolve().parent / 'driven_outcomes.json', 'w'))
    n = sum(len(v['arms']) for v in out.values())
    print(f'{len(out)} groups, {n} arm-runs labelled, {miss} missing run dirs')
    for arm in ('n2', 'e0', 'd', 'straight6', 'n2_fixed2', 'e0_fixed2', 'd_fixed2'):
        u = [v['arms'][arm]['unsafe'] for v in out.values() if arm in v['arms']]
        f = [v['arms'][arm]['fail'] for v in out.values() if arm in v['arms']]
        if u:
            print(f'  {arm:11s} n={len(u):5d}  unsafe {100 * np.mean(u):5.2f}%  fail {100 * np.mean(f):5.2f}%')


if __name__ == '__main__':
    main()
