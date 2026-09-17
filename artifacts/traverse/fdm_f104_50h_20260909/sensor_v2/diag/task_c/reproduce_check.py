"""Does the regenerated pool reproduce the one the sensor_v1 test-2 actually used?

For every group, compare the saved pick json (risk of the chosen candidate under each of the three models, its
rank, the pool sizes and the rejection-sampling try count) with the v1 scores recomputed here at the same index.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = Path('/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_f104_50h_20260909')
ARENAS = ['f104', 'g203', 'g216', 'g217', 'g228', 'g231']
ARM_MODEL = {'n2': ('n2', 'proposal'), 'e0': ('e0', 'proposal'), 'd': ('d', 'proposal'),
             'straight6': ('n2', 'proposal'),
             'n2_fixed2': ('n2', 'fixed2'), 'e0_fixed2': ('e0', 'fixed2'), 'd_fixed2': ('d', 'fixed2')}


def prob(z):
    return 1 - np.exp(-np.exp(z))


def main():
    rel, rankbad, arg_ok, arg_n, tries_ok, tries_n = [], 0, 0, 0, 0, 0
    for a in ARENAS:
        z = np.load(HERE / f'scores/{a}_all.npz')
        gi = {g: i for i, g in enumerate(z['group'].astype(str))}
        for p in sorted((EXP / f'sensor_v1/test2_{a}/picks').glob('*.json')):
            s = json.load(open(p)); g = s['group']
            if g not in gi:
                continue
            i = gi[g]
            tries_n += 1; tries_ok += int(s['tries'] == int(z['tries'][i]))
            for arm, v in s['arms'].items():
                if v is None:
                    continue
                m, pool = ARM_MODEL[arm]
                zz = z[f'{pool}_{m}_v1'][i]
                for k in ('n2', 'e0', 'd'):
                    if f'risk_{k}' not in v:
                        continue
                    mine = prob(z[f'{pool}_{k}_v1'][i][v['index']])
                    rel.append(abs(mine - v[f'risk_{k}']) / max(v[f'risk_{k}'], 1e-12))
                    rankbad += int((z[f'{pool}_{k}_v1'][i] < z[f'{pool}_{k}_v1'][i][v['index']]).sum() != v[f'rank_{k}'])
                if arm in ('n2', 'e0', 'd', 'n2_fixed2', 'e0_fixed2', 'd_fixed2'):
                    arg_n += 1; arg_ok += int(int(zz.argmin()) == v['index'])
    rel = np.array(rel)
    print(f'groups checked: {tries_n}; rejection-sampling try count identical: {tries_ok}/{tries_n}')
    print(f'risk values compared: {len(rel)}; median rel.err {np.median(rel):.2e}, p99 {np.percentile(rel, 99):.2e}, '
          f'max {rel.max():.2e}; rank mismatches {rankbad}')
    print(f'saved argmin reproduced: {arg_ok}/{arg_n}')
    json.dump(dict(groups=tries_n, tries_identical=tries_ok, n_risk=len(rel),
                   median_rel_err=float(np.median(rel)), p99_rel_err=float(np.percentile(rel, 99)),
                   max_rel_err=float(rel.max()), rank_mismatches=int(rankbad),
                   argmin_reproduced=arg_ok, argmin_total=arg_n),
              open(HERE / 'reproduce_check.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
