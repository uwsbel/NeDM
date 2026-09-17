"""Does the corrected geometry narrow the gap between the raw-depth arm and the height arms?

This is the question behind "correcting the geometry and keeping absolute range should make a direct-depth model
match the height model". TASK C reports each arm's v1->v2 change separately; here the quantity of interest is the
DIFFERENCE between arms, under v1 and under v2, with a group-clustered bootstrap on the difference-in-differences.

Two endpoints, both from task_c/scores + task_c/driven_outcomes.json:
  * within-group pair accuracy on driven routes with opposite outcomes (task C's primary label-carrying test);
  * pooled AUC over the arm's own driven routes.
Also: the fraction of groups where the v2 argmin of one model equals the v2 argmin of another (agreement),
to see whether v2 makes the depth model behave more like the height model.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TC = HERE.parent / 'task_c'
ARENAS = ['f104', 'g203', 'g216', 'g217', 'g228', 'g231']
ARM_MODEL = {'n2': ('n2', 'proposal'), 'e0': ('e0', 'proposal'), 'd': ('d', 'proposal'),
             'straight6': ('n2', 'proposal'),
             'n2_fixed2': ('n2', 'fixed2'), 'e0_fixed2': ('e0', 'fixed2'), 'd_fixed2': ('d', 'fixed2')}

D, gidx = {}, {}
for a in ARENAS:
    z = np.load(TC / f'scores/{a}_all.npz')
    D[a] = {k: z[k] for k in z}
    for i, g in enumerate(z['group'].astype(str)):
        gidx[g] = (a, i)
drv = json.load(open(TC / 'driven_outcomes.json'))
bygroup = {}
for g, rec in drv.items():
    if g not in gidx:
        continue
    for arm, v in rec['arms'].items():
        m, p = ARM_MODEL[arm]
        bygroup.setdefault((g, p), {}).setdefault(v['index'], v['unsafe'])

out = {}
rng = np.random.default_rng(3)
print('== pair accuracy per model, and the N2 - D gap, under v1 and v2 ==')
for p in ('proposal', 'fixed2'):
    # per-group pair outcomes for each model
    G = sorted({g for (g, pp) in bygroup if pp == p})
    per = {m: [] for m in ('n2', 'e0', 'd')}
    keep = []
    for g in G:
        seen = bygroup[(g, p)]
        un = [k for k in seen if seen[k] == 1]; sa = [k for k in seen if seen[k] == 0]
        if not un or not sa:
            continue
        a, i = gidx[g]
        keep.append(g)
        for m in ('n2', 'e0', 'd'):
            z1 = D[a][f'{p}_{m}_v1'][i]; z2 = D[a][f'{p}_{m}_v2'][i]
            per[m].append(np.array([[z1[x] > z1[y], z2[x] > z2[y]] for x in un for y in sa], float))
    ng = len(keep)
    acc = lambda m, col, ix: np.concatenate([per[m][j] for j in ix], 0)[:, col].mean()
    allix = np.arange(ng)
    base = {m: (acc(m, 0, allix), acc(m, 1, allix)) for m in per}
    for m in per:
        print(f'  {p:9s} {m:3s}  v1 {base[m][0]:.3f}  v2 {base[m][1]:.3f}  gain {base[m][1]-base[m][0]:+.3f}')
    rec = {'n_groups': ng, 'acc': {m: dict(v1=base[m][0], v2=base[m][1]) for m in per}}
    for hi in ('n2', 'e0'):
        g1 = base[hi][0] - base['d'][0]; g2 = base[hi][1] - base['d'][1]
        bs = []
        for _ in range(4000):
            ix = rng.integers(0, ng, ng)
            bs.append((acc(hi, 1, ix) - acc('d', 1, ix)) - (acc(hi, 0, ix) - acc('d', 0, ix)))
        bs = np.array(bs)
        rec[f'{hi}_minus_d'] = dict(gap_v1=g1, gap_v2=g2, change=g2 - g1,
                                    ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        print(f'    gap {hi} - d: v1 {g1:+.3f} -> v2 {g2:+.3f}   change {g2-g1:+.3f} '
              f'[{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]')
    out[f'pairs/{p}'] = rec

print('\n== how often does the depth model pick the same route as the height model? ==')
for p in ('proposal', 'fixed2'):
    for ver in ('v1', 'v2'):
        z = {m: np.concatenate([D[a][f'{p}_{m}_{ver}'] for a in ARENAS], 0) for m in ('n2', 'e0', 'd')}
        am = {m: z[m].argmin(1) for m in z}
        r = dict(n2_vs_d=float((am['n2'] == am['d']).mean()), e0_vs_d=float((am['e0'] == am['d']).mean()),
                 n2_vs_e0=float((am['n2'] == am['e0']).mean()))
        out[f'agree/{p}/{ver}'] = r
        print(f'  {p:9s} {ver}: n2==d {100*r["n2_vs_d"]:5.1f}%   e0==d {100*r["e0_vs_d"]:5.1f}%   '
              f'n2==e0 {100*r["n2_vs_e0"]:5.1f}%')

print('\n== rank correlation between models within a group (are the scorers converging under v2?) ==')
def spearman(a, b):
    ra = np.argsort(np.argsort(a, 1), 1).astype(float); rb = np.argsort(np.argsort(b, 1), 1).astype(float)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
for p in ('proposal', 'fixed2'):
    for ver in ('v1', 'v2'):
        z = {m: np.concatenate([D[a][f'{p}_{m}_{ver}'] for a in ARENAS], 0) for m in ('n2', 'e0', 'd')}
        r = dict(n2_d=float(spearman(z['n2'], z['d']).mean()), e0_d=float(spearman(z['e0'], z['d']).mean()),
                 n2_e0=float(spearman(z['n2'], z['e0']).mean()))
        out[f'rho_models/{p}/{ver}'] = r
        print(f'  {p:9s} {ver}: rho(n2,d) {r["n2_d"]:.3f}  rho(e0,d) {r["e0_d"]:.3f}  rho(n2,e0) {r["n2_e0"]:.3f}')

json.dump(out, open(HERE / 'gap_d_vs_height.json', 'w'), indent=1)
print('\nwrote', HERE / 'gap_d_vs_height.json')
