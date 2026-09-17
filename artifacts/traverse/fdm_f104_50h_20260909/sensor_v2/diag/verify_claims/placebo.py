"""Placebo test for TASK C's two outcome-carrying claims.

Both claims are measured on routes that were SELECTED as the v1 argmin of one of the arms. Any zero-mean
perturbation of the scores raises the expected rank of an argmin-selected route (optimiser's curse), and
regresses anomalously-low scores (which is what an unsafe pick is) further than typical ones. So the null is
not "no change", it is "a random perturbation of the same size".

Null A: permute the observed dz = z2 - z1 across the 256 candidates within each group (same marginal
        magnitudes, same per-group scale, no geometry information).
Null B: keep each candidate's |dz| but randomise its sign within the group (preserves any correlation of
        movement magnitude with route geometry; destroys the direction).

Reports, for the real v2 and for each null: within-group pair accuracy (v2 - v1), and the driven-route rank
gap (unsafe minus safe).
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
REPS = 400

D, gidx = {}, {}
for a in ARENAS:
    z = np.load(TC / f'scores/{a}_all.npz')
    D[a] = {k: z[k] for k in z}
    for i, g in enumerate(z['group'].astype(str)):
        gidx[g] = (a, i)
drv = json.load(open(TC / 'driven_outcomes.json'))

bygroup, driven_rows = {}, []
for g, rec in drv.items():
    if g not in gidx:
        continue
    for arm, v in rec['arms'].items():
        m, p = ARM_MODEL[arm]
        bygroup.setdefault((g, p), {}).setdefault(v['index'], v['unsafe'])
        driven_rows.append((arm, g, p, m, v['index'], v['unsafe']))


def pair_acc(p, m, z2_of):
    n = c1 = c2 = 0
    for (g, pp), seen in bygroup.items():
        if pp != p:
            continue
        a, i = gidx[g]
        z1 = D[a][f'{p}_{m}_v1'][i]; z2 = z2_of(a, i, p, m)
        un = [k for k in seen if seen[k] == 1]; sa = [k for k in seen if seen[k] == 0]
        for x in un:
            for y in sa:
                n += 1; c1 += z1[x] > z1[y]; c2 += z2[x] > z2[y]
    return c1 / n, c2 / n, n


def rank_gap(arm, z2_of):
    m, p = ARM_MODEL[arm]
    du, ds = [], []
    for a2, g, pp, mm, j, uns in driven_rows:
        if a2 != arm:
            continue
        a, i = gidx[g]
        z1 = D[a][f'{p}_{m}_v1'][i]; z2 = z2_of(a, i, p, m)
        dr = int((z2 < z2[j]).sum()) - int((z1 < z1[j]).sum())
        (du if uns else ds).append(dr)
    return float(np.mean(du)) - float(np.mean(ds)), float(np.mean(du)), float(np.mean(ds))


real = lambda a, i, p, m: D[a][f'{p}_{m}_v2'][i]
rng = np.random.default_rng(7)

# pre-draw per-rep permutations / sign patterns, shared across models so the comparison is paired
DZ = {(a, p, m): D[a][f'{p}_{m}_v2'] - D[a][f'{p}_{m}_v1'] for a in ARENAS for p in ('proposal', 'fixed2')
      for m in ('n2', 'e0', 'd')}

out = {}
print('== within-group pair accuracy (driven unsafe vs driven safe) ==')
for p in ('proposal', 'fixed2'):
    for m in ('n2', 'e0', 'd'):
        v1a, v2a, n = pair_acc(p, m, real)
        nulls = {'permute_dz': [], 'sign_flip_dz': []}
        for _ in range(REPS):
            perm = {a: rng.permutation(256) for a in ARENAS}   # not used; per-group below
            def z2_perm(a, i, pp, mm, _cache={}):
                dz = DZ[(a, pp, mm)][i]
                return D[a][f'{pp}_{mm}_v1'][i] + rng.permutation(dz)
            def z2_sign(a, i, pp, mm):
                dz = DZ[(a, pp, mm)][i]
                return D[a][f'{pp}_{mm}_v1'][i] + np.abs(dz) * rng.choice([-1.0, 1.0], dz.shape)
            nulls['permute_dz'].append(pair_acc(p, m, z2_perm)[1] - v1a)
            nulls['sign_flip_dz'].append(pair_acc(p, m, z2_sign)[1] - v1a)
        rec = dict(n_pairs=n, v1=v1a, v2=v2a, real_gain=v2a - v1a)
        for k, vals in nulls.items():
            vals = np.array(vals)
            rec[k] = dict(mean=float(vals.mean()), sd=float(vals.std()),
                          p95=float(np.percentile(vals, 95)),
                          p_one_sided=float((vals >= v2a - v1a).mean()))
        out[f'pair/{p}/{m}'] = rec
        print(f"{p:9s} {m:3s} v1={v1a:.3f} v2={v2a:.3f} gain={v2a-v1a:+.4f} | "
              f"permute null {rec['permute_dz']['mean']:+.4f}+-{rec['permute_dz']['sd']:.4f} "
              f"p={rec['permute_dz']['p_one_sided']:.3f} | "
              f"signflip null {rec['sign_flip_dz']['mean']:+.4f}+-{rec['sign_flip_dz']['sd']:.4f} "
              f"p={rec['sign_flip_dz']['p_one_sided']:.3f}")

print('\n== driven-route rank movement: unsafe minus safe ==')
for arm in ARM_MODEL:
    g0, u0, s0 = rank_gap(arm, real)
    nulls = []
    for _ in range(REPS // 2):
        def z2_perm(a, i, pp, mm):
            dz = DZ[(a, pp, mm)][i]
            return D[a][f'{pp}_{mm}_v1'][i] + rng.permutation(dz)
        nulls.append(rank_gap(arm, z2_perm)[0])
    nulls = np.array(nulls)
    out[f'rank/{arm}'] = dict(real_gap=g0, mean_drank_unsafe=u0, mean_drank_safe=s0,
                              null_mean=float(nulls.mean()), null_sd=float(nulls.std()),
                              null_p95=float(np.percentile(nulls, 95)),
                              p_one_sided=float((nulls >= g0).mean()))
    print(f"{arm:11s} real gap {g0:+7.2f} (unsafe {u0:+6.2f}, safe {s0:+6.2f}) | permute null "
          f"{nulls.mean():+6.2f}+-{nulls.std():.2f} p={float((nulls >= g0).mean()):.3f}")

json.dump(out, open(HERE / 'placebo.json', 'w'), indent=1)
print('wrote', HERE / 'placebo.json')
