"""Conditional on the argmin moving: how far apart are the two picks in each ordering, and how much does each
scorer think it gains/loses by the swap. Writes flip_detail.json."""
import numpy as np, json
from pathlib import Path
H = Path(__file__).resolve().parent
A = ['f104', 'g203', 'g216', 'g217', 'g228', 'g231']
D = {a: dict(np.load(H / f'scores/{a}_all.npz')) for a in A}
cat = lambda k: np.concatenate([D[a][k] for a in A], 0)
prob = lambda z: 1 - np.exp(-np.exp(z))
out = {}
for p in ('proposal', 'fixed2'):
    for m in ('n2', 'e0', 'd'):
        z1 = cat(f'{p}_{m}_v1'); z2 = cat(f'{p}_{m}_v2')
        i1 = z1.argmin(1); i2 = z2.argmin(1); fl = i1 != i2; g = np.arange(len(z1))
        r1 = (z2 < z2[g, i1][:, None]).sum(1)[fl]; r2 = (z1 < z1[g, i2][:, None]).sum(1)[fl]
        dP1 = prob(z1[g, i2])[fl] - prob(z1[g, i1])[fl]
        dP2 = prob(z2[g, i1])[fl] - prob(z2[g, i2])[fl]
        out[f'{p}/{m}'] = dict(n_flip=int(fl.sum()),
            v1pick_rank_under_v2=dict(median=float(np.median(r1)), p90=float(np.percentile(r1, 90)), max=int(r1.max())),
            v2pick_rank_under_v1=dict(median=float(np.median(r2)), p90=float(np.percentile(r2, 90)), max=int(r2.max())),
            median_dP_v1=float(np.median(dP1)), median_dP_v2=float(np.median(dP2)),
            frac_v1pick_still_top10_under_v2=float((r1 < 10).mean()),
            frac_v2pick_was_top10_under_v1=float((r2 < 10).mean()))
        print(f'{p:9s} {m:5s} flips {int(fl.sum()):5d}  v1pick rank under v2 med/p90/max '
              f'{np.median(r1):.0f}/{np.percentile(r1,90):.0f}/{r1.max()}  still top10 {100*(r1<10).mean():.0f}%  '
              f'median dP_v1 {np.median(dP1):+.2e}  dP_v2 {np.median(dP2):+.2e}')
json.dump(out, open(H / 'flip_detail.json', 'w'), indent=1)
