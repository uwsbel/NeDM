"""Per-candidate corridor diagnostics pooled over all 614,400 replayed candidates. Writes corridor_diag_summary.json."""
import numpy as np, json
from pathlib import Path
H = Path(__file__).resolve().parent
A = ['f104', 'g203', 'g216', 'g217', 'g228', 'g231']
D = {a: dict(np.load(H / f'scores/{a}_all.npz')) for a in A}
names = list(D['f104']['diag_names'].astype(str))
out = {}
for p in ('proposal', 'fixed2'):
    Dg = np.concatenate([D[a][f'{p}_diag'] for a in A], 0)
    rec = {}
    for c in ('mean_abs_delev', 'p95_abs_delev', 'mean_v1_disp_m', 'max_v1_disp_m', 'valid_frac_v1',
              'valid_frac_v2', 'mean_radius_m', 'max_radius_m', 'p95_abs_grade'):
        v = Dg[:, :, names.index(c)].ravel()
        rec[c] = dict(mean=float(v.mean()), p50=float(np.median(v)), p95=float(np.percentile(v, 95)), max=float(v.max()))
    out[p] = dict(n_candidates=int(Dg.shape[0] * Dg.shape[1]), **rec)
    print(f'-- {p}: {Dg.shape[0] * Dg.shape[1]} candidates')
    for c, r in rec.items():
        print(f'   {c:16s} mean {r["mean"]:.4f}  p50 {r["p50"]:.4f}  p95 {r["p95"]:.4f}  max {r["max"]:.4f}')
json.dump(out, open(H / 'corridor_diag_summary.json', 'w'), indent=1)
