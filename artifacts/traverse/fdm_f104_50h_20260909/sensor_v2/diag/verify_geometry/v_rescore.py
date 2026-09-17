"""Spot re-score: rebuild 8 test-2 groups' candidate pools, rebuild both corridors, re-run the three
ensembles, and compare against the scores saved in task_c/scores/*.npz.  Also checks corridor_v2's
elevation channel against an independent bilinear read of the v2 grid.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path('/home/harry/NeDM-traverse_mppi')
EXP = ROOT / 'artifacts/traverse/fdm_f104_50h_20260909'
TC = EXP / 'sensor_v2/diag/task_c'
sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(TC))
import gen_planner as P
from corridor_v2 import V2Map, tensor10_v2, corridor_xy

ARENA, TAG = 'arena_g231', 'g231'
NG = 8
S = dict(np.load(TC / 'scores' / f'{TAG}_all.npz', allow_pickle=True))
gsaved = [str(g) for g in S['group']]

P.set_sensor_map(str(EXP / 'sensor_v1/maps' / ARENA))
vm = V2Map(EXP / 'sensor_v2/grids' / ARENA)
models = {'n2': P.RiskModel(),
          'e0': P.SensorRiskModel(str(EXP / 'sensor_v1/final/E0_s*.pt')),
          'd': P.SensorRiskModel(str(EXP / 'sensor_v1/final/D_s*.pt'))}

def seed(g, t): return int(hashlib.md5((g + t).encode()).hexdigest()[:8], 16)

cases = sorted(str(p) for p in (EXP / f'sensor_v1/cases2_{TAG}/cases').glob('*.json') if p.name != 'cases.json')
rep = {'checked_groups': [], 'max_abs_score_diff': {}, 'elev_channel_max_abs_diff': [], 'argmin_match': 0, 'argmin_total': 0}
diffs = {f'{p}_{m}_{v}': 0.0 for p in ('proposal', 'fixed2') for m in models for v in ('v1', 'v2')}
for cp in cases[:NG]:
    case = json.load(open(cp)); g = case['id']; lay = case['layout']
    if g not in gsaved: continue
    i = gsaved.index(g); rep['checked_groups'].append(g)
    pose = [lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']]
    base = {k: np.asarray(v, float) for k, v in
            json.load(open(Path(cp).parent / 'routes' / g / 'route_00.json')).items()
            if k in ('waypoints', 'speeds', 'stations', 'headings')}
    base['meta'] = {}
    c1, _ = P.proposal_pool(base, pose, np.random.default_rng(seed(g, 's1_proposal')))
    c2 = P.fixed2_pool(base, pose, np.random.default_rng(seed(g, 's1_fixed2')))
    for pool, cands in (('proposal', c1), ('fixed2', c2)):
        X1, L1 = P.corridors10(cands)
        X2 = np.empty_like(X1)
        for j, r in enumerate(cands):
            wp = np.asarray(r['waypoints']); sp = np.asarray(r['speeds']); st = np.asarray(r['stations'])
            X2[j] = tensor10_v2(vm, wp, sp, st)[0]
            if j < 4:  # independent check of the elevation channel
                gx, gy, _ = corridor_xy(wp, st)
                mpp, half, n = vm.mpp, vm.half, vm.n
                col = (gx + half) / mpp - 0.5; row = (gy + half) / mpp - 0.5
                r0 = np.clip(np.floor(row).astype(int), 0, n - 2); c0 = np.clip(np.floor(col).astype(int), 0, n - 2)
                fr = np.clip(row - r0, 0, 1); fc = np.clip(col - c0, 0, 1)
                Z = vm.z.astype(np.float64)
                zz = (Z[r0, c0] * (1 - fr) * (1 - fc) + Z[r0, c0 + 1] * (1 - fr) * fc
                      + Z[r0 + 1, c0] * fr * (1 - fc) + Z[r0 + 1, c0 + 1] * fr * fc)
                ok = np.isfinite(X2[j, 0]) & (X2[j, 4] > 0.5)
                e0 = zz[0, 16]
                rep['elev_channel_max_abs_diff'].append(float(np.abs((zz - e0)[ok] - X2[j, 0][ok]).max()))
        ctx = P.geom_ctx(lay['start_xy'], case['goal_xy'], lay['start_yaw'], L1)
        for k, m in models.items():
            for ver, X in (('v1', X1.astype(np.float16)), ('v2', X2.astype(np.float16))):
                z, _ = m.score(X.astype(np.float32)[:, :5] if k == 'n2' else X.astype(np.float32), ctx)
                sv = S[f'{pool}_{k}_{ver}'][i]
                diffs[f'{pool}_{k}_{ver}'] = max(diffs[f'{pool}_{k}_{ver}'], float(np.abs(z - sv).max()))
                rep['argmin_total'] += 1
                rep['argmin_match'] += int(np.argmin(z) == np.argmin(sv))
rep['max_abs_score_diff'] = diffs
rep['elev_channel_max_abs_diff'] = float(max(rep['elev_channel_max_abs_diff']))
print(json.dumps(rep, indent=1))
json.dump(rep, open(Path(__file__).parent / 'v_rescore.json', 'w'), indent=1)
