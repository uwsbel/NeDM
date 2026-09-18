"""Moving-start ground truth vs velocity-aware models: score the remaining route from each spawned anchor with the
context velocity set to the MEASURED first-frame speed of that drive, for every stage-C ensemble, then compare with the
realised outcome (pooled / per-v0 AUC, calibration by v0, and the per-anchor direction check).
  python scripts/n2_moving_analyze.py --moving artifacts/traverse/crm_night2_v1/moving_v1 --reanchor .../reanchor_rigid.npz --ckpts 'stageC/rigid_*_s*.pt' --out OUT.json"""
import argparse, glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n2_arch_train as NT

ap = argparse.ArgumentParser()
ap.add_argument('--moving', required=True); ap.add_argument('--reanchor', required=True); ap.add_argument('--ckpts', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
idx = json.load(open(a.moving + '/index.json')); D = np.load(a.reanchor, allow_pickle=True)
drives = []
for it in idx:
    for v0 in (0, 2, 4):
        d = f"{a.moving}/out/runs/{it['case']}__v{v0}"
        if not os.path.exists(d + '/episode_complete.json'): continue
        o = json.load(open(d + '/outcome.json')); z = np.load(d + '/trajectory.npz')
        drives.append(dict(row=it['row'], v0=v0, vx0=float(z['state'][0, 0]), fail=int(o['status'] != 'goal_reached'), status=o['status'], anchor=it['anchor'], case=it['case'],
                           vx_anchor=it['vx_anchor'], recorded_fail=it['recorded_fail'], elapsed=o['elapsed_s']))
print(len(drives), 'drives found;', 'fail rate by v0:', {v: round(float(np.mean([d['fail'] for d in drives if d['v0'] == v])), 3) for v in (0, 2, 4)})
rows = np.array([d['row'] for d in drives]); X = D['X'][rows].astype(np.float32); ctx22 = D['ctx'][rows].astype(np.float32).copy()
ctx22[:, 0] = [d['vx0'] for d in drives]; ctx22[:, 1:7] = 0.0                       # spawned vehicle: measured forward speed, no other motion
y = np.array([d['fail'] for d in drives], float); v0 = np.array([d['v0'] for d in drives])
tags = sorted({os.path.basename(p).rsplit('_s', 1)[0] for p in glob.glob(a.ckpts)})
out = {'n_drives': len(drives), 'fail_by_v0': {int(v): float(y[v0 == v].mean()) for v in (0, 2, 4)}, 'arms': {}}
for tag in tags:
    members = sorted(glob.glob(f'{os.path.dirname(a.ckpts)}/{tag}_s*.pt')); zs = []
    for p in members:
        ck = torch.load(p, map_location=NT.DEV, weights_only=False)
        m = NT.build(ck['arch'], ck['cin'], ck['nctx'], ck['energy']).to(NT.DEV); m.load_state_dict(ck['state']); m.eval()
        Xi = X.copy()
        if ck['vplane']:
            Xi = np.concatenate([Xi, np.broadcast_to((ctx22[:, 0] / 6.0)[:, None, None, None], (len(Xi), 1, 96, 32)).astype(np.float32)], 1); Xi[:, [4, 5]] = Xi[:, [5, 4]]
        nm = ck['norm']; ci = nm['cont_index']; Xi[:, ci] = (Xi[:, ci] - nm['mu'][None, :, None, None]) / nm['sd'][None, :, None, None]
        Xi = np.concatenate([Xi, np.ones((len(Xi), 1, 96, 32), np.float32)], 1)
        c = (ctx22[:, ck['ctx_cols']] - ck['ctx_mu']) / ck['ctx_sd']
        with torch.no_grad():
            z = np.concatenate([NT.route_logit(m(torch.tensor(Xi[i:i + 512], device=NT.DEV), torch.tensor(c[i:i + 512], dtype=torch.float32, device=NT.DEV))['haz']).cpu().numpy() for i in range(0, len(Xi), 512)])
        zs.append(z)
    z = np.mean(zs, 0); P = 1 - np.exp(-np.exp(z))
    res = dict(members=len(members), auc_all=NT.auc(y, z), auc_by_v0={int(v): NT.auc(y[v0 == v], z[v0 == v]) for v in (0, 2, 4)},
               mean_P_by_v0={int(v): float(P[v0 == v].mean()) for v in (0, 2, 4)})
    # per-anchor direction check: anchors driven at both v0 = 0 and 4 with different outcomes
    by = {}
    for d, pp in zip(drives, P): by.setdefault(d['anchor'], {})[d['v0']] = (d['fail'], pp)
    agree = tot = 0; delta = []
    for an, dd in by.items():
        if 0 in dd and 4 in dd:
            delta.append(dd[4][1] - dd[0][1])
            if dd[0][0] != dd[4][0]:
                tot += 1; agree += int(np.sign(dd[4][1] - dd[0][1]) == np.sign(dd[4][0] - dd[0][0]))
    res.update(direction_agree=agree, direction_total=tot, mean_dP_v4_minus_v0=float(np.mean(delta)) if delta else float('nan'))
    out['arms'][tag] = res
    print(f"{tag:45s} AUC all {res['auc_all']:.3f}  by v0 {[round(res['auc_by_v0'][v], 3) for v in (0, 2, 4)]}  mean P by v0 {[round(res['mean_P_by_v0'][v], 3) for v in (0, 2, 4)]}  direction {agree}/{tot}  dP(4-0) {res['mean_dP_v4_minus_v0']:+.3f}")
json.dump(out, open(a.out, 'w'), indent=1, default=float)
