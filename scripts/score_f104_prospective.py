"""Score the un-driven held-out test routes BEFORE they are run in Chrono.

Writes a locked prediction file. Chrono then adjudicates: no outcome for these
routes exists anywhere at the time this runs.
"""
import json, glob, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from build_f104_planner_dataset import (sample_map, resample_route, SCAL_KEYS,
                                        N_STATION, N_LATERAL, HALF_WIDTH_M, ELEV_SCALE)
from train_f104_risk_head import RiskNet

ROOT = '/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_f104_50h_20260909'
SEED = int(os.environ.get('SEED', '0'))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def feats(wp, speeds, stations, headings):
    wp = np.asarray(wp, float)
    pts, grid = resample_route(wp, stations, N_STATION)
    d = np.gradient(pts, axis=0); tn = np.linalg.norm(d, axis=1, keepdims=True); tn[tn < 1e-9] = 1e-9
    tang = d / tn; norm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    off = np.linspace(-HALF_WIDTH_M, HALF_WIDTH_M, N_LATERAL)
    gx = pts[:, 0:1] + norm[:, 0:1] * off[None, :]
    gy = pts[:, 1:2] + norm[:, 1:2] * off[None, :]
    patch, vmask = sample_map(gx, gy); patch = patch.astype(np.float32)
    elev = np.where(vmask, patch[3] * ELEV_SCALE, np.nan)
    ds = float(np.mean(np.diff(grid))) if grid.size > 1 else 1.0
    centre = elev[:, N_LATERAL // 2]
    grade = np.gradient(np.nan_to_num(centre, nan=np.nanmean(centre)), max(ds, 1e-6))
    cross = (elev[:, -1] - elev[:, 0]) / (2 * HALF_WIDTH_M)
    rough = np.nanstd(elev, axis=1)
    fin = lambda a: np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    grade, cross, rough = fin(grade), fin(cross), fin(rough)
    sp = np.asarray(speeds, float); hdg = np.unwrap(np.asarray(headings, float))
    rl = float(np.sum(np.linalg.norm(np.diff(wp, axis=0), axis=1)))
    curv = np.abs(np.gradient(np.unwrap(np.arctan2(tang[:, 1], tang[:, 0])), max(ds, 1e-6)))
    s = dict(route_len_m=rl, straight_dist_m=float(np.linalg.norm(wp[-1] - wp[0])),
             tortuosity=float(rl / max(np.linalg.norm(wp[-1] - wp[0]), 1e-6)),
             max_abs_grade=float(np.max(np.abs(grade))), mean_abs_grade=float(np.mean(np.abs(grade))),
             p95_abs_grade=float(np.percentile(np.abs(grade), 95)),
             max_abs_cross=float(np.max(np.abs(cross))), mean_abs_cross=float(np.mean(np.abs(cross))),
             max_rough=float(np.max(rough)), mean_rough=float(np.mean(rough)),
             elev_gain_m=float(np.sum(np.clip(np.diff(np.nan_to_num(centre)), 0, None))),
             elev_range_m=float(np.nanmax(centre) - np.nanmin(centre)),
             max_curv=float(np.max(curv)), mean_curv=float(np.mean(curv)),
             total_heading_change=float(abs(hdg[-1] - hdg[0])) if hdg.size else 0.0,
             mean_ref_speed=float(np.mean(sp)) if sp.size else 0.0,
             min_ref_speed=float(np.min(sp)) if sp.size else 0.0,
             invalid_frac=float(1.0 - vmask.mean()))
    return patch, np.array([s[k] for k in SCAL_KEYS], np.float32)


def main():
    tasks = json.load(open(ROOT + '/prospective/f104_undriven_test.json'))
    # per-group anchor + case, taken from any already-driven route of that group
    ganchor, gcase = {}, {}
    for ep in glob.glob(ROOT + '/production_v2/runs/*'):
        g = os.path.basename(ep).split('_route_')[0]
        if g in ganchor:
            continue
        try:
            a = np.load(ep + '/anchor_state.npz', allow_pickle=True)
            c = json.load(open(ep + '/case.json'))
        except Exception:
            continue
        ganchor[g] = np.asarray(a['state'], np.float32)
        gcase[g] = c
    ck = torch.load(f'{ROOT}/risk_head_v1/risk_head_seed{SEED}.pt', map_location=DEV, weights_only=False)
    P, V, ids = [], [], []
    for t in tasks:
        g = t['group_id']
        if g not in ganchor:
            continue
        patch, scal = feats(t['waypoints'], t['speeds'], t['stations'], t['headings'])
        c = gcase[g]; lay = c.get('layout', {})
        gxy = np.asarray(c['goal_xy'], np.float32); sxy = np.asarray(lay.get('start_xy', [0, 0]), np.float32)
        rel = gxy - sxy
        extra = np.array([rel[0], rel[1], np.linalg.norm(rel), float(lay.get('start_yaw', 0.0))], np.float32)
        P.append(patch); V.append(np.concatenate([scal, ganchor[g], extra])); ids.append(t['id'])
    P = np.stack(P); V = np.stack(V)
    Pn = (P - ck['pmu']) / ck['psd']; Vn = (V - ck['mu']) / ck['sd']
    net = RiskNet(V.shape[1], 0).to(DEV); net.load_state_dict(ck['state_dict']); net.eval()
    with torch.no_grad():
        risk, tpred = [], []
        for i in range(0, len(P), 512):
            lf, lt = net(torch.tensor(Pn[i:i+512], device=DEV), torch.tensor(Vn[i:i+512], device=DEV))
            risk.append(torch.sigmoid(lf).cpu().numpy()); tpred.append(lt.cpu().numpy())
    risk = np.concatenate(risk); tpred = np.concatenate(tpred)
    out = [dict(id=i, predicted_risk=float(r), predicted_time_s=float(tt))
           for i, r, tt in zip(ids, risk, tpred)]
    p = f'{ROOT}/prospective/predictions_seed{SEED}.json'
    json.dump(dict(seed=SEED, n=len(out), model=f'risk_head_seed{SEED}.pt',
                   note='Locked before any Chrono run of these routes.',
                   predictions=out), open(p, 'w'), indent=2)
    print(f'scored {len(out)} un-driven test routes -> {p}')
    print(f'  predicted risk: mean {risk.mean():.4f}  min {risk.min():.4f}  max {risk.max():.4f}')
    print(f'  flagged high-risk (>0.5): {(risk>0.5).sum()} / {len(risk)}')


if __name__ == '__main__':
    main()
