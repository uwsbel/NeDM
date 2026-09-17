"""Vehicle-included depth-input pilot: three matched corridor inputs, identical pools and frozen checkpoints.

  A  vehicle-free frame, no exclusion  (the current v2 input)
  B  vehicle-free frame + exclusion mask
  C  vehicle-included frame + the same exclusion mask
Scores every candidate of the same 256-route pool with the frozen matched ensembles (height and depth), and reports
score shift, rank correlation, and whether the chosen route changes.
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sensor_map_v2 as M2, sensor_dataset_v2 as V2, vehicle_corridor as VC
import gen_planner as P

V = 'artifacts/traverse/fdm_f104_50h_20260909/sensor_v2'


def pool_for(case_path, group):
    case = json.load(open(case_path)); lay = case['layout']
    pose = [lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']]
    base = {k: np.asarray(v, float) for k, v in json.load(open(Path(case_path).parent / 'routes' / group / 'route_00.json')).items()
            if k in ('waypoints', 'speeds', 'stations', 'headings')}
    base['meta'] = {}
    seed = int(hashlib.md5((group + 'v2_proposal').encode()).hexdigest()[:8], 16)
    cands, _ = P.proposal_pool(base, pose, np.random.default_rng(seed))
    return case, pose, cands


def corridors(cands, exclude):
    X, L, info = [], [], []
    for r in cands:
        x, l, i = VC.tensor12_excluded(np.asarray(r['waypoints']), np.asarray(r['speeds']), np.asarray(r['stations']), exclude)
        X.append(x); L.append(l); info.append(i)
    return np.stack(X), np.asarray(L, np.float32), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--margin', type=float, default=1.5)
    ap.add_argument('--models', default='H=matched/matched_H_s*.pt,Dabs=matched/matched_Dabs_s*.pt')
    ap.add_argument('--out', default=V + '/vehicle_pilot_results.json')
    a = ap.parse_args()
    models = {k: P.GridRiskModel(f'{V}/{v}') for k, v in (kv.split('=', 1) for kv in a.models.split(','))}
    sel = json.load(open(f'{V}/pilot_vehicle_cases.json'))
    out = []
    for k, meta in enumerate(sel):
        g, arena = meta['group'], meta['arena']
        frame = f'{V}/vehicle_frames/{g}'
        obs = json.load(open(frame + '/observation.json')); vpose = np.asarray(obs['measured_pose_xy_yaw'], float)
        case, pose, cands = pool_for(meta['case'], g)
        ctx_of = lambda L: P.geom_ctx(case['layout']['start_xy'], case['goal_xy'], case['layout']['start_yaw'], L)
        res = {'group': g, 'arena': arena, 'n_candidates': len(cands), 'measured_pose': vpose.tolist(),
               'margin_m': a.margin, 'max_grade': meta['max_grade'], 'start_slope': meta['start_slope']}
        # --- A and B: vehicle-free grid ---
        V2.init_grid(f'{V}/grids/arena_{arena}')
        mask = VC.exclusion_mask(vpose, a.margin)
        res['excluded_cells'] = int(mask.sum())
        XA, LA, iA = corridors(cands, None)
        XB, LB, iB = corridors(cands, mask)
        # --- C: vehicle-included grid, same mask ---
        grid = M2.grid_from_capture(frame)
        V2.G.update(z=grid['z'], range_m=grid['range_m'], sec=grid['sec'], rgb=grid['rgb'], cover=grid['cover'],
                    mpp=grid['meta']['mpp'], half=grid['meta']['half_extent_m'], n=grid['meta']['n'],
                    cam_h=grid['meta']['camera_height_m'])
        XC, LC, iC = corridors(cands, mask)
        res['invalid_fraction'] = {'A': float(np.mean([i['invalid_fraction'] for i in iA])),
                                   'B': float(np.mean([i['invalid_fraction'] for i in iB])),
                                   'C': float(np.mean([i['invalid_fraction'] for i in iC]))}
        res['reference_station'] = {v: float(np.mean([i['reference_station'] or 0 for i in inf]))
                                    for v, inf in (('A', iA), ('B', iB), ('C', iC))}
        res['fully_invalid_stations'] = {v: float(np.mean([i['invalid_stations'] for i in inf]))
                                         for v, inf in (('A', iA), ('B', iB), ('C', iC))}
        # terrain difference where both are valid, inside the corridor (C vs B): should be ~0 if the mask is right
        both = (XB[:, 4] > 0.5) & (XC[:, 4] > 0.5)
        res['abs_z_diff_valid_C_vs_B'] = dict(mean=float(np.abs(XB[:, 0] - XC[:, 0])[both].mean()),
                                              p99=float(np.percentile(np.abs(XB[:, 0] - XC[:, 0])[both], 99)),
                                              max=float(np.abs(XB[:, 0] - XC[:, 0])[both].max()))
        for name, mdl in models.items():
            zs = {}
            for v, (X, L) in (('A', (XA, LA)), ('B', (XB, LB)), ('C', (XC, LC))):
                zs[v] = mdl.score(X.astype(np.float32), ctx_of(L))[0]
            pick = {v: int(np.argmin(zs[v])) for v in zs}
            rank = lambda z: np.argsort(np.argsort(z))
            res[name] = {
                'pick': pick,
                'pick_same_AB': bool(pick['A'] == pick['B']), 'pick_same_AC': bool(pick['A'] == pick['C']),
                'pick_same_BC': bool(pick['B'] == pick['C']),
                'spearman_AB': float(np.corrcoef(rank(zs['A']), rank(zs['B']))[0, 1]),
                'spearman_AC': float(np.corrcoef(rank(zs['A']), rank(zs['C']))[0, 1]),
                'spearman_BC': float(np.corrcoef(rank(zs['B']), rank(zs['C']))[0, 1]),
                'logit_shift_AC': dict(mean=float((zs['C'] - zs['A']).mean()), p95=float(np.percentile(np.abs(zs['C'] - zs['A']), 95))),
                'logit_shift_BC': dict(mean=float((zs['C'] - zs['B']).mean()), p95=float(np.percentile(np.abs(zs['C'] - zs['B']), 95))),
                'rank_of_A_pick_in_C': int((zs['C'] < zs['C'][pick['A']]).sum()),
                'rank_of_C_pick_in_A': int((zs['A'] < zs['A'][pick['C']]).sum()),
                'risk_A_pick': float(1 - np.exp(-np.exp(zs['A'][pick['A']]))),
                'risk_C_pick': float(1 - np.exp(-np.exp(zs['C'][pick['C']]))),
            }
            if k == 0:
                np.savez_compressed(f'{V}/vehicle_pilot_example_{name}.npz', XA=XA[pick['A']], XB=XB[pick['A']],
                                    XC=XC[pick['A']], zA=zs['A'], zB=zs['B'], zC=zs['C'])
        # keep the chosen routes for the Chrono check
        res['routes'] = {v: {'waypoints': np.asarray(cands[p]['waypoints']).tolist(),
                             'speeds': np.asarray(cands[p]['speeds']).tolist(),
                             'stations': np.asarray(cands[p]['stations']).tolist(),
                             'headings': np.asarray(cands[p]['headings']).tolist()}
                         for v, p in ((f'{n}_{v}', res[n]['pick'][v]) for n in models for v in ('A', 'B', 'C'))}
        out.append(res)
        print(f"{g}: excluded {res['excluded_cells']} cells, invalid corridor fraction A/B/C "
              f"{res['invalid_fraction']['A']:.3f}/{res['invalid_fraction']['B']:.3f}/{res['invalid_fraction']['C']:.3f}; "
              + '  '.join(f"{n}: pick same A->C {res[n]['pick_same_AC']}, rho {res[n]['spearman_AC']:.3f}" for n in models), flush=True)
    json.dump(out, open(a.out, 'w'), indent=1, default=float)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
