"""gen_v1 data collection read-out: offline ranking on the new arenas' designed routes (12 per start/goal).

Scores every collected route with the frozen model, the hand rule and speed alone, and reports ranking AUC on the
unsafe label: P pooled, W within a start/goal, G within a start/goal at the same speed profile. Reference: the same
metrics on f104's held-out test split of designed routes (station_ds_all split == test).
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_planner as P
from f104_n2_analyze import labels
from f104_night_train import auc, cell_auc

G = 'artifacts/traverse/fdm_f104_50h_20260909/gen_v1'
NEW = ['g228', 'g203', 'g217', 'g216', 'g231']


def metrics(y, s, grp, prof):
    y = np.asarray(y, float); s = np.asarray(s, float)
    W, wn = cell_auc(y, s, np.asarray(grp)); Gm, gn = cell_auc(y, s, np.array([f'{g}|{p}' for g, p in zip(grp, prof)]))
    return dict(P=float(auc(y, s)), W=float(W), W_pairs=int(wn), G=float(Gm), G_pairs=int(gn), n=int(len(y)), unsafe_rate=float(y.mean()))


def main():
    runs = sys.argv[1] if len(sys.argv) > 1 else G + '/data/runs'
    model = P.RiskModel(); rule = P.HandRule()
    out, pooled = {}, {k: [] for k in ('y', 'z', 'r', 'v', 'g', 'p')}
    for a in NEW:
        P.set_map(f'assets/traverse/arena_{a}')
        Y, Xs, C, L, grp, prof, spd = [], [], [], [], [], [], []
        for cp in sorted(Path(f'{G}/cases_data_{a}/cases').glob('*_data_group_*.json')):
            case = json.load(open(cp)); g = case['id']; lay = case['layout']
            for ri in range(12):
                d = f'{runs}/{g}__route_{ri:02d}'
                if not (os.path.exists(d + '/outcome.json') and os.path.exists(d + '/trajectory.npz')):
                    continue
                r = json.load(open(cp.parent / 'routes' / g / f'route_{ri:02d}.json'))
                x, l = P.DS.station_tensor(np.asarray(r['waypoints']), np.asarray(r['speeds']), np.asarray(r['stations']))
                Xs.append(x); L.append(l); Y.append(labels(d)['unsafe']); grp.append(g); prof.append(ri % 4)
                C.append(P.geom_ctx(lay['start_xy'], case['goal_xy'], lay['start_yaw'], [l])[0]); spd.append(float(np.asarray(r['speeds'])[1:-1].mean()))
        if not Y:
            continue
        X = np.stack(Xs); z, _ = model.score(X, np.stack(C)); s = rule.score(rule.features(X, np.asarray(L)))
        out[a] = dict(model=metrics(Y, z, grp, prof), hand_rule=metrics(Y, s, grp, prof), speed_only=metrics(Y, -np.asarray(spd), grp, prof))
        for k, v in zip(('y', 'z', 'r', 'v', 'g', 'p'), (Y, z, s, -np.asarray(spd), grp, prof)):
            pooled[k] += list(v)
        print(a, {k: {m: round(v[m], 3) for m in ('P', 'W', 'G')} for k, v in out[a].items()}, 'n', len(Y), flush=True)
    out['new_pooled'] = dict(model=metrics(pooled['y'], pooled['z'], pooled['g'], pooled['p']),
                             hand_rule=metrics(pooled['y'], pooled['r'], pooled['g'], pooled['p']),
                             speed_only=metrics(pooled['y'], pooled['v'], pooled['g'], pooled['p']))
    # f104 reference: held-out designed routes the model never trained on
    d = np.load('artifacts/traverse/fdm_f104_50h_20260909/night2_v1/station_ds_all.npz', allow_pickle=True)
    m = (d['split'].astype(str) == 'test') & (d['source'].astype(str) == 'designed')
    X = d['X'][m].astype(np.float32); ctx = d['ctx'][m][:, 17:22].astype(np.float32); Lr = d['route_len'][m].astype(float)
    y = d['unsafe'][m]; grp = d['group'][m].astype(str); prof = d['profile'][m]
    z, _ = model.score(X, ctx); s = rule.score(rule.features(X, Lr)); v = -X[:, 3, 1:-1, 16].astype(float).mean(1)  # slower = riskier, as for the new arenas
    out['f104_test_split'] = dict(model=metrics(y, z, grp, prof), hand_rule=metrics(y, s, grp, prof), speed_only=metrics(y, v, grp, prof),
                                  note='depth-map corridors as stored in the training tensors')
    json.dump(out, open(G + '/data_ranking.json', 'w'), indent=1)
    for k in ('f104_test_split', 'new_pooled'):
        print(k, {s_: {m_: round(out[k][s_][m_], 3) for m_ in ('P', 'W', 'G', 'n')} for s_ in ('model', 'hand_rule', 'speed_only')})


if __name__ == '__main__':
    main()
