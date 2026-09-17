"""sensor_v1 offline read-out: dev / test-split metrics from the sweep, and zero-shot ranking on the routes already
driven on the five sibling arenas (their corridors sampled from each arena's captured RGB-D image)."""
import glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_planner as P
from sensor_train import metrics

S = 'artifacts/traverse/fdm_f104_50h_20260909/sensor_v1'
NEW = ['g228', 'g203', 'g217', 'g216', 'g231']


def main():
    variants = sys.argv[1].split(',') if len(sys.argv) > 1 else ['E', 'E0', 'D', 'RGBD', 'RGB']
    sweep = []
    for f in glob.glob(S + '/sweep/sweep_*.json'):
        sweep += json.load(open(f))
    out = {'sweep': {}}
    for v in variants:
        rows = [r for r in sweep if r['variant'] == v]
        if rows:
            out['sweep'][v] = {f'{fold}_{k}': [float(np.mean([r[fold][k] for r in rows])), float(np.std([r[fold][k] for r in rows]))]
                               for fold in ('dev', 'test') for k in ('G_unsafe', 'W_unsafe', 'P_unsafe', 'G_fail')} | {'seeds': len(rows)}
    data = {a: dict(np.load(f'{S}/sensor_ds_gen_{a}.npz', allow_pickle=True)) for a in NEW}
    models = {v: P.SensorRiskModel(f'{S}/sweep/{v}_s*.pt') for v in variants if glob.glob(f'{S}/sweep/{v}_s*.pt')}
    models['N2 (deployed)'] = None
    n2 = P.RiskModel()
    zs = {}
    for name, mdl in models.items():
        pooled = {k: [] for k in ('s', 'unsafe', 'fail', 'group', 'profile')}
        zs[name] = {}
        for a in NEW:
            d = data[a]; m = d['source'].astype(str) == 'designed'
            X = d['X10'][m].astype(np.float32); ctx = d['ctx'][m][:, 17:22].astype(np.float32)
            z = n2.score(X[:, :5], ctx)[0] if mdl is None else mdl.score(X, ctx)[0]
            dd = {k: d[k][m] for k in ('group', 'profile', 'unsafe', 'fail')}
            zs[name][a] = metrics(z, dd, np.ones(m.sum(), bool))
            for k in ('unsafe', 'fail', 'group', 'profile'): pooled[k] += list(dd[k])
            pooled['s'] += list(z)
        dd = {k: np.asarray(pooled[k]) for k in ('group', 'profile', 'unsafe', 'fail')}
        zs[name]['new_pooled'] = metrics(np.asarray(pooled['s']), dd, np.ones(len(dd['unsafe']), bool))
        print(f"{name:14s} new arenas pooled: P {zs[name]['new_pooled']['P_unsafe']:.3f}  W {zs[name]['new_pooled']['W_unsafe']:.3f}  "
              f"G {zs[name]['new_pooled']['G_unsafe']:.3f}  (fail W {zs[name]['new_pooled']['W_fail']:.3f})  per-arena W "
              + ' '.join(f"{a}:{zs[name][a]['W_unsafe']:.3f}" for a in NEW), flush=True)
    out['zero_shot'] = zs
    json.dump(out, open(S + '/offline_results.json', 'w'), indent=1, default=float)
    print('\nsweep (mean over seeds): dev G / W | test G / W')
    for v, r in out['sweep'].items():
        print(f"  {v:5s} dev G {r['dev_G_unsafe'][0]:.3f}+-{r['dev_G_unsafe'][1]:.3f}  W {r['dev_W_unsafe'][0]:.3f} | "
              f"test G {r['test_G_unsafe'][0]:.3f} W {r['test_W_unsafe'][0]:.3f}  ({r['seeds']} seeds)")


if __name__ == '__main__':
    main()
