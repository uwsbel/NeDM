"""TASK C step 1: regenerate the sensor_v1 test-2 candidate pools and score every candidate under v1 and v2 corridors.

For each selected start/goal group:
  * rebuild the proposal pool (256) and the fixed-2 m/s pool (256) with the exact seeds the test used
    (md5(group + 's1_proposal') / md5(group + 's1_fixed2'), gen_planner.proposal_pool / fixed2_pool);
  * build the 10-channel corridor tensor twice -- v1 (gen_planner.corridors10, flat-ground image lookup) and
    v2 (corridor_v2.tensor10_v2, back-projected metric grid);
  * score every candidate with the three existing ensembles (N2 height+slopes, E0 height-only, D raw depth)
    on both tensors, and record per-candidate corridor diagnostics.

Nothing is modified outside this directory; checkpoints and captures are read-only.

  python replay_pools.py --arenas f104 g203 g216 g217 g228 g231 --stride 8 --out arena_<a>.npz
"""
import argparse, hashlib, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/harry/NeDM-traverse_mppi')
EXP = ROOT / 'artifacts/traverse/fdm_f104_50h_20260909'
sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(HERE))

ARENA_DIR = {'f104': 'arena_f104_50h_v1', 'g203': 'arena_g203', 'g216': 'arena_g216',
             'g217': 'arena_g217', 'g228': 'arena_g228', 'g231': 'arena_g231'}
CAM_H = 110.0
DIAG = ['mean_abs_grade', 'p95_abs_grade', 'mean_abs_cross', 'p95_abs_cross', 'mean_radius_m', 'max_radius_m',
        'mean_v1_disp_m', 'max_v1_disp_m', 'mean_abs_delev', 'p95_abs_delev', 'valid_frac_v1', 'valid_frac_v2',
        'mean_speed', 'route_len_m']
_S = {}


def _seed(g, tag):
    return int(hashlib.md5((g + tag).encode()).hexdigest()[:8], 16)


def _init(arena):
    import gen_planner as P
    from corridor_v2 import V2Map, tensor10_v2, corridor_xy
    P.set_sensor_map(str(EXP / 'sensor_v1/maps' / ARENA_DIR[arena]))
    _S.update(P=P, vm=V2Map(EXP / 'sensor_v2/grids' / ARENA_DIR[arena]), t10=tensor10_v2, cxy=corridor_xy)


def build(args):
    case_path, arena = args
    P, vm, t10, cxy = _S['P'], _S['vm'], _S['t10'], _S['cxy']
    case = json.load(open(case_path)); g = case['id']; lay = case['layout']
    pose = [lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']]
    base = {k: np.asarray(v, float) for k, v in
            json.load(open(Path(case_path).parent / 'routes' / g / 'route_00.json')).items()
            if k in ('waypoints', 'speeds', 'stations', 'headings')}
    base['meta'] = {}
    c1, tries = P.proposal_pool(base, pose, np.random.default_rng(_seed(g, 's1_proposal')))
    c2 = P.fixed2_pool(base, pose, np.random.default_rng(_seed(g, 's1_fixed2')))
    out = {'group': g, 'arena': arena, 'tries': tries, 'n_proposal': len(c1), 'n_fixed2': len(c2)}
    for name, cands in (('proposal', c1), ('fixed2', c2)):
        X1, L1 = P.corridors10(cands)
        X2 = np.empty_like(X1); D = np.zeros((len(cands), len(DIAG)), np.float32)
        for i, r in enumerate(cands):
            wp = np.asarray(r['waypoints']); sp = np.asarray(r['speeds']); st = np.asarray(r['stations'])
            X2[i] = t10(vm, wp, sp, st)[0]
            gx, gy, grid = cxy(wp, st)
            rad = np.sqrt(gx ** 2 + gy ** 2)
            zz = vm.sample(gx, gy)
            z_true = np.where(zz['valid'], zz['z'], 0.0)
            disp = rad * np.clip(z_true, 0, None) / (CAM_H - np.clip(z_true, None, CAM_H - 1))
            de = np.abs(X1[i, 0] - X2[i, 0])
            D[i] = [np.abs(X1[i, 1]).mean(), np.percentile(np.abs(X1[i, 1]), 95),
                    np.abs(X1[i, 2]).mean(), np.percentile(np.abs(X1[i, 2]), 95),
                    rad.mean(), rad.max(), np.abs(disp).mean(), np.abs(disp).max(),
                    de.mean(), np.percentile(de, 95), X1[i, 4].mean(), X2[i, 4].mean(),
                    float(sp[1:-1].mean()), float(st[-1])]
        out[name] = dict(X1=X1.astype(np.float16), X2=X2.astype(np.float16), L=L1.astype(np.float32), D=D,
                         ctx=P.geom_ctx(lay['start_xy'], case['goal_xy'], lay['start_yaw'], L1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arenas', nargs='+', default=list(ARENA_DIR))
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--tag', default='s8')
    a = ap.parse_args()

    import gen_planner as P
    models = {'n2': P.RiskModel(),
              'e0': P.SensorRiskModel(str(EXP / 'sensor_v1/final/E0_s*.pt')),
              'd': P.SensorRiskModel(str(EXP / 'sensor_v1/final/D_s*.pt'))}
    outdir = HERE / 'scores'; outdir.mkdir(exist_ok=True)

    for arena in a.arenas:
        t0 = time.time()
        cases = sorted(str(p) for p in (EXP / f'sensor_v1/cases2_{arena}/cases').glob('*.json') if p.name != 'cases.json')
        sel = cases[a.offset::a.stride]
        if a.limit:
            sel = sel[:a.limit]
        acc = {p: {k: {v: [] for v in ('v1', 'v2')} for k in models} for p in ('proposal', 'fixed2')}
        diag = {p: [] for p in ('proposal', 'fixed2')}
        groups, tries = [], []
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(arena,)) as ex:
            for j, r in enumerate(ex.map(build, [(c, arena) for c in sel], chunksize=1)):
                groups.append(r['group']); tries.append(r['tries'])
                for pool in ('proposal', 'fixed2'):
                    d = r[pool]
                    assert len(d['X1']) == 256, (r['group'], pool, len(d['X1']))
                    ctx = d['ctx']
                    for k, m in models.items():
                        for ver, X in (('v1', d['X1']), ('v2', d['X2'])):
                            Xf = X.astype(np.float32)
                            z, _ = m.score(Xf[:, :5] if k == 'n2' else Xf, ctx)
                            acc[pool][k][ver].append(z.astype(np.float64))
                    diag[pool].append(d['D'])
                if (j + 1) % 10 == 0:
                    print(f'  {arena}: {j + 1}/{len(sel)}  {time.time() - t0:.0f}s', flush=True)
        out = {'group': np.array(groups), 'tries': np.array(tries), 'diag_names': np.array(DIAG)}
        for pool in ('proposal', 'fixed2'):
            out[f'{pool}_diag'] = np.stack(diag[pool])
            for k in models:
                for ver in ('v1', 'v2'):
                    out[f'{pool}_{k}_{ver}'] = np.stack(acc[pool][k][ver])
        np.savez_compressed(outdir / f'{arena}_{a.tag}.npz', **out)
        print(f'{arena}: {len(groups)} groups in {time.time() - t0:.0f}s -> {outdir / f"{arena}_{a.tag}.npz"}', flush=True)


if __name__ == '__main__':
    main()
