#!/usr/bin/env python3
"""Map-lookup error per arena, the covariate of PLAN section 7 item 6 (arena_gator_20260925, E1b).

What the planner's models see: the elevation channel of the one static OptiX overhead capture, read at world (x, y)
through the flat-ground pixel mapping (f104_n2_dataset.init_map / sample_map: row = ctr - y / mpp, col = ctr + x / mpp,
bilinear, value * elevation_scale). Its error against Chrono's own terrain height is measured on the 4,096 audit
points the capture script stores in native_height_audit.npz (uniform in [-38, 38]^2, rng 104, the same points on
every arena; native_height_m = RigidTerrain.GetHeight). This is the measure of maps/flat_lookup_error.json (E1); it is
recomputed here for all 12 arenas from the files and the 8 E1 values must be reproduced (--check).

Per arena: n valid points, rmse, mean signed error (map - native), p95 and max |error|; the same restricted to the
points within 2 sigma of a hill or crater centre (where the suites' routes go and where the error grows); the
native-height audit (BMP bilinear vs native, the capture's own orientation check); and, as further covariates, the
gen_v1 8-statistic distance to f104 and to the nearest training arena (f104, g203, g228), scales from
ag_arena_rank.py.

  PYTHONPATH=src:scripts python scripts/ag_map_lookup_error.py --out <K3>/arenas/map_lookup_error.json
"""
from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path

import numpy as np

import f104_n2_dataset as DS
from nedm.traverse.terrain import TerrainMap
from ag_arena_rank import stats, distance, SIM

ROOT = Path(__file__).resolve().parents[1]
K3 = ROOT / 'artifacts/traverse/arena_gator_20260925'
ARENAS = {'f104': ('arena_f104_50h_v1', ROOT / 'artifacts/traverse/crm_f104_v1/map_root', 'training'),
          'g203': ('arena_g203', K3 / 'map_roots/g203', 'training'), 'g228': ('arena_g228', K3 / 'map_roots/g228', 'training'),
          'g217': ('arena_g217', K3 / 'map_roots/g217', 'dev')}
ARENAS.update({a: (f'arena_{a}', K3 / f'map_roots/{a}', 'test_near') for a in ('g260', 'g271', 'g251', 'g247')})
ARENAS.update({a: (f'arena_{a}', K3 / f'map_roots/{a}', 'test_spread') for a in ('g258', 'g268', 'g263', 'g241')})
TRAIN = ('f104', 'g203', 'g228')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def summ(e):
    a = np.abs(e)
    return dict(n=int(e.size), rmse=float(np.sqrt(np.mean(e ** 2))), bias=float(np.mean(e)),
                p95=float(np.quantile(a, .95)), max=float(a.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=K3 / 'arenas/map_lookup_error.json')
    ap.add_argument('--check', type=Path, default=K3 / 'maps/flat_lookup_error.json')
    ap.add_argument('--feature-sigmas', type=float, default=2.0)
    a = ap.parse_args()
    scale = json.loads(SIM.read_text())['scale']
    feats = {k: stats(ROOT / 'assets/traverse' / v[0]) for k, v in ARENAS.items()}
    out = dict(schema='ag_map_lookup_error_v1', created=time.strftime('%Y-%m-%d %H:%M:%S'),
               script='scripts/ag_map_lookup_error.py', script_sha256=sha(__file__),
               measure='map (flat-ground lookup of the OptiX capture, f104_n2_dataset.sample_map channel 3 x elevation_scale) '
                       'minus Chrono native height, on the capture\'s 4,096 audit points; valid map points only',
               feature_region=f'points within {a.feature_sigmas} sigma of any hill or crater centre (TerrainMap.features, Chrono frame)',
               distance='gen_v1 8-statistic distance (ag_arena_rank.distance, gen_v1 scales)', arenas={})
    for k, (name, root, role) in ARENAS.items():
        obs = json.loads((root / 'static_map_v1/observation.json').read_text())
        arena = ROOT / 'assets/traverse' / name
        meta = json.loads((arena / 'arena_meta.json').read_text())
        assert obs['arena_bmp_sha256'] == sha(arena / meta['bmp']), f'{k}: map root is not this arena'
        aud = np.load(root / 'static_map_v1/native_height_audit.npz')
        xy, nat, bil = aud['xy'], aud['native_height_m'], aud['bmp_bilinear_height_m']
        DS.init_map(str(root))
        v, valid = DS.sample_map(xy[:, 0], xy[:, 1])
        err = (v[3] * DS.G['elev_scale'] - nat)[valid]
        fts = TerrainMap.from_dir(arena).features   # Chrono frame (meta['features'] is the y-mirrored generation frame)
        c = np.array([[f['x_m'], f['y_m']] for f in fts])
        s = np.array([f['sigma_m'] for f in fts])
        near = (np.linalg.norm(xy[:, None, :] - c[None], axis=-1) <= a.feature_sigmas * s[None]).any(1)
        d = {t: distance(feats[k], feats[t], scale) for t in TRAIN}
        nearest = min((t for t in TRAIN if t != k), key=d.get) if k in TRAIN else min(TRAIN, key=d.get)
        out['arenas'][k] = dict(arena=name, role=role, map_root=str(root.relative_to(ROOT)),
                                map_observation_sha256=obs['observation_sha256'], backend=obs['camera']['backend'],
                                n_audit=int(len(xy)), n_valid=int(valid.sum()), all=summ(err),
                                feature_region=summ((v[3] * DS.G['elev_scale'] - nat)[valid & near]),
                                outside_features=summ((v[3] * DS.G['elev_scale'] - nat)[valid & ~near]),
                                native_vs_bmp_bilinear=summ(bil - nat),
                                height_range_m=float(meta['height_max_m'] - meta['height_min_m']),
                                slope_p99_deg=float(feats[k][5]),
                                distance_to=d, distance_f104=d['f104'],
                                nearest_training_arena=nearest, distance_nearest_training=d[nearest])
    if a.check and a.check.exists():
        ref = json.loads(a.check.read_text())
        diffs = {k: abs(out['arenas'][k]['all']['rmse'] - ref[k]['rmse']) for k in ref}
        out['check_vs_E1_flat_lookup_error'] = dict(file=str(a.check.relative_to(ROOT)), n=len(ref),
                                                    max_abs_rmse_diff=max(diffs.values()),
                                                    n_equal_points=all(out['arenas'][k]['all']['n'] == ref[k]['n'] for k in ref))
        assert max(diffs.values()) < 1e-9, diffs
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    print(f"{'arena':6s} {'role':11s} {'rmse':>6s} {'bias':>7s} {'p95':>6s} {'max':>6s} {'rmse_feat':>9s} {'d_f104':>6s} {'d_near':>6s} nearest")
    for k, r in out['arenas'].items():
        print(f"{k:6s} {r['role']:11s} {r['all']['rmse']:.4f} {r['all']['bias']:+.4f} {r['all']['p95']:.4f} {r['all']['max']:.4f} "
              f"{r['feature_region']['rmse']:9.4f} {r['distance_f104']:6.3f} {r['distance_nearest_training']:6.3f} {r['nearest_training_arena']}")
    if 'check_vs_E1_flat_lookup_error' in out:
        print('check vs E1:', out['check_vs_E1_flat_lookup_error'])
    print('wrote', a.out)


if __name__ == '__main__':
    main()
