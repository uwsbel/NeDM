#!/usr/bin/env python3
"""Mechanical checks of the E1b spread-arena preparation (arena_gator_20260925): arenas, maps, grids, soil smokes,
suites. Writes <K3>/arenas/e1b_checks.json. Reads only files; runs no simulation.

  PYTHONPATH=src:scripts python scripts/ag_e1b_checks.py [--gen-a /tmp/ag_e1/gen_a --gen-b /tmp/ag_e1b/gen_b
                                                          --cases-re /tmp/ag_e1b/cases_re]
"""
import argparse, fnmatch, hashlib, json, math, os, time
from collections import Counter
from pathlib import Path

import numpy as np

import f104_n2_dataset as DS
import ag_blacklist as B
from nedm.traverse.terrain import TerrainMap

ROOT = Path(__file__).resolve().parents[1]
K3 = ROOT / 'artifacts/traverse/arena_gator_20260925'
SPREAD = ['g258', 'g268', 'g263', 'g241']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen-a', type=Path, default=Path('/tmp/ag_e1/gen_a'))
    ap.add_argument('--gen-b', type=Path, default=Path('/tmp/ag_e1b/gen_b'))
    ap.add_argument('--cases-re', type=Path, default=Path('/tmp/ag_e1b/cases_re'))
    ap.add_argument('--out', type=Path, default=K3 / 'arenas/e1b_checks.json')
    a = ap.parse_args()
    sel = {r['arena']: r for r in json.loads((K3 / 'arenas/selection.json').read_text())['ranking']}
    out = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), script='scripts/ag_e1b_checks.py', script_sha256=sha(__file__),
               arenas={}, maps={}, grids={}, smoke={}, suites={})
    for s in SPREAD:
        name = f'arena_{s}'; d = ROOT / 'assets/traverse' / name
        meta = json.loads((d / 'arena_meta.json').read_text())
        files = {}
        for f in ('arena_000.bmp', 'arena_meta.json'):
            h = dict(assets=sha(d / f), first_generation=sha(a.gen_a / name / f) if (a.gen_a / name / f).exists() else None,
                     regenerated=sha(a.gen_b / name / f) if (a.gen_b / name / f).exists() else None,
                     selection_json=sel[name]['bmp_sha256' if f.endswith('bmp') else 'meta_sha256'])
            h['identical'] = len({v for v in h.values()}) == 1
            files[f] = h
        out['arenas'][s] = dict(rank=sel[name]['rank'], distance_f104=sel[name]['distance'], files=files,
                                listed_in_gen_arenas_json=json.loads((ROOT / 'scripts/gen_arenas.json').read_text()).get(name) == files['arena_000.bmp']['assets'],
                                in_assets=sorted(os.listdir(d)), seed=meta['seed'], difficulty=meta['family']['difficulty'],
                                size_m=meta['size_m'], slope_cap_deg=meta['family']['slope_cap_deg'],
                                roughness_m=meta['family']['roughness_m'], roughness_corr_m=meta['family']['roughness_corr_m'],
                                n_hills=meta['family']['n_hills'], n_craters=meta['family']['n_craters'],
                                height_range=[meta['height_min_m'], meta['height_max_m']],
                                slope_max_deg=math.degrees(math.atan(meta['slope_stats']['max'])),
                                slope_p99_deg=math.degrees(math.atan(meta['slope_stats']['p99'])),
                                flat5deg_fraction=meta['slope_stats']['flat5deg_fraction'])
        # map
        m = K3 / f'maps/{name}'; obs = json.loads((m / 'observation.json').read_text())
        npz = np.load(m / 'observation.npz'); rgbd = npz['rgbd']
        ref_cam = json.loads((ROOT / 'artifacts/traverse/crm_f104_v1/maps/arena_f104_50h_v1/observation.json').read_text())
        root = K3 / f'map_roots/{s}'
        DS.init_map(str(root))
        out['maps'][s] = dict(bmp_sha_matches_assets=obs['arena_bmp_sha256'] == files['arena_000.bmp']['assets'],
                              meta_sha_matches_assets=obs['arena_meta_sha256'] == files['arena_meta.json']['assets'],
                              observation_sha_ok=obs['observation_sha256'] == sha(m / 'observation.npz'),
                              native_p95_m=obs['native_geometry']['p95_abs_error_m'], native_rmse_m=obs['native_geometry']['rmse_m'],
                              all_corners_visible=obs['all_corners_visible'], backend=obs['camera']['backend'],
                              camera_equal_to_f104_soil_map=obs['camera'] == ref_cam['camera'],
                              capture_script_sha256_equal_to_f104=obs['capture_script_sha256'] == ref_cam.get('capture_script_sha256'),
                              chrono_git=obs['chrono_git'], rgbd_shape=list(rgbd.shape), rgbd_finite=bool(np.isfinite(rgbd).all()),
                              valid_fraction=float((rgbd[3] > -1.999).mean()),
                              map_root_link=os.readlink(root / 'static_map_v1'),
                              map_root_resolves_to_own_capture=(root / 'static_map_v1').resolve() == m.resolve(),
                              init_map_mpp=DS.G['mpp'], init_map_npx=DS.G['npx'])
        # grid
        g = np.load(K3 / f'grids/{name}/grid.npz'); gj = json.loads((K3 / f'grids/{name}/grid.json').read_text())
        n = g['z'].shape[0]; mpp = gj['mpp']; c = -gj['half_extent_m'] + (np.arange(n) + .5) * mpp
        X, Y = np.meshgrid(c, c)            # grid[row, col]: row = +y bin, col = +x bin
        tm = TerrainMap.from_dir(d); ok = g['cover'] > 0
        cv = lambda x: ((x + 40.) * 511. / 80. + .5) * 80. / 512. - 40.   # Chrono maps 512 px onto 511 intervals
        e5 = (g['z'] - tm.height(cv(X), cv(Y)))[ok]
        e = (g['z'] - tm.height(X, Y))[ok]; ef = (g['z'] - tm.height(X, -Y))[ok]
        out['grids'][s] = dict(coverage=float(ok.mean()), rmse_vs_bmp_chrono511_m=float(np.sqrt(np.mean(e5 ** 2))),
                               rmse_vs_bmp_m=float(np.sqrt(np.mean(e ** 2))),
                               rmse_vs_bmp_y_flipped_m=float(np.sqrt(np.mean(ef ** 2))),
                               camera_equal_to_capture=gj['camera'] == obs['camera'])
        # soil smokes
        sm = json.loads((K3 / f'smoke/{s}/smoke_report.json').read_text())
        log = (K3 / f'smoke/{s}/smoke.log').read_text().splitlines()
        surf = [ln for ln in log if ln.startswith('SURFACE')]
        pr = json.loads((K3 / f'smoke/prod_spacing_0.08/{s}/smoke_report.json').read_text())
        out['smoke'][s] = dict(coarse=dict(spacing=sm['spacing'], surface_lines=surf, n_sph=sm['n_sph'], rtf=sm['rtf']),
                               production=dict(spacing=pr['spacing'], depth=pr['depth'], step=pr['step'], n_sph=pr['n_sph'],
                                               n_bce_boundary=pr['n_bce_boundary'], rtf=pr['rtf'],
                                               chassis_above_bmp_m=[round(r['z'] - r['ground'], 3) for r in pr['log']],
                                               speed_mps=[round(r['speed'], 2) for r in pr['log']],
                                               distance_3s_m=float(math.hypot(pr['log'][-1]['x'] + 30, pr['log'][-1]['y'] + 30))))
        # suite
        cdir = K3 / f'cases/test_{s}/cases'; man = json.loads((cdir / 'cases.json').read_text()); recs = man['records']
        ids = [r['scene_id'] for r in recs]
        feat = Counter(json.loads((cdir / 'routes' / g_ / 'route_00.json').read_text())['meta'].get('feature_index') for g_ in ids)
        kinds = [f['kind'] for f in tm.features]
        lens = [json.loads((cdir / 'routes' / g_ / 'route_00.json').read_text())['stations'][-1] for g_ in ids]
        hits = {k: sum(any(fnmatch.fnmatch(x, p) for p in v) for x in ids) for k, v in B.builder_lists().items()}
        regen = a.cases_re / f'test_{s}/cases'
        same = None
        if regen.exists():
            fa = sorted(p.relative_to(cdir) for p in cdir.rglob('*.json')); fb = sorted(p.relative_to(regen) for p in regen.rglob('*.json'))
            same = fa == fb and all((cdir / p).read_bytes() == (regen / p).read_bytes() for p in fa)
        out['suites'][s] = dict(groups=len(recs), seed=man['seed'], prefix=man['prefix'], strata_mode=man['strata'],
                                strata=dict(Counter(r['evaluation_stratum'] for r in recs)),
                                splits_written=dict(Counter(r['split'] for r in recs)),
                                arena_field=sorted({json.loads((cdir / r['case']).read_text())['arena'] for r in recs}),
                                n_features=len(kinds), feature_kinds=dict(Counter(kinds)),
                                groups_per_feature={f'{i}:{kinds[i]}': feat.get(i, 0) for i in range(len(kinds))},
                                features_without_groups=[f'{i}:{kinds[i]}' for i in range(len(kinds)) if feat.get(i, 0) == 0],
                                route00_length_m_p5_p50_p95=[round(float(np.percentile(lens, q)), 1) for q in (5, 50, 95)],
                                blacklist_hits_existing_builders=hits,
                                blacklist_hits_ag_blacklist_ALL=sum(B.is_suite(x) for x in ids),
                                regenerated_byte_identical=same, lock=(K3 / f'suites/test_{s}.SUITE_LOCKED.sha256').read_text().split()[0])
    # ids unique across every case set in K3
    allids = []
    for cj in sorted((K3 / 'cases').glob('*/cases/cases.json')) + sorted((K3 / 'cases').glob('*/cases.json')):
        allids += [r['scene_id'] for r in json.loads(cj.read_text())['records']]
    out['ids_unique_across_K3_case_sets'] = dict(n=len(allids), unique=len(set(allids)) == len(allids))
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    for s in SPREAD:
        A, M, G, S, C = out['arenas'][s], out['maps'][s], out['grids'][s], out['smoke'][s], out['suites'][s]
        print(s, 'files identical', all(v['identical'] for v in A['files'].values()), '| map', M['bmp_sha_matches_assets'],
              M['observation_sha_ok'], f"p95 {M['native_p95_m']:.4f} valid {M['valid_fraction']:.3f} cam==f104 {M['camera_equal_to_f104_soil_map']}",
              f"| grid cov {G['coverage']:.3f} rmse511 {G['rmse_vs_bmp_chrono511_m']:.4f} plain {G['rmse_vs_bmp_m']:.4f} flip {G['rmse_vs_bmp_y_flipped_m']:.3f}",
              f"| prod above {min(S['production']['chassis_above_bmp_m']):.2f}-{max(S['production']['chassis_above_bmp_m']):.2f} v3s {S['production']['speed_mps'][-1]} d {S['production']['distance_3s_m']:.1f}",
              f"| suite {C['groups']} {C['strata']} empty feats {C['features_without_groups']} regen {C['regenerated_byte_identical']} hits {C['blacklist_hits_existing_builders']} / all {C['blacklist_hits_ag_blacklist_ALL']}")
        print('   groups per feature', C['groups_per_feature'])
    print(out['ids_unique_across_K3_case_sets'])


if __name__ == '__main__':
    main()
