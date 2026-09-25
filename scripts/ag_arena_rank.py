#!/usr/bin/env python3
"""Rank generated arenas by the gen_v1 8-statistic distance to f104 and pick the closest ones (arena_gator E1).

The gen_v1 ranking script was never committed; only its output (fdm_f104_50h_20260909/gen_v1/arena_similarity.json:
reference features, per-statistic scales, ranking of seeds 201-240) survives. This recomputes the same distance:

    statistics (from each arena's arena_meta.json):
      slope_cap_deg, roughness_m, roughness_corr_m, n_hills, n_craters   (family block)
      slope_p99_deg = degrees(atan(slope_stats.p99)), flat5deg_fraction  (slope_stats block)
      height_range_m = height_max_m - height_min_m
    distance = sqrt(mean_i(((stat_i - stat_i(f104)) / scale_i) ** 2))

  --check   arena dirs whose names appear in the stored ranking are compared with the stored distances (the stored
            features are rounded to 3 decimals, so agreement is expected to ~0.003, not exactly)
  --rank    arena dirs to rank; the --select closest are chosen (ties broken by name). Nothing but the distance is used.

  PYTHONPATH=src:scripts python scripts/ag_arena_rank.py --check /tmp/gen_ref/arena_g2* --rank /tmp/gen_a/arena_g2* \
      --select 4 --out <K3>/arenas/selection.json
"""
from __future__ import annotations

import argparse, hashlib, json, math, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / 'artifacts/traverse/fdm_f104_50h_20260909/gen_v1/arena_similarity.json'
REF = ROOT / 'assets/traverse/arena_f104_50h_v1'
NAMES = ['slope_cap_deg', 'roughness_m', 'roughness_corr_m', 'n_hills', 'n_craters', 'slope_p99_deg',
         'flat5deg_fraction', 'height_range_m']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stats(arena_dir):
    m = json.loads((Path(arena_dir) / 'arena_meta.json').read_text())
    f, s = m['family'], m['slope_stats']
    return [float(f['slope_cap_deg']), float(f['roughness_m']), float(f['roughness_corr_m']), float(f['n_hills']),
            float(f['n_craters']), math.degrees(math.atan(float(s['p99']))), float(s['flat5deg_fraction']),
            float(m['height_max_m']) - float(m['height_min_m'])]


def distance(x, ref, scale):
    z = (np.asarray(x, float) - np.asarray(ref, float)) / np.asarray(scale, float)
    return float(np.sqrt(np.mean(z ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--similarity', type=Path, default=SIM)
    ap.add_argument('--ref', type=Path, default=REF)
    ap.add_argument('--check', type=Path, nargs='*', default=[])
    ap.add_argument('--check-tol', type=float, default=0.003)
    ap.add_argument('--rank', type=Path, nargs='*', default=[])
    ap.add_argument('--select', type=int, default=4)
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()

    sim = json.loads(a.similarity.read_text())
    assert sim['feature_names'] == NAMES, sim['feature_names']
    scale = sim['scale']
    ref = stats(a.ref)
    ref_dev = float(np.max(np.abs(np.asarray(ref) - np.asarray(sim['reference_features']))))
    assert ref_dev < 1e-9, f'f104 statistics differ from the stored reference features by {ref_dev}'
    out = {'schema': 'ag_arena_rank_v1', 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
           'script': 'scripts/ag_arena_rank.py', 'script_sha256': sha(__file__),
           'similarity_json': str(a.similarity.relative_to(ROOT)) if a.similarity.is_relative_to(ROOT) else str(a.similarity),
           'similarity_json_sha256': sha(a.similarity), 'reference': a.ref.name,
           'reference_bmp_sha256': sha(a.ref / 'arena_000.bmp'), 'feature_names': NAMES, 'scale': scale,
           'reference_features': ref, 'reference_features_max_abs_dev_vs_stored': ref_dev,
           'distance': 'sqrt(mean(((stat - stat_f104) / scale)^2)) over the 8 statistics'}

    if a.check:
        stored = {r['arena']: r for r in sim['ranking']}
        rows = []
        for d in sorted(a.check):
            if d.name not in stored:
                continue
            x = stats(d); dist = distance(x, ref, scale)
            rows.append({'arena': d.name, 'distance': dist, 'stored_distance': stored[d.name]['distance'],
                         'abs_diff': abs(dist - stored[d.name]['distance']),
                         'max_abs_feature_diff_vs_stored': float(np.max(np.abs(np.asarray(x) - np.asarray(stored[d.name]['features'])))),
                         'bmp_sha256': sha(d / 'arena_000.bmp')})
        diffs = np.array([r['abs_diff'] for r in rows])
        order_new = [r['arena'] for r in sorted(rows, key=lambda r: (r['distance'], r['arena']))]
        order_old = [r['arena'] for r in sim['ranking'] if r['arena'] in {x['arena'] for x in rows}]
        chk = {'n': len(rows), 'n_stored': len(stored), 'max_abs_diff': float(diffs.max()), 'mean_abs_diff': float(diffs.mean()),
               'tolerance': a.check_tol, 'passed': bool(len(rows) == len(stored) and diffs.max() <= a.check_tol),
               'same_order_as_stored': order_new == order_old,
               'top5_recomputed': order_new[:5], 'top5_stored': order_old[:5], 'rows': rows}
        out['check_vs_stored'] = chk
        print(f"check: {chk['n']}/{chk['n_stored']} stored arenas recomputed, max |diff| {chk['max_abs_diff']:.4f} "
              f"(mean {chk['mean_abs_diff']:.4f}, tol {a.check_tol}), same order {chk['same_order_as_stored']}, "
              f"passed {chk['passed']}", flush=True)
        if not chk['passed']:
            raise SystemExit('the recomputed distances do not reproduce the stored gen_v1 ranking')

    if a.rank:
        rows = []
        for d in sorted(a.rank):
            x = stats(d)
            rows.append({'arena': d.name, 'distance': distance(x, ref, scale), 'features': x,
                         'bmp_sha256': sha(d / 'arena_000.bmp'), 'meta_sha256': sha(d / 'arena_meta.json')})
        rows.sort(key=lambda r: (r['distance'], r['arena']))
        for i, r in enumerate(rows):
            r['rank'] = i + 1
        out['ranking'] = rows
        out['selected'] = [r['arena'] for r in rows[:a.select]]
        out['selection_rule'] = (f'the {a.select} arenas with the smallest distance, taken in ranking order; nothing '
                                 'else was looked at before the choice')
        out['replacement_rule'] = ('declared with the choice: a selected arena that fails a mechanical preparation gate '
                                   '(map capture assertions, soil surface check, or the case generator cannot produce '
                                   'its suite) is replaced by the next unselected arena in this ranking; any '
                                   'replacement is recorded here')
        out['replacements'] = []
        for r in rows:
            print(f"{r['rank']:3d} {r['arena']}  {r['distance']:.4f}", flush=True)
        print('selected:', out['selected'], flush=True)

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1) + '\n')
        print('wrote', a.out)


if __name__ == '__main__':
    main()
