#!/usr/bin/env python3
"""Pick the 4 'spread' test arenas from the E1 ranking of seeds 241-280 (arena_gator PLAN section 7 item 1).

Rule (declared in PLAN 7.1 before this script ran): ranks 5-40 of the E1 ranking (arenas/selection.json) are split into
four quarters (ranks 5-13, 14-22, 23-31, 32-40); in each quarter the arena with the smallest BMP sha256 (hex string,
i.e. numerically smallest) is taken. Nothing but the rank and the BMP hash is used. The BMP hashes are re-read from the
generated arena files when those still exist (--gen-root) and must equal the ones stored in selection.json.

Also records the gen_v1 8-statistic distance of each chosen arena to f104, g203 and g228 (the training arenas), using
the same statistics and scales as ag_arena_rank.py.

  PYTHONPATH=src:scripts python scripts/ag_spread_select.py --selection <K3>/arenas/selection.json \
      --gen-root /tmp/ag_e1/gen_a --out <K3>/arenas/selection_spread.json
"""
from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path

import numpy as np

from ag_arena_rank import stats, distance, sha

ROOT = Path(__file__).resolve().parents[1]
QUARTERS = [(5, 13), (14, 22), (23, 31), (32, 40)]
TRAIN = {'f104': ROOT / 'assets/traverse/arena_f104_50h_v1', 'g203': ROOT / 'assets/traverse/arena_g203',
         'g228': ROOT / 'assets/traverse/arena_g228'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--gen-root', type=Path, default=None)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    sel = json.loads(a.selection.read_text())
    ranking = {r['rank']: r for r in sel['ranking']}
    assert sorted(ranking) == list(range(1, 41)), 'expected ranks 1-40'
    scale = sel['scale']
    tr_feat = {k: stats(v) for k, v in TRAIN.items()}
    assert max(abs(x - y) for x, y in zip(tr_feat['f104'], sel['reference_features'])) < 1e-9
    out = {'schema': 'ag_spread_select_v1', 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
           'script': 'scripts/ag_spread_select.py', 'script_sha256': sha(__file__),
           'selection_json': str(a.selection.relative_to(ROOT)) if a.selection.is_relative_to(ROOT) else str(a.selection),
           'selection_json_sha256': sha(a.selection),
           'rule': ('PLAN 7.1: ranks 5-40 of the E1 ranking split into four quarters (5-13, 14-22, 23-31, 32-40); in '
                    'each quarter the arena with the smallest BMP sha256 (hex string compare) is taken; nothing else '
                    'is used'),
           'replacement_rule': ('declared with the choice: a chosen arena that fails a mechanical preparation gate '
                                '(regeneration not byte-identical, map capture assertions, soil surface check, or the '
                                'case generator cannot produce 250 feature groups) is replaced by the arena with the '
                                'next-smallest BMP sha256 in the same quarter; any replacement is recorded here'),
           'replacements': [], 'quarters': [], 'selected': []}
    for lo, hi in QUARTERS:
        cand = []
        for rk in range(lo, hi + 1):
            r = ranking[rk]
            h = r['bmp_sha256']
            if a.gen_root is not None:
                f = a.gen_root / r['arena'] / 'arena_000.bmp'
                h2 = sha(f)
                assert h2 == h, f'{f}: {h2} != selection.json {h}'
            cand.append({'rank': rk, 'arena': r['arena'], 'distance_f104': r['distance'], 'bmp_sha256': h})
        cand_sorted = sorted(cand, key=lambda c: c['bmp_sha256'])
        pick = cand_sorted[0]
        out['quarters'].append({'ranks': [lo, hi], 'candidates': cand, 'hash_order': [c['arena'] for c in cand_sorted],
                                'chosen': pick['arena']})
        out['selected'].append(pick['arena'])
    rows = []
    for q in out['quarters']:
        r = next(x for x in sel['ranking'] if x['arena'] == q['chosen'])
        d = {k: distance(r['features'], f, scale) for k, f in tr_feat.items()}
        near = min(d, key=d.get)
        rows.append({'arena': r['arena'], 'rank': r['rank'], 'quarter': q['ranks'], 'bmp_sha256': r['bmp_sha256'],
                     'meta_sha256': r['meta_sha256'], 'features': r['features'],
                     'distance_to': d, 'nearest_training_arena': near, 'distance_nearest_training': d[near]})
        print(f"quarter {q['ranks']}: {r['arena']} rank {r['rank']} sha {r['bmp_sha256'][:16]} "
              f"d(f104) {d['f104']:.3f} d(g203) {d['g203']:.3f} d(g228) {d['g228']:.3f} nearest {near}", flush=True)
    out['chosen'] = rows
    out['gen_root_hash_check'] = str(a.gen_root) if a.gen_root else None
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    print('selected:', out['selected'])
    print('wrote', a.out)


if __name__ == '__main__':
    main()
