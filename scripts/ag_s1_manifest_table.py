#!/usr/bin/env python3
"""arena_gator_20260925 soil stage 1: one line per ag_subset.py manifest (rows, fitted rows in deploy / holdout mode,
per arena: selected / available training groups, training rows and standing-start rows, holdout-fitted groups, val and
test rows, dev-fold groups kept, output sha256). Plain python3 (json only).

  python3 ag_s1_manifest_table.py <manifest.json> ... [--json out.json]
"""
import json, os, sys


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    out = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
    if out in args:
        args.remove(out)
    res = {}
    for p in args:
        m = json.load(open(p))
        name = os.path.basename(p).replace('.manifest.json', '')
        pa = []
        for a, v in m['per_arena'].items():
            pa.append('%s: groups %s/%s, train rows %d (standing start %d), holdout-fit groups %d, val %dg/%dr, test %dr, dev kept %d'
                      % (a, v['selected'], v['available'], v['train_rows'], v['train_rows_startup'], v['fit_groups_holdout'],
                         v['val_groups'], v['val_rows'], v['test_rows'], v['dev_fold_groups_kept']))
        print('%-32s rows %6d  fit deploy %6d  fit holdout %6d  sha %s | %s' % (
            name, m['rows'], m['fit_rows_deploy'], m['fit_rows_holdout'], m['output']['sha256'][:16], ' ; '.join(pa)))
        res[name] = dict(rows=m['rows'], fit_rows_deploy=m['fit_rows_deploy'], fit_rows_holdout=m['fit_rows_holdout'],
                         eval_rows_val=m['eval_rows_val'], sha256=m['output']['sha256'], tiers=m.get('tiers'), eval_rows=m.get('eval_rows'),
                         ids_file=m.get('ids_file'), per_arena=m['per_arena'])
    if out:
        json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
