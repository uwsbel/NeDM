#!/usr/bin/env python3
"""arena_gator_20260925 module E5a: the Gator ids behind task B's G and H (PLAN 3, 7.7: H is trained on exactly the ids
the Gator validated).

Validated = the episodes in the Gator ci file written by scripts/ag_build_ds.py (completion marker, launch check passed,
finite state / action / pose arrays, map and vehicle-block checks). For every Gator task row of the task file this
tool reports whether it validated and, if not, why (the build record's rejected ids, or no run / no completion
marker), plus the rigid native-height check of the validated runs. Writes the validated ids (one per line, with the
gator__ prefix; ag_subset.py strips it for the HMMWV twins) and a json summary.

  python scripts/ag_e5a_ids.py --ci $G3/e4/f104_gator/ci_f104_gator_rigid.npz --record $G3/e4/f104_gator/f104_gator_rigid_record.json \
     --tasks $G3/tasks/rigid_v2.json --runs $G3/rigid_v1/runs --out-ids $G3/e5/ids/gator_rigid_validated.txt --out-json <summary.json>
  [--compare A.npz B.npz ...]  id sets and row order of ci / subset files (e.g. H against M1)
numpy only.
"""
import argparse, json, os, sys
from collections import Counter
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ci'); ap.add_argument('--record'); ap.add_argument('--tasks'); ap.add_argument('--runs')
    ap.add_argument('--prefix', default='gator__'); ap.add_argument('--world', default='rigid')
    ap.add_argument('--out-ids'); ap.add_argument('--out-json')
    ap.add_argument('--compare', nargs='*', default=[])
    a = ap.parse_args(argv)
    res = {}
    if a.ci:
        z = np.load(a.ci, allow_pickle=True)
        ep = z['episode'].astype(str); af = z['anchor_frame'].astype(int); sp = z['split'].astype(str)
        val = sorted(set(ep))
        rows = [r for r in json.load(open(a.tasks)) if r['id'].startswith(a.prefix) and r.get('tier', -1) >= 0]
        tid = {r['id']: r for r in rows}
        rec = json.load(open(a.record)) if a.record else {}
        sel = rec.get('selection', {}).get(a.world, {})
        rej = dict(sel.get('rejected_ids', []))
        missing = sorted(set(tid) - set(val))
        extra = sorted(set(val) - set(tid))
        why = Counter()
        why_ids = {}
        for i in missing:
            d = os.path.join(a.runs, i) if a.runs else None
            if i in rej:
                w = rej[i]
            elif d and not os.path.isdir(d):
                w = 'no_run_folder'
            elif d and not os.path.isfile(os.path.join(d, 'episode_complete.json')):
                w = 'no_completion_marker'
            else:
                w = 'not_selected_other'
            why[w] += 1; why_ids[i] = w
        nh = Counter()
        if a.runs:
            for i in val:
                p = os.path.join(a.runs, i, 'native_height_check.json')
                try:
                    nh['passed' if json.load(open(p)).get('passed') else 'failed'] += 1
                except (OSError, ValueError):
                    nh['absent'] += 1
        st = Counter()
        for i in val:
            try:
                st[json.load(open(os.path.join(a.runs, i, 'outcome.json'))).get('status')] += 1
            except (OSError, ValueError):
                st['unreadable'] += 1
        res.update(ci=os.path.abspath(a.ci), tasks=os.path.abspath(a.tasks), task_rows=len(tid), validated=len(val),
                   validated_fraction=round(len(val) / max(len(tid), 1), 5), not_validated=len(missing), not_validated_reasons=dict(why),
                   not_validated_ids=why_ids, validated_not_in_tasks=extra[:20], n_validated_not_in_tasks=len(extra),
                   validated_by_split=dict(Counter(tid[i]['split'] for i in val if i in tid)),
                   validated_by_kind=dict(Counter(tid[i]['kind'] for i in val if i in tid)),
                   validated_groups=len({tid[i]['group'] for i in val if i in tid}), native_height_check=dict(nh), status=dict(st),
                   rows=int(len(ep)), rows_by_split={s: int((sp == s).sum()) for s in ('train', 'val', 'test')},
                   startup_rows=int((af == 0).sum()))
        assert not extra, f'validated ids without a task row: {extra[:5]}'
        if a.out_ids:
            os.makedirs(os.path.dirname(os.path.abspath(a.out_ids)), exist_ok=True)
            with open(a.out_ids, 'w') as f:
                f.write('\n'.join(val) + '\n')
            res['out_ids'] = os.path.abspath(a.out_ids)
        print(json.dumps({k: v for k, v in res.items() if k != 'not_validated_ids'}, indent=1), flush=True)
    if a.compare:
        cmp = []
        base = None
        for p in a.compare:
            z = np.load(p, allow_pickle=True)
            ids = z['id'].astype(str)
            d = dict(path=os.path.abspath(p), rows=int(len(ids)), rows_by_split={s: int((z['split'].astype(str) == s).sum()) for s in ('train', 'val', 'test')})
            if base is None:
                base = ids
            else:
                d.update(same_id_set=bool(set(ids) == set(base)), same_order=bool(len(ids) == len(base) and (ids == base).all()),
                         only_here=int(len(set(ids) - set(base))), only_in_first=int(len(set(base) - set(ids))))
            cmp.append(d)
        res['compare'] = cmp
        print(json.dumps(cmp, indent=1), flush=True)
    if a.out_json:
        json.dump(res, open(a.out_json, 'w'), indent=1)


if __name__ == '__main__':
    main()
