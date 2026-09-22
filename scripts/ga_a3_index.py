#!/usr/bin/env python3
"""Rename ga_planner picks (route ids <g>__B) into model-tagged copies (<g>__<MODEL>_B) and build the run index
{route_id: {driven_as, group, sha256, run_rel}} from the merged task file, so scripts/ga_analyze.py can pair every
arm's pick with its driven run (or the run of an identical route).
  python scripts/ga_a3_index.py --world rigid --models T,H,P --a3-dir K/A_adapt/a3 --tasks K/A_adapt/a3/tasks_rigid_all.json \
      --base-index K/A_adapt/suite/run_index_rigid.json --out K/A_adapt/a3/run_index_rigid_all.json
"""
import argparse, json, os, shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', required=True); ap.add_argument('--models', required=True); ap.add_argument('--a3-dir', required=True)
    ap.add_argument('--tasks', required=True); ap.add_argument('--base-index'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = {r['id']: r for r in json.load(open(a.tasks))}
    index = json.load(open(a.base_index)) if a.base_index else {}
    for m in a.models.split(','):
        src = os.path.join(a.a3_dir, f'picks_{a.world}_{m}'); dst = os.path.join(a.a3_dir, f'picks_{a.world}_{m}_named')
        os.makedirs(os.path.join(dst, 'picks'), exist_ok=True); os.makedirs(os.path.join(dst, 'routes'), exist_ok=True)
        n = 0
        for f in sorted(os.listdir(os.path.join(src, 'picks'))):
            p = json.load(open(os.path.join(src, 'picks', f)))
            for arm, e in (p.get('arms') or {}).items():
                if not e: continue
                old = e['route_id']; new = f"{p['group']}__{m}_{arm}"; e['route_id'] = new
                r = rows.get(new)
                if r is None: raise SystemExit(f'no task row for {new}')
                driven = r['id'] if r.get('run', True) else r['ref_id']
                index[new] = dict(driven_as=driven, group=p['group'], sha256=r['sha256'], run_rel=f'runs/{driven}', ref_run_local=None, ref_run_cluster=None)
                shutil.copyfile(os.path.join(src, 'routes', old + '.json'), os.path.join(dst, 'routes', new + '.json')); n += 1
            json.dump(p, open(os.path.join(dst, 'picks', f), 'w'), indent=1)
        for extra in ('summary.json', 'PICKS_LOCKED.sha256'):
            if os.path.exists(os.path.join(src, extra)): shutil.copyfile(os.path.join(src, extra), os.path.join(dst, extra))
        print(m, n, 'picks renamed ->', dst)
    json.dump(index, open(a.out, 'w'), indent=1); print(len(index), 'index entries ->', a.out)


if __name__ == '__main__':
    main()
