#!/usr/bin/env python3
"""Merge the A3 picks of several deploy ensembles (ga_planner.py out dirs) into ONE cluster task file per world.

Rows get unique ids <group>__<MODEL>_<arm>; every route file is copied into one directory <a3>/routes_<world>/<id>.json;
identical routes (planner content sha256) are driven once: later duplicates get run=false and ref_id = the row that
drives them (or an A0 suite row already driven, given with --a0-tasks and --driven-runs). Rigid merges can append the
A0 suite rows so reference and test arms of a group share one array task (plan review finding 2.3).

  python scripts/ga_merge_tasks.py --world crm --models T,H,P --a3-dir K/A_adapt/a3 --suite-dir K/A_adapt/suite \
      --a0-tasks K/A_adapt/suite/tasks_crm.json --driven-runs K/A_adapt/suite_out_crm/runs,<night-2 runs> \
      --cluster-case-prefix generalist/suite/cases --cluster-route-prefix generalist/a3/routes_crm --out K/A_adapt/a3/tasks_crm_a3.json
  python scripts/ga_merge_tasks.py --world rigid --models T,H,P ... --append-a0 --cluster-case-prefix generalist_suite/cases \
      --cluster-route-prefix generalist_suite/routes_a3 --out K/A_adapt/a3/tasks_rigid_all.json
"""
import argparse, hashlib, json, os, shutil


def md5shard(group, shards):
    return int(hashlib.md5(group.encode()).hexdigest(), 16) % shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', required=True, choices=['crm', 'rigid']); ap.add_argument('--models', required=True)
    ap.add_argument('--a3-dir', required=True); ap.add_argument('--suite-dir', required=True)
    ap.add_argument('--a0-tasks', help='suite tasks file of this world (for dedup against already-driven routes / append)')
    ap.add_argument('--driven-runs', default='', help='comma list of run dirs; an A0 row counts as driven if <dir>/<id>/episode_complete.json exists')
    ap.add_argument('--append-a0', action='store_true', help='rigid: append the A0 rows (not yet driven) to the merged file')
    ap.add_argument('--cluster-case-prefix', required=True); ap.add_argument('--cluster-route-prefix', required=True)
    ap.add_argument('--shards', type=int, default=6); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(',') if m.strip()]
    routes_dir = os.path.join(a.a3_dir, f'routes_{a.world}'); os.makedirs(routes_dir, exist_ok=True)
    driven_dirs = [d for d in a.driven_runs.split(',') if d]
    sha_to_ref = {}   # content sha256 -> id of the row that drives (or drove) it
    a0_rows = json.load(open(a.a0_tasks)) if a.a0_tasks else []
    n_a0_driven = 0
    for r in a0_rows:
        driven = any(os.path.exists(os.path.join(d, r['id'], 'episode_complete.json')) for d in driven_dirs)
        if driven or a.append_a0:
            sha_to_ref.setdefault(r['sha256'], r['id']); n_a0_driven += int(driven)
    out, counts = [], {}
    if a.append_a0:
        for r in a0_rows:
            rr = dict(r); rr['source'] = 'A0'
            if a.world == 'rigid':
                assert rr.get('shard') == md5shard(rr['group'], a.shards), rr['id']
            out.append(rr)
        counts['A0'] = len(a0_rows)
    for m in models:
        pdir = os.path.join(a.a3_dir, f'picks_{a.world}_{m}')
        rows = json.load(open(os.path.join(pdir, 'tasks.json')))
        n_new = n_dup = 0
        for r in sorted(rows, key=lambda x: x['tier']):
            arm = r['arms'][0]; nid = f"{r['group']}__{m}_{arm}"
            src = os.path.join(pdir, 'routes', os.path.basename(r['route'])); dst = os.path.join(routes_dir, nid + '.json')
            shutil.copyfile(src, dst)
            row = dict(id=nid, group=r['group'], model=m, arm=arm, run=True, tier=r['tier'], episode_seed=int(hashlib.md5(nid.encode()).hexdigest()[:8], 16),
                       sha256=r['sha256'], file_sha256=hashlib.sha256(open(dst, 'rb').read()).hexdigest(), source='A3',
                       case=f"{a.cluster_case_prefix}/{r['group']}.json", route=f"{a.cluster_route_prefix}/{nid}.json")
            if a.world == 'rigid':
                row.update(arena='f104', shard=md5shard(r['group'], a.shards))
            if r['sha256'] in sha_to_ref:
                row['run'] = False; row['ref_id'] = sha_to_ref[r['sha256']]; n_dup += 1
            else:
                sha_to_ref[r['sha256']] = nid; n_new += 1
            out.append(row)
        counts[m] = dict(rows=len(rows), new_drives=n_new, duplicates=n_dup)
    json.dump(out, open(a.out, 'w'), indent=1)
    lock = hashlib.sha256()
    for p in sorted(os.listdir(routes_dir)):
        lock.update(p.encode()); lock.update(hashlib.sha256(open(os.path.join(routes_dir, p), 'rb').read()).digest())
    lock.update(open(a.out, 'rb').read())
    open(a.out.replace('.json', '_LOCKED.sha256'), 'w').write(lock.hexdigest() + '  routes_%s/*.json + tasks (name + content, sorted)\n' % a.world)
    summary = dict(world=a.world, models=models, counts=counts, n_rows=len(out), n_run=sum(1 for r in out if r['run']),
                   a0_rows=len(a0_rows), a0_driven=n_a0_driven, routes_dir=routes_dir, out=a.out, lock=lock.hexdigest())
    json.dump(summary, open(a.out.replace('.json', '_summary.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
