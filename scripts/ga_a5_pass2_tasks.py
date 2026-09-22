#!/usr/bin/env python3
"""A5 pass-2 task rows (PLAN v2 A5): approach prefix (60 frames of the pass-1 approach route) then a branch to each
arm's pick planned from the recorded frame-60 state. One row per (group, arm) with unique ids <group>__<ARM>_B; picks
identical across arms (planner content sha256) are driven once (run=false + ref_id).

  python scripts/ga_a5_pass2_tasks.py --world crm --arms Spcrm,Sprigid,H,Hmask,P,T --picks-dir K/A_adapt/a5/picks \
      --cluster-case-prefix generalist/suite/cases --cluster-approach-prefix generalist/a5/approach \
      --cluster-route-prefix generalist/a5/pass2/routes_crm --out K/A_adapt/a5/tasks_pass2_crm.json
<picks-dir>/picks_<world>_<ARM>/ are ga_planner.py out dirs (routes/<g>__B.json, tasks.json rows with sha256).
"""
import argparse, hashlib, json, os, shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--world', required=True); ap.add_argument('--arms', required=True); ap.add_argument('--picks-dir', required=True)
    ap.add_argument('--cluster-case-prefix', required=True); ap.add_argument('--cluster-approach-prefix', required=True)
    ap.add_argument('--cluster-route-prefix', required=True); ap.add_argument('--branch-frame', type=int, default=60)
    ap.add_argument('--groups', help='file with the analysis-set group ids (one per line); default all')
    ap.add_argument('--rigid-abs-root', help='rigid: absolute cluster root prefixed to case/approach/route paths (gen_runner_g rows)')
    ap.add_argument('--shards', type=int, default=6); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    arms = [x for x in a.arms.split(',') if x]
    keep = set(l.strip() for l in open(a.groups) if l.strip()) if a.groups else None
    rdir = os.path.join(os.path.dirname(a.out), f'routes_pass2_{a.world}'); os.makedirs(rdir, exist_ok=True)
    sha_ref, rows, counts = {}, [], {}
    for arm in arms:
        pdir = os.path.join(a.picks_dir, f'picks_{a.world}_{arm}'); trows = json.load(open(os.path.join(pdir, 'tasks.json')))
        n_new = n_dup = 0
        for r in sorted(trows, key=lambda x: x['tier']):
            g = r['group']
            if keep is not None and g not in keep:
                continue
            nid = f'{g}__{arm}_B'; src = os.path.join(pdir, 'routes', os.path.basename(r['route'])); dst = os.path.join(rdir, nid + '.json')
            shutil.copyfile(src, dst)
            pre = (a.rigid_abs_root.rstrip('/') + '/') if a.rigid_abs_root else ''
            row = dict(id=nid, group=g, arm=arm, run=True, tier=r['tier'], sha256=r['sha256'], source='A5',
                       episode_seed=int(hashlib.md5(nid.encode()).hexdigest()[:8], 16),
                       case=f'{pre}{a.cluster_case_prefix}/{g}.json', route=f'{pre}{a.cluster_approach_prefix}/{g}.json',
                       extra=['--mode', 'branch', '--branch-frame', str(a.branch_frame), '--branch-route', f'{pre}{a.cluster_route_prefix}/{nid}.json', '--horizon-s', '120'])
            if a.world == 'rigid':
                row.update(arena='f104', shard=int(hashlib.md5(g.encode()).hexdigest(), 16) % a.shards, mode='branch')
            key = (g, r['sha256'])
            if key in sha_ref:
                row['run'] = False; row['ref_id'] = sha_ref[key]; n_dup += 1
            else:
                sha_ref[key] = nid; n_new += 1
            rows.append(row)
        counts[arm] = dict(rows=n_new + n_dup, new_drives=n_new, duplicates=n_dup)
    json.dump(rows, open(a.out, 'w'), indent=1)
    summary = dict(world=a.world, arms=arms, counts=counts, n_rows=len(rows), n_run=sum(r['run'] for r in rows), routes_dir=rdir)
    json.dump(summary, open(a.out.replace('.json', '_summary.json'), 'w'), indent=1); print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
