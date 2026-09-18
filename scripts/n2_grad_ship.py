"""Lock the gradient-refinement picks (planner_grad_arms.py output) and write the cluster task list + flat picks for analysis.

  python scripts/n2_grad_ship.py --grad artifacts/traverse/crm_night2_v1/planner/grad_crm --cluster-prefix night2/grad_crm/routes
Outputs (in --grad): tasks_cluster.json (run=true tasks only, route paths relative to the cluster CRM root, sha256 per route),
PICKS_LOCKED.sha256 (sorted name+content of routes/*.json), picks_flat/<group>.json (arms A/G/H -> route_id, P, z_mean, T) for
scripts/n2_planner_analyze.py; A = the locked night-1 pick (route_id <group>__crm, driven in eval_v1/runs).
"""
import argparse, hashlib, json
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument('--grad', required=True); ap.add_argument('--cluster-prefix', default='night2/grad_crm/routes')
ap.add_argument('--expect', type=int, default=200)
ap.add_argument('--ref-arm', default='crm', help="route-id suffix of the already-driven one-shot pick used as arm A (crm = night-1 eval_v1 drive, A = the A/A re-drive in eval_iter_crm)")
ap.add_argument('--no-lock', action='store_true', help='do not (re)write PICKS_LOCKED.sha256')
a = ap.parse_args(); out = Path(a.grad)
picks = sorted((out / 'picks').glob('*.json'))
assert len(picks) == a.expect, f'{len(picks)} picks, expected {a.expect}'
tasks = json.load(open(out / 'tasks.json'))
flat_dir = out / 'picks_flat'; flat_dir.mkdir(exist_ok=True)
ship, n_abstain = [], {}
for p in picks:
    s = json.load(open(p)); g = s['group']; pp = s['pool_pick']
    flat = {'group': g, 'arms': {'A': dict(route_id=f'{g}__{a.ref_arm}', P=pp['P'], z_mean=pp['z_mean'], T=pp['T'], E_analytic=pp.get('E_analytic'), pool_index=pp['index'])}}
    for arm, r in s['arms'].items():
        if r.get('abstained') or r.get('reuse_eval_v1'):
            flat['arms'][arm] = dict(route_id=f'{g}__{a.ref_arm}', P=pp['P'], z_mean=pp['z_mean'], T=pp['T'], E_analytic=pp.get('E_analytic'), abstained=True, gain=r.get('gain'))
            n_abstain[arm] = n_abstain.get(arm, 0) + 1
        else:
            k = r['pick']
            flat['arms'][arm] = dict(route_id=r['route_id'], P=k['P'], z_mean=k['z_mean'], T=k['T'], E_analytic=k.get('E_analytic'), abstained=False, gain=r.get('gain'))
            assert (out / 'routes' / f"{r['route_id']}.json").exists(), r['route_id']
    json.dump(flat, open(flat_dir / f'{g}.json', 'w'), indent=1)
for t in tasks:
    if not t.get('run'): continue
    rp = out / 'routes' / f"{t['id']}.json"; assert rp.exists(), rp
    ship.append(dict(id=t['id'], group=t['group'], case=t['case'], route=f"{a.cluster_prefix}/{t['id']}.json", run=True, tier=t.get('tier', 0),
                     episode_seed=t['episode_seed'], sha256=hashlib.sha256(rp.read_bytes()).hexdigest()))
json.dump(ship, open(out / 'tasks_cluster.json', 'w'), indent=1)
lock = hashlib.sha256()
for rp in sorted((out / 'routes').glob('*.json')):
    lock.update(rp.name.encode()); lock.update(hashlib.sha256(rp.read_bytes()).digest())
if not a.no_lock:
    (out / 'PICKS_LOCKED.sha256').write_text(lock.hexdigest() + '  routes/*.json (name + content, sorted)\n')
print(f'groups {len(picks)}  drives to run {len(ship)}  abstained {n_abstain}  lock {lock.hexdigest()[:16]}')
