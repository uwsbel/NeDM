#!/usr/bin/env python3
"""B0 tracking-suite task rows for both worlds (PLAN v2 B0/B5): three arms per route through the ext collectors.

  native_pid  unmodified follower rules (crm_collect_ext / gen_collect_ext --mode native)
  held_pid    shadow follower output held for 50 ms (--mode pid_held, 40 s near-stop rule)
  policy      numpy actor (--mode policy --actor <npz>, 40 s near-stop rule)

CRM rows are relative to CRM_ROOT (cases/night2/...), rigid rows carry absolute cluster paths for gen_runner_g.py and
share one shard per route (md5(route id) % shards) so the three arms of a route run in the same array task (same node).
"""
import argparse, hashlib, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', required=True); ap.add_argument('--actor', required=True, help='absolute cluster path of actor.npz')
    ap.add_argument('--arms', default='native_pid,held_pid,policy'); ap.add_argument('--shards', type=int, default=6)
    ap.add_argument('--crm-out', required=True); ap.add_argument('--rigid-out', required=True)
    ap.add_argument('--rigid-cases', default='/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/cases_night2_v1')
    ap.add_argument('--tag', default='b0')
    a = ap.parse_args()
    s = json.load(open(a.suite)); routes = s['routes'] if 'routes' in s else s
    arms = [x for x in a.arms.split(',') if x]
    extra = {'native_pid': [], 'held_pid': ['--mode', 'pid_held', '--near-stop-s', '40'],
             'policy': ['--mode', 'policy', '--actor', a.actor, '--near-stop-s', '40']}
    crm, rigid = [], []
    for i, r in enumerate(routes):
        g, rid = r['group'], r['id']; rfile = r['route_file'].split('/routes/')[-1]   # <g>/route_XX.json
        for arm in arms:
            base = dict(group=g, route_id=rid, arm=arm, stratum=r['stratum'], run=True, tier=i, tag=a.tag,
                        episode_seed=int(hashlib.md5(f'{rid}__{arm}'.encode()).hexdigest()[:8], 16), route_sha256=r['route_sha256'])
            crm.append(dict(base, id=f'{rid}__{arm}', case=f'cases/night2/{g}.json', route=f'cases/night2/routes/{rfile}', extra=list(extra[arm])))
            rigid.append(dict(base, id=f'{rid}__{arm}', case=f'{a.rigid_cases}/{g}.json', route=f'{a.rigid_cases}/routes/{rfile}', arena='f104',
                              shard=int(hashlib.md5(rid.encode()).hexdigest(), 16) % a.shards,
                              mode=('native' if arm == 'native_pid' else ('pid_held' if arm == 'held_pid' else 'policy')), extra=list(extra[arm])))
    json.dump(crm, open(a.crm_out, 'w'), indent=1); json.dump(rigid, open(a.rigid_out, 'w'), indent=1)
    print(f'{len(routes)} routes x {len(arms)} arms -> {len(crm)} CRM rows, {len(rigid)} rigid rows; strata', {k: sum(1 for r in routes if r["stratum"] == k) for k in ("feasible", "infeasible")})


if __name__ == '__main__':
    main()
