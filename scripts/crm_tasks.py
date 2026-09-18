"""Task list for the CRM training collection on the night-2 start/goal pool (1,200 groups x 20 routes).

Every group keeps its own 90/5/5 hash split, so held-out groups are collected with the same recipe. Each group's 20
routes (12 designed: 3 offsets x 4 speed profiles; 8 planner-proposal routes) get a per-group random order; tier k
holds the k-th route of every group. Workers drain tier by tier, so whenever the collection stops, all groups have
(almost) the same number of routes and the route mix stays balanced. Episode ids equal the rigid ids of the same
route (rigid/CRM outcomes pair 1:1); episode_seed = first 8 hex digits of md5(id), unique per episode.
"""
import argparse, hashlib, json, random

ap = argparse.ArgumentParser()
ap.add_argument('--groups', type=int, default=1200)
ap.add_argument('--out', required=True)
a = ap.parse_args()
rows = []
for i in range(a.groups):
    g = f'f104_v2_group_{i:04d}'
    routes = [(f'{g}_route_{k:02d}', f'cases/night2/routes/{g}/route_{k:02d}.json') for k in range(12)]
    routes += [(f'{g}_op_{k:02d}', f'cases/night2_onpolicy/routes/{g}/op_{k:02d}.json') for k in range(8)]
    random.Random(int(hashlib.md5(g.encode()).hexdigest()[:8], 16)).shuffle(routes)
    for tier, (rid, route) in enumerate(routes):
        rows.append(dict(id=rid, group=g, case=f'cases/night2/{g}.json', route=route, tier=tier,
                         episode_seed=int(hashlib.md5(rid.encode()).hexdigest()[:8], 16), run=True))
assert len({r['id'] for r in rows}) == len(rows) and len({r['episode_seed'] for r in rows}) == len(rows)
json.dump(rows, open(a.out, 'w'))
print(len(rows), 'tasks;', len({r['group'] for r in rows}), 'groups; tiers 0-19')
