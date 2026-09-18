"""Convert planner_arms.py output (rigid world) into a gen_runner.py task file + the files to ship to the cluster GEN_ROOT.
  python scripts/n2_rigid_eval_tasks.py --picks-dir <planner_arms out> --case-prefix <path of the case dir relative to GEN_ROOT> --route-prefix <relative dir for routes> --shards 24 --out tasks.json"""
import argparse, hashlib, json
ap = argparse.ArgumentParser()
ap.add_argument('--picks-dir', required=True); ap.add_argument('--case-prefix', required=True); ap.add_argument('--route-prefix', required=True)
ap.add_argument('--shards', type=int, default=24); ap.add_argument('--out', required=True); ap.add_argument('--arena', default='f104')
a = ap.parse_args()
t = json.load(open(a.picks_dir + '/tasks.json')); rows = []
for r in t:
    g = r['group']; shard = int(hashlib.md5(g.encode()).hexdigest(), 16) % a.shards      # all arms of a group on one node (rigid Chrono is node-deterministic only)
    rows.append(dict(id=r['id'], group=g, arena=a.arena, case=f"{a.case_prefix}/{g}.json", route=f"{a.route_prefix}/{r['id']}.json", shard=shard, run=True))
json.dump(rows, open(a.out, 'w'), indent=1); print(len(rows), 'rows,', a.shards, 'shards ->', a.out)
