#!/usr/bin/env python
"""Per-pick Chrono summary for a merged sampling-planner batch (routes deduplicated by
``traverse_wp5_merge_routes.py``: a row's candidate 'a+b' counts for picks a and b).

Reports completion / contact, Chrono time, energy and combined cost per pick, how often each pick
beats the A* pick of the same layout, and -- given the planner results.json files -- the Chrono /
imagined ratio of energy and time at the pick (the optimiser's-curse gauge).
"""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("batch")
ap.add_argument("--results", nargs="*", default=[], help="planner results.json files (imagined values at the picks)")
ap.add_argument("--reference", default="astar_best")
ap.add_argument("--controller", default="tracker")
args = ap.parse_args()

rows = {}
for line in Path(args.batch, "rows.jsonl").read_text().splitlines():
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    if args.controller not in r["controller"]:
        continue
    for name in r["candidate"].split("+"):
        rows.setdefault(name, {})[r["key"]] = r
imagined = {}
for f in args.results:
    for r in json.loads(Path(f).read_text()):
        prefix = r.get("pick_prefix", "")
        for k, v in r.items():
            if k.endswith("_time") or k.endswith("_energy"):
                imagined.setdefault(r["key"], {})[prefix + k] = v
ref = rows.get(args.reference, {})
print(f"{'pick':22s} {'n':>3s} {'done':>5s} {'contact':>7s} {'time':>6s} {'energy':>7s} {'cost':>6s} | {'beats ref':>9s} | {'E ch/img':>8s} {'t ch/img':>8s} {'minclr':>6s}")
for name in sorted(rows, key=lambda n: (n != args.reference, n)):
    rs = rows[name]
    done = [r for r in rs.values() if r["completed"]]
    common = [k for k in rs if k in ref and rs[k]["completed"] and ref[k]["completed"]]
    cost = lambda r: r["time_s"] + r["energy_kj"] / 10
    beats = sum(cost(rs[k]) < cost(ref[k]) for k in common)
    e_ratio = t_ratio = float("nan")
    ie = [(r["energy_kj"], imagined[k].get(f"{name}_energy")) for k, r in rs.items() if r["completed"] and k in imagined]
    it = [(r["time_s"], imagined[k].get(f"{name}_time")) for k, r in rs.items() if r["completed"] and k in imagined]
    ie = [(a, b) for a, b in ie if b is not None]; it = [(a, b) for a, b in it if b is not None]
    if ie:
        e_ratio = np.mean([a for a, _ in ie]) / np.mean([b for _, b in ie]); t_ratio = np.mean([a for a, _ in it]) / np.mean([b for _, b in it])
    print(f"{name:22s} {len(rs):3d} {len(done) / max(len(rs), 1):5.2f} {sum(r['contact'] for r in rs.values()):7d} "
          f"{np.mean([r['time_s'] for r in done]):6.2f} {np.mean([r['energy_kj'] for r in done]):7.1f} {np.mean([cost(r) for r in done]):6.2f} | "
          f"{beats:4d}/{len(common):<4d} | {e_ratio:8.2f} {t_ratio:8.2f} {np.mean([r['min_clearance_m'] for r in done]):6.2f}")
