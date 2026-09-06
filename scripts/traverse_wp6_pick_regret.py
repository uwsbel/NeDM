#!/usr/bin/env python
"""Regret of each picker among the IDENTICAL sweep routes of a terrain challenge set.

Pickers: the rule-based slope-aware profile, always-fastest, always-slowest, the geometry regression (profile
time + Chrono-fitted energy regression) and, when an imagine-sweep result is given, the world model (imagined
cost over the routes it accepts). Regret = Chrono cost of the pick / Chrono cost of the best feasible route;
'infeasible' counts picks that stalled, left the route, made contact or timed out.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nedm.traverse.energy_floor import EnergyFloor
from nedm.traverse.oracle import PlanCandidate
from nedm.traverse.terrain import TerrainMap

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--challenges", required=True)
ap.add_argument("--sweep", required=True)
ap.add_argument("--arena", required=True)
ap.add_argument("--imagined", default=None, help="imagine-sweep rows.json (world model)")
ap.add_argument("--energy-floor", default="artifacts/traverse/wp5_energy_floor/energy_floor.json")
args = ap.parse_args()
tmap = TerrainMap.from_dir(Path(args.arena)); fit = EnergyFloor.load(Path(args.energy_floor), 0.0)
prof = lambda r: float(np.sum(np.diff(np.asarray(r["stations"])) / np.maximum(0.5 * (np.asarray(r["speeds"])[:-1] + np.asarray(r["speeds"])[1:]), 0.5)))
tasks = defaultdict(dict)
for t in json.loads(Path(args.challenges, "tasks.json").read_text()): tasks[t["key"]][t["candidate"]] = t["route"]
rows = defaultdict(dict)
for l in Path(args.sweep, "rows.jsonl").read_text().splitlines():
    if l.strip().startswith("{"): r = json.loads(l); rows[r["key"]][r["candidate"]] = r
img = defaultdict(dict)
if args.imagined:
    for r in json.loads(Path(args.imagined).read_text()): img[r["key"]][r["candidate"]] = r
feasible = lambda r: r["completed"] and not r.get("stalled") and r["status"] == "completed" and not r.get("contact", False)
cost = lambda r: r["time_s"] + r["energy_kj"] / 10
pickers = ["rule (slope_aware)", "always fastest (8 m/s)", "always slowest (2 m/s)", "geometry regression"] + (["world model (frozen)"] if img else [])
reg = {p: [] for p in pickers}; inf = {p: 0 for p in pickers}; n = 0
for key, rs in rows.items():
    f = {c: r for c, r in rs.items() if feasible(r)}
    if not f: continue
    n += 1; best = min(f.values(), key=cost)
    g = {c: prof(tasks[key][c]) + fit.fit_kj(PlanCandidate(waypoints=np.array(tasks[key][c]["waypoints"]), speeds=np.array(tasks[key][c]["speeds"]), headings=np.array(tasks[key][c]["headings"]), stations=np.array(tasks[key][c]["stations"]), meta={}), tmap) / 10 for c in rs}
    picks = {"rule (slope_aware)": "slope_aware", "always fastest (8 m/s)": "direct_v8", "always slowest (2 m/s)": "direct_v2", "geometry regression": min(g, key=g.get)}
    if img:
        w = {c: r["img_time"] + r["img_energy"] / 10 for c, r in img[key].items() if r["img_ok"]}
        picks["world model (frozen)"] = min(w, key=w.get) if w else None
    for p, c in picks.items():
        if c is None or c not in rs or not feasible(rs[c]): inf[p] += 1
        else: reg[p].append(cost(rs[c]) / cost(best))
print(f"{n} challenges with a feasible route ({len(rows) - n} with none); pick among the same {len(next(iter(rows.values())))} routes -> Chrono cost / best feasible")
for p in pickers:
    r = np.array(reg[p]); print(f"  {p:26s} infeasible pick {inf[p]:2d}/{n}   regret mean {r.mean():.2f} median {np.median(r):.2f} max {r.max():.2f} (n={len(r)})" if len(r) else f"  {p:26s} infeasible pick {inf[p]:2d}/{n}")
