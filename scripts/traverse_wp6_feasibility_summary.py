#!/usr/bin/env python
"""Feasibility map from the terrain sweep: per challenge x route, what happened in Chrono.

Feasible = completed, no stall, no roll/pitch abort. Columns per route: time, energy, seconds stalled,
seconds with an unloaded wheel, max roll / pitch, tracker speed error. The last block asks the planning
question directly: on each challenge, which route-and-speed combination is cheapest among the feasible
ones, and what does the rule-based slope-aware profile cost relative to it?
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("batch")
ap.add_argument("--challenges", default="artifacts/traverse/wp6_challenges/challenges.json")
ap.add_argument("--json", default=None)
args = ap.parse_args()
rows = [json.loads(l) for l in Path(args.batch, "rows.jsonl").read_text().splitlines() if l.strip().startswith("{")]
ch = {c["id"]: c for c in json.loads(Path(args.challenges).read_text())}
by = defaultdict(dict)
for r in rows:
    by[r["key"]][r["candidate"]] = r
cands = sorted({r["candidate"] for r in rows}, key=lambda c: (not c.startswith("direct"), c))
cost = lambda r: r["time_s"] + r["energy_kj"] / 10
feasible = lambda r: r["completed"] and not r.get("stalled") and r["status"] == "completed"
print(f"{'challenge':24s} {'up/down deg':>11s} | " + " ".join(f"{c[-7:]:>9s}" for c in cands) + "  | best feasible -> cost | slope_aware cost (x best)")
out = {}
for key in sorted(by, key=lambda k: (ch.get(k, {}).get("kind", ""), k)):
    rs = by[key]; c = ch.get(key, {})
    cells = []
    for cand in cands:
        r = rs.get(cand)
        if r is None:
            cells.append(f"{'-':>9s}"); continue
        if feasible(r):
            cells.append(f"{cost(r):6.1f}{'u' if r.get('unloaded_s', 0) > 0.5 else ' '}{'!' if r['max_pitch_deg'] > 25 or r['max_roll_deg'] > 25 else ' '} ")
        else:
            cells.append(f"{'STALL' if r.get('stalled') else r['status'][:6]:>9s}")
    feas = {n: r for n, r in rs.items() if feasible(r)}
    best = min(feas, key=lambda n: cost(feas[n])) if feas else None
    sa = rs.get("slope_aware")
    tail = f"{best or '-':12s} {cost(feas[best]) if best else float('nan'):6.1f} | " + (f"{cost(sa):6.1f} ({cost(sa) / cost(feas[best]):.2f}x)" if sa and feasible(sa) and best else f"{'infeasible' if sa else '-':>14s}")
    print(f"{key:24s} {c.get('line_max_up_deg', float('nan')):5.1f}/{c.get('line_max_down_deg', float('nan')):4.1f} | " + " ".join(cells) + "  | " + tail)
    out[key] = {"best_feasible": best, "best_cost": cost(feas[best]) if best else None, "slope_aware_cost": cost(sa) if sa and feasible(sa) else None,
                "n_feasible": len(feas), "n_routes": len(rs), "infeasible": [n for n, r in rs.items() if not feasible(r)],
                "routes": {n: {k: r.get(k) for k in ("status", "completed", "stalled", "stall_s", "unloaded_s", "time_s", "energy_kj", "max_roll_deg", "max_pitch_deg", "mean_speed_err_mps", "max_ct_m", "min_tire_fz_n")} for n, r in rs.items()}}
print("cells: cost = time + kJ/10 for feasible routes; 'u' = a wheel unloaded for > 0.5 s; '!' = roll or pitch over 25 deg; STALL / status otherwise")
n_inf = sum(len(v["infeasible"]) for v in out.values()); n_all = sum(v["n_routes"] for v in out.values())
print(f"{n_inf}/{n_all} routes infeasible over {len(out)} challenges; slope-aware infeasible on {sum(1 for v in out.values() if v['slope_aware_cost'] is None)} challenges; "
      f"slope-aware is the cheapest feasible route on {sum(1 for v in out.values() if v['best_feasible'] == 'slope_aware')}")
ratios = [v["slope_aware_cost"] / v["best_cost"] for v in out.values() if v["slope_aware_cost"] and v["best_cost"]]
if ratios:
    print(f"slope-aware cost / best feasible cost: mean {np.mean(ratios):.2f}, max {np.max(ratios):.2f}")
if args.json:
    Path(args.json).write_text(json.dumps(out, indent=1))
