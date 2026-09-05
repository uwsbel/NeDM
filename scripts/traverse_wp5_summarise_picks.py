#!/usr/bin/env python
"""Per-pick Chrono summary for a merged planner batch (routes deduplicated by ``traverse_wp5_merge_routes.py``:
a row's candidate 'a+b' counts for picks a and b).

Per pick: completion / contact, Chrono time, energy and combined cost (time + energy / 10), paired wins and
mean cost difference against ``--reference`` (default: the plain A* baseline), and -- given the planner
results.json files -- how well each planner PREDICTED its own route: Chrono / predicted energy and time
(the imagination for the world-model picks, the geometry regression for the geometry-scorer picks) and
Chrono / profile-implied time (what the rule-based speed profile promised). Safety / feasibility columns:
mean and WORST min clearance, routes under 0.3 m, tracker speed error, p95 cross-track, max roll and pitch.
"""
import argparse, json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("batch")
ap.add_argument("--results", nargs="*", default=[], help="planner results.json files (predicted values at the picks)")
ap.add_argument("--reference", default="astar_plain")
ap.add_argument("--controller", default="tracker")
ap.add_argument("--json", default=None, help="write the per-pick summary here")
ap.add_argument("--layouts", action="store_true", help="also print the per-layout cost table")
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
pred = {}  # key -> {prefix+pick+"_time"/"_energy"/"_profile_time": value}
for f in args.results:
    for r in json.loads(Path(f).read_text()):
        prefix = r.get("pick_prefix", "")
        d = pred.setdefault(r["key"], {})
        for k, v in r.items():
            if k.endswith("_time") or k.endswith("_energy") or k.endswith("_profile_time"):
                d[prefix + k] = v
        if r.get("deploy_fallback") is not None:  # deploy = cem_pess_clear or the plain A* fallback
            src = "astar_plain" if r["deploy_fallback"] else "cem_pess_clear"
            for suffix in ("_time", "_energy", "_profile_time"):
                if f"{src}{suffix}" in r:
                    d[f"{prefix}deploy{suffix}"] = r[f"{src}{suffix}"]
ref = rows.get(args.reference, {})
cost = lambda r: r["time_s"] + r["energy_kj"] / 10
hdr = (f"{'pick':20s} {'n':>3s} {'done':>4s} {'cont':>4s} | {'time':>6s} {'energy':>6s} {'cost':>6s} | {'wins':>7s} {'dcost':>6s} | "
       f"{'E/pred':>6s} {'t/pred':>6s} {'t/prof':>6s} | {'clr':>5s} {'worst':>5s} {'<0.3':>4s} | {'verr':>5s} {'ct95':>5s} {'roll':>5s} {'pitch':>5s}")
print(hdr); print("-" * len(hdr))
summary = {}
def ratio(pairs):
    pairs = [(a, b) for a, b in pairs if b is not None and np.isfinite(b) and b > 0]
    return float(np.mean([a for a, _ in pairs]) / np.mean([b for _, b in pairs])) if pairs else float("nan")
for name in sorted(rows, key=lambda n: (n != args.reference, n)):
    rs = rows[name]
    done = [r for r in rs.values() if r["completed"]]
    common = [k for k in rs if k in ref and rs[k]["completed"] and ref[k]["completed"]]
    wins = sum(cost(rs[k]) < cost(ref[k]) for k in common)
    diffs = np.array([cost(rs[k]) - cost(ref[k]) for k in common])
    dcost = float(diffs.mean()) if common else float("nan")
    dcost_se = float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")
    e_ratio = ratio([(r["energy_kj"], pred.get(k, {}).get(f"{name}_energy")) for k, r in rs.items() if r["completed"]])
    t_ratio = ratio([(r["time_s"], pred.get(k, {}).get(f"{name}_time")) for k, r in rs.items() if r["completed"]])
    p_ratio = ratio([(r["time_s"], pred.get(k, {}).get(f"{name}_profile_time")) for k, r in rs.items() if r["completed"]])
    clr = [r["min_clearance_m"] for r in done]
    s = {"n": len(rs), "completed": len(done), "contact": int(sum(r["contact"] for r in rs.values())),
         "time_s": float(np.mean([r["time_s"] for r in done])) if done else None, "energy_kj": float(np.mean([r["energy_kj"] for r in done])) if done else None,
         "cost": float(np.mean([cost(r) for r in done])) if done else None, "wins": wins, "n_common": len(common), "dcost_vs_ref": dcost, "dcost_se": dcost_se,
         "chrono_over_pred_energy": e_ratio, "chrono_over_pred_time": t_ratio, "chrono_over_profile_time": p_ratio,
         "min_clearance_mean": float(np.mean(clr)) if clr else None, "min_clearance_worst": float(np.min(clr)) if clr else None,
         "n_below_0p3": int(sum(c < 0.3 for c in clr)), "speed_err_mps": float(np.mean([r["mean_speed_err_mps"] for r in done])) if done else None,
         "p95_ct_m": float(np.mean([r["p95_ct_m"] for r in done])) if done else None,
         "max_roll_deg": float(np.max([r.get("max_roll_deg", np.nan) for r in done])) if done and all("max_roll_deg" in r for r in done) else None,
         "max_pitch_deg": float(np.max([r.get("max_pitch_deg", np.nan) for r in done])) if done and all("max_pitch_deg" in r for r in done) else None}
    summary[name] = s
    f = lambda v, w=6, p=2: f"{v:{w}.{p}f}" if v is not None and np.isfinite(v) else " " * (w - 1) + "-"
    print(f"{name:20s} {s['n']:3d} {len(done) / max(len(rs), 1):4.2f} {s['contact']:4d} | {f(s['time_s'])} {f(s['energy_kj'], 6, 1)} {f(s['cost'])} | "
          f"{wins:3d}/{len(common):<3d} {f(dcost, 6, 2)} | {f(e_ratio)} {f(t_ratio)} {f(p_ratio)} | {f(s['min_clearance_mean'], 5)} {f(s['min_clearance_worst'], 5)} {s['n_below_0p3']:4d} | "
          f"{f(s['speed_err_mps'], 5)} {f(s['p95_ct_m'], 5)} {f(s['max_roll_deg'], 5, 1)} {f(s['max_pitch_deg'], 5, 1)}")
print("wins / dcost: paired against", args.reference, "on layouts both completed; E/pred, t/pred: Chrono / the planner's own prediction; t/prof: Chrono / profile-implied time")
for name, s in summary.items():
    if name != args.reference and s["n_common"] > 1:
        print(f"  {name:20s} paired cost difference {s['dcost_vs_ref']:+.2f} +/- {s['dcost_se']:.2f} (SE, n={s['n_common']})")
if args.layouts:
    names = sorted(rows, key=lambda n: (n != args.reference, n))
    keys = sorted(set().union(*(set(rs) for rs in rows.values())))
    print("\n" + f"{'layout':32s} " + " ".join(f"{n[:14]:>14s}" for n in names))
    for k in keys:
        print(f"{k:32s} " + " ".join(f"{cost(rows[n][k]):14.2f}" if k in rows[n] and rows[n][k]['completed'] else f"{'-':>14s}" for n in names))
if args.json:
    Path(args.json).write_text(json.dumps(summary, indent=1))
