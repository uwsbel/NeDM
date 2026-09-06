#!/usr/bin/env python
"""Plan §28 step 3 table: every method picks one route per layout from the SAME candidate bank (the layout's recorded
runs, all with Chrono ground truth); report, in this order, successful selections, cost among successes (regret
against Chrono's best feasible route), and rejection of feasible options.

Methods: fixed heuristics on the bank (rule-based profile ``slope_aware``; fastest constant speed), the cheap predictor
(``--cheap NAME=predictions.json``: pick the lowest predicted cost among routes with p_feasible >= --tau, else the most
feasible), and dynamics models (``--imagine NAME=rows.json`` from ``traverse_wp7_imagine_cache.py``: lowest imagined
cost among the accepted routes; none accepted = no pick). Cost = time + energy / 10 (kJ).
"""
from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

COST = lambda t, e: t + e / 10.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="schema-v2 cache (labels.json = the bank's Chrono ground truth)")
    ap.add_argument("--arenas", nargs="+", required=True)
    ap.add_argument("--cheap", nargs="+", action="extend", default=[], help="NAME=predictions.json (traverse_wp7_cheap_predictor.py)")
    ap.add_argument("--imagine", nargs="+", action="extend", default=[], help="NAME=rows.json (traverse_wp7_imagine_cache.py)")
    ap.add_argument("--tau", type=float, default=0.5, help="cheap predictor feasibility threshold")
    ap.add_argument("--kinds", nargs="*", default=None, help="restrict to layout kinds (crossing, sequence, freeform)")
    ap.add_argument("--feasible", choices=["strict", "completed"], default="strict",
                    help="strict: completed, no stall >= 1 s, no contact (the pilot's definition); completed: completed, no contact")
    ap.add_argument("--hybrids", action="store_true", help="also cross every cheap predictor's gate with every world model's cost and vice versa")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    labels = json.loads((Path(args.cache) / "labels.json").read_text())
    bank = defaultdict(dict)  # layout -> candidate -> label row
    for k, r in labels.items():
        if r["arena"] in args.arenas and (args.kinds is None or r["kind"] in args.kinds):
            r = dict(r); r["key"] = k
            r["feasible"] = bool(r.get("completed")) and not bool(r.get("contact")) and (args.feasible == "completed" or not bool(r.get("stalled")))
            bank[r["layout"]][r["candidate"]] = r
    layouts = sorted(l for l in bank if any(r["feasible"] for r in bank[l].values()))
    nosol = sorted(l for l in bank if not any(r["feasible"] for r in bank[l].values()))  # "no candidate works": abstention task
    n_nosol = len(nosol)
    best = {l: min((r for r in bank[l].values() if r["feasible"]), key=lambda r: COST(r["time_s"], r["energy_kj"])) for l in layouts}

    def speed_of(c: str) -> float:
        if c.startswith("direct_v"):
            return float(c[8:])
        if c.startswith("seq_v"):
            return float(c[5:].split("_")[0])
        if c.startswith("free_") and "_v" in c:
            return float(c.rsplit("_v", 1)[1])
        return -1.0

    methods: dict[str, dict] = {}
    # heuristics: pick a candidate name per layout, reject nothing
    every = layouts + nosol
    methods["rule (slope_aware)"] = {l: ("slope_aware" if "slope_aware" in bank[l] else None) for l in every}
    spd = lambda l, c: bank[l][c].get("mean_speed") if bank[l][c].get("mean_speed") is not None else speed_of(c)
    methods["fastest (mean commanded speed)"] = {l: max((c for c in bank[l] if speed_of(c) > 0), key=lambda c: spd(l, c), default=None) for l in every}
    methods["slowest (mean commanded speed)"] = {l: min((c for c in bank[l] if speed_of(c) > 0), key=lambda c: spd(l, c), default=None) for l in every}
    rejects: dict[str, dict] = {}  # method -> layout -> set of rejected candidates
    for spec in args.cheap:
        name, path = spec.split("=", 1)
        pred = {p["key"]: p for p in json.loads(Path(path).read_text())}
        picks, rej = {}, {}
        for l in every:
            rows = [(c, pred[r["key"]]) for c, r in bank[l].items() if r["key"] in pred]
            if not rows:
                picks[l] = None; continue
            ok = [(c, p) for c, p in rows if p["p_feasible"] >= args.tau]
            rej[l] = {c for c, p in rows if p["p_feasible"] < args.tau}
            picks[l] = min(ok, key=lambda cp: COST(cp[1]["time_pred"], cp[1]["energy_pred"]))[0] if ok else None  # none above tau: abstain
        methods[f"cheap: {name}"] = picks; rejects[f"cheap: {name}"] = rej
    for spec in args.imagine:
        name, path = spec.split("=", 1)
        rows_by = defaultdict(dict)
        for r in json.loads(Path(path).read_text()):
            rows_by[r["layout"]][r["candidate"]] = r
        picks, rej = {}, {}
        for l in every:
            rows = rows_by.get(l, {})
            if not rows:
                picks[l] = None; continue
            ok = [(c, r) for c, r in rows.items() if r["img_ok"]]
            rej[l] = {c for c, r in rows.items() if not r["img_ok"]}
            picks[l] = min(ok, key=lambda cr: COST(cr[1]["img_time"], cr[1]["img_energy"]))[0] if ok else None
        methods[f"world model: {name}"] = picks; rejects[f"world model: {name}"] = rej

    if args.hybrids:
        cheap_preds = {spec.split("=", 1)[0]: {p["key"]: p for p in json.loads(Path(spec.split("=", 1)[1]).read_text())} for spec in args.cheap}
        wm_rows = {}
        for spec in args.imagine:
            name, path = spec.split("=", 1)
            wm_rows[name] = defaultdict(dict)
            for r in json.loads(Path(path).read_text()):
                wm_rows[name][r["layout"]][r["candidate"]] = r
        for cn, pred in cheap_preds.items():
            for wn, rows_by in wm_rows.items():
                g1, g2 = {}, {}
                for l in every:
                    cands = [c for c, r in bank[l].items() if r["key"] in pred and c in rows_by.get(l, {})]
                    gate_c = [c for c in cands if pred[bank[l][c]["key"]]["p_feasible"] >= args.tau]
                    g1[l] = min(gate_c, key=lambda c: COST(rows_by[l][c]["img_time"], rows_by[l][c]["img_energy"])) if gate_c else None
                    gate_w = [c for c in cands if rows_by[l][c]["img_ok"]]
                    g2[l] = min(gate_w, key=lambda c: COST(pred[bank[l][c]["key"]]["time_pred"], pred[bank[l][c]["key"]]["energy_pred"])) if gate_w else None
                methods[f"gate {cn} + cost {wn}"] = g1; rejects[f"gate {cn} + cost {wn}"] = rejects[f"cheap: {cn}"]
                methods[f"gate {wn} + cost {cn}"] = g2; rejects[f"gate {wn} + cost {cn}"] = rejects[f"world model: {wn}"]
    hdr = f"{'method':34s} {'layouts':>7s} {'picked':>6s} {'feasible':>8s} {'regret':>6s} {'max':>5s} {'rej.feas':>8s} {'rej.inf':>7s} {'abstain':>7s}"
    print(f"arenas {args.arenas}: {len(layouts)} layouts with a feasible route ({n_nosol} without), bank sizes {min(len(bank[l]) for l in layouts)}-{max(len(bank[l]) for l in layouts)}")
    print(hdr); print("-" * len(hdr))
    table = {}
    n_feas_total = sum(sum(r["feasible"] for r in bank[l].values()) for l in layouts)
    n_inf_total = sum(sum(not r["feasible"] for r in bank[l].values()) for l in layouts)
    for name, picks in methods.items():
        picked = [l for l in layouts if picks.get(l)]
        feas = [l for l in picked if bank[l][picks[l]]["feasible"]]
        reg = [COST(bank[l][picks[l]]["time_s"], bank[l][picks[l]]["energy_kj"]) / COST(best[l]["time_s"], best[l]["energy_kj"]) for l in feas]
        rej = rejects.get(name)
        rf = sum(sum(bank[l][c]["feasible"] for c in rej.get(l, ())) for l in layouts) if rej else 0
        ri = sum(sum(not bank[l][c]["feasible"] for c in rej.get(l, ())) for l in layouts) if rej else 0
        abstain = sum(1 for l in nosol if not picks.get(l))  # no-solution layouts on which the method picked nothing
        table[name] = {"layouts": len(layouts), "picked": len(picked), "pick_feasible": len(feas), "regret_mean": float(np.mean(reg)) if reg else None,
                       "regret_max": float(np.max(reg)) if reg else None, "rejected_feasible": rf, "rejected_infeasible": ri,
                       "n_feasible_routes": n_feas_total, "n_infeasible_routes": n_inf_total, "no_solution_layouts": n_nosol, "abstained": abstain}
        print(f"{name:34s} {len(layouts):7d} {len(picked):6d} {len(feas):8d} {np.mean(reg) if reg else float('nan'):6.2f} {np.max(reg) if reg else float('nan'):5.2f} "
              f"{(f'{rf}/{n_feas_total}' if rej else '-'):>8s} {(f'{ri}/{n_inf_total}' if rej else '-'):>7s} {abstain:3d}/{n_nosol:<3d}")
    print("feasible = picked route completed in Chrono without stall or contact; regret = Chrono cost of the pick / best feasible cost; "
          "rej.feas / rej.inf = feasible / infeasible bank routes the method rejected (world models: imagination not ok; cheap: p < tau); "
          "abstain = layouts with NO feasible bank route on which the method picked nothing")
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
