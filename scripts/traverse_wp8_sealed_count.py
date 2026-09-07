#!/usr/bin/env python
"""Model-free selection of the fresh sealed pair (plan §32 / notes §13.12): count, per collected arena, the layouts on
which the fastest-commanded-speed candidate is infeasible while a feasible alternative exists (the only layouts on which
a gate can beat the heuristic), split by contact-only vs stall/timeout failures, and write sealed_pair.json (the two
arenas with the most stall/timeout-type such layouts). Uses labels only; no model touches these arenas.
  python scripts/traverse_wp8_sealed_count.py --cache artifacts/traverse/wp8_cache_sealed2
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n-pick", type=int, default=2)
    args = ap.parse_args()
    cache = Path(args.cache)
    labels = json.loads((cache / "labels.json").read_text()); man = json.loads((cache / "cache_manifest.json").read_text())
    feasible = lambda c: bool(c.get("completed")) and not c.get("stalled") and not c.get("contact")
    by = defaultdict(lambda: defaultdict(dict))
    for k, c in labels.items():
        by[c["arena"]][c["layout"]][c["candidate"]] = c
    rows = {}
    for aid, lays in by.items():
        n_lay = n_feas = n_room = n_room_stall = n_nosol = 0
        for lay, cands in lays.items():
            n_lay += 1
            if not any(feasible(c) for c in cands.values()):
                n_nosol += 1; continue
            n_feas += 1
            fast = max(cands.values(), key=lambda c: c["mean_speed"])
            if not feasible(fast):
                n_room += 1
                if not (fast.get("completed") and not fast.get("stalled") and fast.get("contact")):
                    n_room_stall += 1
        rows[aid] = {"layouts": n_lay, "with_feasible": n_feas, "no_solution": n_nosol, "heuristic_fails": n_room, "heuristic_fails_stall_or_timeout": n_room_stall,
                     "runs": sum(len(c) for c in lays.values()), "infeasible_runs": sum(1 for c in lays.values() for r in c.values() if not feasible(r))}
        print(f"{aid}: {n_lay} layouts, {n_feas} with a feasible route ({n_nosol} none), heuristic fails on {n_room} of them ({n_room_stall} stall/timeout, {n_room - n_room_stall} contact-only); {rows[aid]['infeasible_runs']}/{rows[aid]['runs']} runs infeasible")
    pick = sorted(rows, key=lambda a: (-rows[a]["heuristic_fails_stall_or_timeout"], -rows[a]["heuristic_fails"]))[: args.n_pick]
    tot = sum(rows[a]["heuristic_fails"] for a in pick); tot_s = sum(rows[a]["heuristic_fails_stall_or_timeout"] for a in pick)
    print(f"sealed pair: {pick} -> {tot} heuristic-failure layouts with a feasible alternative ({tot_s} stall/timeout); target >= 30")
    (cache / "sealed_pair.json").write_text(json.dumps({"pair": pick, "counts": rows, "heuristic_fail_layouts": tot, "stall_timeout": tot_s, "target_met": tot >= 30}, indent=1))


if __name__ == "__main__":
    main()
