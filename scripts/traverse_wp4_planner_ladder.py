#!/usr/bin/env python
"""Planner ablation ladder (plan section 7): plans from the camera-derived map judged on the TRUE map.

Rungs, all held-out layouts (val split, oracle family so start/goal are oracle-feasible):
  oracle          true heightmap + true footprints (privileged reference)
  pred_occ        predicted occupancy + true (memorized) terrain
  pred_full       predicted occupancy + predicted elevation           <- camera only
  straight        straight line start -> nearest approach-ring point  (naive bracket)
Every plan is validated against the true map with the oracle's footprint sweep:
collision (true clearance < 0), path-length and energy-proxy ratios vs the oracle
plan, and no-path rate. Run for the interim 0.9 m tracker margin and the measured one.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nedm.traverse import nrd_data as D
from nedm.traverse.layout import EpisodeLayout
from nedm.traverse.oracle import OracleGrid, PlanCandidate, PlannerParams, plan_to_ring, plan_to_ring_fallback, validate_candidate, _energy_per_m
from nedm.traverse.planner_b import MapDecoder, plan_on_predicted_map
from nedm.traverse.terrain import TerrainMap


def straight_line(tmap: TerrainMap, start_xy, ring_center, params: PlannerParams) -> PlanCandidate:
    s, c = np.asarray(start_xy), np.asarray(ring_center)
    d = c - s
    end = c - d / np.linalg.norm(d) * params.approach_ring_m
    n = max(2, int(np.linalg.norm(end - s) / params.sample_step_m) + 1)
    pts = s[None] + np.linspace(0, 1, n)[:, None] * (end - s)[None]
    seg = np.hypot(*np.diff(pts, axis=0).T)
    return PlanCandidate(waypoints=pts, speeds=np.full(n, params.v_cruise_mps), headings=np.full(n, math.atan2(*d[::-1])),
                         stations=np.concatenate([[0.0], np.cumsum(seg)]), meta={"length_m": float(seg.sum())})


def judge(true_grid: OracleGrid, plan: PlanCandidate, params: PlannerParams) -> dict:
    chk = validate_candidate(true_grid, plan.waypoints, params)
    pts = plan.waypoints
    seg = np.hypot(*np.diff(pts, axis=0).T)
    gx, gy = true_grid.tmap.gradient(pts[:, 0], pts[:, 1])
    t = np.gradient(pts, axis=0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    s_along = gx * t[:, 0] + gy * t[:, 1]
    e = float(np.sum(seg * np.asarray(_energy_per_m(s_along, replace(params, energy_weight=1.0)))[:-1]))
    return {"true_min_clearance_m": chk["min_footprint_clearance_m"], "collision": not chk["clearance_ok"],
            "slope_ok": chk["slope_ok"], "kappa_ok": chk["kappa_ok"], "length_m": float(seg.sum()), "energy_proxy": e}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--maphead", default="artifacts/traverse/wp4_maphead_v1/ckpt_best.pt")
    ap.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6")
    ap.add_argument("--stores", default="artifacts/traverse")
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--routes", default="artifacts/traverse/wp3_routes")
    ap.add_argument("--split", default="val")
    ap.add_argument("--families", nargs="+", default=["oracle"])
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--margins", type=float, nargs="+", default=[0.9, 0.3])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--elev-smooth", type=float, default=1.0)
    ap.add_argument("--margin-fallback", action="store_true", help="0.9 -> 0.6 -> 0.3 rescue for all rungs")
    ap.add_argument("--goal", choices=["true", "predicted"], default="true", help="ring centre for the predicted rungs")
    ap.add_argument("--map-key", default="map_v2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cache, routes = Path(args.cache), Path(args.routes)
    keys = D.load_cache_keys(cache)
    split = dict(zip(("train", "val", "test"), D.split_keys(keys)))[args.split]
    manifest = json.loads((routes / "routes_manifest.json").read_text())
    allowed = set().union(*(set(manifest["families"][f]) for f in args.families))
    keys = [k for k in split if k in allowed][: args.episodes]
    tmap = TerrainMap.from_dir(Path(args.arena))
    decoder = MapDecoder(Path(args.maphead), Path(args.arena), args.device)

    rows = []
    t0 = time.time()
    for n, key in enumerate(keys):
        store, ep = key.split("__", 1)
        meta = json.loads((Path(args.stores) / store / ep / "meta.json").read_text())
        layout = EpisodeLayout.from_json(meta["layout"])
        with np.load(cache / f"{key}.npz") as d:
            scene_map = d[args.map_key]
        for margin in args.margins:
            params = replace(PlannerParams(), tracker_p95_margin_m=margin)
            true_grid = OracleGrid(tmap, layout.obstacles(), params)
            planner = plan_to_ring_fallback if args.margin_fallback else plan_to_ring
            plans = {"oracle": planner(tmap, layout.obstacles(), layout.start_xy, layout.house_xy, params)}
            t1 = time.time()
            plans["pred_occ"], info = plan_on_predicted_map(decoder, scene_map, layout.start_xy, layout.house_xy, params,
                                                           true_terrain=tmap, threshold=args.threshold, margin_fallback=args.margin_fallback,
                                                           goal_from_camera=args.goal == "predicted")
            plans["pred_full"], info_full = plan_on_predicted_map(decoder, scene_map, layout.start_xy, layout.house_xy, params,
                                                                 threshold=args.threshold, elev_smooth=args.elev_smooth,
                                                                 margin_fallback=args.margin_fallback, goal_from_camera=args.goal == "predicted")
            reasons = {"pred_occ": info["reason"], "pred_full": info_full["reason"]}
            plan_s = time.time() - t1
            plans["straight"] = straight_line(tmap, layout.start_xy, layout.house_xy, params)
            ref = judge(true_grid, plans["oracle"], params) if plans["oracle"] else None
            for rung, plan in plans.items():
                row = {"key": key, "margin": margin, "rung": rung, "found": plan is not None,
                       "occupied_cells": info["occupied_cells"], "plan_s": plan_s, "reason": reasons.get(rung)}
                if plan is not None:
                    j = judge(true_grid, plan, params)
                    row.update(j)
                    row["margin_used"] = plan.meta.get("tracker_margin_m", margin)
                    end_d = float(np.hypot(plan.waypoints[-1, 0] - layout.house_xy[0], plan.waypoints[-1, 1] - layout.house_xy[1]))
                    row["end_ring_err_m"] = abs(end_d - params.approach_ring_m)
                    row["on_true_ring"] = bool(row["end_ring_err_m"] <= params.approach_ring_tol_m)
                    if ref:
                        row["length_ratio"] = j["length_m"] / ref["length_m"]
                        row["energy_ratio"] = j["energy_proxy"] / ref["energy_proxy"]
                rows.append(row)
        if n % 10 == 0:
            print(f"[{n + 1}/{len(keys)}] {time.time() - t0:.0f}s  cells {info['occupied_cells']}", flush=True)

    summary = {}
    for margin in args.margins:
        for rung in ("oracle", "pred_occ", "pred_full", "straight"):
            sel = [r for r in rows if r["margin"] == margin and r["rung"] == rung]
            found = [r for r in sel if r["found"]]
            summary[f"m{margin}/{rung}"] = {
                "n": len(sel), "no_path_rate": 1 - len(found) / max(len(sel), 1),
                "collision_rate": float(np.mean([r["collision"] for r in found])) if found else None,
                "slope_violation_rate": float(np.mean([not r["slope_ok"] for r in found])) if found else None,
                "min_true_clearance_p05_m": float(np.percentile([r["true_min_clearance_m"] for r in found], 5)) if found else None,
                "length_ratio_mean": float(np.mean([r["length_ratio"] for r in found if "length_ratio" in r])) if found else None,
                "energy_ratio_mean": float(np.mean([r["energy_ratio"] for r in found if "energy_ratio" in r])) if found else None,
                "on_true_ring_rate": float(np.mean([r["on_true_ring"] for r in found])) if found else None,
                "end_ring_err_p95_m": float(np.percentile([r["end_ring_err_m"] for r in found], 95)) if found else None,
            }
    from collections import Counter
    for rung in ("pred_occ", "pred_full"):
        c = Counter(r["reason"] for r in rows if r["rung"] == rung and not r["found"])
        print(f"{rung} failure reasons: {dict(c)}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "ladder.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(f"{'rung':22s} {'no-path':>8s} {'collide':>8s} {'slope!':>7s} {'clear p05':>10s} {'len ratio':>10s} {'E ratio':>8s} {'on ring':>8s} {'ring err p95':>12s}")
    for k, v in summary.items():
        f = lambda x, p=3: "  -  " if x is None else f"{x:.{p}f}"
        print(f"{k:22s} {f(v['no_path_rate']):>8s} {f(v['collision_rate']):>8s} {f(v['slope_violation_rate']):>7s} "
              f"{f(v['min_true_clearance_p05_m'], 2):>10s} {f(v['length_ratio_mean']):>10s} {f(v['energy_ratio_mean']):>8s} "
              f"{f(v['on_true_ring_rate']):>8s} {f(v['end_ring_err_p95_m'], 2):>12s}")


if __name__ == "__main__":
    main()
