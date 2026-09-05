#!/usr/bin/env python
"""Plan camera-only (occupancy + memorized terrain) with the margin fallback on every routed
held-out layout and export the plans that needed a margin below 0.9 m, for Chrono validation."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nedm.traverse import nrd_data as D
from nedm.traverse.layout import EpisodeLayout
from nedm.traverse.oracle import PlannerParams
from nedm.traverse.planner_b import MapDecoder, plan_on_predicted_map
from nedm.traverse.terrain import TerrainMap

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="artifacts/traverse/wp4_fallback_plans/routes.json")
ap.add_argument("--maphead", default="artifacts/traverse/wp4_maphead_v2/ckpt_best.pt")
ap.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6")
ap.add_argument("--routes", default="artifacts/traverse/wp3_routes")
ap.add_argument("--split", default="val")
ap.add_argument("--threshold", type=float, default=0.85)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
cache = Path(args.cache)
keys = D.load_cache_keys(cache)
split = dict(zip(("train", "val", "test"), D.split_keys(keys)))[args.split]
manifest = json.loads((Path(args.routes) / "routes_manifest.json").read_text())
routed = set().union(*manifest["families"].values())
keys = [k for k in split if k in routed]
if args.limit: keys = keys[: args.limit]
tmap = TerrainMap.from_dir(Path("assets/traverse/arena_v1"))
dec = MapDecoder(Path(args.maphead), Path("assets/traverse/arena_v1"), "cuda")
out, used, t0 = {}, {}, time.time()
for i, key in enumerate(keys):
    store, ep = key.split("__", 1)
    layout = EpisodeLayout.from_json(json.loads((Path("artifacts/traverse") / store / ep / "meta.json").read_text())["layout"])
    with np.load(cache / f"{key}.npz") as d:
        scene_map = d["map_v2"]
    plan, info = plan_on_predicted_map(dec, scene_map, layout.start_xy, layout.house_xy, PlannerParams(),
                                       true_terrain=tmap, threshold=args.threshold, margin_fallback=True)
    m = plan.meta.get("tracker_margin_m") if plan is not None else None
    used[m] = used.get(m, 0) + 1
    if plan is not None and m < 0.9:
        out[key] = [{"candidate": f"fallback_m{m}", "waypoints": plan.waypoints.tolist(), "speeds": plan.speeds.tolist(),
                     "headings": plan.headings.tolist(), "stations": plan.stations.tolist()}]
    if i % 100 == 0:
        print(f"[{i}/{len(keys)}] {time.time() - t0:.0f}s margins used {used}", flush=True)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(out))
print(f"done: {len(keys)} layouts, margins used {used}; exported {len(out)} rescued plans -> {args.out}", flush=True)
