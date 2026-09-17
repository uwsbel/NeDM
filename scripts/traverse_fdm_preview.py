#!/usr/bin/env python3
"""Score and refine one recorded reference at a real measured anchor; no Chrono execution.

Only the selected measured prefix is passed into the planner. Recorded future
states are not used to generate, rank or validate the candidate paths.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nedm.traverse.fdm_data import build_history, load_episode, tracked_route_indices
from nedm.traverse.fdm_model import load_fdm_checkpoint
from nedm.traverse.fdm_mppi import MPPIConfig, ReferenceMPPI
from nedm.traverse.fdm_scoring import FDMReferenceScorer
from nedm.traverse.terrain import TerrainMap


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--arena", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--anchor", type=int, default=40)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    episode = load_episode(args.episode)
    if not 0 <= args.anchor < len(episode["state"]):
        raise ValueError("Anchor outside recorded episode")
    meta = episode["meta"]
    if meta["route"] is None:
        raise ValueError("A nominal PID reference route is required")
    model, checkpoint = load_fdm_checkpoint(args.checkpoint, args.device)
    history = build_history(episode["state"], episode["actions"], episode["poses"], args.anchor)
    anchor = episode["poses"][args.anchor]
    route, layout = meta["route"], meta["layout"]
    # Reproduce the driver's station using measured history up to the anchor.
    # Global nearest-point search can jump backwards or across a self-crossing.
    index = int(tracked_route_indices(route, episode["poses"][:args.anchor+1])[-1])
    route.setdefault("meta", {})["fdm_station"] = float(route["stations"][index])
    terrain = TerrainMap.from_dir(args.arena)
    scorer = FDMReferenceScorer(model, history, anchor, terrain, layout, route["waypoints"][-1],
                                elapsed_s=args.anchor*.05)
    start = time.perf_counter()
    base_prediction = scorer.predict([route])
    base_cost = float(scorer([route])[0])
    obstacles = [(a["x_m"], a["y_m"], a["footprint_radius_m"]) for a in layout["assets"]]
    result = ReferenceMPPI(MPPIConfig(samples=args.samples, iterations=args.iterations), args.seed).optimize(
        route, anchor, obstacles, scorer)
    selected_prediction = None if result["abstained"] else scorer.predict([result["route"]])
    payload = {"mode": "offline candidate preview; no new physical outcome or closed-loop validation",
               "observation": "privileged BMP and authored asset footprints",
               "checkpoint": str(args.checkpoint.resolve()), "model_arm": model.config.arm,
               "source_episode": str(args.episode.resolve()), "anchor_frame": args.anchor,
               "goal_xy": route["waypoints"][-1], "anchor_pose": anchor,
               "base_cost": base_cost, "base_prediction": base_prediction,
               "selected_prediction": selected_prediction, "mppi": result,
               "wall_s": time.perf_counter()-start,
               "risk_cost_channels": ["contact", "low_progress"],
               "rollover_cost_excluded": "no positive rollover examples in first training/validation pilot"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, default=json_default, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"out": str(args.out), "abstained": result["abstained"],
                      "base_cost": base_cost, "selected_cost": result["cost"],
                      "model_evaluations": result["model_evaluations"], "wall_s": payload["wall_s"]}, indent=2))


if __name__ == "__main__":
    main()
