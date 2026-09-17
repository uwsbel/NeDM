#!/usr/bin/env python3
"""Predeclare scene-grouped sibling-route collection, without outcome selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from nedm.traverse.fdm_mppi import MPPIConfig, validate_reference
from nedm.traverse.fdm_rgbd_planner import check_reference_contract, propose_route_families


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else float(x), allow_nan=False)+"\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026090807)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Refusing to replace predeclared case cohort")
    rng = np.random.default_rng(args.seed)
    records = []
    # Balanced lateral placements prevent reference-index alone determining risk.
    for split, count in (("train", 16), ("val", 4)):
        for number in range(count):
            scene_id = f"focus_obstacle_{split}_{number:03d}"
            yaw = float(rng.uniform(-math.pi, math.pi))
            direction = np.array([math.cos(yaw), math.sin(yaw)])
            normal = np.array([-direction[1], direction[0]])
            center = rng.uniform(-9., 9., 2)
            start, goal = center-18.*direction, center+18.*direction
            ahead = float(rng.uniform(6., 9.))
            lateral = float((0., 0., -3.2, 3.2)[number % 4] + rng.uniform(-.35, .35))
            rock_xy = start+ahead*direction+lateral*normal
            # Keep this cohort distinct from the already inspected frozen demo.
            if np.linalg.norm(rock_xy-np.array([-10., -28.])) < 2.:
                center += np.array([3., 3.])
                start, goal = center-18.*direction, center+18.*direction
                rock_xy = start+ahead*direction+lateral*normal
            edge = float(rng.uniform(1.6, 2.7))
            rock = {"kind": "rock", "x_m": float(rock_xy[0]), "y_m": float(rock_xy[1]),
                    "yaw_rad": float(rng.uniform(-math.pi, math.pi)),
                    "footprint_radius_m": edge/math.sqrt(2.),
                    "dims": {"edge_m": edge, "height_m": float(rng.uniform(1.15, 1.9))}}
            pose = [*start, yaw]
            family_parameters = {"speeds": [4., 6.], "offsets": [0., -8., 8.]}
            routes = propose_route_families(pose, goal, **family_parameters)
            route_paths = []
            for j, route in enumerate(routes):
                check_reference_contract(route)
                check = validate_reference(route, [], MPPIConfig(), pose)
                if not check["valid"]:
                    raise ValueError((scene_id, j, check))
                if np.max(np.abs(route["waypoints"])) > 36.:
                    raise ValueError("Reference violates declared arena boundary margin")
                rel = Path("routes")/scene_id/f"family_{j:02d}.json"
                dump(args.out/rel, route)
                route_paths.append(str(rel))
            case = {"id": scene_id, "split": split, "scene_id": scene_id,
                    "arena": "assets/traverse/arena_v1", "horizon_s": 10., "goal_radius_m": 2.5,
                    "family_parameters": family_parameters, "goal_xy": goal,
                    "layout": {"episode_id": scene_id, "seed": args.seed+len(records), "assets": [rock],
                               "house_xy": goal, "house_yaw": yaw, "start_xy": start, "start_yaw": yaw},
                    "settle_reference": routes[0], "source_domain": "focused_v1_standard_chrono_pid_vulkan",
                    "role": "predeclared focused training" if split == "train" else "predeclared grouped development validation",
                    "construction": {"rock_ahead_m": ahead, "rock_lateral_m": lateral,
                                     "no_terrain_or_outcome_filter": True, "routes": route_paths},
                    "exclusion": "Fresh scene; neither visible_rock_v1 nor hill0_f111_near; no protected-test layouts read."}
            rel = Path(f"{scene_id}.json")
            dump(args.out/rel, case)
            records.append({"scene_id": scene_id, "split": split, "case": str(rel), "routes": route_paths,
                            "case_sha256": hashlib.sha256((args.out/rel).read_bytes()).hexdigest(),
                            "route_sha256": [hashlib.sha256((args.out/p).read_bytes()).hexdigest() for p in route_paths]})
    dump(args.out/"cases.json", {"schema": 1, "cohort": "focused_v1_obstacle_initial", "seed": args.seed,
        "cases": [r["case"] for r in records], "records": records,
        "scene_counts": {"train": 16, "val": 4}, "route_counts": {"train": 96, "val": 24},
        "split_unit": "scene identity; every sibling route remains in the same split",
        "selection": "Predeclared without model scores or physical outcomes; geometry-only curvature/bounds checks",
        "cohort_state": "Obstacle construction cohort frozen; physical outcomes not yet available; terrain cohort declared separately",
        "validation_scope": "Designed failure-focused development distribution, not natural deployment prevalence or untouched test"})
    print(json.dumps({"out": str(args.out), "scenes": len(records), "routes": 6*len(records)}))


if __name__ == "__main__":
    main()
