#!/usr/bin/env python3
"""Night-2 wave A1: fresh start/goal groups on the SAME f104 arena, same 12 designed routes per group.

Geometry gates, strata round-robin, route families and split hashing are copied from
scripts/generate_traverse_f104_collection.py (the frozen campaign generator) so the new groups are drawn from
the same distribution as the existing 1,500. Only the id prefix, the seed and the output directory differ, and
start/goal pairs already used by the existing campaign are skipped.
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from nedm.traverse.terrain import TerrainMap
from nedm.traverse.fdm_diverse_planner import propose_route_families, check_reference_contract
from nedm.traverse.fdm_mppi import MPPIConfig, validate_reference
from generate_traverse_f104_collection import footprint, start_ok, speed_profile, sha, dump


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groups", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=20260912104)
    ap.add_argument("--prefix", default="f104_v2_group")
    ap.add_argument("--arena", type=Path, default=ROOT / "assets/traverse/arena_f104_50h_v1")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/traverse/fdm_f104_50h_20260909/cases_night2")
    ap.add_argument("--existing", type=Path, default=ROOT / "artifacts/traverse/fdm_f104_50h_20260909/cases")
    args = ap.parse_args()
    tmap = TerrainMap.from_dir(args.arena); features = tmap.features
    seen = set()
    for p in sorted(args.existing.glob("f104_v1_group_*.json")):
        c = json.loads(p.read_text())
        seen.add(tuple(np.round(np.r_[c["layout"]["start_xy"], c["goal_xy"]], 1)))
    print(f"{len(seen)} existing start/goal pairs will be avoided", flush=True)

    coords = np.arange(-34., 34.01, 1.)
    gx, gy = np.meshgrid(coords, coords)
    candidates = np.c_[gx.ravel(), gy.ravel()]
    candidates = candidates[tmap.slope(candidates[:, 0], candidates[:, 1]) < math.tan(math.radians(7))]
    rng = np.random.default_rng(args.seed)
    selected = []; attempts = 0; last_accepted_attempt = 0
    max_attempts = max(20000, args.groups * 1000)
    while len(selected) < args.groups and attempts < max_attempts:
        attempts += 1
        index = len(selected)
        feature_index = (index + (attempts - last_accepted_attempt - 1) // 1000) % (len(features) + 2)
        start = candidates[int(rng.integers(len(candidates)))].copy() + rng.uniform(-.35, .35, 2)
        if feature_index < len(features):
            feature = features[feature_index]
            center = np.array([feature["x_m"], feature["y_m"]])
            delta = center - start; distance = np.linalg.norm(delta)
            if not 9 <= distance <= 36:
                continue
            direction = delta / distance
            normal = np.array([-direction[1], direction[0]])
            miss = [0., .65, -.65][(index // (len(features) + 2)) % 3] * feature["sigma_m"]
            target = center + miss * normal
            direction = (target - start) / np.linalg.norm(target - start)
            maximum = min((35 - start[k]) / direction[k] if direction[k] > 1e-8 else
                          (-35 - start[k]) / direction[k] if direction[k] < -1e-8 else 1e9 for k in range(2))
            length = min(maximum, np.linalg.norm(target - start) + float(rng.uniform(10, 23)))
            if length < np.linalg.norm(target - start) + 5:
                continue
            goal = start + length * direction
            stratum = feature["kind"] + ("_cross_slope" if miss else "_entry_cross_exit")
        else:
            goal = candidates[int(rng.integers(len(candidates)))].copy()
            delta = goal - start; length = np.linalg.norm(delta)
            if length < 42:
                continue
            direction = delta / length; feature = None; miss = None
            stratum = "long_traverse" if feature_index == len(features) else "roughness_transfer"
        if not 24 <= length <= 85:
            continue
        yaw = float(math.atan2(direction[1], direction[0]))
        stats = footprint(tmap, start, yaw)
        if not start_ok(stats):
            continue
        key = tuple(np.round(np.r_[start, goal], 1))
        if key in seen:
            continue
        base_routes = propose_route_families([*start, yaw], goal, speeds=[2., 4., 6.], offsets=[0., -4., 4.], step_m=.5)
        checks = []
        for route in base_routes:
            check_reference_contract(route)
            checks.append(validate_reference(route, [], MPPIConfig(arena_half_extent_m=36., max_speed_mps=6.,
                                                                  max_curvature_inv_m=.10), [*start, yaw]))
        if not all(c["valid"] for c in checks):
            continue
        seen.add(key)
        split_key = int.from_bytes(hashlib.sha256(json.dumps(key).encode()).digest()[:4], "big") % 100
        split = "test" if split_key < 5 else "val" if split_key < 10 else "train"
        sid = f"{args.prefix}_{index:04d}"
        route_meta = {"scene_id": sid, "collection_stratum": stratum, "feature_index": feature_index if feature else None,
                      "feature_geometry_only": feature, "feature_lateral_miss_m": miss, "pair_group_id": sid,
                      "route_seed": args.seed + index, "minimum_recovery_tail_s": 8.,
                      "recovery_label_scope": "Commanded entry/exit and retained post-blockage attempts; no asserted physical recovery"}
        routes = []
        for offset_index, offset in enumerate([0., -4., 4.]):
            for profile, speed in [("constant_2", 2.), ("constant_4", 4.), ("constant_6", 6.), ("smooth_2_6_2", 6.)]:
                route = dict(base_routes[3 * offset_index])
                route["speeds"] = speed_profile(route["stations"], speed, profile)
                route["meta"] = {**route["meta"], **route_meta, "speed_profile_id": profile,
                                 "cruise_speed_mps": speed, "lateral_offset_m": offset, "route_index": len(routes)}
                routes.append(route)
        case = {"id": sid, "split": split, "arena": str(args.arena.relative_to(ROOT)), "family": "f104",
                "evaluation_stratum": stratum, "layout": {"episode_id": sid, "seed": args.seed + index,
                    "assets": [], "house_xy": goal, "house_yaw": yaw, "start_xy": start, "start_yaw": yaw},
                "goal_xy": goal, "goal_radius_m": 2.5, "horizon_s": 120., "arena_half_extent_m": 40.,
                "settle_reference": routes[0], "family_parameters": {"speeds": [2., 4., 6.], "offsets": [0., -4., 4.],
                    "speed_profiles": ["constant_2", "constant_4", "constant_6", "smooth_2_6_2"]},
                "collection_contract": {"maximum_duration_s": 120., "record_sustained_failure_until_horizon": False,
                    "minimum_recovery_tail_s": 8., "stop_policy_owned_by_runner": True,
                    "count_actual_completed_task_seconds_only": True, "exclude_settle_and_failed_jobs_from_hours": True},
                "geometry_initialization": stats, "wave": "night2_A1_coverage",
                "role": "Terrain-only fixed-map data enrichment; no physics outcome filtered",
                "split_scope": "Deterministic start/goal group split 90/5/5; all 12 paired references together."}
        case_path = args.out / "cases" / f"{sid}.json"; dump(case_path, case)
        rpaths = []
        for ri, route in enumerate(routes):
            rp = args.out / "cases" / "routes" / sid / f"route_{ri:02d}.json"; dump(rp, route)
            rpaths.append(str(rp.relative_to(args.out / "cases")))
        selected.append({"scene_id": sid, "split": split, "family": "f104", "evaluation_stratum": stratum,
                         "case": case_path.name, "routes": rpaths, "case_sha256": sha(case_path),
                         "route_sha256": [sha(args.out / "cases" / r) for r in rpaths], "arena": case["arena"],
                         "route_lengths_m": [float(r["stations"][-1]) for r in routes],
                         "horizon_s": 120., "start_xy": [float(x) for x in start], "goal_xy": [float(x) for x in goal]})
        last_accepted_attempt = attempts
        if len(selected) % 200 == 0:
            print(f"  {len(selected)} groups after {attempts} attempts", flush=True)
    if len(selected) != args.groups:
        raise RuntimeError(f"Only {len(selected)} groups after {attempts} attempts")
    dump(args.out / "cases" / "cases.json", {"schema": 1, "campaign": "fdm_f104_50h_20260909", "wave": "night2_A1",
                                             "records": selected, "seed": args.seed, "prefix": args.prefix})
    print(json.dumps({"groups": len(selected), "episodes": 12 * len(selected), "attempts": attempts,
                      "splits": {s: sum(r["split"] == s for r in selected) for s in ("train", "val", "test")}}), flush=True)


if __name__ == "__main__":
    main()
