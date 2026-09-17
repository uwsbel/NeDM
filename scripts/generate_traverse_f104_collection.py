#!/usr/bin/env python3
"""Freeze geometry-only PID collection cases on an exact external F104 BMP.

Only this isolated checkout is written. No simulator, neural model, outcome
artifact or procedural terrain/asset generator is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nedm.traverse.terrain import TerrainMap
from nedm.traverse.fdm_diverse_planner import propose_route_families, check_reference_contract
from nedm.traverse.fdm_mppi import MPPIConfig, validate_reference


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    text = json.dumps(obj, indent=2, default=encode, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"Preserve incompatible existing artifact: {path}")
        return
    path.write_text(text)


def footprint(tmap, start, yaw):
    u = np.array([math.cos(yaw), math.sin(yaw)])
    v = np.array([-u[1], u[0]])
    xx, yy = np.meshgrid(np.linspace(-3, 3, 13), np.linspace(-1.5, 1.5, 7))
    xy = start + xx[..., None] * u + yy[..., None] * v
    z = tmap.height(xy[..., 0], xy[..., 1])
    grade = np.degrees(np.arctan(tmap.slope(xy[..., 0], xy[..., 1])))
    plane = np.linalg.lstsq(np.c_[xx.ravel(), yy.ravel(), np.ones(xx.size)], z.ravel(), rcond=None)[0]
    residual = abs(z - plane[0] * xx - plane[1] * yy - plane[2])
    return {"height_range_m": float(np.ptp(z)), "max_grade_deg": float(grade.max()),
            "fitted_grade_deg": float(np.degrees(np.arctan(np.linalg.norm(plane[:2])))),
            "plane_residual_max_m": float(residual.max()), "sample_count": int(z.size),
            "rectangle_m": [6, 3], "method": "Bilinear quantized BMP; native settling pending"}


def start_ok(stats):
    return (stats["height_range_m"] <= .65 and stats["max_grade_deg"] <= 12
            and stats["fitted_grade_deg"] <= 6 and stats["plane_residual_max_m"] <= .20)


def speed_profile(station, cruise, name):
    length = station[-1]
    if name == "smooth_2_6_2":
        x = station / length
        # Spatial ramps are deterministic PID speed references, not random controls.
        ramp = np.clip((x - .12) / .25, 0, 1)
        fall = np.clip((x - .58) / .25, 0, 1)
        smooth = lambda z: z * z * (3 - 2 * z)
        speed = 2 + 4 * smooth(ramp) - 4 * smooth(fall)
    else:
        speed = np.full(len(station), cruise)
    return np.minimum(speed, np.sqrt(4 * np.maximum(length - station, 0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path("/home/harry/NeDM-mppi-claude/assets/traverse/arena_f104"))
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/traverse/fdm_f104_50h_20260909")
    ap.add_argument("--arena", type=Path, default=ROOT / "assets/traverse/arena_f104_50h_v1")
    ap.add_argument("--groups", type=int, default=500)
    ap.add_argument("--pilot-groups", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260909104)
    ap.add_argument("--horizon-s", type=float, default=120.)
    args = ap.parse_args()
    if not 0 < args.horizon_s <= 120 or args.groups < 1:
        raise ValueError("Positive group count and horizon <=120 s required")
    args.out = args.out.resolve(); args.arena = args.arena.resolve()
    for p in (args.out, args.arena):
        p.relative_to(ROOT)
    source_meta = json.loads((args.source / "arena_meta.json").read_text())
    assert source_meta["size_m"] == 80 and source_meta["height_min_m"] == -1.8 and source_meta["height_max_m"] == 3.9
    args.arena.mkdir(parents=True, exist_ok=True)
    source_bmp = args.source / "arena_000.bmp"; copied = args.arena / "arena_000.bmp"
    if copied.exists():
        assert sha(copied) == sha(source_bmp)
    else:
        shutil.copyfile(source_bmp, copied)
    metadata = dict(source_meta)
    metadata["source_provenance"] = {"source_bmp": str(source_bmp), "source_bmp_sha256": sha(source_bmp),
        "source_meta_sha256": sha(args.source / "arena_meta.json"), "bmp_byte_exact": True,
        "terrain_only_no_added_assets": True, "orientation_scope": "Inherited calibrated transform; F104 native audit required before collection"}
    dump(args.arena / "arena_meta.json", metadata)
    tmap = TerrainMap.from_dir(args.arena)
    features = tmap.features
    # The map has a calibrated image orientation: tmap.features transforms the
    # authoring coordinates. Never aim at source_meta.features directly.
    coords = np.arange(-34., 34.01, 1.)
    gx, gy = np.meshgrid(coords, coords)
    candidates = np.c_[gx.ravel(), gy.ravel()]
    candidates = candidates[tmap.slope(candidates[:, 0], candidates[:, 1]) < math.tan(math.radians(7))]
    rng = np.random.default_rng(args.seed)
    selected = []; seen = set(); attempts = 0; last_accepted_attempt = 0
    max_attempts = max(20000, args.groups * 1000)
    while len(selected) < args.groups and attempts < max_attempts:
        attempts += 1
        index = len(selected)
        # Repeated round-robin strata ensure that a quick pilot already spans
        # terrain approaches and that rare feature types are not drowned out.
        feature_index = (index + (attempts - last_accepted_attempt - 1) // 1000) % (len(features) + 2)
        start = candidates[int(rng.integers(len(candidates)))].copy()
        start += rng.uniform(-.35, .35, 2)
        if feature_index < len(features):
            feature = features[feature_index]
            center = np.array([feature["x_m"], feature["y_m"]])
            delta = center - start; distance = np.linalg.norm(delta)
            if not 9 <= distance <= 36:
                continue
            direction = delta / distance
            # Alternate central traversals and flanking cross-slope approaches.
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
            check = validate_reference(route, [], MPPIConfig(arena_half_extent_m=36., max_speed_mps=6., max_curvature_inv_m=.10), [*start, yaw])
            checks.append(check)
        if not all(c["valid"] for c in checks):
            continue
        seen.add(key)
        # Paired offsets and speed profiles remain in one immutable split group.
        cell = [int(math.floor(start[0] / 8)), int(math.floor(start[1] / 8))]
        split_key = int.from_bytes(hashlib.sha256(json.dumps(key).encode()).digest()[:4], "big") % 100
        split = "test" if split_key < 5 else "val" if split_key < 10 else "train"
        sid = f"f104_v1_group_{index:04d}"
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
            "goal_xy": goal, "goal_radius_m": 2.5, "horizon_s": args.horizon_s, "arena_half_extent_m": 40.,
            "settle_reference": routes[0], "family_parameters": {"speeds": [2., 4., 6.], "offsets": [0., -4., 4.],
                "speed_profiles": ["constant_2", "constant_4", "constant_6", "smooth_2_6_2"]},
            "collection_contract": {"maximum_duration_s": args.horizon_s, "record_sustained_failure_until_horizon": False,
                "minimum_recovery_tail_s": 8., "stop_policy_owned_by_runner": True,
                "count_actual_completed_task_seconds_only": True, "exclude_settle_and_failed_jobs_from_hours": True},
            "geometry_initialization": stats, "role": "Terrain-only fixed-map data enrichment; no physics outcome filtered",
            "split_scope": "Deterministic start/goal group split90/5/5; all12paired references together. Same BMP across splits, not unseen-terrain generalization."}
        case_path = args.out / "cases" / f"{sid}.json"; dump(case_path, case)
        rpaths = []
        for ri, route in enumerate(routes):
            rp = args.out / "cases" / "routes" / sid / f"route_{ri:02d}.json"; dump(rp, route)
            rpaths.append(str(rp.relative_to(args.out / "cases")))
        record = {"scene_id": sid, "split": split, "family": "f104", "evaluation_stratum": stratum,
            "case": case_path.name, "routes": rpaths, "case_sha256": sha(case_path),
            "route_sha256": [sha(args.out / "cases" / r) for r in rpaths], "arena": case["arena"],
            "arena_bmp_sha256": sha(copied), "arena_meta_sha256": sha(args.arena / "arena_meta.json"),
            "geometry_initialization": stats, "feature_index": route_meta["feature_index"],
            "route_lengths_m": [float(r["stations"][-1]) for r in routes], "reference_geometry_checks": checks,
            "split_group_start_cell_8m": cell, "horizon_s": args.horizon_s}
        selected.append(record)
        last_accepted_attempt = attempts
        if len(selected) == min(args.groups, args.pilot_groups):
            dump(args.out / "cases" / "pilot_manifest.json", {"schema": 1, "campaign": "fdm_f104_50h_20260909",
                "records": selected.copy(), "geometry_only": True, "no_route_outcomes_inspected": True})
            print(f"Pilot ready: {len(selected)} groups / {len(selected)*12} routes", flush=True)
    if len(selected) != args.groups:
        raise RuntimeError(f"Only {len(selected)} groups after {attempts} attempts; preserve pilot, do not relax gates silently")
    manifest = {"schema": 1, "campaign": "fdm_f104_50h_20260909", "records": selected,
        "geometry_only": True, "no_route_outcomes_inspected": True, "actual_simulation_target_s": 180000,
        "potential_duration_s": len(selected) * 12 * args.horizon_s}
    dump(args.out / "cases" / "cases.json", manifest)
    summary = {"source_bmp_sha256": sha(source_bmp), "copied_bmp_sha256": sha(copied),
        "source_meta_sha256": sha(args.source / "arena_meta.json"), "generator_sha256": sha(__file__),
        "groups": len(selected), "episodes": len(selected) * 12, "attempts": attempts,
        "stratum_schedule": "Round-robin target; after1000 unsuccessful geometry attempts advance target without relaxing geometry gates",
        "maximum_episode_s": args.horizon_s, "potential_hours": manifest["potential_duration_s"] / 3600,
        "required_actual_hours": 50, "actual_hours_are_not_yet_collected": True,
        "mean_actual_seconds_needed_per_episode": 180000 / (len(selected) * 12),
        "stratum_group_counts": {s: sum(r["evaluation_stratum"] == s for r in selected) for s in sorted({r["evaluation_stratum"] for r in selected})},
        "split_episode_counts": {s: 12 * sum(r["split"] == s for r in selected) for s in ["train", "val", "test"]},
        "start_gates": {"footprint_m": [6, 3], "height_range_m": .65, "max_grade_deg": 12,
            "fitted_grade_deg": 6, "plane_residual_max_m": .20, "require_native_settle_gate": True},
        "feature_world_coordinates": features, "no_added_assets": True,
        "orientation": metadata["orientation"], "native_f104_orientation_audit_pending": True,
        "limits": ["Geometry-feasible is not physically traversable", "Same terrain across splits", "Post-failure dwell is bounded by runner", "No guaranteed recovery or failure label before physics"]}
    dump(args.out / "design" / "collection_design.json", summary)
    print(json.dumps({k: summary[k] for k in ["groups", "episodes", "potential_hours", "attempts"]}), flush=True)


if __name__ == "__main__":
    main()
