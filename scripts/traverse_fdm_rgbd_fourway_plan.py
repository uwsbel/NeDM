#!/usr/bin/env python3
"""Freeze a four-route model diagnostic before reading any physical outcomes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nedm.traverse.fdm_rgbd_model import load_rgbd_checkpoint
from nedm.traverse.fdm_rgbd_planner import RGBDReferenceScorer, RGBDCostConfig, propose_route_families
from nedm.traverse.fdm_rgbd_diagnostics import RGBDScoringRecorder, route_digest, serializable
from nedm.traverse.fdm_mppi import MPPIConfig, ReferenceMPPI, validate_reference
from traverse_fdm_rgbd_chrono import dump, read_route, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--stall-family-index", type=int, choices=(0, 1), default=1,
                        help="Predeclared straight stall-test command: family 0 is 4 m/s; family 1 is 6 m/s")
    parser.add_argument("--base-references", type=Path,
                        help="Optional JSON list of six original reference files, ordered offsets 0/-8/+8, speeds 4/6")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Refusing to replace a frozen prediction set")
    with np.load(args.observation) as file:
        obs = {key: file[key].copy() for key in file.files}
    model, _ = load_rgbd_checkpoint(args.checkpoint, device="cpu")
    cost = RGBDCostConfig(goal_radius_m=float(obs["goal_radius_m"]))
    config = MPPIConfig(samples=args.samples, iterations=args.iterations, max_speed_mps=6.)
    scorer = RGBDReferenceScorer(model, obs["rgbd"], obs["history"], obs["pose"], obs["goal_xy"],
                                elapsed_s=float(obs["elapsed_s"]), cost_config=cost)
    provenance = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "observation": str(args.observation.resolve()), "observation_sha256": sha256(args.observation),
        "planning_script_sha256": sha256(__file__), "seed": args.seed,
        "stall_test_family_index": args.stall_family_index,
        "stall_test_command_speed_mps": (4., 6.)[args.stall_family_index],
        "model_input_boundary": "Only saved current RGB-D, measured history/pose, goal and proposed geometric references",
        "future_physical_outcomes_read": False, "mppi_config": asdict(config),
        "proposal_scope": "Same six original focused-data route/speed families, plus bounded MPPI refinements; speed<=6m/s. Bounds are not a guarantee of learned support."}
    if args.base_references:
        entries = json.loads(args.base_references.read_text())
        if not isinstance(entries, list) or len(entries) != 6:
            raise ValueError("Exactly six ordered command references are required")
        paths = [(args.base_references.parent / entry).resolve() for entry in entries]
        families = [read_route(path) for path in paths]
        for index, route in enumerate(families):
            for key in ("waypoints", "speeds", "stations", "headings"):
                route[key] = np.asarray(route[key], dtype=float)
            cruise = (4., 6.)[index % 2]
            remaining = route["stations"][-1] - route["stations"]
            expected_speed = np.minimum(cruise, np.sqrt(4. * np.maximum(remaining, 0.)))
            if not np.allclose(route["speeds"], expected_speed):
                raise ValueError("Expected alternating 4/6 m/s cruising commands with the original goal deceleration")
            if not np.isclose(route["meta"]["lateral_offset_m"], (0., -8., 8.)[index // 2]):
                raise ValueError("Unexpected geometric-family ordering")
        provenance["base_reference_manifest"] = str(args.base_references.resolve())
        provenance["base_reference_manifest_sha256"] = sha256(args.base_references)
        provenance["base_reference_sha256"] = [sha256(path) for path in paths]
        provenance["reference_origin"] = "Reused original geometric commands, scored from measured current pose; no outcome or terrain fields loaded"
    else:
        families = propose_route_families(obs["pose"], obs["goal_xy"], speeds=(4., 6.), offsets=(0., -8., 8.))
    recorder = RGBDScoringRecorder(scorer, provenance=provenance)
    assert all(validate_reference(route, [], config, obs["pose"])["valid"] for route in families)
    start = time.perf_counter()
    costs = recorder(families)
    finite = [int(i) for i in np.argsort(costs) if np.isfinite(costs[i])]
    if not finite:
        raise RuntimeError("Model abstains on every base family; no predicted-best route can be fabricated")
    best_route, best_cost = families[finite[0]], float(costs[finite[0]])
    refinements, geometries = [], set()
    for index in finite:
        geometry = np.asarray(families[index]["waypoints"]).tobytes()
        if geometry in geometries:
            continue
        geometries.add(geometry)
        result = ReferenceMPPI(config, args.seed+len(refinements)).optimize(families[index], obs["pose"], [], recorder)
        result["base_family_index"] = index
        refinements.append(result)
        if result.get("route") is not None and result["cost"] < best_cost:
            best_route, best_cost = result["route"], float(result["cost"])
        if len(refinements) == cost.refine_families:
            break
    specifications = [
        ("01_selected", "best", "MPPI selected", "#087f5b", best_route,
         "Lowest model cost among the scored, allowed candidates; physical optimum is not known"),
        ("02_collision", "collision", "Rock-crossing test", "#d23b3b", families[3],
         "Scene-designed collision hypothesis; actual predicted contact probability is reported without alteration"),
        ("03_stall", "stall", "Terrain-stall test", "#d97706", families[args.stall_family_index],
         "Scene-designed terrain-stall hypothesis; a later stall may lie outside the4s learned event horizon"),
        ("04_progress", "low_progress", "Slower clear route", "#7441b5", families[4],
         "Lower progress means less goal progress at the same4s horizon; a4m/s command may still reach the goal later"),
    ]
    records = []
    for name, category, label, color, route, explanation in specifications:
        identity = route_digest(route)
        route_path = args.out / "routes" / (name+".json")
        dump(route_path, route)
        forecast = recorder.records[identity]
        records.append({"id": name, "category": category, "label": label, "color": color,
            "candidate_id": identity, "reference_json": str(route_path.resolve()), "role_semantics": explanation,
            "route_sha256": sha256(route_path), "model_prediction": {
                "goal_progress_4s_m": forecast["predicted_4s_goal_progress_m"],
                "contact_probability_4s_prefix_max": forecast["scoring"]["contact_probability"],
                "bounded_motion_probability_4s_prefix_max": forecast["scoring"]["low_progress_probability"],
                "cost": forecast["scoring"]["cost"],
                "estimated_time_to_goal_s": forecast["scoring"]["estimated_time_to_goal_s"]}})
    actual_runtime = time.perf_counter()-start
    dump(args.out / "selection.json", {"provenance": provenance, "routes": records,
        "selected_cost": best_cost, "planning_wall_s": actual_runtime,
        "unique_scored_candidates": len(recorder.records), "forecast_horizon_s": 4.,
        "prediction_only_risk_categories": recorder.representative_candidates(),
        "refinements": serializable(refinements),
        "display": "Solid=model forecast0-4s; dashed=full commanded path; role names are test hypotheses, not overwritten NN classifications"})
    recorder.export(args.out / "candidates.json", representatives={"routes": records})
    dump(args.out / "frozen_artifacts.json", {str(p.relative_to(args.out)): sha256(p)
        for p in sorted(args.out.rglob("*")) if p.is_file()})
    for row in records:
        print(row["id"], row["model_prediction"])
    print("unique_scored_candidates", len(recorder.records), "planning_wall_s", round(actual_runtime, 3))


if __name__ == "__main__":
    main()
