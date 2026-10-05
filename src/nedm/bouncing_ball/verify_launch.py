"""Independently replay initial and optimized NRD launches in physical Chrono."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
from pathlib import Path

import numpy as np

from nedm.bouncing_ball.collection import CSV_FIELDS, STATE_FIELDS, atomic_json, provenance, simulate_episode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    result = json.loads((args.run/"optimization.json").read_text())
    folder = args.run/"chrono"
    folder.mkdir(exist_ok=False)
    terminal = result["config"]["terminal_time_s"]
    config = result["physics_config"]
    cache, trajectories, reports = {}, [], []
    for case in result["cases"]:
        endpoints, traces = {}, {}
        for role in ("initial", "optimized"):
            velocity = case[f"{role}_velocity_mps"]
            key = tuple(velocity)
            if key not in cache:
                name = f"launch_{len(cache):03d}"
                rows, metadata = simulate_episode(config, *velocity, episode_id=name, split="test")
                path = folder/f"{name}.csv"
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                metadata["csv_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                metadata["csv_path"] = str(path.relative_to(args.run))
                atomic_json(path.with_suffix(".json"), metadata)
                cache[key] = (rows, metadata)
            rows, metadata = cache[key]
            times = np.array([r["time_s"] for r in rows])
            events = {e["kind"]: e["time_s"] for e in metadata["events"]}
            valid = (metadata["accepted"] and events.get("wall_release", float("inf")) < terminal
                     and terminal < events.get("next_ground_contact", 0)-.02
                     and times[-1] >= terminal)
            if valid:
                endpoint = [float(np.interp(terminal, times, [r[k] for r in rows])) for k in STATE_FIELDS[:2]]
                distance = math.dist(endpoint, case["xz_m"])
            else:
                endpoint, distance = None, None
            endpoints[role] = {"endpoint_m": endpoint, "target_distance_m": distance,
                               "two_impacts_at_endpoint": bool(valid), "metadata": metadata}
            traces[role] = (times, np.array([[r[k] for k in STATE_FIELDS] for r in rows]))
        optimized = endpoints["optimized"]
        model_gap = (math.dist(optimized["endpoint_m"], case["optimized_nrd_endpoint_m"])
                     if optimized["endpoint_m"] else None)
        passed = (endpoints["initial"]["two_impacts_at_endpoint"] and optimized["two_impacts_at_endpoint"] and case["nrd_two_impacts"]
                  and case["optimized_nrd_distance_m"] <= result["config"]["nrd_tolerance_m"]
                  and optimized["target_distance_m"] <= result["config"]["chrono_tolerance_m"]
                  and optimized["target_distance_m"] < endpoints["initial"]["target_distance_m"])
        reports.append({"name": case["name"], "target_xz_m": case["xz_m"], "passed": bool(passed),
                        "initial": endpoints["initial"], "optimized": optimized,
                        "nrd_chrono_endpoint_gap_m": model_gap})
        grid = np.arange(round(terminal/.01)+1)*.01
        trajectories.append(np.stack([np.stack([np.interp(grid, trace[0], trace[1][:, k])
                                                for k in range(5)], -1) for trace in traces.values()]))
    atomic_json(args.run/"chrono_verification.json", {
        "passed": all(r["passed"] for r in reports), "cases": reports, "host": platform.node(),
        "job_id": os.environ.get("SLURM_JOB_ID"), "runtime": provenance(),
        "unique_physical_launches": len(cache), "simulator_feedback_used_by_optimizer": False,
        "scope": "Physical replay after NRD-only optimization; no Chrono corrections or simulator gradients"})
    np.savez_compressed(args.run/"chrono_trajectories.npz", states=np.stack(trajectories), time_s=grid)
    print(json.dumps({"chrono_passed": all(r["passed"] for r in reports),
                      "distances_m": [{"name": r["name"], "before": r["initial"]["target_distance_m"],
                                       "after": r["optimized"]["target_distance_m"]} for r in reports]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
