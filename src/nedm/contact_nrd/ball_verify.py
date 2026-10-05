"""Replay bouncing-ball launches chosen through the unified NRD in Chrono.

For every target: the launch chosen by each optimiser arm, every distinct
converged branch, both data-only baselines, and the target's own source
launch (control). The ball's (x, z) at t is read off the 0.5 ms record by
linear interpolation, and the Chrono episode must show ground -> wall before t.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nedm.bouncing_ball.collection import simulate_episode


def replay(payload):
    key, config, launch, t = payload
    rows, meta = simulate_episode(config, float(launch[0]), float(launch[1]), episode_id=f"replay_{key}", split="test")
    times = np.array([r["time_s"] for r in rows])
    events = {e["kind"]: e["time_s"] for e in meta["events"]}
    ok = (events.get("first_ground_contact", 9) < events.get("wall_contact", 9) < t
          and times[-1] >= t and meta["termination"] in ("next_ground_contact", "simulation_timeout"))
    xz = [float(np.interp(t, times, [r[k] for r in rows])) for k in ("x_m", "z_m")]
    trace = [[float(np.interp(s, times, [r[k] for r in rows])) for k in ("x_m", "z_m")] for s in np.arange(0, t + 1e-9, 0.01)]
    return key, {"xz_at_t": xz, "ground_then_wall_before_t": bool(ok), "termination": meta["termination"], "trace_10ms": trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    parser.add_argument("--video-targets", type=int, default=10)
    args = parser.parse_args()
    report = json.loads((args.run / "optimization.json").read_text())
    config, t = report["physics_config"], report["target_time_s"]
    jobs = {}
    for i, target in enumerate(report["targets"]):
        jobs[("source", i)] = target["source_launch_mps"]
        jobs[("nearest", i)] = report["baselines"][i]["nearest_launch_mps"]
        jobs[("local_linear", i)] = report["baselines"][i]["local_linear_launch_mps"]
        for method, result in report["results"].items():
            case = result["cases"][i]
            jobs[(method, i)] = case["launch_mps"]
            for b, branch in enumerate(case["branches"]):
                jobs[(f"{method}_branch{b}", i)] = branch["launch_mps"]
            if i < args.video_targets:
                for snap in result["snapshots"]:
                    jobs[(f"{method}_iter{snap['iteration']:03d}", i)] = snap["launch_mps"][i]
    unique = {}
    for key, launch in jobs.items():
        unique.setdefault(tuple(np.round(launch, 12)), (launch, []))[1].append(key)
    payloads = [(n, config, launch, t) for n, (launch, _) in enumerate(unique.values())]
    with ProcessPoolExecutor(args.workers) as pool:
        results = dict(pool.map(replay, payloads, chunksize=1))
    by_key = {}
    for n, (_, keys) in enumerate(unique.values()):
        for key in keys:
            by_key[key] = results[n]
    rows = []
    names = ["source", "nearest", "local_linear", *report["results"]]
    for i, target in enumerate(report["targets"]):
        row = {"target": i, "xz_m": target["xy_m"]}
        for name in names:
            r = by_key[(name, i)]
            row[name] = {"miss_m": math.dist(r["xz_at_t"], target["xy_m"]), "ground_then_wall": r["ground_then_wall_before_t"]}
        for method, result in report["results"].items():
            row[f"{method}_branches"] = [math.dist(by_key[(f"{method}_branch{b}", i)]["xz_at_t"], target["xy_m"])
                                         for b in range(len(result["cases"][i]["branches"]))]
        rows.append(row)
    summary = {}
    for name in names:
        m = np.array([r[name]["miss_m"] for r in rows])
        summary[name] = {"median_mm": float(1e3 * np.median(m)), "p90_mm": float(1e3 * np.quantile(m, 0.9)),
                         "max_mm": float(1e3 * m.max()), "within_10mm": int((m <= 0.01).sum()), "within_5mm": int((m <= 0.005).sum()),
                         "within_3mm": int((m <= 0.003).sum()), "count": len(m),
                         "ground_then_wall": int(sum(r[name]["ground_then_wall"] for r in rows))}
    video = {}
    for (name, i), r in by_key.items():
        if "_iter" in name:
            video.setdefault(name.split("_iter")[0], {}).setdefault(str(i), {})[name.split("_iter")[1]] = {
                "miss_m": math.dist(r["xz_at_t"], report["targets"][i]["xy_m"]), "trace_10ms": r["trace_10ms"]}
    (args.run / "chrono_verification.json").write_text(json.dumps({"summary": summary, "targets": rows, "host": platform.node(),
        "job_id": os.environ.get("SLURM_JOB_ID"), "replays": len(payloads), "simulator_feedback_used_by_optimiser": False}, indent=1))
    (args.run / "chrono_video_traces.json").write_text(json.dumps(video))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
