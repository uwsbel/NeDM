"""Replay optimised launches in Chrono (independent physical check).

For every target: the launch chosen by each optimiser arm, every distinct
converged branch, the data-only baselines, and the target's own source launch
(a control that measures how well Chrono reproduces its own recording; the
cohort was recorded on other nodes, so this is cross-node reproducibility). For the first `video_targets` targets, the launches saved every few
optimiser iterations are replayed too, for the iteration video. All replays of
one run happen in one job on one node.
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

from nedm.pool_ball.campaign import atomic_json
from nedm.pool_ball.physics import simulate_episode


def replay(payload):
    key, config, launch, t_target = payload
    arrays, meta = simulate_episode(config, float(launch[0]), float(launch[1]))
    record = config["simulation"]["record_step_s"]
    k = round(t_target / record)
    states = arrays["states"]
    events = [(e["pair"], e["time_s"]) for e in meta["impacts"]]
    return key, {"b_xy_at_t": states[k, 1, :2].tolist(), "a_xy_at_t": states[k, 0, :2].tolist(),
                 "termination": meta["termination"], "ab_contacts_before_t": sum(1 for p, t in events if p == "AB" and t < t_target),
                 "b_cushions_before_t": sum(1 for p, t in events if p.startswith("B_") and t < t_target),
                 "trace_10ms": states[: k + 1 : 10, :, :2].tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    parser.add_argument("--video-targets", type=int, default=10)
    parser.add_argument("--branches", action="store_true")
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
            if args.branches:
                for b, branch in enumerate(case["branches"]):
                    jobs[(f"{method}_branch{b}", i)] = branch["launch_mps"]
            if i < args.video_targets:
                for snap in result["snapshots"]:
                    jobs[(f"{method}_iter{snap['iteration']:03d}", i)] = snap["launch_mps"][i]
    # Launches outside the collected speed range are still physical; replay all.
    unique = {}
    for key, launch in jobs.items():
        unique.setdefault(tuple(np.round(launch, 12)), []).append(key)
    payloads = [(n, config, list(launch), t) for n, launch in enumerate(unique)]
    with ProcessPoolExecutor(args.workers) as pool:
        results = dict(pool.map(replay, payloads, chunksize=1))
    by_key = {}
    for n, (launch, keys) in enumerate(unique.items()):
        for key in keys:
            by_key[key] = results[n]
    rows = []
    for i, target in enumerate(report["targets"]):
        row = {"target": i, "xy_m": target["xy_m"], "b_cushions_before_t": target["b_cushions_before_t"]}
        for name in ["source", "nearest", "local_linear", *report["results"]]:
            r = by_key[(name, i)]
            row[name] = {"miss_m": math.dist(r["b_xy_at_t"], target["xy_m"]), "termination": r["termination"],
                         "ab_contacts_before_t": r["ab_contacts_before_t"], "b_cushions_before_t": r["b_cushions_before_t"]}
        if args.branches:
            for method, result in report["results"].items():
                row[f"{method}_branches"] = [math.dist(by_key[(f"{method}_branch{b}", i)]["b_xy_at_t"], target["xy_m"])
                                             for b in range(len(result["cases"][i]["branches"]))]
        rows.append(row)
    summary = {}
    for name in ["source", "nearest", "local_linear", *report["results"]]:
        misses = np.array([r[name]["miss_m"] for r in rows])
        summary[name] = {"median_mm": float(1e3 * np.median(misses)), "p90_mm": float(1e3 * np.quantile(misses, 0.9)),
                         "max_mm": float(1e3 * misses.max()), "within_10mm": int((misses <= 0.01).sum()),
                         "within_5mm": int((misses <= 0.005).sum()), "count": len(misses)}
    video = {}
    for (name, i), r in by_key.items():
        if "_iter" in name and i < args.video_targets:
            video.setdefault(name.split("_iter")[0], {}).setdefault(str(i), {})[name.split("_iter")[1]] = {
                "miss_m": math.dist(r["b_xy_at_t"], report["targets"][i]["xy_m"]), "trace_10ms": r["trace_10ms"]}
    for name in ["source", *report["results"]]:
        for i in range(min(args.video_targets, len(report["targets"]))):
            video.setdefault(f"final_{name}", {})[str(i)] = by_key[(name, i)]["trace_10ms"]
    atomic_json(args.run / "chrono_verification.json", {"summary": summary, "targets": rows, "host": platform.node(),
                                                        "job_id": os.environ.get("SLURM_JOB_ID"), "replays": len(payloads),
                                                        "simulator_feedback_used_by_optimiser": False})
    (args.run / "chrono_video_traces.json").write_text(json.dumps(video))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
