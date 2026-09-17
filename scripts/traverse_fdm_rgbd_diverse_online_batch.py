#!/usr/bin/env python3
"""Run a predeclared online evaluation cohort in bounded AMD workers."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def execute(task, root, python, chrono_data):
    out = root / task["id"]
    log = root / (task["id"]+".log")
    args = [python, "-u", str(ROOT / "scripts/traverse_fdm_rgbd_diverse_online.py"),
            "--out", str(out), "--chrono-data", chrono_data]
    for key, value in task["arguments"].items():
        if key not in {"case", "checkpoint", "scene-observation", "horizon-s", "planning-device", "planning-mode",
                       "replan-period-s", "cost-mode", "cost-config", "mppi-samples", "mppi-iterations",
                       "seed", "speeds", "offsets", "image-intervention", "video-fps"}:
            raise ValueError(f"Unknown declared task argument {key}")
        args.append("--"+key)
        args.extend(map(str, value if isinstance(value, list) else [value]))
    start = time.time()
    with log.open("x") as handle:
        completed = subprocess.run(args, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
    result = {"id": task["id"], "exit_code": completed.returncode, "wall_s": time.time()-start,
              "log_sha256": sha(log), "out": str(out), "command": args}
    if completed.returncode == 0:
        required = ("outcome.json", "online_protocol.json", "online_planning_summary.json",
                    "anchor_equality.json", "trajectory.npz", "rich_telemetry.npz", "rich_intervals.npz")
        result["sha256"] = {name: sha(out / name) for name in required}
        outcome = json.loads((out / "outcome.json").read_text())
        result["outcome"] = {k: outcome[k] for k in (
            "status", "schema_safe_goal_reached", "goal_time_s", "elapsed_s", "goal_progress_m",
            "positive_work_kj", "schema_contact", "schema_rollover", "bounded_blockage_v1",
            "max_abs_roll_deg", "max_abs_pitch_deg", "planning_decisions", "planning_abstentions", "planning_wall_s")}
    dump(root / (task["id"]+".result.json"), result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--chrono-data", required=True)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        p.error("Run physical online cohorts in an AMD Slurm allocation")
    tasks = json.loads(a.tasks.read_text())["tasks"]
    names = [t["id"] for t in tasks]
    if not tasks or len(set(names)) != len(names) or any(not n.replace("_", "").isalnum() for n in names):
        p.error("Require unique plain task identifiers")
    if a.workers < 1:
        p.error("Workers must be positive")
    a.out.mkdir(parents=True, exist_ok=False)
    dump(a.out / "declaration.json", {"tasks_sha256": sha(a.tasks), "tasks": tasks,
        "workers": a.workers, "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "source_manifest_sha256": sha(ROOT / "source_manifest.json"),
        "batch_script_sha256": sha(__file__), "started_unix": time.time()})
    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending = {pool.submit(execute, task, a.out, a.python, a.chrono_data): task for task in tasks}
        for future in as_completed(pending):
            task = pending[future]
            try:
                result = future.result()
            except Exception as error:
                result = {"id": task["id"], "exit_code": -1, "error": repr(error)}
                dump(a.out / (task["id"]+".result.json"), result)
            results.append(result)
            print(json.dumps({k: result[k] for k in ("id", "exit_code")}), flush=True)
    failed = [r["id"] for r in results if r["exit_code"]]
    dump(a.out / "batch_result.json", {"declared": len(tasks), "finished": len(results),
        "failures": failed, "results": sorted(results, key=lambda r: r["id"]), "finished_unix": time.time()})
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
