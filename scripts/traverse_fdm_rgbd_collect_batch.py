#!/usr/bin/env python3
"""Parallel isolated Chrono processes for a frozen focused RGB-D task manifest."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chrono-data", required=True)
    parser.add_argument("--record-rgbd-stride", type=int, default=20)
    parser.add_argument("--horizon-s", type=float)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest_hash = sha256(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    tasks = []
    for record in manifest["records"]:
        case = args.manifest.parent / record["case"]
        if sha256(case) != record["case_sha256"]:
            raise ValueError(f"Case changed after declaration: {case}")
        for route, expected_hash in zip(record["routes"], record["route_sha256"]):
            path = args.manifest.parent / route
            if sha256(path) != expected_hash:
                raise ValueError(f"Reference changed after declaration: {path}")
            tasks.append((record["scene_id"], case, path))
    if args.limit:
        tasks = tasks[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("SLURM_JOB_ID", f"local_{os.getpid()}")
    ledger = args.out / f"collection_jobs_{run_id}.jsonl"
    started = time.time()

    def run(task):
        scene, case, route = task
        out = args.out / scene / route.stem
        out.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT / "scripts/traverse_fdm_rgbd_chrono.py"), "collect",
                   "--case", str(case), "--route", str(route), "--out", str(out),
                   "--chrono-data", args.chrono_data, "--record-rgbd-stride", str(args.record_rgbd_stride)]
        if args.horizon_s is not None:
            command += ["--horizon-s", str(args.horizon_s)]
        wall = time.time()
        with (out / "process.log").open("w") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
        return {"scene": scene, "route": route.stem, "out": str(out), "returncode": result.returncode,
                "wall_s": time.time()-wall, "case_sha256": sha256(case), "route_sha256": sha256(route)}

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(run, task) for task in tasks]
        for future in as_completed(pending):
            row = future.result()
            rows.append(row)
            with ledger.open("a") as handle:
                handle.write(json.dumps(row)+"\n")
            print(json.dumps({"done": len(rows), "total": len(tasks), **row}), flush=True)
    summary = {"manifest_sha256": manifest_hash, "runs": len(rows),
               "failed": sum(row["returncode"] != 0 for row in rows),
               "wall_s": time.time()-started, "workers": args.workers, "rows": rows}
    (args.out / f"collection_summary_{run_id}.json").write_text(json.dumps(summary, indent=2)+"\n")
    raise SystemExit(1 if summary["failed"] else 0)


if __name__ == "__main__":
    main()
