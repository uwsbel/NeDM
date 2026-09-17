#!/usr/bin/env python3
"""Immutable staged RGB-D observations and parallel headless Chrono episodes.

Each subprocess writes a unique attempt directory. Only validated successful
attempts become final outputs. Completed outputs are reused only after matching
their entire input contract and every saved file hash; incomplete attempts stay
available for diagnosis. Training is not performed here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "scripts/traverse_fdm_rgbd_diverse_chrono.py"
MARKER = "batch_complete.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checked(path, expected):
    path = Path(path).resolve()
    if sha256(path) != expected:
        raise ValueError(f"Frozen input changed: {path}")
    return path


def safe_id(value):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in (".", ".."):
        raise ValueError(f"Unsafe output identifier: {value!r}")
    return value


def source_fingerprint():
    files = sorted((ROOT / "src/nedm").rglob("*.py"))
    files += [COLLECTOR, Path(__file__).resolve(), ROOT / "scripts/check_traverse_fdm_rich_telemetry.py"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def runtime_fingerprint(chrono_data, dry_run=False):
    build = os.environ.get("CHRONO_BUILD")
    if not build and not dry_run:
        raise ValueError("CHRONO_BUILD is required: source nrd/env.sh and call nrd_pychrono")
    files = []
    if build:
        for name in ("lib/libChrono_core.so", "lib/libChrono_vehicle.so", "lib/libChrono_sensor.so",
                     "bin/pychrono/_core.so", "bin/pychrono/_vehicle.so", "bin/pychrono/_sensor.so"):
            path = Path(build) / name
            if path.exists():
                files.append(path.resolve())
        files += sorted((Path(build) / "bin/pychrono").glob("*.py"))
        if not (Path(build) / "bin/pychrono/_vehicle.so").is_file():
            raise ValueError(f"No vehicle module in selected build: {build}")
    data = Path(chrono_data).resolve()
    if not data.is_dir() and not dry_run:
        raise ValueError(f"Missing Chrono data: {data}")
    # Vehicle parameter files and meshes affect both physical and visual context.
    for subdir in (data / "vehicle/hmmwv",):
        if subdir.exists():
            files += [p for p in subdir.rglob("*") if p.is_file()]
    texture = data / "sensor/textures/grass_texture.jpg"
    if texture.is_file():
        files.append(texture)
    return {"python": str(Path(sys.executable).resolve()), "python_version": sys.version,
            "chrono_build": str(Path(build).resolve()) if build else None,
            "chrono_data": str(data), "file_sha256": {str(p): sha256(p) for p in sorted(set(files))},
            "render_backend": "Vulkan_RT_lavapipe",
            "thread_environment": {k: os.environ.get(k) for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LP_NUM_THREADS")}}


def completed(path, contract):
    path = Path(path)
    if not path.exists():
        return False
    marker = path / MARKER
    if not marker.is_file():
        raise ValueError(f"Existing final directory lacks completion marker; preserved: {path}")
    record = json.loads(marker.read_text())
    if record["contract"] != contract:
        raise ValueError(f"Completed output has a different input contract; use a new output root: {path}")
    actual = sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file() and p != marker)
    if actual != sorted(record["output_sha256"]):
        raise ValueError(f"Completed output file inventory changed: {path}")
    for relative, expected in record["output_sha256"].items():
        checked(path / relative, expected)
    return True


@contextmanager
def task_lock(root, task_id):
    lock = root / ".locks" / f"{task_id}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise RuntimeError(f"Task already locked (possibly interrupted); inspect before removing: {lock}") from None
    with os.fdopen(fd, "w") as handle:
        json.dump({"pid": os.getpid(), "host": os.uname().nodename,
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "created_unix": time.time()}, handle)
    try:
        yield
    finally:
        lock.unlink()


def validate_result(attempt, task, observation=None):
    import numpy as np

    required = ["simulation_provenance.json", "anchor_state.npz"]
    if task["stage"] == "observe":
        required += ["observation.npz", "observation.json", "rgb.png"]
    else:
        required += ["trajectory.npz", "outcome.json", "collection_meta.json", "contact_events.json"]
    for name in required:
        if not (attempt / name).is_file():
            raise ValueError(f"Collector omitted required output {name}")
    provenance = json.loads((attempt / "simulation_provenance.json").read_text())
    if provenance["case_sha256"] != task["case_sha256"]:
        raise ValueError("Collector case provenance disagrees with manifest")
    if provenance["script_sha256"] != sha256(COLLECTOR):
        raise ValueError("Collector source changed during execution")
    with np.load(attempt / "anchor_state.npz", allow_pickle=False) as anchor:
        for name in ("state", "pose"):
            if not np.isfinite(anchor[name]).all():
                raise ValueError(f"Nonfinite anchor {name}")
        if task["stage"] == "observe":
            with np.load(attempt / "observation.npz", allow_pickle=False) as image:
                if not np.isfinite(image["rgbd"]).all():
                    raise ValueError("Nonfinite encoded RGB-D")
                if not np.array_equal(anchor["state"], image["state"]) or not np.allclose(anchor["pose"], image["pose"], rtol=0., atol=1e-5):
                    raise ValueError("Observation and recorded anchor disagree")
            return
        # Compare native-precision physical anchors. The model observation pose
        # is float32, which can round a 100 m coordinate by nearly 4 micrometres.
        with np.load(observation / "anchor_state.npz", allow_pickle=False) as observed_anchor:
            differences = {name: float(np.max(np.abs(anchor[name] - observed_anchor[name]))) for name in ("state", "pose")}
            matched = bool(np.allclose(anchor["state"], observed_anchor["state"], rtol=1e-6, atol=1e-5)
                           and np.allclose(anchor["pose"], observed_anchor["pose"], rtol=0., atol=2e-6))
    obs_provenance = json.loads((observation / "simulation_provenance.json").read_text())
    same_runtime = all(provenance[k] == obs_provenance[k] for k in
                       ("pychrono", "python", "script_sha256", "scene_code_sha256", "physics_dt_s", "camera",
                        "source_sha256", "runtime_sha256"))
    join = {"matched": matched and same_runtime, "anchor_max_abs_difference": differences,
            "runtime_and_camera_matched": same_runtime,
            "observation_sha256": sha256(observation / "observation.npz"),
            "observation_completion_sha256": sha256(observation / MARKER),
            "observation": str(observation), "scope": "Single pre-drive RGB-D used for every causal anchor"}
    dump(attempt / "observation_join.json", join)
    if not join["matched"]:
        raise ValueError("Headless trajectory does not match its rendered observation contract")
    with np.load(attempt / "trajectory.npz", allow_pickle=False) as data:
        for name in ("state", "action", "pose", "terminal_state", "terminal_pose", "power_kw", "positive_work_kj_per_interval"):
            if not np.isfinite(data[name]).all():
                raise ValueError(f"Nonfinite trajectory {name}")
        frames = len(data["state"])
        if frames < 1 or len(data["action"]) != frames or len(data["pose"]) != frames:
            raise ValueError("Invalid trajectory array lengths")
    outcome = json.loads((attempt / "outcome.json").read_text())
    if outcome["frames"] != frames or outcome["route_sha256"] != task["route_sha256"]:
        raise ValueError("Outcome does not match measured trajectory/reference")
    if task["rich_telemetry"]:
        from check_traverse_fdm_rich_telemetry import verify
        rich_check = verify(attempt, trajectory=attempt / "trajectory.npz", require_solver_steps=True)
        with np.load(attempt / "rich_intervals.npz", allow_pickle=False) as intervals:
            for key in ("engine_interface_positive_work_kj", "max_abs_roll_rad", "max_abs_pitch_rad"):
                if not np.isfinite(intervals[key]).all():
                    raise ValueError(f"Required rich target unavailable: {key}")
        with np.load(attempt / "rich_telemetry.npz", allow_pickle=False) as samples:
            for wheel in ("fl", "fr", "rl", "rr"):
                for suffix in ("longitudinal_slip", "force_world_vertical_n"):
                    key = f"tire_{wheel}_{suffix}"
                    if not np.isfinite(samples[key]).all():
                        raise ValueError(f"Required tire telemetry unavailable: {key}")
        dump(attempt / "rich_validation.json", rich_check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chrono-data", required=True)
    parser.add_argument("--stage", choices=("observe", "collect", "all"), default="all")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
    parser.add_argument("--evaluation-only", action="store_true", help="Required with test alone; never used in development")
    parser.add_argument("--scene-ids", nargs="+")
    parser.add_argument("--route-indices", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--observation-workers", type=int, default=2)
    parser.add_argument("--horizon-s", type=float)
    parser.add_argument("--render-parity", action="store_true")
    parser.add_argument("--no-rich-telemetry", dest="rich_telemetry", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(rich_telemetry=True)
    args = parser.parse_args()
    if "test" in args.splits and (not args.evaluation_only or args.splits != ["test"]):
        parser.error("Test collection requires --evaluation-only --splits test; keep development separate")
    if args.evaluation_only and args.splits != ["test"]:
        parser.error("--evaluation-only requires --splits test")
    if min(args.workers, args.observation_workers) < 1 or (args.horizon_s is not None and args.horizon_s <= 0):
        parser.error("Worker counts and horizon must be positive")
    manifest = json.loads(args.manifest.read_text())
    records = [r for r in manifest["records"] if r["split"] in args.splits
               and (args.scene_ids is None or r["scene_id"] in args.scene_ids)]
    if not records:
        parser.error("No scenes match the declared selection")
    if args.scene_ids and set(args.scene_ids) != {r["scene_id"] for r in records}:
        parser.error("A requested scene ID is missing or outside the selected split")
    if len({r["scene_id"] for r in records}) != len(records):
        parser.error("Duplicate scene IDs in manifest")
    sources = source_fingerprint()
    common = {"schema": 1, "manifest_sha256": sha256(args.manifest),
              "source_sha256": sources, "runtime": runtime_fingerprint(args.chrono_data, args.dry_run)}
    tasks, observations = [], []
    for record in records:
        scene = safe_id(record["scene_id"])
        case = checked(args.manifest.parent / record["case"], record["case_sha256"])
        case_data = json.loads(case.read_text())
        if case_data["id"] != scene or case_data["split"] != record["split"]:
            raise ValueError(f"Case ID/split disagrees with manifest: {scene}")
        arena = ROOT / record["arena"]
        meta = checked(arena / "arena_meta.json", record["arena_meta_sha256"])
        checked(arena / json.loads(meta.read_text())["bmp"], record["arena_bmp_sha256"])
        basic = {"scene_id": scene, "split": record["split"], "case": str(case),
                 "case_sha256": record["case_sha256"], "arena_meta_sha256": record["arena_meta_sha256"],
                 "arena_bmp_sha256": record["arena_bmp_sha256"]}
        observations.append({**basic, "stage": "observe", "task_id": scene + "__observe"})
        indices = args.route_indices if args.route_indices is not None else list(range(len(record["routes"])))
        if len(record["routes"]) != len(record["route_sha256"]) or len(set(indices)) != len(indices):
            raise ValueError(f"Invalid route hash/index list for {scene}")
        for index in indices:
            if not 0 <= index < len(record["routes"]):
                raise ValueError(f"Route index {index} unavailable for {scene}")
            route = checked(args.manifest.parent / record["routes"][index], record["route_sha256"][index])
            name = safe_id(route.stem)
            tasks.append({**basic, "stage": "collect", "task_id": scene + "__" + name,
                          "route": str(route), "route_index": index, "route_name": name,
                          "route_sha256": record["route_sha256"][index],
                          "horizon_s": args.horizon_s if args.horizon_s is not None else float(case_data["horizon_s"]),
                          "render_parity": args.render_parity, "rich_telemetry": args.rich_telemetry})
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate route output identifiers")
    declaration = {"common_contract": common, "stage": args.stage, "observations": observations, "tasks": tasks,
                   "workers": args.workers, "observation_workers": args.observation_workers,
                   "evaluation_only": args.evaluation_only, "output_root": str(args.out.resolve())}
    if args.dry_run:
        print(json.dumps({"dry_run": True, "observations": len(observations), "routes": len(tasks),
                          "splits": sorted({r["split"] for r in records}), "tasks": tasks}, indent=2))
        return
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_id = f"{os.environ.get('SLURM_JOB_ID', 'manual')}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    runtime_path = out / "batches" / f"runtime_{run_id}.json"
    dump(runtime_path, {"runtime_sha256": common["runtime"]["file_sha256"],
                        "chrono_build": common["runtime"]["chrono_build"],
                        "chrono_data": common["runtime"]["chrono_data"],
                        "python": common["runtime"]["python"],
                        "python_version": common["runtime"]["python_version"]})
    os.environ["FDM_RUNTIME_FINGERPRINT"] = str(runtime_path)
    dump(out / "batches" / f"declaration_{run_id}.json", declaration)
    ledger = out / "batches" / f"ledger_{run_id}.jsonl"
    started = time.monotonic()
    observation_tasks = {t["scene_id"]: t for t in observations}

    def run(task):
        wall = time.monotonic()
        scene, stage = task["scene_id"], task["stage"]
        destination = out / "observations" / scene if stage == "observe" else out / "raw" / scene / task["route_name"]
        contract = {"common": common, "task": task}
        row = {"task_id": task["task_id"], "stage": stage, "scene_id": scene,
               "split": task["split"], "output": str(destination)}
        attempt = None
        try:
            with task_lock(out, task["task_id"]):
                observation = out / "observations" / scene
                if stage == "collect":
                    if not completed(observation, {"common": common, "task": observation_tasks[scene]}):
                        raise ValueError(f"Missing completed observation; run observe stage first: {scene}")
                    contract["observation_completion_sha256"] = sha256(observation / MARKER)
                if completed(destination, contract):
                    return {**row, "status": "reused", "returncode": 0, "wall_s": time.monotonic()-wall}
                attempt = out / ".attempts" / f"{task['task_id']}__{run_id}"
                attempt.mkdir(parents=True, exist_ok=False)
                command = [sys.executable, str(COLLECTOR), stage, "--case", task["case"],
                           "--out", str(attempt), "--chrono-data", args.chrono_data]
                if stage == "collect":
                    command += ["--route", task["route"], "--horizon-s", str(task["horizon_s"])]
                    if task["render_parity"]:
                        command += ["--render-parity"]
                    if not task["rich_telemetry"]:
                        command += ["--no-rich-telemetry"]
                dump(attempt / "invocation.json", {"command": command, "contract": contract})
                with (attempt / "process.log").open("w") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    raise RuntimeError(f"Collector exited {result.returncode}; see {attempt / 'process.log'}")
                if source_fingerprint() != sources:
                    raise ValueError("Source snapshot changed during collection")
                validate_result(attempt, task, observation if stage == "collect" else None)
                output_hashes = {str(p.relative_to(attempt)): sha256(p) for p in attempt.rglob("*") if p.is_file()}
                dump(attempt / MARKER, {"contract": contract, "output_sha256": output_hashes,
                                       "run_id": run_id, "completed_unix": time.time()})
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise ValueError(f"Final output appeared during collection; preserved both: {destination}")
                attempt.rename(destination)
                return {**row, "status": "completed", "returncode": 0, "wall_s": time.monotonic()-wall}
        except Exception as error:
            return {**row, "status": "failed", "returncode": 1, "error": repr(error),
                    "attempt": str(attempt) if attempt else None, "wall_s": time.monotonic()-wall}

    rows = []
    stages = [(observations, args.observation_workers)] if args.stage == "observe" else []
    if args.stage == "all":
        stages.append((observations, args.observation_workers))
    if args.stage in ("collect", "all"):
        stages.append((tasks, args.workers))
    for stage_tasks, workers in stages:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run, task) for task in stage_tasks]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                with ledger.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps({"finished": len(rows), **row}), flush=True)
        if any(row["status"] == "failed" for row in rows):
            break  # Never proceed to physics after an invalid observation stage.
    summary = {"schema": 1, "run_id": run_id, "manifest_sha256": common["manifest_sha256"],
               "stage": args.stage, "declared_observations": len(observations), "declared_routes": len(tasks),
               "processed": len(rows), "completed": sum(r["status"] == "completed" for r in rows),
               "reused": sum(r["status"] == "reused" for r in rows),
               "failed": sum(r["status"] == "failed" for r in rows),
               "wall_s": time.monotonic()-started, "rows": rows}
    dump(out / "batches" / f"summary_{run_id}.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)
    raise SystemExit(1 if summary["failed"] else 0)


if __name__ == "__main__":
    main()
