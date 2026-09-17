#!/usr/bin/env python3
"""Prepare and verify an exact frozen online trial with a passive camera.

Preparation never submits a job or changes the original source/data. The
generated Slurm script executes the frozen online runner and checks strict
parity. Archived reporting code creates the overview/video after local transfer.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np

ONLINE_V9_SHA = "c2ec8b81acd066c75e46cf658c8aa5e67661f92cf3a35a6da97575b03812382a"
ARRAY_FILES = ("trajectory.npz", "rich_telemetry.npz", "rich_intervals.npz", "anchor_state.npz")
IDENTITY_FILES = ("online_protocol.json", "anchor_equality.json", "initial_settle_reference.json")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_hash(path, expected):
    require(sha(path) == expected, f"SHA mismatch: {path}")


def check_source(root, expected):
    require_hash(root / "source_manifest.json", expected)
    # The immutable snapshot also archives every arena/case/route. A demo must
    # not enumerate or copy their payloads: verify executable source only, and
    # separately verify just the explicitly selected trial's allowed inputs.
    files = {name: digest for name, digest in dict(read(root / "source_manifest.json")["files"]).items()
             if name.startswith(("src/", "scripts/", "slurm/"))}
    require(files, "Snapshot has no executable source inventory")
    for relative, digest in files.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root), "Source manifest path escapes snapshot")
        require_hash(path, digest)
    return files


def selected_task(document, trial_id, allow_test=False):
    matches = [task for task in document["tasks"] if task["id"] == trial_id]
    require(len(matches) == 1, "Require one explicitly selected frozen trial")
    task = copy.deepcopy(matches[0])
    require(task.get("split") in ("train", "val", "test"), "Trial split must be explicit")
    require(task["split"] != "test" or allow_test,
            "Protected test trial requires explicit --allow-protected-test after selection is frozen")
    require(float(task["arguments"].get("video-fps", 0)) == 0, "Baseline must be headless")
    return task


def snapshot_payloads(folder):
    paths = [folder / name for name in (*ARRAY_FILES, *IDENTITY_FILES,
        "outcome.json", "online_planning_summary.json", "simulation_provenance.json")]
    decisions = sorted((folder / "decisions").glob("decision_*.json"))
    require(decisions, "Completed reference trial has no decisions")
    return {str(path.relative_to(folder)): sha(path) for path in paths + decisions}


def prepare(args):
    source, headless, out = args.source_root.resolve(), args.headless_root.resolve(), args.out.resolve()
    fps = int(args.video_fps)
    require(fps in (1, 5), "Supported media cadences are 1 Hz and 5 Hz")
    require(not out.exists(), "Use a new demo directory")
    source_files = check_source(source, args.expected_source_sha256)
    document = read(args.tasks)
    task = selected_task(document, args.trial_id, args.allow_protected_test)
    declaration = read(headless / "declaration.json")
    require_hash(args.tasks, declaration["tasks_sha256"])
    require(declaration["source_manifest_sha256"] == args.expected_source_sha256,
            "Headless trial did not execute the requested immutable source")
    recorded = [item for item in declaration["tasks"] if item["id"] == task["id"]]
    require(recorded == [task], "Requested task differs from actual headless declaration")
    result = read(headless / (task["id"] + ".result.json"))
    require(result["exit_code"] == 0, "Reference headless process did not complete")
    baseline = headless / task["id"]
    require(not out.is_relative_to(headless) and not headless.is_relative_to(out),
            "Demo output must be separate from immutable headless results")
    for name, digest in result["sha256"].items():
        require_hash(baseline / name, digest)
    protocol = read(baseline / "online_protocol.json")
    require(protocol["planning_mode"] == task["arguments"].get("planning-mode", "receding"),
            "Baseline planning mode disagrees with declaration")
    for name, digest in protocol["source_sha256"].items():
        require(source_files.get(name) == digest, "Baseline protocol source differs from requested source")
    inputs = {key: {"path": value, "sha256": sha(value)} for key, value in task["arguments"].items()
              if key in ("case", "checkpoint", "scene-observation", "cost-config")}
    observation_meta = str(Path(inputs["scene-observation"]["path"]).with_suffix(".json"))
    inputs["scene-observation-metadata"] = {"path": observation_meta, "sha256": sha(observation_meta)}
    require(inputs["checkpoint"]["sha256"] == protocol["model_checkpoint_sha256"], "Checkpoint changed")
    require(inputs["scene-observation"]["sha256"] == protocol["scene_observation_sha256"], "Observation changed")
    require(inputs["case"]["sha256"] == read(baseline / "outcome.json")["case_sha256"], "Case changed")
    video_task = copy.deepcopy(task)
    video_task["id"] += f"_video{fps}"
    video_task["arguments"]["video-fps"] = fps
    unchanged = dict(video_task["arguments"])
    unchanged.pop("video-fps")
    original_args = dict(task["arguments"])
    original_args.pop("video-fps", None)
    require(unchanged == original_args, "Only video-fps may change")
    out.mkdir(parents=True)
    archived = out / "reporting_source"
    archived.mkdir()
    helper = archived / Path(__file__).name
    renderer = archived / "render_traverse_fdm_diverse_online_report.py"
    shutil.copy2(__file__, helper)
    shutil.copy2(args.renderer, renderer)
    tasks_path = out / "video_tasks.json"
    dump(tasks_path, {"scope": f"Exact selected full trial, passive {fps} Hz camera only",
        "source_manifest_sha256": args.expected_source_sha256, "tasks": [video_task],
        "original_tasks_sha256": sha(args.tasks), "original_trial_id": task["id"]})
    plan = {"schema": 2, "status": "prepared_not_submitted", "trial": task, "video_trial": video_task,
        "video_fps": fps,
        "split": task["split"], "protected_test_explicitly_enabled": args.allow_protected_test,
        "source_root": str(source), "source_manifest_sha256": args.expected_source_sha256,
        "source_files": source_files, "original_tasks": str(args.tasks.resolve()),
        "original_tasks_sha256": sha(args.tasks), "headless_root": str(headless),
        "headless_trial": str(baseline), "headless_payload_sha256": snapshot_payloads(baseline),
        "headless_declaration_sha256": sha(headless / "declaration.json"),
        "headless_result_sha256": sha(headless / (task["id"] + ".result.json")),
        "input_files": inputs, "video_tasks": str(tasks_path), "video_tasks_sha256": sha(tasks_path),
        "video_root": str(out / "raw"), "video_trial_directory": str(out / "raw" / video_task["id"]),
        "report_directory": str(out / "report"), "report_label": "selected_trial",
        "reporting_source_sha256": {str(path): sha(path) for path in (helper, renderer)},
        "parity_policy": "Exact dtype, shape and bytes for every core/rich/anchor array; exact decisions excluding planning wall time; identical frozen source, inputs and runtime. Any difference blocks evidence rendering.",
        "video_policy": f"Actual 640x480 Chrono chase pixels at {fps} Hz, measured terminal frame, variable timestamp encoding; no interpolation or synthetic vehicle motion. Physics and telemetry cadence are unchanged.",
        "allocation": {"partition": "mi2101x", "cpus": 16, "workers": 1, "wall_limit": "01:00:00",
                       "memory_request": "None: cluster scheduler RealMemory is a placeholder"}}
    dump(out / "replay_plan.json", plan)
    q = shlex.quote
    commands = ["#!/bin/bash", "#SBATCH -J fdm_demo_replay", "#SBATCH -A dannegrut",
        "#SBATCH -p mi2101x", "#SBATCH -N 1", "#SBATCH -n 1", "#SBATCH -c 16",
        "#SBATCH -t 01:00:00", f"#SBATCH -o {out}/slurm_%j.log", "set -eo pipefail",
        "source /work1/dannegrut/harry/nrd/env.sh", "nrd_pychrono", "nrd_use_lavapipe",
        f"cd {q(str(source))}", f"export PYTHONPATH={q(str(source / 'src') + ':' + str(source / 'scripts'))}:\"${{PYTHONPATH:-}}\"",
        "export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4",
        f'"$NRD_PYTHON" {q(str(helper))} preflight --plan {q(str(out / "replay_plan.json"))}',
        f'"$NRD_PYTHON" -u scripts/traverse_fdm_rgbd_diverse_online_batch.py --tasks {q(str(tasks_path))} --out {q(str(out / "raw"))} --workers 1 --python "$NRD_PYTHON" --chrono-data "$CHRONO_BUILD/data"',
        f'"$NRD_PYTHON" {q(str(helper))} verify --plan {q(str(out / "replay_plan.json"))} --out {q(str(out / "parity.json"))}', ""]
    (out / "launch.sbatch").write_text("\n".join(commands))
    local_commands = ["#!/bin/bash", "set -eo pipefail",
        'DEMO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
        f'"${{DEMO_PYTHON:-python}}" "$DEMO_DIR/reporting_source/{helper.name}" verify-transfer --root "$DEMO_DIR"',
        f'"${{DEMO_PYTHON:-python}}" "$DEMO_DIR/reporting_source/{renderer.name}" --run "selected_trial=$DEMO_DIR/raw/{video_task["id"]}" --observation "$DEMO_DIR/inputs/observation.npz" --checkpoint "$DEMO_DIR/inputs/checkpoint.pt" --code-root "$DEMO_DIR/source_online_v9" --out "$DEMO_DIR/report" --cpu-threads 4 --encode-video', ""]
    (out / "render_local.sh").write_text("\n".join(local_commands))
    dump(out / "preparation_manifest.json", {"files": {name: sha(out / name) for name in
        ("replay_plan.json", "video_tasks.json", "launch.sbatch", "render_local.sh")}, "helper_sha256": sha(__file__)})
    print(json.dumps({"prepared": str(out), "submitted": False,
                      "launch_command": shlex.join(["sbatch", str(out / "launch.sbatch")])}))


def preflight(plan):
    source = Path(plan["source_root"])
    check_source(source, plan["source_manifest_sha256"])
    require_hash(plan["original_tasks"], plan["original_tasks_sha256"])
    require_hash(plan["video_tasks"], plan["video_tasks_sha256"])
    for value in plan["input_files"].values():
        require_hash(value["path"], value["sha256"])
    for path, digest in plan["reporting_source_sha256"].items():
        require_hash(path, digest)
    root = Path(plan["headless_root"])
    require_hash(root / "declaration.json", plan["headless_declaration_sha256"])
    require_hash(root / (plan["trial"]["id"] + ".result.json"), plan["headless_result_sha256"])
    for relative, digest in plan["headless_payload_sha256"].items():
        require_hash(Path(plan["headless_trial"]) / relative, digest)


def compare_npz(left, right):
    rows = {}
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        require(set(a.files) == set(b.files), f"Array keys differ: {left.name}")
        for key in a.files:
            x, y = a[key], b[key]
            exact = x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()
            require(exact, f"Array parity failed: {left.name}:{key}")
            rows[key] = {"shape": list(x.shape), "dtype": str(x.dtype),
                         "array_sha256": hashlib.sha256(x.tobytes()).hexdigest(), "exact": True}
    return rows


def route_fingerprint(route):
    if route is None:
        return None
    digest = hashlib.sha256()
    for key in ("waypoints", "stations", "headings", "speeds"):
        value = np.ascontiguousarray(route[key], dtype="<f8")
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def compare_decisions(left, right):
    a, b = sorted((left / "decisions").glob("decision_*.json")), sorted((right / "decisions").glob("decision_*.json"))
    require([path.name for path in a] == [path.name for path in b] and a, "Decision inventory differs")
    rows = []
    for x, y in zip(a, b, strict=True):
        first, second = read(x), read(y)
        first.pop("planning_wall_s")
        second.pop("planning_wall_s")
        require(first == second, f"Decision parity failed: {x.name}")
        route = first["decision"].get("route")
        rows.append({"file": x.name, "frame": first["frame"], "route_sha256": route_fingerprint(route),
            "abstained": first["decision"].get("abstained", False),
            "driver_route_fingerprint": (first.get("driver_update") or {}).get("route_fingerprint"),
            "all_other_decision_fields_exact": True})
    return rows


def verify(plan, out):
    require(not out.exists(), "Preserve existing parity reports")
    preflight(plan)
    left, right = Path(plan["headless_trial"]), Path(plan["video_trial_directory"])
    result = read(Path(plan["video_root"]) / (plan["video_trial"]["id"] + ".result.json"))
    require(result["exit_code"] == 0, "Video replay did not complete")
    for relative, digest in result["sha256"].items():
        require_hash(right / relative, digest)
    declared = read(Path(plan["video_root"]) / "declaration.json")
    require(declared["tasks"] == [plan["video_trial"]], "Executed video task differs from preparation")
    require(declared["source_manifest_sha256"] == plan["source_manifest_sha256"], "Video source differs")
    arrays = {name: compare_npz(left / name, right / name) for name in ARRAY_FILES}
    for name in IDENTITY_FILES:
        require(read(left / name) == read(right / name), f"Identity differs: {name}")
    runtime_a, runtime_b = read(left / "simulation_provenance.json"), read(right / "simulation_provenance.json")
    for key in ("runtime_sha256", "source_sha256", "case_sha256", "arena_meta_sha256", "arena_bmp_sha256", "physics_dt_s", "camera", "driver"):
        require(runtime_a[key] == runtime_b[key], f"Simulation provenance differs: {key}")
    outcomes = [read(folder / "outcome.json") for folder in (left, right)]
    for value in outcomes:
        value.pop("wall_s")
        value.pop("planning_wall_s")
    require(outcomes[0] == outcomes[1], "Measured outcome differs beyond permitted wall timings")
    decisions = compare_decisions(left, right)
    if plan["trial"]["arguments"].get("planning-mode", "receding") == "plan_once":
        require(len(decisions) == 1 and decisions[0]["frame"] == 0, "Plan-once replay contains later decisions")
    metadata = read(right / "frame_metadata.json")
    fps = plan.get("video_fps", 5)
    require(fps in (1, 5) and plan["video_trial"]["arguments"]["video-fps"] == fps, "Video plan cadence disagrees with executed task")
    require(metadata["nominal_fps"] == fps, "Video must use the declared passive sensor cadence")
    with np.load(right / "trajectory.npz", allow_pickle=False) as physical:
        n, dt = len(physical["state"]), .05
        frames = metadata["frames"]
        stride = int(round(1. / (dt * fps)))
        require([row["telemetry_frame"] for row in frames] == list(range(0, n, stride)) + [n], "Video frame schedule differs")
        for row in frames:
            index, terminal = row["telemetry_frame"], row["terminal"]
            require(terminal == (index == n), "Terminal video flag differs")
            require(row["recording_time_s"] == index * dt, "Video recording timestamp differs")
            require(np.isclose(row["simulation_time_s"] - row["recording_time_s"], .8, atol=1e-8, rtol=0), "Video simulation timestamp differs")
            for key, expected in (("actual_pose", physical["terminal_pose"] if terminal else physical["pose"][index]),
                                  ("actual_state17", physical["terminal_state"] if terminal else physical["state"][index]),
                                  ("applied_action", physical["action"][-1] if terminal else physical["action"][index])):
                require(np.array_equal(np.asarray(row[key]), expected), f"Video frame is not the measured {key}")
        times = np.load(right / "frame_times_s.npy", allow_pickle=False)
        require(np.array_equal(times, [row["recording_time_s"] for row in frames]), "Video timestamp array differs")
    from PIL import Image
    pngs = {}
    for row in frames:
        path = (right / row["file"]).resolve()
        require(path.is_relative_to(right), "Frame path escapes video output")
        with Image.open(path) as image:
            require(image.size == (640, 480) and image.mode == "RGB", "Unexpected passive camera frame")
            image.verify()
        pngs[row["file"]] = sha(path)
    require(len(list((right / "frames").glob("*.png"))) == len(frames), "Extra/missing video images")
    camera = read(right / "video_camera.json")
    require(camera["nominal_fps"] == fps, "Camera metadata disagrees with declared media cadence")
    require(camera["physics_script_sha256"] == plan["source_files"]["scripts/traverse_fdm_rgbd_diverse_online.py"], "Video physics source differs")
    require(camera["video_adapter_sha256"] == plan["source_files"]["scripts/traverse_fdm_rgbd_diverse_video.py"], "Video adapter source differs")
    report = {"passed": True, "trial_id": plan["trial"]["id"], "video_trial_id": plan["video_trial"]["id"],
        "source_manifest_sha256": plan["source_manifest_sha256"], "arrays": arrays,
        "array_count": sum(len(values) for values in arrays.values()), "decisions": decisions,
        "runtime_and_protocol_exact": True, "outcome_exact_except_wall_timings": True,
        "video_fps": fps, "frame_count": len(frames), "physical_elapsed_s": outcomes[0]["elapsed_s"],
        "frame_sha256": pngs, "video_payload_sha256": snapshot_payloads(right),
        "claim": "The passive camera replay reproduces every stored physical array and decision exactly. Actual rendered frames are registered to those measured states."}
    dump(out, report)
    write_transfer_manifest(plan, out)
    print(json.dumps({key: report[key] for key in ("passed", "trial_id", "array_count", "frame_count", "physical_elapsed_s")}))


def write_transfer_manifest(plan, parity_path):
    destination = parity_path.parent
    files = {}

    def add(remote, local):
        path = Path(remote)
        require(local not in files, f"Duplicate transfer destination: {local}")
        files[local] = {"remote_path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}

    for path in destination.glob("*.json"):
        if path.name != "transfer_manifest.json":
            add(path, path.name)
    for name in ("launch.sbatch", "render_local.sh"):
        add(destination / name, name)
    for path in plan["reporting_source_sha256"]:
        add(path, "reporting_source/" + Path(path).name)
    source = Path(plan["source_root"])
    for relative in ["source_manifest.json", *plan["source_files"]]:
        add(source / relative, "source_online_v9/" + relative)
    for key, name in (("scene-observation", "observation.npz"),
                      ("scene-observation-metadata", "observation.json"), ("checkpoint", "checkpoint.pt")):
        add(plan["input_files"][key]["path"], "inputs/" + name)
    raw = Path(plan["video_root"])
    for path in sorted(raw.rglob("*")):
        if path.is_file() and path.suffix in (".json", ".npz", ".npy", ".png"):
            add(path, "raw/" + str(path.relative_to(raw)))
    dump(destination / "transfer_manifest.json", {"schema": 1, "parity_passed": True,
        "parity_sha256": sha(parity_path), "files": files,
        "file_count": len(files), "total_bytes": sum(value["bytes"] for value in files.values()),
        "instruction": "Copy each remote_path to its relative destination under a new local demo directory, preserve this manifest, then run render_local.sh with DEMO_PYTHON pointing to the local NeDM environment. Physics/NumPy parity already completed on AMD; no Matplotlib import is needed there."})


def verify_transfer(root):
    root = root.resolve()
    manifest = read(root / "transfer_manifest.json")
    require(manifest["parity_passed"], "Transfer requires completed strict physics parity")
    for relative, value in manifest["files"].items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root), "Transfer path escapes local demo")
        require_hash(path, value["sha256"])
    require_hash(root / "parity.json", manifest["parity_sha256"])
    require(read(root / "parity.json")["passed"], "Stored parity did not pass")
    print(json.dumps({"transfer_verified": True, "files": len(manifest["files"]),
                      "bytes": manifest["total_bytes"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--trial-id", required=True)
    p.add_argument("--headless-root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--expected-source-sha256", default=ONLINE_V9_SHA)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allow-protected-test", action="store_true")
    p.add_argument("--video-fps", type=int, choices=(1, 5), default=5,
                   help="Passive media cadence only; full physics and telemetry remain unchanged")
    p.add_argument("--renderer", type=Path, default=Path(__file__).with_name("render_traverse_fdm_diverse_online_report.py"))
    for name in ("preflight", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--plan", type=Path, required=True)
        if name == "verify":
            p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("verify-transfer")
    p.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "preflight":
        preflight(read(args.plan))
        print(json.dumps({"passed": True, "preflight": str(args.plan)}))
    elif args.command == "verify-transfer":
        verify_transfer(args.root)
    else:
        verify(read(args.plan), args.out)


if __name__ == "__main__":
    main()
