#!/usr/bin/env python3
"""Headless collection on the f104 terrain FAMILY through the immutable rich Chrono/native PID runner.

Byte-for-byte the f104 collector (scripts/collect_traverse_f104.py) except the arena gate: instead of requiring
the exact f104 BMP and its [-1.8, 3.9] m height range, the arena's BMP must be listed with its sha256 in
gen_arenas.json next to this script (f104 plus the five sibling arenas). Physics, driver, stop policy, native
terrain audit and launch validation are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

F104_BMP_SHA256 = "5d5bc683b6a8d9e99fd323c4112752e0cad214b986319ea1b2396631104ee8ed"
DT = .05
SOURCE_FILES = ("scripts/traverse_fdm_rgbd_diverse_chrono.py", "src/nedm/traverse/scene.py",
    "src/nedm/hmmwv_data.py", "src/nedm/traverse/terrain.py", "src/nedm/traverse/layout.py",
    "src/nedm/traverse/fdm_data.py", "src/nedm/traverse/fdm_diverse_data.py",
    "src/nedm/traverse/fdm_rich_telemetry.py", "src/nedm/training/constants.py")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(ok, message):
    if not ok:
        raise ValueError(message)


class StopPolicy:
    """Causal bounded-displacement evidence; normal goal/rollover take priority."""
    def __init__(self, args, case):
        self.args, self.case = args, case
        self.minimum_s, self.confirm_s, self.tail_s = args.minimum_elapsed_s, args.confirm_s, args.recovery_tail_s
        self.window_s, self.diameter_m = 2., .25
        self.first_confirmation_s = None
        self.confirmed = False
        self.events = []
        self.commands = []
        self.native_height_report = None
        self.initial_state_report = None

    def config(self):
        return {"enabled": not self.args.disable_early_stop, "minimum_elapsed_s": self.minimum_s,
            "evidence_window_s": self.window_s, "confirmation_s": self.confirm_s,
            "recovery_tail_s": self.tail_s, "maximum_pairwise_xy_diameter_m": self.diameter_m,
            "all_interval_throttle_gt": .3, "exclude_deliberate_parking": True,
            "cancel_pending_on": "Any current 2 s window fails bounded, effortful, nonparking criterion",
            "earliest_possible_stop_s": self.minimum_s+self.confirm_s+self.tail_s,
            "terrain_bounds_xy_m": [-40., 40.],
            "priority": ["rollover", "goal_reached", "terrain_bounds_exit", "prolonged_blockage_terminated", "timeout"],
            "scope": "Episode truncation only; no reverse/recovery command intervention and no fabricated tail"}

    def bind_scene(self, scene, tmap):
        # Preserve original collision initialization. Before its first Advance,
        # unbound Bullet ray queries return the fallback zero height.
        self.tmap = tmap

    def validate_native_height(self, scene):
        import pychrono as chrono
        values = np.linspace(-36., 36., 7)
        points = [(float(x), float(y)) for x in values for y in values]
        points.append(tuple(self.case["layout"]["start_xy"]))
        actual = np.asarray([scene.terrain.GetHeight(chrono.ChVector3d(x, y, 20.)) for x, y in points])
        expected = np.asarray([self.tmap.height(x, y) for x, y in points])
        error = np.abs(actual-expected)
        passed = bool(np.isfinite(actual).all() and np.quantile(error, .95) <= .08 and error.max() <= .15)
        self.native_height_report = {"sample_count": len(points), "xy_m": points,
            "native_height_m": actual.tolist(), "bmp_terrain_map_height_m": expected.tolist(),
            "median_abs_error_m": float(np.median(error)), "p95_abs_error_m": float(np.quantile(error, .95)),
            "max_abs_error_m": float(error.max()), "p95_limit_m": .08, "max_limit_m": .15,
            "passed": passed, "metadata_mutated": False,
            "scope": "Actual native terrain versus declared quantized BMP transform after original 0.8 s settling, before traversal advance; collision initialization unmodified"}
        dump(Path(self.args.out)/"native_height_check.json", self.native_height_report)
        require(passed, "Native F104 terrain does not match the declared BMP transform; report preserved")

    def on_anchor(self, scene, state, pose, history):
        self.validate_native_height(scene)
        speed = float(np.linalg.norm(np.asarray(state)[:2]))
        roll, pitch = float(state[2]), float(state[3])
        yaw_error = float(math.atan2(math.sin(pose[2]-self.case["layout"]["start_yaw"]),
                                    math.cos(pose[2]-self.case["layout"]["start_yaw"])))
        position_error = float(np.linalg.norm(np.asarray(pose[:2])-self.case["layout"]["start_xy"]))
        finite = bool(np.isfinite(state).all() and np.isfinite(pose).all() and np.isfinite(history).all())
        passed = bool(finite and speed <= 1. and max(abs(roll), abs(pitch)) <= math.radians(20.)
                      and abs(yaw_error) <= math.radians(10.) and position_error <= 1.)
        self.initial_state_report = {"passed": passed, "finite": finite, "body_horizontal_speed_mps": speed,
            "roll_rad": roll, "pitch_rad": pitch, "yaw_error_rad": yaw_error,
            "start_xy_error_m": position_error, "pose": np.asarray(pose).tolist(),
            "limits": {"body_horizontal_speed_mps": 1., "absolute_roll_pitch_deg": 20., "yaw_error_deg": 10., "start_xy_error_m": 1.},
            "settle_s": .8, "chassis_contact_is_not_a_launch_rejection": True,
            "history": "Measured t=0 state repeated with synthetic previous brake, exactly the frozen collector convention"}
        dump(Path(self.args.out)/"initial_state_validation.json", self.initial_state_report)
        require(passed, "Invalid settled launch; anchor and validation report preserved before traversal advance")

    def check(self, frame, poses, actions, parked, terminal_pose, wp, desired_speed):
        elapsed = (frame+1)*DT
        self.commands.append((frame*DT, elapsed, int(wp), float(desired_speed), bool(parked[-1])))
        if np.max(np.abs(np.asarray(terminal_pose)[:2])) > 40.:
            self.events.append({"kind": "terrain_bounds_exit", "interval_end_s": elapsed})
            return "terrain_bounds_exit"
        if self.args.disable_early_stop or len(actions) < 40:
            return None
        bounded = False
        if not any(parked[-40:]) and np.all(np.asarray(actions[-40:])[:, 1] > .3):
            points = np.concatenate((np.asarray(poses[-40:])[:, :2], np.asarray(terminal_pose)[None, :2]))
            diameter2 = np.square(points[:, None, :]-points[None, :, :]).sum(-1).max()
            bounded = bool(diameter2 <= self.diameter_m**2)
        if not bounded:
            if self.first_confirmation_s is not None:
                self.events.append({"kind": "pending_stop_cancelled", "interval_end_s": elapsed,
                    "confirmation_began_s": self.first_confirmation_s,
                    "reason": "Current measured bounded-effortful window no longer qualifies"})
            self.first_confirmation_s, self.confirmed = None, False
            return None
        if elapsed+1e-9 < self.minimum_s:
            return None
        if self.first_confirmation_s is None:
            self.first_confirmation_s = elapsed
            self.events.append({"kind": "confirmation_started", "interval_end_s": elapsed,
                                "first_qualifying_window_start_s": elapsed-self.window_s})
        confirmed_time = self.first_confirmation_s+self.confirm_s
        if elapsed+1e-9 >= confirmed_time and not self.confirmed:
            self.confirmed = True
            self.events.append({"kind": "confirmed_tail_started", "interval_end_s": elapsed,
                                "scheduled_stop_s": confirmed_time+self.tail_s})
        if elapsed+1e-9 >= confirmed_time+self.tail_s:
            self.events.append({"kind": "prolonged_blockage_terminated", "interval_end_s": elapsed})
            return "prolonged_blockage_terminated"
        return None


def import_runner(source):
    sys.path.insert(0, str(source/"src"))
    spec = importlib.util.spec_from_file_location("f104_frozen_collector", source/SOURCE_FILES[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def adapted_function(module):
    """Fail closed if the frozen source no longer has these exact hook locations."""
    source = inspect.getsource(module.run_chrono)
    hooks = [
        ("    hmmwv, system, terrain = scene.hmmwv, scene.system, scene.terrain\n",
         "    args.f104_policy.bind_scene(scene, tmap)\n"),
        ('                    if args.command == "observe":\n',
         "                    args.f104_policy.on_anchor(scene, state, pose, history)\n"),
        ('        frame += 1\n',
         '            f104_stop = args.f104_policy.check(frame, record_pose, record_action, record_parked, terminal_pose, wp, 0. if at_end else float(speed[wp]))\n'
         '            if f104_stop is not None:\n'
         '                status = f104_stop\n'
         '                break\n'),
    ]
    # There are two observe guards in the anchor block; insert before the first.
    require(source.count(hooks[0][0]) == 1 and source.count(hooks[2][0]) == 1, "Frozen loop hook locations changed")
    require(source.count(hooks[1][0]) == 2, "Frozen anchor hook locations changed")
    original = source
    for old, insertion in hooks:
        source = source.replace(old, insertion+old, 1)
    namespace = dict(module.__dict__)
    exec(compile(source, str(Path(__file__).resolve())+":adapted_frozen_run_chrono", "exec"), namespace)
    return namespace["run_chrono"], {"original_function_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "adapted_function_sha256": hashlib.sha256(source.encode()).hexdigest(), "hook_count": 3,
        "mutation_scope": "Three additive audit/termination hooks; original native physics, driver and integration statements retained verbatim"}


def make_observer(out, case):
    from nedm.traverse.fdm_rich_telemetry import RichTelemetry

    class F104Telemetry(RichTelemetry):
        def _snapshot(self, scene, frame, action, terminal, command_context=None):
            row = super()._snapshot(scene, frame, action, terminal, command_context)
            vehicle = scene.hmmwv.GetVehicle()
            for name, axle, side in self._wheel_specs:
                normal = self._vector(row, f"{name}_terrain_normal_under_hub_world",
                    lambda axle=axle, side=side: scene.terrain.GetNormal(vehicle.GetSpindlePos(axle, side)),
                    "1", f"{name}: terrain.GetNormal at measured spindle XY; geometric terrain normal, not tire contact-patch normal", "unit")
                force = np.asarray([row[f"{name}_force_world_{axis}_n"] for axis in "xyz"])
                self._put(row, f"{name}_force_projected_terrain_normal_n", np.dot(force, normal), "N",
                    "Reported world tire-force resultant dotted with geometric terrain normal under hub; not an internal tire normal-load getter")
            return row

    return F104Telemetry(out, case_path=case, record_dt_s=DT)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", required=True)
    p.add_argument("--source-manifest-sha256")
    p.add_argument("--case", required=True)
    p.add_argument("--case-sha256")
    p.add_argument("--route", required=True)
    p.add_argument("--route-sha256")
    p.add_argument("--out", required=True)
    p.add_argument("--chrono-data", required=True)
    p.add_argument("--horizon-s", type=float, default=120.)
    p.add_argument("--minimum-elapsed-s", type=float, default=24.)
    p.add_argument("--confirm-s", type=float, default=2.)
    p.add_argument("--recovery-tail-s", type=float, default=8.)
    p.add_argument("--disable-early-stop", action="store_true", help="Parity/long-tail control only; terrain bounds remain active")
    p.add_argument("--check-only", action="store_true", help="Validate files/hooks without importing Chrono or creating outputs")
    return p


def main():
    args = parser().parse_args()
    source, out = Path(args.source_root).resolve(), Path(args.out).resolve()
    args.case, args.route, args.out = str(Path(args.case).resolve()), str(Path(args.route).resolve()), str(out)
    require(0. < args.horizon_s <= 120. and abs(args.horizon_s/DT-round(args.horizon_s/DT)) < 1e-8, "Horizon must be a positive 50 ms multiple, at most 120 s")
    require(args.minimum_elapsed_s >= 24. and args.confirm_s >= 2. and args.recovery_tail_s >= 8., "Do not shorten the declared failure/recovery safeguards")
    manifest_path = source/"source_manifest.json"
    manifest = read(manifest_path)
    if args.source_manifest_sha256:
        require(sha(manifest_path) == args.source_manifest_sha256, "Source manifest mismatch")
    source_hashes = {name: sha(source/name) for name in SOURCE_FILES}
    require(all(manifest["files"].get(name) == value for name, value in source_hashes.items()), "Frozen source file mismatch")
    case = read(args.case)
    require(case["split"] in ("train", "val", "test"), "Missing declared group split")
    arena = (source/case["arena"]).resolve()
    meta = read(arena/"arena_meta.json")
    allowed = read(Path(__file__).resolve().parent/"gen_arenas.json")
    require(float(meta["size_m"]) == 80., "Require the 80 m arena size of the f104 family")
    require(allowed.get(arena.name) == sha(arena/meta["bmp"]), f"Arena {arena.name} BMP not in gen_arenas.json allowlist")
    require(case["layout"]["assets"] == [], "This campaign is the exact F104 terrain without added assets")
    for field in ("case", "route"):
        if getattr(args, field+"_sha256"):
            require(sha(getattr(args, field)) == getattr(args, field+"_sha256"), f"{field} checksum mismatch")
    module = import_runner(source)
    route = module.read_route(args.route)
    run, adapter = adapted_function(module)
    args.f104_policy = StopPolicy(args, case)
    contract = {"schema": "f104_collection_request_v1", "source_root": str(source),
        "source_manifest_sha256": sha(manifest_path), "source_sha256": source_hashes,
        "wrapper_sha256": sha(__file__), "adapter": adapter, "case": args.case, "case_sha256": sha(args.case),
        "route": args.route, "route_sha256": sha(args.route), "scene_id": case["id"], "split": case["split"],
        "arena_bmp_sha256": sha(arena/meta["bmp"]), "arena_meta_sha256": sha(arena/"arena_meta.json"),
        "horizon_s": args.horizon_s, "stop_policy": args.f104_policy.config(), "route_metadata": route.get("meta", {}),
        "observation": "One separately rendered terrain-only RGB-D map may be joined by BMP/camera/runtime hash. Per-episode measured anchor is distinct; no shared vehicle-anchor equality claimed.",
        "time_accounting": "Completed measured traversal intervals only; exclude 0.8 s settling, failed processes and any postprocessing window overlap"}
    if args.check_only:
        print(json.dumps({"check_only": True, "scene_id": case["id"], "contract": contract}))
        return
    out.mkdir(parents=True, exist_ok=True)
    require(not any((out/name).exists() for name in ("outcome.json", "collection_request.json", "trajectory.npz")), "Preserve existing output; use a new directory")
    fingerprint = os.environ.get("FDM_RUNTIME_FINGERPRINT")
    require(fingerprint and Path(fingerprint).is_file(), "FDM_RUNTIME_FINGERPRINT must bind the full Chrono libraries and vehicle assets")
    runtime = read(fingerprint)["runtime_sha256"]
    require(runtime and any("_vehicle.so" in name for name in runtime) and any("/vehicle/hmmwv/" in name for name in runtime), "Runtime fingerprint lacks native vehicle library or HMMWV data")
    contract["runtime_fingerprint_sha256"] = sha(fingerprint)
    contract["runtime_sha256"] = runtime
    contract["thread_environment"] = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LP_NUM_THREADS")}
    dump(out/"collection_request.json", contract)
    shutil.copyfile(args.case, out/"case.json")
    shutil.copyfile(args.route, out/"reference.json")
    args.command, args.backend, args.depth_ray_scale = "collect", "Vulkan_RT_lavapipe", 1.
    args.record_rgbd_stride, args.path_height_source = 0, "truth"
    args.render_parity, args.rich_telemetry = False, True
    args.frame_observer = make_observer(out, args.case)
    started = time.time()
    try:
        run(args)
        outcome = read(out/"outcome.json")
        with np.load(out/"trajectory.npz", allow_pickle=False) as data:
            n = len(data["state"])
            require(n > 0 and n == outcome["frames"] and abs(n*DT-outcome["elapsed_s"]) < 1e-8, "Actual interval count/duration disagree")
            require(data["terminal_state"].shape == (17,) and np.isfinite(data["terminal_pose"]).all(), "Missing finite actual terminal measurement")
        # Normal goal/rollover break precedes the hook, so append its last command
        # from the rich pre-interval record if necessary; never synthesize physics.
        with np.load(out/"rich_telemetry.npz", allow_pickle=False) as rich:
            desired = rich["command_desired_speed_mps"][:n].copy()
        np.savez_compressed(out/"command_reference.npz", interval_start_s=np.arange(n)*DT,
            desired_speed_mps=desired, reference_waypoints=np.asarray(route["waypoints"]),
            reference_stations=np.asarray(route["stations"]), reference_speeds=np.asarray(route["speeds"]),
            reference_headings=np.asarray(route["headings"]))
        require(all(sha(source/name) == value for name, value in source_hashes.items()), "Source changed during collection")
        require(sha(args.case) == contract["case_sha256"] and sha(args.route) == contract["route_sha256"], "Input changed during collection")
        summary = {"schema": "f104_collection_completed_v1", "scene_id": case["id"], "split": case["split"],
            "status": outcome["status"], "actual_elapsed_s": n*DT, "actual_elapsed_h": n*DT/3600.,
            "interval_count": n, "terminal_endpoint_recorded": True, "future_padding": False,
            "requested_horizon_s": args.horizon_s, "horizon_reached": abs(n*DT-args.horizon_s) < 1e-8,
            "early_stop_policy": args.f104_policy.config(), "stop_events": args.f104_policy.events,
            "native_height_valid": args.f104_policy.native_height_report["passed"],
            "initial_state_valid": args.f104_policy.initial_state_report["passed"],
            "outcome_contact_scope": "Original outcome asset_contact excludes chassis. Use rich_intervals max_chassis_contact_resultant_n OR max_asset_contact_max_resultant_n for physical contact labels.",
            "additional_normals_scope": "Geometric RigidTerrain.GetNormal under measured wheel hubs; projected tire force is a derived quantity, not an internal tire normal load/contact-pair measurement.",
            "mechanical_work_is_not_fuel": True, "wall_s_including_finalization": time.time()-started,
            "collection_request_sha256": sha(out/"collection_request.json")}
        dump(out/"f104_episode.json", summary)
        artifacts = {str(path.relative_to(out)): sha(path) for path in sorted(out.iterdir()) if path.is_file() and path.suffix != ".log"}
        dump(out/"episode_complete.json", {"schema": "f104_episode_complete_v1", "actual_elapsed_s": n*DT,
            "artifacts_sha256": artifacts, "request_sha256": sha(out/"collection_request.json")})
        print(json.dumps({"out": str(out), "status": outcome["status"], "actual_elapsed_s": n*DT, "complete": True}))
    except Exception as exc:
        dump(out/"collection_failure.json", {"type": type(exc).__name__, "error": str(exc),
            "count_toward_completed_hours": False, "stop_events": args.f104_policy.events})
        raise


if __name__ == "__main__":
    main()
