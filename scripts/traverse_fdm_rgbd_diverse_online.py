#!/usr/bin/env python3
"""Measured-history FDM / MPPI with native Chrono PID.

The observe/execute processes may read simulation geometry. The plan process
receives only a saved RGB-D observation, causal proprioception, goal and
geometric reference families. It never opens an arena BMP or asset manifest.
No training occurs here. Run physics and rendering on one AMD Chrono/Vulkan
runtime for every arm of a comparison; legacy OptiX observations need a
different depth-ray calibration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DT = .05
SETTLE_S = .8
CAMERA = {"width": 1024, "height": 1024, "hfov_rad": math.radians(47),
          "cam_height_m": 400., "depth_ray_scale": 1., "max_depth_m": 600.,
          "backend": "Vulkan_RT_lavapipe", "depth_measurement": "Euclidean_ray_range_m",
          "model_image_size": 512, "elevation_scale_m": 40.,
          "observation_mode": "single pre-drive global RGB-D snapshot reused causally"}
DRIVER = {"name": "ChPathFollowerDriver", "steering_gains": [.8, 0., 0.],
          "speed_gains": [.6, .05, 0.], "lookahead_m": 5., "steer_rate_per_s": 2.,
          "path_min_spacing_m": 2., "path_z": "privileged terrain height + 0.5 m",
          "initialization": "before 0.8 s settle; steering held straight during settle",
          "control_period_s": DT, "advance": "every physics substep"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=serializable, allow_nan=False) + "\n")


def geometric_line(start, goal, speed=4., step=.5):
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    distance = float(np.linalg.norm(goal-start))
    if distance < 1.:
        raise ValueError("Need a nontrivial start-goal separation")
    count = max(3, int(np.ceil(distance/step))+1)
    xy = np.linspace(start, goal, count)
    yaw = math.atan2(goal[1]-start[1], goal[0]-start[0])
    return {"waypoints": xy.tolist(), "stations": np.linspace(0., distance, count).tolist(),
            "speeds": [float(speed)] * count, "headings": [yaw] * count,
            "meta": {"family": "geometric_straight", "fdm_station": 0.}}


def read_route(path):
    value = json.loads(Path(path).read_text())
    if "route" in value:
        value = value["route"]
    if value is None:
        raise ValueError("Planner abstained: no executable route")
    xy, speeds = np.asarray(value["waypoints"], float), np.asarray(value["speeds"], float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or speeds.shape != (len(xy),):
        raise ValueError("Invalid reference shape")
    station = np.r_[0., np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
    if np.any(np.diff(station) <= 0.) or not np.isfinite(xy).all() or not np.isfinite(speeds).all():
        raise ValueError("Nonfinite or duplicate reference")
    if np.any(speeds < 0.) or np.any(speeds > 10.):
        raise ValueError("Reference speeds outside the declared forward-driving range")
    if not np.allclose(station, value["stations"], atol=1e-4):
        raise ValueError("Reference station array disagrees with its geometry")
    return value


def make_driver(chrono, veh, vehicle, route, tmap):
    """Exact standard collector reference construction, with disclosed height privilege."""
    points = chrono.vector_ChVector3d()
    last_station = -10.
    for (x, y), station in zip(route["waypoints"], route["stations"]):
        if station-last_station < 2. and station != route["stations"][-1]:
            continue
        last_station = station
        points.append(chrono.ChVector3d(float(x), float(y), float(tmap.height(x, y))+.5))
    driver = veh.ChPathFollowerDriver(vehicle, chrono.ChBezierCurve(points), "route", float(route["speeds"][0]))
    driver.GetSteeringController().SetLookAheadDistance(5.)
    driver.GetSteeringController().SetGains(.8, 0., 0.)
    driver.GetSpeedController().SetGains(.6, .05, 0.)
    driver.Initialize()
    return driver


def nearest_index(xy, pos, previous):
    last = min(len(xy), previous+60)
    return previous+int(np.argmin(np.linalg.norm(xy[previous:last]-pos, axis=1)))


def planning_due(frame, replan_stride, at_end, planning_mode="receding"):
    """Plan once at launch, or retain the original periodic online schedule."""
    if planning_mode not in ("receding", "plan_once"):
        raise ValueError(f"Unknown planning mode {planning_mode}")
    if frame < 0 or at_end:
        return False
    return frame == 0 if planning_mode == "plan_once" else frame % replan_stride == 0


def write_observation(out, scene, state, pose, goal, goal_radius, camera, case_id):
    from nedm.traverse.fdm_data import build_history
    from nedm.traverse.fdm_diverse_data import encode_global_rgbd

    scene.manager.Update()
    rgb, depth = scene.rgb_tap.take(), scene.depth_tap.take()
    history = build_history(np.asarray(state, np.float32)[None], np.array([[0., 0., 1.]], np.float32),
                            np.asarray(pose, np.float32)[None], 0)
    rgbd = encode_global_rgbd(rgb, depth, camera)
    out.mkdir(parents=True, exist_ok=True)
    # This is the complete allowed planner input artifact. No layout, assets,
    # arena metadata, terrain samples or future telemetry are stored here.
    np.savez_compressed(out / "observation.npz", rgb=rgb, depth_m=depth,
        rgbd=rgbd, state=np.asarray(state, np.float32), pose=np.asarray(pose, np.float64),
        history=history, goal_xy=np.asarray(goal, np.float32), goal_radius_m=np.float32(goal_radius),
        elapsed_s=np.float32(0.))
    from PIL import Image
    Image.fromarray(rgb).save(out / "rgb.png")
    dump(out / "observation.json", {"case_id": case_id, "camera": camera,
         "proprioception": "17 measured state fields at t=0; XY/yaw are simulator localization",
         "history": "Same startup convention as reused training: repeated current state and synthetic previous brake=1; no measured settle history claimed",
         "observation_sha256": sha256(out / "observation.npz"),
         "valid_depth_fraction": float(np.mean(np.isfinite(depth) & (depth > 0.) & (depth < camera["max_depth_m"]))),
         "planner_input_boundary": "RGB-D, causal proprioception and supplied goal only; no BMP or authored asset geometry"})
    return {"rgb": rgb, "depth_m": depth, "rgbd": rgbd, "history": history}


def run_chrono(args):
    import pychrono as chrono
    import pychrono.vehicle as veh
    from nedm.hmmwv_data import WHEEL_SPECS, capture_row
    from nedm.training.constants import STATE_FIELD_PRESETS
    from nedm.traverse.layout import EpisodeLayout
    from nedm.traverse.scene import RenderSpec, build_config, build_scene
    from nedm.traverse.terrain import TerrainMap

    case_path = Path(args.case).resolve()
    case = json.loads(case_path.read_text())
    route = case["settle_reference"] if args.command == "observe" else read_route(args.route)
    arena = (ROOT / case["arena"]).resolve()
    layout = EpisodeLayout.from_json(case["layout"])
    tmap = TerrainMap.from_dir(arena)
    config = build_config(arena, (*layout.start_xy, float(tmap.height(*layout.start_xy))+.75), layout.start_yaw)
    if args.chrono_data:
        config["chrono_data_root"] = str(Path(args.chrono_data).resolve())
        config["vehicle_data_root"] = str(Path(args.chrono_data).resolve() / "vehicle")
    camera = {**CAMERA, "backend": args.backend, "depth_ray_scale": args.depth_ray_scale}
    render = (RenderSpec(width=camera["width"], height=camera["height"],
        cam_height_m=camera["cam_height_m"], hfov_rad=camera["hfov_rad"],
        max_depth_m=camera["max_depth_m"], plan_markers=False)
        if args.command == "observe" or args.render_parity else None)
    scene = build_scene(config, layout, tmap, arena, plan=None, render=render)
    hmmwv, system, terrain = scene.hmmwv, scene.system, scene.terrain
    vehicle = hmmwv.GetVehicle()
    engine, transmission = vehicle.GetEngine(), vehicle.GetTransmission()
    height_source = tmap
    args.execution_driver = dict(DRIVER)
    if getattr(args, "path_height_source", "truth") == "current_depth":
        from nedm.traverse.fdm_rgbd_controller_height import ObservedDepthHeight
        with np.load(args.anchor_observation) as scored_observation:
            height_source = ObservedDepthHeight(scored_observation["depth_m"], camera)
        args.execution_driver["path_z"] = "nearest current measured depth surface + 0.5 m; no terrain fallback"
    from nedm.traverse.fdm_online_driver import OnlinePathFollower
    driver = OnlinePathFollower(chrono, veh, vehicle, route, height_source)
    if height_source is not tmap:
        dump(Path(args.out) / "controller_depth_height.json", {
            "anchor_observation_sha256": sha256(args.anchor_observation),
            "max_observed_xy_distance_m": height_source.max_xy_distance_m,
            "limitation": "Visible surface may include occluding vehicle/obstacle; ground beneath occlusions is unknown",
            "path_samples": height_source.queries})
    tire_radii = {name: float(vehicle.GetTire(axle, side).GetRadius()) for name, axle, side in WHEEL_SPECS}
    fields = STATE_FIELD_PRESETS["tire_normal_force_omega_pt"]
    xy, speed = np.asarray(route["waypoints"], float), np.asarray(route["speeds"], float)
    goal = np.asarray(case["goal_xy"], float)
    if np.linalg.norm(xy[0]-np.asarray(layout.start_xy)) > .25 or np.linalg.norm(xy[-1]-goal) > .25:
        raise ValueError("Paired from-rest benchmark requires the declared common start and goal")
    dt = float(config["simulation"]["step_size_s"])
    substeps = int(round(DT/dt))
    frame, previous_steer, wp = -int(round(SETTLE_S/DT)), 0., 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    horizon = float(case.get("horizon_s", 20.)) if args.horizon_s is None else args.horizon_s
    frames = int(round(horizon/DT))
    record_state, record_action, record_pose, record_power, record_contact = [], [], [], [], []
    record_parked, record_interval_work = [], []
    image_anchors, image_rgb, image_depth = [], [], []
    contact_events = []
    positive_work_kj, maximum_contact = 0., 0.
    consecutive_slow, longest_slow = 0, 0
    status, goal_time, first_contact = "timeout", None, None
    wall0 = time.time()
    anchor_pose = None
    observation = None
    terminal_pose = None
    from nedm.traverse.fdm_data import build_history
    from nedm.traverse.fdm_diverse_model import load_rgbd_checkpoint
    from nedm.traverse.fdm_diverse_planner import RGBDCostConfig, plan_rgbd_routes, propose_route_families
    from nedm.traverse.fdm_online_candidates import online_route_families, POLICY as CANDIDATE_POLICY
    from nedm.traverse.fdm_mppi import MPPIConfig
    with np.load(args.scene_observation) as stored:
        map_observation = {key: stored[key].copy() for key in stored.files}
    model, checkpoint = load_rgbd_checkpoint(args.checkpoint, device=args.planning_device)
    observation_dir = Path(args.scene_observation).parent
    observation_meta = json.loads((observation_dir / "observation.json").read_text())
    observation_provenance = json.loads((observation_dir / "simulation_provenance.json").read_text())
    current_provenance = simulation_provenance(chrono, case_path, arena, args, dt)
    if observation_meta["case_id"] != case["id"] or observation_meta["observation_sha256"] != sha256(args.scene_observation):
        raise ValueError("Observation identity/hash disagrees with the declared scene")
    for key in ("case_sha256", "arena_meta_sha256", "arena_bmp_sha256", "runtime_sha256"):
        if observation_provenance[key] != current_provenance[key]:
            raise ValueError(f"Observation/execution {key} differs")
    if map_observation["rgbd"].shape != (4, model.config.image_size, model.config.image_size):
        raise ValueError("Stored map resolution differs from the checkpoint")
    for actual, expected in ((observation_meta["camera"]["cam_height_m"], model.config.camera_height_m),
                             (observation_meta["camera"]["hfov_rad"], math.radians(model.config.camera_hfov_deg)),
                             (observation_meta["camera"]["elevation_scale_m"], model.config.elevation_scale_m)):
        if not np.isclose(actual, expected, atol=1e-8, rtol=0.):
            raise ValueError("Stored camera calibration differs from the checkpoint")
    cost_values = json.loads(Path(args.cost_config).read_text()) if args.cost_config else {}
    cost_values["goal_radius_m"] = float(case.get("goal_radius_m", 3.))
    if args.cost_mode == "time_risk":
        cost_values["energy_weight_s_per_kj"] = 0.
    costs = RGBDCostConfig(**cost_values)
    mppi = MPPIConfig(samples=args.mppi_samples, iterations=args.mppi_iterations,
        max_speed_mps=6., arena_half_extent_m=120., knots=5,
        lateral_sigma_m=.6, max_lateral_m=2., path_step_m=.5,
        max_curvature_inv_m=CANDIDATE_POLICY["max_reference_curvature_inv_m"])
    replan_stride = int(round(args.replan_period_s / DT))
    if replan_stride < 1 or not np.isclose(replan_stride * DT, args.replan_period_s):
        raise ValueError("Replan period must be an integer telemetry interval")
    paused, planning_rows = False, []
    original_families = propose_route_families(map_observation["pose"], goal,
        speeds=tuple(args.speeds), offsets=tuple(args.offsets), step_m=.5)
    dump(out / "online_protocol.json", {
        "model_checkpoint_sha256": sha256(args.checkpoint), "checkpoint_step": checkpoint["step"],
        "scene_observation_sha256": sha256(args.scene_observation),
        "planning_mode": args.planning_mode,
        "planning_schedule": ("One existing RGB-D/MPPI decision at frame 0; the selected full reference and its speed profile remain fixed for native PID execution. Initial abstention pauses the whole trial."
            if args.planning_mode == "plan_once" else "Periodic measured-state RGB-D/MPPI decisions and native PID reference updates"),
        "replan_period_s": args.replan_period_s, "planning_device": args.planning_device,
        "seed": args.seed, "cost_mode": args.cost_mode, "image_intervention": args.image_intervention,
        "cost_config": asdict(costs), "mppi_config": asdict(mppi), "candidate_speeds_mps": args.speeds,
        "candidate_offsets_m": args.offsets, "native_controller": DRIVER,
        "candidate_policy": CANDIDATE_POLICY,
        "model_input_boundary": "Fixed pre-drive RGB-D, current measured pose/state/history and supplied goal/reference only",
        "planning_time": "Wall time measured separately; physics is paused during decision computation",
        "request_timing": "Measured input/state precede planning. New reference and desired speed enter native driver.Advance; first affected applied input is next 2ms physics substep.",
        "attitude_risk_scope": "0.2s predicted endpoints; physical substep maxima measured separately",
        "work_scope": "Positive engine-interface mechanical work, not fuel energy",
        "source_sha256": {name: sha256(ROOT/name) for name in (
            "scripts/traverse_fdm_rgbd_diverse_online.py", "src/nedm/traverse/fdm_online_driver.py",
            "src/nedm/traverse/fdm_diverse_model.py", "src/nedm/traverse/fdm_diverse_planner.py",
            "src/nedm/traverse/fdm_diverse_data.py", "src/nedm/traverse/fdm_online_candidates.py")}})
    while frame < frames:
        ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
        pos = np.array([ref.GetPos().x, ref.GetPos().y])
        wp = nearest_index(xy, pos, wp)
        at_end = wp >= len(xy)-2 and np.linalg.norm(pos-xy[-1]) < 3.
        driver.SetDesiredSpeed(0. if frame < 0 or at_end or paused else float(speed[wp]))
        frame_contact, frame_positive_work = 0., 0.
        for sub in range(substeps):
            ts = float(system.GetChTime())
            driver.Synchronize(ts)
            inputs = driver.GetInputs()
            previous_steer = 0. if frame < 0 else float(np.clip(inputs.m_steering, previous_steer-2.*dt, previous_steer+2.*dt))
            inputs.m_steering = previous_steer
            terrain.Synchronize(ts)
            hmmwv.Synchronize(ts, inputs, terrain)
            if sub == 0 and frame >= 0:
                row = capture_row(hmmwv, terrain, "fdm_rgbd_eval", "follower", case["id"], "development",
                                  frame, ts, inputs, include_tires=True, tire_radii=tire_radii)
                row["engine_motor_speed_radps"] = float(engine.GetMotorSpeed())
                row["engine_motorshaft_torque_nm"] = float(engine.GetOutputMotorshaftTorque())
                state = np.array([float(row[field]) for field in fields], np.float32)
                pose = np.array([row["pos_x_m"], row["pos_y_m"], row["yaw_rad"]], np.float64)
                action = np.array([inputs.m_steering, inputs.m_throttle, inputs.m_braking], np.float32)
                power = float(engine.GetOutputMotorshaftTorque()) * float(transmission.GetOutputMotorshaftSpeed())/1000.
                if frame == 0:
                    anchor_pose = pose.copy()
                    from nedm.traverse.fdm_data import build_history
                    history = build_history(state[None], np.array([[0., 0., 1.]], np.float32), pose[None], 0)
                    np.savez_compressed(out / "anchor_state.npz", state=state, pose=pose,
                        history=history, goal_xy=goal, goal_radius_m=np.float32(case.get("goal_radius_m", 2.5)))
                    if args.command == "observe":
                        observation = write_observation(out, scene, state, pose, goal,
                            float(case.get("goal_radius_m", 2.5)), camera, case["id"])
                        frame_ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
                        marker = frame_ref.TransformPointLocalToParent(chrono.ChVector3d(.1, 0., .95))
                        position = frame_ref.GetPos()
                        dump(out / "sensor_validation_truth.json", {
                            "scope": "Simulator camera verification only; excluded from model inputs",
                            "roof_marker_world_xyz_m": [marker.x, marker.y, marker.z],
                            "vehicle_position_world_xyz_m": [position.x, position.y, position.z],
                            "camera": camera})
                    elif args.render_parity:
                        # One passive render for a same-build physics parity benchmark.
                        scene.manager.Update()
                        scene.rgb_tap.take()
                        scene.depth_tap.take()
                    if args.command == "observe":
                        dump(out / "simulation_provenance.json", simulation_provenance(chrono, case_path, arena, args, dt))
                        print(json.dumps({"out": str(out), "case": case["id"], "observed": True, "wall_s": time.time()-wall0}))
                        return
                if args.record_rgbd_stride > 0 and frame % args.record_rgbd_stride == 0:
                    if frame == 0:
                        rgb, depth = observation["rgb"], observation["depth_m"]
                    else:
                        scene.manager.Update()
                        rgb, depth = scene.rgb_tap.take(), scene.depth_tap.take()
                    image_anchors.append(frame)
                    image_rgb.append(rgb.copy())
                    image_depth.append(depth.copy())
                if frame == 0:
                    checks = {key: bool(np.allclose(actual, map_observation[key], rtol=0., atol=2e-6 if key=="pose" else 1e-5))
                        for key, actual in (("state", state), ("pose", pose), ("history", history), ("goal_xy", goal))}
                    dump(out / "anchor_equality.json", {"matched": all(checks.values()), "checks": checks,
                        "observation_sha256": sha256(args.scene_observation)})
                    if not all(checks.values()):
                        raise ValueError("Online vehicle launch differs from fixed-map observation")
                if planning_due(frame, replan_stride, at_end, args.planning_mode):
                    states_so_far = np.asarray(record_state + [state], np.float32)
                    poses_so_far = np.asarray(record_pose + [pose], np.float64)
                    actions_so_far = np.asarray(record_action + [action], np.float32)
                    causal_history = build_history(states_so_far, actions_so_far, poses_so_far, len(record_state))
                    rejected_candidates = []
                    families = online_route_families(pose, goal, route, original_families,
                        speeds=tuple(args.speeds), offsets=tuple(args.offsets),
                        rejected=rejected_candidates)
                    plan_start = time.perf_counter()
                    decision = plan_rgbd_routes(model, map_observation["rgbd"], causal_history, pose, goal,
                        elapsed_s=frame*DT, seed=args.seed+len(planning_rows), families=families,
                        mppi_config=mppi, cost_config=costs, control=args.image_intervention)
                    planning_wall = time.perf_counter()-plan_start
                    paused = bool(decision.get("abstained", False))
                    update = None
                    if decision.get("route") is not None:
                        route = decision["route"]
                        update = driver.update_route(route, pose)
                        xy, speed = np.asarray(route["waypoints"], float), np.asarray(route["speeds"], float)
                        wp = int(np.linalg.norm(xy-pose[:2], axis=1).argmin())
                        at_end = np.linalg.norm(pos-goal) <= float(case.get("goal_radius_m", 3.))
                    driver.SetDesiredSpeed(0. if paused or at_end else float(speed[wp]))
                    record = {"frame": frame, "time_s": frame*DT, "anchor_pose": pose,
                        "anchor_state17": state, "history_sha256": hashlib.sha256(causal_history.tobytes()).hexdigest(),
                        "already_synchronized_action": action,
                        "new_desired_speed_mps": 0. if paused or at_end else float(speed[wp]),
                        "new_request_first_applied_time_s": frame*DT+dt,
                        "candidate_references": families,
                        "rejected_candidate_references": rejected_candidates,
                        "planning_wall_s": planning_wall, "paused": paused,
                        "driver_update": update, "decision": decision}
                    dump(out / "decisions" / f"decision_{len(planning_rows):05d}.json", record)
                    planning_rows.append({"frame": frame, "time_s": frame*DT,
                        "planning_wall_s": planning_wall, "abstained": paused,
                        "reference_updated": update is not None,
                        "reference_geometry_changed": bool(update and update.get("geometry_changed", False)),
                        "family": decision.get("selected_family_index"),
                        "rejected_candidate_count": len(rejected_candidates),
                        "candidate_evaluations": decision.get("model_evaluations", 0)})
                record_state.append(state)
                record_action.append(action)
                record_pose.append(pose)
                record_power.append(power)
                # Optional execution visualization; default collection and
                # model camera behavior do not create or call this hook.
                if getattr(args, "frame_observer", None) is not None:
                    args.frame_observer.on_frame(scene, frame, state, pose, action,
                        command_context={"desired_speed_mps": 0. if at_end or paused else float(speed[wp]), "parked": bool(at_end), "planner_paused": bool(paused)})
            if frame >= 0:
                if getattr(args, "frame_observer", None) is not None:
                    args.frame_observer.on_substep(scene, frame, sub, dt,
                        np.array([inputs.m_steering, inputs.m_throttle, inputs.m_braking]))
                p = float(engine.GetOutputMotorshaftTorque()) * float(transmission.GetOutputMotorshaftSpeed())/1000.
                frame_positive_work += max(p, 0.)*dt
            driver.Advance(dt)
            terrain.Advance(dt)
            hmmwv.Advance(dt)
            if frame >= 0:
                if getattr(args, "frame_observer", None) is not None:
                    args.frame_observer.on_post_substep(scene, frame, sub)
                for asset_index, (_, body) in enumerate(scene.asset_bodies):
                    force = float(body.GetContactForce().Length())
                    frame_contact = max(frame_contact, force)
                    if force > 1.:
                        contact_events.append([frame, sub, asset_index, force])
        if frame >= 0:
            positive_work_kj += frame_positive_work
            record_interval_work.append(frame_positive_work)
            record_parked.append(at_end)
            record_contact.append(frame_contact)
            maximum_contact = max(maximum_contact, frame_contact)
            if frame_contact > 1. and first_contact is None:
                first_contact = frame*DT
            ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
            terminal_pose = np.array([ref.GetPos().x, ref.GetPos().y, ref.GetRot().GetCardanAnglesZYX().z])
            reached = np.linalg.norm(terminal_pose[:2]-goal) <= float(case.get("goal_radius_m", 2.5))
            # Consecutive effortful near-zero speed is distinct from net
            # displacement cancellation during rollback. Never count parking.
            slow = not at_end and abs(float(state[0])) < .3 and float(action[1]) > .3
            consecutive_slow = consecutive_slow+1 if slow else 0
            longest_slow = max(longest_slow, consecutive_slow)
            if abs(float(vehicle.GetRoll())) > math.radians(60.) or abs(float(vehicle.GetPitch())) > math.radians(60.):
                status = "rollover"
                break
            if reached:
                status, goal_time = "goal_reached", (frame+1)*DT
                break
        frame += 1
    states, actions, poses = np.asarray(record_state), np.asarray(record_action), np.asarray(record_pose)
    full_poses = np.concatenate([poses, terminal_pose[None]])
    # Measure the real endpoint with the last causally applied inputs. This
    # performs no extra physics advance and invents no terminal state row.
    terminal_time = float(system.GetChTime())
    terrain.Synchronize(terminal_time)
    hmmwv.Synchronize(terminal_time, inputs, terrain)
    endpoint = capture_row(hmmwv, terrain, "fdm_rgbd_eval", "follower", case["id"], "development",
        len(states), terminal_time, inputs, include_tires=True, tire_radii=tire_radii)
    endpoint["engine_motor_speed_radps"] = float(engine.GetMotorSpeed())
    endpoint["engine_motorshaft_torque_nm"] = float(engine.GetOutputMotorshaftTorque())
    terminal_state = np.array([float(endpoint[field]) for field in fields], np.float32)
    terminal_wp = nearest_index(xy, terminal_pose[:2], wp)
    terminal_parked = terminal_wp >= len(xy)-2 and np.linalg.norm(terminal_pose[:2]-xy[-1]) < 3.
    elapsed = len(states)*DT
    distance0 = float(np.linalg.norm(anchor_pose[:2]-goal))
    distance1 = float(np.linalg.norm(terminal_pose[:2]-goal))
    low_progress_windows = []
    for anchor in range(max(0, len(states)-80+1)):
        if not any(record_parked[anchor:anchor+80]) and np.linalg.norm(full_poses[anchor+80, :2]-full_poses[anchor, :2]) < .6 and actions[anchor:anchor+80, 1].mean() > .3:
            low_progress_windows.append(anchor)
    bounded_windows = []
    for anchor in range(max(0, len(states)-40+1)):
        if any(record_parked[anchor:anchor+40]) or not np.all(actions[anchor:anchor+40, 1] > .3):
            continue
        points = full_poses[anchor:anchor+41, :2]
        squared_distance = np.square(points[:, None, :]-points[None, :, :]).sum(-1)
        if float(squared_distance.max()) <= .25**2:
            bounded_windows.append(anchor)
    np.savez_compressed(out / "trajectory.npz", state=states, action=actions, pose=poses,
        terminal_pose=terminal_pose, power_kw=np.asarray(record_power), contact_n=np.asarray(record_contact),
        terminal_state=terminal_state, terminal_parked=np.asarray(terminal_parked, bool),
        parked=np.asarray(record_parked, bool), positive_work_kj_per_interval=np.asarray(record_interval_work),
        state_fields=np.asarray(fields), dt_s=np.float32(DT))
    if image_anchors:
        np.savez_compressed(out / "rgbd_frames.npz", anchor=np.asarray(image_anchors, np.int64),
            rgb=np.asarray(image_rgb, np.uint8), depth_m=np.asarray(image_depth, np.float32),
            camera_json=np.asarray(json.dumps(camera)))
    if getattr(args, "frame_observer", None) is not None:
        args.frame_observer.finish(scene, len(states), terminal_state, terminal_pose, actions[-1])
    # The learned event schema includes terrain/chassis contact in addition to
    # asset contact. Preserve the old asset-only measurements, but judge the
    # declared milestone using the complete, solver-rate observation schema.
    with np.load(out / "rich_intervals.npz") as measured:
        contact_maxima = np.maximum(measured["max_chassis_contact_resultant_n"],
                                    measured["max_asset_contact_max_resultant_n"])
        if not np.isfinite(contact_maxima).all():
            raise ValueError("Online collision outcome has incomplete measured support")
        schema_contact = bool((contact_maxima > 1.).any())
        peak_roll = float(np.degrees(measured["max_abs_roll_rad"]).max())
        peak_pitch = float(np.degrees(measured["max_abs_pitch_rad"]).max())
    summary = {"case_id": case["id"], "status": status, "goal_reached": goal_time is not None,
        "contact_free_goal_reached": goal_time is not None and maximum_contact <= 1.,
        "safe_goal_reached": goal_time is not None and maximum_contact <= 1. and longest_slow*DT < 2. and status != "rollover",
        "goal_time_s": goal_time, "elapsed_s": elapsed, "goal_radius_m": case.get("goal_radius_m", 2.5),
        "initial_goal_distance_m": distance0, "final_goal_distance_m": distance1,
        "goal_progress_m": distance0-distance1,
        "net_displacement_m": float(np.linalg.norm(terminal_pose[:2]-anchor_pose[:2])),
        "path_length_m": float(np.linalg.norm(np.diff(full_poses[:, :2], axis=0), axis=1).sum()),
        "asset_contact": maximum_contact > 1., "max_asset_contact_n": maximum_contact,
        "first_asset_contact_interval_start_s": first_contact,
        "longest_consecutive_effortful_near_zero_speed_s": longest_slow*DT,
        "sustained_near_stop": longest_slow*DT >= 2.,
        "sustained_near_stop_definition": "Contiguous >=2 s recorded intervals with abs(body vx)<0.3 m/s, commanded throttle>0.3, not deliberate parking; no global startup grace",
        "bounded_blockage_v1": bool(bounded_windows),
        "bounded_blockage_v1_windows": bounded_windows,
        "bounded_blockage_v1_definition": "A contiguous 2 s window: maximum pairwise XY separation of its 41 measured endpoints <=0.25 m, all 40 commanded throttle intervals>0.3, no deliberate parking",
        "safe_goal_reached_with_bounded_blockage_v1": goal_time is not None and maximum_contact <= 1. and longest_slow*DT < 2. and not bounded_windows and status != "rollover",
        "low_net_progress_4s_windows": low_progress_windows,
        "low_progress_warning": "Net motion under 0.6 m in 4 s may include rollback; sustained near-zero-speed is reported separately.",
        "positive_work_kj": positive_work_kj, "frames": len(states), "wall_s": time.time()-wall0,
        "route_sha256": sha256(args.route), "case_sha256": sha256(case_path),
        "driver": args.execution_driver, "camera": camera,
        "measurement_scope": ("One launch MPPI decision from fixed pre-drive global RGB-D and measured state; native PID executes the selected full reference without replanning"
            if args.planning_mode == "plan_once" else "Measured-state receding-horizon MPPI with fixed pre-drive global RGB-D and native PID reference updates"),
        "planning_mode": args.planning_mode,
        "planning_decisions": len(planning_rows), "planning_abstentions": sum(r["abstained"] for r in planning_rows),
        "replanning_decisions": sum(r["frame"] > 0 for r in planning_rows),
        "planning_candidate_evaluations": sum(r["candidate_evaluations"] for r in planning_rows),
        "planning_reference_updates": sum(r["reference_updated"] for r in planning_rows),
        "planning_geometry_changes": sum(r["reference_geometry_changed"] for r in planning_rows),
        "planning_wall_s": sum(r["planning_wall_s"] for r in planning_rows),
        "final_reference": route,
        "mechanical_work_definition": "Positive engine output torque times transmission motorshaft feedback speed, substep-integrated; not fuel energy",
        "rich_telemetry_enabled": args.rich_telemetry,
        "render_parity": args.render_parity}
    summary.update({"event_schema": "fdm_diverse_events_v1_asset_or_chassis",
        "schema_contact": schema_contact, "max_schema_contact_n": float(contact_maxima.max()),
        "max_abs_roll_deg": peak_roll, "max_abs_pitch_deg": peak_pitch,
        "schema_rollover": max(peak_roll, peak_pitch) > 60.,
        "schema_safe_goal_reached": goal_time is not None and not schema_contact
            and not bounded_windows and max(peak_roll, peak_pitch) <= 60.,
        "safe_goal_definition": "Reached supplied goal without asset/chassis resultant>1N, bounded2s blockage, or solver-step roll/pitch>60deg",
        "legacy_safe_fields_scope": "Legacy safe_goal_reached fields use asset-only contact; use schema_safe_goal_reached for the diverse milestone"})
    dump(out / "online_planning_summary.json", {"planning_mode": args.planning_mode,
        "planning_decisions": summary["planning_decisions"],
        "replanning_decisions": summary["replanning_decisions"],
        "candidate_evaluations": summary["planning_candidate_evaluations"],
        "reference_updates": summary["planning_reference_updates"],
        "geometry_changes": summary["planning_geometry_changes"],
        "decisions": planning_rows})
    dump(out / "outcome.json", summary)
    if args.command == "collect_unused":
        dump(out / "collection_meta.json", {"case_id": case["id"], "split": case["split"],
            "route": route, "frame_dt_s": DT, "camera": camera, "driver": args.execution_driver,
            "status": status, "frames": len(states), "image_frame_indices": image_anchors,
            "case_sha256": sha256(case_path), "route_sha256": sha256(args.route),
            "telemetry_timing": "state/action/pose row i precedes interval[i,i+1); contact and exact positive work describe that interval",
            "terminal_pose_timing": "after the last recorded interval; separate endpoint, not an extra state row",
            "image_timing": "Separate single pre-drive global observation; reused at every causal anchor. No future image substitutes.",
            "observation_join": "Check same case, runtime and settled anchor against separately rendered snapshot; scene observation required for training"})
    dump(out / "contact_events.json", {"columns": ["frame", "substep", "asset_index", "force_n"], "events": contact_events})
    dump(out / "simulation_provenance.json", simulation_provenance(chrono, case_path, arena, args, dt))
    print(json.dumps(summary, default=serializable, allow_nan=False))


def simulation_provenance(chrono, case_path, arena, args, dt):
    texture = ROOT / "chrono/data/sensor/textures/grass_texture.jpg"
    source_files = [Path(__file__), ROOT / "src/nedm/traverse/scene.py",
        ROOT / "src/nedm/hmmwv_data.py", ROOT / "src/nedm/traverse/terrain.py",
        ROOT / "src/nedm/traverse/layout.py", ROOT / "src/nedm/traverse/fdm_data.py",
        ROOT / "src/nedm/traverse/fdm_diverse_data.py", ROOT / "src/nedm/traverse/fdm_rich_telemetry.py"]
    fingerprint = os.environ.get("FDM_RUNTIME_FINGERPRINT")
    if fingerprint:
        runtime = json.loads(Path(fingerprint).read_text())["runtime_sha256"]
    else:
        # Use the collection inventory, including HMMWV parameter/mesh files.
        # The older standalone runner inventories only modules/libraries; that
        # is not an equal runtime contract even when shared hashes agree.
        from traverse_fdm_rgbd_diverse_batch import runtime_fingerprint
        runtime = runtime_fingerprint(args.chrono_data)["file_sha256"]
    meta = json.loads((arena / "arena_meta.json").read_text())
    return {"host": platform.node(), "python": sys.version, "pychrono": chrono.__file__,
            "chrono_version": str(getattr(chrono, "CHRONO_VERSION", "unreported")),
            "case_sha256": sha256(case_path), "arena_meta_sha256": sha256(arena / "arena_meta.json"),
            "arena_bmp_sha256": sha256(arena / meta["bmp"]),
            "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_files},
            "runtime_sha256": runtime,
            "runtime_fingerprint_sha256": sha256(fingerprint) if fingerprint else None,
            "script_sha256": sha256(__file__), "scene_code_sha256": sha256(ROOT / "src/nedm/traverse/scene.py"),
            "physics_dt_s": dt, "camera": {**CAMERA, "backend": args.backend, "depth_ray_scale": args.depth_ray_scale},
            "driver": getattr(args, "execution_driver", DRIVER), "vk_driver_files": os.environ.get("VK_DRIVER_FILES"),
            "terrain_texture": str(texture.resolve()) if texture.exists() else "solid_color_fallback",
            "terrain_texture_sha256": sha256(texture) if texture.exists() else None,
            "privilege_boundary": "Terrain and asset truth used only for scene construction, default controller path altitude and outcome measurement. Optional current_depth controller path altitude uses only the scored observation. Planner process opens observation.npz only."}


def main():
    parser = argparse.ArgumentParser(description="Online MPPI on a fixed global RGB-D map with native Chrono PID")
    parser.add_argument("--case", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene-observation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chrono-data", required=True)
    parser.add_argument("--horizon-s", type=float, default=180.)
    parser.add_argument("--planning-device", default="cpu")
    parser.add_argument("--planning-mode", choices=("receding", "plan_once"), default="receding")
    parser.add_argument("--replan-period-s", type=float, default=1.)
    parser.add_argument("--cost-mode", choices=("time_risk", "energy_risk"), default="energy_risk")
    parser.add_argument("--cost-config")
    parser.add_argument("--mppi-samples", type=int, default=64)
    parser.add_argument("--mppi-iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--speeds", type=float, nargs="+", default=[2.,4.,6.])
    parser.add_argument("--offsets", type=float, nargs="+", default=[0.,-22.,22.,-44.,44.])
    parser.add_argument("--image-intervention", choices=("normal","blank"), default="normal")
    parser.add_argument("--video-fps", type=float, default=0.)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        parser.error("Use a new empty online execution output directory")
    out.mkdir(parents=True, exist_ok=True)
    case = json.loads(Path(args.case).read_text())
    dump(out / "initial_settle_reference.json", case["settle_reference"])
    args.route = str(out / "initial_settle_reference.json")
    args.command, args.backend, args.depth_ray_scale = "online", "Vulkan_RT_lavapipe", 1.
    args.record_rgbd_stride, args.path_height_source, args.render_parity = 0, "truth", False
    args.rich_telemetry = True
    from nedm.traverse.fdm_rich_telemetry import RichTelemetry
    args.frame_observer = RichTelemetry(out, case_path=args.case, record_dt_s=DT)
    if args.video_fps > 0:
        from traverse_fdm_rgbd_diverse_video import OnlineVideoObserver
        args.frame_observer = OnlineVideoObserver(args.frame_observer, out, fps=args.video_fps)
    run_chrono(args)


if __name__ == "__main__":
    main()
