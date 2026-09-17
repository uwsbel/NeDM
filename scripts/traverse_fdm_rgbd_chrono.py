#!/usr/bin/env python3
"""Real RGB-D observations and paired Chrono executions for the FDM pilot.

The observe/execute processes may read simulation geometry. The plan process
receives only a saved RGB-D observation, causal proprioception, goal and
geometric reference families. It never opens an arena BMP or asset manifest.
No training occurs here. Run physics and rendering on one AMD Chrono/Vulkan
runtime for every arm of a comparison; legacy OptiX observations need a
different depth-ray calibration.
"""
from __future__ import annotations

import argparse
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
CAMERA = {"width": 256, "height": 256, "hfov_rad": math.radians(47),
          "cam_height_m": 100., "depth_ray_scale": 1., "max_depth_m": 250.,
          "backend": "Vulkan_RT_lavapipe", "depth_measurement": "Euclidean_ray_range_m"}
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


def make_cases(out):
    """Explicit development scenarios, declared before new model scores exist.

    Terrain/asset choices construct the benchmark, never candidate generation.
    f111 has already been examined; neither case is an untouched test arena.
    """
    common = {"horizon_s": 20., "goal_radius_m": 2.5,
              "family_parameters": {"speeds": [4., 6.], "offsets": [0., -8., 8.]},
              "role": "predeclared development demonstration; no unseen-test claim"}
    rock = {"kind": "rock", "x_m": -10., "y_m": -28., "yaw_rad": 0.,
            "footprint_radius_m": math.sqrt(2.), "dims": {"edge_m": 2., "height_m": 1.5}}
    cases = [
        {**common, "id": "visible_rock_v1", "arena": "assets/traverse/arena_v1",
         "goal_xy": [18., -28.],
         "layout": {"episode_id": "fdm_rgbd_visible_rock_v1", "seed": 20260908,
                    "assets": [rock], "house_xy": [18., -28.], "house_yaw": 0.,
                    "start_xy": [-18., -28.], "start_yaw": 0.},
         "construction_note": "A visible 2 m rock is 8 m ahead of the start; three geometric route families must distinguish collision risk. No goal house is placed."},
        {**common, "id": "hill0_f111_near", "arena": "assets/traverse/arena_f111",
         "goal_xy": [-26.943053993768274, 20.62465651186841],
         "layout": {"episode_id": "fdm_rgbd_hill0_f111_near", "seed": 20260908,
                    "assets": [], "house_xy": [-26.943053993768274, 20.62465651186841],
                    "house_yaw": math.pi/2,
                    "start_xy": [-26.943053993768274, -15.375343488131591],
                    "start_yaw": math.pi/2},
         "construction_note": "Previously examined f111 hill0 terrain, start moved to 10 m south of hill center so its rising face falls in the short planning horizon. The old goal house is omitted to isolate terrain behavior."},
    ]
    out = Path(out)
    for case in cases:
        case["settle_reference"] = geometric_line(case["layout"]["start_xy"], case["goal_xy"])
        dump(out / f"{case['id']}.json", case)
    dump(out / "cases.json", {"cases": [f"{c['id']}.json" for c in cases],
         "protocol": "Six geometric references per case; no oracle route filtering. Learned scores are not physical outcomes.",
         "camera": CAMERA, "driver": DRIVER})
    print(json.dumps({"out": str(out), "cases": [c["id"] for c in cases]}))


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


def write_observation(out, scene, state, pose, goal, goal_radius, camera, case_id):
    from nedm.traverse.fdm_data import build_history
    from nedm.traverse.fdm_rgbd_data import rgbd_from_arrays

    scene.manager.Update()
    rgb, depth = scene.rgb_tap.take(), scene.depth_tap.take()
    history = build_history(np.asarray(state, np.float32)[None], np.array([[0., 0., 1.]], np.float32),
                            np.asarray(pose, np.float32)[None], 0)
    rgbd = rgbd_from_arrays(rgb, depth_m=depth, camera=camera)
    out.mkdir(parents=True, exist_ok=True)
    # This is the complete allowed planner input artifact. No layout, assets,
    # arena metadata, terrain samples or future telemetry are stored here.
    np.savez_compressed(out / "observation.npz", rgb=rgb, depth_m=depth,
        rgbd=rgbd, state=np.asarray(state, np.float32), pose=np.asarray(pose, np.float32),
        history=history, goal_xy=np.asarray(goal, np.float32), goal_radius_m=np.float32(goal_radius),
        elapsed_s=np.float32(0.))
    from PIL import Image
    Image.fromarray(rgb).save(out / "rgb.png")
    dump(out / "observation.json", {"case_id": case_id, "camera": camera,
         "proprioception": "17 measured state fields at t=0; XY/yaw are simulator localization",
         "history": "Same startup convention as reused training: repeated current state and synthetic previous brake=1; no measured settle history claimed",
         "observation_sha256": sha256(out / "observation.npz"),
         "valid_depth_fraction": float(np.mean(np.isfinite(depth) & (depth > 0.) & (depth < 250.))),
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
    render = RenderSpec(width=256, height=256, plan_markers=False)
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
    driver = make_driver(chrono, veh, vehicle, route, height_source)
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
    while frame < frames:
        ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
        pos = np.array([ref.GetPos().x, ref.GetPos().y])
        wp = nearest_index(xy, pos, wp)
        at_end = wp >= len(xy)-2 and np.linalg.norm(pos-xy[-1]) < 3.
        driver.SetDesiredSpeed(0. if frame < 0 or at_end else float(speed[wp]))
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
                    observation = write_observation(out, scene, state, pose, goal,
                        float(case.get("goal_radius_m", 2.5)), camera, case["id"])
                    if args.command == "observe":
                        dump(out / "simulation_provenance.json", simulation_provenance(chrono, case_path, arena, args, dt))
                        print(json.dumps({"out": str(out), "case": case["id"], "observed": True, "wall_s": time.time()-wall0}))
                        return
                    if args.command == "execute":
                        with np.load(args.anchor_observation) as expected:
                            equality = {
                                "state17_max_abs_difference": float(np.abs(state-expected["state"]).max()),
                                "pose_max_abs_difference": float(np.abs(pose-expected["pose"]).max()),
                                "rgb_equal": bool(np.array_equal(observation["rgb"], expected["rgb"])),
                                "depth_max_abs_difference_m": float(np.abs(observation["depth_m"]-expected["depth_m"]).max()),
                                "anchor_observation_sha256": sha256(args.anchor_observation),
                            }
                            equality["matched"] = bool(
                                np.allclose(state, expected["state"], rtol=1e-6, atol=1e-5)
                                and np.allclose(pose, expected["pose"], rtol=0., atol=2e-6)
                                and equality["rgb_equal"] and equality["depth_max_abs_difference_m"] <= 1e-4)
                        dump(out / "anchor_equality.json", equality)
                        if not equality["matched"]:
                            raise RuntimeError("Execution anchor differs from scored context; refusing a mismatched paired trial")
                if args.record_rgbd_stride > 0 and frame % args.record_rgbd_stride == 0:
                    if frame == 0:
                        rgb, depth = observation["rgb"], observation["depth_m"]
                    else:
                        scene.manager.Update()
                        rgb, depth = scene.rgb_tap.take(), scene.depth_tap.take()
                    image_anchors.append(frame)
                    image_rgb.append(rgb.copy())
                    image_depth.append(depth.copy())
                record_state.append(state)
                record_action.append(action)
                record_pose.append(pose)
                record_power.append(power)
                # Optional execution visualization; default collection and
                # model camera behavior do not create or call this hook.
                if getattr(args, "frame_observer", None) is not None:
                    args.frame_observer.on_frame(scene, frame, state, pose, action)
            if frame >= 0:
                p = float(engine.GetOutputMotorshaftTorque()) * float(transmission.GetOutputMotorshaftSpeed())/1000.
                frame_positive_work += max(p, 0.)*dt
            driver.Advance(dt)
            terrain.Advance(dt)
            hmmwv.Advance(dt)
            if frame >= 0:
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
    scene.manager.Update()
    from PIL import Image
    Image.fromarray(scene.rgb_tap.take()).save(out / "terminal_rgb.png")
    # Consume the paired depth buffer too; this is an observation-only artifact.
    terminal_depth = scene.depth_tap.take()
    np.save(out / "terminal_depth_m.npy", terminal_depth)
    if getattr(args, "frame_observer", None) is not None:
        args.frame_observer.finish(scene, len(states), terminal_state, terminal_pose, actions[-1])
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
        "measurement_scope": "Actual Chrono execution of a fixed selected reference; this is not a receding-horizon replan result."}
    dump(out / "outcome.json", summary)
    if args.command == "collect":
        dump(out / "collection_meta.json", {"case_id": case["id"], "split": case["split"],
            "route": route, "frame_dt_s": DT, "camera": camera, "driver": args.execution_driver,
            "status": status, "frames": len(states), "image_frame_indices": image_anchors,
            "case_sha256": sha256(case_path), "route_sha256": sha256(args.route),
            "telemetry_timing": "state/action/pose row i precedes interval[i,i+1); contact and exact positive work describe that interval",
            "terminal_pose_timing": "after the last recorded interval; separate endpoint, not an extra state row",
            "image_timing": "current RGB-D captured at its integer telemetry anchor; future images never substitute for missing anchors"})
    dump(out / "contact_events.json", {"columns": ["frame", "substep", "asset_index", "force_n"], "events": contact_events})
    dump(out / "simulation_provenance.json", simulation_provenance(chrono, case_path, arena, args, dt))
    print(json.dumps(summary))


def simulation_provenance(chrono, case_path, arena, args, dt):
    texture = ROOT / "chrono/data/sensor/textures/grass_texture.jpg"
    return {"host": platform.node(), "python": sys.version, "pychrono": chrono.__file__,
            "chrono_version": str(getattr(chrono, "CHRONO_VERSION", "unreported")),
            "case_sha256": sha256(case_path), "arena_meta_sha256": sha256(arena / "arena_meta.json"),
            "script_sha256": sha256(__file__), "scene_code_sha256": sha256(ROOT / "src/nedm/traverse/scene.py"),
            "physics_dt_s": dt, "camera": {**CAMERA, "backend": args.backend, "depth_ray_scale": args.depth_ray_scale},
            "driver": getattr(args, "execution_driver", DRIVER), "vk_driver_files": os.environ.get("VK_DRIVER_FILES"),
            "terrain_texture": str(texture.resolve()) if texture.exists() else "solid_color_fallback",
            "terrain_texture_sha256": sha256(texture) if texture.exists() else None,
            "privilege_boundary": "Terrain and asset truth used only for scene construction, default controller path altitude and outcome measurement. Optional current_depth controller path altitude uses only the scored observation. Planner process opens observation.npz only."}


def plan(args):
    """Image-only planning entry point; no simulation or geometry imports."""
    from nedm.traverse.fdm_rgbd_model import load_rgbd_checkpoint
    from nedm.traverse.fdm_rgbd_planner import RGBDCostConfig, plan_rgbd_routes, propose_route_families
    from nedm.traverse.fdm_mppi import MPPIConfig

    with np.load(args.observation) as data:
        observation = {key: data[key].copy() for key in data.files}
    model, checkpoint = load_rgbd_checkpoint(args.checkpoint, device=args.device)
    families = propose_route_families(observation["pose"], observation["goal_xy"],
        speeds=tuple(args.speeds), offsets=tuple(args.offsets))
    result = plan_rgbd_routes(model, observation["rgbd"], observation["history"],
        observation["pose"], observation["goal_xy"], elapsed_s=float(observation["elapsed_s"]),
        seed=args.seed, families=families,
        mppi_config=MPPIConfig(samples=args.samples, iterations=args.iterations),
        cost_config=RGBDCostConfig(goal_radius_m=(float(observation.get("goal_radius_m", 2.5))
            if args.goal_radius_m is None else args.goal_radius_m)), control=args.control)
    out = Path(args.out)
    dump(out / "selection.json", result)
    dump(out / "families.json", families)
    for index, route in enumerate(families):
        dump(out / "routes" / f"family_{index:02d}.json", route)
    if result.get("route") is not None:
        dump(out / "routes" / "selected.json", result["route"])
    dump(out / "planning_provenance.json", {"observation_sha256": sha256(args.observation),
        "checkpoint_sha256": sha256(args.checkpoint), "seed": args.seed,
        "image_intervention": args.control,
        "inputs": "saved RGB-D, current pose, causal startup history, elapsed time, goal, geometric references",
        "forbidden_inputs": "no terrain BMP, asset list, future state/action, physical outcome or simulator construction manifest",
        "interpretation": "Learned predictions and costs; evaluate the saved references in Chrono for actual outcomes."})
    print(json.dumps({"out": str(out), "abstained": result.get("abstained"), "families": len(families)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cases = sub.add_parser("make-cases")
    cases.add_argument("--out", type=Path, required=True)
    for command in ("observe", "execute", "collect"):
        p = sub.add_parser(command)
        p.add_argument("--case", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--chrono-data", help="Explicit runtime data directory; never changes the shared checkout")
        p.add_argument("--backend", choices=("Vulkan_RT_lavapipe", "OptiX_legacy"), default="Vulkan_RT_lavapipe")
        p.add_argument("--depth-ray-scale", type=float, default=1., help="Vulkan=1; legacy OptiX calibration must be passed explicitly")
        p.add_argument("--horizon-s", type=float)
        p.add_argument("--record-rgbd-stride", type=int, default=20 if command == "collect" else 0)
        if command in ("execute", "collect"):
            p.add_argument("--route", required=True)
        if command == "execute":
            p.add_argument("--anchor-observation", required=True, help="Scored common observation; must match before physical execution")
            p.add_argument("--path-height-source", choices=("truth", "current_depth"), default="truth",
                           help="Matched standard driver uses terrain truth; optional controller-only ablation uses the scored depth image")
    p = sub.add_parser("plan")
    p.add_argument("--observation", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--speeds", type=float, nargs="+", default=[4., 6.])
    p.add_argument("--offsets", type=float, nargs="+", default=[0., -8., 8.])
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--goal-radius-m", type=float, help="Defaults to the tolerance saved in the observation")
    p.add_argument("--control", choices=("normal", "blank"), default="normal")
    args = parser.parse_args()
    if args.command == "make-cases":
        make_cases(args.out)
    elif args.command == "plan":
        plan(args)
    else:
        if args.backend == "OptiX_legacy" and np.isclose(args.depth_ray_scale, 1.):
            parser.error("Legacy OptiX requires its explicitly measured depth-ray scale, not Vulkan's scale=1")
        run_chrono(args)


if __name__ == "__main__":
    main()
