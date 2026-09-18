#!/usr/bin/env python3
"""One HMMWV route episode on CRM (SPH deformable soil) built from an f104-family arena BMP.

The rigid-terrain collector (scripts/gen_collect.py wrapping traverse_fdm_rgbd_diverse_chrono.run_chrono) is mirrored
statement for statement wherever CRM allows it, and its pieces are IMPORTED rather than copied:

  same   case/route JSON, read_route, make_driver (ChPathFollowerDriver, gains .8 / .6,.05, 5 m look-ahead, path z =
         BMP height + 0.5 m), 50 ms control/record period, 0.8 s braked settle, 2 /s steering rate limit, parking rule,
         goal radius, rollover rule, StopPolicy (terrain bounds, prolonged blockage 24 s + 2 s + 8 s), 120 s horizon,
         (one added terminal status, soil_breakthrough_terminated: a wheel dug through the whole soil layer = bogged),
         trajectory.npz / outcome.json / command_reference.npz / anchor_state.npz / case.json / reference.json schemas,
         17-field state (tire_normal_force_omega_pt), action columns [steering, throttle, braking].
  CRM    terrain = veh.CRMTerrain from the same BMP and height range (fresh soil every process = reset per episode);
         the terrain owns the coupled step (terrain.Advance only, never hmmwv.Advance); rigid-mesh tyres registered as
         FSI bodies; tyre forces read from the FSI solver; physics step from the CRM config instead of 2 ms; the
         native-height raycast audit is replaced by a settled-chassis-height check against the BMP height.

Extra CRM-only measurements go to crm_extra.npz so the shared schemas stay untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CRM_DEFAULT = {
    "schema": "crm_f104_config_v1",
    "spacing_m": 0.08, "depth_m": 0.24, "step_s": 5e-4, "active_domain_m": [2.0, 2.0, 1.0],
    "active_domain_delay_s": 0.0, "side_walls": True, "mbs_threads": 4,
    "soil": {"density": 1700.0, "young_modulus_pa": 1.0e6, "poisson_ratio": 0.3, "mu_I0": 0.04,
             "friction": 0.8, "average_diam_m": 0.005, "cohesion_pa": 5.0e3},
    "sph": {"d0_multiplier": 1.2, "free_surface_threshold": 0.8, "artificial_viscosity": 0.5,
            "shifting_method": "PPST", "shifting_ppst_push": 3.0, "shifting_ppst_pull": 1.0,
            "num_proximity_search_steps": 4},
    "tire_mesh": "hmmwv/hmmwv_tire_coarse_closed.obj",
}


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def merged_config(path):
    cfg = json.loads(json.dumps(CRM_DEFAULT))
    if path:
        user = read(path)
        for key, value in user.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg


def build_crm(chrono, veh, fsi, system, vehicle, arena, meta, cfg):
    dt = float(cfg["step_s"])
    system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
    system.SetNumThreads(int(cfg["mbs_threads"]), 1, 1)
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

    terrain = veh.CRMTerrain(system, float(cfg["spacing_m"]))
    terrain.SetVerbose(False)
    terrain.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    terrain.SetStepSizeCFD(dt)
    terrain.RegisterVehicle(vehicle)

    s = cfg["soil"]
    soil = (getattr(fsi, "SoilProperties", None) or fsi.ElasticMaterialProperties)()  # Chrono main / pychrono 10.0.0
    soil.density = float(s["density"])
    soil.Young_modulus = float(s["young_modulus_pa"])
    soil.Poisson_ratio = float(s["poisson_ratio"])
    soil.mu_I0 = float(s["mu_I0"])
    soil.mu_fric_s = float(s["friction"])
    soil.mu_fric_2 = float(s["friction"])
    soil.average_diam = float(s["average_diam_m"])
    soil.cohesion_coeff = float(s["cohesion_pa"])
    (getattr(terrain, "SetCrmSPH", None) or terrain.SetElasticSPH)(soil)

    p = cfg["sph"]
    sph = fsi.SPHParameters()
    sph.integration_scheme = fsi.IntegrationScheme_RK2
    sph.initial_spacing = float(cfg["spacing_m"])
    sph.d0_multiplier = float(p["d0_multiplier"])
    sph.free_surface_threshold = float(p["free_surface_threshold"])
    sph.artificial_viscosity = float(p["artificial_viscosity"])
    sph.shifting_method = getattr(fsi, "ShiftingMethod_" + str(p["shifting_method"]).upper())
    sph.shifting_ppst_push = float(p["shifting_ppst_push"])
    sph.shifting_ppst_pull = float(p["shifting_ppst_pull"])
    sph.use_consistent_gradient_discretization = False
    sph.use_consistent_laplacian_discretization = False
    sph.viscosity_method = fsi.ViscosityMethod_ARTIFICIAL_BILATERAL
    sph.boundary_method = fsi.BoundaryMethod_ADAMI
    sph.num_proximity_search_steps = int(p["num_proximity_search_steps"])
    terrain.SetSPHParameters(sph)

    geometry = chrono.ChBodyGeometry()
    geometry.coll_meshes.append(chrono.TrimeshShape(chrono.VNULL, chrono.QUNIT,
                                                    veh.GetVehicleDataFile(cfg["tire_mesh"]), chrono.VNULL))
    for axle in vehicle.GetAxles():
        for wheel in axle.GetWheels():
            terrain.AddRigidBody(wheel.GetSpindle(), geometry, False)

    terrain.SetActiveDomain(chrono.ChVector3d(*[float(v) for v in cfg["active_domain_m"]]))
    if hasattr(terrain, "SetFreeFlowDuration"):
        terrain.SetFreeFlowDuration(float(cfg["active_domain_delay_s"]))
    elif hasattr(terrain, "SetActiveDomainDelay"):  # pychrono 10.0.0 name
        terrain.SetActiveDomainDelay(float(cfg["active_domain_delay_s"]))
    size = float(meta["size_m"])
    sides = (fsi.BoxSide_ALL & ~fsi.BoxSide_Z_POS) if cfg["side_walls"] else fsi.BoxSide_Z_NEG
    # Same BMP, same [min, max] grey mapping and the same node-on-edge pixel convention as RigidTerrain; the y
    # orientation was checked against TerrainMap on the initial SPH lattice (rmse = lattice quantisation, 0.047 m at
    # 0.16 m spacing; y-mirrored 1.23 m).
    terrain.Construct(str(arena / meta["bmp"]), size, size,
                      chrono.ChVector2d(float(meta["height_min_m"]), float(meta["height_max_m"])),
                      float(cfg["depth_m"]), True, chrono.ChVector3d(0, 0, 0), sides)
    terrain.Initialize()
    return terrain


def crm_tire_fields(chrono, veh, vehicle, terrain, wheel_specs, tire_radii):
    up = chrono.ChVector3d(0, 0, 1)
    fields = {}
    for name, axle, side in wheel_specs:
        spindle = vehicle.GetWheel(axle, side).GetSpindle()
        force = terrain.GetFsiBodyForce(spindle)
        spin_axis = vehicle.GetSpindleRot(axle, side).GetAxisY()
        heading = spin_axis.Cross(up).GetNormalized()
        lateral = up.Cross(heading)
        omega = float(vehicle.GetSpindleAngVel(axle, side).Dot(spin_axis))
        wheel_vx = float(vehicle.GetSpindleLinVel(axle, side).Dot(heading))
        fields[f"{name}_force_wheel_fx_n"] = float(force.Dot(heading))
        fields[f"{name}_force_wheel_fy_n"] = float(force.Dot(lateral))
        fields[f"{name}_force_wheel_fz_n"] = float(force.Dot(up))
        fields[f"{name}_spindle_omega_radps"] = omega
        fields[f"{name}_wheel_vx_mps"] = wheel_vx
        fields[f"{name}_slip_ratio"] = (omega * tire_radii[name] - wheel_vx) / max(abs(wheel_vx), 0.1)
        fields[f"{name}_spindle_z_m"] = float(vehicle.GetSpindlePos(axle, side).z)
    return fields


def run(args, cfg, policy):
    import pychrono as chrono
    import pychrono.fsi as fsi
    import pychrono.vehicle as veh
    from nedm.hmmwv_data import WHEEL_SPECS, capture_row, configure_chrono_data_paths, create_hmmwv
    from nedm.training.constants import STATE_FIELD_PRESETS
    from nedm.traverse.fdm_data import build_history
    from nedm.traverse.layout import EpisodeLayout
    from nedm.traverse.scene import build_config
    from nedm.traverse.terrain import TerrainMap
    import traverse_fdm_rgbd_diverse_chrono as frozen

    DT, SETTLE_S = frozen.DT, frozen.SETTLE_S
    source = Path(args.source_root).resolve()
    case = read(args.case)
    route = frozen.read_route(args.route)
    arena = (source / case["arena"]).resolve()
    layout = EpisodeLayout.from_json(case["layout"])
    tmap = TerrainMap.from_dir(arena)
    meta = tmap.meta
    dt = float(cfg["step_s"])
    config = build_config(arena, (*layout.start_xy, float(tmap.height(*layout.start_xy)) + .75), layout.start_yaw,
                          step_size_s=dt, tire_step_size_s=dt)
    config["vehicle"]["tire_model"] = "RIGID_MESH"   # FSI wheels; TMEASY cannot see SPH soil
    config["vehicle"]["chassis_collision"] = "NONE"  # no rigid obstacles in this campaign; chassis is not an FSI body
    config["chrono_data_root"] = str(Path(args.chrono_data).resolve())
    config["vehicle_data_root"] = str(Path(args.chrono_data).resolve() / "vehicle")
    configure_chrono_data_paths(source, config)
    hmmwv = create_hmmwv(config)
    vehicle, system = hmmwv.GetVehicle(), hmmwv.GetSystem()
    engine, transmission = vehicle.GetEngine(), vehicle.GetTransmission()
    t_build = time.time()
    terrain = build_crm(chrono, veh, fsi, system, vehicle, arena, meta, cfg)
    build_s = time.time() - t_build
    driver = frozen.make_driver(chrono, veh, vehicle, route, tmap)
    # Optional visualisation hooks (default None = collection behaviour unchanged): sensors and visual shapes only,
    # attached to existing bodies, so they add no bodies and no contacts.
    if getattr(args, "scene_hook", None) is not None:
        args.scene_hook(chrono=chrono, veh=veh, hmmwv=hmmwv, system=system, terrain=terrain, tmap=tmap, route=route, case=case)
    tire_radii = {name: float(vehicle.GetTire(axle, side).GetRadius()) for name, axle, side in WHEEL_SPECS}
    fields = STATE_FIELD_PRESETS["tire_normal_force_omega_pt"]
    xy, speed = np.asarray(route["waypoints"], float), np.asarray(route["speeds"], float)
    goal = np.asarray(case["goal_xy"], float)
    if np.linalg.norm(xy[0] - np.asarray(layout.start_xy)) > .25 or np.linalg.norm(xy[-1] - goal) > .25:
        raise ValueError("Paired from-rest benchmark requires the declared common start and goal")
    substeps = int(round(DT / dt))
    if abs(substeps * dt - DT) > 1e-12:
        raise ValueError("CRM step must divide the 50 ms control period")
    frame, previous_steer, wp = -int(round(SETTLE_S / DT)), 0., 0
    out = Path(args.out)
    frames = int(round(args.horizon_s / DT))
    record_state, record_action, record_pose, record_power = [], [], [], []
    record_parked, record_interval_work, record_extra = [], [], []
    positive_work_kj = 0.
    consecutive_slow, longest_slow = 0, 0
    status, goal_time = "timeout", None
    wall0 = time.time()
    anchor_pose, terminal_pose, inputs = None, None, None
    max_sinkage, deep_frames = 0., 0
    sim_t = 0.

    def measure(frame_index, ts):
        row = capture_row(hmmwv, terrain, "crm_f104", "follower", case["id"], "development",
                          frame_index, ts, inputs, include_tires=False)
        row.update(crm_tire_fields(chrono, veh, vehicle, terrain, WHEEL_SPECS, tire_radii))
        row["engine_motor_speed_radps"] = float(engine.GetMotorSpeed())
        row["engine_motorshaft_torque_nm"] = float(engine.GetOutputMotorshaftTorque())
        return row

    while frame < frames:
        ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
        pos = np.array([ref.GetPos().x, ref.GetPos().y])
        wp = frozen.nearest_index(xy, pos, wp)
        at_end = wp >= len(xy) - 2 and np.linalg.norm(pos - xy[-1]) < 3.
        driver.SetDesiredSpeed(0. if frame < 0 or at_end else float(speed[wp]))
        frame_positive_work = 0.
        for sub in range(substeps):
            ts = sim_t
            driver.Synchronize(ts)
            inputs = driver.GetInputs()
            previous_steer = 0. if frame < 0 else float(np.clip(inputs.m_steering, previous_steer - 2. * dt, previous_steer + 2. * dt))
            inputs.m_steering = previous_steer
            terrain.Synchronize(ts)
            hmmwv.Synchronize(ts, inputs, terrain)
            if sub == 0 and frame >= 0:
                row = measure(frame, ts)
                state = np.array([float(row[field]) for field in fields], np.float32)
                pose = np.array([row["pos_x_m"], row["pos_y_m"], row["yaw_rad"]], np.float64)
                action = np.array([inputs.m_steering, inputs.m_throttle, inputs.m_braking], np.float32)
                power = float(engine.GetOutputMotorshaftTorque()) * float(transmission.GetOutputMotorshaftSpeed()) / 1000.
                if not (np.isfinite(state).all() and np.isfinite(pose).all()):
                    raise FloatingPointError(f"non-finite vehicle state at frame {frame}")
                if frame == 0:
                    anchor_pose = pose.copy()
                    history = build_history(state[None], np.array([[0., 0., 1.]], np.float32), pose[None], 0)
                    np.savez_compressed(out / "anchor_state.npz", state=state, pose=pose, history=history,
                                        goal_xy=goal, goal_radius_m=np.float32(case.get("goal_radius_m", 2.5)))
                    policy.on_anchor_crm(row, state, pose, history, tmap)
                record_state.append(state)
                record_action.append(action)
                record_pose.append(pose)
                record_power.append(power)
                if getattr(args, "frame_hook", None) is not None:
                    args.frame_hook(frame=frame, row=row, state=state, pose=pose, action=action,
                                    desired_speed=0. if at_end else float(speed[wp]))
                ground = float(tmap.height(row["pos_x_m"], row["pos_y_m"]))
                record_extra.append([row["pos_z_m"], ground, row["quat_e0"], row["quat_e1"], row["quat_e2"], row["quat_e3"],
                                     *[row[f"{n}_spindle_z_m"] for n, _, _ in WHEEL_SPECS],
                                     *[row[f"{n}_slip_ratio"] for n, _, _ in WHEEL_SPECS],
                                     *[row[f"{n}_force_wheel_fx_n"] for n, _, _ in WHEEL_SPECS]])
            if frame >= 0:
                p = float(engine.GetOutputMotorshaftTorque()) * float(transmission.GetOutputMotorshaftSpeed()) / 1000.
                frame_positive_work += max(p, 0.) * dt
            driver.Advance(dt)
            terrain.Advance(dt)  # CRM owns the coupled FSI + multibody step
            sim_t += dt
        if frame >= 0:
            positive_work_kj += frame_positive_work
            record_interval_work.append(frame_positive_work)
            record_parked.append(at_end)
            ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
            terminal_pose = np.array([ref.GetPos().x, ref.GetPos().y, ref.GetRot().GetCardanAnglesZYX().z])
            if not np.isfinite(terminal_pose).all():
                raise FloatingPointError(f"non-finite pose after frame {frame}")
            if ref.GetPos().z < float(tmap.height(terminal_pose[0], terminal_pose[1])) - 1.0:
                raise RuntimeError(f"vehicle fell through the soil at frame {frame}")  # invalid physics, never a label
            # CRM-only terminal: a wheel that has dug through the whole soil layer is supported by nothing (boundary
            # markers push on soil, not on the tyre) and the vehicle then drops through the floor. Physically the
            # vehicle is bogged to the axles; stop here, before the unphysical fall, and count it as a failure.
            sinkage = max(tire_radii[name] - (float(vehicle.GetSpindlePos(axle, side).z)
                          - float(tmap.height(vehicle.GetSpindlePos(axle, side).x, vehicle.GetSpindlePos(axle, side).y)))
                          for name, axle, side in WHEEL_SPECS)
            max_sinkage = max(max_sinkage, sinkage)
            deep_frames = deep_frames + 1 if sinkage > float(cfg["depth_m"]) + float(cfg.get("breakthrough_margin_m", 0.06)) else 0
            if deep_frames >= 5:  # 0.25 s: a hard landing may dip through transiently, a dug-in wheel stays there
                status = "soil_breakthrough_terminated"
                policy.events.append({"kind": status, "interval_end_s": (frame + 1) * DT, "wheel_sinkage_m": sinkage})
                break
            reached = np.linalg.norm(terminal_pose[:2] - goal) <= float(case.get("goal_radius_m", 2.5))
            slow = not at_end and abs(float(state[0])) < .3 and float(action[1]) > .3
            consecutive_slow = consecutive_slow + 1 if slow else 0
            longest_slow = max(longest_slow, consecutive_slow)
            if abs(float(vehicle.GetRoll())) > math.radians(60.) or abs(float(vehicle.GetPitch())) > math.radians(60.):
                status = "rollover"
                break
            if reached:
                status, goal_time = "goal_reached", (frame + 1) * DT
                break
            stop = policy.check(frame, record_pose, record_action, record_parked, terminal_pose, wp,
                                0. if at_end else float(speed[wp]))
            if stop is not None:
                status = stop
                break
        frame += 1

    states, actions, poses = np.asarray(record_state), np.asarray(record_action), np.asarray(record_pose)
    full_poses = np.concatenate([poses, terminal_pose[None]])
    terrain.Synchronize(sim_t)
    hmmwv.Synchronize(sim_t, inputs, terrain)
    endpoint = measure(len(states), sim_t)
    terminal_state = np.array([float(endpoint[field]) for field in fields], np.float32)
    terminal_wp = frozen.nearest_index(xy, terminal_pose[:2], wp)
    terminal_parked = terminal_wp >= len(xy) - 2 and np.linalg.norm(terminal_pose[:2] - xy[-1]) < 3.
    elapsed = len(states) * DT
    distance0 = float(np.linalg.norm(anchor_pose[:2] - goal))
    distance1 = float(np.linalg.norm(terminal_pose[:2] - goal))
    low_progress_windows = []
    for anchor in range(max(0, len(states) - 80 + 1)):
        if not any(record_parked[anchor:anchor + 80]) and np.linalg.norm(full_poses[anchor + 80, :2] - full_poses[anchor, :2]) < .6 and actions[anchor:anchor + 80, 1].mean() > .3:
            low_progress_windows.append(anchor)
    bounded_windows = []
    for anchor in range(max(0, len(states) - 40 + 1)):
        if any(record_parked[anchor:anchor + 40]) or not np.all(actions[anchor:anchor + 40, 1] > .3):
            continue
        points = full_poses[anchor:anchor + 41, :2]
        if float(np.square(points[:, None, :] - points[None, :, :]).sum(-1).max()) <= .25 ** 2:
            bounded_windows.append(anchor)
    n = len(states)
    np.savez_compressed(out / "trajectory.npz", state=states, action=actions, pose=poses,
        terminal_pose=terminal_pose, power_kw=np.asarray(record_power), contact_n=np.zeros(n),
        terminal_state=terminal_state, terminal_parked=np.asarray(terminal_parked, bool),
        parked=np.asarray(record_parked, bool), positive_work_kj_per_interval=np.asarray(record_interval_work),
        state_fields=np.asarray(fields), dt_s=np.float32(DT))
    extra = np.asarray(record_extra, np.float32)
    np.savez_compressed(out / "crm_extra.npz", pos_z_m=extra[:, 0], bmp_ground_z_m=extra[:, 1], quat=extra[:, 2:6],
        spindle_z_m=extra[:, 6:10], slip_ratio=extra[:, 10:14], fsi_force_wheel_fx_n=extra[:, 14:18],
        wheel_order=np.asarray([w[0] for w in WHEEL_SPECS]), tire_radius_m=np.asarray([tire_radii[w[0]] for w in WHEEL_SPECS]))
    desired = np.asarray([c[3] for c in policy.commands], float)
    if len(desired) < n:  # goal/rollover break precedes the policy hook: add the last commanded speed
        desired = np.r_[desired, 0. if record_parked[-1] else float(speed[wp])]
    np.savez_compressed(out / "command_reference.npz", interval_start_s=np.arange(n) * DT,
        desired_speed_mps=desired[:n], reference_waypoints=np.asarray(route["waypoints"]),
        reference_stations=np.asarray(route["stations"]), reference_speeds=np.asarray(route["speeds"]),
        reference_headings=np.asarray(route["headings"]))
    wall = time.time() - wall0
    sink = extra[:, 6:10].mean(1) - extra[:, 1]  # mean spindle height above the BMP ground at the chassis reference
    summary = {"case_id": case["id"], "status": status, "goal_reached": goal_time is not None,
        "contact_free_goal_reached": goal_time is not None,
        "safe_goal_reached": goal_time is not None and longest_slow * DT < 2. and status != "rollover",
        "goal_time_s": goal_time, "elapsed_s": elapsed, "goal_radius_m": case.get("goal_radius_m", 2.5),
        "initial_goal_distance_m": distance0, "final_goal_distance_m": distance1,
        "goal_progress_m": distance0 - distance1,
        "net_displacement_m": float(np.linalg.norm(terminal_pose[:2] - anchor_pose[:2])),
        "path_length_m": float(np.linalg.norm(np.diff(full_poses[:, :2], axis=0), axis=1).sum()),
        "asset_contact": False, "max_asset_contact_n": 0., "first_asset_contact_interval_start_s": None,
        "longest_consecutive_effortful_near_zero_speed_s": longest_slow * DT,
        "sustained_near_stop": longest_slow * DT >= 2.,
        "bounded_blockage_v1": bool(bounded_windows), "bounded_blockage_v1_windows": bounded_windows,
        "safe_goal_reached_with_bounded_blockage_v1": goal_time is not None and longest_slow * DT < 2. and not bounded_windows and status != "rollover",
        "low_net_progress_4s_windows": low_progress_windows,
        "positive_work_kj": positive_work_kj, "frames": n, "wall_s": wall,
        "driver": dict(frozen.DRIVER), "terrain": "crm",
        "crm": {"config": cfg, "n_sph": int(terrain.GetNumSPHParticles()),
                "n_boundary_bce": int(terrain.GetNumBoundaryBCEMarkers()), "build_s": build_s,
                "rtf_sim_over_wall": (elapsed + SETTLE_S) / max(wall, 1e-9), "physics_dt_s": dt,
                "mean_spindle_height_above_bmp_m": float(sink.mean()), "min_spindle_height_above_bmp_m": float(sink.min()),
                "max_abs_slip_ratio_p95": float(np.percentile(np.abs(extra[:, 10:14]), 95)),
                "max_wheel_sinkage_below_bmp_m": float(max_sinkage),
                "soil_breakthrough_rule": "any wheel lower than tyre radius below the BMP surface by more than soil depth + margin; counted as a failure (bogged)"},
        "measurement_scope": "Headless fixed-reference CRM collection; fresh soil per episode; no online replanning."}
    dump(out / "outcome.json", summary)
    return summary, route


def main(argv=None, scene_hook=None, frame_hook=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--route", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chrono-data", required=True)
    p.add_argument("--crm-config", default="")
    p.add_argument("--horizon-s", type=float, default=120.)
    p.add_argument("--minimum-elapsed-s", type=float, default=24.)
    p.add_argument("--confirm-s", type=float, default=2.)
    p.add_argument("--recovery-tail-s", type=float, default=8.)
    p.add_argument("--disable-early-stop", action="store_true")
    p.add_argument("--episode-seed", type=int, default=None, help="unique per-episode id seed, recorded for provenance")
    args = p.parse_args(argv)
    args.scene_hook, args.frame_hook = scene_hook, frame_hook
    source, out = Path(args.source_root).resolve(), Path(args.out).resolve()
    sys.path.insert(0, str(source / "src"))
    sys.path.insert(0, str(source / "scripts"))
    args.case, args.route, args.out = str(Path(args.case).resolve()), str(Path(args.route).resolve()), str(out)
    import gen_collect  # rigid collector: StopPolicy and hashing helpers reused verbatim

    cfg = merged_config(args.crm_config)
    case = read(args.case)
    gen_collect.require(case["split"] in ("train", "val", "test"), "Missing declared group split")
    gen_collect.require(case["layout"]["assets"] == [], "CRM campaign runs on bare terrain")
    out.mkdir(parents=True, exist_ok=True)
    gen_collect.require(not (out / "episode_complete.json").exists(), "Episode already complete")
    for stale in out.iterdir():  # a killed attempt leaves partial files; the episode restarts from scratch
        if stale.is_file():
            stale.unlink()

    class CrmPolicy(gen_collect.StopPolicy):
        def on_anchor_crm(self, row, state, pose, history, tmap):
            speed = float(np.linalg.norm(np.asarray(state)[:2]))
            roll, pitch = float(state[2]), float(state[3])
            start_yaw = self.case["layout"]["start_yaw"]
            yaw_error = float(math.atan2(math.sin(pose[2] - start_yaw), math.cos(pose[2] - start_yaw)))
            position_error = float(np.linalg.norm(np.asarray(pose[:2]) - self.case["layout"]["start_xy"]))
            clearance = float(row["pos_z_m"] - tmap.height(row["pos_x_m"], row["pos_y_m"]))
            finite = bool(np.isfinite(state).all() and np.isfinite(pose).all())
            passed = bool(finite and speed <= 1. and max(abs(roll), abs(pitch)) <= math.radians(25.)
                          and abs(yaw_error) <= math.radians(10.) and position_error <= 1. and 0. <= clearance <= 1.2)
            self.initial_state_report = {"passed": passed, "finite": finite, "body_horizontal_speed_mps": speed,
                "roll_rad": roll, "pitch_rad": pitch, "yaw_error_rad": yaw_error, "start_xy_error_m": position_error,
                "chassis_reference_height_above_bmp_m": clearance, "pose": np.asarray(pose).tolist(),
                "limits": {"body_horizontal_speed_mps": 1., "absolute_roll_pitch_deg": 25., "yaw_error_deg": 10.,
                           "start_xy_error_m": 1., "chassis_reference_height_above_bmp_m": [0., 1.2]},
                "settle_s": .8, "replaces": "rigid native-height raycast audit (no rigid patch exists under CRM)"}
            dump(Path(self.args.out) / "initial_state_validation.json", self.initial_state_report)
            gen_collect.require(passed, "Invalid settled launch on CRM soil")

    policy = CrmPolicy(args, case)
    request = {"schema": "crm_f104_collection_request_v1", "case": args.case, "case_sha256": gen_collect.sha(args.case),
        "route": args.route, "route_sha256": gen_collect.sha(args.route), "scene_id": case["id"], "split": case["split"],
        "episode_seed": args.episode_seed, "crm_config": cfg, "horizon_s": args.horizon_s, "stop_policy": policy.config(),
        "collector_sha256": gen_collect.sha(__file__), "host": os.uname().nodename,
        "gpu_env": {k: os.environ.get(k) for k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")},
        "time_accounting": "Completed measured traversal intervals only; excludes the 0.8 s settle and failed processes"}
    dump(out / "collection_request.json", request)
    shutil.copyfile(args.case, out / "case.json")
    shutil.copyfile(args.route, out / "reference.json")
    started = time.time()
    try:
        summary, route = run(args, cfg, policy)
        n = summary["frames"]
        dump(out / "f104_episode.json", {"schema": "crm_f104_collection_completed_v1", "scene_id": case["id"],
            "split": case["split"], "status": summary["status"], "actual_elapsed_s": n * gen_collect.DT,
            "actual_elapsed_h": n * gen_collect.DT / 3600., "interval_count": n,
            "early_stop_policy": policy.config(), "stop_events": policy.events,
            "initial_state_valid": policy.initial_state_report["passed"],
            "wall_s_including_finalization": time.time() - started})
        artifacts = {f.name: gen_collect.sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.suffix != ".log"}
        tmp = out / "episode_complete.json.tmp"
        dump(tmp, {"schema": "crm_f104_episode_complete_v1", "actual_elapsed_s": n * gen_collect.DT,
                   "status": summary["status"], "artifacts_sha256": artifacts})
        os.replace(tmp, out / "episode_complete.json")  # atomic completion marker = resumability key
        print(json.dumps({"out": str(out), "status": summary["status"], "actual_elapsed_s": n * gen_collect.DT,
                          "wall_s": time.time() - started, "rtf": summary["crm"]["rtf_sim_over_wall"], "complete": True}))
    except Exception as exc:
        dump(out / "collection_failure.json", {"type": type(exc).__name__, "error": str(exc),
             "count_toward_completed_hours": False, "stop_events": policy.events})
        raise


if __name__ == "__main__":
    main()
