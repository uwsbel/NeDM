#!/usr/bin/env python3
"""HMMWV route episode on CRM (SPH deformable soil) with external-control and prefix-branch modes (PLAN B3, A4).

Fork of scripts/crm_collect.py.  That file is IMPORTED (terrain build, tyre fields, config merge) and never edited;
the episode loop and the driver are forked here.  The name keeps the ``crm_collect`` substring so crm_worker.py
forwards ``--crm-config`` (required here: the physics step comes from the campaign config, PLAN R2).

Modes (``--mode``):
  native         the unmodified collector's behaviour: the same physics/driver statements in the same order, the same
                 files.  The only differences are three provenance keys appended to outcome.json (``mode``,
                 ``collector_sha256``, ``physics_dt_s``) and the request/complete markers, which hash this file.
  branch         ``--branch-frame F --branch-route <json>``: the case route is driven by the native follower to frame F;
                 at the top of frame F (before that frame's route index and desired speed are computed) the follower is
                 rebuilt on the branch route with gc_control.make_follower(initialize=False) (no body is added; the
                 constructor resets the controllers), wp is set to 0 and xy/speed/goal are swapped; ``previous_steer``
                 is kept so the frozen per-substep 2/s steering clamp makes the first branch substep continuous.  The
                 branch frame, route hashes, branch pose and the 40-frame prefix history window (12 observable state
                 columns + 3 actions, masked where the action frame is < 0) go to outcome.json['branch'] and
                 trajectory.npz (branch_frame, branch_pose, branch_hist, branch_hmask); the branch route is copied to
                 branch_reference.json and appended to command_reference.npz.
  pid_held       external control: the shadow follower's output (GetInputs at the top of frame k = its last Advance of
                 frame k-1) is clipped ONCE per frame by gc_control.hold_clip (steering within +-0.1 of the previous
                 held steering, boxes) and the identical triple is written into a veh.DriverInputs at every physics
                 substep of the frame.  The frozen per-substep clamp is replaced by that one clip.
  pid_perturbed  as pid_held with gc_control.OUPerturb (seeded from --episode-seed) applied to the shadow output before
                 the clip.
  policy         as pid_held with a numpy actor (--actor npz, gc_control.NumpyActor) evaluated on the
                 gc_control.PolicyObs observation built from the pose/state captured at the top of the frame (before
                 Synchronize: positions, velocities, rates and spindle speeds are final there; the engine speed is
                 read from the transmission's integrated shaft state because the SHAFTS engine's own value lags one
                 substep before Synchronize; the 12 observable columns equal the recorded state[k], checked and
                 recorded per run) and the last held action; frame 0 is padded with the rest state and the settle
                 action (0, 0, 1).
In the three external-control modes the follower keeps running as a shadow (Synchronize/Advance at every substep,
SetDesiredSpeed every frame), so the settle, the parking rule and the desired-speed record are identical to native.
The recorded ``action[k]`` is the held triple.  A 40 s any-throttle near-stop rule (|vx| < 0.3 m/s on 800 consecutive
non-parked frames -> prolonged_blockage_terminated, evaluated after the native goal/rollover/StopPolicy checks) is
active in the external-control modes only.  ``--near-stop-all-modes`` enables it for native and branch as well; that
CAN change a native outcome (creeping or low-throttle stalls that the bounded-displacement rule never confirms), so it
is off by default and recorded in outcome.json when on.  ``--substep-log <npz>`` records the applied inputs at every
substep (all modes, settle included with negative frame numbers) for the timing audit.
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

import crm_collect as base  # noqa: E402  unmodified collector: build_crm, crm_tire_fields, merged_config, read, dump
import gc_control as gc  # noqa: E402  hold_clip, OUPerturb, NumpyActor, PolicyObs, make_follower, route_sha256

MODES = ("native", "branch", "pid_perturbed", "pid_held", "policy")
EXTERNAL_MODES = ("pid_perturbed", "pid_held", "policy")
HIST_T = 40
HIST_STATE_COLS = tuple(int(c) for c in gc.OBSERVABLE_COLS)  # 0-6, 11-14, 15: the 12 deployable state columns
NEAR_STOP_VX_MPS = 0.3
NEAR_STOP_STATUS = "prolonged_blockage_terminated"

read, dump, merged_config = base.read, base.dump, base.merged_config


def require(ok, message):
    if not ok:
        raise ValueError(message)


def prefix_history(states, actions, F, T=HIST_T, cols=HIST_STATE_COLS):
    """The causal T-frame window ending at the branch frame F in the mixed dataset's convention (ga_build_mixed.py):
    hist[t] = [state[F-(T-1)+t][cols], action[F-T+t]], hmask[t] = (F-T+t >= 0); masked rows are zero."""
    states, actions = np.asarray(states, np.float32), np.asarray(actions, np.float32)
    require(len(states) > F and len(actions) > F, "branch frame beyond the recorded prefix")
    t = np.arange(T)
    si, ai = F - (T - 1) + t, F - T + t
    ok = ai >= 0
    hist, hmask = np.zeros((T, len(cols) + 3), np.float32), np.zeros(T, bool)
    hmask[ok] = True
    hist[ok] = np.concatenate([states[si[ok]][:, list(cols)], actions[ai[ok]]], 1)
    return hist, hmask


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

    mode = args.mode
    external = mode in EXTERNAL_MODES
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
    terrain = base.build_crm(chrono, veh, fsi, system, vehicle, arena, meta, cfg)
    build_s = time.time() - t_build
    driver = frozen.make_driver(chrono, veh, vehicle, route, tmap)
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

    # ------------------------------------------------------------------ mode state (no physics object is touched here)
    branch = None
    if mode == "branch":
        branch_route = frozen.read_route(args.branch_route)
        bxy_end = np.asarray(branch_route["waypoints"], float)[-1]
        require(np.linalg.norm(bxy_end - goal) <= float(args.branch_goal_tol_m),
                f"branch route ends {np.linalg.norm(bxy_end - goal):.3f} m from the case goal (> {args.branch_goal_tol_m} m)")
        branch = {"frame": int(args.branch_frame), "route": branch_route, "reached": False}
    ou = actor = po = None
    if mode == "pid_perturbed":
        ou = gc.OUPerturb(seed=int(args.episode_seed), dt=DT, tau_s=args.perturb_tau_s, steer_sd=args.perturb_steer_sd,
                          throttle_sd=args.perturb_throttle_sd, brake_p=args.perturb_brake_p,
                          brake_len_s=tuple(args.perturb_brake_len_s), brake_level=tuple(args.perturb_brake_level),
                          bound_sds=args.perturb_bound_sds)
    if mode == "policy":
        actor = gc.NumpyActor.from_npz(args.actor)
        po = gc.PolicyObs.from_meta(route, actor.meta)
        require(po.num_obs == actor.num_obs, f"actor expects {actor.num_obs} observations, PolicyObs builds {po.num_obs}")
    last_action = np.asarray(gc.SETTLE_ACTION, np.float64)  # action[k-1] as seen by the external controller; settle at k = 0
    held, held_inputs, pre_state = None, None, None
    record_shadow, record_perturb = [], []
    pre_capture_max_diff = 0.
    substep_log = [] if args.substep_log else None
    near_stop_active = bool(args.near_stop_all_modes or (external and not args.no_near_stop))
    near_stop_frames = int(round(float(args.near_stop_s) / DT))
    near_stop_run, near_stop_longest, near_stop_fired = 0, 0, False
    obs_cols = list(HIST_STATE_COLS)

    def measure(frame_index, ts):
        row = capture_row(hmmwv, terrain, "crm_f104", "follower", case["id"], "development",
                          frame_index, ts, inputs, include_tires=False)
        row.update(base.crm_tire_fields(chrono, veh, vehicle, terrain, WHEEL_SPECS, tire_radii))
        row["engine_motor_speed_radps"] = float(engine.GetMotorSpeed())
        row["engine_motorshaft_torque_nm"] = float(engine.GetOutputMotorshaftTorque())
        return row

    def measure_pre_synchronize(frame_index, ts):
        """The policy's state at the top of the frame, BEFORE this frame's hmmwv.Synchronize.  Positions, velocities,
        angular rates and spindle speeds are final there; the SHAFTS engine's motor speed is not: the vehicle's
        Synchronize imposes the transmission's motorshaft speed on the engine shaft (ChEngineShafts::Synchronize),
        so engine.GetMotorSpeed() still holds the previous substep's imposed value (one-substep lag, VERIFY_gen_ext
        P1).  The transmission's integrated shaft state is exactly the value that will be imposed, so column 15 read
        here equals the recorded state[k][15] (checked per run: pre_capture_vs_recorded_observable_max_abs_diff)."""
        row = measure(frame_index, ts)
        row["engine_motor_speed_radps"] = float(transmission.GetOutputMotorshaftSpeed())
        return row

    while frame < frames:
        if branch is not None and not branch["reached"] and frame == branch["frame"]:
            # A4 driver swap at the top of frame F, before this frame's route index and desired speed are computed.
            require(len(record_state) == branch["frame"], "prefix length disagrees with the branch frame")
            ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
            bpose = np.array([ref.GetPos().x, ref.GetPos().y, ref.GetRot().GetCardanAnglesZYX().z])
            bxy = np.asarray(branch["route"]["waypoints"], float)
            start_offset = float(np.linalg.norm(bxy[0] - bpose[:2]))
            require(start_offset <= float(args.branch_start_tol_m),
                    f"branch route starts {start_offset:.3f} m from the vehicle at frame {frame} (> {args.branch_start_tol_m} m)")
            branch["prefix_driver"] = driver  # keep the prefix follower alive (its Initialize() body stays in the system anyway)
            driver = gc.make_follower(veh, chrono, vehicle, branch["route"], tmap.height, initialize=False)
            xy, speed = bxy, np.asarray(branch["route"]["speeds"], float)
            goal = bxy[-1].copy()
            wp = 0
            branch.update(reached=True, pose=bpose, start_offset_m=start_offset, previous_steer=float(previous_steer),
                          sim_t=float(sim_t))
        ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
        pos = np.array([ref.GetPos().x, ref.GetPos().y])
        wp = frozen.nearest_index(xy, pos, wp)
        at_end = wp >= len(xy) - 2 and np.linalg.norm(pos - xy[-1]) < 3.
        driver.SetDesiredSpeed(0. if frame < 0 or at_end else float(speed[wp]))
        frame_positive_work = 0.
        if external and frame >= 0:
            # One command per frame.  The shadow follower's GetInputs here is the output of its last Advance (the
            # final substep of frame k-1): exactly the value the native collector records as action[k] before its clamp.
            shadow_in = driver.GetInputs()
            shadow = (float(shadow_in.m_steering), float(shadow_in.m_throttle), float(shadow_in.m_braking))
            if mode == "pid_held":
                raw = shadow
            elif mode == "pid_perturbed":
                raw = ou.step(shadow)
            else:
                pre = measure_pre_synchronize(frame, sim_t)  # top of the frame: the 12 observable columns == state[k]
                pre_state = np.array([float(pre[field]) for field in fields], np.float32)
                pre_pose = np.array([pre["pos_x_m"], pre["pos_y_m"], pre["yaw_rad"]], np.float64)
                if frame == 0:
                    po.reset(pre_state)
                raw = actor.act(po.observe(pre_pose, pre_state, last_action))
            held = gc.hold_clip(raw, float(last_action[0]))
            held_inputs = veh.DriverInputs()
            held_inputs.m_steering, held_inputs.m_throttle, held_inputs.m_braking = held
            record_shadow.append(shadow)
            if ou is not None:
                record_perturb.append([ou.last["d_steer"], ou.last["d_throttle"], float(ou.last["tap_active"]), ou.last["tap_level"]])
        for sub in range(substeps):
            ts = sim_t
            driver.Synchronize(ts)
            if external and frame >= 0:
                inputs = held_inputs  # the identical triple at every substep; hold_clip replaced the per-substep clamp
                previous_steer = held[0]
            else:
                inputs = driver.GetInputs()
                previous_steer = 0. if frame < 0 else float(np.clip(inputs.m_steering, previous_steer - 2. * dt, previous_steer + 2. * dt))
                inputs.m_steering = previous_steer
            terrain.Synchronize(ts)
            hmmwv.Synchronize(ts, inputs, terrain)
            if substep_log is not None:
                substep_log.append((frame, sub, float(inputs.m_steering), float(inputs.m_throttle), float(inputs.m_braking)))
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
                if pre_state is not None:
                    pre_capture_max_diff = max(pre_capture_max_diff, float(np.abs(pre_state[obs_cols] - state[obs_cols]).max()))
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
            if external:
                last_action = action.astype(np.float64)  # == held
                if po is not None:
                    po.push(state, action)
            positive_work_kj += frame_positive_work
            record_interval_work.append(frame_positive_work)
            record_parked.append(at_end)
            ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
            terminal_pose = np.array([ref.GetPos().x, ref.GetPos().y, ref.GetRot().GetCardanAnglesZYX().z])
            if not np.isfinite(terminal_pose).all():
                raise FloatingPointError(f"non-finite pose after frame {frame}")
            if ref.GetPos().z < float(tmap.height(terminal_pose[0], terminal_pose[1])) - 1.0:
                raise RuntimeError(f"vehicle fell through the soil at frame {frame}")  # invalid physics, never a label
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
            if near_stop_active:  # any-throttle near-stop rule, after every native rule of this frame
                near_slow = (not at_end) and abs(float(state[0])) < NEAR_STOP_VX_MPS
                near_stop_run = near_stop_run + 1 if near_slow else 0
                near_stop_longest = max(near_stop_longest, near_stop_run)
                if near_stop_run >= near_stop_frames:
                    status, near_stop_fired = NEAR_STOP_STATUS, True
                    policy.events.append({"kind": NEAR_STOP_STATUS, "rule": "near_stop_any_throttle",
                                          "interval_end_s": (frame + 1) * DT, "near_stop_s": near_stop_run * DT,
                                          "vx_threshold_mps": NEAR_STOP_VX_MPS})
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
    traj = dict(state=states, action=actions, pose=poses,
        terminal_pose=terminal_pose, power_kw=np.asarray(record_power), contact_n=np.zeros(n),
        terminal_state=terminal_state, terminal_parked=np.asarray(terminal_parked, bool),
        parked=np.asarray(record_parked, bool), positive_work_kj_per_interval=np.asarray(record_interval_work),
        state_fields=np.asarray(fields), dt_s=np.float32(DT))
    branch_hist = branch_hmask = None
    if mode != "native":
        traj["mode"] = np.asarray(mode)
    if external:
        traj["shadow_action"] = np.asarray(record_shadow, np.float32).reshape(-1, 3)
        if ou is not None:
            traj["perturbation"] = np.asarray(record_perturb, np.float32).reshape(-1, 4)
    if branch is not None:
        traj["branch_frame"] = np.int64(branch["frame"])
        traj["branch_reached"] = np.asarray(branch["reached"], bool)
        if branch["reached"]:
            branch_hist, branch_hmask = prefix_history(states, actions, branch["frame"])
            traj["branch_pose"] = np.asarray(branch["pose"], np.float64)
        else:
            branch_hist, branch_hmask = np.zeros((HIST_T, len(HIST_STATE_COLS) + 3), np.float32), np.zeros(HIST_T, bool)
            traj["branch_pose"] = np.full(3, np.nan)
        traj["branch_hist"], traj["branch_hmask"] = branch_hist, branch_hmask
        traj["branch_hist_cols"] = np.asarray(HIST_STATE_COLS, np.int16)
    np.savez_compressed(out / "trajectory.npz", **traj)
    extra = np.asarray(record_extra, np.float32)
    np.savez_compressed(out / "crm_extra.npz", pos_z_m=extra[:, 0], bmp_ground_z_m=extra[:, 1], quat=extra[:, 2:6],
        spindle_z_m=extra[:, 6:10], slip_ratio=extra[:, 10:14], fsi_force_wheel_fx_n=extra[:, 14:18],
        wheel_order=np.asarray([w[0] for w in WHEEL_SPECS]), tire_radius_m=np.asarray([tire_radii[w[0]] for w in WHEEL_SPECS]))
    desired = np.asarray([c[3] for c in policy.commands], float)
    if len(desired) < n:  # goal/rollover break precedes the policy hook: add the last commanded speed
        desired = np.r_[desired, 0. if record_parked[-1] else float(speed[wp])]
    cmdref = dict(interval_start_s=np.arange(n) * DT,
        desired_speed_mps=desired[:n], reference_waypoints=np.asarray(route["waypoints"]),
        reference_stations=np.asarray(route["stations"]), reference_speeds=np.asarray(route["speeds"]),
        reference_headings=np.asarray(route["headings"]))
    if branch is not None:
        br = branch["route"]
        cmdref.update(branch_frame=np.int64(branch["frame"]), branch_reached=np.asarray(branch["reached"], bool),
                      branch_reference_waypoints=np.asarray(br["waypoints"]), branch_reference_stations=np.asarray(br["stations"]),
                      branch_reference_speeds=np.asarray(br["speeds"]), branch_reference_headings=np.asarray(br["headings"]))
    np.savez_compressed(out / "command_reference.npz", **cmdref)
    if substep_log is not None:
        log = np.asarray(substep_log, np.float64).reshape(-1, 5)
        np.savez_compressed(args.substep_log, frame=log[:, 0].astype(np.int32), sub=log[:, 1].astype(np.int16),
                            inputs=log[:, 2:5], substeps=np.int64(substeps), dt_s=np.float64(dt), mode=np.asarray(mode),
                            layout=np.asarray("inputs[i] = (steering, throttle, braking) passed to hmmwv.Synchronize at (frame, sub); frame < 0 = settle"))
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
    # ------------------------------------------------------------------ provenance and mode blocks (appended)
    summary["mode"] = mode
    summary["collector_sha256"] = args.collector_sha256
    summary["physics_dt_s"] = dt
    if external:
        summary["driver"] = {**summary["driver"], "control": mode}
        shadow = np.asarray(record_shadow, np.float64).reshape(-1, 3)
        held_all = actions[:len(shadow)].astype(np.float64)
        dsteer = np.abs(np.diff(actions[:, 0].astype(np.float64))) if n > 1 else np.zeros(0)
        summary["control"] = {
            "mode": mode, "episode_seed": args.episode_seed, "frames_controlled": int(len(shadow)),
            "hold": "one gc_control.hold_clip per frame (steering within +-0.1 of the previous held steering, boxes); the identical triple at every physics substep; recorded action = held triple",
            "shadow_follower": "native ChPathFollowerDriver kept in Synchronize/Advance as a shadow; its GetInputs at the top of frame k (its last Advance of frame k-1) is the base command; settle, parking and desired speed as native",
            "held_minus_shadow_abs_max": np.abs(held_all - shadow).max(0).tolist() if len(shadow) else None,
            "held_minus_shadow_abs_mean": np.abs(held_all - shadow).mean(0).tolist() if len(shadow) else None,
            "steer_step_abs_max": float(dsteer.max()) if len(dsteer) else 0.,
            "steer_clip_bound_frames": int((dsteer > gc.STEER_RATE_PER_FRAME - 1e-9).sum()) if len(dsteer) else 0,
            "throttle_brake_both_positive_frames": int(((actions[:, 1] > 0) & (actions[:, 2] > 0)).sum())}
        if ou is not None:
            summary["control"]["perturb"] = {**ou.summary(), "tau_s": args.perturb_tau_s, "steer_sd": args.perturb_steer_sd,
                "throttle_sd": args.perturb_throttle_sd, "brake_p_per_frame": args.perturb_brake_p,
                "brake_len_s": list(args.perturb_brake_len_s), "brake_level": list(args.perturb_brake_level),
                "bound_sds": args.perturb_bound_sds,
                "d_steer_abs_max": float(np.abs(np.asarray(record_perturb)[:, 0]).max()) if record_perturb else 0.,
                "d_throttle_abs_max": float(np.abs(np.asarray(record_perturb)[:, 1]).max()) if record_perturb else 0.,
                "tap_frames": int(np.asarray(record_perturb)[:, 2].sum()) if record_perturb else 0}
        if actor is not None:
            summary["control"]["actor"] = {"path": str(args.actor), "sha256": args.actor_sha256, "num_obs": int(actor.num_obs),
                "num_actions": int(actor.num_actions), "activation": actor.activation, "n_layers": len(actor.layers),
                "obs_layout": po.layout(), "obs_nonfinite_zeroed": int(po.n_nonfinite),
                "pre_capture_vs_recorded_observable_max_abs_diff": pre_capture_max_diff,
                "frame0_padding": "PolicyObs.reset(rest state = state at the top of frame 0) + settle action (0, 0, 1)"}
    if branch is not None:
        br = branch["route"]
        F = branch["frame"]
        blk = {"branch_frame": F, "branch_time_s": F * DT, "reached": bool(branch["reached"]),
               "branch_route": str(args.branch_route), "branch_route_sha256": args.branch_route_sha256,
               "branch_route_content_sha256": gc.route_sha256(br), "branch_route_meta": br.get("meta", {}),
               "prefix_route_sha256": args.route_sha256, "case_goal_xy": np.asarray(case["goal_xy"], float).tolist(),
               "branch_goal_xy": np.asarray(br["waypoints"], float)[-1].tolist(),
               "goal_used_after_branch": "branch route's last waypoint (within --branch-goal-tol-m of the case goal)",
               "driver_swap": "gc_control.make_follower(initialize=False) at the top of frame F before the route index / desired speed; wp = 0; xy/speed/goal swapped; previous_steer kept for the per-substep 2/s clamp",
               "hist_T": HIST_T, "hist_cols": list(HIST_STATE_COLS),
               "hist_layout": "hist[t] = [state[F-39+t][hist_cols], action[F-40+t]]; hmask[t] = action frame F-40+t >= 0 (ga_build_mixed convention)",
               "hist": branch_hist.tolist(), "hmask": branch_hmask.tolist()}
        if branch["reached"]:
            blk.update({"branch_pose": np.asarray(branch["pose"], float).tolist(),
                        "branch_pose_equals_recorded_pose_F": bool(np.allclose(branch["pose"], poses[F], atol=1e-9)),
                        "branch_start_offset_m": branch["start_offset_m"], "previous_steer_at_branch": branch["previous_steer"],
                        "prefix_last_action": actions[F - 1].tolist() if F > 0 else None,
                        "first_branch_action": actions[F].tolist(),
                        "steer_step_at_branch": float(abs(actions[F, 0] - actions[F - 1, 0])) if F > 0 else None,
                        "vx_at_branch_mps": float(states[F, 0]), "branch_route_v0_mps": float(br["speeds"][0])})
        else:
            blk.update({"branch_pose": None, "note": f"episode ended at frame {n} before the branch frame; no swap happened"})
        summary["branch"] = blk
    if near_stop_active:
        summary["near_stop_rule"] = {"active": True, "threshold_s": float(args.near_stop_s), "vx_threshold_mps": NEAR_STOP_VX_MPS,
                                     "excludes_parked_frames": True, "evaluated_after_native_rules": True,
                                     "longest_near_stop_run_s": near_stop_longest * DT, "fired": near_stop_fired,
                                     "enabled_by": "--near-stop-all-modes" if args.near_stop_all_modes else "external-control mode default"}
    dump(out / "outcome.json", summary)
    return summary, route


def main(argv=None, scene_hook=None, frame_hook=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-root", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--route", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chrono-data", required=True)
    p.add_argument("--crm-config", default="", help="REQUIRED here (PLAN R2): the physics step comes from the campaign config")
    p.add_argument("--horizon-s", type=float, default=120.)
    p.add_argument("--minimum-elapsed-s", type=float, default=24.)
    p.add_argument("--confirm-s", type=float, default=2.)
    p.add_argument("--recovery-tail-s", type=float, default=8.)
    p.add_argument("--disable-early-stop", action="store_true")
    p.add_argument("--episode-seed", type=int, default=None, help="unique per-episode id seed, recorded for provenance; seeds the perturbation in pid_perturbed")
    p.add_argument("--mode", choices=MODES, default="native")
    p.add_argument("--branch-frame", type=int, default=None, help="branch mode: frame F (>= 0) at whose top the follower is swapped")
    p.add_argument("--branch-route", default=None, help="branch mode: route JSON from the pose at F to the goal (gc_control.sample_continuations)")
    p.add_argument("--branch-start-tol-m", type=float, default=1.0, help="branch mode: max distance between the branch route start and the vehicle at F")
    p.add_argument("--branch-goal-tol-m", type=float, default=0.5, help="branch mode: max distance between the branch route end and the case goal")
    p.add_argument("--actor", default=None, help="policy mode: gc_control.NumpyActor npz (obs normaliser, MLP, squash, obs_layout in meta)")
    p.add_argument("--substep-log", default=None, help="optional npz: applied inputs at every physics substep (timing audit)")
    p.add_argument("--near-stop-s", type=float, default=40., help="any-throttle near-stop rule: |vx| < 0.3 m/s for this long (non-parked) -> prolonged_blockage_terminated")
    p.add_argument("--near-stop-all-modes", action="store_true", help="apply the near-stop rule in native/branch too (can change native outcomes)")
    p.add_argument("--no-near-stop", action="store_true", help="disable the near-stop rule in the external-control modes")
    p.add_argument("--perturb-tau-s", type=float, default=0.5)
    p.add_argument("--perturb-steer-sd", type=float, default=0.15)
    p.add_argument("--perturb-throttle-sd", type=float, default=0.25)
    p.add_argument("--perturb-brake-p", type=float, default=None, help="per-frame brake-tap start probability; default = 1.4 taps per 20 s of driving")
    p.add_argument("--perturb-brake-len-s", type=float, nargs=2, default=(0.5, 1.0))
    p.add_argument("--perturb-brake-level", type=float, nargs=2, default=(0.3, 1.0))
    p.add_argument("--perturb-bound-sds", type=float, default=1.0)
    args = p.parse_args(argv)
    args.scene_hook, args.frame_hook = scene_hook, frame_hook
    source, out = Path(args.source_root).resolve(), Path(args.out).resolve()
    sys.path.insert(0, str(source / "src"))
    sys.path.insert(0, str(source / "scripts"))
    args.case, args.route, args.out = str(Path(args.case).resolve()), str(Path(args.route).resolve()), str(out)
    import gen_collect  # rigid collector: StopPolicy and hashing helpers reused verbatim

    gen_collect.require(bool(args.crm_config), "--crm-config is required for crm_collect_ext.py (PLAN R2)")
    gen_collect.require(Path(args.crm_config).is_file(), f"--crm-config not found: {args.crm_config}")
    args.crm_config = str(Path(args.crm_config).resolve())
    if args.mode == "branch":
        gen_collect.require(args.branch_frame is not None and args.branch_frame >= 0, "branch mode needs --branch-frame >= 0")
        gen_collect.require(bool(args.branch_route) and Path(args.branch_route).is_file(), "branch mode needs --branch-route <json>")
        gen_collect.require(args.branch_frame < int(round(args.horizon_s / gen_collect.DT)),
                            f"--branch-frame {args.branch_frame} is not inside the horizon ({args.horizon_s} s)")
        args.branch_route = str(Path(args.branch_route).resolve())
    else:
        gen_collect.require(args.branch_frame is None and args.branch_route is None, "--branch-frame/--branch-route are branch-mode only")
    if args.mode == "policy":
        gen_collect.require(bool(args.actor) and Path(args.actor).is_file(), "policy mode needs --actor <npz>")
        args.actor = str(Path(args.actor).resolve())
    else:
        gen_collect.require(args.actor is None, "--actor is policy-mode only")
    if args.mode == "pid_perturbed":
        gen_collect.require(args.episode_seed is not None, "pid_perturbed needs --episode-seed (it seeds the perturbation)")
        if args.perturb_brake_p is None:
            args.perturb_brake_p = gc.OUPerturb.brake_p_for_rate(1.4 / 20.0, dt=gen_collect.DT, brake_len_s=tuple(args.perturb_brake_len_s))
    if args.substep_log:
        args.substep_log = str(Path(args.substep_log).resolve())
        Path(args.substep_log).parent.mkdir(parents=True, exist_ok=True)
    args.collector_sha256 = gen_collect.sha(__file__)
    args.route_sha256 = gen_collect.sha(args.route)
    args.branch_route_sha256 = gen_collect.sha(args.branch_route) if args.branch_route else None
    args.actor_sha256 = gen_collect.sha(args.actor) if args.actor else None

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
    ext = {"mode": args.mode, "near_stop_s": args.near_stop_s, "near_stop_all_modes": args.near_stop_all_modes,
           "no_near_stop": args.no_near_stop, "substep_log": args.substep_log,
           "base_collector_sha256": gen_collect.sha(HERE / "crm_collect.py"), "gc_control_sha256": gen_collect.sha(HERE / "gc_control.py")}
    if args.mode == "branch":
        ext.update(branch_frame=args.branch_frame, branch_route=args.branch_route, branch_route_sha256=args.branch_route_sha256,
                   branch_start_tol_m=args.branch_start_tol_m, branch_goal_tol_m=args.branch_goal_tol_m)
    if args.mode == "policy":
        ext.update(actor=args.actor, actor_sha256=args.actor_sha256)
    if args.mode == "pid_perturbed":
        ext.update(perturb={"tau_s": args.perturb_tau_s, "steer_sd": args.perturb_steer_sd, "throttle_sd": args.perturb_throttle_sd,
                            "brake_p_per_frame": args.perturb_brake_p, "brake_len_s": list(args.perturb_brake_len_s),
                            "brake_level": list(args.perturb_brake_level), "bound_sds": args.perturb_bound_sds})
    request = {"schema": "crm_f104_collection_request_v1", "case": args.case, "case_sha256": gen_collect.sha(args.case),
        "route": args.route, "route_sha256": args.route_sha256, "scene_id": case["id"], "split": case["split"],
        "episode_seed": args.episode_seed, "crm_config": cfg, "horizon_s": args.horizon_s, "stop_policy": policy.config(),
        "collector_sha256": args.collector_sha256, "host": os.uname().nodename,
        "gpu_env": {k: os.environ.get(k) for k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")},
        "time_accounting": "Completed measured traversal intervals only; excludes the 0.8 s settle and failed processes",
        "collector": "crm_collect_ext", "mode": args.mode, "ext": ext}
    dump(out / "collection_request.json", request)
    shutil.copyfile(args.case, out / "case.json")
    shutil.copyfile(args.route, out / "reference.json")
    if args.mode == "branch":
        shutil.copyfile(args.branch_route, out / "branch_reference.json")
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
                          "wall_s": time.time() - started, "rtf": summary["crm"]["rtf_sim_over_wall"], "mode": args.mode,
                          "complete": True}))
    except Exception as exc:
        dump(out / "collection_failure.json", {"type": type(exc).__name__, "error": str(exc),
             "count_toward_completed_hours": False, "stop_events": policy.events})
        raise


if __name__ == "__main__":
    main()
