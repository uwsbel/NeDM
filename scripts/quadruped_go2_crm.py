#!/usr/bin/env python
"""Unitree Go2 walking on CRM deformable terrain, headless, with video.

The second half of the WP0 gate for the proposed quadruped case study
(docs/state/progress/future-case-studies.md), which prescribes "a privileged
scripted gait walking on rigid ground, THEN CRM, with zero learning, before any
model work". `quadruped_wp0_gait.py` did the rigid half on RoboSimian. This does
the CRM half on the robot the study actually targets.

WHY GO2 RATHER THAN THE RoboSimian PROTOTYPE. The plan ranks the bootstrapping
problem as the study's real risk, and ranks "import a pretrained Go2 policy" as
option 3, "highest risk, keep off the critical path". That ranking is stale:
`uwsbel/sbel-reproducibility` 2025/multi-terrain-RL already trained a Go2
locomotion policy in Chrono on rigid ground and finetuned it on CRM granular
terrain. `model_2999.pt` is the CRM-finetuned checkpoint. So the seed-controller
problem is solved in-house, and RoboSimian's only stated purpose, shaking out
CRM foot-contact machinery, is served better by the target robot itself.

This is a PORT, not a reuse. That work runs `bochengzou::pychrono`; everything
here runs the `nedm` environment this repo specifies (`projectchrono` 10.0.0).
The observation convention is taken verbatim from `chrono_crmenv.py`, which is
authoritative because it ships with the checkpoint.

FOUR CONVENTIONS THAT SILENTLY BREAK THIS IF MISSED, all verified on kyle-sbel:

1. Joints are NOT actuated by default. `SetAllJointsActuationType` must be
   called BEFORE `PopulateSystem`. Without it `GetChMotor` returns a wrapped
   null pointer that is not None, reports a plausible type, and kills the
   interpreter with no traceback when touched. `if motor is not None` is not a
   valid success test here.
2. Joint positions and velocities are NEGATED into the policy's frame. This is
   a real sign-convention difference, not a reordering artifact.
3. Chrono orders joints [RR, RL, FR, FL]; the policy expects [FR, FL, RR, RL].
   The map is an involution, so the same array converts both ways.
4. The 3-wide command slot is a HARDCODED [0.5, 0, 0] * lin_vel_scale. It is
   NOT `env_cfg['target_lin_vel']`, which has two elements and is used only for
   reward. Wiring the config value in here produces a subtly wrong observation.

Requires the `nedm` env AND an OptiX-capable driver (R590+) for --video; see
docs/state/lessons/chrono-versions.md. CRM alone needs no OptiX.

Usage:
  "$NEDM_PY" scripts/quadruped_go2_crm.py --seconds 8 --out artifacts/go2_crm
  "$NEDM_PY" scripts/quadruped_go2_crm.py --seconds 8 --camera overhead --out artifacts/go2_top

Six everyday flags; everything else sits in the "advanced" group with a settled
default. The sprite parameters are no longer flags at all -- they are constants
in nedm/quadruped/camera.py, each with the measurement that fixed it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- everyday ----
    ap.add_argument("--seconds", type=float, default=8.0, help="simulated seconds")
    ap.add_argument("--terrain", choices=["crm", "rigid"], default="crm",
                    help="rigid reproduces the ground the policy was trained on")
    ap.add_argument("--soil", choices=["training", "eval"], default="training",
                    help="training is the softer soil the CRM finetune actually used")
    ap.add_argument("--camera", choices=["none", "overhead", "follow", "side"],
                    default="none",
                    help="none is headless; anything else records frames. overhead "
                         "looks straight down and tracks the robot, which is what "
                         "makes footprints and disturbed soil legible")
    ap.add_argument("--out", default="artifacts/go2_crm")
    ap.add_argument("--no-policy", action="store_true", help="hold the stand pose throughout")

    # ---- advanced: settled defaults, kept for sweeps and reproduction ----
    adv = ap.add_argument_group("advanced", "settled defaults; each was measured")
    adv.add_argument("--spawn-x", type=float, default=0.0,
                     help="Spawn X. The CRM bed's near edge is at -0.6 BY "
                          "CONSTRUCTION (centre = patch_x/2 - 0.6), independent of "
                          "patch_x, so a turning run that arcs backwards can leave "
                          "the bed from the default spawn.")
    adv.add_argument("--spawn-y", type=float, default=0.0)
    adv.add_argument("--imported-ckpt", default=None,
                     help="TorchScript policy from the legged_gym family, driven "
                          "through nedm.quadruped.imported_policy instead of the "
                          "harness contract. Implies its own conventions.")
    adv.add_argument("--command-family", default=None,
                     help="Structured excitation family; overrides --command.")
    adv.add_argument("--command", type=float, nargs=3, default=[0.5, 0.0, 0.0],
                     metavar=("VX", "VY", "WZ"),
                     help="Velocity command for --imported-ckpt only; the harness "
                          "policy reads a hardcoded literal and ignores this.")
    adv.add_argument("--actuation", choices=["torque", "position"], default="torque",
                     help="torque: PD on ChLinkMotorRotationTorque, kp/kd from the "
                          "legged_gym family, clamped to URDF effort. position: the "
                          "historical kinematic-constraint plant, unbounded torque, "
                          "which every gate before 2026-09-03 was measured on.")
    adv.add_argument("--assets", default="/home/kyle/Documents/sbel/sbel-reproducibility/2025/multi-terrain-RL")
    adv.add_argument("--step", type=float, default=5e-4, help="CFD step")
    adv.add_argument("--exchange-mult", type=int, default=5, help="MBS/CFD exchange = mult * step")
    adv.add_argument("--control-hz", type=float, default=50.0)
    adv.add_argument("--pose-ramp-seconds", type=float, default=0.75,
                     help="ease from the URDF spawn pose to the stand pose")
    adv.add_argument("--settle-seconds", type=float, default=0.5,
                     help="hold the stand pose after the ramp, before the policy")
    # The foot sphere is r=0.025, so 0.02 gives a foot spanning 2.5 spacings.
    adv.add_argument("--spacing", type=float, default=0.02)
    adv.add_argument("--patch-x", type=float, default=8.0)
    adv.add_argument("--patch-y", type=float, default=4.0)
    # 0.2 matches chrono_crmenv.py and sits just under the ~0.22 m depth at
    # which the bed starts heaving.
    adv.add_argument("--depth", type=float, default=0.20)
    adv.add_argument("--soil-bottom", type=float, default=0.0)
    # None means DERIVE it, placing the base so the lowest foot clears the soil.
    # A constant cannot do this: it fixes the BASE height while the FEET are what
    # meets the soil, and the Go2's rest pose has the legs extended. Swept
    # standing, 3 s: base clearance 0.34 gave 0.13-0.16 m of launch and a fall at
    # 0.64-0.85 s; 0.42 gave 0.07 m and 1.08-1.21 s; 0.60 gave zero and 2.97 s.
    adv.add_argument("--spawn-clearance", type=float, default=None,
                     help="base height above soil; omit to derive from leg reach")
    # In SPH SPACINGS, not metres, because what the foot must clear is the kernel
    # support radius and that scales with spacing.
    adv.add_argument("--foot-margin-spacings", type=float, default=5.0,
                     help="foot gap above the soil at spawn, in SPH spacings")
    adv.add_argument("--solver-iters", type=int, default=150)
    # Override the preset's soil stiffness / cohesion. Both presets give only
    # 1-5 mm of surface response under a Go2, which is a tenth of a particle
    # diameter and cannot be seen; these exist to sweep softer.
    adv.add_argument("--soil-young", type=float, default=None,
                     help="override Young's modulus (training preset: 5.0e5)")
    adv.add_argument("--soil-cohesion", type=float, default=None,
                     help="override cohesion (training preset: 2000)")
    # 0.5 is the Viper demo value and leaves an undamped limit cycle under
    # impact; 3.0 and above crash the full-scale run. 2.0 is nearly the only
    # value in the working window. See docs/state/lessons/chrono-versions.md.
    adv.add_argument("--artificial-viscosity", type=float, default=2.0)
    adv.add_argument("--no-calf-fsi", action="store_true",
                     help="couple only the feet to the SPH, not the calves")
    adv.add_argument("--no-check-embedded", dest="check_embedded",
                     action="store_false", default=True,
                     help="keep SPH particles that overlap the feet at init; "
                          "reproduces the launch, for comparison only")
    adv.add_argument("--soil-proxy", action="store_true",
                     help="add a static non-deforming floor box as well as the sprites")
    adv.add_argument("--video-fps", type=float, default=30.0)
    adv.add_argument("--video-width", type=int, default=960)
    adv.add_argument("--video-height", type=int, default=540)
    adv.add_argument("--cam-eye", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                     help="absolute camera position; overrides --camera framing")
    adv.add_argument("--cam-target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                     help="absolute camera aim point")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    from nedm.quadruped import soilprobe
    from nedm.quadruped.camera import attach_camera
    from nedm.quadruped.constants import (FALL_TILT_RAD, FOOT_BODIES, GRAVITY,
                                          SOIL_PRESETS, STAND_ACTION)
    from nedm.quadruped.policy import PolicyController
    from nedm.quadruped.robot import Go2Robot
    from nedm.quadruped.terrain import (add_soil_visual_proxy, build_crm,
                                        build_rigid_ground, measure_leg_reach)


    cwd_at_start = os.getcwd()

    import pychrono as chrono
    import pychrono.vehicle as veh
    fsi = None
    if args.terrain == "crm":
        try:
            import pychrono.fsi as fsi
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: pychrono.fsi unavailable ({type(exc).__name__}). "
                  "CRM needs the nedm env; --terrain rigid does not.")
            return 1

    assets = Path(args.assets)
    urdf = assets / "data/robot/go2_irrvis/urdf/go2_description.urdf"
    ckpt = assets / "data/rl_models/rslrl/model_2999.pt"
    cfgs = assets / "data/rl_models/rslrl/cfgs.pkl"
    for f in (urdf, ckpt, cfgs):
        if not f.is_file():
            print(f"FAIL: missing {f}")
            return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    system = chrono.ChSystemSMC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -GRAVITY))
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    # 150, from playground_crm.py. The Go2 skill says 60, but that is for
    # rigid ground; a 42-body articulated robot on compliant terrain is
    # exactly where an under-converged contact solve shows up as excess bounce.
    system.GetSolver().AsIterative().SetMaxIterations(args.solver_iters)
    chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0025)
    chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0025)

    rigid = args.terrain == "rigid"
    # For CRM, Construct's pos is the BOTTOM of the soil box, and GetHeight does
    # NOT report the free surface (it is the height-functor hook, flat zero by
    # default). The surface is soil_bottom + depth. For rigid, the Go2 skill's
    # ChBodyEasyBox at z=0 puts its top at +0.05.
    soil_top = 0.05 if rigid else args.soil_bottom + args.depth
    if args.spawn_clearance is None:
        os.chdir(urdf.parent)
        try:
            leg_reach = measure_leg_reach(chrono, urdf)
        finally:
            os.chdir(cwd_at_start)
        foot_margin = args.foot_margin_spacings * args.spacing
        spawn_z = soil_top + foot_margin + leg_reach
        print(f"auto-spawn: leg reach {leg_reach:.3f} m, foot margin {foot_margin:.3f} m "
              f"({args.foot_margin_spacings:g} spacings) -> base at {spawn_z:.3f}")
    else:
        leg_reach = None
        spawn_z = soil_top + args.spawn_clearance
    init = chrono.ChFramed(chrono.ChVector3d(args.spawn_x, args.spawn_y, spawn_z),
                           chrono.QuatFromAngleZ(0.0))

    # URDF meshes are referenced relatively; resolve from the urdf directory.
    cwd = cwd_at_start
    os.chdir(urdf.parent)
    try:
        robot = Go2Robot(system, urdf, init, actuation=args.actuation)
    finally:
        os.chdir(cwd)

    foot_z = [robot.body(n).GetPos().z for n in FOOT_BODIES if robot.body(n) is not None]
    foot_clearance = min(foot_z) - soil_top if foot_z else float("nan")
    if foot_clearance < 0.05:
        print(f"WARNING: lowest foot is {foot_clearance:.3f} m above the soil surface. "
              "Below ~0.05 m the foot BCE markers take a launch impulse from the "
              "particle bed; raise --spawn-clearance.")

    if rigid:
        build_rigid_ground(chrono, system)
        terrain, coupled = None, []
    else:
        terrain, coupled = build_crm(chrono, fsi, veh, system, robot, args)
    print(f"soil top {soil_top:.3f}  spawn z {spawn_z:.3f}  "
          f"lowest foot {min(foot_z):.3f} (clearance {foot_clearance:.3f})  "
          f"FSI-coupled bodies: {len(coupled)}")
    if terrain is not None:
        print(f"SPH particles {terrain.GetNumSPHParticles()}  "
              f"boundary BCE {terrain.GetNumBoundaryBCEMarkers()}")

    manager, video_note, cam_mount = (None, "disabled", None)
    if args.camera != "none":
        if args.soil_proxy:
            add_soil_visual_proxy(chrono, system, args, soil_top)
        manager, video_note, cam_mount = attach_camera(chrono, system, args, soil_top, terrain)
        print(f"video: {video_note}")

    if args.no_policy:
        policy = None
    elif args.imported_ckpt:
        from nedm.quadruped.imported_policy import ImportedGo2Policy
        policy = ImportedGo2Policy(Path(args.imported_ckpt), command=args.command,
                                   family=args.command_family, duration=args.seconds)
        print(f"imported policy: {args.imported_ckpt}  "
              f"{'family ' + args.command_family if args.command_family else 'command ' + str(args.command)}")
    else:
        policy = PolicyController(ckpt, cfgs)

    exchange = args.exchange_mult * args.step
    control_every = max(1, int(round((1.0 / args.control_hz) / exchange)))
    n_steps = int(args.seconds / exchange)
    base = robot.base()
    z0 = base.GetPos().z
    x0 = base.GetPos().x
    y0 = base.GetPos().y
    foot_bodies = {n: robot.body(n) for n in FOOT_BODIES}
    try:
        total_mass = sum(b.GetMass() for b in system.GetBodies())
    except Exception:  # noqa: BLE001 - a diagnostic must not break the run
        total_mass = float("nan")
    log, tilts, wall0, fell_at = [], [], time.perf_counter(), None
    # BED-BOUNDARY TERMINATION. Walking off the SPH bed is not a soft failure:
    # it is an illegal memory access in SphBceManager and a core dump, with
    # nothing written. A 32 s straight run died at ~22 s exactly when the base
    # reached the far edge. Rather than pick an episode length that happens to
    # fit the geometry -- fragile, since a faster family or a different spawn
    # breaks it silently -- end the episode and record WHY, keeping the data
    # collected up to that point.
    # Margin is body half-length plus the known turning radius.
    BED_MARGIN = 0.8
    if rigid:
        bed = (-5.0, 5.0, -5.0, 5.0)          # ChBodyEasyBox(10, 10, ...) at origin
    else:
        cx = args.patch_x / 2 - 0.6           # build_crm's centre; near edge is -0.6
        bed = (cx - args.patch_x / 2, cx + args.patch_x / 2,
               -args.patch_y / 2, args.patch_y / 2)
    boundary_at = None

    # The URDF spawns at its own rest configuration, which is NOT the stand
    # pose. Commanding the stand pose directly gives ChParserURDF's position
    # motors a large step error to close in one control tick, and they close it
    # by launching the robot: measured 9.2 cm of RISE in the first 200 ms on a
    # robot commanded only to hold still, followed by a topple at 1.2 s. Nothing
    # about soil compliance makes a stationary robot go up. Ramp instead.
    q0 = robot.joint_pos().astype(np.float64)
    initial_error = np.abs(q0 - STAND_ACTION)
    print(f"initial joint error vs stand pose: max {initial_error.max():.3f} rad, "
          f"mean {initial_error.mean():.3f} rad")

    robot.actuate(q0)
    sph_probe = soilprobe.bind_probe(terrain)

    for i in range(n_steps):
        t = i * exchange
        if i % control_every == 0:
            if t < args.pose_ramp_seconds:
                a = t / max(args.pose_ramp_seconds, 1e-9)
                robot.actuate(q0 + a * (STAND_ACTION - q0))
            elif policy is None or t < args.pose_ramp_seconds + args.settle_seconds:
                robot.actuate(STAND_ACTION)
            else:
                if hasattr(policy, "set_time"):
                    policy.set_time(t)
                robot.actuate(policy.act(robot))
        # PD runs every PHYSICS step, not every control step -- see apply_pd.
        robot.apply_pd()
        if terrain is not None:
            terrain.DoStepDynamics(exchange)   # advances BOTH fluid and multibody
        else:
            system.DoStepDynamics(exchange)
        if cam_mount is not None and args.camera in ("follow", "overhead"):
            # Translate the camera mount with the robot, so the pose the camera
            # holds relative to it is preserved. A fixed frame loses a walking
            # robot: the first CRM walk left shot at t=7.40 of 8 s, and the
            # RoboSimian framing only survived because it barely moved.
            if args.camera == "overhead":
                # Overhead tracks in BOTH axes: looking straight down, lateral
                # drift walks the robot out of frame just as forward travel does.
                cam_mount.SetPos(chrono.ChVector3d(base.GetPos().x - x0,
                                                   base.GetPos().y - y0, 0.0))
            else:
                cam_mount.SetPos(chrono.ChVector3d(base.GetPos().x - x0, 0.0, 0.0))
        if manager is not None:
            manager.Update()
        p, q = base.GetPos(), base.GetRot()
        # Soil surface under each foot, sampled from SPH particle z directly.
        # Renderer-independent, so it answers "is there a depression" without
        # depending on whether the sprite path can draw one. 95th percentile of
        # particles within SOIL_PROBE_R of the foot's XY, minus the same statistic
        # for a fixed undisturbed control patch: the difference is the local
        # surface displacement. A STATIC proxy foot gave only 2 mm; a walking foot
        # lands with well above its static share of body weight, so the two are
        # not interchangeable.
        soil_z, soil_ctrl = ([float("nan")] * len(FOOT_BODIES), float("nan"))
        if i % control_every == 0:
            soil_z, soil_ctrl = soilprobe.sample(sph_probe, robot)
        fz, ffz = [], []
        for n in FOOT_BODIES:
            b = foot_bodies.get(n)
            fz.append(b.GetPos().z if b is not None else float("nan"))
            if b is not None and terrain is not None:
                try:
                    ffz.append(float(terrain.GetFsiBodyForce(b).z))
                except Exception:  # noqa: BLE001
                    ffz.append(float("nan"))
            else:
                ffz.append(float("nan"))
        log.append([t, p.x, p.y, p.z, q.e0, q.e1, q.e2, q.e3, *fz, *ffz, *soil_z, soil_ctrl])
        # Tilt from upright, NOT base height. A 6 s run reported PASS on a robot
        # lying inverted on the soil: base z was 0.0074, still nominally above a
        # soil top of 0.0, because 7 mm off the ground is what lying down looks
        # like. Height measures something adjacent to falling; angle measures
        # falling. Same failure the RoboSimian roll metric had.
        up_z = 1.0 - 2.0 * (q.e1 * q.e1 + q.e2 * q.e2)
        tilt = math.acos(max(-1.0, min(1.0, up_z)))
        tilts.append(tilt)
        if boundary_at is None and not (
                bed[0] + BED_MARGIN <= p.x <= bed[1] - BED_MARGIN
                and bed[2] + BED_MARGIN <= p.y <= bed[3] - BED_MARGIN):
            boundary_at = t
            print(f"bed boundary at t={t:.2f}s, base ({p.x:+.2f}, {p.y:+.2f}); "
                  f"bed x[{bed[0]:.2f},{bed[1]:.2f}] y[{bed[2]:.2f},{bed[3]:.2f}] "
                  f"margin {BED_MARGIN}")
            break
        if fell_at is None and (tilt > FALL_TILT_RAD or p.z < soil_top - 0.05):
            fell_at = t

    wall = time.perf_counter() - wall0
    arr = np.asarray(log)
    np.savez_compressed(out / "trajectory.npz", log=arr,
                        columns=np.array(["t", "x", "y", "z", "e0", "e1", "e2", "e3",
                                          *[f"footz_{n}" for n in FOOT_BODIES],
                                          *[f"fsiFz_{n}" for n in FOOT_BODIES],
                                          *[f"soilz_{n}" for n in FOOT_BODIES],
                                          "soilz_ctrl"]))
    n_frames = len(list((out / "frames").glob("*"))) if (out / "frames").is_dir() else 0
    summary = {
        "sim_seconds": args.seconds, "wall_seconds": round(wall, 1),
        "realtime_factor": round(args.seconds / wall, 5) if wall else None,
        "rtf_cfd": round(float(terrain.GetRtfCFD()), 5) if terrain else None,
        "rtf_mbd": round(float(terrain.GetRtfMBD()), 5) if terrain else None,
        "fsi_coupled_bodies": len(coupled), "coupled_names": coupled,
        "terrain": args.terrain,
        "system_total_mass_kg": round(float(total_mass), 3),
        "weight_n": round(float(total_mass * GRAVITY), 1),
        "solver_iters": args.solver_iters,
        "artificial_viscosity": args.artificial_viscosity,
        "soil_preset": args.soil,
        "soil": {**SOIL_PRESETS[args.soil],
                 **({"young": args.soil_young} if args.soil_young is not None else {}),
                 **({"cohesion": args.soil_cohesion} if args.soil_cohesion is not None else {})},
        "sph_particles": int(terrain.GetNumSPHParticles()) if terrain else 0,
        "soil_top_m": soil_top, "spawn_z_m": spawn_z,
        "forward_travel_m": round(float(arr[-1, 1] - arr[0, 1]), 4),
        "lateral_travel_m": round(float(arr[-1, 2] - arr[0, 2]), 4),
        "base_z_start_end_m": [round(z0, 4), round(float(arr[-1, 3]), 4)],
        "base_z_min_m": round(float(arr[:, 3].min()), 4),
        "fell": bool(fell_at is not None), "fell_at_s": fell_at,
        # Three outcomes, not two. bed_boundary is a clean stop with usable data,
        # not a failure -- distinguishing it from a fall matters because the
        # trajectory up to that point is fine.
        "status": ("fell" if fell_at is not None
                   else "bed_boundary" if boundary_at is not None else "completed"),
        "bed_boundary_at_s": boundary_at,
        "command_family": args.command_family,
        "command_series": getattr(policy, "command_log", None),
        "max_tilt_deg": round(math.degrees(max(tilts)), 1) if tilts else None,
        "final_tilt_deg": round(math.degrees(tilts[-1]), 1) if tilts else None,
        "policy": "none (stand pose)" if policy is None else "model_2999.pt",
        "pose_ramp_s": args.pose_ramp_seconds,
        "check_embedded": args.check_embedded,
        "foot_clearance_above_soil_m": round(float(foot_clearance), 4),
        "leg_reach_m": round(leg_reach, 4) if leg_reach is not None else None,
        "initial_joint_error_max_rad": round(float(initial_error.max()), 4),
        "initial_joint_error_mean_rad": round(float(initial_error.mean()), 4),
        "camera": args.camera, "video": video_note, "frames_written": n_frames,
    }
    if args.camera != "none" and n_frames:
        try:
            from PIL import Image
            dst = out / "jpg"
            dst.mkdir(exist_ok=True)
            # NUMERIC sort. Chrono names frames frame_0.png, frame_1.png,
            # frame_10.png, so lexicographic order is not chronological and the
            # clip plays the robot flipping back and forth at random. Caught
            # only because the frames contradicted the trajectory.
            def _frame_index(path):
                digits = "".join(c for c in path.stem if c.isdigit())
                return int(digits) if digits else -1
            for i, png in enumerate(sorted((out / "frames").glob("*.png"), key=_frame_index)):
                Image.open(png).convert("RGB").save(dst / f"f{i:05d}.jpg", quality=85)
            summary["frames_jpeg"] = str(dst)
        except Exception as exc:  # noqa: BLE001
            summary["frames_jpeg"] = f"transcode skipped ({type(exc).__name__})"

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nGATE: " + (f"FAIL, fell at {fell_at:.2f}s (max tilt {summary['max_tilt_deg']} deg)"
                    if summary["fell"] else
                    f"PASS, upright for the full window (max tilt {summary['max_tilt_deg']} deg)"))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
