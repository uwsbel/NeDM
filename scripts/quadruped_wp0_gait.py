#!/usr/bin/env python
"""Quadruped WP0 vertical slice: a scripted gait, headless, zero learning.

The analogue of Study 3's WP0a oracle slice, for the proposed quadruped case
study (docs/state/progress/future-case-studies.md). That doc names bootstrapping
as the study's real risk: locomotion cannot be trained in Chrono + CRM, and a
random-action quadruped falls in ~0.4 s, so the seed controller has to be a
scripted gait. This proves the scripted gait exists and walks before any model
work starts. If it fails, the case study changes shape.

Two subcommands:

  cycle   Read the actuation file and report its period. Answers the third entry
          in docs/state/decisions/open-questions.md (is a 16-token context at
          50 Hz, 0.32 s, long enough for a gait?) with no simulation at all.

  walk    Run RoboSimian on rigid ground and report whether it stayed up, how
          fast it went, and the realtime factor.

One artifact to expect, so nobody chases it as physics: walking_cycle.txt is
exactly one stride (verified frame-by-frame -- no interior frame returns to frame
0, the nearest is the last one), but the loop is not perfectly closed. The gap
|q[-1] - q[0]| is 0.0524 rad, about 3 degrees across 32 joints, so every time
RS_Driver wraps end to start it steps the joint targets by that much. Measured
over two cycles, this is NOT observable in the chassis trajectory: the largest
single-step lateral move in a 40 s run was 0.14 mm and did not coincide with
either wrap. Noted for anyone reading joint targets directly, not as a transient
to watch for.

Scope, stated plainly: RoboSimian is 32-DOF, so this validates the gait and
contact machinery, NOT the ~40-D Go2-shaped z1 in the plan. And this build has
no pychrono.fsi, so rigid ground is the only terrain available here; CRM needs a
source build with FSI. See docs/state/machines/kyle-sbel.md.

Usage (from repo root):
  "$NEDM_PY" scripts/quadruped_wp0_gait.py cycle
  "$NEDM_PY" scripts/quadruped_wp0_gait.py walk --sim-seconds 10 \
      --out artifacts/quadruped/wp0_walk
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

CYCLE_REL = "robot/robosimian/actuation/walking_cycle.txt"

# Demo values (demo_ROBOT_RoboSimian_Rigid.py), kept so results are comparable.
TIME_STEP = 1e-3
DURATION_POSE = 1.0      # hold the initial pose, robot fixed
DURATION_SETTLE = 0.5    # settle onto terrain once released
TERRAIN_LENGTH = 8.0
TERRAIN_WIDTH = 2.0
FALL_DROP_M = 0.25       # chassis drop below its released height that counts as a fall
OUTPUT_FPS = 100         # robot.Output() rate, matching the shipped demo


def rpy_from_quat(e0: float, e1: float, e2: float, e3: float) -> tuple[float, float, float]:
    """Roll/pitch/yaw from a Chrono quaternion, computed here to avoid API drift."""
    roll = math.atan2(2.0 * (e0 * e1 + e2 * e3), 1.0 - 2.0 * (e1 * e1 + e2 * e2))
    s = max(-1.0, min(1.0, 2.0 * (e0 * e2 - e3 * e1)))
    pitch = math.asin(s)
    yaw = math.atan2(2.0 * (e0 * e3 + e1 * e2), 1.0 - 2.0 * (e2 * e2 + e3 * e3))
    return roll, pitch, yaw


def dominant_period_s(signal: np.ndarray, dt: float) -> float | None:
    """First autocorrelation peak after the zero lag, in seconds."""
    x = signal - signal.mean()
    if not np.any(x):
        return None
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    below = np.where(ac < 0.0)[0]
    if len(below) == 0:
        return None
    start = int(below[0])
    if start >= len(ac) - 1:
        return None
    return float((start + int(np.argmax(ac[start:]))) * dt)


def cmd_cycle(args: argparse.Namespace) -> int:
    import pychrono as chrono

    path = Path(chrono.GetChronoDataFile(CYCLE_REL))
    if not path.is_file():
        print(f"FAIL: actuation file not found at {path}")
        print(f"      Chrono data path is {chrono.GetChronoDataPath()!r}")
        return 1

    data = np.loadtxt(path)
    t, joints = data[:, 0], data[:, 1:]
    dts = np.diff(t)
    dt = float(np.median(dts))
    duration = float(t[-1] - t[0])

    periods = [p for p in (dominant_period_s(joints[:, j], dt) for j in range(joints.shape[1])) if p]

    out = {
        "file": str(path),
        "rows": int(data.shape[0]),
        "n_joints": int(joints.shape[1]),
        "dt_s": dt,
        "dt_uniform": bool(np.allclose(dts, dt, atol=1e-9)),
        "file_duration_s": duration,
        "autocorr_period_median_s": float(np.median(periods)) if periods else None,
        "autocorr_period_min_s": float(np.min(periods)) if periods else None,
        "autocorr_period_max_s": float(np.max(periods)) if periods else None,
        "context_16_tokens_at_50hz_s": 16 / 50.0,
    }
    print(json.dumps(out, indent=2))

    # RS_Driver replays this file on a loop, so its duration IS the period by
    # construction. Autocorrelation is only a cross-check, and a weak one: with
    # roughly a single period in the record there is almost no lag range, which
    # is why the per-joint estimates scatter.
    ctx = out["context_16_tokens_at_50hz_s"]
    print(f"\nRoboSimian gait period = {duration:.2f} s (the cycle file's own duration, "
          f"replayed on a loop) against a {ctx:.2f} s context: {duration / ctx:.0f}x too short.")
    print("Autocorrelation cross-check spans "
          f"{out['autocorr_period_min_s'] or float('nan'):.1f} to "
          f"{out['autocorr_period_max_s'] or float('nan'):.1f} s; do not quote it, "
          "the record holds about one period so the estimate is unresolved.")
    print("\nCAVEAT, and it decides the answer: RoboSimian is a slow statically-stable")
    print("walker. The plan targets a Go2 DYNAMIC TROT at 0.3-0.5 s, which needs only")
    print("15-25 tokens at 50 Hz. So this number does NOT transfer, and 'lengthen the")
    print("context' stays viable for Go2 even though it is hopeless here.")
    return 0


def up_axis_from_quat(e0: float, e1: float, e2: float, e3: float) -> tuple[float, float, float]:
    """The body's local +Z expressed in world coordinates."""
    return (2.0 * (e1 * e3 + e0 * e2),
            2.0 * (e2 * e3 - e0 * e1),
            e0 * e0 - e1 * e1 - e2 * e2 + e3 * e3)



def look_at_frame(chrono, eye, target):
    """A Chrono frame at `eye` looking at `target`.

    Chrono::Sensor cameras look down +X of their own frame with +Z up. The
    rotation is built in numpy and handed over as an explicit quaternion rather
    than going through a Chrono look-at helper, whose name has moved between
    versions.
    """
    eye = np.asarray(eye, dtype=float)
    fwd = np.asarray(target, dtype=float) - eye
    fwd /= np.linalg.norm(fwd)
    up_hint = np.array([0.0, 0.0, 1.0])
    if abs(float(fwd @ up_hint)) > 0.999:
        up_hint = np.array([0.0, 1.0, 0.0])
    left = np.cross(up_hint, fwd)
    left /= np.linalg.norm(left)
    up = np.cross(fwd, left)
    r = np.column_stack([fwd, left, up])
    tr = float(np.trace(r))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        e0, e1, e2, e3 = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (r[k, j] - r[j, k]) / s
        q[i + 1], q[j + 1], q[k + 1] = 0.25 * s, (r[j, i] + r[i, j]) / s, (r[k, i] + r[i, k]) / s
        e0, e1, e2, e3 = q
    return chrono.ChFramed(chrono.ChVector3d(*eye), chrono.ChQuaterniond(e0, e1, e2, e3))


def attach_video_camera(chrono, sys_, args):
    """Offscreen chase camera. Returns (manager, note) or (None, reason)."""
    try:
        import pychrono.sensor as sens
    except Exception as exc:  # noqa: BLE001
        return None, f"pychrono.sensor unavailable ({type(exc).__name__}: {exc})"

    mount = chrono.ChBody()
    mount.SetFixed(True)
    mount.EnableCollision(False)
    sys_.AddBody(mount)

    manager = sens.ChSensorManager(sys_)
    manager.scene.SetAmbientLight(chrono.ChVector3f(0.35, 0.35, 0.38))
    # pychrono 9.0.0's ChScene has no directional light: only AddPointLight and
    # AddAreaLight exist. src/nedm/traverse/scene.py calls AddDirectionalLight
    # and therefore cannot render on either of Kyle's boxes; see
    # docs/state/progress/vision-study3-traverse.md. Prefer it when present so
    # this stays correct on a newer build, and fall back to a key/fill pair.
    if hasattr(manager.scene, "AddDirectionalLight"):
        manager.scene.AddDirectionalLight(chrono.ChColor(1.0, 0.95, 0.85),
                                          math.radians(55.0), math.radians(120.0))
    else:
        manager.scene.AddPointLight(chrono.ChVector3f(2.0, -2.5, 3.0),
                                    chrono.ChColor(1.0, 0.95, 0.85), 25.0)
        manager.scene.AddPointLight(chrono.ChVector3f(-2.0, 1.5, 2.0),
                                    chrono.ChColor(0.45, 0.5, 0.6), 25.0)
    background = sens.Background()
    background.mode = sens.BackgroundMode_SOLID_COLOR
    background.color_zenith = chrono.ChVector3f(0.55, 0.68, 0.85)
    manager.scene.SetBackground(background)

    # Fixed three-quarter view. The robot travels ~0.2 m per cycle, so it stays
    # in frame for the whole run without tracking. A camera parented to the
    # chassis would inherit its 180 deg X initialization and film upside down.
    pose = look_at_frame(chrono, (-1.1, -1.6, 0.75), (0.35, 0.0, 0.05))
    cam = sens.ChCameraSensor(mount, float(args.video_fps), pose,
                              args.video_width, args.video_height, math.radians(55.0))
    cam.SetName("chase")
    cam.SetLag(0.0)
    cam.SetCollectionWindow(0.0)

    frame_dir = Path(args.out) / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    save = getattr(sens, "ChFilterSave", None)
    if save is None:
        return None, "sens.ChFilterSave not available in this build"
    cam.PushFilter(save(str(frame_dir) + "/"))
    manager.AddSensor(cam)
    return manager, f"frames -> {frame_dir}"


def transcode_frames_to_jpeg(out_dir: Path, quality: int) -> str:
    """PNG frames to JPEG, so a run can cross a relayed link without ffmpeg.

    Neither box has ffmpeg and installing it needs a human, but PIL is present.
    JPEG at q85 is roughly a fifth the size of the PNGs, which is the difference
    between a transfer that finishes and one that does not.
    """
    src = out_dir / "frames"
    if not src.is_dir():
        return "no frames directory"
    pngs = sorted(src.glob("*.png"))
    if not pngs:
        return "no frames rendered"
    try:
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        return f"{len(pngs)} png frames, PIL unavailable ({type(exc).__name__})"
    dst = out_dir / "jpg"
    dst.mkdir(exist_ok=True)
    for i, png in enumerate(pngs):
        Image.open(png).convert("RGB").save(dst / f"f{i:05d}.jpg", quality=quality)
    mb = sum(p.stat().st_size for p in dst.glob("*.jpg")) / 1e6
    return f"{len(pngs)} frames -> {dst} ({mb:.1f} MB)"


def cmd_walk(args: argparse.Namespace) -> int:
    import pychrono as chrono
    import pychrono.robot as robosimian

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cycle_file = chrono.GetChronoDataFile(CYCLE_REL)
    if not Path(cycle_file).is_file():
        print(f"FAIL: actuation file not found at {cycle_file}")
        return 1

    # RS_Driver loops this file, so its duration is the gait period. A window
    # shorter than one period measures part of a stride, not locomotion.
    cycle_period = float(np.loadtxt(cycle_file, usecols=0)[-1])
    if args.sim_seconds < cycle_period:
        print(f"WARNING: --sim-seconds {args.sim_seconds} is shorter than the "
              f"{cycle_period:.2f} s gait period. Travel and speed will describe "
              f"part of one stride. Use at least {math.ceil(2 * cycle_period)}.")

    sys_ = chrono.ChSystemSMC()
    sys_.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    # Solver type before iteration count. The shipped demo does the reverse,
    # which sets iterations on the solver it is about to replace.
    sys_.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    sys_.GetSolver().AsIterative().SetMaxIterations(200)
    sys_.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.8))

    # has_sled=True, fixed=True. Fixed matters: the robot holds its pose while
    # the driver poses it, and is released only after terrain exists.
    robot = robosimian.RoboSimian(sys_, True, True)
    robot.SetOutputDirectory(str(out_dir))
    robot.Initialize(
        chrono.ChCoordsysd(chrono.ChVector3d(0, 0, 0), chrono.QuatFromAngleX(chrono.CH_PI))
    )

    # WALK uses the cycle file alone; start and stop are empty, repeat=True.
    driver = robosimian.RS_Driver("", cycle_file, "", True)
    cbk = robosimian.RS_DriverCallback(robot)
    driver.RegisterPhaseChangeCallback(cbk)
    driver.SetTimeOffsets(DURATION_POSE, DURATION_SETTLE)
    robot.SetDriver(driver)

    time_create_terrain = DURATION_POSE
    time_start = time_create_terrain + DURATION_SETTLE
    time_end = time_start + args.sim_seconds
    output_every = max(1, math.ceil((1.0 / OUTPUT_FPS) / TIME_STEP))

    video_manager, video_note = (None, "disabled")
    if args.video:
        video_manager, video_note = attach_video_camera(chrono, sys_, args)
        print(f"video: {video_note}")
        if video_manager is None:
            print("video: continuing without capture")

    log: list[list[float]] = []
    terrain_created = False
    released_z = None
    up0 = None
    fell_at = None
    step = 0
    wall0 = time.perf_counter()

    while sys_.GetChTime() < time_end:
        if not terrain_created and sys_.GetChTime() > time_create_terrain:
            # Ground height comes from the settled robot, not a constant. Creating
            # it up front at a guessed z gives interpenetration or a drop.
            z = robot.GetWheelPos(robosimian.FR).z - 0.15
            _create_terrain(chrono, sys_, TERRAIN_LENGTH, TERRAIN_WIDTH, z, TERRAIN_LENGTH / 4)
            _set_contact_properties(chrono, robot)
            robot.GetChassisBody().SetFixed(False)
            terrain_created = True
            body = robot.GetChassisBody()
            released_z = body.GetPos().z
            r = body.GetRot()
            up0 = up_axis_from_quat(r.e0, r.e1, r.e2, r.e3)

        robot.DoStepDynamics(TIME_STEP)
        if video_manager is not None:
            video_manager.Update()
        step += 1

        t = sys_.GetChTime()
        if terrain_created and t >= time_start:
            if step % output_every == 0:
                robot.Output()  # per-limb .dat channel; empty without this call
            body = robot.GetChassisBody()
            pos, rot = body.GetPos(), body.GetRot()
            roll, pitch, yaw = rpy_from_quat(rot.e0, rot.e1, rot.e2, rot.e3)
            up = up_axis_from_quat(rot.e0, rot.e1, rot.e2, rot.e3)
            # Tilt from the pose the robot was released in. Absolute roll is
            # useless here: RoboSimian is initialized 180 deg about X, so roll
            # sits at +/-pi for the whole run by construction.
            dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(up, up0))))
            tilt = math.acos(dot)
            log.append([t, pos.x, pos.y, pos.z, roll, pitch, yaw, tilt])
            if fell_at is None and released_z is not None and pos.z < released_z - FALL_DROP_M:
                fell_at = t

    wall = time.perf_counter() - wall0
    arr = np.asarray(log, dtype=np.float64)
    np.savez_compressed(
        out_dir / "trajectory.npz", log=arr,
        columns=np.array(["t", "x", "y", "z", "roll", "pitch", "yaw", "tilt_from_release"]),
    )

    try:
        avg_speed = float(cbk.GetAvgSpeed())
    except Exception as exc:  # noqa: BLE001 - reporting beats crashing the gate
        avg_speed = None
        print(f"note: GetAvgSpeed unavailable ({type(exc).__name__}: {exc})")

    # Displacement measured between driver wrap events, not window endpoints.
    # RS_Driver wraps at time_start + k*period, so the boundaries are known
    # without parsing its stdout. Endpoint-to-endpoint over a fractional number
    # of cycles samples intra-stride oscillation at an arbitrary phase: x moves
    # backward mid-stride, and y sways about +/-77 mm and closes on itself, so a
    # window measurement reads sway as drift and inflates stride by ~10%.
    per_cycle = []
    k = 0
    while cycle_period and time_start + (k + 1) * cycle_period <= time_end + 1e-9:
        t0 = time_start + k * cycle_period
        t1 = t0 + cycle_period
        seg = arr[(arr[:, 0] >= t0 - 1e-9) & (arr[:, 0] <= t1 + 1e-9)] if len(arr) else arr
        if len(seg) < 2:
            break
        per_cycle.append({
            "cycle": k + 1,
            "stride_length_m": round(float(seg[-1, 1] - seg[0, 1]), 4),
            "net_lateral_m": round(float(seg[-1, 2] - seg[0, 2]), 4),
            "lateral_sway_p2p_m": round(float(seg[:, 2].max() - seg[:, 2].min()), 4),
        })
        k += 1
    strides = [c["stride_length_m"] for c in per_cycle]
    dx = float(arr[-1, 1] - arr[0, 1]) if len(arr) else 0.0
    dy = float(arr[-1, 2] - arr[0, 2]) if len(arr) else 0.0
    cycles = args.sim_seconds / cycle_period if cycle_period else None
    summary = {
        "sim_seconds": args.sim_seconds,
        "gait_period_s": round(cycle_period, 3),
        "cycles_covered": round(cycles, 3) if cycles else None,
        "window_covers_full_cycle": bool(cycles and cycles >= 1.0),
        "wall_seconds": round(wall, 1),
        "realtime_factor": round(args.sim_seconds / wall, 4) if wall else None,
        "fell": fell_at is not None,
        "fell_at_s": fell_at,
        "complete_cycles": len(per_cycle),
        "per_cycle": per_cycle,
        "stride_length_m_mean": round(float(np.mean(strides)), 4) if strides else None,
        "speed_mps_from_stride": round(float(np.mean(strides)) / cycle_period, 5)
        if strides and cycle_period else None,
        # Window endpoints, kept only because Chrono's own GetAvgSpeed shares
        # their flaw. Neither is stride length; use the per-cycle values.
        "window_endpoint_dx_m": round(dx, 4),
        "window_endpoint_dy_m": round(dy, 4),
        "avg_speed_mps_callback": avg_speed,
        # Tilt from the released pose. max_abs_roll is deliberately absent: it is
        # a constant of the 180 deg initialization, not a measurement.
        "max_tilt_from_release_rad": round(float(arr[:, 7].max()), 4) if len(arr) else None,
        "chassis_z_range_m": [round(float(arr[:, 3].min()), 5),
                              round(float(arr[:, 3].max()), 5)] if len(arr) else None,
        "samples": int(len(arr)),
        "video": video_note,
        # Rendering dominates wall time, so realtime_factor from a --video run
        # understates the solver. Quote the figure from a run without it.
        "realtime_factor_is_comparable": not args.video,
    }
    if args.video and video_manager is not None:
        summary["frames_jpeg"] = transcode_frames_to_jpeg(out_dir, args.video_quality)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    verdict = "FAIL, robot fell" if summary["fell"] else "PASS, stayed upright for the full window"
    if not per_cycle:
        verdict += " (stride NOT measured: window is under one gait period)"
    print("\nGATE: " + verdict)
    if strides:
        print(f"Stride {np.mean(strides):.4f} m/cycle over {len(strides)} cycle(s); "
              f"window endpoints would say {dx / cycles:.4f} m/cycle.")
    return 0


def _create_terrain(chrono, sys_, length, width, height, offset):
    mat = chrono.ChContactMaterial.DefaultMaterial(sys_.GetContactMethod())
    mat.SetFriction(0.8)
    mat.SetRestitution(0)
    if sys_.GetContactMethod() == chrono.ChContactMethod_SMC:
        chrono.CastToChContactMaterialSMC(mat).SetYoungModulus(1e7)

    ground = chrono.ChBody()
    ground.SetFixed(True)
    ground.EnableCollision(True)
    ground.AddCollisionShape(
        chrono.ChCollisionShapeBox(mat, length, width, 0.2),
        chrono.ChFramed(chrono.ChVector3d(offset, 0, height - 0.1), chrono.QUNIT),
    )
    sys_.GetCollisionSystem().BindItem(ground)
    sys_.AddBody(ground)
    return ground


def _set_contact_properties(chrono, robot):
    friction, young, restitution = 0.8, 1e7, 0.0
    for get in (robot.GetSledContactMaterial, robot.GetWheelContactMaterial):
        mat = get()
        mat.SetFriction(friction)
        mat.SetRestitution(restitution)
    if robot.GetSystem().GetContactMethod() == chrono.ChContactMethod_SMC:
        chrono.CastToChContactMaterialSMC(robot.GetSledContactMaterial()).SetYoungModulus(young)
        chrono.CastToChContactMaterialSMC(robot.GetWheelContactMaterial()).SetYoungModulus(young)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cycle", help="report the actuation file's gait period")
    walk = sub.add_parser("walk", help="run the scripted gait on rigid ground")
    walk.add_argument("--sim-seconds", type=float, default=40.0,
                      help="default covers ~2 RoboSimian gait cycles (~19.2 s each)")
    walk.add_argument("--out", default="artifacts/quadruped/wp0_walk")
    walk.add_argument("--video", action="store_true",
                      help="capture offscreen frames via Chrono::Sensor (needs a GPU)")
    walk.add_argument("--video-fps", type=float, default=30.0)
    walk.add_argument("--video-width", type=int, default=960)
    walk.add_argument("--video-height", type=int, default=540)
    walk.add_argument("--video-quality", type=int, default=85,
                      help="JPEG quality for the transferable copy")
    args = parser.parse_args()
    return cmd_cycle(args) if args.cmd == "cycle" else cmd_walk(args)


if __name__ == "__main__":
    raise SystemExit(main())
