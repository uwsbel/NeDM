"""Milestone 1 gate for the scripted trot: stand, then trot. Rigid ground only.

TWO RUNS, AND THE FAILING OUTCOME OF EACH IS NAMED BEFORE EITHER IS RUN.

  STAND  hold the IK rest pose for 5 s.
         FAILS if the base falls more than 0.05 m, or |roll| or |pitch| exceeds
         0.20 rad. If this fails the IK or the frame convention is wrong and
         nothing after it means anything.

  TROT   f = 2.0 Hz, D = 0.6, h = 0.08 m, vx = 0.5 m/s, 10 s.
         Reports distance travelled, whether it stayed upright, and realised vs
         commanded speed. There is no pass threshold on speed -- a hand-tuned
         open-loop trot on a 15 kg quadruped is genuinely hard and the point of
         this gate is to find out WHICH failure mode appears, not to tune.

The failure mode is the output, not the tuning. Reports pitch, roll, foot slip
and knee sign separately so "pitches forward", "rolls", "feet slip" and "knee
inverts" are distinguishable -- they have very different costs.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def run(mode: str, seconds: float, params_kw: dict, command: tuple[float, float, float],
        assets: str) -> dict:
    import pychrono as chrono
    from nedm.quadruped.constants import FOOT_BODIES, GRAVITY, MOTOR_NAMES
    from nedm.quadruped.robot import Go2Robot
    from nedm.quadruped.terrain import build_rigid_ground
    from nedm.quadruped.trot import (GaitParams, TrotController, extract_geometry,
                                     measure_base_com_offset)

    urdf = Path(assets) / "data/robot/go2_irrvis/urdf/go2_description.urdf"
    geo = extract_geometry(urdf)
    params = GaitParams(**params_kw)

    step, exchange_mult, control_hz = 5e-4, 4, 50.0
    exchange = step * exchange_mult

    system = chrono.ChSystemSMC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -GRAVITY))
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    system.GetSolver().AsIterative().SetMaxIterations(150)
    chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0025)
    chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0025)

    ground_top = 0.05
    spawn_z = ground_top + params.stand_height_m
    cwd = os.getcwd()
    os.chdir(urdf.parent)
    try:
        robot = Go2Robot(system, urdf,
                         chrono.ChFramed(chrono.ChVector3d(0, 0, spawn_z),
                                         chrono.QuatFromAngleZ(0.0)),
                         actuation="torque")
    finally:
        os.chdir(cwd)
    geo.base_com_offset = measure_base_com_offset(chrono, robot, (0.0, 0.0, spawn_z))
    build_rigid_ground(chrono, system)

    ctrl = TrotController(geo, params)
    ctrl.set_command(*command)
    base = robot.base()

    # z_spawn is recorded BEFORE the settle. Reading it after made a collapse
    # invisible: the robot fell 0.26 m during the settle and the gate then
    # measured a drop of 0.0000 from the floor to the floor, and passed.
    z_spawn = float(base.GetPos().z)
    hold = ctrl.joint_targets()
    robot.actuate(hold)
    for _ in range(int(0.75 / exchange)):
        robot.apply_pd()
        system.DoStepDynamics(exchange)

    p0 = base.GetPos()
    z0 = float(p0.z)
    z_settle_drop = z_spawn - z0
    control_every = max(1, int(round((1.0 / control_hz) / exchange)))
    n = int(seconds / exchange)
    roll_max = pitch_max = tilt_max = 0.0
    z_min = z0
    slips = []
    for i in range(n):
        if i % control_every == 0:
            if mode == "trot":
                ctrl.advance(control_every * exchange)
            # GetCardanAnglesZYX -- the same call dataset.py:292 uses for the
            # roll_rad/pitch_rad columns. The previous version guarded a
            # non-existent Q_to_Euler123 with hasattr and fell back to 0.0, so
            # roll and pitch read exactly zero for the whole run and BOTH
            # attitude checks passed without measuring anything.
            rot = base.GetFrameRefToAbs().GetRot()
            e = rot.GetCardanAnglesZYX()
            roll, pitch = float(e.x), float(e.y)
            # CARDAN ZYX IS DEGENERATE near pitch = +/- pi and reports roll ~ -pi/2
            # with pitch ~ pi for an upright-looking pose. Tilt -- the angle between
            # the body +z and world +z -- has no such singularity and is the
            # quantity the pass/fail decision uses. Cardan is kept only because
            # dataset.py reports those columns.
            up = rot.Rotate(chrono.ChVector3d(0, 0, 1))
            tilt = math.acos(max(-1.0, min(1.0, float(up.z))))
            tilt_max = max(tilt_max, tilt)
            w = base.GetAngVelLocal()
            robot.actuate(ctrl.joint_targets(roll, pitch, float(w.x), float(w.y)))
            roll_max = max(roll_max, abs(roll))
            pitch_max = max(pitch_max, abs(pitch))
        robot.apply_pd()
        system.DoStepDynamics(exchange)
        z_min = min(z_min, float(base.GetPos().z))
        if i % 200 == 0:
            for nm in FOOT_BODIES:
                b = robot.body(nm)
                if b is not None and b.GetPos().z < ground_top + 0.02:
                    v = b.GetPosDt()
                    slips.append(math.hypot(v.x, v.y))

    p1 = base.GetPos()
    q = robot.joint_pos()
    idx = {nm.removesuffix("_joint"): k for k, nm in enumerate(MOTOR_NAMES)}
    calf_signs = {leg: float(q[idx[f"{leg}_calf"]]) for leg in ("FR", "FL", "RR", "RL")}
    return {
        "mode": mode, "seconds": seconds,
        "dx": float(p1.x - p0.x), "dy": float(p1.y - p0.y),
        "distance": float(math.hypot(p1.x - p0.x, p1.y - p0.y)),
        "z_spawn": z_spawn, "z_settle_drop": z_settle_drop,
        "z_start": z0, "z_end": float(p1.z), "z_min": z_min, "z_drop": z0 - z_min,
        "stand_height_target": params.stand_height_m,
        "roll_max": roll_max, "pitch_max": pitch_max, "tilt_max": tilt_max,
        "mean_foot_slip": float(np.mean(slips)) if slips else float("nan"),
        "calf_angles": calf_signs,
        "unreachable": ctrl.unreachable_count,
        "realised_speed": float(math.hypot(p1.x - p0.x, p1.y - p0.y) / seconds),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=os.environ.get(
        "NEDM_GO2_ASSETS",
        "/home/kyle/Documents/sbel/sbel-reproducibility/2025/multi-terrain-RL"))
    a = ap.parse_args()

    print("GATE 1: STAND, 5 s, hold the IK rest pose")
    s = run("stand", 5.0, dict(frequency_hz=0.0), (0.0, 0.0, 0.0), a.assets)
    # The binding condition is that the base is still NEAR THE COMMANDED HEIGHT,
    # not merely that it stopped moving. A collapsed robot is perfectly stable.
    expected_z = 0.05 + s["stand_height_target"]
    height_ok = abs(s["z_end"] - expected_z) < 0.05
    stand_ok = height_ok and s["z_drop"] < 0.05 and s["tilt_max"] < 0.20
    print(f"  spawn z {s['z_spawn']:.4f} -> after settle {s['z_start']:.4f} "
          f"(settle drop {s['z_settle_drop']:.4f}) -> end {s['z_end']:.4f}")
    print(f"  expected standing z {expected_z:.4f}   |error| {abs(s['z_end']-expected_z):.4f} m"
          f"   {'OK' if height_ok else 'COLLAPSED'}")
    print(f"  TILT max {math.degrees(s['tilt_max']):.2f} deg  "
          f"(cardan roll {s['roll_max']:.4f} pitch {s['pitch_max']:.4f}, degenerate near +/-pi)")
    print(f"  drift {s['distance']:.4f} m   unreachable targets {s['unreachable']}")
    print(f"  -> {'PASS' if stand_ok else 'FAIL'}  (fails if the base is not within "
          f"0.05 m of the commanded standing height, or drops > 0.05 m, or |att| > 0.20 rad)")
    if not stand_ok:
        print("\n  STAND FAILED -- the IK or the frame convention is wrong. Not running the trot.")
        return 1

    print("\nGATE 2: TROT, 10 s, f=2.0 Hz D=0.6 h=0.08 m, commanded vx=0.5 m/s")
    t = run("trot", 10.0, dict(frequency_hz=2.0, duty=0.6, step_height_m=0.08),
            (0.5, 0.0, 0.0), a.assets)
    upright = t["z_drop"] < 0.12 and t["tilt_max"] < 0.6
    print(f"  travelled {t['distance']:.4f} m  (dx {t['dx']:+.4f}, dy {t['dy']:+.4f})")
    print(f"  realised speed {t['realised_speed']:.4f} m/s vs commanded 0.5000")
    print(f"  base z {t['z_start']:.4f} -> {t['z_end']:.4f}   min {t['z_min']:.4f}  drop {t['z_drop']:.4f}")
    print(f"  TILT max {math.degrees(t['tilt_max']):.2f} deg")
    print(f"  mean foot slip while loaded {t['mean_foot_slip']:.4f} m/s")
    print(f"  calf angles at end {({k: round(v,3) for k,v in t['calf_angles'].items()})}")
    print(f"  unreachable targets {t['unreachable']}")
    print(f"  -> upright: {'YES' if upright else 'NO'}")
    if not upright:
        print("\n  FAILURE MODE, reported rather than tuned:")
        if abs(t["pitch_max"]) > abs(t["roll_max"]) * 1.5:
            print("    PITCH dominates -- pitches forward/back")
        elif t["roll_max"] > t["pitch_max"] * 1.5:
            print("    ROLL dominates -- rolls sideways")
        else:
            print("    roll and pitch comparable -- collapses rather than tips")
        if not np.isnan(t["mean_foot_slip"]) and t["mean_foot_slip"] > 0.3:
            print(f"    FEET SLIP: {t['mean_foot_slip']:.3f} m/s while loaded")
        if any(v < 0 for v in t["calf_angles"].values()):
            print(f"    KNEE INVERTED on {[k for k,v in t['calf_angles'].items() if v<0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
