#!/usr/bin/env python3
"""Establish the joint sign convention by PHYSICS, then write it into params/policy.yaml.

The previous harness negated joint positions, velocities and targets, and its own
documentation says no source records why. A sign-flip bug followed, costing three days.
This determines the convention by running the robot and looking at what it does, which is
the one method that cannot be circular.

Why not compare stand poses. Chrono's STAND_ACTION is [0, -1.0, +1.5] per leg and the
policy's default_dof_pos is [0, +0.80, -1.50]. The calf agrees exactly under sign -1,
which LOOKS decisive -- but STAND_ACTION belongs to the old harness and may already encode
the convention being tested, so that argument assumes its conclusion. The thigh magnitudes
differ too (1.0 against 0.8), which shows the two are simply different nominal poses.

The physical test has no such problem. A policy trained to stand and walk will do neither
with its joints negated: it collapses within a fraction of a second. So run both
candidates on the ground the policy was trained on and report what happened.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]      # <repo>/quadruped/diagnostics/this.py -> <repo>
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))                   # so `quadruped.params` imports


def run(sign: float, seconds: float, command, urdf: Path, policy_path: Path,
        control_hz: float, warmup_s: float = 1.0):
    import pychrono as chrono
    from nedm.quadruped.robot import Go2Robot
    from nedm.quadruped.terrain import build_rigid_ground
    from quadruped.lib.policy import Go2Policy

    # ENVELOPE AND MARGIN BEFORE ANY COLLISION MODEL IS BUILT. These are global
    # defaults consulted at model construction, so setting them after the ground or the
    # robot exists leaves those models with whatever was in force earlier. Without them
    # the feet pass straight through the ground: measured, feet at z = -0.22 against a
    # ground surface at +0.05, with the robot resting on its trunk at 0.093.
    chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0025)
    chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0025)

    system = chrono.ChSystemSMC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    system.GetSolver().AsIterative().SetMaxIterations(100)
    build_rigid_ground(chrono, system)

    # SPAWN CLEARS THE FULLY EXTENDED LEG, not the standing height. The URDF parser
    # starts every joint at zero, which is legs straight down, 0.42 m of reach. Spawning
    # at standing height puts the feet at z = -0.046 against a ground surface at +0.05 --
    # penetrating before the first step, and the robot never recovers. It rests on its
    # trunk at 0.093 with its legs through the floor, which reads exactly like a policy
    # that cannot stand.
    GROUND_TOP = 0.05
    LEG_REACH = 0.42
    init = chrono.ChFramed(chrono.ChVector3d(0, 0, GROUND_TOP + LEG_REACH + 0.02),
                           chrono.ChQuaterniond(1, 0, 0, 0))
    # TORQUE, not position. apply_pd() is a no-op on the position plant, so the policy's
    # gains would never be applied and the joints would be driven by a hard constraint
    # instead -- a different controller than the one the policy was trained against.
    # PD_KP/PD_KD are 20.0/0.5, which is exactly the rl_sar rl_kp/rl_kd.
    # CHDIR IS LOAD-BEARING. The URDF references its meshes by relative path, so without
    # this the collision geometry silently fails to load and the robot has no feet -- it
    # lies flat, perfectly level, at belly height, which is exactly what it did.
    import os  # noqa: PLC0415
    cwd = os.getcwd()
    os.chdir(urdf.parent)
    try:
        robot = Go2Robot(system, urdf, init, actuation="torque")
    finally:
        os.chdir(cwd)

    cfg_override = {"sign": {"value": sign, "established_by": "probe"}}
    import yaml
    cfg = yaml.safe_load((REPO / "quadruped" / "params" / "policy.yaml").read_text())
    cfg.update(cfg_override)
    pol = Go2Policy(policy_path, cfg=cfg)
    pol.command = np.asarray(command, dtype=np.float32)

    dt = 1.0 / 400.0
    every = max(1, int(round((1.0 / control_hz) / dt)))
    n = int(seconds / dt)

    # WARMUP. The URDF spawns with its joints near zero, i.e. legs straight, so handing
    # straight to the policy measures the drop rather than the policy. Hold the simulator
    # stand pose first and let it settle, exactly as the collector does, then hand over.
    from nedm.quadruped.constants import STAND_ACTION  # noqa: PLC0415
    for _ in range(int(warmup_s / dt)):
        robot.actuate(STAND_ACTION)
        robot.apply_pd()                 # every physics step, not every control step
        system.DoStepDynamics(dt)

    zs, ups, vxs = [], [], []
    for i in range(n):
        if i % every == 0:
            robot.actuate(pol.act(robot))
        robot.apply_pd()                 # every physics step
        system.DoStepDynamics(dt)
        b = robot.base()
        p = b.GetPos()
        if not math.isfinite(p.z):
            return {"sign": sign, "fell_at_s": i * dt, "min_z": float("nan"),
                    "mean_up": float("nan"), "mean_vx": float("nan"), "diverged": True}
        zs.append(p.z)
        r = b.GetRot()
        from quadruped.params import transforms as T  # noqa: PLC0415
        ups.append(-T.projected_gravity(r.e0, r.e1, r.e2, r.e3)[2])
        v = b.GetPosDt()
        vxs.append(v.x)
    zs, ups, vxs = np.array(zs), np.array(ups), np.array(vxs)
    settle = int(0.5 / dt)
    return {
        "sign": sign,
        "min_z": float(zs.min()),
        "final_z": float(zs[-1]),
        "mean_up": float(ups[settle:].mean()),   # 1.0 = perfectly upright
        "mean_vx": float(vxs[settle:].mean()),
        "diverged": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--urdf", default=None)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--command", type=float, nargs=3, default=[0.5, 0.0, 0.0])
    ap.add_argument("--warmup", type=float, default=1.0,
                    help="seconds holding the simulator stand pose before the policy acts")
    ap.add_argument("--write", action="store_true",
                    help="write the winning sign into params/policy.yaml")
    a = ap.parse_args()

    urdf = Path(a.urdf) if a.urdf else None
    if urdf is None:
        for c in (REPO / "data/robot/go2_irrvis/urdf/go2_description.urdf",
                  Path.home() / "sbel-artifacts/data/robot/go2_irrvis/urdf/go2_description.urdf"):
            if c.exists():
                urdf = c
                break
    if urdf is None or not urdf.exists():
        print("FATAL: go2_description.urdf not found; pass --urdf", file=sys.stderr)
        return 1

    print(f"urdf   {urdf}")
    print(f"policy {a.policy}")
    print(f"command {a.command}, {a.seconds}s on rigid ground\n")

    res = {}
    for sign in (+1.0, -1.0):
        try:
            r = run(sign, a.seconds, a.command, urdf, Path(a.policy), 50.0,
                    warmup_s=a.warmup)
        except Exception as e:  # noqa: BLE001
            r = {"sign": sign, "error": f"{type(e).__name__}: {e}"}
        res[sign] = r
        if "error" in r:
            print(f"sign {sign:+.0f}:  ERROR {r['error']}")
        else:
            print(f"sign {sign:+.0f}:  min_z {r['min_z']:.3f}  final_z {r['final_z']:.3f}  "
                  f"upright {r['mean_up']:+.3f}  vx {r['mean_vx']:+.3f}")

    ok = {s: r for s, r in res.items() if "error" not in r}
    if len(ok) < 2:
        print("\nINCONCLUSIVE: a candidate failed to run at all", file=sys.stderr)
        return 1

    # A standing robot keeps its height and stays upright. A negated one collapses.
    def score(r):
        return (r["mean_up"], r["min_z"])

    win = max(ok, key=lambda s: score(ok[s]))
    lose = -win
    print(f"\nwinner: sign {win:+.0f}")
    print(f"  upright {ok[win]['mean_up']:+.3f} vs {ok[lose]['mean_up']:+.3f}")
    print(f"  min_z   {ok[win]['min_z']:.3f} vs {ok[lose]['min_z']:.3f}")

    if ok[win]["mean_up"] < 0.8 or ok[win]["min_z"] < 0.15:
        print("\nINCONCLUSIVE: neither candidate stands. The defect is not the sign.",
              file=sys.stderr)
        return 1
    if ok[lose]["mean_up"] > 0.8 and ok[lose]["min_z"] > 0.15:
        print("\nINCONCLUSIVE: BOTH candidates stand, so this test does not separate them.",
              file=sys.stderr)
        return 1

    if a.write:
        p = REPO / "quadruped" / "params" / "policy.yaml"
        s = p.read_text()
        ev = (f"rigid-ground probe {a.seconds}s cmd={a.command}: "
              f"upright {ok[win]['mean_up']:+.3f} vs {ok[lose]['mean_up']:+.3f}, "
              f"min_z {ok[win]['min_z']:.3f} vs {ok[lose]['min_z']:.3f}")
        s = s.replace("  value: null", f"  value: {win:+.1f}")
        s = s.replace("  established_by: null", f'  established_by: "{ev}"')
        p.write_text(s)
        print(f"\nwrote sign {win:+.0f} into {p}")
    else:
        print("\n(not written; pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
