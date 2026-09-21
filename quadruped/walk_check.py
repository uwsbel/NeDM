#!/usr/bin/env python3
"""Does the policy walk? On rigid ground, and on CRM soil.

Rigid is the control: the policy was trained on a ChBodyEasyBox, so if it does not walk
here nothing about CRM is the problem. CRM is the terrain the study is actually about.

Reports achieved forward velocity against commanded, base height, and uprightness, so the
answer is a number rather than an impression.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))


def crm_args(spacing, step, soil, patch_x, patch_y, depth):
    """The collector's defaults, which build_crm reads duck-typed off a namespace."""
    return types.SimpleNamespace(
        spacing=spacing, step=step, soil=soil, patch_x=patch_x, patch_y=patch_y,
        depth=depth, soil_bottom=0.0, artificial_viscosity=2.0,
        no_calf_fsi=False, check_embedded=False,
        soil_young=None, soil_cohesion=None,
    )


def run(terrain_kind, sign, seconds, command, urdf, policy_path, *, warmup_s=1.5,
        spacing=0.02, step=5e-4, soil="soft", patch_x=8.0, patch_y=4.0, depth=0.20,
        exchange_mult=4):
    import pychrono as chrono
    from nedm.quadruped.robot import Go2Robot
    from nedm.quadruped.terrain import build_crm, build_rigid_ground, measure_leg_reach
    from nedm.quadruped.constants import STAND_ACTION
    from quadruped.lib.policy import Go2Policy
    from quadruped.params import transforms as T
    import yaml

    chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0025)
    chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0025)

    system = chrono.ChSystemSMC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    system.GetSolver().AsIterative().SetMaxIterations(100)

    rigid = terrain_kind == "rigid"
    cwd = os.getcwd()

    if rigid:
        build_rigid_ground(chrono, system)
        soil_top = 0.05
        os.chdir(urdf.parent)
        try:
            leg_reach = measure_leg_reach(chrono, urdf)
        finally:
            os.chdir(cwd)
        spawn_z = soil_top + leg_reach + 0.02
        dt = 1.0 / 400.0
    else:
        import pychrono.fsi as fsi
        import pychrono.vehicle as veh
        soil_top = 0.0 + depth
        os.chdir(urdf.parent)
        try:
            leg_reach = measure_leg_reach(chrono, urdf)
        finally:
            os.chdir(cwd)
        # The collector's rule exactly: clear the FULLY EXTENDED leg above the bed.
        spawn_z = soil_top + 2.0 * spacing + leg_reach
        dt = exchange_mult * step

    init = chrono.ChFramed(chrono.ChVector3d(0, 0, spawn_z), chrono.ChQuaterniond(1, 0, 0, 0))
    os.chdir(urdf.parent)
    try:
        robot = Go2Robot(system, urdf, init, actuation="torque")
    finally:
        os.chdir(cwd)

    terrain = None
    if not rigid:
        a = crm_args(spacing, step, soil, patch_x, patch_y, depth)
        os.chdir(urdf.parent)
        try:
            terrain, _coupled = build_crm(chrono, fsi, veh, system, robot, a)
        finally:
            os.chdir(cwd)

    cfg = yaml.safe_load((REPO / "quadruped" / "params" / "policy.yaml").read_text())
    if sign is not None:
        cfg["sign"] = {"value": sign, "established_by": "override"}
    pol = Go2Policy(policy_path, cfg=cfg)
    pol.command = np.asarray(command, dtype=np.float32)

    def advance():
        if terrain is not None:
            terrain.DoStepDynamics(dt)
        else:
            system.DoStepDynamics(dt)

    for _ in range(int(warmup_s / dt)):
        robot.actuate(STAND_ACTION)
        robot.apply_pd()
        advance()

    every = max(1, int(round(0.02 / dt)))
    n = int(seconds / dt)
    zs, ups, vxs = [], [], []
    for i in range(n):
        if i % every == 0:
            robot.actuate(pol.act(robot))
        robot.apply_pd()
        advance()
        b = robot.base()
        p, v, r = b.GetPos(), b.GetPosDt(), b.GetRot()
        if not math.isfinite(p.z):
            return {"terrain": terrain_kind, "diverged_at_s": i * dt}
        zs.append(p.z); vxs.append(v.x)
        ups.append(-T.projected_gravity(r.e0, r.e1, r.e2, r.e3)[2])
    s = int(0.5 / dt)
    return {"terrain": terrain_kind, "sign": pol.sign, "spawn_z": spawn_z,
            "soil_top": soil_top, "min_z": float(np.min(zs)),
            "mean_z": float(np.mean(zs[s:])), "mean_up": float(np.mean(ups[s:])),
            "cmd_vx": float(command[0]), "mean_vx": float(np.mean(vxs[s:])),
            "diverged_at_s": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--terrain", choices=["rigid", "crm", "both"], default="both")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--warmup", type=float, default=1.5)
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--sign", type=float, default=None)
    ap.add_argument("--spacing", type=float, default=0.02)
    ap.add_argument("--patch-x", type=float, default=8.0)
    ap.add_argument("--patch-y", type=float, default=4.0)
    a = ap.parse_args()

    kinds = ["rigid", "crm"] if a.terrain == "both" else [a.terrain]
    rc = 0
    for k in kinds:
        try:
            r = run(k, a.sign, a.seconds, [a.vx, 0.0, 0.0], Path(a.urdf),
                    Path(a.policy), warmup_s=a.warmup, spacing=a.spacing,
                    patch_x=a.patch_x, patch_y=a.patch_y)
        except Exception as e:  # noqa: BLE001
            print(f"{k:6s}  ERROR {type(e).__name__}: {str(e)[:110]}")
            rc = 1
            continue
        if r.get("diverged_at_s") is not None:
            print(f"{k:6s}  DIVERGED at {r['diverged_at_s']:.2f}s")
            rc = 1
            continue
        track = r["mean_vx"] / r["cmd_vx"] if r["cmd_vx"] else float("nan")
        print(f"{k:6s}  spawn {r['spawn_z']:.3f}  soil_top {r['soil_top']:.2f}  "
              f"mean_z {r['mean_z']:.3f}  upright {r['mean_up']:+.3f}  "
              f"vx {r['mean_vx']:+.3f} / {r['cmd_vx']:.2f}  tracking {track:.0%}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
