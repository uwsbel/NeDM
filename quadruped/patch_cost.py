#!/usr/bin/env python3
"""What does a bigger CRM patch actually cost?

The collector sizes its soil patch from how far the robot travels, because the patch is
centred and the robot must not walk off it. That rule sets patch length from the episode,
and if per-step cost scales with patch length then episode length is priced in particles.

But `build_crm` already calls `SetActiveDomain(1,1,1)`, inherited from Chrono's Viper CRM
demo. If the active domain does what its name suggests, the per-step cost is set by the
box around the robot and NOT by the patch, and a long patch is then nearly free per step.
Nobody has measured which of those is true here, so the patch-size decision has been made
on an assumption.

This separates the two axes: patch length and active-domain size are varied
independently, and build time and steady-state step time are reported apart, since a long
patch can be expensive to CREATE and still cheap to STEP.

Usage:
  python patch_cost.py --policy <policy.pt> --urdf <go2.urdf> \
      --config 4x4:1.0 --config 8x4:1.0 --config 16x4:1.0 --config 8x4:2.0 --config 8x4:none
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))


def particle_count(terrain):
    """Whatever this build exposes. Reported as a dict so a missing getter is visible
    rather than silently zero."""
    out = {}
    for name in ("GetNumSPHParticles", "GetNumBCE", "GetNumBoundaryBCEMarkers",
                 "GetNumFluidMarkers", "GetNumParticles"):
        fn = getattr(terrain, name, None)
        if fn is None:
            continue
        try:
            out[name] = int(fn())
        except Exception as e:  # noqa: BLE001
            out[name] = f"ERR {type(e).__name__}"
    return out


def run_one(patch_x, patch_y, active, seconds, warmup_s, command, urdf, policy_path,
            spacing, step, soil, depth, exchange_mult):
    import pychrono as chrono
    import pychrono.fsi as fsi
    import pychrono.vehicle as veh
    import yaml

    from nedm.quadruped.robot import Go2Robot
    from nedm.quadruped.terrain import build_crm, measure_leg_reach
    from nedm.quadruped.constants import STAND_ACTION
    from quadruped.lib.policy import Go2Policy
    from quadruped.params import transforms as T

    # The active-domain size is wired inside build_crm. Rather than fork the builder for a
    # measurement, intercept the one call: `active=None` skips it entirely, which is the
    # no-active-domain control. Restored in the finally block so configs do not leak into
    # one another.
    orig = veh.CRMTerrain.SetActiveDomain

    def patched(self, _v):
        if active is None:
            return None
        return orig(self, chrono.ChVector3d(active, active, active))

    veh.CRMTerrain.SetActiveDomain = patched
    try:
        chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0025)
        chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0025)

        system = chrono.ChSystemSMC()
        system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
        system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        system.GetSolver().AsIterative().SetMaxIterations(100)

        cwd = os.getcwd()
        soil_top = depth
        os.chdir(urdf.parent)
        try:
            leg_reach = measure_leg_reach(chrono, urdf)
        finally:
            os.chdir(cwd)
        spawn_z = soil_top + 2.0 * spacing + leg_reach
        dt = exchange_mult * step

        # Spawn at the near end so the robot walks ACROSS the patch. Centred spawning on a
        # 16 m patch would keep it in the middle for the whole measured window, and a
        # moving active domain that never nears an edge is not the case being priced.
        sx = -(patch_x / 2 - 1.0)
        init = chrono.ChFramed(chrono.ChVector3d(sx, 0.0, spawn_z),
                               chrono.ChQuaterniond(1, 0, 0, 0))
        os.chdir(urdf.parent)
        try:
            robot = Go2Robot(system, urdf, init, actuation="torque")
        finally:
            os.chdir(cwd)

        a = types.SimpleNamespace(
            spacing=spacing, step=step, soil=soil, patch_x=patch_x, patch_y=patch_y,
            depth=depth, soil_bottom=0.0, artificial_viscosity=2.0,
            no_calf_fsi=False, check_embedded=False, soil_young=None, soil_cohesion=None)

        t0 = time.perf_counter()
        os.chdir(urdf.parent)
        try:
            terrain, _ = build_crm(chrono, fsi, veh, system, robot, a)
        finally:
            os.chdir(cwd)
        build_s = time.perf_counter() - t0

        cfg = yaml.safe_load((REPO / "quadruped" / "params" / "policy.yaml").read_text())
        pol = Go2Policy(policy_path, cfg=cfg)
        pol.command = np.asarray(command, dtype=np.float32)

        # Warm-up is excluded from the timing: the first steps pay one-off neighbour-list
        # construction and allocation, and including them would price startup as if it
        # recurred every step.
        for _ in range(int(warmup_s / dt)):
            robot.actuate(STAND_ACTION)
            robot.apply_pd()
            terrain.DoStepDynamics(dt)

        every = max(1, int(round(0.02 / dt)))
        n = int(seconds / dt)
        t0 = time.perf_counter()
        zs = []
        for i in range(n):
            if i % every == 0:
                robot.actuate(pol.act(robot))
            robot.apply_pd()
            terrain.DoStepDynamics(dt)
            p = robot.base().GetPos()
            if not math.isfinite(p.z):
                return {"patch": f"{patch_x:g}x{patch_y:g}", "active": active,
                        "diverged_at_s": i * dt}
            zs.append(p.z)
        step_s = time.perf_counter() - t0

        b = robot.base()
        r = b.GetRot()
        up = -T.projected_gravity(r.e0, r.e1, r.e2, r.e3)[2]
        R = T.quat_to_rot(r.e0, r.e1, r.e2, r.e3)
        v = b.GetPosDt()
        vb = R.T @ np.array([v.x, v.y, v.z])

        return {
            "patch": f"{patch_x:g}x{patch_y:g}",
            "patch_x": patch_x, "patch_y": patch_y,
            "active": active,
            "counts": particle_count(terrain),
            "build_s": round(build_s, 2),
            "sim_s": seconds,
            "wall_s": round(step_s, 2),
            "steps": n,
            "ms_per_step": round(1000.0 * step_s / n, 3),
            "slowdown_x": round(step_s / seconds, 2),
            "end_x": round(float(b.GetPos().x), 3),
            "travelled_m": round(float(b.GetPos().x) - sx, 3),
            "mean_z": round(float(np.mean(zs)), 4),
            "up": round(float(up), 4),
            "vx_body": round(float(vb[0]), 3),
            "diverged_at_s": None,
        }
    finally:
        veh.CRMTerrain.SetActiveDomain = orig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--config", action="append", required=True,
                    help="PATCHX x PATCHY : ACTIVE, e.g. 8x4:1.0 or 8x4:none")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--vx", type=float, default=0.6)
    ap.add_argument("--spacing", type=float, default=0.02)
    ap.add_argument("--step", type=float, default=5e-4)
    ap.add_argument("--depth", type=float, default=0.20)
    ap.add_argument("--soil", default="soft")
    ap.add_argument("--exchange-mult", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    for spec in a.config:
        geom, _, act = spec.partition(":")
        px, _, py = geom.partition("x")
        active = None if act.strip().lower() in ("none", "", "off") else float(act)
        try:
            r = run_one(float(px), float(py), active, a.seconds, a.warmup,
                        [a.vx, 0.0, 0.0], Path(a.urdf), Path(a.policy),
                        a.spacing, a.step, a.soil, a.depth, a.exchange_mult)
        except Exception as e:  # noqa: BLE001
            r = {"patch": geom, "active": active,
                 "error": f"{type(e).__name__}: {str(e)[:200]}"}
        rows.append(r)
        print(json.dumps(r), flush=True)

    print("\n%-9s %-7s %10s %9s %9s %9s %7s %7s" %
          ("patch", "active", "particles", "build_s", "ms/step", "slowdown", "trav_m", "up"))
    for r in rows:
        if "error" in r:
            print("%-9s %-7s  ERROR %s" % (r["patch"], r["active"], r["error"]))
            continue
        if r.get("diverged_at_s") is not None:
            print("%-9s %-7s  DIVERGED at %.2fs" % (r["patch"], r["active"], r["diverged_at_s"]))
            continue
        c = r["counts"]
        n = c.get("GetNumSPHParticles") or c.get("GetNumFluidMarkers") or c.get("GetNumParticles") or -1
        print("%-9s %-7s %10s %9.1f %9.3f %9.2f %7.2f %7.3f" %
              (r["patch"], str(r["active"]), f"{n:,}" if n > 0 else "?",
               r["build_s"], r["ms_per_step"], r["slowdown_x"],
               r["travelled_m"], r["up"]))

    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
