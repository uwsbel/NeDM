#!/usr/bin/env python3
"""Is the robot walking onto settled soil, or onto soil that starts settling when it
arrives?

With an active domain on, a particle outside the box has its velocity zeroed every step
(`SphFluidDynamics.cu:245-246`). `SetFreeFlowDuration(t)` is the one escape: while
`time < t` the active domain is disabled entirely and every particle is simulated
(`SphFluidDynamics.cu:243`), which is how the bed is meant to settle under gravity before
the run proper begins.

Ours is 0.1 s, inherited. A 0.2 m bed laid out on a perfect lattice at rest density does
not settle in 0.1 s. If it does not, then soil ahead of the robot stays frozen at its
initial state and begins settling only when the robot walks into it and activates it --
so every episode is recorded on settling rather than settled material, and the bias is
systematic across the entire corpus rather than averaging out.

THE CONFOUND THIS CONTROLS FOR. A longer free-flow phase is also a longer phase during
which the robot is standing there doing nothing, and standing changes the soil under the
feet whether or not the rest of the bed settles. So total warmup is held FIXED across
every arm and only the free-flow fraction of it varies. The robot's standing time is then
identical and the only difference is how much of the bed was allowed to move.

The direct diagnostic is the bed top. If the bed compacts, the SPH bounding box's upper z
face drops, and that is measurable without reference to the robot at all.
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

PATCH_X, PATCH_Y, DEPTH = 8.0, 4.0, 0.20
SPACING, STEP = 0.02, 5e-4
ACTIVE = 1.0          # held at the inherited value; this test is about settling, not size


def run(command, free_flow_s, seconds, warmup_s, urdf, policy_path, spawn_xy,
        exchange_mult=4):
    import pychrono as chrono
    import pychrono.fsi as fsi
    import pychrono.vehicle as veh
    import yaml

    from nedm import chrono_crm_compat as crm_compat
    from nedm.quadruped.robot import Go2Robot
    from nedm.quadruped.terrain import (assert_spawn_on_patch, build_crm,
                                        measure_leg_reach)
    from nedm.quadruped.constants import STAND_ACTION
    from quadruped.lib.policy import Go2Policy
    from quadruped.params import transforms as T

    orig_ff = crm_compat.set_free_flow_duration
    crm_compat.set_free_flow_duration = lambda terrain, _d: orig_ff(terrain, free_flow_s)
    try:
        chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0025)
        chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0025)

        system = chrono.ChSystemSMC()
        system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
        system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        system.GetSolver().AsIterative().SetMaxIterations(100)

        cwd = os.getcwd()
        os.chdir(urdf.parent)
        try:
            leg_reach = measure_leg_reach(chrono, urdf)
        finally:
            os.chdir(cwd)
        spawn_z = DEPTH + 2.0 * SPACING + leg_reach
        dt = exchange_mult * STEP

        init = chrono.ChFramed(chrono.ChVector3d(spawn_xy[0], spawn_xy[1], spawn_z),
                               chrono.ChQuaterniond(1, 0, 0, 0))
        os.chdir(urdf.parent)
        try:
            robot = Go2Robot(system, urdf, init, actuation="torque")
        finally:
            os.chdir(cwd)

        a = types.SimpleNamespace(
            spacing=SPACING, step=STEP, soil="soft", patch_x=PATCH_X, patch_y=PATCH_Y,
            depth=DEPTH, soil_bottom=0.0, artificial_viscosity=2.0, no_calf_fsi=False,
            check_embedded=False, soil_young=None, soil_cohesion=None)
        os.chdir(urdf.parent)
        try:
            terrain, _ = build_crm(chrono, fsi, veh, system, robot, a)
        finally:
            os.chdir(cwd)
        assert_spawn_on_patch(terrain, spawn_xy, margin=0.5)

        bed_top_0 = float(terrain.GetSPHBoundingBox().max.z)

        cfg = yaml.safe_load((REPO / "quadruped" / "params" / "policy.yaml").read_text())
        pol = Go2Policy(policy_path, cfg=cfg)
        pol.command = np.asarray(command, dtype=np.float32)

        # FIXED total warmup across arms. Only the free-flow fraction inside it varies.
        t0 = time.perf_counter()
        for _ in range(int(warmup_s / dt)):
            robot.actuate(STAND_ACTION)
            robot.apply_pd()
            terrain.DoStepDynamics(dt)
        warm_wall = time.perf_counter() - t0
        bed_top_1 = float(terrain.GetSPHBoundingBox().max.z)

        every = max(1, int(round(0.02 / dt)))
        rec_every = max(1, int(round(0.01 / dt)))
        n = int(seconds / dt)
        traj = []
        for i in range(n):
            if i % every == 0:
                robot.actuate(pol.act(robot))
            robot.apply_pd()
            terrain.DoStepDynamics(dt)
            if i % rec_every:
                continue
            b = robot.base()
            p, v, r = b.GetPos(), b.GetPosDt(), b.GetRot()
            if not math.isfinite(p.z):
                return {"diverged_at_s": i * dt}
            R = T.quat_to_rot(r.e0, r.e1, r.e2, r.e3)
            vb = R.T @ np.array([v.x, v.y, v.z])
            traj.append([p.x, p.y, p.z, vb[0], vb[1], float(b.GetAngVelLocal().z),
                         -T.projected_gravity(r.e0, r.e1, r.e2, r.e3)[2]])
        tr = np.asarray(traj, dtype=float)
        return {
            "free_flow_s": free_flow_s,
            "bed_top_initial_m": bed_top_0,
            "bed_top_after_warmup_m": bed_top_1,
            "bed_settle_m": bed_top_1 - bed_top_0,
            "warmup_wall_s": warm_wall,
            "mean_vx": float(np.mean(tr[:, 3])),
            "mean_vy": float(np.mean(tr[:, 4])),
            "mean_wz": float(np.mean(tr[:, 5])),
            "mean_z": float(np.mean(tr[:, 2])),
            "mean_up": float(np.mean(tr[:, 6])),
            "travelled_m": float(np.hypot(tr[-1, 0] - tr[0, 0], tr[-1, 1] - tr[0, 1])),
            "diverged_at_s": None,
        }
    finally:
        crm_compat.set_free_flow_duration = orig_ff


CASES = [
    ("fwd", (0.6, 0.0, 0.0), (-3.0, 0.0)),
    ("fwd_fast", (1.2, 0.0, 0.0), (-3.0, 0.0)),
    ("turn", (0.5, 0.0, 0.7), (-2.0, 0.0)),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--free-flow", default="0.1,0.5,1.0,2.0")
    ap.add_argument("--warmup", type=float, default=2.5,
                    help="held FIXED across arms; free flow is a fraction of it")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ffs = [float(x) for x in a.free_flow.split(",")]
    if max(ffs) > a.warmup:
        raise SystemExit(
            f"free flow {max(ffs)} s exceeds the fixed warmup {a.warmup} s. The arms would "
            f"then differ in how long the robot stood as well as in how long the bed was "
            f"free, which is the confound this test exists to remove.")
    urdf, policy = Path(a.urdf), Path(a.policy)
    out = {}

    print(f"warmup held at {a.warmup} s across all arms; free flow varies within it\n",
          flush=True)
    for name, cmd, spawn in CASES:
        print(f"=== {name}  cmd={cmd} ===", flush=True)
        out[name] = {}
        for ff in ffs:
            r = run(cmd, ff, a.seconds, a.warmup, urdf, policy, spawn)
            if r.get("diverged_at_s") is not None:
                print(f"  free_flow {ff:4.1f}s  DIVERGED at {r['diverged_at_s']:.2f}s",
                      flush=True)
                continue
            out[name][str(ff)] = r
            print("  free_flow %4.1fs | bed settled %+.5f m | vx %+.3f  z %.4f  "
                  "up %.4f  trav %.3f  (warmup wall %5.1fs)" %
                  (ff, r["bed_settle_m"], r["mean_vx"], r["mean_z"], r["mean_up"],
                   r["travelled_m"], r["warmup_wall_s"]), flush=True)
        print(flush=True)

    # The comparison that matters: does anything move as free flow grows? A bed that
    # settles but leaves the robot's behaviour unchanged is a non-issue; a bed that
    # settles AND shifts the gait means 0.1 s has been biasing every episode.
    print("=" * 74)
    print("CHANGE FROM THE INHERITED 0.1 s, per case")
    print("=" * 74)
    base = str(ffs[0])
    for name in out:
        if base not in out[name]:
            continue
        b = out[name][base]
        for ff in ffs[1:]:
            k = str(ff)
            if k not in out[name]:
                continue
            r = out[name][k]
            print("%-9s %4.1fs -> d_vx %+.4f  d_z %+.5f  d_bed_settle %+.5f" %
                  (name, ff, r["mean_vx"] - b["mean_vx"], r["mean_z"] - b["mean_z"],
                   r["bed_settle_m"] - b["bed_settle_m"]))

    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
