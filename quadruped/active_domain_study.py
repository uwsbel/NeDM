#!/usr/bin/env python3
"""How small can the active domain get before it changes the physics?

The patch-cost benchmark established that patch LENGTH is nearly free: 4x the particles
(444k -> 1.77M) costs 4.5% more per step, because every SPH kernel launches over the
active set rather than over all markers. The active DOMAIN is the whole cost, and it is
steep -- 3.81x real time at a 0.5 m box, 6.18x at 1.0 m, 14.11x at 2.0 m.

So the collection budget is set by one number, and halving it nearly halves the cost of
the corpus. The catch is that the active domain is not a free approximation. Particles
outside it have their velocity zeroed every step, which destroys momentum at the face,
and the SPH sum is truncated at an envelope only 2*h_multiplier*h wide, which is a
free-surface-like artifact. A box chosen for speed can quietly stiffen the soil under the
robot, and that would bias the exact quantity the study measures.

This is the artificial-viscosity discipline again: take the SMALLEST value that does not
change the answer, and show that it does not, rather than taking the largest that runs.

THE NOISE FLOOR COMES FIRST. A paired difference is meaningless without knowing what two
identical runs differ by, and CRM runs on the GPU where reductions need not be
bit-reproducible. The reference configuration is therefore run twice on identical inputs,
and any active-domain effect has to clear that floor before it is called an effect.

The design is paired: every case (command, spawn) is run at every active-domain size, and
differences are taken within a case. Unpaired scatter across cases is roughly +/-20% on
mean velocity, which would swamp the effect being looked for.
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

# Fixed across every run so the only thing that varies is the active domain.
PATCH_X, PATCH_Y, DEPTH = 8.0, 4.0, 0.20
SPACING, STEP = 0.02, 5e-4


def run_case(command, active, seconds, warmup_s, urdf, policy_path, spawn_xy,
             free_flow_s=None, exchange_mult=4):
    """One (command, active-domain) run. Returns the trajectory and the timing."""
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

    orig_ad = veh.CRMTerrain.SetActiveDomain
    orig_ff = crm_compat.set_free_flow_duration

    def patched_ad(self, _v):
        if active is None:
            return None
        return orig_ad(self, chrono.ChVector3d(active, active, active))

    def patched_ff(terrain, _d):
        return orig_ff(terrain, 0.1 if free_flow_s is None else free_flow_s)

    veh.CRMTerrain.SetActiveDomain = patched_ad
    crm_compat.set_free_flow_duration = patched_ff
    try:
        chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0025)
        chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0025)

        system = chrono.ChSystemSMC()
        system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
        system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        system.GetSolver().AsIterative().SetMaxIterations(100)

        cwd = os.getcwd()
        soil_top = DEPTH
        os.chdir(urdf.parent)
        try:
            leg_reach = measure_leg_reach(chrono, urdf)
        finally:
            os.chdir(cwd)
        spawn_z = soil_top + 2.0 * SPACING + leg_reach
        dt = exchange_mult * STEP

        init = chrono.ChFramed(
            chrono.ChVector3d(spawn_xy[0], spawn_xy[1], spawn_z),
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

        cfg = yaml.safe_load((REPO / "quadruped" / "params" / "policy.yaml").read_text())
        pol = Go2Policy(policy_path, cfg=cfg)
        pol.command = np.asarray(command, dtype=np.float32)

        for _ in range(int(warmup_s / dt)):
            robot.actuate(STAND_ACTION)
            robot.apply_pd()
            terrain.DoStepDynamics(dt)

        every = max(1, int(round(0.02 / dt)))
        rec_every = max(1, int(round(0.01 / dt)))
        n = int(seconds / dt)
        traj = []
        t0 = time.perf_counter()
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
            w = b.GetAngVelLocal()
            traj.append([p.x, p.y, p.z, vb[0], vb[1], vb[2], w.z,
                         -T.projected_gravity(r.e0, r.e1, r.e2, r.e3)[2]])
        wall = time.perf_counter() - t0
        return {"traj": np.asarray(traj, dtype=float), "wall_s": wall,
                "ms_per_step": 1000.0 * wall / n, "slowdown_x": wall / seconds,
                "diverged_at_s": None}
    finally:
        veh.CRMTerrain.SetActiveDomain = orig_ad
        crm_compat.set_free_flow_duration = orig_ff


def compare(ref, other):
    """Paired difference between two trajectories of the same case.

    Reported against DISTANCE TRAVELLED, not against the noise floor. The floor turned out
    to be exactly zero -- two identical runs are bit-identical on this GPU -- so a ratio to
    it is either zero or infinite and says nothing. Normalising by travel is the study's
    own errdist convention and makes a 0.15 m disagreement legible as the 12% of a 1.3 m
    walk that it is.
    """
    n = min(len(ref), len(other))
    a, b = ref[:n], other[:n]
    d = b - a
    travel = float(np.hypot(a[-1, 0] - a[0, 0], a[-1, 1] - a[0, 1]))
    final = float(np.hypot(d[-1, 0], d[-1, 1]))
    mean_vx_ref = float(np.mean(a[:, 3]))
    return {
        "ref_travel_m": travel,
        "final_xy_err_m": final,
        "err_over_travel": final / travel if travel > 1e-6 else float("nan"),
        "rms_pos_m": float(np.sqrt(np.mean(np.sum(d[:, :3] ** 2, axis=1)))),
        "rms_vx_mps": float(np.sqrt(np.mean(d[:, 3] ** 2))),
        "d_mean_vx_mps": float(np.mean(b[:, 3]) - mean_vx_ref),
        "rel_d_mean_vx": (float(np.mean(b[:, 3]) - mean_vx_ref) / abs(mean_vx_ref)
                          if abs(mean_vx_ref) > 1e-6 else float("nan")),
        "d_mean_z_m": float(np.mean(b[:, 2]) - np.mean(a[:, 2])),
        "d_mean_up": float(np.mean(b[:, 7]) - np.mean(a[:, 7])),
    }


CASES = [
    ("fwd_slow", (0.4, 0.0, 0.0), (-3.0, 0.0)),
    ("fwd_fast", (1.0, 0.0, 0.0), (-3.0, 0.0)),
    ("lateral", (0.3, 0.5, 0.0), (-2.0, -1.5)),
    ("turning", (0.6, 0.0, 0.6), (-2.0, 0.0)),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--active", default="0.5,0.75,1.0,2.0")
    ap.add_argument("--ref", type=float, default=2.0,
                    help="the largest box, taken as the converged reference")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--warmup", type=float, default=1.5)
    ap.add_argument("--free-flow-s", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ads = [float(x) for x in a.active.split(",")]
    urdf, policy = Path(a.urdf), Path(a.policy)
    results = {}

    for name, cmd, spawn in CASES:
        # The noise floor, from two identical runs of the reference. Everything else is
        # judged against this, so it is measured first and per case.
        r0 = run_case(cmd, a.ref, a.seconds, a.warmup, urdf, policy, spawn, a.free_flow_s)
        r0b = run_case(cmd, a.ref, a.seconds, a.warmup, urdf, policy, spawn, a.free_flow_s)
        if r0["diverged_at_s"] is not None or r0b["diverged_at_s"] is not None:
            print(f"{name}: reference diverged, skipping case", flush=True)
            continue
        floor = compare(r0["traj"], r0b["traj"])
        results[name] = {"floor": floor, "ref_ms_per_step": r0["ms_per_step"], "arms": {}}
        print(f"\n=== {name}  cmd={cmd}  spawn={spawn} ===", flush=True)
        print("  noise floor (ref run twice): final_xy %.4f m  rms_pos %.4f m  "
              "rms_vx %.4f m/s  d_mean_vx %+.4f" %
              (floor["final_xy_err_m"], floor["rms_pos_m"], floor["rms_vx_mps"],
               floor["d_mean_vx_mps"]), flush=True)

        for ad in ads:
            if ad == a.ref:
                continue
            r = run_case(cmd, ad, a.seconds, a.warmup, urdf, policy, spawn, a.free_flow_s)
            if r["diverged_at_s"] is not None:
                print(f"  ad={ad:<5} DIVERGED at {r['diverged_at_s']:.2f}s", flush=True)
                continue
            c = compare(r0["traj"], r["traj"])
            # Ratio to the noise floor is the number that matters: below ~1 the active
            # domain is indistinguishable from running the reference again.
            ratio = (c["rms_pos_m"] / floor["rms_pos_m"]) if floor["rms_pos_m"] > 0 else float("inf")
            results[name]["arms"][str(ad)] = {**c, "ms_per_step": r["ms_per_step"],
                                              "slowdown_x": r["slowdown_x"],
                                              "rms_pos_over_floor": ratio}
            print("  ad=%-5s %6.2f ms/step %5.2fx RT | final_xy %.4f  rms_pos %.4f "
                  "(%.1fx floor)  d_mean_vx %+.4f  d_mean_z %+.5f" %
                  (ad, r["ms_per_step"], r["slowdown_x"], c["final_xy_err_m"],
                   c["rms_pos_m"], ratio, c["d_mean_vx_mps"], c["d_mean_z_m"]), flush=True)

    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
