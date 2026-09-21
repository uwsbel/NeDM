#!/usr/bin/env python3
"""Do two machines disagree about CRM from the first step, or only after chaos amplifies?

The same Go2 on the same soil tracks 70.2 +/- 3.4% on a3 and 79.2 +/- 2.4% on north, over
the same eight spawn offsets, while rigid ground agrees to 0.1 points on both
(`docs/EVALUATION.md`). Something in the CRM path is machine-dependent. Two explanations
have opposite consequences:

  ARITHMETIC. The solvers compute different numbers -- different GPU architecture,
  different reduction order, separately compiled binaries. Then the machines are running
  subtly different physics and no amount of replication reconciles them. A corpus
  collected on one machine cannot be evaluated on another.

  AMPLIFICATION. The solvers agree to floating-point noise and the chaotic gait amplifies
  that noise into a visible difference. Then the machines are running the SAME physics and
  the disagreement is a sampling problem, fixable by replicating within a machine and
  never pooling across.

These are distinguishable by looking early. This runs a fixed, low-chaos protocol -- the
robot STANDS, it does not walk, because a standing robot does not amplify perturbations
the way a gait does -- and records the full state at a geometric ladder of step counts.
Comparing the ladder across machines shows whether the disagreement is present at step 1
or grows from nothing.

Particle state is summarised by mean and a checksum rather than dumped, so the artefact is
small enough to move between machines and still catches a single differing bit.

Run the same command on two machines, then diff the JSONs with --compare.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

PATCH_X, PATCH_Y, DEPTH = 8.0, 4.0, 0.20
SPACING, STEP = 0.02, 5e-4
LADDER = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)


def _hash(a):
    """Checksum of the exact bytes, so a single differing bit is visible."""
    return hashlib.md5(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]


def snapshot(terrain, robot, step):
    b = robot.base()
    p, v, r = b.GetPos(), b.GetPosDt(), b.GetRot()
    pos = np.asarray(terrain.GetFluidSystemSPH().GetParticlePositionsNumpy(), dtype=np.float64)
    vel = np.asarray(terrain.GetFluidSystemSPH().GetParticleVelocitiesNumpy(), dtype=np.float64)
    return {
        "step": step,
        "base_pos": [p.x, p.y, p.z],
        "base_vel": [v.x, v.y, v.z],
        "base_rot": [r.e0, r.e1, r.e2, r.e3],
        "sph_mean_z": float(pos[:, 2].mean()),
        "sph_mean_speed": float(np.linalg.norm(vel, axis=1).mean()),
        "sph_max_speed": float(np.linalg.norm(vel, axis=1).max()),
        "sph_pos_hash": _hash(pos),
        "sph_vel_hash": _hash(vel),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                    help="compare two probe outputs instead of running")
    a, unknown = ap.parse_known_args()

    if a.compare:
        return compare(Path(a.compare[0]), Path(a.compare[1]))

    import pychrono as chrono
    import pychrono.fsi as fsi
    import pychrono.vehicle as veh

    from nedm.quadruped.robot import Go2Robot
    from nedm.quadruped.terrain import build_crm, measure_leg_reach
    from nedm.quadruped.constants import STAND_ACTION

    urdf = Path(a.urdf)
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
    init = chrono.ChFramed(chrono.ChVector3d(0.0, 0.0, DEPTH + 2 * SPACING + leg_reach),
                           chrono.ChQuaterniond(1, 0, 0, 0))
    os.chdir(urdf.parent)
    try:
        robot = Go2Robot(system, urdf, init, actuation="torque")
    finally:
        os.chdir(cwd)

    args = types.SimpleNamespace(
        spacing=SPACING, step=STEP, soil="soft", patch_x=PATCH_X, patch_y=PATCH_Y,
        depth=DEPTH, soil_bottom=0.0, artificial_viscosity=2.0, no_calf_fsi=False,
        check_embedded=False, soil_young=None, soil_cohesion=None)
    os.chdir(urdf.parent)
    try:
        terrain, _ = build_crm(chrono, fsi, veh, system, robot, args)
    finally:
        os.chdir(cwd)

    dt = 4 * STEP
    snaps = [snapshot(terrain, robot, 0)]
    for i in range(1, max(LADDER) + 1):
        robot.actuate(STAND_ACTION)      # STANDING, not walking: minimal chaos
        robot.apply_pd()
        terrain.DoStepDynamics(dt)
        if i in LADDER:
            snaps.append(snapshot(terrain, robot, i))
            print(f"  step {i:5d}  base_z {snaps[-1]['base_pos'][2]:.9f}  "
                  f"sph_mean_z {snaps[-1]['sph_mean_z']:.9f}  "
                  f"pos#{snaps[-1]['sph_pos_hash']}", flush=True)

    import subprocess
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        gpu = "unknown"
    out = {"host": socket.gethostname(), "platform": platform.platform(), "gpu": gpu,
           "n_particles": int(terrain.GetNumSPHParticles()), "snapshots": snaps}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")
    return 0


def compare(pa: Path, pb: Path) -> int:
    A, B = json.loads(pa.read_text()), json.loads(pb.read_text())
    print(f"A: {A['host']}  {A['gpu']}  {A['n_particles']:,} particles")
    print(f"B: {B['host']}  {B['gpu']}  {B['n_particles']:,} particles")
    if A["n_particles"] != B["n_particles"]:
        print("\nPARTICLE COUNTS DIFFER -- the scenes are not the same, stop here.")
        return 1
    print(f"\n{'step':>6} {'pos hash':>10} {'|d base_z|':>14} {'|d sph_mean_z|':>16} "
          f"{'|d max_speed|':>14}")
    first_div = None
    for sa, sb in zip(A["snapshots"], B["snapshots"], strict=False):
        same = sa["sph_pos_hash"] == sb["sph_pos_hash"]
        dz = abs(sa["base_pos"][2] - sb["base_pos"][2])
        dm = abs(sa["sph_mean_z"] - sb["sph_mean_z"])
        ds = abs(sa["sph_max_speed"] - sb["sph_max_speed"])
        if not same and first_div is None:
            first_div = sa["step"]
        print(f"{sa['step']:>6} {'same' if same else 'DIFFER':>10} "
              f"{dz:>14.3e} {dm:>16.3e} {ds:>14.3e}")
    print()
    if first_div is None:
        print("Particle state is bit-identical at every checkpoint. The machines agree; "
              "any tracking difference comes from somewhere other than the SPH solve.")
    elif first_div <= 2:
        print(f"Particle state differs from step {first_div}. This is ARITHMETIC, not "
              "amplification: the solvers compute different numbers before any chaos "
              "could act. The machines are running subtly different physics, and a corpus "
              "collected on one cannot be evaluated on another.")
    else:
        print(f"Particle state agrees until step {first_div}, then diverges. Consistent "
              "with AMPLIFICATION of floating-point noise rather than a different "
              "computation. Replicate within a machine and never pool across machines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
