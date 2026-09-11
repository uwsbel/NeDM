"""Collect one Go2 episode on rigid or CRM terrain in the HMMWV dataset schema.

Mirrors scripts/collection/collect_hmmwv_crm_smoke.py: same episode CSV plus
per-episode JSON plus dataset_index.json plus collector_config.resolved.json
layout, same split assignment, same "write the row at a fixed record cadence
inside the physics loop" shape. The per-step schema lives in
nedm.quadruped.dataset; see that module for where the mirror is not literal.

Terrain labels follow terrain_conditioning: rigid ground is "flat", the training
soil preset is "crm".

COMMANDS VARY, VIA STRUCTURED FAMILIES. With --imported-ckpt the policy has a
live command channel, so each episode is driven by one excitation family and the
full command series is recorded alongside the family name. Without it, the
harness policy reads a hardcoded [0.5, 0, 0] and no per-episode variation can
reach it -- in that case the command columns are constant and the episode is
state coverage only.

BOTH ACTION CANDIDATES ARE LOGGED: the twelve joint targets AND the three
velocity commands. Which one is the NRD action is a training-time config
decision; see nedm.quadruped.dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_OUTPUT_DIR = Path("artifacts/datasets/go2_smoke")
# Machine-specific, so it is an env var with this box's value as the fallback --
# the same convention drive_go2_collection.py already documents for NEDM_REPO,
# NEDM_PY and NEDM_CHRONO_PYTHONPATH. The driver passes no --assets, so a
# hardcoded path here is not a default: it is a hard failure on any box whose
# checkout is not at ~/Documents/sbel (dorm-pc's has no "sbel/" segment), and it
# fails at the URDF load AFTER the process has started.
DEFAULT_ASSETS = os.environ.get(
    "NEDM_GO2_ASSETS",
    "/home/kyle/Documents/sbel/sbel-reproducibility/2025/multi-terrain-RL")


# ---------------------------------------------------------------------------
# RENDERING. A copy of collect_go2_smoke.py with cameras bolted on, deliberately
# NOT a flag inside that script: it is launched fresh for every episode of every
# Chrono scoring run across four machines, and editing it mid-flight would take
# the evaluation down with it.
#
# Two things had to be established before this could work at all:
#
#   1. Chrono::Sensor's ray tracer needs OptiX 9.0/9.1, which needs an R590+
#      driver. north is on 580.126.09 and fails with UNSUPPORTED_ABI_VERSION;
#      sbel is on 595.84 and works. This runs on sbel.
#   2. ChOptixEngine::ConstructScene() walks m_system->GetBodies() and
#      GetOtherPhysicsItems() and turns VISUAL SHAPES into OptiX geometry. SPH
#      markers are neither, so a first test rendered a beautiful empty sky.
#
#      The obvious fix -- a ChParticleCloud carrying one ChVisualShapeSphere,
#      repositioned per frame -- DOES NOT WORK, and does not work for a reason
#      worth writing down. A ChParticleCloud lands in GetOtherPhysicsItems(),
#      and the loop there (ChOptixEngine.cpp, "Assumption made here that other
#      physics items don't have a transform -> not always true!!!") builds a
#      throwaway `dummy_body` per SHAPE INSTANCE and hangs the shape off that.
#      A cloud of 25,000 particles has exactly ONE shape instance, so the
#      renderer draws ONE sphere, at the origin, forever. Per-particle
#      instancing of a cloud's visual model is something the Irrlicht and VSG
#      visualisers do; the OptiX path has never done it. No amount of SetPos on
#      the cloud's particles can be seen, which is why the positions were
#      demonstrably updating and the frames were demonstrably empty.
#
#      What this build DOES have is a native FSI-SPH path, added 2026 by Bocheng
#      Zou: ChSensorManager::AttachFsiSphSystem(fluid_system, options) registers
#      the fluid system itself, and ChOptixGeometry::UpdateFsiSphCloud writes
#      OptiX instance transforms straight from the solver's DEVICE pos_rad
#      buffer with a CUDA kernel. Nothing crosses to the host, there is no
#      per-particle Python loop, and it draws fluid markers ONLY -- so the
#      boundary BCE markers that made a global max(z) surface cut meaningless
#      are excluded for free, and no surface extraction is needed at all.
#      Requirements, none of them documented: the sprite templates must be
#      ChVisualShapeTriangleMesh (AddFsiSphVisualization dynamic_pointer_casts
#      to it and silently skips anything else -- a sphere shape renders
#      nothing), and options.render_particle_spacing must be > 0 or
#      EstimateFsiSphRenderCount returns 0 and, again, nothing renders.
# EVERY z BELOW IS MEASURED FROM THE SOIL SURFACE, not from world zero. The
# mount bodies are placed at z = soil_top for exactly this reason: soil_top is
# --soil-bottom + --depth, so with the defaults the bed's surface is at z = 0.20
# and a table written against world zero aims every camera 20 cm too low. The
# first version of this table did that, and put "fixed_ground" at z = 0.16 --
# four centimetres BELOW the surface, i.e. buried in the soil looking at more
# soil.
CAMERAS = [
    # name          mount    position offset (m)      look-at offset (m)
    ("chase",       "yaw",   (-2.15, 0.00, 0.85),     (1.10, 0.00, 0.32)),
    ("side",        "pos",   ( 0.00, 2.60, 0.62),     (0.00, 0.00, 0.30)),
    ("front_3q",    "yaw",   ( 2.40,-1.70, 0.78),     (0.00, 0.00, 0.30)),
    ("low_close",   "yaw",   (-1.55,-1.05, 0.30),     (0.25, 0.00, 0.30)),
    ("topdown",     "pos",   ( 0.00, 0.00, 2.60),     (0.00, 0.001, 0.00)),
    # Steep and behind: the robot sits high in frame and the ground it has just
    # walked over fills the foreground, which is where the tracks are.
    ("trail",       "yaw",   (-2.20, 0.00, 2.20),     (-0.20, 0.00, 0.00)),
    # World-mounted: these do not move, so the robot crosses the frame and the
    # distance covered is directly readable -- which a tracking shot hides.
    # Framed for a run from x = 1.2 to about x = 7.2 on a 10 x 3.6 m bed.
    # Placed at the FAR end of the run and at its START respectively, rather than
    # broadside to it. A broadside fixed camera only holds the robot for the
    # couple of seconds it takes to cross a 3 m frame; down the axis of travel it
    # is in shot for the whole episode and the ground it has covered is the
    # thing the frame is actually measuring.
    ("fixed_wide",  "world", ( 9.50,-4.50, 1.80),     (4.00, 0.00, 0.32)),
    ("fixed_ground","world", (-0.80,-0.85, 0.45),     (6.00, 0.00, 0.08)),
]


def _look(dx, dy, dz):
    """Chrono::Sensor cameras look down their own +X. Yaw about Z, then pitch about
    Y; a positive Y rotation tilts +X toward -Z, so looking down needs pitch > 0."""
    import pychrono as chrono
    yaw = math.atan2(dy, dx)
    pitch = -math.atan2(dz, math.hypot(dx, dy))
    return chrono.QuatFromAngleZ(yaw) * chrono.QuatFromAngleY(pitch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect one Go2 episode while writing the same episode CSV/index "
            "schema used by the HMMWV datasets."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--terrain", choices=["rigid", "crm"], default="crm")
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--step-size-s", type=float, default=5e-4)
    parser.add_argument("--exchange-mult", type=int, default=4)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--record-step-s", type=float, default=0.01,
                        help="100 Hz, matching the HMMWV collector. Control runs at "
                             "50 Hz so joint targets are stair-stepped across sample "
                             "pairs -- that is what the plant does. NOTE the SPH "
                             "surface probe also runs at control rate, so "
                             "foot_*_surface_disp_m is zero-order held across pairs; "
                             "every other channel is genuinely 100 Hz.")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-noise-sigma-rad", type=float, default=0.0,
        help="Gaussian noise added to the twelve joint TARGETS each control step, "
             "in radians. Every other perturbation this collector offers is an "
             "external push on the BODY, which leaves the action an exact function "
             "of the state: measured, only ~4%% of action variance survives "
             "conditioning on the state, over an effectively rank-2 subspace of a "
             "12-dimensional action. A surrogate fit on that data can ignore the "
             "action entirely and still fit, which is why it is undefined once a "
             "fine-tuned policy leaves the a = pi(s) manifold. This flag is the "
             "only thing here that decorrelates action from state. Applied AFTER "
             "the policy acts and BEFORE actuation, so the logged action is the "
             "one actually applied -- logging the clean action would recreate the "
             "confound in the dataset while hiding it.")
    parser.add_argument("--heading-deg", type=float, default=0.0)
    parser.add_argument("--spawn-x-m", type=float, default=0.0)
    parser.add_argument("--spawn-y-m", type=float, default=0.0)
    # 0.2 matches the HMMWV (6,644 val against 26,124 train). NOT 0.0: a zero
    # ratio marks every episode "train" and produces a dataset with no held-out
    # set, which fails SILENTLY -- training runs, the loss falls, and there is
    # nothing to select a checkpoint on or detect overfitting with. Every
    # deployed model in the paper is chosen by held-out rollout error, so this
    # ratio is load-bearing rather than a convenience. The first 1,042 episodes
    # were collected at 0.0 and had their splits recomputed in a retrofit pass.
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--soil", choices=["soft", "hmmwv_reference", "training", "eval"],
                        default="training",
                        help="soil STIFFNESS, not a role. 'soft' (alias 'training') is what "
                             "every collected episode and the Chrono eval use; "
                             "'hmmwv_reference' (alias 'eval') matches the HMMWV study and is "
                             "unused. Default stays 'training' so new episodes record the same "
                             "value as the 304 already collected.")
    parser.add_argument("--depth", type=float, default=0.20)
    parser.add_argument("--soil-bottom", type=float, default=0.0)
    parser.add_argument("--spacing", type=float, default=0.02)
    parser.add_argument("--artificial-viscosity", type=float, default=2.0)
    parser.add_argument("--foot-margin-spacings", type=float, default=2.0)
    parser.add_argument("--spawn-clearance", type=float, default=None)
    parser.add_argument("--pose-ramp-seconds", type=float, default=0.75)
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--solver-iters", type=int, default=150)
    parser.add_argument("--actuation", choices=["torque", "position"], default="torque")
    parser.add_argument("--imported-ckpt", default=None)
    # MID-EPISODE POLICY BRANCH, off unless both are given, so nothing about an
    # ordinary collection changes. Exists so two arms can be bit-identical up to a
    # known instant and differ only after it: a weight-perturbed policy otherwise
    # differs from step 0, and by the time any fixed-length history window has
    # filled, the arms have been diverging for the whole window. Measured: 0.068 m/s
    # of body-velocity difference already present at the first row of a 128-row
    # window, against a surrogate whose two arms start at exactly 0 by construction.
    parser.add_argument("--switch-ckpt", default=None,
                        help="Checkpoint to swap the policy's WEIGHTS to mid-episode.")
    parser.add_argument("--switch-ckpt-at-s", type=float, default=None,
                        help="Simulation time at which to swap. Requires --switch-ckpt.")
    parser.add_argument("--command-family", default=None)
    parser.add_argument("--command-params", default=None,
                        help="JSON dict of this episode's stratified amplitude draw.")
    parser.add_argument("--assets", default=DEFAULT_ASSETS)
    parser.add_argument("--patch-x", type=float, default=8.0)
    parser.add_argument("--patch-y", type=float, default=4.0)
    # Rigid ground extent. 10.0 reproduces every earlier rigid collection; long
    # episodes need more, since travel is duration x commanded speed.
    parser.add_argument("--ground-size-m", type=float, default=10.0)
    # DIVERSITY. All default to OFF so every earlier collection reproduces byte
    # for byte; the new run switches them on explicitly.
    #
    # Peak trunk impulse in newtons. Scale reference: the robot weighs ~158 N, so
    # 80 N is roughly half body weight -- enough to visibly disturb a gait, and the
    # top of the range falls it, which is where the fall coverage comes from.
    parser.add_argument("--perturb-peak-n", type=float, default=0.0)
    parser.add_argument("--perturb-mean-interval-s", type=float, default=2.0)
    parser.add_argument("--perturb-duration-s", type=float, default=0.10)
    # TORQUE PERTURBATION. perturb_torque_x/y/z_nm have been recorded since the
    # collector was written and were identically zero in 210 of 210 episodes sampled
    # at offset 3,000,000 -- the columns were faithful, the channel was never driven.
    # A pure torque rotates the trunk without translating the centre of mass, which
    # is the only excitation that unloads the FRONT or REAR pair together. A trot is
    # always diagonal, so those configurations (modes 3 and 12) are structurally
    # unreachable by pushing, and they measure 0.91% and 0.06% of transitions.
    parser.add_argument("--perturb-torque-peak-nm", type=float, default=0.0)
    # AMPLITUDE SCALE APPLIED AFTER EVERY DRAW, for a matched no-perturbation control.
    # Setting --perturb-peak-n 0 to build that control changes the CODE PATH: the guard
    # at the impulse block is skipped, so its rng.uniform draws are never taken and the
    # shared stream diverges from the perturbed arm at the first event. The two arms
    # then differ in every subsequent random quantity, not only in perturbation -- the
    # same defect as the draw-order bugs documented in that block.
    #
    # This scales the assembled vector instead, so the draws, the branch and the event
    # timing are bit-identical between arms and only the applied force differs. Default
    # 1.0 multiplies by exactly one and takes no draw, so no existing corpus changes.
    parser.add_argument("--perturb-scale", type=float, default=1.0,
                        help="multiply applied perturbation force/torque by this "
                             "(1.0 = normal, 0.0 = matched control with identical draws)")
    # Per-episode ground tilt, degrees, applied as roll and pitch of the static box.
    parser.add_argument("--ground-tilt-roll-deg", type=float, default=0.0)
    parser.add_argument("--ground-tilt-pitch-deg", type=float, default=0.0)
    # NON-QUIESCENT START: seconds of policy-driven walking after the settle and
    # BEFORE recording begins, so an episode starts mid-gait with velocity and
    # arbitrary phase instead of from rest.
    parser.add_argument("--prewalk-s", type=float, default=0.0)
    parser.add_argument("--log-warmup", action="store_true",
                        help="also record the pose ramp, settle and prewalk that are "
                             "normally discarded. Diagnostic only -- see next_record_s.")
    parser.add_argument("--soil-young", type=float, default=None)
    parser.add_argument("--soil-cohesion", type=float, default=None)
    parser.add_argument("--no-calf-fsi", action="store_true")
    parser.add_argument("--render-fps", type=float, default=24.0)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=540)
    parser.add_argument("--render-dir", default="/tmp/go2_render")
    parser.add_argument("--render-particles", type=int, default=0,
                        help="DEAD FLAG, kept so old command lines still parse. The soil "
                             "is no longer drawn as a host-side proxy cloud; see "
                             "--render-particle-spacing.")
    parser.add_argument("--render-particle-radius", type=float, default=0.028,
                        help="DEAD FLAG, kept so old command lines still parse. See "
                             "--render-sprite-scale.")
    parser.add_argument("--render-particle-spacing", type=float, default=0.0,
                        help="Visual spacing, in metres, handed to Chrono::Sensor's native "
                             "FSI-SPH sprite renderer. The number of sprites drawn is "
                             "num_fluid_markers * (initial_spacing / this)^3, so setting it "
                             "EQUAL to --spacing draws one sprite per fluid marker and "
                             "anything larger thins them by the cube of the ratio. 0 means "
                             "'use --spacing'. Nothing renders if this is <= 0. "
                             "MEASURED: at --render-ray-recursions 2 the sprites are "
                             "nearly free (886k of them cost about as much as none at "
                             "all against CRM's own step), so there is no reason to "
                             "thin them. It is the RECURSION DEPTH that costs -- 4 "
                             "instead of 2 added 30%% to a seven-camera run.")
    parser.add_argument("--render-sprite-scale", type=float, default=0.0,
                        help="Uniform scale on the regolith sprite mesh, which is ~7.1 mm "
                             "across as authored. 0 means 'size it to 1.20x the render "
                             "spacing', which is what makes the bed read as a continuous "
                             "surface rather than a sparse ball pit. Below about 1.0 the "
                             "sprites stop touching and the bed goes see-through.")
    parser.add_argument("--render-sprite-meshes", type=int, default=3,
                        help="How many of data/models/regolith/particle_N.obj to use as "
                             "sprite templates. More templates means less visible tiling.")
    parser.add_argument("--render-soil-color", default="0.52,0.41,0.29")
    parser.add_argument("--render-ray-recursions", type=int, default=2,
                        help="OptiX recursion depth. Chrono defaults to 9, which "
                             "buys nothing on a matte soil bed and a matte robot "
                             "and costs real time per frame.")
    parser.add_argument("--render-ground", type=float, default=60.0,
                        help="Edge length of a visual-only plain drawn under the SPH "
                             "bed, so the fixed cameras have a horizon. 0 disables it.")
    parser.add_argument("--render-cameras", default="",
                        help="Comma-separated subset of CAMERAS by name; empty means all.")
    parser.add_argument("--no-check-embedded", dest="check_embedded",
                        action="store_false", default=True)
    parser.add_argument("--progress-interval-s", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    # build_crm reads args.step; this collector names the flag --step-size-s to
    # match the HMMWV collector's simulation.step_size_s. Alias rather than
    # rename, so both conventions stay intact and the terrain builder is shared
    # with the gate script rather than forked.
    args.step = args.step_size_s
    return args


def build_collector_config(args: argparse.Namespace, soil: dict[str, Any]) -> dict[str, Any]:
    exchange_s = args.exchange_mult * args.step_size_s
    config: dict[str, Any] = {
        "dataset_name": "go2_smoke",
        "output_subdir": str(args.output_dir),
        "simulation": {
            "step_size_s": float(args.step_size_s),
            "exchange_step_s": float(exchange_s),
            "record_step_s": float(args.record_step_s),
            "control_hz": float(args.control_hz),
            "validation_ratio": float(args.validation_ratio),
            "solver_iterations": int(args.solver_iters),
        },
        "robot": {
            "model": "Unitree_Go2",
            "urdf": "data/robot/go2_irrvis/urdf/go2_description.urdf",
            "contact_method": "SMC",
            "init": {
                "x_m": float(args.spawn_x_m),
                "y_m": float(args.spawn_y_m),
                "heading_deg": float(args.heading_deg),
            },
        },
        "controller": {
            "policy": (str(args.imported_ckpt) if args.imported_ckpt
                       else "model_2999.pt (harness contract)"),
            "command_family": args.command_family,
            "command_constant": args.imported_ckpt is None,
            "command_source": ("structured family, see command_series"
                               if args.imported_ckpt else
                               "hardcoded literal in chrono_crmenv._compute_observations"),
            "actuation_plant": args.actuation,
        },
        "logging": {"include_foot_channels": True},
        "scenario_generator": {"seed": int(args.seed)},
    }
    if args.terrain == "rigid":
        config["terrain"] = {"type": "rigid", "label": "flat", "top_z_m": 0.05}
        config["logging"]["foot_force_source"] = "chrono_contact"
    else:
        config["terrain"] = {
            "type": "crm",
            "label": "crm",
            "depth_m": float(args.depth),
            "bottom_z_m": float(args.soil_bottom),
            "initial_spacing_m": float(args.spacing),
            # PER EPISODE, never assumed from a global. CRM lateral episodes use a
            # wider bed than every other family (see the lateral note in the
            # collection docs), so bed extent has to be readable from the episode
            # itself rather than inferred from the collector defaults.
            "patch_x_m": float(args.patch_x),
            "patch_y_m": float(args.patch_y),
            "soil_preset": args.soil,
            "soil": dict(soil),
            "sph": {"artificial_viscosity": float(args.artificial_viscosity)},
        }
        config["logging"]["foot_force_source"] = "crm_fsi"
    return config


def first_nonfinite_row(rows: list[dict[str, Any]], allow_nan: set[str]) -> int | None:
    """Index of the first row carrying a non-finite value, or None.

    TRUNCATE, DO NOT DISCARD. A violent fall can diverge the contact solver
    mid-tumble: measured, roll runs 16 -> 42 degrees over six samples and the
    next row is NaN. Everything before that is a real recorded fall and is
    exactly the coverage this collection exists to get, so the episode is cut at
    the first bad row and LABELLED rather than thrown away. Raising here would
    discard the falls and keep only the episodes that never lost stability --
    selecting against the data we are trying to collect.
    """
    for index, row in enumerate(rows):
        for key, value in row.items():
            if key in allow_nan:
                continue
            if isinstance(value, float) and not math.isfinite(value):
                return index
    return None


def summarize_force(rows: list[dict[str, Any]], total_mass_kg: float, gravity: float) -> dict[str, float]:
    from nedm.quadruped.dataset import LEG_ORDER

    settled = [r for r in rows if r["time_s"] >= 1.5] or rows
    fz = [sum(float(r[f"foot_{leg}_force_fz_n"]) for leg in LEG_ORDER) for r in settled]
    finite = [v for v in fz if math.isfinite(v)]
    return {
        "mean_sum_fz_n": float(sum(finite) / max(len(finite), 1)),
        "weight_n": float(total_mass_kg * gravity),
        "samples": len(finite),
    }


def summarize_pose(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        # Reached when an episode ended before the first recorded row. The old
        # failure here was min() on an empty sequence, which named the symptom and
        # not the cause; the divergence guard in the step loop now catches the
        # usual route in, and this stays as a backstop that says what happened.
        raise ValueError("episode produced zero recorded rows: nothing to summarise")
    return {
        "min_pos_z_m": min(float(r["pos_z_m"]) for r in rows),
        "final_pos_x_m": float(rows[-1]["pos_x_m"]),
        "final_pos_y_m": float(rows[-1]["pos_y_m"]),
        "final_speed_mps": float(rows[-1]["speed_mps"]),
        "max_abs_roll_rad": max(abs(float(r["roll_rad"])) for r in rows),
        "max_abs_pitch_rad": max(abs(float(r["pitch_rad"])) for r in rows),
    }


def run_episode(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    from nedm.hmmwv_data import assign_split       # reused, not reimplemented
    from nedm.quadruped import soilprobe
    from nedm.quadruped.constants import (FALL_TILT_RAD, FOOT_BODIES, GRAVITY,
                                          SOIL_PRESETS, STAND_ACTION)
    from nedm.quadruped.dataset import (capture_row, contact_bodies, csv_field_names,
                                        foot_field_names, whole_robot_com)
    from nedm.quadruped.policy import PolicyController
    from nedm.quadruped.provenance import provenance
    from nedm.quadruped.robot import Go2Robot
    from nedm.quadruped.terrain import build_crm, build_rigid_ground, measure_leg_reach

    import pychrono as chrono
    import pychrono.vehicle as veh
    fsi = None
    if args.terrain == "crm":
        # FAIL FAST ON A BOX WITH NO GPU, before terrain construction rather
        # than at the first device call inside it. The check is backend-neutral
        # -- /dev/nvidiactl for CUDA, /dev/kfd for ROCm/HIP -- because the same
        # checkout runs on the NVIDIA desktop fleet AND on the AMD HPC Fund
        # cluster, whose Chrono is built with CHRONO_GPU_BACKEND=HIP. See
        # nedm.quadruped.gpu for why the driver node is the thing to test.
        from nedm.quadruped.gpu import require_gpu_backend, verify_chrono_backend

        _gpu = require_gpu_backend()
        import pychrono.fsi as fsi

        # Second half of the same guard: the node proved a GPU is here, this
        # proves the pychrono just imported is built for it. Still ahead of
        # terrain construction, which is the expensive part being protected.
        verify_chrono_backend(_gpu)

    np.random.seed(args.seed)
    # Seeded from the episode seed so perturbation timing and direction are part
    # of the episode's identity and replay exactly.
    import random as _random
    rng = _random.Random(args.seed)
    # SEPARATE STREAM, and created only when the flag is on. Drawing action noise
    # from `rng` above would shift every subsequent perturbation draw and break
    # bit-identical replay of every episode already collected -- the same class of
    # break as the perturbation draw-order bug fixed in 519ad1d. With sigma = 0 no
    # draw is taken and nothing about this collector's behaviour changes.
    _action_rng = (np.random.default_rng([args.seed, 0xAC7104])
                   if args.action_noise_sigma_rad > 0.0 else None)
    cwd_at_start = os.getcwd()
    assets = Path(args.assets)
    urdf = assets / "data/robot/go2_irrvis/urdf/go2_description.urdf"
    ckpt = assets / "data/rl_models/rslrl/model_2999.pt"
    cfgs = assets / "data/rl_models/rslrl/cfgs.pkl"
    for f in (urdf, ckpt, cfgs):
        if not f.is_file():
            raise FileNotFoundError(f"missing {f}")

    soil = dict(SOIL_PRESETS[args.soil])
    config = build_collector_config(args, soil)

    output_root = args.output_dir.resolve()
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    system = chrono.ChSystemSMC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -GRAVITY))
    # SLOPE BY TILTING GRAVITY, NOT THE GROUND. Rotating the static box is not a
    # local slope: at 200 m across, 2.5 deg is a 4.4 m ramp, and a 40 s episode
    # climbs several metres of it and destabilises -- measured, 6 of 16 probe
    # episodes went non-finite and tilt was the isolated cause. A tilted gravity
    # vector is exactly a uniform slope, identical at every position, and the
    # body attitude the policy senses responds the same way.
    _grav_world = [0.0, 0.0, -9.81]
    if args.ground_tilt_roll_deg or args.ground_tilt_pitch_deg:
        _r = math.radians(args.ground_tilt_roll_deg)
        _p = math.radians(args.ground_tilt_pitch_deg)
        _grav_world = [9.81 * math.sin(_p), -9.81 * math.sin(_r),
                       -9.81 * math.cos(_r) * math.cos(_p)]
        system.SetGravitationalAcceleration(chrono.ChVector3d(*_grav_world))
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
    system.GetSolver().AsIterative().SetMaxIterations(args.solver_iters)
    chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.0025)
    chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.0025)

    rigid = args.terrain == "rigid"
    soil_top = 0.05 if rigid else args.soil_bottom + args.depth
    if args.spawn_clearance is None:
        os.chdir(urdf.parent)
        try:
            leg_reach = measure_leg_reach(chrono, urdf)
        finally:
            os.chdir(cwd_at_start)
        spawn_z = soil_top + args.foot_margin_spacings * args.spacing + leg_reach
    else:
        spawn_z = soil_top + args.spawn_clearance

    init = chrono.ChFramed(
        chrono.ChVector3d(float(args.spawn_x_m), float(args.spawn_y_m), spawn_z),
        chrono.QuatFromAngleZ(math.radians(float(args.heading_deg))),
    )
    os.chdir(urdf.parent)
    try:
        robot = Go2Robot(system, urdf, init, actuation=args.actuation)
    finally:
        os.chdir(cwd_at_start)

    # BEFORE the terrain exists, and that ordering is load-bearing. Summing
    # system bodies afterwards picks up the rigid ground box, which is 10000 kg,
    # and reports a 16 kg robot as weighing 98 kN.
    #
    # It was caught by eye, not by the check: summarize_force only tests for a
    # NON-POSITIVE mean load, so a 600x error in the reference sailed through as
    # PASS with a printed ratio of 0.00. A validation that cannot fail on a 600x
    # error is not a validation. If you add a reference quantity here, give it a
    # two-sided bound -- the ratio should be near 1.0 and both directions are
    # informative.
    robot_mass = sum(b.GetMass() for b in system.GetBodies())

    if rigid:
        build_rigid_ground(chrono, system, size_m=float(args.ground_size_m))
        terrain, coupled = None, []
    else:
        # ASSERT THE SPAWN IS ON THE BED. build_crm centres the patch at
        # x = patch_x/2 - 0.6, so the soil spans [-0.6, patch_x - 0.6] -- NOT
        # symmetric about the origin, which is the trap. A sweep over
        # spawn_x in [-1.5, 0.5] once put 21 episodes off the near edge; all 21
        # recorded zero foot load and were only caught by reading the data
        # afterwards. Bounds that are checked by inspection get checked for the
        # reason you happened to think of; this fails at episode 1 instead.
        cx = args.patch_x / 2 - 0.6
        x_lo, x_hi = cx - args.patch_x / 2, cx + args.patch_x / 2
        y_lo, y_hi = -args.patch_y / 2, args.patch_y / 2
        margin = 0.5
        if not (x_lo + margin <= args.spawn_x_m <= x_hi - margin):
            raise ValueError(
                f"spawn_x_m={args.spawn_x_m} is outside the CRM bed "
                f"[{x_lo:.2f}, {x_hi:.2f}] with {margin} m margin")
        if not (y_lo + margin <= args.spawn_y_m <= y_hi - margin):
            raise ValueError(
                f"spawn_y_m={args.spawn_y_m} is outside the CRM bed "
                f"[{y_lo:.2f}, {y_hi:.2f}] with {margin} m margin")
        terrain, coupled = build_crm(chrono, fsi, veh, system, robot, args)

    if args.imported_ckpt:
        from nedm.quadruped.imported_policy import CHRONO_TO_IMPORTED, ImportedGo2Policy
        params = json.loads(args.command_params) if args.command_params else None
        policy = ImportedGo2Policy(Path(args.imported_ckpt), family=args.command_family,
                                   duration=args.duration_s, params=params)
    else:
        policy = PolicyController(ckpt, cfgs)
    _switch_ck = Path(args.switch_ckpt) if args.switch_ckpt else None
    if (_switch_ck is None) != (args.switch_ckpt_at_s is None):
        raise SystemExit("--switch-ckpt and --switch-ckpt-at-s must be given together")
    _switched = False
    sph_probe = soilprobe.bind_probe(terrain)

    exchange = args.exchange_mult * args.step_size_s
    control_every = max(1, int(round((1.0 / args.control_hz) / exchange)))
    n_steps = int(args.duration_s / exchange)
    base = robot.base()


    terrain_label = config["terrain"]["label"]
    episode_id = f"go2_{terrain_label}_{args.episode_index:03d}"
    scenario_name = episode_id
    # THE COMMANDED FAMILY, NOT A LITERAL. This said "constant_command" for every
    # episode regardless of what was commanded, which is the defect already in
    # docs/state/lessons/experiment-design.md -- and it was still writing one value
    # in data collected 2026-09-07, because the entry recorded the consequence and
    # nothing changed the source.
    #
    # It is not cosmetic. trainer.py:797 buckets on this field and round-robins
    # across families to choose the twelve rollout episodes the DEPLOYED CHECKPOINT
    # is selected on. With one bucket the round-robin degenerates to "take the first
    # twelve in list order", and the output is still a rollout error over twelve
    # episodes -- which is exactly what a correct one looks like.
    #
    # args.command_family is what the driver actually passed, so it cannot drift
    # from the commanded motion the way a separate label can.
    # --command-family defaults to None, so a caller that omits it would get
    # "go2_flat_None" -- a new wrong value in place of the old one. Fall back to the
    # historical literal in that case, which keeps every existing invocation
    # reproducing and confines the change to callers that actually state a family.
    scenario_family = (f"go2_{terrain_label}_{args.command_family}"
                       if args.command_family else f"go2_{terrain_label}_constant_command")
    split = assign_split(episode_id, float(args.validation_ratio))
    csv_path = episodes_dir / f"{episode_id}.csv"

    BED_MARGIN = 0.8
    if rigid:
        # DERIVED FROM THE GROUND SIZE, not restated. These were two hardcoded
        # constants that had to agree (a 10 x 10 box and a +/-5 bed); deriving one
        # from the other means enlarging the ground cannot leave the bed behind.
        half = float(args.ground_size_m) / 2.0
        bed = (-half, half, -half, half)
    else:
        cx = args.patch_x / 2 - 0.6
        bed = (cx - args.patch_x / 2, cx + args.patch_x / 2,
               -args.patch_y / 2, args.patch_y / 2)
    # A SPAWN OUTSIDE THE USABLE BED PRODUCES ZERO ROWS AND NO EXPLANATION.
    # The CRM bed is centred at patch_x/2 - 0.6, so with the 8.0 m default the
    # usable x-range after BED_MARGIN is [0.20, 6.60] -- and --spawn-x-m defaults
    # to 0.0, which is OUTSIDE it. The boundary check below then fires on the first
    # step and the episode dies with "zero recorded rows", which reads as a physics
    # failure rather than an argument that was never going to work. CRM collection
    # was impossible with default arguments and the error said nothing about why.
    # NOTE: drive_go2_collection.py already passes a valid CRM spawn (x = 0.9 for
    # forward travel), so no collection run was ever broken by this -- the trap is
    # for anyone invoking this collector directly, which is how it was found.
    # Check it up front, name the numbers, and refuse.
    if not (bed[0] + BED_MARGIN <= args.spawn_x_m <= bed[1] - BED_MARGIN
            and bed[2] + BED_MARGIN <= args.spawn_y_m <= bed[3] - BED_MARGIN):
        raise SystemExit(
            f"FATAL: spawn ({args.spawn_x_m:.2f}, {args.spawn_y_m:.2f}) is outside the "
            f"usable bed x[{bed[0] + BED_MARGIN:.2f}, {bed[1] - BED_MARGIN:.2f}] "
            f"y[{bed[2] + BED_MARGIN:.2f}, {bed[3] - BED_MARGIN:.2f}] "
            f"(bed x[{bed[0]:.2f}, {bed[1]:.2f}] y[{bed[2]:.2f}, {bed[3]:.2f}], "
            f"margin {BED_MARGIN}). The episode would record zero rows. "
            f"Pass --spawn-x-m/--spawn-y-m inside the bed, or enlarge --patch-x/--patch-y.")

    boundary_at = None
    solver_diverged_at = None

    q0 = robot.joint_pos().astype(np.float64)
    robot.actuate(q0)
    action = q0.copy()
    tau = np.zeros(12)
    policy_raw = np.full(12, float("nan"))
    perturb = np.zeros(6)
    _acc = base.AddAccumulator()
    _pert_until = -1.0
    # BOTH CHANNELS, not just force. This gate was left testing perturb_peak_n alone
    # when torque was added: with force 0 and torque 80 N.m the first trigger time
    # stayed at infinity, so nothing ever fired and the torque bracket measured a
    # perfectly flat zero at every level. The trigger condition below had been
    # updated; this initialiser had not. A partial fix reads exactly like a working
    # one when the control row is also zero.
    _pert_any = (args.perturb_peak_n > 0.0) or (args.perturb_torque_peak_nm > 0.0)
    _pert_events = 0
    _pert_next = (rng.expovariate(1.0 / max(args.perturb_mean_interval_s, 1e-9))
                  if _pert_any else float("inf"))
    rows: list[dict[str, Any]] = []
    # DISCARD THE WARMUP, as the HMMWV collector does ("each episode discards an
    # initial settling transient before recording"). Before this the robot is on
    # the pose ramp and the settle hold, not under policy control, so the command
    # columns would carry the constructor default rather than the family and the
    # dynamics are a drop transient rather than locomotion.
    # Recording starts AFTER the prewalk, so the first recorded row is mid-gait.
    warmup_s = args.pose_ramp_seconds + args.settle_seconds + float(args.prewalk_s)
    # --log-warmup RECORDS THE DISCARDED TRANSIENT. The window above is normally
    # dropped, which means every artefact this project has starts AFTER the policy has
    # been in closed loop for seconds. A fine-tuned policy was found to be at 1e10 rad
    # by its first recorded row, so the failure happens entirely inside this window and
    # nothing in the corpus contains it. Off by default: the rows are a drop transient
    # and a settle hold, not locomotion, and pooling them with training data would be
    # the mistake the discard exists to prevent.
    next_record_s = 0.0 if args.log_warmup else warmup_s
    # A DISTURBANCE PARAMETER THAT SILENTLY PRODUCES ZERO EVENTS IS A VACUOUS TEST.
    # The first event is drawn from expovariate(1/mean_interval) and gated on
    # t > warmup_s * 0.5, so the eligible window is duration - warmup_s/2. With the
    # 2.0 s default mean interval, a 3 s episode leaves 1.88 s of eligible time and
    # frequently fires NOTHING -- the parameter is accepted, recorded in the config,
    # and never used. That produced three "with perturbation" warmup runs on
    # 2026-09-07 that tested no perturbation at all, reported as the corrected
    # version of an under-disturbed test.
    if _pert_any:
        _elig = float(args.duration_s) - warmup_s * 0.5
        _ratio = _elig / max(args.perturb_mean_interval_s, 1e-9)
        if _ratio < 3.0:
            print(f"  WARNING: perturbation eligible window {_elig:.2f}s is only "
                  f"{_ratio:.1f}x the mean interval "
                  f"{args.perturb_mean_interval_s:.2f}s -- this episode may record "
                  f"ZERO disturbance events. Lengthen --duration-s or shorten "
                  f"--perturb-mean-interval-s.", flush=True)
    # ---- RENDER SETUP -----------------------------------------------------
    _rman = None; _rmounts = {}; _rnext = 0.0
    _rsprites = []          # sprite templates must outlive the manager
    try:
        import pychrono.sensor as sens
        rd = Path(args.render_dir); rd.mkdir(parents=True, exist_ok=True)
        _rman = sens.ChSensorManager(system)
        _rman.SetRayRecursions(int(args.render_ray_recursions))
        _rman.scene.SetAmbientLight(chrono.ChVector3f(0.30, 0.31, 0.35))
        # A LOW sun, 34 degrees rather than the 52 this started at. Foot
        # depressions here are about 2 cm deep -- one SPH particle -- so they are
        # nearly invisible under a high sun and read clearly under a raking one,
        # which is the whole reason for filming on CRM rather than on rigid.
        _rman.scene.AddDirectionalLight(
            chrono.ChColor(1.05, 1.00, 0.90), math.radians(34.0), math.radians(130.0))
        # Weak opposite fill so the shaded side of the robot does not go to black.
        _rman.scene.AddDirectionalLight(
            chrono.ChColor(0.20, 0.22, 0.27), math.radians(48.0), math.radians(-55.0))
        for key in ("yaw", "pos", "world"):
            b = chrono.ChBody(); b.SetFixed(True); b.EnableCollision(False)
            b.SetPos(chrono.ChVector3d(0.0, 0.0, soil_top))
            system.AddBody(b); _rmounts[key] = b
        # A visual-only plain under the bed so the fixed cameras, which look at
        # the pit from outside it, have a horizon instead of the bed hanging in
        # a sky gradient. Sunk below the surface so it cannot hide a rut: it
        # only shows where the SPH box ends. No collision, no FSI, no physics.
        if args.render_ground > 0.0:
            # FOUR slabs around the bed, not one slab under it. A single big box
            # has to sit either below the surface -- and then the bed stands on a
            # visible pedestal, which is what the fixed cameras saw -- or flush
            # with it, and then it fills in every rut, which destroys the only
            # thing CRM is here to show. A surround is flush AND leaves the bed
            # itself untouched.
            _gnd = chrono.ChBody(); _gnd.SetFixed(True); _gnd.EnableCollision(False)
            _gcol = [float(v) for v in args.render_soil_color.split(",")]
            _gclr = chrono.ChColor(*[c * 0.92 for c in _gcol])
            _G = float(args.render_ground)
            _cx = args.patch_x / 2 - 0.6
            _hx, _hy = args.patch_x / 2, args.patch_y / 2
            _tz = soil_top - 0.004 - 0.2          # slab is 0.4 thick; this is its centre
            _out = (_G - args.patch_x) / 2
            _outy = (_G - args.patch_y) / 2
            for _sx, _sy, _px, _py in (
                    (_out, _G, _cx - _hx - _out / 2, 0.0),          # behind the bed
                    (_out, _G, _cx + _hx + _out / 2, 0.0),          # beyond the bed
                    (args.patch_x, _outy, _cx, -_hy - _outy / 2),   # near side
                    (args.patch_x, _outy, _cx, _hy + _outy / 2)):   # far side
                _gb = chrono.ChVisualShapeBox(_sx, _sy, 0.4)
                _gb.SetColor(_gclr)
                _gnd.AddVisualShape(_gb, chrono.ChFramed(
                    chrono.ChVector3d(_px, _py, _tz), chrono.QUNIT))
            system.AddBody(_gnd)
        # ---- SOIL, via the native FSI-SPH sprite renderer ------------------
        # Attached BEFORE the cameras. AttachFsiSphSystem calls ReconstructScenes,
        # which is a no-op while no engine exists; the engine is then created by
        # AddSensor and picks the source list up through SetFsiSphSources. Doing
        # it in this order also matches demo_SEN_CRM_Rendering.cpp, which is the
        # only worked example of this API that exists.
        _rspacing = float(args.render_particle_spacing) or float(args.spacing)
        if terrain is not None and _rspacing > 0.0:
            _sc = float(args.render_sprite_scale) or (1.20 * _rspacing / 0.0071)
            _col = [float(v) for v in args.render_soil_color.split(",")]
            _smat = chrono.ChVisualMaterial()
            _smat.SetAmbientColor(chrono.ChColor(*[c * 0.55 for c in _col]))
            _smat.SetDiffuseColor(chrono.ChColor(*_col))
            _smat.SetSpecularColor(chrono.ChColor(0.06, 0.06, 0.06))
            _smat.SetUseSpecularWorkflow(True)
            _smat.SetRoughness(0.95)
            _opts = sens.ChFsiSphRenderOptions()
            for _k in range(1, max(1, int(args.render_sprite_meshes)) + 1):
                _mp = chrono.GetChronoDataFile(f"models/regolith/particle_{_k}.obj")
                # load_normals=True is LOAD-BEARING. Without vertex normals the
                # OptiX hit program has no shading normal for these meshes and
                # every sprite comes back black with a white specular speck --
                # an 8x4 m bed of soot with the robot lit correctly on top of it.
                _mesh = chrono.ChTriangleMeshConnected().CreateFromWavefrontFile(
                    _mp, True, True)
                _msh = chrono.ChVisualShapeTriangleMesh()
                _msh.SetMesh(_mesh)
                _msh.SetMutable(False)
                _msh.SetScale(chrono.ChVector3d(_sc, _sc, _sc))
                # SetMesh leaves the shape holding ONE default material, Kd
                # (0.5, 0.5, 0.5). AddMaterial APPENDS, and the renderer reads
                # slot 0, so adding a soil material renders the default grey and
                # nothing about the frame says why. Overwrite slot 0 in place.
                if _msh.GetNumMaterials() == 0:
                    _msh.AddMaterial(_smat)
                else:
                    _m0 = _msh.GetMaterials()[0]
                    _m0.SetAmbientColor(chrono.ChColor(*[c * 0.55 for c in _col]))
                    _m0.SetDiffuseColor(chrono.ChColor(*_col))
                    _m0.SetSpecularColor(chrono.ChColor(0.06, 0.06, 0.06))
                    _m0.SetUseSpecularWorkflow(True)
                    _m0.SetRoughness(0.95)
                _rsprites.append(_msh)
                _opts.sprite_shapes.append(_msh)
            _opts.render_particle_spacing = _rspacing
            # Jitter breaks the 0.02 m lattice up so the bed does not read as a
            # grid of identical beads under a raking light. Deterministic: it is
            # hashed off the instance index, not sampled, so two policies on the
            # same episode get the same soil grain and the pair stays comparable.
            _opts.sprite_position_jitter = chrono.ChVector3f(
                0.35 * _rspacing, 0.35 * _rspacing, 0.12 * _rspacing)
            _rman.AttachFsiSphSystem(terrain.GetFluidSystemSPH(), _opts)
            _nfl = int(terrain.GetNumSPHParticles())
            _est = int(_nfl * (float(args.spacing) / _rspacing) ** 3)
            print(f"  render: soil = {_est} FSI-SPH sprites from {_nfl} fluid "
                  f"markers (render spacing {_rspacing:.4f} m, sprite scale "
                  f"{_sc:.2f}, {len(_rsprites)} templates)", flush=True)
        _want = [c.strip() for c in args.render_cameras.split(",") if c.strip()]
        for nm, mount, off, tgt in CAMERAS:
            if _want and nm not in _want: continue
            d = (tgt[0] - off[0], tgt[1] - off[1], tgt[2] - off[2])
            pose = chrono.ChFramed(chrono.ChVector3d(*off), _look(*d))
            cam = sens.ChCameraSensor(_rmounts[mount], args.render_fps, pose,
                                      args.render_width, args.render_height,
                                      math.radians(58.0))
            cam.SetName(nm); cam.SetLag(0.0); cam.SetCollectionWindow(0.0)
            cdir = rd / nm; cdir.mkdir(parents=True, exist_ok=True)
            cam.PushFilter(sens.ChFilterSave(str(cdir) + "/"))
            _rman.AddSensor(cam)
        _ncam = sum(1 for nm, _m, _o, _t in CAMERAS if not _want or nm in _want)
        print(f"  render: {_ncam} cameras -> {rd}", flush=True)
    except Exception as _exc:
        print(f"  render DISABLED: {type(_exc).__name__}: {_exc}", flush=True)
        _rman = None

    next_progress_s = 0.0
    sample_index = 0
    fell_at: float | None = None
    wall0 = time.perf_counter()

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_field_names())
        writer.writeheader()

        soil_z, soil_ctrl = [float("nan")] * len(FOOT_BODIES), float("nan")
        for i in range(n_steps):
            t = i * exchange
            if i % control_every == 0:
                # Advance the command schedule UNCONDITIONALLY, including during
                # the ramp. It has no effect there -- the policy is not driving --
                # but it keeps the cmd_* columns showing the schedule rather than
                # a stale constructor default on the first recorded row.
                if policy is not None and hasattr(policy, "set_time"):
                    policy.set_time(t)
                if t < args.pose_ramp_seconds:
                    a = t / max(args.pose_ramp_seconds, 1e-9)
                    action = q0 + a * (STAND_ACTION - q0)
                elif t < args.pose_ramp_seconds + args.settle_seconds:
                    action = STAND_ACTION.astype(np.float64)
                else:
                    if (_switch_ck is not None and not _switched
                            and t >= args.switch_ckpt_at_s):
                        # WEIGHTS ONLY. Replacing the policy object would reset its
                        # 5-step observation history and inject a discontinuity that
                        # has nothing to do with the weight change being measured.
                        policy.model = policy.torch.jit.load(str(_switch_ck),
                                                             map_location="cpu")
                        _switched = True
                    action = policy.act(robot)
                    # The network's raw output, before ACTION_SCALE and the
                    # defaults. Reordered to Chrono order for column naming only;
                    # the sign convention is NOT applied, because "raw" means the
                    # number the policy emitted.
                    _raw = np.zeros(12, dtype=np.float64)
                    _raw[CHRONO_TO_IMPORTED] = policy.last_actions
                    policy_raw = _raw
                    # NOT during the ramp or settle: those phases are a scripted
                    # stand-up, not a policy decision, and perturbing them only
                    # changes how often the robot falls before it starts walking.
                    # policy_raw stays clean -- it means "the number the policy
                    # emitted", and that is still true.
                    if _action_rng is not None:
                        action = action + _action_rng.normal(
                            0.0, args.action_noise_sigma_rad, size=12)
                robot.actuate(action)
                if sph_probe is not None:
                    soil_z, soil_ctrl = soilprobe.sample(sph_probe, robot)

            # PERTURBATION: Poisson-timed impulses of random direction, held for a
            # short window. Re-applied every physics step while active because the
            # accumulator is emptied each step.
            base.EmptyAccumulator(_acc)
            if args.perturb_peak_n > 0.0 or args.perturb_torque_peak_nm > 0.0:
                if t >= _pert_next and t > warmup_s * 0.5:
                    mag = rng.uniform(0.25, 1.0) * args.perturb_peak_n
                    th = rng.uniform(0.0, 2.0 * math.pi)
                    # Torque direction is drawn in the ROLL-PITCH plane, because that
                    # is where the target modes live: pitch unloads front or rear
                    # (modes 3, 12), roll unloads left or right (modes 5, 10). Yaw is
                    # given a small share only -- it spins the trunk without changing
                    # which feet carry load, so it excites nothing we are short of.
                    # EVERY TORQUE DRAW IS INSIDE THE GUARD, and the force z-component
                    # is drawn BEFORE any of them. Drawing tmag/tphi unconditionally
                    # advanced the shared RNG by two before the force z, so a
                    # force-only episode replayed with DIFFERENT forces even though the
                    # torque was zero -- measured on vel_step_140 as 155 of 164 physics
                    # columns differing from row 207. Multiplying a draw by zero still
                    # consumes it, which is why `tmag * rng.uniform(-0.2, 0.2)` has to
                    # be inside the guard too and not merely zeroed.
                    _f = [mag * math.cos(th), mag * math.sin(th),
                          mag * rng.uniform(-0.3, 0.3)]
                    if args.perturb_torque_peak_nm > 0.0:
                        tmag = rng.uniform(0.25, 1.0) * args.perturb_torque_peak_nm
                        tphi = rng.uniform(0.0, 2.0 * math.pi)
                        _t = [tmag * math.cos(tphi), tmag * math.sin(tphi),
                              tmag * rng.uniform(-0.2, 0.2)]
                    else:
                        _t = [0.0, 0.0, 0.0]
                    perturb = np.array(_f + _t)
                    _pert_until = t + args.perturb_duration_s
                    _pert_next = t + rng.expovariate(1.0 / max(args.perturb_mean_interval_s, 1e-9))
                    _pert_events += 1
                if t < _pert_until:
                    _ps = args.perturb_scale
                    if args.perturb_peak_n > 0.0:
                        base.AccumulateForce(_acc, chrono.ChVector3d(*(perturb[:3] * _ps)),
                                             base.GetPos(), False)
                    if args.perturb_torque_peak_nm > 0.0:
                        # local=False: the torque is applied about world axes, so
                        # "pitch" means pitch regardless of the trunk's heading.
                        base.AccumulateTorque(_acc, chrono.ChVector3d(*(perturb[3:6] * _ps)),
                                              False)
                else:
                    perturb = np.zeros(6)
            tau = robot.apply_pd()    # every physics step, not every control step
            if terrain is not None:
                terrain.DoStepDynamics(exchange)
            else:
                system.DoStepDynamics(exchange)

            if _rman is not None and t >= _rnext:
                _rnext = t + 1.0 / args.render_fps
                _bp = base.GetPos()
                # Position-following mount stays world-aligned, so side/top shots do
                # not roll with the trunk. The yaw mount also carries heading, which
                # is what makes the chase shot read as a chase.
                _rmounts["pos"].SetPos(chrono.ChVector3d(_bp.x, _bp.y, 0.0))
                _rq = base.GetRot()
                _yaw = math.atan2(2.0 * (_rq.e0 * _rq.e3 + _rq.e1 * _rq.e2),
                                  1.0 - 2.0 * (_rq.e2 ** 2 + _rq.e3 ** 2))
                _rmounts["yaw"].SetPos(chrono.ChVector3d(_bp.x, _bp.y, 0.0))
                _rmounts["yaw"].SetRot(chrono.QuatFromAngleZ(_yaw))
                # No soil bookkeeping here on purpose. UpdateFsiSphCloud reads the
                # solver's device buffer inside the render thread, so the soil is
                # current by construction and costs the host nothing.
                _rman.Update()

            bp = base.GetPos()
            # DIVERGENCE FIRST, AND SEPARATELY FROM THE BED TEST. A non-finite
            # position fails every comparison below, so `not (lo <= x <= hi)` is
            # True and a diverged solver was being recorded as a bed exit. That
            # mislabelled the ending and, when it happened before the first
            # recorded row, surfaced as "min() iterable argument is empty" three
            # functions away -- 144 of the 238 lost episodes at offset 3,000,000.
            #
            # BREAK, DO NOT RAISE. An earlier version of this guard raised here.
            # That changed behaviour rather than only the diagnosis: a LATE
            # divergence is recoverable, and the post-loop handler below truncates
            # at the first non-finite row and keeps the episode. Raising threw those
            # episodes away and made previously collected ones impossible to
            # reproduce -- caught by the verdict harness's replay check, which
            # requires bit-identical output and refused to run.
            if not (math.isfinite(bp.x) and math.isfinite(bp.y) and math.isfinite(bp.z)):
                solver_diverged_at = t
                break
            if boundary_at is None and not (
                    bed[0] + BED_MARGIN <= bp.x <= bed[1] - BED_MARGIN
                    and bed[2] + BED_MARGIN <= bp.y <= bed[3] - BED_MARGIN):
                boundary_at = t
                break

            if t + 1e-12 >= next_record_s:
                # Contact ground truth is meaningful only on rigid: on CRM the
                # feet couple through FSI and the contact system sees nothing, so
                # None here makes the column NaN rather than a confident False.
                contacts = contact_bodies(chrono, system) if terrain is None else None
                row = capture_row(
                    chrono=chrono,
                    robot=robot,
                    tau=tau,
                    policy_raw=policy_raw,
                    # THE FORCE THAT WAS APPLIED, not the one that was drawn. A
                    # --perturb-scale 0 control draws a full-magnitude impulse and
                    # applies none of it; logging the draw would label every control
                    # episode with perturbations it never felt. Scale 1.0 multiplies
                    # by exactly one, so existing corpora are unchanged.
                    perturb=perturb * args.perturb_scale,
                    gravity=_grav_world,
                    contacts=contacts,
                    com=whole_robot_com(system),
                    terrain=terrain,
                    soil_top_m=soil_top,
                    action=action,
                    command=getattr(policy, "command", (0.0, 0.0, 0.0)),
                    soil_z=soil_z,
                    soil_ctrl=soil_ctrl,
                    scenario_name=scenario_name,
                    scenario_family=scenario_family,
                    episode_id=episode_id,
                    split=split,
                    sample_index=sample_index,
                    time_s=t,
                )
                writer.writerow(row)
                rows.append(row)
                sample_index += 1
                next_record_s += args.record_step_s

            # Tilt, not base height: a robot lying inverted still reports a base
            # z above the soil top. Same criterion as the gate script.
            q = base.GetRot()
            tilt = math.acos(max(-1.0, min(1.0, 1 - 2 * (q.e1 * q.e1 + q.e2 * q.e2))))
            if fell_at is None and tilt > FALL_TILT_RAD:
                fell_at = t

            if args.progress_interval_s > 0 and t + 1e-12 >= next_progress_s:
                p = base.GetPos()
                print(f"t={t:5.2f}s pos=({p.x:+6.3f}, {p.y:+6.3f}, {p.z:5.3f}) "
                      f"tilt={math.degrees(tilt):5.1f} rows={len(rows)}")
                next_progress_s += args.progress_interval_s

    wall_s = time.perf_counter() - wall0
    # surface_disp is NaN before the first probe on CRM and legitimately absent
    # if the probe never bound; nothing else is allowed to be non-finite.
    # NaN is MEANINGFUL in two places and must not be confused with a defect:
    #  - surface_disp_m off CRM: rigid ground genuinely does not deform.
    #  - policy_raw_*: recording can begin between 50 Hz policy control steps
    #    (physics runs at 400 Hz), so the first row or two can precede the first
    #    policy action. NaN there means "the policy was not driving yet", which is
    #    true; filling it with a stale or zero action would be a quiet lie.
    #  - foot_*_in_contact on CRM: the contact system sees nothing there.
    from nedm.quadruped.dataset import POLICY_RAW_ACTION_FIELDS
    allow_nan = {f for f in foot_field_names() if f.endswith("_surface_disp_m")}
    allow_nan |= set(POLICY_RAW_ACTION_FIELDS)
    allow_nan |= {f for f in foot_field_names() if f.endswith("_in_contact")}
    diverged_at = first_nonfinite_row(rows, allow_nan)
    if diverged_at is not None:
        if diverged_at < 50:
            raise ValueError(
                f"non-finite at sample {diverged_at}: too little usable data to keep")
        rows = rows[:diverged_at]
        # The CSV is streamed during the loop, so the non-finite row is ALREADY on
        # disk. Rewrite the file from the truncated list, or the metadata says 185
        # while the file holds 186 and gate G10 fails on a real dataset.
        with csv_path.open("w", newline="", encoding="utf-8") as _h:
            _w = csv.DictWriter(_h, fieldnames=csv_field_names())
            _w.writeheader()
            _w.writerows(rows)

    force_summary = summarize_force(rows, robot_mass, GRAVITY)
    pose_summary = summarize_pose(rows)

    episode_meta = {
        "episode_id": episode_id,
        "scenario_name": scenario_name,
        "scenario_family": scenario_family,
        "split": split,
        "csv_path": str(csv_path.relative_to(output_root)),
        "rows": len(rows),
        "duration_s": float(args.duration_s),
        # COLLECTION PARAMETERS, RECORDED RATHER THAN RE-DERIVED. The verdict
        # harness used to reconstruct these from a seeded RNG duplicated in its own
        # source. That contract broke silently when the driver's pitch range was
        # capped to +-1.5 and the harness kept deriving +-3.0: any corpus collected
        # after that fails the bit-identical replay check for a reason that looks
        # like non-determinism. Recording them removes the duplicated draw instead
        # of asking two files to stay in step.
        "prewalk_s": float(args.prewalk_s),
        "ground_tilt_roll_deg": float(args.ground_tilt_roll_deg),
        "ground_tilt_pitch_deg": float(args.ground_tilt_pitch_deg),
        "perturb_scale": float(args.perturb_scale),
        "perturb_peak_n": float(args.perturb_peak_n),
        "perturb_torque_peak_nm": float(args.perturb_torque_peak_nm),
        "record_step_s": float(args.record_step_s),
        # COUNT, not just the parameter. A recorded --perturb-peak-n proves the
        # parameter was accepted, not that any force was applied.
        "perturb_events": int(_pert_events),
        "warmup_s": float(args.pose_ramp_seconds + args.settle_seconds),
        "terrain_type": args.terrain,
        "terrain_label": terrain_label,
        "foot_force_source": config["logging"]["foot_force_source"],
        "plant": args.actuation,

        "seed": int(args.seed),
        "action_noise_sigma_rad": float(args.action_noise_sigma_rad),
        "heading_deg": float(args.heading_deg),
        "spawn_m": [float(args.spawn_x_m), float(args.spawn_y_m), float(spawn_z)],
        "soil_top_m": float(soil_top),
        "robot_mass_kg": float(robot_mass),
        "fsi_coupled_bodies": len(coupled),
        "crm_particles": int(terrain.GetNumSPHParticles()) if terrain is not None else 0,
        "fell": fell_at is not None,
        # Solver divergence during a tumble. The episode is truncated here, not
        # discarded: the rows before it are a genuine recorded fall.
        "diverged": diverged_at is not None,
        "diverged_at_s": (None if diverged_at is None
                          else float(rows[-1]["time_s"]) if rows else None),
        "policy_switch_ckpt": str(_switch_ck) if _switch_ck is not None else None,
        "policy_switch_at_s": args.switch_ckpt_at_s,
        "policy_switched": _switched,
        "status": ("diverged" if diverged_at is not None
                   else "fell" if fell_at is not None
                   else "bed_boundary" if boundary_at is not None else "completed"),
        # Distinguished from a genuine bed exit: this is the step at which the
        # base position went non-finite, which used to be reported as a boundary.
        "solver_diverged_at_s": solver_diverged_at,
        "bed_boundary_at_s": boundary_at,
        "command_family": args.command_family,
        "command_params": getattr(policy, "params", None),
        "command_series": getattr(policy, "command_log", None),
        "plant_bed_m": list(bed),
        "fell_at_s": fell_at,
        "force_summary": force_summary,
        "pose_summary": pose_summary,
        "wall_clock_s": round(wall_s, 2),
        # WRITTEN AT COLLECTION TIME, NOT BACKFILLED. nedm.quadruped.provenance
        # existed for a whole collection without ever being imported, so every
        # origin field on the shipped episodes was reconstructed afterwards by
        # repair_go2_metadata.py -- which can recover git state from mtimes but
        # CANNOT recover which pychrono was on the path. This box has two Chrono
        # builds that differ in the CRM API, selected only by PYTHONPATH, and
        # working out after the fact which one produced the CRM episodes took an
        # hour and only succeeded because the two happen to disagree about an
        # attribute name. A field that can only be observed while the process is
        # alive has to be written while the process is alive.
        **provenance(),
    }
    (episodes_dir / f"{episode_id}.json").write_text(json.dumps(episode_meta, indent=2) + "\n")

    dataset_index = {
        "dataset_name": config["dataset_name"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode_count": 1,
        "episodes": [
            {
                "episode_id": episode_id,
                "scenario_name": scenario_name,
                "scenario_family": scenario_family,
                "split": split,
                "csv_path": str(csv_path.relative_to(output_root)),
                "rows": len(rows),
                "duration_s": float(args.duration_s),
                "terrain_label": terrain_label,
            }
        ],
    }
    (output_root / "dataset_index.json").write_text(json.dumps(dataset_index, indent=2) + "\n")
    # THE EFFECTIVE ACTION GAIN, which nothing else records.
    #
    # collector_config.resolved.json is named as though it captured the run's resolved
    # parameters. It captures the parameters that flow through the CONFIG system.
    # NEDM_ACTION_MULT is applied at policy level from the environment, outside that
    # path, so it never entered this file -- and the verdict summary did not record it
    # either. Audited 2026-09-07: the checkpoint has two independent records (filename
    # and controller.policy) and the multiplier had exactly one, the filename. The
    # parameter that turned out to be the confound was the one with nothing to
    # cross-check it against.
    #
    # A file that looks like a complete record is complete for one subsystem, and the
    # boundary is invisible from inside the file. Stated here because the next
    # parameter applied outside the config path will look just as absent.
    config["effective_action_mult"] = float(os.environ.get("NEDM_ACTION_MULT", 1.0))
    config["action_mult_source"] = ("NEDM_ACTION_MULT" if "NEDM_ACTION_MULT" in os.environ
                                    else "default (variable unset)")
    (output_root / "collector_config.resolved.json").write_text(json.dumps(config, indent=2) + "\n")
    return dataset_index, episode_meta


def main() -> int:
    args = parse_args()
    dataset_index, meta = run_episode(args)
    ratio = meta["force_summary"]["mean_sum_fz_n"] / max(meta["force_summary"]["weight_n"], 1.0)
    print(f"wrote {meta['rows']} rows to {meta['csv_path']} ({meta['wall_clock_s']} s wall)")
    print(f"mean settled sum Fz = {meta['force_summary']['mean_sum_fz_n']:.0f} N "
          f"vs weight {meta['force_summary']['weight_n']:.0f} N (ratio {ratio:.2f})")
    print("PASS" if not meta["fell"] else f"FELL at {meta['fell_at_s']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
