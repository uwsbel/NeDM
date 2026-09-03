"""Collect one Go2 episode on rigid or CRM terrain in the HMMWV dataset schema.

Mirrors scripts/collection/collect_hmmwv_crm_smoke.py: same episode CSV plus
per-episode JSON plus dataset_index.json plus collector_config.resolved.json
layout, same split assignment, same "write the row at a fixed record cadence
inside the physics loop" shape. The per-step schema lives in
nedm.quadruped.dataset; see that module for where the mirror is not literal.

Terrain labels follow terrain_conditioning: rigid ground is "flat", the training
soil preset is "crm".

THE COMMAND IS CONSTANT IN EVERY EPISODE THIS PRODUCES. model_2999.pt reads a
hardcoded [0.5, 0, 0] built inside the training harness's own observation
function, so no per-episode variation can reach it. What varies here is initial
heading, spawn offset and seed. That is state coverage, not command coverage,
and command_constant is written into the episode metadata so that a later reader
cannot mistake limit-cycle data for a command-conditioned dataset.
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
DEFAULT_ASSETS = "/home/kyle/Documents/sbel/sbel-reproducibility/2025/multi-terrain-RL"


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
    parser.add_argument("--exchange-mult", type=int, default=5)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--record-step-s", type=float, default=0.02,
                        help="Default matches the 50 Hz control step, so every "
                             "sample carries the action actually applied.")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heading-deg", type=float, default=0.0)
    parser.add_argument("--spawn-x-m", type=float, default=0.0)
    parser.add_argument("--spawn-y-m", type=float, default=0.0)
    parser.add_argument("--validation-ratio", type=float, default=0.0)
    parser.add_argument("--soil", choices=["training", "eval"], default="training")
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
    parser.add_argument("--assets", default=DEFAULT_ASSETS)
    parser.add_argument("--patch-x", type=float, default=8.0)
    parser.add_argument("--patch-y", type=float, default=4.0)
    parser.add_argument("--soil-young", type=float, default=None)
    parser.add_argument("--soil-cohesion", type=float, default=None)
    parser.add_argument("--no-calf-fsi", action="store_true")
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
            "policy": "data/rl_models/rslrl/model_2999.pt",
            "contact_method": "SMC",
            "init": {
                "x_m": float(args.spawn_x_m),
                "y_m": float(args.spawn_y_m),
                "heading_deg": float(args.heading_deg),
            },
        },
        "controller": {
            # See the module docstring: this is not configurable, it is a fact
            # about the checkpoint, recorded so the dataset is self-describing.
            "command_constant": True,
            "command_lin_vel_x_mps": 0.5,
            "command_lin_vel_y_mps": 0.0,
            "command_ang_vel_yaw_radps": 0.0,
            "command_source": "hardcoded literal in chrono_crmenv._compute_observations",
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
            "soil_preset": args.soil,
            "soil": dict(soil),
            "sph": {"artificial_viscosity": float(args.artificial_viscosity)},
        }
        config["logging"]["foot_force_source"] = "crm_fsi"
    return config


def assert_finite_rows(rows: list[dict[str, Any]], allow_nan: set[str]) -> None:
    for row in rows:
        for key, value in row.items():
            if key in allow_nan:
                continue
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite value in {key} at sample {row['sample_index']}")


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
    from nedm.quadruped.dataset import capture_row, csv_field_names, foot_field_names
    from nedm.quadruped.policy import PolicyController
    from nedm.quadruped.robot import Go2Robot
    from nedm.quadruped.terrain import build_crm, build_rigid_ground, measure_leg_reach

    import pychrono as chrono
    import pychrono.vehicle as veh
    fsi = None
    if args.terrain == "crm":
        import pychrono.fsi as fsi

    np.random.seed(args.seed)
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
        build_rigid_ground(chrono, system)
        terrain, coupled = None, []
    else:
        terrain, coupled = build_crm(chrono, fsi, veh, system, robot, args)

    policy = PolicyController(ckpt, cfgs)
    sph_probe = soilprobe.bind_probe(terrain)

    exchange = args.exchange_mult * args.step_size_s
    control_every = max(1, int(round((1.0 / args.control_hz) / exchange)))
    n_steps = int(args.duration_s / exchange)
    base = robot.base()


    terrain_label = config["terrain"]["label"]
    episode_id = f"go2_{terrain_label}_{args.episode_index:03d}"
    scenario_name = episode_id
    scenario_family = f"go2_{terrain_label}_constant_command"
    split = assign_split(episode_id, float(args.validation_ratio))
    csv_path = episodes_dir / f"{episode_id}.csv"

    q0 = robot.joint_pos().astype(np.float64)
    robot.actuate(q0)
    action = q0.copy()

    rows: list[dict[str, Any]] = []
    next_record_s = 0.0
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
                if t < args.pose_ramp_seconds:
                    a = t / max(args.pose_ramp_seconds, 1e-9)
                    action = q0 + a * (STAND_ACTION - q0)
                elif t < args.pose_ramp_seconds + args.settle_seconds:
                    action = STAND_ACTION.astype(np.float64)
                else:
                    action = policy.act(robot)
                robot.actuate(action)
                if sph_probe is not None:
                    soil_z, soil_ctrl = soilprobe.sample(sph_probe, robot)

            robot.apply_pd()          # every physics step, not every control step
            if terrain is not None:
                terrain.DoStepDynamics(exchange)
            else:
                system.DoStepDynamics(exchange)

            if t + 1e-12 >= next_record_s:
                row = capture_row(
                    chrono=chrono,
                    robot=robot,
                    terrain=terrain,
                    soil_top_m=soil_top,
                    action=action,
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
    allow_nan = {f for f in foot_field_names() if f.endswith("_surface_disp_m")}
    assert_finite_rows(rows, allow_nan)

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
        "record_step_s": float(args.record_step_s),
        "terrain_type": args.terrain,
        "terrain_label": terrain_label,
        "foot_force_source": config["logging"]["foot_force_source"],
        "plant": args.actuation,
        "command_constant": True,
        "command_lin_vel": [0.5, 0.0, 0.0],
        "seed": int(args.seed),
        "heading_deg": float(args.heading_deg),
        "spawn_m": [float(args.spawn_x_m), float(args.spawn_y_m), float(spawn_z)],
        "soil_top_m": float(soil_top),
        "robot_mass_kg": float(robot_mass),
        "fsi_coupled_bodies": len(coupled),
        "crm_particles": int(terrain.GetNumSPHParticles()) if terrain is not None else 0,
        "fell": fell_at is not None,
        "fell_at_s": fell_at,
        "force_summary": force_summary,
        "pose_summary": pose_summary,
        "wall_clock_s": round(wall_s, 2),
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
