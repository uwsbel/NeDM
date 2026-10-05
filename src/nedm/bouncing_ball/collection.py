"""Collect passive rigid-sphere launches in Chrono, ending before ground impact 2.

The learned state is [x, z, vx, vz, omega_y] in a Z-up world frame. Launch
velocities initialize the body once; there are no subsequent control inputs.
Normal forces on the two fixed bodies identify real floor and wall contacts.
Contact flags in CSV rows describe the preceding recording interval; states
are instantaneous at time_s. A short lookahead observes ground impact 2, then
the saved trajectory is trimmed to its last nonpenetrating pre-impact sample.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_FIELDS = ["x_m", "z_m", "vx_mps", "vz_mps", "omega_y_radps"]
NUMERIC_FIELDS = [
    "sample_index", "time_s", *STATE_FIELDS, "y_m", "vy_mps",
    "omega_x_radps", "omega_z_radps", "ground_gap_m", "wall_gap_m",
    "ground_contact", "wall_contact", "ground_normal_force_n", "wall_normal_force_n",
    "mechanical_energy_j",
]
CSV_FIELDS = ["episode_id", "split", "phase", *NUMERIC_FIELDS]
REPO_ROOT = Path(__file__).resolve().parents[3]


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    sim, scene, launch = config["simulation"], config["scene"], config["launch"]
    acceptance, collection, contact = config["acceptance"], config["collection"], config["contact"]
    for name, value in [
        ("step_s", sim["step_s"]), ("record_step_s", sim["record_step_s"]),
        ("max_duration_s", sim["max_duration_s"]), ("gravity_mps2", sim["gravity_mps2"]),
        *[(key, scene[key]) for key in ["radius_m", "mass_kg", "ground_length_m", "width_m",
                                     "ground_thickness_m", "wall_height_m", "wall_thickness_m"]],
    ]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    ratio = sim["record_step_s"] / sim["step_s"]
    if ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-9):
        raise ValueError("record_step_s must be an integer multiple of step_s")
    lo, hi = acceptance["duration_window_s"]
    if not 0 < lo <= acceptance["target_duration_s"] <= hi <= sim["max_duration_s"]:
        raise ValueError("duration window must contain the target and fit the simulation limit")
    if scene["launch_z_m"] <= scene["ground_z_m"] + scene["radius_m"]:
        raise ValueError("launch position must be above the ground")
    if scene["launch_x_m"] + scene["radius_m"] >= scene["wall_front_x_m"]:
        raise ValueError("launch position must be in front of the wall")
    for name, bounds in [("vx", launch["vx_range_mps"]), ("vz", launch["vz_range_mps"])]:
        if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] >= bounds[1]:
            raise ValueError(f"{name} range must have two finite increasing bounds")
    if launch["vx_range_mps"][0] <= 0 or launch["vz_range_mps"][1] >= 0:
        raise ValueError("this ground-first launch family requires vx > 0 and vz < 0")
    if (contact["method"] != "NSC" or contact["solver"] != "ADMM"
            or contact["contact_activation"] != "touching_only" or not 0 <= contact["restitution"] <= 1
            or any(contact[key] < 0 for key in ["friction", "rolling_friction_m", "spinning_friction_m"])):
        raise ValueError("use NSC/ADMM, touching-only contacts, restitution in [0, 1], and nonnegative friction")
    if collection["episodes"] < 1 or collection["max_attempts_per_episode"] < 1:
        raise ValueError("episode and attempt counts must be positive")
    if not (0 <= collection["validation_fraction"] < 1 and 0 <= collection["test_fraction"] < 1
            and collection["validation_fraction"] + collection["test_fraction"] < 1):
        raise ValueError("validation/test fractions must leave a nonempty training fraction")
    return config


@dataclass
class ContactSequence:
    """Accept one ground bounce followed by one wall bounce and airborne return."""

    events: list[dict[str, Any]] = field(default_factory=list)
    ground_seen: bool = False
    ground_released: bool = False
    ground_bounced: bool = False
    wall_seen: bool = False
    wall_released: bool = False
    wall_rebounded: bool = False
    previous_ground: bool = False
    previous_wall: bool = False

    @property
    def phase(self) -> str:
        return "wall_rebound" if self.wall_seen else "after_ground" if self.ground_seen else "launch"

    def observe(self, time_s: float, ground: bool, wall: bool, vx: float, vz: float,
                acceptance: dict[str, Any]) -> str | None:
        def event(kind: str) -> None:
            self.events.append({"kind": kind, "time_s": time_s, "vx_mps": vx, "vz_mps": vz})

        if ground and wall:
            event("simultaneous_ground_wall")
            return "simultaneous_ground_wall"
        if ground and not self.previous_ground:
            if self.ground_seen:
                event("next_ground_contact")
                return "next_ground_contact" if self.wall_rebounded else "ground_again_before_wall_rebound"
            event("first_ground_contact")
            self.ground_seen = True
        if not ground and self.previous_ground:
            event("ground_release")
            self.ground_released = True
        if self.ground_seen and vz >= acceptance["min_ground_bounce_speed_mps"]:
            self.ground_bounced = True
        if wall and not self.previous_wall:
            if not self.ground_released or not self.ground_bounced:
                event("wall_before_ground_bounce")
                return "wall_before_ground_bounce"
            if self.wall_seen:
                event("extra_wall_contact")
                return "extra_wall_contact"
            event("wall_contact")
            self.wall_seen = True
        if not wall and self.previous_wall:
            event("wall_release")
            self.wall_released = True
        if self.wall_seen and not self.wall_rebounded and vx <= -acceptance["min_rebound_speed_mps"]:
            event("wall_rebound")
            self.wall_rebounded = True
        self.previous_ground, self.previous_wall = ground, wall
        return None


def build_scene(config: dict[str, Any], vx: float, vz: float, visualize: bool = False):
    import pychrono as chrono

    scene, contact = config["scene"], config["contact"]
    system = chrono.ChSystemNSC()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -config["simulation"]["gravity_mps2"]))
    solver = chrono.ChSolverADMM()
    solver.SetMaxIterations(contact["solver_iterations"])
    solver.SetTolerancePrimal(contact["solver_tolerance"])
    solver.SetToleranceDual(contact["solver_tolerance"])
    system.SetSolver(solver)
    system.SetMinBounceSpeed(contact["min_bounce_speed_mps"])

    class TouchingOnly(chrono.NarrowphaseCallback):
        def __init__(self):
            super().__init__()

        def OnNarrowphase(self, info):
            # Bullet retains nearby separated contacts. Rolling/sliding impulses
            # on those pairs caused premature wall impulses in the pilot. Activate
            # contact only at physical touching/overlap; retain the small overlap
            # introduced by discrete stepping in the penetration diagnostics.
            return bool(info.distance <= 0)

    # Keep the Python director alive as long as the Chrono system uses it.
    system._touching_only_callback = TouchingOnly()
    system.GetCollisionSystem().RegisterNarrowphaseCallback(system._touching_only_callback)
    material = chrono.ChContactMaterialNSC()
    material.SetFriction(contact["friction"])
    material.SetRollingFriction(contact["rolling_friction_m"])
    material.SetSpinningFriction(contact["spinning_friction_m"])
    material.SetRestitution(contact["restitution"])
    floor = chrono.ChBodyEasyBox(scene["ground_length_m"], scene["width_m"],
                                scene["ground_thickness_m"], 1000, visualize, True, material)
    floor.SetPos(chrono.ChVector3d(0, 0, scene["ground_z_m"] - scene["ground_thickness_m"] / 2))
    floor.SetFixed(True)
    floor.SetName("ground")
    wall = chrono.ChBodyEasyBox(scene["wall_thickness_m"], scene["width_m"],
                               scene["wall_height_m"], 1000, visualize, True, material)
    wall.SetPos(chrono.ChVector3d(scene["wall_front_x_m"] + scene["wall_thickness_m"] / 2, 0,
                                 scene["ground_z_m"] + scene["wall_height_m"] / 2))
    wall.SetFixed(True)
    wall.SetName("wall")
    density = scene["mass_kg"] / (4 / 3 * math.pi * scene["radius_m"] ** 3)
    ball = chrono.ChBodyEasySphere(scene["radius_m"], density, visualize, True, material)
    ball.SetName("ball")
    ball.SetPos(chrono.ChVector3d(scene["launch_x_m"], 0, scene["launch_z_m"]))
    ball.SetPosDt(chrono.ChVector3d(vx, 0, vz))
    ball.SetAngVelParent(chrono.ChVector3d(0, config["launch"]["omega_y_radps"], 0))
    for body in [floor, wall, ball]:
        body.GetCollisionModel().SetEnvelope(contact["collision_envelope_m"])
        body.GetCollisionModel().SetSafeMargin(contact["collision_safe_margin_m"])
        system.Add(body)
    if scene["constrain_to_xz_plane"]:
        # Only y translation and x/z rotations are constrained. The five-state
        # model retains all free coordinates: x/z translations and y rotation.
        planar = chrono.ChLinkLockPlanar()
        # The link's local Z axis is its normal; align that normal with world Y.
        frame = chrono.ChFramed(ball.GetPos(), chrono.QuatFromAngleAxis(
            -math.pi / 2, chrono.ChVector3d(1, 0, 0)))
        planar.Initialize(ball, floor, frame)
        planar.SetName("xz_plane")
        system.AddLink(planar)
    return system, ball, floor, wall


def simulate_episode(config: dict[str, Any], vx: float, vz: float,
                     episode_id: str = "probe", split: str = "train") -> tuple[list[dict], dict]:
    sim, scene, acceptance = config["simulation"], config["scene"], config["acceptance"]
    dt = sim["step_s"]
    stride = round(sim["record_step_s"] / dt)
    system, ball, floor, wall = build_scene(config, vx, vz)
    sequence = ContactSequence()
    rows: list[dict] = []
    interval_ground = interval_wall = 0.0
    max_penetration = max_position_y = max_velocity_y = max_angular_out_of_plane = 0.0
    reason = "simulation_timeout"
    start = time.perf_counter()

    def state():
        p, v, w = ball.GetPos(), ball.GetPosDt(), ball.GetAngVelParent()
        return p, v, w

    def record(step: int, p, v, w):
        inertia = 0.4 * scene["mass_kg"] * scene["radius_m"] ** 2
        rows.append(dict(zip(CSV_FIELDS, [
            episode_id, split, sequence.phase, len(rows), step * dt,
            p.x, p.z, v.x, v.z, w.y, p.y, v.y, w.x, w.z,
            p.z - scene["ground_z_m"] - scene["radius_m"],
            scene["wall_front_x_m"] - p.x - scene["radius_m"],
            int(interval_ground > config["contact"]["force_threshold_n"]),
            int(interval_wall > config["contact"]["force_threshold_n"]),
            interval_ground, interval_wall,
            0.5 * scene["mass_kg"] * (v.x*v.x + v.y*v.y + v.z*v.z)
            + 0.5 * inertia * (w.x*w.x + w.y*w.y + w.z*w.z)
            + scene["mass_kg"] * sim["gravity_mps2"] * (p.z - scene["ground_z_m"]),
        ], strict=True)))

    record(0, *state())
    step = 0
    for step in range(1, math.ceil(sim["max_duration_s"] / dt) + 1):
        system.DoStepDynamics(dt)
        p, v, w = state()
        if not all(math.isfinite(x) for x in [p.x, p.y, p.z, v.x, v.y, v.z, w.x, w.y, w.z]):
            reason = "nonfinite_state"
            break
        ground_force, wall_force = abs(floor.GetContactForce().z), abs(wall.GetContactForce().x)
        if not math.isfinite(ground_force + wall_force):
            reason = "nonfinite_contact_force"
            break
        reason_now = sequence.observe(step * dt, ground_force > config["contact"]["force_threshold_n"],
                                      wall_force > config["contact"]["force_threshold_n"], v.x, v.z, acceptance)
        if reason_now == "next_ground_contact":
            # Its impulse is lookahead only. Do not use that excluded collision
            # in the accepted episode's contact or planar-motion diagnostics.
            reason = reason_now
            break
        interval_ground = max(interval_ground, ground_force)
        interval_wall = max(interval_wall, wall_force)
        max_position_y = max(max_position_y, abs(p.y))
        max_velocity_y = max(max_velocity_y, abs(v.y))
        max_angular_out_of_plane = max(max_angular_out_of_plane, abs(w.x), abs(w.z))
        wall_overlap = (max(0.0, p.x + scene["radius_m"] - scene["wall_front_x_m"])
                        if scene["ground_z_m"] <= p.z <= scene["ground_z_m"] + scene["wall_height_m"] else 0.0)
        max_penetration = max(max_penetration, scene["ground_z_m"] + scene["radius_m"] - p.z, wall_overlap)
        if max_penetration > acceptance["max_penetration_m"]:
            reason = "excess_penetration"
            break
        if (max_position_y > acceptance["max_out_of_plane_position_m"]
                or max_velocity_y > acceptance["max_out_of_plane_velocity_mps"]
                or max_angular_out_of_plane > acceptance["max_out_of_plane_angular_velocity_radps"]):
            reason = "out_of_plane_motion"
            break
        if reason_now:
            reason = reason_now
            break
        if step % stride == 0:
            record(step, p, v, w)
            interval_ground = interval_wall = 0.0

    # The final ground impulse belongs to the lookahead, not the saved episode.
    if reason == "next_ground_contact":
        while rows and rows[-1]["ground_gap_m"] <= 0:
            rows.pop()
    duration = rows[-1]["time_s"] if rows else 0.0
    event_times = {event["kind"]: event["time_s"] for event in sequence.events}
    wall_time = event_times.get("wall_contact", duration)
    if reason == "next_ground_contact":
        lo, hi = acceptance["duration_window_s"]
        if not sequence.wall_released:
            reason = "wall_not_released"
        elif not lo <= duration <= hi:
            reason = "duration_out_of_window"
        elif duration - wall_time < acceptance["min_post_wall_duration_s"]:
            reason = "insufficient_post_wall_flight"
        elif rows[-1]["vx_mps"] >= -acceptance["min_rebound_speed_mps"] or rows[-1]["vz_mps"] >= 0:
            reason = "final_state_not_returning_and_descending"
        elif scene["wall_front_x_m"] - scene["radius_m"] - rows[-1]["x_m"] < acceptance["min_return_distance_m"]:
            reason = "insufficient_return_distance"
    accepted = reason == "next_ground_contact"
    metadata = {
        "episode_id": episode_id, "split": split, "accepted": accepted, "termination": reason,
        "scenario_family": "ground_wall_rebound", "rows": len(rows), "duration_s": duration,
        "simulated_until_s": step * dt, "events": sequence.events,
        "launch": {"vx_mps": vx, "vz_mps": vz, "omega_y_radps": config["launch"]["omega_y_radps"]},
        "max_penetration_m": max_penetration, "max_out_of_plane_position_m": max_position_y,
        "max_out_of_plane_velocity_mps": max_velocity_y,
        "max_out_of_plane_angular_velocity_radps": max_angular_out_of_plane,
        "wall_time_s": time.perf_counter() - start,
        "final_state": {key: rows[-1][key] for key in STATE_FIELDS} if rows else None,
    }
    return rows, metadata


def assign_split(seed: int, attempt: int, config: dict) -> str:
    value = int.from_bytes(hashlib.sha256(f"{seed}:{attempt}:split".encode()).digest()[:8], "big") / 2**64
    collection = config["collection"]
    if value < collection["test_fraction"]:
        return "test"
    if value < collection["test_fraction"] + collection["validation_fraction"]:
        return "val"
    return "train"


def provenance() -> dict:
    import pychrono

    versions = []
    for path in (Path(sys.prefix) / "conda-meta").glob("pychrono-*.json"):
        package = json.loads(path.read_text())
        versions.append({key: package.get(key) for key in ["name", "version", "build", "channel"]})
    bundle = {}
    for parent in Path(pychrono.__file__).resolve().parents:
        package_path = parent / "conda-package.json"
        if package_path.exists():
            package = json.loads(package_path.read_text())
            versions.append({key: package.get(key) for key in ["name", "version", "build", "channel"]})
            bundle = {"root": str(parent), "package_sha256": hashlib.sha256(package_path.read_bytes()).hexdigest()}
            if (parent / "manifest.json").exists():
                bundle["manifest_sha256"] = hashlib.sha256((parent / "manifest.json").read_bytes()).hexdigest()
            break
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    return {"python": sys.version, "platform": platform.platform(), "chrono_module": pychrono.__file__,
            "chrono_packages": versions, "chrono_binary_bundle": bundle, "git_revision": revision.stdout.strip() or None,
            "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def collect(config: dict, output: Path, episodes: int, max_attempts: int,
            launches: list[dict] | None = None) -> dict:
    if launches is not None:
        if episodes != len(launches):
            raise ValueError("explicit launch list must contain exactly the requested episodes")
        for launch in launches:
            for axis in ("vx", "vz"):
                value = launch[f"{axis}_mps"]
                lo, hi = config["launch"][f"{axis}_range_mps"]
                if not math.isfinite(value) or not lo <= value <= hi:
                    raise ValueError(f"explicit {axis}={value} is outside [{lo}, {hi}]")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty dataset: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "episodes").mkdir()
    (output / "rejected").mkdir()
    atomic_json(output / "collector_config.resolved.json", config)
    atomic_json(output / "schema.json", {
        "version": "bouncing_ball_chrono_v1", "state_fields": STATE_FIELDS, "action_fields": [],
        "state_units": ["m", "m", "m/s", "m/s", "rad/s"], "coordinate_frame": "world, Z up, motion in X-Z",
        "state_timestamp": "instantaneous at time_s; includes t=0 launch state",
        "contact_timestamp": "maximum normal force over (previous time_s, current time_s]",
        "contact_source": "resultant normal force on each fixed body; only the sphere is dynamic",
        "terminal_policy": "observe ground contact 2 in lookahead, save through last clear pre-impact grid sample",
        "launch_action": "[vx_0, vz_0] applied once at reset; no actions during rollout",
        "target_fields": [f"delta_{key}" for key in STATE_FIELDS], "csv_fields": CSV_FIELDS,
    })
    index = {"dataset_name": config["dataset_name"], "schema_version": "bouncing_ball_chrono_v1",
             "generated_at_utc": datetime.now(timezone.utc).isoformat(), "config": config,
             "provenance": provenance(), "requested_episodes": episodes, "episodes": [], "complete": False}
    if launches is not None:
        index["explicit_launches"] = launches
    attempts, reasons = [], Counter()

    def summarize() -> None:
        accepted = index["episodes"]
        durations = sorted(x["duration_s"] for x in accepted)
        index["episode_count"] = len(accepted)
        index["attempt_count"] = len(attempts)
        index["rejection_counts"] = dict(reasons)
        index["split_counts"] = dict(Counter(x["split"] for x in accepted))
        index["duration_summary_s"] = {"min": durations[0], "mean": sum(durations)/len(durations), "max": durations[-1]} if durations else None
        atomic_json(output / "dataset_index.json", index)

    summarize()
    seed = config["collection"]["seed"]
    try:
        with (output / "attempts.jsonl").open("w") as journal:
            for attempt in range(min(max_attempts, len(launches)) if launches is not None else max_attempts):
                rng = random.Random(f"{seed}:{attempt}:launch")
                vx = launches[attempt]["vx_mps"] if launches is not None else rng.uniform(*config["launch"]["vx_range_mps"])
                vz = launches[attempt]["vz_mps"] if launches is not None else rng.uniform(*config["launch"]["vz_range_mps"])
                episode_id = launches[attempt].get("episode_id", f"ball_{seed}_{attempt:06d}") if launches is not None else f"ball_{seed}_{attempt:06d}"
                split = launches[attempt].get("split", assign_split(seed, attempt, config)) if launches is not None else assign_split(seed, attempt, config)
                if split not in ("train", "val", "test") or not episode_id.replace("_", "").isalnum():
                    raise ValueError("explicit episodes need safe IDs and train/val/test splits")
                rows, result = simulate_episode(config, vx, vz, episode_id, split)
                if launches is not None:
                    result["launch_label"] = launches[attempt].get("name", f"launch_{attempt+1}")
                    if "cell" in launches[attempt]:
                        result["launch_cell"] = launches[attempt]["cell"]
                relative = Path("episodes" if result["accepted"] else "rejected") / f"{episode_id}.csv"
                with (output / relative).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                result["csv_path"] = relative.as_posix()
                atomic_json((output / relative).with_suffix(".json"), result)
                journal.write(json.dumps(result, allow_nan=False) + "\n")
                journal.flush()
                attempts.append(result)
                if result["accepted"]:
                    index["episodes"].append(result)
                else:
                    reasons[result["termination"]] += 1
                summarize()
                if len(attempts) % 20 == 0 or len(index["episodes"]) == episodes:
                    print(f"accepted {len(index['episodes'])}/{episodes}; attempts {len(attempts)}; rejections {dict(reasons)}", flush=True)
                if len(index["episodes"]) == episodes:
                    index["complete"] = True
                    break
    finally:
        summarize()
    if not index["complete"]:
        raise RuntimeError(f"Only {index['episode_count']}/{episodes} accepted after {len(attempts)} attempts; inspect {output}")
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/bouncing_ball/chrono_v1.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts/datasets/bouncing_ball_chrono_v1")
    parser.add_argument("--episodes", type=int, help="Accepted-episode quota; default from config")
    parser.add_argument("--max-attempts", type=int, help="Hard limit including rejected launches")
    parser.add_argument("--launches-file", type=Path, help="JSON list of explicit {name, vx_mps, vz_mps} review launches; all must pass")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    launches = json.loads(args.launches_file.read_text()) if args.launches_file else None
    episodes = args.episodes if args.episodes is not None else len(launches) if launches is not None else config["collection"]["episodes"]
    max_attempts = args.max_attempts if args.max_attempts is not None else episodes * config["collection"]["max_attempts_per_episode"]
    if episodes < 1 or max_attempts < episodes:
        parser.error("episodes must be positive and max-attempts must be at least episodes")
    config["collection"]["episodes"] = episodes
    config["collection"]["max_attempts"] = max_attempts
    index = collect(config, args.output_dir.resolve(), episodes, max_attempts, launches)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "episodes": index["episode_count"],
                      "splits": index["split_counts"], "duration_s": index["duration_summary_s"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
