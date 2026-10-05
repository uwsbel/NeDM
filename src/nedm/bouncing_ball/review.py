"""Audit five-state traces and render true Chrono Sensor camera videos.

The camera runs at 120 Hz; 30 FPS encoding gives quarter-speed playback.
Rendering re-simulates each saved launch and checks every recorded physical
state against its CSV. A white visual marker follows the sphere's rotation;
it has no collision shape. No states are clamped, interpolated, or replayed
kinematically to make the video.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nedm.bouncing_ball.collection import STATE_FIELDS, atomic_json, build_scene, simulate_episode
from nedm.bouncing_ball.validation import validate_dataset

CAMERA_POSITION = [1.5, -15, 3]
CAMERA_HFOV = 2 * math.atan(60 * math.tan(0.195 / 2) / 15)


def read_rows(root: Path, episode: dict) -> list[dict]:
    with (root / episode["csv_path"]).open(newline="") as stream:
        return [{key: float(value) for key, value in row.items()
                 if key not in ("episode_id", "split", "phase")} for row in csv.DictReader(stream)]


def audit_physics(root: Path, index: dict) -> dict:
    """Compare the pilot against a halved physics timestep and more iterations."""
    results = []
    limits = {"position_difference_m": 0.01, "velocity_difference_mps": 0.02,
              "spin_difference_radps": 0.1, "event_difference_s": 0.001,
              "energy_increase_j": 0.001, "normal_rebound_ratio": 1.01}
    for episode in index["episodes"]:
        rows = read_rows(root, episode)
        energies = [row["mechanical_energy_j"] for row in rows]
        maximum_energy_increase = max(b - a for a, b in zip(energies, energies[1:]))
        events = {event["kind"]: event for event in episode["events"]}
        ground = events["first_ground_contact"]
        wall = events["wall_contact"]
        incoming_ground_speed = abs(episode["launch"]["vz_mps"]) + index["config"]["simulation"]["gravity_mps2"] * ground["time_s"]
        ratios = {"ground": ground["vz_mps"] / incoming_ground_speed,
                  "wall": -wall["vx_mps"] / ground["vx_mps"]}
        comparisons = []
        for label in ("half_timestep", "double_solver_iterations"):
            config = copy.deepcopy(index["config"])
            if label == "half_timestep":
                config["simulation"]["step_s"] /= 2
            else:
                config["contact"]["solver_iterations"] *= 2
            reference, metadata = simulate_episode(config, episode["launch"]["vx_mps"], episode["launch"]["vz_mps"])
            event_delta = max(abs(events[event["kind"]]["time_s"] - event["time_s"])
                              for event in metadata["events"] if event["kind"] in events)
            # Compare the same timestamps, omitting the tiny interval in which
            # a one-step impulse can straddle different sample times.
            paired = [(a, b) for a, b in zip(rows, reference)
                      if all(abs(a["time_s"] - event["time_s"]) > 0.002 for event in events.values())]
            position = max(max(abs(a[k] - b[k]) for k in ("x_m", "z_m")) for a, b in paired)
            velocity = max(max(abs(a[k] - b[k]) for k in ("vx_mps", "vz_mps")) for a, b in paired)
            spin = max(abs(a["omega_y_radps"] - b["omega_y_radps"]) for a, b in paired)
            passed = (metadata["accepted"] and position <= limits["position_difference_m"]
                      and velocity <= limits["velocity_difference_mps"] and spin <= limits["spin_difference_radps"]
                      and event_delta <= limits["event_difference_s"])
            comparisons.append({"variant": label, "passed": passed, "accepted": metadata["accepted"],
                                "termination": metadata["termination"], "position_difference_m": position,
                                "velocity_difference_mps": velocity, "spin_difference_radps": spin,
                                "event_difference_s": event_delta})
        passed = (all(item["passed"] for item in comparisons)
                  and maximum_energy_increase <= limits["energy_increase_j"]
                  and all(0 < ratio <= limits["normal_rebound_ratio"] for ratio in ratios.values()))
        results.append({"episode_id": episode["episode_id"], "launch_label": episode["launch_label"],
                        "passed": passed, "normal_speed_ratios": ratios,
                        "max_energy_increase_j": maximum_energy_increase,
                        "initial_energy_j": energies[0], "final_energy_j": energies[-1],
                        "max_penetration_m": episode["max_penetration_m"], "comparisons": comparisons})
    return {"passed": all(result["passed"] for result in results), "limits": limits, "episodes": results,
            "note": "NSC friction couples normal and tangential response; configured restitution is not an exact measured speed ratio. These are numerical pilot checks, not real-material calibration."}


def sensor_scene(config: dict, episode: dict, width: int, height: int, sensor_fps: int):
    import pychrono as chrono
    import pychrono.sensor as sensor

    system, ball, floor, wall = build_scene(config, episode["launch"]["vx_mps"],
                                          episode["launch"]["vz_mps"], visualize=True)

    def material(color):
        result = chrono.ChVisualMaterial()
        result.SetDiffuseColor(chrono.ChColor(*color))
        result.SetRoughness(0.65)
        return result

    for body, color in [(ball, (1.0, 0.25, 0.015)), (floor, (0.38, 0.42, 0.48)), (wall, (0.1, 0.4, 0.65))]:
        body.GetVisualShape(0).SetMaterial(0, material(color))
    marker = chrono.ChVisualShapeSphere(0.014)
    marker.AddMaterial(material((0.95, 0.95, 0.95)))
    ball.AddVisualShape(marker, chrono.ChFramed(chrono.ChVector3d(0.073, -0.069, 0),
                                               chrono.ChQuaterniond(1, 0, 0, 0)))
    anchor = chrono.ChBody()
    anchor.SetFixed(True)
    anchor.SetName("review_camera_anchor")
    system.Add(anchor)
    manager = sensor.ChSensorManager(system)
    manager.scene.AddPointLight(chrono.ChVector3f(0, -6, 9), chrono.ChColor(1, 1, 1), 100)
    manager.scene.SetAmbientLight(chrono.ChVector3f(0.3, 0.3, 0.3))
    background = sensor.Background()
    background.mode = sensor.BackgroundMode_SOLID_COLOR
    background.color_zenith = chrono.ChVector3f(0.065, 0.085, 0.12)
    manager.scene.SetBackground(background)
    camera = sensor.ChCameraSensor(anchor, sensor_fps, chrono.ChFramed(
        chrono.ChVector3d(*CAMERA_POSITION),
        chrono.QuatFromAngleAxis(math.pi / 2, chrono.ChVector3d(0, 0, 1))), width, height, CAMERA_HFOV, 2)
    camera.SetLag(0)
    camera.SetCollectionWindow(0)
    camera.PushFilter(sensor.ChFilterRGBA8Access())
    manager.AddSensor(camera)
    return system, ball, manager, camera


def render_episode(root: Path, output: Path, index: dict, episode: dict,
                   width: int, height: int, sensor_fps: int, video_fps: int) -> dict:
    rows = read_rows(root, episode)
    system, ball, manager, camera = sensor_scene(index["config"], episode, width, height, sensor_fps)
    dt = index["config"]["simulation"]["step_s"]
    stride = round(index["config"]["simulation"]["record_step_s"] / dt)
    end_step = round(episode["duration_s"] / dt)
    label = episode["launch_label"]
    path = output / f"{label}.mp4"
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", round(width / 53))
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", round(width / 65))
    header, footer = 100, 60
    frames, timestamps, launch_counts, snapshots = 0, [], [], []
    max_replay_error = {key: 0.0 for key in STATE_FIELDS}
    event_times = {event["kind"]: event["time_s"] for event in episode["events"]}
    snapshot_targets = [0.0, event_times["first_ground_contact"],
                        max(rows, key=lambda row: row["z_m"])["time_s"],
                        event_times["wall_contact"], episode["duration_s"] - 1 / sensor_fps]
    snapshot_images: dict[int, Image.Image] = {}
    snapshot_distances = {i: math.inf for i in range(len(snapshot_targets))}
    stderr_path = output / f"{label}.ffmpeg.log"
    with stderr_path.open("w") as log:
        encoder = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height+header+footer}", "-r", str(video_fps), "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE, stderr=log)
        last_launch_count = 0
        try:
            for step in range(end_step + 1):
                if step:
                    system.DoStepDynamics(dt)
                if step % stride == 0:
                    row = rows[step // stride]
                    p, v, w = ball.GetPos(), ball.GetPosDt(), ball.GetAngVelParent()
                    for key, actual in zip(STATE_FIELDS, [p.x, p.z, v.x, v.z, w.y], strict=True):
                        max_replay_error[key] = max(max_replay_error[key], abs(actual - row[key]))
                manager.Update()
                buffer = camera.GetMostRecentRGBA8Buffer()
                if not buffer.HasData() or buffer.LaunchedCount == last_launch_count:
                    continue
                last_launch_count = buffer.LaunchedCount
                timestamp = float(buffer.TimeStamp)
                if timestamp > episode["duration_s"] + dt:
                    raise RuntimeError("camera frame exceeds the saved pre-impact horizon")
                pixels = np.asarray(buffer.GetRGBA8Data())[::-1, :, :3].copy()
                frame = Image.new("RGB", (width, height + header + footer), (13, 19, 29))
                frame.paste(Image.fromarray(pixels), (0, header))
                draw = ImageDraw.Draw(frame)
                draw.text((24, 14), f"{label.replace('_', ' ').title()}    vx={episode['launch']['vx_mps']:.2f} m/s    vz={episode['launch']['vz_mps']:.2f} m/s", font=font, fill="white")
                draw.text((24, 60), f"Chrono Sensor    {video_fps/sensor_fps:g}x speed    t={timestamp:.3f} / {episode['duration_s']:.3f} s", font=small, fill=(180, 202, 220))
                phase = "launch" if timestamp < event_times["first_ground_contact"] else "after ground bounce" if timestamp < event_times["wall_contact"] else "wall rebound / airborne return"
                draw.text((24, height + header + 15), f"{phase}    |    rolling friction: {index['config']['contact']['rolling_friction_m']:g} m    |    white dot shows spin", font=small, fill="white")
                encoder.stdin.write(frame.tobytes())
                if frames == 0:
                    for _ in range(video_fps):
                        encoder.stdin.write(frame.tobytes())
                frames += 1
                timestamps.append(timestamp)
                launch_counts.append(int(buffer.LaunchedCount))
                for i, target in enumerate(snapshot_targets):
                    distance = abs(timestamp - target)
                    if distance < snapshot_distances[i]:
                        snapshot_distances[i] = distance
                        snapshot_images[i] = frame.copy()
                last_frame = frame
            if frames < math.floor(episode["duration_s"] * sensor_fps) - 1:
                raise RuntimeError(f"camera dropped frames: {frames}")
            if any(b != a + 1 for a, b in zip(launch_counts, launch_counts[1:])):
                raise RuntimeError("nonconsecutive camera launch counts")
            if max(max_replay_error.values()) > 1e-9:
                raise RuntimeError(f"visual re-simulation differs from saved physics: {max_replay_error}")
            for _ in range(video_fps):
                encoder.stdin.write(last_frame.tobytes())
        finally:
            encoder.stdin.close()
            return_code = encoder.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed: inspect {stderr_path}")
    for i, frame in snapshot_images.items():
        snapshot = output / f"{label}_{i}.png"
        frame.save(snapshot)
        snapshots.append(str(snapshot))
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate,duration", "-of", "json", str(path)],
                           capture_output=True, text=True, check=True)
    probe_result = json.loads(probe.stdout)
    if int(probe_result["streams"][0]["nb_read_frames"]) != frames + 2 * video_fps:
        raise RuntimeError("encoded frame count does not match the Sensor frames plus review holds")
    return {"episode_id": episode["episode_id"], "launch_label": label, "video": str(path),
            "video_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sensor_frames": frames,
            "frame_timestamps_s": timestamps, "snapshot_files": snapshots,
            "playback_speed": video_fps / sensor_fps, "start_hold_video_s": 1, "end_hold_video_s": 1,
            "max_physical_replay_error": max_replay_error, "ffprobe": probe_result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--sensor-fps", type=int, default=120)
    parser.add_argument("--video-fps", type=int, default=30)
    args = parser.parse_args(argv)
    root, output = args.dataset.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty review directory: {output}")
    if min(args.width, args.height, args.sensor_fps, args.video_fps) <= 0 or args.width % 2 or args.height % 2:
        parser.error("dimensions must be positive even integers and frame rates must be positive")
    validation = validate_dataset(root)
    if not validation["passed"]:
        raise RuntimeError(f"dataset validation failed: {validation['errors']}")
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads((root / "dataset_index.json").read_text())
    report = {"dataset": str(root), "validation": validation, "physics": audit_physics(root, index),
              "renderer": "Chrono Sensor / OptiX", "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "camera": {"position_m": CAMERA_POSITION, "horizontal_fov_rad": CAMERA_HFOV, "supersample_factor": 2},
              "videos": [], "complete": False}
    atomic_json(output / "review_report.json", report)
    if not report["physics"]["passed"]:
        raise RuntimeError(f"physics review failed: inspect {output / 'review_report.json'}")
    for episode in index["episodes"]:
        video = render_episode(root, output, index, episode, args.width, args.height, args.sensor_fps, args.video_fps)
        report["videos"].append(video)
        atomic_json(output / "review_report.json", report)
        print(f"Rendered {episode['launch_label']}: {video['sensor_frames']} Sensor frames; replay error {max(video['max_physical_replay_error'].values()):.2g}", flush=True)
    report["complete"] = True
    atomic_json(output / "review_report.json", report)
    print(f"Review complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
