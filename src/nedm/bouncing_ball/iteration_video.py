"""Replay saved NRD optimization launches in Chrono and animate physical traces.

This is a labeled 2D physics-trace animation, not a Sensor/VSG camera recording.
Every unique saved launch is simulated with the approved physical collector.
Interpolation between 0.5 ms physical samples only sets video frame times.
The original NRD optimizer and its saved history are never modified.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from nedm.bouncing_ball.collection import CSV_FIELDS, STATE_FIELDS, atomic_json, provenance, simulate_episode


def replay_one(task):
    index, velocity, config, terminal, grid, output = task
    name = f"iteration_launch_{index:03d}"
    rows, metadata = simulate_episode(config, *velocity, episode_id=name, split="test")
    folder = Path(output)/"traces"
    raw = folder/f"{name}.csv.gz"
    with gzip.open(raw, "wt", newline="", compresslevel=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata["raw_csv_path"] = str(raw.relative_to(output))
    metadata["raw_csv_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    events = {e["kind"]: e["time_s"] for e in metadata["events"]}
    metadata["valid_at_terminal"] = bool(metadata["accepted"] and rows[-1]["time_s"] >= terminal
                                         and events.get("wall_release", math.inf) < terminal
                                         and terminal < events.get("next_ground_contact", 0)-.02)
    atomic_json(raw.with_name(f"{name}.json"), metadata)
    if not metadata["valid_at_terminal"]:
        raise RuntimeError(f"physical replay invalid at {terminal}s: {name}")
    times = np.array([r["time_s"] for r in rows])
    states = np.array([[r[key] for key in STATE_FIELDS] for r in rows])
    sampled = np.stack([np.interp(grid, times, states[:, k]) for k in range(5)], axis=-1)
    energy = [r["mechanical_energy_j"] for r in rows]
    metadata["max_raw_energy_increase_j"] = max(b-a for a, b in zip(energy, energy[1:]))
    return index, sampled, metadata


def collect_history(run, output, history, optimization, fps, workers):
    terminal = optimization["config"]["terminal_time_s"]
    record_step = optimization["physics_config"]["simulation"]["record_step_s"]
    grid = np.arange(round(terminal/record_step)+1)*record_step
    if not math.isclose(grid[-1], terminal, abs_tol=1e-9):
        raise ValueError("terminal time must be on the requested video frame grid")
    velocities, lookup = [], {}
    indices = []
    for row in history:
        mapped = []
        for velocity in row["velocity_mps"]:
            key = tuple(velocity)
            if key not in lookup:
                lookup[key] = len(velocities)
                velocities.append(velocity)
            mapped.append(lookup[key])
        indices.append(mapped)
    (output/"traces").mkdir()
    tasks = [(i, v, optimization["physics_config"], terminal, grid, str(output)) for i, v in enumerate(velocities)]
    states, metadata = [None]*len(tasks), [None]*len(tasks)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(replay_one, task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            i, states[i], metadata[i] = future.result()
            if count % 10 == 0 or count == len(tasks):
                print(f"Chrono replays {count}/{len(tasks)}", flush=True)
    states = np.stack(states)
    indices = np.array(indices)
    endpoints = states[indices, -1, :2]
    targets = np.array([case["xz_m"] for case in optimization["cases"]])
    distances = np.linalg.norm(endpoints-targets[None, :, :], axis=-1)
    verification = json.loads((run/"chrono_verification.json").read_text())
    final_reference = np.array([case["optimized"]["endpoint_m"] for case in verification["cases"]])
    if history[-1]["iteration"] == json.loads((run/"history.json").read_text())[-1]["iteration"]:
        replay_error = float(np.abs(endpoints[-1]-final_reference).max())
        if replay_error > 1e-9:
            raise RuntimeError(f"final physical replays differ from accepted verification: {replay_error}")
    else:
        replay_error = None
    np.savez_compressed(output/"physical_replays.npz", time_s=grid, states=states,
                        step_trace_indices=indices, endpoint_xz_m=endpoints,
                        endpoint_distance_m=distances, iterations=[r["iteration"] for r in history])
    report = {"unique_launches": len(tasks), "shown_steps": [r["iteration"] for r in history],
              "physical_launches_per_step": len(targets), "traces": metadata,
              "final_replay_difference_m": replay_error,
              "all_valid_two_bounce_replays": all(m["valid_at_terminal"] for m in metadata),
              "endpoint_distance_m": distances.tolist(),
              "physical_error_increases_over_1mm": (np.diff(distances, axis=0) > .001).sum(axis=0).tolist(),
              "maximum_penetration_m": max(m["max_penetration_m"] for m in metadata),
              "max_raw_energy_increase_j": max(m["max_raw_energy_increase_j"] for m in metadata)}
    return grid, states, indices, distances, report


def render_video(output, optimization, history, grid, states, indices, distances, physics, fps):
    from PIL import Image, ImageDraw, ImageFont
    import imageio_ffmpeg

    width, height = 1920, 1080
    font_path = next((p for p in ["/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
                                  "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"] if Path(p).exists()), "DejaVuSans.ttf")
    fonts = {size: ImageFont.truetype(font_path, size) for size in (17, 18, 20, 22, 25, 32, 38, 46)}
    blue, orange, red = "#2563a8", "#d97706", "#c62828"
    ink, gray, light = "#172238", "#657084", "#e4e9ef"
    canvas_color = "#f5f7fa"
    count = len(optimization["cases"])
    if count != 5:
        raise ValueError("this review video presents the five-target experiment")
    panel_width, gap, left = 352, 20, 40
    scene_y, scene_height = 280, 420
    scale = min((panel_width-24)/5.8, (scene_height-20)/6.5)
    terminal = optimization["config"]["terminal_time_s"]
    tolerance = optimization["config"]["nrd_tolerance_m"]
    total_iterations = history[-1]["iteration"]
    restart_start = next((r["iteration"] for r in history if r.get("restart_case")), None)
    pixel_paths = None

    def text(draw, xy, label, size=20, color=ink):
        draw.text(xy, label, font=fonts[size], fill=color)

    def world(i, x, z):
        origin_x = left+i*(panel_width+gap)+18
        origin_y = scene_y+scene_height-22
        return (origin_x+(x+.3)*scale, origin_y-z*scale)

    def card(title, lines):
        image = Image.new("RGB", (width, height), canvas_color)
        draw = ImageDraw.Draw(image)
        text(draw, (85, 120), title, 46)
        for n, line in enumerate(lines):
            text(draw, (90, 250+n*85), line, 32)
        text(draw, (90, 990), "2D animation of measured Chrono simulation states", 22, gray)
        return image

    def base_frame(j):
        row = history[j]
        image = Image.new("RGB", (width, height), canvas_color)
        draw = ImageDraw.Draw(image)
        text(draw, (40, 24), "NRD launch optimization — physical Chrono replays", 38)
        phase = "Initial launch" if row["iteration"] == 0 else "Gradient descent through NRD"
        if row["iteration"] == restart_start:
            phase = "Fixed restart for target 5"
        elif row.get("restart_case"):
            phase = "NRD gradient descent after target 5 restart"
        text(draw, (40, 79), f"Saved step {row['iteration']:02d}/{total_iterations:02d}  |  {phase}", 25)
        text(draw, (40, 116), "Target time: 1.7 s    •    Gradients: NRD    •    Displayed motion: Chrono physics", 22, gray)
        for i, case in enumerate(optimization["cases"]):
            x = left+i*(panel_width+gap)
            velocity = row["velocity_mps"][i]
            nrd_distance = row["distance_m"][i]
            target = case["xz_m"]
            text(draw, (x, 160), f"Target {i+1}: ({target[0]:.3f}, {target[1]:.3f}) m", 20)
            text(draw, (x, 192), f"vx {velocity[0]:.4f}  vz {velocity[1]:.4f} m/s", 18)
            if j == 0:
                status = "Common initial velocity"
            elif i == 4 and row["iteration"] == restart_start:
                status = "Fixed restart velocity"
            elif row["velocity_mps"][i] == history[j-1]["velocity_mps"][i]:
                status = "Held: NRD tolerance reached" if nrd_distance <= tolerance else "Held: local progress stalled"
            else:
                status = "Gradient update"
                if nrd_distance <= tolerance:
                    status += " • NRD converged"
            text(draw, (x, 220), status, 17, "#247344" if nrd_distance <= tolerance else gray)
            draw.rectangle((x, scene_y, x+panel_width, scene_y+scene_height), fill="white", outline=light, width=2)
            for tick in range(6):
                a, b = world(i, tick, 0), world(i, tick, 6)
                draw.line((a, b), fill="#eff2f6", width=1)
                text(draw, (a[0]-4, scene_y+scene_height+2), str(tick), 17, gray)
            for tick in (0, 2, 4, 6):
                a, b = world(i, -.3, tick), world(i, 5.3, tick)
                draw.line((a, b), fill="#eff2f6", width=1)
                text(draw, (x+3, a[1]-19), str(tick), 17, gray)
            floor_y = world(i, 0, 0)[1]
            draw.rectangle((x+1, floor_y, x+panel_width-1, scene_y+scene_height-1), fill="#a7adb6")
            wall_x = world(i, 5, 0)[0]
            draw.rectangle((wall_x, scene_y+1, wall_x+.2*scale, floor_y), fill="#47758d")
            text(draw, (x+28, scene_y+8), "z ↑ (m)     x → (m)", 17, gray)
            text(draw, (x, 727), f"Endpoint miss at T: {100*distances[j,i]:.2f} cm", 20, blue)
            text(draw, (x, 758), f"NRD predicted miss: {100*nrd_distance:.2f} cm", 18, orange)
            graph = (x+42, 823, x+panel_width-12, 977)
            draw.rectangle(graph, fill="white", outline=light)
            def point(step, value):
                value = max(.1, min(300., value*100))
                return (graph[0]+step/max(1,total_iterations)*(graph[2]-graph[0]),
                        graph[3]-(math.log10(value)+1)/math.log10(3000)*(graph[3]-graph[1]))
            for tick in (.1, 1., 10., 100.):
                yy = point(0, tick/100)[1]
                draw.line((graph[0], yy, graph[2], yy), fill="#e6ebf1")
                text(draw, (x+2, yy-10), f"{tick:g}", 17, gray)
            for tick in (0, 10, 20, 30):
                xx = point(tick, .001)[0]
                draw.line((xx, graph[1], xx, graph[3]), fill="#eff2f6")
                text(draw, (xx-6, graph[3]+3), str(tick), 17, gray)
            for values, color in (([r["distance_m"][i] for r in history[:j+1]], orange), (distances[:j+1,i], blue)):
                points = [point(r["iteration"], value) for r, value in zip(history[:j+1], values, strict=True)]
                if len(points) > 1:
                    draw.line(points, fill=color, width=3)
                px, py = points[-1]
                draw.ellipse((px-4, py-4, px+4, py+4), fill=color)
            if restart_start is not None and i == 4 and row["iteration"] >= restart_start:
                xx = point(restart_start, .001)[0]
                for yy in range(graph[1], graph[3], 10):
                    draw.line((xx, yy, xx, min(yy+5,graph[3])), fill=gray)
            text(draw, (x+2, 797), "Endpoint miss (cm), log scale", 17, gray)
        text(draw, (40, 1026), "Orange: NRD prediction   |   Blue: measured Chrono replay   |   Red cross: target", 20)
        text(draw, (40, 1055), "2D Chrono trace animation • extra contact-review frames • rolling friction 0.001 m", 17, gray)
        return image

    def motion_frame(base, j, t, held=False):
        image = base.copy()
        draw = ImageDraw.Draw(image)
        t = float(t)
        text(draw, (1260, 79), f"Simulation t = {t:.2f} / {terminal:.2f} s", 25)
        if held:
            text(draw, (1510, 116), "ENDPOINT — review hold", 20, blue)
        for i, case in enumerate(optimization["cases"]):
            trace = states[indices[j,i]]
            n = int(np.searchsorted(grid, t, side="right"))
            px, py = world(i, np.interp(t, grid, trace[:,0]), np.interp(t, grid, trace[:,1]))
            points = pixel_paths[i][:n]+[(px,py)]
            if len(points) > 1:
                draw.line(points, fill="#7ca6c8", width=3)
            radius = optimization["physics_config"]["scene"]["radius_m"]*scale
            draw.ellipse((px-radius,py-radius,px+radius,py+radius), fill=blue, outline="#103b66", width=1)
            tx, ty = world(i, *case["xz_m"])
            draw.line((tx-6,ty-6,tx+6,ty+6), fill=red, width=3)
            draw.line((tx-6,ty+6,tx+6,ty-6), fill=red, width=3)
        return image

    path = output/"all_33_steps_chrono.mp4"
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    log = output/"encode.log"
    frame_count = 0
    snapshot_frames = []
    with log.open("w") as stderr:
        encoder = subprocess.Popen([executable, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
                                    "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
                                    "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "8", "-preset", "fast",
                                    "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
                                   stdin=subprocess.PIPE, stderr=stderr)
        def write(image, repeat=1):
            nonlocal frame_count
            raw = image.tobytes()
            for _ in range(repeat):
                encoder.stdin.write(raw)
                frame_count += 1
        try:
            write(card("Watch launch velocities improve through NRD gradients", [
                f"{len(history)} saved steps • five targets • endpoint time 1.7 s",
                "Each displayed trial is a physical Chrono replay of a saved launch.",
                "NRD supplied the optimization gradients; Chrono verifies the motion.",
                "Target 5 has a labeled fixed restart at saved step 23.",
                "Endpoint errors may fluctuate near convergence."]), 2*fps)
            per_step_frames = []
            for j, row in enumerate(history):
                base = base_frame(j)
                pixel_paths = [[world(i, x, z) for x,z in states[indices[j,i],:,:2]] for i in range(count)]
                times = list(np.arange(round(terminal*fps)+1)/fps)
                for trace_index in indices[j]:
                    times.extend(e["time_s"] for e in physics["traces"][trace_index]["events"]
                                 if e["kind"] in ("first_ground_contact","wall_contact") and e["time_s"] <= terminal)
                times = sorted(set(times))
                for t in times:
                    frame = motion_frame(base, j, t)
                    write(frame)
                end_frame = motion_frame(base, j, terminal, True)
                write(end_frame, round(.6*fps))
                per_step_frames.append(len(times)+round(.6*fps))
                if row["iteration"] in (0, 10, 22, 23, 32):
                    end_frame.save(output/f"step_{row['iteration']:02d}_endpoint.png")
                    snapshot_frames.append(end_frame.copy())
                print(f"Encoded saved step {row['iteration']} ({j+1}/{len(history)})", flush=True)
            write(card("Final physical replay: every target within 4 cm", [
                f"Target {i+1}: {100*distances[-1,i]:.2f} cm physical endpoint error"
                for i in range(count)]), 2*fps)
        finally:
            encoder.stdin.close()
            return_code = encoder.wait()
        if return_code:
            raise RuntimeError(f"encoder failed: inspect {log}")
    decode = subprocess.run([executable,"-hide_banner","-loglevel","error","-i",str(path),
                             "-progress","pipe:1","-nostats","-f","null","-"], capture_output=True, text=True, check=True)
    (output/"decode.log").write_text(decode.stdout+decode.stderr)
    decoded = int([line.split("=",1)[1] for line in decode.stdout.splitlines() if line.startswith("frame=")][-1])
    if decoded != frame_count:
        raise RuntimeError(f"decoded frame count {decoded} differs from encoded {frame_count}")
    sheet = Image.new("RGB", (960, 540*len(snapshot_frames)), "white")
    for j, frame in enumerate(snapshot_frames):
        sheet.paste(frame.resize((960,540)), (0,j*540))
    sheet.save(output/"contact_sheet.jpg", quality=92)
    version = subprocess.run([executable,"-version"],capture_output=True,text=True,check=True).stdout.splitlines()[0]
    return {"video": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "width": width, "height": height, "fps": fps, "frames": frame_count,
            "decoded_frames": decoded, "duration_video_s": frame_count/fps,
            "review_hold_per_step_s": .6, "ffmpeg_version": version,
            "per_step_frames": per_step_frames, "extra_contact_review_frames": True,
            "renderer": "Pillow 2D animation of physical Chrono traces", "font": font_path}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = json.loads((args.run/"history.json").read_text())
    optimization = json.loads((args.run/"optimization.json").read_text())
    full_count = len(history)
    if args.smoke:
        history = [history[0],history[1],history[-1]]
    grid, states, indices, distances, physics = collect_history(args.run, args.output_dir, history,
                                                              optimization, args.fps, args.workers)
    report = {"complete": False, "host": platform.node(), "job_id": os.environ.get("SLURM_JOB_ID"),
              "original_run": str(args.run), "original_history_sha256": hashlib.sha256((args.run/"history.json").read_bytes()).hexdigest(),
              "optimization_sha256": hashlib.sha256((args.run/"optimization.json").read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "full_saved_steps": full_count, "complete_history_shown": not args.smoke,
              "chrono_feedback_used_to_optimize": False, "runtime": provenance(), "physics": physics,
              "cases": [{"name": c["name"], "target_xz_m": c["xz_m"],
                         "initial_physical_distance_m": float(distances[0,i]),
                         "final_physical_distance_m": float(distances[-1,i])} for i,c in enumerate(optimization["cases"])]}
    atomic_json(args.output_dir/"video_report.json", report)
    report["encoding"] = render_video(args.output_dir, optimization, history, grid, states, indices, distances, physics, args.fps)
    report["complete"] = True
    atomic_json(args.output_dir/"video_report.json", report)
    print(json.dumps({"complete": True,"steps":len(history),"unique_launches":physics["unique_launches"],
                      "video":report["encoding"],"physical_error_increases_over_1mm":physics["physical_error_increases_over_1mm"]},indent=2),flush=True)


if __name__ == "__main__":
    main()
