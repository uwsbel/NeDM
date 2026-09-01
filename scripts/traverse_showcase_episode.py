#!/usr/bin/env python
"""One-episode showcase for the HMMWV traversal arena (plan §14, WP0a).

Samples a layout, proves it feasible with the privileged oracle, drives the
plan with a ChPathFollowerDriver while the fixed overhead RGB + depth cameras
record, and encodes an mp4 so the scene can be eyeballed BEFORE mass data
collection. Plan markers (cyan path dots, magenta goal ring, yellow approach
pose) are showcase-only decorations.

Usage (from repo root, nedm env):
  PYTHONPATH=src python scripts/traverse_showcase_episode.py --probe   # 1 frame
  PYTHONPATH=src python scripts/traverse_showcase_episode.py           # movie
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pychrono as chrono  # noqa: E402
import pychrono.vehicle as veh  # noqa: E402

from nedm.traverse.layout import sample_episode  # noqa: E402
from nedm.traverse.scene import (  # noqa: E402
    RenderSpec,
    build_config,
    build_scene,
    calibrate_orientation,
)
from nedm.traverse.terrain import META_NAME, TerrainMap  # noqa: E402

# Viridis-ish anchor colors for the depth panel (near ground = bright).
_CMAP = np.array(
    [
        [0.267, 0.005, 0.329],
        [0.254, 0.265, 0.530],
        [0.164, 0.471, 0.558],
        [0.128, 0.567, 0.551],
        [0.369, 0.789, 0.383],
        [0.993, 0.906, 0.144],
    ]
)


def colorize_depth(depth: np.ndarray, max_depth_m: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.1) & (depth < 0.98 * max_depth_m)
    out = np.full((*depth.shape, 3), (40, 40, 60), dtype=np.uint8)
    if valid.sum() < 16:
        return out
    lo = float(np.percentile(depth[valid], 2))
    hi = float(np.percentile(depth[valid], 98))
    v = 1.0 - np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)  # near = bright
    stops = np.linspace(0.0, 1.0, len(_CMAP))
    rgb = np.stack([np.interp(v, stops, _CMAP[:, c]) for c in range(3)], axis=-1)
    out[valid] = (rgb[valid] * 255).astype(np.uint8)
    return out


def compose_frame(
    rgb: np.ndarray, depth_rgb: np.ndarray | None, hud_lines: list[str]
) -> Image.Image:
    panels = [Image.fromarray(rgb)]
    if depth_rgb is not None:
        panels.append(Image.fromarray(depth_rgb))
    width = sum(p.width for p in panels)
    canvas = Image.new("RGB", (width, panels[0].height))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(hud_lines):
        draw.text((6, 5 + 13 * i), line, fill=(255, 255, 255))
    draw.text((panels[0].width + 6, 5), "depth", fill=(255, 255, 255))
    return canvas


def nearest_index(waypoints: np.ndarray, pos: np.ndarray, last: int, window: int = 60) -> int:
    lo = last
    hi = min(len(waypoints), last + window)
    d = np.hypot(waypoints[lo:hi, 0] - pos[0], waypoints[lo:hi, 1] - pos[1])
    return lo + int(np.argmin(d))


def cross_track_m(waypoints: np.ndarray, pos: np.ndarray, idx: int) -> float:
    best = math.inf
    for i in range(max(0, idx - 1), min(len(waypoints) - 1, idx + 1)):
        a, b = waypoints[i], waypoints[i + 1]
        ab = b - a
        denom = float(ab @ ab)
        t = 0.0 if denom < 1e-9 else float(np.clip((pos - a) @ ab / denom, 0.0, 1.0))
        best = min(best, float(np.hypot(*(pos - (a + t * ab)))))
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena", default="assets/traverse/arena_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-id", default="showcase_000")
    parser.add_argument("--out", default="artifacts/traverse/showcase_000")
    parser.add_argument("--res", type=int, default=512, help="camera resolution (square)")
    parser.add_argument("--render-hz", type=float, default=20.0)
    parser.add_argument("--cam-height", type=float, default=100.0)
    parser.add_argument("--hfov-deg", type=float, default=47.0)
    parser.add_argument("--timeout-scale", type=float, default=2.5)
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--probe", action="store_true", help="render one settled frame and exit")
    args = parser.parse_args()

    arena_dir = (REPO_ROOT / args.arena).resolve()
    out_dir = (REPO_ROOT / args.out).resolve()
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(exist_ok=True)

    with (arena_dir / META_NAME).open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if not meta.get("orientation", {}).get("calibrated", False):
        result = calibrate_orientation(arena_dir)
        print(f"orientation calibrated: {result}")
    tmap = TerrainMap.from_dir(arena_dir)

    layout, plan = sample_episode(tmap, args.episode_id, args.seed)
    layout.save(out_dir / "layout.json")
    plan.save(out_dir / "plan.json")
    approach = np.asarray(plan.meta["approach_pose"][:2])
    print(
        f"plan: {plan.meta['length_m']:.1f} m, est {plan.meta['est_duration_s']:.1f} s, "
        f"{sum(a.kind == 'rock' for a in layout.assets)} rocks / "
        f"{sum(a.kind == 'tree' for a in layout.assets)} trees, "
        f"house ({layout.house_xy[0]:.1f},{layout.house_xy[1]:.1f})"
    )

    start_z = float(tmap.height(*layout.start_xy)) + 0.75
    config = build_config(arena_dir, (*layout.start_xy, start_z), layout.start_yaw)
    render = RenderSpec(
        width=args.res,
        height=args.res,
        cam_height_m=args.cam_height,
        hfov_rad=math.radians(args.hfov_deg),
        with_depth=not args.no_depth,
        plan_markers=True,
    )
    scene = build_scene(config, layout, tmap, arena_dir, plan=plan, render=render)
    hmmwv, system, terrain = scene.hmmwv, scene.system, scene.terrain
    vehicle = hmmwv.GetVehicle()

    if args.probe:
        scene.manager.Update()
        rgb = scene.rgb_tap.take()
        depth_rgb = (
            colorize_depth(scene.depth_tap.take(), render.max_depth_m)
            if scene.depth_tap is not None
            else None
        )
        compose_frame(rgb, depth_rgb, ["probe t=0"]).save(out_dir / "probe.png")
        print(f"probe frame written: {out_dir / 'probe.png'}")
        return 0

    # Oracle path as a Bezier curve (waypoints subsampled to ~2 m spacing).
    pts = chrono.vector_ChVector3d()
    last_s = -10.0
    for (x, y), s in zip(plan.waypoints, plan.stations):
        if s - last_s < 2.0 and s != plan.stations[-1]:
            continue
        last_s = s
        pts.append(chrono.ChVector3d(float(x), float(y), float(tmap.height(x, y)) + 0.5))
    path = chrono.ChBezierCurve(pts)
    driver = veh.ChPathFollowerDriver(vehicle, path, "oracle_plan", float(plan.speeds[0]))
    driver.GetSteeringController().SetLookAheadDistance(5.0)
    driver.GetSteeringController().SetGains(0.8, 0.0, 0.0)
    driver.GetSpeedController().SetGains(0.6, 0.05, 0.0)
    driver.Initialize()

    dt = float(config["simulation"]["step_size_s"])
    substeps_per_frame = max(1, int(round(1.0 / (args.render_hz * dt))))
    settle_s = 0.8
    timeout_s = settle_s + plan.meta["est_duration_s"] * args.timeout_scale + 8.0
    success_radius = 2.0

    track_rows: list[dict[str, float]] = []
    frame_idx = 0
    wp_idx = 0
    status = "timeout"
    success_time = None
    max_contact_n = 0.0
    step = 0

    while True:
        t = float(system.GetChTime())
        if t >= timeout_s:
            break

        ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
        pos = np.array([ref.GetPos().x, ref.GetPos().y])
        wp_idx = nearest_index(plan.waypoints, pos, wp_idx)
        v_cmd = 0.0 if t < settle_s else float(plan.speeds[wp_idx])
        driver.SetDesiredSpeed(v_cmd)

        driver.Synchronize(t)
        inputs = driver.GetInputs()
        terrain.Synchronize(t)
        hmmwv.Synchronize(t, inputs, terrain)

        dist_goal = float(np.hypot(*(pos - approach)))
        roll, pitch = float(vehicle.GetRoll()), float(vehicle.GetPitch())

        if step % substeps_per_frame == 0:
            contact = max(
                (body.GetContactForce().Length() for _, body in scene.asset_bodies), default=0.0
            )
            max_contact_n = max(max_contact_n, contact)
            speed = float(vehicle.GetSpeed())
            xtrack = cross_track_m(plan.waypoints, pos, wp_idx)
            track_rows.append(
                {
                    "time_s": t,
                    "x_m": pos[0],
                    "y_m": pos[1],
                    "speed_mps": speed,
                    "v_cmd_mps": v_cmd,
                    "cross_track_m": xtrack,
                    "dist_goal_m": dist_goal,
                    "roll_rad": roll,
                    "pitch_rad": pitch,
                    "contact_n": contact,
                    "steering": float(inputs.m_steering),
                    "throttle": float(inputs.m_throttle),
                    "braking": float(inputs.m_braking),
                }
            )
            scene.manager.Update()
            rgb = scene.rgb_tap.take()
            depth_rgb = (
                colorize_depth(scene.depth_tap.take(), render.max_depth_m)
                if scene.depth_tap is not None
                else None
            )
            hud = [
                f"t={t:6.2f}s  v={speed:4.1f}/{v_cmd:4.1f} m/s",
                f"goal={dist_goal:5.1f} m  xtrack={xtrack:4.2f} m",
            ]
            compose_frame(rgb, depth_rgb, hud).save(frames_dir / f"frame_{frame_idx:05d}.png")
            frame_idx += 1

        if abs(roll) > math.radians(60) or abs(pitch) > math.radians(60):
            status = "rollover"
            break
        if success_time is None and dist_goal < success_radius:
            success_time = t
        if success_time is not None and t - success_time > 0.5:
            status = "success"
            break

        driver.Advance(dt)
        terrain.Advance(dt)
        hmmwv.Advance(dt)
        step += 1

    final_t = float(system.GetChTime())
    xtracks = np.array([r["cross_track_m"] for r in track_rows]) if track_rows else np.zeros(1)
    metrics = {
        "status": status,
        "final_time_s": final_t,
        "success_time_s": success_time,
        "frames": frame_idx,
        "plan_length_m": plan.meta["length_m"],
        "plan_est_duration_s": plan.meta["est_duration_s"],
        "cross_track_mean_m": float(xtracks.mean()),
        "cross_track_max_m": float(xtracks.max()),
        "max_asset_contact_n": max_contact_n,
        "final_dist_goal_m": track_rows[-1]["dist_goal_m"] if track_rows else None,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    if track_rows:
        with (out_dir / "track.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(track_rows[0].keys()))
            writer.writeheader()
            writer.writerows(track_rows)

    # 256^2 preview of what the DATA-collection observation will look like.
    if frame_idx > 0:
        mid = frames_dir / f"frame_{frame_idx // 2:05d}.png"
        with Image.open(mid) as img:
            left = img.crop((0, 0, args.res, args.res)).resize((256, 256), Image.BILINEAR)
            left.save(out_dir / "preview_obs_256.png")

    movie_path = out_dir / f"{args.episode_id}.mp4"
    if frame_idx > 0 and shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(args.render_hz),
            "-i", str(frames_dir / "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(movie_path),
        ]
        subprocess.run(cmd, check=True)
        print(f"movie: {movie_path} ({frame_idx} frames @ {args.render_hz:g} fps)")
        if not args.keep_frames:
            for i in range(frame_idx):
                if i % 100 != 0:  # keep a sparse set of stills
                    (frames_dir / f"frame_{i:05d}.png").unlink(missing_ok=True)

    print(
        f"status={status} t={final_t:.1f}s xtrack mean/max="
        f"{metrics['cross_track_mean_m']:.2f}/{metrics['cross_track_max_m']:.2f} m "
        f"contact_max={max_contact_n:.0f} N"
    )
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
