#!/usr/bin/env python3
"""Scientific prediction overview and actual-frame Chrono video comparison.

No scene synthesis or simulator access. Inputs are frozen predictions, current
RGB-D, authored references, measured telemetry, and actual rendered PNG frames.
The manifest's four labels are supplied by the selection protocol, never chosen
from the measured outcomes by this compositor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from nedm.traverse.fdm_rgbd_data import rgbd_from_arrays
from nedm.traverse.fdm_bounded_targets import bounded_motion_endpoints

COLORS = ["#24a58a", "#e55455", "#e7a432", "#9362d5"]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def probability_text(value):
    percent = value*100.
    if percent < .001:
        return "<0.001%"
    return f"{percent:.4f}%" if percent < 1. else f"{percent:.2f}%"


def display_sentence(text):
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    return re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)


def resolve(base, value):
    p = Path(value)
    return (p if p.is_absolute() else base/p).resolve()


def world_xy(local_xy, anchor):
    c, s = np.cos(anchor[2]), np.sin(anchor[2])
    q = np.asarray(local_xy)
    return np.stack((c*q[..., 0]-s*q[..., 1], s*q[..., 0]+c*q[..., 1]), -1)+anchor[:2]


class CameraProjection:
    """Perspective RGB projection using only the current observed depth surface.

    Forecasts provide XY, not elevation. Three fixed-point depth lookups infer
    the visible surface height. Unknown/out-of-view depth uses elevation zero;
    this is stated in the overview, not presented as true future 3D position.
    """
    def __init__(self, rgb, depth, camera):
        self.rgb, self.camera = rgb, camera
        self.height, self.width = rgb.shape[:2]
        hfov = float(camera.get("hfov_rad", math.radians(camera.get("hfov_deg", 47.))))
        self.focal = self.width/2/math.tan(hfov/2)
        self.camera_z = float(camera["cam_height_m"])
        self.observed = rgbd_from_arrays(rgb, depth_m=depth, camera=camera)[3]

    def project(self, xy):
        xy = np.asarray(xy, float)
        z = np.zeros(xy.shape[:-1])
        for _ in range(3):
            distance = np.maximum(self.camera_z-z, 1.)
            u = (self.width-1)/2+self.focal*xy[..., 0]/distance
            v = (self.height-1)/2-self.focal*xy[..., 1]/distance
            # Registered depth features use box-resized 128px RGB coordinates.
            iu = np.rint((u+.5)*128/self.width-.5).astype(int)
            iv = np.rint((v+.5)*128/self.height-.5).astype(int)
            inside = (iu >= 0) & (iu < 128) & (iv >= 0) & (iv < 128)
            observed = self.observed[iv.clip(0, 127), iu.clip(0, 127)]
            z = np.where(inside & (observed > -1.5), observed*10., 0.)
        return np.stack(((self.width-1)/2+self.focal*xy[..., 0]/np.maximum(self.camera_z-z, 1.),
                         (self.height-1)/2-self.focal*xy[..., 1]/np.maximum(self.camera_z-z, 1.)), -1)


def load_inputs(manifest_path):
    m = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    observation_path = resolve(base, m["observation_npz"])
    with np.load(observation_path, allow_pickle=False) as f:
        obs = {k: f[k].copy() for k in f.files}
    if "camera" in m:
        camera = m["camera"]
    elif "camera_json" in obs:
        camera = json.loads(str(obs["camera_json"].item()))
    else:
        camera = json.loads(resolve(base, m.get("observation_json", str(observation_path.with_suffix(".json")))).read_text())["camera"]
    if "depth_m" not in obs or "rgb" not in obs:
        raise ValueError("Require actual current RGB and metric ray depth")
    anchor = np.asarray(m.get("anchor_pose", obs.get("pose")), float)
    goal = np.asarray(m.get("goal_xy", obs.get("goal_xy")), float)
    if anchor.shape != (3,) or goal.shape != (2,):
        raise ValueError("Expected current XY/yaw and supplied XY goal")
    prediction_path = resolve(base, m["predictions_npz"])
    with np.load(prediction_path, allow_pickle=False) as f:
        predictions = {k: f[k].copy() for k in f.files}
    diagnostics_path = resolve(base, m.get("diagnostics_json", str(prediction_path.with_suffix(".json"))))
    diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else {}
    candidate_records = diagnostics.get("candidates", [])
    if "forecast_world_xy" in predictions:
        trajectory_world = predictions["forecast_world_xy"][:, 1:]
        probability = np.stack([predictions[k+"_probability_by_prefix"] for k in ("contact", "rollover", "progress_event")], -1)
        probability = np.nan_to_num(probability, nan=0.)
        candidate_ids = predictions["candidate_ids"].astype(str).tolist()
        dt = float(np.diff(predictions["forecast_trace_times_s"])[0])
    else:
        trajectory = predictions["trajectory"]
        probability = predictions["event_probability"]
        if trajectory.ndim != 3 or trajectory.shape[-1] != 4 or probability.shape != (*trajectory.shape[:2], 3):
            raise ValueError("Expected candidate forecasts[C,T,4] and event probabilities[C,T,3]")
        trajectory_world = world_xy(trajectory[..., :2], anchor)
        candidate_ids = [str(i) for i in range(len(trajectory))]
        dt = float(m.get("prediction_dt_s", .2))
    routes = []
    if len(m["routes"]) != 4:
        raise ValueError("The frozen selection must contain exactly four routes")
    for i, r in enumerate(m["routes"]):
        r = dict(r)
        r.setdefault("id", str(i+1))
        r.setdefault("display_id", str(i+1))
        r.setdefault("color", COLORS[i])
        r.setdefault("label", r.get("category", f"Route {i+1}"))
        index = int(r["prediction_index"]) if "prediction_index" in r else candidate_ids.index(str(r["candidate_id"]))
        if "reference_json" in r:
            r["reference_path"] = resolve(base, r["reference_json"])
            reference = json.loads(r["reference_path"].read_text())
            r["reference"] = reference.get("route", reference)
        else:
            r["reference_path"] = diagnostics_path
            r["reference"] = candidate_records[index]["route"]
        r["actual_path"] = resolve(base, r["actual_dir"]) if r.get("actual_dir") else None
        r["forecast_xy"] = trajectory_world[index]
        r["progress_m"] = float(np.linalg.norm(anchor[:2]-goal)-np.linalg.norm(r["forecast_xy"][-1]-goal))
        r["contact_score"] = float(probability[index, :, 0].max())
        eligible = (np.arange(trajectory_world.shape[1])+1)*dt >= 2.-1e-6
        r["blockage_score"] = float(probability[index, eligible, 2].max()) if eligible.any() else 0.
        routes.append(r)
    projection = CameraProjection(obs["rgb"], obs["depth_m"], camera)
    return dict(manifest=m, base=base, observation=obs, camera=camera, anchor=anchor, goal=goal,
                predictions=predictions, routes=routes, projection=projection,
                forecast_world_xy=trajectory_world, horizon_s=dt*trajectory_world.shape[1],
                input_hashes={"manifest": sha(manifest_path), "observation": sha(observation_path), "predictions": sha(prediction_path),
                    **{f"reference_{r['id']}": sha(r["reference_path"]) for r in routes}})


def remaining_reference(r):
    reference = np.asarray(r["reference"]["waypoints"], float)
    start = int(np.argmin(np.linalg.norm(reference-r["forecast_xy"][-1], axis=1)))
    return reference[start:]


def overview(data, out):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig = plt.figure(figsize=(16, 10), facecolor="#f4f6f8")
    ax = fig.add_axes([.045, .355, .54, .52])
    proj, rgb = data["projection"], data["observation"]["rgb"]
    ax.imshow(rgb, origin="upper")
    for trace in data["forecast_world_xy"]:
        xy = proj.project(trace)
        ax.plot(xy[:, 0], xy[:, 1], color="#f4f4f4", alpha=.15, lw=.65, zorder=2)
    for r in data["routes"]:
        rest = proj.project(remaining_reference(r))
        if len(rest):
            ax.plot(rest[:, 0], rest[:, 1], ls=(0, (5, 4)), lw=2.2, color="black", alpha=.6, zorder=3)
            ax.plot(rest[:, 0], rest[:, 1], ls=(0, (5, 4)), lw=1.6, color=r["color"], alpha=.95, zorder=4)
        xy = proj.project(np.vstack((data["anchor"][:2], r["forecast_xy"])))
        ax.plot(xy[:, 0], xy[:, 1], lw=4.2, color="black", alpha=.75, zorder=5)
        ax.plot(xy[:, 0], xy[:, 1], lw=2.8, color=r["color"], zorder=6)
        ax.scatter(*xy[-1], s=100, color=r["color"], edgecolors="white", linewidth=1.4, zorder=7)
        ax.annotate(str(r["display_id"]), xy[-1], xytext=(8, -6), textcoords="offset points", color="white", fontsize=12,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=.22", facecolor=r["color"], edgecolor="white"), zorder=8)
    start, goal = proj.project(np.stack((data["anchor"][:2], data["goal"])))
    ax.scatter(*start, s=90, facecolors="white", edgecolors="black", linewidth=1.5, zorder=9)
    ax.annotate("START", start, xytext=(-9, 12), textcoords="offset points", ha="right", color="white", fontsize=11,
                bbox=dict(facecolor="#111827", alpha=.85, edgecolor="none", pad=3), zorder=10)
    ax.scatter(*goal, marker="*", s=320, color="white", edgecolors="black", linewidth=1.4, zorder=9)
    ax.annotate("GOAL", goal, xytext=(9, -12), textcoords="offset points", color="white", fontsize=11,
                bbox=dict(facecolor="#111827", alpha=.85, edgecolor="none", pad=3), zorder=10)
    for i, annotation in enumerate(data["manifest"].get("scene_annotations", [])):
        xy = proj.project(np.asarray(annotation["xy"], float))
        ax.annotate(annotation["label"], xy, xytext=(30, 35 if i%2 == 0 else -37), textcoords="offset points",
            fontsize=10, color="#243146", bbox=dict(facecolor="white", alpha=.9, edgecolor="#758399", boxstyle="round,pad=.25"),
            arrowprops=dict(arrowstyle="->", color="#243146", linewidth=1.2), zorder=11)
    extent_points = np.concatenate([proj.project(np.asarray(r["reference"]["waypoints"], float)) for r in data["routes"]])
    lower = np.maximum(extent_points.min(0)-17., [-.5, -.5])
    upper = np.minimum(extent_points.max(0)+17., [rgb.shape[1]-.5, rgb.shape[0]-.5])
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(upper[1], lower[1])
    ax.set_xlabel("Camera image x (pixels)")
    ax.set_ylabel("Camera image y (pixels)")
    ax.set_title(f"Current RGB view (cropped) | {len(data['forecast_world_xy'])} candidate forecasts", loc="left", fontsize=12, pad=8)
    fig.text(.045, .952, "One scene. Four candidate routes.", fontsize=24, weight="bold", color="#142136")
    fig.text(.045, .921, "Frozen model forecasts before Chrono execution", fontsize=13, color="#526174")
    for i, r in enumerate(data["routes"]):
        bottom = .715-i*.172
        panel = fig.add_axes([.635, bottom, .325, .15])
        panel.axis("off")
        panel.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=.018", fc="white", ec="#dce2e9", transform=panel.transAxes, clip_on=False))
        panel.text(.035, .85, f"{r['display_id']}  {r['label']}", weight="bold", color=r["color"], fontsize=15)
        panel.text(.035, .63, f"Predicted goal progress at {data['horizon_s']:g}s", color="#5b6779", fontsize=11)
        panel.text(.94, .63, f"{r['progress_m']:.2f} m", ha="right", fontsize=15, weight="bold", color="#142136")
        panel.text(.035, .44, f"P(contact <= {data['horizon_s']:g}s)", color="#5b6779", fontsize=10)
        panel.text(.94, .44, probability_text(r['contact_score']), ha="right", fontsize=11, color="#142136")
        panel.text(.035, .27, f"P(confirmed stop <= {data['horizon_s']:g}s)", color="#5b6779", fontsize=10)
        panel.text(.94, .27, probability_text(r['blockage_score']), ha="right", fontsize=11, color="#142136")
        if r.get("prediction_caveat"):
            panel.text(.035, .005, textwrap.fill(display_sentence(r["prediction_caveat"]), width=68), color="#9a431c", fontsize=8.2, va="bottom")
    depth_ax = fig.add_axes([.045, .14, .22, .16])
    height_map = np.where(proj.observed > -1.5, proj.observed*10., np.nan)
    depth_image = depth_ax.imshow(height_map, origin="upper", extent=(-.5, rgb.shape[1]-.5, rgb.shape[0]-.5, -.5),
        cmap="viridis", vmin=min(0., float(np.nanmin(height_map))), vmax=max(1., float(np.nanmax(height_map))))
    depth_ax.set_xlim(lower[0], upper[0]); depth_ax.set_ylim(upper[1], lower[1])
    depth_ax.set_xticks([]); depth_ax.set_yticks([])
    depth_ax.set_title("Current observed depth as height", loc="left", fontsize=10)
    cb = fig.colorbar(depth_image, ax=depth_ax, fraction=.045, pad=.03)
    cb.set_label("m", fontsize=9); cb.ax.tick_params(labelsize=8)
    handles = [Line2D([], [], color="#34445a", lw=2.8, label=f"Solid: learned {data['horizon_s']:g}s motion forecast"),
               Line2D([], [], color="#34445a", lw=2, ls="--", label="Dashed: remaining commanded reference\n(not predicted motion)")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.30, .195), frameon=False, fontsize=10)
    fig.text(.31, .155, "Faint curves: all scored candidate forecasts.\nRGB and depth are the same current observation.", fontsize=9.3, color="#526174")
    if data["manifest"].get("scene_annotations"):
        fig.text(.31, .11, "Feature names use scene metadata for annotation only;\nthey never enter the model or candidate scoring.", fontsize=9, color="#526174")
    fig.text(.045, .037, "XY traces use perspective camera projection with current observed depth; unknown surface height uses z=0. Overlays are not future images.", fontsize=9.5, color="#526174")
    fig.text(.045, .018, "Scores are uncalibrated model outputs. Route labels are test hypotheses, not predicted or measured event claims; actual outcomes appear in the videos.", fontsize=9.5, color="#526174")
    for extension in ("png", "pdf"):
        fig.savefig(out/f"prediction_overview.{extension}", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def dashed(draw, points, fill, width=2, dash=7):
    travelled = 0.
    for a, b in zip(points[:-1], points[1:]):
        a, b = np.asarray(a), np.asarray(b)
        length = np.linalg.norm(b-a)
        if length < 1e-6:
            continue
        position = 0.
        while position < length-1e-8:
            phase = (travelled+position) % (dash*2)
            visible = phase < dash
            step = min(length-position, (dash if visible else dash*2)-phase)
            if step < 1e-8:
                step = min(length-position, 1e-7)
            if visible:
                draw.line([tuple(a+(b-a)*position/length), tuple(a+(b-a)*(position+step)/length)], fill=fill, width=width)
            position += step
        travelled += length


def forecast_inset(data, route, size=352):
    image = Image.fromarray(data["observation"]["rgb"]).resize((size, size), Image.Resampling.LANCZOS).convert("RGB")
    draw = ImageDraw.Draw(image)
    scale = np.array([size/data["projection"].width, size/data["projection"].height])
    rest = data["projection"].project(remaining_reference(route))*scale
    dashed(draw, rest, route["color"], 3)
    points = data["projection"].project(np.vstack((data["anchor"][:2], route["forecast_xy"])))*scale
    draw.line([tuple(x) for x in points], fill="black", width=7)
    draw.line([tuple(x) for x in points], fill=route["color"], width=4)
    for xy, fill, radius in [(points[0], "white", 5), (points[-1], route["color"], 6), (data["projection"].project(data["goal"])*scale, "white", 6)]:
        draw.ellipse([*tuple(xy-radius), *tuple(xy+radius)], fill=fill, outline="black", width=2)
    return image


def actual_trace_inset(initial, data, measured_poses, current_frame):
    image = initial.copy()
    draw = ImageDraw.Draw(image)
    scale = np.array([image.width/data["projection"].width, image.height/data["projection"].height])
    xy = data["projection"].project(measured_poses[:current_frame+1, :2])*scale
    if len(xy) > 1:
        draw.line([tuple(p) for p in xy], fill="black", width=5)
        draw.line([tuple(p) for p in xy], fill="#67e8f9", width=2)
    point = xy[-1]
    draw.ellipse([*tuple(point-5), *tuple(point+5)], fill="#67e8f9", outline="black", width=2)
    goal = data["projection"].project(data["goal"])*scale
    star = []
    for i in range(10):
        radius = 8 if i%2 == 0 else 3.5
        angle = -math.pi/2+i*math.pi/5
        star.append((goal[0]+radius*math.cos(angle), goal[1]+radius*math.sin(angle)))
    draw.polygon(star, fill="white", outline="black")
    return image


def load_actual(path, goal, radius):
    with np.load(path/"trajectory.npz", allow_pickle=False) as f:
        d = {k: f[k].copy() for k in f.files}
    n = len(d["action"])
    d["poses"] = np.concatenate((d["pose"], d["terminal_pose"][None]))
    d["states"] = np.concatenate((d["state"], d.get("terminal_state", d["state"][-1])[None]))
    d["parked_full"] = np.r_[d.get("parked", np.zeros(n, bool)), bool(d.get("terminal_parked", False))]
    d["bounded_endpoints"] = bounded_motion_endpoints(d["poses"], d["action"], d["parked_full"], goal_xy=goal, goal_radius_m=radius)
    d["contact_cumulative"] = np.r_[False, np.maximum.accumulate(d["contact_n"] > 1.)]
    d["bounded_cumulative"] = np.maximum.accumulate(d["bounded_endpoints"])
    d["outcome"] = json.loads((path/"outcome.json").read_text())
    d["frames"] = sorted((path/"frames").glob("frame_*.png"))
    d["frame_times"] = np.load(path/"frame_times_s.npy", allow_pickle=False)
    if len(d["frames"]) != len(d["frame_times"]) or not len(d["frames"]):
        raise ValueError("Actual frame/time inventory mismatch")
    if np.any(np.diff(d["frame_times"]) <= 0):
        raise ValueError("Video timestamps must be strictly increasing")
    return d


def encode_video(data, route, out):
    actual_path = route["actual_path"]
    if actual_path is None:
        raise ValueError("Missing actual execution directory")
    radius = float(data["manifest"].get("goal_radius_m", data["observation"].get("goal_radius_m", 2.5)))
    measured = load_actual(actual_path, data["goal"], radius)
    target = out/f"route_{route['id']}_{route.get('category', 'comparison')}.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", "1280x720", "-framerate", "10", "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast",
               "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    inset = forecast_inset(data, route)
    dt = float(measured.get("dt_s", .05))
    source_digest = hashlib.sha256()
    try:
        for k, (frame_path, timestamp) in enumerate(zip(measured["frames"], measured["frame_times"])):
            source_digest.update(bytes.fromhex(sha(frame_path)))
            image = Image.open(frame_path).convert("RGB")
            scale = min(832/image.width, 624/image.height)
            image = image.resize((int(round(image.width*scale)), int(round(image.height*scale))), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (1280, 720), "#101722")
            draw = ImageDraw.Draw(canvas)
            draw.rounded_rectangle((24, 16, 67, 58), radius=9, fill=route["color"])
            draw.text((38, 19), str(route["display_id"]), font=font(28, True), fill="white")
            draw.text((83, 17), route["label"], font=font(28, True), fill="white")
            draw.text((83, 51), "Actual Chrono execution | 10 fps", font=font(17), fill="#aebbd0")
            canvas.paste(image, (24+(832-image.width)//2, 88+(624-image.height)//2))
            frame = min(len(measured["action"]), max(0, int(round(float(timestamp)/dt))))
            draw.text((888, 29), f"INITIAL {data['horizon_s']:g}s FORECAST", font=font(21, True), fill="#e0e7f1")
            draw.text((888, 57), "Colored: forecast | cyan: measured so far", font=font(14), fill="#aebbd0")
            canvas.paste(actual_trace_inset(inset, data, measured["poses"], frame), (888, 82))
            draw.text((888, 448), f"Predicted progress: {route['progress_m']:.2f} m", font=font(20, True), fill="white")
            draw.text((888, 480), f"P(contact <={data['horizon_s']:g}s): {probability_text(route['contact_score'])}", font=font(18), fill="#aebbd0")
            draw.text((888, 505), f"P(stop <={data['horizon_s']:g}s): {probability_text(route['blockage_score'])}", font=font(18), fill="#aebbd0")
            draw.line((888, 536, 1240, 536), fill="#3b4658", width=1)
            contact = bool(measured["contact_cumulative"][frame])
            bounded = bool(measured["bounded_cumulative"][frame])
            pose, state = measured["poses"][frame], measured["states"][frame]
            distance = float(np.linalg.norm(pose[:2]-data["goal"]))
            draw.text((888, 552), "MEASURED SO FAR", font=font(19, True), fill="white")
            draw.text((888, 589), "Contact: "+("OBSERVED" if contact else "none"), font=font(22, True), fill="#fa7373" if contact else "#8de0c6")
            draw.text((888, 626), "Bounded stop: "+("YES" if bounded else "not confirmed"), font=font(17, True), fill="#f4bd5f" if bounded else "#d0d8e5")
            draw.text((888, 657), "2s motion diameter <=0.25m under effort", font=font(12), fill="#8e9db2")
            note = ("Hypothesis test; initial event risk was low." if route.get("category") in ("collision", "stall")
                    else "Lower progress is not a stagnation claim." if route.get("category") in ("progress", "low_progress")
                    else "The learned forecast ends at 4 seconds.")
            draw.text((888, 688), note, font=font(12), fill="#d4bc8f")
            hud = Image.new("RGBA", (800, 68), (8, 15, 26, 220))
            hd = ImageDraw.Draw(hud)
            hd.text((16, 8), f"TIME  {timestamp:5.2f} s", font=font(24, True), fill="white")
            hd.text((238, 8), f"FORWARD  {state[0]:+.2f} m/s", font=font(24, True), fill="white")
            hd.text((548, 8), f"TO GOAL  {distance:.1f} m", font=font(24, True), fill="white")
            suffix = "GOAL REACHED" if distance <= radius else ("END OF RECORDING" if k == len(measured["frames"])-1 else "Live measured state; overlays are annotations")
            hd.text((16, 42), suffix, font=font(15, True), fill="#99e6ce" if distance <= radius else "#c3ccda")
            canvas.paste(hud, (40, 630), hud)
            process.stdin.write(np.asarray(canvas, np.uint8).tobytes())
        # Explicit hold aids inspection; these frames are not additional simulation.
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((24, 86, 856, 120), fill="#101722")
        draw.text((42, 90), "FINAL FRAME HELD - no additional simulation", font=font(19, True), fill="#f1c777")
        for _ in range(10):
            process.stdin.write(np.asarray(canvas, np.uint8).tobytes())
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    except BaseException:
        process.kill()
        raise
    return {"id": route["id"], "category": route.get("category"), "video": target.name,
        "video_sha256": sha(target), "actual_frames": len(measured["frames"]),
        "encoded_frames": len(measured["frames"])+10, "explicit_final_hold_frames": 10,
        "actual_timestamp_start_s": float(measured["frame_times"][0]), "actual_timestamp_end_s": float(measured["frame_times"][-1]),
        "source_png_sha256_chain": source_digest.hexdigest(), "trajectory_sha256": sha(actual_path/"trajectory.npz"),
        "frame_times_sha256": sha(actual_path/"frame_times_s.npy"), "outcome_sha256": sha(actual_path/"outcome.json"),
        "video_semantics": "Actual rendered Chrono frames in timestamp order, annotated, encoded10fps; no synthesized/interpolated scene frames"}


def comparison_video(videos, out):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for video in videos:
        command += ["-i", str(out/video["video"])]
    duration = max(v["encoded_frames"]/10 for v in videos)
    graph = []
    for i in range(4):
        graph.append(f"[{i}:v]scale=640:360,tpad=stop_mode=clone:stop_duration={duration:.3f},trim=duration={duration:.3f}[v{i}]")
    graph += ["[v0][v1]hstack=inputs=2[top]", "[v2][v3]hstack=inputs=2[bottom]", "[top][bottom]vstack=inputs=2[out]"]
    target = out/"four_route_comparison.mp4"
    command += ["-filter_complex", ";".join(graph), "-map", "[out]", "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    subprocess.run(command, check=True)
    return {"video": target.name, "sha256": sha(target), "note": "Completed recordings hold their last frame with the displayed completion status; individual videos retain native duration."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overview-only", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_inputs(args.manifest)
    overview(data, args.out)
    report = {"schema": 1, "inputs": data["input_hashes"], "compositor_sha256": sha(Path(__file__)),
        "prediction_horizon_s": data["horizon_s"], "camera": data["camera"],
        "projection": "Perspective camera with current observed-depth surface lookup; unknown depth uses elevation0; future XY forecasts do not supply future height/images",
        "overview_png_sha256": sha(args.out/"prediction_overview.png"), "overview_pdf_sha256": sha(args.out/"prediction_overview.pdf")}
    if not args.overview_only:
        report["routes"] = [encode_video(data, r, args.out) for r in data["routes"]]
        report["comparison"] = comparison_video(report["routes"], args.out)
    (args.out/"visualization_provenance.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"out": str(args.out), "videos": len(report.get("routes", []))}))


if __name__ == "__main__":
    main()
