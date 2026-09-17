"""Large-arena sensor and command contract, independent of authored terrain.

Unlike the legacy 100 m camera encoding, float metric depth is never packed
through a 65 m uint16 window. One pre-drive global image remains fixed while
causal vehicle history and candidate commands change at each planning anchor.
"""
from __future__ import annotations

import math
import numpy as np
from PIL import Image

HORIZON = 60
OUTPUT_DT = .2
RECORD_DT = .05
ELEVATION_SCALE_M = 40.
IMAGE_SIZE = 512


def encode_global_rgbd(rgb, depth_m, camera):
    rgb, depth = np.asarray(rgb), np.asarray(depth_m, np.float64)
    h, w = depth.shape
    if rgb.shape != (h, w, 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected registered uint8 RGB and metric ray-depth arrays")
    if (h, w) != (camera["height"], camera["width"]):
        raise ValueError("Camera dimensions disagree with observed depth")
    if not np.isclose(camera["depth_ray_scale"], 1.):
        raise ValueError("New cohort requires the calibrated Vulkan pinhole depth geometry")
    f = (w / 2) / math.tan(float(camera["hfov_rad"]) / 2)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    sec = np.sqrt(1 + ((u - (w-1)/2)/f)**2 + ((v - (h-1)/2)/f)**2)
    valid = np.isfinite(depth) & (depth > 0) & (depth < float(camera["max_depth_m"]))
    z = float(camera["cam_height_m"]) - depth / sec
    size = int(camera.get("model_image_size", IMAGE_SIZE))
    scale = float(camera.get("elevation_scale_m", ELEVATION_SCALE_M))
    def resize(a):
        return np.asarray(Image.fromarray(np.asarray(a, np.float32)).resize((size, size), Image.Resampling.BOX))
    ok = resize(valid.astype(np.float32)) >= 1 - 1e-6
    elevation = resize(np.where(valid, np.clip(z / scale, -1, 1), 0))
    colors = np.stack([resize(rgb[..., i] / 255.) for i in range(3)])
    result = np.concatenate((colors, np.where(ok, elevation, -2)[None])).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite encoded global observation")
    return result


def build_command_features(route, anchor_pose, *, station=None, elapsed_s=0., horizon=HORIZON, output_dt=OUTPUT_DT):
    """Same candidate-conditioned FDM formulation, extended finite horizon."""
    xy, ss = np.asarray(route["waypoints"], float), np.asarray(route["stations"], float)
    vv, hh = np.asarray(route["speeds"], float), np.unwrap(np.asarray(route["headings"], float))
    pose = np.asarray(anchor_pose, float)
    if len(xy) < 2 or not np.all(np.diff(ss) > 0):
        raise ValueError("Invalid reference stations")
    st = float(ss[np.argmin(np.linalg.norm(xy-pose[:2], axis=1))] if station is None else station)
    st = float(np.clip(st, ss[0], ss[-1]))
    start = st
    c, s = np.cos(pose[2]), np.sin(pose[2])
    def ego(point):
        dx, dy = point-pose[:2]
        return [c*dx+s*dy, -s*dx+c*dy]
    commands = []
    steps = int(round(output_dt / RECORD_DT))
    if not np.isclose(steps * RECORD_DT, output_dt):
        raise ValueError("Output period must be a multiple of recording period")
    for _ in range(horizon):
        for _ in range(steps):
            st = min(ss[-1], st+max(0., float(np.interp(st, ss, vv)))*RECORD_DT)
        pt = np.array([np.interp(st, ss, xy[:, k]) for k in range(2)])
        heading = float(np.interp(st, ss, hh))-pose[2]
        speed = float(np.interp(st, ss, vv)) if st < ss[-1] else 0.
        commands.append([*ego(pt), np.sin(heading), np.cos(heading), speed])
    commands = np.asarray(commands, np.float32)
    return {"commands": commands, "nominal_pose": commands[:, :4].copy(),
            "global_features": np.asarray([ss[-1]-start, *ego(xy[-1]), elapsed_s,
                pose[0]/40., pose[1]/40., np.sin(pose[2]), np.cos(pose[2])], np.float32)}
