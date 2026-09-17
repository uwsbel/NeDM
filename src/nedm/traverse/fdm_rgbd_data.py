"""RGB-D-only observation and reference-command contract for the PID FDM.

Depth registration uses the recorded ray depth and camera calibration only.
There is no terrain map, obstacle inventory or future observation dependency.
The caller supplies the measured current world pose for image/path alignment.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from PIL import Image

IMAGE_SIZE = 128
COMMAND_DIM = 5
GLOBAL_DIM = 8
OUTPUT_STEPS = 20
OUTPUT_DT = .2
DT = .05
CAMERA = {"width": 256, "height": 256, "hfov_deg": 47., "cam_height_m": 100.,
          "depth_ray_scale": 1.2, "depth_convention": "ray", "depth_offset_m": 80.,
          "depth_no_hit": 65535, "image_up": "+Y world (north-up)"}
COMMAND_FIELDS = ["reference_ego_x_m", "reference_ego_y_m", "sin_reference_heading", "cos_reference_heading", "reference_speed_mps"]
GLOBAL_FIELDS = ["remaining_route_m", "goal_ego_x_m", "goal_ego_y_m", "elapsed_recording_s",
                 "anchor_world_x_over40", "anchor_world_y_over40", "sin_anchor_world_yaw", "cos_anchor_world_yaw"]


@lru_cache(maxsize=8)
def _registration(width, height, hfov_deg, cam_height_m, depth_ray_scale):
    if depth_ray_scale <= 0.: raise ValueError("Invalid depth ray scale")
    f = (width/2.)/math.tan(math.radians(hfov_deg)/2.)
    cx, cy = (width-1)/2., (height-1)/2.
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    sec = np.sqrt(1.+(depth_ray_scale*(u-cx)/f)**2+(depth_ray_scale*(v-cy)/f)**2).astype(np.float32)
    su, sv = cx+(u-cx)/depth_ray_scale, cy+(v-cy)/depth_ray_scale
    inside = (su >= 0.) & (su <= width-1) & (sv >= 0.) & (sv <= height-1)
    su, sv = su.clip(0, width-1), sv.clip(0, height-1)
    x0, y0 = np.floor(su).astype(int), np.floor(sv).astype(int)
    x1, y1 = np.minimum(x0+1, width-1), np.minimum(y0+1, height-1)
    ax, ay = su-x0, sv-y0
    samples = [(y0,x0,(1-ax)*(1-ay)),(y0,x1,ax*(1-ay)),(y1,x0,(1-ax)*ay),(y1,x1,ax*ay)]
    return sec, inside, [(y,x,w.astype(np.float32)) for y,x,w in samples]


def register_observed_depth(depth_mm, camera=None):
    """Observed metric elevation registered onto RGB pixels, plus validity."""
    cfg = CAMERA | (camera or {})
    if camera and "hfov_rad" in camera:
        cfg["hfov_deg"] = math.degrees(float(camera["hfov_rad"]))
    d = np.asarray(depth_mm)
    if d.shape != (cfg["height"], cfg["width"]): raise ValueError("Depth/camera shape mismatch")
    sec, inside, samples = _registration(cfg["width"], cfg["height"], cfg["hfov_deg"], cfg["cam_height_m"], cfg["depth_ray_scale"])
    valid = d != cfg["depth_no_hit"]
    z = cfg["cam_height_m"]-(cfg["depth_offset_m"]+d.astype(np.float32)/1000.)/sec
    out, ok = np.zeros(d.shape, np.float32), inside.copy()
    for y,x,w in samples:
        ok &= (w <= 1e-12) | valid[y,x]
        out += np.where(valid[y,x], z[y,x], 0.)*w
    return np.where(ok, out, np.nan), ok


def _box_resize(array, size=IMAGE_SIZE):
    h, w = array.shape[:2]
    if h == size and w == size: return array.copy()
    if h % size == 0 and w % size == 0:
        if array.ndim == 2:
            return array.reshape(size,h//size,size,w//size).mean((1,3))
        return array.reshape(size,h//size,size,w//size,array.shape[-1]).mean((1,3))
    if array.ndim == 2:
        return np.asarray(Image.fromarray(array.astype(np.float32), mode="F").resize((size,size), Image.Resampling.BOX))
    return np.stack([_box_resize(array[...,i], size) for i in range(array.shape[-1])], -1)


def prepare_observation(rgb, depth_mm, camera=None):
    """Float32 [4,128,128]: RGB [0,1], observed registered elevation /10.

    Elevation uses a fixed [-10,10] m clip, independent of terrain statistics.
    Invalid registration or any invalid contributor to a downsampled pixel is
    represented by -2 in the fourth channel, outside the valid [-1,1] range.
    This is a four-channel sensor encoding, not an authored heightmap.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB must be HxWx3 uint8")
    if rgb.shape[:2] != np.asarray(depth_mm).shape: raise ValueError("RGB/depth shape mismatch")
    z, valid = register_observed_depth(depth_mm, camera)
    small_valid = _box_resize(valid.astype(np.float32)) >= 1.-1e-6
    depth = _box_resize(np.where(valid, np.clip(z/10., -1., 1.), 0.))
    depth = np.where(small_valid, depth, -2.)
    color = _box_resize(rgb.astype(np.float32)/255.)
    return np.concatenate((color.transpose(2,0,1), depth[None]), axis=0).astype(np.float32)


def _ego_xy(xy, pose):
    delta = np.asarray(xy)-np.asarray(pose)[:2]
    c,s = np.cos(pose[2]), np.sin(pose[2])
    return np.stack((c*delta[...,0]+s*delta[...,1],-s*delta[...,0]+c*delta[...,1]), -1)


def build_reference_features(route, anchor_pose, *, station=None, elapsed_s=0.):
    """Only intended path/speed and current localization; no physical map read."""
    xy, ss = np.asarray(route["waypoints"],float), np.asarray(route["stations"],float)
    vv, hh = np.asarray(route["speeds"],float), np.unwrap(np.asarray(route["headings"],float))
    pose = np.asarray(anchor_pose,float)
    if len(xy)<2 or not np.all(np.diff(ss)>0): raise ValueError("Invalid nominal route")
    start = ss[np.argmin(np.linalg.norm(xy-pose[:2],axis=1))] if station is None else float(station)
    start = float(np.clip(start,ss[0],ss[-1]))
    commands, st = [], start
    for _ in range(OUTPUT_STEPS):
        for _ in range(4): st = min(ss[-1],st+max(0.,float(np.interp(st,ss,vv)))*DT)
        pt = np.array([np.interp(st,ss,xy[:,j]) for j in range(2)])
        h = float(np.interp(st,ss,hh))-pose[2]
        speed = float(np.interp(st,ss,vv)) if st<ss[-1] else 0.
        commands.append([*_ego_xy(pt,pose),np.sin(h),np.cos(h),speed])
    command = np.asarray(commands,np.float32)
    return {"commands": command,"nominal_pose": command[:,:4].copy(),
            "global_features": np.array([ss[-1]-start,*_ego_xy(xy[-1],pose),elapsed_s,
                                         pose[0]/40.,pose[1]/40.,np.sin(pose[2]),np.cos(pose[2])],np.float32)}


def build_model_inputs(rgb, depth_mm, route, anchor_pose, history, *, station=None, elapsed_s=0., camera=None):
    """Shared live/preparation boundary: current raw observation and known command."""
    result = build_reference_features(route,anchor_pose,station=station,elapsed_s=elapsed_s)
    result.update(rgbd=prepare_observation(rgb,depth_mm,camera),history=np.asarray(history,np.float32))
    if result["history"].shape != (16,24): raise ValueError("Expected causal 16x24 history")
    return result


def build_command_features(route, anchor_pose, *, station=None, elapsed_s=0.):
    """Public planner name for the geometry-free reference feature helper."""
    return build_reference_features(route,anchor_pose,station=station,elapsed_s=elapsed_s)


def rgbd_from_arrays(rgb, depth_mm=None, camera=None, *, depth_m=None):
    """Live/preparation image boundary; pass encoded uint16 OR metric ray depth.

    camera fields: width, height (pixels), hfov_deg (or hfov_rad),
    cam_height_m, depth_ray_scale (legacy OptiX=1.2, calibrated Vulkan=1.).
    New backends must explicitly pass their calibration. Default is the legacy
    recorded-data contract. Invalid metric rays become the recording sentinel.
    """
    if (depth_mm is None)==(depth_m is None):
        raise ValueError("Supply exactly one of depth_mm or depth_m")
    if depth_m is not None:
        cfg=CAMERA | (camera or {})
        d=np.asarray(depth_m,float)
        mm=np.rint((d-cfg["depth_offset_m"])*1000.)
        valid=np.isfinite(mm)&(mm>=0)&(mm<cfg["depth_no_hit"])
        depth_mm=np.where(valid,mm,cfg["depth_no_hit"]).astype(np.uint16)
    return prepare_observation(rgb,depth_mm,camera)
