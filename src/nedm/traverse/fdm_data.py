"""Causal reference-path features and labels for the standard Chrono PID FDM.

Inputs use only the measured prefix and the commanded path. No future measured
station, pose, image or reactive actuator sequence enters candidate features.
The BMP and asset footprints are privileged geometry for this first experiment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from nedm.training.constants import STATE_FIELD_PRESETS, DEFAULT_ACTION_FIELDS

DT = 0.05
HISTORY_STEPS = 16
HISTORY_DIM = 24
CANDIDATE_STEPS = 64
CANDIDATE_DIM = 13
GLOBAL_DIM = 4
OUTPUT_STEPS = 20
OUTPUT_DT = 0.2
OUTPUT_OFFSETS = np.arange(1, OUTPUT_STEPS + 1) * 4
TERRAIN_INDICES = (5, 6, 7, 8, 9, 10)
STATE_FIELDS = STATE_FIELD_PRESETS["tire_normal_force_omega_pt"]
EVENT_NAMES = ("contact", "rollover", "low_progress")
HISTORY_FIELDS = STATE_FIELDS + ["previous_" + k for k in DEFAULT_ACTION_FIELDS] + ["ego_x", "ego_y", "sin_yaw", "cos_yaw"]
CANDIDATE_FIELDS = ["ego_x", "ego_y", "sin_heading", "cos_heading", "speed", "height", "along_slope", "cross_slope", "roughness", "left_height", "right_height", "asset_clearance", "valid"]


def canonical_id(key: str) -> str:
    return key.replace("full_v4_partial__", "full_v4__", 1)


def frozen_split_assignments(manifest: dict) -> dict[str, str]:
    """Preserve the original WP2 permutation, including partial/full aliases."""
    keys = manifest["episodes"]
    order = np.random.default_rng(20260902).permutation(len(keys))
    nt, nv = int(.7 * len(keys)), int(.15 * len(keys))
    result = {}
    for split, indices in zip(("train", "val", "test"), (order[:nt], order[nt:nt+nv], order[nt+nv:])):
        for i in indices:
            key = canonical_id(keys[i])
            if key in result and result[key] != split:
                raise ValueError("Conflicting canonical split: " + key)
            result[key] = split
    return result


def novel_split(key: str) -> str:
    # Stable extension: adding episodes never changes an existing assignment.
    p = int(hashlib.sha256(("fdm_split_v1:" + canonical_id(key)).encode()).hexdigest()[:8], 16) % 100
    return "train" if p < 70 else "val" if p < 85 else "test"


def load_episode(path: str | Path) -> dict:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    with np.load(path / "states.npz") as d:
        table, fields = d["table"].copy(), d["fields"].tolist()
    index = {k: i for i, k in enumerate(fields)}
    pick = lambda names: table[:, [index[k] for k in names]].astype(np.float32)
    power = table[:, index["engine_motorshaft_torque_nm"]] * table[:, index["trans_motorshaft_speed_radps"]] / 1000.
    if not np.isclose(float(meta["frame_dt_s"]), DT):
        raise ValueError("Unexpected recording interval")
    return {"state": pick(STATE_FIELDS), "actions": pick(DEFAULT_ACTION_FIELDS),
            "poses": pick(["pos_x_m", "pos_y_m", "yaw_rad"]),
            "power_kw": power.astype(np.float32), "meta": meta}


def ego_xy(xy: np.ndarray, pose: np.ndarray) -> np.ndarray:
    d = np.asarray(xy) - np.asarray(pose)[:2]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.stack((c*d[..., 0] + s*d[..., 1], -s*d[..., 0] + c*d[..., 1]), -1)


def build_history(state, actions, poses, anchor: int) -> np.ndarray:
    """State at t plus action from [t-dt,t), so changing the next path is causal.

    Missing startup rows repeat the first measured state/pose and use a
    synthetic straight/brake command. This is padding, not measured settle
    telemetry. global_features[3] exposes elapsed recording time, so its valid
    history count is min(16, 1 + round(elapsed_s / .05)).
    """
    idx = np.arange(anchor-HISTORY_STEPS+1, anchor+1)
    st = np.asarray(state)[idx.clip(0)]
    ac = np.asarray(actions)[(idx-1).clip(0)].copy()
    ac[idx <= 0] = (0., 0., 1.)
    ps = np.asarray(poses)[idx.clip(0)]
    yaw = ps[:, 2] - poses[anchor, 2]
    return np.concatenate((st, ac, ego_xy(ps[:, :2], poses[anchor]), np.sin(yaw)[:, None], np.cos(yaw)[:, None]), -1).astype(np.float32)


def tracked_route_indices(route, poses) -> np.ndarray:
    """Original collector's monotone 60-waypoint search, using a prefix only."""
    xy = np.asarray(route["waypoints"])
    result = []
    last = 0
    for pose in poses:
        hi = min(len(xy), last+60)
        last += int(np.argmin(np.linalg.norm(xy[last:hi] - pose[:2], axis=1)))
        result.append(last)
    return np.asarray(result, np.int64)


def parking_mask(route, poses, indices=None) -> np.ndarray:
    indices = tracked_route_indices(route, poses) if indices is None else indices
    return (indices >= len(route["waypoints"])-2) & (np.linalg.norm(np.asarray(poses)[:, :2] - np.asarray(route["waypoints"])[-1], axis=1) < 3.)


def build_candidate_features(route, anchor_pose, terrain, layout, *, station=None, elapsed_s=0.) -> dict:
    """Shared training/planning interface; all geometry comes from the command.

    station is the driver's currently tracked nominal station, never a future
    observed station. A new candidate defaults to its nearest anchor waypoint.
    The nominal trajectory advances through spatial reference speeds at 20 Hz.
    """
    xy = np.asarray(route["waypoints"], np.float64)
    ss = np.asarray(route["stations"], np.float64)
    speeds = np.asarray(route["speeds"], np.float64)
    headings = np.unwrap(np.asarray(route["headings"], np.float64))
    pose = np.asarray(anchor_pose, np.float64)
    if len(xy) < 2 or not np.all(np.diff(ss) > 0):
        raise ValueError("Route needs strictly increasing stations")
    start = float(ss[np.argmin(np.linalg.norm(xy-pose[:2], axis=1))] if station is None else station)
    start = float(np.clip(start, ss[0], ss[-1]))
    query = start + np.linspace(0., 40., CANDIDATE_STEPS)
    valid = query <= ss[-1] + 1e-6
    q = np.minimum(query, ss[-1])

    def at(stations):
        pts = np.stack([np.interp(stations, ss, xy[:, j]) for j in range(2)], -1)
        h = np.interp(stations, ss, headings)
        return pts, h, np.interp(stations, ss, speeds)

    pts, h, sp = at(q)
    tx, ty = np.cos(h), np.sin(h)
    z0 = float(terrain.height(pose[0], pose[1]))
    z = np.asarray(terrain.height(pts[:, 0], pts[:, 1]))
    gx, gy = terrain.gradient(pts[:, 0], pts[:, 1])
    left = terrain.height(pts[:, 0]-1.5*ty, pts[:, 1]+1.5*tx)
    right = terrain.height(pts[:, 0]+1.5*ty, pts[:, 1]-1.5*tx)
    fore = terrain.height(pts[:, 0]+1.9*tx, pts[:, 1]+1.9*ty)
    aft = terrain.height(pts[:, 0]-1.9*tx, pts[:, 1]-1.9*ty)
    rough = np.std(np.stack((z, left, right, fore, aft)), axis=0)
    clearance = np.full(len(pts), 40.)
    for asset in (layout or {}).get("assets", []):
        center = np.array([asset["x_m"], asset["y_m"]])
        radius = float(asset.get("footprint_radius_m", 0.)) + 1.3
        for off in (-1.9, 0., 1.9):
            body = pts + off*np.stack((tx, ty), -1)
            clearance = np.minimum(clearance, np.linalg.norm(body-center, axis=1)-radius)
    hd = h-pose[2]
    feat = np.column_stack((ego_xy(pts, pose), np.sin(hd), np.cos(hd), sp, z-z0,
                            gx*tx+gy*ty, -gx*ty+gy*tx, rough, left-z0, right-z0, clearance, valid))
    feat[~valid, :-1] = 0.
    nominal = []
    st = start
    for _ in range(OUTPUT_STEPS):
        for _ in range(4):
            st = min(ss[-1], st + max(0., float(np.interp(st, ss, speeds)))*DT)
        pp, hh, _ = at(np.asarray([st]))
        nominal.append([*ego_xy(pp, pose)[0], np.sin(hh[0]-pose[2]), np.cos(hh[0]-pose[2])])
    return {"candidate": feat.astype(np.float32),
            "global_features": np.array([ss[-1]-start, *ego_xy(xy[-1], pose), elapsed_s], np.float32),
            "nominal_pose": np.asarray(nominal, np.float32)}


def build_targets(poses, states, actions, power_kw, meta, anchor: int, parked) -> dict:
    """Contact frame i labels interval [state_i,state_{i+1}), not state_i.

    All event heads are horizon-specific. Contact and rollover are cumulative
    within the future prefix. Low progress compares prefix net displacement to
    0.15 m/s under mean throttle >0.3; prefixes below 2 s or including deliberate
    route-end parking have no low-progress supervision. A capped contact list
    cannot supply negatives after its last known frame. Known positives remain
    supervised after censoring/termination; trajectories never extrapolate.
    """
    n = len(poses)
    target_idx = anchor+OUTPUT_OFFSETS
    valid = target_idx < n
    ps = np.asarray(poses)[target_idx.clip(max=n-1)]
    yaw = ps[:, 2]-poses[anchor, 2]
    trajectory = np.column_stack((ego_xy(ps[:, :2], poses[anchor]), np.sin(yaw), np.cos(yaw))).astype(np.float32)
    events = np.zeros((OUTPUT_STEPS, 3), np.float32)
    event_mask = np.zeros_like(events)
    work = np.zeros((OUTPUT_STEPS, 1), np.float32)
    contacts = np.asarray(meta.get("contact", {}).get("events", []), float).reshape(-1, 3)
    frames = contacts[contacts[:, 2] > 1., 0].astype(int)
    capped = len(contacts) >= 2000
    # Past the final listed frame, missing contact entries may have been lost.
    contact_known_until = int(contacts[-1, 0]) if capped else n
    terminal_rollover = meta.get("status") == "rollover"
    for j, end in enumerate(target_idx):
        seen = min(end, n)
        positive_contact = bool(np.any((frames >= anchor) & (frames < end)))
        events[j, 0] = positive_contact
        event_mask[j, 0] = positive_contact or (end <= n and (not capped or end <= contact_known_until))
        interval_roll = np.any(np.abs(np.asarray(states)[anchor+1:min(end+1, n), 2:4]) > np.deg2rad(60.))
        terminal = terminal_rollover and anchor <= n-1 < end
        events[j, 1] = bool(interval_roll or terminal)
        event_mask[j, 1] = bool(events[j, 1]) or end < n
        work[j, 0] = np.maximum(np.asarray(power_kw)[anchor:seen], 0.).sum()*DT
        if valid[j] and OUTPUT_OFFSETS[j]*DT >= 2. and not np.any(np.asarray(parked)[anchor:end+1]):
            event_mask[j, 2] = 1.
            events[j, 2] = (np.linalg.norm(poses[end, :2]-poses[anchor, :2]) < .15*OUTPUT_OFFSETS[j]*DT) and np.mean(actions[anchor:end, 1]) > .3
    trajectory[~valid] = 0.
    work[~valid] = 0.
    return {"trajectory": trajectory, "work": work, "events": events,
            "trajectory_mask": valid[:, None].astype(np.float32), "event_mask": event_mask}
