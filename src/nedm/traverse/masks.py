"""Analytic class masks from layout manifests (plan §6.3 / review §3.3).

Masks are never stored: they are rasterized on demand from the per-episode
layout manifest + the frozen camera model. Each asset is a convex solid, and
the perspective silhouette of a convex solid is the convex hull of its
projected surface points — so masks are exact up to hull sampling and pixel
quantization ("IoU ~ 1 up to anti-aliasing" vs ChSegmentationCamera, the
one-shot validation owed for G0b).

Solid geometry mirrors ``scene._add_assets`` / ``scene.build_scene`` (single
source of truth for the simulator; the constants below must track it):
  rock   box  edge x edge x h,        base z = ground - 0.15
  tree   trunk cylinder r x h,        base z = ground - 0.1
         canopy sphere r,             center z = trunk_top + 0.55 r
  house  walls box l x w x wall_h,    base z = ground - 0.1
         roof  box (l+0.8)x(w+0.8)x0.45 on top of the walls
  vehicle: oriented chassis-box proxy from the recorded pose (the true
         silhouette is a mesh; the box is for coverage/recall metrics —
         vehicle center/yaw probe targets come from states.npz, not masks).

Classes: 1=rock, 2=tree, 3=house, 4=vehicle; label image resolves overlap
by "highest solid top wins" (nadir-ish view, max off-nadir ~24 deg).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from nedm.traverse.camera import CameraModel

ROCK_SINK_M = 0.15
TREE_SINK_M = 0.1
HOUSE_SINK_M = 0.1
ROOF_PAD_M = 0.8
ROOF_H_M = 0.45
CANOPY_LIFT = 0.55  # canopy center = trunk_top + 0.55 * canopy_r

# Chassis-box proxy for the HMMWV (hull half-width 1.1 used across planning).
VEHICLE_LEN_M = 4.8
VEHICLE_WID_M = 2.2
VEHICLE_HGT_M = 1.9

CLASS_NAMES = {1: "rock", 2: "tree", 3: "house", 4: "vehicle"}


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew monotone chain; points (N,2) -> hull vertices (M,2), CCW."""
    pts = np.unique(np.round(points, 6), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(iterable):
        chain: list[np.ndarray] = []
        for p in iterable:
            while len(chain) >= 2 and np.cross(chain[-1] - chain[-2], p - chain[-2]) <= 0:
                chain.pop()
            chain.append(p)
        return chain

    lower = half(pts)
    upper = half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _fill_hull(draw: ImageDraw.ImageDraw, uv: np.ndarray) -> None:
    hull = _convex_hull(uv)
    if len(hull) >= 3:
        draw.polygon([(float(u), float(v)) for u, v in hull], fill=1)


def _box_corners(cx: float, cy: float, yaw: float, lx: float, ly: float, z0: float, z1: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    corners = []
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            x = cx + c * sx * lx - s * sy * ly
            y = cy + s * sx * lx + c * sy * ly
            corners.extend([(x, y, z0), (x, y, z1)])
    return np.array(corners)


def _cylinder_points(cx: float, cy: float, r: float, z0: float, z1: float, n: int = 32) -> np.ndarray:
    ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    ring = np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1)
    return np.concatenate(
        [np.column_stack([ring, np.full(n, z0)]), np.column_stack([ring, np.full(n, z1)])]
    )


def _sphere_points(cx: float, cy: float, cz: float, r: float, n_lat: int = 9, n_lon: int = 24) -> np.ndarray:
    lat = np.linspace(-math.pi / 2, math.pi / 2, n_lat)
    lon = np.linspace(0.0, 2.0 * math.pi, n_lon, endpoint=False)
    la, lo = np.meshgrid(lat, lon)
    return np.stack(
        [
            cx + r * np.cos(la.ravel()) * np.cos(lo.ravel()),
            cy + r * np.cos(la.ravel()) * np.sin(lo.ravel()),
            cz + r * np.sin(la.ravel()),
        ],
        axis=1,
    )


def _asset_solids(asset: dict[str, Any], ground_z: float) -> tuple[list[np.ndarray], float]:
    """Surface point sets for one asset (world xyz) + its top height."""
    kind, x, y, yaw = asset["kind"], asset["x_m"], asset["y_m"], asset.get("yaw_rad", 0.0)
    dims = asset["dims"]
    if kind == "rock":
        e, h = dims["edge_m"], dims["height_m"]
        z0 = ground_z - ROCK_SINK_M
        return [_box_corners(x, y, yaw, e, e, z0, z0 + h)], z0 + h
    if kind == "tree":
        tr, th, cr = dims["trunk_radius_m"], dims["trunk_height_m"], dims["canopy_radius_m"]
        z0 = ground_z - TREE_SINK_M
        cz = z0 + th + CANOPY_LIFT * cr
        return [
            _cylinder_points(x, y, tr, z0, z0 + th),
            _sphere_points(x, y, cz, cr),
        ], cz + cr
    if kind == "house":
        length, w, wh = dims["length_m"], dims["width_m"], dims["wall_height_m"]
        z0 = ground_z - HOUSE_SINK_M
        roof_z0 = z0 + wh
        return [
            _box_corners(x, y, yaw, length, w, z0, roof_z0),
            _box_corners(x, y, yaw, length + ROOF_PAD_M, w + ROOF_PAD_M, roof_z0, roof_z0 + ROOF_H_M),
        ], roof_z0 + ROOF_H_M
    raise ValueError(f"unknown asset kind: {kind}")


def _rasterize_solids(solids: list[np.ndarray], cam: CameraModel) -> np.ndarray:
    img = Image.new("1", (cam.width, cam.height), 0)
    draw = ImageDraw.Draw(img)
    for pts in solids:
        u, v = cam.world_to_pixel(pts[:, 0], pts[:, 1], pts[:, 2])
        _fill_hull(draw, np.stack([u, v], axis=1))
    return np.asarray(img, dtype=bool)


def class_masks(
    layout: dict[str, Any],
    ground_z_fn,
    cam: CameraModel | None = None,
    vehicle_pose: tuple[float, float, float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Per-class boolean masks {rock, tree, house[, vehicle]} for one frame.

    layout: the ``meta.json`` layout manifest (``EpisodeLayout.to_json()``).
    ground_z_fn: (x, y) -> terrain height (``TerrainMap.height``).
    vehicle_pose: (x, y, ground_z, yaw) from states.npz; omit for static masks.
    """
    cam = cam or CameraModel()
    masks = {name: np.zeros((cam.height, cam.width), dtype=bool) for name in ("rock", "tree", "house")}
    for asset in layout["assets"]:
        solids, _ = _asset_solids(asset, float(ground_z_fn(asset["x_m"], asset["y_m"])))
        masks[asset["kind"]] |= _rasterize_solids(solids, cam)
    if vehicle_pose is not None:
        x, y, gz, yaw = vehicle_pose
        box = _box_corners(x, y, yaw, VEHICLE_LEN_M, VEHICLE_WID_M, gz, gz + VEHICLE_HGT_M)
        masks["vehicle"] = _rasterize_solids([box], cam)
    return masks


def label_image(
    layout: dict[str, Any],
    ground_z_fn,
    cam: CameraModel | None = None,
    vehicle_pose: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Single uint8 label map (0=background), overlap -> highest solid top."""
    cam = cam or CameraModel()
    entries: list[tuple[float, int, list[np.ndarray]]] = []
    class_id = {"rock": 1, "tree": 2, "house": 3}
    for asset in layout["assets"]:
        solids, top = _asset_solids(asset, float(ground_z_fn(asset["x_m"], asset["y_m"])))
        entries.append((top, class_id[asset["kind"]], solids))
    if vehicle_pose is not None:
        x, y, gz, yaw = vehicle_pose
        box = _box_corners(x, y, yaw, VEHICLE_LEN_M, VEHICLE_WID_M, gz, gz + VEHICLE_HGT_M)
        entries.append((gz + VEHICLE_HGT_M, 4, [box]))
    out = np.zeros((cam.height, cam.width), dtype=np.uint8)
    for _, cid, solids in sorted(entries, key=lambda e: e[0]):  # low tops first
        out[_rasterize_solids(solids, cam)] = cid
    return out


def bev_occupancy(
    layout: dict[str, Any],
    size_m: float,
    grid: int = 128,
    inflate_m: float = 0.0,
) -> np.ndarray:
    """Bird's-eye occupancy grid from footprint discs (probe target, world frame).

    Cell (i, j) covers world y (north, row 0 = north edge) x x (east); matches
    the image convention of ``CameraModel`` (up = +Y).
    """
    half = size_m / 2.0
    xs = (np.arange(grid) + 0.5) / grid * size_m - half
    ys = half - (np.arange(grid) + 0.5) / grid * size_m
    gx, gy = np.meshgrid(xs, ys)
    occ = np.zeros((grid, grid), dtype=bool)
    for asset in layout["assets"]:
        r = asset["footprint_radius_m"] + inflate_m
        occ |= (gx - asset["x_m"]) ** 2 + (gy - asset["y_m"]) ** 2 <= r * r
    return occ
