"""Chrono scene assembly for the HMMWV traversal arena (plan §3).

Builds the fixed-heightmap terrain + per-episode assets + HMMWV + the single
fixed overhead RGB and depth cameras. Reuses ``nedm.hmmwv_data`` for vehicle
construction and data-path plumbing.

Render notes carried over from earlier studies:
- ``create_hmmwv`` sets every visualization type NONE; we re-enable meshes here.
- Decorations (plan markers, goal ring) are visual shapes attached to the
  EXISTING terrain patch body — never new bodies (a new body perturbs the
  solver even with collision off).
- Cameras use the manual-trigger pattern from the double-pendulum study:
  nominal rate = one physics substep so the internal scheduler is always
  behind the clock, SetLag(0), SetCollectionWindow(0), and every
  ``manager.Update()`` fires exactly one render consumed via LaunchedCount.

The BMP→world orientation is empirically calibrated against
``RigidTerrain.GetHeight`` (8 dihedral transforms) and frozen into
``arena_meta.json`` so oracle and simulator provably share one heightfield.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import pychrono as chrono
import pychrono.sensor as sens
import pychrono.vehicle as veh

from nedm.hmmwv_data import configure_chrono_data_paths, create_hmmwv, repo_root_from_module
from nedm.traverse.layout import Asset, EpisodeLayout
from nedm.traverse.oracle import PlanCandidate
from nedm.traverse.terrain import META_NAME, TerrainMap, _apply_orientation

ROCK_RGB = (0.42, 0.40, 0.38)
TRUNK_RGB = (0.35, 0.22, 0.10)
CANOPY_RGB = (0.10, 0.38, 0.12)
HOUSE_WALL_RGB = (0.92, 0.90, 0.82)
HOUSE_ROOF_RGB = (0.85, 0.10, 0.08)
VEHICLE_MARKER_RGB = (0.05, 0.2, 1.0)  # blue: survives lighting; nothing else in-arena is blue
PATH_MARKER_RGB = (0.05, 0.85, 0.95)
RING_MARKER_RGB = (0.90, 0.10, 0.85)
SKY_RGB = (0.53, 0.71, 0.92)


def build_config(
    arena_dir: Path,
    start_xyz: tuple[float, float, float],
    start_yaw: float,
    step_size_s: float = 0.002,
    tire_step_size_s: float = 0.001,
) -> dict[str, Any]:
    """hmmwv_data-schema config for this arena (one heightmap, fixed size)."""
    with (arena_dir / META_NAME).open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return {
        "dataset_name": "hmmwv_traverse",
        "chrono_data_root": "chrono/data",
        "vehicle_data_root": "chrono/data/vehicle",
        "simulation": {
            "step_size_s": step_size_s,
            "tire_step_size_s": tire_step_size_s,
            "record_step_s": 0.05,
            "driver_sample_step_s": 0.05,
            "validation_ratio": 0.2,
        },
        "vehicle": {
            "model": "HMMWV_Full",
            "contact_method": "SMC",
            "chassis_fixed": False,
            "init": {
                "x_m": start_xyz[0],
                "y_m": start_xyz[1],
                "z_m": start_xyz[2],
                "yaw_rad": start_yaw,
            },
            "engine_model": "SHAFTS",
            "transmission_model": "AUTOMATIC_SHAFTS",
            "drive_type": "AWD",
            "steering_type": "PITMAN_ARM",
            "tire_model": "TMEASY",
            # The traversal arena has rigid obstacles; without chassis hulls
            # the vehicle is a ghost to them (WP0c finding: a deliberate graze
            # at 0.62 m center-to-center recorded 0 N).
            "chassis_collision": "HULLS",
        },
        "terrain": {
            "type": "rigid_heightmap",
            "length_m": meta["size_m"],
            "width_m": meta["size_m"],
            "height_min_m": meta["height_min_m"],
            "height_max_m": meta["height_max_m"],
            "friction": 0.9,
            "restitution": 0.01,
            "young_modulus_pa": 2.0e7,
        },
        "arena": meta,
    }


# ---------------------------------------------------------------------------
# BMP orientation calibration (plan §3.1): which of the 8 dihedral transforms
# of the raw BMP array matches Chrono's in-simulator heightfield.
# ---------------------------------------------------------------------------
def calibrate_orientation(arena_dir: Path, samples: int = 400, tol_m: float = 0.08) -> dict[str, Any]:
    with (arena_dir / META_NAME).open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    config = build_config(arena_dir, (0.0, 0.0, 1.6), 0.0)
    configure_chrono_data_paths(repo_root_from_module(), config)

    system = chrono.ChSystemSMC()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    terrain = veh.RigidTerrain(system)
    mat = chrono.ChContactMaterialSMC()
    mat.SetFriction(0.9)
    terrain.AddPatch(
        mat,
        chrono.CSYSNORM,
        str(arena_dir / meta["bmp"]),
        float(meta["size_m"]),
        float(meta["size_m"]),
        float(meta["height_min_m"]),
        float(meta["height_max_m"]),
    )
    terrain.Initialize()
    # Bullet raycasts see the patch only after the collision system binds it,
    # and GetHeight casts DOWN from the query point — starting at z=0 begins
    # inside any terrain above datum and silently returns 0.
    system.GetCollisionSystem().BindAll()
    probe_z = float(meta["height_max_m"]) + 5.0

    rng = np.random.default_rng(0)
    half = 0.47 * float(meta["size_m"])
    pts = rng.uniform(-half, half, size=(samples, 2))
    measured = np.array([terrain.GetHeight(chrono.ChVector3d(x, y, probe_z)) for x, y in pts])

    raw = np.asarray(Image.open(arena_dir / meta["bmp"]).convert("L"), dtype=np.float64)
    results = []
    for rot90 in range(4):
        for flipud in (False, True):
            orientation = {"rot90": rot90, "flipud": flipud}
            gray = _apply_orientation(raw, orientation)
            height = meta["height_min_m"] + gray / 255.0 * (meta["height_max_m"] - meta["height_min_m"])
            tmap = TerrainMap(height, meta["size_m"], meta)
            err = tmap.height(pts[:, 0], pts[:, 1]) - measured
            results.append((float(np.sqrt(np.mean(err**2))), float(np.median(np.abs(err))), orientation))
    results.sort(key=lambda r: r[0])
    rmse, median_abs, best = results[0]
    if median_abs > tol_m:
        raise RuntimeError(
            f"orientation calibration failed: best {best} median|err|={median_abs:.3f} m "
            f"(tol {tol_m}); top-3: {[(r[2], round(r[0], 3)) for r in results[:3]]}"
        )
    meta["orientation"] = {
        **best,
        "calibrated": True,
        "rmse_m": rmse,
        "median_abs_m": median_abs,
        "runner_up_rmse_m": results[1][0],
        "samples": samples,
    }
    with (arena_dir / META_NAME).open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return meta["orientation"]


# ---------------------------------------------------------------------------
# Frame taps (manual-trigger consumption; see double_pendulum_data.FrameTap)
# ---------------------------------------------------------------------------
class _Tap:
    def __init__(self, get_buffer, extract) -> None:
        self._get_buffer = get_buffer
        self._extract = extract
        self.taken_count = 0

    def take(self, timeout_s: float = 10.0) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            buffer = self._get_buffer()
            if buffer.HasData() and buffer.LaunchedCount > self.taken_count:
                if buffer.LaunchedCount != self.taken_count + 1:
                    raise RuntimeError(
                        f"frames skipped: expected launch {self.taken_count + 1}, got {buffer.LaunchedCount}"
                    )
                self.taken_count = buffer.LaunchedCount
                return self._extract(buffer)
            time.sleep(0.0005)
        raise RuntimeError(f"frame for launch {self.taken_count + 1} never arrived")


def _rgb_tap(camera: "sens.ChCameraSensor") -> _Tap:
    # OptiX buffers are bottom-up; flip to row 0 = image top.
    return _Tap(camera.GetMostRecentRGBA8Buffer, lambda b: np.ascontiguousarray(b.GetRGBA8Data()[::-1, :, :3]))


def _depth_tap(camera: "sens.ChDepthCamera") -> _Tap:
    def extract(buffer):
        data = np.asarray(buffer.GetDepthData(), dtype=np.float32)
        data = data.reshape(buffer.Height, buffer.Width) if data.ndim == 1 else data.squeeze()
        return np.ascontiguousarray(data[::-1, :])

    return _Tap(camera.GetMostRecentDepthBuffer, extract)


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------
@dataclass
class RenderSpec:
    width: int = 512
    height: int = 512
    cam_height_m: float = 100.0
    hfov_rad: float = math.radians(47.0)
    with_depth: bool = True
    max_depth_m: float = 250.0
    plan_markers: bool = False  # showcase only; OFF for data collection
    light_elevation_deg: float = 55.0  # WP0b geometry probes use ~80 (less shading bias)


@dataclass
class TraverseScene:
    hmmwv: Any
    system: Any
    terrain: Any
    patch_body: Any
    asset_bodies: list[tuple[Asset, Any]]
    manager: Any
    rgb_tap: _Tap | None
    depth_tap: _Tap | None
    config: dict[str, Any]


def _set_body_color(body: Any, rgb: tuple[float, float, float]) -> None:
    material = chrono.ChVisualMaterial()
    material.SetDiffuseColor(chrono.ChColor(*rgb))
    body.GetVisualShape(0).SetMaterial(0, material)


def _visual_sphere(radius: float, rgb: tuple[float, float, float]) -> Any:
    shape = chrono.ChVisualShapeSphere(radius)
    material = chrono.ChVisualMaterial()
    material.SetDiffuseColor(chrono.ChColor(*rgb))
    material.SetEmissiveColor(chrono.ChColor(*[0.5 * c for c in rgb]))
    shape.SetMaterial(0, material)
    return shape


def _add_assets(
    system: Any, tmap: TerrainMap, layout: EpisodeLayout, contact_mat: Any
) -> list[tuple[Asset, Any]]:
    bodies: list[tuple[Asset, Any]] = []
    for asset in layout.assets:
        ground_z = float(tmap.height(asset.x_m, asset.y_m))
        if asset.kind == "rock":
            edge = asset.dims["edge_m"]
            h = asset.dims["height_m"]
            body = chrono.ChBodyEasyBox(edge, edge, h, 2200.0, True, True, contact_mat)
            body.SetPos(chrono.ChVector3d(asset.x_m, asset.y_m, ground_z + h / 2.0 - 0.15))
            _set_body_color(body, ROCK_RGB)
        elif asset.kind == "tree":
            trunk_r = asset.dims["trunk_radius_m"]
            trunk_h = asset.dims["trunk_height_m"]
            canopy_r = asset.dims["canopy_radius_m"]
            body = chrono.ChBodyEasyCylinder(chrono.ChAxis_Z, trunk_r, trunk_h, 800.0, True, True, contact_mat)
            body.SetPos(chrono.ChVector3d(asset.x_m, asset.y_m, ground_z + trunk_h / 2.0 - 0.1))
            _set_body_color(body, TRUNK_RGB)
            canopy = chrono.ChVisualShapeSphere(canopy_r)
            cmat = chrono.ChVisualMaterial()
            cmat.SetDiffuseColor(chrono.ChColor(*CANOPY_RGB))
            canopy.SetMaterial(0, cmat)
            body.AddVisualShape(
                canopy, chrono.ChFramed(chrono.ChVector3d(0, 0, trunk_h / 2.0 + 0.55 * canopy_r), chrono.QUNIT)
            )
        elif asset.kind == "house":
            length = asset.dims["length_m"]
            width = asset.dims["width_m"]
            wall_h = asset.dims["wall_height_m"]
            body = chrono.ChBodyEasyBox(length, width, wall_h, 500.0, True, True, contact_mat)
            body.SetPos(chrono.ChVector3d(asset.x_m, asset.y_m, ground_z + wall_h / 2.0 - 0.1))
            body.SetRot(chrono.QuatFromAngleZ(asset.yaw_rad))
            _set_body_color(body, HOUSE_WALL_RGB)
            roof = chrono.ChVisualShapeBox(length + 0.8, width + 0.8, 0.45)
            rmat = chrono.ChVisualMaterial()
            rmat.SetDiffuseColor(chrono.ChColor(*HOUSE_ROOF_RGB))
            roof.SetMaterial(0, rmat)
            body.AddVisualShape(roof, chrono.ChFramed(chrono.ChVector3d(0, 0, wall_h / 2.0 + 0.225), chrono.QUNIT))
        else:
            raise ValueError(f"unknown asset kind: {asset.kind}")
        if asset.kind != "house":
            body.SetRot(chrono.QuatFromAngleZ(asset.yaw_rad))
        body.SetFixed(True)
        system.Add(body)
        bodies.append((asset, body))
    return bodies


def _add_plan_markers(patch_body: Any, tmap: TerrainMap, plan: PlanCandidate) -> None:
    """Showcase decoration on the EXISTING terrain body (never a new body)."""
    last_station = -3.0
    for (x, y), s in zip(plan.waypoints, plan.stations):
        if s - last_station < 2.5:
            continue
        last_station = s
        z = float(tmap.height(x, y)) + 0.35
        patch_body.AddVisualShape(
            _visual_sphere(0.22, PATH_MARKER_RGB),
            chrono.ChFramed(chrono.ChVector3d(float(x), float(y), z), chrono.QUNIT),
        )
    cx, cy = plan.meta["ring_center"]
    ring = plan.meta.get("params", {})
    radius = 7.0
    for k in range(28):
        ang = 2.0 * math.pi * k / 28
        x = cx + radius * math.cos(ang)
        y = cy + radius * math.sin(ang)
        z = float(tmap.height(x, y)) + 0.35
        patch_body.AddVisualShape(
            _visual_sphere(0.18, RING_MARKER_RGB),
            chrono.ChFramed(chrono.ChVector3d(x, y, z), chrono.QUNIT),
        )
    ax, ay, _ = plan.meta["approach_pose"]
    patch_body.AddVisualShape(
        _visual_sphere(0.5, (1.0, 1.0, 0.1)),
        chrono.ChFramed(chrono.ChVector3d(ax, ay, float(tmap.height(ax, ay)) + 0.8), chrono.QUNIT),
    )
    _ = ring


def overhead_camera_pose(cam_height_m: float) -> Any:
    """Nadir view with image up = +Y (north-up map): x_cam=-Z, z_cam=+Y."""
    rot = chrono.QuatFromAngleZ(math.pi / 2.0) * chrono.QuatFromAngleY(math.pi / 2.0)
    return chrono.ChFramed(chrono.ChVector3d(0.0, 0.0, cam_height_m), rot)


def build_scene(
    config: dict[str, Any],
    layout: EpisodeLayout,
    tmap: TerrainMap,
    arena_dir: Path,
    plan: PlanCandidate | None = None,
    render: RenderSpec | None = None,
) -> TraverseScene:
    configure_chrono_data_paths(repo_root_from_module(), config)
    hmmwv = create_hmmwv(config)
    system = hmmwv.GetSystem()

    # create_hmmwv sets every visualization NONE; re-enable for the cameras.
    hmmwv.SetChassisVisualizationType(chrono.VisualizationType_MESH)
    hmmwv.SetWheelVisualizationType(chrono.VisualizationType_MESH)
    hmmwv.SetTireVisualizationType(chrono.VisualizationType_MESH)

    # Bright roof marker so the ~15x7 px vehicle stays unambiguous at 256^2.
    marker = chrono.ChVisualShapeBox(1.6, 1.0, 0.12)
    mmat = chrono.ChVisualMaterial()
    mmat.SetDiffuseColor(chrono.ChColor(*VEHICLE_MARKER_RGB))
    # 0.4x emissive saturated the marker to sand-white under the directional
    # light (WP0b alignment probe couldn't detect it); keep it subtle.
    mmat.SetEmissiveColor(chrono.ChColor(*[0.1 * c for c in VEHICLE_MARKER_RGB]))
    marker.SetMaterial(0, mmat)
    hmmwv.GetChassisBody().AddVisualShape(
        marker, chrono.ChFramed(chrono.ChVector3d(0.1, 0.0, 0.95), chrono.QUNIT)
    )

    terrain_cfg = config["terrain"]
    terrain = veh.RigidTerrain(system)
    patch_mat = chrono.ChContactMaterialSMC()
    patch_mat.SetFriction(terrain_cfg["friction"])
    patch_mat.SetRestitution(terrain_cfg["restitution"])
    patch_mat.SetYoungModulus(terrain_cfg["young_modulus_pa"])
    patch = terrain.AddPatch(
        patch_mat,
        chrono.CSYSNORM,
        str(arena_dir / config["arena"]["bmp"]),
        float(terrain_cfg["length_m"]),
        float(terrain_cfg["width_m"]),
        float(terrain_cfg["height_min_m"]),
        float(terrain_cfg["height_max_m"]),
    )
    repo_root = repo_root_from_module()
    texture = repo_root / "chrono/data/sensor/textures/grass_texture.jpg"
    if texture.is_file():
        patch.SetTexture(str(texture), 40.0, 40.0)
    else:
        patch.SetColor(chrono.ChColor(0.42, 0.5, 0.32))
    terrain.Initialize()
    patch_body = patch.GetGroundBody()

    asset_mat = chrono.ChContactMaterialSMC()
    asset_mat.SetFriction(0.8)
    asset_mat.SetRestitution(0.01)
    asset_mat.SetYoungModulus(2.0e7)
    asset_bodies = _add_assets(system, tmap, layout, asset_mat)

    # Fixed assets sit embedded in the fixed terrain mesh; without masking,
    # Bullet narrowphases every asset against the 522k-triangle heightmap
    # every substep (~42 of 43 ms/substep measured — 96% of collection
    # cost, RTF 0.057 -> ~4 once masked). RigidTerrain already isolates the
    # patch in its own collision family for exactly this purpose; vehicle
    # collides with both sides regardless (chassis mask is all-families).
    terrain_family = patch_body.GetCollisionModel().GetFamily()
    for _, body in asset_bodies:
        model = body.GetCollisionModel()
        model.SetFamilyMask(model.GetFamilyMask() & ~(1 << terrain_family))

    manager = None
    rgb_tap = None
    depth_tap = None
    if render is not None:
        if plan is not None and render.plan_markers:
            _add_plan_markers(patch_body, tmap, plan)
        manager = sens.ChSensorManager(system)
        manager.scene.SetAmbientLight(chrono.ChVector3f(0.35, 0.35, 0.38))
        manager.scene.AddDirectionalLight(
            chrono.ChColor(1.0, 0.95, 0.85), math.radians(render.light_elevation_deg), math.radians(120.0)
        )
        background = sens.Background()
        background.mode = sens.BackgroundMode_SOLID_COLOR
        background.color_zenith = chrono.ChVector3f(*SKY_RGB)
        manager.scene.SetBackground(background)

        trigger_rate_hz = 1.0 / float(config["simulation"]["step_size_s"])
        pose = overhead_camera_pose(render.cam_height_m)
        rgb_cam = sens.ChCameraSensor(patch_body, trigger_rate_hz, pose, render.width, render.height, render.hfov_rad)
        rgb_cam.SetName("overhead_rgb")
        rgb_cam.SetLag(0.0)
        rgb_cam.SetCollectionWindow(0.0)
        rgb_cam.PushFilter(sens.ChFilterRGBA8Access())
        manager.AddSensor(rgb_cam)
        rgb_tap = _rgb_tap(rgb_cam)

        if render.with_depth:
            depth_cam = sens.ChDepthCamera(
                patch_body, trigger_rate_hz, pose, render.width, render.height, render.hfov_rad, render.max_depth_m
            )
            depth_cam.SetName("overhead_depth")
            depth_cam.SetLag(0.0)
            depth_cam.SetCollectionWindow(0.0)
            # ChDepthCamera installs its own depth-access filter; pushing
            # another ChFilterDepthAccess breaks the filter graph.
            manager.AddSensor(depth_cam)
            depth_tap = _depth_tap(depth_cam)

    return TraverseScene(
        hmmwv=hmmwv,
        system=system,
        terrain=terrain,
        patch_body=patch_body,
        asset_bodies=asset_bodies,
        manager=manager,
        rgb_tap=rgb_tap,
        depth_tap=depth_tap,
        config=config,
    )
