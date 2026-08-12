"""Export a Chrono HMMWV scene (rigid / bumpy / CRM terrain) for Blender rendering.

Stage 1 of the Blender render pipeline. This runs Chrono, settles the HMMWV on the
requested terrain, and writes everything the Blender stage needs:

* ``exported.assets.py`` + ``output/stateNNNNN.py`` -- Chrono's ``ChBlender``
  postprocess export of the vehicle (and, for ``bumpy``, the heightmap patch mesh).
* ``crm_surface/surface_NNNNN.obj`` -- for CRM only, the soil free surface
  reconstructed from the SPH particles with ``splashsurf`` (see ``nedm.crm_surface``).
  The particle field itself is never rendered: 11 M point primitives is not a
  practical Blender scene, whereas the marching-cubes surface is a few MB of mesh.
* ``scene_manifest.json`` -- terrain kind, paths, and vehicle pose, so the Blender
  stage can pick materials/camera without being told the terrain type again.

Stage 2 is ``blender-render/render_hmmwv_scene.py``.

Examples
--------
    PYTHONPATH=src python scripts/rendering/export_hmmwv_blender_scene.py \
        --terrain rigid --output-dir artifacts/blender_exports/snapshot_rigid
    PYTHONPATH=src python scripts/rendering/export_hmmwv_blender_scene.py \
        --terrain bumpy --output-dir artifacts/blender_exports/snapshot_bumpy
    PYTHONPATH=src python scripts/rendering/export_hmmwv_blender_scene.py \
        --terrain crm --output-dir artifacts/blender_exports/snapshot_crm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import pychrono as chrono  # noqa: E402
import pychrono.vehicle as veh  # noqa: E402

from nedm.blender_export import BlenderFrameExporter  # noqa: E402
from nedm.crm_surface import export_crm_surface  # noqa: E402
from nedm.hmmwv_data import (  # noqa: E402
    assign_height_map_index,
    configure_chrono_data_paths,
    create_hmmwv,
    create_rigid_terrain,
    load_config,
    resolve_project_path,
)
from nedm.hmmwv_crm import configure_crm_terrain, load_crm_config  # noqa: E402


DEFAULT_CONFIGS = {
    "rigid": "configs/hmmwv_overfit_v1.json",
    "bumpy": "configs/hmmwv_bumpy_eval.json",
    "crm": "configs/hmmwv_crm_eval.json",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--terrain", choices=sorted(DEFAULT_CONFIGS), required=True, help="Terrain kind to build.")
    parser.add_argument("--config", type=Path, default=None, help="Collector config. Defaults per --terrain.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Blender export directory to write.")
    parser.add_argument("--settle-s", type=float, default=None, help="Seconds to settle before export (per-terrain default).")
    parser.add_argument("--vehicle-x-m", type=float, default=0.0, help="Vehicle spawn x.")
    parser.add_argument("--vehicle-y-m", type=float, default=0.0, help="Vehicle spawn y.")
    parser.add_argument("--vehicle-yaw-rad", type=float, default=0.0, help="Vehicle spawn yaw.")
    parser.add_argument(
        "--drop-height-m",
        type=float,
        default=0.70,
        help="Spawn height above the local terrain surface (chassis reference frame). The settled "
        "ride height is ~0.58 m, so this starts the vehicle just clear of the surface and lets it "
        "drop onto it rather than starting interpenetrated.",
    )
    parser.add_argument(
        "--height-map-index",
        type=int,
        default=None,
        help="bumpy only: heightmap index to load. Mutually exclusive with --episode-id.",
    )
    parser.add_argument(
        "--episode-id",
        type=str,
        default=None,
        help="bumpy only: reproduce the heightmap this dataset episode was collected on.",
    )
    parser.add_argument(
        "--crm-length-m",
        type=float,
        default=38.0,
        help="crm only: soil bed length override. The 150 m eval bed is ~11 M particles; "
        "a small bed is physically identical near the vehicle and builds in under a second. "
        "Keep it wider than 2x --surface-half-extent-m so the reconstructed slab is bounded "
        "by real soil rather than by the bed wall.",
    )
    parser.add_argument("--crm-width-m", type=float, default=30.0, help="crm only: soil bed width override.")
    parser.add_argument(
        "--surface-half-extent-m",
        type=float,
        nargs=2,
        default=(16.0, 13.0),
        help="crm only: XY half-window around the vehicle to reconstruct the soil surface over. "
        "The reconstruction is a closed volume, so its crop faces are visible vertical walls -- "
        "size this so they fall outside the camera frame.",
    )
    parser.add_argument(
        "--surface-cube-size",
        type=float,
        default=1.0,
        help="crm only: splashsurf marching-cubes cell size (x particle radius). Chrono's default of "
        "0.5 (=0.02 m here) resolves far below the 0.08 m particle spacing and costs ~170 MB of mesh; "
        "1.0 is visually identical at a tenth of the size.",
    )
    parser.add_argument("--surface-smoothing-length", type=float, default=1.5, help="crm only: splashsurf smoothing length (x particle radius).")
    parser.add_argument("--surface-threshold", type=float, default=0.6, help="crm only: splashsurf iso-surface density threshold.")
    parser.add_argument(
        "--surface-format",
        choices=("ply", "obj"),
        default="ply",
        help="crm only: soil surface mesh format. Binary PLY is ~3x smaller than OBJ for the same mesh.",
    )
    parser.add_argument("--no-surface-cleanup", action="store_true", help="crm only: skip splashsurf's marching-cubes mesh decimation.")
    parser.add_argument("--keep-particles", action="store_true", help="crm only: keep the intermediate .xyz particle dump.")
    parser.add_argument("--blender-width", type=int, default=1600, help="Picture width stored in the assets script.")
    parser.add_argument("--blender-height", type=int, default=1000, help="Picture height stored in the assets script.")
    args = parser.parse_args(argv)
    if args.height_map_index is not None and args.episode_id is not None:
        parser.error("pass at most one of --height-map-index / --episode-id")
    return args


def resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config is not None:
        return args.config if args.config.is_absolute() else REPO_ROOT / args.config
    return REPO_ROOT / DEFAULT_CONFIGS[args.terrain]


def resolve_height_map_path(config: dict[str, Any], args: argparse.Namespace) -> tuple[int, Path]:
    terrain_cfg = config["terrain"]
    count = int(terrain_cfg["height_map_count"])
    if args.episode_id is not None:
        index = assign_height_map_index(args.episode_id, count)
    else:
        index = int(args.height_map_index or 0) % count
    height_map_dir = resolve_project_path(REPO_ROOT, terrain_cfg["height_map_dir"])
    path = height_map_dir / (terrain_cfg.get("height_map_pattern", "bumpy_field_%03d.bmp") % index)
    if not path.is_file():
        raise FileNotFoundError(f"height map not found: {path}")
    return index, path


def probe_terrain_height(config: dict[str, Any], height_map_path: Path | None, x: float, y: float) -> float:
    """Terrain height at (x, y), measured on a throwaway system.

    The vehicle must be created before its terrain (the terrain is built on the
    vehicle's own ChSystem), so a spawn height that hugs the surface cannot be
    computed from the real terrain without a chicken-and-egg. Building the same
    patch on a scratch system is cheap for rigid/heightmap terrain and avoids
    dropping the HMMWV from a fixed height onto an unknown bump.
    """
    scratch = chrono.ChSystemSMC()
    scratch.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    terrain = create_rigid_terrain(scratch, config, height_map_path=height_map_path)
    # RigidTerrain::GetHeight ray-casts through the collision system. On a system that
    # has never been stepped the patch's collision BVH is not built yet, and the cast
    # segfaults for a trimesh (heightmap) patch -- one tiny step populates it.
    scratch.DoStepDynamics(1.0e-5)
    # The cast starts *at* the query point and goes down, so probing at z=0 silently
    # misses every part of the terrain above z=0 and reports 0 ("no hit"). Start above
    # the heightmap's declared maximum.
    probe_z = float(config["terrain"].get("height_max_m", 0.0)) + 10.0
    return float(terrain.GetHeight(chrono.ChVector3d(float(x), float(y), probe_z)))


def enable_vehicle_visuals(hmmwv: Any) -> None:
    """Re-enable the HMMWV visual assets.

    ``create_hmmwv`` initializes every subsystem with ``VisualizationType_NONE``
    (the collector never renders), so an export made straight after it contains
    only empties. Call this *before* any stepping: the suspension/steering
    primitive visuals freeze their hardpoints in absolute coordinates when the
    asset is added, so adding them after the vehicle has moved leaves the linkage
    visuals floating away from the wheels.
    """
    for setter in (
        hmmwv.SetChassisVisualizationType,
        hmmwv.SetSuspensionVisualizationType,
        hmmwv.SetSteeringVisualizationType,
        hmmwv.SetWheelVisualizationType,
        hmmwv.SetTireVisualizationType,
    ):
        setter(chrono.VisualizationType_MESH)


def build_scene(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Create the HMMWV and its terrain. Returns the scene handles plus metadata."""
    height_map_index: int | None = None
    height_map_path: Path | None = None

    if args.terrain == "bumpy":
        height_map_index, height_map_path = resolve_height_map_path(config, args)

    if args.terrain == "crm":
        # The soil bed top sits at center_z + depth (Construct builds upward from the
        # given center), so the spawn height is known without probing.
        surface_z = float(config["terrain"]["center_m"][2]) + float(config["terrain"]["depth_m"])
    else:
        surface_z = probe_terrain_height(config, height_map_path, args.vehicle_x_m, args.vehicle_y_m)

    config["vehicle"]["init"] = dict(config["vehicle"]["init"])
    config["vehicle"]["init"].update(
        {
            "x_m": float(args.vehicle_x_m),
            "y_m": float(args.vehicle_y_m),
            "z_m": surface_z + float(args.drop_height_m),
            "yaw_rad": float(args.vehicle_yaw_rad),
        }
    )

    hmmwv = create_hmmwv(config)
    # Visuals must be enabled before the terrain build and before any stepping.
    enable_vehicle_visuals(hmmwv)

    wheels = None
    if args.terrain == "crm":
        terrain, wheels = configure_crm_terrain(hmmwv, config)
    else:
        terrain = create_rigid_terrain(hmmwv.GetSystem(), config, height_map_path=height_map_path)

    return {
        "hmmwv": hmmwv,
        "terrain": terrain,
        "wheels": wheels,
        "surface_z": surface_z,
        "height_map_index": height_map_index,
        "height_map_path": height_map_path,
    }


def settle(scene: dict[str, Any], config: dict[str, Any], terrain_kind: str, settle_s: float) -> None:
    """Hold the vehicle on the brakes until it has come to rest on the terrain."""
    hmmwv = scene["hmmwv"]
    terrain = scene["terrain"]
    step_size_s = float(config["simulation"]["step_size_s"])
    driver_inputs = veh.DriverInputs()
    driver_inputs.m_steering = 0.0
    driver_inputs.m_throttle = 0.0
    driver_inputs.m_braking = 1.0
    driver_inputs.m_clutch = 0.0

    num_steps = max(int(round(settle_s / step_size_s)), 1)
    started = time.time()
    for step in range(num_steps):
        time_s = float(hmmwv.GetSystem().GetChTime())
        terrain.Synchronize(time_s)
        hmmwv.Synchronize(time_s, driver_inputs, terrain)
        terrain.Advance(step_size_s)
        if terrain_kind != "crm":
            # CRMTerrain.Advance co-steps the coupled FSI + multibody system, so the
            # vehicle is already integrated; calling hmmwv.Advance() as well on CRM
            # would integrate it twice per substep.
            hmmwv.Advance(step_size_s)
        if step % max(num_steps // 10, 1) == 0:
            elapsed = time.time() - started
            print(f"  settle {step}/{num_steps} t={time_s:.3f}s wall={elapsed:.1f}s", flush=True)
    print(f"  settled {num_steps} steps in {time.time() - started:.1f}s", flush=True)


def export_blender_frame(scene: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    chassis_pos = scene["hmmwv"].GetVehicle().GetPos()
    exporter = BlenderFrameExporter(
        scene["hmmwv"].GetSystem(),
        output_dir,
        fps=20.0,
        max_frames=1,
        picture_size=(int(args.blender_width), int(args.blender_height)),
        camera_location=(float(chassis_pos.x) - 9.0, float(chassis_pos.y) - 11.0, 5.0),
        camera_aim=(float(chassis_pos.x), float(chassis_pos.y), 1.0),
        clean=True,
    )
    exporter.maybe_export(force=True)
    exporter.write_summary()
    return exporter.summary()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = resolve_config_path(args)
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.terrain == "crm":
        config = load_crm_config(config_path)
        config["terrain"]["length_m"] = float(args.crm_length_m)
        config["terrain"]["width_m"] = float(args.crm_width_m)
    else:
        config = load_config(config_path)
    configure_chrono_data_paths(REPO_ROOT, config)

    settle_s = args.settle_s if args.settle_s is not None else (1.5 if args.terrain == "crm" else 2.0)

    print(f"[{args.terrain}] config={config_path}")
    print(f"[{args.terrain}] building scene ...", flush=True)
    scene = build_scene(config, args)
    print(f"[{args.terrain}] terrain surface z={scene['surface_z']:.3f} m", flush=True)
    if args.terrain == "crm":
        print(f"[{args.terrain}] SPH particles: {scene['terrain'].GetNumSPHParticles()}", flush=True)

    print(f"[{args.terrain}] settling {settle_s:.2f}s ...", flush=True)
    settle(scene, config, args.terrain, settle_s)

    chassis_pos = scene["hmmwv"].GetVehicle().GetPos()
    print(f"[{args.terrain}] chassis at ({chassis_pos.x:.3f}, {chassis_pos.y:.3f}, {chassis_pos.z:.3f})", flush=True)

    print(f"[{args.terrain}] exporting Blender frame ...", flush=True)
    blender_summary = export_blender_frame(scene, output_dir, args)

    manifest: dict[str, Any] = {
        "terrain": args.terrain,
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "assets_script": blender_summary["assets_script"],
        "state_file_count": blender_summary["state_file_count"],
        "settle_s": float(settle_s),
        "sim_time_s": float(scene["hmmwv"].GetSystem().GetChTime()),
        "terrain_surface_z_m": float(scene["surface_z"]),
        "vehicle_pos_m": [float(chassis_pos.x), float(chassis_pos.y), float(chassis_pos.z)],
        "vehicle_yaw_rad": float(args.vehicle_yaw_rad),
    }

    if args.terrain == "bumpy":
        manifest["height_map_index"] = scene["height_map_index"]
        manifest["height_map_path"] = str(scene["height_map_path"])
        manifest["terrain_extent_m"] = [float(config["terrain"]["length_m"]), float(config["terrain"]["width_m"])]

    if args.terrain == "rigid":
        manifest["terrain_extent_m"] = [float(config["terrain"]["length_m"]), float(config["terrain"]["width_m"])]

    if args.terrain == "crm":
        surface_dir = output_dir / "crm_surface"
        surface_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = surface_dir / f"surface_00000.{args.surface_format}"
        print(f"[{args.terrain}] reconstructing soil surface with splashsurf ...", flush=True)
        started = time.time()
        reconstruction = export_crm_surface(
            scene["terrain"],
            mesh_path,
            initial_spacing_m=float(config["terrain"]["initial_spacing_m"]),
            center_xy=(float(chassis_pos.x), float(chassis_pos.y)),
            half_extent_xy=tuple(float(v) for v in args.surface_half_extent_m),
            keep_particle_file=bool(args.keep_particles),
            smoothing_length=float(args.surface_smoothing_length),
            cube_size=float(args.surface_cube_size),
            surface_threshold=float(args.surface_threshold),
            mesh_cleanup=not args.no_surface_cleanup,
        )
        print(
            f"[{args.terrain}] surface: {reconstruction.num_particles} of "
            f"{reconstruction.num_particles_total} particles -> {mesh_path.name} "
            f"({mesh_path.stat().st_size / 1e6:.1f} MB, {time.time() - started:.1f}s)",
            flush=True,
        )
        manifest["crm_surface"] = reconstruction.as_dict()
        manifest["crm_bed_extent_m"] = [float(config["terrain"]["length_m"]), float(config["terrain"]["width_m"])]
        manifest["crm_initial_spacing_m"] = float(config["terrain"]["initial_spacing_m"])

    manifest_path = output_dir / "scene_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[{args.terrain}] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
