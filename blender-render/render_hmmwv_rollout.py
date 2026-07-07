from __future__ import annotations

import argparse
import atexit
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


DEFAULT_ASSET = Path(
    "/home/harry/NeDM/artifacts/blender_exports/hmmwv_flat_val_rest_preroll0_rollout/exported.assets.py"
)
DEFAULT_ROLLOUT = Path(
    "/home/harry/NeDM/artifacts/blender_exports/hmmwv_flat_val_rest_preroll0_rollout_eval/chrono_tracking_00.npz"
)
DEFAULT_OUTPUT = Path(
    "/home/harry/NeDM/artifacts/blender_exports/hmmwv_flat_val_rest_preroll0_rollout/renders/"
    "hmmwv_flat_val_rest_preroll0_frame120_ortho.png"
)


def blender_script_args() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Chrono HMMWV Blender postprocess export as a still image."
    )
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET, help="Chrono exported.assets.py file.")
    parser.add_argument(
        "--rollout",
        type=Path,
        default=DEFAULT_ROLLOUT,
        help="Rollout .npz with pose/ref_pose arrays. Pass --no-trajectories to skip.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    parser.add_argument("--frame", type=int, default=120, help="Blender frame to render.")
    parser.add_argument("--frame-end", type=int, default=219, help="Scene end frame.")
    parser.add_argument("--width", type=int, default=1480, help="Render width.")
    parser.add_argument("--height", type=int, default=980, help="Render height.")
    parser.add_argument("--samples", type=int, default=72, help="CPU Cycles samples.")
    parser.add_argument("--no-trajectories", action="store_true", help="Skip red/green trajectory curves.")
    return parser.parse_args(blender_script_args())


def look_at(obj, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def object_names(obj) -> str:
    names = [obj.name]
    parent = obj.parent
    while parent is not None:
        names.append(parent.name)
        parent = parent.parent
    return " ".join(names).lower()


def mesh_bounds(objects):
    pts = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        eval_obj = obj.evaluated_get(depsgraph)
        for corner in eval_obj.bound_box:
            pts.append(eval_obj.matrix_world @ Vector(corner))
    if not pts:
        raise RuntimeError("No mesh bounds available")
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def make_material(name, color, roughness=0.65, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
    return mat


def assign_material(obj, mat) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def add_polyline(name: str, xy: np.ndarray, z: float, mat, bevel_depth: float) -> bpy.types.Object:
    if len(xy) < 2:
        raise ValueError(f"{name} needs at least two points")
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 3
    spline = curve.splines.new("POLY")
    spline.points.add(len(xy) - 1)
    for point, (x, y) in zip(spline.points, xy, strict=True):
        point.co = (float(x), float(y), float(z), 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def set_render_engine(scene, samples: int) -> None:
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs:
        try:
            prefs.preferences.compute_device_type = "NONE"
        except TypeError:
            pass
    scene.cycles.device = "CPU"
    scene.cycles.samples = int(samples)
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2


def freeze_chrono_frame() -> list:
    removed = []
    for handler in list(bpy.app.handlers.frame_change_post):
        if getattr(handler, "__name__", "") == "callback_post":
            bpy.app.handlers.frame_change_post.remove(handler)
            removed.append(handler)
    return removed


def restore_chrono_handlers(handlers: list) -> None:
    for handler in handlers:
        if handler not in bpy.app.handlers.frame_change_post:
            bpy.app.handlers.frame_change_post.append(handler)


def collect_vehicle_meshes(frame_collection):
    frame_objects = list(getattr(frame_collection, "all_objects", frame_collection.objects))
    meshes = [obj for obj in frame_objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("No frame meshes found after import")

    for obj in frame_objects:
        if obj.type == "EMPTY":
            obj.empty_display_size = 0.0001
            obj.hide_select = True

    terrain_meshes = []
    vehicle_meshes = []
    for obj in meshes:
        names = object_names(obj)
        if "patch" in names or max(obj.dimensions) > 40:
            terrain_meshes.append(obj)
        else:
            vehicle_meshes.append(obj)

    if not vehicle_meshes:
        vehicle_meshes = [obj for obj in meshes if obj not in terrain_meshes]
    if not vehicle_meshes:
        vehicle_meshes = meshes

    # Fallback for unpatched Chrono HMMWV assets: obj_import leaves most chassis
    # submeshes loose in Scene Collection, while only one small mesh is linked to
    # Chassis body. Re-parent the loose pieces for rendering.
    chassis_parent = next((obj for obj in frame_objects if obj.name == "Chassis body"), None)
    extra_chassis_meshes = []
    if chassis_parent is not None:
        for obj in bpy.data.objects:
            if obj.type != "MESH" or obj.parent is not None:
                continue
            if any(collection.name.startswith("chrono") for collection in obj.users_collection):
                continue
            extra_chassis_meshes.append(obj)
            obj.parent = chassis_parent
            obj.matrix_parent_inverse.identity()
            obj.location = (0.0, 0.0, 0.0)
            obj.rotation_euler = (0.0, 0.0, 0.0)
            obj.scale = (1.0, 1.0, 1.0)
            if obj.name not in frame_collection.objects:
                frame_collection.objects.link(obj)
            vehicle_meshes.append(obj)

        for obj in list(vehicle_meshes):
            if extra_chassis_meshes and obj.parent is chassis_parent and obj.name.startswith("shape_"):
                obj.hide_render = True
                obj.hide_viewport = True
                vehicle_meshes.remove(obj)

    return frame_objects, terrain_meshes, vehicle_meshes


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    bpy.ops.preferences.addon_enable(module="chrono_import")
    bpy.ops.import_chrono.data(filepath=str(args.asset), setting_materials=True, setting_merge=False)

    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = max(int(args.frame_end), int(args.frame))
    scene.frame_set(int(args.frame))
    bpy.context.view_layer.update()
    removed_chrono_handlers = freeze_chrono_frame()
    atexit.register(restore_chrono_handlers, removed_chrono_handlers)

    frame_collection = bpy.data.collections.get("chrono_frame_objects")
    if frame_collection is None:
        raise RuntimeError("chrono_frame_objects collection not found after import")

    _, terrain_meshes, vehicle_meshes = collect_vehicle_meshes(frame_collection)
    lo, hi = mesh_bounds(vehicle_meshes)
    center = (lo + hi) * 0.5
    size = hi - lo
    radius = max(size.x, size.y, size.z)
    target = center + Vector((0.0, 0.0, 0.28))

    mat_body = make_material("hmmwv_desert_olive", (0.47, 0.50, 0.42, 1.0), 0.76)
    mat_dark = make_material("dark_rubber_metal", (0.055, 0.055, 0.052, 1.0), 0.84)
    mat_metal = make_material("muted_axle_metal", (0.30, 0.31, 0.30, 1.0), 0.58, 0.05)
    mat_glass = make_material("smoked_glass", (0.08, 0.10, 0.11, 1.0), 0.42, 0.02)
    mat_light = make_material("muted_headlight", (0.92, 0.80, 0.48, 1.0), 0.35)
    mat_ground = make_material("flat_terrain_matte", (0.48, 0.43, 0.35, 1.0), 0.93)
    mat_ref_line = make_material("reference_trajectory_green", (0.02, 0.72, 0.22, 1.0), 0.48)
    mat_rollout_line = make_material("rollout_trajectory_red", (0.82, 0.10, 0.055, 1.0), 0.52)

    for obj in terrain_meshes:
        obj.hide_render = True
        obj.hide_viewport = True

    vehicle_mesh_set = set(vehicle_meshes)
    for obj in bpy.data.objects:
        if obj.type in {"CURVE", "GPENCIL", "GREASEPENCIL"}:
            obj.hide_render = True
            obj.hide_viewport = True
        elif obj.type == "MESH" and obj not in vehicle_mesh_set:
            obj.hide_render = True
            obj.hide_viewport = True

    for obj in vehicle_meshes:
        obj.hide_render = False
        names = object_names(obj)
        max_dim = max(obj.dimensions)
        if any(token in names for token in ("wi_", ".wi", "window", "mirror")):
            assign_material(obj, mat_glass)
        elif any(token in names for token in ("hlight", "signal", "lites", "turn_signa")):
            assign_material(obj, mat_light)
        elif "chassis" in names:
            assign_material(obj, mat_body)
        elif "spindle" in names and max_dim > 0.45:
            assign_material(obj, mat_dark)
        elif any(token in names for token in ("susp", "steering", "upright", "lca", "uca")):
            assign_material(obj, mat_metal)
        elif max_dim < 0.18:
            assign_material(obj, mat_metal)
        else:
            assign_material(obj, mat_dark)

    bpy.ops.mesh.primitive_plane_add(size=max(radius * 4.0, 18.0), location=(target.x, target.y, -0.015))
    ground = bpy.context.object
    ground.name = "render_flat_terrain_plane"
    assign_material(ground, mat_ground)

    if not args.no_trajectories:
        with np.load(args.rollout) as rollout:
            ref_xy = np.asarray(rollout["ref_pose"][:, :2], dtype=float)
            pose_xy = np.asarray(rollout["pose"][:, :2], dtype=float)
        add_polyline("render_reference_trajectory", ref_xy, 0.08, mat_ref_line, 0.045)
        add_polyline("render_rollout_trajectory", pose_xy, 0.10, mat_rollout_line, 0.032)

    cam_data = bpy.data.cameras.new("render_camera")
    cam = bpy.data.objects.new("render_camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    view_dir = Vector((0.95, -1.20, 0.58)).normalized()
    cam.location = target + view_dir * max(radius * 4.0, 22.0)
    look_at(cam, target)
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = max(radius * 1.42, 6.5)
    cam_data.clip_end = 5000
    scene.camera = cam

    sun_data = bpy.data.lights.new("sun_key", "SUN")
    sun = bpy.data.objects.new("sun_key", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(48), 0.0, math.radians(135))
    sun_data.energy = 1.7

    area_data = bpy.data.lights.new("softbox", "AREA")
    area = bpy.data.objects.new("softbox", area_data)
    bpy.context.scene.collection.objects.link(area)
    area.location = target + Vector((-5.0, -6.0, 8.0))
    look_at(area, target)
    area_data.energy = 430
    area_data.size = 7.0

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.color = (0.70, 0.74, 0.78)

    set_render_engine(scene, args.samples)
    scene.render.resolution_x = int(args.width)
    scene.render.resolution_y = int(args.height)
    scene.render.film_transparent = False
    scene.render.filepath = str(args.output)

    for transform in ("Standard", "AgX", "Filmic"):
        try:
            scene.view_settings.view_transform = transform
            break
        except TypeError:
            continue
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1

    bpy.ops.render.render(write_still=True)
    print(f"RENDERED {args.output}")
    print(f"VEHICLE_BOUNDS lo={tuple(round(v, 3) for v in lo)} hi={tuple(round(v, 3) for v in hi)}")
    print(f"CAMERA location={tuple(round(v, 3) for v in cam.location)} ortho_scale={cam_data.ortho_scale:.3f}")
    restore_chrono_handlers(removed_chrono_handlers)


if __name__ == "__main__":
    main()
