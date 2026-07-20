"""Render a Chrono M113+arm Blender postprocess export as a single still image.

Adapted from ``render_hmmwv_rollout.py`` for the combined tracked-drive -> arm-reach
demo (``scripts/eval_tracked_then_arm_chrono.py --blender-output-dir ...``). The
material classification is keyed on the M113 track/wheel/suspension body names and
the LRV arm link names (see the exported ``output/stateNNNNN.py`` object names),
so the hull and arm render light and the tracks/road-wheels render dark -- matching
the earlier ``tracked_arm_two_policy_consecutive`` render.

Run headless, e.g.:

    /home/harry/blender-4.0.0-linux-x64/blender --background --python \
        blender-render/render_tracked_arm_rollout.py -- --frame 0
"""

from __future__ import annotations

import argparse
import atexit
import math
import os
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector


REPO_ROOT = Path("/home/harry/NeDM")
DEFAULT_ASSET = REPO_ROOT / "artifacts/combined_tracked_arm_demo/blender/exported.assets.py"
DEFAULT_OUTPUT = (
    REPO_ROOT / "artifacts/combined_tracked_arm_demo/blender/renders/tracked_arm_frame0000_ortho.png"
)


def blender_script_args() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Chrono M113+arm Blender export as a still.")
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET, help="Chrono exported.assets.py file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    parser.add_argument("--frame", type=int, default=0, help="Blender frame to render (0 = first frame).")
    parser.add_argument("--frame-end", type=int, default=690, help="Scene end frame.")
    parser.add_argument("--width", type=int, default=1480, help="Render width.")
    parser.add_argument("--height", type=int, default=980, help="Render height.")
    parser.add_argument("--samples", type=int, default=96, help="CPU Cycles samples.")
    # Goal-anchored "line" camera: sit a bit offset from the tracked-vehicle goal position
    # and look along the goal->origin direction (down the driven path toward the start).
    parser.add_argument("--goal-x", type=float, default=-15.593, help="Tracked-vehicle goal x (world m).")
    parser.add_argument("--goal-y", type=float, default=3.147, help="Tracked-vehicle goal y (world m).")
    parser.add_argument("--origin-x", type=float, default=0.0, help="Drive start (origin) x (world m).")
    parser.add_argument("--origin-y", type=float, default=0.0, help="Drive start (origin) y (world m).")
    parser.add_argument("--ground-z", type=float, default=0.57, help="Settled chassis ground height (world m).")
    parser.add_argument("--cam-back", type=float, default=9.0,
                        help="Camera distance behind the goal along -(goal->origin) (m).")
    parser.add_argument("--cam-side", type=float, default=2.5,
                        help="Camera lateral offset (left of the goal->origin line) for a 3/4 view (m).")
    parser.add_argument("--cam-height", type=float, default=6.5, help="Camera height above ground (m).")
    parser.add_argument("--look-height", type=float, default=1.4,
                        help="Height of the aim point at the goal (m); tilts the view down onto the rig.")
    parser.add_argument("--ortho-scale", type=float, default=None,
                        help="Override the orthographic scale (default: auto from rig bounds).")
    parser.add_argument("--perspective", action="store_true",
                        help="Use a PERSPECTIVE camera (far rig small, near rig big) instead of ortho. "
                             "Needed to show the vehicle growing from origin (frame 1) to the goal (arm phase).")
    parser.add_argument("--cam-lens", type=float, default=32.0,
                        help="Perspective focal length in mm (36x24 mm sensor); smaller = wider FOV.")
    parser.add_argument("--bg-color", type=float, nargs=3, default=(0.53, 0.72, 0.95),
                        metavar=("R", "G", "B"), help="Sky/background color (light blue by default).")
    parser.add_argument("--chassis-alpha", type=float, default=0.3,
                        help="Chassis (hull) material alpha; <1 makes the hull see-through.")
    parser.add_argument("--shadows", action="store_true",
                        help="Enable cast shadows (default: disabled).")
    # Animation (render the whole state sequence as PNGs in one Blender session).
    parser.add_argument("--animate", action="store_true",
                        help="Render every frame in [--anim-start, --anim-end] as a PNG sequence.")
    parser.add_argument("--anim-start", type=int, default=0, help="First frame of the sequence.")
    parser.add_argument("--anim-end", type=int, default=690, help="Last frame of the sequence.")
    parser.add_argument("--anim-dir", type=Path,
                        default=REPO_ROOT / "artifacts/combined_tracked_arm_demo/blender/renders/anim",
                        help="Output directory for the PNG sequence (frameNNNN.png).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip frames whose PNG already exists (resume after a crash / chunked runs).")
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


def make_material(name, color, roughness=0.65, metallic=0.0, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (color[0], color[1], color[2], float(alpha))
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = color
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
        if "Alpha" in bsdf.inputs:  # Cycles mixes with transparency from this input
            bsdf.inputs["Alpha"].default_value = float(alpha)
    if alpha < 1.0:
        try:
            mat.blend_method = "BLEND"  # only affects EEVEE viewport; harmless/absent in some builds
        except (AttributeError, TypeError):
            pass
    return mat


def assign_material(obj, mat) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(mat)


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
    scene.cycles.transparent_max_bounces = 12  # see through both faces of the alpha'd chassis


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


# M113 running-gear tokens (dark); everything else on the rig (hull + arm links) is light.
DARK_TOKENS = (
    "trackshoe", "roadwheel", "sprocket", "idler", "suspension", "_shoe", "_wheel", "_gear",
)
ARM_LINK_TOKENS = (
    "base-1", "shoulder-1", "bicep-1", "elbow-1", "wrist-1", "endeffector-1", "finger-1", "finger-2",
)


def collect_meshes(frame_collection):
    frame_objects = list(getattr(frame_collection, "all_objects", frame_collection.objects))
    for obj in frame_objects:
        if obj.type == "EMPTY":
            obj.empty_display_size = 0.0001
            obj.hide_select = True
    meshes = [obj for obj in frame_objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("No frame meshes found after import")

    terrain_meshes, rig_meshes = [], []
    for obj in meshes:
        names = object_names(obj)
        if "patch" in names or max(obj.dimensions) > 40:
            terrain_meshes.append(obj)
        else:
            rig_meshes.append(obj)
    if not rig_meshes:
        rig_meshes = meshes
    return terrain_meshes, rig_meshes


def build_materials(args):
    """Create the styling materials once (they survive Chrono's per-frame object rebuild)."""
    return {
        "chassis": make_material("m113_hull_glass", (0.70, 0.71, 0.69, 1.0), 0.6,
                                 alpha=float(args.chassis_alpha)),
        "arm": make_material("lrv_arm_light", (0.80, 0.80, 0.81, 1.0), 0.55, 0.05),
        "track": make_material("track_dark_rubber", (0.045, 0.045, 0.045, 1.0), 0.85),
        "wheel": make_material("road_wheel_metal", (0.20, 0.20, 0.21, 1.0), 0.55, 0.05),
        "ground": make_material("flat_terrain_matte", (0.48, 0.43, 0.35, 1.0), 0.93),
        "goal": make_material("goal_marker_green", (0.08, 0.72, 0.20, 1.0), 0.35),
        "ee": make_material("ee_marker_red", (0.86, 0.07, 0.06, 1.0), 0.35),
    }


def apply_style(materials):
    """Re-hide terrain/curves and (re)assign materials + marker colors on the CURRENT
    chrono_frame_objects. Must run every frame: Chrono's frame handler deletes and rebuilds
    those objects (with default materials) on each frame change. Returns the rig meshes.
    Scoped to Chrono objects + all curves/GP so the ground plane/camera/lights are untouched.
    """
    frame_collection = bpy.data.collections.get("chrono_frame_objects")
    if frame_collection is None:
        raise RuntimeError("chrono_frame_objects collection not found after import")
    terrain_meshes, rig_meshes = collect_meshes(frame_collection)
    for obj in terrain_meshes:
        obj.hide_render = obj.hide_viewport = True
    for obj in bpy.data.objects:
        if obj.type in {"CURVE", "GPENCIL", "GREASEPENCIL"}:
            obj.hide_render = obj.hide_viewport = True

    marker_objs, ee_ref_positions = [], []
    for obj in rig_meshes:
        obj.hide_render = False
        names = object_names(obj)
        dims = obj.dimensions
        max_dim, min_dim = max(dims), min(dims)
        is_marker = (
            not any(tok in names for tok in ("m113", "chassis"))
            and not any(tok in names for tok in ARM_LINK_TOKENS)
            and 0.0 < max_dim < 0.3
            and (min_dim / max_dim) > 0.7
        )
        if "trackshoe" in names or "_shoe" in names:
            assign_material(obj, materials["track"])
        elif any(tok in names for tok in DARK_TOKENS):
            assign_material(obj, materials["wheel"])
        elif any(tok in names for tok in ARM_LINK_TOKENS):
            assign_material(obj, materials["arm"])
            if "endeffector-1" in names or "finger-1" in names or "finger-2" in names:
                ee_ref_positions.append(obj.matrix_world.translation.copy())
        elif is_marker:
            marker_objs.append(obj)
        else:  # chassis / deck / unclassified hull part -> see-through hull
            assign_material(obj, materials["chassis"])

    if marker_objs:
        if ee_ref_positions:
            ee_ref = sum(ee_ref_positions, Vector((0.0, 0.0, 0.0))) / len(ee_ref_positions)
            marker_objs.sort(key=lambda o: (o.matrix_world.translation - ee_ref).length)
        for i, obj in enumerate(marker_objs):
            assign_material(obj, materials["ee"] if i == 0 else materials["goal"])
    if os.environ.get("NEDM_RENDER_DEBUG"):
        smalls = [(o.name, tuple(round(d, 3) for d in o.dimensions)) for o in rig_meshes
                  if 0.0 < max(o.dimensions) < 0.5]
        print(f"[dbg] rig={len(rig_meshes)} markers={len(marker_objs)} ee_ref={len(ee_ref_positions)} "
              f"smalls={smalls[:6]}", flush=True)
    return rig_meshes


def setup_scene(scene, args, goal, origin, target, radius, materials):
    """One-time scene setup (ground plane, goal-line camera, lights, sky, engine)."""
    path_mid = (goal + origin) * 0.5
    plane_size = max(radius * 6.0, float((goal - origin).length) * 6.0, 400.0)
    bpy.ops.mesh.primitive_plane_add(size=plane_size, location=(path_mid.x, path_mid.y, -0.015))
    ground = bpy.context.object
    ground.name = "render_flat_terrain_plane"
    assign_material(ground, materials["ground"])

    line_dir = Vector((origin.x - goal.x, origin.y - goal.y, 0.0))
    if line_dir.length < 1e-6:
        line_dir = Vector((1.0, 0.0, 0.0))
    line_dir.normalize()
    perp = Vector((-line_dir.y, line_dir.x, 0.0))

    cam_data = bpy.data.cameras.new("render_camera")
    cam = bpy.data.objects.new("render_camera", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (
        goal - line_dir * args.cam_back + perp * args.cam_side + Vector((0.0, 0.0, args.cam_height))
    )
    look_at(cam, target)
    if args.perspective:
        cam_data.type = "PERSP"
        cam_data.lens = float(args.cam_lens)
    else:
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = float(args.ortho_scale) if args.ortho_scale else max(radius * 1.55, 7.0)
    cam_data.clip_end = 5000
    scene.camera = cam

    sun_data = bpy.data.lights.new("sun_key", "SUN")
    sun = bpy.data.objects.new("sun_key", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(48), 0.0, math.radians(135))
    sun_data.energy = 1.7

    area_data = bpy.data.lights.new("softbox", "AREA")
    area = bpy.data.objects.new("softbox", area_data)
    scene.collection.objects.link(area)
    area.location = target + Vector((-5.0, -6.0, 8.0))
    look_at(area, target)
    area_data.energy = 430
    area_data.size = 7.0

    if not args.shadows:
        for light_data in (sun_data, area_data):
            try:
                light_data.use_shadow = False
            except (AttributeError, TypeError):
                pass
            try:
                light_data.cycles.cast_shadow = False
            except (AttributeError, TypeError):
                pass

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    r, g, b = args.bg_color
    world.color = (r, g, b)
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node is not None:
        bg_node.inputs["Color"].default_value = (r, g, b, 1.0)
        bg_node.inputs["Strength"].default_value = 1.0

    set_render_engine(scene, args.samples)
    scene.render.resolution_x = int(args.width)
    scene.render.resolution_y = int(args.height)
    scene.render.film_transparent = False
    for transform in ("Standard", "AgX", "Filmic"):
        try:
            scene.view_settings.view_transform = transform
            break
        except TypeError:
            continue
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    return cam


def render_animation(args, scene, goal, origin, target, materials) -> None:
    """Render the whole frame range as a PNG sequence in ONE Blender session.

    Chrono's frame_change handler stays ACTIVE (not frozen) so it rebuilds the geometry each
    frame; we re-apply styling after every frame_set, then render. Import happens only once.
    """
    anim_dir = Path(args.anim_dir).expanduser().resolve()
    anim_dir.mkdir(parents=True, exist_ok=True)
    start, end = int(args.anim_start), int(args.anim_end)
    scene.frame_start, scene.frame_end = start, end

    scene.frame_set(start)
    bpy.context.view_layer.update()
    rig_meshes = apply_style(materials)
    lo, hi = mesh_bounds(rig_meshes)
    radius = max((hi - lo).x, (hi - lo).y, (hi - lo).z)
    cam = setup_scene(scene, args, goal, origin, target, radius, materials)
    print(f"[anim] CAMERA location={tuple(round(v, 3) for v in cam.location)}  "
          f"frames {start}..{end}  -> {anim_dir}", flush=True)

    total = end - start + 1
    done = 0
    t0 = time.time()
    for f in range(start, end + 1):
        out_path = anim_dir / f"frame{f:04d}.png"
        if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            done += 1
            continue
        scene.frame_set(f)                 # Chrono rebuilds objects for frame f
        bpy.context.view_layer.update()
        apply_style(materials)             # restyle the freshly-created objects
        scene.render.filepath = str(out_path)
        # Freeze Chrono's frame handler across the render: bpy.ops.render.render re-fires
        # frame_change_post, which would delete our restyled objects and rebuild them with
        # default materials right before the pixels are shaded. Restore it for the next frame_set.
        frozen = freeze_chrono_frame()
        bpy.ops.render.render(write_still=True)
        restore_chrono_handlers(frozen)
        done += 1
        if done % 20 == 0 or done == total:
            el = time.time() - t0
            eta = (total - done) * el / done / 60.0
            print(f"[anim] {done}/{total} rendered  ({el / done:.1f}s/frame, ETA {eta:.1f} min)",
                  flush=True)
    print(f"ANIM_DONE {done} frames -> {anim_dir}", flush=True)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Enable the chrono_import add-on. Run Blender with --factory-startup so it is NOT
    # pre-loaded from userprefs (a pre-loaded copy triggers a broken unregister/re-register
    # cycle on addon_enable). Verify via the real registered operator type, since
    # hasattr(bpy.ops.import_chrono, "data") is always truthy even when unregistered.
    import addon_utils

    if not hasattr(bpy.types, "IMPORT_CHRONO_OT_data"):
        try:
            addon_utils.enable("chrono_import", default_set=False, persistent=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] chrono_import enable raised (continuing): {exc}")
    if not hasattr(bpy.types, "IMPORT_CHRONO_OT_data"):
        raise RuntimeError("chrono_import operator unavailable; cannot import Chrono assets.")
    bpy.ops.import_chrono.data(filepath=str(args.asset), setting_materials=True, setting_merge=False)

    scene = bpy.context.scene
    goal = Vector((args.goal_x, args.goal_y, args.ground_z))
    origin = Vector((args.origin_x, args.origin_y, args.ground_z))
    target = Vector((goal.x, goal.y, args.look_height))
    materials = build_materials(args)

    if args.animate:
        # Whole-sequence render: Chrono's frame handler stays active and rebuilds geometry per
        # frame; we restyle + render each one. (No freeze_chrono_frame here.)
        render_animation(args, scene, goal, origin, target, materials)
        return

    # ---- single still: freeze on one frame ----
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.frame_start = 0
    scene.frame_end = max(int(args.frame_end), int(args.frame))
    scene.frame_set(int(args.frame))
    bpy.context.view_layer.update()
    removed_chrono_handlers = freeze_chrono_frame()
    atexit.register(restore_chrono_handlers, removed_chrono_handlers)

    rig_meshes = apply_style(materials)
    lo, hi = mesh_bounds(rig_meshes)
    radius = max((hi - lo).x, (hi - lo).y, (hi - lo).z)
    cam = setup_scene(scene, args, goal, origin, target, radius, materials)
    scene.render.filepath = str(args.output)

    bpy.ops.render.render(write_still=True)
    print(f"RENDERED {args.output}")
    print(f"RIG_BOUNDS lo={tuple(round(v, 3) for v in lo)} hi={tuple(round(v, 3) for v in hi)}")
    print(f"CAMERA location={tuple(round(v, 3) for v in cam.location)}")
    restore_chrono_handlers(removed_chrono_handlers)


if __name__ == "__main__":
    main()
