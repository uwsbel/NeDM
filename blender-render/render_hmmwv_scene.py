"""Render an HMMWV Chrono scene / policy rollout exported for Blender.

Stage 2 of the Blender render pipeline. Imports a Chrono ``ChBlender`` postprocess
export, dresses the scene per terrain kind, optionally overlays the reference and
rollout trajectories from an eval ``.npz``, and renders either a still or a frame
sequence.

Scene sources (one of):
  --manifest   scene_manifest.json written by scripts/rendering/export_hmmwv_blender_scene.py
  --assets     exported.assets.py written by eval_hmmwv_rl_chrono_tracking.py
               --blender-output-dir (pass --terrain to say which terrain it is)

Terrain handling differs by kind, which is why this exists alongside
``render_hmmwv_rollout.py`` (that one always hides the terrain and substitutes an
untextured flat plane, which is right for ``rigid`` and wrong for the other two):

* ``rigid``  -- the exported patch is a 240 m box; hide it and lay down a textured plane.
* ``bumpy``  -- keep the exported heightmap patch mesh, which carries the real relief.
* ``crm``    -- the SPH soil has no Chrono visual asset at all; import the splashsurf
                surface mesh named in the manifest.

Run with the Blender 5.1.2 build and ``--factory-startup`` (the chrono_import add-on
in the saved userprefs is half-registered and cannot be re-enabled; 4.0.0 lacks the
``shade_auto_smooth`` operator the add-on calls):

    ~/blender-5.1.2-linux-x64/blender --background --factory-startup \
        --python blender-render/render_hmmwv_scene.py -- \
        --assets artifacts/blender_exports/rollout_rigid_ref00_mixture/exported.assets.py \
        --terrain rigid --rollout <...>/chrono_tracking_00.npz \
        --animate --anim-dir <...>/frames

Blender 5.1.2 segfaults in native code under sustained Cycles-CPU load on this box, so
sequences are driven by ``scripts/rendering/render_rollout_video.sh``: a fresh process
per chunk plus ``--skip-existing``, retried until the sequence is complete.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUND_TEXTURE = REPO_ROOT / "chrono/data/vehicle/terrain/textures/dirt.jpg"

TERRAIN_TINTS = {
    # (tint RGBA, roughness). The tint multiplies the ground texture, so values above 1
    # lift Chrono's dirt.jpg -- it is a dark, desaturated scan and renders as near-black
    # mud at unit tint. Without a texture the tint is used as a flat colour instead, so
    # these stay close to plausible albedos.
    "rigid": ((1.45, 1.30, 1.05, 1.0), 0.93),
    "bumpy": ((1.30, 1.18, 0.94, 1.0), 0.95),
    "crm": ((1.25, 0.98, 0.72, 1.0), 0.97),
}

REFERENCE_COLOR = (0.03, 0.72, 0.26, 1.0)
ROLLOUT_COLOR = (0.86, 0.11, 0.06, 1.0)


def blender_script_args() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an exported HMMWV Chrono scene or rollout.")
    source = parser.add_argument_group("scene source")
    source.add_argument("--manifest", type=Path, default=None, help="scene_manifest.json from the export stage.")
    source.add_argument("--assets", type=Path, default=None, help="Chrono exported.assets.py (use with --terrain).")
    source.add_argument("--terrain", choices=sorted(TERRAIN_TINTS), default=None, help="Terrain kind for --assets.")
    source.add_argument("--crm-surface", type=Path, default=None, help="Single soil surface mesh (stills).")
    source.add_argument(
        "--crm-surface-dir",
        type=Path,
        default=None,
        help="Directory of per-frame soil meshes + crm_surface_manifest.json, as written by "
        "eval_hmmwv_rl_chrono_tracking.py --blender-output-dir on CRM. Required to animate CRM: "
        "the soil deforms every frame, so one static mesh would freeze the ruts.",
    )
    source.add_argument(
        "--terrain-z-m",
        type=float,
        default=None,
        help="Terrain surface height for the --assets path. The trajectory tubes sit just above it, so "
        "on CRM this must be the soil top (bed center_z + depth, e.g. 0.25) or the lines end up buried.",
    )

    parser.add_argument("--rollout", type=Path, default=None, help="Eval .npz with pose/ref_pose to draw as trajectories.")
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Read --rollout for the camera path but do not draw the reference/driven tubes. "
        "The chase camera still needs the npz; this only hides the overlay.",
    )
    parser.add_argument(
        "--trajectory-radius-scale",
        type=float,
        default=1.0,
        help="Scale the trajectory tubes and the reference marker. The defaults are sized for a "
        "30 m-wide top-down frame and are far too fat in a chase close-up; ~0.35 works there.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output PNG for a single frame.")
    parser.add_argument("--frame", type=int, default=0, help="Blender frame (state index) to render.")

    anim = parser.add_argument_group("animation")
    anim.add_argument("--animate", action="store_true", help="Render a frame sequence instead of one still.")
    anim.add_argument("--anim-dir", type=Path, default=None, help="Directory for frameNNNN.png output.")
    anim.add_argument("--anim-start", type=int, default=0, help="First frame.")
    anim.add_argument("--anim-end", type=int, default=None, help="Last frame (default: last exported state).")
    anim.add_argument("--skip-existing", action="store_true", help="Skip frames whose PNG already exists.")

    parser.add_argument("--width", type=int, default=1600, help="Render width.")
    parser.add_argument("--height", type=int, default=1000, help="Render height.")
    parser.add_argument("--samples", type=int, default=96, help="Cycles samples per frame.")
    parser.add_argument("--gpu", action="store_true", help="Render on the GPU (OptiX/CUDA) instead of the CPU.")

    cam = parser.add_argument_group("camera")
    cam.add_argument("--camera-azimuth-deg", type=float, default=52.0, help="Camera bearing around the subject.")
    cam.add_argument("--camera-elevation-deg", type=float, default=20.0, help="Camera elevation above the subject.")
    cam.add_argument("--camera-distance-m", type=float, default=16.0, help="Camera distance from the subject.")
    cam.add_argument("--perspective", action="store_true", help="Use a perspective camera instead of orthographic.")
    cam.add_argument("--cam-lens-mm", type=float, default=50.0, help="Perspective focal length.")
    cam.add_argument("--ortho-scale-m", type=float, default=9.0, help="Orthographic view width (ignored when auto-framing).")
    cam.add_argument(
        "--follow",
        action="store_true",
        help="Chase camera locked to the vehicle. Default for a rollout is a fixed auto-framed "
        "camera, so the three policy videos share one framing and are directly comparable.",
    )
    cam.add_argument("--frame-margin-m", type=float, default=2.5, help="Auto-frame margin around the trajectories.")

    follow = parser.add_argument_group("follow camera (--follow)")
    follow.add_argument(
        "--follow-mode",
        choices=("chase", "world"),
        default="chase",
        help="'chase' rides with the vehicle's heading, so the camera holds the same view of the "
        "body and the tyres however the vehicle turns. 'world' keeps --camera-azimuth-deg and only "
        "translates, which preserves the sun/rut raking angle but shows the vehicle from behind on "
        "one leg and head-on on the next.",
    )
    follow.add_argument(
        "--follow-azimuth-deg",
        type=float,
        default=38.0,
        help="Chase bearing measured from straight behind the vehicle; positive swings to its left. "
        "0 is a pure tail view (hides the tyres behind the body), 90 is a pure side view (loses the "
        "sense of travel). ~35-45 is the three-quarter rear that shows body, front wheel and the "
        "ruts being cut at the same time.",
    )
    follow.add_argument("--follow-distance-m", type=float, default=9.0, help="Chase camera distance from the look-at point.")
    follow.add_argument(
        "--follow-elevation-deg",
        type=float,
        default=15.0,
        help="Chase camera elevation. Low keeps the ruts in silhouette against the light; above "
        "~25 the tracks flatten out the same way they do for the fixed camera.",
    )
    follow.add_argument(
        "--follow-target-z-offset-m",
        type=float,
        default=-0.30,
        help="Look-at height relative to the vehicle's bounding-box centre. Negative aims at the "
        "wheels and the contact patch instead of the roof.",
    )
    follow.add_argument(
        "--follow-lead-m",
        type=float,
        default=0.0,
        help="Shift the look-at point along the heading. Positive shows more of the ground the "
        "vehicle is about to cross, negative more of the ruts behind it. Keep it under ~1 m: the "
        "camera swings about the look-at point, so a large lead walks the vehicle out of frame.",
    )
    follow.add_argument(
        "--follow-smooth-frames",
        type=int,
        default=9,
        help="Centred moving-average window over the vehicle track. The raw pose carries yaw jitter "
        "and suspension bob that a rigidly locked camera turns into visible shake; smoothing lets "
        "the vehicle move slightly within the frame instead, which is what a real chase shot does.",
    )
    follow.add_argument(
        "--follow-ortho",
        action="store_true",
        help="Keep the orthographic camera while following. Follow mode is perspective by default: "
        "an ortho chase shot has no depth cue at all, so the soil relief reads as a flat texture.",
    )

    ground = parser.add_argument_group("ground")
    ground.add_argument(
        "--ground-texture",
        type=str,
        default=str(DEFAULT_GROUND_TEXTURE),
        help="Ground albedo image. Pass 'none' for a flat colour.",
    )
    ground.add_argument("--ground-texture-scale-m", type=float, default=6.0, help="Texture tile size in metres.")
    ground.add_argument(
        "--contour-interval-m",
        type=float,
        default=0.0,
        help="Draw topographic contour lines on the terrain every N metres of height (0 = off). "
        "This is how the bumpy terrain is made legible: it is a ~1-3%% grade undulation, so it is "
        "honestly near-flat at vehicle scale and lighting alone cannot show it. 0.1 suits a "
        "terrain with +/-0.5 m of relief.",
    )
    ground.add_argument(
        "--relief-shading",
        dest="relief_exaggeration",
        type=float,
        default=0.0,
        help="Hillshade the bumpy terrain by tilting its shading normals as if slopes were N "
        "times steeper (0/1 = off, 8-12 reads well). Geometry, and so the vehicle's contact "
        "with it, is untouched -- only the normal handed to the shader changes. Say the factor "
        "in the caption.",
    )
    ground.add_argument(
        "--elevation-tint-strength",
        type=float,
        default=0.0,
        help="Shade the bumpy terrain by world height, dark low ground to bright high ground "
        "(0 = off, ~0.8 reads well). Contour lines alone look like cracks in a flat surface; "
        "a continuous luminance ramp reads as a surface. Geometry is untouched either way.",
    )
    ground.add_argument(
        "--elevation-range-m",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="World-Z window the elevation tint spans. Defaults to the measured relief of the "
        "terrain patch inside the framed area, which is what you want unless you are matching "
        "two shots to one scale.",
    )
    ground.add_argument(
        "--contour-strength", type=float, default=0.55, help="How dark the contour lines are (0-1)."
    )
    ground.add_argument("--terrain-radius-m", type=float, default=140.0, help="Radius of the substituted ground plane (rigid).")

    parser.add_argument("--sun-energy", type=float, default=1.7, help="Key (sun) light strength.")
    parser.add_argument("--sky-strength", type=float, default=0.35, help="World background light strength.")
    # Rut visibility is a lighting problem, not a mesh-resolution one: the CRM surface
    # carries the tracks as real geometry, but a high sun plus a strong fill flattens
    # them out. Grazing light (low elevation, raking across the path) plus a weak fill
    # is what makes centimetre-deep tracks read.
    parser.add_argument(
        "--crm-filler-plane",
        action="store_true",
        help="Hide the CRM slab's crop walls behind a ground plane at the soil surface.",
    )
    parser.add_argument(
        "--crm-filler-drop-m",
        type=float,
        default=0.06,
        help="How far below the soil top to park the filler plane.",
    )
    parser.add_argument("--sun-elevation-deg", type=float, default=48.0, help="Sun elevation. Low (~15) rakes the ruts.")
    parser.add_argument(
        "--sun-azimuth-offset-deg",
        type=float,
        default=80.0,
        help="Sun bearing relative to the camera azimuth. ~90 lights the ruts side-on.",
    )
    parser.add_argument("--fill-energy", type=float, default=430.0, help="Area softbox strength. 0 disables the fill.")
    parser.add_argument(
        "--soil-shading",
        choices=("smooth", "flat"),
        default="smooth",
        help="Shading for the CRM soil surface. 'flat' keeps per-face normals so the "
        "marching-cubes relief is not averaged away.",
    )
    parser.add_argument("--save-blend", type=Path, default=None, help="Also save the assembled .blend for inspection.")

    args = parser.parse_args(blender_script_args())
    if (args.manifest is None) == (args.assets is None):
        parser.error("pass exactly one of --manifest / --assets")
    if args.assets is not None and args.terrain is None:
        parser.error("--assets requires --terrain")
    if args.animate and args.anim_dir is None:
        parser.error("--animate requires --anim-dir")
    return args


def resolve_scene(args: argparse.Namespace) -> dict:
    """Normalize the two scene sources into one manifest-shaped dict."""
    if args.manifest is not None:
        manifest_path = args.manifest.expanduser().resolve()
        scene = json.loads(manifest_path.read_text(encoding="utf-8"))
        scene.setdefault("_base_dir", str(manifest_path.parent))
        return scene

    assets = args.assets.expanduser().resolve()
    base_dir = assets.parent
    state_dir = base_dir / "output"
    state_count = len(list(state_dir.glob("state*.py"))) if state_dir.is_dir() else 1
    scene = {
        "terrain": args.terrain,
        "assets_script": str(assets),
        "state_file_count": state_count,
        "terrain_surface_z_m": float(args.terrain_z_m) if args.terrain_z_m is not None else 0.0,
        "_base_dir": str(base_dir),
    }
    if args.terrain == "crm":
        if args.crm_surface_dir is not None:
            surface_dir = args.crm_surface_dir.expanduser().resolve()
            manifest = json.loads((surface_dir / "crm_surface_manifest.json").read_text(encoding="utf-8"))
            scene["crm_surface_sequence"] = {
                "dir": str(surface_dir),
                "pattern": manifest["pattern"],
                "frame_count": int(manifest["frame_count"]),
            }
            first = surface_dir / (manifest["pattern"] % 0)
            scene["crm_surface"] = {"mesh_path": str(first)}
        elif args.crm_surface is not None:
            scene["crm_surface"] = {"mesh_path": str(args.crm_surface.expanduser().resolve())}
        else:
            raise SystemExit("--terrain crm with --assets needs --crm-surface or --crm-surface-dir")
    return scene


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


def make_material(name, color, roughness=0.65, metallic=0.0, emission_strength=0.0):
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
        # A little emission keeps the trajectory tubes legible where the vehicle
        # shadows them, without blowing out the rest of the frame.
        if emission_strength > 0.0:
            if "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = color
            elif "Emission" in bsdf.inputs:
                bsdf.inputs["Emission"].default_value = color
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = emission_strength
    return mat


def add_elevation_contours(mat, base_socket, interval_m: float, strength: float, line_width: float = 0.06):
    """Darken thin bands at every ``interval_m`` of world height.

    The point is legibility, not decoration. This "bumpy" terrain is a long-wavelength
    undulation -- roughly +/-0.5 m over 30-60 m, i.e. a ~1-3% grade -- so at vehicle scale
    it is honestly near-flat and no amount of lighting makes it read as bumpy. Contour
    lines show the relief exactly as a topographic map does, without touching the geometry
    (vertical exaggeration would decouple the vehicle from the ground it is driving on).

    ``PINGPONG`` turns world Z into a symmetric triangle wave, so a line lands on both
    sides of each multiple of the interval and every band has the same width.
    """
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    geometry = nodes.new("ShaderNodeNewGeometry")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    scale = nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = 1.0 / float(interval_m)
    pingpong = nodes.new("ShaderNodeMath")
    pingpong.operation = "PINGPONG"
    pingpong.inputs[1].default_value = 0.5

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ramp.color_ramp.elements[1].position = max(float(line_width), 1e-3)
    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MULTIPLY"
    mix.inputs["Fac"].default_value = float(np.clip(strength, 0.0, 1.0))

    links.new(geometry.outputs["Position"], separate.inputs["Vector"])
    links.new(separate.outputs["Z"], scale.inputs[0])
    links.new(scale.outputs["Value"], pingpong.inputs[0])
    links.new(pingpong.outputs["Value"], ramp.inputs["Fac"])
    links.new(base_socket, mix.inputs["Color1"])
    links.new(ramp.outputs["Color"], mix.inputs["Color2"])
    return mix.outputs["Color"]


def add_elevation_tint(mat, base_socket, low_m: float, high_m: float, strength: float):
    """Shade the ground by world height, low ground dark and cool, high ground bright.

    Contour lines alone read as cracks in a flat surface, because that is very nearly what
    this terrain is: ~1-3% grade, so the true shading gradient across a 60 m frame is a few
    percent and the eye takes the lines for texture rather than relief. A hypsometric tint
    -- the same trick as a shaded relief map -- restates the same world-Z field as a
    continuous luminance ramp, which the eye does read as a surface. Geometry is untouched,
    so the vehicle stays welded to the ground it is actually driving on.

    The ramp multiplies the albedo, so the ground texture survives underneath it. Because
    every stop is <= 1.0 it can only darken; raise ``--sun-energy`` to compensate.
    """
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    geometry = nodes.new("ShaderNodeNewGeometry")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    # Map the measured elevation window onto 0..1 so the ramp spends its whole range on the
    # relief that is actually in frame, rather than on the patch's full +/-0.6 m.
    map_range = nodes.new("ShaderNodeMapRange")
    map_range.inputs["From Min"].default_value = float(low_m)
    map_range.inputs["From Max"].default_value = float(high_m)
    map_range.inputs["To Min"].default_value = 0.0
    map_range.inputs["To Max"].default_value = 1.0
    map_range.clamp = True

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.34, 0.38, 0.46, 1.0)
    mid = ramp.color_ramp.elements.new(0.55)
    mid.color = (0.80, 0.76, 0.68, 1.0)
    ramp.color_ramp.elements[2].position = 1.0
    ramp.color_ramp.elements[2].color = (1.0, 0.97, 0.88, 1.0)

    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MULTIPLY"
    mix.inputs["Fac"].default_value = float(np.clip(strength, 0.0, 1.0))

    links.new(geometry.outputs["Position"], separate.inputs["Vector"])
    links.new(separate.outputs["Z"], map_range.inputs["Value"])
    links.new(map_range.outputs["Result"], ramp.inputs["Fac"])
    links.new(base_socket, mix.inputs["Color1"])
    links.new(ramp.outputs["Color"], mix.inputs["Color2"])
    return mix.outputs["Color"]


def add_relief_shading(mat, exaggeration: float):
    """Light the terrain as if its slopes were ``exaggeration`` times steeper.

    A cartographer's hillshade. This terrain's true grade is 1-3%, so a real sun produces a
    few percent of shading across the whole frame and the surface reads as flat no matter
    where the sun is put. Tilting the *shading normal* -- N' = normalize(k*nx, k*ny, nz) --
    makes hills and hollows light and shadow the way the eye expects, while the geometry,
    and therefore the vehicle's contact with it, is exactly the physics mesh.

    Only the normal fed to the BSDF changes; nothing moves. The patch must be smooth-shaded
    or every 1.96 m triangle turns into its own hard facet under the amplified normal.
    """
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")

    geometry = nodes.new("ShaderNodeNewGeometry")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    scale_x = nodes.new("ShaderNodeMath")
    scale_x.operation = "MULTIPLY"
    scale_x.inputs[1].default_value = float(exaggeration)
    scale_y = nodes.new("ShaderNodeMath")
    scale_y.operation = "MULTIPLY"
    scale_y.inputs[1].default_value = float(exaggeration)
    combine = nodes.new("ShaderNodeCombineXYZ")
    normalize = nodes.new("ShaderNodeVectorMath")
    normalize.operation = "NORMALIZE"

    links.new(geometry.outputs["Normal"], separate.inputs["Vector"])
    links.new(separate.outputs["X"], scale_x.inputs[0])
    links.new(separate.outputs["Y"], scale_y.inputs[0])
    links.new(scale_x.outputs["Value"], combine.inputs["X"])
    links.new(scale_y.outputs["Value"], combine.inputs["Y"])
    links.new(separate.outputs["Z"], combine.inputs["Z"])
    links.new(combine.outputs["Vector"], normalize.inputs[0])
    links.new(normalize.outputs["Vector"], bsdf.inputs["Normal"])


def terrain_elevation_range(terrain_meshes, lo, hi, pad_m: float = 6.0):
    """World-Z range of the terrain patch inside the framed XY window.

    Taken over the patch's own vertices rather than its bounding box: the exported patch is
    500 m across and its full +/-0.6 m range would spread the elevation ramp over relief
    that is nowhere near the shot.
    """
    zs = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in terrain_meshes:
        eval_obj = obj.evaluated_get(depsgraph)
        matrix = eval_obj.matrix_world
        for vert in eval_obj.data.vertices:
            world = matrix @ vert.co
            if lo[0] - pad_m <= world.x <= hi[0] + pad_m and lo[1] - pad_m <= world.y <= hi[1] + pad_m:
                zs.append(world.z)
    if not zs:
        return None
    return float(min(zs)), float(max(zs))


def make_ground_material(
    name,
    tint,
    roughness,
    texture_path: Path | None,
    tile_m: float,
    contour_interval_m: float = 0.0,
    contour_strength: float = 0.55,
    elevation_range: tuple[float, float] | None = None,
    elevation_tint_strength: float = 0.0,
    relief_exaggeration: float = 0.0,
):
    """Ground material: tiled albedo image multiplied by a tint, or a flat colour."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = roughness
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.0

    if texture_path is None:
        rgb = nodes.new("ShaderNodeRGB")
        rgb.outputs["Color"].default_value = tint
        base_socket = rgb.outputs["Color"]
    else:
        image = bpy.data.images.load(str(texture_path), check_existing=True)
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = image
        tex.extension = "REPEAT"

        # Object coordinates are in metres for both the substituted plane and the heightmap
        # patch (both are unit-scaled and centred at the origin), so the tile size below is
        # a real length rather than a UV fraction.
        coord = nodes.new("ShaderNodeTexCoord")
        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (1.0 / tile_m, 1.0 / tile_m, 1.0 / tile_m)

        mix = nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MULTIPLY"
        mix.inputs["Fac"].default_value = 1.0
        mix.inputs["Color2"].default_value = tint

        links.new(coord.outputs["Object"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        links.new(tex.outputs["Color"], mix.inputs["Color1"])
        base_socket = mix.outputs["Color"]

    if elevation_range is not None and float(elevation_tint_strength) > 0.0:
        base_socket = add_elevation_tint(
            mat, base_socket, elevation_range[0], elevation_range[1], elevation_tint_strength
        )

    if float(contour_interval_m) > 0.0:
        base_socket = add_elevation_contours(mat, base_socket, contour_interval_m, contour_strength)

    if float(relief_exaggeration) > 1.0:
        add_relief_shading(mat, float(relief_exaggeration))

    links.new(base_socket, bsdf.inputs["Base Color"])
    return mat


def assign_material(obj, mat) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(mat)


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


def freeze_chrono_frame() -> list:
    """Detach Chrono's frame_change_post handler.

    The add-on's handler deletes and rebuilds every object in ``chrono_frame_objects``
    on each frame change *including the one Cycles fires internally when rendering*,
    which would throw away the materials assigned below. It must therefore be frozen
    across each render() call and re-attached before the next frame_set.
    """
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


def enable_chrono_import() -> None:
    bpy.ops.preferences.addon_enable(module="chrono_import")
    # bpy.ops.import_chrono.data stays truthy even when the operator is unregistered,
    # so check the type registry instead.
    if not hasattr(bpy.types, "IMPORT_CHRONO_OT_data"):
        raise RuntimeError("chrono_import add-on did not register (run Blender with --factory-startup)")


def split_frame_objects(frame_collection):
    """Partition the imported Chrono objects into (terrain meshes, vehicle meshes)."""
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
        # The terrain patch is the only object that spans tens of metres; the largest
        # vehicle part (the chassis hull) is under 5 m.
        if "patch" in names or max(obj.dimensions) > 20:
            terrain_meshes.append(obj)
        else:
            vehicle_meshes.append(obj)

    # Fallback for unpatched Chrono HMMWV assets: obj_import leaves most chassis
    # submeshes loose in the scene collection, with only one small mesh linked to the
    # Chassis body. Re-parent the loose pieces so they follow the chassis.
    chassis_parent = next((obj for obj in frame_objects if obj.name == "Chassis body"), None)
    extra_chassis_meshes = []
    if chassis_parent is not None:
        for obj in bpy.data.objects:
            if obj.type != "MESH" or obj.parent is not None:
                continue
            if any(collection.name.startswith("chrono") for collection in obj.users_collection):
                continue
            # Our own ground plane / soil surface also live loose in the scene
            # collection; they must not be parented to the chassis.
            if obj.name.startswith("render_"):
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

    if not vehicle_meshes:
        vehicle_meshes = meshes
    return terrain_meshes, vehicle_meshes


class VehicleStyler:
    """Re-applies the HMMWV materials after each Chrono frame rebuild."""

    def __init__(self) -> None:
        self.body = make_material("hmmwv_desert_olive", (0.47, 0.50, 0.42, 1.0), 0.76)
        self.dark = make_material("dark_rubber_metal", (0.055, 0.055, 0.052, 1.0), 0.84)
        self.metal = make_material("muted_axle_metal", (0.30, 0.31, 0.30, 1.0), 0.58, 0.05)
        self.glass = make_material("smoked_glass", (0.08, 0.10, 0.11, 1.0), 0.42, 0.02)
        self.light = make_material("muted_headlight", (0.92, 0.80, 0.48, 1.0), 0.35)

    def apply(self, vehicle_meshes) -> None:
        for obj in vehicle_meshes:
            obj.hide_render = False
            obj.hide_viewport = False
            names = object_names(obj)
            max_dim = max(obj.dimensions)
            if any(token in names for token in ("wi_", ".wi", "window", "mirror")):
                assign_material(obj, self.glass)
            elif any(token in names for token in ("hlight", "signal", "lites", "turn_signa")):
                assign_material(obj, self.light)
            elif "chassis" in names:
                assign_material(obj, self.body)
            elif "spindle" in names and max_dim > 0.45:
                assign_material(obj, self.dark)
            elif any(token in names for token in ("susp", "steering", "upright", "lca", "uca")):
                assign_material(obj, self.metal)
            elif max_dim < 0.18:
                assign_material(obj, self.metal)
            else:
                assign_material(obj, self.dark)


def import_surface_mesh(path: Path):
    """Import the splashsurf soil surface (PLY or OBJ) and return the new object."""
    before = set(bpy.data.objects)
    suffix = path.suffix.lower()
    if suffix == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    elif suffix == ".obj":
        # Chrono/splashsurf write Z-up data; the OBJ importer defaults to Y-up.
        bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z")
    else:
        raise ValueError(f"unsupported surface mesh format: {path}")
    created = [obj for obj in bpy.data.objects if obj not in before]
    if not created:
        raise RuntimeError(f"no object created importing {path}")
    return created[0]


class SoilSequence:
    """Swaps the CRM soil mesh for each rendered frame.

    The SPH surface is re-reconstructed every exported frame, so a single static mesh
    would freeze the ruts while the vehicle drove over them. Each swap fully removes
    the previous mesh datablock -- Blender does not free it just because the object is
    gone, and at ~19 MB per mesh a 180-frame sequence would otherwise exhaust RAM.
    """

    def __init__(self, directory: Path, pattern: str, frame_count: int, material, smooth: bool = True):
        self.directory = Path(directory)
        self.pattern = pattern
        self.frame_count = int(frame_count)
        self.material = material
        self.smooth = bool(smooth)
        self.obj = None

    def set_frame(self, frame: int) -> None:
        index = int(min(max(frame, 0), self.frame_count - 1))
        path = self.directory / (self.pattern % index)
        if not path.is_file():
            raise FileNotFoundError(f"CRM surface mesh not found: {path}")
        if self.obj is not None:
            mesh_data = self.obj.data
            bpy.data.objects.remove(self.obj, do_unlink=True)
            bpy.data.meshes.remove(mesh_data)
            self.obj = None
        soil = import_surface_mesh(path)
        soil.name = "render_crm_soil_surface"
        assign_material(soil, self.material)
        for polygon in soil.data.polygons:
            polygon.use_smooth = self.smooth
        self.obj = soil


def undisturbed_soil_z(soil_obj, bin_m: float = 0.01) -> float:
    """Height of the flat, undriven soil surface.

    The mesh bounding box is useless here: its top is the tallest berm thrown up by a
    tyre, which is well above the ground the vehicle is actually driving on, and a filler
    plane parked there hides the whole scene. The undisturbed surface is instead the
    single most common vertex height, since it covers nearly the whole window. The closed
    volume contributes an equally large flat *bottom* face, so take the highest height
    that is still strongly populated rather than the global mode.
    """
    matrix = soil_obj.matrix_world
    heights = np.array([(matrix @ v.co).z for v in soil_obj.data.vertices], dtype=np.float64)
    if heights.size == 0:
        raise RuntimeError("CRM soil mesh has no vertices")
    lo, hi = float(heights.min()), float(heights.max())
    if hi - lo < bin_m:
        return hi
    counts, edges = np.histogram(heights, bins=max(int((hi - lo) / bin_m), 2))
    populated = np.flatnonzero(counts >= 0.3 * counts.max())
    top_bin = int(populated[-1])
    return float(0.5 * (edges[top_bin] + edges[top_bin + 1]))


def add_crm_filler_plane(soil_obj, center: Vector, mat_ground, args):
    """Hide the CRM slab's crop faces behind a large plane at the undisturbed soil level.

    splashsurf reconstructs a *closed volume*, so the cropped render window has vertical
    walls at its edges. Above ~35 deg of camera elevation they sit outside the frame, but
    the whole point of lowering the camera is to rake light across the ruts -- which drags
    those walls into shot as a striped band. Widening the SPH window instead would mean
    re-running the (25-minute) CRM eval, so the cheap fix is the one the rigid path already
    uses: an identically-textured plane, parked just under the soil top so the slab still
    reads as sitting on the ground rather than floating above it.
    """
    if not args.crm_filler_plane or soil_obj is None:
        return None
    top_z = undisturbed_soil_z(soil_obj)
    bpy.ops.mesh.primitive_plane_add(
        size=max(args.terrain_radius_m * 2.0, 20.0),
        location=(center.x, center.y, top_z - float(args.crm_filler_drop_m)),
    )
    filler = bpy.context.object
    filler.name = "render_crm_filler_plane"
    assign_material(filler, mat_ground)
    print(f"CRM_FILLER soil_top_z={top_z:.3f} plane_z={top_z - float(args.crm_filler_drop_m):.3f}")
    return filler


def hide_scene_clutter(keep) -> None:
    """Hide everything that is neither vehicle, terrain, nor ours.

    Chrono exports coordinate-frame glyphs and helper geometry alongside the bodies. Like
    the terrain material, hiding them once at setup is not enough: the add-on's frame
    handler recreates them -- visible -- on every frame change, and a green Y-axis glyph
    sitting near the origin can fill an entire orthographic frame.

    Our own additions are named ``render_*`` and are exempt, which matters because the
    trajectory tubes are curves and would otherwise be caught by the same rule that hides
    Chrono's glyphs.
    """
    for obj in bpy.data.objects:
        if obj.name.startswith("render_"):
            continue
        if obj.type in {"CURVE", "GPENCIL", "GREASEPENCIL"}:
            obj.hide_render = True
            obj.hide_viewport = True
        elif obj.type == "MESH" and obj not in keep:
            obj.hide_render = True
            obj.hide_viewport = True


class TerrainStyler:
    """Re-applies terrain material and visibility after each Chrono frame rebuild.

    The add-on's frame handler deletes and rebuilds every object in
    ``chrono_frame_objects`` on each frame change, throwing away whatever
    ``build_terrain`` did -- and ``build_terrain`` only runs once, at setup. For rigid and
    CRM the patch is hidden and substituted, so losing that is invisible in the render.
    For **bumpy the exported patch IS the terrain**, so without re-applying, every frame
    after the first renders it with Chrono's default flat grey instead of the ground
    material (and drops the elevation contours with it).
    """

    def __init__(self, kind: str, mat_ground, smooth: bool = False):
        self.kind = kind
        self.mat_ground = mat_ground
        self.smooth = smooth

    def apply(self, terrain_meshes) -> None:
        for obj in terrain_meshes:
            if self.kind == "bumpy":
                obj.hide_render = False
                obj.hide_viewport = False
                assign_material(obj, self.mat_ground)
                if self.smooth:
                    # Interpolated vertex normals only -- no geometry change. Without this the
                    # patch's 1.96 m triangles each shade as a flat facet, which relief
                    # shading then amplifies into a field of hard plates.
                    obj.data.polygons.foreach_set("use_smooth", [True] * len(obj.data.polygons))
                    obj.data.update()
            else:
                obj.hide_render = True
                obj.hide_viewport = True


def build_terrain(scene_info: dict, terrain_meshes, center: Vector, args, frame_bounds=None):
    kind = scene_info["terrain"]
    tint, roughness = TERRAIN_TINTS[kind]
    texture_path = None
    if args.ground_texture and str(args.ground_texture).lower() != "none":
        texture_path = Path(args.ground_texture).expanduser()
        if not texture_path.is_file():
            raise FileNotFoundError(f"ground texture not found: {texture_path}")

    elevation_range = None
    if float(args.elevation_tint_strength) > 0.0 and kind == "bumpy":
        if args.elevation_range_m is not None:
            elevation_range = (float(args.elevation_range_m[0]), float(args.elevation_range_m[1]))
        elif frame_bounds is not None:
            elevation_range = terrain_elevation_range(terrain_meshes, *frame_bounds)
        if elevation_range is not None and elevation_range[1] - elevation_range[0] < 1e-3:
            elevation_range = None
        if elevation_range is not None:
            print(f"TERRAIN_ELEVATION {elevation_range[0]:.3f} {elevation_range[1]:.3f} m")

    mat_ground = make_ground_material(
        f"{kind}_terrain",
        tint,
        roughness,
        texture_path,
        float(args.ground_texture_scale_m),
        contour_interval_m=float(args.contour_interval_m),
        contour_strength=float(args.contour_strength),
        elevation_range=elevation_range,
        elevation_tint_strength=float(args.elevation_tint_strength),
        relief_exaggeration=float(args.relief_exaggeration) if kind == "bumpy" else 0.0,
    )

    styler = TerrainStyler(kind, mat_ground, smooth=float(args.relief_exaggeration) > 1.0)
    if kind == "bumpy":
        if not terrain_meshes:
            raise RuntimeError("bumpy scene has no exported terrain patch mesh")
        styler.apply(terrain_meshes)
        return None, styler

    styler.apply(terrain_meshes)

    if kind == "rigid":
        bpy.ops.mesh.primitive_plane_add(
            size=max(args.terrain_radius_m * 2.0, 20.0),
            location=(center.x, center.y, scene_info.get("terrain_surface_z_m", 0.0) - 0.005),
        )
        ground = bpy.context.object
        ground.name = "render_flat_terrain_plane"
        assign_material(ground, mat_ground)
        return None, styler

    sequence = scene_info.get("crm_surface_sequence")
    if sequence is not None:
        soil_seq = SoilSequence(
            sequence["dir"],
            sequence["pattern"],
            sequence["frame_count"],
            mat_ground,
            smooth=(args.soil_shading == "smooth"),
        )
        soil_seq.set_frame(0)
        add_crm_filler_plane(soil_seq.obj, center, mat_ground, args)
        return soil_seq, styler

    surface = scene_info.get("crm_surface")
    if not surface:
        raise RuntimeError("crm scene has no crm_surface entry")
    mesh_path = Path(surface["mesh_path"])
    if not mesh_path.is_file():
        raise FileNotFoundError(f"CRM surface mesh not found: {mesh_path}")
    soil = import_surface_mesh(mesh_path)
    soil.name = "render_crm_soil_surface"
    assign_material(soil, mat_ground)
    # splashsurf's marching-cubes output is faceted at the cell scale; smooth shading
    # reads as soil rather than as a voxel field.
    for polygon in soil.data.polygons:
        polygon.use_smooth = args.soil_shading == "smooth"
    add_crm_filler_plane(soil, center, mat_ground, args)
    return None, styler


def make_polyline(name: str, xy: np.ndarray, z: float, mat, bevel_depth: float):
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


class TrajectoryOverlay:
    """Green reference path (static) plus the red driven path, grown frame by frame."""

    def __init__(self, rollout_path: Path, terrain_z: float, draw: bool = True, radius_scale: float = 1.0):
        with np.load(rollout_path) as data:
            # (x, y, yaw) per step. The yaw column is what the chase camera rides on.
            self.pose = np.asarray(data["pose"], dtype=float)
            self.ref_xy = np.asarray(data["ref_pose"][:, :2], dtype=float)
            self.pose_xy = self.pose[:, :2]
        self.num_steps = int(min(len(self.ref_xy), len(self.pose_xy)))
        # A chase camera can be driven from the pose log without any of the overlay being
        # drawn, which is the point of `draw`: tubes sized for a 30 m-wide top-down frame
        # become foreground pipes in a close-up, lying exactly along the ruts they are
        # meant to let you see.
        self.draw = bool(draw)
        self.radius_scale = float(radius_scale)
        self.marker = None
        self.trace = None
        if not self.draw:
            return

        self.mat_ref = make_material("reference_trajectory_green", REFERENCE_COLOR, 0.45, emission_strength=0.35)
        self.mat_rollout = make_material("rollout_trajectory_red", ROLLOUT_COLOR, 0.5, emission_strength=0.35)
        self.ref_z = terrain_z + 0.06
        self.rollout_z = terrain_z + 0.10

        make_polyline("render_reference_trajectory", self.ref_xy, self.ref_z, self.mat_ref, 0.075 * self.radius_scale)

        # Marker at the reference *target* for the current step. A static path alone
        # cannot show along-track lag; this can.
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.32 * self.radius_scale, location=(0.0, 0.0, self.ref_z))
        self.marker = bpy.context.object
        self.marker.name = "render_reference_marker"
        assign_material(self.marker, self.mat_ref)
        for polygon in self.marker.data.polygons:
            polygon.use_smooth = True

    def set_frame(self, frame: int) -> None:
        if not self.draw:
            return
        step = int(np.clip(frame, 0, self.num_steps - 1))
        ref = self.ref_xy[step]
        self.marker.location = (float(ref[0]), float(ref[1]), self.ref_z)

        # Rebuild rather than trim an existing curve: Blender's bevel_factor trimming
        # maps along the *evaluated* curve length, which does not line up with the sim
        # step index.
        if self.trace is not None:
            curve_data = self.trace.data
            bpy.data.objects.remove(self.trace, do_unlink=True)
            bpy.data.curves.remove(curve_data)
            self.trace = None
        if step >= 1:
            self.trace = make_polyline(
                "render_rollout_trajectory",
                self.pose_xy[: step + 1],
                self.rollout_z,
                self.mat_rollout,
                0.055 * self.radius_scale,
            )

    def bounds(self):
        pts = np.vstack([self.ref_xy, self.pose_xy])
        return pts.min(axis=0), pts.max(axis=0)


class FollowPath:
    """Smoothed vehicle track (x, y, yaw) that drives the chase camera.

    The camera is driven from the eval's own pose log rather than from the per-frame
    vehicle mesh bounds, for two reasons. The bounding box of the wheels and suspension
    breathes as the body rolls, which a rigidly locked camera turns into shake; and a
    heading recovered from consecutive bounding boxes is far noisier than the yaw the
    simulator already recorded.

    Smoothing is a centred moving average, so it needs the whole track up front -- which
    is exactly what the rollout ``.npz`` gives us, and why this is precomputed instead of
    being updated frame by frame. Yaw is averaged as a unit vector so the window does not
    blow up when the heading crosses +/-pi.
    """

    def __init__(self, pose: np.ndarray, smooth_frames: int = 9):
        pose = np.asarray(pose, dtype=np.float64)
        if pose.ndim != 2 or pose.shape[1] < 3:
            raise ValueError(f"expected an (N, 3) pose array, got {pose.shape}")
        self.num_steps = int(pose.shape[0])
        self.xy = self._smooth(pose[:, :2], smooth_frames)
        heading = np.stack([np.cos(pose[:, 2]), np.sin(pose[:, 2])], axis=1)
        heading = self._smooth(heading, smooth_frames)
        self.yaw = np.arctan2(heading[:, 1], heading[:, 0])

    @staticmethod
    def _smooth(values: np.ndarray, window: int) -> np.ndarray:
        window = max(int(window), 1)
        if window <= 1 or values.shape[0] <= 2:
            return values.copy()
        half = window // 2
        # Edge-clamped padding: a zero-padded or reflected window would drag the camera
        # away from the vehicle at the first and last frames.
        padded = np.pad(values, ((half, half), (0, 0)), mode="edge")
        kernel = np.ones(2 * half + 1) / float(2 * half + 1)
        return np.stack([np.convolve(padded[:, c], kernel, mode="valid") for c in range(values.shape[1])], axis=1)

    def sample(self, frame: int) -> tuple[float, float, float]:
        step = int(np.clip(frame, 0, self.num_steps - 1))
        return float(self.xy[step, 0]), float(self.xy[step, 1]), float(self.yaw[step])


def follow_path_from_meshes(vehicle_meshes) -> tuple[float, float, float] | None:
    """Fallback pose for --follow without a rollout npz: bbox centre, heading unknown."""
    lo, hi = mesh_bounds(vehicle_meshes)
    centre = (lo + hi) * 0.5
    return float(centre.x), float(centre.y), 0.0


def camera_basis(azimuth_deg: float, elevation_deg: float):
    """Return (view offset direction, screen-right, screen-up) for the given angles."""
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    offset = Vector(
        (
            math.cos(azimuth) * math.cos(elevation),
            math.sin(azimuth) * math.cos(elevation),
            math.sin(elevation),
        )
    )
    right = Vector((-math.sin(azimuth), math.cos(azimuth), 0.0))
    up = offset.cross(right).normalized()
    return offset, right, up


def auto_frame_ortho_scale(overlay: TrajectoryOverlay, args, target: Vector) -> float:
    """Smallest ortho width that fits both trajectories plus a margin."""
    _, right, up = camera_basis(args.camera_azimuth_deg, args.camera_elevation_deg)
    lo, hi = overlay.bounds()
    corners = [
        Vector((float(x), float(y), target.z))
        for x in (lo[0] - args.frame_margin_m, hi[0] + args.frame_margin_m)
        for y in (lo[1] - args.frame_margin_m, hi[1] + args.frame_margin_m)
    ]
    rel = [corner - target for corner in corners]
    half_width = max(abs(v.dot(right)) for v in rel)
    half_height = max(abs(v.dot(up)) for v in rel)
    aspect = float(args.width) / float(args.height)
    return max(2.0 * half_width, 2.0 * half_height * aspect)


def camera_summary(args, use_perspective: bool, ortho_scale: float) -> str:
    lens = f"persp lens={args.cam_lens_mm:.0f}mm" if use_perspective else f"ortho_scale={ortho_scale:.2f}"
    if not args.follow:
        return f"camera=fixed {lens}"
    azimuth = args.follow_azimuth_deg if args.follow_mode == "chase" else args.camera_azimuth_deg
    return (
        f"camera=follow/{args.follow_mode} {lens} az={azimuth:.0f} "
        f"el={args.follow_elevation_deg:.0f} d={args.follow_distance_m:.1f} "
        f"smooth={args.follow_smooth_frames}"
    )


def set_render_engine(scene, samples: int, use_gpu: bool = False) -> None:
    """Configure Cycles.

    GPU is worth asking for on a long sequence -- it is roughly an order of magnitude
    faster here than CPU, and it also sidesteps the sustained-CPU-load segfaults this
    box is prone to. Falls back to CPU if no CUDA/OptiX device is present.
    """
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons.get("cycles")
    device = "CPU"
    if use_gpu and prefs:
        cycles_prefs = prefs.preferences
        for backend in ("OPTIX", "CUDA"):
            try:
                cycles_prefs.compute_device_type = backend
            except TypeError:
                continue
            cycles_prefs.get_devices()
            if any(dev.type == backend for dev in cycles_prefs.devices):
                for dev in cycles_prefs.devices:
                    dev.use = dev.type == backend
                device = "GPU"
                print(f"CYCLES_DEVICE {backend}: " + ", ".join(d.name for d in cycles_prefs.devices if d.use))
                break
        if device == "CPU":
            print("CYCLES_DEVICE no GPU backend available, falling back to CPU")
    elif prefs:
        try:
            prefs.preferences.compute_device_type = "NONE"
        except TypeError:
            pass
    scene.cycles.device = device
    scene.cycles.samples = int(samples)
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2


def main() -> None:
    args = parse_args()
    scene_info = resolve_scene(args)
    kind = scene_info["terrain"]
    state_count = int(scene_info.get("state_file_count", 1))
    terrain_z = float(scene_info.get("terrain_surface_z_m", 0.0))

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    enable_chrono_import()
    bpy.ops.import_chrono.data(filepath=scene_info["assets_script"], setting_materials=True, setting_merge=False)

    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = max(state_count - 1, 0)
    first_frame = args.anim_start if args.animate else args.frame
    scene.frame_set(int(first_frame))
    bpy.context.view_layer.update()

    frame_collection = bpy.data.collections.get("chrono_frame_objects")
    if frame_collection is None:
        raise RuntimeError("chrono_frame_objects collection not found after import")

    styler = VehicleStyler()
    terrain_meshes, vehicle_meshes = split_frame_objects(frame_collection)
    styler.apply(vehicle_meshes)

    hide_scene_clutter(set(vehicle_meshes) | set(terrain_meshes))

    lo, hi = mesh_bounds(vehicle_meshes)
    vehicle_center = (lo + hi) * 0.5

    overlay = None
    if args.rollout is not None:
        overlay = TrajectoryOverlay(
            args.rollout.expanduser().resolve(),
            terrain_z,
            draw=not args.no_trajectory,
            radius_scale=float(args.trajectory_radius_scale),
        )
        overlay.set_frame(int(first_frame))

    fixed_camera = overlay is not None and not args.follow
    if overlay is not None:
        # Centre the substituted ground / CRM filler plane on the whole trajectory, not on
        # the start pose: a chase camera ends the rollout tens of metres from where it began
        # and would otherwise drive off the edge of a plane centred on frame 0.
        traj_lo, traj_hi = overlay.bounds()
        ground_center = Vector((
            float((traj_lo[0] + traj_hi[0]) * 0.5),
            float((traj_lo[1] + traj_hi[1]) * 0.5),
            terrain_z,
        ))
    else:
        ground_center = Vector((vehicle_center.x, vehicle_center.y, terrain_z))

    frame_bounds = overlay.bounds() if overlay is not None else None
    soil_sequence, terrain_styler = build_terrain(
        scene_info, terrain_meshes, ground_center, args, frame_bounds=frame_bounds
    )

    offset_dir, _, _ = camera_basis(args.camera_azimuth_deg, args.camera_elevation_deg)
    cam_data = bpy.data.cameras.new("render_camera")
    cam = bpy.data.objects.new("render_camera", cam_data)
    scene.collection.objects.link(cam)
    cam_data.clip_end = 5000
    scene.camera = cam

    if fixed_camera:
        target = ground_center + Vector((0.0, 0.0, 0.6))
        ortho_scale = auto_frame_ortho_scale(overlay, args, target)
        # An orthographic camera has no near/far falloff, but it does clip: push it
        # back far enough that nothing in a wide frame lands behind the lens.
        distance = max(args.camera_distance_m, ortho_scale * 1.5)
    else:
        target = vehicle_center + Vector((0.0, 0.0, 0.15))
        ortho_scale = float(args.ortho_scale_m)
        distance = float(args.camera_distance_m)

    # A chase shot is perspective unless told otherwise: an orthographic camera removes
    # every depth cue, so the ruts it is meant to show read as a flat painted texture.
    use_perspective = bool(args.perspective) or (args.follow and not args.follow_ortho)
    if use_perspective:
        cam_data.type = "PERSP"
        cam_data.lens = float(args.cam_lens_mm)
    else:
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = ortho_scale

    follow_path = None
    if args.follow:
        if overlay is not None:
            follow_path = FollowPath(overlay.pose, args.follow_smooth_frames)
        else:
            print("FOLLOW no --rollout npz: falling back to per-frame vehicle bounds, heading fixed")

    def place_fixed_camera(look_target: Vector) -> None:
        cam.location = look_target + offset_dir * distance
        look_at(cam, look_target)

    def place_follow_camera(frame: int, frame_vehicle_meshes) -> Vector:
        if follow_path is not None:
            x, y, yaw = follow_path.sample(frame)
        else:
            x, y, yaw = follow_path_from_meshes(frame_vehicle_meshes)
        frame_lo, frame_hi = mesh_bounds(frame_vehicle_meshes)
        center_z = float(((frame_lo + frame_hi) * 0.5).z)
        if args.follow_mode == "chase":
            # Bearing of the camera *from* the vehicle: 180 deg puts it behind, and the
            # offset swings it round to the three-quarter view.
            azimuth = math.degrees(yaw) + 180.0 + float(args.follow_azimuth_deg)
        else:
            azimuth = float(args.camera_azimuth_deg)
        direction, _, _ = camera_basis(azimuth, args.follow_elevation_deg)
        look_target = Vector((
            x + math.cos(yaw) * float(args.follow_lead_m),
            y + math.sin(yaw) * float(args.follow_lead_m),
            center_z + float(args.follow_target_z_offset_m),
        ))
        cam.location = look_target + direction * float(args.follow_distance_m)
        look_at(cam, look_target)
        return look_target

    sun_data = bpy.data.lights.new("sun_key", "SUN")
    sun = bpy.data.objects.new("sun_key", sun_data)
    scene.collection.objects.link(sun)
    # A SUN's rotation points its -Z down the beam, so an X rotation of (90 - elevation)
    # puts the light that many degrees above the horizon. The sun is placed once and stays
    # world-fixed even in follow mode -- a key light that swung round with a chase camera
    # would flip the rut shadows every time the vehicle turned. --camera-azimuth-deg is
    # therefore still meaningful while following: it is the bearing the sun is offset from.
    sun.rotation_euler = (
        math.radians(90.0 - float(args.sun_elevation_deg)),
        0.0,
        math.radians(args.camera_azimuth_deg + float(args.sun_azimuth_offset_deg)),
    )
    sun_data.energy = float(args.sun_energy)
    # A tighter disc keeps the rut shadows crisp instead of smearing them at grazing angles.
    sun_data.angle = math.radians(0.8)

    fill_light = None
    # World-axis offset, so the fill keeps the same bearing on the vehicle as the sun does
    # rather than swinging round with a chase camera.
    fill_offset = Vector((-6.0, -7.0, 9.0))
    if float(args.fill_energy) > 0.0:
        area_data = bpy.data.lights.new("softbox", "AREA")
        fill_light = bpy.data.objects.new("softbox", area_data)
        scene.collection.objects.link(fill_light)
        area_data.energy = float(args.fill_energy)
        area_data.size = 7.0

    def update_fill(look_target: Vector) -> None:
        """Keep the softbox with the vehicle.

        It is a 7 m area light a few metres away, so unlike the sun it does not carry: left
        at the start pose it stops lifting the vehicle's shadow side within a second or two
        of driving.
        """
        if fill_light is None:
            return
        fill_light.location = look_target + fill_offset
        look_at(fill_light, look_target)

    if args.follow:
        update_fill(place_follow_camera(int(first_frame), vehicle_meshes))
    else:
        place_fixed_camera(target)
        update_fill(target)

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background:
        background.inputs["Color"].default_value = (0.62, 0.70, 0.80, 1.0)
        background.inputs["Strength"].default_value = float(args.sky_strength)

    set_render_engine(scene, args.samples, use_gpu=bool(args.gpu))
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

    if args.save_blend is not None:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.save_blend.expanduser().resolve()))

    handlers = freeze_chrono_frame()
    atexit.register(restore_chrono_handlers, handlers)

    def render_frame(frame: int, output_path: Path) -> None:
        nonlocal handlers
        output_path.parent.mkdir(parents=True, exist_ok=True)
        restore_chrono_handlers(handlers)
        scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        handlers = freeze_chrono_frame()
        # The handler just deleted and rebuilt every chrono_frame_object with default
        # materials, so re-collect and re-style before rendering.
        frame_terrain_meshes, frame_vehicle_meshes = split_frame_objects(frame_collection)
        styler.apply(frame_vehicle_meshes)
        # The handler just rebuilt the terrain patch and Chrono's helper glyphs too, with
        # default materials and visibility, so both policies have to be re-applied.
        hide_scene_clutter(set(frame_vehicle_meshes) | set(frame_terrain_meshes))
        terrain_styler.apply(frame_terrain_meshes)
        if soil_sequence is not None:
            soil_sequence.set_frame(frame)
        if overlay is not None:
            overlay.set_frame(frame)
        if args.follow:
            update_fill(place_follow_camera(frame, frame_vehicle_meshes))
        elif not fixed_camera:
            # No trajectory to frame against, so keep the vehicle centred.
            frame_lo, frame_hi = mesh_bounds(frame_vehicle_meshes)
            place_fixed_camera((frame_lo + frame_hi) * 0.5 + Vector((0.0, 0.0, 0.15)))
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)

    if not args.animate:
        output = args.output
        if output is None:
            output = Path(scene_info["_base_dir"]) / "renders" / f"hmmwv_{kind}_frame{args.frame:04d}.png"
        output = output.expanduser().resolve()
        render_frame(int(args.frame), output)
        print(f"RENDERED {output}")
        print(f"TERRAIN {kind} {camera_summary(args, use_perspective, ortho_scale)} "
              f"camera={tuple(round(v, 2) for v in cam.location)}")
        restore_chrono_handlers(handlers)
        return

    anim_dir = args.anim_dir.expanduser().resolve()
    anim_dir.mkdir(parents=True, exist_ok=True)
    last = args.anim_end if args.anim_end is not None else max(state_count - 1, 0)
    rendered = 0
    for frame in range(int(args.anim_start), int(last) + 1):
        output_path = anim_dir / f"frame{frame:04d}.png"
        if args.skip_existing and output_path.is_file():
            continue
        render_frame(frame, output_path)
        rendered += 1
        print(f"FRAME {frame} -> {output_path}", flush=True)
    print(f"RENDERED_SEQUENCE {anim_dir} frames={rendered} range=[{args.anim_start},{last}]")
    print(f"TERRAIN {kind} {camera_summary(args, use_perspective, ortho_scale)}")
    restore_chrono_handlers(handlers)


if __name__ == "__main__":
    main()
