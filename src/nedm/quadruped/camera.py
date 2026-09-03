"""Camera and CRM sprite rendering.

The sprite settings below are SETTLED and are constants rather than CLI flags.
Each carries the measurement that fixed it."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np

from nedm import chrono_crm_compat as crm_compat

# --------------------------------------------------------------------------
# SETTLED SPRITE PARAMETERS. These were CLI flags while they were being found;
# they are constants now, each with the measurement that fixed it.
#
# render_particle_spacing is a RESAMPLING TARGET, not a sprite size:
#   ChOptixEngine.cpp EstimateFsiSphRenderCount computes
#     render_count = num_markers * (source_spacing / render_particle_spacing)^3
#   so SMALLER draws MORE sprites, cubically. Raising it to "enlarge" the
#   sprites decimated the bed instead and exposed background monotonically:
#   bright 30.3% at 0.01 -> 45.8% at 0.02 -> 68.2% at 0.026.
SPH_RENDER_SPACING = 0.01
# Sprite SIZE is the mesh's own scale, applied in fsi_sph_render.cu from
# template_scale. The options struct has no size field, so it is baked in.
SPRITE_SCALE = 2.0
# 24 pre-rotated copies. The options struct has no ORIENTATION field either, but
# we own the mesh list and the renderer cycles it, so N rotated copies are
# orientation variety. Position jitter and sprite overlap both failed to break
# the triangular lattice, because a shared attitude cannot be decorrelated by
# moving things. A longer list also lengthens the cycle: with 3 meshes the row
# profile had a clean 3.0 px period, with 24 it moved to 24.0 px exactly.
SPRITE_VARIANTS = 24
# 0.005 breaks the position lattice that survives rotation. At 0.026 sprites on
# 0.02 spacing there is 30% overlap, so this cannot reopen the gaps.
SPRITE_JITTER = 0.005
# TRUE, and the upstream demo passes false while shipping meshes that CONTAIN
# normals, so its own published output is unlit. Loading them moved the terrain
# mean from 18.8 to 128.2 and dark pixels from 45.4% to 0.3%; the single largest
# change in the whole rendering effort.
SPRITE_NORMALS = True
# The sprite path IGNORES the attached ChVisualMaterial: setting diffuse and
# ambient to a grey-brown produced a frame numerically IDENTICAL to white, to
# one decimal on all three channels. Left white because that is what the
# renderer uses regardless. Recorded as an upstream finding.
SOIL_DIFFUSE = (1.0, 1.0, 1.0)
SOIL_AMBIENT = (1.0, 1.0, 1.0)
# --------------------------------------------------------------------------


def attach_sph_rendering(chrono, sens, manager, terrain, args):
    """Make the CRM soil visible to Chrono::Sensor.

    SPH markers are not visual shapes, so a camera sees nothing by default and
    the robot appears to tumble in a void. The sensor manager renders particles
    as sprites instead, via AttachFsiSphSystem; see Chrono's own
    src/demos/sensor/demo_SEN_CRM_Rendering.cpp. This is the real mechanism and
    it supersedes the static proxy box, which showed where the surface was but
    could not show the soil deform.
    """
    attach = getattr(manager, "AttachFsiSphSystem", None)
    if attach is None:
        return "AttachFsiSphSystem unavailable in this build"
    getter = getattr(terrain, "GetFluidSystemSPH", None) or getattr(terrain, "GetFsiSystemSPH", None)
    if getter is None:
        return "no SPH system accessor on CRMTerrain"
    try:
        sph = getter()
        opts_cls = getattr(sens, "ChFsiSphRenderOptions", None)
        if opts_cls is None:
            # REFUSE rather than fall back. A default-constructed options object
            # is a NULL CONFIGURATION -- ChFsiSphRender.h documents sprite_shapes
            # as required (default empty) and render_particle_spacing as "must be
            # positive to render particles" (default 0.f) -- so attaching without
            # it returns a valid handle and renders NOTHING. Measured: 86% sky,
            # 0.2% dark, indistinguishable from not attaching at all.
            #
            # This branch used to attach anyway and report "attached (defaults)",
            # which is precisely the silent no-op this project spent two days
            # cataloguing, sitting in our own code. The absence of this class
            # means the build lacks patches/0001-expose-fsi-sph-render-options
            # -- so say that, and stop.
            raise RuntimeError(
                "ChFsiSphRenderOptions is absent from pychrono.sensor, so the "
                "only reachable configuration is the one that renders nothing. "
                "This build is missing patches/0001-expose-fsi-sph-render-"
                "options.patch. Apply it to chrono-src and rebuild the sensor "
                "module and python bindings; see "
                "docs/state/machines/crm-rendering-handoff.md.")
        opts = opts_cls()
        # BOTH of these are required for anything to be drawn, and the defaults
        # are a null configuration -- ChFsiSphRender.h documents sprite_shapes as
        # "Required" (default: empty) and render_particle_spacing as "Must be
        # positive to render particles" (default: 0.f). Attaching with a
        # default-constructed options object returns a valid handle and renders
        # NOTHING, which is what an 886,611-particle run produced before the
        # options struct was exposed: 86% flat sky, 0.2% dark pixels.
        # render_particle_spacing is a RESAMPLING TARGET, not a sprite size.
        # ChOptixEngine.cpp EstimateFsiSphRenderCount:
        #     render_count = num_markers * (source_spacing / render_particle_spacing)^3
        # so SMALLER values draw MORE sprites, cubically. Setting it ABOVE the
        # actual particle spacing DECIMATES the bed and exposes background, which
        # is why bright% rose monotonically 30.3 -> 45.8 -> 68.2 as we raised it
        # from 0.01 to 0.02 to 0.026.
        spacing = float(SPH_RENDER_SPACING)
        n_shapes = 0
        if hasattr(opts, "sprite_shapes"):
            # Match demo_SEN_CRM_Rendering.cpp EXACTLY rather than substituting a
            # primitive. It loads three regolith OBJs as ChVisualShapeTriangleMesh
            # with a white material; a ChVisualShapeSphere was accepted here and
            # rendered nothing, so the sprite path likely handles meshes only.
            # Comparing a first render against a known-good reference is worth
            # more than comparing against a variant we invented.
            mat = chrono.ChVisualMaterial()
            mat.SetAmbientColor(chrono.ChColor(*SOIL_AMBIENT))
            mat.SetDiffuseColor(chrono.ChColor(*SOIL_DIFFUSE))
            # ChFsiSphRenderOptions exposes no orientation control -- only
            # sprite_shapes, sprite_position_jitter and render_particle_spacing.
            # But we own the mesh LIST, and the renderer cycles it, so N
            # pre-rotated copies ARE orientation variety. Position jitter and
            # sprite overlap both failed to break the triangular lattice because
            # you cannot decorrelate a shared attitude by moving things; rotating
            # the vertex data before handing it over is what does it. A longer
            # list also lengthens the cycle, which is what produced the measured
            # period-3 banding against exactly three meshes.
            rng = random.Random(20260903)
            srcs = [f"models/regolith/particle_{i}.obj" for i in (1, 2, 3)]
            for k in range(int(SPRITE_VARIANTS)):
                path = chrono.GetChronoDataFile(srcs[k % len(srcs)])
                if not Path(path).is_file():
                    continue
                mesh = chrono.ChTriangleMeshConnected.CreateFromWavefrontFile(
                    path, bool(SPRITE_NORMALS), True)
                if SPRITE_VARIANTS > len(srcs):
                    # Verified to MUTATE the vertex data, not just return a copy:
                    # a vertex read before and after Transform differs.
                    q = chrono.QuatFromAngleZ(rng.uniform(0, 6.283185))
                    q = q * chrono.QuatFromAngleX(rng.uniform(0, 6.283185))
                    q = q * chrono.QuatFromAngleY(rng.uniform(0, 6.283185))
                    mesh.Transform(chrono.ChVector3d(0, 0, 0), chrono.ChMatrix33d(q))
                if SPRITE_SCALE != 1.0:
                    # SPRITE SIZE IS THE MESH'S OWN SCALE, not render_particle_spacing.
                    # fsi_sph_render.cu builds each OptiX instance transform from
                    # template_scale, i.e. the template mesh; render_particle_spacing
                    # only sets HOW MANY sprites are resampled (see below). The options
                    # struct has no size field, but we own the meshes, so scaling the
                    # vertex data is how sprite size gets set.
                    mesh.Transform(chrono.ChVector3d(0, 0, 0),
                                   chrono.ChMatrix33d(float(SPRITE_SCALE)))
                shape = chrono.ChVisualShapeTriangleMesh()
                shape.SetMesh(mesh)
                shape.SetName(f"RegolithMesh{k}")
                shape.SetMutable(False)
                shape.AddMaterial(mat)
                opts.sprite_shapes.append(shape)
            n_shapes = len(opts.sprite_shapes)
        if hasattr(opts, "sprite_position_jitter"):
            opts.sprite_position_jitter = chrono.ChVector3f(
                float(SPRITE_JITTER), float(SPRITE_JITTER), 0.0)
        if hasattr(opts, "render_particle_spacing"):
            opts.render_particle_spacing = spacing
        handle = attach(sph, opts)
        if handle is not None and handle < 0:
            return f"FAILED: AttachFsiSphSystem returned {handle} (no-op)"
        return (f"attached handle={handle}, sprite_shapes={n_shapes}, "
                f"render_particle_spacing={spacing}")
    except Exception as exc:  # noqa: BLE001
        return f"FAILED: {type(exc).__name__}: {exc}"



# Camera height above the soil for the overhead view. 1.6 m with a 58 deg FOV
# frames roughly 1.8 m across, i.e. the robot plus about a metre of soil.
OVERHEAD_HEIGHT = 1.6


def _camera_pose(args, soil_top: float):
    """(eye, target, fov) for the requested camera mode.

    --cam-eye / --cam-target override any mode. Modes:
      overhead  straight down, TRACKING the robot in XY. Looking down is what
                makes footprints and disturbed soil legible; the three-quarter
                view cannot show them because it looks across the surface
                rather than into it.
      follow    the three-quarter view, tracking. A fixed frame loses a walking
                robot: the first CRM walk left shot at t=7.40 of an 8 s clip.
      side      the same pose, NOT tracking, for a gait profile.
    """
    mode = getattr(args, "camera", "follow")
    if args.cam_eye and args.cam_target:
        return np.array(args.cam_eye), np.array(args.cam_target), math.radians(58.0)
    if mode == "overhead":
        # Straight down. The target sits directly under the eye; the mount
        # translates both together, so the robot stays centred.
        # Aimed at the robot's spawn XY, not the three-quarter view's look-at
        # point. Tuned by looking at a frame: targeting x=0.30 put the robot 0.28 m
        # off-axis, which at 1.6 m and 58 deg is a quarter of the half-width, and
        # it showed -- the robot sat well left of centre with dead soil to the right.
        eye = np.array([0.0, 0.0, soil_top + OVERHEAD_HEIGHT])
        target = np.array([0.0, 0.0, soil_top])
        return eye, target, math.radians(58.0)
    eye = np.array([-1.15, -1.45, soil_top + 0.62])
    target = np.array([0.30, 0.0, soil_top + 0.08])
    return eye, target, math.radians(58.0)


def attach_camera(chrono, system, args, soil_top: float, terrain=None):
    try:
        import pychrono.sensor as sens
    except Exception as exc:  # noqa: BLE001
        return None, f"pychrono.sensor unavailable ({type(exc).__name__})", None
    mount = chrono.ChBody()
    mount.SetFixed(True)
    mount.EnableCollision(False)
    system.AddBody(mount)

    manager = sens.ChSensorManager(system)
    manager.scene.SetAmbientLight(chrono.ChVector3f(0.35, 0.35, 0.40))
    if hasattr(manager.scene, "AddDirectionalLight"):
        manager.scene.AddDirectionalLight(chrono.ChColor(1.0, 0.95, 0.85),
                                          math.radians(55.0), math.radians(120.0))
    else:
        manager.scene.AddPointLight(chrono.ChVector3f(2, -2.5, 3),
                                    chrono.ChColor(1.0, 0.95, 0.85), 25.0)
    crm_compat.set_solid_background(manager.scene, chrono.ChVector3f(0.55, 0.68, 0.85))

    # Attach the SPH system BEFORE the camera exists, matching the C++ demo
    # (AttachFsiSphSystem at demo line 295, camera built after). The attach calls
    # ReconstructScenes() internally, so a camera added first is built against a
    # scene that has no FSI source in it, and the later rebuild does not retrofit
    # the pipeline that camera already holds. Attaching after AddSensor produced
    # frames byte-identical to attaching not at all -- options populated, handle
    # 0, no error, and nothing drawn.
    sph_note = attach_sph_rendering(chrono, sens, manager, terrain, args) if terrain is not None else "n/a"
    print(f"sph rendering: {sph_note}")

    eye, target, fov = _camera_pose(args, soil_top)
    pose = chrono.ChFramed(chrono.ChVector3d(*eye), _look_at(chrono, eye, target))
    cam = sens.ChCameraSensor(mount, args.video_fps, pose,
                              args.video_width, args.video_height, fov)
    cam.SetLag(0.0)
    cam.SetCollectionWindow(0.0)
    frames = Path(args.out) / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    save = getattr(sens, "ChFilterSave", None)
    if save is None:
        return None, "ChFilterSave unavailable", None
    cam.PushFilter(save(str(frames) + "/"))
    manager.AddSensor(cam)
    return manager, f"frames -> {frames} | sph: {sph_note}", mount


def _look_at(chrono, eye, target):
    """Chrono::Sensor cameras look down +X with +Z up."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    up_hint = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.999 else np.array([0.0, 1.0, 0.0])
    left = np.cross(up_hint, f); left /= np.linalg.norm(left)
    up = np.cross(f, left)
    r = np.column_stack([f, left, up])
    tr = float(np.trace(r))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r))); j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
        q = [0.0] * 4
        q[0] = (r[k, j] - r[j, k]) / s
        q[i + 1], q[j + 1], q[k + 1] = 0.25 * s, (r[j, i] + r[i, j]) / s, (r[k, i] + r[i, k]) / s
    return chrono.ChQuaterniond(*q)

