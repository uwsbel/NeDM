"""Newton's Warp ray tracer, used as a renderer only.

Newton holds the scene (one copy of the bodies per world, plus what every world shares) and
the cameras. No solver is created and nothing is simulated. It is pure Warp kernels, so it
runs wherever Warp does: NVIDIA through the stock wheel, AMD Instinct through AMD's ROCm
port, and CPU. Apache-2.0, as are Newton and Warp.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Sequence

import numpy as np

# where a ray of an invalid pixel starts (camera frame, m): far behind the camera, pointing away from everything
INVALID_RAY_ORIGIN = 1.0e4
LIGHT_TYPES = {"spot": 0, "directional": 1, "point": 2}     # 2: Newton's point-light branch (any id but 0 and 1)
_KERNELS: dict = {}


def _mask_kernel():
    """Warp kernel: colour black (alpha 255) and depth 0 at the invalid pixels of every world."""
    if "mask" not in _KERNELS:
        import warp as wp

        @wp.kernel(enable_backward=False)
        def mask_invalid(valid: wp.array3d(dtype=wp.int32), color: wp.array4d(dtype=wp.uint32),
                         depth: wp.array4d(dtype=wp.float32)):
            w, c, y, x = wp.tid()
            if valid[c, y, x] == 0:
                color[w, c, y, x] = wp.uint32(0xFF000000)
                depth[w, c, y, x] = 0.0

        _KERNELS["mask"] = mask_invalid
    return _KERNELS["mask"]


class Frames:
    """The images of one render call. Views into the renderer's buffers: copy to keep them."""

    def __init__(self, color, depth, hdr=None, shape_index=None):
        self._color, self._depth, self._hdr, self._shape_index = color, depth, hdr, shape_index

    def hdr_torch(self):
        """(N, C, H, W, 3) float32 linear colour before the 8-bit clamp, on the render device (``hdr=True`` only)."""
        import warp as wp
        if self._hdr is None:
            raise RuntimeError("build the renderer with hdr=True")
        h = wp.to_torch(self._hdr)
        return h.reshape(*h.shape[:4], 3)

    @property
    def rgb(self) -> np.ndarray:
        """(N, C, H, W, 3) uint8."""
        c = self._color.numpy()
        return c.view(np.uint8).reshape(*c.shape, 4)[..., :3]

    @property
    def depth(self) -> np.ndarray:
        """(N, C, H, W) float32, distance along the ray in metres, 0 where nothing was hit."""
        return self._depth.numpy()

    def rgb_torch(self):
        """(N, C, H, W, 3) uint8 on the render device, zero-copy."""
        import torch
        import warp as wp
        c = wp.to_torch(self._color)
        return c.view(torch.uint8).reshape(*c.shape, 4)[..., :3]

    def depth_torch(self):
        """(N, C, H, W) float32 on the render device, zero-copy."""
        import warp as wp
        return wp.to_torch(self._depth)

    def shape_index_torch(self):
        """(N, C, H, W) int64 Newton shape index per pixel (-1: nothing hit), on the render device (``segment=True``
        only)."""
        import torch
        import warp as wp
        if self._shape_index is None:
            raise RuntimeError("build the renderer with segment=True")
        s = wp.to_torch(self._shape_index).view(torch.int32).long()
        return torch.where(s < 0, torch.full_like(s, -1), s)          # 0xFFFFFFFF (no hit) -> -1


class NewtonBackend:
    """``camera_models``: None (every camera a pinhole of ``fov_deg``: Newton's own rays, the original behaviour) or one
    model per camera (:class:`~nedm.render.cameras.PinholeCamera` / :class:`~nedm.render.cameras.CalibratedCamera`);
    pixels without a ray in their model are drawn black with depth 0 (:attr:`valid`). ``hdr``: also keep the linear
    colour before the 8-bit clamp (:meth:`Frames.hdr_torch`). ``segment``: also keep Newton's shape index per pixel
    (:meth:`Frames.shape_index_torch`; :attr:`shape_body` maps it to the body index of the scene, statics after the
    bodies)."""

    def __init__(self, scene, num_worlds: int, width: int, height: int, cameras: int, fov_deg: float,
                 device: Any, shadows: bool, sky: tuple[int, int, int], camera_models: Sequence | None = None,
                 hdr: bool = False, segment: bool = False) -> None:
        import newton
        import warp as wp
        from newton.sensors import SensorTiledCamera
        self._wp, self._newton = wp, newton
        self.num_worlds, self.cameras, self.num_bodies = int(num_worlds), int(cameras), len(scene.bodies)

        bodies = newton.ModelBuilder()
        for body in scene.bodies:
            vertices, faces = body.arrays()
            bodies.add_shape_mesh(bodies.add_body(label=body.name), mesh=newton.Mesh(vertices, faces.reshape(-1)),
                                  color=body.color)
        b = newton.ModelBuilder()
        for _ in range(self.num_worlds):
            b.begin_world()
            b.add_builder(bodies)
            b.end_world()
        if scene.ground is not None:            # shared by every world
            ground = scene.ground
            b.add_ground_plane(height=ground.height, color=ground.color)
            half = (ground.tile_size or 0.0) / 2
            for center in ground.tile_centers():
                b.add_shape_box(-1, xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
                                hx=half, hy=half, hz=ground.tile_thickness / 2, color=ground.tile_color)
        for mesh, xf in scene.statics:
            vertices, faces = mesh.arrays()
            b.add_shape_mesh(-1, xform=wp.transform(wp.vec3(*xf[:3]), wp.quat(*xf[3:])),
                             mesh=newton.Mesh(vertices, faces.reshape(-1)), color=mesh.color)
        self.model = b.finalize(device=device)
        self.state = self.model.state()
        self.device = self.model.device

        self._camera = SensorTiledCamera(model=self.model)
        self._camera.utils.create_default_light(enable_shadows=shadows)
        self.width, self.height, self.fov_deg = int(width), int(height), float(fov_deg)
        self.camera_models = None
        self.valid = None            # (C, H, W) bool numpy; None: pinholes of fov_deg (every pixel has a ray)
        self._valid = None           # its int32 Warp copy, only when some pixel is invalid
        self._ray_cache: dict = {}   # camera-model key -> (rays, valid, valid int32, ray_z): intrinsics DR per call
        if camera_models is None:
            self._rays = self._camera.utils.compute_pinhole_camera_rays(width, height,
                                                                        [math.radians(fov_deg)] * self.cameras)
        else:
            self.set_camera_models(camera_models)
        self._color = self._camera.utils.create_color_image_output(width, height, self.cameras)
        self._depth = self._camera.utils.create_depth_image_output(width, height, self.cameras)
        self._hdr = self._camera.utils.create_hdr_color_image_output(width, height, self.cameras) if hdr else None
        self._shape_index = None
        if segment:
            self._shape_index = self._camera.utils.create_shape_index_image_output(width, height, self.cameras)
            sb = self.model.shape_body.numpy()
            nb = self.num_bodies
            lut = np.empty(len(sb), np.int64)
            statics = 0
            for k, b in enumerate(sb):                  # bodies are world-major: global body = world * nb + local
                if b >= 0:
                    lut[k] = int(b) % nb
                else:
                    lut[k] = nb + statics
                    statics += 1
            self.shape_body = lut                       # Newton shape index -> scene body index (statics: nb + k)
        r, g, bl = sky
        self._clear = SensorTiledCamera.ClearData(clear_color=0xFF000000 | (bl << 16) | (g << 8) | r)
        self._camera_xf = wp.zeros((self.cameras, self.num_worlds), dtype=wp.transformf, device=self.device)
        self._bvh_built = False
        self.interop: str | None = None      # "zero-copy" or "host-copy"

    def render(self, body_q, camera) -> Frames:
        self._put(self.state.body_q, body_q.reshape(self.num_worlds * self.num_bodies, 7))
        self._put(self._camera_xf, camera.transpose(0, 1))
        geometry = self._newton.geometry
        if self._bvh_built:
            geometry.refit_bvh_shape(self.model, self.state)
        else:
            geometry.build_bvh_shape(self.model, self.state)
            self._bvh_built = True
        if self._hdr is None and self._shape_index is None:       # the original call, unchanged
            self._camera.update(self.state, self._camera_xf, self._rays, color_image=self._color,
                                depth_image=self._depth, clear_data=self._clear)
        else:
            self._camera.update(self.state, self._camera_xf, self._rays, color_image=self._color,
                                depth_image=self._depth, clear_data=self._clear, hdr_color_image=self._hdr,
                                shape_index_image=self._shape_index)
        if self._valid is not None:
            self._wp.launch(_mask_kernel(), dim=(self.num_worlds, self.cameras, self.height, self.width),
                            inputs=[self._valid, self._color, self._depth], device=self.device)
        self._wp.synchronize()
        return Frames(self._color, self._depth, self._hdr, self._shape_index)

    # ---- camera models -------------------------------------------------------------------------------------------
    RAY_CACHE = 16                  # ray sets kept on the device (2 cameras x 448 x 448: 9.6 MB each)

    def set_camera_models(self, models: Sequence) -> None:
        """One camera model per camera (rays shared by all worlds; call again to change intrinsics between calls).
        Pinholes get Newton's own pinhole rays (their own fov each); calibrated cameras get their undistorted rays.
        The last :attr:`RAY_CACHE` ray sets are kept, so cycling through a few intrinsics draws costs no ray
        computation after the first use of each."""
        models = list(models)
        if len(models) != self.cameras:
            raise ValueError(f"{len(models)} camera models for {self.cameras} cameras")
        key = tuple(repr(m) for m in models)
        hit = self._ray_cache.get(key)
        if hit is None:
            hit = self._compute_rays(models)
            if len(self._ray_cache) >= self.RAY_CACHE:
                self._ray_cache.pop(next(iter(self._ray_cache)))
            self._ray_cache[key] = hit
        self._rays, self.valid, self._valid, self.ray_z = hit
        self.camera_models = models

    def _compute_rays(self, models):
        from nedm.render.cameras import camera_rays
        wp = self._wp
        fovs = [math.radians(m.fovy_deg) if m.kind == "pinhole" else 1.0 for m in models]
        rays = self._camera.utils.compute_pinhole_camera_rays(self.width, self.height, fovs)
        dirs, valid = camera_rays(models, self.width, self.height)
        custom = [c for c, m in enumerate(models) if m.kind != "pinhole"]
        if custom:
            r = rays.numpy()                                        # (C, H, W, 2, 3): origin, direction
            for c in custom:
                ok = valid[c]
                r[c, :, :, 0] = np.where(ok[..., None], 0.0, np.array([0.0, 0.0, INVALID_RAY_ORIGIN]))
                r[c, :, :, 1] = np.where(ok[..., None], dirs[c], np.array([0.0, 0.0, 1.0]))
            rays = wp.array(r.astype(np.float32), dtype=wp.vec3f, device=self.device)
        valid_wp = None if valid.all() else wp.array(valid.astype(np.int32), dtype=wp.int32, device=self.device)
        # z-depth per unit ray length (camera -z component of the unit ray), 0 at invalid pixels
        ray_z = np.where(valid, -dirs[..., 2], 0.0).astype(np.float32)
        return rays, valid, valid_wp, ray_z

    # ---- lights (shared by all worlds: Newton has no per-world lights) -------------------------------------------
    def set_lights(self, lights: Sequence[dict], shadows: bool | None = None, ambient: bool | None = None) -> None:
        """Replace the lights. Each light: ``{"type": "directional" | "spot" | "point", "direction": (3,) (the way the
        light travels; directional and spot), "position": (3,) (spot, point), "shadow": bool}``. Newton lights have no
        intensity or colour (each adds albedo * max(0, n.L), spot/point times 1/(1 + 0.02 d^2)). ``shadows``: the
        global shadow switch; ``ambient``: Newton's fixed hemispheric ambient term on/off.

        A directional light may carry ``"intensity"`` w in (0, 1) (and ``"center"`` (3,), default the origin): it is
        drawn as a POINT light at distance d = sqrt((1/w - 1) / 0.02) m from ``center``, against the light's direction,
        whose attenuation 1/(1 + 0.02 d^2) is w there (an approximation: over +-0.25 m around ``center`` the direction
        varies by atan(0.25/d) and the intensity by about +-0.5 d / (d^2 + 50) relative; d = 7.1 m at w = 0.5)."""
        wp = self._wp
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            rc = self._camera.render_context
        lights = [self._dimmed(dict(lt)) for lt in lights]
        n = len(lights)
        if n == 0:
            rc.lights_active = None
        else:
            dev = self.device
            rc.lights_active = wp.array([True] * n, dtype=wp.bool, device=dev)
            rc.lights_type = wp.array([LIGHT_TYPES[lt.get("type", "directional")] for lt in lights], dtype=wp.int32,
                                      device=dev)
            rc.lights_cast_shadow = wp.array([bool(lt.get("shadow", True)) for lt in lights], dtype=wp.bool,
                                             device=dev)
            pos = [np.asarray(lt.get("position", (0.0, 0.0, 0.0)), float) for lt in lights]
            way = [np.asarray(lt.get("direction", (0.0, 0.0, -1.0)), float) for lt in lights]
            rc.lights_position = wp.array([wp.vec3f(*p) for p in pos], dtype=wp.vec3f, device=dev)
            rc.lights_orientation = wp.array([wp.vec3f(*(d / np.linalg.norm(d))) for d in way], dtype=wp.vec3f,
                                             device=dev)
        if shadows is not None:
            self._camera.render_config.enable_shadows = bool(shadows)
        if ambient is not None:
            self._camera.render_config.enable_ambient_lighting = bool(ambient)

    @staticmethod
    def _dimmed(light: dict) -> dict:
        """A directional light with ``intensity`` < 1 -> the equivalent far point light (see :meth:`set_lights`)."""
        w = float(light.pop("intensity", 1.0))
        center = np.asarray(light.pop("center", (0.0, 0.0, 0.0)), float)
        if w >= 1.0:
            return light
        if w <= 0.0 or light.get("type", "directional") != "directional":
            raise ValueError(f"intensity {w}: only directional lights, 0 < intensity <= 1")
        d = np.asarray(light["direction"], float)
        dist = math.sqrt((1.0 / w - 1.0) / 0.02)
        return {"type": "point", "position": center - dist * d / np.linalg.norm(d),
                "shadow": light.get("shadow", True)}

    def _put(self, dst, src) -> None:
        """Write a torch tensor into a Warp array, in place when both are on the render device."""
        wp = self._wp
        src = src.detach().float().contiguous()
        if self.interop != "host-copy":
            try:
                wp.copy(dst, wp.from_torch(src, dtype=dst.dtype))
                self.interop = "zero-copy"
                return
            except Exception as error:  # noqa: BLE001
                print(f"torch to warp without a copy is unavailable here ({type(error).__name__}: "
                      f"{str(error)[:120]}), copying through the host instead")
                self.interop = "host-copy"
        dst.assign(src.cpu().numpy())
