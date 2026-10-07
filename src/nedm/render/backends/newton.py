"""Newton's Warp ray tracer, used as a renderer only.

Newton holds the scene (one copy of the bodies per world, plus what every world shares) and
the cameras. No solver is created and nothing is simulated. It is pure Warp kernels, so it
runs wherever Warp does: NVIDIA through the stock wheel, AMD Instinct through AMD's ROCm
port, and CPU. Apache-2.0, as are Newton and Warp.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class Frames:
    """The images of one render call. Views into the renderer's buffers: copy to keep them."""

    def __init__(self, color, depth):
        self._color, self._depth = color, depth

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


class NewtonBackend:
    def __init__(self, scene, num_worlds: int, width: int, height: int, cameras: int, fov_deg: float,
                 device: Any, shadows: bool, sky: tuple[int, int, int]) -> None:
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
        self._rays = self._camera.utils.compute_pinhole_camera_rays(width, height, [math.radians(fov_deg)] * self.cameras)
        self._color = self._camera.utils.create_color_image_output(width, height, self.cameras)
        self._depth = self._camera.utils.create_depth_image_output(width, height, self.cameras)
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
        self._camera.update(self.state, self._camera_xf, self._rays, color_image=self._color,
                            depth_image=self._depth, clear_data=self._clear)
        self._wp.synchronize()
        return Frames(self._color, self._depth)

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
