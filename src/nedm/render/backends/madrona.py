"""Madrona's batch renderer, driven directly from PyTorch.

The compiled renderer is the C++ engine inside ``madrona_mjx`` (MIT, as is Madrona). That
project wraps it in JAX and MuJoCo MJX. Neither is used here: the engine's own interface is
mesh arrays at setup, then GPU pointers to per-world positions, rotations and camera poses,
and that is what this backend passes. NVIDIA only.

Build it once (``scripts/render/madrona/build.sh``) and point ``NEDM_MADRONA_BUILD`` at the
build directory, or put that directory on ``PYTHONPATH``.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

import numpy as np
import torch

PLANE, MESH = 0, 7                      # MuJoCo geom types, which the engine reuses
PLANE_HALF_SIZE = 200.0
LIGHT_DIRECTION = (-0.57735026, 0.57735026, -0.57735026)    # the Newton backend's default light


def _engine():
    build = os.environ.get("NEDM_MADRONA_BUILD")
    if build and build not in sys.path:
        sys.path.insert(0, build)
    try:
        return importlib.import_module("_madrona_mjx_batch_renderer")
    except ImportError as error:
        raise ImportError(
            "the Madrona renderer module was not found. Build it with scripts/render/madrona/build.sh "
            "and set NEDM_MADRONA_BUILD to its build directory.") from error


class Frames:
    """The images of one render call. Views into the renderer's buffers: copy to keep them."""

    def __init__(self, rgba: torch.Tensor, depth: torch.Tensor):
        self._rgba, self._depth = rgba, depth

    def rgb_torch(self) -> torch.Tensor:
        """(N, C, H, W, 3) uint8 on the GPU, zero-copy."""
        return self._rgba[..., :3]

    def depth_torch(self) -> torch.Tensor:
        """(N, C, H, W) float32 on the GPU, zero-copy."""
        return self._depth[..., 0]

    @property
    def rgb(self) -> np.ndarray:
        return self.rgb_torch().cpu().numpy()

    @property
    def depth(self) -> np.ndarray:
        return self.depth_torch().cpu().numpy()


def _wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., 3:], q[..., :3]], dim=-1)


class MadronaBackend:
    def __init__(self, scene, num_worlds: int, width: int, height: int, cameras: int, fov_deg: float,
                 device: Any, shadows: bool, sky: tuple[int, int, int], ray_trace: bool = True) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("the Madrona backend needs an NVIDIA GPU")
        self.device = torch.device("cuda:0")
        self.num_worlds, self.cameras, self.num_bodies = int(num_worlds), int(cameras), len(scene.bodies)
        self.interop = "zero-copy"

        # One "geom" per drawn thing. Bodies first, in scene order, then everything that never moves.
        meshes, types, data_ids, sizes, rgba, static_xf = [], [], [], [], [], []

        def add_mesh(mesh):
            vertices, faces = mesh.arrays()
            meshes.append((vertices, faces))
            types.append(MESH)
            data_ids.append(len(meshes) - 1)
            sizes.append((1.0, 1.0, 1.0))
            rgba.append((*mesh.color, 1.0))

        for body in scene.bodies:
            add_mesh(body)
        if scene.ground is not None:
            ground = scene.ground
            types.append(PLANE)
            data_ids.append(-1)
            sizes.append((PLANE_HALF_SIZE, PLANE_HALF_SIZE, 1.0))
            rgba.append((*ground.color, 1.0))
            static_xf.append((0.0, 0.0, ground.height, 0.0, 0.0, 0.0, 1.0))
            tiles = ground.tile_mesh()          # one object, not one per tile
            if tiles is not None:
                add_mesh(tiles)
                static_xf.append((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        for mesh, xf in scene.statics:
            add_mesh(mesh)
            static_xf.append(xf)
        if len(types) == 1:
            # The engine reads out of bounds when a world holds exactly one object (found
            # 2026-10-07, "illegal memory access" at the first render). Give it a second one
            # that no ray will find: a 0.1 mm triangle 10 km below the origin.
            from nedm.render.scene import MeshBody
            speck = np.array([[0.0, 0.0, 0.0], [1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0]], dtype=np.float32)
            add_mesh(MeshBody("padding", vertices=speck, faces=np.array([[0, 1, 2]], dtype=np.int32)))
            static_xf.append((0.0, 0.0, -1e4, 0.0, 0.0, 0.0, 1.0))
        self.num_geoms = len(types)

        vertex_offsets = np.cumsum([0] + [len(v) for v, _ in meshes])[:-1]
        face_offsets = np.cumsum([0] + [len(f) for _, f in meshes])[:-1]
        i32 = lambda a: np.ascontiguousarray(a, dtype=np.int32)             # noqa: E731
        f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)           # noqa: E731
        engine = _engine()
        self._renderer = engine.MadronaBatchRenderer(
            gpu_id=0,
            mesh_vertices=f32(np.concatenate([v for v, _ in meshes])),
            mesh_faces=i32(np.concatenate([f for _, f in meshes])),
            mesh_vertex_offsets=i32(vertex_offsets),
            mesh_face_offsets=i32(face_offsets),
            mesh_texcoords=np.zeros((0, 2), dtype=np.float32),
            mesh_texcoord_offsets=i32(np.full(len(meshes), -1)),
            mesh_texcoord_num=i32(np.zeros(len(meshes))),
            geom_types=i32(types),
            geom_groups=i32(np.zeros(self.num_geoms)),
            geom_data_ids=i32(data_ids),
            geom_sizes=f32(sizes),
            geom_mat_ids=i32(np.full(self.num_geoms, -1)),      # -1: a flat material from geom_rgba
            geom_rgba=f32(rgba),
            mat_rgba=np.zeros((0, 4), dtype=np.float32),
            mat_tex_ids=np.zeros((0, 10), dtype=np.int32),
            tex_data=np.zeros((0,), dtype=np.uint8),
            tex_offsets=np.zeros((0,), dtype=np.int32),
            tex_widths=np.zeros((0,), dtype=np.int32),
            tex_heights=np.zeros((0,), dtype=np.int32),
            tex_nchans=np.zeros((0,), dtype=np.int32),
            num_lights=1,
            num_cams=self.cameras,
            num_worlds=self.num_worlds,
            batch_render_view_width=int(width),
            batch_render_view_height=int(height),
            cam_fovy=f32([fov_deg] * self.cameras),
            enabled_geom_groups=i32([0]),
            add_cam_debug_geo=False,
            use_rt=ray_trace,
        )

        w, dev = self.num_worlds, self.device
        static = torch.tensor(static_xf, dtype=torch.float32, device=dev).reshape(-1, 7)
        self._static_pos = static[:, :3].expand(w, -1, 3).contiguous()
        self._static_rot = _wxyz(static[:, 3:]).expand(w, -1, 4).contiguous()
        self._sizes = torch.tensor(sizes, dtype=torch.float32, device=dev).expand(w, -1, 3).contiguous()
        self._materials = torch.full((w, self.num_geoms), -1, dtype=torch.int32, device=dev)
        colors = (np.array(rgba)[:, :3] * 255).astype(np.uint32)
        packed = (colors[:, 0] << 16) + (colors[:, 1] << 8) + colors[:, 2]
        self._colors = torch.tensor(packed.astype(np.int64), device=dev).to(torch.uint32).expand(w, -1).contiguous()
        self._light_dir = torch.tensor(LIGHT_DIRECTION, device=dev).expand(w, 3).contiguous()
        self._light_pos = torch.zeros(w, 3, device=dev)
        self._light_directional = torch.ones(w, dtype=torch.bool, device=dev)
        self._light_shadow = torch.full((w,), bool(shadows), dtype=torch.bool, device=dev)
        self._light_cutoff = torch.full((w,), 45.0, device=dev)
        # Our cameras look along -Z with +Y up, as MuJoCo's do. Madrona's look along +Y with +Z up.
        self._to_y_forward = torch.tensor([-0.70710678, 0.0, 0.0, 0.70710678], device=dev)    # xyzw, -90 deg about X
        self._sky = torch.tensor(sky, dtype=torch.uint8, device=dev)
        self._started = False

    def render(self, body_q: torch.Tensor, camera: torch.Tensor) -> Frames:
        from nedm.render.poses import compose
        position = torch.cat([body_q[..., :3], self._static_pos], dim=1).contiguous()
        rotation = torch.cat([_wxyz(body_q[..., 3:]), self._static_rot], dim=1).contiguous()
        turn = torch.zeros(7, device=self.device)
        turn[3:] = self._to_y_forward
        cam = compose(camera, turn.expand_as(camera))
        cam_position, cam_rotation = cam[..., :3].contiguous(), _wxyz(cam[..., 3:]).contiguous()
        if self._started:
            self._renderer.render(position, rotation, cam_position, cam_rotation)
        else:
            self._renderer.init(position, rotation, cam_position, cam_rotation, self._materials, self._colors,
                                self._sizes, self._light_pos, self._light_dir, self._light_directional,
                                self._light_shadow, self._light_cutoff)
            self._rgba = self._renderer.rgb_tensor().to_torch()
            self._depth = self._renderer.depth_tensor().to_torch()
            self._started = True
        # The ray tracer leaves rays that hit nothing black, with depth 0. Paint them the sky color.
        self._rgba[..., :3][self._depth[..., 0] == 0] = self._sky
        torch.cuda.synchronize()
        return Frames(self._rgba, self._depth)
