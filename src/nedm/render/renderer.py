"""Batch renderer: meshes and per-world poses in, one image per world out.

Newton's Warp ray tracer is used as a renderer only. Newton holds the scene (one copy of the
bodies per world, plus whatever every world shares) and the cameras. No solver is created
and nothing is simulated. The NRD model stays the dynamics.

The renderer is pure Warp kernels, so it runs wherever Warp does: NVIDIA through the stock
wheel, AMD Instinct through AMD's ROCm port, and CPU. See ``docs/batch_rendering.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

Color = tuple[float, float, float]
_GROUND = (0.60, 0.55, 0.46)
_TILE = (0.47, 0.42, 0.34)


@dataclass
class MeshBody:
    """One rigid body drawn as a triangle mesh, given in the body's own frame."""

    name: str
    vertices: np.ndarray | None = None      # (V, 3)
    faces: np.ndarray | None = None         # (F, 3)
    path: str | Path | None = None          # anything trimesh loads, instead of vertices/faces
    color: Color = (0.8, 0.8, 0.8)
    scale: float = 1.0

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.path is not None:
            import trimesh
            mesh = trimesh.load(str(self.path), force="mesh", process=False)
            vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        else:
            vertices, faces = np.asarray(self.vertices), np.asarray(self.faces)
        return (vertices * self.scale).astype(np.float32), faces.reshape(-1).astype(np.int32)


class Scene:
    """What one world contains, and what all worlds share.

    Two ways to describe the moving part:

    * :meth:`from_urdf` for an articulated robot. Poses are then given as joint coordinates
      and Newton runs the forward kinematics.
    * :meth:`from_meshes` for a set of rigid bodies whose transforms the caller computes,
      for example a chassis and four wheels, or the links of an arm with its own kinematics.
    """

    def __init__(self) -> None:
        import newton
        self._newton = newton
        self._bodies = newton.ModelBuilder()
        self._shared: list[Callable[[Any], None]] = []
        self.articulated = False
        self.floating = False

    @classmethod
    def from_urdf(
        cls,
        path: str | Path,
        floating: bool = True,
        draw: str = "visual",
        colors: Mapping[str, Color] | Callable[[str], Color | None] | None = None,
    ) -> "Scene":
        """``draw``: ``"visual"`` (the URDF's visual geometry), ``"collision"`` or ``"all"``.

        ``colors`` maps a link name to a color, as a dict or a function. Meshes are drawn in
        flat colors because hardware textures are the one Warp feature AMD's port lacks.
        """
        if draw not in ("visual", "collision", "all"):
            raise ValueError(f"draw must be visual, collision or all, got {draw!r}")
        scene = cls()
        newton, b = scene._newton, scene._bodies
        b.add_urdf(str(path), floating=floating)
        visible, collides = int(newton.ShapeFlags.VISIBLE), int(newton.ShapeFlags.COLLIDE_SHAPES)
        pick = colors if callable(colors) else (colors or {}).get
        for i in range(b.shape_count):
            flags = int(b.shape_flags[i])
            is_collider = bool(flags & collides)
            show = draw == "all" or (draw == "collision") == is_collider
            b.shape_flags[i] = (flags | visible) if show else (flags & ~visible)
            color = pick(b.body_label[b.shape_body[i]].split("/")[-1]) if b.shape_body[i] >= 0 else None
            if color is not None:
                b.shape_color[i] = tuple(color)
        scene.articulated, scene.floating = True, floating
        return scene

    @classmethod
    def from_meshes(cls, bodies: Sequence[MeshBody]) -> "Scene":
        scene = cls()
        newton, b = scene._newton, scene._bodies
        for body in bodies:
            vertices, indices = body.arrays()
            b.add_shape_mesh(b.add_body(label=body.name), mesh=newton.Mesh(vertices, indices), color=body.color)
        return scene

    # ---- shared by every world -------------------------------------------------------

    def add_ground(
        self,
        height: float = 0.0,
        color: Color = _GROUND,
        tile_size: float | None = 0.5,
        tile_color: Color = _TILE,
        extent: tuple[float, float, float, float] = (-5.0, 13.0, -9.0, 9.0),
    ) -> "Scene":
        """A ground plane, with a checker of thin tiles over ``extent`` (x0, x1, y0, y1).

        The tiles are geometry, not a texture, and they are what makes motion visible from a
        camera that follows the robot. ``tile_size=None`` leaves a plain plane.
        """
        def add(b):
            import warp as wp
            b.add_ground_plane(height=height, color=color)
            if tile_size is None:
                return
            t = float(tile_size)
            for i in range(round(extent[0] / t), round(extent[1] / t)):
                for j in range(round(extent[2] / t), round(extent[3] / t)):
                    if (i + j) % 2 == 0:
                        b.add_shape_box(
                            -1,
                            xform=wp.transform(wp.vec3((i + 0.5) * t, (j + 0.5) * t, height + 0.001),
                                               wp.quat_identity()),
                            hx=t / 2, hy=t / 2, hz=0.001, color=tile_color)

        self._shared.append(add)
        return self

    def add_static_mesh(self, mesh: MeshBody, transform: Sequence[float] | None = None) -> "Scene":
        """A fixed mesh every world sees, for terrain or obstacles. ``transform`` is 7 numbers."""
        newton = self._newton
        vertices, indices = mesh.arrays()
        xf = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0) if transform is None else tuple(float(v) for v in transform)

        def add(b):
            import warp as wp
            b.add_shape_mesh(-1, xform=wp.transform(wp.vec3(*xf[:3]), wp.quat(*xf[3:])),
                             mesh=newton.Mesh(vertices, indices), color=mesh.color)

        self._shared.append(add)
        return self

    # ---- what the caller needs to build poses ----------------------------------------

    @property
    def body_names(self) -> list[str]:
        return [label.split("/")[-1] for label in self._bodies.body_label]

    @property
    def coord_count(self) -> int:
        return len(self._bodies.joint_q)

    @property
    def joint_index(self) -> dict[str, int]:
        """URDF joint name -> index of its first coordinate in a world's ``joint_q`` row."""
        b = self._bodies
        return {label.split("/")[-1]: int(b.joint_q_start[j]) for j, label in enumerate(b.joint_label)}


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


class BatchRenderer:
    """``num_worlds`` copies of a :class:`Scene`, each with its own poses and cameras.

    Poses and camera transforms may be torch tensors or numpy arrays. Torch tensors on the
    render device are read in place. Anything else is copied through the host.
    """

    def __init__(
        self,
        scene: Scene,
        num_worlds: int,
        width: int = 128,
        height: int = 128,
        cameras: int = 1,
        fov_deg: float = 45.0,
        device: Any = None,
        shadows: bool = True,
        sky: tuple[int, int, int] = (200, 215, 230),
    ) -> None:
        import newton
        import warp as wp
        from newton.sensors import SensorTiledCamera
        self._wp, self._newton = wp, newton
        self.scene, self.num_worlds = scene, int(num_worlds)
        self.width, self.height, self.cameras = int(width), int(height), int(cameras)
        self.body_names = scene.body_names
        self.body_index = {name: i for i, name in enumerate(self.body_names)}
        self.joint_index, self.coord_count = scene.joint_index, scene.coord_count

        b = newton.ModelBuilder()
        for _ in range(self.num_worlds):
            b.begin_world()
            b.add_builder(scene._bodies)
            b.end_world()
        for add in scene._shared:
            add(b)
        self.model = b.finalize(device=device)
        self.state = self.model.state()
        self.device = self.model.device

        self._camera = SensorTiledCamera(model=self.model)
        self._camera.utils.create_default_light(enable_shadows=shadows)
        self._rays = self._camera.utils.compute_pinhole_camera_rays(
            self.width, self.height, [math.radians(fov_deg)] * self.cameras)
        self._color = self._camera.utils.create_color_image_output(self.width, self.height, self.cameras)
        self._depth = self._camera.utils.create_depth_image_output(self.width, self.height, self.cameras)
        r, g, bl = sky
        self._clear = SensorTiledCamera.ClearData(clear_color=0xFF000000 | (bl << 16) | (g << 8) | r)
        self._camera_xf = wp.zeros((self.cameras, self.num_worlds), dtype=wp.transformf, device=self.device)
        self._bvh_built = False
        self.interop: str | None = None      # "zero-copy" or "host-copy", once a torch tensor came in

    # ---- poses -----------------------------------------------------------------------

    def set_joint_q(self, joint_q) -> None:
        """Joint coordinates (N, coord_count) of a URDF scene. Runs forward kinematics."""
        if not self.scene.articulated:
            raise RuntimeError("this scene has no joints, give body transforms with set_body_q")
        self._put(self.state.joint_q, joint_q, (self.num_worlds * self.coord_count,))
        self._newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    def set_body_q(self, body_q) -> None:
        """World transforms (N, bodies, 7) of every body, in :attr:`body_names` order."""
        self._put(self.state.body_q, body_q, (self.num_worlds * len(self.body_names), 7))

    def body_q(self):
        """Current body transforms (N, bodies, 7) as a torch tensor on the render device."""
        return self._wp.to_torch(self.state.body_q).reshape(self.num_worlds, len(self.body_names), 7)

    def joint_map(self, state_fields: Sequence[str], mapping: Mapping[str, Any]):
        """A :class:`~nedm.render.poses.JointMap` from state fields to this scene's joints."""
        from nedm.render.poses import JointMap
        return JointMap(self.joint_index, self.coord_count, state_fields, mapping,
                        base_coords=7 if self.scene.floating else 0)

    # ---- render ----------------------------------------------------------------------

    def render(self, camera, *, joint_q=None, body_q=None) -> Frames:
        """Render every world. ``camera`` is (N, 7), or (N, C, 7) with several cameras per world."""
        if joint_q is not None:
            self.set_joint_q(joint_q)
        if body_q is not None:
            self.set_body_q(body_q)
        self._put_cameras(camera)
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

    # ---- data in ---------------------------------------------------------------------

    def _put_cameras(self, camera) -> None:
        shape = (self.num_worlds, self.cameras, 7)
        if isinstance(camera, np.ndarray):
            camera = np.ascontiguousarray(camera.reshape(shape).transpose(1, 0, 2), dtype=np.float32)
        else:
            camera = camera.reshape(shape).transpose(0, 1).contiguous()
        self._put(self._camera_xf, camera, (self.cameras, self.num_worlds, 7))

    def _put(self, dst, src, shape: tuple[int, ...]) -> None:
        wp = self._wp
        if isinstance(src, np.ndarray):
            dst.assign(np.ascontiguousarray(src, dtype=np.float32).reshape(shape))
            return
        src = src.detach().float().reshape(shape)
        if self.interop != "host-copy":
            try:
                wp.copy(dst, wp.from_torch(src.contiguous(), dtype=dst.dtype))
                self.interop = "zero-copy"
                return
            except Exception as error:  # noqa: BLE001
                print(f"torch to warp without a copy is unavailable here ({type(error).__name__}: "
                      f"{str(error)[:120]}), copying through the host instead")
                self.interop = "host-copy"
        dst.assign(src.cpu().numpy())
