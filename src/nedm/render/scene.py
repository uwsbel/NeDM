"""What gets drawn, described without reference to any renderer.

A :class:`Scene` is plain data: the moving bodies as triangle meshes, optionally the URDF
joint tree that poses them, and whatever every world shares (a ground, fixed meshes). Each
renderer backend builds its own representation from it, so the same scene and the same poses
can be drawn by more than one renderer and the pictures compared.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from nedm.render.poses import transform_from_matrix
from nedm.render.urdf import UrdfModel, load_urdf

Color = tuple[float, float, float]
IDENTITY = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


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
        """Vertices (V, 3) float32 and faces (F, 3) int32, with the scale applied."""
        if self.path is not None:
            import trimesh
            mesh = trimesh.load(str(self.path), force="mesh", process=False)
            vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        else:
            vertices, faces = np.asarray(self.vertices), np.asarray(self.faces)
        return (vertices * self.scale).astype(np.float32), faces.reshape(-1, 3).astype(np.int32)


@dataclass
class Ground:
    """A plane at ``height``, with a checker of thin tiles over ``extent`` (x0, x1, y0, y1).

    The tiles are geometry, not a texture. They are what makes motion visible from a camera
    that follows the robot, and they work on renderers and GPUs without texture support.
    """

    height: float = 0.0
    color: Color = (0.60, 0.55, 0.46)
    tile_size: float | None = 0.5
    tile_color: Color = (0.47, 0.42, 0.34)
    extent: tuple[float, float, float, float] = (-5.0, 13.0, -9.0, 9.0)
    tile_thickness: float = 0.002

    def tile_centers(self) -> list[tuple[float, float, float]]:
        if self.tile_size is None:
            return []
        t = float(self.tile_size)
        return [((i + 0.5) * t, (j + 0.5) * t, self.height + self.tile_thickness / 2)
                for i in range(round(self.extent[0] / t), round(self.extent[1] / t))
                for j in range(round(self.extent[2] / t), round(self.extent[3] / t))
                if (i + j) % 2 == 0]

    def tile_mesh(self) -> MeshBody | None:
        """All tiles as ONE mesh, for renderers that pay per object rather than per triangle."""
        centers = self.tile_centers()
        if not centers:
            return None
        h = np.array([self.tile_size / 2, self.tile_size / 2, self.tile_thickness / 2])
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * h
        box = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
        vertices = np.concatenate([corners + np.array(c) for c in centers])
        faces = np.concatenate([box + 8 * k for k in range(len(centers))])
        return MeshBody("ground_tiles", vertices=vertices, faces=faces, color=self.tile_color)


class Scene:
    """What one world contains, and what all worlds share.

    Two ways to describe the moving part:

    * :meth:`from_urdf` for an articulated robot. Poses are given as joint coordinates and
      :meth:`body_transforms` runs the forward kinematics.
    * :meth:`from_meshes` for rigid bodies whose transforms the caller computes, for example
      a chassis and four wheels, or the links of an arm with its own kinematics.
    """

    def __init__(self) -> None:
        self.bodies: list[MeshBody] = []
        self.statics: list[tuple[MeshBody, tuple[float, ...]]] = []
        self.ground: Ground | None = None
        self.urdf: UrdfModel | None = None
        self._drawn_links: list[int] = []

    @classmethod
    def from_urdf(
        cls,
        path: str | Path,
        floating: bool = True,
        draw: str = "visual",
        colors: Mapping[str, Color] | Callable[[str], Color | None] | None = None,
    ) -> "Scene":
        """``draw``: ``"visual"`` (the URDF's visual geometry) or ``"collision"``.

        ``colors`` maps a link name to a color, as a dict or a function. Links are drawn in
        flat colors.
        """
        scene = cls()
        scene.urdf = load_urdf(path, floating=floating, draw=draw)
        pick = colors if callable(colors) else (colors or {}).get
        for i, link in enumerate(scene.urdf.links):
            if link.vertices is None:
                continue
            color = pick(link.name)
            body = MeshBody(link.name, vertices=link.vertices, faces=link.faces)
            if color is not None:
                body.color = tuple(color)
            scene.bodies.append(body)
            scene._drawn_links.append(i)
        return scene

    @classmethod
    def from_meshes(cls, bodies: Sequence[MeshBody]) -> "Scene":
        scene = cls()
        scene.bodies = list(bodies)
        return scene

    # ---- shared by every world -------------------------------------------------------

    def add_ground(self, height: float = 0.0, **options) -> "Scene":
        """A ground plane with a tile checker. See :class:`Ground` for the options."""
        self.ground = Ground(height=height, **options)
        return self

    def add_static_mesh(self, mesh: MeshBody, transform: Sequence[float] | None = None) -> "Scene":
        """A fixed mesh every world sees, for terrain or obstacles. ``transform`` is 7 numbers."""
        self.statics.append((mesh, IDENTITY if transform is None else tuple(float(v) for v in transform)))
        return self

    # ---- what the caller needs to build poses ----------------------------------------

    @property
    def articulated(self) -> bool:
        return self.urdf is not None

    @property
    def floating(self) -> bool:
        return self.urdf is not None and self.urdf.floating

    @property
    def body_names(self) -> list[str]:
        return [body.name for body in self.bodies]

    @property
    def coord_count(self) -> int:
        return self.urdf.coord_count if self.urdf is not None else 0

    @property
    def joint_index(self) -> dict[str, int]:
        """URDF joint name -> index of its coordinate in a world's ``joint_q`` row."""
        return self.urdf.joint_index if self.urdf is not None else {}

    def body_transforms(self, joint_q: torch.Tensor) -> torch.Tensor:
        """Forward kinematics: joint coordinates (N, coord_count) -> body transforms (N, bodies, 7)."""
        if self.urdf is None:
            raise RuntimeError("this scene has no joints, give body transforms directly")
        rotation, position = self.urdf.forward_kinematics(joint_q)
        rotation, position = rotation[:, self._drawn_links], position[:, self._drawn_links]
        matrix = torch.zeros(*rotation.shape[:2], 4, 4, dtype=torch.float32, device=joint_q.device)
        matrix[..., :3, :3], matrix[..., :3, 3], matrix[..., 3, 3] = rotation, position, 1.0
        return transform_from_matrix(matrix)
