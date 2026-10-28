"""URDF -> link meshes and batched forward kinematics, with no renderer and no simulator.

Only what drawing needs: each link's visual (or collision) geometry as one triangle mesh in
the link frame, and the joint tree. Forward kinematics is batched torch, so it runs on the
device the NRD model is on and the result goes to the renderer without leaving it.

Joint coordinates of one world are a row: a floating base first (position, then quaternion
xyzw) when the model has one, then every movable joint in depth-first order, children in
file order. That is the order Newton's importer produces for the same file, which keeps
saved trajectories readable by either renderer.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

MOVABLE = ("revolute", "continuous", "prismatic")


def _floats(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    return np.array([float(v) for v in text.split()] if text else default, dtype=np.float64)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll, pitch, yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = np.cos(rpy[0]), np.sin(rpy[0]), np.cos(rpy[1]), np.sin(rpy[1]), np.cos(rpy[2]), np.sin(rpy[2])
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _origin(element: ET.Element | None) -> np.ndarray:
    matrix = np.eye(4)
    origin = element.find("origin") if element is not None else None
    if origin is not None:
        matrix[:3, :3] = _rpy_matrix(_floats(origin.get("rpy"), (0.0, 0.0, 0.0)))
        matrix[:3, 3] = _floats(origin.get("xyz"), (0.0, 0.0, 0.0))
    return matrix


def _resolve(filename: str, urdf_dir: Path) -> Path:
    if filename.startswith("package://"):
        tail = Path(filename[len("package://"):])
        for up in (urdf_dir, *urdf_dir.parents):        # the package root is some ancestor
            for candidate in (up / tail, up / Path(*tail.parts[1:])):
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(f"cannot find {filename} above {urdf_dir}")
    path = Path(filename.removeprefix("file://"))
    return path if path.is_absolute() else (urdf_dir / path).resolve()


def _geometry_mesh(geometry: ET.Element, urdf_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh
    shape = geometry[0]
    if shape.tag == "mesh":
        mesh = trimesh.load(str(_resolve(shape.get("filename"), urdf_dir)), force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64) * _floats(shape.get("scale"), (1.0, 1.0, 1.0))
        return vertices, np.asarray(mesh.faces, dtype=np.int64)
    if shape.tag == "box":
        mesh = trimesh.creation.box(extents=_floats(shape.get("size"), (1.0, 1.0, 1.0)))
    elif shape.tag == "cylinder":
        mesh = trimesh.creation.cylinder(radius=float(shape.get("radius")), height=float(shape.get("length")), sections=24)
    elif shape.tag == "sphere":
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(shape.get("radius")))
    else:
        raise ValueError(f"unsupported URDF geometry <{shape.tag}>")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


@dataclass
class Link:
    name: str
    vertices: np.ndarray | None = None       # (V, 3) in the link frame, None if nothing to draw
    faces: np.ndarray | None = None          # (F, 3)


@dataclass
class Joint:
    name: str
    kind: str
    parent: int
    child: int
    origin: np.ndarray                        # (4, 4), child frame in the parent frame at q = 0
    axis: np.ndarray                          # (3,), in the child frame
    coord: int = -1                           # index into a joint_q row, -1 if not movable


@dataclass
class UrdfModel:
    links: list[Link]
    joints: list[Joint] = field(default_factory=list)      # in the order forward kinematics visits them
    root: int = 0
    floating: bool = True
    coord_count: int = 0

    @property
    def joint_index(self) -> dict[str, int]:
        index = {"floating_base": 0} if self.floating else {}
        index.update({j.name: j.coord for j in self.joints if j.coord >= 0})
        return index

    def forward_kinematics(self, joint_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint coordinates (N, coord_count) -> every link's world rotation (N, L, 3, 3) and position (N, L, 3)."""
        n, device = joint_q.shape[0], joint_q.device
        q = joint_q.float()
        rotation: list[torch.Tensor | None] = [None] * len(self.links)
        position: list[torch.Tensor | None] = [None] * len(self.links)
        if self.floating:
            x, y, z, w = (q[:, 3 + i] for i in range(4))
            rotation[self.root] = torch.stack([
                1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1).reshape(n, 3, 3)
            position[self.root] = q[:, :3]
        else:
            rotation[self.root] = torch.eye(3, device=device).expand(n, 3, 3)
            position[self.root] = torch.zeros(n, 3, device=device)
        for joint in self.joints:
            origin = torch.as_tensor(joint.origin, dtype=torch.float32, device=device)
            parent_r, parent_p = rotation[joint.parent], position[joint.parent]
            r = parent_r @ origin[:3, :3]
            p = parent_p + parent_r @ origin[:3, 3]
            if joint.coord >= 0:
                value = q[:, joint.coord]
                axis = torch.as_tensor(joint.axis, dtype=torch.float32, device=device)
                if joint.kind == "prismatic":
                    p = p + r @ axis * value[:, None]
                else:
                    r = r @ _axis_angle(axis, value)
            rotation[joint.child], position[joint.child] = r, p
        return torch.stack(rotation, dim=1), torch.stack(position, dim=1)


def _axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues: rotation matrices (N, 3, 3) about a fixed unit axis."""
    x, y, z = axis.tolist()
    skew = torch.tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], device=angle.device)
    eye = torch.eye(3, device=angle.device)
    s, c = torch.sin(angle)[:, None, None], torch.cos(angle)[:, None, None]
    return eye + s * skew + (1 - c) * (skew @ skew)


def load_urdf(path: str | Path, floating: bool = True, draw: str = "visual") -> UrdfModel:
    """Parse a URDF. ``draw`` picks which geometry becomes each link's mesh: ``visual`` or ``collision``."""
    if draw not in ("visual", "collision"):
        raise ValueError(f"draw must be visual or collision, got {draw!r}")
    path = Path(path)
    robot = ET.parse(path).getroot()
    names = [link.get("name") for link in robot.findall("link")]
    index = {name: i for i, name in enumerate(names)}

    links = []
    for element in robot.findall("link"):
        vertices, faces, offset = [], [], 0
        for item in element.findall(draw):
            geometry = item.find("geometry")
            if geometry is None or len(geometry) == 0:
                continue
            v, f = _geometry_mesh(geometry, path.parent)
            frame = _origin(item)                       # bake the visual's own offset into the vertices
            vertices.append(v @ frame[:3, :3].T + frame[:3, 3])
            faces.append(f + offset)
            offset += len(v)
        links.append(Link(element.get("name"), np.concatenate(vertices).astype(np.float32),
                          np.concatenate(faces).astype(np.int32)) if vertices else Link(element.get("name")))

    children: dict[int, list[Joint]] = {}
    has_parent = set()
    for element in robot.findall("joint"):
        axis = _floats(element.find("axis").get("xyz") if element.find("axis") is not None else None, (1.0, 0.0, 0.0))
        length = np.linalg.norm(axis)           # fixed joints are often written with a zero axis
        joint = Joint(element.get("name"), element.get("type"), index[element.find("parent").get("link")],
                      index[element.find("child").get("link")], _origin(element), axis / length if length > 0 else axis)
        if joint.kind not in (*MOVABLE, "fixed"):
            raise ValueError(f"joint {joint.name!r} has unsupported type {joint.kind!r}")
        if joint.kind in MOVABLE and length == 0:
            raise ValueError(f"joint {joint.name!r} moves but has a zero axis")
        children.setdefault(joint.parent, []).append(joint)
        has_parent.add(joint.child)
    roots = [i for i in range(len(links)) if i not in has_parent]
    if len(roots) != 1:
        raise ValueError(f"expected one root link, found {[names[i] for i in roots]}")

    model = UrdfModel(links, root=roots[0], floating=floating, coord_count=7 if floating else 0)
    stack = list(reversed(children.get(roots[0], [])))
    while stack:                                        # depth first, children in file order
        joint = stack.pop()
        if joint.kind in MOVABLE:
            joint.coord = model.coord_count
            model.coord_count += 1
        model.joints.append(joint)
        stack.extend(reversed(children.get(joint.child, [])))
    return model
