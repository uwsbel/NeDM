"""The Study Case II arm as a render scene. Shared by arm_fk_demo.py and arm_nrd_rollout.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SHAPES = REPO / "src/nedm/tracked_arm/arm_model/data/lrv_robotarm/lrv_arm_shapes"
# geometry link name -> the SolidWorks shape it was exported with (lrv_arm.py). Both fingers share one.
LINK_MESH = {"shoulder": "body_4_1.obj", "biceps": "body_2_1.obj", "elbow": "body_5_1.obj", "wrist": "body_6_1.obj",
             "endoffactor": "body_1_1.obj", "finger_1": "body_7_1.obj", "finger_2": "body_7_1.obj"}
BASE_MESH = "body_3_1.obj"
COLORS = [(0.85, 0.55, 0.20), (0.80, 0.80, 0.82), (0.30, 0.45, 0.75), (0.80, 0.80, 0.82),
          (0.25, 0.25, 0.28), (0.75, 0.25, 0.25), (0.75, 0.25, 0.25)]
ARM_SCALE = 2.0             # ARM_SCALE in nedm.tracked_arm.arm_data, restated because that module needs pychrono


def box_mesh(center, half):
    c, h = np.asarray(center, dtype=np.float32), np.asarray(half, dtype=np.float32)
    vertices = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float32) * h + c
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int32)
    return vertices, faces


def arm_scene(kin, arm_scale: float = ARM_SCALE, extra_bodies=()):
    """The links in the order of ``kin.link_names``, then ``extra_bodies``, then everything fixed.

    The meshes are in each link's REF frame at export scale. The Chrono scene scales the arm.
    """
    from nedm.render import MeshBody, Scene, transform_from_matrix
    links = [MeshBody(name, path=SHAPES / LINK_MESH[name], scale=arm_scale, color=color)
             for name, color in zip(kin.link_names, COLORS, strict=True)]
    scene = Scene.from_meshes(links + list(extra_bodies))
    base = transform_from_matrix(kin.base_to_world[None])[0].tolist()
    scene.add_static_mesh(MeshBody("arm_base", path=SHAPES / BASE_MESH, scale=arm_scale, color=(0.2, 0.2, 0.22)), base)
    if len(kin.vehicle_box_center):
        v, f = box_mesh(kin.vehicle_box_center[0].tolist(), kin.vehicle_box_half[0].tolist())
        scene.add_static_mesh(MeshBody("vehicle", vertices=v, faces=f, color=(0.35, 0.40, 0.30)))
    scene.add_ground(height=kin.ground_z, tile_size=1.0, extent=(-8.0, 8.0, -8.0, 8.0))
    return scene
