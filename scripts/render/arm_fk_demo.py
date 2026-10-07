#!/usr/bin/env python
"""Batch-render the Study Case II arm from joint angles, using the repository's own kinematics.

The second way to plug a model in: no URDF. ``ArmKinematics`` already turns the arm NRD
model's joint state into a pose for every link, so the renderer only needs the link meshes
and those transforms. An ``ArmReachingEnv`` would pass ``env.current_q()`` where this demo
passes a made-up joint motion.

    PYTHONPATH=src python scripts/render/arm_fk_demo.py --out out/arm_demo --envs 16 --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
SHAPES = REPO / "src/nedm/tracked_arm/arm_model/data/lrv_robotarm/lrv_arm_shapes"
# geometry link name -> the SolidWorks shape it was exported with (lrv_arm.py). Both fingers share one.
LINK_MESH = {"shoulder": "body_4_1.obj", "biceps": "body_2_1.obj", "elbow": "body_5_1.obj", "wrist": "body_6_1.obj",
             "endoffactor": "body_1_1.obj", "finger_1": "body_7_1.obj", "finger_2": "body_7_1.obj"}
BASE_MESH = "body_3_1.obj"
COLORS = [(0.85, 0.55, 0.20), (0.80, 0.80, 0.82), (0.30, 0.45, 0.75), (0.80, 0.80, 0.82),
          (0.25, 0.25, 0.28), (0.75, 0.25, 0.25), (0.75, 0.25, 0.25)]


def box_mesh(center, half):
    c, h = np.asarray(center, dtype=np.float32), np.asarray(half, dtype=np.float32)
    vertices = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float32) * h + c
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int32)
    return vertices, faces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geometry", type=Path, default=REPO / "artifacts/arm_geometry/arm_geometry_v1.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--device", default=None, help="warp device, default: the GPU if there is one")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=["newton", "madrona"], default="newton")
    parser.add_argument("--arm-scale", type=float, default=2.0,
                        help="ARM_SCALE in nedm.tracked_arm.arm_data, restated because that module needs pychrono")
    a = parser.parse_args()

    if a.backend == "newton":
        import warp as wp
        wp.config.quiet = True
    from nedm.render import (BatchRenderer, CollageRecorder, MeshBody, Scene, cameras, save_sheet,
                                  transform_from_matrix)
    from nedm.tracked_arm.rl.arm_kinematics import ArmKinematics

    kin = ArmKinematics.from_json(a.geometry)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # The meshes are in each link's REF frame at export scale. The Chrono scene scales the arm.
    scene = Scene.from_meshes([MeshBody(name, path=SHAPES / LINK_MESH[name], scale=a.arm_scale, color=color)
                               for name, color in zip(kin.link_names, COLORS, strict=True)])
    base = transform_from_matrix(kin.base_to_world[None])[0].tolist()
    scene.add_static_mesh(MeshBody("arm_base", path=SHAPES / BASE_MESH, scale=a.arm_scale, color=(0.2, 0.2, 0.22)), base)
    if len(kin.vehicle_box_center):
        v, f = box_mesh(kin.vehicle_box_center[0].tolist(), kin.vehicle_box_half[0].tolist())
        scene.add_static_mesh(MeshBody("vehicle", vertices=v, faces=f, color=(0.35, 0.40, 0.30)))
    scene.add_ground(height=kin.ground_z, tile_size=1.0, extent=(-8.0, 8.0, -8.0, 8.0))
    renderer = BatchRenderer(scene, a.envs, width=a.res, height=a.res, device=a.device, backend=a.backend)

    # A different smooth joint motion in every world: base yaw, biceps, elbow, wrist. The biceps
    # stays raised (positive is up) so the arm clears the vehicle deck, which nothing here enforces.
    g = torch.Generator().manual_seed(a.seed)
    center = torch.tensor([0.0, 0.35, -0.2, 0.0])
    swing = torch.tensor([1.0, 0.30, 0.5, 0.5])
    phase = torch.rand(a.envs, kin.num_joints, generator=g) * 6.283
    rate = 0.5 + torch.rand(a.envs, kin.num_joints, generator=g)

    camera = cameras.look_at(torch.tensor([[4.5, -7.0, 4.0]]).repeat(a.envs, 1), torch.tensor([[0.3, 0.0, 1.8]]))
    video = CollageRecorder(out / "collage_rgb.mp4", a.envs, sample=min(a.envs, 50), cols=min(a.envs, 10),
                            fps=25, seed=a.seed)
    for k in range(a.frames):
        q = kin.q_home + center + swing * torch.sin(rate * (k / 25.0) + phase)
        body_q = transform_from_matrix(kin.link_transforms(q))          # (envs, links, 7)
        frames = renderer.render(camera, body_q=body_q)
        video.add(frames.rgb[:, 0])
        if k == a.frames // 2:
            save_sheet(frames.rgb[:, 0], out / "all_worlds_midframe.png", cols=min(a.envs, 8))
    video.close()
    print(f"wrote {out}/collage_rgb.mp4 ({a.envs} worlds, {a.frames} frames, device {renderer.device})")


if __name__ == "__main__":
    main()
