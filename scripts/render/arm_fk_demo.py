#!/usr/bin/env python
"""Batch-render the Study Case II arm from joint angles, using the repository's own kinematics.

The second way to plug a model in: no URDF. ``ArmKinematics`` already turns the arm NRD
model's joint state into a pose for every link, so the renderer only needs the link meshes
and those transforms. This demo passes a made-up joint motion and needs no checkpoint.
``arm_nrd_rollout.py`` is the same scene driven by the reaching policy in the NRD model.

    PYTHONPATH=src python scripts/render/arm_fk_demo.py --out out/arm_demo --envs 16 --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from arm_scene import ARM_SCALE, REPO, arm_scene


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
    parser.add_argument("--arm-scale", type=float, default=ARM_SCALE)
    a = parser.parse_args()

    if a.backend == "newton":
        import warp as wp
        wp.config.quiet = True
    from nedm.render import BatchRenderer, CollageRecorder, cameras, save_sheet, transform_from_matrix
    from nedm.tracked_arm.rl.arm_kinematics import ArmKinematics

    kin = ArmKinematics.from_json(a.geometry)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    renderer = BatchRenderer(arm_scene(kin, a.arm_scale), a.envs, width=a.res, height=a.res, device=a.device,
                             backend=a.backend)

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
