"""Batch rendering for NRD rollouts: one image per world, every step, on the GPU.

    scene = Scene.from_urdf("go2.urdf").add_ground()
    renderer = BatchRenderer(scene, num_worlds=1024, width=128, height=128)
    frames = renderer.render(cameras.follow(xy), joint_q=joint_q)
    frames.depth_torch()        # (1024, 1, 128, 128), on the device, no copy

Needs ``newton`` and ``warp`` (``requirements-render.txt``). Nothing else in ``nedm`` does,
so this package imports them only when a scene is built.
"""

from nedm.render import cameras
from nedm.render.poses import (
    JointMap,
    PlanarPose,
    base_transform,
    compose,
    quat_from_rpy,
    tilt_from_gravity,
    transform_from_matrix,
)
from nedm.render.renderer import BatchRenderer, Frames, MeshBody, Scene
from nedm.render.video import CollageRecorder, depth_to_gray, grid, save_sheet

__all__ = [
    "BatchRenderer",
    "CollageRecorder",
    "Frames",
    "JointMap",
    "MeshBody",
    "PlanarPose",
    "Scene",
    "base_transform",
    "cameras",
    "compose",
    "depth_to_gray",
    "grid",
    "quat_from_rpy",
    "save_sheet",
    "tilt_from_gravity",
    "transform_from_matrix",
]
