"""Camera poses, one per world.

A camera looks along its own -Z axis with +Y up (the OpenGL convention Newton's ray tracer
uses). Every helper returns transforms (N, 7) that :meth:`BatchRenderer.render` accepts.
"""

from __future__ import annotations

import torch

from nedm.render.poses import compose, transform_from_matrix


def look_at(eye: torch.Tensor, target: torch.Tensor, up=(0.0, 0.0, 1.0)) -> torch.Tensor:
    """Cameras at ``eye`` (N, 3) looking at ``target`` (N, 3), world +Z up by default."""
    # float64 until the final cast: a camera axis off by one float32 ulp flips silhouette pixels
    eye, target = torch.as_tensor(eye, dtype=torch.float64), torch.as_tensor(target, dtype=torch.float64)
    eye, target = torch.broadcast_tensors(eye.reshape(-1, 3), target.reshape(-1, 3).to(eye.device))
    up_t = torch.as_tensor(up, dtype=torch.float64, device=eye.device).expand_as(eye)
    forward = torch.nn.functional.normalize(target - eye, dim=-1)
    right = torch.nn.functional.normalize(torch.cross(forward, up_t, dim=-1), dim=-1)
    true_up = torch.cross(right, forward, dim=-1)
    matrix = torch.eye(4, dtype=torch.float64, device=eye.device).repeat(eye.shape[0], 1, 1)
    matrix[:, :3, 0], matrix[:, :3, 1], matrix[:, :3, 2], matrix[:, :3, 3] = right, true_up, -forward, eye
    return transform_from_matrix(matrix)


def follow(
    xy: torch.Tensor,
    eye_offset=(-1.05, -1.20, 0.70),
    look_offset=(0.10, 0.0, 0.20),
) -> torch.Tensor:
    """Third-person view that tracks each world's ``xy`` (N, 2) at a fixed world-frame offset.

    The offset does not turn with the robot, so turning stays visible in the picture.
    """
    xy = torch.as_tensor(xy, dtype=torch.float32)
    eye = torch.as_tensor(eye_offset, dtype=torch.float64, device=xy.device)
    look = torch.as_tensor(look_offset, dtype=torch.float64, device=xy.device)
    one = look_at(eye[None], look[None])            # orientation is the same in every world
    out = one.repeat(xy.shape[0], 1)
    out[:, :2] = xy + eye[:2].float()
    return out


def mounted(body: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    """Camera fixed to a body: ``body`` (N, 7) world poses, ``local`` (7,) the camera in the body frame.

    For a forward-looking camera on a robot whose +X is forward and +Z is up, build ``local``
    with ``look_at(position, position + (1, 0, 0))``.
    """
    local = torch.as_tensor(local, dtype=torch.float32, device=body.device).reshape(-1, 7)
    return compose(body.float(), local.expand(body.shape[0], 7))
