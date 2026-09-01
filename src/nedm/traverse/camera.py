"""Overhead camera model: the world<->image frames contract (plan §3.3).

One fixed nadir pinhole camera at arena center, ``cam_height_m`` above datum,
image up = +Y world (north-up map, see ``scene.overhead_camera_pose``). Frame
buffers are consumed through the taps in ``scene``, which flip OptiX's
bottom-up rows so row 0 = image top. Under that convention:

    u = cx + f * x / (H - z)          (east  -> right)
    v = cy - f * y / (H - z)          (north -> up)

with f = (width/2) / tan(hfov/2), cx = (width-1)/2, cy = (height-1)/2.

Whether ``ChDepthCamera`` reports distance along the per-pixel ray ("ray")
or planar z-depth ("planar") is measured against the calibrated heightmap by
``scripts/traverse_wp0b_sensor_smoke.py`` (alignment + depth-at-edges), which
records the winning convention in its summary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class CameraModel:
    width: int = 256
    height: int = 256
    hfov_rad: float = math.radians(47.0)
    cam_height_m: float = 100.0

    @property
    def f_px(self) -> float:
        return (self.width / 2.0) / math.tan(self.hfov_rad / 2.0)

    @property
    def cx(self) -> float:
        return (self.width - 1) / 2.0

    @property
    def cy(self) -> float:
        return (self.height - 1) / 2.0

    def world_to_pixel(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World point(s) -> (u, v) pixel coordinates (float, unrounded)."""
        x, y, z = np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(z, np.float64)
        scale = self.f_px / (self.cam_height_m - z)
        return self.cx + scale * x, self.cy - scale * y

    def pixel_rays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-pixel ray tangents (dx/dz-drop, dy/dz-drop) and 1/cos(theta)."""
        u = np.arange(self.width, dtype=np.float64)
        v = np.arange(self.height, dtype=np.float64)
        uu, vv = np.meshgrid(u, v)
        tx = (uu - self.cx) / self.f_px
        ty = -(vv - self.cy) / self.f_px
        sec = np.sqrt(1.0 + tx**2 + ty**2)
        return tx, ty, sec

    def depth_to_world(
        self, depth: np.ndarray, convention: str = "ray"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Depth image -> per-pixel world (x, y, z) of the hit point.

        convention "ray": depth is metric distance along the pixel ray
        (ChDepthCamera, validated at smoke tier). "planar": depth is the
        distance along the optical axis (z-drop).
        """
        depth = np.asarray(depth, np.float64)
        tx, ty, sec = self.pixel_rays()
        drop = depth / sec if convention == "ray" else depth
        z = self.cam_height_m - drop
        return tx * drop, ty * drop, z
