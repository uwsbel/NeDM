"""Optional execution-only path altitude from a single measured depth image.

No terrain assets, heightmap, semantic oracle or fallback elevation is used.
The visible surface can include the vehicle or an obstacle; this intentionally
does not claim to recover ground beneath occlusions.
"""
from __future__ import annotations

import numpy as np

from nedm.traverse.camera import CameraModel


class ObservedDepthHeight:
    def __init__(self, depth_m, camera, *, max_xy_distance_m=.75):
        model = CameraModel(width=int(camera["width"]), height=int(camera["height"]),
                            hfov_rad=float(camera["hfov_rad"]),
                            cam_height_m=float(camera["cam_height_m"]))
        depth = np.asarray(depth_m, np.float64)
        if depth.shape != (model.height, model.width):
            raise ValueError("Depth shape disagrees with calibrated camera")
        x, y, z = model.depth_to_world(depth, convention="ray",
                                      ray_scale=float(camera["depth_ray_scale"]))
        valid = np.isfinite(depth) & (depth > 0) & (depth < float(camera["max_depth_m"]))
        if not valid.any():
            raise ValueError("No measured surface for the depth-only controller")
        self.xy = np.column_stack([x[valid], y[valid]])
        self.z = z[valid]
        self.max_xy_distance_m = float(max_xy_distance_m)
        self.queries = []

    def height(self, x, y):
        squared_distance = np.square(self.xy-[float(x), float(y)]).sum(axis=1)
        index = int(np.argmin(squared_distance))
        distance = float(np.sqrt(squared_distance[index]))
        if distance > self.max_xy_distance_m:
            raise ValueError(f"Path point lacks nearby measured depth: XY distance {distance:.3f}m")
        height = float(self.z[index])
        self.queries.append({"x_m": float(x), "y_m": float(y),
                             "visible_surface_z_m": height,
                             "nearest_observed_xy_distance_m": float(distance)})
        return height
