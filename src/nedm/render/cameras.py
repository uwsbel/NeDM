"""Camera poses, one per world, and camera models (the rays of each camera).

A camera looks along its own -Z axis with +Y up (the OpenGL convention Newton's ray tracer
uses). Every pose helper returns transforms (N, 7) that :meth:`BatchRenderer.render` accepts.
Calibrations use the OpenCV convention (x right, y down, z forward): :func:`pose_from_opencv`
and :func:`pose_to_opencv` are the one conversion.

Camera models (``BatchRenderer(..., camera_models=[...])``, Newton only): :class:`PinholeCamera`
(own vertical fov) and :class:`CalibratedCamera` (``opencv_fisheye`` / ``opencv`` lens with K, D
of the real image, a crop box and the renderer's output size).
"""

from __future__ import annotations

import math

import numpy as np
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


# ---- camera conventions -------------------------------------------------------------------------------------------
# OpenCV camera frame: x right, y down, z forward (optical axis). Renderer (Newton, MuJoCo) camera frame: x right, y up,
# z backward. The two differ by a turn of 180 deg about the camera x axis: R_renderer = R_opencv @ diag(1, -1, -1).
def pose_from_opencv(pose) -> torch.Tensor:
    """Camera poses in the OpenCV convention -> renderer transforms. ``pose`` (..., 7) = x y z, qw qx qy qz (the
    calibration contract's ``T_parent_cam``: camera frame x right, y down, z forward, in the parent frame, metres).
    Returns (..., 7) = x y z, qx qy qz qw of the same camera in the renderer's convention (-Z forward, +Y up), float64.
    The inverse is :func:`pose_to_opencv`."""
    p = torch.as_tensor(pose, dtype=torch.float64)
    w, x, y, z = p[..., 3:7].unbind(-1)
    n = torch.sqrt(w * w + x * x + y * y + z * z)
    # q_renderer = q_opencv * q_x(pi), q_x(pi) = (w 0, x 1, y 0, z 0): (w, x, y, z) * (0, 1, 0, 0) = (-x, w, z, -y)
    return torch.cat([p[..., :3], torch.stack([w, z, -y, -x], -1) / n[..., None]], -1)


def pose_to_opencv(transform) -> torch.Tensor:
    """Renderer transforms (..., 7) = x y z, qx qy qz qw -> OpenCV-convention poses (..., 7) = x y z, qw qx qy qz
    (inverse of :func:`pose_from_opencv`; the quaternion sign is chosen with qw >= 0), float64."""
    t = torch.as_tensor(transform, dtype=torch.float64)
    x, y, z, w = t[..., 3:7].unbind(-1)
    # q_opencv = q_renderer * q_x(pi)^-1 = q_renderer * (0, -1, 0, 0): (w, x, y, z) * (0, -1, 0, 0) = (x, -w, -z, y)
    q = torch.stack([x, -w, -z, y], -1)
    q = q / q.norm(dim=-1, keepdim=True)
    q = torch.where(q[..., :1] < 0, -q, q)
    return torch.cat([t[..., :3], q], -1)


def opencv_look_at(eye, target, up=(0.0, 0.0, 1.0)) -> torch.Tensor:
    """OpenCV-convention poses (N, 7) x y z, qw qx qy qz of cameras at ``eye`` (N, 3) looking at ``target`` (N, 3)
    (image up towards world ``up``), float64 throughout: the contract's layout, for placeholder calibrations and
    tests."""
    eye = torch.as_tensor(eye, dtype=torch.float64).reshape(-1, 3)
    target = torch.as_tensor(target, dtype=torch.float64).reshape(-1, 3)
    eye, target = torch.broadcast_tensors(eye, target)
    z = torch.nn.functional.normalize(target - eye, dim=-1)
    x = torch.nn.functional.normalize(torch.cross(z, torch.as_tensor(up, dtype=torch.float64).expand_as(z), dim=-1),
                                      dim=-1)
    y = torch.cross(z, x, dim=-1)                                   # image down
    m = torch.stack([x, y, z], -1)                                  # columns = camera axes in the world
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m.reshape(-1, 9).unbind(-1)
    four = torch.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1)
    rows = torch.stack([torch.stack([four[:, 0], m21 - m12, m02 - m20, m10 - m01], -1),     # 4w (w, x, y, z)
                        torch.stack([m21 - m12, four[:, 1], m01 + m10, m02 + m20], -1),     # 4x
                        torch.stack([m02 - m20, m01 + m10, four[:, 2], m12 + m21], -1),     # 4y
                        torch.stack([m10 - m01, m02 + m20, m12 + m21, four[:, 3]], -1)], -2)  # 4z
    big = four.argmax(-1)
    q = rows[torch.arange(len(big)), big] / (2 * torch.sqrt(four[torch.arange(len(big)), big]))[:, None]
    q = torch.where(q[:, :1] < 0, -q, q)
    return torch.cat([eye, q], -1)


# ---- camera models: per-camera rays ---------------------------------------------------------------------------------
class PinholeCamera:
    """The renderer's own pinhole: vertical field of view ``fovy_deg``, square pixels, principal point at the image
    centre. Newton draws it with its own ray kernel (``compute_pinhole_camera_rays``), so a renderer whose cameras are
    all pinholes of one fov draws exactly the pictures it drew before camera models existed."""

    kind = "pinhole"

    def __init__(self, fovy_deg: float):
        self.fovy_deg = float(fovy_deg)

    def __repr__(self) -> str:
        return f"PinholeCamera(fovy_deg={self.fovy_deg})"

    def rays(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Unit ray directions (H, W, 3) float64 in the renderer's camera frame and validity (H, W) bool (all True).
        Newton's formula in float64: d = (u 2h W/H, -v 2h, -1), u = (px + 0.5)/W - 0.5, v = (py + 0.5)/H - 0.5."""
        h = math.tan(math.radians(self.fovy_deg) / 2)
        u = (np.arange(width) + 0.5) / width - 0.5
        v = (np.arange(height) + 0.5) / height - 0.5
        x, y = u[None, :] * 2 * h * width / height, -v[:, None] * 2 * h
        d = np.stack(np.broadcast_arrays(x, y, -np.ones((1, 1))), -1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True), np.ones((height, width), bool)


class CalibratedCamera:
    """A calibrated real camera, drawn through its own lens model.

    ``model``: ``"opencv_fisheye"`` (Kannala-Brandt, ``D`` = k1..k4 as ``cv2.fisheye``: r_d = theta (1 + k1 theta^2 +
    k2 theta^4 + k3 theta^6 + k4 theta^8), valid beyond 90 deg where cv2.fisheye itself is not) or ``"opencv"``
    (radial-tangential, ``D`` = k1 k2 p1 p2 [k3 [k4 k5 k6 ...]] as ``cv2.projectPoints``). ``K`` = (fx, fy, cx, cy) and ``width`` x ``height`` of the REAL image (pixel centres at
    integer coordinates, OpenCV's convention). ``crop`` = (x0, y0, w, h) in real pixels (None: the whole image).

    The output image (W_out x H_out, the renderer's size) samples the crop box like ``cv2.resize(real[y0:y0+h,
    x0:x0+w], (W_out, H_out))``: output pixel (i, j) looks along the undistorted ray of the real-image point
    u = x0 + (j + 0.5) w / W_out - 0.5, v = y0 + (i + 0.5) h / H_out - 0.5. Pixels outside the real sensor, or whose
    point has no ray in the model (beyond the lens model's monotonic range), are invalid: the renderer draws them black
    with depth 0.
    """

    MODELS = ("opencv_fisheye", "opencv")

    def __init__(self, model: str, K, D, width: int, height: int, crop=None):
        if model not in self.MODELS:
            raise ValueError(f"camera model {model!r}: one of {self.MODELS}")
        self.kind = self.model = model
        self.K = tuple(float(k) for k in K)
        self.D = tuple(float(d) for d in D)
        if len(self.K) != 4:
            raise ValueError("K = (fx, fy, cx, cy)")
        if model == "opencv_fisheye" and len(self.D) != 4:
            raise ValueError("opencv_fisheye: D = (k1, k2, k3, k4)")
        if model == "opencv" and len(self.D) not in (4, 5, 8, 12, 14):
            raise ValueError("opencv: D has 4, 5, 8, 12 or 14 coefficients (cv2.projectPoints)")
        self.width, self.height = int(width), int(height)
        self.crop = tuple(float(c) for c in crop) if crop is not None else (0.0, 0.0, float(width), float(height))

    @classmethod
    def from_contract(cls, cam: dict, crop=None) -> "CalibratedCamera":
        """From one ``cameras`` entry of the calibration contract (``model``, ``K``, ``D``, ``width``, ``height``)."""
        return cls(cam["model"], cam["K"], cam["D"], cam["width"], cam["height"], crop)

    def __repr__(self) -> str:
        return (f"CalibratedCamera({self.model!r}, K={self.K}, D={self.D}, {self.width}x{self.height}, "
                f"crop={self.crop})")

    @property
    def matrix(self) -> np.ndarray:
        fx, fy, cx, cy = self.K
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)

    # -- pixel grids --
    def real_pixels(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Real-image coordinates (u, v), each (H_out, W_out) float64, of the output pixel centres."""
        x0, y0, w, h = self.crop
        u = x0 + (np.arange(width) + 0.5) * w / width - 0.5
        v = y0 + (np.arange(height) + 0.5) * h / height - 0.5
        return np.broadcast_to(u[None, :], (height, width)).copy(), np.broadcast_to(v[:, None], (height, width)).copy()

    def output_pixels(self, u, v, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Real-image coordinates -> output-image coordinates (column, row; pixel centres at integers)."""
        x0, y0, w, h = self.crop
        return (np.asarray(u) - x0 + 0.5) * width / w - 0.5, (np.asarray(v) - y0 + 0.5) * height / h - 0.5

    # -- the lens model --
    def _theta_max(self) -> float:
        """Fisheye: the largest incidence angle (rad, <= pi) up to which r_d(theta) increases."""
        k1, k2, k3, k4 = self.D
        t = np.linspace(0.0, math.pi, 200001)
        t2 = t * t
        slope = 1 + 3 * k1 * t2 + 5 * k2 * t2 ** 2 + 7 * k3 * t2 ** 3 + 9 * k4 * t2 ** 4
        bad = np.nonzero(slope <= 0)[0]
        return float(t[bad[0] - 1]) if len(bad) else math.pi

    def _fisheye_r(self, theta):
        k1, k2, k3, k4 = self.D
        t2 = theta * theta
        return theta * (1 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4)

    def _fisheye_theta(self, rd: np.ndarray, tmax: float) -> np.ndarray:
        """theta with r_d(theta) = rd on [0, tmax] (rd < r_d(tmax)): Newton steps from theta = rd, then bisection for
        any point that has not converged to 1e-13."""
        k1, k2, k3, k4 = self.D
        theta = np.clip(rd, 0.0, tmax)
        for _ in range(12):
            t2 = theta * theta
            slope = 1 + 3 * k1 * t2 + 5 * k2 * t2 ** 2 + 7 * k3 * t2 ** 3 + 9 * k4 * t2 ** 4
            theta = np.clip(theta - (self._fisheye_r(theta) - rd) / np.maximum(slope, 1e-9), 0.0, tmax)
        bad = np.abs(self._fisheye_r(theta) - rd) > 1e-13
        if bad.any():
            lo, hi, r = np.zeros(bad.sum()), np.full(bad.sum(), tmax), rd[bad]
            for _ in range(64):                               # bisection on the monotonic part: 2^-64 of tmax
                mid = 0.5 * (lo + hi)
                big = self._fisheye_r(mid) > r
                hi, lo = np.where(big, mid, hi), np.where(big, lo, mid)
            theta[bad] = 0.5 * (lo + hi)
        return theta

    def directions(self, u, v) -> tuple[np.ndarray, np.ndarray]:
        """Undistorted unit ray directions (..., 3) float64 in the OPENCV camera frame of real-image points (u, v),
        and validity (...) bool (inside the sensor and inside the lens model's range)."""
        u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
        fx, fy, cx, cy = self.K
        inside = (u >= -0.5) & (u <= self.width - 0.5) & (v >= -0.5) & (v <= self.height - 0.5)
        if self.model == "opencv_fisheye":
            xd, yd = (u - cx) / fx, (v - cy) / fy
            rd = np.hypot(xd, yd)
            tmax = self._theta_max()
            ok = rd < self._fisheye_r(tmax)
            theta = self._fisheye_theta(np.where(ok, rd, 0.0), tmax)
            s = np.where(rd > 0, np.sin(theta) / np.where(rd > 0, rd, 1.0), 1.0)
            d = np.stack([xd * s, yd * s, np.cos(theta)], -1)
            return d / np.linalg.norm(d, axis=-1, keepdims=True), inside & ok
        import cv2
        pts = np.stack([u.ravel(), v.ravel()], -1)[:, None, :]
        D = np.array(self.D, np.float64)
        def solve(p, iters):
            crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, iters, 1e-15)
            xy = cv2.undistortPointsIter(p, self.matrix, D, None, None, crit).reshape(-1, 2)
            back, _ = cv2.projectPoints(np.c_[xy, np.ones(len(xy))].reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
                                        self.matrix, D)
            return xy, np.linalg.norm(back.reshape(-1, 2) - p[:, 0], axis=-1)

        xy, err = solve(pts, 40)
        slow = ~(err < 1e-4)
        if slow.any():                                       # strong distortion: more fixed-point iterations
            xy[slow], err[slow] = solve(pts[slow], 400)
        d = np.concatenate([xy, np.ones((len(xy), 1))], -1)
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        # the radial factor must still grow with r at the solution (else it is the far, folded branch of the model)
        k1, k2, k3, k4, k5, k6 = (*D[:2], *(D[4:8].tolist() + [0.0] * 4)[:4])
        r2 = (xy ** 2).sum(-1)
        num = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        den = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        dnum = 2 * k1 * r2 + 4 * k2 * r2 ** 2 + 6 * k3 * r2 ** 3          # r d(num)/dr
        dden = 2 * k4 * r2 + 4 * k5 * r2 ** 2 + 6 * k6 * r2 ** 3
        slope = (num + dnum) * den - num * dden                          # sign of d(r num/den)/dr
        ok = (err < 1e-3) & (slope > 0) & np.isfinite(xy).all(-1)
        return d.reshape(*u.shape, 3), inside & ok.reshape(u.shape)

    def project(self, points) -> np.ndarray:
        """Real-image coordinates (N, 2) of points (N, 3) in the OPENCV camera frame (cv2's own projection)."""
        import cv2
        p = np.asarray(points, np.float64).reshape(-1, 1, 3)
        D = np.array(self.D, np.float64)
        if self.model == "opencv_fisheye":
            out, _ = cv2.fisheye.projectPoints(p, np.zeros(3), np.zeros(3), self.matrix, D)
        else:
            out, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), self.matrix, D)
        return out.reshape(-1, 2)

    def rays(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Unit ray directions (H, W, 3) float64 in the RENDERER's camera frame (x right, y up, z backward) of the
        output pixels, and validity (H, W) bool."""
        d, ok = self.directions(*self.real_pixels(width, height))
        return d * np.array([1.0, -1.0, -1.0]), ok


def camera_rays(models, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Rays of several cameras: directions (C, H, W, 3) float64 (renderer frame) and validity (C, H, W) bool."""
    out = [m.rays(width, height) for m in models]
    return np.stack([d for d, _ in out]), np.stack([v for _, v in out])
