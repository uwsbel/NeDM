"""The student's view of a real camera: ONE crop box of the 640 x 480 frame and ONE output size, shared by the
renderer (``nedm.render.so101.RealScene``: the rays of each output pixel) and the real-robot code
(``scripts/so101_policy/real2sim/hw/hw_lib.py``: crop + resize of the camera frame).

numpy only (cv2 inside :func:`crop_resize`): the robot's Jetson has no mujoco, newton or warp, and ``hw_lib`` loads
this file by path (no ``nedm.render`` package import).

Spec layout (``real_scene_v1.json`` / ``calib_v1.json`` key ``"student_view"``), per camera::

    {"crop": [x0, y0, w, h], "out": [W, H], "interp": "area"}

``"crop_xywh"`` is the same as ``"crop"``; ``"crop_xyxy": [x0, y0, x1, y1]`` is also accepted; ``"out_wh"`` is the
same as ``"out"``. No entry: a centred square of side min(width, height) (the old ``hw_lib`` default).

Pixel convention (both sides): output pixel (i, j) covers the crop-box area [x0 + j w/W, x0 + (j+1) w/W) x [y0 + i h/H,
y0 + (i+1) h/H) in real-pixel EDGE coordinates; its centre is the real point u = x0 + (j + 0.5) w / W - 0.5,
v = y0 + (i + 0.5) h / H - 0.5 (pixel centres at integers, OpenCV). ``cv2.resize(..., INTER_AREA)`` averages over that
area; the renderer samples it at ``supersample``^2 sub-pixel centres and averages (same centre).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WIDTH, HEIGHT = 640, 480                # the real camera frames (LeRobot, both cameras)
_INTERP = {"area": 3, "linear": 1, "nearest": 0, "cubic": 2}          # cv2.INTER_* values


@dataclass
class ViewSpec:
    """Crop box (x, y, w, h in the camera image) and output size (W, H) of one student camera."""
    crop_xywh: tuple
    out_wh: tuple
    interp: str = "area"
    source: str = ""

    def __post_init__(self):
        self.crop_xywh = tuple(int(v) for v in self.crop_xywh)
        self.out_wh = tuple(int(v) for v in self.out_wh)
        x, y, w, h = self.crop_xywh
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            raise ValueError(f"bad crop {self.crop_xywh}")
        if self.interp not in _INTERP:
            raise ValueError(f"interp {self.interp!r}: one of {sorted(_INTERP)}")

    def check_inside(self, width: int = WIDTH, height: int = HEIGHT) -> None:
        x, y, w, h = self.crop_xywh
        if x + w > width or y + h > height:
            raise ValueError(f"crop {self.crop_xywh} outside the {width} x {height} image")

    def to_dict(self) -> dict:
        return {"crop": list(self.crop_xywh), "out": list(self.out_wh), "interp": self.interp}


def centre_square(width=WIDTH, height=HEIGHT, out=224, interp="area") -> ViewSpec:
    s = min(width, height)
    return ViewSpec(((width - s) // 2, (height - s) // 2, s, s), (out, out), interp, "centre square (no student_view)")


def parse_view(d: dict, out_res: int | None = None, interp: str = "area", source: str = "student_view") -> ViewSpec:
    """One camera's entry (module docstring) -> ViewSpec. ``out_res``: the default square output if the entry has
    none."""
    if "crop" in d:
        box = d["crop"]
    elif "crop_xywh" in d:
        box = d["crop_xywh"]
    elif "crop_xyxy" in d:
        x0, y0, x1, y1 = d["crop_xyxy"]
        box = (x0, y0, x1 - x0, y1 - y0)
    else:
        raise ValueError(f"student_view entry {d!r}: needs 'crop', 'crop_xywh' or 'crop_xyxy'")
    wh = d.get("out", d.get("out_wh"))
    if wh is None:
        if out_res is None:
            raise ValueError(f"student_view entry {d!r}: no 'out' size and no default")
        wh = (out_res, out_res)
    return ViewSpec(box, wh, d.get("interp", interp), source)


def view_specs(calib, real_cams, out_res, width=WIDTH, height=HEIGHT, interp="area") -> dict:
    """{camera: ViewSpec} from ``calib["student_view"][camera]`` (``calib``: any dict with that key, e.g. the
    calibration contract or real_scene_v1.json); cameras without an entry get the centred square."""
    sv = (calib or {}).get("student_view") or {}
    out = {}
    for cam in real_cams:
        d = sv.get(cam)
        out[cam] = centre_square(width, height, out_res, interp) if d is None else parse_view(d, out_res, interp)
    return out


def crop_resize(img, spec: ViewSpec, bgr: bool = False) -> np.ndarray:
    """Camera image uint8 [H, W, 3] (BGR if ``bgr``: cv2 decode; RGB: LeRobot camera with color_mode rgb) -> the
    student's RGB uint8 [out_h, out_w, 3]: crop box, then resize (cv2, ``spec.interp``)."""
    import cv2
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError(f"expected uint8 [H, W, 3], got {a.dtype} {a.shape}")
    x, y, w, h = spec.crop_xywh
    if y + h > a.shape[0] or x + w > a.shape[1]:
        raise ValueError(f"crop {spec.crop_xywh} outside the image {a.shape[1]} x {a.shape[0]}")
    c = a[y:y + h, x:x + w]
    if bgr:
        c = c[..., ::-1]
    if (w, h) != spec.out_wh:
        c = cv2.resize(np.ascontiguousarray(c), spec.out_wh, interpolation=_INTERP[spec.interp])
    return np.ascontiguousarray(c)


def output_pixel_centres(spec: ViewSpec, supersample: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Real-image coordinates (u, v), each (H_out * s, W_out * s) float64, of the (sub-)pixel centres the renderer
    samples: s x s sub-pixels per output pixel, whose mean is the output pixel centre of the module docstring."""
    x0, y0, w, h = spec.crop_xywh
    W, H = spec.out_wh
    s = int(supersample)
    u = x0 + (np.arange(W * s) + 0.5) * w / (W * s) - 0.5
    v = y0 + (np.arange(H * s) + 0.5) * h / (H * s) - 0.5
    return np.broadcast_to(u[None, :], (H * s, W * s)).copy(), np.broadcast_to(v[:, None], (H * s, W * s)).copy()
