"""Plumb-line points of the external camera: edges that are straight in 3D (tent wall / floor edges, the table edge,
the mat sides), refined to sub-pixel on the median background image of dataset green_cube_red_plate_1.

Rough polylines were read off the median image by eye (original pixel coordinates); each sample is moved along the
normal to the strongest gradient within +-RAD px. Output: ~/so101_r2s_claude/calib/ext_lines.json + a check image.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib_common import CALIB, read_frames, save_json  # noqa: E402

RAD = 6
LINES = {
    "back_left_vertical": [(178, 18), (181, 148)],
    "back_floor": [(186, 153), (355, 160)],
    "back_right_vertical": [(360, 155), (377, 58)],
    "right_floor": [(361, 168), (395, 280), (430, 430)],
    "table_edge_left": [(122, 292), (108, 360), (93, 428)],
    "front_floor": [(110, 446), (220, 468), (330, 466), (430, 446)],
    "mat_top": [(216, 194), (308, 197)],
    "mat_right": [(315, 205), (328, 343)],
    "mat_bottom": [(176, 347), (323, 351)],
    "mat_left": [(207, 202), (172, 340)],
}


def median_image():
    p = CALIB / "dbg/ext_median.png"
    if p.exists():
        return cv2.imread(str(p))
    idx = list(range(0, 6632, 97))
    fr = read_frames("green_cube_red_plate_1", "external", idx)
    med = np.median(np.stack([fr[i] for i in sorted(fr)]), 0).astype(np.uint8)
    cv2.imwrite(str(p), med)
    return med


def refine(gray, poly, step=3.0):
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    pts = []
    poly = np.asarray(poly, np.float64)
    for a, b in zip(poly[:-1], poly[1:]):
        L = np.linalg.norm(b - a)
        t = (b - a) / L
        nrm = np.array([-t[1], t[0]])
        for s in np.arange(4, L - 4, step):
            c = a + t * s
            offs = np.arange(-RAD, RAD + 0.01, 0.5)
            P = c[None] + offs[:, None] * nrm[None]
            xi, yi = P[:, 0], P[:, 1]
            if xi.min() < 1 or yi.min() < 1 or xi.max() > gray.shape[1] - 2 or yi.max() > gray.shape[0] - 2:
                continue
            gxs = cv2.remap(gx, xi.astype(np.float32)[None], yi.astype(np.float32)[None], cv2.INTER_LINEAR)[0]
            gys = cv2.remap(gy, xi.astype(np.float32)[None], yi.astype(np.float32)[None], cv2.INTER_LINEAR)[0]
            mag = np.abs(gxs * nrm[0] + gys * nrm[1])
            k = int(np.argmax(mag))
            if mag[k] < 20 or k == 0 or k == len(mag) - 1:
                continue
            y0, y1, y2 = mag[k - 1], mag[k], mag[k + 1]
            den = y0 - 2 * y1 + y2
            sub = 0.5 * (y0 - y2) / den if abs(den) > 1e-6 else 0.0
            pts.append((c + (offs[k] + 0.5 * sub) * nrm).tolist())
    return pts


def main():
    med = median_image()
    gray = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
    out = {}
    vis = cv2.resize(med, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    for name, poly in LINES.items():
        pts = refine(gray, poly)
        out[name] = pts
        for x, y in pts:
            cv2.circle(vis, (int(round(2 * x)), int(round(2 * y))), 2, (0, 0, 255), -1)
        print(name, len(pts))
    save_json(CALIB / "ext_lines.json", out)
    cv2.imwrite(str(CALIB / "dbg/ext_lines.png"), vis)


if __name__ == "__main__":
    main()
