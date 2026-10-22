"""Mean sRGB of the scene materials in both real cameras (decoded video frames, OpenCV BGR -> RGB 0-255).

Regions per material (frames: every 10th frame of the static phases unless noted):
  cube   : largest green blob, eroded 2 px (pre-grasp frames)
  plate  : largest red blob, eroded 2 px (pre-grasp frames)
  mat    : ext - mat quadrilateral shrunk 8 px minus coloured / robot pixels; wrist - mat mask eroded 4 px
  floor  : ext - floor band in front of the mat; wrist - ring 6-30 px outside the mat hull, bright pixels only
  arm_white / servo_black: ext - projected robot visual meshes (calibrated camera, FK at the mapped real q),
           bright (V > 110) vs dark (V < 70) pixels inside the projected white-part / servo masks (frames at rest);
           wrist - jaw pixels (white inside the mat hull) and the dark gripper-servo plate in the bottom band
Output: ~/so101_r2s_claude/calib/colors.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib_common import CALIB, DATASETS, load_dataset, load_json, read_frames, real_to_sim, save_json  # noqa
from calib_detect import masks  # noqa: E402
from calib_ext import MAT_EXT  # noqa: E402
from calib_overlay import Scene, cam_from_json, proj, world_to_cam  # noqa: E402
from calib_wrist_det import mat_mask  # noqa: E402


def srgb_to_lin(c):
    c = np.asarray(c, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


class Acc:
    def __init__(self):
        self.s = {}

    def add(self, key, im, m):
        px = im[m > 0][:, ::-1].astype(np.float64)  # RGB
        if len(px) == 0:
            return
        d = self.s.setdefault(key, [np.zeros(3), 0, [], 0])
        d[0] += px.sum(0)
        d[1] += len(px)
        d[2].append(px[:: max(1, len(px) // 200)])
        d[3] += 1

    def result(self):
        out = {}
        for k, (s, n, smp, nf) in self.s.items():
            mean = s / n
            med = np.median(np.concatenate(smp), 0)
            out[k] = dict(mean_srgb=[round(float(v), 1) for v in mean], median_srgb=[round(float(v), 1) for v in med],
                          mean_linear=[round(float(v), 4) for v in srgb_to_lin(mean)], n_pixels=int(n), n_frames=int(nf))
        return out


def robot_masks(scene, cam, q):
    """Masks of the projected white parts and the black servos (convex hulls of each geom's projected points)."""
    Rwc, c = world_to_cam(cam, scene, q)
    white = np.zeros((480, 640), np.uint8)
    black = np.zeros((480, 640), np.uint8)
    m = scene.kin.model
    for (gid, v, body), (X, _) in zip(scene.geoms, scene.world_points(q)):
        uv = proj(cam, Rwc, c, X)
        uv = uv[np.isfinite(uv).all(1)]
        if len(uv) < 3:
            continue
        hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
        mesh = m.mesh(m.geom_dataid[gid]).name
        cv2.fillConvexPoly(black if "sts3215" in mesh else white, hull, 1)
    white[black > 0] = 0
    return white, black


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=str(CALIB / "calib_v1.json"))
    a = ap.parse_args()
    cal = load_json(a.calib)
    jm = cal["joint_map"]
    cam = cam_from_json(cal["cameras"]["external"])
    scene = Scene()
    segs = load_json(CALIB / "segments.json")
    acc = {"external": Acc(), "wrist": Acc()}
    k3 = np.ones((3, 3), np.uint8)
    mat_poly = MAT_EXT.astype(np.float32)
    cen = mat_poly.mean(0)
    shrunk = (cen + (mat_poly - cen) * 0.9).astype(np.int32)
    floor_poly = np.array([[175, 362], [328, 365], [333, 425], [150, 420]], np.int32)
    for s in segs:
        ds, a0 = s["dataset"], s["a"]
        d = load_dataset(ds)
        pre = list(range(0, max(1, s["grasp_frame"] - 40), 10))
        idx = [a0 + f for f in pre]
        for camn in ("external", "wrist"):
            fr = read_frames(ds, camn, idx)
            for gi in idx:
                im = fr[gi]
                g, r = masks(im, camn)
                for key, mk in (("cube", g), ("plate", r)):
                    n, lab, st, _ = cv2.connectedComponentsWithStats(mk)
                    if n < 2:
                        continue
                    j = 1 + int(np.argmax(st[1:, 4]))
                    blob = cv2.erode((lab == j).astype(np.uint8), k3, iterations=2)
                    acc[camn].add(key, im, blob)
                hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
                if camn == "external":
                    q = real_to_sim(d["state"][gi], jm)[0]
                    white, black = robot_masks(scene, cam, q)
                    mm = np.zeros((480, 640), np.uint8)
                    cv2.fillConvexPoly(mm, shrunk, 1)
                    mm[(g > 0) | (r > 0) | (white > 0) | (black > 0)] = 0
                    mm[hsv[..., 2] > 120] = 0
                    acc[camn].add("mat", im, mm)
                    fm = np.zeros((480, 640), np.uint8)
                    cv2.fillConvexPoly(fm, floor_poly, 1)
                    acc[camn].add("floor", im, fm)
                    if gi - a0 < 20:  # robot at rest, not over the mat edge
                        we = cv2.erode(white, k3, iterations=1) & (hsv[..., 2] > 110).astype(np.uint8)
                        be = cv2.erode(black, k3, iterations=1) & (hsv[..., 2] < 70).astype(np.uint8)
                        acc[camn].add("arm_white", im, we)
                        acc[camn].add("servo_black", im, be)
                else:
                    m = mat_mask(im)
                    if m is None:
                        continue
                    mm = cv2.erode(m, k3, iterations=4)
                    mm[(g > 0) | (r > 0)] = 0
                    acc[camn].add("mat", im, mm)
                    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
                    H = np.zeros_like(m)
                    cv2.fillConvexPoly(H, hull, 1)
                    ring = cv2.dilate(H, np.ones((61, 61), np.uint8)) & ~cv2.dilate(H, np.ones((13, 13), np.uint8))
                    ring = ring & ((hsv[..., 2] > 150) & (hsv[..., 1] < 60)).astype(np.uint8)
                    ring[400:] = 0  # keep the jaws out
                    acc[camn].add("floor", im, ring)
                    jaw = ((hsv[..., 2] > 150) & (hsv[..., 1] < 80)).astype(np.uint8) & H
                    jaw[:300] = 0
                    acc[camn].add("arm_white", im, cv2.erode(jaw, k3, iterations=2))
                    band = np.zeros_like(m)
                    band[430:, 240:380] = 1
                    band &= (hsv[..., 2] < 60).astype(np.uint8)
                    acc[camn].add("servo_black", im, cv2.erode(band, k3, iterations=1))
    out = {k: v.result() for k, v in acc.items()}
    out["notes"] = ("decoded H.264 (yuv420p) frames, OpenCV BGR->RGB; pre-grasp frames of all 15 episodes, every 10th "
                    "frame; both cameras run auto exposure / white balance (UVC defaults), so values are scene-and-time "
                    "dependent")
    save_json(CALIB / "colors.json", out)
    for camn in ("external", "wrist"):
        for k, v in out[camn].items():
            print(camn, k, v["mean_srgb"], v["n_pixels"], v["n_frames"])


if __name__ == "__main__":
    main()
