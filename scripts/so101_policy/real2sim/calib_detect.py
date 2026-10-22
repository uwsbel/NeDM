"""Detect the green cube and the red plate in every frame of both real cameras (HSV thresholds + connected
components). Output per dataset / camera: ~/so101_r2s_claude/calib/det/{ds}_{cam}.npz with, per frame and colour,
the top-K blobs (area, centroid, bbox, second moments) and, for the largest blob, a 4-corner polygon fit.

Usage: python calib_detect.py [--sheet]   (``--sheet`` draws a mask check sheet on a few frames only)
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from calib_common import CALIB, CAMS, DATASETS, iter_frames, load_dataset, read_frames  # noqa: E402

K = 3
# thresholds tuned on the sample frames (OpenCV HSV: H 0-179)
GREEN = dict(h=(50, 90), s=90, v=40)
RED = dict(h_lo=8, h_hi=168, s=110, v=50)
AMIN = {"external": 12, "wrist": 80}
EXT_XMIN = 80  # external view: equipment LEDs outside the tent at x < 70 (bottom-left); all task pixels at x > 180


def masks(im, cam=None):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    g = (h >= GREEN["h"][0]) & (h <= GREEN["h"][1]) & (s >= GREEN["s"]) & (v >= GREEN["v"])
    r = ((h <= RED["h_lo"]) | (h >= RED["h_hi"])) & (s >= RED["s"]) & (v >= RED["v"])
    ker = np.ones((3, 3), np.uint8)
    g = cv2.morphologyEx(g.astype(np.uint8), cv2.MORPH_OPEN, ker)
    r = cv2.morphologyEx(r.astype(np.uint8), cv2.MORPH_OPEN, ker)
    if cam == "external":
        g[:, :EXT_XMIN] = 0
        r[:, :EXT_XMIN] = 0
    return g, r


def blobs(m, amin):
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for j in range(1, n):
        a = st[j, cv2.CC_STAT_AREA]
        if a < amin:
            continue
        out.append((a, j))
    out.sort(reverse=True)
    rows = np.zeros((K, 10), np.float32)  # area, cx, cy, x0, y0, w, h, mu20, mu02, mu11 (central, normalised)
    poly = np.full((4, 2), np.nan, np.float32)
    hull = np.full((24, 2), np.nan, np.float32)
    for k, (a, j) in enumerate(out[:K]):
        x0, y0, w, h = st[j, :4]
        sub = (lab[y0:y0 + h, x0:x0 + w] == j).astype(np.uint8)
        mo = cv2.moments(sub, binaryImage=True)
        cx, cy = mo["m10"] / mo["m00"] + x0, mo["m01"] / mo["m00"] + y0
        rows[k] = [a, cx, cy, x0, y0, w, h, mo["mu20"] / a, mo["mu02"] / a, mo["mu11"] / a]
        if k == 0:
            cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cnts, key=cv2.contourArea) + np.array([x0, y0])
            hl = cv2.convexHull(c)[:, 0, :].astype(np.float32)
            if len(hl) > 24:
                hl = hl[np.linspace(0, len(hl) - 1, 24).astype(int)]
            hull[:len(hl)] = hl
            ap = cv2.approxPolyDP(cv2.convexHull(c), 0.04 * cv2.arcLength(cv2.convexHull(c), True), True)[:, 0, :]
            if len(ap) == 4:
                poly[:] = ap
    return rows, poly, hull, len(out)


def process(args):
    ds, cam, start, stop = args
    amin = AMIN[cam]
    n = stop - start
    G = np.zeros((n, K, 10), np.float32)
    R = np.zeros((n, K, 10), np.float32)
    Gp = np.zeros((n, 4, 2), np.float32)
    Rp = np.zeros((n, 4, 2), np.float32)
    Gh = np.zeros((n, 24, 2), np.float32)
    Rh = np.zeros((n, 24, 2), np.float32)
    ng = np.zeros(n, np.int16)
    nr = np.zeros(n, np.int16)
    for i, im in iter_frames(ds, cam, start, stop):
        g, r = masks(im, cam)
        k = i - start
        G[k], Gp[k], Gh[k], ng[k] = blobs(g, amin)
        R[k], Rp[k], Rh[k], nr[k] = blobs(r, amin)
    return start, dict(G=G, R=R, Gp=Gp, Rp=Rp, Gh=Gh, Rh=Rh, ng=ng, nr=nr)


def run_all(workers=12):
    out_dir = CALIB / "det"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        d = load_dataset(ds)
        n = len(d["ep"])
        for cam in CAMS:
            chunks = [(ds, cam, s, min(n, s + 400)) for s in range(0, n, 400)]
            with Pool(workers) as p:
                res = sorted(p.map(process, chunks), key=lambda t: t[0])
            cat = {k: np.concatenate([r[1][k] for r in res]) for k in res[0][1]}
            assert len(cat["G"]) == n, (ds, cam, len(cat["G"]), n)
            np.savez_compressed(out_dir / f"{ds}_{cam}.npz", **cat)
            print(ds, cam, n, "green found", (cat["G"][:, 0, 0] > 0).mean(), "red found", (cat["R"][:, 0, 0] > 0).mean())


def sheet():
    rng = np.random.default_rng(1)
    for cam in CAMS:
        tiles = []
        for ds in DATASETS:
            n = len(load_dataset(ds)["ep"])
            idx = np.sort(rng.choice(n, 6, replace=False))
            fr = read_frames(ds, cam, idx)
            for i in idx:
                im = fr[i]
                g, r = masks(im, cam)
                vis = im.copy()
                vis[g > 0] = (255, 0, 255)
                vis[r > 0] = (255, 255, 0)
                G, Gp, _, _ = blobs(g, AMIN[cam])
                R, Rp, _, _ = blobs(r, AMIN[cam])
                for row, col in [(G[0], (0, 255, 0)), (R[0], (0, 0, 255))]:
                    if row[0] > 0:
                        cv2.circle(vis, (int(row[1]), int(row[2])), 4, col, -1)
                for P in (Gp, Rp):
                    if np.isfinite(P).all():
                        cv2.polylines(vis, [P.astype(np.int32)], True, (0, 255, 255), 1)
                tiles.append(np.hstack([im, vis]))
        rows = [np.hstack(tiles[r * 2:(r + 1) * 2]) for r in range(6)]
        cv2.imwrite(str(CALIB / f"dbg/masks_{cam}.jpg"), cv2.resize(np.vstack(rows), None, fx=0.4, fy=0.4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    sheet() if a.sheet else run_all()
