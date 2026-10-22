"""Wrist-camera feature detection: mat corners (convex corners of the black mat region) per frame.

Output: ~/so101_r2s_claude/calib/det/{ds}_wrist_mat.npz  with corners (N, 8, 2) NaN-padded and their turning angle.
Usage: python calib_wrist_det.py [--sheet]
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib_common import CALIB, DATASETS, iter_frames, load_dataset, read_frames  # noqa: E402

MAXC = 8
BORDER = 6
KSTEP = 12


def mat_mask(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    v = cv2.GaussianBlur(hsv[..., 2], (5, 5), 0)
    thr, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ((v < min(thr, 140)) & (hsv[..., 1] < 140)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    j = 1 + int(np.argmax(st[1:, 4]))
    if st[j, 4] < 3000:
        return None
    return (lab == j).astype(np.uint8)


def corners(im):
    """Convex corners of the mat region: contour points where the two arms (KSTEP px each, line fits) meet at an
    interior angle < 150 deg, away from the image border. Columns x, y, interior angle (deg)."""
    m = mat_mask(im)
    out = np.full((MAXC, 3), np.nan, np.float32)
    if m is None:
        return out
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    n = len(c)
    if n < 6 * KSTEP:
        return out
    h, w = m.shape
    on_border = (c[:, 0] < BORDER) | (c[:, 1] < BORDER) | (c[:, 0] > w - 1 - BORDER) | (c[:, 1] > h - 1 - BORDER)
    idx = np.arange(n)
    # direction of each arm from a 2-point chord (fast) for the candidate scan
    a = c[(idx - KSTEP) % n] - c
    b = c[(idx + KSTEP) % n] - c
    cosang = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    area2 = np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1])
    convex = cross * area2 < 0
    win = 2 * KSTEP + 4
    bord = np.convolve(np.r_[on_border[-win:], on_border, on_border[:win]].astype(float), np.ones(2 * win + 1),
                       "same")[win:-win] > 0
    cand = (ang < 150) & convex & ~bord
    k = 0
    used = np.zeros(n, bool)
    for i in np.argsort(ang):
        if not cand[i] or used[i]:
            continue
        d = np.abs(idx - i)
        used |= np.minimum(d, n - d) <= 2 * KSTEP
        armA = c[[(i - j) % n for j in range(2, KSTEP + 1)]]
        armB = c[[(i + j) % n for j in range(2, KSTEP + 1)]]
        # straight arms: line-fit residual small (a curved edge or an occluder blob fails this)
        if _line_rms(armA) > 1.0 or _line_rms(armB) > 1.0:
            continue
        p = _intersect(armA, armB)
        if p is None or np.linalg.norm(p - c[i]) > 4:
            p = c[i]
        out[k] = [p[0], p[1], ang[i]]
        k += 1
        if k == MAXC:
            break
    return out


def _line_rms(P):
    mu = P.mean(0)
    _, sv, _ = np.linalg.svd(P - mu)
    return sv[-1] / np.sqrt(len(P))


def _fit_line(P):
    mu = P.mean(0)
    _, _, vt = np.linalg.svd(P - mu)
    return mu, vt[0]


def _intersect(A, B):
    pa, da = _fit_line(A)
    pb, db = _fit_line(B)
    M = np.array([da, -db]).T
    if abs(np.linalg.det(M)) < 1e-6:
        return None
    t = np.linalg.solve(M, pb - pa)
    return pa + t[0] * da


def process(args):
    ds, start, stop = args
    res = np.full((stop - start, MAXC, 3), np.nan, np.float32)
    for i, im in iter_frames(ds, "wrist", start, stop):
        res[i - start] = corners(im)
    return start, res


def run_all(workers=12):
    for ds in DATASETS:
        n = len(load_dataset(ds)["ep"])
        chunks = [(ds, s, min(n, s + 400)) for s in range(0, n, 400)]
        with Pool(workers) as p:
            res = sorted(p.map(process, chunks), key=lambda t: t[0])
        C = np.concatenate([r[1] for r in res])
        np.savez_compressed(CALIB / "det" / f"{ds}_wrist_mat.npz", corners=C)
        print(ds, n, "frames with >=1 corner", np.isfinite(C[:, 0, 0]).mean(), "mean corners",
              np.isfinite(C[:, :, 0]).sum(1).mean())


def sheet():
    rng = np.random.default_rng(3)
    tiles = []
    for ds in DATASETS:
        n = len(load_dataset(ds)["ep"])
        idx = np.sort(rng.choice(n, 8, replace=False))
        fr = read_frames(ds, "wrist", idx)
        for i in idx:
            im = fr[i].copy()
            m = mat_mask(im)
            if m is not None:
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cnts, -1, (255, 0, 255), 1)
            for x, y, a in corners(fr[i]):
                if np.isfinite(x):
                    cv2.circle(im, (int(x), int(y)), 6, (0, 255, 255), 2)
                    cv2.putText(im, f"{a:.0f}", (int(x) + 6, int(y)), 0, 0.5, (0, 255, 255), 1)
            tiles.append(im)
    rows = [np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(4)]
    cv2.imwrite(str(CALIB / "dbg/wrist_mat_corners.jpg"), cv2.resize(np.vstack(rows), None, fx=0.5, fy=0.5))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", action="store_true")
    a = ap.parse_args()
    sheet() if a.sheet else run_all()
