"""Jaw tips in the wrist camera vs the gripper value (evidence for the gripper map).

Detection: white pixels (V > 150, S < 80) inside the filled convex hull of the black mat region are jaw pixels (the
floor is outside the hull). The two largest such components that reach the bottom part of the image are the jaws;
the topmost pixel of each is its tip. A tip is kept only if a 7 px ring above it is mat (dark), i.e. the tip is seen
against the mat. Output: ~/so101_r2s_claude/calib/det/{ds}_wrist_jaw.npz with tips (N, 2, 2) [left, right] NaN.

Model side (``fit``): with the calibrated wrist camera, the topmost projected vertex of the moving-jaw visual mesh
as a function of the sim jaw angle; per frame the observed right tip gives a jaw angle, and theta = a * g + b is fit
robustly.
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib_common import CALIB, DATASETS, Kin, iter_frames, load_dataset, load_json, project, quat_wxyz_to_R  # noqa
from calib_wrist_det import mat_mask  # noqa: E402


def tips(im):
    out = np.full((2, 2), np.nan, np.float32)
    m = mat_mask(im)
    if m is None:
        return out
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    H = np.zeros_like(m)
    cv2.fillConvexPoly(H, hull, 1)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    w = ((hsv[..., 2] > 150) & (hsv[..., 1] < 80)).astype(np.uint8) & H
    w = cv2.morphologyEx(w, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(w)
    comps = [j for j in range(1, n) if st[j, 4] > 150 and st[j, 1] + st[j, 3] > 330]
    comps = sorted(comps, key=lambda j: -st[j, 4])[:2]
    res = []
    for j in comps:
        ys, xs = np.where(lab == j)
        k = np.argmin(ys)
        top = ys == ys[k]
        tx, ty = xs[top].mean(), ys[k]
        # ring above the tip must be mat
        yy = int(ty) - 7
        if yy < 0:
            continue
        x0, x1 = int(tx) - 6, int(tx) + 7
        if x0 < 0 or x1 > im.shape[1]:
            continue
        if m[yy, x0:x1].mean() < 0.8:
            continue
        res.append((tx, ty))
    res.sort()
    if len(res) == 2:
        out[0], out[1] = res[0], res[1]
    elif len(res) == 1:
        out[1 if res[0][0] > 330 else 0] = res[0]
    return out


def process(args):
    ds, start, stop = args
    r = np.full((stop - start, 2, 2), np.nan, np.float32)
    for i, im in iter_frames(ds, "wrist", start, stop):
        r[i - start] = tips(im)
    return start, r


def detect():
    for ds in DATASETS:
        n = len(load_dataset(ds)["ep"])
        with Pool(12) as p:
            res = sorted(p.map(process, [(ds, s, min(n, s + 400)) for s in range(0, n, 400)]), key=lambda t: t[0])
        T = np.concatenate([r[1] for r in res])
        np.savez_compressed(CALIB / "det" / f"{ds}_wrist_jaw.npz", tips=T)
        print(ds, "left tip found", np.isfinite(T[:, 0, 0]).mean(), "right", np.isfinite(T[:, 1, 0]).mean())


class JawModel:
    """Projected topmost points of the fixed-jaw and moving-jaw meshes in the wrist image vs jaw angle."""

    def __init__(self, cam):
        self.kin = Kin()
        m = self.kin.model
        import mujoco
        self.cam = cam
        self.R_mc = quat_wxyz_to_R(cam["T_parent_cam"][3:])
        self.t_mc = np.array(cam["T_parent_cam"][:3])
        self.geo = {}
        for gid in range(m.ngeom):
            if m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[gid] != 2:
                continue
            b = m.body(m.geom_bodyid[gid]).name
            mid = m.geom_dataid[gid]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
            self.geo.setdefault(b, []).append((gid, v))

    def tip_uv(self, body, jaw):
        q = np.zeros(6)
        q[5] = jaw
        self.kin.set(q)
        d = self.kin.data
        pg, Rg = self.kin.body_pose("gripper")
        best = None
        for gid, v in self.geo[body]:
            X = v @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
            Xm = (X - pg) @ Rg  # gripper / camera_mount frame
            Xc = (Xm - self.t_mc) @ self.R_mc
            ok = Xc[:, 2] > 0.005
            uv = project(Xc[ok], self.cam["model"], np.array(self.cam["K"]), np.array(self.cam["D"]))
            inside = (uv[:, 0] > 0) & (uv[:, 0] < 640) & (uv[:, 1] > 0) & (uv[:, 1] < 480)
            uv = uv[inside]
            if len(uv) == 0:
                continue
            k = np.argmin(uv[:, 1])
            if best is None or uv[k, 1] < best[1]:
                best = uv[k]
        return best


def fit(calib_path):
    cal = load_json(calib_path)
    cam = cal["cameras"]["wrist"]
    jm = JawModel(cam)
    ang = np.radians(np.arange(-15, 101, 1.0))
    curve = np.array([jm.tip_uv("moving_jaw_so101_v1", a) for a in ang])
    fixed = jm.tip_uv("gripper", 0.0)
    G, T = [], []
    for ds in DATASETS:
        d = load_dataset(ds)
        tp = np.load(CALIB / "det" / f"{ds}_wrist_jaw.npz")["tips"]
        G.append(d["state"][:, 5])
        T.append(tp)
    G, T = np.concatenate(G), np.concatenate(T)
    lt = T[:, 0]
    okl = np.isfinite(lt[:, 0])
    print("fixed jaw tip: model", np.round(fixed, 1), "observed median", np.round(np.median(lt[okl], 0), 1),
          "n", int(okl.sum()), "MAD", np.round(np.median(np.abs(lt[okl] - np.median(lt[okl], 0)), 0), 1))
    rt = T[:, 1]
    ok = np.isfinite(rt[:, 0])
    # nearest point on the model curve -> jaw angle
    dd = np.linalg.norm(rt[ok][:, None, :] - curve[None], axis=2)
    k = np.argmin(dd, 1)
    th = ang[k]
    dist = dd[np.arange(len(k)), k]
    g = G[ok]
    sel = dist < 6
    A = np.c_[g[sel], np.ones(sel.sum())]
    coef = np.linalg.lstsq(A, th[sel], rcond=None)[0]
    for _ in range(5):  # robust re-weighting
        r = th[sel] - A @ coef
        s = 1.4826 * np.median(np.abs(r))
        w = np.abs(r) < 3 * s
        coef = np.linalg.lstsq(A[w], th[sel][w], rcond=None)[0]
    print("moving jaw: %d frames with a tip, %d near the model curve; theta = %.5f * g + %.4f rad (%.3f deg/unit, %.2f deg)"
          % (ok.sum(), sel.sum(), coef[0], coef[1], np.degrees(coef[0]), np.degrees(coef[1])))
    bins = [0, 3, 10, 13, 15, 20, 25, 30, 40, 50, 70]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = (g[sel] >= lo) & (g[sel] < hi)
        if b.sum() >= 5:
            rows.append((lo, hi, int(b.sum()), float(np.degrees(np.median(th[sel][b]))), float(np.median(g[sel][b]))))
            print(f"  g {lo:>3}-{hi:<3}: n {b.sum():5d}, median g {np.median(g[sel][b]):5.1f}, jaw from image "
                  f"{np.degrees(np.median(th[sel][b])):6.1f} deg, linear fit {np.degrees(coef[0] * np.median(g[sel][b]) + coef[1]):6.1f}")
    return dict(a=float(coef[0]), b=float(coef[1]), n=int(sel.sum()), bins=rows, fixed_model=fixed.tolist(),
                fixed_obs=np.median(lt[okl], 0).tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="")
    a = ap.parse_args()
    if a.fit:
        fit(a.fit)
    else:
        detect()
