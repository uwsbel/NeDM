"""Joint bundle adjustment of the real SO101 setup: external camera, wrist camera (pose in the MJCF camera_mount
body), joint offsets, per-episode cube / plate placements and grasp points, the mat rectangle.

Frames: world = arm base (so101.xml), table top z = 0, z up. Cameras use the OpenCV convention (x right, y down,
z forward). Joint map: sim = sign * radians(real) + offset (5 arm joints); pan offset fixed at 0 (a gauge with the
whole scene's rotation about the pan axis), wrist_roll offset fixed at 0 (a gauge with the wrist-camera mount roll and
the grasp point).

Unknowns
  global : ext K(4) D(4) rv tv | wrist K(4) D(4) rv tv (camera in camera_mount) | offsets lift, elbow, wflex |
           mat (x, y, yaw, w, h)
  episode: cube start (x, y, yaw) at z = 0.015 | cube end (x, y, yaw) at z = 0.015 + plate | plate (x, y, yaw) |
           p_grip (3, cube centre in the gripper frame while held)
Residuals (pixels unless noted)
  ext : cube start / end centroid (median over the static frames), carried-cube centroid, plate corners, mat corners,
        plumb lines (straight 3D edges)
  wrist: cube silhouette centroid before the grasp / after the release, plate corners before the grasp, mat corners
  kin : FK(q_grasp) (+) p_grip = cube start, FK(q_release) (+) p_grip = cube end (sigma 1.5 mm, 3D)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative, group_columns
from scipy.sparse import lil_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib_common import (CALIB, CUBE_HALF, DATASETS, FK, PLATE_HALF, Kin, R_to_quat_wxyz, load_dataset,  # noqa
                          load_json, project, quat_wxyz_to_R, rodrigues, save_json, transform, unproject_fisheye,
                          unproject_opencv)
from calib_ext import MAT_EXT, order_corners  # noqa: E402

PLATE_T = 0.002
MAT_T = 0.001
FIXED_TIP = np.array([-0.0089, -0.0002, -0.1044])   # fixed-jaw tip vertex, gripper frame (visual mesh, min z)
MOVING_TIP = np.array([-0.0092, -0.082, 0.0189])    # moving-jaw tip vertex, moving-jaw frame (visual mesh, min y)


def _jaw_tip_vertices():
    """Visual-mesh vertices of the jaw tip regions: fixed jaw (gripper frame, z < -0.085 m) and moving jaw
    (moving-jaw frame, y < -0.065 m), subsampled."""
    import mujoco
    kin = Kin()
    m = kin.model
    out = {}
    for gid in range(m.ngeom):
        if m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[gid] != 2:
            continue
        mesh = m.mesh(m.geom_dataid[gid]).name
        mid = m.geom_dataid[gid]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
        vb = v @ quat_wxyz_to_R(m.geom_quat[gid]).T + m.geom_pos[gid]
        if mesh == "wrist_roll_follower_so101_v1":
            out["fixed"] = vb[vb[:, 2] < -0.085]
        elif mesh == "moving_jaw_so101_v1":
            out["moving"] = vb[vb[:, 1] < -0.065]
    for k in out:
        if len(out[k]) > 400:
            out[k] = out[k][np.linspace(0, len(out[k]) - 1, 400).astype(int)]
    return out


JAW_PTS = _jaw_tip_vertices()
SIG_KIN = 0.0015
# kinematic links of the held cube: at the grasp frame only the height is sure (the closing jaw can push the cube
# sideways on the table), so z = 0.015 (sigma 1.5 mm) and xy within ~1 cm of the resting position; at the release
# frame the cube rests on the plate where the open jaw leaves it (3D, sigma 3 mm: it may drop a few mm)
SIG_GRASP = np.array([0.010, 0.010, 0.0015])
SIG_REL = np.array([0.010, 0.010, 0.006])  # released cube: drops a few mm and follows the opening jaw (measured)
CUBE_SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float64)
# faces as corner index quads (outward normal axis, sign)
FACES = []
for ax in range(3):
    for sg in (-1, 1):
        idx = [i for i in range(8) if CUBE_SIGNS[i, ax] == sg]
        # order the 4 corners cyclically in the face plane
        other = [a for a in range(3) if a != ax]
        P = CUBE_SIGNS[idx][:, other]
        ang = np.arctan2(P[:, 1], P[:, 0])
        FACES.append(([idx[i] for i in np.argsort(ang)], ax, sg))
NSUB = 2  # boundary samples per face edge
MONO_DEG, MONO_MIN = 95.0, 0.25
F_SCALE = 2.0  # soft_l1 scale used by stage_joint
NB = np.arange(-3, 4)  # neighbour frames kept for the camera-to-joint-state time offset


def deg_at(nb, dt):
    """Joint states at a fractional frame offset dt (|dt| <= 3) from neighbour samples nb (N, 7, 6)."""
    t = np.clip(dt + 3.0, 0.0, 5.999)
    i = int(np.floor(t))
    f = t - i
    return nb[:, i] * (1 - f) + nb[:, i + 1] * f


def _face_boundary_local():
    """(6, 4*NSUB, 3) boundary points of each face (unit cube corners at +-1), and normals (6, 3)."""
    B, N = [], []
    for idx, ax, sg in FACES:
        pts = []
        for k in range(4):
            a, b = CUBE_SIGNS[idx[k]], CUBE_SIGNS[idx[(k + 1) % 4]]
            for t in np.arange(NSUB) / NSUB:
                pts.append(a + (b - a) * t)
        B.append(pts)
        n = np.zeros(3)
        n[ax] = sg
        N.append(n)
    return np.array(B), np.array(N)


FACE_B, FACE_N = _face_boundary_local()


def poly_area_centroid(P):
    """P (..., M, 2) closed polygons -> signed area (...), centroid (..., 2)."""
    x, y = P[..., 0], P[..., 1]
    x1, y1 = np.roll(x, -1, -1), np.roll(y, -1, -1)
    cr = x * y1 - x1 * y
    A = 0.5 * cr.sum(-1)
    cx = ((x + x1) * cr).sum(-1) / (6 * np.where(np.abs(A) > 1e-9, A, 1e-9))
    cy = ((y + y1) * cr).sum(-1) / (6 * np.where(np.abs(A) > 1e-9, A, 1e-9))
    return A, np.stack([cx, cy], -1)


_POOL_FUN = None


def _pool_eval(z):
    return _POOL_FUN(z)


def _pm(model):
    return "opencv" if model == "opencv8" else model


def yaw_R(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.zeros(np.shape(yaw) + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1], R[..., 2, 2] = c, -s, s, c, 1
    return R


class Cam:
    """World-to-pixel for the external camera (fixed) or the wrist camera (moving with camera_mount)."""

    def __init__(self, model="opencv_fisheye"):
        self.model = model


def silhouette(cam_proj, Xc_fn, centre, Rcube, cam_c):
    """Cube silhouette (union of visible faces) centroid and area in pixels.
    cam_proj(Xw (..., 3)) -> uv; centre (N, 3), Rcube (N, 3, 3) world; cam_c (N, 3) camera centres in the world."""
    Bw = centre[:, None, None, :] + np.einsum("nij,fmj->nfmi", Rcube, FACE_B * CUBE_HALF)  # (N, 6, M, 3)
    Nw = np.einsum("nij,fj->nfi", Rcube, FACE_N)
    fc = centre[:, None, :] + Nw * CUBE_HALF
    vis = np.einsum("nfi,nfi->nf", cam_c[:, None, :] - fc, Nw) > 0
    uv = cam_proj(Bw)  # (N, 6, M, 2)
    A, C = poly_area_centroid(uv)
    A = np.abs(A) * vis
    tot = A.sum(1)
    cen = (A[..., None] * C).sum(1) / np.maximum(tot, 1e-9)[:, None]
    return cen, tot


class Data:
    def __init__(self, stride_ext=2, stride_wr=2):
        self.segs = load_json(CALIB / "segments.json")
        self.D = {ds: load_dataset(ds) for ds in DATASETS}
        self.eps = [(s["dataset"], s["episode"]) for s in self.segs]
        E = len(self.segs)
        dext = {ds: np.load(CALIB / "det" / f"{ds}_external.npz") for ds in DATASETS}
        dwr = {ds: np.load(CALIB / "det" / f"{ds}_wrist.npz") for ds in DATASETS}
        dmat = {ds: np.load(CALIB / "det" / f"{ds}_wrist_mat.npz")["corners"] for ds in DATASETS}
        self.lines = {k: np.array(v) for k, v in load_json(CALIB / "ext_lines.json").items() if len(v) >= 10}
        o = {k: [] for k in ["es_deg", "es_uv", "es_kind", "es_ep", "ec_deg", "ec_uv", "ec_ep", "ep_uv", "ep_ep",
                             "ws_deg", "ws_uv", "ws_area", "ws_kind", "ws_ep", "ws_frame", "wp_deg", "wp_uv", "wp_ep",
                             "wm_deg", "wm_cand", "wm_ep", "wm_frame", "k_deg", "k_kind", "k_ep", "wc_uv", "wc_area",
                             "wc_mad", "wc_ep", "ec_nb", "ws_nb", "wp_nb", "wm_nb"]}
        for k, s in enumerate(self.segs):
            ds, a, n = s["dataset"], s["a"], s["n"]
            st = self.D[ds]["state"]
            gr, rl = s["grasp_frame"], s["release_frame"]
            for fr, kind in [(gr, 0), (rl, 1)]:
                o["k_deg"].append(st[a + fr])
                o["k_kind"].append(kind)
                o["k_ep"].append(k)
            # external static
            for key, kind in [("ext_pre", 0), ("ext_post", 1)]:
                o["es_deg"].append(st[a + (gr if kind == 0 else rl)])
                o["es_uv"].append(s[key]["med"])
                o["es_kind"].append(kind)
                o["es_ep"].append(k)
            # external carry
            G = dext[ds]["G"][:, 0]
            ref = 0.5 * (s["ext_pre"]["area"] + s["ext_post"]["area"])
            for fr in range(gr + 8, rl - 8, stride_ext):
                g = G[a + fr]
                if g[0] < 0.6 * ref or g[0] > 1.4 * ref or not (0.5 < g[5] / max(g[6], 1) < 2.0):
                    continue
                o["ec_deg"].append(st[a + fr])
                o["ec_nb"].append(st[a + np.clip(fr + NB, 0, n - 1)])
                o["ec_uv"].append(g[1:3])
                o["ec_ep"].append(k)
            # external plate corners (pre-grasp)
            Rp = dext[ds]["Rp"][a:a + max(10, gr - 40)]
            ok = np.isfinite(Rp).all(axis=(1, 2))
            if ok.sum() >= 5:
                o["ep_uv"].append(np.median(np.array([order_corners(P) for P in Rp[ok]]), 0))
                o["ep_ep"].append(k)
            # wrist cube (static: before the grasp, after the release), plate (before the grasp), mat corners
            Gw = dwr[ds]["G"][:, 0]
            Rw = dwr[ds]["R"][:, 0]
            Rpw = dwr[ds]["Rp"]
            for fr in list(range(0, gr - 12, stride_wr)) + list(range(rl + 12, n, stride_wr)):
                kind = 0 if fr < gr else 1
                g = Gw[a + fr]
                x0, y0, w, h = g[3:7]
                if g[0] > 120 and x0 > 3 and y0 > 3 and x0 + w < 637 and y0 + h < 477 and g[0] < 15000:
                    o["ws_deg"].append(st[a + fr])
                    o["ws_nb"].append(st[a + np.clip(fr + NB, 0, n - 1)])
                    o["ws_uv"].append(g[1:3])
                    o["ws_area"].append(g[0])
                    o["ws_kind"].append(kind)
                    o["ws_ep"].append(k)
                    o["ws_frame"].append(a + fr)
                if kind == 0:
                    P = Rpw[a + fr]
                    r = Rw[a + fr]
                    if np.isfinite(P).all() and r[0] > 150:
                        pa = cv2.contourArea(P.astype(np.float32))
                        if 0.9 < r[0] / max(pa, 1) < 1.35 and r[3] > 3 and r[4] > 3 and r[3] + r[5] < 637 and \
                                r[4] + r[6] < 477:
                            o["wp_deg"].append(st[a + fr])
                            o["wp_nb"].append(st[a + np.clip(fr + NB, 0, n - 1)])
                            o["wp_uv"].append(order_corners(P.astype(np.float64)))
                            o["wp_ep"].append(k)
            car = []
            for fr in range(gr + 10, rl - 10, 2):
                g = Gw[a + fr]
                x0, y0, w, h = g[3:7]
                if g[0] > 500 and x0 > 3 and y0 > 3 and x0 + w < 637 and y0 + h < 477:
                    car.append(g[[1, 2, 0]])
            if len(car) >= 10:
                car = np.array(car)
                o["wc_uv"].append(np.median(car[:, :2], 0))
                o["wc_area"].append(np.median(car[:, 2]))
                o["wc_mad"].append(np.median(np.abs(car[:, :2] - np.median(car[:, :2], 0)), 0))
                o["wc_ep"].append(k)
            for fr in range(0, n, 3):
                c = dmat[ds][a + fr]
                c = c[np.isfinite(c[:, 0])]
                if len(c):
                    o["wm_deg"].append(st[a + fr])
                    o["wm_nb"].append(st[a + np.clip(fr + NB, 0, n - 1)])
                    cc = np.full((8, 2), np.nan)
                    cc[:len(c)] = c[:, :2]
                    o["wm_cand"].append(cc)
                    o["wm_ep"].append(k)
                    o["wm_frame"].append(a + fr)
        # jaw tips in the wrist view (calib_jaw.py): fixed tip median, moving tip median per gripper-value bin
        T = np.concatenate([np.load(CALIB / "det" / f"{ds}_wrist_jaw.npz")["tips"] for ds in DATASETS])
        G = np.concatenate([self.D[ds]["state"][:, 5] for ds in DATASETS])
        okf = np.isfinite(T[:, 0, 0])
        self.jf_uv = np.median(T[okf, 0], 0)
        jg, ju = [], []
        for lo in np.arange(0, 46, 2.0):
            b = np.isfinite(T[:, 1, 0]) & (G >= lo) & (G < lo + 2)
            if b.sum() >= 15:
                jg.append(np.median(G[b]))
                ju.append(np.median(T[b, 1], 0))
        self.jm_g, self.jm_uv = np.array(jg), np.array(ju)
        for kk, v in o.items():
            setattr(self, kk, np.array(v))
        self.E = E
        print("obs: ext static %d carry %d plates %d | wrist cube %d plate %d mat-frames %d held %d | lines %d pts" % (
            len(self.es_uv), len(self.ec_uv), len(self.ep_uv), len(self.ws_uv), len(self.wp_uv), len(self.wm_deg),
            len(self.wc_uv),
            sum(len(v) for v in self.lines.values())))


class Joint:
    GLOBAL0 = [("eK", 4), ("eD", 4), ("erv", 3), ("etv", 3), ("wK", 4), ("wD", 4), ("wrv", 3), ("wtv", 3),
               ("off", 3), ("mat", 5), ("dt", 2), ("offr", 1), ("offp", 1), ("grip", 2), ("tbl", 3), ("plt", 1)]
    LOCAL = [("cs", 3), ("ce", 3), ("pl", 3), ("pg", 3)]

    def __init__(self, data: Data, signs=(1, 1, 1, 1, 1), model="opencv_fisheye", wmodel=None):
        self.d = data
        self.signs = np.asarray(signs, np.float64)
        # camera models: 'opencv_fisheye' (k1..k4), 'opencv' (k1 k2 p1 p2 k3), 'opencv8' (+ k4 k5 k6 rational)
        self.emodel, self.wmodel = model, (wmodel or model)
        nd = {"opencv_fisheye": 4, "opencv": 5, "opencv8": 8}
        self.GLOBAL = [(k, nd[self.emodel] if k == "eD" else nd[self.wmodel] if k == "wD" else n)
                       for k, n in self.GLOBAL0]
        self.fk = FK()
        self.sl = {}
        i = 0
        for name, n in self.GLOBAL:
            self.sl[name] = slice(i, i + n)
            i += n
        self.nglob = i
        self.lsl = {}
        j = 0
        for name, n in self.LOCAL:
            self.lsl[name] = slice(j, j + n)
            j += n
        self.nloc = j
        self.n = self.nglob + self.nloc * data.E
        self.w = dict(es=1.0, ec=0.5, ep=1.0, em=1.0, line=1.0, ws=1.0, wp=1.0, wm=1.0, wc=1.0, kin=1.0, prior=1.0,
                      jaw=2.0)
        self.mat_assoc = None
        self.workers = int(os.environ.get("CALIB_WORKERS", "10"))
        self.tilt_sigma = float(os.environ.get("CALIB_TILT_SIGMA_RAD", str(np.radians(2.0))))
        # parameters held fixed inside a block: list of (block name, indices); e.g. CALIB_FREEZE="wD:2,3"
        self.freeze = []
        for item in filter(None, os.environ.get("CALIB_FREEZE", "").split(";")):
            name, idx = item.split(":")
            self.freeze.append((name, [int(v) for v in idx.split(",")]))
        self.train = np.ones(data.E, bool)
        self.use = dict(es=True, ec=True, ep=True, em=True, line=True, ws=True, wp=True, wm=True, wc=True, kin=True,
                        jaw=True)
        self._prep_states()

    # ---- parameter helpers ----
    def g(self, x, name):
        return x[self.sl[name]]

    def loc(self, x, name):
        L = x[self.nglob:].reshape(self.d.E, self.nloc)
        return L[:, self.lsl[name]]

    def loc_index(self, e, name):
        s = self.lsl[name]
        return self.nglob + e * self.nloc + np.arange(s.start, s.stop)

    # ---- kinematics ----
    def _prep_states(self):
        """Joint states of each observation, plus neighbours for the time-offset interpolation."""
        d = self.d
        self.frames = {}
        for key in ["es", "ec", "ws", "wp", "wm", "k"]:
            pass

    def q(self, deg, x):
        q = np.zeros((len(deg), 6))
        q[:, :5] = self.signs * np.radians(deg[:, :5])
        q[:, 1:4] += self.g(x, "off")
        q[:, 4] += self.g(x, "offr")[0]
        q[:, 0] += self.g(x, "offp")[0]
        return q

    def gripper_pose(self, deg, x):
        return self.fk(self.q(deg, x))["gripper"]

    # ---- cameras ----
    def ext_proj(self, x, Xw):
        R = rodrigues(self.g(x, "erv"))
        return project(Xw @ R.T + self.g(x, "etv"), _pm(self.emodel), self.g(x, "eK"), self.g(x, "eD"))

    def ext_center(self, x):
        R = rodrigues(self.g(x, "erv"))
        return -R.T @ self.g(x, "etv")

    def wr_pose(self, deg, x):
        """World pose of the wrist camera (R_wc (N,3,3), c (N,3)): X_w = R_wc X_c + c."""
        pm, Rm = self.gripper_pose(deg, x)  # camera_mount = gripper frame (identity child)
        Rmc = rodrigues(self.g(x, "wrv"))
        tmc = self.g(x, "wtv")
        Rwc = Rm @ Rmc
        c = pm + Rm @ tmc
        return Rwc, c

    def wr_proj(self, x, Rwc, c, Xw):
        """Xw (N, ..., 3) with per-row camera poses."""
        sh = Xw.shape
        if sh[0] == 0:
            return np.zeros(sh[:-1] + (2,))
        Xf = Xw.reshape(sh[0], -1, 3)
        Xc = np.einsum("nji,nmj->nmi", Rwc, Xf - c[:, None, :])
        uv = project(Xc, _pm(self.wmodel), self.g(x, "wK"), self.g(x, "wD"))
        return uv.reshape(sh[:-1] + (2,))

    # ---- world objects ----
    def plate_t(self, x):
        """Plate thickness: PLATE_T + the fitted correction (global 'plt')."""
        return PLATE_T + self.g(x, "plt")[0]

    def table_z(self, x, xy):
        t = self.g(x, "tbl")
        return t[0] + t[1] * xy[..., 0] + t[2] * xy[..., 1]

    def cube_pose(self, x, ep, kind):
        cs, ce = self.loc(x, "cs"), self.loc(x, "ce")
        P = np.where((kind == 0)[:, None], cs[ep], ce[ep])
        z = np.where(kind == 0, CUBE_HALF, CUBE_HALF + self.plate_t(x)) + self.table_z(x, P[:, :2])
        return np.c_[P[:, :2], z], yaw_R(P[:, 2])

    def plate_corners(self, pl, xp=None):
        x, y, yaw = pl[..., 0], pl[..., 1], pl[..., 2]
        loc = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * PLATE_HALF
        c, s = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
        X = loc[:, 0] * c - loc[:, 1] * s + x[..., None]
        Y = loc[:, 0] * s + loc[:, 1] * c + y[..., None]
        Z = np.full_like(X, PLATE_T)
        if xp is not None:
            Z = np.full_like(X, self.plate_t(xp)) + self.table_z(xp, np.stack([X, Y], -1))
        return np.stack([X, Y, Z], -1)

    def mat_corners(self, m, xp=None):
        x, y, yaw, w, h = m
        loc = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * [w / 2, h / 2]
        c, s = np.cos(yaw), np.sin(yaw)
        xy = loc @ np.array([[c, s], [-s, c]]) + [x, y]
        z = np.full(4, MAT_T) + (self.table_z(xp, xy) if xp is not None else 0.0)
        return np.c_[xy, z]

    @staticmethod
    def match4(proj, obs):
        best = None
        for sh in range(4):
            for flip in (False, True):
                o = obs[::-1] if flip else obs
                r = (proj - np.roll(o, sh, 0)).ravel()
                if best is None or r @ r < best @ best:
                    best = r
        return best

    @staticmethod
    def match4_batch(proj, obs):
        """proj, obs (N, 4, 2) -> (N, 8) residual with the best cyclic / flipped assignment per row."""
        cands = []
        for flip in (False, True):
            o = obs[:, ::-1] if flip else obs
            for sh in range(4):
                cands.append(proj - np.roll(o, sh, 1))
        C = np.stack(cands, 1)  # (N, 8, 4, 2)
        cost = np.nansum(C ** 2, axis=(2, 3))
        k = np.argmin(cost, 1)
        return C[np.arange(len(C)), k].reshape(len(C), 8)

    def line_res(self, x):
        K, D = self.g(x, "eK"), self.g(x, "eD")
        f = 0.5 * (K[0] + K[1])
        out = []
        for name, P in self.lines_used.items():
            r = unproject_fisheye(P, K, D) if self.emodel == "opencv_fisheye" else unproject_opencv(P, K, D)
            _, _, vt = np.linalg.svd(r, full_matrices=False)
            out.append(f * (r @ vt[-1]))
        return np.concatenate(out)

    # ---- residuals ----
    def residuals(self, x, parts=False, eps_mask=None):
        d = self.d
        m = self.train if eps_mask is None else eps_mask
        out = {}
        if self.use["es"]:
            sel = m[d.es_ep]
            X, R = self.cube_pose(x, d.es_ep[sel], d.es_kind[sel])
            C = np.broadcast_to(self.ext_center(x), X.shape)
            cen, _ = silhouette(lambda Xw: self.ext_proj(x, Xw), None, X, R, C)
            out["es"] = self.w["es"] * (cen - d.es_uv[sel]).ravel()
        if self.use["ec"] and len(d.ec_uv):
            sel = m[d.ec_ep]
            pg = self.loc(x, "pg")[d.ec_ep[sel]]
            p, R = self.gripper_pose(deg_at(d.ec_nb[sel], self.g(x, "dt")[0]), x)
            X = p + np.einsum("nij,nj->ni", R, pg)
            out["ec"] = self.w["ec"] * (self.ext_proj(x, X) - d.ec_uv[sel]).ravel()
        if self.use["ep"]:
            sel = m[d.ep_ep]
            P = self.plate_corners(self.loc(x, "pl")[d.ep_ep[sel]], x)
            out["ep"] = self.w["ep"] * self.match4_batch(self.ext_proj(x, P), d.ep_uv[sel]).ravel()
        if self.use["em"]:
            out["em"] = self.w["em"] * self.match4(self.ext_proj(x, self.mat_corners(self.g(x, "mat"), x)), MAT_EXT)
        if self.use["line"]:
            out["line"] = self.w["line"] * self.line_res(x)
        if self.use["ws"] and len(d.ws_uv):
            sel = m[d.ws_ep] & self.ws_keep
            X, R = self.cube_pose(x, d.ws_ep[sel], d.ws_kind[sel])
            Rwc, c = self.wr_pose(deg_at(d.ws_nb[sel], self.g(x, "dt")[1]), x)
            cen, area = silhouette(lambda Xw: self.wr_proj(x, Rwc, c, Xw), None, X, R, c)
            out["ws"] = self.w["ws"] * (cen - d.ws_uv[sel]).ravel()
        if self.use["wp"] and len(d.wp_uv):
            sel = m[d.wp_ep] & self.wp_keep
            P = self.plate_corners(self.loc(x, "pl")[d.wp_ep[sel]], x)
            Rwc, c = self.wr_pose(deg_at(d.wp_nb[sel], self.g(x, "dt")[1]), x)
            uv = self.wr_proj(x, Rwc, c, P)
            out["wp"] = self.w["wp"] * self.match4_batch(uv, d.wp_uv[sel]).ravel()
        if self.use["wm"] and self.mat_assoc is not None and len(self.mat_assoc[0]):
            rows, corner, obs = self.mat_assoc
            sel = m[d.wm_ep[rows]]
            Rwc, c = self.wr_pose(deg_at(d.wm_nb[rows[sel]], self.g(x, "dt")[1]), x)
            M = self.mat_corners(self.g(x, "mat"), x)[corner[sel]]
            uv = self.wr_proj(x, Rwc, c, M[:, None, :])[:, 0]
            out["wm"] = self.w["wm"] * (uv - obs[sel]).ravel()
        if self.use.get("wc", True) and len(d.wc_uv):
            sel = m[d.wc_ep]
            pg = self.loc(x, "pg")[d.wc_ep[sel]]
            Rmc = rodrigues(self.g(x, "wrv"))
            tmc = self.g(x, "wtv")
            n = int(sel.sum())
            Rwc = np.broadcast_to(Rmc, (n, 3, 3))
            c = np.broadcast_to(tmc, (n, 3))
            cen, _ = silhouette(lambda Xw: self.wr_proj(x, Rwc, c, Xw), None, pg, np.broadcast_to(np.eye(3), (n, 3, 3)),
                                c)
            out["wc"] = self.w["wc"] * (cen - d.wc_uv[sel]).ravel()
        if self.use.get("jaw", True):
            Rmc = rodrigues(self.g(x, "wrv"))
            tmc = self.g(x, "wtv")
            a, b = self.g(x, "grip")
            th = a * d.jm_g + b
            q = np.zeros((len(th), 6))
            q[:, 5] = th
            pj, Rj = self.fk(q)["moving_jaw_so101_v1"]
            pg0, Rg0 = self.fk(np.zeros((1, 6)))["gripper"]
            # silhouette extreme: the topmost projected vertex of each jaw's tip region (the detector's tip)
            Xw = pj[:, None, :] + np.einsum("nij,mj->nmi", Rj, JAW_PTS["moving"])
            Xm = (Xw - pg0[0]) @ Rg0[0]  # moving-jaw tip vertices in the gripper (camera_mount) frame
            uvs = [project((JAW_PTS["fixed"] - tmc) @ Rmc, _pm(self.wmodel), self.g(x, "wK"), self.g(x, "wD"))]
            uvs += list(project((Xm - tmc) @ Rmc, _pm(self.wmodel), self.g(x, "wK"), self.g(x, "wD")))
            top = []
            for uv in uvs:
                v = np.where(np.isfinite(uv[:, 1]), uv[:, 1], 1e9)
                k = int(np.argmin(v))
                top.append(uv[k])
            uv = np.array(top)
            obs = np.vstack([d.jf_uv[None], d.jm_uv])
            out["jaw"] = self.w["jaw"] * (uv - obs).ravel()
        if self.use["kin"]:
            sel = m[d.k_ep]
            p, R = self.gripper_pose(d.k_deg[sel], x)
            pg = self.loc(x, "pg")[d.k_ep[sel]]
            X = p + np.einsum("nij,nj->ni", R, pg)
            T, _ = self.cube_pose(x, d.k_ep[sel], d.k_kind[sel])
            sig = np.where((d.k_kind[sel] == 0)[:, None], SIG_GRASP, SIG_REL)
            out["kin"] = self.w["kin"] * ((X - T) / sig).ravel()
        # weak priors: grasp point between the jaw pads, offsets small, distortion moderate
        pg = self.loc(x, "pg")[m]
        out["prior"] = self.w["prior"] * np.r_[((pg - [0.004, 0.0, -0.075]) / [0.006, 0.006, 0.03]).ravel(),
                                               self.g(x, "off") / np.radians(20), self.g(x, "dt") / 3.0,
                                               self.g(x, "tbl") / [0.005, self.tilt_sigma, self.tilt_sigma],
                                               self.g(x, "plt") / 0.004]
        if parts:
            return out
        r = np.concatenate([v for v in out.values()])
        return np.nan_to_num(r, nan=50.0, posinf=50.0, neginf=-50.0)

    # ---- sparsity of the Jacobian (local params only affect their episode's residuals) ----
    def sparsity(self, x, free):
        """Boolean structure: residual blocks -> params; computed by perturbing each local block once."""
        r0 = self.residuals(x)
        S = lil_matrix((len(r0), int(np.sum(free))), dtype=np.int8)
        fidx = np.where(free)[0]
        col = {p: j for j, p in enumerate(fidx)}
        glob = [p for p in fidx if p < self.nglob]
        for p in glob:
            S[:, col[p]] = 1
        for e in range(self.d.E):
            ids = [p for p in fidx if self.nglob + e * self.nloc <= p < self.nglob + (e + 1) * self.nloc]
            if not ids:
                continue
            xx = x.copy()
            xx[ids] += 1e-3 * (1 + np.abs(xx[ids]))
            dr = self.residuals(xx) - r0
            rows = np.where(np.abs(dr) > 0)[0]
            for p in ids:
                S[rows, col[p]] = 1
        return S.tocsr()

    def fit(self, x, free, loss="soft_l1", f_scale=2.0, max_nfev=200, verbose=0):
        free = free.copy()
        for name, idx in self.freeze:
            free[self.sl[name].start + np.asarray(idx)] = False
        fidx = np.where(free)[0]
        x = x.copy()

        def fun(z):
            xx = x.copy()
            xx[fidx] = z
            return self.residuals(xx)

        S = self.sparsity(x, free).tocsc()
        groups = group_columns(S)
        ng = int(groups.max()) + 1
        global _POOL_FUN
        _POOL_FUN = fun
        pool = mp.get_context("fork").Pool(self.workers) if self.workers > 1 else None

        def jac(z):
            # grouped forward differences (columns touching disjoint residual rows share one evaluation), in parallel
            h = 1e-6 * np.maximum(1.0, np.abs(z))
            zs = []
            for gk in range(ng):
                zz = z.copy()
                zz[groups == gk] += h[groups == gk]
                zs.append(zz)
            r0 = fun(z)
            R1 = pool.map(_pool_eval, zs) if pool else [fun(zz) for zz in zs]
            Jm = np.zeros((len(r0), len(z)))
            for c in range(len(z)):
                rows = S.indices[S.indptr[c]:S.indptr[c + 1]]
                Jm[rows, c] = (R1[groups[c]][rows] - r0[rows]) / h[c]
            return Jm

        t0 = time.time()
        try:
            r = least_squares(fun, x[fidx], jac=jac, loss=loss, f_scale=f_scale, max_nfev=max_nfev, x_scale="jac",
                              method="trf", tr_solver="exact", verbose=verbose)
        finally:
            if pool:
                pool.close()
                pool.join()
        x[fidx] = r.x
        self.last = r
        print(f"  fit: {len(fidx)} params, {len(r.fun)} residuals, cost {r.cost:.1f}, nfev {r.nfev}, "
              f"{time.time() - t0:.1f} s, status {r.status}")
        return x

    def free_mask(self, globals_=(), locals_=(), eps=None):
        f = np.zeros(self.n, bool)
        for gname in globals_:
            f[self.sl[gname]] = True
        eps = range(self.d.E) if eps is None else eps
        for e in eps:
            for lname in locals_:
                f[self.loc_index(e, lname)] = True
        return f

    # ---- associations / outlier gating ----
    def associate_mat(self, x, gate=15.0):
        d = self.d
        Rwc, c = self.wr_pose(deg_at(d.wm_nb, self.g(x, "dt")[1]), x)
        M = self.mat_corners(self.g(x, "mat"), x)
        uv = self.wr_proj(x, Rwc, c, np.broadcast_to(M, (len(c), 4, 3)).copy())  # (N, 4, 2)
        rows, corner, obs = [], [], []
        for i in range(len(c)):
            for k in range(4):
                if not np.isfinite(uv[i, k]).all():
                    continue
                if not (8 < uv[i, k, 0] < 632 and 8 < uv[i, k, 1] < 472):
                    continue
                cand = d.wm_cand[i]
                dd = np.linalg.norm(cand - uv[i, k], axis=1)
                if np.nanmin(dd) < gate:
                    j = int(np.nanargmin(dd))
                    # unique: the second-best candidate must be clearly farther
                    rows.append(i)
                    corner.append(k)
                    obs.append(cand[j])
        self.mat_assoc = (np.array(rows, int), np.array(corner, int), np.array(obs).reshape(-1, 2))
        return len(rows)

    def gate_wrist(self, x, gate_ws=8.0, gate_area=0.25, gate_wp=8.0):
        d = self.d
        X, R = self.cube_pose(x, d.ws_ep, d.ws_kind)
        Rwc, c = self.wr_pose(deg_at(d.ws_nb, self.g(x, "dt")[1]), x)
        cen, area = silhouette(lambda Xw: self.wr_proj(x, Rwc, c, Xw), None, X, R, c)
        e = np.linalg.norm(cen - d.ws_uv, axis=1)
        ar = d.ws_area / np.maximum(area, 1)
        self.ws_keep = (e < gate_ws) & (np.abs(np.log(np.maximum(ar, 1e-3))) < gate_area) & np.isfinite(e)
        P = self.plate_corners(self.loc(x, "pl")[d.wp_ep], x)
        Rwc, c = self.wr_pose(deg_at(d.wp_nb, self.g(x, "dt")[1]), x)
        uv = self.wr_proj(x, Rwc, c, P)
        r = self.match4_batch(uv, d.wp_uv).reshape(-1, 4, 2)
        self.wp_keep = np.nanmax(np.linalg.norm(r, axis=2), 1) < gate_wp
        return self.ws_keep.mean(), self.wp_keep.mean()

    def report(self, x, eps_mask=None):
        parts = self.residuals(x, parts=True, eps_mask=eps_mask)
        rep = {}
        for k, v in parts.items():
            if k in ("prior", "mono") or len(v) == 0:
                continue
            w = self.w[k]
            if k == "kin":
                sel = (self.train if eps_mask is None else eps_mask)[self.d.k_ep]
                kind = self.d.k_kind[sel]
                sig = np.where((kind == 0)[:, None], SIG_GRASP, SIG_REL)
                r = (v / w).reshape(-1, 3) * sig * 1000
                rep["kin_grasp_z_mm"] = dict(rms=float(np.sqrt(np.mean(r[kind == 0, 2] ** 2))),
                                             max=float(np.abs(r[kind == 0, 2]).max()))
                rep["kin_grasp_xy_mm"] = dict(rms=float(np.sqrt(np.mean((r[kind == 0, :2] ** 2).sum(1)))))
                rep["kin_release_mm"] = dict(rms=float(np.sqrt(np.mean((r[kind == 1] ** 2).sum(1)))),
                                             z_rms=float(np.sqrt(np.mean(r[kind == 1, 2] ** 2))))
                continue
            if k == "line":
                e = np.abs(v / w)
            else:
                e = np.linalg.norm((v / w).reshape(-1, 2), axis=1)
            rep[k] = dict(n=int(len(e)), rms=float(np.sqrt(np.mean(e ** 2))), median=float(np.median(e)),
                          p90=float(np.percentile(e, 90)))
        return rep


# ---------------------------------------------------------------------------------------------------------------
def init_params(J: Joint, ext_fit: dict | None = None, x_glob=None):
    x = np.zeros(J.n)
    d = J.d
    if x_glob is not None:
        x[:J.nglob] = x_glob[:J.nglob]
    kin = Kin()
    # wrist camera: MJCF wrist_cam pose in camera_mount (MuJoCo camera looks along -z with y up -> OpenCV flip)
    m = kin.model
    cid = m.camera("wrist_cam").id
    R_mj = quat_wxyz_to_R(m.cam_quat[cid])
    R_cv = R_mj @ np.diag([1.0, -1.0, -1.0])
    if x_glob is None:
        x[J.sl["wrv"]] = cv2.Rodrigues(R_cv)[0].ravel()
        x[J.sl["wtv"]] = m.cam_pos[cid]
        x[J.sl["wK"]] = [280.0, 280.0, 320.0, 240.0]
    if ext_fit is not None:
        for k_src, k_dst in [("K", "eK"), ("D", "eD")]:
            x[J.sl[k_dst]] = ext_fit[k_src]
        R = np.array(ext_fit["R_world_to_cam"])
        C = np.array(ext_fit["cam_center_world"])
        x[J.sl["erv"]] = cv2.Rodrigues(R)[0].ravel()
        x[J.sl["etv"]] = -R @ C
        x[J.sl["off"]] = np.radians(ext_fit["off_deg"])
        x[J.sl["mat"]] = ext_fit["mat"]
    # episode locals: cube start / end from FK with a default grasp point
    pg0 = np.array([0.004, 0.0, -0.075])
    L = x[J.nglob:].reshape(d.E, J.nloc)
    for e in range(d.E):
        L[e, J.lsl["pg"]] = pg0
    for i in range(len(d.k_deg)):
        e, kind = d.k_ep[i], d.k_kind[i]
        p, R = J.gripper_pose(d.k_deg[i:i + 1], x)
        X = p[0] + R[0] @ pg0
        yaw = np.arctan2(R[0][1, 0], R[0][0, 0])
        name = "cs" if kind == 0 else "ce"
        L[e, J.lsl[name]] = [X[0], X[1], yaw]
        if kind == 1:
            L[e, J.lsl["pl"]] = [X[0], X[1], yaw]
    x[J.nglob:] = L.ravel()
    return x


def stage_ext(J: Joint, x, f0=200.0):
    """External camera from the static cubes / plates / mat / lines (wrist off). Robust start from PnP."""
    d = J.d
    use0 = dict(J.use)
    J.use.update(ws=False, wp=False, wm=False, wc=False)
    if not np.any(x[J.sl["eK"]]):
        x[J.sl["eK"]] = [f0, f0, 320, 240]
        X, _ = J.cube_pose(x, d.es_ep, d.es_kind)
        rays = unproject_fisheye(d.es_uv, x[J.sl["eK"]], np.zeros(4))
        ok, rv, tv = cv2.solvePnP(X, rays[:, :2] / rays[:, 2:], np.eye(3), None, flags=cv2.SOLVEPNP_SQPNP)
        x[J.sl["erv"]], x[J.sl["etv"]] = rv.ravel(), tv.ravel()
        # mat from back-projection
        R = rodrigues(x[J.sl["erv"]])
        C = -R.T @ x[J.sl["etv"]]
        dw = unproject_fisheye(MAT_EXT, x[J.sl["eK"]], np.zeros(4)) @ R
        P = C + (-C[2] / dw[:, 2])[:, None] * dw
        e1 = P[1] - P[0]
        x[J.sl["mat"]] = [*P.mean(0)[:2], np.arctan2(e1[1], e1[0]), np.linalg.norm(P[1] - P[0]),
                          np.linalg.norm(P[2] - P[1])]
    # 1) ext pose + focal with locals (no distortion), 2) everything ext + offsets
    free = J.free_mask(["eK", "erv", "etv", "mat"], ["cs", "ce", "pl", "pg"])
    x = J.fit(x, free, max_nfev=100)
    free = J.free_mask(["eK", "eD", "erv", "etv", "mat", "off"], ["cs", "ce", "pl", "pg"])
    x = J.fit(x, free, max_nfev=200)
    J.use = use0
    return x


def stage_wrist(J: Joint, x, nfev=150):
    """Wrist camera with the world objects of the external stage held fixed (cube starts / ends, plates, mat)."""
    use0 = dict(J.use)
    J.use.update(es=False, ec=False, ep=False, em=False, line=False, wm=False)
    free = J.free_mask(["wK", "wrv", "wtv"], ["pg"])
    x = J.fit(x, free, max_nfev=60, f_scale=5.0)
    free = J.free_mask(["wK", "wD", "wrv", "wtv", "dt"], ["pg"])
    x = J.fit(x, free, max_nfev=nfev, f_scale=3.0)
    J.use = use0
    return x


def wrist_check(J: Joint, x):
    d = J.d
    X, R = J.cube_pose(x, d.ws_ep, d.ws_kind)
    Rwc, c = J.wr_pose(deg_at(d.ws_nb, x[J.sl["dt"]][1]), x)
    cen, area = silhouette(lambda Xw: J.wr_proj(x, Rwc, c, Xw), None, X, R, c)
    e = np.linalg.norm(cen - d.ws_uv, axis=1)
    print("  wrist cube: median err %.1f px, p90 %.1f, area ratio median %.2f" % (
        np.nanmedian(e), np.nanpercentile(e, 90), np.nanmedian(d.ws_area / np.maximum(area, 1))))


def stage_joint(J: Joint, x, rounds=((15.0, 20.0, 80), (8.0, 10.0, 150)), fix=("offp",)):
    """Everything free; wrist observations gated and mat corners associated with the current estimate per round."""
    for gate_w, gate_m, nfev in rounds:
        kw, kp = J.gate_wrist(x, gate_ws=gate_w, gate_wp=gate_w)
        nm = J.associate_mat(x, gate=gate_m)
        print(f"  round gate {gate_w}/{gate_m}: wrist cube kept {kw:.2f}, plate kept {kp:.2f}, mat corners {nm}")
        g = [k for k, _ in J.GLOBAL if k not in fix]
        free = J.free_mask(g, ["cs", "ce", "pl", "pg"], eps=np.where(J.train)[0])  # no free params without data
        x = J.fit(x, free, max_nfev=nfev, f_scale=2.0)
        print("  ", {k: round(v["rms"], 2) if "rms" in v else v for k, v in J.report(x).items()})
    return x


def load_x(J: Joint, path):
    """Load a parameter vector; older layouts (fewer globals appended at the end of GLOBAL) are padded with 0."""
    x0 = np.load(path)
    if len(x0) == J.n:
        return x0
    ng_old = len(x0) - J.d.E * J.nloc
    x = np.zeros(J.n)
    x[:ng_old] = x0[:ng_old]
    x[J.nglob:] = x0[ng_old:]
    return x


def holdout_eval(J: Joint, x, hold_eps, nfev=60):
    """Cross-camera holdout on episodes not used for the global fit (J.train must exclude them already).
    ext holdout : their cube / plate placements and grasp points are fit from the WRIST view + kinematics only, then
                  the external cube centroids (static + carried) and plate corners are predicted.
    wrist holdout: placements fit from the EXTERNAL view + kinematics only, then the wrist cube silhouettes, plate
                  corners and held-cube centroid are predicted.
    The mat corners in the held-out frames (wrist) depend on global parameters only (direct holdout)."""
    H = np.zeros(J.d.E, bool)
    H[list(hold_eps)] = True
    train0, use0 = J.train.copy(), dict(J.use)
    out = {}
    for which, off_keys, eval_keys in [("ext", ("es", "ec", "ep"), ("es", "ec", "ep")),
                                        ("wrist", ("ws", "wp", "wc"), ("ws", "wp", "wc", "wm"))]:
        J.train = H.copy()
        J.use = dict(use0)
        for k in off_keys:
            J.use[k] = False
        J.use.update(em=False, line=False, wm=False)
        free = J.free_mask([], ["cs", "ce", "pl", "pg"], eps=np.where(H)[0])
        # neutral start for the held-out placements (FK with the default grasp point; no camera information)
        xs = x.copy()
        xi = init_params(J, x_glob=x)
        xs[free] = xi[free]
        xh = J.fit(xs, free, max_nfev=nfev, f_scale=2.0)
        J.use = dict(use0)
        rep = J.report(xh, eps_mask=H)
        out[which] = {k: rep[k] for k in eval_keys if k in rep}
        if which == "ext":
            out["ext_kin"] = {k: v for k, v in rep.items() if k.startswith("kin")}
    J.train, J.use = train0, use0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signs", default="1,1,1,1,1")
    ap.add_argument("--tag", default="joint_v1")
    ap.add_argument("--holdout", default="")
    ap.add_argument("--from-ext", default="")
    ap.add_argument("--from-wr", default="")
    a = ap.parse_args()
    data = Data()
    J = Joint(data, [int(s) for s in a.signs.split(",")])
    J.lines_used = data.lines
    J.ws_keep = np.ones(len(data.ws_uv), bool)
    J.wp_keep = np.ones(len(data.wp_uv), bool)
    if a.from_wr:
        x = load_x(J, a.from_wr)
    else:
        if a.from_ext:
            x = np.load(a.from_ext)
        else:
            x = init_params(J)
            x = stage_ext(J, x)
            np.save(CALIB / f"fits/{a.tag}_x_ext.npy", x)
        print("ext stage", J.report(x))
        x[J.sl["dt"]] = [-1.0, -1.0]
        x = stage_wrist(J, x)
        wrist_check(J, x)
        np.save(CALIB / f"fits/{a.tag}_x_wr.npy", x)
    x = stage_joint(J, x)
    np.save(CALIB / f"fits/{a.tag}_x_joint.npy", x)
    save_json(CALIB / f"fits/{a.tag}_report.json", J.report(x))


if __name__ == "__main__":
    main()
