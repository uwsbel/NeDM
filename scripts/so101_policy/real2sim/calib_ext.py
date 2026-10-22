"""External (fixed) camera + robot-kinematics bundle adjustment from the 15 real recordings.

Unknowns: camera intrinsics (fx, fy, cx, cy + distortion) and world pose, joint offsets of shoulder_lift / elbow_flex /
wrist_flex (pan / roll offsets fixed at 0: pan is a gauge with the camera yaw about the base axis, roll barely moves the
cube), the cube centre in the gripper frame while held (p_grip), the plate pose per episode, the mat rectangle.
Observations (external view): cube centroid at rest before the grasp and after the release (median over frames),
cube centroid while carried (frames with a full-size blob), plate corners (before the grasp), mat corners.
Priors: resting cube centre z = 0.015 m at the grasp frame and 0.015 + plate thickness at the release frame (sigma
2 mm), cube release xy near the plate centre (sigma 8 mm).

Usage: python calib_ext.py --signs 1,1,1,1,1 --model opencv_fisheye [--holdout ds:ep,...] [--no-carry] --tag NAME
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib_common import (CALIB, CUBE_HALF, DATASETS, FK, PLATE_HALF, load_dataset, load_json, project,  # noqa
                          rodrigues, save_json, transform, unproject_fisheye)

PLATE_T = 0.002
MAT_EXT = np.array([[210.5, 194.5], [313.5, 197.5], [330.0, 351.0], [169.5, 347.0]])  # median over 21 frames, +-1 px
NDIST = {"opencv_fisheye": 4, "opencv": 5, "opencv8": 8}


def order_corners(P):
    """Sort 4 points counter-clockwise in the image starting from the one with the smallest angle."""
    c = P.mean(0)
    ang = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
    return P[np.argsort(ang)]


class Obs:
    def __init__(self, signs, stride=2, holdout=(), area_lo=0.6, area_hi=1.4):
        self.signs = np.asarray(signs, np.float64)
        self.segs = load_json(CALIB / "segments.json")
        self.D = {ds: load_dataset(ds) for ds in DATASETS}
        self.det = {ds: np.load(CALIB / "det" / f"{ds}_external.npz") for ds in DATASETS}
        self.eps = [(s["dataset"], s["episode"]) for s in self.segs]
        self.holdout = set(holdout)
        st_deg, st_uv, st_kind, st_ep = [], [], [], []
        ca_deg, ca_uv, ca_ep = [], [], []
        pl_uv, pl_ep = [], []
        for k, s in enumerate(self.segs):
            d = self.D[s["dataset"]]
            a = s["a"]
            G = self.det[s["dataset"]]["G"][:, 0]
            for fr, key, kind in [(s["grasp_frame"], "ext_pre", 0), (s["release_frame"], "ext_post", 1)]:
                st_deg.append(d["state"][a + fr])
                st_uv.append(s[key]["med"])
                st_kind.append(kind)
                st_ep.append(k)
            ref_area = 0.5 * (s["ext_pre"]["area"] + s["ext_post"]["area"])
            for fr in range(s["grasp_frame"] + 8, s["release_frame"] - 8, stride):
                g = G[a + fr]
                if g[0] < area_lo * ref_area or g[0] > area_hi * ref_area:
                    continue
                if not (0.5 < g[5] / max(g[6], 1) < 2.0):
                    continue
                ca_deg.append(d["state"][a + fr])
                ca_uv.append(g[1:3])
                ca_ep.append(k)
            # plate corners before the grasp (frames 0 .. grasp-40), median of ordered corners
            Rp = self.det[s["dataset"]]["Rp"][a:a + max(10, s["grasp_frame"] - 40)]
            ok = np.isfinite(Rp).all(axis=(1, 2))
            if ok.sum() >= 5:
                C = np.array([order_corners(P) for P in Rp[ok]])
                pl_uv.append(np.median(C, 0))
                pl_ep.append(k)
        self.st_deg, self.st_uv = np.array(st_deg), np.array(st_uv, np.float64)
        self.st_kind, self.st_ep = np.array(st_kind), np.array(st_ep)
        self.ca_deg, self.ca_uv, self.ca_ep = np.array(ca_deg), np.array(ca_uv, np.float64), np.array(ca_ep)
        self.pl_uv, self.pl_ep = np.array(pl_uv), np.array(pl_ep)
        self.train_ep = np.array([e not in self.holdout for e in self.eps])
        print(f"static {len(self.st_uv)}, carry {len(self.ca_uv)}, plates {len(self.pl_uv)}")


class Model:
    """Parameter vector layout and residuals."""

    def __init__(self, obs: Obs, model="opencv_fisheye", use_carry=True, w_carry=0.5, sig_z=0.002, sig_plate=0.008,
                 fit_offsets=True, fit_plate=True, fit_mat=True):
        self.o = obs
        self.model = "opencv" if model == "opencv8" else model
        self.nd = NDIST[model]
        self.fk = FK()
        self.use_carry, self.w_carry = use_carry, w_carry
        self.sig_z, self.sig_plate = sig_z, sig_plate
        self.fit_offsets, self.fit_plate, self.fit_mat = fit_offsets, fit_plate, fit_mat
        n_ep = len(obs.eps)
        self.sl = {}
        i = 0
        for name, n in [("K", 4), ("D", self.nd), ("rv", 3), ("tv", 3), ("off", 3), ("pg", 3), ("plate", 3 * n_ep),
                        ("mat", 5)]:
            self.sl[name] = slice(i, i + n)
            i += n
        self.n = i

    def unpack(self, x):
        return {k: x[s] for k, s in self.sl.items()}

    def q_of(self, deg, off):
        q = np.zeros((len(deg), 6))
        q[:, :5] = self.o.signs * np.radians(deg[:, :5])
        q[:, 1:4] += off
        return q

    def cube_world(self, deg, p):
        q = self.q_of(deg, p["off"] if self.fit_offsets else np.zeros(3))
        return transform(self.fk(q)["gripper"], p["pg"][None])[:, 0]

    def cam(self, p, X):
        R = rodrigues(p["rv"])
        return project(X @ R.T + p["tv"], self.model, p["K"], p["D"])

    def plate_corners(self, pl):
        x, y, yaw = pl
        c, s = np.cos(yaw), np.sin(yaw)
        loc = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * PLATE_HALF
        xy = loc @ np.array([[c, s], [-s, c]]) + [x, y]
        return np.c_[xy, np.full(4, PLATE_T)]

    def mat_corners(self, m):
        x, y, yaw, w, h = m
        c, s = np.cos(yaw), np.sin(yaw)
        loc = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * [w / 2, h / 2]
        xy = loc @ np.array([[c, s], [-s, c]]) + [x, y]
        return np.c_[xy, np.zeros(4)]

    @staticmethod
    def match4(proj, obs):
        """Residual between 4 projected and 4 observed corners under the best cyclic assignment."""
        best = None
        for sh in range(4):
            for flip in (False, True):
                o = obs[::-1] if flip else obs
                o = np.roll(o, sh, 0)
                r = (proj - o).ravel()
                if best is None or r @ r < best @ best:
                    best = r
        return best

    def residuals(self, x, eps_mask=None, parts=False):
        p = self.unpack(x)
        o = self.o
        m = o.train_ep if eps_mask is None else eps_mask
        out = {}
        Xs = self.cube_world(o.st_deg, p)
        sel = m[o.st_ep]
        out["static_px"] = (self.cam(p, Xs[sel]) - o.st_uv[sel]).ravel()
        zt = np.where(o.st_kind == 1, CUBE_HALF + PLATE_T, CUBE_HALF)
        out["static_z"] = ((Xs[:, 2] - zt)[sel] / self.sig_z)
        if self.use_carry and len(o.ca_uv):
            selc = m[o.ca_ep]
            Xc = self.cube_world(o.ca_deg[selc], p)
            out["carry_px"] = self.w_carry * (self.cam(p, Xc) - o.ca_uv[selc]).ravel()
        if self.fit_plate:
            pl = p["plate"].reshape(-1, 3)
            rr = []
            for k, uv in zip(o.pl_ep, o.pl_uv):
                if not m[k]:
                    continue
                rr.append(self.match4(self.cam(p, self.plate_corners(pl[k])), uv))
            out["plate_px"] = np.concatenate(rr) if rr else np.zeros(0)
            end = np.where(o.st_kind == 1)[0]
            pr = [(Xs[i, :2] - pl[o.st_ep[i], :2]) / self.sig_plate for i in end if m[o.st_ep[i]]]
            out["plate_prior"] = np.concatenate(pr) if pr else np.zeros(0)
        if self.fit_mat:
            out["mat_px"] = self.match4(self.cam(p, self.mat_corners(p["mat"])), MAT_EXT)
        out["prior"] = np.r_[(p["pg"] - [0.004, 0.0, -0.08]) / [0.01, 0.01, 0.05], p["off"] / np.radians(15)]
        if parts:
            return out
        r = np.concatenate(list(out.values()))
        return np.nan_to_num(r, nan=1e3)

    def init(self, f=210.0):
        """Pinhole-on-rays PnP initialisation from the static points (equidistant guess, k = 0)."""
        o = self.o
        x = np.zeros(self.n)
        x[self.sl["K"]] = [f, f, 320, 240]
        x[self.sl["pg"]] = [0.004, 0.0, -0.08]
        p = self.unpack(x)
        Xs = self.cube_world(o.st_deg, p)
        D0 = np.zeros(4)
        rays = unproject_fisheye(o.st_uv, p["K"], D0)
        xy = rays[:, :2] / rays[:, 2:]
        ok, rv, tv = cv2.solvePnP(Xs, xy, np.eye(3), None, flags=cv2.SOLVEPNP_SQPNP)
        x[self.sl["rv"]] = rv.ravel()
        x[self.sl["tv"]] = tv.ravel()
        if self.model == "opencv":
            # start the radial-tangential model from the pinhole PnP solution
            K = np.array([[f, 0, 320], [0, f, 240], [0, 0, 1.0]])
            ok, rv, tv = cv2.solvePnP(Xs, o.st_uv, K, None, flags=cv2.SOLVEPNP_SQPNP)
            x[self.sl["rv"]], x[self.sl["tv"]] = rv.ravel(), tv.ravel()
        # plates: back-project the plate centroid to the plate plane; mat: back-project its corners to z = 0
        pl = np.zeros((len(o.eps), 3))
        for k in range(len(o.eps)):
            ends = np.where((o.st_ep == k) & (o.st_kind == 1))[0]
            pl[k, :2] = Xs[ends[0], :2]
        x[self.sl["plate"]] = pl.ravel()
        R = rodrigues(x[self.sl["rv"]])
        C = -R.T @ x[self.sl["tv"]]
        Kf = x[self.sl["K"]]
        if self.model == "opencv":
            und = cv2.undistortPoints(MAT_EXT[None].astype(np.float64), np.array([[f, 0, 320], [0, f, 240], [0, 0, 1.0]]),
                                      None)[0]
            rays = np.c_[und, np.ones(4)]
        else:
            rays = unproject_fisheye(MAT_EXT, Kf, D0)
        dw = rays @ R
        lam = -C[2] / dw[:, 2]
        P = C + lam[:, None] * dw
        c = P.mean(0)
        e1 = P[1] - P[0]
        yaw = np.arctan2(e1[1], e1[0])
        w = 0.5 * (np.linalg.norm(P[1] - P[0]) + np.linalg.norm(P[2] - P[3]))
        h = 0.5 * (np.linalg.norm(P[2] - P[1]) + np.linalg.norm(P[3] - P[0]))
        x[self.sl["mat"]] = [c[0], c[1], yaw, w, h]
        return x

    def fit(self, x0, loss="soft_l1", f_scale=2.0, max_nfev=400, stages=True):
        x = x0.copy()
        if stages:
            # stage 1: pose + focal only (distortion frozen at 0)
            free = np.zeros(self.n, bool)
            for k in ("K", "rv", "tv", "pg", "plate", "mat"):
                free[self.sl[k]] = True
            x = self._fit_subset(x, free, loss, f_scale, 200)
        free = np.ones(self.n, bool)
        if not self.fit_offsets:
            free[self.sl["off"]] = False
        return self._fit_subset(x, free, loss, f_scale, max_nfev)

    def _fit_subset(self, x, free, loss, f_scale, max_nfev):
        idx = np.where(free)[0]

        def fun(z):
            xx = x.copy()
            xx[idx] = z
            return self.residuals(xx)

        r = least_squares(fun, x[idx], loss=loss, f_scale=f_scale, max_nfev=max_nfev, x_scale="jac", method="trf")
        x = x.copy()
        x[idx] = r.x
        self.last = r
        return x

    def report(self, x, mask=None, label="train"):
        parts = self.residuals(x, eps_mask=mask, parts=True)
        out = {}
        for k in ("static_px", "carry_px", "plate_px", "mat_px"):
            if k in parts and len(parts[k]):
                r = parts[k].reshape(-1, 2)
                if k == "carry_px":
                    r = r / self.w_carry
                e = np.linalg.norm(r, axis=1)
                out[k] = dict(n=len(e), rms=float(np.sqrt((e ** 2).mean())), median=float(np.median(e)),
                              p90=float(np.percentile(e, 90)))
        if "static_z" in parts:
            out["static_z_mm"] = dict(rms=float(np.sqrt((parts["static_z"] ** 2).mean()) * self.sig_z * 1000))
        return out


def summary(model, x):
    p = model.unpack(x)
    R = rodrigues(p["rv"])
    C = -R.T @ p["tv"]
    return dict(K=p["K"].tolist(), D=p["D"].tolist(), cam_center_world=C.tolist(), R_world_to_cam=R.tolist(),
                off_deg=np.degrees(p["off"]).tolist(), p_grip=p["pg"].tolist(), mat=p["mat"].tolist(),
                plates=p["plate"].reshape(-1, 3).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signs", default="1,1,1,1,1")
    ap.add_argument("--model", default="opencv_fisheye")
    ap.add_argument("--holdout", default="")
    ap.add_argument("--no-carry", action="store_true")
    ap.add_argument("--no-offsets", action="store_true")
    ap.add_argument("--tag", default="ext")
    ap.add_argument("--f0", type=float, default=210.0)
    a = ap.parse_args()
    hold = [(DATASETS[0] if t.split(":")[0] == "0" else DATASETS[1], int(t.split(":")[1]))
            for t in a.holdout.split(",") if t]
    obs = Obs([int(s) for s in a.signs.split(",")], holdout=hold)
    m = Model(obs, a.model, use_carry=not a.no_carry, fit_offsets=not a.no_offsets)
    x0 = m.init(a.f0)
    x = m.fit(x0)
    res = dict(signs=a.signs, model=a.model, holdout=a.holdout, carry=not a.no_carry, train=m.report(x),
               cost=float(m.last.cost), x=x.tolist(), **summary(m, x))
    if hold:
        res["holdout_report"] = m.report(x, mask=~obs.train_ep, label="holdout")
    save_json(CALIB / f"fits/{a.tag}.json", res)
    print(a.tag, "train", res["train"])
    if hold:
        print(a.tag, "holdout", res["holdout_report"])
    print("K", np.round(res["K"], 2), "D", np.round(res["D"], 4), "C", np.round(res["cam_center_world"], 4),
          "off deg", np.round(res["off_deg"], 2), "pg", np.round(res["p_grip"], 4), "mat", np.round(res["mat"], 4))


if __name__ == "__main__":
    main()
