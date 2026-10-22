"""Shared helpers for the real-setup calibration (SO101 real2sim, calibration agent).

Data: the 15 real LeRobot v3 recordings (untrusted data; read only, never executed). Robot model: the pinned
MuJoCo Menagerie ``so101.xml`` (world = arm base frame, z up, table top z = 0).

Joint map convention (calib_v1.json): sim_rad[i] = sign[i] * radians(real_deg[i]) + offset_rad[i] (5 arm joints),
sim jaw rad = a * real_gripper(0-100) + b.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

SCRATCH = Path(os.path.expanduser("~/so101_r2s_claude"))
RAW = SCRATCH / "raw"
CALIB = SCRATCH / "calib"
DATASETS = ["green_cube_red_plate", "green_cube_red_plate_1"]
CAMS = ["external", "wrist"]
FPS = 30
W, H = 640, 480
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

_WT = Path(__file__).resolve().parents[3]
_XML_CANDIDATES = [
    _WT / "contact_nrd/so101_grasp/vendor/mujoco_menagerie/robotstudio_so101/so101.xml",
    Path("/home/harry/.t3/worktrees/NeDM-push-ppo/t3code-2f613f84/contact_nrd/so101_grasp/vendor/"
         "mujoco_menagerie/robotstudio_so101/so101.xml"),
]
CUBE_HALF = 0.015
PLATE_HALF = 0.025
PLATE_THICK = 0.001  # assumed (thin printed plate); only used for the cube rest height on the plate


def so101_xml() -> Path:
    for p in _XML_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("so101.xml not found")


def load_dataset(ds: str) -> dict:
    """Joint data of one dataset: state / action (N, 6) LeRobot units, ep / fi (N,), episode ranges."""
    import pyarrow.parquet as pq
    root = RAW / ds
    t = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pydict()
    e = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pydict()
    out = {
        "state": np.asarray(t["observation.state"], np.float64),
        "action": np.asarray(t["action"], np.float64),
        "ep": np.asarray(t["episode_index"], np.int64),
        "fi": np.asarray(t["frame_index"], np.int64),
        "ts": np.asarray(t["timestamp"], np.float64),
        "gidx": np.asarray(t["index"], np.int64),
        "ep_from": np.asarray(e["dataset_from_index"], np.int64),
        "ep_to": np.asarray(e["dataset_to_index"], np.int64),
    }
    for cam in CAMS:
        k = f"videos/observation.images.{cam}/from_timestamp"
        vf = np.round(np.asarray(e[k]) * FPS).astype(np.int64)
        assert np.all(vf == out["ep_from"]), (ds, cam, vf, out["ep_from"])
    return out


def video_path(ds: str, cam: str) -> Path:
    return RAW / ds / f"videos/observation.images.{cam}/chunk-000/file-000.mp4"


def iter_frames(ds: str, cam: str, start: int = 0, stop: int | None = None):
    """Yield (video frame index, BGR uint8 image) sequentially."""
    import cv2
    cap = cv2.VideoCapture(str(video_path(ds, cam)))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    i = start
    while stop is None or i < stop:
        ok, im = cap.read()
        if not ok:
            break
        yield i, im
        i += 1
    cap.release()


def read_frames(ds: str, cam: str, idxs) -> dict:
    """Random access to a few frames (sorted sequential decode, robust to seeking inaccuracies)."""
    import cv2
    idxs = sorted(set(int(i) for i in idxs))
    out = {}
    cap = cv2.VideoCapture(str(video_path(ds, cam)))
    cur = 0
    for i in idxs:
        if i < cur or i - cur > 300:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, i - 0))
            cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        while cur < i:
            cap.grab()
            cur += 1
        ok, im = cap.read()
        cur += 1
        if ok:
            out[i] = im
    cap.release()
    return out


# ---- joint map -------------------------------------------------------------------------------------------------
def real_to_sim(state_deg: np.ndarray, jm: dict) -> np.ndarray:
    """(N, 6) LeRobot units -> (N, 6) sim rad (5 arm + jaw)."""
    s = np.atleast_2d(np.asarray(state_deg, np.float64))
    sign = np.asarray(jm["sign"], np.float64)[:5]
    off = np.asarray(jm["offset_rad"], np.float64)[:5]
    q = np.empty_like(s)
    q[:, :5] = sign * np.radians(s[:, :5]) + off
    a, b = jm["gripper_lin"]
    q[:, 5] = a * s[:, 5] + b
    return q


# ---- MuJoCo kinematics -----------------------------------------------------------------------------------------
class Kin:
    """Forward kinematics of so101.xml (no physics). Bodies / geoms by name."""

    def __init__(self):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(so101_xml()))
        self.data = mujoco.MjData(self.model)
        self.jadr = [self.model.joint(n).qposadr[0] for n in JOINTS]

    def set(self, q6: np.ndarray):
        self.data.qpos[:] = 0
        for k, a in enumerate(self.jadr):
            self.data.qpos[a] = q6[k]
        self.mj.mj_kinematics(self.model, self.data)
        self.mj.mj_comPos(self.model, self.data)
        self.mj.mj_camlight(self.model, self.data)

    def body_pose(self, name: str):
        b = self.model.body(name).id
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def site_pos(self, name: str):
        return self.data.site(name).xpos.copy()

    def cam_pose(self, name: str):
        c = self.model.camera(name).id
        return self.data.cam_xpos[c].copy(), self.data.cam_xmat[c].reshape(3, 3).copy()

    def poses(self, q: np.ndarray, bodies):
        """Batch FK: returns {body: (pos (N,3), R (N,3,3))}."""
        q = np.atleast_2d(q)
        out = {b: (np.empty((len(q), 3)), np.empty((len(q), 3, 3))) for b in bodies}
        for i, qi in enumerate(q):
            self.set(qi)
            for b in bodies:
                p, R = self.body_pose(b)
                out[b][0][i] = p
                out[b][1][i] = R
        return out


def quat_wxyz_to_R(q):
    w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_quat_wxyz(R):
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q


def load_json(p):
    return json.loads(Path(p).read_text())


def save_json(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=1))


# ---- batched numpy FK of the so101 chain (validated against mujoco in calib_selftest) ---------------------------
CHAIN = ["shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"]  # joint i moves body i


def _axis_angle_R(axis, ang):
    """(N,) angles about a fixed unit axis -> (N, 3, 3)."""
    ax = np.asarray(axis, np.float64)
    ax = ax / np.linalg.norm(ax)
    c, s = np.cos(ang), np.sin(ang)
    x, y, z = ax
    C = 1 - c
    R = np.empty(ang.shape + (3, 3))
    R[..., 0, 0] = c + x * x * C
    R[..., 0, 1] = x * y * C - z * s
    R[..., 0, 2] = x * z * C + y * s
    R[..., 1, 0] = y * x * C + z * s
    R[..., 1, 1] = c + y * y * C
    R[..., 1, 2] = y * z * C - x * s
    R[..., 2, 0] = z * x * C - y * s
    R[..., 2, 1] = z * y * C + x * s
    R[..., 2, 2] = c + z * z * C
    return R


class FK:
    """Batched FK: q (N, 6) sim rad -> dict body -> (p (N,3), R (N,3,3)) in the world (= base) frame."""

    def __init__(self, kin: Kin | None = None):
        kin = kin or Kin()
        m = kin.model
        self.links = []
        for b in CHAIN:
            bid = m.body(b).id
            jid = m.body_jntadr[bid]
            self.links.append(dict(name=b, pos=m.body_pos[bid].copy(), R=quat_wxyz_to_R(m.body_quat[bid]),
                                   axis=m.jnt_axis[jid].copy(), jpos=m.jnt_pos[jid].copy()))
        assert all(np.allclose(l["jpos"], 0) for l in self.links)
        bid = m.body("base").id
        self.base = (m.body_pos[bid].copy(), quat_wxyz_to_R(m.body_quat[bid]))

    def __call__(self, q):
        q = np.atleast_2d(q)
        n = len(q)
        p = np.broadcast_to(self.base[0], (n, 3)).copy()
        R = np.broadcast_to(self.base[1], (n, 3, 3)).copy()
        out = {"base": (p.copy(), R.copy())}
        for i, l in enumerate(self.links):
            if l["name"] == "moving_jaw_so101_v1":
                pg, Rg = out["gripper"]
                p, R = pg, Rg
            p = p + R @ l["pos"]
            R = R @ l["R"] @ _axis_angle_R(l["axis"], q[:, i])
            out[l["name"]] = (p.copy(), R.copy())
        out["camera_mount"] = out["gripper"]
        return out


def transform(pR, pts_local):
    """Apply body poses (p (N,3), R (N,3,3)) to local points (M,3) -> (N, M, 3)."""
    p, R = pR
    return np.einsum("nij,mj->nmi", R, np.asarray(pts_local, np.float64)) + p[:, None, :]


# ---- camera models (OpenCV conventions; camera frame x right, y down, z forward) --------------------------------
def rodrigues(rv):
    rv = np.asarray(rv, np.float64)
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def project(Xc, model, K, D):
    """Camera-frame points (..., 3) -> pixels (..., 2). model 'opencv_fisheye' (D = k1..k4) or 'opencv'
    (D = k1, k2, p1, p2, k3[, k4, k5, k6]). Points behind the camera get NaN."""
    fx, fy, cx, cy = K
    z = Xc[..., 2]
    a = Xc[..., 0] / z
    b = Xc[..., 1] / z
    if model == "opencv_fisheye":
        r = np.sqrt(a * a + b * b)
        th = np.arctan(r)
        t2 = th * th
        k1, k2, k3, k4 = D
        thd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        s = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0)
        xd, yd = a * s, b * s
    else:
        Dd = list(D) + [0.0] * (8 - len(D))
        k1, k2, p1, p2, k3, k4, k5, k6 = Dd
        r2 = a * a + b * b
        rad = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        xd = a * rad + 2 * p1 * a * b + p2 * (r2 + 2 * a * a)
        yd = b * rad + p1 * (r2 + 2 * b * b) + 2 * p2 * a * b
    u = fx * xd + cx
    v = fy * yd + cy
    bad = z <= 1e-6
    out = np.stack([u, v], -1)
    if np.any(bad):
        out = np.where(bad[..., None], np.nan, out)
    return out


def unproject_fisheye(uv, K, D, iters=20):
    """Pixels (N, 2) -> unit rays in the camera frame (N, 3) for the Kannala-Brandt model (Newton on theta)."""
    fx, fy, cx, cy = K
    xd = (uv[:, 0] - cx) / fx
    yd = (uv[:, 1] - cy) / fy
    thd = np.sqrt(xd * xd + yd * yd)
    th = thd.copy()
    k1, k2, k3, k4 = D
    for _ in range(iters):
        t2 = th * th
        f = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - thd
        df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
        th = th - f / df
    s = np.where(thd > 1e-12, np.sin(th) / np.maximum(thd, 1e-12), 1.0)
    rays = np.stack([xd * s, yd * s, np.cos(th)], -1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def unproject_opencv(uv, K, D, iters=30):
    """Pixels (N, 2) -> unit rays for the radial-tangential / rational model (OpenCV-style fixed-point undistortion;
    valid while the model is monotonic over the field of view)."""
    fx, fy, cx, cy = K
    xd = (uv[:, 0] - cx) / fx
    yd = (uv[:, 1] - cy) / fy
    Dd = list(D) + [0.0] * (8 - len(D))
    k1, k2, p1, p2, k3, k4, k5, k6 = Dd
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        rad = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (xd - dx) / rad
        y = (yd - dy) / rad
    rays = np.stack([x, y, np.ones_like(x)], -1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)
