"""Shared code of the v8 real-data system identification (read only on all robot data; writes nothing by itself).

Episode list over both robot-log mirrors (``~/so101_r2s_claude/robot_logs`` and ``robot_logs_lab/<bundle>``), forward
kinematics of the student's kinematic point set, the calibration file, and the external-camera geometry. Adapted copy
of ``~/so101_r2s_claude/grasp_diag/tally/tally_lib.py`` (2026-10-23): hw_lib from THIS worktree, all episodes (no
start-time cut), dry / replay folders listed separately.

    PYTHONPATH=<worktree>/src ~/miniconda3/envs/so101_vision/bin/python <script>.py
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[4]
for _p in (WT / "src", WT / "scripts/so101_policy/real2sim/hw"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import hw_lib as H  # noqa: E402

R2S = Path("/home/harry/so101_r2s_claude")
LOGS = R2S / "robot_logs"
LAB_LOGS = R2S / "robot_logs_lab"
RAW = R2S / "raw"
OUT = R2S / "v8" / "sysid"
CALIB = R2S / "calib" / "calib_v1.json"
KIN_POINTS = Path("/home/harry/so101_policy/ckpt/kin_points_v1.npz")
PAN_XY = (0.0388, 0.0)
CUBE_HALF = 0.015
DT = 0.02
EXT_CROP = (52, 103, 288, 288)
THUMB = 112
STUDENT_OF_BUNDLE = {"so101_hw_robot": "s28", "so101_hw_c1": "C1", "so101_hw_bgfix_jump": "bg_b2_s403",
                     "so101_hw_bgfix_only": "bg_b1_s401", "so101_hw_v7gz": "gz_s721", "so101_hw_s28c1": "s28+C1"}

_LAYER = None
_CAL = None


def layer():
    global _LAYER
    if _LAYER is None:
        _LAYER = H.ActionLayer(KIN_POINTS, table_z=0.0)
    return _LAYER


def fk(q):
    """Sim poses [N, 6] -> grasp point [N, 3], rotation [N, 3, 3]."""
    import torch
    p, R = layer().kin.fk(torch.as_tensor(np.asarray(q, np.float64)).reshape(-1, 6))
    return p.numpy(), R.numpy()


def lowest(q):
    return layer().lowest(np.asarray(q, np.float64))


def calib():
    global _CAL
    if _CAL is None:
        _CAL = json.loads(CALIB.read_text())
    return _CAL


def table_coef():
    return np.array(calib()["scene"]["table_plane"]["coef"])


def table_z_at(xy):
    c = table_coef()
    xy = np.asarray(xy, np.float64)
    return c[0] + c[1] * xy[..., 0] + c[2] * xy[..., 1]


def real_to_sim(q_real):
    """LeRobot units [..., 6] -> sim rad [..., 6] (calib_v1 joint map)."""
    jm = calib()["joint_map"]
    q = np.asarray(q_real, np.float64)
    out = np.empty_like(q)
    s, o = np.asarray(jm["sign"][:5], float), np.asarray(jm["offset_rad"][:5], float)
    out[..., :5] = s * np.radians(q[..., :5]) + o
    a, b = jm["gripper_lin"]
    out[..., 5] = a * q[..., 5] + b
    return out


# ---- episodes ---------------------------------------------------------------------------------------------------------
def _bundle(meta, fallback):
    kp = meta.get("setup", {}).get("kin_points", "") or ""
    for b in STUDENT_OF_BUNDLE:
        if "/" + b + "/" in kp:
            return b
    return fallback


def list_episodes(with_empty=False):
    """All robot episodes (eval_*/ep_* and live_*) of both mirrors -> list of dicts, sorted by start."""
    dirs = [(LOGS, None)] + [(p, p.name) for p in sorted(LAB_LOGS.glob("so101_hw_*"))]
    out = []
    for root, lab_bundle in dirs:
        for f in sorted(glob.glob(str(root / "live_*"))) + sorted(glob.glob(str(root / "eval_*/ep_*"))):
            p = Path(f)
            rid = str(p.relative_to(root))
            npz, js = p / "episode_live.npz", p / "episode_live.json"
            meta = json.loads(js.read_text()) if js.exists() else {}
            setup = meta.get("setup", {})
            has = npz.exists() and npz.stat().st_size > 10000
            bundle = _bundle(meta, lab_bundle)
            g = setup.get("guard", {})
            session = p.name if rid.startswith("live") else p.parent.name
            e = dict(id=(lab_bundle + "/" if lab_bundle else "") + rid, path=str(npz), dir=str(p), has_data=bool(has),
                     mirror="lab" if lab_bundle else "main", session=session,
                     session_dir=str(p if rid.startswith("live") else p.parent),
                     start=meta.get("episode_start") or meta.get("start"), bundle=bundle,
                     student=STUDENT_OF_BUNDLE.get(bundle, str(setup.get("student", "?"))),
                     table_z=float(setup.get("table_z", 0.0) or 0.0), cmd_floor_m=g.get("cmd_floor_m"),
                     hard_floor_m=g.get("hard_floor_m"), servo_i=g.get("i_time_s") is not None,
                     end_reason=meta.get("end_reason"), outcome=meta.get("outcome"),
                     kind="live" if rid.startswith("live") else "eval")
            if e["start"] is None:
                continue
            if has or with_empty:
                out.append(e)
    out.sort(key=lambda e: e["start"])
    return out


def sessions():
    """Session folders with first_frames.npz (dry / live / eval; replay excluded) of both mirrors, by time."""
    out = []
    for f in sorted(glob.glob(str(LOGS / "*/first_frames.npz"))) + sorted(glob.glob(str(LAB_LOGS / "*/*/first_frames.npz"))):
        p = Path(f).parent
        if p.name.startswith("replay"):
            continue
        out.append(p)
    out.sort(key=lambda p: p.name.split("_", 1)[1])
    return out


def load(e):
    return np.load(e["path"] if isinstance(e, dict) else e, allow_pickle=True)


def pct(a, ps=(5, 25, 50, 75, 95)):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return {}
    d = {f"p{p}": float(np.percentile(a, p)) for p in ps}
    d.update(n=int(len(a)), mean=float(a.mean()), std=float(a.std()), min=float(a.min()), max=float(a.max()))
    return d


def r3(x, nd=3):
    if isinstance(x, dict):
        return {k: r3(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r3(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(round(float(x), nd))
    if isinstance(x, np.integer):
        return int(x)
    return x


# ---- external camera geometry -----------------------------------------------------------------------------------------
def quat_wxyz_to_R(q):
    w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def ext_cam():
    c = calib()["cameras"]["external"]
    T = c["T_parent_cam"]
    return dict(K=np.array(c["K"]), D=np.array(c["D"]), t=np.array(T[:3]), R=quat_wxyz_to_R(T[3:]))


def unproject_fisheye(uv, K, D, iters=20):
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


def project_cam(Xc, K, D):
    a, b = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
    r = np.sqrt(a * a + b * b)
    th = np.arctan(r)
    t2 = th * th
    k1, k2, k3, k4 = D
    thd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    s = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0)
    fx, fy, cx, cy = K
    return np.stack([fx * a * s + cx, fy * b * s + cy], -1)


def project_ext(Xw):
    c = ext_cam()
    Xc = (np.asarray(Xw, np.float64) - c["t"]) @ c["R"]
    return project_cam(Xc, c["K"], c["D"])


def thumb_to_raw(uv):
    uv = np.asarray(uv, np.float64)
    x0, y0, w, h = EXT_CROP
    return np.stack([x0 + (uv[..., 0] + 0.5) * w / THUMB - 0.5, y0 + (uv[..., 1] + 0.5) * h / THUMB - 0.5], -1)


def raw_to_thumb(uv):
    uv = np.asarray(uv, np.float64)
    x0, y0, w, h = EXT_CROP
    return np.stack([(uv[..., 0] + 0.5 - x0) * THUMB / w - 0.5, (uv[..., 1] + 0.5 - y0) * THUMB / h - 0.5], -1)


def colour_masks(img_rgb):
    """-> (green, red) soft scores in [0, 1] (cube green: sRGB ~(15, 105, 41); plate red: ~(151, 16, 25))."""
    x = img_rgb.astype(np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    green = np.clip((g - np.maximum(r, b) - 25.0) / 40.0, 0, 1)
    red = np.clip((r - np.maximum(g, b) - 45.0) / 50.0, 0, 1)
    return green, red


def blob(score, thr=0.25, min_px=3):
    import cv2
    m = (score > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[j, cv2.CC_STAT_AREA] < min_px:
        return None
    w = np.where(lab == j, score, 0.0)
    ys, xs = np.mgrid[0:score.shape[0], 0:score.shape[1]]
    mass = float(w.sum())
    return dict(cx=float((w * xs).sum() / mass), cy=float((w * ys).sum() / mass), area=int(st[j, cv2.CC_STAT_AREA]),
                mass=mass)
