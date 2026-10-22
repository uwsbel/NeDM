"""Overlay check: project the robot's visual mesh vertices (FK at the mapped real joint angles) and the cube corners
into the real frames with the calibrated cameras (calib_v1.json contract). Writes PNG panels.

Usage: python calib_overlay.py --calib ~/so101_r2s_claude/calib/calib_v1.json --out ~/so101_r2s_claude/calib/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib_common import (CALIB, CUBE_HALF, DATASETS, Kin, load_dataset, load_json, project, quat_wxyz_to_R,  # noqa
                          read_frames, real_to_sim)

BODY_COL = {"base": (255, 128, 0), "shoulder": (255, 0, 255), "upper_arm": (0, 255, 255), "lower_arm": (0, 128, 255),
            "wrist": (255, 255, 0), "gripper": (0, 0, 255), "camera_mount": (128, 0, 255),
            "moving_jaw_so101_v1": (0, 255, 0)}


def cam_from_json(c):
    t = np.array(c["T_parent_cam"][:3])
    R = quat_wxyz_to_R(c["T_parent_cam"][3:])
    return dict(model=c["model"], K=np.array(c["K"]), D=np.array(c["D"]), R_pc=R, t_pc=t, parent=c["parent"])


class Scene:
    def __init__(self):
        import mujoco
        self.mj = mujoco
        self.kin = Kin()
        m = self.kin.model
        self.geoms = []
        rng = np.random.default_rng(0)
        for gid in range(m.ngeom):
            if m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[gid] not in (1, 2):
                continue
            mid = m.geom_dataid[gid]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
            if len(v) > 600:
                v = v[rng.choice(len(v), 600, replace=False)]
            body = m.body(m.geom_bodyid[gid]).name
            self.geoms.append((gid, v, body))

    def world_points(self, q6):
        self.kin.set(q6)
        d = self.kin.data
        out = []
        for gid, v, body in self.geoms:
            R = d.geom_xmat[gid].reshape(3, 3)
            out.append((v @ R.T + d.geom_xpos[gid], body))
        return out

    def body_pose(self, name):
        return self.kin.body_pose(name)


def world_to_cam(cam, scene, q6):
    """Return R_wc, c (camera pose in the world) for this camera at joint state q6."""
    if cam["parent"] == "world":
        return cam["R_pc"], cam["t_pc"]
    scene.kin.set(q6)
    p, R = scene.body_pose(cam["parent"])
    return R @ cam["R_pc"], p + R @ cam["t_pc"]


def proj(cam, Rwc, c, X, min_depth=0.01, max_angle_deg=95.0):
    """Project world points; points nearer than min_depth along the axis or beyond max_angle_deg off the optical
    axis are dropped (the fisheye polynomial is only fitted inside the field of view)."""
    Xc = (X - c) @ Rwc
    uv = project(Xc, cam["model"], cam["K"], cam["D"])
    ang = np.degrees(np.arctan2(np.linalg.norm(Xc[:, :2], axis=1), Xc[:, 2]))
    uv[(Xc[:, 2] < min_depth) | (ang > max_angle_deg)] = np.nan
    return uv


def cube_corners(xyz, yaw):
    s = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * CUBE_HALF
    c, sn = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
    return s @ R.T + xyz


CUBE_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]


def draw(im, cam, Rwc, c, scene, q6, cube=None, plate=None, label=""):
    vis = im.copy()
    for X, body in scene.world_points(q6):
        if cam["parent"] != "world" and body == "camera_mount":
            continue  # the camera's own mount (in the real setup the camera sits on another bracket)
        uv = proj(cam, Rwc, c, X, min_depth=0.03 if cam["parent"] != "world" else 0.01)
        col = BODY_COL.get(body, (255, 255, 255))
        for u, v in uv[np.isfinite(uv).all(1)]:
            if 0 <= u < 640 and 0 <= v < 480:
                vis[int(v), int(u)] = col
    if cube is not None:
        P = proj(cam, Rwc, c, cube_corners(*cube))
        for i, j in CUBE_EDGES:
            if np.isfinite(P[[i, j]]).all():
                cv2.line(vis, tuple(np.round(P[i]).astype(int)), tuple(np.round(P[j]).astype(int)), (0, 255, 255), 1,
                         cv2.LINE_AA)
    if plate is not None:
        x, y, yaw, zt = plate
        loc = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * 0.025
        cs, sn = np.cos(yaw), np.sin(yaw)
        xy = loc @ np.array([[cs, sn], [-sn, cs]]) + [x, y]
        P = proj(cam, Rwc, c, np.c_[xy, np.full(4, zt)])
        if np.isfinite(P).all():
            cv2.polylines(vis, [np.round(P).astype(np.int32)], True, (255, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, label, (8, 20), 0, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=str(CALIB / "calib_v1.json"))
    ap.add_argument("--out", default=str(CALIB))
    ap.add_argument("--prefix", default="overlay")
    a = ap.parse_args()
    cal = load_json(a.calib)
    jm = cal["joint_map"]
    cams = {k: cam_from_json(v) for k, v in cal["cameras"].items()}
    scene = Scene()
    eps = {(e["dataset"], e["episode"]): e for e in cal["episodes"]}
    segs = load_json(CALIB / "segments.json")
    D = {ds: load_dataset(ds) for ds in DATASETS}
    # frames: 8 episodes x phases (rest, approach, grasp, carry, release, after)
    pick_eps = [(DATASETS[0], 0), (DATASETS[0], 3), (DATASETS[1], 0), (DATASETS[1], 2), (DATASETS[1], 4),
                (DATASETS[1], 6), (DATASETS[1], 8), (DATASETS[1], 9)]
    phases = ["rest", "approach", "grasp", "carry", "release", "after"]
    dt = cal.get("time_offset_frames", {"external": 0.0, "wrist": 0.0})
    for cam_name, cam in cams.items():
        panels = []
        panels_lab = []
        for k, (ds, ep) in enumerate(pick_eps):
            s = next(t for t in segs if t["dataset"] == ds and t["episode"] == ep)
            e = eps[(ds, ep)]
            gr, rl, n = s["grasp_frame"], s["release_frame"], s["n"]
            frs = {"rest": 2, "approach": max(0, gr - 25), "grasp": gr, "carry": (gr + rl) // 2, "release": rl,
                   "after": min(n - 1, rl + 45)}
            plan = [("rest", "carry"), ("approach", "release"), ("grasp", "after"), ("rest", "grasp"),
                    ("approach", "carry"), ("release", "after"), ("grasp", "carry"), ("approach", "after")]
            for ph in plan[k]:
                fr = frs[ph]
                gi = s["a"] + fr
                im = read_frames(ds, cam_name, [gi])[gi]
                # joint state at the image time (fractional frame offset)
                st = D[ds]["state"]
                t = gi + dt[cam_name]
                i0 = int(np.clip(np.floor(t), s["a"], s["b"] - 2))
                f = float(np.clip(t - i0, 0, 1))
                deg = st[i0] * (1 - f) + st[i0 + 1] * f
                q = real_to_sim(deg, jm)[0]
                Rwc, c = world_to_cam(cam, scene, q)
                if fr < gr:
                    cube = (np.array(e["cube_start_xyz"]), np.radians(e["cube_start_yaw_deg"]))
                elif fr > rl:
                    cube = (np.array(e["cube_end_xyz"]), np.radians(e["cube_end_yaw_deg"]))
                else:
                    scene.kin.set(q)
                    p, R = scene.body_pose("gripper")
                    X = p + R @ np.array(e["p_grip"])
                    cube = (X, np.arctan2(R[1, 0], R[0, 0]))
                tp = cal["scene"].get("table_plane", {}).get("coef", [0.0, 0.0, 0.0])
                zt = tp[0] + tp[1] * e["plate_xy"][0] + tp[2] * e["plate_xy"][1] + cal["scene"]["plate_top_z"]
                plate = (e["plate_xy"][0], e["plate_xy"][1], np.radians(e["plate_yaw_deg"]), zt)
                lab = f"{cam_name} {ds[-2:]} ep{ep} f{fr} {ph}"
                panels.append(draw(im, cam, Rwc, c, scene, q, cube, plate, lab))
                panels_lab.append(lab)
        rows = [np.hstack(panels[r * 4:(r + 1) * 4]) for r in range(len(panels) // 4)]
        for r in range(0, len(rows), 2):
            out = Path(a.out) / f"{a.prefix}_{cam_name}_{r // 2}.png"
            cv2.imwrite(str(out), np.vstack(rows[r:r + 2]))
            print(out)
        if cam_name == "external":
            # zoom on the workspace (the robot is small in the external view): x 60-340, y 110-330, 2x
            crops = [cv2.resize(p_[110:330, 60:340], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST) for p_ in panels]
            for i, cp in enumerate(crops):
                cv2.putText(cp, panels_lab[i], (6, 18), 0, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            rows = [np.hstack(crops[r * 4:(r + 1) * 4]) for r in range(len(crops) // 4)]
            for r in range(0, len(rows), 2):
                out = Path(a.out) / f"{a.prefix}_external_zoom_{r // 2}.png"
                cv2.imwrite(str(out), np.vstack(rows[r:r + 2]))
                print(out)


if __name__ == "__main__":
    main()
