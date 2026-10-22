"""Write calib_v1.json (contract of the real2sim study) from a joint bundle-adjustment solution.

Usage: python calib_export.py --x fits/FULL_x_joint.npy --holdout fits/HOLD.json --jaw fits/jaw.json \
           --colors calib/colors.json --out calib/calib_v1.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calib_common import CALIB, CUBE_HALF, JOINTS, R_to_quat_wxyz, load_json, rodrigues, save_json  # noqa: E402
from calib_joint import MAT_T, PLATE_T, Data, Joint  # noqa: E402

ANNULUS = dict(center_xy=(0.0388, 0.0), r_min=0.15, r_max=0.27, angle_max_deg=75.0)
GOAL_DIST = (0.06, 0.30)
GOAL_TURN_DEG = 30.0


def wrap90(deg):
    """Yaw of a square (mod 90) in [-45, 45)."""
    return float((deg + 45.0) % 90.0 - 45.0)


def on_annulus(xy):
    c = np.asarray(ANNULUS["center_xy"])
    d = np.asarray(xy) - c
    r = float(np.hypot(*d))
    ang = float(np.degrees(np.arctan2(d[1], d[0])))
    return r, ang, bool(ANNULUS["r_min"] <= r <= ANNULUS["r_max"] and abs(ang) <= ANNULUS["angle_max_deg"])


def export(J: Joint, x, rep_full, hold=None, jaw=None, colors=None, signs=(1, 1, 1, 1, 1), extra=None):
    d = J.d
    off = J.g(x, "off")
    sign6 = [int(v) for v in J.signs] + [1]
    offset6 = [float(J.g(x, "offp")[0]), float(off[0]), float(off[1]), float(off[2]), float(J.g(x, "offr")[0]), 0.0]
    ga, gb = J.g(x, "grip")
    gl = [float(jaw["a"]), float(jaw["b"])] if jaw else [float(ga), float(gb)]
    jm = dict(names=list(JOINTS), sign=sign6, offset_rad=offset6, gripper_lin=gl,
              offset_deg=[float(np.degrees(v)) for v in offset6])
    # cameras
    R = rodrigues(J.g(x, "erv"))
    C = -R.T @ J.g(x, "etv")
    q_ext = R_to_quat_wxyz(R.T)
    Rmc = rodrigues(J.g(x, "wrv"))
    tmc = J.g(x, "wtv")
    q_wr = R_to_quat_wxyz(Rmc)

    def stats(keys):
        n, se = 0, 0.0
        for k in keys:
            if k in rep_full:
                n += rep_full[k]["n"]
                se += rep_full[k]["n"] * rep_full[k]["rms"] ** 2
        return (float(np.sqrt(se / max(n, 1))), int(n))

    def hstats(part, keys):
        if not hold:
            return None
        n, se = 0, 0.0
        for k in keys:
            if k in hold[part]:
                n += hold[part][k]["n"]
                se += hold[part][k]["n"] * hold[part][k]["rms"] ** 2
        return float(np.sqrt(se / max(n, 1)))

    e_rms, e_n = stats(["es", "ec", "ep", "em"])
    w_rms, w_n = stats(["ws", "wp", "wm", "wc"])
    cams = {
        "external": dict(model=J.emodel if J.emodel != "opencv8" else "opencv", width=640, height=480,
                         K=J.g(x, "eK").tolist(), D=J.g(x, "eD").tolist(), parent="world",
                         T_parent_cam=[*C.tolist(), *q_ext.tolist()], rms_px=e_rms, n_points=e_n,
                         holdout_rms_px=hstats("ext", ["es", "ec", "ep"]),
                         rms_by_kind={k: rep_full[k] for k in ("es", "ec", "ep", "em", "line") if k in rep_full},
                         holdout_by_kind=hold["ext"] if hold else None),
        "wrist": dict(model=J.wmodel if J.wmodel != "opencv8" else "opencv", width=640, height=480,
                      K=J.g(x, "wK").tolist(), D=J.g(x, "wD").tolist(), parent="camera_mount",
                      T_parent_cam=[*tmc.tolist(), *q_wr.tolist()], rms_px=w_rms, n_points=w_n,
                      holdout_rms_px=hstats("wrist", ["ws", "wp", "wc", "wm"]),
                      rms_by_kind={k: rep_full[k] for k in ("ws", "wp", "wm", "wc") if k in rep_full},
                      holdout_by_kind=hold["wrist"] if hold else None),
    }
    from calib_common import unproject_fisheye
    for name, c in cams.items():
        K, D = np.array(c["K"]), np.array(c["D"])
        pts = {"left": [0, K[3]], "right": [639, K[3]], "top": [K[2], 0], "bottom": [K[2], 479],
               "tl": [0, 0], "tr": [639, 0], "bl": [0, 479], "br": [639, 479]}
        ang = {k: float(np.degrees(np.arccos(unproject_fisheye(np.array([v], float), K, D)[0, 2]))) for k, v in pts.items()}
        th = np.radians(np.arange(0, 100.1, 0.5))
        t2 = th * th
        der = 1 + t2 * (3 * D[0] + t2 * (5 * D[1] + t2 * (7 * D[2] + t2 * 9 * D[3])))
        fold = th[np.where(der <= 0)[0][0]] if (der <= 0).any() else None
        c["field_of_view_deg"] = dict(horizontal=ang["left"] + ang["right"], vertical=ang["top"] + ang["bottom"],
                                      off_axis_angle_at=ang)
        c["monotonic_up_to_deg"] = float(np.degrees(fold)) if fold is not None else 100.0
    dtv = J.g(x, "dt")
    # episodes
    cs, ce, pl, pg = J.loc(x, "cs"), J.loc(x, "ce"), J.loc(x, "pl"), J.loc(x, "pg")
    eps = []
    # cube yaw from the jaw orientation (cube faces parallel to the pads at the grasp / release frame)
    pk, Rk = J.gripper_pose(d.k_deg, x)
    jaw_yaw = {}
    for i in range(len(d.k_deg)):
        ax = Rk[i][:, 0]  # jaw closing axis (gripper x) in the world
        jaw_yaw[(int(d.k_ep[i]), int(d.k_kind[i]))] = wrap90(np.degrees(np.arctan2(ax[1], ax[0])))
    for e, s in enumerate(d.segs):
        r_c, a_c, in_c = on_annulus(cs[e, :2])
        r_p, a_p, in_p = on_annulus(pl[e, :2])
        dist = float(np.linalg.norm(pl[e, :2] - cs[e, :2]))
        turn = wrap90(np.degrees(pl[e, 2]) - jaw_yaw[(e, 0)])
        eps.append(dict(
            dataset=s["dataset"], episode=s["episode"],
            cube_start_xyz=[float(cs[e, 0]), float(cs[e, 1]), float(CUBE_HALF + J.table_z(x, cs[e, :2]))],
            cube_start_yaw_deg=jaw_yaw[(e, 0)], cube_start_yaw_silhouette_deg=wrap90(np.degrees(cs[e, 2])),
            plate_xy=[float(pl[e, 0]), float(pl[e, 1])], plate_yaw_deg=wrap90(np.degrees(pl[e, 2])),
            grasp_frame=s["grasp_frame"], release_frame=s["release_frame"],
            cube_end_xyz=[float(ce[e, 0]), float(ce[e, 1]), float(CUBE_HALF + J.plate_t(x) + J.table_z(x, ce[e, :2]))],
            cube_end_yaw_deg=jaw_yaw[(e, 1)], cube_end_yaw_silhouette_deg=wrap90(np.degrees(ce[e, 2])),
            p_grip=pg[e].tolist(),
            n_frames=s["n"], start_pose_deg=d.D[s["dataset"]]["state"][s["a"]].tolist(),
            sim_check=dict(cube_r=r_c, cube_angle_deg=a_c, cube_in_annulus=in_c, plate_r=r_p, plate_angle_deg=a_p,
                           plate_in_annulus=in_p, cube_plate_dist=dist, dist_in_6_30cm=bool(GOAL_DIST[0] <= dist <= GOAL_DIST[1]),
                           yaw_turn_deg=turn, turn_within_30=bool(abs(turn) <= GOAL_TURN_DEG),
                           task_in_sim_distribution=bool(in_c and in_p and GOAL_DIST[0] <= dist <= GOAL_DIST[1]
                                                         and abs(turn) <= GOAL_TURN_DEG))))
    m = J.g(x, "mat")
    corners = J.mat_corners(m)[:, :2].tolist()
    # tent / table edges: plumb-line points of the external view back-projected onto the table plane
    from calib_common import unproject_fisheye
    edges = {}
    t = J.g(x, "tbl")
    for name, P in d.lines.items():
        rays = unproject_fisheye(P, J.g(x, "eK"), J.g(x, "eD")) @ R
        nrm = np.array([-t[1], -t[2], 1.0])
        lam = (t[0] - nrm @ C) / (rays @ nrm)
        Xe = C + lam[:, None] * rays
        mu = Xe[:, :2].mean(0)
        _, _, vt = np.linalg.svd(Xe[:, :2] - mu)
        u = vt[0] if vt[0][np.argmax(np.abs(vt[0]))] > 0 else -vt[0]
        proj = (Xe[:, :2] - mu) @ u
        edges[name] = dict(mean_xy=mu.tolist(), direction_deg=float(np.degrees(np.arctan2(u[1], u[0]))),
                           end_a=(mu + u * proj.min()).tolist(), end_b=(mu + u * proj.max()).tolist(),
                           note="on the table plane" if not name.endswith("vertical") else
                           "vertical edge: the table-plane intersection is not meaningful")
    scene = dict(mat=dict(corners_xy=corners, center_xy=m[:2].tolist(), yaw_deg=float(np.degrees(m[2])),
                          size_xy=[float(m[3]), float(m[4])], z=MAT_T),
                 plate_top_z=float(J.plate_t(x)), plate_thickness=float(J.plate_t(x)), plate_size=0.05, cube_size=0.03,
                 table_plane=dict(coef=[float(v) for v in J.g(x, "tbl")],
                                  meaning="table top z = c0 + c1 * x + c2 * y in the arm-base frame (m)",
                                  tilt_deg=[float(np.degrees(v)) for v in J.g(x, "tbl")[1:]]),
                 colors_srgb=colors or {},
                 tent_edges_on_table=edges,
                 notes="")
    out = dict(version="calib_v1", joint_map=jm, cameras=cams,
               time_offset_frames=dict(external=float(dtv[0]), wrist=float(dtv[1])),
               episodes=eps, scene=scene)
    if extra:
        out.update(extra)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", required=True)
    ap.add_argument("--model", default="opencv_fisheye")
    ap.add_argument("--wmodel", default="")
    ap.add_argument("--holdout", default="")
    ap.add_argument("--jaw", default="")
    ap.add_argument("--colors", default="")
    ap.add_argument("--assoc", default="", help="npz with ws_keep / wp_keep / mat_assoc saved by the fit")
    ap.add_argument("--out", default=str(CALIB / "calib_v1.json"))
    a = ap.parse_args()
    data = Data()
    J = Joint(data, model=a.model, wmodel=a.wmodel or None)
    from calib_joint import load_x
    J.lines_used = data.lines
    x = load_x(J, a.x)
    if a.assoc:
        z = np.load(a.assoc)
        J.ws_keep, J.wp_keep = z["ws_keep"], z["wp_keep"]
        J.mat_assoc = (z["ma_rows"], z["ma_corner"], z["ma_obs"])
    else:
        J.gate_wrist(x, 8.0, 0.25, 8.0)
        J.associate_mat(x, 10.0)
    rep = J.report(x)
    hold = load_json(a.holdout) if a.holdout else None
    jaw = load_json(a.jaw) if a.jaw else None
    col = load_json(a.colors) if a.colors else None
    out = export(J, x, rep, hold, jaw, col)
    save_json(a.out, out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
