"""(1) Table / mat top height over the workspace from all real data. CPU only. Writes OUT/table.json + table_rows.csv.

Evidence (sim frame = arm base frame of the calibrated joint map, sim table top z = 0):
  A. Robot holds (every robot episode with a logged hold: measured jaw 5-14 deg with the command > 3 deg below it,
     > 1 s; the episodes.py rule of the grasp tally). At the hold start (median of the first 10 steps):
     grasp point z (FK), lowest gripper point z (measured pose), cube centre = grasp point + R XG (sim median cube
     centre in the grasp frame). These are bounds, not measurements: the action-layer floor keeps the COMMANDED
     lowest gripper point >= 1.5 mm + table_z, so the jaw can only reach the cube's upper part if the mat is lower.
  B. Image height of the resting cube (robot): the green blob of the external 112 px thumbs before the close (fully
     visible frames), the calibrated camera ray through its centroid, the point of the ray above the FK cube xy at the
     hold -> cube centroid height -> mat top = height - 15 mm.
  C. Teleop recordings (no floor rule; 15 episodes): lowest gripper point near the grasp frame of each calib episode
     (the operator's jaw near / on the mat) and at the release; the calib's cube rest / end positions.
Plane fits z = c0 + c1 x + c2 y (least squares + residual) for each evidence set; comparison with
calib scene.table_plane.

    PYTHONPATH=<wt>/src ~/miniconda3/envs/so101_vision/bin/python table_height.py
"""
from __future__ import annotations

import csv
import json

import numpy as np

import sysid_lib as L

HELD_JAW = (5.0, 14.0)
SQUEEZE = 3.0
HOLD_MIN_S = 1.0
OPEN_JAW = 25.0
XG = np.array([-0.0043, 0.0005, -0.0025])     # sim median cube centre in the grasp frame (bench/inventory.py)
FLOOR = 0.0015                                 # joint_env.FLOOR (lowest commanded gripper point, sim frame)
PLATE_TOP = 0.0039                             # calib scene.plate_top_z (plate thickness, m)


def runs(mask, t, min_s):
    out, k, n = [], 0, len(mask)
    while k < n:
        if mask[k]:
            k1 = k
            while k1 < n and mask[k1]:
                k1 += 1
            if t[k1 - 1] - t[k] >= min_s:
                out.append((k, k1))
            k = k1
        else:
            k += 1
    return out


def fill_gaps(mask, t, max_gap_s):
    m = mask.copy()
    on = np.flatnonzero(mask)
    for a, b in zip(on[:-1], on[1:]):
        if b - a > 1 and t[b] - t[a] <= max_gap_s:
            m[a:b] = True
    return m


_ROT = None


def session_rot(session):
    """Rotation of the external camera of a session against the calibration (teleop) frames, from appearance.json
    (appearance.py runs first; missing -> identity). New pixel u' = proj(R ray(u)) -> calibration ray = R^T ray(u')."""
    global _ROT
    if _ROT is None:
        f = L.OUT / "appearance.json"
        _ROT = json.loads(f.read_text())["ext_pose_per_session"] if f.exists() else {}
    w = _ROT.get(session, {}).get("rot_vs_teleop_deg")
    if w is None:
        return np.eye(3)
    import cv2
    R, _ = cv2.Rodrigues(np.radians(np.asarray(w, np.float64)).reshape(3, 1))
    return R


def ray_height_at_xy(uv_raw, cxy, Rs=None):
    """Camera ray through raw pixel uv -> z of the ray point whose xy is closest to cxy (m). Rs: the session's camera
    rotation (session_rot), applied to the camera-frame ray first."""
    c = L.ext_cam()
    rc = L.unproject_fisheye(np.atleast_2d(uv_raw), c["K"], c["D"])[0]
    if Rs is not None:
        rc = Rs.T @ rc
    d = rc @ c["R"].T
    o = c["t"]
    s = float((np.asarray(cxy) - o[:2]) @ d[:2] / (d[:2] @ d[:2]))
    return float(o[2] + s * d[2]), float(np.degrees(np.arctan2(-d[2], np.hypot(*d[:2]))))


def ray_to_plane(uv_raw, h):
    """Raw external pixel -> world point on the plane 'calibrated table + h'."""
    c = L.ext_cam()
    d = L.unproject_fisheye(np.atleast_2d(uv_raw), c["K"], c["D"])[0] @ c["R"].T
    o = c["t"]
    tc = L.table_coef()
    n = np.array([-tc[1], -tc[2], 1.0])
    s = (tc[0] + h - o @ n) / (d @ n)
    return o + s * d


def plate_xy(th, h=PLATE_TOP):
    """Plate centre (world xy) from the red blob of the first 10 external thumbs (median), or None."""
    pts = []
    for j in range(min(10, len(th))):
        _, r = L.colour_masks(th[j, 0])
        b = L.blob(r, min_px=4)
        if b is not None:
            pts.append(ray_to_plane(L.thumb_to_raw(np.array([b["cx"], b["cy"]])), h)[:2])
    return np.median(pts, 0) if pts else None


def plane_fit(xy, z, w=None):
    xy, z = np.asarray(xy, float), np.asarray(z, float)
    ok = np.isfinite(z) & np.isfinite(xy).all(1)
    xy, z = xy[ok], z[ok]
    if len(z) < 4:
        return None
    A = np.c_[np.ones(len(z)), xy]
    c, *_ = np.linalg.lstsq(A, z, rcond=None)
    res = z - A @ c
    # bootstrap spread of the coefficients
    g = np.random.default_rng(0)
    bs = []
    for _ in range(500):
        i = g.integers(0, len(z), len(z))
        cb, *_ = np.linalg.lstsq(A[i], z[i], rcond=None)
        bs.append(cb)
    bs = np.array(bs)
    tc = L.table_coef()
    calib_at = A @ tc
    return dict(n=int(len(z)), coef=c.tolist(), coef_sd=bs.std(0).tolist(), resid_rms_mm=float(res.std() * 1e3),
                resid_p5_p95_mm=np.percentile(res * 1e3, [5, 95]).tolist(),
                minus_calib_plane_mm=L.pct((z - calib_at) * 1e3),
                z_mm=L.pct(z * 1e3), x_range=[float(xy[:, 0].min()), float(xy[:, 0].max())],
                y_range=[float(xy[:, 1].min()), float(xy[:, 1].max())],
                slope_mm_per_m=[float(c[1] * 1e3), float(c[2] * 1e3)])


STEP_DEV = []          # per robot step: measured lowest gripper point minus the calibrated plane (mm)
TEL_DEV = []           # per teleop frame: same


def robot_holds():
    rows, allrows = [], []
    for e in L.list_episodes():
        d = L.load(e)
        q = d["q_sim"]
        u = d["u_sim"]
        if "u_layer" in d.files:
            ul = d["u_layer"]
        else:                    # the action-layer output = the written command without the integral correction
            ul = u.copy()
            if "i_corr" in d.files:
                ul[:, :5] = ul[:, :5] - d["i_corr"]
        t = d["t"] - d["t"][0]
        N = len(t)
        jaw = np.degrees(q[:, 5])
        u = d["u_sim"]
        jc = np.degrees(np.where(np.isfinite(u[:, 5]), u[:, 5], q[:, 5]))
        P, R = L.fk(q)
        low = L.lowest(q)
        okc = np.isfinite(ul).all(1)
        low_cmd = np.full(N, np.nan)
        if okc.any():
            low_cmd[okc] = L.lowest(ul[okc])
        floor_z = FLOOR + e["table_z"]
        pl = L.table_z_at(P[:, :2])
        STEP_DEV.append((low - pl) * 1e3)
        pxy = plate_xy(d["qry_thumbs"])
        allrows.append(dict(id=e["id"], student=e["student"], table_z=e["table_z"], low_min_mm=float(low.min() * 1e3),
                            plate_x=float(pxy[0]) if pxy is not None else np.nan,
                            plate_y=float(pxy[1]) if pxy is not None else np.nan,
                            plane_at_plate_mm=float(L.table_z_at(pxy) * 1e3) if pxy is not None else np.nan,
                            low_cmd_min_mm=float(np.nanmin(low_cmd) * 1e3) if okc.any() else None,
                            floor_mm=floor_z * 1e3, floor_rule_frac=float(d["ev_floor"].mean()),
                            xy_at_low_min=P[int(np.argmin(low)), :2].tolist(),
                            plane_at_low_min_mm=float(L.table_z_at(P[int(np.argmin(low)), :2]) * 1e3)))
        squeeze = jc < jaw - SQUEEZE
        held = fill_gaps((jaw >= HELD_JAW[0]) & (jaw <= HELD_JAW[1]) & squeeze, t, 0.3)
        hr = runs(held, t, HOLD_MIN_S)
        hr = [(a, b) for a, b in hr if jaw[max(0, a - 50):a + 1].max() > 15.0]   # a close, not a jaw already shut
        if not hr:
            continue
        a, b = hr[0]
        kk = np.arange(a, min(a + 10, N))
        c3 = np.median(P[kk] + np.einsum("kij,j->ki", R[kk], XG), 0)
        gp = np.median(P[kk], 0)
        lo = float(np.median(low[kk]))
        # the 1 s before the hold: floor rule share, commanded lowest point
        pre = np.arange(max(0, a - 50), a)
        o = np.flatnonzero(jaw[:a] > OPEN_JAW)
        k_cl = int(o[-1]) if len(o) else 0
        # B. image height of the resting cube (external thumbs before the close)
        steps = d["qry_step"].astype(int)
        th = d["qry_thumbs"]
        sel = np.flatnonzero(steps < k_cl)
        det = []
        for j in sel:
            g, _ = L.colour_masks(th[j, 0])
            bb = L.blob(g, min_px=3)
            det.append(bb)
        area = np.array([x["area"] if x else 0 for x in det], float)
        zimg, elev, nimg, zimg_rot = np.nan, np.nan, 0, np.nan
        if (area > 0).any():
            full = area >= 0.85 * np.percentile(area[area > 0], 95)
            js = [i for i in np.flatnonzero(full)][-10:]            # the last 10 fully visible frames
            if js:
                hs, hr = [], []
                Rs = session_rot(e["session"])
                for i in js:
                    uv = L.thumb_to_raw(np.array([det[i]["cx"], det[i]["cy"]]))
                    h, el = ray_height_at_xy(uv, c3[:2])
                    hs.append(h)
                    hr.append(ray_height_at_xy(uv, c3[:2], Rs)[0])
                zimg, elev, nimg = float(np.median(hs)), el, len(js)
                zimg_rot = float(np.median(hr))
        rows.append(dict(id=e["id"], session=e["session"], student=e["student"], start=e["start"],
                         table_z=e["table_z"], t_hold=float(t[a]), hold_s=float(t[b - 1] - t[a]),
                         cube_x=float(c3[0]), cube_y=float(c3[1]), cube_z_fk=float(c3[2]), gp_z=float(gp[2]),
                         low_z=lo, plane_z=float(L.table_z_at(c3[:2])), floor_z=floor_z,
                         floor_rule_pre=float(d["ev_floor"][pre].mean()) if len(pre) else 0.0,
                         low_cmd_pre_min=float(np.nanmin(low_cmd[pre])) if len(pre) and np.isfinite(low_cmd[pre]).any() else np.nan,
                         low_pre_min=float(low[pre].min()) if len(pre) else np.nan,
                         cube_z_img=zimg, cube_z_img_rotcorr=zimg_rot, img_elev_deg=elev, n_img=nimg))
    return rows, allrows


def teleop():
    import pyarrow.parquet as pq
    cal = L.calib()
    eps = {(e["dataset"], e["episode"]): e for e in cal["episodes"]}
    out = []
    for ds in ("green_cube_red_plate", "green_cube_red_plate_1"):
        root = L.RAW / ds
        t = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pydict()
        st = np.asarray(t["observation.state"], np.float64)
        ep = np.asarray(t["episode_index"], np.int64)
        fi = np.asarray(t["frame_index"], np.int64)
        for k in np.unique(ep):
            m = ep == k
            q = L.real_to_sim(st[m])
            P, R = L.fk(q)
            low = L.lowest(q)
            TEL_DEV.append((low - L.table_z_at(P[:, :2])) * 1e3)
            e = eps.get((ds, int(k)))
            if e is None:
                continue
            gf, rf = int(e["grasp_frame"]), int(e["release_frame"])
            n = int(m.sum())
            cs, ce = np.array(e["cube_start_xyz"]), np.array(e["cube_end_xyz"])
            w = np.arange(max(0, gf - 30), min(n, gf + 3))             # 1 s before the grasp frame
            wr = np.arange(max(0, rf - 15), min(n, rf + 15))
            kmin = int(w[np.argmin(low[w])])
            kr = int(wr[np.argmin(low[wr])])
            out.append(dict(dataset=ds, episode=int(k), n=n, grasp_frame=gf, release_frame=rf,
                            cube_start=cs.tolist(), cube_end=ce.tolist(),
                            plane_at_cube_mm=float(L.table_z_at(cs[:2]) * 1e3),
                            low_grasp_min_mm=float(low[kmin] * 1e3), xy_grasp=P[kmin, :2].tolist(),
                            plane_at_grasp_xy_mm=float(L.table_z_at(P[kmin, :2]) * 1e3),
                            gp_z_at_grasp_mm=float(P[gf, 2] * 1e3), low_at_grasp_mm=float(low[gf] * 1e3),
                            low_ep_min_mm=float(low.min() * 1e3), xy_ep_min=P[int(np.argmin(low)), :2].tolist(),
                            plane_at_ep_min_mm=float(L.table_z_at(P[int(np.argmin(low)), :2]) * 1e3),
                            low_release_min_mm=float(low[kr] * 1e3), xy_release=P[kr, :2].tolist(),
                            plate_top_calib_mm=float((ce[2] - L.CUBE_HALF) * 1e3)))
    return out


def main():
    L.OUT.mkdir(parents=True, exist_ok=True)
    rows, allrows = robot_holds()
    tel = teleop()
    xy = np.array([[r["cube_x"], r["cube_y"]] for r in rows])
    res = dict(n_robot_episodes=len(allrows), n_holds=len(rows))
    res["robot_gp_minus15"] = plane_fit(xy, [r["gp_z"] - L.CUBE_HALF for r in rows])
    res["robot_cubefk_minus15"] = plane_fit(xy, [r["cube_z_fk"] - L.CUBE_HALF for r in rows])
    res["robot_low_at_hold"] = plane_fit(xy, [r["low_z"] for r in rows])
    res["robot_img_minus15"] = plane_fit(xy, [r["cube_z_img"] - L.CUBE_HALF for r in rows])
    res["robot_img_rotcorr_minus15"] = plane_fit(xy, [r["cube_z_img_rotcorr"] - L.CUBE_HALF for r in rows])
    txy = np.array([r["xy_grasp"] for r in tel])
    res["teleop_low_grasp_min"] = plane_fit(txy, [r["low_grasp_min_mm"] / 1e3 for r in tel])
    res["teleop_cube_start_calib"] = plane_fit(np.array([r["cube_start"][:2] for r in tel]),
                                               [r["cube_start"][2] - L.CUBE_HALF for r in tel])
    res["calib_plane"] = dict(coef=L.table_coef().tolist(), tilt_deg=L.calib()["scene"]["table_plane"]["tilt_deg"])
    res["robot_all_episodes"] = dict(
        low_min_mm=L.pct([r["low_min_mm"] for r in allrows]),
        low_min_minus_plane_mm=L.pct([r["low_min_mm"] - r["plane_at_low_min_mm"] for r in allrows]),
        low_cmd_min_mm=L.pct([r["low_cmd_min_mm"] for r in allrows if r["low_cmd_min_mm"] is not None]),
        floor_rule_frac=L.pct([r["floor_rule_frac"] for r in allrows]))
    res["holds"] = dict(
        gp_minus_plane_mm=L.pct([(r["gp_z"] - r["plane_z"]) * 1e3 for r in rows]),
        cubefk_minus_plane_minus15_mm=L.pct([(r["cube_z_fk"] - L.CUBE_HALF - r["plane_z"]) * 1e3 for r in rows]),
        low_minus_plane_mm=L.pct([(r["low_z"] - r["plane_z"]) * 1e3 for r in rows]),
        low_minus_floor_mm=L.pct([(r["low_z"] - r["floor_z"]) * 1e3 for r in rows]),
        low_cmd_pre_min_minus_floor_mm=L.pct([(r["low_cmd_pre_min"] - r["floor_z"]) * 1e3 for r in rows]),
        floor_rule_pre=L.pct([r["floor_rule_pre"] for r in rows]),
        img_minus_plane_minus15_mm=L.pct([(r["cube_z_img"] - L.CUBE_HALF - r["plane_z"]) * 1e3 for r in rows]),
        img_rotcorr_minus_plane_minus15_mm=L.pct([(r["cube_z_img_rotcorr"] - L.CUBE_HALF - r["plane_z"]) * 1e3
                                                  for r in rows]),
        img_elev_deg=L.pct([r["img_elev_deg"] for r in rows]),
        cube_x=L.pct(xy[:, 0]), cube_y=L.pct(xy[:, 1]))
    res["teleop"] = dict(
        n=len(tel),
        low_grasp_min_minus_plane_mm=L.pct([r["low_grasp_min_mm"] - r["plane_at_grasp_xy_mm"] for r in tel]),
        low_ep_min_minus_plane_mm=L.pct([r["low_ep_min_mm"] - r["plane_at_ep_min_mm"] for r in tel]),
        low_release_min_mm=L.pct([r["low_release_min_mm"] for r in tel]),
        plate_top_calib_mm=L.pct([r["plate_top_calib_mm"] for r in tel]),
        plane_at_cube_mm=L.pct([r["plane_at_cube_mm"] for r in tel]))
    sd, td = np.concatenate(STEP_DEV), np.concatenate(TEL_DEV)
    edge = lambda a: dict(n=int(len(a)), p0_01=float(np.percentile(a, 0.01)), p0_1=float(np.percentile(a, 0.1)),
                          p1=float(np.percentile(a, 1)), min=float(a.min()), n_below_0=int((a < 0).sum()),
                          n_below_m2=int((a < -2).sum()), n_within_3=int((np.abs(a) < 3).sum()))
    res["floor_edge"] = dict(note="measured lowest gripper point minus the calibrated plane, every step / frame (mm): "
                             "a physical floor at the plane gives a hard lower edge near 0", robot=edge(sd),
                             teleop=edge(td))
    pz = np.array([r["plane_z"] for r in rows]) * 1e3
    res["plane_at_holds_mm"] = L.pct(pz)
    res["held_cube_offset_mm"] = dict(
        note="grasp point minus (calibrated plane + 15 mm) at the hold start = how far above its centre the jaw "
             "holds the cube (sim teacher: XG z = -2.5 mm in the grasp frame, i.e. ~+2.5 mm)",
        **L.pct([(r["gp_z"] - r["plane_z"] - L.CUBE_HALF) * 1e3 for r in rows]))
    pp = np.array([r["plane_at_plate_mm"] for r in allrows])
    res["plane_at_plates_mm"] = L.pct(pp)
    res["plate_top_at_plates_mm"] = L.pct(pp + PLATE_TOP * 1e3)
    res["plate_xy"] = dict(x=L.pct([r["plate_x"] for r in allrows]), y=L.pct([r["plate_y"] for r in allrows]))
    allxy = np.concatenate([xy, np.array([[r["plate_x"], r["plate_y"]] for r in allrows])])
    allxy = allxy[np.isfinite(allxy).all(1)]
    res["plane_at_cubes_and_plates_mm"] = L.pct(L.table_z_at(allxy) * 1e3)
    res["rows_teleop"] = tel
    (L.OUT / "table.json").write_text(json.dumps(L.r3(res, 5), indent=1))
    with open(L.OUT / "table_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows([L.r3(r, 5) for r in rows])
    with open(L.OUT / "table_allrows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys()))
        w.writeheader()
        w.writerows([L.r3(r, 5) for r in allrows])
    for k in ("robot_gp_minus15", "robot_cubefk_minus15", "robot_low_at_hold", "robot_img_minus15", "robot_img_rotcorr_minus15",
              "teleop_low_grasp_min", "teleop_cube_start_calib"):
        v = res[k]
        if v:
            print(f"{k:26s} n {v['n']:3d} coef {np.round(np.array(v['coef']) * [1e3, 1, 1], 4)} (c0 mm) "
                  f"sd {np.round(np.array(v['coef_sd']) * [1e3, 1, 1], 4)} rms {v['resid_rms_mm']:.1f} mm  "
                  f"minus calib p50 {v['minus_calib_plane_mm']['p50']:+.1f} [{v['minus_calib_plane_mm']['p5']:+.1f}, "
                  f"{v['minus_calib_plane_mm']['p95']:+.1f}]")
    print("calib", res["calib_plane"])
    for k, v in res["holds"].items():
        print(f"hold {k:34s} p50 {v.get('p50', np.nan):+8.2f} p5 {v.get('p5', np.nan):+8.2f} p95 {v.get('p95', np.nan):+8.2f} n {v.get('n')}")
    print("floor edge", json.dumps(L.r3(res["floor_edge"], 2)))
    print("plane at holds", L.r3(res["plane_at_holds_mm"], 2))
    print("plane at plates", L.r3(res["plane_at_plates_mm"], 2), "plate xy", L.r3(res["plate_xy"], 3))
    print("plane at cubes + plates", L.r3(res["plane_at_cubes_and_plates_mm"], 2))
    print("held cube offset", L.r3(res["held_cube_offset_mm"], 2))
    for k, v in res["robot_all_episodes"].items():
        print(f"all  {k:34s} p50 {v.get('p50', np.nan):+8.2f} p5 {v.get('p5', np.nan):+8.2f} min {v.get('min', np.nan):+8.2f} n {v.get('n')}")
    for k, v in res["teleop"].items():
        if isinstance(v, dict):
            print(f"tel  {k:34s} p50 {v.get('p50', np.nan):+8.2f} p5 {v.get('p5', np.nan):+8.2f} p95 {v.get('p95', np.nan):+8.2f} min {v.get('min', np.nan):+8.2f} n {v.get('n')}")


if __name__ == "__main__":
    main()
