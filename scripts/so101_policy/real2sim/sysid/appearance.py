"""(4) Image appearance per session and the external camera pose drift. CPU only. Writes OUT/appearance.json and
OUT/img/first_frames_ext.jpg / first_frames_wrist.jpg.

A. Student thumbs (112 px, both cameras, every query of every robot episode): mean gray, mean sRGB, contrast (gray
   std), share of clipped pixels (>= 250), "equivalent linear brightness" = mean of srgb_inv(y ** (1 / g_nom)) with the
   scene's nominal tone gamma g_nom (external 1.354, wrist 1.056): a ratio of two such means is the exposure factor of
   the renderer's tone map (camera_fx: x = hdr * exposure ..., y = srgb(x) ** gamma) that maps one onto the other.
   Per episode: the first query (arm at HOME, comparable across episodes) and the within-episode spread (auto exposure:
   the camera re-exposes as the view changes). Per session and day / night (session start 08:00-18:00 = day).
B. Sim nominal renders of 1,795 real hold states (10-22 afternoon, ~/so101_studentlab_claude/diag/renders_B_112.npz,
   training scene, nominal visuals, READ ONLY) against the real thumbs of the same 12 episodes while the cube is held.
C. First frames (640 x 480, preflight, one per session): gray, regions (scene.py ROIs of the grasp tally).
D. External camera pose per session: 3-angle rotation of the calibrated fisheye fitted to 48 px patch shifts (phase
   correlation of gradient images, static regions, arm box masked) of each session's first frame against
   (a) the reference session eval_20261022_151223 and (b) teleop recording frames (the calibration data; median of
   the fits against 5 teleop first frames). Code copied from ~/so101_r2s_claude/grasp_diag/tally/scene_rotfit.py.

    PYTHONPATH=<wt>/src ~/miniconda3/envs/so101_vision/bin/python appearance.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import sysid_lib as L

G_NOM = {"external": 1.35405, "wrist": 1.05565}
RENDERS_B = Path("/home/harry/so101_studentlab_claude/diag/renders_B_112.npz")
RENDER_RUNS = ["live_20261022_144731"] + [f"eval_20261022_151223/ep_0{i}" for i in range(1, 7)] + \
    ["live_20261022_152157", "live_20261022_152431"] + [f"eval_20261022_153448/ep_0{i}" for i in range(1, 4)]
REF_SESSION = "eval_20261022_151223"
TH_EXT_PATCH = {"dark_lower_left": (0, 70, 8, 112), "dark_top_left": (0, 0, 20, 20), "floor_lower_left": (20, 90, 40, 110),
                "floor_top_right": (100, 0, 112, 30), "mat": (60, 75, 100, 92)}
FF_EXT_ROI = {"right_walls": (360, 20, 630, 460), "lower_left_floor_edge": (20, 330, 200, 475),
              "top_left_dark": (0, 0, 120, 110)}
P = 48
ARM = (60, 110, 280, 300)
GROUPS = {"floor_edge_lower_left": (0, 290, 170, 480), "mat_edges": (160, 180, 340, 370),
          "walls_right": (340, 0, 640, 480), "back_wall_top": (100, 0, 340, 110), "dark_top_left": (0, 0, 100, 110)}


def srgb_inv(y):
    y = np.asarray(y, np.float64)
    return np.where(y <= 0.04045, y / 12.92, ((y + 0.055) / 1.055) ** 2.4)


def lin_mean(img, cam):
    y = img.astype(np.float64) / 255.0
    return float(srgb_inv(y ** (1.0 / G_NOM[cam])).mean())


def img_stats(img, cam):
    g = img.astype(np.float64).mean(-1)
    return dict(gray=float(g.mean()), rgb=img.reshape(-1, 3).mean(0).tolist(), std=float(g.std()),
                clip=float((img.max(-1) >= 250).mean()), lin=lin_mean(img, cam))


def day_or_night(name):
    hh = int(name.split("_")[2][:2])
    return "day" if 8 <= hh < 18 else "night"


# ---- external camera rotation (scene_rotfit.py of the grasp tally) ------------------------------------------------
def grad(img_rgb, mask=None):
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if mask is not None:
        g = g.copy()
        bl = cv2.blur(np.where(mask, 0, g), (15, 15)) / np.maximum(cv2.blur((~mask).astype(np.float32), (15, 15)), 1e-3)
        g[mask] = bl[mask]
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def colour_mask(img_rgb, dil=3):
    gm, rm = L.colour_masks(img_rgb)
    m = ((gm > 0.1) | (rm > 0.1)).astype(np.uint8)
    if dil:
        m = cv2.dilate(m, np.ones((2 * dil + 1, 2 * dil + 1), np.uint8))
    return m.astype(bool)


def shift(a, b, roi):
    x0, y0, x1, y1 = roi
    A, Bb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    win = cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(A.astype(np.float32), Bb.astype(np.float32), win)
    return float(dx), float(dy), float(resp)


def rotm(w):
    R, _ = cv2.Rodrigues(np.asarray(w, np.float64).reshape(3, 1))
    return R


def fit_rot(pts, K, D):
    u = pts[:, :2]
    du = pts[:, 2:4]
    rays = L.unproject_fisheye(u, K, D)
    w = np.zeros(3)
    for _ in range(10):
        f0 = L.project_cam(rays @ rotm(w).T, K, D) - u
        J = np.zeros((len(u) * 2, 3))
        for i in range(3):
            dw = np.zeros(3)
            dw[i] = 1e-5
            J[:, i] = ((L.project_cam(rays @ rotm(w + dw).T, K, D) - u) - f0).reshape(-1) / 1e-5
        res = (du - f0).reshape(-1)
        w = w + np.linalg.lstsq(J, res, rcond=None)[0]
    return w


def rot_vs(ref, img, K, D, extra_mask=None):
    mref = colour_mask(ref, 8) if extra_mask is None else (colour_mask(ref, 8) | extra_mask)
    gr = grad(ref, mref)
    g = grad(img, colour_mask(img, 8) | mref)
    pts = []
    for y in range(0, 480 - P + 1, P // 2):
        for x in range(0, 640 - P + 1, P // 2):
            cx, cy = x + P // 2, y + P // 2
            if ARM[0] <= cx < ARM[2] and ARM[1] <= cy < ARM[3]:
                continue
            if gr[y:y + P, x:x + P].mean() < 4.0:
                continue
            dx, dy, r = shift(gr, g, (x, y, x + P, y + P))
            if r > 0.3 and np.hypot(dx, dy) < 15:
                pts.append((cx, cy, dx, dy, r))
    pts = np.array(pts)
    keep = np.ones(len(pts), bool)
    w = np.zeros(3)
    for _ in range(3):
        w = fit_rot(pts[keep], K, D)
        pa = L.project_cam(L.unproject_fisheye(pts[:, :2], K, D) @ rotm(w).T, K, D) - pts[:, :2]
        resid = np.hypot(*(pts[:, 2:4] - pa).T)
        keep = resid < max(3 * np.median(resid[keep]), 0.5)
    pa = L.project_cam(L.unproject_fisheye(pts[:, :2], K, D) @ rotm(w).T, K, D) - pts[:, :2]
    resid = np.hypot(*(pts[:, 2:4] - pa).T)
    crop_c = np.array([[52 + 144, 103 + 144]], float)
    cc = L.project_cam(L.unproject_fisheye(crop_c, K, D) @ rotm(w).T, K, D) - crop_c
    return dict(rot_deg=np.degrees(w).tolist(), rot_norm_deg=float(np.degrees(np.linalg.norm(w))), n=int(len(pts)),
                n_inlier=int(keep.sum()), resid_px_p50=float(np.median(resid[keep])),
                crop_centre_shift_px=cc[0].tolist())


def teleop_first_frames(n=5):
    cap = cv2.VideoCapture(str(L.RAW / "green_cube_red_plate_1/videos/observation.images.external/chunk-000/file-000.mp4"))
    import pyarrow.parquet as pq
    e = pq.read_table(L.RAW / "green_cube_red_plate_1/meta/episodes/chunk-000/file-000.parquet").to_pydict()
    starts = list(np.asarray(e["dataset_from_index"], int))
    want = set(starts[:n * 2])
    out, i = [], 0
    while len(out) < n:
        ok, im = cap.read()
        if not ok:
            break
        if i in want:
            out.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return out


def main():
    eps = L.list_episodes()
    out = {}
    # ---- A. thumbs ---------------------------------------------------------------------------------------------------
    per_ep, all_q = [], {"external": [], "wrist": []}
    for e in eps:
        d = L.load(e)
        th = d["qry_thumbs"]
        st = {}
        for ci, cam in enumerate(("external", "wrist")):
            s0 = img_stats(th[0, ci], cam)
            gq = th[:, ci].astype(np.float64).mean((1, 2, 3))
            lq = np.array([lin_mean(th[j, ci], cam) for j in range(0, len(th), 5)])
            all_q[cam].append(gq)
            st[cam] = dict(first=s0, gray_p50=float(np.median(gq)), gray_p5=float(np.percentile(gq, 5)),
                           gray_p95=float(np.percentile(gq, 95)), lin_p50=float(np.median(lq)),
                           lin_spread_p5_p95_rel=[float(np.percentile(lq, 5) / np.median(lq)),
                                                  float(np.percentile(lq, 95) / np.median(lq))])
        st["ext_patches_first"] = {k: float(th[0, 0, y0:y1, x0:x1].mean()) for k, (x0, y0, x1, y1) in TH_EXT_PATCH.items()}
        per_ep.append(dict(id=e["id"], session=e["session"], start=e["start"], student=e["student"],
                           dn=day_or_night(e["session"]), **st))
    sess = {}
    for r in per_ep:
        sess.setdefault(r["session"], []).append(r)
    S = {}
    for s, rs in sess.items():
        S[s] = dict(n_ep=len(rs), dn=rs[0]["dn"], start=rs[0]["start"],
                    ext_first_gray=float(np.median([r["external"]["first"]["gray"] for r in rs])),
                    ext_first_lin=float(np.median([r["external"]["first"]["lin"] for r in rs])),
                    ext_first_rgb=np.median([r["external"]["first"]["rgb"] for r in rs], 0).tolist(),
                    wr_first_gray=float(np.median([r["wrist"]["first"]["gray"] for r in rs])),
                    wr_first_lin=float(np.median([r["wrist"]["first"]["lin"] for r in rs])),
                    wr_first_rgb=np.median([r["wrist"]["first"]["rgb"] for r in rs], 0).tolist(),
                    ext_dark=float(np.median([r["ext_patches_first"]["dark_lower_left"] for r in rs])),
                    ext_floor=float(np.median([r["ext_patches_first"]["floor_lower_left"] for r in rs])),
                    ext_mat=float(np.median([r["ext_patches_first"]["mat"] for r in rs])),
                    wr_gray_ep_p50=float(np.median([r["wrist"]["gray_p50"] for r in rs])),
                    ext_gray_ep_p50=float(np.median([r["external"]["gray_p50"] for r in rs])))
    out["sessions_thumbs"] = S
    for cam, key in (("external", "ext_first_lin"), ("wrist", "wr_first_lin")):
        v = np.array([S[s][key] for s in S])
        med = float(np.median(v))
        out.setdefault("session_exposure_factor", {})[cam] = dict(
            note="session median of the first-query linear brightness / the median over sessions",
            p_min_max=[float(v.min() / med), float(v.max() / med)], p5_p95=np.percentile(v / med, [5, 95]).tolist(),
            n_sessions=int(len(v)))
        allep = np.array([r[cam]["first"]["lin"] for r in per_ep])
        out.setdefault("episode_exposure_factor", {})[cam] = dict(
            note="per episode first-query linear brightness / the median over episodes",
            p5_p95=np.percentile(allep / np.median(allep), [5, 95]).tolist(),
            min_max=[float(allep.min() / np.median(allep)), float(allep.max() / np.median(allep))], n=int(len(allep)))
        within = np.array([r[cam]["lin_spread_p5_p95_rel"] for r in per_ep])
        out.setdefault("within_episode_exposure_factor", {})[cam] = dict(
            note="per episode p5 / p50 and p95 / p50 of the linear brightness over its queries (view change + auto "
                 "exposure)", p5_rel_p50=float(np.median(within[:, 0])), p95_rel_p50=float(np.median(within[:, 1])),
            p5_rel_worst=float(within[:, 0].min()), p95_rel_worst=float(within[:, 1].max()))
    dn = {}
    for k in ("day", "night"):
        rs = [r for r in per_ep if r["dn"] == k]
        dn[k] = dict(n_ep=len(rs), n_sessions=len({r["session"] for r in rs}),
                     ext_first_gray=L.pct([r["external"]["first"]["gray"] for r in rs], (5, 50, 95)),
                     wr_first_gray=L.pct([r["wrist"]["first"]["gray"] for r in rs], (5, 50, 95)),
                     ext_dark=L.pct([r["ext_patches_first"]["dark_lower_left"] for r in rs], (5, 50, 95)),
                     ext_floor=L.pct([r["ext_patches_first"]["floor_lower_left"] for r in rs], (5, 50, 95)),
                     ext_rgb=np.median([r["external"]["first"]["rgb"] for r in rs], 0).tolist(),
                     wr_rgb=np.median([r["wrist"]["first"]["rgb"] for r in rs], 0).tolist())
    out["day_night_first_query"] = dn
    out["all_queries_gray"] = {c: L.pct(np.concatenate(v), (1, 5, 50, 95, 99)) for c, v in all_q.items()}
    # ---- B. sim nominal vs real (same 12 episodes, held-cube queries) ---------------------------------------------------
    sim = np.load(RENDERS_B)["th"]
    real = []
    for r in RENDER_RUNS:
        d = np.load(L.LOGS / r / "episode_live.npz", allow_pickle=True)
        q = np.degrees(d["q_sim"][:, 5])
        steps = d["qry_step"].astype(int)
        held = (q[steps] >= 5) & (q[steps] <= 14)
        real.append(d["qry_thumbs"][held])
    real = np.concatenate(real)
    cmp = {}
    for ci, cam in enumerate(("external", "wrist")):
        ls = np.array([lin_mean(x, cam) for x in sim[:, ci]])
        lr = np.array([lin_mean(x, cam) for x in real[:, ci]])
        cmp[cam] = dict(n_sim=int(len(sim)), n_real=int(len(real)),
                        sim_gray=L.pct(sim[:, ci].astype(np.float64).mean((1, 2, 3)), (5, 50, 95)),
                        real_gray=L.pct(real[:, ci].astype(np.float64).mean((1, 2, 3)), (5, 50, 95)),
                        sim_rgb=sim[:, ci].reshape(-1, 3).mean(0).tolist(), real_rgb=real[:, ci].reshape(-1, 3).mean(0).tolist(),
                        sim_std=float(np.median(sim[:, ci].astype(np.float64).mean(-1).std((1, 2)))),
                        real_std=float(np.median(real[:, ci].astype(np.float64).mean(-1).std((1, 2)))),
                        real_over_sim_lin=float(np.median(lr) / np.median(ls)),
                        sim_clip=float((sim[:, ci].max(-1) >= 250).mean()), real_clip=float((real[:, ci].max(-1) >= 250).mean()))
    out["sim_nominal_vs_real"] = cmp
    # ---- C. first frames ---------------------------------------------------------------------------------------------
    ses = L.sessions()
    FF = {}
    for p in ses:
        ff = np.load(p / "first_frames.npz")
        e, w = ff["external"], ff["wrist"]
        FF[p.name] = dict(dn=day_or_night(p.name), ext=img_stats(e, "external"), wrist=img_stats(w, "wrist"),
                          **{f"ext_{k}": float(e[y0:y1, x0:x1].mean()) for k, (x0, y0, x1, y1) in FF_EXT_ROI.items()})
    out["first_frames"] = FF
    # montage of the external / wrist first frames
    for cam in ("external", "wrist"):
        tiles = []
        for p in ses:
            im = cv2.resize(np.load(p / "first_frames.npz")[cam], (213, 160), interpolation=cv2.INTER_AREA)
            cv2.putText(im, p.name[5:18] if p.name.startswith("eval") else p.name[:18], (3, 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (255, 0, 255), 1)
            tiles.append(im)
        while len(tiles) % 8:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.concatenate(tiles[i:i + 8], 1) for i in range(0, len(tiles), 8)]
        (L.OUT / "img").mkdir(exist_ok=True)
        cv2.imwrite(str(L.OUT / "img" / f"first_frames_{cam}.jpg"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    # ---- D. external camera pose ----------------------------------------------------------------------------------
    c = L.ext_cam()
    K, D = c["K"], c["D"]
    ref = np.load(L.LOGS / REF_SESSION / "first_frames.npz")["external"]
    tele = teleop_first_frames(5)
    ref_vs_tele = [rot_vs(t, ref, K, D) for t in tele]
    rt = np.median([r["rot_deg"] for r in ref_vs_tele], 0)
    out["ext_pose_ref_vs_teleop"] = dict(rot_deg=rt.tolist(), each=[r["rot_deg"] for r in ref_vs_tele],
                                         resid_px_p50=[r["resid_px_p50"] for r in ref_vs_tele])
    EP = {}
    for p in ses:
        e = np.load(p / "first_frames.npz")["external"]
        r = rot_vs(ref, e, K, D)
        r["rot_vs_teleop_deg"] = (np.asarray(r["rot_deg"]) + rt).tolist()
        r["rot_vs_teleop_norm_deg"] = float(np.linalg.norm(r["rot_vs_teleop_deg"]))
        r["dn"] = day_or_night(p.name)
        EP[p.name] = r
        print(f"{p.name:24s} rot vs ref {np.round(r['rot_deg'], 3)} |{r['rot_norm_deg']:.3f}| deg, vs teleop "
              f"{np.round(r['rot_vs_teleop_deg'], 3)} |{r['rot_vs_teleop_norm_deg']:.3f}|, resid {r['resid_px_p50']:.2f} px, "
              f"crop shift {np.round(r['crop_centre_shift_px'], 2)}", flush=True)
    out["ext_pose_per_session"] = EP
    nv = np.array([v["rot_vs_teleop_norm_deg"] for v in EP.values()])
    cs = np.array([np.hypot(*v["crop_centre_shift_px"]) for v in EP.values()])
    out["ext_pose_summary"] = dict(rot_vs_teleop_norm_deg=L.pct(nv, (5, 50, 95)),
                                   rot_vs_ref_norm_deg=L.pct([v["rot_norm_deg"] for v in EP.values()], (5, 50, 95)),
                                   crop_centre_shift_px_vs_ref=L.pct(cs, (5, 50, 95)),
                                   dr_cam_rot_external_rad=0.026, dr_cam_rot_external_deg=float(np.degrees(0.026)),
                                   dr_cam_pos_external_m=0.008)
    out["per_episode"] = per_ep
    (L.OUT / "appearance.json").write_text(json.dumps(L.r3(out, 4), indent=1))
    print(json.dumps(L.r3({k: v for k, v in out.items() if k in ("session_exposure_factor", "episode_exposure_factor",
                                                                  "within_episode_exposure_factor", "day_night_first_query",
                                                                  "sim_nominal_vs_real", "ext_pose_ref_vs_teleop",
                                                                  "ext_pose_summary", "all_queries_gray")}, 3), indent=1))


if __name__ == "__main__":
    main()
