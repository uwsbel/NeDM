"""(2b) Direct measurement of the camera capture pipeline lag from moving image features. CPU only.
Writes OUT/lag_fit.json.

Wrist camera, robot: the resting cube (before the close of every episode with a logged hold; cube xy = FK at the hold
start, z = calibrated plane + 15 mm) seen by the moving wrist camera. External camera, robot: the held cube (hold start
-> release / hold end) seen by the static external camera; the cube's offset in the grasp frame is unknown, so a
per-episode constant pixel bias is removed (both cameras). The detected green blob centroid (112 px thumbs) is compared
with the projection of the cube centre at the arm pose of time

    t_exposure = t_joint(query) + (grab - joint) - frame_age - tau      (tau = capture pipeline, the unknown)

(arm pose linearly interpolated over the logged joint reading times t + ms_read). tau is chosen to minimise the robust
(Huber) sum of squared residuals after the per-episode bias. The total image lag relative to the joint reading of the
query is then tau + frame_age - (grab - joint).

Teleop check (same wrist method on the 15 LeRobot recordings, 30 fps, state and frames on the same row): tau_teleop
should match the calibration's time_offset_frames (pipeline + the teleop frame age at read). This shows that the
method works when the arm moves fast (teleop); it says nothing about the slow robot motion (feature speed below).

Main robot result: the free-cube fit (cube position free in each episode). Checks of that fit, all in lag_fit.json:
  - per time block (10-22 / 10-23, day 08-18 h / night, by session start) and per student, bootstrap over episodes;
  - only the fast samples (predicted feature speed > its p75): does tau move?
  - frame age: the fit with the logged per-query age, with age 0 and with the episode median age (min cost);
  - identifiability: synthetic blob positions = the per-episode fitted cube projected at q(t - tau_true) on the logged
    robot time base + 1 px noise, for tau_true 10 / 25 / 40 ms (recovered?), and with a 1 deg wrist-camera turn or a
    1 deg shoulder_lift offset in the synthetic data only (systematic model errors the bootstrap does not see);
  - bias-removal fit with the FK cube moved by +-5 mm (x, y, z): sensitivity to the cube-position error.
A second, independent estimate of the pipeline uses LeRobot's timing (get_observation reads the motors first, then
calls the non-blocking read_latest, so the calib offset contains the teleop frame age at the read; mean 1/2 frame =
16.7 ms if the phases drift): pipeline = calib offset - 16.7 ms. Both estimates go to the total lag ("direct_fit",
"calib_minus_mean_teleop_age").

    PYTHONPATH=<wt>/src ~/miniconda3/envs/so101_vision/bin/python lag_fit.py
"""
from __future__ import annotations

import json

import numpy as np
import torch

import sysid_lib as L
import table_height as TH

TAUS = np.arange(-40.0, 120.1, 2.0)      # ms
WR_SX, WR_SY = 640 / 112, 480 / 112     # wrist thumb = the full 640 x 480 frame resized to 112 x 112


def quat_R(q):
    return L.quat_wxyz_to_R(q)


def wrist_cam(rot_deg=None):
    """Calibrated wrist camera; ``rot_deg`` (3,): an extra small turn of the camera in its own frame (synthetic
    systematic-error test only)."""
    c = L.calib()["cameras"]["wrist"]
    T = c["T_parent_cam"]
    R = quat_R(T[3:])
    if rot_deg is not None:
        import cv2
        dR, _ = cv2.Rodrigues(np.radians(np.asarray(rot_deg, np.float64)).reshape(3, 1))
        R = R @ dR
    return dict(K=np.array(c["K"]), D=np.array(c["D"]), t=np.array(T[:3]), R=R)


_WCAM = None


def _wcam():
    global _WCAM
    if _WCAM is None:
        _WCAM = wrist_cam()
    return _WCAM


def mount_poses(q):
    kin = L.layer().kin
    p, R = kin.body_poses(torch.as_tensor(np.asarray(q, np.float64)).reshape(-1, 6))
    i = list(__import__("nedm.so101_policy.kin", fromlist=["BODIES"]).BODIES).index("camera_mount")
    return p[:, i].numpy(), R[:, i].numpy()


def project_wrist(Xw, pm, Rm, cam=None):
    """World points [N, 3] with mount poses [N] -> wrist raw px [N, 2] and camera depth [N]."""
    c = cam if cam is not None else _wcam()
    Xm = np.einsum("nji,nj->ni", Rm, Xw - pm)             # R_m^T (Xw - p_m)
    Xc = (Xm - c["t"]) @ c["R"]                             # R_c^T (Xm - t_c)
    return L.project_cam(Xc, c["K"], c["D"]), Xc[:, 2]


def wrist_thumb_px(raw):
    return np.stack([(raw[:, 0] + 0.5) / WR_SX - 0.5, (raw[:, 1] + 0.5) / WR_SY - 0.5], -1)


def detect(img, min_px=6):
    g, _ = L.colour_masks(img)
    b = L.blob(g, min_px=min_px)
    if b is None:
        return None
    m = (g > 0.25)
    ys, xs = np.nonzero(m)
    if xs.min() <= 0 or ys.min() <= 0 or xs.max() >= img.shape[1] - 1 or ys.max() >= img.shape[0] - 1:
        return None                                          # touches the border: centroid biased
    return b


def interp_q(tq, ts, q):
    """Linear interpolation of the joint vector at times tq over sample times ts."""
    out = np.empty((len(tq), q.shape[1]))
    for i in range(q.shape[1]):
        out[:, i] = np.interp(tq, ts, q[:, i])
    return out


def ep_costs(res, delta=1.5):
    """Per-episode (Huber sum, n) after removing the episode's median residual (episodes with < 3 samples: 0, 0)."""
    hs, ns = [], []
    for r in res:
        if len(r) < 3:
            hs.append(0.0)
            ns.append(0)
            continue
        rr = r - np.median(r, 0)
        a = np.hypot(rr[:, 0], rr[:, 1])
        hs.append(float(np.where(a < delta, 0.5 * a * a, delta * (a - 0.5 * delta)).sum()))
        ns.append(len(a))
    return np.array(hs), np.array(ns)


def collect_robot():
    """-> list of per-episode dicts with the samples for both cameras."""
    eps = L.list_episodes()
    out = []
    for e in eps:
        d = L.load(e)
        q = d["q_sim"]
        t = d["t"] - d["t"][0]
        N = len(t)
        jaw = np.degrees(q[:, 5])
        u = d["u_sim"]
        jc = np.degrees(np.where(np.isfinite(u[:, 5]), u[:, 5], q[:, 5]))
        held = TH.fill_gaps((jaw >= TH.HELD_JAW[0]) & (jaw <= TH.HELD_JAW[1]) & (jc < jaw - TH.SQUEEZE), t, 0.3)
        hr = TH.runs(held, t, TH.HOLD_MIN_S)
        hr = [(a, b) for a, b in hr if jaw[max(0, a - 50):a + 1].max() > 15.0]
        if not hr:
            continue
        a, b = hr[0]
        P, R = L.fk(q)
        kk = np.arange(a, min(a + 10, N))
        c3 = np.median(P[kk] + np.einsum("kij,j->ki", R[kk], TH.XG), 0)
        cube = np.array([c3[0], c3[1], float(L.table_z_at(c3[:2])) + L.CUBE_HALF])
        o = np.flatnonzero(jaw[:a] > TH.OPEN_JAW)
        k_cl = int(o[-1]) if len(o) else 0
        mr = d["robot_ms_read"][:N] * 1e-3
        ts = t + mr                                         # joint reading times
        steps = d["qry_step"].astype(int)
        age = d["qry_frame_age_ms"] * 1e-3
        mc, mq, mf, mp, mm, mg = (d[k] for k in ("ms_ctrl", "ms_qdot", "ms_frames", "ms_pre", "ms_model", "ms_guard"))
        G = (mc[steps] - mf[steps] - mp[steps] - mm[steps] - mg[steps]) * 1e-3     # grab - joint (qdot + checks)
        t_ts = ts[steps] + G                                # grab time (episode clock)
        th = d["qry_thumbs"]
        # wrist: resting cube before the close, after 0.5 s
        W = []
        for j in np.flatnonzero((steps < k_cl - 5) & (t[steps] > 0.5)):
            bb = detect(th[j, 1])
            if bb is None or bb["area"] < 6:
                continue
            W.append((j, bb["cx"], bb["cy"], bb["area"]))
        # external: held cube from the hold start + 0.3 s to the hold end
        E = []
        for j in np.flatnonzero((steps > a + 15) & (steps < b - 2)):
            bb = detect(th[j, 0], min_px=4)
            if bb is None:
                continue
            E.append((j, bb["cx"], bb["cy"], bb["area"]))
        if not W and not E:
            continue
        # held cube point in the grasp frame for the external projection (sim median offset; bias removed later)
        out.append(dict(id=e["id"], ts=ts, q=q, t_grab=t_ts, age=age, cube=cube, W=np.array(W, float).reshape(-1, 4),
                        E=np.array(E, float).reshape(-1, 4), steps=steps, session=e["session"], student=e["student"],
                        block=block_of(e["session"])))
    return out


def block_of(session):
    """Session name (live_ / eval_YYYYMMDD_HHMMSS) -> time block 'MM-DD day' (08-18 h) or 'MM-DD night'."""
    _, ymd, hms = session.split("_")[:3]
    hh = int(hms[:2])
    return f"{ymd[4:6]}-{ymd[6:8]} " + ("day" if 8 <= hh < 18 else "night")


STUDENT_GROUPS = (("s28", ("s28",)), ("C1", ("C1",)), ("other", None))     # None = every other student


def student_group(st):
    for g, members in STUDENT_GROUPS:
        if members is None or st in members:
            return g
    return "other"


def residuals_robot(rec, tau_ms, cam):
    res = []
    for r in rec:
        S = r["W"] if cam == "wrist" else r["E"]
        if len(S) < 3:
            res.append(np.zeros((0, 2)))
            continue
        j = S[:, 0].astype(int)
        ci = 1 if cam == "wrist" else 0
        te = r["t_grab"][j] - r["age"][j, ci] - tau_ms * 1e-3
        qe = interp_q(te, r["ts"], r["q"])
        if cam == "wrist":
            pm, Rm = mount_poses(qe)
            raw, z = project_wrist(np.repeat(r["cube"][None], len(j), 0), pm, Rm)
            pred = wrist_thumb_px(raw)
        else:
            P, R = L.fk(qe)
            X = P + np.einsum("kij,j->ki", R, TH.XG)
            pred = L.raw_to_thumb(L.project_ext(X))
        res.append(S[:, 1:3] - pred)
    return res


def motion_px(rec, cam):
    """Image speed of the predicted feature (thumb px per 10 ms) at tau = 40 ms: identifiability."""
    sp = []
    for r in rec:
        S = r["W"] if cam == "wrist" else r["E"]
        if len(S) < 3:
            continue
        j = S[:, 0].astype(int)
        ci = 1 if cam == "wrist" else 0
        te = r["t_grab"][j] - r["age"][j, ci] - 0.04
        p = []
        for dt in (0.0, 0.01):
            qe = interp_q(te + dt, r["ts"], r["q"])
            if cam == "wrist":
                pm, Rm = mount_poses(qe)
                raw, _ = project_wrist(np.repeat(r["cube"][None], len(j), 0), pm, Rm)
                p.append(wrist_thumb_px(raw))
            else:
                P, R = L.fk(qe)
                p.append(L.raw_to_thumb(L.project_ext(P + np.einsum("kij,j->ki", R, TH.XG))))
        sp.append(np.hypot(*(p[1] - p[0]).T))
    return np.concatenate(sp) if sp else np.zeros(0)


def cost_matrix(resfun, rec, cam):
    C, n = [], None
    for tau in TAUS:
        h, n = ep_costs(resfun(rec, tau, cam))
        C.append(h)
    return np.array(C), n


def argmin_tau(costs):
    k = int(np.argmin(costs))
    if 0 < k < len(TAUS) - 1:                              # parabola refinement around the minimum
        y0, y1, y2 = costs[k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        return float(TAUS[k] + (0.5 * (y0 - y2) / den * (TAUS[1] - TAUS[0]) if den > 0 else 0.0))
    return float(TAUS[k])


def fit_tau(resfun, rec, cam, nb=500, seed=0):
    """-> (tau, cost curve, bootstrap taus over episodes)."""
    C, n = cost_matrix(resfun, rec, cam)
    costs = C.sum(1) / max(n.sum(), 1)
    g = np.random.default_rng(seed)
    use = np.flatnonzero(n > 0)
    bs = []
    for _ in range(nb):
        idx = g.choice(use, len(use), replace=True)
        bs.append(argmin_tau(C[:, idx].sum(1) / n[idx].sum()))
    return argmin_tau(costs), costs, np.array(bs)


# ---- teleop -------------------------------------------------------------------------------------------------------
def collect_teleop():
    import cv2
    import pyarrow.parquet as pq
    cal = L.calib()
    eps = {(e["dataset"], e["episode"]): e for e in cal["episodes"]}
    rec = []
    for ds in ("green_cube_red_plate", "green_cube_red_plate_1"):
        root = L.RAW / ds
        t = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pydict()
        st = np.asarray(t["observation.state"], np.float64)
        ep = np.asarray(t["episode_index"], np.int64)
        gidx = np.asarray(t["index"], np.int64)
        cap = cv2.VideoCapture(str(root / "videos/observation.images.wrist/chunk-000/file-000.mp4"))
        frames = {}
        need = {}
        for k in np.unique(ep):
            e = eps.get((ds, int(k)))
            if e is None:
                continue
            m = np.flatnonzero(ep == k)
            gf = int(e["grasp_frame"])
            sel = m[max(0, gf - 150):max(0, gf - 8)]          # the 5 s before the grasp (approach)
            need[int(k)] = (m, sel, e)
        want = set(int(gidx[i]) for _, (m, sel, e) in need.items() for i in sel)
        i = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if i in want:
                frames[i] = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (112, 112), interpolation=cv2.INTER_AREA)
            i += 1
        cap.release()
        for k, (m, sel, e) in need.items():
            q = L.real_to_sim(st[m])
            ts = np.arange(len(m)) / 30.0
            cs = np.array(e["cube_start_xyz"])
            W = []
            for ii in sel:
                f = frames.get(int(gidx[ii]))
                if f is None:
                    continue
                bb = detect(f)
                if bb is None:
                    continue
                W.append((int(ii - m[0]), bb["cx"], bb["cy"], bb["area"]))
            if len(W) >= 3:
                rec.append(dict(id=f"{ds}/{k}", ts=ts, q=q, cube=cs, W=np.array(W, float)))
    return rec


def residuals_teleop(rec, tau_ms, cam="wrist"):
    res = []
    for r in rec:
        j = r["W"][:, 0].astype(int)
        te = r["ts"][j] - tau_ms * 1e-3
        qe = interp_q(te, r["ts"], r["q"])
        pm, Rm = mount_poses(qe)
        raw, _ = project_wrist(np.repeat(r["cube"][None], len(j), 0), pm, Rm)
        res.append(r["W"][:, 1:3] - wrist_thumb_px(raw))
    return res


# ---- free cube position per episode (robust to the cube position error) ------------------------------------------
def free_cube_cost(S_px, cube0, pm, Rm, iters=6, delta=1.5):
    """Wrist samples [n, 2] thumb px, mount poses -> (Huber sum, n, cube) with the cube position fitted (3 params,
    Gauss-Newton on the Huber-weighted reprojection error)."""
    c = np.asarray(cube0, np.float64).copy()
    n = len(S_px)

    def pred(cc):
        raw, _ = project_wrist(np.repeat(cc[None], n, 0), pm, Rm)
        return wrist_thumb_px(raw)

    for _ in range(iters):
        f0 = pred(c)
        r = (S_px - f0).reshape(-1)
        a = np.abs(r)
        w = np.where(a < delta, 1.0, delta / np.maximum(a, 1e-9))
        J = np.zeros((2 * n, 3))
        for i in range(3):
            dc = np.zeros(3)
            dc[i] = 1e-4
            J[:, i] = (pred(c + dc) - f0).reshape(-1) / 1e-4
        JW = J * w[:, None]
        step = np.linalg.lstsq(JW.T @ J + 1e-6 * np.eye(3), JW.T @ r, rcond=None)[0]
        c = c + np.clip(step, -0.02, 0.02)
    rr = S_px - pred(c)
    a = np.hypot(rr[:, 0], rr[:, 1])
    return float(np.where(a < delta, 0.5 * a * a, delta * (a - 0.5 * delta)).sum()), n, c


def exposure_times(r, kind, tau, age_mode="query"):
    """Image exposure time of the wrist samples of record r (s, episode clock): robot t_grab - age - tau
    (age_mode 'query' = the logged per-query frame age, 'zero' = no age, 'median' = the episode's median age);
    teleop: the state row's time - tau."""
    j = r["W"][:, 0].astype(int)
    if kind != "robot":
        return r["ts"][j] - tau * 1e-3
    a = r["age"][j, 1]
    if age_mode == "zero":
        a = np.zeros_like(a)
    elif age_mode == "median":
        a = np.full_like(a, np.median(r["age"][:, 1]))
    return r["t_grab"][j] - a - tau * 1e-3


def cost_matrix_free(rec, kind, taus=None, age_mode="query"):
    """kind 'robot' or 'teleop' -> (C [tau, ep], n [ep])."""
    taus = TAUS if taus is None else taus
    C, ns = [], None
    for tau in taus:
        row, ns = [], []
        for r in rec:
            S = r["W"]
            if len(S) < 3:
                row.append(0.0)
                ns.append(0)
                continue
            qe = interp_q(exposure_times(r, kind, tau, age_mode), r["ts"], r["q"])
            pm, Rm = mount_poses(qe)
            h, n, _ = free_cube_cost(S[:, 1:3], r["cube"], pm, Rm)
            row.append(h)
            ns.append(n)
        C.append(row)
    return np.array(C), np.array(ns)


def argmin_tau_grid(costs, taus):
    k = int(np.argmin(costs))
    if 0 < k < len(taus) - 1:
        y0, y1, y2 = costs[k - 1:k + 2]
        den = y0 - 2 * y1 + y2
        return float(taus[k] + (0.5 * (y0 - y2) / den * (taus[1] - taus[0]) if den > 0 else 0.0))
    return float(taus[k])


def fit_from_C(C, n, cols=None, taus=None, nb=500, seed=0):
    """Fit tau from the per-episode cost matrix on the episode columns ``cols`` (None = all) with a bootstrap over
    those episodes -> dict (tau, bootstrap p5 / p50 / p95, normalised cost curve, n samples / episodes, cost rise from
    the minimum at a few taus)."""
    taus = TAUS if taus is None else taus
    cols = np.arange(C.shape[1]) if cols is None else np.asarray(cols, int)
    use = cols[n[cols] > 0]
    if len(use) == 0:
        return dict(n_episodes=0)
    costs = C[:, use].sum(1) / n[use].sum()
    g = np.random.default_rng(seed)
    bs = np.array([argmin_tau_grid(C[:, (idx := g.choice(use, len(use), replace=True))].sum(1) / n[idx].sum(), taus)
                   for _ in range(nb)])
    tau = argmin_tau_grid(costs, taus)
    cmin = float(costs.min())
    at = {f"{int(t)}": float(np.interp(t, taus, costs)) for t in (0.0, 10.0, 20.0, 30.0, 40.0, 60.0) if taus[0] <= t <= taus[-1]}
    return dict(tau_ms=tau, tau_boot_p5_p50_p95=np.percentile(bs, [5, 50, 95]).tolist(), n_episodes=int(len(use)),
                n_samples=int(n[use].sum()), cost_min=cmin, cost_at_ms=at,
                cost_rise_at_ms={k: v - cmin for k, v in at.items()}, cost=costs.tolist())


def fit_free(rec, kind, nb=500, seed=0, taus=None, age_mode="query"):
    taus = TAUS if taus is None else taus
    C, n = cost_matrix_free(rec, kind, taus, age_mode)
    f = fit_from_C(C, n, taus=taus, nb=nb, seed=seed)
    return f, C, n


def fitted_cubes(rec, kind, tau, age_mode="query"):
    """Per-episode free cube position at pipeline ``tau`` (None for episodes with < 3 samples)."""
    out = []
    for r in rec:
        if len(r["W"]) < 3:
            out.append(None)
            continue
        qe = interp_q(exposure_times(r, kind, tau, age_mode), r["ts"], r["q"])
        pm, Rm = mount_poses(qe)
        out.append(free_cube_cost(r["W"][:, 1:3], r["cube"], pm, Rm)[2])
    return out


def feature_speed_free(rec, cubes, tau):
    """Per-episode predicted image speed (thumb px per 10 ms) of the fitted cube at every wrist sample."""
    out = []
    for r, c in zip(rec, cubes):
        if c is None:
            out.append(np.zeros(len(r["W"])))
            continue
        te = exposure_times(r, "robot", tau)
        p = []
        for dt in (0.0, 0.01):
            pm, Rm = mount_poses(interp_q(te + dt, r["ts"], r["q"]))
            p.append(wrist_thumb_px(project_wrist(np.repeat(c[None], len(te), 0), pm, Rm)[0]))
        out.append(np.hypot(*(p[1] - p[0]).T))
    return out


def synth(rec, cubes, tau_true, noise_px=1.0, seed=0, cam_rot_deg=None, q_off_deg=None, noise_pool=None):
    """Synthetic copy of the robot wrist records: blob = the fitted cube projected at q(t_exposure(tau_true)) on the
    logged time base (+ N(0, noise_px) per axis, or residual vectors resampled from ``noise_pool`` [M, 2]: the real
    heavy-tailed residuals); ``cam_rot_deg`` / ``q_off_deg`` (6,) put a systematic model error into the synthetic data
    only (the fit keeps the nominal model)."""
    g = np.random.default_rng(seed)
    cam = wrist_cam(cam_rot_deg) if cam_rot_deg is not None else None
    out = []
    for r, c in zip(rec, cubes):
        if c is None:
            out.append(dict(r, W=r["W"][:0]))
            continue
        qe = interp_q(exposure_times(r, "robot", tau_true), r["ts"], r["q"])
        if q_off_deg is not None:
            qe = qe + np.radians(np.asarray(q_off_deg, np.float64))
        pm, Rm = mount_poses(qe)
        px = wrist_thumb_px(project_wrist(np.repeat(c[None], len(qe), 0), pm, Rm, cam)[0])
        W = r["W"].copy()
        W[:, 1:3] = px + (g.normal(0.0, noise_px, px.shape) if noise_pool is None
                          else noise_pool[g.integers(0, len(noise_pool), len(px))])
        out.append(dict(r, W=W))
    return out


def subset_samples(rec, masks):
    """Records with only the wrist samples of the boolean masks (one per record)."""
    return [dict(r, W=r["W"][m]) for r, m in zip(rec, masks)]


def shifted_cube(rec, d):
    return [dict(r, cube=r["cube"] + np.asarray(d, np.float64)) for r in rec]


def _brief(f):
    return {k: v for k, v in f.items() if k != "cost"}


def fitted_cubes_resid(rec, tau):
    """(per-episode fitted cubes, residual RMS per axis over all samples, thumb px) at pipeline tau."""
    cubes = fitted_cubes(rec, "robot", tau)
    rr = []
    for r, c in zip(rec, cubes):
        if c is None:
            continue
        pm, Rm = mount_poses(interp_q(exposure_times(r, "robot", tau), r["ts"], r["q"]))
        rr.append(r["W"][:, 1:3] - wrist_thumb_px(project_wrist(np.repeat(c[None], len(pm), 0), pm, Rm)[0]))
    rr = np.concatenate(rr)
    return cubes, float(np.sqrt(np.mean(rr ** 2))), float(np.median(np.abs(rr)) * 1.4826), rr


def main():
    rec = collect_robot()
    tel = collect_teleop()
    out = dict(taus_ms=TAUS.tolist(), method_note=__doc__)
    # A. per-episode pixel-bias removal (cube position from FK + calibrated plane)
    for cam in ("wrist", "external"):
        n_s = int(sum(len(r["W"] if cam == "wrist" else r["E"]) for r in rec))
        n_e = int(sum((len(r["W"] if cam == "wrist" else r["E"]) >= 3) for r in rec))
        tau, costs, bs = fit_tau(residuals_robot, rec, cam)
        sp = motion_px(rec, cam)
        out["bias_removal_robot_" + cam] = dict(n_samples=n_s, n_episodes=n_e, tau_pipeline_ms=tau,
                                                tau_boot_p5_p50_p95=np.percentile(bs, [5, 50, 95]).tolist(),
                                                cost=costs.tolist(), feature_speed_px_per_10ms=L.pct(sp, (25, 50, 75)))
        print(f"[bias removal] robot {cam}: n {n_s} / {n_e} episodes, tau {tau:.1f} ms (boot "
              f"{np.percentile(bs, 5):.1f}-{np.percentile(bs, 95):.1f}), feature speed p50 {np.median(sp):.3f} px/10 ms",
              flush=True)
    # A2. check: the bias-removal fit with the FK cube moved by +-5 mm (cube-position error -> tau)
    tau0 = out["bias_removal_robot_wrist"]["tau_pipeline_ms"]
    sh = {}
    for ax, nm in enumerate("xyz"):
        for sg in (-1, 1):
            d = np.zeros(3)
            d[ax] = sg * 0.005
            sh[f"{nm}{'+' if sg > 0 else '-'}5mm"] = fit_tau(residuals_robot, shifted_cube(rec, d), "wrist", nb=0)[0]
    dev = np.array(list(sh.values())) - tau0
    out["bias_removal_cube_shift_check"] = dict(
        note="bias-removal wrist fit with the FK cube position moved by 5 mm along x / y / z (sim frame): tau (ms); "
             "the free-cube fit (main result) does not depend on the cube position", tau_nominal_ms=tau0,
        tau_shifted_ms=sh, change_ms_min_max=[float(dev.min()), float(dev.max())])
    print(f"[cube shift] bias-removal tau {tau0:.1f} ms -> {np.round(list(sh.values()), 1)} (change "
          f"{dev.min():+.1f} .. {dev.max():+.1f} ms)", flush=True)
    tau, costs, bs = fit_tau(residuals_teleop, tel, "wrist")
    out["bias_removal_teleop_wrist"] = dict(n_samples=int(sum(len(r["W"]) for r in tel)), n_episodes=len(tel), tau_ms=tau,
                                            tau_boot_p5_p50_p95=np.percentile(bs, [5, 50, 95]).tolist(), cost=costs.tolist())
    print(f"[bias removal] teleop wrist: tau {tau:.1f} ms (boot {np.percentile(bs, 5):.1f}-{np.percentile(bs, 95):.1f})")
    # B. free cube position per episode (main result)
    fr, C, n = fit_free(rec, "robot")
    tau_r = fr["tau_ms"]
    bs_lo, bs_hi = fr["tau_boot_p5_p50_p95"][0], fr["tau_boot_p5_p50_p95"][2]
    cubes, rms_px, mad_px, resid = fitted_cubes_resid(rec, tau_r)
    sp = feature_speed_free(rec, cubes, tau_r)
    allsp = np.concatenate(sp)
    out["free_cube_robot_wrist"] = dict(fr, tau_pipeline_ms=tau_r, resid_rms_px=rms_px, resid_robust_sd_px=mad_px,
                                        feature_speed_px_per_10ms=L.pct(allsp, (25, 50, 75, 95)))
    print(f"[free cube] robot wrist tau {tau_r:.1f} ms (boot {bs_lo:.1f}-{bs_hi:.1f}); cost {fr['cost_at_ms']}; "
          f"residual rms {rms_px:.2f} px (robust sd {mad_px:.2f}); speed p50 {np.median(allsp):.3f} px/10 ms", flush=True)
    # B1. per time block and per student (columns of the same cost matrix; bootstrap within the subset)
    order = sorted({r["block"] for r in rec}, key=lambda b: (b[:5], b[6:] == "night"))
    byb = {b: _brief(fit_from_C(C, n, [i for i, r in enumerate(rec) if r["block"] == b])) for b in order}
    out["free_cube_by_block"] = byb
    out["free_cube_by_block_note"] = ("free-cube robot wrist fit per time block (session start; day = 08-18 h): tau, "
                                      "bootstrap p5 / p50 / p95 over the block's episodes (ms)")
    tb = np.array([v["tau_ms"] for v in byb.values()])
    lo_hi = np.array([[v["tau_boot_p5_p50_p95"][0], v["tau_boot_p5_p50_p95"][2]] for v in byb.values()])
    out["free_cube_block_spread"] = dict(tau_min_max_ms=[float(tb.min()), float(tb.max())],
                                         bootstrap_ranges_overlap=bool(lo_hi[:, 0].max() <= lo_hi[:, 1].min()))
    bys = {g: _brief(fit_from_C(C, n, [i for i, r in enumerate(rec) if student_group(r["student"]) == g]))
           for g, _ in STUDENT_GROUPS}
    out["free_cube_by_student"] = bys
    for k, v in list(byb.items()) + list(bys.items()):
        print(f"  [{k}] tau {v['tau_ms']:.1f} ms boot {np.round(v['tau_boot_p5_p50_p95'], 1)} n {v['n_episodes']} eps",
              flush=True)
    # B2. only the fast samples (predicted image speed > p75)
    thr = float(np.percentile(allsp, 75))
    ff, _, _ = fit_free(subset_samples(rec, [x > thr for x in sp]), "robot")
    out["free_cube_fast_samples"] = dict(_brief(ff), speed_threshold_px_per_10ms=thr,
                                         tau_change_ms=ff["tau_ms"] - tau_r)
    print(f"[fast samples > {thr:.3f} px/10 ms] tau {ff['tau_ms']:.1f} ms (boot "
          f"{np.round(ff['tau_boot_p5_p50_p95'], 1)}), n {ff['n_samples']} / {ff['n_episodes']} eps", flush=True)
    # B3. frame age check: per-query age vs no age vs the episode median age
    fa = {"query": _brief(fr)}
    for mode in ("zero", "median"):
        fa[mode] = _brief(fit_free(rec, "robot", nb=200, age_mode=mode)[0])
    out["frame_age_check"] = dict(
        note="free-cube robot wrist fit with the logged per-query frame age ('query', main), without it ('zero') and "
             "with the episode median age ('median'): minimum cost (lower = better) and tau",
        cost_min={k: v["cost_min"] for k, v in fa.items()}, tau_ms={k: v["tau_ms"] for k, v in fa.items()},
        cost_min_minus_query={k: v["cost_min"] - fa["query"]["cost_min"] for k, v in fa.items()})
    print(f"[frame age] min cost {out['frame_age_check']['cost_min']} tau {out['frame_age_check']['tau_ms']}", flush=True)
    # B4. identifiability on the logged robot time base (synthetic blobs)
    TS = np.arange(-20.0, 80.1, 2.0)
    ident = {}
    resid = resid - resid.mean(0)          # centred real residuals (heavy tails: rms vs robust sd above)
    for nl, kw in (("gauss_1px", dict(noise_px=1.0)), ("real_residuals", dict(noise_pool=resid))):
        for tt in (10.0, 25.0, 40.0):
            fs = fit_free(synth(rec, cubes, tt, seed=int(tt), **kw), "robot", nb=200, taus=TS)[0]
            ident[f"tau_true_{int(tt)}_noise_{nl}"] = dict(tau_true_ms=tt, noise=nl, tau_fit_ms=fs["tau_ms"],
                                                         error_ms=fs["tau_ms"] - tt,
                                                         boot_p5_p50_p95=fs["tau_boot_p5_p50_p95"],
                                                         cost_rise_at_ms=fs["cost_rise_at_ms"])
            print(f"[ident] noise {nl} tau_true {tt:.0f} -> {fs['tau_ms']:.1f} ms (boot "
                  f"{np.round(fs['tau_boot_p5_p50_p95'], 1)})", flush=True)
    for lab, kw in (("cam_turn_x_1deg", dict(cam_rot_deg=[1.0, 0.0, 0.0])),
                    ("cam_turn_y_1deg", dict(cam_rot_deg=[0.0, 1.0, 0.0])),
                    ("cam_turn_z_1deg", dict(cam_rot_deg=[0.0, 0.0, 1.0])),
                    ("shoulder_lift_offset_1deg", dict(q_off_deg=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0])),
                    ("wrist_flex_offset_1deg", dict(q_off_deg=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))):
        fs = fit_free(synth(rec, cubes, 25.0, 1.0, seed=25, **kw), "robot", nb=200, taus=TS)[0]
        ident["systematic_" + lab] = dict(tau_true_ms=25.0, noise="gauss_1px", tau_fit_ms=fs["tau_ms"],
                                          error_ms=fs["tau_ms"] - 25.0, boot_p5_p50_p95=fs["tau_boot_p5_p50_p95"])
        print(f"[ident] {lab}: tau_true 25 -> {fs['tau_ms']:.1f} ms", flush=True)
    out["identifiability"] = dict(
        note="synthetic wrist blobs = the per-episode fitted cube projected at q(t_exposure(tau_true)) on the logged "
             "robot time base and sample set + noise (Gaussian 1 px per axis, or the real fit residuals resampled); "
             "fitted with the same free-cube fit. "
             "'systematic_*': the synthetic data has a model error (wrist camera turned 1 deg in its own frame, or a "
             "1 deg joint offset) that the fit does not know", **ident)
    # teleop check
    calib = {k: -float(v) * 1000.0 / 30.0 for k, v in L.calib()["time_offset_frames"].items()}
    ft, _, _ = fit_free(tel, "teleop")
    out["free_cube_teleop_wrist"] = dict(ft, calib_time_offset_ms=calib["wrist"])
    print(f"[free cube] teleop wrist {ft['tau_ms']:.1f} ms (boot {np.round(ft['tau_boot_p5_p50_p95'], 1)}), calib "
          f"{calib['wrist']:.1f} ms; cost {ft['cost_at_ms']}", flush=True)
    # C. total image lag relative to the joint reading of the query, every robot query; two pipeline estimates
    ages, G = {"external": [], "wrist": []}, []
    for e in L.list_episodes():
        d = L.load(e)
        steps = d["qry_step"].astype(int)
        mc, mf, mp, mm, mg = (d[k] for k in ("ms_ctrl", "ms_frames", "ms_pre", "ms_model", "ms_guard"))
        G.append(mc[steps] - mf[steps] - mp[steps] - mm[steps] - mg[steps])
        ages["external"].append(d["qry_frame_age_ms"][:, 0])
        ages["wrist"].append(d["qry_frame_age_ms"][:, 1])
    G = np.concatenate(G)
    mean_tel_age = 1000.0 / 30.0 / 2.0

    def lag_stats(tt, cam):
        lag = tt + np.concatenate(ages[cam]) - G
        return dict(pipeline_ms=float(tt), lag_ms=L.pct(lag, (5, 25, 50, 75, 95)),
                    lag_steps=L.pct(lag / 20.0, (5, 25, 50, 75, 95)),
                    steps_rounded_frac={str(k): float(np.mean(np.round(lag / 20.0) == k)) for k in range(0, 5)})

    tot, tot2 = {}, {}
    tau_ext = tau_r + (calib["external"] - calib["wrist"])
    for cam, tau in (("wrist", tau_r), ("external", tau_ext)):
        for lab, tt in (("central", tau), ("low", tau + bs_lo - tau_r), ("high", tau + bs_hi - tau_r)):
            tot.setdefault(cam, {})[lab] = lag_stats(tt, cam)
        tot2[cam] = lag_stats(calib[cam] - mean_tel_age, cam)
    out["total_lag"] = tot
    out["total_lag_calib_minus_mean_teleop_age"] = dict(
        note="pipeline = calib time offset - the mean teleop frame age at the read (1/2 frame at 30 fps = 16.7 ms; "
             "LeRobot get_observation reads the motors first, then the non-blocking read_latest)", **tot2)
    out["lag_estimates_p50_steps"] = {cam: dict(direct_fit=tot[cam]["central"]["lag_steps"]["p50"],
                                                calib_minus_mean_teleop_age=tot2[cam]["lag_steps"]["p50"])
                                      for cam in ("wrist", "external")}
    out["external_pipeline_note"] = ("direct fit: external pipeline = robot wrist pipeline + (calib external - calib "
                                     f"wrist offset) = {tau_r:.1f} + {calib['external'] - calib['wrist']:.1f} ms; the held "
                                     "cube moves too slowly in the external view to fit it directly (bias-removal row)")
    (L.OUT / "lag_fit.json").write_text(json.dumps(L.r3(out, 3), indent=1))
    print(json.dumps(L.r3(out["lag_estimates_p50_steps"], 2), indent=1))


if __name__ == "__main__":
    main()
