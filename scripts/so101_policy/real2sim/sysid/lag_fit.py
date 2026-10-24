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
should match the calibration's time_offset_frames (pipeline + the teleop frame age at read).

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


def wrist_cam():
    c = L.calib()["cameras"]["wrist"]
    T = c["T_parent_cam"]
    return dict(K=np.array(c["K"]), D=np.array(c["D"]), t=np.array(T[:3]), R=quat_R(T[3:]))


def mount_poses(q):
    kin = L.layer().kin
    p, R = kin.body_poses(torch.as_tensor(np.asarray(q, np.float64)).reshape(-1, 6))
    i = list(__import__("nedm.so101_policy.kin", fromlist=["BODIES"]).BODIES).index("camera_mount")
    return p[:, i].numpy(), R[:, i].numpy()


def project_wrist(Xw, pm, Rm):
    """World points [N, 3] with mount poses [N] -> wrist raw px [N, 2] and camera depth [N]."""
    c = wrist_cam()
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
                        E=np.array(E, float).reshape(-1, 4), steps=steps))
    return out


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


def cost_matrix_free(rec, kind):
    """kind 'robot' or 'teleop' -> (C [tau, ep], n [ep], cube fits at the best tau)."""
    C, ns = [], None
    for tau in TAUS:
        row, ns = [], []
        for r in rec:
            S = r["W"]
            if len(S) < 3:
                row.append(0.0)
                ns.append(0)
                continue
            j = S[:, 0].astype(int)
            te = (r["t_grab"][j] - r["age"][j, 1] if kind == "robot" else r["ts"][j]) - tau * 1e-3
            qe = interp_q(te, r["ts"], r["q"])
            pm, Rm = mount_poses(qe)
            h, n, _ = free_cube_cost(S[:, 1:3], r["cube"], pm, Rm)
            row.append(h)
            ns.append(n)
        C.append(row)
    return np.array(C), np.array(ns)


def fit_free(rec, kind, nb=500, seed=0):
    C, n = cost_matrix_free(rec, kind)
    costs = C.sum(1) / max(n.sum(), 1)
    g = np.random.default_rng(seed)
    use = np.flatnonzero(n > 0)
    bs = [argmin_tau(C[:, (idx := g.choice(use, len(use), replace=True))].sum(1) / n[idx].sum()) for _ in range(nb)]
    return argmin_tau(costs), costs, np.array(bs)


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
    tau, costs, bs = fit_tau(residuals_teleop, tel, "wrist")
    out["bias_removal_teleop_wrist"] = dict(n_samples=int(sum(len(r["W"]) for r in tel)), n_episodes=len(tel), tau_ms=tau,
                                            tau_boot_p5_p50_p95=np.percentile(bs, [5, 50, 95]).tolist(), cost=costs.tolist())
    print(f"[bias removal] teleop wrist: tau {tau:.1f} ms (boot {np.percentile(bs, 5):.1f}-{np.percentile(bs, 95):.1f})")
    # B. free cube position per episode (main result)
    tau_r, costs, bs_r = fit_free(rec, "robot")
    out["free_cube_robot_wrist"] = dict(n_samples=int(sum(len(r["W"]) for r in rec)),
                                        n_episodes=int(sum(len(r["W"]) >= 3 for r in rec)), tau_pipeline_ms=tau_r,
                                        tau_boot_p5_p50_p95=np.percentile(bs_r, [5, 50, 95]).tolist(), cost=costs.tolist())
    tau_t, costs, bs_t = fit_free(tel, "teleop")
    calib = {k: -float(v) * 1000.0 / 30.0 for k, v in L.calib()["time_offset_frames"].items()}
    out["free_cube_teleop_wrist"] = dict(n_samples=int(sum(len(r["W"]) for r in tel)), n_episodes=len(tel), tau_ms=tau_t,
                                         tau_boot_p5_p50_p95=np.percentile(bs_t, [5, 50, 95]).tolist(),
                                         cost=costs.tolist(), calib_time_offset_ms=calib["wrist"])
    print(f"[free cube] robot wrist tau {tau_r:.1f} ms (boot {np.percentile(bs_r, 5):.1f}-{np.percentile(bs_r, 95):.1f}); "
          f"teleop wrist {tau_t:.1f} ms (boot {np.percentile(bs_t, 5):.1f}-{np.percentile(bs_t, 95):.1f}), calib "
          f"{calib['wrist']:.1f} ms", flush=True)
    # C. total image lag relative to the joint reading of the query, every robot query
    ages, G = {"external": [], "wrist": []}, []
    for e in L.list_episodes():
        d = L.load(e)
        steps = d["qry_step"].astype(int)
        mc, mf, mp, mm, mg = (d[k] for k in ("ms_ctrl", "ms_frames", "ms_pre", "ms_model", "ms_guard"))
        G.append(mc[steps] - mf[steps] - mp[steps] - mm[steps] - mg[steps])
        ages["external"].append(d["qry_frame_age_ms"][:, 0])
        ages["wrist"].append(d["qry_frame_age_ms"][:, 1])
    G = np.concatenate(G)
    tot = {}
    tau_ext = tau_r + (calib["external"] - calib["wrist"])
    for cam, tau in (("wrist", tau_r), ("external", tau_ext)):
        for lab, tt in (("central", tau), ("low", tau + np.percentile(bs_r, 5) - tau_r),
                        ("high", tau + np.percentile(bs_r, 95) - tau_r)):
            lag = tt + np.concatenate(ages[cam]) - G
            tot.setdefault(cam, {})[lab] = dict(pipeline_ms=float(tt), lag_ms=L.pct(lag, (5, 25, 50, 75, 95)),
                                                lag_steps=L.pct(lag / 20.0, (5, 25, 50, 75, 95)),
                                                steps_rounded_frac={str(k): float(np.mean(np.round(lag / 20.0) == k))
                                                                    for k in range(0, 5)})
    out["total_lag"] = tot
    out["external_pipeline_note"] = ("external pipeline = robot wrist pipeline + (calib external - calib wrist offset) "
                                     f"= {tau_r:.1f} + {calib['external'] - calib['wrist']:.1f} ms; the held cube "
                                     "moves too slowly in the external view to fit it directly (bias-removal row)")
    (L.OUT / "lag_fit.json").write_text(json.dumps(L.r3(out, 3), indent=1))
    print(json.dumps(L.r3(tot, 2), indent=1))


if __name__ == "__main__":
    main()
