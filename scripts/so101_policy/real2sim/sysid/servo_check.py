"""(5) Check of the fitted real servo (RobotDynConfig.servo, ~/so101_r2s_claude/sim_debug/scout/tr_servo.md: fit on 12
robot episodes of 10-22 afternoon + teleop) on ALL robot episodes. CPU only. Writes OUT/servo_check.json.

The sent command (u_sim, after the guard incl. the integral correction) drives the fitted model on the robot's own time
base (q read at t + ms_read / 2, the command acts from t + robot_ms_step), as the fit did (servo_sim.py = a copy of
the fit's simulator). Reported per joint (deg): free-run RMS (model never reset), decision-horizon RMS (reset to the
measured q at every query), one-step RMS; for the 12 fit episodes and for the 103 later / other episodes, and per
session. Also a small grid around the evaluation setting (delay, lag, dead band, gravity compliance scales) on the
new episodes: does the optimum move?

Outliers: the new episodes whose free-run shoulder_lift RMS (fitted setting) exceeds OUTLIER_LIFT_DEG are listed with
their wrist / external views at the queries of the largest shoulder_lift error (img/servo_outlier_views.jpg) and the
same metrics are repeated without them. Per-episode values: servo_per_episode.json.

The MuJoCo model (gravity torques): --xml, else $SO101_XML, else the repo's pinned vendor copy
(nedm.render.so101.vendor_xml: contact_nrd/so101_grasp/vendor, fetched by contact_nrd/so101_grasp/fetch_sources.py).

    PYTHONPATH=<wt>/src ~/miniconda3/envs/so101_vision/bin/python servo_check.py [--xml <so101.xml>]
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch

import servo_sim as SS
import sysid_lib as L

XML = None                                     # set in main (resolve_xml)
OUTLIER_LIFT_DEG = 1.0                         # free-run shoulder_lift RMS above this = outlier episode (deg)
GRASP_POINT = np.array([0.0105, 0.0, -0.092])
FIT_EPISODES = {"live_20261022_144731", "live_20261022_152157", "live_20261022_152431"} | \
    {f"eval_20261022_151223/ep_0{i}" for i in range(1, 7)} | {f"eval_20261022_153448/ep_0{i}" for i in range(1, 4)}
D2R = math.pi / 180
PROP = dict(                                   # eval_final.py "proposed" = RobotDynConfig.servo evaluation setting
    delay_ms=[25, 25, 20, 25, 20, 45],
    tau_ms=[50, 50, 45, 40, 40, 45],
    deadband_deg=[0.35, 0.30, 0.74, 0.37, 0.44, 0.37],
    deadband_per_Nm_deg=[0.0, 1.7, 1.3, 0.8, 0.0, 0.0],
    breakaway_extra_deg=[0.14, 0.0, 0.02, 0.09, 0.09, 0.29],
    compliance_rad_per_Nm=[0.0, 0.060, 0.055, 0.055, 0.0, 0.0],
    bias_deg=[-0.07, -0.22, 0.18, 0.0, -0.07, 0.03],
    vmax_rad_s=[3.0, 2.3, 3.0, 3.0, 1.0, 2.2],
    cube_mass_kg=0.03)
LIM = np.array([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533])


def resolve_xml(arg=None):
    """--xml, else $SO101_XML, else the repo's pinned vendor copy (read only)."""
    import os
    from pathlib import Path
    for c in (arg, os.environ.get("SO101_XML")):
        if c:
            if not Path(c).exists():
                raise FileNotFoundError(c)
            return str(c)
    from nedm.render.so101 import vendor_xml
    return str(vendor_xml())


def gravity(q):
    import mujoco
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    jac = np.zeros((3, m.nv))
    bias = np.zeros((len(q), 6))
    cube = np.zeros((len(q), 6))
    for k, qq in enumerate(q):
        d.qpos[:6] = qq
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        bias[k] = d.qfrc_bias[:6]
        p = d.xpos[gid] + d.xmat[gid].reshape(3, 3) @ GRASP_POINT
        mujoco.mj_jac(m, d, jac, None, p, gid)
        cube[k] = jac[:, :6].T @ np.array([0, 0, 9.81])
    return bias, cube


def load_ep(e):
    """load.py of the fit, on a sysid episode dict."""
    d = L.load(e)
    t = d["t"] - d["t"][0]
    n = len(t)
    pad = lambda x: np.concatenate((x[:n], np.full(max(0, n - len(x)), np.nan)))
    ms_read = pad(d["robot_ms_read"])
    ms_step = pad(d["robot_ms_step"])
    tr = t + 0.5 * np.nan_to_num(ms_read, nan=1.2) / 1e3
    tw = t + ms_step / 1e3
    q = d["q_sim"].astype(np.float64)
    u = d["u_sim"].astype(np.float64)
    ok = np.all(np.isfinite(u), 1) & np.isfinite(ms_step)
    # rows after the first non-ok row are not used (the fit's loader assumed a contiguous ok prefix)
    bad = np.flatnonzero(~ok)
    if len(bad):
        ok[bad[0]:] = False
    gq, gu = np.degrees(q[:, 5]), np.degrees(np.nan_to_num(u[:, 5], nan=0.0))
    return dict(name=e["id"], session=e["session"], t=t, tr=tr, tw=tw, q=q, u=np.nan_to_num(u), query=d["query"].astype(bool),
                ok=ok, n=n, jaw_block=(gq < 11.0) & (gu < gq - 1.0), held=gq < 11.0)


def blocked(E, thr_deg=4.0, still=6):
    q, u = E["q"], E["u"]
    n = len(q)
    e = np.degrees(np.abs(np.nan_to_num(u - q)))
    dq = np.abs(np.diff(q, axis=0, prepend=q[:1])) > 0
    run = np.zeros((n, 6), int)
    for k in range(1, n):
        run[k] = np.where(dq[k], 0, run[k - 1] + 1)
    b = (e > thr_deg) & (run >= still)
    b |= np.abs(q) > LIM + 1e-3
    return b


def to_P(pr, C=1, scale=None):
    """Parameter dict [C, 6]; scale: list of C dicts {delay_add_ms, tau_x, db_x, kg_x}."""
    rows = []
    for c in range(C):
        s = (scale or [{}] * C)[c]
        kg = np.asarray(pr["compliance_rad_per_Nm"]) * s.get("kg_x", 1.0)
        rows.append(dict(delay=(np.asarray(pr["delay_ms"]) + s.get("delay_add_ms", 0.0)).clip(0) * 1e-3,
                         tau=np.asarray(pr["tau_ms"]) * s.get("tau_x", 1.0) * 1e-3,
                         vmax=np.asarray(pr["vmax_rad_s"], float),
                         db=np.asarray(pr["deadband_deg"]) * s.get("db_x", 1.0) * D2R,
                         db1=np.asarray(pr["deadband_per_Nm_deg"]) * s.get("db_x", 1.0) * D2R,
                         dbs=np.asarray(pr["breakaway_extra_deg"]) * D2R, kg=kg, kh=kg * pr["cube_mass_kg"],
                         b=np.asarray(pr["bias_deg"]) * D2R, tau2=np.zeros(6)))
    return {k: torch.tensor(np.stack([r[k] for r in rows]), dtype=torch.float32) for k in rows[0]}


def perfect_P():
    z = np.zeros(6)
    t = lambda x: torch.tensor(np.asarray(x, float)[None], dtype=torch.float32)
    return dict(delay=t(z), tau=t(np.full(6, 1e-4)), vmax=t(np.full(6, 100.0)), db=t(z), db1=t(z), dbs=t(z), kg=t(z),
                kh=t(z), b=t(z), tau2=t(z))


def per_ep_rms(D, out, c=0, decision=False):
    m = (D.mask_dec if decision else D.mask)
    err = (out[:, c] - D.qm) * m
    se = (err ** 2).sum(1)
    n = m.sum(1)
    return torch.rad2deg(torch.sqrt(se / n.clamp_min(1))).numpy(), n.numpy()


def pooled(r, n, idx):
    r, n = r[idx], n[idx]
    return np.sqrt((r ** 2 * n).sum(0) / np.maximum(n.sum(0), 1))


def batched(Es, G, P, modes, B=29):
    """Simulate P on the episodes in batches of B -> {mode: (rms [E, C, 6], n [E, 6])}."""
    acc = {m: ([], []) for m in modes}
    C = P["tau"].shape[0]
    for k0 in range(0, len(Es), B):
        D = SS.Data(Es[k0:k0 + B], G, blocked, dev="cpu")
        for m in modes:
            o = SS.simulate(D, P, reset_query=(m == "decision"), reset_all=(m == "one_step"))
            rs, ns = [], None
            for c in range(C):
                r, ns = per_ep_rms(D, o, c, decision=(m == "decision"))
                rs.append(r)
            acc[m][0].append(np.stack(rs, 1))
            acc[m][1].append(ns)
            del o
        del D
    return {m: (np.concatenate(v[0]), np.concatenate(v[1])) for m, v in acc.items()}


def pooled_c(r, n, idx):
    """r [E, C, 6], n [E, 6] -> [C, 6]."""
    r, n = r[idx], n[idx][:, None]
    return np.sqrt((r ** 2 * n).sum(0) / np.maximum(n.sum(0), 1))


def outlier_views(ids, E_by_id, path, n_per_ep=2):
    """Wrist + external query thumbs of each outlier episode at the queries with the largest shoulder_lift
    |command - measured| -> one jpg."""
    import cv2
    eps = {e["id"]: e for e in L.list_episodes()}
    tiles = []
    for i in ids:
        d = L.load(eps[i])
        E = E_by_id[i]
        err = np.degrees(np.abs(E["u"][:, 1] - E["q"][:, 1]))
        steps = d["qry_step"].astype(int)
        order = []
        for j in np.argsort(-err[np.clip(steps, 0, len(err) - 1)]):      # largest errors, >= 10 queries apart
            if all(abs(int(j) - k) >= 10 for k in order):
                order.append(int(j))
            if len(order) == n_per_ep:
                break
        for j in sorted(order):
            row = np.concatenate([cv2.resize(d["qry_thumbs"][j, c], (224, 224), interpolation=cv2.INTER_NEAREST)
                                  for c in (0, 1)], 1).copy()
            cv2.putText(row, f"{i[-24:]} t{E['t'][steps[j]]:.0f}s lift err {err[steps[j]]:.1f}", (3, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
            tiles.append(row)
    if tiles:
        (path.parent).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(tiles, 0), cv2.COLOR_RGB2BGR))


def per_episode_rows(Es, R, names):
    """Per-episode record (fitted setting): RMS per joint (free / decision), max |integral correction| per arm joint,
    the guard's i_max, the lowest measured gripper point minus the calibrated plane (episode minimum, mm)."""
    eps = {e["id"]: e for e in L.list_episodes()}
    rows = []
    for k, E in enumerate(Es):
        d = L.load(eps[E["name"]])
        meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
        g = meta.get("setup", {}).get("guard", {}) if isinstance(meta, dict) else {}
        ic = np.degrees(np.abs(d["i_corr"])).max(0).tolist() if "i_corr" in d.files else None
        P, _ = L.fk(E["q"])
        low = L.lowest(E["q"])
        rows.append(dict(id=E["name"], fit=E["name"] in FIT_EPISODES, i_max_deg=g.get("i_max_deg"),
                         i_corr_max_deg=ic, low_minus_plane_min_mm=float(((low - L.table_z_at(P[:, :2])) * 1e3).min()),
                         free_rms_deg=R["free"][0][k, 0].tolist(), decision_rms_deg=R["decision"][0][k, 0].tolist()))
    return rows


def main():
    import argparse
    global XML
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=None, help="so101.xml (MuJoCo, gravity torques); default $SO101_XML or the repo vendor copy")
    XML = resolve_xml(ap.parse_args().xml)
    torch.set_num_threads(1)            # tiny tensors: one thread is ~10x faster than 16
    eps = L.list_episodes()
    Es, G = [], {}
    for e in eps:
        E = load_ep(e)
        if E["ok"].sum() < 50:
            continue
        b, c = gravity(E["q"])
        G[E["name"] + "|bias"], G[E["name"] + "|cube"] = b, c
        Es.append(E)
    names = [E["name"] for E in Es]
    fit_idx = [i for i, n in enumerate(names) if n in FIT_EPISODES]
    new_idx = [i for i, n in enumerate(names) if n not in FIT_EPISODES]
    print(f"{len(Es)} episodes ({len(fit_idx)} of the fit, {len(new_idx)} new)", flush=True)
    out = dict(n_episodes=len(Es), n_fit=len(fit_idx), n_new=len(new_idx), params=PROP,
               n_steps=int(sum(E["ok"].sum() for E in Es)))
    res = {}
    for label, P in (("fitted_eval_setting", to_P(PROP)), ("perfect_tracking", perfect_P())):
        R = batched(Es, G, P, ("free", "decision", "one_step"))
        res[label] = {grp: {m: pooled_c(R[m][0], R[m][1], ix)[0].tolist() for m in R}
                      for grp, ix in (("fit_eps", fit_idx), ("new_eps", new_idx))}
        res[label]["new_eps_free_per_episode_p50_p90"] = np.percentile(R["free"][0][new_idx, 0], [50, 90], axis=0).tolist()
        if label == "fitted_eval_setting":
            ses = {}
            for i, E in enumerate(Es):
                ses.setdefault(E["session"], []).append(i)
            res[label]["per_session_decision"] = {s: pooled_c(R["decision"][0], R["decision"][1], ix)[0].round(3).tolist()
                                                  for s, ix in ses.items()}
        print(label, json.dumps(L.r3({k: v for k, v in res[label].items() if not k.startswith("per_session")}, 3)),
              flush=True)
    out["rms_deg"] = res
    # outliers (rule: new episodes with free-run shoulder_lift RMS > OUTLIER_LIFT_DEG, fitted setting). On 10-23 these
    # were open-jaw hovers with the fixed jaw tip on the cube (img/servo_outlier_views.jpg, looked at by eye): the arm
    # stays 2-3.6 deg from its command with the integral correction at its 3 deg cap. Same metrics with a stricter
    # contact mask (|u - q| > 2 deg while still >= 6 reads; the fit used 4 deg) and without the outliers.
    R0 = batched(Es, G, to_P(PROP), ("free", "decision"))
    rows = per_episode_rows(Es, R0, names)
    (L.OUT / "servo_per_episode.json").write_text(json.dumps(L.r3(rows, 4), indent=1))
    contact = {names[i] for i in new_idx if R0["free"][0][i, 0, 1] > OUTLIER_LIFT_DEG}
    lift_sorted = np.sort(R0["free"][0][new_idx, 0, 1])[::-1]
    outlier_views(sorted(contact), {E["name"]: E for E in Es}, L.OUT / "img" / "servo_outlier_views.jpg")
    print(f"outliers (free shoulder_lift > {OUTLIER_LIFT_DEG} deg): {sorted(contact)}; top free lift RMS "
          f"{np.round(lift_sorted[:6], 2).tolist()}", flush=True)
    global blocked
    b4 = blocked
    blocked = lambda E: b4(E, thr_deg=2.0)
    R = batched(Es, G, to_P(PROP), ("free", "decision"))
    blocked = b4
    new_nc = [i for i in new_idx if names[i] not in contact]
    out["rms_deg_contact_mask_2deg"] = dict(
        note="fitted setting, contact mask 2 deg (fit: 4 deg)", contact_episodes=sorted(contact),
        outlier_rule=f"new episodes with free-run shoulder_lift RMS > {OUTLIER_LIFT_DEG} deg (fitted setting)",
        free_lift_rms_top6_new=lift_sorted[:6].tolist(),
        new_eps={m: pooled_c(R[m][0], R[m][1], new_idx)[0].tolist() for m in R},
        new_eps_without_contact_eps={m: pooled_c(R[m][0], R[m][1], new_nc)[0].tolist() for m in R},
        fit_eps={m: pooled_c(R[m][0], R[m][1], fit_idx)[0].tolist() for m in R}, n_new_without=len(new_nc))
    print("contact mask 2 deg", json.dumps(L.r3(out["rms_deg_contact_mask_2deg"], 3)), flush=True)
    # grid around the evaluation setting
    grid = []
    for dadd in (-20, 0, 20, 40):
        for tx in (0.6, 1.0, 1.5):
            for dbx in (0.7, 1.0, 1.4):
                for kgx in (0.6, 1.0, 1.4):
                    grid.append(dict(delay_add_ms=dadd, tau_x=tx, db_x=dbx, kg_x=kgx))
    P = to_P(PROP, len(grid), grid)
    R = batched(Es, G, P, ("free", "decision"), B=12)
    dn = pooled_c(R["decision"][0], R["decision"][1], new_idx)
    fn = pooled_c(R["free"][0], R["free"][1], new_idx)
    df = pooled_c(R["decision"][0], R["decision"][1], fit_idx)
    rows = [dict(s, decision_new=dn[c].tolist(), free_new=fn[c].tolist(), decision_fit=df[c].tolist())
            for c, s in enumerate(grid)]
    sc = [np.mean(r["decision_new"][:5]) + np.mean(r["free_new"][:5]) for r in rows]
    out["grid_best_new_arm_mean"] = rows[int(np.argmin(sc))]
    cen = dict(delay_add_ms=0, tau_x=1.0, db_x=1.0, kg_x=1.0)
    out["grid_center_new"] = [r for r in rows if all(r[k] == cen[k] for k in cen)][0]
    # per joint: the best grid point for each joint (decision + free)
    pj = []
    for j in range(6):
        scj = [r["decision_new"][j] + r["free_new"][j] for r in rows]
        k = int(np.argmin(scj))
        pj.append(dict(joint=j, **{kk: rows[k][kk] for kk in cen}, decision_new=rows[k]["decision_new"][j],
                       free_new=rows[k]["free_new"][j]))
    out["grid_best_per_joint_new"] = pj
    marg = {}
    for fac, vals in (("delay_add_ms", (-20, 0, 20, 40)), ("tau_x", (0.6, 1.0, 1.5)), ("db_x", (0.7, 1.0, 1.4)),
                      ("kg_x", (0.6, 1.0, 1.4))):
        m = {}
        for v in vals:
            want = dict(cen, **{fac: v})
            r = [r for r in rows if all(r[k] == want[k] for k in want)][0]
            m[str(v)] = dict(decision_new=r["decision_new"], free_new=r["free_new"])
        marg[fac] = m
    out["grid_marginals_new"] = marg
    out["grid"] = rows
    (L.OUT / "servo_check.json").write_text(json.dumps(L.r3(out, 4), indent=1))
    print("best", json.dumps(L.r3(out["grid_best_new_arm_mean"], 3)))
    print("centre", json.dumps(L.r3(out["grid_center_new"], 3)))
    print("per joint", json.dumps(L.r3(pj, 3)))
    for fac, m in marg.items():
        print(fac, {v: (np.round(x["decision_new"], 2).tolist(), np.round(x["free_new"], 2).tolist()) for v, x in m.items()})


if __name__ == "__main__":
    main()
