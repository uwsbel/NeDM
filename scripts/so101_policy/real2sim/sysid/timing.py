"""(2) camera timing and (3) loop timing of all robot episodes. CPU only. Writes OUT/timing.json.

Robot loop (run_student_real.control_loop): at each step t (loop top) -> read q (ms_read) -> Controller.step: qdot
(ms_qdot), measured checks, on a query step the newest camera frames (ms_frames; frame age = now - the LeRobot
background thread's timestamp of that frame, taken after the decode), preprocess (ms_pre), the student (ms_model),
then the action layer + guard (ms_guard) -> goal write (ms_write). ms_ctrl = the whole Controller.step.

Image lag relative to the joint reading (per query): lag = capture_pipeline + frame_age - (grab time - joint time).
grab time - joint time = ms_qdot + the measured checks (= ms_ctrl - ms_qdot - ms_frames - ms_pre - ms_model -
ms_guard on query steps) ~ a few ms. capture_pipeline (exposure -> timestamp) is NOT in the logs: the calibration's
time_offset_frames (teleop, 30 fps) is pipeline + the teleop frame age at read (LeRobot get_observation: motors first,
then read_latest, age uniform 0-33 ms if the phases drift). Two variants are reported: pipeline = 45 ms (the
calibration offset, an upper bound) and pipeline = 45 - 16.7 ms (mean teleop age removed); lag_fit.py measures the
pipeline directly from the wrist camera.

    PYTHONPATH=<wt>/src ~/miniconda3/envs/so101_vision/bin/python timing.py
"""
from __future__ import annotations

import json

import numpy as np

import sysid_lib as L

STEP_MS = 20.0


def main():
    cal = L.calib()
    off = cal["time_offset_frames"]
    calib_ms = {k: -float(v) * 1000.0 / 30.0 for k, v in off.items()}
    eps = L.list_episodes()
    age = {"external": [], "wrist": []}
    grab_minus_joint = []
    dt_all, dt_q, dt_nq, dt_after_q = [], [], [], []
    qry_dt, rates, overruns, late, per_ep = [], [], [], [], []
    cmd_delay_q, cmd_delay_nq = [], []
    tw_after_read, tw_hold = [], []          # per chunk: write time of targets 0 .. 4 (+ the next chunk's 0) after the read
    ms_model = []
    by_session = {}
    for e in eps:
        d = L.load(e)
        t = d["t"]
        N = len(t)
        qmask = d["query"].astype(bool)
        a = d["qry_frame_age_ms"]
        cams = ("external", "wrist")
        for i, c in enumerate(cams):
            age[c].append(a[:, i])
        steps = d["qry_step"].astype(int)
        mc, mq, mf, mp, mm, mg = (d[k] for k in ("ms_ctrl", "ms_qdot", "ms_frames", "ms_pre", "ms_model", "ms_guard"))
        chk = mc[steps] - mq[steps] - mf[steps] - mp[steps] - mm[steps] - mg[steps]
        grab_minus_joint.append(mq[steps] + chk)
        ms_model.append(mm[steps])
        dt = np.diff(t) * 1e3
        dt_all.append(dt)
        isq = qmask[:-1]
        dt_q.append(dt[isq])
        dt_nq.append(dt[~isq])
        # command delay: joint read end -> goal write (ms_ctrl + half the read; the write itself ~0.2 ms)
        mr = d["robot_ms_read"][:N]
        cd = mc + 0.5 * mr
        cmd_delay_q.append(cd[qmask])
        cmd_delay_nq.append(cd[~qmask])
        # chunk timeline: read of the query step s at t[s] + mr[s] / 2, goal write of step s + k at t[s + k] + mr + ms_ctrl
        tw_ = t + (mr + mc) * 1e-3
        tr_ = t + 0.5 * mr * 1e-3
        for s0 in steps:
            if s0 + 5 < N and np.isfinite(tw_[s0:s0 + 6]).all():
                w = (tw_[s0:s0 + 6] - tr_[s0]) * 1e3
                tw_after_read.append(w)
                tw_hold.append(np.diff(w))
        qt = d["qry_t"]
        qry_dt.append(np.diff(qt) * 1e3)
        T = t[-1] - t[0]
        rates.append((N - 1) / T if T > 0 else np.nan)
        ov = int(d["robot_overruns"]) if "robot_overruns" in d.files else -1
        overruns.append(ov)
        late.append(d["robot_late_ms"])
        per_ep.append(dict(id=e["id"], steps=N, T_s=float(T), rate_hz=float(rates[-1]), overruns=ov,
                           overrun_per_query=float(ov / max(qmask.sum(), 1)),
                           age_ext_p50=float(np.median(a[:, 0])), age_wr_p50=float(np.median(a[:, 1])),
                           age_ext_p95=float(np.percentile(a[:, 0], 95)), age_wr_p95=float(np.percentile(a[:, 1], 95)),
                           dt_q_p50=float(np.median(dt[isq])), qry_dt_p50=float(np.median(np.diff(qt) * 1e3))))
        s = by_session.setdefault(e["session"], dict(n_ep=0, ages_ext=[], ages_wr=[], rates=[]))
        s["n_ep"] += 1
        s["ages_ext"].append(a[:, 0])
        s["ages_wr"].append(a[:, 1])
        s["rates"].append(rates[-1])
    cat = lambda x: np.concatenate(x)
    A = {c: cat(v) for c, v in age.items()}
    G = cat(grab_minus_joint)
    out = dict(n_episodes=len(eps), n_queries=int(len(A["external"])), n_steps=int(sum(len(x) + 1 for x in dt_all)))
    out["frame_age_ms"] = {c: L.pct(v, (1, 5, 25, 50, 75, 95, 99)) for c, v in A.items()}
    out["frame_age_hist_ms"] = {c: np.histogram(v, bins=np.arange(0, 80, 5))[0].tolist() for c, v in A.items()}
    out["grab_minus_joint_ms"] = L.pct(G)
    out["calib_time_offset_ms"] = calib_ms
    lag = {}
    for variant, sub in (("pipeline_calib", 0.0), ("pipeline_calib_minus_mean_teleop_age", 1000.0 / 30.0 / 2)):
        lag[variant] = {}
        for i, c in enumerate(("external", "wrist")):
            pipe = calib_ms[c] - sub
            v = pipe + A[c] - G
            st = v / STEP_MS
            lag[variant][c] = dict(pipeline_ms=pipe, lag_ms=L.pct(v, (5, 25, 50, 75, 95)),
                                   lag_steps=L.pct(st, (5, 25, 50, 75, 95)),
                                   steps_rounded_frac={str(k): float(np.mean(np.round(st) == k)) for k in range(0, 7)})
    out["image_lag"] = lag
    DT = cat(dt_all)
    out["loop"] = dict(
        step_period_ms_all=L.pct(DT, (1, 5, 25, 50, 75, 95, 99)),
        step_period_ms_query_steps=L.pct(cat(dt_q), (5, 25, 50, 75, 95)),
        step_period_ms_other_steps=L.pct(cat(dt_nq), (5, 25, 50, 75, 95)),
        chunk_period_ms=L.pct(cat(qry_dt), (5, 25, 50, 75, 95)),
        chunk_period_sim_ms=5 * STEP_MS,
        rate_hz_per_episode=L.pct(rates),
        overruns_per_episode=L.pct(overruns),
        overruns_per_query=L.pct([p["overrun_per_query"] for p in per_ep]),
        late_ms=L.pct(cat(late), (50, 95, 99)),
        frac_steps_over_25ms=float(np.mean(DT > 25)), frac_steps_over_40ms=float(np.mean(DT > 40)),
        model_ms=L.pct(cat(ms_model), (5, 50, 95, 99)),
        cmd_delay_ms_query_steps=L.pct(cat(cmd_delay_q), (5, 50, 95)),
        cmd_delay_ms_other_steps=L.pct(cat(cmd_delay_nq), (5, 50, 95)),
        chunk_period_steps=dict(
            note="chunk period / 20 ms (sim decision = 5 steps)", pct=L.pct(cat(qry_dt) / STEP_MS, (5, 25, 50, 75, 95)),
            rounded_frac={str(k): float(np.mean(np.round(cat(qry_dt) / STEP_MS) == k)) for k in range(4, 11)},
            rounded_frac_ge_11=float(np.mean(np.round(cat(qry_dt) / STEP_MS) >= 11))),
        target_timeline_ms=dict(
            note="per chunk: goal write of targets 0 .. 4 and of the next chunk's target 0 after the joint read of the "
                 "query (write_after_read, p50 / p5 / p95 for each index), and how long each of targets 0 .. 4 is the "
                 "newest written goal (hold, ms)",
            write_after_read={str(k): L.pct(np.array(tw_after_read)[:, k], (5, 50, 95)) for k in range(6)},
            hold={str(k): L.pct(np.array(tw_hold)[:, k], (5, 50, 95)) for k in range(5)}, n_chunks=len(tw_after_read)),
        period_hist_ms=dict(edges=[0, 15, 18, 19, 19.5, 20.5, 21, 22, 25, 30, 40, 45, 50, 60, 100, 1000],
                            counts=np.histogram(DT, bins=[0, 15, 18, 19, 19.5, 20.5, 21, 22, 25, 30, 40, 45, 50, 60, 100,
                                                           1000])[0].tolist()))
    out["by_session"] = {k: dict(n_ep=v["n_ep"], age_ext_p50=float(np.median(cat(v["ages_ext"]))),
                                 age_wr_p50=float(np.median(cat(v["ages_wr"]))),
                                 age_ext_p95=float(np.percentile(cat(v["ages_ext"]), 95)),
                                 age_wr_p95=float(np.percentile(cat(v["ages_wr"]), 95)),
                                 rate_hz_p50=float(np.median(v["rates"]))) for k, v in by_session.items()}
    out["per_episode"] = per_ep
    (L.OUT / "timing.json").write_text(json.dumps(L.r3(out, 3), indent=1))
    print(json.dumps(L.r3({k: v for k, v in out.items() if k not in ("per_episode", "by_session")}, 2), indent=1))


if __name__ == "__main__":
    main()
