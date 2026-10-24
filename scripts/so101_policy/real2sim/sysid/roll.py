"""(6) Wrist roll usage vs the real stop at +101.2 sim deg (LeRobot calibrated range end 177 deg + the -75.79 deg
joint-map offset). CPU only. Writes OUT/roll.json.

Per robot step: measured roll q_sim[:, 4], written command u_sim[:, 4], student target u_target[:, 4] (sim deg).
Per episode: max measured / commanded roll, time near the stop, guard stall on the roll, roll at the hold start.
Teleop: the operator's roll range (sim deg).

    PYTHONPATH=<wt>/src ~/miniconda3/envs/so101_vision/bin/python roll.py
"""
from __future__ import annotations

import json

import numpy as np

import sysid_lib as L
import table_height as TH


def main():
    cal = L.calib()
    off = float(np.degrees(cal["joint_map"]["offset_rad"][4]))
    eps = L.list_episodes()
    lim_hi = None
    rows, Q, U, T = [], [], [], []
    for e in eps:
        d = L.load(e)
        meta = json.loads(str(d["meta"]))
        hi = meta.get("limits", {}).get("hi", [None] * 5)[4]
        lo = meta.get("limits", {}).get("lo", [None] * 5)[4]
        stop = hi + off if hi is not None else None
        lim_hi = stop if lim_hi is None else lim_hi
        q = np.degrees(d["q_sim"][:, 4])
        u = np.degrees(d["u_sim"][:, 4])
        ut = np.degrees(d["u_target"][:, 4])
        t = d["t"] - d["t"][0]
        jaw = np.degrees(d["q_sim"][:, 5])
        jc = np.degrees(np.where(np.isfinite(d["u_sim"][:, 5]), d["u_sim"][:, 5], d["q_sim"][:, 5]))
        held = TH.fill_gaps((jaw >= TH.HELD_JAW[0]) & (jaw <= TH.HELD_JAW[1]) & (jc < jaw - TH.SQUEEZE), t, 0.3)
        hr = TH.runs(held, t, TH.HOLD_MIN_S)
        k = hr[0][0] if hr else None
        Q.append(q)
        U.append(u[np.isfinite(u)])
        T.append(ut)
        stall = "wrist_roll" in str(e.get("end_reason"))
        rows.append(dict(id=e["id"], student=e["student"], stop_deg=stop, lo_sim_deg=(lo + off) if lo is not None else None,
                         home_roll=float(q[0]), roll_max=float(q.max()), roll_min=float(q.min()),
                         cmd_max=float(np.nanmax(u)), target_max=float(ut.max()),
                         s_within_2deg_of_stop=float((q > stop - 2.0).sum() * 0.02) if stop else None,
                         s_target_beyond_stop=float((ut > stop).sum() * 0.02) if stop else None,
                         stall=stall, end_reason=e.get("end_reason"), grasp=k is not None,
                         roll_at_hold=float(q[k]) if k is not None else None,
                         roll_max_after_hold=float(q[k:].max()) if k is not None else None))
    q, u, ut = np.concatenate(Q), np.concatenate(U), np.concatenate(T)
    stop = lim_hi
    out = dict(n_episodes=len(rows), n_steps=int(len(q)), stop_sim_deg=stop, joint_map_offset_deg=off,
               sim_range_deg=[-154.35, 159.92],
               measured_roll_deg=L.pct(q, (1, 5, 25, 50, 75, 95, 99)),
               command_roll_deg=L.pct(u, (1, 5, 25, 50, 75, 95, 99)),
               target_roll_deg=L.pct(ut, (1, 5, 25, 50, 75, 95, 99)),
               frac_steps_measured_within_2deg_of_stop=float(np.mean(q > stop - 2)),
               frac_steps_target_beyond_stop=float(np.mean(ut > stop)),
               episodes_reaching_stop=int(sum(r["roll_max"] > stop - 2 for r in rows)),
               episodes_target_beyond_stop=int(sum(r["target_max"] > stop for r in rows)),
               episodes_roll_stall=int(sum(r["stall"] for r in rows)),
               roll_at_hold=L.pct([r["roll_at_hold"] for r in rows if r["roll_at_hold"] is not None]),
               roll_max_after_hold=L.pct([r["roll_max_after_hold"] for r in rows if r["roll_max_after_hold"] is not None]),
               home_roll=L.pct([r["home_roll"] for r in rows]),
               hist_measured=dict(edges=list(range(-60, 121, 10)),
                                  counts=np.histogram(q, bins=list(range(-60, 121, 10)))[0].tolist()))
    by = {}
    for r in rows:
        b = by.setdefault(r["student"], dict(n=0, stall=0, reach=0, target_beyond=0))
        b["n"] += 1
        b["stall"] += int(r["stall"])
        b["reach"] += int(r["roll_max"] > stop - 2)
        b["target_beyond"] += int(r["target_max"] > stop)
    out["by_student"] = by
    # teleop roll range
    import pyarrow.parquet as pq
    tq = []
    for ds in ("green_cube_red_plate", "green_cube_red_plate_1"):
        t = pq.read_table(L.RAW / ds / "data/chunk-000/file-000.parquet").to_pydict()
        tq.append(np.degrees(L.real_to_sim(np.asarray(t["observation.state"], np.float64))[:, 4]))
    tq = np.concatenate(tq)
    out["teleop_roll_deg"] = L.pct(tq, (1, 5, 50, 95, 99))
    out["rows"] = rows
    (L.OUT / "roll.json").write_text(json.dumps(L.r3(out, 3), indent=1))
    print(json.dumps(L.r3({k: v for k, v in out.items() if k != "rows"}, 2), indent=1))


if __name__ == "__main__":
    main()
