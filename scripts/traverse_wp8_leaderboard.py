#!/usr/bin/env python
"""Leaderboard of the wp8 stall ablation: in-training validation stall metrics (best checkpoint) + the local
stall tests on f105 / f104 + the f105 pick table. One row per run."""
from __future__ import annotations
import json, re, sys
from pathlib import Path

OUT = Path("artifacts/traverse/wp8_eval")

def parse_analyze(path: Path) -> dict:
    """pull the per-class row of the single model in an analyze_*.txt"""
    if not path.exists():
        return {}
    res, cls = {}, None
    for line in path.read_text().splitlines():
        m = re.match(r"--- (\w+) \(n=(\d+)\)", line)
        if m:
            cls = m.group(1); continue
        if cls and line.startswith(("wp", "ckpt")) or (cls and line[:1].isalnum() and "|" in line and not line.startswith("model")):
            parts = line.split("|")
            head = parts[0].split()
            try:
                rest_vx, rest_gt1, healthy, wrong, done = map(float, head[1:6])
                pre = parts[1].split(); pre_vx, pre_lt, pre_wrong, trk_vx, trk_lt = map(float, pre[:5])
                stall = re.search(r"in-stall \+4s ([\d.]+) \(<0.5: ([\d.]+)\)", line)
                res[cls] = {"rest_rec_vx3": rest_vx, "rest_trk_done": done, "pre_rec_vx": pre_vx, "pre_rec_stop": pre_lt, "pre_trk_stop": trk_lt,
                            "in_stall_vx": float(stall.group(1)) if stall else None, "in_stall_hold": float(stall.group(2)) if stall else None}
            except (ValueError, IndexError):
                pass
    return res

rows = []
for d in sorted(OUT.glob("*/")):
    run = d.name
    r = {"run": run}
    log = Path("artifacts/traverse") / run / "train_log.jsonl"
    if log.exists():
        vals = [json.loads(l) for l in log.read_text().splitlines() if '"phase": "val"' in l]
        if vals:
            best = min(vals, key=lambda v: v.get("stall_score", float("inf")))
            r.update(step=best["step"], stall_score=best.get("stall_score"), v_stuck=best.get("stall_stuck_hold"), v_appr=best.get("stall_approach_hold"),
                     v_launch=best.get("stall_launch_hold"), v_match=best.get("stall_matched_vx"), v_recov=best.get("stall_recovery_vx"), z1_100=best.get("z1_mae_norm@100"))
    for arena in ("f105", "f104"):
        a = parse_analyze(d / f"analyze_{arena}.txt")
        for cls, key in (("stop", "S"), ("launch", "L"), ("feasible", "F")):
            if cls in a:
                r[f"{arena}_{key}_pre_stop"] = a[cls]["pre_rec_stop"]; r[f"{arena}_{key}_trk_done"] = a[cls]["rest_trk_done"]; r[f"{arena}_{key}_hold"] = a[cls]["in_stall_hold"]
    pk = d / "pick_f105.json"
    if pk.exists():
        try:
            p = json.loads(pk.read_text())
            row = p.get(f"world model: {run}") or next((v for k, v in p.items() if k.startswith("world model")), None)
            if row:
                r.update(pick_feasible=row.get("pick_feasible"), pick_regret=row.get("regret_mean"), pick_layouts=row.get("layouts"),
                         rej_feas=row.get("rejected_feasible"), rej_inf=row.get("rejected_infeasible"))
            fast = p.get("fastest (mean commanded speed)")
            if fast:
                r["fastest_feasible"] = fast.get("pick_feasible")
        except Exception:
            pass
    rows.append(r)

cols = ["run", "step", "stall_score", "v_stuck", "v_appr", "v_launch", "v_match", "v_recov", "z1_100",
        "f105_S_pre_stop", "f105_S_hold", "f105_L_pre_stop", "f105_F_pre_stop", "f105_S_trk_done", "f105_F_trk_done",
        "f104_S_pre_stop", "f104_S_hold", "f104_L_pre_stop", "f104_F_pre_stop", "pick_feasible", "pick_regret", "rej_feas", "rej_inf"]
fmt = lambda v: ("" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v)))
print("  ".join(f"{c:>15s}" if c != "run" else f"{c:28s}" for c in cols))
for r in sorted(rows, key=lambda r: (r.get("stall_score") is None, r.get("stall_score") or 0)):
    print("  ".join(f"{fmt(r.get(c)):>15s}" if c != "run" else f"{fmt(r.get(c)):28s}" for c in cols))
print("\nv_* = in-training validation (f105) at the best stall_score checkpoint: hold = fraction of event windows kept under 0.5 m/s "
      "(stuck: seeded 0.2 s after the stop, 3 s; appr: context 2 s before the stop, 4 s; launch: 3 s); v_match / v_recov = |pred-rec| vx (m/s) on "
      "matched feasible crossings / recoveries (guards against 'always stop'). f105/f104_*: local tests from the recorded context "
      "(S = stops, L = launch failures, F = feasible controls): pre_stop = fraction predicting < 0.5 m/s 2 s after the stop (F: false stops), "
      "hold = in-stall hold fraction, trk_done = closed-loop tracker from rest completes. pick_* = f105 shared-bank pick table.")
