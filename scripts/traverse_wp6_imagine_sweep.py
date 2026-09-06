#!/usr/bin/env python
"""Does the world model predict the feasibility map? Imagine the exact sweep routes and compare with Chrono.

For every challenge in a challenge cache, the routes of the terrain sweep (``tasks.json``) are rolled out by
the tracker inside the dynamics model from the rest state at the camera start pose -- the same imagination the
planner uses -- and compared route by route with what Chrono did (``rows.jsonl`` of the sweep): did the
imagination reject (not completed / failed) the routes that stalled or left the route in Chrono, and how close
are imagined time and energy on the feasible ones? Run on arena_v1 (the training terrain) and on the steep
arena the model has never seen.
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nedm.traverse import nrd_data as D
from nedm.traverse.layout import EpisodeLayout
from nedm.traverse.oracle import PlanCandidate
from nedm.traverse.power_calib import KINDS, PowerModel
from nedm.traverse.terrain import TerrainMap
from traverse_wp5_sample_planner import Imaginer


def auc(score, label):
    s, l = np.asarray(score, float), np.asarray(label, int)
    m = np.isfinite(s); s, l = s[m], l[m]
    if l.sum() == 0 or l.sum() == len(l):
        return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        idx = np.nonzero(s == v)[0]; ranks[idx] = ranks[idx].mean()
    n1, n0 = l.sum(), (1 - l).sum()
    return float((ranks[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--challenge-cache", required=True)
    ap.add_argument("--challenges", required=True, help="challenge dir (tasks.json, <key>/meta.json)")
    ap.add_argument("--sweep", required=True, help="Chrono sweep batch dir (rows.jsonl)")
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="artifacts/traverse/wp3_tracker_v1")
    ap.add_argument("--dynamics-checkpoints", nargs="+", default=["artifacts/traverse/wp2_mapv2_pt_dag_ro8_amd/ckpt_best.pt"])
    ap.add_argument("--z1-extra-cache", nargs="*", default=None)
    ap.add_argument("--power-calib", default="artifacts/traverse/wp4_power_calib/power_calib.json")
    ap.add_argument("--energy-floor", default="artifacts/traverse/wp5_energy_floor/energy_floor.json")
    ap.add_argument("--floor-sigmas", type=float, default=1.5)
    ap.add_argument("--pess-terms", nargs="+", default=["head", "state"])
    ap.add_argument("--horizon-s", type=float, default=30.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    a = SimpleNamespace(**vars(args), routes="artifacts/traverse/wp3_routes", map_key="map_v2", from_rest=True, cache=args.challenge_cache)
    cache = Path(args.challenge_cache)
    keys = D.load_cache_keys(cache)
    start_est = json.loads((cache / "start_poses.json").read_text())
    tasks = defaultdict(list)
    for t in json.loads((Path(args.challenges) / "tasks.json").read_text()):
        tasks[t["key"]].append(t)
    chrono = defaultdict(dict)
    for l in Path(args.sweep, "rows.jsonl").read_text().splitlines():
        if l.strip().startswith("{"):
            r = json.loads(l); chrono[r["key"]][r["candidate"]] = r
    tmap = TerrainMap.from_dir(Path(args.arena))
    power_models = {k: PowerModel.load(Path(args.power_calib), k) for k in KINDS}
    imagine = Imaginer(a, power_models, tmap, cache)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in keys:
        if key not in tasks or key not in chrono:
            continue
        meta = json.loads((Path(args.challenges) / key / "meta.json").read_text())
        layout = EpisodeLayout.from_json(meta["layout"])
        plans = [PlanCandidate(waypoints=np.array(t["route"]["waypoints"]), speeds=np.array(t["route"]["speeds"]), headings=np.array(t["route"]["headings"]),
                               stations=np.array(t["route"]["stations"]), meta={"candidate": t["candidate"]}) for t in tasks[key]]
        res = imagine(key, plans, tuple(start_est[key]["est"]), layout.obstacles(), layout)  # the house is the only obstacle
        for i, t in enumerate(tasks[key]):
            c = chrono[key].get(t["candidate"])
            if c is None:
                continue
            feasible = c["completed"] and not c.get("stalled") and c["status"] == "completed"
            rows.append({"key": key, "candidate": t["candidate"], "chrono_feasible": bool(feasible), "chrono_status": c["status"], "chrono_stalled": bool(c.get("stalled")),
                         "chrono_time": c["time_s"], "chrono_energy": c["energy_kj"], "chrono_max_pitch": c["max_pitch_deg"], "chrono_max_roll": c["max_roll_deg"],
                         "img_ok": bool(res["ok"][i]), "img_completed": bool(res["completed"][i]), "img_failed": bool(res["failed"][i]),
                         "img_time": float(res["time_s"][i]), "img_energy": float(res["energy_pess"][i]), "img_energy_head": float(res["energy_kj"][i]),
                         "img_max_pitch": float(res["pred_max_pitch_deg"][i]), "img_max_roll": float(res["pred_max_roll_deg"][i]), "img_min_fz": float(res["pred_min_fz_n"][i])})
        n_ok = sum(r["img_ok"] for r in rows if r["key"] == key); n_f = sum(r["chrono_feasible"] for r in rows if r["key"] == key)
        print(f"{key:26s} imagined ok {n_ok:2d}/{len(tasks[key])}  Chrono feasible {n_f:2d}/{len(tasks[key])}", flush=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    feas = np.array([r["chrono_feasible"] for r in rows]); ok = np.array([r["img_ok"] for r in rows])
    print(f"\n{len(rows)} routes: Chrono infeasible {int((~feas).sum())}, imagination rejects {int((~ok).sum())}")
    print(f"confusion: rejected & infeasible {int((~ok & ~feas).sum())} | rejected & feasible {int((~ok & feas).sum())} | accepted & infeasible {int((ok & ~feas).sum())} | accepted & feasible {int((ok & feas).sum())}")
    print(f"AUC of 'imagination rejects' for Chrono infeasibility: {auc(~ok, ~feas):.2f}; of imagined time (longer = worse): {auc([r['img_time'] for r in rows], ~feas):.2f}; "
          f"of imagined max pitch: {auc([r['img_max_pitch'] for r in rows], ~feas):.2f}; of imagined energy: {auc([r['img_energy'] for r in rows], ~feas):.2f}")
    both = [r for r in rows if r["chrono_feasible"] and r["img_ok"]]
    if both:
        ct, it = np.array([r["chrono_time"] for r in both]), np.array([r["img_time"] for r in both])
        ce, ie = np.array([r["chrono_energy"] for r in both]), np.array([r["img_energy"] for r in both])
        print(f"on the {len(both)} routes both call feasible: time Chrono/imagined {ct.mean() / it.mean():.2f} (MAE {np.abs(ct - it).mean():.2f} s, corr {np.corrcoef(ct, it)[0, 1]:.2f}); "
              f"energy Chrono/imagined {ce.mean() / ie.mean():.2f} (MAE {100 * np.mean(np.abs(ce - ie) / ce):.0f} %, corr {np.corrcoef(ce, ie)[0, 1]:.2f})")
    # per challenge: does the imagination's best route agree with Chrono's best feasible route?
    by = defaultdict(list)
    for r in rows:
        by[r["key"]].append(r)
    agree, regret = 0, []
    for key, rs in by.items():
        f = [r for r in rs if r["chrono_feasible"]]
        if not f:
            continue
        cb = min(f, key=lambda r: r["chrono_time"] + r["chrono_energy"] / 10)
        ib = [r for r in rs if r["img_ok"]]
        if not ib:
            regret.append(None); continue
        ib = min(ib, key=lambda r: r["img_time"] + r["img_energy"] / 10)
        agree += ib["candidate"] == cb["candidate"]
        regret.append(None if not ib["chrono_feasible"] else (ib["chrono_time"] + ib["chrono_energy"] / 10) / (cb["chrono_time"] + cb["chrono_energy"] / 10))
    n_inf = sum(r is None for r in regret); reg = [r for r in regret if r is not None]
    print(f"imagined best route: same as Chrono's best feasible on {agree}/{len(by)} challenges; imagined best infeasible in Chrono on {n_inf}; "
          f"Chrono cost of imagined best / best feasible: mean {np.mean(reg):.2f} max {np.max(reg):.2f}" if reg else "")


if __name__ == "__main__":
    main()
