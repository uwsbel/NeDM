#!/usr/bin/env python
"""Does a dynamics model predict the recorded outcomes on held-out arenas? Imagine every recorded run of the
schema-v2 cache on the chosen arenas -- tracker in the model, from rest at the camera start pose, on the layout's
single-frame scene map (the planner's exact imagination) -- and compare route by route with what Chrono did:
rejection of the infeasible runs (AUC), time / energy on the feasible ones, and the pick among each layout's
routes (regret against Chrono's best feasible). Same rows for the frozen arena_v1 model and the fine-tuned one,
same routes as the cheap predictor's val predictions -> the matched comparison of plan §28.

  PYTHONPATH=src python scripts/traverse_wp7_imagine_cache.py --cache artifacts/traverse/wp7_cache_v1 --arenas arena_f105 \
      --dynamics-checkpoints artifacts/traverse/wp2_mapv2_pt_dag_ro8_amd/ckpt_best.pt --out artifacts/traverse/wp7_imagine_f105_frozen
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nedm.traverse.layout import EpisodeLayout
from nedm.traverse.oracle import PlanCandidate
from nedm.traverse.power_calib import KINDS, PowerModel
from nedm.traverse.terrain import TerrainMap
from traverse_wp5_sample_planner import FZ0, FZ1, PITCH, ROLL, Imaginer
from traverse_wp6_imagine_sweep import auc


def imagine_arena_batched(a, cache: Path, groups: list[tuple[str, list[str], EpisodeLayout, tuple]], labels: dict,
                          power_models: dict) -> dict[str, dict[str, np.ndarray]]:
    """All (layout, route) pairs of one arena in ONE imagination env per dynamics model (the per-layout ``Imaginer``
    reloads model + policy for every layout). Returns key -> {ok, time_s, energy_pess, energy_kj, completed, failed,
    pred_max_roll_deg, pred_max_pitch_deg, pred_min_fz_n}; same computation as ``Imaginer.__call__``."""
    import torch
    from nedm.traverse.tracker_env import TraverseTrackingEnv, merge_env_cfg
    from traverse_wp4_score_candidates import load_policy, rollout, route_dict
    dev = a.device
    entries, starts, discs, keys = [], [], [], []
    for lay, ks, layout, start in groups:
        for k in ks:
            with np.load(cache / f"{k}.npz") as z:
                plan = PlanCandidate(waypoints=z["route_waypoints"].astype(float), speeds=z["route_speeds"].astype(float),
                                     headings=z["route_headings"].astype(float), stations=z["route_stations"].astype(float), meta={"candidate": labels[k]["candidate"]})
            entries.append((ks[0], route_dict(plan))); starts.append(start); discs.append(np.asarray(layout.obstacles(), np.float32).reshape(-1, 3)); keys.append(k)
    n = len(entries); m = max(len(d) for d in discs)
    obst = np.full((n, m, 3), -1.0, np.float32)
    for i, d in enumerate(discs):
        obst[i, : len(d)] = d
    obst_t = torch.tensor(obst, device=dev)
    horizon = int(round(a.horizon_s / 0.05))
    per_model, limits = [], None
    for ckpt in a.dynamics_checkpoints:
        cfg = merge_env_cfg({"num_envs": n, "device": dev, "auto_reset": False, "split": "val", "dynamics_checkpoint": ckpt, "arena": a.arena,
                             "cache": str(cache), "routes": a.routes, "fragment_steps_max": horizon, "z1_extra_cache": None, "map_key": a.map_key,
                             "termination": {"max_abs_roll_rad": np.radians(a.roll_limit_deg), "max_abs_pitch_rad": np.radians(a.pitch_limit_deg)}})
        env = TraverseTrackingEnv(cfg, device=dev, entries=entries)
        policy = load_policy(Path(a.policy), env, dev)
        sp = torch.tensor(np.asarray(starts, np.float32), device=dev)
        res = rollout(env, policy, horizon, obst_t, obst_t, power_models, sp, rest_start=True)
        if limits is None:
            z1, active = res["_traj"]["z1"], res["_traj"]["active"]
            mm = np.where(active[..., None], z1, np.nan)
            with np.errstate(all="ignore"):
                limits = {"pred_max_roll_deg": np.degrees(np.nanmax(np.abs(mm[..., ROLL]), axis=0)), "pred_max_pitch_deg": np.degrees(np.nanmax(np.abs(mm[..., PITCH]), axis=0)),
                          "pred_min_fz_n": np.nanmin(mm[..., FZ0:FZ1], axis=(0, 2)) if z1.shape[-1] >= FZ1 else np.full(n, np.nan)}
        per_model.append({k: v.cpu().numpy() for k, v in res.items() if not k.startswith("_")})
        del env, policy
        torch.cuda.empty_cache()
    names = {"head": "energy_kj", "act": "energy_act_kj", "state": "energy_state_kj"}
    terms = [r[names[t]] for r in per_model for t in a.pess_terms if names[t] in r]
    out = {"time_s": per_model[0]["time_s"], "completed": per_model[0]["completed"], "failed": per_model[0]["failed"], "energy_kj": per_model[0]["energy_kj"],
           "energy_pess": np.max(terms, axis=0), "ok": np.all([r["completed"] & ~r["failed"] & ~r["collided"] for r in per_model], axis=0), **limits}
    return {k: {f: out[f][i] for f in out} for i, k in enumerate(keys)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--arenas", nargs="+", required=True, help="arena ids to imagine (val / sealed test)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="artifacts/traverse/wp3_tracker_v1")
    ap.add_argument("--dynamics-checkpoints", nargs="+", required=True)
    ap.add_argument("--power-calib", default="artifacts/traverse/wp4_power_calib/power_calib.json")
    ap.add_argument("--energy-floor", default="", help="geometry floor json (default: none -- judge the model alone)")
    ap.add_argument("--floor-sigmas", type=float, default=1.5)
    ap.add_argument("--pess-terms", nargs="+", default=["head", "state"])
    ap.add_argument("--horizon-s", type=float, default=30.0)
    ap.add_argument("--max-layouts", type=int, default=0)
    ap.add_argument("--per-layout", action="store_true", help="one imagination env per layout (slow; default: all routes of an arena in one env)")
    ap.add_argument("--roll-limit-deg", type=float, default=np.degrees(0.6), help="imagination attitude termination (tracker-env default 0.6 rad); Chrono aborts at 60 deg")
    ap.add_argument("--pitch-limit-deg", type=float, default=np.degrees(0.4))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cache = Path(args.cache)
    manifest = json.loads((cache / "cache_manifest.json").read_text())
    labels = json.loads((cache / "labels.json").read_text())
    start_est = json.loads((cache / "start_poses.json").read_text())
    power_models = {k: PowerModel.load(Path(args.power_calib), k) for k in KINDS}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for aid in args.arenas:
        arena_dir = Path(manifest["arenas"][aid])
        tmap = TerrainMap.from_dir(arena_dir)
        a = SimpleNamespace(**{**vars(args), "routes": "artifacts/traverse/wp3_routes", "map_key": "map_v2", "from_rest": True, "cache": str(cache),
                               "arena": str(arena_dir), "z1_extra_cache": None, "energy_floor": args.energy_floor or None})
        imagine = Imaginer(a, power_models, tmap, cache)
        by_layout = defaultdict(list)
        for k in manifest["episodes"]:
            if manifest["arena_of"][k] == aid:
                by_layout[labels[k]["layout"]].append(k)
        layouts = sorted(by_layout)
        if args.max_layouts:
            layouts = layouts[: args.max_layouts]
        groups = []
        for lay in layouts:
            with np.load(cache / f"{by_layout[lay][0]}.npz") as z:
                groups.append((lay, by_layout[lay], EpisodeLayout.from_json(json.loads(str(z["layout_json"]))), tuple(start_est[lay]["est"])))
        batched = {} if args.per_layout else imagine_arena_batched(a, cache, groups, labels, power_models)
        for lay, keys, layout, start in groups:
            if args.per_layout:
                plans = []
                for k in keys:
                    with np.load(cache / f"{k}.npz") as z:
                        plans.append(PlanCandidate(waypoints=z["route_waypoints"].astype(float), speeds=z["route_speeds"].astype(float),
                                                   headings=z["route_headings"].astype(float), stations=z["route_stations"].astype(float), meta={"candidate": labels[k]["candidate"]}))
                r_ = imagine(keys[0], plans, start, layout.obstacles(), layout)
                res = {k: {f: r_[f][i] for f in r_} for i, k in enumerate(keys)}
            else:
                res = {k: batched[k] for k in keys}
            for k in keys:
                c, r_ = labels[k], res[k]
                feasible = bool(c.get("completed")) and not bool(c.get("stalled")) and not bool(c.get("contact"))
                rows.append({"key": k, "arena": aid, "layout": lay, "kind": c.get("kind"), "candidate": c["candidate"], "chrono_feasible": feasible, "chrono_status": c["status"],
                             "chrono_time": c["time_s"], "chrono_energy": c["energy_kj"], "chrono_max_pitch": c.get("max_pitch_deg"), "chrono_max_roll": c.get("max_roll_deg"),
                             "img_ok": bool(r_["ok"]), "img_completed": bool(r_["completed"]), "img_failed": bool(r_["failed"]),
                             "img_time": float(r_["time_s"]), "img_energy": float(r_["energy_pess"]), "img_energy_head": float(r_["energy_kj"]),
                             "img_max_pitch": float(r_["pred_max_pitch_deg"]), "img_max_roll": float(r_["pred_max_roll_deg"]), "img_min_fz": float(r_["pred_min_fz_n"])})
            n_ok = sum(bool(res[k]["ok"]) for k in keys); n_f = sum(rows[-len(keys) + i]["chrono_feasible"] for i in range(len(keys)))
            print(f"{lay:44s} imagined ok {n_ok:2d}/{len(keys):2d}  Chrono feasible {n_f:2d}/{len(keys):2d}", flush=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    feas = np.array([r["chrono_feasible"] for r in rows]); ok = np.array([r["img_ok"] for r in rows])
    summary = {"n_routes": len(rows), "chrono_infeasible": int((~feas).sum()), "img_rejects": int((~ok).sum()),
               "rejected_infeasible": int((~ok & ~feas).sum()), "rejected_feasible": int((~ok & feas).sum()),
               "accepted_infeasible": int((ok & ~feas).sum()), "accepted_feasible": int((ok & feas).sum()),
               "auc_reject": auc(~ok, ~feas), "auc_img_time": auc([r["img_time"] for r in rows], ~feas),
               "auc_img_pitch": auc([r["img_max_pitch"] for r in rows], ~feas), "auc_img_energy": auc([r["img_energy"] for r in rows], ~feas),
               "auc_img_roll": auc([r["img_max_roll"] for r in rows], ~feas)}
    print(f"\n{len(rows)} routes: Chrono infeasible {summary['chrono_infeasible']}, imagination rejects {summary['img_rejects']} "
          f"(rightly {summary['rejected_infeasible']}, wrongly {summary['rejected_feasible']}); accepted & infeasible {summary['accepted_infeasible']}")
    print(f"AUC for infeasibility: reject {summary['auc_reject']:.2f} | imagined time {summary['auc_img_time']:.2f} | pitch {summary['auc_img_pitch']:.2f} | roll {summary['auc_img_roll']:.2f} | energy {summary['auc_img_energy']:.2f}")
    both = [r for r in rows if r["chrono_feasible"] and r["img_ok"]]
    if both:
        ct, it = np.array([r["chrono_time"] for r in both]), np.array([r["img_time"] for r in both])
        ce, ie = np.array([r["chrono_energy"] for r in both]), np.array([r["img_energy"] for r in both])
        summary.update(n_both=len(both), time_ratio=float(ct.mean() / it.mean()), time_mae_s=float(np.abs(ct - it).mean()), time_corr=float(np.corrcoef(ct, it)[0, 1]),
                       energy_ratio=float(ce.mean() / ie.mean()), energy_mape=float(np.mean(np.abs(ce - ie) / ce)), energy_corr=float(np.corrcoef(ce, ie)[0, 1]))
        print(f"on the {len(both)} routes both call feasible: time Chrono/imagined {summary['time_ratio']:.2f} (MAE {summary['time_mae_s']:.2f} s, corr {summary['time_corr']:.2f}); "
              f"energy Chrono/imagined {summary['energy_ratio']:.2f} (MAE {100 * summary['energy_mape']:.0f} %, corr {summary['energy_corr']:.2f})")
    by_kind = {}
    for kind in sorted(set(r["kind"] for r in rows)):
        m = np.array([r["kind"] == kind for r in rows])
        by_kind[kind] = {"n": int(m.sum()), "infeasible": int((~feas[m]).sum()), "auc_reject": auc(~ok[m], ~feas[m]), "auc_img_time": auc(np.array([r["img_time"] for r in rows])[m], ~feas[m]),
                         "accepted_infeasible": int((ok[m] & ~feas[m]).sum()), "rejected_feasible": int((~ok[m] & feas[m]).sum())}
    summary["by_kind"] = by_kind
    # pick among each layout's routes: imagined best (cost = time + energy / 10 over the accepted) vs Chrono's best feasible
    by = defaultdict(list)
    for r in rows:
        by[r["layout"]].append(r)
    agree, regret, n_inf, n_none = 0, [], 0, 0
    for lay, rs in by.items():
        f = [r for r in rs if r["chrono_feasible"]]
        if not f:
            continue
        cb = min(f, key=lambda r: r["chrono_time"] + r["chrono_energy"] / 10)
        ib = [r for r in rs if r["img_ok"]]
        if not ib:
            n_none += 1; continue
        ib = min(ib, key=lambda r: r["img_time"] + r["img_energy"] / 10)
        agree += ib["candidate"] == cb["candidate"]
        if ib["chrono_feasible"]:
            regret.append((ib["chrono_time"] + ib["chrono_energy"] / 10) / (cb["chrono_time"] + cb["chrono_energy"] / 10))
        else:
            n_inf += 1
    n_lay = sum(1 for rs in by.values() if any(r["chrono_feasible"] for r in rs))
    summary["pick"] = {"layouts_with_feasible": n_lay, "agree_with_chrono_best": agree, "pick_infeasible": n_inf, "no_pick": n_none,
                       "regret_mean": float(np.mean(regret)) if regret else None, "regret_max": float(np.max(regret)) if regret else None}
    print(f"pick among each layout's routes: best-feasible agreement {agree}/{n_lay}; imagined pick infeasible in Chrono {n_inf}; no accepted route {n_none}; "
          f"regret mean {np.mean(regret):.2f} max {np.max(regret):.2f}" if regret else "no regret rows")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
