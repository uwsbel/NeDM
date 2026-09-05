#!/usr/bin/env python
"""Planner-S evaluation: sample thousands of routes, imagine them all, pick the best -- versus A*.

All-sensor setting on the held-out layouts: camera map -> obstacle cells + goal, camera start pose;
routes sampled by ``planner_s.sample_routes``; every survivor and every A* candidate is tracked by the
PPO tracker inside the NRD from the same start context, scored on imagined time + energy/10 with hard
rejects (not completed, roll/pitch/cross-track failure, footprint through a predicted cell). The two
picks per layout are exported for Chrono.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nedm.traverse import nrd_data as D
from nedm.traverse.layout import EpisodeLayout
from nedm.traverse.nrd_model import DT_S
from nedm.traverse.planner_b import MapDecoder, goal_from_map, occupancy_discs
from nedm.traverse.planner_s import sample_routes
from nedm.traverse.power_calib import KINDS, PowerModel
from nedm.traverse.terrain import TerrainMap
from nedm.traverse.tracker_env import TraverseTrackingEnv, merge_env_cfg
from traverse_wp4_score_candidates import CANDIDATE_SWEEP, build_candidates, load_policy, rollout, route_dict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="artifacts/traverse/wp3_tracker_v1")
    ap.add_argument("--dynamics-checkpoint", default="artifacts/traverse/wp2_mapv2_dagger2_ro8_amd/ckpt_best.pt")
    ap.add_argument("--maphead", default="artifacts/traverse/wp4_maphead_v2/ckpt_best.pt")
    ap.add_argument("--start-poses", default="artifacts/traverse/wp4_start_poses/val_start_poses.json")
    ap.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6")
    ap.add_argument("--routes", default="artifacts/traverse/wp3_routes")
    ap.add_argument("--stores", default="artifacts/traverse")
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--power-calib", default="artifacts/traverse/wp4_power_calib/power_calib.json")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--max-imagined", type=int, default=2000)
    ap.add_argument("--occ-threshold", type=float, default=0.85)
    ap.add_argument("--horizon-s", type=float, default=20.0)
    ap.add_argument("--energy-field", default="energy_act_kj")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    cache, routes = Path(args.cache), Path(args.routes)
    keys = D.load_cache_keys(cache)
    val = D.split_keys(keys)[1]
    manifest = json.loads((routes / "routes_manifest.json").read_text())
    keys = [k for k in val if k in set(manifest["families"]["oracle"])][: args.episodes]
    tmap = TerrainMap.from_dir(Path(args.arena))
    decoder = MapDecoder(Path(args.maphead), Path(args.arena), dev)
    start_est = json.loads(Path(args.start_poses).read_text())
    power_models = {k: PowerModel.load(Path(args.power_calib), k) for k in KINDS}
    horizon = int(round(args.horizon_s / DT_S))
    rng = np.random.default_rng(args.seed)
    results, export = [], {}
    for n_ep, key in enumerate(keys):
        t0 = time.perf_counter()
        store, ep = key.split("__", 1)
        meta = json.loads((Path(args.stores) / store / ep / "meta.json").read_text())
        layout = EpisodeLayout.from_json(meta["layout"])
        with np.load(cache / f"{key}.npz") as d:
            occ, _ = decoder(d["map_v2"])
        discs = occupancy_discs(occ, decoder.size_m, args.occ_threshold, mode="cells")
        goal = goal_from_map(occ, decoder.size_m, args.occ_threshold)
        sx, sy, syaw = start_est[key]["est"]
        t1 = time.perf_counter()
        sampled, stats = sample_routes((sx, sy), syaw, goal, args.samples, rng, tmap, discs, max_routes=args.max_imagined)
        t_sample = time.perf_counter() - t1
        astar, _ = build_candidates(meta, tmap, CANDIDATE_SWEEP, obstacles=discs, repair_iterations=40, margin_fallback=True,
                                    goal=goal, start=(sx, sy))
        t_astar = time.perf_counter() - t1 - t_sample
        plans = [("S", p) for p in sampled] + [("A", p) for _, p in astar]
        if not plans:
            results.append({"key": key, "sampled_ok": 0, "astar": 0}); continue
        entries = [(key, route_dict(p)) for _, p in plans]
        t2 = time.perf_counter()
        cfg = merge_env_cfg({"num_envs": len(entries), "device": dev, "auto_reset": False, "split": "val",
                             "dynamics_checkpoint": args.dynamics_checkpoint, "arena": args.arena, "cache": args.cache,
                             "routes": args.routes, "fragment_steps_max": horizon})
        env = TraverseTrackingEnv(cfg, device=dev, entries=entries)
        policy = load_policy(Path(args.policy), env, dev)
        obst = torch.tensor(np.asarray(discs, np.float32), device=dev)[None].expand(len(entries), -1, -1).contiguous()
        true_d = np.asarray(layout.obstacles(), np.float32)
        obst_true = torch.tensor(true_d, device=dev)[None].expand(len(entries), -1, -1).contiguous()
        sp = env.pose.clone(); sp[:] = torch.tensor([sx, sy, syaw], device=dev, dtype=sp.dtype)
        res = rollout(env, policy, horizon, obst, obst_true, power_models, sp)
        res = {k: v.cpu().numpy() for k, v in res.items() if not k.startswith("_")}
        t_imag = time.perf_counter() - t2
        ok = res["completed"] & ~res["failed"] & ~res["collided"]
        cost = res["time_s"] + res[args.energy_field] / 10.0
        cost = np.where(ok, cost, np.inf)
        tags = np.array([t for t, _ in plans])
        np.savez_compressed(out / f"cands_{ep}.npz", tag=tags, time_s=res["time_s"], energy_kj=res["energy_kj"],
                            energy_act_kj=res["energy_act_kj"], ok=ok, collided_true=res["collided_true"],
                            clear_true=res["min_clearance_true_m"], length=np.array([pl.length_m for _, pl in plans]),
                            v_cruise=np.array([float(pl.meta.get("v_cruise", pl.speeds.max())) for _, pl in plans]))
        # alternative objectives for the sampled pick: pessimistic energy (max of the two estimates), time only
        e_pess = np.maximum(res["energy_kj"], res["energy_act_kj"])
        alt_costs = {"sampled_pess": np.where(ok, res["time_s"] + e_pess / 10.0, np.inf),
                     "sampled_time": np.where(ok, res["time_s"], np.inf)}
        row = {"key": key, **{f"n_{k}": int(v) for k, v in stats.items()}, "sampled_imagined": int((tags == "S").sum()),
               "sampled_ok": int(ok[tags == "S"].sum()), "astar": int((tags == "A").sum()), "astar_ok": int(ok[tags == "A"].sum()),
               "t_sample_s": t_sample, "t_astar_s": t_astar, "t_imagine_s": t_imag, "t_total_s": time.perf_counter() - t0}
        export[key] = []
        for tag, name in (("S", "sampled_best"), ("A", "astar_best")):
            idx = np.nonzero(tags == tag)[0]
            if len(idx) == 0 or not np.isfinite(cost[idx]).any():
                row[f"{name}_cost"] = None; continue
            b = idx[np.argmin(cost[idx])]
            row[f"{name}_cost"] = float(cost[b]); row[f"{name}_time"] = float(res["time_s"][b]); row[f"{name}_energy"] = float(res[args.energy_field][b])
            row[f"{name}_collided_true"] = bool(res["collided_true"][b]); row[f"{name}_clear_true"] = float(res["min_clearance_true_m"][b])
            row[f"{name}_length"] = float(plans[b][1].length_m); row[f"{name}_vcruise"] = float(plans[b][1].meta.get("v_cruise", plans[b][1].speeds.max()))
            export[key].append({"candidate": name, **{k: np.asarray(v).tolist() for k, v in route_dict(plans[b][1]).items()}})
        for name, c in alt_costs.items():
            idx = np.nonzero(tags == "S")[0]
            if len(idx) and np.isfinite(c[idx]).any():
                b = idx[np.argmin(c[idx])]
                row[f"{name}_time"] = float(res["time_s"][b]); row[f"{name}_energy"] = float(res[args.energy_field][b])
                row[f"{name}_energy_head"] = float(res["energy_kj"][b])
                export[key].append({"candidate": name, **{k: np.asarray(v).tolist() for k, v in route_dict(plans[b][1]).items()}})
        # sampled-route statistics in imagination: how many true-map collisions among imagined-OK samples?
        s_ok = (tags == "S") & ok
        row["sampled_ok_true_collision_rate"] = float(res["collided_true"][s_ok].mean()) if s_ok.any() else None
        results.append(row)
        print(f"[{n_ep + 1}/{len(keys)}] {ep}: sampled {stats['sampled']} -> clear {stats['clear']} -> imagined-ok {row['sampled_ok']} | "
              f"A* {row['astar']} ok {row['astar_ok']} | best cost S {row.get('sampled_best_cost')} A {row.get('astar_best_cost')} | "
              f"t sample {t_sample:.1f}s A* {t_astar:.1f}s imagine {t_imag:.1f}s", flush=True)
        del env, policy; torch.cuda.empty_cache()
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "routes.json").write_text(json.dumps(export))
    both = [r for r in results if r.get("sampled_best_cost") is not None and r.get("astar_best_cost") is not None]
    print(f"\n{len(both)} layouts with both picks: sampled better on {sum(r['sampled_best_cost'] < r['astar_best_cost'] for r in both)}; "
          f"mean cost S {np.mean([r['sampled_best_cost'] for r in both]):.2f} vs A {np.mean([r['astar_best_cost'] for r in both]):.2f}; "
          f"time S {np.mean([r['sampled_best_time'] for r in both]):.2f} vs A {np.mean([r['astar_best_time'] for r in both]):.2f}; "
          f"energy S {np.mean([r['sampled_best_energy'] for r in both]):.1f} vs A {np.mean([r['astar_best_energy'] for r in both]):.1f}; "
          f"true-map collision of picks S {sum(r['sampled_best_collided_true'] for r in both)} A {sum(r['astar_best_collided_true'] for r in both)}; "
          f"mean wall per layout {np.mean([r['t_total_s'] for r in results]):.1f} s")


if __name__ == "__main__":
    main()
