#!/usr/bin/env python
"""Planner-S evaluation: sample thousands of routes, imagine them all, pick the best -- versus A*.

All-sensor setting on the held-out layouts: camera map -> obstacle cells + goal, camera start pose;
routes sampled by ``planner_s.sample_routes``; every survivor and every A* candidate is tracked by the
PPO tracker inside the NRD from the same start context and scored with hard rejects (not completed,
roll/pitch/cross-track failure, footprint through a predicted cell) on

    pessimistic cost = imagined time + max(power-head energy, throttle-model energy) / 10

(the objective that survived Chrono in round 2; the plain throttle-model energy gets exploited by the
sampler). Optional cross-entropy rounds (``--cem-rounds``) resample around the best imagined routes so
the batched imagination budget concentrates where the objective is already good; optional extra
dynamics checkpoints (``--dynamics-checkpoints``) make the energy pessimism an ensemble maximum; an
optional clearance penalty adds a safety margin to the sampled routes (the A* routes carry a 0.9 m one).
Picks per layout are exported as a route file for ``traverse_wp3_chrono_eval.py --route-file``.
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
from nedm.traverse.planner_s import resample_routes, sample_routes
from nedm.traverse.power_calib import KINDS, PowerModel
from nedm.traverse.terrain import TerrainMap
from nedm.traverse.tracker_env import TraverseTrackingEnv, merge_env_cfg
from traverse_wp4_score_candidates import CANDIDATE_SWEEP, build_candidates, load_policy, rollout, route_dict

RES_KEYS = ("time_s", "completed", "failed", "collided", "collided_true", "min_clearance_m", "min_clearance_true_m",
            "energy_kj", "energy_act_kj", "energy_state_kj")


class Imaginer:
    """Batched imagined rollouts of route candidates on one layout, over one or more dynamics models."""

    def __init__(self, args, power_models):
        self.args, self.power_models = args, power_models
        self.horizon = int(round(args.horizon_s / DT_S))
        self.sidecars = args.z1_extra_cache or [""]
        if len(self.sidecars) == 1:
            self.sidecars = self.sidecars * len(args.dynamics_checkpoints)

    def __call__(self, key: str, plans, start, discs, layout) -> dict[str, np.ndarray]:
        a, dev = self.args, self.args.device
        entries = [(key, route_dict(p)) for p in plans]
        per_model = []
        for ckpt, sidecar in zip(a.dynamics_checkpoints, self.sidecars):
            cfg = merge_env_cfg({"num_envs": len(entries), "device": dev, "auto_reset": False, "split": "val",
                                 "dynamics_checkpoint": ckpt, "arena": a.arena, "cache": a.cache, "routes": a.routes,
                                 "fragment_steps_max": self.horizon, "z1_extra_cache": sidecar or None})
            env = TraverseTrackingEnv(cfg, device=dev, entries=entries)
            policy = load_policy(Path(a.policy), env, dev)
            obst = torch.tensor(np.asarray(discs, np.float32), device=dev)[None].expand(len(entries), -1, -1).contiguous()
            obst_true = torch.tensor(np.asarray(layout.obstacles(), np.float32), device=dev)[None].expand(len(entries), -1, -1).contiguous()
            sp = env.pose.clone(); sp[:] = torch.tensor(start, device=dev, dtype=sp.dtype)
            res = rollout(env, policy, self.horizon, obst, obst_true, self.power_models, sp)
            per_model.append({k: v.cpu().numpy() for k, v in res.items() if not k.startswith("_")})
            del env, policy
        torch.cuda.empty_cache()
        out = {k: per_model[0][k] for k in RES_KEYS if k in per_model[0]}  # time, flags, clearances: primary model
        # pessimistic energy = max over the chosen estimators and over all models
        names = {"head": "energy_kj", "act": "energy_act_kj", "state": "energy_state_kj"}
        terms = [r[names[t]] for r in per_model for t in a.pess_terms if names[t] in r]
        out["energy_pess"] = np.max(terms, axis=0)
        out["ok"] = np.all([r["completed"] & ~r["failed"] & ~r["collided"] for r in per_model], axis=0)
        return out


def concat(a: dict, b: dict) -> dict:
    return {k: np.concatenate([a[k], b[k]]) for k in a if k in b}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="artifacts/traverse/wp3_tracker_v1")
    ap.add_argument("--dynamics-checkpoints", nargs="+", default=["artifacts/traverse/wp2_mapv2_dagger2_ro8_amd/ckpt_best.pt"],
                    help="first = primary (time, flags); all of them enter the pessimistic energy maximum")
    ap.add_argument("--z1-extra-cache", nargs="*", default=None, help="sidecar per checkpoint ('' for none)")
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
    ap.add_argument("--prefilter-margin", type=float, default=0.2, help="footprint slack vs camera cells for sampled routes")
    ap.add_argument("--occ-threshold", type=float, default=0.85)
    ap.add_argument("--horizon-s", type=float, default=20.0)
    ap.add_argument("--cem-rounds", type=int, default=0)
    ap.add_argument("--cem-elites", type=int, default=16)
    ap.add_argument("--cem-children", type=int, default=24, help="children per elite and round")
    ap.add_argument("--cem-sigma-m", type=float, default=1.5, help="control-point perturbation (m) in round 1")
    ap.add_argument("--cem-sigma-v", type=float, default=0.6, help="cruise-speed perturbation (m/s) in round 1")
    ap.add_argument("--cem-shrink", type=float, default=0.6, help="per-round multiplier on both sigmas")
    ap.add_argument("--clearance-target", type=float, default=0.5, help="m of predicted clearance below which the penalty applies")
    ap.add_argument("--clearance-penalty", type=float, default=4.0, help="s of cost per metre short of the target")
    ap.add_argument("--pess-terms", nargs="+", default=["head", "act"], choices=["head", "act", "state"],
                    help="estimators entering the pessimistic maximum ('state' = engine speed x torque from a 17-D model)")
    ap.add_argument("--no-astar", action="store_true")
    ap.add_argument("--pick-prefix", default="", help="prefix for the exported pick names (e.g. 'ens_' for an ensemble run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache, routes = Path(args.cache), Path(args.routes)
    keys = D.load_cache_keys(cache)
    val = D.split_keys(keys)[1]
    manifest = json.loads((routes / "routes_manifest.json").read_text())
    keys = [k for k in val if k in set(manifest["families"]["oracle"])][: args.episodes]
    tmap = TerrainMap.from_dir(Path(args.arena))
    decoder = MapDecoder(Path(args.maphead), Path(args.arena), args.device)
    start_est = json.loads(Path(args.start_poses).read_text())
    power_models = {k: PowerModel.load(Path(args.power_calib), k) for k in KINDS}
    imagine = Imaginer(args, power_models)
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
        sampled, stats = sample_routes((sx, sy), syaw, goal, args.samples, rng, tmap, discs,
                                       prefilter_margin_m=args.prefilter_margin, max_routes=args.max_imagined)
        t_sample = time.perf_counter() - t1
        astar = [] if args.no_astar else build_candidates(meta, tmap, CANDIDATE_SWEEP, obstacles=discs, repair_iterations=40,
                                                          margin_fallback=True, goal=goal, start=(sx, sy))[0]
        t_astar = time.perf_counter() - t1 - t_sample
        plans = list(sampled) + [p for _, p in astar]
        tags = np.array(["S"] * len(sampled) + ["A"] * len(astar))
        rounds = np.zeros(len(plans), int)
        if not plans:
            results.append({"key": key, "sampled_ok": 0, "astar": 0}); continue
        t2 = time.perf_counter()
        res = imagine(key, plans, (sx, sy, syaw), discs, layout)
        t_imag = time.perf_counter() - t2
        pess = lambda r: np.where(r["ok"], r["time_s"] + r["energy_pess"] / 10.0, np.inf)
        row = {"key": key, "pick_prefix": args.pick_prefix, **{f"n_{k}": int(v) for k, v in stats.items()}, "sampled_imagined": int(len(sampled)),
               "sampled_ok": int(res["ok"][tags == "S"].sum()), "astar": int(len(astar)), "astar_ok": int(res["ok"][tags == "A"].sum()),
               "t_sample_s": t_sample, "t_astar_s": t_astar, "t_imagine_s": t_imag, "round_best_pess": [], "round_imagined": []}
        c0 = pess(res)[tags == "S"]
        row["round_best_pess"].append(float(c0.min()) if np.isfinite(c0).any() else None); row["round_imagined"].append(int(len(sampled)))
        # --- cross-entropy rounds around the best sampled routes -----------------------------------------
        sig_m, sig_v, t_cem = args.cem_sigma_m, args.cem_sigma_v, 0.0
        for r in range(1, args.cem_rounds + 1):
            c = pess(res); c[tags != "S"] = np.inf
            elite = np.argsort(c)[: args.cem_elites]
            elite = elite[np.isfinite(c[elite])]
            if len(elite) == 0:
                break
            tc = time.perf_counter()
            children, cst = resample_routes([plans[i] for i in elite], args.cem_children, rng, goal, tmap, discs,
                                            sigma_m=sig_m, sigma_v=sig_v, prefilter_margin_m=args.prefilter_margin, tag=f"cem{r}")
            if not children:
                break
            res_c = imagine(key, children, (sx, sy, syaw), discs, layout)
            plans += children; tags = np.concatenate([tags, ["S"] * len(children)]); rounds = np.concatenate([rounds, [r] * len(children)])
            res = concat(res, res_c)
            t_cem += time.perf_counter() - tc
            cc = pess(res_c)
            row["round_best_pess"].append(float(cc.min()) if np.isfinite(cc).any() else None); row["round_imagined"].append(int(len(children)))
            sig_m *= args.cem_shrink; sig_v *= args.cem_shrink
        row["t_cem_s"] = t_cem; row["t_total_s"] = time.perf_counter() - t0
        cost_pess = pess(res)
        penalty = args.clearance_penalty * np.maximum(0.0, args.clearance_target - res["min_clearance_m"])
        cost_clear = np.where(res["ok"], cost_pess + penalty, np.inf)
        np.savez_compressed(out / f"cands_{ep}.npz", tag=tags, round=rounds, ok=res["ok"], time_s=res["time_s"], energy_kj=res["energy_kj"],
                            energy_act_kj=res["energy_act_kj"], energy_pess=res["energy_pess"], collided_true=res["collided_true"],
                            clear=res["min_clearance_m"], clear_true=res["min_clearance_true_m"],
                            length=np.array([p.length_m for p in plans]), v_cruise=np.array([float(p.meta.get("v_cruise", p.speeds.max())) for p in plans]))
        picks = {"astar_best": (tags == "A", cost_pess),
                 "sampled_pess": ((tags == "S") & (rounds == 0), cost_pess),
                 "cem_pess": (tags == "S", cost_pess),
                 "cem_pess_clear": (tags == "S", cost_clear)}
        export[key] = []
        for name, (mask, cost) in picks.items():
            idx = np.nonzero(mask)[0]
            if len(idx) == 0 or not np.isfinite(cost[idx]).any():
                row[f"{name}_cost"] = None; continue
            b = idx[np.argmin(cost[idx])]
            row[f"{name}_cost"] = float(cost_pess[b]); row[f"{name}_time"] = float(res["time_s"][b]); row[f"{name}_energy"] = float(res["energy_pess"][b])
            row[f"{name}_energy_head"] = float(res["energy_kj"][b]); row[f"{name}_energy_act"] = float(res["energy_act_kj"][b])
            row[f"{name}_clear"] = float(res["min_clearance_m"][b]); row[f"{name}_clear_true"] = float(res["min_clearance_true_m"][b])
            row[f"{name}_collided_true"] = bool(res["collided_true"][b]); row[f"{name}_round"] = int(rounds[b])
            row[f"{name}_length"] = float(plans[b].length_m); row[f"{name}_vcruise"] = float(plans[b].meta.get("v_cruise", plans[b].speeds.max()))
            export[key].append({"candidate": args.pick_prefix + name, **{k: np.asarray(v).tolist() for k, v in route_dict(plans[b]).items()}})
        s_ok = (tags == "S") & res["ok"]
        row["sampled_ok_true_collision_rate"] = float(res["collided_true"][s_ok].mean()) if s_ok.any() else None
        results.append(row)
        rb = " -> ".join("-" if v is None else f"{v:.2f}" for v in row["round_best_pess"])
        print(f"[{n_ep + 1}/{len(keys)}] {ep}: sampled {stats['sampled']} -> clear {stats['clear']} -> ok {row['sampled_ok']} | A* {row['astar']} ok {row['astar_ok']} | "
              f"pess best per round {rb} | A* {row.get('astar_best_cost')} | t sample {t_sample:.1f} A* {t_astar:.1f} imagine {t_imag:.1f} cem {t_cem:.1f} s", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "routes.json").write_text(json.dumps(export))
    names = ["astar_best", "sampled_pess", "cem_pess", "cem_pess_clear"]
    both = [r for r in results if all(r.get(f"{n}_cost") is not None for n in names)]
    print(f"\n{len(both)} layouts with all picks (imagined, pessimistic cost = time + E/10):")
    for n in names:
        print(f"  {n:15s} cost {np.mean([r[f'{n}_cost'] for r in both]):6.2f}  time {np.mean([r[f'{n}_time'] for r in both]):5.2f}  "
              f"E_pess {np.mean([r[f'{n}_energy'] for r in both]):6.1f}  clear(pred) {np.mean([r[f'{n}_clear'] for r in both]):.2f}  "
              f"clear(true) {np.mean([r[f'{n}_clear_true'] for r in both]):.2f}  true-collisions {sum(r[f'{n}_collided_true'] for r in both)}  "
              f"beats A* {sum(r[f'{n}_cost'] < r['astar_best_cost'] for r in both)}/{len(both)}")
    print(f"mean wall per layout {np.mean([r['t_total_s'] for r in results if 't_total_s' in r]):.1f} s")


if __name__ == "__main__":
    main()
