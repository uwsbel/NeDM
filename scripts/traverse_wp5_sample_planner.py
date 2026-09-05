#!/usr/bin/env python
"""Planner-S evaluation: sample thousands of routes, imagine them all, pick the best -- versus A*.

Live-input setting on held-out layouts: camera map -> obstacle cells + goal, camera start pose, rest
context (``--from-rest``). Three planners share the same map, start, goal, terrain source and tracker:

* **plain A*** (``astar_plain``): the classical baseline -- one A* search with the default parameters
  (slope caps, 0.9 m tracker margin with the 0.6 / 0.3 m fallback ladder), rule-based speed profile,
  no dynamics, no imagination. Exported whatever the imagination thinks of it.
* **A* sweep + world model** (``astar_best``): six A* parameter variants, the tracker rolled out inside
  the NRD on each, best pessimistic cost.
* **sampling + world model** (``sampled_pess`` round 0, ``cem_pess`` after the CEM rounds,
  ``cem_pess_clear`` with the clearance penalty): thousands of smooth sampled routes, every survivor
  tracked by the PPO tracker inside the NRD from the same start context and scored with hard rejects
  (not completed, roll/pitch/cross-track failure, footprint through a predicted cell) on

      pessimistic cost = imagined time + max(chosen energy estimators, geometry floor) / 10

``--scorer geometry`` replaces the world model by a cheap route-geometry scorer (profile-implied time +
the Chrono-fitted geometry energy regression) over the SAME candidate family and CEM procedure: the
matched-candidate control that isolates what the dynamics model adds. ``deploy`` is the pick a live
vehicle would drive: the clearance-penalised CEM pick, or plain A* when no sampled route validates.
Picks per layout are exported as a route file for ``traverse_wp3_chrono_eval.py --route-file``.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nedm.traverse import nrd_data as D
from nedm.traverse.layout import EpisodeLayout
from nedm.traverse.energy_floor import EnergyFloor
from nedm.traverse.nrd_model import DT_S
from nedm.traverse.oracle import PlannerParams, plan_to_ring_fallback
from nedm.traverse.planner_b import MapDecoder, goal_from_map, occupancy_discs
from nedm.traverse.planner_s import FOOTPRINT_DISCS, FOOTPRINT_HALF_W, resample_routes, sample_routes
from nedm.traverse.power_calib import KINDS, PowerModel
from nedm.traverse.terrain import TerrainMap
from nedm.traverse.tracker_env import TraverseTrackingEnv, merge_env_cfg
from traverse_wp4_score_candidates import CANDIDATE_SWEEP, build_candidates, load_policy, rollout, route_dict

RES_KEYS = ("time_s", "completed", "failed", "collided", "collided_true", "min_clearance_m", "min_clearance_true_m",
            "energy_kj", "energy_act_kj", "energy_state_kj", "energy_floor")


def profile_time_s(plan) -> float:
    """Duration the speed profile implies (what a rule-based planner 'predicts')."""
    v, seg = np.asarray(plan.speeds, float), np.diff(np.asarray(plan.stations, float))
    return float(np.sum(seg / np.maximum(0.5 * (v[:-1] + v[1:]), 0.5)))


def geom_clearance(plans, obs) -> np.ndarray:
    """Min footprint clearance (m) of each route against disc obstacles (x, y, r); inf without obstacles."""
    obs = np.asarray(obs, float)
    out = np.full(len(plans), np.inf)
    if len(obs) == 0:
        return out
    for i, p in enumerate(plans):
        pts = np.asarray(p.waypoints, float)
        tang = np.gradient(pts, axis=0); tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
        centres = np.concatenate([pts + off * tang for off in FOOTPRINT_DISCS], 0)
        d = np.hypot(centres[:, None, 0] - obs[None, :, 0], centres[:, None, 1] - obs[None, :, 1]) - obs[None, :, 2] - FOOTPRINT_HALF_W
        out[i] = d.min()
    return out


class Imaginer:
    """Batched imagined rollouts of route candidates on one layout, over one or more dynamics models."""

    def __init__(self, args, power_models, tmap=None, cache_dir: Path | None = None):
        self.args, self.power_models, self.tmap = args, power_models, tmap
        self.cache_dir = str(cache_dir or args.cache)
        self.floor = EnergyFloor.load(Path(args.energy_floor), args.floor_sigmas) if args.energy_floor else None
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
                                 "dynamics_checkpoint": ckpt, "arena": a.arena, "cache": self.cache_dir, "routes": a.routes,
                                 "fragment_steps_max": self.horizon, "z1_extra_cache": sidecar or None, "map_key": a.map_key})
            env = TraverseTrackingEnv(cfg, device=dev, entries=entries)
            policy = load_policy(Path(a.policy), env, dev)
            obst = torch.tensor(np.asarray(discs, np.float32).reshape(-1, 3), device=dev)[None].expand(len(entries), -1, -1).contiguous()
            obst_true = torch.tensor(np.asarray(layout.obstacles(), np.float32), device=dev)[None].expand(len(entries), -1, -1).contiguous()
            sp = env.pose.clone(); sp[:] = torch.tensor(start, device=dev, dtype=sp.dtype)
            res = rollout(env, policy, self.horizon, obst, obst_true, self.power_models, sp, rest_start=a.from_rest)
            per_model.append({k: v.cpu().numpy() for k, v in res.items() if not k.startswith("_")})
            del env, policy
        torch.cuda.empty_cache()
        out = {k: per_model[0][k] for k in RES_KEYS if k in per_model[0]}  # time, flags, clearances: primary model
        # pessimistic energy = max over the chosen estimators and over all models
        names = {"head": "energy_kj", "act": "energy_act_kj", "state": "energy_state_kj"}
        terms = [r[names[t]] for r in per_model for t in a.pess_terms if names[t] in r]
        if self.floor is not None:  # geometry floor: no route of this length / speed / climb has cost less in Chrono
            out["energy_floor"] = np.array([self.floor.floor_kj(p, self.tmap) for p in plans])
            terms.append(out["energy_floor"])
        out["energy_pess"] = np.max(terms, axis=0)
        out["ok"] = np.all([r["completed"] & ~r["failed"] & ~r["collided"] for r in per_model], axis=0)
        return out


class GeometryScorer:
    """Matched-candidate control: no dynamics model. Time = what the speed profile implies; energy = the
    route-geometry regression fitted on Chrono runs (``EnergyFloor`` with k = 0); a route is 'ok' when its
    footprint clears the camera cells; clearance is geometric."""

    def __init__(self, args, tmap):
        self.tmap = tmap
        self.fit = EnergyFloor.load(Path(args.energy_floor), 0.0)
        self.floor = EnergyFloor.load(Path(args.energy_floor), args.floor_sigmas)

    def __call__(self, key: str, plans, start, discs, layout) -> dict[str, np.ndarray]:
        n = len(plans)
        clear = geom_clearance(plans, discs)
        clear_true = geom_clearance(plans, layout.obstacles())
        fit = np.array([self.fit.fit_kj(p, self.tmap) for p in plans])
        return {"time_s": np.array([profile_time_s(p) for p in plans]), "completed": np.ones(n, bool), "failed": np.zeros(n, bool),
                "collided": clear < 0, "collided_true": clear_true < 0, "min_clearance_m": clear, "min_clearance_true_m": clear_true,
                "energy_kj": fit, "energy_act_kj": fit, "energy_floor": np.array([self.floor.floor_kj(p, self.tmap) for p in plans]),
                "energy_pess": fit, "ok": clear >= 0}


def concat(a: dict, b: dict) -> dict:
    return {k: np.concatenate([a[k], b[k]]) for k in a if k in b}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="artifacts/traverse/wp3_tracker_v1")
    ap.add_argument("--dynamics-checkpoints", nargs="+", default=["artifacts/traverse/wp2_mapv2_pt_dag_ro8_amd/ckpt_best.pt"],
                    help="first = primary (time, flags); all of them enter the pessimistic energy maximum")
    ap.add_argument("--z1-extra-cache", nargs="*", default=None, help="sidecar per checkpoint ('' for none)")
    ap.add_argument("--maphead", default="artifacts/traverse/wp4_maphead_v2/ckpt_best.pt")
    ap.add_argument("--start-poses", default="artifacts/traverse/wp4_start_poses/val_start_poses.json")
    ap.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6", help="dynamics cache: key selection + (without --map-dir) maps and rest state")
    ap.add_argument("--map-dir", default=None, help="self-contained cache whose map_v2 is the LIVE single-frame scene map (traverse_wp5_live_inputs.py)")
    ap.add_argument("--map-key", default="map_v2")
    ap.add_argument("--routes", default="artifacts/traverse/wp3_routes")
    ap.add_argument("--stores", default="artifacts/traverse")
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--power-calib", default="artifacts/traverse/wp4_power_calib/power_calib.json")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--terrain", choices=["true", "predicted"], default="true",
                    help="height field for the speed profiles, A* slope caps and the geometry floor: the arena's or the camera map head's")
    ap.add_argument("--scorer", choices=["nrd", "geometry"], default="nrd")
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--max-imagined", type=int, default=2000)
    ap.add_argument("--prefilter-margin", type=float, default=0.2, help="footprint slack vs camera cells for sampled routes")
    ap.add_argument("--occ-threshold", type=float, default=0.85)
    ap.add_argument("--clear-start-radius", type=float, default=3.5,
                    help="drop predicted cells this close to the start estimate (the parked vehicle is in the live frame)")
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
    ap.add_argument("--energy-floor", default=None, help="energy_floor.json from traverse_wp5_energy_floor.py: adds fit - k*sigma to the pessimistic maximum")
    ap.add_argument("--floor-sigmas", type=float, default=1.5)
    ap.add_argument("--from-rest", action="store_true",
                    help="seed the imagination with a rest context at the camera start pose (a live vehicle has no recording)")
    ap.add_argument("--no-astar", action="store_true", help="skip the A* parameter sweep (astar_best)")
    ap.add_argument("--no-astar-plain", action="store_true", help="skip the plain A* baseline")
    ap.add_argument("--pick-prefix", default="", help="prefix for the exported pick names (e.g. 'geo_' for the geometry-scorer run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.scorer == "geometry" and not args.energy_floor:
        ap.error("--scorer geometry needs --energy-floor (its energy regression)")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache, routes = Path(args.cache), Path(args.routes)
    map_src = Path(args.map_dir) if args.map_dir else cache
    keys = D.load_cache_keys(cache)
    split = dict(zip(("train", "val", "test"), D.split_keys(keys)))[args.split]
    manifest = json.loads((routes / "routes_manifest.json").read_text())
    keys = [k for k in split if k in set(manifest["families"]["oracle"])][: args.episodes]
    if args.map_dir:
        have = set(D.load_cache_keys(map_src))
        missing = [k for k in keys if k not in have]
        if missing:
            raise SystemExit(f"{len(missing)} selected layouts have no live map in {map_src}: {missing[:3]}")
    tmap = TerrainMap.from_dir(Path(args.arena))
    decoder = MapDecoder(Path(args.maphead), Path(args.arena), args.device)
    start_est = json.loads(Path(args.start_poses).read_text())
    power_models = {k: PowerModel.load(Path(args.power_calib), k) for k in KINDS}
    rng = np.random.default_rng(args.seed)
    results, export = [], {}
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    for n_ep, key in enumerate(keys):
        t0 = time.perf_counter()
        store, ep = key.split("__", 1)
        meta = json.loads((Path(args.stores) / store / ep / "meta.json").read_text())
        layout = EpisodeLayout.from_json(meta["layout"])
        with np.load(map_src / f"{key}.npz") as d:
            occ, elev = decoder(d[args.map_key])
        discs = occupancy_discs(occ, decoder.size_m, args.occ_threshold, mode="cells")
        goal = goal_from_map(occ, decoder.size_m, args.occ_threshold)
        sx, sy, syaw = start_est[key]["est"]
        n_start_cells = sum(math.hypot(d[0] - sx, d[1] - sy) < args.clear_start_radius for d in discs)
        discs = [d for d in discs if math.hypot(d[0] - sx, d[1] - sy) >= args.clear_start_radius]
        plan_tmap = decoder.terrain(elev) if args.terrain == "predicted" else tmap
        score = Imaginer(args, power_models, plan_tmap, map_src) if args.scorer == "nrd" else GeometryScorer(args, plan_tmap)
        fit = EnergyFloor.load(Path(args.energy_floor), 0.0) if args.energy_floor else None
        t1 = time.perf_counter()
        sampled, stats = sample_routes((sx, sy), syaw, goal, args.samples, rng, plan_tmap, discs,
                                       prefilter_margin_m=args.prefilter_margin, max_routes=args.max_imagined)
        t_sample = time.perf_counter() - t1
        t1 = time.perf_counter()
        astar = [] if args.no_astar else build_candidates(meta, plan_tmap, CANDIDATE_SWEEP, obstacles=discs, repair_iterations=40,
                                                          margin_fallback=True, goal=goal, start=(sx, sy))[0]
        t_astar = time.perf_counter() - t1
        t1 = time.perf_counter()
        plain = None if args.no_astar_plain else plan_to_ring_fallback(plan_tmap, discs, (sx, sy), goal,
                                                                     replace(PlannerParams(), curvature_repair_iterations=40))  # same smoothing budget as the sweep
        t_plain = time.perf_counter() - t1
        if plain is not None:
            plain.meta = {**plain.meta, "candidate": "astar_plain"}
        plans = list(sampled) + [p for _, p in astar] + ([plain] if plain is not None else [])
        tags = np.array(["S"] * len(sampled) + ["A"] * len(astar) + (["P"] if plain is not None else []))
        rounds = np.zeros(len(plans), int)
        row = {"key": key, "pick_prefix": args.pick_prefix, "scorer": args.scorer, "n_start_cells_cleared": int(n_start_cells),
               "n_discs": len(discs), **{f"n_{k}": int(v) for k, v in stats.items()}, "sampled_imagined": int(len(sampled)),
               "astar": int(len(astar)), "astar_plain": plain is not None, "astar_plain_margin": None if plain is None else plain.meta.get("tracker_margin_m"),
               "t_sample_s": t_sample, "t_astar_s": t_astar, "t_astar_plain_s": t_plain, "round_best_pess": [], "round_imagined": []}
        if not plans:
            row["t_total_s"] = time.perf_counter() - t0
            results.append(row); export[key] = []
            print(f"[{n_ep + 1}/{len(keys)}] {ep}: no candidate at all", flush=True); continue
        t2 = time.perf_counter()
        res = score(key, plans, (sx, sy, syaw), discs, layout)
        t_imag = time.perf_counter() - t2
        pess = lambda r: np.where(r["ok"], r["time_s"] + r["energy_pess"] / 10.0, np.inf)
        row.update({"sampled_ok": int(res["ok"][tags == "S"].sum()), "astar_ok": int(res["ok"][tags == "A"].sum()),
                    "astar_plain_ok": bool(res["ok"][tags == "P"].any()) if plain is not None else None, "t_imagine_s": t_imag})
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
            children, cst = resample_routes([plans[i] for i in elite], args.cem_children, rng, goal, plan_tmap, discs,
                                            sigma_m=sig_m, sigma_v=sig_v, prefilter_margin_m=args.prefilter_margin, tag=f"cem{r}")
            if not children:
                break
            res_c = score(key, children, (sx, sy, syaw), discs, layout)
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
                            energy_act_kj=res.get("energy_act_kj", res["energy_kj"]), energy_pess=res["energy_pess"], collided_true=res["collided_true"],
                            clear=res["min_clearance_m"], clear_true=res["min_clearance_true_m"],
                            length=np.array([p.length_m for p in plans]), v_cruise=np.array([float(p.meta.get("v_cruise", p.speeds.max())) for p in plans]))
        picks = {"astar_plain": (tags == "P", np.zeros(len(plans))),  # the baseline is driven whatever the imagination says
                 "astar_best": (tags == "A", cost_pess),
                 "sampled_pess": ((tags == "S") & (rounds == 0), cost_pess),
                 "cem_pess": (tags == "S", cost_pess),
                 "cem_pess_clear": (tags == "S", cost_clear)}
        export[key] = []
        chosen = {}
        for name, (mask, cost) in picks.items():
            idx = np.nonzero(mask)[0]
            if len(idx) == 0 or not np.isfinite(cost[idx]).any():
                row[f"{name}_cost"] = None; continue
            b = int(idx[np.argmin(cost[idx])]); chosen[name] = b
            row[f"{name}_cost"] = float(cost_pess[b]) if np.isfinite(cost_pess[b]) else None
            row[f"{name}_ok"] = bool(res["ok"][b]); row[f"{name}_time"] = float(res["time_s"][b]); row[f"{name}_energy"] = float(res["energy_pess"][b])
            row[f"{name}_energy_head"] = float(res["energy_kj"][b]); row[f"{name}_energy_act"] = float(res.get("energy_act_kj", res["energy_kj"])[b])
            if "energy_state_kj" in res:
                row[f"{name}_energy_state"] = float(res["energy_state_kj"][b])
            if "energy_floor" in res:
                row[f"{name}_energy_floor"] = float(res["energy_floor"][b])
            row[f"{name}_profile_time"] = profile_time_s(plans[b])  # the rule-based prediction, for every arm
            if fit is not None:
                row[f"{name}_fit_energy"] = fit.fit_kj(plans[b], plan_tmap)
            row[f"{name}_clear"] = float(res["min_clearance_m"][b]); row[f"{name}_clear_true"] = float(res["min_clearance_true_m"][b])
            row[f"{name}_collided_true"] = bool(res["collided_true"][b]); row[f"{name}_round"] = int(rounds[b])
            row[f"{name}_length"] = float(plans[b].length_m); row[f"{name}_vcruise"] = float(plans[b].meta.get("v_cruise", plans[b].speeds.max()))
            export[key].append({"candidate": args.pick_prefix + name, **{k: np.asarray(v).tolist() for k, v in route_dict(plans[b]).items()}})
        # what a live vehicle drives: the clearance-penalised CEM pick, plain A* when nothing sampled validates
        if "cem_pess_clear" in chosen:
            row["deploy_fallback"] = False; b = chosen["cem_pess_clear"]
        elif "astar_plain" in chosen:
            row["deploy_fallback"] = True; b = chosen["astar_plain"]
        else:
            row["deploy_fallback"] = None; b = None
        if b is not None:
            row["deploy_cost"] = row.get(("cem_pess_clear" if not row["deploy_fallback"] else "astar_plain") + "_cost")
            export[key].append({"candidate": args.pick_prefix + "deploy", **{k: np.asarray(v).tolist() for k, v in route_dict(plans[b]).items()}})
        s_ok = (tags == "S") & res["ok"]
        row["sampled_ok_true_collision_rate"] = float(res["collided_true"][s_ok].mean()) if s_ok.any() else None
        results.append(row)
        rb = " -> ".join("-" if v is None else f"{v:.2f}" for v in row["round_best_pess"])
        pc = row.get("astar_plain_cost"); pc = "-" if pc is None else f"{pc:.2f}"
        print(f"[{n_ep + 1}/{len(keys)}] {ep}: cells {len(discs)} (+{n_start_cells} at start) | sampled {stats['sampled']} -> clear {stats['clear']} -> ok {row['sampled_ok']} | "
              f"A* {row['astar']} ok {row['astar_ok']} plain ok {row['astar_plain_ok']} cost {pc} | best per round {rb} | A* best {row.get('astar_best_cost')} | "
              f"t sample {t_sample:.1f} A* {t_astar:.1f} plain {t_plain:.1f} score {t_imag:.1f} cem {t_cem:.1f} s", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    (out / "routes.json").write_text(json.dumps(export))
    names = ["astar_plain", "astar_best", "sampled_pess", "cem_pess", "cem_pess_clear"]
    both = [r for r in results if all(r.get(f"{n}_cost") is not None for n in names)]
    print(f"\n{len(both)} layouts with all picks ({args.scorer} score = time + E/10):")
    for n in names:
        print(f"  {n:15s} cost {np.mean([r[f'{n}_cost'] for r in both]):6.2f}  time {np.mean([r[f'{n}_time'] for r in both]):5.2f}  "
              f"E {np.mean([r[f'{n}_energy'] for r in both]):6.1f}  clear(pred) {np.mean([r[f'{n}_clear'] for r in both]):.2f}  "
              f"clear(true) {np.mean([r[f'{n}_clear_true'] for r in both]):.2f} min {np.min([r[f'{n}_clear_true'] for r in both]):.2f}  "
              f"true-collisions {sum(r[f'{n}_collided_true'] for r in both)}  beats plain A* {sum(r[f'{n}_cost'] < r['astar_plain_cost'] for r in both)}/{len(both)}")
    print(f"fallbacks to plain A*: {sum(bool(r.get('deploy_fallback')) for r in results)}/{len(results)}; "
          f"mean wall per layout {np.mean([r['t_total_s'] for r in results if 't_total_s' in r]):.1f} s")


if __name__ == "__main__":
    main()
