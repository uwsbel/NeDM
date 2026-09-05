"""Planner-C: score candidate plans by rolling the tracker out inside the NRD (plan §9.5).

For each held-out episode the privileged oracle produces k candidate routes to the
house (cost-weight / speed / margin sweeps -- plan §7.6 "diverse candidates") and
the RECORDED route the collection driver actually drove is added as a candidate
with ground truth attached. Every candidate is then tracked by the WP3 policy
inside the frozen WP2 model from the episode's real start context and scored on

    time      steps until the route end (or the horizon)          -> predicted z1
    energy    integral of the auxiliary power head (kJ)            -> predicted power
    tracking  mean / max cross-track error of the imagined vehicle
    safety    roll / pitch / cross-track bound violations
    collision three-disc footprint vs the layout's obstacle discs (privileged v1)

The recorded-route candidate is the calibration: the imagined time-to-end and
energy are compared with what the recorded vehicle actually did on that same
route, which is the direct test of "planner rollout inside NRD" against reality
without touching Chrono.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nedm.traverse import nrd_data as D
from nedm.traverse.layout import EpisodeLayout
from nedm.traverse.nrd_model import DT_S, VX
from nedm.traverse.oracle import PlanCandidate, PlannerParams, plan_to_ring, plan_to_ring_fallback
from nedm.traverse.planner_b import MapDecoder, goal_from_map, occupancy_discs
from nedm.traverse.power_calib import KINDS, PowerModel
from nedm.traverse.terrain import TerrainMap
from nedm.traverse.tracker_env import TraverseTrackingEnv, merge_env_cfg, pure_pursuit_actions

CANDIDATE_SWEEP = {
    "oracle": {},
    "shortest": {"energy_weight": 0.0},
    "energy_averse": {"energy_weight": 4.0},
    "slow": {"v_cruise_mps": 4.0},
    "fast": {"v_cruise_mps": 9.0},
    "wide_berth": {"inflation_m": 3.0},
}
FOOTPRINT_HALF_W = 1.3
FOOTPRINT_DISCS = (-1.9, 0.0, 1.9)  # disc centres along the body x axis


def route_dict(plan: PlanCandidate) -> dict[str, np.ndarray]:
    return {"waypoints": plan.waypoints.astype(np.float32), "speeds": plan.speeds.astype(np.float32),
            "headings": plan.headings.astype(np.float32), "stations": plan.stations.astype(np.float32)}


def recorded_truth(cache: Path, key: str, route: dict[str, np.ndarray]) -> dict[str, float]:
    """What the recorded vehicle did on this route: time to the end, energy, cross-track."""
    with np.load(cache / f"{key}.npz") as d:
        pose, power = d["pose"], d["power"][:, 0]
    w, s, h = route["waypoints"], route["stations"], route["headings"]
    dist = np.linalg.norm(pose[:, None, :2] - w[None], axis=-1)
    j = dist.argmin(axis=1)
    reached = np.nonzero(j >= len(w) - 2)[0]
    end = int(reached[0]) if len(reached) else len(pose) - 1
    dx, dy = pose[:end + 1, 0] - w[j[:end + 1], 0], pose[:end + 1, 1] - w[j[:end + 1], 1]
    ct = np.abs(-dx * np.sin(h[j[:end + 1]]) + dy * np.cos(h[j[:end + 1]]))
    return {"time_s": float((end + 1) * DT_S), "completed": bool(len(reached)),
            "energy_kj": float(power[: end + 1].sum() * DT_S),
            "mean_ct_m": float(ct.mean()), "max_ct_m": float(ct.max())}


def build_candidates(meta: dict, tmap: TerrainMap, sweep: dict, obstacles=None, margin: float | None = None,
                     repair_iterations: int | None = None, margin_fallback: bool = False, goal=None, start=None,
                     ) -> tuple[list[tuple[str, PlanCandidate]], EpisodeLayout]:
    """``obstacles`` None -> the true footprint discs (privileged oracle); otherwise the
    disc list to plan against (e.g. occupied cells of the camera-derived map)."""
    layout = EpisodeLayout.from_json(meta["layout"])
    obstacles = layout.obstacles() if obstacles is None else obstacles
    goal = layout.house_xy if goal is None else goal
    start = layout.start_xy if start is None else start
    out, seen = [], set()
    for name, overrides in sweep.items():
        params = replace(PlannerParams(), **overrides)
        if margin is not None:
            params = replace(params, tracker_p95_margin_m=margin)
        if repair_iterations is not None:
            params = replace(params, curvature_repair_iterations=repair_iterations)
        planner = plan_to_ring_fallback if margin_fallback else plan_to_ring
        plan = planner(tmap, obstacles, start, goal, params)
        if plan is None:
            continue
        sig = (len(plan.waypoints), round(plan.length_m, 2), round(float(plan.speeds.mean()), 3))
        if sig in seen:
            continue
        seen.add(sig)
        plan.meta = {**plan.meta, "candidate": name}
        out.append((name, plan))
    return out, layout


def footprint_clearance(env: TraverseTrackingEnv, obstacles: torch.Tensor) -> torch.Tensor:
    n, dev = env.num_envs, env.device
    cos_y, sin_y = torch.cos(env.pose[:, 2]), torch.sin(env.pose[:, 2])
    clear = torch.full((n,), float("inf"), device=dev)
    for off in FOOTPRINT_DISCS:
        cx, cy = env.pose[:, 0] + off * cos_y, env.pose[:, 1] + off * sin_y
        d = torch.hypot(obstacles[..., 0] - cx[:, None], obstacles[..., 1] - cy[:, None]) - obstacles[..., 2] - FOOTPRINT_HALF_W
        d = torch.where(obstacles[..., 2] >= 0, d, torch.full_like(d, float("inf")))
        clear = torch.minimum(clear, d.min(dim=1).values)
    return clear


@torch.no_grad()
def rollout(env: TraverseTrackingEnv, policy, horizon: int, obstacles: torch.Tensor,
            obstacles_true: torch.Tensor | None = None, power_models: dict[str, PowerModel] | None = None,
            start_poses: torch.Tensor | None = None, rest_start: bool = False) -> dict[str, torch.Tensor]:
    """Roll every env from its episode start for ``horizon`` steps; obstacles (N, M, 3) padded with r<0
    are the set the SCORE uses (true discs or camera-derived cells); ``obstacles_true`` adds the
    privileged metric. ``power_models`` -> calibrated kinematic energies alongside the head's.
    ``rest_start``: instead of the recorded frames 0-15 (vehicle already launched), seed the context with
    the episode's frame-0 state (at rest, brake on) repeated -- what a live vehicle at t=0 looks like."""
    n, dev = env.num_envs, env.device
    obstacles_true = obstacles if obstacles_true is None else obstacles_true
    power_models = power_models or {}
    env.reset_idx(torch.arange(n, device=dev), episode_ids=torch.arange(n, device=dev),
                  start_frames=torch.full((n,), env.context, device=dev, dtype=torch.long),
                  fragment_steps=torch.full((n,), horizon, device=dev, dtype=torch.long))
    if rest_start:
        b, c = env.bank, env.context
        z0 = b.z1[env.env_ep, 0]  # normalized frame-0 state: settled, at rest
        pose0 = b.pose[env.env_ep, 0]
        brake = (torch.tensor([0.0, 0.0, 1.0], device=dev) - env.act_mean) / env.act_std
        env.z1_hist[:] = z0[:, None, :].expand(-1, c, -1)
        env.act_hist[:] = brake[None, None, :].expand(n, c, -1)
        env.pose[:] = pose0 if start_poses is None else start_poses  # the live vehicle knows only the camera's estimate
        with torch.no_grad():
            env.token_hist[:] = env.model.cropper(env.env_maps, env.pose[:, None, :].expand(-1, c, -1))
        env.z1_phys[:] = z0 * env.z1_std + env.z1_mean
        env.last_actions[:] = torch.tensor([0.0, 0.0, 1.0], device=dev); env.actions[:] = env.last_actions
        d = (b.route_xy[env.env_ep] - env.pose[:, None, :2]).norm(dim=-1)
        valid = torch.arange(b.route_xy.shape[1], device=dev)[None, :] < b.route_len[env.env_ep][:, None]
        env.route_idx[:] = torch.where(valid, d, torch.full_like(d, float("inf"))).argmin(dim=1)
        env.start_station_m[:] = b.route_s[env.env_ep, env.route_idx]
    if start_poses is not None:  # dead-reckon from the camera's start estimate instead of the recorded pose
        env.pose[:] = start_poses
    env._compute_observations()
    active = torch.ones(n, dtype=torch.bool, device=dev)
    end_step = torch.full((n,), horizon, device=dev, dtype=torch.long)
    completed = torch.zeros(n, dtype=torch.bool, device=dev)
    failed = torch.zeros(n, dtype=torch.bool, device=dev)
    collided = torch.zeros(n, dtype=torch.bool, device=dev)
    min_clear = torch.full((n,), float("inf"), device=dev)
    ct_sum = torch.zeros(n, device=dev); ct_max = torch.zeros(n, device=dev)
    energy = torch.zeros(n, device=dev)
    progress = torch.zeros(n, device=dev)
    collided_true = torch.zeros(n, dtype=torch.bool, device=dev)
    min_clear_true = torch.full((n,), float("inf"), device=dev)
    e_kin = {k: torch.zeros(n, device=dev) for k in power_models}
    has_pt = env.z1_phys.shape[1] >= 17  # powertrain state present: power = engine speed x motorshaft torque
    e_state = torch.zeros(n, device=dev)
    prev_vx = env.z1_phys[:, VX].clone()
    traj = {"z1": [], "act": [], "active": [], "power": []}
    for step in range(horizon):
        obs = env.obs_buf
        act = policy(obs)
        _, _, dones, extras = env.step(act)
        err = env._route_errors()
        ct = err["e_ct"].abs()
        ct_sum += ct * active; ct_max = torch.where(active, torch.maximum(ct_max, ct), ct_max)
        vx = env.z1_phys[:, VX]
        ax = (vx - prev_vx) / DT_S
        prev_vx = vx.clone()
        for k, pm in power_models.items():
            e_kin[k] += pm.predict(env.z1_phys, env.actions, ax, torch) * DT_S * active
        if has_pt:
            e_state += (env.z1_phys[:, 15] * env.z1_phys[:, 16] / 1000.0) * DT_S * active
        traj["z1"].append(env.z1_phys.clone()); traj["act"].append(env.actions.clone())
        traj["active"].append(active.clone()); traj["power"].append(env.last_power.clone() if hasattr(env, "last_power") else torch.zeros(n, device=dev))
        # footprint clearance against the scoring obstacle set and the true discs
        clear = footprint_clearance(env, obstacles)
        min_clear = torch.where(active, torch.minimum(min_clear, clear), min_clear)
        collided |= active & (clear < 0)
        clear_t = footprint_clearance(env, obstacles_true)
        min_clear_true = torch.where(active, torch.minimum(min_clear_true, clear_t), min_clear_true)
        collided_true |= active & (clear_t < 0)
        just_done = active & dones.bool()
        end_step = torch.where(just_done, torch.full_like(end_step, step + 1), end_step)
        completed |= just_done & err["route_end"]
        failed |= just_done & ~env.time_out_buf
        energy = torch.where(active, env.energy_kj, energy)
        progress = torch.where(active, env.progress_m, progress)
        active &= ~dones.bool()
        if not active.any():
            break
    steps = end_step.float().clamp(min=1.0)
    out = {"time_s": end_step.float() * DT_S, "completed": completed, "failed": failed,
           "collided": collided, "min_clearance_m": min_clear, "energy_kj": energy,
           "collided_true": collided_true, "min_clearance_true_m": min_clear_true,
           "mean_ct_m": ct_sum / steps, "max_ct_m": ct_max, "progress_m": progress}
    for k, e in e_kin.items():
        out[f"energy_{k.replace('+', '')}_kj"] = e
    if has_pt:
        out["energy_state_kj"] = e_state
    out["_traj"] = {k: torch.stack(v).cpu().numpy() for k, v in traj.items()}
    return out


def load_policy(run_dir: Path, env: TraverseTrackingEnv, device: str):
    from rsl_rl.runners import OnPolicyRunner
    train_cfg = json.loads((run_dir / "train_cfg.json").read_text())
    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=device)
    runner.load(str(ckpts[-1]), load_optimizer=False)
    print(f"policy {ckpts[-1]}", flush=True)
    return runner.get_inference_policy(device=device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="pure_pursuit",
                    help="'pure_pursuit', 'replay' (recorded actions open-loop; recorded route only) or a WP3 run dir")
    ap.add_argument("--dynamics-checkpoint", default="artifacts/traverse/wp2_mapv2_index_amd/ckpt_best.pt")
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6")
    ap.add_argument("--routes", default="artifacts/traverse/wp3_routes")
    ap.add_argument("--stores", default="artifacts/traverse")
    ap.add_argument("--split", default="val")
    ap.add_argument("--families", nargs="+", default=["oracle"])
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--horizon-s", type=float, default=20.0)
    ap.add_argument("--candidates", choices=["oracle", "predicted"], default="oracle",
                    help="oracle = privileged true map; predicted = camera-derived map (Planner-B)")
    ap.add_argument("--terrain", choices=["true", "predicted"], default="true",
                    help="terrain for predicted-map planning (true = memorized-terrain rung)")
    ap.add_argument("--collision", choices=["true", "predicted"], default="true",
                    help="obstacle set the SCORE's collision check uses (true discs are always reported)")
    ap.add_argument("--maphead", default="artifacts/traverse/wp4_maphead_v1/ckpt_best.pt")
    ap.add_argument("--map-key", default="map_v2")
    ap.add_argument("--occ-threshold", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=None, help="tracker margin override for candidate generation")
    ap.add_argument("--margin-fallback", action="store_true", help="0.9 -> 0.6 -> 0.3 margin rescue when no plan validates")
    ap.add_argument("--goal", choices=["true", "predicted"], default="true", help="ring centre: true house or largest camera blob")
    ap.add_argument("--start-poses", default=None,
                    help="json from traverse_wp4_start_pose_from_camera.py: plan and roll out from the camera's start estimate")
    ap.add_argument("--power-calib", default="artifacts/traverse/wp4_power_calib/power_calib.json")
    ap.add_argument("--z1-extra-cache", default=None, help="sidecar dir when the dynamics model has a 17-D (powertrain) state")
    ap.add_argument("--dump-trajectories", default=None, help="npz path for per-step imagined z1/actions")
    ap.add_argument("--export-routes", default=None, help="json path: every candidate route, for the Chrono eval")
    ap.add_argument("--energy-field", default="energy_act_kj", help="energy used in the selection objective")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cache, routes = Path(args.cache), Path(args.routes)
    keys = D.load_cache_keys(cache)
    split = dict(zip(("train", "val", "test"), D.split_keys(keys)))[args.split]
    manifest = json.loads((routes / "routes_manifest.json").read_text())
    allowed = set().union(*(set(manifest["families"][f]) for f in args.families))
    keys = [k for k in split if k in allowed][: args.episodes]
    tmap = TerrainMap.from_dir(Path(args.arena))

    t0 = time.time()
    need_map = args.candidates == "predicted" or args.collision == "predicted"
    decoder = MapDecoder(Path(args.maphead), Path(args.arena), args.device) if need_map else None
    entries, index, obstacle_lists, true_lists, truths, cells = [], [], [], [], {}, []
    start_est = json.loads(Path(args.start_poses).read_text()) if args.start_poses else {}
    start_list = []
    for key in keys:
        store, ep = key.split("__", 1)
        meta = json.loads((Path(args.stores) / store / ep / "meta.json").read_text())
        pred_discs, plan_tmap, pred_goal = None, tmap, None
        if decoder is not None:
            with np.load(cache / f"{key}.npz") as d:
                occ, elev = decoder(d[args.map_key])
            pred_discs = occupancy_discs(occ, decoder.size_m, args.occ_threshold, mode="cells")
            cells.append(len(pred_discs))
            pred_goal = goal_from_map(occ, decoder.size_m, args.occ_threshold) if args.goal == "predicted" else None
            if args.terrain == "predicted":
                plan_tmap = decoder.terrain(elev)
        cands, layout = build_candidates(meta, plan_tmap, CANDIDATE_SWEEP,
                                         obstacles=pred_discs if args.candidates == "predicted" else None,
                                         margin=args.margin,
                                         repair_iterations=40 if args.candidates == "predicted" else None,
                                         margin_fallback=args.margin_fallback, goal=pred_goal,
                                         start=tuple(start_est[key]["est"][:2]) if key in start_est else None)
        with np.load(routes / f"{key}.npz") as r:
            recorded = {n: r[n] for n in ("waypoints", "speeds", "headings", "stations")}
        truths[key] = recorded_truth(cache, key, recorded)
        for name, plan in [("recorded", None)] + (cands if args.policy != "replay" else []):
            route = recorded if plan is None else route_dict(plan)
            entries.append((key, route))
            index.append({"key": key, "candidate": name, "length_m": float(route["stations"][-1]),
                          "mean_speed_mps": float(route["speeds"].mean())})
            true_lists.append(layout.obstacles())
            obstacle_lists.append(pred_discs if args.collision == "predicted" else layout.obstacles())
            start_list.append(start_est[key]["est"] if key in start_est else None)
    if args.export_routes:
        exp = {}
        for (key, route), row in zip(entries, index):
            exp.setdefault(key, []).append({"candidate": row["candidate"], **{k: np.asarray(v).tolist() for k, v in route.items()}})
        Path(args.export_routes).parent.mkdir(parents=True, exist_ok=True)
        Path(args.export_routes).write_text(json.dumps(exp))
    print(f"{len(keys)} episodes -> {len(entries)} candidate rollouts (plans in {time.time() - t0:.1f}s)"
          + (f"; predicted occupied cells mean {np.mean(cells):.0f}" if cells else ""), flush=True)

    def pad(lists):
        m = max(len(o) for o in lists)
        arr = np.full((len(entries), m, 3), -1.0, np.float32)
        for i, o in enumerate(lists):
            if len(o):
                arr[i, : len(o)] = np.asarray(o, np.float32)
        return torch.tensor(arr, device=args.device)
    obst, obst_true = pad(obstacle_lists), pad(true_lists)
    power_models = {}
    if args.power_calib and Path(args.power_calib).exists():
        power_models = {k: PowerModel.load(Path(args.power_calib), k) for k in KINDS}
    cfg = merge_env_cfg({"num_envs": len(entries), "device": args.device, "auto_reset": False,
                         "dynamics_checkpoint": args.dynamics_checkpoint, "arena": args.arena,
                         "cache": args.cache, "routes": args.routes, "split": args.split,
                         "z1_extra_cache": args.z1_extra_cache,
                         "fragment_steps_max": int(round(args.horizon_s / DT_S))})
    env = TraverseTrackingEnv(cfg, device=args.device, entries=entries)
    if args.policy == "pure_pursuit":
        policy = lambda obs: pure_pursuit_actions(env)
    elif args.policy == "replay":
        # open-loop replay of the recorded driver's actions: isolates the model's own
        # time/energy error from any controller difference
        frames = torch.arange(env.bank.n_frames, device=env.device)
        def policy(obs):
            f = torch.minimum(env.episode_length_buf + env.context, frames[-1])
            return env.physical_to_policy(env.bank.act_raw[env.env_ep, f])
    else:
        policy = load_policy(Path(args.policy), env, args.device)
    start_poses = None
    if start_est:
        sp = env.pose.clone()
        for i, est in enumerate(start_list):
            if est is not None:
                sp[i] = torch.tensor(est, device=env.device, dtype=sp.dtype)
        start_poses = sp
    res = rollout(env, policy, int(round(args.horizon_s / DT_S)), obst, obst_true, power_models, start_poses)
    traj = res.pop("_traj")
    res = {k: v.cpu().numpy() for k, v in res.items()}
    if args.dump_trajectories:
        np.savez_compressed(args.dump_trajectories, keys=np.array([r["key"] for r in index]),
                            candidates=np.array([r["candidate"] for r in index]), **traj)

    rows = []
    for i, row in enumerate(index):
        rows.append({**row, **{k: (float(v[i]) if v.dtype != bool else bool(v[i])) for k, v in res.items()}})
    # calibration on the recorded route: imagined vs recorded time / energy
    cal = []
    for r in rows:
        if r["candidate"] == "recorded":
            t = truths[r["key"]]
            cal.append({"key": r["key"], "rec_time_s": t["time_s"], "img_time_s": r["time_s"],
                        "rec_energy_kj": t["energy_kj"], "img_energy_kj": r["energy_kj"],
                        "rec_completed": t["completed"], "img_completed": r["completed"],
                        "rec_mean_ct_m": t["mean_ct_m"], "img_mean_ct_m": r["mean_ct_m"]})
    both = [c for c in cal if c["rec_completed"] and c["img_completed"]]
    def stat(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return {"n": int(len(a)), "mean_rec": float(a.mean()), "mean_img": float(b.mean()),
                "mae": float(np.abs(a - b).mean()), "corr": float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")}
    summary = {
        "episodes": len(keys), "rollouts": len(entries), "horizon_s": args.horizon_s, "policy": args.policy,
        "candidates": args.candidates, "terrain": args.terrain, "collision": args.collision, "margin": args.margin, "goal": args.goal, "start_poses": args.start_poses,
        "predicted_cells_mean": float(np.mean(cells)) if cells else None,
        "recorded_route_calibration": {
            "completed_recorded": int(sum(c["rec_completed"] for c in cal)),
            "completed_imagined": int(sum(c["img_completed"] for c in cal)),
            "time": stat([c["rec_time_s"] for c in both], [c["img_time_s"] for c in both]) if both else None,
            "energy": stat([c["rec_energy_kj"] for c in both], [c["img_energy_kj"] for c in both]) if both else None,
            "mean_ct_imagined": float(np.mean([c["img_mean_ct_m"] for c in cal])),
            "mean_ct_recorded": float(np.mean([c["rec_mean_ct_m"] for c in cal])),
        },
        "per_candidate": {},
    }
    for name in ["recorded"] + list(CANDIDATE_SWEEP):
        sel = [r for r in rows if r["candidate"] == name]
        if not sel:
            continue
        summary["per_candidate"][name] = {
            "n": len(sel), "completed": float(np.mean([r["completed"] for r in sel])),
            "failed": float(np.mean([r["failed"] for r in sel])), "collided": float(np.mean([r["collided"] for r in sel])),
            "collided_true": float(np.mean([r["collided_true"] for r in sel])),
            "time_s": float(np.mean([r["time_s"] for r in sel])), "energy_kj": float(np.mean([r["energy_kj"] for r in sel])),
            **{k: float(np.mean([r[k] for r in sel])) for k in sel[0] if k.startswith("energy_") and k != "energy_kj"},
            "mean_ct_m": float(np.mean([r["mean_ct_m"] for r in sel])), "max_ct_m": float(np.mean([r["max_ct_m"] for r in sel])),
            "length_m": float(np.mean([r["length_m"] for r in sel])), "progress_m": float(np.mean([r["progress_m"] for r in sel])),
        }
    # selection: among oracle-family candidates, pick min(time + energy/10 kJ) with hard safety rejects
    wins = {}
    for key in keys:
        cands = [r for r in rows if r["key"] == key and r["candidate"] != "recorded"]
        if not cands:
            continue
        ok = [r for r in cands if r["completed"] and not r["failed"] and not r["collided"]]
        pool = ok or cands
        ef = args.energy_field if args.energy_field in pool[0] else "energy_kj"
        best = min(pool, key=lambda r: r["time_s"] + r[ef] / 10.0)
        wins[best["candidate"]] = wins.get(best["candidate"], 0) + 1
        best["selected"] = True
    summary["selection_wins"] = wins
    summary["selection_energy_field"] = ef if wins else None
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "candidate_scores.json").write_text(json.dumps({"summary": summary, "rows": rows, "calibration": cal}, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
