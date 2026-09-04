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
from nedm.traverse.nrd_model import DT_S
from nedm.traverse.oracle import PlanCandidate, PlannerParams, plan_to_ring
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


def build_candidates(meta: dict, tmap: TerrainMap, sweep: dict) -> tuple[list[tuple[str, PlanCandidate]], EpisodeLayout]:
    layout = EpisodeLayout.from_json(meta["layout"])
    obstacles = layout.obstacles()
    out, seen = [], set()
    for name, overrides in sweep.items():
        plan = plan_to_ring(tmap, obstacles, layout.start_xy, layout.house_xy,
                            replace(PlannerParams(), **overrides))
        if plan is None:
            continue
        sig = (len(plan.waypoints), round(plan.length_m, 2), round(float(plan.speeds.mean()), 3))
        if sig in seen:
            continue
        seen.add(sig)
        plan.meta = {**plan.meta, "candidate": name}
        out.append((name, plan))
    return out, layout


@torch.no_grad()
def rollout(env: TraverseTrackingEnv, policy, horizon: int, obstacles: torch.Tensor) -> dict[str, torch.Tensor]:
    """Roll every env from its episode start for ``horizon`` steps; obstacles (N, M, 3) padded with r<0."""
    n, dev = env.num_envs, env.device
    env.reset_idx(torch.arange(n, device=dev), episode_ids=torch.arange(n, device=dev),
                  start_frames=torch.full((n,), env.context, device=dev, dtype=torch.long),
                  fragment_steps=torch.full((n,), horizon, device=dev, dtype=torch.long))
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
    for step in range(horizon):
        obs = env.obs_buf
        act = policy(obs)
        _, _, dones, extras = env.step(act)
        err = env._route_errors()
        ct = err["e_ct"].abs()
        ct_sum += ct * active; ct_max = torch.where(active, torch.maximum(ct_max, ct), ct_max)
        # footprint clearance against the obstacle discs
        cos_y, sin_y = torch.cos(env.pose[:, 2]), torch.sin(env.pose[:, 2])
        clear = torch.full((n,), float("inf"), device=dev)
        for off in FOOTPRINT_DISCS:
            cx, cy = env.pose[:, 0] + off * cos_y, env.pose[:, 1] + off * sin_y
            d = torch.hypot(obstacles[..., 0] - cx[:, None], obstacles[..., 1] - cy[:, None]) - obstacles[..., 2] - FOOTPRINT_HALF_W
            d = torch.where(obstacles[..., 2] >= 0, d, torch.full_like(d, float("inf")))
            clear = torch.minimum(clear, d.min(dim=1).values)
        min_clear = torch.where(active, torch.minimum(min_clear, clear), min_clear)
        collided |= active & (clear < 0)
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
    return {"time_s": end_step.float() * DT_S, "completed": completed, "failed": failed,
            "collided": collided, "min_clearance_m": min_clear, "energy_kj": energy,
            "mean_ct_m": ct_sum / steps, "max_ct_m": ct_max, "progress_m": progress}


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
    entries, index, obstacle_lists, truths = [], [], [], {}
    for key in keys:
        store, ep = key.split("__", 1)
        meta = json.loads((Path(args.stores) / store / ep / "meta.json").read_text())
        cands, layout = build_candidates(meta, tmap, CANDIDATE_SWEEP)
        with np.load(routes / f"{key}.npz") as r:
            recorded = {n: r[n] for n in ("waypoints", "speeds", "headings", "stations")}
        truths[key] = recorded_truth(cache, key, recorded)
        for name, plan in [("recorded", None)] + (cands if args.policy != "replay" else []):
            route = recorded if plan is None else route_dict(plan)
            entries.append((key, route))
            index.append({"key": key, "candidate": name, "length_m": float(route["stations"][-1]),
                          "mean_speed_mps": float(route["speeds"].mean())})
            obstacle_lists.append(layout.obstacles())
    print(f"{len(keys)} episodes -> {len(entries)} candidate rollouts (plans in {time.time() - t0:.1f}s)", flush=True)

    m = max(len(o) for o in obstacle_lists)
    obst = np.full((len(entries), m, 3), -1.0, np.float32)
    for i, o in enumerate(obstacle_lists):
        obst[i, : len(o)] = np.asarray(o, np.float32)
    cfg = merge_env_cfg({"num_envs": len(entries), "device": args.device, "auto_reset": False,
                         "dynamics_checkpoint": args.dynamics_checkpoint, "arena": args.arena,
                         "cache": args.cache, "routes": args.routes, "split": args.split,
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
    res = rollout(env, policy, int(round(args.horizon_s / DT_S)), torch.tensor(obst, device=env.device))
    res = {k: v.cpu().numpy() for k, v in res.items()}

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
            "time_s": float(np.mean([r["time_s"] for r in sel])), "energy_kj": float(np.mean([r["energy_kj"] for r in sel])),
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
        best = min(pool, key=lambda r: r["time_s"] + r["energy_kj"] / 10.0)
        wins[best["candidate"]] = wins.get(best["candidate"], 0) + 1
    summary["selection_wins"] = wins
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "candidate_scores.json").write_text(json.dumps({"summary": summary, "rows": rows, "calibration": cal}, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
