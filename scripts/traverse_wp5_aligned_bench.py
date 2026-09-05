#!/usr/bin/env python
"""Frame-aligned model test: imagination vs tracker-driven Chrono episodes from the SAME context.

The Chrono batches start from rest while the imagination starts from the recorded context at frame 16
(vehicle already at ~2 m/s, ~30 kJ spent), so their time / energy comparison carries a built-in head
start of ~0.8 s and ~20 kJ. Here the ground truth is a tracker-driven Chrono episode on a held-out
layout (``traverse_wp4_collect_tracker_episodes.py --split val``, 400 frames of z1 / action / pose /
power plus the route); every model imagines the same route from that episode's own frames 0-15 with
the same tracker, and time-to-route-end and energy are compared from frame 16 in both. No confound.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nedm.traverse.nrd_model import DT_S
from nedm.traverse.power_calib import KINDS, PowerModel
from nedm.traverse.tracker_env import TraverseTrackingEnv, merge_env_cfg
from traverse_wp4_score_candidates import load_policy, rollout

CONTEXT = 16


def build_mini_cache(src: Path, base: Path, dst: Path, map_key: str = "map_v2") -> list[str]:
    """Tracker episodes + their layout's scene map in one self-contained cache dir (what the env loads)."""
    dst.mkdir(parents=True, exist_ok=True)
    keys = []
    for f in sorted(src.glob("*__dag_*.npz")):
        key = f.stem
        keys.append(key)
        if (dst / f.name).exists():
            continue
        with np.load(f) as d:
            arrays = {k: d[k] for k in d.files}
        with np.load(base / f"{str(arrays['source_key'])}.npz") as b:
            arrays[map_key] = b[map_key]
        np.savez(dst / f.name, **arrays)
    (dst / "cache_manifest.json").write_text(json.dumps({"episodes": keys, "source_cache": str(base)}))
    return keys


def truth(npz: Path, start: int = CONTEXT) -> dict:
    with np.load(npz) as d:
        pose, power, w = d["pose"], d["power"][:, 0], d["route_waypoints"]
        stations = d["route_stations"]
    dist = np.linalg.norm(pose[:, None, :2] - w[None], axis=-1)
    j = dist.argmin(axis=1)
    reached = np.nonzero(j >= len(w) - 2)[0]
    reached = reached[reached >= start]
    end = int(reached[0]) if len(reached) else None
    return {"completed": end is not None, "time_s": None if end is None else (end - start + 1) * DT_S,
            "energy_kj": None if end is None else float(power[start:end + 1].sum() * DT_S),
            "energy_first16_kj": float(power[:CONTEXT].sum() * DT_S), "vx16": float(d["z1"][CONTEXT, 0]) if False else None,
            "length_m": float(stations[-1])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes-dir", default="artifacts/traverse/wp5_tracker_val_pt")
    ap.add_argument("--mini-cache", default="artifacts/traverse/wp5_tracker_val_pt_cache")
    ap.add_argument("--base-cache", default="artifacts/traverse/wp2_z2_cache_v6")
    ap.add_argument("--dynamics-checkpoints", nargs="+", required=True)
    ap.add_argument("--z1-extra-cache", nargs="*", default=None, help="per checkpoint ('' = none); unused here (cache is 17-D) but kept for symmetry")
    ap.add_argument("--policy", default="artifacts/traverse/wp3_tracker_v1")
    ap.add_argument("--power-calib", default="artifacts/traverse/wp4_power_calib/power_calib.json")
    ap.add_argument("--routes", default="artifacts/traverse/wp3_routes")
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--horizon-s", type=float, default=19.0)
    ap.add_argument("--from-rest", action="store_true",
                    help="seed the imagination with the frame-0 (at rest) state instead of frames 0-15; compare from frame 0")
    ap.add_argument("--out", default="artifacts/traverse/wp5_aligned_bench")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    keys = build_mini_cache(Path(args.episodes_dir), Path(args.base_cache), Path(args.mini_cache))
    start = 0 if args.from_rest else CONTEXT
    gt = {k: truth(Path(args.mini_cache) / f"{k}.npz", start) for k in keys}
    keys = [k for k in keys if gt[k]["completed"]]
    print(f"{len(keys)} tracker-driven held-out episodes reach the route end after frame {start}; "
          f"Chrono from frame {start}: time {np.mean([gt[k]['time_s'] for k in keys]):.2f} s, energy {np.mean([gt[k]['energy_kj'] for k in keys]):.1f} kJ "
          f"(launch frames 0-15: {np.mean([gt[k]['energy_first16_kj'] for k in keys]):.1f} kJ)", flush=True)
    entries = []
    for k in keys:
        with np.load(Path(args.mini_cache) / f"{k}.npz") as d:
            entries.append((k, {n: d[f"route_{n}"] for n in ("waypoints", "speeds", "headings", "stations")}))
    power_models = {kk: PowerModel.load(Path(args.power_calib), kk) for kk in KINDS}
    horizon = int(round(args.horizon_s / DT_S))
    t_gt = np.array([gt[k]["time_s"] for k in keys]); e_gt = np.array([gt[k]["energy_kj"] for k in keys])
    results = {"n": len(keys), "chrono_time_s": float(t_gt.mean()), "chrono_energy_kj": float(e_gt.mean()), "models": {}}
    print(f"\n{'model':32s} {'done':>5s} {'time bias':>9s} {'t corr':>6s} | {'head':>5s} {'corr':>5s} | {'act':>5s} {'corr':>5s} | {'state':>5s} {'corr':>5s} | {'pess(h,a)':>9s}")
    for ckpt in args.dynamics_checkpoints:
        name = Path(ckpt).parent.name
        cfg = merge_env_cfg({"num_envs": len(entries), "device": args.device, "auto_reset": False, "split": "val",
                             "dynamics_checkpoint": ckpt, "arena": args.arena, "cache": args.mini_cache, "routes": args.routes,
                             "fragment_steps_max": horizon})
        env = TraverseTrackingEnv(cfg, device=args.device, entries=entries)
        policy = load_policy(Path(args.policy), env, args.device)
        dummy = torch.full((len(entries), 1, 3), -1.0, device=env.device)
        res = rollout(env, policy, horizon, dummy, dummy, power_models, None, rest_start=args.from_rest)
        res = {k: v.cpu().numpy() for k, v in res.items() if not k.startswith("_")}
        ok = res["completed"]
        r = {"completed": float(ok.mean()), "time_bias": float((res["time_s"][ok] / t_gt[ok]).mean() - 1),
             "time_corr": float(np.corrcoef(res["time_s"][ok], t_gt[ok])[0, 1])}
        line = f"{name.replace('wp2_mapv2_', ''):32s} {ok.mean():5.2f} {r['time_bias']:+9.3f} {r['time_corr']:6.3f}"
        for est, fld in (("head", "energy_kj"), ("act", "energy_act_kj"), ("state", "energy_state_kj")):
            if fld in res:
                r[f"{est}_ratio"] = float(e_gt[ok].mean() / res[fld][ok].mean()); r[f"{est}_corr"] = float(np.corrcoef(res[fld][ok], e_gt[ok])[0, 1])
                line += f" | {r[f'{est}_ratio']:5.2f} {r[f'{est}_corr']:5.2f}"
            else:
                line += " |     -     -"
        pess = np.maximum(res["energy_kj"], res["energy_act_kj"])
        r["pess_ratio"] = float(e_gt[ok].mean() / pess[ok].mean()); line += f" | {r['pess_ratio']:9.2f}"
        print(line, flush=True)
        results["models"][name] = r
        np.savez_compressed(out / f"per_episode_{name}.npz", keys=np.array(keys), t_gt=t_gt, e_gt=e_gt, **res)
        del env, policy; torch.cuda.empty_cache()
    (out / "aligned_bench.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
