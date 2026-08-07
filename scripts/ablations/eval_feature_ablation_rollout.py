#!/usr/bin/env python3
"""Low-noise, paired open-loop rollout comparison for the feature-ablation runs.

Why this exists: the trainer's `rollout_sel` uses only 12 episodes/domain and is
recorded once per epoch, and S = min-over-80-epochs of it. A min over 80 noisy
draws is an outlier statistic -- it partly measures which run got a lucky epoch.
This script re-evaluates the SELECTED checkpoints on many more episodes, so the
comparison is about the models rather than the selection noise.

Paired by construction: every model is scored on the SAME episodes at the SAME
horizon. That is valid across feature sets because the body7 caches symlink their
`rollout`/`episodes` arrays to the 15-D caches' real files -- the ground-truth
poses and episode list are literally the same bytes; only the state columns each
model consumes differ.

Each checkpoint carries its own config, so the right cache dirs and terrain ids
are read from it (no hardcoding of which run is 7-D or unconditioned).

Reports, per model per domain:
  errdist   pooled xy_rmse / mean_gt_distance -- the same definition the trainer's
            rollout_sel uses, so it is comparable to S.
  per-episode errdist median + IQR, and a paired win-rate vs the baseline model.

Usage:
    python scripts/ablations/eval_feature_ablation_rollout.py \
        --ckpts baseline=artifacts/.../L8_H8_E256_ctx128/checkpoints/best_val.pt \
                ab1=artifacts/.../L8_H8_E256_ctx128_no_onehot/checkpoints/best_val.pt \
        --episodes 100 --horizon 10.0
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nedm.training.dataset import load_rollout_split  # noqa: E402
from nedm.training.model import HMMWVDynamicsModel  # noqa: E402


def load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta, config = ck["metadata"], ck["config"]
    terrain_cfg = config.get("terrain_conditioning", {})
    enabled = bool(terrain_cfg.get("enabled", False))
    terrains = [str(t) for t in terrain_cfg.get("terrains", [])] if enabled else []
    model = HMMWVDynamicsModel(
        state_dim=len(meta["state_fields"]),
        action_dim=len(meta["action_fields"]),
        target_dim=len(meta["state_fields"]),
        transformer_cfg=config["model"],
        normalization=meta["normalization"],
        num_terrains=len(terrains),
    )
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    return {
        "model": model,
        "metadata": meta,
        "config": config,
        "terrains": terrains,
        "terrain_to_id": {name: i for i, name in enumerate(terrains)},
        "seq": int(config["model"]["block_size"]),
        "state_index": {f: i for i, f in enumerate(meta["state_fields"])},
        "dt": float(meta["dt_s"]),
    }


def select_episodes(episodes: list[dict], max_episodes: int) -> list[dict]:
    """Round-robin over scenario families -- identical to Trainer._select_rollout_episodes."""
    by_family: dict[str, list[dict]] = {}
    for ep in episodes:
        by_family.setdefault(ep["scenario_family"], []).append(ep)
    families = sorted(by_family)
    selected: list[dict] = []
    index = 0
    while len(selected) < max_episodes and families:
        family = families[index % len(families)]
        bucket = by_family[family]
        if bucket:
            selected.append(bucket.pop(0))
        if not bucket:
            families.remove(family)
            index -= 1
        index += 1
    return selected


@torch.no_grad()
def rollout_episode(bundle: dict, episode: dict, horizon_steps: int, terrain_id: int | None):
    """Open-loop rollout with recorded actions; mirrors Trainer._rollout_episode."""
    model, seq, dt = bundle["model"], bundle["seq"], bundle["dt"]
    device = bundle["model"].state_mean.device
    states = torch.from_numpy(episode["states"]).to(device)
    actions = torch.from_numpy(episode["actions"]).to(device)
    poses = torch.from_numpy(episode["rollout"]).to(device)
    if states.shape[0] <= seq + 1:
        return None
    steps = min(horizon_steps, states.shape[0] - seq)
    if steps <= 0:
        return None

    idx = bundle["state_index"]
    yaw_i, vx_i, vy_i = idx["yaw_rate_radps"], idx["vel_body_x_mps"], idx["vel_body_y_mps"]
    hist_s, hist_a = states[:seq].clone(), actions[:seq].clone()
    pose = poses[seq - 1].clone()

    sq_err = 0.0
    for k in range(steps):
        delta = model.predict_delta(
            hist_s[-seq:].unsqueeze(0), hist_a[-seq:].unsqueeze(0), terrain=terrain_id
        )[:, -1, :].squeeze(0)
        next_state = hist_s[-1] + delta
        yaw = pose[2] + dt * next_state[yaw_i]
        vx_w = torch.cos(yaw) * next_state[vx_i] - torch.sin(yaw) * next_state[vy_i]
        vy_w = torch.sin(yaw) * next_state[vx_i] + torch.cos(yaw) * next_state[vy_i]
        pose = torch.stack([pose[0] + dt * vx_w, pose[1] + dt * vy_w, yaw])
        sq_err += float(((pose[:2] - poses[seq + k][:2]) ** 2).sum())
        if seq + k < actions.shape[0]:
            hist_a = torch.cat([hist_a, actions[seq + k].unsqueeze(0)], 0)
        hist_s = torch.cat([hist_s, next_state.unsqueeze(0)], 0)

    gt_xy = poses[seq : seq + steps, :2]
    dist = float((gt_xy[1:] - gt_xy[:-1]).pow(2).sum(dim=-1).sqrt().sum()) if steps >= 2 else 0.0
    return {"sq_err": sq_err, "steps": steps, "dist": dist}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True, help="name=path pairs; the FIRST is the baseline")
    ap.add_argument("--episodes", type=int, default=100, help="episodes per domain (trainer uses 12)")
    ap.add_argument("--horizon", type=float, default=10.0, help="rollout horizon in seconds")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=None, help="Optional path to dump the full result dict")
    args = ap.parse_args()

    bundles = {}
    for spec in args.ckpts:
        name, _, path = spec.partition("=")
        bundles[name] = load_checkpoint(Path(path), args.device)

    results: dict[str, dict[str, Any]] = {}
    per_episode: dict[str, dict[str, list[float]]] = {}

    for name, bundle in bundles.items():
        domains = bundle["config"]["rollout_eval"]["datasets"]
        horizon_steps = max(1, int(round(args.horizon / bundle["dt"])))
        results[name], per_episode[name] = {}, {}
        for domain in domains:
            dname = str(domain["name"])
            root = (REPO_ROOT / domain["processed_dataset_dir"]).resolve()
            split = load_rollout_split(root, str(domain.get("split", "val")))
            episodes = select_episodes(split["episodes"], args.episodes)
            terrain_id = bundle["terrain_to_id"].get(str(domain.get("terrain", dname))) if bundle["terrains"] else None

            sq_err = dist_sum = 0.0
            steps_total = ep_count = 0
            ep_errdists: list[float] = []
            for ep in episodes:
                out = rollout_episode(bundle, ep, horizon_steps, terrain_id)
                if out is None:
                    continue
                sq_err += out["sq_err"]
                steps_total += out["steps"]
                dist_sum += out["dist"]
                ep_count += 1
                if out["dist"] > 1e-6:
                    ep_errdists.append(math.sqrt(out["sq_err"] / out["steps"]) / out["dist"])

            xy_rmse = math.sqrt(sq_err / steps_total) if steps_total else float("nan")
            mean_dist = dist_sum / max(ep_count, 1)
            results[name][dname] = {
                "errdist": xy_rmse / mean_dist if mean_dist > 1e-6 else float("nan"),
                "xy_rmse_m": xy_rmse,
                "mean_dist_m": mean_dist,
                "episodes": ep_count,
                "weight": float(domain.get("weight", 1.0)),
            }
            per_episode[name][dname] = ep_errdists
            print(f"{name:12s} {dname:5s} errdist={results[name][dname]['errdist']:.4f} "
                  f"({ep_count} eps, {xy_rmse:.3f} m rmse)", flush=True)

        terms = [d["weight"] * d["errdist"] for d in results[name].values() if math.isfinite(d["errdist"])]
        weights = sum(d["weight"] for d in results[name].values() if math.isfinite(d["errdist"]))
        results[name]["sel"] = sum(terms) / weights if weights else float("nan")

    base = list(bundles)[0]
    print(f"\n{'model':14s} {'sel':>8s} {'d_vs_base':>10s} " +
          " ".join(f"{d:>9s}" for d in results[base] if d != "sel"))
    print("-" * 70)
    for name in bundles:
        row = f"{name:14s} {results[name]['sel']:8.4f} {results[name]['sel'] - results[base]['sel']:+10.4f} "
        row += " ".join(f"{results[name][d]['errdist']:9.4f}" for d in results[name] if d != "sel")
        print(row)

    # Paired per-episode view: same episodes, so a win-rate is meaningful.
    print(f"\nPaired per-episode errdist (median [IQR], win-rate vs {base}):")
    for name in bundles:
        if name == base:
            continue
        for dname in per_episode[name]:
            mine, theirs = per_episode[name][dname], per_episode[base][dname]
            n = min(len(mine), len(theirs))
            if not n:
                continue
            wins = sum(1 for a, b in zip(mine[:n], theirs[:n]) if a < b)
            q = st.quantiles(mine[:n], n=4)
            qb = st.quantiles(theirs[:n], n=4)
            print(f"  {name:12s} {dname:5s} {st.median(mine[:n]):.4f} [{q[0]:.4f},{q[2]:.4f}]  vs  "
                  f"base {st.median(theirs[:n]):.4f} [{qb[0]:.4f},{qb[2]:.4f}]   "
                  f"win {wins}/{n} ({100*wins/n:.0f}%)")

    if args.json:
        Path(args.json).write_text(json.dumps({"pooled": results, "per_episode": per_episode}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
