"""Paired G3 analysis: per-channel rollout error and bootstrap CIs.

The triad's aggregate z1 MAE and its pose error disagree (joint wins the first,
loses the second), so the aggregate is hiding structure. Pose integrates only
vx, vy and yaw-rate; 8 of the 15 channels are tire Fz and spindle omega, which
are the terrain-dependent ones z2 is supposed to inform. This script re-runs
each saved checkpoint on the SAME held-out episodes and reports:

  * per-channel MAE at the rollout horizons (normalized and physical units),
  * per-episode pose error, so variant differences get a paired bootstrap CI.

No retraining -- it reads ckpt_best.pt from each run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nedm.traverse import nrd_data as D
from nedm.training.constants import STATE_FIELD_PRESETS
from traverse_wp2_train import (Batcher, TERRAIN_CHANNELS, WP2Model, build_tokens,
                                 integrate_pose)

STATE_FIELDS = STATE_FIELD_PRESETS["tire_normal_force_omega"]
POSE_CHANNELS = [0, 1, 6]  # vx, vy, yaw_rate -- the only channels dead reckoning reads


@torch.no_grad()
def rollout_detail(model, data: Batcher, norm: D.Normalizer, variant: str, context: int,
                   horizons: list[int], n_episodes: int, device: str, seed: int = 7) -> dict:
    model.eval()
    rng = np.random.default_rng(seed)
    eps = rng.choice(data.n_episodes, size=min(n_episodes, data.n_episodes), replace=False)
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    z1_gt, z2_gt = to_t(data.z1[eps]), to_t(data.z2[eps])
    act, priv, pose_gt = to_t(data.act[eps]), to_t(data.priv[eps]), to_t(data.pose[eps])
    z1_mean, z1_std = to_t(norm.z1_mean.astype(np.float32)), to_t(norm.z1_std.astype(np.float32))
    power_gt = to_t(data.power_raw[eps])[..., 0]
    p_mean, p_std = float(norm.power_mean[0]), float(norm.power_std[0])
    energy_pred = torch.zeros(len(eps), device=device)
    energy_gt = torch.zeros(len(eps), device=device)

    z1_hist, z2_hist = z1_gt[:, :context].clone(), z2_gt[:, :context].clone()
    pose = pose_gt[:, context - 1].clone()
    out: dict = {"episodes": len(eps)}

    for step in range(max(horizons)):
        window = slice(step, step + context)
        tokens = build_tokens(z1_hist[:, -context:], z2_hist[:, -context:],
                              priv[:, window], act[:, window], variant)
        pred_delta, pred_z2, pred_power = model(tokens)
        z1_next = z1_hist[:, -1] + pred_delta[:, -1]
        z1_hist = torch.cat([z1_hist, z1_next.unsqueeze(1)], dim=1)
        z2_hist = torch.cat(
            [z2_hist, (pred_z2[:, -1] if (variant == "joint" and pred_z2 is not None) else z2_hist[:, -1]).unsqueeze(1)],
            dim=1,
        )
        pose = integrate_pose(pose, z1_next * z1_std + z1_mean)
        energy_pred = energy_pred + (pred_power[:, -1, 0] * p_std + p_mean) * 0.05
        energy_gt = energy_gt + power_gt[:, context + step] * 0.05
        h = step + 1
        if h in horizons:
            gt = pose_gt[:, context - 1 + h]
            per_ep = (pose[:, :2] - gt[:, :2]).norm(dim=1)
            err_norm = (z1_hist[:, context - 1 + h] - z1_gt[:, context - 1 + h]).abs()
            out[f"pose_per_episode@{h}"] = per_ep.cpu().numpy().tolist()
            out[f"z1mae_per_episode@{h}"] = err_norm.mean(1).cpu().numpy().tolist()
            out[f"terrain_per_episode@{h}"] = err_norm[:, TERRAIN_CHANNELS].mean(1).cpu().numpy().tolist()
            out[f"energy_per_episode@{h}"] = (energy_pred - energy_gt).abs().cpu().numpy().tolist()
            out[f"channel_mae_norm@{h}"] = err_norm.mean(0).cpu().numpy().tolist()
            out[f"channel_mae_phys@{h}"] = (err_norm.mean(0) * z1_std).cpu().numpy().tolist()
    return out


def bootstrap_paired(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 11) -> dict:
    """CI on mean(a) - mean(b) over paired episodes (negative => a is better)."""
    rng = np.random.default_rng(seed)
    diff = a - b
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return {
        "delta_mean_m": float(diff.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "p_a_better": float((means < 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6")
    parser.add_argument("--runs", nargs="+",
                        default=["artifacts/traverse/wp2_g3_state_amd",
                                 "artifacts/traverse/wp2_g3_joint_amd",
                                 "artifacts/traverse/wp2_g3_priv_amd"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--eval-episodes", type=int, default=256)
    parser.add_argument("--out", default="artifacts/traverse/wp2_g3_analysis.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    keys = D.load_cache_keys(Path(args.cache))
    _, val_keys, _ = D.split_keys(keys)

    results: dict[str, dict] = {}
    val_cache: dict[bool, D.CacheSplit] = {}
    for run_dir in args.runs:
        run = Path(run_dir)
        cfg = json.loads((run / "config.json").read_text())
        ckpt = torch.load(run / "ckpt_best.pt", map_location=device, weights_only=False)
        variant = ckpt["variant"]
        need_z2 = variant == "joint"
        if need_z2 not in val_cache:
            val_cache[need_z2] = D.load_split(Path(args.cache), val_keys, with_z2=need_z2)
        split = val_cache[need_z2]
        norm = D.Normalizer.from_dict(cfg["normalization"])
        data = Batcher(split, norm, cfg["context"], need_z2)
        model = WP2Model(
            split.z1.shape[-1], split.z2.shape[-1] if need_z2 else 0,
            4 if variant == "priv" else 0, split.act.shape[-1], ckpt["config"],
        ).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        results[variant] = rollout_detail(model, data, norm, variant, cfg["context"],
                                          args.horizons, args.eval_episodes, device)
        results[variant]["step"] = ckpt["step"]
        print(f"{variant}: rolled out {results[variant]['episodes']} episodes "
              f"(ckpt step {ckpt['step']})", flush=True)

    h = max(args.horizons)
    print(f"\nper-channel MAE (physical units) at {h * 0.05:.1f} s")
    header = f"{'channel':<28}" + "".join(f"{v:>12}" for v in results)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(STATE_FIELDS):
        marker = " *" if i in POSE_CHANNELS else "  "
        row = f"{name + marker:<28}"
        for v in results:
            row += f"{results[v][f'channel_mae_phys@{h}'][i]:>12.4f}"
        print(row)
    print("  (* = read by dead reckoning)")

    comparisons = {}
    variants = list(results)
    families = ["z1mae", "terrain", "pose", "energy"]
    for family in families:
        print(f"\npaired bootstrap on {family} (negative delta => first is better)")
        for i, a in enumerate(variants):
            for b in variants[i + 1:]:
                for hh in args.horizons:
                    key = f"{family}:{a}_vs_{b}@{hh}"
                    stats = bootstrap_paired(
                        np.array(results[a][f"{family}_per_episode@{hh}"]),
                        np.array(results[b][f"{family}_per_episode@{hh}"]),
                    )
                    comparisons[key] = stats
                    flag = "" if stats["ci95"][0] * stats["ci95"][1] > 0 else "   (CI spans 0)"
                    print(f"  {key:<34} delta {stats['delta_mean_m']:+.4f}  "
                          f"CI95 [{stats['ci95'][0]:+.4f}, {stats['ci95'][1]:+.4f}]"
                          f"  P {stats['p_a_better']:.3f}{flag}")

    payload = {
        "state_fields": STATE_FIELDS,
        "pose_channels": POSE_CHANNELS,
        "comparisons": comparisons,
        "runs": {v: {k: val for k, val in r.items() if "_per_episode@" not in k}
                 for v, r in results.items()},
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
