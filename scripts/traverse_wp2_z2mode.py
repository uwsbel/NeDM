"""Does predicting z2 beat just HOLDING it? (plan 8.3 z2-prediction baselines)

On a static, fully-observed map the layout never changes, so "persistence"
(carry the last encoded z2 forward unchanged) is an unusually strong baseline
-- the plan says so explicitly. G4 showed the predicted latent decays fast, so
the question is whether the z2-prediction branch is earning its place at all.

Eval-time only: the SAME trained joint checkpoint is rolled out twice, once
feeding its own predicted z2 back in and once holding the last encoded z2
fixed. Paired over episodes, so the difference is attributable.
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
from traverse_wp2_train import (Batcher, POSE_CHANNELS, TERRAIN_CHANNELS, WP2Model,
                                build_tokens, integrate_pose)
from traverse_wp2_analyze import bootstrap_paired


@torch.no_grad()
def rollout(model, data, norm, context, horizons, eps, device, z2_mode):
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    z1_gt, z2_gt = to_t(data.z1[eps]), to_t(data.z2[eps])
    act, priv, pose_gt = to_t(data.act[eps]), to_t(data.priv[eps]), to_t(data.pose[eps])
    z1_mean, z1_std = to_t(norm.z1_mean.astype(np.float32)), to_t(norm.z1_std.astype(np.float32))
    z1_hist, z2_hist = z1_gt[:, :context].clone(), z2_gt[:, :context].clone()
    pose = pose_gt[:, context - 1].clone()
    held = z2_gt[:, context - 1].clone()
    out = {}
    for step in range(max(horizons)):
        window = slice(step, step + context)
        tokens = build_tokens(z1_hist[:, -context:], z2_hist[:, -context:],
                              priv[:, window], act[:, window], "joint")
        pred_delta, pred_z2, _ = model(tokens)
        z1_next = z1_hist[:, -1] + pred_delta[:, -1]
        z1_hist = torch.cat([z1_hist, z1_next.unsqueeze(1)], dim=1)
        nxt = held if z2_mode == "persist" else pred_z2[:, -1]
        z2_hist = torch.cat([z2_hist, nxt.unsqueeze(1)], dim=1)
        pose = integrate_pose(pose, z1_next * z1_std + z1_mean)
        h = step + 1
        if h in horizons:
            frame = context - 1 + h
            err = (z1_hist[:, frame] - z1_gt[:, frame]).abs()
            out[f"z1mae@{h}"] = err.mean(1).cpu().numpy()
            out[f"terrain@{h}"] = err[:, TERRAIN_CHANNELS].mean(1).cpu().numpy()
            out[f"posechan@{h}"] = err[:, POSE_CHANNELS].mean(1).cpu().numpy()
            out[f"pose_m@{h}"] = (pose[:, :2] - pose_gt[:, frame, :2]).norm(dim=1).cpu().numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6")
    parser.add_argument("--runs", nargs="+", default=["artifacts/traverse/wp2_g3b_joint_amd",
                                                      "artifacts/traverse/wp2_joint_s2_amd",
                                                      "artifacts/traverse/wp2_joint_s3_amd"])
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 40, 100])
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--out", default="artifacts/traverse/wp2_z2mode_readout.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    keys = D.load_cache_keys(Path(args.cache))
    _, val_keys, _ = D.split_keys(keys)
    split = D.load_split(Path(args.cache), val_keys, with_z2=True)

    payload = {}
    for run_dir in args.runs:
        run = Path(run_dir)
        cfg = json.loads((run / "config.json").read_text())
        ckpt = torch.load(run / "ckpt_best.pt", map_location=device, weights_only=False)
        norm = D.Normalizer.from_dict(cfg["normalization"])
        data = Batcher(split, norm, cfg["context"], True)
        model = WP2Model(split.z1.shape[-1], split.z2.shape[-1], 0, split.act.shape[-1],
                         ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        eps = np.random.default_rng(7).choice(data.n_episodes,
                                              size=min(args.episodes, data.n_episodes), replace=False)
        modes = {m: rollout(model, data, norm, cfg["context"], args.horizons, eps, device, m)
                 for m in ("predict", "persist")}
        print(f"\n=== {run.name}")
        print(f"{'metric':<16}{'predict':>10}{'persist':>10}{'delta':>10}   CI95 (predict − persist)")
        entry = {}
        for metric in ("z1mae", "terrain", "posechan", "pose_m"):
            for h in args.horizons:
                k = f"{metric}@{h}"
                a, b = modes["predict"][k], modes["persist"][k]
                st = bootstrap_paired(a, b)
                entry[k] = {"predict": float(a.mean()), "persist": float(b.mean()), **st}
                flag = "" if st["ci95"][0] * st["ci95"][1] > 0 else "  (spans 0)"
                print(f"{k:<16}{a.mean():>10.4f}{b.mean():>10.4f}{st['delta_mean_m']:>+10.4f}"
                      f"   [{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}]{flag}")
        payload[run.name] = entry
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}   (negative delta => PREDICTING z2 is better)")


if __name__ == "__main__":
    main()
