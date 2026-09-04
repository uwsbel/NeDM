"""G4 cross-modal: does the AUTONOMOUSLY PREDICTED z2 still decode the scene?

Plan section 12.1 G4 -- "decoded vehicle blob tracks dead-reckoned pose;
per-class object permanence over autonomous z2 rollouts". The frozen WP1
LatentProbe (trained on ENCODED z2) is applied to PREDICTED z2, which is the
cross-modal test: the dynamics model never saw the probe, and the probe never
saw a predicted latent.

Three readings per horizon:
  * blob_vs_gt      decoded vehicle centre from predicted z2 vs true pose
  * blob_vs_deadrec decoded centre vs the pose dead-reckoned from predicted z1
                    -- the two branches agreeing is the cross-modal claim
  * bev_iou         decoded occupancy vs the analytic layout raster
                    (object permanence), with the same quantity decoded from
                    the ENCODED z2 as the no-prediction reference

Per-class permanence is not reported for the global latent: WP1 established
z2 carries essentially no rock content (IoU 0.005, recall 0.007), so there is
nothing for it to lose. That question belongs to the spatial-map path.
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
from nedm.traverse import perception as P
from nedm.traverse.storage import EpisodeReader
from nedm.traverse.terrain import TerrainMap
from traverse_wp2_train import Batcher, WP2Model, build_tokens, integrate_pose


def episode_dir(traverse_root: Path, key: str) -> Path:
    store, episode = key.split("__", 1)
    return traverse_root / store / episode


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6")
    parser.add_argument("--run", default="artifacts/traverse/wp2_g3b_joint_amd")
    parser.add_argument("--probes", default="artifacts/traverse/wp1_v6/ckpt_probes.pt")
    parser.add_argument("--arena", default="assets/traverse/arena_v1")
    parser.add_argument("--traverse-root", default="artifacts/traverse")
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 40, 100])
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--out", default="artifacts/traverse/wp2_g4_readout.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = Path(args.run)
    cfg = json.loads((run / "config.json").read_text())
    ckpt = torch.load(run / "ckpt_best.pt", map_location=device, weights_only=False)
    if ckpt["variant"] != "joint":
        raise SystemExit("G4 needs the joint run: only it predicts z2")

    keys = D.load_cache_keys(Path(args.cache))
    _, val_keys, _ = D.split_keys(keys)
    rng = np.random.default_rng(7)
    picked = rng.choice(len(val_keys), size=min(args.episodes, len(val_keys)), replace=False)
    picked_keys = [val_keys[i] for i in picked]

    split = D.load_split(Path(args.cache), picked_keys, with_z2=True)
    norm = D.Normalizer.from_dict(cfg["normalization"])
    context = cfg["context"]
    data = Batcher(split, norm, context, True)

    model = WP2Model(split.z1.shape[-1], split.z2.shape[-1], 0, split.act.shape[-1],
                     ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    probe = P.LatentProbe(z_dim=split.z2.shape[-1]).to(device)
    probe.load_state_dict(torch.load(args.probes, map_location=device,
                                     weights_only=False)["z2_probe"], strict=True)
    probe.eval()

    tmap = TerrainMap.from_dir(Path(args.arena))
    half = tmap.size_m / 2.0
    bev_gt = []
    for key in picked_keys:
        with (episode_dir(Path(args.traverse_root), key) / "meta.json").open() as handle:
            layout = json.load(handle)["layout"]
        bev_gt.append(P.bev_occupancy(layout, tmap.size_m, grid=P.BEV_GRID))
    bev_gt_t = torch.from_numpy(np.stack(bev_gt).astype(np.float32)).to(device)

    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    z1_gt, z2_gt = to_t(data.z1), to_t(data.z2)
    act, priv, pose_gt = to_t(data.act), to_t(data.priv), to_t(data.pose)
    z1_mean, z1_std = to_t(norm.z1_mean.astype(np.float32)), to_t(norm.z1_std.astype(np.float32))
    z2_mean, z2_std = to_t(norm.z2_mean.astype(np.float32)), to_t(norm.z2_std.astype(np.float32))

    z1_hist, z2_hist = z1_gt[:, :context].clone(), z2_gt[:, :context].clone()
    pose = pose_gt[:, context - 1].clone()
    results: dict[str, float] = {}

    def decode(z2_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bev_logits, pose4 = probe(z2_norm * z2_std + z2_mean)
        centre = pose4[:, :2] * half
        return bev_logits, centre

    for step in range(max(args.horizons)):
        window = slice(step, step + context)
        tokens = build_tokens(z1_hist[:, -context:], z2_hist[:, -context:],
                              priv[:, window], act[:, window], "joint")
        pred_delta, pred_z2, _ = model(tokens)
        z1_next = z1_hist[:, -1] + pred_delta[:, -1]
        z1_hist = torch.cat([z1_hist, z1_next.unsqueeze(1)], dim=1)
        z2_hist = torch.cat([z2_hist, pred_z2[:, -1].unsqueeze(1)], dim=1)
        pose = integrate_pose(pose, z1_next * z1_std + z1_mean)

        h = step + 1
        if h in args.horizons:
            frame = context - 1 + h
            bev_pred, centre_pred = decode(z2_hist[:, frame])
            bev_enc, centre_enc = decode(z2_gt[:, frame])
            gt_xy = pose_gt[:, frame, :2]

            results[f"blob_vs_gt_m@{h}"] = float((centre_pred - gt_xy).norm(dim=1).mean())
            results[f"blob_vs_deadrec_m@{h}"] = float((centre_pred - pose[:, :2]).norm(dim=1).mean())
            results[f"blob_encoded_vs_gt_m@{h}"] = float((centre_enc - gt_xy).norm(dim=1).mean())
            results[f"deadrec_vs_gt_m@{h}"] = float((pose[:, :2] - gt_xy).norm(dim=1).mean())
            for name, logits in (("pred", bev_pred), ("encoded", bev_enc)):
                hit = ((torch.sigmoid(logits) > 0.5) & (bev_gt_t > 0.5)).sum()
                union = ((torch.sigmoid(logits) > 0.5) | (bev_gt_t > 0.5)).sum()
                results[f"bev_iou_{name}@{h}"] = float(hit / union.clamp(min=1))
            results[f"z2_cos@{h}"] = float(
                torch.nn.functional.cosine_similarity(z2_hist[:, frame], z2_gt[:, frame], dim=-1).mean()
            )

    print(f"G4 cross-modal ({len(picked_keys)} held-out episodes, run {run.name})\n")
    rows = ["z2_cos", "blob_vs_gt_m", "blob_encoded_vs_gt_m", "blob_vs_deadrec_m",
            "deadrec_vs_gt_m", "bev_iou_pred", "bev_iou_encoded"]
    head = f"{'metric':<24}" + "".join(f"{h * 0.05:>10.1f}s" for h in args.horizons)
    print(head); print("-" * len(head))
    for row in rows:
        print(f"{row:<24}" + "".join(f"{results[f'{row}@{h}']:>11.4f}" for h in args.horizons))
    Path(args.out).write_text(json.dumps({"run": str(run), "episodes": len(picked_keys),
                                          "metrics": results}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
