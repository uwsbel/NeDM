"""Static-scene check: does the scene map depend on WHEN in the episode it is built?

The v1.3 design encodes one vehicle-free scene map per episode on the assumption
that the layout is static. Here two maps are built per held-out episode from
disjoint halves of the recording (frames 0-199 and 200-399, same masked-median
procedure) and compared (a) as feature maps and (b) as the 8x8 ego windows the
dynamics model actually reads along the recorded trajectory, against two
references: the same map (1.0) and the map of a different layout (the floor).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nedm.traverse import nrd_data as D
from nedm.traverse import perception as P
from nedm.traverse.nrd_model import load_map_model
from nedm.traverse.storage import EpisodeReader
from traverse_wp2_encode_map import MAP_STAGE, MEDIAN_FRAMES, EpisodeMedian


class HalfMedian(EpisodeMedian):
    def __init__(self, keys, root, arena, half: int):
        super().__init__(keys, root, arena)
        self.half = half

    def __getitem__(self, index):
        key = self.keys[index]
        store, episode = key.split("__", 1)
        reader = EpisodeReader(self.root / store / episode)
        lo, hi = (0, reader.frames // 2 - 1) if self.half == 0 else (reader.frames // 2, reader.frames - 1)
        picks = np.linspace(lo, hi, MEDIAN_FRAMES).astype(int)
        fields = {n: i for i, n in enumerate(reader.states()[0])}
        rgb, elev, masks = [], [], []
        for t in picks:
            win = reader.read_window(int(t), 1)
            rgb.append(win["rgb"][0].astype(np.float32) / 255.0)
            elev.append(self._elevation(win["depth_mm"][0]))
            masks.append(self._vehicle_mask(win["states"][0], fields))
        reader.close()
        keep = ~np.stack(masks)
        stack_rgb, stack_elev = np.stack(rgb), np.stack(elev)
        never = ~keep.any(0)
        med_rgb = np.nanmedian(np.where(keep[..., None], stack_rgb, np.nan), axis=0)
        med_rgb[never] = np.median(stack_rgb, axis=0)[never]
        med_elev = np.nanmedian(np.where(keep, stack_elev, np.nan), axis=0)
        med_elev[never] = np.median(stack_elev, axis=0)[never]
        return {"index": index, "input": torch.from_numpy(
            np.concatenate([med_rgb.transpose(2, 0, 1), med_elev[None]]).astype(np.float32))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6")
    ap.add_argument("--encoder", default="artifacts/traverse/wp1_v6/ckpt_warmup.pt")
    ap.add_argument("--dynamics-checkpoint", default="artifacts/traverse/wp2_mapv2_index_amd/ckpt_best.pt")
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dev = "cuda"
    keys = D.load_cache_keys(Path(args.cache))
    _, val, _ = D.split_keys(keys)
    keys = val[: args.episodes]
    enc = P.Encoder(z_dim=256, n_q=8).to(dev)
    enc.load_state_dict(torch.load(args.encoder, map_location=dev, weights_only=False)["encoder"], strict=True)
    enc.eval(); stem = enc.backbone[:MAP_STAGE]
    maps = []
    for half in (0, 1):
        feats = torch.zeros(len(keys), 64, 64, 64, device=dev)
        loader = DataLoader(HalfMedian(keys, Path("artifacts/traverse"), Path(args.arena), half),
                            batch_size=16, num_workers=args.workers)
        with torch.no_grad():
            for item in loader:
                feats[item["index"]] = stem(item["input"].to(dev)).float()
        maps.append(feats)
    a, b = maps
    model, _, _ = load_map_model(args.dynamics_checkpoint, args.arena, dev)
    poses = torch.tensor(np.stack([np.load(Path(args.cache) / f"{k}.npz")["pose"] for k in keys]), device=dev)
    with torch.no_grad():
        ta, tb = model.cropper(a, poses), model.cropper(b, poses)
        roll = torch.roll(torch.arange(len(keys)), 1)
        tc = model.cropper(b[roll], poses)  # a different layout's map read at this vehicle's poses
        feat_cos = F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)
        feat_cos_other = F.cosine_similarity(a.flatten(1), b[roll].flatten(1), dim=1)
        pix_cos = F.cosine_similarity(a.permute(0, 2, 3, 1).reshape(-1, 64), b.permute(0, 2, 3, 1).reshape(-1, 64), dim=1)
        tok_cos = F.cosine_similarity(ta, tb, dim=-1)
        tok_cos_other = F.cosine_similarity(ta, tc, dim=-1)
        mu = ta.reshape(-1, 256).mean(0)
        tok_cos_c = F.cosine_similarity(ta - mu, tb - mu, dim=-1)
        tok_cos_c_other = F.cosine_similarity(ta - mu, tc - mu, dim=-1)
    print(f"episodes {len(keys)} (held-out); maps from frames 0-199 vs 200-399")
    print(f"whole-map feature cosine   same episode {feat_cos.mean():.4f}  (p5 {feat_cos.quantile(0.05):.4f})   different layout {feat_cos_other.mean():.4f}")
    print(f"per-cell feature cosine    same episode {pix_cos.mean():.4f}  (p5 {pix_cos.quantile(0.05):.4f})")
    print(f"ego-window token cosine    same episode {tok_cos.mean():.4f}  (p5 {tok_cos.quantile(0.05):.4f})   different layout {tok_cos_other.mean():.4f}")
    print(f"  mean-centred             same episode {tok_cos_c.mean():.4f}  (p5 {tok_cos_c.quantile(0.05):.4f})   different layout {tok_cos_c_other.mean():.4f}")
    print(f"token abs diff / token std: {((ta - tb).abs().mean() / ta.reshape(-1, 256).std(0).mean()):.3f}")


if __name__ == "__main__":
    main()
