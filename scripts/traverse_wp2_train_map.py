"""WP2 spatial-token runs: ego-crop of the static scene map instead of pooled z2.

Same backbone, data, split and budget as the G3b triad; only the sensor token
changes. Two modes, and the comparison between them is the experiment:

  index    the scene map is encoded once and CROPPED at the dead-reckoned pose
           each step. Nothing is predicted forward, so there is no latent drift
           -- the failure mode G4 measured simply cannot occur.
  predict  a head predicts the next crop and it is fed back autoregressively,
           preserving the "NRD rolls out its sensor latent" claim.

index is the strong baseline (the layout is static, so indexing is exact). If
predict cannot beat it, the prediction branch is not earning its place here --
the same question the z2 persistence test asked of the pooled token, which
predict won at 1 s.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nedm.traverse import nrd_data as D
from nedm.traverse.nrd_model import WP2MapModel  # noqa: F401 (checkpoint class)
from nedm.training.model_transformer import ContinuousTransformer, TransformerConfig
from traverse_wp2_train import DT_S, POSE_CHANNELS, TERRAIN_CHANNELS, integrate_pose


class MapBatcher:
    def __init__(self, split: D.CacheSplit, norm: D.Normalizer, context: int, maps: np.ndarray):
        self.context = context
        self.z1 = ((split.z1 - norm.z1_mean) / norm.z1_std).astype(np.float32)
        self.act = ((split.act - norm.act_mean) / norm.act_std).astype(np.float32)
        self.power = ((split.power - norm.power_mean) / norm.power_std).astype(np.float32)
        self.power_raw = split.power.astype(np.float32)
        self.pose = split.pose.astype(np.float32)
        self.maps = maps  # (N, C, 64, 64) float16
        self.n_episodes, self.n_frames = split.z1.shape[0], split.z1.shape[1]

    def sample(self, rng, batch: int, device: str):
        ep = rng.integers(0, self.n_episodes, batch)
        t0 = rng.integers(0, self.n_frames - self.context, batch)
        idx = t0[:, None] + np.arange(self.context + 1)[None, :]
        out = {}
        for name, src in (("z1", self.z1), ("act", self.act), ("power", self.power),
                          ("pose", self.pose)):
            out[name] = torch.from_numpy(src[ep[:, None], idx]).to(device, non_blocking=True)
        out["map"] = torch.from_numpy(self.maps[ep]).to(device, non_blocking=True).float()
        return out


def step_loss(model, batch, mode: str):
    token = model.cropper(batch["map"], batch["pose"])
    delta, power, token_next = model(batch["z1"][:, :-1], token[:, :-1], batch["act"][:, :-1])
    loss = F.huber_loss(delta, batch["z1"][:, 1:] - batch["z1"][:, :-1], delta=1.0)
    parts = {"z1": float(loss.detach())}
    lp = F.huber_loss(power, batch["power"][:, 1:], delta=1.0)
    loss = loss + lp
    parts["power"] = float(lp.detach())
    if mode == "predict" and token_next is not None:
        target = ((token[:, 1:] - model.tok_mean) / model.tok_std).detach()
        lt = F.mse_loss(token_next, target)
        loss = loss + lt
        parts["token"] = float(lt.detach())
    return loss, parts


@torch.no_grad()
def fit_token_stats(model, data: MapBatcher, batch: int, n_batches: int, device: str,
                    seed: int) -> None:
    rng = np.random.default_rng(seed)
    crops = []
    for _ in range(n_batches):
        b = data.sample(rng, batch, device)
        crops.append(model.cropper(b["map"], b["pose"]).reshape(-1, model.token_dim))
    t = torch.cat(crops)
    model.tok_mean.copy_(t.mean(0))
    model.tok_std.copy_(t.std(0).clamp_min(1e-3))
    print(f"token stats over {t.shape[0]} crops: mean|.|={float(model.tok_mean.abs().mean()):.4f} "
          f"std={float(model.tok_std.mean()):.4f}", flush=True)


@torch.no_grad()
def rollout_eval(model, data: MapBatcher, norm, mode: str, context: int, horizons: list[int],
                 n_episodes: int, device: str, seed: int = 7,
                 crop_pose: str = "deadreckon") -> dict:
    """crop_pose: where the index crop is taken during rollout. "deadreckon" is the
    honest setting (pose integrated from predicted z1); "gt" reads the map at the
    true pose and isolates how much long-horizon error is *reading the wrong place*
    rather than the token lacking information (the pose-drift test)."""
    model.eval()
    eps = np.random.default_rng(seed).choice(
        data.n_episodes, size=min(n_episodes, data.n_episodes), replace=False)
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    z1_gt, act = to_t(data.z1[eps]), to_t(data.act[eps])
    pose_gt, maps = to_t(data.pose[eps]), to_t(data.maps[eps]).float()
    power_gt = to_t(data.power_raw[eps])[..., 0]
    z1_mean, z1_std = to_t(norm.z1_mean.astype(np.float32)), to_t(norm.z1_std.astype(np.float32))
    p_mean, p_std = float(norm.power_mean[0]), float(norm.power_std[0])

    z1_hist = z1_gt[:, :context].clone()
    pose_hist = pose_gt[:, :context].clone()
    token_hist = model.cropper(maps, pose_hist)
    pose = pose_gt[:, context - 1].clone()
    cv_pose = pose_gt[:, context - 1].clone()
    cv_state = z1_gt[:, context - 1] * z1_std + z1_mean
    e_pred = torch.zeros(len(eps), device=device)
    e_gt = torch.zeros(len(eps), device=device)
    results: dict[str, float] = {}

    for step in range(max(horizons)):
        window = slice(step, step + context)
        delta, power, token_next = model(z1_hist[:, -context:], token_hist[:, -context:],
                                         act[:, window])
        z1_next = z1_hist[:, -1] + delta[:, -1]
        z1_hist = torch.cat([z1_hist, z1_next.unsqueeze(1)], dim=1)
        pose = integrate_pose(pose, z1_next * z1_std + z1_mean)
        cv_pose = integrate_pose(cv_pose, cv_state)
        # index: re-crop at the pose we just dead-reckoned (or the true pose for
        # the pose-drift test). predict: the head's normalized output, de-normalized.
        if mode == "index":
            crop_at = pose_gt[:, context + step] if crop_pose == "gt" else pose
            nxt = model.cropper(maps, crop_at.unsqueeze(1))[:, 0]
        else:
            nxt = token_next[:, -1] * model.tok_std + model.tok_mean
        token_hist = torch.cat([token_hist, nxt.unsqueeze(1)], dim=1)

        kw = power[:, -1, 0] * p_std + p_mean
        e_pred = e_pred + kw * DT_S
        e_gt = e_gt + power_gt[:, context + step] * DT_S
        h = step + 1
        if h in horizons:
            frame = context - 1 + h
            gt = pose_gt[:, frame]
            err = (z1_hist[:, frame] - z1_gt[:, frame]).abs()
            results[f"z1_mae_norm@{h}"] = float(err.mean())
            results[f"terrain_mae_norm@{h}"] = float(err[:, TERRAIN_CHANNELS].mean())
            results[f"pose_chan_mae_norm@{h}"] = float(err[:, POSE_CHANNELS].mean())
            results[f"pose_err_m@{h}"] = float((pose[:, :2] - gt[:, :2]).norm(dim=1).mean())
            results[f"cv_pose_err_m@{h}"] = float((cv_pose[:, :2] - gt[:, :2]).norm(dim=1).mean())
            results[f"yaw_err_deg@{h}"] = float(torch.rad2deg(
                torch.abs((pose[:, 2] - gt[:, 2] + math.pi) % (2 * math.pi) - math.pi)).mean())
            results[f"energy_err_kj@{h}"] = float((e_pred - e_gt).abs().mean())
            if mode == "predict":
                true_token = model.cropper(maps, pose_gt[:, frame].unsqueeze(1))[:, 0]
                results[f"token_cos@{h}"] = float(
                    F.cosine_similarity(token_hist[:, frame], true_token, dim=-1).mean())
    model.train()
    return results


def load_maps(cache: Path, keys: list[str], key: str = "map") -> np.ndarray:
    return np.stack([np.load(cache / f"{k}.npz")[key] for k in keys])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--map-mode", choices=["index", "predict"], required=True)
    ap.add_argument("--map-key", default="map")
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--token-dim", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-embd", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-episodes", type=int, default=256)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 40, 100])
    ap.add_argument("--selection", default="z1_mae_norm")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--max-train-episodes", type=int, default=0)
    ap.add_argument("--init-from", default="",
                    help="ckpt_best.pt of a finished run to start from (two-stage predict / eval-only)")
    ap.add_argument("--freeze-cropper", action="store_true",
                    help="freeze the map projection so the prediction target is stationary")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="also freeze backbone + z1/power heads: only the token head trains")
    ap.add_argument("--token-stat-batches", type=int, default=64)
    ap.add_argument("--eval-only", action="store_true",
                    help="no training: roll out --init-from under both crop-pose settings "
                         "and write posedrift_readout.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    keys = D.load_cache_keys(cache)
    train_keys, val_keys, test_keys = D.split_keys(keys)
    if args.max_train_episodes:
        train_keys = train_keys[: args.max_train_episodes]
        val_keys = val_keys[: max(args.eval_episodes, 8)]
    print(f"split: {len(train_keys)}/{len(val_keys)}/{len(test_keys)} (test untouched)", flush=True)

    t0 = time.time()
    train_split = D.load_split(cache, train_keys, with_z2=False)
    val_split = D.load_split(cache, val_keys, with_z2=False)
    train_maps = load_maps(cache, train_keys, args.map_key)
    val_maps = load_maps(cache, val_keys, args.map_key)
    print(f"loaded cache + maps in {time.time() - t0:.1f}s  maps {train_maps.shape}", flush=True)

    payload = torch.load(args.init_from, map_location="cpu") if args.init_from else None
    norm = (D.Normalizer.from_dict(payload["normalization"]) if payload
            else D.Normalizer.fit(train_split))
    z1_dim, act_dim = train_split.z1.shape[-1], train_split.act.shape[-1]
    train_data = MapBatcher(train_split, norm, args.context, train_maps)
    val_data = MapBatcher(val_split, norm, args.context, val_maps)
    del train_split, val_split

    cfg = {"block_size": args.context, "n_layer": args.n_layer, "n_head": args.n_head,
           "n_embd": args.n_embd, "dropout": args.dropout, "bias": False,
           "head_hidden_dim": args.n_embd}
    model = WP2MapModel(z1_dim, act_dim, cfg, Path(args.arena), args.token_dim,
                        predict_token=(args.map_mode == "predict")).to(device)
    if payload is not None:
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        print(f"init from {args.init_from} (step {payload.get('step')}) "
              f"missing={missing} unexpected={unexpected}", flush=True)
    if args.map_mode == "predict":
        fit_token_stats(model, train_data, args.batch, args.token_stat_batches, device,
                        args.seed + 2)
    if args.freeze_cropper:
        for p in model.cropper.parameters():
            p.requires_grad_(False)
    if args.freeze_backbone:
        for module in (model.backbone, model.state_head, model.power_head):
            for p in module.parameters():
                p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"map-mode={args.map_mode} token_dim={args.token_dim} params={n_params/1e6:.2f}M "
          f"trainable={n_train/1e6:.2f}M", flush=True)

    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        p = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(p, 1.0)))

    (out_dir / "config.json").write_text(json.dumps(
        {**vars(args), "model": cfg, "n_params": n_params,
         "split_counts": [len(train_keys), len(val_keys), len(test_keys)],
         "normalization": norm.to_dict()}, indent=2))
    if args.eval_only:
        assert payload is not None, "--eval-only needs --init-from"
        readout = {"init_from": args.init_from, "episodes": args.eval_episodes,
                   "map_mode": args.map_mode}
        for cp in ("deadreckon", "gt"):
            readout[cp] = rollout_eval(model, val_data, norm, args.map_mode, args.context,
                                       args.horizons, args.eval_episodes, device, crop_pose=cp)
            print(cp, json.dumps(readout[cp]), flush=True)
        (out_dir / "posedrift_readout.json").write_text(json.dumps(readout, indent=2))
        return

    log = out_dir / "train_log.jsonl"; log.write_text("")
    rng = np.random.default_rng(args.seed)
    best = {"metric": float("inf"), "step": -1}
    start = time.time()

    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss, parts = step_loss(model, train_data.sample(rng, args.batch, device), args.map_mode)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if (step + 1) % 100 == 0:
            with log.open("a") as fh:
                fh.write(json.dumps({"phase": "train", "step": step + 1,
                                     "loss": float(loss.detach()), **parts,
                                     "sps": (step + 1) * args.batch / (time.time() - start)}) + "\n")
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            vr = np.random.default_rng(args.seed + 1)
            with torch.no_grad():
                vls = [float(step_loss(model, val_data.sample(vr, args.batch, device),
                                       args.map_mode)[0]) for _ in range(args.val_batches)]
            m = rollout_eval(model, val_data, norm, args.map_mode, args.context,
                             args.horizons, args.eval_episodes, device)
            sel = m[f"{args.selection}@{max(args.horizons)}"]
            rec = {"phase": "val", "step": step + 1, "val_loss": float(np.mean(vls)),
                   "selection": sel, **m}
            with log.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            if sel < best["metric"]:
                best = {"metric": sel, "step": step + 1, **m}
                torch.save({"model": model.state_dict(), "config": cfg, "map_mode": args.map_mode,
                            "normalization": norm.to_dict(), "step": step + 1, "metrics": m},
                           out_dir / "ckpt_best.pt")
    (out_dir / "g3_readout.json").write_text(json.dumps(
        {"variant": f"map-{args.map_mode}", "best": best, "wall_s": time.time() - start,
         "split_counts": [len(train_keys), len(val_keys), len(test_keys)]}, indent=2))
    print(f"done: best {args.selection} {best['metric']:.4f} @ step {best['step']}", flush=True)


if __name__ == "__main__":
    main()
