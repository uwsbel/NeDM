"""WP2 G3 triad: matched NRD dynamics runs on cached traverse latents.

Three variants share the backbone, data, split and budget and differ only in
what the token carries (plan section 8.3):

    state   [z1, a]              -- the RQ2 comparator
    joint   [z1, z2, a]          -- frozen WP1 encoder latent
    priv     [z1, (x,y,yaw), a]              -- localization-only privileged row
    privterr [z1, (x,y,yaw), terrain, a]     -- the real ceiling (plan 8.3)

Selection follows repo lore: judge on rollouts, not val_loss. The checkpoint
metric is dead-reckoned pose error at 1.0 s of AUTONOMOUS rollout -- z1 and
(for joint) z2 are both predicted recursively; only the actions come from the
episode. The priv row is fed ground-truth pose every step, which is what makes
it a ceiling rather than a fourth model.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nedm.traverse import nrd_data as D
from nedm.training.model_transformer import ContinuousTransformer, TransformerConfig

DT_S = 0.05
VX, VY, YAW_RATE = 0, 1, 6  # indices into the tire_normal_force_omega preset
POSE_CHANNELS = [0, 1, 6]  # the only channels dead reckoning reads
TERRAIN_CHANNELS = [2, 3, 4, 5, 7, 8, 9, 10]  # attitude + tire normal loads


class WP2Model(nn.Module):
    def __init__(self, z1_dim: int, z2_dim: int, priv_dim: int, act_dim: int, cfg: dict) -> None:
        super().__init__()
        self.z1_dim, self.z2_dim, self.priv_dim = z1_dim, z2_dim, priv_dim
        self.backbone = ContinuousTransformer(
            TransformerConfig(
                input_dim=z1_dim + z2_dim + priv_dim + act_dim,
                block_size=int(cfg["block_size"]),
                n_layer=int(cfg["n_layer"]),
                n_head=int(cfg["n_head"]),
                n_embd=int(cfg["n_embd"]),
                dropout=float(cfg["dropout"]),
                bias=bool(cfg["bias"]),
            )
        )
        hidden = int(cfg["head_hidden_dim"])
        n_embd = int(cfg["n_embd"])
        self.state_head = nn.Sequential(nn.Linear(n_embd, hidden), nn.GELU(), nn.Linear(hidden, z1_dim))
        self.z2_head = (
            nn.Sequential(nn.Linear(n_embd, hidden), nn.GELU(), nn.Linear(hidden, z2_dim))
            if z2_dim
            else None
        )
        # Auxiliary and identical across variants: power is supervised out of the
        # shared feature, never fed back into the token (plan section 4), so the
        # state-only baseline stays exactly 15-D for RQ2.
        self.power_head = nn.Sequential(nn.Linear(n_embd, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        feat = self.backbone(tokens)
        return (
            self.state_head(feat),
            (self.z2_head(feat) if self.z2_head is not None else None),
            self.power_head(feat),
        )


def build_tokens(z1: torch.Tensor, z2: torch.Tensor, priv: torch.Tensor, act: torch.Tensor,
                 variant: str) -> torch.Tensor:
    parts = [z1]
    if variant == "joint":
        parts.append(z2)
    elif variant in ("priv", "privterr"):
        parts.append(priv)
    parts.append(act)
    return torch.cat(parts, dim=-1)


class Batcher:
    """Vectorized window sampler. The cache is small enough to live in RAM, so
    a DataLoader would only add per-item Python overhead."""

    def __init__(self, split: D.CacheSplit, norm: D.Normalizer, context: int, with_z2: bool,
                 shuffle_z2_seed: int | None = None) -> None:
        self.context = context
        self.z1 = ((split.z1 - norm.z1_mean) / norm.z1_std).astype(np.float32)
        self.act = ((split.act - norm.act_mean) / norm.act_std).astype(np.float32)
        self.priv = D.pose_features(split.pose)
        if split.terrain.shape[-1]:
            self.priv = np.concatenate([self.priv, split.terrain], axis=-1).astype(np.float32)
        self.pose = split.pose.astype(np.float32)
        self.power = ((split.power - norm.power_mean) / norm.power_std).astype(np.float32)
        self.power_raw = split.power.astype(np.float32)
        self.z2 = (
            ((split.z2 - norm.z2_mean) / norm.z2_std).astype(np.float32)
            if with_z2 and split.z2.shape[-1]
            else np.zeros(split.z1.shape[:2] + (0,), dtype=np.float32)
        )
        if shuffle_z2_seed is not None and self.z2.shape[-1]:
            # Plan 8.3 integrity ablation: z2 from a DIFFERENT layout, whole-episode
            # so it stays a coherent latent trajectory rather than noise.
            perm = np.random.default_rng(shuffle_z2_seed).permutation(self.z2.shape[0])
            self.z2 = self.z2[perm]
        self.n_episodes, self.n_frames = split.z1.shape[0], split.z1.shape[1]

    def sample(self, rng: np.random.Generator, batch: int, device: str) -> dict[str, torch.Tensor]:
        ep = rng.integers(0, self.n_episodes, batch)
        t0 = rng.integers(0, self.n_frames - self.context, batch)
        idx = t0[:, None] + np.arange(self.context + 1)[None, :]
        out = {}
        for name, source in (("z1", self.z1), ("z2", self.z2), ("act", self.act),
                             ("priv", self.priv), ("power", self.power)):
            out[name] = torch.from_numpy(source[ep[:, None], idx]).to(device, non_blocking=True)
        return out


def step_loss(model: WP2Model, batch: dict[str, torch.Tensor], variant: str) -> tuple[torch.Tensor, dict]:
    tokens = build_tokens(batch["z1"][:, :-1], batch["z2"][:, :-1], batch["priv"][:, :-1],
                          batch["act"][:, :-1], variant)
    pred_delta, pred_z2, pred_power = model(tokens)
    target_delta = batch["z1"][:, 1:] - batch["z1"][:, :-1]
    loss_z1 = F.huber_loss(pred_delta, target_delta, delta=1.0)
    loss_power = F.huber_loss(pred_power, batch["power"][:, 1:], delta=1.0)
    parts = {"z1": float(loss_z1.detach()), "power": float(loss_power.detach())}
    loss = loss_z1 + loss_power
    if variant == "joint" and pred_z2 is not None:
        target_z2 = batch["z2"][:, 1:]
        mse = F.mse_loss(pred_z2, target_z2)
        cos = (1.0 - F.cosine_similarity(pred_z2, target_z2, dim=-1)).mean()
        loss = loss + mse + 0.1 * cos
        parts.update(z2_mse=float(mse.detach()), z2_cos=float(cos.detach()))
    return loss, parts


def integrate_pose(pose: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    """Mirror of trainer._integrate_pose, batched over episodes."""
    yaw = pose[:, 2] + DT_S * z1[:, YAW_RATE]
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    vx_world = cos_yaw * z1[:, VX] - sin_yaw * z1[:, VY]
    vy_world = sin_yaw * z1[:, VX] + cos_yaw * z1[:, VY]
    return torch.stack([pose[:, 0] + DT_S * vx_world, pose[:, 1] + DT_S * vy_world, yaw], dim=1)


@torch.no_grad()
def rollout_eval(model: WP2Model, data: Batcher, norm: D.Normalizer, variant: str, context: int,
                 horizons: list[int], n_episodes: int, device: str, seed: int = 7) -> dict:
    """Autonomous rollout from frame 0; only the actions come from the episode.

    Reports three families, because the triad showed they disagree: dead-reckoned
    pose (which the priv row proved is insensitive to localization), the
    terrain-coupled channels z2 actually moves, and the energy the Planner-C
    scorer would consume.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    eps = rng.choice(data.n_episodes, size=min(n_episodes, data.n_episodes), replace=False)
    max_h = max(horizons)
    if context + max_h > data.n_frames:
        raise ValueError("context + horizon exceeds episode length")

    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    z1_gt, z2_gt = to_t(data.z1[eps]), to_t(data.z2[eps])
    act, priv, pose_gt = to_t(data.act[eps]), to_t(data.priv[eps]), to_t(data.pose[eps])
    power_gt = to_t(data.power_raw[eps])[..., 0]
    z1_mean, z1_std = to_t(norm.z1_mean.astype(np.float32)), to_t(norm.z1_std.astype(np.float32))
    p_mean = float(norm.power_mean[0])
    p_std = float(norm.power_std[0])

    z1_hist, z2_hist = z1_gt[:, :context].clone(), z2_gt[:, :context].clone()
    pose = pose_gt[:, context - 1].clone()
    cv_pose = pose_gt[:, context - 1].clone()
    cv_state = z1_gt[:, context - 1] * z1_std + z1_mean
    energy_pred = torch.zeros(len(eps), device=device)
    energy_gt = torch.zeros(len(eps), device=device)
    drive_pred = torch.zeros(len(eps), device=device)
    drive_gt = torch.zeros(len(eps), device=device)
    results: dict[str, float] = {}

    for step in range(max_h):
        window = slice(step, step + context)
        tokens = build_tokens(z1_hist[:, -context:], z2_hist[:, -context:],
                              priv[:, window], act[:, window], variant)
        pred_delta, pred_z2, pred_power = model(tokens)
        z1_next = z1_hist[:, -1] + pred_delta[:, -1]
        z1_hist = torch.cat([z1_hist, z1_next.unsqueeze(1)], dim=1)
        z2_hist = torch.cat(
            [z2_hist,
             (pred_z2[:, -1] if (variant == "joint" and pred_z2 is not None) else z2_hist[:, -1]).unsqueeze(1)],
            dim=1,
        )
        pose = integrate_pose(pose, z1_next * z1_std + z1_mean)
        cv_pose = integrate_pose(cv_pose, cv_state)

        power_kw = pred_power[:, -1, 0] * p_std + p_mean
        power_true = power_gt[:, context + step]
        energy_pred = energy_pred + power_kw * DT_S
        energy_gt = energy_gt + power_true * DT_S
        drive_pred = drive_pred + power_kw.clamp(min=0.0) * DT_S
        drive_gt = drive_gt + power_true.clamp(min=0.0) * DT_S

        h = step + 1
        if h in horizons:
            gt = pose_gt[:, context - 1 + h]
            results[f"pose_err_m@{h}"] = float((pose[:, :2] - gt[:, :2]).norm(dim=1).mean())
            results[f"pose_err_p95_m@{h}"] = float((pose[:, :2] - gt[:, :2]).norm(dim=1).quantile(0.95))
            results[f"cv_pose_err_m@{h}"] = float((cv_pose[:, :2] - gt[:, :2]).norm(dim=1).mean())
            results[f"yaw_err_deg@{h}"] = float(
                torch.rad2deg(torch.abs((pose[:, 2] - gt[:, 2] + math.pi) % (2 * math.pi) - math.pi)).mean()
            )
            err = (z1_hist[:, context - 1 + h] - z1_gt[:, context - 1 + h]).abs()
            results[f"z1_mae_norm@{h}"] = float(err.mean())
            results[f"terrain_mae_norm@{h}"] = float(err[:, TERRAIN_CHANNELS].mean())
            results[f"pose_chan_mae_norm@{h}"] = float(err[:, POSE_CHANNELS].mean())
            results[f"power_mae_kw@{h}"] = float((power_kw - power_true).abs().mean())
            results[f"energy_err_kj@{h}"] = float((energy_pred - energy_gt).abs().mean())
            results[f"energy_rel@{h}"] = float(
                (energy_pred - energy_gt).abs().mean() / energy_gt.abs().mean().clamp(min=1e-6)
            )
            results[f"drive_energy_err_kj@{h}"] = float((drive_pred - drive_gt).abs().mean())
            if variant == "joint":
                results[f"z2_cos@{h}"] = float(
                    F.cosine_similarity(z2_hist[:, context - 1 + h], z2_gt[:, context - 1 + h], dim=-1).mean()
                )
    model.train()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--variant", choices=["state", "joint", "priv", "privterr"], required=True)
    parser.add_argument("--shuffle-z2", action="store_true",
                        help="integrity ablation: draw z2 from a different layout")
    parser.add_argument("--out", required=True)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=256)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 20, 40, 100])
    parser.add_argument("--selection", default="z1_mae_norm",
                        help="metric family for checkpoint selection, scored at the longest horizon")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--max-train-episodes", type=int, default=0, help="smoke only")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = Path(args.cache)
    keys = D.load_cache_keys(cache)
    train_keys, val_keys, test_keys = D.split_keys(keys)
    if args.max_train_episodes:
        train_keys = train_keys[: args.max_train_episodes]
        val_keys = val_keys[: max(args.eval_episodes, 8)]
    need_z2 = args.variant == "joint"
    need_terrain = args.variant == "privterr"
    print(f"split: {len(train_keys)}/{len(val_keys)}/{len(test_keys)} (test untouched)", flush=True)

    t0 = time.time()
    train_split = D.load_split(cache, train_keys, with_z2=need_z2, with_terrain=need_terrain)
    val_split = D.load_split(cache, val_keys, with_z2=need_z2, with_terrain=need_terrain)
    print(f"loaded cache in {time.time() - t0:.1f}s", flush=True)

    norm = D.Normalizer.fit(train_split)
    z1_dim, act_dim = train_split.z1.shape[-1], train_split.act.shape[-1]
    z2_dim = train_split.z2.shape[-1] if need_z2 else 0
    shuffle_seed = args.seed if args.shuffle_z2 else None
    train_data = Batcher(train_split, norm, args.context, need_z2, shuffle_seed)
    val_data = Batcher(val_split, norm, args.context, need_z2,
                       None if shuffle_seed is None else shuffle_seed + 1)
    # The Batcher holds normalized copies; the raw stacks would otherwise
    # double the ~3 GB z2 cache in RAM for the whole run.
    del train_split, val_split

    priv_dim = train_data.priv.shape[-1] if args.variant in ("priv", "privterr") else 0
    model_cfg = {
        "block_size": args.context,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "dropout": args.dropout,
        "bias": False,
        "head_hidden_dim": args.n_embd,
    }
    model = WP2Model(z1_dim, z2_dim, priv_dim, act_dim, model_cfg)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"variant={args.variant} token_dim={model.backbone.config.input_dim} params={n_params/1e6:.2f}M",
          flush=True)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))

    (out_dir / "config.json").write_text(
        json.dumps({**vars(args), "model": model_cfg, "n_params": n_params,
                    "split_counts": [len(train_keys), len(val_keys), len(test_keys)],
                    "normalization": norm.to_dict()}, indent=2)
    )
    log_path = out_dir / "train_log.jsonl"
    log_path.write_text("")
    rng = np.random.default_rng(args.seed)
    val_rng_seed = args.seed + 1
    best = {"metric": float("inf"), "step": -1}
    start = time.time()

    for step in range(args.steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        batch = train_data.sample(rng, args.batch, device)
        loss, parts = step_loss(model, batch, args.variant)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if (step + 1) % 100 == 0:
            record = {"phase": "train", "step": step + 1, "loss": float(loss.detach()), **parts,
                      "lr": lr_at(step), "sps": (step + 1) * args.batch / (time.time() - start)}
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            val_rng = np.random.default_rng(val_rng_seed)
            val_losses = []
            with torch.no_grad():
                for _ in range(args.val_batches):
                    vb = val_data.sample(val_rng, args.batch, device)
                    vloss, _ = step_loss(model, vb, args.variant)
                    val_losses.append(float(vloss))
            metrics = rollout_eval(model, val_data, norm, args.variant, args.context,
                                   args.horizons, args.eval_episodes, device)
            selection = metrics[f"{args.selection}@{max(args.horizons)}"]
            record = {"phase": "val", "step": step + 1, "val_loss": float(np.mean(val_losses)),
                      "selection": selection, **metrics}
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if selection < best["metric"]:
                best = {"metric": selection, "step": step + 1, **metrics}
                torch.save({"model": model.state_dict(), "config": model_cfg, "variant": args.variant,
                            "normalization": norm.to_dict(), "step": step + 1, "metrics": metrics},
                           out_dir / "ckpt_best.pt")

    torch.save({"model": model.state_dict(), "config": model_cfg, "variant": args.variant,
                "normalization": norm.to_dict(), "step": args.steps}, out_dir / "ckpt_last.pt")
    (out_dir / "g3_readout.json").write_text(
        json.dumps({"variant": args.variant, "best": best, "wall_s": time.time() - start,
                    "split_counts": [len(train_keys), len(val_keys), len(test_keys)]}, indent=2)
    )
    print(f"done: best {args.selection} {best['metric']:.4f} @ step {best['step']}", flush=True)


if __name__ == "__main__":
    main()
