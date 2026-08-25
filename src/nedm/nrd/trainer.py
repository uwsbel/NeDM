"""Two-stage NRD training (study plan section 9).

Stage ``ae``   -- encoder/decoder warm-up on individual Chrono frames (L1).
Stage ``joint``-- joint transition training on [z1, z2, a] windows with the
                  NRD loss: Huber on normalized delta-z1, MSE + cosine on the
                  predicted next latent against stopgrad(E(x_{t+1})), L1 on
                  decoded predicted frames (subsampled positions), and a
                  unit-circle penalty on predicted (cos, sin) pairs.

Model selection follows the NeDM lesson (judge on rollouts, not val_loss): the
checkpoint metric is the AUTONOMOUS rollout tip error -- frames are encoded only
for the context window, after which both z1 and z2 are predicted recursively.

Run in the NeDM conda env, e.g.:

    PYTHONPATH=src python -m nedm.nrd.trainer --config configs/nrd/dpend_nrd_v1.json --stage ae
    PYTHONPATH=src python -m nedm.nrd.trainer --config configs/nrd/dpend_nrd_v1.json --stage joint
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler

from nedm.nrd.model import NRDDynamicsModel
from nedm.nrd.vision import ConvDecoder, ConvEncoder, frames_to_float
from nedm.training.dataset import WindowedHMMWVDataset, load_metadata, load_rollout_split
from nedm.training.trainer import (
    build_optimizer,
    infinite_loader,
    move_batch,
    pendulum_tip_positions,
    resolve_device,
    seed_everything,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def trig_pair_indices(state_fields: list[str]) -> list[tuple[int, int]]:
    """(cos_X, sin_X) channel index pairs found in the state layout."""
    pairs: list[tuple[int, int]] = []
    for index, field in enumerate(state_fields):
        if field.startswith("cos_"):
            partner = "sin_" + field[len("cos_") :]
            if partner in state_fields:
                pairs.append((index, state_fields.index(partner)))
    return pairs


def estimate_background(frames_array: np.ndarray, samples: int = 256, seed: int = 0) -> torch.Tensor:
    """Per-pixel median over sampled frames -> (H, W, 3) float in [0, 1].

    The camera is fixed and the background static, so the pixel-wise median over
    a modest sample is an essentially exact background model.
    """
    index = np.sort(
        np.random.default_rng(seed).choice(
            frames_array.shape[0], size=min(samples, frames_array.shape[0]), replace=False
        )
    )
    sample = np.array(frames_array[index]).astype(np.float32) / 255.0
    return torch.from_numpy(np.median(sample, axis=0))


def foreground_weight_map(
    target_images: torch.Tensor, background: torch.Tensor, weight: float, threshold: float = 0.06
) -> torch.Tensor:
    """(N, 3, H, W) targets -> per-pixel loss weights (N, 1, H, W).

    The pendulum covers only ~3% of the frame; an unweighted L1 reaches ~0.005
    by reconstructing the constant background and ERASING the pendulum entirely
    (observed). Pixels that differ from the static background get weight
    1 + ``weight``, so the foreground dominates the reconstruction objective.
    """
    bg = background.to(target_images.device).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    fg_mask = (target_images - bg).abs().amax(dim=1, keepdim=True) > threshold
    return 1.0 + weight * fg_mask.to(target_images.dtype)


def weighted_l1(recon: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return ((recon - target).abs() * weights).sum() / (weights.sum() * recon.shape[1])


def load_rollout_split_with_frames(processed_root: Path, split: str) -> dict[str, Any]:
    """load_rollout_split plus a lazy per-episode frames memmap slice."""
    split_data = load_rollout_split(processed_root, split)
    frames = np.load(processed_root / f"{split}_frames.npy", mmap_mode="r")
    metadata = load_json(processed_root / f"{split}_episodes.json")
    offsets = metadata["rollout_episode_offsets"]
    for episode_index, episode in enumerate(split_data["episodes"]):
        start, stop = int(offsets[episode_index]), int(offsets[episode_index + 1])
        episode["frames"] = frames[start:stop]  # memmap view: (length + 1, H, W, 3)
    return split_data


# ---------------------------------------------------------------------------
# Stage 1: autoencoder warm-up
# ---------------------------------------------------------------------------
class AutoencoderTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.processed_root = Path(config["processed_dataset_dir"]).resolve()
        self.output_dir = Path(config["output_dir"]).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = load_metadata(self.processed_root)

        ae_cfg = config.get("ae_training", {})
        self.device = resolve_device(ae_cfg.get("device", config.get("training", {}).get("device", "auto")))
        self.seed = int(ae_cfg.get("seed", 20260825))
        seed_everything(self.seed)
        self.batch_size = int(ae_cfg.get("batch_size", 128))
        self.num_epochs = int(ae_cfg.get("num_epochs", 8))
        self.steps_per_epoch = int(ae_cfg.get("steps_per_epoch", 500))
        self.max_val_batches = int(ae_cfg.get("max_val_batches", 40))
        self.lr = float(ae_cfg.get("lr", 3e-4))
        num_workers = int(ae_cfg.get("num_workers", 4))

        vision_cfg = config.get("vision", {})
        z2_dim = int(config["model"].get("z2_dim", 64))
        self.encoder = ConvEncoder(
            z2_dim,
            image_size=int(vision_cfg.get("image_size", 128)),
            base_channels=int(vision_cfg.get("base_channels", 32)),
        ).to(self.device)
        self.decoder = ConvDecoder(
            z2_dim,
            image_size=int(vision_cfg.get("image_size", 128)),
            base_channels=int(vision_cfg.get("base_channels", 32)),
        ).to(self.device)

        # Window length 1 -> item frames shape (2, H, W, 3); both frames are used.
        self.train_dataset = WindowedHMMWVDataset(
            self.processed_root, "train", sequence_length=1, seed=self.seed, load_frames=True
        )
        sampler = RandomSampler(
            self.train_dataset, replacement=True, num_samples=self.steps_per_epoch * self.batch_size
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
        )
        self.val_dataset = WindowedHMMWVDataset(
            self.processed_root, "val", sequence_length=1, seed=self.seed + 1, load_frames=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers
        )
        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=1e-5)
        self.metrics_path = self.output_dir / "ae_metrics.jsonl"
        # Foreground-weighted loss (see foreground_weight_map): without it the
        # decoder converges to background-only output and erases the pendulum.
        self.foreground_weight = float(ae_cfg.get("foreground_weight", 30.0))
        self.background = estimate_background(self.train_dataset.frames, seed=self.seed).to(self.device)

    def _batch_images(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        frames = batch["frames"].to(self.device, non_blocking=True)  # (B, 2, H, W, 3)
        return frames_to_float(frames.flatten(0, 1))  # (2B, 3, H, W)

    def train(self) -> Path:
        best_val = float("inf")
        checkpoint_path = self.output_dir / "ae_best.pt"
        train_iterator = infinite_loader(self.train_loader)
        for epoch in range(1, self.num_epochs + 1):
            self.encoder.train()
            self.decoder.train()
            losses = []
            for _ in range(self.steps_per_epoch):
                images = self._batch_images(next(train_iterator))
                self.optimizer.zero_grad(set_to_none=True)
                recon = self.decoder(self.encoder(images))
                weights = foreground_weight_map(images, self.background, self.foreground_weight)
                loss = weighted_l1(recon, images, weights)
                loss.backward()
                clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()), 1.0
                )
                self.optimizer.step()
                losses.append(float(loss.item()))

            self.encoder.eval()
            self.decoder.eval()
            val_l1 = val_weighted = val_fg = val_mse = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch_index, batch in enumerate(self.val_loader):
                    if batch_index >= self.max_val_batches:
                        break
                    images = self._batch_images(batch)
                    recon = self.decoder(self.encoder(images))
                    weights = foreground_weight_map(images, self.background, self.foreground_weight)
                    val_weighted += float(weighted_l1(recon, images, weights).item())
                    val_l1 += float(F.l1_loss(recon, images).item())
                    val_mse += float(F.mse_loss(recon, images).item())
                    fg_mask = (weights > 1.0).expand_as(images)
                    if bool(fg_mask.any()):
                        val_fg += float((recon - images)[fg_mask].abs().mean().item())
                    val_batches += 1
            val_l1 /= max(val_batches, 1)
            val_weighted /= max(val_batches, 1)
            val_fg /= max(val_batches, 1)
            val_mse /= max(val_batches, 1)
            psnr = 10.0 * math.log10(1.0 / max(val_mse, 1e-12))
            record = {
                "epoch": epoch,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "train_weighted_l1": sum(losses) / max(len(losses), 1),
                "val_weighted_l1": val_weighted,
                "val_l1": val_l1,
                "val_foreground_l1": val_fg,
                "val_psnr_db": psnr,
            }
            with self.metrics_path.open("a") as fp:
                fp.write(json.dumps(record) + "\n")
            print(json.dumps(record))
            if val_weighted < best_val:
                best_val = val_weighted
                torch.save(
                    {
                        "config": self.config,
                        "metadata": self.metadata,
                        "encoder_state_dict": self.encoder.state_dict(),
                        "decoder_state_dict": self.decoder.state_dict(),
                        "background_image": self.background.cpu(),
                        "metrics": record,
                    },
                    checkpoint_path,
                )
        print(f"autoencoder warm-up done; best val weighted L1 {best_val:.5f} -> {checkpoint_path}")
        return checkpoint_path


# ---------------------------------------------------------------------------
# Baseline 5 (study plan section 10): pose-conditioned decoder D(z1).
# The honesty check for the fixed-camera scene: if a decoder driven by the
# 6-D physical state alone matches the latent pipeline's frames, the scene's
# pixels are fully explained by z1 (expected in Study 1, and reported as such).
# ---------------------------------------------------------------------------
class PoseDecoderTrainer(AutoencoderTrainer):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        state_dim = len(self.metadata["state_fields"])
        vision_cfg = config.get("vision", {})
        self.decoder = ConvDecoder(
            state_dim,
            image_size=int(vision_cfg.get("image_size", 128)),
            base_channels=int(vision_cfg.get("base_channels", 32)),
        ).to(self.device)
        normalization = self.metadata["normalization"]
        self.state_mean = torch.tensor(normalization["state_mean"], device=self.device)
        self.state_std = torch.tensor(normalization["state_std"], device=self.device)
        self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=self.lr, weight_decay=1e-5)
        self.metrics_path = self.output_dir / "pose_decoder_metrics.jsonl"

    def train(self) -> Path:
        best_val = float("inf")
        checkpoint_path = self.output_dir / "pose_decoder_best.pt"
        train_iterator = infinite_loader(self.train_loader)

        def batch_pairs(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            states = batch["states"][:, 0].to(self.device, non_blocking=True)
            images = frames_to_float(batch["frames"][:, 0].to(self.device, non_blocking=True))
            return (states - self.state_mean) / self.state_std, images

        for epoch in range(1, self.num_epochs + 1):
            self.decoder.train()
            losses = []
            for _ in range(self.steps_per_epoch):
                states, images = batch_pairs(next(train_iterator))
                self.optimizer.zero_grad(set_to_none=True)
                recon = self.decoder(states)
                weights = foreground_weight_map(images, self.background, self.foreground_weight)
                loss = weighted_l1(recon, images, weights)
                loss.backward()
                clip_grad_norm_(self.decoder.parameters(), 1.0)
                self.optimizer.step()
                losses.append(float(loss.item()))

            self.decoder.eval()
            val_weighted = val_fg = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch_index, batch in enumerate(self.val_loader):
                    if batch_index >= self.max_val_batches:
                        break
                    states, images = batch_pairs(batch)
                    recon = self.decoder(states)
                    weights = foreground_weight_map(images, self.background, self.foreground_weight)
                    val_weighted += float(weighted_l1(recon, images, weights).item())
                    fg_mask = (weights > 1.0).expand_as(images)
                    if bool(fg_mask.any()):
                        val_fg += float((recon - images)[fg_mask].abs().mean().item())
                    val_batches += 1
            val_weighted /= max(val_batches, 1)
            val_fg /= max(val_batches, 1)
            record = {
                "epoch": epoch,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "train_weighted_l1": sum(losses) / max(len(losses), 1),
                "val_weighted_l1": val_weighted,
                "val_foreground_l1": val_fg,
            }
            with self.metrics_path.open("a") as fp:
                fp.write(json.dumps(record) + "\n")
            print(json.dumps(record))
            if val_weighted < best_val:
                best_val = val_weighted
                torch.save(
                    {
                        "config": self.config,
                        "metadata": self.metadata,
                        "decoder_state_dict": self.decoder.state_dict(),
                        "background_image": self.background.cpu(),
                        "metrics": record,
                    },
                    checkpoint_path,
                )
        print(f"pose decoder done; best val weighted L1 {best_val:.5f} -> {checkpoint_path}")
        return checkpoint_path


# ---------------------------------------------------------------------------
# Stage 2/3: joint transition training
# ---------------------------------------------------------------------------
class NRDTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.processed_root = Path(config["processed_dataset_dir"]).resolve()
        self.output_dir = Path(config["output_dir"]).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.metadata = load_metadata(self.processed_root)
        self.state_fields = list(self.metadata["state_fields"])
        self.dt_s = float(self.metadata["dt_s"])

        training_cfg = config["training"]
        self.device = resolve_device(training_cfg.get("device", "auto"))
        self.seed = int(training_cfg["seed"])
        seed_everything(self.seed)
        self.batch_size = int(training_cfg["batch_size"])
        self.num_epochs = int(training_cfg["num_epochs"])
        self.steps_per_epoch = int(training_cfg["steps_per_epoch"])
        self.max_val_batches = int(training_cfg["max_val_batches"])
        self.block_size = int(config["model"]["block_size"])
        self.rollout_horizon = int(training_cfg.get("rollout_horizon", 1))
        self.horizon_gamma = float(training_cfg.get("rollout_horizon_gamma", 0.95))
        num_workers = int(training_cfg.get("num_workers", 4))

        window = self.block_size + max(self.rollout_horizon - 1, 0)
        self.train_dataset = WindowedHMMWVDataset(
            self.processed_root, "train", sequence_length=window, seed=self.seed, load_frames=True
        )
        sampler = RandomSampler(
            self.train_dataset, replacement=True, num_samples=self.steps_per_epoch * self.batch_size
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
        )
        self.val_dataset = WindowedHMMWVDataset(
            self.processed_root, "val", sequence_length=window, seed=self.seed + 1, load_frames=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers
        )

        self.model = NRDDynamicsModel(
            state_dim=len(self.state_fields),
            action_dim=len(self.metadata["action_fields"]),
            transformer_cfg=config["model"],
            vision_cfg=config.get("vision", {}),
            normalization=self.metadata["normalization"],
            state_fields=self.state_fields,
        ).to(self.device)

        vision_cfg = config.get("vision", {})
        ae_checkpoint = vision_cfg.get("ae_checkpoint")
        self.background: torch.Tensor | None = None
        if ae_checkpoint:
            payload = torch.load(Path(ae_checkpoint).expanduser().resolve(), map_location=self.device, weights_only=False)
            self.model.encoder.load_state_dict(payload["encoder_state_dict"])
            self.model.decoder.load_state_dict(payload["decoder_state_dict"])
            if "background_image" in payload:
                self.background = payload["background_image"].to(self.device)
            print(f"loaded AE warm-up weights from {ae_checkpoint}")
        if self.background is None:
            self.background = estimate_background(self.train_dataset.frames, seed=self.seed).to(self.device)
        self.freeze_encoder = bool(vision_cfg.get("freeze_encoder", True))
        if self.freeze_encoder:
            for parameter in self.model.encoder.parameters():
                parameter.requires_grad_(False)
            self.model.encoder.eval()

        # Fit the z2 normalization on a sample of training frames (see model.py:
        # the raw latents share a large constant component that would otherwise
        # dominate latent losses and metrics).
        with torch.no_grad():
            frames_array = self.train_dataset.frames
            sample_index = np.sort(
                np.random.default_rng(self.seed).choice(
                    frames_array.shape[0], size=min(4096, frames_array.shape[0]), replace=False
                )
            )
            chunks = []
            for start in range(0, len(sample_index), 512):
                chunk = torch.from_numpy(np.array(frames_array[sample_index[start : start + 512]]))
                chunks.append(
                    self.model.encode_frame_sequence(chunk.unsqueeze(0).to(self.device))[0]
                )
            self.model.set_z2_normalization(torch.cat(chunks, dim=0))
        print(
            f"z2 normalization fitted on {len(sample_index)} frames: "
            f"mean-norm {self.model.z2_mean.norm().item():.3f}, "
            f"per-dim std mean {self.model.z2_std.mean().item():.5f}"
        )

        loss_cfg = config.get("loss", {})
        self.lambda_z1 = float(loss_cfg.get("lambda_z1", 1.0))
        self.lambda_z2 = float(loss_cfg.get("lambda_z2", 1.0))
        self.lambda_frame = float(loss_cfg.get("lambda_frame", 0.1))
        self.lambda_circle = float(loss_cfg.get("lambda_circle", 0.01))
        self.huber_delta = float(loss_cfg.get("huber_delta", 1.0))
        self.frame_loss_positions = int(loss_cfg.get("frame_loss_positions", 4))
        self.frame_foreground_weight = float(loss_cfg.get("frame_foreground_weight", 30.0))
        self.trig_pairs = trig_pair_indices(self.state_fields)

        self.optimizer = build_optimizer(self.model, config["optimizer"])
        self.grad_clip_norm = float(config["optimizer"].get("grad_clip_norm", 1.0))
        self.rollout_eval_cfg = config.get("rollout_eval", {})
        self.checkpoint_metric = str(training_cfg.get("checkpoint_metric", "rollout_sel"))
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.best_metric = float("inf")
        self.global_step = 0
        self.start_epoch = 0

        resume_path = training_cfg.get("resume_from_checkpoint")
        if resume_path:
            payload = torch.load(Path(resume_path).expanduser().resolve(), map_location=self.device, weights_only=False)
            self.model.load_state_dict(payload["model_state_dict"])
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            self.start_epoch = int(payload.get("epoch", 0))
            self.global_step = int(payload.get("global_step", 0))
            if self.metrics_path.exists():
                history = [json.loads(line) for line in self.metrics_path.read_text().splitlines() if line.strip()]
                values = [float(r[self.checkpoint_metric]) for r in history if self.checkpoint_metric in r]
                if values:
                    self.best_metric = min(values)
            print(
                f"resumed from {resume_path} at epoch {self.start_epoch} "
                f"(best {self.checkpoint_metric}={self.best_metric:.5f})"
            )

        self._rollout_episodes: list[dict[str, Any]] | None = None

    # -- optimization --------------------------------------------------------
    def scheduled_lr(self) -> float:
        optimizer_cfg = self.config["optimizer"]
        warmup_steps = int(optimizer_cfg.get("warmup_steps", 0))
        min_lr = float(optimizer_cfg.get("min_lr", optimizer_cfg["lr"]))
        max_lr = float(optimizer_cfg["lr"])
        total_steps = self.num_epochs * self.steps_per_epoch
        if warmup_steps > 0 and self.global_step < warmup_steps:
            return max_lr * float(self.global_step + 1) / float(warmup_steps)
        progress = (self.global_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_lr - min_lr)

    def _loss_terms(
        self,
        delta_norm: torch.Tensor,
        z2_pred: torch.Tensor,
        states: torch.Tensor,
        target_norm: torch.Tensor,
        z2_target: torch.Tensor,
        frame_targets: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """All positions share shape (B, T, ·); frame_targets is (B, T, H, W, 3) uint8.

        ``z2_pred`` is in NORMALIZED z2 space (the visual head's output);
        ``z2_target`` is raw encoder output and is normalized here.
        """
        terms: dict[str, torch.Tensor] = {}
        terms["z1"] = F.huber_loss(delta_norm, target_norm, delta=self.huber_delta)
        z2_detached = self.model.normalize_z2(z2_target.detach())
        mse = F.mse_loss(z2_pred, z2_detached)
        cosine = 1.0 - F.cosine_similarity(z2_pred, z2_detached, dim=-1).mean()
        terms["z2"] = mse + cosine
        terms["z2_cos_sim"] = F.cosine_similarity(z2_pred.detach(), z2_detached, dim=-1).mean()

        if self.lambda_circle > 0.0 and self.trig_pairs:
            next_state = states + self.model.denormalize_target(delta_norm)
            circle = 0.0
            for cos_index, sin_index in self.trig_pairs:
                radius_sq = next_state[..., cos_index] ** 2 + next_state[..., sin_index] ** 2
                circle = circle + ((radius_sq - 1.0) ** 2).mean()
            terms["circle"] = circle / len(self.trig_pairs)
        else:
            terms["circle"] = torch.zeros((), device=delta_norm.device)

        if frame_targets is not None and self.lambda_frame > 0.0:
            batch, steps = z2_pred.shape[:2]
            count = min(self.frame_loss_positions, steps)
            positions = torch.randint(0, steps, (batch, count), device=z2_pred.device)
            gather_index = positions.unsqueeze(-1).expand(-1, -1, z2_pred.shape[-1])
            picked_latents = torch.gather(z2_pred, 1, gather_index)  # (B, count, z2) normalized
            decoded = self.model.decode_latents(self.model.denormalize_z2(picked_latents))
            frame_index = positions.view(batch, count, 1, 1, 1).expand(
                -1, -1, *frame_targets.shape[2:]
            )
            picked_frames = torch.gather(frame_targets, 1, frame_index)
            target_images = frames_to_float(picked_frames)
            flat_decoded = decoded.flatten(0, 1)
            flat_targets = target_images.flatten(0, 1)
            weights = foreground_weight_map(
                flat_targets, self.background, self.frame_foreground_weight
            )
            terms["frame"] = weighted_l1(flat_decoded, flat_targets, weights)
        else:
            terms["frame"] = torch.zeros((), device=delta_norm.device)
        return terms

    def _window_forward(
        self, batch: dict[str, torch.Tensor], train: bool
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """One-step (H=1) or curriculum (H>1) loss over a window batch."""
        states = batch["states"]
        actions = batch["actions"]
        targets = batch["targets"]
        frames = batch["frames"]  # (B, window + 1, H, W, 3) uint8

        encode = self.model.encode_frame_sequence
        if self.freeze_encoder:
            with torch.no_grad():
                z2_all = encode(frames)
        else:
            z2_all = encode(frames)
        block = self.block_size
        horizon = self.rollout_horizon

        total_loss = torch.zeros((), device=states.device)
        logs: dict[str, float] = {}
        weight_sum = 0.0

        # Step 0: teacher-forced over the context window, losses at every position.
        window_states = states[:, :block]
        window_z2 = z2_all[:, :block]
        window_actions = actions[:, :block]
        delta_norm, z2_pred = self.model.forward(window_states, window_z2, window_actions)
        target_norm = self.model.normalize_target(targets[:, :block])
        terms = self._loss_terms(
            delta_norm, z2_pred, window_states, target_norm, z2_all[:, 1 : block + 1],
            frames[:, 1 : block + 1],
        )
        step_loss = (
            self.lambda_z1 * terms["z1"]
            + self.lambda_z2 * terms["z2"]
            + self.lambda_frame * terms["frame"]
            + self.lambda_circle * terms["circle"]
        )
        total_loss = total_loss + step_loss
        weight_sum += 1.0
        for key in ("z1", "z2", "frame", "circle", "z2_cos_sim"):
            logs[key] = float(terms[key].item())

        # Steps k >= 1: slide the window, feeding predictions back in.
        # rolled_z2 stays in RAW latent space (forward() normalizes inputs).
        if horizon > 1:
            pred_state = window_states[:, -1] + self.model.denormalize_target(delta_norm[:, -1])
            pred_z2 = self.model.denormalize_z2(z2_pred[:, -1])
            rolled_states = torch.cat([states[:, 1:block], pred_state.unsqueeze(1)], dim=1)
            rolled_z2 = torch.cat([z2_all[:, 1:block], pred_z2.unsqueeze(1)], dim=1)
            for k in range(1, horizon):
                window_actions = actions[:, k : block + k]
                delta_norm, z2_pred = self.model.forward(rolled_states, rolled_z2, window_actions)
                last_target = self.model.normalize_target(targets[:, block + k - 1 : block + k])
                terms = self._loss_terms(
                    delta_norm[:, -1:],
                    z2_pred[:, -1:],
                    rolled_states[:, -1:],
                    last_target,
                    z2_all[:, block + k : block + k + 1],
                    frames[:, block + k : block + k + 1],
                )
                gamma_k = self.horizon_gamma**k
                step_loss = (
                    self.lambda_z1 * terms["z1"]
                    + self.lambda_z2 * terms["z2"]
                    + self.lambda_frame * terms["frame"]
                    + self.lambda_circle * terms["circle"]
                )
                total_loss = total_loss + gamma_k * step_loss
                weight_sum += gamma_k
                pred_state = rolled_states[:, -1] + self.model.denormalize_target(delta_norm[:, -1])
                pred_z2 = self.model.denormalize_z2(z2_pred[:, -1])
                rolled_states = torch.cat(
                    [rolled_states[:, 1:], pred_state.unsqueeze(1)], dim=1
                )
                rolled_z2 = torch.cat([rolled_z2[:, 1:], pred_z2.unsqueeze(1)], dim=1)

        return total_loss / weight_sum, logs

    def training_step(self, batch: dict[str, torch.Tensor]) -> tuple[float, dict[str, float]]:
        batch = move_batch(batch, self.device)
        lr = self.scheduled_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        loss, logs = self._window_forward(batch, train=True)
        loss.backward()
        clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.grad_clip_norm)
        self.optimizer.step()
        self.global_step += 1
        return float(loss.item()), logs

    @torch.no_grad()
    def evaluate_windows(self) -> dict[str, Any]:
        self.model.eval()
        totals: dict[str, float] = {}
        count = 0
        for batch_index, batch in enumerate(self.val_loader):
            if batch_index >= self.max_val_batches:
                break
            batch = move_batch(batch, self.device)
            loss, logs = self._window_forward(batch, train=False)
            totals["loss"] = totals.get("loss", 0.0) + float(loss)
            for key, value in logs.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
        return {f"val_{key}": value / max(count, 1) for key, value in totals.items()}

    # -- autonomous rollout evaluation --------------------------------------
    def _rollout_episode_set(self) -> list[dict[str, Any]]:
        if self._rollout_episodes is None:
            split_data = load_rollout_split_with_frames(
                self.processed_root, str(self.rollout_eval_cfg.get("split", "val"))
            )
            num_episodes = int(self.rollout_eval_cfg.get("num_episodes", 24))
            horizon_steps = max(
                int(round(float(h) / self.dt_s)) for h in self.rollout_eval_cfg.get("horizons_s", [1.0])
            )
            minimum_rows = self.block_size + horizon_steps + 1
            usable = [ep for ep in split_data["episodes"] if ep["states"].shape[0] >= minimum_rows]
            self._rollout_episodes = usable[:num_episodes]
        return self._rollout_episodes

    @torch.no_grad()
    def evaluate_rollouts(self) -> dict[str, Any]:
        cfg = self.rollout_eval_cfg
        if not cfg:
            return {}
        self.model.eval()
        episodes = self._rollout_episode_set()
        if not episodes:
            return {}
        horizons_s = [float(value) for value in cfg.get("horizons_s", [0.5, 1.0, 2.0])]
        selection_horizon_s = float(cfg.get("selection_horizon_s", horizons_s[-1]))
        link_lengths = cfg.get("link_lengths")
        max_steps = max(int(round(h / self.dt_s)) for h in horizons_s)
        block = self.block_size

        states = torch.stack(
            [torch.from_numpy(ep["states"][: block + max_steps]) for ep in episodes]
        ).to(self.device)
        actions = torch.stack(
            [torch.from_numpy(ep["actions"][: block + max_steps]) for ep in episodes]
        ).to(self.device)
        gt_tip = torch.stack(
            [torch.from_numpy(np.array(ep["rollout"][: block + max_steps], copy=True)) for ep in episodes]
        ).to(self.device)
        frames_np = np.stack([np.array(ep["frames"][: block + max_steps]) for ep in episodes])
        frames = torch.from_numpy(frames_np).to(self.device)

        # Context: true states + encoded context frames. Autonomous afterwards.
        z2_context = self.model.encode_frame_sequence(frames[:, :block])
        rolled_states = states[:, :block].clone()
        rolled_z2 = z2_context.clone()
        predicted_states: list[torch.Tensor] = []
        predicted_z2: list[torch.Tensor] = []
        for step in range(max_steps):
            window_actions = actions[:, step : block + step]
            next_state, next_z2 = self.model.predict_next(rolled_states, rolled_z2, window_actions)
            predicted_states.append(next_state)
            predicted_z2.append(next_z2)
            rolled_states = torch.cat([rolled_states[:, 1:], next_state.unsqueeze(1)], dim=1)
            rolled_z2 = torch.cat([rolled_z2[:, 1:], next_z2.unsqueeze(1)], dim=1)
        pred_states = torch.stack(predicted_states, dim=1)  # (E, max_steps, S)
        pred_z2 = torch.stack(predicted_z2, dim=1)  # (E, max_steps, Z)

        # Ground-truth latents for the rolled span (chunked encode).
        gt_z2 = self.model.encode_frame_sequence(frames[:, block : block + max_steps])

        metrics: dict[str, Any] = {}
        trig_idx = torch.tensor(
            [self.state_fields.index(f) for f in ("cos_q1", "sin_q1", "cos_q2", "sin_q2")],
            dtype=torch.long,
            device=self.device,
        ) if link_lengths else None
        for horizon_s in horizons_s:
            steps = int(round(horizon_s / self.dt_s))
            gt_slice = states[:, block : block + steps]
            z1_rmse = float(torch.sqrt(((pred_states[:, :steps] - gt_slice) ** 2).mean()).item())
            # Cosine in NORMALIZED latent space; raw latents share a constant
            # component that would pin this metric at ~1.0 regardless of quality.
            cos_sim = float(
                F.cosine_similarity(
                    self.model.normalize_z2(pred_z2[:, :steps]),
                    self.model.normalize_z2(gt_z2[:, :steps]),
                    dim=-1,
                ).mean().item()
            )
            entry: dict[str, Any] = {"z1_rmse": z1_rmse, "z2_cos_sim": cos_sim, "episodes": len(episodes)}
            if link_lengths:
                pred_tip = pendulum_tip_positions(pred_states[:, :steps][..., trig_idx], link_lengths)
                gt_tip_slice = gt_tip[:, block : block + steps]
                tip_rmse = float(torch.sqrt((pred_tip - gt_tip_slice).pow(2).sum(-1).mean()).item())
                path = (
                    (gt_tip_slice[:, 1:] - gt_tip_slice[:, :-1]).pow(2).sum(-1).sqrt().sum(1).mean()
                )
                errdist = tip_rmse / max(float(path.item()), 1e-6)
                entry["tip_rmse_m"] = tip_rmse
                entry["errdist"] = errdist
            metrics[f"rollout_{horizon_s:.1f}s"] = entry
            if abs(horizon_s - selection_horizon_s) < 1e-9:
                metrics["rollout_sel"] = entry.get("errdist", entry["z1_rmse"])
        return metrics

    # -- checkpoints / loop --------------------------------------------------
    def save_checkpoint(self, name: str, epoch: int, metrics: dict[str, Any]) -> Path:
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "config": self.config,
                "metadata": self.metadata,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            path,
        )
        return path

    def train(self) -> Path:
        train_iterator = infinite_loader(self.train_loader)
        for epoch in range(self.start_epoch + 1, self.num_epochs + 1):
            self.model.train()
            if self.freeze_encoder:
                self.model.encoder.eval()
            epoch_losses = []
            epoch_logs: dict[str, float] = {}
            for _ in range(self.steps_per_epoch):
                loss, logs = self.training_step(next(train_iterator))
                epoch_losses.append(loss)
                for key, value in logs.items():
                    epoch_logs[key] = epoch_logs.get(key, 0.0) + value

            record = {
                "epoch": epoch,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "train_loss": sum(epoch_losses) / max(len(epoch_losses), 1),
                **{
                    f"train_{key}": value / max(self.steps_per_epoch, 1)
                    for key, value in epoch_logs.items()
                },
                **self.evaluate_windows(),
                **self.evaluate_rollouts(),
            }
            with self.metrics_path.open("a") as fp:
                fp.write(json.dumps(record) + "\n")
            self.save_checkpoint("last", epoch, record)
            if self.checkpoint_metric in record:
                value = float(record[self.checkpoint_metric])
                if value < self.best_metric:
                    self.best_metric = value
                    self.save_checkpoint("best_val", epoch, record)
            print(json.dumps(record, indent=2))
        return self.checkpoint_dir / "last.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NRD joint dynamics model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["ae", "joint", "posedec"], default="joint")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ae-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="joint stage: continue from a previous last.pt")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_json(args.config.resolve())
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.device is not None:
        config.setdefault("training", {})["device"] = args.device
        config.setdefault("ae_training", {})["device"] = args.device
    if args.stage in ("ae", "posedec"):
        if args.num_epochs is not None:
            config.setdefault("ae_training", {})["num_epochs"] = args.num_epochs
        if args.steps_per_epoch is not None:
            config.setdefault("ae_training", {})["steps_per_epoch"] = args.steps_per_epoch
        trainer_cls = AutoencoderTrainer if args.stage == "ae" else PoseDecoderTrainer
        trainer_cls(config).train()
        return 0
    if args.num_epochs is not None:
        config["training"]["num_epochs"] = args.num_epochs
    if args.steps_per_epoch is not None:
        config["training"]["steps_per_epoch"] = args.steps_per_epoch
    if args.ae_checkpoint is not None:
        config.setdefault("vision", {})["ae_checkpoint"] = str(args.ae_checkpoint)
    if args.resume_from is not None:
        config["training"]["resume_from_checkpoint"] = str(args.resume_from)
    trainer = NRDTrainer(config)
    final = trainer.train()
    print(f"NRD training completed; last checkpoint: {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
