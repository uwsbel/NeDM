#!/usr/bin/env python3
"""GPU training of the image-conditioned HMMWV reference-style forward GRU.

Four capacity-matched image arms: rgbd, rgb_only, depth_only, and blank.
No simulation collection or authored terrain/clearance inputs are used here.
CPU operation is restricted to checkpoint evaluation; all optimizer updates
belong in an AMD-cluster GPU allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from nedm.traverse.fdm_diverse_model import (RGBD_ARMS, RGBDFDMConfig, RGBDFiniteHorizonFDM,
    fdm_loss, load_rgbd_checkpoint, rgbd_model_config_dict, rotate_world_batch, PROGRESS_TARGET_VERSIONS)
# Reuse audited masked metrics, event eligibility, calibration, and JSON helpers.
# The previous pilot's training entry point is not executed or modified.
from traverse_fdm_train import (json_clean, make_normalization as profile_normalization,
                               outcome_metrics, sha256, write_json)


LOW_DIM_KEYS = ("history", "commands", "global_features", "nominal_pose", "trajectory", "work",
                "events", "trajectory_mask", "event_mask", "attitude", "attitude_mask")


def load_low_dim(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as file:
        missing = set(LOW_DIM_KEYS) - set(file.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        arrays = {key: np.ascontiguousarray(file[key], dtype=np.float32) for key in LOW_DIM_KEYS}
        for key in ("episode_index", "anchor", "image_index", "scene_index", "source_domain"):
            if key in file.files:
                arrays[key] = np.ascontiguousarray(file[key], dtype=np.int64)
        if "work_mask" in file.files:
            arrays["work_mask"] = np.ascontiguousarray(file["work_mask"], dtype=np.float32)
        for key in ("sustained_stall", "sustained_stall_mask", "bounded_motion", "bounded_motion_mask"):
            if key in file.files:
                arrays[key] = np.ascontiguousarray(file[key], dtype=np.float32)
    count = len(arrays["history"])
    if count == 0 or any(len(value) != count for value in arrays.values()):
        raise ValueError(f"{path}: empty or inconsistent batch dimensions")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"{path}: non-finite inputs or targets, including masked targets")
    horizon = arrays["nominal_pose"].shape[1]
    shapes = {"history": (count, 16, 24), "commands": (count, horizon, 5), "global_features": (count, 8),
              "nominal_pose": (count, horizon, 4), "trajectory": (count, horizon, 4), "work": (count, horizon, 1),
              "events": (count, horizon, 3), "trajectory_mask": (count, horizon, 1), "event_mask": (count, horizon, 3)}
    shapes.update(attitude=(count, horizon, 2), attitude_mask=(count, horizon, 2))
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{path}: {key} must be {shape}, got {arrays[key].shape}")
    for key in ("events", "trajectory_mask", "event_mask"):
        if not np.isin(arrays[key], [0.0, 1.0]).all():
            raise ValueError(f"{path}: {key} must be binary")
    return arrays


def select_progress_target(arrays: dict[str, np.ndarray], target: str) -> dict[str, np.ndarray]:
    """Explicitly choose the third head's frozen label, never silently mix definitions."""
    if target == "net_progress":
        return arrays
    if target not in PROGRESS_TARGET_VERSIONS:
        raise ValueError(f"Unknown progress target {target}")
    target_mask = target + "_mask"
    if target not in arrays or target_mask not in arrays:
        raise ValueError(f"{target} training/evaluation requires its separately recorded targets and masks")
    expected = (*arrays["events"].shape[:2], 1)
    for key in (target, target_mask):
        if arrays[key].shape != expected or not np.isin(arrays[key], [0.0, 1.0]).all():
            raise ValueError(f"Expected binary {key} of shape {expected}")
    result = dict(arrays)
    result["events"], result["event_mask"] = arrays["events"].copy(), arrays["event_mask"].copy()
    result["events"][..., 2:3] = arrays[target]
    result["event_mask"][..., 2:3] = arrays[target_mask]
    return result


class Images:
    """Memory-mapped camera cache with optional immutable half-precision GPU copy."""

    def __init__(self, path: Path, count: int, device: torch.device, image_size: int = 512,
                 preload: bool = False, image_index: np.ndarray | None = None):
        self.path, self.device = path, device
        self.array = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.array.ndim != 4 or self.array.shape[1:] != (4, image_size, image_size):
            raise ValueError(f"{path}: expected (N,4,{image_size},{image_size}), got {self.array.shape}")
        self.index = np.arange(count) if image_index is None else np.asarray(image_index)
        if len(self.index) != count or self.index.min() < 0 or self.index.max() >= len(self.array):
            raise ValueError(f"{path}: invalid image indexing")
        if image_index is None and len(self.array) != count:
            raise ValueError(f"{path}: image count differs from low-dimensional rows")
        self.tensor = None
        if preload:
            self.tensor = torch.from_numpy(np.array(self.array, copy=True)).to(device)

    def batch(self, rows: np.ndarray) -> torch.Tensor:
        image_ids = self.index[rows]
        if self.tensor is not None:
            return self.tensor[torch.as_tensor(image_ids, device=self.device)].float()
        pixels = torch.from_numpy(np.array(self.array[image_ids], copy=True))
        return pixels.to(self.device, dtype=torch.float32)


def normalization(train: dict[str, np.ndarray], max_pos_weight: float) -> dict[str, Any]:
    # The shared helper accepts a path tensor named candidate; here it receives
    # only the five geometry/speed command channels, never terrain features.
    proxy = {**train, "candidate": train["commands"]}
    result = profile_normalization(proxy, max_pos_weight)
    result["commands"] = result.pop("candidate")
    result["pixels"] = {"mean": [0.5, 0.5, 0.5, 0.0], "std": [0.5, 0.5, 0.5, 1.0],
                        "source": "fixed documented camera encoding; no validation fitting"}
    result["source"] = "training low-dimensional tensors only; fixed physical pixel normalization"
    return result


@torch.inference_mode()
def evaluate_rgbd(model: RGBDFiniteHorizonFDM, arrays: dict[str, np.ndarray], images: Images,
                  batch_size: int, device: torch.device, loss_weights: dict[str, float],
                  control: str = "normal", shuffle_seed: int = 20260908) -> tuple[dict, dict[str, np.ndarray]]:
    if control not in ("normal", "shuffle", "blank", "current_history"):
        raise ValueError(f"Unknown control: {control}")
    model.eval()
    count = len(arrays["history"])
    image_order = np.arange(count)
    if control == "shuffle":
        # Derange scene-image identities, not window rows: each global map is
        # reused by many windows, so a row shuffle can leave the image unchanged.
        identities, representative_rows = np.unique(images.index, return_index=True)
        if len(identities) < 2:
            raise ValueError("Image shuffle unavailable: fewer than two distinct scene images")
        shift = 1 + shuffle_seed % (len(identities) - 1)
        replacement = {int(key): int(row) for key, row in
                       zip(identities, np.roll(representative_rows, shift))}
        image_order = np.asarray([replacement[int(key)] for key in images.index])
        if np.any(images.index[image_order] == images.index):
            raise AssertionError("Shuffle retained an original scene image")
    outputs = {key: [] for key in ("trajectory", "work", "event_logits", "attitude")}
    sums, denominators = {}, {}
    start = time.perf_counter()
    for offset in range(0, count, batch_size):
        rows = np.arange(offset, min(offset + batch_size, count))
        batch = {key: torch.from_numpy(value[rows]).to(device) for key, value in arrays.items()}
        batch["rgbd"] = images.batch(image_order[rows])
        output = model(batch, control="normal" if control == "shuffle" else control)
        if any(not torch.isfinite(value).all() for value in output.values()):
            raise FloatingPointError(f"Non-finite {control} evaluation prediction")
        _, losses = fdm_loss(model, output, batch, loss_weights)
        counts = {"xy": float(batch["trajectory_mask"].sum()) * 2,
                  "yaw": float(batch["trajectory_mask"].sum()) * 2,
                  "work": float(batch.get("work_mask", batch["trajectory_mask"]).sum()),
                  "events": float((batch["event_mask"] * model.supported_events).sum()),
                  "attitude": float(batch["attitude_mask"].sum())}
        for key, denominator in counts.items():
            sums[key] = sums.get(key, 0.0) + float(losses[key]) * denominator
            denominators[key] = denominators.get(key, 0.0) + denominator
        for key in output:
            outputs[key].append(output[key].cpu().numpy())
    merged = {key: np.concatenate(value, axis=0) for key, value in outputs.items()}
    metrics = outcome_metrics(merged, arrays, model.supported_events.cpu().numpy(), model.config.dt)
    valid_attitude = np.broadcast_to(arrays["attitude_mask"] > 0, arrays["attitude"].shape)
    attitude_delta = merged["attitude"] - arrays["attitude"]
    attitude_abs = np.abs(np.arctan2(np.sin(attitude_delta), np.cos(attitude_delta)))
    metrics["attitude"] = {"mae_deg": float(np.degrees(attitude_abs[valid_attitude]).mean()) if valid_attitude.any() else None,
                           "valid_values": int(valid_attitude.sum())}
    metrics["progress_event_definition"] = model.config.progress_event_definition
    metrics["progress_event_version"] = model.config.progress_event_version
    metrics["events"]["low_progress"]["target_definition"] = model.config.progress_event_definition
    metrics["loss"] = {key: value / max(denominators[key], 1.0) for key, value in sums.items()}
    metrics["loss"]["total"] = sum(loss_weights.get(key, 0.0) * value for key, value in metrics["loss"].items())
    metrics.update(control=control, windows=count, evaluation_seconds=time.perf_counter() - start)
    if control == "shuffle":
        metrics["shuffle"] = {"seed": shuffle_seed, "distinct_images": int(len(np.unique(images.index))),
                              "same_image_windows": int((images.index[image_order] == images.index).sum()),
                              "index_sha256": hashlib.sha256(image_order.tobytes()).hexdigest()}
    model.train()
    return metrics, merged


def save(path: Path, model: RGBDFiniteHorizonFDM, optimizer: torch.optim.Optimizer,
         generator: torch.Generator, step: int, best_loss: float, norm: dict, provenance: dict,
         args: argparse.Namespace, draw_digest: str, training_seconds: float,
         augmentation_generator: torch.Generator | None = None, augmentation_digest: str = "") -> None:
    checkpoint = {
        "format_version": 1, "model_family": "diverse_rgbd_reference_gru", "model_config": rgbd_model_config_dict(model),
        "normalization": norm, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "step": step, "best_validation_loss": best_loss, "provenance": provenance, "args": vars(args),
        "draw_digest": draw_digest, "augmentation_digest": augmentation_digest, "training_seconds": training_seconds,
        "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(), "python": random.getstate(), "sampler": generator.get_state()},
    }
    if augmentation_generator is not None:
        checkpoint["rng"]["augmentation"] = augmentation_generator.get_state()
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=RGBD_ARMS, default="rgbd")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--candidate-patches", action=argparse.BooleanOptionalAction, default=True,
                        help="Encode 8m observed RGB-D patches aligned with each proposed command")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--patch-span-m", type=float, default=8.0)
    parser.add_argument("--patch-hidden", type=int, default=64)
    parser.add_argument("--rotation-augmentation", action=argparse.BooleanOptionalAction, default=False,
                        help="Matched random quarter-turns of current image and measured world pose")
    parser.add_argument("--progress-target", choices=tuple(PROGRESS_TARGET_VERSIONS), default="bounded_motion",
                        help="Explicit third-event labels; sustained_stall and bounded_motion require separate target/mask arrays")
    parser.add_argument("--preload-images", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--positive-sampling-fraction", type=float, default=0.0)
    parser.add_argument("--max-event-pos-weight", type=float, default=20.0)
    parser.add_argument("--xy-weight", type=float, default=1.0)
    parser.add_argument("--yaw-weight", type=float, default=0.5)
    parser.add_argument("--work-weight", type=float, default=0.2)
    parser.add_argument("--event-weight", type=float, default=1.0)
    parser.add_argument("--attitude-weight", type=float, default=.5)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-only", type=Path)
    parser.add_argument("--eval-controls", nargs="+", choices=("normal", "shuffle", "blank", "current_history"),
                        default=["normal", "shuffle", "blank"], help="Interventions on best checkpoint after training")
    parser.add_argument("--shuffle-seed", type=int, default=20260908)
    parser.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.steps, args.batch, args.eval_every, args.eval_batch, args.threads) < 1:
        parser.error("steps, batches, eval-every, and threads must be positive")
    if not 0 <= args.positive_sampling_fraction < 1:
        parser.error("positive-sampling-fraction must be in [0,1)")
    if args.max_event_pos_weight < 1:
        parser.error("max-event-pos-weight must be at least one")
    return args


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda" and args.evaluate_only is None:
        raise RuntimeError("Optimizer updates require an AMD-cluster GPU; CPU is evaluation-only")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; submit within an AMD-cluster GPU allocation")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.out.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only is None and (args.out / "last.pt").exists() and args.resume is None:
        raise FileExistsError("Output already contains last.pt; choose a fresh output or resume explicitly")

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
        print(line, flush=True)
        with (args.out / "train.log").open("a") as stream:
            stream.write(line + "\n")

    status = {"state": "loading", "model_family": "diverse_rgbd_reference_gru", "hostname": socket.gethostname(),
              "arm": args.arm, "seed": args.seed, "pid": os.getpid(), "device": str(device),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    write_json(args.out / "status.json", status)
    log(f"Loading measured RGB-D cache {args.data}; arm={args.arm} seed={args.seed}")
    manifest = json.loads((args.data / "manifest.json").read_text())
    if manifest.get("event_schema") != "fdm_diverse_events_v1_asset_or_chassis":
        raise ValueError("Expanded model requires the explicit asset-or-chassis event schema")
    val = load_low_dim(args.data / "val.npz")
    val_images = Images(args.data / "val_rgbd.npy", len(val["history"]), device, image_index=val.get("image_index"))
    loss_weights = {"xy": args.xy_weight, "yaw": args.yaw_weight, "work": args.work_weight, "events": args.event_weight, "attitude": args.attitude_weight}

    def final_controls(model: RGBDFiniteHorizonFDM, checkpoint_step: int, filename: str,
                       checkpoint_label: str = "best") -> None:
        report = {"checkpoint_step": checkpoint_step, "checkpoint_label": checkpoint_label,
                  "progress_event_definition": model.config.progress_event_definition,
                  "progress_event_version": model.config.progress_event_version, "controls": {}}
        for control in args.eval_controls:
            if control == "shuffle" and len(np.unique(val_images.index)) < 2:
                report["controls"][control] = {"available": False,
                    "reason": "Fewer than two distinct validation scene images"}
                continue
            metrics, predictions = evaluate_rgbd(model, val, val_images, args.eval_batch, device,
                                                 loss_weights, control, args.shuffle_seed)
            report["controls"][control] = metrics
            if args.save_predictions:
                prefix = "" if checkpoint_label == "best" else f"{checkpoint_label}_"
                np.savez_compressed(args.out / f"val_predictions_{prefix}{control}.npz", **predictions)
            log(f"{checkpoint_label}-step={checkpoint_step} control={control} ADE={metrics['pose']['ade_m']} FDE={metrics['pose']['fde_m']}")
        if "normal" in report["controls"]:
            normal = report["controls"]["normal"]
            report["image_dependence"] = {
                control: {"ade_delta_m": metrics["pose"]["ade_m"] - normal["pose"]["ade_m"],
                          "fde_delta_m": (metrics["pose"]["fde_m"] - normal["pose"]["fde_m"])
                          if metrics["pose"]["fde_m"] is not None and normal["pose"]["fde_m"] is not None else None}
                for control, metrics in report["controls"].items() if control != "normal" and "pose" in metrics}
        write_json(args.out / filename, report)

    if args.evaluate_only is not None:
        model, checkpoint = load_rgbd_checkpoint(args.evaluate_only, device)
        if model.config.progress_event_definition == "bounded_motion" and manifest.get("bounded_motion_version") != 2:
            raise ValueError("Bounded-motion evaluation requires its explicit version2 data manifest")
        val = select_progress_target(val, model.config.progress_event_definition)
        final_controls(model, checkpoint["step"], "evaluation_controls.json")
        write_json(args.out / "status.json", {**status, "state": "evaluated", "step": checkpoint["step"]})
        return

    train = load_low_dim(args.data / "train.npz")
    if args.progress_target == "bounded_motion" and manifest.get("bounded_motion_version") != 2:
        raise ValueError("Bounded-motion training requires its explicit version2 data manifest")
    train = select_progress_target(train, args.progress_target)
    val = select_progress_target(val, args.progress_target)
    norm = normalization(train, args.max_event_pos_weight)
    norm["event_head_definitions"] = {"contact": "contact", "rollover": "rollover", "low_progress": args.progress_target}
    norm["progress_event_version"] = PROGRESS_TARGET_VERSIONS[args.progress_target]
    if args.rotation_augmentation:
        # Exact statistics of the four quarter-turns of training poses. This
        # also handles focused datasets whose unrotated anchor position is
        # constant: augmentation must not create a thousand-sigma input.
        globals_ = train["global_features"].astype(np.float64)
        xy_std = max(float(np.sqrt(np.mean(globals_[:, 4] ** 2 + globals_[:, 5] ** 2) / 2)), 1e-3)
        yaw_std = max(float(np.sqrt(np.mean(globals_[:, 6] ** 2 + globals_[:, 7] ** 2) / 2)), 1e-3)
        norm["global_features"]["mean"][4:8] = [0.0] * 4
        norm["global_features"]["std"][4:8] = [xy_std, xy_std, yaw_std, yaw_std]
        norm["rotation_normalization"] = "Exact four-turn training-pose moments; no validation fitting"
    camera = manifest.get("camera", {})
    cfg = RGBDFDMConfig(arm=args.arm, image_size=int(manifest.get("image_size", 512)),
                       elevation_scale_m=float(manifest.get("elevation_scale_m", 40.)), horizon=train["nominal_pose"].shape[1], dt=float(manifest.get("output_dt_s", 0.2)),
                       candidate_patches=args.candidate_patches, patch_size=args.patch_size,
                       patch_span_m=args.patch_span_m, patch_hidden=args.patch_hidden,
                       camera_hfov_deg=float(camera.get("hfov_deg", math.degrees(camera.get("hfov_rad", math.radians(47.))))),
                       camera_height_m=float(camera.get("cam_height_m", 400.0)),
                       progress_event_definition=args.progress_target)
    model = RGBDFiniteHorizonFDM(cfg, norm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1729)
    augmentation_generator = torch.Generator(device="cpu").manual_seed(args.seed + 9172)
    log("Hashing immutable RGB-D data files and training sources")
    sources = [Path(__file__), ROOT / "src/nedm/traverse/fdm_diverse_model.py", ROOT / "scripts/traverse_fdm_train.py",
               ROOT / "src/nedm/traverse/fdm_model.py", ROOT / "src/nedm/traverse/fdm_diverse_data.py"]
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    data_files = ("manifest.json", "train.npz", "val.npz", "train_rgbd.npy", "val_rgbd.npy")
    provenance = {
        "data": {name: sha256(args.data / name) for name in data_files},
        "code": {str(path.relative_to(ROOT)): sha256(path) for path in sources if path.exists()},
        "git_head": git_head, "data_manifest": manifest, "hostname": socket.gethostname(),
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "device_name": torch.cuda.get_device_name(device), "input_source": "fixed pre-drive global RGB-D and causal measured vehicle history",
        "excluded_inputs": ["authored terrain", "BMP profile", "authored obstacle clearance", "future measured images", "future executed controls"],
        "architecture": "HMMWV adaptation: spatial CNN + history GRU + command-conditioned forward GRU + joint residual-velocity/risk heads",
        "candidate_patches": args.candidate_patches,
        "patch_projection": "Calibrated perspective with observed-depth height refinement; RGB-only/blank use zero-elevation approximation",
        "rotation_augmentation": args.rotation_augmentation,
        "augmentation_rng_seed": args.seed + 9172,
        "progress_event_definition": args.progress_target,
        "progress_event_version": model.config.progress_event_version,
        "validation_sampling": manifest.get("validation_distribution", manifest.get("selection", "See data manifest")),
        "selection": "best fixed validation masked loss; image controls are evaluation only; no protected test loading",
    }
    step, best_loss, training_seconds, draw_digest, augmentation_digest = 0, math.inf, 0.0, "", ""
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("model_family") != "diverse_rgbd_reference_gru" or checkpoint["model_config"] != rgbd_model_config_dict(model):
            raise ValueError("Resume model family or configuration differs")
        if checkpoint["provenance"]["data"] != provenance["data"] or checkpoint["provenance"]["code"] != provenance["code"]:
            raise ValueError("Resume data/source hashes differ; start a new output for changed experiments")
        for key in ("seed", "batch", "positive_sampling_fraction", "max_event_pos_weight", "learning_rate", "weight_decay",
                    "xy_weight", "yaw_weight", "work_weight", "event_weight", "attitude_weight", "grad_clip", "rotation_augmentation", "progress_target"):
            if checkpoint["args"].get(key) != vars(args)[key]:
                raise ValueError(f"Resume training setting changed: {key}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        torch.set_rng_state(checkpoint["rng"]["torch"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["rng"]["cuda"]])
        np.random.set_state(checkpoint["rng"]["numpy"])
        random.setstate(checkpoint["rng"]["python"])
        generator.set_state(checkpoint["rng"]["sampler"].cpu())
        if "augmentation" in checkpoint["rng"]:
            augmentation_generator.set_state(checkpoint["rng"]["augmentation"].cpu())
        elif args.rotation_augmentation:
            raise ValueError("Rotation-augmented resume is missing its augmentation RNG state")
        step, best_loss = int(checkpoint["step"]), float(checkpoint["best_validation_loss"])
        training_seconds, draw_digest = float(checkpoint.get("training_seconds", 0.0)), checkpoint.get("draw_digest", "")
        augmentation_digest = checkpoint.get("augmentation_digest", "")
        log(f"Resumed step={step} best-loss={best_loss:.6g}")
    write_json(args.out / "config.json", {"arguments": vars(args), "model": rgbd_model_config_dict(model),
                                          "loss_weights": loss_weights, "parameter_count": model.parameter_count()})
    write_json(args.out / "normalization.json", norm)
    write_json(args.out / "provenance.json", provenance)
    count = len(train["history"])
    train_images = Images(args.data / "train_rgbd.npy", count, device, image_index=train.get("image_index"))
    image_bytes = train_images.array.nbytes + val_images.array.nbytes
    free, _ = torch.cuda.mem_get_info(device)
    preload = args.preload_images == "cuda" or (args.preload_images == "auto" and image_bytes < free * 0.35)
    if preload:
        train_images.tensor = torch.from_numpy(np.array(train_images.array, copy=True)).to(device)
        val_images.tensor = torch.from_numpy(np.array(val_images.array, copy=True)).to(device)
    positive_mask = ((train["events"] > 0) & (train["event_mask"] > 0) & norm["supported_events"]).any(axis=(1, 2))
    positive_indices = torch.from_numpy(np.flatnonzero(positive_mask))
    if args.positive_sampling_fraction and not len(positive_indices):
        raise ValueError("Positive sampling requested with no supported positive training windows")
    tensors = {key: torch.from_numpy(value).to(device) for key, value in train.items()}
    del train
    log(f"train={count} val={len(val['history'])} params={model.parameter_count()} imageGB={image_bytes/1e9:.2f} preload={preload}")
    log(f"Event support={norm['supported_events']} counts={norm['event_counts']}")
    start_step, wall_start = step, time.perf_counter()
    model.train()
    while step < args.steps:
        start = time.perf_counter()
        indices = torch.randint(count, (args.batch,), generator=generator)
        sample_weight = None
        if args.positive_sampling_fraction:
            selected = torch.rand(args.batch, generator=generator) < args.positive_sampling_fraction
            indices[selected] = positive_indices[torch.randint(len(positive_indices), (int(selected.sum()),), generator=generator)]
            probability = np.full(args.batch, (1 - args.positive_sampling_fraction) / count)
            probability += positive_mask[indices.numpy()] * (args.positive_sampling_fraction / len(positive_indices))
            sample_weight = torch.from_numpy((1 / (count * probability)).astype(np.float32)).to(device)
        draw_digest = hashlib.sha256(draw_digest.encode() + indices.numpy().tobytes()).hexdigest()
        gpu_index = indices.to(device)
        batch = {key: value[gpu_index] for key, value in tensors.items()}
        batch["rgbd"] = train_images.batch(indices.numpy())
        if args.rotation_augmentation:
            rotations = torch.randint(4, (args.batch,), generator=augmentation_generator)
            augmentation_digest = hashlib.sha256(augmentation_digest.encode() + rotations.numpy().tobytes()).hexdigest()
            batch = rotate_world_batch(batch, rotations.to(device))
        if sample_weight is not None:
            batch["sample_weight"] = sample_weight
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss, components = fdm_loss(model, output, batch, loss_weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at update {step+1}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize(device)
        training_seconds += time.perf_counter() - start
        step += 1
        if step % args.eval_every == 0 or step == args.steps:
            metrics, predictions = evaluate_rgbd(model, val, val_images, args.eval_batch, device, loss_weights)
            metric_loss = metrics["loss"]["total"]
            is_best = metric_loss < best_loss
            best_loss = min(best_loss, metric_loss)
            summary = {"step": step, "train_loss": {key: float(value.detach()) for key, value in components.items()},
                       "validation": metrics, "gradient_norm": float(gradient_norm), "draw_digest": draw_digest,
                       "augmentation_digest": augmentation_digest,
                       "training_seconds": training_seconds, "updates_per_second": step / max(training_seconds, 1e-9),
                       "best_validation_loss": best_loss, "is_best": is_best,
                       "wall_seconds_this_invocation": time.perf_counter() - wall_start}
            with (args.out / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(json_clean(summary), sort_keys=True) + "\n")
            write_json(args.out / "metrics_last.json", summary)
            save(args.out / "last.pt", model, optimizer, generator, step, best_loss, norm, provenance, args, draw_digest, training_seconds, augmentation_generator, augmentation_digest)
            if step == args.steps and args.save_predictions:
                np.savez_compressed(args.out / "val_predictions_last.npz", **predictions)
            if is_best:
                save(args.out / "best.pt", model, optimizer, generator, step, best_loss, norm, provenance, args, draw_digest, training_seconds, augmentation_generator, augmentation_digest)
                write_json(args.out / "metrics_best.json", summary)
                if args.save_predictions:
                    np.savez_compressed(args.out / "val_predictions_best.npz", **predictions)
            write_json(args.out / "status.json", {**status, "state": "running", "step": step, "target_steps": args.steps,
                                                   "best_validation_loss": best_loss, "draw_digest": draw_digest})
            log(f"step={step}/{args.steps} train={float(loss):.5f} val={metric_loss:.5f} ADE={metrics['pose']['ade_m']} "
                f"FDE={metrics['pose']['fde_m']} updates/s={step/max(training_seconds,1e-9):.1f} best={is_best}")
    # The fixed final update and validation-selected best receive distinct
    # image interventions. Neither intervention changes checkpoint selection.
    final_controls(model, step, "last_controls.json", "last")
    best_model, best_checkpoint = load_rgbd_checkpoint(args.out / "best.pt", device)
    final_controls(best_model, best_checkpoint["step"], "best_controls.json")
    write_json(args.out / "status.json", {**status, "state": "complete", "step": step, "new_updates": step - start_step,
               "training_seconds": training_seconds, "wall_seconds_this_invocation": time.perf_counter() - wall_start,
               "updates_per_second": step / max(training_seconds, 1e-9), "draw_digest": draw_digest,
               "augmentation_digest": augmentation_digest,
               "best_validation_loss": best_loss, "best_step": best_checkpoint["step"]})
    log(f"Complete; best={args.out/'best.pt'} last={args.out/'last.pt'}")


def main() -> None:
    args = arguments()
    try:
        run(args)
    except Exception as error:
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(args.out / "status.json", {"state": "failed", "exception": repr(error), "traceback": traceback.format_exc(),
                                               "arm": args.arm, "seed": args.seed})
        raise


if __name__ == "__main__":
    main()
