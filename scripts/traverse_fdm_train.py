#!/usr/bin/env python3
"""Train/evaluate the isolated finite-horizon HMMWV PID outcome-model pilot.

Training belongs on AMD cluster GPUs. This script never collects simulation
data. ``--evaluate-only`` is suitable for read-only CPU checkpoint evaluation.
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

from nedm.traverse.fdm_model import (ARMS, EVENT_NAMES, FDMConfig, FiniteHorizonFDM,
                                     fdm_loss, load_fdm_checkpoint, model_config_dict)

ARRAY_KEYS = ("history", "candidate", "global_features", "nominal_pose", "trajectory",
              "work", "events", "trajectory_mask", "event_mask")
INPUT_KEYS = ("history", "candidate", "global_features", "nominal_pose")


def json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_clean(x) for x in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_clean(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = set(ARRAY_KEYS) - set(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays {sorted(missing)}")
        result = {k: np.ascontiguousarray(data[k], dtype=np.float32) for k in ARRAY_KEYS}
        if "work_mask" in data.files:
            result["work_mask"] = np.ascontiguousarray(data["work_mask"], dtype=np.float32)
        for key in ("episode_index", "anchor"):
            if key in data.files:
                result[key] = np.ascontiguousarray(data[key], dtype=np.int64)
    count = len(result["history"])
    if count == 0 or any(len(x) != count for x in result.values()):
        raise ValueError(f"{path}: empty or inconsistent batch dimensions")
    if any(not np.isfinite(x).all() for x in result.values()):
        raise ValueError(f"{path}: every feature and masked target must be finite")
    if result["trajectory"].shape != result["nominal_pose"].shape or result["trajectory"].shape[-1] != 4:
        raise ValueError(f"{path}: expected matching actual/nominal (N,H,4) poses")
    horizon = result["trajectory"].shape[1]
    for key, channels in (("work", 1), ("events", 3), ("trajectory_mask", 1), ("event_mask", 3)):
        if result[key].shape != (count, horizon, channels):
            raise ValueError(f"{path}: {key} must have shape {(count, horizon, channels)}")
    for key in ("trajectory_mask", "event_mask"):
        if not np.isin(result[key], [0.0, 1.0]).all():
            raise ValueError(f"{path}: {key} must be binary")
    if not np.isin(result["events"], [0.0, 1.0]).all():
        raise ValueError(f"{path}: event labels must be binary")
    return result


def make_normalization(train: dict[str, np.ndarray], max_pos_weight: float) -> dict[str, Any]:
    """Statistics from training tensors only; no validation-dependent fitting."""
    result: dict[str, Any] = {}
    for key in ("history", "candidate", "global_features"):
        array = train[key].reshape(-1, train[key].shape[-1]).astype(np.float64)
        # Padded candidate rows must not distort profile feature normalization.
        if key == "candidate" and train[key].shape[-1] == 13:
            valid = train[key][..., 12].reshape(-1) > 0
            if valid.any():
                array = array[valid]
        result[key] = {"mean": array.mean(axis=0).tolist(), "std": np.maximum(array.std(axis=0), 1e-3).tolist()}
        if key == "candidate" and train[key].shape[-1] == 13:
            result[key]["mean"][12], result[key]["std"][12] = 0.0, 1.0
    mask = train["trajectory_mask"][..., 0] > 0
    residual = (train["trajectory"][..., :2] - train["nominal_pose"][..., :2])[mask]
    result["xy_scale"] = max(float(np.sqrt(np.mean(residual.astype(np.float64) ** 2))), 0.5) if residual.size else 1.0
    work_valid = train.get("work_mask", train["trajectory_mask"])[..., 0] > 0
    valid_work = train["work"][..., 0][work_valid]
    result["work_scale"] = max(float(np.quantile(valid_work, 0.9)), 1.0) if valid_work.size else 1.0
    event_valid = train["event_mask"].astype(np.float64)
    positive = (train["events"] * event_valid).sum(axis=(0, 1))
    total = event_valid.sum(axis=(0, 1))
    pos_weight = np.where(positive > 0, (total - positive) / np.maximum(positive, 1), 1.0)
    result["event_pos_weight"] = np.clip(pos_weight, 1.0, max_pos_weight).tolist()
    result["supported_events"] = ((positive > 0) & (total > positive)).tolist()
    result["event_counts"] = {name: {"positive": int(positive[i]), "negative": int(total[i] - positive[i])}
                              for i, name in enumerate(EVENT_NAMES)}
    if "episode_index" in train:
        for i, name in enumerate(EVENT_NAMES):
            observed = event_valid[..., i].any(axis=1)
            event_positive = ((train["events"][..., i] > 0) & (event_valid[..., i] > 0)).any(axis=1)
            result["event_counts"][name]["observed_unique_episodes"] = int(np.unique(train["episode_index"][observed]).size)
            result["event_counts"][name]["positive_unique_episodes"] = int(np.unique(train["episode_index"][event_positive]).size)
    result["source"] = "training split only; candidate padding excluded"
    result["event_logit_correction"] = "BCE(logits + log(pos_weight), labels, pos_weight); inference uses logits"
    return result


def binary_metrics(label: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    label = np.asarray(label, dtype=np.int64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    n = len(label)
    positive = int(label.sum())
    negative = n - positive
    result: dict[str, Any] = {"count": n, "positive": positive, "negative": negative,
                             "auroc": None, "average_precision": None, "brier": None,
                             "ece": None, "unsupported": None}
    if n == 0:
        result["unsupported"] = "no observed labels"
        return result
    result["brier"] = float(np.mean((score - label) ** 2))
    bins = []
    ece = 0.0
    for lower in np.linspace(0, 0.9, 10):
        upper = lower + 0.1
        included = (score >= lower) & ((score < upper) if upper < 0.999 else (score <= 1))
        if included.any():
            prediction, frequency = float(score[included].mean()), float(label[included].mean())
            ece += included.mean() * abs(prediction - frequency)
            bins.append({"lower": float(lower), "count": int(included.sum()), "prediction": prediction, "frequency": frequency})
    result["ece"], result["calibration_bins"] = float(ece), bins
    if positive and negative:
        order = np.argsort(score, kind="stable")
        sorted_score = score[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_score)) + 1]
        ends = np.r_[starts[1:], n]
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.repeat((starts + 1 + ends) / 2.0, ends - starts)
        result["auroc"] = float((ranks[label == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))
        descending = np.argsort(-score, kind="stable")
        desc_score, desc_label = score[descending], label[descending]
        end = np.r_[np.flatnonzero(np.diff(desc_score)) + 1, n]
        tp = np.cumsum(desc_label)[end - 1]
        recall, precision = tp / positive, tp / end
        result["average_precision"] = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    else:
        result["unsupported"] = "zero positive events" if not positive else "zero negative events"
    for threshold in (0.1, 0.5):
        accepted = score < threshold
        result[f"threshold_{threshold}"] = {
            "accepted": int(accepted.sum()), "false_accepts": int(label[accepted].sum()),
            "false_accept_rate": float(label[accepted].mean()) if accepted.any() else None,
            "failure_recall": float((~accepted & (label == 1)).sum() / positive) if positive else None,
        }
    return result


def retention_metrics(label: np.ndarray, score: np.ndarray) -> list[dict[str, Any]]:
    """Risk sorting at fixed acceptance fractions; compare arms at equal coverage."""
    order = np.argsort(score, kind="stable")
    rows = []
    for fraction in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        count = min(len(label), max(1, int(math.floor(len(label) * fraction)))) if len(label) else 0
        kept = order[:count]
        rows.append({"target_retention": fraction, "retained": count,
                     "retention": count / len(label) if len(label) else None,
                     "false_accepts": int(label[kept].sum()),
                     "false_accept_rate": float(label[kept].mean()) if count else None,
                     "threshold": float(score[kept[-1]]) if count else None,
                     "auc_supported": bool(label.sum() and (1 - label).sum())})
    return rows


def pose_metrics(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    valid = mask[..., 0] > 0
    error = np.linalg.norm(predicted[..., :2] - target[..., :2], axis=-1)
    yaw = np.arctan2(predicted[..., 2], predicted[..., 3]) - np.arctan2(target[..., 2], target[..., 3])
    yaw_error = np.abs(np.arctan2(np.sin(yaw), np.cos(yaw))) * 180.0 / np.pi
    final = valid[:, -1]
    counts = valid.sum(axis=0)
    return {
        "observed_points": int(valid.sum()), "complete_horizon_windows": int(final.sum()),
        "ade_m": float(error[valid].mean()) if valid.any() else None,
        "fde_m": float(error[final, -1].mean()) if final.any() else None,
        "yaw_mae_deg": float(yaw_error[valid].mean()) if valid.any() else None,
        "final_yaw_mae_deg": float(yaw_error[final, -1].mean()) if final.any() else None,
        "ade_by_step_m": [float(error[valid[:, i], i].mean()) if counts[i] else None for i in range(error.shape[1])],
        "observed_points_by_step": counts.tolist(),
    }


def outcome_metrics(output: dict[str, np.ndarray], data: dict[str, np.ndarray],
                    supported_events: np.ndarray | None = None, dt: float = 0.2) -> dict[str, Any]:
    result: dict[str, Any] = {"pose": pose_metrics(output["trajectory"], data["trajectory"], data["trajectory_mask"]),
                             "nominal_kinematics": pose_metrics(data["nominal_pose"], data["trajectory"], data["trajectory_mask"])}
    work_mask = data.get("work_mask", data["trajectory_mask"])[..., 0] > 0
    error = output["work"][..., 0] - data["work"][..., 0]
    last = work_mask[:, -1]
    result["work"] = {
        "observed_points": int(work_mask.sum()),
        "unit": "kJ",
        "mae_kj": float(np.abs(error[work_mask]).mean()) if work_mask.any() else None,
        "final_mae_kj": float(np.abs(error[last, -1]).mean()) if last.any() else None,
        "final_bias_kj": float(error[last, -1].mean()) if last.any() else None,
        "final_truth_mean_kj": float(data["work"][last, -1, 0].mean()) if last.any() else None,
    }
    probability = 1.0 / (1.0 + np.exp(-np.clip(output["event_logits"], -80.0, 80.0)))
    supported = np.asarray([True, True, True] if supported_events is None else supported_events, dtype=bool)
    eligible = np.ones(data["events"].shape[1:], dtype=bool)
    eligible[:, 2] = (np.arange(1, len(eligible) + 1) * dt) >= 2.0 - 1e-6
    result["events"] = {}
    for i, name in enumerate(EVENT_NAMES):
        valid = data["event_mask"][..., i] > 0
        label = data["events"][..., i]
        metrics = binary_metrics(label[valid], probability[..., i][valid])
        # A positive is known once observed; a negative needs the complete
        # horizon. Censored non-events are excluded from window-risk metrics.
        positive = ((label > 0) & valid).any(axis=1)
        known = (positive | (valid | ~eligible[:, i]).all(axis=1)) & eligible[:, i].any()
        window_probability = np.where(eligible[:, i], probability[..., i], 0.0).max(axis=1)[known]
        window_label = positive[known].astype(np.int64)
        metrics["training_supported"] = bool(supported[i])
        if not supported[i]:
            metrics["operational_use"] = "disabled: training split lacks positive or negative examples"
        metrics["window"] = binary_metrics(window_label, window_probability)
        metrics["window"]["false_accepts_at_retention"] = retention_metrics(window_label, window_probability)
        metrics["window"]["excluded_censored"] = int((~known).sum())
        last_known = valid[:, -1] & eligible[-1, i]
        metrics["last_horizon"] = binary_metrics(label[last_known, -1], probability[last_known, -1, i])
        metrics["last_horizon"]["horizon_seconds"] = float(len(eligible) * dt)
        metrics["last_horizon"]["false_accepts_at_retention"] = retention_metrics(
            label[last_known, -1].astype(np.int64), probability[last_known, -1, i])
        if "episode_index" in data:
            episodes = data["episode_index"]
            metrics["observed_unique_episodes"] = int(np.unique(episodes[valid.any(axis=1)]).size)
            metrics["positive_unique_episodes"] = int(np.unique(episodes[positive]).size)
            metrics["window"]["known_unique_episodes"] = int(np.unique(episodes[known]).size)
            metrics["window"]["positive_unique_episodes"] = int(np.unique(episodes[positive & known]).size)
            metrics["last_horizon"]["known_unique_episodes"] = int(np.unique(episodes[last_known]).size)
            metrics["last_horizon"]["positive_unique_episodes"] = int(np.unique(episodes[last_known & (label[:, -1] > 0)]).size)
        result["events"][name] = metrics
    valid = data["event_mask"] > 0
    evaluated = eligible & supported[None, :]
    positive = ((data["events"] > 0) & valid & evaluated).any(axis=(1, 2))
    known = (positive | (valid | ~evaluated).all(axis=(1, 2))) & evaluated.any()
    window_score = np.where(evaluated, probability, 0.0).max(axis=(1, 2))[known]
    window_label = positive[known].astype(np.int64)
    result["any_event_window"] = binary_metrics(window_label, window_score)
    result["any_event_window"]["false_accepts_at_retention"] = retention_metrics(window_label, window_score)
    result["any_event_window"]["excluded_censored"] = int((~known).sum())
    result["any_event_window"]["included_heads"] = [name for i, name in enumerate(EVENT_NAMES) if supported[i]]
    if "episode_index" in data:
        result["any_event_window"]["known_unique_episodes"] = int(np.unique(data["episode_index"][known]).size)
        result["any_event_window"]["positive_unique_episodes"] = int(np.unique(data["episode_index"][positive & known]).size)
    result["risk_interpretation"] = "Window score is maximum per-step event probability, not an independent-hazard product."
    return result


@torch.inference_mode()
def evaluate(model: FiniteHorizonFDM, arrays: dict[str, np.ndarray], batch_size: int,
             device: torch.device, loss_weights: dict[str, float]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    outputs: dict[str, list[np.ndarray]] = {"trajectory": [], "work": [], "event_logits": []}
    sums: dict[str, float] = {}
    denominators: dict[str, float] = {}
    count = len(arrays["history"])
    start = time.perf_counter()
    for offset in range(0, count, batch_size):
        batch = {key: torch.from_numpy(value[offset:offset + batch_size]).to(device) for key, value in arrays.items()}
        output = model(batch)
        _, losses = fdm_loss(model, output, batch, loss_weights)
        mask_counts = {"xy": float(batch["trajectory_mask"].sum()) * 2,
                       "yaw": float(batch["trajectory_mask"].sum()) * 2,
                       "work": float(batch.get("work_mask", batch["trajectory_mask"]).sum()),
                       "events": float((batch["event_mask"] * model.supported_events).sum())}
        for key, denominator in mask_counts.items():
            sums[key] = sums.get(key, 0.0) + float(losses[key]) * denominator
            denominators[key] = denominators.get(key, 0.0) + denominator
        for key in outputs:
            outputs[key].append(output[key].cpu().numpy())
    merged = {key: np.concatenate(value, axis=0) for key, value in outputs.items()}
    metrics = outcome_metrics(merged, arrays, model.supported_events.cpu().numpy(), model.config.dt)
    metrics["loss"] = {key: value / max(denominators[key], 1.0) for key, value in sums.items()}
    metrics["loss"]["total"] = sum(loss_weights.get(key, 0.0) * value for key, value in metrics["loss"].items())
    metrics["windows"] = count
    metrics["evaluation_seconds"] = time.perf_counter() - start
    model.train()
    return metrics, merged


def save_checkpoint(path: Path, model: FiniteHorizonFDM, optimizer: torch.optim.Optimizer,
                    generator: torch.Generator, step: int, best_loss: float, normalization: dict,
                    provenance: dict, args: argparse.Namespace, draw_digest: str, training_seconds: float) -> None:
    checkpoint = {
        "format_version": 1, "model_config": model_config_dict(model), "normalization": normalization,
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "step": step, "best_validation_loss": best_loss, "provenance": provenance,
        "args": vars(args), "draw_digest": draw_digest, "training_seconds": training_seconds,
        "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "numpy": np.random.get_state(), "python": random.getstate(), "sampler": generator.get_state()},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing train.npz, val.npz, manifest.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, default="history")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=1000, help="Total update count, including resumed updates")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batch", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--preload-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--positive-sampling-fraction", type=float, default=0.0,
                        help="Optional event-positive mixture with inverse-probability loss correction")
    parser.add_argument("--max-event-pos-weight", type=float, default=20.0)
    parser.add_argument("--xy-weight", type=float, default=1.0)
    parser.add_argument("--yaw-weight", type=float, default=0.5)
    parser.add_argument("--work-weight", type=float, default=0.2)
    parser.add_argument("--event-weight", type=float, default=1.0)
    parser.add_argument("--resume", type=Path, help="Trusted checkpoint, usually OUT/last.pt")
    parser.add_argument("--evaluate-only", type=Path, help="Read-only model evaluation of fixed val split; no optimizer updates")
    parser.add_argument("--save-predictions", action="store_true", help="Save best natural-validation prediction arrays")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.steps < 1 or args.batch < 1 or args.eval_every < 1 or args.eval_batch < 1:
        parser.error("steps, batch, eval-every, and eval-batch must be positive")
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
        raise RuntimeError("Training is restricted to cluster GPUs. CPU is supported only with --evaluate-only.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable; run training in an AMD-cluster GPU allocation")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.out.mkdir(parents=True, exist_ok=True)
    if args.evaluate_only is None and (args.out / "last.pt").exists() and args.resume is None:
        raise FileExistsError("Output already contains last.pt; choose a new output or pass --resume")
    log_path = args.out / "train.log"

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a") as stream:
            stream.write(line + "\n")

    loss_weights = {"xy": args.xy_weight, "yaw": args.yaw_weight, "work": args.work_weight, "events": args.event_weight}
    status = {"state": "loading", "hostname": socket.gethostname(), "arm": args.arm, "seed": args.seed,
              "pid": os.getpid(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "device": str(device)}
    write_json(args.out / "status.json", status)
    log(f"Loading fixed data cache {args.data}; arm={args.arm}, seed={args.seed}, device={device}")
    val = load_arrays(args.data / "val.npz")
    if args.evaluate_only is not None:
        model, checkpoint = load_fdm_checkpoint(args.evaluate_only, device)
        metrics, predictions = evaluate(model, val, args.eval_batch, device, loss_weights)
        write_json(args.out / "evaluation.json", {"checkpoint": str(args.evaluate_only), "step": checkpoint["step"], "metrics": metrics})
        if args.save_predictions:
            np.savez_compressed(args.out / "evaluation_predictions.npz", **predictions)
        write_json(args.out / "status.json", {**status, "state": "evaluated", "step": checkpoint["step"]})
        log(f"Evaluation complete; ADE={metrics['pose']['ade_m']} FDE={metrics['pose']['fde_m']}")
        return
    train = load_arrays(args.data / "train.npz")
    norm = make_normalization(train, args.max_event_pos_weight)
    config = FDMConfig(history_dim=train["history"].shape[-1], candidate_dim=train["candidate"].shape[-1],
                       global_dim=train["global_features"].shape[-1], horizon=train["trajectory"].shape[1],
                       dt=args.dt, arm=args.arm, hidden_dim=args.hidden_dim)
    model = FiniteHorizonFDM(config, norm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1729)
    manifest = json.loads((args.data / "manifest.json").read_text()) if (args.data / "manifest.json").exists() else {}
    source_paths = [Path(__file__), ROOT / "src/nedm/traverse/fdm_model.py", ROOT / "src/nedm/traverse/fdm_data.py"]
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    provenance = {
        "data": {name: sha256(args.data / name) for name in ("train.npz", "val.npz", "manifest.json") if (args.data / name).exists()},
        "code": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths if path.exists()},
        "git_head": git_head, "data_manifest": manifest, "hostname": socket.gethostname(),
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "device_name": torch.cuda.get_device_name(device), "training_domain": "one fixed recorded PID controller domain",
        "selection": "best fixed validation masked loss; protected test split not read",
    }
    step, best_loss, accumulated_training_seconds = 0, math.inf, 0.0
    draw_digest = ""
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["model_config"] != model_config_dict(model) or checkpoint["provenance"]["data"] != provenance["data"]:
            raise ValueError("Resume model configuration or data hashes differ")
        if checkpoint["provenance"]["code"] != provenance["code"]:
            raise ValueError("Resume source hashes differ; use a new run for changed code")
        for key in ("seed", "batch", "positive_sampling_fraction", "max_event_pos_weight", "learning_rate", "weight_decay",
                    "xy_weight", "yaw_weight", "work_weight", "event_weight", "grad_clip"):
            if checkpoint["args"].get(key) != vars(args)[key]:
                raise ValueError(f"Resume training setting changed: {key}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        torch.set_rng_state(checkpoint["rng"]["torch"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["rng"]["cuda"]])
        np.random.set_state(checkpoint["rng"]["numpy"])
        random.setstate(checkpoint["rng"]["python"])
        generator.set_state(checkpoint["rng"]["sampler"].cpu())
        step, best_loss = int(checkpoint["step"]), float(checkpoint["best_validation_loss"])
        accumulated_training_seconds = float(checkpoint.get("training_seconds", 0.0))
        draw_digest = checkpoint.get("draw_digest", "")
        log(f"Resumed update {step}; best validation loss={best_loss:.6g}")
    write_json(args.out / "config.json", {"arguments": vars(args), "model": model_config_dict(model),
                                          "loss_weights": loss_weights, "parameter_count": model.parameter_count()})
    write_json(args.out / "normalization.json", norm)
    write_json(args.out / "provenance.json", provenance)
    positive_mask = ((train["events"] > 0) & (train["event_mask"] > 0)).any(axis=(1, 2))
    positive_indices = torch.from_numpy(np.flatnonzero(positive_mask))
    count = len(train["history"])
    if args.positive_sampling_fraction and not len(positive_indices):
        raise ValueError("Positive sampling requested but training data contain no observed positive windows")
    tensors = {key: torch.from_numpy(value) for key, value in train.items()}
    cache_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
    preload = args.preload_device == "cuda"
    if args.preload_device == "auto":
        free, _ = torch.cuda.mem_get_info(device)
        preload = cache_bytes < free * 0.45
    if preload:
        tensors = {key: value.to(device) for key, value in tensors.items()}
    del train
    cache_device = device if preload else torch.device("cpu")
    log(f"windows train={count} val={len(val['history'])}; params={model.parameter_count()}; cacheMB={cache_bytes/1e6:.1f} preload={preload}")
    log(f"Event counts {norm['event_counts']}; positive weights {norm['event_pos_weight']}")
    training_seconds = accumulated_training_seconds
    start_step = step
    wall_start = time.perf_counter()
    model.train()
    latest_loss = None
    while step < args.steps:
        start = time.perf_counter()
        indices = torch.randint(count, (args.batch,), generator=generator)
        sample_weight = None
        if args.positive_sampling_fraction:
            selected = torch.rand(args.batch, generator=generator) < args.positive_sampling_fraction
            positive_draw = torch.randint(len(positive_indices), (int(selected.sum()),), generator=generator)
            indices[selected] = positive_indices[positive_draw]
            probability = np.full(args.batch, (1 - args.positive_sampling_fraction) / count)
            probability += positive_mask[indices.numpy()] * (args.positive_sampling_fraction / len(positive_indices))
            sample_weight = torch.from_numpy((1 / (count * probability)).astype(np.float32)).to(device)
        # A chain hash is resumable and proves identical indexed draws across arms.
        draw_digest = hashlib.sha256(draw_digest.encode() + indices.numpy().tobytes()).hexdigest()
        index = indices.to(cache_device)
        batch = {key: value[index].to(device) for key, value in tensors.items()}
        if sample_weight is not None:
            batch["sample_weight"] = sample_weight
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss, components = fdm_loss(model, output, batch, loss_weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at update {step + 1}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize(device)
        training_seconds += time.perf_counter() - start
        step += 1
        latest_loss = {key: float(value.detach()) for key, value in components.items()}
        if step % args.eval_every == 0 or step == args.steps:
            metrics, predictions = evaluate(model, val, args.eval_batch, device, loss_weights)
            metric_loss = metrics["loss"]["total"]
            is_best = metric_loss < best_loss
            if is_best:
                best_loss = metric_loss
            summary = {"step": step, "train_loss": latest_loss, "validation": metrics,
                       "gradient_norm": float(gradient_norm), "draw_digest": draw_digest,
                       "training_seconds": training_seconds, "updates_per_second": step / max(training_seconds, 1e-9),
                       "wall_seconds_this_invocation": time.perf_counter() - wall_start,
                       "best_validation_loss": best_loss, "is_best": is_best}
            with (args.out / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(json_clean(summary), sort_keys=True) + "\n")
            write_json(args.out / "metrics_last.json", summary)
            save_checkpoint(args.out / "last.pt", model, optimizer, generator, step, best_loss, norm,
                            provenance, args, draw_digest, training_seconds)
            if is_best:
                save_checkpoint(args.out / "best.pt", model, optimizer, generator, step, best_loss, norm,
                                provenance, args, draw_digest, training_seconds)
                write_json(args.out / "metrics_best.json", summary)
                if args.save_predictions:
                    np.savez_compressed(args.out / "val_predictions_best.npz", **predictions)
            write_json(args.out / "status.json", {**status, "state": "running", "step": step,
                       "target_steps": args.steps, "best_validation_loss": best_loss,
                       "updates_per_second": step / max(training_seconds, 1e-9), "draw_digest": draw_digest})
            log(f"step={step}/{args.steps} train={latest_loss['total']:.5f} val={metric_loss:.5f} "
                f"ADE={metrics['pose']['ade_m']} FDE={metrics['pose']['fde_m']} "
                f"updates/s={step/max(training_seconds,1e-9):.1f} best={is_best}")
    write_json(args.out / "status.json", {**status, "state": "complete", "step": step,
               "new_updates": step - start_step, "training_seconds": training_seconds,
               "wall_seconds_this_invocation": time.perf_counter() - wall_start,
               "updates_per_second": step / max(training_seconds, 1e-9), "draw_digest": draw_digest,
               "best_validation_loss": best_loss})
    log(f"Complete; best checkpoint={args.out / 'best.pt'}; last checkpoint={args.out / 'last.pt'}")


def main() -> None:
    args = arguments()
    try:
        run(args)
    except Exception as error:
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(args.out / "status.json", {"state": "failed", "exception": repr(error),
                                               "traceback": traceback.format_exc(), "arm": args.arm, "seed": args.seed})
        raise


if __name__ == "__main__":
    main()
