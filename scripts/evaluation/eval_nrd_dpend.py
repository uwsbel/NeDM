"""NRD double-pendulum evaluation: autonomous rollouts, curves, and demo video.

Implements the study plan's section 11 evaluation for the trained joint model:

- AUTONOMOUS mode (the real surrogate test): only the first ``block_size`` camera
  frames are encoded; afterwards both z1 and z2 are predicted recursively and
  frames are DECODED from predicted latents.
- Observation-anchored mode: the true frame is encoded at every step (z1 still
  rolls autonomously) -- isolates how much visual grounding helps the latent head.
- Baselines: the matched state-only NeDM model (z1 fidelity reference, gate G3)
  and persistence (state frozen at the last context step).
- Cross-modal consistency (11.3): the tip pixel found in each DECODED frame is
  compared against the pinhole projection of the PREDICTED z1 tip -- do the two
  heads describe the same future?

Outputs (to --output-dir): summary.json, error-growth figure, and side-by-side
GIFs (Chrono truth | NRD decoded prediction | absolute difference).

Run:
    PYTHONPATH=src conda run -n nedm python scripts/evaluation/eval_nrd_dpend.py \
        --nrd-checkpoint artifacts/training_runs/dpend_nrd_v1/checkpoints/best_val.pt \
        --state-checkpoint artifacts/training_runs/dpend_state_v1/checkpoints/best_val.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nedm.double_pendulum_data import find_tip_pixel, project_to_pixel  # noqa: E402
from nedm.nrd.checkpoint import load_nrd_model as load_nrd  # noqa: E402
from nedm.nrd.model import NRDDynamicsModel  # noqa: E402
from nedm.nrd.trainer import load_rollout_split_with_frames  # noqa: E402
from nedm.nrd.vision import frames_to_uint8  # noqa: E402
from nedm.training.model import HMMWVDynamicsModel  # noqa: E402
from nedm.training.trainer import pendulum_tip_positions  # noqa: E402

SERIES_NRD = "#2a78d6"       # categorical slot 1 (blue)
SERIES_STATE = "#eb6834"     # categorical slot 2 (orange)
SERIES_ANCHOR = "#1baf7a"    # categorical slot 3 (aqua)
SERIES_REFERENCE = "#8a8a85"  # persistence: recessive reference


def load_state_only(checkpoint_path: Path, device: torch.device) -> HMMWVDynamicsModel:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = payload["config"]
    metadata = payload["metadata"]
    model = HMMWVDynamicsModel(
        state_dim=len(metadata["state_fields"]),
        action_dim=len(metadata["action_fields"]),
        target_dim=len(metadata["state_fields"]),
        transformer_cfg=config["model"],
        normalization=metadata["normalization"],
        state_fields=list(metadata["state_fields"]),
        dt_s=float(metadata["dt_s"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def autonomous_rollout(
    model: NRDDynamicsModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    frames: torch.Tensor,
    block: int,
    steps: int,
    anchored: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(predicted z1 (E, steps, S), predicted z2 (E, steps, Z)).

    ``anchored=True`` feeds encoder(true frame) as the latent INPUT each step
    (the latent head still predicts, so its output is comparable), z1 rolls
    autonomously in both modes.
    """
    z2_window = model.encode_frame_sequence(frames[:, :block])
    state_window = states[:, :block].clone()
    out_states, out_z2 = [], []
    for step in range(steps):
        window_actions = actions[:, step : block + step]
        next_state, next_z2 = model.predict_next(state_window, z2_window, window_actions)
        out_states.append(next_state)
        out_z2.append(next_z2)
        if anchored:
            true_latent = model.encode_frame_sequence(
                frames[:, block + step : block + step + 1]
            )[:, 0]
            fed_z2 = true_latent
        else:
            fed_z2 = next_z2
        state_window = torch.cat([state_window[:, 1:], next_state.unsqueeze(1)], dim=1)
        z2_window = torch.cat([z2_window[:, 1:], fed_z2.unsqueeze(1)], dim=1)
    return torch.stack(out_states, dim=1), torch.stack(out_z2, dim=1)


@torch.no_grad()
def state_only_rollout(
    model: HMMWVDynamicsModel, states: torch.Tensor, actions: torch.Tensor, block: int, steps: int
) -> torch.Tensor:
    state_window = states[:, :block].clone()
    out = []
    for step in range(steps):
        window_actions = actions[:, step : block + step]
        delta = model.predict_next_delta(state_window, window_actions)
        next_state = state_window[:, -1] + delta
        out.append(next_state)
        state_window = torch.cat([state_window[:, 1:], next_state.unsqueeze(1)], dim=1)
    return torch.stack(out, dim=1)


def tip_error_curve(
    pred_states: torch.Tensor, gt_tip: torch.Tensor, trig_idx: torch.Tensor, link_lengths: list[float]
) -> np.ndarray:
    """Mean tip position error (m) at each horizon step, over episodes."""
    pred_tip = pendulum_tip_positions(pred_states[..., trig_idx], link_lengths)
    err = (pred_tip - gt_tip).pow(2).sum(-1).sqrt()  # (E, steps)
    return err.mean(0).cpu().numpy()


def make_video(
    gt_frames: np.ndarray, pred_frames: np.ndarray, out_path: Path, fps: int = 25, scale: int = 2
) -> None:
    """Side-by-side GIF: truth | prediction | absolute difference."""
    from PIL import Image

    frames = []
    gap = 255 * np.ones((gt_frames.shape[1], 4, 3), dtype=np.uint8)
    for k in range(gt_frames.shape[0]):
        diff = np.abs(gt_frames[k].astype(np.int16) - pred_frames[k].astype(np.int16)).astype(np.uint8)
        row = np.concatenate([gt_frames[k], gap, pred_frames[k], gap, diff], axis=1)
        image = Image.fromarray(row)
        image = image.resize((row.shape[1] * scale, row.shape[0] * scale), Image.NEAREST)
        frames.append(image)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nrd-checkpoint", type=Path, required=True)
    parser.add_argument("--state-checkpoint", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None,
                        help="Defaults to the checkpoint config's processed_dataset_dir.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/nrd_eval/dpend_v1"))
    parser.add_argument("--num-episodes", type=int, default=24)
    parser.add_argument("--horizon-s", type=float, default=3.0)
    parser.add_argument("--video-episodes", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          args.device if args.device != "auto" else "cpu")
    torch.set_grad_enabled(False)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, payload = load_nrd(args.nrd_checkpoint.resolve(), device)
    config = payload["config"]
    metadata = payload["metadata"]
    processed_root = (
        args.processed_dir.resolve()
        if args.processed_dir is not None
        else Path(config["processed_dataset_dir"]).resolve()
    )
    block = int(config["model"]["block_size"])
    dt_s = float(metadata["dt_s"])
    steps = int(round(args.horizon_s / dt_s))
    link_lengths = config.get("rollout_eval", {}).get("link_lengths", [0.3, 0.3])
    state_fields = list(metadata["state_fields"])
    trig_idx = torch.tensor(
        [state_fields.index(f) for f in ("cos_q1", "sin_q1", "cos_q2", "sin_q2")],
        dtype=torch.long, device=device,
    )

    split_data = load_rollout_split_with_frames(processed_root, "val")
    minimum_rows = block + steps + 1
    episodes = [ep for ep in split_data["episodes"] if ep["states"].shape[0] >= minimum_rows]
    episodes = episodes[: args.num_episodes]
    if not episodes:
        raise SystemExit(f"no val episodes with >= {minimum_rows} rows")
    print(f"evaluating {len(episodes)} val episodes, horizon {args.horizon_s}s ({steps} steps)")

    states = torch.stack([torch.from_numpy(ep["states"][: block + steps]) for ep in episodes]).to(device)
    actions = torch.stack([torch.from_numpy(ep["actions"][: block + steps]) for ep in episodes]).to(device)
    gt_tip_all = torch.stack(
        [torch.from_numpy(np.array(ep["rollout"][: block + steps], copy=True)) for ep in episodes]
    ).to(device)
    frames = torch.from_numpy(
        np.stack([np.array(ep["frames"][: block + steps]) for ep in episodes])
    ).to(device)
    gt_tip = gt_tip_all[:, block:]
    gt_states = states[:, block:]

    # --- rollouts -----------------------------------------------------------
    pred_states, pred_z2 = autonomous_rollout(model, states, actions, frames, block, steps)
    anchor_states, anchor_z2 = autonomous_rollout(
        model, states, actions, frames, block, steps, anchored=True
    )
    persistence = states[:, block - 1 : block].expand(-1, steps, -1)

    # z2 ablation (plan section 10): replace every latent input with the dataset
    # mean (normalized zero) and roll z1. In Study 1 z1 is fully observed, so
    # little degradation is the EXPECTED result -- this quantifies it.
    blind_z2 = model.z2_mean.view(1, 1, -1).expand(len(episodes), states.shape[1], -1)
    blind_state_window = states[:, :block].clone()
    blind_window = blind_z2[:, :block].contiguous()
    blind_out = []
    for step in range(steps):
        next_state, _ = model.predict_next(
            blind_state_window, blind_window, actions[:, step : block + step]
        )
        blind_out.append(next_state)
        blind_state_window = torch.cat(
            [blind_state_window[:, 1:], next_state.unsqueeze(1)], dim=1
        )
    blind_states = torch.stack(blind_out, dim=1)

    curves = {
        "nrd_autonomous": tip_error_curve(pred_states, gt_tip, trig_idx, link_lengths),
        "nrd_anchored": tip_error_curve(anchor_states, gt_tip, trig_idx, link_lengths),
        "nrd_z2_meanblind": tip_error_curve(blind_states, gt_tip, trig_idx, link_lengths),
        "persistence": tip_error_curve(persistence, gt_tip, trig_idx, link_lengths),
    }
    state_model = None
    if args.state_checkpoint is not None:
        state_model = load_state_only(args.state_checkpoint.resolve(), device)
        state_pred = state_only_rollout(state_model, states, actions, block, steps)
        curves["state_only"] = tip_error_curve(state_pred, gt_tip, trig_idx, link_lengths)

    # --- latent / frame quality --------------------------------------------
    gt_z2 = model.encode_frame_sequence(frames[:, block:])
    # Cosine in normalized latent space: raw latents share a constant component
    # that pins raw cosine at ~1.0 regardless of quality.
    z2_cos = (
        F.cosine_similarity(model.normalize_z2(pred_z2), model.normalize_z2(gt_z2), dim=-1)
        .mean(0)
        .cpu()
        .numpy()
    )

    decoded = model.decode_latents(pred_z2)  # (E, steps, 3, H, W)
    gt_images = frames[:, block:].permute(0, 1, 4, 2, 3).float() / 255.0
    mse_per_step = (decoded - gt_images).pow(2).mean(dim=(0, 2, 3, 4)).cpu().numpy()
    psnr = 10.0 * np.log10(1.0 / np.maximum(mse_per_step, 1e-12))

    # --- cross-modal consistency (plan 11.3) --------------------------------
    pred_tip = pendulum_tip_positions(pred_states[..., trig_idx], link_lengths).cpu().numpy()
    decoded_uint8 = frames_to_uint8(decoded).cpu().numpy()  # (E, steps, H, W, 3)
    consistency_px = []
    missing = 0
    check_steps = range(0, steps, 5)
    for episode_index in range(len(episodes)):
        for step in check_steps:
            found = find_tip_pixel(decoded_uint8[episode_index, step])
            if found is None:
                missing += 1
                continue
            expected = project_to_pixel(*pred_tip[episode_index, step])
            consistency_px.append(math.hypot(found[0] - expected[0], found[1] - expected[1]))
    consistency_px = np.array(consistency_px)

    # --- summary ------------------------------------------------------------
    def at(curve: np.ndarray, seconds: float) -> float:
        return float(curve[min(int(round(seconds / dt_s)) - 1, len(curve) - 1)])

    summary = {
        "episodes": len(episodes),
        "horizon_s": args.horizon_s,
        "tip_error_m": {
            name: {f"{h}s": at(curve, h) for h in (0.5, 1.0, 2.0, 3.0) if h <= args.horizon_s}
            for name, curve in curves.items()
        },
        "z2_cos_sim": {f"{h}s": at(z2_cos, h) for h in (0.5, 1.0, 2.0, 3.0) if h <= args.horizon_s},
        "decoded_psnr_db": {f"{h}s": at(psnr, h) for h in (0.5, 1.0, 2.0, 3.0) if h <= args.horizon_s},
        "cross_modal_px": {
            "median": float(np.median(consistency_px)) if consistency_px.size else None,
            "p95": float(np.percentile(consistency_px, 95)) if consistency_px.size else None,
            "checked": int(consistency_px.size),
            "tip_not_found_in_decoded": int(missing),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # --- figure -------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_axis = (np.arange(1, steps + 1)) * dt_s
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), dpi=150)
    series_style = {
        "nrd_autonomous": dict(color=SERIES_NRD, label="NRD (autonomous)", lw=2),
        "nrd_anchored": dict(color=SERIES_ANCHOR, label="NRD (frame-anchored)", lw=2),
        "nrd_z2_meanblind": dict(color="#eda100", label="NRD (z2 mean-blinded)", lw=2, ls=":"),
        "state_only": dict(color=SERIES_STATE, label="State-only NeDM", lw=2),
        "persistence": dict(color=SERIES_REFERENCE, label="Persistence", lw=2, ls="--"),
    }
    for name in ("nrd_autonomous", "nrd_anchored", "nrd_z2_meanblind", "state_only", "persistence"):
        if name in curves:
            axes[0].plot(time_axis, curves[name] * 1000.0, **series_style[name])
    axes[0].set_xlabel("rollout horizon (s)")
    axes[0].set_ylabel("mean tip error (mm)")
    axes[0].set_title("Physical state (z1): open-loop tip error")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(time_axis, z2_cos, color=SERIES_NRD, lw=2)
    axes[1].set_xlabel("rollout horizon (s)")
    axes[1].set_ylabel("cosine similarity")
    axes[1].set_title("Camera latent (z2) vs encoder(true frame)")
    axes[1].set_ylim(0.0, 1.02)

    axes[2].plot(time_axis, psnr, color=SERIES_NRD, lw=2)
    axes[2].set_xlabel("rollout horizon (s)")
    axes[2].set_ylabel("PSNR (dB)")
    axes[2].set_title("Decoded predicted frames vs Chrono")
    for axis in axes:
        axis.grid(alpha=0.25, lw=0.5)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("NRD double pendulum: autonomous rollout fidelity", y=1.02)
    fig.tight_layout()
    figure_path = output_dir / "nrd_dpend_rollout_curves.png"
    fig.savefig(figure_path, bbox_inches="tight")
    print(f"figure -> {figure_path}")

    # --- demo videos --------------------------------------------------------
    gt_frames_np = frames[:, block:].cpu().numpy()
    for episode_index in range(min(args.video_episodes, len(episodes))):
        episode_id = episodes[episode_index]["episode_id"]
        video_path = output_dir / f"rollout_{episode_id}.gif"
        make_video(gt_frames_np[episode_index], decoded_uint8[episode_index], video_path)
        print(f"video -> {video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
