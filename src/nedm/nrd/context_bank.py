"""Recorded [z1, z2, a] context windows for resetting NRD-in-the-loop RL environments.

The NRD transformer needs a full ``block_size`` history of aligned states, camera
latents, and actions. Repeating one state/latent is outside the training
distribution (plan section 4), so RL resets draw from a bank of windows cut out of
recorded episodes: the states/actions come straight from the processed cache and
the latents are the frozen encoder's outputs on the recorded frames. Encoding
is done once here and cached, because the frames memmap is ~22 GB.

The bank also carries the per-dimension extremes of the NORMALIZED latents over
its windows; the environment uses them as its out-of-distribution guard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from nedm.nrd.model import NRDDynamicsModel
from nedm.nrd.trainer import load_rollout_split_with_frames

BANK_FORMAT_VERSION = 1


@torch.no_grad()
def build_context_bank(
    model: NRDDynamicsModel,
    processed_root: Path | str,
    split: str,
    num_contexts: int,
    seed: int,
    device: torch.device | str,
    *,
    nrd_checkpoint: str = "",
    encode_batch: int = 256,
    log: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Sample ``num_contexts`` windows uniformly over all valid (episode, start) pairs."""
    device = torch.device(device)
    block = int(model.backbone.config.block_size)
    split_data = load_rollout_split_with_frames(Path(processed_root), split)
    episodes = split_data["episodes"]
    windows_per_episode = np.array([ep["states"].shape[0] - block + 1 for ep in episodes], dtype=np.int64)
    windows_per_episode = np.clip(windows_per_episode, 0, None)
    total_windows = int(windows_per_episode.sum())
    if total_windows == 0:
        raise ValueError(f"no {split} episode has >= block_size={block} rows")

    rng = np.random.default_rng(seed)
    episode_index = rng.choice(len(episodes), size=num_contexts, p=windows_per_episode / total_windows)
    start_index = np.floor(rng.random(num_contexts) * windows_per_episode[episode_index]).astype(np.int64)
    # Sequential-ish memmap reads: sort by (episode, start).
    order = np.lexsort((start_index, episode_index))
    episode_index, start_index = episode_index[order], start_index[order]

    state_dim = int(episodes[0]["states"].shape[1])
    action_dim = int(episodes[0]["actions"].shape[1])
    states = np.empty((num_contexts, block, state_dim), dtype=np.float32)
    actions = np.empty((num_contexts, block, action_dim), dtype=np.float32)
    latents = np.empty((num_contexts, block, model.z2_dim), dtype=np.float32)
    for chunk_start in range(0, num_contexts, encode_batch):
        chunk = slice(chunk_start, min(chunk_start + encode_batch, num_contexts))
        pairs = list(zip(episode_index[chunk], start_index[chunk]))
        frames = np.stack([np.asarray(episodes[e]["frames"][s : s + block]) for e, s in pairs])
        latents[chunk] = model.encode_frame_sequence(torch.from_numpy(frames).to(device)).cpu().numpy()
        states[chunk] = np.stack([episodes[e]["states"][s : s + block] for e, s in pairs])
        actions[chunk] = np.stack([episodes[e]["actions"][s : s + block] for e, s in pairs])
        if log is not None and ((chunk_start // encode_batch) % 8 == 0 or chunk.stop == num_contexts):
            log(f"  encoded {chunk.stop}/{num_contexts} windows")

    z2_norm = model.normalize_z2(torch.from_numpy(latents).to(device)).reshape(-1, model.z2_dim).abs().cpu().numpy()
    bank = {
        "states": states,
        "actions": actions,
        "z2": latents,
        "episode_index": episode_index.astype(np.int64),
        "start_index": start_index.astype(np.int64),
        "z2_norm_absmax": z2_norm.max(axis=0).astype(np.float32),
        "z2_norm_p999": np.percentile(z2_norm, 99.9, axis=0).astype(np.float32),
        "z2_mean": model.z2_mean.detach().cpu().numpy().astype(np.float32),
        "z2_std": model.z2_std.detach().cpu().numpy().astype(np.float32),
        "meta": {
            "format_version": BANK_FORMAT_VERSION,
            "block_size": block,
            "split": split,
            "processed_root": str(Path(processed_root).resolve()),
            "nrd_checkpoint": str(nrd_checkpoint),
            "seed": int(seed),
            "num_contexts": int(num_contexts),
            "total_windows": total_windows,
            "state_fields": list(model.state_fields or []),
            "episode_ids": [episodes[int(e)]["episode_id"] for e in episode_index],
        },
    }
    return bank


def save_context_bank(bank: dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value for key, value in bank.items() if key != "meta"}
    np.savez(path, meta_json=np.array(json.dumps(bank["meta"])), **arrays)
    return path


def load_context_bank(path: Path | str) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as payload:
        bank = {key: payload[key] for key in payload.files if key != "meta_json"}
        bank["meta"] = json.loads(str(payload["meta_json"]))
    return bank


__all__ = ["BANK_FORMAT_VERSION", "build_context_bank", "load_context_bank", "save_context_bank"]
