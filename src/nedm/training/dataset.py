from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def load_metadata(processed_root: Path) -> dict[str, Any]:
    return json.loads((processed_root / "metadata.json").read_text())


def load_split_metadata(processed_root: Path, split: str) -> dict[str, Any]:
    return json.loads((processed_root / f"{split}_episodes.json").read_text())


class WindowedHMMWVDataset(Dataset):
    def __init__(
        self,
        processed_root: Path,
        split: str,
        sequence_length: int,
        max_windows: int | None = None,
        episode_fraction: float | None = None,
        seed: int = 0,
        load_into_memory: bool = False,
        load_frames: bool = False,
    ) -> None:
        self.processed_root = processed_root
        self.split = split
        self.sequence_length = int(sequence_length)
        mmap_mode = None if load_into_memory else "r"
        self.states = np.load(processed_root / f"{split}_states.npy", mmap_mode=mmap_mode)
        self.actions = np.load(processed_root / f"{split}_actions.npy", mmap_mode=mmap_mode)
        self.targets = np.load(processed_root / f"{split}_targets.npy", mmap_mode=mmap_mode)
        self.episode_starts = np.load(processed_root / f"{split}_episode_starts.npy")
        self.episode_lengths = np.load(processed_root / f"{split}_episode_lengths.npy")
        self.split_metadata = load_split_metadata(processed_root, split)
        # NRD datasets: camera frames in the rollout layout (one extra row per
        # episode), indexed by rollout_episode_offsets. Frames stay memory-mapped
        # regardless of load_into_memory -- they are typically several GB.
        self.frames: np.ndarray | None = None
        self.frame_offsets: np.ndarray | None = None
        if load_frames:
            self.frames = np.load(processed_root / f"{split}_frames.npy", mmap_mode="r")
            self.frame_offsets = np.asarray(
                self.split_metadata["rollout_episode_offsets"][:-1], dtype=np.int64
            )
        # Optional data-quantity ablation: restrict training to a nested seeded
        # subset of whole episodes. Same seed -> the same permutation, so a
        # smaller fraction is a prefix of a larger one (20% ⊂ 40% ⊂ ...). Only
        # the small index arrays are subset; the states/targets/actions memmaps
        # are untouched (dropped-episode rows are simply never sampled).
        if episode_fraction is not None and 0.0 < episode_fraction < 1.0:
            n_ep = int(self.episode_lengths.shape[0])
            keep = round(episode_fraction * n_ep)
            perm = np.random.default_rng(seed).permutation(n_ep)
            sel = np.sort(perm[:keep])
            self.episode_starts = self.episode_starts[sel]
            self.episode_lengths = self.episode_lengths[sel]
            if self.frame_offsets is not None:
                self.frame_offsets = self.frame_offsets[sel]
        self.valid_counts = np.maximum(self.episode_lengths.astype(np.int64) - self.sequence_length + 1, 0)
        self.cumulative_windows = np.cumsum(self.valid_counts, dtype=np.int64)
        self.total_windows = int(self.cumulative_windows[-1]) if self.cumulative_windows.size else 0
        self.window_indices: np.ndarray | None = None

        if max_windows is not None and max_windows < self.total_windows:
            rng = np.random.default_rng(seed)
            self.window_indices = np.sort(rng.choice(self.total_windows, size=max_windows, replace=False))

    def __len__(self) -> int:
        if self.window_indices is not None:
            return int(self.window_indices.shape[0])
        return self.total_windows

    def _locate(self, index: int) -> tuple[int, int]:
        """(episode position in the current index arrays, local window start)."""
        if self.window_indices is not None:
            index = int(self.window_indices[index])
        if index < 0 or index >= self.total_windows:
            raise IndexError(index)
        episode_index = int(np.searchsorted(self.cumulative_windows, index, side="right"))
        previous_total = 0 if episode_index == 0 else int(self.cumulative_windows[episode_index - 1])
        return episode_index, index - previous_total

    def _window_start(self, index: int) -> int:
        episode_index, local_start = self._locate(index)
        return int(self.episode_starts[episode_index]) + local_start

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, local_start = self._locate(index)
        start = int(self.episode_starts[episode_index]) + local_start
        stop = start + self.sequence_length
        # torch.tensor copies into torch-owned storage. The previous
        # from_numpy path attaches a custom deleter per tensor whose context
        # was implicated in rare heap-corruption aborts (free(): invalid
        # pointer in c10::deleteInefficientStdFunctionContext).
        item = {
            "states": torch.tensor(self.states[start:stop], dtype=torch.float32),
            "actions": torch.tensor(self.actions[start:stop], dtype=torch.float32),
            "targets": torch.tensor(self.targets[start:stop], dtype=torch.float32),
        }
        if self.frames is not None:
            # sequence_length + 1 frames: window rows t..t+L-1 plus the frame of
            # the last transition's NEXT state, so every transition in the window
            # has its (x_t, x_{t+1}) pair.
            frame_start = int(self.frame_offsets[episode_index]) + local_start
            item["frames"] = torch.tensor(
                self.frames[frame_start : frame_start + self.sequence_length + 1],
                dtype=torch.uint8,
            )
        return item


def load_rollout_split(processed_root: Path, split: str) -> dict[str, Any]:
    split_metadata = load_split_metadata(processed_root, split)
    states = np.load(processed_root / f"{split}_states.npy", mmap_mode="r")
    actions = np.load(processed_root / f"{split}_actions.npy", mmap_mode="r")
    targets = np.load(processed_root / f"{split}_targets.npy", mmap_mode="r")
    rollout = np.load(processed_root / f"{split}_rollout.npy", mmap_mode="r")
    episode_starts = np.load(processed_root / f"{split}_episode_starts.npy")
    episode_lengths = np.load(processed_root / f"{split}_episode_lengths.npy")

    episodes: list[dict[str, Any]] = []
    rollout_offsets = split_metadata["rollout_episode_offsets"]
    for episode_index, episode_id in enumerate(split_metadata["episode_ids"]):
        start = int(episode_starts[episode_index])
        length = int(episode_lengths[episode_index])
        rollout_start = int(rollout_offsets[episode_index])
        rollout_stop = int(rollout_offsets[episode_index + 1])
        episode_states = np.empty((length + 1, states.shape[1]), dtype=np.float32)
        episode_actions = np.empty((length + 1, actions.shape[1]), dtype=np.float32)
        episode_states[:-1] = states[start : start + length]
        episode_states[-1] = states[start + length - 1] + targets[start + length - 1]
        episode_actions[:-1] = actions[start : start + length]
        episode_actions[-1] = actions[start + length - 1]
        episodes.append(
            {
                "episode_id": episode_id,
                "scenario_family": split_metadata["scenario_families"][episode_index],
                "states": episode_states,
                "actions": episode_actions,
                "rollout": np.array(rollout[rollout_start:rollout_stop], copy=True).astype(np.float32),
            }
        )
    return {"episodes": episodes}
