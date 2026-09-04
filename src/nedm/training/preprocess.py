from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nedm.quadruped.dataset import (
    CONTACT_ENGAGE_N,
    CONTACT_RELEASE_N,
    LEG_ORDER,
    contact_mode,
)
from nedm.training.constants import (
    DEFAULT_ACTION_FIELDS,
    DEFAULT_ROLLOUT_FIELDS,
    DEFAULT_STATE_FIELDS,
    STATE_FIELD_PRESETS,
)


@dataclass(frozen=True)
class SplitBuffers:
    states: np.ndarray
    actions: np.ndarray
    targets: np.ndarray
    episode_starts: np.ndarray
    episode_lengths: np.ndarray
    rollout: np.ndarray
    # Packed 4-bit contact mode per transition, int8 in [0, 15], aligned with
    # ``states`` (the mode of the FROM state). None unless --contact-mode.
    contact_modes: np.ndarray | None = None
    arrays_saved: bool = False
    # Camera frames (NRD datasets): uint8 (total_transitions + episodes, H, W, 3)
    # in the same one-extra-row-per-episode layout as ``rollout``, so
    # ``rollout_episode_offsets`` indexes both. None for state-only datasets.
    frames: np.ndarray | None = None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def compute_dt_s(dataset_root: Path) -> float:
    resolved_cfg_path = dataset_root / "collector_config.resolved.json"
    if resolved_cfg_path.exists():
        resolved_cfg = load_json(resolved_cfg_path)
        return float(resolved_cfg["simulation"]["record_step_s"])
    # Datasets without a resolved collector config (e.g. the arm collector) record
    # their control period in the dataset index's config block instead.
    dataset_index_path = dataset_root / "dataset_index.json"
    if dataset_index_path.exists():
        index_cfg = load_json(dataset_index_path).get("config", {})
        if "control_dt_s" in index_cfg:
            return float(index_cfg["control_dt_s"])
    return 0.01


def compute_common_dt_s(dataset_roots: list[Path]) -> float:
    dts = [compute_dt_s(dataset_root) for dataset_root in dataset_roots]
    reference = dts[0]
    for dataset_root, dt_s in zip(dataset_roots, dts, strict=True):
        if abs(dt_s - reference) > 1e-12:
            raise ValueError(f"Dataset {dataset_root} has dt={dt_s}, expected {reference}")
    return reference


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact HMMWV training dataset cache.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        nargs="+",
        default=[Path("artifacts/datasets/hmmwv_overfit_6k")],
        help="Root(s) of raw episode datasets to merge into one processed cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/training_datasets/hmmwv_overfit_6k_seq_v1"),
        help="Directory where the processed arrays will be written.",
    )
    parser.add_argument(
        "--max-episodes-per-split",
        type=int,
        default=None,
        help="Optional cap per split for quick smoke tests.",
    )
    parser.add_argument(
        "--contact-mode",
        action="store_true",
        help="Precompute the packed 4-bit foot contact mode per transition "
             "(quadruped datasets only; needs the foot_*_force_fz_n columns).",
    )
    parser.add_argument(
        "--contact-release-n", type=float, default=CONTACT_RELEASE_N,
        help="Schmitt-trigger release threshold, newtons.",
    )
    parser.add_argument(
        "--contact-engage-n", type=float, default=CONTACT_ENGAGE_N,
        help="Schmitt-trigger engage threshold, newtons.",
    )
    parser.add_argument(
        "--state-field-preset",
        type=str,
        choices=sorted(STATE_FIELD_PRESETS),
        default="default",
        help="Named state-field preset. Use tire_force_omega for all tire force axes, "
        "or tire_normal_force_omega for tire Fz plus spindle omega.",
    )
    parser.add_argument(
        "--state-fields",
        type=str,
        nargs="+",
        default=None,
        help="Explicit state fields. Overrides --state-field-preset when provided.",
    )
    parser.add_argument(
        "--action-fields",
        type=str,
        nargs="+",
        default=None,
        help="Explicit action fields. Defaults to the standard driver controls.",
    )
    parser.add_argument(
        "--rollout-fields",
        type=str,
        nargs="+",
        default=None,
        help="Explicit rollout fields. Defaults to x/y/yaw pose.",
    )
    parser.add_argument(
        "--disk-backed-arrays",
        action="store_true",
        help="Write .npy arrays through memory maps instead of keeping the full cache in RAM.",
    )
    parser.add_argument(
        "--frames",
        action="store_true",
        help="Also pack per-episode camera frames (episodes must carry frames_path) "
        "into {split}_frames.npy, laid out like the rollout array.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Rebuild metadata.json from already-written arrays and split metadata.",
    )
    parser.add_argument(
        "--stats-chunk-rows",
        type=int,
        default=1_000_000,
        help="Rows per chunk when computing normalization statistics.",
    )
    return parser.parse_args(argv)


# Foot force columns in the order the 4-bit contact mode packs them: bits 3..0.
# DERIVED FROM LEG_ORDER, the constant that already defines this ordering and
# that names the CSV columns in the first place -- not a literal beside it. Two
# orderings for the same four legs already coexist here: LEG_ORDER is fl fr rl
# rr, and quadruped.constants.FOOT_BODIES is FR FL RR RL. Packing against the
# wrong one transposes the left/right bits, and NO HISTOGRAM CAN SEE THAT,
# because the marginal distribution is unchanged under a relabelling -- the trot
# signature survives it too, since 0110 and 1001 swap into each other under a
# left/right flip. A hardcoded copy would make it three orderings and one more
# thing to reach for by mistake.
CONTACT_FORCE_FIELDS = [f"foot_{leg}_force_fz_n" for leg in LEG_ORDER]


def read_episode_csv(
    csv_path: Path,
    state_fields: list[str],
    action_fields: list[str],
    rollout_fields: list[str],
    contact_fields: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    state_rows: list[list[float]] = []
    action_rows: list[list[float]] = []
    rollout_rows: list[list[float]] = []
    contact_rows: list[list[float]] = []

    with csv_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        missing = [
            field
            for field in state_fields + action_fields + rollout_fields + list(contact_fields or [])
            if field not in reader.fieldnames
        ]
        if missing:
            raise KeyError(f"{csv_path} is missing required fields: {missing}")
        for row in reader:
            state_rows.append([float(row[field]) for field in state_fields])
            action_rows.append([float(row[field]) for field in action_fields])
            rollout_rows.append([float(row[field]) for field in rollout_fields])
            if contact_fields:
                contact_rows.append([float(row[field]) for field in contact_fields])

    states = np.asarray(state_rows, dtype=np.float32)
    actions = np.asarray(action_rows, dtype=np.float32)
    rollout = np.asarray(rollout_rows, dtype=np.float32)
    contact = np.asarray(contact_rows, dtype=np.float64) if contact_fields else None
    if len(states) < 2:
        raise ValueError(f"{csv_path} has fewer than 2 rows after warmup trimming")
    return states, actions, rollout, contact


def build_split_buffers(
    episodes: list[dict[str, Any]],
    state_fields: list[str],
    action_fields: list[str],
    rollout_fields: list[str],
    output_dir: Path | None = None,
    split: str | None = None,
    disk_backed_arrays: bool = False,
    include_frames: bool = False,
    contact_mode_bounds: tuple[float, float] | None = None,
) -> SplitBuffers:
    total_transitions = sum(int(ep["rows"]) - 1 for ep in episodes)
    state_dim = len(state_fields)
    action_dim = len(action_fields)
    rollout_dim = len(rollout_fields)

    def allocate(name: str, shape: tuple[int, ...], dtype: type[np.generic]) -> np.ndarray:
        if not disk_backed_arrays:
            return np.empty(shape, dtype=dtype)
        if output_dir is None or split is None:
            raise ValueError("output_dir and split are required when disk_backed_arrays=True")
        return np.lib.format.open_memmap(output_dir / f"{split}_{name}.npy", mode="w+", dtype=dtype, shape=shape)

    states = allocate("states", (total_transitions, state_dim), np.float32)
    actions = allocate("actions", (total_transitions, action_dim), np.float32)
    targets = allocate("targets", (total_transitions, state_dim), np.float32)
    rollout = allocate("rollout", (total_transitions + len(episodes), rollout_dim), np.float32)
    episode_starts = np.empty((len(episodes),), dtype=np.int64)
    episode_lengths = np.empty((len(episodes),), dtype=np.int32)
    contact_modes = (allocate("contact_modes", (total_transitions,), np.int8)
                     if contact_mode_bounds is not None else None)

    frames: np.ndarray | None = None
    if include_frames and episodes:
        if output_dir is None or split is None:
            raise ValueError("output_dir and split are required when include_frames=True")
        first_frames = np.load(Path(episodes[0]["_dataset_root"]) / episodes[0]["frames_path"], mmap_mode="r")
        frame_shape = first_frames.shape[1:]
        # Frames are always disk-backed: a pilot split alone is multiple GB.
        frames = np.lib.format.open_memmap(
            output_dir / f"{split}_frames.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(total_transitions + len(episodes), *frame_shape),
        )

    cursor = 0
    rollout_cursor = 0
    for episode_index, episode in enumerate(episodes):
        csv_path = Path(episode["_dataset_root"]) / episode["csv_path"]
        episode_states, episode_actions, episode_rollout, episode_forces = read_episode_csv(
            csv_path,
            state_fields=state_fields,
            action_fields=action_fields,
            rollout_fields=rollout_fields,
            contact_fields=CONTACT_FORCE_FIELDS if contact_mode_bounds is not None else None,
        )
        length = episode_states.shape[0] - 1
        episode_starts[episode_index] = cursor
        episode_lengths[episode_index] = length

        states[cursor : cursor + length] = episode_states[:-1]
        actions[cursor : cursor + length] = episode_actions[:-1]
        targets[cursor : cursor + length] = episode_states[1:] - episode_states[:-1]
        if contact_modes is not None:
            # PER EPISODE, BEFORE SLICING, and both parts matter. The Schmitt
            # trigger is a temporal filter carrying state across samples, so
            # running it on the concatenated buffer would let one episode's final
            # stance set the next episode's opening mode across a discontinuity
            # that is not a footfall. Sliced [:-1] to align with ``states``: the
            # mode of the FROM state of each transition, not the TO state, so a
            # model conditioning on it sees the contact configuration it is
            # predicting forward from.
            release_n, engage_n = contact_mode_bounds
            _, episode_mode = contact_mode(episode_forces, release_n=release_n, engage_n=engage_n)
            contact_modes[cursor : cursor + length] = episode_mode[:-1].astype(np.int8)
        rollout[rollout_cursor : rollout_cursor + length + 1] = episode_rollout
        if frames is not None:
            frames_path = episode.get("frames_path")
            if not frames_path:
                raise ValueError(f"--frames requested but episode {episode['episode_id']} has no frames_path")
            episode_frames = np.load(Path(episode["_dataset_root"]) / frames_path, mmap_mode="r")
            if episode_frames.shape[0] != length + 1:
                raise ValueError(
                    f"{frames_path}: {episode_frames.shape[0]} frames but {length + 1} CSV rows"
                )
            frames[rollout_cursor : rollout_cursor + length + 1] = episode_frames

        cursor += length
        rollout_cursor += length + 1

    return SplitBuffers(
        states=states,
        actions=actions,
        targets=targets,
        episode_starts=episode_starts,
        episode_lengths=episode_lengths,
        rollout=rollout,
        contact_modes=contact_modes,
        arrays_saved=disk_backed_arrays,
        frames=frames,
    )


def save_split(output_dir: Path, split: str, buffers: SplitBuffers, episodes: list[dict[str, Any]]) -> None:
    if buffers.arrays_saved:
        for array in (buffers.states, buffers.actions, buffers.targets, buffers.rollout):
            flush = getattr(array, "flush", None)
            if flush is not None:
                flush()
    else:
        np.save(output_dir / f"{split}_states.npy", buffers.states)
        np.save(output_dir / f"{split}_actions.npy", buffers.actions)
        np.save(output_dir / f"{split}_targets.npy", buffers.targets)
        np.save(output_dir / f"{split}_rollout.npy", buffers.rollout)
    if buffers.frames is not None:
        buffers.frames.flush()  # frames are always written through a memmap
    if buffers.contact_modes is not None and not buffers.arrays_saved:
        np.save(output_dir / f"{split}_contact_modes.npy", buffers.contact_modes)
    np.save(output_dir / f"{split}_episode_starts.npy", buffers.episode_starts)
    np.save(output_dir / f"{split}_episode_lengths.npy", buffers.episode_lengths)

    split_metadata = {
        "split": split,
        "episode_count": len(episodes),
        "transition_count": int(buffers.targets.shape[0]),
        "episode_ids": [episode["episode_id"] for episode in episodes],
        "scenario_families": [episode["scenario_family"] for episode in episodes],
        "source_datasets": [episode["_dataset_name"] for episode in episodes],
        "source_csv_paths": [
            str(Path(episode["_dataset_root"]) / episode["csv_path"])
            for episode in episodes
        ],
        # Occupancy of the 16 modes, so a degenerate extraction (one mode at
        # 90%, or flight/full-stance never observed) is visible in the cache
        # rather than only after a model fails to learn from it.
        **({"contact_mode_histogram": {
            str(m): int(c) for m, c in
            zip(*np.unique(buffers.contact_modes, return_counts=True))}}
           if buffers.contact_modes is not None else {}),
        "rollout_episode_offsets": np.cumsum(
            np.concatenate(([0], buffers.episode_lengths.astype(np.int64) + 1))
        ).tolist(),
    }
    (output_dir / f"{split}_episodes.json").write_text(json.dumps(split_metadata, indent=2))


def summarize_stats(array: np.ndarray, chunk_rows: int = 1_000_000) -> tuple[list[float], list[float]]:
    if array.ndim != 2:
        raise ValueError(f"expected 2D array for stats, got shape {array.shape}")
    chunk_rows = max(1, int(chunk_rows))
    total = 0
    sums = np.zeros((array.shape[1],), dtype=np.float64)
    sum_squares = np.zeros((array.shape[1],), dtype=np.float64)
    for start in range(0, array.shape[0], chunk_rows):
        chunk = np.asarray(array[start : start + chunk_rows], dtype=np.float64)
        total += int(chunk.shape[0])
        sums += chunk.sum(axis=0, dtype=np.float64)
        sum_squares += np.square(chunk).sum(axis=0, dtype=np.float64)
    mean64 = sums / max(total, 1)
    variance64 = np.maximum(sum_squares / max(total, 1) - mean64 * mean64, 0.0)
    mean = mean64.astype(np.float32)
    std = np.maximum(np.sqrt(variance64).astype(np.float32), 1e-6)
    return mean.tolist(), std.tolist()


def build_metadata(
    dataset_indices: list[dict[str, Any]],
    dataset_roots: list[Path],
    output_dir: Path,
    dt_s: float,
    state_fields: list[str],
    action_fields: list[str],
    rollout_fields: list[str],
    state_field_preset: str,
    stats_chunk_rows: int,
    include_frames: bool = False,
    contact_mode_bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    train_states = np.load(output_dir / "train_states.npy", mmap_mode="r")
    train_actions = np.load(output_dir / "train_actions.npy", mmap_mode="r")
    train_targets = np.load(output_dir / "train_targets.npy", mmap_mode="r")
    train_split = load_json(output_dir / "train_episodes.json")
    val_split = load_json(output_dir / "val_episodes.json")

    state_mean, state_std = summarize_stats(train_states, chunk_rows=stats_chunk_rows)
    action_mean, action_std = summarize_stats(train_actions, chunk_rows=stats_chunk_rows)
    target_mean, target_std = summarize_stats(train_targets, chunk_rows=stats_chunk_rows)

    frames_info: dict[str, Any] | None = None
    if include_frames:
        train_frames = np.load(output_dir / "train_frames.npy", mmap_mode="r")
        frames_info = {"shape": list(train_frames.shape[1:]), "dtype": "uint8"}

    # THE THRESHOLDS TRAVEL WITH THE CACHE. contact_mode's docstring keeps the
    # mode out of the CSV so two tuned constants are not frozen into the dataset
    # where no consumer can revisit them. Writing the mode into a derived cache
    # without the bounds would reintroduce exactly that, one layer down: a
    # contact channel that cannot be reproduced or compared against another
    # choice of bounds. 5/60 N is measured, not chosen -- see that docstring.
    contact_info = None
    if contact_mode_bounds is not None:
        release_n, engage_n = contact_mode_bounds
        contact_info = {
            "release_n": release_n,
            "engage_n": engage_n,
            "force_fields": list(CONTACT_FORCE_FIELDS),
            "bit_order": "fl fr rl rr -> bits 3..0",
            "num_modes": 16,
            "dtype": "int8",
            "alignment": "mode of the FROM state of each transition",
        }

    return {
        "contact_mode": contact_info,
        "frames": frames_info,
        "dataset_name": "+".join(dataset_index["dataset_name"] for dataset_index in dataset_indices),
        "raw_dataset_root": str(dataset_roots[0]),
        "raw_dataset_roots": [str(dataset_root) for dataset_root in dataset_roots],
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dt_s": dt_s,
        "state_field_preset": state_field_preset,
        "state_fields": state_fields,
        "action_fields": action_fields,
        "target_fields": [f"delta_{field}" for field in state_fields],
        "rollout_fields": rollout_fields,
        "splits": {
            "train": {
                "episode_count": int(train_split["episode_count"]),
                "transition_count": int(train_split["transition_count"]),
            },
            "val": {
                "episode_count": int(val_split["episode_count"]),
                "transition_count": int(val_split["transition_count"]),
            },
        },
        "normalization": {
            "state_mean": state_mean,
            "state_std": state_std,
            "action_mean": action_mean,
            "action_std": action_std,
            "target_mean": target_mean,
            "target_std": target_std,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_roots = [dataset_root.resolve() for dataset_root in args.dataset_root]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_indices: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
    for dataset_root in dataset_roots:
        dataset_index_path = dataset_root / "dataset_index.json"
        if not dataset_index_path.exists():
            raise FileNotFoundError(f"Dataset index not found: {dataset_index_path}")
        dataset_index = load_json(dataset_index_path)
        dataset_indices.append(dataset_index)
        for episode in dataset_index["episodes"]:
            episode_record = dict(episode)
            # Arm-collector episodes carry collision_kind but not scenario_family
            # (which save_split requires); group by termination kind so a future
            # rollout eval can balance across ground/track/joint_limit/full-length.
            episode_record.setdefault(
                "scenario_family", episode.get("collision_kind") or "full_length"
            )
            episode_record["_dataset_root"] = str(dataset_root)
            episode_record["_dataset_name"] = str(dataset_index["dataset_name"])
            all_episodes.append(episode_record)

    episode_id_counts = Counter(episode["episode_id"] for episode in all_episodes)
    duplicate_ids = sorted(episode_id for episode_id, count in episode_id_counts.items() if count > 1)
    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:10])
        raise ValueError(f"Duplicate episode IDs across dataset roots: {preview}")

    split_episodes: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for episode in all_episodes:
        split_episodes[episode["split"]].append(episode)

    if args.max_episodes_per_split is not None:
        for split in split_episodes:
            split_episodes[split] = split_episodes[split][: args.max_episodes_per_split]

    contact_bounds = ((float(args.contact_release_n), float(args.contact_engage_n))
                      if args.contact_mode else None)
    if contact_bounds is not None and contact_bounds[0] >= contact_bounds[1]:
        raise ValueError(
            f"--contact-release-n ({contact_bounds[0]}) must be below "
            f"--contact-engage-n ({contact_bounds[1]}). Equal bounds collapse the "
            "Schmitt trigger to a plain threshold, which invents 66% spurious "
            "transitions on CRM -- see quadruped.dataset.contact_mode."
        )

    state_fields = (
        list(args.state_fields)
        if args.state_fields is not None
        else list(STATE_FIELD_PRESETS[args.state_field_preset])
    )
    action_fields = list(args.action_fields) if args.action_fields is not None else list(DEFAULT_ACTION_FIELDS)
    rollout_fields = list(args.rollout_fields) if args.rollout_fields is not None else list(DEFAULT_ROLLOUT_FIELDS)
    dt_s = compute_common_dt_s(dataset_roots)

    if args.metadata_only:
        metadata = build_metadata(
            dataset_indices=dataset_indices,
            dataset_roots=dataset_roots,
            output_dir=output_dir,
            dt_s=dt_s,
            state_fields=state_fields,
            action_fields=action_fields,
            rollout_fields=rollout_fields,
            state_field_preset=args.state_field_preset,
            stats_chunk_rows=args.stats_chunk_rows,
            include_frames=bool(args.frames),
            contact_mode_bounds=contact_bounds,
        )
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(
            f"wrote metadata to {output_dir} "
            f"with {metadata['splits']['train']['transition_count']} train transitions and "
            f"{metadata['splits']['val']['transition_count']} val transitions"
        )
        return 0

    train_buffers = build_split_buffers(
        episodes=split_episodes["train"],
        state_fields=state_fields,
        action_fields=action_fields,
        rollout_fields=rollout_fields,
        output_dir=output_dir,
        split="train",
        disk_backed_arrays=bool(args.disk_backed_arrays),
        include_frames=bool(args.frames),
        contact_mode_bounds=contact_bounds,
    )
    val_buffers = build_split_buffers(
        episodes=split_episodes["val"],
        state_fields=state_fields,
        action_fields=action_fields,
        rollout_fields=rollout_fields,
        output_dir=output_dir,
        split="val",
        disk_backed_arrays=bool(args.disk_backed_arrays),
        include_frames=bool(args.frames),
        contact_mode_bounds=contact_bounds,
    )

    save_split(output_dir, "train", train_buffers, split_episodes["train"])
    save_split(output_dir, "val", val_buffers, split_episodes["val"])

    metadata = build_metadata(
        dataset_indices=dataset_indices,
        dataset_roots=dataset_roots,
        output_dir=output_dir,
        dt_s=dt_s,
        state_fields=state_fields,
        action_fields=action_fields,
        rollout_fields=rollout_fields,
        state_field_preset=args.state_field_preset,
        stats_chunk_rows=args.stats_chunk_rows,
        include_frames=bool(args.frames),
        contact_mode_bounds=contact_bounds,
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(
        f"wrote processed dataset to {output_dir} "
        f"with {metadata['splits']['train']['transition_count']} train transitions and "
        f"{metadata['splits']['val']['transition_count']} val transitions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
