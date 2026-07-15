#!/usr/bin/env python
"""Derive a state-channel-subset processed cache from an existing one.

The feature ablations need caches whose ``state_fields`` are a subset of an
existing cache's. Re-running ``preprocess.py`` against the raw datasets would
re-read ~300 GB of raw episodes; instead we column-slice the processed memmaps,
which is exactly how ``*_normal_force_omega_seq_v1`` was itself derived from
``*_force_omega_seq_v1`` (note its symlinked actions/rollout/episode arrays).

Only ``states``/``targets`` depend on the state-field list. ``actions``,
``rollout``, ``episode_starts``, ``episode_lengths`` and ``episodes.json`` are
byte-identical, so they are symlinked to the source's real files rather than
copied. Per-channel ``normalization`` entries are exact row subsets, so the
derived cache is bit-identical to what preprocess.py would emit for the same
field list (verified by --verify).

Example:
    python scripts/ablation_ofat/derive_state_subset_dataset.py \
        --source-dir artifacts/training_datasets/hmmwv_crm_2000_normal_force_omega_seq_v1 \
        --output-dir artifacts/training_datasets/hmmwv_crm_2000_body7_seq_v1 \
        --state-field-preset default
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nedm.training.constants import STATE_FIELD_PRESETS  # noqa: E402

# Arrays that do not depend on the state-field list: symlink, never copy.
SHARED_ARRAYS = ("actions", "rollout", "episode_starts", "episode_lengths")
SHARED_JSON = ("episodes",)
SPLITS = ("train", "val")
CHUNK_ROWS = 4_000_000  # ~240 MB/chunk at 15 float32 columns


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, required=True, help="Existing processed dataset dir")
    parser.add_argument("--output-dir", type=Path, required=True, help="Derived dataset dir to create")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--state-field-preset",
        choices=sorted(STATE_FIELD_PRESETS),
        help="Keep exactly this preset's fields (must be a subset of the source's).",
    )
    group.add_argument("--keep-fields", nargs="+", help="Explicit state fields to keep.")
    group.add_argument("--drop-fields", nargs="+", help="State fields to drop.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output dir")
    parser.add_argument("--verify", action="store_true", help="Re-read the written arrays and compare to the source")
    return parser.parse_args(argv)


def resolve_keep_fields(source_fields: list[str], args: argparse.Namespace) -> list[str]:
    if args.state_field_preset:
        keep = list(STATE_FIELD_PRESETS[args.state_field_preset])
    elif args.keep_fields:
        keep = list(args.keep_fields)
    else:
        drop = set(args.drop_fields)
        unknown = drop - set(source_fields)
        if unknown:
            raise ValueError(f"--drop-fields not present in source: {sorted(unknown)}")
        keep = [field for field in source_fields if field not in drop]

    missing = [field for field in keep if field not in source_fields]
    if missing:
        raise ValueError(f"requested fields are not in the source cache: {missing}")
    if not keep:
        raise ValueError("refusing to derive a cache with zero state fields")
    return keep


def link_shared(source_dir: Path, output_dir: Path, name: str) -> None:
    """Symlink a state-independent artifact, resolving through source symlinks."""
    source = source_dir / name
    if not source.exists():
        raise FileNotFoundError(f"source cache is missing {source}")
    target = Path(os.path.realpath(source))
    (output_dir / name).symlink_to(target)


def slice_array(source_path: Path, output_path: Path, keep_index: np.ndarray) -> None:
    source = np.load(source_path, mmap_mode="r")
    output = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=source.dtype, shape=(source.shape[0], keep_index.size)
    )
    for start in range(0, source.shape[0], CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, source.shape[0])
        output[start:stop] = source[start:stop][:, keep_index]
    output.flush()
    del output, source


def verify_array(source_path: Path, output_path: Path, keep_index: np.ndarray) -> None:
    source = np.load(source_path, mmap_mode="r")
    derived = np.load(output_path, mmap_mode="r")
    if derived.shape != (source.shape[0], keep_index.size):
        raise AssertionError(f"{output_path.name}: shape {derived.shape} != expected")
    for start in range(0, source.shape[0], CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, source.shape[0])
        if not np.array_equal(derived[start:stop], source[start:stop][:, keep_index]):
            raise AssertionError(f"{output_path.name}: rows [{start}:{stop}) differ from the source slice")


def build_metadata(source_meta: dict[str, Any], keep_fields: list[str], keep_index: np.ndarray,
                   source_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    metadata = json.loads(json.dumps(source_meta))  # deep copy
    metadata["state_fields"] = keep_fields
    metadata["target_fields"] = [source_meta["target_fields"][index] for index in keep_index]
    metadata["state_field_preset"] = args.state_field_preset or "custom_subset"

    normalization = metadata["normalization"]
    for key in ("state_mean", "state_std", "target_mean", "target_std"):
        values = source_meta["normalization"][key]
        normalization[key] = [values[index] for index in keep_index]

    metadata["derived_from"] = {
        "source_dataset_dir": str(source_dir),
        "source_state_field_preset": source_meta.get("state_field_preset"),
        "dropped_state_fields": [f for f in source_meta["state_fields"] if f not in set(keep_fields)],
        "tool": "scripts/ablation_ofat/derive_state_subset_dataset.py",
        "note": (
            "Column subset of the source processed cache; states/targets sliced, "
            "actions/rollout/episode arrays symlinked to the source's real files."
        ),
    }
    return metadata


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    source_meta = json.loads((source_dir / "metadata.json").read_text())
    source_fields = list(source_meta["state_fields"])
    keep_fields = resolve_keep_fields(source_fields, args)
    keep_index = np.asarray([source_fields.index(field) for field in keep_fields], dtype=np.int64)

    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_dir} exists (pass --overwrite to replace)")
        for child in sorted(output_dir.iterdir()):
            child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"source : {source_dir}")
    print(f"output : {output_dir}")
    print(f"keep   : {len(keep_fields)}/{len(source_fields)} state fields -> {keep_fields}")
    print(f"drop   : {[f for f in source_fields if f not in set(keep_fields)]}")

    for split in SPLITS:
        for array in ("states", "targets"):
            source_path = source_dir / f"{split}_{array}.npy"
            output_path = output_dir / f"{split}_{array}.npy"
            print(f"slicing {source_path.name} -> {output_path.name}", flush=True)
            slice_array(source_path, output_path, keep_index)
        for name in SHARED_ARRAYS:
            link_shared(source_dir, output_dir, f"{split}_{name}.npy")
        for name in SHARED_JSON:
            link_shared(source_dir, output_dir, f"{split}_{name}.json")

    metadata = build_metadata(source_meta, keep_fields, keep_index, source_dir, args)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if args.verify:
        for split in SPLITS:
            for array in ("states", "targets"):
                print(f"verifying {split}_{array}.npy", flush=True)
                verify_array(source_dir / f"{split}_{array}.npy", output_dir / f"{split}_{array}.npy", keep_index)
        print("verify: derived arrays match the source column slice exactly")

    print(f"done: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
