"""Fetch the NeDM datasets from Hugging Face into this repo's artifact layout.

    PYTHONPATH=src python scripts/release/download_nedm_datasets.py --dataset tracked --processed
    PYTHONPATH=src python scripts/release/download_nedm_datasets.py --dataset arm --rehydrate
    PYTHONPATH=src python scripts/release/download_nedm_datasets.py --dataset all --processed

What lands where (all under the repo root unless ``--dest`` is given):

* raw Parquet, episodes table and metadata bundle -> ``artifacts/hf_release/download/raw/<name>/``
* ``--processed``  the training caches the deployed models read
                   -> ``artifacts/training_datasets/<cache>/`` (every config in ``configs/`` works verbatim)
* ``--rehydrate``  the original per-episode CSV tree rebuilt from the Parquet files
                   -> ``artifacts/datasets/<original name>/`` so ``scripts/preprocess/*`` and the
                   collectors' validators run unchanged (values are the float32 the trainer uses;
                   caches rebuilt from a rehydrated tree are bit-identical to the published ones)

Anonymous downloads work for this public repo; set ``HF_TOKEN`` or run ``hf auth login``
only if you hit rate limits. Use ``--local-release`` to skip the download and rehydrate
from a staged copy (used by the release validation).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm import hf_release as hr  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="+", default=["all"], choices=[*hr.DATASETS, "all"])
    parser.add_argument("--repo-id", default=hr.HF_REPO_ID)
    parser.add_argument("--revision", default=None, help="Hub revision (branch/tag/commit); default main.")
    parser.add_argument("--dest", type=Path, default=REPO_ROOT, help="Repo root to populate (default: this checkout).")
    parser.add_argument("--processed", action="store_true", help="Also fetch the processed training caches.")
    parser.add_argument("--no-raw", dest="raw", action="store_false", default=True, help="Skip the raw Parquet files.")
    parser.add_argument("--rehydrate", action="store_true", help="Rebuild the per-episode CSV tree under artifacts/datasets/.")
    parser.add_argument("--rehydrate-max-episodes", type=int, default=None, help="Debug: stop after N episodes.")
    parser.add_argument("--rehydrate-shards", nargs="+", default=None,
                        help="Only rebuild these shards/parts (e.g. shard_017); default all.")
    parser.add_argument("--local-release", type=Path, default=None,
                        help="Use an already staged/downloaded release tree instead of downloading.")
    return parser.parse_args(argv)


def download(repo_id: str, revision: str | None, patterns: list[str], local_dir: Path) -> Path:
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=patterns,
            local_dir=str(local_dir),
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = list(hr.DATASETS) if "all" in args.dataset else args.dataset
    dest = args.dest.resolve()
    release_dir = args.local_release.resolve() if args.local_release else dest / "artifacts" / "hf_release" / "download"

    if args.local_release is None:
        patterns: list[str] = ["README.md", "release_manifest.json"]
        if args.raw or args.rehydrate:
            patterns += [f"raw/{n}/**" for n in names]
            if "hmmwv_bumpy" in names:
                patterns.append("assets/bumpy_terrain/**")
        if args.processed:
            patterns += [f"processed/{hr.DATASETS[n].processed_cache}/**" for n in names if hr.DATASETS[n].processed_cache]
        print(f"downloading {args.repo_id} -> {release_dir}: {patterns}")
        download(args.repo_id, args.revision, patterns, release_dir)

    if args.processed:
        for name in names:
            cache = hr.DATASETS[name].processed_cache
            if cache is None:
                continue
            src = release_dir / "processed" / cache
            target = dest / "artifacts" / "training_datasets" / cache
            if src.resolve() == target.resolve():
                continue
            target.mkdir(parents=True, exist_ok=True)
            for file in sorted(src.iterdir()):
                if file.is_file() and not (target / file.name).exists():
                    shutil.copyfile(file, target / file.name)
            print(f"processed cache ready: {target}")

    if args.rehydrate:
        for name in names:
            hr.rehydrate_dataset(
                hr.DATASETS[name],
                release_dir,
                dest / "artifacts" / "datasets",
                max_episodes=args.rehydrate_max_episodes,
                only_shards=args.rehydrate_shards,
            )
        if "hmmwv_bumpy" in names:
            src = release_dir / "assets" / "bumpy_terrain"
            target = dest / "assets" / "bumpy_terrain"
            if src.exists() and not target.exists():
                shutil.copytree(src, target)
                print(f"heightmaps ready: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
