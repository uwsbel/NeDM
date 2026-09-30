"""Stage the NeDM datasets for the Hugging Face release.

Converts the raw per-episode CSV trees into the Parquet layout described in
``nedm.hf_release`` (float32 transitions per shard and split, an episodes table,
and a byte-exact metadata bundle), optionally copies the processed training
caches, and records every produced file in ``release_manifest.json``.

    PYTHONPATH=src python scripts/release/export_hf_dataset.py --dataset tracked arm
    PYTHONPATH=src python scripts/release/export_hf_dataset.py --dataset all --processed --workers 8

Re-running skips shards whose Parquet files already exist (``--force`` redoes them).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pyarrow as pa  # noqa: E402

from nedm import hf_release as hr  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="+", default=["all"], choices=[*hr.DATASETS, "all"],
                        help="Release dataset names (default: all).")
    parser.add_argument("--staging", type=Path, default=hr.DEFAULT_STAGING_DIR,
                        help="Local mirror of the Hub repo tree.")
    parser.add_argument("--workers", type=int, default=6, help="Parallel shard conversions.")
    parser.add_argument("--threads-per-worker", type=int, default=4, help="Arrow threads per worker.")
    parser.add_argument("--processed", action="store_true", help="Also stage the processed caches.")
    parser.add_argument("--raw", dest="raw", action="store_true", default=True)
    parser.add_argument("--no-raw", dest="raw", action="store_false", help="Skip the raw Parquet export.")
    parser.add_argument("--max-shards", type=int, default=None, help="Debug: only the first N shards per dataset.")
    parser.add_argument("--force", action="store_true", help="Redo shards that are already staged.")
    parser.add_argument("--max-attempts", type=int, default=6, help="Pool restarts tolerated per dataset.")
    return parser.parse_args(argv)


def _worker(args: tuple[str, int, str, bool, int]) -> hr.ShardExportResult:
    release_name, shard_index, staging, force, threads = args
    pa.set_cpu_count(max(1, threads))
    pa.set_io_thread_count(max(1, threads))
    spec = hr.DATASETS[release_name]
    shard = hr.list_raw_shards(spec)[shard_index]
    return hr.export_shard(spec, shard, Path(staging), force=force, log=lambda m: print(m, flush=True))


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = list(hr.DATASETS) if "all" in args.dataset else args.dataset
    staging = args.staging.resolve()
    staging.mkdir(parents=True, exist_ok=True)
    manifest = hr.load_manifest(staging)
    manifest.setdefault("tool_versions", {})
    import huggingface_hub  # noqa: PLC0415
    manifest["tool_versions"].update({"pyarrow": pa.__version__, "huggingface_hub": huggingface_hub.__version__})
    manifest["git_commit"] = git_commit()
    manifest["parquet"] = {
        "compression": hr.PARQUET_COMPRESSION,
        "compression_level": hr.PARQUET_COMPRESSION_LEVEL,
        "row_group_rows": hr.PARQUET_ROW_GROUP_ROWS,
        "float_dtype": "float32 (time_s float64)",
        "byte_stream_split": True,
    }

    if args.raw:
        for name in names:
            spec = hr.DATASETS[name]
            shards = hr.list_raw_shards(spec)
            if args.max_shards is not None:
                shards = shards[: args.max_shards]
            print(f"== {name}: {len(shards)} shards from {spec.raw_dir}", flush=True)
            t0 = time.time()
            jobs = {i: (name, i, str(staging), args.force, args.threads_per_worker) for i in range(len(shards))}
            results_by_index: dict[int, hr.ShardExportResult] = {}
            # Worker processes have segfaulted inside libarrow on this box (flaky
            # hardware); shard writes are atomic and re-runs skip finished shards,
            # so a broken pool is simply rebuilt for the shards still pending.
            for attempt in range(1, args.max_attempts + 1):
                pending = [jobs[i] for i in sorted(jobs) if i not in results_by_index]
                if not pending:
                    break
                try:
                    with cf.ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
                        futures = {pool.submit(_worker, job): job[1] for job in pending}
                        for future in cf.as_completed(futures):
                            results_by_index[futures[future]] = future.result()
                except cf.process.BrokenProcessPool as exc:
                    print(f"!! worker pool broke on attempt {attempt} ({exc}); "
                          f"{len(jobs) - len(results_by_index)} shards pending, retrying", flush=True)
                    if attempt == args.max_attempts:
                        raise
            results = [results_by_index[i] for i in sorted(results_by_index)]
            columns = results[0].columns
            for r in results:
                if r.columns != columns:
                    raise ValueError(f"{name}: shard {r.shard} has a different column set")

            episodes_table = hr.build_episodes_table(spec, shards)
            episodes_path = staging / "raw" / name / "episodes.parquet"
            hr.write_parquet(episodes_table, episodes_path)
            bundle_info = hr.build_metadata_bundle(spec, shards, staging / "raw" / name / "metadata.tar.gz")
            extra = hr.export_extra_files(spec, staging)

            files: dict[str, dict] = {}
            for r in results:
                files.update(r.files)
            files[str(episodes_path.relative_to(staging))] = {
                "bytes": episodes_path.stat().st_size,
                "sha256": hr.sha256_file(episodes_path),
                "rows": episodes_table.num_rows,
            }
            files[f"raw/{name}/metadata.tar.gz"] = bundle_info
            files.update(extra)
            split_counts = {"train": {"episodes": 0, "rows": 0}, "val": {"episodes": 0, "rows": 0}}
            for path, info in files.items():
                for split in split_counts:
                    if path.startswith(f"raw/{name}/{split}/"):
                        split_counts[split]["episodes"] += info["episodes"]
                        split_counts[split]["rows"] += info["rows"]
            manifest["datasets"][name] = {
                "description": spec.description,
                "raw_dir": spec.raw_dir,
                "sharded": spec.sharded,
                "shards": [s.name for s in shards],
                "columns": columns,
                "episodes": sum(len(hr.shard_episodes(s)) for s in shards),
                "rows": sum(sum(r.episode_rows.values()) for r in results),
                "splits": split_counts,
                "bytes": sum(f["bytes"] for f in files.values()),
                "processed_cache": spec.processed_cache,
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            manifest["files"].update(files)
            hr.save_manifest(staging, manifest)
            print(f"== {name}: done in {time.time()-t0:.0f}s, {manifest['datasets'][name]['bytes']/1e9:.2f} GB staged", flush=True)

    if args.processed:
        for name in names:
            spec = hr.DATASETS[name]
            if spec.processed_cache is None:
                continue
            files = hr.export_processed_cache(spec.processed_cache, staging, force=args.force)
            manifest["processed"][spec.processed_cache] = {
                "dataset": name,
                "bytes": sum(f["bytes"] for f in files.values()),
                "files": sorted(files),
            }
            manifest["files"].update(files)
            hr.save_manifest(staging, manifest)
            print(f"== processed/{spec.processed_cache}: {sum(f['bytes'] for f in files.values())/1e9:.2f} GB", flush=True)

    card = REPO_ROOT / "docs" / "hf_dataset_card.md"
    if card.exists():
        (staging / "README.md").write_text(card.read_text())
    hr.save_manifest(staging, manifest)
    total = sum(f["bytes"] for f in manifest["files"].values())
    print(f"manifest: {len(manifest['files'])} files, {total/1e9:.2f} GB at {staging}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())
