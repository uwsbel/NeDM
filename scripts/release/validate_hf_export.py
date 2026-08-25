"""Check a staged (or downloaded) NeDM release against the raw CSV trees.

For every dataset recorded in ``release_manifest.json``:
  * split episode/row totals in the Parquet files match the raw ``dataset_index.json``s;
  * no column contains nulls;
  * a random sample of episodes compares exactly (float32(csv) == parquet, time_s
    as float64, strings and ints equal) against the raw CSV, read the way the
    trainer reads it;
  * ``episodes.parquet`` lists every episode once, and ``metadata.tar.gz`` holds
    byte-exact copies of the JSON originals (sampled);
  * processed caches: array shapes/dtypes match ``metadata.json``, no symlinks,
    no machine-local paths.

    PYTHONPATH=src python scripts/release/validate_hf_export.py --staging artifacts/hf_release/NeDM --sample 20
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tarfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm import hf_release as hr  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staging", type=Path, default=hr.DEFAULT_STAGING_DIR)
    parser.add_argument("--dataset", nargs="+", default=None, help="Subset of release datasets (default: all in manifest).")
    parser.add_argument("--sample", type=int, default=12, help="Episodes per dataset to compare value-by-value.")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--skip-processed", action="store_true")
    parser.add_argument("--full-read", action="store_true",
                        help="Decode every Parquet file completely and check every episode's row count "
                             "(catches corrupt pages; ~minutes for the flat set).")
    return parser.parse_args(argv)


def read_csv_like_trainer(csv_path: Path, columns: list[str]) -> dict[str, list]:
    values: dict[str, list] = {c: [] for c in columns}
    with csv_path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            for c in columns:
                values[c].append(row[c])
    return values


def compare_episode(spec: hr.DatasetSpec, shard: hr.RawShard, episode: dict, staging: Path, columns: list[str]) -> list[str]:
    problems: list[str] = []
    parquet_path = staging / "raw" / spec.release_name / episode["split"] / f"{shard.name}.parquet"
    table = pq.read_table(parquet_path, filters=[("episode_id", "==", episode["episode_id"])])
    if table.num_rows != int(episode["rows"]):
        problems.append(f"{episode['episode_id']}: parquet rows {table.num_rows} != index rows {episode['rows']}")
        return problems
    raw = read_csv_like_trainer(shard.root / episode["csv_path"], columns)
    for column in columns:
        got = table.column(column)
        if column in hr.STRING_COLUMNS:
            if got.cast(pa.string()).to_pylist() != raw[column]:
                problems.append(f"{episode['episode_id']}: string column {column} differs")
        elif column in hr.INT_COLUMNS:
            if not np.array_equal(np.asarray(got.to_numpy()), np.asarray([int(v) for v in raw[column]], dtype=np.int32)):
                problems.append(f"{episode['episode_id']}: int column {column} differs")
        elif column in hr.FLOAT64_COLUMNS:
            expected = np.asarray([float(v) for v in raw[column]], dtype=np.float64)
            if not np.array_equal(np.asarray(got.to_numpy()), expected):
                problems.append(f"{episode['episode_id']}: float64 column {column} differs")
        else:
            expected = np.asarray([float(v) for v in raw[column]], dtype=np.float32)
            actual = np.asarray(got.to_numpy())
            if actual.dtype != np.float32 or not np.array_equal(actual, expected):
                problems.append(f"{episode['episode_id']}: float32 column {column} differs (max abs {np.max(np.abs(actual.astype(np.float64)-expected)) if actual.shape==expected.shape else 'shape'})")
    return problems


def validate_dataset(name: str, staging: Path, sample: int, rng: random.Random, full_read: bool = False) -> tuple[bool, dict]:
    spec = hr.DATASETS[name]
    shards = hr.list_raw_shards(spec)
    ok = True
    report: dict = {"dataset": name}

    # Totals from raw indices.
    raw_totals = {"train": {"episodes": 0, "rows": 0}, "val": {"episodes": 0, "rows": 0}}
    all_ids: list[str] = []
    for shard in shards:
        for ep in hr.shard_episodes(shard):
            raw_totals[ep["split"]]["episodes"] += 1
            raw_totals[ep["split"]]["rows"] += int(ep["rows"])
            all_ids.append(ep["episode_id"])
    # Totals + null check from parquet metadata.
    pq_totals = {"train": {"episodes": 0, "rows": 0}, "val": {"episodes": 0, "rows": 0}}
    null_columns: set[str] = set()
    columns: list[str] | None = None
    for split in ("train", "val"):
        for path in sorted((staging / "raw" / name / split).glob("*.parquet")):
            meta = pq.read_metadata(path)
            pq_totals[split]["rows"] += meta.num_rows
            schema_names = pq.read_schema(path).names
            columns = columns or schema_names
            if schema_names != columns:
                ok = False
                report.setdefault("problems", []).append(f"{path.name}: column set differs")
            for rg in range(meta.num_row_groups):
                group = meta.row_group(rg)
                for ci in range(group.num_columns):
                    col = group.column(ci)
                    if col.statistics is not None and col.statistics.null_count:
                        null_columns.add(col.path_in_schema)
            ids = pq.read_table(path, columns=["episode_id"]).column("episode_id")
            pq_totals[split]["episodes"] += len(ids.unique())
    if null_columns:
        ok = False
        report.setdefault("problems", []).append(f"null values in columns: {sorted(null_columns)}")
    if pq_totals != raw_totals:
        ok = False
        report.setdefault("problems", []).append(f"totals differ: parquet {pq_totals} vs raw {raw_totals}")
    report["totals"] = raw_totals
    report["columns"] = len(columns or [])

    # Episodes table.
    episodes = pq.read_table(staging / "raw" / name / "episodes.parquet")
    table_ids = episodes.column("episode_id").to_pylist()
    if sorted(table_ids) != sorted(all_ids) or len(set(table_ids)) != len(table_ids):
        ok = False
        report.setdefault("problems", []).append("episodes.parquet ids != raw episode ids")
    report["episodes_table_columns"] = episodes.num_columns

    # Metadata bundle: sample byte-exactness.
    with tarfile.open(staging / "raw" / name / "metadata.tar.gz", "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        report["metadata_members"] = len(members)
        want = [f"{spec.raw_name}/{s.subdir}/dataset_index.json".replace("//", "/") for s in shards]
        want = list(dict.fromkeys(want))
        for arc in want + rng.sample(sorted(members), min(10, len(members))):
            if arc not in members:
                ok = False
                report.setdefault("problems", []).append(f"metadata bundle missing {arc}")
                continue
            data = tar.extractfile(members[arc]).read()
            rel = Path(arc)
            if rel.parts[0] == "plans":
                src = REPO_ROOT / "artifacts" / "datasets" / Path(*rel.parts[1:])
            else:
                src = spec.raw_root.parent / rel
            if not src.exists() or src.read_bytes() != data:
                ok = False
                report.setdefault("problems", []).append(f"metadata bundle member differs from source: {arc}")

    # Optional exhaustive decode + per-episode row counts.
    if full_read:
        expected_rows = {}
        for shard in shards:
            for ep in hr.shard_episodes(shard):
                expected_rows[ep["episode_id"]] = int(ep["rows"])
        seen: dict[str, int] = {}
        for split in ("train", "val"):
            for path in sorted((staging / "raw" / name / split).glob("*.parquet")):
                table = pq.read_table(path)  # full decode of every column
                for episode_id, _start, length in hr.episode_runs(table.column("episode_id")):
                    seen[episode_id] = seen.get(episode_id, 0) + length
        bad = [e for e, n in expected_rows.items() if seen.get(e) != n]
        extra = [e for e in seen if e not in expected_rows]
        if bad or extra:
            ok = False
            report.setdefault("problems", []).append(f"full-read row-count mismatches: {len(bad)} bad, {len(extra)} extra (e.g. {(bad+extra)[:5]})")
        report["full_read_episodes"] = len(seen)

    # Value-level sample.
    picks = []
    for shard in shards:
        for ep in hr.shard_episodes(shard):
            picks.append((shard, ep))
    sampled = rng.sample(picks, min(sample, len(picks)))
    problems: list[str] = []
    for shard, ep in sampled:
        problems.extend(compare_episode(spec, shard, ep, staging, columns or []))
    if problems:
        ok = False
        report.setdefault("problems", []).extend(problems[:20])
    report["sampled_episodes"] = len(sampled)
    return ok, report


def validate_processed(cache_name: str, staging: Path) -> tuple[bool, dict]:
    root = staging / "processed" / cache_name
    ok = True
    report: dict = {"cache": cache_name}
    if not root.exists():
        return False, {"cache": cache_name, "problems": ["missing"]}
    meta = json.loads((root / "metadata.json").read_text())
    for split in ("train", "val"):
        n = meta["splits"][split]["transition_count"]
        n_eps = meta["splits"][split]["episode_count"]
        expect = {
            "states": ((n, len(meta["state_fields"])), np.float32),
            "actions": ((n, len(meta["action_fields"])), np.float32),
            "targets": ((n, len(meta["target_fields"])), np.float32),
            "rollout": ((n + n_eps, len(meta["rollout_fields"])), np.float32),
            "episode_starts": ((n_eps,), np.int64),
            "episode_lengths": ((n_eps,), np.int32),
        }
        for stem, (shape, dtype) in expect.items():
            path = root / f"{split}_{stem}.npy"
            if path.is_symlink():
                ok = False
                report.setdefault("problems", []).append(f"{path.name} is a symlink")
            arr = np.load(path, mmap_mode="r")
            if arr.shape != shape or arr.dtype != dtype:
                ok = False
                report.setdefault("problems", []).append(f"{path.name}: {arr.shape}/{arr.dtype} != {shape}/{dtype}")
        eps = json.loads((root / f"{split}_episodes.json").read_text())
        if eps["episode_count"] != n_eps or len(eps["episode_ids"]) != n_eps:
            ok = False
            report.setdefault("problems", []).append(f"{split}_episodes.json episode_count mismatch")
    for js in root.glob("*.json"):
        if "/home/" in js.read_text():
            ok = False
            report.setdefault("problems", []).append(f"{js.name} still contains a machine-local path")
    report["bytes"] = sum(p.stat().st_size for p in root.iterdir())
    return ok, report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    staging = args.staging.resolve()
    manifest = hr.load_manifest(staging)
    names = args.dataset or list(manifest["datasets"])
    rng = random.Random(args.seed)
    all_ok = True
    for name in names:
        ok, report = validate_dataset(name, staging, args.sample, rng, full_read=args.full_read)
        all_ok &= ok
        print(("PASS" if ok else "FAIL"), json.dumps(report))
    if not args.skip_processed:
        for cache_name in manifest.get("processed", {}):
            ok, report = validate_processed(cache_name, staging)
            all_ok &= ok
            print(("PASS" if ok else "FAIL"), json.dumps(report))
    # Every manifest file exists with the recorded size.
    missing = [p for p, info in manifest["files"].items() if not (staging / p).exists() or (staging / p).stat().st_size != info["bytes"]]
    if missing:
        all_ok = False
        print("FAIL manifest files missing/size-changed:", missing[:10])
    print("OVERALL", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
