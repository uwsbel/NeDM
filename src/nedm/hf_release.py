"""Hugging Face release of the NeDM datasets.

The five raw episode datasets behind the manuscript's dynamics models are
per-episode CSV trees (``artifacts/datasets/<name>[/shard_NNN]/episodes/*.csv``
plus ``dataset_index.json`` / ``collector_config.resolved.json`` and a JSON
sidecar per episode). This module converts them into the Parquet layout that
is published at https://huggingface.co/datasets/harryzhang1018/NeDM and back:

* ``export_dataset``  – raw CSV shards -> ``raw/<release>/{train,val}/<shard>.parquet``
  (float32, exact CSV column names/order), ``raw/<release>/episodes.parquet``
  (one row per episode: index entry + sidecar) and ``raw/<release>/metadata.tar.gz``
  (byte-exact originals of every JSON file, so the original tree can be
  rebuilt).
* ``export_processed_cache`` – the ``.npy`` training caches the deployed models
  read, with symlinks materialised and machine-local paths made repo-relative.
* ``rehydrate_dataset`` – Parquet + metadata bundle -> the original per-episode
  CSV tree under ``artifacts/datasets/``, so ``nedm.training.preprocess`` and the
  RL reference builders run unchanged. Values are the float32 the trainer uses;
  the caches rebuilt from a rehydrated tree are bit-identical to the published ones.

Everything is driven by ``DATASETS`` below. Scripts: ``scripts/release/``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
HF_REPO_ID = "harryzhang1018/NeDM"
DEFAULT_STAGING_DIR = Path("artifacts/hf_release/NeDM")

# CSV columns that are not physical channels.
STRING_COLUMNS = {"episode_id", "scenario_name", "scenario_family", "split", "collision_kind"}
INT_COLUMNS = {"sample_index", "collision"}
FLOAT64_COLUMNS = {"time_s"}

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3
PARQUET_ROW_GROUP_ROWS = 262_144


@dataclass(frozen=True)
class DatasetSpec:
    """One published dataset: where its raw tree lives and how it is chunked."""

    release_name: str
    raw_dir: str  # relative to REPO_ROOT; the original on-disk name is preserved on rehydration
    sharded: bool  # True: raw_dir/shard_NNN/{dataset_index.json,...}; False: raw_dir/{...}
    description: str
    exclude_shards: tuple[str, ...] = ()
    part_episodes: int | None = None  # unsharded datasets are chunked into parts of this many episodes
    plan_dirs: tuple[str, ...] = ()  # sibling shard-plan directories bundled into metadata.tar.gz
    processed_cache: str | None = None  # artifacts/training_datasets/<name> used by the deployed model
    extra_files: tuple[str, ...] = ()  # repo-relative files/dirs copied verbatim into the release

    @property
    def raw_root(self) -> Path:
        return REPO_ROOT / self.raw_dir

    @property
    def raw_name(self) -> str:
        return Path(self.raw_dir).name


DATASETS: dict[str, DatasetSpec] = {
    "hmmwv_flat": DatasetSpec(
        release_name="hmmwv_flat",
        raw_dir="artifacts/datasets/hmmwv_tire_rigid_300g_shards",
        sharded=True,
        description="HMMWV on flat rigid terrain (TMEASY tires, 100 Hz), 128 shards",
        plan_dirs=("artifacts/datasets/hmmwv_tire_rigid_300g_plan",),
        processed_cache="hmmwv_tire_rigid_300g_normal_force_omega_seq_v1",
    ),
    "hmmwv_bumpy": DatasetSpec(
        release_name="hmmwv_bumpy",
        raw_dir="artifacts/datasets/hmmwv_bumpy_10g_shards",
        sharded=True,
        description="HMMWV on bumpy rigid heightmap terrain (100 Hz), 4 shards; zero-shot test only",
        exclude_shards=("smoke",),
        plan_dirs=("artifacts/datasets/hmmwv_bumpy_10g_plan",),
        extra_files=("assets/bumpy_terrain",),
    ),
    "hmmwv_crm": DatasetSpec(
        release_name="hmmwv_crm",
        raw_dir="artifacts/datasets/hmmwv_crm_2000",
        sharded=False,
        description="HMMWV on CRM deformable soil (SPH terramechanics, rigid-mesh tires, 100 Hz)",
        part_episodes=500,
        processed_cache="hmmwv_crm_2000_normal_force_omega_seq_v1",
    ),
    "arm": DatasetSpec(
        release_name="arm",
        raw_dir="artifacts/datasets/arm_dynamics_v3_home_reset_fulltraj_shards",
        sharded=True,
        description="4-DOF LRV arm on the M113 (PD joint torques, 50 Hz), 15 shards",
        processed_cache="arm_dyn_v3_8d_seq16_v1",
    ),
    "tracked": DatasetSpec(
        release_name="tracked",
        raw_dir="artifacts/datasets/tracked_vehicle_drive_v2_shards",
        sharded=True,
        description="M113 tracked vehicle drive mode, arm welded at home (50 Hz), 60 shards",
        processed_cache="tracked_drive_v2_seq16_v1",
    ),
}


# --------------------------------------------------------------------------- shards


@dataclass(frozen=True)
class RawShard:
    """A unit of conversion: one directory holding ``dataset_index.json`` + ``episodes/``.

    For unsharded datasets several ``RawShard``s share the same ``root`` and carry
    disjoint episode slices (``episode_slice``)."""

    name: str  # shard_000 / part_000
    root: Path  # directory containing dataset_index.json
    subdir: str  # path of root relative to the dataset raw_root ("" for unsharded)
    episode_slice: slice = field(default=slice(None))


def list_raw_shards(spec: DatasetSpec) -> list[RawShard]:
    if spec.sharded:
        shards = []
        for child in sorted(spec.raw_root.iterdir()):
            if not child.is_dir() or child.name in spec.exclude_shards:
                continue
            if not (child / "dataset_index.json").exists():
                continue
            shards.append(RawShard(name=child.name, root=child, subdir=child.name))
        if not shards:
            raise FileNotFoundError(f"no shard directories with dataset_index.json under {spec.raw_root}")
        return shards
    index = load_json(spec.raw_root / "dataset_index.json")
    count = len(index["episodes"])
    size = spec.part_episodes or count
    return [
        RawShard(name=f"part_{i:03d}", root=spec.raw_root, subdir="", episode_slice=slice(start, start + size))
        for i, start in enumerate(range(0, count, size))
    ]


def shard_episodes(shard: RawShard) -> list[dict[str, Any]]:
    index = load_json(shard.root / "dataset_index.json")
    return list(index["episodes"])[shard.episode_slice]


# --------------------------------------------------------------------------- CSV -> Arrow


def read_csv_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", newline="") as fp:
        return fp.readline().rstrip("\r\n").split(",")


def csv_convert_options(columns: list[str]) -> pacsv.ConvertOptions:
    """Parse every column with an explicit type; floats as float64 first.

    Reading doubles then casting to float32 reproduces exactly what the trainer
    does (``float(row[field])`` -> ``np.float32``); parsing decimal text straight
    into float32 could double-round differently in rare cases."""
    types: dict[str, pa.DataType] = {}
    for column in columns:
        if column in STRING_COLUMNS:
            types[column] = pa.string()
        elif column in INT_COLUMNS:
            types[column] = pa.int32()
        else:
            types[column] = pa.float64()
    return pacsv.ConvertOptions(
        column_types=types,
        include_columns=columns,
        null_values=[],  # nothing is null: "nan"/"inf" tokens would be parse errors, not silent nulls
        strings_can_be_null=False,
    )


def read_episode_csv(csv_path: Path, columns: list[str]) -> pa.Table:
    table = pacsv.read_csv(
        csv_path,
        read_options=pacsv.ReadOptions(use_threads=True),
        parse_options=pacsv.ParseOptions(newlines_in_values=False),
        convert_options=csv_convert_options(columns),
    )
    if table.column_names != columns:
        raise ValueError(f"{csv_path}: column order differs from the shard header")
    return table


def release_schema(columns: list[str]) -> pa.Schema:
    fields = []
    for column in columns:
        if column in STRING_COLUMNS:
            fields.append(pa.field(column, pa.dictionary(pa.int32(), pa.string()), nullable=False))
        elif column in INT_COLUMNS:
            fields.append(pa.field(column, pa.int32(), nullable=False))
        elif column in FLOAT64_COLUMNS:
            fields.append(pa.field(column, pa.float64(), nullable=False))
        else:
            fields.append(pa.field(column, pa.float32(), nullable=False))
    return pa.schema(fields)


def to_release_table(table: pa.Table, schema: pa.Schema) -> pa.Table:
    arrays = []
    for f in schema:
        column = table.column(f.name)
        if pa.types.is_dictionary(f.type):
            column = column.dictionary_encode() if not pa.types.is_dictionary(column.type) else column
            column = column.cast(f.type)
        else:
            column = column.cast(f.type)  # float64 -> float32 is round-to-nearest-even, same as numpy astype
        arrays.append(column)
    return pa.Table.from_arrays(arrays, schema=schema)


def write_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        table,
        tmp,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
        row_group_size=PARQUET_ROW_GROUP_ROWS,
        # Dictionary pages only for the string columns: dictionary-encoding the
        # float channels defeats zstd (5.4 vs 3.1 MB on a tracked shard);
        # BYTE_STREAM_SPLIT + zstd gets ~28% below raw float32.
        use_dictionary=[f.name for f in table.schema if pa.types.is_dictionary(f.type) or pa.types.is_string(f.type)],
        use_byte_stream_split=[f.name for f in table.schema if pa.types.is_floating(f.type)],
        write_statistics=True,
    )
    tmp.replace(path)


# --------------------------------------------------------------------------- export


@dataclass
class ShardExportResult:
    shard: str
    files: dict[str, dict[str, Any]]  # release-relative path -> {bytes, sha256, rows, episodes}
    episode_rows: dict[str, int]
    columns: list[str]


def export_shard(
    spec: DatasetSpec,
    shard: RawShard,
    staging_dir: Path,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> ShardExportResult:
    """Convert one raw shard into ``raw/<release>/{train,val}/<shard>.parquet``."""
    episodes = shard_episodes(shard)
    if not episodes:
        raise ValueError(f"{spec.release_name}/{shard.name}: no episodes")
    columns = read_csv_header(shard.root / episodes[0]["csv_path"])
    schema = release_schema(columns)
    out_dir = staging_dir / "raw" / spec.release_name
    outputs = {split: out_dir / split / f"{shard.name}.parquet" for split in ("train", "val")}

    files: dict[str, dict[str, Any]] = {}
    episode_rows: dict[str, int] = {}
    if not force and all(p.exists() for p in outputs.values()):
        # Already exported: re-derive the bookkeeping from the parquet files.
        for split, path in outputs.items():
            meta = pq.read_metadata(path)
            ids = pq.read_table(path, columns=["episode_id"]).column("episode_id")
            counts = ids.value_counts().to_pylist()
            for entry in counts:
                episode_rows[entry["values"]] = int(entry["counts"])
            files[str(path.relative_to(staging_dir))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rows": meta.num_rows,
                "episodes": len(counts),
            }
        log(f"[{spec.release_name}/{shard.name}] already exported, skipped")
        return ShardExportResult(shard.name, files, episode_rows, columns)

    tables: dict[str, list[pa.Table]] = {"train": [], "val": []}
    for episode in episodes:
        csv_path = shard.root / episode["csv_path"]
        table = read_episode_csv(csv_path, columns)
        expected_rows = int(episode["rows"])
        if table.num_rows != expected_rows:
            raise ValueError(
                f"{csv_path}: {table.num_rows} rows but dataset_index says {expected_rows}"
            )
        split_col = table.column("split").unique().to_pylist()
        ep_col = table.column("episode_id").unique().to_pylist()
        if split_col != [episode["split"]] or ep_col != [episode["episode_id"]]:
            raise ValueError(f"{csv_path}: split/episode_id columns disagree with dataset_index")
        episode_rows[episode["episode_id"]] = expected_rows
        tables[episode["split"]].append(table)

    for split, parts in tables.items():
        path = outputs[split]
        if parts:
            table = to_release_table(pa.concat_tables(parts), schema)
        else:
            table = schema.empty_table()
        write_parquet(table, path)
        files[str(path.relative_to(staging_dir))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": table.num_rows,
            "episodes": len(parts),
        }
    total_rows = sum(episode_rows.values())
    out_bytes = sum(f["bytes"] for f in files.values())
    log(f"[{spec.release_name}/{shard.name}] {len(episodes)} episodes, {total_rows} rows -> {out_bytes/1e6:.1f} MB")
    return ShardExportResult(shard.name, files, episode_rows, columns)


def _flatten_for_table(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def build_episodes_table(spec: DatasetSpec, shards: list[RawShard]) -> pa.Table:
    """One row per episode: dataset_index entry merged with its JSON sidecar.

    Sidecar keys that duplicate the index entry with the same value are dropped;
    conflicting ones are kept with a ``sidecar_`` prefix. Nested values become
    JSON strings so the table stays flat for the dataset viewer."""
    rows: list[dict[str, Any]] = []
    for shard in shards:
        index = load_json(shard.root / "dataset_index.json")
        for episode in list(index["episodes"])[shard.episode_slice]:
            record: dict[str, Any] = {
                "episode_id": episode["episode_id"],
                "split": episode["split"],
                "source_dataset": index["dataset_name"],
                "source_shard": shard.name,
                "parquet_file": f"{episode['split']}/{shard.name}.parquet",
            }
            for key, value in episode.items():
                record.setdefault(key, value)
            sidecar_path = (shard.root / episode["csv_path"]).with_suffix(".json")
            if sidecar_path.exists():
                sidecar = load_json(sidecar_path)
                for key, value in sidecar.items():
                    if key in record:
                        if record[key] != value:
                            record[f"sidecar_{key}"] = value
                    else:
                        record[key] = value
            rows.append({k: _flatten_for_table(v) for k, v in record.items()})
    # Stable column order: first-seen.
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    arrays = {c: [row.get(c) for row in rows] for c in columns}
    return pa.table(arrays)


def _tar_add_tree(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        tar.add(file, arcname=f"{arcname}/{file.relative_to(path).as_posix()}", recursive=False)


def build_metadata_bundle(spec: DatasetSpec, shards: list[RawShard], out_path: Path) -> dict[str, Any]:
    """Byte-exact copy of every JSON metadata file (index, resolved config, sidecars, plans).

    Archive paths start with the original dataset directory name, so extracting the
    bundle into ``artifacts/datasets/`` recreates the original tree minus the CSVs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    count = 0
    seen_roots: set[Path] = set()
    with tarfile.open(tmp, "w:gz", compresslevel=6) as tar:
        for shard in shards:
            if shard.root in seen_roots:
                continue
            seen_roots.add(shard.root)
            prefix = spec.raw_name if not shard.subdir else f"{spec.raw_name}/{shard.subdir}"
            for name in ("dataset_index.json", "collector_config.resolved.json"):
                candidate = shard.root / name
                if candidate.exists():
                    tar.add(candidate, arcname=f"{prefix}/{name}", recursive=False)
                    count += 1
            for sidecar in sorted((shard.root / "episodes").glob("*.json")):
                tar.add(sidecar, arcname=f"{prefix}/episodes/{sidecar.name}", recursive=False)
                count += 1
        for plan_dir in spec.plan_dirs:
            plan_path = REPO_ROOT / plan_dir
            if plan_path.exists():
                _tar_add_tree(tar, plan_path, f"plans/{plan_path.name}")
                count += sum(1 for p in plan_path.rglob("*") if p.is_file())
    tmp.replace(out_path)
    return {"bytes": out_path.stat().st_size, "sha256": sha256_file(out_path), "members": count}


def scrub_paths(value: Any, repo_root: Path = REPO_ROOT) -> Any:
    """Make machine-local absolute paths repo-relative inside JSON metadata."""
    prefix = str(repo_root) + "/"
    if isinstance(value, str):
        return value[len(prefix):] if value.startswith(prefix) else value
    if isinstance(value, list):
        return [scrub_paths(v, repo_root) for v in value]
    if isinstance(value, dict):
        return {k: scrub_paths(v, repo_root) for k, v in value.items()}
    return value


def export_processed_cache(cache_name: str, staging_dir: Path, force: bool = False,
                           log: Callable[[str], None] = print) -> dict[str, dict[str, Any]]:
    """Copy ``artifacts/training_datasets/<cache_name>`` into ``processed/<cache_name>``.

    Symlinks are materialised (the 300g cache links its unchanged arrays into the
    23-D parent cache) and absolute paths in the JSON files are made repo-relative."""
    src = REPO_ROOT / "artifacts" / "training_datasets" / cache_name
    if not src.exists():
        raise FileNotFoundError(src)
    dst = staging_dir / "processed" / cache_name
    dst.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for entry in sorted(src.iterdir()):
        target = dst / entry.name
        real = entry.resolve()
        if entry.suffix == ".json":
            payload = scrub_paths(load_json(real))
            text = json.dumps(payload, indent=2)
            if force or not target.exists() or target.read_text() != text:
                target.write_text(text)
        elif entry.suffix == ".npy":
            if force or not target.exists() or target.stat().st_size != real.stat().st_size:
                log(f"[processed/{cache_name}] copying {entry.name} ({real.stat().st_size/1e9:.2f} GB)")
                shutil.copyfile(real, target)
        else:
            continue
        files[str(target.relative_to(staging_dir))] = {
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
    return files


def export_extra_files(spec: DatasetSpec, staging_dir: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for rel in spec.extra_files:
        src = REPO_ROOT / rel
        for file in sorted(p for p in src.rglob("*") if p.is_file()) if src.is_dir() else [src]:
            target = staging_dir / file.relative_to(REPO_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != file.stat().st_size:
                shutil.copyfile(file, target)
            files[str(target.relative_to(staging_dir))] = {"bytes": target.stat().st_size, "sha256": sha256_file(target)}
    return files


# --------------------------------------------------------------------------- rehydrate


def episode_runs(episode_ids: pa.ChunkedArray) -> list[tuple[str, int, int]]:
    """Contiguous (episode_id, start, length) runs of a sorted-by-episode column."""
    codes = episode_ids.combine_chunks()
    if pa.types.is_dictionary(codes.type):
        indices = np.asarray(codes.indices)
        dictionary = codes.dictionary.to_pylist()
    else:
        encoded = codes.dictionary_encode()
        indices = np.asarray(encoded.indices)
        dictionary = encoded.dictionary.to_pylist()
    if indices.size == 0:
        return []
    change = np.flatnonzero(np.diff(indices)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [indices.size]))
    return [(dictionary[int(indices[s])], int(s), int(e - s)) for s, e in zip(starts, ends)]


def write_episode_csv(table: pa.Table, csv_path: Path) -> None:
    """Write one episode back to CSV with shortest round-trip float text."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    for f in table.schema:
        column = table.column(f.name)
        if pa.types.is_dictionary(f.type):
            column = column.cast(pa.string())
        columns.append(column)
    plain = pa.Table.from_arrays(columns, names=table.column_names)
    pacsv.write_csv(
        plain,
        csv_path,
        write_options=pacsv.WriteOptions(include_header=True, quoting_style="needed"),
    )


def rehydrate_dataset(
    spec: DatasetSpec,
    release_dir: Path,
    dest_datasets_dir: Path,
    log: Callable[[str], None] = print,
    max_episodes: int | None = None,
    only_shards: Iterable[str] | None = None,
) -> Path:
    """Rebuild the original per-episode CSV tree from the release files.

    ``release_dir`` holds ``raw/<release_name>/`` as downloaded from the Hub;
    the tree is written to ``dest_datasets_dir/<original raw name>/``."""
    raw_dir = release_dir / "raw" / spec.release_name
    bundle = raw_dir / "metadata.tar.gz"
    if not bundle.exists():
        raise FileNotFoundError(bundle)
    dest_datasets_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle, "r:gz") as tar:
        members = [m for m in tar.getmembers() if not m.name.startswith("plans/")]
        plans = [m for m in tar.getmembers() if m.name.startswith("plans/")]
        tar.extractall(dest_datasets_dir, members=members, filter="data")
        for member in plans:
            member.name = member.name[len("plans/"):]
        if plans:
            tar.extractall(dest_datasets_dir, members=plans, filter="data")
    dataset_root = dest_datasets_dir / spec.raw_name

    episodes_table = pq.read_table(raw_dir / "episodes.parquet", columns=["episode_id", "source_shard", "parquet_file"])
    shard_of = dict(zip(episodes_table.column("episode_id").to_pylist(), episodes_table.column("source_shard").to_pylist()))
    written = 0
    wanted = set(only_shards) if only_shards is not None else None
    for split_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir() and p.name in ("train", "val")):
        for parquet_path in sorted(split_dir.glob("*.parquet")):
            if wanted is not None and parquet_path.stem not in wanted:
                continue
            table = pq.read_table(parquet_path)
            for episode_id, start, length in episode_runs(table.column("episode_id")):
                shard = shard_of[episode_id]
                subdir = "" if not spec.sharded else shard
                csv_path = dataset_root / subdir / "episodes" / f"{episode_id}.csv"
                write_episode_csv(table.slice(start, length), csv_path)
                written += 1
                if max_episodes is not None and written >= max_episodes:
                    log(f"[{spec.release_name}] wrote {written} episodes (limit) -> {dataset_root}")
                    return dataset_root
    log(f"[{spec.release_name}] wrote {written} episode CSVs -> {dataset_root}")
    return dataset_root


# --------------------------------------------------------------------------- misc


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fp:
        while True:
            block = fp.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_manifest(staging_dir: Path) -> dict[str, Any]:
    path = staging_dir / "release_manifest.json"
    if path.exists():
        return load_json(path)
    return {"repo_id": HF_REPO_ID, "datasets": {}, "processed": {}, "files": {}}


def save_manifest(staging_dir: Path, manifest: dict[str, Any]) -> None:
    path = staging_dir / "release_manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(path)


def iter_release_files(staging_dir: Path) -> Iterable[Path]:
    for path in sorted(staging_dir.rglob("*")):
        if path.is_file() and ".cache" not in path.parts and not path.name.endswith(".tmp"):
            yield path
