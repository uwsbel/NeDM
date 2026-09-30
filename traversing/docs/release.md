# The data and model release

The data and trained models of the traversing study are in the `traversing/` folder of the paper's Hugging Face
dataset [harryzhang1018/NeDM](https://huggingface.co/datasets/harryzhang1018/NeDM), at revision `6620faead5225ac9aa5ae8ab19bc2ef2db38a863`.
The paper's files in that repository are unchanged; the tag `paper-v1` pins them. The dataset card at the repository
root describes the paper's release in Part A and this one in Part B. The card was added one commit after the pinned
data revision, in [`f44f87c`](https://huggingface.co/datasets/harryzhang1018/NeDM/blob/f44f87c232a1df8ce86139d1121789f7372bfbda/README.md), which changes only `README.md`; at the pinned revision itself the root
card is still the paper's.

This page says how the release was built, how to verify it, where each item restores to, and how to rebuild it.
[evidence.md](evidence.md) names the item behind each cited record, and the [README](../README.md#download) gives the
download commands.

## Layout

```text
traversing/README.md                  index of items: bundle, milestones, files, bytes, restore folder, description
traversing/release_manifest.json      every file (bytes, SHA256, item) and every item
traversing/<bundle>/<item>/...        bundle: models, evaluation, processed, raw or assets
```

Each item records its bundle, the milestones it serves, a description, what it is needed for (`recount`: recount a
compact table from its per-drive records; `rerun`: drive a table arm again; `retrain`: train a model again), its
restore folder and its source. `download_traversing_data.py --list --all` prints each item's bundle, packing, file
count, size, restore folder and milestones; the descriptions are in the manifest and in `traversing/README.md` on the
Hub. An item is
packed one of two ways:

- **Plain files.** Each source file is uploaded as it is, at `traversing/<bundle>/<item>/<path>`, where `<path>` is
  its path under the item's source folder. Models, single large training files and small sets of evaluation files
  are packed this way.
- **Tar shards.** Many small files (drive folders, route picks, suites) go into gzip tar shards
  `part-00000.tar.gz`, `part-00001.tar.gz`, ... of about 1 GiB of file data each. Each member is stored under its
  path relative to the repository root, so extracting at the root restores it. `index.csv.gz` lists every member with
  its bytes, SHA256 and shard. Drive collections also carry `ids.txt.gz`, their drive list, and the raw drive
  collections `episodes.csv.gz`: one row per drive with its vehicle, ground, arena, end status, simulated time, file
  count and the SHA256 of its trajectory and outcome files.

Every released file is an exact copy of a study file, or a tar of exact copies; nothing was re-encoded. The `.npz`
files hold object arrays, so load them with `numpy.load(path, allow_pickle=True)`.

Reading an `episodes.csv.gz` table (one row per drive of a raw collection):

```python
from datasets import load_dataset
eps = load_dataset("csv", split="train", data_files="hf://datasets/harryzhang1018/NeDM@6620faead5225ac9aa5ae8ab19bc2ef2db38a863/"
                   "traversing/raw/hmmwv_soil_f104_collection/episodes.csv.gz")
# or without the datasets library:
import csv, gzip, io
from huggingface_hub import HfFileSystem
path = ("datasets/harryzhang1018/NeDM@6620faead5225ac9aa5ae8ab19bc2ef2db38a863/"
        "traversing/raw/hmmwv_soil_f104_collection/episodes.csv.gz")
with HfFileSystem().open(path, "rb") as f:
    rows = list(csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=f))))
```

## How it was built

**Tools.** `build_release.py` packed the items described by `release_spec.json`. The spec lists, for each item, its
source folder, which files to take (glob patterns, a file list, or a list of drive folders with the files to take
inside each; the lists are in `id_lists/`), where it restores to, and the checksums and parameter counts known before
the build. The build uses the Python standard library only. These build inputs are needed only to rebuild the release
from the original study folders, so they are not on main: they are kept under the tag
[`traversing-release-build-v1`](https://github.com/uwsbel/NeDM/tree/traversing-release-build-v1/traversing) (commit `e37bac0`), in `scripts/release/build_release.py` and
`manifests/`. Each item's drive-folder list is also on the Hub as `ids.txt.gz` in the item's folder (same lines).

**Scope.** The items come from an inventory made on 2026-09-29 of what reproduces the milestone results:

- the per-drive evaluation records behind every compact table;
- every model behind a table column, plus the Gator rigid planner, and the evaluation inputs needed to drive every
  table arm again exactly;
- the exact training files of those models, and the raw collection drives they were built from, limited to the files
  the dataset builders read.

**Where each item came from.** Workstation items were taken from the experiment checkout at commit
[`901d6c9`][commit], including the study folders that are not in git. Cluster items were taken from the study
folders on the AMD HPC Fund cluster. Where an item existed in both places and the workstation copy was complete, the
workstation copy was used; the cluster was used where the workstation copy was missing or incomplete, which is the
case for most raw collections, the navigation model's training files and most unseen-arena and vehicle training
subsets. Each machine packed and uploaded its own items, so the cluster shards never passed through the workstation.

| Bundle | Workstation items | Workstation size | Cluster items | Cluster size |
|---|---|---|---|---|
| models | 30 | 206 MB | – | – |
| evaluation | 32 | 5.6 GB | 1 | 148 MB |
| processed | 9 | 11.8 GB | 19 | 22.0 GB |
| raw | 4 | 809 MB | 21 | 16.0 GB |
| assets | 3 | 8.1 MB | – | – |
| all | 78 | 18.5 GB | 41 | 38.2 GB |

Sizes are download sizes. Each item's `source` in the manifest names the machine, the source root and the folder.

**Deterministic shards.** Members are sorted by path, and a drive folder never straddles two shards. Every member is
stored as a regular file with modification time 0, owner and group 0, empty owner names and mode 0644, in PAX format,
with no folder entries. The gzip layer uses level 6, modification time 0 and no file name. The compressed bytes still
depend on the zlib version (the cluster and the workstation differ), so the manifest also records the size and
SHA256 of each shard's uncompressed tar stream, which a rebuild on any machine must reproduce.

**Checks during the build.** The build stops on:

- a pattern that selects nothing, a missing listed file or a missing required file in any drive folder;
- a drive count other than the spec's;
- a symlink or unreadable folder in a selection;
- an unsafe path;
- a file that changes while it is packed;
- any checksum that differs from the spec. The spec holds the SHA256 of every local-only record cited in
  [evidence.md](evidence.md) and [`results/README.md`](../results/README.md), every mission outcome of the
  milestone-1 table, and the checksums of record of the models and training files.

Each item's build record stores a fingerprint of its spec entry, its list and the build script. The merge step
refuses records built from anything else, and refuses two items that restore the same path. It then writes
`release_manifest.json` and the Hub index.

Upload. Each machine uploaded its own items with `huggingface_hub`'s `upload_large_folder`: the workstation items
from the workstation, the cluster items from inside the cluster batch job that built them (job 443887), authenticated
with a Hub write token kept privately in the cluster home directory. The manifest and the Hub index went up last, in
Hub commit `6620fae`, after every item's upload had been checked against the Hub. `build_release.py pin` then
confirmed that the Hub's manifest at that commit is byte-identical to the staged one, and wrote the GitHub copy
[`manifests/hf_release_manifest.json`](../manifests/hf_release_manifest.json): the same manifest plus `hf_revision`,
the Hub commit. The download and verification helpers default to that commit. The two-part dataset card followed in
its own commit, `f44f87c`. Revoke or rotate the write token on the cluster once it is no longer needed.

## How to verify it

```bash
python traversing/scripts/release/verify_release.py --hub                                     # the Hub copy
python traversing/scripts/release/verify_release.py --local artifacts/hf_release/download     # downloaded files
python traversing/scripts/release/download_traversing_data.py --bundle models --verify-members # download anything missing, then re-hash the restored files
```

- **`--hub`** (needs `huggingface_hub`) checks, at the pinned revision:
  - every manifest file is on the Hub with the recorded size and SHA256;
  - the Hub's own manifest lists the same files and items as the pinned copy;
  - nothing else is under `traversing/`;
  - every file outside `traversing/` except the root `README.md` is unchanged from the paper release (commit
    `8091c3b4c9ac5dd270fd6a85eb95afe77652c6f2`, which `paper-v1` pins); `.gitattributes` may differ only by
    large-file rules for `traversing/` paths that the Hub appended after the paper's lines (9 at this revision);
  - the paper's `load_dataset` configurations in the card are unchanged, and any added one reads only `traversing/`
    files. The card adds none: the `datasets` library reads every configuration of a repository with one file format
    (the paper's Parquet), so the `episodes.csv.gz` tables are loaded with its CSV reader instead (see Layout).
  - `--hub --revision f44f87c232a1df8ce86139d1121789f7372bfbda` runs the same checks on the commit with the live card.
- **`--local DIR`** checks a downloaded or staged tree:
  - the tree's manifest lists the same files and items as the pinned copy;
  - every file present has the recorded size and SHA256, and every shard the recorded tar stream (`--complete` also
    requires every file to be present);
  - no two items restore the same path;
  - every local-only record cited with a SHA256 in [evidence.md](evidence.md) and
    [`results/README.md`](../results/README.md), including the hash columns of the result tables, is in the release at
    the cited path with that SHA256, and every other hash in those two documents is accounted for;
  - on a partial download, a cited record inside an item whose index was not downloaded is reported as not checked
    rather than failed (`--complete` requires everything);
  - every model checkpoint loads with torch in its safe mode (`weights_only=True`) and has the parameter count the
    manifest records. Without torch every checkpoint is reported unverified and the check fails; `--no-models` skips
    it.
- **`--verify-members`** on a download re-hashes every restored file against the manifest and the item's index.

Both helpers exit non-zero on any failure.

## Restore layout

Each item has a restore folder relative to the repository root. Plain files go to `<restore folder>/<path>`; tar
members already carry their full path. Items from the workstation keep their path in the experiment checkout. Items
from the cluster restore into the local study folder of the same name, keeping their path inside it:

| Cluster study folder | Restores to |
|---|---|
| `/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909` | `artifacts/traverse/fdm_f104_50h_20260909` |
| `/work1/dannegrut/harry/experiments/crm_f104_20260916` | `artifacts/traverse/crm_f104_v1` |
| `/work1/dannegrut/harry/experiments/generalist_20260921` | `artifacts/traverse/generalist_20260921` |
| `/work1/dannegrut/harry/experiments/crm_improve_20260922` | `artifacts/traverse/crm_improve_20260922` |
| `/work1/dannegrut/harry/experiments/arena_gator_20260925` | `artifacts/traverse/arena_gator_20260925` |
| `/work1/dannegrut/harry/experiments/offroad_vehicles_20260927` | `artifacts/traverse/offroad_vehicles_20260927` |

For example, the Gator soil drive `arena_gator_20260925/soil_v1/runs/<drive>/trajectory.npz` on the cluster restores
to `artifacts/traverse/arena_gator_20260925/soil_v1/runs/<drive>/trajectory.npz`.

No two items restore the same path. Files that two tables use are released once, and an item that relies on another
item's files lists it in `requires`, which the download helper adds to the selection (for example, the vehicle f104
drives require the unseen-arena soil drives, which hold the HMMWV f104 drives both tables use). This is done only for
evaluation-size dependencies: the smoke test's stock Polaris and stored HMMWV columns count drives of the Polaris and
HMMWV soil collections, which the smoke-test item names in its description instead (about 3.1 GB to download,
3.7 GB restored).

The helper writes each file through a new temporary file and never writes through a symlink or outside `--dest`. It
keeps existing files with other content unless `--overwrite` is given. The `assets` items restore into the tracked
folders `assets/traverse/` and `configs/`, so after restoring them into a checkout they show up as untracked files in
`git status`; do not commit them.

## Paths inside released files

Task lists and pick records name cases, routes, maps and models by the paths they had when the studies ran: cluster
paths, or paths relative to a study folder. `path_remap` in the manifest says how to turn each into a restore path:

1. Absolute paths are used as they are. A relative path starting with `artifacts/` or `assets/` is relative to the
   repository root; any other relative path is relative to the folder the task file was written against
   (`relative_roots`, by task-file pattern).
2. The longest matching entry of `prefixes` replaces the start of the path, for example
   `/work1/dannegrut/harry/experiments/arena_gator_20260925/` with `artifacts/traverse/arena_gator_20260925/`.
3. If the result is not a released file, it may be a renamed copy of a released route pick; `copies` gives the
   pattern, the pick it copies and the command that made it.

Every path in every released task file was resolved this way on 2026-09-29. What does not resolve is either a
collection input that is not released (training cases and route pools of the collections; each released drive keeps
its own `case.json`), a row of an arm that is not a table column, or the Gator runtime fingerprint, a record of the
cluster's Chrono build that the Gator rigid drives check. That fingerprint is machine-specific: write one for the
machine that drives, with `scripts/ag_runtime_fingerprint.py` at the experiment commit.

### Links to recreate

The release contains no symlinks, but the planners, the dataset builders and
[`results/README.md`](../results/README.md) go through some. `path_remap.links` lists them: the 12 planner map
roots, the cluster's copy of the f104 map root, and the tracker's by-arm views of its drive folders, where `<route>`
stands for each of the 423 route ids of `tracker_drive_folders`. Each link has a path and a target relative to the
link's folder. After a restore, recreate them from the repository root with:

```bash
python - <<'EOF'
import json, os
from pathlib import Path
links = json.load(open("traversing/manifests/hf_release_manifest.json"))["path_remap"]["links"]
runs = Path("artifacts/traverse/generalist_20260921/B_tracker/b0/rigid_runs")
routes = sorted({p.name.rsplit("__", 1)[0] for p in runs.glob("*__native_pid")})
for link in links:
    for route in routes if "<route>" in link["path"] else [""]:
        path = Path(link["path"].replace("<route>", route))
        if not path.is_symlink() and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(link["target"].replace("<route>", route), path)
EOF
```

The tracker views need `tracker_drive_folders` restored first; the map-root links need the planner map items.

## What is not released

- Arms that are not columns of the compact tables: their route picks and drives, and the round-1 learned tracker.
- Renamed copies of route picks and the run indexes that point at them; `path_remap.copies` recreates the copies.
- Superseded runs, such as the navigation runs before the rescue-route fix, pilot drives and exploratory campaigns.
- Intermediate training files that the dataset builders recreate from released files, and training subsets that
  no table column uses.
- Training snapshots: intermediate PPO iterations, the NRD's last checkpoint, hold-out variants of the ensembles and
  superseded ensembles.
- Collection inputs beyond the drives themselves: the training case sets and route pools of the arena and vehicle
  collections.
- Drives no released model was trained on: the g203 and g228 soil drives beyond the first 7 routes of each group, the
  rigid drives on the held-out arenas g216 and g231, and the continuation drives from the 3 s decision states.
- Files inside raw drive folders that no dataset builder reads, such as per-drive provenance and request records.

## Rebuild from the spec

A rebuild needs the original sources: the experiment checkout at [`901d6c9`][commit] with its untracked study
folders, and the study folders on the cluster. Check out the tag `traversing-release-build-v1`, which holds the
builder, the spec and the lists. The release was built with these commands (staging folders are examples):

```bash
# workstation items
python traversing/scripts/release/build_release.py build --spec traversing/manifests/release_spec.json \
    --source-repo ~/NeDM-traverse_mppi --items <workstation items> --staging /data/trv_staging --jobs 12 --link
# cluster items, with the builder, spec and lists copied to the cluster (python3.12)
python3.12 build_release.py build --spec manifests/release_spec.json \
    --items <cluster items> --staging $WORK/trv_staging --jobs 16
# after copying the cluster's _fragments/ records to the workstation staging folder
python traversing/scripts/release/build_release.py merge --staging /data/trv_staging \
    --spec traversing/manifests/release_spec.json
# upload: each machine with huggingface_hub upload_large_folder(repo_id="harryzhang1018/NeDM", repo_type="dataset",
#   folder_path=<staging>, allow_patterns=["traversing/**"], ignore_patterns=[the manifest and traversing/README.md]);
#   then traversing/release_manifest.json and traversing/README.md from the merged staging folder in one commit
python traversing/scripts/release/build_release.py pin --staging /data/trv_staging --revision <Hub commit>
```

`build --dry-run` resolves and counts an item without writing anything, and `--root KEY=PATH` points a source root
somewhere else. To compare a rebuild with the release, compare its `release_manifest.json` with the pinned copy:
every plain file's SHA256 and every shard's tar stream SHA256 must match.

[commit]: https://github.com/uwsbel/NeDM/tree/901d6c9423a16c0fafc3d60056065415d5a725f2
