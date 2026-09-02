# kyle-sbel

**Verified:** 2026-09-02 · **Owner:** Kyle · **Role:** Development, docs,
manuscript builds, light training and analysis.

| | |
|---|---|
| GPU | NVIDIA RTX 3090, 24 GB |
| CPU / RAM | 16 cores / 30 GB |
| Disk (free) | 1.8 T total, **1.3 T free** |
| Repo path | `/home/kyle/Documents/sbel/NeDM` |
| Interpreter | `/home/kyle/miniconda3/envs/chrono/bin/python` (pychrono + torch 2.6.0+cu124, CUDA available) |
| Reachable from | local only |

```bash
export NEDM_ROOT=/home/kyle/Documents/sbel/NeDM
export NEDM_PY=/home/kyle/miniconda3/envs/chrono/bin/python
export PYTHONPATH=$NEDM_ROOT/src
```

## What this machine is for

Reading and writing code, docs, and the manuscript; small training runs and
analysis; driving work on other boxes. Sibling repos live alongside it under
`/home/kyle/Documents/sbel/` — `Manuscripts`, `chrono_fork`, `chrono_hil`,
`ccta`, `sbel-reproducibility`.

## What it must not do

No raw dataset collection (see [`newton.md`](newton.md)). It has the disk for it
but not the role; collection belongs where the pipeline already runs.

## Gotchas

1. **`git-lfs` is not installed.** Checkpoint `.pt` files under `artifacts/` are
   LFS pointer stubs, not weights. Install `git-lfs`, then
   `git lfs install && git lfs pull`, before expecting any checkpoint to load.
2. **The conda env is named `chrono`, not `nedm`.** Every other doc in this repo
   says `nedm` (that is Harry's env name). Same role, different name — do not
   "fix" scripts to hardcode either one; use `$NEDM_PY`.
3. **`newton` is not resolvable from here** (`Temporary failure in name
   resolution`, 2026-09-02). No SSH host entry and/or not on the network. Work
   destined for newton must be pushed to GitHub and pulled there by someone with
   access.
4. Local Chrono checkout is at `/home/kyle/Documents/sbel/chrono_fork/chrono`,
   which is the source tree, not the pychrono package the env uses.
