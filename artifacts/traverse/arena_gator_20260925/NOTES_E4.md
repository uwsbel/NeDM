# NOTES E4: training files, subsets, and the first rigid training (2026-09-25, 02:25-03:00)

Module E4 of PLAN.md (section 4 E4/E5, amendments 7.5 and 7.6). K3 = this folder, G3 =
`/work1/dannegrut/harry/experiments/arena_gator_20260925`. New scripts: `scripts/ag_build_ds.py`, `scripts/ag_subset.py`,
`scripts/ag_train.sbatch`. No existing script was edited. Earlier artefact folders were only read.

## 1. Summary

- **Dataset pipeline** `scripts/ag_build_ds.py`: one call per (arena, vehicle, world or both worlds). It selects the
  episodes, checks them, then runs the three unchanged earlier tools on exactly those episodes: labels and corridors
  per episode, re-anchored rows, and the file the trainer reads. Arena, vehicle and tier columns are added.
- **f104 HMMWV rebuilt** from every local run:
  - rigid: 24,000 episodes, 93,397 rows;
  - soil: 15,235 episodes, 58,268 rows;
  - file `e4/f104_hmmwv/ci_f104_hmmwv_both.npz`, 151,665 rows, 2.75 GB.
- **Twin check passed.** Restricted to the 15,024 episodes that the earlier shared-model file used, the rebuild gives
  exactly its rows: 58,424 rigid + 57,444 soil. Every array is identical (corridors, context, history windows, labels,
  ids).
- **Subset tool** `scripts/ag_subset.py`: training groups per arena by the lowest md5 of the group id. It builds
  M1, M2, M3, A3, the leave-one-arena-out sets, the f104 learning curve and task B's H set. Val and test groups are
  kept whole.
- **Training on the cluster:**
  - The smoke run (job 436133) passed.
  - Job **436135** is training five 5-seed ensembles on one MI350X node: rigid M1 twice (seeds 0-4 and 5-9, deploy)
    and three holdout runs for the offline scores and the learning curve (1,089 / 545 / 272 groups).
  - Exact commands: `e5/README.md`.
  - No soil model was trained.

## 2. The dataset pipeline (`scripts/ag_build_ds.py`)

### 2.1 Selection and checks, before any tensor is built

- **Episodes.**
  - Id pattern `<prefix><arena>_v2_group_NNNN_route_NN|_op_NN`. The prefix is empty for the HMMWV and `gator__` for the
    Gator (`--id-prefix` overrides it, e.g. the pilot's `hmmwv__` rows).
  - This alone excludes drift rows, dev-arena rows, test and held-out groups and planner rows in the shared output
    folders.
  - With task files (`--tasks`, or per world `--tasks-rigid` / `--tasks-crm`), only their training rows are used: tier
    >= 0, the same arena, and the same vehicle.
  - Optional tier range and per-world exclusion lists (`--exclude-ids-crm <qa.json>` takes soil QA's flagged ids).
- **Per episode:**
  - every file the builders read is present;
  - the completion marker is present (required automatically when the folder's collector writes markers: all new
    runs; the older local f104 runs have none);
  - the launch check passed;
  - soil: `crm_qa.py`'s validity check, where its records exist;
  - rigid: finite state, action and pose arrays.
  - Rejections are counted by reason in the record. In-progress runs are simply not selected (missing files or no
    marker).
- **Map check** (`scripts/ag_map_check.py`, on every selected run's `case.json`): all runs name one arena, it is the
  requested arena, and its BMP sha256 equals the map root's `arena_bmp_sha256`. A wrong map stops the build (tested:
  f104 runs with the g203 map fail).
- **Vehicle check:** Gator runs must carry `vehicle.name == "gator"`, HMMWV runs no vehicle block (tested: HMMWV runs
  asked for as the Gator fail).
- **Id checks:** ids are unique across the run folders, every case is a training-pool group of the arena, and no suite
  id or group is present.

### 2.2 The three stages, each the unchanged earlier tool run as a subprocess

The tools read a folder of links to the selected episodes:

1. `f104_n2_dataset.py --root <map root>`: one row per episode. Every selected episode must be labelled.
2. `n2_reanchor_dataset.py`, default flags (4 anchors, 12 m minimum remaining; the flags behind the night-2 files, as
   the twin check confirms). Every episode must give its standing-start row.
3. The trainer file:
   - **Both worlds:** `ga_build_mixed.py`. Its exit status must be 0 and its build record must show no missing
     episode (it would otherwise still write the file and exit 2). Then the three columns are appended to its npz
     without rewriting it.
   - **One world:** the same arrays for that world alone (`ci_one_world`, which reuses `ga_build_mixed`'s own episode
     cutter, constants and invariants).
     - Check: run on the f104 rigid and soil re-anchored files, it equals the rigid and the soil half of the
       two-world file, every array (`e4/f104_hmmwv/thin_variant_check.json`).

### 2.3 Columns, final checks, record

- **Added columns:**
  - `arena` (e.g. `f104`);
  - `vehicle` (`hmmwv` / `gator`);
  - `tier` (int16): the task-file tier. Ids without a task row (the f104 pools, the Gator ids stripped of their
    prefix) get the per-group shuffled order of `crm_tasks.py` (`ag_tasklib.group_routes`). That order is also
    asserted equal to every task-file tier. f104 rigid `route_NN` / `op_NN` ids therefore carry the same tier as
    their soil twins.
- **Final checks on the written file:** unique ids; no id, group or episode matches `ga_build_mixed.BLACKLIST` or the
  generic suite patterns (`*_test_group_*`, `*_heldout_group_*`, `*_dev_group_*`, `*_eval_group_*`, `*_pair_group_*`,
  `drift__*`); one arena and vehicle.
- **Record** `<out>/<stem>_record.json`: selection counts, rejected ids, map check, tool sha256, stage times, row
  counts per world x split x start, tiers, output sha256.

### 2.4 Tested on cluster data

These are tests; their outputs are in `G3/e4/tests/` and are not training data.

- g203 soil (partial, 02:55): 298 episodes selected. 7 in-progress runs were left out. 1,115 rows. The soil task file
  was used and the marker required.
- The Gator pilot on f104 (288 rigid + 54 soil `gator__` runs): vehicle blocks checked, both-world file built through
  `ga_build_mixed.py`, 1,327 rows.

Cluster recipe (numpy environment of the collectors):

```bash
source /work1/dannegrut/harry/nrd/env.sh; nrd_pychrono
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925; cd $G3/source
export PYTHONPATH=$G3/source/src:$G3/source/scripts:${PYTHONPATH:-}
$NRD_PYTHON -u scripts/ag_build_ds.py --arena g203 --vehicle hmmwv --world rigid --rigid-runs $G3/rigid_v1/runs \
  --tasks $G3/tasks/rigid_hmmwv_v1.json --map-root $G3/map_roots/g203 --source-root $G3/source --out $G3/e4/g203_hmmwv --workers 16
# soil: --world crm --crm-runs $G3/soil_v1/runs --tasks $G3/tasks/soil_v1.json (or the extended soil_v2.json)
# Gator f104: --arena f104 --vehicle gator --map-root $G3/e4/map_roots/f104 (a copy of the crm_f104_v1 f104 map, staged by E4)
```

The cluster copy of `ag_build_ds.py` (sha256 `6a4da5a8...`, synced 02:48) predates two local additions
(`--id-prefix`; the per-world `--tasks-rigid` / `--tasks-crm`). Local file: `5e2eae03...`. I did not replace the
cluster copy, because jobs reading `G3/source` are running.

With that copy:
- build **one world per call** with `--tasks` (as above);
- `ag_subset.py` combines one-world files: one or more files per arena.

A two-world call with both task files needs the newer copy, because the rigid and soil rows share ids. Sync it when
no job reads the tree.

## 3. f104 HMMWV files (local, `e4/f104_hmmwv/`)

Command (331 s on 12 local cores; record `f104_hmmwv_both_record.json`, log `e4/f104_hmmwv/logs/ag_build_ds_f104_hmmwv_both.log`):

```bash
A=artifacts/traverse
PYTHONPATH=src:scripts python scripts/ag_build_ds.py --arena f104 --vehicle hmmwv --world both \
  --rigid-runs $A/fdm_f104_50h_20260909/production_v3/runs $A/fdm_f104_50h_20260909/production_v4/runs \
  --crm-runs $A/crm_f104_v1/collect_v1/runs --exclude-ids-crm $A/crm_f104_v1/collect_v1/qa.json \
  --map-root $A/crm_f104_v1/map_root --out $A/arena_gator_20260925/e4/f104_hmmwv --workers 12 \
  --compare-to $A/generalist_20260921/A_adapt/datasets/mixed_reanchor.npz
```

**Map root.** `crm_f104_v1/map_root` for both worlds.
- It is the root behind the night-2 re-anchored files of both worlds: recomputed rows match it with difference 0.
  The older `fdm_f104_50h_20260909/static_map_v1` capture differs by up to 0.002 in the float16 corridors.
- Its BMP hash equals the f104 arena's.

| world | episodes | groups (train / val / test) | rows (re-anchored) | per episode | outcomes |
|---|---|---|---|---|---|
| rigid | 24,000 (14,400 designed + 9,600 on-policy), tiers 0-19 x 1,200 | 1,089 / 56 / 55 | 93,397 (train 84,787, val 4,358, test 4,252) | 3.89 | goal 19,230, blockage 4,120, timeout 529, arena exit 85, rollover 36 |
| soil | 15,235 (9,168 designed + 6,067 on-policy) | 1,089 / 56 / 55 | 58,268 (train 52,923, val 2,708, test 2,637) | 3.82 | goal 4,867, breakthrough 8,372, blockage 1,985, timeout 9, rollover 2 |

Rows by world x split x start: rigid train 21,780 start + 63,007 moving, val 1,120 + 3,238, test 1,100 + 3,152;
soil train 13,821 + 39,102, val 713 + 1,995, test 701 + 1,936.

- Soil tiers per group: tiers 1-11 on all 1,200 groups. Tier 0 on 1,199: the one QA-flagged run,
  `f104_v2_group_0211_route_11`, is not in the local runs. Tier 12 on 836 groups.
- **Twin check** (`compare` block of the record). On the 15,024 twin episodes:
  - rows 58,424 rigid + 57,444 soil = `generalist_20260921/A_adapt/datasets/mixed_reanchor.npz`;
  - no id missing on either side;
  - every common array identical (`X, ctx, E, T, hist, hmask, privileged, fail, unsafe, event_idx, anchor_frame,
    ...`).
  - The rebuild adds 8,976 rigid episodes (the rest of the 24,000 pool) and the 211 soil episodes that had no rigid
    twin when the old file was made.
- sha256: `ci_f104_hmmwv_both.npz` `350d58f4...`, station rigid `0013ddd4...`, soil `da3a587a...`, re-anchored rigid
  `61b73560...`, soil `2c27d8ce...`. Synced to `G3/e4/f104_hmmwv/`.
- The record lists the builder's sha256 at the start of this build, `2b4b20fa...`. Two later edits did not touch the
  two-world path: the one-world function takes a list of run folders, and the options above were added.

## 4. Subsets (`scripts/ag_subset.py`, `e4/subsets/`)

**Rule.** Within each arena, rank the training groups that still have rows after the world, tier and id filters by
md5 of the group id (hex, ascending), and take the first N.
- The ranking depends only on the name, so all sets nest, and both worlds get the same groups when the same groups
  exist (checked).
- Val and test groups are kept whole. `--eval-rows filtered` applies the tier filter to them too.
- An id list (`--ids-file`, a leading `gator__` stripped) applies to every row by default (`--ids-apply all`), so H
  and G are scored on the same held-out routes.
- The tool stops on suite ids (tested with a planted `f104_pair_group` id), on missing arenas, and on requests larger
  than the available groups (`--allow-short` to take all).

**Presets:**

| preset | groups per arena | note |
|---|---|---|
| M1 | f104 1,089 | |
| M2 | f104 545, g203 545 | |
| M3 | 363 on each of f104, g203, g228 | |
| A3 | all on each | rigid: 1,089 + 1,083 + 1,063 |
| LOAO1_f104 / _g203 / _g228 | 545 on one arena | |
| LOAO2_f104_g203, LOAO2_f104_g228, LOAO2_g203_g228 | 273 + 272 (first named arena 273) | |
| LC272 / LC545 / LC1089 | f104 272 / 545 / 1,089 | holdout-mode files, see below |
| H | f104, all groups | with `--ids-file` = the validated Gator ids |

All presets were exercised on synthetic three-arena files (f104's columns renamed to g203 / g228):
`e4/subset_mechanics_test.py`, results in `e4/subset_mechanics_test.json` (e.g. M3 fits 288 / 296 / 292 groups in
holdout mode; the sets nest; M3 picks the same groups in both worlds; H keeps only listed episodes). The one-world
check script is `e4/thin_variant_test.py`.

**Written now (rigid, f104; sha256 equal locally and in `G3/e4/subsets/`):**

| file | rows | fitted, deploy | fitted, holdout | held-out rows | sha256 |
|---|---|---|---|---|---|
| `M1_f104_hmmwv_rigid.npz` | 93,397 | 84,787 rows / 1,089 groups | 67,368 / 865 | val 4,358 (56 groups) | `3174999e...` |
| `LC545_f104_hmmwv_rigid.npz` | 59,606 | (holdout only) | 33,577 / 431 | dev fold 17,419 (224 groups) + val 4,358 | `2ab94685...` |
| `LC272_f104_hmmwv_rigid.npz` | 42,641 | (holdout only) | 16,612 / 213 | the same | `4d10dced...` |

**The learning-curve files keep all 224 dev-fold groups** (the training groups ga_train holds out in holdout mode,
md5 % 5 == 0).
- So every point of the curve is scored on the same rows. The top point is the M1 file itself.
- Fitted groups are about 80 % of the nominal count, because holdout mode leaves the dev fold out.
- Those two files must not be used in deploy mode, which would fit the dev fold. Their manifests say so.

## 5. Training (details and commands in `e5/README.md`)

- **Local smoke.** 1 epoch, LC272 holdout, on the 5090. Checkpoint round trip exact.
- **Cluster smoke, job 436133** (mi3501x, 49 s). M1 deploy 1 epoch with round trip, and LC272 holdout 1 epoch. Both
  exit 0.
- **Job 436135** (mi3501x, 3 h 50 min limit, one node, five runs at once), submitted 02:52:
  - M1a / M1b rigid deploy, seeds 0-4 / 5-9;
  - M1 rigid holdout, seeds 0-4;
  - LC545 and LC272 rigid holdout, seeds 0-4.
  - Recipe: `--arch gru --cond none --domain-filter rigid --ctx geom --epochs 30 --bs 256 --split-eval val`, trainer
    default learning rate and weight decay (= deploy_a1).
  - Status at the time of writing: in section 7.
- Queue discipline: at 02:49 we had 44 tasks queued, at 02:52 43. Each submission added one task.

## 6. Problems and things the next modules need

1. **The soil tier range is not decided.**
   - f104 soil has tiers 0-11 complete (tier 0 minus the one flagged run) and tier 12 on 836 of 1,200 groups.
   - Matched soil designs cut f104 to the new arenas' last complete tier: `--tiers 0-K`, applied to training rows only
     by default.
2. **g228 soil has only 520 training groups** in its 606 collected groups (E1 caveat 1).
   - Soil LOAO1_g228 (545) fails without `--allow-short`.
   - A3's g228 share is 520 unless the 28 extra groups are collected.
3. **Rigid test-arena rows cannot be scored through the trainer as they are.**
   - This concerns the 12,000 designed routes on the test arenas.
   - Their cases carry hash splits (mostly `train`) and their groups are blacklisted, so the trainer would refuse them
     as fit rows.
   - Offline scores on unseen arenas need either a scorer that loads the checkpoints (`ci_train.load_ci_model` +
     `score`) or evaluation-only files with the split set to `val`. Neither exists yet (E5/E6).
4. **Rows grow with drive length** (REVIEW_R1 11): rigid 3.89 and soil 3.82 rows per episode on f104. The subset
   manifests report rows per arena.
5. **Gator rigid runs have 24 NaN suspension telemetry fields.** The builders read only the 17 state columns, the
   actions and pose (finite, checked), so the Gator pilot build went through.
6. **Two cluster copies are behind the local ones.** `ag_build_ds.py` on the cluster lacks `--id-prefix` and
   `--tasks-rigid/--tasks-crm` (section 2.4). `ag_subset.py` and `ag_train.sbatch` are identical to the local files.
7. **The f104 map root on the cluster** is staged at `G3/e4/map_roots/f104`. It is a byte copy of
   `crm_f104_v1/maps/arena_f104_50h_v1` (observation sha256 `53e23f99...`). `G3/map_roots` has no f104 entry.

## 7. Status of job 436135 at hand-off (03:02)

- Running on k007-005-v8 since 02:52. The GPU is at 100 % use, 13 % of its memory is in use, and each of the five
  processes holds about 2.8 GB of host memory.
- First finished seed: LC272 holdout seed 0 at 03:00, 1,920 steps in 444 s (0.23 s per step with five runs sharing
  the GPU; one run alone does about 0.06 s per step).
- Expected total: about 168,000 steps at about 20 steps per second for the whole node, so about 2.3 h (end about
  05:15). The limit is 3 h 50 min (06:42).
- Every seed is saved as it finishes, so a timeout would keep the finished seeds.
- Cost: one mi3501x node for about 2.5 h (0.3 billed at S3's 0.14 billed per 65 min).
- When it ends:
  - check `grep -h "exit:" $G3/e5/logs/*_436135.log` (5 x `exit: 0`) and `sacct -j 436135`;
  - fetch with `rsync -a amd:$G3/e5/train/ $K3/e5/train/`;
  - the offline scores are in `<out>/<tag>.json` under `ensemble.dev` / `ensemble.heldout`: within-group AUC of
    unsafe, `W_unsafe`, per start/moving row.
  - The learning curve = LC272 / LC545 / M1 holdout, on identical dev-fold + val rows.
