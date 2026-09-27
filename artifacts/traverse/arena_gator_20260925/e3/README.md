# E3: soil and rigid collections of arena_gator_20260925 (HMMWV on the new training arenas, Gator on f104; 2026-09-25)

Launched 01:53-01:58 CDT (E3a: HMMWV soil_v1, rigid_hmmwv_v1), relaunched 04:47 (E3b2: soil_v2 with the Gator rows and
the spread headroom rows; rigid_v2 with the Gator pool and the spread designed rows run inside the soil allocations).
Cluster root `G3=/work1/dannegrut/harry/experiments/arena_gator_20260925`; local root
`K3=artifacts/traverse/arena_gator_20260925`. Every command below is run on the cluster login node (`ssh amd`) unless
it says "local". All submissions and job steps are also listed in `$G3/e3/submissions.tsv`.

## 0. Launch map (E3b2, updated 05:10) - read this first

**Soil** (all jobs write `$G3/soil_v1`; claims are per id, so jobs on either task file never repeat an episode).

| job (raw ids) | partition | tasks | task file | collector | state / claims stop |
|---|---|---|---|---|---|
| 436080 (436081-436091, 436116, 436147-436152) | mi2101x, 1 MI210 each, `-c 16 -t 12h` | 0-17 running; 18-19 cancelled 04:47 | `soil_v1.json` (16,066 rows, HMMWV only) | `crm_collect.py` (unchanged) | running; claims stop 13:41 (tasks 0-10), 14:09 (11), 14:52-15:03 (12-17) |
| 436092 (436092-436095) | mi3501x, `-t 4h` | 0-3 | `soil_v1.json` | `crm_collect.py` | claims stop 05:41, ends by 05:58 |
| **436207** (436207, 436209) | mi3501x, `-c 24 -t 4h` | 0-1 | **`soil_v2.json`** (31,901 rows) | **`ag_crm_collect.py`** (frozen, `b52e1fa6...`) | running since 04:47; claims stop 08:30 |
| **436212** | mi3501x, `-t 4h`, `--dependency=aftercorr:436092` | 0-1 | `soil_v2.json` | `ag_crm_collect.py` | starts when 436092_0 / _1 end (~05:58); claims stop ~09:41 |
| **436208** | mi2101x, `-t 12h` | 0-2 | `soil_v2.json` | `ag_crm_collect.py` | pending (Priority) |
| **436210** | mi2104x (4 MI210), `-c 128 -t 12h` | 0-2 | `soil_v2.json` | `ag_crm_collect.py` | pending (behind 436075_[12-13] and two other users' jobs) |
| **436215** | mi2104x, `-c 128 -t 12h` | 0-3 | `soil_v2.json` | `ag_crm_collect.py` | pending |
| 436213 | devel, 30 min | 0 | `soil_v2_check.json` (15 rows of soil_v2) | `ag_crm_collect.py` | first-episode check, ends ~05:18 |

Order in which the soil_v2 jobs work: tier -2 (3 bitid rows) -> tier -1 (597 spread headroom rows; the dev and drift
rows are done) -> tier 0 (Gator 1,199 + g203/g228 leftovers) -> tier 1 -> ... -> tier 12. The soil_v1 jobs keep
driving g203/g228 rows only (they read their file once at start), so the HMMWV tiers run ahead of the Gator tiers
until the soil_v1 jobs end; new capacity always goes to the lowest open tier, i.e. mostly to the Gator.

**Rigid** (all write `$G3/rigid_v1`).

| what | where | rows | collector | state |
|---|---|---|---|---|
| 436075 shards 0-13 | mi2104x, 126 processes per node | `rigid_hmmwv_v1.json` (g203/g228 pools, near test designed, f104 drift) | `gen_collect_ext.py` | 05:10: 35,213 of 60,200 rows complete, shards 6-11 running, 12-13 waiting for nodes (about 1 h per shard) |
| pool steps, shards 1000-1299 then 2000-2124 | `srun --overlap` inside the soil allocations (20 steps at 05:10: 12 workers in each of the 18 436080 tasks, 16 in each 436207 task, plus a 4-worker step that ran spread shard 2000 first) | `rigid_v2.json`: Gator pool 24,000 (1000s), spread designed 12,000 (2000s) | 1000s `ag_gen_collect_ext.py --vehicle gator` + Gator fingerprint; 2000s `G3/r2/source/scripts/gen_collect_ext.py` + HMMWV fingerprint | started 04:48 (test) / 04:59; 05:09: 8 shards done, 1,531 Gator + 40 spread runs complete, 0 incomplete; about 11,200 routes per hour; Gator done ~07:15, spread ~08:15 |

A bash loop on the login node **login1** (`$G3/source/scripts/ag_rigid_pool_autostep.sh`, log
`$G3/rigid_v1/pool/logs/autostep.out`) adds a pool step to every running soil job of this effort that has none and more
than 60 min left (12 workers on mi2101x, 16 on mi3501x, 96 on mi2104x), every 5 min, and exits when all 425 shards are
done or `$G3/rigid_v1/pool/STOP` exists. The `srun` clients of the steps also live on login1 (`pgrep -fc "srun --jobid"`).
Restart it after a login-node reboot: `setsid nohup bash $G3/source/scripts/ag_rigid_pool_autostep.sh >>
$G3/rigid_v1/pool/logs/autostep.out 2>&1 < /dev/null &`.

### Watching

```bash
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
squeue -u harry -h -r | wc -l                               # keep <= 47 (cap 50, 3 free)
python3 $G3/source/scripts/ag_collect_status.py --tasks $G3/tasks/soil_v2.json --out $G3/soil_v1   # per kind/arena + tiers
ls $G3/soil_v1/failed | wc -l; ls $G3/soil_v1/runs/*/collection_failure.json 2>/dev/null | wc -l
grep -l "3 consecutive failures\|Traceback" $G3/soil_v1/logs/*.out                              # retired soil workers
ls $G3/rigid_v1/pool/done | wc -l                            # rigid shards finished (425 in all: 300 Gator + 125 spread)
cat $G3/rigid_v1/pool/done/*.json | grep -c '"incomplete": \[\]'                                 # shards with every row complete
squeue -u harry -s | grep " bash"                            # running pool steps
tail -n 3 $G3/rigid_v1/pool/logs/step_*.out                  # one line per finished shard
python3 $G3/source/scripts/ag_overlap_rate.py --out $G3/soil_v1 --hosts <node> --t0 <HH:MM>      # soil speed on a node before/after a step
```

Local first-episode check (syncs a sample, then `ag_e3b2_verify.py`: status mix, launch checks, vehicle block present
in Gator runs and absent in HMMWV runs, bitid arrays):

```bash
# local
PYTHONPATH=src:scripts /home/harry/miniconda3/envs/nedm/bin/python scripts/ag_e3b2_verify.py \
  --soil-runs /tmp/ag_e3b2/soil_new --soil-refs /tmp/ag_e3b2/soil_refs --rigid-runs /tmp/ag_e3b2/rigid_new --out /tmp/v.json
```

### Stop commands

- **Soil, all jobs** (old and new; every worker finishes its episode, then exits): `touch $G3/soil_v1/STOP_CLAIMS`;
  delete it before any relaunch. For the PLAN 7.8 cut ("the tier in progress completes after ~12:00"): watch the tier
  line of `ag_collect_status.py`, touch STOP_CLAIMS when the tier in progress is complete for the arena/vehicle pair
  that matters, then `scancel` the soil jobs once their episodes have ended (< ~7 min).
- **Only the new launch:** `scancel 436207 436208 436210 436212 436215` (a killed episode is re-taken by another
  worker after 25 min and re-run from the start). There is no per-job soft stop; the soil_v1 jobs cannot be stopped
  alone either except by `scancel 436080`.
- **Rigid pool, soft:** `touch $G3/rigid_v1/pool/STOP` (each step finishes its shard, then exits); `rm` it to allow
  new steps. **Hard:** `scancel <raw job id>.<step id>` for one step (see `squeue -u harry -s`); its shard is taken
  over by another step after 15 min, the partial run folders are moved to `$G3/rigid_v1/pool/moved`. Cancelling a soil
  job also ends its pool step.
- **Drop the Gator rigid rows only** (if the Gator rigid GO is withdrawn): the steps skip any shard that has a done
  record, so mark every Gator shard that nobody has claimed yet (shards already running finish):
  `cd $G3/rigid_v1/pool; for s in $(seq 1000 1299); do [ -e claims/$s ] || [ -e done/$s.json ] || echo '{"skipped": "Gator rigid withdrawn"}' > done/$s.json; done`
  (`STOP` would stop the spread shards too; the helper counts these records as done).
- **Rigid HMMWV v1:** `scancel 436075`.

### Relaunch / more capacity

```bash
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
# more soil workers on soil_v2 (same folder; <part>:<cpus>:<hours>:<n>; refuses above cap-5 unless AG_QUEUE_RESERVE is lowered):
env -u NEDM_VEHICLE bash $G3/source/scripts/ag_soil_launch.sh $G3/tasks/soil_v2.json $G3/soil_v1 \
  $G3/source/scripts/ag_crm_collect.py mi3501x:24:4:2          # e.g. when 436207 stops claiming at 08:30
# rigid pool steps in soil allocations that have none (12 workers on mi2101x, 16 on mi3501x, 100 on mi2104x):
squeue -u harry -h -t R -o "%i %A %P %N %e"                  # raw job ids (%A)
bash $G3/source/scripts/ag_rigid_pool_launch.sh 16 <raw ids of mi3501x soil tasks>
bash $G3/source/scripts/ag_rigid_pool_launch.sh 100 <raw ids of mi2104x soil tasks>
```

A soil_v3 (e.g. evaluation drives) follows the same recipe: build it as a superset of soil_v2 (`--check-superset`),
new path, read-only, same output folder, cancel only pending tasks. Never edit a task file that a job has read.

Sections 1-7 below are the E3a record (01:53-02:25); section 6 (the relaunch) was carried out by E3b2 as described in
section 0 and in `$K3/NOTES_E3b2.md`.

Launched 01:53-01:58 CDT. Cluster root `G3=/work1/dannegrut/harry/experiments/arena_gator_20260925`; local root
`K3=artifacts/traverse/arena_gator_20260925`. Every command below is run on the cluster login node (`ssh amd`) unless
it says "local". All submissions are also listed in `$G3/e3/submissions.tsv`.

## 1. What was running at 02:25 (E3a; see section 0 for the current map)

| job | partition | array | what | task file (sha256) | output |
|---|---|---|---|---|---|
| **436080** | mi2101x (1 MI210 each), `-c 16 -t 12:00:00` | 0-19 (11 started at once, 9 waiting for nodes) | soil, one worker per GPU | `$G3/tasks/soil_v1.json` (`db0ba97f4b763ae6...`) | `$G3/soil_v1` |
| **436092** | mi3501x (1 MI350X each), `-c 24 -t 04:00:00` | 0-3 | soil, same file and folder | same | `$G3/soil_v1` |
| **436075** | mi2104x (128 cores), `-c 128 -t 03:00:00` | 0-13 | rigid, 126 collector processes per node | `$G3/tasks/rigid_hmmwv_v1.json` (`168044497dd338de...`) | `$G3/rigid_v1` |
| 436073 / 436074 | devel | 0 | rigid / soil smokes (done, all 6 episodes complete) | `$G3/e3/smoke_{rigid,soil}.json` | `$G3/e3/smoke_{rigid,soil}` |

- The soil collector is the **unchanged** `crm_collect.py` (sha256 `cb6792be...`, the same as in the crm_improve and
  generalist source trees), with `configs/crm_main.json` (sha256 `90cd049e...`, a copy of `crm_f104_v1/configs`,
  identical to the file behind collect_v1). They are passed explicitly: `CRM_COLLECTOR=$G3/source/scripts/crm_collect.py`,
  `CRM_CONFIG=configs/crm_main.json`. The job script `ag_soil.sbatch` refuses to start without them.
- The rigid collector is the unchanged `gen_collect_ext.py` (`b9f36a02...`) in its default native mode, runner
  `gen_runner_g.py`, job script `gen_array_g.sbatch` with `GEN_ROOT=$G3`.
- Source tree `$G3/source`: `src/` and `scripts/` from this worktree, the 8 arena folders (f104, g203, g228, g217,
  g260, g271, g251, g247), `source_manifest.json` copied from `crm_improve_20260922/source`. The 9 frozen files match
  that manifest; `scripts/gen_arenas.json` lists 10 arenas, each equal to its BMP. The files the HMMWV rows run and the
  two task files are read-only (`chmod a-w`). **Do not rsync into `$G3/source` while these jobs run** (rsync replaces
  read-only files); put new files next to them by name, or use a second tree.

## 2. Task files

**Soil, `soil_v1.json` (16,066 rows), in the order the workers take them (lowest tier first):**

| tier | rows | what |
|---|---|---|
| -1 | 300 | g217 dev headroom: 150 picks of the frozen soil specialist (`<g>__Scrm_B`) + 150 straight 6 m/s routes (`<g>__straight6`) |
| -1 | 10 | f104 drift check: `drift__<id>` = 10 collect_v1 ids re-driven (compare with `ag_drift_check.py --mode soil`) |
| 0-12 | 1,212 per tier | g203 and g228 groups 0-605, one route per group per tier (13 of the 20 routes per group, crm_tasks.py order) |

Training rows: 15,756 (g203 7,878, g228 7,878); 559 g203 and 520 g228 training-split groups. Nothing above tier 12.

**Rigid, `rigid_hmmwv_v1.json` (60,200 rows, 14 shards = 14 array tasks):**

| shards | rows | what |
|---|---|---|
| 0-11 | 48,000 | g203 and g228, all 1,200 groups x 20 routes; ids and tiers equal the soil twins |
| 12-13 | 12,000 | the 12 designed routes of every test group of g260, g271, g251, g247 (blacklisted ids) |
| 0 | 200 | f104 drift check: `drift__<id>` = 200 night-2 designed ids (compare with `ag_drift_check.py --mode rigid`) |

## 3. Watching

```bash
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
squeue -u harry
python3 $G3/source/scripts/ag_collect_status.py --tasks $G3/tasks/soil_v1.json --out $G3/soil_v1
python3 $G3/source/scripts/ag_collect_status.py --tasks $G3/tasks/rigid_hmmwv_v1.json --out $G3/rigid_v1
grep -l "3 consecutive failures\|Traceback" $G3/soil_v1/logs/*.out          # retired workers
ls $G3/soil_v1/failed | wc -l; ls $G3/soil_v1/runs/*/collection_failure.json 2>/dev/null | wc -l
grep -h "FAIL\|done in" $G3/rigid_v1/logs/*.out | tail
```

Drift checks (local, after syncing the drift runs):

```bash
# local
rsync -a --include='drift__*/***' --exclude='*' amd:$G3/soil_v1/runs/ /tmp/ag_drift_soil/
PYTHONPATH=src:scripts /home/harry/miniconda3/envs/nedm/bin/python scripts/ag_drift_check.py --mode soil --runs /tmp/ag_drift_soil --out $K3/e3/drift_soil.json
rsync -a --include='drift__*/***' --exclude='*' amd:$G3/rigid_v1/runs/ /tmp/ag_drift_rigid/
PYTHONPATH=src:scripts /home/harry/miniconda3/envs/nedm/bin/python scripts/ag_drift_check.py --mode rigid --runs /tmp/ag_drift_rigid --out $K3/e3/drift_rigid.json
```

Validated hours per arena (crm_qa.py pools a whole folder, so filter its rows by id prefix):
`python3 $G3/source/scripts/crm_qa.py $G3/soil_v1` (needs numpy: `source /work1/dannegrut/harry/nrd/env.sh; nrd_pychrono; $NRD_PYTHON ...`).

## 4. Launch commands (as run)

```bash
G3=/work1/dannegrut/harry/experiments/arena_gator_20260925
# rigid (01:53)
sbatch --parsable -p mi2104x -c 128 -t 03:00:00 --array=0-13 -J ag_rigid_v1 -o $G3/rigid_v1/logs/%x_%A_%a.out \
  --export=ALL,GEN_ROOT=$G3,GEN_TASKS=$G3/tasks/rigid_hmmwv_v1.json,GEN_OUT=$G3/rigid_v1,GEN_COLLECTOR=$G3/source/scripts/gen_collect_ext.py,GEN_RUNNER=$G3/source/scripts/gen_runner_g.py \
  $G3/source/scripts/gen_array_g.sbatch
# soil (01:58): refuses if queued + new tasks > 50 - 5; records each job id in $G3/e3/submissions.tsv
bash $G3/source/scripts/ag_soil_launch.sh $G3/tasks/soil_v1.json $G3/soil_v1 $G3/source/scripts/crm_collect.py \
  mi2101x:16:12:20 mi3501x:24:4:4
```

More soil capacity later (same file, same folder; the workers skip finished and claimed ids): e.g.
`bash $G3/source/scripts/ag_soil_launch.sh $G3/tasks/soil_v1.json $G3/soil_v1 $G3/source/scripts/crm_collect.py mi3501x:24:4:3`
(the mi3501x tasks stop claiming at 3 h 43 min, i.e. about 05:40; resubmit them then if the MI350X nodes are not
needed for training). mi2104x nodes (4 GPUs each) can take soil too once the rigid shards are done:
`... mi2104x:128:12:<n>`.

## 5. Stop commands

- **Soft stop of all soil jobs** (every worker finishes its current episode, then exits; affects every job writing to
  `soil_v1`, old and new): `touch $G3/soil_v1/STOP_CLAIMS`. It stays in force until deleted
  (`rm $G3/soil_v1/STOP_CLAIMS`); delete it before any relaunch.
- **The 12:00 cut** (PLAN 1.7): `touch $G3/soil_v1/STOP_CLAIMS` at 12:00, wait for the running episodes to finish
  (under ~7 min), then `scancel 436080 436092` (and any later soil job ids). The last, partly filled tier is dropped
  from every arena at dataset time (REVIEW_R2 amendment 2); `ag_collect_status.py` prints the per-tier counts.
- Hard stop: `scancel <job ids>`. A killed episode leaves its claim; another worker retakes it only after 25 min
  (`CRM_STALE_S`) and re-runs it from the start.
- Rigid: `scancel 436075` (runs with `episode_complete.json` are skipped by a resubmission of the same file).
- Drop one kind of soil row (e.g. the dev rows) without stopping the rest: a new task file with those rows set to
  `"run": false`, then the relaunch below. STOP_CLAIMS cannot target rows.

## 6. Relaunch with the Gator rows (E3a recipe; carried out at 04:47 by E3b2, see section 0)

1. Build the extended file locally so every soil_v1 row stays unchanged:
   `PYTHONPATH=src:scripts $PY scripts/ag_soil_tasks.py --head $K3/e3/dev_headroom/tasks_dev.json --append <gator rows json>
   --check-superset $K3/e3/tasks/soil_v1.json --out $K3/e3/tasks/soil_v2.json`
   (same defaults as soil_v1: 606 groups, tiers 0-12, 10 drift rows; `--groups-for g228=634` would add 28 g228 groups
   if wanted). Gator ids must carry a prefix (`gator__f104_v2_group_0001_route_03`): the id parsers need the trailing
   `_route_NN` / `_op_NN`, and a suffix would break them. Gator rows carry their own tiers (0-12 of the f104 order), so
   tier k of g203, g228 and the Gator run before tier k+1.
2. Copy to `$G3/tasks/soil_v2.json`, `chmod a-w`. Never edit `soil_v1.json`.
3. `scancel -t PENDING 436080 436092` (only the waiting tasks; the running ones drain soil_v1 rows and exit at their
   limit). Check `ls $G3/soil_v1/STOP_CLAIMS` is absent.
4. Submit with `ag_soil_launch.sh $G3/tasks/soil_v2.json $G3/soil_v1 <collector> ...`, the **same output folder**
   `soil_v1` (a new folder would re-run everything). If the collector is the Gator dispatcher `ag_crm_collect.py`
   (its name contains `crm_collect`, so the config still arrives), it must be frozen (read-only, sha256 recorded)
   before the first job starts: every episode re-reads it. HMMWV rows through it run `crm_collect.main` unpatched
   (E2 bit-identity check, `$K3/e2/bitid`), so HMMWV rows from both launches are interchangeable.
5. Put Gator rows in only after the Gator soil pilot shows < 5 % failed or rejected launches: deterministic failures
   are retried once and retire a worker after three in a row.

## 7. Extra jobs from the first checks

- 436103 (devel) and 436106 (mi3501x): 3 re-runs of `drift__f104_v2_group_0885_op_07`, output `$G3/e3/drift_rerun`.
  All 3 are identical to tonight's soil_v1 run and differ from collect_v1's run of that id (see NOTES_E3a section 6).
- 436108 (devel, 30 min): 15 more soil drift ids, task file `$G3/tasks/drift_more_soil.json`, output `$G3/e3/drift_more`.
