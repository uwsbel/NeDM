# Cluster operations for mass collection — f104 rigid pipeline, and what changes for CRM

Scout report, 2026-09-16 ~23:30 CDT. Read-only reconnaissance. Nothing was submitted; the only scheduler calls were
`sinfo`, `scontrol show`, `sacct`, `squeue`, and `sbatch --test-only` (dry run, creates no job).
Paths: local = `/home/harry/NeDM-traverse_mppi`, cluster campaign = `C=/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909`.

## 0. Bottom line for tonight

- Every node is whole-node exclusive, no GRES, no memory tracking. A job gets all CPUs and all GPUs of the node; GPU binding
  is entirely the job script's business (`HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` per worker).
- The rigid campaign packed **1 Chrono process per CPU core** (2008 workers, 66 h of driving in 13.6 min wall). CRM is
  GPU-bound: **1 worker per GPU (possibly several per GPU, see 7.4)**, ~8 CPU threads each.
- The **shared flock queue failed across nodes on this filesystem** (production_v1, cancelled). Everything after it used
  static shards (no shared mutable state) + skip-if-done. For a dynamic GPU queue use `os.mkdir` claims, never `flock`,
  never a shared `state.json`.
- The FSI/HIP build is verified on all three GPU architectures tonight (jobs 423533-35 by a sibling agent): MI210 rtf 0.174,
  MI300X 0.201, MI350X 0.254 at spacing 0.08, step 5e-4, 4.0 M SPH particles.
- Balance: **396.0 of 1500 billed node-hours used (26.4 %)**, harry 47.2. Cheapest GPU-hour is MI210/MI250 (0.1 billed/GPU-h).
- Idle right now: 23 x mi2101x (1 GPU), 4 x mi2104x (4 GPU), 3 x mi3001x, 3 x mi3501x, 1 x mi3508x (8 GPU), 4 x devel.

## 1. Partitions, hardware, limits, cost

`ssh amd 'sinfo -N -o "%P %N %c %m %G %t"'` reports `GRES=(null)` and `MEMORY=1` for every node: Slurm tracks neither
GPUs nor memory here (`GresTypes=(null)`, `RealMemory=1`). `SelectType=select/cons_tres`, but every partition is
`OverSubscribe=EXCLUSIVE`. GPU type/count come from `/etc/motd` + the user guide (https://amdresearch.github.io/hpcfund);
`gfx` versions for MI210/MI300X/MI350X were confirmed by `rocm-smi` in tonight's smoke logs.

| partition | nodes (names) | CPUs/node | GPUs/node | GPU, VRAM/device | arch | max walltime (filter) | charge / node-h | charge / GPU-h |
|---|---|---|---|---|---|---|---|---|
| `devel` | 4 (`k006-00[4-7]-v2`) | 16 | 1 | MI210, 64 GB | gfx90a | 30 min | 0.1 | 0.1 |
| `mi2101x` | 24 (`k006-00[4-7]-v[3-8]`, VMs) | 16 | 1 | MI210, 64 GB | gfx90a | 12 h (guide) | 0.1 | 0.1 |
| `mi2104x` | 21 (`k002-00[5-6]`, `k003-0[01-10]`, `k005-0[02-10]`) | 128 | 4 | MI210, 64 GB | gfx90a | 24 h (guide), 16 nodes/job | 0.4 | 0.1 |
| `mi2508x` | 10 (`k004-0[01-10]`) | 128 | 8 (GCDs) | MI250, 64 GB | gfx90a | 12 h (dry-run accepted `-t 12:00:00`) | 0.8 | 0.1 |
| `mi3001x` | 8 (`k002-004-v[1-8]`, VMs) | 16 | 1 | MI300X, 192 GB | gfx942 | 4 h | 0.125 | 0.125 |
| `mi3008x` | 2 (`k002-00[2-3]`) | 192 | 8 | MI300X, 192 GB | gfx942 | 12 h batch | 1.0 | 0.125 |
| `mi3258x` | 1 (`k002-001`) | 256 | 8 | MI325X, 256 GB | gfx942 | 12 h batch | 1.2 | 0.15 |
| `mi3501x` (default) | 8 (`k007-005-v[1-8]`, VMs) | 24 | 1 | MI350X, 288 GB | gfx950 | **4 h** (dry-run rejected `-t 06:00:00`) | 0.175 | 0.175 |
| `mi3508x` | 4 (`k007-00[1-4]`) | 256 | 8 | MI350X, 288 GB | gfx950 | 12 h batch | 1.4 | 0.175 |

- `scontrol show partition` says `MaxTime=4-00:00:00`, `MaxNodes=UNLIMITED` everywhere except `devel` (30 min). Those are
  NOT the real limits; the lua submit filter enforces the table above.
- Charge factors: the ledger uses the user-guide factors, not `TRESBillingWeights` (which are 0.01/0.04/0.08/0.1/0.0125/0.12).
  Verified: harry's `sacct` node-hours per partition x the factors above = 47.18, ledger says 47.196.
- State at 23:28 CDT (from `sinfo -h -o "%P %a %l %D %T"`): devel 4 idle; mi2101x 23 idle / 1 alloc; mi2104x 4 idle
  (`k003-001`, `k003-004`, `k003-005`, `k005-002`, `k005-007` seen idle) / 16 alloc / 1 down; mi2508x 0 idle / 8 alloc / 2 down;
  mi3001x 3 idle / 3 alloc / 1 drain / 1 down; mi3008x 0 idle; mi3258x 0 idle; mi3501x 3 idle + 3 completing (sibling CRM
  tests) / 1 alloc / 1 drained; mi3508x 1 idle (`k007-003`) / 3 alloc. Queue pressure: mi3001x 22 pending, mi2508x 9
  pending, mi3258x 9 pending, mi2101x 5 pending.
- Idle GPU count right now: 23 + 16 + 3 + 3(+3) + 8 + 4(devel) = about 57-60.

## 2. Submit filter rules (verified with `sbatch --test-only`)

Filter = `JobSubmitPlugins=lua,require_timelimit`. Full transcript order: runtime limit -> partition provided -> billing
account -> single partition -> job limits (walltime) -> partition access -> job size (nodes) -> active allocation ->
balance -> max job limit ("number of jobs in queue = N").

| rule | evidence |
|---|---|
| `-t` is mandatory | `--> runtime limit is required ... please specify with -t` |
| exactly one partition per job | `-p mi2101x,mi3001x` -> `error: please submit to a single partition only` |
| per-partition walltime cap | mi3501x `-t 06:00:00` -> `requested timelimit exceeds max of 4 hours`; mi2508x and mi2101x accept `12:00:00` |
| account | `-A dannegrut` (every f104 sbatch has it) |
| max jobs in queue | checked, numeric cap not printed; array tasks count. nav_v1 throttled with `--array=0-29%23`; production_v2 had 46 tasks queued at once without rejection. `MaxArraySize=1001` |
| `--mem` | never use: `RealMemory=1`, any `--mem=NG` pends forever |
| interactive | guide: 1 concurrent interactive job |

Consequence that shaped all f104 launch scripts: a heterogeneous fleet needs **one `sbatch` per partition**, each with
its own `-c <all CPUs of that node type>`; the worker count is derived from `SLURM_CPUS_PER_TASK`.
`-c` must be given: without it `SLURM_CPUS_PER_TASK` is unset (the f104 scripts then fall back to 16 -> 14 workers even on a
256-CPU node) and the batch step may be affinity-bound to one core (`TaskPlugin=task/affinity`; not tested).
CPUs per node type for `-c`: devel 16, mi2101x 16, mi2104x 128, mi2508x 128, mi3001x 16, mi3008x 192, mi3258x 256,
mi3501x 24, mi3508x 256.

## 3. Environment recipe (rigid) and what differs for the FSI build

`/work1/dannegrut/harry/nrd/env.sh` (43 lines): sets `NRD_ROOT=/work1/dannegrut/harry/nrd`,
`CHRONO_BUILD=$NRD_ROOT/chrono-build`, `NRD_PYTHON=/work1/dannegrut/harry/venvs/nedm/bin/python` (3.12),
`NRD_PYSTACK=/share/sw/ai/pytorch/2.10.0`, `MESA_SHADER_CACHE_DIR=$NRD_ROOT/mesa_cache`; functions `nrd_pychrono`
(`PYTHONPATH=$CHRONO_BUILD/bin:$NRD_PYSTACK:$PYTHONPATH`) and `nrd_use_lavapipe` (CPU Vulkan ICD for the camera).

Rigid header used by every f104 sbatch:
```bash
set -eo pipefail
source /work1/dannegrut/harry/nrd/env.sh
nrd_pychrono
nrd_use_lavapipe
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=4   # nav: LP_NUM_THREADS=16, NAV_TORCH_THREADS=8
export FDM_RUNTIME_FINGERPRINT=$C/pilot_runtime_412394.json
"$NRD_PYTHON" -P -u <runner>
```

**RIGID-SPECIFIC / must change for CRM**
1. `nrd_pychrono` points at `chrono-build` (no FSI). For CRM: `export CHRONO_BUILD=$NRD_ROOT/chrono-build-fsi` *before*
   `nrd_pychrono`, or as tonight's smoke did: `export PYTHONPATH=$NRD_ROOT/chrono-build-fsi/bin:$NRD_PYSTACK:${PYTHONPATH:-}`.
   `--chrono-data` must become `/work1/dannegrut/harry/nrd/chrono-build-fsi/data` (hard-coded to `chrono-build/data` in
   `scripts/gen_runner.py:27`, `scripts/gen_mission_array.py:18`, `nav_run.sbatch:16`, contract `chrono_data`).
2. `FDM_RUNTIME_FINGERPRINT=$C/pilot_runtime_412394.json` hashes 108 files under `chrono-build/` (6 `.so` + pychrono `*.py`
   + `data/vehicle/hmmwv/**` + grass texture). `gen_collect.py:271-276` and `run_traverse_f104_queue.py:222-229`
   refuse to run without it and verify every hash. A CRM collector that keeps this gate needs a new fingerprint generated
   against `chrono-build-fsi` — `scripts/traverse_fdm_rgbd_diverse_batch.py:62 runtime_fingerprint(chrono_data)` reads
   `$CHRONO_BUILD`, and its file list (`:68-69`) has no `libChrono_fsi*.so` / `_fsi.so`; add them.
   Recipe used originally: `$C/launch/run_pilot.py:4-5`.
3. `OMP_NUM_THREADS=1` was mandatory for 1-process-per-core packing (otherwise 64-thread idle pools; Slurm also sets
   `OMP_NUM_THREADS` from `-c`). CRM workers want ~8 MBS threads (tonight's smoke: `--threads 8`, `OMP_NUM_THREADS=8`);
   set it explicitly per worker = floor(CPUs / workers), cap 8-16.
4. `LP_NUM_THREADS` / lavapipe only matter if a camera renders inside the episode (not in collection).
5. ROCm: `libChrono_fsisph.so` links `/opt/rocm-7.2.0/lib/{libamdhip64.so.7,libhsa-runtime64.so.1,libamd_comgr.so.3}` by
   absolute RPATH; no `module load` needed (smoke ran without one). Do NOT mix with `module load pytorch/2.10.0`
   (pulls rocm 7.1) in the same process unless tested.

## 4. How the rigid collections were launched

### 4.1 production_v1: shared flock queue — FAILED across nodes (do not reuse)
- Worker: `scripts/run_traverse_f104_queue.py` (cluster copy `$C/source_v1/scripts/`). CLI: `--contract <json>`
  `--workers N` `[--initialize | --status]`.
- Mechanism: `queue/queue.lock` with `fcntl.flock` (`:124-133`), one shared `queue/state.json` rewritten by
  tmp+`os.replace`+dir fsync (`atomic_json`, `:52-64`), append-only `queue/completed_ledger.jsonl` replayed from
  `ledger_offset` (`:149-162`), `claim()` hands out `next_index` (`:176-186`), quota stop at `target_frames`.
- Launch: `$C/launch/production.sbatch` (`-t 04:00:00`, `--workers "$SLURM_CPUS_PER_TASK"`), submitted per partition by
  `$C/launch/submit_remaining.py` (`sbatch --parsable -p <part> -c <cpus> [--array 0-(n-1)] production.sbatch`).
- Outcome (`artifacts/traverse/fdm_f104_50h_20260909/production_v1_incident_audit/{report.md,audit.json}`): 42 node
  processes; **28 died with `FileNotFoundError ... queue/state.json`** (another client's `os.replace` made the file
  briefly absent on WekaFS), 2 with `Completion does not own an active claim`, **14 duplicate claims** (same index handed
  to two nodes). The local contract check (`queue_worker_contract_check.json`: 6 processes on ONE machine) had passed.
  => `flock` does not give cross-node mutual exclusion on `/work1` (`wekafs`, `dentry_max_age_positive=1000`), and
  read-after-rename of a hot shared file is not safe.
- Useful positive evidence: all 14 duplicates were caught by the output-dir guard
  `require(not out.exists()); out.mkdir(parents=True)` (`:328-329`) — i.e. **`mkdir` exclusivity held across nodes**.

### 4.2 production_v2/v3/v4: static weighted shards — the proven mass-collection design
- Worker: `scripts/run_traverse_f104_shard.py` (cluster: `$C/control_v2/`). CLI: `--contract`, `--shard-index`, `--workers`.
- Ownership is arithmetic (`ownership()`, `:40-47`): task `i` belongs to the shard whose cumulative-weight interval
  contains `i % sum(shard_weights)`; weights = CPUs of the node that will run the shard
  (`[128]*5+[256,192,256]+[16]*24+[16]*7+[24]*7`, sum 2008, `$C/launch/freeze_shards_v2.py`). No cross-node lock or counter.
- Per shard: `shards/shard_%05d/{state.json,completed_ledger.jsonl,failures.jsonl,error_*.json,node_binding.json}`;
  `mkdir(exist_ok=False)` of the shard dir is the duplicate-invocation guard (`:71`). Threads in one process share an
  in-memory state under a `threading.Lock`.
- Global stop: a single monitor (`$C/launch/monitor_shards_v2.py`, run on the login node, `monitor_owner/` mkdir as
  singleton guard) re-reads all ledgers every 10 s, asserts no duplicate completions, writes `progress.json`, and creates
  `STOP_CLAIMS` at quota; workers check `(root/'STOP_CLAIMS').exists()` before each claim (`:81`). In-flight episodes finish.
  Never-started pending array tasks were then cancelled by `$C/launch/cancel_unstarted_at_quota.py`.
- Launch (`$C/launch/submit_shards_v2.py`, recorded in `production_v2/submission.json`):
  ```
  sbatch --parsable -p mi2104x -c 128 --export ALL,F104_SHARD_OFFSET=0  --array 0-4  $C/launch/production_v2.sbatch
  sbatch --parsable -p mi3258x -c 256 --export ALL,F104_SHARD_OFFSET=5               $C/launch/production_v2.sbatch
  sbatch --parsable -p mi3008x -c 192 --export ALL,F104_SHARD_OFFSET=6               ...
  sbatch --parsable -p mi3508x -c 256 --export ALL,F104_SHARD_OFFSET=7               ...
  sbatch --parsable -p mi2101x -c 16  --export ALL,F104_SHARD_OFFSET=8  --array 0-23 ...
  sbatch --parsable -p mi3001x -c 16  --export ALL,F104_SHARD_OFFSET=32 --array 0-6  ...
  sbatch --parsable -p mi3501x -c 24  --export ALL,F104_SHARD_OFFSET=39 --array 0-6  ...
  ```
  `production_v2.sbatch:15-16`: `shard=$((F104_SHARD_OFFSET + ${SLURM_ARRAY_TASK_ID:-0}))`, `--workers "$SLURM_CPUS_PER_TASK"`.
- Result (`production_v2/progress.json`): 11,412 episodes, 66.42 h of driving, 0 failed episodes, 46 shards,
  2008 workers, **13.6 min wall**. v3/v4 (Sep 12): 35 shards `[128]*9+[16]*19+[24]*7`, same sbatch with the contract path changed.
- Contract = frozen inputs (`$C/launch/freeze_production.py`): source tree manifest sha, collector sha, worker sha, task
  file sha (18,000 tasks = 1500 groups x 12 routes, round-robin one-route-per-group ordering so an early stop is balanced),
  `episode_wall_timeout_s=3600`, `max_attempts=1`, `target_seconds=180000`.
- Per-episode subprocess (`:138-146`): `$NRD_PYTHON -P -u <collector> --source-root --source-manifest-sha256 --case
  --case-sha256 --route --route-sha256 --out runs/<task_id> --chrono-data --horizon-s 120`, stdout->`runs/<id>/collector.log`.
  After exit 0, `verify_episode()` (`queue.py:250-300`) re-hashes every artifact listed in `episode_complete.json`, checks
  shapes (`state (n,17)`, `action (n,3)`, `pose (n,3)`, `dt_s=0.05`), rich telemetry coverage, then appends to the ledger.
- NOT resumable by design: `require(not out.exists())` — an existing run dir is never reused; retries go to
  `<id>__retry_NNN`; a shard dir that exists refuses to start. 5 consecutive errors stop the shard.

### 4.3 gen_v1 / night2 / sensor / nav: light array runners — the resumable pattern (use this shape)
- Files: `scripts/gen_array.sbatch`, `scripts/gen_runner.py`, `scripts/gen_collect.py`, `scripts/gen_mission.sbatch`,
  `scripts/gen_mission_array.py`, `scripts/nav_array.py`; cluster copies `$C/gen_v1/{gen_array.sbatch,gen_runner.py,gen_collect.py,
  gen_mission.sbatch,planner/gen_mission_array.py}`, `$C/nav_v1/{nav_run.sbatch,nav_run32.sbatch,nav_pilot.sbatch,code/nav_array.py}`.
- Task file = JSON list of rows with a precomputed integer `shard`; runner selects `rows where shard == SLURM_ARRAY_TASK_ID`
  and runs them in a `ThreadPoolExecutor(max_workers=GEN_WORKERS)` of subprocesses.
  `GEN_WORKERS=${GEN_WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-16} - 2 ))}` (`gen_array.sbatch:15`); nav: `NAV_WORKERS=1`
  (render + 8 torch threads per run).
- Row schemas: gen `{id, group, arena, case, route, shard, run}` (paths relative to `$GEN_ROOT`); missions
  `{mission_id, arm, mission, shard}`; nav `{mission, arm, mode, period, latency_s, ..., shard, save_frames}`.
- Shard assignment keeps all arms of a group/mission in ONE shard (= one node): `gen_pools.py:80`
  `shard = md5(group) % shards`; `nav_tasks.py:34` `shard = i % shards` per mission. Reason: rigid Chrono on this
  cluster is bit-deterministic per node, not across node types.
- Because of the one-partition rule, big runs were split into per-partition task files with their own shard numbering:
  `tasks_data_mi2101x.json` (3996 rows / 30 shards of 132-144), `tasks_data_mi2508x.json` (2004 / 1 shard),
  `tasks_data_mi3008x.json` (3000 / 1 shard), plus `tasks_*_catchup_<partition>.json` to move not-started shards to
  a free big node (gen_v1/LOG.md 04:21Z).
- Exact submit lines (from `sacct --format=SubmitLine`; `GEN_TASKS`/`GEN_OUT` were exported in the calling shell and
  passed by `--export=ALL`):
  ```
  sbatch -p mi2101x -J gen_test   -c 16  --array=0-47 -o $C/gen_v1/test/arr_%A_%a.out  --export=ALL gen_array.sbatch
  sbatch -p mi3008x -J gen_data_a -c 192 --array=0    -o $C/gen_v1/data/arrA_%A_%a.out --export=ALL gen_array.sbatch
  sbatch -p mi2508x -J gen_data_b -c 128 --array=0    -o $C/gen_v1/data/arrB_%A_%a.out --export=ALL gen_array.sbatch
  sbatch -p mi2101x -J gen_data_c -c 16  --array=0-29 -o $C/gen_v1/data/arrC_%A_%a.out --export=ALL gen_array.sbatch
  sbatch -p mi3501x -J gen_miss   -c 24  --array=0-28 -o $C/gen_v1/missions_run/arr_%A_%a.out --export=ALL gen_mission.sbatch
  sbatch -p devel -t 00:30:00 -J gen_mpilot -c 16 --array=0 -o .../mission_pilot/arr_%A_%a.out --export=ALL gen_mission.sbatch
  # nav_v1 (RUNNING.md):
  sbatch -p mi2101x --array=0-29%23 -o logs/main_%A_%a.out \
    --export=ALL,NAV_TASKS=$N/tasks_main.json,NAV_OUT=$N/main,NAV_WORKERS=1 nav_run.sbatch
  # night2 closed loop (scripts/f104_n2_submit.sh): rsync routes + scp tasks, then
  ssh amd "cd $R && sbatch -p mi2101x --array=0-39 n2_closed.sbatch"
  ```
- Throughput reference (rigid): mi3008x, 190 workers, 3000 episodes in 1941 s; 16-CPU mi2101x shard of ~140 episodes ~18 min.
- **Resumability** = skip-if-done marker, re-submit the same array: `gen_runner.py:23`
  `if exists(d+'/episode_complete.json'): return 'cached'`; missions/nav use `mission_outcome.json`
  (`gen_mission_array.py:14`, `nav_array.py:16`). The marker is the LAST file the collector writes
  (`gen_collect.py:315-317`, contains sha256 of every artifact), so a killed episode has no marker.
  GOTCHA: `gen_collect.py:270` refuses an out dir that already has `outcome.json` / `collection_request.json` /
  `trajectory.npz` ("Preserve existing output; use a new directory"), so an episode killed mid-run (timeout, node loss)
  is NOT re-runnable in place — the partial dir must be moved aside first. gen_v1 handled this by hand
  (`missions_run/superseded_before_fix/`). A CRM runner should rename a marker-less dir to `<id>__stale_<ts>` itself.
- Atomicity as implemented: only the queue/shard workers write JSON atomically (tmp + `os.replace` + fsync).
  `gen_collect.dump()` (`:45-46`) is a plain `write_text`, and `np.savez_compressed` writes in place; correctness rests
  solely on "marker written last". Failures: nonzero return code is printed as `FAIL <id> rc=..` in the array `.out`,
  per-episode log at `$OUT/logs/<id>.log`; `collection_failure.json` is written in the run dir (`gen_collect.py:319-322`).
- Per-episode timeouts: gen 3600 s, missions 7200 s, nav `NAV_TIMEOUT` default 10800 s (subprocess `timeout=`).

## 5. Per-episode output and storage

- Run dir (rigid, after rich telemetry is deleted; `production_v2/runs/<id>/`, 17-18 files): `trajectory.npz` 40-50 KB,
  `collection_request.json` 22 KB, `simulation_provenance.json` 20 KB, `case.json` 13-15 KB, `collection_meta.json` 13-15 KB,
  `reference.json` 10-12 KB, `native_height_check.json` 5 KB, `command_reference.npz` 4 KB, `outcome.json` 2.8 KB,
  `episode_complete.json`, `f104_episode.json`, `anchor_state.npz`, `initial_state_validation.json`, `contact_events.json`,
  (`queue_launch.json`, `queue_validation.json` for queue/shard runs), `collector.log` 2.7 KB.
  Total **~190 KB/episode** (2.2 GB / 11,412; gen_v1 data 1.6 GB / 9000 = 180 KB). Mean episode 21 s = 419 frames at 20 Hz.
- Rich telemetry (`rich_telemetry.npz`, `rich_telemetry.json`, `rich_intervals.npz`): +0.8-1.5 MB/episode
  (8.94 GB / 11,412 in v2; 14.7 GB / 9,600 in v4). Labels need only `trajectory.npz`, `outcome.json`,
  `command_reference.npz`, `case.json`, `anchor_state.npz`; `gen_runner.py:30-33` deletes the rich files after a successful
  episode, and `$C/launch/cleanup_2026-09-15.sbatch` purged them campaign-wide (freed 42 GB of 52 GB).
  NOTE: `verify_episode` in the queue/shard worker REQUIRES the rich files, so deletion must happen after verification.
- `/work1` = WekaFS, 1.9 TB total, 1.4 TB free; campaign dir is 28 GB. `/home1` is a 24 GB quota, ~93 % full — write nothing there.
- CRM: do NOT dump SPH particle states per frame (4 M particles x 3 floats = 48 MB per snapshot). Keep the rigid file set;
  if soil state is wanted, save a cropped height/rut raster at episode end.

## 6. Sync (code up, results down)

No wrapper script exists for f104 sync; it was ad hoc rsync/scp over the `amd` ssh alias. `$WORK` is not defined in
non-interactive ssh, always use absolute paths. Observed layout mapping:
`artifacts/traverse/fdm_f104_50h_20260909/<sub>/` (local) <-> `$C/<sub>/` (cluster).
```bash
C=/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909
# up (pattern of scripts/f104_n2_submit.sh:6-8)
rsync -az "$L/routes/" "amd:$R/routes/";  scp -q "$L/tasks_cluster.json" "amd:$R/tasks_cluster.json"
# down (pattern of scripts/traverse_wp8_eval_runs.sh:36); local production_v2/runs has no collector.log => logs were excluded
rsync -az --exclude 'collector.log' --exclude 'rich_*' amd:$C/production_v2/runs/ artifacts/traverse/fdm_f104_50h_20260909/production_v2/runs/
```
- The cluster trees are rsync copies, not git checkouts. Each campaign carries its own frozen code copy
  (`$C/source_v1`, `$C/gen_v1/source`, `$C/nav_v1/{code,source}`) with `source_manifest.json`
  (`{files: {relpath: sha256}}`) that the collector verifies before and after each episode (`gen_collect.py:237-242,301`).
  Tonight's CRM tree already exists: `/work1/dannegrut/harry/experiments/crm_f104_20260916/{source/{src,scripts,assets},smoke,logs}`.
- ~11k run dirs x 17 small files: rsync of ~200k small files is metadata-bound (minutes). For CRM volumes, tar per shard on
  the cluster, or pull only the 5 label files with `--include`.
- Transfer rate seen earlier: ~50 MB/s workstation <-> cluster.

## 7. GPUs for CRM

### 7.1 What is already verified about FSI on these GPUs
- Build: `/work1/dannegrut/harry/nrd/chrono-build-fsi` (configured 2026-09-08, `configure_fsi.log`; built by
  `nrd/build_unified.sbatch`, log `nrd/logs/unified.408935.out`). Source `/home1/harry/chrono` (branch `project/nrd`, Chrono
  `main`, not 10.0.0). `CHRONO_GPU_BACKEND=HIP`, `CHRONO_HIP_ARCHITECTURES=gfx90a;gfx942;gfx950`, ROCm 7.2.0
  (`/opt/rocm-7.2.0`, HIP clang 22), `CH_ENABLE_MODULE_FSI=ON`, `FSI_SPH=ON`, `FSI_TDPF=OFF` (no HDF5),
  `CH_USE_SPH_DOUBLE=OFF` (single-precision SPH), Release, `-march=native`; vehicle SCM GPU backend also HIP; sensor =
  Vulkan RT (lavapipe), no OptiX. Same tree has all 7 python modules
  (`bin/pychrono/{_core,_fea,_fsi,_postprocess,_robot,_sensor,_vehicle}.so`), so it is a superset of `chrono-build`.
  `lib/libChrono_fsisph.so` carries code objects for all three archs.
- `nrd/smoke_fsi.py` (import + list SPH names), `nrd/smoke_fsi_api.py` (checks 13 names incl. `ChFsiProblemCartesian`,
  `SPHParameters`, `BoundaryMethod_ADAMI`, `ShiftingMethod_PPST_XSPH`, `ViscosityMethod_ARTIFICIAL_BILATERAL`,
  `ElasticMaterialProperties` — the last is the OLD name), `nrd/smoke_fsi_api2.py` (finds `SoilProperties`, `SetCrmSPH`).
  Their outputs were not saved to a log; the recorded conclusion is in `/home/harry/NeDM/docs/collection_pipelines_amd_port.md`:
  API rename on `main`: `fsi.ElasticMaterialProperties()` -> `fsi.SoilProperties()`, `SetElasticSPH` -> `SetCrmSPH`
  (`src/nedm/hmmwv_crm.py` accepts both). Those Sep-8 checks were import/API only — no simulation.
- **Tonight (sibling agent, jobs 423533/34/35, `crm_f104_20260916/smoke/crm_smoke.sbatch` + `source/scripts/crm_smoke.py`)**:
  full HMMWV-on-CRMTerrain run from the f104 BMP on one GPU of each arch, exit 0:

  | job | partition / node | GPU (`rocm-smi`) | n_sph / n_bce | build_s | sim 4 s wall | rtf (sim/wall) |
  |---|---|---|---|---|---|---|
  | 423533 | mi2101x k006-004-v4 | MI210 gfx90a | 4,008,004 / 3,006,003 | 3.1 | 23.0 s | 0.174 |
  | 423534 | mi3001x k002-004-v4 | MI300X gfx942 | same | 3.4 | 19.9 s | 0.201 |
  | 423535 | mi3501x k007-005-v2 | "AMD Radeon Graphics" gfx950 (MI350X) | same | 3.7 | 15.7 s | 0.254 |

  Settings: spacing 0.08 m, depth 0.24 m, step 5e-4, active box [2,2,1], 8 threads, whole 80x80 m arena.
  MI350X breakdown: `rtf_cfd` 3.05 (wall/sim, GPU SPH), `rtf_mbd` 0.30. Follow-ups 423543-46 on mi3501x (9 s hill run):
  step 5e-4 rtf 0.261, 7.5e-4 0.384, 1e-3 0.531; cropped domain (479k particles instead of 4.0 M) only 0.289.
- The vehicle trajectory printed every 0.5 s is identical to 16 digits on all three GPU types / CPU types for that 4 s
  flat-ground run. Encouraging, NOT a determinism proof (no slip/sinkage-critical event, single run each).
- NOT verified anywhere: MI250 (mi2508x) and MI325X (mi3258x) execution (same archs gfx90a / gfx942, should load);
  multiple FSI processes on one node with `HIP_VISIBLE_DEVICES`; multiple FSI processes sharing one GPU; VRAM use;
  long-episode stability; the known pytorch-on-MI210 failure (`HIP error: file not found`, pytorch/2.10.0+rocm7.1 on
  MI210 compute nodes) does not affect Chrono HIP (the smoke ran on MI210) but DOES matter if the closed-loop planner
  wants torch on the same MI210 GPU — run the hazard net on CPU there (nav_v1 already did: `--torch-threads 8`, CPU).

### 7.2 Binding one worker per GPU inside a whole-node job
- All GPUs are visible to the job (no GRES, no cgroup device filtering). The cluster guide's own recipe for N independent
  single-GPU tasks is `export HIP_VISIBLE_DEVICES=<k>` per task; harry's precedent
  `/work1/dannegrut/harry/experiments/mppi_claude_20260908/slurm/mppi_train_suite.sbatch:24-34` packs 8 trainings on
  mi3508x with a `( export HIP_VISIBLE_DEVICES=$gpu; ... ) &` subshell loop and `wait` per pid.
- `ROCR_VISIBLE_DEVICES=<k>` filters at the ROCr/HSA layer (the process sees exactly one device, index 0), which is the
  safer choice for a library like Chrono FSI that simply uses device 0. Set BOTH consistently:
  `ROCR_VISIBLE_DEVICES=$k HIP_VISIBLE_DEVICES=0`. Do not set `HIP_VISIBLE_DEVICES=$k` together with
  `ROCR_VISIBLE_DEVICES=$k` (HIP indices are renumbered after the ROCr filter; k>0 would select nothing).
- GPU count at run time: `NGPU=$(rocminfo | grep -c -E '^\s+Name:\s+gfx')` (checked on the login node: 2).
  First line of each worker log should print `rocm-smi --showproductname` or the device name to prove the binding.
- CPU side: with `-n 1 -c <all>` the batch step owns all cores; give each worker `OMP_NUM_THREADS=T`,
  `T = min(8, CPUS / NWORKERS)`. Optional `taskset -c` slices; NUMA-GPU affinity not investigated
  (`rocm-smi --showtoponuma` on a compute node would tell).
- Guide warning: a "low-power state" rocm-smi warning appears on idle GPUs; harmless.

### 7.3 Cost / throughput arithmetic (from tonight's rtf, step 5e-4, 1 worker per GPU)
| GPU | sim-s per GPU-h | billed per GPU-h | sim-h per billed node-hour |
|---|---|---|---|
| MI210 (mi2101x, mi2104x) | 626 | 0.1 | 1.74 |
| MI250 (mi2508x) | unmeasured (gfx90a) | 0.1 | ? |
| MI300X | 724 | 0.125 | 1.61 |
| MI350X | 914 | 0.175 | 1.45 |

- 50 sim-hours (the rigid campaign's size) at step 5e-4 = ~29-35 billed node-hours: affordable (1104 left). The
  constraint is wall time, not budget: ~57 idle GPUs x ~650 sim-s/h = ~10 sim-h per wall-hour if everything idle is taken.
  Step 1e-3 doubles that if the physics team accepts it.
- Episode wall estimate at rtf 0.17-0.26: mean 21 s episode = 80-125 s + ~4 s terrain build + python/Chrono import;
  120 s horizon episode = 8-12 min. `episode_wall_timeout_s` 3600 stays adequate; mi3501x/mi3001x 4 h cap = ~100+ episodes/job.
- Billing is per node regardless of GPUs used: a mi2104x job that drives only 1 of 4 GPUs costs 4x. Always fill the node.

### 7.4 Open experiment worth 10 minutes before the mass launch
Cropping 4.0 M -> 0.48 M particles changed rtf only 0.261 -> 0.289, and `rtf_cfd` dominates: the per-step cost is
active-domain work + kernel-launch latency, not total particle count, so one worker likely leaves the GPU mostly idle.
Test 1/2/4 workers on ONE GPU (mi2101x or devel, 16 CPUs: 4 workers x 4 threads) and read aggregate sim-s per wall-s.
If it scales, set `WORKERS_PER_GPU` accordingly (VRAM is not the limit: 64-288 GB per device). Watch CPU: `rtf_mbd`
0.30-0.44 with 8 threads means the MBS side needs real cores; 16-CPU single-GPU VMs saturate at about 4 workers x 4 threads.

## 8. Proposed design for tonight: N GPU workers per node, shared queue, atomic claims

Design rules derived from sections 4.1-4.3:
1. No `flock`, no shared mutable file. Claim = `os.mkdir(claims/<task_id>)`; `FileExistsError` = someone else has it.
   (mkdir exclusivity is the one primitive that demonstrably held across nodes in the v1 incident.)
2. Done marker = `runs/<task_id>/episode_complete.json`, written last via tmp + `os.replace` in the same directory.
3. Per-worker append-only ledger `workers/<jobid>_<host>_g<k>w<j>.jsonl` (one writer each); a login-node monitor aggregates
   (copy of `monitor_shards_v2.py` logic) and drops `STOP_CLAIMS` at quota. Workers check `STOP_CLAIMS` and a wall-clock
   deadline (passed in by the sbatch; Slurm 22.05.8 here has no `SLURM_JOB_END_TIME`) before each claim so no episode is
   started that cannot finish.
4. Resume = re-submit the same sbatch. A claim whose job is gone is stale: the janitor (or the worker at startup) does
   `os.rename(claims/<id>, stale/<id>.<ts>)` (atomic; exactly one renamer wins) and
   `os.rename(runs/<id>, runs_stale/<id>.<ts>)` if no marker; then the task is claimable again. Stale test:
   `claim.json` records `{job_id, host, pid, t}`; stale iff `squeue -h -j <job_id>` is empty (run from the login node),
   or age > `episode_wall_timeout_s`.
5. Paired closed-loop tests: the claim unit is the GROUP (all arms run back-to-back by the same worker on the same GPU),
   mirroring the rigid "all arms share a node" rule; record `host`, GPU index, `gfx` arch and job id in every outcome.
6. Randomise the scan start (`offset = hash(worker_id) % n`) so 60 workers do not hammer the same directory entries;
   claim traffic is ~1/s cluster-wide, negligible for Weka.
7. One `sbatch` per partition (filter rule), same script, `-c` = all CPUs; the worker count comes from the GPU count.

### 8.1 sbatch template (`crm_collect.sbatch`)
```bash
#!/bin/bash
#SBATCH -A dannegrut
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00                      # <= 4 h on mi3501x/mi3001x; 12 h allowed on mi2101x/mi2508x/mi3x08x, 24 h mi2104x
# submit with:  -p <one partition>  -c <all CPUs of that node type>  [--array=0-(nodes-1)]  -o $CRM/<run>/logs/%x_%A_%a.out
set -o pipefail                          # no -u: env.sh expands unset LD_LIBRARY_PATH/PYTHONPATH; no -e: one dead worker must not kill the rest
source /work1/dannegrut/harry/nrd/env.sh
export CHRONO_BUILD=$NRD_ROOT/chrono-build-fsi          # RIGID->CRM: FSI/HIP build, not chrono-build
nrd_pychrono                                            # PYTHONPATH=$CHRONO_BUILD/bin:$NRD_PYSTACK:...
CRM=/work1/dannegrut/harry/experiments/crm_f104_20260916
RUN=${CRM_RUN:?campaign subdir, e.g. $CRM/collect_v1}
export PYTHONPATH="$CRM/source/scripts:$CRM/source/src:${PYTHONPATH:-}"
export FDM_RUNTIME_FINGERPRINT=${CRM_FINGERPRINT:-$CRM/runtime_fsi.json}   # regenerate for chrono-build-fsi
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NGPU=$(rocminfo | grep -c -E '^\s+Name:\s+gfx')
[ "${NGPU:-0}" -ge 1 ] || { echo "no GPU visible on $(hostname)"; exit 2; }
WPG=${WORKERS_PER_GPU:-1}
NW=$(( NGPU * WPG ))
CPUS=${SLURM_CPUS_PER_TASK:-16}
T=$(( CPUS / NW )); [ "$T" -gt 8 ] && T=8; [ "$T" -lt 1 ] && T=1
mkdir -p "$RUN"/{claims,runs,workers,logs,stale,runs_stale}
echo "host=$(hostname) job=${SLURM_JOB_ID} part=${SLURM_JOB_PARTITION} ngpu=$NGPU wpg=$WPG threads=$T"
rocm-smi --showproductname | grep -i -E "series|gfx" || true
pids=()
for g in $(seq 0 $((NGPU-1))); do for j in $(seq 0 $((WPG-1))); do
  (
    export ROCR_VISIBLE_DEVICES=$g HIP_VISIBLE_DEVICES=0      # process sees exactly one device
    export OMP_NUM_THREADS=$T
    exec "$NRD_PYTHON" -P -u "$CRM/source/scripts/crm_queue_worker.py" \
      --run "$RUN" --tasks "$RUN/tasks.json" --worker-id "${SLURM_JOB_ID}_$(hostname -s)_g${g}w${j}" \
      --threads "$T" --chrono-data "$CHRONO_BUILD/data" \
      --deadline-unix $(( $(date +%s) + ${CRM_BUDGET_S:-13200} )) --episode-budget-s ${CRM_EPISODE_BUDGET_S:-900}
  ) > "$RUN/logs/worker_${SLURM_JOB_ID}_$(hostname -s)_g${g}w${j}.log" 2>&1 &
  pids+=($!)
done; done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "CRM_COLLECT_DONE rc=$rc"; exit $rc
```
Submit lines (one per partition; array index is only used to get N nodes, workers pull from the shared queue):
```bash
S=$CRM/source/scripts/crm_collect.sbatch; X=--export=ALL,CRM_RUN=$CRM/collect_v1
sbatch -p mi2101x -c 16  --array=0-22 -t 04:00:00 -J crm_c $X -o $CRM/collect_v1/logs/%x_%A_%a.out $S   # 23 GPUs
sbatch -p mi2104x -c 128 --array=0-3  -t 04:00:00 -J crm_c $X -o ... $S                                  # 16 GPUs
sbatch -p mi3508x -c 256              -t 04:00:00 -J crm_c $X -o ... $S                                  # 8 GPUs
sbatch -p mi3001x -c 16  --array=0-2  -t 04:00:00 -J crm_c $X -o ... $S
sbatch -p mi3501x -c 24  --array=0-5  -t 04:00:00 -J crm_c $X -o ... $S
sbatch -p mi2508x -c 128 --array=0-1  -t 04:00:00 -J crm_c $X -o ... $S                                  # queued behind 9 pending
```

### 8.2 Worker claim loop (core of `crm_queue_worker.py`)
```python
def atomic_json(path, obj):                      # same recipe as run_traverse_f104_queue.py:52-64
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f: json.dump(obj, f, indent=1); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def try_claim(run, task_id, me):
    if (run/'runs'/task_id/'episode_complete.json').exists(): return False      # resume: done
    try: os.mkdir(run/'claims'/task_id)                                         # atomic across nodes
    except FileExistsError: return False
    atomic_json(run/'claims'/task_id/'claim.json', dict(worker=me, job=os.environ.get('SLURM_JOB_ID'),
                host=socket.gethostname(), pid=os.getpid(), gpu=os.environ.get('ROCR_VISIBLE_DEVICES'), t=time.time()))
    out = run/'runs'/task_id
    if out.exists(): os.rename(out, run/'runs_stale'/f'{task_id}.{int(time.time())}')   # marker-less leftover
    return True

tasks = json.load(open(args.tasks)); n = len(tasks); off = int(hashlib.md5(me.encode()).hexdigest(), 16) % n
for k in range(n):
    t = tasks[(off + k) % n]
    if (run/'STOP_CLAIMS').exists() or time.time() + args.episode_budget_s > args.deadline_unix: break
    if not try_claim(run, t['id'], me): continue
    rc = subprocess.run([sys.executable, '-P', '-u', COLLECTOR, '--case', ..., '--route', ..., '--out', str(run/'runs'/t['id']), ...],
                        stdout=open(run/'logs'/f"{t['id']}.log", 'w'), stderr=subprocess.STDOUT, timeout=args.episode_timeout_s).returncode
    ledger.write(json.dumps(dict(id=t['id'], rc=rc, worker=me, wall_s=..., t=time.time())) + '\n'); ledger.flush(); os.fsync(ledger.fileno())
```
- One subprocess per episode (as in every rigid runner): a HIP fault or Chrono abort kills only that episode, GPU memory is
  released, and the 4 s terrain rebuild is small next to 80+ s of simulation.
- Failed episode (`rc != 0`, no marker): leave the claim in place (no automatic retry, same policy as
  `max_attempts=1`), list them from the ledger, and release deliberately with the janitor rename after triage.
- For group-paired tests replace `t['id']` by the group id and loop the arms inside the claimed unit.

### 8.3 Simpler fallback (zero new mechanism)
Static shards exactly like gen_v1: precompute `shard` per row, one array task per node, inside the node a
`ThreadPoolExecutor(max_workers=NGPU*WPG)` whose submit function takes a GPU index from a `queue.Queue` of free indices
and passes `ROCR_VISIBLE_DEVICES` in the subprocess `env`. Skip-if-marker gives resume. Downside: per-partition task files
(one-partition rule) and poor balance across GPU speeds (0.17 vs 0.25 rtf) and episode lengths (5-120 s); catch-up
task files were needed even in the rigid run.

## 9. Accounting

`ssh amd slurm_balance2.py` (2026-09-16 23:2x CDT; ledger `/share/accounting/allocation_usage.json` refreshed 23:00):
```
-> alloc_dannegrut_06222026_06302027:  Used   396.0 of   1500 total ( 26.4 %)
billing_total 396.02   node_hours_total 1056.28 (unweighted)
users: slaton 255.74 | harry 47.20 | dannegrut 38.26 | auc7us 31.96 | kyle 22.87
```
- Remaining 1104 billed node-hours; allocation ends 2027-06-30; QOS `GrpTRESMins billing=90000` (= 1500 h).
- harry's unweighted node-hours since 06-22 (`sacct`): mi3501x 122.4, mi2101x 114.9, mi2104x 19.6, mi3001x 8.1,
  mi3508x 2.2, mi2508x 1.7, mi3008x 0.7 — total 270 node-h -> 47.2 billed. harry stood at 16.3 on 2026-09-07, so everything
  since then (the entire f104 rigid line plus other work) cost <= 31 billed hours. Only `slurm_balance2.py` is correct (v1/v3 read decaying fairshare usage, v4 crashes).
- Post-mortems: `sacct -u harry -S <t> -X --format=JobID,JobName%20,Partition,NCPUS,Elapsed,State,SubmitLine`
  (`SubmitLine` recovers the exact sbatch command of any past job).

## 10. Gotcha checklist

1. `flock` + shared `state.json` on `/work1` broke with 42 node processes (4.1). Use mkdir claims or static shards.
2. One partition per `sbatch`; `-t` mandatory; 4 h cap on mi3501x/mi3001x, 30 min devel; never `--mem`.
3. Pass `-c <all CPUs>` on the command line; worker/thread counts derive from `SLURM_CPUS_PER_TASK`.
4. Whole-node billing: fill every GPU of a multi-GPU node; mi2101x/mi2104x/mi2508x are the cheapest per GPU-hour.
5. `OMP_NUM_THREADS` is set by Slurm from `-c`; always override per worker.
6. `gen_collect.py:270` refuses a dirty out dir — killed episodes must be moved aside before re-running.
7. `FDM_RUNTIME_FINGERPRINT` and `--chrono-data` are pinned to `chrono-build`; both must be re-pointed to `chrono-build-fsi`.
8. `gen_collect.py:83-99 validate_native_height` calls `scene.terrain.GetHeight` on a `RigidTerrain` and enforces
   p95 <= 0.08 m / max <= 0.15 m vs the BMP; `:199-206` calls `scene.terrain.GetNormal` — RIGID-SPECIFIC hooks that the
   queue verifier then demands (`summary['native_height_valid']`, `tire_*_force_projected_terrain_normal_n`). The contract
   `physics` block (`freeze_production.py:55`: `'terrain':'RigidTerrain height map'`, `solver_dt_s .002`, `tire TMEASY`)
   is descriptive text but will be wrong for CRM. (Details belong to the episode-runner scout; listed here because the
   cluster-side verifier enforces them.)
9. Cross-node determinism: rigid Chrono is bit-identical per node type only (k003 vs k005 differ); keep paired arms on one
   node/GPU and record host + gfx arch. GPU SPH determinism is untested beyond tonight's 4 s coincidence.
10. bare `python`/`python3` = 3.9 and cannot import pychrono; always `$NRD_PYTHON`. Cluster trees are rsync copies.
11. torch on MI210 compute nodes is broken with pytorch/2.10.0; planner-in-the-loop CRM tests on mi2101x/mi2104x must run
    the network on CPU (as nav_v1 did) or go to MI3xx nodes (`module load pytorch/2.10.0` is rocm 7.1 vs Chrono's 7.2 —
    keep torch in a separate process if both are needed on GPU).
12. Sibling agents are using mi3501x right now (jobs 423543-46, `hill_*`); "jobs in queue" counts them toward the filter's
    max-job check.
