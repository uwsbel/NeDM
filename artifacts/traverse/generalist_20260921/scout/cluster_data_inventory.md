# Cluster operations and recorded-episode inventory (both worlds)

Scout, 2026-09-21, read-only. Local = `/home/harry/NeDM-traverse_mppi`; `C` = `/work1/dannegrut/harry/experiments/crm_f104_20260916`; `R` = `/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909`; `G` = `/work1/dannegrut/harry/experiments/generalist_20260921`.

## 1. Status

- Balance: **489.9 of 1500 billed node-hours used (32.7 %)**. The script is `/usr/local/bin/slurm_balance2.py` (on PATH); `python3 ~/slurm_balance2.py` as written in the task fails (no such file in `/home1/harry`). harry's unweighted node-hours since 09-16 (`sacct`): mi2101x 115.9, mi2104x 25.0, mi3501x 24.6, mi3001x 12.2, mi2508x 7.9, mi3508x 6.6, mi3008x 2.9.
- Idle now (`sinfo`): mi3501x 7, mi2104x 12, mi2508x 2, mi3508x 1, mi2101x 2, devel 4. Pending: mi3008x 39, mi3001x 13, mi3258x 5.
- Queue cap "max 50 queued jobs per user, array tasks count" is a note from the CRM night (`memory/crm-f104-night-state.md:46`); `sacctmgr` shows no MaxSubmit (lua filter; `MaxArraySize=1001`). `scripts/crm_launch.sh:15-21` submits 24+6+3+2+1+5+6 = 47 tasks, sized to it.
- Walltime caps (dry-run verified 09-16, `artifacts/traverse/crm_f104_v1/scout/cluster_ops.md:28-38,61`): mi3501x/mi3001x 4 h; mi2101x/mi2508x/mi3008x/mi3258x/mi3508x 12 h; mi2104x 24 h; devel 30 min. `-t`, `-A dannegrut`, one partition per sbatch, `-c <all CPUs>`, never `--mem`. The CRM collection used `-t 06:00:00` on mi2508x/mi3008x (`sacct` 423643/423645).

## 2. CRM world (all f104 arena)

| set | cluster | episodes (complete) | sim h | local copy |
|---|---|---|---|---|
| training collection | `C/collect_v1/runs` | **15,235** (15,235 have `trajectory.npz`+`command_reference.npz`+`episode_complete.json`) | **91.51** (`C/collect_v1/qa.json`: train 83.08 h/13,821/1,089 groups; val 4.33/713/56; test 4.10/701/55) | `artifacts/traverse/crm_f104_v1/collect_v1/runs` 15,235, **6 files, no marker**, 1.8 GB |
| planner eval | `C/eval_v1/runs` | 1,240 | 6.61 | `crm_f104_v1/eval_v1/runs` 1,240 |
| iterated / gradient planner | `C/night2/eval_iter_crm`, `eval_grad_crm` | 1,137 / 363 | 5.04 / 2.35 | `crm_night2_v1/planner/eval_*_crm/runs` 1,137 / 563 (4 files, no cmdref) |
| pilots A/B/C | `C/pilot/{A,B,C}/runs` | 144 (143) / 144 (143) / 31 (25) | 0.87 / 0.82 | `crm_f104_v1/pilot/*` |
| **total** | | | **~107 h** | |

- Episode file list (`C/collect_v1/runs/f104_v2_group_0000_op_00`, ~165 KB): anchor_state.npz 1.2K, case.json 13.8K, collection_request.json 2.5K, command_reference.npz 5.7K, crm_extra.npz 44.6K, episode_complete.json 1.1K, f104_episode.json 1.4K, initial_state_validation.json 0.7K, outcome.json 9.1K, reference.json 8.1K, trajectory.npz 76.5K. Arrays: `state (n,17) f32, action (n,3), pose (n,3), dt_s`; cmdref `interval_start_s, desired_speed_mps (n), reference_waypoints (m,2), stations, speeds, headings`; crm_extra `slip_ratio (n,4), fsi_force_wheel_fx_n (n,4), spindle_z, quat, pos_z, bmp_ground_z`.
- `C/moving_v1/out/runs` (1,080, 8.10 h) is **rigid** moving-start data driven through the CRM worker (`CRM_COLLECTOR=rigid_moving_collect.py, CRM_NO_GPU_BIND=1`, `sacct` 425329); 9 files, no cmdref/case.
- Inputs: `C/tasks_train.json` 24,000 rows = 1,200 groups x 20 (`{id, group, case, route, tier, episode_seed, run}`); `C/cases/night2` 1,200 cases + routes; `C/cases/night2_onpolicy` 1,200; `C/cases_eval/cases` 202; `C/configs/crm_main.json` (step 1e-3, spacing 0.08, depth 0.24, mbs_threads 4).
- Datasets `C/datasets/station_ds_crm_v1.npz` 471 MB; `C/night2/datasets/twin_{crm,rigid}.npz` 169/170 MB, `reanchor_{crm,rigid}.npz` 997/1015 MB. Checkpoints `C/train_v1/deploy/CRM_N2_s{0..4}.pt` (+json, logits); `C/night2/train/stage{A,B,C}` 14/36/21 files. Frozen code `C/source/{src,scripts,assets}` 2.4 MB, no manifest (`scripts/crm_collect.py:382-393` verifies none).

## 3. Rigid world

| set | cluster | episodes | sim h | cmdref | local files/ep |
|---|---|---|---|---|---|
| production_v2 (f104, 1,500 groups) | `R/production_v2/runs` | 11,412 | 66.42 | yes | 16 |
| production_v3 (f104 designed 1,200x12) | `R/production_v3/runs` | 14,400 | 82.98 | yes | 5 |
| production_v4 (f104 on-policy 1,200x8, mean 45 s) | `R/production_v4/runs` | 9,600 | 119.97 | yes | 5 |
| gen_v1 data | `R/gen_v1/data/runs` | 9,000 (1,800 each g203/g216/g217/g228/g231) | 55.93 | yes | 5 |
| gen_v1 test | `R/gen_v1/test/runs` | 6,639 (f104 1,103; g203 1,098; g216 1,109; g217 1,132; g228 1,111; g231 1,086) | 37.14 | yes | 14 |
| night2 evals | `R/gen_v1/night2_eval_{f104_fixed2,g216,g231}` | 1,000/961/941 | 5.94/3.67/3.39 | yes | 14 |
| closed-loop + sensor tests | `R/night2_{closed,ext,haz,tilt}_v1`; local `sensor_v1/test{,2}` | 4,086; 12,187 | 22.3; 56.4 | no | |
| **total** | | | **~467 h** (375 h with cmdref; f104 with cmdref ~281 h) | | |

- Cluster dirs are complete (18 files incl. marker + `collector.log`, `R/production_v4/runs/f104_v2_group_0000_op_00`); local production copies hold 5 label files (`artifacts/traverse/fdm_f104_50h_20260909/production_v{2,3,4}/runs`: 11,412/14,400/9,600; 2.1/1.2/1.3 GB).
- CRM ids equal rigid ids of the same route (`scripts/crm_tasks.py:6-7`): `collect_v1` pairs 1:1 with production_v3+v4 (24,000 rows); 211 CRM routes have no rigid twin (`crm_f104_v1/REPORT.md:148`).
- Rigid checkpoints `R/gen_v1/models/N2_s{0..4}.pt`; missions `R/gen_v1/missions/{f104,g203..g231}`; routes `R/gen_v1/night2_iter_*`. `/work1` 116 TB free; local disk 409 GB free.

## 4. Environment and templates

- `/work1/dannegrut/harry/nrd/env.sh:3-5` `CHRONO_BUILD=chrono-build`; `:19-21` `NRD_PYTHON=/work1/dannegrut/harry/venvs/nedm/bin/python` (3.12.12); `:38` `NRD_PYSTACK=/share/sw/ai/pytorch/2.10.0`; `:41-43` `nrd_pychrono`. The venv has **no numpy/torch itself** (verified `ModuleNotFoundError`); they come from the module tree (`torch 2.10.0+rocm7.1`; `module pytorch/2.10.0` pulls `rocm/7.1.0`). Torch is broken on MI210 compute nodes (`memory/amd-hpcfund-cluster.md`): train on mi3501x/mi3508x.
- CRM build `$NRD_ROOT/chrono-build-fsi` (7 pychrono modules incl. `_fsi.so`; `libChrono_fsisph.so` gfx90a/gfx942/gfx950; ROCm 7.2 by RPATH). Rigid build `chrono-build` + fingerprint `R/pilot_runtime_412394.json` (`scripts/gen_array.sbatch:12`; required by `scripts/gen_collect.py:271-272`).
- **CRM = one collector subprocess per GPU**: `scripts/crm_collect.sbatch` (md5 equal to `C/source/scripts/crm_collect.sbatch`) execs one `crm_worker.py` per node (`:17`); it starts `CRM_GPUS` threads (`:14`, `rocminfo` count or `CRM_GPUS_OVERRIDE`), each running one `crm_collect.py` subprocess with `ROCR_VISIBLE_DEVICES=g HIP_VISIBLE_DEVICES=0` (`scripts/crm_worker.py:75`), `OMP_NUM_THREADS=CRM_OMP` (`:76`; sbatch `:15` = 4). mkdir claims (`:47`), stale after `CRM_STALE_S` 1500 s (`:23,50`), `CRM_MAX_ATTEMPTS` 2 (`:24`), episode timeout 2400 s (`:26`), stop at `CRM_BUDGET_S` or `STOP_CLAIMS` (`:25,84`), 3 failures retire a GPU (`:131`). Fed by env vars `CRM_TASKS, CRM_OUT, CRM_CONFIG, CRM_COLLECTOR` (`:17-21,91`). **`CRM_ROOT` is hard-coded** (`crm_collect.sbatch:11`; `crm_launch.sh:5`) and all case/route/config/collector paths resolve against it (`crm_worker.py:16,91-99`).
- **Rigid = array over shards**: `scripts/gen_array.sbatch` (`-t 03:00:00`; `GEN_ROOT=R/gen_v1` hard-coded `:11-13`; `GEN_WORKERS=CPUs-2` `:15`) runs `gen_runner.py`: `GEN_TASKS/GEN_OUT` (`:11-12`), rows with `shard == SLURM_ARRAY_TASK_ID` (`:14-15`), skip-if-marker (`:23`), hard-coded `$GEN_ROOT/gen_collect.py`, `source`, `chrono-build/data` (`:25-27`). Precedent: `sbatch --parsable -p mi2101x -c 16 -t 03:00:00 --array=0-11 -o .../arr_%A_%a.out --export=ALL,GEN_TASKS=...,GEN_OUT=... gen_array.sbatch` (`sacct` 425467).
- Rigid through the CRM worker packed **one worker per node** (`C/moving_v1/out/workers/*_gpu0.json` only, since `CRM_GPUS`=1 on mi2101x) and used the FSI build's pychrono/data (`crm_collect.sbatch:9,12`), not the fingerprinted rigid build.
- **Torch training**: `artifacts/traverse/crm_night2_v1/cluster_train/n2_train.sbatch` (mi3501x, `-c 24`, 3.5 h, `--array=0-6` `:5-9`; `source /etc/profile; module load pytorch/2.10.0; source venv/bin/activate` `:13-15`; MIOpen cache in /tmp `:16`; 4 concurrent `python3.12` runs per MI350X `:32`; tasks path hard-coded `:22`). Single run: `scripts/crm_train.sbatch:7` (`CRM_DS, CRM_TRAIN_OUT, CRM_MODE, CRM_SEEDS, CRM_TAG, CRM_EXTRA`; `cd $C/source/scripts` hard-coded `:15`). 8-per-node: `/work1/dannegrut/harry/experiments/mppi_claude_20260908/slurm/mppi_train_suite.sbatch:24-34`. CPU dataset build: `scripts/crm_dataset.sbatch:6,14`.

## 5. Recipes (from the local repo root)

```bash
G=/work1/dannegrut/harry/experiments/generalist_20260921; C=/work1/dannegrut/harry/experiments/crm_f104_20260916
# (a) code up (rsync copies, not git); gen_collect.py needs source/source_manifest.json = {"files":{relpath:sha256}} (gen_collect.py:237-242)
ssh amd "mkdir -p $G/source $G/cases $G/configs $G/train/logs && cp $C/configs/crm_main.json $G/configs/"
rsync -az --exclude __pycache__ src/ amd:$G/source/src/; rsync -az --exclude __pycache__ scripts/ amd:$G/source/scripts/; rsync -az assets/ amd:$G/source/assets/
rsync -az artifacts/traverse/generalist_20260921/cases/ amd:$G/cases/
ssh amd "cd $G/source && python3 -c \"import hashlib,json,os;f={os.path.relpath(os.path.join(d,n)):hashlib.sha256(open(os.path.join(d,n),'rb').read()).hexdigest() for d,_,ns in os.walk('.') for n in ns if '__pycache__' not in d and n!='source_manifest.json'};json.dump({'files':f},open('source_manifest.json','w'),indent=1)\""
ssh amd "sed -i 's#^export CRM_ROOT=.*#export CRM_ROOT=$G#' $G/source/scripts/crm_collect.sbatch; sed -i 's#^CRM=.*#CRM=$G#' $G/source/scripts/crm_launch.sh"
ssh amd "sed 's#^export CHRONO_BUILD=.*#export CHRONO_BUILD=\$NRD_ROOT/chrono-build\nexport FDM_RUNTIME_FINGERPRINT=/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/pilot_runtime_412394.json#' $G/source/scripts/crm_collect.sbatch > $G/source/scripts/rigid_collect.sbatch"

# (b) drives. Rows {id, group, case, route, tier, run[, episode_seed, config]}, paths relative to $G; rigid rows must NOT carry
#     episode_seed (gen_collect.py has no such flag; worker.py:95-96 would pass it).
scp tasks_crm.json tasks_rigid.json amd:$G/
S=$G/source/scripts/crm_collect.sbatch; X="--export=ALL,CRM_TASKS=$G/tasks_crm.json,CRM_OUT=$G/crm_v1,CRM_CONFIG=configs/crm_main.json,CRM_BUDGET_S=13400"
ssh amd "mkdir -p $G/crm_v1/logs && sbatch --parsable -p mi3501x -c 24 -t 04:00:00 --array=0-6 -J crm_g $X -o $G/crm_v1/logs/%x_%A_%a.out $S \
  && sbatch --parsable -p mi2104x -c 128 -t 04:00:00 --array=0-5 -J crm_g $X -o $G/crm_v1/logs/%x_%A_%a.out $S"
#   all partitions (47 tasks): ssh amd "bash $G/source/scripts/crm_launch.sh $G/tasks_crm.json $G/crm_v1 configs/crm_main.json 4"
SR=$G/source/scripts/rigid_collect.sbatch; XR="--export=ALL,CRM_TASKS=$G/tasks_rigid.json,CRM_OUT=$G/rigid_v1,CRM_COLLECTOR=$G/source/scripts/gen_collect.py,CRM_NO_GPU_BIND=1,CRM_OMP=1,CRM_BUDGET_S=9800"
ssh amd "mkdir -p $G/rigid_v1/logs && sbatch --parsable -p mi2101x -c 16 -t 03:00:00 --array=0-9 -J rigid_g $XR,CRM_GPUS_OVERRIDE=14 -o $G/rigid_v1/logs/%x_%A_%a.out $SR"
#   (mi2104x: -c 128, CRM_GPUS_OVERRIDE=126). Resume = resubmit. Progress: ssh amd "ls $G/crm_v1/runs | wc -l; cat $G/crm_v1/workers/*.json"

# (c) training: edit n2_train.sbatch lines 10 (-o dir), 18 (CRM=, N2=$G), 22 (tasks path), 27 (trainer) locally, then
scp train/n2_train.sbatch train/tasks.json amd:$G/train/ && ssh amd "cd $G/train && sbatch n2_train.sbatch"
#   single run: ssh amd "sbatch -p mi3501x -t 02:00:00 --export=ALL,CRM_DS=$G/datasets/x.npz,CRM_TRAIN_OUT=$G/train/x,CRM_TAG=GEN $G/source/scripts/crm_train.sbatch"  (after sed CRM= in it)

# (d) results down, label files only
rsync -az --include '*/' --include 'trajectory.npz' --include 'command_reference.npz' --include 'outcome.json' --include 'case.json' \
  --include 'anchor_state.npz' --include 'crm_extra.npz' --include 'episode_complete.json' --exclude '*' amd:$G/crm_v1/runs/ artifacts/traverse/generalist_20260921/crm_v1/runs/
```

Disk cost: label subset 118 KB/episode (CRM, 6 files) or 83-135 KB (rigid, 5 files); full dirs ~165 KB (CRM) / ~215 KB (rigid). Everything above already sits locally (5.9 GB `crm_f104_v1`, 18 GB `fdm_f104_50h_20260909`); 50 new simulated hours (~8,000 episodes) is 1-1.7 GB.

## 6. Gaps / unknowns

- No record of the CRM result pull (no rsync line in `crm_f104_v1/LOG.md`); the include filter is inferred from the local layout.
- Local CRM `collect_v1`, rigid `production_v3/v4`, `gen_v1/data` lack `episode_complete.json`; provenance hashes exist only on the cluster.
- `gen_v1/missions_run` (600) and `nav_v1` hours not summed.
- The 50-job cap is a memory note, not re-verified today (no dry-run submissions made).
- Rigid episodes driven with the FSI build's pychrono (moving_v1 path) vs `chrono-build` production episodes: bit-identity untested. CRM cross-GPU determinism is still a 4 s coincidence (`cluster_ops.md:278-279`).

## 7. What must change for the plan

- Plan B1 has `command_reference.npz` for 15,235 CRM + 35,412 f104 rigid production + 15,639 sibling-arena episodes, but NOT for `moving_v1`, closed-loop or sensor tests.
- New drives need `CRM_ROOT`/`CRM=` re-pointed (`crm_collect.sbatch:11`, `crm_launch.sh:5`) and a rigid sbatch variant (`CHRONO_BUILD=chrono-build`, fingerprint, `CRM_GPUS_OVERRIDE`), or per-partition shard files for `gen_array.sbatch` (`GEN_ROOT` hard-coded `:13`, `gen_runner.py:25-27`).
- Plan A4 (prefix branches in CRM) has no collector: `crm_collect.py` always starts from fresh soil; `rigid_moving_collect.py --v0` is rigid-only.
- Training on mi3501x (4 h cap, 7 idle) using the `n2_train.sbatch` shape; budget ~1,010 node-hours left (50 h CRM cost ~37, `crm_f104_v1/REPORT.md:17`; night 2 cost 21.9, `crm_night2_v1/LOG.md:27`).
