# Rigid launch path for A4 (branch_auto) and B3 (pid_perturbed) + the CRM B3 twin

Written 2026-09-21 23:10. Nothing was submitted. `G` = `/work1/dannegrut/harry/experiments/generalist_20260921`,
`R` = `/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909`, `C` = `/work1/dannegrut/harry/experiments/crm_f104_20260916`.
Files of this round (all under `artifacts/traverse/generalist_20260921/C_collectors/launch/`): `build_rigid_tasks.py`
(builder, run locally), `tasks/{tasks_a4_rigid,tasks_b3_rigid,tasks_b3_crm}.json` + `tasks/build_summary.json`,
`verify_cluster_paths.py` + `verify_cluster_paths_20260921.txt` (cluster-side check, output), `check_only_cluster_20260921.txt`
(the full `--check-only` outputs), `submit_rigid_a4_b3.sh` (the submit lines below, not executed). No repo file was edited.

## 1. What was shipped

- `G/source/{src,scripts,assets}` = rsync of the local worktree (`--exclude __pycache__`; scripts 415, src 85, assets 305 files).
  Before this round the cluster copies of `gen_collect_ext.py`, `gc_control.py`, `crm_collect_ext.py` and `gen_array_g.sbatch`
  were stale (different md5); now all equal the local files: `gen_collect_ext.py` sha256 `b9f36a02...`, `gc_control.py`
  `683f24ad...`, `gen_collect.py` `b6ba0622...`, `f104_n2_sampler.py` `c8018449...`, `gen_runner_g.py` `b47c9fd5...`,
  `gen_array_g.sbatch` `e3bcb4d8...`, `crm_collect_ext.py` `cbf57f7f...`.
- `G/source/source_manifest.json`: generated on the cluster with exactly recipe (a) of the inventory map
  (`os.walk` from `G/source`, sha256 of every file, `__pycache__` and the manifest itself excluded): **805 files**
  (scripts 415, src 85, assets 305, nothing else); manifest sha256 `5fcbe6693752bd5221b20b72b4119be733298456a83728e8bcc0b9a6d14b99e6`.
- Frozen-source comparison against the recordings (all 800 A4 recordings carry one identical `source_sha256` set, equal to
  `R/source_v1/source_manifest.json`): 8 of the 9 `gen_collect.SOURCE_FILES` in `G/source` are byte-identical to the recording
  source; `src/nedm/traverse/scene.py` differs (G `cb164313...`, source_v1 `b4f4f6b8...`). The diff is the 09-16 nav_v1
  change only: a `with_rgb: bool = True` render option that wraps the overhead RGB camera construction in `if render.with_rgb:`;
  with the default the executed statements are identical, and the rigid collector never renders (`record_rgbd_stride 0`,
  `render_parity False`). Physics, driver and integration code are unchanged, so the A4 prefix replay is expected to
  reproduce the recordings within the 0.5 m tolerance as it did locally (0.095 m at 3 s); the replay comparison in every
  `branch_auto.json` is the actual evidence, and `skipped.json` reasons give the drop count.
- Launch files: `G/source/scripts/gen_array_g.sbatch` and `G/source/scripts/gen_runner_g.py` (where the sbatch's own
  comment and the `GEN_RUNNER` default point), plus byte-identical convenience copies `G/gen_array_g.sbatch`, `G/gen_runner_g.py`.
- Environment handling verified against `gen_array.sbatch` (diff = the comment block, the four `GEN_*` defaults and the
  runner path only): `source env.sh; nrd_pychrono; nrd_use_lavapipe`, the four thread variables = 1,
  `FDM_RUNTIME_FINGERPRINT=R/pilot_runtime_412394.json` (exists; 108 runtime entries, contains `_vehicle.so` and
  `/vehicle/hmmwv/` so the collector's gate passes), chrono data `/work1/dannegrut/harry/nrd/chrono-build/data`
  (= `gen_runner.py` line 27; `.../data/vehicle/hmmwv` exists), `GEN_WORKERS = CPUs - 2`, `GEN_TIMEOUT_S` 3600 (x4 for
  branch_auto rows). `GEN_ROOT` is passed explicitly in every submit line (P6).
- Output roots pre-created (empty): `G/rigid_a4/logs`, `G/rigid_b3/logs`, `G/crm_b3/logs` (sbatch `-o` needs the directory).

## 2. `--check-only` on the login node (login1, `$NRD_PYTHON` = venv python 3.12.12 after `env.sh`/`nrd_pychrono`, the job's interpreter; full text in `check_only_cluster_20260921.txt`)

Common arguments: `--source-root G/source --chrono-data /work1/dannegrut/harry/nrd/chrono-build/data --horizon-s 120 --check-only`,
`PYTHONPATH=G/source/scripts:G/source/src`, `FDM_RUNTIME_FINGERPRINT` as above (not consulted by check-only, which returns before the gate).

| # | mode | arguments | result |
|---|---|---|---|
| 1 | `native` | case `R/cases_night2_v1/f104_v2_group_0063.json`, route `.../routes/f104_v2_group_0063/route_01.json` | rc 0; gates `source_manifest: checked`, `runtime_fingerprint: checked`; manifest sha `5fcbe669...`; adapter `gen_collect_ext.replacement_adapter`, hook_count `{insertions: 6, replacements: 1}`, original `run_chrono` `f060168d...`, adapted `8cf1bd50...`, gen_collect's adapted `162c594c...`; arena bmp `5d5bc683...` (allowlisted); split test |
| 2 | `branch_auto` | same case/route, `--recorded R/production_v3/runs/f104_v2_group_0063_route_01 --branch-frame 40 --n-cont 3 --cont-seed 3840197225` (the first A4 row) | rc 0: `{"check_only": true, "mode": "branch_auto", "recorded": ".../production_v3/runs/f104_v2_group_0063_route_01", "branch_frame": 40}` (validates the argument set and the recorded dir; the manifest/arena/hook gates are the ones of row 1, which the replay and continuation subprocesses run through) |
| 3 | `pid_perturbed` | train group `f104_v2_group_0000` route_00, `--episode-seed 12345 --near-stop-s 40` | rc 0; same gates/hook counts; `ext.perturb_seed 12345`, `near_stop_s 40.0`; split train |
| 4 | `pid_perturbed` without a seed | as 3 minus the seed | rc 1, `ValueError: pid_perturbed needs --perturb-seed or --episode-seed (no default seed: every episode must have its own)` (fails closed, as designed) |
| 5 | legacy `gen_collect.py --check-only` against `G/source` (PLAN R3) | as 1 | rc 0; manifest `5fcbe669...`; adapter hook_count 3, original `f060168d...`, adapted `162c594c...` |

No output directory was created by the checks; after importing the collector neither `torch` nor `pychrono` is in `sys.modules`.

## 3. Task files (`G/tasks/`, local copies in `launch/tasks/`; `tasks/build_summary.json`)

| file | rows | sha256 | content |
|---|---|---|---|
| `tasks_a4_rigid.json` | **800** | `a4bc4dab...` | one `branch_auto` row per rigid anchor (700 train / 50 val / 50 test; 480 clean_moving / 320 low_progress; 412 designed routes from `production_v3`, 388 on-policy from `production_v4`; F 40..2018 frames, always < recorded frames and < 120 s) |
| `tasks_b3_rigid.json` | **1,500** | `64489178...` | `pid_perturbed` on 1,500 distinct (train group, designed route) pairs: groups drawn uniformly from the 1,089 train groups (852 distinct, at most 7 rows per group), route uniformly from route_00..11 (97-143 per index), `numpy.default_rng(20260921)` |
| `tasks_b3_crm.json` | **1,500** | `4060ad03...` | the SAME ids, pairs and seeds for `crm_worker.py`, paths relative to `CRM_ROOT=C` |

Row conventions. A4: `{id: <episode>__a4, group, case: R/cases_night2_v1/<g>.json, route: R/cases_night2_v1/routes/<g>/route_XX.json or
R/cases_night2_onpolicy_v1/routes/<g>/op_XX.json, run, tier = index, arena: f104, shard = int(md5(group), 16) % 6, mode: branch_auto,
n_cont: 3, extra: [--branch-frame F, --n-cont 3, --cont-seed int(md5(anchor_id)[:8], 16), --recorded R/production_v{3,4}/runs/<episode>]}` plus
informational `anchor_id, episode, split, cls, F, cont_seed, route_kind, case_sha256, route_sha256, recorded_frames, recorded_status`.
The runner passes `--horizon-s 120` itself; `n_cont` gives the row the 4 x 3600 s timeout; all continuations of an anchor run in the
row's own subprocesses on the shard's node. B3 rigid: `{id: <g>_route_XX__b3, ..., mode: pid_perturbed, extra: [--episode-seed
int(md5(id)[:8], 16), --near-stop-s 40]}` (no `episode_seed` key on rigid rows, per the inventory's convention). B3 CRM:
`{id, group, case: cases/night2/<g>.json, route: cases/night2/routes/<g>/route_XX.json, run, tier, episode_seed: <the same seed>,
extra: [--mode, pid_perturbed, --near-stop-s, 40]}`. `crm_collect_ext.py` has NO `--perturb-seed` flag: its perturbation is
seeded from `--episode-seed`, which `crm_worker.py` appends from the `episode_seed` key (lines 95-96), so the `--perturb-seed`
suggested in the brief was deliberately not used; the near-stop rule is active by default in `pid_perturbed` (`--near-stop-s`,
default 40; `--no-near-stop` disables it), the explicit `--near-stop-s 40` only documents it. `crm_worker.py` forwards
`--crm-config C/configs/crm_main.json` because the collector path contains `crm_collect`.

Assertions in the builder (all passed): twin split from `twin_crm.npz` = 1,089 train / 56 val / 55 test groups; every anchor's
split equals the twin split; no id/group with a planner-suite prefix (`f104_crm_eval_group_`, `f104_g1_test_group_`,
`f104_pair_group_`) in any file; no val/test group in either B3 file; 1,500 seeds unique, identical across the two B3 files and
disjoint from the 800 A4 continuation seeds; every A4 `route_sha256` equals the recording's; the local case files equal the
recordings' `case_sha256`.

Cluster verification (`verify_cluster_paths.py`, `verify_rc=0`, **0 misses**): all 800 A4 case, route and recorded dirs exist
(`trajectory.npz`, `outcome.json`, `collection_request.json`, `episode_complete.json`); sha256 of the cluster case/route files
== the row's == the recording's `outcome.json` hashes; recorded frames == the anchor's `n_frames` > F; recorded statuses 506
goal_reached / 267 prolonged_blockage_terminated / 23 timeout / 3 terrain_bounds_exit / 1 rollover; all 1,500 rigid rows (under
R) and all 1,500 CRM rows (under C) exist with matching hashes and `split == train`. `C/cases/night2` and `R/cases_night2_v1`
are byte-identical copies. `C/source/scripts/crm_worker.py` md5 equals the local/G copy (handles `extra` and `episode_seed`).

## 4. Submit lines (NOT run; also in `submit_rigid_a4_b3.sh`)

```bash
G=/work1/dannegrut/harry/experiments/generalist_20260921; C=/work1/dannegrut/harry/experiments/crm_f104_20260916
# rigid A4 (mi2104x, 24 h cap; command-line -t overrides the sbatch's 03:00:00 header)
mkdir -p $G/rigid_a4/logs
sbatch --parsable -p mi2104x -c 128 -t 06:00:00 --array=0-5 -J rigid_a4 \
  --export=ALL,GEN_ROOT=$G,GEN_TASKS=$G/tasks/tasks_a4_rigid.json,GEN_OUT=$G/rigid_a4,GEN_COLLECTOR=$G/source/scripts/gen_collect_ext.py \
  -o $G/rigid_a4/logs/%x_%A_%a.out $G/source/scripts/gen_array_g.sbatch
# rigid B3
mkdir -p $G/rigid_b3/logs
sbatch --parsable -p mi2104x -c 128 -t 06:00:00 --array=0-5 -J rigid_b3 \
  --export=ALL,GEN_ROOT=$G,GEN_TASKS=$G/tasks/tasks_b3_rigid.json,GEN_OUT=$G/rigid_b3,GEN_COLLECTOR=$G/source/scripts/gen_collect_ext.py \
  -o $G/rigid_b3/logs/%x_%A_%a.out $G/source/scripts/gen_array_g.sbatch
# CRM B3 twin (C/source sbatch: CRM_ROOT=C hard-coded, FSI build; MI350X partitions only; 4 h windows, resumable)
S=$C/source/scripts/crm_collect.sbatch
X="--export=ALL,CRM_TASKS=$G/tasks/tasks_b3_crm.json,CRM_OUT=$G/crm_b3,CRM_CONFIG=configs/crm_main.json,CRM_COLLECTOR=$G/source/scripts/crm_collect_ext.py,CRM_BUDGET_S=13400"
mkdir -p $G/crm_b3/logs
sbatch --parsable -p mi3501x -c 24  -t 04:00:00 --array=0-5 -J crm_b3 $X -o $G/crm_b3/logs/%x_%A_%a.out $S
sbatch --parsable -p mi3508x -c 256 -t 04:00:00 --array=0-1 -J crm_b3 $X -o $G/crm_b3/logs/%x_%A_%a.out $S
```

Shapes: the rigid array runs `gen_runner_g.py` with 126 single-threaded collector subprocesses per node; shard sizes are
109-158 (A4) and 196-294 (B3) rows, so each shard is one to three waves of at most ~8 min of simulated time per process
(A4 rows: replay of F frames + three continuations of at most 120 s, sequential) — expected wall well under the 6 h asked
for (the local rate is ~0.5x real time single-threaded). The CRM shape copies today's A0/pass-1 launches (`sacct` 430402
mi3501x `-c 24`, 430403 mi3508x `-c 256`, 4 h; PLAN budget ~5 billed CRM for B3). Resume = resubmit (both runners skip rows
with `episode_complete.json`; the CRM worker also re-claims stale claims). Balance at 23:10: 519.4 / 1500 used.

Preconditions and notes for the submitter:
- CRM twin: PLAN R2 requires the `crm_collect_ext.py` cluster smoke (first `outcome.json` with `physics_dt_s == 0.001`) before a
  CRM launch; this round did not run one (no jobs). The rigid side needs nothing further: the check-only gates above are the
  PLAN R3 requirement.
- Queue: 7 of my array tasks are running now (a4pass1 430728, nrd_tag 430704); the rigid arrays add 12 tasks, the CRM twin 8.
- A4 rigid drops are expected for anchors whose replay ends before F, whose replayed pose leaves the 0.5 m tolerance or whose
  class flips (`runs/<id>/skipped.json`, reasons `replay_ended_before_branch_frame`, `replay_mismatch: ...`,
  `no_valid_continuation: ...`); report the count per reason after the job (`grep -o '"reason": "[a-z_]*' runs/*/skipped.json | sort | uniq -c`).
- Only the `mode`, `extra`, `case`, `route`, `shard`, `run`, `n_cont` keys drive the rigid runner; the other row keys are for the
  labeller (`ga_branch_dataset.py`) and the report.
