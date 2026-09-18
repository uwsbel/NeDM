# Scout report: planner + held-out single-goal evaluation (gen_v1 / sensor_v2 / night-2), for the CRM port

Read-only reconnaissance, 2026-09-16. Repo `/home/harry/NeDM-traverse_mppi` (branch `traverse_mppi`), artifacts root
`R = artifacts/traverse/fdm_f104_50h_20260909`, cluster campaign `C = /work1/dannegrut/harry/experiments/fdm_f104_50h_20260909`.
Nothing was run, submitted or modified. Local `scripts/{gen_runner.py, gen_collect.py, gen_array.sbatch, f104_n2_sampler.py,
f104_n2_dataset.py, gen_riskmodel.py}` are md5-identical to the cluster copies under `C/gen_v1/` (checked). Local
`scripts/gen_planner.py` == `C/nav_v1/code/gen_planner.py` (newest; adds sensor_v1/v2 classes); `C/gen_v1/planner/gen_planner.py` is the
older pre-sensor version (mission runner only; single-goal picks never used it on the cluster).

## 0. Facts that contradict the brief (read first)

- **There is NO time term and NO risk/time mixing anywhere in the f104 planner.** Selection is `argmin` of the ensemble-mean route
  logit (`gen_planner.plan` L232-235, `gen_pools.py` L75-79, `sensor_pools_v2.py` L58-59, `nav_online.py` L378). `docs/progress.md`
  L361: "Selection. argmin predicted risk (no time term)". gen_v1 REPORT caveat: "Time is not in the planner's objective". Time to goal
  is only a reported metric. (The "time-only MPPI" lives in the other worktree `~/NeDM-mppi-claude`, not in this pipeline.)
- **Pessimistic ensemble and the abstain rule exist only in the night-2 test** (`scripts/f104_n2_pick.py`), not in
  `gen_planner.py`/gen_v1/sensor_v2/nav_v1:
  - pessimist: `risk(..., agg='max')` = max over the 5 seeds of the route logit, then argmin (L46-57, L78, L89). Result on 184 groups:
    0.0% fail / 0.0% unsafe vs mean-ensemble 0.5% / 0.5% (1 vs 0, n.s.); differed from the mean pick in 97/184.
  - abstain: `ABSTAIN = 0.05` (L26); if best OR second-best P(unsafe) >= 0.05, drive the anchor with the lowest model risk, else the
    argmin (L84-88). Pre-registered (night2 PLAN D5). **Never fired** (best candidate < 5% in all 184 groups).
- Picks are made **offline on the workstation** (torch GPU, conda env `nedm`: `/home/harry/miniconda3/envs/nedm/bin/python`, torch
  2.12+cu130), written as route JSON files, then rsynced to the cluster and only *driven* there. The cluster never re-scores.

## 1. Pipeline, end to end (as actually run)

```
gen_cases.py      -> <cases>/cases/<group>.json + routes/<group>/route_00..11.json + cases.json          (local)
gen_pools.py      -> <out>/picks/<group>.json, <out>/routes/<group>__<arm>.json, <out>/tasks_cluster.json   (local, GPU)
(one-off merge)   -> tasks_test.json  (run=True rows only, case/route paths relative to GEN_ROOT, shard = md5(group) % 48)
rsync to amd:$C/gen_v1 ; sbatch gen_array.sbatch -> gen_runner.py -> gen_collect.py per episode               (AMD)
rsync runs back   -> gen_v1/test/runs/<route_id>/{outcome.json,trajectory.npz,command_reference.npz,case.json,anchor_state.npz}
gen_analyze_test.py -> gen_v1/test_results.json                                                           (local)
```
sensor_v2 pilot is the same with `sensor_pools_v2.py` (pools) and `sensor_analyze_v2.py` (analysis); `sensor_eval_v2.py` is the
OFFLINE ensemble read-out on saved datasets (no Chrono).

### 1.1 Cases: `scripts/gen_cases.py`
CLI: `--groups N --seed S --prefix P --arena DIR --out DIR --strata {all,feature} --avoid DIR... --margin-m 2.0 --wave TAG`.
- Start candidates: 1 m grid on [-34, 34]^2 with `tmap.slope < tan(7 deg)` (L50-53), jitter +-0.35 m. Start footprint gate
  (`generate_traverse_f104_collection.footprint/start_ok`, 6x3 m rectangle): height range <= 0.65 m, max grade <= 12 deg, fitted grade
  <= 6 deg, plane residual <= 0.20 m. All from `TerrainMap` (heightmap only, no physics).
- `--strata feature` (used by every held-out test): round-robin over `tmap.features` (hills/craters from `arena_meta.json`); start
  9-36 m from the feature centre; lateral miss in {0, +0.65, -0.65} x sigma; goal = start + (dist + U(10,23) m) along the line, clipped
  to +-35 m; stratum name `<kind>_entry_cross_exit` or `<kind>_cross_slope`. Route length 24-85 m. Start yaw = direction to goal.
- `--avoid D...`: globs `D/*.json` (pass the directory that directly holds the group JSONs; `cases.json` is skipped by the
  try/except), rejects any new (start,goal) within `--margin-m` (4-D Euclidean) of an old pair; also enforces the margin among the new
  groups (L97-101). gen_v1 used 2 m; night-2 "fresh" used >= 4 m; hazard test >= 3 m.
- All 9 base routes (3 offsets x 3 speeds, `propose_route_families(..., speeds=[2,4,6], offsets=[0,-4,4], step_m=.5)`) must pass
  `validate_reference` with `MPPIConfig(arena_half_extent_m=36, max_speed_mps=6, max_curvature_inv_m=.10)` (L102-109).
- Writes 12 designed routes per group (3 offsets x {constant_2, constant_4, constant_6, smooth_2_6_2}); `route_00.json` = offset 0,
  constant 2 m/s = the **base route of every candidate pool**. Case keys used downstream: `id`, `split`, `arena` (repo-relative),
  `layout.{start_xy,start_yaw,assets=[]}`, `goal_xy`, `goal_radius_m=2.5`, `horizon_s=120`, `settle_reference`.
- GOTCHA: `split` is a hash (90/5/5) and is meaningless for test sets (e.g. `f104_g1_test_group_0000` has split "train"); the collector
  only requires it to be one of train/val/test. Do not filter test cases by `split`.
- Actual gen_v1 f104 test set: `R/gen_v1/cases_test_f104/cases/` (200 groups, prefix `f104_g1_test_group`, seed 20260915104, strata
  feature: 76 hill_cross_slope, 52 crater_cross_slope, 44 hill_entry_cross_exit, 28 crater_entry_cross_exit). sensor_v2 pilot:
  `R/sensor_v2/cases_g216`, `cases_g231` (100 each, prefix `<arena>_v2_group`, seed 20260919216...).
- Existing f104 start/goal pools to avoid (directory that holds the group JSONs): `R/cases` (501), `R/cases_reserve_v1` (1001),
  `R/cases_night2/cases` (1201), `R/cases_test_final` (185), `R/cases_ext_final` (523), `R/cases_haz_final` (300),
  `R/gen_v1/cases_test_f104/cases`, `R/sensor_v1/cases_f104/cases`, `R/sensor_v1/cases2_f104/cases` (201 each, incl. cases.json).

### 1.2 Candidate pools (256, shared across arms)
`scripts/gen_planner.py`: `N_CAND = 256`, `CFG = MPPIConfig(max_speed_mps=6, min_speed_mps=0, max_curvature_inv_m=.125,
arena_half_extent_m=40)` (L30-32); validator = `safe_validate` (L94-99; ValueError -> invalid) anchored at the start pose. Vehicle
footprint in the validator: half-length 2.6 m, half-width 1.3 m, +0.1 m margin, corners must stay within +-40 m; accel <= 1.5,
decel <= 2.0 m/s^2.
- **proposal pool** = `proposal_pool(base, pose, rng)` -> `f104_n2_sampler.propose` (L96-108): the 9 anchors first (offsets (0,-4,+4) m
  with a sin^2 lateral bump x speeds (2,4,6) m/s, meta `candidate='n2_anchor'`; order: index 0 = straight 2 m/s, 1 = straight 4, 2 =
  straight 6, 3.. = -4 m ...), then random `sample_one` until 256 valid or 8*256 tries: lateral = 3-mode sine basis, amplitudes
  N(0, 5.0/j) capped at 0.55*kappa_max*L^2/(j*pi)^2, clip +-10 m; speed = base 2 m/s + 4 smoothstep knots N(0,1.5) clip +-4, clipped
  to [0.5, 6] m/s, terminal cone v <= sqrt(2*2.0*(L-s)), forward accel 1.5 / backward decel 2.0 projection. No end envelope.
- **fixed-2 pool** = `fixed2_pool` (L130-138): same lateral family, `sp_sigma=0, base_speed=2.0`, up to 6*256 tries; geometry-only.
- Base route for single-goal tests = the case's `routes/<g>/route_00.json` (NOT `base_route()`; that fallback chain is for missions).
- Seeds (deterministic, per group): gen_v1 `md5(g+'gen_night2')[:8]` and `md5(g+'gen_fixed2')[:8]` (`gen_pools._seed` L30-31);
  sensor_v2 `'v2_proposal'`/`'v2_fixed2'`; night-2 `md5(g)`, `g+'n2'`, `g+'f2'`; hazard `'h1','h2','h3'`. Pools are rebuildable
  bit-for-bit from these (the 18 GB candidate caches were deleted on that basis; sensor_v2 replay reproduced the shipped test exactly).
- **Same pool for every arm**: each pool is built once per group, scored by every scorer, and each arm is an argmin (or a fixed anchor
  index) into that one list. `anchor_index(cands, 0.0, 6.0)` / `(0.0, 2.0)` finds the straight anchors (returns None if that anchor
  failed validation -> arm missing for that group).
- Ties: `np.argmin` takes the first index, so ties favour anchors.

### 1.3 Model input and scoring (the contract the pools feed)
- Corridor: `f104_n2_dataset.station_tensor(wp, sp, st)` -> `X` float32 `(5, 96, 32)`: 96 stations resampled uniformly in arc length,
  32 lateral samples over +-6 m; channels `[elev - e0, along-grade (clip +-2), cross-grade (clip +-2), commanded speed, valid]`, `e0` =
  elevation at the route start centre. Second return = route length (m). Stored pools use float16 for X.
- Map source for gen_v1: `gen_planner.set_map(arena_dir)` (L35-42): `TerrainMap.height_grid` flipped so row 0 = +y, stored as channel 3
  `/ ELEV_SCALE (10.0)`, `mpp = size_m / n` (80/512 = 0.15625 m), `ctr = (n-1)/2`. Same encoding as the static overhead depth map
  (on f104: identical pick 32/40, top-5 40/40, logit corr 0.987 vs the captured map).
- Context: `geom_ctx` (L149-151) float32 `(N,5)` = `[goal_dx, goal_dy, |goal-start|, start_yaw, route_len]`. No vehicle state.
- `RiskModel` (L154-184): loads `GEN_MODELS` env glob or default `R/night2_v1/final/N2_s*.pt` (5 seeds). Checkpoint keys required:
  `arch, layers, cin(=6), nctx(=5), state, norm{mu(4), sd(4)}, ctx_mu(5), ctx_sd(5)` (N2 also has `ctx_variant='none'`,
  `ctx_cols=[17..21]`, unused here). Per member: z-score channels 0-3, append a constant-ones channel (-> 6 ch), z-score ctx,
  `route_logit = log(sum_s softplus(hazard_s) + 1e-6)`. **Ensemble = mean of member logits**; `P = 1 - exp(-exp(z))`.
- `GridRiskModel`/`SensorRiskModel` (L268-323, sensor_v2 `matched_{H,H0,Dabs,Drel}_s{0,1,2}.pt`, 3 seeds): same net, inputs from
  `sensor_dataset_v2.tensor12` on a back-projected world grid (`set_grid_map(R/sensor_v2/grids/arena_f104_50h_v1)`; grid.npz exists
  locally for f104 + 5 siblings); extra ckpt keys `channels`, `norm.cont_index`. `matched_H` channels = `[z_rel, grade, cross, speed,
  valid]` (cin 6), `matched_Dabs` = `[range_abs, sec1, speed, valid]` (cin 5).
- Network: `scripts/gen_riskmodel.py` `Net(cin, nctx, arch='gru')`, 256k params.
- Hand rule: `gen_planner.HandRule` + `gen_terrain_features.route_features` (22 features of the same X) with
  `R/gen_v1/hand_rule.json` (`features, mu, sd, w`; pairwise-logistic fit by `scripts/gen_hand_rule.py` on
  `night2_v1/station_ds_all.npz` split==train, 33,840 routes, 102,167 pairs, within-group AUC 0.911). Higher = riskier.

### 1.4 Arms
gen_v1 (`gen_pools.py` L20, L75-79): `n2` argmin model logit on proposal pool; `rule` argmin hand-rule score on the SAME proposal
pool; `straight6` = anchor (0 m, 6 m/s) of the proposal pool; `n2_fixed2` / `rule_fixed2` argmin on the fixed-2 pool; `straight2` =
anchor (0 m, 2 m/s) of the proposal pool. sensor_v2 (`sensor_pools_v2.py` L48, L56-59): `straight6` + for each `NAME=glob` in
`--models`: `NAME` (proposal pool) and `NAME_fixed2` (fixed-2 pool). night-2 (`f104_n2_pick.py`): control/model/sampler/abstain/
anchor6/pessimist/fixed2_control/fixed2_new. hazard (`f104_n2_haz_pick.py`): control/sampler/sampler_noanchor/anchor6/fixed2_old/new.

### 1.5 Locking picks, dedup, sharding
- Within a group, arms that chose the same `(pool, index)` share ONE route file and ONE drive: `seen.setdefault((pool, idx), arm)`,
  `route_id = f'{g}__{first_arm_with_that_pick}'`, task row `run = first` (gen_pools L81-103). The pick summary of every arm carries
  the shared `route_id`; analysis reads `same_as = route_id.split('__')[1]`. Identical picks therefore tie exactly.
  gen_v1: 7,200 arm-picks -> 6,639 drives (5.5/group); sensor_v2: 1,400 -> 1,066.
- Shard = `int(md5(group).hexdigest(), 16) % n_shards` -> **all arms of a group are in one array task = one node**. Verified on the
  shipped task files: 6,639/6,639 rows match `% 48`, 0 groups span two shards; sensor_v2 1,066/1,066 match `% 4`.
- Task-row schema consumed by `gen_runner.py`: `{id, case, route, shard, run}` with `case`/`route` relative to `GEN_ROOT`
  (extra keys `group`, `arena` ignored). `gen_pools.py` writes `tasks_cluster.json` rows `{id, group_id, arena, arm, case(basename),
  shard, run}` WITHOUT `route`; the six per-arena files were merged by an unsaved one-off into `gen_v1/tasks_test.json` (run=True only,
  `case='cases_test_<arena>/cases/<g>.json'`, `route='test_<arena>/routes/<id>.json'`, re-sharded % 48). `sensor_pools_v2.py` writes
  runner-ready rows directly via `--case-prefix/--route-prefix` (`'../sensor_v2/cases_g216/cases'`, `'../sensor_v2/pilot_g216/routes'`,
  relative to `GEN_ROOT=gen_v1`, which is hardcoded in the sbatch).

### 1.6 Driving on AMD (one node per group)
`scripts/gen_array.sbatch` (= `C/gen_v1/gen_array.sbatch`): `-A dannegrut -N1 -n1 -t 03:00:00`; `source
/work1/dannegrut/harry/nrd/env.sh; nrd_pychrono; nrd_use_lavapipe`; `OMP/OPENBLAS/MKL/LP_NUM_THREADS=1`;
`FDM_RUNTIME_FINGERPRINT=$C/pilot_runtime_412394.json`; `GEN_ROOT=$C/gen_v1`; `PYTHONPATH=$GEN_ROOT/source/scripts:$GEN_ROOT/source/src`;
`GEN_WORKERS = SLURM_CPUS_PER_TASK - 2` (14); runs `$NRD_PYTHON -P -u $GEN_ROOT/gen_runner.py`.
Submit lines recovered from `sacct` (GEN_TASKS/GEN_OUT were exported in the shell and passed by `--export=ALL`):
```
cd $C/gen_v1
GEN_TASKS=$C/gen_v1/tasks_test.json GEN_OUT=$C/gen_v1/test \
  sbatch -p mi2101x -J gen_test -c 16 --array=0-47 -o $C/gen_v1/test/arr_%A_%a.out --export=ALL gen_array.sbatch      # job 420022
GEN_TASKS=$C/sensor_v2/tasks_pilot.json GEN_OUT=$C/sensor_v2/pilot_runs \
  sbatch -p mi2101x -J v2_pilot -c 16 --array=0-3 -o $C/sensor_v2/pilot_runs/arr_%A_%a.out --export=ALL gen_array.sbatch  # job 421193
```
A one-shard pilot preceded the main array (`-J gen_pilot --array=0`, job 420020). gen_v1: ~140 episodes per shard, 11-25 min per
shard on 16 cores. Unstarted shards 28-47 were cancelled and re-queued as two single-node tasks by rewriting the task file with
`shard=0` for the moved rows (whole shards moved, so groups stayed on one node; cf. `C/gen_v1/data_move.py`).
`scripts/gen_runner.py`: filters `rows` by `shard == SLURM_ARRAY_TASK_ID and run`, skips dirs with `episode_complete.json`, runs
`gen_collect.py --source-root $GEN_ROOT/source --case ... --route ... --out $GEN_OUT/runs/<id> --chrono-data
/work1/dannegrut/harry/nrd/chrono-build/data --horizon-s 120` in a `ThreadPoolExecutor(GEN_WORKERS)`, per-episode `timeout=3600`,
log `GEN_OUT/logs/<id>.log`, deletes `rich_telemetry.*`/`rich_intervals.npz` on success.
`scripts/gen_collect.py`: wrapper around the frozen runner `source/scripts/traverse_fdm_rgbd_diverse_chrono.py::run_chrono` with three
source-patched hooks (`adapted_function` L166-190); gates: source manifest hashes (L237-242), arena BMP sha in `gen_arenas.json` (L247-249),
`size_m == 80`, `layout.assets == []`, refuses to write into a dir that already has `outcome.json|collection_request.json|trajectory.npz`
(L270) — **a crashed episode dir must be moved away before a retry**. StopPolicy (L54-154): stall = all 41 endpoints of a 2 s window
within 0.25 m, throttle > 0.3 for all 40 intervals, not parked; not before 24 s; +2 s confirmation +8 s tail -> status
`prolonged_blockage_terminated` (earliest 34 s); `terrain_bounds_exit` if |x| or |y| > 40 m; frozen runner adds `rollover` (|roll| or
|pitch| > 60 deg), `goal_reached` (chassis within `goal_radius_m` 2.5 m), else `timeout` at 120 s.
Execution contract recorded in every `outcome.json.driver`: `ChPathFollowerDriver`, steering PID 0.8/0/0 look-ahead 5 m, speed PID
0.6/0.05/0, steer rate 2.0 /s, path min spacing 2 m, path z = `TerrainMap.height + 0.5 m`, control 0.05 s, 0.8 s settle (steering
straight), physics 2 ms, tire TMEASY (`scene.build_config` L58-89), spawn z = `tmap.height(start) + 0.75`.

### 1.7 Per-run files (what labels need)
`trajectory.npz`: `state (T,17) f32` [0 `vel_body_x_mps`, 1 `vel_body_y_mps`, 2 `roll_rad`, 3 `pitch_rad`, 4 roll rate, 5 pitch rate, 6
yaw rate, 7-10 tire Fz (N), 11-14 spindle omega, 15 engine speed, 16 engine torque], `action (T,3) f32` = **[steering, throttle,
braking]**, `pose (T,3) f64` (x, y, yaw), `terminal_pose (3,)`, `terminal_state (17,)`, `parked (T,) bool`, `power_kw`, `contact_n`,
`positive_work_kj_per_interval`, `state_fields`, `dt_s = 0.05`. `outcome.json`: `status`, `elapsed_s`, `goal_time_s`, `frames`,
`wall_s`, `driver{}`, `camera{}`, `route_sha256`, `case_sha256`... `command_reference.npz`: `reference_{waypoints (K,2), stations, speeds,
headings}`, `desired_speed_mps (T,)`. `anchor_state.npz`: `state (17,)`, `pose (3,)`, `history (16,24)`. Plus `case.json`,
`episode_complete.json` (still lists hashes of the deleted telemetry).

### 1.8 Metrics: `scripts/f104_n2_analyze.py::labels(run_dir)` (L16-24) — the single label function for every test
`S0 = 20` frames (1 s after the settle) are skipped. `vx = state[:,0]`, `thr = action[:,1]`.
- `fail` = `outcome.status != 'goal_reached'`
- `back_s` = 0.05 * count(`vx < -0.10` and `thr > 0.3`); `min_vx` = min `vx[S0:]`
- `unsafe` = NOT (not fail AND back_s < 0.05 AND min_vx > -0.30)   ("failed or slid backwards")
- `max_tilt` = max |roll|,|pitch| over `state[S0:, 2:4]` in deg; `tilt30` = `max_tilt > 30` (gen_v1, sensor_v2); hazard test used
  `unsafe_or_tilt` with 35 deg
- `elapsed` = `outcome.elapsed_s`; time to goal = median over non-failed runs of the arm
- GOTCHA: `vx[S0:].min()` raises on a run shorter than 21 frames (never happened on rigid; an early CRM rollover/exit would crash the
  analysis).
Identical to the training label in `f104_n2_dataset.one` (L100-110).

### 1.9 Statistics
- Exact two-sided McNemar on discordant groups: `p = min(1, 2 * sum_{i<=min(b,c)} C(b+c, i) / 2^(b+c))` (`mcnemar`, in
  `f104_n2_analyze.py` L27, `gen_analyze_test.py` L20, `sensor_analyze_v2.py` L12). `a_worse` = groups where arm a = 1 and b = 0.
- Holm step-down over the pre-declared family (`gen_analyze_test.holm` L50-54; 6 unsafe tests in gen_v1).
- Paired bootstrap of the rate difference, 4,000 resamples of groups, `default_rng(0)`, percentiles 2.5/97.5: plain resample in
  `f104_n2_analyze.py` L64 / `f104_n2_haz_analyze.py` L46; **stratified within arena** in `sensor_analyze_v2.cmp` L40-49 and the
  gen_v1 generalisation gap (L86-95).
- Arena-level sign test = the same binomial function on arena win/loss counts (gen_v1 L97-101).
- Group inclusion: gen_v1 = pairwise-complete (a missing arm drops the group only from comparisons involving it, L42-47);
  night-2/hazard/sensor_v2 = groups with ALL arms present.
- Offline ensemble read-out (`sensor_eval_v2.py`): per cell (group, speed profile) argmin of the ensemble-mean logit among the designed
  routes, `picked_unsafe` vs reference variant, bootstrap over groups (4,000), worse/better cell counts.

### 1.10 Result file schemas
- `<out>/picks/<group>.json` (gen_pools): `{group, arena, n_proposal, n_fixed2, tries, arms{arm: {route_id, pool, index, risk, logit,
  rule_score, rule_rank, model_rank, mean_speed, length_m} | null}, proposal_logit_rule_spearman, proposal_risk_quantiles[5],
  fixed2_...}`. sensor_pools_v2: `arms{arm: {route_id, pool, index, mean_speed, risk_<MODEL> for every model}}` (every model's risk for
  every arm's pick — the more useful layout for a two-model comparison).
- `<out>/routes/<route_id>.json`: `{waypoints[[x,y]], speeds[], stations[], headings[], meta{candidate, scene_id, pool, cand_index}}`.
- `gen_v1/test_results.json`: `n_groups{arena}`, `n_complete{arena}`, `unsafe_family[{test, n, rate_a, rate_b, a_worse, b_worse, p,
  p_holm}]`, `secondary{fail|tilt30: [...]}`, `rates{arena|new_pooled: {arm: {n, unsafe, fail, tilt30, median_time_s,
  median_max_tilt}}}`, `gap_n2_unsafe_new_minus_f104{estimate, ci95}`, `P1_arena_sign_test`, `identical_picks`.
- `sensor_v2/pilot_results.json`: `groups, complete, per_arena`, `{fixed2_unsafe|speedfree_unsafe|speedfree_fail|speedfree_tilt30}_<M>_vs_H:
  {n, rate_a, rate_b, diff, ci95, a_worse, b_worse, p}` (percent), `rates{all|arena: {arm: {...}}}`, `identical_to_H{M: n}`.
- night-2 / hazard `closed*/results.json`: `{group: {arm: {fail, unsafe, status, elapsed, back_s, min_vx, max_tilt, risk}}}`;
  `picks.json` rows `{group, abstained | pick_is_anchor, p_control, p_sampler, ...}`.

### 1.11 Reference numbers (rigid) for power planning
f104, 200 feature groups (gen_v1): unsafe speed-free model/rule/straight6 = 0.0 / 0.5 / 1.0 %; **fixed 2 m/s = 2.5 / 10.0 / 48.0 %**;
model tilt30 5.5%; median time 12.6 s (model) vs ~8 s (straight6). sensor_v2 pilot (200 groups, held-out arenas): H vs Dabs fixed-2
6.0 vs 5.0 % (5 vs 7, p = 0.77). On rigid, speed-free rates are too low to resolve anything with 200 groups; fixed 2 m/s is where route
choice shows.

## 2. Recipe for tonight: CRM-trained model vs frozen rigid model, same held-out CRM missions, identical pools

Design: one pool pair per group (proposal-256 + fixed2-256), both ensembles score the SAME tensors, picks locked to files, identical
picks driven once, everything of a group driven on one machine/node, `labels()` unchanged, exact McNemar + paired group bootstrap
(one arena, so the arena-stratified bootstrap of `sensor_analyze_v2.py` reduces to a plain group resample).
Arms: `crm`, `rigid`, `crm_fixed2`, `rigid_fixed2`, `straight6`, `straight2` (+ optional `rule`, `rule_fixed2` from the frozen
hand rule). `E = artifacts/traverse/crm_f104_v1/eval_v1`.

### Step 1 — held-out start/goals (local, seconds)
Option A (recommended if the CRM training set excluded them): reuse `R/gen_v1/cases_test_f104/cases` (200 groups). N2 never trained on
them, and with the gen_v1 seed tags the `rigid` arm reproduces gen_v1's `n2`/`n2_fixed2` picks exactly, so the rigid-terrain outcome of
the very same routes already exists in `R/gen_v1/test/runs/` (free rigid-vs-CRM physics contrast). NOTE for the `matched_*` (sensor_v2) models
`ds_v2_gen_f104.npz` (the gen-wave f104 runs) is in the training file list with f104 as a training arena, so assume they have seen
these groups; Option A is only clean with `N2_s*.pt` as the frozen rigid model.
Option B (fresh):
```
PY=/home/harry/miniconda3/envs/nedm/bin/python
R=artifacts/traverse/fdm_f104_50h_20260909; E=artifacts/traverse/crm_f104_v1/eval_v1
$PY scripts/gen_cases.py --groups 200 --seed 20260917104 --prefix f104_crm_test_group --strata feature --wave crm_f104_v1_test \
  --arena assets/traverse/arena_f104_50h_v1 --out $E/cases_test --margin-m 2.0 \
  --avoid $R/cases $R/cases_reserve_v1 $R/cases_night2/cases $R/cases_test_final $R/cases_ext_final $R/cases_haz_final \
          $R/gen_v1/cases_test_f104/cases $R/sensor_v1/cases_f104/cases $R/sensor_v1/cases2_f104/cases <CRM_TRAINING_CASES_DIR...>
```
Held-out must hold for BOTH models: avoid every rigid training pool above AND every group the CRM model trained on.

### Step 2 — pools + both models' picks (local GPU; ~2-3 min per 200 groups with 12 workers)
Zero-code-change path (relies on bit-reproducible pools; assert it in the merge):
```
GEN_MODELS="$PWD/$R/night2_v1/final/N2_s*.pt"            $PY scripts/gen_pools.py --cases $E/cases_test/cases \
   --arena assets/traverse/arena_f104_50h_v1 --out $E/picks_rigid --arena-tag f104 --workers 12 --shards 1
GEN_MODELS="$PWD/artifacts/traverse/crm_f104_v1/<train>/CRM_s*.pt" $PY scripts/gen_pools.py --cases $E/cases_test/cases \
   --arena assets/traverse/arena_f104_50h_v1 --out $E/picks_crm   --arena-tag f104 --workers 12 --shards 1
```
then a ~40-line merge (new file, e.g. `scripts/crm_merge_picks.py`): per group assert `(n_proposal, n_fixed2, tries)` equal and the
model-free arms (`rule`, `straight6`, `straight2`, `rule_fixed2`) have equal `index` in both outputs (pool identity check); map arms
`rigid <- picks_rigid.n2`, `crm <- picks_crm.n2`, `*_fixed2 <- n2_fixed2`, model-free arms from either; dedup by `(pool, index)` exactly
as gen_pools L81-103 (`rid = f'{g}__{first_arm}'`, copy `picks_<src>/routes/<info.route_id>.json` -> `$E/routes/<rid>.json`); write
`$E/picks/<g>.json` (`arms{arm: {route_id, pool, index, risk, mean_speed, length_m}}`) and `$E/tasks_eval.json` rows
`{id, group, arena:'f104', case:'cases_test/cases/<g>.json', route:'routes/<rid>.json', shard, run:True}` (first occurrences only).
Cleaner path (preferred if there is time): copy `scripts/sensor_pools_v2.py` -> `scripts/crm_pools.py` and change: L16
`P.set_grid_map(griddir)` -> `P.set_map(arena)`; L34 `P.corridors12` -> `P.corridors`; L41 `--grid` -> `--arena`; L47
`P.GridRiskModel(v)` -> `P.RiskModel(pattern=v)`; seed tags L30-31 -> `'gen_night2'`/`'gen_fixed2'` (reproduces gen_v1 pools on gen_v1
cases); add `pick['straight2'] = ('proposal', P.anchor_index(c1, 0.0, 2.0))` and put it in `arms`. Call:
`--models rigid=<N2 glob>,crm=<CRM glob> --case-prefix cases_test/cases --route-prefix routes --shards <n>`. This records
`risk_rigid` and `risk_crm` for every arm's pick (needed to see how each model rates the other's choice). If the CRM model is a
v2-grid model instead, build both `P.corridors(cands)` and `P.corridors12(cands)` for the same `cands` and route each model to its own
tensor (`set_map` writes `f104_n2_dataset.G`, `set_grid_map` writes `sensor_dataset_v2.G`; they do not collide).
Checkpoint compatibility: the CRM-trained ensemble must be loadable by `RiskModel` (keys in 1.3; `nctx` must be 5 = geometry-only
context; channels 0-3 normalised by its own `norm`). Pick ONCE, keep the files; never re-score on another device (CPU/GPU float
differences can flip near-tie argmins).
Sanity checks before driving: `n_proposal == 256` for all groups; `straight6`/`straight2` indices are 2/0; count `crm == rigid`
identical picks (if > ~70% the test has little power); on Option A compare `rigid` indices with `R/gen_v1/test_f104/picks/*.json`
(`arms.n2.index`, `arms.n2_fixed2.index`) — they must match exactly.

### Step 3 — drive all arms of a group on one machine
Needs the CRM collector (other scout) to honour the `gen_collect.py` CLI and outputs: `--case --route --out --horizon-s 120` ->
`outcome.json{status, elapsed_s}`, `trajectory.npz{state (T,17), action (T,3)=[steer,throttle,brake], pose}`, `episode_complete.json`.
Then `scripts/gen_runner.py` is reusable with two edits: the `cmd` list (L25-27) and the worker count. Local (luffy, one GPU):
```
GEN_ROOT=$PWD/$E GEN_TASKS=$PWD/$E/tasks_eval.json GEN_OUT=$PWD/$E/eval SLURM_ARRAY_TASK_ID=0 GEN_WORKERS=<k per GPU> \
  NRD_PYTHON=/usr/bin/python3.12 PYTHONPATH=/home/harry/chrono/build/bin:src  /usr/bin/python3.12 -u scripts/gen_runner.py   # all rows shard 0
```
AMD (if the HIP FSI build is used — see 3.4): keep `shard = md5(group) % n_shards`, one array task per shard, request a GPU, and size
`GEN_WORKERS` to the GPUs of the node. Pilot one shard first (as job 420020 did). Budget: ~5.3-5.5 unique drives per group; rigid
episodes were 8-30 s simulated (fixed-2 arms ~21-29 s, failures >= 34 s, timeouts 120 s). crm_smoke (`artifacts/traverse/crm_f104_v1/smoke/orient_016/smoke_report.json`, a 1 s run, 753k SPH markers) measured ~1.45 s wall per
simulated second at 0.16 m spacing on luffy -> 200 groups ~ 1,100 drives ~ 25 s mean ~ 11 GPU-hours serial at that setting unless
several episodes share the GPU; size the group count accordingly (100 groups = ~550 drives).

### Step 4 — analysis
Copy `scripts/sensor_analyze_v2.py` -> `scripts/crm_analyze.py`; change `V` and the picks glob (`f'{V}/pilot_{ar}/picks'` ->
`$E/picks`), `--models` default `rigid,crm`, reference `'H'` -> `'rigid'` (L52-56, L68), add `straight2`. It already gives, per
comparison, rates, diff, stratified bootstrap CI (4,000, seed 0), discordant counts and exact McNemar for `fixed2 unsafe`,
`speed-free unsafe`, `speed-free fail`, `speed-free tilt30`, plus per-arm `median_time_s`, `median_max_tilt`, and the identical-pick
count. Add `fixed2 fail` and `fixed2 tilt30`. Guard `labels()` against T < 21 frames. Pre-declare (before any outcome is read):
primary = unsafe, `crm` vs `rigid`, in the regime whose MODEL-FREE base rate (`straight2`/`straight6` on CRM) is >= ~5%; Holm over
{speed-free unsafe, fixed-2 unsafe}; secondary fail / tilt30 / time. Write the plan + sha256 first (gen_v1 convention: `PLAN.md`,
`PLAN.sha256`).

### Minimal file set to copy/adapt
Unchanged: `scripts/gen_cases.py` (+ `generate_traverse_f104_collection.py`), `scripts/gen_planner.py`, `scripts/f104_n2_sampler.py`,
`scripts/f104_n2_dataset.py`, `scripts/gen_riskmodel.py`, `scripts/gen_terrain_features.py`, `scripts/f104_n2_analyze.py` (labels,
mcnemar), `src/nedm/traverse/{terrain.py, fdm_mppi.py, fdm_diverse_planner.py}`, `R/night2_v1/final/N2_s{0..4}.pt`,
`R/gen_v1/hand_rule.json`, `assets/traverse/arena_f104_50h_v1`. Adapt: `sensor_pools_v2.py` -> `crm_pools.py` (or `gen_pools.py` x2 +
merge), `gen_runner.py` (cmd + workers), `gen_array.sbatch` (only if AMD), `sensor_analyze_v2.py` -> `crm_analyze.py`. Replace:
`gen_collect.py` + frozen `run_chrono` by the CRM collector.

## 3. Rigid-terrain-specific assumptions in the evaluation path (must change or be re-verified for CRM)

1. `gen_collect.StopPolicy.validate_native_height` (L83-99): compares `scene.terrain.GetHeight(x, y, 20)` (RigidTerrain ray query) with
   `TerrainMap.height` at 50 points, limits p95 0.08 m / max 0.15 m, hard-fails the episode. Not meaningful for an SPH terrain; replace
   by a marker-surface check (crm_smoke `--check-surface`) done once per build, not per episode.
2. `gen_collect.make_observer` (L193-209): `scene.terrain.GetNormal(...)` per wheel ("Geometric RigidTerrain.GetNormal"); rich telemetry
   is deleted after each run anyway -> switch it off.
3. `gen_collect.adapted_function` patches `run_chrono` by exact source strings and `main` enforces `source_manifest.json` hashes of 9
   frozen files incl. `scene.py` and the runner (L27-30, L237-242). Any CRM edit trips "Frozen source file mismatch". A CRM source tree
   needs its own manifest or its own collector. (`scene.py` is already modified in the worktree: `RenderSpec.with_rgb`.)
4. Runtime binding: `FDM_RUNTIME_FINGERPRINT=$C/pilot_runtime_412394.json` describes the NON-FSI build
   `/work1/dannegrut/harry/nrd/chrono-build`; `nrd_pychrono` puts `$CHRONO_BUILD/bin` on PYTHONPATH. A HIP FSI build exists at
   `/work1/dannegrut/harry/nrd/chrono-build-fsi` (Sep 8; `CH_ENABLE_MODULE_FSI_SPH=ON`, `CHRONO_GPU_BACKEND=HIP`, archs
   gfx90a;gfx942;gfx950, `bin/pychrono/_fsi.so` present) — untested by me. The fingerprint is only recorded, not compared, so it would
   silently mislabel a CRM run; regenerate it. `--chrono-data` in `gen_runner.py` L27 also points at the non-FSI build's data dir.
5. Concurrency model: 14 single-threaded CPU episodes per node (`OMP_NUM_THREADS=1`, lavapipe, no GPU requested, `-t 03:00:00`,
   per-episode timeout 3600 s). CRM needs a GPU per episode (mi2101x = 1 GPU/node; MI350 partition has a 4 h cap). `nrd_use_lavapipe`
   is irrelevant without a camera (`record_rgbd_stride=0`, `render=None` in collect mode).
6. **Determinism premise.** Pairing relies on "Chrono is deterministic per node" (rigid, single-thread CPU): identical routes give
   identical outcomes, paired arms differ only by the route. GPU SPH may not be run-to-run deterministic even on one device. Run an
   A/A check first (same case+route twice on the same GPU; compare `trajectory.npz`). If not bit-identical: keep the dedup (so identical
   picks still tie exactly), keep all arms of a group on one device, and measure the label noise floor (e.g. 30 duplicated drives) —
   McNemar stays valid but loses power.
7. Vehicle/terrain setup in the frozen runner: TMEASY tires, 2 ms step, spawn at `height + 0.75 m`, 0.8 s settle, initial-state gate
   (speed <= 1 m/s, |roll|,|pitch| <= 20 deg, yaw error <= 10 deg, xy error <= 1 m; `on_anchor` L101-118). CRM needs rigid-mesh tires with
   BCE markers, step 5e-4..1e-3 (keep `DT/dt` an integer: `substeps = round(0.05/dt)`), and sinkage during settle may trip the gate
   or need a longer settle. State columns 7-10 (tire Fz) may be zero/meaningless under CRM — harmless for N2 (no state input) and for
   `labels()` (uses columns 0, 2, 3 only), but keep the (T,17) layout.
8. Path-follower z: `TerrainMap.height + 0.5 m` ("privileged terrain height"). On CRM the wheels run below the initial surface; the
   controller contract (gains, look-ahead, path from the same waypoints) should stay as is; just be aware the path floats higher.
9. Label thresholds were tuned on rigid ground: slide = `vx < -0.10` with throttle > 0.3 (>= 0.05 s) or `vx < -0.30`; stall detector =
   0.25 m in 2 s with throttle > 0.3, not before 24 s (+2 s +8 s). On soil the likely failure is digging in with forward creep/wheel
   spin, which is a `fail` only when the stall detector or the 120 s timeout fires. Keep `labels()` byte-identical for both arms
   (and identical to the label the CRM model was trained on); report status counts.
10. Arena edge: planner validator lets the footprint reach +-40 m (`CFG.arena_half_extent_m=40`); `terrain_bounds_exit` at 40 m. The
    CRM particle box ends at +-40 m (crm_smoke aabb) — soil at the edge is unsupported unless walled. nav_v1 already logged "routes were
    allowed to touch the terrain edge" as a defect. Tightening the extent changes the pools (and breaks reproduction of gen_v1 picks);
    if changed, change it once for both arms.
11. Map registration: `set_map` uses `TerrainMap` (samples at cell centres, 80/512); Chrono `RigidTerrain` puts samples at patch edges
    (80/511) — the known 511/512 radial offset (<= 0.078 m). `CRMTerrain.Construct(heightmap...)` may use yet another convention and the
    BMP row order is being checked by crm_smoke (`--flip-bmp`). Both models read the same map so the comparison stays fair, but a
    y-flip would make BOTH planners plan on mirrored terrain — verify orientation before any pick is driven.
12. Map source itself: no sensor capture of SPH soil exists (captures in `static_map_v1/`, `sensor_v2/grids/` are renders of the rigid
    mesh). Use the heightmap path (`set_map`) for both arms; the v2 grid of f104 is the same undeformed geometry if a grid model is used.
    Terrain deformation never enters the input (each episode starts on fresh soil; single-goal routes do not revisit ground).
13. `hand_rule.json` and the N2 ensemble are fit on rigid outcomes; on CRM they are "frozen rigid" baselines by construction. The
    optimal speed regime may invert (on rigid "carry momentum" dominates: straight 6 m/s was nearly as safe as the planner); do not
    assume the fixed-2 regime is still the high-base-rate one — read the model-free arms' base rates first.
14. `gen_collect` gates that still pass for f104 but are rigid-campaign artefacts: `gen_arenas.json` BMP allowlist, `size_m == 80`,
    `assets == []`, horizon <= 120 s and a multiple of 50 ms, `--minimum-elapsed-s >= 24`.
15. `gen_array.sbatch` hardcodes `GEN_ROOT=$C/gen_v1` (sensor_v2 worked around it with `../sensor_v2/...` relative paths).

## 4. Other gotchas
- `gen_pools.py` hardcodes the scorer (`P.RiskModel()` -> env `GEN_MODELS`, else N2) and the hand rule path (env `GEN_RULE`); its
  `tasks_cluster.json` is not runner-ready (no `route`, includes `run=False` rows).
- `gen_runner.py` treats a dir with `episode_complete.json` as cached; a failed dir (has `collection_request.json` +
  `collection_failure.json`) blocks its own retry until moved.
- Missing arms: a straight anchor that fails validation yields `arms[arm] = null`; the fixed-2 pool can be short or empty near the
  arena edge (`if c2 else None`).
- `risk` values are badly calibrated (median predicted 0.008% vs realised 0.33%) — use as an ordering only; a 5% abstain threshold
  is meaningless for a differently calibrated CRM model.
- Optimiser's curse: the argmin over 256 lands on each model's own optimistic outliers (night-2 motivation for the pessimist arm; max
  over seeds cost nothing on rigid). If added tonight, add it for both models (`np.max(zs, 0)` instead of `np.mean`).
- `action` columns are [steering, throttle, braking]; night-1 label scripts read column 0 by mistake — `labels()` uses column 1.
- Records to mirror: `PLAN.md` + `PLAN.sha256` before any drive, `LOG.md` with UTC stamps, `REPORT.md`.
