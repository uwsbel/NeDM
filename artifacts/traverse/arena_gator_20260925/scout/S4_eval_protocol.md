# S4 scout: evaluation protocol for cross-arena generalisation (task A) and for the Gator (task B)

Read-only scout, 2026-09-25. Worktree `/home/harry/NeDM-traverse_mppi`, branch `crm_improve_v1`. Paths are relative to
the worktree unless absolute. Nothing in the repo was modified, no cluster job was submitted, no Chrono collection was
run. One import/instantiate smoke of the Chrono Gator and HMMWV_Full was run with the local source build
(`PYTHONPATH=/home/harry/chrono/build/bin /usr/bin/python3.12`, no simulation step). Items I could not check are marked
**[unverified]**.

Short names used below: K1 = `artifacts/traverse/generalist_20260921`, K2 = `artifacts/traverse/crm_improve_20260922`,
R = `artifacts/traverse/fdm_f104_50h_20260909`, "soil" = CRM, "rigid" = rigid terrain.

---------------------------------------------------------------------------------------------------------------------

## 1. How the current closed-loop benchmark is built

### 1.1 The 800-group paired suite

Built by `scripts/ga_suite.py --suite planner` (stage `assemble`, lines 77-110) into `K1/A_adapt/suite/`:

| part | groups | source | how sampled | strata |
|---|---|---|---|---|
| fresh | 600 `f104_pair_group_0000-0599` | `K1/cases/pair_v1/cases` | `scripts/gen_cases.py --strata all`, seed 20260921104, margin 2 m vs 4,507 existing pairs (log `K1/cases/gen_pair_v1_600b.log`) | hill 250 (165 cross-slope, 85 entry-cross-exit), crater 200 (132 / 68), long traverse 100, roughness transfer 50 |
| reused | 200 `f104_crm_eval_group_0000-0199` | `artifacts/traverse/crm_f104_v1/cases_eval/cases` | CRM night-1 evaluation set, seed 20260926104, hill/crater only, >= 2.03 m from every f104 pair (median 3.2 m) (`crm_f104_v1/PLAN.md:69`, `REPORT.md:66`) | hill 120, crater 80 |

Counts verified from `K1/A_adapt/suite/suite.json` (min 4-D distance fresh-to-reused 2.05 m, 0 exact matches).
Every file is sha256-listed in `suite.json`; `SUITE_LOCKED.sha256` hashes the manifest and the per-world task files
(`ga_suite.py:318`; check with `tail -n +2 SUITE_LOCKED.sha256 | sha256sum -c`).

**How one start/goal pair is drawn** (`scripts/gen_cases.py`):
- start cells: a 1 m grid over +-34 m with local slope < 7 deg, jittered +-0.35 m (lines 50-53, 62);
- feature strata (round-robin over the arena's `TerrainMap.features`, f104 has 10: 5 hills, 5 craters): start 9-36 m
  from the feature centre, aimed at the centre (`entry_cross_exit`) or 0.65 sigma to either side (`cross_slope`), goal
  10-23 m beyond the feature, clipped to +-35 m (lines 63-80); non-feature strata: a random goal >= 42 m away
  (`long_traverse` / `roughness_transfer`, lines 81-87); total length 24-85 m (line 88);
- vehicle footprint check at the start (`start_ok(footprint(...))`, lines 91-93; HMMWV footprint);
- separation: >= 2 m in 4-D (start x, y, goal x, y) from every pair in the `--avoid` case dirs and from every pair
  already selected (lines 97-101); this is the only "blacklist vs training pairs" at generation time;
- all 12 designed routes (3 lateral offsets 0/-4/+4 m x 4 speed profiles 2, 4, 6, 2-6-2 m/s) must pass the validator
  (curvature <= 0.10 /m, +-36 m), lines 102-109; the files are written for every group (`routes/<g>/route_NN.json`);
- a hash split 90/5/5 is written into each case (lines 111-112) but is irrelevant for suite groups.

**Blacklist in the training builders**: suite ids are refused by pattern in every dataset builder
(`scripts/ga_build_mixed.py:29`, `BLACKLIST = ['f104_crm_eval_group_*', 'f104_g1_test_group_*', 'f104_pair_group_*']`;
the same patterns in `scripts/ci_a5data.py:93`). Caveat carried from CRM night 1 (`crm_f104_v1/REPORT.md:124-130`): a
2 m start+goal margin is not route-level separation; 99.9 % of picked route points lie within 1 m of some training
route, so "held-out" on f104 is interpolation on a memorisable arena. This is exactly what task A is meant to fix.

### 1.2 Decision protocols

**Standing start** (K1 stage A3): the planner decides at the case pose at rest, before the 0.8 s braked settle; a
history model sees an all-masked window (it was trained with `--hist-drop 0.2`, so this is in distribution). One pass:
the picked route is driven from rest. Planner = CEM 4 rounds x 64 (arm letter `B`, rng tag `n2iter_cem4x64`), or the
one-shot 256 draw (arm `A`).

**Moving start** (K2 final protocol, 0.5 s approach):
1. Pass 1: per group one approach route, independent of the model: the straight start-goal line at a constant 3 m/s or
   `route_00` at 3 m/s, whichever has the lower mean |grade| over the first 12 m (straight line wins ties)
   (`scripts/ga_approach.py:67`, stage `routes`; generalised to any length in `scripts/ci_a5data.py --stage
   approach-routes`). Driven with `--horizon-s 0.5` (10 frames, about 0.1 m of travel; 1 s = 0.7 m).
2. The decision state at frame 10 (pose, 17-column state, a 40-frame window of 12 state columns + 3 actions, masked
   before frame 1) is extracted (`ci_a5data.py --stage analyze`), `poses_<world>_all.json` in
   `K2/a5data/decision_suite_L0p5/`.
3. Every arm plans offline from that SAME recorded state: `scripts/ci_planner.py` (CEM) or `scripts/ci_grad.py`
   (CEM 4x64 pick + 60 Adam steps on the ensemble's route logit from 17 starts, keep-best by the pessimistic member,
   abstain unless 0.3 logit better; 2.57 s per decision on the 5090, GPU peak 0.45 GB, log
   `K2/s4/L0p5/picks_crm_HnG.log`; 666/800 picks changed, 134 abstained).
4. Pass 2: the approach prefix is re-driven and the collector switches to the pick at frame 10
   (`scripts/ga_a5_pass2_tasks.py`, `extra = ['--mode','branch','--branch-frame','10','--branch-route',...]`), soil
   through `crm_collect_ext.py`, rigid through `gen_collect_ext.py`. Soil prefixes are byte-identical across GPU types;
   rigid is deterministic per node only (memory note `amd-not-reproducible-across-nodes`), so the rigid rows of a group
   share one shard (`md5(group) % 6`).

The 3 s approach of K1 cost 10-16 points on soil because the vehicle had already committed to a climb when it decided
(`K2/REPORT.md:26-39`); 0.5 s keeps it on the start pad while a 0.25 s window already identifies the world.

### 1.3 Integrity and statistics

- **Picks hashed before driving**: every pick directory writes `PICKS_LOCKED.sha256` (one hash over `routes/*.json`
  names + contents, `ga_suite.py:198-201`, `ga_planner.py:605`, `ci_grad` identical); task files are locked the same way
  (`K1/A_adapt/a3/tasks_*_LOCKED.sha256`). Identical picks (same route sha256) are driven once and mapped back through
  `run_index_<world>.json`.
- **Labels** (`scripts/f104_n2_analyze.py:16-24`): fail = status != goal_reached; unsafe = fail or, after the 1 s
  settle, any backward motion under throttle (vx < -0.10 with throttle > 0.3, >= 0.05 s) or vx < -0.30; tilt30 =
  max |roll|,|pitch| > 30 deg. On soil unsafe == fail almost always (stalled vehicles dig in).
- **`scripts/ga_analyze.py`**: arms `NAME=<picks dir>:<arm letter>:<runs dirs>`; only groups where EVERY arm has a
  drive are analysed (line 209). `cmp()` (lines 84-93): paired group bootstrap of the rate difference (4,000 resamples of
  groups, seed 0), 95 % CI, one-sided 5th/95th percentiles, exact two-sided McNemar on discordant groups, identical-pick
  count. `time_ratio()` (lines 96-108): median over groups of t_X/t_Y on groups both reached, bootstrap 95th percentile
  vs a bound (1.10). Per stratum (`fresh`/`reused` from `--suite`) and per `evaluation_stratum`. Pick agreement by route
  geometry. With `--cluster-ci --cases` it runs `scripts/n2_cluster_ci.py`.
- **Clustered interval** (`scripts/n2_cluster_ci.py:16-31`): each group is assigned to the terrain feature nearest to
  its start-goal midpoint (ONE arena, `--arena`, default f104); 4,000 bootstrap resamples of clusters (9 clusters on the
  suite) and a cluster-level sign test. Used as robustness only.
- Decision rules used so far: K1 non-inferiority (one-sided 95th percentile of reference-minus-test < 3 points on the
  600 fresh groups, 200 reused reported separately); K2 superiority (one-sided lower bound of the improvement > 0).

Reference numbers on the 800 groups (from `K1/A_adapt/a3/results_*_A0A3.json`, `K2/s4/results_s4*.json`):

| world | protocol / planner | goal reached | unsafe | median time |
|---|---|---|---|---|
| soil | standing, soil specialist CEM | 95.8 % | 4.2 % | 15.4 s |
| soil | standing, soil specialist one-shot 256 | 91.6 % | 8.4 % | 17.6 s |
| soil | standing, rigid-trained specialist on soil | 80.1 % | 20.0 % | 14.3 s |
| soil | 0.5 s approach, short-window history ensemble, CEM | 95.8 % | 4.4 % | 15.3 s |
| soil | 0.5 s approach, same + gradient refinement (final) | **97.5 %** | 2.6 % | 14.3 s |
| rigid | standing, every arm | 99.9-100 % | 0.0-0.2 % | 12.6-14.5 s |
| rigid | 0.5 s approach + gradient (final) | 100.0 % | 0.1 % | 13.5 s |

**Consequence for both tasks: on rigid ground the speed-free goal-reached metric is at its ceiling on f104 with the
HMMWV.** It cannot show a 3-5 point difference there; rigid needs a harder regime (fixed 2 m/s, unsafe, or a weaker
vehicle).

Observed discordance (fraction of groups where exactly one of two arms fails), useful for power: two competent soil
models with 4-7 % failure disagree on 3.2-8.0 % of groups (e.g. final vs its CEM pick 30/800, soil specialist vs
shared history 46/800); models with very different training data disagree on 16-17 % (soil specialist vs rigid-trained
on soil 133/800). On unseen rigid arenas at 2 m/s the f104 model and the hand rule disagreed on 127/1,000 (12.7 %).

---------------------------------------------------------------------------------------------------------------------

## 2. What gen_v1 did on the five sibling arenas (`R/gen_v1/PLAN.md`, `REPORT.md`, `test_results.json`)

- **Arenas**: 40 seeds (201-240) of the f104 generator (`scripts/traverse_wp7_arenas.py --prefix arena_g`, difficulty
  1.0) ranked by a scaled distance to f104 over 8 terrain statistics (slope cap, roughness, roughness correlation, hill
  and crater counts, 99th-percentile slope, flat fraction, height range; scale vector in
  `R/gen_v1/arena_similarity.json`); the 5 closest were used: g228, g203, g217, g216, g231 (distance 0.81-0.94). The
  ranking script itself is not in `scripts/` (only its output json and `gen_figure_arenas.py`, which reads it)
  **[the ranking code must be re-written for new seeds; 8 features + scale are in the json]**.
- **Model**: the frozen night-2 rigid ensemble `R/night2_v1/final/N2_s*.pt` (geometry-only GRU, trained on f104 only),
  no retraining. **Map input on the new arenas = each arena's heightmap encoded like the depth map**
  (`scripts/gen_planner.py:35` `set_map`), because no captured image existed then; on f104 the two gave the same pick in
  32/40 pools.
- **Test pairs**: 200 per arena, hill/crater strata only (`gen_cases.py --strata feature`), f104's 200 fresh ones >= 2 m
  from any earlier f104 pair. Standing start, one-shot 256 proposal pool per group (md5 seeds), identical pools for all
  scorers (`scripts/gen_pools.py`).
- **Arms** (all arms of a group in ONE cluster array task, identical picks driven once): model (argmin risk), hand rule
  (argmin of a pairwise-logistic score over 22 hand terrain+speed features, fit on the f104 training routes,
  `scripts/gen_hand_rule.py`, `R/gen_v1/hand_rule.json`), straight 6 m/s; and the same three at a fixed 2 m/s
  (geometry-only pool).
- **Statistics**: pre-registered primary P1 = pooled new arenas, unsafe, model vs hand rule, exact McNemar, Holm over 6
  unsafe tests; generalisation gap bootstrapped within arena; arena-level sign test (`scripts/gen_analyze_test.py`).
- **Result** ("model ties hand rule speed-free, wins at 2 m/s"): speed-free unsafe 1.3 % model / 1.8 % rule / 4.5 %
  straight (model vs rule 10 vs 15 discordant, p = 0.42, primary NOT met); fixed 2 m/s 5.9 % / 13.6 % / 59.8 % (25 vs 102,
  p < 1e-10, rule worse on 5/5 arenas). f104: 0.0 / 0.5 / 1.0 % speed-free, 2.5 / 10.0 / 48.0 % at 2 m/s. Generalisation
  gap +1.3 points [0.7, 2.0] speed-free. Per-arena spread at 2 m/s: 2.5-8.0 % (model), 7.5-20.5 % (rule).
- **Offline** on 9,000 designed routes (150 groups x 12 routes x 5 arenas, `R/gen_v1/data`): within-group AUC 0.955 on the
  new arenas vs 0.991 on f104 held-out groups; same-speed 0.905 vs 0.985; hand rule 0.902 / 0.778. The labelled rows are
  in `R/gen_v1/station_ds_gen_v1.npz` (15,639 routes, never used for training).
- **Five-goal missions** (`scripts/gen_missions.py`, `gen_mission_runner.py`): one continuous simulation per mission,
  replanning from the measured pose at each goal (clock paused), legs 20-35 m, turns <= 110 deg, >= 3 of 5 legs over a
  hill or crater; 100 missions on f104 + 20 per new arena, arms model / hand rule / straight. f104 all 5 goals 99 % vs 91
  % vs 90 % (M1 10 vs 1 p = 0.012, M2 8 vs 0 p = 0.008) at 54 s vs 36/30 s; new arenas 94 / 89 / 94 %, slides 7 / 15 /
  16 %, tilt > 30 deg 9 / 21 / 42 %. Missions compare policies, not routes (poses diverge after leg 1).
- **Later use of the same arenas** (they are no longer "never seen"): sensor_v1/v2 captured maps for all five
  (`R/sensor_v1/maps/arena_g*`, Vulkan lavapipe backend) and trained matched variants split by whole arena (g216 + g231
  held out); nav_v1 used four more (g213, g204, g234, g223, "next four of the ranking") as unseen mission arenas
  (16/16 waypoint missions). No currently deployed model was trained on any g-arena.

What this means for task A: gen_v1 only measured *one* training set (f104) on new arenas, with a rigid one-shot planner
and heightmap input, and found a small rigid gap (+1.3 points speed-free, +3.4 points at 2 m/s). It never trained on
more arenas and never ran soil on another arena.

---------------------------------------------------------------------------------------------------------------------

## 3. Proposed evaluation for task A (more training arenas vs generalisation)

### 3.1 Arenas (fixed before any model is trained)

| role | proposal | why |
|---|---|---|
| training arenas | T1 = f104, T2, T3 = two existing close siblings (suggest g203 and g228, distance 0.81; both have maps `R/sensor_v1/maps/`, v2 grids `R/sensor_v2/grids/` and gen_v1 rigid data) | the data scouts choose; the evaluation only needs them frozen first |
| dev arena | one more existing sibling (g217 or g216) | headroom pilot and any model or recipe selection, so test arenas stay sealed |
| test arenas | **6 fresh arenas** (minimum 4): generate 40 new seeds (e.g. 241-280) with `traverse_wp7_arenas.py --prefix arena_g --difficulty 1.0`, rank by the gen_v1 distance to f104, take the 6 closest (same-family generalisation, comparable to T2/T3) | never captured, driven or inspected; no g-arena from 201-240 qualifies as sealed any more |
| optional stress stratum | 2 arenas from ranks ~20-30 or difficulty 1.15 | exploratory only, reported separately |

Why 6 x 250 rather than 3 x 500: arena-to-arena variation is large (gen_v1 2 m/s unsafe 2.5-8.0 % across five arenas
for the same model), and the drive count is set by groups, not arenas; per extra arena the setup is only a map capture,
a v2 grid and an allowlist entry. More arenas give more (arena, feature) clusters for the clustered interval.

Per-arena setup that is f104-hardcoded today (implementation work, not optional):
- overhead depth map for the planner: `f104_n2_dataset.init_map` holds ONE global map (lines 17-24), and `ci_grad` /
  `ci_planner` / `ga_planner` take one `--map-root`, so plan per arena. Capture every training and test arena with the
  same backend as the f104 map the models were trained on (`crm_f104_v1/map_root`, OptiX luffy fork build, sha
  53e23f99...; `scripts/crm_capture_map_local.py`). The sensor_v1 g-arena maps are Vulkan lavapipe captures (backend
  field in `observation.json`); OptiX vs Vulkan agreed to 4.6e-5 m on f104 (memory note `optix-depth-camera-fov-bug`), so
  either works, but mixing backends between training and test arenas should be avoided or checked on one arena;
- v2 Chrono-frame grid per arena for the approach-route grade and CRM sinkage (`ga_approach.py:61-62` default f104;
  `ci_a5data.py` accepts `--grid/--arena`, line 1179), from `scripts/sensor_map_v2.py`;
- rigid collector allowlist: `gen_collect.py:247-249` refuses any arena whose BMP sha256 is not in
  `scripts/gen_arenas.json` (today f104 + g228/g203/g217/g216/g231); the CRM collector takes the arena from the case file
  (`crm_collect.py:172`, `crm_collect_ext.py:109`) with no allowlist;
- `ci_a5data.py` pass-1 task builder hardcodes f104 case locations and `ARENA_TAG = 'f104'` (lines 76-97); rigid pass-2
  rows hardcode `arena='f104'` (`ga_a5_pass2_tasks.py`); `ci_grad --arena-tag` defaults to f104;
- `n2_cluster_ci.py` clusters by the features of ONE arena; for several arenas it must key clusters by (arena, nearest
  feature). `ga_analyze.py` keys groups by id, so group ids must be unique across arenas (use `--prefix
  <arena>_test_group` in `gen_cases.py`); putting the arena name into the suite's `stratum` field gives per-arena blocks
  from the existing `--suite` code path for free.

### 3.2 Test suites

- **Unseen arenas**: `gen_cases.py --arena <test arena> --strata all --groups 250 --seed <new, declared> --prefix
  <arena>_test_group` per test arena (same generator, gates and strata mix as the 600 fresh f104 groups, so numbers are
  comparable with the f104 suite; hill/crater groups also reported alone). 1,500 groups per world. Lock with the
  `SUITE_LOCKED.sha256` scheme before any model exists.
- **In distribution**: f104 = the existing 800-group suite (pass-1 decision states at 0.5 s already exist in both worlds,
  `K2/a5data/decision_suite_L0p5`, and the deployed model's drives exist: 97.5 % soil / 100 % rigid). T2 and T3 = 250
  fresh groups each with `--avoid <that arena's training case dirs>` (2 m margin), blacklisted by id pattern in every
  builder (extend the `BLACKLIST` idea to `g203_pair_group_*`, etc.).
- **Dev arena**: 200 groups, for the headroom pilot (section 3.8) and any selection.
- **Feasibility subset**: on 50 groups per test arena, drive all 12 designed routes that `gen_cases.py` already writes.
  This gives the "some designed route works" ceiling and the offline within-group ranking AUC on unseen arenas (as
  gen_v1's 9,000 routes did).

### 3.3 Training sets

Nested so that every larger set contains the smaller one (fixed data seed), train-group split per arena as the twin data
(90/5/5 by group); the val groups of each training arena are the only selection data besides the dev arena.

| name | arenas | groups (f104 has 1,089 train groups today) | question |
|---|---|---|---|
| M1 = A1 | f104 | 1,089 | today's model (retrained with the new recipe if the recipe changes) |
| M2 | f104 + T2 | 545 + 545 | diversity at matched total data |
| M3 | f104 + T2 + T3 | 363 x 3 | diversity at matched total data |
| A2 | f104 + T2 | 1,089 + 1,089 | practical: more data from more arenas |
| A3 | f104 + T2 + T3 | 1,089 x 3 | practical: more data from more arenas |

Matched total isolates arena diversity; additive answers "is collecting on new arenas worth it". Fix the recipe in
optimiser steps per row or in epochs before training and state it (30 epochs over 3x more rows is 3x more steps).
Offline only (cheap, answers "which arena" vs "how many"): the rotations {T2}, {T3}, {T2,T3}, {f104,T3} at matched total.

### 3.4 Seeds and ensembles

Deployed ensembles have 5 seeds; the paired group bootstrap covers test-group sampling only, not training randomness.
Offline seed spread was 0.002-0.005 AUC (crm_night2 memory), but closed-loop spread between two ensembles trained on the
same data is unmeasured **[unverified]**. Proposal: two independent 5-seed ensembles (seeds 0-4 and 5-9) for the
primary pair M1 and M3; a group's outcome under a condition = the mean over its two ensembles (0, 0.5, 1); report the
ensemble-a vs ensemble-b difference within each condition as the training-noise floor. Other conditions: one ensemble.
Offline: 3 ensembles for every training set.

### 3.5 Planner protocol per world

- **Soil = primary testbed.** Use the protocol that matches the trained model: if task A trains the shared rigid+soil
  history model (needs both worlds on every training arena), the final protocol (0.5 s approach, CEM 4x64 + gradient
  refinement, `ci_grad.py`); if it trains a soil-only specialist, standing start with CEM (+ gradient if the specialist
  is loadable by `ci_grad`). The approach route and the decision state are model-independent, so one pass 1 per group
  serves every arm.
- **Rigid = secondary.** Speed-free goal reached is at the ceiling (section 1.3). Primary rigid read-out = unsafe at a
  fixed 2 m/s (geometry-only CEM from a standing start; `ga_planner.py --fixed2`, line 431; `ci_planner.py` passes
  unknown options through to `ga_planner`, but a fixed-2 run with `ci_train` checkpoints has never been done
  **[unverified]**; `ci_grad.py` has no fixed-speed option, lines 529-545). Speed-free rigid kept as a no-harm check.
  Expected rigid effect is small: the f104-only model's 2 m/s gap to f104 was only 3.4 points in gen_v1.

### 3.6 Arms and baselines (every arm of a group plans from the same state; picks hashed before any drive)

| arm | worlds | purpose |
|---|---|---|
| M1a, M1b, M3a, M3b, M2, A2, A3 | both | the comparison |
| frozen deployed f104 model (`K2/deploy_v1/deploy_a1_haux_gru_s*.pt` + gradient) | both | status quo; equals M1 if the recipe and data are unchanged (then drive once) |
| straight route: 6 m/s speed-free, 2 m/s in the fixed-speed comparison | both | no-model difficulty floor per arena |
| random pick from the same round-0 pool | both | model-free control with the identical loop (nav_v1 `--pick random` idea, `scripts/nav_runner.py:81`) |
| hand rule (22 features, pairwise logistic) refit on the 1-arena and on the 3-arena training rows, argmin over the same CEM round-0 pool | rigid (soil optional, needs a soil refit) | does more arenas help only the network? |
| feasibility ceiling = any of 12 designed routes reaches the goal (50 groups per test arena) | both | upper bound, and the offline AUC set |
| oracle of arms (best arm per group) | both | free ceiling from existing drives |
| in-arena reference: M1 (never saw T2/T3) vs M3/A3 (trained on them) on the T2/T3 held-out groups | both | what having the arena in training buys, i.e. the in-arena upper bound, without collecting on a test arena |

A true "trained on the test arena" oracle would need a full collection on a test arena; not proposed.

### 3.7 Metrics and statistics (pre-register in a PLAN.md with sha256 before any test drive)

- **Primary**: soil goal reached, pooled over the 6 test arenas, M3 (matched) vs M1, and A3 (additive) vs M1 as a
  co-primary; Holm over the two. Decision: one-sided 95 % lower bound of the paired group-bootstrap improvement > 0
  (`ga_analyze.py` `p05_one_sided_pts` with the contrast written M1:M3 on the fail label), exact McNemar reported.
  Primary groups = all 1,500; hill/crater-only reported as a stratum.
- **Robustness**: clustered bootstrap over (arena, nearest feature) clusters (about 6 x 10 = 60; needs the multi-arena
  variant of `n2_cluster_ci.py`); per-arena differences with the count of arenas improving (6/6 has two-sided sign p
  0.031, 4/4 only 0.125); optional mixed logistic model (arena fixed effect, group random intercept).
- **Secondary**: dose response M1 -> M2 -> M3 (paired bootstrap of the slope per added arena); unsafe, tilt30; time
  (median paired ratio on joint successes, bound 1.10, `ga_analyze.time_ratio`); positive work; generalisation gap per
  model = unseen-arena rate minus f104 suite rate (bootstrap, groups resampled within arena); offline within-group AUC
  on the feasibility subset and on T2/T3 val groups.
- **In distribution (no-harm)**: on the f104 800-group suite, M3 and A3 vs M1 non-inferior with a 2-point margin
  (`--margin-pts 2`; M1 sits at 97.5 % soil, so a 2-point loss is detectable); same on T2/T3 held-out groups where M1
  is the out-of-arena reference.
- **Rigid**: fixed-2 unsafe with the same contrasts, reported as secondary with its own power statement.

### 3.8 Smallest suite that can see 3-5 points

Paired binary outcomes, McNemar normal approximation (Connor), one-sided alpha 0.05, power 0.8. psi = fraction of
groups where the two arms disagree. Numbers are groups; two-sided alpha 0.05 in brackets.

| difference | psi 5 % | psi 8 % | psi 10 % | psi 15 % | psi 20 % |
|---|---|---|---|---|---|
| 3 points | 342 [434] | 548 [696] | 685 [870] | 1,029 [1,306] | 1,372 [1,742] |
| 4 points | 192 [243] | 308 [391] | 385 [489] | 578 [734] | 771 [979] |
| 5 points | - | 196 [249] | 246 [312] | 369 [469] | 493 [626] |

Terrain clustering inflates these by a design effect (my assumption 1.5; on the f104 suite the 9-cluster CIs were 0.7x
to 5.8x the group-bootstrap variance, too few clusters to trust **[unverified for multiple arenas]**). Detectable
difference at 1,500 groups, design effect 1.5: 2.5 points at psi 10 %, 3.0 at psi 15 %, 3.5 at psi 20 %. Expected psi:
competent same-arena soil models disagree on 3-8 % of f104 groups at 3-7 % failure; on unseen arenas failure and
disagreement will be higher, and models trained on different data disagreed on 16-17 % (section 1.3). So:
**1,500 groups (6 x 250) detects about 3 points; 1,200 (4 x 300) detects 2.8-3.9 points (psi 10-20 %); 800 detects
3.4-4.8.** Averaging two ensembles per condition lowers psi a little.

**Headroom pilot first (dev arena, before any test drive)**: drive the frozen deployed f104 model and the straight
6 m/s route on 200 dev-arena groups in soil (and fixed-2 rigid). If the deployed model is already within 3 points of
its f104 level (97.5 %) on the dev arena, a 3-5 point improvement cannot exist on soil speed-free either; then switch the
primary to soil at a fixed 2 m/s or to hill/crater-only groups before locking the plan. The rigid gap in gen_v1 (+1.3
speed-free, +3.4 at 2 m/s) suggests rigid alone will not carry the question.

---------------------------------------------------------------------------------------------------------------------

## 4. Proposed evaluation for task B (Gator on f104)

### 4.1 What the Gator changes (smoke-checked, source build)

| | HMMWV_Full (current) | Gator (default) |
|---|---|---|
| mass | 2,573 kg | 906 kg |
| wheelbase | 3.378 m | 2.776 m |
| driven wheels | 4WD, open differentials (`ShaftsDriveline4WD`, axles 0 and 1) | rear only (`GatorCustomDriveline`, axle 1; the model's SIMPLE driveline is a limited-slip differential, bias 2.0, `Gator_SimpleDriveline.cpp:28`) |
| wheel radius | 0.470 m | 0.286 m front, 0.318 m rear |
| min turning radius | 7.62 m | 7.60 m (the planner's 0.125 /m curvature limit stays feasible) |
| engine | map, 793 N m peak | EngineSimple 200 N m, 14 kW, 3,500 rpm; forward ratio 0.07 (`Gator_EngineSimple.cpp:25-27`, `Gator_AutomaticTransmissionSimple.cpp:23`) -> no-load top speed about 8 m/s **[computed, not driven]** |
| brake | 4,000 N m per wheel | 800 N m (`Gator_BrakeSimple.cpp:29`) |

Difficulty will change in both directions (lighter, limited-slip, but rear-drive only and small wheels), so absolute
goal-reached numbers are not comparable across vehicles without anchors.

Vehicle-specific pieces inside the evaluation path that must be re-parameterised, not just the collector:
`ga_approach.py:79` (`WHEEL_OFFSETS`, HMMWV spindle positions for the CRM sinkage/moving check),
`crm_collect.py:45` (HMMWV rigid tyre mesh for the soil coupling; Gator ships `gator/gator_wheel_*.obj` rigid-tyre meshes,
their suitability for the soil coupling **[unverified]**), `crm_collect.py:374` (soil-breakthrough rule uses the tyre
radius), the start footprint check in `gen_cases.py:91-93` (HMMWV footprint; a smaller vehicle only makes it looser),
`gen_collect.py:274` (runtime fingerprint requires HMMWV data), and the history-channel normalisation (wheel omegas
about 1.5x larger at the same speed because of the smaller wheels).

### 4.2 Suite and protocol

The same 800 groups (`K1/A_adapt/suite/cases`, unchanged files, same `SUITE_LOCKED` hash), the same approach routes,
the same candidate family and validator, the same goal radius (2.5 m), parking rule (3 m), stop policy and labels.
Protocol per trained model as in 3.5: shared history model -> 0.5 s approach + gradient; single-world specialist ->
standing start + CEM (+ gradient). Keep the follower gains unchanged if the Gator tracks acceptably; if they are
re-tuned, say so and report tracking error per vehicle on the designed routes (`gb_track_analyze.py` metrics). Rigid
headroom: if the Gator is also at the ceiling speed-free, report fixed-2 m/s unsafe as for task A.

### 4.3 Arms per world (all from the same decision state per group, hashed before driving)

| arm | purpose |
|---|---|
| Gator-trained planner, final configuration (gradient pick G) and its CEM pick B (drive B only where it differs, about 83 % of groups in K2) | the result |
| **frozen HMMWV final model driven by the Gator** (`deploy_a1_haux_gru` + gradient, 0.5 s approach) | transfer baseline; its history input is out of distribution (wheel speeds, engine speed, rear drive) |
| frozen HMMWV geometry-only specialist driven by the Gator, standing start (`crm_f104_v1/train_v1/deploy/CRM_N2_s*.pt` on soil, `R/night2_v1/final/N2_s*.pt` on rigid) | cleaner transfer baseline: does HMMWV terrain knowledge carry over without the history shift |
| straight 6 m/s (and straight 2 m/s at fixed speed, rigid) | vehicle difficulty floor |
| random pick from the same pool | model-free control |
| hand rule refit on Gator rows | non-learned baseline (rigid; soil optional) |

**Primary B contrast**: Gator-trained vs frozen HMMWV-trained, both driven by the Gator, soil goal reached, 800 groups,
exact McNemar + paired group bootstrap (the CRM-night-1 design, where soil-trained beat rigid-trained 91.0 vs 68.0 %).
"Works" (pre-register): (1) the Gator planner beats straight 6 m/s and the random pick on the Gator (one-sided lower
bound > 0); (2) it beats or is non-inferior (2-point margin) to the frozen HMMWV model on the Gator; (3) offline
within-group AUC on the 111 held-out twin groups >= 0.95 (HMMWV 0.97-0.99); (4) it closes a comparable share of the
headroom as the HMMWV planner does on the HMMWV (4.4).

### 4.4 Comparing with the HMMWV fairly

1. **Same groups, report both vehicles side by side** (goal reached, unsafe, time), paired by group; a cross-vehicle
   McNemar compares vehicle+planner systems, not planner quality, and must be labelled that way.
2. **Difficulty anchors per vehicle on the same 800 groups**: straight 6 m/s, random pick, and the feasibility ceiling.
   The HMMWV anchors do NOT exist on the 800 suite (straight/random were driven only on the 200 reused groups in CRM
   night 1: straight 6 m/s 66.5 %, straight 2 m/s 13.0 %), so drive HMMWV straight-6 and random on all 800 in both
   worlds (cheap, same jobs).
3. **Normalised effect**: headroom closed = (planner - straight6) / (ceiling - straight6) per vehicle, and failure
   relative to the random pick; bootstrap both by group. Report the feasible stratum (groups where some route works for
   that vehicle) separately.
4. **Ceiling / difficulty from the collection itself**: the Gator collection replays the 1,200 twin groups x ~12.5
   routes with identical routes and splits, so per-route Gator vs HMMWV outcomes (by speed profile) and "any designed
   route reached" per group come for free and give the difficulty shift without extra drives.
5. **Decomposition**: vehicle effect at a fixed planner (frozen HMMWV model on HMMWV vs on Gator) and data effect at a
   fixed vehicle (Gator-trained vs HMMWV-trained on the Gator).
6. **Ranking quality**: within-group AUC on held-out twin groups for each vehicle's model on its own labels is the
   least difficulty-dependent comparison.

---------------------------------------------------------------------------------------------------------------------

## 5. Number of closed-loop drives and cost

Cost basis: soil about 0.4 billed node-hours per simulated hour (K1 `PLAN.md:17`; consistent with K2's 39.0 billed for
about 20,000 soil episodes, i.e. about 2 billed per 1,000 drives of ~18 s); soil throughput about 23-36 simulated hours
per wall hour on 60-111 GPUs (memory `crm-f104-night-state`); rigid about 0.02 billed per simulated hour and minutes of
wall time (the 66 h rigid dataset took 13.6 wall minutes). Mean drive length on the 800 suite 14-19 s (from K2 per-group
elapsed); assume 19 s incl. settle for soil on unseen arenas (more failures). Queue cap 50 tasks per user: one tasks
file per world per wave. Planning: `ci_grad` 2.6 s per decision (5090, 0.45 GB), CEM about 0.4 s; determinism of
`ci_grad` was self-tested on CUDA only, not on ROCm **[unverified]**, so plan picks locally or check on the cluster
first.

**Task A, full design (1,500 test groups):**

| item | soil drives | rigid drives |
|---|---|---|
| pass 1 (0.5 s approach) test + T2/T3 held-out + dev | 2,200 | 2,200 |
| test arms: 7 model ensembles + straight + random (soil); + 2 hand rules, fixed-2 set (rigid) | 13,500 | about 22,500 |
| in distribution: f104 suite x 4 arms (M2, M3a, A2, A3), T2/T3 500 groups x 4 arms | 5,200 | 5,200 |
| dev headroom pilot 200 x 2 | 400 | 400 |
| feasibility subset 300 groups x 12 designed routes (rigid: all 1,500 x 12) | 3,600 | 18,000 |
| **total** | **about 25,000 drives, about 130 simulated h, about 50-55 billed, 4-6 wall h** | **about 48,000, about 5 billed, < 1 wall h** |

Planning decisions: soil about 15,000 gradient decisions (11 GPU-h serial; run 4-6 processes in parallel on the 5090 or
on MI350X), rigid about 4,500 gradient + 16,500 CEM (about 5 GPU-h).

**Task A, lean design (1,200 test groups = 4 x 300; arms M1, M3, A3, straight; single ensembles; no feasibility
subset):** soil about 8,200 drives (43 simulated h, about 17 billed, 1.5-2 wall h); rigid about 12,000 (well under 1
billed). Detects about 2.8-3.9 points (psi 10-20 %, design effect 1.5).

**Task B (per world, 800 groups):** Gator pass 1 800 + about 6-7 arms x 800 = about 6,400 soil drives (Gator times
unknown; at 20 s about 36 simulated h, about 14 billed, 1-1.5 wall h) plus HMMWV anchors (straight-6 and random on 800)
1,600 soil drives (about 3 billed); rigid about 10,000 drives including fixed-2 arms (about 1 billed). The difficulty
calibration on designed routes comes from the Gator collection itself.

These evaluation costs sit on top of the collections: soil data on two more arenas at the f104 amount is about
2 x 91.5 simulated hours (about 75 billed) and a Gator soil set of the HMMWV size about 91.5 simulated hours (about 37
billed), if the Gator runs at the HMMWV's real-time factor **[unverified]**. The full evaluation of both tasks
(about 75 billed) plus both collections (about 110 billed) exceeds a 100-node-hour night; the lean A design plus B
(about 35 billed) fits beside the collections only if the collections are cut.

---------------------------------------------------------------------------------------------------------------------

## 6. Unverified or open

- Soil headroom on unseen arenas for the f104-trained model is unknown (no soil run on any other arena yet); the whole
  power argument for A depends on the dev-arena pilot.
- Disagreement rates (psi) on unseen arenas and the multi-arena design effect are extrapolated.
- Fixed-2 m/s planning with `ci_train` checkpoints through `ci_planner.py` has never been run; `ci_grad.py` has no
  fixed-speed mode.
- Gator: top speed on slopes, CRM tyre-mesh suitability, real-time factor on soil, follower tracking with HMMWV gains,
  settle height check (`crm_collect.py:425-427`) all untested.
- `ci_grad` determinism on ROCm untested.
- The gen_v1 arena-similarity code is not in `scripts/`; only its output and scale vector are.
- The five-goal mission runner (`gen_mission_runner.py`) was built for the frozen geometry-only model; whether it can
  drive a history model with gradient refinement is untested, so missions are exploratory for both tasks.
