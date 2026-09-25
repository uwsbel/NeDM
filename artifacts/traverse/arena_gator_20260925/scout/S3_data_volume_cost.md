# S3 scout: data volume and cost (for task A, more arenas, and task B, Gator on f104)

Read-only scout, 2026-09-25. Local root `L = /home/harry/NeDM-traverse_mppi`. Cluster roots
`R = /work1/dannegrut/harry/experiments/fdm_f104_50h_20260909`, `C = .../crm_f104_20260916`,
`G = .../generalist_20260921`, `G2 = .../crm_improve_20260922`.
Every count and hour below was recomputed from the local run folders (`outcome.json` `elapsed_s` / `wall_s`), the
dataset build records, or `sacct`, unless it is marked **[record only]** or **[UNVERIFIED]**.
Billed node-hours are recomputed from `sacct` with the partition weights (method in section 2.1).

---

## 0. Answers in brief

- **The final HMMWV planner (soil 97.5 %, rigid 100 %) was trained on 217,814 rows. Those rows are re-cuts of only
  15,024 rigid + 15,024 soil episodes, plus 5,414 branch drives.** The rigid episodes total 125.1 simulated h and
  the soil episodes 89.8 h. So each training episode appears about 7 times, at different decision frames.
- **Rigid data is almost free.** All 35,412 rigid f104 episodes (269 simulated h) cost 5.2 billed node-hours
  (13-33 min of wall time per wave). **Soil data is the cost.** The 91.5 h soil set cost 31.8 billed node-hours and
  2 h 40 min on 111 GPUs, about 0.35 billed per simulated hour.
- **Sensible meaning of "the same amount of data": the same episode design.** That means the same start-goal groups
  and route task list, drawn tier by tier and stopped at the same number of routes per group. Report simulated hours
  and training rows as outcomes, not targets (section 1.5).
- **Estimates (collection only, at HMMWV rates):**

  | Case | Rigid | Soil |
  |---|---|---|
  | One extra arena, full volume | ~4 billed, <1 h wall | ~32 billed, 2.6 h on 111 GPUs or ~5 h on the ~52 GPUs idle today |
  | Half volume | ~2 billed | ~16 billed |
  | Third volume | ~1.2 billed | ~11 billed |
  | Gator on f104, full volume | ~4-7 billed | ~32-64 billed (depends on how long Gator episodes last) |
- **Allocation.** 639.0 of 1,500 billed node-hours used account-wide (harry 161.0), so 861 remain. The queue is empty.
  The limit that binds is wall time and GPU availability, not budget (section 4.4).

---

## 1. What the HMMWV planner on f104 was trained on

### 1.1 Rigid ground: the three f104 collection waves (all local copies under `L/artifacts/traverse/fdm_f104_50h_20260909/`)

| Wave | Folder | Groups | Routes per group | Episodes | Sim h | Mean s/ep | Failed (not goal) |
|---|---|---|---|---|---|---|---|
| Night 1 "50 h" | `production_v2/runs` (ids `f104_v1_group_*`) | 1,500 | **5-10** (87/270/348/381/267/147 groups with 5/6/7/8/9/10) | 11,412 | 66.42 | 20.95 | 11.1 % |
| Night 2 A1, designed | `production_v3/runs` (`f104_v2_group_*_route_*`) | 1,200 | 12 | 14,400 | 82.98 | 20.74 | 10.7 % |
| Night 2 A2, on-policy | `production_v4/runs` (`f104_v2_group_*_op_*`) | same 1,200 | 8 | 9,600 | 119.97 | 44.99 | 33.7 % |
| **Total** | | 2,700 | | **35,412** | **269.4** | | |

- **Correction to the memory note.** `f104-fixed-arena-study-state.md:18` calls night 1 a "complete factorial,
  1,500 groups x 12 routes". It is not. 18,000 designed tasks existed (6,000 initial plus 12,000 reserve,
  `design/README.md`). The collection stopped at its quota of 66.4 h after 13.6 min, and every group ended with 5-10
  of its 12 routes (`production_v2/progress.json`: `quota_reached: true`, 11,412 episodes). `docs/progress.md`
  ("36,199 driven routes, about 200 h simulated") also understates the hours: the three waves alone hold 269.4 h.
- **Designed routes.** `scripts/generate_traverse_f104_collection.py:163-193` defines them, and
  `f104_n2_cases.py` / `gen_cases.py` copy them. Each group gets:
  - 3 lateral offsets (0, -4, +4 m) x 4 speed profiles: constant 2, 4 and 6 m/s, and a smooth 2-6-2 m/s profile;
  - speed capped by a stopping cone, sqrt(4 (L - s)) (`:78`);
  - curvature cap 0.10 /m and arena margin 36 m (`:167`).

  Start and goal pairs come from 6 strata drawn in turn: hill cross-slope, crater cross-slope, hill
  entry/cross/exit, crater entry/cross/exit, long traverse and roughness transfer. The start pad is 6 x 3 m, with
  grade <= 6 deg fitted and <= 12 deg maximum. The horizon is 120 s after a 0.8 s settle. Splits are 90/5/5 by an
  md5 hash of the group, so all routes of a group share a split.
- **On-policy routes.** `scripts/f104_n2_onpolicy.py` draws 8 routes per group from the planner's own proposal
  (`f104_n2_sampler.sample_one`: 3-mode sine lateral offset plus 4 speed knots). No network is involved. The
  settings are `MPPIConfig(max_speed_mps=6, max_curvature_inv_m=.125, arena_half_extent_m=40)` (`:17`) and
  `base_speed=2.0` (`:34`).
- **Merged rigid tensors** (`night2_v1/station_ds_all.npz`, 1.1 GB): 36,199 rows.

  | Wave | Rows | Where they come from |
  |---|---|---|
  | night1_50h | 12,542 | 11,721 designed (11,412 production_v2 + 309 prospective) + 821 Chrono MPPI variants; `night_v1/LOG.md:4` |
  | night2_A1 | 14,348 | designed |
  | night2_A2 | 9,309 | on-policy; 211 stragglers were not finished when the file was built, `night2_v1/LOG.md:40-43` |

  Split: train 33,840 / val 764 / test 1,595. The deployed rigid specialist `night2_v1/final/N2_s*.pt` was fitted
  on **31,851** rows (`N2_meta.json` `n_fit`). That is the training split minus the 1,989 non-designed rows of the
  dev fold (`scripts/f104_n2_deploy.py:21`, `f104_n2_train.py:81-82`).

### 1.2 Soil (CRM): `crm_f104_v1/collect_v1`

- **Tasks.** 24,000 = the same 1,200 night-2 groups x 20 routes (12 designed + 8 on-policy), with ids equal to the
  rigid ids (`scripts/crm_tasks.py`). Each group's routes were put in random order, and tier k holds the k-th route
  of every group. The collection stopped via `STOP_CLAIMS` at 90.6 h (`crm_f104_v1/LOG.md:19`).
- **Validated episodes** (`collect_v1/qa.json`): **15,235 = 91.51 simulated h**.
  - Groups end with 12 or 13 routes (835 groups x 13, 365 x 12).
  - Designed 9,168 (44.7 h), on-policy 6,067 (46.8 h).
  - Train 13,821 episodes / 83.08 h / 1,089 groups; val 713 / 56 groups; test 701 / 55 groups.
  - 68.1 % did not reach the goal.
  - Mean episode length by outcome: goal 16.6 s, soil breakthrough 20.9 s, blockage 36.4 s. 69 h of the 91.5 are
    bogged episodes (`REPORT.md:16`).
- **Soil specialist** (`train_v1/deploy/CRM_N2_s*.pt`): trained on `datasets/station_ds_crm_v1.npz`, 15,235 rows,
  of which 13,821 were fitted.
- **211 soil routes have no rigid twin** (`crm_f104_v1/REPORT.md:148`). This is why the shared-model data below has
  15,024 episodes per world.

### 1.3 Data added by the generalist (09-21/22) and soil-improvement (09-22/24) efforts

| Data | Drives (new Chrono) | Sim h | Rows | Built by | Used by the final model? |
|---|---|---|---|---|---|
| Re-anchored rows, rigid (`crm_night2_v1/datasets/reanchor_rigid.npz`) | none (re-cut of the 15,024 twins) | - | 58,424 | `n2_reanchor_dataset.py` | yes, inside the mixed file |
| Re-anchored rows, soil (`reanchor_crm.npz`) | none | - | 57,444 | same | yes |
| Mixed file + 2 s history (`generalist_20260921/A_adapt/datasets/mixed_reanchor.npz`) | none | - | 115,868 (anchors k = 0 and multiples of 40 frames) | `ga_build_mixed.py` | yes |
| Rigid branch drives (A4, `A_adapt/a4/rigid_runs`) | 3,158 (including prefixes) | 24.85 | 2,358 (789 anchors x 3) | `ga_branch_anchors.py` -> `gen_collect_ext.py --mode branch_auto` -> `ga_branch_dataset.py` | yes |
| Soil branch drives (A4, `A_adapt/a4/crm_pass2_runs`) | 800 pass-1 prefix replays + 2,256 pass-2 | 13.18 (pass 2) | 2,256 (752 anchors x 3) | two passes, `crm_collect_ext.py` + `ga_branch_continuations.py` | yes |
| Early decision rows (`crm_improve_20260922/datasets/short_anchor.npz`) | none | - | 89,998 (frames 10/20/30 = 0.5/1/1.5 s) | `ci_short_anchors.py --ks 10 20 30` | yes |
| 2-4 s decision rows (`anchor_k40_60_80.npz`) | none | - | 88,797 | `ci_short_anchors.py --ks 40 60 80` | only the k = 60 slice |
| 3 s decision rows (`anchor_k60.npz`) | none | - | 29,699 (rigid 14,972 + soil 14,727) | k = 60 slice of the above. **[extraction command not recorded; counts match the slice]** | yes |
| 3 s continuations (`cont3_{crm,rigid}.npz`) | 7,182 soil + 7,182 rigid | 45.32 + 66.85 | 14,364 | `ci_a5data.py` + collectors | **no** (no offline gain, `crm_improve REPORT.md:68-76`) |

### 1.4 The final model's training files (`crm_improve_20260922/deploy_v1/deploy_a1_haux_gru.json`)

| File (on G2/data) | Rows | Rows used (test split removed) | Fit rows (val split removed) |
|---|---|---|---|
| `mixed_reanchor_plus_branch_both.npz` (115,868 + 2,358 + 2,256) | 120,482 | 114,929 | 109,244 |
| `short_anchor.npz` | 89,998 | 85,844 | 81,634 |
| `anchor_k60.npz` | 29,699 | 28,326 | 26,936 |
| **Total** | 240,179 | 229,099 | **217,814** |

- **Fit rows by world:** rigid 109,492, soil 108,322.
- **Fit rows by source:** designed 130,051, on-policy 83,712, branch 4,051.
- **Model:** 262,018 parameters; 5 seeds x 30 epochs, ~800 s per seed on an MI350X; 16.3 GB of data on the GPU.
- **Earlier model for comparison:** the generalist's `H_deploy` was fitted on `mixed_reanchor.npz` only, 105,193 rows.
- **Underlying unique episodes** (recomputed from the local `outcome.json` files):

  | World | Episodes | Sim h | Failed | Where they come from |
  |---|---|---|---|---|
  | Rigid | 15,024 | **125.11** | 19.4 % | 9,135 `production_v3` + 5,889 `production_v4` |
  | Soil | 15,024 | **89.84** | 67.8 % | `collect_v1` |
  | Branch drives | 5,414 | ~38 | | rigid 3,158 + soil 2,256; the 800 soil prefix replays are not counted |

- **Not in the final model:**
  - night-1 `production_v2` (11,412 episodes);
  - the rigid-only extras (on-policy stragglers, the 52 missing A1 routes, and every A1/A2 route whose group
    exceeded the soil tier count);
  - the 211 soil routes without a twin;
  - `cont3`;
  - `gen_v1` sibling-arena data;
  - every evaluation drive.

### 1.5 What "the same amount of data" can mean

| Unit | HMMWV f104 value | Behaviour for a new vehicle or arena |
|---|---|---|
| Raw episodes (design) | final model: 15,024 per world (soil 15,235 collected; rigid 24,000 in the night-2 pool, 35,412 in all waves) | Fixed before collection. Paired with the HMMWV route by route when the ids match. |
| Simulated hours | final model: rigid 125.1 h, soil 89.8 h (collected: rigid 269.4 h, soil 91.5 h) | An outcome. It depends on how often and how long the vehicle stalls: blockage episodes last 36-56 s, successes 16 s, the on-policy tier 45 s versus 21 s. A vehicle that stalls more "earns" more hours with less information. |
| Training rows | 217,814 fit (the specialists: rigid 31,851, soil 13,821) | Derived deterministically from episodes. Anchor admission depends on outcomes (e.g. soil k = 80 rejects 3.3 % against rigid 0.7 %). |

**Recommendation.**

- **Task B (Gator on f104): match the design.** Use the same 1,200 `f104_v2` groups and the same task list with
  the same ids and tier order. On soil, stop at the same 12-13 routes per group (15,235 episodes). On rigid, run at
  least the 15,024-twin set, better the full 24,000 pool. Keep the same branch-anchor recipe. Then report simulated
  hours and rows as they come out. This keeps every Gator episode paired with an HMMWV episode on the identical route.
  "Same hours" would not be the same information: it would buy fewer episodes if the Gator bogs down more.
- **Task A (new arenas): per arena, match the design** (1,200 new groups x the same 20-route recipe from
  `gen_cases.py` + `f104_n2_onpolicy.py`, same tiering).
  - **Control total volume, or "more arenas" is confounded with "more data".** Night 2 showed ranking quality was
    still data-limited: dev AUC 0.94 on 25 % of the rows, 0.98 on 100 % (`night2_v1/LOG.md:28-31`).
  - The clean contrast is 1 arena x V against 3 arenas x V/3, with the same total episodes. Add 3 x V (more data
    and more arenas) only if the budget allows.
  - Existing rigid sibling data is a small head start: `gen_v1/data`, 1,800 designed routes per arena on
    g203/g216/g217/g228/g231, 150 groups x 12, 55.9 h in total, no on-policy tier.

---

## 2. Cost of past collections

### 2.1 Billing method (verified today)

- **The ledger.** `ssh amd slurm_balance2.py` reads `/share/accounting/allocation_usage.json`. On 2026-09-25 it
  shows `billing_total` 639.0 of 1,500 and `node_hours_total` 1,739.8. Per user: slaton 325.9, **harry 161.0**,
  kyle 81.8, dannegrut 38.3, auc7us 32.0.
- **Partition weights.** `scontrol show partition` gives `TRESBillingWeights=NODE=`: mi2101x 0.01, mi3001x and
  mi3501x 0.0125, mi2104x 0.04, mi2508x 0.08, mi3008x 0.1, mi3258x and mi3508x 0.12, devel 0.01.
- **Scale.** The ledger's unit is 10x these weights per raw node-hour, the same as
  `generalist_20260921/tools/my_spend.sh`. Checked against the ledger:
  - `sacct -a -A dannegrut` summed over the account gives node-hours of exactly 1,739.8;
  - the 10x weights reproduce auc7us (32.0) and kyle (81.8) exactly;
  - they under-count harry (146.7 against 161.0), slaton and dannegrut by 10-20. **[discrepancy unexplained;
    treat my estimates as +-10 %]**
- **Billed per GPU-hour:** 0.10 (mi2101x, mi2104x, mi2508x), 0.125 (mi3001x, mi3501x, mi3008x), 0.15 (mi3508x).
- **Billed per CPU core-hour:** 0.003 (mi2104x) to 0.006 (mi2101x).

### 2.2 Per collection (sacct, 10x weights; job ids from the records)

| Collection | Episodes | Sim h | Wall | Raw node-h | **Billed** | Billed per sim h |
|---|---|---|---|---|---|---|
| Rigid `production_v2`: jobs 412450/1/4/5/6/80/81 on 7 partitions, 2,008 workers | 11,412 | 66.42 | 13.6 min | 8.47 | **1.57** | 0.024 |
| Rigid `production_v3` (A1): jobs 416128/29/56, 1,624 cores | 14,400 | 82.98 | ~24 min (01:31-01:56) | 8.53 | **1.61** | 0.019 |
| Rigid `production_v4` (A2): jobs 416167/68/69 | 9,600 | 119.97 | ~33 min (01:41-02:15) | 11.12 | **1.97** | 0.016 |
| Rigid `gen_v1` data, 5 arenas: jobs 420037/8/9 | 9,000 | 55.93 | ~47 min | 1.52 | **0.92** | 0.016 |
| Rigid `gen_v1` test arms: job 420022 | 6,639 | 37.14 | ~44 min | 8.34 | 0.83 | 0.022 |
| **Soil `collect_v1`**: jobs 423618, 423642-423647; 111 GPUs = 24 mi2101x + 7x4 mi2104x + 3x8 mi2508x + 2x8 mi3508x + 1x8 mi3008x + 5 mi3001x + 6 mi3501x | 15,235 | 91.51 | 2 h 40 min (23:59-02:39, including the drain) | 123.7 (~284 GPU-h) | **31.75** | **0.347** |
| Soil night 1 in full (pilot 1.1 + collection 31.75 + dataset/training 0.02 + evaluation 2.36 for 1,240 drives) | | | | 141.0 | 35.3 (the record says ~37, taken from the account-wide change 396 -> 433, `REPORT.md:17`) | |
| Rigid branch drives A4 (`rigid_a4`) | 3,158 | 24.85 | ~7 min | 2.0 | 0.80 | 0.032 |
| Soil branch drives A4 (pass 1 + pass 2 + pass2_8) | 800 + 2,256 | 13.18 + prefixes | ~1.5 h | 21.2 | 4.89 | ~0.37 |
| Soil 3 s continuations (`ci_cont3`) | 7,182 | 45.32 | 10:33-15:10 (queued behind other work) | 92.3 | **15.11** | 0.333 |
| Rigid 3 s continuations (`ci_rcont3`) | 7,182 | 66.85 | ~5 min | 2.2 | 0.88 | 0.013 |

**Whole efforts** (my `sacct` windows against the records):

| Effort | sacct billed | Record | Note |
|---|---|---|---|
| Soil night 1 | 35.3 | ~37 | |
| Night 2 | **3.1** | **21.9** (`crm_night2_v1/REPORT.md:39`) | That figure equals the raw node-hours, not billed |
| Generalist | 28.7 | 29 | |
| Soil improvement | 39.0 | 39.0 | |

About 20 of the soil-improvement effort's billed hours were closed-loop evaluation drives:

| Jobs | Soil drives | Billed |
|---|---|---|
| `ci_s2` | 5,464 | 9.7 |
| `ci_s2n` | 3,200 | 4.7 |
| `ci_e2` | | 3.0 |
| `ci_s4` | | 2.4 |

That works out to **~1.5-1.9 billed per 1,000 soil evaluation drives** and ~0.1-0.2 per 1,000 rigid drives.

**Measured throughput:**

- **Rigid**, one core per episode. Wall time = ~20-23 s fixed + 3.1-3.2 s per simulated second, fitted over the
  v3/v4 episodes.
- **Soil**, one GPU per episode. Wall time = 1.4 s fixed + 2.81 s per simulated second, averaged over the GPU mix
  (fit over collect_v1). Real-time factor 0.32 on MI210 and ~0.5 on MI350X (`crm_f104_v1/REPORT.md:48-50`).
- **Rates.** About 36 simulated soil hours per wall hour on ~111 GPUs, and ~23 per wall hour on ~60 GPUs
  (memory note).

### 2.3 Allocation and queue today (2026-09-25)

- **Allocation.** `alloc_dannegrut_06222026_06302027`: **639.0 / 1,500 used (42.6 %), 861 left**, shared with 4
  other users. My own spend on the f104 line since 09-08 is 133.2 billed.
- **Queue.** `squeue -u harry`: **0 jobs**. No harry jobs since 2026-09-24 12:00.
- **Idle nodes** (`sinfo`): mi2104x 7 (28 MI210), mi2101x 17 (17 MI210), mi3501x 7 (7 MI350X), devel 4. So
  **~52 GPUs are free right now**.
- **Busy or down.** Every 8-GPU node is allocated to other users: mi2508x 8 allocated + 2 down, mi3508x 4/4, mi3008x
  2/2, mi3258x 1/1. mi3001x has 2 allocated and 5 down.
- **Queue cap of 50 queued tasks per user** (array tasks count): recorded, not visible in `sacctmgr` (a lua
  submit filter, `JobSubmitPlugins=lua,require_timelimit` in `/etc/slurm/slurm.conf:172`). **[record only:
  memory `crm-f104-night-state.md`; `generalist_20260921/scout/cluster_data_inventory.md:9`]**
  `scripts/crm_launch.sh:15-21` alone submits 47 tasks.
- **Walltime caps:** mi3501x/mi3001x 4 h; mi2101x/mi2508x/mi3008x/mi3258x/mi3508x 12 h; mi2104x 24 h.
  **[record only: dry run on 09-16, `cluster_data_inventory.md:10`]** `MaxTime` in slurm.conf is 96 h, so the caps
  come from the filter.
- **Every partition is `OverSubscribe=EXCLUSIVE`.** Rigid CPU jobs on mi2104x/mi2101x therefore occupy the same
  nodes the soil collection needs.

---

## 3. Minimal pipeline from raw episodes to the final training files

**Legend:** [A] = step depends on the arena; [V] = depends on the vehicle.
Paths are relative to `L`. Commands are the recorded ones where a record exists.

| # | Step | Script / command | A/V notes |
|---|---|---|---|
| 0 | Arena asset (BMP + `arena_meta.json`) | `assets/traverse/arena_*`. f104 plus 9 siblings exist: g203 g204 g213 g216 g217 g223 g228 g231 g234. | [A] The rigid collector and the map capture only accept BMPs whose sha256 is listed in `scripts/gen_arenas.json` (`gen_collect.py:247-249`, `gen_collect_ext.py:500-502`, `sensor_capture_map.py:29-30`). Listed today: f104, g228, g203, g217, g216, g231. **g204, g213, g223 and g234 must be added.** |
| 1 | Static overhead depth map, vehicle-free: `<root>/static_map_v1/observation.{npz,json}` | Cluster: `sensor_capture_map.py` (lavapipe). Local: `crm_capture_map_local.py` (OptiX on luffy, 7 ms per frame). | [A] Existing captures: `fdm_f104_50h_20260909/sensor_v1/maps/arena_{f104_50h_v1,g203,g216,g217,g228,g231}` and `crm_f104_v1/map_root/static_map_v1`. The labeller reads only this (`f104_n2_dataset.py:17-24`, camera from `observation.json`), so a new arena needs its own root. **[Using sensor_v1 captures as soil roots is structurally compatible but not verified]** |
| 2 | Start-goal groups + 12 designed routes | f104: `f104_n2_cases.py`. Any arena: `gen_cases.py --arena <dir> --strata all --avoid <case dirs>`. | [A] strata, gates, arena bounds. [V] speed profiles 2/4/6/2-6-2 m/s, curvature cap 0.10 /m, 6 x 3 m start footprint (`generate_traverse_f104_collection.py:78,163-193`). |
| 3 | 8 on-policy routes per group | `f104_n2_onpolicy.py --cases <cases> --out <dir> --n 8` (no network) | [A] via the cases and the arena bound. [V] max 6 m/s, curvature 0.125 /m (`:17`), base 2 m/s (`:34`). |
| 4 | Task lists | Soil: `crm_tasks.py` (20 routes per group, tiered, `episode_seed = md5(id)`). Rigid: shard lists for `gen_array.sbatch` -> `gen_runner.py`. | - |
| 5 | Collection | Rigid: `gen_array.sbatch` -> `gen_runner.py` -> `gen_collect.py`. f104's own frozen collector is `collect_traverse_f104.py` (f104 BMP hash at `:24`). Soil: `crm_launch.sh <tasks> <out> configs/crm_main.json [h]` -> `crm_collect.sbatch` -> `crm_worker.py` -> `crm_collect.py`. | [A] soil terrain comes from `case["arena"]` (`crm_collect.py:172,188`). **[V] HMMWV is hard-coded:** `src/nedm/hmmwv_data.py:294-297` (raises unless `HMMWV_Full`), `src/nedm/traverse/scene.py:354`, `crm_collect.py:160,184`; HMMWV tyre mesh for soil coupling (`crm_collect.py:45`, one mesh for all wheels at `:111-116`); runtime fingerprint requires `/vehicle/hmmwv/` (`gen_collect.py:274`, `collect_traverse_f104.py:272`, `gen_collect_ext.py:545`); follower gains and 5 m look-ahead in the frozen `traverse_fdm_rgbd_diverse_chrono.py:98-112`. A soil task row can carry per-task `extra` arguments and the collector is set by `CRM_COLLECTOR` (`crm_worker.py:91-99`), so a vehicle switch can travel in the task file. |
| 6 | Soil QA / validated hours | `crm_qa.py <collect dir> [--quarantine]` -> `qa.json` | [V] explosion thresholds (15 m/s, 2.5 m); breakthrough rule uses the tyre radius and soil depth (`crm_collect.py:289`). |
| 7 | Labels + corridor tensors | `f104_n2_dataset.py --root <map root> --runs "<runs>/*_route_*:designed" "<runs>/*_op_*:on_policy" --out X.npz` (soil version: `crm_dataset.sbatch`). Rigid waves are merged by `f104_n2_merge.py`. | [A] one map root per call. [V] event thresholds are in m/s (vx < -0.1/-0.3, near-stop |vx| < 0.3 at throttle > 0.3). |
| 8 | Twin id set (ids present in both worlds) -> `twin_{crm,rigid}.npz` (15,024 each) | **[builder script not found; recorded in `crm_night2_v1/REPORT.md:207`]** | Needed only because the shared model pairs the worlds. |
| 9 | Re-anchored rows | `n2_reanchor_dataset.py --root <map root> --ids twin_X.npz --runs <dirs> --out reanchor_X.npz` | [A] root. [V] admission depends on outcomes. |
| 10 | Mixed file + 2 s history + privileged context | `ga_build_mixed.py --rigid ... --crm ... --rigid-runs ... --crm-runs ... --out mixed_reanchor.npz` (defaults are the f104 paths) | [V] history = state columns 0-6 and 11-15 (spindle speeds, engine speed) plus actions; the scales are vehicle-specific. |
| 11 | Branch rows (optional; 4,051 of the 217,814 fit rows; the branch-trained model changed nothing closed loop, 83.9 %) | `ga_branch_anchors.py` -> rigid `gen_collect_ext.py --mode branch_auto`; soil pass 1 (`crm_collect_ext.py` prefix replay) -> `ga_branch_continuations.py` -> pass 2 `--mode branch` -> `ga_branch_dataset.py --runs ... --anchors ... [--merge <mixed> --merged-out <npz>]` | [A][V] new Chrono drives. Rigid runs all branches of one anchor on one node (determinism). |
| 12 | Early and 3 s decision rows | `ci_short_anchors.py --ks 10 20 30 --out short_anchor.npz`, then `--ks 40 60 80` (take the k = 60 slice) | [A] `--root` (default `crm_f104_v1/map_root`). ~1.5 min each on the local CPU. |
| 13 | Train | `ci_train.py --mode deploy --arch gru --hist-enc gru --cond hist_aux --ctx geom --crm-batch-frac 0.5 --hist-drop 0.2 --aux-weight 0.5 --seeds 5 --epochs 30 --bs 256 --ds <file 1> <file 2> <file 3>` (arguments from `deploy_a1_haux_gru.json`) | Arena-agnostic. [V] retrain from scratch; no weights carry over. |

- **Build order:**

  ```
  0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10 -> (11) -> 12 -> 13
  ```

  Steps 7-12 are local CPU work of minutes each; their build logs are in each `*_build.json`. Only steps 5 and 11
  need Chrono.
- **Multi-arena.** Steps 7, 9 and 12 take a single `--root`, so run them once per arena and pass several files to
  `ci_train.py` (it accepts several `--ds`). Ids stay unique because the case ids carry the arena prefix
  (e.g. `g203_data_group_*`).
- **Rebuild gen_v1 rows before reuse.** The existing `gen_v1/station_ds_gen_v1.npz` was built from heightmaps
  (`gen_build_dataset.py` -> `gen_planner.set_map`), not from the depth-map root. Rebuild it with step 7 for
  consistency.
- **Gator smoke** (local, conda pychrono 10.0.0, flat rigid ground, SMC, TMEASY, full throttle for 12 s;
  `/tmp/s3_gator_smoke2.py`):

  | | HMMWV | Gator |
  |---|---|---|
  | Mass | 2,573 kg | 906 kg |
  | Wheelbase | 3.38 m | 2.78 m |
  | Max steer | 0.528 rad | 0.436 rad |
  | Kinematic minimum turning radius | ~5.8 m | ~6.0 m |
  | Real-time factor | 4.2 | 11.3 |
  | Speed after 12 s at full throttle | 18.7 m/s | 8.1 m/s (max 8.3) |

  - `veh.Gator` exists in both cluster builds (`chrono-build`, `chrono-build-fsi`; checked on the login node), and
    Gator tyre meshes exist: `gator_tireF_coarse.obj` and `gator_tireR_coarse.obj`, which differ front and rear.
  - Its components list `Gator_Driveline2WD`: probably rear-wheel drive, against the HMMWV's all-wheel drive.
    **[the default drive type is not verified]**
  - The 0.10-0.125 /m curvature caps (8-10 m radius) and the 6 m/s ceiling are kinematically feasible for the Gator.
    Its hill-climbing is unmeasured. **[soil + Gator not smoke-tested]**

---

## 4. Cost estimates

### 4.1 Rates used

| World | Billed | Wall | Notes |
|---|---|---|---|
| Rigid | 0.016-0.024 per sim h, ~0.10-0.21 per 1,000 episodes | ~25-35 min per ~15-24k episodes on ~1,600-2,000 cores | Depends on CPU nodes being free |
| Soil | 0.33-0.37 per sim h (use 0.35) | 36 sim h per wall hour on 111 GPUs; ~18 on the ~52 GPUs free today (45 MI210 x 0.32 + 7 MI350X x 0.5) | Cheapest per sim h: mi3501x, ~0.25 |

The f104 volumes behind the rows below:

| Volume | Episodes | Sim h |
|---|---|---|
| Soil, full | 15,235 | 91.5 |
| Rigid, full: the night-2 pool | 24,000 = 1,200 x 20 | 203 |
| Rigid, twin-sized: what the final model saw | ~15,235 | ~125 |

Sibling arenas: rigid designed episodes averaged 22.4 s on `gen_v1` against 20.7 s on f104 (+8 %). Arena
difficulty moves the hours by roughly +-20 %.

### 4.2 Collection estimates

| Case | Rigid billed | Rigid wall | Soil billed | Soil wall (111 / ~52 GPUs) |
|---|---|---|---|---|
| **1 extra arena, full HMMWV volume** | ~3.6 (24,000 episodes; 3.2-4.9) or ~2.3 twin-sized | 35-60 min | **~32** (28-38) | 2.6 h / ~5 h |
| 1 arena, half volume (600 groups) | ~1.8 | ~20-30 min | ~16 | 1.3 h / 2.6 h |
| 1 arena, third volume (400 groups) | ~1.2 | ~15-20 min | ~11 | 0.9 h / 1.7 h |
| **Gator on f104, full volume, same task lists** | 3.6 if Gator episodes last as long; **~4-7** at 1-2x longer episodes; ceiling ~13-19 (all 24,000 run to the 120 s cap = 800 h) | 35-70 min | **32 / 48 / 64** at 1x / 1.5x / 2x the HMMWV mean of 21.6 s per episode | 2.6-5.2 h / 5-10 h |

**Assumption behind the Gator figures.** The soil cost is set by the SPH particles: 4.0 M particles plus active
boxes of 2 x 2 x 1 m per wheel. So it should be about the same per simulated hour for the Gator; the open factor is
episode length. On rigid ground the Gator stepped 2.7x faster on the flat locally, but cluster episodes are dominated
by the heightmap terrain and per-episode overhead. **[Gator cost on the cluster: unmeasured; use HMMWV rates as the
upper bound per simulated hour]**

### 4.3 Add-ons per arena or vehicle, to reproduce the final model's data exactly

| Add-on | Billed | Wall / note |
|---|---|---|
| Pilot | ~1-2 | Night-1 soil pilot: 1.1 |
| Branch drives, 800 anchors | ~0.8 rigid + ~4.9 soil | |
| Map capture | negligible | |
| Dataset builds | 0 | local CPU |
| Training | < 1 | Five seeds x ~13 min on mi3501x = 0.14 billed at 217k rows. At 3 arenas (~650k rows, ~49 GB on the GPU; the MI350X has 288 GB): ~0.5 billed, ~40 min per seed. |
| Soil evaluation | ~1.5-1.9 per 1,000 drives | e.g. 800 pairs x 3 arms = ~4-5 billed |
| Rigid evaluation | ~0.1-0.2 per 1,000 drives | |

### 4.4 Overnight packages (collection + add-ons; the evaluation size is the caller's choice)

| Package | Billed | Soil sim h | Soil wall (111 / ~52 GPUs) |
|---|---|---|---|
| A, rigid only: 2 extra arenas at full volume | ~8-10 | 0 | - (<1 h, CPU) |
| A, both worlds: 2 extra arenas at full volume | ~80-85 | 183 | 5.1 h / ~10 h |
| A, both worlds: 2 extra arenas at 1/3 volume (equal-total design with f104 cut to 1/3) | ~30 | 61 | 1.7 h / 3.4 h |
| B, both worlds: Gator full volume | ~45-80 | 92-183 | 2.6-5.2 h / 5-10 h |
| A (1/3) + B (full) | ~75-110 | 153-244 | 4.2-6.8 h / 8.5-13.5 h |

- **Budget does not bind; wall time does.** Every package fits the 861 remaining, and the previous efforts'
  self-imposed cap of ~100 billed each.
- **Soil hours limit what fits in a night.** A night fits at most ~150-180 simulated soil hours, and only if the
  8-GPU nodes free up. With today's ~52 GPUs it is ~90-110 h, i.e. one full-volume soil set.
- **Scheduling limits:**
  - The 50-task cap is nearly used by one soil launch (47 tasks). Put all arenas and vehicles in **one** tiered
    soil task file with per-row `extra` arguments.
  - Run the rigid waves first (~30 min), or on partitions the soil launch does not use: mi2104x/mi2101x nodes are
    exclusive and shared by both kinds of work.
  - Soil episodes are bit-identical across GPU types, so mixing partitions is safe. Rigid episodes are deterministic
    only within a node type, so run paired rigid arms in one job.

---

## 5. Not verified / open

- The discrepancy between my `sacct` reconstruction and the harry ledger (+14.3 billed) is unexplained. The
  10x-weight method matches two users exactly and the effort records (29, 39.0).
- The 50-task queue cap and the per-partition walltime caps are taken from records, not re-tested.
- The builder of `twin_{crm,rigid}.npz` and the command that sliced `anchor_k60.npz` are not recorded.
- Gator: cluster cost per simulated second, drive type, hill-climbing, soil coupling with per-axle tyre meshes, and
  the episode-length multiplier are all unmeasured.
- Using the sensor_v1 lavapipe captures as soil map roots for the sibling arenas is plausible (OptiX matched lavapipe
  to 4.6e-5 m on f104) but untested.
- Memory and progress-doc corrections found here:
  - production_v2 is not a complete 12-route factorial: it has 5-10 routes per group;
  - the rigid f104 waves hold 269.4 simulated h, not "about 200";
  - night 2's "21.9 billed" is raw node-hours; billed was ~3.1.
