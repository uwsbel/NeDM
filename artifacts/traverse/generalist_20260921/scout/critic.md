# Completeness critic: what the seven maps miss or contradict

Read-only, 2026-09-21. Every claim below was re-checked in code or data this session; paths are relative to
`/home/harry/NeDM-traverse_mppi`. The maps are good on their own subsystems; the problems are at the seams.

## 1. Contradictions between maps (or with the plan)

1. **CRM physics step.** `nrd_tracker_repo` says the CRM collector re-synchronises the PID every 0.5 ms and a CRM
   policy runner must hold inputs over "100 x 0.5 ms substeps"; `original_mixed_domain` says "0.5-1 ms". The
   collection ran at **1 ms / 50 substeps**: `artifacts/traverse/crm_f104_v1/configs/crm_main.json` `step_s 0.001`,
   `scripts/crm_collect.py:201` (`substeps = round(0.05/dt)`), `physics_dt_s 0.001` in every `outcome.json`
   (`collect_v1/runs/*`, `crm_night2_v1/planner/eval_iter_crm/runs/*`). The in-script default is 0.5 ms
   (`crm_collect.py:38`), and the pilot showed the two steps agree on only 96.5 % of outcomes
   (`crm_f104_v1/LOG.md:13`), so a runner built from the wrong map would evaluate on different soil physics.
2. **What the specialists take as context.** `terrain_crop_maps` calls the plan wrong ("vehicle state is already
   an input"). It is the plan that is right: the dataset ctx is 22-d, but `scripts/crm_train.py:18,83` selects
   `GEOM=[17..21]` and saves `nctx=5` (`:124-127`); `scripts/gen_planner.py:7` documents "geometry-only context";
   `planner_closed_loop` and `risk_model_dataset` verified nctx=5 on both deployed ensembles.
3. **Map source per world.** `planner_closed_loop` says depth image vs heightmap are "equal to float16 rounding".
   Two different facts were merged: both specialists were *trained* on the same OptiX depth-map corridors
   (`crm_f104_v1/LOG.md:12`, max 0.002), but the rigid *closed-loop* harness scores on the heightmap via
   `GP.set_map` (`scripts/planner_arms.py:82-90`), which differs from the depth map by 0.035 m mean height and
   0.985 logit correlation (`gen_planner.py:9-11`), while the depth map itself is 0.050 m rmse from Chrono
   (`terrain_crop_maps`). In a paired cross-domain suite the two worlds would feed the same model different
   corridors for the same route.
4. **Deployable state columns.** `risk_model_dataset` counts column 16 (motorshaft torque) as deployable
   ("13 + 3"); `collector_timing` classifies it simulator-only (`src/nedm/training/constants.py:44-48`).
   One decision is needed: 12 + 3 or 13 + 3.
5. **What `moving_v1` holds.** `risk_model_dataset` lists it among raw drives of the held-out planner missions;
   it actually holds `mv_<hash>__v{0,2,4}` moving-start drives of held-out *re-anchored training routes*
   (`scripts/n2_moving_tasks.py:14-35`), started as braked skids (`collector_timing`). Exclude it from any
   dynamics dataset for that reason, not the other.
6. **Action timing nuance.** `risk_model_dataset` and `nrd_tracker_repo` describe `action[k]` as "the PID output
   at substep 0 of interval k / the action applied in that interval"; `collector_timing` is the precise one:
   `ChDriver::Synchronize` is empty (`/home/harry/chrono/src/chrono_vehicle/ChDriver.h:66`), the PID computes in
   `Advance` (`driver/ChPathFollowerDriver.cpp:84-114`), and the row is captured before `Advance`
   (`crm_collect.py:232-246`), so `action[k]` is one substep stale and computed under interval k-1's speed target.
   Same data, but B step 2 must state this convention.

## 2. Plan steps no map covers

- **A4 low-progress prefixes.** `scripts/n2_reanchor_dataset.py:41-42` admits anchors only *before* the event,
  with lateral deviation < 1 m and not parked; no existing row has a history window containing a stall, and the
  prefix-then-branch collector (`collector_timing` section 4) is not written. The cases where the plan expects
  history to matter have no data source.
- **Milestone A calibration.** No calibration metric exists (`grep -i "calib|brier"` over `n2_arch_train.py`,
  `n2_planner_analyze.py`, `crm_train.py` is empty).
- **Milestone A "established history".** The only in-motion planner is `scripts/nav_runner.py`, rigid-only
  (live OptiX/Vulkan depth render, `:1-13,201,477`) and it replaces the follower's throttle with
  `nav_online.SpeedPI` (`nav_online.py:125-138`); no CRM path to an in-motion decision exists.
- **Offline reference for A1.** The deployed rigid N2 was fitted on the rigid twins of the CRM val/test groups
  (`crm_f104_v1/REPORT.md:140`); the uncontaminated per-domain references are
  `crm_night2_v1/stageA/{rigid,crm}_gru_geom_lr0.002_holdout.json`, which no map names.
- **B3 perturbation data.** Nobody read the original excitation design (`src/nedm/hmmwv_data.py:215-250,571`)
  or `scripts/traverse_wp4_collect_tracker_episodes.py`; neither arena collector has a perturbed-PID mode.
- **Milestone B suite.** No map defines the held-out tracking routes; rigid PID drives of the CRM eval routes
  do not exist, so paired PID references must be collected in both worlds under the 1 ms config.
- **B5 failure harvesting in CRM** needs the external-control mode plus a stop rule that does not require
  throttle > 0.3 (`scripts/gen_collect.py:129-131`).

## 3. Verified this session

- Twin `id`, `group`, `split` arrays are identical across worlds; held-out = 1,395 rows / 111 groups, fail 0.73 CRM
  vs 0.186 rigid.
- Eval cases carry `split='train'` for 189/200 (CRM) and 180/200 (rigid test).
- At rest the 17-D state barely separates the domains: per-column AUC <= 0.60 (best `fz_fl` 0.595,
  `pitchrate` 0.601). Startup context is genuinely uninformative; tyre forces are the only mild leak.
- `n2_arch_train.py:305` saves `model_kind='n2_arch_train'` GRUNet weights; `gen_planner.py:165-166` rebuilds the
  legacy `Net`. Nothing new loads in the planner today.

## Gaps / unknowns

- torch is not importable in system `python3`; checkpoint key layouts were taken from the two maps that loaded them.
- Whether the rigid bit-for-bit baseline (arm A == gen_v1 picks) survives switching the rigid world to the depth
  map is untested; the 0.985 logit correlation suggests picks would change on some groups.
- Substep action path inside a frame is still unmeasured (audit sketched, not run).

## What must change for the plan

1. Declare one map source for both worlds in the paired suite (the v2 grid or the depth map via `--map-root` on
   rigid too) and re-run both specialist baselines on it; report the old heightmap numbers as history only.
2. Name the offline reference models (stage A per-domain holdout ensembles), add the id blacklist and split
   override, and add a per-domain Brier/reliability metric before step 1 runs.
3. Fix the deployable feature list (cols 0-6, 11-15 [+16?], 3 actions) in writing; treat cols 7-10 as teacher-only.
4. Write the CRM external-hold and prefix-branch modes with `--crm-config crm_main.json` mandatory; run the
   substep audit on ~20 episodes per world first.
5. Either build an in-motion CRM decision path (prefix replay + plan at the branch frame) or drop the
   "established history" read-out from Milestone A and say so.
6. Widen the predeclared gap or the group count; at 200 groups the paired CI half-width (2-2.5 pts) equals the
   2-pt margin.
