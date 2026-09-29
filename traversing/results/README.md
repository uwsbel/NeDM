# Traversing study: compact result tables

These tables let a reviewer recount the milestone numbers of the traversing study (learned route-risk planning
for off-road vehicles in Project Chrono, on rigid ground and on CRM deformable soil) without running a simulation
or downloading the raw drives. Each table has one row per evaluation task (a start/goal pair, a navigation
mission or a reference route) and records how every evaluated planner or controller ("arm") did on it.

All numbers come from the experiment branch `offroad_vehicles_v1` at commit
[`901d6c9`](https://github.com/uwsbel/NeDM/tree/901d6c9423a16c0fafc3d60056065415d5a725f2). Every cell was derived
from the study's per-drive records and cross-checked against the study's own analysis files, with no mismatches.
Most raw drive records are not yet published; they are listed below as **local-only** with their SHA256, so they
can be matched once the archive is released.

## Recount the headline numbers

```bash
python traversing/scripts/recount_milestones.py               # per-arm counts, metric checks, verdict
python traversing/scripts/recount_milestones.py --check-only  # only mismatches and the verdict
```

Python 3.8 or newer, standard library only. The script prints n / goal / unsafe / safe_goal for every arm of every
table, checks the 251 expectations in [`expected_counts.json`](expected_counts.json), and exits non-zero on
any mismatch. Counts must match exactly; means and medians must lie within the stated tolerance of the expected value
and, where given, of `study_value`, which is the study's own full-precision number. Each expectation has a
`claim_id` naming the milestone statement it supports, the column it counts, and optional row filters (exact text
match).

## Outcome codes

Every arm cell (or the `outcome_code` column of a long table) holds one code:

| code | meaning |
|---|---|
| `S` | reached the goal with no unsafe event |
| `s` | reached the goal, but an unsafe event was recorded |
| `U` | did not reach the goal, unsafe |
| `F` | did not reach the goal, not unsafe (stall, timeout, ...) |
| `-` | this arm was not run on this task |

Counts: **n** = cells other than `-`; **goal** = `S` + `s`; **unsafe** = `s` + `U`; **safe_goal** = `S`.

Each study's own definitions are kept. Except in the tracker study, **every drive that does not reach the goal
counts as unsafe**, so `F` does not occur there. Two event definitions recur:

- **Rolling back.** After the first 1 s of driving, the vehicle rolls backwards faster than 0.10 m/s with throttle
  above 0.3 for at least 0.05 s in total, or its forward speed drops below -0.30 m/s. The state is recorded
  every 0.05 s.
- **Belly flag** (Gator and Polaris only). The lowest hull point stays more than 0.05 m below the undisturbed soil
  surface for more than 1 s in a row.

Body tilt beyond 30 degrees is **not** part of any label here.

Source links: files tracked at the commit link to GitHub. Local-only files are named by their repository path and
SHA256. All paths are relative to the repository root at that commit.

---

## `m1_navigation_missions.csv`: continuous navigation with live depth sensing

**One row** = one mission driven by one decision-timing arm (30 missions x 4 arms = 120 rows; long table). The
HMMWV drives a chain of 5-8 waypoints on rigid hill-and-crater terrain. At each decision, Chrono renders a
fresh depth image from a whole-arena overhead camera. The planner turns it into a height map, builds candidate
routes, scores them with a frozen learned risk ensemble and drives the best one with the conventional path follower.

| column | meaning |
|---|---|
| `arm` | decision timing (below) |
| `mission_id` | the study's mission id, e.g. `g213_nav_002`; raw folder `local_luffy/runs/<mission_id>__<study arm id>` |
| `arena` | terrain arena; `f104` is the main training arena |
| `arena_seen_in_training` | 1 if the risk ensemble was trained on drives from this arena (f104, g228, g203, g217), else 0 |
| `arena_group` | `risk_model_training`; `held_out_development` (g216, g231: not trained on, but used in earlier tests); `new_unseen` (g213, g204, g234, g223: generated for this milestone, never driven or inspected before; the report's "unseen" split) |
| `waypoints_total`, `waypoints_reached` | waypoints in the mission and waypoints reached |
| `status` | raw end status of the mission |
| `outcome_code` | code as defined below |
| `backward_slide` | 1 if the runner's rolling-back detector fired at any frame after the first 1 s |
| `elapsed_s` | simulated mission time to completion or termination (s), excluding the 0.8 s settle |
| `mission_outcome_sha256` | SHA256 of that run's raw `mission_outcome.json` |

| arm | in words | study arm id |
|---|---|---|
| `plan_once_per_waypoint` | one planning decision at the start of each leg | `W` |
| `replan_every_2s` | a fresh decision every 2 s of simulated time, planning time not charged | `R2` |
| `replan_every_1s` | a fresh decision every 1 s, planning time not charged | `R1` |
| `replan_every_1s_delay_charged` | every 1 s, but each new route takes effect only after the measured back-projection and planning time (the software render time is not charged) | `R1L` |

**Status to code**, following the study's mission analysis (`scripts/nav_analyze.py`: a mission is unsafe if it
slid backwards or did not complete): `mission_complete` without a slide -> `S`; `mission_complete` with a slide
-> `s`. Every other status -> `U`: `prolonged_blockage_terminated` (pinned under throttle), `terrain_bounds_exit`
(left the +-40 m terrain), `timeout` (a leg over 120 s), `no_route` (no valid route even after the rescue
steps), `rollover` (roll or pitch above 60 deg). An event-only reading (unsafe = slide or rollover) can be derived
from `status` and `backward_slide`. It would turn 5 of the 21 `U` rows into `F`.

These are the corrected re-run (routes that double back were rejected). The risk model is the frozen direct-depth
model, not the later history model.

| source | status | SHA256 |
|---|---|---|
| `artifacts/traverse/fdm_f104_50h_20260909/nav_v1/local_luffy/runs/*/mission_outcome.json` (120 files) | local-only | per file in the table |
| `artifacts/traverse/fdm_f104_50h_20260909/nav_v1/local_luffy/summary.json` (the study's read-out) | local-only | `f8d03b267b7bcc096d7c64f5d683cf0fc8dece143fdda73fa1f1753a1c6aa85b` |
| [`.../nav_v1/local_luffy/tasks_main.json`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/fdm_f104_50h_20260909/nav_v1/local_luffy/tasks_main.json) | tracked | `ea670e3ec84f4ef5664113807b50f34acaedfa92a79244068fecb4798ccd9f2b` |
| [`.../nav_v1/REPORT.md`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/fdm_f104_50h_20260909/nav_v1/REPORT.md) | tracked | `4e24c31417bfc5309f8ff90a2e84966419f5bc7ab35df8401fbab4c64796b948` |
| [`scripts/nav_analyze.py`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/nav_analyze.py), [`scripts/nav_runner.py`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/nav_runner.py) | tracked | |

---

## `m2_shared_risk_soil.csv` and `m2_shared_risk_rigid.csv`: one risk model for rigid ground and soil

**One row** = one start/goal pair of the 800-pair suite on the f104 training arena (wide table). The soil table
covers the CRM deformable-soil world and the rigid table the rigid-ground world, on the same 800 pairs. Every arm
plans one route per pair by cross-entropy sampling search (4 rounds of 64 candidate routes), and the conventional
path follower drives it.

| column | meaning |
|---|---|
| `group_id` | the study's pair id. `f104_pair_group_0000`-`0599` are 600 pairs created on 2026-09-21. `f104_crm_eval_group_0000`-`0199` are 200 pairs reused from the 2026-09-16 soil evaluation |
| `arena`, `world` | `f104`; `crm_soil` or `rigid` |
| `suite_stratum` | `fresh` (the 600) or `reused` (the 200) |
| `terrain_stratum` | the suite's terrain type for the pair (hill or crater side slope, entry/cross/exit, long traverse, roughness transfer) |

Arms. "Shared history model" = one model trained on both worlds that reads the last 2 s of the vehicle's own motion
and controls, with no world label. The approach is a straight drive at 3 m/s before the planner decides.

| arm column | in words | study arm id (file) |
|---|---|---|
| `specialist_soil_standing` | earlier deployed soil-only planner, standing start | `Scrm` (A0A3) |
| `specialist_rigid_standing` | earlier deployed rigid-only planner, standing start | `Srigid` (A0A3) |
| `oracle_tag_standing` | shared model given the true world label, standing start | `T` (A0A3) |
| `shared_hist_standing` | shared history model, standing start (empty history) | `H` (A0A3) |
| `specialist_soil_3s`, `specialist_rigid_3s` | single-world models trained on the same rows, after a 3 s approach | `Spcrm`, `Sprigid` (A5) |
| `oracle_tag_3s` | shared model with the true world label, 3 s approach | `T` (A5) |
| `pooled_3s` | shared model with neither history nor label, 3 s approach | `P` (A5) |
| `shared_masked_3s` | shared history model with its history masked at the decision, 3 s approach | `Hmask` (A5) |
| `shared_hist_3s` | shared history model, 3 s approach (the 3 s baseline) | `H` (A5) |
| `specialist_soil_1s`, `oracle_tag_1s`, `pooled_1s`, `shared_hist_1s` | the same models, deciding after a 1 s approach (soil only) | `L1_Spcrm`, `L1_T`, `L1_P`, `L1_H` (s2) |
| `shared_hist_early_rows_1s` | the same history design retrained with extra decision rows at 0.5/1/1.5/3 s and rows branched from moving states, 1 s approach | `L1_Hn` (s2) |
| `transformer_1s` | a transformer over route stations and history steps, trained on the same rows, 1 s approach | `L1_X` (s2) |
| `oracle_tag_0p5s`, `pooled_0p5s`, `shared_hist_0p5s`, `shared_hist_early_rows_0p5s`, `transformer_0p5s` | the same arms after a 0.5 s approach (soil) | `L0p5_T`, `L0p5_P`, `L0p5_H`, `L0p5_Hn`, `L0p5_X` (s2) |
| `shared_hist_early_rows_0p5s` (rigid table) | retrained history model, 0.5 s approach, sampling search only | `Hn05` (s4 rigid) |
| `shared_hist_early_rows_0p5s_grad` | **final label-free configuration**: retrained history model, 0.5 s approach, sampling search plus gradient route refinement | `HnG05` (s4, s4 rigid) |
| `transformer_0p5s_grad` | transformer, 0.5 s approach, with gradient refinement (soil) | `XG05` (s4) |

**Status to code**, following the study's label (`scripts/f104_n2_analyze.py` `labels`, through
`scripts/ga_analyze.py` `safe_labels`): `goal_reached` and no rolling back -> `S`; `goal_reached` with rolling back
-> `s`. Any other status -> `U`: on soil `soil_breakthrough_terminated` (stalled while a spinning wheel dug through
the soil layer), `prolonged_blockage_terminated`, `timeout`; on rigid ground also `rollover` and
`terrain_bounds_exit`. Tilt beyond 30 deg is not counted: 10 of the 800 final-configuration soil drives and 51 of the 800 rigid drives
exceeded it.
The rigid 1 s and 0.5 s arms of the other models are left out, because the study analysed them on only 556
pairs.

Checks: every cell equals the study's per-pair record in the result files below (18,400 soil and 9,600 rigid cells).
It was also recomputed from the raw `outcome.json` and `trajectory.npz` of the run folder that record names.

| source (prefix `artifacts/traverse/`) | status | SHA256 |
|---|---|---|
| [`generalist_20260921/A_adapt/suite/suite.json`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/A_adapt/suite/suite.json) (the 800 pairs) | tracked | `64921952321fa4e20597bea8b0b4c1a61dad0ac2a763c6eb9351edfa30844c1a` |
| [`generalist_20260921/A_adapt/a3/results_crm_A0A3.json`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/A_adapt/a3/results_crm_A0A3.json) | tracked | `cd6434bf353f375e8556f082e6a90bf29cc28c940a0e44fe945f55feaf387882` |
| [`generalist_20260921/A_adapt/a3/results_rigid_A0A3.json`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/A_adapt/a3/results_rigid_A0A3.json) | tracked | `400fc640e0a86e0ebf332747fee24fe9abe72caff9215706db6966b75edb7c13` |
| [`generalist_20260921/A_adapt/a5/results_crm_A5.json`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/A_adapt/a5/results_crm_A5.json) | tracked | `d8a1585fab38c75cb3a47dd1c7e33d9bce5eb838bd9827c995601474e66c2968` |
| [`generalist_20260921/A_adapt/a5/results_rigid_A5.json`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/A_adapt/a5/results_rigid_A5.json) | tracked | `70ed12392f39cf60ad181586a4203669bafb12d43bafd3691197fe6357b248de` |
| `crm_improve_20260922/s2/results_s2_crm_vs3s.json` | local-only | `da58edcada7d71199d5fa2ab4d8f9fcdf1ce6cc22768c499ebb10c70b7ecebc4` |
| `crm_improve_20260922/s4/results_s4.json` | local-only | `d4b2007f3885bdc45b542addda6f8ce6f6323320f0dde58fe0d9c5477658695a` |
| `crm_improve_20260922/s4/results_s4_rigid.json` | local-only | `438ce5c84a5b4e9b910c2ca5e2a385da93474f3bf3e88fd5ae912b1a6ef75575` |
| raw run folders (`outcome.json`, `trajectory.npz`); a per-file SHA256 list ships with the data release | local-only | |

---

## `m3_tracker_routes.csv`: learned low-level tracker vs the conventional follower

**One row** = one reference route driven by one controller in one world (2 worlds x 423 routes x 3 controllers =
2,538 rows; long table). The HMMWV follows a fixed route and speed profile without replanning.

| column | meaning |
|---|---|
| `world` | `rigid` (rigid f104 ground) or `crm` (CRM soil on the same arena) |
| `route_id` | the study's route id `f104_v2_group_NNNN_route_NN` (prefix = start/goal group) |
| `stratum` | `feasible` (141 routes) or `infeasible` (282; the report's "hard" routes), as frozen in the tracking suite |
| `arm` | controller (below) |
| `status` | raw end status |
| `completed` | 1 if and only if `status` is `goal_reached` |
| `unsafe` | 1 if and only if `status` contains rollover, breakthrough, blockage, off_route or bounds_exit (the analysis script's rule) |
| `outcome_code` | code as defined below |
| `near_stop_40s_fired` | 1 if an extra stop rule ended the run: speed below 0.3 m/s (not parked) for 40 s in a row, whatever the throttle; it then ends as `prolonged_blockage_terminated`. The rule was active only for the two arms with 50 ms held commands; `na` for the native follower |
| `xtrack_station_winsor_mean_m` | per route, 5 % Winsorised mean distance from each reference station to the driven path (m), capped at 6 m |
| `speed_abs_err_mean_mps` | mean absolute error between body forward speed and desired speed over 50 ms frames, from 1 s onward (m/s) |
| `mean_abs_action_change` | mean over consecutive 50 ms frames and the three command channels (steer, throttle, brake) of the absolute change in the recorded command |
| `positive_work_kj` | whole-run positive engine output work (kJ; engine torque times motorshaft speed, not fuel energy) |

Floats are rounded to 4, 4, 6 and 2 decimals. Means over the CSV therefore differ from the study's
full-precision means by up to a few 1e-6; `expected_counts.json` records both values.

| arm | in words | study arm id |
|---|---|---|
| `pid_native` | stock Chrono path-follower PID, updating its command inside each 50 ms interval | `native_pid` |
| `pid_held_50ms` | the same PID, command read once per 50 ms and held, steering change clipped to 0.1 | `held_pid` |
| `nrd_policy_v2` | the round-2 learned tracker (PPO policy trained inside the learned vehicle-dynamics model), command held 50 ms; NumPy actor SHA256 `ea9388204a952d0a096a59aecfb7d27a54e4fe1f1cb406f1c6772d212baae3d6` | `policy_v2` |

**Status to code**, following the study's final analysis (`scripts/gb_track_analyze.py`): `goal_reached` -> `S`;
`prolonged_blockage_terminated`, `soil_breakthrough_terminated`, `rollover`, `terrain_bounds_exit` -> `U`;
`timeout` -> `F`. The study's plan did not list leaving the terrain as unsafe; its final analysis does (3 rigid
hard-route runs), and this table follows the analysis. `s` cannot occur, because unsafe depends only on the end status. `-` is unused, because every
arm drove every route. Observed: 1,536 `S`, 965 `U` (722 breakthrough, 235 blockage, 5 rollover, 3 bounds exit),
37 `F`. The round-1 policy arm is left out to keep the file small.

Checks: every row equals the tracked per-route records. The near-stop flags equal the raw `outcome.json` files, and
all values were also recomputed from the raw run files.

| source (prefix `artifacts/traverse/generalist_20260921/B_tracker/`) | status | SHA256 |
|---|---|---|
| [`b0/results_rigid_b0v2.json`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/B_tracker/b0/results_rigid_b0v2.json) (`per_route`) | tracked | `20c95e79d044ffb46a9fd02ca81f9e426ffd6df0d18e0464336e185d3a88adf5` |
| [`b0/results_crm_b0v2.json`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/B_tracker/b0/results_crm_b0v2.json) (`per_route`) | tracked | `ebcf6fffdb0a45f6f964bf48306d645eb9e76f9b65db08541719720e7d994d18` |
| [`suite/tracking_suite.json`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/B_tracker/suite/tracking_suite.json) (routes and strata) | tracked | `a53b0eb69381e7b437c6afbe3e339c1671c28c727c10f2c5912472368b9ca2e5` |
| [`scripts/gb_track_analyze.py`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/scripts/gb_track_analyze.py) | tracked | `c0203ab6da1a9eceb2342765eeb3a179d12ce22ce414bd9813c01af4d5d544ab` |
| raw run folders `b0/{rigid,crm}_by_arm/<arm>/<route>/` (7,614 files; a per-file SHA256 list ships with the data release) | local-only | |

---

## `m4_unseen_arenas_hmmwv_soil.csv` and `m4_unseen_arenas_hmmwv_rigid.csv`: more training arenas, tested on unseen arenas

**One row** = one evaluation start/goal pair (wide table). The HMMWV plans once from a standing start by sampling
search (4 x 64 routes) with a 5-network risk ensemble, and the conventional follower drives the route. The eight
test arenas are new layouts from the same hill-and-crater generator. No model was trained on them.

| column | meaning |
|---|---|
| `pair_id` | the study's group id, e.g. `g260_test_group_0000` |
| `arena` | arena id |
| `arena_role` | `near_test` (g260, g271, g251, g247: the 4 unseen arenas closest to f104); `spread_test` (g258, g268, g263, g241: 4 unseen arenas farther away); `training` (f104, g203, g228); `dev` (g217, rigid only) |
| `eval_set` | `unseen_test`: soil = the declared 125-pair soil half of each unseen arena (1,000 rows), rigid = all 250 pairs per unseen arena (2,000 rows). `f104_in_distribution`: 200 held-out f104 pairs. `training_arena_heldout`: 150 held-out pairs each on g203 and g228. `dev_unseen` (rigid only): 150 pairs on g217, which none of this study's planners trained on (the separate
navigation model of `m1` did) |
| `terrain_cluster` | arena and nearest terrain feature, the study's bootstrap cluster |

| arm column (soil; rigid adds `_fixed2mps` / `_speedfree`) | in words | study arm id |
|---|---|---|
| `f104_only_ens1`, `f104_only_ens2` | two independently trained ensembles on the f104 arena only (1,089 groups). The study's "f104 only" rate is their per-pair mean | `M1a`, `M1b` |
| `two_arenas_same_total` | f104 + g203, 545 groups each, same total data | `M2` |
| `three_arenas_same_total_ens1`, `_ens2` | f104 + g203 + g228, 363 groups each, same total data, two ensembles. The study's rate is their mean | `M3a`, `M3b` |
| `three_arenas_all_data` | all data: soil f104 1,089 + g203 559 + g228 520 groups; rigid f104 1,089 + g203 1,083 + g228 1,063 groups | `A3` |
| `straight_route_6mps` | straight line to the goal at 6 m/s, no model | `straight6` |
| `straight_route_2mps` (rigid) | straight line at 2 m/s, no model | `straight2` |

Soil arms choose speeds freely up to 6 m/s, and soil models were trained on the first 7 routes of each group. On
rigid ground each trained planner ran twice: `_fixed2mps` fixes the speed at 2 m/s, so the planner chooses only
the path, and `_speedfree` lets it choose speeds up to 6 m/s (study suffixes `_fx2`, `_free`). In the soil table the
second ensembles and the two-arena model were driven only on the unseen test pairs (`-` elsewhere).

**Status to code**, following the study's labels (`scripts/ag_eval_index.py` through `ga_analyze.safe_labels`):
`goal_reached` without rolling back -> `S`; `goal_reached` with rolling back -> `s`. Any other status
(`soil_breakthrough_terminated`, `prolonged_blockage_terminated`, `timeout`, `rollover`, `terrain_bounds_exit`)
-> `U`. `F` does not occur. Totals: soil 7,580 `S`, 2 `s`, 1,418 `U`, 1,500 `-`; rigid 34,492 `S`, 1,370 `s`,
1,238 `U`. The Gator rows of the same indexes are not included here.

Checks: every cell (9,000 soil, 37,100 rigid) equals the local evaluation index. All 100 recount expectations for
these two tables equal the rates in the tracked analysis files.

| source (prefix `artifacts/traverse/arena_gator_20260925/e6/`) | status | SHA256 |
|---|---|---|
| `index/soil_eval_v1.json` (per-drive index) | local-only | `d9c86320694f352d24826c4d828fabe1a2f0cd7b7b359891eba76bdde4cd4029` |
| `index/rigid_eval_v1.json` (per-drive index) | local-only | `7c09e4d3abfaf3da205adca2ee672813c93eb96481df24e3c1d4de405567b900` |
| [`analysis/results_soil_v1.json`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/e6/analysis/results_soil_v1.json) (records the soil index hash) | tracked | `96facb251356d9cd9eabde5d0d4a73b11511b1b797963321e203571a4451755a` |
| [`analysis/results_rigid_v1.json`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/e6/analysis/results_rigid_v1.json) (records the rigid index hash) | tracked | `712e3fb674e83ccac9baf056f25d6a872547f3047bfd96e204da563934ba326d` |
| `tasks/soil_eval_p1.json.mapping.json` ... `p5` | local-only | `959fc9f62507a65926c155c3d44e29a1605329b73a4a3ba88331d8c6b2311ad2`, `95652f1807b688ef4b953a5d98a698d91715746e0a5989a81ffa1007e1b27e92`, `9955b8c303705e209fb53d2a1e3a173b136ada71094089fa145387a5706c2306`, `b30428f0c70213936250fca197224f5bc1a30b428866c8c322480dc201b39727`, `23fcd8b334f2ad42b7e7c76d36202fa34a45528643fdb8d9706bb1e453ed4510` |
| `tasks/rigid_eval_{unseen,f104,heldout,dev}.json.mapping.json` | local-only | `4ab8211599a1314c2f1c6f307ac183ea0923dd42bd0766ef10c72f8b14b09c65`, `16476b6fcc3cbed81cd34454fd79539c924144693f8f8688e4c98bf7f224d26b`, `efc929c314abf157990310ff285f05660ac86c4aa7754c6cd6d7681202ec5c1f`, `8d74779b3ac45a7814b2dbfc3cc9307c21974e6f7af9a29b5b066aa44e317b21` |
| raw drive folders `runs_soil/<run_id>/`, `runs_rigid/<run_id>/` (`outcome.json`, `trajectory.npz`) | local-only | |

---

## `m4_vehicles_f104_soil.csv`: the pipeline on other vehicles (f104 soil)

**One row** = one start/goal pair of the 800-pair f104 soil suite (wide table; CRM soil). For every arm the vehicle
starts at rest, plans once from the whole-arena height map and drives the route once with the frozen follower. "Own
model" = a 5-network risk ensemble trained on the training groups of that vehicle's own f104 soil collection
(15,235 route ids in all; the Polaris had 15,229 valid episodes; validation and test groups are held out). "Sampling" = sampling
search, 4 rounds x 64 routes. "Grad" = plus gradient route refinement.

| column | meaning |
|---|---|
| `task_id` | the study's group id. `f104_crm_eval_group_0000`-`0199` are the 200 pairs on which the sampling-search settings were tuned in earlier studies. `f104_pair_group_0000`-`0599` are the 600 fresh pairs |
| `arena`, `world` | `f104`, `crm_soil` |
| `suite_part` | `tuned200` or `fresh600` |
| `terrain_cluster` | arena and nearest terrain feature (9 clusters) |
| `task_type` | terrain type of the pair |

| arm column | in words | study arm id |
|---|---|---|
| `gator_own_model_sampling` | Gator, own model, sampling (the Gator study's declared primary arm) | `Gfull_free_gator` |
| `gator_own_model_sampling_grad` | Gator, own model, with gradient refinement | `Gfull_grad_gator` |
| `gator_own_model_tiers0to6_sampling` | Gator model trained on only the first 7 routes of each group (8,399 drives) | `G_free_gator` |
| `gator_hmmwv_model_sampling` | Gator driving the picks of the planner trained on the HMMWV's f104 soil collection | `Hfull_free_gator` |
| `gator_straight_6mps` | Gator on the straight line at 6 m/s | `straight6_gator` |
| `hmmwv_own_model_sampling` | HMMWV driving the same HMMWV-trained picks (identical routes to `gator_hmmwv_model_sampling`) | `Hfull_free` |
| `hmmwv_own_model_sampling_grad` | HMMWV, with gradient refinement | `Hfull_grad_hmmwv` |
| `hmmwv_straight_6mps` | HMMWV on the straight line at 6 m/s | `straight6` |
| `polaris_own_model_sampling_grad` | Polaris, own model, sampling plus refinement (the declared Polaris planner) | `polaris_grad` |
| `polaris_own_model_sampling` | Polaris, own model, sampling only. On the 275 pairs where refinement kept the sampling pick, it shares the declared arm's drive | `polaris_cem` |
| `polaris_corrected_driveline_grad_routes` | Polaris with the power-corrected driveline driving the declared planner's routes (model trained on stock-driveline data) | `polaris_grad_pc` |
| `polaris_straight_6mps` | Polaris on the straight line at 6 m/s | `straight6_polaris` |

**Status to code**, following the study's final labels: `goal_reached` with neither rolling back nor a belly flag
-> `S`; `goal_reached` with either -> `s`. `prolonged_blockage_terminated`, `soil_breakthrough_terminated`,
`timeout`, `rollover`, `terrain_bounds_exit` -> `U`, because every drive that misses the goal is unsafe in the
study's label. `F` does not occur. HMMWV drives have no belly record. The Gator-study index has no belly field.
A recheck of the raw Gator belly records found every flag on a drive that missed the goal, so no code changes.

Checks: all 9,600 cells equal the two local indexes. All 15 recount expectations equal the rates in the tracked
read-outs `offroad_vehicles_20260927/e6/analysis/results_ov_v1.json` and
`arena_gator_20260925/e6/analysis/results_soil_v1_Bfull.json`.

| source (prefix `artifacts/traverse/`) | status | SHA256 |
|---|---|---|
| `arena_gator_20260925/e6/index/soil_eval_bfull.json` (Gator sampling arms, HMMWV sampling and straight arms) | local-only | `ce1a5ee1618b9e132ff1fb457a36fc78c30e50d3a69d988cf588b193f158d000` |
| `offroad_vehicles_20260927/e6/index/soil_eval_ov_final.json` (refined arms, Polaris arms) | local-only | `bb5e35ecb29fd20e139238a7b373778aa22ae55c2c45e8a4f5bedb931a6fc3f5` |
| [`offroad_vehicles_20260927/e6/analysis/results_ov_v1.json`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/e6/analysis/results_ov_v1.json) (records both index hashes) | tracked | `dccf2fc708b14d1422926d4f4205376b69024e90b136d2752b5a6c3016534c90` |
| [`arena_gator_20260925/e6/analysis/results_soil_v1_Bfull.json`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/e6/analysis/results_soil_v1_Bfull.json) | tracked | `9f67f31f7b55bdab3efb50c97cf4e25f8d44a593439d31fea1f483c962285566` |
| `arena_gator_20260925/e6/tasks/soil_eval_bf1.json.mapping.json`, `soil_eval_bf2.json.mapping.json` | local-only | `6907d61598af98e3cef62758fd120829338b36e0bb295f4927d56012566a09c4`, `cc5321f2df9f82f9f34fcfdb39d8173811a7ab55cf7ceb10967415d8e3af4a49` |
| `offroad_vehicles_20260927/e6/tasks/soil_eval_polaris_s2a.json.mapping.json`, `soil_eval_polaris_s2b.json.mapping.json`, `soil_eval_gradref_v1_noreuse.json.mapping.json` | local-only | `8f6d4167c82a3c741bae327994d31b8a37f1788f9ad74526f9351b6b5711c9b1`, `a7dee6d0652562d6673eb8599a3fbc11077f411c88be46805545b7ff15c2d4df`, `62c1bed5d7e7f13be0e4bcd61617519df0511794ec2868ab43f772761fede81e` |
| raw drive folders named in the indexes (`outcome.json`, `trajectory.npz`, `vehicle_extra.npz`) | local-only | |

---

## `m4_polaris_unseen_soil.csv`: Polaris on unseen arenas (soil)

**One row** = one declared soil start/goal pair on the 8 arenas the Polaris planner never trained on (125 per arena,
1,000 rows; wide table). All arms use the unchanged f104-trained 5-network Polaris ensemble, each arena's own
overhead height map, a standing start and one drive.

| column | meaning |
|---|---|
| `task_id` | the study's group id, e.g. `g241_test_group_0000` |
| `arena` | g241, g247, g251, g258, g260, g263, g268 or g271 |
| `arena_kind` | `near` or `spread` (terrain distance to f104 of 0.65-0.91 or 1.11-1.79) |
| `world` | `crm_soil` |
| `terrain_cluster` | arena and nearest terrain feature |
| `task_type` | terrain type of the pair |

| arm column | in words | study arm id |
|---|---|---|
| `polaris_own_model_sampling_grad` | the declared planner: sampling plus gradient refinement | `polaris_u_grad` |
| `polaris_own_model_sampling` | sampling only. On the 380 pairs where refinement kept the sampling pick, it shares the declared arm's drive | `polaris_u_cem` |
| `polaris_straight_6mps` | straight route at 6 m/s | `straight6_polaris_u` |

**Status to code** as in `m4_vehicles_f104_soil.csv`. No drive that reached the goal had a belly flag or rolled back, so there are no `s`
cells. The only belly flag was on a straight-route drive that rolled over, which is `U` anyway. The 2 straight-route drives flagged "dug through without a stall first" count as failures (`U`), as in the
study. `F` and `-` do not occur. The HMMWV arms on the same pairs are in `m4_unseen_arenas_hmmwv_soil.csv`.

Checks: all 3,000 cells equal the local index. All 21 recount expectations equal the tracked read-out
`offroad_vehicles_20260927/e6/analysis/results_unseen_v1.json`, including per arena and per near/spread group.

| source (prefix `artifacts/traverse/offroad_vehicles_20260927/`) | status | SHA256 |
|---|---|---|
| `e6/index/unseen_polaris_v1.json` | local-only | `49c5b4741423f13bf8555a0e9d5358e24baea51f6961bcebd8e184d100afb4f5` |
| [`e6/analysis/results_unseen_v1.json`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/e6/analysis/results_unseen_v1.json) (records the index hash) | tracked | `7954083ab976a9c451f0297bafa979f56b99a3bf33d1c604d271f00a8f7e5169` |
| `e6/tasks/soil_eval_polaris_unseen_v1.json.mapping.json` | local-only | `e13845895a15834f451a28312ac3523ac6d3d87a4269880b6e8b465e1cc37e4a` |
| raw drive folders `e6/sync/runs_unseen/` | local-only | |

---

## `m4_vehicle_smoke.csv`: vehicle smoke test (Gator, HMMWV, Polaris variants, M113)

**One row** = one route of smoke sample A: the 144 f104 soil routes of the Gator pilot, 24 start/goal groups x 6
routes (wide table). Each vehicle drives the same fixed route and speed profile. There is no planning in this
test.

| column | meaning |
|---|---|
| `route_id` | the collection's route id, e.g. `f104_v2_group_0177_route_11` (designed) or `..._op_01` (planner proposal) |
| `group_id` | start/goal group |
| `route_kind` | `designed` (86 geometric routes at a set speed profile) or `planner_proposal` (58 routes drawn from the planner's own candidates) |
| `speed_profile` | `constant_2`, `constant_4`, `constant_6`, `smooth_2_6_2` (m/s) or `planner_proposal` |
| `task_type` | terrain type of the group |

| arm column | in words | study arm id |
|---|---|---|
| `gator_redrive_1ms` | Gator driven again in the Polaris launch (1 ms step); the paired reference | `gatorctl` |
| `gator_redrive_0p5ms` | Gator at the M113's 0.5 ms step, in the M113 launch | `gatorh` |
| `hmmwv_stored` | the HMMWV drives of these routes from the 09-16/17 soil collection | stored runs |
| `polaris_stock` | Polaris, stock driveline (primary) | `polaris` |
| `polaris_power_corrected` | Polaris, power-corrected driveline | `polaris_pc` |
| `polaris_open_diff` | Polaris with Chrono's open-differential 4WD driveline | `polaris_4wd` |
| `polaris_soil_wheels_0p33m` | Polaris with 0.33 m soil wheels instead of 0.25 m | `polaris_w08` |
| `m113_stock_gearing` | M113 tracked vehicle, stock gearing | `m113` |
| `m113_regeared_4x` | M113 with every gearbox ratio x 0.25 | `m113_g4` |

**Status to code**: `goal_reached` without a belly flag -> `S`; with a belly flag -> `s` (none occurred);
`prolonged_blockage_terminated`, `soil_breakthrough_terminated`, `rollover` -> `U`. The smoke read-out's endpoint was
"goal not reached". It is coded `U` for consistency with the other studies' labels. Rolling back was not
computed here. `-` = no valid drive: `m113_regeared_4x` on `f104_v2_group_0427_route_11` crashed with a bookkeeping
error in the new M113 module. The study excluded it from both sides of the comparison.

Checks: all cells equal the per-route extract below. The totals equal the tracked smoke read-outs and
`RESULTS_smoke.md` (including the designed / planner-proposal split).

| source (prefix `artifacts/traverse/offroad_vehicles_20260927/`) | status | SHA256 |
|---|---|---|
| `analysis/k4_extract.jsonl` (per-route records extracted read-only from the cluster run folders) | local-only | `511cfaae7af7dea8a5a2919e96e229e1477a0447e317fc7ed73f4c616d3da496` |
| [`analysis/scripts/k4_extract.py`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/analysis/scripts/k4_extract.py) | tracked | `a72c6e7c4591947e7aa876989ade4c44aea0d63f3c529a69530da8008122f6b6` |
| [`analysis/smoke_polaris_v1_final.json`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/analysis/smoke_polaris_v1_final.json) | tracked | `cfb4ad6f8f11d304170378d260a7e41be82523fecd5f6c03d3c06e3bce2556df` |
| [`analysis/smoke_m113_v1_final.json`](https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/analysis/smoke_m113_v1_final.json) | tracked | `c5c98948fb5f9143f8d57074ed5605eb61c827f43328995c8824e28c466000db` |
| `scratch/S3/sample_A.json` (the 144 routes and the stored HMMWV/Gator outcomes) | local-only | `afb0607f659b7bbbc989c8b9428d32728bece9943839d43b3e21ab18c7dffdec` |
