# Module note: frozen suites and decision statistics (ga_suite, ga_analyze, gb_track_analyze)

Written 2026-09-21 (first attempt 14:20-14:50, interrupted by the session usage limit; resumed and finished 17:20-17:50).
Files: `scripts/ga_suite.py`, `scripts/ga_analyze.py`, `scripts/gb_track_analyze.py` (all new; no existing repo file was
edited). Artefacts under `artifacts/traverse/generalist_20260921/A_adapt/suite/` (planner suite),
`.../B_tracker/suite/tracking_suite.json` (tracking suite), self-tests under `A_adapt/selftest/{ga_suite,ga_analyze,gb_track}/`.
All commands run from the repo root with `PYTHONPATH=src:scripts /home/harry/miniconda3/envs/nedm/bin/python`.
Nothing was submitted to the cluster; `ssh amd` was used read-only (sha256 of the rigid checkpoints, directory listings).
No Chrono process was started by this module (no CRM lock needed).

## 1. What was built

### ga_suite.py --suite planner (PLAN A0)

Three stages (`--stage all|assemble|picks|merge`):

- assemble: the 600 fresh `f104_pair_group_0000-0599` cases (`cases/pair_v1/cases`, seed 20260921104, manifest with 600
  records) and the 200 reused `f104_crm_eval_group_0000-0199` cases (`crm_f104_v1/cases_eval/cases`) are copied
  byte-for-byte (case json + `routes/<g>/route_00.json`) into `suite/cases/`. `suite/suite.json` lists every group with
  stratum (fresh / reused), evaluation_stratum, the split field of the source case file, source paths and the sha256 of the
  case and route file, plus the minimum 4-D start+goal distance between the two strata.
- picks: the reference arms A (deployed one-shot 256, rng tag `crm_proposal` in the CRM world, `gen_night2` in the rigid
  world, exactly as `planner_arms.py`) and B (CEM 4x64, tag `n2iter_cem4x64` in both worlds) for the two specialists
  S_crm (`crm_f104_v1/train_v1/deploy/CRM_N2_s*.pt`) and S_rigid (`fdm_f104_50h_20260909/night2_v1/final/N2_s*.pt`, which is
  byte-identical to the cluster's `gen_v1/models/N2_s*.pt`; sha256 re-checked on 09-21 17:30, all five files equal). The loop is
  the `planner_arms.main` loop (imported pieces: `arm_specs`, `load_case`, `route_json`, `f104_n2_iter.oneshot / plan_iter /
  Scorer / selftest`) with one difference: the corridors come from the OptiX static depth map
  (`f104_n2_dataset.init_map(crm_f104_v1/map_root)`) in BOTH worlds; the heightmap path of `planner_arms.py --rigid-arena` is
  never called. I did not wait for `ga_planner.py`; after it appeared I used it as an independent cross-check of the rigid
  depth-map path (section 3, test 4). The one-shot pool is checked route by route against `gen_planner.proposal_pool` on the
  first 3 groups of every pass (`selftest_groups` in `summary.json`), and the batched corridor builder is checked bit-for-bit
  against `gen_planner.corridors` on the first call (`corridors_batched: true`). Output per (world, model):
  `suite/picks_<world>_<model>/{picks/<g>.json, routes/<g>__<model>_<arm>.json, tasks.json, summary.json, PICKS_LOCKED.sha256}`
  in the planner_arms formats (route ids carry the model tag so the four passes can share one runs directory). On the
  reused groups every pick is compared with the night-2 picks (`crm_night2_v1/planner/iter_crm/picks`) by candidate index
  and route sha256 (`n2_reference` in `summary.json`). `--jobs 4` runs the four passes as parallel subprocesses.
- merge: per world ONE tasks file, deduplicated by route sha256 across the two models (`aliases` keeps the other route ids),
  the route files copied to `suite/routes_<world>/`, and `suite/run_index_<world>.json` mapping every route id (of any
  model/arm) to the id that is actually driven or to the already-driven reference. CRM rows
  (`suite/tasks_crm.json`) are relative to `CRM_ROOT=/work1/dannegrut/harry/experiments/crm_f104_20260916`:
  `case = generalist/suite/cases/<g>.json`, `route = generalist/suite/routes_crm/<id>.json`, `episode_seed`, `tier` = group
  index, `arms = ['Scrm:A', ...]`, `extra: []` (crm_worker.py reads id/case/route/run/tier/episode_seed/extra). Rows whose
  route sha256 equals a night-2 drive of arm A or B in `crm_night2_v1/planner/eval_iter_crm` (`--ref-arms A,B`; same collector
  config `crm_main_step1ms_spacing008`, physics 1 ms, checked from the reference `outcome.json`) get `run=false`,
  `ref_run = night2/eval_iter_crm/runs/<id>` (cluster, relative to CRM_ROOT) and `ref_run_local`; a route that only night-2
  arms C-F drove is driven again and only annotated (`n2_route_driven_by_other_arm`; count 0 in this suite). Rigid rows
  (`suite/tasks_rigid.json`) are relative to `/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/gen_v1`:
  `arena = f104`, `case = generalist_suite/cases/<g>.json`, `route = generalist_suite/routes/<id>.json`,
  `shard = md5(group) % 6`, no `episode_seed` (gen_collect.py has no such flag; gen_runner.py reads id/case/route/shard/run).
  `suite/merge_<world>.json` holds the counts.

### ga_suite.py --suite tracking (PLAN B0)

Takes the 55 test-split groups from `crm_night2_v1/datasets/twin_crm.npz` (`group`, `split`), their designed routes
`route_00..11` from `fdm_f104_50h_20260909/cases_night2/cases/routes/<g>/`, and keeps a route only if
`trajectory.npz + command_reference.npz + outcome.json` exist for it in both `crm_f104_v1/collect_v1/runs/<g>_route_NN`
(CRM PID recording) and `fdm_f104_50h_20260909/production_v3/runs/<g>_route_NN` (rigid PID recording), and if the recorded
`reference_waypoints` equal the route file (max |diff| <= 1e-6 m in both worlds). Stratum feasible = recorded PID reached
the goal in both worlds, infeasible = the rest. Every route carries the sha256 of the route file and the case file, the
recorded status / elapsed / frames / positive work / trajectory sha256 per world, speed profile and lateral offset.

### ga_analyze.py (Milestone A statistics)

Arms are `NAME=<picks dir>:<arm letter>:<runs dir>[,<runs dir>]` (the picks dir may be the pass directory, its `picks/`
subdir is used). The outcome of a group under an arm is the label of the driven route of that arm
(`f104_n2_analyze.labels`: fail = status != goal_reached, unsafe adds the backward-motion tests after the 1 s settle,
exactly as `n2_planner_analyze.py`). Run resolution: `run_index[route_id].ref_run_local` (reused reference drive) else
`<runs dir>/<driven_as>` else `<runs dir>/<route_id>`. Contrast `X:Y` keeps the n2 convention (`rate_a` = X, `rate_b` = Y,
`diff = label(X) - label(Y)` in points; with `--label fail` this equals goal-reached(Y) minus goal-reached(X)), so the
decision rule is written TEST:REFERENCE and the added `p95_one_sided_pts` (95th percentile of the 4,000-resample group
bootstrap of diff) must be below `--margin-pts` (3.0). Also added: the same block per stratum (fresh / reused / all from
`--suite`) and per evaluation_stratum; the paired elapsed-time ratio on groups where both arms reached the goal (median over
groups of t_X / t_Y, 95th percentile of its bootstrap < `--time-bound` 1.10; the ratio of medians is reported too); exact
two-sided McNemar on discordant groups; pick agreement by route sha256 for every arm pair, and with `--agreement H:OWN:OTHER`
the bootstrap of agree(H, OWN) - agree(H, OTHER) (5th percentile > 0) (SUPERSEDED in fix round 1, section 5: agreement by
route geometry, the sha256 counts stay as an extra); with `--cluster-ci --cases <dir>` the 9-cluster
terrain-feature CI of `scripts/n2_cluster_ci.py` (subprocess) as a robustness line. `results.json = {summary: {...,
PRIMARY_X:Y, contrasts, rates, time_ratio, strata, by_evaluation_stratum, agreement, cluster_ci}, per_group: {g: {arm:
{fail, unsafe, status, elapsed, max_tilt, tilt30, work_kj, route_id, route_sha256, P, z, T_cmd, run_dir}}}}` (n2 format plus
the extra keys), so `n2_cluster_ci.py` reads it unchanged.

### gb_track_analyze.py (Milestone B metrics)

Per run: station-based cross-track = for every reference station (`command_reference.reference_waypoints`) the distance
to the nearest trajectory point (one vectorised (m, T) distance matrix, argmin per station; SUPERSEDED in fix round 1,
section 5: distance to the trajectory polyline), unreached stations capped at
`--cap-m` 6 m; per route the Winsorised mean (5 % each side), mean, median, p95, max, the number of capped stations and the
station completion (fraction within the cap). Heading error = |wrap(yaw at the nearest trajectory point - reference
heading)| over reached stations (deg). Speed error = |vx - desired_speed_mps| per frame from frame 20 (the settle skipped
by the labels). Completion = outcome status goal_reached (decision metric) and station completion. Unsafe = status
contains rollover / breakthrough / blockage / off_route / bounds_exit (timeout is not unsafe); every status outside the six
strings the two unmodified collectors write today is listed in `unknown_statuses` and printed as a warning. Positive work,
mean |da| per channel and L1, elapsed time. Action bounds: whenever at least two arms of a route carry `action_bounds` /
`action_squash` / `action_affine` / `policy_meta.action_*` in `outcome.json` they are asserted equal (raises on a
mismatch, test 8). Paired statistics per test arm vs `--reference` (default native_pid) per stratum: cross-track ratio
mean(test)/mean(reference) with a route-level paired bootstrap (4,000; one-sided 95th percentile < 0.90) and a
group-clustered bootstrap as robustness, median ratio; completion difference in points (5th percentile > -3); unsafe-rate
difference (95th percentile <= +1); speed-error ratio (95th percentile < 1.10); heading, work, time and |da| ratios
reported; McNemar on completion and unsafe; `decision[test:reference].pass_all`. `--truncate FRAC --truncate-arm NAME`
cuts that arm's trajectories for the cap self-test.

## 2. How to run

```bash
PY="PYTHONPATH=src:scripts /home/harry/miniconda3/envs/nedm/bin/python"
# A0 suite: assemble + 4 pick passes in parallel + merge (about 15 min on the 5090 while shared)
$PY scripts/ga_suite.py --suite planner --out artifacts/traverse/generalist_20260921/A_adapt/suite --jobs 4
# single pass / single stage
$PY scripts/ga_suite.py --suite planner --stage picks --pass crm:Scrm --out artifacts/traverse/generalist_20260921/A_adapt/suite
$PY scripts/ga_suite.py --suite planner --stage merge --out artifacts/traverse/generalist_20260921/A_adapt/suite   # --ref-arms A,B
# B0 suite
$PY scripts/ga_suite.py --suite tracking      # -> artifacts/traverse/generalist_20260921/B_tracker/suite/tracking_suite.json
# Milestone A analysis (example: test arm H vs reference S_crm on CRM; runs synced to <runs>)
$PY scripts/ga_analyze.py --arm S=artifacts/traverse/generalist_20260921/A_adapt/suite/picks_crm_Scrm:B:<runs> \
    --arm H=<H picks dir>:B:<runs> --primary H:S --label fail --suite artifacts/traverse/generalist_20260921/A_adapt/suite/suite.json \
    --run-index artifacts/traverse/generalist_20260921/A_adapt/suite/run_index_crm.json --margin-pts 3.0 --time-bound 1.10 \
    --agreement H:S:<other specialist arm name> --cluster-ci --cases artifacts/traverse/generalist_20260921/A_adapt/suite/cases --world crm --out <results.json>
# Milestone B analysis
$PY scripts/gb_track_analyze.py --suite artifacts/traverse/generalist_20260921/B_tracker/suite/tracking_suite.json \
    --arm native_pid=<runs> --arm held_pid=<runs> --arm policy=<runs> --reference native_pid --out <results.json>
```

Shipping the A0 suite (NOT run by this module; the drives use the unmodified collectors from C and R):

```bash
S=artifacts/traverse/generalist_20260921/A_adapt/suite
C=/work1/dannegrut/harry/experiments/crm_f104_20260916; R=/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/gen_v1
ssh amd "mkdir -p $C/generalist/suite $R/generalist_suite"
rsync -az $S/cases/ amd:$C/generalist/suite/cases/
rsync -az $S/routes_crm/ amd:$C/generalist/suite/routes_crm/
rsync -az $S/tasks_crm.json $S/suite.json amd:$C/generalist/suite/
rsync -az $S/cases/ amd:$R/generalist_suite/cases/
rsync -az $S/routes_rigid/ amd:$R/generalist_suite/routes/
rsync -az $S/tasks_rigid.json $S/suite.json amd:$R/generalist_suite/
# CRM: bash $C/source/scripts/crm_launch.sh $C/generalist/suite/tasks_crm.json $C/generalist/suite_out configs/crm_main.json 4
# rigid: GEN_TASKS=$R/generalist_suite/tasks_rigid.json GEN_OUT=$R/generalist_suite_out sbatch -p mi2104x -c 128 --array=0-5 --export=ALL $R/gen_array.sbatch
# results back (label files only), then ga_analyze with --runs <local runs> and --run-index
```

For the analysis after the drives: sync `runs/<id>/` (trajectory.npz, outcome.json, episode_complete.json) of each world
into one local directory; the run index maps every model/arm route id to the driven id or to the night-2 reference drive.

## 3. Tests (exact commands and numbers)

1. Case set readiness (brief: wait for 600 groups, verify the first 400). `cases/pair_v1/cases/cases.json` (note: the
   manifest sits inside `cases/`, not one level up) reports `records: 600`, seed 20260921104, prefix `f104_pair_group`;
   the generator log `cases/gen_pair_v1_600b.log` ends with `{"groups": 600, "episodes": 7200, ...}`. Byte check
   (`cmp` loop over `cases/pair_v1_first400/cases/*.json` and `.../routes/*/*.json` against `cases/pair_v1/cases/`):
   400 case files checked, 0 differ; 4,800 route files checked, 0 differ.
2. Suite assembly (`ga_suite.py --suite planner --stage assemble`, then in the resumed session a re-verification): 800 groups
   (600 fresh + 200 reused); all 800 case files and 800 `route_00.json` copies byte-identical to their sources (cmp loop, 0
   differ); every sha256 in `suite.json` re-hashed from the copies, 0 mismatches; minimum 4-D fresh-reused distance 2.05 m,
   0 exact pair matches; evaluation strata crater_cross_slope 184, crater_entry_cross_exit 96, hill_cross_slope 241,
   hill_entry_cross_exit 129, long_traverse 100, roughness_transfer 50. The picks (below) were computed at 14:31-14:46 from
   these copies, after the final 600-group regeneration (cases 14:18, manifest 14:22), so they are on the final case set.
3. Pick passes (`ga_suite.py --suite planner --stage picks --jobs 4`, logs in `suite/logs/picks_<world>_<model>.log`):
   4 passes x 800 groups, wall 880 / 889 / 886 / 888 s (four in parallel on the shared 5090, 0.52-0.59 s per group per arm).
   One-shot pool checked route by route against `gen_planner.proposal_pool` on the first 3 groups of every pass (256/256
   each), batched corridors identical to `gen_planner.corridors` in every pass. Per pass: distinct routes 1,571 (crm/Scrm),
   1,576 (crm/Srigid), 1,573 (rigid/Scrm), 1,575 (rigid/Srigid); A=B agreement 29 / 24 / 27 / 25 groups. Night-2 reproduction
   on the 200 reused groups: crm/Scrm A 200/200 index and 200/200 route sha256, B 200/200 and 200/200 (the CRM specialist on
   the depth map reproduces the night-2 picks exactly); rigid/Scrm B 200/200 (arm B has the same tag, corridors and model in
   both worlds; A 19/200 because the rigid one-shot tag `gen_night2` draws a different pool, by planner_arms design);
   crm/Srigid A 26/200, B 0/200 and rigid/Srigid A 2/200, B 0/200 (different model: expected to differ). Mean pick logit
   z_mean: Scrm A -5.639 (both worlds) B -6.329 (both worlds, identical); Srigid A -9.387 (crm tag) / -9.412 (rigid tag),
   B -9.655 (both). Mean predicted failure P of the picks: Scrm A 0.051 B 0.017; Srigid A 0.0002 B 0.0001.
4. Independent cross-check of the rigid depth-map path (resumed session): `ga_planner.py --world rigid --map-root
   crm_f104_v1/map_root --models 'night2_v1/final/N2_s*.pt' --arms A,B --groups f104_crm_eval_group_0000,f104_pair_group_0000,
   f104_pair_group_0599 --out A_adapt/selftest/ga_suite/xcheck_ga_planner_rigid`: route sha256, pool index and z_mean equal
   (difference 0.0) to `suite/picks_rigid_Srigid/picks/<g>.json` for arms A and B on all three groups.
5. Merge (`ga_suite.py --suite planner --stage merge`, re-run in the resumed session with the explicit `--ref-arms A,B` rule):
   CRM 3,200 (model, arm) picks -> 3,058 distinct routes (89 shared between the two models), 2,663 drives + 395 reused
   night-2 drives (all 395 are routes that night-2 arm A or B drove on the reused groups: 369 touch only S_crm picks, 26 are
   also an S_rigid pick; 1,137 night-2 drives of any arm usable, 0 routes that only arms C-F drove); by stratum fresh 2,293
   distinct / 2,293 drives, reused 765 distinct / 370 drives. Rigid 3,200 picks -> 3,051 distinct routes (97 shared),
   3,051 drives, 0 reused (no rigid depth-map reference drive exists); shards 0-5 hold 477 / 598 / 537 / 504 / 488 / 447 rows.
   Distinct routes per model: CRM Scrm 1,571 / Srigid 1,576; rigid Scrm 1,573 / Srigid 1,575.
6. Tracking suite (`ga_suite.py --suite tracking`, re-run in the resumed session): 55 test groups, 423 routes with recordings
   in both worlds, 141 feasible / 282 infeasible; dropped 237 routes with no CRM recording (collect_v1 drove 6-10 of the 12
   designed routes per test group: 8 groups with 6, 17 with 7, 19 with 8, 6 with 9, 5 with 10), 0 missing rigid, 0 reference
   mismatches. By speed profile (n / feasible): constant_2 112 / 18, constant_4 107 / 39, constant_6 114 / 55, smooth_2_6_2
   90 / 29. Status pairs (crm | rigid): goal|goal 141, breakthrough|goal 194, blockage|goal 35, breakthrough|blockage 36, others 17.
7. ga_analyze night-2 reproduction (`selftest/ga_analyze/n2_repro.{json,log}`):
   `ga_analyze.py --arm A=crm_night2_v1/planner/iter_crm:A:crm_night2_v1/planner/eval_iter_crm/runs ... (arms A-F)
   --primary B:A --contrasts C:A,D:A,E:B,F:B --label fail --agreement B:A:C --cluster-ci --cases crm_f104_v1/cases_eval/cases`:
   200/200 groups, 0 missing drives. B:A fail 1.5 % vs 9.0 % (goal reached 98.5 vs 91.0), diff -7.5 [-12.0, -3.5], worse
   2 vs 17, McNemar p = 0.000729, one-sided p95 -4.0 points (< 3.0: pass); identical to `eval_iter_crm/results.json` in every
   field of the primary contrast, the four contrasts and the per-arm rates (same seed and resample order). Time ratio B/A:
   median paired ratio 0.886 (p95 0.930), ratio of medians 0.866, n 180. Feature-cluster CI B:A [-11.8, -4.1], 9 clusters,
   wins/losses 8/0, sign p 0.0078, identical to night-2's `cluster_ci_fail.json`. Agreement demo B:A:C: 2.5 % vs 1.0 %,
   one-sided p05 +0.5.
8. gb_track_analyze (`selftest/gb_track/`): fixtures are symlinks to 20 recorded CRM PID episodes
   (`crm/native_pid/<id>` -> `crm_f104_v1/collect_v1/runs/<id>`; `crm/policy/<id>` -> the same run) and 20 rigid ones
   (`production_v3/runs`), 12 feasible + 8 infeasible routes of the tracking suite each (`<world>/suite_subset.json`).
   - identical copies, CRM: cross-track ratio 1.0000, p95 1.0000, CI [1.0, 1.0], degenerate = true, group-clustered p95 1.0;
     completion +0.0 (p05 +0.0), unsafe +0.0 (p95 +0.0), speed-error ratio 1.0000 in every stratum (feasible n 12,
     infeasible n 8, all n 20); native PID cross-track 0.331 m feasible / 3.860 m infeasible (median 6.0 = capped), station
     completion 100 % / 44.1 %.
   - identical copies, rigid: the same exact-1 / degenerate result (feasible 0.170 m, infeasible 1.090 m, all 0.538 m).
   - truncation (`--truncate 0.3 --truncate-arm policy` on `f104_v2_group_0048_route_00`): frames 515 -> 154, reached
     stations 102 -> 39, 63 stations capped at 6.0 m, median 6.0, p95 6.0, Winsorised mean 0.430 -> 4.058 m, ratio 9.44.
   - action bounds: two copies of one route with `action_bounds` injected in `outcome.json`: equal -> "action bounds checked
     on 1 routes"; throttle bound changed to [0, 0.8] in one arm (`bounds_mismatch/`) -> AssertionError, exit code 1.

## 4. Known limits

- Rigid picks use the depth map (plan convention), so they are not comparable with the night-2 rigid heightmap picks and no
  rigid reference drive exists yet: every rigid row is a new drive (3,051).
- The A arm's candidate pool depends on the world tag (`crm_proposal` vs `gen_night2`, planner_arms semantics), so the same
  model's A pick differs between worlds; the B arm (same tag, same corridors, same model) is identical across worlds.
- The reused night-2 drives are matched by route sha256 against arms A/B of `eval_iter_crm`; a route only arms C-F drove
  would be driven again (none in this suite).
- The tracking suite's cluster paths: the CRM copy of the night-2 cases was listed on the cluster; no rigid copy was found
  under gen_v1. The B0 drives go through the new-mode collectors from G, so the B5 task builder must ship the case and route
  files named in `route_file` / `case_file`.
- ga_analyze's time ratio uses groups where both arms reached the goal (conditioning on success, as the n2 median time does).
- gb_track_analyze's unsafe set is a status-substring rule; unknown status strings are reported, not classified.
- Heading error is only defined on reached stations; speed error uses `desired_speed_mps` of the recording (the follower's
  own target), so for the policy arm it measures deviation from the same reference schedule only if the ext collector
  writes the same `command_reference.npz` fields.
- The planner suite drives cost more than the plan's estimate: 2,663 CRM drives (about 13 simulated hours at ~17 s each,
  ~5 billed) rather than "800 x 2 models" because arms A and B pick different routes on ~96 % of the groups.

## 5. Fix round 1 (2026-09-21, after `VERIFY_ga_suite.md`)

Applied the verifier's findings 2.1, 2.2 and 2.5 (and the note asked for in 2.4). Files changed: `scripts/gb_track_analyze.py`,
`scripts/ga_analyze.py`, `scripts/ga_suite.py`, this note. No pick, task, route or case file was regenerated or rewritten; the
CRM drives launched from the current files stay valid: `sha256(suite/tasks_crm.json)` =
`287737e388778187c4fe307f8c9b40e67c1862a6756576f8131f0308c18c8a80` before and after this round (also `tasks_rigid.json`
`cc39cad5...c68f26e` and `suite.json` `64921952...844c1a` unchanged, file times 14:31 / 17:27 untouched). New self-test outputs
carry a `_fix1` suffix or sit in `selftest/ga_analyze/fix1/`; the old files were left in place for comparison. Finding 2.3
(rigid reference drives must run in the same array task as the later H/T/P arms) is a launch-policy item and is not a code
change in this round: do not launch `tasks_rigid.json` on its own; at A3 concatenate its rows into the A3 rigid tasks file.

Informational only: the `split` field that `suite.json` copies from the source case files (719 of the 800 suite groups say
`train`) is provenance and must not be used to select training data; both dataset builders blacklist the suite groups by
id prefix (`f104_crm_eval_group_*`, `f104_g1_test_group_*`, `f104_pair_group_*`), which is the rule that counts.

### 5.1 Cross-track distance to the trajectory polyline (finding 2.1)

`gb_track_analyze.station_xtrack` now returns, for every reference station, the distance to the driven trajectory as a
polyline: the station is projected onto every sample-to-sample segment (`t = clip(((p - a) . ab) / |ab|^2, 0, 1)`, distance
to `a + t ab`, vectorised as an (m, T-1, 2) array; zero-length segments use their start point) and the minimum over segments
is taken. The nearest-sample index is kept only for the heading lookup, so the heading error is unchanged. Unreached stations
are still capped at `--cap-m` 6 m. The new distance is never larger than the old one and does not depend on how densely the
trajectory is sampled, which is what let a slower drive look better tracked. Check against a plain Python loop on 50 random
station/trajectory sets (`/tmp/fix1/xtrack_check.py`): max |vectorised - loop| = 3.6e-15 m, polyline <= nearest sample on
every station, nearest-sample index identical.

Feasible stratum of the tracking suite (141 routes, native PID recordings, Winsorised route mean averaged over routes),
nearest sample -> polyline: rigid 0.2179 -> 0.1997 m (8.4 % of the old value was sampling artefact; by profile constant_2
3.3 %, constant_4 6.7 %, constant_6 12.0 %, smooth_2_6_2 6.2 %); CRM 0.4516 -> 0.4418 m (2.2 %; 0.8 / 1.8 / 3.0 / 1.7 %).
Plain route means rigid 0.2737 -> 0.2559 m, CRM 0.4888 -> 0.4793 m; capped stations 0 -> 0 and reached stations
12,435 -> 12,435 in both worlds. This confirms the verifier's numbers exactly.

Self-tests re-run (same commands as section 3, test 8, outputs `results_identical_fix1.{json,log}`,
`results_truncated_fix1.{json,log}`, `bounds/results_fix1.json`, `bounds_mismatch/run_fix1.log`):
- identical copies, CRM: cross-track ratio 1.0000, p95 1.0000, CI [1.0, 1.0], degenerate, group-clustered p95 1.0; completion
  +0.0 (p05 +0.0), unsafe +0.0 (p95 +0.0), speed-error ratio 1.0000 in every stratum; native PID cross-track 0.316 m feasible
  (was 0.331), 3.858 m infeasible (was 3.860; median 6.0 = capped), 1.733 m all (was 1.742); station completion 100 / 44.1 /
  77.6 %, heading 3.6 / 2.8 / 3.3 deg, speed error and |da| unchanged.
- identical copies, rigid: the same exact-1 / degenerate result; feasible 0.147 m (was 0.170), infeasible 1.078 m (was 1.090),
  all 0.519 m (was 0.538).
- truncation (`--truncate 0.3 --truncate-arm policy` on `f104_v2_group_0048_route_00`): frames 515 -> 154, reached stations
  102 -> 39, 63 stations capped at 6.0 m, median 6.0, p95 6.0, Winsorised mean 0.426 -> 4.055 m (was 0.430 -> 4.058),
  ratio 9.527 (was 9.442); heading error 3.9 -> 0.3 deg unchanged.
- action bounds: equal -> "action bounds checked on 1 routes"; mismatch -> AssertionError, exit code 1 (unchanged).

### 5.2 Pick agreement by route geometry (finding 2.2)

`ga_analyze.agreement` no longer decides on identical routes. For two picked routes X and Y, `d(X, Y)` is the symmetrised
mean nearest-waypoint distance (`0.5 * (mean over X's waypoints of the distance to the nearest Y waypoint + the same from Y
to X)`, in metres, 0 = same path), read from the route files next to the picks (`<pass>/routes/<route_id>.json`; extra
directories via `--routes-dir`). Per arm pair the results carry n, median / mean / p90 of d and the sha256-identical count
(kept as an extra; `identical_picks` in every contrast block stays too). With `--agreement H:OWN:OTHER` the per-group
difference `d(H, OTHER) - d(H, OWN)` is bootstrapped over groups (4,000 resamples); pass = its 5th percentile > 0; next to it
the sign fraction (groups with `d(H, OWN) < d(H, OTHER)`, ties counted separately) with the exact two-sided sign test.
The rng stream is consumed in the same order as before (agreement last), so all other statistics are bit-identical.

Test on the real A0 picks, with the CRM specialist's one-shot arm standing in for H (`selftest/ga_analyze/fix1/agreement_A0_<world>.{json,log}`):
```bash
S=artifacts/traverse/generalist_20260921/A_adapt/suite; O=artifacts/traverse/generalist_20260921/A_adapt/selftest/ga_analyze/fix1
$PY scripts/ga_analyze.py --arm ScrmA=$S/picks_crm_Scrm:A:/tmp/noruns --arm ScrmB=$S/picks_crm_Scrm:B:/tmp/noruns \
    --arm SrigidA=$S/picks_crm_Srigid:A:/tmp/noruns --arm SrigidB=$S/picks_crm_Srigid:B:/tmp/noruns \
    --primary ScrmB:SrigidB --agreement ScrmA:ScrmB:SrigidB --world crm --out $O/agreement_A0_crm.json     # same with picks_rigid_* / --world rigid
```
(no drives exist yet, so the contrast blocks are empty and only the agreement block is filled.)
- CRM-world picks, median d over 800 groups: S_crm A vs S_crm B 1.303 m (29 identical), S_rigid A vs S_rigid B 1.452 m (24),
  S_crm B vs S_rigid B 1.466 m (1), S_crm A vs S_rigid A 1.917 m (88), S_crm A vs S_rigid B 1.805 m (0), S_crm B vs
  S_rigid A 1.914 m (3). Statistic ScrmA:ScrmB:SrigidB: d(H, own) mean 1.939 m (median 1.303) vs d(H, other) mean 2.486 m
  (median 1.805), diff +0.547 m, one-sided p05 +0.433 m, CI95 [+0.411, +0.685] -> PASS; closer to own 485 / 800 (60.6 %,
  4 ties, sign p < 1e-4); sha256-identical own 29 / other 0 (the old statistic would have given 3.6 vs 0.0 %).
- rigid-world picks: S_crm A vs S_crm B 1.299 m (27), S_rigid A vs S_rigid B 1.442 m (25), S_crm B vs S_rigid B 1.466 m (1;
  arm B is identical across worlds, so this pair is the same in both), S_crm A vs S_rigid A 1.813 m (96), S_crm A vs S_rigid B
  1.874 m (1), S_crm B vs S_rigid A 1.838 m (3). Statistic: d(H, own) mean 1.994 m (median 1.299) vs d(H, other) mean 2.505 m
  (median 1.874), diff +0.512 m, p05 +0.390 m, CI95 [+0.367, +0.654] -> PASS; closer to own 492 / 800 (61.5 %, 4 ties).
The two medians the verifier quoted (1.30 m own-model A vs B, 1.47 m S_crm-B vs S_rigid-B) are confirmed. The stand-in test
is only a check that the statistic is non-degenerate and points the expected way (the same model's two arms are closer to
each other than to the other model's arm); the real H picks come from A2/A3.

Night-2 reproduction re-run with the new code (`selftest/ga_analyze/fix1/n2_repro.{json,log}`, exact command of section 3,
test 7): all 18 summary blocks other than `agreement` are identical to the stored `n2_repro.json` (primary B:A diff -7.5,
one-sided p95 -4.0, identical picks 5; feature-cluster CI identical). The geometric demo `B:A:C` now reads d(B, A) mean
1.727 m (median 1.166) vs d(B, C) mean 1.435 m (median 0.744), diff -0.292 m, p05 -0.528 -> FAIL, closer to A 69 / 200:
expected, because C is a second CEM arm of the same model and therefore closer to B than the one-shot arm A is; the demo
exercises the code path, it is not a decision.

### 5.3 Suite lock (finding 2.5)

`ga_suite.py` merge stage now ends by writing `<suite>/SUITE_LOCKED.sha256`, and `--stage lock` writes only that file.
Line 1 is one sha256 over (file name + content digest) of `suite.json`, `tasks_crm.json`, `tasks_rigid.json` sorted by name
(same scheme as `PICKS_LOCKED.sha256`); the following lines are `<sha256>  <name>` per file. Written on the current suite with
`$PY scripts/ga_suite.py --suite planner --stage lock --out artifacts/traverse/generalist_20260921/A_adapt/suite`
(nothing else was run; the merge stage was NOT re-run):
```
4327c110b962631ca6b4f849e049ff55320d748e0a0db62df35cc1953e910763  suite.json + tasks_crm.json + tasks_rigid.json (name + content, sorted)
64921952321fa4e20597bea8b0b4c1a61dad0ac2a763c6eb9351edfa30844c1a  suite.json
287737e388778187c4fe307f8c9b40e67c1862a6756576f8131f0308c18c8a80  tasks_crm.json
cc39cad584a6110744f4565bbe82249992a1372875099bffcb1215b79c68f26e  tasks_rigid.json
```
One-line check after the drives come back: `cd <suite> && tail -n +2 SUITE_LOCKED.sha256 | sha256sum -c` (ran: all three OK).
