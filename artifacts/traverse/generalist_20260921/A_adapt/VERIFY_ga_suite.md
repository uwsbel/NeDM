# Verification: ga_suite / ga_analyze / gb_track_analyze (module ga_suite) — round 2

Verifier run 2026-09-21 (late evening) against the CURRENT code (after fix round 1, `NOTES_ga_suite.md` section 5).
This file replaces the round-1 report; its findings 2.1 (nearest-sample cross-track), 2.2 (sha256 pick agreement) and
2.5 (suite lock) were fixed and are re-verified below; 2.3 (rigid reference drives must share the A3 job) is a launch rule
that is being respected; 2.4 (`split` field is provenance only) is noted in the module note.

Rules respected: nothing submitted to the cluster (`ssh amd` read-only: sha256sums, `ls`, `squeue`, one read-only python
over `outcome.json` files); no simulator started, no CRM lock needed; GPU use = two limited pick passes (about 1.2 GB
while the A2 training lanes held 14 GB). No repo file modified except this one; scratch in `/tmp/verify3/`.

Verdict: **pass with issues**. Every number in the implementer's report and in the fix-round note reproduces bit-for-bit
(pick passes, merge, tracking suite, night-2 reproduction, all gb_track self-tests); the launched CRM suite is byte-identical
to the locked local files and its 2,663 drives are complete on the cluster; the reuse of night-2 drives is valid end to end.
Two defects remain in the analysis code (section 2): one changes a decision regression bound (off-route events never count
as unsafe), one is a small reproducibility hole (pick-file order enters the agreement bootstrap). Both have one-line to
three-line fixes and do not touch any suite, pick, route or task file.

## 1. Reproduced (evidence)

| Claim | What I did | Result |
|---|---|---|
| Case set 600 groups, first 400 identical | `cmp` loops `cases/pair_v1_first400` vs `cases/pair_v1/cases` | 400/400 cases, 4,800/4,800 routes identical; `cases.json` records 600, seed 20260921104; 600 case files in the dir (no silent `[:600]` truncation) |
| Generator margin | 4-D start+goal distance of the 600 fresh cases to `cases_night2` (1,200) and `cases_eval` (200); generator log | min 2.004 m / 2.048 m; fresh self-min 2.008 m; log: "4507 existing pairs avoided (margin 2.0 m)" |
| Suite assembly | byte-compare all 800 case + 800 `route_00` copies with their sources; re-hash every sha256 in `suite.json` | 0 differ, 0 sha mismatches; header 800 = 600 + 200, strata 184/96/241/129/100/50, min fresh-reused 2.048 m |
| Geometry source | read `f104_n2_iter.corridors` / `gen_planner.corridors` / `f104_n2_dataset.sample_map` | both corridor builders sample `DS.G`, which only `DS.init_map` (depth map) fills; `GP.set_map` (heightmap) is never called by `ga_suite.py`; `corridors_batched = True` in every pick file I opened |
| Pick determinism | `--stage picks --pass crm:Scrm` and `rigid:Srigid --limit 2` into `/tmp/verify3/picks` | 8 route files per pass byte-identical to the stored ones; route sha256 / pool index / z_mean / P / T / kind / round equal on 4 groups x 2 arms x 2 passes; one-shot pool 256/256 on 3 groups; crm/Scrm night-2 match A 2/2, B 2/2 |
| Merge | copied `suite.json`, `cases/`, four pick dirs to `/tmp/verify3/merge`, re-ran `--stage merge` | `tasks_crm.json`, `tasks_rigid.json`, both `run_index_*.json` byte-identical; 6,109 route files identical; merge counts identical (3058 distinct / 2663 drives / 395 reused; 3051 / 3051 / shards 477-598); lock line 1 identical `4327c110...910763` |
| Lock | `tail -n +2 SUITE_LOCKED.sha256 \| sha256sum -c` | OK for all three files; sha256 now = lock = fix-round note |
| Run index / reuse | all 3,200 pick route ids per world resolve to a task row with the same sha256; the 395 `run=false` rows: local night-2 run files present, sha256 equal to the night-2 route file, night-2 arm in {A, B}, case bytes equal to `cases_eval`; 0 `run=true` rows whose sha is a night-2 A/B drive; rigid: 0 groups split across shards | all 0 / all present |
| Cluster state (read-only) | sha256 of `C/generalist/suite/{tasks_crm.json,suite.json}`; `routes_crm` 3,058 and `cases` 801 entries; `suite_out/runs` | shipped files equal the locked local files; 2,663/2,663 run dirs with `episode_complete.json`; every `outcome.json`: config `crm_main_step1ms_spacing008`, `physics_dt_s` 0.001; `collection_request.episode_seed` equals the task row on 2,663/2,663; statuses goal_reached 2,201 / soil_breakthrough 352 / prolonged_blockage 110; `R/generalist_suite` does not exist (rigid not launched alone, as required by 2.3); queue empty |
| Rigid checkpoints | sha256 of the 5 local `night2_v1/final/N2_s*.pt` vs cluster `gen_v1/models/N2_s*.pt` | identical |
| Tracking suite | `ga_suite.py --suite tracking --out /tmp/...` | identical to the stored file except `created`; 55 groups == the twin `test` split (1089/56/55); 423 routes = every test-group route present in `collect_v1` (423) and `production_v3` (660); `production_v4` holds 0 test routes; no other CRM run dir holds test routes |
| ga_analyze night-2 repro | exact command of note test 7 with `--cluster-ci` into `/tmp/verify3/ga_analyze` | 18 of 19 summary blocks identical to stored `fix1/n2_repro.json` (only `world`, not passed); every common field of `PRIMARY_B:A`, the four contrasts and the six per-arm rates equal to night-2's `eval_iter_crm/results.json`; cluster CI [-11.76, -4.11] 8/0 p 0.0078 equal to night-2's `cluster_ci_fail.json`; time ratio 0.886 (p95 0.930) |
| ga_analyze through the run index (new) | `--arm A=picks_crm_Scrm:A:/tmp/noruns --arm B=...:B:... --suite suite.json --run-index run_index_crm.json` | the 200 reused groups resolve to the night-2 drives (all `run_dir` under `crm_night2_v1`), fresh groups missing 600/600 as expected; the `PRIMARY_B:A` block is bit-identical to the night-2 reproduction (-7.5 [-12.0, -3.5], p 0.000729, p95 -4.0) |
| Agreement A0 (fix 2.2) | note 5.2 command, CRM world | `own_vs_other` identical to stored `fix1/agreement_A0_crm.json`: diff +0.547 m, p05 +0.433, CI [+0.411, +0.685], closer 485/800, identical 29/0 |
| gb_track self-tests | identical-copies crm + rigid, truncation, bounds, bounds_mismatch | all four result files byte-identical to the stored `_fix1` files; ratio 1.0000 / CI [1, 1] / degenerate in every stratum; truncation 515 -> 154 frames, 102 -> 39 reached, 63 capped, Winsorised 0.4256 -> 4.0546 m, ratio 9.527; bounds "checked on 1 routes"; mismatch `AssertionError`, rc 1 |
| Polyline metric (fix 2.1) | hand-made L-route: interior/vertex stations 0; beside-segment 3.0 / 1.0; past-end and before-start 5.0; far 10 -> capped 6.0; straight path sampled at 0.1 / 1 / 3 m gives 0.7 / 0.2 / 0.0 at every density; single sample; duplicate samples; loop-back trajectory min 1.0 | all as expected |
| Axis / shape conventions | opened `collect_v1` and `production_v3` recordings | `pose (T,3)` = x, y, yaw (yaw[0] -1.823 vs `reference_headings[0]` -1.821), `state[:,0]` = vx, `desired_speed_mps (T,)`, `reference_waypoints (102,2)` |
| Unsafe rule vs collector statuses | read `gen_collect.py`, `crm_collect.py`, both ext collectors | the six known statuses classify as intended (timeout safe; rollover / bounds_exit / prolonged_blockage / soil_breakthrough unsafe); the ext collectors' 40 s near-stop rule writes `prolonged_blockage_terminated` (event carries `rule: near_stop_any_throttle`), so it is counted as unsafe, consistent with the plan |

Note on the implementer's report: its gb_track numbers (0.430 -> 4.058, ratio 9.44) are the pre-fix-round values; the
current code gives 0.4256 -> 4.0546 (ratio 9.527), which is what the stored `_fix1` files and my re-run show. Not a defect.

## 2. Problems

### 2.1 (major) The "off-route" unsafe event can never fire — one of the plan's four unsafe events is silently excluded

PLAN Milestone B: unsafe = rollover, soil breakthrough, prolonged blockage, **off-route**; regression bound: paired
unsafe-rate difference <= +1 point. `gb_track_analyze.UNSAFE_SUBSTR` contains `'off_route'`, but no collector writes such
a status: `grep off_route` over `gen_collect.py`, `crm_collect.py`, `crm_collect_ext.py`, `gen_collect_ext.py`,
`gc_control.py` finds nothing, and the ext collectors' stop rules are the native `StopPolicy` plus the 40 s near-stop rule
only. So a policy that leaves the corridor by more than the 6 m cap and later reaches the goal is scored `goal_reached`,
`unsafe = 0`, with its cross-track penalty limited to 6 m per station (Winsorised). On the suite's own native-PID recordings
this is not hypothetical: the maximum trajectory-to-route-polyline distance exceeds 6 m on 15/423 rigid routes and 2/423
CRM routes, and 6 of the rigid ones end in `goal_reached` (e.g. `f104_v2_group_0220_route_05` 8.0 m, `_0652_route_05`
8.1 m; all in the infeasible stratum). A policy that shortcuts around an obstacle would therefore gain on cross-track and
completion without any unsafe count.

Smallest fix (in `route_metrics`, three lines; no suite file changes): derive the event from the trajectory with the
function that already exists —
`_, d_traj, _ = station_xtrack(pose[:n, :2], st, np.inf); off_route = bool(d_traj.max() > cap_m)`
(the roles of stations and samples swapped: distance of every driven sample to the reference polyline), report `off_route`
per route and set `unsafe = int(status-substring or off_route)`. Keep the status-substring rule. Re-run the identical-copies
self-tests (still exactly 1 / 0 pts) and note the new count on the native-PID recordings (rigid 15, CRM 2 of 423) in the
module note. Optional: also list the recorded PID off-route routes in `tracking_suite.json` so the strata are pre-registered
with this flag.

### 2.2 (minor) The pick-agreement bootstrap depends on directory listing order

`ga_analyze.main` line 205 builds `picks_by` from `arm['picks'].glob('*.json')` **unsorted** (whereas `collect()` on
line 70 sorts). The bootstrap in `agreement()` draws indices into `G3`, whose order is the dict insertion order, so the same
data in a different order gives a different resample set. On the real A0 picks (ext4 hash order vs sorted vs reversed, same
seed): diff +0.5471 m in all three, but p05 +0.4335 / +0.4333 / +0.4294 and CI upper +0.6847 / +0.6859 / +0.6836. On this
machine the order is stable (ext4 dir_index), which is why every reproduction so far agreed; after an rsync to another
file system the decision percentile can move in the third decimal. The point estimate, sign counts and every other summary
block are unaffected (`collect()` is sorted and `G` follows it).
Smallest fix: `for p in sorted(arm['picks'].glob('*.json')):` on line 205. Re-running the stored `fix1/agreement_A0_*`
will then give p05 +0.4333 (CRM) instead of +0.4335; record the new values.

### 2.3 (open, not a code defect) Rigid reference drives still wait for the A3 job

Round-1 finding 2.3 stands as a launch rule: `tasks_rigid.json` (3,051 rows, shard = md5(group) % 6) must be concatenated
into the A3 rigid tasks file so every arm of a group runs in one array task. Verified respected so far: `R/generalist_suite`
does not exist on the cluster and the queue is empty. The A3 builder must use the same shard rule and must not reuse an id.

## 3. Notes

- The CRM suite drives are finished on the cluster (2,663/2,663, no failures, 4 h windows on mi3501x/mi3508x); nothing has
  been synced back yet. `ga_analyze` needs `runs/<id>/{trajectory.npz,outcome.json,episode_complete.json}` locally plus
  `--run-index run_index_crm.json` (the 395 reused rows resolve to the local night-2 copies, verified above).
- `ga_analyze` per-stratum blocks share one rng stream, so their numbers depend on the stratum order (sorted, stable) —
  deterministic under `--seed`.
- `gb_track_analyze.paired.ratio` draws a fresh index set when a metric has non-finite entries (heading with no reached
  station, `positive_work_kj = None`); exercised with a synthetic 12-route set: n drops to 11 for those metrics, the other
  metrics keep the shared paired resamples, group-clustered p95 computed. Deterministic.
- `winsor_mean` at 5 % does not soften the cap when more than 5 % of a route's stations are capped (90 x 0.2 + 10 x 6.0
  gives 0.78 = plain mean); this is the plan's definition, just worth knowing when reading infeasible-stratum numbers.
- `write_suite_lock` silently omits a missing file (round-1 fix remark); the current lock covers all three files.
- Timing convention, anchor/branch frames, float16 overflow and observation contracts are not exercised by this module;
  float16 scoring is the night-2 path and reproduces it bit-for-bit.

Scratch: `/tmp/verify3/{picks,merge,track,ga_analyze,ga_e2e,revorder,tracking_suite.json}`.
