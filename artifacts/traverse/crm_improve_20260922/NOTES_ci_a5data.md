# Module note: `scripts/ci_a5data.py` (task files for the short-approach closed loop and the new continuation data)

2026-09-22. One new file, `scripts/ci_a5data.py`; no existing repo file was edited; nothing was submitted, shipped or run in
Chrono. Outputs are under `artifacts/traverse/crm_improve_20260922/a5data/` (called `A` below). It builds on
`ga_approach.py` (generalist_20260921, "K1"): same straight approach route, same analysis formulas, generalised from
"3 s approach, decision at frame 60" to any approach length L (0.5 s -> frame 10, 1 s -> frame 20, 3 s -> frame 60),
and extended to the 1,200 twin training groups and to the six continuation routes per decision state (PLAN S2, S3).

## What each stage does

- `approach-routes --cases DIR`: per group the two ga_approach candidates (straight start-goal line at a constant 3 m/s,
  a waypoint every 0.5 m; route_00 at 3 m/s), the same checks (reference contract, planner validator, collector
  start/goal 0.25 m rule, the torch-free gc_control copies, and the frozen collector's own route reader) and the same
  rule (lower mean |grade| over the first 12 m wins, ties go to the straight line). Also checks that route_00 IS the
  straight line (largest lateral distance of a route_00 waypoint from the line < 0.01 m, start yaw within 0.01 deg of
  the line heading, both ends within 0.01 m) and lists every group where it is not. Twin groups: split taken from
  `crm_night2_v1/datasets/twin_crm.npz` and asserted equal to the case file's split.
  Writes `A/approach_<set>/<g>.json` and `A/approach_<set>_index.json` (set = suite or twin).
- `pass1-tasks`: the approach drives. Ids `<g>__p1_<len>` with len = `0p5`, `1`, `3` (the same id in both worlds;
  seed = first 8 hex of md5(id), identical across the worlds, unique in every file).
  - CRM rows (paths relative to CRM_ROOT = `/work1/dannegrut/harry/experiments/crm_f104_20260916`): case =
    `generalist/suite/cases/<g>.json` or `cases/night2/<g>.json` (the existing cluster copies, see below), route =
    `crm_improve/pass1_<set>/approach/<g>.json`, `extra ['--horizon-s', L]` (crm_worker puts it after its own
    `--horizon-s 120`, so L wins, as in K1), plus `episode_seed`, `tier` (suite order / sorted twin names) and provenance
    (route and case sha256, set, length, frame, split or stratum).
  - Rigid rows (absolute paths): case = `/work1/dannegrut/harry/experiments/generalist_20260921/a5/cases/<g>.json` (suite)
    or `/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/cases_night2_v1/<g>.json` (twin), route =
    `/work1/dannegrut/harry/experiments/crm_improve_20260922/pass1_<set>/approach/<g>.json`, arena `f104`,
    shard = md5(group) % 6, mode `native`, `extra ['--horizon-s', L]` (the K1 rigid pass-1 row format).
  - Files: one per (set, world, L) and one merged per (set, world) in `A/tasks/`, summary with sha256 in
    `A/tasks/pass1_tasks_summary.json`, shipping commands in `A/SHIP_pass1.txt` (NOT run).
- `analyze --runs DIR [DIR ...] --worlds crm rigid --length L`: per pass-1 run whose directory name ends in `__p1_<len>`
  (option `--run-suffix`, K1 runs use `__pass1`) the decision state at F = round(L / 0.05): pose, full 17-column
  state, vx, last applied action, station / lateral offset / remaining length along the approach route, the CRM
  sinkage increase (ga_approach's per-wheel formula), the moving flag, and the history window(s) cut at F
  (`--hist-T`, default 40; row t = [state[j] 12 columns | action[j-1]], j = F-T+1+t, valid iff 1 <= j <= F, the
  ga_planner and ga_branch_dataset convention). A recording with exactly F rows (what `--horizon-s L` produces) uses
  the collector's terminal state / pose. Writes `decision_<world>.json` (every record), `hist_<world>_T<T>/<g>.npz`,
  `poses_<world>.json` (analysis set = moving in every world given, K1 convention), `poses_<world>_all.json` (every
  group with a decision state), `poses_<world>_masked.json` (no history: the all-masked arm), `poses_<world>_T<T>.json`
  for extra window lengths, and `pass1_state.json`. Pose entries follow ga_planner's `--poses` format (explicit pose +
  history npz for F-row recordings, `{run, frame}` otherwise) and carry `vx` (ci_planner takes its start speed from
  the last valid history row; its cross-check against the recording was 0.0 in the self-test).
- `continuations --decision <analyze dir> --world crm|rigid`: for every twin decision state (default: the analysis set;
  `--all-groups` for all) six routes from the decision pose to the goal, slots in a fixed order:
  c0, c1 = `ci_planner.candidates(family='free')` (2 prior draws), c2, c3 = `candidates(family='cont_head')`,
  c4 = CEM 4x64 pick of the K1 history model H (`generalist_20260921/A_adapt/train/deploy_v1/H_deploy_s*.pt`,
  `ci_planner.plan_decision(arm='B', family='free')`), c5 = the same with `family='cont_head'`. Sampler seed =
  md5(`<g>__p1_<len>|<world>|<family>|samples`); the CEM seed is ci_planner's own (group and arm tag, as the CLI).
  Every route is re-checked (contract, validator, start within 1 m of the pose, end within 0.5 m of the goal, the
  branch collectors' tolerances). Duplicates by route content sha256 keep their row with `run: false, ref_id`.
  Rows `<g>__p1_<len>__c<slot>`:
  - CRM: case / route (= the approach route, i.e. the prefix) relative to CRM_ROOT as in pass 1, `extra ['--mode',
    'branch', '--branch-frame', F, '--branch-route', /work1/.../crm_f104_20260916/crm_improve/cont_twin_L<len>/routes_crm/<id>.json,
    '--horizon-s', '120']` (absolute branch path: K1's relative paths made its first pass-2 launch fail, K1 LOG 01:07);
    launched with `CRM_COLLECTOR=$G2/source/scripts/crm_collect_ext.py`.
  - Rigid: absolute case / route, arena f104, shard md5(group) % 6, mode `branch`, `extra ['--branch-frame', F,
    '--branch-route', /work1/.../crm_improve_20260922/cont_twin_L<len>/routes_rigid/<id>.json, '--horizon-s', '120']`.
  - Anchors file `anchors_<world>_L<len>.json` in the ga_branch_dataset format: episode `<g>__p1_<len>`, world, group,
    split (twin file), cls `a5_<len>`, F, pose_F, vx_F, remaining_m, plus state_F, goal, the six slot ids and hashes.
  - Asserts: no suite group (ids, groups, episodes; patterns = ga_build_mixed.BLACKLIST, checked equal), twin split on
    every row and anchor, unique ids / seeds / episodes, every run row has its route file with the right hash, the
    pass-1 run drove exactly the approach route (its `command_reference.npz` must be synced; the stage refuses
    otherwise). Outputs in `A/cont_twin_L<len>/`: routes, tasks, anchors, report, `SHIP_cont_<world>_L<len>.txt`.
- `merge --inputs ... --out-file X`: one task (or anchors) file per world from several lengths (the one-file-per-world
  queue-cap convention); asserts unique ids / seeds, one world, no suite group (unless `--allow-suite-rows`, meant for
  the pass-1 evaluation drives only), twin split.

## Commands run (all local, CPU except the continuation self-test, which used < 0.2 GB of GPU memory)

```bash
cd /home/harry/NeDM-traverse_mppi
PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts OMP_NUM_THREADS=6
$PY scripts/ci_a5data.py --stage approach-routes --cases artifacts/traverse/generalist_20260921/A_adapt/suite/cases    # 2.4 s
$PY scripts/ci_a5data.py --stage approach-routes --cases artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases    # 3.6 s
$PY scripts/ci_a5data.py --stage pass1-tasks                       # all builds: suite L 0.5, 1; twin L 0.5, 1, 3; both worlds
$PY scripts/ci_a5data.py --stage selftest-analyze                  # SELFTEST analyze OK (A/selftest/analyze/RESULTS.json)
$PY scripts/ci_a5data.py --stage selftest-continuations            # SELFTEST continuations OK (A/selftest/continuations/RESULTS.json), 24 s
```

Later, once pass 1 is synced back (examples; the length and the run dirs are the orchestrator's):

```bash
A=artifacts/traverse/crm_improve_20260922/a5data
# (sync back first: the rsync lines at the end of $A/SHIP_pass1.txt; command_reference.npz must come along)
$PY scripts/ci_a5data.py --stage analyze --runs $A/pass1_twin_crm/runs $A/pass1_twin_rigid/runs --worlds crm rigid --length 0.5 \
    --approach-dir $A/approach_twin --out $A/decision_twin_L0p5
$PY scripts/ci_a5data.py --stage continuations --decision $A/decision_twin_L0p5 --world crm      # then --world rigid
$PY scripts/ci_a5data.py --stage merge --inputs $A/cont_twin_L0p5/tasks_cont_twin_crm_L0p5.json $A/cont_twin_L3/tasks_cont_twin_crm_L3.json \
    --out-file $A/tasks_cont_twin_crm.json
# suite closed loop (S2): analyze the suite runs the same way (--out $A/decision_suite_L0p5), then
# ci_planner.py ... --poses $A/decision_suite_L0p5/poses_crm.json (poses_crm_masked.json for an all-masked arm)
```

## Results

Approach routes.

| set | groups | straight chosen | route_00 differs from the straight line | invalid | max 12 m grade > 12 deg | > 17 deg | median max grade | route length min / median / max |
|---|---|---|---|---|---|---|---|---|
| suite | 800 | 800 (all ties) | 0 (largest lateral 1.7e-14 m, start yaw 2.5e-14 deg) | 0 / 0 | 301 (37.6 %) | 196 (24.5 %) | 9.65 deg | 28.0 / 42.1 / 83.7 m |
| twin | 1,200 | 1,200 (all ties) | 0 (largest lateral 1.7e-14 m, start yaw 2.5e-14 deg) | 0 / 0 | 418 (34.8 %) | 285 (23.8 %) | 9.10 deg | 28.1 / 43.2 / 81.4 m |

All 800 suite routes have the same content hash as K1's `A_adapt/a5/approach/<g>.json` (the routes K1 drove for its 3 s
pass 1), so the new 0.5 / 1 s drives start exactly like the K1 3 s drives. The torch-free validator copies agreed on
all 4,000 checks; the frozen collector's reader accepted all 2,000 routes. Twin split 1,089 train / 56 val / 55 test,
equal to the case files. As in K1, the lower-grade rule cannot choose anything (both candidates coincide in every group).

Case files are referenced at their existing cluster copies (not copied under the new roots): the sha256 of all 2,000
local case files equals the four cluster copies (`C/generalist/suite/cases`, `generalist_20260921/a5/cases`,
`C/cases/night2`, `fdm_f104_50h_20260909/cases_night2_v1`), checked with `sha256sum` over ssh today; every row carries
`case_sha256` and the shipping file has a cluster-side check of it.

Pass-1 task files (`A/tasks/`, sha256 prefixes):

| file | rows | sha256 |
|---|---|---|
| tasks_pass1_suite_crm_L0p5 / _L1 / merged | 800 / 800 / 1,600 | cf70ba93 / a84393eb / 6d2aaf8a |
| tasks_pass1_suite_rigid_L0p5 / _L1 / merged | 800 / 800 / 1,600 | 4b2c71fe / 4565c73f / 03b10ac8 |
| tasks_pass1_twin_crm_L0p5 / _L1 / _L3 / merged | 1,200 each / 3,600 | 2996b424 / f2d48b60 / 6e62e094 / 554ac199 |
| tasks_pass1_twin_rigid_L0p5 / _L1 / _L3 / merged | 1,200 each / 3,600 | f9c17324 / 2572b7d2 / 71d4d861 / f9ad7ea7 |

Rigid shard sizes: suite merged 250 / 314 / 284 / 264 / 254 / 234, twin merged 597 / 486 / 666 / 570 / 672 / 609.
The cluster verification snippet in `SHIP_pass1.txt` (paths exist, case sha256, route content hash recomputed in plain
python) was run on a local mirror of the CRM layout: `verify: bad = 0` on 5,200 rows, and it reports all three rows of
a group whose route was edited (`bad = 3`).

Analyze self-test (`A/selftest/analyze/RESULTS.json`, all checks true). Stand-ins: 5 K1 3 s pass-1 recordings, every
160th of the sorted 800 (f104_crm_eval_group_0000, _0160, f104_pair_group_0120, _0280, _0440), both worlds.
- At L = 3 s the new stage reproduces K1's own analysis exactly: pose, vx and CRM sinkage differences 0.0, the same
  moving flags, and the 40-step windows equal K1's `hist_crm/`, `hist_rigid/` files (difference 0.0, 10/10); ga_planner's
  `load_history` reads them; progress along the route + remaining length = route length.
- At L = 0.5 and 1 s: copies cut at F (F rows + terminal state = row F) give the same pose and vx as reading row F of the
  full recording (difference 0.0), the window equals `ga_planner.history_from_trajectory(full, F)` and
  `ga_branch_dataset.prefix_hist` (difference 0.0, mask equal, 10 / 20 valid steps of 40), and the window with T = F
  equals ga_planner's too. CRM vx at frame 10: 0.77-0.88 m/s; at frame 20: 1.13-1.75 m/s.
- Sinkage of an F-row recording is read at crm_extra row F-1 (the last one), as K1 did; against row F the difference
  was up to 0.006 m at 0.5 s and 0.026 m at 1 s (the vehicle is still accelerating).

Continuation self-test (`A/selftest/continuations/RESULTS.json`, all checks true; real ci_planner and the K1 H ensemble).
- The suite guard fires on the five suite decision states (the normal path refuses them); the self-test then runs them
  with the guard lifted into `selftest/` only (marker file `SELFTEST_OUTPUT_DO_NOT_SHIP.txt`, no shipping file).
- Both worlds: 6 rows per decision (30 each, no duplicates, no invalid route, no sampler failure); every route re-read
  from disk passes the collector reader, the contract, the validator and the branch tolerances (start error 0.0 m);
  cont_head routes start exactly at the vehicle speed and within 20 deg of its heading; the free samples ask for speed
  steps of -2.2 / -1.2 / +0.4 m/s (p10 / p50 / p90, CRM) and the free CEM picks +0.6 / +1.4 / +2.6 m/s, so the six
  slots span the speed-step range the plan wants covered.
- The free CEM pick from each K1 decision state equals K1's A5 H pick for that group (route hash, 5/5 per world), and
  the ci_planner command line fed with this stage's `poses_crm.json` picks exactly the cont_head route of slot 5 (5/5,
  start speed from the history window, cross-check 0.0).
- The anchors match K1's pass-2 branch drives from the same state (pose, vx and the collector's own prefix window,
  differences 0.0 in both worlds), and ga_branch_dataset accepts them: a K1 pass-2 drive renamed as slot c4 of our anchor
  is labelled with cls a5_3, anchor frame 60, route hash checked, pose difference 0.0 (the CRM copy needed a zero
  `crm_extra.npz`, format test only).
- Twin path smoke (the real path, guard on): six K1 A4 CRM prefix replays of twin episodes (4 train, 1 val, 1 test, cut at
  frame 40, run as a 2 s "approach") -> 36 rows, splits kept, anchor pose / vx equal to the A4 recordings (0.0), CRM case
  and prefix paths as in pass 1; because those replays drove designed routes, the prefix check flags all 6 and the real
  path refuses to write tasks for them, as intended.

Costs for the orchestrator (measured on K1 A5): CRM pass 1 of 800 suite groups took 1.65 GPU wall-hours in total (median
7.4 s per drive, set-up dominated), so the 1,600 suite and 3,600 twin pass-1 drives are about 3 and 7.5 GPU-hours. CRM
pass 2 of K1 A5 (4,652 branch drives) took 77.9 GPU wall-hours (median 56 s, 24.0 simulated hours). Scaled, the CRM
continuations of all 1,200 twin groups are up to 7,200 drives, about 120 GPU wall-hours or about 15-18 billed node-hours
per approach length (0.125-0.15 billed per MI350X GPU-hour; night 1's 0.4 billed per simulated hour gives the same),
so all three lengths would take about half of the 100-hour budget. Rigid is cheap (6 shards on mi2104x). Planning the
continuations locally: 0.58 s per CEM plan, two per decision, about 23 minutes per world and length on the 5090.

## Where things go on the cluster (from the SHIP files; nothing was shipped)

- CRM: `C/crm_improve/pass1_<set>/approach/`, `C/crm_improve/pass1_<set>/tasks_pass1_<set>_crm*.json`, out
  `C/crm_improve/pass1_<set>/out`; continuations `C/crm_improve/cont_twin_L<len>/{routes_crm, tasks, anchors, out}`.
- Rigid: `G2/pass1_<set>/approach/`, `G2/tasks/tasks_pass1_<set>_rigid*.json`, out `G2/rigid_pass1_<set>`; continuations
  `G2/cont_twin_L<len>/routes_rigid`, `G2/tasks/tasks_cont_twin_rigid_L<len>.json`, out `G2/rigid_cont_twin_L<len>`.
- Pass 1 in CRM is native mode, so it runs with the unmodified collector (as K1); the branch drives need
  `crm_collect_ext.py` / `gen_collect_ext.py` in `G2/source` (not shipped by this module).

## Open issues and caveats

- The moving threshold for short approaches is my choice, not the plan's: `--vx-min` defaults to 1.0 m/s at frame 60
  (K1) but 0.3 m/s below, because in K1's CRM pass-1 recordings 89 % of groups are below 1 m/s at frame 10 (median
  0.89 m/s) and 2 % at frame 20; with 1.0 m/s the 0.5 s analysis set would lose most soil groups. Override if wanted.
- Continuations use the cross-world analysis set by default (groups moving in both worlds, K1's pairing convention);
  `--all-groups` takes every decision state, including stalled ones whose cont routes would start near 0 m/s.
- The K1 H ensemble was trained on full 40-step windows (plus fully masked ones); at frames 10 and 20 its CEM picks see
  windows with 10 or 20 valid steps. Those picks are proposals for data collection only (labels come from Chrono).
- Rigid Chrono is deterministic per node only: the pass-2 prefix replay may land a little off the pass-1 pose on another
  node; ga_branch_dataset reports it (pose_F_vs_recorded). The 5 K1 groups checked matched exactly.
- ga_branch_dataset's discordance summary counts anchors with exactly 3 continuations (K1's design); with 6 per anchor
  only its "2 or more" figures are informative. The labels are unaffected; changing the summary is outside this module.
- Budget: continuations for all 1,200 groups at every length do not fit with the rest of the plan (above); choose the
  lengths (the plan says 3 s plus the chosen short one) or a group subset before shipping.
