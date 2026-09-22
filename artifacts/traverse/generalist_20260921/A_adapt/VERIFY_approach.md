# Verification of `scripts/ga_approach.py` (plan A5, pass 1)

2026-09-21, adversarial verifier. Everything below ran on the CPU (`CUDA_VISIBLE_DEVICES=`), nothing was submitted to the
cluster (one read-only `ssh amd ls`), no Chrono run, no existing repo file edited. Scratch outputs under `/tmp/verify_a5`.

Verdict: **pass with issues**. The module does what the note says and every number in the report reproduces; the two
issues are about how the result is to be used (which partitions pass 1 must run on, and how missing runs are counted),
not about the routes, task rows or the analysis code.

## What was re-run and compared

| check | result |
|---|---|
| `--stage routes --a5 /tmp/verify_a5` (3.6 s) | all 800 route files byte-identical to `a5/approach/`; `approach_index.json` identical apart from `created` / `wall_s` |
| `--stage tasks-crm --a5 /tmp/verify_a5` | `tasks_pass1_crm.json` identical (800 rows) |
| `--stage selftest-analyze --a5 /tmp/verify_a5` | `RESULTS.json` identical apart from `created`, `all_ok` true |
| independent grade recomputation (own code, v2 grid, 12 m straight at the start yaw, 0.5 m steps) | fresh 221 / 143 / 68, reused 80 / 53 / 31 (= plan review finding 10), all 800: 301 / 196 / 99, histogram 96 / 323 / 80 / 105 / 156 / 40 — exactly the module's numbers |
| start yaw vs start-goal heading, `route_00` vs the straight line | 2.5e-14 deg and 1.7e-14 m over the 800 groups: the two candidates really coincide in every group |
| frozen collector reader (`traverse_fdm_rgbd_diverse_chrono.read_route`) + original `check_reference_contract` + the collector's 0.25 m start/goal rule on all 800 written routes | 800 ok, 0 bad; every speed 3.0 m/s; largest waypoint step 0.500 m; 58–169 waypoints |
| task rows | keys as in the A0 suite rows plus provenance; `episode_seed == md5(id)[:8]` for all 800 (ga_suite convention); 800 unique ids and seeds; tier 0–799; `sha256` equals the geometry hash of the route file (0 mismatches); `extra == ['--horizon-s', '3']` everywhere |
| worker semantics | `crm_worker.py:92-94` builds `... '--horizon-s', '120'] + extra`, so the row's `3` is the last value and wins in argparse; `crm_launch.sh` usage `<tasks> <out> <config rel. to C> [hours]` matches the SHIP line |
| cluster preconditions (read-only) | `C/generalist/suite/cases` has 800 case files; `C/configs/crm_main.json`, `C/source/scripts/crm_launch.sh`, `crm_collect.py` exist; `C/generalist/a5` does not exist yet (correct: nothing was shipped) |

## Conventions against PLAN.md

- Route contract: collector route format (lists), stations from the geometry, constant heading; the straight line at a
  constant 3 m/s passes `validate_reference` with `gen_planner.CFG` anchored at the layout pose and `gc_control`'s
  torch-free copies give the same verdict for all 1,600 candidate checks (0 disagreements).
- History window: `HIST_STATE_COLS = [0-6, 11-14, 15]` matches the plan's deployable columns and the 17-field preset
  (`vel_body_x, vel_body_y, roll, pitch, roll_rate, ang_vel_body_y, yaw_rate | 4 tyre Fz | 4 spindle omegas | engine speed
  | torque`). `history_window` row t = `state[F-39+t]` paired with `action[F-40+t]`, valid iff the action index is >= 0,
  which is the plan's rule and is line-for-line `ga_planner.history_from_trajectory`.
- Frame 60 on a 60-row recording: the collector records row k at substep 0 of interval k and measures
  `terminal_state` / `terminal_pose` after the loop at the same simulation time a row 60 would have (no `Advance`
  in between, same chassis reference frame `GetFrameRefToAbs`, same yaw formula `GetCardanAnglesZYX().z`, and the state
  preset contains no driver-input column), so the explicit `{pose: terminal_pose, history: npz}` entry is physically the
  row-60 entry. The frozen rigid collector uses the same convention (`traverse_fdm_rgbd_diverse_chrono.py:228, 307, 328-332`),
  so the same path serves the rigid pass-1 runs. I ran `ga_planner.decision_for` on the four explicit entries from the
  scratch self-test and on the plain `{run, frame: 60}` entries of the full recordings: pose, history, mask (40/40 valid)
  and the base route are identical (difference 0.0) and the sources read `poses_file` / `npz`.
- Grade on the v2 grid: `gb_crop.sample_height` (cell-centre convention, Chrono-frame grid) at the route's own xy, which is
  the Chrono frame the collector drives in; 24 segments of 0.5 m over the first 12 m (9 m are driven in 3 s).
- Sinkage (my strongest independent check): the per-wheel reconstruction (chassis pose + quaternion + spindle offsets,
  tyre radius minus spindle height above the TerrainMap) was compared with the collector's own
  `max_wheel_sinkage_below_bmp_m` on 60 recorded CRM episodes: median difference -0.005 m, 90th percentile 0.05 m (the
  larger differences are the breakthrough-terminated episodes, where the collector's maximum includes the final
  measurement that is not among the rows). The chassis-centre proxy is off by 0.43 m (median) and a y-mirrored offset
  table by 0.21 m, so the lateral sign and the offsets are right. The offsets match the Chrono data files
  (suspension at +-1.688965 m, spindle COM (-0.040 / +0.036, 0.910, -0.026)). The `pose` rows are the chassis reference
  frame, the same frame the offsets are defined in.
- Splits / blacklist: pass 1 is a closed-loop drive on the frozen 800-group suite (which every builder blacklists); it
  builds no training rows, and the case files carry their split. Anchor classes do not apply to A5; the moving criterion is
  the plan's (vx > 1 m/s, CRM sinkage increase < 0.1 m).
- Cost: from 300 recorded outcomes, a CRM episode costs ~3 s of terrain build plus ~2.8 s of wall time per simulated
  second, so a 3.8 s pass-1 episode is ~14 s and the 800 rows ~3 GPU-hours. Within the A5 budget.

## Issues

1. **Partitions for pass 1 (plan-level, affects the SHIP file's launch line).** The suggested launch
   `crm_launch.sh ... 2` submits to every GPU partition (MI210, MI250, MI300, MI350). Pass 2 of A5 re-drives this
   approach for 60 frames before branching, i.e. the same two-pass replay as A4, and PLAN A4 (amended after the
   determinism check) restricts CRM two-pass designs to the MI350X partitions (mi3501x / mi3508x) because replays are
   reproducible only per node type. If pass 1 runs on a mixed set of partitions, the pass-2 state at frame 60 can differ
   from the recorded one the arms planned from. Pass 1 should be launched on mi3501x / mi3508x only (or pass 2 pinned to
   the node type of each pass-1 run, with the node type recorded). The line is a comment and was not run, so nothing is
   broken yet.
2. **Missing runs are not counted as excluded.** `stage_analyze` builds its universe from the run directories that
   contain a `trajectory.npz`. A run that failed (e.g. `collection_failure.json` with "Invalid settled launch on CRM soil")
   or a group with no run directory at all is absent from `n_groups`, `n_excluded` and the excluded list (verified with a
   probe directory: the failed group appears nowhere in `pass1_state.json`). The plan asks for the excluded count; the
   universe should be the 800 suite groups (or the tasks file) with reasons `missing_run` / `collection_failure`.
3. **Sinkage definition differs from A4's pre-screen** (`ga_branch_anchors.py`: chassis-centre proxy, mean over wheels,
   drop over the last 20 frames < 0.1 m). The module's per-wheel, spindle-xy definition over 60 frames is the better one
   (see the check above); the two should be reconciled or the difference stated in the report. Already flagged by the
   module.
4. Notes, no action: the "lower-grade heading" rule is a no-op on this suite because the case generator points the
   start yaw at the goal, so `route_00` is the straight line (the module says so); `--horizon-s 3.05` would make every
   entry the plain form but 3 s is what the plan says and the explicit-entry path is verified; the 'terminal' sinkage uses
   frame 59 (crm_extra has no terminal row), 8 mm on the stand-ins.

## Commands run

```bash
cd ~/NeDM-traverse_mppi
CUDA_VISIBLE_DEVICES= PYTHONPATH=src:scripts python scripts/ga_approach.py --stage routes --a5 /tmp/verify_a5
CUDA_VISIBLE_DEVICES= PYTHONPATH=src:scripts python scripts/ga_approach.py --stage tasks-crm --a5 /tmp/verify_a5
CUDA_VISIBLE_DEVICES= PYTHONPATH=src:scripts python scripts/ga_approach.py --stage selftest-analyze --a5 /tmp/verify_a5
CUDA_VISIBLE_DEVICES= PYTHONPATH=src:scripts python scripts/ga_approach.py --stage analyze --runs /tmp/verify_a5/probe/runs --worlds crm --out /tmp/verify_a5/probe/out
# plus inline python: byte comparison of the 800 routes / index / tasks / RESULTS, independent grade recomputation,
# frozen read_route + contract + 0.25 m rule on the 800 routes, sinkage reconstruction vs outcome.json on 60 episodes,
# ga_planner.decision_for on explicit vs plain pose entries, cost estimate from 300 outcomes; ssh amd (ls only).
```
