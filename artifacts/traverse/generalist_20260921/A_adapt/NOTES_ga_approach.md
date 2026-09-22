# Module note: `scripts/ga_approach.py` (plan A5, pass 1)

2026-09-21. One new file, `scripts/ga_approach.py`; no existing repo file was edited. Outputs under
`A_adapt/a5/` (`approach/<group>.json` x 800, `approach_index.json`, `tasks_pass1_crm.json`, `SHIP_pass1_crm.txt`,
`selftest/analyze/`). Nothing was submitted to the cluster and no Chrono run was made; everything ran on the CPU
(`CUDA_VISIBLE_DEVICES=` set).

## What it does

Pass 1 of the established-history closed loop drives every suite group for 3 s (60 frames at 50 ms) along one common
approach route, so that every planning arm later starts from the same recorded pose and history at frame 60.

- `--stage routes`: two candidates per group, (1) the straight start-goal line at a constant 3 m/s (a waypoint every
  0.5 m, the last one exactly the goal, constant heading) and (2) the group's `route_00` with all speeds set to 3 m/s.
  Both are checked with the reference contract (`nedm.traverse.fdm_diverse_planner.check_reference_contract`), the
  planner's validator (`gen_planner.safe_validate` with `gen_planner.CFG`, anchored at the layout pose) and the
  collectors' rule that the route starts within 0.25 m of the start and ends within 0.25 m of the goal; the torch-free
  copies in `gc_control` gave the same verdict for all 1,600 checks. The grade of each candidate is measured on the
  Chrono-frame v2 grid (`crm_f104_v1/grids/arena_f104_50h_v1/grid.npz`, `gb_crop.sample_height`): heights at 0.5 m
  steps along the first 12 m, grade = atan(height difference / 0.5 m), 24 segments. The candidate with the lower mean
  |grade| wins, the straight line wins ties, `route_00` is the fallback when the straight line fails validation.
- `--stage tasks-crm`: the 800 CRM task rows `{id: <group>__pass1, group, case: generalist/suite/cases/<g>.json,
  route: generalist/a5/approach/<g>.json, run: true, tier: suite index, episode_seed: md5(id)[:8], extra:
  ['--horizon-s', '3']}` (plus provenance: route sha256, stratum, approach). `crm_worker.py` puts `extra` after its
  own `--horizon-s 120`, so the 3 s wins in argparse; the unmodified collector then records exactly 60 frames. The
  rsync commands are printed and saved in `SHIP_pass1_crm.txt`; they were not run.
- `--stage analyze --runs DIR [DIR ...]`: per run dir the frame-60 state, the sinkage increase and the moving
  criterion (vx > 1 m/s and sinkage increase < 0.1 m); writes `pass1_state.json` and `poses.json` for
  `ga_planner.py --poses`. Several `--runs` dirs (one per world, `--worlds crm rigid`) give `poses_<world>.json`
  each and an analysis set = groups moving in every world; `poses.json` is the first world's file restricted to that
  set; the excluded groups are listed with their reasons.
- `--stage selftest-analyze`: the analyze stage on 5 recorded CRM episodes and on 60-row copies of them.

## Commands run

```bash
cd ~/NeDM-traverse_mppi
CUDA_VISIBLE_DEVICES= PYTHONPATH=src:scripts python scripts/ga_approach.py --stage routes            # 2.1 s
CUDA_VISIBLE_DEVICES= PYTHONPATH=src:scripts python scripts/ga_approach.py --stage tasks-crm
CUDA_VISIBLE_DEVICES= PYTHONPATH=src:scripts python scripts/ga_approach.py --stage selftest-analyze  # SELFTEST OK
# later, after the cluster drive is synced back:
# python scripts/ga_approach.py --stage analyze --runs <pass1_out>/runs [<rigid pass1 runs>] --worlds crm rigid --out A_adapt/a5/pass1
```

## Results

Approach choice. In all 800 groups the start yaw points exactly at the goal (largest difference between the start
yaw and the start-goal direction: 3e-14 degrees), so `route_00` (the frozen generator's Hermite with zero lateral
offset) IS the straight line in every group (largest lateral distance of any `route_00` waypoint from the line:
6e-15 m). The two candidates therefore have identical grades and the rule resolves every group as a tie in favour of
the straight line: 800 straight, 0 route_00, 0 validation failures on either candidate. The "lower-grade heading"
idea from the plan review cannot buy anything on this suite because there is no second heading to choose from; the
only way to lower the approach grade would be a different approach direction, which pass 1 does not have.

Grades over the first 12 m of the chosen routes (share of groups; the plan review's numbers for the 200 reused groups
are reproduced exactly):

| stratum | max |grade| > 12 deg | max |grade| > 17 deg | mean uphill > 8 deg | median max |grade| |
|---|---|---|---|---|
| reused (200) | 80 (40.0 %) — review: 80/200 | 53 (26.5 %) — review: 53/200 | 31 (15.5 %) — review: 31/200 | 10.6 deg |
| fresh (600) | 221 (36.8 %) | 143 (23.8 %) | 68 (11.3 %) | 9.2 deg |
| all (800) | 301 (37.6 %) | 196 (24.5 %) | 99 (12.4 %) | 9.7 deg |

Histogram of the 12 m max |grade| (all 800): 0-5 deg 96, 5-10 deg 323, 10-12 deg 80, 12-17 deg 105, 17-25 deg 156,
over 25 deg 40. Since CRM stalls from rest on 12-17 degree slopes (crm_f104_v1/LOG.md), the plan's expectation that
roughly a quarter to two fifths of the CRM groups reach frame 60 not moving stands; the analysis set will be the
groups moving in both worlds, and the excluded count is reported by the analyze stage.

All 800 written routes are accepted by the frozen collector's own reader (`traverse_fdm_rgbd_diverse_chrono.read_route`).

Analyze self-test (`a5/selftest/analyze/RESULTS.json`, all checks true). Stand-ins: the first recorded episode of
`f104_v2_group_0000..0004` from `crm_f104_v1/collect_v1/runs` (333-680 frames). Frame-60 states: vx 0.23 (group
0000, prolonged blockage later; excluded as not moving), 2.63, 3.26, 1.81, 2.00 m/s; sinkage increases -0.03 to
+0.02 m (max over wheels); 4/5 moving. The 60-row copies (state/pose cut at 60, `terminal_state`/`terminal_pose` =
row 60, `crm_extra` cut at 60) give the same pose (diff 0.0), the same vx (0.0), the same moving flags, and a
sinkage increase within 0.008 m (their last sinkage row is frame 59). Their explicit history windows equal
`ga_planner.history_from_trajectory(full, 60)` to 0.0 with 40/40 valid steps and are read by `ga_planner.load_history`.

## Definitions worth knowing

- Frame 60 and the 60-frame recording. The collector records state row k at substep 0 of interval k, so a 3 s
  recording has rows 0-59 and the state after 3 s is the collector's `terminal_state` / `terminal_pose` (measured
  after the loop at the same simulation time a row 60 would be). `ga_planner.py --poses {run, frame: 60}` indexes
  `pose[60]` directly and would fail on a 60-row recording, so for those the analyze stage writes the equivalent
  explicit entry `{pose: terminal_pose, history: <npz with the (40, 15) window and mask>, frame: 60, pass1_run,
  pass1_source: 'terminal'}`; the window is built the `history_from_trajectory` way from rows 21-59 plus the terminal
  state, paired with `action[20..59]`. Recordings with 61 or more rows get the plain `{run, frame: 60}` form.
  Shipping `--horizon-s 3.05` (61 frames) instead would make the plain form work everywhere; `--horizon-s` is an
  option of the tasks stage, the rows were written with 3 as the plan says.
- Sinkage increase. `crm_extra.npz` has the spindle heights but only the ground height under the chassis centre;
  on curved terrain the chassis-centre proxy is off by up to 0.2 m (the mean spindle height minus the centre ground
  height drops by 0.1-0.27 m on moving episodes that are not sinking at all). The analyze stage therefore
  reconstructs each spindle's xy from the chassis pose and quaternion plus the HMMWV spindle offsets (front
  1.649 m, rear -1.653 m, lateral +-0.910 m, from the Chrono HMMWV vehicle and double-wishbone files) and uses the
  collector's own formula, tyre radius minus (spindle z minus arena TerrainMap height at that xy), per wheel; the
  increase is frame min(60, last row) minus frame 0, and the criterion uses the maximum over the four wheels (the
  mean and the chassis-centre proxy are recorded too). On the stand-ins a moving vehicle shows -0.03 to -0.07 m per
  wheel and the later-blocked one +0.02 m at frame 60 (its per-wheel value reaches +0.31 m during the blockage), so
  the 0.1 m threshold separates the two regimes with margin.
- Speeds. The approach is a constant 3 m/s everywhere, including the last waypoint (no terminal deceleration cone),
  because only the first 3 s are driven; the validator accepts it (zero acceleration) and the collector's `at_end`
  logic would zero the desired speed near the goal anyway.

## Open issues

- The two candidates coincide on this suite, so the "lower-grade" choice is a no-op; if the approach grade turns out
  to exclude too many CRM groups, a different approach direction (for example a `route_0k` with lateral offset) would
  be needed, which is a plan decision.
- The 60-row recording versus `ga_planner --poses {run, frame: 60}`: handled by the explicit pose+history entries
  above; alternatively ship the rows with `--horizon-s 3.05` (stage option) so every entry can be the plain form.
- The rigid-world pass 1 (same routes, `gen_collect.py` from R with `--horizon-s 3`, `gen_array.sbatch`) is not
  built by this module; the analyze stage accepts its run dir as a second `--runs` entry.
- The sinkage reconstruction ignores the small planar shift of the spindles under pitch/roll (below 0.08 m of xy at
  17 degrees, well under one grid cell of slope effect) and uses the arena TerrainMap like the collector, not the v2
  grid (the two agree to 0.01 m on the stand-ins).
