# Module note: `scripts/crm_collect_ext.py` (CRM collector with external-control and branch modes)

Written 2026-09-21 (first attempt ~17:50, interrupted by the usage limit after one smoke; resumed ~22:30: the file
was reviewed against the plan, the timing scout map, `scripts/gc_control.py` as revised in the fix round, and the
rigid sibling's verification findings, three changes were made (below), and every smoke was re-run with the final
file). PLAN stages B3 and A4.

Changes made in the resumed attempt: (1) policy mode reads the engine speed for the observation from the
transmission's integrated shaft state instead of `engine.GetMotorSpeed()`: before the frame's `Synchronize` the
SHAFTS engine's own value is the one imposed at the previous substep (one-substep lag, the rigid sibling's
verification finding P1); with the transmission's value the 12 observable columns of the pre-frame capture equal
the recorded `state[k]` exactly (asserted per run below); (2) the prefix follower object is kept alive across the
branch swap (no destructor side effects mid-run; it has no physics effect either way); (3) `--branch-frame` must
lie inside the horizon (rejected before Chrono loads).

## What it is

A fork of the unmodified deformable-soil collector `scripts/crm_collect.py`. That file is imported, never edited:
the soil build (`build_crm`), the tyre-force fields and the config merge come from it; the episode loop and the
driver are forked. The file name keeps `crm_collect` in it so `crm_worker.py` forwards `--crm-config` (which is now
asserted: the physics step must come from the campaign config, PLAN R2).

Five modes (`--mode`):

- `native` (default): the unmodified collector's behaviour. The physics and driver statements are the same
  statements in the same order; the on-disk files are byte-identical except for wall-clock fields and three
  provenance keys appended at the end of `outcome.json` (`mode`, `collector_sha256`, `physics_dt_s`; the brief asks
  for them), plus `collection_request.json` / `episode_complete.json`, which hash this file. Self-test S1 below.
- `branch` (`--branch-frame F --branch-route <json>`): the case route is driven by the native follower up to frame F.
  At the top of frame F, before that frame's route index and desired speed are computed, the follower is rebuilt on
  the branch route with `gc_control.make_follower(initialize=False)` (no body added mid-run; the constructor resets
  the controllers), `wp` is set to 0, the waypoint/speed arrays and the goal are swapped (the goal becomes the branch
  route's last waypoint, which must lie within `--branch-goal-tol-m` 0.5 m of the case goal), and `previous_steer`
  is kept so the frozen per-substep 2/s steering clamp makes the first branch substep continuous. The branch route
  must start within `--branch-start-tol-m` (1.0 m) of the vehicle at F, else the run fails (wrong route for the
  case). Written: `outcome.json['branch']` (branch_frame, branch_time_s, reached, branch_route, branch_route_sha256
  = sha256 of the route file, branch_route_content_sha256 = `gc_control.route_sha256` of the arrays, the route meta,
  prefix_route_sha256, branch_pose, branch_start_offset_m, previous_steer_at_branch, prefix_last_action,
  first_branch_action, steer_step_at_branch, vx_at_branch_mps, branch_route_v0_mps, and the prefix history window
  `hist` (40 x 15) + `hmask` (40) + `hist_cols`); `trajectory.npz` gets `branch_frame`, `branch_reached`,
  `branch_pose`, `branch_hist`, `branch_hmask`, `branch_hist_cols`; `command_reference.npz` gets `branch_frame`,
  `branch_reached` and the four `branch_reference_*` arrays; the route file is copied to `branch_reference.json`.
  History window convention = the mixed dataset builder's (`ga_build_mixed.py`): row t holds
  `state[F-39+t]` on the 12 observable columns (0-6, 11-14, 15) and `action[F-40+t]`; row t is valid iff the action
  frame `F-40+t >= 0` (so F = 0 is all-masked, F = 10 has 10 valid rows, F >= 40 all 40). If the episode ends before
  F the block says `reached: false` and the swap never happens.
- `pid_held`: external control with the shadow follower's own command. At the top of frame k the follower's
  `GetInputs()` (the output of its last `Advance`, i.e. the final substep of frame k-1, exactly what the native
  collector records as `action[k]` before its clamp) is clipped once by `gc_control.hold_clip` (steering within
  +-0.1 of the previous held steering, throttle/brake in [0, 1]) and the identical triple is written into a
  `veh.DriverInputs` at every one of the 50 physics substeps. The frozen per-substep clamp is replaced by that one
  clip (the verifier's note 4 on gc_control). The follower keeps running as a shadow (`Synchronize`/`Advance` every
  substep, `SetDesiredSpeed` every frame), so the settle, the parking rule and the desired-speed record are as native.
  The recorded `action[k]` is the held triple.
- `pid_perturbed`: as `pid_held` with `gc_control.OUPerturb` applied to the shadow output before the clip, seeded
  from `--episode-seed` (required). Defaults: tau 0.5 s, steering sd 0.15, throttle sd 0.25, bound 1 sd, brake taps
  0.5-1 s at level 0.3-1.0 with per-frame start probability `OUPerturb.brake_p_for_rate(1.4 / 20 s)` = 0.0037
  (1.4 tap starts per 20 s of driving, the plan's >= 1.33 onsets per episode; the gc_control default 0.05 gives
  46 % braked time and is NOT used). All parameters are CLI flags (`--perturb-*`) and are recorded in
  `outcome.json['control']['perturb']` together with the tap count and braked fraction; `trajectory.npz` gets
  `shadow_action` (n, 3) and `perturbation` (n, 4: d_steer, d_throttle, tap_active, tap_level).
- `policy`: as `pid_held` with a numpy actor (`--actor <npz>`, `gc_control.NumpyActor`; no torch in the process)
  evaluated on the `gc_control.PolicyObs` observation rebuilt from the actor's `obs_layout` meta. The pose/state for
  the observation are captured at the top of the frame before `Synchronize` (positions, velocities, rates and
  spindle speeds are final there; the engine speed, column 15, is read from `transmission.GetOutputMotorshaftSpeed()`
  because `engine.GetMotorSpeed()` lags one substep before `Synchronize`; the maximum difference of the 12
  observable columns against the recorded substep-0 capture is written to
  `outcome.json['control']['actor']['pre_capture_vs_recorded_observable_max_abs_diff']` and is 0 in the smoke);
  frame 0 is padded with that rest state and the settle action (0, 0, 1); after every frame `(state[k], action[k])`
  is pushed.

Common to the three external-control modes: `outcome.json['driver']['control']` = the mode;
`outcome.json['control']` holds the held-minus-shadow statistics, the largest per-frame steering step, the number
of frames at the clip bound and the number of frames with throttle and brake both positive (0 by construction).
Every mode: `outcome.json` ends with `mode`, `collector_sha256` (this file) and `physics_dt_s`;
`collection_request.json` has `collector: crm_collect_ext`, `mode` and an `ext` block with the mode arguments, the
route/actor hashes, and the sha256 of `crm_collect.py` and `gc_control.py`.

`--substep-log <npz>` (any mode) records the inputs passed to `hmmwv.Synchronize` at every physics substep
(`frame`, `sub`, `inputs (N, 3)`, `substeps`, `dt_s`, `mode`; the settle is included with negative frame numbers).
It only appends to a list; it does not touch the physics.

### The 40 s any-throttle near-stop rule

|vx| < 0.3 m/s (body-frame vx at substep 0) on 800 consecutive non-parked frames, evaluated after every native check
of the frame (soil breakthrough, rollover, goal, the native StopPolicy) -> status `prolonged_blockage_terminated`
with a stop event `{"kind": ..., "rule": "near_stop_any_throttle", "near_stop_s": ...}`. It is active by default in
`pid_held`, `pid_perturbed` and `policy` only (`--no-near-stop` disables it there). It is NOT applied to `native`
or `branch` by default, because it can change native outcomes: it was replayed offline over all 15,235 recorded
`collect_v1` episodes (`selftest/crm/near_stop_offline.py`, 7 s): 9 episodes would have ended earlier
(all nine are 120 s timeouts in which the vehicle sat below 0.3 m/s for 40 s without the native bounded-displacement
rule confirming; the new rule would have stopped them at 52-64 s), 0 of the 1,985 native prolonged-blockage
episodes are touched (the native rule fires first, from 34 s), 0 goal-reached / breakthrough / rollover episodes
are touched. The longest near-stop run is 5.6 s at the median and 28.9 s at the 99th percentile; 40 episodes have a
run of >= 30 s. So "it cannot change any native outcome" is false and the rule is external-control-only;
`--near-stop-all-modes` turns it on for `native`/`branch` (for a B0 arm that wants identical stop rules; the run is
then documented as not byte-identical and `outcome.json['near_stop_rule']` records it). Details in
`selftest/crm/near_stop_offline.json` (the nine episode ids are listed under `changed`).

## How to run

    cd /home/harry/NeDM-traverse_mppi
    PY=/home/harry/miniconda3/envs/nedm/bin/python
    CFG=artifacts/traverse/crm_f104_v1/configs/crm_main.json
    COMMON="--source-root . --chrono-data /home/harry/chrono/data --crm-config $CFG --case <case.json> --route <route.json> --out <dir>"
    # native (byte-identical to scripts/crm_collect.py)
    PYTHONPATH=src:scripts flock /tmp/luffy_crm.lock $PY -P -u scripts/crm_collect_ext.py $COMMON --horizon-s 120 --episode-seed <id>
    # held follower / perturbed follower / policy, with the substep log
    ... --mode pid_held --substep-log <npz>
    ... --mode pid_perturbed --episode-seed <id> [--perturb-brake-p 0.0037 ...]
    ... --mode policy --actor <actor.npz>
    # branch at frame F into a continuation route (gc_control.sample_continuations -> route_to_json)
    ... --mode branch --branch-frame F --branch-route <branch.json>

On the cluster the worker launches it unchanged: `CRM_COLLECTOR=$G/source/scripts/crm_collect_ext.py`, per-task
`extra` = the mode arguments above (absolute paths), `CRM_CONFIG` as before (`crm_worker.py` forwards
`--crm-config` because the collector path contains `crm_collect`).

## Self-test (all under `flock /tmp/luffy_crm.lock`, horizon 12 s, on the RTX 5090)

Inputs: `selftest/crm/inputs/` = the recorded `collect_v1` episode `f104_v2_group_0000_op_03` (train group; the
route JSON is rebuilt from its `command_reference.npz` because the run directories do not keep `reference.json`;
recorded: goal reached at 20.5 s, physics step 0.001 s). Runner: `selftest/crm/run_one.sh NAME base|ext ARGS`
(= `flock /tmp/luffy_crm.lock bash run_locked.sh`, which samples `nvidia-smi` every second while the collector
runs, records the collector's PID so its own GPU peak can be separated from the other processes on the shared
GPU, and appends one line to `runs/summary.txt`). Chain: `selftest/crm/chain2.sh` (the resumed attempt; the
unmodified-collector reference run `base_native` was launched separately just before it and the lock serialised
them); branch inputs: `selftest/crm/make_branch.py 60 ext_native_1` (continuations sampled with `v0` = the
vehicle's speed at frame 60 and the 15 degree start-heading acceptance of the revised sampler); checks:
`selftest/crm/check.py` -> `selftest/crm/selftest_report.json`. Argument gates without Chrono (all rejected
before any import of pychrono and without creating the output directory): missing `--crm-config`, `pid_perturbed`
without `--episode-seed`, `--branch-frame` beyond the horizon, `policy` without `--actor`, `--actor` outside
policy mode; `prefix_history` checked at F = 0, 10, 39, 40, 60 (valid rows = min(F, 40), masked rows zero, last
row = state[F] + action[F-1], first valid row = state[F-39+t0] + action[F-40+t0]); importing the module leaves
torch unloaded.

RESULTS_PLACEHOLDER

## Known limits

- The three external-control modes and the branch mode are exclusive: a policy cannot yet take over after a
  recorded prefix (`PolicyObs.seed_history` exists in gc_control for that, but no plan stage needs it now).
- The branch swap gives the frozen follower's start-up transient on the continuation: the new driver's first
  `GetInputs` is (0, 0, 0) before its first `Advance` (one coasting substep, steering held by the clamp), and if the
  continuation's first speed is below the vehicle speed the follower brakes hard for a while (gc_control verification
  note 1); the collector records `vx_at_branch_mps` and `branch_route_v0_mps` so the label script can see it.
- The near-stop rule counts |vx| on the recorded substep-0 state and excludes parked frames only; a policy that
  creeps at 0.3-0.5 m/s without progress runs to the native rule (needs throttle > 0.3) or the horizon.
- `policy` mode calls `capture_row` twice per frame (pre-capture for the observation, substep-0 capture for the
  record); the extra call costs a few hundred microseconds per frame, nothing against the 50 SPH steps.
- Local timing: with the A2 training lanes sharing the GPU the smokes ran far below the collector's normal
  0.4-0.5x real time (see the run summaries); determinism and byte-identity are unaffected by contention.
