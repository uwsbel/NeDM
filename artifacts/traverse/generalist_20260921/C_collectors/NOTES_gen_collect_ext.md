# Module note: rigid collector with external-control and branch modes (`scripts/gen_collect_ext.py`) and the G-side launch files

Written 2026-09-21 (evening) for PLAN B3 (external-control collectors) and A4 (rigid in-job two-pass branch collection).
Files: `scripts/gen_collect_ext.py`, `scripts/gen_runner_g.py`, `scripts/gen_array_g.sbatch`. No existing repo file was
edited; `gen_collect.py`, `gc_control.py` and the frozen loop are imported. Self-test artefacts:
`artifacts/traverse/generalist_20260921/C_collectors/selftest/rigid/` (first version), `selftest/fix1/` (fix round 1)
and `selftest/rigid_final/` (the complete suite re-run on the FINAL file after the resume, see the last section; the
numbers quoted in the "What was tested" section below are from the first version and were all reproduced there).

## What was built

### `scripts/gen_collect_ext.py`

Same command line as `gen_collect.py` (source root, case, route, out, chrono data, horizon, the three stop-policy
safeguards, `--check-only`, the optional sha arguments) plus `--mode native|branch|branch_auto|pid_perturbed|pid_held|policy`,
`--branch-frame`, `--branch-route`, `--actor`, `--substep-log`, `--local`, and the mode parameters listed below.

Everything physical is the frozen loop `traverse_fdm_rgbd_diverse_chrono.run_chrono`, loaded through
`gen_collect.import_runner`. `gen_collect.adapted_function` is insertion-only and cannot change the per-substep follower
input, so this wrapper has its own adapter, `replacement_adapter`, placed beside it:

- the three `gen_collect` insertions are kept verbatim (scene bind, launch-state gate, stop check); the stop-check
  insertion is extended by one line that runs the external-mode near-stop rule when the native rule is silent;
- three more insertions: bind the scene objects once (after the driver, tyre radii, state fields and goal exist), the
  follower swap at the top of frame F, and one external command per frame after `SetDesiredSpeed`;
- ONE replacement of the four-line block `driver.Synchronize(ts) / inputs = driver.GetInputs() / clamp / assign`: when an
  external command is set for the frame, a fresh `DriverInputs` with that exact triple is used and `previous_steer`
  becomes its steering; otherwise the original four statements run unchanged.

Every hook location is asserted by count (6 insertions, 1 replacement). The record in `collection_request.json`
carries the sha256 of the original `run_chrono` source (`f060168d...`), of the adapted source (`8cf1bd50...`), of
`gen_collect`'s own adapted source (`162c594c...`), the `mutation_scope` string `gen_ext_v1: ...`, and the sha256 of
this wrapper, of `gen_collect.py`, of `gc_control.py` and of `f104_n2_sampler.py` (the continuation route family).

Modes, all sharing the frozen settle (0.8 s, follower steering forced to 0), recording, termination and file schemas:

- `native`: no external command; output byte-identical to `gen_collect.py` on the same node (checked, S1/S2 below).
- `pid_held`: at the top of every frame k >= 0 the shadow follower's `GetInputs()` (the output of its last `Advance`
  in interval k-1, i.e. what the native loop would record as `action[k]` before its clamp) is clipped once by
  `gc_control.hold_clip` (steering within +-0.1 of the previously held steering, throttle/brake in [0, 1]) and the
  identical `DriverInputs` is written at all 25 substeps. The follower keeps `Synchronize`/`Advance`-ing as a shadow, so
  the settle, the parking logic and `SetDesiredSpeed` are unchanged.
- `pid_perturbed`: as `pid_held` with `gc_control.OUPerturb` around the shadow output before the clip. Seed:
  `--perturb-seed`, else `--episode-seed` (the CRM task-row convention); one of them is REQUIRED (a row without a
  seed is rejected before Chrono loads, so no two episodes can silently share a stream); `--brake-p` defaults to
  `OUPerturb.brake_p_for_rate(1.4 / 20 s)` = 0.0037 per frame (the plan's >= 1.33 brake onsets per episode, not the
  0.05 that gives ~46 % braked time); `--ou-tau-s 0.5 --ou-steer-sd 0.15 --ou-throttle-sd 0.25`.
- `policy`: as `pid_held` with a numpy actor (`gc_control.NumpyActor.from_npz(--actor)` and `PolicyObs.from_meta`);
  frame 0 is padded with the rest state and the settle action (0, 0, 1); the state the policy sees is captured at the
  top of the frame BEFORE `hmmwv.Synchronize` (one extra `capture_row` call): positions, velocities, angular rates,
  wheel speeds and engine speed equal the recorded `state[k]`; the simulator-only tyre-force and torque columns lag one
  substep. No torch is imported.
- `branch`: the frozen follower drives the initial route to frame F; at the top of frame F a new
  `ChPathFollowerDriver` on `--branch-route` is built by `gc_control.make_follower(initialize=False)` (same decimated
  points, Bezier, gains and look-ahead as `make_driver`; the constructor already `Reset()`s the controllers; no body is
  added mid-run); the loop's `xy`/`speed` switch to the branch route and `wp` restarts at 0; the case goal is kept (the
  branch route must end within 0.25 m of it) and the branch route must start within `--branch-start-tol-m` (1 m) of the
  vehicle. Steering continuity is the frozen per-substep clamp: at substep 0 of frame F the new driver returns
  (0, 0, 0), so `action[F]` = (previous steering -+ 0.004, 0, 0) for one substep, then the new PID output. A branch
  drive whose prefix ends before F (goal, rollover, bounds, stop rule) is a skip (`skipped.json`,
  `episode_complete.json {skipped: true}`, exit 0), not a failure.
- `branch_auto`: the rigid two-pass on one node, see below.

External-mode stop rule (the CRM collector's rule, same status): 800 consecutive recorded intervals (40 s) with
|vx| < 0.3 m/s and not parked, regardless of throttle -> status `prolonged_blockage_terminated`, with an event
`{"kind": ..., "rule": "near_stop_any_throttle"}` in `outcome.json` (`ext_stop_events`, `ext.events`) and
`f104_episode.json` (`stop_events`). The native rule (throttle > 0.3 in every interval, earliest 34 s) is checked first
on every frame, so for the follower it fires first. Never active in `native`/`branch`.

Outputs beyond the frozen set (only in non-native modes; native writes exactly what `gen_collect.py` writes):

- `trajectory.npz` is re-saved atomically with the original arrays plus `ext_mode`, and for `branch`
  `branch_frame`, `branch_pose` (x, y, yaw at the top of frame F), `branch_route_sha256` (the content hash
  `gc_control.route_sha256`, equal to `meta.route_sha256` of a sampled continuation) and the prefix history window
  `branch_hist` (40 x 15 float32: row t = `state[F-39+t]` on the 12 observable columns 0-6, 11-14, 15 followed by
  `action[F-40+t]`), `branch_hmask` (40 bool: the action frame `F-40+t >= 0`, so F >= 40 is all valid) and
  `branch_hist_cols` (12 int16); this is the mixed dataset's window convention (`ga_build_mixed.py`) and the same
  block the CRM collector writes, so the branch labeller can read both worlds identically. Rows `[:F]` of the
  recorded arrays are the full prefix.
- `outcome.json` gains `ext` (mode, gates, near-stop rule report, events, branch record with the follower output
  before the swap, the vehicle speed and the branch route's first speed at the swap, the same `hist`/`hmask` window
  as lists with `hist_T`, `hist_cols` and `hist_layout`; perturbation summary; policy actor sha, observation layout and
  the `current_state_row_check`; substep-log summary) and, for `branch`, the top-level keys `branch_frame`,
  `branch_pose`, `branch_route_sha256`.
- `ext_control.npz` (external modes): `frame`, `shadow_action` (follower output read at the top of the frame, raw,
  before any clamp), `held_action` (the triple written at every substep = the recorded action, asserted), `desired_speed_mps`,
  `parked`, `perturbation` (d_steer, d_throttle, tap_active, tap_level) and `policy_obs` (N x 158, float32).
- `substep_actions.npz` (`--substep-log`): `applied` (N x 25 x 3) = the clamped triple handed to `hmmwv.Synchronize` at
  every substep, logged through the frozen loop's own `frame_observer.on_substep` call by a wrapper around the rich
  telemetry observer (reads only). The summary in `outcome.json.ext.substep_log` has the per-channel maximum
  intra-interval range and the count of intervals with any range.
- `command_reference.npz` additionally carries `branch_frame`, `branch_waypoints/stations/speeds/headings` in branch mode;
  `desired_speed_mps` comes from the rich telemetry and switches to the branch profile at F.
- `branch_route.json` is copied into the run directory in branch mode.

Gates. Without `--local` the collector behaves exactly like `gen_collect.py`: `source_manifest.json` must list the
frozen source files with matching hashes, `FDM_RUNTIME_FINGERPRINT` must point to a fingerprint with the vehicle library
and HMMWV data, the arena BMP must be in `gen_arenas.json`, the case must declare a split and no assets. `--local`
bypasses only the manifest and the fingerprint; `collection_request.json.ext.gates`, `outcome.json.ext.gates` and
`f104_episode.json.gates` record `BYPASSED (--local)`. The arena allowlist and the case checks are never bypassed.

`branch_auto` (`--recorded RUNDIR --branch-frame F --n-cont 3 --cont-seed S`, optional `--cont-parallel`,
`--replay-pose-tol-m 0.5`, `--stall-run-frames 20`, `--sub-timeout-s 3600`): checks that the recorded episode's
`route_sha256`/`case_sha256` equal the given route and case; reads the recorded `pose[F]`, `vx` and the stalled/moving
class at F (stalled = the last 20 recorded frames before F all have |vx| < 0.3 and throttle > 0.3); pass 1 replays the
prefix in a subprocess in native mode with horizon F/20 s into `<out>__replay_F<F>` (keyed by F; a cached replay is
reused only if its `requested_horizon_s` equals F/20); the replay is accepted only if it ran exactly F frames with
status `timeout` (a prefix that reached the goal or a stop rule at or before F is skipped with reason
`replay_ended_before_branch_frame`), the replayed pose (its `terminal_pose`) is within the tolerance of the recorded
`pose[F]` and the class matches; otherwise `<out>/skipped.json` (with the full comparison) and
`<out>/episode_complete.json {"skipped": true}` are written and the process exits 0. Otherwise
`gc_control.sample_continuations(replayed pose, goal, k, seed, v0 = replayed vx at F, max_start_heading_err_deg =
--cont-heading-tol-deg (15))` draws the continuations (speed profile floored at the 2 m/s^2 deceleration ramp from the
vehicle's speed so the new follower never starts with a hard brake; draws whose first tangent kinks more than the
tolerance away from the vehicle's heading are redrawn; opt-in `--cont-accel-cap` also caps the profile at the
acceleration ramp from v0; a `RuntimeError`/`ValueError` there also skips), each is written to
`<out>__c<i>/branch_route.json`, and `<out>__c<i>` is driven in a subprocess with `--mode branch --branch-frame F
--branch-route ...` (the full `--horizon-s` of the row, the same safeguards, `--local`/`--substep-log` forwarded). `<out>/branch_auto.json` has the replay comparison (pose and yaw
difference, vx, classes, prefix deviation), the continuation routes (content hash, base speed, draws), statuses and
wall times; `<out>/episode_complete.json` is written only when all continuations completed (exit 1 otherwise). Sibling
directories are used so every drive is a normal run directory; the runner deletes rich telemetry from them too.

### `scripts/gen_runner_g.py`, `scripts/gen_array_g.sbatch`

Copies of `gen_runner.py` / `gen_array.sbatch` parameterised by `GEN_ROOT` (default
`/work1/dannegrut/harry/experiments/generalist_20260921`), `GEN_COLLECTOR` (default `$GEN_ROOT/source/scripts/gen_collect_ext.py`),
`GEN_RUNNER` (default `$GEN_ROOT/source/scripts/gen_runner_g.py`), `GEN_CHRONO_DATA` (default the chrono-build data path
of `gen_array.sbatch`), `GEN_TASKS`, `GEN_OUT`, `GEN_WORKERS` (CPUs - 2), `GEN_TIMEOUT_S` (3600 per task, x4 for
`branch_auto` rows). The sbatch keeps `source env.sh; nrd_pychrono; nrd_use_lavapipe`, the thread variables,
`FDM_RUNTIME_FINGERPRINT=$campaign/pilot_runtime_412394.json` and the `chrono-build/data` path exactly as
`gen_array.sbatch` (diff checked: only the parameterised lines and the runner path differ). Task rows:
`{id, case, route, shard[, run, group, tier, mode, extra: [...], n_cont]}`; `case`/`route` are relative to `GEN_ROOT`
unless absolute; `mode` becomes `--mode`; `extra` is appended verbatim (absolute paths for `--recorded`,
`--branch-route`, `--actor`, seeds, `--substep-log`). Output `runs/<id>` (+ `runs/<id>__replay`, `runs/<id>__c*` for
branch_auto rows), logs `logs/<id>.log`, skip-if-`episode_complete.json`, rich telemetry deleted after success.

Launch example (do not launch without the balance check and the `gen_collect.py --check-only` against G/source):

    sbatch --parsable -p mi2104x -c 128 -t 03:00:00 --array=0-5 -J rigid_g \
      --export=ALL,GEN_ROOT=$G,GEN_TASKS=$G/tasks_rigid.json,GEN_OUT=$G/rigid_v1 -o $G/rigid_v1/logs/%x_%A_%a.out \
      $G/source/scripts/gen_array_g.sbatch

`GEN_ROOT` is passed explicitly: with `--export=ALL` a `GEN_ROOT` left in the submitting shell (the gen_v1 root) would
otherwise be inherited by the `${GEN_ROOT:-...}` default and silently redirect the source root, cases and routes.

## How to run locally

    cd /home/harry/NeDM-traverse_mppi
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONPATH=src:scripts
    PY=/home/harry/miniconda3/envs/nedm/bin/python
    CASE=$PWD/artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases/f104_v2_group_0000.json
    ROUTE=$PWD/artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases/routes/f104_v2_group_0000/route_00.json
    COMMON="--source-root . --case $CASE --route $ROUTE --chrono-data /home/harry/chrono/data --horizon-s 120 --local"
    $PY -P -u scripts/gen_collect_ext.py $COMMON --out OUT --mode native
    $PY -P -u scripts/gen_collect_ext.py $COMMON --out OUT --mode pid_held --substep-log
    $PY -P -u scripts/gen_collect_ext.py $COMMON --out OUT --mode pid_perturbed --perturb-seed 7 --substep-log
    $PY -P -u scripts/gen_collect_ext.py $COMMON --out OUT --mode policy --actor ACTOR.npz
    $PY -P -u scripts/gen_collect_ext.py $COMMON --out OUT --mode branch --branch-frame 60 --branch-route ROUTE_F.json
    $PY -P -u scripts/gen_collect_ext.py $COMMON --out OUT --mode branch_auto \
        --recorded artifacts/traverse/fdm_f104_50h_20260909/production_v3/runs/f104_v2_group_0000_route_00 \
        --branch-frame 60 --n-cont 3 --cont-seed 20260921 --cont-parallel 3

Rigid runs are CPU-only (no lock needed); a 20 s episode takes 35-40 s wall on luffy with one thread. Runner test:

    GEN_ROOT=<root with source -> repo> GEN_COLLECTOR=$PWD/scripts/gen_collect_ext.py GEN_CHRONO_DATA=/home/harry/chrono/data \
    GEN_TASKS=tasks.json GEN_OUT=out NRD_PYTHON=$PY SLURM_ARRAY_TASK_ID=0 GEN_WORKERS=2 $PY -P -u scripts/gen_runner_g.py

## What was tested (episode f104_v2_group_0000 / route_00; the cluster recording reached the goal at 19.50 s)

All commands above with the outputs under `selftest/rigid/`. The reference for S1 is the unmodified path:
`gen_collect.import_runner` + `gen_collect.adapted_function` + `gen_collect.StopPolicy` + `gen_collect.make_observer`,
gates bypassed exactly as `rigid_moving_collect.py` does (`selftest/rigid/S1_ref`, the heredoc is in the log).

- S1 native vs unmodified path: `trajectory.npz` byte-identical (`cmp`), also `anchor_state.npz`, `rich_telemetry.npz`,
  `rich_intervals.npz`, `contact_events.json`; `outcome.json` differs only in `wall_s`. Both goal_reached at 19.55 s
  (391 frames). Re-checked on the final file version (`S1_native_final`): byte-identical again.
- S2 native twice: `trajectory.npz` byte-identical; `outcome.json` differs only in `wall_s`.
- S3 pid_held (`--substep-log`): 391 frames, goal_reached 19.55 s; substep log (391 x 25 x 3) max intra-interval range
  [0, 0, 0] per channel, 0 intervals with any range; `applied[:, 0] == action` and `held_action == action` exactly;
  max steering step per frame 0.100; throttle and brake never both > 0; pose deviation from the native drive at most
  0.156 m over the episode. The largest raw shadow-vs-held steering gaps (up to 1.39) are in the last ~15 frames before
  the goal where the follower's raw output saturates at +-1 and flips sign (its end-of-path behaviour); the native loop
  clamps that to the same 0.1 per frame, so the held drive is actuator-matched, not divergent.
- S4 branch at frame 60 (route sampled from the reference drive's pose at 60 with `sample_continuations`, seed 4):
  rows [:60] of state/action/pose identical to native, first differing frame 61; `action[60]` = (0.064, 0, 0)
  (previous steering 0.068 minus one substep of clamp, the new driver's zero throttle for one substep), substep 1 of
  frame 60 already (0.060, 0.753, 0); `branch_start_error_m` 0.0; desired speed switches 2.0 -> 2.99 m/s at frame 60;
  `outcome.json` and `trajectory.npz` carry `branch_frame` 60, `branch_pose`, `branch_route_sha256`
  (`aaf361ab...`); the drive ended in `prolonged_blockage_terminated` (native rule) at 54.7 s on that wide continuation.
- S5 branch_auto on the recorded production_v3 episode, F = 60, 3 continuations, seed 20260921, `--cont-parallel 3`:
  replay 60 frames in 17 s wall; replayed pose 0.095 m from the cluster recording's `pose[60]` (yaw 0.003 rad, vx 1.73 vs
  1.69 m/s, prefix max pose deviation 0.134 m, class moving/moving) -> accepted; replay prefix identical to the native
  drive and its terminal pose equals native `pose[60]`; continuations (base speeds 2, 4, 6 m/s; 1, 3, 4 draws) all
  goal_reached (goal times 41.6, 10.8, 9.7 s from the start; 832, 216, 194 frames), every continuation's prefix [:60]
  identical to native and `branch_start_error_m` 0.0; total 80 s wall; `branch_auto.json` and `episode_complete.json`
  written. Skip paths: F = 1000 > 390 recorded frames -> `skipped.json` (`recording_shorter_than_branch_frame`), exit 0,
  nothing driven; `--replay-pose-tol-m 0` -> replay driven, `skipped.json` (`replay_mismatch: pose diff 0.095 m`), exit 0.
- S6 policy with the 158-input random actor from the `gc_control` self-test (`selftest/selftest_actor.npz`, ELU
  512-256-128, weights x3): 114 frames, `terrain_bounds_exit` at 5.7 s (a random policy floors the throttle and steers
  hard); substep range [0, 0, 0]; observations 114 x 158, 0 non-finite entries; steering rate-clipped to 0.1 per frame
  from the settle's 0 (-0.1, -0.2, -0.3, ...). This actor emits throttle and brake both > 0 in every frame (its action
  squash has no exclusivity); the collector writes the triple as is.
- S7 pid_perturbed seed 7 (`--substep-log`): 416 frames, goal_reached 20.8 s; substep range [0, 0, 0]; max |held -
  shadow| throttle/brake 0.25 (the OU bound), 0 taps in this episode at brake_p 0.0037 (21 % braked frames come from the
  follower's own braking plus noise), pose deviation from native up to 4.7 m. Seeding through `--episode-seed 7`
  instead (`S7b_episode_seed`) reproduces the same `trajectory.npz` byte for byte (see the final-check block in the
  rerun log).
- S8 runner: two rows (native with `extra: ["--local"]`, pid_perturbed with `mode` and `extra` seed/substep-log), shard
  0, 2 workers, `GEN_ROOT` a local root whose `source` links to the repo (the link was removed after the test so no
  recursive copy of the artefacts can follow it; recreate it with `ln -s $PWD S8_runner/root/source` to rerun): both
  `ok`, `episode_complete.json` present, rich telemetry deleted, 36 s; an empty shard exits cleanly; `bash -n` on the
  sbatch passes.
- Gates: without `--local` the run stops at `source_manifest.json` (missing locally); `--mode branch` without
  `--branch-route` is rejected before anything is loaded.

## Known limits

- `policy` mode starts at frame 0 only; a policy taking over after a recorded prefix (`seed_history`) is not wired
  (the B0 suite drives all three arms from rest).
- The branch swap gives one substep of zero throttle/brake at frame F and a fresh PID integrator; a brake tap of one
  substep can occur when the vehicle is faster than the branch route's first speed (documented in `make_follower`).
  `action[F]` is therefore a non-hold transition; the `hold_ok` filter of the dynamics training handles it.
- `branch_auto` compares the local replay with a recording made on another node; on luffy the prefix deviates by
  ~0.1 m from the cluster recording at 3 s, within the 0.5 m tolerance for this moving anchor. The anchor list for A4
  should be built with the drop counts from `skipped.json` files (`reason` field).
- The near-stop rule is only active in the external modes; a policy that stops with zero throttle is caught by it
  (regardless of throttle), a follower that stalls with throttle > 0.3 by the native rule first.
- The extra `capture_row` call in policy mode reads tyre forces one substep stale; only the 12 observable columns feed
  the policy.
- `trajectory.npz` is rewritten (atomically) in non-native modes to add the `ext_mode`/branch keys; readers that
  compare file hashes across modes must expect that. Native mode never rewrites it.
- The runner treats `branch_auto` rows as one worker with sequential continuations unless `extra` carries
  `--cont-parallel`; with `GEN_WORKERS = CPUs - 2` a parallel setting oversubscribes the node.
- The CRM sibling collector (`crm_collect_ext.py`, written in parallel) seeds its perturbation from `--episode-seed`;
  this collector accepts both `--perturb-seed` and `--episode-seed` so the same task-row convention works in both worlds.

## Fix round 1 (2026-09-21, late evening; answers VERIFY_gen_ext.md P1-P4)

File after the fixes: `scripts/gen_collect_ext.py` sha256 `b9f36a024570...` (before: `f010b729a2aa...`); it imports
`scripts/gc_control.py` `683f24ad3992...` (fix round 1 of that module). The frozen-loop hooks are untouched: the
adapted `run_chrono` source still hashes to `8cf1bd508877...` (original `f060168debf4...`, gen_collect's adapted
`162c594c03a4...`, 6 insertions + 1 replacement). `gen_runner_g.py` and `gen_array_g.sbatch` were not edited.
Artefacts of this round: `selftest/fix1/rigid/` (final files; the same smokes with the intermediate file before the
opt-in `--cont-accel-cap` flag existed are in `selftest/fix1/rigid_prelim/`, identical continuation hashes). All runs
CPU-only with `--horizon-s 12` and `--local`; no CRM run.

Provenance of the earlier artefacts (P3's request): `S4_branch`, `S5_auto*`, `S6_policy`, `S7_perturbed` were made
with wrapper `b21c86af...`, `S1_native`/`S2_native` with `3bcb7ae4...`; `S1_native_final`, `S3_held_final`,
`S7b_episode_seed` and the verifier's runs 1-2 with `f010b729...`; this round's `fix1/rigid/*` with `b9f36a02...`.

1. **P1, policy engine-speed column** (`GenExt.bind` line 211, `_capture` line 283): hook A now also binds
   `vehicle.GetTransmission()` and the policy's state row takes engine speed from
   `transmission.GetOutputMotorshaftSpeed()` (the integrated shaft state that the SHAFTS engine's `Synchronize`
   imposes on the engine shaft) instead of `engine.GetMotorSpeed()` (which, read before the frame's `Synchronize`,
   still held the previous substep's imposed value). Every policy run now asserts that the newest row of the
   observation's past-state block equals the recorded `state[k]` on all 12 observable columns
   (`policy_state_check`, line 460; result in `outcome.json` `ext.policy.current_state_row_check`). S6 re-run
   (`fix1/rigid/S6_policy`, same random actor, `terrain_bounds_exit` at 5.7 s, 114 frames): per-column max |diff|
   `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`, 114/114 frames exactly equal (an independent recomputation from
   `ext_control.npz` and `trajectory.npz` agrees). The pre-fix S6 shows the verifier's lag: column 15 max 1.008 rad/s,
   mean 0.522, the other 11 columns 0. Docstring (lines 30-38) and the `state_capture` string corrected.
2. **P2, perturbation seed fails closed** (`perturb_seed` line 106-113, `validate_mode_args` line 449-451, parser help
   line 421-422): `pid_perturbed` without `--perturb-seed` or `--episode-seed` is rejected before Chrono is loaded;
   `perturb_seed` raises instead of returning 0; `seed_source` in `outcome.json` can no longer say "default 0".
   `--check-only --mode pid_perturbed` without a seed: exit 1 with
   `ValueError: pid_perturbed needs --perturb-seed or --episode-seed (no default seed: every episode must have its
   own)` (`fix1/rigid/S9_noseed.log`); with `--episode-seed 7`: exit 0, contract `perturb_seed 7`
   (`S9b_seed.log`). The B3 task builder must still assert that the seeds are distinct across rows.
3. **P3, stale docstring** (lines 61-67): the external-mode stop rule is described as it is coded (40 s = 800
   consecutive recorded frames with |vx| < 0.3 m/s and not parked, regardless of throttle, status
   `prolonged_blockage_terminated`, event rule `near_stop_any_throttle`).
4. **P4, branch_auto replay directory and the ended-before-F case** (lines 686, 701-718, 568-580, 770-786):
   - the replay lives in `<out>__replay_F<F>`; a cached replay is additionally checked to have been run with
     `requested_horizon_s == F/20` (S12b: a cached dir whose horizon was edited to 4 s -> exit 1, `Cached replay ...
     was not run with horizon F`);
   - the replay is accepted only with `status == "timeout"` and `frames == F`; otherwise the anchor is skipped with
     `replay_ended_before_branch_frame: status X after N frames (branch frame F)` (S12: S5's replay copied with
     status edited to `goal_reached` -> `skipped.json`, `episode_complete.json {skipped: true}`, exit 0, no `__c*`
     directory created);
   - a `--mode branch` run whose prefix drive ends before F no longer exits 1: it writes `outcome.json` (with `ext`),
     `skipped.json` (`episode_ended_before_branch_frame: status ... after N frames (branch frame F)`) and
     `episode_complete.json {skipped: true, actual_elapsed_s: 0}` and returns 0 (S11: group_0996 route_02, whose
     drive reaches the goal at 5.45 s locally, `--branch-frame 150` -> skipped, `status goal_reached after 109
     frames`); `branch_auto` reads such a continuation as `skipped` (counted in `n_skipped_continuations`, excluded
     from `actual_elapsed_s`) instead of failing on a missing branch key.
   S10 (a recording that reached the goal at frame 107, `--branch-frame 108`): on this machine the replay ends 2.515
   m from the goal (radius 2.5) at frame 108 with status `timeout`, so it was accepted (pose diff 0.078 m) and the
   three continuations all reached the goal one frame later; the status check is therefore exercised by S12, not
   by S10.
5. **Continuation sampling at the branch** (lines 733-746, parser 432-433): `sample_continuations` is called with
   `v0 = replayed vx at F` and `max_start_heading_err_deg = --cont-heading-tol-deg` (default 15); the opt-in
   `--cont-accel-cap` also caps the profile at the acceleration ramp from v0 (first speed == v0 exactly; off by
   default, as the task specified the floor only). `branch_auto.json` records `continuation_sampling` (v0, tolerance,
   floor formula, cap flag) and per continuation `start_speed_mps`, `start_heading_err_deg`,
   `speed_floor_raised_points`; the branch info in `outcome.json` gains `vehicle_vx_at_swap_mps` and
   `branch_start_speed_mps` and its swap note now states the follower's real braking behaviour.
   S5 re-run (group_0000 route_00, F = 60, seed 20260921, `--cont-parallel 3`, 58 s wall): replay 60 frames,
   `timeout`, pose 0.095 m from the cluster recording, accepted; replayed vx 1.733 m/s; continuations (base 2, 4,
   6 m/s; 1, 11, 13 draws) start at 1.733 (floor raised 2 points), 4.645 and 5.439 m/s with start-heading errors
   5.9, 12.9, 10.8 deg; each prefix `[:60]` identical to the replay, `branch_start_error_m` 0.0; statuses at the 12 s
   horizon timeout, timeout, goal_reached (10.65 s). In the 3 s after the branch the maximum brake was 0.39, 0.25,
   0.00 (the verifier's pre-fix burst was 0.96-1.0 for 0.5 s): continuation 0's sampled profile drops toward the
   family's 0.5 m/s floor within a metre, a legitimate 2 m/s^2 deceleration, not a target step below the vehicle's
   speed.

Not changed in that round: P5 (stop-rule symmetry by argument), P6 (`GEN_ROOT` inheritance in the sbatch) and P7
(`f104_n2_sampler` hash in the contract). P6 and P7 were then done in the resumed implementer round below (the sbatch
launch comment passes `GEN_ROOT` explicitly; the contract records `f104_n2_sampler_sha256`); P5 stays a documented
property (the follower saturates throttle above 0.3 when stalled, so the native rule fires first for the PID arms).

## Resumed implementer round (2026-09-21 ~22:33-22:55): merge, two additions, full suite re-run on the final file

The original implementer was resumed after a usage-limit interruption while the fix-round agent above was still
active; both edited `scripts/gen_collect_ext.py` within minutes of each other (fix round: P1-P4 and the opt-in
`--cont-accel-cap`, written 22:36:30; this round: the two additions below, applied to the same file). Both edit sets are
present in the final file and every self-test below was run on it: `scripts/gen_collect_ext.py` sha256
`b9f36a024570...` (804 lines), `scripts/gen_runner_g.py` `b47c9fd53c33...` (unchanged since the first version),
`scripts/gen_array_g.sbatch` `e3bcb4d8d742...`; imported `gc_control.py` `683f24ad3992...`, `gen_collect.py`
`b6ba062260aa...`, `f104_n2_sampler.py` `c80184491ba5...`. The frozen-loop adapter is untouched: adapted `run_chrono`
`8cf1bd508877...` (original `f060168debf4...`, gen_collect's own `162c594c03a4...`; 6 insertions + 1 replacement).

Additions of this round (no change to any hook or to the physics):

1. Branch prefix history in the CRM collector's convention: `trajectory.npz` gains `branch_hist` (40 x 15),
   `branch_hmask` (40) and `branch_hist_cols` (12); `outcome.json.ext.branch` gains `hist`, `hmask`, `hist_T`,
   `hist_cols`, `hist_layout` (function `prefix_history`, computed from the recorded arrays after the drive, so a
   branch labeller reads both worlds with one code path).
2. Provenance (verifier P7): `collection_request.json` records `f104_n2_sampler_sha256`. Verifier P6: the sbatch
   launch comment passes `GEN_ROOT=$G` explicitly (the `${GEN_ROOT:-...}` default itself is kept, as the brief asks).

Suite re-run, all under `selftest/rigid_final/` (`env.sh`, `batch1-4.sh`, `s1_ref.py`, `analyze.py`; the checks are
recomputed by `analyze.py` from the files and saved as `analysis_*.json`). Episode f104_v2_group_0000 / route_00,
`--horizon-s 120 --local`, single-threaded rigid runs, at most 4 of mine concurrent (the fix-round agent's runs
overlapped in time; Chrono rigid is deterministic on this node regardless, see S1/S2/S8).

- No-Chrono checks: `--check-only` hook counts `{insertions: 6, replacements: 1}` and the three adapter hashes above;
  no output directory created. Rejected before Chrono loads: no `--local` locally (stops at the missing
  `source_manifest.json`), `--mode branch` without `--branch-route`, `pid_perturbed` without a seed, `--near-stop-s 10`,
  a branch frame beyond the horizon. Gated path (S9_gates: a temporary root with `source_manifest.json` listing the 9
  frozen files): valid manifest -> both gates `checked`; a wrong file hash -> `Frozen source file mismatch`;
  `--source-manifest-sha256` mismatch -> `Source manifest mismatch`; a real run without `FDM_RUNTIME_FINGERPRINT` and
  without `--local` -> fails at the fingerprint gate. Torch-free: after importing the collector and sampling a
  continuation, `torch` is not in `sys.modules` (nor is pychrono).
- S1 native vs the unmodified path (`s1_ref.py`: `gen_collect.import_runner` + `adapted_function` + `StopPolicy` +
  `make_observer`, gates bypassed like `rigid_moving_collect.py`): `trajectory.npz`, `anchor_state.npz`,
  `rich_telemetry.npz`, `rich_intervals.npz`, `contact_events.json`, `command_reference.npz` all byte-identical;
  `outcome.json` differs only in `wall_s`; goal_reached at 19.55 s, 391 frames.
- S2 native twice: the same six files byte-identical; only `wall_s` differs.
- S3 pid_held (`--substep-log`): 391 x 25 x 3; per-channel max intra-interval range [0, 0, 0], 0 intervals with any
  range; substep 0 == recorded action exactly; held == recorded to 1.5e-8 (float32 storage); max steering step per
  frame 0.100; 0 frames with throttle and brake both > 0; pose deviation from native at most 0.156 m; goal_reached.
- S4 branch at F = 60 (route re-sampled from the replayed frame-60 pose with `v0` 1.733 m/s, seed 4: the same
  content hash `aaf361ab...` as the first version's route, start heading error 6.2 deg, floor raised 0 points):
  prefix `[:60]` of state/pose identical to native, `state[60]` identical, first differing action frame 60;
  `action[60]` = (0.0640, 0, 0), substep 1 of frame 60 = (0.0600, 0.753, 0); `branch_pose == pose[60]`; desired speed
  2.0 -> 2.989 at F; `branch_frame`/`branch_pose`/`branch_route_sha256` at the top level of `outcome.json`, in
  `trajectory.npz` and (`branch_*` reference arrays) in `command_reference.npz`; `branch_hist` 40 x 15 with `hmask`
  all true, first row == [state[21][cols], action[20]] and last row == [state[60][cols], action[59]], the
  `outcome.json` copy equal to the npz; `branch_route.json` copied; ended `prolonged_blockage_terminated` (native rule)
  at 1094 frames on that wide continuation, as before.
- S5 branch_auto on the recorded cluster episode (`production_v3/runs/f104_v2_group_0000_route_00`, F = 60, 3
  continuations, seed 20260921, `--cont-parallel 3`, 76 s wall): replay 60 frames, status `timeout`, 17.3 s wall;
  replayed pose 0.095 m from the recording's `pose[60]` (yaw 0.003 rad, vx 1.733 vs 1.690 m/s, prefix max pose
  deviation 0.134 m, class moving/moving) -> accepted; the replay's arrays are identical to the native drive's
  `[:60]` and its terminal pose equals native `pose[60]` exactly; continuations (base 2/4/6 m/s; start speeds
  1.733/4.645/5.439 m/s; start-heading errors 5.9/12.9/10.8 deg; 1/11/13 draws) all goal_reached at 824/359/213 frames
  (41.2/17.95/10.65 s), each prefix `[:60]` identical to native, `branch_pose` == the replay's terminal pose,
  `branch_start_error_m` 0.0, `branch_hist` present; `branch_auto.json` + `episode_complete.json` (69.8 s of new
  data). Skip paths, all exit 0 with `skipped.json` + `episode_complete.json {skipped: true}`: F = 1000 > 390 recorded
  frames -> `recording_shorter_than_branch_frame`, nothing driven; `--replay-pose-tol-m 0` -> replay driven, then
  `replay_mismatch: pose diff 0.095 m`, no continuation directory; F = 390 (the recording's goal frame; the local
  replay stops 1 frame short of the goal with `timeout`, pose accepted) -> `no_valid_continuation` (192 draws, all
  rejected by the planner validator from inside the goal radius), no continuation directory. The
  `replay_ended_before_branch_frame` path is exercised by the fix round's S12.
- S6 policy (random 158-input ELU actor `selftest/fix1/gc_control/selftest_actor.npz`, `--substep-log`): 114
  frames, `terrain_bounds_exit` at 5.7 s; substep range [0, 0, 0]; `current_state_row_check` max |diff| 0.0 on all 12
  observable columns (the engine-speed lag of the first version is gone); observations 114 x 158, 0 non-finite;
  steering rate-clipped from the settle's 0 (-0.1, -0.2, ...); this actor has no throttle/brake exclusivity (both > 0
  in all 114 frames), written as is.
- S7 pid_perturbed seed 7 (`--substep-log`): 416 frames, goal_reached 20.8 s; substep range [0, 0, 0]; |d_steer| <=
  0.15, |d_throttle| <= 0.25, |held - shadow| throttle/brake <= 0.25; 0 taps, 21.4 % braked frames; 0 frames with both
  > 0; `--episode-seed 7` (`S7b_episode_seed`) gives a byte-identical `trajectory.npz` with `seed_source
  --episode-seed`.
- S8 runner (`gen_runner_g.py`, `GEN_ROOT` a local root with `source -> repo`, shard 0, 2 workers): rows native
  (`extra: ["--local"]`) and pid_perturbed (`mode` + `extra` seed/substep-log/`--horizon-s 60`) both `ok` in 35 s;
  `episode_complete.json` present, rich telemetry deleted; the native run's `trajectory.npz` is byte-identical to S1;
  the row's `--horizon-s 60` in `extra` overrides the runner's 120 (388 frames); a `run: false` row and a row of
  another shard are not run; shard 5 (empty) exits 0; the `source` link was removed afterwards; `bash -n` on the
  sbatch passes and its diff against `gen_array.sbatch` is only the comment block, the four `GEN_*` defaults and the
  runner path.

Open points for the reviewer: (a) two agents edited the collector concurrently; the final file was hashed before the
suite and re-hashed after it (unchanged), and the note's line numbers in the fix-round section are now offset by ~20
lines; (b) the B3 task builder must give every `pid_perturbed` row its own seed and the A4 row builder an
anchor-specific `--cont-seed` (nothing in this module enforces distinctness across rows); (c) `branch_auto` rows in
the runner get `GEN_TIMEOUT_S x (1 + n_cont)`; with `--cont-parallel` in `extra` a row uses more than one core.
