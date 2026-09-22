# Verification of module crm_ext (`scripts/crm_collect_ext.py`)

Adversarial verifier, 2026-09-21 late evening (22:44-23:15). Files read: `PLAN.md`, `NOTES_crm_collect_ext.md`,
`scripts/crm_collect_ext.py` (sha256 `cbf57f7f3c86...`, the file the implementer calls final), the unmodified
`scripts/crm_collect.py` (`cb6792bebeb1...`, unchanged since c81c3f9f3), `scripts/gc_control.py` (`683f24ad3992...`),
`scripts/gen_collect.py` (StopPolicy), the frozen `traverse_fdm_rgbd_diverse_chrono.py` (`make_driver`, `nearest_index`,
`read_route`), `scripts/ga_build_mixed.py` (history convention), `scripts/crm_worker.py`, the self-test scripts under
`selftest/crm/`, `VERIFY_gen_ext.md`, `VERIFY_gc_control.md`, and the Chrono sources `ChPowertrainAssembly.cpp`,
`ChEngineShafts.cpp/.h`, `ChAutomaticTransmissionShafts.cpp`, `ChDriver.h/.cpp`, `ChPathFollowerDriver.cpp/.h`.
Budget: the implementer's 7-run chain was already queued when I started (I waited for it, under the lock); I added
ONE extra CRM run (`/tmp/vcrm/runs/nearstop_policy`, 5 s of sim, 211 s wall while the substep-audit verifier shared
the GPU); 0 rigid runs; no cluster submission; collector GPU share 1,678 MiB per run (< 8 GB). Nothing in the repo
was modified by me: the implementer's `check.py` was run from a copy patched to write `/tmp/vcrm/selftest_report.json`
(the self-test directory still has no `selftest_report.json`; `NOTES` still says `RESULTS_PLACEHOLDER`). My scratch
outputs: `/tmp/vcrm/` (`analyze.py`, `verify_analysis.json`, `brake_actor.npz`, `run_nearstop.sh`, `runs/`).

Verdict: **pass with issues**. PLAN's ship-blocking property holds and is reproduced here: native mode is byte-identical
to the unmodified collector (four npz files + `initial_state_validation.json` by `cmp`; `outcome.json` differs only in
`wall_s`, `crm/build_s`, `crm/rtf_sim_over_wall` and the three appended provenance keys; `physics_dt_s` 0.001, 50
substeps). Held-triple, perturbed, policy and branch contracts hold when checked with a float32-aware comparison. The
implementer's own `check.py` however reports S3/S5/S6 as FAILED because of two tolerance bugs in the checker (P1), and
the branch smoke is a 9 s stall caused by the sampled continuation's 0.5 m/s target, which the S4 check does not look at
(P2). Both must be dealt with before the NOTES results block is written; P2 also needs a decision from the A4/sampler
owner before the CRM branch launch.

## Reproduced

1. Argument gates (7 cases, CPU, ~0.1 s each, no `pychrono` import, no output directory created): missing `--crm-config`;
   `pid_perturbed` without `--episode-seed`; `--branch-frame 240` at horizon 12 s; `policy` without `--actor`; `--actor`
   in native mode; `--branch-frame -1`; `--branch-frame` in native mode. All rc = 1 with the intended `ValueError`.
2. `prefix_history` on synthetic arrays at F = 0, 1, 10, 39, 40, 41, 60, 99: valid rows = min(F, 40), every valid row t
   equals `[state[F-39+t][cols], action[F-40+t]]`, masked rows are zero, F = 0 is all-masked, F = len raises. The
   column set equals `ga_build_mixed.HIST_STATE_COLS`. `import crm_collect_ext` leaves torch and pychrono unloaded.
3. Near-stop offline replay: my own vectorised implementation over all 15,235 `collect_v1` episodes (6.6 s) gives the
   same 9 changed episodes (all 120 s timeouts, rule firing at frames 1030-1282), the same per-status table
   (0 of 1,985 blockage / 4,867 goal / 8,372 breakthrough / 2 rollover touched) and the same run-length statistics
   (p50 5.55 s, p99 28.85 s, max 108.45 s, 40 episodes >= 30 s).
4. S1 (`base_native` vs `ext_native_1`, both 12 s, both from this evening's chain): `trajectory.npz`,
   `command_reference.npz`, `crm_extra.npz`, `anchor_state.npz`, `initial_state_validation.json`, `case.json`,
   `reference.json` byte-identical; `f104_episode.json` identical except `wall_s_including_finalization`;
   `episode_complete.json` artefact hashes differ only for `collection_request.json`, `f104_episode.json`,
   `outcome.json`; `outcome.json` key order ends `..., measurement_scope, mode, collector_sha256, physics_dt_s`.
5. S2 (`ext_native_1` vs `ext_native_2`): the four npz files byte-identical (local CRM determinism); the native
   substep log shows the follower refreshing inside every interval (max intra-interval spread steer 0.098, throttle
   1.0, brake 0.05 over 240 intervals) and `action[k]` == the substep-0 applied inputs after a float32 cast.
6. S3 `pid_held` (240 frames x 50 substeps): per-channel intra-interval spread exactly [0, 0, 0]; applied substep-0
   triple == recorded `action` after float32 cast (max |d| 3.0e-8 = float32 storage); max steering step per frame
   0.10000002 (float32 of 0.1), 0 frames above 0.1 + 1e-6; throttle and brake never both > 0; offline
   `hold_clip(shadow[k], action[k-1][0])` reproduces `action[k]` to 2.4e-8; `driver.control` = `pid_held`;
   `near_stop_rule` block present (active, not fired, longest run 0.4 s).
7. S5 `pid_perturbed` (seed 20260921): same hold properties; OU replay from the recorded parameters reproduces the
   action (5.4e-8) and the logged perturbation (7.4e-9); d_steer within +-0.15, d_throttle within +-0.25; throttle and
   brake never both > 0; `brake_p_per_frame` 0.003694 recorded. Note: 0 taps occurred in the 240 frames (expected 0.9),
   so the tap branch was not exercised by this smoke (see P6).
8. S6 `policy` (random self-test actor, 158-d): hold properties as S3; `pre_capture_vs_recorded_observable_max_abs_diff`
   = 0.0 (the transmission-shaft engine-speed read works on the CRM vehicle: `ChPowertrainAssembly::Synchronize` reads
   `m_transmission->GetOutputMotorshaftSpeed()` and imposes it with `SetPosDt` on the engine shaft, so the pre-Synchronize
   transmission value is exactly the post-Synchronize `engine.GetMotorSpeed()`); offline `PolicyObs` twin on the
   recorded pose/state/action reproduces the action to 2.9e-8; `obs_nonfinite_zeroed` 0; status timeout.
9. S4 `branch60` (continuation c0 regenerated by `make_branch.py` from the final `ext_native_1`): prefix `state`,
   `action`, `pose` `[:60]` bit-identical to native; `state[60]` (all 17 columns) and `pose[60]` identical to native;
   `branch_pose == pose[60]` exactly; first branch substep inputs (-0.20183, 0, 0) = the frozen clamp moving the last
   substep's steering (-0.20383) by 0.002 toward the new driver's 0; `action[60]` = that triple; steer step vs
   `action[59]` 0.027; `branch_hist`/`branch_hmask` equal `prefix_history(native state, native action, 60)`, last row =
   `state[60][cols] + action[59]`, 40 valid rows; `branch_route_sha256` = sha of `branch_reference.json`,
   `branch_route_content_sha256` = `gc_control.route_sha256` = the route meta hash; `command_reference.npz` carries the
   four `branch_reference_*` arrays, `branch_frame`, `branch_reached`; desired speed 2.896 -> 1.886 (= vx at F) at frame
   60. `zero_duration`/`ramp_duration` of `ChPathFollowerDriver` default to 0, so the fresh driver has no target ramp.
10. My extra run (the near-stop rule's `break` path, which no chain smoke executes because 12 s < 40 s): `policy` mode
    with a constant (0, 0, 1) actor (`/tmp/vcrm/brake_actor.npz`, one zero-weight layer) and `--near-stop-s 5`:
    status `prolonged_blockage_terminated` at exactly 5.0 s (100 frames), event
    `{kind: prolonged_blockage_terminated, rule: near_stop_any_throttle, interval_end_s: 5.0, near_stop_s: 5.0}` in
    `outcome`/`f104_episode.json`, `near_stop_rule.fired` true, `episode_complete.json` status matches, native effortful
    rule 0.0 s (throttle 0 -> the native rules could never have stopped it), 100 x 50 substeps all (0, 0, 1), vx max
    0.025 m/s, `pre_capture` diff 0.0, collector GPU 1,678 MiB.
11. Provenance: all 7 chain runs carry `collector_sha256 cbf57f7f...`, `ext.gc_control_sha256 683f24ad...`,
    `ext.base_collector_sha256 cb6792be...` (the current files); `base_native` was produced by `crm_collect.py`
    `cb6792be...`. rtf 0.44-0.49 once the training lanes finished (0.036-0.08 while they were resident).

## Problems

### P1 (major) The implementer's `check.py` fails S3, S5 and S6 on two float32 artefacts

`selftest/crm/check.py:71` requires `np.array_equal(per[:, 0, :], action[:n])` between the substep log (float64
doubles from `DriverInputs`) and `trajectory.npz` `action` (float32 schema); that can only be true when the held
values happen to be float32-exact, which PID/OU/actor outputs never are. Line 75 counts `dsteer > 0.1 + 1e-12` on
float32 actions, so a legal step of exactly 0.1 (stored as 0.10000002) is counted as a violation (8 / 16 / 14 frames).
Result of running the checker on the finished chain: `S3_pid_held false`, `S5_pid_perturbed false`,
`S6_policy false`, `all_pass false` (`/tmp/vcrm/selftest_report.json`); S1, S2, S4 true. The underlying contract holds
(reproduced item 6-8 with `per[:, 0].astype(np.float32) == action` and a 1e-6 tolerance). Smallest fix: in
`substep_stats` compare `per.astype(np.float32)` with `action` (or `np.abs(...) <= 1e-6`) and use `dsteer > 0.1 + 1e-6`;
then run `check.py`, replace `RESULTS_PLACEHOLDER`, and state the float32 storage precision (~3e-8) in the NOTES.

### P2 (major, design hazard, cross-module) The branch smoke is a 9 s stall caused by the continuation's 0.5 m/s target

`inputs/branch_60_c0.json` (base speed 2.0, `f104_n2_sampler` V_MIN 0.5) demands speeds [1.886, 1.252, 0.5, 0.5, ...]
from station 1 m onward (39 % of its points below 1 m/s, 71 % below 2 m/s; the v0 floor only protects the first 0.9 m).
In `runs/branch60` the fresh follower brakes 0.43-0.53 at frames 64-70, vx falls from 1.89 m/s to -0.09 at frame 76
and stays below 0.3 m/s for 123 consecutive frames (166 of the 180 post-branch frames; throttle 0.21-0.39 = the P term
on a 0.4 m/s error plus the slow Ki 0.05 integral, not enough to move the HMMWV on this soil); path length after the
branch 2.0 m vs 21.6 m for the native run over the same frames; `low_net_progress_4s_windows` 42,
`sustained_near_stop` true. `check.py` S4 has no progress criterion, so it reports `pass`. The collector executed the
route faithfully (this is not a code defect in this module), but two consequences matter: (a) PLAN A4 says
continuations have "speeds 2-6 m/s" while `sample_continuations` produces 0.5-6 m/s draws, and on CRM a low target is
a stall generator independent of terrain risk, so a share of A4 CRM "goal-reached-from-branch" labels would measure
the speed draw, not the terrain; (b) the module's S4 result is not evidence that a continuation is driven. The other
two continuations (c1 start 3.4 m/s, c2 start 6.0 m/s, 3 % of points below 2 m/s) look normal. Smallest fixes: in
`gc_control.sample_continuations` floor the non-terminal speeds at 2 m/s (PLAN's range) or reject draws whose
minimum speed outside the terminal deceleration cone is < 1.5 m/s (a `validator` hook suffices); add a progress
criterion to the S4 check (e.g. post-branch path length > 5 m or median vx > 1 m/s at 12 s) and re-run the branch smoke
on c1; record in the A4 anchor note which continuation speed floor was used.

### P3 (minor) NOTES claim "throttle and brake both positive (0 by construction)" is false for policy mode

`NOTES_crm_collect_ext.md:74`. Only `OUPerturb` enforces exclusivity; `hold_clip` does not, and the S6 smoke recorded
238 of 240 frames with throttle 0.9 and brake ~0.52 both positive (the random actor). This is consistent with the WP3
tracker action convention (PLAN B5 squashes throttle and brake independently over [0, 1]), so it is a note error, not a
contract breach; but B5's evaluation must know the collector applies whatever the actor outputs. Fix the sentence (or
add an explicit `--policy-exclusive-throttle-brake` option if B5 wants the max-of rule).

### P4 (minor) `--near-stop-s` is not gated

The rigid sibling rejects `--near-stop-s < 40` (`gen_collect_ext.validate_mode_args`); this collector accepts any value
(I used 5 s for the test above). A task row with a shortened value would silently change the B0/B3 stop rule.
Smallest fix: `require(args.near_stop_s >= 40. or args.allow_short_near_stop, ...)` with an explicit test-only flag.

### P5 (minor) The reached test after a branch uses the branch route's last waypoint instead of the case goal

`crm_collect_ext.py:212` sets `goal = bxy[-1]`; `goal` is only used by the reached test, `distance0/1` and
`goal_progress_m` (not by the follower or the parking rule, which use `xy[-1]`). The sampler ends every route within
0.25 m of the case goal (0.0 m in the smoke), so the effect is at most 0.25 m on a 2.5 m radius, but it makes the
goal-reached definition of branch drives differ from the recorded native data for no benefit. Smallest fix: leave
`goal` untouched at the swap (keep the `--branch-goal-tol-m` assertion).

### P6 (minor) Documentation and self-test coverage gaps

- `NOTES:58` "the gc_control default 0.05" is stale: `gc_control.DEFAULT_BRAKE_P` is now `brake_p_for_rate(1.4/20)`
  = 0.003694, the same value the collector computes.
- `RESULTS_PLACEHOLDER` is still in the NOTES; the report's "S1 pending" is now resolved (item 4).
- The perturbed smoke exercised 0 brake taps (12 s, p 0.0037); the tap branch (throttle 0, brake = max(follower,
  level)) is covered only by gc_control's self-test. One `--perturb-brake-p 0.05` smoke (or a 60 s one) would cover it.
- `runs/summary.txt` keeps the first-attempt line `ext_native_1 ... wall_s 1426 peak 24447` although that run's files
  were deleted and re-made by the chain (195 s, peak 13,923 total, collector 1,678); the 24,447 MiB figure in the
  report is no longer reproducible from disk. `check.py`'s summary print shows `gpu_peak_per_run: null` (the dict
  has no `pass` key); the JSON has the values.
- `collection_request.json.ext` records the gc_control and base-collector hashes but not `f104_n2_sampler.py`, on
  which branch routes depend (same gap as the rigid sibling's P7; the route content hash keeps drives reproducible).

## Contract checks (PLAN) — no violation found

- Timing: external modes read the shadow follower's `GetInputs()` at the top of frame k (`ChDriver::Synchronize` is a
  no-op, `GetInputs` returns a copy; the value is the last `Advance` of interval k-1 = what native records as `action[k]`
  before its clamp), clip once with `hold_clip(raw, action[k-1][0])` (settle action at k = 0), write the identical
  `DriverInputs` at all 50 substeps, record the triple at substep 0. Verified on 720 held intervals (spread 0).
- Native path: the same physics/driver statements in the same order (branch/external/near-stop branches are inert in
  native mode); byte identity verified (item 4) and local determinism (item 5).
- Branch swap: at the top of frame F before the route index and desired speed; `make_follower(initialize=False)` (no
  body added; constructor `Reset()`); `wp = 0`; `previous_steer` carried into the frozen +-2*dt clamp (verified
  continuity, item 9); prefix identical to native through `state[F]`; history window in the mixed-dataset convention.
- Observation: pre-Synchronize capture with the transmission's motorshaft speed equals the recorded 12 observable
  columns exactly (0.0 in two runs); frame-0 padding = rest state + (0, 0, 1); `(state[k], action[k])` pushed after the
  frame; the offline twin reproduces the actions.
- Near-stop rule: |vx| < 0.3 on 800 consecutive non-parked frames, after every native rule of the frame, external
  modes only by default, `--near-stop-all-modes` opt-in; the default is justified by the offline replay (9/15,235
  native outcomes would change); the firing path verified end to end (item 10).
- Physics/config: `--crm-config` required; `physics_dt_s` 0.001 in `outcome.json` (top level and `crm`), 50 substeps.
- Provenance: `outcome.json` provenance keys, `collection_request.json` `collector`/`mode`/`ext` block with mode
  arguments and hashes, `episode_complete.json` marker; `crm_worker.py` forwards `--crm-config` (`crm_collect` in the
  path) and appends the row's `extra` after `--horizon-s 120`, so a row can override the horizon.
- Splits/blacklist: not this module's job; the case's declared split is asserted and copied; the smoke case is a train
  group. No float16, no torch, no cluster paths.
- GPU: collector process 1,678-1,684 MiB in every run (< 8 GB); runs serialised under `/tmp/luffy_crm.lock`.

## Not verified here

Cluster-side behaviour (MI350X node determinism of a replayed prefix vs pass 2, the worker's `extra` rows, `G/source`
paths); the A4 anchor/label scripts; whether a low-speed continuation stall also occurs on rigid ground (the rigid
sibling's S4 continuation started at 2.99 m/s); B0/B3 task-row construction (unique perturbation seeds,
`--near-stop-all-modes` for the native arm).
