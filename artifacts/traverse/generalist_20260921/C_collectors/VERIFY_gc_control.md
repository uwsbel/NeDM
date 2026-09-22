# Verification: `scripts/gc_control.py` (adversarial re-check, 2026-09-21 late evening)

Verdict: **pass with issues**. The module (unchanged since 17:26, sha of the version the earlier 17:41 note examined)
reproduces every claimed number bit-for-bit, its copies of the route code are exact, the observation/history/timing
conventions match PLAN.md and the frozen code, the actor export is torch-free and exact on three real trained WP3
checkpoints, and the follower swap has real Chrono evidence. Two design hazards in the A4 continuation sampler
(speed step and heading kink at the branch) remain open; one docstring claim about the swap is wrong; one default is a
misreading of the plan that both collectors already override. Nothing in the repo was modified; scratch outputs are
under `/tmp/gc_verify2` (this pass) and `/tmp/gc_swap` (the earlier pass, re-used read-only).

## Reproduced

- Self-test re-run into `/tmp/gc_verify2` (3.6 s wall, 3.19 s inside, no Chrono): `gc_control_selftest.json`
  identical to the implementer's file apart from `runtime_s` and the npz path; `selftest_actor.npz` and
  `selftest_actor_tanh.npz` byte-identical (`cmp`). Numbers: hold_clip max step 0.1000; OU 594 taps / 20,000 frames,
  braked 0.5286, sd 0.107, lag-1 0.882, 31 % at the bound, taps 10-20 frames, chain max step 0.100,
  `brake_p_for_rate(0.07/s)` = 0.003694 -> 1.425 taps per 400 frames; actor 3.28e-6 / 5.07e-6 / 2.02e-7, 33 % clipped;
  64 base routes identical, 38 fallbacks, 12 contract agreements, 15 recorded anchors solved with <= 5 draws;
  96 -> 24 follower points, identical call sequence, `initialize=False` drops only `Initialize`; PolicyObs diff 0.0;
  `torch_free_import` True.
- **Stronger actor check (new)**: `export_torch_actor` on the three real trained checkpoints under
  `artifacts/traverse/wp3_tracker_v1`, `wp3_tracker_v1_s2` (38 obs) and `wp3_tracker_hist4` (98 obs), compared with
  `traverse_wp3_chrono_eval.PolicyController.act` (rate limit off) on 2,000 observations each drawn around the
  normaliser's own mean/sd: max |da| = 8.7e-7, 8.0e-7, 1.5e-6 (30-35 % of outputs at the clip). The `_std` buffer
  check, `eps` = 0.01, `action_center = "dataset_mean"` and `nn.Sequential` iteration all work on real files.
- rsl_rl in the nedm env: `EmpiricalNormalization.forward` = `(x - _mean) / (_std + eps)`, `eps` 1e-2, `_std` a
  buffer; `resolve_nn_activation` maps exactly the eight names in `_ACTIVATIONS` (crelu -> `CELU()` alpha 1 == ELU;
  lrelu -> `LeakyReLU()` slope 0.01); `ActorCritic.actor` is an `nn.Sequential` of Linear/ELU; `act_inference` is the
  plain forward. The export path matches.
- Copies read side by side: `check_reference_contract` verbatim from `fdm_diverse_planner.py:16-37`; `constant_speed`,
  `_hermite`, `safe_validate`, `base_route` match `gen_planner.py`; `_arc_line` differs only in the 2-vector
  `np.cross` replacement (same sign convention); `_hermite(scale=1)` is the same expression as
  `propose_route_families` (`fdm_diverse_planner.py:248-278`, `t` grid `max(33, ceil(L/0.5)+1)`, `np.gradient`
  headings). `planner_cfg()` uses the four `gen_planner.py:30` arguments; the self-test asserts `asdict` equality.
  Both `gen_planner` and `fdm_diverse_planner` import torch at module level, so the copies are justified.
- **Route hash contract (new)**: 45 continuations written with `route_to_json`, read back with the frozen
  `traverse_fdm_rgbd_diverse_chrono.read_route`, then `route_sha256(read-back) == meta["route_sha256"]` in 45/45 (the
  A4 labels join on this hash; `route_sha256` is the expression of `f104_n2_iter.route_sha256`).
- State preset `tire_normal_force_omega_pt` (`src/nedm/training/constants.py`): columns 0-6 vx, vy, roll, pitch, roll
  rate, ang_vel_body_y, yaw rate; 7-10 tyre Fz; 11-14 spindle omegas; 15 engine speed; 16 torque. `OBSERVABLE_COLS`
  = (0-6, 11-14, 15) is right; `VX, YAW_RATE = 0, 6` (`nrd_model.py:30`) match `VX_COL, YAW_RATE_COL`.
- History convention: `history_blocks` = state[k-7..k] with action[k-8..k-1], padding state[0] and (0, 0, 1), the same
  as `fdm_data.build_history` (`idx = anchor-H+1..anchor`, actions at `idx-1`, `ac[idx <= 0] = (0, 0, 1)`) and PLAN A1.
  `[0:38]` equals `tracker_env._compute_observations` and the WP3 evaluator block (self-test diff 0.0 re-run here).
  The tracker-env fork `gb_tracker_env.py:361-369` now seeds `state_hist8` = s-H+1..s, `act_hist8` = s-H..s-1 and
  `last_actions = a8[:, -1]` (action[s-1] or the settle action), i.e. the same convention as `PolicyObs`, and carries
  `check_against_policy_obs`; the earlier cross-module note on `act_raw[start-1]` seeding is resolved in the fork.
- Timing: the collectors capture the row at sub 0 after `Synchronize` (`traverse_fdm_rgbd_diverse_chrono.py:214-232`,
  `crm_collect.py:229-246`); `ChDriver::Synchronize` is a no-op and the PID output comes from `Advance`, so `action[k]`
  = last `Advance` of interval k-1 (scout `collector_timing.md`). The module docstring states exactly this. The
  `hold_clip` rate 0.1 per frame = the frozen `previous_steer +- 2*dt` clamp accumulated over 25 x 2 ms / 50 x 1 ms.
- Collector-side use (read, not run): `crm_collect_ext.py:159-233, 281` and `gen_collect_ext.py:176-237` call
  `hold_clip(raw, last_action[0])` once per frame, write the identical triple at every substep in place of the frozen
  clamp (`previous_steer = held[0]`), reset `PolicyObs` with the pre-Synchronize state at frame 0, push
  `(state[k], action[k])` after the frame, and override `brake_p` with `OUPerturb.brake_p_for_rate(1.4/20)`
  (`crm_collect_ext.py:542-543`, `gen_collect_ext.py:378`). Branch swap at the top of frame F with
  `len(record_state) == F`, `wp = 0`, `previous_steer` carried for the per-substep clamp (`crm_collect_ext.py:184-199`).
- Chrono C++ (`/home/harry/chrono/src/chrono_vehicle/driver/ChPathFollowerDriver.cpp`, `utils/ChSteeringController.cpp`,
  `core/ChBezierCurve.cpp`, `utils/ChSpeedController.cpp`): `ChPathFollowerDriver` constructor calls `Reset()`;
  `ChSteeringController::Reset` puts the sentinel at `m_dist` (0 at construction) and `ChBezierCurveTracker::Reset`
  sorts all curve points by distance (global search); `ChClosedLoopDriver::Initialize` only adds a fixed body with a
  visual shape; `ChSpeedController::Reset` zeroes the PID errors. All docstring citations confirmed except the
  brake claim in problem 1.
- **Rigid Chrono swap test re-run (1 rigid run, no CRM, no cluster)**: the definitions of `/tmp/gc_swap/swap_test.py`
  exec'd and `run("module_as_is")` called once (HMMWV_Full via `create_hmmwv`, TMEASY, flat `RigidTerrain`, frozen-loop
  semantics, straight route at 4 m/s, swap at frame 120 to `base_route(pose, (36, 8))` at 2 m/s with
  `make_follower(initialize=False)`); result `/tmp/gc_verify2/swap_rerun_module_as_is.json` is identical to the
  earlier `/tmp/gc_swap/result.json`: bodies 21 -> 21 across the swap (22 with `initialize=True`), steering jump
  -3.7e-4 and per-substep step <= 0.004 (the clamp), the vehicle follows the new route, deterministic.
- Determinism and numerics: `OUPerturb` and `sample_continuations` are pure functions of their seeds (equal-seed
  streams identical, `reset()` replays); single vs batched actor evaluation agrees to 1e-12; no float16 anywhere
  (float64 weights and observations); non-finite commands raise, non-finite observation entries are zeroed and counted
  as the env does. No split/blacklist logic lives in this module.

## Problems

1. **[major] Wrong docstring claim; systematic brake burst (or throttle step) at the branch.** `make_follower` and the
   NOTES say "a one-substep brake can occur if the vehicle is above the new target speed". `ChClosedLoopDriver::Advance`
   (`ChPathFollowerDriver.cpp:94-108`) enters the brake branch when the vehicle is above target and `m_throttle <= 0.2`;
   the new driver starts with `m_throttle = 0` and the brake branch sets `m_throttle = 0` again, so it re-enters itself
   at every substep until the vehicle is below the target. Measured in the rigid re-run (vx 4.01 m/s, continuation v0
   2.0): brake 0.96-1.0 for 0.502 s, 251 braking substeps in 3 s, vx 4.0 -> 1.8 m/s within 2 s. On the 15 real CRM
   anchors (45 continuations from `sample_continuations` with anchor-hashed seeds): 12/45 start with a target below
   the vehicle speed (gap median 1.24, max 2.31 m/s -> brake burst), 33/45 start above it (median +2.63, max +4.68 m/s
   -> `Kp * err` saturates, full throttle from a moving state on soft soil). The 1 m / 3 s grace window hides this from
   the event clock, not from the physics or the goal-reached-from-branch label. Smallest fix: add `v0=None` to
   `sample_continuations`; when given, after `S.sample_one` pin the profile to the vehicle speed with the sampler's own
   ramps, `v = max(v, sqrt(max(v0^2 - 2 A_DEC s, 0)))`, `v = min(v, sqrt(v0^2 + 2 A_ACC s))`, `v = clip(v, 0, 6)`,
   `v[0] = v0` (`s` = station from the branch, `A_ACC, A_DEC = f104_n2_sampler.A_ACC, A_DEC`), then validate as now;
   the collectors pass `v0 = state[F, 0]` (rigid: `float(pts[0])` of the replay). Checked here: all 45 pinned
   continuations stay planner-valid and contract-OK with a first-speed mismatch of 0.0. Fix the docstring either way.
2. **[major] Heading kink at the branch.** The night-2 lateral basis `sum a_j sin(j pi f)` is zero at both ends but its
   slope at f = 0 is `sum a_j j pi / L`; at the curvature budget that is a 58-67 deg heading change for L = 40-60 m.
   `validate_reference` (`fdm_mppi.py:121-168`) uses the anchor only to slice the route and never compares the start
   heading with the anchor yaw, so nothing rejects it. Measured on the 45 continuations: base route start heading =
   vehicle yaw (0.0 deg), continuation |heading[0] - yaw| median 21.5 deg, p90 35 deg, max 41.8 deg, > 15 deg in 27/45,
   > 30 deg in 10/45. At 2.5-3.6 m/s the 5 m sentinel then sees ~1-1.5 m of lateral error and the PID commands large
   steering within the first frames, again a branch artefact confounding the labels. The family is what PLAN A4 names,
   so this is a design hazard, not a contract breach. Smallest fix: a `max_start_heading_err` acceptance test in
   `sample_continuations` (`abs(wrap(headings[0] - pose[2])) <= tol`), or pass it through the existing `validator`
   hook. Checked here with tol 15 deg through the hook: 15/15 anchors solved, 4-15 draws for 3 routes (median 7),
   within the `64 k` budget.
3. **[minor] `OUPerturb(brake_p=0.05)` default is the wrong reading of the plan.** A bare `OUPerturb(seed)` gives
   ~12 taps and 46 % braked time per 20 s episode (the note documents it; both collectors override with
   `brake_p_for_rate(1.4/20)`, so no data is affected). Smallest fix: default `brake_p` to
   `OUPerturb.brake_p_for_rate(1.4 / 20.0)` or make it required. Two small related notes: the renewal cycle is
   `1/p + L - 1` frames, not `1/p + L` (exact p = 0.00368 vs 0.00369; measured 1.417 taps per 400 frames over 4,000
   episodes, target 1.40, harmless); and a tap that starts while the follower already brakes (15 % of frames in the
   self-test stream) is not a brake onset, so the onset yield is ~85 % of the tap rate; the collectors must count
   onsets from the recorded action, not taps.
4. **[minor, cross-module] The continuation seed must be anchor-specific and nobody enforces it yet.** The sampler's
   lateral and speed draws depend on the seed only (the docstring says so); `gen_collect_ext.py:640` takes
   `--cont-seed` verbatim from the task row and the `continuations` stage of `ga_branch_anchors.py` is not written
   (`ga_branch_anchors.py:357-358`, "after the pending gc_control fix"). The row builder must derive the seed from
   (episode id, F), as the self-test does (`md5("run:F")`). No change needed in gc_control; one assertion in the row
   builder (distinct seeds across anchors) would close it.
5. **[minor, note]** `make_follower(veh_module, chrono_module, ...)` reverses the frozen `make_driver(chrono, veh, ...)`
   argument order. A mirrored call fails loudly (`AttributeError` on `ChPathFollowerDriver`), and both collectors call
   it correctly, so nothing to fix; a keyword-only signature would remove the trap.

## Not a problem (checked)

- Branch-frame indexing: `seed_history(states[:F], actions[:F])` + `observe(pose[F], state[F], action[F-1])` equals
  `history_blocks(k=F)`; the CRM collector swaps at the top of frame F with exactly F recorded prefix frames, so the
  continuation is sampled from the pose at the top of frame F, the same instant the self-test uses (`pose_rec[F]`).
- Observable columns are unaffected by `Synchronize` (positions, velocities, omegas, engine speed), so the policy's
  pre-Synchronize current state and the post-Synchronize pushed history rows are the same numbers; the CRM collector
  measures `pre_capture_max_diff` on those columns to prove it per run.
- `OUPerturb` exclusivity, bounds, tap lengths, draw order and `reset()` are as documented; the steering perturbation
  applies during taps too (intended). With `bound_sds=1` 31 % of steering frames sit at +-0.15 (documented alternative
  `bound_sds=2`).
- `hold_clip` validates `prev_steer` in [-1, 1] and rejects NaN; a `rate` of 0 freezes steering (used by nothing).
- `NumpyActor` accepts only the eight rsl_rl activations, rejects ELU/CELU/LeakyReLU with non-default parameters,
  and refuses a normaliser whose `_std` is not `sqrt(_var)` (no silent guess).
- `route_to_json` produces lists only; nested `meta` (branch pose, goal, base meta) survives the frozen reader.
- Unsolvable poses raise `RuntimeError` with the rejection reasons; a validator that rejects everything raises.
