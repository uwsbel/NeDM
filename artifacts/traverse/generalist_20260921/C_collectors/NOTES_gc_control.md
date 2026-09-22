# Module note: `scripts/gc_control.py` (shared, torch-free controller helpers)

Written 2026-09-21 (first version 14:37, revised after verification the same evening) for PLAN stages B3
(external-control collectors, perturbed and policy modes) and A4 (moving-prefix branch collection). Both new
collectors import this module inside the Chrono process; it imports numpy only, plus the numpy-only route code of the
repo (`f104_n2_sampler`, `nedm.traverse.fdm_mppi`). torch is imported in exactly one function (the actor export) and
in the self-test. A subprocess in the self-test asserts that importing the module and calling every collector-side
entry point leaves torch unloaded.

## What is in it

1. `hold_clip(cmd, prev_steer, rate=0.1)`: the one clip per 50 ms frame. Steering is limited to the previous
   frame's held steering +- 0.1 and then to [-1, 1]; throttle and brake to [0, 1]. The returned triple of Python
   floats is what the collector writes into the manual `DriverInputs` at every physics substep of the frame, so the
   substep audit sees per-interval max-min = 0. Non-finite commands raise.

2. `OUPerturb(seed, dt=0.05, tau_s=0.5, steer_sd=0.15, throttle_sd=0.25, brake_p=0.05, brake_len_s=(0.5, 1.0))`:
   one call per frame around the shadow follower's command. Two Ornstein-Uhlenbeck states (steering, throttle)
   advanced with the exact discretisation, applied perturbation clipped to +- one standard deviation (steer within
   +-0.15, throttle within +-0.25, the plan's numbers; `bound_sds=2` gives a softer bound). Brake taps start with
   probability `brake_p` PER FRAME when no tap is active, last 0.5-1 s (10-20 frames), level uniform in [0.3, 1].
   Throttle and brake are never both positive: during a tap throttle is 0 and brake is max(follower brake, tap
   level); when the follower itself brakes, throttle stays 0 and the throttle noise is applied to the brake channel;
   otherwise the throttle is perturbed and brake is 0. One `default_rng(seed)` stream with a fixed draw order, so
   equal seeds replay identical perturbations; `reset()` replays from the seed; `summary()` reports taps and the
   braked fraction. The steering-rate clip is not inside: the collector passes the result through `hold_clip`.
   `OUPerturb.brake_p_for_rate(taps_per_s)` converts a wanted tap-start rate into the per-frame probability
   (renewal-process formula, see the docstring), so the collectors can hit the plan's onset sizing directly.

3. `NumpyActor` and `export_torch_actor(actor_critic, normalizer, meta, path)`: an rsl_rl `ActorCritic` actor plus
   its `EmpiricalNormalization` exported to one npz (`obs_mean`, `obs_var`, `obs_eps`, `W{i}`/`b{i}` in torch
   layout, `activation`, `action_center/scale/low/high`, `meta_json`) and evaluated in numpy as
   `clip(center + scale * tanh(mlp((obs - mean) / (sqrt(var) + eps))), low, high)`. That is rsl_rl's normaliser in
   eval mode (`(x - _mean) / (_std + eps)`, eps 1e-2) followed by `act_inference` and the tracker env's squash.
   `meta["action_center"]` may be the WP3 string `"dataset_mean"` with the mean in `meta["act_mean"]`. Supported
   activations: elu, relu, tanh, selu, lrelu, sigmoid, identity, crelu (rsl_rl maps it to CELU alpha 1). The
   steering-rate clamp is applied by the collector afterwards (`hold_clip`), as the WP3 evaluator did after the squash.

4. `sample_continuations(pose_xy_yaw, goal_xy, k, seed, speeds=(2, 4, 6), validator=None)`: k routes of the night-2
   route family from a moving-prefix pose to the goal. Base route = the planner's straight route
   (`gen_planner.base_route` copied line for line: route_00 Hermite first, then the wider/tighter Hermite starts and
   arc-then-straight fallbacks in the same order). Continuation i = one wide night-2 sample
   (`f104_n2_sampler.sample_one`, sine-basis lateral offset zero at both ends, four free-end speed knots) on a
   constant base cruise speed `speeds[i % len(speeds)]`, redrawn until the validator accepts it and the reference
   contract holds. Default validator = `validate_reference` with the arguments of `gen_planner.CFG` (max speed 6,
   curvature 0.125, half-extent 40, no obstacles, anchored at the pose). Start and end are asserted within 0.25 m of
   the pose and the goal (they are exact by construction). Raises `RuntimeError` when `64 k` draws do not give k valid
   routes (the caller drops the anchor). Output is the collector route format (`waypoints`, `speeds`, `stations`,
   `headings`, `meta` with `route_sha256`, `continuation`, `base_speed_mps`, `seed`, `branch_pose`, `goal_xy`);
   `route_to_json` turns it into the JSON the frozen `read_route` accepts.
   The seed must be anchor-specific (hash of episode id and cut frame): the lateral/speed draws depend on the seed
   alone, so two anchors with similar base routes and the same seed get near-identical continuation shapes.
   Why not import `gen_planner`: it imports `nedm.traverse.fdm_diverse_planner`, which imports torch at module
   level (checked). The validator config is rebuilt with the same arguments and the 20-line
   `check_reference_contract` is copied; the self-test asserts both agree with the originals.

5. `make_follower(veh_module, chrono_module, vehicle, route, height_fn, look_ahead=5.0, steer_gains=(0.8, 0, 0),
   speed_gains=(0.6, 0.05, 0), z_offset=0.5, initialize=True)`: the frozen collector's `make_driver`
   (`traverse_fdm_rgbd_diverse_chrono.py:98-112`) with `height_fn(x, y)` in place of `tmap.height`. The waypoint
   decimation (`follower_points`) is the same loop: keep a waypoint when its station is >= 2 m past the last kept
   one, always keep the last waypoint, z = ground + 0.5 m. `initialize=False` is for the branch swap:
   `ChClosedLoopDriver::Initialize` only adds a fixed body with the path's visual asset
   (`ChPathFollowerDriver.cpp:59-77`), which is not wanted mid-run; the controllers are already reset because the
   `ChPathFollowerDriver` constructor calls `Reset()` (`ChPathFollowerDriver.cpp:131-141`; it zeroes the PID errors
   and re-seeds the curve tracker at the sentinel, which at construction is the vehicle's own reference point since
   the look-ahead distance is still 0, `ChSteeringController.cpp:49`; `ChBezierCurveTracker::Reset` does a global
   closest-point search over the curve points, `ChBezierCurve.cpp:611-639`). The 5 m look-ahead set afterwards is
   used from the first `Advance`, the same order of operations as the settle-time `make_driver`. The new driver
   starts with throttle 0 and steering 0; steering continuity is enforced by the collector's clamp.

6. `PolicyObs` (+ `RouteTracker`, `history_blocks`): the tracking-policy observation, 158 values:
   `[0:3]` e_along/10, e_ct/10, e_h/pi to the nearest waypoint (search window idx-2..idx+39; the first call after a
   reset searches the whole route); `[3:33]` 10 preview points at 1 m spacing as (bx/10, by/10, v_ref/5) in the
   body frame; `[33:35]` vx/10 and yaw rate (state columns 0 and 6); `[35:38]` the last held action (action[k-1],
   (0, 0, 1) at frame 0). `[38:62]` the past 8 actions, action[k-8..k-1] oldest first, flattened row-major; `[62:158]`
   the past 8 observable states, state[k-7..k] on the 12 deployable columns (0-6, 11-14, 15), oldest first, the
   newest row being the current state; raw physical values unless `state_mean`/`state_std` are given. Pre-frame-0
   entries are the rest state (state[0]) and the settle action (0, 0, 1), the `fdm_data.build_history` convention.
   Per frame k the collector calls `observe(pose[k], state[k], action[k-1])`, drives with the held command, then
   `push(state[k], action[k])`. `seed_history(prefix_states, prefix_actions)` loads a recorded prefix for the branch
   swap. `history_blocks(states, actions, k)` is the offline twin for the tracker env; `layout()` returns the slice
   dictionary, `from_meta(route, meta)` rebuilds the helper from the `obs_layout` dict stored in the actor npz.

## How to run

    cd /home/harry/NeDM-traverse_mppi
    PYTHONPATH=src:scripts /home/harry/miniconda3/envs/nedm/bin/python scripts/gc_control.py --selftest \
        --out artifacts/traverse/generalist_20260921/C_collectors/selftest

Runs in about 3.5 s wall (3.1 s inside), no Chrono. Writes `selftest/gc_control_selftest.json`,
`selftest/selftest_actor.npz` and `selftest/selftest_actor_tanh.npz`. The recorded-pose check inside it reads
`artifacts/traverse/crm_f104_v1/demo_v1/run_*` (read-only) and is skipped when that directory is absent.

## What was tested (numbers from the final run)

- hold_clip: fixed cases (rate clip, box clip, rate 0, NaN and out-of-range previous steering raise) and 2,000 random
  commands: max steering step per frame 0.1000.
- OUPerturb: two instances with seed 7 on 20,000 follower commands give identical streams; seed 8 differs; `reset()`
  replays; no frame with throttle and brake both > 0; steering perturbation within +-0.15 and throttle within +-0.25;
  every tap drawn 10-20 frames, throttle 0 and brake >= 0.3 on every tap frame, no brake outside taps when the follower
  does not brake. Statistics at the defaults: steering perturbation sd 0.107 (after the clip), lag-1 autocorrelation
  0.88 (0.905 for the unclipped process), 31 % of frames at the steering bound, 594 taps in 20,000 frames (one tap
  start every 33.7 frames), braked fraction 0.53 on a stream where the follower itself brakes 15 % of the time.
  Chain perturbation -> `hold_clip` over 5,000 frames: max steering step 0.100, channels exclusive.
  `brake_p_for_rate(1.4 / 20 s)` = 0.00369 per frame; measured 1.425 tap starts per 400-frame episode over 200
  episodes (target 1.4).
- NumpyActor: a real `rsl_rl.modules.ActorCritic` (158 -> 512 -> 256 -> 128 -> 3, ELU, weights scaled x3) and an
  `EmpiricalNormalization` fitted on 4,096 random rows, exported and evaluated on 512 random observations:
  max |numpy - torch| = 3.3e-6 on the squashed action (5.1e-6 pre-squash), 33 % of outputs at the clip; a second
  tanh network: 2.0e-7. Single-observation and batched evaluation agree to 1e-12.
- sample_continuations: 4 fixed poses (one facing away from the goal that needs the arc fallback, one near the arena
  edge) x 3 continuations: deterministic per seed, 3 distinct hashes, all valid under the planner validator, contract
  OK, endpoints exact, 3 draws for 3 routes in every case; the base routes are bitwise equal to
  `gen_planner.base_route` for the 4 fixed poses and for 60 random pose/goal pairs over the arena (38 of them using a
  fallback shape); `planner_cfg()` equals `gen_planner.CFG` field by field; 12 routes pass
  `fdm_diverse_planner.check_reference_contract` and `gen_planner.safe_validate`; a cusped route is rejected by both
  contract checkers; a pose at the arena corner facing outward has no valid base route in either implementation and
  raises `RuntimeError`. Recorded moving-prefix poses: the 5 local CRM demo episodes cut at 2, 4 and 6 s (15 anchors,
  vx 1.3-3.6 m/s, 36-50 m to the goal): 3 valid continuations after at most 5 draws (3 draws in 8 of 15), ~3 ms each.
  A one-off check outside the self-test: 200 random pose/goal pairs (118 fallback shapes, 22 with no valid shape at
  all) bitwise equal to `gen_planner.base_route`, and 1,536 (pose, radius, long-way) arc-then-straight shapes bitwise
  equal to `gen_planner._arc_line` (the copy replaces the numpy-2-deprecated 2-vector `np.cross` by the explicit
  expression).
- make_follower: with stub `chrono`/`veh` modules recording every call, the call sequence (points, curve, constructor
  arguments, look-ahead, gains, Initialize) is identical to the frozen `make_driver` on a 96-point route (24 follower
  points); `initialize=False` drops exactly the final `Initialize` call. pychrono in the `nedm` env exposes `Reset`,
  `Initialize`, `SetDesiredSpeed`, `GetInputs` on `ChPathFollowerDriver` (checked by attribute, no simulation).
- PolicyObs: 120 frames of a noisy synthetic trajectory along a sampled route with random 17-column states: the
  streaming observation equals the WP3 evaluator's `RouteTracker` block plus the offline `history_blocks` to 0.0
  at every frame; frame-0 padding, partial padding at frames 1-7, branch seeding from a 40-frame prefix and from a
  3-frame prefix, `from_meta` round trip and normalised state block all asserted.
- Torch-free: a subprocess imports the module, samples continuations, clips, perturbs and builds a `PolicyObs`;
  `'torch' in sys.modules` is False.

## Known limits and notes for the collector authors

- `brake_p` is a PER-FRAME probability. Measured on 200 x 400-frame follower-like episodes: 0.05 -> 11.8 taps per
  episode and 46 % braked time; 0.01 -> 3.5 taps, 17 %; 0.0025 -> 1.02 taps, 8.6 %. The plan needs >= 2,000 brake
  onsets from 1,500 episodes per world (>= 1.33 per episode): pass `brake_p=OUPerturb.brake_p_for_rate(1.4 / 20.0)`
  (0.0037 per frame for 20 s episodes) or scale it to the actual episode length. Adjacent taps can follow each other
  without a gap (no refractory period).
- With `bound_sds=1` the OU steering perturbation sits at its +-0.15 bound ~31 % of the time; `bound_sds=2` (bound
  0.30, sd 0.15) is the softer alternative if the audit prefers fewer saturated frames. The subsequent `hold_clip`
  limits the per-frame steering change to 0.1 regardless.
- `sample_continuations` speeds are the base cruise speeds of the night-2 family (2, 4, 6 m/s cycled over the k
  continuations); the four speed knots move the profile by up to +-4 m/s within [0.5, 6]. The first speed of a
  continuation is not matched to the vehicle's current speed; the follower handles the step (a brake tap of one
  substep is possible when the vehicle is above the new target, see the `make_follower` docstring).
- The observation's past-state block is raw physical units by default; the rsl_rl empirical normaliser absorbs the
  scale. If the tracker env normalises the block, it must write `state_mean`/`state_std` into `obs_layout` so
  `PolicyObs.from_meta` reproduces it.
- `NumpyActor` uses float64; the torch policy runs float32, so the residual ~3e-6 is torch's rounding.
- `make_follower` cannot be exercised against Chrono without a scene; the equivalence test is at the call level with
  stubs. The branch-swap behaviour (Reset in the constructor, global tracker search, no body added) is documented from
  the C++ source (`/home/harry/chrono/src`), not from a run.

## Fix round 1 (2026-09-21, late evening; answers VERIFY_gc_control.md problems 1-4)

File after the fixes: `scripts/gc_control.py` sha256 `683f24ad3992...` (before: `44136eba83e5...`). Nothing else in the
repo was edited; the self-test was re-run into `selftest/fix1/gc_control/` (the earlier `selftest/` artefacts are kept
unchanged). Statements in the sections above that say "a brake tap of one substep" at the branch are superseded by
item 4 below.

1. **Speed floor from the vehicle's speed** (`sample_continuations(..., v0=None)`, lines 527-600). When `v0` is
   given, every draw's speed profile is set to `min(max(speeds, sqrt(max(v0^2 - 2 A_DEC (s - s0), 0))), 6.0)` with
   `A_DEC = f104_n2_sampler.A_DEC` (2 m/s^2) BEFORE the heading test, the planner validator and the contract check
   (`speed_floor_from`, line 520). A negative v0 (reversing vehicle) counts as 0. The element-wise max of two profiles
   that each respect the accel/decel limits respects them too, so floored routes stay planner-valid. Opt-in extra
   (`v0_accel_cap=True`, off by default; the later verifier note's second ramp): also cap at
   `sqrt(v0^2 + 2 A_ACC (s - s0))`, which makes the first speed exactly `min(v0, 6)`. The collectors pass
   `v0 = replayed/recorded vx at F`; `meta` gains `v0_mps`, `start_speed_mps`, `speed_floor_raised_points`,
   `v0_accel_cap`, `speed_cap_lowered_points`.
2. **Start-heading acceptance** (`max_start_heading_err_deg=15.0`, `start_heading_err_deg` line 515): a draw whose
   first heading is more than 15 deg from the vehicle's yaw is rejected inside the draw loop (counted as
   `start_heading` in the rejection reasons; `None` disables it). `meta` gains `start_heading_err_deg` and
   `max_start_heading_err_deg`. With both tests off and no v0 the sampler is bit-for-bit the earlier one (checked:
   the 4 synthetic cases and the 15 demo anchors reproduce the earlier self-test's draw counts, mean speeds and
   lengths).
3. **`OUPerturb` default** (lines 103-124, 157, 210): `brake_p` now defaults to `DEFAULT_BRAKE_P =
   brake_p_for_rate(1.4 / 20.0)` = 0.00369 per frame (the PLAN B3 sizing, ~1.4 taps per 20 s episode); the function
   is module-level and still reachable as `OUPerturb.brake_p_for_rate`. The docstring keeps the per-frame semantics
   and now also says that a tap starting while the follower already brakes is not a brake onset (count onsets from
   the recorded action). The self-test's 0.05 statistics are kept by passing `brake_p=0.05` explicitly.
4. **`make_follower` docstring** (lines 665-680): the follower keeps braking while `m_throttle <= 0.2` and the vehicle
   is faster than the target (not one substep; the verifier's rigid measurement is quoted), and two notes for the
   collector authors: `hold_clip` must REPLACE the frozen per-substep clamp, and the frozen route-start check (0.25 m
   from the layout start) must be bypassed for the branch leg. The module docstring's `hold_clip` and `make_follower`
   entries say the same.

Self-test after the fixes (`selftest/fix1/gc_control/gc_control_selftest.json`, 4.1 s, torch-free): every earlier
number unchanged (hold_clip 0.1; OU at 0.05: 594 taps, braked 0.5286; default p 0.003694 -> 1.425 taps per 400
frames; actor 3.28e-6 / 5.07e-6 / 2.02e-7; 64 base routes identical to gen_planner, 38 fallbacks, 12 contract
agreements; follower 96 -> 24 points; PolicyObs diff 0.0). New checks: floor and cap ramps on hand cases; the four
synthetic poses at v0 = 0, 3.3, 5, 7 m/s (floor only, and floor + cap) all planner-valid, contract-OK, first speed
>= min(v0, 6) (== with the cap), heading within 15 deg; the acceptance can be switched off; a tolerance nothing
meets raises.

Test of (1)-(2) on real anchors:

- The verifier's 15 anchors (5 local CRM demo episodes cut at 2, 4, 6 s; vx 1.3-3.6 m/s): before the fix 12/45
  continuations started below the vehicle's speed and 27/45 kinked more than 15 deg (the verifier's numbers); after
  the fix 45/45 start at or above v0 and 45/45 within 15 deg, draws for 3 routes 4-15 (before: 3-5).
- The A4 anchor list `A_adapt/a4/anchors/anchors_crm.json` (pose_F, vx_F, goal_xy; anchor-hashed seeds). Self-test
  subset of 15 anchors spread through the file (vx 0.2-3.0 m/s): before 9/45 below v0 and 27/45 over 15 deg; after
  45/45 and 45/45, draws 6-100 (one stalled anchor 16 m from its goal needed 100 of the 192 allowed). One-off over
  all 800 anchors (`/tmp/a4_all.py`, 16 s): before 510/2400 continuations below v0 and 1400/2400 over 15 deg; after
  798/800 anchors solved, 2394/2394 at or above v0 and within 15 deg, all planner-valid; draws median 12 (before 5),
  p90 36, max 159, 17 anchors above 64; 2 anchors (`group_0284_op_02@521`, `group_0848_op_04@229`) now fail within the
  192-draw budget because no draw meets the 15 deg tolerance (they should be dropped by the row builder, as the
  docstring says). With the opt-in cap: same solvability, first speed == v0 on 795/798 (the 3 others are reversing
  anchors, vx < 0, where the ramp starts at 0).

Not changed (verifier items 4-5 of the later note): the continuation seed must be anchor-specific (row builder's
job); `make_follower`'s argument order is unchanged (a mirrored call fails loudly).
