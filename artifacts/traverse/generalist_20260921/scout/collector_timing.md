# Scout map: Chrono collectors (rigid + CRM), action timing, PID follower, hooks

Read-only, 2026-09-21. `chrono.py` = `scripts/traverse_fdm_rgbd_diverse_chrono.py` (sha256 2996c567... = the frozen cluster source per `artifacts/traverse/crm_f104_v1/scout/episode_runner.md`). Chrono C++ read from `/home/harry/chrono/src`.

## 1. Physics step, substeps, and WHEN the logged action is sampled

- Rigid: vehicle step 2 ms, TMEASY tyre step 1 ms (`src/nedm/traverse/scene.py:58-59, 81`), 25 substeps per 50 ms frame (`chrono.py:195-196`).
- CRM: 1 ms from `configs/crm_main.json` (`step_s 0.001`; verified in `demo_v1/run_145/outcome.json`, `physics_dt_s 0.001`), 50 substeps (`scripts/crm_collect.py:219-221`). The in-script default `CRM_DEFAULT` says 5e-4 (`crm_collect.py:38`) but was overridden by `--crm-config`. RIGID_MESH FSI tyres, chassis uncoupled (`:178-179`).

Both loops are statement-for-statement the same. Per frame: `wp = nearest_index`, `at_end`, `driver.SetDesiredSpeed(0 if frame<0 or at_end else speed[wp])` once (`chrono.py:214-218`, `crm_collect.py:229-233`). Per substep: `driver.Synchronize(ts)`; `inputs = driver.GetInputs()`; steering clamped to `previous_steer +- 2*dt` (forced 0 in the settle); `hmmwv.Synchronize(ts, inputs, terrain)`; at `sub == 0` the row is captured with `action = [m_steering, m_throttle, m_braking]`; then `driver.Advance(dt)` and the physics advance (`chrono.py:220-235, 285-287`; `crm_collect.py:235-246, 273-274`).

Chrono source facts that fix the timing: `ChDriver::Synchronize` is a no-op (`chrono_vehicle/ChDriver.h:66`); the PID output is computed in `ChClosedLoopDriver::Advance(step)` (`driver/ChPathFollowerDriver.cpp:84-114`); `GetInputs()` returns a copy (`ChDriver.cpp:36`), so the external clamp never feeds back. Hence **`action[k]` is the PID output of the LAST `Advance` of interval k-1** (time k*DT - dt, that instant's state, interval k-1's desired speed). The desired speed of interval k first acts at substep 1. `action[0]` is the settle's last output (target 0, steering 0). Inside interval k the output is refreshed 25x/50x; steering drift per interval is bounded at 0.1 by the clamp, throttle/brake have no rate limit and the exclusive throttle/brake switch (`ChPathFollowerDriver.cpp:97-108`, threshold 0.2 at `:41`) can flip mid-interval.

Net drift is already measurable offline because `action[k+1]` is the output at the END of interval k. On local files (5 CRM demo, 48 rigid demo, 1,080 rigid moving_v1 episodes): |dsteer| p50/p90/p99 = 0.004-0.007 / 0.02-0.04 / 0.10 (the clamp binds in ~1 % of frames); |dthrottle| p50/p90/p99 = 0.00-0.01 / 0.03-0.05 / 0.10-0.11, max 1.0; throttle<->brake flips in 1.0-1.6 % of frames. So a 50 ms hold of `action[k]` is within 0.1 of the applied signal for ~99 % of intervals; the tail is where recorded transitions are not hold-transitions.

Full substep log, zero repo patch:
- rigid: the frozen loop calls `args.frame_observer.on_substep(scene, frame, sub, dt, action3)` every substep with the clamped inputs (`chrono.py:279-282`); `RichTelemetry.on_substep` discards `action3` (`src/nedm/traverse/fdm_rich_telemetry.py:225-236`; no action keys in any `rich_intervals.npz`). Sketch: wrap `gen_collect.make_observer(...)`, append `(frame, sub, action3)` in `on_substep`, save `substep_actions.npz` (N x 25 x 3) in `finish`, run ~20 episodes, report per-interval max-min and mean minus `action[k]` per channel.
- CRM: no substep hook, but `run()` calls `frozen.make_driver` (`crm_collect.py:190`): pre-import the frozen module and swap `make_driver` for one returning a proxy whose `GetInputs()` logs; or add a one-line `substep_hook` beside `frame_hook` (`:266-268`). `crm_collect.py` is not hash-frozen (`:439` only records its sha).

## 2. The follower

`make_driver` (`chrono.py:98-112`, reused at `crm_collect.py:190`): waypoints subsampled to >= 2 m spacing, z = BMP + 0.5 m, `ChBezierCurve(points)`, `ChPathFollowerDriver(vehicle, curve, "route", speeds[0])`, look-ahead 5 m, steering gains (0.8, 0, 0), speed gains (0.6, 0.05, 0), `Initialize()` before the settle, `zero/ramp_duration = 0` (`ChPathFollowerDriver.h:121-122`).
- Steering: sentinel 5 m ahead, target = closest curve point, signed horizontal error in metres, output 0.8*err clamped to [-1, 1] (`utils/ChSteeringController.cpp:138-176`), then the collector's 2/s clamp.
- Speed: signed forward velocity component (`utils/ChSpeedController.cpp:50`), PID (`:110-129`); `out>0 and target>0` -> throttle=|out|; elif throttle>0.2 -> throttle=1-|out|; else brake=|out| (`ChPathFollowerDriver.cpp:94-108`). Target 0 (settle, parking) brakes in proportion to forward speed: ~0 at rest, 1 at 4 m/s.
- Desired speed `speed[wp]`, `wp` = monotone argmin over the next 60 waypoints (`chrono.py:115-117`); parking `wp >= len-2 and |pos-xy[-1]| < 3 m` (`:216-218`) vs goal radius 2.5 m (`:308`).
- Settle 0.8 s = 16 negative frames, nothing recorded (`chrono.py:29, 197`). `build_history` pads pre-anchor actions with (0, 0, 1) (`src/nedm/traverse/fdm_data.py:87-90`), not the settle's real command.

## 3. External 50 ms zero-order-hold mode (smallest patch)

Template: `scripts/traverse_wp3_chrono_eval.py:332, 405-437, 457-470` captures the state at the top of the frame BEFORE `Synchronize`, computes `cmd`, writes a `veh.DriverInputs()` at every substep, and records at sub 0 after `Synchronize` exactly as the collectors do. Positions, velocities and omegas are unaffected by `Synchronize`; tyre-force and engine-torque fields lag one substep in the pre-Synchronize capture; document, do not reorder.
- CRM: add `args.control_hook`. Before the substep loop, if `frame >= 0`: `cmd = control_hook(frame, sim_t, row_pre, wp, desired_speed, last_cmd)`; in the loop replace the `inputs` block with a `DriverInputs` holding `cmd` for all 50 substeps, steering clipped to `previous_steer +- 0.1`; keep `driver.Synchronize/Advance` running as a shadow so settle and parking stay byte-identical. Recording (`:242-262`), termination (`:277-306`) and schemas unchanged; tag `outcome["driver"]["control"]`.
- Rigid: the block `driver.Synchronize(ts)\n inputs = driver.GetInputs()\n` occurs once in `run_chrono` (verified). `gen_collect.adapted_function` (`scripts/gen_collect.py:166-190`) is insertion-only with asserted counts, so put a sibling replacement adapter in a new wrapper next to it and record a new `mutation_scope`.
- Caveat: the blockage stop needs throttle > 0.3 (`gen_collect.py:129`); a low-throttle stalled tracker runs to the 120 s timeout.

## 4. Prefix-then-branch

Existing hooks cannot do it: `frame_observer.on_frame` (`chrono.py:271-276`), `frame_hook`/`scene_hook` (`crm_collect.py:192-193, 266-268`) and `StopPolicy.check` (`gen_collect.py:120-154`) receive values only; `driver, xy, speed, goal, wp` are loop locals. Patch (~10 lines at the top of the frame loop, both worlds): at `branch_frame` build `driver = make_driver(branch_route)`, swap `xy, speed, goal`, set `wp = 0`, keep `previous_steer`. Do NOT bake the continuation into the initial route: `ChBezierCurve(points)` solves a global linear system (`chrono/core/ChBezierCurve.cpp:87-140`), so a different tail perturbs the prefix curve. Call `Reset()` on the branch driver (public, `ChPathFollowerDriver.h:64`; resets sentinel and integrators, `ChSteeringController.cpp:57-67`) and skip `Initialize()`, which adds a fixed body (`ChPathFollowerDriver.cpp:59-63`). The new driver starts with `m_throttle = 0`, so expect a one-substep brake tap if the vehicle is above the new target.
- CRM: fresh soil per process, so cost = K x (prefix + branch) at RTF ~0.5 on MI350X (run_145: rtf 0.509, build 3.4 s, 4.0 M particles); bit-identical across MI210/MI300X/MI350X (`crm_f104_v1/LOG.md:11`, `REPORT.md:48`), so prefixes are exactly shared across jobs.
- Rigid: deterministic per node only (memory `amd-not-reproducible-across-nodes`): run all branches of one anchor on one node.
- Night-2 moving starts were NOT prefix replays: `scripts/n2_moving_tasks.py:35-38` spawns at the route point; `scripts/rigid_moving_collect.py:24-33` uses `SetInitFwdVel(v0)` and a 0.1 s settle. `ChVehicle::Initialize` ignores the forward velocity (`ChVehicle.cpp:163-168`); only the chassis receives it (`HMMWV_VehicleFull.cpp:146`), wheels do not spin up. Measured at frame 0 over the 1,080 `moving_v1` drives: v0 = 4 -> vx 2.93 m/s, front-left omega 0.03 rad/s (rolling: 8.6), `action[0]` brake 0.99; v0 = 2 -> vx 1.38, omega 0.21, brake 0.84. The docstring's "free settle" is a full PID brake on locked wheels; those anchors are skidding vehicles.

## 5. The 17 state fields and observability

`STATE_FIELD_PRESETS["tire_normal_force_omega_pt"]` (`src/nedm/training/constants.py:3-11, 31-40, 44-47, 54`), filled by `capture_row` (`src/nedm/hmmwv_data.py:438-544`) and `crm_tire_fields` (`crm_collect.py:137-155`): 0-1 body vx, vy (INS/odometry, observable); 2-6 roll, pitch, roll rate, pitch rate, yaw rate (IMU, observable); 7-10 world-Z tyre force (TMEASY `ReportTireForce` / FSI `GetFsiBodyForce`), simulator-only; 11-14 spindle omega (wheel-speed sensors, observable); 15 engine speed (CAN, observable); 16 motorshaft torque, simulator-only. Pose (x, y, yaw) and the applied action are observable. `crm_extra.npz` (pos_z, quaternion, spindle z, slip ratio, FSI fx) is simulator-only except the quaternion.

## 6. Actuator limits and action ranges

Steering [-1, 1] -> Pitman-arm +-30 deg (`HMMWV_PitmanArm.cpp:35`); throttle [0, 1] -> engine map peak 793 Nm at 1600 rpm (`HMMWV_EngineShafts.cpp:34-53`); brake [0, 1] -> 4000 Nm per wheel (`HMMWV_BrakeSimple.cpp:29`, type SIMPLE `HMMWV.cpp:61`); tyre radius 0.467 m. Collector: steering rate 2/s (0.1 per frame); PID never has throttle and brake > 0 together (0/361 frames in run_145). The tracker env uses the same box and rate limit (`src/nedm/traverse/tracker_env.py:64-68`), as does its Chrono eval (`traverse_wp3_chrono_eval.py:134-141`).

## Gaps / unknowns

- The intermediate substep path inside an interval is in no file; the offline numbers are net drift. The substep log closes this.
- pychrono is not importable in the system python here; `Reset()`/`SetThrottle` on the branch driver are asserted from the C++ header only.
- Effect of adding a body mid-run on CRM/FSI stepping untested (the recommendation avoids it).
- No CRM moving-start data exists; moving_v1 is rigid only.

## What must change for the plan

- B step 2: plan wording is correct; recorded `(state[k], action[k]) -> state[k+1]` used an action applied only at substep 0 then refreshed. Quantify with the substep log, then collect new hold-mode data (section 3) or filter old frames by `|action[k+1]-action[k]|` (~1 % throttle steps, 1-1.6 % mode flips).
- A step 4: implement the driver swap of section 4; do not reuse `rigid_moving_collect.py`.
- Both modes touch two files: `crm_collect.py` (not frozen) and a new rigid wrapper beside `gen_collect.py`; `crm_worker.py` already selects the collector via `CRM_COLLECTOR` and forwards per-task `extra` args (`scripts/crm_worker.py:91-98`).
