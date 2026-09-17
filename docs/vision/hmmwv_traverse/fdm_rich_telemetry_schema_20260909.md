# Rich FDM collection telemetry

The passive observer in `src/nedm/traverse/fdm_rich_telemetry.py` records additional physical responses for later energy, stability and traction losses. It reads getters only: it never synchronizes, advances, renders, adds bodies or changes controls. The old collector and its frozen results remain unchanged.

## Attachment and alignment

```python
observer = RichTelemetry(out, case_path=case, record_dt_s=0.05)
# After driver/terrain/vehicle Synchronize, before the first Advance of frame i:
observer.on_frame(scene, i, state17, pose3, action3,
                  command_context={"desired_speed_mps": desired, "parked": parked})
# For each physics step, after Synchronize and before Advance:
observer.on_substep(scene, i, sub, physics_dt_s, action3)
# After driver/terrain/vehicle Advance:
observer.on_post_substep(scene, i, sub)
# After measuring the real terminal state; no extra Advance:
observer.finish(scene, N, terminal_state17, terminal_pose3, last_action3)
```

Settling frames below zero are excluded. `rich_telemetry.npz` has **N+1** measured states, including the actual endpoint; `rich_intervals.npz` has **N** rows covering `[state i, state i+1)`. The state timestamp comes directly from Chrono. Row i records the command applied at that pre-interval sample; the terminal command repeats the last applied command and is not a new control interval. Optional command context is absent at the terminal endpoint.

Every numeric field is a float64 vector. `rich_telemetry.json` stores field units, getter/formula provenance, finite/missing counts, getter errors, observer/case hashes and full scene configuration. Configuration—including friction—is **label/provenance only**, not an automatic model input. Unknown getters yield **NaN with capability metadata**, never zero supervision. A complete physics-step sequence is checked against both interval endpoints; incomplete coverage fails collection.

## Measured sample fields

| Group | Main fields and units | Interpretation |
|---|---|---|
| Powertrain | engine motor speed (rad/s), engine motorshaft torque (N m), transmission motorshaft speed (rad/s), transmission driveshaft torque (N m), driveline driveshaft speed (rad/s), gear | Raw factors preserve the distinction between engine rotor and transmission interfaces. |
| Attitude/motion | quaternion, roll/pitch/yaw (rad), native Euler rates (rad/s), body/world angular velocity, world angular acceleration, REF position/velocity, COM acceleration | REF position and COM acceleration are explicitly distinguished. Angular velocity components are not silently relabeled Euler derivatives. |
| Control | steering, throttle, braking; optional desired speed and parked flag | Actual applied effort is separate from desired speed. |
| Each wheel | native longitudinal slip, slip/camber angles, deflection, radius, world tire force/moment, spindle pose/velocity/spin, axle speed and spindle torque | Native signs are retained. World-Z tire force is **not** terrain-normal load. |
| Derived wheel diagnostics | circumference speed, horizontal heading speed, horizontal slip speed/ratio, heading force | Useful diagnostics with declared approximations on grades; not substitutes for native tire slip. |
| Suspension | double-wishbone spring force/length/deformation and shock force/length/velocity | Typed scalar getters avoid the AMD build's unsupported `vector<ForceTSDA>` binding. Other suspension types remain missing with metadata. |
| Contact | chassis world resultant force/torque, largest asset-body resultant | These are body resultants, not identified contact pairs or tire support forces. |

## Mechanical energy quantities

All powers are in kW; interval work is in kJ. The observer saves signed, positive and negative-magnitude work for each channel. Positive work is `sum(max(power, 0) * physics_dt)`; negative work magnitude is saved separately, without assuming regeneration.

| Prefix | Matched factors | Meaning |
|---|---|---|
| `engine_interface` | engine `GetOutputMotorshaftTorque` × transmission `GetOutputMotorshaftSpeed` | Engine/transmission interface mechanical power. The transmission getter reports feedback speed **to the engine**, not wheel-side speed. This matches the previous runner definition. |
| `engine_rotor` | engine `GetOutputMotorshaftTorque` × engine `GetMotorSpeed` | Engine torque × rotor speed; retain separately because dynamic rotor and feedback speed can differ. |
| `driveshaft` | transmission `GetOutputDriveshaftTorque` × driveline `GetOutputDriveshaftSpeed` | Transmission/driveline interface mechanical power. |

There is **no measured fuel consumption**: the configured engine API supplies mechanical torque/speed, not established fuel flow. Engine mechanical energy should not be called fuel/chemical energy. The Chrono 10 headers (`ChEngine.h`, `ChTransmission.h`, `ChDriveline.h`) define these interface directions; AMD source `ChPowertrainAssembly.cpp` pairs engine torque and motorshaft feedback speed.

The 500 Hz hooks integrate every synchronized pre-step power sample using actual step duration. This is solver-step quadrature, not continuous-time exact integration. Attaching only the original 20 Hz frame hook remains supported, but `solver_step_work=0` explicitly labels the coarse left-endpoint estimate. `--require-solver-steps` rejects such data for the enriched main cohort.

## Interval targets and future horizon labels

`rich_intervals.npz` includes the three channels' `{positive,signed,negative_magnitude}_work_kj`, mean power, `max_abs_roll_rad`, `max_abs_pitch_rad`, maximum chassis/asset contact resultants, endpoint maximum absolute native wheel slip and minimum world-vertical tire force. Record-level wheel summaries use only the two measured endpoints. Attitude/contact extrema include every post-physics state only when `on_post_substep` is attached; coverage counts are saved.

For a future prediction window starting at i and ending at j, sum work over intervals `[i,j)` and take attitude/contact maxima over those same intervals. Never include interval j. Save target-validity masks when any required measurement is missing. Use measured motion plus effort and deliberate-parking exclusions for stall labels; high slip alone is not a stall label. Load/slip histories allow later slip-duty and wheel-unloading definitions without recollection.

Recommended first new supervised outputs are cumulative positive engine-interface work and peak absolute roll/pitch alongside the existing progress/contact/stall formulation. Keep raw torque/speed, native slip and suspension responses even before decoder heads use them. Rigid heightfields with TMeasy demonstrate tire/vehicle dynamics; they do **not** simulate deformable-soil sinkage or soil constitutive terramechanics.

## Verification

```bash
python scripts/check_traverse_fdm_rich_telemetry.py --self-test
python scripts/check_traverse_fdm_rich_telemetry.py EPISODE_DIR \
  --trajectory EPISODE_DIR/trajectory.npz --require-solver-steps
```

The standalone contract check covers variable physics-step durations, signed/positive/negative work, missing-getter propagation, post-step event maxima, coarse-integration labels, excluded settling and rejection of incomplete intervals. The recording check validates N/N+1 timing, metadata coverage, work identities and agreement with the collector's pose/action/power/work arrays. Neither substitutes for an actual Chrono observer-on/off trajectory parity run.

The initial immutable pilot uses `fdm_rich_telemetry_v1`: power, slip, load, attitude and contact recorded successfully; its generic suspension-vector getter was unsupported and was explicitly recorded as unavailable. The revised `fdm_rich_telemetry_v2` uses `CastToChDoubleWishbone` and six scalar spring/shock getters per wheel. Pilot source/results are retained, and the full cohort must use a new frozen collection snapshot. This is a telemetry-binding correction, with no changed controls or physics parameters.
