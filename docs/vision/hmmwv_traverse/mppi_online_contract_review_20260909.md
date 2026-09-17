# Independent online FDM/MPPI review

The online loop uses a fixed pre-drive RGB-D map, current measured state/pose and causal applied-control history to rank new geometric route/speed families. The native PID adapter updates the route against the current vehicle frame while preserving its speed PID. Model work and arrival beyond the prediction horizon are explicitly extrapolated planning costs; measured full-run work, goal arrival and physical hazards determine evaluation.

## Verified CPU contracts

The independent [checker](../../../scripts/check_traverse_fdm_diverse_online_contract.py) performs no optimizer updates or simulated physics. Its [report](../../../artifacts/traverse/fdm_diverse_v1_20260909/geometry/online_contract_cpu_01.json) records tested source hashes.

- Poisoning future state/pose and the current/future action leaves history unchanged; changing the preceding applied action changes it. Building history from the online prefix exactly matches the training helper applied to a complete recording. Startup padding is the declared synthetic brake command.
- More goal progress reduces the tested cost. Positive work affects the energy-enabled arm and leaves the zero-energy arm unchanged. Rotating measured pose and goal preserves physical scoring.
- Contact, stall, rollover and hard attitude gates reject their synthetic hazardous forecasts. Required unsupported heads cause abstention. Unsupported rollover probability is reported unavailable while the attitude proxy remains active.
- Contact at first predicted arrival counts; events after arrival do not. Stall risk does not count before the required two-second future. All fifteen tested 240 m family references pass current kinematic checks.

Source inspection confirms frame-zero state/pose/history/goal are compared against the saved observation before any planning-dependent physical advance. Prediction sees neither authored hazard geometry nor future telemetry. New path geometry is checked separately from learned safety: valid curvature and arena footprint do not establish passability.

## Issues reported before the online snapshot

1. The first loop draft used asset contact alone in outcome and safe-goal fields, while the model's contact target is the union of asset and chassis resultants. Root was asked to make the measured outcome use the same union and retain explicit separate channels.
2. The first loop draft requests a new route/speed after `GetInputs()` and vehicle synchronization. The already-applied input therefore governs the current physical substep; the next native driver advance makes the new request available on the following substep. This is a 2 ms scheduling delay in the current build. Root was asked to record requested speed and effective timing separately from the actual applied input, preserving native controller dynamics.

These checks establish causal contracts and arithmetic, not learned decision accuracy, route-update success or long-route reliability. Replanning can change cruise speed and path under an already moving vehicle; fixed-reference training does not by itself establish coverage of those transitions. Online validation must measure them and report deliberate planner abstention separately from physical blockage.

## Findings addressed and physical adapter gate

The subsequent online draft adds `schema_contact`, `schema_rollover` and `schema_safe_goal_reached` from the actual rich solver-interval union and attitude maxima. Legacy asset-only fields are explicitly labeled; evaluation must select the schema fields. Decision records now distinguish `already_synchronized_action`, `new_desired_speed_mps` and `new_request_first_applied_time_s`, and the protocol declares the native substep delay.

[AMD job 412077](../../../artifacts/traverse/fdm_diverse_v1_20260909/driver_adapter_moving_v1/validation.json) validates a changed route while moving at 6.09 m/s: exact physical prefix before replacement, persistent speed PID, steering-only reset, and seven seconds of finite continued traversal on the changed reference. [Job 412076](../../../artifacts/traverse/fdm_diverse_v1_20260909/driver_adapter_v1/validation.json) separately established byte-identical initial/no-change behavior over a complete 41.35 s route. These gates are sufficient to proceed to the planned development-only online smoke; the smoke must still establish measured-history integration and planner behavior in the complete loop.

## Version 7 retained-reference policy

The subsequent [`fdm_online_candidates.py`](../../../src/nedm/traverse/fdm_online_candidates.py) makes the active reference available first, then original launch-time geometric families within 2 m and 15° of measured localization, followed by fresh current-pose families. Exact geometry/speed duplicates are removed. The active path is retained even when outside the nearby-family thresholds; availability itself is not a passability or tracking-support claim. Nearest-waypoint distance is conservatively quantized by the 0.5 m route sampling. All candidates still pass the common kinematic/model-risk checks.

[Independent CPU checks](../../../artifacts/traverse/fdm_diverse_v1_20260909/geometry/online_candidates_cpu_v7.json) confirm input references are not mutated, active geometry/speeds are exact, current station and FDM command features are correct, angle wrapping and join boundaries behave as declared, repeated reference identity survives, and unrelated future/truth-like metadata does not enter command tensors. The live loop initializes originals from the saved launch observation, computes eligibility from current measured pose, and updates the native driver's waypoint index to the current nearest point after reference replacement.

All version 7 comparisons use the same **0.025 m⁻¹** reference-curvature cap, derived from the geometric training-reference envelope rather than physical outcomes. The online `MPPIConfig` passes this limit to base, sampled and weighted-mean reference validation. Independent circular-arc checks reject a 20 m radius and accept a 60 m radius under this cap. This is a geometric support restriction; it does not establish safety on terrain.

The visualizer now uses the actual saved `candidate_references` for retention-policy runs and verifies their policy-source and geometry/speed fingerprints. It plots retained paths from their recorded current station forward, preserving full reference data in the artifact. Missing recorded candidates cause an error rather than substitution with new local curves. The old version 5 fresh-family reconstruction remains supported through its exact source hash. Version 6 failure evidence and version 7 candidate-policy results must be kept separate.
