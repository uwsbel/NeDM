# Native Chrono PID with online reference updates

The new [`OnlinePathFollower`](../../../src/nedm/traverse/fdm_online_driver.py) retains one native `ChPathFollowerDriver`, its speed PID, steering controller and shared `ChBezierCurve`. Its initial construction matches the collector: select original waypoints at least 2 m apart, always retain the last point, use terrain altitude + 0.5 m, set 5 m lookahead, steering gains `(0.8, 0, 0)`, speed gains `(0.6, 0.05, 0)`, and initialize before settlement. It performs no route selection or model inference.

```python
driver = OnlinePathFollower(chrono, veh, vehicle, route, height_source)
provenance = driver.update_route(new_route, measured_xy_yaw)
driver.SetDesiredSpeed(desired_speed)
driver.Synchronize(time)
inputs = driver.GetInputs()
driver.Advance(physics_dt)
```

Construction already initializes the native driver. The caller retains the existing control interval, 0.8 s settlement, steering slew limit and speed-profile scheduling. The update method does not advance physics or change the desired speed. It validates finite XY points, stations and forward speeds; `current_pose` is measured `[world_x, world_y, yaw]` for provenance, and steering resets against the actual current vehicle reference frame.

## Actual AMD bindings

Inspection of `/home1/harry/chrono` and the installed Python bindings confirmed lowercase `ChBezierCurve.setPoints(points, inCV, outCV)`, plus `driver.GetSteeringController().Reset(vehicle.GetRefFrame())`. The whole driver's `Reset()` resets both controllers and is never called by the adapter.

The build exposes native spline derivatives but not its control-vertex getters. For a changed route, a temporary native interpolating spline supplies endpoint derivatives. The cubic identities `outCV[i] = point[i] + derivative(i, 0)/3` and `inCV[i+1] = point[i+1] - derivative(i, 1)/3` recover its control polygon. The adapter mutates the existing shared curve and resets only the steering tracker, then verifies interpolated positions against the temporary native spline. Identical or speed-only references do not query terrain, mutate a curve or reset either PID.

Relevant source locations are `src/chrono/core/ChBezierCurve.h:85`, `src/chrono/core/ChBezierCurve.cpp:233`, `src/chrono_vehicle/utils/ChSteeringController.cpp:57`, and `src/chrono_vehicle/driver/ChPathFollowerDriver.cpp:79` in the AMD Chrono checkout.

## Matched physical parity

AMD job **412076** completed successfully in **1 min 01 s**, starting `2026-09-09T16:36:56` and ending `16:37:57` in Slurm's reported clock. The train rolling-hills scene, reference `family_14`, used the same immutable `campaign_v2` runner/runtime as the earlier headless control. The adapter was injected only into that audit process; frozen source was unchanged.

Forty-two identical-reference updates during the **41.35 s** successful traversal left all eleven physical arrays exactly equal: state, pose, controls, power, contact, interval work and endpoint/parking fields. The entire `trajectory.npz` SHA also matches. The rich telemetry validator passes. After this rollout, a speed-only update reset neither PID; a changed-geometry update successfully changed the controller's shared curve, preserved the speed-controller pointer and current-speed value, and matched a fresh native interpolant within **1.59 × 10⁻¹⁴ m**.

[Exact report and hashes](../../../artifacts/traverse/fdm_diverse_v1_20260909/driver_adapter_v1/validation.json), [audit declaration](../../../artifacts/traverse/fdm_diverse_v1_20260909/driver_adapter_v1/declaration.json), and [recorded updates](../../../artifacts/traverse/fdm_diverse_v1_20260909/driver_adapter_v1/no_change_update_records.json) preserve this evidence. This first check establishes initial/no-change parity and binding behavior; changed-route driving is checked separately.

## Reference replacement while moving

AMD job **412077** completed successfully in **33 s** (`2026-09-09T16:43:47` to `16:44:20`, Slurm clock). At 8 s after settlement the vehicle was traveling at **6.09136 m/s**. A new measured-pose Hermite route changed the native curve from 115 to 89 knots. Steering reset against the live vehicle frame; native speed-controller pointer and current speed were unchanged. Spline reconstruction error was **1.42 × 10⁻¹⁴ m**.

The first 8 s of state, action, pose, power, contact and work exactly match the baseline. The vehicle then completed another 7 s with finite telemetry and a changed physical path; rich telemetry validation passed. The cruise request stayed 6 m/s. [Moving-update report](../../../artifacts/traverse/fdm_diverse_v1_20260909/driver_adapter_moving_v1/validation.json) preserves the complete path and measurements. Its diagnostic `previous_knots` field is the original input waypoint count, 458; actual native knot counts are 115 before and 89 after. This confirms mutable native PID operation during motion, without claiming the replacement route is optimal or safe over its entire remaining length.
