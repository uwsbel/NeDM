# Vehicle-included depth-input pilot (2026-09-15)

Model architecture, candidate generation, controller and checkpoints unchanged; this is preprocessing only.
Artifacts: `vehicle_frames/<case>/` (raw sensor frame + measured pose), `vehicle_pilot.png`,
`vehicle_pilot_results.json`, `vehicle_pilot_routes/`, `vehicle_pilot_runs/`, `vehicle_pilot_chrono.json`,
`vehicle_pilot_time.json`, `mask_coverage.json`, `mask_choice_effect.json`.
Code: `scripts/vehicle_capture.py`, `scripts/vehicle_corridor.py`, `scripts/vehicle_pilot.py`,
`scripts/vehicle_pilot_figure.py`, `scripts/sensor_eval_mask.py`.

## 1. Capture
One overhead RGB-D frame per planning decision with the HMMWV present, after the frozen 0.8 s settle, using the same
scene builder, physics, driver and camera as the arena capture (1024x1024, 110 m, 47 deg, 180 m max range). The
measured chassis pose at capture time is saved with the frame and is the only pose the mask uses. 20 cases on the two
held-out arenas: route grades 0.26-0.59, start tilts up to 0.166, relief 1.5-5.1 m, including curved routes.

## 2. Exclusion zone: measured, not assumed
Vehicle-affected cells (height changed by > 0.15 m, or coverage lost to the vehicle's shadow) number 440-548 per case,
of which 0-97 are shadow. Their greatest distance outside the bare footprint (2.6 x 1.3 m half-extents) is
**0.50 m** (p99 0.29 m) across all 20 cases. A **1.5 m** margin was used - 3x the worst observed case - giving an
exclusion zone of ~1,880 cells (46 m^2). `vehicle_pilot.png` shows, per case, the rendered vehicle, the affected cells
and the zone boundary: every affected cell lies inside it.

## 3. Corridor handling
96 x 32 coordinates and the full route are preserved. A sample is invalid unless all four grid cells it interpolates
from are covered AND outside the zone. Height/range references come from the first station with >= 4 valid samples
outside the zone (station 0 in every case here: the corridor is +-6 m wide, the zone +-2.8 m across). Hidden ground is
never filled - not from simulator heights, not from the vehicle-free image.

## 4. Three-way comparison (identical pools, frozen checkpoints)
| | invalid corridor samples | pick == A | rank corr. vs A |
|---|---|---|---|
| A vehicle-free, no mask | 0.2% | - | - |
| B vehicle-free + mask | 5.2% | H 6/20, Dabs 11/20 | 0.968 / 0.971 |
| C vehicle present + mask | 5.2% | H 6/20, Dabs 11/20 | 0.968 / 0.971 |

**B and C are identical**: 0.0000 m height difference on every valid sample, rank correlation 1.000, same chosen route
in 20/20 cases for both models. With the mask in place, a vehicle-included frame is indistinguishable from a
vehicle-free one. All differences come from the mask itself, not from the vehicle.

## 5. Driving check (63 drives, 0 collection failures)
Both models, A vs C: **0 unsafe runs and 0 failures in both conditions**. Median time 13.9 -> 14.1 s (height) and
10.9 -> 11.5 s (depth); paired mean +2.65 s [+0.16, +5.29] (height) and -0.02 s [-1.30, +1.25] (depth), with single
cases moving up to 16 s either way. 20 cases with zero unsafe events cannot resolve a safety difference.

## 6. Does the mask cost planning quality? (1,200 labelled choices, frozen checkpoints)
Applying the same exclusion to the held-out arenas' already-driven designed routes and re-running the route-choice
metric: height 14.50% -> 14.83% unsafe picks (+0.33), depth 14.92% -> 14.83% (-0.09). The two models move in opposite
directions by less than a third of a point, i.e. no measurable cost.

## 7. Verdict
The preprocessing works. A vehicle-included frame plus a footprint-based exclusion reproduces the vehicle-free input
exactly, keeps the corridor geometry intact, and neither the 1,200-choice offline metric nor 63 drives show a
performance cost. **Retraining is not justified by this evidence.** The frozen checkpoints tolerate ~5% invalid
samples at the route start even though they never saw an invalid sample in training.

## 8. Limits and the next step
- 20 cases, all on two arenas, all from a still vehicle at the start pose; no moving-vehicle or mid-route decision.
- Zero unsafe events in the drive check: it bounds gross breakage, not small differences.
- The margin is validated for this vehicle, this camera height and radii up to ~35 m; a lower camera or a larger
  vehicle needs the measurement repeated (the shadow scales with height x radius / camera height).
- Smallest next step, only if the vehicle-included path is to be adopted: fold the mask into the existing fixed-speed
  Chrono comparison (~1,200 start/goals) as one extra arm, rather than a separate campaign; retraining with masked
  corridors should wait until that arm shows a cost.
