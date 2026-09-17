# nav_v1 — continuous sensor-driven waypoint navigation (pre-registration, written before any nav_v1 Chrono result)

The milestone: drive a sequence of waypoints in ONE continuous Chrono rollout, replanning toward the active
waypoint from a fresh overhead depth frame with the vehicle in it. Everything before this ran the planner once per
start/goal (or once per waypoint) from a vehicle-free arena image.

## Frozen (not changed by this milestone)
* Risk model architecture and checkpoints: `sensor_v2/matched/matched_Dabs_s{0,1,2}.pt` (the matched direct-depth
  ensemble; channels `range_abs, sec1, speed, valid`). No retraining unless a measured failure justifies it, and
  then only as a separate, labelled arm.
* Candidate generation: `f104_n2_sampler.propose`, 256 candidates, curvature rejection at 0.125 /m.
* Corridor geometry: 96 stations x 32 lateral samples over +-6 m; the vehicle-exclusion rules of
  `vehicle_corridor.tensor12_excluded` with the pilot's 1.5 m margin.
* Physics, vehicle, terrain, settle (0.8 s), per-leg stop policy (`gen_mission_runner.LegStop`), rollover/bounds
  limits, and the unsafe/failure label definitions.

## Supporting fixes (measured, reported, no model change)
1. **Speed-integrator carry-over.** Rebuilding `ChPathFollowerDriver` resets its speed PID integral. Measured on
   an UNCHANGED route, rebuilding every 2 s costs 0.09 m/s of mean speed and 1.1 m of trajectory divergence, with
   the throttle dropping to zero at every rebuild. Without this fix a periodic-replanning arm would lose to a
   plan-once arm for reasons that have nothing to do with planning. `nav_online.SpeedPI` reimplements Chrono's
   controller (verified equal to 1e-16 over 4,150 substeps) so the integral survives a route change; the Chrono
   follower is rebuilt for steering only, which is proportional and stateless.
2. **Reachable speed command.** At a replan the candidate's commanded speed is clipped to what the vehicle can
   reach from its current speed using the sampler's own acceleration cone. Without it the command can jump by
   several m/s at a replan — a transient absent from training.
3. **Sensed path heights.** The reference path handed to the follower is lifted with heights read from the same
   depth frame, not from the arena heightmap (`--path-heights map` keeps the old behaviour as a control).

## Missions
30 missions, 10 arenas, 5-8 waypoints each (mean 6.6), 150-228 m of chained legs, legs 22-35 m, turn <= 120 deg at
each waypoint, at least half the legs crossing a hill or crater, all points on slope < 7 deg and inside +-33 m.
* Development arenas (already used in this project): f104 (4 missions), g228, g203, g217, g216, g231 (2 each).
* **Previously unseen arenas** generated for this milestone and never driven, captured or inspected before:
  g213, g204, g234, g223 (4 each). They are the next four entries of the existing f104-similarity ranking.

## Arms (matched missions, same mission file, same frozen checkpoints)
| arm | decisions |
|---|---|
| `W`  | one decision per waypoint (sensor-driven, the existing behaviour) |
| `R2` | a decision every 2.0 s of simulated time |
| `R1` | a decision every 1.0 s of simulated time |
| `R1L`| `R1` with each decision's own measured wall time charged to the simulation |

Replanning is skipped inside 6 m of the active waypoint (a route that short has no corridor); the skip is counted.

## Primary comparison
Per mission, `R2` and `R1` against `W`: missions completed (all waypoints reached), unsafe missions (any backward
slide by the frozen definition, or any failed leg), total travel time, and waypoints reached. Paired over the 30
missions; reported separately for development and unseen arenas. Nothing here is a hypothesis test with a
pre-set alpha — 30 missions cannot resolve small differences, and the milestone's question is whether the
continuous pipeline works at all and at what cost.

## Latency
Every decision records render, back-projection, candidate generation, corridor extraction and scoring separately,
plus their sum (`sense_to_plan_s`). Reported as a distribution, converted to the real-time replanning rate it
implies, and — in `R1L` — charged to the simulation so the vehicle keeps following the previous route until the
new one activates. Simulated planning frequency and wall-clock feasibility are reported as two different numbers.

## Deliverables
A reproducible runner (`scripts/nav_runner.py`, `scripts/nav_online.py`), the mission set, per-decision records,
videos showing the route updating while driving, and a report of what is demonstrated and what is not.

## Amendment 1 (2026-09-16, during implementation, before any nav_v1 arm was run to completion)
* **The reachable-speed clip is off.** Clipping a candidate's commanded speed to the acceleration cone from the
  current speed looked obviously right and is a trap: every route starts at the vehicle and the path follower
  reads its speed command at the vehicle's own station, so the cone pins the command to the measured speed and the
  vehicle never accelerates (measured: 0.3 m/s after 5 s of 1 Hz replanning). The training episodes all commanded
  2/4/6 m/s from rest, so the unclipped profile is the in-distribution one. Supporting fix 2 in the plan above is
  therefore withdrawn; the flag remains for the diagnostic.
* **`R1L` charges the algorithmic latency, not the render.** The overhead frame is produced by Chrono's software
  rasteriser (lavapipe, no GPU): 2.6 s for 1024x1024 depth on a 16-core node. That is a property of this
  simulator, not of the planner, and a real depth camera delivers a frame in milliseconds. `R1L` therefore charges
  back-projection + planning (measured 1.4-2.4 s); the render cost is reported separately and never hidden.
* **The overhead RGB camera is switched off** for runs whose checkpoints do not read colour (the deployed depth
  ensemble reads `range_abs, sec1, speed, valid`). This cuts the render from 13 s to 2.6 s and provably cannot
  change the model input; the runner refuses to do it if any checkpoint reads a colour channel. RGB is rendered
  for the video runs.
* `RenderSpec` gained an additive `with_rgb=True` field. The frozen `gen_v1/source` tree is untouched; nav_v1 runs
  from its own copy (`nav_v1/source`).
* Exploratory arms added after the pre-registered four, if the results call for them: `R2L` (2 s with latency
  charged) and `R1S` (1 s with the route the vehicle is already on scored alongside the fresh candidates).

## Amendment 2 (2026-09-16, written while the campaigns that were discarded were being diagnosed)
Four infrastructure defects were found, each by a failure that hit the replanning arms harder than the plan-once
arm for reasons unrelated to planning quality. Every one is recorded with its measurement in LOG.md, the campaigns
that contained them were discarded, and the reported campaign runs a single code version throughout:
anti-windup on the carried speed integral; a rescue ladder for waypoints the frozen route builder cannot reach;
rescue shapes commanded at 2 m/s because the follower cannot hold an 8 m radius at 4 m/s; and a candidate
validation bound 3 m inside the terrain, widened to the vehicle's own radius when tracking error has already
carried it past that. `R1S` and `R2L` were dropped for time; `R1rand` (the same loop picking at random) was added
as the control for whether the risk model is doing anything inside the loop, and a range-limited retraining was
added because the offline range sweep called for it.
