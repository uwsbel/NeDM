# nav_v1 log — continuous sensor-driven waypoint navigation

2026-09-16
- Measured the blocker first: rebuilding `ChPathFollowerDriver` mid-drive resets its speed PID integral. On an
  UNCHANGED route, rebuilding every 2 s costs 0.09 m/s of mean speed and 1.1 m of trajectory divergence, with the
  throttle dropping to 0 at each rebuild. A Python replica of Chrono's controller matches it to 1e-16 over 4,150
  substeps, so `nav_online.SpeedPI` carries the integral across route changes and the Chrono follower is rebuilt
  for steering only (proportional, stateless).
- `sensor_map_v2.grid_from_arrays` factored out of `grid_from_capture` (verified bit-identical on a stored
  capture); `sensor_dataset_v2.set_grid` accepts an in-memory grid. Both are what makes per-decision sensing
  possible inside a running simulation.
- `nav_online.corridors12_batch`: all 256 candidates' corridors in one pass; output identical (max |dX| = 0) to
  the per-candidate masked extractor, 1.5x faster.
- `nav_runner.py`: one continuous rollout, `--mode waypoint|periodic`, per-decision render/back-project/propose/
  corridor/score timing, latency charging, sensed path heights, mask-leak check against the heightmap
  (evaluation only).
- Four new arenas generated for final confirmation (g213, g204, g234, g223 — the next four of the f104-similarity
  ranking, never driven or captured before). 30 missions, 10 arenas, 5-8 waypoints, 150-228 m.
- Latency on a 16-core mi2101x node: render 2.6 s (1024x1024 depth, software rasteriser), back-projection 0.05 s,
  candidate generation 0.4 s, corridor extraction 0.5 s, scoring 1.5 s (8 torch threads) -> 4.2-5.0 s per decision,
  of which 1.4-2.4 s is the planner itself.
- Trap found and removed: the reachable-speed clip pins the speed command to the measured speed (see PLAN
  amendment 1). Caught because the vehicle crawled at 0.3 m/s under 1 Hz replanning.
- First complete sensor-driven mission (g216_nav_001, one decision per waypoint, pilot code with the speed clip
  still on, so the numbers are superseded): 7/7 waypoints, 83.6 s, 285 m driven, with the last leg sliding
  backwards for 9.9 s before recovering. Proof the loop closes, not a result.
- Geometry gap found on the first campaign run and fixed before the campaign proper. A vehicle that arrives at a
  waypoint pointing ~110 deg away from the next one, near the arena edge, has NO valid route to it: the frozen
  `base_route` tries tight arcs of radius 12/10/9 the short way round and 12/10/9/8.5 the long way, and all of
  them leave the +-40 m arena. Radius 8.5 the short way is valid (curvature 0.1176 /m, inside the 0.125 /m limit).
  `nav_online.fallback_base` adds those tighter short-way radii, `surrogate_goal` aims at the closest reachable
  point if even that fails, and the runner replans when a route ends short of its waypoint. The frozen builder is
  unchanged. On such legs the proposal collapses to ~7 valid candidates (the tight arc uses most of the curvature
  budget), which is recorded per decision.
- A pilot pair on g216_nav_001 looked like a clean win for 1 Hz replanning (W slid for 9.9 s, R1 did not).
  It is NOT a valid pair and is not used: the W run predates disabling the reachable-speed clip, so the two arms
  had different speed commands. Re-run inside the campaign with both arms on the same code and node, W completes
  the same mission in 68.7 s with no slide at all. Only campaign pairs are used below.
- Control for the controller fix, same route throughout, rebuilding the follower every 1 s:
  | | mean speed | max trajectory divergence | frames to goal |
  |---|---|---|---|
  | Chrono follower rebuilt (no fix) | -0.09 m/s | 1.08 m | +9 |
  | with SpeedPI carrying the integral | +0.0011 m/s | 0.097 m | 0 |
  So the periodic-vs-waypoint comparison is not confounded by the act of rebuilding the follower.
- Second geometry gap, found on the first full campaign and fixed before the campaign that is reported. Leg-by-leg
  planning has no idea where the NEXT waypoint is, so a curved route can deliver the vehicle to a waypoint facing
  almost 180 deg away from the following one, sometimes 9 m from the arena wall where the 8 m minimum turning
  radius makes a U-turn impossible. Periodic replanning hit this more often than one-decision-per-waypoint (it
  takes different paths), so the first campaign made periodic replanning look worse for a reason that had nothing
  to do with planning quality. The fallback ladder is now: frozen `base_route` -> tight short-way arcs -> closest
  reachable surrogate point (scanned to +-180 deg at 10/12/15/20/25 m, ~0.3-0.6 s) -> drive straight ahead. On the
  two failing poses the surrogate cuts the distance to the waypoint from 28 m to 3 m and from 34 m to 17 m.
  Everything before this fix was discarded (job 421595, itself later discarded; the reported campaign is jobs 421779/421780).
- Planner-only latency, same stored frame and a real 256-candidate pool (`scripts/nav_latency_bench.py`,
  cluster job 421622):
  | | back-project | candidates | corridors | risk model | total | rate |
  |---|---|---|---|---|---|---|
  | MI350X GPU | 0.068 s | 0.130 s | 0.181 s | **0.042 s** | 0.42 s | 2.4 Hz |
  | same node, 16 CPU threads | 0.066 s | 0.128 s | 0.183 s | 0.335 s | 0.71 s | 1.4 Hz |
  | campaign node (mi2101x, 16 cores) | 0.05 s | 0.4 s | 0.5 s | ~1.5 s (8 threads) | 1.4-2.4 s | 0.4-0.7 Hz |
  Chrono's software depth rasteriser adds a further 2.6 s per decision on the campaign nodes. That is a property
  of the simulator, not of the planner: a real depth camera delivers a frame in milliseconds.
- The footprint exclusion holds while driving. Over the campaign's decisions the evaluation-only leak check (any
  cell outside the zone whose sensed height sits more than 0.6 m above the simulator's terrain) finds **zero**
  leaking cells; the largest height difference anywhere outside the zone is 0.069 m. Speeds at the moment of
  decision reach 9.8 m/s and 73% of decisions are taken above 2 m/s, so the 1.5 m margin measured on a parked
  vehicle also covers a moving one. The picked route's corridor is 6.2% invalid on average (p90 8.2%), rising to
  ~28% when the waypoint is 6-8 m away, which is why replanning stops inside 6 m of a waypoint.
  *(Re-checked 09-16: the figures in this paragraph came from campaign (4) below, not the reported one. On the
  reported campaign's 5,043 decisions (`main/runs/*/decisions.json`): zero leaking cells, largest difference
  outside the zone 0.078 m, speeds up to 6.9 m/s, 82% of decisions above 2 m/s, picked corridor 9.3% invalid on
  average (p90 17.9%) over the 5,016 decisions that picked a route. The conclusion is unchanged.)*
- Third defect, this one mine, found by reading the boundary exits of the second campaign. Carrying the speed
  integral across a whole mission (the fix above) introduces classic integral windup: with Ki = 0.05 the integral
  only has to reach 20 m/s.s to saturate the command by itself, which it does on any long stretch where the vehicle
  runs slower than commanded. Observed: **full throttle at 9-10.8 m/s against a 6 m/s command** on a downhill,
  steering saturated, 7 m off the route, ending outside the arena. Chrono's own controller has no anti-windup and
  does not need one — the collector rebuilds the driver every 10-20 s episode. `SpeedPI` now uses conditional
  integration plus a +-1/Ki clamp: identical to Chrono whenever the command is inside [-1, 1], at most 0.101 of
  throttle different during the saturated launch from rest (7.4% of substeps of a 7.5 s drive), and the
  rebuild-every-1 s artifact is still +0.0012 m/s and 0.065 m. The second campaign was discarded; its diagnosis is
  kept: 4/21 periodic missions left the arena, always at 8-10.8 m/s with the steering saturated, never because a
  planned route came near the wall (no route exceeded 38 m of the +-40 m arena).
- Fourth defect, in the rescue path, found in the anti-windup campaign (kept as `main_nocap`). A fallback arc at the
  8 m minimum radius validates as a route, but the path follower (5 m look-ahead, proportional gain 0.8, steering
  rate limited to 2 /s) cannot hold it at 4 m/s: the steering saturates at full lock, the vehicle runs 6 m wide of
  its own route and off the terrain. Measured asymmetry in that campaign — a fallback shape is used in 2/30 `W`
  runs but 6/11 `R2`, 2/3 `R1` and 5/7 `R1L` runs, because replanning from arbitrary poses meets hard geometry far
  more often — so a bad rescue path penalises exactly the arms under test. Fallback shapes are now commanded at
  **2 m/s** (`--fallback-speed-mps`, candidates resampled at that constant speed around the rescue shape); nothing
  on the normal path changes. Campaign `main_nocap` numbers for the record: W 27/30 complete (1 timeout,
  2 prolonged blockage, 0 arena exits), periodic arms 3 arena exits in 14 runs, all of them on a rescue route.
- Fifth defect of the same family: routes were allowed to run to the terrain edge. The frozen validator's
  arena bound IS the edge (+-40 m), which is fine for a single leg starting near the middle but not for a mission
  that works the whole arena — the follower's tracking error reaches ~3 m at 4-5 m/s on this terrain, so a route
  touching 37 m puts the vehicle over the edge and ends the run (observed: on-route at 35.5 m, 1 s later at 39.4 m
  with the steering saturated). Candidate generation now uses a 37 m planning bound for **every arm**; rescue
  shapes are still tried out to the edge if nothing else exists. Mission waypoints are all inside +-33 m, so this
  costs nothing, and a normal decision still yields the full 256-candidate pool.
- Order of campaigns: (1) discarded, no fallback ladder; (2) discarded, `main_nocap` kept for its diagnosis, no
  anti-windup and no fallback speed cap; (3) discarded, no planning margin; (4) jobs 421710/421711, superseded by the sixth fix below; (5) **reported**,
  jobs 421779/421780.
- Sixth infrastructure fix (the last before the reported campaign; a seventh was found afterwards, see the end of this log), and the reason the fifth was not enough. A fixed 37 m planning bound has a
  nasty failure mode: every candidate route starts at the vehicle, so once tracking error has carried the vehicle
  past the bound, EVERY candidate fails the arena test and the planner collapses onto its rescue ladder at exactly
  the moment it needs a normal route pointing back inside. (The validator also rejects a route whose swept corridor
  — half width 1.3 m — crosses the bound, so a vehicle at radius r needs a bound of at least r + 1.3 before any
  route from it validates at all.) `nav_online.plan_bound` now sets the bound per decision to
  `max(37 m, |pose|_inf + 2.5 m)`, and the rescue ladder uses the same. A vehicle 3 m from the corner pointing
  straight out of the arena still has no valid route at an 8 m minimum turning radius — that case is geometric, not
  fixable, and the margin is what keeps the vehicle from reaching it.
- Offline, well-powered answer to the sensing-range question (`nav_eval_range.py`, 1,200 labelled choices on the
  held-out arenas, frozen checkpoints, both masks applied):
  | usable sensing radius | valid corridor | height model: avoidable unsafe | depth model: avoidable unsafe |
  |---|---|---|---|
  | whole arena | 96% | 2.33% | 2.75% |
  | 30 m | 66% | 5.67% | 5.83% |
  | 25 m | 54% | 8.17% | 7.75% |
  | 20 m | 42% | 10.00% | 9.58% |
  | 15 m | 30% | 11.75% | 11.00% |
  (random pick 17.3% avoidable, i.e. 29.72% picked-unsafe against a 12.42% unavoidable floor.) A 20 m sensor
  recovers ~43% of the avoidable risk where the whole arena recovers ~85%. Both models degrade identically, so
  this is about missing terrain, not about depth vs height. Retraining ON range-limited corridors is therefore the
  one place where retraining is clearly indicated; datasets masked at 20 m built and training launched.
- Determinism check: the two video runs (cluster `video/runs/`; the f104_nav_001/002 video runs are copied locally to `video/`, g216_nav_001__W is cluster-only) reproduce their campaign counterparts exactly —
  `f104_nav_001__W` 5/7 waypoints, 102.2 s, 163.0 m, 10.70 s backward in both; `g216_nav_001__W` 7/7, 69.4 s.
  Same mission, same arm, different job, same partition.

## Reported campaign (jobs 421779 / 421780, all 120 runs)
| arm | complete | waypoints | unsafe | legs with a slide | median time | decisions | stalls | arena exits |
|---|---|---|---|---|---|---|---|---|
| W (once per waypoint) | 27/30 | 96.5% | 6 | 7/195 | 76.6 s | 7 | 3 | 0 |
| R2 (every 2 s) | 23/30 | 88.4% | 13 | 8/183 | 74.1 s | 38 | 2 | 5 |
| R1 (every 1 s) | 24/30 | 87.9% | 15 | 13/181 | 75.6 s | 73 | 2 | 4 |
| R1L (1 s, latency charged) | 23/30 | 87.4% | 12 | 10/181 | 76.4 s | 51 | 2 | 5 |
Travel time vs W on paired missions: R2 -5.7 s [-15.6, +3.2], R1 -3.3 s [-11.9, +4.9], R1L **+3.1 s**
[-11.5, +19.1] — charging the algorithmic latency costs about 6 s a mission and removes the (already
non-significant) speed advantage of replanning.
Unseen arenas: W 16/16 complete, R2 9/16, R1 12/16, R1L 12/16. Development arenas: W 11/14, R2 14/14, R1 12/14,
R1L 11/14.
Random-pick control (`R1rand`, identical loop): 2 of the first 6 missions complete, 156-207 s where the model's
arms take 60-80 s, up to 54 s of backward sliding in one mission.
Per-decision median wall clock: render 2.6 s, back-project 0.04 s, candidates 0.85 s, corridors 0.29 s, risk model
0.60 s, total 4.2-4.4 s.
- Random-pick control (`R1rand`, identical loop, all 30 missions): **6/30 complete against 24/30, 45.2% of
  waypoints against 87.9%, median 166 s against 75.6 s**, 28 of 30 missions with a backward slide and 580 s of
  backward sliding in total. Statuses: 6 complete, 14 prolonged blockage, 8 timeouts, 2 arena exits.
- **The latency-charged arm is the one arm that is not reproducible**, and it matters. `R1L` charges each decision's
  own *measured* wall-clock planning time, which depends on machine load, so the plans activate at different
  simulated frames from run to run. Re-running `f104_nav_001__R1L` for the videos produced charged delays of
  1.35/1.45/1.80/1.90/1.65/1.50 s where the campaign run charged 1.45/1.55/1.70/1.80/1.50/1.45 s — and the two rolls
  end differently: **the campaign roll left the arena at 2/7, the video roll completed all 7 waypoints in 74.0 s**.
  Every other arm reproduced bit-for-bit on the same partition (`W` on `f104_nav_001`: 5/7, 102.2 s, 163.0 m,
  10.70 s backward in both; `f104_nav_002__W` 76.2 s in both). So `R1L`'s campaign row carries run-to-run variance
  that the other rows do not, and a fixed latency budget (`--latency-s 1.5`) should be used if that arm is ever to
  be compared like for like.
- Video deliverable (`videos/amd_heightmap_render/` and its `README.md`, `scripts/nav_video_compare.py`): two missions x four planners
  plus a four-way synchronised comparison each, all from fresh rollouts with `--save-frames --rgb on`.
  * `f104_nav_002` (7 waypoints, 195 m) — all four finish: 76.2 s / 7 decisions, 73.8 s / 35, 74.8 s / 69,
    71.2 s / 50. No sliding anywhere.
  * `f104_nav_001` (7 waypoints, 212 m) — one decision per waypoint reaches 5/7 and is ended by the stall rule
    after 10.7 s of sliding backwards; 2 s finishes in 63.0 s with no sliding; 1 s finishes in 81.5 s after sliding
    for 7.5 s and driving out of it; the latency-charged arm finished this roll in 74.0 s (its campaign roll left
    the arena at 2/7 — see the non-reproducibility note above).
  All eight rollouts reproduce their campaign counterparts exactly except the latency-charged pair.

## Local re-run on luffy (2026-09-16, OptiX on the RTX 5090)
- Same 30 missions x 4 planners, same frozen checkpoints, run locally with the user's source-built Chrono (OptiX,
  depth-FOV fix #819) via `scripts/nav_local.py` (system Python 3.12 + numpy 1.26 for the Chrono build, conda torch
  appended for CUDA scoring) and `scripts/nav_local_batch.py` (6 runs at a time). 120/120 runs, 70 min wall. Results
  in `local_luffy_v0_foldback/` (`summary.json`, `tables.md`; renamed after the rescue-route bug below, since
  `local_luffy/` now holds the fixed re-run).
- Per decision: render 0.01 s, back-projection 0.05 s, candidate generation 0.36-0.66 s, corridors 0.29 s, risk model
  0.05 s (GPU); 0.86-1.05 s total with six runs sharing the 8-core CPU. The render is no longer the cost.
- | planner | completed | waypoints | unsafe missions | legs with a slide | median time | stalls | arena exits | rollovers |
  |---|---|---|---|---|---|---|---|---|
  | once per waypoint | 27/30 | 96.0% | 5 | 4/194 | 76.6 s | 2 | 1 | 0 |
  | every 2 s | 25/30 | 93.5% | 12 | 12/191 | 78.4 s | 2 | 2 | 1 |
  | every 1 s | 24/30 | 89.4% | 11 | 10/184 | 75.0 s | 4 | 2 | 0 |
  | every 1 s, delay charged | 23/30 | 86.4% | 8 | 7/179 | 82.3 s | 4 | 3 | 0 |
  Unseen arenas: once per waypoint 16/16 again. Same picture as the AMD campaign: replanning more often does not improve completion (but this run had the same rescue-route bug, 6 of its 9 exits/rollovers followed a doubled-back route, so it is not independent confirmation; see the end of this log). Per-run completion agrees with the AMD campaign in 92 of 120 runs; the rest differ because
  Chrono is not bit-reproducible across machines and the risk model now scores on the GPU. The failure mix is less
  lopsided locally (replanning arms: 2-4 stalls and 2-3 arena exits each).
- Recording both RGB cameras (overhead + a new chase camera) leaves a rollout bit-identical (1,581 frames, zero pose,
  state and action difference, same picks).

## Rescue routes that double back on themselves (bug, found 2026-09-16 by the video audit)
- Symptom: in the first RGB videos a replanning run on `g204_nav_003` turned around and drove off the arena on a
  route that went forward a few metres and then straight back past the vehicle.
- Cause: when no normal candidate is valid the runner falls back to a stand-in goal scanned to +-180 deg. With the
  stand-in directly behind the vehicle, the frozen route builder returns a route with a sharp reversal (a cusp at
  point 4 of 33, or 7 of 51). The frozen route checker estimates curvature in a way that reads the cusp as 0.000
  against the 0.125 /m limit, so the route passed (`videos/_superseded_v0_foldback/_audit/completeness_critic/c5_*`). In the c5/c6 scripts, `local_luffy/` means the folder now named `local_luffy_v0_foldback/`; their saved stdout is the record, re-running them needs the local per-run files.
- Extent (any route whose heading turns > 150 deg between consecutive points, `c6_cusp_eval_stdout.txt`):
  **all 14 arena exits of the AMD campaign** followed one or more such routes (R2 5, R1 4, R1L 5), and 3 completed
  AMD runs used one without harm; in the first local re-run 6 of the 8 arena exits did (the one rollover did not), and so did 1 of the 12 local stalls (`g203_nav_001__R1`). No AMD stall involved one.
- Consequence: the reported conclusion "replanning trades stalls for driving off the arena" is **confounded by this
  bug** and is withdrawn (REPORT.md now carries a correction at the top). Latency, rest-vs-moving and sensing-range
  results do not involve rescue routes.
- Fix (`scripts/nav_online.py`): `safe_validate_no_reversal` wraps the frozen checker and rejects any route whose
  heading turns more than 45 deg between consecutive points. Normal routes turn at most 6.6 deg per step across both campaigns (the eight AMD video rollouts: 3.9-5.0 deg max), so only degenerate rescue shapes are affected. Smoke test:
  `g204_nav_003__R2` now ends with no valid route at 7/8 (boxed in facing the north edge) instead of driving off.
- The AMD height-map videos (`videos/amd_heightmap_render/`) contain no such route and stand. The local RGB videos
  and the first local re-run were moved to `videos/_superseded_v0_foldback/` and `local_luffy_v0_foldback/`.
- Re-run with the fix: `local_luffy/` (same 120 tasks, luffy, OptiX). The session restart on 09-16 killed it at
  37/120; the six incomplete run folders were moved to `local_luffy/_killed_partial_0916/` and the batch resumed
  (finished runs are skipped).
- Per-arm evidence that replanning asks for rescue routes more often (AMD campaign, stand-in goals committed):
  once per waypoint 1 (1 run), every 2 s 10 (6 runs), every 1 s 15 (5 runs), delay charged 60 (12 runs). Each exit
  followed the last committed route, a doubled-back one, by 2.1-3.9 s. The fixed re-run's first 90 finished runs have
  no route turning more than 6.1 deg. (Independent verification, 09-16, from `main/runs/*/routes.json`; local only.)
