# nav_v1 — continuous, sensor-driven waypoint navigation in Chrono

*(numbers filled in from `summary.json`, `diag_*.json` and `bench_*.json`; see PLAN.md for what was fixed in
advance and LOG.md for what was found along the way)*

> **Correction, 2026-09-16 (later the same day).** The explanation below that replanning "drives off the arena"
> is withdrawn. An audit of the videos found a bug in the runner's rescue routes: when no normal candidate was
> valid, the stand-in goal could sit directly behind the vehicle, the route builder then produced a route that doubles back on itself at a sharp point, and the route checker scored that point as curvature 0.000 and accepted it. Each exit came 2-4 s after the vehicle started following such a route. **All 14 arena exits in the AMD campaign below followed one of these routes** (and 6 of the 9
> exits/rollovers in the first local re-run). The replanning arms ask for rescue routes far more often, which is
> why the exits landed on them. The fix rejects any route that turns more than 45 deg between consecutive points
> (`nav_online.safe_validate_no_reversal`, which the frozen route builder's shapes now also pass through); the builder's code and the model are unchanged. The tables in section 5 are the measured results of the buggy runner and are kept as a record. **The re-run with
> the fix is done** (120 rollouts on the workstation, `local_luffy/summary.json`, section 5a): arena exits fall from
> 14 to 3 across 120 runs, and the comparison itself is unchanged — planning once per waypoint still completes more
> missions (27/30) than replanning every 2 s (25/30), every 1 s (25/30) or every 1 s with the delay charged (22/30). The latency measurements (section 4), rest vs moving (section 6) and sensing range (section 8) do not depend on the rescue routes, but the example of the delay-charged arm's run-to-run variance in sections 4 and 9 (a campaign roll that left the arena) went through the bug.

## 1. What runs now

One Chrono rollout per mission. The vehicle is never reset and never repositioned. At every planning decision:

1. the simulator renders **one overhead RGB-D frame with the HMMWV in it**, at the pose the vehicle is actually in;
2. the depth image is back-projected through the camera intrinsics into a 0.156 m metric grid (`sensor_map_v2`);
3. the vehicle's own footprint plus a 1.5 m margin is marked **unobserved** at the measured pose — not filled from
   the heightmap, not filled from a vehicle-free image;
4. 256 candidate routes are generated from the measured pose to the active waypoint and their 96x32 corridors are
   read out of **that one frame**, with samples inside the exclusion zone marked invalid and the coordinates and
   route length left alone;
5. the frozen 3-seed direct-depth ensemble (`matched_Dabs`, channels `range_abs, sec1, speed, valid`) scores all
   256 and the lowest-risk route is handed to the path follower;
6. the vehicle keeps driving; when it comes within 2.5 m of the active waypoint the next one becomes active.

Nothing is remembered between decisions: each decision sees one frame and nothing else. The model, the candidate
generator, the risk definition and the checkpoints are unchanged from `sensor_v2`.

## 2. What had to be fixed to make the comparison honest

**The speed integrator.** Chrono's `ChPathFollowerDriver` has to be rebuilt to follow a new path, and rebuilding
resets its speed PID integral. On an *unchanged* route, rebuilding every 2 s costs 0.09 m/s of mean speed and 1.1 m
of trajectory divergence, with the throttle dropping to zero at each rebuild — so a periodic-replanning arm would
have lost to a plan-once arm for a reason that has nothing to do with planning. `nav_online.SpeedPI` reimplements
that controller (verified equal to Chrono's own to 1e-16 over 4,150 substeps) and carries the integral across route
changes; the Chrono follower is rebuilt for steering only, which is proportional and stateless. Residual artifact
of rebuilding every 1 s: +0.001 m/s and 0.065 m.

Carrying the integral then needs **anti-windup**, which Chrono's controller does not have and does not need,
because the collector rebuilds the driver for every 10-20 s episode. Over a 90 s mission the integral only has to
reach 20 m/s.s to saturate the command by itself, and it did: full throttle at 9-10.8 m/s against a 6 m/s command
on a downhill, 7 m off the route, off the terrain. Conditional integration plus a clamp fixes it and leaves the
controller identical to Chrono's whenever the command is inside [-1, 1] (at most 0.101 of throttle different during
the saturated launch from rest).

**A trap that was not a fix.** Clipping each candidate's commanded speed to the acceleration cone from the current
speed looks obviously right. It is a trap: every route starts at the vehicle, the follower reads its speed command
at the vehicle's own station, and the cone pins that command to the measured speed. The vehicle then never
accelerates — 0.3 m/s after 5 s of 1 Hz replanning. It is also unnecessary: every training episode commanded
2/4/6 m/s from rest, so the unclipped profile is the in-distribution one.

**Route geometry at waypoints.** Leg-by-leg planning does not know where the *next* waypoint is, so a curved route
can deliver the vehicle to a waypoint facing up to 180 deg away from the following one, sometimes 9 m from the
arena wall where the 8 m minimum turning radius makes a U-turn impossible. The frozen route builder then has no
valid candidate at all and the mission ends. The runner now falls back, in order, to tight short-way arcs, to the
closest reachable stand-in point (scanned to +-180 deg), and finally to driving straight ahead, and replans when a
route ends short of its waypoint. The frozen builder itself is untouched. (After the campaign, rescue routes that
double back on themselves were found to pass the route checker; they are now rejected — see the correction at the top.)

## 3. Arms

| arm | decisions | notes |
|---|---|---|
| `W`    | one per waypoint | the existing multi-goal behaviour, but sensor-driven |
| `R2`   | every 2.0 s of simulated time | |
| `R1`   | every 1.0 s of simulated time | |
| `R1L`  | every 1.0 s, **planning delay charged to the simulation** | the vehicle keeps following the previous route until the new one activates; the delay charged is the measured back-projection + planning time, not Chrono's software depth render |

Every arm shares the mission file, the frozen checkpoints, the corridor rules and the controller, and all four arms
of a mission run on the same node (Chrono on this cluster is deterministic per node, not across nodes). Replanning
is skipped inside 6 m of the active waypoint: that close, the footprint mask invalidates a quarter of a corridor
that short.

Exploratory arms, built after the four above; only `R1rand` has been run (reported as exploratory), the others are implemented but unrun: `W20`/`R1_20` (the overhead frame cropped
to 20 m around the vehicle, so a 25-35 m route reaches past what has been seen), `R1rand` (the same loop picking a
candidate at random - the control for whether the risk model is doing anything), `R1S` (the route the vehicle is
already on is scored alongside the fresh candidates).

### Missions and arenas

30 missions, 5-8 waypoints each (mean 6.6), 150-228 m of chained legs, legs 22-35 m, turn <= 120 deg at each
waypoint, at least half the legs crossing a hill or a crater, every point on slope < 7 deg and inside +-33 m
(`missions.png`). Six development arenas contribute 14 missions; four arenas **generated for this milestone and
never driven, captured or inspected before** contribute 16. The new four are the next entries of the existing
f104-similarity ranking and match it closely:

| arena | slope cap | roughness | hills / craters | slope p99 | height range |
|---|---|---|---|---|---|
| f104 (reference) | 30.9 deg | 0.240 m | 5 / 5 | 28.6 deg | 5.70 m |
| g213 (unseen) | 29.6 deg | 0.24 m | 5 / 6 | 28.4 deg | 6.80 m |
| g204 (unseen) | 28.8 deg | 0.26 m | 6 / 5 | 27.1 deg | 6.45 m |
| g234 (unseen) | 30.6 deg | 0.21 m | 7 / 5 | 29.2 deg | 6.35 m |
| g223 (unseen) | 31.2 deg | 0.20 m | 6 / 5 | 29.8 deg | 6.45 m |

Missions 0-12 ran on mi2104x nodes and 13-29 on mi2101x; all arms of a mission always share one node, because
Chrono on this cluster is deterministic per node but not across nodes.

## 4. Sensing-to-planning latency: simulated rate vs wall clock

Two different numbers, and they should not be confused.

**Simulated planning frequency** is what the arms are named for: `R1` takes a decision every 1.0 s of simulated
time, `R2` every 2.0 s. The simulation pauses while the decision is computed, so in `W`, `R2` and `R1` planning is
free — that is the pre-registered condition.

**Wall clock.** Every decision records its own stages. On the campaign nodes (16 or 32 cores of a shared
mi2101x/mi2104x node), the median decision costs ~4.2-5.0 s, of which **~2.6 s is Chrono's software depth render** (Vulkan ray tracing on lavapipe, a CPU software driver; there is no GPU rendering path here). The render is a property of the simulator, not of the planner: a real depth camera delivers a frame in milliseconds. (Measured later in `../render_latency_v1/`:
about 1.9 s of that is CPU software ray tracing and 0.8 s is Chrono rebuilding the whole ~800k-triangle scene every
frame; the same 1024x1024 depth frame takes 7 ms with OptiX on an RTX 5090.)

The planner alone — back-projection, candidate generation, corridor extraction, ensemble scoring — was measured on
the same stored frame and a real 256-candidate pool (`nav_latency_bench.py`):

| | back-project | candidates | corridors | risk model | total | implied rate |
|---|---|---|---|---|---|---|
| MI350X GPU | 0.068 s | 0.130 s | 0.181 s | **0.042 s** | 0.42 s | **2.4 Hz** |
| same node, 16 CPU threads | 0.066 s | 0.128 s | 0.183 s | 0.335 s | 0.71 s | 1.4 Hz |
| campaign node (8 torch threads, shared) | ~0.05 s | ~0.3 s | ~0.4 s | ~1.0 s | 1.4-2.4 s | 0.4-0.7 Hz |

So 1 Hz replanning is already achievable with a GPU for the network and the present CPU code for everything else,
and the remaining cost is not the neural network — it is candidate generation and corridor extraction, both plain
numpy.

**`R1L` charges the delay.** In that arm the plan computed at simulated time *t* only takes effect at
*t + (back-projection + planning)*, measured per decision; until then the vehicle keeps following the previous
route, and no new decision starts while one is pending. **That makes it the one arm that is not reproducible**:
the charged delay is measured wall-clock time and depends on machine load, so plans activate at different simulated
frames from run to run. Re-running one mission for the videos charged 1.35/1.45/1.80/1.90 s where the campaign run
charged 1.45/1.55/1.70/1.80 s, and the two rolls ended differently — the campaign roll left the arena at 2/7, the re-run completed all 7 waypoints. (The campaign roll's exit followed a doubled-back rescue route, see the correction at the top: the differing delays are real, the different outcome went through the bug.) Every other arm reproduces bit-for-bit on the same partition. `R1L`'s row
therefore carries run-to-run variance the others do not; a fixed budget (`--latency-s 1.5`) would remove it. Its effective replanning period is therefore the measured
algorithmic latency rather than the nominal 1 s, and every route it commits to was planned from a pose the vehicle
has already left. (Latency on decisions that had to fall back to a rescue shape is under-counted: only the
successful planning call is timed.)

## 5. Result: periodic replanning against one decision per waypoint

*(all 120 rollouts; regenerate with `nav_analyze.py` + `nav_table.py`; `results.png` is the figure and
`tables.md` the same tables on their own)*

| arm | missions completed | waypoints reached | unsafe missions | legs with a slide | median time | decisions | sense-to-plan (p50) | travel time vs W | completed only by (arm / W) |
|---|---|---|---|---|---|---|---|---|---|
| W | 27/30 | 96.5% | 6 | 7/195 | 76.6 s | 7 | 4.3 s | - | - |
| R2 | 23/30 | 88.4% | 13 | 8/183 | 74.1 s | 38 | 4.4 s | -5.7 [-15.6, +3.2] | 3 / 7 |
| R1 | 24/30 | 87.9% | 15 | 13/181 | 75.6 s | 73 | 4.4 s | -3.3 [-11.9, +4.9] | 2 / 5 |
| R1L | 23/30 | 87.4% | 12 | 10/181 | 76.4 s | 51 | 4.2 s | +3.1 [-11.5, +19.1] | 2 / 6 |

| arm | dev: completed | dev: waypoints | unseen: completed | unseen: waypoints |
|---|---|---|---|---|
| W | 11/14 | 92.6% | 16/16 | 100.0% |
| R2 | 14/14 | 100.0% | 9/16 | 78.1% |
| R1 | 12/14 | 88.3% | 12/16 | 87.6% |
| R1L | 11/14 | 87.2% | 12/16 | 87.6% |

| arm | mission_complete | prolonged_blockage_terminated | terrain_bounds_exit |
|---|---|---|---|
| W | 27 | 3 | 0 |
| R2 | 23 | 2 | 5 |
| R1 | 24 | 2 | 4 |
| R1L | 23 | 2 | 5 |

| arm | render | back-project | candidates | corridors | risk model | total (p50) |
|---|---|---|---|---|---|---|
| W | 2.67 s | 0.05 s | 0.50 s | 0.33 s | 0.65 s | 4.40 s |
| R2 | 2.62 s | 0.04 s | 0.85 s | 0.29 s | 0.59 s | 4.36 s |
| R1 | 2.62 s | 0.04 s | 0.86 s | 0.29 s | 0.60 s | 4.36 s |
| R1L | 2.62 s | 0.04 s | 0.68 s | 0.30 s | 0.61 s | 4.26 s |

**The pipeline works.** Every arm drives a real mission end to end on nothing but live depth frames: 30 missions,
~200 legs, 150-228 m each, 10 arenas. One decision per waypoint completes 27 of 30 missions and reaches 96.5% of
waypoints, and on the four arenas that had never been driven, captured or inspected before it completes
**16 of 16**.

**Replanning more often did not improve completion in this campaign, but the arena exits that made the
difference were caused by a runner bug (see the correction at the top).** The plan-once arm fails three missions,
all by stalling on a hill after a long backward slide, and never leaves the terrain. The replanning arms stall no
more often (two missions each) but four or five runs each drive off the arena — every one of them after following
a rescue route that doubled back on itself. Legs with a backward slide are 7/195 (plan-once) against 8/183, 13/181 and
10/181 — no difference either. Travel time moves 3-6 s a mission in favour of replanning when planning is free,
and back the other way (+3.1 s) once the planning delay is charged; every interval includes zero.

The split by arena is worth stating rather than smoothing over: the plan-once arm loses three missions on the six
**development** arenas and none of the sixteen on the four **unseen** ones, while the replanning arms lose most of
theirs on the unseen arenas. That is not a generalisation gap in the model — the losses are arena exits, and they
cluster on eight of the thirty missions, seven of which are on new arenas. With 30 missions this is a small
sample of a failure mode with a known, fixable cause.

*Withdrawn explanation, kept for the record — the exits were the doubled-back rescue routes, not the lack of a
boundary term:* **Why the boundary failure belongs to replanning.** One-shot planning draws a route from one waypoint to the next,
and every waypoint is inside +-33 m of an +-40 m arena, so the vehicle never approaches the edge with a route that
points at it. Replanning re-anchors the route at wherever the vehicle *actually is*, including 2-3 m of accumulated
tracking error, and nothing in the objective knows the boundary exists: the risk model scores terrain, and the
boundary is a hard constraint inside the route validator, not a cost. The thirteen exits [sic: 14 exits on 8 missions] come from seven of the thirty missions and are not explained by those missions being closer to the edge (mean maximum waypoint radius
31.9 m against 31.7 m for the rest). The fix is to make the boundary a cost the planner can trade against, or to
let the model see it — the grid already marks everything beyond the terrain as unobserved, and training never
contained a route that went near it.

**The risk model is doing the work.** `R1rand` runs the identical loop — same sensing, same mask, same 256
candidates, same controller — and picks one of them at random. All 30 missions: **6 completed against 24, 45.2% of
waypoints against 87.9%, median 166 s against 75.6 s**, 28 of 30 missions with a backward slide and 580 s of
backward sliding in total (the model's 1 Hz arm: 13 legs of 181). Statuses: 14 stalls, 8 timeouts, 2 arena exits.
The median candidate in these pools carries a predicted unsafe probability near 0.9 while the model's pick is
usually below 0.01, and the drives agree.

**Scale of the campaign:** 120 continuous rollouts, 5,043 planning decisions, 2.7 h of simulated driving and
29.9 km driven. Every decision rendered a frame, back-projected it, generated 256 candidates, extracted 256 corridors
and ran a three-model ensemble.

## 5a. The same comparison after the rescue-route fix (2026-09-17)

The 120 rollouts were repeated on the workstation with doubled-back rescue routes rejected (`local_luffy/`,
OptiX rendering, 50 min wall; `local_luffy/tables.md`). Chrono is not bit-reproducible across machines, so this is
not a re-play of the campaign above but an independent repetition of the same 30 missions.

| arm | missions completed | waypoints reached | legs with a slide | median time | travel time vs W | stalls | arena exits | other failures |
|---|---|---|---|---|---|---|---|---|
| W | **27/30** | **96.0%** | 4/194 | 76.6 s | - | 2 | 1 | - |
| R2 | 25/30 | 93.5% | 12/191 | 77.1 s | +3.3 s [-6.5, +13.6] | 2 | 0 | 2 no route, 1 rollover |
| R1 | 25/30 | 89.9% | 11/184 | 75.9 s | +1.3 s [-6.6, +9.3] | 3 | 1 | 1 timeout |
| R1L | 22/30 | 86.9% | 10/181 | 82.0 s | **+14.5 s [+2.1, +27.8]** | 3 | 1 | 2 timeouts, 1 no route, 1 rollover |

- **The fix worked.** Arena exits over 120 runs: 14 (campaign, buggy) -> 8 plus a rollover (first local re-run,
  same bug) -> **3** (fixed). No route in the fixed re-run turns more than 6.1 deg between consecutive points.
- **The conclusion is unchanged, and now it is not confounded.** Planning once per waypoint completes 27/30 and
  reaches 96.0% of waypoints; replanning completes 25/30 (2 s), 25/30 (1 s) and 22/30 (1 s with the delay charged),
  and slides backwards on 10-12 legs against 4. On paired missions R2 completes 2 that W loses and loses 4 that W
  completes (R1: 1 and 3; R1L: 3 and 8) - differences this small are not resolvable with 30 missions, but there is
  no sign of a gain.
- **Replanning is not faster here, and charging the delay makes it slower.** +3.3 s and +1.3 s per mission are
  consistent with no difference; +14.5 s is not (interval excludes zero).
- **What replanning does buy** is visible per mission: it rescues `f104_nav_003`, where the plan-once arm leaves
  the arena at 2/6 while all three replanning arms finish, and `g203_nav_001`, where the plan-once arm stalls at
  4/6. It loses `g204_nav_001` and `g204_nav_003` (no valid route from a pose plan-once never visits) and
  `g223_nav_003` (a rollover). The failures move around; the total does not improve.
- **Unseen arenas** (16 missions): W 16/16, R2 13/16, R1 13/16, R1L 11/16. **Development arenas** (14): 11, 12, 12, 11.
- Per decision on the workstation: render 0.01 s, back-projection 0.05 s, candidates 0.38-0.69 s, corridors
  0.29 s, risk model 0.05 s (GPU), total 0.9-1.1 s with six runs sharing the CPU.

## 6. Planning from rest vs replanning while moving

The risk model reads corridor geometry and the commanded speed profile. It reads **no vehicle state at all**, so a
route gets exactly the same predicted risk whether the vehicle is parked or doing 6 m/s — while the real risk
plainly differs — and every label behind it comes from an episode that started at rest. Whether that matters was
measured on the runs themselves (`nav_moving_diag.py`), by asking whether a backward slide starts before the next
decision plus a 2 s tail.

| decision taken... | n | followed by a slide |
|---|---|---|
| at the start of the mission | 30 | 0 |
| at a waypoint | 153 | 1 (0.7%) |
| periodically, above 2 m/s | 802 | 10 (1.2%) |
| periodically, below 2 m/s | 147 | 78 (53%) |

(R2 arm, 1,132 decisions over 30 missions.) Replanning at speed is where almost all decisions are taken — 83% of
them are above 2 m/s — and it does not produce a burst of trouble: ten slide events follow 802 moving decisions,
the same rate as decisions taken at a waypoint from a rolling start. With ten events the *ordering* quality at
speed cannot be estimated from this data (the AUC moves between 0.63 and 0.82 depending on how many missions are
included, and I am not quoting it). The clear signal is the last row: a decision taken when the vehicle has
already dropped below 2 m/s is followed by a slide **half the time**, and no route choice fixes that — by then the
vehicle is already stuck, and the planner is being asked the wrong question.

**Conclusion: no targeted collection or retraining is warranted for the moving-start shift.** The place where
retraining *is* warranted is different, and section 8 measures it.

## 7. What this does not show

- **The world is static and, at every decision, fully visible.** The overhead camera sees the whole 80 m arena, so
  a decision taken at the start of a leg already has all the terrain information a decision taken halfway through
  would have. Replanning can only correct for *where the vehicle actually ended up*, not for new terrain. The
  exploratory limited-range arms (`W20`, `R1_20`) exist precisely to remove that advantage.
- **The overhead camera is a simulator convenience.** 110 m above the arena with a 47 deg field of view is not a
  sensor a vehicle carries. It is the sensing geometry every earlier result in this project used, kept so the
  frozen checkpoints see the input they were evaluated on.
- **No obstacles.** These missions place no rocks, trees or buildings; the hazard is terrain shape only.
- **30 missions and ~200 legs** cannot resolve small differences in completion or unsafe rate.
- **The leak check uses the privileged heightmap** as a reference for whether the exclusion zone still covers the
  vehicle. Nothing in the planning path reads it; the follower's reference path is lifted with heights read from
  the same depth frame the planner uses.
- **The risk model reads no vehicle state.** A route scores the same whether the vehicle is parked or doing
  6 m/s. That is measured, not assumed, in section 6.

## 8. Limited sensing range, and the one place retraining is indicated

The overhead camera sees the whole arena, which is not what a vehicle-mounted sensor gives. The runner can crop
the frame to a radius around the vehicle (`--sense-radius-m`), leaving the far half of a 25-35 m route unobserved.
Thirty missions cannot resolve what that costs, so it was measured where 1,200 labelled route choices already
exist — the held-out arenas' stored corridors, masked in corridor space exactly as the runner masks a live frame
(the vehicle is at the route start, so a sample at station *s* and lateral offset *l* is `hypot(s, l)` away), then
scored with the frozen checkpoints (`nav_eval_range.py`):

| usable radius | valid corridor | height model | direct-depth model |
|---|---|---|---|
| whole arena | 96% | 2.33% | 2.75% |
| 30 m | 66% | 5.67% | 5.83% |
| 25 m | 54% | 8.17% | 7.75% |
| 20 m | 42% | 10.00% | 9.58% |
| 15 m | 30% | 11.75% | 11.00% |

(*avoidable* unsafe picks: the picked-unsafe rate minus the 12.42% that no candidate in the pool avoids. Picking
at random gives 17.3%.) With the whole arena visible the planner removes ~85% of the avoidable risk; with a 20 m
sensor it removes ~43%. Both models degrade identically, so this is about missing terrain, not about depth versus
height.

Part of that loss is information the planner simply does not have, and part is distribution shift — these
checkpoints never saw an unobserved sample in training. To separate them, the whole matched training was repeated
on corridors masked at 20 m (same rows, labels, split by arena, seeds and budget; `nav_mask_datasets.py` then the
unchanged `sensor_train_v2.py`).

Retrained on 20 m corridors and evaluated at 20 m, the ensembles reach **8.33% avoidable** (direct depth, frozen
9.58%) and **8.67%** (height, frozen 10.00%) — a real gain of 1.25-1.33 points, but only ~18% of the 6.8-7.7
points that the limited range costs in the first place. So the loss is mostly **missing terrain, not distribution shift**: retraining on
range-limited data is worth doing if the sensing range is to be limited, and it is not a substitute for seeing
further. This is the one retraining that the evidence supports; the moving-start shift of section 6 does not need
one.

## 9. Videos

`videos/amd_heightmap_render/` (see its `README.md` for how to read a frame). Ten videos: two missions x four
planners, plus a four-way comparison per mission. Every one is a real continuous rollout, made with the runner
before the rescue-route fix; none of the routes in these eight rollouts turns more than 5.1 deg between consecutive
points, so none of them followed a doubled-back rescue route. A second set rendered from Chrono's RGB cameras on the workstation
(`videos/_superseded_v0_foldback/`; its videos are not in git, only the rescue-route check from its audit is) was withdrawn when its audit found the rescue-route bug; it is to
be regenerated from the fixed runner.

**Set 1, `f104_nav_002` — all four planners do the job.** 7 waypoints, 195 m of legs. Waypoint-only 7/7 in 76.2 s
with 7 decisions; 2 s 7/7 in 73.8 s with 35; 1 s 7/7 in 74.8 s with 69; latency-charged 7/7 in 71.2 s with 50 —
all four within 5 s of each other, no sliding anywhere. This is what the loop looks like when nothing goes wrong.

**Set 2, `f104_nav_001` — the same mission separates them.** 7 waypoints, 212 m. Waypoint-only reaches 5 of 7 and
**stops**: on leg 6 the route it committed to cannot be climbed, it slides backwards for 10.7 s and the stall rule
ends the run — it has no way to change its mind. Replanning every 2 s finishes in 63.0 s without ever sliding;
every 1 s finishes in 81.5 s, sliding for 7.5 s on the way and driving out of it. The latency-charged arm finished
this roll in 74.0 s, and its campaign roll left the arena at 2/7 after following a doubled-back rescue route (the bug in the correction at the top) — see also the non-reproducibility note in section 4.

Each single-planner video shows the planner's whole input (the height map back-projected from that decision's own
depth frame, white where the sensor saw nothing), a zoom on the footprint exclusion it applies to its own
corridors, the routes committed and the track driven, and a speed trace with a tick per decision. The comparison
videos put all four on one screen, synchronised on simulated time, each panel freezing with its outcome when that
run ends.

The 1 s videos also show the cost of replanning: the red route is re-drawn from the vehicle's actual pose every
second, consecutive picks differ by a median of 7.6 m over the first 30 m, and the speed command oscillates between
2 and 6 m/s because each decision is an independent `argmin` with no term for keeping the route it is already on.
In the latency-charged video the red route visibly starts *behind* the vehicle: it was planned 1.5-1.9 s ago from a
pose the vehicle has already left.

## 10. Reproducing

`RUNNING.md` has the exact commands. Code: `scripts/nav_online.py`, `scripts/nav_runner.py`,
`scripts/nav_missions.py`, `scripts/nav_tasks.py`, `scripts/nav_array.py`, `scripts/nav_analyze.py`,
`scripts/nav_moving_diag.py`, `scripts/nav_eval_range.py`, `scripts/nav_mask_datasets.py`,
`scripts/nav_latency_bench.py`, `scripts/nav_video.py`, `scripts/nav_figure.py`, `scripts/nav_table.py`.

## 11. What I would do next, in order

0. **Finish and analyse the fixed re-run** (`local_luffy/`, doubled-back rescue routes rejected) before drawing
   any conclusion about replanning versus planning once per waypoint.
1. **Give the planner the arena boundary** (lower priority after the correction: the exits it was meant to
   prevent were caused by the rescue-route bug). The objective contains nothing about the boundary. The cheapest version is a cost on how close a candidate's
   corridor comes to the edge of the observed grid; the principled version is to let the model see it — the grid
   already marks everything past the terrain as unobserved — which needs training routes that go near it, and there
   are none.
2. **Run the replanning comparison where replanning can actually pay.** With the whole arena visible at every
   decision, a decision taken at a waypoint has all the terrain information a decision taken halfway to it would
   have, so the only thing replanning can correct is the vehicle's own drift. The limited-range arms (`W20`,
   `R1_20`, built and ready, not yet run) remove that: at a 20 m sensing radius the far half of every route is
   unobserved when the leg starts and is revealed while driving. Section 8 shows offline that the terrain missing
   at 20 m costs 7.3 points of avoidable risk, so there is something for replanning to recover.
3. **Make a decision cheaper than the replanning period.** On the campaign nodes a decision costs 1.5-1.8 s of
   planner, and that is with the neural network already down at 42 ms on a GPU. Candidate generation (0.85 s) and
   corridor extraction (0.29 s) are plain numpy and are now the bottleneck.
4. **Add commitment to the objective.** Consecutive picks differ by a median of 7.6 m over the first 30 m of route,
   and the speed command oscillates between 2 and 6 m/s as a result. Scoring the route the vehicle is already on
   alongside the fresh candidates (`--keep-current --switch-margin`) is implemented and unrun.
