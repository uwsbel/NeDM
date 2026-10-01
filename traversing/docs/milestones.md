# Milestones: evidence, scope and limits

State as of 2026-09-28. Every number is a closed-loop drive in Chrono unless marked offline. Counts are recounted from
the per-task tables in [`../results/`](../results/README.md), which were rebuilt from the per-drive records and checked
against each study's own analysis files with no mismatches. Run `python scripts/traversing/analysis/recount_milestones.py` to
repeat the recount. Records named below are at the experiment commit [`901d6c9`][commit].
[evidence.md](evidence.md) lists every record, the code that produced it, and the release item that holds it.

Conventions:

- **Pair**: one start/goal task.
- **Goal reached safely**: the goal was reached without an unsafe event. The unsafe event is rolling back on a climb:
  at least 0.05 s in total backwards faster than 0.10 m/s under throttle, or backwards at 0.30 m/s or faster, after the first
  1 s. The Polaris study also counts the hull sitting more than 5 cm inside the soil for over 1 s.
- **Unsafe.** In every study except the tracker comparison, a drive that misses the goal also counts as unsafe. That
  includes stalls, timeouts and a spinning wheel digging through the soil layer. In the tracker comparison, unsafe
  means the run ended by rollover, prolonged blockage, a wheel digging through the soil or leaving the terrain;
  rolling back is not counted there, and a timeout is a failure but not unsafe. Tilt beyond 30 degrees is not part of
  any label.
- **Uncertainty.** Intervals are the study's own paired bootstrap over start/goal groups, or over terrain-feature
  clusters where the study used them. Test p-values from the studies' bootstraps are one-sided;
  exact McNemar p-values on discordant pairs are two-sided and marked so. Every study wrote its decision rules and
  analysis specifications before the drives they judged. The single-goal studies also locked each planner's routes by
  checksum before driving them; in the live navigation loop routes are chosen during the drive. Deviations from a plan
  are named under each milestone.
- **Pre-registered vs post hoc.** "Predeclared" marks a rule written before the drives. Anything else is post hoc.

---

## 1. A Chrono depth camera in the planning loop

**Status: achieved in simulation, on rigid ground, with a whole-arena overhead camera.**

**What was tested.** One continuous Chrono rollout per mission, with the vehicle never reset. At each decision:

1. Chrono renders one overhead depth frame with the HMMWV in view.
2. The frame is back-projected along each camera ray into a 0.156 m height grid. Cells the camera did not see stay
   empty and are marked invalid.
3. The vehicle's own footprint plus a 1.5 m margin is marked as unseen.
4. Corridors for 256 candidate routes are cut from that one frame, and a frozen three-network risk ensemble scores
   them.
5. The lowest-risk route goes to the path follower. Nothing is carried over between decisions.

Setup:

- **Missions:** 30 missions of 5-8 waypoints (199 waypoints in total, 150-228 m of legs) on 10 rigid arenas:
  - f104 (4 missions);
  - the other three arenas the depth model was trained on (2 each);
  - two development arenas the model did not train on (2 each);
  - four arenas generated for this test and never driven before (4 each).
- **Arms:** four decision schedules, each driving all 30 missions: once per waypoint; every 2 s; every 1 s; every
  1 s with the measured planning delay charged to the simulation.
- **Frozen before the drives:** models, route generator, corridor rules and stop rules.
- **Run:** 120 rollouts on the workstation with OptiX rendering.

| Decision schedule | Missions completed | Waypoints reached | Completed without a backward slide | Median mission time |
|---|---|---|---|---|
| **Once per waypoint** | **27/30** | 191/199 | 25/30 | 76.6 s |
| Every 2 s | 25/30 | 186/199 | 18/30 | 77.1 s |
| Every 1 s | 25/30 | 179/199 | 19/30 | 75.9 s |
| Every 1 s, planning delay charged | 22/30 | 173/199 | 19/30 | 82.0 s |

- **Never-seen arenas.** Planning once per waypoint completed all 16 missions on the four arenas generated for the
  test (105/105 waypoints; 15/16 without a slide). On the 20 missions on arenas absent from the risk model's training
  it completed 19.
- **Faster replanning did not help.** With the whole arena visible at every decision, a replan has no terrain
  information the waypoint decision lacked. Replanning mid-leg also often left the route builder with no valid
  candidate (140 of 2,266 decisions every 1 s, 127 of 2,361 with the delay charged, none when planning once per
  waypoint), so this result is specific to this route builder. Against planning once per waypoint, the 2 s schedule wins 2 missions and
  loses 4, and the delay-charged schedule wins 3 and loses 8. With the delay charged, mission time rises by 14.5 s on
  average, 95 % interval [+2.1, +27.8] over all 30 missions. Mission time includes failed missions that ran long: on
  the 19 missions both schedules completed, the difference is +7.4 s, 95 % interval [-2.8, +18.8], which includes
  zero.
- **A runner defect was found and fixed before this run.** When no normal route was valid, an emergency route could
  double back on itself and still pass the route check. Every one of the 14 arena exits in the first run of these 120 drives,
  on the AMD cluster before the fix, followed such a route. Routes that turn more than 45 degrees between consecutive points are now rejected. Over the
  5,762 routes of the corrected run, the largest turn between consecutive points is 6.6 degrees, and arena exits fell
  from 14 to 3 per 120 drives. Planning once per waypoint completed the most missions both times: 27/23/24/23 on
  the cluster before the fix, 27/25/25/22 after it. Total failed missions only went from 23 to 21 of 120; the rest
  moved to no-route, rollover and timeout endings.
- **Depth-to-map geometry was corrected.** Each pixel is now placed along its measured ray instead of on assumed flat
  ground. The mean height error against the arenas' authored heightmaps fell from 0.027-0.042 m to 0.0071-0.0087 m on
  six arenas. In an offline test on 1,200 route choices, a model reading the corrected depth directly chose about as
  well as one reading height: 14.92 % against 14.50 % unsafe picks, difference +0.42 points [-0.42, +1.25].
- **Frames with the vehicle in view need no retraining.** With the footprint mask, vehicle-in-view frames gave
  corridor inputs identical to vehicle-free frames. Every valid sample differed by 0.0000 m, and the same route was
  picked in 20/20 cases for both models. A short driving check had 0 unsafe drives in 63 (40 without the mask, 23
  with it). In all 6,035 live decisions, no sensed cell outside the mask within 8 m of the vehicle sat more than
  0.08 m above the true terrain.
- **Latency.** On an AMD MI350X the whole planner takes 0.42 s per decision, about 2.4 decisions per second. The risk
  networks take 0.042 s of that; candidate generation and corridor extraction take most of the rest. Rendering the
  depth frame takes about 2.6 s with CPU software ray tracing on the cluster and under 0.01 s with OptiX on an RTX 5090.
  In three of the four schedules the simulation pauses while planning. The delay-charged schedule pays the measured
  back-projection and planning time, not the render: on the workstation, with six runs sharing the CPU, a median
  1.05 s per decision (95th percentile 1.55 s).

**What this does not show.**

- Onboard or limited-visibility sensing. The camera is fixed 110 m above the arena centre, looking straight down. It
  sees the whole 80 m arena at every decision, with ideal noise-free depth, and every decision uses the simulator's
  exact vehicle pose.
- The effect of a limited view, tested offline on the same 1,200 route choices. Keeping only 20 m around the vehicle
  raised avoidable unsafe picks from 2.75 % to 9.58 %, and retraining on 20 m views recovered only 1.25 points. The
  live limited-range schedules were implemented but never run.
- A benefit from replanning, obstacles, deformable soil, or the later shared history model, iterated search and learned
  tracker in the loop. This loop uses the frozen depth-reading model with one-pass ranking and the PID follower.
- Differences of 2-3 missions are within run-to-run variation. The same 120 tasks with the same (pre-fix) runner, on
  the cluster and on the workstation, agreed on per-mission completion in 92 of 120 runs.
- The random-pick control (same loop, random candidate: 6/30 missions) was run only before the runner fix, and its raw
  files are on the cluster only. It is not verified here.

**Records:** [nav report][nav-report] (section 5a: the corrected run), [nav plan][nav-plan],
[sensor geometry report][sensor-report], [vehicle-in-view pilot][vehicle-pilot], [offline range sweep][range-choice],
[planner timing][bench]; table [`m1_navigation_missions.csv`](../results/m1_navigation_missions.csv).

---

## 2. One risk model for rigid ground and soil, and the decision-timing finding

**Status: achieved on the f104 arena for the final configuration. The history input's own closed-loop contribution
was not shown.**

**The model.** A causal GRU reads the last 2 s of driving, 40 samples at 50 ms:

- 12 observable vehicle signals: forward and sideways speed, roll, pitch, the three body rates, four wheel speeds and
  engine speed;
- the 3 applied controls (steering, throttle, brake).

It condenses them into 16 numbers, which join the route context of the corridor risk network ([architecture](architecture.md#history-input)).
The model is trained on rigid and soil drives together. An auxiliary head predicts the ground type from the 16
numbers, but only as a training loss. At deployment the planner is never told which ground it is on.

**The suite.** 800 start/goal pairs on f104, identical on both grounds:

- **600 fresh:** new start/goal pairs, created on 09-21;
- **200 reused:** from the first soil study, and used earlier to choose the search method.

Every comparison drives the same pairs with routes locked before driving. Search is iterated sampling, 4 rounds of
64 candidates, unless gradient refinement is named. The follower is the PID path follower.

**Decision protocols.**

- **Standing start:** the planner decides before the vehicle moves.
- **After an approach:** the vehicle first drives straight toward the goal at 3 m/s for 3 s, 1 s or 0.5 s. The
  planner then decides from the recorded state, and the drive continues on the chosen route.

Soil, goal reached out of 800:

| Model | Standing start | After 3 s | After 1 s | After 0.5 s | After 0.5 s + gradient refinement |
|---|---|---|---|---|---|
| Soil-only model (at rest: the earlier deployed soil planner, trained on other data; 3 s and 1 s: retrained on the shared model's rows) | 766 | 664 | 749 | – | – |
| Model told the true ground type | 769 | 675 | 763 | 769 | – |
| Both grounds, no history, no ground type | – | 668 | 741 | 750 | – |
| **Shared history model** | 750 | 671 | 746 | 753 | – |
| Shared history model, history blanked | – | 671 | – | – | – |
| Shared history model, retrained with early-decision rows | – | – | 742 | 766 | **780** |
| Transformer over route stations and history | – | – | 758 | 754 | 778 |

On rigid ground every arm is at 99-100 % goal reached. The final configuration reaches **800/800**, and the
original shared model after the 3 s approach reaches 797/800.

- **Final configuration:** soil **780/800** (fresh 582/600) against the 3 s baseline's 671/800. The failure rate falls
  by 13.6 points, 95 % interval [-16.1, -11.1]. The predeclared rule on the fresh 600 passes (one-sided bound -11.0).
  Terrain-clustered interval [-20.2, -8.1]; all 9 terrain clusters improve.
- **Goal reached is not zero risk.** Under the study's label, 21/800 soil drives and 1/800 rigid drives of the final
  configuration were unsafe:
  - soil: the 20 failures (16 wheels digging through the soil layer, 3 blockages, 1 timeout), plus 1 goal reached
    after rolling back;
  - rigid: 1 goal reached after 20.7 s of rolling back under throttle.
- **Deciding earlier explains most of the gain.** The 671 → 780 gain splits into three steps; only the first keeps
  the original model unchanged:
  - 671 → 753 by deciding after 0.5 s instead of 3 s (101 pairs gained, 19 lost);
  - 753 → 766 by retraining with early-decision rows (26 vs 13, exact McNemar p = 0.053 two-sided; not a
    predeclared test);
  - 766 → 780 by gradient refinement (22 vs 8, p = 0.016 two-sided on all 800; on the fresh 600, 14 vs 6, not
    significant).

  The 3 s approach carried vehicles toward steep terrain before the planner decided. In the 86 pairs where every 3 s
  planner trained with soil data failed, the first steep cell was 2.0 m ahead at the decision, against 8.0 m where every planner succeeded.
  At least one planner completed 76 of those 86 pairs from a standing start.
- **Against planning from rest, the gain is small.** The original shared model decides after 0.5 s about as well as
  from rest (753 vs 750). With the same search, the retrained model deciding after 0.5 s ties the earlier soil-only
  planner deciding at rest (766 vs 766, 19 vs 19 pairs). The final configuration beats that planner by 780 vs 766
  (22 vs 8 pairs, p = 0.016 two-sided), but only the final configuration had gradient refinement. "83.9 % → 97.5 %"
  is a gain over the 3 s protocol, not over planning from rest.
- **After a common 3 s approach, one shared model matched the single-ground models.** Soil 671 vs 664/800, one-sided
  bound +0.5 points against a 3-point margin. Rigid 797 vs 796/800. Drive times were within 1-2 %. Both predeclared
  rules pass.
- **From a standing start, the shared model missed its predeclared rules.**
  - Soil: 750/800 against 766/800 for the earlier soil-only planner. The one-sided bound is +3.5 points on the fresh
    600 (+3.4 on all 800), against a 3.0-point margin.
  - Rigid: the shared model drove 14 % slower than the earlier rigid-only planner, against a 10 % bound.

  Without a ground label it still recovered 13.6 of the 15.6 soil points between the right and the wrong single-ground
  planner.

**What this does not show.**

- **That the history input is needed.** After the 3 s approach, the same model with its history blanked also reaches
  671/800, and a model with neither history nor ground type reaches 668/800. At 0.5 s the original history model and
  the no-history model are level (753 vs 750). No blanked or no-history version of the retrained model was driven.
  Offline, once 2 s of motion exist, the history model ranks routes within a pair better than the no-history model
  (within-pair AUC 0.989 / 0.986 against 0.969 / 0.968, rigid / soil).
- **That the 16 numbers estimate soil properties.** They separate the two grounds almost perfectly offline, but partly
  through simulator set-up differences: engine idle speed alone separates them 0.1 s after release, before the vehicle
  moves. Only one soil setting was ever used.
- **A confirmatory result.** At least 25 planner configurations were driven on this same suite while the final one was
  chosen. The 200 reused pairs had already been used to choose the search method. No untouched suite was driven
  afterwards.
- **A live 0.5 s decision.** Planning ran offline between two drive passes: gradient refinement takes a median 2.56 s
  per decision on an RTX 5090. The drive assumes the route switches instantly at 0.5 s. The final configuration was
  never driven from a standing start.
- **Other terrain.** One arena. Every metre of f104 was driven in training, so "fresh" pairs are new start/goal
  pairs, not new terrain.
- **About the name.** The soil-improvement report calls the final model "short-window history", but its encoder still
  spans 40 samples. At a 0.5 s decision only 10 hold real motion. What changed is the training rows.

**Records:**

- Reports: [shared-model report][gen-report], [soil-improvement report][ci-report];
- standing-start results: [soil][a3-crm], [rigid][a3-rigid];
- results after the 3 s approach: [soil][a5-crm], [rigid][a5-rigid];
- early-decision summary: [text summary][s2-txt];
- the final per-pair results were local-only and are now in the data release, listed in [evidence.md](evidence.md#milestone-2-shared-rigidsoil-model-and-early-decision).

Tables: [`m2_shared_risk_soil.csv`](../results/m2_shared_risk_soil.csv) and
[`m2_shared_risk_rigid.csv`](../results/m2_shared_risk_rigid.csv).

---

## 3. A learned route tracker trained inside the NRD, against PID

**Status: partial. Better tracking than PID on rigid ground. On soil it does not replace PID, and the predeclared
replacement test fails on both grounds.**

**What was tested.**

- **Dynamics model (NRD).** A 4.9-million-parameter causal transformer, fitted to recorded f104 drives on both grounds.
  Every 50 ms it predicts the change in a 17-number vehicle state from:
  - the last 16 steps;
  - the command;
  - an 8 x 8 patch of terrain heights over 12 m x 12 m;
  - a ground-type tag.

  Round 2, the reported one, added 1,500 perturbed PID drives per ground and 1,000 rigid drives of the round-1 tracker
  to the recordings. It passed its three predeclared validation gates on both grounds: stall escape, displacement and
  brake response, over 3 s fed-back rollouts.
- **Tracker.** Warm-started by imitating the stock PID's recorded commands on training drives, then trained with PPO
  only inside the frozen NRD, on 2,048 parallel imagined fragments of 1-3 s. The fragments
  start from recorded training drives, half on each ground.
  - It was exported as a NumPy network. In Chrono it runs every 50 ms without the NRD and without a ground label.
  - It matches the torch policy to 5e-7.
- **Test.** 423 designed routes from 55 sealed test start/goal groups on f104, each driven on both grounds. Before any
  tracker drive the routes were split into:
  - **feasible** (141): the PID had completed them on both grounds in earlier recordings;
  - **hard** (282): the rest.
- **Arms:**
  - the stock PID path follower, which updates its command inside each 50 ms interval;
  - the same PID with its command held for 50 ms, the timing the tracker must use;
  - the round-2 tracker.

Feasible routes (141):

| Ground | Controller | Completed | Unsafe | Cross-track error | Speed error |
|---|---|---|---|---|---|
| rigid | stock PID | 141 | 0 | 0.200 m | 0.464 m/s |
| rigid | PID held 50 ms | 141 | 0 | 0.200 m | 0.488 m/s |
| rigid | **learned tracker** | **141** | **0** | **0.106 m** | **0.348 m/s** |
| soil | stock PID | 140 | 1 | 0.451 m | 1.161 m/s |
| soil | PID held 50 ms | 133 | 8 | 0.537 m | 1.287 m/s |
| soil | learned tracker | 129 | 12 | 0.379 m | 1.140 m/s |

Cross-track error is the mean over routes of each route's 5 % Winsorised mean distance from the reference waypoints
to the driven path (distances beyond the 5th and 95th percentiles are clipped to them, not dropped), with unreached
waypoints counted as 6 m. Speed error is the mean absolute error from 1 s onwards.

- **Rigid, feasible routes: every predeclared rule passes.**
  - Cross-track ratio 0.53 (one-sided bound 0.57, rule < 0.90).
  - Speed-error ratio 0.75 (bound 0.80, rule < 1.10).
  - No completion or safety loss.
- **Rigid, hard routes: the completion bound fails.** 229 vs 231 of 282 completed. The one-sided bound is -4.3 points
  against -3 allowed. The tracker ends fewer runs unsafe, 29 vs 46, but partly because its failures end differently:
  24 timeouts (which count as failures but not as unsafe) against 5, and 53 failed runs against 51 in total.
- **Soil, feasible routes: completion, safety and tracking rules all fail.**
  - Completion 129 vs 140, bound -11.3 points. 11 routes were completed only by the PID and none only by the tracker.
  - Unsafe 12 vs 1: 9 wheels digging through the soil and 3 blockages. 10 of the 12 unsafe endings came from the
    stop rules all arms share.
  - Mean cross-track error was lower, but the ratio's one-sided bound, 0.99, is not below the 0.90 rule.
- **Soil, hard routes.** The tracker completes 25 vs 1 of 282. The stratum was chosen by PID failure, which favours
  any other controller: the held PID also completes 7.
- **Part of the soil deficit comes from the 50 ms hold, not the policy.** The held PID alone loses 7 feasible soil
  routes, and 5 of the tracker's 11 losses are shared with it.
- **Busier actuation.** On rigid feasible routes the tracker changes its command 3.3 times as much per step, and its
  median positive engine work is 13 % higher. Throttle and brake are both above 0.01 on 30.7 % of its steps, and
  both above 0.05 on 5.0 %. The PID never applies both.
- **Round 1** had a weaker speed reward and a dynamics model trained on PID drives only. It already cut rigid
  cross-track error (0.111 m) but tracked speed worse than PID and completed only 108/141 feasible soil routes.

**What this does not show.**

- **Identical execution.** The stock PID refreshes its command inside each 50 ms interval; the tracker and the held
  PID hold one command per interval. A 40 s any-throttle standstill stop applied only to the tracker and the held PID,
  although the plan required the same stop rules for all arms. It ended 14 of the tracker's rigid hard-route runs and
  2 of its feasible soil runs. The analysis's automatic check that every arm used the same actuator limits could
  not run: no limit metadata was recorded, so it covered 0 routes. The recorded
  commands of all arms stay in the same range and steering-rate limit.
- **A fair completion comparison on feasible routes.** They were chosen because the stock PID had completed them, and
  its re-drives reproduce the earlier recorded end status on 423/423 rigid and 419/423 soil routes. On the feasible
  routes it therefore completes 141/141 and 140/141 almost by construction.
- **Reproducibility across runs.** The round-2 tracker drives ran in separate cluster jobs from the reused PID drives.
  Chrono soil runs are not bit-reproducible across nodes: re-driving the stock PID reproduced 369/423 soil
  trajectories exactly and 419/423 end states.
- **Generality.** One PPO seed and one NRD fit per round. f104 only: new start/goal groups, not new terrain. The PID
  kept its default gains. The tracker was never combined with the route planner, whose risk labels describe PID
  drives.

**Records:**

- report: [shared-model report][gen-report] (tracker sections);
- per-route results: [rigid][b0v2-rigid], [soil][b0v2-crm];
- the [route suite][track-suite];
- policy and NRD records: [policy][policy-meta], [NRD config][nrd-config], [NRD gates][nrd-metrics].

Table: [`m3_tracker_routes.csv`](../results/m3_tracker_routes.csv).

---

## 4a. Arenas the planner never trained on (HMMWV)

**Status: achieved, with a small gain within one terrain family. On soil the absolute rate sits at the 90 % bar.**

**What was tested.**

- **Planners compared:**
  - f104 only, with two independently trained ensembles;
  - two arenas;
  - three arenas (f104, g203, g228) with the same number of training start/goal groups, again two ensembles;
  - three arenas with all their data.

  g203 and g228 are the two most f104-like of 40 earlier generated arenas.
- **Test arenas.** Eight arenas from 40 new generator seeds, chosen by a script before anyone looked at them: the four
  nearest to f104 and four spread ones.
- **Protocol.** One risk network without history per ground type, planning once from a standing start by iterated
  sampling, then driven by the PID follower.
  - Soil: 1,000 pairs, speed free.
  - Rigid: 2,000 pairs at a forced 2 m/s, so the planner only chooses the path, plus a speed-free read-out.
- **Statistics.** Four declared tests, Holm-corrected, with intervals from a bootstrap over (arena, terrain-feature)
  clusters. Two-ensemble arms are per-pair means over both ensembles.

| Planner | Soil: goal not reached (1,000) | Rigid 2 m/s: unsafe (2,000) | Rigid speed free: not reached |
|---|---|---|---|
| f104 only (mean of 2 ensembles) | 12.05 % | 8.325 % | 0.60-0.65 % |
| Two arenas, same data | 12.0 % | 6.60 % | 0.25 % |
| **Three arenas, same data** (mean of 2) | **9.70 %** | **6.175 %** | 0.25-0.40 % |
| Three arenas, all data | 9.40 % | 5.35 % | 0.30 % |
| Straight route (6 m/s on soil and in the speed-free column, 2 m/s in the 2 m/s column) | 40.0 % | 54.5 % | 1.60 % |

- **All four declared tests pass.**
  - Soil: -2.35 points, 90 % interval [-3.7, -1.1], Holm p 0.004.
  - Rigid: -2.15 points, [-3.4, -1.0], Holm p 0.004.
  - All data against f104 only: soil -2.65, rigid -2.98.
- **The rigid gain is entirely fewer backward slides on climbs.** Goal reaching at 2 m/s is unchanged, 3.075 % vs
  3.025 % not reached.
- **The gain does not grow step by step.** On soil the second arena added nothing (12.05 → 12.0 %) and the third
  brought the gain. On rigid ground most of it came with the second (8.325 → 6.60 %). More data on the same three
  arenas added little: soil -0.3 points, rigid -0.8, both inside a ±2-point band.
- **Rigid speed free is at the ceiling for every planner.** The three-arena planner's drives were 1.20 times as long as
  f104-only's, which fails the declared 1.10 time check. The all-data planner passes at 1.065.
- **Against the 90 % bar (post hoc).**
  - Soil, pooled goal reached safely: 90.3 % (three arenas, same data), 90.6 % (all data), 87.9 % (f104 only). Five and
    four of the 8 arenas are below 90 %.
  - Rigid at 2 m/s: the three-arena planners stay at or above 90 % on every arena; the lowest is 90.0 %, on g247.
    The two-arena planner reaches 89.2 % on g247. The f104-only planner falls below 90 % on g258 and g247.
  - Pooled, every rigid planner is above 90 % (91.7-94.7 %).

**What this does not show.**

- **New kinds of terrain.** Every arena comes from the same hill-and-crater generator. The planner reads heights
  through a flat-ground map lookup whose error grows off f104: 0.050 m on f104, up to 0.094 m on the test arenas.
- **A reliable per-arena verdict.** Two identically trained f104-only ensembles differ by up to 6.4 points on one
  arena. On soil, the headline survives the estimated training noise but not a noise twice as large.
- **Closing the gap to the training arena.**
  - Every planner that was also driven on held-out f104 pairs (the first ensemble of each two-ensemble arm, and the
    all-data planner) does worse on the unseen arenas: soil 3.0-5.0 % against 9.4-11.5 % not reached.
  - On rigid ground the three-arena planners also do worse there than on held-out g203/g228 pairs.
  - On soil those held-out pairs are harder than the test arenas for every planner.
- **Calibration.** Off its training arena the f104-only planner predicted about 1.1 % failure for its picks on unseen
  soil, against 11.5 % observed.
- **The later protocol.** All runs start from rest. The early decision and history model of milestone 2 were not used.
  Soil models used only the first 7 of 13 routes per training group.

**Records:**

- write-ups: [terrain and Gator report][ag-report], [soil results][ag-soil], [rigid results][ag-rigid];
- frozen-spec analyses: [soil][ag-soil-json], [rigid][ag-rigid-json];
- the [Holm family][ag-family].

Tables: [`m4_unseen_arenas_hmmwv_soil.csv`](../results/m4_unseen_arenas_hmmwv_soil.csv) and
[`m4_unseen_arenas_hmmwv_rigid.csv`](../results/m4_unseen_arenas_hmmwv_rigid.csv).

---

## 4b. Other vehicles: Gator, Polaris and an M113 smoke test

**Status: the collection → training → planning → Chrono pipeline was reused by retraining for each vehicle. The
Polaris meets the 90 % bar on soil; the Gator does not. This is not one model for all vehicles.**

**How vehicles were swapped.** A small wrapper swaps the vehicle at run time around the unchanged soil collector; the
route follower, stop rules and labels stay those of the HMMWV. On soil, the Gator's and Polaris's wheels meet the
particles as plain cylinders calibrated to the HMMWV's settled sinkage (the HMMWV uses its rigid tyre mesh). The
M113's track shoes meet the soil as flat, uncalibrated pads, at a 0.5 ms step. No vehicle body touches the soil.

**Data.**

| Vehicle | f104 soil routes | Simulated soil hours | f104 rigid routes | Goal not reached on the same pre-planned collection routes |
|---|---|---|---|---|
| HMMWV | 15,235 | 91.5 | 24,000 | 68.05 % |
| Gator (Chrono stock: rear drive, 14 kW, small wheels) | 15,235 | 141.1 | 24,000 | 88.15 % |
| Polaris (Chrono stock: 4WD, limited-slip) | 15,235 (15,229 valid) | 93.0 | none | 14.95 % |

The last column covers the 15,229 collection routes valid for all three vehicles. These are designed and random routes,
many deliberately aggressive, not planner-chosen ones.

**Smoke test** on 144 f104 soil routes, each vehicle paired route by route with the Gator driven again on the same
routes (in the same launch for the Polaris; the M113 ran in its own jobs, and a Gator re-drive in its launch gave the
same verdicts). Goal not reached:

| Vehicle | Not reached | Verdict |
|---|---|---|
| Gator | 135/144 | reference |
| **Polaris** | **24/144** | better |
| Polaris, power-corrected driveline | 20/144 | holds |
| Polaris, open differentials | 37/144 | holds |
| M113, stock | 129/144 | not better |
| M113, 4x lower gearing | 69/143 | better, at about 4 times the Gator's simulation cost |

The M113 went no further than this test: no collection and no planner.

**Planners on the 800-pair f104 soil suite.** Each planner uses a five-network risk ensemble without history, trained
on the vehicle's own f104 drives. It plans once from a standing start and was driven once.

| Vehicle driven | Planner | Reached | Reached safely |
|---|---|---|---|
| Gator | trained on Gator drives | 539/800 | 536/800 |
| Gator | same, with gradient refinement | 544/800 | 543/800 |
| Gator | trained on HMMWV drives | 349/800 | 349/800 |
| Gator | straight route, 6 m/s | 114/800 | 114/800 |
| HMMWV | trained on HMMWV drives | 765/800 | 765/800 |
| HMMWV | same, with gradient refinement | 770/800 | 770/800 |
| **Polaris** | **trained on Polaris drives, with gradient refinement** (declared) | 798/800 | **798/800** |
| Polaris | same, without refinement | 800/800 | 800/800 |
| Polaris | power-corrected driveline, same routes | 793/800 | 793/800 |
| Polaris | straight route, 6 m/s | 796/800 | 796/800 |

**The Polaris on 8 unseen arenas** (the arenas of milestone 4a, 1,000 soil pairs, planner trained on f104 only):

| Planner | Reached safely | Lowest arena |
|---|---|---|
| **Own planner, with gradient refinement** (declared) | **997/1,000** | 124/125 |
| Own planner, without refinement | 1,000/1,000 | 125/125 |
| Straight route, 6 m/s | 970/1,000 | 117/125 |

On the same pairs, the HMMWV planners of milestone 4a reach 874-906/1,000, and the HMMWV on the straight route 600.

- **Gator.**
  - Its own planner beats the HMMWV-trained one by 23.8 points of goal-not-reached, 90 % interval [-30.4, -16.5],
    Holm p 0.001, winning 206 pairs and losing 16. So retraining matters, and the HMMWV's risk ranking also transfers
    partly: 349 against 114 for the straight route.
  - It stays far below the bar, and the limit is mostly the vehicle, which stalls on climbs the HMMWV completes.
  - The model also contributes: in 127 of its 261 recorded failures it had rated the chosen route at most 50 % risky.
  - Neither gradient refinement (544) nor more data brought it near 90 %. Wider route shapes are estimated at about
    74 % from 361 re-driven pairs.
  - On rigid f104 ground its own rigid planner reached the goal safely on 800/800, but the straight route at 6 m/s
    already reaches 795/800 (797 reached), so rigid ground cannot separate planners ([rigid results][ag-rigid],
    section 3.1; not in the compact tables).
  - It was never tested off f104.
- **Polaris on f104.** 798/800 (fresh pairs 599/600), 95 % interval [99.4, 100.0] %.
  - The straight route already reaches 796/800, so the suite cannot separate planners for this vehicle. Planner
    against straight route: -0.2 points, no meaningful difference.
  - The "meets the bar" verdict uses the decision rule as amended before these drives. The original plan wording also
    asked the planner to beat the straight route, which it does not do on f104.
- **Polaris on unseen arenas.** 997/1,000, and every arena at 124 or 125 of 125. Here the planner beats the straight
  route by 2.7 points, 95 % interval [-4.6, -1.2] on failure, Holm p 0.0005, winning 30 pairs and losing 3. Most of the success is
  the vehicle's mobility.
- **Gradient refinement did not help the Polaris and made it slightly worse.** Sampling alone reached 800/800 and
  1,000/1,000. All 5 failures of the refined planner were on routes the refinement had changed. On f104 its changed
  routes took a median 1.12 times as long, and 14.4 % of its 800 drives tilted past 30 degrees against 11.4 % for
  sampling alone. Tilt is not part of "reached safely".

**What this does not show.**

- **One model for several vehicles.** Each vehicle is retrained on its own drives.
- **Rigid ground for the Polaris.** No rigid Polaris data was collected.
- **Physically exact vehicles and soil.**
  - Chrono's stock Polaris driveline gives the wheels about 16 times the engine's power; all Polaris training and
    planner drives used it. The power-corrected variant reached 793/800 on the same f104 routes but was never trained
    or driven on the unseen arenas.
  - The soil is a 0.24 m particle layer over a rigid floor. The wheels meet it as calibrated cylinders, and the body
    does not touch it; this may favour light vehicles.
  - The Polaris has a limited-slip split, the HMMWV open differentials.
- **Calibrated risk.** The Polaris model predicted a mean 0.0014 % failure for its f104 picks, against 0.25 %
  observed. The Gator model predicted 18 %, against 32.6 % observed.
- **Sample size.** One ensemble per vehicle.

**Records:**

- Gator: [Gator results][ag-gator], [Gator analysis][ag-bfull];
- Polaris and M113: [report][ov-report], [smoke test][ov-smoke], [collection][ov-collection], [planner][ov-planner],
  [f104 analysis][ov-f104], [unseen-arena analysis][ov-unseen].

Tables: [`m4_vehicles_f104_soil.csv`](../results/m4_vehicles_f104_soil.csv),
[`m4_polaris_unseen_soil.csv`](../results/m4_polaris_unseen_soil.csv) and
[`m4_vehicle_smoke.csv`](../results/m4_vehicle_smoke.csv).

[commit]: https://github.com/uwsbel/NeDM/tree/901d6c9423a16c0fafc3d60056065415d5a725f2
[nav-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/fdm_f104_50h_20260909/nav_v1/REPORT.md
[nav-plan]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/fdm_f104_50h_20260909/nav_v1/PLAN.md
[sensor-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/REPORT.md
[vehicle-pilot]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/VEHICLE_PILOT.md
[range-choice]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/fdm_f104_50h_20260909/nav_v1/range_choice.json
[bench]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/fdm_f104_50h_20260909/nav_v1/bench_cuda.json
[gen-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/REPORT.md
[ci-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/crm_improve_20260922/REPORT.md
[a3-crm]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/A_adapt/a3/results_crm_A0A3.json
[a3-rigid]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/A_adapt/a3/results_rigid_A0A3.json
[a5-crm]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/A_adapt/a5/results_crm_A5.json
[a5-rigid]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/A_adapt/a5/results_rigid_A5.json
[s2-txt]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/crm_improve_20260922/s2/results_s2_crm_vs3s.txt
[b0v2-rigid]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/B_tracker/b0/results_rigid_b0v2.json
[b0v2-crm]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/B_tracker/b0/results_crm_b0v2.json
[track-suite]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/B_tracker/suite/tracking_suite.json
[policy-meta]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/B_tracker/ppo_v2/policy_meta_999.json
[nrd-config]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/B_tracker/nrd_tag_v3/config.json
[nrd-metrics]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/B_tracker/nrd_tag_v3/metrics.json
[ag-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/REPORT.md
[ag-soil]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/RESULTS_soil.md
[ag-rigid]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/RESULTS_rigid.md
[ag-soil-json]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/e6/analysis/results_soil_v1.json
[ag-rigid-json]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/e6/analysis/results_rigid_v1.json
[ag-family]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/e6/analysis/family_v1_S2.json
[ag-gator]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/RESULTS_gator_full.md
[ag-bfull]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/e6/analysis/results_soil_v1_Bfull.txt
[ov-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/REPORT.md
[ov-smoke]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/RESULTS_smoke.md
[ov-collection]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/RESULTS_collection.md
[ov-planner]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/RESULTS_planner.md
[ov-f104]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/e6/analysis/results_ov_v1.txt
[ov-unseen]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/e6/analysis/results_unseen_v1.txt
