# gen_v1 — longer tasks and new arenas (pre-registration, written 2026-09-14 before any Chrono run of this plan)

Frozen for everything below: the night-2 risk model `night2_v1/final/N2_s{0..4}.pt` (no retraining), the night-2
route proposal `scripts/f104_n2_sampler.py`, and the collector physics/driver/stop policy of `source_v1`.
Planner code packaged in `scripts/gen_planner.py`. Map source for ALL arenas including f104: the arena heightmap
encoded like the static depth map (checked on 40 f104 hazard pools before this plan: identical pick 32/40, pick in
depth-map top-5 40/40, logit correlation median 0.987).

Labels (unchanged from night 2): fail = not goal_reached; unsafe = fail OR any frame after the 1 s settle with
(vx < -0.10 and throttle > 0.3) or vx < -0.30 (throttle = action column 1); tilt30 = max(|roll|,|pitch|) > 30 deg
after the settle.

## A. Generalisation to new arenas (task 3)

Arenas. 40 seeds (201-240) of the f104 terrain-family generator (`scripts/traverse_wp7_arenas.py`, difficulty 1.0),
ranked by distance to f104 on slope cap, roughness, roughness correlation, hill and crater counts, 99th-percentile
slope, flat fraction and height range (`gen_v1/arena_similarity.json`). The five closest are used:
g228, g203, g217, g216, g231. f104 itself is the in-arena reference.

Test start/goals. 200 per arena, hill/crater strata only, same gates and route families as the frozen campaign
generator (`scripts/gen_cases.py`). f104's are at least 2 m (4-D start+goal) from every f104 start/goal used before.

Arms (every arm of a group in ONE cluster array task; identical picks driven once):
  n2           deployed planner: argmin model risk over the 256-candidate proposal
  rule         argmin of a non-learned terrain+speed rule over the SAME 256 candidates
               (`gen_v1/hand_rule.json`: 22 hand features, pairwise logistic fit on the same f104 training routes)
  straight6    the 6 m/s straight route (no model)
  n2_fixed2    argmin model risk over 256 geometry-only candidates at 2 m/s
  rule_fixed2  argmin hand rule over the same fixed-2 candidates
  straight2    the 2 m/s straight route

Primary (P1): pooled over the five new arenas (1,000 groups), unsafe rate n2 vs rule, exact two-sided McNemar on
discordant groups. "The learned model adds value beyond a hand rule on unseen terrain" requires n2 better, p < 0.01.
Holm family for unsafe (6 tests): P1; new arenas n2 vs straight6; new arenas n2_fixed2 vs rule_fixed2; new arenas
n2_fixed2 vs straight2; f104 n2 vs rule; f104 n2 vs straight6.
Secondary, reported without correction: failure rate and tilt30 for the same pairs; median time to goal;
generalisation gap = n2 unsafe rate on new arenas minus on f104 (bootstrap CI, groups resampled within arena);
per-arena rates; arena-level sign test for P1 as a clustering check.
A missing arm (route failed validation) removes that group from comparisons involving the arm only; counts reported.

Data collection on the new arenas: 150 further start/goals per arena (all strata, disjoint from the test groups),
12 designed routes each (3 offsets x 4 speed profiles) = 9,000 episodes, same collector. Offline read-out:
within-group ranking AUC (unsafe) of the model, the hand rule and speed alone, per arena and pooled.
These episodes are future training data; they are not used to change anything in this plan.

## B. Five-goal missions (task 2)

A mission = start pose + 5 goals, driven as ONE continuous Chrono simulation. The vehicle is never reset. When the
chassis is within 2.5 m of the current goal, the next route is planned from the vehicle's actual pose at that moment
(simulation paused while planning) and handed to a fresh copy of the same path follower (steering-rate limiter state
carried over; PID integrators restart). Per leg, the single-episode stop rules apply unchanged: stall detector
(2 s window within 0.25 m at throttle > 0.3, 2 s confirmation + 8 s tail, not before 24 s of leg time), rollover
(> 60 deg), leaving the terrain, 120 s leg horizon. A failed leg ends the mission.

Mission geometry: legs 25-45 m; heading change at each goal <= 60 deg; all six points on slope < 7 deg and within
+-32 m; at least 3 of 5 leg chords cross a hill (> median ground + 2 m) or crater (< median ground - 0.7 m) zone.
100 missions on f104 (primary) and 20 on each new arena (secondary).

Arms (same job per mission): n2 (replan with the model), rule (replan with the hand rule), straight6 (straight
6 m/s route from the actual pose each leg).

Primary (M1): f104 mission success (all 5 goals reached), n2 vs straight6, exact McNemar. M2: n2 vs rule.
Secondary: goals reached per mission, missions with any slide, max tilt, total mission time; the same on the new
arenas pooled.

## Not changed after results
Arms, group counts, labels, thresholds, primary metrics and Holm families above. Anything added later is labelled
exploratory in the report.

## Amendment 1 (2026-09-15, before any mission was run)
The declared mission geometry (legs 25-45 m, heading change <= 60 deg, all points within +-32 m) is infeasible:
0 of 200,000 generation attempts on f104 and g203 produced a five-leg mission (a chain that long with turns that
gentle leaves the arena). Changed to legs 20-35 m and heading change <= 110 deg at each goal; everything else in
section B unchanged. No mission outcome existed when this was changed.

## Amendment 2 (2026-09-15, after a one-mission pilot, before the mission array)
In the pilot, legs that start with the vehicle facing well away from the next goal had no valid route: the frozen
Hermite route shape exceeds the curvature limit there. The route builder (`gen_planner.base_route`) now tries, in a
fixed order, the frozen shape, wider/tighter Hermite starts, and a tight-arc-then-straight route (radius 12/10/9 m),
using the first shape the validator accepts. Used identically by all three mission arms; single start/goal tests
are unaffected (their starts face the goal). The pilot mission (f104_mission_000 before regeneration) is excluded
from all analysis.
