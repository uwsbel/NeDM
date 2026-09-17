# gen_v1 report — five-goal missions and five never-seen arenas (2026-09-15)

Frozen throughout: the night-2 risk model (`night2_v1/final/N2_s*.pt`, no retraining), the night-2 route proposal,
the collector physics and driver. Plan written and hashed before any run: `PLAN.md` (+ two amendments, both before
the affected runs). Everything ran on the AMD cluster with all arms of a start/goal (or mission) on the same node.
Figures: `arenas.png`, `results.png`. Per-run data: `test/runs`, `missions_run/runs`, `data/runs` (trajectory,
outcome, reference only).

Arms. **model** = the deployed planner (lowest predicted risk of 256 candidates). **hand rule** = a non-learned
terrain+speed score (22 hand features, fit on the same f104 training routes) choosing from the same 256.
**straight line** = the 6 m/s straight route (2 m/s in the fixed-speed comparison).

## 1. New arenas (task 3)

Five arenas from the f104 terrain generator, the closest 5 of 40 seeds to f104 (g228, g203, g217, g216, g231: 5-6
hills, 5-6 craters, 29-31 deg slope cap). 200 hill/crater start/goals per arena, plus 200 fresh ones on f104.
6,639 runs, 0 collection failures.

Single goal, failed or slid backwards (unsafe):

| | model | hand rule | straight line |
|---|---|---|---|
| f104, speed free | 0.0% | 0.5% | 1.0% |
| new arenas, speed free | **1.3%** | 1.8% | 4.5% |
| f104, 2 m/s | 2.5% | 10.0% | 48.0% |
| new arenas, 2 m/s | **5.9%** | 13.6% | 59.8% |

Pre-registered unsafe family (exact McNemar, Holm over 6 tests):

| Test | Discordant groups | p | Holm |
|---|---|---|---|
| **P1 new arenas: model vs hand rule** | 10 vs 15 | 0.42 | 1.0 — **primary criterion not met** |
| new arenas: model vs straight 6 m/s | 9 vs 41 | < 1e-5 | < 1e-4 |
| new arenas 2 m/s: model vs hand rule | 25 vs 102 | < 1e-10 | < 1e-10 |
| new arenas 2 m/s: model vs straight 2 m/s | 12 vs 551 | < 1e-10 | < 1e-10 |
| f104: model vs hand rule | 0 vs 1 | 1.0 | 1.0 |
| f104: model vs straight 6 m/s | 0 vs 2 | 0.5 | 1.0 |

Secondary (uncorrected). New arenas, speed free: failures 0.6% / 1.4% / 3.0% (model vs rule 5 vs 13, p = 0.096;
vs straight 5 vs 29, p < 1e-4); leaned past 30 deg 4.4% / 9.8% / 17.8% (model vs rule 32 vs 86, p < 1e-5). Median
time to goal 12.3 s / 8.0 s / 7.8 s: the model buys its margin by driving slower. Per arena the model is safer than
the straight line on 5/5 arenas and safer than the hand rule at 2 m/s on 5/5; against the hand rule with speed free
it is safer on 2, worse on 1, tied on 2 (arena sign test p = 1.0). Generalisation gap in the model's unsafe rate,
new arenas minus f104: +1.3 points [0.7, 2.0].

Offline ranking on the 9,000 designed routes collected on the new arenas (12 per start/goal, unsafe label):

| | pooled | within start/goal | within start/goal, same speed |
|---|---|---|---|
| model, new arenas | 0.950 | 0.955 | 0.905 |
| model, f104 held-out test groups | 0.995 | 0.991 | 0.985 |
| hand rule, new arenas | 0.865 | 0.902 | 0.778 |
| speed alone, new arenas | 0.756 | 0.839 | 0.506 |

**Reading.** The model's terrain skill transfers to unseen arenas of the same kind: with speed held fixed it roughly
halves the hand rule's failures on every new arena, and it ranks routes far better than the rule at matched speed
(0.905 vs 0.778), though below its f104 level (0.985). With speed free, the choice that matters most is still "carry
momentum", and a hand rule that knows this is statistically as safe as the model (1.3% vs 1.8%) while being 4 s
faster — the model's remaining advantage is much less body tilt. Both clearly beat the plain straight line on the new
arenas, which they did not on f104.

## 2. Five goals in a row (task 2)

One continuous simulation per mission; at each goal the next route is planned from the truck's measured pose
(planning pauses the clock). 100 missions on f104, 20 on each new arena; legs 20-35 m, turns up to 110 deg, at least
3 of 5 legs crossing a hill or crater (mean path ~140 m). 600 mission runs.

| | model | hand rule | straight line |
|---|---|---|---|
| f104: all 5 goals reached | **99%** | 91% | 90% |
| f104: slid backwards at least once | 2% | 7% | 11% |
| f104: leaned past 30 deg | 10% | 18% | 28% |
| f104: median time, completed missions | 54 s | 36 s | 30 s |
| new arenas: all 5 goals reached | 94% | 89% | 94% |
| new arenas: slid at least once | 7% | 15% | 16% |
| new arenas: leaned past 30 deg | 9% | 21% | 42% |
| new arenas: median time | 52 s | 36 s | 31 s |

Pre-registered: **M1 f104 success, model vs straight line: 10 vs 1 missions, p = 0.012. M2 model vs hand rule:
8 vs 0, p = 0.008.** New arenas: model vs straight line 5 vs 5 (tie), vs hand rule 8 vs 3 (p = 0.23); fewer slides
than the straight line 14 vs 5 (p = 0.064). Mission failures: the straight line and the hand rule lost missions to
stalls, timeouts and 2 rollovers each on f104; the model lost one f104 mission (a stall).

**Reading.** Chaining goals is where the learned planner separates most clearly on its own arena: small per-leg
risks compound over five legs, and the model's pick is the only one that almost never compounds into a failure. On
the new arenas it completes missions as often as the straight line while sliding and leaning far less, at ~20 s more
per mission.

## 3. Data collected on the new arenas

9,000 designed routes (150 start/goals x 12 routes x 5 arenas) plus the 6,639 test runs = 15,639 labelled routes,
unsafe rate 16-29% per arena. Training tensors in the same format as the f104 set: `station_ds_gen_v1.npz`
(467 MB, with an `arena` column). Not used for anything tonight.

## Caveats

- The map input is each arena's heightmap (privileged, noise-free), not a captured sensor image. On f104 it picks the
  same route as the depth map in 32/40 pools (top-5 in 40/40).
- "New" means new instances of the same terrain generator, chosen for similarity to f104. It is not a test of
  different terrain types.
- Time is not in the planner's objective; the model's safety margin costs 4 s per single goal and ~20 s per mission.
- Missions: planning pauses the simulation; the path-follower's integrators restart at each goal; sharp-turn legs use
  a fallback route shape (amendment 2); 36 missions were re-run whole after a planner crash fix; 5 missions still end
  in "no route" legs and count as failures. Missions are not re-runs of each other across arms after leg 1 (arrival
  poses differ), so they compare policies, not routes.
- The primary single-goal test (P1) was null. The strong wins are at fixed speed and in chained missions on f104.
