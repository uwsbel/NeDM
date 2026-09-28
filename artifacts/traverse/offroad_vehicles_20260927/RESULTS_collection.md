# RESULTS collection (Q2): the Polaris drives the Gator's 15,235 soil routes; Gator planner checks (2026-09-28)

Q2 of `PLAN.md`: for a vehicle that passes the smoke test (`RESULTS_smoke.md`: the Polaris did, the re-geared M113
too but its collection was out of scope tonight), can it collect the same data as the Gator, i.e. the same 15,235
f104 soil task ids, tiers 0-12? Section 6 adds the two Gator planner checks driven tonight.

Words used below:
- **ids / routes** = the 15,235 soil start/goal/route rows the HMMWV (`collect_v1`, 09-16/17) and the Gator
  (`arena_gator_20260925`, 09-25/26) already drove on f104: 1,200 start/goal groups, 12-13 routes each, tiers 0-12.
  Here each row was driven by the Polaris (`polaris__<id>`, stock driveline = the primary arm of the smoke test).
- **designed routes** (9,168) = geometric routes at a set speed profile: constant 2, 4 or 6 m/s, or 2 -> 6 -> 2 m/s.
  **planner proposals** (6,067) = routes drawn at random from the planner's own route family (sideways bends, 4
  random speed knots between 0.5 and 6 m/s; `scripts/f104_n2_sampler.py`, route `meta.candidate = n2_wide`).
- **tiers** = batches of the collection order: each group's routes are numbered, and tier k holds route k of every
  group (about 1,200 rows per tier; tiers 0-6 = the first 7 routes of each group).
- **launch check** = the collector's test after the 0.8 s braked settle: body 0-1.2 m above the ground, roll and
  pitch within 25 deg, speed at most 1 m/s.
- **valid** = the drive completed and passes the collector's QA check `crm_qa.check` (finite states, launch check
  passed, no explosion, and no soil breakthrough at speed without a stall before it, which would mean a wheel
  punched through the rigid floor).
- **fail** = goal not reached. **goal reached safely** = goal reached, no roll-back on a climb and no belly-in-soil
  flag (lowest hull point more than 0.05 m under the undisturbed surface for more than 1 s).
- G4 = `/work1/dannegrut/harry/experiments/offroad_vehicles_20260927` (this study, cluster);
  G3 = `.../arena_gator_20260925` (Gator runs `G3/soil_v1/runs/gator__<id>`); HMMWV runs
  `.../crm_f104_20260916/collect_v1/runs/<id>`.

Sources: task file `tasks/soil_v5_polaris.json` (the final superset; the 15,235 Polaris rows are those with tier
0-12), the Polaris runs `G4/soil_v1/runs/polaris__<id>/`, the Gator and HMMWV runs above, and the route files named in
each row. Every per-route number was recomputed tonight by a read-only pass over those run folders (outcome.json,
initial_state_validation.json, vehicle_extra.npz, `crm_qa.check`) and route files; the LOG's read-outs are quoted
where they agree and flagged where they differ.

![identical-route comparison](figures/fig_vehicles.png)

## 0. Headline

- **Yes: the Polaris collected the same data as the Gator.** All 15,235 ids were driven (0 failed ids); 15,229
  (99.96 %) are valid; 0 crashes, 0 launch-check failures, belly flag 0.03 %. Every declared criterion is met with a
  wide margin.
- **It is by far the strongest vehicle on these routes.** Goal not reached on the identical 15,229 routes: **Polaris
  15.0 %, HMMWV 68.0 %, Gator 88.1 %**. 8,137 routes fail only for the HMMWV, 51 only for the Polaris.
- **Its failures are on slow planner proposals.** Designed routes: 1.6 % not reached. Planner proposals: 35.1 %,
  rising to 74 % when the commanded mean speed is 0.5-1.5 m/s and falling to 4 % at 3 m/s or more.
- **It cost less than the Gator.** 93.0 simulated hours against the Gator's 141.1 h on the same ids (the HMMWV:
  91.5 h), at the same wall time per simulated second; the whole 15,235-id collection ran from 01:30 to 08:23.
- **Datasets built:** stage 1 (tiers 0-6) 31,575 decision rows from 8,395 episodes; stage 2 (tiers 0-12) 57,228
  rows from 15,229 episodes (the Gator's: 30,827 and 55,826).
- **Gator planner checks.** Gradient refinement of the planner's pick adds less than a point for the Gator (68.0 vs
  67.4 % goal reached) and the HMMWV (96.2 vs 95.6 %). Driving the offline wider-search probe: wider route shapes
  rescue 53 of the 134 pairs where the recorded pick was already rated risky (the bigger search in the old shapes:
  21), but the estimated 800-pair rate is only 73.9 % (recorded 67.4 %): far from 90 % for the Gator.

## 1. What was driven

| item | value | source |
|---|---|---|
| rows | 15,235 (9,168 designed, 6,067 planner proposals), tiers 0-12; the Gator's rows with the vehicle changed | `tasks/soil_v5_polaris.json` |
| of which reused from the smoke | the 144 sample-A rows of `polaris` (driven by the same frozen dispatcher) | rows with `retiered_from` |
| order | tiers 0-6 first (8,399 ids), then tiers 7-12, enabled at 04:40 once the smoke was "better, robust" (amendment 9.1 item 4) | LOG 01:30, 04:40 |
| timeline | launched 01:30; tiers 0-2 complete at 04:30; tiers 0-6 at 05:47; all 15,235 at 08:23 | LOG |
| budget | 49.8 billed node-hours since 09-27 22:00 for the whole study at 08:23 (ov_* jobs 46.6), including the smoke, the evaluation drives and two trainings; PLAN 5 expected 30-50 for the collection alone | LOG 08:23 |

## 2. End states and validity against the declared criteria

| end state (all 15,235 drives) | count | share |
|---|---|---|
| goal reached | 12,952 | 85.0 % |
| prolonged blockage (stalled) | 1,365 | 9.0 % |
| soil breakthrough (a wheel dug through the layer) | 898 | 5.9 % |
| timeout (120 s) | 10 | 0.07 % |
| rollover | 8 | 0.05 % |
| arena exit | 2 | 0.01 % |

| criterion (PLAN 3.4, from K3 PLAN 7.7) | limit | Polaris | met |
|---|---|---|---|
| ids valid (complete and pass `crm_qa.check`) | >= 95 % | 15,229 / 15,235 = **99.96 %** (6 flagged: soil breakthrough at speed with no stall before it) | yes |
| crashed or non-finite states | < 1 % | 0 (0 failed ids, 0 non-finite, 0 explosion) | yes |
| launch-check failures | < 5 % | 0 | yes |
| belly-in-soil flag | <= 10 % | 4 of 15,229 = **0.03 %** (2 rollovers, 1 breakthrough, 1 goal reached) | yes |
| vehicle record | present | `polaris` in all 15,235 outcome files | yes |

- The 6 rejected ids are `f104_v2_group_0727_route_02`, `_0555_route_02`, `_0187_route_03`, `_0291_route_10`,
  `_1181_route_11`, `_0033_route_10` (tiers 0, 2, 4, 6, 9, 10). The dataset builder rejects the same 6
  (`G4/e4/soil_s2_build.log`: "rejected {'crm_qa:unstalled_break': 6}").
- For comparison on the same ids: the Gator's belly flag was 8.5 % (1,296 drives, recomputed; the Gator report
  `arena_gator_20260925/REPORT.md` 3.1 says 8.5 %), because its hull sits 0.14-0.17 m above the ground against the
  Polaris's about 0.31 m (scout S1 section 3.2).
- Matches the LOG 08:45 read-out exactly (goal 12,952, breakthrough 898, blockage 1,365, rollover 8, timeout 10,
  arena exit 2, 0 launch failures, belly 4, 15,229 valid).

## 3. Identical-route comparison: Polaris, HMMWV and Gator

The 15,229 ids valid for all three vehicles (the Gator and HMMWV runs are valid on all 15,235; the 6 Polaris-rejected
ids are left out). Including them changes no rate by more than 0.1 point (all 15,235: 15.0 / 68.1 / 88.2 %).

| routes | n | Polaris | HMMWV | Gator | only Polaris fails / only HMMWV fails | only Polaris / only Gator |
|---|---|---|---|---|---|---|
| **all** | 15,229 | **15.0 %** | **68.0 %** | **88.1 %** | 51 / 8,137 | 17 / 11,164 |
| designed | 9,162 | 1.6 % | 57.6 % | 85.5 % | 6 / 5,134 | 2 / 7,688 |
| planner proposals | 6,067 | 35.1 % | 83.9 % | 92.2 % | 45 / 3,003 | 15 / 3,476 |
| designed, constant 2 m/s | 2,302 | 5.0 % | 86.2 % | 91.3 % | 0 / 1,869 | 0 / 1,986 |
| designed, constant 4 m/s | 2,254 | 0.1 % | 56.3 % | 83.8 % | 0 / 1,266 | 0 / 1,887 |
| designed, constant 6 m/s | 2,322 | 0.7 % | 33.5 % | 82.6 % | 4 / 764 | 1 / 1,903 |
| designed, 2 -> 6 -> 2 m/s | 2,284 | 0.5 % | 54.5 % | 84.2 % | 2 / 1,235 | 1 / 1,912 |
| proposals, mean speed 0.5-1.5 m/s | 1,368 | 74.0 % | 93.3 % | 97.3 % | 22 / 287 | 5 / 324 |
| proposals, 1.5-2.0 m/s | 1,579 | 41.0 % | 88.3 % | 93.4 % | 11 / 758 | 3 / 829 |
| proposals, 2.0-2.5 m/s | 1,470 | 23.0 % | 84.9 % | 91.5 % | 8 / 918 | 4 / 1,011 |
| proposals, 2.5-3.0 m/s | 907 | 11.2 % | 76.6 % | 89.3 % | 3 / 596 | 2 / 710 |
| proposals, 3.0 m/s or more | 743 | 4.2 % | 63.8 % | 85.1 % | 1 / 444 | 1 / 602 |

The speed profile of each designed route was read from its route file (`meta.speed_profile_id`; it agrees with
route index mod 4 on all 9,168, and every constant profile's maximum speed equals its label). The proposals' mean
commanded speed is the mean of the route's speed knots without the two end points.

Intervals (bootstrap over the 1,200 start/goal groups, 2,000 resamples):

| | Polaris | HMMWV | Gator | Polaris - HMMWV | Polaris - Gator |
|---|---|---|---|---|---|
| all routes | [14.3, 15.6] | [66.7, 69.5] | [87.1, 89.2] | -53.1 [-54.4, -51.8] | -73.2 [-74.3, -72.2] |
| designed | [1.3, 1.9] | [55.8, 59.3] | [84.2, 86.9] | | |
| planner proposals | [33.9, 36.5] | [82.6, 85.2] | [91.2, 93.1] | | |

How each vehicle fails (end states on the 15,229 routes):

| vehicle | goal | blockage (stall) | soil breakthrough (dug through) | other |
|---|---|---|---|---|
| Polaris | 12,952 | 1,365 | 892 (898 with the 6 rejected ids) | rollover 8, timeout 10, arena exit 2 |
| HMMWV | 4,866 | 1,985 | 8,367 | timeout 9, rollover 2 |
| Gator | 1,805 | 13,225 | 198 | timeout 1 |

- **Designed routes: the Polaris almost never fails.** 146 failures in 9,162, of which 115 on constant 2 m/s
  routes; 113 of the 146 are soil breakthroughs.
- **Planner proposals: the slower the route, the more it fails** (74 % at 0.5-1.5 m/s, 4 % at 3 m/s or more).
  Failures there are 1,348 blockages and 779 breakthroughs. The HMMWV and the Gator show the same direction, much
  higher. A likely reading (not tested here): on soil, a slow climb gives the wheels time to dig in, while speed
  carries the vehicle over; the planner will have to learn to avoid crawling.
- **Stable across the collection.** Per tier the Polaris fails 12.4-16.1 % (tiers 0-6: 14.7 %, tiers 7-12:
  15.2 %). Per terrain type 13.6 % (long traverse) to 16.5 % (hill entry/exit). Test-split groups: 15.7 / 73.9 /
  90.9 %.
- **Different failure physics.** The HMMWV mostly digs through (8,367 breakthroughs; in the scout's slope smoke one
  wheel spins with open differentials); the Gator mostly stalls (13,225 blockages; rear-wheel drive, 14 kW); the
  Polaris does both, rarely (scout S5 sections 3.5 and 4.2 for the mechanisms).
- **Speed when all succeed.** On the 4,815 routes where both the Polaris and the HMMWV reach the goal, the Polaris
  takes a median 0.92 of the HMMWV's time.
- **These final numbers supersede the LOG 07:15 interim** (11,233 ids then): Polaris 14.9 -> 15.0 %, HMMWV 68.1 ->
  68.0 %, Gator 88.1 % (same); designed 1.6 / 57.9 / 85.5 -> 1.6 / 57.6 / 85.5 %; proposals 35.3 / 83.7 / 92.1 ->
  35.1 / 83.9 / 92.2 %; only-Polaris / only-HMMWV 34 / 6,011 -> 51 / 8,137 (more ids).

## 4. Cost

| | Polaris | Gator (stored) | HMMWV (stored) |
|---|---|---|---|
| simulated hours on the 15,235 ids | **93.0 h** | 141.1 h | 91.5 h |
| mean simulated seconds per route (15,229 valid for all) | 22.0 | 33.4 | 21.6 |
| episode wall hours (sum of `wall_s`) | 279.4 h | 434.4 h | - |
| wall s per simulated s (whole collection, mixed nodes) | 3.00 | 3.08 | 2.87 |

- The simulated hours match LOG 08:45 (93.0 / 141.1 / 91.5 h) and the HMMWV's own QA record
  (`crm_f104_20260916/collect_v1/qa.json`: 91.51 h).
- Same cost per simulated second as the Gator (the smoke test measured 0.99x in the same launch;
  `RESULTS_smoke.md` section 6). The saving comes from shorter drives: the Polaris reaches the goal where the Gator
  stalls and runs into the blockage stop at 34 s (scout S5 section 1.1). The per-collection wall-per-simulated-second figures come from
  different node mixes and dates, so they are only indicative.

## 5. Datasets built from the Polaris drives

| | stage 1 (tiers 0-6) | stage 2 (all tiers 0-12) |
|---|---|---|
| episodes selected | **8,395** of 8,399 (4 rejected: breakthrough without a stall) | **15,229** of 15,235 (6 rejected, the same kind) |
| decision rows | **31,575** (train 7,619 at the standing start + 21,082 re-anchored mid-drive; val 392 + 1,058; test 384 + 1,040) | **57,228** (train 13,817 + 38,204; val 712 + 1,915; test 700 + 1,880) |
| file | `G4/e4/soil_s1/f104_polaris/ci_f104_polaris_crm.npz` (0.58 GB) | `G4/e4/soil_s2/f104_polaris/ci_f104_polaris_crm.npz` (1.04 GB) |
| training subset | `e4/subsets/polaris_s1_f104_soil.npz`: 1,089 training groups, 28,701 fit rows (deploy), 22,809 (hold-out lane), val 1,450 / test 1,424 | `e4/subsets/polaris_full_f104_soil.npz`: 1,089 groups, 52,021 fit rows (deploy), 41,338 (hold-out lane), val 2,627 / test 2,580 |
| Gator equivalent | 30,827 rows | 55,826 rows |
| model | job 441823 (9.5 min), `e5/deploy/polaris_s1_soil` | job 441919 (16.8 min), `e5/deploy/polaris_full_soil` |

Sources: `G4/e4/soil_s1_build.log`, `G4/e4/soil_s2_build.log`, `G4/e4/subsets/*.manifest.json`, LOG 05:55, 06:06, 08:45,
08:48; Gator rows from `arena_gator_20260925/REPORT.md` 3.1. Every vehicle block in both files is `polaris`; the map
check passed. The Polaris data have slightly more rows than the Gator's at the same ids. The planner results from
these models are in `RESULTS_planner.md`.

## 6. Gator planner checks run tonight

Both on the 800-pair f104 soil suite (200 tuning + 600 fresh pairs), standing start, the Gator-trained model of the
Gator study (`G_full`, 5 networks trained on all 15,235 Gator soil drives) and the HMMWV-trained one (`H_full`).
"Recorded" = the stored drives of the Gator study's planner, whose search is a cross-entropy method (CEM) with
4 rounds of 64 candidate routes, each round re-centred on the best 15 % (`arena_gator_20260925/e6/index/soil_eval_bfull.json`,
arms `Gfull_free_gator` and `Hfull_free`). All numbers below were recomputed from the index files.

### 6.1 Gradient refinement of the pick (1,600 new drives)

The planner's CEM pick is refined by gradient steps on the model's predicted risk (`ci_grad`, standing start). Picks
locked before any drive (`e6/picks/LOCK_gradref_v1.sha256`, cf0396c4); drives in
`e6/index/soil_eval_gradref_v1.json` (1,600 driven, 0 missing, all launch-checked and QA-valid).

| vehicle, model | goal reached: refined / recorded CEM | refined better / worse (pairs) | difference in failure, points [pair bootstrap] [9-cluster bootstrap] | one-sided McNemar p | goal reached safely (refined) | fresh 600 only |
|---|---|---|---|---|---|---|
| Gator, G_full | **68.0 %** (544) / 67.4 % (539) | 42 / 37 | -0.6 [-2.9, +1.5] [-1.8, +0.7] | 0.33 | 67.9 % (543; 10 belly flags) | 68.2 / 66.7 % |
| HMMWV, H_full | **96.2 %** (770) / 95.6 % (765) | 21 / 16 | -0.6 [-2.1, +0.9] [-2.8, +1.1] | 0.26 | 96.2 % (no belly record) | 95.7 / 95.5 % |

- **Gradient refinement adds less than a point from a standing start.** Recomputed 90 % cluster intervals: Gator
  [-1.6, +0.5], HMMWV [-2.3, +0.8]. Under the declared "no meaningful difference" rule (90 % interval within
  +-2 points; `e6/analysis/spec_ov_v1.json`) this reads "no meaningful difference" for the Gator and "inconclusive"
  for the HMMWV (lower end just past -2). The formal read-out of the declared family belongs to `RESULTS_planner.md`.
- The refinement changed 734 of the 800 Gator picks and 643 of the HMMWV's. **The unchanged picks re-drove to the
  identical end state on every pair** (Gator 66/66, HMMWV 157/157), on other nodes and another day: a clean
  reproducibility check.
- Refined picks are slightly faster (median time ratio on pairs both reach: Gator 0.95, HMMWV 0.98).
- Matches the LOG (04:33: 68.0 %, 37 vs 42 discordant, 96.2 %; 08:25: Gator safely 67.9 %).

### 6.2 Driving the offline wider-search probe (722 new Gator drives)

The offline probe (`artifacts/traverse/search_probe_20260927/RESULTS.md`, 09-27) re-planned 361 of the 800 pairs
with the same Gator model and found, on paper, safe routes for many pairs the recorded planner had failed. Tonight
two of its pick sets were driven by the Gator (analysis spec `e6/analysis/spec_probe_v1.json`, frozen 04:58 before
any drive; descriptive, outside the declared test family):
- **bigger search** (probe arm A2): CEM 16 rounds x 512 routes (32 times the routes scored), the original route
  shapes (3 sideways bend terms, at most 10 m sideways);
- **wider shapes** (probe arm A4): the same bigger search with 5 bend terms and up to 20 m sideways.

The 361 pairs, by what the recorded planner (CEM 4 x 64) did (`search_probe_20260927/groups/*.txt`):
- **risky failures** (134): the recorded pick failed and the model had rated it over 50 % likely to fail;
- **other failures** (127): the recorded pick failed although rated 50 % or less;
- **controls** (100): recorded successes, the 100 lowest md5 of the 539.

![wider-search probe](figures/fig_probe.png)

| set | recorded planner | bigger search (16 x 512) | wider shapes (5 bends, 20 m) |
|---|---|---|---|
| risky failures (134) | 0 (all failed) | 21 = **15.7 %** [10.5, 22.8] | 53 = **39.6 %** [31.7, 48.0] |
| other failures (127) | 0 (all failed) | 42 = 33.1 % [25.5, 41.6] | 42 = 33.1 % [25.5, 41.6] |
| controls (100) | 100 (all reached) | 93 (7 broken) [86.3, 96.6] | 92 (8 broken) [85.0, 95.9] |

Goal reached safely, 95 % Wilson intervals; `e6/index/soil_eval_probe_v1.json` (722 driven, 0 missing, all
launch-checked and QA-valid; belly flags: wider 7, bigger 11). In these drives every goal-reaching drive was also
safe, so the goal-only counts are identical.

- **Wider shapes beat the bigger search on the risky failures:** 35 pairs rescued only by the wider shapes, 3 only
  by the bigger search (exact two-sided McNemar p 6.7e-8); over all 261 failures 48 vs 16 (p 7.7e-5). On the
  controls they are equal (3 vs 4).
- **Swinging wide is what helps.** Wider-shape picks that swing more than 10 m sideways (outside the original shape
  family and every Gator training route): 40 of 58 reached the goal on the risky failures; picks within 10 m: 13 of
  76. On the other failures: 15 of 18 vs 27 of 109. (Sideways swing from the probe's pick files,
  `A4_wide5m20_cem16x512/picks/*.json`, `max_lateral_m`, threshold 10.05 m as in the probe.)
- **The model's "safe" is still over-optimistic, most where it failed before without warning.** Of the 71 risky
  failures where the wider search found a route rated below 5 % (the offline probe's key count), 51 were reached
  safely when driven (46 of the 58 rated below 1 %). On the other failures, 108 wider picks were rated below 1 %
  and only 41 reached the goal.
- **Estimated 800-pair goal rate** if the new search replaced the recorded one (spec formula:
  (539 x control success share + rescued pairs) / 800): **wider shapes 73.9 %** = (539 x 0.92 + 53 + 42) / 800,
  bigger search 70.5 % = (539 x 0.93 + 21 + 42) / 800, recorded 67.4 % (539/800, driven). Caveats: only 100 of the
  539 recorded successes were re-driven, and the recorded 539 is goal reached, not goal reached safely (536 are safe
  by the unsafe label; the stored drives have no belly record).
- **Bottom line for the Gator:** wider route shapes help (+6.5 points estimated), bigger search alone less (+3.1);
  gradient refinement about +0.6. None comes near the 90 % bar. Half of the recorded failures (127 of 261) had picks
  the model rated safe, and there a better search cannot help: those need a better model or a stronger vehicle.
- All counts match LOG 07:05 (53/134, 42/127, 92/100; 21/134, 42/127, 93/100; 35 vs 3, p 6.7e-8; 48 vs 16,
  p 7.7e-5; 40/58 and 13/76; 73.9 / 70.5 / 67.4 %).

## 7. Caveats

- **Vehicle model.** The collection used the primary Polaris: Chrono's stock driveline with its reduction defect
  (the wheels get about 16 times the engine's power; below about 6 m/s it acts like a one-gear 4WD vehicle with
  plausible torque). The smoke test showed the power-corrected driveline does as well (13.9 vs 16.7 %) and open
  differentials somewhat worse (25.7 %), both far better than the Gator; the collection itself was not repeated with
  another driveline. The Polaris drives with Chrono's limited-slip torque split (spinning wheels do not starve the
  others) where the HMMWV has open differentials: part of the Polaris-HMMWV gap may be that modelling choice.
- **Soil stand-ins.** Soil wheels are 0.25 m cylinders (calibrated to the HMMWV's settled sinkage); at 0.08 m
  spacing the soil sees narrow tyres as wider, which favours the Polaris most. The body is not coupled to the soil
  (all vehicles). The soil is a thin 0.24 m layer over a rigid floor, which favours light vehicles.
- **The comparison vehicles are stored runs.** The Gator (09-25/26) and HMMWV (09-16/17) runs come from other
  launches, nodes and dates. The smoke test's Gator re-drive reproduced 144/144 stored end states and tonight's
  unchanged gradient picks 223/223, so cross-date drift is not a concern at the outcome level.
- **Ceiling.** The Polaris reaches the goal on 98.4 % of the designed routes, so the planner question (Q3) for the
  Polaris is close to its ceiling: the straight 6 m/s route alone reaches 99.0 % on sample B (`RESULTS_smoke.md`).
- **Probe estimate.** The 800-pair probe estimates rest on 100 re-driven controls. The control share's 95 % Wilson
  interval alone moves the wider-shape estimate between 69.1 and 76.5 % and the bigger-search one between 66.0 and
  72.9 %.
