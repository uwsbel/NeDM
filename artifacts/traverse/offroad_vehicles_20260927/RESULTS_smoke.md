# RESULTS smoke (Q1): do Chrono's Polaris and M113 get through f104 soil better than the Gator? (2026-09-28)

Q1 of `PLAN.md`: on f104 soil, with the same routes, the same soil and the same route follower, does Chrono's
Polaris, and does Chrono's M113 tracked vehicle, reach the goal more often than the Chrono Gator? The rule for
"better" was written down before any drive (PLAN 2.4, amendment 9.1) and is applied by the frozen script
`scripts/ov_smoke_analyze.py` (sha256 `d7b9141e...`, LOG 01:05).

Words used below:
- **soil** = Chrono's particle soil (CRM), production settings: 0.08 m particle spacing, 0.24 m layer over a rigid
  floor that follows the terrain, 1 ms step (0.5 ms for the M113 rows).
- **sample A** = the 144 soil routes of the Gator pilot: 24 start/goal groups (4 per terrain type) x 6 routes each
  (`scratch/S3/sample_A.json`). 86 are **designed routes** (geometric routes at a set speed profile: constant 2, 4 or
  6 m/s, or 2 -> 6 -> 2 m/s) and 58 are **planner proposals** (routes drawn at random from the planner's own route
  family: sideways bends and 4 random speed knots between 0.5 and 6 m/s; `scripts/f104_n2_sampler.py`).
- **sample B** = 96 of the 800 f104 evaluation pairs (16 per terrain type), each with the straight route at 6 m/s
  and the pick of the HMMWV-trained planner (`scratch/S3/sample_B.json`).
- **Gator re-drive** = the existing Gator driven again on sample A through the new dispatcher, in the same launch
  as the Polaris rows (arm name `gatorctl`); `gatorh` = the Gator at the M113's 0.5 ms step, in the M113 launch.
- **fail** = goal not reached (any stop rule: blockage = stalled with throttle for 2 s after 24 s, soil breakthrough
  = a wheel or track dug through the whole layer, rollover, timeout, arena exit).
- **launch check** = the collector's test after the 0.8 s braked settle (body 0-1.2 m above the ground, roll and
  pitch within 25 deg, speed at most 1 m/s). **valid** = the drive completed and passes the collector's QA check
  `crm_qa.check` (finite states, launch check passed, no explosion, no breakthrough at speed without a stall first).
- **belly flag** = the lowest hull point more than 0.05 m under the undisturbed soil surface for more than 1 s.

Sources: `analysis/smoke_polaris_v1_final.txt` / `.json` (all 916 Polaris-file rows, 04:30),
`analysis/smoke_m113_v1_final.txt` / `.json` (all 432 M113-file rows: 431 complete, 1 crashed; 06:51), task files
`tasks/smoke_v2_polaris.json` and `tasks/analysis_m113_combined.json`. Jobs (LOG 01:05-04:30): Polaris file 441596,
441605, 441609 and two more mi3501x nodes; M113 file 441599 and 441610, relaunched at 04:30 after the 4 h cap. Per-route breakdowns ("from the runs") were
recomputed read-only from the cluster run folders `G4/soil_v1/runs/<arm>__<id>/` (outcome.json, launch record,
`crm_qa.check`, vehicle_extra.npz) and the route files named in the task rows; every recomputed rate matches the
analysis files. G4 = `/work1/dannegrut/harry/experiments/offroad_vehicles_20260927`.

![goal not reached on sample A per arm](figures/fig_smoke.png)

## 0. Headline

| vehicle | verdict (frozen rule) | goal not reached on sample A | Gator re-drive | difference [95 % group interval] | one-sided McNemar p |
|---|---|---|---|---|---|
| **Polaris** (primary: stock driveline) | **better**, robust to the driveline | **16.7 %** (24/144) | 93.8 % | **-77.1** [-83.3, -70.8] | 3.9e-34 |
| M113, stock gearing | not better | 89.6 % (129/144) | 93.8 % | -4.2 [-11.1, +2.8] | 0.090 |
| **M113, re-geared** (gear ratios / 4) | **better** (the better M113 arm) | **48.3 %** (69/143) | 93.7 % | **-45.5** [-58.0, -33.3] | 2.7e-20 |

Source: `analysis/smoke_polaris_v1_final.txt`, `analysis/smoke_m113_v1_final.txt`. The frozen analysis pairs every
arm with the Gator re-drive of the Polaris launch (`gatorctl`). The M113 ran in its own jobs (amendment 9.1 item 1);
its own-launch Gator (`gatorh`, at the M113's 0.5 ms step) also fails 93.8 % and gives the same verdicts (section 2).

- **The Polaris is far better than the Gator.** 111 routes fail only for the Gator, 0 only for the Polaris. It
  reached the goal in all 24 groups (Gator: 6 of 24). It is also far better than the stored HMMWV runs on these
  routes (69.4 % not reached, `scratch/S3/sample_A.json`).
- **The result does not depend on the Polaris's broken driveline model.** The power-corrected driveline fails
  13.9 %, open differentials 25.7 %, larger (0.33 m) soil wheels 18.8 %. All three are far below the Gator.
- **Every Polaris failure is on a planner proposal** (24/58); it reached the goal on all 86 designed routes.
- **Sample B agrees and hits a ceiling.** The straight 6 m/s route fails 1.0 % for the Polaris (Gator stored
  83.3 %); the HMMWV-trained planner's picks fail 0 % (96/96 reached). By the declared ceiling rule, the straight
  route alone nearly meets the 90 % bar for the Polaris.
- **The stock M113 is no better than the Gator.** It lacks drive torque: the stock model's sprocket push is at most
  0.35 of its weight (scout S5 section 3.3). The re-geared M113 is clearly better than the Gator, but clearly worse
  than the Polaris (48.3 vs 16.8 % on the same 143 routes).
- **The M113 costs 4.1x the Gator per simulated second on the same GPU type** (6.2x over the mixed nodes it ran
  on). A full M113 collection would take about 2,600 GPU-hours (LOG 06:00), so it was not run.

## 1. The decision rule (PLAN 2.4, frozen before any drive)

Primary endpoint: goal not reached on sample A, paired route by route with the same-launch Gator re-drive. A vehicle
is **better** only if all five hold:

| # | criterion | Polaris (`polaris`) | M113 stock (`m113`) | M113 re-geared (`m113_g4`) |
|---|---|---|---|---|
| 1 | validity gates pass (section 3) | yes | yes | yes |
| 2 | failure rate at least 10 points lower (point estimate) | yes (-77.1) | no (-4.2) | yes (-45.5) |
| 3 | exact one-sided McNemar p at or below its Holm level (alpha 0.05) | yes (3.9e-34 vs 0.05) | no (0.090 vs 0.05) | yes (2.7e-20 vs 0.025) |
| 4 | 95 % bootstrap interval over the 24 groups lies below 0 | yes [-83.3, -70.8] | no [-11.1, +2.8] | yes [-58.0, -33.3] |
| 5 | failure rate below 80.6 % (the Gator with 0.08 m larger soil wheels, stored) | yes (16.7 %) | no (89.6 %) | yes (48.3 %) |
| | **verdict** | **better** | not better | **better** |

Source: `decision` blocks of `analysis/smoke_polaris_v1_final.json` and `analysis/smoke_m113_v1_final.json`.

- **Holm levels.** The tool ran the Polaris as a family of one (level 0.05) and the M113 as a family of two arms
  (Holm levels 0.025 / 0.05). PLAN 2.4 speaks of "Holm over 3 comparisons"; both passing arms also pass the strictest
  step of a three-way family (0.05 / 3 = 0.017; field `passes_at_family_of_3_bonferroni_first_step: true`), so the
  verdicts do not depend on how the family is read.
- **Consistency with sample B (PLAN 2.4).** Polaris: straight 6 m/s fails 1.0 %, below the Gator's stored 83.3 %:
  consistent. **M113: not tested.** The M113 task file had no sample-B rows (`consistency_B: null` in
  `smoke_m113_v1_final.json`). The M113 verdict therefore rests on sample A only; it has no consequence tonight
  because no M113 collection was planned (PLAN 1.3).
- **Driveline robustness (amendment 9.1 item 4).** The Polaris result counts as robust only if the power-corrected
  arm also meets criteria 2 and 5 (it does: -79.9 points, 13.9 %), and as driveline-dependent if the primary and a
  driveline arm differ by more than 10 points (point estimates: power-corrected -2.8, open differentials +9.0).
  **Robust, not driveline-dependent.** Caveat: the open-differential difference has a 95 % interval of
  [+4.2, +13.9], so a gap above 10 points is not excluded.
- **Ceiling rule.** The Polaris fails fewer than 12 % of sample B's straight routes (1.0 %), so the report must say
  that the straight route already nearly meets the bar. Per the plan the collection and planner still ran.

## 2. Results per arm on sample A

| arm | what it is | valid drives | goal not reached | vs `gatorctl`: only Gator fails / only arm fails | groups with a goal | end states |
|---|---|---|---|---|---|---|
| `gatorctl` | Gator re-drive, 1 ms (Polaris launch) | 144/144 | 93.8 % | - | 6/24 | goal 9, blockage 134, breakthrough 1 |
| `gatorh` | Gator at 0.5 ms (M113 launch) | 144/144 | 93.8 % | 3 / 3 (vs `gatorctl`) | 6/24 | goal 9, blockage 134, breakthrough 1 |
| `polaris` | Polaris, stock driveline (primary) | 144/144 | **16.7 %** | 111 / 0 | 24/24 | goal 120, breakthrough 12, blockage 12 |
| `polaris_pc` | power-corrected driveline | 144/144 | 13.9 % | 115 / 0 | 24/24 | goal 124, blockage 11, breakthrough 9 |
| `polaris_4wd` | Chrono's open-differential 4WD driveline | 144/144 | 25.7 % | 98 / 0 | 24/24 | goal 107, blockage 20, breakthrough 17 |
| `polaris_w08` | soil wheels 0.33 m instead of 0.25 m | 144/144 | 18.8 % | 108 / 0 | 24/24 | goal 117, blockage 25, rollover 2 |
| `m113` | M113, stock gearing | 144/144 | 89.6 % | 10 / 4 | 7/24 | goal 15, blockage 128, breakthrough 1 |
| `m113_g4` | M113, gear ratios / 4 | 143/144 (1 crash) | 48.3 % | 65 / 0 | 19/24 | goal 74, blockage 55, breakthrough 14 |

Source: `arms.*.describe` and `vs_gatorctl` in the two analysis JSON files.

Polaris sensitivity arms against the primary (same 144 routes; `sensitivity` block of
`smoke_polaris_v1_final.json`):

| arm vs `polaris` | difference | only `polaris` fails / only the arm fails | 95 % group interval | two-sided p |
|---|---|---|---|---|
| power-corrected | -2.8 | 4 / 0 | [-6.2, +0.0] | 0.13 |
| open differentials | +9.0 | 2 / 15 | [+4.2, +13.9] | 0.0023 |
| 0.33 m soil wheels | +2.1 | 4 / 7 | [-3.5, +7.6] | 0.55 |

- Open differentials are measurably worse (15 routes fail only with them), as scout S5 predicted: with open
  differentials one spinning wheel takes the torque, the HMMWV's failure pattern. Still 68 points better than the
  Gator.
- M113 against the Gator of its own launch (`gatorh`, 0.5 ms), recomputed from the runs: stock -4.2 (12 / 6
  discordant, one-sided p 0.12); re-geared -45.5 (65 / 0, p 2.7e-20). Same conclusions as against `gatorctl`.
- Re-geared vs stock M113: 60 routes fail only for the stock M113, 1 only for the re-geared one (recomputed from the
  runs). The stock M113 fails for lack of drive torque, not grip (NOTES_M2 gate c: stalled at 19-20 deg pitch, engine
  pinned at the torque converter's stall speed).

## 3. Validity gates (PLAN 2.3) and reference checks

| arm | ids complete and pass `crm_qa.check` (>= 95 %) | crashed / non-finite (< 1 %) | launch-check failures (< 5 %) | belly flag (<= 10 %) | vehicle record | pass |
|---|---|---|---|---|---|---|
| `gatorctl` | 144/144 | 0 | 0 | 4.2 % | gator 144 | yes |
| `gatorh` | 144/144 | 0 | 0 | 3.5 % | gator 144 | yes |
| `polaris` | 144/144 | 0 | 0 | 0.0 % | polaris 144 | yes |
| `polaris_pc` | 144/144 | 0 | 0 | 0.0 % | polaris_pc 144 | yes |
| `polaris_4wd` | 144/144 | 0 | 0 | 0.0 % | polaris_4wd 144 | yes |
| `polaris_w08` | 144/144 | 0 | 0 | 0.0 % | polaris_w08 144 | yes |
| `m113` | 144/144 | 0 | 0 | 0.0 % | m113 144 | yes |
| `m113_g4` | 143/144 (99.3 %) | 1 (0.7 %) | 0 | 0.0 % | m113_g4 143 | yes |

Source: `arms.*.gates` in the two analysis JSON files.

- The one M113 crash: `m113_g4__f104_v2_group_0427_route_11`, "track quarter ('rear', 1) has no shoe" in the M113
  module's per-quarter lowest-shoe lookup (LOG 04:30; `G4/soil_v1/failed/...json`: 2 attempts, both failed). It is a
  bookkeeping gap in the new module, not a physics blow-up. The drive is excluded from both sides of the pairing.

Reference checks (`reference` block of both analysis files):

| check | result |
|---|---|
| Gator re-drive vs the stored Gator runs (144 sample-A routes, other nodes and dates) | same end state 144/144, identical arrays 144/144; failure 93.8 % both |
| Gator reuse rule (amendment 9.1 item 5: keep >= 8 of the 9 stored goals and >= 95 % of end states) | 9/9 goals, 100 % end states: stored Gator drives may be reused |
| 3 HMMWV + 3 Gator bit-identity rows through the new dispatcher vs stored runs | same end state 6/6; identical arrays 4/6 (two HMMWV rows differ in trajectory length, 457 vs 458 and 361 vs 319 frames: the known drift across nodes; same outcome) |
| pooled end-state agreement (gate: >= 95 %) | 100 % of 150 |
| Gator at 0.5 ms vs 1 ms (`gatorh` vs `gatorctl`, recomputed from the runs) | 93.8 % both; same end state 138/144; 3 / 3 discordant. The M113's smaller step does not by itself change the Gator's result |

## 4. Where the failures are (from the runs)

| arm | designed routes (86) | planner proposals (58) | failure types on designed / on proposals |
|---|---|---|---|
| Gator re-drive | 81 (94.2 %) | 54 (93.1 %) | blockage 80, breakthrough 1 / blockage 54 |
| Gator stored runs | 81 | 54 | (same end states as the re-drive) |
| HMMWV stored runs | 50 (58.1 %) | 50 (86.2 %) | (`scratch/S3/sample_A.json`) |
| Polaris, stock driveline | **0** | **24 (41.4 %)** | - / breakthrough 12, blockage 12 |
| Polaris, power-corrected | 0 | 20 (34.5 %) | - / blockage 11, breakthrough 9 |
| Polaris, open differentials | 5 (5.8 %) | 32 (55.2 %) | breakthrough 4, blockage 1 / blockage 19, breakthrough 13 |
| Polaris, 0.33 m soil wheels | 3 (3.5 %) | 24 (41.4 %) | rollover 2, blockage 1 / blockage 24 |
| M113, stock | 76 (88.4 %) | 53 (91.4 %) | blockage 75, breakthrough 1 / blockage 53 |
| M113, re-geared | 40 of 85 (47.1 %) | 29 (50.0 %) | blockage 30, breakthrough 10 / blockage 25, breakthrough 4 |

Recomputed from `G4/soil_v1/runs/<arm>__<id>/outcome.json`; route kind from the task rows, route family
(`rgbd_geometric_family` = designed, `n2_wide` = planner proposal) from each route file's `meta.candidate`.

- **Polaris: planner proposals only.** Its 24 failures sit in 15 of the 24 groups, spread over all six terrain
  types (hill cross-slope 7, crater entry/exit 5, hill entry/exit 4, roughness transfer 4, crater cross-slope 3, long
  traverse 1). The proposals are much slower than the designed routes (median commanded mean speed 1.9 vs 3.9 m/s on
  sample A). The full collection shows the same pattern at scale: slow proposals are where the Polaris fails
  (`RESULTS_collection.md` section 3).
- **The designed-route failures of the sensitivity arms are at 2 m/s** for the open differentials (5 of 18 constant
  2 m/s routes). The 0.33 m wheels fail 1 constant 2 m/s route (blockage) and 2 constant 6 m/s routes (two rollovers, both in group
  `f104_v2_group_0259`, a crater entry/exit).
- **M113: no pattern by route kind.** Both M113 arms fail designed routes and proposals at about the same rate.
- **Soil breakthrough stop on the M113.** Amendment 9.1 item 3 expected this stop to be "effectively inactive"
  (pads rest 0.09-0.16 m above the surface). In the drives it fired 14 times for the re-geared M113 and once for the
  stock M113 (largest pad sinkage below the undisturbed surface 0.49 m and 0.36 m; `crm.max_wheel_sinkage_below_bmp_m`
  in the outcome files). All 15 passed the QA check for a stall before the breakthrough, i.e. the M113 had stopped
  and dug in, as the wheeled vehicles do.

## 5. Sample B (Polaris only)

| route on the 96 pairs | Polaris goal not reached | Gator, stored | HMMWV, stored |
|---|---|---|---|
| straight route at 6 m/s | **1.0 %** (1 blockage: `f104_pair_group_0579`) | 83.3 % | 34.4 % |
| HMMWV-trained planner's pick (H_full) | **0.0 %** (96/96 reached) | 51.0 % | 3.1 % |
| Gator-trained planner's pick (G_full), for reference | not driven | 25.0 % | - |

Sources: `sample_B` block of `smoke_polaris_v1_final.json`; stored rates `scratch/S3/sample_B.json`. 190 drives
cover the 192 slots, because on 2 pairs the H_full pick is the straight route itself (`tasks/smoke_v2_polaris.json`,
rows with both arm names).

## 6. Cost per simulated second

| arm | mean simulated s per route | loop wall s per simulated s, all nodes | x Gator re-drive | same, MI350X nodes only | x Gator (MI350X) | simulated hours |
|---|---|---|---|---|---|---|
| Gator re-drive | 35.3 | 2.74 | 1.00 | 1.94 | 1.00 | 1.41 |
| Polaris, stock | 22.9 | 2.71 | 0.99 | 1.96 | 1.01 | 0.92 |
| Polaris, power-corrected | 23.4 | 2.77 | 1.01 | 1.98 | 1.02 | 0.94 |
| Polaris, open differentials | 24.2 | 2.81 | 1.03 | 1.96 | 1.01 | 0.97 |
| Polaris, 0.33 m wheels | 25.2 | 2.77 | 1.01 | 1.96 | 1.01 | 1.01 |
| Gator at 0.5 ms | 35.1 | 5.41 | 1.98 | 3.92 | 2.02 | 1.40 |
| M113, stock | 37.8 | 16.88 | 6.16 | 7.99 | 4.11 | 1.51 |
| M113, re-geared | 36.7 | 16.92 | 6.18 | 7.99 | 4.11 | 1.46 |

Source: `arms.*.cost` in the analysis JSON files. "Loop wall" excludes the soil build; with the build it adds about
1-4 % (`wall_per_sim_with_setup`).

- **The Polaris costs the same as the Gator per simulated second**, and needs fewer simulated seconds per route
  because it reaches the goal instead of stalling into the 34 s blockage stop. This held in the full collection
  (93.0 vs 141.1 simulated hours on the same ids; `RESULTS_collection.md`).
- **The M113 is 4.1x the Gator on the same GPU type.** A factor of about 2 is its 0.5 ms step (the Gator alone at
  0.5 ms costs 2.0x); the other factor of about 2 is the tracked vehicle itself (155 bodies, 127 of them
  soil-coupled track shoes; S2 section 1.2). The 6.2x over all nodes mixes GPU types: the
  M113 file ran on mi2104x (MI210) and mi3501x, the Gator re-drive on a different mix. Both are inside the plan's
  8x quick-look limit (PLAN 5).
- **Full M113 collection: not tonight.** About 2,600 GPU-hours (LOG 06:00; it is the Gator's 141 simulated hours
  scaled by the M113's longer drives, 37.8 / 35.3 s, and its 16.9 wall s per simulated second), far over the
  study's budget (soft cap 150 billed node-hours for everything, PLAN 5). A reduced M113 collection is a question
  for the user.

## 7. Stage 0: flat soil (before the routes)

From module checks M1 (job 441578) and M2 (jobs 441588/441589); amendment 9.1 item 2 made only launch failure,
non-finite states, not reaching 2 m/s, and (M113) not holding braked on 10 deg tilted soil exclusion reasons.

| vehicle | launch height on flat soil | mean speed at 2 / 4 / 6 m/s asked (M113: climb test) | braked hold on tilted soil (mean speed) | admitted |
|---|---|---|---|---|
| Polaris (all four arms) | 0.370 m (0.33 m wheels 0.399 m) | 1.89 / 3.88 / 5.90 (stock) | - | yes |
| HMMWV | 0.568 m | 1.89 / 3.85 / 5.94 | - | reference |
| Gator | 0.303 m | 1.87 / 3.23 / 4.01 (cannot hold 4 or 6 m/s on soil) | 10 deg -0.018 m/s, 15 deg -0.036 m/s | reference |
| M113 (shafts brake, flat pads) | not reported (builds at 11,343 kg; launch check passed on f104) | climbs 10 deg from rest to 1.6 m/s in 3 s (re-geared 2.5 m/s) | 10 deg -0.009 m/s, 15 deg -0.028 m/s | yes |

Sources: `NOTES_M1.md` section 4, `NOTES_M2.md` gates a-d.

- Every vehicle, the HMMWV included, misses the 0.5 m cross-track mark on a 10 m radius half circle (HMMWV 0.77 m,
  Polaris 0.79-0.81 m, Gator 0.95 m): the frozen follower's 5 m look-ahead cuts the curve. Reported, not gating.
- The power-corrected Polaris dipped to 1.796 m/s for one frame at the 2 m/s step (limit 1.80); its means were
  within 7 %. Reported, not gating.

## 8. What the vehicles are, and the caveats that go with each claim

**Polaris** (Chrono's JSON Polaris, based on measurements of a 2013 Polaris MRZR: 1,378 kg, all four wheels driven,
0.33 m tyres; scout S1 section 3). Changes needed to run it in our loop (PLAN 1.2, NOTES_M1):
- **Re-framed reference point.** The stock chassis reference sits at the front axle, 0.397 m below the axle line,
  so the settled vehicle's reference point is below the ground and fails the collector's launch check. It was moved
  to mid-wheelbase (every chassis-frame location shifted by (1.35763, 0, -0.42) m; physics-identical by test). This
  also puts the goal and exit rules on the body centre, as for the HMMWV and the Gator.
- **Spawn** at ground + 0.40 m (the frozen collector uses + 0.75 m); launch height 0.385 m on f104 soil.
- **Stock driveline defect.** Chrono's simple driveline as shipped reports the driveshaft speed as wheel speed x 0.25
  but multiplies the torque by 4, so the wheels get about 16 times the engine's power; the engine never revs up and
  the gearbox stays in first gear (scout S5 section 3.2, S1 section 3.3; a Chrono change of 2024-12-10). Below about
  6 m/s it behaves like a one-gear 4WD vehicle with a plausible wheel torque (about 2.8 kN m, the same as the
  power-corrected model; CRITIC section 1.1). It is the primary arm, labelled as such; the power-corrected arm
  (driveline reduction 1.0, gear ratios x 0.25) does slightly better, so the smoke verdict does not rest on the
  defect.
- **Driveline type.** Chrono's simple driveline splits torque 50/50 front/rear with a limited-slip bias of 2 per
  axle, so one spinning wheel cannot starve the others. With open differentials the Polaris fails 9 points more.
- **Soil contact wheels** = one cylinder per wheel, radius 0.25 m (stock 0.33 m minus 0.08 m), stock width
  0.2121 m, chosen so the settled sinkage matches the HMMWV's (S1 section 6.2). With 0.33 m cylinders: +2.1 points.
  The soil sees every wheel as wider than it is at 0.08 m spacing, which favours the narrow Polaris tyre most (S5
  section 3.4). The vehicle itself runs on rigid-mesh tyres, as in the other collectors.
- **No chassis collision shape** in Chrono's Polaris data: the body cannot touch the soil (see "all vehicles").

**M113** (NOTES_M2, amendment 9.1 item 3):
- **Chrono's C++ M113 class, built exactly as `veh.M113` builds it** (identical poses by test): single-pin track
  shoes (63 + 64), SMC contact, brake-steering driveline (steering brakes the inner track), shafts engine and
  automatic shafts transmission with torque converter; 11,343 kg.
- **One physics change: the brake.** The stock simple brake never locks (the braked stock M113 rolls back at
  0.89 m/s even on rigid 10 deg ground), and asking for the shafts brake is silently ignored for single-pin shoes.
  It was replaced by Chrono's own shafts-brake construction (a clutch per sprocket, 10,000 N m).
- **Track shoes on soil = one flat pad box each** (127 bodies, 18 soil markers per shoe). Chrono's own track recipe
  gives no soil markers at 0.08 m spacing (the M113 fell through the soil in the scout, S5 section 0.3). No grousers
  (the braked M113 holds 10 and 15 deg soil without them). An uncalibrated stand-in, not Chrono's validated track
  model.
- **0.5 ms step** for the M113 rows (1 ms gives non-finite states at once); soil config `configs/crm_m113.json` =
  the production file with only the step changed. The Gator at 0.5 ms gives the same 93.8 %.
- **Breakthrough stop.** The plan declared it effectively inactive; it fired on 15 of 287 M113 drives (section 4).
- **Reference point at the front sprocket** (the centre of mass is 2.0 m behind it). Goals count when the front
  sprocket is within 2.5 m of the goal, i.e. about 2 m early: slightly optimistic for the M113.
- **Re-geared variant = every gearbox ratio / 4** (the stock model's sprocket push is at most 0.35 of its weight; a
  real M113 climbs about 31 deg, S5 section 0.5). The stock arm measures Chrono's gearing, not tracks versus wheels.
- The frozen follower tracks the M113 1.5-2 times worse than the HMMWV on rigid ground (S2 section 1.4).

**All vehicles:**
- **The body is not coupled to the soil** for any vehicle: the chassis passes through soil without resistance. The
  belly flag records how often that matters (0 % for the Polaris and the M113, about 4 % for the Gator here).
  Results are optimistic for every vehicle where the body would drag.
- **Thin soil over a rigid floor.** 0.24 m of soil (4 particle layers) over a floor that pushes only on soil, never
  on the vehicle. The floor carries most of every vehicle's weight, which caps sinkage, favours light vehicles and
  hides the M113's low ground pressure (S5 section 0.1).
- **Same follower, same routes.** The frozen route follower and stop rules are unchanged for every vehicle; the
  answer is "better on our routes with our follower", not a general vehicle ranking.
- **Sample size.** 144 routes in 24 groups. Group intervals are given for every comparison; the Polaris and
  re-geared M113 margins are far outside them.
