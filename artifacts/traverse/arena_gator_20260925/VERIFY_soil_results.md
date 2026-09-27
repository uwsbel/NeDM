# VERIFY soil results: skeptical check of RESULTS_soil.md (2026-09-26, 19:40-19:58)

**Verdict: PASS.** Every number I recomputed from the raw soil outcome files with my own code matches `RESULTS_soil.md`
(to Monte-Carlo precision for bootstrap p-values). Both soil primary tests and the whole declared family of four reject
after Holm, also with my own bootstrap and with my own rigid P3 / P4 recomputed from the raw rigid drives. Task B's
primary (Gator-trained vs HMMWV-trained planner, both on the Gator) holds. The pick locks predate every new drive, and
the reuse of the headroom drives is correct by content. I fixed the wording in 11 places of `RESULTS_soil.md` (section 9).
None of the fixes changes a declared decision. The largest is the training-noise paragraph, which contradicted itself.

K3 = this folder, G3 = `/work1/dannegrut/harry/experiments/arena_gator_20260925`. No job or job step was submitted. The
only cluster-side action was read-only extraction on the login node, into new paths (`G3/tools/verify_sr/`).

## 0. How I checked

Four new scripts. None imports the analysis, index or label code under test (`ag_eval_index`, `ag_analyze`,
`ga_analyze`).

| script | where | what |
|---|---|---|
| `scripts/ag_vsr_recount.py` | local | Reads the plan (the 5 soil mapping files) and the raw run folders `e6/runs_soil/<id>/outcome.json` (fail = status is not `goal_reached`). Recomputes the declared group sets from the case folders (md5 rule), clusters (nearest terrain feature to the start-goal midpoint, own code), rates, paired group and cluster bootstraps (20,000 resamples, seed 20260926), McNemar and Holm. Also recomputes rigid P3 / P4 from the raw `e6/runs_rigid` drives, with my own unsafe label. |
| `scripts/ag_vsr_picks.py` | local | Re-derives every line of `e6/picks/LOCK_crm.sha256` from the pick folders. Checks the mapping against the pick manifests and the route files, the task rows against the picks, the soil_v2 reuse by content, and the soil_v3 composition. |
| `scripts/ag_vsr_cluster.py` | login node (`python3`, numpy) | Evaluation drives: collection requests, file hashes, my own re-implementation of the soil physics checks, Gator belly flags, times. Gator collection: all 15,235 ids against the HMMWV `collect_v1` twins, with the speed profile read from each route file's meta. Wheel-radius pilot. |
| `scripts/ag_vsr_readouts.py` | local | Descriptive read-outs (unsafe, tilt, time ratios, predicted risk, simulated hours) and the summaries of the cluster outputs. |

Outputs are in `verify_soil_results/`:
- `recount.json` / `recount.txt`;
- `picks_reuse.json`;
- `readouts.json`;
- `cluster/{eval,gator,pilot}.json` and `cluster/logs/`.

The cluster copies are in `G3/tools/verify_sr/out`.

## 1. Completeness and integrity of the 12,310 drives

- **Plan and drives.** The plan has 12,600 arm results (arm = one model or route rule), with no duplicate
  (vehicle, group, arm). They map to 12,310 distinct drives, and every drive folder is present and complete (0 missing,
  0 without a completion marker).
- **Local copies = the cluster files.** The sha256 of `outcome.json`, `trajectory.npz` and `case.json` equals the
  completion record, and the status matches it (0 mismatches). The sha256 of `outcome.json` and `episode_complete.json`
  equals the cluster copies for all 12,310 drives.
- **Each drive ran what was planned.**
  - Route: `reference.json` (the collector's copy of `--route`) = the task row's route file = the collection request's
    route hash = the route file on G3 now, byte for byte: 0 mismatches over 12,310.
  - Case file bytes: 0 mismatches. Case id = group: 0 mismatches.
  - Soil config: `crm_main_step1ms_spacing008` (1 ms, 0.08 m) on all drives.
  - Collector: `cb6792be` on all drives.
- **Vehicle records.**
  - All 2,377 Gator drives carry the Gator block with rear / front cylinder radius 0.2275 / 0.19575 m, spawn 0.35 m,
    wrapper `b52e1fa6` and switch `072716ee`.
  - None of the 9,933 HMMWV drives has a vehicle block.
- **No retries.** `soil_v1/failed` and `soil_v1/stale` contain nothing: no evaluation drive was retried or replaced.
- **Physics checks, own re-implementation** (shape, finite states, launch check, explosion, breakthrough without a
  stall): 12 of 12,310 are flagged, all "breakthrough without a stall", all on HMMWV drives. This is the same set as
  the report's.
  - Each soil primary loses 4 groups to the flags (a different 4 each). Dropping them gives P1 -2.36 and P2 -2.86, as
    reported.
- **Agreement with the frozen index.** My plan-based table equals `e6/index/soil_eval_v1.json` on all 12,600 rows,
  with 0 label and 0 cluster mismatches.

## 2. Declared group sets

- **Unseen subset.** I recomputed it from the case folders: the 125 lowest-md5 of the 250 groups of each test arena.
  It equals `suites/soil_unseen_subset.json` (`c991e542...`).
- **f104 in-distribution set.** I recomputed it as the 200 lowest-md5 hill/crater groups among 450 candidates. It
  equals `f104_indist_200.json`.
- **Planned groups per arm.** Each arm's planned groups equal the declared set exactly:

| arm | groups |
|---|---|
| M1b, M3b, M2 | the 1,000 unseen |
| M1a | unseen + the 800 f104 suite + 2 x 150 held-out = 2,100 |
| M3a, A3 | unseen + the 200 in-distribution + held-out = 1,500 |
| straight 6 (HMMWV) | 2,100 |
| G, H and straight 6 on the Gator | the 800 f104 suite |

No extra arm appears.

## 3. Pick locks and timing

- **Lock file.** I re-derived all 69 lines of `e6/picks/LOCK_crm.sha256` from the pick folders as they are now:
  - the routes lock (file names + content, sorted);
  - the manifest sha256;
  - each folder's `PICKS_LOCKED.sha256`;
  - the ALL line (`2c766a52...`).

  All match.
- **Picks, mapping and task rows agree.** For all 12,600 mapping entries:
  - route hash = the pick manifest's;
  - pick route file unchanged;
  - the task row's route has the same drive content;
  - same group and same case.

  0 mismatches.
- **Order in time** (09-25):

| event | time |
|---|---|
| pick manifests created | 14:23:16-14:34:13 |
| lock file written | 14:34:23 |
| spec `spec_soil_v1.json` written (sha256 `cf813e0a...`, equal to the LOG line) | 14:40:28 |
| first new evaluation drive requested (collection request time on G3) | 14:41:25 |
| last new evaluation outcome | 23:32:48 |

  So the locks and the spec predate every new drive. The spec's M1 / M3 definitions (the mean of the two ensembles)
  already stood in `spec_rigid_v1.json` at 13:26.
- **Drives that existed before the spec.** The reused soil_v2 headroom drives ran 04:48-06:07, and their pooled rates
  were known from the headroom scan. This was disclosed in PLAN 7.10 but not in the RESULTS header, so I added it there
  (section 9).

## 4. Content-based reuse (soil_v2 headroom drives)

- **What is reused.** 348 arm results reuse 310 soil_v2 drives:
  - 297 straight-route headroom drives;
  - 13 drives of the frozen f104 model's headroom picks (`__Scrm_B`).

  Every reused row is an HMMWV row, with no Gator switch in `extra`. It has the same group and the same case file, and
  its route has the same waypoints, speeds, stations and headings. Those four fields are the only ones the soil
  collector reads (`read_route`, `make_driver`, the speed schedule).
- **Other fields.** 318 of the 348 are byte-identical files; the other 30 differ only in `meta`. The episode seed is
  provenance only (it is recorded, not used by `crm_collect`).
- **Who reuses what.**
  - The 300 straight 6 m/s arms: 297 on straight drives, and 3 on frozen-model drives whose pick was the straight
    route.
  - The 48 model picks: 27 equal the straight route, and 21 equal the frozen model's pick.
- **The report's accounting was wrong in two places** (both fixed):
  - It said the 48 model picks equal "one of those [straight] headroom routes".
  - It said "Beyond that, 290 arm results share a drive". 290 = 12,600 - 12,310 also counts 38 of the soil_v2 reuses.
    The correct count is 252 = 131 merged within one priority tier + 121 with an earlier tier.
- **soil_v3.json** (`654ca61b...`) = soil_v2 unchanged as a prefix + exactly the 12,000 evaluation rows (tiers -9 to
  -5; 9,623 HMMWV, 2,377 Gator).

## 5. Task A soil, goal reached per arm (recomputed from `outcome.json` status)

The 8 unseen arenas, 125 declared groups each; the pooled column has 1,000 groups.

| arm (plain label) | g260 | g271 | g251 | g247 | g258 | g268 | g263 | g241 | pooled |
|---|---|---|---|---|---|---|---|---|---|
| M1a (f104 only, ensemble 1) | 96.8 | 88.0 | 86.4 | 89.6 | 87.2 | 84.8 | 94.4 | 80.8 | **88.5** |
| M1b (f104 only, ensemble 2) | 92.0 | 89.6 | 80.0 | 85.6 | 88.0 | 88.8 | 92.0 | 83.2 | 87.4 |
| M1 (f104 only, average) | 94.4 | 88.8 | 83.2 | 87.6 | 87.6 | 86.8 | 93.2 | 82.0 | 87.95 |
| M2 (two arenas, same total) | 93.6 | 88.0 | 82.4 | 88.0 | 89.6 | 86.4 | 95.2 | 80.8 | 88.0 |
| M3a (three arenas, same total, ensemble 1) | 97.6 | 88.8 | 88.8 | 89.6 | 86.4 | 89.6 | 93.6 | 88.0 | **90.3** |
| M3b (three arenas, same total, ensemble 2) | 95.2 | 90.4 | 85.6 | 89.6 | 88.0 | 92.8 | 95.2 | 85.6 | 90.3 |
| A3 (three arenas, all data) | 93.6 | 94.4 | 87.2 | 92.0 | 88.8 | 86.4 | 95.2 | 87.2 | **90.6** |
| straight 6 m/s | 68.0 | 59.2 | 56.0 | 44.0 | 75.2 | 52.8 | 64.0 | 60.8 | 60.0 |

**Pooled, one ensemble each** (unseen, fail):
- M1a vs M3a vs A3: 11.5 / 9.7 / 9.4 %.
- M3a vs M1a: -1.8 points, cluster 90 % [-3.6, 0.0], 61 better / 43 worse groups.
- A3 vs M1a: -2.1 points, [-3.9, -0.4].

**Near vs spread** (M1 / M3 / A3): near 11.5 / 9.3 / 8.2 %, spread 12.6 / 10.1 / 10.6 %.

**In distribution and held out:**
- f104 in-distribution: M1a 3.5, M3a 3.0, A3 5.0, straight 34.5 %.
- g203 / g228 held-out: M1a 14.0 / 26.0, M3a 8.0 / 22.0, A3 8.7 / 13.3, straight 46.0 / 46.7 %.

Every rate in RESULTS 2.1 and 2.4 matches.

## 6. Primary tests, cluster bootstrap, Holm

My own cluster bootstrap:
- 80 clusters of (arena, nearest feature), identical to the index's;
- clusters resampled with replacement; statistic = the paired differences summed over the drawn clusters / the number
  of groups drawn;
- 20,000 resamples.

| test | diff | my cluster 90 % | my p (one-sided) | report | my group 95 % |
|---|---|---|---|---|---|
| P1 soil M3 vs M1 (three arenas vs f104 only, same total) | -2.35 | [-3.66, -1.08] | 0.0018 | [-3.68, -1.08], 0.0017 | [-3.95, -0.75] |
| P2 soil A3 vs M1 (three arenas all data vs f104 only) | -2.65 | [-3.98, -1.31] | 0.0007 | [-4.05, -1.31], 0.0012 | [-4.40, -0.90] |
| P3 rigid fixed 2 m/s M3 vs M1, unsafe (own recount from raw rigid drives) | -2.15 (6.175 vs 8.325 %) | [-3.42, -1.01] | 0.0007 | 0.0015 | - |
| P4 rigid fixed 2 m/s A3 vs M1, unsafe (own recount) | -2.975 (5.35 vs 8.325 %) | [-4.12, -1.92] | 0.00005 | 0.00025 | - |

**Holm over the four:**

| inputs | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| my soil p + their rigid p | 0.0030 | 0.0021 | 0.0030 | 0.0010 |
| my soil p + my rigid p | 0.0021 | 0.0021 | 0.0021 | 0.0002 |
| my soil p, rigid entered as p = 1 | 0.0054 | 0.0028 | - | - |
| report | 0.0037 | 0.0037 | 0.0037 | 0.0010 |

All reject at 0.05, in every variant. The differences from the report are Monte-Carlo noise: the report used 4,000
resamples, whose floor is 2.5e-4.

- **Family file.** `family_v1_S2.json` differs from `family_final_E6b.json` only in the `created` time stamp. The
  report called them "identical"; fixed.
- **McNemar.** McNemar better / worse: P1 102 / 64 and P2 106 / 53, on half-integer composite outcomes (reported as
  counts only).
- **±2-point rule.** Within ±2 points (90 % cluster): M2 vs M1 [-1.60, +1.53], A3 vs M3 [-1.57, +1.00], M3a vs M3b
  [-1.66, +1.65]. P1 and P2 are not within ±2 points, and their lower ends are 1.1 / 1.3 points. The report's reading
  is correct: a gain is shown, but not a gain of 2 points or more.
- **Near vs spread** (P1): -2.20 vs -2.50. The difference is +0.3, 95 % [-2.75, +3.48].
- **Per arena.**
  - Signs: P1 is better on 7 of 8 arenas, P2 on 6 of 8.
  - Only P2 on g271 has a 95 % group interval that excludes 0.
  - M1a - M1b per arena ranges from -6.4 (g251) to +4.0 (g268).
  - Pearson correlation with the distance to the nearest training arena: -0.32 (P1) / -0.22 (P2); with the map-lookup
    error: -0.04 / +0.40.
- **Dose.**
  - The OLS slope over M1 / M2 / M3 is -1.175 points per arena.
  - M3 vs M2 -2.30 [-3.78, -0.80]; A3 vs M3 -0.30.
- **Unsafe label.**
  - P1 -2.4, P2 -2.7 on "unsafe".
  - The largest unsafe-minus-fail on a pooled arm is 0.33 (M3a held out); per arena it is 0.8 (M1a on g263). This
    matches "at most 0.8".
- **Gaps.**
  - Unseen minus in-distribution: +8.0 / +6.7 / +4.4 (M1a / M3a / A3).
  - Unseen minus held-out: -8.5 / -5.3 / -1.6.
- **In-arena.** A3 vs M1a on the held-out groups: -9.0 [-13.1, -5.0].
- **No harm on f104.**
  - A3 vs M1a +1.5: one-sided upper bound +2.89 (cluster) / +4.0 (group), 6 worse / 3 better groups, McNemar 0.51.
  - M3a vs M1a -0.5: upper bound +1.47 (cluster) / +2.0 (group).
- **Descriptive.**
  - Median time ratio on joint successes: M3a / M1a 0.868 (14.55 vs 16.95 s, 842 groups); A3 / M1a 1.00; M1a /
    straight 1.835.
  - Mean predicted risk of the picks: M1a 1.07 % (unseen) and 0.11 % (in-distribution); A3 2.24 %.
  - Simulated hours: HMMWV 47.5, Gator 18.7.

## 7. Task B soil (800 f104 groups)

**Rates** (fail = goal not reached):
- G on the Gator 34.75 %, with statuses 522 goal / 264 blocked / 14 broke through.
- H on the Gator 51.88 % (385 / 405 / 10).
- Straight 6 m/s on the Gator 85.75 % (114 / 679 / 7).
- H on the HMMWV 5.50 %; straight 6 m/s on the HMMWV 32.25 %.

**Primary.**
- G vs H on the Gator: -17.13 points, my cluster 90 % [-21.6, -12.4], group 95 % [-20.1, -14.0].
- 159 better / 22 worse groups, McNemar 8.3e-27.

**Criteria and anchors.**
- Criterion 1: G vs straight 6 m/s on the Gator -51.0 [-55.4, -46.3], 417 / 9 groups.
- H on the Gator vs H on the HMMWV: +46.4 (372 vs 1 groups).
- Straight route, Gator vs HMMWV: +53.5 (433 vs 5).
- H picks the same routes on both vehicles: 800 / 800, and the straight arms are identical too.

**Headroom closed:** G on the Gator 59.5 %, H on the Gator 39.5 %, H on the HMMWV 82.9 %.

**Split by part of the suite:**
- the 200 in-distribution groups: G 36.0 vs H 60.0 %;
- the other 600: 34.3 vs 49.2 %.

**Time and risk.**
- Time ratio on joint successes: 0.832 (17.6 vs 21.75 s, 363 groups).
- Mean predicted risk of the picks: G 18.4 %, H 0.21 %.
- Tilt > 30 deg: G 2.5 %, H 2.6 %, straight on the Gator 12.1 %.

**Belly flag on the Gator evaluation drives:** 5.47 % overall (G 2.1, H 3.1, straight 11.1 %). None of the flagged
drives reached the goal.

All of this matches the report. The report's per-arm belly ranges were per sub-suite; I replaced them with per-arm
values.

Criterion 3 (offline AUC) I did not recompute. It was re-scored by VERIFY_S1. The report quotes it correctly (met on the
56 val groups only).

## 8. Gator collection read-out, all 15,235 ids (own QA and belly code, login node)

**Validation.**
- The 15,235 `gator__` training rows of `G3/tasks/soil_v3.json` are all complete and all pass my re-implemented
  physics checks: 0 flags, 0 launch-check failures, 0 non-finite states.
- Every one carries the Gator block with the calibrated rear radius 0.2275 m and `wheel: calibrated`.

**Belly-in-soil flag.**
- Definition: the lowest hull point more than 0.05 m under the surface for more than 1 s in a row (more than 20
  consecutive 0.05 s frames). The frames are contiguous in every file.
- 8.51 % of drives; 9.83 % if the total time over 1 s is counted instead.
- Designed routes alone: 10.06 % (922 / 9,168), just above 10 %. On-policy routes: 6.16 %.
- All 1,296 flagged drives failed.

**Failure, Gator vs the HMMWV `collect_v1` twins** (identical ids, 15,235 pairs):
- 88.15 vs 68.05 %; 3,263 Gator-only vs 201 HMMWV-only failures.
- Statuses: Gator 13,231 blocked / 1,805 goal / 198 broke through / 1 timeout; HMMWV 8,372 broke through / 4,867 goal /
  1,985 blocked / 9 timeout / 2 rollover.
- Simulated hours: 141.13 vs 91.51.

**By designed speed profile.** The profile was read from each route's meta `speed_profile_id`; it equals route
index % 4 on all 9,168 designed routes, so the tool's shortcut is right.

| profile | routes | Gator fails | HMMWV fails | belly flag |
|---|---|---|---|---|
| constant 2 m/s | 2,302 | 91.27 % | 86.19 % | 8.99 % |
| constant 4 m/s | 2,254 | 83.81 % | 56.26 % | 11.49 % |
| constant 6 m/s | 2,326 | 82.67 % | 33.53 % | 11.39 % |
| smooth 2-6-2 m/s | 2,286 | 84.21 % | 54.55 % | 8.36 % |
| planner proposals (on-policy) | 6,067 | 92.17 % | 83.88 % | 6.16 % |

**By tier and split.**
- Tiers 0-12: 87.32-89.33 %.
- Split, Gator / HMMWV: train 88.0 / 67.5, val 88.4 / 72.8, test 90.9 / 73.9 %.
- Tiers 0-6: 8,399 ids; 87.93 vs 67.91 %; belly 8.27 %; 77.65 vs 50.33 h.

**Wheel-radius pilot** (`G3/pilot_gator/soil/runs`):
- 144 routes, calibrated rear radius 0.2275 m vs 0.3075 m.
- Failure 93.75 vs 80.56 % (-13.2 points); 20 routes flip to goal, 1 the other way. HMMWV 69.4 %.
- All 288 runs pass the checks.

Every number in RESULTS 3.2 / 3.3 matches.

## 9. Wording fixed in RESULTS_soil.md

1. **Header.** "before any soil evaluation outcome existed": now gives the lock / spec / first-drive times and
   discloses the reused headroom drives, whose outcomes were known before the spec.
2. **Headline "Size".** "real but small" became "pass the declared tests but are small", plus the training-noise
   caveat.
3. **Headline "Where the gain comes from".**
   - "the second arena buys nothing": now "no measurable gain", with its interval, and it notes that M2 is a single
     ensemble.
   - "The third arena buys all of it": now "the gain appears at the step to three arenas". The arenas were added in one
     fixed order (g203, then g228), so the design cannot tell a third arena apart from g228 in particular.
   - "Doubling the data adds nothing": now "about doubling … no meaningful further gain", with its interval.
4. **Headline "Near vs spread".** "The gain is the same" became "similar". The difference interval [-2.8, +3.4] cannot
   exclude a gap of about 3 points.
5. **Section 1, reuse accounting.** 310 distinct soil_v2 drives, 27 straight + 21 frozen-model model picks, and 252
   (not 290) further shared arm results.
6. **Section 2.2.** The family file is identical to the rigid track's "except for its creation time".
7. **Section 2.3, training noise.** The text called the variance-doubled case (0.78 points per ensemble) "noise doubled
   … still rejected", and then concluded "not a noise twice as large". The sd-doubled case (1.1 points) is the one that
   is not rejected. The labels now say which is which, and the conclusion reads "survives the estimated noise and a
   noise about 1.4 times as large, but not twice as large (in standard deviation)".
8. **Section 2.5.** "doubling the data does not add" became "adds no meaningful gain (up to 1.6 points not excluded)".
9. **Section 2.6.** "2-3 times as often" became "about 1.9 to 3.3 times (3.3 / 3.2 / 1.9 x)". "The gap shrinks with more
   arenas" is now qualified: the intervals overlap, and A3's smaller gap comes partly from failing more on f104.
10. **Section 2.7.** It now states that the intervals are cluster 90 % from single ensembles. The M3a no-harm claim now
    notes that the group-bootstrap bound is +2.0, exactly at the margin, so it rests on the 9-cluster interval.
11. **Section 3.**
    - Belly row: per-arm evaluation values, the designed-route share (10.1 %, above the 10 % limit on the constant 4 / 6
      m/s routes; the declared criterion is over all soil drives and is met), and the fact that all flagged drives
      failed.
    - "stalls where the HMMWV drives through" became "on many routes the HMMWV completes (3,263 vs 201)".
    - "still works" became "meets the first two declared 'works' criteria (criterion 3 only on the val groups)".

**The report I was given** repeats the flaw in fix 7: it says "both soil tests survive that, and twice that noise". The
"twice" there is twice the variance, not twice the noise.

## 10. What remains true but thin

- The soil gains (2.35 / 2.65 points) rest on two ensembles per arm at most. They survive the estimated training noise,
  not a doubled one.
- The step-by-step attribution rests on one M2 ensemble and one arena order.
- f104 no-harm and task B rest on 9 terrain-feature clusters. The group bootstrap and McNemar agree for task B.
- The Gator soil result depends on the wheel stand-in. It is 13.2 points against a 15-point rule, on 144 pilot routes,
  and the belly flag is at 10 % on the designed routes.
- Rigid P3 / P4 were recomputed only from the local synced rigid drives: the labels and the family, not their
  provenance, which is the rigid checker's job.
