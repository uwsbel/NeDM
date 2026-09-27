# RESULTS soil: more training arenas (task A) and the Gator on f104 (task B), stage-1 models (2026-09-26)

Soil (deformable ground, CRM), speed free, CEM 4 x 64 planning from a standing start, one drive per start/goal group and
arm. "Fail" = goal not reached. Every model here is a stage-1 soil model: trained on tiers 0-6 (7 routes per group) of
every arena and vehicle (PLAN 7.10). The analysis follows the frozen spec `e6/analysis/spec_soil_v1.json` (sha256
`cf813e0a...`, written 14:40 on 09-25, after the picks were locked at 14:34 and before the first new evaluation drive
started at 14:41). The only outcomes that existed earlier are the reused spread-arena headroom drives of 04:48-06:07
(the straight route and the frozen f104 model on 300 spread groups; their rates were known from the headroom scan,
PLAN 7.10); 48 model-arm results reuse 30 of those drives (18 straight, 12 frozen-model) because their picks are
identical. Everything that is not in that spec is marked **supplementary (post hoc)**.

Models (plain label, code):

| plain label | code | training data (soil, tiers 0-6) |
|---|---|---|
| f104 only, ensemble 1 / ensemble 2 / average of both | M1a / M1b / M1 | f104, 1,089 groups (29,210 fitted rows) |
| two arenas, same total data | M2 | f104 545 + g203 545 groups (29,048 rows) |
| three arenas, same total data, ensemble 1 / 2 / average | M3a / M3b / M3 | f104 + g203 + g228, 363 groups each (29,071 rows) |
| three arenas, all data | A3 | f104 1,089 + g203 559 + g228 520 groups (57,898 rows) |
| Gator-trained | G | Gator drives of the 8,399 f104 ids of tiers 0-6 |
| HMMWV-trained (task B) | H = M1a | HMMWV drives of exactly those 8,399 ids (the file is byte-identical to the M1 file) |
| straight route at 6 m/s (no model) | straight 6 | - |

## 0. Headline

**Task A (soil, 8 unseen arenas x 125 declared groups = 1,000 groups).** Goal reached: f104 only 87.95 % (the two
ensembles 88.5 / 87.4 %), two arenas at the same total 88.0 %, three arenas at the same total 90.3 %, three arenas with
all data 90.6 %, straight 6 m/s 60.0 %.

- **P1, three arenas vs f104 only at the same total data (M3 vs M1):** 9.70 vs 12.05 % not reached, **-2.35 points**
  (19 % fewer failures), cluster-bootstrap 90 % interval [-3.7, -1.1], one-sided p 0.0017.
- **P2, three arenas with all data vs f104 only (A3 vs M1):** 9.40 vs 12.05 %, **-2.65 points** (22 % fewer
  failures), 90 % [-4.0, -1.3], p 0.0012.
- **Declared Holm family of four** (with the rigid fixed 2 m/s tests from `results_rigid_v1.json`): all four improve.
  Holm-adjusted p: P1 0.0037, P2 0.0037, P3 (rigid, three arenas same total) 0.0037, P4 (rigid, three arenas all
  data) 0.0010. The family is complete.
- **Size.** Both soil gains pass the declared tests but are small. The 90 % intervals reach down to 1.1 / 1.3 points,
  so a gain of 2 points or more is not shown. They also rest on two training runs (ensembles) per arm at most: they
  survive the estimated training noise, but not a noise twice as large (section 2.3).
- **Where the gain comes from.** Adding the second arena (g203) at the same total gives no measurable gain (M2 vs M1
  -0.05, 90 % [-1.6, +1.5], within ±2 points; M2 is a single ensemble). The gain appears at the step to three arenas
  (M3 vs M2 -2.3, 90 % [-3.8, -0.8]). The arenas were added in one fixed order (g203, then g228), so this cannot tell
  "a third arena" apart from "g228 in particular". About doubling the data on three arenas gives no meaningful further
  gain on unseen arenas (A3 vs M3 -0.3, 90 % [-1.6, +1.0], within ±2 points).
- **Near vs spread.** The gain is similar on the 4 arenas closest to f104 and on the 4 spread ones (P1 -2.2 / -2.5).
  The difference (+0.3, 95 % [-2.8, +3.4]) is too uncertain to exclude a gap of about 3 points either way. The gain
  shows no clear trend with distance to the nearest training arena, nor with each arena's map-lookup error (8 arenas).

**Task B (soil, 800-group f104 suite, same pick poses).** Goal reached:

| planner | vehicle | goal reached |
|---|---|---|
| Gator-trained (G) | Gator | **65.3 %** |
| HMMWV-trained (H) | Gator | 48.1 % |
| straight 6 m/s | Gator | 14.3 % |
| H | HMMWV | 94.5 % |
| straight 6 m/s | HMMWV | 67.8 % |

- **Declared primary B (G vs H, both on the Gator):** -17.1 points not reached, 90 % [-21.7, -12.5]; G is better on
  159 groups and worse on 22 (McNemar p 8e-27).
- **"Works" criteria:** (1) G beats straight 6 m/s on the Gator: yes. (2) G within 2 points of H or better: yes,
  17 points better. (3) Offline AUC >= 0.95: met on the val groups only (0.952 / 0.957), not on the larger dev-fold +
  val set (0.925). (4) Headroom closed: 59 % for G on the Gator, against 83 % for the HMMWV planner on the HMMWV.
- **Collection, all 15,235 Gator ids:** 100 % validated, 0 crashed or NaN, 0 launch-check failures, belly-in-soil flag
  8.5 % (limit 10 %). The Gator fails 88.2 % of the routes against 68.1 % for the HMMWV on the identical routes, and
  needs 141.1 simulated hours against 91.5 (1.54 x).

## 1. What was run and checked

- **Drives.**
  - 12,310 soil drives (G3/soil_v1/runs), 12,600 arm results (identical routes are driven once), 0 missing.
  - The index files only (outcome, trajectory, completion marker, case, vehicle_extra) were synced into
    `e6/runs_soil` by `scripts/ag_s2b_sync.sh`. `scripts/ag_eval_index.py` (unchanged) built `e6/index/soil_eval_v1.json`.
  - 348 arm results reuse 310 earlier soil_v2 spread-arena headroom drives with identical route content (same group,
    case and vehicle; waypoints, speeds, stations and headings equal): the 300 straight 6 m/s arms (297 on the straight
    headroom drives, 3 on the frozen f104 model's headroom drive, whose pick was the straight route), plus 48 model
    picks (27 equal to the straight route, 21 equal to the frozen f104 model's headroom pick). Beyond that, 252 arm
    results share a new drive with another arm whose route is identical (131 within one priority tier, 121 with an arm
    of an earlier priority tier); 12,600 arm results come from 12,310 drives. (Verifier correction: the first version
    said "290 arm results" here, which also counted 38 of the soil_v2 reuses, and said all 48 model picks equal a
    straight route.)
- **Groups.** They equal the declared sets exactly:
  - the 1,000-group unseen subset (`suites/soil_unseen_subset.json` `c991e542...`);
  - the 200 f104 in-distribution groups (`f104_indist_200.json` `f21a438f...`);
  - the 800 f104 suite groups;
  - the 2 x 150 held-out groups of g203 and g228.
- **Provenance** (`scripts/ag_s2b_provenance.py`, login node, all 12,310 drives): every drive used the locked pick's
  route file (path and bytes), the right case, the frozen soil config `crm_main_step1ms_spacing008` (1 ms, 0.08 m)
  and the frozen collector (`cb6792be`). All 2,377 Gator drives carry the Gator block with the calibrated wheel radii
  (0.19575 / 0.2275 m), spawn 0.35 m, and wrapper `b52e1fa6` + switch `072716ee`. The 9,933 HMMWV drives carry no
  vehicle block. 0 mismatches.
- **Independent recount.** Rates recomputed straight from `outcome.json` status (not through the label code) equal the
  index for every arm: M1a 11.5, M1b 12.6, M3a 9.7, M3b 9.7, A3 9.4, M2 12.0, straight 6 40.0 % on unseen, and the task
  B arms. P1 -2.35 and P2 -2.65 come out the same.
- **Physics QA of the evaluation drives** (`scripts/ag_s2_extras.py`, unchanged, run on the login node):
  - 12 of 12,310 carry a `crm_qa` flag, all "breakthrough without a stall" on HMMWV drives (0.1 %).
  - 0 launch-check failures, 0 non-finite states.
  - The declared analysis keeps every driven outcome. Dropping the 4 affected unseen groups of each test (a different
    4 for P1 and P2) changes P1 to -2.36 and P2 to -2.86 (supplementary).
- **Simulated time of the evaluation drives:** HMMWV 47.5 h, Gator 18.7 h.

## 2. Task A soil

### 2.1 Rates (fail = goal not reached, %)

| model | unseen (1,000) | near 4 (500) | spread 4 (500) | f104 in distribution (200) | g203 / g228 held out (150 each) |
|---|---|---|---|---|---|
| f104 only, ens. 1 (M1a) | 11.5 | 9.8 | 13.2 | 3.5 | 14.0 / 26.0 |
| f104 only, ens. 2 (M1b) | 12.6 | 13.2 | 12.0 | - | - |
| f104 only, average (M1) | 12.05 | 11.5 | 12.6 | - | - |
| two arenas, same total (M2) | 12.0 | 12.0 | 12.0 | - | - |
| three arenas, same total, ens. 1 (M3a) | 9.7 | 8.8 | 10.6 | 3.0 | 8.0 / 22.0 |
| three arenas, same total, ens. 2 (M3b) | 9.7 | 9.8 | 9.6 | - | - |
| three arenas, same total, average (M3) | 9.7 | 9.3 | 10.1 | - | - |
| three arenas, all data (A3) | 9.4 | 8.2 | 10.6 | 5.0 | 8.7 / 13.3 |
| straight 6 m/s | 40.0 | 43.2 | 36.8 | 34.5 | 46.0 / 46.7 |

"Unsafe" (goal not reached or rolling backwards) differs from "fail" by at most 0.8 points on any arm, and the
contrasts on "unsafe" are the same: P1 -2.4, P2 -2.7.

### 2.2 Primary tests and the declared family of four (PLAN 7.3)

One-sided p from the bootstrap over (arena, nearest terrain feature) clusters: 80 clusters on unseen soil, 4,000
resamples. Holm is taken over the four tests.

| test | plain question | rates (test vs ref) | diff (points) | 90 % cluster interval | p (one-sided) | Holm-adjusted p | decision |
|---|---|---|---|---|---|---|---|
| P1 soil M3 vs M1 | three arenas vs f104 only, same total | 9.70 vs 12.05 % | **-2.35** | [-3.68, -1.08] | 0.0017 | 0.0037 | **improves** |
| P2 soil A3 vs M1 | three arenas with all data vs f104 only | 9.40 vs 12.05 % | **-2.65** | [-4.05, -1.31] | 0.0012 | 0.0037 | **improves** |
| P3 rigid fixed 2 m/s M3 vs M1 (unsafe) | from `results_rigid_v1.json` | 6.18 vs 8.33 % | -2.15 | [-3.37, -1.01] | 0.0015 | 0.0037 | improves |
| P4 rigid fixed 2 m/s A3 vs M1 (unsafe) | from `results_rigid_v1.json` | 5.35 vs 8.33 % | -2.98 | [-4.14, -1.92] | 0.00025 | 0.0010 | improves |

- Family file: `e6/analysis/family_v1_S2.json` (`scripts/ag_s2_family.py`, rigid input `results_rigid_v1.json`
  sha256 `712e3fb6...`, soil input `results_soil_v1.json` `96facb25...`). It is identical to the rigid track's
  `family_final_E6b.json` except for its creation time (every p-value, adjusted p and decision equal).
- The soil decisions do not depend on the rigid results: with P3 / P4 entered as p = 1, the soil adjusted p are
  0.0052 / 0.0050.
- Other intervals for P1 / P2:
  - group bootstrap 95 %: [-3.95, -0.75] / [-4.50, -0.95];
  - cluster 95 %: [-3.92, -0.81] / [-4.36, -1.06];
  - the other clustering (the case's design feature, 81 clusters) 90 %: [-3.72, -0.96] (p 0.0020) / [-3.92, -1.39]
    (p 0.0002).
- Cluster wins / losses: P1 35 / 13, P2 31 / 16.
- **±2-point rule.** P1 and P2 are decided by Holm ("improves"). Their intervals do not lie within ±2 points, and their
  lower ends (1.1 / 1.3 points of improvement) are below 2. So the gain is real, but it is not shown to be 2 points or
  more. Within ±2 points ("no meaningful difference") are: M2 vs M1 (-0.05, [-1.6, +1.5]), A3 vs M3 (-0.3, [-1.6, +1.0])
  and the two M3 ensembles against each other (0.0, [-1.6, +1.7]).

### 2.3 Robustness of P1 / P2 (supplementary unless marked)

- **Every single-ensemble pair points the same way** (unseen, fail, cluster 90 %; exact McNemar on the discordant
  groups):

  | pair | diff | 90 % | p | worse / better groups (McNemar p) |
  |---|---|---|---|---|
  | M3a vs M1a (spec) | -1.8 | [-3.6, -0.1] | 0.050 | 43 / 61 (0.095) |
  | M3a vs M1b | -2.9 | [-5.0, -0.7] | 0.015 | 43 / 72 (0.009) |
  | M3b vs M1a | -1.8 | [-3.8, +0.1] | 0.062 | 48 / 66 (0.11) |
  | M3b vs M1b | -2.9 | [-4.5, -1.3] | 0.003 | 40 / 69 (0.007) |
  | A3 vs M1a (spec) | -2.1 | [-3.9, -0.4] | 0.025 | 42 / 63 (0.050) |
  | A3 vs M1b | -3.2 | [-4.9, -1.5] | 0.001 | 38 / 70 (0.003) |

  The contrast against M1a alone is smaller, because M1a is the better of the two f104-only ensembles.
- **Ensemble a-vs-b noise floor (spec).**
  - Two ensembles trained on the same data differ by -1.1 points (M1a vs M1b, 90 % [-3.2, +0.9], 47 / 58 discordant)
    and 0.0 points (M3a vs M3b, 33 / 33).
  - Per arena they differ far more: M1a vs M1b ranges from -6.4 (g251) to +4.0 (g268), between-arena spread 2.7
    points, I² 50 %.
  - So a single arena's reading carries about as much training noise as the effect itself. Only the pooled numbers
    carry weight.
- **Training-noise sensitivity (supplementary).** The cluster bootstrap resamples groups, not training runs.
  - The two same-data pairs give a rough training noise (standard deviation) of 0.55 points per ensemble. Added to
    the bootstrap variance (normal approximation), P1 has p 0.007 and P2 p 0.007 (Holm with P3 / P4: 0.014 each, both
    rejected).
  - With that noise *variance* doubled (0.78 points per ensemble): 0.017 / 0.019 (Holm 0.034, still rejected).
  - With the noise *standard deviation* doubled (1.1 points per ensemble, as large as the whole M1a-M1b difference):
    0.041 / 0.048 (Holm 0.083, not rejected).
  - Two estimates of training noise are too few to settle this. The conclusion survives the estimated noise and a
    noise about 1.4 times as large, but not a noise twice as large (in standard deviation) as estimated. (Verifier
    correction: the first version called the variance-doubled case "noise doubled", which contradicted this
    sentence.)
- **Random-effects pooling over the 8 arenas (spec):** P1 -2.16 [-3.69, -0.63] (spread between arenas 0), P2 -2.31
  [-4.01, -0.60] (spread 0.5 points, I² 4 %).

### 2.4 Per arena (spec; diff in points, 95 % group-bootstrap interval, 125 groups each)

| arena (role) | distance to f104 / to nearest training arena | map-lookup error (m) | fail: M1 / M3 / A3 / straight 6 (%) | P1 M3 vs M1 | P2 A3 vs M1 |
|---|---|---|---|---|---|
| g260 (near) | 0.65 / 0.61 (g228) | 0.060 | 5.6 / 3.6 / 6.4 / 32.0 | -2.0 [-5.2, +0.8] | +0.8 [-3.2, +4.8] |
| g271 (near) | 0.74 / 0.74 (f104) | 0.065 | 11.2 / 10.4 / 5.6 / 40.8 | -0.8 [-5.2, +3.6] | -5.6 [-10.4, -0.8] |
| g251 (near) | 0.78 / 0.78 (f104) | 0.068 | 16.8 / 12.8 / 12.8 / 44.0 | -4.0 [-8.8, +0.4] | -4.0 [-8.8, +0.8] |
| g247 (near) | 0.91 / 0.72 (g228) | 0.071 | 12.4 / 10.4 / 8.0 / 56.0 | -2.0 [-6.4, +2.4] | -4.4 [-10.0, +1.2] |
| g258 (spread) | 1.11 / 0.88 (g228) | 0.085 | 12.4 / 12.8 / 11.2 / 24.8 | +0.4 [-4.0, +5.2] | -1.2 [-6.4, +4.0] |
| g268 (spread) | 1.32 / 0.92 (g228) | 0.094 | 13.2 / 8.8 / 13.6 / 47.2 | -4.4 [-9.2, 0.0] | +0.4 [-5.2, +6.0] |
| g263 (spread) | 1.46 / 1.19 (g203) | 0.059 | 6.8 / 5.6 / 4.8 / 36.0 | -1.2 [-5.2, +2.8] | -2.0 [-5.6, +1.6] |
| g241 (spread) | 1.79 / 1.34 (g228) | 0.062 | 18.0 / 13.2 / 12.8 / 39.2 | -4.8 [-10.4, +0.4] | -5.2 [-10.4, 0.0] |

- Signs: P1 better on 7 of 8 arenas, P2 on 6 of 8. Only one per-arena interval excludes 0 (P2 on g271). Given the
  ensemble noise floor above, this is the expected pattern for a small pooled effect.
- **Near vs spread (spec).**
  - P1: near -2.2 (cluster 95 % [-4.4, -0.1]), spread -2.5 ([-4.9, -0.3]); difference +0.3 [-2.8, +3.4].
  - P2: near -3.3 ([-5.6, -1.0]), spread -2.0 ([-4.4, +0.2]); difference -1.3 [-4.4, +2.0].
  - No sign that the gain depends on how far the test arena is from f104.
- **Effect vs distance and map-lookup error** (per-arena regressions, 8 points, descriptive only):
  - Against the distance to the nearest training arena: P1 slope -2.4 points per distance unit (Pearson -0.32,
    Spearman -0.33); P2 -2.2 (-0.22 / -0.14).
  - Against the map-lookup error: P1 Pearson -0.04, P2 +0.40.
  - Supplementary regressors:
    - how much closer the nearest of the three training arenas is than f104: P1 Pearson -0.40, P2 +0.14;
    - the f104-only model's own failure rate against its distance to f104: Pearson +0.34.
  - None of these is distinguishable from no relation with 8 arenas.

### 2.5 Dose response (spec)

- **Ensemble averages:** f104 only 12.05 % -> two arenas 12.0 % -> three arenas 9.7 %. Slope -1.17 points per added
  arena [-2.0, -0.4], but the steps are not linear:
  - first step (two arenas vs f104 only): -0.05 [-1.6, +1.5], no meaningful difference;
  - second step (three arenas vs two): -2.3 [-3.8, -0.8], p 0.005.
- **Single ensembles:** M1a 11.5 -> M2 12.0 -> M3a 9.7, slope -0.9 [-1.9, +0.1].
- **More data at three arenas (A3 vs M3):** -0.3 [-1.6, +1.0]. This is within ±2 points: about doubling the data
  adds no meaningful gain on unseen arenas (a further gain of up to 1.6 points is not excluded).

### 2.6 Generalisation gap

**Against the f104 in-distribution groups (spec):**

| model | fail: unseen / in distribution (%) | gap (points) | against straight 6 m/s (difference in differences) |
|---|---|---|---|
| f104 only (M1a) | 11.5 / 3.5 | **+8.0** [+4.6, +11.2] | +2.5 [-5.1, +10.2] |
| three arenas, same total (M3a) | 9.7 / 3.0 | +6.7 [+3.5, +9.5] | +1.2 [-6.3, +8.6] |
| three arenas, all data (A3) | 9.4 / 5.0 | +4.4 [+0.7, +7.6] | -1.1 [-9.1, +6.8] |

- Every model fails about 1.9 to 3.3 times as often on the unseen arenas as on f104's own held-out groups (f104 only
  3.3 x, M3a 3.2 x, A3 1.9 x). The gap is smaller with more arenas (8.0 -> 6.7 -> 4.4 points), but the three
  intervals overlap widely, and A3's smaller gap comes partly from failing more on f104 (5.0 vs 3.5 %), not only from
  failing less on unseen arenas.
- Straight 6 m/s also fails more on the unseen groups (40.0 vs 34.5 %). Measured against it, the model-specific part of
  the gap cannot be told from zero (wide intervals: two independent group sets).
- In "share of the straight route's failures removed", the f104-only model removes 90 % on f104 and 71 % on unseen
  arenas (per arena 48-90 %), the three-arena models 91 / 76 % (M3a) and 86 / 77 % (A3).

**Against the g203 / g228 held-out groups (supplementary):**

- These groups are in distribution for M3a and A3, and unseen sibling arenas for the f104-only model. They are harder
  than the 8 test arenas for every arm:
  - f104 only 20.0 %, M3a 15.0 %, A3 11.0 %, straight 6 46.3 %;
  - so unseen minus held-out: f104 only -8.5 [-13.5, -3.8], M3a -5.3 [-9.9, -0.9], A3 -1.6 [-5.6, +2.3].
- Against straight 6 m/s: -2.2 [-9.3, +5.0] / +1.0 [-5.5, +7.7] / +4.7 [-1.8, +11.2].
- So on the arenas it was trained on, A3 is not clearly better than on arenas it never saw, once group difficulty is
  accounted for.

### 2.7 In-arena effect and no-harm on f104 (spec, plus supplementary per-arena M3a)

- **Training on an arena helps a lot on that arena's held-out groups** (cluster 90 % intervals; single ensembles):
  - A3 vs f104 only: -9.0 [-13.1, -5.1] (g203 -5.3 [-9.0, -1.9], g228 -12.7 [-18.5, -5.7]);
  - M3a vs f104 only: -5.0 [-8.5, -1.4] (g203 -6.0 [-8.5, -3.5], g228 -4.0 [-10.3, +2.6]).
  - On arenas nobody trained on, the gain is 2-3 points (section 2.2).
- **No harm on f104's own groups (200 in-distribution groups, 2-point margin):**
  - M3a vs f104 only: 3.0 vs 3.5 %. The one-sided upper bound +1.5 (cluster) is below 2, so by the frozen
    non-inferiority line M3a is not worse by more than 2 points. The group-bootstrap bound is +2.0, exactly at the
    margin, so this rests on the fragile 9-cluster interval.
  - A3 vs f104 only: 5.0 vs 3.5 %, +1.5 points. The one-sided upper bound is +2.9 (cluster; +4.0 group bootstrap), so
    non-inferiority is **not shown**. There are only 6 vs 3 discordant groups (McNemar p 0.51), so there is no evidence
    of harm either.
  - f104 has only 9 terrain-feature clusters, so its cluster intervals are fragile. The frozen decision label for both
    is "inconclusive".

### 2.8 Time and predicted risk (descriptive)

- **Median time on joint successes** (median paired ratio):
  - three arenas at the same total (M3a) drives 13 % faster routes than f104 only: 0.87 [0.85, 0.89], 14.6 vs 17.0 s;
  - three arenas with all data takes the same time as f104 only (1.00 [0.98, 1.02]);
  - f104 only takes 1.84 x straight 6 m/s.
- **Predicted vs observed risk.** Every model is over-confident about its own pick, most of all off its training
  arenas:
  - f104 only: mean predicted risk 1.1 % against 11.5 % observed on unseen arenas, 0.1 % against 3.5 % on f104;
  - A3: 2.2 % against 9.4 % on unseen arenas.

## 3. Task B soil (stage 1: 8,399 ids of tiers 0-6)

### 3.1 Closed loop on the 800-group f104 suite (spec)

- H on the Gator drives exactly the routes H picked for the HMMWV (800 / 800 identical picks). The two straight arms
  are identical too. So the cross-vehicle contrasts compare vehicles on identical routes.
- The f104 suite has 9 terrain-feature clusters.

| arm | fail (%) | goal reached (%) | tilt > 30 deg (%) | statuses (blocked / broke through / goal) |
|---|---|---|---|---|
| Gator-trained on the Gator (G) | 34.75 | **65.25** | 2.5 | 264 / 14 / 522 |
| HMMWV-trained on the Gator (H) | 51.9 | 48.1 | 2.6 | 405 / 10 / 385 |
| straight 6 m/s on the Gator | 85.75 | 14.25 | 12.1 | 679 / 7 / 114 |
| HMMWV-trained on the HMMWV (H = M1a) | 5.5 | 94.5 | 1.0 | - |
| straight 6 m/s on the HMMWV | 32.25 | 67.75 | 5.4 | - |

- **Declared primary B, G vs H on the Gator:** -17.1 points.
  - Cluster 90 % [-21.7, -12.5], group 95 % [-20.3, -14.0], p 0.0002 (bootstrap floor); 22 worse / 159 better groups
    (McNemar p 8e-27). Decision: **improves** (declared primary, outside the Holm family).
  - By part of the suite (supplementary): the 200 in-distribution groups 36.0 vs 60.0 %, the other 600 groups 34.3 vs
    49.2 %.
- **On the unsafe label:** -17.0.
- **Time on joint successes:** G takes 0.83 x H's time (17.6 vs 21.8 s).
- **The vehicle itself, on identical routes:**
  - H on the Gator vs H on the HMMWV: +46.4 points [+39.7, +52.4] (372 vs 1 discordant);
  - straight 6 m/s: +53.5 (433 vs 5);
  - even the Gator-trained planner on the Gator fails 29 points more often than the HMMWV planner on the HMMWV.
- **Predicted risk:** H's picks carry a mean predicted risk of 0.2 % and fail 52 % of the time on the Gator (the
  HMMWV model cannot know the vehicle). G's picks carry 18.4 % and fail 34.8 %.

### 3.2 PLAN 3 / 7.7 criteria

| criterion | limit | result | met |
|---|---|---|---|
| (1) G beats straight 6 m/s on the Gator | lower bound of the improvement > 0 | -51.0 points, 90 % [-55.5, -46.1], 9 vs 417 discordant | **yes** |
| (2) G beats H on the Gator, or is within 2 points | - | G better by 17.1 points [12.5, 21.7] | **yes** |
| (3) offline within-group AUC of unsafe on held-out twin groups (NOTES_S1 7.3) | >= 0.95 | 0.952 (holdout model) / 0.957 (deploy) on the 56 val groups; 0.925 on dev fold + val (280 groups; only 96 groups have both outcomes for the Gator) | **val groups only** |
| (4) headroom closed, (straight - planner) / straight on failure | reported, no threshold | G on the Gator 59 % [56, 63]; H on the Gator 40 % [35, 43]; HMMWV planner on the HMMWV 83 % [78, 88] | lower than the HMMWV's |
| collects the same data: validated ids | >= 95 % | 100 % (15,235 / 15,235; tiers 0-6: 8,399 / 8,399) | **yes** |
| crashed / NaN | < 1 % | 0 | **yes** |
| launch-check failures | < 5 % | 0 | **yes** |
| belly-in-soil flag (lowest hull point > 0.05 m under the surface for > 1 s in a row) | <= 10 % of soil drives | collection 8.5 % (9.8 % counting > 1 s in total; designed routes alone 10.1 %, constant 4 / 6 m/s 11.5 / 11.4 %, on-policy routes 6.2 %); evaluation drives 5.5 % (G 2.1 %, H 3.1 %, straight 6 11.1 % over the 800 groups); every flagged drive failed anyway (collection 1,296 / 1,296, evaluation all) | **yes** (overall; close to the limit, and above it on the designed constant-speed routes) |
| wheel-radius sensitivity (calibrated + 0.08 m, pilot, 144 routes, NOTES_E3b1) | change <= 15 points | 93.8 -> 80.6 % failure (-13.2 points) | yes, narrowly |
| H trained on exactly the Gator-validated ids | - | yes (all 8,399 validated; H file = M1 file, sha256 `48a29d16`) | yes |

### 3.3 Collection read-out, all 15,235 Gator ids (the HMMWV `collect_v1` drives of the identical ids as reference)

`scripts/ag_s1_gator_soil_qa.py` (unchanged), run on the login node on the Gator training rows of soil_v3
(`G3/e5/ids/soil_v3_gator_tiers0-12.json`). Output: `e6/analysis/s2b/gator_soil_qa_t0-12.json`.

| | Gator | HMMWV (same ids) |
|---|---|---|
| drives complete / validated | 15,235 / 15,235 | - |
| QA flags (unreadable, shape, NaN, launch, explosion, breakthrough without a stall) | 0 | - |
| goal not reached | **88.2 %** | 68.1 % |
| routes failing for one vehicle only | 3,263 Gator only | 201 HMMWV only |
| statuses | 13,231 blocked, 1,805 goal, 198 broke through, 1 timeout | 8,372 broke through, 4,867 goal, 1,985 blocked, 9 timeout, 2 rollover |
| simulated hours | **141.1** | 91.5 (1.54 x) |
| belly-in-soil flag | 8.5 % | - |

- **Failure by designed speed profile, Gator vs HMMWV on identical routes:**
  - constant 2 m/s: 91.3 vs 86.2 % (2,302 routes);
  - constant 4 m/s: 83.8 vs 56.3 % (2,254);
  - constant 6 m/s: 82.7 vs 33.5 % (2,326);
  - smooth 2-6-2 m/s: 84.2 vs 54.6 % (2,286);
  - planner proposals (on-policy routes): 92.2 vs 83.9 % (6,067).
- **Belly flag by profile:** 9.0 / 11.5 / 11.4 / 8.4 / 6.2 %.
- **Per tier** the Gator failure rate is flat (87.3-89.3 %, tiers 0-12). By split: train 88.0 / 67.5, val 88.4 / 72.8,
  test 90.9 / 73.9 %.
- **Stage 1 (tiers 0-6, 8,399 ids, the data behind G and H):** 100 % validated, 0 / 0, belly flag 8.3 %, 87.9 vs
  67.9 %, 77.7 vs 50.3 simulated hours (unchanged from NOTES_S1).

**Reading.** The Gator can collect the same data: every id was validated, and the physics checks pass. But what it
collects is a much harder task. It stalls with its rear wheels spinning on many routes the HMMWV completes (3,263
routes fail for the Gator only, 201 for the HMMWV only), and its labels are close to saturated: in 65 % of the
scored f104 groups every Gator route of tiers 0-6 fails.

A planner trained on those labels still meets the first two declared "works" criteria (criterion 3 only on the val
groups). It reaches the goal on 65 % of the suite against 48 % for the HMMWV planner on the same vehicle, and 14 % for
the straight route. It closes a smaller share of the headroom than the HMMWV planner does on the HMMWV (59 % vs 83 %).

The result depends on a wheel model that is only just inside its sensitivity rule (-13.2 points at +0.08 m).

## 4. Deviations and declarations

1. **Supplementary analyses (post hoc, after the outcomes were seen).** They are all in
   `e6/analysis/results_soil_v1_supp.json` (`scripts/ag_s2b_supp.py`, unchanged `ag_analyze.py` statistics, bootstrap
   seed + 1):
   - the single-ensemble pairs beyond the two in the spec;
   - the training-noise sensitivity;
   - the QA-clean sensitivity;
   - the gap against the g203 / g228 held-out groups (the spec's gap uses the f104 in-distribution groups only);
   - M3a per held-out arena;
   - extra per-arena regressors;
   - the task B split.

   None of them changes a declared decision. Where the same contrast appears in the spec and the supplement, the table
   quotes the spec value. Monte-Carlo differences are at the third decimal of p.
2. **Decision labels of the frozen tool.**
   - The rule labels every contrast whose whole interval is above 0 "inconclusive", e.g. "the Gator fails more than the
     HMMWV" (the one-sided test asks only "is the test arm better"). Those are described here in words as worse.
   - The rule gives "improves" to secondary contrasts at their unadjusted p. The tables mark them secondary.
3. **Cluster intervals on f104** rest on 9 terrain-feature clusters. The group-bootstrap intervals are given beside
   them where it matters (no-harm, task B).
4. **Tools.**
   - No existing script was edited.
   - New scripts: `scripts/ag_s2b_sync.sh` (sync of the index files into `e6/runs_soil`, then the unchanged index
     tool), `scripts/ag_s2b_provenance.py` (login node), `scripts/ag_s2b_supp.py`.
   - `ag_s2_extras.py` and `ag_s1_gator_soil_qa.py` ran unchanged from `G3/tools/s2b/scripts` (hash-equal copies),
     the first on a cluster copy of the index with the run folders remapped to `G3/soil_v1/runs`
     (`G3/tools/s2b/out/soil_eval_v1_cluster.json`).
   - No job was submitted: login-node reads only.
5. **"M1" in the soil primary** = the per-group mean of the two f104-only ensembles (M1a, M1b), and likewise M3 (as
   in the frozen specs). A3 has one ensemble.
6. **Stage-1 models only.** Every model here uses tiers 0-6. Retraining on the full 15,235 ids (task B stage 2) is a
   separate track and is not reported here.

## 5. Files

- Frozen-spec analysis: `e6/analysis/results_soil_v1.json` (+ `.txt`), spec `e6/analysis/spec_soil_v1.json`
  (`cf813e0a...`).
- Family of four: `e6/analysis/family_v1_S2.json`.
- Supplementary: `e6/analysis/results_soil_v1_supp.json` (+ `.txt`).
- Checks: `e6/analysis/s2b/`:
  - `provenance_v1.json`;
  - `soil_extras_v1.json` (evaluation-drive QA, belly flags, simulated hours per arm);
  - `gator_soil_qa_t0-12.json` (collection read-out, per-id rows);
  - `logs/`.
- Index: `e6/index/soil_eval_v1.json` (run-id lists in `e6/index/sync_soil_eval_v1/`); run files `e6/runs_soil/`
  (index files only).
- Offline read-outs quoted in section 3.2: `e5/offline/soil_s1_B_auc.md`, NOTES_S1 section 7.3.
