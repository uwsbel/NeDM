# REVIEW R1 (design and statistics) of PLAN.md, 2026-09-25 ~01:30

Reviewer R1. Read: PLAN.md (sha in PLAN.sha256), scout S1-S4, LOG.md up to the E1/E2 start lines, the memory notes, and
`arenas/selection.json` (E1 has already picked the 4 test arenas). Nothing was edited except this file. Numbers marked
"measured here" were computed in this review from files in K3 and the earlier read-only roots.

## 0. Verdict

The plan answers both questions in principle. Task A controls the main confound (amount of data against number of
arenas): the matched series M1/M2/M3 and the additive arm A3 are right. Task B pairs every Gator run with an HMMWV run
through the same task ids, which is also right.

Three design choices make a null result for task A close to built in, and the plan has no rule for telling
"inconclusive" apart from "no benefit":
1. The test arenas are the 4 arenas of the new batch that are closest to f104.
2. The test drives do not measure how much results vary between training seeds.
3. The decision rule treats start/goal groups as independent, but failures cluster at terrain features.

For task B, the plan does not define what "can collect the same data" means. It also leaves two representation
choices (the soil wheel radius and the body that is not coupled to the soil) that could decide the soil answer
unnoticed.

Every amendment below fits tonight's budget. Together they add about 12 billed node-hours (most of it optional,
lowest-priority soil drives). Items 1-3, 5, 6, 10 and 11 cost nothing in simulation.

## 1. Numbers measured here

**Test-arena distances** (the plan's 8-statistic distance, recomputed with `scripts/ag_arena_rank.py` functions):

| arena | role | distance to f104 | to g203 | to g228 | nearest training arena |
|---|---|---|---|---|---|
| g260 | test | 0.653 | 0.967 | 0.606 | g228 |
| g271 | test | 0.739 | 1.114 | 1.012 | f104 |
| g251 | test | 0.779 | 1.296 | 1.163 | f104 |
| g247 | test | 0.906 | 1.170 | 0.721 | g228 |
| g203 | training | 0.815 | 0 | 0.750 | |
| g228 | training | 0.811 | 0.750 | 0 | |

- The test arenas are ranks 1-4 of 40. The median distance of the new batch to f104 is 1.42.
- Three of the four test arenas are closer to f104 than either added training arena.
- g203 and g228 were themselves the 2 closest of the previous batch of 40.

**Map lookup error.** This is the flat-ground depth lookup against Chrono's own terrain height, on the 4,096 audit
points of each capture. It is the same method as S1 2.1.

| arena | error (rms) |
|---|---|
| f104 | 0.050 m |
| g203 | 0.062 m |
| g228 | 0.072 m |
| g217 | 0.066 m |
| g260 | 0.060 m |
| g271 | 0.065 m |

**f104 is the least distorted arena.** Every other arena is 20-44 % worse.

**Where the soil failures are on f104** (the soil specialist in the generalist study, CEM, 800 groups, by stratum):

| stratum | failure rate |
|---|---|
| hill cross-slope | 5.8 % |
| hill entry-cross-exit | 7.8 % |
| crater cross-slope | 2.7 % |
| crater entry-cross-exit | 1.0 % |

- The 200 `f104_crm_eval_group` pairs score 98.5 %, against 94.8 % on the 600 fresh pairs.
- Those 200 pairs are the ones on which night 2 chose CEM 4x64.
- The "95.8 %" soil specialist named in PLAN section 1.1 is `crm_f104_v1/train_v1/deploy/CRM_N2_s*.pt`. It was
  trained on standing-start rows only (one row per drive), not with tonight's recipe (re-anchored rows through the
  trainer). Source: `generalist_20260921/A_adapt/suite/picks_crm_Scrm/summary.json`.

**Power at 1,000 paired test groups.** Paired binary outcomes, 80 % power. The design effect of 1.5 (the correction
for failures clustering at terrain features) is the S4 assumption.

| test level | effect needed at a 5 % / 8 % / 10 % / 15 % / 20 % disagreement rate |
|---|---|
| one-sided 0.05, no clustering correction | 1.8 / 2.2 / 2.5 / 3.0 / 3.5 points |
| one-sided 0.025 (Holm first step), design effect 1.5 | 2.4 / 3.1 / 3.4 / 4.2 / 4.8 points |

The disagreement rate is the share of groups where exactly one of the two models fails.

Power at the planned test level (one-sided 0.025) with design effect 1.5:

| true effect | power, disagreement 6 % / 10 % / 15 % |
|---|---|
| 2 points | 0.56 / 0.37 / 0.27 |
| 3 points | 0.89 / 0.69 / 0.52 |

**Why the effect is likely small.** On f104 the specialist fails 5-8 % of hill groups and 1-3 % of crater groups. The
gain M3 can show on unseen arenas is capped by how much worse M1 does there than on f104, which is probably a few
points. So the primary is underpowered for the effects most likely to exist.

**Arena-level sign test.** 4/4 arenas improving gives one-sided p = 0.0625, so it can never reach 0.05. With 8
arenas, 8/8 gives p = 0.004 and 7/8 gives p = 0.035. With a true 3-point effect and 250 groups per arena, all 4
arenas point the same way only about 76 % of the time. The "4/4 needed" rule therefore mostly measures noise.

## 2. Ranked amendments

### 1. Test arenas: draw them from the whole family, and use 8 x 125 instead of 4 x 250 (task A; no extra drives)

**Problem.** The rule "the 4 closest to f104" picked the most f104-like 10 % of the family.
- f104-like test arenas keep M1's shortfall on unseen arenas small, which also caps the gain M3 can show.
- A null result then says little about new arenas in general.
- With 4 test arenas, no arena-level statement can be significant.

**Amendment.**
- Keep g260, g271, g251 and g247 as a "near" stratum.
- Add 4 "spread" arenas by a declared rule, before any case file exists. Take one from each quarter of ranks 5-40
  (5-13, 14-22, 23-31, 32-40). Within each quarter, take the arena with the smallest sha256 of its BMP. Use E1's
  replacement rule.
- Use 125 feature groups per arena. That is still 1,000 groups per world, so the drive count is unchanged.

**Reporting.**
- The primary is pooled over all 8 arenas.
- Secondary: near against spread, and the M3-M1 effect against the distance to the nearest training arena.

**Cost.** 4 more OptiX captures, grids and soil surface smokes (about 10 minutes locally), cases, and 4 more planner
runs.

**Plan sections to change:** 1.3, 2.2.

### 2. Make the decision rule honest about clustering and power; fix one family of tests now (task A; free)

**The contradictions.**
- The plan applies Holm over P1 and P2 but decides on a one-sided 95 % bound. The first Holm step needs a 97.5 %
  bound.
- A group-level McNemar test ignores that failures sit at a few terrain features. Soil night 1 found a pair-level
  p of 3.5e-13 against a clustered p of 0.003-0.008.

**Amendment.**
- Take one-sided p-values from a bootstrap over (arena, nearest feature) clusters. With 8 arenas that is about 90
  clusters.
- Apply Holm at a family-wise 0.05.
- Report the group-level bootstrap and McNemar alongside.
- Declare the family now: soil P1 and P2, and rigid fixed-2 m/s P1 and P2, Holm over all 4. The dev-arena check must
  not change which tests are confirmatory, only the power statement and the order of presentation. The current
  wording ("unless there is no gap at all") is an undefined switch.

**Pre-register a power and equivalence statement** (numbers in section 1):
- A difference is "no meaningful difference" only if the 90 % interval lies inside +-2 points.
- Otherwise it is "inconclusive".
- Report estimates with intervals, not only pass/fail.

**Per-arena reporting.** Replace "4/4 arenas needed" with per-arena estimates and intervals, plus a pooled estimate
that treats arena as random, and the spread between arenas. The sign count is supplementary only.

### 3. Make "can the Gator collect the same data" a pre-declared criterion, and keep G and H on identical ids (task B; free)

**(a) Matched ids.** H must be trained on exactly the ids for which the Gator run validated, in both worlds (the
intersection with the HMMWV ids), wherever the Gator collection stops. Task A's M1, which may be cut at a lower tier,
is not H unless the ids coincide.

**(b) "Can collect" means all of the following** (the plan currently has only pilot go/no-go gates):
- at least 95 % of the ids validated;
- crashed, NaN or exploded drives below 1 %;
- launch-check failures below 5 %;
- informative labels: overall failure between 10 % and 90 %, and inside that band for at least 3 of the 4 designed
  speed profiles;
- simulated hours, rows, and billed cost per simulated hour reported against the HMMWV.

**(c) Body-in-soil flag.** Flag a soil drive when the body's lowest point is more than 0.05 m below the undeformed
surface for more than 1 s. The body is not coupled to the soil and the Gator has 0.14-0.17 m of clearance. If more than
10 % of Gator soil drives are flagged, report the soil answer as "collected, but not physically trustworthy". Also
report G against H with flagged drives counted as failures.

**(c2) Wheel-radius sensitivity.** The calibrated soil wheel radius is a free parameter that directly sets how hard
the soil is for the Gator.
- Run the 24 x 6 soil pilot at the calibrated radius and again at the calibrated radius + 0.08 m (one particle
  spacing). That is about 144 extra drives.
- If the failure rate differs by more than 15 points between the two, report every Gator soil result as dependent on
  the wheel representation. This does not stop the run.

**(d) Headroom rules.**
- If Gator straight 6 m/s on soil reaches the goal in 95 % or more, or 5 % or less, of groups, "works" criterion (1)
  is not applicable. Say so.
- Normalise the "headroom closed" measure with each vehicle's feasibility, not with 1.0. Feasibility = the share of
  groups where any designed route reached the goal, from the collection itself (1,200 groups, free).
- Define "rigid at the ceiling" numerically before the rigid fixed-2 arms are added: G and H both at 99 % or more.

**(e) Rigid labels and determinism.**
- The Gator brakes only its rear wheels and creeps downhill. That produces "slid backwards" unsafe labels with no
  terrain cause.
- On rigid ground, report the Gator's unsafe rate with and without the backward-motion clause, and lead with goal
  reached and tilt.
- Run all rigid arms of a task B group in one array task, as in task A.

### 4. Measure how much results vary between training seeds in closed loop (task A; about 4 billed)

**Problem.** Two competent soil models disagree on 3-8 % of groups. A 2-3 point M3-M1 difference cannot be told apart
from "which five seeds" without a second ensemble per condition driven on the same groups.

**Amendment.**
- Make the M1b and M3b drives (seeds 5-9) on the unseen soil suite mandatory, not "if time allows".
- The primary outcome per condition becomes the mean over the two ensembles (0, 0.5 or 1 per group). Averaging over
  seeds is the right target for "does this training set help".
- Report M1a against M1b and M3a against M3b as the seed-noise floor.
- Rigid fixed-2 copies are nearly free and should be driven too.

**Cost.** At most 2,000 soil drives, fewer after identical picks: about 11 simulated hours, about 4 billed, 0.3-0.6
wall hours.

### 5. Offline read-outs that separate "how many arenas" from "which arena" and from "how much data" (task A; no soil drives)

**(a) Leave one training arena out**, at a matched 545 groups.
- Train 1-arena models on f104, on g203 and on g228 (545 groups each).
- Train 2-arena models on f104+g203, f104+g228 and g203+g228 (272 + 273 groups each).
- Score each model on the standing-start rows of the arena it did not see (all splits). Use within-group AUC and the
  failure rate of the lowest-risk offline pick.
- That is 6 models x 3-5 seeds, minutes each on the MI350X. It answers "1 arena against 2 at equal data, averaged over
  which arena", which the single nested order f104 -> g203 -> g228 cannot.

**(b) f104 learning curve.** Train on 272, 545 and 1,089 groups and score on the same held-out rows. If M1 is not
data-limited on unseen arenas, an A3 gain can be credited to the arenas. Without this, P2 cannot be interpreted.

**(c) Labelled data on unseen arenas.**
- Rigid: drive the 12 designed routes of every unseen test group. That is 12,000 drives, about 1.5 billed, under 1
  hour of CPU wall time, run with the rigid wave before the soil launch because the partitions are exclusive. It gives
  within-group AUC on truly unseen arenas for every model variant, including (a) and (b). It also gives a per-group
  feasibility ceiling for normalising headroom.
- Soil: 25 groups per test arena x 12 routes (2,400 drives, about 14 simulated hours, about 5 billed), as the last
  soil item only.
- The plan's line "offline AUC on unseen arenas is not available" then no longer holds.

### 6. Do not use the CEM-selection groups as the in-distribution reference (task A; swap, no extra drives)

**Problem.** CEM 4x64 was chosen on the 200 `f104_crm_eval_group` pairs, and the specialist scores 98.5 % there
against 94.8 % on fresh pairs. Using them as "f104, in distribution" makes every generalisation gap look larger than
it is.

**Amendment.**
- Use 200 hill/crater groups drawn by hash from the 600 `f104_pair_group`, the same strata as the test suites.
- Compute the gap relative to each arena's straight 6 m/s arm (a difference in differences): the difference between
  arenas in raw rates mixes model loss with arena difficulty.

**Dev check.**
- Run the dev check with tonight's M1. It can be trained now: the f104 data exists and the dev check does not need the
  tier cut. The frozen model is a different recipe.
- Delete the sentence that calls 95.8 % "the reference at this protocol".

### 7. Check integrity before production: map/arena match and data drift since 09-16/17 (both tasks; about 10 soil + 200 rigid drives)

**(a) Map/arena match.** Every dataset build and every pick directory must check that the map root's
`observation.json` BMP hash equals the case arena's BMP hash. S1 found no such check anywhere, so planning an arena
with f104's map fails silently. Put the check in the new wrappers, or in a verifier that runs over every pick
directory and tensor file before drives are launched.

**(b) Data drift.**
- Soil: before production, re-drive 10 f104 soil ids through tonight's exact HMMWV path. Require byte-identical
  trajectories (soil drives are bit-identical across GPU types).
- Rigid: re-drive 200 night-2 rigid ids on one node type. Require at least 95 % outcome agreement (rigid is
  deterministic only per node).
- Otherwise a build or collector change since 09-16/17 would be mixed into "arena" (task A) and "vehicle" (task B).

### 8. Separate the map-distortion effect from the terrain-variety effect (task A; offline only)

**Problem.** f104 has the smallest lookup error (0.050 m against 0.060-0.072 m on every other arena). M1 learns only
the least distorted arena. M3 learns arenas that are as distorted as the test arenas. Part of any M3 gain could be
robustness to the lookup error rather than terrain variety.

**Amendment.**
- Record each arena's lookup error in the arena table, and test the per-arena M3-M1 effect against it.
- In the offline analysis of item 5, build a second set of tensors with metric-grid corridors (exact heights; S1 cites
  a 5e-15 m equivalence of the grid sampler) for f104, g203, g228 and the unseen designed-route rows.
- Retrain M1 and M3 offline and report whether the offline gain survives.
- Closed loop stays on the depth lookup, for continuity.

### 9. Put the test groups where the failures are (task A; free)

**Problem.** Half of a "feature" suite is crater groups, which fail 1-3 % of the time on f104 and add almost no
disagreement. That costs about 25-30 % of the effective sample.

**Amendment.** Before the suites are locked, draw the unseen suites at 3 hill groups to 1 crater group, using a rule
declared on `evaluation_stratum`. Report craters separately.

### 10. Declare the soil drive order now, so a short night keeps the confirmatory contrasts (both tasks; free)

**Order:**
1. Unseen suite M1a, M3a, A3 and straight 6 m/s, together with task B's G, H and straight 6 m/s on the Gator.
2. M1b and M3b.
3. H and straight 6 m/s on the HMMWV (800 groups).
4. M2.
5. In-distribution arms.
6. Dev arena.
7. Soil designed-route subset (item 5c).

**Also fix a fallback that does nothing.** PLAN section 6 says "Gator soil gets priority over soil A3 extra groups",
but soil A3 uses the same 545 g203/g228 groups as M2, so there are no extra groups to give up. The real trade-off is
tiers, which the interleaved task file already handles.

### 11. Minor (free)

- **Matched groups are not matched rows.** Re-anchored rows grow with drive length (bogged drives give more 2 s
  anchors), so arenas and vehicles that bog more contribute more rows and more optimiser steps.
  - Report rows per arena and per model.
  - Add a standing-start-rows-only variant to the offline analysis as a sensitivity check.
- **Diversity disclosure (optional swap).** g203 and g228 are the two most f104-like of 40 (0.81 each; 0.75 from each
  other), so M3 measures the smallest step in variety available.
  - If they are kept, the report must say so.
  - Swapping g228 for a randomly drawn unused sibling before the soil launch costs one capture and one smoke.

### Note on timing: E1 locked the suites at 01:28

E1 locked the suites at 01:28 (`suites/`, ALL_LOCKED c2d0022c5a31..., 4 test arenas x 250 feature groups). No model
exists and no drive has been made, so items 1 and 9 can still be adopted without breaking the lock. Do not regenerate
the locked files. Declare a subset rule on top of them instead.

- Keep the locked 4 x 250 files.
- Add a new locked suite for the 4 spread arenas.
- The primary set is:
  - 125 groups per near arena: all hill groups first, then crater groups taken by sha256(group id) up to a 3:1 ratio;
  - plus 125 per spread arena, drawn the same way.
- Record this rule with its own hash before any pick is made.
- The unused near-arena groups stay available as a reserve and are reported separately if driven.

## 3. Things checked and found sound

- Nested matched subsets with a fixed data seed.
- The tier cut applied to f104 as well.
- The tier-interleaved soil task file.
- Picks hashed and suites locked before any model exists.
- All rigid arms of a group in one array task.
- G against H on the Gator as the task B primary. H uses geometry only, with no vehicle-state input, so the transfer
  arm is not handicapped by out-of-range state inputs.
- H on the HMMWV against H on the Gator uses the same routes. It is a clean vehicle effect at a fixed planner.
- The same ids give route-by-route pairs.
- The HMMWV collector code for the new arenas is the frozen code: `gen_collect.py` states it is byte-for-byte the f104
  collector except the arena gate.
- The soil wheel geometry change applies only to the Gator. It does not bias G against H, which are driven on the same
  vehicle. It does limit cross-vehicle system comparisons, and the report must say so.

## 4. Budget effect of the amendments

| item | extra cost |
|---|---|
| 4: seed-noise floor | about 4 billed |
| 5c: rigid designed routes on test arenas | about 1.5 billed |
| 5c: soil designed-route subset (optional, last) | about 5 billed |
| 3c: wheel-radius sensitivity, 144 pilot drives | about 0.5 billed |
| 7: drift check | under 0.2 billed |
| offline training (items 5 and 8) | under 1 billed |
| **total** | **about 12 billed** |

The plan's estimate becomes about 117-132 billed, inside the 150 soft cap. Added soil wall time is about 0.5-1.5 hours,
most of it in the optional last item.
