# RESULTS planner (Q3): the Polaris planner on f104 soil (2026-09-28)

Milestone of PLAN amendment 9.2 item 2: **"the pipeline works for a new vehicle"**. A planner trained only on the
Polaris's own f104 soil drives plans a route from a standing start, the Polaris drives it, and it must reach the goal
safely on at least 90 % of the 800 f104 suite pairs and of the 600 fresh pairs among them. The answer comes from the
analysis spec frozen before any drive of the gradient arms (`e6/analysis/spec_ov_v1.json`, sha256 3dbd3d79, LOG 00:57),
applied to the stage-2 arms (PLAN 9.1 item 7). The frozen read-out is `e6/analysis/results_ov_v1.txt` / `.json`
(15:10), from the per-pair index `e6/index/soil_eval_ov_final.json`. Re-running `scripts/ov_analyze.py` on that index
with the frozen spec reproduces the `.txt` byte for byte.

Words used below:
- **pair** = one start/goal task of the 800-pair f104 soil suite (Gator study `K3` suites): **200 tuning pairs**
  (`f104_crm_eval_group_*`, the pairs the sampling search's settings were tuned on in earlier studies) and **600 fresh
  pairs** (`f104_pair_group_*`). None of these pairs is in the training data: the dataset builder refuses suite and
  evaluation groups (NOTES_M4). They are still on f104, the arena the data came from.
- **goal reached safely** (the declared bar label, amendment 9.1 item 6) = goal reached, no roll-back on a climb
  (backward time < 0.05 s and lowest forward speed > -0.30 m/s after the 1 s settle), and no **belly flag** (lowest hull
  point more than 0.05 m under the undisturbed soil surface for more than 1 s). **unsafe** = goal not reached or rolled
  back. Stored drives from older studies have no belly record, so for them "safely" means "not unsafe".
- **stall** = stopped for 2 s with throttle applied, after 24 s (the collector's blockage stop). **dug in** = a wheel dug
  through the whole 0.24 m soil layer (breakthrough stop).
- **power-corrected Polaris** = the smoke test's `polaris_pc` arm: Chrono's Polaris with the driveline reduction defect
  fixed (`RESULTS_smoke.md`). The primary Polaris uses the stock driveline, which gives the wheels about 16 times the
  engine's power.
- Arm names in the files: `polaris_grad` / `polaris_cem` = stage-2 planner with / without gradient refinement,
  `polaris_s1_grad` / `polaris_s1_cem` = the same with the stage-1 model, `*_pc` = the same routes driven by the
  power-corrected Polaris, `straight6_polaris` = the straight route at 6 m/s, `Gfull_*_gator` / `Hfull_*` = the Gator /
  HMMWV with their own soil models (Gator study).

![Polaris planner on f104](figures/fig_planner_f104.png)

## 0. Headline

- **Milestone met, clearly above the bar.** The declared planner (stage 2, sampling + gradient refinement) reaches the
  goal safely on **99.8 %** of the 800 pairs (798; 95 % pair interval [99.4, 100.0]) and on **99.8 %** of the 600 fresh
  pairs ([99.5, 100.0]). Valid drives: 0 crashed, 0 launch-check failures, 0 belly flags. Verdict of the frozen tool:
  "meets the bar", "clearly above" (`results_ov_v1.txt`, CRITERIA and bar lines).
- **But f104 is at the ceiling for the Polaris.** The straight route at 6 m/s, with no model at all, already reaches the
  goal safely on **99.5 %** (796 of 800). The planner cannot show a gain here. It fixes all 4 pairs where the straight
  route fails and fails 2 others: -0.2 points, "no meaningful difference" (declared family, section 4). So this result
  shows that **the pipeline runs end to end for a new vehicle and does not break anything**. It does not show that the
  planner adds value on f104 for the Polaris.
- **Every Polaris variant passes.** Sampling only: 100.0 %. Stage-1 model (routes 0-6 only): 99.9 %. The
  power-corrected Polaris on the same refined routes: 99.1 % [98.4, 99.8] (stage 1: 100.0 %).
- **Same pairs, other vehicles with their own planners:** Gator 67.9 %, HMMWV 96.2 % goal reached safely. The Polaris
  planner beats the Gator's by 31.8 points (Holm-adjusted, 0 vs 254 discordant pairs).
- **Gradient refinement changes many routes but no outcome.** It changed 131 of 800 stage-1 picks and 525 of 800
  stage-2 picks. For the Polaris it made no measurable difference (100.0 -> 99.8 %). The Gator went 67.4 -> 68.0 % and
  the HMMWV 95.6 -> 96.2 % goal reached (section 6).
- **The model's risk numbers are far too optimistic in absolute terms.** Mean predicted failure probability of the
  stage-2 picks: 1.4e-5, or 0.01 expected failures over 800 drives. Observed: 2 for the refined routes (0.25 %) and 7
  when the power-corrected Polaris drives them.

## 1. What was trained

| item | stage 1 (interim) | stage 2 (answers the milestone) | source |
|---|---|---|---|
| drives used | routes 0-6 of every start/goal group (collection tiers 0-6): **8,395** episodes of 8,399 (4 rejected: breakthrough without a stall first) | all tiers 0-12: **15,229** valid episodes of 15,235 (6 rejected, same kind) | LOG 05:47, 08:45; `RESULTS_collection.md` 5 |
| decision rows | **31,575** (each drive gives its standing-start row plus rows re-anchored mid-drive) | **57,228** | LOG 05:55, 08:45 |
| rows the deployed model was fitted on | 28,701 (1,089 training groups; 17,387 from designed routes, 11,314 from planner proposals) | 52,021 (31,447 / 20,574) | `e5/deploy/*/…_deploy.json` (`n_fit`, `fit_by_source`) |
| model | CNN-GRU risk model: a small convolutional network reads the height map along the candidate route, a GRU runs along the route, and the route's shape and speed profile go in as context (`ci_train --arch gru --cond none --domain-filter crm --ctx geom`). Output: the probability that the drive ends unsafely. 256,161 parameters per network. **5 networks (seeds) x 30 epochs**, fresh input normalisers | same recipe | PLAN 1.7; `e6/picks/f104/*/summary.json` (members) |
| training data | **only the Polaris's own f104 soil drives** (every vehicle record in both files is `polaris`; 0 rigid-ground rows) | same | `…_deploy.json` (`fit_by_domain`); `RESULTS_collection.md` 5 |
| training job | MI350X job 441823, 9.5 min, both lanes exit 0 | MI350X job 441919, 16.8 min | LOG 06:06, 08:48 |

**Offline ranking score (held-out groups).** A second training lane (the hold-out lane) uses the same recipe but leaves
out 224 training-pool groups ("dev") and scores them, plus the 56 validation groups. The score is the within-pair AUC:
among the collected routes of one start/goal group, how often a failing route is rated riskier than a successful one
(1.0 = always, 0.5 = chance). Source: the cluster records
`G4/e5/train/offline_soil/polaris_{s1,full}_soil_holdout.json` (read-only copy), `ensemble` block, soil rows, label
"goal not reached".

| hold-out lane | fitted rows | standing-start rows: dev / validation | all rows: dev / validation | lowest-rated collected route fails (standing start): dev / validation | a random route fails |
|---|---|---|---|---|---|
| stage 1 | 22,809 | **0.981** (1,568 rows) / **0.989** (392) | 0.980 / 0.995 | 1 of 224 groups / 0 of 56 | 16.5 % / 13.8 % |
| stage 2 | 41,338 | **0.977** (2,847) / **0.981** (712) | 0.981 / 0.978 | 2 of 224 / 0 of 56 | 15.4 % / 14.6 % |

- On routes like the collected ones, the model is also calibrated on average: mean predicted 14.2 % vs observed 14.6 %
  failures (stage 2, validation groups, standing start). Section 7 shows that this breaks down in the far tail, where
  the planner's picks sit.
- The deployed checkpoints score 1.0 on "dev" because deployment trains on those groups. Only the hold-out lane is
  quoted.

## 2. How it plans (the declared default, PLAN 1.7)

From a standing start at the pair's start pose, with the arena's overhead height map:
1. **Sampling search (cross-entropy method, "CEM 4 x 64").** Candidate routes come from the planner's route family:
   the straight start-goal line bent sideways by up to 3 sine terms, with 4 speed knots on a 2 m/s base profile
   (0-6 m/s). There are 4 rounds of 64 candidates, and each round re-centres on the best 15 %. The score is the 5-network
   mean of the predicted risk. Arm `polaris_cem` stops here.
2. **Gradient refinement (`ci_grad`, defaults).** It starts from the CEM pick plus the 16 best routes of the first
   round and takes up to 60 gradient steps on the predicted risk. Penalties stop the route from curving too sharply or
   leaving the arena. The planner replaces the CEM pick only if the most pessimistic of the 5 networks rates the refined
   route at least 0.3 better on the logit scale. Otherwise it **abstains** and keeps the CEM pick. Arm `polaris_grad`.
3. The route is driven once by the frozen soil collector and route follower. The picks are locked by sha256 before any
   drive: stage 1 CEM 57801d6c, gradient 42709c24; stage 2 CEM 54902c81, gradient 1f419a03. Every lock passed a re-plan
   check (identical routes), and the CEM stage of the gradient run matched the locked CEM picks on 800/800 (LOG 06:19,
   06:51, 09:09, 14:20; `e6/picks/f104/polaris_*_grad/summary.json` `ref_b`).

Planning cost on the local GPU: about 2.3 s per pair for stage 1 and 2.6 s for stage 2 (CEM 0.40 s + refinement 2.11 s),
31 and 35 min for 800 pairs (`summary.json` `seconds_per_group`, `wall_s_total`). One stage-2 run hung on one pair while
the GPU was shared with other jobs. It was re-run, and the locked picks come from the re-run (LOG 13:43, 14:20).

## 3. Results on the 800 pairs and the 600 fresh pairs

Stage 2 = the frozen read-out (`e6/analysis/results_ov_v1.txt`, `rates` / `bar` / `validity` blocks of the `.json`).
Stage 1 = the same frozen spec and tool run for this report on `e6/index/soil_eval_polaris_s1.json`, with the stage-1
arm names mapped onto the spec's names (`polaris_s1_grad` -> `polaris_grad` etc.). The Gator / HMMWV gradient drives
come from the final index, and the stored Gator-study drives were added with `--reuse`, as in the frozen run. This
output is not stored in K4. Its numbers match LOG 08:25 (99.9 % [99.6, 100.0], sampling only 100.0 %, straight 99.5 %,
power-corrected 100.0 %, straight vs planner 4 vs 1 discordant). Intervals: 95 % bootstrap over pairs (4,000
resamples).

| arm (all on the same 800 pairs) | goal reached safely, 800 [95 %] | goal reached safely, 600 fresh [95 %] | goal reached, 800 | unsafe, 800 | belly flags | crashed / launch-check failures | median time to goal |
|---|---|---|---|---|---|---|---|
| **Polaris, stage 2: sampling + gradient (declared)** | **99.8 %** [99.4, 100.0] | **99.8 %** [99.5, 100.0] | 99.75 % (798) | 0.25 % | 0 / 800 | 0 / 0 | 16.9 s |
| Polaris, stage 2: sampling only | 100.0 % [100.0, 100.0] | 100.0 % | 100.0 % | 0.0 % | 0 / 800 | 0 / 0 | 15.7 s |
| Polaris (power-corrected), stage-2 refined routes | 99.1 % [98.4, 99.8] | 99.2 % [98.3, 99.8] | 99.1 % (793) | 0.9 % | 1 / 800 | 0 / 0 | 16.7 s |
| Polaris, stage 1: sampling + gradient | 99.9 % [99.6, 100.0] | 99.8 % [99.5, 100.0] | 99.9 % (799) | 0.1 % | 0 / 800 | 0 / 0 | 14.9 s |
| Polaris, stage 1: sampling only | 100.0 % | 100.0 % | 100.0 % | 0.0 % | 0 / 800 | 0 / 0 | 15.1 s |
| Polaris (power-corrected), stage-1 refined routes | 100.0 % | 100.0 % | 100.0 % | 0.0 % | 0 / 800 | 0 / 0 | 14.8 s |
| **Polaris, straight route 6 m/s (no model)** | **99.5 %** [99.0, 99.9] | 99.7 % [99.2, 100.0] | 99.5 % (796) | 0.5 % | 0 / 800 | 0 / 0 | 7.9 s |
| Gator, own model: sampling + gradient | 67.9 % [64.6, 71.2] | 68.0 % [64.0, 71.7] | 68.0 % (544) | 32.1 % | 10 / 800 | 0 / 0 | 18.0 s |
| Gator, own model: sampling only (stored) | 67.0 % [63.6, 70.4] | 66.3 % [62.5, 70.0] | 67.4 % (539) | 33.0 % | no record | stored | 18.9 s |
| Gator, straight route 6 m/s (stored) | 14.2 % [11.9, 16.8] * | 14.3 % | 14.2 % (114) | 85.8 % | no record | stored | 17.0 s |
| HMMWV, own model: sampling + gradient | 96.2 % [94.9, 97.5] | 95.7 % [94.0, 97.3] | 96.2 % (770) | 3.8 % | no record | 0 / 0 | 15.8 s |
| HMMWV, own model: sampling only (stored) | 95.6 % [94.1, 97.0] | 95.5 % [93.7, 97.0] | 95.6 % (765) | 4.4 % | no record | stored | 17.1 s |

\* The frozen spec has no bar block for the Gator's straight route. This interval was computed for this report with the
same pair bootstrap.

- End states of the Polaris failures (`rates.status_counts`): declared planner 1 left the arena and 1 rolled over.
  Straight route: 1 timeout, 1 rollover, 1 arena exit, 1 stall. Power-corrected: 5 stalls and 2 dug in. The Gator
  mostly stalls (241 of 256 failures with its gradient planner).
- Median time is over drives that reach the goal. The planner's routes average about 3.1 m/s (`summary.json`
  `mean_speed`), so they take about twice as long as the straight 6 m/s route for the same success.
- The 704 pairs outside smoke sample B (whose straight-route outcomes were seen before the picks were locked): 99.7 %
  [99.3, 100.0] (`results_ov_v1.txt`, `f104_800_notB`).
- Validity (PLAN 4.3(c), `results_ov_v1.txt` validity lines): every arm has all 800 drives, all QA-checked, 0 crashed or
  non-finite, 0 launch-check failures, belly flags <= 10 %. With belly-flagged drives counted as failures, the bar
  still holds (99.8 / 99.8 %). The HMMWV drives carry no belly record, so the spec waives its belly check.

## 4. The declared family of 4 tests (Holm at 0.05)

Quoted exactly from `e6/analysis/results_ov_v1.txt`. "fail" = goal not reached, so a negative diff favours the first
arm. The Holm correction runs over the one-sided 9-cluster bootstrap p. Clusters group the pairs by the nearest terrain
feature. "No meaningful difference" = the 90 % cluster interval lies within +-2 points.

```
FAMILY F_polaris_grad_vs_straight6 (crm/polaris, fail, f104_800): polaris_grad 0.2 % vs straight6_polaris 0.5 %, diff -0.2 pts, cluster 90 % [-0.5, +0.0] (9 clusters), p1 0.0947, Holm adj 0.2842 -> no meaningful difference; group 95 % [-0.9, +0.3]; McNemar worse/better 2/4; n 800
FAMILY F_polaris_grad_vs_gator_grad (crm/any, fail, f104_800): polaris_grad 0.2 % vs Gfull_grad_gator 32.0 %, diff -31.8 pts, cluster 90 % [-40.7, -23.9] (9 clusters), p1 0.0002, Holm adj 0.0010 -> improves; group 95 % [-35.1, -28.5]; McNemar worse/better 0/254; n 800
FAMILY F_polaris_grad_vs_cem (crm/polaris, fail, f104_800): polaris_grad 0.2 % vs polaris_cem 0.0 %, diff +0.2 pts, cluster 90 % [+0.0, +0.8] (9 clusters), p1 1.0000, Holm adj 1.0000 -> no meaningful difference; group 95 % [+0.0, +0.6]; McNemar worse/better 2/0; n 800
FAMILY F_gator_grad_vs_gator_cem (crm/gator, fail, f104_800): Gfull_grad_gator 32.0 % vs Gfull_free_gator 32.6 %, diff -0.6 pts, cluster 90 % [-1.6, +0.5] (9 clusters), p1 0.2009, Holm adj 0.4019 -> no meaningful difference; group 95 % [-2.8, +1.5]; McNemar worse/better 37/42; n 800
```

In words:
1. **Planner vs the straight route (same Polaris): no meaningful difference.** This is decided by the ceiling
   (section 5). With the straight route at 0.5 % failures, a gain can be at most 0.5 points, well inside the +-2 point
   margin.
2. **Polaris planner vs the Gator's own gradient planner: improves**, by 31.8 points. The Gator reaches the goal on none
   of the pairs where the Polaris fails, and fails 254 pairs the Polaris reaches. This compares vehicles, not planners.
3. **Gradient refinement vs sampling only (Polaris): no meaningful difference.** The refined routes are the ones that
   failed (2 vs 0).
4. **The same for the Gator: no meaningful difference** (-0.6 points).

## 5. Verdict against the bar, and the ceiling

Quoted from `results_ov_v1.txt`:

```
bar polaris/polaris_grad (unsafe_belly): f104_800 99.8 % [99.4, 100.0] n 800; f104_600 99.8 % [99.5, 100.0] n 600 -> meets True, clearly above True
CRITERIA polaris/polaris_grad (unsafe_belly): (a) beats straight False (no meaningful difference), (b) bar True {'f104_800': 99.8, 'f104_600': 99.8} clearly above True, (c) valid True -> meets the bar (rule r1; PLAN: does not beat the straight route; R1-S3: meets the bar); the straight route already reaches 99.5 % of the 800 pairs: the bar is met by the straight route if the planner arm is not better; report the planner gain with its interval
```

- **"Meets the bar": yes.** 99.8 % on the 800 pairs and 99.8 % on the 600 fresh pairs, both >= 90 %.
- **"Clearly above": yes.** The rule in force (PLAN 9.1 item 8, spec option `cluster95`) uses the one-sided 95 % lower
  bound of the 9-cluster bootstrap: 99.24 % (800) and 99.50 % (600), both >= 90. The pair-level two-sided lower bounds
  (99.4 / 99.5 %) also clear it. Caveat: the spec's free-text note still describes the pair-level rule. Both rules give
  the same answer.
- **Rule in force.** Amendment 9.1 item 6 (adopted before any drive) answers Q3 with (b) the bar and (c) validity only,
  and reports (a) "beats the straight route" separately. Under the original PLAN 4.3 wording, (a) was part of "the
  planner works", and the verdict would read **"does not beat the straight route"**. The tool prints both.

**The ceiling, pair by pair** (`e6/index/soil_eval_ov_final.json` and `soil_eval_polaris_s1.json`; figure panel c):
- **The straight route fails on 4 of 800 pairs**, all where the line crosses a slope:
  - pair 0137 (tuning set, crater side slope): timeout at 120 s;
  - pair 0147 (tuning set, crater in-across-out): rolled over at 7.4 s, 58 deg tilt;
  - pair 0558 (fresh, crater side slope): left the arena;
  - pair 0579 (fresh, hill in-across-out): stalled. This is the one sample-B failure the smoke test had already seen.
- **Every Polaris planner arm reaches the goal safely on all 4**: stage 1 and 2, with and without refinement, and both
  drivelines. For comparison, the Gator's own planner fails 3 of the 4, the HMMWV's gradient planner none, and its
  sampling-only planner 1.
- **Where the planner itself fails.** The stage-2 refined routes fail on 2 pairs, both hill side slopes where the
  straight route and the sampling-only pick succeed: pair 0193 left the arena after 39 s, and pair 0423 rolled over at
  5.9 s (57 deg). The stage-1 refined routes fail once (pair 0127, dug in). Net stage 2: 4 fixed - 2 broken = +2 pairs
  (+0.25 points), which is noise at n = 800 (McNemar 2 / 4).
- **Where anything failed.** 14 of the 800 pairs had a failure for any Polaris arm. 13 of them are side-slope or
  crossing tasks (7 crater side slope, 4 hill side slope, 1 each crater and hill in-across-out), and 12 lie in 2 of the 9
  terrain clusters (f104:3: 7 pairs, f104:7: 5 pairs).
- **What this means.** On f104 soil the Polaris is strong enough that driving straight at 6 m/s nearly always works,
  so this suite cannot tell a good planner from none. The unseen-arena test (PLAN 9.2 item 3, below) is where the
  planner can show value. The collection already showed where the Polaris fails: slow planner-proposal routes (74 %
  failures at 0.5-1.5 m/s, `RESULTS_collection.md` 3), a route type the planner avoids.

## 6. The power-corrected Polaris on the same routes, and what gradient refinement changed

**Power-corrected Polaris** (PLAN 9.1 item 4, secondary bar; spec `extra_bars`). The model was trained on
stock-driveline drives only. These drives check that the result does not rest on the driveline defect.
- Stage-2 refined routes: 99.1 % [98.4, 99.8] on 800, 99.2 % on the 600 fresh pairs, "meets the bar, clearly above"
  (cluster one-sided lower bound 98.49 %). 0 crashes, 0 launch failures, 1 belly flag, on pair 0499, a stall
  (`results_ov_v1.txt`).
- Stage-1 refined routes: 100.0 % (the stage-1 run described in section 3).
- Paired with the stock Polaris on the identical stage-2 routes: 7 pairs fail only with the power-corrected driveline
  (5 stalls, 2 dug in, all on side slopes or the rough patch). 2 pairs fail only with the stock driveline. Exact
  two-sided McNemar p 0.18 (computed for this report). One possible reason: the power-corrected driveline puts less
  power at the wheels on a slow climb. This was not tested.

**Gradient refinement** (`e6/picks/f104/*_grad/summary.json` `per_arm.G`; the rates from `results_ov_v1.txt`):

| model | picks changed / kept (abstained) | predicted failure, mean: sampling pick -> refined | goal reached: sampling only -> + gradient (pairs worse vs better with refinement) | declared read-out |
|---|---|---|---|---|
| Polaris, stage 1 | **131 / 669** | 7.4e-5 -> 7.6e-5 | 100.0 -> 99.9 % (1 vs 0 discordant) | not in the family (interim) |
| Polaris, stage 2 | **525 / 275** | 1.42e-5 -> 1.36e-5 | 100.0 -> 99.75 % (2 vs 0) | no meaningful difference (+0.2, family test 3) |
| Gator (Gator study model) | 734 / 66 | 0.181 -> 0.114 | **67.4 -> 68.0 %** (37 vs 42) | no meaningful difference (-0.6 [-1.6, +0.5], family test 4) |
| HMMWV (Gator study model) | 643 / 157 | 0.0045 -> 0.00014 | **95.6 -> 96.2 %** (16 vs 21) | inconclusive (-0.6, 90 % [-2.3, +0.8], secondary contrast) |

- **No measurable difference for any vehicle, with two caveats.** The HMMWV's interval reaches just past -2, so the
  declared rule reads "inconclusive" rather than "no meaningful difference". On the 600 fresh pairs alone, the Gator's
  secondary, unadjusted contrast reads "improves" (-1.5 points, 90 % [-2.6, -0.2], p 0.036). The declared 800-pair test
  says no meaningful difference.
- **For the Polaris the refinement had nothing to work with.** The model already rates the sampling picks at about
  1e-5. The acceptance step (0.3 better on the logit scale for the most pessimistic network) then means a change of a
  few in a million in predicted risk. Still, the refined stage-2 routes differ in ways that matter physically:
  - on the 525 changed pairs where both routes reach the goal, they take a median 1.12 times as long as the sampling
    pick;
  - they tip past 30 deg more often (86 vs 62 of the 525 changed pairs; over all 800 pairs 14.4 % vs 11.4 %,
    `results_ov_v1.txt` `tilt30`);
  - both failures are on changed routes.

  The model has no tilt term, and nothing in its scores flags these routes.
- Time ratio (`results_ov_v1.json` `time_ratio`): the median paired time ratio for the Polaris is 1.00 (798 pairs), and
  the declared time rule passes. Refined picks were faster for the Gator (0.95) and the HMMWV (0.98).

## 7. The model's predicted risk vs what happened

Predicted = the 5-network mean probability of an unsafe drive for the route that was driven (index field `P`, equal to
the planner's `summary.json` `P_mean`). The "expected" column is the sum of these probabilities over the 800 drives.
Failure counts are goal not reached, from the index files. Interval: 95 % Wilson interval, computed for this report.

| picks | predicted failure, mean (median) | expected failures in 800 | observed failures in 800 |
|---|---|---|---|
| Polaris stage 2, sampling only | 1.4e-5 (1.2e-5) | 0.01 | 0 |
| Polaris stage 2, refined | 1.4e-5 (1.2e-5; max 4.8e-5) | 0.01 | **2** (0.25 %, 95 % [0.07, 0.91] %) |
| same routes, power-corrected Polaris | same | 0.01 | 7 (0.9 %) |
| Polaris stage 1, sampling only / refined | 7.4e-5 / 7.6e-5 | 0.06 | 0 / 1 |
| Gator, refined | 0.114 (6.0e-4) | 91 | 256 (32.0 %) |
| HMMWV, refined | 1.4e-4 (5.4e-6) | 0.1 | 30 (3.8 %) |

- The Polaris model rates every pick as essentially certain to succeed. Observed failures of the refined picks run about
  180 times the predicted rate (0.25 % vs 0.0014 %), though both are tiny. 800 drives cannot measure a rate near 1e-5.
  The point is that the model's absolute numbers in this tail carry no information.
- The same optimism shows for the vehicles that fail often: the Gator model predicts 11 % and gets 32 %, the HMMWV model
  predicts 0.01 % and gets 3.8 %. The planner uses the model to rank routes, and the rankings are good (section 1).
  **The predicted probabilities of picked routes should not be read as success rates.**

## 8. Caveats (beside the claims above)

- **Same arena.** The 800 pairs are new start/goal tasks on f104, the arena all training drives came from. This shows
  in-arena generalisation only. The unseen-arena test is below.
- **Ceiling.** The straight route reaches 99.5 %, so "the planner works" here means "the pipeline runs for a new
  vehicle and keeps the vehicle's own near-ceiling success". It does not show that the planner helps (section 5).
- **Vehicle model** (`RESULTS_smoke.md`, `RESULTS_collection.md` 7). The stock driveline gives the wheels about 16 times
  the engine's power. The power-corrected Polaris still meets the bar on the same routes (99.1 %), but it was never used
  for training. The soil wheels are calibrated 0.25 m cylinders. The body is not coupled to the soil. The soil is a thin
  0.24 m layer over a rigid floor, which favours light vehicles.
- **Comparison vehicles.** The Gator and HMMWV sampling-only and Gator straight-route rows are stored drives from the
  Gator study (other nodes and dates). Their reuse was justified by the 144/144 identical end states of the Gator
  re-drive and 223/223 for unchanged gradient picks (`RESULTS_collection.md` 6.1). Their models were trained on their
  own vehicles' drives of the same 15,235 routes.
- **Stage 1 is interim** (PLAN 9.1 item 7). Its table rows come from this report's run of the frozen tool on the
  stage-1 index, not from a frozen K4 output file.

## Unseen arenas

Milestone of PLAN amendment 9.2 item 3: **"the planner generalises to arenas it never trained on"**. The stage-2
Polaris planner, trained only on the Polaris's f104 soil drives, plans from a standing start on the 8 test arenas of
the Gator study. It uses each arena's own overhead height map, and the Polaris drives the route once. The bar: goal
reached safely on at least 90 % of the 1,000 declared soil pairs pooled ("meets"). "Clearly above" needs the lower end
of the 95 % interval over the 8 arenas to be at least 90 % too.

The read-out spec was frozen at 14:03, before any drive of these arms (`e6/analysis/spec_unseen_v1.json`, sha256
bdc2a738). The analysis tool `scripts/ov_unseen_analyze.py` (sha256 ce15dc28) was frozen before the full drives. The
frozen read-out is `e6/analysis/results_unseen_v1.txt` / `.json` (16:05), from the per-pair index
`e6/index/unseen_polaris_v1.json`. The build was checked by an independent verifier before launch (`VERIFY_U1.md`: PASS).

**All of this is on deformable soil (CRM), like every other Polaris planner number in this file.** No rigid-ground
Polaris data exists.

![Polaris planner on the 8 unseen arenas](figures/fig_unseen.png)

### U0. Headline

- **Milestone met, clearly above the bar.** The Polaris planner reaches the goal safely on **99.7 %** of the 1,000
  pairs (997). The 95 % interval over arenas is **[99.4, 99.9]**. Every arena is at 99.2-100 %, and every arena's
  Wilson interval lies above 90 %. Verdict of the frozen tool: "clearly above the bar" (`results_unseen_v1.txt`,
  VERDICT line).
- **Valid drives:** 1,000/1,000 driven, 0 crashes, 0 collector failures, 0 launch-check failures, 0 belly flags.
- **Sampling search only (no gradient refinement): 100.0 %** (1,000/1,000).
- **The straight route at 6 m/s, no model: 97.0 %.** This measures the Polaris's mobility: it drives straight across
  unseen soil terrain and reaches the goal safely on 97 of 100 pairs. The HMMWV manages 60.0 % on the same straight
  routes.
- **Context on the same 1,000 pairs** (HMMWV planners of the Gator study, not a test): trained on f104 only 87.9 %
  goal reached safely (87.95 % goal only); trained on three arenas 90.3 %. All three reproduce the Gator study's stored
  rates exactly.

### U1. What was run

| item | value | source |
|---|---|---|
| model | the stage-2 Polaris ensemble (5 networks, f104 soil only, all tiers, 52,021 fitted rows), unchanged from the f104 milestone | spec `model`; `VERIFY_U1.md` 2 |
| arenas | 4 near (g260, g271, g251, g247: terrain distance to f104 0.65-0.91) and 4 spread (g258, g268, g263, g241: 1.11-1.79). Chosen by the Gator study before anyone looked at them; none is in any Polaris training data | PLAN 9.2 item 3; `arena_gator_20260925/REPORT.md` 1 |
| pairs | the Gator study's declared soil subset: 125 pairs per arena, 1,000 in total. By task type: crater side slope 333, hill side slope 323, hill in-across-out 178, crater in-across-out 166 | spec `subset`; index `stratum` (counted for this report) |
| map | each arena's own overhead map, as in the Gator study (map error 0.06-0.09 m RMS) | `NOTES_U1.md` 2; index `map_err_rmse_m` |
| arms | planner = sampling search (4 rounds x 64 routes) + gradient refinement, the declared default; planner without refinement (sampling only); straight route at 6 m/s (the Gator study's straight routes, identical to the HMMWV's on 1,000/1,000) | spec `arms`; `VERIFY_U1.md` 3 |
| refinement | changed 620 picks and kept 380 (abstained). Mean predicted failure about 1.3e-5 before and after | `NOTES_U1.md` 2 |
| picks | locked per arena and as a set (`e6/picks/LOCK_unseen_v1.sha256`) before any drive; re-plan checks identical, also in the verifier's own re-plan (24/24) | `NOTES_U1.md` 2; `VERIFY_U1.md` 2 |
| drives | 2,618 new Polaris drives (8 in the pilot, the other 2,610 in the full launch): 380 abstentions share the sampling-only drive, and 2 straight routes equal a planner route. Pilot of 16 rows first (job 442088): the 8 HMMWV reference drives reproduced their stored end state 8/8 (identical arrays 7/8), and the Polaris reached the goal safely 8/8 | `NOTES_U1.md` 3, 5 |
| launch | 15:21, three job arrays: 442096 (mi2104x x 8), 442104 (mi3501x x 4) and 442108 (mi2104x x 4), i.e. mi2104x x 12 + mi3501x x 4 (LOG 15:21 lists only the first two). 3.4 billed node-hours with the pilot (pilot 0.03 + 2.06 + 0.32 + 1.01); all drives done by 16:02 | LOG 15:21; `sacct` (the cluster's job accounting) |

### U2. Pooled result and the bar

| arm | goal reached safely | Wilson 95 % | pair bootstrap | cluster bootstrap (80 clusters) | over the 8 arenas | t interval over arena rates | arenas at or above 90 % |
|---|---|---|---|---|---|---|---|
| **Polaris planner (declared)** | **99.7 %** (997) | [99.1, 99.9] | [99.3, 100.0] | [99.3, 100.0] | **[99.4, 99.9]** | [99.4, 100.0] | 8/8 |
| Polaris planner, sampling only | 100.0 % (1,000) | [99.6, 100.0] | [100.0, 100.0] | [100.0, 100.0] | [100.0, 100.0] | [100.0, 100.0] | 8/8 |
| Polaris, straight route 6 m/s | 97.0 % (970) | [95.7, 97.9] | [95.9, 98.0] | [95.0, 98.5] | [95.4, 98.5] | [95.0, 99.0] | 8/8 |

Source: `results_unseen_v1.txt`, `arm ... unsafe_belly` lines. Clusters = (arena, nearest terrain feature), as in the
Gator study.

- **Goal only** (ignoring the roll-back and belly checks) gives the same numbers: 99.7 / 100.0 / 97.0 %. No drive of
  any arm reached the goal and was then counted unsafe.
- **Near vs spread arenas:** planner 99.8 % [99.4, 100.0] near and 99.6 % [99.2, 100.0] spread; straight route 96.2 %
  [94.2, 98.4] near and 97.8 % [95.6, 99.6] spread (all intervals over arenas, `results_unseen_v1.json`
  `near_spread.*.arena_boot95`). No sign of a drop on the arenas least like f104. Note: the near/spread lines of
  `results_unseen_v1.txt` print the cluster-bootstrap interval instead (straight route [92.9, 98.7] near and
  [96.1, 99.2] spread; planner spread [99.0, 100.0]); for the planner near it is the same [99.4, 100.0].
- **Verdict:** meets the bar (99.7 >= 90) and clearly above it (arena lower bound 99.4 >= 90). The goal-only verdict
  is the same.

### U3. Per arena

Goal reached safely, % of the arena's 125 pairs. The Polaris columns are from `results_unseen_v1.json` (`arms`); the
HMMWV columns from its `context` block (the Gator study's drives). The HMMWV planners are each the per-pair mean of two
independently trained ensembles, so their counts can be half-integers.

| arena | kind | terrain distance to f104 | **Polaris planner** [Wilson 95 %] | Polaris, sampling only | Polaris, straight 6 m/s [Wilson 95 %] | HMMWV planner, f104 only | HMMWV planner, three arenas | HMMWV, straight 6 m/s |
|---|---|---|---|---|---|---|---|---|
| g260 | near | 0.65 | **100.0** [97.0, 100.0] | 100.0 | 93.6 [87.9, 96.7] * | 94.4 | 96.4 | 68.0 |
| g271 | near | 0.74 | **100.0** [97.0, 100.0] | 100.0 | 99.2 [95.6, 99.9] | 88.8 | 89.6 | 59.2 |
| g251 | near | 0.78 | **99.2** [95.6, 99.9] | 100.0 | 96.0 [91.0, 98.3] | 83.2 | 87.2 | 56.0 |
| g247 | near | 0.91 | **100.0** [97.0, 100.0] | 100.0 | 96.0 [91.0, 98.3] | 87.6 | 89.6 | 44.0 |
| g258 | spread | 1.11 | **100.0** [97.0, 100.0] | 100.0 | 100.0 [97.0, 100.0] | 87.6 | 87.2 | 75.2 |
| g268 | spread | 1.32 | **100.0** [97.0, 100.0] | 100.0 | 94.4 [88.9, 97.3] * | 86.8 | 91.2 | 52.8 |
| g263 | spread | 1.46 | **99.2** [95.6, 99.9] | 100.0 | 99.2 [95.6, 99.9] | 92.8 | 94.4 | 64.0 |
| g241 | spread | 1.79 | **99.2** [95.6, 99.9] | 100.0 | 97.6 [93.2, 99.2] | 82.0 | 86.8 | 60.8 |
| **all 1,000** | | | **99.7** | 100.0 | 97.0 | 87.9 | 90.3 | 60.0 |
| near 500 / spread 500 | | | 99.8 / 99.6 | 100.0 / 100.0 | 96.2 / 97.8 | 88.5 / 87.3 | 90.7 / 89.9 | 56.8 / 63.2 |

\* The Wilson interval reaches below 90 % ("not distinguishable from 90" in the tool). Every other Polaris cell is
"above". Terrain distance = the Gator study's 8-statistic distance to f104 (index `dist_f104`). Arenas at or above
90 % (point estimate): HMMWV f104 only 2/8, three arenas 3/8, straight 0/8 (`arena_rate_min` / `arenas_point_ge_bar`
in the context block).

### U4. Validity (PLAN 4.3(c), spec `validity`)

| arm | driven | crashed / collector failures | launch-check failures | belly flags | QA flags | valid |
|---|---|---|---|---|---|---|
| Polaris planner | 1,000/1,000 | 0 / 0 | 0 | 0 | none | yes |
| sampling only | 1,000/1,000 | 0 / 0 | 0 | 0 | none | yes |
| straight 6 m/s | 1,000/1,000 | 0 / 0 | 0 | 1 (0.1 %; g268 pair 0128, a rollover) | 2 "dug through without a stall first" (g260 pairs 0025 and 0085), both counted as failures | yes |

Source: `results_unseen_v1.txt` validity lines; the flagged pairs from `e6/index/unseen_polaris_v1.json`.

### U5. Declared tests (Holm at 0.05, one-sided cluster bootstrap; exact McNemar beside)

Quoted from `results_unseen_v1.txt`. A negative difference means fewer failures for the first arm.

```
FAMILY U1_grad_vs_straight6 (unsafe_belly): polaris_u_grad 99.7 % vs straight6_polaris_u 97.0 % reached; failure diff -2.7 pts, cluster 95 % [-4.6, -1.2] (80 clusters), p1 0.0002, Holm 0.0005 -> improves; McNemar only-test-fails / only-ref-fails 3/30 (two-sided p 1.4011748135089874e-06, one-sided 7.005874067544937e-07); n 1000
FAMILY U2_grad_vs_cem (unsafe_belly): polaris_u_grad 99.7 % vs polaris_u_cem 100.0 % reached; failure diff +0.3 pts, cluster 95 % [+0.0, +0.7] (80 clusters), p1 1.0000, Holm 1.0000 -> no meaningful difference; McNemar only-test-fails / only-ref-fails 3/0 (two-sided p 0.25, one-sided 1.0); n 1000
```

1. **Planner vs the straight route (same Polaris):** 2.7 points fewer failures (30 pairs only the straight route
   fails, 3 only the planner). The straight route itself is at 97.0 %, so both are high. Following the user (09-28),
   the straight route is read mainly as a measure of the Polaris's mobility.
2. **Gradient refinement vs sampling only: no meaningful difference** (+0.3 points; the 3 planner failures are all on
   routes the refinement changed).
3. Secondary, unadjusted: the same two on goal only (-2.7 and +0.3, same verdicts); sampling only vs straight -3.0
   points [-4.9, -1.5] (0 / 30 discordant).

### U6. Where drives failed (from `e6/index/unseen_polaris_v1.json`, counted for this report)

- **Planner, 3 failures, all side slopes:** g241 pair 0069 (hill side slope, stalled at 34 s), g251 pair 0192
  (crater side slope, stalled at 69 s) and g263 pair 0017 (hill side slope, timeout at 120 s). On all three the refinement changed the
  route, and the sampling-only route and the straight route both reach the goal.
- **Straight route, 30 failures, all on crater or hill crossings:** 24 crater (12 side slope, 12 in-across-out) and 6
  hill (4 side slope, 2 in-across-out). By end state: 10 stalls, 7 arena exits, 5 dug in, 4 timeouts, 4 rollovers.
  8 of the 30 are on g260 (5 of them dug in), 7 on g268 (3 rollovers).
- **Time and tilt.** Median time to the goal: planner 15.0 s, sampling only 14.2 s, straight route 7.6 s (it drives at
  6 m/s; the planner's routes average about 3.1-3.5 m/s, `NOTES_U1.md` 2). Drives tilting past 30 deg: planner 60,
  sampling only 50, straight route 121 of 1,000.

### U7. HMMWV context on the same 1,000 pairs (not tests: another vehicle)

From the `context` and `cross_vehicle_context` blocks of `results_unseen_v1.json`. These are the Gator study's HMMWV
planners, sampling search only (no gradient refinement), standing start. The HMMWV drives carry no belly record, so
"safely" there means "not unsafe".

| HMMWV planner (Gator study) | goal reached safely | 95 % over arenas | Polaris planner minus it, failure points [95 % cluster] | pairs only the Polaris fails / only the HMMWV fails |
|---|---|---|---|---|
| trained on f104 only (mean of 2 ensembles) | 87.9 % (goal only 87.95 %) | [85.3, 90.7] | vs its first ensemble (88.4 %): -11.3 [-14.2, -8.4] | 2 / 115 |
| trained on three arenas, same total data (mean of 2) | 90.3 % | [88.2, 92.7] | vs its first ensemble (90.3 %): -9.4 [-12.9, -6.2] | 2 / 96 |
| trained on three arenas, all data | 90.6 % | [88.3, 93.0] | -9.1 [-12.3, -6.2] | 3 / 94 |
| straight route 6 m/s | 60.0 % | [53.9, 66.0] | -39.7 [-46.7, -33.1] | 1 / 398 |

- The tool re-derived the Gator study's stored goal rates exactly (M1 87.95, M3 90.3, A3 90.6, straight 60.0 %;
  `reproduces_k3_goal_rates`).
- The two independently trained HMMWV f104-only ensembles differ by 1.0 point on these pairs (88.4 vs 87.4 %): a
  rough size of the training noise.
- Not like for like: another vehicle, and the HMMWV models were fitted on tiers 0-6 only (29,210 soil rows for the
  f104-only model; `arena_gator_20260925/REPORT.md` 1), the Polaris model on all tiers (52,021). Most of the gap is
  mobility: the HMMWV's straight route reaches 60.0 %, the Polaris's 97.0 %.

### U8. Caveats (unseen arenas)

- **Same terrain generator.** The 8 arenas are new seeds of the generator that made f104 (the Gator study picked them
  by terrain distance, 0.65-1.79). "Never trained on" means new terrain of the same family, not a new kind of terrain.
  Every declared pair is a hill or crater crossing.
- **Soil only, same set-up as f104:** 0.24 m of soil over a rigid floor, stock Polaris driveline, 0.25 m soil-contact
  cylinders, body not coupled to the soil (section 8 above; `REPORT.md` 7). The power-corrected Polaris was not driven
  on these arenas.
- **One ensemble.** The Polaris planner is one trained ensemble of 5 networks. At 99.7 % the margin to 90 % is far
  larger than the 1-point ensemble-to-ensemble spread seen for the HMMWV.
- **The model's absolute risk numbers carry no information here either** (mean predicted failure about 1.3e-5;
  observed 0.3 %). It still ranks routes well enough that the planner loses nothing against driving straight.
- **Standing start, one plan, whole-arena map.** No replanning and no sensing limits, as on f104.
