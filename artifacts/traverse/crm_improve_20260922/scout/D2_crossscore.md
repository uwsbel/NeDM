# D2: how well the existing risk models rank the routes that were actually driven on soil (diagnosis only)

Written 2026-09-22 08:50, task D2 of PLAN.md stage S0. Nothing in this note was used to choose or tune anything, and it
must not be used that way later: the 800 suite groups are evaluation data.

Command (7.7 min on the local 5090; five 260k-parameter members per model, batches of 64):

    cd /home/harry/NeDM-traverse_mppi
    OMP_NUM_THREADS=6 PYTHONPATH=src:scripts /home/harry/miniconda3/envs/nedm/bin/python \
        artifacts/traverse/crm_improve_20260922/scout/D2_crossscore.py

Outputs: `scout/D2_crossscore.json` (every number below, plus a table of all 4,652 routes with their outcome, speed
step, frame-60 vx and the logit from every model) and this note. The script `scout/D2_crossscore.py` is the only other
file written.

## What was scored

- **Routes.** K1 A5 pass 2 on CRM soil (`generalist_20260921/A_adapt/a5`). In each of the 800 suite groups (600 fresh
  `f104_pair_group_*`, 200 reused `f104_crm_eval_group_*`) six arms planned from the same recorded frame-60 state. If two
  arms picked the same route it was driven once (`run_index_crm_pass2.json`, route id -> driven_as). That leaves 4,652
  distinct routes: 732 groups with 6, 28 with 5, 14 with 4, 15 with 3, 8 with 2 and 3 with 1. Failure means
  `outcome.json` status other than goal_reached. That covers 888 routes (19.1 %): 700 soil breakthroughs, 185 prolonged
  blockages, 3 timeouts.
- **Decision state.** Pose and the 40-step history at frame 60 from `a5/poses_crm.json`, goal from the suite case, all
  exactly as in `ga_planner.decision_for`.
- **Models.** They are loaded and scored with `ga_planner.Ensemble` and `ga_planner.Scorer`, with no copied code:
  depth-map corridors from `crm_f104_v1/map_root`, float16 rounding of the corridor, history encoded once per decision,
  and the crm tag for T. The route logit is the ensemble mean of the five members' log cumulative hazard, and
  P = 1 - exp(-exp(logit)).

| name | checkpoints | what it is |
|---|---|---|
| H | `A_adapt/train/deploy_v1/H_deploy_s*.pt` | shared rigid+soil model with the 2 s history and domain head, fed the frame-60 history |
| H masked | same files | the same network with the history fully masked (the A5 "Hmask" arm) |
| T | `T_deploy_s*.pt` | shared model given the true world label (soil) |
| P | `P_deploy_s*.pt` | shared model with no label and no history |
| S'_crm | `Sp_crm_deploy_s*.pt` | soil specialist trained on the same rows |
| S'_rigid | `Sp_rigid_deploy_s*.pt` | rigid-ground specialist trained on the same rows |
| legacy CRM | `crm_f104_v1/train_v1/deploy/CRM_N2_s*.pt` | the deployed soil ensemble of the first CRM effort (13,821 whole-route rows from standing starts) |

- **Speed step** = route speed at station 0 minus the vehicle's forward speed vx at frame 60. Its positive part is
  max(step, 0). The step bins reproduce the PLAN.md anatomy exactly: < -1.5 m/s 16.7 % fail (n 132), -1.5..-0.5 6.3 %
  (632), -0.5..0.5 7.1 % (1,346), 0.5..1.5 19.7 % (1,223), > 1.5 37.1 % (1,319).

## Checks (all passed)

- All 4,800 route files hash to the sha256 in the run index. The duplicates are byte-identical to the route that was
  driven for them.
- In every one of the 4,652 pass-2 drives, the frame-60 state and pose equal the pass-1 terminal state exactly (maximum
  difference 0.0). The vx taken from the history file equals the pass-1 terminal vx. So all routes of a group really do
  start from an identical state.
- The failure flags match `results_crm_A5.json` for every arm and group (0 mismatches).
- Re-scoring each arm's own pick with its own model reproduces the logit recorded at planning time with a median
  difference of 0 to 5e-6 and a maximum of 2.4e-3. This needed one fix. The GPU convolution runs in TF32 and cuDNN picks
  its algorithm by batch shape, so each group is scored in one batch of 64 (its routes repeated), the planner's batch
  size. Scoring the routes one at a time moved logits by up to 4e-3.

## 1. Per model, all 4,652 driven routes

Within-group AUC counts the (failed, succeeded) route pairs of the same group: the share where the failed route got the
higher logit, with ties counting one half. It is pooled over the 1,369 such pairs in the 252 groups whose routes had
mixed outcomes. The 95 % intervals come from 2,000 group-bootstrap resamples. Regret is the failure of the model's
lowest-logit route among the group's driven routes minus the failure of the best driven route, averaged over all 800
groups, in points. The oracle (best driven route) fails 10.25 %. A random driven route fails 19.97 % on average.

| model | within-group AUC [95 %] | minus H [95 %] | pooled AUC | mean P / realised fail | Brier | lowest-logit route fails | regret, points [95 %] | share of random-pick regret removed |
|---|---|---|---|---|---|---|---|---|
| H | 0.836 [0.797, 0.871] | – | 0.932 | 9.3 % / 19.1 % | 0.100 | 15.12 % | 4.88 [3.38, 6.38] | 0.50 |
| H masked | 0.834 [0.798, 0.869] | -0.001 [-0.015, 0.013] | 0.929 | 3.8 % | 0.137 | 14.12 % | 3.88 [2.62, 5.25] | 0.60 |
| T (soil label) | 0.840 [0.808, 0.873] | +0.004 [-0.026, 0.036] | 0.921 | 8.3 % | 0.114 | 14.50 % | 4.25 [3.00, 5.62] | 0.56 |
| P (pooled) | 0.828 [0.793, 0.862] | -0.008 [-0.040, 0.025] | 0.921 | 3.0 % | 0.148 | 14.75 % | 4.50 [3.12, 6.00] | 0.54 |
| S'_crm | 0.814 [0.779, 0.849] | -0.021 [-0.053, 0.007] | 0.921 | 8.5 % | 0.114 | 15.88 % | 5.62 [4.12, 7.38] | 0.42 |
| S'_rigid | 0.252 [0.211, 0.293] | -0.584 [-0.648, -0.517] | 0.583 | 0.1 % | 0.190 | 35.38 % | 25.12 [22.25, 28.25] | -1.59 |
| legacy CRM | 0.850 [0.817, 0.881] | +0.015 [-0.019, 0.047] | 0.932 | 15.5 % | 0.075 | 14.50 % | 4.25 [2.88, 5.62] | 0.56 |

- **Fresh / reused groups** (within-group AUC; 1,003 and 366 pairs): H 0.839 / 0.825, H masked 0.836 / 0.828, T
  0.824 / 0.885, P 0.814 / 0.866, S'_crm 0.796 / 0.866, S'_rigid 0.260 / 0.230, legacy CRM 0.830 / 0.907.
- **Single members** score 0.79-0.85 (H members 0.786-0.846). The ensemble mean is at least as good as the median
  member. Taking the ensemble maximum instead of the mean is worse for every model (for H, 0.826).
- **Soil-competent models are statistically indistinguishable.** The five soil-competent models and the legacy
  ensemble all rank at 0.81-0.85, and no difference from H excludes zero. The 2 s history gives no ranking gain within a
  group (H vs H masked -0.001), which matches the closed-loop tie in K1 A5.
- **The rigid specialist is inverted on soil.** The routes it thinks are safest fail most: within-group AUC 0.25. Its
  own picks start with a median speed step of +1.32 m/s, against +0.25 for H, and 42 % of them have steps above
  1.5 m/s.

## 2. Only the routes picked by the soil-competent arms (the choice that matters)

The 792 routes picked only by the rigid specialist, of which 36.7 % fail, are dropped. That leaves 3,860 routes
(15.5 % fail). Only 105 groups still have mixed outcomes, with 459 pairs.
The oracle fails 10.75 %, random 16.30 %.

| model | within-group AUC [95 %] | pooled AUC | mean P / fail | lowest-logit route fails | regret, points [95 %] | share of random regret removed |
|---|---|---|---|---|---|---|
| H | 0.643 [0.567, 0.712] | 0.928 | 5.5 % / 15.5 % | 15.00 % | 4.25 [2.88, 5.62] | 0.23 |
| H masked | 0.643 [0.574, 0.704] | 0.926 | 2.1 % | 14.00 % | 3.25 [2.12, 4.50] | 0.41 |
| T | 0.654 [0.591, 0.719] | 0.915 | 4.4 % | 14.37 % | 3.62 [2.38, 4.88] | 0.35 |
| P | 0.612 [0.543, 0.679] | 0.914 | 1.5 % | 14.88 % | 4.12 [2.75, 5.50] | 0.26 |
| S'_crm | 0.573 [0.506, 0.638] | 0.916 | 4.4 % | 15.88 % | 5.12 [3.75, 6.75] | 0.08 |
| S'_rigid | 0.577 [0.507, 0.647] | 0.708 | 0.1 % | 16.12 % | 5.38 [3.88, 7.00] | 0.03 |
| legacy CRM | 0.678 [0.615, 0.741] | 0.923 | 11.4 % | 14.75 % | 4.00 [2.62, 5.38] | 0.28 |

Most of the 0.83 in section 1 comes from recognising the rigid specialist's bad routes. Among the routes that the
soil-trained models themselves proposed, they rank failures only a little better than chance (0.57-0.68). They remove
between 8 % (S'_crm) and 41 % (H masked) of the regret a random choice would carry. At the same time, the pooled AUC
stays at 0.91-0.93: the models know well which decision states are hard, but not which of several near-optimal routes
from that state will fail.

## 3. Calibration: mean predicted P / realised failure, by decile of predicted P (all 4,652 routes, % )

| decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| H | 0.0/0.6 | 0.0/0.9 | 0.0/1.3 | 0.0/1.9 | 0.0/3.4 | 0.0/4.3 | 0.0/9.9 | 0.2/25.2 | 7.8/52.5 | 85.2/91.0 |
| H masked | 0.0/0.0 | 0.0/0.6 | 0.0/1.5 | 0.0/2.2 | 0.0/4.3 | 0.0/4.1 | 0.0/10.3 | 0.2/28.4 | 3.4/50.8 | 34.5/88.8 |
| T | 0.0/0.4 | 0.0/1.7 | 0.0/1.9 | 0.0/1.9 | 0.0/2.4 | 0.0/4.7 | 0.0/12.3 | 0.2/27.5 | 6.0/52.3 | 76.3/85.8 |
| P | 0.0/0.9 | 0.0/0.6 | 0.0/1.5 | 0.0/3.0 | 0.0/3.0 | 0.0/5.4 | 0.0/12.5 | 0.1/25.8 | 1.9/52.3 | 27.6/86.0 |
| S'_crm | 0.0/0.6 | 0.0/0.9 | 0.0/1.7 | 0.0/3.0 | 0.0/2.2 | 0.0/3.7 | 0.0/13.5 | 0.2/29.2 | 6.0/51.0 | 78.7/85.2 |
| S'_rigid | 0.0/22.3 | 0.0/20.6 | 0.0/16.6 | 0.0/11.4 | 0.0/11.6 | 0.0/10.8 | 0.0/11.2 | 0.0/13.8 | 0.0/22.2 | 0.6/50.5 |
| legacy CRM | 0.1/0.0 | 0.1/0.2 | 0.1/2.2 | 0.2/3.4 | 0.3/4.1 | 0.6/3.7 | 1.7/9.9 | 8.1/19.8 | 46.2/56.8 | 97.2/91.0 |

Each decile holds 465-466 routes. Expected calibration error over the deciles: H 0.098, H masked 0.153, T 0.108,
P 0.161, S'_crm 0.106, S'_rigid 0.190, legacy CRM 0.049.

Every model trained on re-anchored rows is badly over-confident in the upper-middle range. For example, routes that H
gives 0.2 % fail 25 %, and routes it gives 7.8 % fail 52 %. The legacy soil ensemble, trained on whole routes from a
standing start, is the least over-confident. Masking the history (H masked) or removing all context (P) makes the level
much worse (mean P 3.8 % and 3.0 % against 19.1 % realised). So on soil, the history shifts the risk level rather than
the ranking within a group. Note that the models were trained on the "unsafe" label of the training rows, and here they
are judged against goal not reached.

## 4. Does the branch-point speed step explain the misses?

Method: an exact conditional (within-group) logistic regression, where each group contributes the probability of its
observed failure set given its number of failures. Features were standardised on the training folds, with ridge 1e-3
for numerical stability only. Group-level 5-fold cross-validation uses fold = md5(group) % 5. The within-group AUC is
computed on the pooled out-of-fold scores, and the rise is measured against the same fit on the logit alone. That
logit-only fit reproduces the raw AUC for every soil-competent model; for S'_rigid it flips the sign, giving 0.748. The
vehicle vx is the same for every route in a group, so on its own it cannot change any ranking inside a group. It
therefore enters only through interactions: vx x step, vx x positive step and vx x logit.

**Speed step alone within groups:** step 0.605 [0.562, 0.647], positive part 0.604, |step| 0.586, mean route speed
0.516 (all routes). On the soil-competent routes: step 0.501, |step| 0.558.

**Rise of the within-group AUC over the model's logit** (95 % paired group-bootstrap interval):

| model | all routes: + step, positive step | + vx interactions | + mean speed + step (vs + mean speed) | soil-competent routes: + step, positive step | + vx interactions |
|---|---|---|---|---|---|
| H | +0.005 [-0.005, 0.015] | +0.007 [-0.003, 0.018] | +0.004 [-0.006, 0.015] | +0.039 [+0.007, +0.079] | +0.022 [-0.015, 0.062] |
| H masked | -0.004 [-0.010, 0.004] | -0.004 [-0.013, 0.004] | +0.002 [-0.006, 0.010] | +0.009 [-0.019, 0.040] | -0.002 [-0.037, 0.034] |
| T | -0.004 [-0.013, 0.006] | -0.003 [-0.012, 0.006] | +0.001 [-0.009, 0.009] | -0.004 [-0.032, 0.022] | -0.024 [-0.053, 0.004] |
| P | -0.006 [-0.017, 0.005] | -0.007 [-0.016, 0.003] | +0.003 [-0.007, 0.012] | +0.004 [-0.030, 0.042] | +0.000 [-0.031, 0.035] |
| S'_crm | -0.006 [-0.013, 0.001] | -0.006 [-0.014, 0.002] | +0.001 [-0.006, 0.008] | -0.002 [-0.034, 0.032] | -0.007 [-0.042, 0.033] |
| legacy CRM | -0.004 [-0.010, 0.001] | -0.002 [-0.008, 0.004] | +0.000 [-0.007, 0.006] | -0.015 [-0.035, 0.004] | -0.026 [-0.047, -0.004] |
| S'_rigid (reference) | -0.033 [-0.058, -0.008] | -0.028 [-0.053, -0.006] | -0.048 [-0.077, -0.020] | +0.013 [-0.051, 0.079] | -0.007 [-0.076, 0.059] |

**Misranked pairs.** Among the within-group pairs a model gets wrong (the failed route had the lower logit), the failed
route asked for the larger speed step in only 42 % (H), 47 % (H masked), 53 % (T), 50 % (P), 50 % (S'_crm) and 47 %
(legacy) of pairs. Among correctly ranked pairs the figure is 61-64 %. So the models already get most of the pairs that
the step separates right. Their misses are pairs where the failed route did not ask for a bigger speed change. In the
39 groups where H's lowest-logit route failed although another driven route succeeded, the failed pick had a median
step of +0.89 m/s, against +1.22 m/s for the routes that succeeded.

**Absolute level (pooled logistic, same folds, intercept, vx also allowed as a main effect).** Adding the step, the vx or
their interactions to the logit changes the pooled out-of-fold AUC by -0.003 to +0.004 for every soil-competent model on
all routes (for example H 0.931 -> 0.930; P 0.920 -> 0.923, +0.0035 [0.0002, 0.0077]). Only for the rigid specialist,
whose logit carries nothing on soil, does it add much (0.580 -> 0.745). The strong pooled link between step and failure
(7 % vs 37 %) is therefore already carried by the soil models' logits, which see the route's speed profile in the
corridor. It does not point to extra information the models lack.

**Verdict:** no. At the branch point, the speed step does not explain the models' ranking misses on the driven routes.
The one interval that excludes zero (H on the soil-competent routes, +0.039) does not survive controlling for the
route's mean speed: +0.020 [-0.017, 0.060] over logit + mean speed. It is also not seen for T, P, S'_crm or the legacy
model (-0.011 to +0.017 on the same comparison). Hypothesis H1 in PLAN.md
(the handover) is not supported as a *ranking* problem. This does not rule out the handover as a *cause* of failure
that every candidate from the same state shares; that is D1's question, not this test's.

## 5. Only routes with speed step < 0.5 m/s (and |step| < 0.5 m/s)

With step < 0.5 m/s there are 2,110 routes in 593 groups, failing 7.4 %, but only 57 groups have mixed outcomes, with 182
pairs. The oracle fails 7.93 %, random 11.43 %.

| model | within-group AUC [95 %] | pooled AUC | mean P / fail | regret, points | + step, positive step | + vx interactions |
|---|---|---|---|---|---|---|
| H | 0.720 [0.608, 0.822] | 0.932 | 3.1 % / 7.4 % | 2.87 | +0.049 [+0.006, +0.105] | +0.060 [+0.015, +0.117] |
| H masked | 0.742 [0.643, 0.830] | 0.939 | 1.3 % | 2.02 | +0.066 [+0.016, +0.126] | +0.060 [+0.007, +0.121] |
| T | 0.819 [0.739, 0.891] | 0.928 | 3.8 % | 1.18 | +0.027 [-0.013, 0.072] | +0.027 [-0.017, 0.073] |
| P | 0.791 [0.693, 0.880] | 0.930 | 1.3 % | 1.69 | +0.027 [-0.015, 0.074] | +0.027 [-0.015, 0.074] |
| S'_crm | 0.758 [0.649, 0.856] | 0.937 | 3.6 % | 2.19 | +0.044 [0.000, 0.094] | +0.033 [-0.011, 0.083] |
| S'_rigid | 0.341 [0.231, 0.457] | 0.540 | 0.0 % | 5.73 | -0.132 [-0.252, -0.010] | -0.187 [-0.307, -0.065] |
| legacy CRM | 0.791 [0.690, 0.887] | 0.939 | 5.4 % | 2.02 | +0.005 [-0.040, 0.059] | +0.005 [-0.040, 0.059] |

Here the step alone ranks at 0.497. The fitted step terms put a negative weight on the positive part (-0.43 to -0.61
per standard deviation, 5-fold mean, for the soil models) and almost none on the step itself. In this subset, routes
that ask for a small speed-up (0 to 0.5 m/s) fail less than the models expect, compared with routes that hold or cut
speed at the branch. Mean speed alone adds nothing (for H +0.000). The rises
are borderline and rest on 57 groups, so read them as a hint, not a finding.

With |step| < 0.5 m/s there are 1,346 routes in 551 groups, failing 7.1 %, with 34 mixed groups and 75 pairs.
Within-group AUC is H 0.773, H masked 0.773, T 0.813, P 0.840, S'_crm 0.827, S'_rigid 0.240, legacy CRM 0.853. The step
terms change it by -0.013 to +0.067, and every interval includes zero. Regret for the soil models is 0.73-1.09 points
(oracle 8.35 %, random 10.76 %). Calibration is closer here (H mean P 3.2 % vs 7.1 % realised; legacy 5.5 %).

## 6. Side finding: the 4 x 64 search often misses a route that its own model scores lower

Under its own model, each arm's closed-loop pick is the lowest-logit route among the group's driven routes in only 58 %
(H), 50 % (H masked), 66 % (T), 65 % (P), 72 % (S'_crm) and 93 % (S'_rigid) of groups. In the other groups a route
proposed by a different arm scores lower, by a median of 0.24-0.36 logit units (90th percentile 1.6-2.3). That route
also fails less often than the arm's own pick: H 17.4 % vs 19.8 % over 333 groups, T 15.9 vs 19.2 % (271), P 19.4 vs
24.5 % (278), S'_crm 19.3 vs 23.2 % (228). Taking H's lowest logit over the six arms' routes would have failed 15.1 %
against H's own closed-loop 16.1 %, so about one point is lost to search. The ranking problem in section 2 is the larger
one.

## Reading

1. The existing soil models are about equal: H, H masked, T, P, S'_crm and the legacy ensemble. None of them tells
   apart the near-optimal routes that the soil-trained planners propose from one moving start. Within-group AUC is
   0.57-0.68 on those routes, against 0.98+ offline on training-distribution rows. They do know which starts are hard
   (pooled AUC 0.92).
2. The speed step at the branch does not explain the ranking misses. Adding it, its positive part or vx interactions to
   any soil model's logit leaves the within-group AUC unchanged on all routes, and it gives no consistent gain on the
   soil-competent routes. The misses are mostly pairs without a larger speed step on the failing route. A weak
   exception is the step < 0.5 m/s subset, where braking-at-branch routes fail more than predicted (57 groups).
3. Calibration against realised failure is poor for every model trained on re-anchored rows: large under-prediction
   between the 7th and 9th deciles. The standing-start legacy ensemble is the best calibrated and ranks at least as well
   (0.850 all routes, 0.678 soil-competent), with no significant difference from H.
4. Consequence for the plan, as a diagnosis rather than a selection: a candidate family that removes the speed step
   (S1 `cont`) should not be expected to help through better *ranking*. Any gain has to come from removing a failure
   cause that the step creates for every candidate. The ranking gap among good candidates looks like a
   training-distribution problem (H3: decision states after an approach, with several continuations labelled in
   Chrono), which is what S3 targets.

## Caveats

- One arena, and the models were fitted on rows from the same arena. The suite groups are new start-goal pairs, not new
  terrain.
- The driven routes are not a random sample of candidates. Each is some arm's optimum, so the within-group pairs are
  deliberately hard, and absolute AUCs are not comparable with the offline ones.
- The speed step and vx are the only handover features tested. The start-heading step, pitch and grade at the branch
  are stored per route in the JSON (heading) or in D1, and were not tested here.
- The within-group AUC uses only the 252 (all) / 105 (soil-competent) / 57 (step < 0.5) groups with mixed outcomes.
  Groups where every driven route failed (86 under the soil-competent arms) carry no ranking information and are D1's
  subject.
- Numerical: the re-scored logits differ from the planning-time values by at most 2.4e-3 (TF32 convolution). That is far
  below the logit gaps that decide any ranking here.
