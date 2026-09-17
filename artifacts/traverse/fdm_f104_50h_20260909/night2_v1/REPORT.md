# Night 2 — what was tried and what it gave

## Your three model questions
1. DATA QUANTITY — yes, it was a limit, and the cheapest fix. The whole 66-h dataset takes 13.6 wall minutes to
   collect on this cluster, so I doubled it: 1,200 fresh start/goal groups x 12 designed routes (14,348 labelled).
   Ranking quality on held-out groups, same architecture and protocol throughout:
     2,491 routes .905 | 4,930 .939 | 9,773 .965 | 19,463 .980   — still rising at the end.
   But the more valuable data was a different kind: 9,309 routes drawn from the PLANNER'S OWN proposal
   distribution (wave A2). Those routes are much harder (56% unsafe vs 23% for designed routes) and the model had
   never seen anything like them. Scored on that distribution, training on them lifts ranking AUC .933 -> .969 and
   cuts the top pick's unsafe rate 10.3% -> 8.1%. On designed routes the two models are indistinguishable, which is
   why this gap was invisible in every offline number I reported last night.
2. TRANSFORMER — tested, does not help here. Same front end, BiGRU replaced by a 2-layer transformer encoder
   (d=96, 4 heads): dev ranking .947 vs .958 for the GRU, over 3 seeds. A no-sequence control (per-station MLP)
   scores .944, so sequence modelling is worth ~.014 and the transformer gives that back. Not adopted.
3. VEHICLE STATE — carries nothing, and dropping it is an upgrade. Full 17-d state .948, chassis-only
   (vx, vy, roll, pitch, roll/pitch/yaw rates) .954, no state at all .947 — all within noise. The reason is
   mechanical: every episode starts from the same settled rest state (vx, roll, pitch IQR ~0.07), so tire forces
   and engine terms are constants with noise on top. The night-2 model therefore uses route geometry only, which
   also lets it score start/goals that were never driven.

## The MPPI side — this was the real limit
Last night's 4 closed-loop failures were all sampler failures, and the diagnosis was exact:
  - lateral: knot-shaped bumps hit the validator's curvature limit at about 4 m, so 98% of wide samples were
    rejected and the effective spread was +-3-4 m;
  - speed: a sin^2 envelope forced every candidate back to the 2 m/s base at both ends, so no candidate could
    approach a hill-top or crater-rim goal with momentum.
New proposal: lateral offsets from a smooth sine basis whose amplitudes are capped by the curvature budget
(a_j <= 0.55 kappa_max L^2 / (j pi)^2), speed knots with no end envelope (only the physical deceleration cone
into the goal), and the 9 designed-style routes always included as anchors.
  detour from the straight chord median 1.8 -> 4.9 m (p95 3.7 -> 9.0, max 6.0 -> 10.0); route mean speed
  p95 2.57 -> 3.48 m/s (max 3.47 -> 5.80); commanded speed at 85% of the route: night-1 never exceeds 2.6 m/s,
  night-2 reaches 6.0.
  CORRECTION from the audit: I first wrote that the old sampler accepted only 2% of wide samples. That is wrong
  -- its measured acceptance is 48.8% on average. What actually limited it was its +-6 m clip and the speed
  envelope, not a rejection collapse. The redesign stands; my stated reason for it did not.

## What each planner chose (fresh test groups, before outcomes were known)
  night-1 planner (last night's model + proposal)   route mean speed 2.62 m/s (p10 2.15, p90 3.06)
  night-2 model on the SAME night-1 candidates      2.56 m/s   -> changed the pick in 126 of 184 groups
  night-2 model + night-2 proposal                  3.64 m/s (p10 2.72, p90 4.08)
  pessimistic ensemble                              3.33 m/s   -> differs from the mean-ensemble pick in 97/184
  always 6 m/s straight                             5.93 m/s
  fixed 2 m/s arms                                  2.00 m/s by construction (model changes the pick in 148/184)
The abstain rule never fired: with the new proposal the best candidate scored under 5% in all 184 groups.

## Caveats carried into this report
- One arena. "Fresh" means a start/goal pair at least 6 m away (jointly) from every group in training; it is not
  new terrain.
- Batch route selection: one route chosen and driven, no replanning during the drive.
- The night-1 model needs a settled initial state that fresh groups do not have; it was imputed with the training
  median, which changes 4 of 60 night-1 picks on last night's groups.
- The night-2 model is calibrated as P(failed or slid), like the night-1 one, and both are dominated by speed.

## Closed loop in Chrono: 184 fresh start/goals, 8 planners, 1,102 episodes, no errors
                            failed      unsafe (failed or slid back)   time to goal   max body tilt
  last night's planner       3.3%        8.2%                           23.4 s         23.2 deg
  + night-2 model only       2.7%        6.5%                           23.4 s         22.9
  + night-2 route proposals  0.5%        0.5%                           16.4 s         19.3
  + pessimistic ensemble     0.0%        0.0%                           17.4 s         19.3
  always 6 m/s straight      0.5%        2.7%                           10.2 s         23.6
  fixed 2 m/s, night-1 model 0.5%        3.8%                           29.2 s         17.1
  fixed 2 m/s, night-2 model 0.0%        2.7%                           28.8 s         15.5

What is and is not established (paired McNemar on the same groups):
  ESTABLISHED  the proposal change, with the model held fixed: unsafe +6.0 pts, 11 groups fixed vs 0 broken,
               p = 0.001. End to end 8.2% -> 0.5%, 14 vs 0, p = 0.0001. The drive is also 30% faster and the
               truck leans 4 degrees less.
  NOT SHOWN    the model change on its own (same candidates offered): +1.6 pts, 5 vs 2, p = 0.45.
  NOT SHOWN    at a fixed 2 m/s the night-2 model beats the night-1 model only directionally
               (2.7% vs 3.8% unsafe, 3 vs 1, p = 0.63).
  UNDERPOWERED real failures sit at 0-3.3% for every arm, so failure comparisons prove nothing here.
  HUMBLING     "always drive the straight line at 6 m/s" beats last night's planner (2.7% vs 8.2% unsafe,
               p = 0.021) and is only directionally worse than tonight's planner (2.7% vs 0.5%, p = 0.22),
               while arriving 6 s sooner. On this arena, when speed is free, most of the planning value is
               in choosing speed.
  FREE WIN     ranking candidates by the ensemble's most pessimistic member instead of its mean was the only
               arm with zero failures and zero unsafe runs. Not significant on its own (1 vs 0), but it costs
               nothing and it targets the exact failure mode from last night (the argmin landing on one seed's
               over-optimistic score).
  The abstain rule never fired: with the new proposal the best candidate scored under 5% in all 184 groups.

## CORRECTION to the headline, forced by the hazard-test audit
I first wrote "the sampler, not the network, was the bottleneck". That is too strong. Decomposing the hazard
test with arms that change one thing at a time:
    29 unsafe runs (night-1 planner)
    -> 13   change the candidate geometry only, night-1 model held      (25 vs 9, p = 0.009)
    ->  2   change the model only, candidates held                      (12 vs 1, p = 0.003)
    ->  1   add the full wide+fast proposal, night-2 model held         ( 2 vs 1, p = 1.0)
The route proposal is the larger single factor, but the network accounts for roughly 11 of the 28 avoided
unsafe runs. Both changes were needed; neither alone explains the result.
Second correction: a fixed 6 m/s straight line, with no model and no search at all, rescues 26 of the 29 and
all 11 failures. The planner's only significant advantage over it is body tilt (15 vs 2, p = 0.0023), and that
falls to p = 0.070 once near-duplicate start/goals are collapsed. What keeps the planner interesting: the
fixed-2 m/s arm is SLOWER than the control and still halves unsafe runs, so lateral width earns its keep
independently of speed -- and the 6 m/s baseline is the only arm that ever rolled the vehicle.

## The model comparison, done properly (523 more fresh groups, fixed 2 m/s, identical candidates)
The main test could not answer "is the model better?" -- with the new proposal every arm sits at the failure
floor. So the comparison was repeated at a fixed 2 m/s, where planning has to work, on 523 further fresh groups:
  slid back or failed : night-1 model 3.06%  ->  night-2 model 1.34%   (+1.7 pts, 11 vs 2, p = 0.022)
  did not reach goal  : 0.38% -> 0.38% (1 vs 1, at the floor)
  max body tilt       : median 19.2 -> 16.3 degrees
The two models never picked the same route in any of the 523 groups. So the model did improve -- it roughly
halves backward slides when speed is constrained -- but the effect is small and invisible on outright failures.

## What the audits changed (three independent checkers, plus a fourth on the offline numbers)
Everything numerical reproduced: rates, McNemar counts, times, tilt, route hashes, node pairing, the argmin of
every pick, and zero train/test id overlap. The corrections are about meaning, not arithmetic:
 - The pre-registered PRIMARY metric was the failure rate, and it is null everywhere. Every significant result
   tonight is on the secondary metric (failed-or-slid). Stated plainly rather than buried.
 - My "2% acceptance" rationale for the sampler redesign was wrong (real figure 48.8%).
 - The "new proposal" arm bundles two changes: a wider sampler AND 9 injected designed routes as anchors.
   34% of its picks were anchors. The hazard test below separates them.
 - Scoring body tilt above 35 degrees as unsafe flips one verdict: the 6 m/s straight baseline has 11 such runs
   (up to 45.9 degrees) against 0-1 for every other arm.
 - With ~16 tests reported and no multiplicity correction, the "6 m/s straight beats last night's planner"
   result (p = 0.021) does not survive correction; the main sampler result (p = 0.001) does.
 - The 184-group test set is NOT representative: selecting groups at least 6 m from all training groups keeps
   mostly long flat traverses (15% hill/crater versus 75% in the pool). That is why failures sit at the floor.
 - "Unseen" overstates it. Every metre of every test route was driven during training, in the same direction,
   at comparable speed. The honest description is: held-out start/goal pairs on one arena the model has driven
   exhaustively -- not new terrain.
 - Pre-registration is not verifiable from timestamps (PLAN.md was appended to during the night), and the
   scripts are untracked in git because committing needs your say-so.

## Test 3, the one that matters: 300 start/goals aimed at a hill or crater (1,523 episodes)
The first test set had filtered hazard terrain out, so it was repeated on 300 groups drawn only from the
hill/crater strata, with a sixth arm that removes the injected designed routes.
                            failed   slid or failed   + leaned past 35 deg   time to goal
  last night's planner       3.7%      9.7%            10.3%                  16.8 s
  night-2 planner            0.3%      0.3%             0.7%                  12.0 s
  night-2, no designed fallbacks 0.3%  0.7%             1.7%                  13.1 s
  always 6 m/s straight      0.3%      1.3%             5.0%                   7.7 s
  2 m/s, night-1 model       1.3%      4.3%             4.3%                  20.8 s
  2 m/s, night-2 model       0.0%      0.7%             0.7%                  20.6 s
  - The PRE-REGISTERED PRIMARY metric finally moves: failures 3.7% -> 0.3%, 10 groups fixed vs 0 broken,
    p = 0.002. Slides: 9.7% -> 0.3%, 28 vs 0, p < 0.0001.
  - The injected designed routes contribute nothing (+0.3 pts, 1 vs 0, p = 1.0): the wider sampler alone
    produces the entire gain. That was the audit's main fairness objection, and it is answered.
  - The model change on its own, at a fixed 2 m/s: slides 4.3% -> 0.7%, 12 vs 1, p = 0.0034. So the model did
    improve, and on hazard terrain it is significant.
  - Against "always drive 6 m/s straight": on slides alone the two are level (p = 0.38), but that baseline
    leans past 35 degrees in 13 of 300 runs (up to 45.9) against 1 for the planner. Counting that, the planner
    wins 0.7% vs 5.0%, 15 vs 2, p = 0.0023 -- for 4 seconds more travel time.

## Corrections from the fourth audit (offline claims)
 - TRANSFORMER: my verdict was too strong. The sweep gave all architectures the GRU's learning rate. At the
   deployed context variant the two tie exactly (.9551 each, 8 seeds), and at lr 1e-3 a transformer ensemble
   edges the GRU ensemble (.9641 vs .9608). Honest answer: no benefit at equal tuning, no penalty either.
 - STATE INPUTS: the conclusion (they carry nothing) survives an independent check (R^2 = .003 predicting a
   group's unsafe rate from the 17 values on 523 held-out groups), but my explanation was wrong: the settled
   state is not near-constant (tire loads vary by 3.4-3.8 kN; 3% of groups start with a wheel unloaded).
   It is uninformative, not constant.
 - SCALING CURVE: real but ~20% inflated by dev groups crowding closer to training ones as the set grows.
 - ON-POLICY DATA: strengthened. With row counts matched, the mixed set beats designed-only .9374 -> .9687,
   while 7,320 extra designed rows buy nothing. It is the distribution, not the volume.
 - "On-policy routes are harder" is mostly a speed artifact (24.9% vs 23.6% once speed-matched).

## What is still broken, and what I would do next
Across 300 hazard groups the night-2 planner has 9 bad runs out of 900 (the night-1 planner had 31):
  - 1 real failure: a crater entry it rated at 0.012% risk and then slid backwards out of - a confident miss.
  - 4 runs that leaned 35-38 degrees, each rated under 0.15%. The model cannot see this: body tilt is not in
    the training label at all. This is the clearest remaining gap and the cheapest fix.
  - 2 slides in the fixed 2 m/s arm, which the model had flagged at 0.8-2.3% but had no safer option for.
On this arena the planner is close to saturated, so more accuracy work has little room to prove itself. The
three things worth doing next, in your hands:
  (a) put tilt/rollover into the label and the cost - it explains 4 of the 9 remaining bad runs;
  (b) move to the ONLINE receding-horizon planner: everything here is batch route selection, and the online
      path still has the defects diagnosed earlier (curvature cap 0.025 vs 0.125, abstain sets speed to zero,
      and the support gate that rejects every candidate when a label head lacks support);
  (c) a second arena - every number in this report is on terrain the model has driven exhaustively.

## Housekeeping
- 18 GB of new artifacts sit in night2_v1 (mostly cached candidate sets, all regenerable from their seeds).
- The cluster is idle; all jobs completed; the campaign used ~4 node-hours of the 1,500-hour allocation.
- Nothing is committed to git. All the night-2 scripts are untracked, so no result here is tied to a committed
  code version - the auditors flagged this and it needs your go-ahead to fix.

## Putting tilt in the label (the gap found at 03:45, tested at 03:57)
The model could not see body tilt because the label never mentioned it. Relabelling every training route with
"unsafe OR leaned past 30 degrees", retraining, and re-picking on the same 300 hazard candidate sets:
  runs past 30 degrees : 22/300 -> 0/300   (p < 0.0001)
  worst tilt           : 35.7 -> 28.9 degrees; p90 28.5 -> 26.4; median unchanged (it removes the tail only)
  slid or failed       : 0.33% -> 0.67% (n.s.);  failures 0.33% -> 0.33%
  cost                 : 1.6 s slower to the goal
The label defines what the planner optimises. Saved as final/N2T_s{0..4}.pt next to the untilted N2_s*.pt.

## Two things to keep in mind when reading any number here
 - Effective sample size is 69-224 clusters, not 300 independent groups; the main result survives that, the
   failure-rate and tilt results do not survive it together with a multiplicity correction.
 - Both models are badly overconfident: the median predicted risk of the route actually driven is 0.008%
   against a realised 0.33% (night-1: 0.083% against 9.67%). The score is a usable ranking, not a probability.
