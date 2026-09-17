# Night 2 (2026-09-12, started 01:35) — pre-registered plan

Milestone: lower the planner's unsafe / failure rate further on FRESH held-out start/goals, and answer the
user's three questions with evidence. Everything below is written before any result is seen.

## Facts that shape the plan
- The whole 66-h dataset (11,412 episodes) was collected in 13.6 wall minutes on 2,008 workers. Data is cheap;
  the question is WHICH data, not how much.
- Last night's failures came from the sampler, not the model: lateral spread <= ~3-4 m, route-average speed
  <= ~3.3 m/s, and the sin^2 envelope forces ~2 m/s near start and goal (so a goal on a hill can't be reached fast).
- The 3 "confident" misses were each the only candidate under 5% among ~100%-risk candidates (argmin fluke).

## A. Data (user's Q1: is quantity the bottleneck?)
A1 coverage wave: 1,200 NEW start/goal groups, same 12 designed routes each (3 offsets x 4 speed profiles)
    -> ~14.4k episodes. Ids f104_v2_group_XXXX, split by the same deterministic hash.
A2 on-policy wave: ~700 groups x 12 routes sampled from the PLANNER's proposal distribution, including the
    widened sampler support (lateral up to +-10 m, knots 3-5, speeds free at the ends) -> ~8.4k episodes.
    Rationale: the model is queried on these routes but was never trained on them.
Deliverable: data-scaling curve at 25 / 50 / 100 / 200% of episodes, same architecture, same protocol.
Decision rule: quantity is the bottleneck only if the curve is still rising at 200%.

## B. Architecture (user's Q2: transformer?)
Same station front end; swap the BiGRU for a transformer encoder (2-4 layers, d 64-128, 4 heads, learned
positional). Controls: current BiGRU (H1), and a no-sequence per-station MLP to measure how much sequence
modelling matters at all. 3 seeds each, dev fold only.

## C. State inputs (user's Q3: which vehicle states?)
Context vector variants: (i) full 17 (current: incl. tire forces, engine speed/torque), (ii) chassis only
(vx, vy, roll, pitch, roll rate, pitch rate, yaw rate), (iii) no state at all. 3 seeds each.
Note: every episode starts from the same settled rest state, so (iii) may lose nothing -- that is the point.

## D. MPPI sampling
D1 widen: lateral sigma 2.6 -> 4.0, max lateral 6 -> 10 m, knots 3 -> 4.
D2 free end speed: replace the sin^2 speed envelope with one that only forces the terminal stop, so a route
   can approach a hill-top goal fast.
D3 anchors: always include straight at 2 / 4 / 6 m/s and +-4 m offsets at 2 / 4 / 6 m/s.
D4 refinement: 2 CEM rounds (sample 256 -> keep best 32 -> refit -> resample 256).
D5 abstain rule (pre-registered threshold, decided NOW): if the best OR second-best candidate scores >= 5%,
   fall back to the safest anchor instead of the argmin.

## E. Final closed-loop Chrono test (the milestone)
FRESH groups only (the new test/val groups from A1, never used in any earlier session). All arms share one
job per group (node determinism). Arms:
  1. deployed model + tonight's sampler baseline (last night's settings)   [control]
  2. best new model + same sampler                                        [model gain]
  3. best new model + new sampler (D1-D4)                                 [sampler gain]
  4. arm 3 + abstain rule (D5)                                            [policy gain]
Primary metric: failure rate (did not reach goal). Secondary: unsafe rate (failed or slid backwards).
Report McNemar on paired groups + group bootstrap CIs. No arm is dropped after seeing results.

## Order of work
1 cases + collection wave A1 (longest lead)  2 on-policy wave A2  3 B/C sweep on existing data
4 rebuild dataset with 2x data, scaling curve, train final  5 sampler + closed loop  6 report.

## Honesty rules carried over
Single arena. Batch route selection, not an online receding-horizon planner. Any threshold chosen after
seeing data is labelled post-hoc. Report the pre-registered arms even if they lose.

## Addendum (01:50, before any result): definition of the fresh test set
New groups land within ~0.2-0.5 m of old ones in places, so "new" is not automatically "unseen". FIXED RULE:
a fresh held-out group is one whose split is test or val AND whose (start, goal) 4-D distance to EVERY old
campaign group (all 1,500) and every new training group is >= 4 m. The surviving list is frozen in
night2_v1/fresh_test_groups.json before any model is trained. The final closed-loop test uses only these.

## Decisions recorded 02:35, before the new data landed
Sweep result on existing data (dev fold, 3 seeds, G_unsafe): GRU .958 > transformer .947 > no-sequence MLP .944;
state inputs chassis .954 ~ full .948 ~ none .947 (all within ~2 sigma). So: keep the GRU; use the CHASSIS state
variant (drops tire forces / engine, loses nothing); the transformer is not adopted. Reported either way.
Closed-loop arms use the GRU+chassis model trained on ALL available data (old + A1 + A2). The data-scaling curve
(old only / +A1 / +A1+A2) is reported whatever it shows.
Arm 4's abstain threshold stays 5% on the best-or-second-best candidate, as fixed in section D5 above.
Arm 5 (reference) = always drive the 6 m/s straight anchor.

## Arm 6 added 03:25, before any night-2 closed-loop result
Arm 6 "pessimist": night-2 model on night-2 candidates, ranked by the ensemble's MOST PESSIMISTIC member
(max over the 5 seeds) instead of the mean. Reason: all three of last night's confident misses were routes where
one seed was far more optimistic than the rest, and the argmin selects exactly such outliers. It could not be
pre-tested offline: on the designed-route pool both aggregations pick a safe route in 158/158 held-out groups.

## Secondary test declared 04:20, before any night-2 closed-loop result
Offline check on the new candidate sets: night-1 and night-2 models rank them almost identically (rank
correlation 0.972) and both are dominated by speed (median risk 0.001 for routes above 3.5 m/s mean speed,
0.95-0.98 below 2 m/s). So with speed free the planner will mostly choose "go fast", and arms 1-6 may end up
measuring the speed choice rather than the risk model. The regime where the model has to work is constrained
speed, so a secondary paired test is added on the SAME 184 fresh groups:
  7 fixed2-control : 2 m/s fixed, geometry-only candidates (night-2 lateral family), night-1 model
  8 fixed2-new     : the same candidates, night-2 model
Primary metric is the same (failure rate; unsafe rate second). Declared before any outcome was observed.

## Extension declared 02:45 (after the main result, before the extension's own data exists)
The main test cannot answer "is the model better?": with the new proposal every arm is near the floor, and the
model-only comparison came out 5 vs 2 (p=0.45). The regime where the model has to work is constrained speed, so
the secondary 2 m/s test is repeated on a LARGER fresh set for power:
  groups: from the same 4,000-group pool, never trained on, 4-D (start,goal) margin >= 4 m from every training
          group (weaker than the primary test's 6 m -- stated because it is a weaker guarantee), >= 3 m apart
          from each other, and excluding the 184 groups already used.
  arms  : exactly two, on identical fixed-2 m/s candidate sets: night-1 model vs night-2 model. Paired, one node.
  metric: unsafe rate (primary here, since 2 m/s produces enough events), then failures. McNemar on the pair.
No other change. If this comes out null too, the honest conclusion is that tonight's model work did not improve
closed-loop behaviour and only the proposal did.

## Process note (written 03:00, after the main result)
PLAN.md has been appended to during the night, so its file mtime is LATER than the main closed-loop result.
The pre-registration is therefore not verifiable from timestamps alone; treat the sections above as declared
when they say they were, and discount accordingly. Scripts are untracked in git (no commit without the user's
say-so). Deviations to record: the frozen test rule shipped as ">= 6 m from a separate 4,000-group pool"
rather than the addendum's ">= 4 m and split in {test,val}" (stricter but different), and PLAN item D4
(2-round CEM refinement) was never implemented -- the night-2 proposal is one-shot rejection sampling.

## Hazard test declared 03:00, before its data exists — fixes two flaws the audits found
Flaw 1: the 184-group test set is unrepresentative. Selecting groups >= 6 m from all training groups keeps
mostly long flat traverses: 15.2% of them target a hill or crater versus 75.3% of the pool. That is why every
arm sits at the failure floor and why the model comparisons are underpowered.
Flaw 2: the "proposal" arm bundles two changes -- a wider random sampler AND 9 injected designed-route anchors.
34% of its picks were anchors, and 5 of its 14 winning discordant groups were anchors.
HAZARD TEST: ~250 groups drawn ONLY from the hill/crater strata, >= 4 m margin from every training group,
>= 3 m apart, excluding all groups already used tonight. Six arms on identical groups, one node per group:
  1 control        night-1 model, night-1 proposal
  2 sampler        night-2 model, night-2 proposal (anchors included)
  3 sampler_noanchor  night-2 model, night-2 proposal with the 9 anchors REMOVED  <- separates flaw 2
  4 anchor6        always the 6 m/s straight line
  5 fixed2_old / 6 fixed2_new   2 m/s fixed geometry-only candidates, night-1 vs night-2 model
Metrics reported together, no cherry-picking: failures; unsafe (failed or slid); and unsafe-or-tilt where a
run also counts as bad if |roll| or |pitch| exceeds 35 degrees (the audit showed the 6 m/s baseline buys its
safety record with 11 runs above 35 degrees, up to 45.9).

## Tilt experiment declared 03:40, before its data exists
Finding that motivates it: on the 300 hazard groups the night-2 planner has 9 bad runs, and 4 of them are runs
that leaned 35-38 degrees while the model rated them under 0.15% risk. The model cannot see tilt: the training
label is "failed or slid backwards" and says nothing about roll or pitch. The fast straight-line baseline is
worse still (13 of 300 runs past 35 degrees, up to 45.9).
EXPERIMENT: relabel every training route with unsafe_tilt = unsafe OR max(|roll|,|pitch|) after the 1 s settle
> 30 degrees, retrain the same architecture (GRU, geometry-only context, 5 seeds), and re-pick on the SAME
cached hazard candidate sets. Two arms, 300 groups, identical candidates, one node per group:
  A = current night-2 model      B = tilt-aware model
Metrics, all reported: median max body tilt (paired, the high-power one), fraction of runs past 35 degrees
(the pre-registered threshold from the hazard test), slid-or-failed, and failures. A 30 degree training
threshold with a 35 degree test threshold is deliberate: more training signal, unchanged test definition.
Risk to watch: trading slides for tilt. If slides rise, that is the result and it gets reported.
