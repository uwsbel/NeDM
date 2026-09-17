# Night 2 (2026-09-12) — more data, transformer, state inputs, and a wider/faster MPPI proposal

PLAN.md holds the pre-registration (written before any result). Times are CDT, taken from artifact mtimes
(an earlier draft of this log carried guessed times that ran several hours fast).

## [01:35] Setup
The whole 66-h night-1 dataset was collected in 13.6 wall minutes on 2,008 workers, so data quantity is cheap;
the choice is WHAT to collect. Waves:
  A1 coverage  : 1,200 fresh start/goal groups x the same 12 designed routes = 14,400 episodes (jobs 416128/9/56)
  A2 on-policy : 1,200 groups x 8 routes drawn from the planner's own proposal = 9,600 episodes (jobs 416167/8/9)
Test set: a separate 4,000-group pool, never collected, never trained on; 184 groups kept whose (start, goal)
4-D distance is >= 6 m from every training group and >= 3 m from each other (fresh_test_groups.json).

## [01:40] Architecture and state inputs (user questions 2 and 3), dev fold, 3 seeds, night-1 data
G_unsafe (same group, same speed profile): GRU .958 > transformer .947 > no-sequence MLP .944
State inputs: chassis-only .954 ~ full 17-d .948 ~ none at all .947  (per-config sd .005-.008)
  -> The transformer does NOT beat the GRU here. Vehicle state carries nothing: every episode starts from the
     same settled rest state (vx, roll, pitch IQR ~0.07), so tire forces and engine terms are noise. Dropping
     the state entirely also makes the model usable on start/goals that were never driven, which the closed-loop
     test needs. Adopted: GRU, geometry-only context.

## [01:55] The proposal was the real limit (night-1 failure analysis)
Night-1 candidates: lateral spread ~+-4 m, and a sin^2 speed envelope pinned the speed to the 2 m/s base at both
ends, so no candidate could approach a hill-top goal with momentum. Replacing the knot bumps with a
curvature-safe sine basis (a_j <= 0.55 kappa_max L^2/(j pi)^2) and dropping the end envelope:
  acceptance rate 2% -> 73-93%, median detour 2.0 -> 4.6 m (max 10 m), speed at 85% of the route 2.0 -> up to 5.3 m/s.

## [02:22] Data-scaling curve (user question 1), same architecture and dev fold throughout
  25% of groups (4,930 routes)  G_unsafe .936 / .941
  50%           (9,773 routes)  G_unsafe .957 / .972
 100%          (19,463 routes)  G_unsafe .983
  -> Ranking quality was still data-limited: doubling the designed data moves the dev AUC from ~.94 to ~.98.

## [02:00] Proposal coverage on the 184 fresh test groups (same groups, both proposals, 256 candidates each)
  detour from the straight chord : median 1.76 -> 4.89 m,  p95 3.63 -> 8.97 m,  max 6.0 -> 10.0 m
  route mean speed               : p95 2.70 -> 3.68 m/s,   max 3.73 -> 5.95 m/s
  candidate generation is now reproducible (md5 group seed; the first version used Python's salted hash()).

## [02:05] Waves collected
A1 14,348 designed routes labelled (unsafe 23.4%, fail 10.7% — same distribution as night 1: 25.0% / 11.7%).
A2 9,309 on-policy routes labelled of 9,520 driven (211 stragglers not yet complete when the tensors were built;
9,600 were requested). The planner's own proposals are much harder than designed routes:
unsafe 56.4%, fail 33.2%. Merged training set: 36,199 routes over 2,700 groups.

## [02:21] Does the on-policy data help? (the designed-route dev metric cannot see this)
Same architecture, dev fold held out of fitting, scored on the dev fold's ON-POLICY routes (1,989 routes, 255 groups):
  trained on designed only        within-group AUC unsafe .933   top pick unsafe 10.3%
  trained on designed + on-policy within-group AUC unsafe .969   top pick unsafe  8.1%
On designed routes the two are indistinguishable (.978-.980 vs .976), so the gain is specific to the distribution
the planner actually queries. This is why the night-1 model looked excellent offline and still picked bad routes.

## [02:12] Closed-loop test launched (job 416210): 8 arms x 184 fresh groups, 1,102 episodes, 40 array tasks
0 abstentions: with the night-2 proposal the best candidate scored < 5% in every group, so arm 4 == arm 3.

## [02:13] What each arm chose (before any outcome was known)
picked route mean speed, median (p10-p90) m/s:
  control (night-1 planner)  2.62 (2.15-3.06)      model (night-2 model, night-1 candidates) 2.56 (2.12-3.09)
  sampler / abstain          3.64 (2.72-4.08)      pessimist 3.33 (2.47-4.41)
  anchor6                    5.93                  fixed2_control / fixed2_new  2.00 (by construction)
Model risk of its own pick: night-1 median 0.00072, night-2 median 0.00006.
So the new proposal mainly buys speed: the planner now chooses ~3.6 m/s where it used to be stuck near 2.6 m/s.

## [02:38] CLOSED LOOP, 184 fresh start/goals, 8 arms, 1,102 episodes, 0 episode errors
                       failed        unsafe (failed or slid)   median time to goal   median max tilt
  control (night-1)     3.3% (6)      8.2% (15)                 23.4 s                23.2 deg
  + night-2 model       2.7% (5)      6.5% (12)                 23.4 s                22.9
  + night-2 proposal    0.5% (1)      0.5% (1)                  16.4 s                19.3
  + pessimistic ens.    0.0% (0)      0.0% (0)                  17.4 s                19.3
  always 6 m/s straight 0.5% (1)      2.7% (5)                  10.2 s                23.6
  2 m/s, night-1 model  0.5% (1)      3.8% (7)                  29.2 s                17.1
  2 m/s, night-2 model  0.0% (0)      2.7% (5)                  28.8 s                15.5
Paired tests (McNemar, discordant counts):
  proposal change, model held fixed : unsafe +6.0 pts [+2.7, +9.8], 11 vs 0, p=0.0010   <- the one solid result
  model change, candidates held fixed: unsafe +1.6 pts [-1.1, +4.3],  5 vs 2, p=0.45    (not significant)
  night-1 planner -> night-2 planner : unsafe +7.6 pts [+3.8, +11.4], 14 vs 0, p=0.0001
  night-1 planner vs always 6 m/s    : unsafe +5.4 pts, 13 vs 3, p=0.021  <- last night's planner LOSES to a
                                                                            fixed fast straight line
  night-2 planner vs always 6 m/s    : unsafe +2.2 pts, 5 vs 1, p=0.22    (directional, not proven)
  pessimistic vs mean ensemble       : +0.5 pts, 1 vs 0, p=1.0 (only arm with zero failures and zero unsafe)
  SECONDARY at 2 m/s, night-1 -> night-2 model: unsafe +1.1 pts, 3 vs 1, p=0.63 (directional, not proven)
Failure rates are at the floor (0-3.3%), so the failure comparisons are underpowered; the unsafe rate carries
the signal. The abstain rule never fired (best candidate < 5% in all 184 groups), so arm 4 == arm 3 exactly.

## Verdicts
- The PROPOSAL was the bottleneck, not the network. Widening it (and freeing the end speed) cut unsafe runs
  8.2% -> 0.5% and made the drive 30% faster.
- More data helps the offline ranking (.905 -> .980) and on-policy data helps on the planner's own distribution
  (.933 -> .969), but neither produced a significant closed-loop gain on its own (model arm p=0.45).
- Transformer: no. Vehicle state: no. Both null results, measured 3 seeds each.
- Humbling: last night's planner is beaten by "always drive the straight line at 6 m/s" (8.2% vs 2.7% unsafe,
  p=0.021). The night-2 planner beats that baseline directionally (0.5% vs 2.7%) but not significantly, and it
  is slower (16.4 s vs 10.2 s) -- it buys safety margin and lower body tilt, not time.

## [02:45] Extension launched (job 416252): the 2 m/s model comparison on 523 more fresh groups, 929 episodes
Declared in PLAN.md before the data existed. Two arms only, identical candidate sets, night-1 model vs night-2
model, to give the model comparison the power the main test lacked.

## [03:05] EXTENSION RESULT: the 2 m/s model comparison, 523 fresh groups, identical candidate sets
  unsafe  night-1 model 3.06%  ->  night-2 model 1.34%   +1.72 pts [+0.38, +3.06], 11 vs 2, McNemar p=0.0225
  failed  0.38% -> 0.38% (1 vs 1, at the floor)
  max body tilt median 19.2 -> 16.3 deg; time to goal unchanged (24.8 vs 24.6 s)
The two models never picked the same route in any of the 523 groups. So in the speed-constrained regime the
night-2 model IS better than the night-1 model, with power: it halves backward slides. The effect is small in
absolute terms and invisible on outright failures, which sit at the floor.
This is the answer to "did the model improve?": yes, but modestly, and only measurable where speed is fixed.

## [03:10] Audit findings (3 independent auditors) — corrections to the record
CONFIRMED: every closed-loop number reproduces exactly (rates, McNemar counts, times, tilt, dedup, argmin,
node pairing, route hashes); no train/test id overlap; margins exact; fit masks cannot see test groups.
CORRECTED:
 1. "night-1 acceptance rate 2%" is WRONG: measured 48.8% mean (min 12.7%). The redesign is still justified by
    the real limits (6 m lateral CLIP, route-mean-speed ceiling 3.47 m/s, speed pinned to 2.0 m/s at the goal),
    not by an acceptance collapse. The claim appears in LOG [01:55] and the f104_n2_sampler.py docstring.
 2. Mean-speed coverage numbers were stale (pre-seed-fix): actual p95 2.57 -> 3.48, max 3.47 -> 5.80 m/s.
 3. The "sampler" arm bundles TWO changes: a wider random sampler AND 9 injected designed-route anchors
    (34% of its picks were anchors; 5 of 14 winning discordant groups). The hazard test adds a
    sampler_noanchor arm to separate them.
 4. The pre-registered PRIMARY metric (failures) is null everywhere; all significant results are on the
    secondary metric (unsafe). Stated plainly now.
 5. Scoring a run as bad if body tilt exceeds 35 deg flips one verdict: the 6 m/s straight baseline has 11 such
    runs (max 45.9 deg) vs 0-1 for the other arms, so "last night's planner loses to 6 m/s straight" becomes
    p=1.0 under that metric, while "night-2 planner beats 6 m/s straight" becomes 1 vs 15, p=0.0005.
 6. No multiplicity correction over ~16 tests: the 13v3 (p=0.021) claim does not survive it; the headline
    11v0 (p=0.001) does.
 7. The 184-group test set is unrepresentative: 15.2% hill/crater strata vs 75.3% in the pool. That is why
    failure rates sit at the floor. The hazard test (300 groups, all hill/crater) addresses this.
 8. "Unseen" overstates it: 100% of every test route's centreline was driven during training, same direction,
    comparable speed. Honest wording: held-out start/goal pairs on one arena the model has driven exhaustively.
 9. PLAN.md was appended to during the night, so its mtime postdates the main result; pre-registration is not
    verifiable from timestamps. Scripts remain untracked in git.
10. Minor: 72 stale route files from an aborted pick pass sit in closed/routes (none referenced by any task);
    f104_n2_analyze.py prints "six arms" while checking eight; N2_meta.json G_unsafe is in-sample.

## [03:10] Where the new model wins (extension test, 11 groups fixed vs 2 broken)
It is not choosing bigger detours: median detour 6.50 m (old model) vs 6.21 m (new), and in the fixed groups
7.6 m vs 7.2 m. It chooses a DIFFERENT detour of the same size. 5 of the 11 fixed groups are hill cross-slope
or hill entry/exit cases. In several of them the old model was confidently wrong -- it rated the route it
picked at 0.06-0.62% risk and that route then slid backwards at up to 2.7 m/s.

## [03:35] HAZARD TEST: 300 groups that actually target a hill or crater, 6 arms, 1,523 episodes
                     failed   unsafe   unsafe-or-tilt>35deg   time to goal   tilt>35deg runs
  control (night-1)   3.7%     9.7%      10.3%                 16.8 s         2/300
  sampler (night-2)   0.3%     0.3%       0.7%                 12.0 s         1/300
  sampler, NO anchors 0.3%     0.7%       1.7%                 13.1 s         3/300
  always 6 m/s        0.3%     1.3%       5.0%                  7.7 s        13/300
  2 m/s, night-1 model 1.3%    4.3%       4.3%                 20.8 s         0/300
  2 m/s, night-2 model 0.0%    0.7%       0.7%                 20.6 s         0/300
Paired (McNemar):
  night-1 -> night-2 planner      failed +3.3 pts, 10 vs 0, p=0.002   <- the PRE-REGISTERED PRIMARY metric is
                                  unsafe +9.3 pts, 28 vs 0, p<0.0001     now significant, on representative terrain
  the injected designed anchors   +0.3 pts unsafe, 1 vs 0, p=1.0  -> they contribute essentially NOTHING;
                                  the wider sampler alone gives the whole gain (audit confound resolved)
  at 2 m/s, model change only     unsafe +3.7 pts, 12 vs 1, p=0.0034  <- the model gain is significant here too
  night-2 planner vs 6 m/s straight   unsafe +1.0 pts p=0.38, but unsafe-or-tilt +4.3 pts, 15 vs 2, p=0.0023
                                  -> once leaning past 35 deg counts as bad, the planner clearly beats the
                                     fast straight line (13/300 of its runs exceed 35 deg, up to 45.9)
On the terrain the first test filtered out, every claim gets stronger and the primary metric finally moves.

## [03:40] Fourth audit (offline claims) — further corrections
 1. ARCHITECTURE, weakened. The sweep numbers are means over 9 runs (3 seeds x 3 context variants), not
    "3 seeds each" as I wrote. Re-run with 8 seeds: GRU .9528, transformer .9444, MLP .9394 — the gap is half
    what I reported. AT THE DEPLOYED CONTEXT (geometry only) THE TWO TIE EXACTLY (.9551 each). And the sweep
    used one learning rate (2e-3, tuned for the GRU) for all architectures: at lr 1e-3 a 5-seed transformer
    ensemble scores .9641 vs the GRU ensemble's .9608. So "the transformer does not help" is really "the
    transformer at the GRU's learning rate does not help" — an optimiser artifact, not an architecture verdict.
    Also "sequence modelling is worth ~.014" is not established (CI includes zero).
 2. STATE INPUTS: conclusion holds, my stated reason was wrong. The settled state is NOT near-constant — tire
    loads vary by 3.4-3.8 kN between groups and 81 groups (3%) start with a wheel already unloaded. It is
    uninformative, not constant: a linear model of group unsafe-rate on those 17 values gets R^2 = .003 on 523
    held-out groups. Also the ranking flips with 8 seeds (none > full ~ chassis), and the dev metric is
    structurally blind to state (zero within-cell variance), so that ablation could barely have detected it.
 3. SCALING CURVE: numbers exact, interpretation weakened. The dev fold is held out by name hash, not geometry,
    so as training grows the dev groups get closer to training ones (median 5.60 -> 2.83 m). A pure 3-nearest-
    neighbour label lookup improves .596 -> .800 over the same range. Within the distant stratum about .054 of
    the .066 gain survives, so most of the curve is real generalisation and roughly a fifth is crowding.
 4. ON-POLICY DATA: confirmed and strengthened by a control I had not run. Matching row counts (19,463 rows of
    designed-only vs a mixed set of the same size): .9374 -> .9687, and adding 7,320 more DESIGNED rows buys
    nothing (.9348 -> .9374). So it is the distribution, not the volume. Paired bootstrap on top-pick unsafe:
    -2.6 pts [-4.8, -0.5]. Context: on that dev set random picking is 57.4% unsafe, always-fastest 35.3%, and
    the oracle floor is 6.3% (16 of 255 groups have no safe route), so 10.6% -> 8.0% is about half the headroom.
 5. "On-policy routes are harder (56.4% vs 23.4%)" is mostly a SPEED artifact: reweighted to the designed speed
    mix it is 24.9% vs 23.6%. The A2 wave's real contribution is continuous speed and shape coverage.
 6. Training is not seed-deterministic (cuDNN GRU) and f104_n2_train.py was edited after sweep.json was written,
    so those per-seed numbers are not exactly reproducible from the current script.

## [03:45] What is still broken (night-2 planner, 300 hazard groups, 9 bad runs of 900)
  1 real failure : group 0620, crater entry/cross/exit, timed out after sliding to -3.09 m/s at 3.9 m/s
                   commanded. The model rated that route 0.012% -- a confident miss, the same shape as last
                   night's lone-outlier misses. Both sampler arms picked it, so it is not an anchor artifact.
  4 tilt events  : 35-38 deg lean on hill cross-slopes and a crater exit, each rated 0.006-0.121% by the model.
                   The model CANNOT see this: tilt is not in the training label at all. This is the clearest
                   remaining gap and a cheap fix (add a tilt term to the unsafe label and retrain).
  2 slides at the fixed 2 m/s arm, rated 0.8-2.3% -- the model knew, but had no better option at that speed.
Clean night-2 picks run at a median 3.70 m/s (p10 2.86, p90 4.30).
For comparison the night-1 planner had 31 bad runs on the same groups.

## Where this line stands
On this arena, with held-out start/goal pairs, the planner is close to saturated: 0.3% failures and 0.7% on the
strictest metric over 300 hazard-targeted groups. Further accuracy work here has almost no headroom to prove
itself. The three things actually worth doing next, in the user's hands:
  (a) tilt/rollover as an explicit part of the label and cost (cheap, addresses 4 of the 9 remaining bad runs);
  (b) the ONLINE receding-horizon planner -- everything tonight is batch route selection, and the online
      planner still carries the diagnosed defects (curvature cap 0.025 vs 0.125, abstain -> SetDesiredSpeed(0),
      the supported[0] and supported[2] gate);
  (c) a second arena -- every result tonight is on terrain the model has driven exhaustively.

## [03:57] TILT EXPERIMENT: adding tilt to the label, 300 hazard groups, identical candidates, 532 episodes
  runs past 30 deg : 22/300 -> 0/300   (22 vs 0, p < 0.0001)
  runs past 35 deg :  1/300 -> 0/300   (too rare to test)
  max tilt         : worst 35.7 -> 28.9 deg; p90 28.5 -> 26.4; MEDIAN UNCHANGED 20.8 vs 20.9 (it cuts the tail only)
  slid or failed   : 0.33% -> 0.67% (1 vs 2, n.s.)     failures 0.33% -> 0.33% (1 vs 1)
  cost             : 1.6 s slower to the goal (12.0 -> 13.6 s); picks 3.37 m/s instead of 3.61 m/s
The label defines what the planner optimises: put tilt in it and the tilt tail disappears at negligible cost.
Deployed as final/N2T_s{0..4}.pt alongside N2_s*.pt.

## [04:00] Audit of the hazard test — numbers exact, ATTRIBUTION CORRECTED
Every rate, McNemar count, route hash, argmin and shard assignment reproduces bit-exactly, and the candidate
sets are equal-sized and all valid. Three findings change what I may claim:
 1. "The proposal, NOT the network" is REFUTED. Decomposing with arms that hold one factor fixed:
      29 unsafe (night-1 planner)
      -> 13  change the candidate geometry only, night-1 model held   (25 vs 9, p = 0.009)
      ->  2  change the model only, candidates held                   (12 vs 1, p = 0.003)
      ->  1  add the full wide+fast proposal, night-2 model held      ( 2 vs 1, p = 1.0)
    So the geometry change is the larger single factor but the network accounts for ~11 of the 28 avoided
    unsafe runs. The correct claim is "the proposal was the larger factor", not "it was the only one".
 2. A fixed 6 m/s straight line reproduces most of the gain with no model and no search: it rescues 26 of the
    29 unsafe control groups and all 11 failures, and ties the planner on failures (1v1) and on slides
    (4v1, p = 0.375). The planner's only significant advantage is tilt (15 vs 2, p = 0.0023), and that falls to
    p = 0.070 once near-duplicate groups are collapsed. Counter-evidence worth keeping: the fixed-2 m/s arm is
    SLOWER than the control (1.99 vs 2.53 m/s) and still halves unsafe, so lateral width matters independently
    of speed; and the 6 m/s baseline is the only arm that ever rolled the vehicle over.
 3. Effective sample size is 69-224 clusters, not 300 independent groups. At cluster level the main unsafe
    result holds (9 vs 0, p = 0.0039) but the failure result (p = 0.031) and the tilt advantage (p = 0.070) do
    not survive multiplicity correction. The model-gain result survives Holm (0.034) but not Bonferroni (0.062).
 4. Both models are badly overconfident: median predicted risk of the route actually driven vs its realised
    unsafe rate -- night-1 planner 0.083% vs 9.67%, night-2 0.0080% vs 0.33%. Treat the score as a ranking only.
 5. Deviation: the hazard set was declared at ">= 4 m margin, ~250 groups" and shipped at a 2 m rule (realised
    minimum 3.37 m) with 300 groups. On the 47 groups that do meet the declared rule the direction is unchanged
    (control 10.6% unsafe vs sampler 0.0%) but nothing reaches significance.
 6. "The injected anchors add nothing" is underpowered rather than established: only 81 of 300 groups are
    informative for that contrast (3 vs 0, p = 0.25).
