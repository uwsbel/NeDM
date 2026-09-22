# One adaptive route-risk model and one learned tracker: results (2026-09-21/22)

Draft 1, written 2026-09-22 01:20 while the last cluster jobs run. Sections marked PENDING are filled in as results land.
Everything here is on the f104 arena, both worlds (rigid ground and CRM deformable soil), branch `generalist_v1`, artefact
root `artifacts/traverse/generalist_20260921/` (K). The handoff plan is `PLAN_handoff.md`, the executed plan with its
review-driven amendments `PLAN.md`, the chronology `LOG.md`, the subsystem maps `scout/`, the module notes and independent
verifications `*/NOTES_*.md` and `*/VERIFY_*.md`.

## Summary

1. **Milestone A, offline.** One CNN-GRU risk model trained on both worlds with a two-second history of observable
   vehicle state and applied controls matches or beats the two specialists once the vehicle has moved for two seconds,
   and equals the pooled (label-free) model at a standing start. On the sealed test groups (within-group AUC vs the
   unsafe label, rigid / CRM): specialists 0.985 / 0.980 on established rows, history model 0.989 / 0.986; at startup
   0.984 / 0.976 vs 0.964 / 0.960. An oracle domain tag gives 0.985 / 0.981 and 0.983 / 0.979. Two seconds of motion
   identify the world with AUC 1.00 (hand features 0.74-0.77); the settled state at rest carries almost nothing (0.64).
2. **Milestone A, closed loop, standing start (A3).** On the 800-pair suite with CEM 4x64: CRM specialist 95.8 %, oracle
   tag 96.1 %, history model 93.8 %, rigid specialist 80.1 % goal reached on soil. The history model recovers 13.7 of the
   15.7 points between the wrong and the right specialist without a label, but misses the predeclared 3-point margin by
   0.4 (one-sided 95th percentile of the paired difference +3.4). On rigid ground every arm is at 99.8-100 %; the history
   model drives 14 % slower than the deployed rigid specialist there (bound 1.10 fails), the oracle 9 % slower.
3. **Milestone A, closed loop, established history (A5).** Rigid: from a moving anchor three seconds into the mission,
   with all arms trained on the same rows, the history model reaches 99.6 % vs 99.5 % for the rigid specialist and
   drives at 1.01 x its time (bound passes); the pooled and masked-history arms do the same, so on rigid ground the
   moving-anchor decision, not the history, removes the startup slowdown. CRM: history model 83.9 % vs CRM specialist
   83.0 % (rigid specialist 63.3 %), one-sided bound +0.5 (pass), time 0.99 x; masked-history and pooled arms match.
   With established history the shared model preserves both specialists without a label.
4. **Milestone B, tracking.** The tag-conditioned dynamics model passes all three validation gates in both worlds
   (stalled-escape 0.03 / 0.00, moving-displacement error 0.08 / 0.04, brake-response error 0.19 / 0.11, rigid / CRM).
   The PPO tracker trained inside it, evaluated in Chrono on the 423 designed test routes: on rigid ground it cuts the
   station cross-track on the 141 feasible routes from 0.200 m to 0.111 m (ratio 0.56, one-sided bound 0.66, passes the
   0.90 rule) with no safety cost, but tracks speed 21 % worse (bound 1.10 fails) and completes 5 points fewer of the
   282 hard routes. On soil it fails outright in round 1: 76.6 % vs 99.3 % completion on the feasible routes and a
   cross-track ratio of 2.3, although it completes more of the hard routes (7.8 vs 0.4 %) and is safer there. The
   second round (dynamics refit on 3,000 hold-mode perturbed episodes and 1,000 harvested policy failures, PPO retrain
   with a stronger speed term) passes every feasible-stratum rule on rigid ground (cross-track ratio 0.53, speed error
   ratio 0.75, completion 100 %, no unsafe events) and misses only the hard-route completion bound there (-0.7 points,
   5th percentile -4.3, with 6 points fewer unsafe events); on soil it reaches 91.5 % completion on the feasible
   routes (PID 99.3 %) with cross-track 0.379 vs 0.451 m and completes 8.5 points more of the hard routes, which is a
   large step from round 1 but still a fail on the soil replacement rules.
5. **Moving-prefix branch data (A4).** Rigid: 789 anchors x 3 continuations = 2,358 labelled rows; continuations of the
   same anchor disagree on the failure label in 24 % (clean-moving) and 32 % (low-progress) of anchors, so the branch
   choice carries information. Soil: 752 anchors x 3 = 2,256 rows; clean-moving continuations disagree in 43 % of
   anchors, but a prefix that has started to bog is lost whatever follows (97.7 % failure, 3 % disagreement). On these
   rows, where 480 anchors share the identical prefix in both worlds, the two-second history still identifies the
   world with AUC 0.99, so the earlier separation was not a route-selection artefact. Retrained on re-anchored plus
   both worlds' branch rows, the history model reaches 0.982 / 0.988 (rigid / CRM) on the enlarged established
   validation set against 0.973 / 0.980 for the same-row specialists.

Cost: 29 billed node-hours of the 100 cap at 03:45 (my own jobs, from the accounting records with the partition
billing weights; the account-wide ledger also moves with other users' jobs).

## 0. What was built and checked before any result

- Seven subsystem maps and a completeness critique (`scout/`), then a three-lens adversarial review of the plan
  (`scout/plan_review.md`, 38 findings, 5 blocking); every accepted fix is marked in `PLAN.md`.
- Nine modules on disjoint files, each with a self-test and an independent verifier (`*/VERIFY_*.md`), plus three fix
  rounds. Highlights: the new CRM collector is byte-identical to the campaign collector in native mode; the shared
  controller helpers keep torch out of every collector process (the numpy actor agrees with the torch policy to
  5e-7); the planner integration reproduces the night-2 picks bit-for-bit.
- Timing audit (`B_tracker/audit/`): the recorded action equals the action applied at the first physics substep of
  its interval; the follower refreshes inside the interval, but the applied mean differs from the recorded value by
  more than 0.1 in only 0.5 % (rigid) and 0.9 % (CRM) of intervals, almost all of them flagged by the hold filter.
  Brake onsets are never hold transitions in the recordings (the follower's exclusive throttle/brake switch), which is
  why the hold-mode collection exists.
- Determinism: two collectors re-driving ten recorded CRM episodes on MI350X agree byte-for-byte with each other and
  with nine of the ten recordings; one prolonged-blockage episode drifts 0.22 m. The CRM branch collection therefore
  replays every prefix on MI350X before branching.
- Dynamics cache (`B_tracker/cache_v1`): 39,235 episodes (15,235 CRM, 24,000 rigid), hold-mode retention 96-98 % of
  transitions, per-frame stall and hold flags, group split from the twin file (56 val groups for every gate, 55 test
  groups sealed).

## 1. Milestone A: one adaptive risk model

### 1.1 Data and identifiability

`A_adapt/datasets/mixed_reanchor.npz`: 115,868 re-anchored rows (58,424 rigid, 57,444 CRM, the same 15,024 episodes
per world), each with a 40-frame history of the 12 observable state columns and 3 applied actions cut from the raw
recording (all-masked at k = 0), and an 8-column privileged teacher context. Probe (`A_adapt/probe/probe.json`): a
small GRU on the two-second window separates the worlds with AUC 1.000 on the 29,966 moving anchors present in both
worlds (any 11 of the 15 channels suffice; a logistic on window means and standard deviations reaches 0.98); the
frame-0 state gives 0.64-0.67. The separation may partly reflect simulator differences (1 ms FSI vs 2 ms rigid physics)
rather than soil alone; the branch rows are the cleaner test of that and are re-probed in section 1.5.

### 1.2 Offline arms (validation groups, 5 seeds, within-group AUC vs unsafe; startup | established)

| arm | rigid | CRM |
|---|---|---|
| rigid specialist (same rows) | 0.974 | 0.977 | – |
| CRM specialist (same rows) | – | 0.987 | 0.985 |
| pooled, no label | 0.969 | 0.972 | 0.974 | 0.974 |
| oracle domain tag | 0.982 | 0.984 | 0.987 | 0.982 |
| history, risk loss only | 0.962 | 0.981 | 0.976 | 0.987 |
| history + domain head (H) | 0.968 | 0.982 | 0.973 | 0.987 |
| history, privileged teacher | 0.959 | 0.981 | 0.975 | 0.985 |

Sealed test groups, run once (startup | established): rigid specialist 0.984 | 0.985, CRM specialist 0.976 | 0.980,
pooled 0.969 | 0.969 / 0.953 | 0.968, oracle tag 0.983 | 0.985 / 0.979 | 0.981, H 0.964 | 0.989 / 0.960 | 0.986.
Calibration against the fitted unsafe label (Brier, established rows, test): H 0.046 / 0.041 vs specialists 0.059 /
0.047. Reading: with two seconds of motion the shared history model is as good as or better than either specialist and
the oracle; at a standing start it is the pooled model, one to two points behind the specialist of the right world.

### 1.3 Closed loop from a standing start (A3)

Suite: 600 fresh start-goal pairs (`f104_pair_group`, 2 m margin from every training pair) plus the 200 night-2 CRM
evaluation pairs, identical in both worlds, CEM 4x64 on depth-map corridors, picks hashed before driving, groups with
every arm present. Deploy ensembles are five seeds fitted on all training rows.

CRM (`A_adapt/a3/results_crm_A0A3.json`), goal reached:

| arm | all 800 | fresh 600 | reused 200 |
|---|---|---|---|
| CRM specialist S_crm | 95.8 | 94.8 | 98.5 |
| oracle tag T | 96.1 | 95.5 | 98.0 |
| history H | 93.8 | 93.0 | 96.0 |
| rigid specialist S_rigid | 80.1 | 80.7 | 78.5 |
| one-shot S_crm (256) | 91.6 | 91.8 | 91.0 |

Primary H vs S_crm: +2.0 points failure, 95 % one-sided bound +3.4 (margin 3.0: fail), fresh stratum +3.5, terrain
clustered CI [+0.7, +4.1]; T vs S_crm -0.3 (pass). H's picks are geometrically closer to S_crm's than to S_rigid's in
63.5 % of groups. Time: H 1.02 x S_crm.

Rigid (`A_adapt/a3/results_rigid_A0A3.json`): S_rigid 99.9, S_crm 100, T 100, H 99.9 % (all at the ceiling; the
goal-reached primary passes trivially). Time secondary vs the deployed rigid specialist: H 1.14 (fails 1.10), T 1.09,
S_crm 1.11: at a standing start the label-free model inherits the soil model's caution on rigid ground.

### 1.4 Closed loop with established history (A5)

Pass 1 drives a common three-second approach (the straight line to the goal, which every suite start already faces)
in both worlds; all 800 groups are moving at frame 60 (CRM median 2.9 m/s). Every arm then plans from the recorded
frame-60 state and pass 2 drives approach plus branch. All arms are trained on the same re-anchored rows.

Rigid (`A_adapt/a5/results_rigid_A5.json`, 800 groups): S'_rigid 99.5, S'_crm 98.9, H 99.6, H with the history masked
99.8, pooled 99.6, oracle 99.5 % goal reached; time vs S'_rigid: H 1.010 (bound 1.10: pass), masked 1.008, pooled
1.000, oracle 0.975, S'_crm 1.053. The A3 slowdown disappears from a moving anchor even without the history, so on
rigid ground the history has nothing to add beyond the moving state itself.

CRM (`A_adapt/a5/results_crm_A5.json`, 800 groups): S'_crm 83.0, S'_rigid 63.3, H 83.9, H masked 83.9, pooled 83.5,
oracle 84.4 % goal reached. Primary H vs S'_crm: -0.9 points failure (H better), one-sided 95th percentile +0.5
(margin 3.0: pass); fresh stratum +0.2 (pass); reused stratum alone +4.0 (n = 200, fails); terrain-clustered CI
[-2.8, +0.5]. Time H 0.99 x S'_crm. The cross-specialist gap is 19.7 points here. All absolute rates are lower than
in the standing-start protocol because every arm first drives the straight three-second approach (37 % of groups
start on grades above 12 degrees) and then plans from the moving state with the same-row models; specialist and
shared arms suffer alike, so the comparison stays paired. The masked-history and pooled arms match H (H vs masked
+0.0, H vs pooled -0.4), so closed loop the moving state at the decision carries the adaptation and the explicit
two-second window adds nothing measurable beyond it, even though offline it lifts the within-group AUC by one to two
points.

Milestone A verdict: with established history the label-free shared model preserves each specialist's goal-reaching
in both worlds (soil -0.9, rigid +0.1 points, times within 1-2 %); at a standing start it misses the 3-point margin on
soil by 0.4 at the 95th percentile (2.0 points behind) and is 14 % slower on rigid ground; the oracle tag shows the
shared network itself loses nothing.

### 1.5 Moving-prefix branch data (A4)

Rigid (in-job two-pass, `A_adapt/a4/rigid_runs`): 789 of 800 anchors replayed and branched (11 sampler rejections),
2,358 rows; clean-moving continuations fail 11 %, low-progress 53 %; within-anchor disagreement 24 % / 32 %; 2 m/s
continuations fail four times more often than 6 m/s ones (a within-anchor speed confound to keep in mind). Retraining
the history model on re-anchored plus branch rows: established 0.984 / 0.988 (val), startup 0.969 / 0.977.

CRM (two-pass on MI350X: 800 prefix replays, then 752 accepted anchors x 3 continuations, `A_adapt/a4/crm_pass2_runs`):
2,256 rows; clean-moving continuations fail 63.5 % with 43 % of anchors disagreeing across their three continuations;
low-progress continuations fail 97.7 % with only 3 % disagreement: once the soil has started to give way, no
continuation rescues the vehicle, as the plan review anticipated. Statuses: 1,444 soil breakthroughs, 545 goals, 266
blockages.

Probe on the branch rows of both worlds (4,614 rows; `A_adapt/probe_branch/probe_branch.json`): a small GRU on the
two-second prefix history separates the worlds with AUC 0.993 (val) / 0.989 (test), clean-moving 0.986 / 0.984,
low-progress 1.00 / 0.994; hand features 0.89 / 0.84. For 480 clean-moving anchors the prefix is the same episode cut
at the same frame in both worlds, so this separation is the physics response itself, not route selection.

Retraining on re-anchored plus both worlds' branch rows (`A_adapt/train/branch_v2`, validation groups, established set
now including the branch rows; startup | established): history model 0.973 | 0.982 rigid, 0.972 | 0.988 CRM; same-row
rigid specialist 0.977 | 0.973; same-row CRM specialist 0.987 | 0.980. With established history the shared model is
0.8-0.9 points above each specialist on the enlarged set, and the specialists lose ground on the branch rows while the
history model does not. These retrained ensembles were not driven closed loop (section 4).

## 2. Milestone B: learned tracker

### 2.1 Dynamics model

`B_tracker/nrd_tag/`: 4.9 M-parameter causal transformer over 16-frame tokens of normalised 17-D state, a 64-D token
from an 8x8 relative-elevation crop of the Chrono-frame terrain grid at the dead-reckoned pose, the 3-D action and the
domain one-hot; delta-state and power heads; trained 30,000 steps in 32 min on one MI350X on the 35,601 training
episodes with domain-balanced batches and hold-weighted targets. Validation on the 56 val groups at 60 fed-back steps:
rigid stalled-escape 0.027, moving displacement error 0.077, brake response error 0.194; CRM 0.000 / 0.042 / 0.112.
All gates pass (the early rigid brake gate failed at 6,000 steps and recovered by the end).

### 2.2 Tracker

PPO (rsl_rl) inside the frozen model: 2,048 fragments of 1-3 s from real training contexts in both worlds, observation
= 38-D route block + last 8 actions + last 8 observable states, action squash spanning the full box, PID-imitation
warm start on the runner's own normalised observations, progress term; 1,000 iterations in 34 min on the 5090; actor
exported to numpy and checked against torch (5e-7). The collectors run the policy without torch in either world.

### 2.3 Chrono evaluation, round 1 (`B_tracker/b0/results_*_b0.json`, 423 designed test routes, three arms per route)

Rigid, feasible stratum (141 routes the PID completes in both worlds): native PID cross-track 0.200 m, held PID 0.200,
policy 0.111 m; ratio 0.557, one-sided bound 0.655 (rule < 0.90: pass); completion -0.7 points (pass); unsafe +0.0
(pass); speed-error ratio 1.21, bound 1.29 (rule < 1.10: fail). Infeasible stratum (282): policy completion -5.3
points (fail), unsafe -3.9 (safer), cross-track not distinguishable. Held PID equals native PID on feasible routes and
is 4.6 points worse on the hard ones (the follower's inside-interval refresh helps in stalls).

CRM, feasible stratum: native PID 99.3 % goal, cross-track 0.451 m; held PID 94.3 %, 0.537 m; policy 76.6 %, 1.047 m
(ratio 2.32, fail on every rule). Infeasible stratum: policy 7.8 % vs 0.4 % completion (+7.4, pass), unsafe -6.7
(pass), but 30.7 s vs 16.2 s. Decision round 1: the tracker beats the PID on path following on rigid ground but not on
speed tracking or hard-route completion, and does not transfer to soft soil.

### 2.4 Round 2

Cache v3 (43,235 episodes) adds 1,500 hold-mode perturbed episodes per world (true 50 ms holds with brake taps) and
1,000 rigid drives of the round-1 policy on training routes (114 failures). The dynamics model was refit on it (gates:
rigid 0.035 / 0.092 / 0.214, CRM 0.000 / 0.045 / 0.113, all pass) and the tracker retrained with the speed-tracking
weight raised from 0.5 to 1.5 (`B_tracker/ppo_v2`, 22 min). The PID arms are the round-1 drives.

Rigid (`B_tracker/b0/results_rigid_b0v2.json`), feasible stratum: cross-track 0.106 m vs 0.200 m (ratio 0.53, bound
0.57: pass), completion 100 % vs 100 % (pass), unsafe 0 vs 0 (pass), speed error 0.348 vs 0.464 m/s (ratio 0.75, bound
0.80: pass). Hard stratum: completion 81.2 vs 81.9 % (-0.7 points, 5th percentile -4.3: misses the -3 bound), unsafe
10.3 vs 16.3 % (-6.0, safer), cross-track ratio 0.84. All 423 routes: cross-track ratio 0.80 (bound 0.90), completion
-0.5 (bound -2.8), unsafe -4.0, speed ratio 0.82. Verdict on rigid ground: the primary and every feasible-stratum rule
pass; the only miss is the hard-route completion bound, by 1.3 points at the 5th percentile with a point estimate of
-0.7, alongside 6 points fewer unsafe events.

CRM (`B_tracker/b0/results_crm_b0v2.json`), feasible stratum: completion 91.5 % vs 99.3 % (-7.8 points, bound -11.3:
fail), unsafe +7.8 (fail), cross-track 0.379 m vs 0.451 m (ratio 0.84, bound 0.99: not below 0.90), speed error ratio
0.98 (pass). Hard stratum: completion 8.9 % vs 0.4 % (+8.5, pass), unsafe -7.8 (pass), cross-track ratio 0.93. All 423
routes: completion 36.4 % vs 33.3 % (+3.1, 5th percentile +0.7), unsafe 63.6 % vs 66.2 %, cross-track ratio 0.92,
speed 0.95. Round 2 moved the soil result from a clear failure (76.6 % completion on feasible routes) to within 8
points of the PID with better path tracking and fewer unsafe events overall, but it does not meet the predeclared
replacement rules on soil. Verdict for Milestone B: passes on rigid ground except the hard-route completion bound;
fails on soil on the feasible-route completion and unsafe rules while beating the PID on the hard routes.

## 3. Honest caveats

- One arena, both worlds; everything is interpolation on f104. The fresh 600 pairs are new start-goal pairs, not new
  terrain.
- The startup condition of Milestone A is a label-free prior by construction; the history model cannot know the soil
  before it moves. The margin miss of 0.4 points at startup is real but small, and the oracle-tag arm shows the shared
  network itself loses nothing.
- Rigid goal-reaching is at the ceiling for every arm; only time separates the arms there.
- The history probe's perfect separation may include simulator artefacts; branch rows and the CRM established
  protocol are the cleaner evidence.
- Milestone B on soil is a negative result in round 1 despite a dynamics model that passes its gates; the imagination
  to Chrono gap on soil is not closed by this round.

## 4. Remaining work (not done in this session)

- Drive the branch-retrained history model closed loop (A3/A5 used the re-anchored-only deploy ensembles); the
  offline gain from branch rows is small, so the expected closed-loop change is small too.
- Soil tracker: a third round with policy failures harvested on soil training routes (only rigid failures were
  harvested), a longer imagination horizon check on soil, and a soil-specific reward term for wheel spin; the
  imagination-to-Chrono gap on soil is not closed.
- Startup on soil: the 0.4-point margin miss could be tested with a conservative startup rule (plan step 5) or a
  short common approach segment before the first decision, which the established-history protocol already shows to
  work.
- A second CRM arena and unseen soil parameters (the handoff's stated limit) remain untested.
- Integration step from the handoff (retrain the risk model under the learned tracker) was not started because the
  tracker does not yet replace the PID on soil.

## 5. Artefacts

`A_adapt/` (datasets, probe, train/{holdout_v1,holdout_test_v1,deploy_v1,branch_v1}, suite, a3, a4, a5, results),
`B_tracker/` (cache_v1..v3 manifests, audit, nrd_tag, ppo_v1, b0, b3, suite), `C_collectors/` (collector notes,
verifications, launch files), `scout/`, `PLAN.md`, `LOG.md`. Cluster roots: `C/generalist/*` (CRM drives),
`G/` (rigid drives, training, caches), `R/gen_v1/generalist_suite*` (rigid suite drives).
