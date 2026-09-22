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
   moving-anchor decision, not the history, removes the startup slowdown. CRM: PENDING (drives running).
4. **Milestone B, tracking.** The tag-conditioned dynamics model passes all three validation gates in both worlds
   (stalled-escape 0.03 / 0.00, moving-displacement error 0.08 / 0.04, brake-response error 0.19 / 0.11, rigid / CRM).
   The PPO tracker trained inside it, evaluated in Chrono on the 423 designed test routes: on rigid ground it cuts the
   station cross-track on the 141 feasible routes from 0.200 m to 0.111 m (ratio 0.56, one-sided bound 0.66, passes the
   0.90 rule) with no safety cost, but tracks speed 21 % worse (bound 1.10 fails) and completes 5 points fewer of the
   282 hard routes. On soil it fails outright in round 1: 76.6 % vs 99.3 % completion on the feasible routes and a
   cross-track ratio of 2.3, although it completes more of the hard routes (7.8 vs 0.4 %) and is safer there. A second
   round (dynamics refit on 3,000 hold-mode perturbed episodes and 1,000 harvested policy failures, then PPO retrain)
   is PENDING.
5. **Moving-prefix branch data (A4).** Rigid: 789 anchors x 3 continuations = 2,358 labelled rows; continuations of the
   same anchor disagree on the failure label in 24 % (clean-moving) and 32 % (low-progress) of anchors, so the branch
   choice carries information. Adding them raises the history model on the enlarged established set (0.984 / 0.988 val)
   while the rigid specialist drops (0.977 to 0.972). CRM: 752 anchors accepted from the prefix replays (46 class
   mismatches, 2 sampler failures); 2,256 drives PENDING.

Cost so far: 15 billed node-hours of the 100 cap at 01:20 (my own jobs, from the accounting records; the account-wide
ledger also moves with other users' jobs).

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

CRM: PENDING (4,800 drives running).

### 1.5 Moving-prefix branch data (A4)

Rigid (in-job two-pass, `A_adapt/a4/rigid_runs`): 789 of 800 anchors replayed and branched (11 sampler rejections),
2,358 rows; clean-moving continuations fail 11 %, low-progress 53 %; within-anchor disagreement 24 % / 32 %; 2 m/s
continuations fail four times more often than 6 m/s ones (a within-anchor speed confound to keep in mind). Retraining
the history model on re-anchored plus branch rows: established 0.984 / 0.988 (val), startup 0.969 / 0.977.

CRM: PENDING (752 anchors, 2,256 drives). The probe on branch rows and the retrained closed-loop arms are round-two
items if time allows.

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

### 2.4 Round 2: PENDING

Cache v3 (43,235 episodes) adds 1,500 hold-mode perturbed episodes per world (true 50 ms holds with brake taps) and
1,000 rigid drives of the round-1 policy on training routes (114 failures). The dynamics refit is running; the PPO
retrain will raise the speed-tracking weight. Results replace this section when the Chrono re-evaluation is back.

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

## 4. Artefacts

`A_adapt/` (datasets, probe, train/{holdout_v1,holdout_test_v1,deploy_v1,branch_v1}, suite, a3, a4, a5, results),
`B_tracker/` (cache_v1..v3 manifests, audit, nrd_tag, ppo_v1, b0, b3, suite), `C_collectors/` (collector notes,
verifications, launch files), `scout/`, `PLAN.md`, `LOG.md`. Cluster roots: `C/generalist/*` (CRM drives),
`G/` (rigid drives, training, caches), `R/gen_v1/generalist_suite*` (rigid suite drives).
