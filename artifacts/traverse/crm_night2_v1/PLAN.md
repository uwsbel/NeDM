# Night 2 plan (written 2026-09-17 ~19:40 CDT, before any training run or planner drive of this session)

Two studies on the f104 route-risk pipeline, both worlds (rigid ground, CRM soil). Scout reports in `scout/` fix the
facts; prior results that must not be repeated are listed there (architecture tie at 8k rows/one lr; energy branch closed
negative on 09-07 because a learned work predictor lost to an analytic model through the optimiser's curse; every
recorded episode starts from rest, so the earlier "state is uninformative" result says nothing about velocity).

## Study 1 - network design at equal data

**Data (twin-matched).** CRM training split (13,821 routes, 1,089 groups) and the rigid twins of the same route ids
(`night2_v1/station_ds_all.npz`); ids without a twin dropped from both (~13.6k each). Held-out = the CRM val+test groups
(111 groups, 1,414 routes) and their rigid twins; the CRM split labels are used for both worlds (the rigid file called
these groups train). Dev fold = md5(group) % 5 inside the training groups. Second tier: full rigid (33.8k) for the two
best arms. Energy/time targets: `datasets/energy_{rigid,crm}.npz` (per-station first-arrival cumulative W+ and time).

**Stage A - architecture x world** (30 epochs, batch 256, AdamW, OneCycle; 5 seeds; GRU lr 2e-3 wd 1e-4; transformers
lr {1e-3, 2e-3} wd 0.05): A0 deployed CNN-GRU (256k) / A1 GRU hidden 120 (351k, parameter-matched to the transformer) /
A4 tokenised transformer d96 L2 h4 with a context token, sinusoidal position, pre-LN + final LN (356k) / A6 d128 L4 (932k)
/ A8 patch transformer without the CNN (4-station patches, 24 tokens, 893k) / A9 MLP mixer (173k, no cross-station
mixing). If time: A3 (existing conv-transformer + final LN, lr 1e-3), A5, A7, A8'. Read-outs per run: dev and held-out
pooled / within-group / same-speed AUC for goal-not-reached (and unsafe), route-choice avoidable-unsafe, seconds, params.
**Selection rule (declared now):** held-out within-group AUC of the 5-seed ensemble; ties (< 0.005) broken by
route-choice avoidable-unsafe; the answer is given per world.

**Stage B - energy head.** Target T2(a): per-station clean-driving positive work in kJ per metre x station spacing,
first-arrival increments, censored at the hazard event station (masked loss), softplus output, Huber on log1p; a time head
trained the same way. Joint training with the hazard head, lambda in {0.1, 0.3, 1.0}, on A0 and the best transformer of
stage A, both worlds, 5 seeds. Read-outs: hazard AUC (interference check), route-energy RMSE of log E and within-group
Spearman on goal-reached held-out routes, and the CURSE metric (predicted / true at the argmin vs over all routes),
against three baselines: the analytic work model (`scripts/traverse_wp9_analytic.py` refit on f104 W+), "kJ/m x length",
"a + b x commanded time". Energy is a score only behind the risk gate (never alone).

**Stage C - velocity input.** Re-anchored samples from every recorded episode: anchors every 2 s before the event,
k = 0 always included, at most 6 anchors per episode stratified by speed, remaining length >= 12 m, remaining route from
the projection point with the profile's commanded speeds, corridor over the remaining length (96 stations), labels by
the same event rule on the remaining part, ctx = full 22-d layout with the anchor state, energy/time targets relative
to the anchor. Leakage: anchors follow their episode, episodes their group. Arms: V0 velocity-blind (same rows), V1
chassis state in the context (vx, vy, roll, pitch, roll rate, pitch rate, yaw rate + geometry), V2 vx as a 7th input
plane, V3 (transformer) state in the context token. Read-outs within vx strata (0-1, 1-3, 3-6 m/s) on held-out anchors,
plus the start-anchored metric (must not regress) and a physics probe (P vs v0 on fixed uphill / cross-slope routes).
**Rigid moving-start check:** ~1,000 rigid episodes on held-out groups spawned WITH initial forward speed v0 in {0, 2, 4}
m/s (Chrono `SetInitFwdVel`, no braked settle) on re-anchored routes; AUC and calibration of V1/V2 vs V0 by v0, and the
sign of dP/dv0 against the realised outcome differences.

**Stage D.** Full-size rigid replicate of the two best arms, 8 seeds. **Deployed CRM ensembles** (best arm, energy
head, velocity input) retrained on the cluster (MI350X) for the planner study.

## Study 2 - planner

**Arms (offline picks on identical pools, hashed before any drive):** A one-shot 256 (deployed; CRM drives reused from
`crm_f104_v1/eval_v1`) / B iterated CEM 4x64 (equal budget) / C CEM 8x64 / D one-shot 512 (budget control) / E = B with
the pessimistic objective / F = B with expected cost C_fail*P + T (C_fail 60 s) / G gradient refinement from the pool's
top-16 (Adam, 100 steps, penalties, pessimistic keep-best, abstain below 0.3 logit gain), objective = logit / H expected
cost + analytic energy (lambda_E 1, 1 kJ = 0.2 s). Guards for G/H: leave-one-member-out, pessimistic re-score, rigid
ensemble second opinion, validator projection in float64, nearest-training-route failure rate.
**Where:** rigid first (CPU): `gen_v1/cases_test_f104` at fixed 2 m/s (theta in R^3) and speed-free on g216+g231; then CRM
on the 200 `crm_f104_v1/cases_eval` pairs (+ up to 200 fresh hazard pairs if time). One physics config per world.
**Primary contrasts (paired exact McNemar, Holm over the family; terrain-feature clustered CI reported):** B vs A (goal
not reached on CRM; unsafe on rigid); secondary C vs A, D vs A (iteration vs budget), E vs B, F vs B (cost of time
awareness), G vs A, H vs G. Secondary metrics: time to goal, realised positive work (`outcome.positive_work_kj`),
tilt > 30 deg, max speed, optimiser-created failures (pick P_pess < 1 % that fail).
**Power note:** the deployed CRM arm is at 91 %; at most ~11 of its 18 failures are addressable, so 3-5 points is the
realistic ceiling on 200 pairs; the predicted-logit gain (pilot: -0.8 at equal budget, 22/24 cases) is reported next to
the realised change either way.

## Compute and policy
Training sweeps run on the workstation RTX 5090 (30-120 s per run; the cluster is faster only when 8 runs are packed) -
a documented deviation from the train-on-cluster policy for sub-minute jobs; deployed ensembles and all Chrono drives
run on the AMD cluster (budget cap ~100 billed node-hours). Hourly alive cron at :23.
