# Night 2 (2026-09-17/18): network architecture and I/O, energy head, velocity input, iterated and gradient planners

Pre-registered plan: `PLAN.md` (sha256 in `PLAN.sha256`). Chronology: `LOG.md`. Scripts: `scripts/n2_*.py`, `scripts/planner_arms.py`,
`scripts/planner_grad_arms.py`, `scripts/f104_n2_iter.py`, `scripts/f104_n2_grad.py`, `scripts/rigid_moving_collect.py`. Figures: `figures/`.
All network studies use twin-matched datasets: the same 15,024 route ids on rigid ground and on CRM soil (13,629 training / 1,395 held-out rows,
held-out start-goal groups from night 1), so "at equal data" holds by construction.

## Summary

1. **Architecture (question 1a).** At matched data, no tokenise-then-transformer variant beats the current CNN-GRU. On CRM soil all 14 arms
   sit within 0.007 within-group AUC of each other (ensembles 0.977-0.984; CNN-GRU 0.983); on rigid ground the CNN-GRU is the best single
   network (0.914) and ties the best ensemble (0.916-0.917). Transformers cost 1.3-4x the training time for the same score. **Keep the CNN-GRU.**
2. **Velocity input (1b).** Adding the vehicle's velocity (vx, vy, yaw rate as context) to re-anchored samples raises held-out AUC on CRM from
   0.981 to 0.986 (ensemble; same on 1-3 m/s and 3-6 m/s moving anchors) and on rigid ground from 0.914 to 0.916 (noise). On a fresh
   ground-truth test (1,080 rigid drives from 360 held-out mid-route anchors at 0, 2, 4 m/s) the realised failure rate is flat in initial
   speed (36.4 / 36.7 / 35.0 %), the velocity-aware models score those drives slightly better (AUC 0.823-0.842 vs 0.812-0.815) but their per-anchor
   speed direction is at chance (19-22 of 43) and they all predict a downward trend that the drives do not show. **Velocity as an input is
   cheap and mildly helpful for scoring, but the network does not learn a usable speed effect from re-anchored data alone.**
3. **Energy output (1c).** A two-head output (positive motor-shaft work per metre, time per metre) trained jointly with the hazard at
   lambda = 1 reaches held-out log-RMSE 0.19-0.20 on CRM (analytic work model 0.17) and 0.24-0.26 on rigid ground (analytic 0.29), with
   within-group Spearman 0.61-0.69 vs 0.45-0.47 for the analytic model and no optimiser's curse (predicted/true at the argmin 0.97-1.00).
   The hazard AUC is unchanged (within 0.003). Lambda 0.1 is not enough (Spearman 0.39-0.47, negative for the GRU on CRM).
   **Energy can be predicted well enough to be a second score (rank correlation within a start-goal pair 0.6-0.7).**
4. **Iterated sampling vs one-shot (2a).** Re-sampling around the elite routes (CEM, 4 rounds x 64) with the same 256-evaluation budget cuts
   closed-loop failures on CRM soil from 9.0 % to 1.5 % on 200 held-out start-goal pairs (2 vs 17 discordant pairs, exact p = 0.0007; terrain-
   feature-clustered CI [-11.8, -4.1] points). The budget control (one-shot 512) does not help (10.0 %): the gain is from iteration, not
   from more samples. On rigid ground (fixed 2 m/s) the deployed sampler is already at the ceiling (99 % goal reached); CEM still removes
   the unsafe events (2.5 % -> 0.5 %, 0 vs 4 discordant, p = 0.13). On the sibling rigid arenas the deployed sampler is already near the ceiling (g216 unsafe 1.0 -> 0.0 %, g231 3.0 -> 2.5 %, both within noise).
5. **Gradient refinement (2b).** The network is differentiable end-to-end through the corridor extraction (bilinear map sampling), so
   a route parameterised by 3 lateral offsets and 4 speed deltas can be refined by gradient descent on the ensemble's route logit
   (arm G) or on expected cost = time + 120 s x P(fail) + analytic energy (arm H). Closed loop on the same 200 CRM pairs: **G reaches the
   goal on 99.5 % vs 91.0 % for the deployed one-shot sampler** (1 vs 18 discordant pairs, p = 1e-4; feature-clustered CI [-13.2, -4.5])
   and is 2.6 s faster on average; H reaches 98.0 % and is 6 s faster (p = 0.003 vs A). G vs the best sampling arm (CEM 8 x 64, 98.5 %):
   1 vs 3 discordant, not distinguishable at n = 200. Cost 23 s per start-goal pair on a shared RTX 5090 (vs ~1 s for one-shot).
   **The realised energy did not move**: H's analytic-model saving (516 -> 434 kJ predicted) came out as +2 kJ paired median in Chrono,
   because the analytic model's time term does not describe CRM work (traction on the same soil dominates). Energy as an objective needs
   the learned head from item 3, which was not deployed tonight.

Budget: 21.9 billed node-hours on the AMD cluster for the night (allocation at 436.4 / 1500 afterwards) plus the local RTX 5090.

## 1. Architecture at matched data (stage A)

Arms (all trained from scratch, 30 epochs, 5 seeds, held-out groups): CNN-GRU (current, 256 k params), CNN-GRU wide (351 k), CNN-GRU 2-layer
(473 k), MLP (173 k), CNN-tokens + transformer (353 k), transformer on per-station tokens d96 L2 / d96 L4 / d128 L4 (356 k / 580 k / 932 k),
patch-4 tokens d128 L4 (959 k). Two learning rates for the transformers; the better is reported. (`stageA/report.txt`, `figures/fig_arch.png`)

| arm | CRM single W (mean +- sd) | CRM ensemble W | rigid single W | rigid ensemble W | train s/seed (CRM) |
|---|---|---|---|---|---|
| CNN-GRU (current) | 0.979 +- 0.002 | 0.983 | 0.914 +- 0.002 | 0.916 | 12 |
| CNN-GRU wide | 0.982 +- 0.002 | 0.984 | 0.907 +- 0.004 | 0.910 | 13 |
| CNN-GRU 2 layers | 0.981 +- 0.002 | 0.984 | 0.910 +- 0.003 | 0.915 | 14 |
| MLP | 0.974 +- 0.003 | 0.977 | 0.906 +- 0.003 | 0.909 | 12 |
| CNN tokens + transformer | 0.978 +- 0.002 | 0.984 | 0.906 +- 0.004 | 0.914 | 40 |
| transformer d96 L2 | 0.978 +- 0.002 | 0.981 | 0.907 +- 0.004 | 0.916 | 16 |
| transformer d96 L4 | 0.975 +- 0.002 | 0.980 | 0.908 +- 0.002 | 0.914 | 70 |
| transformer d128 L4 | 0.976 +- 0.002 | 0.981 | 0.910 +- 0.005 | 0.917 | 50 |
| patch-4 tokens d128 L4 | 0.978 +- 0.003 | 0.980 | 0.904 +- 0.004 | 0.913 | 17 |

W = within-group AUC for goal-not-reached on held-out groups. Seed sd is 0.002-0.005, so differences below ~0.005 are noise. The
lowest-risk-pick failure rate on held-out groups (offline) is 0.21-0.24 for every CRM arm (oracle 0.18, random 0.73) and 0.00 for every
rigid arm. Read-out: the input representation (5-plane corridor, 96 stations) and the survival loss carry the result; the sequence model
on top is not the bottleneck at 13.6 k routes.

## 2. Velocity as an input (stage C) and the moving-start ground truth

Re-anchored datasets (`scripts/n2_reanchor_dataset.py`): each episode contributes its standing-start sample plus up to 3 samples anchored at
intermediate frames (2 s apart, before the event, stratified by speed), with the remaining route as the corridor and the anchor state as
context: 57,444 CRM / 58,424 rigid rows. Variants: geom (no vehicle state, current), vel (vx, vy, yaw rate + geometry), chassis (full 7-d
state + geometry), geom + vx plane (vx written into a 7th corridor plane). (`stageC/report.txt`, `figures/fig_velocity.png`)

| world | variant | ensemble W (all held-out rows) | standing start | moving 1-3 m/s | moving 3-6 m/s |
|---|---|---|---|---|---|
| CRM | geom (current) | 0.981 | 0.980 | 0.970 | 0.975 |
| CRM | vel | 0.986 | 0.984 | 0.976 | 0.983 |
| CRM | chassis | 0.986 | 0.981 | 0.977 | 0.979 |
| CRM | geom + vx plane | 0.986 | 0.981 | 0.976 | 0.985 |
| CRM | transformer d96 L2, chassis | 0.988 | 0.984 | 0.981 | 0.976 |
| CRM | transformer d96 L2, geom | 0.982 | 0.980 | 0.979 | 0.967 |
| rigid | geom (current) | 0.914 | 0.912 | 0.908 | 0.917 |
| rigid | vel (3 seeds) | 0.911 | 0.909 | 0.899 | 0.922 |
| rigid | transformer d96 L2, geom | 0.916 | 0.916 | 0.905 | 0.923 |
| rigid | chassis | 0.914 | 0.909 | 0.907 | 0.937 |
| rigid | geom + vx plane | 0.916 | 0.913 | 0.908 | 0.921 |

Moving-start ground truth (`moving_v1/`, `scripts/rigid_moving_collect.py`, 1,080 rigid drives, 360 held-out anchors x initial speed
0 / 2 / 4 m/s, measured first-frame speed 0 / 1.5 / 3.0 m/s): realised failure 36.4 / 36.7 / 35.0 %; most anchors are all-or-nothing across
speeds (only 43 of 360 change outcome with speed). Scoring those drives: velocity-blind 0.815 AUC (transformer 0.812), chassis 0.832, vx plane 0.837, vel 0.842, transformer
chassis 0.823; direction of the speed effect right in 19-22 of the 43 speed-sensitive anchors (chance); all velocity-aware models predict
mean P falling from 0.49-0.55 at 0 m/s to 0.38-0.41 at 4 m/s (`figures/fig_moving.png`). The likely cause is survivorship in the re-anchored
training rows: anchors that are moving fast are anchors that had been driving well. A direct moving-start training set would remove it.

## 3. Energy and time as outputs (stage B)

Targets: per-station first-arrival cumulative positive motor-shaft work W+ (engine speed x torque integrated over the positive part,
`scripts/n2_energy_targets.py`) and time, both as increments per station; heads: softplus kJ/m and s/m per station, Huber loss on
log1p increments over the clean-driving prefix + a route-level term, weight lambda against the hazard loss. Metrics on held-out clean
arrivals (goal reached without an unsafe event), like-for-like with the analytic baseline (`energy_baselines/`, `scripts/n2_energy_analytic.py`).
(`stageB/report.txt`, `figures/fig_energy.png`)

| world | arm | lambda | hazard W (ens.) | energy log-RMSE | within-group Spearman | curse ratio | time log-RMSE |
|---|---|---|---|---|---|---|---|
| CRM | analytic work model | - | - | 0.172 | 0.47 | 0.99 | - |
| CRM | transformer d128 L4 | 1 | 0.980 | 0.190 | 0.62 | 1.00 | 0.085 |
| CRM | transformer d96 L2 | 1 | 0.982 | 0.201 | 0.61 | 0.99 | 0.088 |
| CRM | CNN-GRU | 1 | 0.982 | 0.188 | 0.50 | 0.97 | 0.084 |
| CRM | CNN-GRU | 0.3 | 0.98 | 0.245 | 0.38 | 1.00 | 0.099 |
| CRM | CNN-GRU / d96 L2 / d128 L4 | 0.1 | 0.98 | 0.26-0.30 | -0.26 / 0.43 / 0.47 | 0.90-0.99 | 0.10-0.13 |
| rigid | analytic work model (clean fit) | - | - | 0.294 | 0.45 | - | - |
| rigid | transformer d128 L4 | 1 | 0.92 | 0.240 | 0.69 | 0.98 | 0.060 |
| rigid | CNN-GRU | 1 | 0.91 | 0.248 | 0.65 | 0.97 | 0.061 |
| rigid | transformer d96 L2 | 1 | 0.912 | 0.258 | 0.66 | 0.98 | 0.062 |
| rigid | d128 L4 / d96 L2 / CNN-GRU | 0.1-0.3 | 0.91-0.92 | 0.30-0.40 | 0.39-0.61 | 0.98-1.00 | 0.08-0.10 |

Single-network metrics (mean over 5 seeds); an ensemble-mean energy prediction would be somewhat better. Full table: `stageB/report.txt`.

## 4. Iterated sampling vs one-shot (planner study, CRM soil and rigid ground)

Arms (`scripts/planner_arms.py`): A one-shot 256 (deployed sampler, reproduces the locked night-1 picks 200/200); B CEM 4 rounds x 64
(same 256 evaluations); C CEM 8 x 64; D one-shot 512 (budget control for C); E CEM 4 x 64 with the pessimistic (max-member) objective;
F expected cost (time + 60 s x P). Picks hashed before driving (`planner/*/PICKS_LOCKED.sha256`), identical physics config to night 1.

CRM soil, 200 held-out start-goal pairs, 1,137 drives (`planner/eval_iter_crm/analysis.txt`, `figures/fig_planner.png`):

| arm | goal reached | median time | median W+ | mean predicted P | vs A: worse / better, exact p |
|---|---|---|---|---|---|
| A one-shot 256 (deployed) | 91.0 % | 17.0 s | 440 kJ | 0.051 | - |
| B CEM 4 x 64 | **98.5 %** | 14.7 s | 431 kJ | 0.019 | 2 / 17, p = 0.0007 |
| C CEM 8 x 64 | 98.5 % | 14.0 s | 421 kJ | 0.006 | 1 / 16, p = 0.0003 |
| D one-shot 512 | 90.0 % | 16.8 s | 418 kJ | 0.044 | 9 / 7, p = 0.80 |
| E CEM pessimistic | 95.5 % | 15.0 s | 423 kJ | 0.029 | 6 / 15, p = 0.08 |
| F expected cost (60 s) | 94.5 % | 9.8 s | 523 kJ | 0.040 | - |

B vs A: -7.5 points [-12.0, -3.5] pair bootstrap; clustered by the nearest of the 9 terrain features to the route midpoint: [-11.8, -4.1],
B better on all 8 features with any difference (sign test p = 0.008). C vs D (equal 512 budget): -8.5 points, p = 8e-5. The offline picture
matches: B lowers the pick logit by 0.69 mean / 0.34 median and improves 184/200 groups, D only 0.18 / 0.06 (119/200); the gain sits in the
worst quartile of one-shot picks. F is a different trade: 44 % faster routes for +3.5 points of failure and 19 % more energy.

Rigid ground, f104 fixed 2 m/s, 200 test pairs, 1,000 drives (`planner/eval_rigid_f104_fixed2/analysis.txt`): goal reached 99-100 % for
every arm; unsafe events A 2.5 / B 0.5 / C 2.0 / D 1.5 / E 0.5 % (B vs A 0 vs 4 discordant, p = 0.13). Small headroom: the rigid model already
rates its one-shot picks at P ~ 0.2 %.

## 5. Rigid arenas g216 and g231 (speed-free planner)

Same arms A-E with the frozen rigid-trained ensemble on the two sibling arenas from the night-1 generalisation study (200 test pairs each;
all arms of a group driven on one node because rigid Chrono is only node-deterministic).

g216 (`planner/eval_rigid_g216/analysis.txt`, 961 drives): goal reached 100 % for A, B, C, E and 99.5 % for D; unsafe events
A 1.0 / B 0.0 / C 0.0 / D 1.0 / E 0.5 % (B vs A: 0 vs 2 discordant, p = 0.5). Offline the iterated arms lowered the pick logit by 0.3-0.4,
but the deployed one-shot sampler is already at the ceiling on this arena.

g231 (`planner/eval_rigid_g231/analysis.txt`, 941 drives): goal reached 98.0-98.5 % for every arm; unsafe events A 3.0 / B 2.5 / C 1.5 /
D 2.0 / E 1.5 % (B vs A 2 vs 3 discordant, p = 1.0; C vs A 0 vs 3, p = 0.25). The 3-4 failures per arm are all on routes the rigid model
rates below 1 % risk (model blind spots on this arena, which the night-1 generalisation study already documented), so no sampler can remove them.

Read-out for question 2a: iterating the sampler is decisive where the one-shot picks are still risky (CRM soil: 9 % -> 1.5 % failures)
and harmless where they are not (rigid arenas: 0-3 % unsafe, differences within noise, never worse than one-shot).

## 6. Gradient refinement (arms G and H, CRM soil)

Method (`scripts/planner_grad_arms.py`, `scout/differentiable_planner.md`): a route is parameterised by 3 lateral offsets (cubic spline
about the straight line) and 4 speed deltas; corridor extraction is rewritten in torch (bilinear `grid_sample` on the elevation map,
grade and cross-slope by finite differences, speed plane from the profile) so the frozen CRM ensemble's route logit is differentiable in
the 7 parameters. From the 17 best one-shot pool routes, 60 Adam steps (lr 0.02 / 0.1) with curvature, lateral-bound and map-edge penalties;
candidates are re-validated with the frozen validator in float64 and chosen by the pessimistic (max-member) objective; the pool argmin is kept
if the gain is below 0.3 logit (G) or 1.0 s of expected cost (H). Cost per start-goal pair 22.8 s on a shared RTX 5090 (15.6 s refinement).

| arm | objective | picks changed | mean predicted P (pool -> new) | route time | analytic W+ | mean speed |
|---|---|---|---|---|---|---|
| G | ensemble logit | 163 / 200 (37 abstained) | 0.051 -> 0.003 | 18.4 -> 17.4 s | 516 -> 515 kJ | 3.3 -> 3.8 m/s |
| H | time + 120 s x P + analytic energy | 200 / 200 | 0.051 -> 0.007 | 18.4 -> 11.8 s | 516 -> 434 kJ | 3.3 -> 4.0 m/s |

Offline exploitation guards for G: leave-one-member-out held-out gain -2.3 logit (fit -2.4, 95 % of held-out members improve), the frozen
rigid-trained ensemble agrees with the direction of change on 61 % of changed picks, the 47 riskiest pool picks (mean P 0.21, 11 realised
night-1 failures among them) all changed and now score P 0.011. H trades risk for time: on 107/200 groups its pessimistic risk is worse
than the pool pick's.

Closed loop on CRM soil, 363 new drives + the 200 re-driven one-shot picks as arm A (`planner/eval_grad_crm/analysis.txt`,
`cluster_ci_fail.json`, `figures/fig_planner.png`):

| arm | goal reached | median time | median W+ | mean predicted P | vs A: worse / better, exact p | feature-clustered CI |
|---|---|---|---|---|---|---|
| A one-shot 256 (deployed) | 91.0 % | 17.0 s | 440 kJ | 0.051 | - | - |
| G gradient on risk logit | **99.5 %** | 13.9 s | 446 kJ | 0.003 | 1 / 18, p = 0.0001 | [-13.2, -4.5], 8/0 features |
| H gradient on time + 120 s x P + analytic W+ | 98.0 % | 12.0 s | 443 kJ | 0.007 | 3 / 17, p = 0.0026 | [-12.3, -2.3], 7/1 features |

The one G failure (group 0111, predicted P 0.001) bogs down on a route the ensemble rates as safe; H fails there too and on three other
groups (two soil break-throughs at P 0.006-0.08). Against the sampling arms on the same groups (identical A drives, 200/200): G vs CEM 4 x 64
1 vs 3 discordant (p = 0.63), G vs CEM 8 x 64 1 vs 3 (p = 0.63), G vs pessimistic CEM 1 vs 9 (p = 0.02); H vs CEM 4 x 64 4 vs 3 (n.s.).

Paired energy and time on groups where both arms reached the goal (positive motor-shaft work from the drives):

| contrast | n | realised W+ (mean) | paired median difference | time (mean) |
|---|---|---|---|---|
| G vs A | 181 | 462 vs 474 kJ | +0 kJ [-6, +0] | 15.5 vs 18.1 s |
| H vs A | 179 | 458 vs 472 kJ | +2 kJ [-6, +13] | 12.0 vs 18.1 s |
| CEM 4 x 64 vs A | 180 | 448 vs 470 kJ | -8 kJ [-23, 0] | 15.3 vs 18.2 s |
| CEM 8 x 64 vs A | 181 | 445 vs 473 kJ | -18 kJ [-30, -8] | 14.6 vs 18.2 s |
| F expected cost (60 s) vs A | 179 | 527 vs 470 kJ | +65 kJ [+45, +78] | 10.1 vs 18.2 s |

Read-out for question 2b: gradient refinement of the risk logit works in closed loop and is at least as good as the best iterated sampler;
the analytic energy term inside the objective buys time, not energy. Risk-minimising arms lower realised energy as a side effect (fewer
steep stations); the fast-route arm F costs 14 % more energy. An energy objective should use the learned energy head (section 3), which
ranks routes within a start-goal pair at Spearman 0.6-0.7 where the analytic model reaches 0.45-0.47.

## Artefacts

- Datasets: `datasets/twin_{crm,rigid}.npz` (15,024 rows each), `datasets/reanchor_{crm,rigid}.npz` (57,444 / 58,424),
  `datasets/energy_{crm,rigid}.npz` (per-station W+ and time targets).
- Training results: `stageA/`, `stageB/`, `stageC/` (`<tag>.json` per ensemble, `<tag>_logits.npz`, stage C checkpoints `<tag>_s{k}.pt`),
  tables `stage*/report.txt`; cluster copy of the trainer in `cluster_train/` (job 425457 on mi3501x).
- Moving-start ground truth: `moving_v1/` (cases, routes, tasks, `out/runs` 1,080 drives, `moving_analysis.json`).
- Planner picks and drives: `planner/iter_crm`, `planner/iter_rigid_{f104_fixed2,g216,g231}`, `planner/grad_crm` (+ `PICKS_LOCKED.sha256`),
  drives and analyses in `planner/eval_*`.
- Energy baseline: `energy_baselines/`.

## Remaining work

- Not done tonight: deploying a velocity-aware or energy-head ensemble in a closed-loop replanning test; a moving-start training set
  (the re-anchored data carries a survivorship bias, section 2); the full-size rigid replicate of stage A (stage D); a gradient arm with the
  learned energy head as the energy term (H used the analytic model, which did not transfer); confirming G on a second CRM arena.
- Closed-loop numbers are on one memorisable arena (f104) with the interpolation caveat of night 1; the terrain-feature-clustered intervals
  are the honest ones (9 clusters).
