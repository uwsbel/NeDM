# Analytic energy baselines (night 2, Study 1 stage B)

Produced by `scripts/n2_energy_analytic.py` from `datasets/twin_{rigid,crm}.npz` (CPU only). The analytic model is
`scripts/traverse_wp9_analytic.py`'s route physics terms (route waypoints + commanded speed profile + f104 heightmap;
`resample`/`route_terms` imported unchanged) fitted by its NNLS with the within-group centred copy of the design
(weight alpha), per world. Fit rows: `split == 'train' & fail == 0`; (pool, alpha) chosen on the md5(group) % 5 == 0
dev fold by the original's score (0.5 MAE/30 kJ + 0.5 within-group regret/20 kJ), then refitted on all training rows.
Held-out: `split != 'train' & fail == 0` (goal reached). Target E_last = first-arrival cumulative W+ at the furthest
observed station (kJ). Curse ratio = mean over groups (>= 3 goal-reached routes) of (pred/true at the route with the
lowest predicted energy) / mean(pred/true over the group); < 1 means the pick looks cheaper than it is. curse diff is
the 09-07 form (difference instead of ratio).

## Held-out metrics (goal-reached routes, unsafe-but-reached included)

| world | model | n | groups>=3 | log-RMSE | MdAPE % | MAPE % | rho within group | rho pooled | curse ratio | curse diff | regret kJ | bias kJ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rigid | analytic (E_last) | 1136 | 111 | 0.833 | 71.8 | 107.4 | +0.396 | +0.502 | 0.878 | -0.355 | 66.9 | +12.9 |
| rigid | kJ/m x route length | 1136 | 111 | 0.931 | 80.9 | 109.9 | +0.052 | -0.016 | 0.945 | -0.108 | 361.7 | -65.1 |
| rigid | a + b x commanded time | 1136 | 111 | 0.889 | 72.6 | 112.3 | +0.162 | +0.142 | 0.864 | -0.416 | 135.3 | -38.1 |
| rigid | analytic, total W+ target (scored vs total W+) | 1136 | 111 | 0.832 | 71.9 | 107.4 | +0.398 | +0.504 | 0.880 | -0.353 | 66.6 | +13.5 |
| rigid | analytic, clean-only fit (unsafe == 0) | 1136 | 111 | 0.849 | 26.0 | 34.9 | +0.320 | +0.540 | 1.043 | +0.024 | 286.3 | -295.2 |
| crm | analytic (E_last) | 377 | 61 | 0.172 | 10.5 | 13.3 | +0.469 | +0.746 | 0.986 | -0.016 | 41.1 | -1.3 |
| crm | kJ/m x route length | 377 | 61 | 0.240 | 15.2 | 17.4 | -0.011 | +0.521 | 1.010 | +0.016 | 74.9 | -28.8 |
| crm | a + b x commanded time | 377 | 61 | 0.247 | 17.8 | 20.5 | -0.072 | +0.137 | 0.961 | -0.048 | 99.1 | -8.8 |
| crm | analytic, total W+ target (scored vs total W+) | 377 | 61 | 0.172 | 10.5 | 13.3 | +0.469 | +0.746 | 0.987 | -0.015 | 40.6 | -1.3 |
| crm | analytic, clean-only fit (unsafe == 0) | 377 | 61 | 0.172 | 10.5 | 13.3 | +0.469 | +0.746 | 0.987 | -0.015 | 41.1 | -1.4 |

## Held-out metrics, clean routes only (fail == 0 & unsafe == 0)

| world | model | n | groups>=3 | log-RMSE | MdAPE % | MAPE % | rho within group | rho pooled | curse ratio | curse diff | regret kJ | bias kJ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rigid | analytic (E_last) | 894 | 110 | 0.817 | 92.0 | 122.5 | +0.190 | +0.325 | 0.731 | -0.622 | 61.0 | +247.8 |
| rigid | kJ/m x route length | 894 | 110 | 0.820 | 96.3 | 122.7 | -0.057 | +0.005 | 0.913 | -0.167 | 108.5 | +229.6 |
| rigid | a + b x commanded time | 894 | 110 | 0.835 | 89.1 | 126.6 | -0.198 | -0.184 | 0.716 | -0.671 | 105.6 | +225.4 |
| rigid | analytic, clean-only fit | 894 | 110 | 0.294 | 19.6 | 23.5 | +0.447 | +0.623 | 0.987 | -0.015 | 40.2 | -9.7 |
| crm | analytic (E_last) | 377 | 61 | 0.172 | 10.5 | 13.3 | +0.469 | +0.746 | 0.986 | -0.016 | 41.1 | -1.3 |
| crm | kJ/m x route length | 377 | 61 | 0.240 | 15.2 | 17.4 | -0.011 | +0.521 | 1.010 | +0.016 | 74.9 | -28.8 |
| crm | a + b x commanded time | 377 | 61 | 0.247 | 17.8 | 20.5 | -0.072 | +0.137 | 0.961 | -0.048 | 99.1 | -8.8 |
| crm | analytic, clean-only fit | 377 | 61 | 0.172 | 10.5 | 13.3 | +0.469 | +0.746 | 0.987 | -0.015 | 41.1 | -1.4 |

## Fitted coefficients

**rigid** -- pool `tract`, alpha 0.0 (dev score 8.4535; dev log-RMSE optimum: `full` alpha 0.0); fit on 10976 rows, held-out 1136 rows / 111 groups (111 with >= 3).

- analytic (E_last), non-zero terms: tract2 3.314, tract_lowv 1.588, cross 1.548
- analytic (total W+ target): tract2 3.43, tract_lowv 1.591, cross 1.537
- analytic (clean-only fit, 8701 rows): tract2 3.583, tract_hiv 0.1879, corn 0.1031, cross 0.7076, one 55.84
- kJ/m x length: 10.757 kJ/m
- a + b x commanded time: a 253.0 kJ, b 13.03 kJ/s
- time baseline (secondary, for the time head): T = 1.32 s + 1.082 x commanded time; held-out log-RMSE 0.297, MdAPE 22.2 %, within-group rho +0.894

**crm** -- pool `full`, alpha 0.0 (dev score 2.4620; dev log-RMSE optimum: `full` alpha 0.0); fit on 4463 rows, held-out 377 rows / 91 groups (61 with >= 3).

- analytic (E_last), non-zero terms: roll 1.557, aero 25.35, time 2.321, cross 1.017, tract2 1.729, tract_lowv 0.007594, grade_lowv 0.8993, rough_v 0.5443, one 129.4
- analytic (total W+ target): roll 1.513, aero 25.71, time 2.326, cross 1.031, tract2 1.767, grade_lowv 0.9147, rough_v 0.5314, one 129.2
- analytic (clean-only fit, 4462 rows): roll 1.525, aero 25.41, time 2.327, cross 1.015, tract2 1.729, tract_lowv 0.005911, grade_lowv 0.9007, rough_v 0.544, one 129.9
- kJ/m x length: 11.739 kJ/m
- a + b x commanded time: a 509.0 kJ, b 2.81 kJ/s
- time baseline (secondary, for the time head): T = 1.82 s + 0.952 x commanded time; held-out log-RMSE 0.091, MdAPE 5.3 %, within-group rho +0.769

## Caveats

- **rigid**: the prescribed fit rows (`fail == 0`) mix 2275 unsafe-but-reached routes (median E_last 1246 kJ, 21 % of the rows) with clean successes (median 217 kJ). A level NNLS fit on that mixture sits far above the clean median, which is what the MdAPE of the main rows measures; the clean-only fit is the like-for-like baseline for an energy head trained on clean-driving (event-censored) targets, and the held-out set holds 242 unsafe-but-reached routes whose struggle energy no route-geometry model (or clean-driving head) predicts.
- **crm**: 1 unsafe-but-reached route(s) among the 4463 fit rows and 0 among the held-out rows; main and clean-only fits coincide.
- Selection picked alpha = 0 in both worlds: the within-group centred copy of the design (the original's within-mission weighting) does not help here.
- The original's `tract` term carries no standing-start kinetic energy on f104 (profiles begin at the commanded speed); the `tract_launch` pool offered `accel`/`kemax` and NNLS never used them.
- `resample` drops the last partial station (< 0.5 m, in the v -> 0 taper); goal-reached episodes end 5-9 stations short of station 95 (2.5 m goal circle), so E_last is the energy at the furthest observed station.

## Files

- `{world}_analytic_pred.npz`: id, group, split, source, profile, fail, unsafe, pred_analytic, pred_analytic_wplus,
  pred_analytic_clean, pred_len, pred_time, pred_T_from_cmd_time, true_E (E_last), true_wplus, true_T, t_cmd, route_len,
  last_station, heldout / train / dev masks, the selected terms per route (terms, terms_names). All rows of the twin
  file, same order; failed rows carry predictions too (their true_E is the energy at the censoring station).
- `summary.json`: counts, the full dev selection table, coefficients, all metrics.
- `route_terms.npz`: the 23 physics terms for every route id (cache; identical for both worlds).
