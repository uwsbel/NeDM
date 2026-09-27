## T1. Rates on the 8 unseen arenas (rigid, HMMWV, 2,000 groups; near = 4 arenas closest to f104, spread = 4 farther)

| arm (code) | plain label | mode | goal not reached: all / near / spread | unsafe: all / near / spread | of which backward-only (all) | median time, s (all) |
|---|---|---|---|---|---|---|
| M1a_free | f104 only (ensemble a) | speed free | 0.60 / 0.20 / 1.00 | 1.05 / 0.50 / 1.60 | 0.45 | 12.5 |
| M1b_free | f104 only (ensemble b) | speed free | 0.65 / 0.30 / 1.00 | 1.45 / 0.80 / 2.10 | 0.80 | 13.2 |
| M1_free | f104 only (mean of ensembles a and b) | speed free | 0.62 / 0.25 / 1.00 | 1.25 / 0.65 / 1.85 | 0.62 | - |
| M2_free | two arenas, same total data (f104 + g203) | speed free | 0.25 / 0.10 / 0.40 | 0.60 / 0.60 / 0.60 | 0.35 | 12.6 |
| M3a_free | three arenas, same total data (ensemble a) | speed free | 0.40 / 0.30 / 0.50 | 0.65 / 0.50 / 0.80 | 0.25 | 15.2 |
| M3b_free | three arenas, same total data (ensemble b) | speed free | 0.25 / 0.00 / 0.50 | 0.65 / 0.40 / 0.90 | 0.40 | 15.2 |
| M3_free | three arenas, same total data (mean of a and b) | speed free | 0.33 / 0.15 / 0.50 | 0.65 / 0.45 / 0.85 | 0.33 | - |
| A3_free | three arenas, all data | speed free | 0.30 / 0.10 / 0.50 | 0.70 / 0.30 / 1.10 | 0.40 | 13.5 |
| straight6 | straight route at 6 m/s | speed free | 1.60 / 1.30 / 1.90 | 2.10 / 2.00 / 2.20 | 0.50 | 7.7 |
| M1a_fx2 | f104 only (ensemble a) | fixed 2 m/s | 3.00 / 2.70 / 3.30 | 8.25 / 6.50 / 10.00 | 5.25 | 21.2 |
| M1b_fx2 | f104 only (ensemble b) | fixed 2 m/s | 3.15 / 3.30 / 3.00 | 8.40 / 7.60 / 9.20 | 5.25 | 21.2 |
| M1_fx2 | f104 only (mean of ensembles a and b) | fixed 2 m/s | 3.08 / 3.00 / 3.15 | 8.33 / 7.05 / 9.60 | 5.25 | - |
| M2_fx2 | two arenas, same total data (f104 + g203) | fixed 2 m/s | 3.50 / 3.60 / 3.40 | 6.60 / 6.90 / 6.30 | 3.10 | 21.0 |
| M3a_fx2 | three arenas, same total data (ensemble a) | fixed 2 m/s | 2.75 / 2.80 / 2.70 | 6.30 / 6.50 / 6.10 | 3.55 | 21.0 |
| M3b_fx2 | three arenas, same total data (ensemble b) | fixed 2 m/s | 3.30 / 3.10 / 3.50 | 6.05 / 5.80 / 6.30 | 2.75 | 21.1 |
| M3_fx2 | three arenas, same total data (mean of a and b) | fixed 2 m/s | 3.02 / 2.95 / 3.10 | 6.17 / 6.15 / 6.20 | 3.15 | - |
| A3_fx2 | three arenas, all data | fixed 2 m/s | 2.95 / 2.60 / 3.30 | 5.35 / 4.90 / 5.80 | 2.40 | 20.8 |
| straight2 | straight route at 2 m/s | fixed 2 m/s | 26.05 / 28.60 / 23.50 | 54.50 / 56.00 / 53.00 | 28.45 | 22.6 |

Composite rows (M1, M3) are per-group means of the two ensembles (0, 0.5 or 1 per group); they have no times.

## T2. Fixed 2 m/s unsafe per unseen arena (%, 250 groups each), with the designed-route feasibility of the arena

| arena | role | distance to nearest training arena | map error rmse, m | M1a | M1b | M2 | M3a | M3b | A3 | straight 2 m/s | groups with no safe designed 2 m/s route | groups with no goal-reaching designed 2 m/s route |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g260 | near | 0.61 | 0.060 | 1.6 | 4.4 | 2.4 | 1.2 | 2.0 | 2.4 | 39.2 | 18.4 | 5.6 |
| g271 | near | 0.74 | 0.065 | 8.4 | 8.0 | 5.2 | 7.6 | 7.2 | 5.2 | 49.2 | 24.4 | 4.0 |
| g251 | near | 0.78 | 0.068 | 6.0 | 7.6 | 9.2 | 7.2 | 5.2 | 5.2 | 68.8 | 27.6 | 4.0 |
| g247 | near | 0.72 | 0.071 | 10.0 | 10.4 | 10.8 | 10.0 | 8.8 | 6.8 | 66.8 | 29.6 | 4.8 |
| g258 | spread | 0.88 | 0.085 | 13.2 | 12.8 | 8.0 | 9.2 | 8.4 | 7.6 | 47.6 | 15.6 | 0.8 |
| g268 | spread | 0.92 | 0.094 | 8.4 | 7.6 | 4.0 | 2.8 | 4.0 | 5.6 | 48.0 | 12.4 | 4.0 |
| g263 | spread | 1.19 | 0.059 | 8.8 | 9.6 | 3.2 | 5.6 | 6.0 | 6.0 | 66.4 | 20.8 | 3.2 |
| g241 | spread | 1.34 | 0.062 | 9.6 | 6.8 | 10.0 | 6.8 | 6.8 | 4.0 | 50.0 | 13.6 | 0.8 |

Same table for goal not reached at fixed 2 m/s:

| arena | M1a | M1b | M2 | M3a | M3b | A3 | straight 2 m/s |
|---|---|---|---|---|---|---|---|
| g260 | 1.2 | 1.6 | 1.2 | 0.8 | 1.6 | 1.6 | 21.2 |
| g271 | 4.0 | 4.8 | 2.8 | 2.4 | 3.2 | 2.0 | 30.0 |
| g251 | 1.2 | 1.6 | 4.8 | 3.6 | 2.0 | 2.4 | 28.4 |
| g247 | 4.4 | 5.2 | 5.6 | 4.4 | 5.6 | 4.4 | 34.8 |
| g258 | 3.2 | 2.8 | 5.6 | 3.6 | 3.6 | 4.4 | 17.2 |
| g268 | 4.8 | 3.6 | 1.6 | 1.2 | 1.2 | 1.6 | 22.8 |
| g263 | 3.2 | 3.6 | 0.8 | 2.8 | 4.4 | 4.4 | 30.0 |
| g241 | 2.0 | 2.0 | 5.6 | 3.2 | 4.8 | 2.8 | 24.0 |

Speed free, goal not reached / unsafe per arena (%):

| arena | M1a | M1b | M2 | M3a | M3b | A3 | straight 6 m/s |
|---|---|---|---|---|---|---|---|
| g260 | 0.0 / 0.4 | 0.0 / 0.4 | 0.0 / 0.4 | 0.4 / 0.8 | 0.0 / 1.2 | 0.0 / 0.0 | 2.0 / 3.6 |
| g271 | 0.4 / 0.8 | 0.8 / 1.6 | 0.4 / 0.8 | 0.0 / 0.4 | 0.0 / 0.0 | 0.4 / 1.2 | 2.0 / 2.4 |
| g251 | 0.4 / 0.4 | 0.0 / 0.8 | 0.0 / 0.8 | 0.0 / 0.0 | 0.0 / 0.4 | 0.0 / 0.0 | 0.4 / 1.2 |
| g247 | 0.0 / 0.4 | 0.4 / 0.4 | 0.0 / 0.4 | 0.8 / 0.8 | 0.0 / 0.0 | 0.0 / 0.0 | 0.8 / 0.8 |
| g258 | 2.0 / 3.6 | 1.2 / 4.8 | 0.4 / 1.2 | 0.4 / 1.6 | 0.0 / 0.8 | 0.8 / 2.4 | 0.4 / 0.4 |
| g268 | 0.0 / 0.4 | 1.2 / 1.6 | 0.8 / 0.8 | 0.8 / 0.8 | 0.4 / 0.8 | 0.0 / 0.4 | 4.8 / 5.2 |
| g263 | 0.8 / 0.8 | 1.2 / 1.6 | 0.4 / 0.4 | 0.4 / 0.4 | 1.2 / 1.6 | 0.8 / 1.2 | 0.0 / 0.4 |
| g241 | 1.2 / 1.6 | 0.4 / 0.4 | 0.0 / 0.0 | 0.4 / 0.4 | 0.4 / 0.4 | 0.4 / 0.4 | 2.4 / 2.8 |

## T3. The declared family of four (PLAN 7.3; Holm at 0.05 over one-sided cluster-bootstrap p; 4,000 resamples)

| test | comparison (plain) | groups | rate test / reference, % | difference, points | 90 % cluster interval | 95 % group interval | one-sided p (cluster) | Holm-adjusted p | decision |
|---|---|---|---|---|---|---|---|---|---|
| P1_soil_M3_vs_M1 | three arenas, same total data (mean of a and b) vs f104 only (mean of ensembles a and b), soil speed free, goal not reached | 1000 | 9.70 / 12.05 | -2.35 | [-3.7, -1.1] | [-4.0, -0.7] | 0.002 | 0.0037 | improves |
| P2_soil_A3_vs_M1 | three arenas, all data vs f104 only (mean of ensembles a and b), soil speed free, goal not reached | 1000 | 9.40 / 12.05 | -2.65 | [-4.0, -1.3] | [-4.5, -1.0] | 0.001 | 0.0037 | improves |
| P3_rigid_fx2_M3_vs_M1 | three arenas, same total data (mean of a and b) vs f104 only (mean of ensembles a and b), rigid fixed 2 m/s, unsafe | 2000 | 6.17 / 8.33 | -2.15 | [-3.4, -1.0] | [-3.1, -1.2] | 0.001 | 0.0037 | improves |
| P4_rigid_fx2_A3_vs_M1 | three arenas, all data vs f104 only (mean of ensembles a and b), rigid fixed 2 m/s, unsafe | 2000 | 5.35 / 8.33 | -2.98 | [-4.1, -1.9] | [-4.0, -1.9] | < 0.001 | 0.0010 | improves |

Rigid-only run of the frozen spec (soil entered at p = 1, so these adjusted p are upper bounds): P3_rigid_fx2_M3_vs_M1 Holm 0.0045 -> improves; P4_rigid_fx2_A3_vs_M1 Holm 0.0010 -> improves.

## T4. Rigid primary tests per unseen arena (difference in unsafe points, 95 % group-bootstrap interval, groups test worse / better)

| arena | role | three arenas same total (M3) vs f104 only (M1) | three arenas all data (A3) vs f104 only (M1) |
|---|---|---|---|
| g260 | near | -1.4 [-2.8, -0.2] (2 / 8) | -0.6 [-2.6, +1.2] (4 / 7) |
| g271 | near | -0.8 [-3.0, +1.4] (12 / 17) | -3.0 [-5.6, -0.6] (8 / 20) |
| g251 | near | -0.6 [-3.2, +2.0] (16 / 16) | -1.6 [-4.4, +1.2] (8 / 16) |
| g247 | near | -0.8 [-4.0, +2.4] (19 / 21) | -3.4 [-6.6, -0.2] (11 / 26) |
| g258 | spread | -4.2 [-7.6, -0.8] (15 / 35) | -5.4 [-8.8, -2.0] (10 / 32) |
| g268 | spread | -4.6 [-7.4, -2.0] (6 / 21) | -2.4 [-5.2, +0.4] (7 / 18) |
| g263 | spread | -3.4 [-6.6, -0.4] (11 / 24) | -3.2 [-6.6, +0.0] (11 / 25) |
| g241 | spread | -1.4 [-4.2, +1.6] (17 / 27) | -4.2 [-7.0, -1.4] (8 / 32) |

M3 vs M1: arenas better 8 / worse 0; random-effects pooled -1.93 [-2.96, -0.90] (tau 0.73 points, I2 25 %); near -0.90 [-2.21, +0.29], spread -3.40 [-6.23, -1.07], near minus spread +2.50 [-0.22, +5.58]; per-arena effect vs distance to the nearest training arena: slope -2.05 points per unit distance, Spearman -0.55 (8 arenas); vs map error: Spearman -0.26; cluster wins / losses 31 / 13 of 81 clusters; design-feature clustering 90 % [-3.4, -1.0], p < 0.001.

A3 vs M1: arenas better 8 / worse 0; random-effects pooled -2.64 [-3.76, -1.53] (tau 0.74 points, I2 21 %); near -2.15 [-3.49, -0.86], spread -3.80 [-6.08, -1.70], near minus spread +1.65 [-0.94, +4.26]; per-arena effect vs distance to the nearest training arena: slope -3.02 points per unit distance, Spearman -0.48 (8 arenas); vs map error: Spearman -0.19; cluster wins / losses 40 / 9 of 81 clusters; design-feature clustering 90 % [-4.2, -1.9], p < 0.001.

## T5. Post-hoc robustness of the rigid primary tests (decided after seeing T3; descriptive, unadjusted)

| comparison (fixed 2 m/s, 8 unseen arenas) | label | rate test / reference, % | difference | 90 % cluster interval | groups test worse / better | exact McNemar two-sided |
|---|---|---|---|---|---|---|
| M3a_fx2 (three arenas, same total data (ensemble a)) vs M1a_fx2 (f104 only (ensemble a)) | unsafe | 6.30 / 8.25 | -1.95 | [-3.4, -0.6] | 62 / 101 | 0.0028 |
| M3a_fx2 (three arenas, same total data (ensemble a)) vs M1a_fx2 (f104 only (ensemble a)) | fail | 2.75 / 3.00 | -0.25 | [-1.2, +0.7] | 37 / 42 | 0.65 |
| M3a_fx2 (three arenas, same total data (ensemble a)) vs M1a_fx2 (f104 only (ensemble a)) | backward_only | 3.55 / 5.25 | -1.70 | [-2.8, -0.7] | 48 / 82 | 0.0036 |
| M3a_fx2 (three arenas, same total data (ensemble a)) vs M1b_fx2 (f104 only (ensemble b)) | unsafe | 6.30 / 8.40 | -2.10 | [-3.5, -0.8] | 67 / 109 | 0.0019 |
| M3a_fx2 (three arenas, same total data (ensemble a)) vs M1b_fx2 (f104 only (ensemble b)) | fail | 2.75 / 3.15 | -0.40 | [-1.4, +0.5] | 38 / 46 | 0.45 |
| M3a_fx2 (three arenas, same total data (ensemble a)) vs M1b_fx2 (f104 only (ensemble b)) | backward_only | 3.55 / 5.25 | -1.70 | [-2.8, -0.6] | 51 / 85 | 0.0045 |
| M3b_fx2 (three arenas, same total data (ensemble b)) vs M1a_fx2 (f104 only (ensemble a)) | unsafe | 6.05 / 8.25 | -2.20 | [-3.7, -0.7] | 66 / 110 | 0.0011 |
| M3b_fx2 (three arenas, same total data (ensemble b)) vs M1a_fx2 (f104 only (ensemble a)) | fail | 3.30 / 3.00 | +0.30 | [-0.8, +1.3] | 51 / 45 | 0.61 |
| M3b_fx2 (three arenas, same total data (ensemble b)) vs M1a_fx2 (f104 only (ensemble a)) | backward_only | 2.75 / 5.25 | -2.50 | [-3.7, -1.4] | 40 / 90 | 1.4e-05 |
| M3b_fx2 (three arenas, same total data (ensemble b)) vs M1b_fx2 (f104 only (ensemble b)) | unsafe | 6.05 / 8.40 | -2.35 | [-3.8, -1.0] | 66 / 113 | 0.00055 |
| M3b_fx2 (three arenas, same total data (ensemble b)) vs M1b_fx2 (f104 only (ensemble b)) | fail | 3.30 / 3.15 | +0.15 | [-0.9, +1.2] | 50 / 47 | 0.84 |
| M3b_fx2 (three arenas, same total data (ensemble b)) vs M1b_fx2 (f104 only (ensemble b)) | backward_only | 2.75 / 5.25 | -2.50 | [-3.5, -1.5] | 39 / 89 | 1.2e-05 |
| A3_fx2 (three arenas, all data) vs M1a_fx2 (f104 only (ensemble a)) | unsafe | 5.35 / 8.25 | -2.90 | [-4.2, -1.7] | 52 / 110 | 6.1e-06 |
| A3_fx2 (three arenas, all data) vs M1a_fx2 (f104 only (ensemble a)) | fail | 2.95 / 3.00 | -0.05 | [-1.0, +0.9] | 42 / 43 | 1 |
| A3_fx2 (three arenas, all data) vs M1a_fx2 (f104 only (ensemble a)) | backward_only | 2.40 / 5.25 | -2.85 | [-4.1, -1.8] | 35 / 92 | 4.4e-07 |
| A3_fx2 (three arenas, all data) vs M1b_fx2 (f104 only (ensemble b)) | unsafe | 5.35 / 8.40 | -3.05 | [-4.3, -1.9] | 46 / 107 | 8.9e-07 |
| A3_fx2 (three arenas, all data) vs M1b_fx2 (f104 only (ensemble b)) | fail | 2.95 / 3.15 | -0.20 | [-1.0, +0.7] | 38 / 42 | 0.74 |
| A3_fx2 (three arenas, all data) vs M1b_fx2 (f104 only (ensemble b)) | backward_only | 2.40 / 5.25 | -2.85 | [-4.1, -1.7] | 33 / 90 | 2.7e-07 |
| M2_fx2 (two arenas, same total data (f104 + g203)) vs M1a_fx2 (f104 only (ensemble a)) | unsafe | 6.60 / 8.25 | -1.65 | [-3.2, -0.2] | 72 / 105 | 0.016 |
| M2_fx2 (two arenas, same total data (f104 + g203)) vs M1a_fx2 (f104 only (ensemble a)) | fail | 3.50 / 3.00 | +0.50 | [-0.6, +1.6] | 56 / 46 | 0.37 |
| M2_fx2 (two arenas, same total data (f104 + g203)) vs M1a_fx2 (f104 only (ensemble a)) | backward_only | 3.10 / 5.25 | -2.15 | [-3.5, -1.0] | 43 / 86 | 0.00019 |
| M1a_fx2 (f104 only (ensemble a)) vs M1b_fx2 (f104 only (ensemble b)) | unsafe | 8.25 / 8.40 | -0.15 | [-1.4, +1.1] | 84 / 87 | 0.88 |
| M1a_fx2 (f104 only (ensemble a)) vs M1b_fx2 (f104 only (ensemble b)) | fail | 3.00 / 3.15 | -0.15 | [-0.8, +0.5] | 37 / 40 | 0.82 |
| M1a_fx2 (f104 only (ensemble a)) vs M1b_fx2 (f104 only (ensemble b)) | backward_only | 5.25 / 5.25 | +0.00 | [-0.9, +0.9] | 77 / 77 | 1 |
| M3a_fx2 (three arenas, same total data (ensemble a)) vs M3b_fx2 (three arenas, same total data (ensemble b)) | unsafe | 6.30 / 6.05 | +0.25 | [-0.7, +1.2] | 63 / 58 | 0.72 |
| M3a_fx2 (three arenas, same total data (ensemble a)) vs M3b_fx2 (three arenas, same total data (ensemble b)) | fail | 2.75 / 3.30 | -0.55 | [-1.4, +0.3] | 32 / 43 | 0.25 |
| M3a_fx2 (three arenas, same total data (ensemble a)) vs M3b_fx2 (three arenas, same total data (ensemble b)) | backward_only | 3.55 / 2.75 | +0.80 | [+0.1, +1.6] | 57 / 41 | 0.13 |

The family contrasts split into their two parts (composite arms): M3_fx2 vs M1 goal not reached -0.05 [-0.9, +0.8], backward-only -2.10 [-3.0, -1.2]; A3_fx2 vs M1 goal not reached -0.13 [-1.0, +0.7], backward-only -2.85 [-4.0, -1.8] (unsafe = goal not reached + backward-only, exactly).

## T6. Fixed 2 m/s outcomes by designed-route feasibility of the unseen group (post-hoc; the 3 designed constant 2 m/s routes of each group, HMMWV, driven in the rigid collection)

| groups | n | f104 only (M1): unsafe / not reached | two arenas (M2) | three arenas same total (M3) | three arenas all data (A3) | straight 2 m/s |
|---|---|---|---|---|---|---|
| some designed 2 m/s route safe | 1594 | 5.4 / 1.9 | 3.8 / 1.9 | 3.7 / 1.7 | 2.6 / 1.3 | 42.9 / 19.8 |
| no designed 2 m/s route safe, some reaches the goal | 338 | 20.3 / 7.4 | 15.7 / 8.0 | 15.8 / 7.4 | 15.7 / 9.5 | 100.0 / 40.8 |
| no designed 2 m/s route reaches the goal | 68 | 18.4 / 9.6 | 26.5 / 19.1 | 16.9 / 12.5 | 17.6 / 8.8 | 100.0 / 100.0 |

## T7. Secondary contrasts (frozen spec, then the add-on spec; unadjusted). Tool decision: "improves" if the one-sided p <= 0.05 and the difference is negative, else "no meaningful difference" if the 90 % interval lies inside +-2 points, else "inconclusive". The column "within +-2" shows the interval test on its own: a small effect can be both detectable and inside the 2-point band.

| name | comparison (plain) | set | label | groups | rates, % | difference | 90 % cluster interval | p (one-sided) | within +-2 points (90 %) | tool decision |
|---|---|---|---|---|---|---|---|---|---|---|
| A_fx2_M2_vs_M1 | two arenas, same total data (f104 + g203) vs f104 only (mean of ensembles a and b) | unseen | unsafe | 2000 | 6.60 / 8.33 | -1.73 | [-3.3, -0.3] | 0.025 | no | improves (secondary, unadjusted) |
| A_fx2_A3_vs_M3 | three arenas, all data vs three arenas, same total data (mean of a and b) | unseen | unsafe | 2000 | 5.35 / 6.17 | -0.83 | [-1.5, -0.1] | 0.024 | yes | improves (secondary, unadjusted) |
| A_fx2_M3_vs_M1_fail | three arenas, same total data (mean of a and b) vs f104 only (mean of ensembles a and b) | unseen | fail | 2000 | 3.02 / 3.08 | -0.05 | [-0.9, +0.7] | 0.468 | yes | no meaningful difference (secondary, unadjusted) |
| A_fx2_A3_vs_M1_fail | three arenas, all data vs f104 only (mean of ensembles a and b) | unseen | fail | 2000 | 2.95 / 3.08 | -0.13 | [-1.0, +0.7] | 0.424 | yes | no meaningful difference (secondary, unadjusted) |
| A_fx2_seed_M1a_vs_M1b | f104 only (ensemble a) vs f104 only (ensemble b) | unseen | unsafe | 2000 | 8.25 / 8.40 | -0.15 | [-1.4, +1.1] | 0.407 | yes | no meaningful difference (secondary, unadjusted) |
| A_fx2_seed_M3a_vs_M3b | three arenas, same total data (ensemble a) vs three arenas, same total data (ensemble b) | unseen | unsafe | 2000 | 6.30 / 6.05 | +0.25 | [-0.7, +1.2] | 0.690 | yes | no meaningful difference (secondary, unadjusted) |
| A_fx2_M1_vs_straight2 | f104 only (mean of ensembles a and b) vs straight route at 2 m/s | unseen | unsafe | 2000 | 8.33 / 54.50 | -46.18 | [-50.6, -41.9] | < 0.001 | no | improves (secondary, unadjusted) |
| A_fx2_noharm_f104_M3_vs_M1 | three arenas, same total data (mean of a and b) vs f104 only (mean of ensembles a and b) | indist_f104 | unsafe | 200 | 0.50 / 0.50 | +0.00 | [-0.6, +0.5] | 0.636 | yes | no meaningful difference (secondary, unadjusted) |
| A_fx2_noharm_f104_A3_vs_M1 | three arenas, all data vs f104 only (mean of ensembles a and b) | indist_f104 | unsafe | 200 | 0.50 / 0.50 | +0.00 | [-0.9, +1.1] | 0.546 | yes | no meaningful difference (secondary, unadjusted) |
| A_fx2_inarena_M3_vs_M1 | three arenas, same total data (mean of a and b) vs f104 only (mean of ensembles a and b) | heldout | unsafe | 300 | 3.50 / 11.33 | -7.83 | [-11.6, -4.4] | < 0.001 | no | improves (secondary, unadjusted) |
| A_fx2_inarena_A3_vs_M1 | three arenas, all data vs f104 only (mean of ensembles a and b) | heldout | unsafe | 300 | 1.67 / 11.33 | -9.67 | [-13.9, -5.8] | < 0.001 | no | improves (secondary, unadjusted) |
| A_fx2_dev_M3_vs_M1 | three arenas, same total data (mean of a and b) vs f104 only (mean of ensembles a and b) | dev | unsafe | 150 | 3.33 / 3.00 | +0.33 | [-0.8, +1.9] | 0.748 | yes | no meaningful difference (secondary, unadjusted) |
| A_fx2_dev_A3_vs_M1 | three arenas, all data vs f104 only (mean of ensembles a and b) | dev | unsafe | 150 | 1.33 / 3.00 | -1.67 | [-3.4, +0.6] | 0.120 | no | inconclusive (secondary, unadjusted) |
| A_free_M3_vs_M1_fail | three arenas, same total data (mean of a and b) vs f104 only (mean of ensembles a and b) | unseen | fail | 2000 | 0.33 / 0.62 | -0.30 | [-0.6, -0.0] | 0.039 | yes | improves (secondary, unadjusted) |
| A_free_A3_vs_M1_fail | three arenas, all data vs f104 only (mean of ensembles a and b) | unseen | fail | 2000 | 0.30 / 0.62 | -0.33 | [-0.6, -0.1] | 0.006 | yes | improves (secondary, unadjusted) |
| A_free_M3_vs_M1_unsafe | three arenas, same total data (mean of a and b) vs f104 only (mean of ensembles a and b) | unseen | unsafe | 2000 | 0.65 / 1.25 | -0.60 | [-1.1, -0.2] | 0.006 | yes | improves (secondary, unadjusted) |
| A_free_A3_vs_M1_unsafe | three arenas, all data vs f104 only (mean of ensembles a and b) | unseen | unsafe | 2000 | 0.70 / 1.25 | -0.55 | [-1.0, -0.2] | 0.003 | yes | improves (secondary, unadjusted) |
| A_free_M2_vs_M1_fail | two arenas, same total data (f104 + g203) vs f104 only (mean of ensembles a and b) | unseen | fail | 2000 | 0.25 / 0.62 | -0.38 | [-0.7, -0.1] | 0.010 | yes | improves (secondary, unadjusted) |
| A_free_M1_vs_straight6_fail | f104 only (mean of ensembles a and b) vs straight route at 6 m/s | unseen | fail | 2000 | 0.62 / 1.60 | -0.97 | [-1.8, -0.3] | 0.008 | yes | improves (secondary, unadjusted) |
| A_free_noharm_f104_M3_vs_M1 | three arenas, same total data (mean of a and b) vs f104 only (mean of ensembles a and b) | indist_f104 | fail | 200 | 0.00 / 0.00 | +0.00 | [+0.0, +0.0] | 1.000 | yes | no meaningful difference (secondary, unadjusted) |
| A_free_noharm_f104_A3_vs_M1 | three arenas, all data vs f104 only (mean of ensembles a and b) | indist_f104 | fail | 200 | 0.00 / 0.00 | +0.00 | [+0.0, +0.0] | 1.000 | yes | no meaningful difference (secondary, unadjusted) |
| A_free_inarena_A3_vs_M1 | three arenas, all data vs f104 only (mean of ensembles a and b) | heldout | fail | 300 | 0.33 / 0.83 | -0.50 | [-1.6, +0.4] | 0.238 | yes | no meaningful difference (secondary, unadjusted) |
| B_rigid_G_vs_H_on_gator_fail | Gator-trained vs HMMWV-trained (= M1a) | f104_800 | fail | 800 | 0.00 / 0.12 | -0.12 | [-0.4, +0.0] | 0.349 | yes | no meaningful difference (secondary, unadjusted) |
| B_rigid_G_vs_H_on_gator_unsafe | Gator-trained vs HMMWV-trained (= M1a) | f104_800 | unsafe | 800 | 0.00 / 0.38 | -0.38 | [-0.9, +0.0] | 0.108 | yes | no meaningful difference (secondary, unadjusted) |
| B_rigid_G_vs_straight6_on_gator_fail | Gator-trained vs straight route at 6 m/s | f104_800 | fail | 800 | 0.00 / 0.38 | -0.38 | [-0.7, -0.1] | 0.027 | yes | improves (secondary, unadjusted) |
| B_rigid_G_vs_straight6_on_gator_unsafe | Gator-trained vs straight route at 6 m/s | f104_800 | unsafe | 800 | 0.00 / 0.62 | -0.62 | [-1.3, -0.1] | 0.028 | yes | improves (secondary, unadjusted) |
| B_rigid_H_vs_straight6_on_gator_fail | HMMWV-trained (= M1a) vs straight route at 6 m/s | f104_800 | fail | 800 | 0.12 / 0.38 | -0.25 | [-0.5, +0.0] | 0.100 | yes | no meaningful difference (secondary, unadjusted) |
| B_rigid_H_gator_vs_H_hmmwv_fail | HMMWV-trained (= M1a) vs f104 only (ensemble a) | f104_800 | fail | 800 | 0.12 / 0.00 | +0.12 | [+0.0, +0.4] | 1.000 | yes | no meaningful difference (secondary, unadjusted) |
| B_rigid_H_gator_vs_H_hmmwv_unsafe | HMMWV-trained (= M1a) vs f104 only (ensemble a) | f104_800 | unsafe | 800 | 0.38 / 0.12 | +0.25 | [+0.0, +0.5] | 1.000 | yes | no meaningful difference (secondary, unadjusted) |
| B_rigid_G_gator_vs_H_hmmwv_fail | Gator-trained vs f104 only (ensemble a) | f104_800 | fail | 800 | 0.00 / 0.00 | +0.00 | [+0.0, +0.0] | 1.000 | yes | no meaningful difference (secondary, unadjusted) |
| B_rigid_straight6_gator_vs_hmmwv_fail | straight route at 6 m/s vs straight route at 6 m/s | f104_800 | fail | 800 | 0.38 / 0.75 | -0.38 | [-1.3, +0.5] | 0.307 | yes | no meaningful difference (secondary, unadjusted) |
| B_rigid_H_vs_straight6_on_hmmwv_fail | f104 only (ensemble a) vs straight route at 6 m/s | f104_800 | fail | 800 | 0.00 / 0.75 | -0.75 | [-1.5, +0.0] | 0.104 | yes | no meaningful difference (secondary, unadjusted) |
| X_free_seed_M1a_vs_M1b_fail (add-on) | f104 only (ensemble a) vs f104 only (ensemble b) | unseen | fail | 2000 | 0.60 / 0.65 | -0.05 | [-0.4, +0.3] | 0.441 | yes | no meaningful difference (secondary, unadjusted) |
| X_free_seed_M3a_vs_M3b_fail (add-on) | three arenas, same total data (ensemble a) vs three arenas, same total data (ensemble b) | unseen | fail | 2000 | 0.40 / 0.25 | +0.15 | [-0.1, +0.4] | 0.879 | yes | no meaningful difference (secondary, unadjusted) |
| X_free_seed_M1a_vs_M1b_unsafe (add-on) | f104 only (ensemble a) vs f104 only (ensemble b) | unseen | unsafe | 2000 | 1.05 / 1.45 | -0.40 | [-0.9, +0.1] | 0.122 | yes | no meaningful difference (secondary, unadjusted) |
| X_free_seed_M3a_vs_M3b_unsafe (add-on) | three arenas, same total data (ensemble a) vs three arenas, same total data (ensemble b) | unseen | unsafe | 2000 | 0.65 / 0.65 | +0.00 | [-0.3, +0.3] | 0.554 | yes | no meaningful difference (secondary, unadjusted) |
| X_fx2_seed_M1a_vs_M1b_fail (add-on) | f104 only (ensemble a) vs f104 only (ensemble b) | unseen | fail | 2000 | 3.00 / 3.15 | -0.15 | [-0.8, +0.5] | 0.384 | yes | no meaningful difference (secondary, unadjusted) |
| X_fx2_seed_M3a_vs_M3b_fail (add-on) | three arenas, same total data (ensemble a) vs three arenas, same total data (ensemble b) | unseen | fail | 2000 | 2.75 / 3.30 | -0.55 | [-1.4, +0.3] | 0.153 | yes | no meaningful difference (secondary, unadjusted) |
| X_fx2_M2_vs_M1_fail (add-on) | two arenas, same total data (f104 + g203) vs f104 only (mean of ensembles a and b) | unseen | fail | 2000 | 3.50 / 3.08 | +0.43 | [-0.7, +1.5] | 0.742 | yes | no meaningful difference (secondary, unadjusted) |
| X_fx2_A3_vs_M3_fail (add-on) | three arenas, all data vs three arenas, same total data (mean of a and b) | unseen | fail | 2000 | 2.95 / 3.02 | -0.08 | [-0.7, +0.6] | 0.427 | yes | no meaningful difference (secondary, unadjusted) |
| X_fx2_M3_fx2_vs_M1_backward_only (add-on) | three arenas, same total data (mean of a and b) vs f104 only (mean of ensembles a and b) | unseen | backward_only | 2000 | 3.15 / 5.25 | -2.10 | [-3.0, -1.2] | < 0.001 | no | improves (secondary, unadjusted) |
| X_fx2_M3_fx2_vs_straight2_unsafe (add-on) | three arenas, same total data (mean of a and b) vs straight route at 2 m/s | unseen | unsafe | 2000 | 6.17 / 54.50 | -48.33 | [-52.6, -44.1] | < 0.001 | no | improves (secondary, unadjusted) |
| X_fx2_inarena_g203_M3_fx2_vs_M1_unsafe (add-on) | three arenas, same total data (mean of a and b) vs f104 only (mean of ensembles a and b) | heldout_g203 | unsafe | 150 | 2.00 / 6.67 | -4.67 | [-8.4, -1.5] | 0.002 | no | improves (secondary, unadjusted) |
| X_fx2_inarena_g228_M3_fx2_vs_M1_unsafe (add-on) | three arenas, same total data (mean of a and b) vs f104 only (mean of ensembles a and b) | heldout_g228 | unsafe | 150 | 5.00 / 16.00 | -11.00 | [-17.1, -5.4] | < 0.001 | no | improves (secondary, unadjusted) |
| X_fx2_dev_M3_fx2_vs_M1_fail (add-on) | three arenas, same total data (mean of a and b) vs f104 only (mean of ensembles a and b) | dev | fail | 150 | 0.67 / 1.33 | -0.67 | [-2.2, +0.9] | 0.312 | no | inconclusive (secondary, unadjusted) |
| X_fx2_A3_fx2_vs_M1_backward_only (add-on) | three arenas, all data vs f104 only (mean of ensembles a and b) | unseen | backward_only | 2000 | 2.40 / 5.25 | -2.85 | [-4.0, -1.9] | < 0.001 | no | improves (secondary, unadjusted) |
| X_fx2_A3_fx2_vs_straight2_unsafe (add-on) | three arenas, all data vs straight route at 2 m/s | unseen | unsafe | 2000 | 5.35 / 54.50 | -49.15 | [-53.3, -44.8] | < 0.001 | no | improves (secondary, unadjusted) |
| X_fx2_inarena_g203_A3_fx2_vs_M1_unsafe (add-on) | three arenas, all data vs f104 only (mean of ensembles a and b) | heldout_g203 | unsafe | 150 | 1.33 / 6.67 | -5.33 | [-8.5, -2.7] | < 0.001 | no | improves (secondary, unadjusted) |
| X_fx2_inarena_g228_A3_fx2_vs_M1_unsafe (add-on) | three arenas, all data vs f104 only (mean of ensembles a and b) | heldout_g228 | unsafe | 150 | 2.00 / 16.00 | -14.00 | [-21.0, -7.4] | < 0.001 | no | improves (secondary, unadjusted) |
| X_fx2_dev_A3_fx2_vs_M1_fail (add-on) | three arenas, all data vs f104 only (mean of ensembles a and b) | dev | fail | 150 | 1.33 / 1.33 | +0.00 | [-1.9, +1.7] | 0.564 | yes | no meaningful difference (secondary, unadjusted) |
| X_fx2_inarena_M2_vs_M1_unsafe (add-on) | two arenas, same total data (f104 + g203) vs f104 only (mean of ensembles a and b) | heldout | unsafe | 300 | 8.00 / 11.33 | -3.33 | [-5.9, -1.1] | 0.005 | no | improves (secondary, unadjusted) |
| X_fx2_inarena_M3_vs_M1_fail (add-on) | three arenas, same total data (mean of a and b) vs f104 only (mean of ensembles a and b) | heldout | fail | 300 | 1.33 / 3.00 | -1.67 | [-3.3, +0.0] | 0.051 | no | inconclusive (secondary, unadjusted) |
| X_fx2_inarena_A3_vs_M1_fail (add-on) | three arenas, all data vs f104 only (mean of ensembles a and b) | heldout | fail | 300 | 1.00 / 3.00 | -2.00 | [-4.0, -0.2] | 0.045 | no | improves (secondary, unadjusted) |
| X_free_inarena_M3_vs_M1_fail (add-on) | three arenas, same total data (mean of a and b) vs f104 only (mean of ensembles a and b) | heldout | fail | 300 | 0.50 / 0.83 | -0.33 | [-1.2, +0.4] | 0.303 | yes | no meaningful difference (secondary, unadjusted) |
| X_free_inarena_A3_vs_M1_unsafe (add-on) | three arenas, all data vs f104 only (mean of ensembles a and b) | heldout | unsafe | 300 | 0.33 / 1.50 | -1.17 | [-2.3, +0.0] | 0.054 | no | inconclusive (secondary, unadjusted) |
| X_free_M1_vs_straight6_unsafe (add-on) | f104 only (mean of ensembles a and b) vs straight route at 6 m/s | unseen | unsafe | 2000 | 1.25 / 2.10 | -0.85 | [-2.0, +0.2] | 0.095 | yes | no meaningful difference (secondary, unadjusted) |
| X_free_M3_vs_straight6_fail (add-on) | three arenas, same total data (mean of a and b) vs straight route at 6 m/s | unseen | fail | 2000 | 0.33 / 1.60 | -1.28 | [-2.1, -0.6] | < 0.001 | no | improves (secondary, unadjusted) |
| X_free_A3_vs_straight6_fail (add-on) | three arenas, all data vs straight route at 6 m/s | unseen | fail | 2000 | 0.30 / 1.60 | -1.30 | [-2.1, -0.6] | < 0.001 | no | improves (secondary, unadjusted) |
| X_free_M2_vs_M1_unsafe (add-on) | two arenas, same total data (f104 + g203) vs f104 only (mean of ensembles a and b) | unseen | unsafe | 2000 | 0.60 / 1.25 | -0.65 | [-1.1, -0.2] | 0.007 | yes | improves (secondary, unadjusted) |
| X_free_A3_vs_M3_fail (add-on) | three arenas, all data vs three arenas, same total data (mean of a and b) | unseen | fail | 2000 | 0.30 / 0.33 | -0.02 | [-0.2, +0.2] | 0.445 | yes | no meaningful difference (secondary, unadjusted) |
| X_free_dev_M3_vs_M1_fail (add-on) | three arenas, same total data (mean of a and b) vs f104 only (mean of ensembles a and b) | dev | fail | 150 | 0.67 / 0.00 | +0.67 | [+0.0, +1.3] | 1.000 | yes | no meaningful difference (secondary, unadjusted) |
| X_free_dev_A3_vs_M1_fail (add-on) | three arenas, all data vs f104 only (mean of ensembles a and b) | dev | fail | 150 | 0.00 / 0.00 | +0.00 | [+0.0, +0.0] | 1.000 | yes | no meaningful difference (secondary, unadjusted) |

## T8. Generalisation gap: rate on the 8 unseen arenas minus rate in distribution (95 % intervals; independent bootstraps)

In distribution = the 200 declared f104 hill/crater groups (frozen spec) or the g203 / g228 held-out groups (add-on spec; in distribution for M3 and A3, an arena never seen by M1; g203 was seen by M2). "vs straight" = [model - straight](unseen) - [model - straight](in distribution).

| model | label | f104 in distribution: unseen / in-dist, % | gap | vs straight | held-out g203+g228: in-dist, % | gap | vs straight |
|---|---|---|---|---|---|---|---|
| M1_fx2 (f104 only (mean of ensembles a and b)) | unsafe | 8.33 / 0.50 | +7.83 [+6.5, +9.0] | +11.3 [+4.2, +18.7] | 11.33 | -3.01 [-6.2, +0.1] | +0.5 [-6.2, +6.9] |
| M1_fx2 (f104 only (mean of ensembles a and b)) | fail | 3.08 / 0.00 | +3.08 [+2.5, +3.7] | +5.0 [-1.5, +11.7] | 3.00 | +0.08 [-1.7, +1.7] | +8.4 [+2.6, +14.1] |
| M2_fx2 (two arenas, same total data (f104 + g203)) | unsafe | 6.60 / 2.00 | +4.60 [+2.3, +6.7] | +8.1 [+0.9, +15.5] | 8.00 | -1.40 [-4.8, +1.8] | +2.1 [-4.7, +8.4] |
| M2_fx2 (two arenas, same total data (f104 + g203)) | fail | 3.50 / 0.00 | +3.50 [+2.8, +4.3] | +5.5 [-1.1, +11.9] | 2.67 | +0.83 [-1.3, +2.7] | +9.1 [+3.3, +14.9] |
| M3_fx2 (three arenas, same total data (mean of a and b)) | unsafe | 6.17 / 0.50 | +5.67 [+4.5, +6.8] | +9.2 [+1.9, +16.2] | 3.50 | +2.67 [+0.7, +4.6] | +6.2 [+0.1, +12.2] |
| M3_fx2 (three arenas, same total data (mean of a and b)) | fail | 3.02 / 0.00 | +3.02 [+2.4, +3.6] | +5.0 [-1.5, +11.7] | 1.33 | +1.69 [+0.6, +2.7] | +10.0 [+4.3, +15.9] |
| A3_fx2 (three arenas, all data) | unsafe | 5.35 / 0.50 | +4.85 [+3.3, +6.2] | +8.3 [+1.1, +15.8] | 1.67 | +3.68 [+1.9, +5.3] | +7.2 [+1.2, +13.2] |
| A3_fx2 (three arenas, all data) | fail | 2.95 / 0.50 | +2.45 [+1.1, +3.5] | +4.4 [-2.1, +11.2] | 1.00 | +1.95 [+0.5, +3.2] | +10.2 [+4.6, +16.1] |
| M1_free (f104 only (mean of ensembles a and b)) | fail | 0.62 / 0.00 | +0.62 [+0.4, +0.9] | +0.0 [-1.2, +1.7] | 0.83 | -0.21 [-1.3, +0.7] | +2.9 [+0.3, +5.5] |
| M1_free (f104 only (mean of ensembles a and b)) | unsafe | 1.25 / 0.00 | +1.25 [+0.9, +1.6] | +0.6 [-1.0, +2.7] | 1.50 | -0.25 [-1.6, +0.9] | +4.3 [+1.5, +7.4] |
| M2_free (two arenas, same total data (f104 + g203)) | fail | 0.25 / 0.00 | +0.25 [+0.1, +0.5] | -0.4 [-1.7, +1.4] | 1.00 | -0.75 [-2.1, +0.2] | +2.3 [-0.1, +5.0] |
| M2_free (two arenas, same total data (f104 + g203)) | unsafe | 0.60 / 0.00 | +0.60 [+0.3, +0.9] | -0.0 [-1.6, +2.0] | 2.00 | -1.40 [-3.1, +0.1] | +3.2 [+0.1, +6.3] |
| M3_free (three arenas, same total data (mean of a and b)) | fail | 0.33 / 0.00 | +0.33 [+0.1, +0.5] | -0.3 [-1.6, +1.5] | 0.50 | -0.17 [-0.8, +0.4] | +2.9 [+0.5, +5.6] |
| M3_free (three arenas, same total data (mean of a and b)) | unsafe | 0.65 / 0.50 | +0.15 [-0.7, +0.8] | -0.5 [-2.3, +1.7] | 0.83 | -0.18 [-1.0, +0.5] | +4.4 [+1.6, +7.4] |
| A3_free (three arenas, all data) | fail | 0.30 / 0.00 | +0.30 [+0.1, +0.5] | -0.3 [-1.6, +1.4] | 0.33 | -0.03 [-0.8, +0.5] | +3.0 [+0.9, +5.6] |
| A3_free (three arenas, all data) | unsafe | 0.70 / 0.00 | +0.70 [+0.4, +1.1] | +0.1 [-1.6, +2.1] | 0.33 | +0.37 [-0.5, +1.0] | +4.9 [+2.3, +7.9] |

Held-out groups per arena, fixed 2 m/s unsafe (%): g203: M1a_fx2 7.3, M1b_fx2 6.0, M2_fx2 2.0, M3a_fx2 2.0, M3b_fx2 2.0, A3_fx2 1.3, straight2 59.3; g228: M1a_fx2 14.7, M1b_fx2 17.3, M2_fx2 14.0, M3a_fx2 7.3, M3b_fx2 2.7, A3_fx2 2.0, straight2 56.7.

In-distribution rates (f104 200 groups; held-out g203 / g228 150 each; dev g217 150), goal not reached / unsafe (%):

| arm | f104 in distribution | g203 held-out | g228 held-out | g217 dev | 8 unseen arenas |
|---|---|---|---|---|---|
| M1a_free (f104 only (ensemble a)) | 0.0 / 0.0 | 0.7 / 0.7 | 0.7 / 0.7 | 0.0 / 0.0 | 0.6 / 1.1 |
| M1b_free (f104 only (ensemble b)) | 0.0 / 0.0 | 0.7 / 2.0 | 1.3 / 2.7 | 0.0 / 2.0 | 0.7 / 1.5 |
| M1_free (f104 only (mean of ensembles a and b)) | 0.0 / 0.0 | 0.7 / 1.3 | 1.0 / 1.7 | 0.0 / 1.0 | 0.6 / 1.2 |
| M2_free (two arenas, same total data (f104 + g203)) | 0.0 / 0.0 | 0.7 / 2.0 | 1.3 / 2.0 | 0.0 / 0.0 | 0.2 / 0.6 |
| M3a_free (three arenas, same total data (ensemble a)) | 0.0 / 1.0 | 0.7 / 0.7 | 0.7 / 0.7 | 0.0 / 0.0 | 0.4 / 0.7 |
| M3b_free (three arenas, same total data (ensemble b)) | 0.0 / 0.0 | 0.0 / 0.7 | 0.7 / 1.3 | 1.3 / 1.3 | 0.2 / 0.7 |
| M3_free (three arenas, same total data (mean of a and b)) | 0.0 / 0.5 | 0.3 / 0.7 | 0.7 / 1.0 | 0.7 / 0.7 | 0.3 / 0.7 |
| A3_free (three arenas, all data) | 0.0 / 0.0 | 0.7 / 0.7 | 0.0 / 0.0 | 0.0 / 0.0 | 0.3 / 0.7 |
| straight6 (straight route at 6 m/s) | 1.0 / 1.5 | 5.3 / 7.3 | 4.0 / 6.0 | 0.0 / 2.0 | 1.6 / 2.1 |
| M1a_fx2 (f104 only (ensemble a)) | 0.0 / 0.5 | 1.3 / 7.3 | 4.0 / 14.7 | 0.7 / 2.0 | 3.0 / 8.2 |
| M1b_fx2 (f104 only (ensemble b)) | 0.0 / 0.5 | 2.7 / 6.0 | 4.0 / 17.3 | 2.0 / 4.0 | 3.1 / 8.4 |
| M1_fx2 (f104 only (mean of ensembles a and b)) | 0.0 / 0.5 | 2.0 / 6.7 | 4.0 / 16.0 | 1.3 / 3.0 | 3.1 / 8.3 |
| M2_fx2 (two arenas, same total data (f104 + g203)) | 0.0 / 2.0 | 1.3 / 2.0 | 4.0 / 14.0 | 0.0 / 0.7 | 3.5 / 6.6 |
| M3a_fx2 (three arenas, same total data (ensemble a)) | 0.0 / 1.0 | 0.7 / 2.0 | 2.7 / 7.3 | 0.7 / 4.0 | 2.8 / 6.3 |
| M3b_fx2 (three arenas, same total data (ensemble b)) | 0.0 / 0.0 | 0.7 / 2.0 | 1.3 / 2.7 | 0.7 / 2.7 | 3.3 / 6.0 |
| M3_fx2 (three arenas, same total data (mean of a and b)) | 0.0 / 0.5 | 0.7 / 2.0 | 2.0 / 5.0 | 0.7 / 3.3 | 3.0 / 6.2 |
| A3_fx2 (three arenas, all data) | 0.5 / 0.5 | 0.0 / 1.3 | 2.0 / 2.0 | 1.3 / 1.3 | 2.9 / 5.3 |
| straight2 (straight route at 2 m/s) | 28.0 / 58.0 | 38.0 / 59.3 | 30.7 / 56.7 | 18.7 / 41.3 | 26.1 / 54.5 |

## T9. Dose response f104 only -> two arenas -> three arenas (same total data)

| set | mode, label | rates M1 / M2 / M3, % | slope, points per added arena (95 %) | steps M2-M1, M3-M2 |
|---|---|---|---|---|
| unseen | fixed 2 m/s, unsafe | 8.33 / 6.60 / 6.17 | -1.08 [-1.56, -0.59] | -1.73 [-3.3, -0.3], -0.43 [-1.6, +0.8] |
| unseen | fixed 2 m/s, fail | 3.08 / 3.50 / 3.02 | -0.03 [-0.36, +0.32] | +0.43 [-0.7, +1.5], -0.48 [-1.3, +0.4] |
| unseen | speed free, fail | 0.62 / 0.25 / 0.33 | -0.15 [-0.30, -0.01] | -0.38 [-0.7, -0.1], +0.07 [-0.2, +0.3] |
| unseen | speed free, unsafe | 1.25 / 0.60 / 0.65 | -0.30 [-0.51, -0.10] | -0.65 [-1.2, -0.2], +0.05 [-0.3, +0.4] |
| near | fixed 2 m/s, unsafe | 7.05 / 6.90 / 6.15 | -0.45 [-1.05, +0.17] | -0.15 [-1.9, +1.6], -0.75 [-2.3, +0.8] |
| spread | fixed 2 m/s, unsafe | 9.60 / 6.30 / 6.20 | -1.70 [-2.45, -0.95] | -3.30 [-5.8, -1.1], -0.10 [-1.9, +1.8] |

## T10. Median time ratio on joint successes (test / reference)

- M3a_free (three arenas, same total data (ensemble a)) / M1a_free (f104 only (ensemble a)), unseen: {"n": 1980, "median_paired_ratio": 1.196, "median_paired_ratio_p95": 1.212, "ratio_of_medians": 1.214, "ratio_of_medians_p95": 1.234, "mean_ratio": 1.285, "median_time_x": 15.2, "median_time_y": 12.525, "bound": 1.1, "pass_time": false, "note": "t(M3a_free) / t(M1a_free); 95th percentile of the group bootstrap of the median paired ratio must be < 1.1"}
- A3_free (three arenas, all data) / M1a_free (f104 only (ensemble a)), unseen: {"n": 1984, "median_paired_ratio": 1.065, "median_paired_ratio_p95": 1.076, "ratio_of_medians": 1.076, "ratio_of_medians_p95": 1.093, "mean_ratio": 1.108, "median_time_x": 13.5, "median_time_y": 12.55, "bound": 1.1, "pass_time": true, "note": "t(A3_free) / t(M1a_free); 95th percentile of the group bootstrap of the median paired ratio must be < 1.1"}
- M1a_free (f104 only (ensemble a)) / straight6 (straight route at 6 m/s), unseen: {"n": 1957, "median_paired_ratio": 1.604, "median_paired_ratio_p95": 1.619, "ratio_of_medians": 1.634, "ratio_of_medians_p95": 1.651, "mean_ratio": 1.686, "median_time_x": 12.5, "median_time_y": 7.65, "bound": 1.1, "pass_time": false, "note": "t(M1a_free) / t(straight6); 95th percentile of the group bootstrap of the median paired ratio must be < 1.1"}
- G_free_gator (Gator-trained) / H_free_gator (HMMWV-trained (= M1a)), f104_800: {"n": 799, "median_paired_ratio": 0.975, "median_paired_ratio_p95": 0.996, "ratio_of_medians": 0.981, "ratio_of_medians_p95": 1.004, "mean_ratio": 0.987, "median_time_x": 13.1, "median_time_y": 13.35, "bound": 1.1, "pass_time": true, "note": "t(G_free_gator) / t(H_free_gator); 95th percentile of the group bootstrap of the median paired ratio must be < 1.1"}
- G_fx2_gator (Gator-trained) / H_fx2_gator (HMMWV-trained (= M1a)), f104_800: {"n": 800, "median_paired_ratio": 1.0, "median_paired_ratio_p95": 1.002, "ratio_of_medians": 1.008, "ratio_of_medians_p95": 1.016, "mean_ratio": 1.001, "median_time_x": 21.475, "median_time_y": 21.3, "bound": 1.1, "pass_time": true, "note": "t(G_fx2_gator) / t(H_fx2_gator); 95th percentile of the group bootstrap of the median paired ratio must be < 1.1"}
- M3a_fx2 (three arenas, same total data (ensemble a)) / M1a_fx2 (f104 only (ensemble a)), unseen: {"n": 1903, "median_paired_ratio": 0.998, "median_paired_ratio_p95": 1.0, "ratio_of_medians": 0.993, "ratio_of_medians_p95": 0.998, "mean_ratio": 0.998, "median_time_x": 21.0, "median_time_y": 21.15, "bound": 1.1, "pass_time": true, "note": "t(M3a_fx2) / t(M1a_fx2); 95th percentile of the group bootstrap of the median paired ratio must be < 1.1"}
- A3_fx2 (three arenas, all data) / M1a_fx2 (f104 only (ensemble a)), unseen: {"n": 1898, "median_paired_ratio": 0.997, "median_paired_ratio_p95": 0.998, "ratio_of_medians": 0.986, "ratio_of_medians_p95": 0.993, "mean_ratio": 0.992, "median_time_x": 20.8, "median_time_y": 21.1, "bound": 1.1, "pass_time": true, "note": "t(A3_fx2) / t(M1a_fx2); 95th percentile of the group bootstrap of the median paired ratio must be < 1.1"}
- M1a_fx2 (f104 only (ensemble a)) / straight2 (straight route at 2 m/s), unseen: {"n": 1452, "median_paired_ratio": 1.01, "median_paired_ratio_p95": 1.013, "ratio_of_medians": 0.93, "ratio_of_medians_p95": 0.943, "mean_ratio": 0.909, "median_time_x": 21.0, "median_time_y": 22.575, "bound": 1.1, "pass_time": true, "note": "t(M1a_fx2) / t(straight2); 95th percentile of the group bootstrap of the median paired ratio must be < 1.1"}

## T11. Task B rigid (800 f104 groups: 600 f104_pair + 200 f104_crm_eval), rates

| arm (code) | plain label | vehicle driven | mode | goal reached, % | unsafe, % | unsafe without the backward clause (= not reached), % | backward-only, % | tilt > 30 deg, % | median time, s | chassis ground contact, % of drives |
|---|---|---|---|---|---|---|---|---|---|---|
| G_free_gator | Gator-trained on the Gator | gator | speed free | 100.00 | 0.00 | 0.00 | 0.00 | 12.1 | 13.1 | 3.5 |
| H_free_gator | HMMWV-trained on the Gator | gator | speed free | 99.88 | 0.38 | 0.12 | 0.25 | 8.8 | 13.4 | 4.8 |
| straight6_gator | straight 6 m/s on the Gator | gator | speed free | 99.62 | 0.62 | 0.38 | 0.25 | 12.8 | 7.7 | 13.6 |
| G_fx2_gator | Gator-trained on the Gator | gator | fixed 2 m/s | 100.00 | 0.12 | 0.00 | 0.12 | 8.0 | 21.5 | 1.5 |
| H_fx2_gator | HMMWV-trained on the Gator | gator | fixed 2 m/s | 100.00 | 0.12 | 0.00 | 0.12 | 5.1 | 21.3 | 3.1 |
| straight2_gator | straight 2 m/s on the Gator | gator | fixed 2 m/s | 89.00 | 15.50 | 11.00 | 4.50 | 6.0 | 20.8 | 10.5 |
| M1a_free | HMMWV-trained (H = M1a) on the HMMWV | hmmwv | speed free | 100.00 | 0.12 | 0.00 | 0.12 | 5.6 | 13.8 | 1.2 |
| straight6 | straight 6 m/s on the HMMWV | hmmwv | speed free | 99.25 | 0.88 | 0.75 | 0.12 | 17.6 | 8.1 | 2.5 |

Task B contrasts (difference test - reference in points; 90 % cluster interval; one-sided p; non-inferiority = one-sided upper 95 % bound < +2 points):

| name | comparison | label | rates, % | difference | 90 % cluster interval | p | upper 95 % bound | within 2 points (non-inferior) | groups test worse / better | identical picks |
|---|---|---|---|---|---|---|---|---|---|---|
| B_rigid_G_vs_H_on_gator_fail | G_free_gator vs H_free_gator | fail | 0.00 / 0.12 | -0.12 | [-0.4, +0.0] | 0.349 | +0.00 | yes | 0 / 1 | 0 |
| B_rigid_G_vs_H_on_gator_unsafe | G_free_gator vs H_free_gator | unsafe | 0.00 / 0.38 | -0.38 | [-0.9, +0.0] | 0.108 | +0.00 | yes | 0 / 3 | 0 |
| B_rigid_G_vs_straight6_on_gator_fail | G_free_gator vs straight6_gator | fail | 0.00 / 0.38 | -0.38 | [-0.7, -0.1] | 0.027 | -0.12 | yes | 0 / 3 | 1 |
| B_rigid_G_vs_straight6_on_gator_unsafe | G_free_gator vs straight6_gator | unsafe | 0.00 / 0.62 | -0.62 | [-1.3, -0.1] | 0.028 | -0.12 | yes | 0 / 5 | 1 |
| B_rigid_H_vs_straight6_on_gator_fail | H_free_gator vs straight6_gator | fail | 0.12 / 0.38 | -0.25 | [-0.5, +0.0] | 0.100 | +0.00 | yes | 1 / 3 | 3 |
| B_rigid_H_gator_vs_H_hmmwv_fail | H_free_gator vs M1a_free | fail | 0.12 / 0.00 | +0.12 | [+0.0, +0.4] | 1.000 | +0.38 | yes | 1 / 0 | 800 |
| B_rigid_H_gator_vs_H_hmmwv_unsafe | H_free_gator vs M1a_free | unsafe | 0.38 / 0.12 | +0.25 | [+0.0, +0.5] | 1.000 | +0.52 | yes | 3 / 1 | 800 |
| B_rigid_G_gator_vs_H_hmmwv_fail | G_free_gator vs M1a_free | fail | 0.00 / 0.00 | +0.00 | [+0.0, +0.0] | 1.000 | +0.00 | yes | 0 / 0 | 0 |
| B_rigid_straight6_gator_vs_hmmwv_fail | straight6_gator vs straight6 | fail | 0.38 / 0.75 | -0.38 | [-1.3, +0.5] | 0.307 | +0.51 | yes | 3 / 6 | 800 |
| B_rigid_H_vs_straight6_on_hmmwv_fail | M1a_free vs straight6 | fail | 0.00 / 0.75 | -0.75 | [-1.5, +0.0] | 0.104 | +0.00 | yes | 0 / 6 | 3 |
| B2_rigid_fx2_G_vs_H_on_gator_unsafe | G_fx2_gator vs H_fx2_gator | unsafe | 0.12 / 0.12 | +0.00 | [+0.0, +0.0] | 1.000 | +0.00 | yes | 1 / 1 | 0 |
| B2_rigid_fx2_G_vs_H_on_gator_fail | G_fx2_gator vs H_fx2_gator | fail | 0.00 / 0.00 | +0.00 | [+0.0, +0.0] | 1.000 | +0.00 | yes | 0 / 0 | 0 |
| B2_rigid_fx2_G_vs_straight2_on_gator_unsafe | G_fx2_gator vs straight2_gator | unsafe | 0.12 / 15.50 | -15.38 | [-24.5, -8.1] | < 0.001 | -8.08 | yes | 0 / 123 | 0 |
| B2_rigid_fx2_G_vs_straight2_on_gator_fail | G_fx2_gator vs straight2_gator | fail | 0.00 / 11.00 | -11.00 | [-17.1, -5.9] | < 0.001 | -5.85 | yes | 0 / 88 | 0 |
| B2_rigid_fx2_H_vs_straight2_on_gator_unsafe | H_fx2_gator vs straight2_gator | unsafe | 0.12 / 15.50 | -15.38 | [-24.3, -7.9] | < 0.001 | -7.88 | yes | 1 / 124 | 1 |
| B2_rigid_fx2_H_vs_straight2_on_gator_fail | H_fx2_gator vs straight2_gator | fail | 0.00 / 11.00 | -11.00 | [-17.0, -5.9] | < 0.001 | -5.85 | yes | 0 / 88 | 1 |

Headroom closed = (straight - model) / straight on the failure (or unsafe) rate:

- G_free_gator vs straight6_gator (fail): 0.00 % vs 0.38 %, closed 100 % [100, 100], n 800
- H_free_gator vs straight6_gator (fail): 0.12 % vs 0.38 %, closed 67 % [-100, 100], n 800
- M1a_free vs straight6 (fail): 0.00 % vs 0.75 %, closed 100 % [100, 100], n 800
- G_free_gator vs straight6_gator (unsafe): 0.00 % vs 0.62 %, closed 100 % [100, 100], n 800
- M1a_free vs straight6 (unsafe): 0.12 % vs 0.88 %, closed 86 % [25, 100], n 800
- G_fx2_gator vs straight2_gator (unsafe): 0.12 % vs 15.50 %, closed 99 % [97, 100], n 800
- H_fx2_gator vs straight2_gator (unsafe): 0.12 % vs 15.50 %, closed 99 % [97, 100], n 800
- G_fx2_gator vs straight2_gator (fail): 0.00 % vs 11.00 %, closed 100 % [100, 100], n 800
- H_fx2_gator vs straight2_gator (fail): 0.00 % vs 11.00 %, closed 100 % [100, 100], n 800

## T12. Task B collection: the same 24,000 rigid f104 routes driven by both vehicles (1,200 groups x 20 routes)

| speed profile | routes | goal not reached: Gator / HMMWV, % | only Gator / only HMMWV fails | unsafe: Gator / HMMWV, % | backward-only (rolled back but reached the goal): Gator / HMMWV, % | tilt > 30 deg: Gator / HMMWV, % | simulated h: Gator / HMMWV | median time ratio Gator / HMMWV (joint goals) |
|---|---|---|---|---|---|---|---|---|
| all | 24000 | 6.6 / 19.9 | 667 / 3857 | 11.3 / 36.8 | 4.7 / 17.0 | 13.1 / 10.9 | 165.7 / 202.9 | 0.962 |
| constant_2 | 3600 | 10.9 / 26.6 | 197 / 759 | 15.7 / 55.5 | 4.8 / 28.9 | 10.3 / 7.4 | 28.1 / 36.9 | 0.968 |
| constant_4 | 3600 | 1.2 / 8.2 | 35 / 286 | 4.2 / 19.8 | 3.0 / 11.6 | 15.4 / 12.4 | 12.3 / 17.9 | 0.971 |
| constant_6 | 3600 | 0.6 / 0.9 | 15 / 26 | 0.9 / 1.5 | 0.4 / 0.6 | 16.3 / 18.4 | 8.3 / 9.0 | 0.969 |
| smooth_2_6_2 | 3600 | 1.5 / 7.1 | 31 / 234 | 2.5 / 16.9 | 1.0 / 9.8 | 14.3 / 14.1 | 14.1 / 19.2 | 0.958 |
| designed | 14400 | 3.5 / 10.7 | 278 / 1305 | 5.8 / 23.4 | 2.3 / 12.7 | 14.1 / 13.1 | 62.8 / 83.0 | 0.966 |
| on_policy | 9600 | 11.1 / 33.7 | 389 / 2552 | 19.4 / 56.9 | 8.3 / 23.3 | 11.6 / 7.7 | 102.9 / 120.0 | 0.949 |

Statuses Gator {'goal_reached': 22420, 'prolonged_blockage_terminated': 1144, 'timeout': 387, 'rollover': 46, 'terrain_bounds_exit': 3}; HMMWV {'prolonged_blockage_terminated': 4120, 'goal_reached': 19230, 'timeout': 529, 'terrain_bounds_exit': 85, 'rollover': 36}. Gator checks: launch failed 0, native height failed 0, non-finite 0, rollovers 46, chassis ground contact on 2151 of 24000 routes (9.0 %; max 394 kN), lowest hull point below the surface on 1877 routes (min -0.38 m). HMMWV: chassis contact not recorded in that collection (the rich telemetry was not kept), rollovers 36.

## T13. Offline ranking on the 8 unseen arenas (standing start, the 12 designed routes of each of the 2,000 groups; within-group AUC)

| model (code) | plain label | AUC unsafe: all / near / spread | AUC not reached | lowest-risk route unsafe, % (random route) |
|---|---|---|---|---|
| M1a_deploy | f104 only (ensemble a) | 0.947 / 0.952 / 0.942 | 0.891 | 0.9 (22.6) |
| M1b_deploy | f104 only (ensemble b) | 0.944 / 0.946 / 0.941 | 0.887 | 1.2 (22.6) |
| M2_deploy | two arenas, same total data (f104 + g203) | 0.952 / 0.954 / 0.951 | 0.891 | 0.9 (22.6) |
| M3a_deploy | three arenas, same total data (ensemble a) | 0.952 / 0.956 / 0.948 | 0.893 | 0.9 (22.6) |
| M3b_deploy | three arenas, same total data (ensemble b) | 0.952 / 0.954 / 0.950 | 0.892 | 1.1 (22.6) |
| A3_deploy | three arenas, all data | 0.956 / 0.959 / 0.952 | 0.895 | 1.1 (22.6) |
| G_deploy | Gator-trained (scored on HMMWV drives) | 0.704 / 0.707 / 0.702 | 0.668 | 9.6 (22.6) |
| M1_holdout | f104 only, holdout mode (865 fitted groups) | 0.944 / 0.946 / 0.942 | 0.886 | 0.9 (22.6) |
| M2_holdout | two arenas, holdout mode | 0.949 / 0.949 / 0.949 | 0.892 | 1.1 (22.6) |
| M3_holdout | three arenas same total, holdout mode | 0.945 / 0.946 / 0.943 | 0.889 | 1.8 (22.6) |
| A3_holdout | three arenas all data, holdout mode | 0.955 / 0.961 / 0.949 | 0.893 | 0.9 (22.6) |
| LC545_holdout | f104 only, 431 fitted groups | 0.941 / 0.943 / 0.939 | 0.886 | 1.4 (22.6) |
| LC272_holdout | f104 only, 213 fitted groups | 0.933 / 0.932 / 0.933 | 0.875 | 1.5 (22.6) |
| LOAO1_g203_holdout | g203 only (~433 groups) | 0.939 / 0.939 / 0.938 | 0.878 | 1.6 (22.6) |
| LOAO1_g228_holdout | g228 only (~436 groups) | 0.944 / 0.948 / 0.941 | 0.883 | 0.7 (22.6) |
| LOAO2_f104_g203_holdout | f104 + g203 (~435 groups) | 0.949 / 0.951 / 0.948 | 0.888 | 0.7 (22.6) |
| LOAO2_f104_g228_holdout | f104 + g228 (~441 groups) | 0.945 / 0.946 / 0.944 | 0.886 | 0.9 (22.6) |
| LOAO2_g203_g228_holdout | g203 + g228 (~449 groups) | 0.948 / 0.951 / 0.946 | 0.889 | 0.5 (22.6) |

Offline paired contrasts (group bootstrap, 2,000 resamples; AUC difference): M3_vs_M1: unsafe +0.0064 [+0.0035, +0.0092]; A3_vs_M1: unsafe +0.0102 [+0.0070, +0.0132]; M2_vs_M1: unsafe +0.0070 [+0.0040, +0.0100]; A3_vs_M3: unsafe +0.0037 [+0.0012, +0.0061]; seed_M1a_vs_M1b: unsafe +0.0035 [+0.0009, +0.0061]; seed_M3a_vs_M3b: unsafe +0.0002 [-0.0021, +0.0025].

## T14. Drives and integrity

- Index rows 43100 (arm x group), all driven, 0 missing; distinct drives 42645 (identical routes of two arms of a group driven once); pool shards 507, rows 42645, complete 42645; shards with incomplete rows 0, host mismatch 0, takeovers 0, second attempts 0, non-zero runner exit 0; 23 nodes, 52.1 shard wall hours.
- Hosts: every run has a recorded host (0 without); every group's drives within one task file come from one host; 712 f104 groups have their task B fixed 2 m/s drives (added at 14:10, separate shards) on a different node than their speed-free drives: no contrast pairs arms across those two task files.
