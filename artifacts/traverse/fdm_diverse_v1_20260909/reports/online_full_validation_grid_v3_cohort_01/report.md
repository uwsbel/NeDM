# Declared online cohort report

Every declared trial is counted: **24 trials**, **11 verified schema-safe full goals**.
Process completion and physical success are separate. Failed, missing, timed-out and abstaining trials remain in denominators.

| Energy weight (s/kJ) | Safe full goals / all trials | Valid artifacts | Timeouts | Trials with abstentions |
|---:|---:|---:|---:|---:|
| 0 | 4 / 6 | 6 | 2 | 5 |
| 0.02 | 4 / 6 | 6 | 1 | 5 |
| 0.2 | 2 / 6 | 6 | 3 | 5 |
| 0.5 | 1 / 6 | 6 | 5 | 6 |

## Every individual trial

| Scene | Weight | Process exit | Physical status | Safe goal | Progress (m) | Work consumed (kJ) | Abstentions |
|---|---:|---:|---|---|---:|---:|---:|
| diverse_v1_val_rolling_hills_00 | 0.0 | 0 | goal_reached | True | 225.32 | 809.05 | 6 |
| diverse_v1_val_ridge_passes_00 | 0.0 | 0 | timeout | False | 132.21 | 5881.97 | 95 |
| diverse_v1_val_cross_slopes_00 | 0.0 | 0 | goal_reached | True | 225.28 | 1226.27 | 10 |
| diverse_v1_val_valley_network_00 | 0.0 | 0 | goal_reached | True | 225.35 | 303.88 | 0 |
| diverse_v1_val_rough_mosaic_00 | 0.0 | 0 | timeout | False | 132.18 | 6041.73 | 81 |
| diverse_v1_val_mixed_obstacles_00 | 0.0 | 0 | goal_reached | True | 225.29 | 2184.15 | 24 |
| diverse_v1_val_rolling_hills_00 | 0.02 | 0 | goal_reached | False | 225.34 | 877.64 | 7 |
| diverse_v1_val_ridge_passes_00 | 0.02 | 0 | goal_reached | True | 225.39 | 1898.29 | 21 |
| diverse_v1_val_cross_slopes_00 | 0.02 | 0 | goal_reached | True | 225.42 | 1415.11 | 8 |
| diverse_v1_val_valley_network_00 | 0.02 | 0 | goal_reached | True | 225.32 | 308.23 | 0 |
| diverse_v1_val_rough_mosaic_00 | 0.02 | 0 | timeout | False | 67.62 | 6670.56 | 64 |
| diverse_v1_val_mixed_obstacles_00 | 0.02 | 0 | goal_reached | True | 225.54 | 2254.06 | 22 |
| diverse_v1_val_rolling_hills_00 | 0.2 | 0 | timeout | False | 61.23 | 10311.87 | 168 |
| diverse_v1_val_ridge_passes_00 | 0.2 | 0 | timeout | False | 136.80 | 1533.21 | 133 |
| diverse_v1_val_cross_slopes_00 | 0.2 | 0 | goal_reached | True | 225.26 | 2353.01 | 21 |
| diverse_v1_val_valley_network_00 | 0.2 | 0 | goal_reached | True | 225.26 | 598.61 | 0 |
| diverse_v1_val_rough_mosaic_00 | 0.2 | 0 | timeout | False | 108.45 | 6021.83 | 77 |
| diverse_v1_val_mixed_obstacles_00 | 0.2 | 0 | goal_reached | False | 225.31 | 5113.59 | 57 |
| diverse_v1_val_rolling_hills_00 | 0.5 | 0 | timeout | False | 61.23 | 10693.48 | 167 |
| diverse_v1_val_ridge_passes_00 | 0.5 | 0 | timeout | False | 126.35 | 925.74 | 141 |
| diverse_v1_val_cross_slopes_00 | 0.5 | 0 | goal_reached | True | 225.26 | 666.49 | 5 |
| diverse_v1_val_valley_network_00 | 0.5 | 0 | timeout | False | 189.77 | 4927.75 | 70 |
| diverse_v1_val_rough_mosaic_00 | 0.5 | 0 | timeout | False | 96.84 | 5970.99 | 84 |
| diverse_v1_val_mixed_obstacles_00 | 0.5 | 0 | timeout | False | 220.85 | 3473.75 | 116 |

Individual failed-run work is descriptive consumption, not evidence of an efficiency improvement.

## Paired safe-goal tradeoffs

| Scene | Weight | Work saving | Time increase |
|---|---:|---:|---:|
| diverse_v1_val_cross_slopes_00 | 0.02 | -15.4% | 5.0% |
| diverse_v1_val_valley_network_00 | 0.02 | -1.4% | 0.1% |
| diverse_v1_val_mixed_obstacles_00 | 0.02 | -3.2% | -4.7% |
| diverse_v1_val_cross_slopes_00 | 0.2 | -91.9% | 51.6% |
| diverse_v1_val_valley_network_00 | 0.2 | -97.0% | 13.4% |
| diverse_v1_val_cross_slopes_00 | 0.5 | 45.6% | -11.4% |

## Selection status

`complete_valid_grid_no_qualifier_default_retained`. Selected weight: `0.02`; gate passed: `False`.
The full preregistered validation gate requires at least three safe pairs out of six scenes, mean work saving at least 5%, mean time increase at most 25%, and no unconditional safe-success count drop. Incomplete or invalid grids block selection and freeze.

This report does not open protected test scenes or establish generalization from a pilot.

## Rejected fresh reference counters

Counts concern invalid fresh geometry proposals, not predicted or measured terrain risk. Missing counter evidence stays unavailable.

| Energy weight | Rejected fresh proposals | Decisions with rejections | Trials with verified counters |
|---:|---:|---:|---:|
| 0 | 156 | 16 | 6 / 6 |
| 0.02 | 192 | 25 | 6 / 6 |
| 0.2 | 168 | 17 | 6 / 6 |
| 0.5 | 72 | 8 | 6 / 6 |
