# Declared online cohort report

Every declared trial is counted: **24 trials**, **12 verified schema-safe full goals**.
Process completion and physical success are separate. Failed, missing, timed-out and abstaining trials remain in denominators.

| Energy weight (s/kJ) | Safe full goals / all trials | Valid artifacts | Timeouts | Trials with abstentions |
|---:|---:|---:|---:|---:|
| 0 | 5 / 6 | 6 | 1 | 0 |
| 0.02 | 4 / 6 | 6 | 2 | 0 |
| 0.2 | 2 / 6 | 6 | 2 | 0 |
| 0.5 | 1 / 6 | 6 | 4 | 0 |

## Every individual trial

| Scene | Weight | Process exit | Physical status | Safe goal | Progress (m) | Work consumed (kJ) | Abstentions |
|---|---:|---:|---|---|---:|---:|---:|
| diverse_v1_val_rolling_hills_00 | 0.0 | 0 | goal_reached | True | 225.32 | 336.88 | 0 |
| diverse_v1_val_ridge_passes_00 | 0.0 | 0 | goal_reached | True | 225.34 | 245.64 | 0 |
| diverse_v1_val_cross_slopes_00 | 0.0 | 0 | goal_reached | True | 225.37 | 260.86 | 0 |
| diverse_v1_val_valley_network_00 | 0.0 | 0 | goal_reached | True | 225.35 | 290.23 | 0 |
| diverse_v1_val_rough_mosaic_00 | 0.0 | 0 | timeout | False | 114.70 | 941.75 | 0 |
| diverse_v1_val_mixed_obstacles_00 | 0.0 | 0 | goal_reached | True | 225.33 | 269.46 | 0 |
| diverse_v1_val_rolling_hills_00 | 0.02 | 0 | goal_reached | True | 225.33 | 335.66 | 0 |
| diverse_v1_val_ridge_passes_00 | 0.02 | 0 | goal_reached | True | 225.32 | 244.14 | 0 |
| diverse_v1_val_cross_slopes_00 | 0.02 | 0 | goal_reached | True | 225.32 | 259.74 | 0 |
| diverse_v1_val_valley_network_00 | 0.02 | 0 | goal_reached | True | 225.28 | 290.63 | 0 |
| diverse_v1_val_rough_mosaic_00 | 0.02 | 0 | timeout | False | 95.80 | 6180.10 | 0 |
| diverse_v1_val_mixed_obstacles_00 | 0.02 | 0 | timeout | False | 86.11 | 10585.00 | 0 |
| diverse_v1_val_rolling_hills_00 | 0.2 | 0 | timeout | False | 60.97 | 9739.47 | 0 |
| diverse_v1_val_ridge_passes_00 | 0.2 | 0 | goal_reached | True | 225.32 | 239.40 | 0 |
| diverse_v1_val_cross_slopes_00 | 0.2 | 0 | goal_reached | True | 225.37 | 1551.62 | 0 |
| diverse_v1_val_valley_network_00 | 0.2 | 0 | goal_reached | False | 225.32 | 593.83 | 0 |
| diverse_v1_val_rough_mosaic_00 | 0.2 | 0 | rollover | False | 87.34 | 515.11 | 0 |
| diverse_v1_val_mixed_obstacles_00 | 0.2 | 0 | timeout | False | 71.74 | 8877.07 | 0 |
| diverse_v1_val_rolling_hills_00 | 0.5 | 0 | timeout | False | 60.89 | 10198.81 | 0 |
| diverse_v1_val_ridge_passes_00 | 0.5 | 0 | goal_reached | True | 225.27 | 240.47 | 0 |
| diverse_v1_val_cross_slopes_00 | 0.5 | 0 | timeout | False | 158.45 | 10283.64 | 0 |
| diverse_v1_val_valley_network_00 | 0.5 | 0 | timeout | False | 66.87 | 11850.80 | 0 |
| diverse_v1_val_rough_mosaic_00 | 0.5 | 0 | rollover | False | 86.61 | 542.54 | 0 |
| diverse_v1_val_mixed_obstacles_00 | 0.5 | 0 | timeout | False | 71.90 | 12360.79 | 0 |

Individual failed-run work is descriptive consumption, not evidence of an efficiency improvement.

## Paired safe-goal tradeoffs

| Scene | Weight | Work saving | Time increase |
|---|---:|---:|---:|
| diverse_v1_val_rolling_hills_00 | 0.02 | 0.4% | 0.0% |
| diverse_v1_val_ridge_passes_00 | 0.02 | 0.6% | 0.0% |
| diverse_v1_val_cross_slopes_00 | 0.02 | 0.4% | -1.8% |
| diverse_v1_val_valley_network_00 | 0.02 | -0.1% | -1.8% |
| diverse_v1_val_ridge_passes_00 | 0.2 | 2.5% | 5.9% |
| diverse_v1_val_cross_slopes_00 | 0.2 | -494.8% | 8.4% |
| diverse_v1_val_ridge_passes_00 | 0.5 | 2.1% | 5.3% |

## Selection status

`complete_valid_grid_no_qualifier_default_retained`. Selected weight: `0.02`; gate passed: `False`.
The full preregistered validation gate requires at least three safe pairs out of six scenes, mean work saving at least 5%, mean time increase at most 25%, and no unconditional safe-success count drop. Incomplete or invalid grids block selection and freeze.

This report does not open protected test scenes or establish generalization from a pilot.

Declared checkpoint kind: **last**; chosen step **5000** from completed update budget **5000**. Planning mode: **plan_once**. This declaration is specific to this grid and does not replace another grid's declared primary model.

## Rejected fresh reference counters

Counts concern invalid fresh geometry proposals, not predicted or measured terrain risk. Missing counter evidence stays unavailable.

| Energy weight | Rejected fresh proposals | Decisions with rejections | Trials with verified counters |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 6 / 6 |
| 0.02 | 0 | 0 | 6 / 6 |
| 0.2 | 0 | 0 | 6 / 6 |
| 0.5 | 0 | 0 | 6 / 6 |
