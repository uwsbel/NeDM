# Declared online cohort report

Every declared trial is counted: **24 trials**, **6 verified schema-safe full goals**.
Process completion and physical success are separate. Failed, missing, timed-out and abstaining trials remain in denominators.

| Energy weight (s/kJ) | Safe full goals / all trials | Valid artifacts | Timeouts | Trials with abstentions |
|---:|---:|---:|---:|---:|
| 0 | 3 / 6 | 6 | 2 | 0 |
| 0.02 | 2 / 6 | 6 | 4 | 0 |
| 0.2 | 1 / 6 | 6 | 4 | 0 |
| 0.5 | 0 / 6 | 6 | 6 | 0 |

## Every individual trial

| Scene | Weight | Process exit | Physical status | Safe goal | Progress (m) | Work consumed (kJ) | Abstentions |
|---|---:|---:|---|---|---:|---:|---:|
| diverse_v1_val_rolling_hills_00 | 0.0 | 0 | timeout | False | 61.08 | 11058.68 | 0 |
| diverse_v1_val_ridge_passes_00 | 0.0 | 0 | goal_reached | True | 225.32 | 244.41 | 0 |
| diverse_v1_val_cross_slopes_00 | 0.0 | 0 | goal_reached | True | 225.30 | 259.73 | 0 |
| diverse_v1_val_valley_network_00 | 0.0 | 0 | goal_reached | True | 225.26 | 292.22 | 0 |
| diverse_v1_val_rough_mosaic_00 | 0.0 | 0 | rollover | False | 55.44 | 264.57 | 0 |
| diverse_v1_val_mixed_obstacles_00 | 0.0 | 0 | timeout | False | 86.12 | 10651.80 | 0 |
| diverse_v1_val_rolling_hills_00 | 0.02 | 0 | timeout | False | 61.08 | 11058.68 | 0 |
| diverse_v1_val_ridge_passes_00 | 0.02 | 0 | goal_reached | True | 225.34 | 244.40 | 0 |
| diverse_v1_val_cross_slopes_00 | 0.02 | 0 | goal_reached | True | 225.34 | 259.06 | 0 |
| diverse_v1_val_valley_network_00 | 0.02 | 0 | timeout | False | 67.34 | 12203.18 | 0 |
| diverse_v1_val_rough_mosaic_00 | 0.02 | 0 | timeout | False | -0.55 | 5861.55 | 0 |
| diverse_v1_val_mixed_obstacles_00 | 0.02 | 0 | timeout | False | 89.80 | 7842.42 | 0 |
| diverse_v1_val_rolling_hills_00 | 0.2 | 0 | timeout | False | 61.19 | 10739.00 | 0 |
| diverse_v1_val_ridge_passes_00 | 0.2 | 0 | goal_reached | True | 225.35 | 238.52 | 0 |
| diverse_v1_val_cross_slopes_00 | 0.2 | 0 | timeout | False | 219.64 | 2781.51 | 0 |
| diverse_v1_val_valley_network_00 | 0.2 | 0 | timeout | False | 67.25 | 12128.54 | 0 |
| diverse_v1_val_rough_mosaic_00 | 0.2 | 0 | rollover | False | 92.33 | 627.99 | 0 |
| diverse_v1_val_mixed_obstacles_00 | 0.2 | 0 | timeout | False | 140.96 | 5328.32 | 0 |
| diverse_v1_val_rolling_hills_00 | 0.5 | 0 | timeout | False | 61.19 | 10982.60 | 0 |
| diverse_v1_val_ridge_passes_00 | 0.5 | 0 | timeout | False | 36.25 | 8302.24 | 0 |
| diverse_v1_val_cross_slopes_00 | 0.5 | 0 | timeout | False | 54.33 | 7687.54 | 0 |
| diverse_v1_val_valley_network_00 | 0.5 | 0 | timeout | False | 66.90 | 11831.63 | 0 |
| diverse_v1_val_rough_mosaic_00 | 0.5 | 0 | timeout | False | 96.93 | 8866.74 | 0 |
| diverse_v1_val_mixed_obstacles_00 | 0.5 | 0 | timeout | False | 139.29 | 6618.78 | 0 |

Individual failed-run work is descriptive consumption, not evidence of an efficiency improvement.

## Paired safe-goal tradeoffs

| Scene | Weight | Work saving | Time increase |
|---|---:|---:|---:|
| diverse_v1_val_ridge_passes_00 | 0.02 | 0.0% | -0.1% |
| diverse_v1_val_cross_slopes_00 | 0.02 | 0.3% | 0.0% |
| diverse_v1_val_ridge_passes_00 | 0.2 | 2.4% | 5.5% |

## Selection status

`complete_valid_grid_no_qualifier_default_retained`. Selected weight: `0.02`; gate passed: `False`.
The full preregistered validation gate requires at least three safe pairs out of six scenes, mean work saving at least 5%, mean time increase at most 25%, and no unconditional safe-success count drop. Incomplete or invalid grids block selection and freeze.

This report does not open protected test scenes or establish generalization from a pilot.

Declared checkpoint kind: **best**; chosen step **4000** from completed update budget **5000**. Planning mode: **plan_once**. This declaration is specific to this grid and does not replace another grid's declared primary model.

## Rejected fresh reference counters

Counts concern invalid fresh geometry proposals, not predicted or measured terrain risk. Missing counter evidence stays unavailable.

| Energy weight | Rejected fresh proposals | Decisions with rejections | Trials with verified counters |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 6 / 6 |
| 0.02 | 0 | 0 | 6 / 6 |
| 0.2 | 0 | 0 | 6 / 6 |
| 0.5 | 0 | 0 | 6 / 6 |
