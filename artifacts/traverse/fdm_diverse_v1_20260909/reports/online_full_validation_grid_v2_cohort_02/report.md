# Declared online cohort report

Every declared trial is counted: **24 trials**, **0 verified schema-safe full goals**.
Process completion and physical success are separate. Failed, missing, timed-out and abstaining trials remain in denominators.

| Energy weight (s/kJ) | Safe full goals / all trials | Valid artifacts | Timeouts | Trials with abstentions |
|---:|---:|---:|---:|---:|
| 0 | 0 / 6 | 2 | 2 | 2 |
| 0.02 | 0 / 6 | 0 | 0 | 0 |
| 0.2 | 0 / 6 | 3 | 3 | 3 |
| 0.5 | 0 / 6 | 4 | 4 | 4 |

## Every individual trial

| Scene | Weight | Process exit | Physical status | Safe goal | Progress (m) | Work consumed (kJ) | Abstentions |
|---|---:|---:|---|---|---:|---:|---:|
| diverse_v1_val_rolling_hills_00 | 0.0 | 1 | process_failed | False | unavailable | unavailable | unavailable |
| diverse_v1_val_ridge_passes_00 | 0.0 | 0 | timeout | False | 132.21 | 5881.97 | 95 |
| diverse_v1_val_cross_slopes_00 | 0.0 | 1 | process_failed | False | unavailable | unavailable | unavailable |
| diverse_v1_val_valley_network_00 | 0.0 | 1 | process_failed | False | unavailable | unavailable | unavailable |
| diverse_v1_val_rough_mosaic_00 | 0.0 | 0 | timeout | False | 132.18 | 6041.73 | 81 |
| diverse_v1_val_mixed_obstacles_00 | 0.0 | 1 | process_failed | False | unavailable | unavailable | unavailable |
| diverse_v1_val_rolling_hills_00 | 0.02 | 1 | process_failed | False | unavailable | unavailable | unavailable |
| diverse_v1_val_ridge_passes_00 | 0.02 | 1 | process_failed | False | unavailable | unavailable | unavailable |
| diverse_v1_val_cross_slopes_00 | 0.02 | 1 | process_failed | False | unavailable | unavailable | unavailable |
| diverse_v1_val_valley_network_00 | 0.02 | 1 | process_failed | False | unavailable | unavailable | unavailable |
| diverse_v1_val_rough_mosaic_00 | 0.02 | 1 | process_failed | False | unavailable | unavailable | unavailable |
| diverse_v1_val_mixed_obstacles_00 | 0.02 | 1 | process_failed | False | unavailable | unavailable | unavailable |
| diverse_v1_val_rolling_hills_00 | 0.2 | 0 | timeout | False | 61.23 | 10311.87 | 168 |
| diverse_v1_val_ridge_passes_00 | 0.2 | 0 | timeout | False | 136.80 | 1533.21 | 133 |
| diverse_v1_val_cross_slopes_00 | 0.2 | 1 | process_failed | False | unavailable | unavailable | unavailable |
| diverse_v1_val_valley_network_00 | 0.2 | 1 | process_failed | False | unavailable | unavailable | unavailable |
| diverse_v1_val_rough_mosaic_00 | 0.2 | 0 | timeout | False | 108.45 | 6021.83 | 77 |
| diverse_v1_val_mixed_obstacles_00 | 0.2 | 1 | process_failed | False | unavailable | unavailable | unavailable |
| diverse_v1_val_rolling_hills_00 | 0.5 | 0 | timeout | False | 61.23 | 10693.48 | 167 |
| diverse_v1_val_ridge_passes_00 | 0.5 | 0 | timeout | False | 126.35 | 925.74 | 141 |
| diverse_v1_val_cross_slopes_00 | 0.5 | 1 | process_failed | False | unavailable | unavailable | unavailable |
| diverse_v1_val_valley_network_00 | 0.5 | 0 | timeout | False | 189.77 | 4927.75 | 70 |
| diverse_v1_val_rough_mosaic_00 | 0.5 | 0 | timeout | False | 96.84 | 5970.99 | 84 |
| diverse_v1_val_mixed_obstacles_00 | 0.5 | 1 | process_failed | False | unavailable | unavailable | unavailable |

Individual failed-run work is descriptive consumption, not evidence of an efficiency improvement.

## Paired safe-goal tradeoffs

**No eligible paired safe full goals.** Time/work tradeoff fractions are unavailable; shorter or failed runs cannot be called more efficient.

## Selection status

`blocked_incomplete_or_invalid_grid`. Selected weight: `None`; gate passed: `False`.
The full preregistered validation gate requires at least three safe pairs out of six scenes, mean work saving at least 5%, mean time increase at most 25%, and no unconditional safe-success count drop. Incomplete or invalid grids block selection and freeze.

This report does not open protected test scenes or establish generalization from a pilot.

Audit issues:
- diverse_v1_val_rolling_hills_00_time: Subprocess did not complete successfully; retained as failed trial
- diverse_v1_val_cross_slopes_00_time: Subprocess did not complete successfully; retained as failed trial
- diverse_v1_val_valley_network_00_time: Subprocess did not complete successfully; retained as failed trial
- diverse_v1_val_mixed_obstacles_00_time: Subprocess did not complete successfully; retained as failed trial
- diverse_v1_val_rolling_hills_00_energy002: Subprocess did not complete successfully; retained as failed trial
- diverse_v1_val_ridge_passes_00_energy002: Subprocess did not complete successfully; retained as failed trial
- diverse_v1_val_cross_slopes_00_energy002: Subprocess did not complete successfully; retained as failed trial
- diverse_v1_val_valley_network_00_energy002: Subprocess did not complete successfully; retained as failed trial
- diverse_v1_val_rough_mosaic_00_energy002: Subprocess did not complete successfully; retained as failed trial
- diverse_v1_val_mixed_obstacles_00_energy002: Subprocess did not complete successfully; retained as failed trial
- diverse_v1_val_cross_slopes_00_energy020: Subprocess did not complete successfully; retained as failed trial
- diverse_v1_val_valley_network_00_energy020: Subprocess did not complete successfully; retained as failed trial
- diverse_v1_val_mixed_obstacles_00_energy020: Subprocess did not complete successfully; retained as failed trial
- diverse_v1_val_cross_slopes_00_energy050: Subprocess did not complete successfully; retained as failed trial
- diverse_v1_val_mixed_obstacles_00_energy050: Subprocess did not complete successfully; retained as failed trial
- Grid does not use a single verifiable checkpoint
