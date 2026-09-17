# Declared online cohort report

Every declared trial is counted: **4 trials**, **0 verified schema-safe full goals**.
Process completion and physical success are separate. Failed, missing, timed-out and abstaining trials remain in denominators.

| Energy weight (s/kJ) | Safe full goals / all trials | Valid artifacts | Timeouts | Trials with abstentions |
|---:|---:|---:|---:|---:|
| 0 | 0 / 2 | 2 | 2 | 2 |
| 0.02 | 0 / 2 | 2 | 2 | 2 |

## Every individual trial

| Scene | Weight | Process exit | Physical status | Safe goal | Progress (m) | Work consumed (kJ) | Abstentions |
|---|---:|---:|---|---|---:|---:|---:|
| diverse_v1_val_mixed_obstacles_00 | 0.0 | 0 | timeout | False | 53.04 | 3508.57 | 111 |
| diverse_v1_val_mixed_obstacles_00 | 0.02 | 0 | timeout | False | 65.93 | 2935.26 | 117 |
| diverse_v1_val_ridge_passes_00 | 0.0 | 0 | timeout | False | 31.42 | 4340.31 | 105 |
| diverse_v1_val_ridge_passes_00 | 0.02 | 0 | timeout | False | 31.42 | 5590.32 | 104 |

Individual failed-run work is descriptive consumption, not evidence of an efficiency improvement.

## Paired safe-goal tradeoffs

**No eligible paired safe full goals.** Time/work tradeoff fractions are unavailable; shorter or failed runs cannot be called more efficient.

## Selection status

`not_requested_diagnostic_only`. Selected weight: `None`; gate passed: `False`.
The full preregistered validation gate requires at least three safe pairs out of six scenes, mean work saving at least 5%, mean time increase at most 25%, and no unconditional safe-success count drop. Incomplete or invalid grids block selection and freeze.

This report does not open protected test scenes or establish generalization from a pilot.
