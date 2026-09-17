# Final configuration selection from validation

All three configurations must have complete, audited 24-trial grids. No protected test data is read and no test freeze is written.

| Planning mode | Checkpoint | Selected energy weight | Safe goals at selected weight | Safe goals at zero weight | Energy gate |
|---|---|---:|---:|---:|---|
| receding | best step 4000 | 0.02 | 4 / 6 | 4 / 6 | False |
| plan_once | best step 4000 | 0.02 | 2 / 6 | 3 / 6 | False |
| plan_once | last step 5000 | 0.02 | 4 / 6 | 5 / 6 | False |

Validation-selected configuration: **plan_once, last step 5000, energy weight 0.02 s/kJ**.
A concrete checksum-bound test freeze is still required. Original best-checkpoint receding planning remains a reported baseline.
