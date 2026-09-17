# Matched forecast horizon comparison

H60 first4s versus H20 full4s at identical measured anchors; best and fixed-final checkpoints separate

Primary planning model fixed before full training: H60 RGB-D, seed11, validation-best.

| Arm | Seed | Checkpoint | Group | H60 FDE m | H20 FDE m | H60 work MAE kJ | H20 work MAE kJ |
|---|---:|---|---|---:|---:|---:|---:|
| rgbd | 11 | last | all | 3.238 | 2.841 | 33.153 | 34.496 |
| rgbd | 11 | last | pre_first_observed_failure | 4.889 | 4.408 | 29.186 | 28.667 |
| rgbd | 11 | last | prior_observed_failure | 2.440 | 2.084 | 35.069 | 37.312 |
| rgbd | 11 | best | all | 3.026 | 2.841 | 34.749 | 34.496 |
| rgbd | 11 | best | pre_first_observed_failure | 4.234 | 4.408 | 38.274 | 28.667 |
| rgbd | 11 | best | prior_observed_failure | 2.443 | 2.084 | 33.046 | 37.312 |
| blank | 11 | last | all | 2.987 | 2.521 | 30.261 | 27.444 |
| blank | 11 | last | pre_first_observed_failure | 4.902 | 4.243 | 32.814 | 30.239 |
| blank | 11 | last | prior_observed_failure | 2.062 | 1.689 | 29.027 | 26.094 |
| blank | 11 | best | all | 3.544 | 2.721 | 33.746 | 30.446 |
| blank | 11 | best | pre_first_observed_failure | 5.664 | 4.536 | 41.997 | 32.167 |
| blank | 11 | best | prior_observed_failure | 2.519 | 1.844 | 29.760 | 29.615 |
| rgbd | 29 | last | all | 2.966 | 2.735 | 30.575 | 38.126 |
| rgbd | 29 | last | pre_first_observed_failure | 4.013 | 4.477 | 26.736 | 34.844 |
| rgbd | 29 | last | prior_observed_failure | 2.460 | 1.894 | 32.430 | 39.712 |
| rgbd | 29 | best | all | 2.965 | 3.044 | 30.846 | 44.377 |
| rgbd | 29 | best | pre_first_observed_failure | 4.440 | 4.435 | 28.504 | 41.380 |
| rgbd | 29 | best | prior_observed_failure | 2.253 | 2.371 | 31.978 | 45.825 |
| blank | 29 | last | all | 2.736 | 2.582 | 30.691 | 29.259 |
| blank | 29 | last | pre_first_observed_failure | 4.236 | 4.606 | 35.184 | 31.951 |
| blank | 29 | last | prior_observed_failure | 2.012 | 1.604 | 28.521 | 27.958 |
| blank | 29 | best | all | 3.186 | 2.582 | 29.733 | 29.259 |
| blank | 29 | best | pre_first_observed_failure | 4.921 | 4.606 | 35.019 | 31.951 |
| blank | 29 | best | prior_observed_failure | 2.347 | 1.604 | 27.179 | 27.958 |

Per-scene causal strata, supported hazard counts, endpoint/prefix risk metrics, signed attitude errors, and image controls are retained in comparison.json and each horizon's report.json.

These are validation forecast measurements. Full-route completion, failure, time, and mechanical work require the separate Chrono planner evaluation.
