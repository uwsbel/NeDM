# Frozen protected cohort: all 30 trials

Concrete freeze: `29684ce59cdd68e9b1f7310dcd2e63c55d021f6ff6712b83f99d286722efeb33`.

| Arm | Safe full goals / all trials | Complete verified artifacts | Timeouts | Process failures |
|---|---:|---:|---:|---:|
| selected_rgbd_time | 4 / 6 | 6 | 2 | 0 |
| selected_rgbd_energy | 4 / 6 | 6 | 2 | 0 |
| matched_blank_time | 3 / 6 | 6 | 3 | 0 |
| matched_blank_energy | 3 / 6 | 6 | 3 | 0 |
| original_best_receding_time | 4 / 6 | 6 | 2 | 0 |

## selected_rgbd_energy_vs_time

| Scene | Eligible safe pair | Work saving | Time increase | Reason if unavailable |
|---|---|---:|---:|---|
| diverse_v1_test_cross_slopes_00 | True | 0.65% | -0.12% |  |
| diverse_v1_test_mixed_obstacles_00 | True | -0.72% | -1.67% |  |
| diverse_v1_test_ridge_passes_00 | True | 1.01% | -0.12% |  |
| diverse_v1_test_rolling_hills_00 | False | unavailable | unavailable | Both trials did not reach verified schema-safe full goals |
| diverse_v1_test_rough_mosaic_00 | False | unavailable | unavailable | Both trials did not reach verified schema-safe full goals |
| diverse_v1_test_valley_network_00 | True | 0.36% | 0.00% |  |

## matched_blank_energy_vs_time

| Scene | Eligible safe pair | Work saving | Time increase | Reason if unavailable |
|---|---|---:|---:|---|
| diverse_v1_test_cross_slopes_00 | False | unavailable | unavailable | Both trials did not reach verified schema-safe full goals |
| diverse_v1_test_mixed_obstacles_00 | True | 0.00% | 0.00% |  |
| diverse_v1_test_ridge_passes_00 | False | unavailable | unavailable | Both trials did not reach verified schema-safe full goals |
| diverse_v1_test_rolling_hills_00 | True | 0.12% | 0.11% |  |
| diverse_v1_test_rough_mosaic_00 | True | 1.22% | 5.54% |  |
| diverse_v1_test_valley_network_00 | False | unavailable | unavailable | Both trials did not reach verified schema-safe full goals |

No parameters or model choices were selected from this protected evaluation. Original best-checkpoint receding planning is a separate diagnostic baseline. All individual trials are retained in the adjacent CSV/JSON.
