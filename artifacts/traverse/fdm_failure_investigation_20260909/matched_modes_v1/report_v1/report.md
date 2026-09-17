# Matched planning modes: post-hoc diagnostics

Post-hoc matched-mode diagnosis on previously opened data; does not replace protected scores. Mode changes are measured within an identical final checkpoint and first decision. Pause and route switching associations do not prove counterfactual causes; both unsafe modes remain failures. Failure timestamps are interval endpoints, not exact substep onset.

All 24 new trials and 24 existing controls retained. Audit errors: 0.

| Split | Energy weight | Plan once safe | Receding safe | Lost / gained safe | Matched pairs |
|---|---:|---:|---:|---:|---:|
| val | 0 | 5/6 | 4/6 | 1 / 0 | 6/6 |
| val | 0.02 | 4/6 | 4/6 | 0 / 0 | 6/6 |
| test | 0 | 4/6 | 4/6 | 2 / 2 | 6/6 |
| test | 0.02 | 4/6 | 5/6 | 0 / 1 | 6/6 |

| Scene | Weight | Plan once → receding | Contact first s | Blockage detected s | Pause s | Geometry changes after launch |
|---|---:|---|---:|---:|---:|---:|
| diverse_v1_test_cross_slopes_00 | 0.02 | retained_safe | None | None | 0.0 | 21 |
| diverse_v1_test_cross_slopes_00 | 0 | retained_safe | None | None | 0.0 | 28 |
| diverse_v1_test_mixed_obstacles_00 | 0.02 | retained_safe | None | None | 0.0 | 32 |
| diverse_v1_test_mixed_obstacles_00 | 0 | lost_safe | 118.54999999990325 | None | 24.000000000050292 | 60 |
| diverse_v1_test_ridge_passes_00 | 0.02 | retained_safe | None | None | 0.0 | 34 |
| diverse_v1_test_ridge_passes_00 | 0 | retained_safe | None | None | 0.0 | 30 |
| diverse_v1_test_rolling_hills_00 | 0.02 | both_unsafe | 30.89999999999331 | None | 19.00000000000616 | 41 |
| diverse_v1_test_rolling_hills_00 | 0 | gained_safe | None | None | 7.000000000006779 | 43 |
| diverse_v1_test_rough_mosaic_00 | 0.02 | gained_safe | None | None | 0.0 | 28 |
| diverse_v1_test_rough_mosaic_00 | 0 | gained_safe | None | None | 0.0 | 29 |
| diverse_v1_test_valley_network_00 | 0.02 | retained_safe | None | None | 0.0 | 26 |
| diverse_v1_test_valley_network_00 | 0 | lost_safe | None | None | 28.000000000021785 | 31 |
| diverse_v1_val_cross_slopes_00 | 0.02 | retained_safe | None | None | 0.0 | 29 |
| diverse_v1_val_cross_slopes_00 | 0 | retained_safe | None | None | 0.0 | 25 |
| diverse_v1_val_mixed_obstacles_00 | 0.02 | both_unsafe | None | None | 143.00000000013318 | 31 |
| diverse_v1_val_mixed_obstacles_00 | 0 | retained_safe | None | None | 9.000000000002117 | 34 |
| diverse_v1_val_ridge_passes_00 | 0.02 | retained_safe | None | None | 11.000000000008114 | 36 |
| diverse_v1_val_ridge_passes_00 | 0 | lost_safe | 6.949999999999368 | 16.05000000000154 | 53.000000000068326 | 86 |
| diverse_v1_val_rolling_hills_00 | 0.02 | retained_safe | None | None | 0.0 | 31 |
| diverse_v1_val_rolling_hills_00 | 0 | retained_safe | None | None | 0.0 | 31 |
| diverse_v1_val_rough_mosaic_00 | 0.02 | both_unsafe | 11.250000000000693 | None | 120.00000000016458 | 37 |
| diverse_v1_val_rough_mosaic_00 | 0 | both_unsafe | None | 97.44999999995242 | 148.0000000001393 | 24 |
| diverse_v1_val_valley_network_00 | 0.02 | retained_safe | None | None | 0.0 | 33 |
| diverse_v1_val_valley_network_00 | 0 | retained_safe | None | None | 0.0 | 31 |
