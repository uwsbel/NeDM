# Initial fixed-reference diagnostic

All 90 original references on six validation scenes. CPU inference uses the exact frozen online scorer and measured float32 initial RGB-D. This is a fixed-library diagnostic, not an MPPI rollout or model-selection result.

Measured safe routes: 78/90 over the first 12 seconds; 21/90 over the full traversal. 57 references are safe for 12 seconds but fail the full-route safety/completion criterion.

| Model | ADE / FDE, m | Progress MAE, m | Work MAE, kJ | Chosen / safe full goals (energy 0) | Chosen / safe full goals (energy .02) |
|---|---:|---:|---:|---:|---:|
| rgbd_best4000 | 2.031 / 3.279 | 2.843 | 108.116 | 6 / 3 | 6 / 2 |
| rgbd_last5000 | 2.243 / 4.867 | 3.142 | 72.486 | 6 / 5 | 6 / 4 |
| blank_best2000 | 3.287 / 7.473 | 3.460 | 76.663 | 6 / 0 | 6 / 0 |

The oracle below substitutes measured first-12-second trajectories, work, event indicators and endpoint attitudes into the unchanged planning cost. It reveals the cost heuristic's full-route extrapolation limit even with perfect short-horizon forecasts. It is diagnostic truth access, never a model input or deployed planner.

| Scene | Energy | RGB-D best winner / safe full | Best actual safe route rank | Actual safe routes passing model risk | Truth-12s oracle winner / safe full |
|---|---:|---|---:|---:|---|
| diverse_v1_val_rolling_hills_00 | 0.0 | 2 / False | 2 | 6 | 14 / True |
| diverse_v1_val_rolling_hills_00 | 0.02 | 2 / False | 2 | 6 | 14 / True |
| diverse_v1_val_ridge_passes_00 | 0.0 | 14 / True | 1 | 3 | 14 / True |
| diverse_v1_val_ridge_passes_00 | 0.02 | 14 / True | 1 | 3 | 14 / True |
| diverse_v1_val_cross_slopes_00 | 0.0 | 11 / True | 1 | 3 | 11 / True |
| diverse_v1_val_cross_slopes_00 | 0.02 | 11 / True | 1 | 3 | 11 / True |
| diverse_v1_val_valley_network_00 | 0.0 | 11 / True | 1 | 3 | 5 / False |
| diverse_v1_val_valley_network_00 | 0.02 | 2 / False | 3 | 3 | 5 / False |
| diverse_v1_val_rough_mosaic_00 | 0.0 | 2 / False | 4 | 3 | 8 / False |
| diverse_v1_val_rough_mosaic_00 | 0.02 | 2 / False | 4 | 3 | 8 / False |
| diverse_v1_val_mixed_obstacles_00 | 0.0 | 5 / False | 2 | 3 | 5 / False |
| diverse_v1_val_mixed_obstacles_00 | 0.02 | 5 / False | 5 | 3 | 5 / False |

`diagnostic.json` retains every route, both cost modes and three checkpoints, measured full-route consequences, censoring, numerical input comparisons, CPU-versus-saved forecast comparisons, checkpoint/source hashes and complete cost breakdowns. Predictions are exported in three small 90-route NPZ files. Engine-interface work is mechanical work, not fuel consumption.
