# Full train/validation collection and reference-control support

All **450 prescribed routes and 30 one-time RGB-D observations** completed on AMD in **39m05s**, with no failed tasks. There are **109 goal completions, of which 103 meet the recorded safety criteria**. Every one of the 30 arenas has a safe reference control at each of 2, 4, and 6 m/s. These are measured Chrono PID reference rollouts; they do not establish learned MPPI success.

The immutable `campaign_v2` cohort contains 360 train and 90 validation routes. Every arena retains all 15 routes: five lateral offsets (0, -22, +22, -44, +44 m), each at three speeds (2, 4, 6 m/s), with a 180 s cap. No outcome is removed or replaced. Protected test data was not inspected.

**Safety and work definitions.** A safe completion reaches the goal with no asset or chassis contact resultant above 1 N, no two-second bounded-motion event (diameter at most 0.25 m under sustained throttle above 0.3, excluding parking and arrival), and no solver-step absolute roll or pitch above 60 degrees. Strict near-stop is recorded separately. Work is positive engine-interface mechanical work integrated at every physics step, not fuel consumption.

**Integrity and telemetry.** All 450 observation/physics joins agree, with one source fingerprint and one runtime fingerprint. All 1,347,545 logged intervals contain solver-step work and post-step risk observations. The 198 telemetry fields include finite engine power/torque/speed, all four native tire slips and forces, suspension measurements, and attitude. Only the desired-speed and parked-command terminal placeholders are missing by design. Raw route outputs occupy 3,260,077,395 bytes; the separate full payload audit includes observations and verifies 3,345,806,241 bytes. Source/runtime audit verifies 74 source and 108 runtime/HMMWV files.

**Validation control support.** Each row has all 15 routes evaluated. Lowest-work and fastest controls show an actual time/work tradeoff available to later planning. All safe validation controls use an outer offset; this result establishes feasible routes in the prescribed set, with learned route selection still to be evaluated.

| Validation family | Goals / safe | Safe offset(s), m | Lowest work: offset, speed; time; work | Fastest safe: offset, speed; time; work |
|---|---:|---|---|---|
| rolling_hills | 6 / 6 | -44, +44 | +44 m, 2 m/s; 123.10 s; 231.92 kJ | +44 m, 6 m/s; 41.30 s; 292.82 kJ |
| ridge_passes | 3 / 3 | +44 | +44 m, 2 m/s; 123.20 s; 179.96 kJ | +44 m, 6 m/s; 41.30 s; 243.55 kJ |
| cross_slopes | 3 / 3 | -44 | -44 m, 2 m/s; 123.20 s; 185.50 kJ | -44 m, 6 m/s; 41.30 s; 262.37 kJ |
| valley_network | 3 / 3 | -44 | -44 m, 2 m/s; 123.20 s; 205.38 kJ | -44 m, 6 m/s; 41.35 s; 292.59 kJ |
| rough_mosaic | 3 / 3 | +44 | +44 m, 2 m/s; 123.25 s; 226.57 kJ | +44 m, 6 m/s; 41.35 s; 304.83 kJ |
| mixed_obstacles | 3 / 3 | -44 | -44 m, 2 m/s; 123.20 s; 188.34 kJ | -44 m, 6 m/s; 41.35 s; 271.13 kJ |

**All-scene support.** Every scene below has 15 completed references. The full JSON retains each route’s outcome, contact maxima, attitude maxima, blockage flags, time, work, and completion-marker hash.

| Scene | Goals | Safe goals | Safe indices (0–14) |
|---|---:|---:|---|
| diverse_v1_train_rolling_hills_00 | 7 | 6 | 9, 10, 11, 12, 13, 14 |
| diverse_v1_train_rolling_hills_01 | 3 | 3 | 12, 13, 14 |
| diverse_v1_train_rolling_hills_02 | 7 | 6 | 9, 10, 11, 12, 13, 14 |
| diverse_v1_train_rolling_hills_03 | 6 | 6 | 9, 10, 11, 12, 13, 14 |
| diverse_v1_train_ridge_passes_00 | 3 | 3 | 9, 10, 11 |
| diverse_v1_train_ridge_passes_01 | 3 | 3 | 12, 13, 14 |
| diverse_v1_train_ridge_passes_02 | 3 | 3 | 9, 10, 11 |
| diverse_v1_train_ridge_passes_03 | 3 | 3 | 9, 10, 11 |
| diverse_v1_train_cross_slopes_00 | 5 | 4 | 9, 10, 11, 14 |
| diverse_v1_train_cross_slopes_01 | 3 | 3 | 9, 10, 11 |
| diverse_v1_train_cross_slopes_02 | 3 | 3 | 9, 10, 11 |
| diverse_v1_train_cross_slopes_03 | 3 | 3 | 12, 13, 14 |
| diverse_v1_train_valley_network_00 | 3 | 3 | 9, 10, 11 |
| diverse_v1_train_valley_network_01 | 3 | 3 | 9, 10, 11 |
| diverse_v1_train_valley_network_02 | 3 | 3 | 12, 13, 14 |
| diverse_v1_train_valley_network_03 | 3 | 3 | 9, 10, 11 |
| diverse_v1_train_rough_mosaic_00 | 4 | 3 | 9, 10, 11 |
| diverse_v1_train_rough_mosaic_01 | 3 | 3 | 9, 10, 11 |
| diverse_v1_train_rough_mosaic_02 | 3 | 3 | 9, 10, 11 |
| diverse_v1_train_rough_mosaic_03 | 4 | 3 | 12, 13, 14 |
| diverse_v1_train_mixed_obstacles_00 | 3 | 3 | 9, 10, 11 |
| diverse_v1_train_mixed_obstacles_01 | 3 | 3 | 9, 10, 11 |
| diverse_v1_train_mixed_obstacles_02 | 4 | 3 | 12, 13, 14 |
| diverse_v1_train_mixed_obstacles_03 | 3 | 3 | 9, 10, 11 |
| diverse_v1_val_rolling_hills_00 | 6 | 6 | 9, 10, 11, 12, 13, 14 |
| diverse_v1_val_ridge_passes_00 | 3 | 3 | 12, 13, 14 |
| diverse_v1_val_cross_slopes_00 | 3 | 3 | 9, 10, 11 |
| diverse_v1_val_valley_network_00 | 3 | 3 | 9, 10, 11 |
| diverse_v1_val_rough_mosaic_00 | 3 | 3 | 12, 13, 14 |
| diverse_v1_val_mixed_obstacles_00 | 3 | 3 | 9, 10, 11 |

**Family and speed coverage.**

| Family | Goals / 75 | Safe goals | Safe at 2 / 4 / 6 m/s |
|---|---:|---:|---|
| cross_slopes | 17 | 16 | 5 / 5 / 6 |
| mixed_obstacles | 16 | 15 | 5 / 5 / 5 |
| ridge_passes | 15 | 15 | 5 / 5 / 5 |
| rolling_hills | 29 | 27 | 9 / 9 / 9 |
| rough_mosaic | 17 | 15 | 5 / 5 / 5 |
| valley_network | 15 | 15 | 5 / 5 / 5 |

**Reproducible evidence.**

- [Full collection audit](../../../artifacts/traverse/fdm_diverse_v1_20260909/full_collection_audit_412066.json)
- [All 450 route outcomes and safe time/work support](../../../artifacts/traverse/fdm_diverse_v1_20260909/full_route_support.json)
- [Frozen source and runtime audit](../../../artifacts/traverse/fdm_diverse_v1_20260909/full_frozen_inputs_audit_412066.json)
- [Read-only audit scripts and hashes](../../../artifacts/traverse/fdm_diverse_v1_20260909/audits/full_collection_readonly_v1/audit_sha256.json)

Source snapshot: `/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909/snapshots/campaign_v2`. Data: the sibling `full_cohort_v2` directory. Manifest SHA-256: `2e066280e8f667eb020fd455a12017579f6543546262bfb3687363e3821f02e1`. Audit observations: 2026-09-09T22:06:34.543141+00:00.
