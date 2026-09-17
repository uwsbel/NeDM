# Diverse-terrain HMMWV FDM / MPPI evidence

The first broader-terrain campaign collected 540 fixed-reference Chrono rollouts across 36 heightmap arenas, trained eight matched models on AMD, and evaluated frozen planners on six protected test arenas. RGB-D plan-once planning reaches 4/6 safe full goals versus 3/6 for the matched blank model. Mean per-scene mechanical-work savings over four paired safe completions are only 0.33%; the joint time/risk/energy milestone is not established.

- [Campaign results and limits](../../../docs/vision/hmmwv_traverse/fdm_diverse_results_20260909.md)
- [All 30 protected planner trials](reports/online_protected_test_v1_cohort_01/report.md)
- [Protected outcome matrix and paired work/time](reports/protected_test_visuals_v2/all_30_trials.png)
- [Terrain families](geometry/terrain_families_preview.png)
- [Chrono demo index, forecast provenance and telemetry](demos/protected_v1/index.md)
- [Successful time/risk Chrono video, 5 Hz](demos/protected_v1/cross_time/report/selected_trial/actual_chrono.mp4)
- [Successful energy-aware Chrono video, 5 Hz](demos/protected_v1/cross_energy/report/selected_trial/actual_chrono.mp4)
- [Contact/blockage failure Chrono video, full 180 s at 1 Hz](demos/protected_v1/rolling_failure_1hz/report/selected_trial/actual_chrono.mp4)
- [MPPI candidates, forecast and actual route](demos/protected_v1/cross_time/report/selected_trial/overview.png)
- [Protected forecast and reference-support audit](reports/protected_test_offline_final_v2/report.md)
- [Rich telemetry schema](../../../docs/vision/hmmwv_traverse/fdm_rich_telemetry_schema_20260909.md)
- [Forecast validation evidence](reports/full_learning_analysis_v2/evidence.md)
- [Initial-reference diagnostic](reports/initial_reference_v1/diagnostic.md)
- [Validation configuration selection](reports/final_validation_selection_v1/selection.md)
- [Concrete test freeze](protected_test_freeze_v1.json)
- [Predeclared evaluation gates](evaluation_plan_v1.json)
- [Machine-readable progress](progress.json)

All working files are isolated in `/home/harry/NeDM-traverse_mppi` on branch `traverse_mppi`. Physics, sensors and optimizer updates ran on AMD under `/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909`. Large raw data/packs and every immutable job source remain there; this local directory contains checksummed reports, metadata, checkpoints and selected media.

Mechanical work means integrated positive engine-interface work, not fuel use. The map is a single measured overhead RGB-D snapshot, and native PID path altitude still uses simulator terrain height. RigidTerrain/TMEASY exposes vehicle/tire dynamics, not deformable-soil sinkage or ruts. Test data were unsealed only after the concrete model/cost/protocol freeze, and never entered optimizer updates.
