2026-09-15T18:36:20Z PLAN written; captures done (f104 re-capture byte-identical); tensors built; no sensor model trained yet
2026-09-15T18:37:43Z smoke passed (MI350, 5.6 steps/s); sweep submitted: E, E0, D, RGBD, RGB x 3 seeds
2026-09-15 sweep done (3 seeds each, dev G_unsafe mean): E 0.978, E0 0.979, D 0.982, RGBD 0.967, RGB 0.968. E within 0.01 of night-2 full-data 0.983 -> pipeline OK.
  Selection rule -> D (highest among D/RGBD/RGB). Zero-shot new arenas W/G: N2 0.950/0.893, E 0.949/0.890, D 0.932/0.861, RGB 0.851/0.626.
  Three jobs initially failed on miopenStatusInternalError (concurrent MIOpen cache); fixed with per-job MIOPEN_USER_DB_PATH. E0/RGBD sweeps re-run to save checkpoints (zero-shot only; selection unchanged).
  Deploy D (5 seeds, N2's deployment rows) submitted.
2026-09-15T19:09:34Z D deployed (5 seeds). Closed-loop picks built from captured maps for 1,200 fresh groups (5,428 episodes); submitted on mi2508x (4 x 128-core nodes)
2026-09-15T19:29:23Z primary failed (see PLAN results); follow-up declared: E0 deploy + fresh 1,200-group test
2026-09-15T20:04:42Z E0 deployed; follow-up picks built (e0, d, n2 + fixed2 + straight6) for 1,200 fresh groups; submitted on mi2508x
2026-09-15T20:49:09Z follow-up analysed (test2_results.json); REPORT.md written
2026-09-15T20:49:31Z deleted local copies of sensor_ds_gen_*.npz (900 MB; kept on the cluster, rebuildable with sensor_dataset.py)
