# Demo: planner scoring + three CRM rollouts for one held-out start-goal pair (f104_crm_eval_group_0013)

- `planner_scoring.png` - all 256 candidate routes scored by the CRM-trained network (5-member ensemble), the three driven
  routes highlighted; right panels: score of every route by rank, and each route's speed / detour.
- `video_1_planner_pick_goal_reached.mp4` - route A, the planner's pick (risk 0.4 %, rank 1): goal reached in 18.05 s.
- `video_2_suboptimal_goal_reached_slower.mp4` - route B (risk 46 %, rank 6): goal reached in 29.3 s.
- `video_3_risky_bogged_down.mp4` - route C, straight line at 2 m/s (risk 100 %, rank 185): stalls on the climb, one
  wheel spins (slip ratio in the hundreds), bogged down at 23.3 s.
- Other driven candidates (frames + outcomes kept): `run_164` (risk 10.6 %, goal in 28.55 s), `run_150` (risk 26 %, goal in 20.2 s).

How made: `scripts/crm_demo_pool.py` (rebuilds the locked evaluation pool from its md5 seed; its lowest-risk route equals
the locked evaluation pick), `scripts/crm_demo_render.py` (same CRM physics as the collection via `crm_collect.py`, run
locally on the RTX 5090, OptiX chase camera, ~50 s wall per rollout), `scripts/crm_demo_figure.py`.
The camera cannot see soil particles, so the ground drawn is the undeformed heightmap (no ruts); wheels sinking into
the soil show as wheels cutting below that surface. Local CUDA results matched the cluster run for route A (18.05 s vs 18.1 s).

## Soil-visible versions (VSG run-time visualiser)

- `video_soil_A_planner_pick.mp4` (goal reached in 18.25 s), `video_soil_B_suboptimal.mp4` (goal reached in 30.15 s),
  `video_soil_C_risky.mp4` (stalls on the climb, wheels spinning; ended by the blockage rule at 34.0 s).
- Made with `scripts/crm_demo_vsg.py` + `scripts/crm_demo_vsg_video.py` under the conda `nedm` Python: pychrono 10.0.0 there
  has VSG and the SPH particle plug-in (the source build used for the collection has VSG off). All 4.0 M soil particles
  are drawn, ~1 s per frame. Same routes / controller / soil settings, but a different Chrono version than the collection,
  so times differ slightly from the OptiX videos (18.05 / 29.3 s) and route C ends by blockage instead of dig-through.
- Gotchas: `vis.WriteImageToFile` segfaults in that conda build -> use the plug-in's `SetImageOutputDirectory` +
  `SetImageOutput(True)` (one `img_NNNNN.png` per `Render()`); Chrono data must be the conda copy
  (`$CONDA_PREFIX/share/chrono/data`, has `colormaps/` and `vsg/fonts`); needs `DISPLAY=:1`.
