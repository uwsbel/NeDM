# Goldens

Reference outputs of the ORIGINAL experiment code at commit
[`901d6c9`](https://github.com/uwsbel/NeDM/tree/901d6c9423a16c0fafc3d60056065415d5a725f2), against which the tests compare
the evaluation package bit for bit. Each folder was written by a generator that imports that commit's `scripts/` and
`src/` (they are not on main, so the generators are not kept here); two runs of each generator gave identical bytes.

| Folder | Original code | Holds |
|---|---|---|
| `config/` | `planner_arms.arm_specs`, `f104_n2_iter`, `ci_planner.py`, `nav_runner.py` | seed tags and seeds of every planner arm and M1 schedule |
| `suites/` | `nav_tasks`, the PR #4 small-check selections | lowest-md5 subsets and shard blocks |
| `labels/` | `ga_analyze.safe_labels`, `ov_eval_index.extras`, `gb_track_analyze.route_metrics` | per-drive labels of 45 real and 20 synthetic drives |
| `routes/` | `f104_n2_sampler`, `f104_n2_iter`, `gen_planner`, `ga_planner` | candidate pools, corridors, base routes, history windows |
| `planner/` | `f104_n2_iter.plan_iter` / `oneshot` with a float64 stub scorer; `ga_planner` / `ci_planner` scorers | search logs and picks; per-member route scores |
| `controllers/` | `gc_control.py` and the per-frame logic of `gen_collect_ext` / `crm_collect_ext` | `hold_clip`, actor outputs, held and tracker commands |
| `nav/` | `sensor_map_v2`, `nav_online`, `gen_planner.GridRiskModel` | depth-to-grid, pools, corridors, scores, `SpeedPI` |
