2026-09-15T03:48:09Z plan written, no gen_v1 Chrono job submitted yet
2026-09-15T03:55:37Z submitted generalisation test (6,639 episodes, jobs 420022) and data collection (9,000 episodes, jobs 420037-420039)
2026-09-15T04:03:15Z PLAN amendment 1 (mission geometry infeasible; no mission run yet)
2026-09-15T04:14:49Z mission pilot passed (runner + sharp-turn route fallback); submitted 600 mission runs (200 missions x 3 arms)
2026-09-15T04:17:58Z mission planner fix: degenerate candidates rejected instead of crashing (17 crashed runs), long-way arc fallback added (5 no-route legs near the arena edge); affected mission runs to be re-run with the fixed code
2026-09-15T04:21:36Z test shards 28-47 (not started) cancelled on mi2101x and re-queued as two single-node tasks (mi3508x, mi2508x); all arms of a group still share one node
2026-09-15T04:26:22Z 36 missions (any arm crashed or hit no-route under the pre-fix planner) moved to missions_run/superseded_before_fix/ and re-run with all three arms together
2026-09-15T04:43:29Z all runs complete: test 6,639/6,639, missions 600/600 (36 re-run), data 9,000/9,000; 0 collection failures
2026-09-15T04:45:56Z analysis done: REPORT.md, results.png, test_results.json, mission_results.json, data_ranking.json, station_ds_gen_v1.npz
