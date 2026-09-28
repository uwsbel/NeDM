# LOG: Polaris and M113 on f104 soil (started 2026-09-27 22:37 CDT)

Branch `offroad_vehicles_v1` (from `arena_gator_v1` 051dfdba5). Local root K4 = `artifacts/traverse/offroad_vehicles_20260927/`.

## Request (user, 2026-09-27 evening)
"for tonight, I want you to schedule a long session to run. I want you to investigate feasibility for soil traversing
for chrono vehicle's polaris and M113 tracked vehicle. It's supposed to design as off-road vehicle. First do some smoke
test to see if it has better traversibility than gator on soil condition. if it's better than gator, then collect same
amount of data as gator and see planner trained on that data could achieve good success rate or not."

Standing context: success bar = the trained planner, in the loop, reaches the goal safely on 90 %+ of pairs; the user
asked (09-27) that the best planner formulation be the default, and suggested bigger / wider route search (offline
probe `artifacts/traverse/search_probe_20260927`: wider route shapes found model-rated-safe routes on 71 of 134 risky
Gator pairs; not yet driven).

## Log
- 22:37 branch + root created. Cluster: 0 of my jobs; mi3501x 5 idle, rest mostly allocated.
- 23:42 scouts S1-S5 + critic done (Polaris feasible and much better than the Gator in local smokes; M113 7.7x cost, slides on soil slopes, stock gearing weak). PLAN.md written and hashed (e166e1fb6338).
- 23:43 build workflow launched (M1 Polaris+dispatcher, M2 M113 bounded, M3 staging/tasks/smoke analysis, M4 planner/eval chain, R1 plan review), each with verifier + one fix round. Stage 0 driveability folded into the M1/M2 check jobs.
d7b9141e50f04f56c61b5afbf47459331417ab684d0da597edb5f127e4179a25  scripts/ov_smoke_analyze.py
- 00:53 build workflow done: M1 (Polaris + dispatcher), M2 (M113: GO, with the shafts-brake fix; 4.1x the Gator's cost), M3 (staging/tasks/launch/smoke analysis), M4 (planner/eval chain; Gator G_full_grad and HMMWV H_full_grad picks for the 800 suite locked, cf0396c4) all pass independent verification; REVIEW_R1: 2 blockers + 9 should-fix.
- 01:05 PLAN amendment 9.1 (R1 B1, B2, S1-S6, S8, S9; M113 definition); PLAN.md re-hashed d78ec15b. Smoke task files: tasks/smoke_v2_polaris.json (a4eaf300, 916 rows, sample B at tier -2) and tasks/smoke_m113_v1.json (bedca166, 432 rows). The smoke decision rule is scripts/ov_smoke_analyze.py (sha above), frozen before any drive.
- 01:05 G4 staged for real (103 frozen files incl. ov_m113.py, crm_m113.json; record stage_records/stage_real_20260928_005535.txt). Launched: Polaris smoke job 441596 (mi3501x x3, 4 h), M113 smoke job 441599 (mi3501x x2, 4,000 s episode timeout).
