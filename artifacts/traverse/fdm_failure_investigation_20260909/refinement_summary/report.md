# Physical MPPI refinement counterfactuals

All nine headless AMD trials pass source/runtime/input and artifact checks. All four selected-reference controls reproduce every one of 241 stored physical arrays exactly. Each parent reference also reproduces identically across the two cost settings. Original protected scores remain unchanged.

Contact timestamps below are starts of the first positive 50 ms contact interval, not exact contact instants. For example, rough-time first contact lies in [11.70, 11.75] s.

| Scene / cost | Executed reference | Frozen model cost | Safe goal | Duration | First contact interval start |
|---|---|---:|---|---:|---:|
| Rolling hills / time | replay_selected | 45.477 | False | 180.00 s | 29.70 s |
| Rolling hills / time | parent_family | 45.555 | False | 180.00 s | 31.50 s |
| Rough mosaic / time | replay_selected | 46.475 | False | 180.00 s | 11.70 s |
| Rough mosaic / time | parent_family | 48.721 | False | 120.25 s | 11.75 s |
| Rolling hills / energy | replay_selected | 59.627 | False | 180.00 s | 30.45 s |
| Rolling hills / energy | parent_family | 59.726 | False | 180.00 s | 31.50 s |
| Rough mosaic / energy | replay_selected | 62.394 | False | 180.00 s | 12.50 s |
| Rough mosaic / energy | parent_family | 66.388 | False | 120.25 s | 11.75 s |
| Rough mosaic / time | best_unrefined | 47.608 | True | 41.30 s | none |

The rough-mosaic time/risk comparison establishes a harmful ranking change in this case. The original best family (index 12, −44 m at 6 m/s) has model cost 47.608 and safely finishes in 41.30 s. Another parent (index 6, −22 m at 6 m/s) costs 48.721, but MPPI refinement lowers its predicted cost to 46.475 and makes it the winner. That executed route contacts within 12 s, blocks and times out at 180 s.

All four selected parent references are already unsafe. The rough-mosaic parent eventually reaches the goal at 120.25 s after contact and blockage, while its refinements time out. Thus local reference changes worsen that outcome but do not create its initial unsafe classification. Rolling-hills parents and refinements both contact and block. Small MPPI changes are not necessary for every failure.

This isolates the executed reference, including both geometry and speed. It does not show that removing MPPI improves a general success rate. The single additional best-base reference was chosen by the original model-cost argmin; it was not chosen using its subsequently measured success.

The wrapper substitutes one archived launch decision and imports the immutable online_v9 physical loop and native PID adapter. It verifies the exact launch history, pose, map, goal and candidate-family arrays. Sidecar counterfactual_audit.json files identify the intervention; the inherited online_protocol.json describes the unchanged engine, not a fresh neural inference. Parent probabilities must be read from the matching original family_scores row.

The earlier eight-job wrapper attempt 412221 stopped before traversal because its exact goal guard compared a float32 observation copy with the runner’s float64 case goal. Version 2 uses the same case goal as the frozen runner, retaining exact equality. That failed harness and all logs remain archived.

[Cost and physical-route figure](ranking_counterfactual.png) · [Machine-readable report](report.json)
