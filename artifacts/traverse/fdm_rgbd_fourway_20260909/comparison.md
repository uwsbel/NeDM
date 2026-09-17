# Four-route forecast / Chrono comparison

One shared measured starting observation. Route labels are frozen test hypotheses, not positive neural-network classifications.

| Route | Predicted4s progress (m) | Actual4s progress (m) | Progress error (m) | FDE (m) | Predicted contact | Contact≤4s | Predicted bounded | Bounded≤4s |
|---|---:|---:|---:|---:|---:|---|---:|---|
| MPPI selected | 17.246 | 17.183 | 0.063 | 0.403 | 6.927e-05% | no | 4.433e-05% | no |
| Rock-crossing test | 6.453 | 6.756 | -0.303 | 0.357 | 0.01033% | yes | 0.002261% | no |
| Terrain-stall test | 5.121 | 5.046 | 0.075 | 0.101 | 0.02562% | no | 0.05904% | no |
| Slower clear route | 11.971 | 11.655 | 0.316 | 0.319 | 0.0001985% | no | 0.004768% | no |

Full actual runs:

| Route | Goal | Goal time (s) | First contact (s) | Bounded motion confirmed (s) | Strict stall confirmed (s) |
|---|---|---:|---:|---:|---:|
| MPPI selected | yes | 6.850 | — | — | — |
| Rock-crossing test | no | — | 2.072 | 4.350 | 4.700 |
| Terrain-stall test | no | — | — | 7.550 | 9.500 |
| Slower clear route | yes | 9.650 | — | — | — |

Declared high-risk thresholds: contact35%, bounded motion50%. Bounded motion requires a fully observed, wholly future two-second interval; it can be confirmed later than slowdown begins.

An event after4s is outside the learned forecast horizon. The displayed time-to-goal cost extrapolates analytically beyond4s and is not a full-route learned prediction.

Among these four executed routes, 01_selected was the fastest safe goal completion; selected-route time regret was 0.000s. This does not establish optimality over all possible routes.

Shared initial observation independently verified: True. Frozen predictions, selected references, and execution provenance passed SHA checks.
