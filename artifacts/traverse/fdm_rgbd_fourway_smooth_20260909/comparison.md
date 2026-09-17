# Four-route forecast / Chrono comparison

One shared measured starting observation. Route labels are frozen test hypotheses, not positive neural-network classifications.

| Route | Predicted4s progress (m) | Actual4s progress (m) | Progress error (m) | FDE (m) | Predicted contact | Contact≤4s | Predicted bounded | Bounded≤4s |
|---|---:|---:|---:|---:|---:|---|---:|---|
| MPPI selected | 17.043 | 14.613 | 2.430 | 2.463 | 0.001533% | no | 1.919e-05% | no |
| Rock-crossing test | 8.731 | 6.608 | 2.122 | 2.251 | 1.141% | yes | 0.0009398% | no |
| Terrain-stall test | 9.645 | 11.710 | -2.064 | 2.494 | 0.08186% | no | 0.004351% | no |
| Slower clear route | 11.597 | 10.738 | 0.860 | 0.879 | 0.0008906% | no | 0.003666% | no |

Full actual runs:

| Route | Goal | Goal time (s) | First contact (s) | Bounded motion confirmed (s) | Strict stall confirmed (s) |
|---|---|---:|---:|---:|---:|
| MPPI selected | yes | 8.100 | — | — | — |
| Rock-crossing test | no | — | 2.090 | 4.450 | 4.900 |
| Terrain-stall test | no | — | — | 11.450 | 17.200 |
| Slower clear route | yes | 14.500 | — | — | — |

Declared high-risk thresholds: contact35%, bounded motion50%. Bounded motion requires a fully observed, wholly future two-second interval; it can be confirmed later than slowdown begins.

An event after4s is outside the learned forecast horizon. The displayed time-to-goal cost extrapolates analytically beyond4s and is not a full-route learned prediction.

Among these four executed routes, 01_selected was the fastest safe goal completion; selected-route time regret was 0.000s. This does not establish optimality over all possible routes.

Shared initial observation independently verified: True. Frozen predictions, selected references, and execution provenance passed SHA checks.
