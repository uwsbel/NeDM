# Four-route forecast / Chrono comparison

One shared measured starting observation. Route labels are frozen test hypotheses, not positive neural-network classifications.

| Route | Predicted4s progress (m) | Actual4s progress (m) | Progress error (m) | FDE (m) | Predicted contact | Contact≤4s | Predicted bounded | Bounded≤4s |
|---|---:|---:|---:|---:|---:|---|---:|---|
| MPPI selected | 17.043 | 14.613 | 2.430 | 2.463 | 0.001533% | no | 1.919e-05% | no |
| Rock-crossing test | Awaiting actual run | | | | | | | |
| Terrain-stall test | Awaiting actual run | | | | | | | |
| Slower clear route | Awaiting actual run | | | | | | | |

Full actual runs:

| Route | Goal | Goal time (s) | First contact (s) | Bounded motion confirmed (s) | Strict stall confirmed (s) |
|---|---|---:|---:|---:|---:|
| MPPI selected | yes | 8.100 | — | — | — |

Declared high-risk thresholds: contact35%, bounded motion50%. Bounded motion requires a fully observed, wholly future two-second interval; it can be confirmed later than slowdown begins.

An event after4s is outside the learned forecast horizon. The displayed time-to-goal cost extrapolates analytically beyond4s and is not a full-route learned prediction.

Comparison pending complete actual runs.

Shared initial observation independently verified: False. Frozen predictions, selected references, and execution provenance passed SHA checks.
