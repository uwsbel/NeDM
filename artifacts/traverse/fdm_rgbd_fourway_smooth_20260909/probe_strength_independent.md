# Independent smooth-hill probe audit

Physical outcomes below use measured telemetry; no NN scoring or geometry changes.

| Case | Speed | Outcome | Bounded confirmation | Strict stop | Max chassis contact | Clean launch | Last5s diameter |
|---|---:|---|---:|---:|---:|---|---:|
| smooth_strength_gaussian_h6p5_s4p5_cx0 | 6.0 | goal_reached | None | None | 0.0N | True | 19.574m |
| smooth_strength_gaussian_h7_s4p5_cx0 | 6.0 | timeout | None | None | 0.0N | True | 1.286m |

All times are seconds from the measured launch. See JSON for source hashes, exact definitions, initial state, effort, rollback, loaded-wheel slip and force measurements.

Zero20Hz chassis resultant cannot exclude every transient/cancelling contact; no contact-pair census or exact underbody clearance.
Large slip of an unloaded wheel does not establish traction saturation. Report loaded slip separately.
The runner shaft torque-speed product and gear are diagnostics, not proof of an engine-power limit.
Geometry and speed were screened to construct a demonstration; these runs are not an independent performance test set.
