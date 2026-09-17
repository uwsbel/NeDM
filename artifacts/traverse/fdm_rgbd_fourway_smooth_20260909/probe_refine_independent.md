# Independent smooth-hill probe audit

Physical outcomes below use measured telemetry; no NN scoring or geometry changes.

| Case | Speed | Outcome | Bounded confirmation | Strict stop | Max chassis contact | Clean launch | Last5s diameter |
|---|---:|---|---:|---:|---:|---|---:|
| smooth_refine_gaussian_h6p6_s4p5_cx0 | 6.0 | goal_reached | None | None | 0.0N | True | 19.029m |
| smooth_refine_gaussian_h6p7_s4p5_cx0 | 6.0 | timeout | 14.55 | 15.3 | 0.0N | True | 0.016m |
| smooth_refine_gaussian_h6p8_s4p5_cx0 | 6.0 | goal_reached | None | None | 0.0N | True | 18.581m |
| smooth_refine_gaussian_h6p9_s4p5_cx0 | 6.0 | timeout | 19.8 | None | 0.0N | True | 1.225m |
| smooth_strength_gaussian_h6p5_s4p5_cx0 | 4.0 | timeout | 11.450000000000001 | 17.2 | 0.0N | True | 0.572m |
| smooth_strength_gaussian_h7_s4p5_cx0 | 4.0 | timeout | 13.100000000000001 | 13.450000000000001 | 0.0N | True | 0.539m |

All times are seconds from the measured launch. See JSON for source hashes, exact definitions, initial state, effort, rollback, loaded-wheel slip and force measurements.

Zero20Hz chassis resultant cannot exclude every transient/cancelling contact; no contact-pair census or exact underbody clearance.
Large slip of an unloaded wheel does not establish traction saturation. Report loaded slip separately.
The runner shaft torque-speed product and gear are diagnostics, not proof of an engine-power limit.
Geometry and speed were screened to construct a demonstration; these runs are not an independent performance test set.
