# Causal-anchor strata: first AMD FDM pilot

The very high pooled low-progress AUC mainly reflects recognizing vehicles already moving slowly. This pilot contains very few prospective low-progress entries.

These are post-hoc diagnostics of **best validation-loss checkpoints**, on the same development validation split used for checkpoint selection. The prediction horizon is four seconds. All inputs defining anchor strata are causal; recorded futures supply targets only.

| Causal anchor group | Known low-progress windows | Positive windows / episodes | Known contact windows | Positive windows / episodes |
|---|---:|---:|---:|---:|
| all | 3378 | 61 / 9 | 4280 | 136 / 11 |
| moving | 3055 | 4 / 4 | 3558 | 42 / 10 |
| slow | 58 | 55 / 8 | 449 | 92 / 9 |
| already_low_progress | 48 | 48 / 8 | 80 | 80 / 8 |
| moving_before_first_contact | 3052 | 2 / 2 | 3551 | 40 / 10 |
| before_first_contact | 3318 | 2 / 2 | 4178 | 44 / 11 |
| recent_contact | 59 | 59 / 8 | 92 | 92 / 9 |

Moving means current vx≥1 m/s and at least 0.5 m displacement in the past 0.75 s, with fully recorded history. Already low progress means past displacement<0.1125 m under mean previous throttle>0.3. Before first contact means no previous recorded asset-contact interval. Slow means |current vx|<0.5 m/s. Groups overlap. Future route-end parking is excluded only by the original low-progress target mask.

| Best-loss checkpoint | Step | All ADE / 4 s FDE, m | Moving 4 s FDE, m | Low-progress AUC, all | Low-progress AUC, moving | Contact AUC, moving before first contact |
|---|---:|---:|---:|---:|---:|---:|
| history_s11 | 750 | 0.7758 / 0.9804 | 0.9013 | 0.9997 | 0.9961 | 0.9434 |
| history_s29 | 1000 | 0.7440 / 0.9443 | 0.8670 | 0.9996 | 0.9951 | 0.9568 |
| no_history_s11 | 750 | 0.7792 / 1.0009 | 0.9314 | 0.9997 | 0.9963 | 0.9658 |
| no_history_s29 | 1000 | 0.7427 / 0.9590 | 0.8924 | 0.9996 | 0.9951 | 0.9739 |
| no_terrain_s11 | 750 | 0.7746 / 0.9934 | 0.9141 | 0.9998 | 0.9967 | 0.9275 |
| no_terrain_s29 | 1000 | 0.7569 / 0.9745 | 0.8838 | 0.9997 | 0.9968 | 0.9379 |
| profile_s11 | 1000 | 0.7027 / 0.8767 | 0.8141 | 0.9993 | 0.9895 | 0.9565 |
| profile_s29 | 1000 | 0.7108 / 0.9089 | 0.8356 | 0.9991 | 0.9861 | 0.9532 |

Nominal-reference kinematics: all ADE/FDE 0.959/1.305 m; moving FDE 0.999 m.

| Causal heuristic for low progress | All AUC | Moving AUC |
|---|---:|---:|
| negative_current_speed | 0.9615 | 0.8560 |
| negative_past_displacement | 0.8929 | 0.4979 |

The moving low-progress positives are listed below. Two already had contact, so only two episodes combine clear current motion, no prior contact and an impending low-progress target. Low progress is a net-displacement label, not a sustained-stall label: the spline example rolls back after forward motion (3.62 m traveled, 0.308 m net, final vx −1.08 m/s). The other previously contact-free example nearly stops (0.565 m net over four seconds, 0.012 m net in the last two seconds). Individual predictions and raw motion diagnostics are preserved in strata.json. This is insufficient support for a general terrain-stall or recovery claim.

| Episode / anchor | Prior contact | Actual 4 s displacement, m | Nominal displacement, m | History s11 / s29 displacement, m |
|---|---|---:|---:|---:|
| full_v3__ep_0626_near_obstacle / 140 | True | 0.188 | 15.303 | 10.270 / 7.959 |
| full_v3__ep_4705_spline / 280 | False | 0.308 | 12.526 | 6.540 / 6.415 |
| full_v3__ep_3646_near_obstacle / 180 | True | 0.075 | 12.190 | 9.632 / 4.249 |
| full_v3__ep_1166_near_obstacle / 160 | False | 0.565 | 12.363 | 12.165 / 8.471 |

| Episode / anchor | Actual path traveled, m | Actual last 2 s net motion, m | Actual endpoint vx, m/s |
|---|---:|---:|---:|
| full_v3__ep_0626_near_obstacle / 140 | 0.592 | 0.009 | 0.025 |
| full_v3__ep_4705_spline / 280 | 3.619 | 1.947 | -1.084 |
| full_v3__ep_3646_near_obstacle / 180 | 0.739 | 0.048 | -0.023 |
| full_v3__ep_1166_near_obstacle / 160 | 1.359 | 0.012 | 0.122 |

Fixed-budget last checkpoints are separate from these selected checkpoints:

| Last checkpoint | Step | ADE / FDE, m | Contact / low-progress AUC at 4 s |
|---|---:|---:|---:|
| history_s11 | 1000 | 0.7434 / 0.9192 | 0.9823 / 0.9998 |
| history_s29 | 1000 | 0.7440 / 0.9443 | 0.9843 / 0.9996 |
| no_history_s11 | 1000 | 0.7435 / 0.9516 | 0.9873 / 0.9998 |
| no_history_s29 | 1000 | 0.7427 / 0.9590 | 0.9893 / 0.9996 |
| no_terrain_s11 | 1000 | 0.7392 / 0.9329 | 0.9809 / 0.9997 |
| no_terrain_s29 | 1000 | 0.7569 / 0.9745 | 0.9790 / 0.9997 |
| profile_s11 | 1000 | 0.7027 / 0.8767 | 0.9804 / 0.9993 |
| profile_s29 | 1000 | 0.7108 / 0.9089 | 0.9808 / 0.9991 |

No new model or optimizer updates were performed for this analysis. No protected test episodes were read. Overlapping windows are not independent trials; episode counts matter. The raw stores cover one BMP arena and one PID driver domain. This is neither unseen-terrain validation nor closed-loop MPPI performance. Neither training nor validation has rollover-positive episodes.
