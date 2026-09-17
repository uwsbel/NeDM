# RGB-D finite-horizon validation comparison

Fixed-budget last checkpoints are compared first. Separate selected-checkpoint image interventions and prospective hazard strata follow; all remain development validation on one arena.

| Arm | Seed | Last step | ADE / FDE (m) | Contact AUC at 4s | Low-progress AUC at 4s | Best step |
|---|---:|---:|---:|---:|---:|---:|
| blank | 11 | 3000 | 0.3894 / 0.6059 | 0.9806 | 0.9997 | 1500 |
| blank | 29 | 3000 | 0.4000 / 0.6390 | 0.9840 | 0.9998 | 3000 |
| depth_only | 11 | 3000 | 0.4456 / 0.7941 | 0.9739 | 0.9997 | 1500 |
| depth_only | 29 | 3000 | 0.4236 / 0.6527 | 0.9643 | 0.9997 | 2000 |
| rgb_only | 11 | 3000 | 0.4209 / 0.6755 | 0.9799 | 0.9998 | 1500 |
| rgb_only | 29 | 3000 | 0.4229 / 0.6468 | 0.9640 | 0.9997 | 2000 |
| rgbd | 11 | 3000 | 0.4721 / 0.8657 | 0.9773 | 0.9998 | 1500 |
| rgbd | 29 | 3000 | 0.4455 / 0.7503 | 0.9229 | 0.9953 | 1500 |

| Causal anchor stratum | Contact positives / episodes | Low-progress positives / episodes |
|---|---:|---:|
| all | 136 / 11 | 61 / 9 |
| moving | 42 / 10 | 4 / 4 |
| already_low_progress | 80 / 8 | 48 / 8 |
| before_first_contact | 44 / 11 | 2 / 2 |
| moving_before_first_contact | 40 / 10 | 2 / 2 |

| RGB-D comparison (same seed, best checkpoints) | Seed | Moving FDE delta (m) | Prospective-contact AUC delta | Moving low-progress AUC delta |
|---|---:|---:|---:|---:|
| rgbd minus blank | 11 | -0.0485 | -0.0045 | 0.0003 |
| rgbd minus rgb_only | 11 | -0.0026 | -0.0027 | 0.0005 |
| rgbd minus depth_only | 11 | -0.0432 | -0.0045 | 0.0006 |
| rgbd minus blank | 29 | 0.3378 | -0.0008 | -0.0242 |
| rgbd minus rgb_only | 29 | 0.2908 | 0.0043 | -0.0223 |
| rgbd minus depth_only | 29 | 0.2692 | -0.0067 | -0.0213 |

For the preceding modality table, negative error deltas and positive AUC deltas favor RGB-D. Best steps may differ; fixed-last results above retain equal update budgets.

| RGB-D image intervention minus normal (same checkpoint) | Seed | Moving FDE delta (m) | Prospective-contact AUC delta | Moving low-progress AUC delta |
|---|---:|---:|---:|---:|
| shuffle minus normal | 11 | -0.0000 | -0.0000 | 0.0000 |
| blank minus normal | 11 | -0.0001 | -0.0000 | -0.0001 |
| shuffle minus normal | 29 | -0.0013 | -0.0132 | -0.0011 |
| blank minus normal | 29 | -0.0146 | 0.0110 | 0.0138 |

Intervention error increases or AUC decreases suggest useful image information in the selected checkpoint. They do not establish unseen-terrain or closed-loop route-choice success.

The JSON includes per-episode support, calibration, fixed-retention false accepts, paired episode-weighted motion errors, and every prospective positive case. Moving low-progress examples are scarce and can include rollback. Rollover is unsupported.

No model training, checkpoint selection, protected test reading, or raw state-file rereading occurred during this report.
