# Current model dataset

The current RGB-D model was trained on **360 Chrono episodes across 24 arenas**, containing **14.94 hours of recorded simulated traversal**. This is the seed-11 checkpoint after 5,000 updates, predicting 12 seconds ahead, from `packs/full_pair_v1/h60`.

| Split | Episodes | Arenas | Actual recorded seconds | Simulated hours |
|---|---:|---:|---:|---:|
| Training — optimizer data | 360 | 24 | 53,767.20 | 14.935333 |
| Validation — held out | 90 | 6 | 13,610.05 | 3.780569 |
| Test — excluded from training pack | 90 | 6 | 12,885.80 | 3.579389 |
| Training + validation | 450 | 30 | 67,377.25 | 18.715903 |
| All reference data | 540 | 36 | 80,263.05 | 22.295292 |

Each arena is 240 m square and has 15 references: five path offsets at 2, 4 and 6 m/s. Six terrain families cover rolling hills, ridge passes, cross-slopes, valley networks, rough mosaic and mixed obstacles. Each family has four training arenas, one validation arena and one test arena. One measured global RGB-D snapshot is shared by each arena's routes.

Durations sum the actual `outcome.json` values, including early goal arrivals and rollovers. Every value agrees with its recorded interval count at 20 Hz. The totals exclude the unrecorded 0.8-second settling stage, observation-only passes, retries and later planner diagnostics. They are simulation time, not AMD wall-clock time. The 53,830 training and 13,625 validation windows overlap and are not additional episodes or additional simulated hours.

The checkpoint's data-manifest hash, both episode-list hashes, all episode memberships/frame counts, and all 540 raw outcome/case/reference hashes were verified directly. [Exact totals, source paths and checksums](dataset_inventory_20260909.json).

AMD source paths, beneath `/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909/`:

- Checkpoint: `runs/full_learning_h60_v3/patch_rgbd_s11/last.pt`.
- Training pack: `packs/full_pair_v1/h60/manifest.json`, `train_episodes.json`, `val_episodes.json`.
- Training/validation measurements: `full_cohort_v2/raw/<scene>/<family>/outcome.json`.
- Held-out test measurements: `protected_test_cohort_v2/raw/<scene>/<family>/outcome.json`.

Verified on 2026-09-10 at 04:20 UTC. No training or physics was run for this inventory.
