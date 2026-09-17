The frozen RGB-D pack contains no contact or low-progress positives at anchor0 in train or validation. At anchors below2s, only one train window is positive for either event and none in validation.

| Split | Anchor group | Windows | Contact positive windows / episodes | Low-progress positive windows / episodes |
|---|---|---:|---:|---:|
| train | anchor0 | 1000 | 0 / 0 | 0 / 0 |
| train | anchor_lt2s | 2000 | 1 / 1 | 1 / 1 |
| train | anchor_ge2s | 18000 | 702 / 49 | 496 / 61 |
| train | all | 20000 | 703 / 49 | 497 / 61 |
| val | anchor0 | 250 | 0 / 0 | 0 / 0 |
| val | anchor_lt2s | 500 | 0 / 0 | 0 / 0 |
| val | anchor_ge2s | 4500 | 136 / 11 | 82 / 10 |
| val | all | 5000 | 136 / 11 | 82 / 10 |

Of703/136 contact-positive train/val windows,193/43 start before first contact;510/93 start at or after it. Of497/82 low-progress-positive windows,114/20 start before the first observable completed2s low-progress interval;383/62 start at or after it. These categories overlap across windows within episodes.

There are no supervised from-rest failure positives. The model can exploit time/history/current-contact shortcuts; weak new-scene visual ranking cannot be attributed to encoder architecture alone. Need paired safe/failing references with near-term failure visible before contact, grouped by new scene.

Definitions and source hashes are in early_anchor_support.json. Low progress is a future prefix label (net movement under0.15m/s with effort, duration at least2s, parking excluded), not a proof of sustained immobility. The observed-onset comparison uses completed2s intervals from actual20Hz telemetry. No protected-test examples were inspected.
