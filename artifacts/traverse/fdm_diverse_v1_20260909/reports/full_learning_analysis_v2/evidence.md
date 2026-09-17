# Full-cohort forecast evidence

RGB-D provides useful advance hazard information on six unseen validation arenas. Closed-loop route completion and energy efficiency still require the separate Chrono MPPI evaluation.

All eight models used 5,000 updates, 53,830 train windows and 13,625 validation windows. Seeds and training-window/rotation draws match. The 4,447 pre-onset windows have no previously completed contact, rollover, bounded-motion, or sustained-stall event; overlapping windows are not independent trials.

## Matched final checkpoints: 12-second forecasts

| Seed | Model | Pre-onset contact AUROC | Pre-onset bounded-motion AUROC | Pre-onset FDE m | All-window FDE m |
|---:|---|---:|---:|---:|---:|
| 11 | rgbd | 0.914 | 0.915 | 10.14 | 6.00 |
| 11 | blank | 0.776 | 0.729 | 11.89 | 6.62 |
| 29 | rgbd | 0.924 | 0.922 | 9.12 | 6.28 |
| 29 | blank | 0.756 | 0.725 | 11.93 | 6.37 |

RGB-D improves pre-onset contact and bounded-motion AUROC on all six arenas for both seeds. Shuffling scene images increases overall 12-second FDE by 1.76 / 2.35 m and mechanical-work MAE by 18.0 / 39.9 kJ; the learned model uses information specific to the scene.

## Predeclared planning checkpoint and remaining limits

The primary planning model remains H60 RGB-D seed 11, validation-best at update 4,000. Its pre-onset contact / bounded-motion AUROC is 0.871 / 0.897. With shuffled images these fall to 0.611 / 0.612.

At the existing contact / bounded-motion probability caps (0.35 / 0.50), this primary checkpoint recalls 63.4% / 51.1% of positive pre-onset windows while retaining 92.0% / 95.2% of negative windows. Good ranking is not reliable default-threshold safety. Rollover recall is only 17.0% at the 0.35 cap, with four positive validation routes in two arenas.

The trained stall proxy is entirely future two-second bounded motion under throttle; separately recorded sustained low-speed stall is a different target. Contact means asset or chassis resultant above 1 N. Signed roll/pitch endpoints do not guarantee detection of solver-step attitude peaks.

Primary 12-second mechanical-work MAE is 102.2 kJ overall (17.6% weighted absolute error) and 111.5 kJ before first failure (34.5%). This is positive engine-interface mechanical work, not fuel consumption. Aggregate work error includes many already-failed/stalled windows; energy-efficient route selection must be measured among safely completed Chrono routes.

H20 improves short-horizon average FDE but does not beat its blank controls on that metric. H60 has 3,052,556 parameters versus H20's 1,649,316, so horizon comparisons also change capacity and future supervision. Validation-best and fixed-final checkpoints are reported separately; the primary seed is unchanged.

## Coverage boundary

The prescribed safe routes all use the outer ±44 m offsets. A fixed −44 m route succeeds on 4/6 validation arenas; fixed +44 m succeeds on 3/6. Both should remain visible baselines. The dataset provides diverse failure/vehicle signals, but does not yet demonstrate arbitrary safe interior-route topology.

Next gate: measure matched candidate selection and online Chrono completion, contact/stall/rollover, time and work. Report failures and abstentions before comparing energy or time among paired safe goal completions.
