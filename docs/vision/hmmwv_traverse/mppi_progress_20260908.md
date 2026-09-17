# First implementation and AMD pilot

This records the earlier geometry-only baseline. [Current RGB-D implementation,
trained models and measured Chrono results](mppi_rgbd_progress_20260908.md)
supersede its implementation-status statements.

**The effort now has a working data pipeline, trained trajectory predictors and an offline MPPI scoring integration.** The first eight-model AMD pilot is complete. Physical closed-loop MPPI execution has not yet been evaluated.

**RGB-D requirement clarified after the pilot.** The user requires RGB-D observations for the intended model. None of the eight pilot checkpoints consumes RGB or depth images: they are privileged geometry baselines, not the requested perceptive model. The four architectures are custom HMMWV adaptations of the reference's finite-horizon prediction and external MPPI-cost idea, not reproductions of its neural architecture or pretrained weights. In particular, our history GRU encodes measured history; our predictor has no command-sequence forward GRU. The reference includes a depth-image ResNet configuration, which is not a ready-made RGB-D model. The next perceptive implementation must consume actual RGB-D (or explicitly RGB-D-derived spatial features), measured vehicle history and candidate path/speed, predicting motion and separately evaluated collision, low-progress and rollover risks. It must preserve observation provenance and must not retain BMP elevations or authored obstacle clearances as hidden inputs in the RGB-D arm. Existing geometry-only results remain a baseline. A reference-style forward-GRU arm and the simpler direct decoder should be distinguished in comparisons.

All implementation is in `/home/harry/NeDM-traverse_mppi`, branch `traverse_mppi`. The main `/home/harry/NeDM` checkout was read for data and existing evidence only. Its source, datasets, checkpoints and job directories were not changed. Remote work is isolated at `/work1/dannegrut/harry/experiments/traverse_mppi_20260908`.

**What runs.** A shared feature builder reads the measured 0.8 s vehicle/action history, the known reference path and speed profile, and privileged BMP/asset geometry. A direct model predicts twenty ego XY/heading poses over four seconds, cumulative positive shaft work, and separately supervised event risks. No future recorded pose, station or reactive control enters the inputs. Goal cost remains outside the network.

The MPPI reference optimizer produces smooth local path/speed perturbations, checks curvature, acceleration, bounds and footprint clearance, and scores feasible candidates in batches. It rechecks its weighted mean and retains a checked sample if needed. Zero perturbation preserves the original reference waypoint sampling, because resampling would change the standard driver's Bezier control points. A causal station marker prevents nearest-point jumps during rollback. A quarter of proposals change only speed, preserving alternatives when a path is already at its curvature limit.

The optimizer checks nominal reference geometry. It does not yet establish the safety of the current vehicle footprint or the motion connecting an off-reference vehicle to that path. This is particularly relevant after rollback. The current preview is therefore an offline integration test, not a physical navigation result.

**Reused data.** The identity-selected first pack contains 1,000 training and 250 validation episodes from `full_v1`–`full_v3`, all using the standard substep `ChPathFollowerDriver`. It provides 20,000 training and 5,000 validation windows. The fixed legacy split is preserved and protected test payloads are not opened. The pack is `artifacts/traverse/fdm_fast_data_v2`; earlier smoke/v1 packs are superseded.

| Positive episode support | Training | Validation |
|---|---:|---:|
| Contact within an eligible future prefix | 49 | 11 |
| Low progress under effort | 61 | 10 |
| Rollover | 0 | 0 |

Overlapping windows are not independent trials. Rollover loss and operational risk use are disabled because the first pack has no positive examples. Low progress is supervised only from two seconds onward, with deliberate route-end parking excluded. Observed failure positives survive censoring; unavailable future motion and unknown event negatives are masked. Positive work is measured in kJ and uses positive shaft power, not signed energy. `complete` in the old raw stores is treated as recording survival, not goal completion.

**Executed jobs.**

| AMD job | Work | Result | Slurm elapsed |
|---|---|---|---|
| `409459` | Four arms × seeds 11/29, 100 updates each | All eight completed, exit `0:0` | 19 s |
| `409462` | Same eight variants, 1,000 updates each from scratch | All eight completed, exit `0:0` | 27 s |

Each suite used one MI350 eight-GPU node and one model per GPU. Both training arrays were checksum-verified after transfer. The immutable training code snapshot is `code/3fb9df94454b` on AMD; source/data hashes, configuration, normalization, optimizer state and RNG state are stored with checkpoints. All arms with the same seed have identical final sample-draw hashes. All optimizer updates ran on AMD; local work was preprocessing, CPU contract checks, inference and reporting.

**First fixed-budget results.** Validation uses the same one-arena standard-driver distribution, with actual four-second endpoints as truth.

| Predictor | Mean 4 s endpoint error, seeds 11 / 29 |
|---|---:|
| Nominal path/speed integration | 1.305 m |
| Small direct profile model, current state retained | **0.877 / 0.909 m** |
| Direct predictor with measured history | 0.919 / 0.944 m |
| Same larger predictor, current state only | 0.952 / 0.959 m |
| History model without explicit terrain columns | 0.933 / 0.974 m |

The small profile model has 35,680 parameters; the other three arms have 211,680 each. The first budget does not establish a benefit from the larger history architecture. Removing explicit terrain columns still leaves terrain information in measured history and slope-limited reference speeds; this is not a claim that terrain is unnecessary.

Pooled contact AUC at four seconds is approximately 0.98 across the trained variants, and pooled low-progress AUC is nearly 1.0. These headline values are insufficient evidence of anticipating new failures. Among 61 positive four-second low-progress windows from nine validation episodes, 55 already have absolute body vx below 0.5 m/s, and 48 satisfy a causal recent low-displacement/high-throttle criterion. Only four positive windows from four episodes are clearly moving at the anchor, and only two of those episodes have no prior recorded asset contact. Prospective contact support is somewhat broader: 44 pre-first-contact positive windows from 11 episodes, including 40 moving windows from 10 episodes. The audit separates these cases explicitly rather than presenting recognition of an existing stall as prediction of stall entry.

The individual prospective failures expose a real model limitation. In the two moving, previously contact-free cases, actual four-second net displacement is 0.308 m and 0.565 m. The validation-selected history models instead predict 6.415–6.540 m and 8.471–12.165 m, respectively. Their low-progress probabilities are only 0.199–0.229 and 0.018–0.210. The first case is rollback cancellation: 3.62 m traveled but only 0.308 m net displacement, ending with vx −1.08 m/s. The second is a sustained near-stop, with only 0.012 m net displacement in the last two seconds. Thus the direct prediction architecture still overpredicts progress after impending rollback or stopping, despite excellent pooled classification ranking. These two cases cannot establish general failure performance; clean prospective sustained-stop support is only one episode.

Full last/best metrics, calibration counts and per-arm readouts are in [comparison.md](../../../artifacts/traverse/fdm_runs/pilot_v1/comparison.md) and [comparison.json](../../../artifacts/traverse/fdm_runs/pilot_v1/comparison.json). Fixed-budget checkpoints are the table's source; validation-selected checkpoints are reported separately. The independent [causal-anchor audit](../../../artifacts/traverse/fdm_runs/pilot_v1/strata.md) reports the individual failure cases and reproduces all eight archived four-second AUC values.

**Trained MPPI preview.** The predeclared first oracle-family episode in the validation identity list (`full_v3/ep_2824_oracle`, measured frame 40) was scored with `history_s11/last.pt`. The optimizer evaluated 253 geometrically valid model queries and returned a checked speed-profile change. Its predicted cost changed from -19.575 to -20.621. This is model cost only; no improved physical outcome is claimed. The local CPU preview took about 0.53 s, including feature construction and checks; it is not a validated online control-rate benchmark. See [preview](../../../artifacts/traverse/fdm_runs/pilot_v1/mppi_preview.json), which includes checkpoint and current planner-code hashes.

**Verification.** The contract checks cover split aliases, future-input exclusion, exact contact/rollover timing, positive work, censoring, parking, rollback, retained ablation inputs, supported event masks, inference batch parity, gradient finiteness, validation-loss aggregation, station handling, risk aggregation and MPPI geometry/fallback behavior. Python compilation, Slurm shell parsing and whitespace checks also pass. Evidence: [contract checks](../../../artifacts/traverse/fdm_contract_checks_final.json), [execution record](../../../artifacts/traverse/fdm_execution_20260908.json).

**Next decision gate.** Keep the small direct model as the inexpensive baseline and add the required RGB-D observation arm, checking image alignment, camera coverage and retained spatial information. Use the existing multi-arena failure recordings in a separately identified, matched PID domain to evaluate pre-stall forecasts and actual low-progress displacement. Expand moving-to-failure/recovery coverage only where the recorded data lack it. Then run paired candidate executions in Chrono under the exact chosen driver and compare CEM versus MPPI with one fixed scorer. Existing failure-rich 20 Hz `NativePID` recordings must not be silently relabeled as standard substep-driver data. Fresh-terrain confirmation and live replanning remain later gates. All new training stays on AMD in the isolated experiment.

Reproduction entry points:

```bash
# Data preparation and checks are CPU-only; output must be outside the source checkout.
/home/harry/miniconda3/envs/nedm/bin/python scripts/traverse_fdm_prepare.py \
  --source-root /home/harry/NeDM --out /new/isolated/data/path
/home/harry/miniconda3/envs/nedm/bin/python scripts/check_traverse_fdm_contract.py \
  --pack artifacts/traverse/fdm_fast_data_v2
/home/harry/miniconda3/envs/nedm/bin/python scripts/check_traverse_fdm_mppi.py

# Report completed AMD runs locally.
/home/harry/miniconda3/envs/nedm/bin/python scripts/traverse_fdm_report.py \
  --runs artifacts/traverse/fdm_runs/pilot_v1 \
  --out artifacts/traverse/fdm_runs/pilot_v1/comparison.json
```

Training uses `slurm/traverse_fdm_suite.sbatch`, with `CODE_ROOT` pointing to an immutable source snapshot, `DATA_ROOT` to the frozen pack, and a new `RUN_ROOT` for each suite. `STEPS`, `BATCH` and `EVAL_EVERY` set the fixed budget. Do not sync this experimental code over the main AMD mirror or another experiment's source tree.
