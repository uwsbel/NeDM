# Independent review of the diverse RGB-D FDM

The expanded model preserves the agreed finite-horizon FDM design: a measured global RGB-D snapshot and causal vehicle history condition a GRU over candidate path/speed commands; a joint decoder predicts motion, cumulative positive work, events and attitude. It does not predict future images or feed predicted observations back through a recurrent world model.

The review covered `fdm_diverse_model.py`, `fdm_diverse_data.py` and `traverse_fdm_rgbd_diverse_train.py`. Two real issues were found and corrected before the training snapshot:

1. **Image shuffling must use image identity.** One static scene image is reused by many time windows. Shuffling window rows can leave the image unchanged, and a validation set with one scene cannot support a shuffled-image control. Evaluation now deranges unique image identities, counts unchanged image assignments, and reports the one-image control unavailable.
2. **Roll must cover rollover orientations and wrap correctly.** A shared ±90° attitude output could not represent measured roll beyond 90°. Roll now spans ±π, pitch spans ±π/2, and loss plus evaluation use wrapped angular differences. The event head remains a separate rollover probability.

## Verified model contracts

[CPU inference-only report](../../../artifacts/traverse/fdm_diverse_v1_20260909/geometry/model_contract_cpu_02.json) records the exact source hashes and checks. No optimizer was constructed and no training updates were performed.

- A synthetic plane at 10 m elevation, observed with 400 m camera height and 47° FOV, encodes correctly with the 40 m elevation scale. The 1024² metric ray-depth image is reduced to the 512² input without the legacy depth-window clipping.
- Candidate crop projection has maximum normalized-channel error **2.98 × 10⁻⁷** on an affine world-coordinate image. Crop top points forward and crop left points vehicle-left. All three nontrivial quarter-turn augmentations preserve the same physical crop to **2.39 × 10⁻⁷** or better.
- The RGB-only arm excludes depth from crop positioning as well as channel values. Blank intervention removes global/local image evidence. Global context is downsampled to 128²; candidate crops use the 512² input.
- The 60-step × 0.2 s command sequence covers **12 s**. A straight 4 m/s command reaches a nominal 48 m. Nominal body-twist integration is consistent with these positions and headings.
- Replacing future target arrays leaves all inference outputs exactly unchanged. Allowed inputs remain RGB-D, history, candidate commands, measured anchor pose, goal-relative context and elapsed time.
- Cumulative predicted positive work is nonnegative and monotone. Attitude masks exclude masked target values. A **135° roll** is representable, and orientations immediately on opposite sides of ±π have the expected small loss.
- Saving and loading the candidate-patch checkpoint preserves outputs exactly. The shared-image shuffle changes every tested window's image identity; a one-image shuffle is unavailable.

## Training and interpretation boundaries

Normalization is computed from training low-dimensional tensors; physical pixel normalization is fixed. Event definitions and semantic versions are explicit. The trainer reads training and validation arrays, never the sealed test set. Fixed final and validation-selected checkpoints are saved separately, and the image interventions do not choose the checkpoint. Resume requires identical source/data hashes and relevant training settings.

Scene-disjoint data assignment is enforced by the packer rather than inferred by the neural network. These source and synthetic checks do not validate Chrono calibration, dataset labels, learned risk calibration, long-route success or the eventual MPPI energy/time tradeoff. The separate AMD geometry checker measures the actual terrain and camera residuals without rewriting frozen orientation metadata.

The observation is a fixed aerial map-like sensor. Causal vehicle state and command context change with each planning anchor, but the image still shows the launch-time vehicle. This is appropriate for the current static rigid-terrain experiment; dynamic terrain, moving obstacles and onboard limited visibility require additional treatment.
