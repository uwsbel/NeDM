# NRD Study 1 Plan: Chrono Double Pendulum with Camera Dynamics

**Purpose:** First implementation and falsification study for Neural Reduced Dynamics (NRD)  
**Simulator:** Project Chrono with Chrono::Sensor  
**State-only reference:** `~/NeDM`

## 1. Corrected study objective

Build an actuated planar double pendulum in Project Chrono and learn the joint evolution of:

- a complete physical dynamics state, \(z_{1,t}\); and
- a camera latent, \(z_{2,t}=E_\phi(x_t)\), encoded from the synchronized Chrono camera frame.

The NRD state is the concatenation

\[
z_t=\operatorname{concat}(z_{1,t},z_{2,t}),
\]

and the action-conditioned temporal model predicts both parts:

\[
(\hat z_{1,t+1},\hat z_{2,t+1},h_{t+1})
=F_\theta([z_{1,t},z_{2,t}],a_t,h_t),
\qquad
\hat x_{t+1}=D_\psi(\hat z_{2,t+1}).
\]

The sensor latent is therefore **appended to—not substituted for—\(z_1\)**.

For this first case, \(z_1\) intentionally contains enough information to determine the ideal double-pendulum mechanics. With fixed geometry, material properties, gravity, damping, camera, and lighting, the camera image is also largely determined by \(z_1\). Therefore:

> Study 1 validates synchronized physical-state and camera-frame prediction. It does not claim that vision improves double-pendulum control.

The control value of information available only through \(z_2\) should be tested later in manipulation and terrain-navigation scenes.

## 2. Research questions

1. Can the state-only temporal model in `~/NeDM` be extended cleanly from \([z_1,a]\) to \([z_1,z_2,a]\)?
2. Can one model accurately predict physical evolution and camera-latent evolution over multi-step rollouts?
3. Does adding the visual branch preserve \(z_1\) accuracy relative to the matched state-only model?
4. Do decoded future frames remain temporally aligned with the predicted physical state?
5. Can a swing-up policy trained in the frozen NRD transfer to the original Chrono system?
6. What throughput is gained relative to Chrono with camera rendering enabled?

### Expected result

- \(z_1\)-only and \([z_1,z_2]\) policies should perform similarly because \(z_1\) is fully observed.
- Masking the input \(z_2\) may have little effect on \(z_1\) prediction; this is not a failure in Study 1.
- The important visual result is that \(\hat z_2\) and \(D(\hat z_2)\) track the future Chrono camera sequence during autonomous rollout.

## 3. Chrono system specification

### 3.1 Implementation references

Use:

- the official [Project Chrono repository](https://github.com/projectchrono/chrono);
- Chrono's [Python revolute-joint/CAD pendulum example](https://github.com/projectchrono/chrono/blob/main/src/demos/python/cascade/demo_CAS_stepfile.py) as a joint-construction reference;
- the official [Chrono::Sensor tutorial index](https://api.projectchrono.org/tutorial_table_of_content_chrono_sensor.html); and
- the [Chrono Sensor demos](https://github.com/projectchrono/chrono/tree/main/src/demos/sensor) for camera creation, filtering, and frame retrieval.

Implement the study in the language already used by the NeDM data pipeline. Prefer PyChrono for fast iteration; use the C++ Sensor interface if Python frame extraction or throughput becomes a limitation.

### 3.2 Mechanical model

Construct:

- one fixed ground body;
- two slender rigid links;
- a revolute joint from ground to link 1;
- a revolute joint from link 1 to link 2; and
- one torque motor at the second joint using Chrono's rotational torque-motor interface.

Keep motion planar through the revolute-joint axes. Disable contact in the main study. Fix and version:

- link lengths, masses, center-of-mass locations, and inertias;
- joint damping;
- gravitational acceleration;
- solver, integrator, tolerances, and iteration limits; and
- motor torque limits.

Recommended initial timing:

- Chrono integration step: \(\Delta t_{\mathrm{sim}}=0.001\) s;
- action/control interval: \(\Delta t=0.02\) s;
- camera update rate: 50 Hz, aligned with the control interval; and
- action held constant for 20 Chrono integration steps.

These are starting values, not hidden defaults. Confirm convergence by comparing selected trajectories against \(\Delta t_{\mathrm{sim}}=0.0005\) s before generating the full dataset.

### 3.3 Action

Use one normalized continuous action

\[
a_t\in[-1,1], \qquad \tau_t=a_t\tau_{\max},
\]

applied as elbow-joint torque and held over \([t,t+\Delta t)\).

Do not add motor lag in Study 1. If actuator dynamics are added later, their internal states must also be included in \(z_1\); otherwise \(z_1\) is no longer Markov.

## 4. Physical state \(z_1\)

### 4.1 Recommended state

Define:

- \(q_1\): link-1 angle relative to the fixed downward direction;
- \(q_2\): link-2 angle relative to link 1; and
- \(\omega_1,\omega_2\): the corresponding generalized angular velocities.

Use

\[
z_1=
[\cos q_1,\sin q_1,
  \cos q_2,\sin q_2,
  \omega_1,\omega_2]\in\mathbb R^6.
\]

This state is sufficient for the fixed, ideal planar constrained system under the current torque action. The sine/cosine representation avoids discontinuities at \(\pm\pi\).

Extract angles from relative body orientations and angular velocities by projecting onto the known hinge axis. Store the underlying Chrono body positions, quaternions, linear velocities, and angular velocities as audit fields, but do not use them as NRD inputs in the main experiment.

### 4.2 Sufficiency checks

Before model training:

1. reconstruct both link poses from \(z_1\) and fixed geometry;
2. verify reconstructed joint and endpoint positions against Chrono;
3. verify that angle unwrapping and velocity signs remain continuous;
4. replay identical \((z_{1,t},a_t)\) conditions and confirm identical \(z_{1,t+1}\) within solver tolerance; and
5. confirm that no unrecorded actuator, contact, or out-of-plane state affects the next step.

If any check fails, expand \(z_1\) before collecting the production dataset.

## 5. Camera state \(z_2\)

### 5.1 Camera configuration

Attach a fixed world camera that sees the full reachable workspace:

- RGB resolution: \(128\times128\);
- update rate: 50 Hz;
- fixed pose, field of view, exposure, lighting, and background;
- anti-aliasing and postprocessing settings versioned in the dataset manifest; and
- no camera randomization in the main in-distribution study.

Let the synchronized frame be

\[
x_t\in[0,1]^{3\times128\times128},
\qquad z_{2,t}=E_\phi(x_t)\in\mathbb R^{64}.
\]

### 5.2 Role of \(z_2\)

In Study 1, \(z_2\) represents the rendered camera state: link appearance, pose in the image, lighting, and background. It is included in every transition-model token:

\[
u_t=[z_{1,t},z_{2,t},a_t].
\]

The visual prediction head produces \(\hat z_{2,t+1}\), and the decoder reconstructs \(\hat x_{t+1}\). During autonomous rollout, only the initial frame is encoded; subsequent camera latents are predicted recursively.

## 6. Dataset plan

### 6.1 Dataset scale

At 50 recorded transitions per second:

| Dataset | Trajectories | Duration each | Approx. transitions | Purpose |
|---|---:|---:|---:|---|
| Smoke test | 10 | 2 s | 1,000 | Alignment tests and one-batch overfit |
| Pilot | 100 | 10 s | 50,000 | Architecture and loss selection |
| Full | 1,000 | 10 s | 500,000 | Main training and evaluation |

Raw full-resolution RGB for 500,000 frames is roughly 25 GB before compression. Measure the actual compressed size during the pilot before committing to the full run.

### 6.2 Initial conditions and actions

Split complete trajectories among:

- 30% unforced response with randomized angles and velocities;
- 50% piecewise-constant random torque, with dwell times from 0.1–0.5 s; and
- 20% smooth sinusoidal, chirp, or controller-generated torque.

Sample broad but safe initial conditions:

- \(q_1,q_2\) across \([-\pi,\pi]\);
- \(\omega_1,\omega_2\) within a pilot-validated bounded range; and
- reject only states that violate the intended planar mechanism or numerical limits.

Keep physical parameters fixed in the main dataset. Parameter-randomized dynamics require either parameter context in the model or enough recurrent history to infer hidden parameters; that is a later experiment.

### 6.3 Timestamp and transition contract

For every recorded transition:

1. synchronize Chrono dynamics and the Sensor manager at time \(t\);
2. record \(z_{1,t}\) and camera frame \(x_t\);
3. apply \(a_t\) over \([t,t+\Delta t)\);
4. advance Chrono using the smaller integration steps;
5. update the camera at \(t+\Delta t\); and
6. record \(z_{1,t+1}\) and \(x_{t+1}\).

Write an automated timestamp test to detect a one-frame Sensor pipeline delay. Do not infer alignment by visual inspection alone.

### 6.4 Recorded fields

```text
trajectory_id, seed, step, t, dt_sim, dt_record
z1_t, action_t, z1_t_plus_1
rgb_t, rgb_t_plus_1
q_t, qdot_t, q_t_plus_1, qdot_t_plus_1
body_pose_t, body_velocity_t                 # audit only
applied_motor_torque
solver_status, solver_iterations
mechanism_parameters, integrator_parameters
camera_parameters
```

Store states as `float32`, images as `uint8`, and configuration values at sufficient precision to reproduce trajectories. Use Zarr or HDF5 with trajectory or short-sequence chunks. Include the Chrono version, source commit, configuration hash, units, coordinate conventions, and normalization statistics in the manifest.

### 6.5 Splits

Split complete trajectories:

- 70% training;
- 15% validation; and
- 15% test.

Never split adjacent frames across partitions. Create later OOD sets for camera pose, lighting, link texture, small dynamics-parameter changes, and action frequencies, but do not mix these with the first fixed-condition result.

## 7. Neural architecture

### 7.1 Camera encoder

Use a deterministic convolutional encoder trained from scratch:

```text
RGB 3x128x128
Conv 3->32,   kernel 4, stride 2, padding 1   # 64x64
Conv 32->64,  kernel 4, stride 2, padding 1   # 32x32
Conv 64->128, kernel 4, stride 2, padding 1   # 16x16
Conv 128->256,kernel 4, stride 2, padding 1   # 8x8
Flatten -> Linear(256*8*8, 256) -> Linear(256, 64)
LayerNorm -> z2 in R^64
```

Use GroupNorm and SiLU after each convolution. Begin with a deterministic autoencoder rather than a VAE. The official [PyTorch VAE example](https://github.com/pytorch/examples/tree/main/vae) is useful structural scaffolding, but remove the stochastic sampling and KL loss for the main model.

### 7.2 Temporal dynamics backbone

The preferred implementation is to reuse the temporal backbone and sequence conventions in `~/NeDM`:

```text
z1_t [6] + z2_t [64] + action_t [1]
        -> concatenate [71]
        -> input projection to NeDM embedding dimension
        -> NeDM temporal backbone
        -> shared predicted temporal feature
             |-> physical head: predicted delta z1 [6]
             |-> visual head: predicted next z2 [64]
```

The core architecture change is at the input and output boundaries; do not build two disconnected dynamics models.

- Compare against an otherwise matched \([z_1,a]\) state-only model.
- Start with a temporal context of 16–32 recorded steps unless the audited NeDM implementation requires another contract.
- Predict a residual for normalized \(z_1\), then renormalize each sine/cosine pair.
- Predict \(z_{2,t+1}\) against a stop-gradient encoding of the true next frame.
- Use a small GRU only as an early smoke-test fallback; the primary result should extend `~/NeDM`.

### 7.3 Camera decoder

Use the mirrored decoder:

```text
z2 [64] -> Linear(64, 256*8*8) -> reshape 256x8x8
ConvTranspose 256->128 -> 64 -> 32 -> 3
four blocks, kernel 4, stride 2, padding 1
Sigmoid -> RGB 3x128x128
```

The decoder is used during training and visual evaluation. It is disabled during high-throughput RL unless reconstructed frames are explicitly required.

## 8. Training objective

For an open-loop horizon \(H\):

\[
\mathcal L_{\mathrm{NRD}}
=\sum_{k=1}^{H}\gamma^{k-1}
\left[
\lambda_1\mathcal L_{z_1}^{(k)}
+\lambda_2\mathcal L_{z_2}^{(k)}
+\lambda_3\mathcal L_{\mathrm{frame}}^{(k)}
+\lambda_4\mathcal L_{\mathrm{circle}}^{(k)}
\right].
\]

Use:

- \(\mathcal L_{z_1}\): Huber loss on normalized state channels;
- \(\mathcal L_{z_2}\): latent MSE plus cosine distance to `stopgrad(E(x_true))`;
- \(\mathcal L_{\mathrm{frame}}\): RGB L1, with SSIM added only if necessary; and
- \(\mathcal L_{\mathrm{circle}}\): unit-circle error for the two predicted angle pairs.

Initial weights:

\[
\lambda_1=1,\qquad
\lambda_2=1,\qquad
\lambda_3=0.1,\qquad
\lambda_4=0.01,\qquad
\gamma=0.95.
\]

Tune using normalized validation metrics and gradient magnitudes. There is **no reward or task loss in NRD training**.

## 9. Training sequence

### Stage 0 — Chrono and state-only baseline

- Validate the mechanism, state extraction, camera timing, and timestep convergence.
- Reproduce the \([z_1,a]\) state-only model using the `~/NeDM` pipeline.
- Establish one-step and open-loop physical-state errors before adding vision.

### Stage 1 — Encoder–decoder warm-up

- Train \(E_\phi,D_\psi\) on individual Chrono frames.
- First overfit a few trajectories.
- Verify that link endpoints, joint location, orientation, and colors are reconstructed.

### Stage 2 — Joint one-step model

- Encode all training frames or compute latents in the loader.
- Extend each NeDM token with \(z_2\).
- Train the two transition heads at \(H=1\), initially freezing the encoder.
- Decode the predicted next latent, not only the true encoded latent.

### Stage 3 — Multi-step curriculum

- Increase \(H:1\rightarrow5\rightarrow10\rightarrow25\rightarrow50\).
- Feed predicted \(\hat z_1\) and \(\hat z_2\) back into the model during the unrolled portion.
- Fine-tune the encoder at a lower learning rate only if frozen-latent prediction plateaus.

### Initial optimization settings

- AdamW;
- transition and decoder learning rate `3e-4`;
- encoder fine-tuning learning rate at or below `1e-4`;
- batch size 32–64 sequence windows;
- gradient clipping at 1.0;
- early stopping on combined 1-, 10-, 25-, and 50-step validation scores; and
- at least three random seeds for final comparisons.

## 10. Baselines and ablations

### Required baselines

1. Chrono ground-truth dynamics with Chrono::Sensor rendering.
2. State-only NeDM model using \([z_1,a]\).
3. Autoencoder plus a separately trained state-transition model.
4. Joint NRD using concatenated \([z_1,z_2,a]\) and two prediction heads.
5. Pose-conditioned image decoder \(D(z_1)\), which tests the expected redundancy of the fixed camera scene.
6. Persistence for both state and frame.

### Required ablations

- \(z_2\) input correct, zeroed, and trajectory-shuffled;
- frame-reconstruction loss on and off;
- one-step versus multi-step training;
- \(z_2\) dimensions 32, 64, and 128;
- NeDM temporal backbone versus a small GRU;
- decoder on versus off for throughput; and
- one encoded initialization frame versus periodic re-encoding from Chrono.

The pose-conditioned decoder is an important honesty check. If \(D(z_1)\) matches or beats the latent model, report that the first scene's pixels are fully explained by the physical state.

### Optional pretrained comparison

Do not use a VLM in the main double-pendulum study. After the scratch model works, [DINOv2](https://github.com/facebookresearch/dinov2) can be tested as a frozen feature encoder. [DINO-WM](https://github.com/gaoyuezhou/dino_wm) is a useful reference for predicting future DINOv2 patch features, but it does not provide a drop-in image decoder and is larger than necessary here.

No general pretrained decoder is recommended for this controlled Chrono scene. A decoder trained on the actual Chrono camera distribution should be cheaper and better aligned.

## 11. Evaluation

### 11.1 Physical dynamics

- one-step and 5-, 10-, 25-, 50-, and 100-step \(z_1\) RMSE/NRMSE;
- circular error for \(q_1,q_2\);
- angular-velocity error;
- endpoint-position error reconstructed from predicted \(z_1\);
- unit-circle constraint violation; and
- error-growth curves over rollout horizon.

The hybrid model should not materially degrade \(z_1\) accuracy relative to the matched state-only NeDM model.

### 11.2 Camera dynamics

- latent MSE and cosine similarity;
- frame PSNR and SSIM;
- link-endpoint pixel error extracted from rendered and predicted frames;
- temporal flicker between consecutive predicted frames; and
- qualitative videos showing Chrono truth, NRD reconstruction, and absolute difference.

Evaluate two modes:

1. **Observation-anchored:** encode the true Chrono frame at each step.
2. **Autonomous:** encode only the initial frame, then recursively predict both \(z_1\) and \(z_2\).

The autonomous mode is the real surrogate-simulation test.

### 11.3 Cross-modal consistency

From \(\hat z_1\), compute the expected joint and endpoint pixel locations using fixed kinematics and camera calibration. Compare them with the same landmarks in \(D(\hat z_2)\).

This tests whether the physical and visual prediction heads describe the same future, rather than independently achieving low average losses.

### 11.4 RL phase

After NRD training, freeze the encoder and dynamics model. Train a continuous-action swing-up policy inside NRD.

- Reward is introduced only in this RL phase and is computed from predicted \(z_1\).
- Compare policies receiving \(z_1\) and \([z_1,z_2]\).
- Transfer the learned policies without fine-tuning to Chrono.
- During Chrono evaluation, feed each Chrono camera frame through the same encoder to obtain \(z_2\), then pass \([z_1,z_2]\) to the policy.

Expect similar performance from the two policy inputs. A control advantage from \(z_2\) is not a required Study 1 result.

### 11.5 Throughput

Benchmark:

1. Chrono rigid-body dynamics without rendering;
2. Chrono dynamics plus Chrono::Sensor camera;
3. NRD transition with encoder once and decoder off;
4. NRD transition with decoder on; and
5. each configuration over increasing batch size.

Chrono's simple rigid-body dynamics may be faster for one pendulum. The meaningful Study 1 comparison is whether NRD avoids repeated camera rendering and scales efficiently in batches. The broader speed claim remains targeted at later expensive Chrono vehicle/terrain simulations.

## 12. Acceptance gates

| Gate | Requirement |
|---|---|
| G0 Chrono model | Repeatable planar dynamics, stable solver, converged timestep, and reproducible camera |
| G1 State sufficiency | \(z_1\) reconstructs mechanism pose and no hidden simulator state changes \(z_{1,t+1}\) |
| G2 Data alignment | Automated test confirms \((z_{1,t},x_t,a_t,z_{1,t+1},x_{t+1})\) timestamp alignment |
| G3 State fidelity | Joint NRD remains close to state-only NeDM and beats persistence over multi-step rollout |
| G4 Camera fidelity | Autonomous decoded frames preserve link pose and remain aligned with predicted \(z_1\) |
| G5 Policy transfer | NRD-trained swing-up policy transfers to Chrono with a pilot-defined acceptable gap |
| G6 Efficiency | Decoder-off batched throughput and camera-rendering savings are measured reproducibly |

Failure of G4 means the model is only a state surrogate, not a vision-integrated NRD. Similar \(z_1\)-only and \([z_1,z_2]\) policy results are expected and should not be labeled a failure.

## 13. Implementation work packages

### WP1 — Chrono mechanism and camera (Week 1)

- Implement and validate the two-link mechanism and elbow torque motor.
- Add the Chrono::Sensor RGB camera.
- Finalize coordinate conventions, \(z_1\) extraction, and timing.
- Add deterministic replay and timestep-convergence tests.

### WP2 — Data pipeline and state-only model (Week 2)

- Implement trajectory recorder, storage schema, split generator, and manifest.
- Generate smoke and pilot datasets.
- Train and evaluate the state-only `~/NeDM` baseline.

### WP3 — Camera autoencoder and joint NRD (Week 3)

- Train the convolutional encoder–decoder.
- Extend NeDM tokens with \(z_2\).
- Add physical and visual prediction heads.
- Complete one-step joint training.

### WP4 — Multi-step rollout and consistency (Week 4)

- Train the rollout-horizon curriculum.
- Add autonomous frame rollout and cross-modal consistency evaluation.
- Produce error curves and synchronized comparison videos.

### WP5 — RL, ablations, and report (Week 5)

- Train swing-up policies in frozen NRD.
- Transfer policies to Chrono using live camera encoding.
- Run required ablations and throughput benchmarks.
- Issue a go/no-go decision for the tabletop manipulation study.

## 14. Expected outputs

- Reproducible Chrono double-pendulum and Sensor-camera application.
- Versioned physical/camera trajectory dataset.
- State-only NeDM and joint NRD checkpoints.
- Camera encoder/decoder and autonomous visual-rollout evaluator.
- Cross-modal physical/visual consistency report.
- Frozen-NRD swing-up policy and Chrono transfer results.
- Decoder-on/off and Chrono-rendering throughput benchmark.
- Explicit conclusion separating architecture validation from evidence that vision improves control.

