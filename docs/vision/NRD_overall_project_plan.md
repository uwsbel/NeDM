# Neural Reduced Dynamics (NRD): Overall Project Plan

**Status:** Research plan  
**State-only reference:** `~/NeDM`  
**Scope:** High-level direction; implementation details belong in the individual study plans.

## 1. Project objective

Develop a fast learned dynamics model that combines:

- an explicit physical state, \(z_{1,t}\), such as position, velocity, orientation, and joint state;
- a compact sensor latent, \(z_{2,t}\), encoded from cameras or other high-dimensional sensors; and
- the control action, \(a_t\).

The sensor latent is **appended to—not substituted for—the physical state**:

\[
z_{2,t}=E_\phi(x_t),\qquad
z_t=\operatorname{concat}(z_{1,t},z_{2,t}), \qquad
(\hat z_{t+1},h_{t+1})=F_\theta(z_t,a_t,h_t),
\]

where \(h_t\) is optional recurrent memory. The two output branches predict \(\hat z_{1,t+1}\) and \(\hat z_{2,t+1}\), and a decoder maps \(\hat z_{2,t+1}\) to the future sensor frame. The decoder should normally be disabled during policy rollouts.

For each system, \(z_1\) must first be chosen to be a sufficient Markov state for the modeled mechanics under fixed system parameters. \(z_2\) then carries the high-dimensional sensor representation and its evolution. In the first double-pendulum study, this intentionally makes vision redundant with respect to mechanics: the purpose is to validate synchronized physical and camera prediction. Later studies must place information in the sensors that is not already available in \(z_1\).

NRD extends the state-only dynamics approach in `~/NeDM`: preserve its explicit-state interface and learned transition-model role, then add a sensor encoder, sensor-latent prediction, optional decoding, and multimodal evaluation.

## 2. Core research claim

The intended contribution is not simply “an autoencoder attached to NeDM.” The stronger claim to test is:

> A hybrid recurrent state containing explicit physical variables and learned sensor latents can support accurate long-horizon dynamics, preserve task-relevant information that is absent from the explicit state, and enable faster batched policy training than the original high-fidelity simulator while retaining policy transfer.

This claim has four required parts:

1. **Dynamics fidelity:** NRD predicts \(z_1\) and \(z_2\) over useful rollout horizons.
2. **Visual usefulness:** \(z_2\) contains information that is not already present in \(z_1\).
3. **Control transfer:** policies trained or evaluated with NRD remain effective in the original simulator and, later, on real sensor input.
4. **Computational value:** NRD provides a meaningful throughput advantage at the target scale, especially with many parallel environments and the decoder disabled.

## 3. Model boundary

### NRD pretraining

The general dynamics model is trained from transition sequences without reward labels:

\[
\mathcal L_{\mathrm{NRD}}=
\lambda_1\mathcal L_{z_1}
+\lambda_2\mathcal L_{z_2}
+\lambda_3\mathcal L_{\mathrm{reconstruction}}
+\lambda_4\mathcal L_{\mathrm{rollout}}
+\lambda_5\mathcal L_{\mathrm{constraints}}.
\]

- \(\mathcal L_{z_1}\): physical-state prediction.
- \(\mathcal L_{z_2}\): future sensor-latent prediction.
- \(\mathcal L_{\mathrm{reconstruction}}\): optional decoded-frame or decoded-sensor accuracy.
- \(\mathcal L_{\mathrm{rollout}}\): multi-step consistency, implemented by unrolling the same transition model.
- \(\mathcal L_{\mathrm{constraints}}\): known normalization or physical constraints when appropriate.

There is **no reward, success, or task loss in the core NRD objective**. Task-specific reward logic belongs to the downstream RL or planning phase. If imagined RL needs a learned reward estimator, train it afterward as a separate task adapter; do not redefine it as part of the general NRD dynamics model.

### Downstream use

After NRD pretraining:

1. freeze the encoder and transition model for the first policy experiments;
2. expose \([z_1,z_2]\) to the policy;
3. provide reward through the RL environment or a separate reward adapter;
4. keep the decoder off unless a task or diagnostic explicitly requires reconstructed observations; and
5. validate the resulting policy in the source simulator before making transfer claims.

## 4. Architecture strategy

Use the smallest architecture that answers the research question at each stage.

| Stage | Sensor representation | Transition model | Decoder |
|---|---|---|---|
| Chrono double pendulum | Small convolutional encoder trained from scratch | Extend the `~/NeDM` temporal backbone with concatenated \([z_1,z_2,a]\); GRU smoke-test optional | Mirrored convolutional decoder |
| Tabletop manipulation | Patch or object-centric features; compare scratch encoder with DINOv2 | Token-based recurrent transformer | Optional lightweight decoder |
| HMMWV terrain | Multi-camera/depth/terrain tokens plus proprioception | Larger recurrent transformer with modality fusion | Diagnostic only |
| Language-conditioned extension | Vision-language encoder only if language is an actual input | Multimodal recurrent model | Task dependent |

A VLM is not the default encoder. It adds cost and semantic priors that are unnecessary for a synthetic pendulum and may discard motion details. Pretrained visual encoders become a controlled baseline when scenes contain varied objects, textures, viewpoints, or semantics.

## 5. Research progression

### Phase 0 — NeDM interface and baseline

- Audit `~/NeDM` and document its state normalization, action interface, sequence format, transition API, rollout API, and throughput measurement.
- Reproduce one state-only training and rollout result.
- Define an NRD interface that remains compatible with the state-only case when \(z_2\) is absent.

**Exit criterion:** a reproducible state-only baseline and a versioned data/model interface.

### Phase 1 — Double pendulum with vision

- Build an actuated planar double pendulum in Project Chrono with a synchronized Chrono::Sensor camera.
- Define a complete physical state \(z_1\), append the camera latent \(z_2\), and jointly predict their next values.
- Validate autonomous physical-state and reconstructed-camera rollouts.
- Test whether an NRD-trained swing-up policy transfers to Chrono, while expecting no policy advantage from \(z_2\) in this fully observed case.

**Exit criterion:** stable multi-step physical and camera rollouts, consistent decoded pose, and policy transfer to Chrono. This phase validates the architecture, not the added control value of vision.

### Phase 2 — Tabletop arm manipulation

- Add occlusion, object identity, contact, and camera-view variation.
- Compare a scratch CNN, DINOv2-style patch features, and object-centric features.
- Evaluate goal-conditioned manipulation and policy transfer across object and scene variations.

**Exit criterion:** useful visual latents under contact-rich dynamics and novel scene configurations.

### Phase 3 — HMMWV terrain traversal

- Integrate proprioception with camera, depth, lidar, or terrain observations.
- Model partial observability, moving obstacles, terrain geometry, tire–soil interaction, and longer temporal context.
- Benchmark batched NRD against Chrono-based data generation and policy evaluation.

**Exit criterion:** clear throughput gain at scale with acceptable trajectory and policy-transfer error.

### Phase 4 — Generality and VLA bridge

- Reuse the same \((z_1,z_2,a)\) contract across domains.
- Test sensor dropout, encoder freezing, cross-camera transfer, and limited real-data adaptation.
- Add language only for explicitly language-conditioned goals or instructions.

**Exit criterion:** evidence that the interface and training recipe transfer across at least two substantially different domains.

## 6. Evaluation framework

Every study should report the same five categories:

1. **One-step fidelity:** normalized physical-state and latent prediction errors.
2. **Open-loop fidelity:** error growth, stability, constraint violation, and visual quality versus rollout horizon.
3. **Representation quality:** reconstruction, physical/visual consistency, frozen-latent probes, and controlled \(z_2\) masking or shuffling. From Phase 2 onward, this must also demonstrate control-relevant information absent from \(z_1\).
4. **Control performance:** learned-environment return, source-simulator return, success rate, and transfer gap.
5. **Efficiency:** training cost, encoder latency, batched transition throughput, memory use, and decoder-on/off throughput.

Required baselines are:

- state-only model following `~/NeDM`;
- direct simulator;
- image autoencoder plus one-step latent dynamics;
- recurrent multi-step NRD;
- a pretrained visual-encoder baseline when the scene is complex enough to justify it.

## 7. Main risks and responses

| Risk | Consequence | Planned response |
|---|---|---|
| \(z_2\) duplicates \(z_1\) | Study 1 cannot claim added control information | Accept this as a controlled architecture test; include visual-only information from Study 2 onward and conduct \(z_2\) ablations |
| Reconstruction emphasizes background pixels | Good-looking frames but poor control representation | Keep latent dynamics primary; lower reconstruction weight; use crops or masks only as ablations |
| Recursive rollout drift | Policies exploit model errors | Multi-step training, held-out rollouts, uncertainty/error checks, simulator transfer |
| Encoder changes destabilize latent targets | Transition target moves during training | Warm-start encoder; use stop-gradient targets; freeze then cautiously fine-tune |
| Pretrained encoder loses motion detail | Weak future prediction | Compare patch features and scratch CNN; supply temporal context and explicit velocities |
| NRD is slower than a simple simulator | No efficiency case at small scale | Treat double pendulum as validation; make speed claims only against the intended expensive simulator |
| Task specialization harms generality | Model cannot be reused | Keep reward/task heads outside core NRD and test multiple downstream tasks per trained model |

## 8. Project-level deliverables

- Versioned NRD dataset schema and loaders.
- State-only compatibility layer for `~/NeDM`.
- Modular sensor encoder, recurrent transition model, and optional decoder.
- Reproducible training and multi-step evaluation scripts.
- RL/planning adapter with external reward handling.
- Benchmark suite covering fidelity, control transfer, throughput, and ablations.
- Study reports for double pendulum, tabletop manipulation, and HMMWV terrain traversal.

## 9. Go/no-go logic

Proceed from one phase to the next only when:

- the hybrid model does not materially degrade \(z_1\) prediction relative to the state-only baseline;
- in Study 1, \(z_2\) passes camera reconstruction, prediction, and cross-modal consistency tests; from Study 2 onward, it also passes a control-usefulness test;
- open-loop rollouts remain usable over the task's control horizon;
- a policy or planner transfers back to the source simulator; and
- the projected computational advantage justifies increasing scene and model complexity.
