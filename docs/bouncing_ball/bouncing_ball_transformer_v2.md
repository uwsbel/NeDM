# Full-state contact residuals with the NRD Transformer

The full-state version of the proposed scalar contact switch and shared bounce NN now reaches **1.508 mm p95 trajectory RMSE**, compared with **118.658 mm** for the previous full-state-contact diagnostic on the same 900 fresh Chrono episodes. This is a **78.7-fold reduction**. The bounce input changed from `[vx,vz]` to `[x,z,vx,vz,omega_y]`; its output remains a **5D state correction**. Both the contact branch and bounce branch now see the full state.

All 19 training configurations, fresh collection, checkpoint evaluation, plotting and physical replay ran on AMD compute nodes. The work is isolated in `/home/harry/NeDM-ball-transformer-v2`, branch `study/ball-transformer-v2`, base `5cae9e524aa790719aedea7c7bc1b6488c946266`. The active traversal checkout stayed clean.

## The proposed architecture, implemented literally

See the [connected architecture diagrams](bouncing_ball_architecture.md) for
network-level I/O and the complete state-feedback connection.
See the [data and timing audit](bouncing_ball_costs.md) for simulation hours,
collection/training wall time and measured single-target gradient-update cost.

```text
state s [B,5] = [x,z,vx,vz,omega_y]
   ├─ normalized state token [B,1,5]
   │    → embedding [B,1,64]
   │    → existing ContinuousTransformer (2 layers, 4 heads)
   │    → linear output [B,5] = Δfree
   │
   ├─ normalized full state [B,5]
   │    → contact MLP (5 → 256 → 256 → 32, ReLU)
   │    → latent [B,32] → scalar logit [B,1]
   │    → sigmoid probability → threshold 0.5 → gate g [B,1]
   │
   └─ normalized full state [B,5]
        → shared bounce MLP (5 → 512 → 512 → 512 → 5, tanh)
          + learned affine skip (5 → 5)
        → Δbounce [B,5]

s_next = s + Δfree + g * Δbounce
   → predicted s_next becomes the next input to every branch
```

Checkpoint: `frozen/direct_binary_mlp_shared5/best.pt` in the review bundle. Its SHA256 is `ff2e41070f08b70a88708fc49da73b4dc4e61b4e3b09f67d80c5f22fa9e98774`. [Model source](../src/nedm/bouncing_ball/transformer_temporal.py), [configuration](../configs/bouncing_ball/temporal_direct_binary_mlp_shared5.json).

The gate predicts **whether contact occurs in the next 10 ms interval**. The learned MLP makes that decision from the current predicted state. This scalar version disables the separate learned linear gate and has one shared bounce map. No analytical collision test, surface distance, time-to-impact, gravity integration, event-time head, position clamp, true contact, future state or target enters its rollout. Contact labels supervise training only. Normalization and affine initializations are fitted from training transitions.

The Transformer in this scalar candidate reuses the trained K1 free-flight core. A separate K8 / 4-layer / width128 candidate was selected as the primary before fresh testing. It uses full-state learned ground/wall gates and two full-state bounce NNs:

```text
causal predicted history [B,8,5] → Transformer → Δfree [B,5]
current state [B,5] → learned contact latent [B,32] → gates [B,2]
current state [B,5] → ground NN and wall NN → corrections [B,2,5]
s_next = s + Δfree + g_ground*Δground + g_wall*Δwall
```

## What reduced the error

1. **Bounce inputs needed position as well as velocity and spin.** Equal velocities can occur at different surfaces and at different phases inside a 10 ms interval. Position helps predict the correction to position as well as the velocity jump. In a matched affine-core/two-head control, changing bounce inputs from 2D velocity, to 3D velocity/spin, to full 5D gave fresh p95 trajectory errors **52.21, 63.81, and 1.02 mm**. Adding spin alone did not solve the missing-position problem. These controls isolate the bounce input change; they do not establish Transformer attention as its cause.
2. **Preserve the learned flight fit.** The previous rollout refinement damaged the smooth-motion branch. The new trainer fits flight separately, then freezes that core for contact training; configurations that unfreeze it retain a free-flight loss on every rollout update.
3. **Keep numerical state information through the Transformer.** The strongest precision configuration replaces the final LayerNorm with an identity and uses a linear output head. Its readout is initialized by least squares on observed free transitions. This supplies a useful data-derived starting point without supplying physics equations. The core is subsequently trained with SGD and L-BFGS.
4. **Fit the contact branches more thoroughly.** The scalar control uses a larger ReLU contact MLP, 30,000 gate updates, a 512-wide shared bounce NN, contact-focused training and L-BFGS. Every candidate also tests masked half-second autoregressive refinement. The winning scalar and primary checkpoints were selected during bounce L-BFGS, before that refinement; a longer loss did not automatically improve them.

Transformer contexts **1, 8 and 16**, depths **2, 4 and 6**, and embeddings **64, 128 and 256** were tested. With standard final LayerNorm and nonlinear output heads, fresh p95 errors were **6.21, 5.67 and 3.99 mm**, respectively. Capacity helped, but the input and training changes mattered more. The factors change together in that sweep, so it does not isolate context length.

The best precision Transformer cores remain close to their data-fitted affine initialization. They have trained, nonzero attention/MLP projections, but changing older history while keeping the current state fixed changes the K8 core output by only `6.39e-10` in the reported probe. The trained K1 and K8 errors are almost identical. Thus this campaign preserves and trains the Transformer backbone, but does **not** establish that temporal attention is necessary for this memoryless 5D problem. A separate nonzero-residual-initialized, fully trained K8 Transformer achieved **1.59 mm p95**, with appreciable nonzero block projections. Untrained affine controls remain explicitly labeled in `selection.json` and `summary.json`.

## Fresh comparison and its limits

The training set has **5,400** episodes and validation **900**. All statistics, fits and checkpoint selection use those splits only. All 19 checkpoints were frozen before evaluating **900 newly collected episodes**, seed **202610015**, two per cell in the 30×15 launch grid. Launches span `vx=4..7 m/s`, `vz=-10.5..-9 m/s`, with initial state `[0,1,vx,vz,0]`. The fixed sphere/floor/wall scene and launch distribution are unchanged.

Each episode's trajectory error is the RMS Euclidean error in `(x,z)` over its **native 10 ms samples**. The table gives quantiles across episodes. Terminal errors use each episode's valid end, roughly 1.88–2.18 s, at least 20 ms before the next ground contact. They differ from the separate fixed-1.7 s target tests.

| Model | Median trajectory RMSE | p95 trajectory RMSE | p95 terminal error | Worst terminal error |
|---|---:|---:|---:|---:|
| Previous full-state-contact diagnostic, bounce input2 |58.894 mm|118.658 mm|214.149 mm|302.860 mm|
| Standard Transformer K16 / L6 / width256, bounce input5 |1.771 mm|3.991 mm|8.883 mm|309.592 mm|
| **Scalar full-state MLP gate + one shared bounce NN** |**0.666 mm**|**1.508 mm**|**3.220 mm**|**6.526 mm**|
| Validation-selected trained K8 Transformer + two bounce NNs |0.501 mm|0.900 mm|1.580 mm|476.079 mm|
| Trained K1 core + learned two-bit OR gate + one bounce NN |0.562 mm|1.027 mm|1.929 mm|808.350 mm|
| Analytical flight/contact timing + learned MLP reference |0.488 mm|0.818 mm|1.410 mm|2.603 mm|

The analytical MLP reference is still more accurate and more consistent. Its native training step was 50 ms; its timestep-flexible analytical rollout was evaluated at the common 10 ms interval, explicitly recorded in certification metadata. The historical analytical-contact Transformer V2 has fresh p95 trajectory error **31.02 mm**. This older V2 is a different architecture, not the new state-only model.

**The selected two-head model has 13/900 large terminal outliers.** Detailed paired traces show an early learned ground switch near the return to the floor. In worst episode7088, at1.91 s, its flight state is still about0.52 mm from Chrono. The ground logit becomes `+0.7396` at ball height0.441 m, even though the actual next ground impact is at1.949125 s. The ground response then adds `Δz=-0.47578 m`, causing a476 mm terminal error and221 mm penetration. The scalar control's logit stays `-200.83`; it finishes that episode1.732 mm from Chrono. These are gate failures followed by response extrapolation, rather than flight drift.

The scalar version has **0/900 terminal errors above10 mm**, worst trajectory RMSE2.997 mm, and one contact-order failure (**899/900** correct). It is the more consistent measured version of the requested architecture. The shared map by itself did not eliminate tails: the two-bit-OR/shared-NN candidate still has five large outliers. Different gate architectures/training budgets prevent assigning that result solely to the number of bounce NNs. The primary validation selection remains frozen; this robustness comparison is reported after testing.

Between native samples, impacts are unresolved. Linear interpolation of the selected model's10 ms positions against raw0.5 ms Chrono yields **2.351 mm p95 trajectory RMSE** and **51.454 mm p95 maximum error**. Therefore the submillimeter native p95 result is neither a continuous-time bound nor a guarantee for every episode. No analytical impact reconstruction was used to reduce this number.

## Gradient launch optimization and physical replay

All three trained designs were optimized toward the same five reachable validation benchmarks at **1.7 s** from common launch `[5.5,-9.75]`. The objective is squared Euclidean terminal distance. The optimizer receives target positions, launch bounds and NRD rollouts; target-generation launches and Chrono feedback do not enter optimization.

| Target `(x,z)` m | Selected two-head: Chrono miss | Scalar contact MLP/shared NN: Chrono miss | Learned OR/shared NN: Chrono miss |
|---|---:|---:|---:|
|(3.536375,3.602006)|0.593 mm|0.588 mm|0.950 mm|
|(2.779868,2.521737)|0.662 mm|0.734 mm|0.487 mm|
|(1.767235,3.466129)|0.240 mm|0.491 mm|0.028 mm|
|(0.761837,2.230626)|0.624 mm|1.792 mm|0.189 mm|
|(0.108226,3.289743)|0.208 mm|0.262 mm|0.884 mm|

The selected and scalar models each took **45 model-only gradient iterations**; the OR/shared model took46. All15 final launches passed independent physical Chrono replay with ground then wall impact and the predeclared3 mm target tolerance. Each model's replay has six unique physical launches: one common initial launch and five optimized launches. These are five physical replay checks per model, separate from the fresh900-episode certification. Hard threshold gates give piecewise derivatives; differentiability through a fixed event sequence does not imply a smooth derivative across every gate boundary.

## Evidence and reproduction

AMD root: `/work1/dannegrut/harry/experiments/ball_transformer_v2_20261001T214000Z`.

Local review bundle: `artifacts/training_runs/bouncing_ball_transformer_v2_20261001T214000Z_amd/review_bundle_final/`.

- `selection.json`: all19 validation-only frozen selections, SHA256s, trained-versus-affine classification.
- `certification/paired_results.json`: all22 paired new/reference models, all900 per-episode metrics, teacher-forced diagnostics and launch gradients.
- `summary.json`, `comparison.csv`, `comparison.png`: measured comparisons, tail counts and Transformer-function diagnostics, including paired terminal gate traces.
- `frozen/*/best.pt`: exact evaluated checkpoints; `references/`: three paired old checkpoints.
- `targeting/*/`: gradient histories, optimized velocities, independent Chrono CSVs and physical verification JSONs.
- `runs/*/run_config.json`: configuration, source/dataset hashes, runtime, parameter count, host and AMD job ID.
- `source_snapshots/code_v1..code_v10`: immutable campaign source; `manifest.json`: every packaged file's hash and size.

The original data SHA256 is `cce4f5b5dbc7a1f84c6f22800fad3ae212d8b184547d36c2f7b61b0d11b06082`. Large datasets remain on AMD; the bundle includes their index/provenance rather than payloads. The selected two-head checkpoint SHA256 is `fe186208cc0d6c3205f43b8a7e21f54b69cd9973f869a9ef55fbf82548f0702b`.

Core loading uses `nedm.bouncing_ball.model.load_model(path, device)`, then `model.rollout(initial_state,170)` for1.7 s. The scalar model takes `[B,5]`; K8 retains its causal history internally. Loading and the contract tests were executed on AMD. Do not run training, simulation or model/plot evaluation on Luffy during other jobs.

To reproduce a training variant on AMD with a new output directory:

```bash
R=/work1/dannegrut/harry/experiments/ball_transformer_v2_20261001T214000Z
B=/work1/dannegrut/harry/experiments/ball_span_v1_20260930
sbatch --export=ALL,BALL_CODE=$R/code_v10,BALL_DATA=$B/final_data,BALL_CONFIG=$R/code_v10/configs/bouncing_ball/temporal_trained_ap_k8_w128_l4.json,BALL_RUN=$R/runs/reproduction_new,BALL_TEST=1 \
  $R/code_v10/scripts/bouncing_ball/cluster/transformer_temporal.sbatch
```

Certification job447404, primary targeting447380, shared/scalar targeting447405/447406, tests/report/export447416, and final plot/export447423 completed. No checkpoints were tuned after fresh testing, and no commit, PR, publication or release was made.

The research motivation comes from [learned contact/smooth-dynamics separation](https://arxiv.org/html/2102.11206v2), [causal Transformer physics models](https://arxiv.org/html/2010.03957), and [identity-initialized residual networks](https://arxiv.org/abs/2003.04887). The implementation uses zero-initialized output projections, not the exact ReZero parameterization. These precedents motivate experiments; the measured results above determine the claims for this model.
