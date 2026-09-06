# NeRD (Neural Robot Dynamics) — Technical Spec

Xu, Heiden, Akinola, Fox, Macklin, Narang. NVIDIA + UW. CoRL 2025. arXiv:2508.15755v1 (21 Aug 2025).
Code: https://github.com/NVlabs/neural-robot-dynamics (Apache-2.0), commit as of 2026-09-05.

Sources used: full paper text (9 pages + appendix A–C.6) and the released code. Where the paper and
code disagree, both are given. Everything marked **[UNDETERMINED]** could not be established from
either source — I have not guessed.

Empirically-verified numbers below (input/output dims, embed sizes, contact counts) were extracted by
unpickling the released `pretrained_models/NeRD_models/{Anymal,Ant}/model/nn/model.pt` checkpoints
(git-lfs), not inferred. Cartpole/Franka/Pendulum checkpoints were not fetched.

---

## 0. One-paragraph summary of what NeRD actually is

NeRD is **not** an end-to-end world model. It is a drop-in replacement for the *solver + integrator*
inside an analytical simulator (NVIDIA Warp `warp.sim`). The analytical simulator keeps doing:
URDF/model loading, **collision detection**, the low-level controller (PD → joint torques), forward
kinematics, and rendering. NeRD replaces only "given (state, contacts, joint torques), produce the
state one macro-timestep later." Implemented as a `warp.sim.Integrator` subclass with the same
`simulate(model, state_in, state_out, dt, control)` signature as Featherstone/XPBD, so it is
transparent to environments built on top (`integrators/integrator_neural.py:329`).

The critical architectural consequence, which the paper does not stress but which explains most of
its results: **the analytical collision detector re-anchors the network's environment input every
single step from the network's own predicted state.** This is a closed error-correcting loop around
an open-loop-trained network. See §5.

---

## 1. STATE REPRESENTATION

### 1.1 Nominal state (paper, §3.2)

`s_t = (x_t, R_t, q_t, φ_t, q̇_t)`

| symbol | meaning | dim |
|---|---|---|
| `x_t` | base position, world | 3 |
| `R_t` | base orientation, **quaternion** | 4 |
| `q_t` | articulated joint angles (reduced/generalized coords) | n_j |
| `φ_t` | base **spatial twist** (6-D velocity: `ω` angular then `ν` linear) | 6 |
| `q̇_t` | joint velocities | n_j |

Also: `τ_t` joint-space torque, `C_t` contact set, `g⃗` unit gravity vector.

In code the state is literally Warp's `(joint_q, joint_qd)` concatenated:
`state_dim = dof_q_per_env + dof_qd_per_env` (`integrator_neural.py`, `_build_dof_types`).
Warp's `joint_qd` for a free joint stores **angular first, then linear** (see
`compute_states_body` kernel), and the linear part is a *twist* (velocity of the body-frame origin
coincident with the world origin), not the COM velocity — the envs explicitly convert with
`lin_vel = ν − p × ω` when computing observations.

### 1.2 ANYmal case, concretely (verified against checkpoint)

- 18 DoF = 6 base + 12 actuated joints.
- `dof_q_per_env = 19` = 3 (pos) + 4 (quat) + 12 (joint angles)
- `dof_qd_per_env = 18` = 6 (base twist) + 12 (joint velocities)
- **`state_dim = 37`**
- `joint_act_dim = 12`
- **`num_contacts_per_env = 14`** (hard-coded `envs/warp_sim_envs/env_anymal.py:448`)
- Up axis is **y** (`⃗x` forward, `⃗z` sideways, `⃗y` up), per Appendix C.1.4.
- `frame_dt = 1/60 s`; ground-truth Featherstone uses `sim_substeps_featherstone = 10`
  (`sim_dt = 1/600 s`). NeRD mode uses `sim_substeps_neural = 1`.
- URDF: `assets/anymal/urdf/anymal_joint_limits.urdf`, floating base, `collapse_fixed_joints=True`,
  `enable_self_collisions=False`, stiffness 85, damping 2, armature 0.06, contact_mu 0.75.

### 1.3 What "robot-centric and spatially-invariant" means concretely

**Frame:** the robot's own **base (root body) frame at that same timestep**, `B_k = (x_k, R_k)`.
In code this is `states_frame: body`, `anchor_frame_step: every` — i.e. *every* history entry `k` in
the window is expressed in *its own* `B_k`, not in a single shared anchor. (`"first"`/`"last"`
anchoring modes exist in the code but are not what the released models use.)

**What is subtracted/normalized (Appendix A, Eq. 4–7), for the base only:**

```
x_k^{B_k} = 0                              (Eq. 4)   -> base position identically zero
R_k^{B_k} = Identity                       (Eq. 5)   -> base orientation identically identity
ν_k^{B_k} = R_k^{-1} (ν_k − x_k × ω_k)     (Eq. 6)   -> twist linear part, de-offset then rotated in
ω_k^{B_k} = R_k^{-1} ω_k                   (Eq. 7)
```

Joint angles `q_k` and joint velocities `q̇_k` are **untouched** — reduced coordinates are already
spatially invariant. Code: `_convert_states_w2b` only rewrites indices `[0:7]` and
`[dof_q : dof_q+6]`, and only if `joint_types[0] == JOINT_FREE`.

**Gravity augmentation:** because rotating the whole system into `B_k` would otherwise destroy the
information "which way is down," gravity is treated as an external force and the **unit gravity
direction expressed in `B_k`** is appended: `g⃗^{B_k} = R_k^{-1} g⃗`. In world frame it is a constant
`(0,−1,0)`-type vector; in base frame it is exactly the "projected gravity" that locomotion policies
use.

**Why this gives invariance:** the map `s → s^{B}` quotients out the full 3-D translation and the
full base rotation (not just yaw). Rotations about non-gravity axes are *not* physically symmetries,
which is precisely why `g⃗^{B}` must be added back — with it, the pair `(s^{B}, g⃗^{B})` is a
*complete* invariant of the equivalence class under SE(3) acting on the base, and the true dynamics
are a function of that class. The paper's abstract phrases the invariance as "translation and
rotation around the gravity axis" (the physical symmetry group), but the parameterization actually
implemented is full-base-frame + gravity vector, which is a strictly stronger reduction that recovers
the correct answer because gravity is re-injected.

**Prediction is also base-frame, but anchored at `B_t` (not `B_{t+1}`)** — Appendix A, Eq. 8–11:

```
x^{B_t}_{t+1} = R_t^{-1}(x_{t+1} − x_t)
R^{B_t}_{t+1} = R_t^{-1} R_{t+1}
ν^{B_t}_{t+1} = R_t^{-1}(ν_{t+1} − x_t × ω_{t+1})
ω^{B_t}_{t+1} = R_t^{-1} ω_{t+1}
```

The paper explains the reason explicitly: in `B_{t+1}` the base pose is identically
(0, Identity), so `Δx^{B_{t+1}} ≡ 0` and `ΔR^{B_{t+1}} ≡ 0` and the model could not predict base
motion at all. Anchoring at `B_t` is what makes base motion learnable.

**Sizes for ANYmal (verified from `input_rms` shapes in the checkpoint):**

| input key | dim | note |
|---|---|---|
| `states_embedding` | 37 | `states_embedding_type: identical` → pass-through of `s^{B_k}` |
| `contact_normals` | 42 | 14 × 3, rotated into `B_k` |
| `contact_points_1` | 42 | 14 × 3, transformed into `B_k` |
| `contact_depths` | 14 | scalar per slot, frame-invariant |
| `joint_acts` | 12 | joint torques, world/joint space (frame-invariant) |
| `gravity_dir` | 3 | `g⃗^{B_k}` |
| **total per timestep** | **150** | = transformer `vocab_size` (verified) |

Ant for comparison: state 29 (= 15 q + 14 qd), 25 contact slots, joint_act 8 → **215 in / 29 out**.

A `sinusoidal` state embedding (sin/cos of angular DoFs) exists in the code but the released models
all use `identical`.

---

## 2. CONTACT HANDLING

### 2.1 Input vs. predicted — unambiguous

**Contact information is 100% INPUT. NeRD predicts NO contact quantity.** There is no contact head,
no impulse output, no force output, no contact-mode classification. The only network output is the
robot state difference (§3.4). Contacts are re-derived from scratch each step by the analytical
collision detector applied to NeRD's own predicted state.

### 2.2 Where the input contact info comes from

Analytic collision detection, run once per macro step by the wrapper
`AbstractContactEnvironment.update()` (`envs/abstract_contact_environment.py:334`), *before* the
integrator is called. The GPU kernel is `collision_detection_ground`.

**Contact set is fixed-cardinality and fixed-order.** At simulator construction,
`generate_contact_pairs` enumerates a static list of candidate contact points on the robot, one slot
per geometric feature:

| collision primitive | slots | point(s) `p_0` |
|---|---|---|
| sphere | 1 | center |
| capsule | 2 | the two axis endpoints |
| box | 8 | the eight corners |
| anything else (mesh) | 1 | body origin (marked `# TODO: temporary fix`) |

Static shapes (`shape_body == -1`) and visual-only shapes are skipped. The slot list never changes
during simulation — that is the whole point of the `AbstractContact` abstraction ("ensuring that the
number of contact points does not change").

**ANYmal's 14 slots resolve exactly** (verified from the URDF collision geometry):

```
 0,1   base cylinder(len .75, r .10) -> capsule endpoints
 2,3   LF_KFE cylinder(.12, r .06)   4  LF_FOOT sphere(r .03)
 5,6   RF_KFE                        7  RF_FOOT
 8,9   LH_KFE                       10  LH_FOOT
11,12  RH_KFE                       13  RH_FOOT
```
(Consistent with the termination test in `anymal_forward_cost`, which checks slots {0,1} for torso
contact and {2,3,5,6,8,9,11,12} for knee contact.) Hips and thighs carry **no** collision geometry.

**Per-slot contact event quantities** `c^i_t = (p^i_0, p^i_1, n⃗^i, d^i)`:
- `p^i_0` — the contact point on the robot, **in the owning link's body frame**. Static per slot.
- `p^i_1` — the corresponding point on the non-robot shape (for a plane: `p_0` projected onto the plane).
- `n⃗^i` — contact normal in world frame (for a plane: the plane's up-vector).
- `d^i` — signed contact distance, `dot(p_0_in_plane_frame, plane_up)`. Zero/negative on penetration.

**Not everything in `c^i_t` is actually fed to the network.** The released input list
(`low_dim_input_names`, verified in the checkpoint) is
`[states_embedding, contact_normals, contact_points_1, contact_depths, joint_acts, gravity_dir]`.
So `p^i_0` is **omitted** (it is a per-slot constant in link frame, and the joint angles already
determine where it is), and `contact_thicknesses` is **also omitted** (used only to build the mask).
The paper's `c^i_t = (p^i_0, p^i_1, n⃗^i, d^i)` overstates the actual input by one field.

**No contact Jacobians, no penetration velocities, no friction coefficients, no material properties
are provided to the network.** Friction/restitution are implicit in the training data only.

### 2.3 The distance gate (this is the "is there a contact?" signal)

A slot is **masked to all-zero** (all of `p_1`, `n⃗`, `d` zeroed) when it is far from the surface:

- Paper (Appendix A): mask if `d^i > ξ` with `ξ = max(4 · contact_thickness, 0.1)`, fixed across all
  experiments, no per-task tuning.
- Code: `contact_masks = contact_depths < max(CONTACT_DEPTH_UPPER_RATIO · thickness,
  MIN_CONTACT_EVENT_THRESHOLD)` with `CONTACT_DEPTH_UPPER_RATIO = 4.0` and
  **`MIN_CONTACT_EVENT_THRESHOLD = 0.12`** (`utils/commons.py`). **Discrepancy: 0.1 in the paper,
  0.12 in the code.** Masking is applied in `process_neural_model_inputs`.

For ANYmal this means: base slots gate at ξ = 0.40 m, knee slots at 0.24 m, foot slots at
max(0.12, 0.12) = 0.12 m. Since `d` is measured to the *sphere center*, a 3 cm foot in contact has
d ≈ 0.03, so the gate admits up to ~9 cm of surface clearance. The band is deliberately generous:
the paper says ξ should exceed the max per-timestep contact-point displacement so no near-contact
event is missed within a step.

### 2.4 Interface with the analytical simulator

Per macro step, in order (`AbstractContactEnvironment.update`):

1. `before_update()`
2. **collision detection** (analytic, GPU) → fills `model.rigid_contact_{point1,normal,depth,...}`
3. `for _ in range(sim_substeps): self.step(..., eval_collisions=False)` — with NeRD,
   `sim_substeps = 1`, so exactly one network evaluation
4. `NeuralIntegrator.simulate()`:
   - `_update_states` pulls `joint_q/joint_qd` → torch, wraps continuous angles to (−π, π],
     grabs `root_body_q`, `joint_act`, and the freshly-computed contacts; pushes them onto the
     history deque
   - `process_neural_model_inputs`: world → base frame, embed, apply contact mask
   - `neural_model.evaluate()` → prediction at last position only
   - `convert_prediction_to_next_states` → `convert_states_back_to_world` → `wrap2PI` →
     `assign_states_from_torch`
   - `warp_utils.eval_fk(model, state_out)` — **forward kinematics is run analytically** so
     `body_q` is consistent for the next step's collision detection and for rendering
5. `after_update()`

Switching backends is `NeuralEnvironment.set_env_mode('neural' | 'ground-truth')`, which swaps the
integrator object and `sim_substeps`.

**Environment support in the released code: a single infinite ground plane per env only.**
`collision_detection_ground` is the only detector, guarded by `if model.ground:`, and
`initialize_contacts` carries the comment `# NOTE: only work for ground env for now`. The Double
Pendulum "varying contact environments" experiment randomizes the *plane's* normal (uniform random
unit vector) and offset `d` per trajectory
(`generate/trajectory_sampler_pendulum.py`) — that is the entire extent of environment variation.
No meshes, no heightfields, no multi-body contact, no self-collision.

---

## 3. NETWORK

### 3.1 Family and size

Causal decoder-only Transformer, a lightweight nanoGPT/GPT-2 (`models/model_transformer.py`, derived
from Karpathy's nanoGPT). Verified hyperparameters from the ANYmal checkpoint config:

```
n_layer   = 6
n_head    = 12
n_embd    = 384          (ANYmal, Ant, Franka);  192 (Cartpole, Pendulum)
block_size= 32           (max sequence positions; only 10 are used)
dropout   = 0.0
bias      = False        (no bias in LayerNorm or the attention/MLP Linears)
vocab_size= <per-timestep input dim>   (150 for ANYmal, 215 for Ant)
```

- `wte` is repurposed as `nn.Linear(input_dim, n_embd)` (a *learned linear token embedding*, not a
  lookup table).
- `wpe = nn.Embedding(32, n_embd)` — **learned absolute positional embeddings**.
- Blocks are pre-LN: `x = x + attn(ln1(x)); x = x + mlp(ln2(x))`, GELU, MLP hidden = 4·n_embd.
- Causal masking via `F.scaled_dot_product_attention(..., is_causal=True)`.
- `lm_head = nn.Linear(n_embd, n_embd, bias=False)` (not a vocabulary projection).
- Then an **output MLP head**: `MLPBase` with `layer_sizes=[64]`, ReLU → `nn.Linear(64, output_dim)`.
- The "encoder" in the config (`encoder.low_dim.layer_sizes: []`) is an **identity** for all released
  transformer models: the six input blocks are simply concatenated and fed to `wte`.

Parameter count (my arithmetic from the verified shapes, ANYmal): ≈ **10.87 M**
(wte 57,984 + wpe 12,288 + 6×1,770,240 + ln_f 384 + lm_head 147,456 + head 24,640 + 2,405).
Checkpoint file size 43.5 MB ≈ 10.9 M fp32 params — consistent. Ant ≈ 10.89 M.

**Table 2 vs. released code discrepancy:** the paper's Table 2 gives input embedding size 192 for
Cartpole/Pendulum/CubeTossing/Franka/Ant and 384 only for ANYmal. The released Ant and Franka configs
and the Ant checkpoint use **384**. Cartpole/Pendulum use 192, as stated.

### 3.2 History length

`h = 10` for all six robots (`num_states_history: 10`, `sample_sequence_length: 10`). Fixed across
experiments. Ablation in §5.4 below.

### 3.3 Sequence handling at inference

`TransformerNeuralIntegrator` keeps a `collections.deque(maxlen=10)` of per-step input dicts. On
`reset()` the deque is **emptied**, then grows 1, 2, …, 10 as the rollout proceeds — the causal
transformer handles the variable-length prefix natively. There is no zero-padding warm-up in the
transformer path (the non-transformer `StatefulNeuralIntegrator` *does* zero-fill; the transformer
subclass overrides that).

**Implementation caveat I found:** during RL, `RLGamesEnvWrapper` calls
`neural_env.reset_envs(done_buf)` for per-env episode resets, and that path **does not clear the
history deque**. The deque holds batched `(num_envs, dim)` tensors, so an env that resets mid-batch
carries up to 9 steps of stale pre-reset history. Only a full `NeuralEnvironment.reset()` clears it.
Not mentioned in the paper.

### 3.4 Prediction target — exactly what is regressed

`Δs^{B_t}_{t+1} ≜ s^{B_t}_{t+1} ⊖ s^{B_t}_t`, i.e. **relative (delta) state, in the base frame at time
t**, at the **macro** timestep (not per substep). Not an acceleration, not an impulse, not a force.

`⊖` is heterogeneous per DoF type (`convert_next_states_to_prediction`):

| DoF group | operator | output dims (ANYmal) |
|---|---|---|
| free-joint translation (3) | plain subtraction | 3 |
| free-joint orientation (quat) | `delta_quat(q_t, q_{t+1})` = a **quaternion** (`orientation_prediction_parameterization: quaternion`) | 4 |
| revolute/prismatic joints | plain subtraction, then `wrap2PI` for continuous joints | 12 |
| all velocities (`joint_qd`) | plain subtraction | 18 |
| **total** | | **37** (verified: `output_net.weight` is `(37, 64)`) |

`⊕` on reconstruction: `q_{t+1} = normalize( q_pred ⊗ q_t )` for orientation (note the left-multiply,
with a commented-out right-multiply alternative in the source), plain addition elsewhere, then
`normalize` and `wrap2PI`. Alternatives `exponential` (3-dim so(3)) and `naive` (additive on quats)
exist in code; released models use `quaternion`.

**Multi-substep prediction (Appendix A):** NeRD predicts the *macro-step* difference directly, spanning
the 10 (ANYmal) / 16 (Ant) / 20 (Cube Toss) / 9 (Franka) substeps the analytical simulator would take.
The paper explicitly contrasts this with prior work that predicts per-substep accelerations and
integrates. This decouples training-data fidelity from inference cost.

---

## 4. TRAINING LOSS

### 4.1 It is **one-step**, teacher-forced. There is no multi-step rollout in the loss.

`algorithms/vanilla_trainer.py:compute_loss` (SequenceModelTrainer only overrides the dataset):

```python
prediction        = self.neural_model(data)              # (B, T=10, 37), all 10 positions at once
prediction_target = data['target']                       # ground-truth Δs at each of the 10 positions
loss_weights      = 1. / torch.sqrt(output_rms.var + 1e-5)
loss              = MSELoss()(prediction * loss_weights, prediction_target * loss_weights)
```

The transformer consumes a length-10 window of **ground-truth** states and emits 10 independent
one-step predictions in parallel; the loss averages all 10. The paper calls this "teacher forcing"
[48], which is accurate — the sequence is a *history context*, not an autoregressive rollout. **The
model is never unrolled during training and no gradient ever flows through a predicted state.**
Horizon = 1. No curriculum. No scheduled sampling. No DAgger.

Loss = Eq. 2:
`L_θ = (1/NS) Σ_N ‖ NeRD_θ({s^{B_k}_k, C^{B_k}_k, τ_k, g⃗^{B_k}}_{k=t−h+1}^{t}) − Δŝ^{B_t}_{t+1} ‖²`
(N = batch size, S = state dim).

### 4.2 Normalization (the load-bearing regularizer)

- **Output normalization.** Running mean/var (`RunningMeanStd`) is computed over the *dataset*
  before training (`compute_dataset_statistics`) and used two ways:
  (a) the network's raw output is de-normalized at the end of `forward` (`output_rms.normalize(...,
  un_norm=True)`), i.e. the net predicts a whitened delta;
  (b) the loss is **per-dimension inverse-std weighted**: `w = 1/sqrt(var + 1e-5)`.
  Purpose per §4: stop the high-magnitude, high-variance velocity deltas from dominating the loss.
  Paper calls this "critical".
- **Input normalization.** Per-input-key `RunningMeanStd`, applied to all six input blocks at the top
  of `forward`/`evaluate`.
- **Gradient clipping**: `truncate_grad: True`, `grad_norm: 1.0`.
- **Dropout: 0.0** for pretraining (config comments suggest 0.1+ for fine-tuning).
- **No weight decay, no L2, no spectral norm, no EMA.** The only things standing between this model
  and autoregressive divergence are: input norm, output norm + inverse-std loss weighting, relative
  prediction, quaternion renormalization, grad clipping, and the per-step re-anchoring described in
  §5.2.
- **Weight init**: `N(0, 0.02)`, with the GPT-2 scaled init `0.02/sqrt(2·n_layer)` on residual
  `c_proj` weights.

### 4.3 Noise injection: **present in the code but never enabled**

`ModelMixedInput.forward` has an `inject_noise=False` argument that would add
`torch.randn_like(x) * 0.01` to the *normalized* inputs. I grepped the whole repo: **no call site
passes `inject_noise=True`.** It is dead code. The paper never mentions noise injection. So: no noise
injection is used to stabilize autoregression.

### 4.4 Optimization and data

| item | value |
|---|---|
| optimizer | **plain `torch.optim.Adam`**, default betas (0.9, 0.999), **no weight decay** (`vanilla_trainer.py:126`). Note: nanoGPT's `configure_optimizers` (AdamW with a decay/no-decay param split) exists in `model_transformer.py:289` but is **never called** — so there is *no* weight decay and *no* explicit regularizer anywhere in the training loop beyond the normalizations and grad-norm clipping. |
| LR | linear decay 1e-3 → 1e-4, recomputed per iteration (`get_scheduled_learning_rate`; `constant`/`linear`/`cosine` supported, `linear` used) |
| batch size | 512 sequences of length 10 |
| epochs | 1000 × 5000 iters/epoch (config); **[UNDETERMINED]** whether the released checkpoints ran the full schedule |
| dataset | 100K trajectories × 100 steps = **10 M transitions** per robot (paper §4). ANYmal released cfg points at a 20 M-transition file capped by `max_capacity: 10_000_000`. Ant released cfg: `trajectory_len-100_10M_train.hdf5`. README's tutorial uses 1 M. Cube Toss pretrain: 10 K trajectories × 100. |
| dataset sampling | sliding window: every length-10 sub-window of every trajectory is a training example |
| model selection | best-validation-loss and best-rollout-MSE checkpoints saved separately |

**Data generation** (`generate/trajectory_sampler_*.py`), ground-truth = Warp **Featherstone**:
- randomized initial states (per-robot ranges in `utils/commons.py`: `JOINT_Q_MIN/MAX`,
  `JOINT_QD_MIN/MAX`)
- **uniform random joint torques** within motor limits, resampled every step (`JOINT_ACT_SCALE`;
  ANYmal `1.5 × [50,40,8]×4` N·m)
- optionally randomized environment (Double Pendulum: random ground plane normal + offset)
- **ANYmal controller-gain randomization:** in `task="dataset"` mode,
  `AnymalJointPositionControlEnvironment.assign_control` redraws **Kp ~ U[30, 200]** and
  **Kd ~ U[0, 1]** at *every step*. `action_scale = 0.5`; target `q* = 0.5·a + q_default`. This — not
  anything in the network — is the mechanism behind "generalizes to unseen controllers/gains." The
  network only ever sees the resulting torque `τ`, never the action or the gains.
- trajectories containing NaN/Inf/|state|>1e5 are discarded wholesale

---

## 5. HOW 1000-STEP STABILITY IS ACHIEVED

This is the question with the least direct answer in the paper. What follows separates *what they
credit it to* from *what the code shows*.

### 5.1 What the paper credits it to (the six design decisions, ablated in Fig. 7 / App. C.5)

Errors are normalized so NeRD = 1.0. Two test cases: Double Pendulum contact-free 100-step passive
motion (DP), and Ant running-policy reward gap vs. GT (Ant). I recovered the exact bar values by
column-aligning the figure text:

**(a) Architecture**

| model | DP | Ant |
|---|---|---|
| **NeRD (causal Transformer)** | **1.0** | **1.0** |
| RNN (LSTM) | 1.8 | 7.4 |
| RNN (GRU) | 4.8 | 11.9 |
| MLP (h = 1, current step only) | 8.9 | 11.5 |

Their stated hypothesis: the velocity inputs are high-variance, and history lets the model infer a
*smoothed* velocity to combine with the instantaneous one. (My read: this is really a
partial-observability fix — a single macro-step transition spanning 10–16 substeps of a stiff
contact solver is not a clean function of the instantaneous state.)

**(b) Other design decisions**

| variant | DP | Ant |
|---|---|---|
| **NeRD** | **1.0** | **1.0** |
| E2E (world-frame state + torque → next state, no contact input) | **56.1** | **19.8** |
| Abs Pred (predict absolute next state instead of delta) | **26.8** | 5.4 |
| World Frame (Eq. 1 instead of Eq. 2) | 1.1 | **23.3** |
| No Input Norm | 3.3 | 3.6 |
| No Output Norm | 1.6 | 5.3 |

Readings: (i) the contact-input channel is what makes multiple environments learnable at all (E2E
collapses on DP because ground configs are unidentifiable from state alone); (ii) relative prediction
is essential even for a 2-DoF system (26× on DP); (iii) the robot-centric frame does **nothing** for a
base-fixed pendulum and is **decisive** for a floating base that walks out of the training region
(23.3× on Ant); (iv) both normalizations matter, output norm more on the contact-rich case.

**(c) History window**

| h | DP | Ant |
|---|---|---|
| 1 | 4.4 | 4.1 |
| 5 | 2.1 | **5.2** |
| 10 | **1.0** | **1.0** |

Note h = 5 is *worse* than h = 1 on Ant — the trend is not monotone. They also report that h = 20
"will occasionally result in an exploded training loss." So h = 10 is an empirical sweet spot, not a
principled choice, and the method is somewhat brittle in this knob.

### 5.2 What the code shows, that the paper does not say

The stability story is materially incomplete without these three points:

1. **The analytical simulator closes the loop every step.** After each network call, NeRD's predicted
   state is written back into Warp, `eval_fk` recomputes maximal coordinates, and the *analytic*
   collision detector recomputes `(p_1, n⃗, d)` from that state. A drifting prediction therefore
   produces a *corrected* environment input at the next step: if the model over-predicts sinkage, the
   next step's contact depths become correspondingly negative and push back. This is a physical
   feedback path that a pure autoregressive world model does not have, and it is the main reason a
   one-step-trained network survives 1000 autoregressive steps. The paper never frames it this way.

2. **The base-frame re-anchoring resets the state distribution every step.** With
   `anchor_frame_step: every`, the base pose input is *identically* (0, I) at every timestep of every
   rollout. There is no accumulating, unbounded input (unlike world-frame position, which is exactly
   what kills the World Frame ablation on Ant at 23.3×). The network's input distribution at step
   1000 is statistically identical to step 1 — drift cannot leave the training manifold *by
   translation*, only by dynamics error.

3. **The contact gate quantizes the environment signal.** Zeroing every slot beyond ξ means the
   network sees an exactly-zero vector in free flight and only sees smooth values in a narrow band
   near the surface. This suppresses one common failure mode (small state errors producing large
   spurious contact inputs far from the surface).

### 5.3 How strong is the 1000-step evidence, actually

Worth stating plainly, because the headline claim is load-bearing:

- The **1000-step open-loop state-error numbers are Cartpole only** — a 2-DoF, **contact-free**
  system: 0.033 m prismatic, 0.075 rad revolute after 1000 steps (16.67 s).
- **Ant is evaluated at 500 steps**, and the paper itself notes "the motion typically converges to a
  static state within this duration" — i.e. the terminal state is an attractor, which makes the
  averaged error easy. 0.057 m position / 0.095 rad orientation / 0.077 rad joints.
- **For ANYmal there is no reported open-loop state error at any horizon.** The only ANYmal number is
  closed-loop *reward* agreement (Table 1, 1000-step episodes): forward walk −0.02 %, sideways walk
  −0.07 %.
- That reward is `exp(−((v_x−1)² + v_z²)) + 0.5·exp(−ω_y²) − (0.002 Στ)²`. A well-trained policy sits
  near the peak of those exponentials, where the reward is **first-order insensitive** to velocity
  error. A ±0.05 % reward match is therefore a *weak* fidelity test: it is consistent with
  substantial state divergence, and it is measured under closed-loop policy correction, which
  actively suppresses divergence. The near-identical standard deviations (60.5 vs 62.4; 81.2 vs 70.4)
  are what you would expect if the reward spread is dominated by task/initial-condition structure
  rather than by dynamics fidelity. Also note the Ant Spinning task, where the reward error is
  **+17.21 %** — the one task whose reward is *not* saturating (`R = ω_y + p_up`, unbounded and
  linear) shows an order of magnitude more disagreement than every saturating reward in the table.
  That is a strong hint that the small errors elsewhere are metric artifacts, not fidelity.

So: "stable and accurate over a thousand simulation steps" is well-supported for a 2-DoF contact-free
system and for closed-loop policy reward on a quadruped; it is **not** demonstrated as open-loop
state accuracy for a quadruped over 1000 steps.

---

## 6. RL INSIDE THE MODEL

PPO (rl-games), 3 seeds per task, trained **entirely** inside the NeRD simulator from a NeRD model
that only ever saw random-torque trajectories. Evaluated over 2048 trajectories in both NeRD and the
ground-truth Warp simulator, **zero-shot, no fine-tuning or adaptation**.

| Robot | Task | Control | Horizon | GT Reward | NeRD Reward | Err |
|---|---|---|---|---|---|---|
| Cartpole | Swing Up | joint torque | 300 | 1212.5 ± 210.4 | 1212.6 ± 210.2 | +0.01 % |
| Franka | Reach | joint **position** (PD) | 128 | 89.3 ± 10.5 | 91.1 ± 9.9 | +2.02 % |
| Franka | Reach | joint **torque** (App. C.3) | 128 | 94.9 ± 7.8 | 95.0 ± 7.8 | +0.11 % |
| Ant | Running | joint torque | 500 | 2541.5 ± 309.1 | 2649.5 ± 227.4 | +4.25 % |
| Ant | Spinning | joint torque | 500 | 2624.7 ± 641.0 | 3076.2 ± 433.5 | **+17.21 %** |
| Ant | Spin Tracking | joint torque | 500 | 1630.2 ± 203.1 | 1670.5 ± 192.6 | +2.47 % |
| **ANYmal** | Forward walk 1 m/s | joint **position** (PD) | **1000** | 1323.4 ± 60.5 | 1323.1 ± 62.4 | **−0.02 %** |
| **ANYmal** | Sideways walk 1 m/s | joint **position** (PD) | **1000** | 1360.2 ± 81.2 | 1359.2 ± 70.4 | **−0.07 %** |

Four generalization axes exercised simultaneously: (1) task-induced state distributions never in the
random-trajectory training set; (2) low-level controllers — the same Franka model works under both
position and torque control, and the same ANYmal model works under a fixed-gain PD it never saw
(training randomized Kp/Kd every step); (3) spatial regions far outside the training range (Ant
running, ANYmal walking); (4) horizons up to 1000 steps.

**Sim-to-real (§5.4):** the Franka reach policy trained *only* inside NeRD was deployed zero-shot on a
real Franka, 50 random targets in-workspace. Steady-state error **NeRD-trained: 1.927 ± 0.699 mm** vs
**GT-simulator-trained: 4.647 ± 2.667 mm**. The NeRD-trained policy is *better* on hardware. The
paper does not explain why; I would not read this as NeRD being more accurate than Warp — with n = 3
seeds and a single task it is more plausibly noise or a mild-smoothing effect. **No real-robot
transfer was attempted for ANYmal or Ant.**

**Fine-tuning on real data (§5.5, Cube Toss, ContactNets dataset [15]):**
pretrain on 10 K synthetic Warp trajectories, fine-tune on 400 real trajectories (60 K transitions
total, 400/85/85 split), evaluate on 85 held-out, 80-step sub-trajectories.

| model | pos err (m) | rot err (rad) |
|---|---|---|
| Warp (analytical, hand-tuned) | 0.036 | 0.383 |
| NeRD fine-tuned | **0.018** | 0.266 |
| NeRD from scratch on real only | 0.023 | 0.276 |
| GNN-Rigid [33] | 0.032 | (not measured, code unavailable) |
| ContactNets [15] | **0.017** | **0.242** |

Fine-tuning converges in < 5 epochs, **10× faster** than from-scratch, and takes **< 10 min** vs
ContactNets' 12 h. ContactNets is still slightly more accurate on the metric.

**Speed (App. C.6):** 512 parallel Ant envs — Warp/Featherstone with 16 substeps **28 K FPS**, NeRD
**46 K FPS**. ~1.6×. The paper explicitly declines to call this definitive.

**Degradation observed:** the only non-trivial degradation is Ant Spinning (+17.21 %), and NeRD is
systematically *optimistic* (higher reward than GT) on all three Ant tasks. No degradation on
ANYmal or Cartpole by the reward metric.

---

## 7. STATED LIMITATIONS, AND DEFORMABLE/GRANULAR TERRAIN

### 7.1 Limitations the authors state (§6)

1. **Untested on high-DoF robots.** Max tested is ANYmal at 18 DoF. Humanoids (20–50 DoF, complex
   mechanisms) untested.
2. **Random trajectory sampling will not scale.** Uniform random torques give task-agnostic coverage
   at low DoF, but "may become ineffective when the state dimensionality grows." They call for better
   task-agnostic dataset construction. (This is the limitation that matters most for §8.)
3. **Fine-tuning assumes full state observability.** It requires the same state space in the real
   world as in sim — full robot state *and* the environment setup needed to run collision detection.
   Real robot data is partially observable. Fine-tuning from partial observations is future work.

### 7.2 Limitations they do **not** state, that the code makes plain

4. **One robot, one model.** NeRD is robot-*specific*: a separate model, with a different input and
   output dimensionality, per robot. No cross-embodiment sharing, no morphology conditioning.
5. **Single ground plane.** The released collision detection supports exactly one infinite plane per
   env. No meshes, heightfields, movable objects, multi-object contact, or self-collision.
6. **Fixed contact cardinality and ordering.** The contact vector is a fixed-length, fixed-order slot
   list determined at build time from the robot's collision primitives. Adding an object, or a
   robot whose contact count varies, requires re-architecting the input.
7. **No material/friction conditioning.** μ, restitution, and contact stiffness are baked into the
   training data. Changing them requires retraining. (Contrast: the *ground pose* is generalizable
   because it enters through `(p_1, n⃗, d)`.)
8. **Not differentiable end-to-end w.r.t. the simulator.** `AbstractContactEnvironment.update_grad`
   raises `NotImplementedError`.
9. **Warp-version fragility.** README: released Ant/Pendulum models need Warp 1.5.1, ANYmal needs
   Warp 1.8.0, or the numbers do not reproduce.
10. **Mesh collision is a placeholder** — a mesh body gets exactly one contact slot at the body
    origin (`# TODO: temporary fix for mesh body`).

### 7.3 Deformable / granular terrain — **confirmed: none. Zero.**

Confirmed by exhaustive search of both the paper text and the entire repository:

- Paper: the words *terrain*, *granular*, *deformable*, *soil*, *heightfield*, *soft body*, *MPM*
  appear **only** in (a) the related-work sentence listing cloth/fluid/continuum as *other people's*
  domains, (b) reference [6]'s title ("challenging terrain"), and (c) reference titles [30] (MPMNet)
  and [31] (PAC-NeRF). Never in the method, experiments, or limitations.
- Code: `grep -riE "terrain|granular|deformab|heightfield|soil|soft.?body|mpm|sph"` over all `.py`
  and `.yaml` returns **only** matches on `GEO_SPHERE` and `add_shape_sphere`. There is no terrain
  representation of any kind.
- Every environment (Cartpole, Double Pendulum, Ant, Franka, ANYmal, Cube Toss) sits on a **rigid,
  frictional, infinite plane** with `contact_ke/kd/kf/mu` constants.

NeRD contains no modeling of, no input channel for, and no evaluation on deformable or granular
media. Confirmed as expected.

---

## 8. ASSESSMENT: what would have to change for a REDUCED state on deformable granular terrain (Chrono CRM/SPH) with a quadruped

### 8.1 What transfers cleanly

These four ideas are the durable contribution and port directly:

1. **Hybrid, not end-to-end.** Replace only the solver; let the analytic layer keep doing FK,
   control, and environment queries. The E2E ablation (19.8–56.1×) is the strongest result in the
   paper and it argues for this structure independent of the domain.
2. **Robot-centric base-frame parameterization + explicit `g⃗^B`, re-anchored every step.** This
   should be adopted verbatim. It is what kept Ant stable at 23.3× advantage over world frame, and
   the argument (bounded, distribution-stationary inputs at every step of an arbitrarily long
   rollout) is domain-independent.
3. **Relative prediction with per-DoF-type `⊖`, plus per-dimension inverse-std loss weighting.**
   26.8× on a 2-DoF system. On soil, where sinkage deltas (mm) and joint-velocity deltas (rad/s)
   differ by four orders of magnitude, output whitening is not optional.
4. **Direct macro-step prediction across many substeps.** This is where the speedup lives, and it is
   *far* more valuable against CRM/SPH than against Featherstone. Featherstone at 10–16 substeps of
   1/600 s only buys 1.6×; a CRM/SPH step is O(1e-4 s) or smaller with 1e4–1e6 particles, so the
   ratio is three-to-five orders of magnitude, not 1.6×. The economics of NeRD are far better in your
   domain than in theirs.

### 8.2 The four structural blockers, in order of severity

**(1) The environment channel has no analogue, and the analytic producer is gone.**

NeRD's entire environment representation is `{(p_1^i, n⃗^i, d^i)}` for a fixed slot list — the output
of a *cheap, closed-form, memoryless* plane query. In CRM/SPH there is no such object. There is no
"the contact point on the terrain," no single normal, and the "penetration depth" is a continuum
sinkage with a distributed pressure field. Worse: NeRD requires this query to run **every macro
step, cheaply, from the network's own predicted state**. That query is the entire error-correction
mechanism of §5.2. If you have to run CRM to produce it, you have not replaced anything.

You need a substitute environment channel with three properties: cheap, robot-centric, and
*derivable from the predicted reduced state*. Concretely, per foot (12–14 slots, keeping the
fixed-cardinality design):
- local terrain height and normal under the foot, from a maintained heightfield
- sinkage `z` and sinkage rate `ż`
- slip ratio / tangential velocity of the contact patch relative to the soil surface
- contact patch area or an equivalent-plate width proxy
- a small vector of **local soil-state** features: relative packing fraction / bulk density, prior
  disturbance count, saturation if relevant

This is essentially the Bekker–Wong / SCM feature set, and that is the right shape: it is what a
reduced terramechanics model consumes. Note this changes the character of the model: NeRD's contact
input is *purely kinematic* (geometry only); yours must carry *material state*.

**(2) The terrain has its own state, and NeRD has none.**

This is the deepest gap. A rigid plane is a static, instantaneously-queryable field: the environment
input at step 1000 depends only on the robot pose at step 1000. Granular soil is **history-dependent
and spatially persistent** — ruts, compaction, berms, and flow persist and are re-encountered.
NeRD as published is Markov in a 10-step *robot* window with **zero environment memory**.

So the model must be promoted from `s_t → Δs_{t+1}` to a **co-evolving pair**:

```
(s^B_t, F^B_t, τ_t, g⃗^B_t)  →  (Δs^{B_t}_{t+1}, ΔF^{B_t}_{t+1})
```

where `F^B` is a **robot-centric local terrain field** — a heightmap plus a compaction/density map,
cropped in a box around the base and *rotated with the base* so the invariance argument survives.
Two design consequences:
- The field must be stitched back into a persistent world-frame terrain buffer after each step (a
  scatter/gather, cheap on GPU), so that a rut dug on step 200 is still there on step 800.
- The base-frame invariance now requires the field to be resampled into `B_t` each step, which is a
  bilinear gather, not a rigid transform of a fixed-size vector. Cost is small but nonzero.

This turns a 150-dim MLP-token problem into a small vision problem (a conv/patch encoder over the
local field, concatenated into the token). That is a real increase in scope, but it is the minimum
that makes rutting representable.

**(3) A reduced state breaks the Markov property that the h = 10 window barely covers.**

NeRD's `h = 10` is already compensating for partial observability introduced by macro-stepping across
10–16 stiff substeps — that is the honest reading of the architecture ablation (MLP 8.9–11.5×). A
*reduced* state (say: base 6-DoF + a low-dim gait/limb descriptor, dropping full 12+12 joint state)
strictly increases the hidden state. Consequences:
- You cannot simply extend `h`. The authors report h = 20 sometimes explodes the training loss, and
  h = 5 was *worse* than h = 1 on Ant. Scaling the window is not a reliable lever.
- The right move is an explicit **learned latent** for the unobserved part (soil state under each
  foot, limb-internal state), carried across steps like the terrain field, rather than a longer
  window. This is a real departure from NeRD's stateless design.
- **You lose the FK/collision feedback loop.** From a reduced state you cannot run `eval_fk` and get
  foot positions, so you cannot query the terrain field, so §5.2's error correction disappears. You
  must add a **kinematic decoder**: reduced state → foot contact-point poses. Analytic if the
  reduction preserves enough (e.g. keep joint angles, drop joint velocities), learned otherwise. I
  would strongly prefer *keeping full joint positions in the state and reducing only the terrain
  side*, precisely to preserve this loop. Reducing the robot state and the terrain state
  simultaneously removes the one mechanism that makes 1000 steps work.

**(4) The data-generation strategy does not survive the move.**

Two independent problems, both fatal if ignored:

- *Cost.* 10 M transitions at 1/60 s = 46 h of simulated time per robot. Featherstone at 28 K FPS
  produces that in minutes. CRM/SPH quadruped-on-soil is at best low-multiples of realtime and
  realistically sub-realtime; call it 10³–10⁵× slower per transition. Naively reproducing NeRD's
  dataset is weeks-to-months of GPU per soil parameterization, and you need a *family* of soils.
- *Coverage.* Uniform random joint torques work on rigid ground because the robot tumbles and slides
  through a rich contact manifold cheaply. On deformable soil, random torques put the quadruped in a
  hole in the first second and keep it there. The interesting regime — sustained locomotion, slip
  ratio 0.1–0.5, rut formation, cyclic re-loading of pre-compacted soil — is a vanishingly small
  subset of random-torque state space. This is exactly the authors' own stated limitation (2) and it
  bites hardest here.

The mitigation is the one recipe in the paper that is already validated for a distribution shift:
**pretrain cheap, fine-tune expensive.** Pretrain on a fast analytic terramechanics model (Chrono
SCM, or a Bekker/Wong pressure-sinkage + Janosi-Hanamoto shear model) over a wide randomized soil
parameter family, then fine-tune on a much smaller CRM/SPH dataset — the Cube Toss result
(converges in < 5 epochs, 10× faster than scratch, 400 real trajectories) is direct evidence that
this works across a sim-to-different-physics gap. Combine with **policy-in-the-loop data
collection** (roll out a locomotion policy in the NeRD-integrated sim, collect the CRM ground truth
only on the states actually visited) — accepting that this sacrifices the task-agnostic property
the authors were protecting. For a single locomotion domain that trade is worth making.

### 8.3 Things that need to be re-derived, not just re-parameterized

- **The invariance argument itself.** NeRD's spatial invariance is exact because a plane's effect on
  the robot is fully captured by per-point `(p_1, n⃗, d)`. With granular soil: translation invariance
  survives (soil is homogeneous in the mean), and gravity-axis rotation invariance survives **only if
  the soil is isotropic** — which it is not once rutted, and not on a slope with directional flow.
  Keep the base-frame parameterization and rotate the local field with the base, but do not claim
  exact invariance; it becomes an approximate symmetry with the field carrying the anisotropy.
- **The contact gate ξ.** `max(4·thickness, 0.12)` is a rigid-body geometric heuristic. For soil the
  analogous gate is a *sinkage* threshold and it must interact with the soil-state channel (a foot 5
  cm above undisturbed soil and a foot 5 cm above a 10 cm rut are different situations). A binary
  distance gate will destroy that distinction.
- **The `⊖` operator on the reduced state.** NeRD's per-joint-type decomposition (quaternion delta
  for the free joint, plain subtraction elsewhere, `wrap2PI` for continuous joints) is hard-coded
  against the Warp articulation layout. Any reduced state needs its own Lie-group-consistent delta
  for the base pose and its own wrapping rules. Keep the quaternion-delta + renormalize pattern; it
  is the right choice and is worth 26.8× on its own.

### 8.4 Verification, before anything else

NeRD's headline agreement numbers (−0.02 %, −0.07 % on ANYmal) are produced by a **saturating,
closed-loop reward metric**, and the one non-saturating reward in their table (Ant Spinning) shows
+17.21 %. Before adopting any of this, fix the evaluation protocol: report **open-loop state-space
divergence vs. CRM ground truth at 100 / 500 / 1000 steps**, per-DoF, for a quadruped, and report
terrain-side error (rut depth, sinkage) separately from robot-side error. A reward-parity result on
soil would be evidence about the reward's insensitivity, not about the model.

### 8.5 Bottom line

The *hybrid architecture*, the *base-frame re-anchoring*, the *relative whitened prediction*, and the
*macro-step* design all port and are worth porting — and the speed argument is far stronger against
CRM/SPH than against Featherstone. But NeRD's environment representation is a rigid-plane geometric
query with **no material state, no memory, and no cost model for producing it**, and its stability
depends on that query being cheap enough to re-run every step. Replacing it with a persistent,
robot-centric, co-evolved soil field — and paying for the data with a pretrain-on-SCM /
fine-tune-on-CRM schedule rather than uniform random torques — is not a modification of NeRD. It is a
different model that borrows NeRD's four good structural ideas. Scoping it as the latter is the
honest framing.

---

## Appendix: quick-reference numbers

| | Cartpole | Pendulum | Cube Toss | Franka | Ant | **ANYmal** |
|---|---|---|---|---|---|---|
| DoF | 2 | 2 | 6 (free rigid) | 7 | 14 | **18** |
| `state_dim` | 4 | 4 | 13 | 14 | 29 | **37** |
| contact slots | 0* | 4 (2 capsules) | 8 (box) | — | 25 (1 sphere + 12 capsules) | **14** |
| NeRD input dim | — | — | — | — | **215** | **150** |
| NeRD output dim | — | — | — | — | **29** | **37** |
| `n_embd` | 192 | 192 | (192 per paper) | **384** (code) | **384** (code) | **384** |
| GT substeps | — | 5 | 20 | 9 | 16 | 10 |
| `frame_dt` | 1/60 | 1/60 | **0.0067568** (= dataset rate) | **1/120** | 1/60 | 1/60 |
| RL episode | 300 | — | — | 128 | 500 | **1000** |

\* Cartpole is contact-free. Blank cells = not fetched / not stated. Cube Toss is a single free rigid
box (`add_shape_box` + `add_joint_free`) — it still goes through the articulation machinery, which is
why NeRD covers "articulated *and* single rigid bodies." Franka contact slots not counted (fixed
base, reach task in free space; the env adds one sphere shape).

Shared across all six: `h = 10`, `n_layer = 6`, `n_head = 12`, `block_size = 32`, `dropout = 0`,
`bias = False`, output MLP `[64]`, batch 512, LR linear 1e-3 → 1e-4, grad-norm clip 1.0,
input + output normalization on, one-step teacher-forced MSE, no noise injection, no curriculum.

Files worth reading first, in order:
- `/private/tmp/.../nerd-code/integrators/integrator_neural.py` (1104 lines — the whole method)
- `/private/tmp/.../nerd-code/envs/abstract_contact_environment.py` (contact slot enumeration + the
  one and only collision detector)
- `/private/tmp/.../nerd-code/algorithms/vanilla_trainer.py:304` (`compute_loss` — 20 lines, settles
  the one-step question)
- `/private/tmp/.../nerd-code/generate/trajectory_sampler_anymal.py` +
  `envs/warp_sim_envs/env_anymal_joint_position_control.py:81` (the Kp/Kd randomization)
