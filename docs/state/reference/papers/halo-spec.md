# HALO: technical spec

**Paper:** Werner\*, Esteban\*, de Sa, Cohen, Ames. "HALO: Hybrid Auto-encoded Locomotion with
Learned Latent Dynamics, Poincaré Maps, and Regions of Attraction." arXiv:2604.18887v1 [cs.RO],
20 Apr 2026. L4DC 2026 (PMLR vol 331:1-20). Caltech + NC State. Funded by TII.

**Sources used**
- PDF: `/private/tmp/claude-501/-Users-kylesha/e7393dd9-22e3-4b6c-b121-0d82bc75fba8/scratchpad/papers/halo.pdf` (20 pages: 10 body + refs + Appendices A/B/C)
- Extracted text: `.../papers/halo.txt`
- Released code, read directly (this is where most of the operational detail lives, not the paper):
  `github.com/sesteban951/halo-latent-locomotion`. Local copies in `.../papers/repo/`:
  `data_warp_parse_g1_23dof_data.py`, `data_mjx_parse_hopper_data.py`, `data_warp_parallel_sim.py`,
  `utils_dataloader.py`, `utils_auto_encoder.py`, `utils_training.py`, `scripts_train.py`.

Anything marked **[CODE]** comes from the repository and is not stated in the paper. Anything marked
**[UNDETERMINED]** I could not establish from either source and am not guessing at.

---

## 0. One-paragraph orientation

HALO is a straightforward autoencoder-plus-latent-map pipeline applied to pre-impact states of a
clock-driven RL walking policy in simulation. The Poincaré machinery is classical (Westervelt/Grizzle
step-to-step dynamics). The novel claim is that a learned latent Lyapunov sublevel set, decoded, is a
better ROA estimate than naive sampling. The theory (Lemmas 1-3, Thm 5) is about idealized exact
encoders on exact invariant manifolds and does not constrain the learned networks. There is no
limitations section, no ablation, no baseline ROM (no LIP/SLIP comparison), and no hardware number
beyond a single 9-step qualitative plot.

---

## 1. THE POINCARÉ SECTION

### 1.1 Formal definition (paper, Appendix A)

Switching surface, Eq. (18):

    S = { x ∈ R^{n_x} : s(x) = 0, ṡ(x) < 0 }

with `s : R^{n_x} → R` continuously differentiable. Reset at impact, Eq. (19): `x⁺ = Δ(x⁻)`.
Time-to-impact, Eq. (22): `T_I(x) := inf{ t ≥ 0 : φ_t(Δ(x)) ∈ S }`. Poincaré map, Eq. (23):

    x_{k+1} = f(x_k),   f(x) := φ_{T_I(x)}( Δ(x) ),   f : S̃ → S

The section is therefore **foot touchdown (impact), and the recorded state is the pre-impact state
`x_k := x⁻`**, i.e. the state at the last instant before ground contact. Not apex, not lift-off.
Figure 3 draws `x⁻` for all three systems as the instant just before contact.

Note `f` is explicitly a **partial function** (paper's words), citing Ames et al. 2014 and
Westervelt et al. 2003.

Theorem 8 (quoted from Westervelt et al. 2018) requires `Δ(S) ∩ S = ∅` and gives
stable/asymptotically stable/exponentially stable equivalence between the fixed point of `f` and the
periodic orbit `O`.

### 1.2 What the state vector actually is

For the G1 the state is the **full generalized state** `x = (q, v)`, concatenated:
`n_q = 30` (floating base position 3 + quaternion 4 + 23 joints), `n_v = 29`, so `n_x = 59`.
Table 6 footnote: "The G1 state vector is 59-dimensional due to the quaternion representation of base
orientation (n_q = 30, n_v = 29), though the underlying state space is 58-dimensional."

**[CODE]** The quaternion is left as 4 raw components with no manifold handling. The decoder emits an
unconstrained 4-vector and the L2 losses treat it like any other coordinate.

Frames (paper, "Poincaré Data Collection"): world frame for paddle-ball, relative to foot position
for the hopper, **yaw-aligned foot frame** for the humanoid.

**[CODE]** `transform_base_to_yaw_foot_frame(..., foot_site_name="right_foot")`: origin at the right
foot site, z up, x along the foot heading (yaw only, roll/pitch discarded). Base position, base
quaternion, base linear velocity and base angular velocity are all re-expressed in that frame.
Joint entries `q[7:]` and `v[6:]` are left untouched. So the section state is
"stance-foot-relative floating base + raw joint state".

### 1.3 How the crossing is detected in data **[CODE]** (the paper says almost nothing here)

The paper says only: "At each impact, we use MuJoCo sensor measurements to record the generalized
position and velocity in a contact-relative frame... After pruning faulty Poincaré data, such as
impacts with sliding contact or contact chatter, we obtain clean sequences of Poincaré returns."
Remark 7 adds that on hardware one could use kinematic (foot height/velocity), dynamic (generalized
velocity/momentum change) or force (torque/GRF) methods, but they do not do this.

The actual pipeline, from `data/warp/parse_g1_23dof_data.py`:

1. Simulate at `sim_dt = 1e-3` (1 kHz) with a 50 Hz policy. Log `q_log`, `v_log`, and `c_log`
   (MuJoCo `mjSENS_TOUCH` scalar per foot) **at the 1 kHz sim rate**, not the control rate.
2. Binarize: `c_bool = (c_data > 1e-6)`. So the contact "threshold" is essentially any nonzero
   normal force.
3. Rising edge: `transition = c_bool[:, 1:, :] - c_bool[:, :-1, :]`, `preimpact = (transition == +1)`.
   Index `k` in the differenced array is the last 1 kHz sample **before** contact. That sample's
   `(q, v)` is the pre-impact state. No sub-sample interpolation.
4. Pruning, in order:
   - undesired contacts (empty list for the G1; for the hopper it is the torso touch sensor with
     threshold 0.05, and an entire trajectory is dropped if it ever fires),
   - base height `q[2] < 0.3 m` at any time (fallen) → drop trajectory,
   - the **last** right-foot inter-event interval `< gait_period/2 = 0.3 s` → drop trajectory,
   - **chatter**: two consecutive time-to-impact intervals `< 0.1 s` on any channel → drop trajectory,
   - drop the first impact of every trajectory (sim init transient).
   Pruning is per-trajectory, all-or-nothing. Raw 30 × 4096 = 122,880 trajectories reduce to the
   108,403 training trajectories of Table 6, so roughly 12% are discarded.
5. `K = min` number of pre-impacts across the batch and channels; per-trajectory event lists are
   truncated (or right-padded by repeating the last index) to `K`. After dropping the first, the
   training horizon is `K = 8` for the G1.

I found **no code implementing the "sliding contact" pruning** the paper mentions. The G1's
"undesired contact index" list is empty. Only the z-height, short-interval and chatter filters exist.

### 1.4 THE CRUX: multiple legs, unsynchronized events

**This is the direct answer: they do not handle it. They pick one leg and throw the other away.**

**[CODE]** `parse_g1_23dof_data.py`:

```python
contact_idx_dict = {
    "g1_23dof": ([], [0, 1]),  # (None, Left foot / Right foot)
}
...
right_foot_ch = 1  # switching_idx: [0=left, 1=right]
```

The parser builds a pre-impact tensor with a **channel axis** of size `nc_switch = 2`:
`q_preimpact` has shape `(B, 2, K, n_q)`, i.e. it stores a *separate* event sequence for the left
foot and for the right foot. Then `utils/dataloader.py` does:

```python
# choose the contact channel to use, NOTE: just choose the default first one for now
t_preimpact = t_preimpact[:, 0, :]     # (B, K)
q_preimpact = q_preimpact[:, 0, :, :]  # (B, K, nq)
v_preimpact = v_preimpact[:, 0, :, :]  # (B, K, nv)
```

Channel 0 is the **left foot**. The right foot's events are computed, used only for pruning, and
then discarded. The comment "just choose the default first one for now" is theirs, verbatim.

Consequences, stated precisely:

- The section for the G1 is **left-foot touchdown only**. Consecutive section crossings are one
  **full stride** (two steps) apart, not one step. The paper's own prose ("f maps from the state at
  one foot strike to the state at the next foot strike") is therefore inaccurate for what was
  actually implemented: it is same-foot to same-foot.
- The state at left-foot touchdown is expressed relative to the **right** foot, i.e. the stance foot
  at that instant. Self-consistent, but it means the frame is hard-coded to the choice of section leg.
- There is no mechanism, discussion, or experiment involving multiple simultaneous or interleaved
  events. There is no per-leg phase variable, no event-type label, no multi-section (hybrid-automaton)
  formulation. **Yes: the method as implemented only handles systems with a single designated event
  per cycle.**
- The infrastructure (`nc_switch` channels) would let you index a *different* leg, but not to use
  more than one leg's events in one model. Interleaving left and right events into one sequence
  would produce a map with two distinct fixed points (mirrored states), which their single-`z*`
  linearization and single Lyapunov function cannot represent.

### 1.5 The section is quasi-time-triggered, not genuinely event-triggered

**[CODE]**, and this matters a lot for transplanting the method. `data/warp/parallel_sim.py`:

```python
phase = (sim_time % self.gait_period) / self.gait_period
gait_phase = torch.stack([sin(2π·phase), cos(2π·phase)], dim=-1)
```

with `gait_period = 0.6 s`, `control_dt = 0.02 s` (50 Hz), fixed velocity command
`cmd = [0.5, 0.0, 0.0]` m/s, flat ground. The policy observation (Table 5, 80-dim) contains this
2-D gait phase plus the previous action.

Three implications the paper does not discuss:

1. The gait period is **imposed exogenously by an open-loop clock**, not emergent. Touchdowns occur
   at nearly fixed clock phase, so the left-to-left "Poincaré map" is approximately a fixed-Δt = 0.6 s
   flow map wearing event-triggered clothing.
2. The closed-loop system is **non-autonomous and has controller memory**: the true state is
   `(x, clock phase, previous action)`. The learned map takes only `x = (q, v)`. The construction
   `x_{k+1} = f(x_k)` is therefore not literally well-posed; it works only because the clock phase at
   the section is nearly constant across the dataset.
3. Initial conditions are drastically narrow: `q_lb == q_ub == q_nominal` exactly (a single fixed
   configuration, "left leg touching ground, right leg raised"), with randomization applied **only to
   velocities**, uniform ±0.5 on base linear x/y, 0 on base z, ±0.5 on all base angular and all 23
   joint velocities. Every rollout starts at the same pose and the same clock phase.

---

## 2. LATENT REPRESENTATION

### 2.1 Architecture (paper §4 "Autoencoder Architecture" + Table 6, confirmed in code)

Three networks, all MLPs, all swish activation, He-uniform kernel init, zero bias init **[CODE]**:

- Encoder `z_k = E_φ(x_k)`
- Decoder `x̂_k = D_ψ(z_k)`
- Latent dynamics, **residual form**, Eq. (8): `z_{k+1} = g_ρ(z_k) := z_k + ḡ_ρ(z_k)`

| Property | Paddle-Ball | Hopper | G1 Humanoid |
|---|---|---|---|
| FOM state dim `n_x` | 4 | 8 | 59 |
| Latent dim `n_z` | **2** | **4** | **12** |
| Encoder hidden | [64, 32, 16] | [64, 32, 16] | [256, 128, 64] |
| Decoder hidden | [16, 32, 64] | [16, 32, 64] | [64, 128, 256] |
| Dynamics hidden | [64]×3 | [64]×3 | [128]×3 |
| Activation | swish | swish | swish |
| Training steps | 5,000 | 15,000 | 30,000 |
| Learning rate | 1e-3 | 1e-3 | 1e-3 |
| Trajectory length `K` | 6 | 8 | 8 |
| Mini-batch `B` | 512 | 1024 | 512 |
| Training trajectories | 40,960 | 81,920 | 108,403 |
| Testing trajectories | 8,192 | 8,192 | 36,230 |
| Validation fraction | 5% | 10% | 17% |

Compression ratios: 4→2, 8→4, 59→12. Note both toy systems are exactly 2× compression; only the G1
is aggressive (~4.9×).

Optimizer Adam, JAX + Flax. Per-feature normalization to zero mean / unit variance using
training-set statistics aggregated over all trajectories and steps, `x ← (x − µ)/σ`; the same
statistics are reused for validation, test and for scaling residuals at evaluation time.
**[CODE]** `std_floor = 1e-6`.

### 2.2 Joint training: YES

All three networks are trained **jointly, end to end, with a single total loss**, in one optimizer.
Paper: "The encoder, decoder, and latent dynamics networks are trained jointly using a weighted sum of
losses." **[CODE]** confirms: one `params` pytree, one `loss_fn`, one Adam update; there is **no
`stop_gradient` anywhere** in `utils/training.py`. Gradients from the dynamics and prediction losses
flow back into the encoder and decoder.

The stated objective the whole thing approximates, Eq. (9):

    inf_{φ,ψ,ρ}  Σ_{x_k ∈ D} ‖ f(x_k) − (D_ψ ∘ g_ρ ∘ E_φ)(x_k) ‖²

---

## 3. LATENT POINCARÉ MAP `g_ρ`

- **Input:** `z_k ∈ R^{n_z}` only.
- **Output:** `z_{k+1} ∈ R^{n_z}`.
- **Architecture:** 3 hidden layers, swish, width 64 (paddle-ball, hopper) or 128 (G1), linear output
  head of width `n_z`, wrapped in a residual skip: `z_{k+1} = z_k + ḡ_ρ(z_k)`. **[CODE]** the
  non-residual alternative is present but commented out.
- **No control input.** The RL policy is baked in; `g_ρ` models the closed-loop map only.
- **No exogenous input, no phase, no terrain, no command.**

### Is the inter-event time predicted? **NO.**

This is unambiguous. The event timestamps *are* extracted and *are* carried through the dataset, then
never fed to the model:

- **[CODE]** `parse_g1_23dof_data.py` saves `t_data` of shape `(B, 2, K)`.
- **[CODE]** `dataloader.py` loads it, slices channel 0, stores `self.t_data_train/_val`.
- **[CODE]** `t_data` is used **only** for the short-interval and chatter pruning filters and for a
  train/val overlap sanity check and plotting. It never enters `loss_fn`, and the model input is
  `x = concat(q, v)` with no time channel.

So variable step duration is **collected, used for data cleaning, and then ignored** in the model.
The latent map is a pure state-to-state map with implicit, unmodeled step duration. Given the
0.6 s clock this is a defensible approximation for their setup and an unmodeled error source in any
setup where the step period varies.

The paper never discusses this choice.

---

## 4. LOSS: all seven terms

Total, Eq. (10):

    L = L_rec,x + L_rec,z + L_fwd + L_bck + L_pred + L_iso + L_reg

Weights: **all λ = 1.0 except λ_reg = 1e-6** (paper, and confirmed in `scripts/train.py`:
`lambda_x=lambda_z=lambda_fwd=lambda_bck=lambda_pred=lambda_iso=1.0`, `lambda_reg=1e-6`).
`B` = mini-batch size, `K` = trajectory length, `W` = trainable weights excluding biases.
Table 1 of the paper:

**1. `L_rec,x(φ, ψ)` — state reconstruction.**

    (λ_rec,x / (B·K)) Σ_{b,k=1}^{B,K} ‖ x_k^(b) − (D_ψ ∘ E_φ)(x_k^(b)) ‖²

Encode-decode round trip in full-order space. This is the term the "Reconstruction error" plot
(Fig. 5 top, Eq. 13) measures.

**2. `L_rec,z(φ, ψ)` — latent reconstruction / cycle consistency.**

    (λ_rec,z / (B·K)) Σ_{b,k=1}^{B,K} ‖ E_φ(x_k^(b)) − (E_φ ∘ D_ψ ∘ E_φ)(x_k^(b)) ‖²

Round trip in the *other* direction: encoding a decoded latent must return the same latent. Pushes
`E ∘ D` toward identity on the image of the encoder. Note this is `E(D(E(x)))` vs `E(x)`, i.e. it is
evaluated only on latents reachable from data, not on arbitrary `z`.

**3. `L_fwd(φ, ρ)` — forward conjugacy.**

    (λ_fwd / (B·(K−1))) Σ_{b, k=1}^{B, K−1} ‖ E_φ(x_{k+1}^(b)) − (g_ρ ∘ E_φ)(x_k^(b)) ‖²

One latent step must match the encoding of the true next pre-impact state. Error measured **in latent
space**. Note this term is trivially minimizable by collapsing the latent space, which is why `L_iso`
exists.

**4. `L_bck(φ, ρ, ψ)` — backward conjugacy.**

    (λ_bck / (B·(K−1))) Σ_{b, k=1}^{B, K−1} ‖ x_{k+1}^(b) − (D_ψ ∘ g_ρ ∘ E_φ)(x_k^(b)) ‖²

The same one-step consistency but measured **in full-order space** after decoding. This is the direct
finite-sample surrogate for the stated objective Eq. (9).

**5. `L_pred(φ, ρ, ψ)` — multi-step prediction. (the one you asked about)**

    (λ_pred / (B·(K−1))) Σ_{b=1, k=2}^{B, K} ‖ x_k^(b) − D_ψ ∘ g_ρ^{(k−1)} ∘ E_φ(x_1^(b)) ‖²

- **How many steps:** the full training trajectory from a single encoded initial condition.
  `K − 1` steps: **5 steps** for the paddle-ball (`K=6`), **7 steps** for the hopper and the
  **G1** (`K=8`). Every intermediate step `k = 2..K` contributes a term; it is not just the endpoint.
- **Is it backpropagated through:** **Yes, fully.** **[CODE]** `prediction_loss_fn` encodes `x_0`
  once, then `jax.lax.scan`s `latent_step` for `T-1` iterations, concatenates, decodes all steps at
  once, and takes the squared error against `x[:, 1:, :]`. `lax.scan` is differentiable and there is
  no `stop_gradient` or truncation. This is genuine BPTT through 5 or 7 compositions of `g_ρ`, and
  the gradient reaches `E_φ` through the single initial encode and `D_ψ` through every decode.
- This is the term the paper credits for its flat multi-step error: "we attribute [minimal error
  propagation] to our loss choice penalizing discrepancy in long-horizon prediction, combined with the
  stability of the systems."

**6. `L_iso(φ)` — whitening / isometry.**

    λ_iso ‖ I − (1/(B·K)) Σ_{b,k} (E_φ(x_k^(b)) − µ)(E_φ(x_k^(b)) − µ)^T ‖_F²

Drives the **empirical latent covariance toward the identity**, computed per mini-batch with `µ` the
batch latent mean. Purpose: prevent latent collapse (which `L_fwd` otherwise rewards). Side effect
worth noting: it also fixes the latent scale, which is what makes the Lyapunov sublevel set in §5
dimensionally meaningful at all. **[CODE]** implemented exactly as written, `jnp.sum((C - I_z)**2)`,
no epsilon, no log-det variant.

**7. `L_reg(φ, ρ, ψ)` — L2 weight decay.**

    λ_reg Σ_{w_i ∈ W(φ,ψ,ρ)} ‖ w_i ‖²,  λ_reg = 1e-6

Kernels only, **biases excluded**. **[CODE]** an L1 variant exists but is commented out.

Naming note: the paper's Table 1 uses `λ_rec,x`/`λ_rec,z`/`λ_fwd`/`λ_bck`/`λ_pred`/`λ_iso`/`λ_reg`;
the code calls them `lambda_x`/`lambda_z`/`lambda_fwd`/`lambda_bck`/`lambda_pred`/`lambda_iso`/
`lambda_reg`. Same seven terms, same normalizers.

There is **no ablation** of any loss term anywhere in the paper.

---

## 5. REGION OF ATTRACTION

### 5.1 How the Lyapunov function is obtained

Not learned. It is **derived in closed form from the linearization of the learned latent map**:

1. Find the latent equilibrium `z*`. **[UNDETERMINED]**: the paper does not say how `z*` is computed
   (fixed-point iteration on `g_ρ`, mean of latent data, Newton solve?), and there is **no ROA code in
   the released repository** (the repo contains only data generation, parsing, the autoencoder and the
   trainer, no `roa`/`lyapunov` script).
2. Linearize: `Q = ∂g_ρ/∂z |_{z*}`, via `jax.jacfwd` (footnote 2). Shift so `z* = 0`, giving
   `z_{k+1} ≈ Q z_k`.
3. Solve the **discrete-time Lyapunov equation** `Qᵀ P Q − P + I = 0` for `P ≻ 0`, and take
   `V_z(z) = zᵀ P z`.
4. Define the set where `V_z` does not increase **under the nonlinear latent map**, Eq. (11):

       D = { z : ΔV_z(z) ≤ 0 } = { z : −zᵀz + g_ρ(z)ᵀ P g_ρ(z) − zᵀ Qᵀ P Q z ≤ 0 }

   (This is algebraically consistent: `ΔV = g_ρ(z)ᵀPg_ρ(z) − zᵀPz` with `zᵀPz` expanded using the
   Lyapunov equation as `zᵀQᵀPQz + zᵀz`.)
5. Largest sublevel set inside `D`, Eq. (12): `c* = inf_{z ∈ ∂D} zᵀ P z`, solved by **Monte Carlo
   sampling sweep**, not optimization. Footnote 3: "Direct optimization of c* is challenging as ∂D is
   non-convex and implicitly determined by a neural network. We thus use a simple but effective
   sampling-based sweep to obtain a conservative ROA estimate."
6. Latent ROA estimate `Ω_{c*} := { z : zᵀPz ≤ c* }`.

So: linear Lyapunov certificate, verified on the nonlinear latent map by sampling, level set found by
sampling. No SOS, no neural Lyapunov function, no formal verification anywhere.

### 5.2 The transfer claim

**Theoretical claim.** Theorem 5 (proved in Appendix B.4): under the setting of Lemma 3, with
`g̃ = g|_{E(M)}` and `f̃ = f|_M`, if `z*` is a locally asymptotically stable fixed point of `g̃` then
`x* = D(z*)` is a locally asymptotically stable fixed point of `f̃`, and if `z` is in the ROA of `z*`
for `g̃` then `D(z)` is in the ROA of `x*` for `f̃`.

**Be clear about what this proves.** Lemma 3 assumes an exact smooth invariant manifold `M` with
`f(M) = M`, and constructs `E`, `D` from the **Whitney embedding theorem** (hence `n_z ≤ 2·dim(M)`)
such that `D ∘ E = id` on `M` and `f = D ∘ g ∘ E` on `M`. Under those assumptions `Ẽ : M → E(M)` is
a **diffeomorphism**, `f̃ = D̃ ∘ g̃ ∘ Ẽ` is a **topological conjugacy**, and the theorem is the
standard "asymptotic stability is preserved by conjugacy" fact. It is true by construction and says
nothing about a trained network. There is no approximation bound, no robustness margin, no statement
of the form "if reconstruction error < ε then ...".

Two explicit disclaimers by the authors:
- Remark 6: "Here, stability is with respect to open subsets of `E(M)` and `M`, **not** `R^{n_z}` and
  `R^{n_x}`." So it is stability *on the manifold*, not in the ambient state space.
- §3 closing: to get ambient stability you need normal hyperbolicity / fast normal convergence, and
  "**Within the scope of this work, such convergence remains an assumption.**"

### 5.3 The 99.9 ± 0.1% vs 75.6 ± 10.1% numbers: exactly what was measured

Procedure (paper §5, second results paragraph):

- **Lyapunov arm:** sample **2,000 points inside `Ω_{c*}`** (the latent sublevel set), push each
  through the decoder `D_ψ` to get a full-order state, use each as an initial condition in parallel
  MuJoCo simulation, and record whether the rollout stays "stable".
- **Naive arm:** grid the latent space with **4 points per dimension**, over a hypercube whose side
  length equals the **major axis length of the Lyapunov region**, decode those points, roll them out
  the same way.
- **Stability criterion:** `p_z > p_z^thresh` never falling below a prescribed threshold over the
  course of "the 150-step trajectory".
- **Result:** the Lyapunov-set initial conditions are stable **99.9 ± 0.1%** of the time; the naive
  hypercube points are stable **75.6 ± 10.1%** on the G1.

So the metric is a **purity / precision rate over decoded latent samples**: "what fraction of the
initial conditions I proposed did not fall over".

What this measurement does **not** establish, and this is the important part:

- **No recall, no volume, no coverage.** There is no comparison against the true ROA of the FOM, no
  volume ratio, no measure of how many genuinely stable states the Lyapunov set misses. A
  sufficiently small ball around `x*` scores 100% trivially. The authors concede the direction:
  "while the Lyapunov estimate is more conservative, it is far more accurate than the naive
  sampling-based estimate". More conservative + higher purity is exactly what you get for free by
  shrinking the set. The 75.6% naive number is the only thing making 99.9% look meaningful, and the
  naive set is by construction a *superset*-ish box that includes obviously bad points.
- **Every tested state is in the image of the decoder.** Both arms decode latent samples. Nothing
  probes full-order states off the decoder's image, which is the entire question of whether latent
  stability says anything about the 59-dimensional system.
- **[UNDETERMINED]** what the `±` is over. Seeds? The three systems? Repeated sampling draws?
  Not stated. The 99.9 ± 0.1% is phrased as if it holds across systems; the 75.6 ± 10.1% is
  explicitly "on the G1" and introduced with "for instance", so the naive rates for the other two
  systems are not reported.
- **[UNDETERMINED]** whether "the 150-step trajectory" means 150 Poincaré steps (~90 s at the 0.6 s
  gait period) or 150 control steps (3 s at 50 Hz). Given the 7.5 s data rollouts, 3 s is more
  plausible, but the paper does not say and the code is not released.
- **Arithmetic problem with the naive arm:** 4 points per dimension in `n_z = 12` is `4^12 ≈ 1.68e7`
  decoded initial conditions to simulate for the G1. The paper says "we roll out all of these samples
  in parallel simulation". Possible on GPU but implausible as described, and unaddressed. For the
  hopper it is `4^4 = 256` and paddle-ball `4^2 = 16`, so the same sentence describes wildly different
  experiment sizes. Flagging as a likely misstatement.

Figures 6 (G1) and 7 (hopper) show blue "Sampled ROA" clouds with a small dense red "Lyap. ROA" blob
inside. Visually the Lyapunov set on the G1 is a very small fraction of the sampled-stable region
(order a few percent by eye in the `x`, `y`, `z`, `φ`, `θ`, `ψ` projections of stance-foot-relative
COM position). That is the conservatism, visible.

---

## 6. RESULTS

### 6.1 The rollout test (Figure 5)

Metrics, Eqs. (13) and (14), both on **normalized** (z-scored) states and, per Appendix C.2,
"normalize[d] by state dimension when computing the per-step RMS error across state components", so
the y axis is a dimensionless per-state-component RMS residual:

    ℓ_rec = ‖ x_k − D_ψ ∘ E_φ(x_k) ‖              (single-step, encode-decode only)
    ℓ_dyn = ‖ x_k − D_ψ ∘ g_ρ^{(k)} ∘ E_φ(x_0) ‖  (multi-step, from x_0 only)

- **Horizon:** the plot runs `k = 0 … 6`, i.e. 7 Poincaré indices, so **6 propagated steps** from
  `x_0`. For the G1 that is 6 strides ≈ 3.6 s of walking. Not longer.
- **Sample count:** the caption says "over 3,000 test trajectories for each system". **This conflicts
  with Table 6**, which lists 8,192 / 8,192 / 36,230 testing trajectories. **[UNDETERMINED]** which is
  right; possibly a subsample was used for the figure. Flagging the inconsistency.
- **Actual numbers:** there is **no results table anywhere in the paper**. The only numbers are the
  curves. Read off the figure at 400 dpi:

  | | k=0 | k=1 | k=2 | k=6 |
  |---|---|---|---|---|
  | Recon, paddle-ball | ~0.25 (σ band ~±0.37) | ~0.03 | ~0.02 | ~0.02 |
  | Recon, hopper | ~0.11 | ~0.02 | ~0.01 | ~0.01 |
  | Recon, G1 | ~0.22 (band to ~0.62) | ~0.13 | ~0.08 | ~0.02 (band ~±0.15) |
  | Fwd prop, paddle-ball | ~0.25 | ~0.09 | ~0.03 | ~0.02 |
  | Fwd prop, hopper | ~0.11 | ~0.06 | ~0.03 | ~0.01 |
  | **Fwd prop, G1** | ~0.24 | ~0.28 | **~0.30 peak (band to ~0.96)** | ~0.04 |

- **The error decreases with k.** The paper reads this as a success: "We observe good performance of
  the reconstruction and latent dynamics, improving over the rollout as the system stabilizes... error
  propagation is minimal despite the length of the rollout, which we attribute to our loss choice ...
  combined with the stability of the systems."

  **Read it skeptically instead.** All trajectories start from randomized velocities around a single
  fixed pose and converge to the same clock-driven limit cycle. By `k = 6` the test distribution has
  collapsed onto the fixed point, so *any* model that outputs the fixed point scores near zero. The
  metric is at its most informative at `k = 0-2`, where the G1's forward-propagation error is
  ~0.24-0.30 in units of per-component standard deviations, with a ±σ band reaching ~0.96. That is
  a large error, not a small one: at k=2 the model's error is roughly 30% of the full spread of the
  data. The decreasing curve is largely a statement about the data, not the model.
- The ±σ bands are drawn extending **below zero** on a quantity that is a norm, so the per-step error
  distribution is heavily right-skewed and σ is not a meaningful summary.

### 6.2 The hardware G1 demonstration (Figure 8)

**What was demonstrated:** a single 9-step (k = 0…9, so 10 pre-impact indices) rollout on a real,
tethered Unitree G1, comparing three curves on three scalar channels, `p_x` [m], `p_z` [m], and
`θ` [rad]:

- **True** (hardware measurement at pre-impact),
- **Recon** (`D_ψ ∘ E_φ` of the true state at each step, i.e. autoencoder only, no dynamics),
- **Latent** (`D_ψ ∘ g_ρ^{(k)} ∘ E_φ(x_0)`, i.e. the ROM rolled out from the first step only).

Four photographs at k = 0, 3, 6, 9 show the robot walking with a safety tether.

**What the figure actually shows, read carefully:**

- The **Latent** curve is essentially flat from k ≈ 2 onward on all three channels. `p_x` sits at
  ~0.02 m while the true signal oscillates between about −0.07 and +0.10 m. `p_z` sits at ~0.767 m
  while the true signal oscillates between ~0.745 and ~0.777 m. `θ` sits at ~0.002 rad while the true
  signal oscillates between about −0.015 and +0.047 rad.
  **The ROM converges to its fixed point and predicts the mean, not the step-to-step trajectory.**
- The **Recon** curve (no dynamics at all, just encode-decode) is visibly wrong on `p_z`: at k=2 it
  dips to ~0.732 m when the truth is ~0.747 m, an error of ~15 mm on a signal whose entire dynamic
  range is ~30 mm. Autoencoder reconstruction error on hardware data is comparable to the signal.
- There is **no quantitative hardware metric**: no RMSE, no percentage, no comparison to any baseline.
  The paper's only sentence about it is: "In Figure 8, we demonstrate the prediction performance of
  the ROM relative to real-world Unitree G1 experiments."

**[UNDETERMINED]** for the hardware experiment: how the pre-impact state was detected on hardware
(force sensing? kinematic? Remark 7 lists options but does not say which was used), how the 59-dim
state was estimated (base pose and velocity require a state estimator), whether the model was the
sim-trained one applied zero-shot or retrained/fine-tuned on hardware data, how many hardware trials
were run, and what the terrain/velocity command was. **No hardware code exists in the repository.**

### 6.3 What is missing from the results

- No baseline. No LIP, no SLIP, no PCA/DMD, no linear autoencoder, no fixed-`Δt` (non-Poincaré) latent
  model. Nothing to show the Poincaré indexing helps.
- No latent-dimension sweep. `n_z = 2/4/12` are asserted; §3's linear/manifold theory is offered as
  "practical guidance for selecting the latent dimension" but is never actually used to pick 12.
- No loss ablation.
- No seed variance on the model itself.
- No table of numbers anywhere.

---

## 7. STATED LIMITATIONS

**There is no limitations section and no future-work section.** The conclusion is five sentences and
is purely positive. `grep -i "limitation|future work|caveat"` over the full text returns only
references to *other* work's limitations. The following are the only acknowledgments of scope, and
they are scattered inline:

1. **Normal convergence is assumed, not shown.** §3: "Informally, stability on the invariant manifold
   plus sufficiently fast convergence normal to the invariant manifold implies stability of the full
   system. **Within the scope of this work, such convergence remains an assumption.**" This is the
   load-bearing gap: without it, Theorem 5 gives stability *on the manifold* only.
2. **Stability is on the manifold, not in the ambient space.** Remark 6: "stability is with respect to
   open subsets of `E(M)` and `M`, not `R^{n_z}` and `R^{n_x}`."
3. **The ROA level set is found by sampling, and is deliberately conservative.** Footnote 3: "Direct
   optimization of `c*` is challenging as `∂D` is non-convex and implicitly determined by a neural
   network. We thus use a simple but effective sampling-based sweep to obtain a **conservative** ROA
   estimate." And in §5: "the Lyapunov estimate is more conservative".
4. **Perfect compression is unattainable.** §2: "For complex, real-world systems, perfect compression
   in this sense is unrealistic. Likewise, one cannot analytically determine optimal encoders,
   decoders, and ROMs."
5. **Faulty impacts are discarded rather than modeled.** §4: "After pruning faulty Poincaré data, such
   as impacts with sliding contact or contact chatter, we obtain clean sequences of Poincaré returns."
6. **Impact detection used privileged simulator sensing.** Remark 7 exists precisely to hand-wave this:
   "In practice, impacts can be estimated without dedicated contact sensors. Kinematic methods use foot
   height and velocity; dynamic methods use changes in generalized velocities, accelerations, or
   momentum; force methods use joint torques and ground reaction force estimates." They did not do
   any of this; they used MuJoCo touch sensors thresholded at 1e-6.
7. **The Poincaré map is a partial function** (Appendix A), acknowledged with "an abuse of notation".
8. **[CODE, unacknowledged]** the single-channel selection with the comment "just choose the default
   first one for now".

Unstated but material limitations, for your purposes: single flat terrain, single fixed velocity
command (0.5 m/s), single clocked gait period (0.6 s), initial-condition randomization on velocities
only, no external disturbances, no terrain variation, no sim-to-real quantification, no control input
in the ROM (so it cannot be used for planning or control synthesis as-is), and no comparison of the
learned `g_ρ` against a numerically computed Poincaré map of the actual simulator.

---

## 8. ASSESSMENT: Unitree Go2, granular terrain, 50 Hz, trot

Short answer: **the natural section exists and is easy to define, but I do not think an
event-indexed deterministic Poincaré formulation is viable on granular terrain as HALO formulates it.
The blocker is not the asynchrony. It is that on deformable terrain the impact-to-impact map is not a
function of the pre-impact robot state, which is the one assumption the entire construction rests on.**
Asynchrony is a real but second-order problem, and HALO gives you no help with it because it does
not solve it either.

### 8.1 The natural section

For a trot with a clocked policy, the direct transplant is: **designate one foot, say front-left, and
take its touchdown as the section**. In a nominal trot FL and RR form a diagonal pair and land
together; FR/RL land a half-period later. FL touchdown to FL touchdown is exactly one full gait cycle,
which is the same structure HALO uses on the G1 (left-foot to left-foot = one stride = two steps).
Go2 dimensions: `n_q = 19` (3 + 4 quaternion + 12 joints), `n_v = 18`, so `n_x = 37`, well under the
G1's 59. Latent `n_z` in the 8-12 range would be the analogous guess. State expressed relative to a
yaw-aligned frame on the diagonal stance foot (RR at FL touchdown), by direct analogy.

That much is mechanical and would work on flat ground. Everything below is why I would not bet the
project on it for sand.

### 8.2 What breaks, in order of severity

**(1) The map is not a function of the robot state. This is fatal, not fixable by architecture.**

HALO's whole object is `x_{k+1} = f(x_k)`, a deterministic autonomous map. On granular terrain the
reset map `Δ` depends on the terrain: sinkage depth, local packing density, whether this patch was
already disturbed by a previous footfall, moisture. So the true relation is
`x_{k+1} = f(x_k, terrain_state_k)` with `terrain_state` itself having memory (the robot compacts what
it walks on, and a trot re-loads nearby ground every cycle). Regress `x_{k+1}` on `x_k` alone and you
learn a conditional mean; the residual is the terrain interaction, which is precisely the thing you
presumably care about. A 12-dimensional deterministic latent map cannot represent it, no matter how
you train it.

Concretely: two identical pre-impact states landing on loose vs packed sand give measurably different
next states (different sinkage, different effective restitution, different slip). Your `g_ρ` will
average them. You will then see exactly HALO's Figure 8 pathology, and worse: a latent rollout that
sits at the mean while the truth oscillates.

The honest reformulation, if you want to keep the framing, is a **conditional/stochastic map**:
`p(z_{k+1} | z_k, terrain descriptor)`, with the terrain descriptor as an explicit conditioning input
(measured sinkage, contact force integral over stance, proprioceptive terrain estimate). That is a
different, much larger project than HALO, and it breaks the Lyapunov analysis in §5 immediately,
because `ΔV ≤ 0` is not a meaningful set condition for a stochastic map.

**(2) The event itself is not well-defined on granular media.**

`S = {s(x) = 0, ṡ(x) < 0}` presumes an instantaneous, geometric switching surface and an impulsive
reset. On sand, touchdown is a 20-80 ms penetration process, not an event. Whatever detection rule you
pick (force threshold, foot-height threshold, contact-flag) fires at a time that depends on the
material: a fixed force threshold triggers systematically later on loose sand than on packed sand, so
your "section" moves with the terrain and you are comparing states sampled at different phases across
your dataset. HALO's threshold was `force > 1e-6` in a rigid-contact simulator, which is not a choice
you can transplant. Also `Δ(S) ∩ S = ∅` (required by their Theorem 8) is questionable when the foot
sinks, rebounds and re-loads.

**(3) 50 Hz event detection is a real, quantifiable error source.**

HALO detects the rising edge at **1 kHz sim dt**, then takes the last sample before contact with no
interpolation. At 50 Hz your event-time quantization is up to 20 ms. A Go2 trot at ~2.5 Hz stride has
a ~400 ms cycle, so 20 ms is ~5% of a cycle. Worse, you are sampling at the instant of maximum state
derivative: pre-touchdown vertical body velocity is order 0.3-0.8 m/s and body pitch rate is at its
peak, so 20 ms of jitter converts directly into roughly 6-16 mm of body-height error and comparable
velocity error, injected as irreducible noise into a map whose signal (per Figure 8's `p_z` channel on
the G1) is ~30 mm peak-to-peak. That noise floor is a substantial fraction of the signal before you
have modeled anything.

Mitigation exists but is partial: interpolate the crossing by fitting foot height or contact force
across the two bracketing samples and propagating the pre-impact state forward by the sub-sample
offset using the continuous dynamics. You can only do this on the pre-impact side, and it costs you a
dynamics model at exactly the moment the dynamics are stiffest. If you go this route, log contact at a
higher rate than 50 Hz even if the policy runs at 50 Hz; the Go2 gives you foot force estimates faster
than the policy rate, and HALO's own pipeline logged contact at 20× the control rate for exactly this
reason.

**(4) Asynchrony: the real problem is not the nominal trot, it is the off-nominal one.**

On flat ground with a clocked trot, the diagonal pair is synchronous and the single-channel trick
works fine. On granular terrain it stops being true, and this is where the design fails quietly rather
than loudly:

- The diagonal pair de-synchronizes. On uneven or deformable ground the leading foot of a diagonal
  pair can land tens of milliseconds before its partner. Whichever leg you designate as the section,
  the configuration of the other three legs at that instant is no longer consistent across samples, so
  states you are treating as lying on the same section are not.
- **Missed and spurious events desynchronize the step index.** If the designated foot steps into a
  depression, or the force never crosses threshold, you lose a cycle and every subsequent `k` in that
  trajectory is off by one. HALO handles this by **discarding the entire trajectory** (their chatter
  filter: two consecutive inter-event intervals under 0.1 s → drop; plus a short-last-interval filter
  and a fallen filter). That is affordable when you generate 122,880 trajectories in a GPU loop and
  can throw away 12%. On a physical Go2 collecting granular-terrain data, foot bounce and chatter on
  gravel are the norm, not the exception, and you cannot afford to delete most of your hardware data.
  You would need to prune per-event rather than per-trajectory, which HALO's fixed-`K` tensor layout
  does not support.
- HALO offers **zero** methodological content on this. The relevant line in their code is literally
  `# choose the contact channel to use, NOTE: just choose the default first one for now`. If you cite
  HALO as precedent for handling asynchronous multi-leg events, you are citing something that does not
  exist.

**(5) The clocked-gait subtlety cuts against the whole idea.**

HALO's G1 policy is driven by an exogenous clock (`phase = (t mod 0.6)/0.6`, fed in as sin/cos) and
also consumes its own previous action. So the closed-loop system is non-autonomous with controller
memory: the true state is `(x, phase, a_{k-1})`, but the learned map takes only `x = (q, v)`. This is
only self-consistent because touchdowns happen at nearly constant clock phase, which in turn means
their "Poincaré map" is approximately a fixed-Δt = 0.6 s flow map. You could reproduce it with a
fixed-time latent model sampled at the right phase and skip contact detection entirely.

Now apply that to your Go2. If your trot is clock-driven (it almost certainly is), the same reduction
holds on flat ground, and the event indexing buys you nothing over a fixed-Δt model at matched phase.
The moment the terrain makes actual touchdown drift away from the commanded clock phase, event
indexing *does* become different from time indexing, and that is exactly the same moment when the map
stops being a function of `x` (point 1) and the event stops being well-defined (point 2).
**Event indexing is only interesting in the regime where it is also least valid.** That is the core
structural objection, and I do not see a way around it within HALO's formulation.

**(6) The ROA half does not transfer at all.**

HALO's ROA evidence requires resetting the simulator to a decoded full-order state and rolling out.
On granular terrain "the state" includes the terrain, so a decoded 37-vector does not specify an
initial condition; you cannot reset to it, in sim or on hardware. Even setting that aside, the
99.9% vs 75.6% comparison measures purity of decoded samples with no coverage measure, and the
Lyapunov set is visibly a small blob inside the sampled-stable cloud (Figs. 6-7). There is no evidence
that the decoded sublevel set is a *useful* ROA estimate, only that it is a *safe* one, which a small
ball also is. I would not present this result as validated stability transfer.

### 8.3 What I would actually do

**A one-day, no-autoencoder experiment that decides this.** Before building anything:

1. Log Go2 trot data on your granular terrain, several hundred trajectories, contact logged as fast as
   the hardware allows.
2. Detect FL touchdown, extract pre-impact `x_k ∈ R^{37}` in the RR-stance-foot yaw frame, exactly as
   HALO does.
3. **Measure how much of the step-to-step variation is predictable from `x_k` alone.** Fit the
   cheapest possible predictors (ridge regression, then k-NN, then a small MLP) for `x_{k+1}` given
   `x_k` and report per-component `R²` against the naive baseline "predict the dataset mean".
   Also report the same thing on flat ground as a control.

If `R²` over the mean-predictor baseline is small on sand and large on flat ground, the deterministic
event-indexed map is dead in your setting, and you have learned that for the cost of a regression fit
rather than a model architecture. If `R²` is respectable, then the latent/Lyapunov machinery becomes
worth building and you have a quantitative floor to beat.

**Two diagnostics to run alongside it**, both cheap and both things HALO never reports:
- Histogram the inter-event interval for the designated foot on sand vs flat. If the spread is more
  than a few percent of the cycle, you must model `Δt_k`, which HALO does not, and you will need to
  add it as a predicted output of `g_ρ`.
- Histogram the offset between the two diagonal-pair touchdowns. That number tells you directly
  whether single-channel sectioning is defensible for your gait and terrain, and it is the number
  HALO would have had to report to claim anything about asynchrony.

**If you proceed anyway**, three deltas from HALO that I would treat as mandatory, not optional:
predict `Δt_k` as an extra output of `g_ρ`; condition `g_ρ` on a terrain descriptor and accept that
the Lyapunov analysis is then out of scope; and prune per-event with an explicit "missed event" gap
marker rather than dropping whole trajectories.

**Bottom line.** The natural section is FL touchdown and it is easy to build. The asynchrony question
you flagged as the crux is real, unsolved by this paper, and manageable in the nominal case. But the
thing that actually kills it on granular terrain is upstream of asynchrony: `x_{k+1} = f(x_k)` is not
true when the terrain is a state with memory, and the 50 Hz sampling puts a noise floor on the section
state that is a large fraction of the signal HALO itself shows on hardware. I would rate this the
highest-risk item on the list and would want the `R²` experiment above before any further investment.
