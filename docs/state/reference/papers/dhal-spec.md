# DHAL technical spec

**Paper:** *Discrete-Time Hybrid Automata Learning: Legged Locomotion Meets Skateboarding*,
Hang Liu, Sangli Teng, Ben Liu, Wei Zhang, Maani Ghaffari (UMich CURLY + SUSTech).
arXiv:2503.01842v2 [cs.RO], 6 Apr 2025 (v1: 3 Mar 2025). 16 pages, RSS-format.

**Sources used**
- PDF: `/private/tmp/claude-501/-Users-kylesha/e7393dd9-22e3-4b6c-b121-0d82bc75fba8/scratchpad/papers/dhal.pdf`
- Extracted text: `.../papers/dhal_raw.txt`, `.../papers/dhal.txt`
- **Official code, cloned and read:** `https://github.com/UMich-CURLY/DHAL` →
  `/private/tmp/claude-501/-Users-kylesha/e7393dd9-22e3-4b6c-b121-0d82bc75fba8/scratchpad/dhal_code`
  (single commit `62a08b2`). This resolves several ambiguities and contradicts the paper in
  three places; both are reported below.
- alphaxiv page: adds nothing beyond the paper (checked; its "prevents mode collapse" gloss is
  a paraphrase of the paper's own claim, not independent analysis).

**Legend for provenance tags**
`[P]` = stated in the paper. `[C]` = read from the released code. `[M]` = my own measurement
off a figure (pixel extraction; error bars given). `[?]` = could not determine.

**Note on your framing of the K ablation:** you referred to "Figure 2". In v2 the K ablation
is **Figure 7** (p. 8); Figure 2 is a photograph of a pothole recovery. I did not check the v1
PDF's figure numbering, so if you were reading v1 the number may differ there. Content-wise
Fig. 7 is the ablation you described.

---

## 0. One-paragraph summary of the actual mechanism

DHAL is a **hard-gated mixture of K dynamics autoencoders**, bolted onto a PPO locomotion
policy as a representation learner. A single MLP ("Discrete Hybrid Automata", DHA) reads a
flattened 20-step proprioceptive history and emits a softmax over K=3 modes. A **one-hot is
drawn by plain categorical sampling** (not Gumbel) and passed through a **straight-through
estimator**. That one-hot selects which of K *fully separate* β-VAEs encodes the same history
into a 20-d latent `z`, and which decoder predicts next observation + next foot contacts.
The **gate is trained only by the per-expert reconstruction loss** — it is explicitly detached
from the PPO gradient — plus a term that *minimizes* the softmax entropy. The actor sees only
`z` and the current proprio; it never sees the mode one-hot. There is no learned transition
function, no guard, no dependence on the previous mode: despite the name "automata", the mode
is a memoryless per-step classifier over a sliding window.

---

## 1. MODE LATENT — exact mechanism

### 1.1 Input window

`[P]` Eq. (3)/(4): the DHA takes `(o_{t-k:t}, a_{t-k-1:t-1})`.

`[C]` Concretely (`go1_config.py:37-46`, `legged_robot.py:348-404`, `821`):

| quantity | value |
|---|---|
| `history_len` | **20** control steps |
| `n_proprio` (per step) | **48** |
| DHA / VAE input | flat vector of **20 × 48 = 960** floats |
| control rate | `dt=0.005`, `decimation=4` → **50 Hz**, so the window is **0.4 s** |

The 48-d per-step proprio vector is
`[roll, pitch (2)] ‖ [base ang.vel (3)] ‖ [projected gravity (3)] ‖ [cmd vx,vy,wz (3)] ‖
[dof_pos − default (12)] ‖ [dof_vel (12)] ‖ [last action (12)] ‖ [phase (1)]`
= 2+3+3+3+12+12+12+1 = 48.

Two things follow that the paper does not spell out:
- **Actions are inside the observation vector**, so "o and a history" is one 960-d tensor, not
  two streams. The buffer is 19 past steps (`obs_history_buf`, shape `(N, history_len-1, 48)`)
  concatenated with the current *noise-corrupted* observation.
- **The input contains an explicit sinusoidal gait clock** (`phase`, `_get_phase()`:
  `sin(2π · t·dt / cycle_time)`, `cycle_time = 4 s`). This matters a lot — see §5.3 and §8.

### 1.2 Network producing the mode distribution

`[P]` Table V: "DHA Architecture: MLP, DHA Hidden Dims [256, 64, 32]".
`[C]` `actor_critic_hds.py:130-142`:

```
DHA:  Linear(960, 256) → ELU → Linear(256, 64) → ELU → Linear(64, 32) → ELU
      → Linear(32, K=3) → StraightThroughSoftmax
```
≈ 264 k parameters, dominated by the 960×256 first layer. It is an MLP over the **flattened**
history — no convolution, no recurrence, no attention.

### 1.3 Softmax over K modes — yes

`[P]` Eq. (4): `p = f_softmax ∘ f_DHA_logit(o_{t-k:t}, a_{t-k-1:t-1}; θ_DHA)`.
`[C]` `F.softmax(logits, dim=-1)` over `K = num_modes = 3` (`go1_config.py:472`).

### 1.4 How the one-hot is sampled — **plain categorical + straight-through, NOT Gumbel**

`[P]` Eq. (5): `δ ~ Categorical(p)`, `Σδ_i = 1`, `δ_i ∈ {0,1}`; "the discrete categorical
sample results in discontinuity of gradients, we apply the straight-through-gradient method
[Bengio et al. 2013]".

`[C]` `actor_critic_hds.py:270-277` — this is the whole of it:

```python
class StraightThroughSoftmax(nn.Module):
    def forward(self, logits):
        probs = F.softmax(logits, dim=-1)
        sampled_index = torch.multinomial(probs, 1)
        one_hot = torch.zeros_like(probs).scatter_(1, sampled_index, 1)
        return (one_hot - probs).detach() + probs, probs
```

So: **no Gumbel noise, no temperature, no annealing schedule.** Forward value is exactly the
one-hot; backward gradient is the identity w.r.t. `probs` (the classic biased ST estimator).
There is no `self.training` branch — **the multinomial sample is drawn at inference too**
(see §6).

### 1.5 Is it differentiable end to end? Partially, and deliberately so

- ST makes the discrete choice differentiable in the biased sense. `[P]`+`[C]`
- **But the gate is cut off from the RL objective on purpose.** `[P]` Appendix C: "the DHA is
  decoupled from both the encoder and the actor when PPO loss propagation and is only updated
  using the dynamics loss." `[C]` `actor_critic_hds.py:196`: `mode_latent = mode_latent.detach()`
  inside `update_distribution`, and the PPO-side loss therefore cannot reach `θ_DHA`. In
  `ppo_mlp_hds.py:159` the DHA is called a *second*, undetached time solely for the VAE loss.
- The **VAE encoders do** receive gradient from both PPO (through `z` into the actor) and the
  reconstruction loss. `[P]` "the gradient backpropagates through the encoder to extract useful
  temporal features."

Net: the gate is trained by *prediction error only*; the encoders are trained by prediction
error *and* control performance; the actor by control performance only.

---

## 2. MODE-SPECIFIC DYNAMICS — K fully separate networks (not heads, not a hypernet)

`[P]` "With the hybrid automata, we build **K corresponding dynamics β-VAEs** for K modes."
Fig. 3(a) shows K independent encoder/decoder pairs, one highlighted ("Activate"), the others
greyed as "Non-activated Dynamics Modules".

`[C]` `actor_critic_hds.py:145-149`:
```python
self.TsDyn_modules = nn.ModuleList()
for i in range(num_modes):
    self.TsDyn_modules.append(VAE(960, 33, [256,128,64], 20, 20, kl_w=0.9, prior_mu=0))
```
**Zero parameter sharing between modes.** Each mode gets its own encoder, its own μ/logvar
heads, its own observation decoder and its own contact decoder.

### 2.1 Per-mode module, exact shapes `[C]` (`vae.py`, `cnn1d.py`)

**Encoder — a 1-D CNN, not a transformer.** `[P]` §IV-A text says "For the encoder, we adopt
a **transformer** architecture due to its superior ability to capture temporal dependencies."
`[P]` Table V says "VAE Encoder Architecture: **1-D CNN**". `[C]` The code is a 1-D CNN
matching Table V exactly. **The word "transformer" in the body text is wrong / stale.**

```
input  (B, 20, 48)
per-step projection:  Linear(48 → 30) + ReLU          # 3*channel_size, channel_size=10
transpose → (B, 30, 20)
Conv1d(30 → 20, k=6, s=2) + ReLU                       # T: 20 → 8
Conv1d(20 → 10, k=4, s=2) + ReLU                       # T: 8 → 3
Flatten → 30
Linear(30 → 64) + ReLU                                 # 64 = hidden_dims[-1]
fc_mu:     Linear(64 → 20)
fc_logvar: Linear(64 → 20)
reparameterize → z ∈ R^20
```
(Table V's "Input channel = [30, 20], Kernel=(6,4), Stride=(2,2)" matches this exactly.)

**Observation decoder** (predicts `ô_{t+1}`, 33-d):
`Linear(20→64) ReLU Linear(64→128) ReLU Linear(128→256) ReLU Linear(256→33)`

**Contact decoder** (predicts `ĉ_{t+1}`), a *separate* MLP of identical shape ending in
`Linear(256 → 3) → Sigmoid`.

### 2.2 What the decoder predicts

- `n_recon_num = 33` `[C]`: `[roll,pitch (2)] ‖ [ang.vel (3)] ‖ [gravity (3)] ‖ [dof_pos−default (12)]
  ‖ [dof_vel (12)] ‖ [phase (1)]`. **It predicts the absolute next observation, not a delta**,
  and the target is the *clean* (un-noised) `current_obs_buf` at t+1 while the input history is
  noise-corrupted. Commands and actions are excluded from the target.
- **Contact target is 3-dimensional, not 4.** `[C]` `legged_robot.py:157` returns
  `self.contact_filt[:, [0,2,3]]`. Unitree body order is FR, FL, RR, RL, so index 1 = **FL**,
  the foot rigidly attached to the skateboard by a spherical joint, is dropped. The predicted
  contacts are the three free feet. `[P]` frames it generically as "the probability that each
  leg is in contact at time t+1".
- `[P]` explicit rationale: "We include contact information to make learning hybrid switching
  easier for the networks." **Contact is a supervision target for the dynamics head, never an
  input to anything.**

### 2.3 Gating at forward time

`[C]` `actor_critic_hds.py:198-205`. All K encoders are run every step; the results are stacked
and combined by `torch.bmm(one_hot.unsqueeze(1), representation)` — i.e. the one-hot **selects**
one. So the "non-activated modules" in Fig. 3 are conceptually inactive but **computationally
all K still run every step**. Cost scales linearly in K. (This is a straightforward thing to
optimize with a real gather; they did not.)

### 2.4 The actor

`[C]` input = `concat(current 48-d proprio, z ∈ R^20)` = 68-d →
`Linear(68,512) ELU Linear(512,256) ELU Linear(256,128) ELU Linear(128,24) SoftplusWithOffset`,
producing (α, β) for 12 joints. **The actor never sees the mode one-hot or `p`** — the mode
reaches the policy *only* through which expert produced `z`.

Critics: 4 MLPs `[512,256,128] → 1` over the 1630-d privileged observation
(a legacy unused `critic`, plus glide / push / sim2real).

### 2.5 There is no transition/guard function

`[P]` §III-B: "we **omit the explicit jump mapping** between different modes, as it is captured
within the discrete-time dynamics." The DHA is memoryless given the window — it has no input
for the previous mode and no learned transition matrix. Fig. 3(a)'s P1→P2→P3 cycle diagram is
illustrative only; nothing in the model enforces or learns edges.

---

## 3. TRAINING — loss and supervision

### 3.1 Mode supervision: **fully unsupervised**. This is the paper's central claim.

`[P]` "the mode label is hard to obtain even in simulation. To address this ... we utilize the
unsupervised learning method to train the mode selector. We train the mode selector and the
β-VAE simultaneously ..., where the mode is self-determined by constructing the loss function
... The hybrid automata is trained by minimizing the prediction error of `o_{t+1}, c_{t+1}`,
where the correct mode label will result in a lower prediction error."

No contact labels, no phase labels, no segmentation, no event function are used to supervise
the *mode*. Contact **is** ground-truth-supervised, but only as an auxiliary *output* of the
per-mode decoder.

### 3.2 Losses

`[P]` Eq. (8): `L_vae = Σ_t MSE(ô_{t+1}, o_{t+1}) + BCE(ĉ_{t+1}, c_{t+1}) + β·L_KL`
`[P]` Eq. (9): `L_DHA = L_vae + H(p)`

`[C]` `ppo_mlp_hds.py:159-193, 246-247` — the real thing:

```python
mode_latent, prob = self.actor_critic.DHA(obs_batch)          # NOT detached here
DHA_entropy = -(prob * torch.log(prob + 1e-8)).sum(-1).mean() # H(p)

for i, sub_net in enumerate(self.actor_critic.TsDyn_modules):  # all K experts
    recon_x, recon_contact, mu, logvar = sub_net.forward(obs_batch)
    recon_loss_list.append( F.mse_loss(recon_x, obs_future_batch, reduction='none').sum(1) )
    recon_contact_loss_list.append( F.binary_cross_entropy(recon_contact,
                                     contact_future_batch, reduction='none').sum(1) )

recon_loss         = bmm(mode_latent, stack(recon_loss_list)).mean()      # select active expert
recon_contact_loss = bmm(mode_latent, stack(recon_contact_loss_list)).mean()
kl_div = (-0.5*sum(1+logvar-mu**2-logvar.exp(), -1) * mode_latent).mean()

vae_loss = recon_loss + 0.8*recon_contact_loss + 0.9*kl_div
loss = surrogate_loss + 1.0*value_loss - 0.01*policy_entropy + vae_loss + 0.01*DHA_entropy
```

Points that only the code makes clear:

1. **All K experts are evaluated, but only the selected expert's loss is counted.** The
   straight-through one-hot means `d(recon)/d p_i = L_i`, so the gate descends toward whichever
   expert predicts best on that sample. This is a **winner-take-all mixture-of-experts
   assignment**, exactly like hard-EM / competing-experts, with the gate trained on the vector
   of per-expert losses.
2. **`+0.01 · H(p)` is added to the loss ⇒ the entropy of the mode posterior is MINIMIZED.**
   The coefficient is `entropy_coef = 0.01`, reused from PPO. Note the sign asymmetry in the
   same line: `−0.01·H(π)` (maximize policy entropy) but `+0.01·H(p)` (minimize mode entropy).
3. Loss weights in code: MSE ×1.0, contact BCE ×**0.8**, KL ×**0.9**. `[P]` Table V says
   "VAE KL Divergence Weight (β) = **1e-2**" and Eq. (8) shows no weight on BCE.
   **The paper's β and the code's KL weight disagree by ~90×.** (Also, because the KL tensor
   is masked by the one-hot and then averaged over both batch and K, the *effective* KL weight
   is 0.9/K = 0.3.) The `kl_w=0.9` argument passed into `VAE.__init__` is stored and never used;
   the real weight is the hard-coded 0.9 in the PPO loss.
4. Reconstruction MSE is **summed over the 33 output dims** (`.sum(dim=1)`), then averaged over
   the batch. This is the y-axis of Fig. 7, so a plotted value of 0.55 is ≈0.017 per dimension.
5. One optimizer (`Adam`, lr 2e-4 adaptive-KL schedule) over *all* parameters; DHA isolation is
   achieved purely by the `.detach()` in the actor path, not by a separate optimizer.

### 3.3 What actually prevents mode collapse — **and my answer is: essentially nothing explicit**

This is the most important thing to be precise about, because the paper's own wording is
misleading and alphaxiv's summary repeats it.

`[P]` "we encourage the mode to be distinguished by **minimizing the information entropy** of
the mode probability p ... aiming to ensure that the modes' probabilities are as distinct as
possible and to prevent confusion between different modes."

Read carefully: `H(p)` here is the **per-sample** entropy of the K-way posterior. Minimizing it
makes each individual prediction *confident/peaked*. It says nothing whatsoever about the
**marginal** usage of modes across the batch. In fact a degenerate solution — always mode 1
with p = (1,0,0) — achieves the **global minimum** of this term. **The entropy term is a
sharpening term, and if anything it is pro-collapse, not anti-collapse.** There is no
load-balancing loss, no marginal-entropy maximization, no expert-capacity term, no
diversity/orthogonality penalty, no KL-to-uniform on the batch marginal — I grepped the whole
repo and there is nothing else.

What *actually* keeps the modes alive, in decreasing order of confidence:
1. **Stochastic categorical sampling instead of argmax.** Because `torch.multinomial(probs)` is
   used (in training *and* at inference), any mode with nonzero probability keeps receiving
   data and gradient, so an untrained expert is not starved to death instantly. This is the de
   facto exploration mechanism, and it is fragile: it is a race between the sharpening term
   driving p to one-hot and the experts differentiating fast enough.
2. **Per-expert specialization pressure.** With randomly initialized experts, whichever expert
   happens to be better on a given regime gets that regime's data and improves there — the
   standard competitive-MoE positive feedback that *can* produce a partition but can equally
   produce a single winner.
3. `[C]` Independent random init of K identical-architecture VAEs (symmetry breaking only).

`[?]` **The paper never reports mode-usage statistics** — no histogram over modes, no report of
how often a run collapses to fewer than K modes, no seed-to-seed variance in the discovered
partition. `[P]` They do hedge: "we only set the *maximum* number of modes to 3, rather than
requiring that all 3 modes must be present" — an acknowledgement that fewer than K may be used,
with no measurement of when that happens.

---

## 4. THE K ABLATION (Fig. 7, p. 8)

### 4.1 Setup `[P]`

- Metric: **`MSE(ô_{t+1}, o_{t+1})`** — the *training-time* dynamics prediction loss of the
  **selected** expert (contact BCE and KL excluded from the plotted curve; summed over the 33
  output dims, averaged over the batch).
- Conditions: `max_mode |δ| ∈ {1, 2, 3, 4}`; K=1 "represents using one network to model whole
  dynamics like [DreamWaQ]".
- **3 random seeds per condition** (network-init seeds). Shaded band = confidence interval
  across seeds.
- Everything is trained end to end inside PPO, so each K produces a *different policy* and
  therefore a *different data distribution*. This is not a held-out prediction benchmark on a
  fixed dataset.

### 4.2 Numbers

`[P]` gives **no numeric table** — only the curve. The following are `[M]` my own pixel
extraction from the vector figure at 400 dpi (I located the axis frame and gridlines
programmatically; y-axis 0.00–2.00 over 740 px, x-axis 0–6000 iterations at 172.6 px/1000).
Read accuracy ≈ ±0.01 on the mean line. Training runs to ≈6540 iterations.

| iteration | K=1 | K=2 | K=3 | K=4 |
|---|---|---|---|---|
| 1000 | 1.45 | 0.74 | 0.91 | 0.64 |
| 2000 | 1.50 | 0.67 | 0.67 | 0.63 |
| 3000 | 1.50 | 0.66 | 0.63 | 0.61 |
| 4000 | 1.50 | 0.66 | 0.65 | 0.62 |
| 5000 | 1.50 | 0.64 | 0.57 | 0.61 |
| 6000 | 1.55 | 0.60 | 0.56 | 0.61 |
| **final (~6540)** | **≈1.55** | **≈0.60** | **≈0.545** | **≈0.59** |

Qualitative shape: K=1 (green) drops to ~1.48 within ~300 iterations and then **never improves
again** — it drifts slightly *upward* to ~1.55. K=2/3/4 all land in a tight band 0.54–0.61.
K=4 converges fastest (already ~0.62 by iteration 600); K=3 converges slowest (still ~0.91 at
1000) but ends lowest. The seed CIs for K=2/3/4 visibly overlap for the entire second half of
training.

### 4.3 How much does explicit mode switching buy?

- **K=1 → K≥2 is a large effect: ≈2.5–2.8× lower prediction MSE** (1.55 → 0.55–0.61), far
  outside the seed bands, and the K=1 curve is flat-lined from very early — it is not a slower
  convergence, it is a capacity/expressivity wall.
- **K=2 → K=3 → K=4 is noise.** `[P]` "Starting from the maximum number of modes 2, the
  improvement in prediction accuracy begins to plateau... When |δ| ≥ 4, the prediction accuracy
  can hardly be improved and converges to the state of mode=3." My readings: the final spread
  across K∈{2,3,4} is 0.055, with per-condition seed bands of comparable width and only 3 seeds.
  I would not call K=3 significantly better than K=2 or K=4 from this figure.
- **The choice of K=3 is justified by narrative, not by the metric.** `[P]` "we believe that a
  mode count of 3 is the reasonable maximal number of modes for this system, which corresponds
  to three motion modes: on the skateboard, pushing the skateboard under the skateboard, and
  being in the air between the two."

### 4.4 Two confounds you should hold onto

1. **The metric is a selected-minimum-of-K statistic.** For K>1, the plotted loss is the loss of
   the expert the gate chose — and the gate is trained to choose the expert with *lowest* loss.
   Even with K identical, non-specialized experts, `E[L_selected] ≤ E[L_single]`. Part of the
   1.55 → 0.6 gap is this selection effect plus K× the parameters, not "hybrid structure".
   There is no K=1 control with matched parameter count, and no ensemble/soft-mixture control.
2. **The comparison is not on fixed data.** `z` feeds the actor, so K changes the policy, which
   changes the trajectory distribution being predicted. A K=1 policy that behaves worse (and it
   does — DreamWaQ-style baselines fail the task outright, Table III) will visit harder-to-predict
   states. The prediction-loss gap and the control-quality gap are entangled.
3. **`[P]` never reports a control metric as a function of K.** Table III (success rates) ablates
   *multi-critic* and *Beta distribution*, never K. Fig. 10 (training returns) compares against
   PPO-oracle-beta / DreamWaQ / PPO-curiosity, i.e. whole different architectures. **There is no
   evidence in the paper that K>1 improves the robot's behaviour** — only that it improves an
   auxiliary prediction loss.

---

## 5. WHAT THE LEARNED MODES CORRESPOND TO

### 5.1 Skateboarding (main task) — modes = phases, evidenced qualitatively

`[P]` §V-B, Fig. 8 (real hardware, LEDs coloured by mode) and Fig. 9 (t-SNE of an actor hidden
layer, coloured by mode):

| mode | LED | corresponds to |
|---|---|---|
| **3** | red | **gliding** — all four feet planted on the skateboard |
| **1** | green | **airborne / swing** — right front foot lifting off, transitioning on/off the board |
| **2** | blue | **pushing** — both right legs in contact with the ground |

`[P]` "This sequence of mode selections and transitions is smooth and explicitly aligns with the
decomposition of skateboarding motion: (1) gliding phase, (2) airborne phase transitioning on
and off the skateboard, and (3) pushing phase." `[P]` The t-SNE shows three clearly separated
clusters, "remarkably similar to that reported in [VAE-LOCO, Mitchell et al. 2023]".

`[M]` From Fig. 8's mode-coloured background: mode 3 occupies the long standing/gliding stretches
at the start and end; during the acceleration stage the trace alternates broad mode-2 (blue)
bands with **narrow, repeated mode-1 (green) slivers** at the swing transitions. The modes are
cyclic and locked to the push cycle.

**All of this evidence is visual.** `[?]` There is **no** confusion matrix, no mutual information
between mode and contact state, no purity/NMI score, no per-mode occupancy statistic, no
seed-to-seed consistency check of the discovered partition, anywhere in the paper.

### 5.2 Plain legged locomotion — yes, one example, in the appendix, qualitative only

`[P]` Appendix E-E + Fig. 13 applies DHAL to three more tasks:

1. **Robot hand-stand locomotion** (quadruped walking on two legs). `[P]` "In 1), we can identify
   the gait or contact mode." `[M]` Fig. 13 Task1 shows 5 frames with contacting feet circled:
   **Mode 1 = right foot only in contact; Mode 2 = both feet in contact; Mode 3 = left foot only
   in contact**, cycling 1→2→3→2→1. So here the learned modes **do** land exactly on
   foot-contact combinations — but this is a *two*-support-foot system, so K=3 covers
   {left, both, right} and the flight phase is absent. It is the degenerate 2-bit case.
2. **Dexterous manipulation hand-over.** `[P]` modes = "in the air, pushed by the right hand,
   caught by the left hand" — stages, not contacts of a legged system.
3. **Switching linear dynamical system** (textbook, with analytical ground truth). `[M]` Fig. 13
   Task3 plots predicted vs. true mode over ~132 steps: they match on essentially every switch,
   with a single visible discrepancy at t = 0–2 (predicted mode 0 vs. true mode 1), consistent
   with an unfilled history window at episode start. `[?]` **No accuracy number is given**, and
   the mode *indices* happen to coincide with the ground-truth indices, which for an unsupervised
   categorical means someone chose the label permutation by hand.

**There is no experiment anywhere in the paper on ordinary four-legged walking / trotting on flat
ground**, and no case where K learned modes are compared against the 2^4 = 16 foot-contact
combinations of a quadruped.

### 5.3 The alternative explanation the paper does not rule out

`[C]` The DHA's 960-d input **contains the sinusoidal gait clock** (`phase`, 4 s period), and the
reward's own glide/push split (`contact_phase`, `legged_robot.py:578-590`) is computed **from
that same clock** (`phase < 0.5 ∨ still` → glide; `phase ≥ 0.5 ∧ ¬still` → push), low-pass
filtered. So the behaviour is clock-driven, the mode selector can read the clock directly, and
"the mode aligns with gliding/pushing" is *at least partly* the statement that the mode aligns
with a scalar that is literally in its input. `[P]` never ablates the clock out of the DHA input,
and never tests whether the modes survive when the clock is removed or randomized. Given that
Fig. 8's mode sequence is visibly periodic, I regard "the modes encode the commanded phase" as a
live and untested competing hypothesis.

Two counterweights, to be fair: (a) the Task3 switching-LDS result has no clock and still
recovers the true modes; (b) `[P]` §V-A deploys with a **2.5 s** clock at test time when training
used **4 s**, and the predicted trajectories still track (Fig. 6), which shows the modules are
not merely memorizing the training period.

---

## 6. RUNTIME BEHAVIOUR

### 6.1 How the mode is produced at deployment

`[P]` Fig. 6 caption: "During deployment, the system utilizes the mode selection results from
the automata to choose the corresponding decoder for prediction, **consistent with the training
process**."

`[C]` `act_inference` (`actor_critic_hds.py:224-234`) calls the same `self.DHA(...)`, which ends
in the same `StraightThroughSoftmax`. There is **no `if self.training` branch anywhere**.
Therefore at runtime:
- the mode is **re-sampled from `p` by `torch.multinomial` at every 50 Hz control step**;
- it is **not** an argmax;
- there is **no hysteresis, no dwell time, no temporal filter, no dependence on the previous
  mode**;
- `play.py:88-90` just reads the returned one-hot to drive the LED colour (`env._draw_mode`).

Note the contrast with the *reward* phase indicator, which **is** low-pass filtered
(`contact_phase = 0.35·new + 0.75·old`). The mode selector gets no such smoothing.

The only thing making this stable is the `+0.01·H(p)` sharpening term: it drives `p` toward a
near-one-hot, so `multinomial(p) ≈ argmax(p)` in practice. In other words, **the anti-flicker
mechanism and the "anti-collapse" mechanism are the same term**, and it is doing the former job.

### 6.2 Does misprediction cause instability?

`[?]` **The paper never addresses this.** There is no experiment on mode misprediction, no
injected-wrong-mode ablation, no measurement of switching rate or dwell-time statistics, no
discussion of chattering, and no analysis of behaviour near mode boundaries. The words
"boundary", "hysteresis", "chatter"/"chattering", "Zeno" do not appear.

What can be said from the structure:
- A wrong mode does **not** directly corrupt the action. The actor's input is `(o_t, z)`, and
  `z` is a 20-d VAE latent. A mode flip swaps which encoder produced `z`, which changes the
  actor's input smoothly-in-magnitude but discontinuously-in-source; the actor is an MLP with
  no memory, so it will produce a different action, but the closed loop is a PD position
  controller with `action_scale = 0.25` and the joint targets are bounded by the Beta policy's
  ±3 range. That is a real cushion.
- Conversely: because the actor never sees the mode explicitly and the K latents are trained
  independently, there is **no continuity constraint tying `z^i` and `z^j` for the same input**.
  Nothing in the loss encourages the K encoders to agree near a boundary. A flip is a
  discontinuous jump in actor input by construction.
- `[P]` The strongest indirect evidence that this does not blow up is the robustness results:
  100% success on ceramic/carpet/disturbance, 100% single-step, 80% slope, 60% uneven (Table III,
  5 trials per scenario — note the caption says five, while §V-C's text says ten; **the paper
  contradicts itself on trial count**). Plus Fig. 2's pothole-recovery anecdote where both hind
  legs come off the board and it recovers. `[P]` "reliable mode identification and transitions
  under disturbances" (Fig. 1 caption) is asserted, not measured.

---

## 7. STATED LIMITATIONS (§VI, verbatim structure)

1. **Perception.** The left front foot is attached to the skateboard by a **spherical joint** in
   both sim and hardware, to keep the board from detaching. Walking→skateboarding transition is
   not solved and would need hardware changes (camera layout, multiple cameras to localize the
   board). No obstacle avoidance. They tried a RealSense T265 for state estimation and dropped
   it as unnecessary; for a free (unattached) foot, proper state estimation would need to be
   integrated.
2. **Complex skill generalization.** Cannot do extreme tricks (e.g. an ollie); the simulation
   cannot faithfully reproduce passive-wheel contact dynamics in such regimes, so they used
   approximations.
3. **Limitations in dynamics learning.** "The learned dynamics are **not yet precise enough for
   model-based control**. Furthermore, the **coupling between the controller and the dynamics
   predictor prevents iterative optimization**, such as that used in MPC, limiting the
   flexibility and efficiency of our approach."
4. **Non-trivial environment design.** The skateboarding env requires manual design and
   inspection; they suggest LLM/large-model-driven environment generation as future work.

**Additional limitations I would add, which the paper does not state:**
- No load-balancing / anti-collapse term, and no reported mode-usage statistics (§3.3).
- No quantitative mode-identification metric on any robot task; the only ground-truth comparison
  is the 1-D toy switching-LDS, and even there no number is given (§5.2).
- No control-performance ablation over K (§4.4).
- The gait clock is inside the mode selector's input and is never ablated (§5.3).
- All K experts run every step; cost is linear in K (§2.3).
- Trained on flat ground only; `[P]` Appendix D-B they tried stairs/slopes in training and saw
  "no obvious advantage, while the training time increased."

**Documentation defects found (paper vs. released code):**
| item | paper | code |
|---|---|---|
| VAE encoder | "transformer" (§IV-A text) / "1-D CNN" (Table V) | 1-D CNN |
| KL weight β | 1e-2 (Table V) | 0.9 in the loss (effective 0.3 after the /K averaging) |
| contact BCE weight | unweighted (Eq. 8) | 0.8 |
| contact dimension | "each leg" | 3 (FL, welded to the board, is excluded) |
| trial count Table III | "five times per scenario" (caption) | "ten times" (§V-C text) |
| Eq. (6)/(7) index | sums over `M` | `M` is never defined; it is `K` |

---

## 8. ASSESSMENT FOR YOUR SETUP

**Your current design as I understand it:** a GPT-style dynamics model that predicts state
*deltas*, conditioned on a *predicted* 4-bit foot-contact code that is concatenated to the input
as a soft (probability) vector, with all weights shared across contact states.

### 8.1 The one structural thing DHAL has that you don't

Strip away the framing and DHAL's only load-bearing difference is:
**mode-specific *parameters*, selected by a hard gate, with the reconstruction loss routed so
that only the selected expert is updated.** Everything else (contact prediction, history window,
VAE latent) you either already have or don't need.

Your soft-concatenated contact code gives the model *information* about contact. It does not give
it *capacity separation*: a shared MLP/transformer conditioned on a contact vector must fit a
single smooth function of (state, contact) and, at a contact discontinuity, must represent a jump
using its shared weights. That is precisely the wall the K=1 curve in Fig. 7 hits — flat at
~1.55 from iteration 300 onward while the gated versions reach ~0.6. **If your ablation is
"contact code in vs. out", you have not tested DHAL's hypothesis at all.** The right ablation is
**shared weights vs. mode-specific weights**, holding the conditioning information fixed.

### 8.2 Concrete changes, in order of value-per-risk

**Tier 1 — do this first, it isolates the actual claim and carries near-zero risk.**

Keep your predicted contact code. Do **not** introduce an unsupervised categorical yet. Instead
**hard-route on the contact code you already have**:
- Define a routing function `r: {0,1}^4 → {1..K}` with small K (e.g. K=3: `all-4-down`,
  `flight/0-1 down`, `2-3 down`; or K=4 by number of feet down; or K=5 by
  {stance, LF-swing, RF-swing, LH-swing, RH-swing} if your gaits are mostly single-swing).
- Give each route its **own output head / final block(s)** over a shared trunk. Compute all K
  head outputs, combine with a one-hot from `r(ĉ)` via straight-through so soft contact
  probabilities still get gradient; route the *loss* the same way
  (`L = Σ_i δ_i · L_i`, not `L(Σ_i δ_i · ŷ_i)` — DHAL routes the **loss**, and that is what
  trains the gate and forces specialization).
- Measure delta-prediction error vs. your current shared model on the same data and the same
  parameter budget (add a width-matched shared baseline so you're not just measuring K× params).

This buys you DHAL's capacity separation while your modes are, by construction, contact modes.
Identifiability risk: zero. Collapse risk: zero. If this doesn't help, DHAL's mechanism won't
help you either, and you have saved yourself the unsupervised-gate rabbit hole.

**Tier 2 — if Tier 1 helps and you want the learned gate.**

Only then add DHAL's DHA, and add the pieces DHAL is missing:
- Gate: MLP (or a small pooled read of your transformer's own history features) →
  softmax over K → straight-through **categorical** sample. Use `(one_hot - p).detach() + p`.
  I'd start with plain ST as DHAL does; Gumbel-softmax with temperature annealing is the
  obvious upgrade if ST is too high-variance, and DHAL gives you no reason to prefer one.
- **Detach the gate from every downstream objective except the per-expert prediction loss.**
  This is DHAL's `mode_latent.detach()` and Appendix C; it is the single design decision that
  keeps the gate from being co-opted into a general-purpose feature.
- **Add the load-balancing term DHAL lacks.** Maximize the entropy of the *batch marginal*
  `p̄ = mean_batch(p)` while minimizing the per-sample entropy:
  `L_gate = +λ₁·mean(H(p)) − λ₂·H(p̄)`. DHAL only has the λ₁ term, and as argued in §3.3 that
  term's global minimum is total collapse. λ₁ ≈ 0.01 (DHAL's value) is a fine starting point;
  λ₂ should be at least comparable. Log per-mode occupancy every epoch — DHAL never does, and
  it is the first thing that will tell you the method is silently failing.
- **Keep contact as an auxiliary BCE head on the per-mode latent** (DHAL's `FC_decoder`), whether
  or not you also feed it in. `[P]` "We include contact information to make learning hybrid
  switching easier for the networks" — the contact head is what gives the gate a contact-shaped
  gradient to grab onto. This is, I think, the most under-appreciated part of the design: DHAL's
  modes probably align with contact *because the decoder is forced to predict contact*.
- **At inference, use `argmax`, not `multinomial`, and add hysteresis.** DHAL literally samples
  at 50 Hz on hardware (§6.1). Do not copy that. Use argmax plus a margin/dwell rule
  (switch only if `p_new − p_cur > ε` for `n` consecutive steps), or an EMA on the logits.

**Tier 3 — measurements DHAL never made, which you should make.**
- Confusion matrix / NMI between the learned mode and the ground-truth 4-bit contact code.
- Mode-usage histogram and switch rate; per-seed stability of the discovered partition.
- Prediction error **conditioned on distance to a mode boundary**, and error on the step *after*
  a mode flip vs. steps with no flip. This directly answers "does misprediction hurt", which the
  paper leaves open.
- Rollout-horizon error, not just one-step. DHAL only ever reports one-step prediction; a hard
  gate that flickers can compound badly over a multi-step autoregressive rollout — and your
  GPT-style model is presumably rolled out autoregressively, which is a regime DHAL never tests.

### 8.3 Risk that learned categorical modes do NOT correspond to contact modes in a useful way

I rate this **moderate-to-high**, for six specific reasons:

1. **Nothing in the objective asks for contact modes.** The gate minimizes next-step prediction
   error. The lowest-error partition of a 960-d history is whatever splits the *prediction
   problem* best — that could be contact, but it could equally be speed regime, terrain
   stiffness, body pitch, command magnitude, or (in DHAL's case) gait-clock phase. DHAL got
   contact-aligned modes on a task where contact and phase are nearly the same variable.
2. **K=3 cannot represent a 4-bit contact code.** You have up to 16 contact states (realistically
   4–6 common ones for a trot). A learned K-way categorical with small K will coarsen them along
   whatever axis minimizes prediction error, which need not be the axis you care about. If you
   raise K to 16, you lose the capacity-per-expert that made it work and you re-inherit the
   collapse problem at scale. DHAL's own evidence for "modes = contact combinations" is the
   *hand-stand* task with **two** support feet (§5.2) — the 2-bit degenerate case.
3. **The identifiability is unmeasured.** DHAL provides zero quantitative mode-vs-contact
   agreement on any legged task. "Consistent with human intuition" is the paper's actual claim,
   and it is supported by LED photos and a t-SNE. You should not budget on it.
4. **The clock confound (§5.3).** If your observation window contains a gait phase, command, or
   any periodic signal, expect the gate to latch onto it. This is cheap to falsify: hold out the
   clock from the gate's input and see whether the partition survives.
5. **No load balancing (§3.3).** With DHAL's loss as literally written, the degenerate
   all-one-mode solution is the *minimizer* of the entropy term. Any run that collapses will look
   fine on the loss curve and simply be your old shared model with wasted parameters. You will
   not notice unless you instrument mode usage.
6. **Delta prediction changes the economics.** DHAL predicts the **absolute** next observation
   (§2.2). Predicting a delta already removes most of the smooth, mode-independent part of the
   signal, so what's left for the experts to specialize on is a smaller, noisier residual — the
   contrast between experts is compressed, and the winner-take-all assignment gets noisier.
   I'd expect DHAL's K=1→K≥2 gap to shrink substantially in a delta-prediction setting. This is
   speculation on my part, not something the paper addresses, but it is a specific reason not to
   expect a 2.5× improvement to transfer.

**Counter-consideration, so this isn't one-sided:** if your contact-code predictor is
well-calibrated, you already have the information, and a *supervised* hard route (Tier 1) gets
you all of DHAL's structural benefit with none of its identifiability risk. The learned
categorical is only worth reaching for if you suspect the true dynamics discontinuities are
**not** exactly the foot-contact boundaries — e.g. slip vs. stick, terrain compliance, or
actuator saturation — in which case an unsupervised gate might discover a partition your 4-bit
code cannot express. That is a real possibility, and it is the honest argument for Tier 2. But
it is a hypothesis to test, not the paper's finding: DHAL never demonstrates that a learned
partition beats a contact-supervised one, because it never runs that comparison.

### 8.4 Bottom line

DHAL's transferable content is one idea — **route the loss through a hard one-hot to
mode-specific weights, and keep that gate's gradient isolated from everything downstream** —
plus one strong supporting trick — **make the per-mode decoder predict contact as an auxiliary
target**. Its empirical support is a single auxiliary prediction-loss curve on one task, 3 seeds,
no numeric table, no control metric as a function of K, and no quantitative mode-identification
score on any robot. Its anti-collapse story does not survive reading the loss. The runtime mode
selection is a per-step multinomial sample with no hysteresis, which you should not replicate.
Adopt the structure, supervise the route from your contact code first, add load balancing before
you ever let the route go unsupervised, and instrument mode occupancy from day one.
