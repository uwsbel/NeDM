# Quadruped joint level: reduce the terrain, not the robot

**Written 2026-09-05.** Supersedes
[`quadruped-case-study-plan.md`](quadruped-case-study-plan.md), whose stated
contribution (contact-mode conditioning as a soft context input) was measured and is
insufficient. Closes out of
[`go2-finetune-postmortem.md`](go2-finetune-postmortem.md).

Grounded in three papers read in full **with their released code**, not from abstracts.
Specs at `scratchpad/papers/{nerd,dhal,halo}-spec.md`; each cites paper-vs-code
discrepancies, and there are many.

## Where this starts

**Closed:** fine-tuning the imported policy. Failed both declared rules, and no available
surrogate is better than the one it used at the only horizon anything passes.

**The binding constraint, stated as a ratio rather than an error.** Trustworthy horizon
**0.1 s**, five steps at 50 Hz, against a **0.3-0.5 s** gait cycle. The model cannot cover
one stride. This ratio, not rollout RMSE, is the quantity that decides whether a surrogate
is usable for a given plant, and it should be reported for every case study in the paper.

**What is not the constraint:** context length (16 tokens is already 0.32 s), data volume
(10.5 M transitions), or the command channel (solved).

## What the literature actually establishes, and what it does not

### NeRD (Xu et al., CoRL 2025) — the strongest transferable result

**Contact information is 100% INPUT.** NeRD predicts no contact quantity: no impulse head,
no force head, no mode classifier. Contact points, normals and depths come from **analytic
collision detection re-run every step on the network's own predicted state**. Their
ablations:

| variant | Double Pendulum | Ant |
|---|---|---|
| end-to-end, no contact input, world frame | **56.1x** | **19.8x** |
| world frame (contact input kept) | 1.1x | **23.3x** |
| absolute rather than relative prediction | 26.8x | 5.4x |

**That analytic query is a closed error-correcting loop wrapped around an open-loop-trained
network.** Over-predict penetration and the next step's depths come back negative and push
against it. Together with base-frame re-anchoring, which makes step 1000's input
distribution statistically identical to step 1's, this is the whole stability story.

**Two corrections to how this paper is usually cited, including by us:**

1. **The 1000-step open-loop state error is Cartpole only** — 2 DoF, contact-free. Ant is
   500 steps and the paper concedes the motion converges to a static attractor. **For
   ANYmal there is no reported open-loop state error at any horizon.** The only ANYmal
   number is closed-loop reward agreement.
2. **That reward saturates.** `exp(-((v_x-1)^2 + v_z^2)) + 0.5 exp(-w_y^2)` is
   first-order insensitive at the peak where a trained policy sits. The one
   **non-saturating** reward in the table, Ant spinning `R = w_y + p_up`, shows
   **+17.21%**, an order of magnitude worse than every saturating one.

**So there is no published open-loop quadruped result to be behind.** Our 0.1 s is not a
gap against a known number; nobody reports the comparable one.

**And NeRD trains one-step, teacher-forced. No rollout loss, no noise injection** (the
noise path exists in the code with no call site). It does not need them, because the
analytic query does that job.

### DHAL (Liu et al., 2025) — one transferable idea, weaker evidence than advertised

**The idea:** route the **loss** through a hard one-hot to **mode-specific weights**, and
isolate the gate's gradient from every downstream objective. K fully separate VAEs, not
heads on a shared trunk; `mode_latent.detach()` keeps PPO out of the gate.

**The K ablation, read off the figure** (the paper gives no table):

```
  K=1   1.55, flat from iteration ~300, never improves     <- a capacity wall
  K=2   0.60
  K=3   0.545
  K=4   0.59
```

**K=1 to K>=2 is 2.5-2.8x and real. K=2 vs 3 vs 4 is noise**, with 3 seeds and overlapping
bands. Anywhere this repo says "K~3 is optimal", that is wrong; **K>=2 is what is
supported.**

**Three things that do not survive reading the code:**

1. **The entropy term does not prevent collapse.** `+0.01 H(p)` is *per-sample* entropy,
   minimized. It sharpens each posterior. The degenerate always-mode-1 solution attains
   its **global minimum**. There is no load-balancing term, no marginal-entropy term, no
   diversity penalty, and **no reported mode-usage statistics anywhere in the paper**.
2. **The "modes are contact combinations" evidence is the hand-stand task**, a **two**
   support-foot system where K=3 covers {left, both, right}. That is the degenerate 2-bit
   case. There is **no experiment on ordinary four-legged walking**, and no comparison of
   learned modes against the 16 foot-contact combinations.
3. **The gait clock is inside the gate's input and is never ablated.** The modes may
   encode commanded phase rather than contact.

Also: the loss is on **absolute** next observation, not a delta. Predicting deltas removes
the smooth mode-independent signal, so the contrast experts specialize on is a smaller,
noisier residual. **Expect the 2.5x to shrink in a delta-prediction setting.**

### HALO (2026) — do not cite it for what it does not do

Its G1 Poincare section uses **one foot**. The other foot's events are extracted and
discarded, with the literal code comment `# just choose the default first one for now`. It
contributes **nothing** on asynchronous multi-leg events.

Its gait is driven by an **exogenous 0.6 s clock** fed to the policy as sin/cos, so
touchdowns land at near-constant clock phase and the "Poincare map" is approximately a
fixed-dt flow map. Inter-event time is extracted, used for data cleaning, then **never fed
to the model**. Its hardware rollout is flat: the ROM sits at `p_z` 0.767 m while the truth
oscillates 0.745 to 0.777. **It predicts the mean, which is our own pathology.**

The structural objection, which is the reason item W4 is gated:

> **Event indexing only differs from time indexing in the regime where it is also least
> valid.** With a clocked trot on flat ground, event ~ time and the machinery buys
> nothing. Once terrain makes touchdown drift off the clock, event indexing starts to
> matter, and that is the same moment `x_{k+1} = f(x_k)` stops being true, because
> granular terrain is a state with memory.

## The reframing

**Reduce the terrain, not the robot.**

NeRD's correcting query must be cheap and computable **from the predicted state, every
step**. From a reduced robot state you cannot run forward kinematics, so you cannot get
foot poses, so you cannot query the terrain, so the correcting loop disappears. **Keep
full joint positions in the state.**

The half nobody has reduced, and the half CRM makes expensive, is the terrain. Every paper
above is rigid-body contact on a plane; in NeRD's code `initialize_contacts` carries
`# NOTE: only work for ground env for now`. Soil differs in the way that matters: it has
**its own state, with memory**. Ruts and compaction persist and a trot re-loads nearby
ground every cycle.

So the object becomes a co-evolving pair,

```
  (s_t^B, F_t^B, tau_t, g_t^B)  ->  (delta_s, delta_F)
```

with `F^B` a base-rotated local height and compaction field scattered back into a
persistent world buffer, so a rut dug at step 200 is still there at step 800. **That is
the contribution, and it is not a modification of NeRD.** It borrows four of NeRD's
structural ideas and replaces its environment representation entirely. Scoping it as the
latter is the honest framing.

## Work items, in order

### W0 — Is the map a function of the state at all? (one day, no architecture)

**Runs first because it can kill W4 and re-scope everything else.**

Log Go2 trot data on soil **and** on rigid as a control, contact logged **faster than the
50 Hz policy rate**. Detect FL touchdown with the repo's hysteretic
`dataset.contact_mode` (5/60 N), never a plain threshold: on CRM a single threshold fires
at **1.66x** the rate the gait's own spectral peak implies, so it would manufacture events
on soil and not on rigid, fabricating exactly the difference this test measures. Extract
the pre-impact state in the RR-stance yaw-aligned frame, fit ridge / kNN / MLP, and report
per-component R^2.

Harness: `scripts/evaluation/w0_poincare_decidability.py`, with a synthetic
discrimination test at `test/test_w0_discriminates.py`.

**Three things the harness had to be corrected on while it was built, all of which change
how the result must be read:**

1. **Score the INCREMENT, not the level.** R^2 for `x_{k+1}` given `x_k` is inflated by
   persistence: a random walk scores near 1.0 while carrying no learnable dynamics. On the
   synthetic pair the level metric gives **0.966 vs 0.906 (indistinguishable)** where the
   increment gives **0.701 vs 0.112**. The level R^2 is printed only as a persistence
   diagnostic and is never the verdict.
2. **The threat model is SPATIAL variation, not soft terrain.** A uniform per-episode soil
   property is *not* a threat: the state accumulates it and the map stays predictable. It
   was measured. What breaks the formulation is a field the robot **walks into**, whose
   next value `x_k` cannot carry. Soil hysteresis, where the robot changes ground it will
   re-encounter, is the same failure in a worse form.
3. **A component with no spread is unscoreable, not unpredictable.** If every episode
   converges to the same limit cycle, `ss_tot` collapses and R^2 sits near zero however
   good the model is. That is HALO's Figure 5 read correctly. Such components are reported
   as `---`, and if most of the state is degenerate the harness withholds a verdict rather
   than returning a null.

**Decision rule, fixed now:** if R^2 over the mean-predictor is small on soil and large on
rigid, the deterministic event-indexed map is dead here and W4 does not run.

Two diagnostics alongside, both things HALO never reports:

- spread of the inter-event interval, soil vs rigid. More than a few percent means `dt_k`
  must be a predicted output.
- offset between the diagonal pair's touchdowns. This decides directly whether
  single-leg sectioning is defensible for a trot.

**Sampling caveat that applies to all of W0.** HALO detects events at 1 kHz and takes the
last sample before contact with no interpolation. At 50 Hz our quantization is up to
20 ms, sampled at peak state derivative, which converts to roughly 6-16 mm of body-height
error. HALO's own hardware signal is 30 mm peak to peak. **Log contact faster than the
control rate or W0 measures our sampling, not the plant.**

### W1 — Contact and terrain as INPUT, not prediction (the main line)

Adopt, in order:

1. **Contact geometry moves from a learned output to `G()`.** Foot position is forward
   kinematics on predicted joints, terrain height is known, so contact indicator and
   penetration are **computed** from the predicted state, not predicted. This is the role
   the framework already defines for vehicle pose and the arm end-effector, never applied
   to contact.
2. **Base-frame re-anchoring every step**, with the gravity direction in base frame
   re-injected so the reduction stays lossless.
3. **Relative prediction with per-DoF-type composition and inverse-std loss weighting.**
   Not optional here: sinkage deltas in mm and joint-velocity deltas in rad/s differ by
   four orders of magnitude.
4. **Direct macro-step prediction.** NeRD's economics are 1.6x against Featherstone. Ours
   are three to five orders of magnitude against CRM.

**The open design problem, and it is the research question:** what is the cheap,
memory-carrying stand-in for the CRM query. It must run every step from the predicted
state. Candidate feature set per foot: local height and normal, sinkage and its rate, slip
ratio, patch area, plus local soil state (packing fraction, prior disturbance). That is
the Bekker-Wong / SCM feature set, with one character change worth stating plainly:
**NeRD's contact input is purely kinematic; ours must carry material state.**

Note the invariance argument must be **re-derived, not re-parameterized**: translation
survives, but gravity-axis rotation survives only if the soil is isotropic, and it is not
once rutted. It becomes an approximate symmetry with the field carrying the anisotropy.

### W2 — Hard-routed mode-specific weights, supervised from the contact code we have

**Our conditioning experiment tested the wrong hypothesis.** We concatenated a predicted
contact code as a soft input with all weights shared. That gives the model *information*
about contact. It does not give it *capacity separation*, and a shared trunk must still
fit one smooth function of (state, contact) and represent a jump with shared weights.
**"Contact code in vs out" does not test DHAL's claim.** The right ablation is **shared
weights vs mode-specific weights, holding conditioning information fixed.**

Tier 1, which is what this plan authorizes:

- Route on the contact code we already predict, via `r: {0,1}^4 -> {1..K}` with small K.
  Our own measured histogram
  ([`go2-contact-mode-coverage.md`](go2-contact-mode-coverage.md)) says a trot occupies
  **8 of 16 modes, with 4 nearly absent**, and that all-four-down plus the two diagonals
  plus flight is 75% of transitions. **The data supports a small K on its own terms**,
  independent of DHAL.
- Give each route its own head or final block over a shared trunk.
- **Route the LOSS, not the output:** `L = sum_i delta_i L_i`, never `L(sum_i delta_i
  y_i)`. This is the mechanism.
- Include a **width-matched shared baseline**, because DHAL has none and part of its 2.5x
  is a selected-minimum-of-K statistic plus K times the parameters.

Identifiability risk zero, collapse risk zero. **If Tier 1 does not help, the learned
categorical gate will not help either**, and we stop there rather than entering the
unsupervised-gate rabbit hole.

Tier 2 only if Tier 1 helps: straight-through categorical, gate detached from every
objective but per-expert prediction loss, **plus the load-balancing term DHAL lacks**
(minimize per-sample entropy, maximize entropy of the batch marginal), and per-mode
occupancy logged every epoch. At inference use **argmax with hysteresis**, never DHAL's
per-step multinomial.

### W3 — Rollout loss

HALO backpropagates through 7 composed latent steps and credits its flat multi-step error
to it. NeRD does not use it. So this is supported by one paper of the three and is
**subordinate to W1**: if the analytic query closes the loop, one-step training may
suffice, which is the cheaper outcome. Run it as an ablation on top of W1, not before it.

### W4 — Event-indexed Poincare formulation

**Gated behind W0.** Highest ceiling, highest risk, and the risk is structural rather than
implementational.

## Measurement protocol, fixed before any result exists

**Never a saturating reward as the headline.** NeRD's agreement numbers come from one, and
its single non-saturating reward is an order of magnitude worse. A reward-parity result on
soil would be evidence about the reward's insensitivity.

Report instead:

- **Open-loop state divergence against CRM at 100 / 500 / 1000 steps, per DoF.**
- **Terrain-side error separately from robot-side** (rut depth and sinkage apart from
  joint and body error). A model can look good on one while failing the other.
- **The horizon-to-instability ratio**, for this case study and retrospectively for the
  HMMWV and the arm.
- For W2, **mode occupancy, switch rate, and prediction error conditioned on distance to a
  mode boundary** — none of which DHAL reports.

**And the channel-set assertion on every A/B**, per
[`../lessons/experiment-design.md`](../lessons/experiment-design.md#the-same-selection-rule-run-on-two-inputs-is-not-the-same-selection).
Two models with different state definitions must be compared on sets asserted **equal**,
never on the same selection rule run twice.

## Data, which is the quiet blocker

NeRD used 10 M transitions, 46 hours on Featherstone. On CRM that is weeks to months per
soil parameterization, and **random-torque collection does not transfer**: on soil it
buries the robot in the first second, and the regime we want (sustained locomotion, slip
0.1-0.5, rut formation, re-loading compacted soil) is vanishingly rare in random-torque
space. This is the authors' own stated limitation and it bites hardest here.

**The one recipe in that paper already validated across a physics gap: pretrain cheap on
an SCM / Bekker-Wong terrain family, fine-tune on a small CRM set.** Their real-data
fine-tune converged in **under 5 epochs from 400 trajectories, 10x faster than scratch**.
This costs task-agnosticism, and that trade should be made explicitly rather than by
default.

## What is NOT being done, and why

- **No further fine-tuning of the imported policy.** Closed.
- **No targeted rare-contact-mode collection** until W2 Tier 1 shows mode-specific weights
  help. The justification chain was rebuilt three times on a first link that is now
  measured at **+0.245, not +0.793**, and it should not be rebuilt a fourth time before
  the mechanism it depends on is demonstrated.
- **No unsupervised mode gate** before supervised routing is shown to work.
- **No trajectory-reproduction metric** while the controller is stateful. See
  [`../lessons/rl-in-nrd.md`](../lessons/rl-in-nrd.md#a-stateful-policy-is-part-of-the-dynamical-system-and-the-reduced-state-does-not-cover-it).

## What would reopen the closed items

A surrogate that passes the action-sensitivity gate at a horizon covering a full gait
cycle. W1 is the attempt at that; nothing else on this list is.
