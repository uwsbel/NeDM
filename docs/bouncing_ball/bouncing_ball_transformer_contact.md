# Transformer NRD with a learned contact residual

The requested Transformer backbone is preserved. The tested x,z-only contact
branch improves the tail error of the saved early V1, but has worse median
error and does not beat the accepted V2 Transformer. This is a negative
precision result for this implementation and training recipe, not a rejection
of Transformer plus contact residuals as an architecture.

All training, model evaluation, Chrono collection, plotting and checkpoint
loading ran on AMD compute nodes. Work is isolated in
`/home/harry/NeDM-ball-transformer-contact`, branch
`study/ball-transformer-contact`, base
`5cae9e524aa790719aedea7c7bc1b6488c946266`. Existing ball checkpoints,
arm/HMMWV studies, the active traversal checkout and PR work were preserved.

## Implemented primary

```text
state [B,5] = [x,z,vx,vz,omega_y]
  -> existing ContinuousTransformer (2 layers, 4 heads, embedding 64)
  -> delta1 [B,5]

[x,z] [B,2] -> MLP (2 -> 64 -> 64 -> 16) -> latent [B,16]
  -> contact logit [B,1] -> sigmoid -> hard switch [B,1]

[vx,vz] [B,2] -> shared bounce MLP (2 -> 128 -> 128 -> 5)
  -> delta2 [B,5]

next_state = state + delta1 + switch * delta2
  -> feed predicted next_state back into all branches
```

The final residual retains `state` because both outputs are deltas. Delta2
updates all five components; this tests whether position/spin corrections can
be learned from the requested two velocity inputs without an explicit timing
head. The scalar switch means contact occurs during the next **10 ms**
interval, rather than currently touching. Ground and wall share one bounce
map in the literal primary. Its validation-selected refined candidate uses
`sigmoid(logit) >= 0.2`; the initial candidate uses 0.5.

There is no analytical gravity integration, surface-distance feature, contact
root, event-time head, geometry guard, position clamp, true contact, target or
future state in these new models' public inference. Ordinary flight is also
learned by the Transformer. Data normalization and the free-transition mean
delta are fitted from training states only. The backbone is the actual shared
`nedm.core.training.model_transformer.ContinuousTransformer`, with one state
token (`block_size=1`), as in the earlier ball model. This adds no temporal
attention or state history.

Code: `src/nedm/bouncing_ball/transformer_contact.py`. The public loader in
`model.py` dispatches architecture `transformer_contact_residual_v1` without
changing legacy model implementations.

## Research review

A separate research agent reviewed the idea and implementation. No general
objection was found to a Transformer plus gated contact residual. The nearest
primary precedent, [Learning Contact Dynamics using Physically Structured
Neural Networks](https://arxiv.org/html/2102.11206v2), learns a contact signal
and separates smooth/contact updates. It also discusses the need to keep the
smooth branch from absorbing contact effects. Its dynamics is physically
structured, so it supports the decomposition rather than a claim about this
Transformer or exact input split. [Official implementation](https://github.com/libeanim/contact-symplectic-integrator-network).

The restricted inputs remain hypotheses: the same position can be approached
or departed, and the same velocity can have different outcomes at different
surfaces or impact phases. A broad position gate with a velocity map that
learns zero correction on outgoing states could still work in this fixed
launch domain. The experiment does not prove that the whole model must fail.

## Common data and evaluation

- Existing **5,400 train / 900 validation** episodes, with whole-episode splits.
  No test episode contributes to fitting, normalization or selection.
- **900 genuinely new Chrono episodes**, seed **202610014**, two launches per
  cell of the 30 by 15 velocity grid, paired across all 12 evaluated models.
- Native 100 Hz model updates through each valid horizon, approximately
  1.88–2.18 s; stop 20 ms before the second ground impact. The first two
  impacts are ground then wall. Initial launch span remains
  `vx=4..7 m/s`, `vz=-10.5..-9 m/s`.
- Approved sphere/floor/wall scene and materials unchanged; physical step
  0.125 ms, raw recordings every 0.5 ms, rolling friction 0.001 m.
- All nine new checkpoints were frozen from validation before fresh-test
  inspection. No training, inference tuning or reselection followed testing.

RMSE below is per-episode Euclidean position RMSE over the whole valid native
trajectory, then summarized across the same 900 episodes. Median and p95 are
both shown to avoid interpreting a smaller failure tail as better typical
motion.

| Model | Median RMSE | p95 RMSE | p95 endpoint | Correct bounce order |
|---|---:|---:|---:|---:|
| Historical V1, `nrd_v1` | 126.961 mm | 1026.517 mm | 1889.444 mm | 854/900 |
| Historical V1 refined, `nrd_v1_refine` | 50.251 mm | 1095.891 mm | 1498.302 mm | 851/900 |
| Accepted Transformer V2, `nrd_v2_certified` | 16.571 mm | 30.923 mm | 59.205 mm | 900/900 |
| Matched pure Transformer residual, refined | 1691.620 mm | 2448.988 mm | 5193.465 mm | 0/900 |
| **Literal x,z gate + vx,vz bounce, refined** | **323.388 mm** | **658.663 mm** | **1320.072 mm** | **876/900** |
| Two x,z surface gates and bounce maps, refined | 236.955 mm | 486.465 mm | 925.152 mm | 885/900 |
| Full-state contact gate, otherwise shared velocity bounce, refined | 59.722 mm | 116.891 mm | 211.951 mm | 896/900 |

The new literal candidate has 1.56x lower p95 RMSE than saved early V1 but
2.55x worse median RMSE. Against accepted V2 its p95 is **21.30x worse**.
The full-state-gate diagnostic helps but remains **3.78x worse** than V2.
The matched pure Transformer control is also a failed model, so beating it
does not establish practical adequacy or superiority over a well-trained
Transformer family.

Historical V1/V2 are not pure `state -> Transformer -> delta` models: they
already contain gravity/free flight and geometry-based contact priors. V2
also uses impact-time features. V1 was trained on an earlier packed-data
revision (`944ac649...`), whereas the new experiments use `cce4f5b5...`.
These are preserved historical performance comparisons, not identical-data
architecture ablations. The separately trained pure residual control uses
the same backbone, current training data, native timestep and update budgets
as the new augmented models.

All 12 models were finite on all 900 launches. A separate raw position check
uses identical linear interpolation of native 10 ms predictions against the
0.5 ms recordings. This is not an exact dense physics reconstruction. It
gives p95 RMSE 657.062 mm for the literal candidate and 116.715 mm for the
full-state-gate diagnostic; native metrics do not hide a precision success.

## Contact, ordinary motion and gradients

On fresh true states, the selected literal gate has precision **0.512**,
recall **0.554**, F1 **0.532**. The full-state-gate diagnostic reaches
precision **0.992**, recall **0.927**, F1 **0.958**. Gate scores on predicted
rollout states are separately recorded; a shifted contact time is counted as
a miss plus a false activation when compared with the reference interval.
These classification metrics are not interchangeable with bounce order.

Longer cosine pretraining plus L-BFGS improved the frozen noncontact
Transformer's one-step validation position p95 to **0.083643 mm** and velocity
p95 to **6.672e-6 m/s**. Joint/rollout training subsequently disturbed the
ordinary-motion fit: the selected literal model's true-state noncontact
one-step position p95 is **5.338 mm**, versus **0.925 mm** for the full-state
gate. Thus the remaining error cannot be attributed to gate information
alone; branch training and implicit position/timing correction also matter.

Validation branch traces show all 900 literal rollouts cross 10 mm position
error before the first reference impact, mostly at 10–30 ms. Their gate
averages 3.46 ON steps instead of two event intervals, with 2,221 false ON
steps and 911 reference-interval misses across the cohort. The full-state
diagnostic averages 2.02 ON steps and has smaller early drift. These are
diagnostic observations, not a causal proof isolating every source of error.

For three declared launches at T=1.7 s, all selected hard-gated models have
finite launch gradients. The literal model's maximum endpoint-Jacobian
autograd/finite-difference discrepancy at epsilon 1e-5 is **8.84e-10**. The
hard threshold itself has no derivative; this validates local derivatives
while the switch pattern is unchanged, not global smoothness or accurate
physical targeting. New target optimization/Chrono targeting was not run
because the requested model fails the trajectory-accuracy gate.

The next justified work would preserve this Transformer backbone while
testing a contact head with velocity context and stronger protection of the
ordinary-motion core during contact/rollout fitting. That is a recommendation,
not a completed further experiment or proof that an explicit timing head is
required.

## Training budgets and reproducibility

Initial runs: 3,000 noncontact core updates, 3,000 contact-classifier updates,
3,000 oracle-routed bounce warm-up updates, 12,000 joint updates, then 400
masked half-second recursive updates. The pure control substitutes 18,000
joint updates for the classifier/bounce warm-ups: **21,000 Adam updates** in
both cases. A retrained sigmoid-gate ablation also ran; its p95 RMSE was
3246.286 mm, rather than reusing hard-gate weights for soft inference.

Refined runs: 30,000 cosine core updates, 80 outer L-BFGS steps (up to 15
inner iterations) on a fixed 4,096-example training batch, 3,000 classifier
and 3,000 bounce warm-up updates, 6,000 predicted-gate bounce updates with
neighbor/noncontact negatives, 12,000 joint and 400 masked half-second
updates. The pure control instead uses 24,000 joint updates: **54,000 Adam
updates** plus the same core L-BFGS and rollout budgets. It receives the same
noncontact retention penalty as the refined augmented models. Independent
data RNG makes refined core pretraining minibatches identical across models;
the resulting free-core metrics match. One seed, 17, was used per config.

Budgets are update-matched, not FLOP-matched or identical-objective: the
augmented models have classifier supervision and extra parameters. Initial
runs lacked the matched pure-control core-retention penalty; they are kept
as earlier results, not presented as the stronger controlled comparison.

The research agent found and helped correct two evaluator issues before
certification: classifier-statistic key overwriting and variable-length
trace concatenation. Signatures were also disambiguated with geometry in
metrics only. Initial training selected from its old sign-only order metric;
discarded stage weights were not reconstructed. Final selection reevaluated
each preserved best checkpoint on validation with the corrected evaluator.

AMD campaign:
`/work1/dannegrut/harry/experiments/ball_transformer_contact_20261001T192800Z`.
Local bundle:
`artifacts/training_runs/bouncing_ball_transformer_contact_20261001T192800Z_amd`.
The bundle includes all nine frozen checkpoints, complete paired metrics,
contact/branch/loader diagnostics, configs, logs, source snapshots and hashes.
Raw recordings, packed data and replay trace arrays remain on AMD.

Actual executed source: v2 failed only an overly strict floating-point
equality contract test; v3 initial training; v4 refinement smoke; v5 refined
training; v6 selection and certification; v7 public-loader checks and figure.
Collection used v2. The v1 transfer had an incorrect folder layout and never
executed. All fixes preserve the stated inference contract.

Jobs: initial smoke 447267 (failed equality assertion), collection 447268,
successful smoke 447271/447273, initial training 447272/447276–447279;
refinement smoke 447286/447287; refined training 447290–447293; selection
447302; certification 447303; final checks/plot 447306. Process completion
does not imply an accuracy gate passed.

Seven contract tests passed on AMD. Actual public-loader CPU checks passed
for four selected checkpoints, including float32 launch input, finite
gradients, `[1,171,5]` rollout shape, the Transformer backbone, absent analytic
geometry/flight attributes and zero state difference against the equivalent
float64 input. No heavy local jobs or release/PR actions were performed.

```python
from nedm.bouncing_ball.model import load_model

model, metadata = load_model("frozen/transformer_xz_binary_refined/best.pt", "cpu")
trajectory = model.rollout(initial_state_5d, 170)  # [B,171,5], T=1.7 s
```

Use the matching source snapshot and run on an allocated compute node.
