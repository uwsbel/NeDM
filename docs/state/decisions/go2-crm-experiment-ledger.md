# CRM fine-tuning: experiment ledger

Every arm scored on CRM, the state of the open questions, and what is in flight. The
reproducible recipe lives in [`go2-crm-finetune-RECIPE.md`](go2-crm-finetune-RECIPE.md);
this file is the surrounding evidence.

**Read the table as one experiment.** Every row below was scored on the SAME machine
(MI210, HIP) against the SAME base-policy run, paired per episode on identical seeds and
command schedules. Cross-machine numbers exist for several arms and agree to within about
two percentage points, but mixing machines inside one comparison is not safe at the 0.003
level, so this table does not.

## Results

| arm | n | Δ mae_vx | wins | yaw vs base | p |
|---|---|---|---|---|---|
| `w_h15_dw2` | 75 | **-43.8%** | 68/75 | 76% | <1e-4 |
| `b_bal10` | 75 | **-43.7%** | 67/75 | 73% | <1e-4 |
| `s_h15r0_s3` | 75 | **-42.3%** | 69/75 | 73% | <1e-4 |
| `x_bal_s2` | 75 | **-40.6%** | 67/75 | 75% | <1e-4 |
| `s_h15r0_s1` | 74 | **-40.2%** | 61/74 | 74% | <1e-4 |
| `w_h15r0` | 75 | **-40.3%** | 68/75 | 77% | <1e-4 |
| `s_h15r0_s2` | 75 | **-40.3%** | 64/75 | 76% | <1e-4 |
| `b_bal05` | 75 | **-38.3%** | 67/75 | 77% | <1e-4 |
| `x_r0_s2` | 75 | **-35.9%** | 63/75 | 82% | <1e-4 |
| `w_h15_dw4` | 76 | **-32.5%** | 65/76 | 142% | <1e-4 |
| `x_r0_s3` | 75 | **-32.7%** | 66/75 | 113% | <1e-4 |
| `w_r0` | 74 | **-25.3%** | 60/74 | 97% | <1e-4 |
| `z_ppoA03` | 75 | **-22.3%** | 54/75 | 83% | <1e-4 |
| `x_bal_s3` | 75 | **-22.0%** | 61/75 | 145% | 0.018 |
| `w_r0fz` | 74 | **-21.3%** | 57/74 | 106% | <1e-4 |
| `z_ppoA30` | 77 | **-9.5%** | 60/77 | 88% | <1e-4 |
| `z_ppo25` | 72 | **-8.9%** | 45/72 | 107% | 0.268 |
| `z_ppo15` | 75 | **-4.6%** | 34/75 | 160% | 0.333 |
| `d_anch10` | 77 | **-1.7%** | 46/77 | 97% | 0.056 |
| `d_anch50` | 77 | **-1.6%** | 45/77 | 97% | 0.068 |
| `r_rsl_r0` | 75 | **+0.7%** | 31/75 | 123% | 0.893 |
| `c_rsl_anch_r0` | 77 | **+0.7%** | 37/77 | 100% | 0.447 |
| `c_rsl_anch` | 77 | **+1.5%** | 32/77 | 110% | 0.697 |
| `e_noboot` | 75 | **+2.6%** | 36/75 | 103% | 0.496 |
| `e_both` | 76 | **+3.1%** | 34/76 | 97% | 0.193 |
| `r_rsl_full` | 77 | **+10.2%** | 33/77 | 105% | 0.001 |
| `e_g96` | 73 | **+20.0%** | 28/73 | 175% | 0.004 |
| `w_r1` | 75 | **+39.6%** | 22/75 | 428% | <1e-4 |

## What is settled

**The method works, and not by a hair.** Best arms sit at -43.8% velocity tracking error
with 68 of 75 episodes won. Yaw and lateral improve too. No falls: completion is 1.000 for
base and fine-tuned alike, which is why completion does not discriminate on this terrain.

**It is not a lucky seed.** Three independent training seeds: -40.2%, -40.3%, -42.3%.

**It is not specific to the surrogate it was developed against.** Every arm was built
against `baseline_s1`. Retrained against two independently seeded surrogates:
gradient balancing -40.6% / -22.0%, penalties-dropped -35.9% / -32.7%. All six significant.

**It is not exploitation of the learned model.** The fine-tuned policy does leave the
training distribution (8.7% of visited states beyond the corpus p99 against the base
policy's 0.97%) and the surrogate is 1.8-2.1x less accurate there even after controlling
for speed. But the degradation is not specific to the model that was optimised -- two
held-out surrogates are slightly WORSE on those states, difference-in-differences negative
-- and the error runs in the wrong direction to help: the model OVER-states the fine-tuned
policy's tracking error. Ensemble disagreement rises in lockstep, so the ensemble correctly
signals its own ignorance. The test was stacked toward finding exploitation, since the
corpus was collected by the base policy.

**Moving the weights is not what helps.** Three random perturbations at the same weight
displacement as the optimised arms cost nothing: -1.9%, -0.2%, +2.0%. The direction is
what matters, not the distance.

**Displacement has an optimum.** dW 2.0 (-43.8%) beats dW 1.0 (-40.3%) beats dW 4.0
(-32.5%, and yaw degrades to 142%).

**Gradient balancing removes the need to delete reward terms.** Rescaling each term so its
share of the gradient matches its share of the return reaches -43.7% while keeping every
term, deriving a multiplier of 0.018 for `dof_acc` from the policy's own gradients. Prefer
it to `--reg-scale 0`: same result, unmodified objective, and the best yaw of any arm.

## What is not settled

**Reinforcement learning in the surrogate underperforms, and the reason is not yet proven.**
With a well-tested implementation and no trust region, PPO on the same reward in the same
model makes the policy significantly WORSE (+10.2%). Two explanations were tested:

- *Trust horizon* -- that credit assignment reaches past the ~0.5 s the model supports.
  FALSIFIED in isolation: removing the time-out bootstrap does nothing (+2.6%, p 0.50),
  combining it with a matched discount does nothing (+3.1%, p 0.19), and a matched discount
  alone is significantly worse (+20.0%, p 0.004). The monotone pattern that motivated this
  spanned three different implementations and was a coincidence, not a causal ordering.
- *Drift* -- that exploration noise walks the policy into regions the model predicts badly.
  Better supported. Ensemble pessimism cuts the post-peak decay by roughly 7x, and the only
  PPO arm that has never turned over (`f_all`) combines pessimism with the horizon levers
  that did nothing on their own. Chrono scoring pending.

**One terrain.** Everything here is CRM. Transfer to rigid ground is the case study's
actual claim and is untested.

**No held-out command families.** All eight appear in both the branch pool and the scoring
set, so this measures adaptation, not generalisation to unseen commands.

## In flight

| what | where | answers |
|---|---|---|
| `f_pess`, `f_h15`, `f_all`, `e_sig05` | cluster + sliger | does the PPO stack that stopped collapsing transfer to Chrono? |
| `w_h25_dw1` | cluster + a3 | does the 0.50 s analytic horizon beat 0.30 s, or turn over? |
| `g_long`, `g_long2` | sbel | `f_all` at 4x budget, two seeds -- it was budget-limited, not converged |

## Next, in priority order

1. **Rigid terrain.** The single largest untested claim. The surrogate, the recipe and the
   scoring harness all exist for it; what is missing is a rigid corpus run through the same
   pipeline and a rigid verdict set.
2. **Held-out command families.** Train on six, score on two. Cheap, and it converts
   "adapts" into "generalises".
3. **Resolve the PPO question or close it.** If `g_long` transfers, the drift explanation
   stands and PPO is recoverable. If it does not, record that analytic gradients beat RL for
   this problem class and stop spending on it.
4. **A second surrogate architecture.** Everything assumes the transformer. Whether the
   result is a property of the method or of this model class is unknown.

## Operational notes

- **The cluster scores, the desktops train.** 24 mi2101x nodes complete a full sweep in one
  ~90 minute pass at 0.1x charge; a desktop box does exactly one arm in that time. Measured:
  0.149 charged node-hours per policy.
- Backends validated against each other: CUDA desktop, MI210/HIP cluster, RX 9070 XT/HIP,
  A100/CUDA (Euler). Baseline agreement within 0.35%, episode flip rate 0.00% against the
  fleet's own 0.93% across CUDA builds. One box runs ~2% high; compare arms on one machine.
- Scoring is the bottleneck, not training. A fine-tune is 55 seconds; its verdict is 70-90
  minutes.
