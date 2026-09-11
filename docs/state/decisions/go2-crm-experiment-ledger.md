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
| `w_h25_dw1` | 75 | **-44.1%** | 69/75 | -- | <1e-4 |
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
| `e_sig05` | 77 | **-1.7%** | 42/77 | -- | 0.345 |
| `f_pess` | 77 | **+5.8%** | 37/77 | -- | 0.014 |
| `r_rsl_full` | 77 | **+10.2%** | 33/77 | 105% | 0.001 |
| `f_all` | 75 | **+15.5%** | 32/75 | -- | 0.005 |
| `e_g96` | 73 | **+20.0%** | 28/73 | 175% | 0.004 |
| `w_r1` | 75 | **+39.6%** | 22/75 | 428% | <1e-4 |
| `f_h15` | 77 | **+63.3%** | 5/77 | -- | <1e-4 |

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

**Reinforcement learning works, once it is constrained in the right space.** Unanchored
PPO on the same reward in the same model is insignificant at best (-4.6%, p 0.33; -8.9%,
p 0.26) and significantly WORSE at full reward (+10.2%, p 0.001). Adding a KL anchor to
the base policy makes it significant: `z_ppoA30` (KL 0.30) -9.5%, `z_ppoA03` (KL 0.03)
-22.4%, both p<1e-4. Lighter anchor wins, so the sweep continues below 0.03.

The diagnostic value is in WHICH constraint worked, because two were tried:

- *Trust horizon* -- that credit assignment reaches past the ~0.5 s the model supports.
  FALSIFIED. Removing the time-out bootstrap does nothing (+1.3%, p 0.73), combining it
  with a matched discount does nothing (+3.1%, p 0.20), and a matched discount alone is
  significantly worse (+20.0%, p 0.006). The monotone pattern that motivated this spanned
  three different implementations and was a coincidence, not a causal ordering. This was
  predicted before it was measured, and the prediction was wrong.
- *Policy-space drift* -- that the policy walks away from the behaviour the model was fit
  on. CONFIRMED by the KL anchor, which constrains exactly that and is the only thing that
  moved PPO. But note WHICH drift control worked, because the other one did not:

**Ensemble pessimism does not transfer, and its surrogate-internal evidence was misleading.**
Pessimism was adopted because it cut post-peak decay roughly 7x INSIDE the surrogate, and
`f_all` was the first PPO arm that never turned over there at all. Scored in Chrono all
four arms in that family are worse than base: `e_sig05` -1.7% (p 0.35), `f_pess` +5.8%
(p 0.014), `f_all` +15.5% (p 0.005), `f_h15` +63.3% (5 of 77 episodes won). The arm with
the cleanest internal training curve is the second worst of the group.

This is the same lesson as the verification baselines: a suspiciously well-behaved internal
metric is evidence about the instrument before it is evidence about the method. Not turning
over inside the model means the optimiser stopped finding things the model rewards, which
is equally consistent with the model having run out of exploitable error and with the
policy having stopped improving. It cannot distinguish them, so it should never have been
read as progress.

The two mechanisms differ in what they measure and that is why one transferred. Pessimism
penalises ENSEMBLE DISAGREEMENT, a proxy for where the model is uncertain -- and the
attribution result shows the binding error is a near-constant 8-9% bias present for both
policies everywhere, which is precisely the error an ensemble cannot see, since all members
share it. The KL anchor penalises DISTANCE FROM THE BASE POLICY, which is measured on the
policy and needs no model at all.

These are two different horizons and only one binds RL. The analytic objective
differentiates THROUGH the model, so its limit is in time and the 0.50 s measurement sets
it. PPO samples, so its limit is distance in policy space, which no rollout length
controls. Anchored PPO reaches about half the analytic gain from identical inputs; that
gap is what differentiating through a known model buys over sampling against it.

**The residual is model error, not an optimisation shortfall.** Attribution against
commanded velocity, both policies, same episodes:

| | commanded | surrogate says | Chrono gives |
|---|---|---|---|
| base | 0.4386 | 0.3908 (89%) | 0.3565 (81%) |
| fine-tuned | 0.4386 | 0.4657 (**106%**) | 0.4282 (98%) |

Stable from 0.5 s to 1.5 s of rollout. Inside the surrogate the fine-tuned policy already
OVERSHOOTS the command, so there is no optimisation gap left to close against this model.
The surrogate carries a near-constant 8-9% optimistic velocity bias for BOTH policies --
constant across policies, so not exploitation and not accumulated drift, but a fixed
offset in what the model believes the soil returns. That is the kind of error better model
capacity can remove, which is why the current work is on the surrogate and not the
optimiser. Run with `scripts/evaluation/attribute_tracking_error.py`.

**One terrain.** Everything here is CRM. Transfer to rigid ground is the case study's
actual claim and is untested.

**The stop criterion has never been checked against Chrono.** Every rule in use --
fixed ||dW||, best surrogate-internal reward, fixed budget -- selects one iterate with an
instrument that cannot see Chrono, and a rule that returns one point cannot report that
the run peaked early or was still improving at the end. `--ckpt-every` plus
`scripts/evaluation/score_trajectory.sh` turn this into a measurement; the first run
(`t_traj`, the pinned config with the dw stop removed, 1500 updates, every 100th iterate
kept) asks whether ||dW||=1.0 at update ~94 is where the Chrono optimum actually sits.

**No held-out command families.** All eight appear in both the branch pool and the scoring
set, so this measures adaptation, not generalisation to unseen commands.

## In flight

| what | where | answers |
|---|---|---|
| `abl_l3`, `abl_w128` | a3 | does LESS surrogate capacity hurt, and how fast? |
| `abl_l12`, `abl_w512` | sbel | does MORE capacity remove the 8-9% velocity bias? |
| `abl_ctx32`, `abl_ctx64` | sliger | is the 1.28 s context window doing any work? |
| `z_ppoA01`, `z_ppoA003`, `z_ppoA03b25`, `z_ppoA03L` | sbel | where does the KL anchor bottom out, and does it compose with branch length and budget? |
| `t_traj` | sbel | is ||dW||=1.0 the right place to stop, measured rather than assumed? |

## Next, in priority order

1. **Rigid terrain.** The single largest untested claim. The surrogate, the recipe and the
   scoring harness all exist for it; what is missing is a rigid corpus run through the same
   pipeline and a rigid verdict set.
2. **Held-out command families.** Train on six, score on two. Cheap, and it converts
   "adapts" into "generalises".
3. **Resolve the PPO question or close it.** If `g_long` transfers, the drift explanation
   stands and PPO is recoverable. If it does not, record that analytic gradients beat RL for
   this problem class and stop spending on it.
4. **Model capacity and context, now running.** Promoted from last to concurrent by the
   attribution result: with the optimiser at 106% of command inside the model, the 8-9%
   bias is the binding constraint and capacity is the first lever to try against it. Six
   arms in flight varying depth, width and context against an otherwise identical
   baseline. A genuinely different model CLASS (not just a resized transformer) remains
   untested and is the honest version of this question.
5. **Evaluate the fine-tuned policy in its ORIGINAL simulator.** Everything here is scored
   in Chrono, the domain fine-tuned toward. Whether the transplant degrades the policy on
   the rigid ground it was trained on is a real blind spot, and needs a Genesis or Isaac
   install -- neither is on the fleet today.

## Operational notes

- **The cluster scores, the desktops train.** 24 mi2101x nodes complete a full sweep in one
  ~90 minute pass at 0.1x charge; a desktop box does exactly one arm in that time. Measured:
  0.149 charged node-hours per policy.
- Backends validated against each other: CUDA desktop, MI210/HIP cluster, RX 9070 XT/HIP,
  A100/CUDA (Euler). Baseline agreement within 0.35%, episode flip rate 0.00% against the
  fleet's own 0.93% across CUDA builds. One box runs ~2% high; compare arms on one machine.
- Scoring is the bottleneck, not training. A fine-tune is 55 seconds; its verdict is 70-90
  minutes.
