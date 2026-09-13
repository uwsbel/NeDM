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
-22.4%, both p<1e-4. Of the two anchors tried on that path, the lighter won.

**The follow-up anchor sweep was confounded and its conclusion is withdrawn.** `z_ppoA03` and `z_ppoA30` ran with `--objective ppo`, `--det-every 50`, `--branch-steps 25`. The follow-ups `z_ppoA01` (0.01), `z_ppoA003` (0.003) and `z_ppoA03b25` ran with `--objective rslrl` and `--det-every 200`. All three failed to improve on their starting point -- best deterministic reward at update 1 -- and that was briefly read as the anchor having a floor at 0.03. It is not evidence for that: those arms differ from A03 in the optimiser IMPLEMENTATION, not just in `kl-base`. What they do show, consistently with `r_rsl_full` (+10.2%), is that the rsl_rl path fails at every anchor weight tried, while the hand-written PPO path responds to one. `y_a03rep` / `y_a010` / `y_a001` / `y_a0003` re-run the sweep with A03's exact
configuration, varying only `kl-base`, with A03 itself reproduced as the anchor point.

The confound diagnosis is now confirmed from the other side. `y_a001` uses kl-base 0.01,
the same weight as the withdrawn `z_ppoA01`, and on this path it TRAINS: best deterministic
reward +1.3077 at update 1500, where the rslrl arm at the identical anchor peaked at update
1. The anchor weight was never the problem.

Surrogate-internal results are monotone in the anchor, and that is the warning rather than
the finding:

| arm | kl-base | internal reward | best update | final \|\|dW\|\| |
|---|---|---|---|---|
| `y_a001` | 0.01 | +1.3077 | 1500 | 6.81 |
| `y_a03rep` | 0.03 | +1.2993 | 1500 | 5.82 |
| `y_a010` | 0.10 | +1.2785 | 1450 | 5.09 |

A looser anchor scores higher inside the model and travels further from the data it was fit
on. That is the exact shape of model exploitation, and an internal metric cannot tell it
apart from genuine improvement -- the same confusion that made ensemble pessimism look
protective. RESULT, and the prediction held. Scored on the full split against the same base:

| arm | kl-base | vs base | wins | p | internal reward | final \|\|dW\|\| |
|---|---|---|---|---|---|---|
| `y_a010` | 0.10 | -11.0% | 57/77 | <1e-4 | +1.2785 | 5.09 |
| `y_a03rep` | 0.03 | **-22.1%** | 60/75 | <1e-4 | +1.2993 | 5.82 |
| `y_a001` | 0.01 | -4.4% | 53/74 | **0.84** | **+1.3077** | **6.81** |

Three things follow.

**The anchor has an interior optimum at 0.03.** Not a floor to sweep past and not a
monotone knob: 0.10 is too tight to help much, 0.01 is too loose to help at all.

**`y_a03rep` reproduces `z_ppoA03` to within 0.3 points** (-22.1% against -22.4%), so the
configuration and the original result are both sound.

**Model exploitation is now demonstrated rather than hypothesised.** `y_a001` won EVERY
surrogate-internal number -- highest deterministic reward of the three -- while travelling
furthest from the data the model was fit on, and in Chrono it is statistically
indistinguishable from the unmodified base policy (p 0.84). Looser anchor, better internal
score, more drift, no real gain. That is the whole mechanism in one arm.

Note what this does to the claim withdrawn above. "0.03 is a floor, lighter is worse" turns
out to be TRUE, but the evidence offered for it at the time was confounded and worthless --
those arms differed in optimiser implementation, not anchor weight. Right conclusion,
invalid reasoning. It is recorded twice on purpose: a claim that later proves correct does
not retroactively make the bad evidence good, and the withdrawal was still the right call.

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

**The 27.88 h corpus is gravity-randomised, not rigid, and its metadata is thin.** Worth
stating because it has been referred to as "rigid terrain data we already have", including
by me, and that is wrong twice over.

Every episode carries a DIFFERENT gravity vector: 30 distinct vectors in 30 sampled, all of
magnitude 9.810 but tilted up to about 3 degrees off vertical, recorded in three
`grav_shuf_*` columns the CRM corpus does not have. A surrogate fitted to it learns
locomotion under a tilting gravity field. That is its own experiment, not a rigid-ground
analogue of the CRM work.

Its records are also far thinner. Measured, not assumed:

| | deformable soil | gravity-randomised flat |
|---|---|---|
| recorded hours | 1.55 | 27.88 |
| CSV columns | 174 | 177 (superset) |
| key signals populated | 100% | 92-100% |
| **per-episode metadata fields** | **56** | **9** |
| fall / divergence flags | recorded | absent |
| commanded velocities per episode | recorded | absent (family recoverable from the name) |
| git provenance | recorded | absent |
| `foot_force_source` | recorded (FSI) | absent |
| impossible joint jumps, sampled | 2 of 388 | 0 of 120 |
| truncated episodes | common, bed-limited | none seen |

It cuts both ways. The larger corpus is physically CLEANER -- no diverged solves in the
sample and no truncation, because there is no particle bed to walk out of. What it lacks is
the record. On CRM two blown-up solves were caught because the flags and checks existed;
here health had to be measured directly, since there is nothing to look up. A separate
`basefail_index.json` holds 416 base-controller failures and those are already excluded
from the main index, so that part was handled correctly when it was built.

**One terrain.** Everything here is CRM. Transfer to rigid ground is the case study's
actual claim and is untested.

**Neither stop rule finds the Chrono optimum.** MEASURED, on `t_traj`: the pinned config
with the displacement stop removed, 1500 updates, every 100th iterate kept, eight of them
scored in Chrono on an identical 22-episode head of the split.

| update | mae_vx | vs base | wins | p |
|---|---|---|---|---|
| 1 | 0.1474 | -0.8% | 11/23 | 0.66 |
| 200 | 0.0857 | -39.2% | 18/22 | <1e-4 |
| 400 | 0.0857 | -39.2% | 17/22 | <1e-4 |
| 600 | 0.0814 | -42.3% | 19/22 | <1e-4 |
| **800** | **0.0768** | **-45.5%** | 19/22 | <1e-4 |
| 1000 | 0.1122 | -20.4% | 17/22 | 0.63 |
| 1200 | 0.1093 | -22.5% | 18/22 | 0.18 |
| 1500 | 0.0880 | -37.6% | 19/22 | 0.001 |

The recipe stops at ||dW|| 1.0, which happens near update 94. The surrogate-internal
metric, run to the full budget, selected update 1500. On the 22-episode subset the best iterate was update 800; on the full split that arm is
-41.9% against the pinned arm's -40.3%. See the confirmation below before reading the
subset numbers as results. Both rules were plausible and both are
wrong, which is the argument for keeping the trajectory rather than trusting any rule.

Note the shape, not just the peak. Update 1000 wins 17 of 22 episodes yet has p 0.63:
the median improves while a few episodes fail badly enough to carry the mean. Mid-training
iterates are not uniformly worse, they are higher VARIANCE, and a rule reading a single
scalar cannot see that distinction either.

CONFIRMED ON THE FULL SPLIT, and the confirmation cut the headline down. Updates 800 and
600 re-scored on all 80 episodes, paired per episode against the same base:

| arm | n | mae_vx | vs base | wins | p |
|---|---|---|---|---|---|
| update 800 | 75 | 0.0910 | **-41.9%** | 67/75 | <1e-4 |
| update 600 | 75 | 0.0943 | **-39.7%** | 66/75 | <1e-4 |
| `w_h15r0`, pinned, stops ~update 94 | 75 | 0.0937 | **-40.3%** | 68/75 | <1e-4 |

Update 800 read -45.5% on the 22-episode subset and -41.9% on the full split. The subset
inflated the arm it had been used to select by 3.6 points, which is the whole reason the
confirmation step exists and is worth restating: a shape measurement cannot also be the
number, ever.

So the corrected reading of this experiment, which is weaker than the first pass:

- The displacement stop is NOT leaving large gains on the table. It lands within 1.6
  points of the best iterate sampled, and adjacent iterates differ by 2.2 points (800 vs
  600), so 1.6 sits inside iterate-to-iterate noise. "Stops eight times too early" was a
  statement about update NUMBER that did not survive contact with the Chrono scores.
- What does survive is narrower: no stop rule tested tracks the Chrono optimum, and the
  surrogate-internal metric run to a full budget is the worst of them, selecting update
  1500 (-37.6% on the subset, never re-scored at full split). Keeping the trajectory is
  worth it as a DIAGNOSTIC -- it is how the 3.6-point subset inflation and the flat region
  between updates 200 and 800 became visible at all -- not as a source of headroom.
- The variance observation stands and is arguably the most useful part. Update 1000 wins
  17 of 22 episodes at p 0.63: the median improves while a few episodes fail badly enough
  to carry the mean. Mid-training iterates are higher variance, not uniformly worse, and no
  scalar stop rule can see that.

**The stop criterion, before this measurement.** Every rule in use --
fixed ||dW||, best surrogate-internal reward, fixed budget -- selects one iterate with an
instrument that cannot see Chrono, and a rule that returns one point cannot report that
the run peaked early or was still improving at the end. `--ckpt-every` plus
`scripts/evaluation/score_trajectory.sh` turn this into a measurement; the first run
(`t_traj`, the pinned config with the dw stop removed, 1500 updates, every 100th iterate
kept) asks whether ||dW||=1.0 at update ~94 is where the Chrono optimum actually sits.

**GENERALISATION IS WITHIN BEHAVIOUR CLASS, NOT ACROSS IT.** Two command families held out
of the fine-tune's branch pool, scored on all eight, three seeds per set. The two sets
disagree completely, which is the finding:

| held out of the branch pool | seen families | held-out families | gap | p |
|---|---|---|---|---|
| `pivot`, `weave` | -30.5% | -31.7% | **-1.2 pts** | 0.85 |
| `lateral`, `stop_and_go` | -42.6% | **-14.5%** | **+28.1 pts** | <1e-4 |

Two sets rather than one was the point. Either alone would have supported a confident and
wrong conclusion -- set A that the fine-tune generalises freely, set B that it does not
generalise at all.

What separates them is whether a behavioural NEIGHBOUR survives in the pool. Holding out
`pivot` and `weave` leaves `arc` and `yaw_step`, which are also turning behaviours, and
generalisation is perfect. Holding out `lateral` and `stop_and_go` leaves nothing comparable:
`lateral` is the only family with dominant sideways velocity and `stop_and_go` the only one
with velocity discontinuities. Generalisation then collapses to about a third of the
seen-family gain.

So the honest form of the headline claim: the fine-tune ADAPTS to the commands in its branch
pool and generalises to unseen commands only when a behaviourally similar family is present.
That is weaker than "-40% tracking error" reads, and it is a design instruction rather than
a limitation -- a branch pool needs coverage of behaviour CLASSES, not of command count.

This is a hypothesis fitted to two data points and it should be tested by holding out a
family whose neighbour is explicitly present or absent by construction. It is not proven by
these two sets; it is the only account consistent with both.

**The first capacity ablation tested capacity at one learning rate, which is not the same
thing.** Every arm held `lr=3e-4` and `warmup=1000`, the values tuned for the 6x256
baseline. Both larger arms then peaked almost immediately and degraded for the rest of
training:

| arm | n_layer x n_embd | best rollout_sel | at epoch |
|---|---|---|---|
| `abl_l12` | 12 x 256 | 1.124 | **1** of 80 |
| `abl_w512` | 6 x 512 | 0.652 | **4** of 80 |
| `abl_w128` | 6 x 128 | 0.601 | 26 |
| `abl_l3` | 3 x 256 | 0.701 | 33 |
| `baseline_s1` | 6 x 256 | **0.524** | 29 |

Peaking at epoch 1 of 80 is what a learning rate too high for the model looks like, not
what a model failing to benefit from capacity looks like. The smaller arms, which converge
normally, are informative; the larger two are not, and "bigger is worse" must not be read
off this table. `l12lr1` / `l12lr03` / `w512lr1` / `w512lr03` re-run the two larger arms at
a third and a tenth of the baseline LR with warmup lengthened in proportion. If they still
fail to beat 6x256 at their own best LR, capacity genuinely is not the lever and the
attribution result needs a different response.

**Less data trains a better surrogate, and that needs explaining before it is believed.**
`dq25` at 25% of the corpus reaches rollout_sel 0.480 against the full corpus baseline's
0.524, on byte-identical processed data (same metadata md5, same 17 shards). Taken at face
value the model is not data-limited and collecting more of the same buys nothing. The obvious confound is ruled out. `steps_per_epoch` is fixed in config and the sampler
draws `steps_per_epoch x batch_size` samples per epoch WITH REPLACEMENT, so all four arms
take exactly **160,000 gradient steps** (80 x 2000). `train_episode_fraction` changes only
how many distinct episodes those steps draw from; `dq25` is not under-trained, it is
equally trained on a quarter of the data. Same seed, same shards, same schedule.

What is NOT ruled out is the instrument. `rollout_sel` is a surrogate-internal open-loop
score, and today produced two separate cases of an internal number pointing the wrong way:
the pessimism training curve, and the trajectory validation pick. The result that would
settle this is a Chrono one -- fine-tune a policy in the `dq25` surrogate and score it
against the same policy fine-tuned in `baseline_s1`. Until then the claim is that the model
is not data-limited ON ITS OWN METRIC, which is weaker than it sounds.

`dq50` and `dq75` fill in the curve.

**The surrogate selection metric is too noisy to rank anything from one run, and
`best_val.pt` is chosen by its minimum.** This invalidates several comparisons recorded
above and is the most important methodological finding of this line of work.

`rollout_sel` is computed on 12 rollout episodes, and within a single run it swings by a
factor of 2 to 7 between consecutive epochs -- `baseline_s1`'s last five epochs read 1.676,
3.574, 2.773, 3.038, 2.206. Its reported "best" of 0.524 is the MINIMUM OF 80 NOISY DRAWS,
a statistic dominated by how noisy the run was rather than by how good the model is. The
trainer then saves `best_val.pt` at that epoch, so every surrogate checkpoint in this
project is the luckiest of 80 twelve-episode evaluations, not the best model.

Re-ranked on the median of the last 40 epochs, which is what should have been compared:

| run | min(80) as reported | median last 40 | IQR |
|---|---|---|---|
| `dq25` | 0.480 | **0.694** | **0.142** |
| `dq50` | 0.583 | 1.015 | 0.278 |
| `abl_w512` | 0.652 | 1.109 | 0.208 |
| `abl_w128` | 0.601 | 1.429 | 0.466 |
| `abl_l3` | 0.701 | 1.525 | 0.743 |
| `baseline_s3` | 1.039 | 1.819 | 0.691 |
| `dq75` | 0.779 | 2.054 | 0.824 |
| `baseline_s1` | **0.524** | **2.173** | **1.679** |
| `abl_l12` | 1.124 | 2.346 | 0.751 |

What changes:

- **`baseline_s1` won on min by being the noisiest run in the set.** Its median is the
  worst of all nine and its IQR is twelve times `dq25`'s. The surrogate every experiment in
  this document was built against was selected this way.
- **"Bigger is worse" was wrong for width.** `abl_w512` beats the baseline on median
  (1.109 against 2.173) while the min statistic said the opposite. Depth still looks bad.
- **`dq25` survives and strengthens.** Best median AND tightest IQR, so it is not a
  min-of-noise artefact. Less data trains a more stable surrogate here, at matched gradient
  steps.

Two fixes follow, neither yet applied: raise `rollout_eval.num_episodes` well above 12 so
a single evaluation means something, and select checkpoints on a smoothed metric rather
than a raw minimum. Until both land, no single-run surrogate comparison in this document
should be treated as settled, including the capacity and context ablations.

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

**Capacity is not the lever for the bias, and the LR hypothesis was wrong.** Two
corrections, both against things recorded above.

The claim that the larger arms needed a lower learning rate is FALSIFIED for width. At the
baseline LR the wide model is the best and most stable arm measured; at a third of it, it
falls apart:

| arm | lr | median last 40 | IQR |
|---|---|---|---|
| `abl_w512` | 3e-4 | **1.109** | **0.208** |
| `w512lr1` | 1e-4 | 2.692 | 2.799 |
| `l12lr1` | 1e-4 | 2.169 | 1.071 |
| `baseline_s1` | 3e-4 | 2.173 | 1.679 |
| `abl_l12` | 3e-4 | 2.346 | 0.751 |

So `abl_w512` was never badly trained. Its "best at epoch 4" was the min-of-noise artefact,
nothing more, and read by median it halves the baseline's open-loop error at an eighth of
its spread. Width helps, depth is a wash, and the peak-early symptom meant the metric was
noisy rather than the run being broken.

**SOIL DIVERSITY REDUCES THE BIAS, and capacity helps too.** CORRECTING THE PREVIOUS
ENTRY, which was measured on the wrong checkpoint. The copy of the bias script on sbel was
the pre-patch version reading `best_val.pt`, so those numbers were lucky-epoch draws rather
than the `last.pt` comparison they were reported as. Re-run properly, on `last.pt` for
every arm, identical episodes, identical Chrono truth of 0.2814:

| surrogate | trained on | surrogate says | over-promise | vs baseline |
|---|---|---|---|---|
| `baseline_s1` | 1.55 h, one soil | 0.3271 (92.5%) | **+0.0458** | -- |
| `abl_w512` | 1.55 h, one soil, 2x width | 0.3190 (90.2%) | **+0.0376** | -18% |
| `go2_crm_soil` | **0.96 h, soil 0.5x-1.4x** | 0.3112 (88.0%) | **+0.0298** | **-35%** |

Chrono delivers 79.5% of command. Every model still over-promises; the soil-varied one
does so by a third less than the baseline, on two thirds the data.

The two checkpoints disagree about the SIGN of the capacity effect -- on `best_val` the
wide model looks worse than baseline, on `last.pt` it is better -- which is the cleanest
demonstration yet of why selecting on the minimum of a noisy metric had to go, and why the
bias ranking now defaults to `last.pt`.

What stands, carefully:

- **The collection campaign paid off.** Varying the one axis the corpus never varied cut
  the binding residual by about a third, using LESS data than the baseline. This is the
  first intervention to move the bias at all.
- **Open-loop accuracy still dissociates from the bias.** The soil surrogate is WORSE
  open-loop (median 2.842 against the baseline's 2.173) and better on the bias. Fitting the
  data well and being unbiased about what the soil returns are different things, and
  `rollout_sel` measures the former.
- **Caveats.** One run per arm at 24 episodes. The 35% gap is far larger than the ~10%
  that was inside noise for the earlier best_val comparison, so it is probably real, but a
  seed repeat would settle it. The soil arm also differs in data QUANTITY (0.96 h against
  1.55 h), so it is not a pure diversity contrast -- though since `dq25` showed less data
  is not worse, that confound makes the result more impressive rather than less.

**SEED REPLICATION OVERTURNS THE REVERSAL, AND THE STATISTIC THAT PRODUCED IT.** Three
independent fine-tune seeds per surrogate, each scored on the full 80-episode split and
paired per episode:

| seed | soil mae_vx | control mae_vx | soil - control | soil wins | p |
|---|---|---|---|---|---|
| 0 | 0.0971 | 0.0914 | **+6.3%** worse | 33/74 | 0.0117 |
| 1 | 0.0851 | 0.0922 | **-7.7%** better | 51/75 | 0.0001 |
| 2 | 0.0850 | 0.0931 | **-8.7%** better | 50/75 | 0.0034 |
| **pooled** | | | **-3.4%** better | 134/224 | **0.0206** |

Two findings, and the second matters more than the first.

**Soil diversity does not hurt transfer.** The earlier entry -- less bias, worse transfer --
rested on seed 0 alone. Two further seeds reverse it and the pooled evidence favours the
soil surrogate. That conclusion is withdrawn.

**Every seed pair is individually significant and they disagree about the sign.** p 0.0117,
0.0001 and 0.0034, pointing +6.3%, -7.7%, -8.7%. The paired-episode bootstrap treats
episode variation as the ONLY noise source; it is structurally blind to seed-to-seed
variance in the fine-tune, which is roughly +/-8% here. So a per-seed p-value cannot
establish which surrogate is better no matter how small it gets, and one was used to
overturn a published result. Any future surrogate-vs-surrogate claim in this document needs
replicate SEEDS, not more episodes.

REFINEMENT, and it corrects the paragraph above as well as an over-broad audit that
followed it. The seed spread is NOT a uniform property of the fine-tune. It is a property
of the SURROGATE being fine-tuned in:

| surrogate | three fine-tune seeds | spread | sd |
|---|---|---|---|
| `baseline_s1` | -41.6%, -41.1%, -40.5% | **1.1 pts** | **0.43** |
| `go2_crm_soil` | -37.3%, -45.6%, -45.7% | **8.4 pts** | **3.92** |

**REPRODUCIBILITY IS A PROPERTY OF THE SURROGATE.** The baseline surrogate yields fine-tunes
that land within a point of each other across seeds. The soil surrogate is nine times more
variable: two of its seeds reach -45.6% and -45.7%, clearly better than anything the
baseline produces, and one reaches -37.3%, clearly worse.

That changes the reading in three ways:

- **Soil is better on average, by about 1.8 points** (-42.9% mean against -41.1%), which
  agrees with the pooled paired test. The headline is not the mean though, it is the
  variance.
- **The soil surrogate's best fine-tunes beat the baseline's best.** -45.7% is the lowest
  tracking error any arm in this document has reached. Whatever soil diversity buys, it is
  available but not reliably so.
- **The audit that followed the previous entry was alarmist.** Applying the soil surrogate's
  +/-8% spread to every arm suggested 268 arm pairs were indistinguishable. With the
  baseline surrogate's actual sd of 0.43, the rankings among baseline-surrogate arms stand.
  What needs replicate seeds is any comparison INVOLVING the soil surrogate, or any
  surrogate whose seed stability has not been measured.

A surrogate that sometimes produces an excellent policy and sometimes a mediocre one is a
different kind of object from one that reliably produces a good one, and the distinction is
invisible to a single run. Measuring it costs three fine-tunes, which is 55 seconds each.

What survives, stated at the strength the evidence supports: the soil-varied surrogate is
probably mildly better for transfer, by about 3% of control, with a spread across seeds
larger than the effect. It is NOT the case that a 35% reduction in velocity bias bought a
proportional transfer gain -- the bias moved a lot and transfer moved a little, so the two
remain loosely coupled at best. Three seeds is enough to kill the reversal, not enough to
size the benefit.

**THE COMPUTE-LIMITED HYPOTHESIS IS LARGELY FALSIFIED, so "more data is worse" stands.**
At epoch 247 of 320, on the quieter 32-episode evaluation:

| arm | data | gradient steps | median rollout_sel | IQR |
|---|---|---|---|---|
| `qdq25` | 25% | 160k | **0.755** | 0.220 |
| `qbase4x` | 100% | **640k** | 2.305 | 0.387 |
| `qbase` | 100% | 160k | 2.652 | 0.585 |

Quadrupling the budget buys the full corpus 13% (2.652 -> 2.305) and leaves it three times
worse than a quarter of the data at a quarter of the compute. The obvious explanation --
that a fixed `steps_per_epoch` means the full corpus simply gets fewer passes per episode
and is undertrained -- does not survive.

So something about using less of THIS corpus is better, and two explanations call for
opposite responses:

- **a subset of the corpus is harmful** and `dq25`'s particular draw happens to miss it. Then
  the corpus is salvageable and the work is to find those episodes.
- **the fraction is what matters** for any draw, which would make it a fact about
  optimisation dynamics on this data rather than about bad episodes, and hunting for
  culprits would waste effort.

`qdq25b` and `qdq25c` take a different 25% under seeds 101 and 202. If they also land near
0.755 it is the fraction; if they land near 2.6 the first draw was lucky and there are
specific episodes to find. This needs no new code, which is why it is the next test rather
than a corpus audit.

**RIGID DATA IMPROVES THE CRM SURROGATE, on its open-loop metric.** Both arms at the
32-episode evaluation with the smoothing window, differing only in whether level rigid
ground is mixed in:

| surrogate | trained on | median rollout_sel | IQR |
|---|---|---|---|
| `mixctl` | CRM only | 2.767 | 0.738 |
| `mixed` | **CRM + rigid, 50/50, terrain-conditioned** | **2.380** | 0.942 |

A 14% reduction. For scale, two CRM-only surrogates at different seeds (`mixctl` 2.767,
`qbase` 2.652) differ by 4%, so the gap is larger than surrogate-seed noise -- though with
one run each that is an argument, not a measurement.

Why this matters beyond the number: rigid ground costs about 1,900 episodes per charged
node-hour against CRM's 200, and records 92% of each episode against CRM's ~25%, because
there is no SPH solver and no particle bed to walk out of. 800 rigid episodes gave 6.04 h
for roughly 0.4 node-hours. If the cheap terrain improves the expensive one, the collection
economics of this line of work change: the answer to "we need more data" stops being "CRM
is too expensive" and becomes "collect the terrain that is cheap".

Design choices worth recording, since both could have invalidated it. The mix is 50/50
rather than the 75/25 an older flat+crm config used -- that config targeted flat, this one
targets CRM, and the rigid corpus is four times larger, so 75% would let rigid dominate the
gradient. And validation and checkpoint selection are CRM-only, since letting rigid into the
selection metric rewards a model that is good at the easy terrain.

`mixedft_s0/s1/s2` fine-tune in it against the already-scored `basedft` seeds. Three seeds
because the soil experiment showed a surrogate can be better on average and far less
reliable, and open-loop error has twice disagreed with Chrono in this document.

**AT SIX SEEDS BOTH INTERVENTIONS ARE NULL, AND THE THREE-SEED RESULT WAS UNDERPOWERED.**
This supersedes every earlier entry about soil diversity and about mixing rigid data.

| surrogate | n | seeds (% vs base) | mean | sd | best seed |
|---|---|---|---|---|---|
| `baseline_s1` | 6 | -41.6 -41.1 -40.5 -37.2 -41.3 -40.7 | **-40.4** | 1.48 | -41.6 |
| `go2_crm_soil` | 6 | -37.3 -45.6 -45.7 -31.9 -37.5 -41.7 | **-39.9** | **4.93** | -45.7 |
| `go2_crm_mixed` | 3 | -39.2 -39.5 -42.8 | **-40.5** | 1.65 | -42.8 |

Pooled paired: soil vs baseline +0.0005, p 0.62. Mixed vs baseline +0.0006, p 0.63. Neither
surrogate produces a better transplanted policy than the plain CRM one.

**The three-seed pooled test was still underpowered.** At n=3 it gave p 0.0206 favouring
soil and an entry was written around "soil helps by about 3%". At n=6 the same test gives
p 0.62 and the effect is gone. That is the third time this claim moved: one seed said soil
hurt, three said soil helped, six say no difference. The lesson is not "use more seeds", it
is that a significant p-value from the smallest sample that produces one is worth nothing,
and that was the reasoning used each time.

**Soil's variance is the only surviving effect.** sd 4.93 against the baseline's 1.48. Its
best seed reaches -45.7%, the lowest tracking error in this document, and its worst reaches
-31.9%, the worst of any working arm. Same mean, three times the spread. A surrogate with
that profile is worth something only if the best draw can be identified WITHOUT a Chrono
verdict, and nothing measured here predicts which seed will be good.

**Open-loop accuracy does not predict transfer.** The mixed surrogate is 14% better
open-loop (median rollout_sel 2.380 against 2.767) and 0% better in Chrono. Together with
the bias result -- 35% less optimistic velocity bias, no transfer gain -- that is two
independent surrogate-quality metrics that improve substantially while the transplanted
policy does not move. The quantity the trainer minimises, the model's systematic error, and
the policy's Chrono score are three loosely related things.

What this does NOT overturn: the headline. Fine-tuning in a learned surrogate and
transplanting to Chrono gives about -40% tracking error against the base policy, from every
surrogate tried, reproducibly across six seeds. The method works. What fails is the attempt
to make it work BETTER by improving the surrogate along any axis measured so far.

**ORIGINAL-DOMAIN EVALUATION: THE FINE-TUNED POLICY IS WORSE ON RIGID GROUND, AND THE
DAMAGE SCALES WITH SPEED.** The NRD author's question, finally answerable.

Not the training simulator. The policy is from `wty-yy/go2_rl_gym`, a legged_gym-family
repo, so its true training domain is Isaac Gym -- archived, developer-account download,
Python 3.8. What that repo ships is a MuJoCo sim-to-sim deploy config, and we already had it
verbatim at `checkpoints/go2_mujoco.yaml`: kp 20, kd 0.5, dt 0.002, decimation 10, and the
same scales and default angles the Chrono adapter reimplemented. MuJoCo flat ground with the
authors' own config is a genuine third domain, independent of both Chrono and the surrogate.

Six fine-tune seeds against the base policy, five commanded speeds, 8 s each:

| commanded vx | base mae_vx | fine-tuned mean | delta |
|---|---|---|---|
| 0.3 | 0.343 | 0.189 | **-44.9%** better |
| 0.5 | 0.328 | 0.338 | +3.2% |
| 0.7 | 0.290 | 0.391 | +34.8% worse |
| 0.9 | 0.258 | 0.399 | +55.0% worse |
| 1.1 | 0.165 | 0.367 | **+122.0%** worse |

Falls: base 0 of 5, fine-tuned 6 of 30. Mean survival 400/400 against 367/400.

**The dose-response is the evidence.** The degradation is monotone in commanded speed and
crosses over near 0.5 m/s, which is the mechanism showing itself rather than a single
suspicious comparison. CRM fine-tuning teaches the policy to ask MORE of the ground -- that
is precisely how it wins on soil, where the medium yields underfoot. On rigid ground there
is nothing to sink into, so the same extra command is pure overshoot, and overshoot grows
with speed.

**What this is NOT.** The obvious follow-up guess -- that the CRM evaluation only samples
the low-speed regime where the damage is invisible -- is wrong, and was checked before being
written down. CRM scoring commands have median 0.594 m/s and mean 0.709, with 45% of
episodes commanding at or above 0.7. The evaluation covers exactly the band where MuJoCo
shows the regression. So this is not a coverage artefact: the policy is genuinely better on
deformable soil across the speed range AND genuinely worse on rigid ground, which is domain
SPECIALISATION rather than a hidden defect.

That reframes the -40% headline without weakening it. The method buys a real and large gain
on the terrain it is tuned for, and the cost is paid somewhere the CRM verdict harness
cannot see by construction. Anyone deploying a surrogate-fine-tuned policy on mixed terrain
needs this measurement; nothing in the Chrono pipeline would ever surface it.

Harness: `scripts/evaluation/eval_go2_mujoco.py`. Joint order verified from actuator names
at load (menagerie's FL/FR/RL/RR happens to match the policy's, so the permutation is the
identity -- but a wrong one yields a plausible 45-vector with left and right legs swapped),
and the hip sign convention, unverifiable from the MJCF's zero-hip home pose, checked
physically by requiring a sensible settle at the policy's own defaults.

**NO SURROGATE-SIDE METRIC PREDICTS TRANSFER.** The wide surrogate closes this out. Both
arms at batch 32, three seeds each, because the wide model needs more than the 3090's
23.56 GB at the recipe's batch 64 and running only ONE arm at the smaller batch would
confound the surrogate with the batch.

| surrogate | open-loop quality | three seeds | mean | sd |
|---|---|---|---|---|
| `abl_w512` | **2x more accurate** (median 1.109 vs 2.173) | -41.4, -35.5, -31.9 | **-36.3%** | 3.93 |
| `baseline_s1` | reference | -40.9, -37.5, -42.0 | **-40.2%** | 1.91 |

Stated at the strength the evidence supports: the pooled paired-episode test gives p 0.0012,
and that number is NOT the answer. A paired-episode bootstrap is blind to seed variance,
which is the error that moved the soil claim three times in this document. At the seed level
two of three favour the baseline and the 3.9-point gap sits at about one standard deviation.
So: the twice-as-accurate surrogate does not produce a better policy, and may produce a
worse one. Three seeds cannot settle which.

The pattern across three independent interventions is what is solid:

| surrogate improvement | magnitude | effect on the transplanted policy |
|---|---|---|
| soil diversity -> less velocity bias | -35% | none, p 0.62, six seeds |
| rigid mixing -> better open-loop | +14% | none, p 0.63, three seeds |
| 2x width -> much better open-loop | 2x | none, trending worse, three seeds |

Every quantity we can compute about a surrogate WITHOUT running Chrono -- its systematic
bias, its open-loop reconstruction error, and the model capacity that drives both -- is
uninformative about how well a policy fine-tuned inside it will transplant. That is a
negative result about the whole improve-the-model programme, and it is more useful than any
of the individual arms would have been, because it says where not to spend effort.

It also leaves the real question open and sharper: something determines whether a given
surrogate yields a -45.7% policy or a -31.9% one, and nothing measured here is it. The
variance findings are the only lead -- the baseline surrogate yields fine-tunes at sd 1.48
while soil yields sd 4.93 and the wide model sd 3.93, so whatever the property is, it shows
up as reproducibility rather than as accuracy.

**THE ANCHOR DOES NOT CONTROL ORIGINAL-DOMAIN RETENTION; THE SIZE OF THE CRM GAIN DOES.**
Prediction logged before the run and falsified by it.

The KL anchor penalises distance from the base policy in policy space, so it looked like it
should also bound how much of the base policy's rigid-ground competence is forgotten -- a
free win if true. Measured in MuJoCo, five commanded speeds per policy:

| policy | CRM gain | rigid mae_vx at 0.7/0.9/1.1 vs base | falls |
|---|---|---|---|
| base | -- | -- | 0/5 |
| kl 0.10 | -11.0% | **+1.9%** | 1/5 |
| kl 0.03 | -22.1% | **+64.4%** | **4/5** |
| kl 0.01 | -4.4% | -35.2% | 0/5 |
| analytic, unanchored | -41.6% | +53.8% | 0/5 |

Retention does NOT order with anchor strength: kl 0.03 is the worst retainer in the set
despite being anchored more tightly than kl 0.01. The prediction was that tighter anchor
means less forgetting, and it is wrong.

What retention does track is HOW MUCH THE POLICY GAINED ON CRM. kl 0.01 gained essentially
nothing (-4.4%, p 0.84, statistically indistinguishable from base) and kept its rigid
competence; the arms with real CRM gains lost heavily. So this is a tradeoff rather than a
mechanism, and the anchor is not a dial that buys both.

One cell is practically useful. **kl 0.10 buys -11.0% on CRM for +1.9% on rigid**, which is
close to free, against the analytic arm's -41.6% for +53.8%. If a deployed policy must walk
on both terrains, the best CRM number is not the right choice and the anchored PPO arm
nobody favoured on the CRM leaderboard is.

CAVEAT: one run per (policy, speed) cell, and falls are stochastic -- kl 0.03's 4 of 5 is
the kind of number that can move. The broad tradeoff is visible across four policies and
five speeds; the specific ordering between kl 0.03 and the analytic arm is not established.

**THREE QUARTER-CORPUS DRAWS, ALL BEATING THE FULL CORPUS, ALL DIFFERENT FROM EACH OTHER.**
The discriminator resolves, and the answer is neither of the two options it was framed
around.

| arm | data | median rollout_sel | IQR |
|---|---|---|---|
| `qdq25` seed A | 25% | **0.755** | 0.220 |
| `qdq25c` seed C | 25% | **0.979** | 0.205 |
| `qdq25b` seed B | 25% | **1.447** | 0.577 |
| `qbase` | 100% | 2.652 | 0.585 |
| `qbase4x` | 100%, 4x compute | 2.261 | 0.410 |

Not one lucky draw: all three quarter-corpus subsets beat the full corpus decisively. Not
the fraction alone either: they differ from each other by nearly 2x, so WHICH episodes are
drawn matters substantially on top of how many.

The reading that fits: using less of this corpus is reliably better, and the specific subset
then modulates by how much. Together with 4x compute buying only 15%, that makes it a
property of the corpus rather than an optimisation artefact, and points at some episodes
being actively harmful and diluted to different degrees by each draw.

NOTE ON THE EARLIER RUN OF THIS TEST. The first attempt produced qdq25b and qdq25c identical
to qdq25 to four decimal places, because the generator wrote `config["seed"]` while the
trainer reads `config["training"]["seed"]`. Two "different" runs agreeing that precisely is
what surfaced it. The numbers above are from the reseeded runs.

**The mixed surrogate reproduces.** `mixed_s2` at 2.519 against `mixed`'s 2.380, about 5.8%
apart and within surrogate-seed spread, so the 14% open-loop gain over `mixctl`'s 2.767 is
real and not a lucky draw. It still buys nothing in transfer.

**DOES ANY SURROGATE PROPERTY PREDICT ANYTHING? ONE CANDIDATE SURVIVES A LEVERAGE CHECK.**
All four surrogates with both an open-loop profile and a measured fine-tune distribution:

| surrogate | open-loop median | open-loop IQR | transfer mean | transfer sd | seeds |
|---|---|---|---|---|---|
| `baseline_s1` | 2.173 | 1.679 | -40.4% | 1.48 | 6 |
| `go2_crm_soil` | 2.842 | 0.535 | -39.9% | 4.93 | 6 |
| `go2_crm_mixed` | 2.380 | 0.942 | -40.5% | 1.65 | 3 |
| `abl_w512` | 1.109 | 0.208 | -36.3% | 3.93 | 3 |

Two candidate relationships, and the LEVERAGE CHECK separates them:

| relationship | r, all four | r, leave-one-out range |
|---|---|---|
| open-loop accuracy -> transfer gain | -0.872 | **+0.895 without `w512`** |
| open-loop IQR -> fine-tune sd | -0.798 | -0.723 to -0.895, sign stable |

**REJECTED: accuracy anti-predicts transfer.** At face value r -0.872 says the more accurate
the surrogate the worse the policy, which would have been a striking headline. It is one
point. Drop `abl_w512` and the correlation reverses to +0.895. Not claimable, and it is the
kind of number that would have been reported if the leverage check had not been run.

**TENTATIVE, and the only survivor: a surrogate whose own evaluation is MORE stable produces
fine-tunes that are LESS reproducible.** Negative under every single-point deletion. If real,
it is useful in a way none of the accuracy metrics were, because fine-tune reproducibility is
the property that actually distinguished the surrogates -- but it is backwards from the
obvious guess, that a well-behaved model gives well-behaved optimisation.

With n=4 even a leave-one-out-robust correlation is weak. The `lco93/94/95/96` runs will add
four more surrogates; three fine-tune seeds in each takes this to n=8 and costs 12 fine-tunes
at 55 s plus one scoring pass. That is the cheapest way to find out whether this is the first
real predictor in the whole programme or the second artefact.

**NO SINGLE COLLECTION IS THE CULPRIT; THE EFFECT IS IN THE FRACTION.** Leave-one-out
across all four source collections, each arm keeping 75% of the corpus:

| arm | drops | median rollout_sel | IQR |
|---|---|---|---|
| `lco94` | `go2_crm_s9400000` | 2.183 | 0.293 |
| `lco93` | `go2_crm_s9300000` | 2.191 | 0.575 |
| `lco96` | `go2_crm_s9600000` | 2.523 | 0.701 |
| `lco95` | `go2_crm_s9500000` | **2.885** | 0.390 |
| `qbase` | nothing, 100% | 2.652 | 0.585 |
| `qdq25` | random 75%, keeps 25% | **0.755** | 0.220 |

Grouped by how much data survives:

| fraction kept | n | range | mean |
|---|---|---|---|
| 100% | 2 | 2.261 - 2.652 | 2.457 |
| 75% | 4 | 2.183 - 2.885 | 2.446 |
| 25% | 3 | 0.755 - 1.447 | 1.060 |

Removing a quarter of the corpus changes essentially nothing -- 2.446 against 2.457, and one
arm is worse than the full corpus. Removing three quarters changes everything. If specific
episodes were poisoning training, dropping a whole collection would find them, and no
collection-level cut comes within 1.4 of what a random quarter achieves.

So the harmful-subset hypothesis is REJECTED at collection granularity. Combined with 4x
compute recovering only 15%, the surviving explanation is about optimisation dynamics on
this corpus rather than about bad data: something in fitting the full distribution is worse
than fitting a quarter of it, at matched gradient steps.

That is unexplained and it is now the most interesting open question in this line of work,
because it inverts the assumption the whole collection effort rested on. Two waves of
collection were run on the premise that more and more varied data makes a better surrogate.
On this terrain, at this model size and step budget, it does not.

A per-episode cut is the next granularity, but the collection-level null argues against
spending on it: harm spread evenly enough that removing 187 episodes does nothing while
removing 476 random ones transforms the model does not look like a set of bad episodes.

**THE LAST CANDIDATE PREDICTOR DIES AT n=8.** Twelve fine-tunes in the four leave-one-out
surrogates, three seeds each, scored on the full split.

| arm | drops | surrogate median | three transfer seeds | mean | sd |
|---|---|---|---|---|---|
| `lco93` | `s9300000` | 2.191 | -39.2, -27.3, -39.5 | -35.3% | 5.66 |
| `lco94` | `s9400000` | 2.183 | -42.5, -42.2, -38.8 | **-41.2%** | 1.68 |
| `lco95` | `s9500000` | **2.885** | -42.5, -41.3, -39.2 | -41.0% | 1.35 |
| `lco96` | `s9600000` | 2.192 | -40.0, -39.9, -38.2 | -39.3% | 0.83 |
| `baseline_s1` | nothing | 2.652 | six seeds | -40.4% | 1.48 |

**Dropping a collection does not reach the policy.** Surrogate medians span 2.183 to 2.885,
a 32% spread, while transfer means span -35.3% to -41.2% and every one sits inside seed
noise of the baseline's -40.4%. `lco95` has the WORST surrogate of the four and ties for the
best policy. `lco93`'s -35.3% is one outlier seed at -27.3 against its own -39.2 and -39.5.

**And the reproducibility correlation does not survive doubling the sample.**

| | r(surrogate open-loop IQR, fine-tune sd) |
|---|---|
| n=4, the original surrogates | **-0.798**, stable under every single-point deletion |
| n=8, adding the four lco arms | **-0.331** |

So the one relationship that survived a leverage check at n=4 was still noise. Four points
is not enough for a leave-one-out to mean anything, which is worth stating plainly because
the leave-one-out was run precisely to avoid claiming an artefact and it did not save us.

THE PROGRAMME'S RESULT, now closed. Across eight surrogates, every property measurable
without running Chrono is uninformative about transfer:

| surrogate property | range across arms | effect on transfer |
|---|---|---|
| systematic velocity bias | +0.0298 to +0.0458 | none |
| open-loop median error | 0.755 to 2.885 (3.8x) | none |
| open-loop IQR | 0.208 to 1.679 (8x) | none |
| model capacity | 3x256 to 6x512 | none |
| training-data fraction | 25% to 100% | none |
| which collection is dropped | four ways | none |

Every arm that trains at all transplants to about -40%. The method is robust to the
surrogate in a way that is genuinely useful -- you do not need a good model, you need a
model -- and completely opaque in that nothing measurable predicts the residual variation.

## Operational notes

- **sbel is the only box that can run the analytic fine-tune recipe.** Batch 64 with
  15-step BPTT OOMs at 7.53 GiB (a3, RTX 5060 Ti) and 5.6 GiB (sliger, RTX 2060) even for a
  baseline-size surrogate; the 2x-width surrogate OOMs on both by a wider margin. Only the
  3090 fits it, which is why every successful fine-tune in this document ran there and why
  fan-out attempts kept dying in ways that looked like unrelated bugs. Reducing the batch
  would fit, but it changes the arm rather than the surrogate under test, which is the one
  thing a surrogate comparison must hold fixed. PPO arms are smaller and do run on sliger.
- **A fine-tune also needs the surrogate's PROCESSED DATASET, not just its checkpoint.**
  The script loads `training_datasets/<name>/metadata.json` for normalisation, so relaying
  a `.pt` alone produces a FileNotFoundError that reads like a missing model.
- **The cluster cannot train, only score and collect.** torch there raises
  `hipErrorFileNotFound` on the first GPU op: the build carries no kernels for the node
  architecture. Chrono's own HIP path is unaffected, which is why scoring and collection
  work and why this was not obvious. Preprocessing is CPU-only and does run there, so the
  usable division is: collect and preprocess on the cluster, relay the processed dataset
  (~115 MB) to a desktop, train there.
- **a3 has no git credentials**, so `git pull` fails and any commit made there never
  reaches origin. A local commit on a3 also makes later `--ff-only` pulls abort, which
  silently starves it of new configs -- its runs then die on a missing config file rather
  than on anything real. Commit from sbel; reset a3 to origin when it diverges.
- **Only sbel holds a complete CRM corpus** unless repaired. a3 is missing a 246 MB
  collection and its CRM CSVs predate the `grav_body_*` columns; sliger was missing 689
  episodes and had 108 more at the old schema, both since fixed from sbel over Tailscale.
- **Capacity, tentatively.** With a learning rate suited to its size, 12 layers reaches
  parity with 6 and not better: median of the last forty epochs 2.169 against the
  baseline's 2.173, up from 2.346 at the baseline LR. All three sit well inside their own
  IQRs, so this separates nothing yet; `qbase` and `qw512` at 32 rollout episodes are the
  version to believe.

- **The cluster scores, the desktops train.** 24 mi2101x nodes complete a full sweep in one
  ~90 minute pass at 0.1x charge; a desktop box does exactly one arm in that time. Measured:
  0.149 charged node-hours per policy.
- Backends validated against each other: CUDA desktop, MI210/HIP cluster, RX 9070 XT/HIP,
  A100/CUDA (Euler). Baseline agreement within 0.35%, episode flip rate 0.00% against the
  fleet's own 0.93% across CUDA builds. One box runs ~2% high; compare arms on one machine.
- Scoring is the bottleneck, not training. A fine-tune is 55 seconds; its verdict is 70-90
  minutes.
