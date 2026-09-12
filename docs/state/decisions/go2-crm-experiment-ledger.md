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

**No held-out command families.** All eight appear in both the branch pool and the scoring
set, so this measures adaptation, not generalisation to unseen commands.

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

**THE BIAS IS NOT THE BINDING CONSTRAINT ON TRANSFER, and the prediction was wrong in the
opposite direction.** The end-to-end test is in, and it reverses the reading of the soil
result recorded above.

| arm | fine-tuned in | mae_vx | vs base | wins | p |
|---|---|---|---|---|---|
| `basedft` | baseline surrogate | 0.0914 | **-41.6%** | 67/75 | <1e-4 |
| `soilft` | soil-varied surrogate | 0.0971 | **-37.3%** | 64/74 | <1e-4 |

Head to head, paired on the 74 shared episodes: soil is **+6.3% worse** than the control,
33 wins of 74, p 0.0115. Both arms used the pinned recipe, both stopped at ||dW|| ~1.0
(updates 88 and 110), both used `last.pt` of their own surrogate, so the only difference is
which model the policy was optimised against.

So the soil surrogate carries a 35% SMALLER velocity bias and produces a WORSE transplanted
policy. The chain this work was building -- collect the axis the corpus never varied,
reduce the model's systematic error, transfer better -- does not hold. Reducing the bias
made transfer worse.

Line up the three quantities and the pattern is the opposite of what was recorded:

| surrogate | open-loop median | bias | transfer |
|---|---|---|---|
| `baseline_s1` | 2.173 | +0.0458 | **-41.6%** |
| `go2_crm_soil` | 2.842 (worse) | +0.0298 (better) | **-37.3%** (worse) |

Transfer follows OPEN-LOOP ACCURACY and not the bias. The earlier entry claimed the
quantity the trainer minimises is not the quantity that governs transfer; on this evidence
`rollout_sel` tracked transfer correctly and the bias did not. That claim is withdrawn.

What this does to the attribution result. "The residual is model error, not an optimisation
shortfall" still stands -- nothing here touches it. "Therefore target the bias" was the
wrong inference from it, and it was mine. A constant offset is measurable, real, and
apparently not what limits the transplanted policy.

CONFOUND ONE, being removed by collection. The soil surrogate trained on 0.96 h against
the baseline's 1.55 h, so "soil diversity hurt transfer" and "less data hurt transfer" are
not separated by this pair. A second soil wave of 800 episodes (seed offset 10200000) is
collecting now, which takes the soil corpus to roughly 2.9 h and past the baseline. If a
soil surrogate with MORE data than the baseline still transfers worse, diversity is the
cause; if it catches up, the first result was a quantity artefact and the collection
strategy is vindicated after all.

CONFOUND TWO, and it is resolvable. The soil arm differs from the baseline in data AND in
open-loop error, so this pairing cannot separate them. `w512ft` fine-tunes in `abl_w512`,
which holds the corpus fixed and improves open-loop accuracy (median 1.109 against 2.173)
with a bias in between (+0.0376). PREDICTION, logged before the run: `w512ft` beats
`basedft`'s -41.6%. If it lands near baseline instead, open-loop accuracy does not predict
transfer either, and neither surrogate-side metric currently available does.

**Superseded-schema data archived: 46 roots, 78.78 GB.** `datasets/` went from 138 GB to
63 GB and from 76 roots to 30. MOVED to `datasets_archive/`, not deleted, with
`ARCHIVE_MANIFEST.json` recording each root's file count, size, column count and reason;
reversible with a single `mv`.

Two guards produced the list, because an archive that breaks a live experiment is worse
than no archive:

- the root must be UNIFORMLY old-schema. The four `go2_gravworld_off*` roots are MIXED,
  5-15% of their episodes predating `grav_body_*`, and the merged index draws all 2,607 of
  its episodes from them, so they stay.
- nothing live may reference it, checked against every processed training dataset and every
  merged index.

Verified after the move: `go2_crm_merged` resolves 795 of 795 and `go2_gravworld_merged`
2,607 of 2,607. The four `go2_crm_*_c` indices appear broken to an absolute-path check
because they store RELATIVE `csv_path` values; their files are all present.

**Correction on the gravity corpus, arrived at properly this time.** An earlier entry
concluded the 27.88 h corpus is gravity-randomised by reading the `grav_shuf_*` columns.
Those are a PLACEBO: `shuffle_gravworld.py` writes them as `grav_world` permuted across
episodes, a dimensionality control so that a gain from adding three tilt channels cannot be
confused with a gain from adding three channels of anything. The real channel is
`grav_world_*`, and it carries 25 distinct vectors in 25 sampled episodes, |g| 9.810 tilted
up to about 3 degrees. So the conclusion stands, but it was reached from the control column
rather than the real one, which was luck.

Physically that tilt is rigid ground at a VARYING SLOPE, which is genuine terrain variation
rather than a defect. What is missing is a level-ground corpus to pair against CRM, which
`collect_rigid.sbatch` now collects with tilt pinned to zero.

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
