# State

**Updated:** 2026-09-22 (early morning) · **Branch:** `kyle/quadruped-pipeline` on uwsbel/NeDM (off `kyle/locomotion`) · **Head:** `073ce6f4`

## Where this is

**PPO fine-tuning inside the NN-ROM improves CRM tracking, verified in Chrono, and the best
recipe is a long-horizon surrogate with long branches.** Fine-tuning a trained surrogate on
its own 50-step rollouts makes it trustworthy across 10 s instead of ~2 s; PPO with 2 s
branches in it halves forward tracking error (-50%, -53%, 16/16 episodes, both seeds),
cuts yaw error 56-59%, and leaves sideways tracking unchanged. The same surrogate with
0.30 s branches is WORSE, so short rollouts are not merely defensible, they are the wrong
choice once the surrogate supports long ones. Effects reproduce across machines (NVIDIA
north and AMD hpcfund, different Chrono builds) to within a few points.

**The recipe, verified on held-out paths from all ten command families:** a rollout-trained
surrogate, PPO with 2 s branches, and 512-1024 parallel rollouts. Across 37 paired Chrono
paths it improves forward tracking ~30%, sideways ~17% and yaw ~53%, and it REPLICATES:
2 s PPO in each of eight rollout-trained surrogates improves all three axes in every one
(forward -16% to -30%), where one-step surrogates with 0.30 s branches ranged from -51% to
+80% on the straight test and failed to improve forward tracking across paths at all.

An ensemble of rollout-trained surrogates adds nothing over one of them (below). Still
open: 2048 envs (three seeds) and the 1 s rollout-trained surrogate at 1024 envs are running
(hpcfund 431212, sbel).

The v1 fine-tunes of 2026-09-21 are void (three rollout bugs, below); everything since is on
the v2 corpus with rows captured before the physics step.

The base policy walks on both terrains. Quoted as replicate means, because a single CRM
run does not support a number (see `docs/EVALUATION.md`):

```
  rigid   97.0% +/- 0.1  (n=8)        CRM   74.7% +/- 5.7  (n=8, wide spawn spread)
  gap     22.2 +/- 5.6
```

**That gap is the headroom the study is about.**

On the evaluation protocol (CRM, vx 0.5, 16 spawns over +/-1 m, 6 s, north), the base
policy scores mae_vx 0.132 (sd 0.019), mae_vy 0.044, mae_wz 0.210, 16/16 upright, Gate 4
0.0% outside the corpus. That is the baseline every fine-tune is paired against.

## v1 fine-tunes: void, and why

| arm | usable pairs | mae_vx change | mae_wz change | Gate 4 |
|---|---|---|---|---|
| analytic | 11/16 (5 fell or failed) | +1.13 m/s, worse in 11/11 | +0.69 | FAIL, 29.4% outside |
| PPO | 16/16 | +0.077 (+58%), t +9.7, worse in 16/16 | +0.049 (+24%), t +10.2 | PASS, 0.0% |

In the model both had improved (analytic's tracking loss fell 3.5x). Three faults, each
sufficient on its own to break transfer:

1. **Post-step capture.** `collect.py` recorded each row after `DoStepDynamics`, so at a
   control row the state already carried the PD kick from the action being chosen there.
   The policy's output from a recorded row missed its real output by 36% of its spread on
   rigid ground and ~60% on CRM. Now captured before the step; `diagnostics/obs_truth.py` shows every
   observation block agreeing exactly against a live run.
2. **100 Hz policy.** The fine-tune called the policy every model step (rows are 100 Hz,
   control is 50 Hz). It now holds each action for two model steps.
3. **Shifted action history.** State row j was paired with action row j+1 throughout the
   context window. Now aligned, as `train.py`'s own rollout already was.

`finetune.py` now refuses to run unless the base policy, on observations rebuilt from the
corpus, reproduces its own recorded outputs at the branch starts. On a v2 corpus that
agreement is 8.8e-07 against a spread of 1.55 (rigid) and 0.0000 against 1.57 and 1.94
(CRM, hpcfund smoke job 429999). On v1 it cannot pass, and v1 corpora are refused
outright (no `row_capture=pre_step` stamp).

## v2 fine-tunes: the first valid results

CRM, vx 0.5, 16 spawns over +/-1 m, 6 s, north, all arms paired against one base run
(mae_vx 0.128, mae_wz 0.212, 16/16 upright). PPO at dw 4.0 (0.30 s branches, recorded
commands, OOD penalty 1.0 unless noted):

| fine-tune | mae_vx | mae_wz | mae_vy | upright | Gate 4 |
|---|---|---|---|---|---|
| PPO s0, north surrogate | -36% (16/16) | -46% (16/16) | +15% | 16/16 | 0.0% |
| PPO s1, north surrogate | -20% (16/16) | -53% (16/16) | -13% | 16/16 | 0.0% |
| PPO s0, a3 surrogate | **+57%** (0/16) | -32% (16/16) | +31% | 16/16 | 0.0% |
| PPO s0, north, no OOD penalty | -25% (15/16) | -40% (16/16) | +24% | 16/16 | 0.0% |
| analytic, north (dw 2.89, iteration cap) | -- | -- | -- | **0/16** | 37% FAIL |

**Robust:** PPO removes about half the base policy's yaw drift, in every fine-tune. That is
a property of the policy, not of the soil -- it improves on rigid ground too (-55%).

**Not robust:** the forward-speed gain depends on which surrogate the policy was tuned in.
The two v2 surrogates agree to 2% at the 0.30 s use horizon (0.375, 0.367), so accuracy at
the use horizon does not predict transfer. On rigid ground the north-surrogate PPO policy
tracks vx worse (0.026 -> 0.038), consistent with a CRM-specific adaptation when it helps.

**Analytic fails outright** and not for lack of a penalty: PPO without the OOD penalty still
transfers. The loop check separates faithful from broken loops on a real model (ratio 1.26
faithful, 1.52 with the 100 Hz fault, 1.37 with the shifted history), so it can now gate.
That separation is at 0.30 s. Over 2 s branches the ratio reads 1.3-2.4 on loops that
transfer well, and it rises as the surrogate improves: closed-loop error stays at 0.27-0.30
in every surrogate while open-loop error falls from 0.23 to 0.12. The check now prints
the errors along the branch beside a decorrelated reference (another start's recording;
0.64 at 2 s against 0.29 closed-loop) and warns on the 0.30 s ratio only.

## What the spread and the ablations showed (2026-09-21 evening)

Eleven single-surrogate PPO runs (all paired against one base run on north unless noted):

| surrogate | epoch | seed | branch | mae_vx | mae_wz | mae_vy |
|---|---|---|---|---|---|---|
| north s0 | 11 | 0 | 0.30 s | -36% | -46% | +15% |
| north s0 | 11 | 1 | 0.30 s | -20% | -53% | -13% |
| a3 | 72 | 0 | 0.30 s | **+57%** | -32% | +31% |
| a3 | 72 | 1 | 0.30 s | **+80%** | -40% | +26% |
| north s1 | 12 | 0 | 0.30 s | +3% (ns) | -35% | +25% |
| north s1 | 12 | 1 | 0.30 s | -21% | -50% | -2% |
| north s0 | **80** | 0 | 0.30 s | -5% (ns) | -36% | +16% |
| north s1 | **80** | 0 | 0.30 s | **-54%** | -42% | -8% (ns) |
| north s0 | 11 | 0 | **1.00 s** | -24% | **-53%** | **-20%** |
| north s0 | 11 | 0 | no budget (dw 9.3) | -25% | -40% | **+66%** |

- **Yaw drift falls 32-53% in every run.** Robust to surrogate, seed, epoch and branch.
- **Forward speed is surrogate-dependent, -54% to +80%,** and not explained by the
  selected epoch: a late (overfit) epoch hurt one surrogate and helped another. Nothing
  measured about a checkpoint predicts it. Hence the ensemble.
- **1.0 s branches improve all three axes**, the first run to do so; longer branches are
  not the risk they were in the old pipeline (sweep to 2 s running).
- **No displacement budget** does not break PPO but degrades vy badly (dw 9.3 against 4.0;
  in-model OOD cost rose to 0.24): unlimited search drifts into model error, mildly here.

**Chrono is predictable; the surrogate is the limit.** v1/v2 twin episodes (same seed,
dynamics identical, differing only by GPU rounding) diverge by errdist 0.012 at 0.3 s and
0.003 at 2 s (median, 53 twins; mostly the one-physics-step timestamp offset), 0.06 at
10 s. The surrogates score 0.37 and ~1.0 there. Part of the gap is hidden soil state the
36-D state does not carry; the rest is headroom (`diagnostics/chaos_floor.py`).

## The better surrogate (multi-step training)

Chrono is predictable for seconds (twin episodes: errdist 0.003 at 2 s); one-step
surrogates are not (~1.0 at 2 s). Teacher forcing never shows a model its own errors, so
`train.py --init-from <best.pt> --rollout-loss-steps K` fine-tunes a trained surrogate on
K-step rollouts of its own predictions (8 epochs x 1000 steps, lr 1e-4; ~1.2 h on one
MI300X). Seed-7 surrogate, errdist against the no-motion floor of 1.0:

| horizon | as trained | + one-step control | + 30-step | **+ 50-step** |
|---|---|---|---|---|
| 0.3 s | 0.366 | 0.366 | 0.359 | **0.359** |
| 1.0 s | 0.570 | 0.519 | 0.500 | **0.432** |
| 2.0 s | 1.33 | 1.17 | 0.725 | **0.535** |
| 5.0 s | worse | 1.65 | 0.996 | **0.624** |
| 10 s | worse | 1.40 | 1.01 | **0.863** |

A 10-step loss from scratch helps to ~2 s but diverges by 5 s; the length of the training
rollout is what matters. The 100-step variant was still improving at the time limit.

**Across seeds, 1 s (100-step) rollout training is the robust one** (best.pt, from the
`msft210_*` logs on hpcfund; 0.5 s = 50-step, 1 s = 100-step, each from that seed's
one-step surrogate):

| surrogate | 0.3 s | 2 s (0.5 s / 1 s) | 10 s (0.5 s / 1 s) |
|---|---|---|---|
| seed 6 | 0.35 | 0.862 / **0.483** | 11.7 / **0.576** |
| seed 7 | 0.35-0.36 | 0.535 / **0.504** | 0.863 / **0.615** |
| seed 8 | 0.35 | 0.632 / **0.500** | 0.791 / **0.566** |
| seed 9 | 0.35-0.37 | 0.847 / **0.500** | 1.93 / **0.551** |
| north s0 | 0.35 | 0.656 / **0.520** | 1.62 / **0.562** |
| north s1 | 0.35 | 0.798 / **0.507** | 3.77 / **0.572** |

1 s training beats the no-motion baseline across 10 s in 6 of 6 seeds (2 s 0.48-0.52, 10 s
0.55-0.62); 0.5 s training does so in 2 of 6 and blows up by 10 s in the rest (seed 6:
11.7). The 0.3 s score is the same for both, so it cannot choose between them. PPO with
2 s branches transferred in all of the 0.5 s surrogates anyway, because a 2 s branch
only uses the first 2 s. The 1 s surrogates are the standard from here: the margin they
buy is what makes longer branches possible. Euler's four (seeds 2-5) finish training on
2026-09-22 morning.

## The recipe that works: long-horizon surrogate, long branches

PPO in the seed-7 surrogate, paired against north's base arm (CRM, vx 0.5, 16 spawns):

| surrogate | branch | seed | mae_vx | mae_vy | mae_wz |
|---|---|---|---|---|---|
| as trained (~2 s) | 0.30 s | 0 | -44% | **+40%** | -41% |
| 50-step (10 s) | 0.30 s | 0 | **+36%** | +32% | -40% |
| 50-step (10 s) | 0.30 s | 1 | **+19%** | +4% (ns) | -54% |
| **50-step (10 s)** | **2.0 s** | **0** | **-50% (16/16)** | +8% (ns) | **-56%** |
| **50-step (10 s)** | **2.0 s** | **1** | **-53% (16/16)** | +4% (ns) | **-59%** |

All 16/16 upright, Gate 4 0.0%. Reading: the multi-step fine-tune trades a little one-step
accuracy (val loss 0.010 -> 0.016) for long-horizon accuracy; short branches see only
what it traded away, long branches use what it gained.

The branch-length sweep on a one-step surrogate (seed 1, usable to ~3 s) agrees: yaw
improves with branch length (-35/-50% at 0.3 s to -53/-61% at 2.0 s), the sideways
penalty of short branches disappears, and 2.0 s improved all three axes for both seeds.

## Reproduction across machines, and the surrogate spread

The same three policies scored on north (NVIDIA, build d1d0bd0a) and hpcfund (AMD MI210,
build c716f05e): mae_vx -36/-40%, -26/-25%, -24/-24%; mae_wz -46/-45%, -61/-61%,
-53/-54%. vy, the smallest channel, is noisier.

PPO at 0.30 s in eight one-step surrogates: forward tracking improves in seven (-14% to
-51%), and a3's surrogate is the outlier (+57%, +80%). Yaw improves in all. The selected
epoch does not predict it (late epochs 44, 56, 72, 80 went both ways). The consistent
cost of 0.30 s branches is sideways tracking (+13-18%, ~0.007 m/s), which 2 s branches remove.

## Evaluation on paths

`evaluate.py --paths N` scores N held-out schedules per command family (all ten, seeds
777000000+, none seen in training), 15 s each, with the command changing along the path
as it does in collection; `paired_eval.py --by-family` breaks results down per family.
Beds are sized at full commanded speed and a robot leaving the bed ends the episode as
failed.

**First results** (hpcfund 430923, 4 paths x 10 families, 15 s; 37 of 40 paired -- the
same three fast paths were refused by the bed builder in every arm, and no robot left a
bed):

| policy | branch | mae_vx | mae_vy | mae_wz |
|---|---|---|---|---|
| multi-step surrogate, seed 0 | 2.0 s | **-22%** (29/37) | **-12%** | **-42%** (37/37) |
| multi-step surrogate, seed 1 | 2.0 s | **-33%** (32/37) | **-13%** | **-45%** (37/37) |
| one-step surrogate (s1) | 2.0 s | -20% (31/37) | -7% (ns) | -33% (37/37) |
| one-step surrogate (north s0) | 1.0 s | -12% | -9% | -37% |
| multi-step surrogate | 0.30 s | **+10%** | **+13%** | -32% |
| one-step surrogate (north s0) | 0.30 s | +3% (ns) | +10% (ns) | -23% |
| one-step surrogate (seed 7) | 0.30 s | **+30%** | +2% (ns) | -36% |

Seed 7 at 0.30 s scored -44% forward on the straight test and is 30% WORSE across paths.
Per family, the 2 s recipe's forward gains are largest on speed steps (-47/-53%), yaw
steps (-40/-49%), constant (-44/-47%) and arcs (-22/-34%); yaw improves in every family
(-32% to -64%). Forward changes on lateral and pivot paths are noisy, since their
commanded forward speed is zero and the base error small.

**Rule from this:** every result is scored on the paths, not only the straight command.

**Branch length across paths** (seed 7, rollout-trained on 0.5 s; 37 paired paths):

| branch | mae_vx (s0 / s1) | mae_vy (s0 / s1) | mae_wz (s0 / s1) |
|---|---|---|---|
| 0.30 s | +10% (s0) | +13% | -32% |
| 1.0 s | -23% / -22% | -1% / -3% (ns) | -38% / -41% |
| 2.0 s | -22% / -33% | -12% / -13% | -42% / -45% |
| 5.0 s | -31% / -21% | -13% / -15% | -47% / -47% |
| 2.0 s, rollout-trained on 0.3 s | -28% | -17% | -41% |

Across paths 2 s and 5 s are about equal on forward tracking (mean -27.5% and -26%), 5 s a
little better on sideways and yaw at ~2.5x the compute. The straight test had 5 s clearly
ahead; the paths flatten it. Recipe: branches of 2 s or more in a rollout-trained surrogate.

**Replication across surrogates** (2 s branches, 64 envs, seed 0, 36-37 paired paths):

| surrogate | rollout-trained on | mae_vx | mae_vy | mae_wz |
|---|---|---|---|---|
| seed 6 | 0.5 s | -16% | -8% | -42% |
| seed 7 | 0.5 s | -21% | -12% | -43% |
| seed 7 | 1.0 s | -18% | -14% | -42% |
| seed 8 | 0.5 s | -21% | -10% | -44% |
| seed 9 | 0.5 s | -30% | -13% | -43% |
| north s0 | 0.5 s | -23% | -11% | -42% |
| north s1 | 0.5 s | -18% | -8% | -36% |
| north s1 | 1.0 s | -26% | -13% | -37% |

All eight improve all three axes significantly. Rollout training on 1 s does not transfer
better than 0.5 s at 2 s branches; what it buys is that every surrogate beats the no-motion
baseline across 10 s (0.57, 0.62 at 10 s where 0.5 s training left some at 1.5-3.8).

**Env scaling** (seed 7, 0.5 s rollout-trained, 2 s branches, same weight budget, 10 epochs
x 4 minibatches; mean of two seeds, 37 paired paths):

| parallel rollouts | mae_vx | mae_vy | mae_wz |
|---|---|---|---|
| 64 | -21.5% | -11.4% | -42.7% |
| 256 | -25.9% | -16.2% | -49.0% |
| 512 | -29.1% | -17.9% | -50.5% |
| 1024 | -30.5% | -17.0% | -52.9% |

Monotone to 512, flattening at 1024; all eight runs improve all three axes. Averaging more
rollouts into each update at the same displacement gives better policies. The straight
test is saturated (-52% to -60% forward for all) and cannot see this; the paths can.

**Ensembles of one-step surrogates with 0.30 s branches do not help** (a3, 8 members, with
and without the disagreement penalty): forward tracking across paths -2%, +22%, +1%, +18%;
two of the four nearly doubled forward error on the straight test. Short branches are the
problem, and averaging over one-step models does not fix them.

**Nor do ensembles of rollout-trained surrogates** (hpcfund 431093/431094: six 0.5 s
rollout-trained members, seeds 6, 7, 8, 9 and north s0, s1; a random member per branch; 64
envs; 37 paired paths):

| branch | disagreement penalty | seed | mae_vx | mae_vy | mae_wz |
|---|---|---|---|---|---|
| 0.30 s | 0 | 0 / 1 | +18.5% / +31.6% | +2% / -3% | -24% / -28% |
| 0.30 s | 1 | 0 / 1 | +8.7% / +12.2% | +19% / +6% | -27% / -25% |
| 2 s | 0 | 0 / 1 | -16.8% / -13.2% | -8% / -15% | -41% / -44% |
| 2 s | 1 | 0 / 1 | -21.3% / -12.8% | -8% / -9% | -41% / -37% |

With 0.30 s branches forward tracking still gets worse on paths even though every member is
rollout-trained, which settles that the short branch, not the surrogate, is at fault. With
2 s branches the ensemble lands inside the range of its single members (-16% to -30%
forward) and the penalty changes nothing consistently. One rollout-trained surrogate is
the recipe; parallel rollouts, not members, are where the extra compute pays.

## Corpora and models

| corpus | episodes | segments | rows | capture | use |
|---|---|---|---|---|---|
| `go2_crm_v1` | 800 | 1914 | 1,425,186 | post-step | NN-ROM capacity data only |
| `go2_crm_v1_partial` | 550 | 1318 | 980,523 | post-step | superseded |
| `go2_crm_v2` | 1150 | 2716 | 2,041,752 | pre-step | the corpus fine-tunes run on |

**v2 collection** (array job 430005, 6 x `mi2104x`, 4 shards per node, ~3.9 charged
node-hours): 23 of 24 shards clean. Shard 1 died in its 33rd episode on a Chrono GPU fault
(illegal memory access, `SphBceManager.cu:543`) -- the only such fault in ~1,950 CRM
episodes across v1 and v2 -- and was excluded; collect.py now writes its manifest after
every episode so a crash costs one episode, not a shard (`c2928b3d`). v2 truncates more
episodes than v1: 210 of the 1146 its shards scored (18.3%, 202 of them on `off_bed`) against
12.6%, and kept 94.4% of rows against 96.9% on the first 391 episodes. Summed over the 23
shard manifests: 115 episodes in each of the ten command families, 276 long episodes (48 in
validation). The merged manifest recorded shard 0's tallies alone until the fix of this date.
The same seeds plan some episodes longer in v2 (7.98 s against 7.03 s for one weave), which
points at v1 having run staged code that matches no commit; v1 recorded none, so this
cannot be settled.

The 800-episode v1 NN-ROM (a3): errdist 0.391 at 0.3 s, 0.469 at 0.5 s, 0.732 at 1.0 s,
over the floor by 2.0 s. 1.45x the data of the 550-episode model bought 13% at the use
horizon.

The 550-episode NN-ROM is usable to about 0.3 s (errdist 0.451 there, against the 1.0 a
predict-no-motion model scores) and crosses the floor at 1.5-2 s. Fine-tune branches are
0.30 s. `train.py` now judges the floor at that use horizon, not at the 10 s selection
horizon, which had stamped this model worse-than-nothing when it is not at 0.3 s.

## What was decided

| decision | value | on what evidence |
|---|---|---|
| base policy | keep rl_sar `robot_lab/policy.pt` | see "the policy drifts" below |
| active domain | 0.5 m | no measurable bias over 1.0 m at 1.7x the cost, replicated |
| free-flow duration | 0.1 s, unchanged | the bed compacts 0.3 mm total, complete within 0.5 s |
| soil preset | `soft`, not `hmmwv_reference` | 5.6x lower variance on the gap; see `docs/SOIL.md` |
| moving patch | not used | +x only, and our commands cover vy and wz |
| bed sizing | from the planned path, widened for yaw drift (0.35 rad/s) | see below |
| checkpoint selection | smoothed rollout errdist at 10 s; usability judged at 0.30 s | one-step val_loss ranks corpora with the wrong sign |
| row capture | before the physics step | `diagnostics/obs_truth.py`, `07f440a4` |
| fine-tune commands | the recorded command at the branch start | random commands score a 0.3 s transient the policy was never asked to win |
| evaluation | paired, same 16 spawns per arm, `paired_eval.py` | spawn is most of the CRM variance |

## The policy drifts, and it is the policy

With the yaw command held at ZERO the robot turns at 0.126 rad/s on rigid ground -- the
terrain it was trained on -- and at 0.124 to 0.309 rad/s on CRM. Over a 20 s episode that
is 70 to 350 degrees of unplanned heading. A straight command walks an arc.

**This is a property of the policy, not an integration fault.** Yaw tracking is left-right
symmetric (+1.0 -> +0.941, -1.0 -> -0.935; +0.5 -> +0.559, -0.5 -> -0.541) and the offset
collapses from 0.126 at zero command to 0.003 at unit command. That is a deadband, not an
asymmetry. The URDF hips are non-mirrored (`axis 1 0 0` both sides, symmetric limits) and
`default_pos` is identical across all four legs, so the uniform sign convention is
consistent.

**Kept rather than replaced.** A base policy with a measurable yaw deficiency is headroom
for fine-tuning, not an obstacle to it. Switching to the HIM policy would work
mechanically -- a 6-step observation history is a function of the NN-ROM's state sequence
and the estimator head is differentiable -- but its estimator was fit on rigid dynamics,
so CRM fine-tuning would train estimator and policy together, and a policy with more
expressive surface makes NN-ROM exploitation easier, which is our known failure mode.

## Collection

The bed is sized from the PLANNED PATH, dead-reckoned from the command schedule, and
widened by integrating at the commanded yaw rate plus and minus 0.35 rad/s (raised from
0.25 after the pilot truncated 6 of 30 episodes on `off_bed`; 0.309 was the worst seen). Three
successive versions of this were wrong and each was caught by a guard rather than by
inspection:

1. The bed was not where the spawn rule thought it was (`89c898b8`) -- caught by a cost
   benchmark returning bit-identical results for five different configurations.
2. The travel budget ignored warmup travel and let the duration floor override the bed
   limit (`d0147967`) -- caught by the spawn assertion added in (1).
3. The budget was blind to yaw, so `weave` left the bed after 7.33 s (`df6bb874`) --
   caught by the `off_bed` check added in (1).

After all three, the 6-episode smoke corpus keeps every row it collects, where the first
version kept 86% and truncated two of six episodes.

## Cost

Full detail in `docs/COST.md`. The short version, with one correction:

- The active domain is the cost. 3.81x real time at 0.5 m, 6.18x at 1.0 m, 36.95x with
  none at all.
- **Patch length is nearly free only up to about 2 M particles.** An earlier version of
  this file said it was free generally, extrapolated from a benchmark that only spanned
  444k to 1.77M. Measured further: 3.5 M is +34% per step, 7.1 M is +82%, 13.3 M is +170%.
  So a drift-proof disc-shaped bed is not affordable and the bed is sized to the path.

## Tooling added today

`diagnostics/namecheck.py` compares names read against names bound per function with a real scope
chain, because `py_compile` accepts a function that reads a name nothing assigns and one
such bug cost three minutes of GPU. Its first real catch was worse than the bug that
motivated it: **`doctor.py` was verifying nothing.** `_no_driver` had been inserted into
the middle of `check_chrono`, so the trap-hash refusal and the build-md5 comparison sat
below a `return` and never ran. Any run that passed doctor since `39f317eb` was checked
against nothing (`5e3df653`).

## Fleet

- **hpcfund** collects CRM corpora and now also trains surrogates (MI300X, ~30 s/epoch, the
  fastest here) and runs paired Chrono evaluations (4 per `mi2104x` node, one per MI210).
  mi3001x is often congested; mi2104x is the fallback.
- **euler** trains on the `sbel` partition (4 x A100 on euler19, not preempted). Share
  euler19 politely: another user runs CPU jobs there; size requests so nothing is preempted.
- **north is PARKED IDLE** (2026-09-22, Kyle's instruction) until further notice: nothing
  is to run there. d33 (RX 9070 XT, 16 GB, ROCm 7, NAS-mounted) replaces it for fine-tunes.
- **a3, sbel** have 30 GB RAM. evaluate.py once needed ~30 GB (fixed, `d6b15f28`); it
  OOM-killed evaluations on a3 and appears to have taken sbel down.
- **NAS** (`/mnt/nas/Main/nedm/{data,models,results}`, STANDARD.md sec. 3) holds the corpora,
  surrogates and results; the branch is on GitHub.

## Running

hpcfund: multi-step fine-tunes of 8 surrogates
(430862), after which a launcher runs 2 s PPO in each and the multi-step ensemble;
env scaling 64-1024 envs (430927). sbel: 2048 envs seed 0. north: 2048 envs seed 1 after
the branch-length runs in the multi-step surrogate. euler: multi-step fine-tunes of its
four surrogates.
