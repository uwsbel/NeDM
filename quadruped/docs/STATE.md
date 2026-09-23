# State

**Updated:** 2026-09-22 (morning) · **Branch:** `kyle/quadruped-pipeline` on uwsbel/NeDM (off `kyle/locomotion`) · **Head:** `048272ee` + this

## Where this is

**The recipe is frozen and replicates.** Fine-tune the surrogate on its own 1 s rollouts;
run PPO in it with 2 s branches (100 control steps) and 1024 parallel rollouts, weight
budget dw 4.0, 10 epochs x 4 minibatches. In nine independently trained surrogates (ten
runs), scored in Chrono on 37 held-out paths from all ten command families, it improves
forward tracking 25-32% (mean 29%), sideways 14-21% (mean 18%) and yaw 49-55% (mean 53%),
every axis clear of zero in every run; on straight walking forward error falls 38-60%.
Nothing fell; two runs walked off the finite CRM bed on one extra path each, late (9.9 s,
14.3 s). On rigid ground the same ten keep the yaw gain (50-58%) and leave forward tracking
neutral on average (+2%, one of ten worse at 2 se). Details in the next section.

Why each piece (evidence further down):
- **1 s rollout training.** All ten seeds beat the predict-no-motion baseline across 10 s;
  0.5 s training does in 2 of 10, and one-step training fails by ~2 s.
- **2 s branches.** 0.30 s branches fail to improve forward tracking across paths in every
  surrogate tried, one-step, rollout-trained and ensembles alike, however well they score
  on the straight test.
- **1024 envs.** At the same weight budget, forward gain grows from 64 to 1024 rollouts
  (-21.5% to -30.5%) and the rigid-ground forward penalty seen at 64-256 goes away. 2048
  (three seeds, -33.8%) is a few points better, inside the spread between seeds: the
  curve has flattened, and 1024 costs half as much.
- **One surrogate, not an ensemble.** Ensembles land inside the single-surrogate range.

Effects reproduce across machines (NVIDIA north and AMD hpcfund, different Chrono builds)
to within a few points.

**The write-up is a live page**, not a snapshot: https://claude.ai/artifact/G8PHCRfk7M8Mu7b8rW2PbU
(14 sections, every table on it measured in Chrono, plus replayable paired trajectories from
`evaluate.py --dump-traj`). Kyle's instruction is to keep it current as results land and as
old results are refuted or improved, so a result that changes STATE changes the page too.
Stale there today: the nine-surrogate table is at the dw 4.0 stop. Nothing is running; next is the write-up.

## The frozen recipe in nine surrogates (2026-09-22)

1 s rollout-trained surrogates (seed 7 at two PPO seeds on hpcfund 431212; seeds 6, 8, 9
and north s0 fine-tuned on hpcfund 431232;
euler seeds 2-5 on euler 66715), seed 0, 1024 envs, 2 s branches. CRM scored on hpcfund
(431232, 431302/431303) against hpcfund's base arm; rigid scored on sbel against sbel's.

| surrogate | paths vx | paths vy | paths wz | straight vx | rigid vx | rigid wz |
|---|---|---|---|---|---|---|
| seed 6 | -26.7% | -13.9% | -48.8% | -45.9% | -1.9% | -50.3% |
| seed 7 (PPO seed 0) | -28.6% | -16.4% | -52.5% | -51.3% | +5.8% | -55.2% |
| seed 7 (PPO seed 1) | -31.9% | -16.1% | -50.6% | -37.9% | -14.1% | -52.4% |
| seed 8 | -32.1% | -15.5% | -50.2% | -45.9% | -15.9% | -53.3% |
| seed 9 | -30.4% | -16.6% | -53.0% | -52.2% | -6.7% | -56.2% |
| north s0 | -25.4% | -17.6% | -54.1% | -56.0% | **+19.4%** | -55.0% |
| euler s2 | -27.8% | -17.8% | -54.4% | -58.4% | +16.4% | -56.5% |
| euler s3 | -31.1% | -20.5% | -54.8% | -53.4% | +8.7% | -57.7% |
| euler s4 | -30.6% | -21.1% | -53.2% | -59.5% | +0.4% | -55.5% |
| euler s5 | -27.5% | -20.7% | -53.8% | -57.9% | +9.1% | -55.5% |

Every CRM entry is clear of zero at 2 se (paths: 36-37 usable pairs; straight: 16/16,
also improving vy 10-22% in all ten). Rigid: wz better on 40/40 paths in all ten, vy
-5% to -19% (four clear of zero), vx clear of zero only where bold. The two extra CRM
failures are euler s2 (random path, left the bed at 9.95 s) and s4 (yaw step, 14.3 s).
The spread across surrogates is now 7 points on forward tracking, against 131 points
(-51% to +80%) for one-step surrogates with 0.30 s branches on the straight test.

**Per command family** (paths, mean over the eight surrogates other than seed 7 of each
arm's paired change;
4 paths per family, 2-3 where the bed cap skipped or a pair dropped; "vx better" counts
arms whose family mean improved):

| family | vx mean [range] | vx better | vy | wz |
|---|---|---|---|---|
| constant | -54% [-60, -46] | 8/8 | -13% | -56% |
| vel_step | -46% [-48, -43] | 8/8 | -9% | -55% |
| yaw_step | -44% [-51, -37] | 8/8 | -3% | -46% |
| arc | -28% [-35, -20] | 8/8 | -26% | -56% |
| pivot | -27% [-42, -7] | 8/8 | -11% | -53% |
| random | -25% [-30, -19] | 8/8 | -27% | -33% |
| weave | -20% [-28, -7] | 8/8 | -16% | -49% |
| random_walk | -12% [-17, -4] | 8/8 | -10% | -50% |
| stop_and_go | -7% [-8, -4] | 8/8 | -28% | -73% |
| **lateral** | **+19% [+8, +34]** | **0/8** | -20% | -61% |

Lateral paths are the one consistent cost: forward error there rises in all eight, from a
base of ~0.05 m/s (the command's vx is near zero) by ~0.01 m/s, concentrated on two of the
four paths, while sideways error on the same paths falls by up to 0.1 m/s. Small in m/s
against the 0.07-0.15 m/s forward gains on speed steps and constant commands, but it is
systematic and belongs in the write-up.

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

## More data does not help; coverage does (2026-09-23)

The method's obvious exposure is that it lives on what the corpus covers, so both halves
were measured rather than argued.

**Size.** go2_crm_v23 extends go2_crm_v2 with 1,174 newly collected episodes contributed as
TRAINING data only (`merge_corpus.py --train-only`), so the held-out set is exactly v2's and
only the training data grows: 2,094 training episodes against 920, 4,129,428 rows, ~11.5 h
of robot time. Four surrogates trained on it with the v2 optimiser budget. Nothing moved.
Accuracy: errdist at 2 s 0.469-0.517 against 0.477-0.520, marginally worse at 10 s. Transfer
at the new stop: -53.3%, -53.0%, -54.1%, -47.4% forward on the paths against -52.7%, -53.9%,
-55.3% for the original corpus. Two arms drew branch starts from the larger corpus and two
from the original, so it is not a branch-pool effect. **The corpus is past the knee for this
task.**

**Coverage, measured on the real robot.** Gate 4 reports the fraction of states visited in
Chrono that fall outside the corpus region:

| policy | outside | distance ratio |
|---|---|---|
| base | 0.0% | 0.81 |
| iteration 500 | 0.0% | 0.82 |
| iteration 1500 | 0.0% | 0.87 |
| iteration 3000 | 0.0-0.1% (one arm 0.5%) | 0.91 |
| seed 6 at iteration 1000 | 0.0% | 0.88 |
| seed 6 at iteration 1500 | 50.4% | 1.96 |
| 120 N push | 0.0% | - |
| 240 N push | 6.1-13.8% | - |
| 300 N push, BASE policy | 9.7% | - |

Three readings. Optimising hard against the model does not walk the policy out of the data:
3000 iterations at dw 11 still visits 0.0-0.1% outside, with the distance ratio drifting
0.81 -> 0.91, a measurable approach to the boundary rather than a departure. When a run does
leave it leaves abruptly, inside at 1000 and half outside at 1500, so there is no slope to
threshold on and the guard reads the training signal instead. And what takes the robot
outside is DISTURBANCE, not the fine-tune: the untouched base policy is 9.7% outside under a
300 N shove, because the corpus tops out at 140 N. The coverage requirement binds on the
disturbances one intends to claim robustness to, not on the controller's search within the
task -- the same wall the kick experiment hit.

## The stopping budget is too tight (2026-09-22, still running)

Every fine-tune here stops when the actor's weights have moved dw 4.0, about 430 iterations
at 1024 rollouts. That budget was inherited from the old pipeline as a guard against PPO
exploiting the surrogate, and it had never been tested; in-model reward is still climbing
steeply where it fires. hpcfund 431640 runs the recipe to 3000 iterations in four 1 s
surrogates (seeds 6, 8, 9, north s0), saving the policy every 500 iterations and at each dw
mark, and every checkpoint is scored in Chrono. Each run repeats the scored recipe run up
to dw 4, and its dw-4 checkpoint came out BIT-IDENTICAL to the policy already scored, so
the runs reproduce exactly.

Paths (37 pairs), mean over the three healthy surrogates:

| checkpoint | iterations | dw | mae_vx | mae_wz | rigid mae_vx |
|---|---|---|---|---|---|
| dw 2 | ~100 | 2.0 | -6% | -19% | +3% |
| **dw 4 (the current stop)** | ~430 | 4.0 | -29% | -53% | ~0% |
| iteration 500 | 500 | 4.3 | -35% | -55% | -14% |
| **iteration 1000** | 1000 | 6.3 | **-54%** | **-69%** | **-32%** |

The full curve, with robustness beside it (push test, upright of 16 at 300 N; the base
policy is 11/16):

| checkpoint | iterations | dw | paths mae_vx | paths mae_wz | rigid mae_vx | upright at 300 N |
|---|---|---|---|---|---|---|
| dw 2 | ~100 | 2.0 | -6% | -19% | +3% | - |
| dw 4 (the old stop) | ~430 | 4.0 | -29% | -53% | ~0% | 11-13 |
| iteration 500 | 500 | 4.3 | -35% | -55% | -14% | 12 |
| **iteration 1000** | 1000 | 6.3 | **-54%** | **-69%** | **-32%** | **11-12** |
| iteration 1500 | 1500 | 7.0 | -59% | -72% | -32 to -45% | 8-10 |
| iteration 2000 | 2000 | 7.6 | -59% | -73% | -33 to -44% | - |
| iteration 2500 | 2500 | 9.9 | -60% | -74% | -34 to -46% | - |
| iteration 3000 | 3000 | 10.8 | -59% | -75% | -29 to -39% | - |

**So the stop belongs at about iteration 1000.** Tracking roughly doubles the old stop's
gain and then plateaus by 1500; rigid-ground tracking improves throughout; and robustness
to a 300 N shove holds at the base policy's level to iteration 1000 and is clearly worse
by 1500. Neither axis alone finds that point: tracking says "keep going", robustness says
"not that far".

Tracking is FLAT from 1500 to 3000 (-59%, -60%, -59%) while the runs travel from dw 7.0 to
10.8, so nothing is bought after about 1500 and the robustness cost keeps accruing. The path
evaluation carries its own quiet version of that cost: beyond the three paths the base arm
also skips, the extra episodes an arm loses to a fall or to leaving the bed run 0, 0, 0 at
iterations 500 and 1000 across the three healthy surrogates, then 1, 2, 6 at 1500, 2000 and
2500-3000. Averages never show it -- a failed episode is dropped from the pairing, not
averaged in -- so it is only visible by counting what was dropped.

**Why robustness erodes is not a mystery.** robot_lab trains the base policy with a
velocity kick of +/-0.5 m/s every 10-15 s, randomised friction, base mass -1 to +3 kg, link
masses 0.7-1.3x, COM +/-5 cm and actuator gains 0.5-2.0x. Our fine-tune has none of it: one
surrogate, one soil, no disturbance, and a reward containing only tracking error and
uprightness. Whatever the base policy knows that pays off only when disturbed has no
gradient protecting it. `finetune.py --branch-push-prob` puts the disturbance back (one
kick per chosen branch at a uniform control step, +/-0.5 m/s, matching upstream); the
surrogate can roll the recovery because recoveries are in the corpus, only the force
windows were cut. Whether that holds robustness while tracking still improves is hpcfund
432532 (15% and 50% of branches, seeds 8 and 9).


**One run of four came apart, and not at a distance the budget would have caught.** Seed
6's run showed out-of-distribution spikes from its first hundred iterations (10/99 against
2-4 for the others), rising to 95/100 by iteration 1000, with value loss reaching 480,000,
reward falling from -0.046 to -1.01, and dw stalling at 5.6 while the others climbed past
6.8 still improving. `diagnostics/probe_ood.py` rolls a policy in several surrogates from
identical starts and prices the states against the corpus the way PPO does:

| policy | in seed 6's model | seed 8's | seed 9's | north s0's | worst channel |
|---|---|---|---|---|---|
| seed 6, iteration 1000 | **0.371** | 0.447 | 0.477 | 0.477 | `grav_body_z`, 69-127 sd |
| seed 8, iteration 1000 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | a hip velocity, ~9 sd |
| recipe policy | 0.0003 | 0.0003 | 0.0003 | 0.0003 | a hip velocity, ~10 sd |
| base policy | 0.0004 | 0.0004 | 0.0004 | 0.0004 | a hip velocity, ~10 sd |

Reading: the corpus covers what working policies do (0.0% of their steps fall outside it),
and seed 6's policy learned to TUMBLE -- the body-frame gravity direction leaves its
recorded range by 69 to 127 sd -- which every surrogate agrees about. Its own surrogate is
the most forgiving of the four, which is what exploitation looks like from the inside. So
this is not evidence that the corpus is too small, and a fixed weight distance is the wrong
guard: it stopped three healthy runs early and is not what kept them healthy. The OOD term
separated the cases from the first hundred iterations and is the candidate stopping signal.
Whether seed 6's surrogate does this under every PPO seed is hpcfund 432022 (seeds 1 and 2
in it, and in seed 8's as a control).

## The guard, and what one collapse in fifty means for the method (2026-09-23)

One run in about fifty came apart (seed 6 above). That is rare, it needed training 2.5x
past the old stop in the most exploitable of ten surrogates, and two other PPO seeds in
that same surrogate did not collapse -- but "rare and undetermined" is not "safe", and
until now the only proof a run was good came from Chrono.

It does not have to. Every sign was in `finetune.jsonl` hours before any scoring: the
fraction of iterations whose branches left the corpus region (0-4 per 100 in healthy runs,
26 rising to 95 in the collapse), value loss (under 3 against 480,000), reward direction,
and dw stalling. `run_ppo` now watches a rolling window of the first two TOGETHER -- a
burst of spikes alone is ordinary exploration, reward falling alone is a hard batch of
starts; the failure is both -- keeps the last healthy checkpoint, and on a trip stops the
run and writes that checkpoint as the result (`--guard-window/-spike/-drop`, off with
`--guard-spike 1.0`). The OOD cost is now MEASURED whether or not it is priced, because a
run with the penalty disabled is the one whose health matters most.

`diagnostics/guard_replay.py` replays it over every run this study has logged. **6 of 60
trip, and every one is independently bad in Chrono**: the collapse (caught at iteration 871,
keeping a good policy); the two old 0.30 s runs `ppo_h7_s0` and `ppo_h9_s0`, whose paths
scores were +30% WORSE than the base policy; and three runs deliberately pushed over with a
raised learning rate (hpcfund 432752), caught at iterations 57, 58 and 80.

| induced failure | in-model reward | value loss | guard | Chrono, 40 paths |
|---|---|---|---|---|
| 10x lr, seed 8 | -52.9 | 1,170,280 | iter 58 | 0 of 40 usable |
| 10x lr, seed 9 | -293.2 | 6,434,217 | iter 57 | 0 of 30 usable |
| 3x lr, seed 8 | -0.267 | 36,434 | iter 80 | 17 of 40 failed, vx +44% |
| 3x lr, seed 6 | -0.009 | 0.015 | no trip | vx -42%, wz -73%, 37/37 usable |

The last row carries as much weight as the trips: that run travelled to dw 16.9, four times
the old budget and 15% of the policy's weight norm, stayed healthy and was left alone. A
fixed distance would have stopped it long before, and would still have missed the collapse,
which was clean at dw 5 and broke later.

**The OOD penalty is not what makes PPO work here.** The first attempt at inducing failure
removed it (hpcfund 432599) and produced four healthy runs at 800 iterations, including in
the surrogate that had collapsed, scoring -42.7%, -50.6%, -54.8% and -55.7% forward on the
paths -- the band the penalised runs occupy at that length. finetune.py's own docstring
calls the penalty "the difference between PPO working and PPO cheating"; that was written
for one-step surrogates with 0.30 s branches and 64 rollouts and does not hold at the
current settings. It may still matter for longer runs, weaker surrogates or short branches,
and the OOD MEASUREMENT is what the guard reads either way, so it stays measured whether or
not it is priced. Four runs at one training length is what that statement rests on.

One limit remains: the spike statistic is a mean over branches, so 64-branch runs look
noisier than 1024-branch ones for purely statistical reasons. It should be the fraction of
branch-steps outside, which does not depend on batch size.

## The Chrono GPU fault, investigated (2026-09-23)

`GPU failure in chrono_fsi/sph/physics/SphBceManager.cu:543 -- an illegal memory access`,
then `thrust::system_error: HIP free failed`. FIVE occurrences in ~3,600 CRM episodes, about
1 in 720: one in v2 collection (job 430005 shard 1, 2026-09-21), two on 2026-09-23 in 300 N
push evaluations (job 432517, arms ck500_n and ck1000_n), and two more the same day in v3
COLLECTION (job 432712, shards 7 and 13, which stopped at 34 and 40 of their 50 episodes).

**The force correlation is REFUTED.** With three occurrences it looked like the fault only
appeared at the hardest shove we apply; the two collection faults carry ordinary 8-140 N
collection pushes, so force is not the discriminator and the earlier reading was three
points of coincidence. What survives is that it is rare, that it always lands in the same
place, and that it is not deterministic.

Line 543 is the error CHECK; the fault is inside `CalcRigidForces_D` (SphBceManager.cu
304-383), whose only out-of-range candidate is `sorted_index = mapOriginalToSorted[...]`
indexing `derivVelRhoD` and `posRadD`, which are sized to the active-domain particle count
and shrink as the domain follows the robot. `calcHashD` early-returns WITHOUT writing its
hash when a position is non-finite or out of bounds (SphCollisionSystem.cu:74-94), which
leaves a stale index behind; the flag meant to catch that is unreliable in our pinned
Chrono and was hardened upstream in 42ce46b5 (PR #829), after our pin. A marker leaving
its own box is ruled out: rigid BCE markers are attached to the body. Upstream has no fix
for this crash, and 10.0.0 predates our pin.

It is not deterministic: the rerun passed both crash points with the same code. Violence
raises the probability, it does not determine it, and the arms that died also happened to
hold GPUs 2 and 3, so force and device are confounded.

CONTAINMENT, which matters more than the cause: the Python exception is catchable but the
process is NOT recoverable, because thrust throws from a destructor and terminate() ends
the run regardless. So evaluate.py writes the partial record and calls `os._exit(90)`
before teardown, and the job scripts re-run that arm with `--resume`, which keeps the
episodes already recorded. A fault now costs one episode instead of an arm. The failure is
also made visible: job 432517 reported COMPLETED while two arms had died, and their
truncated records were nearly compared against full ones, so the scripts now print
`ARM_FAILED`.

## Putting the disturbance back does NOT hold robustness (2026-09-23)

The hypothesis was specialisation: robot_lab trains the base policy with a +/-0.5 m/s kick
every 10-15 s plus broad randomisation, our fine-tune has no disturbance at all, so whatever
only pays off when disturbed has no gradient protecting it. `--branch-push-prob` puts the
kick back. hpcfund 432532 ran it at 15% and 50% of branches in seeds 8 and 9 to 1500
iterations.

| iteration | upright at 300 N, no kick | with kick | paths mae_vx, no kick | with kick |
|---|---|---|---|---|
| 1000 | 11, 11, 12 | 10, 11, 11, 13 | -54% | -49% |
| 1500 | 8, 10 | 10, 8, 10, 12 | -59% | -50% |

Robustness is unchanged within noise, at 240 N the kicked arms are slightly worse, and
tracking costs 5 points at iteration 1000 and 9 at 1500. At 120 and 180 N, where nothing
falls and recovery quality is the measurement, they are indistinguishable too: seed 8 reads
-45.7% without the kick against -44.9% and -42.8% with it at 120 N, seed 9 -51.1% against
-50.3% and -45.9%, differences of 1-5 points in no consistent direction against a resolution
of about 2, and the heavier kick is slightly worse in three of four comparisons.

**Two reasons it could not have worked, and they are worth keeping.** The kick is +/-0.5 m/s
while the 240-300 N test shoves produce peak errors of 2.3-2.9 m/s, and the corpus tops out
at 140 N pushes -- so the surrogate has never seen a recovery from anything like the test and
cannot teach one. A SURROGATE CAN ONLY TEACH THE ROBUSTNESS ITS CORPUS CONTAINS. And the kick
is an instantaneous velocity change applied to the model's state, while a real push is a
force over 0.1-0.2 s with soil interaction: the model can roll the recovery, which is in its
data, but it never represents the disturbance event itself. Teaching robustness inside a
learned model may need the model to carry the disturbance as an input, which this one does
not.

## Robustness to pushes (2026-09-22)

`evaluate.py --push-force` shoves the trunk once per episode on CRM, at a fixed time, in a
fixed body-frame direction, through the same accumulator the collector uses: 8 directions x
2 lattice positions, identical for every arm, scored over the 3 s after the force ends
(`eval_push.sbatch`, `paired_eval.py --metrics push`). The recipe policies (1024 rollouts,
dw 4) against the base policy:

| force | base upright | seed 6 | seed 8 | seed 9 | recipe mae_vx after the push |
|---|---|---|---|---|---|
| 120 N | 16/16 | 16/16 | 16/16 | 16/16 | -29% to -34% |
| 180 N | 16/16 | 15/16 | 16/16 | 16/16 | -25% to -27% |
| 240 N | 16/16 | 11/16 | 14/16 | 16/16 | -17% to -20% |
| 300 N | 11/16 | 10/16 | 15/16 | 13/16 | not clear of zero |
| total falls | 5/64 | **12/64** | 3/64 | 3/64 | |

Fine-tuning does not cost robustness in general: two of three surrogates fall slightly less
than the base policy, and all three recover with markedly lower tracking error. The
exception is seed 6 again, the surrogate whose training went unstable -- two independent
measurements pointing at the same model.

Recovery TIME is not usable as reported: it was measured against each episode's own
pre-push error band, and a policy that tracks better has a tighter band, so the arms were
held to different bars. `push_recover_fixed_s` (a common 0.30 m/s bar) is recorded from
now on. Gate 4 is reported but not enforced in push mode, since a 240 N shove is meant to
leave the corpus region.

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
`msft210_*` logs on hpcfund and `kyle-v2-msft-*` on euler; 0.5 s = 50-step, 1 s = 100-step, each from that seed's
one-step surrogate):

| surrogate | 0.3 s | 2 s (0.5 s / 1 s) | 10 s (0.5 s / 1 s) |
|---|---|---|---|
| seed 6 | 0.35 | 0.862 / **0.483** | 11.7 / **0.576** |
| seed 7 | 0.35-0.36 | 0.535 / **0.504** | 0.863 / **0.615** |
| seed 8 | 0.35 | 0.632 / **0.500** | 0.791 / **0.566** |
| seed 9 | 0.35-0.37 | 0.847 / **0.500** | 1.93 / **0.551** |
| north s0 | 0.35 | 0.656 / **0.520** | 1.62 / **0.562** |
| north s1 | 0.35 | 0.798 / **0.507** | 3.77 / **0.572** |
| euler s2 | 0.34-0.36 | 0.899 / **0.501** | 1.52 / **0.597** |
| euler s3 | 0.35 | 0.789 / **0.491** | 5.21 / **0.581** |
| euler s4 | 0.36-0.37 | 0.815 / **0.477** | 2.51 / **0.616** |
| euler s5 | 0.35 | 0.622 / **0.503** | 2.01 / **0.579** |

1 s training beats the no-motion baseline across 10 s in 10 of 10 seeds (2 s 0.48-0.52,
10 s 0.55-0.62); 0.5 s training does so in 2 of 10 and blows up by 10 s in the rest (seed
6: 11.7). The 0.3 s score is the same for both, so it cannot choose between them. PPO with
2 s branches transferred in all of the 0.5 s surrogates anyway, because a 2 s branch
only uses the first 2 s. The 1 s surrogates are the standard from here: the margin they
buy is what makes longer branches possible. The recipe at 1024 envs is being replicated
in eight of them (hpcfund 431232: seeds 6, 8, 9, north s0; euler 66715: seeds 2-5).

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
| 2048 (three seeds) | -33.8% | -19.3% | -54.7% |

Monotone to 512, flattening from 1024 (2048 seeds: -33.8, -31.9, -35.7% forward, against
-26.4 and -34.5% at 1024); all eleven runs improve all three axes. Averaging more
rollouts into each update at the same displacement gives better policies. The straight
test is saturated (-52% to -60% forward for all) and cannot see this; the paths can.

**Ensembles of one-step surrogates with 0.30 s branches do not help** (a3, 8 members, with
and without the disagreement penalty): forward tracking across paths -2%, +22%, +1%, +18%;
two of the four nearly doubled forward error on the straight test. Short branches are the
problem, and averaging over one-step models does not fix them.

**On rigid ground the recipe does no harm at 512+ envs** (sbel, the same 40 paths on rigid
terrain, paired against sbel's own base arm; `rigid_sbel*.sh`; 40/40 usable in every arm):

| parallel rollouts | mae_vx (seed 0 / 1) | mae_vy | mae_wz |
|---|---|---|---|
| 64 | **+36.7%** / +13.8% | -1% / -6% | -42% / -39% |
| 256 | **+21.8%** / +6.4% | -9% / -10% | -54% / -48% |
| 512 | -5.3% / +9.7% | -10% / -4% | -51% / -52% |
| 1024 | +12.2% / +1.3% | +3% / -3% | -56% / -57% |
| 2048 | -0.8% / +7.1% / -3.5% (seeds 0-2) | 0% / -10% / -13% | -59% / -61% / -60% |

Bold: clear of zero at 2 se. The yaw fix carries over whole (better on 39-40 of 40 paths in
every arm) and grows with the rollout count, as on CRM: it corrects a deficiency of the
base policy, not a CRM quirk. The forward-speed adaptation to CRM costs some rigid forward
tracking at 64-256 envs (2 of 4 runs clear of zero); at 512 and above one of seventeen is
(these seven and the ten recipe runs above). The
rigid scores are deterministic: a base run killed after writing its 40 episodes (five
evaluations at once ran sbel out of memory) and its clean rerun agree exactly.

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
