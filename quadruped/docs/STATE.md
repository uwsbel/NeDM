# State

**Updated:** 2026-09-21 (afternoon) · **Branch:** `kyle/quadruped-pipeline` (off `kyle/locomotion`) · **Head:** `07f440a4`

## Where this is

The whole pipeline exists and has run end to end at scale: an 800-episode CRM corpus
collected on hpcfund, NN-ROMs trained on it, analytic and PPO fine-tunes, and a paired
Chrono evaluation. **The first fine-tune results are void** -- not because either method
failed, but because the policy rolled inside the NN-ROM was not the policy the robot runs
(below, and `LESSONS.md`). The fault is fixed and verified (`07f440a4`); the corpus is
being recollected as v2, because the fix is in how rows are captured.

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
   rigid ground and ~60% on CRM. Now captured before the step; `obs_truth.py` shows every
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

## Corpora and models

| corpus | episodes | segments | rows | capture | use |
|---|---|---|---|---|---|
| `go2_crm_v1` | 800 | 1914 | 1,425,186 | post-step | NN-ROM capacity data only |
| `go2_crm_v1_partial` | 550 | 1318 | 980,523 | post-step | superseded |
| `go2_crm_v2` | 1200 (planned) | | | pre-step | the corpus fine-tunes run on |

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
| row capture | before the physics step | `obs_truth.py`, `07f440a4` |
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

`namecheck.py` compares names read against names bound per function with a real scope
chain, because `py_compile` accepts a function that reads a name nothing assigns and one
such bug cost three minutes of GPU. Its first real catch was worse than the bug that
motivated it: **`doctor.py` was verifying nothing.** `_no_driver` had been inserted into
the middle of `check_chrono`, so the trap-hash refusal and the build-md5 comparison sat
below a `return` and never ran. Any run that passed doctor since `39f317eb` was checked
against nothing (`5e3df653`).

## Fleet

- **hpcfund** runs CRM collection. v1 used 16 x `mi2101x` (1.75 h per 50-episode shard,
  ~2.8 charged node-hours). v2 packs 4 shards per `mi2104x` node, one per MI210, at the
  same 10 GPU-h per charged node-hour, because `mi2101x` was fully allocated. Staged code
  is marked with `qrun/.source_commit`.
- **a3** trains NN-ROMs (~1.7 min per 2000-step epoch on the 800-episode corpus).
- **north** runs the paired CRM evaluation (~36 s per 6 s episode) and the rigid tests.
- **euler** holds the branch. It still reads dirty because of 15 untracked
  `configs/go2_crm_*.json`, which is Kyle's call (commit, ignore, or `--untracked-files=no`).
- a3 cannot run the unapproximated reference (display watchdog); sbel can.

## Running

hpcfund: v2 collection, array job 430005 (6 x `mi2104x`, 24 shards, started 13:00).
The packed smoke (429999) ran four CRM collects at once, one per GPU, at the same
per-episode speed as a single-GPU node. a3: an NN-ROM on the 800-episode v1 corpus, kept
running only as a capacity data point, since no fine-tune can use a v1 model.
