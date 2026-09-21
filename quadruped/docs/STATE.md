# State

**Updated:** 2026-09-20 (evening) · **Branch:** `kyle/quadruped-pipeline` (off `kyle/locomotion`)

## Where this is

`collect.py` produces clean CRM corpora and `train.py` fits an NN-ROM on them. Both were
exercised end to end today. `finetune.py` and Gate 3 remain unbuilt.

The base policy walks on both terrains. Quoted as replicate means, because a single CRM
run does not support a number (see `docs/EVALUATION.md`):

```
  rigid   97.0% +/- 0.1  (n=8)        CRM   74.7% +/- 5.7  (n=8, wide spawn spread)
  gap     22.2 +/- 5.6
```

**That gap is the headroom the study is about.** It is smaller and far noisier than the
single-run "95 against 64" it replaces.

## What was decided today

| decision | value | on what evidence |
|---|---|---|
| base policy | keep rl_sar `robot_lab/policy.pt` | see "the policy drifts" below |
| active domain | 0.5 m (provisional) | no measurable bias over 1.0 m at 1.7x the cost; replication running |
| free-flow duration | 0.1 s, unchanged | the bed compacts 0.3 mm total, complete within 0.5 s |
| soil preset | `soft`, not `hmmwv_reference` | 5.6x lower variance on the gap; see `docs/SOIL.md` |
| moving patch | not used | +x only, and our commands cover vy and wz |
| bed sizing | from the planned path, widened for yaw drift | see below |
| checkpoint selection | smoothed rollout errdist at 10 s | one-step val_loss ranks corpora with the wrong sign |

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
widened by integrating at the commanded yaw rate plus and minus 0.25 rad/s. Three
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

- **hpcfund can run CRM** -- verified, not assumed: `_fsi.so` and `libChrono_fsisph.so`
  are built and a `go2_crm_scoreset` from earlier work is on disk. The `quadruped/` tree,
  `src/nedm`, the policy and the URDF assets are NOT staged there yet, and the branch is
  euler-local so it has to be moved deliberately.
- **a3 cannot run the unapproximated reference.** Its display GPU's watchdog kills the
  long kernels: `cudaErrorLaunchTimeout`. Ensemble replication therefore runs on sbel.
- sbel, north, a3 all import their pinned source builds.

## Running

sbel: active-domain replication at seed 77. north: the 6-episode CRM corpus at v2 bed
sizing.
