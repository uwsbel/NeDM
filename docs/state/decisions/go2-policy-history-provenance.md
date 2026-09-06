# The Go2 policy is fragile to observation-history provenance

Status: measured 2026-09-06. Recorded separately from the excitation
collection because it bears on how the whole fine-tune line should be read,
not on how data is collected.

## The measurement

Two excitation arms, matched at 160 perturbed rows and ~1.03 s per episode,
differing only in how many times control is handed back to the policy:

```
EX-A   L=40, N=4     4 handovers    300/300 kept     0.0% rejected
EX-B   L=10, N=16   16 handovers   259/300 kept    13.7% rejected (all joint_limit)
EX-C   L=10, N=4     4 handovers   300/300 kept     0.0% rejected
```

Burst length is not the driver. EX-A and EX-C share N=4 at different L and both
reject nothing at 300 episodes; EX-B differs from EX-C only in handover count and
rejects 13.7%.

**EX-A's rate is ~0.1%, not 0%.** The 0/300 above is the 300-episode diagnostic;
at larger n rejections do occur. The 0% figure was an artefact of the sample
size available when it was first quoted, and is corrected here so it does not
harden -- a quoted zero is unusually sticky, because nobody re-examines a rate
with no exceptions. Two orders of magnitude below EX-B either way, so nothing in
this finding changes. Whether EX-A's rare rejections also sit at the handover is
the open test of the mechanism on the arm we kept.

The timing localises the mechanism:

```
phase at rejection:       perturb 0    recover 41
rows since handover:      median 3, max 11   (~30 ms at 100 Hz logging)
```

Every rejection occurs after the policy resumes, none during perturbation.
The destabilising event is not the robot being perturbed. It is **the policy
being handed a robot it did not drive there** -- resuming with a 5-step
observation history containing rows it never produced, and commanding a joint
past its URDF limit within ~30 ms.

This was predicted to cluster at perturbation onsets. It clusters at the
opposite transition.

## Replicated three times, 85 for 85 (2026-09-06)

Three collections, different burst structures, different seeds, and the third run by a
collector that differs from the first two:

```
  arm                          rejections   phase              rows since handover
  EX-B  L=10, N=16, seed 22        41       41 recover, 0 perturb   median  3, max 11*
  EX-A  L=40, N=4,  seeds 21/31/33/34   17  17 recover, 0 perturb   median 19, max 46
  GT    L=40, N=4,  seeds 41-46        27  27 recover, 0 perturb   median 19, max 30
  ------------------------------------------------------------------------------
                                       85   85 recover, 0 perturb
```

`*` EX-B's window is 12 rows, so its maximum is right-censored at 11; see the note below.

**85 of 85 after the policy resumes, none during perturbation**, and the median is
identical at 19 rows on the two arms whose recovery window is long enough to measure it.
This is no longer a property of one collector. It is a measured property of the imported
controller, sampled three times independently.

## Why it matters beyond the collector

Two severities of one phenomenon, now both quantified:

- the policy recovering badly -- recovery caps at **0.79x** the walking ceiling;
- the policy not recovering at all -- **13.7%** joint-limit violation within 30 ms.

It also joins the earlier result that this policy, fed *true* states, drifts
off its own recorded behaviour once it accumulates its own actions, worse than
predicting the mean action.

The consequence for fine-tuning: during fine-tuning the policy's observation
history is produced by the **surrogate's** dynamics, not Chrono's. A controller
this sensitive to history provenance should be expected to transfer fragilely
**for reasons unrelated to the surrogate's accuracy**. That is a confound in
every fine-tune result we have read so far, and it is a property of the
controller, not of the surrogate under test. A fine-tune that fails may be
failing here rather than at the dynamics model.

## Consequence for the collection

EX-B is dropped, and the reason is survivorship rather than the rejection
rate. Excluding episodes that fail at handover selects against exactly the
states where the policy struggles, so EX-B's survivors are biased toward easy
handovers -- a survivorship filter operating on the mechanism under study.
EX-A delivers the same perturbed rows at the same cost at ~0.1% rejection and
no meaningful selection.

Standing plan: **EX-A for bulk, EX-C for the burst-length comparison at
matched handover count, EX-B dropped.** EX-A versus EX-C isolates burst length with
handover count held fixed, which is the comparison the original A/B pair could
not make -- there `L x N` was held constant, so `L` and `N` were perfectly
anti-correlated and completely confounded by construction.

## Remedy, not built

If EX-B's structure is ever wanted back, the fix is a **ramped handover**:
blend the target from perturbed to policy over a few steps rather than
switching abruptly, and mark the blended rows as neither. Not built now.
Worth knowing it exists if the burst-length comparison says short bursts
matter a lot.
