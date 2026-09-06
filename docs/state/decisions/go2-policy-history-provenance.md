# The Go2 policy is fragile to observation-history provenance

Status: measured 2026-09-06. Recorded separately from the excitation
collection because it bears on how the whole fine-tune line should be read,
not on how data is collected.

## The measurement

Two excitation arms, matched at 160 perturbed rows and ~1.03 s per episode,
differing only in how many times control is handed back to the policy:

```
arm A   L=40, N=4     4 handovers    300/300 kept     0.0% rejected
arm B   L=10, N=16   16 handovers   259/300 kept    13.7% rejected (all joint_limit)
arm C   L=10, N=4     4 handovers   300/300 kept     0.0% rejected
```

Burst length is not the driver. A and C share N=4 at different L and both
reject nothing; B differs from C only in handover count and rejects 13.7%.

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

Arm B is dropped, and the reason is survivorship rather than the rejection
rate. Excluding episodes that fail at handover selects against exactly the
states where the policy struggles, so arm B's survivors are biased toward easy
handovers -- a survivorship filter operating on the mechanism under study.
Arm A delivers the same perturbed rows at the same cost with no rejection and
no selection.

Standing plan: **arm A for bulk, arm C for the burst-length comparison at
matched handover count, arm B dropped.** A-versus-C isolates burst length with
handover count held fixed, which is the comparison the original A/B pair could
not make -- there `L x N` was held constant, so `L` and `N` were perfectly
anti-correlated and completely confounded by construction.

## Remedy, not built

If arm B's structure is ever wanted back, the fix is a **ramped handover**:
blend the target from perturbed to policy over a few steps rather than
switching abruptly, and mark the blended rows as neither. Not built now.
Worth knowing it exists if the burst-length comparison says short bursts
matter a lot.
