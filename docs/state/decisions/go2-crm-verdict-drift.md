# The CRM verdict drifted, and the headline -40% is stale

**Status: measured 2026-09-15 on kyle-sbel. The recipe's -40.5% is not wrong, it is
no longer current. Re-measured within one day it is -10.2%.**

## What happened

Reproducing [`go2-crm-finetune-RECIPE.md`](go2-crm-finetune-RECIPE.md) end to end
produced a policy **byte-identical** to the pinned one, and it scored far worse than
documented. The first reading of that was a reproducibility failure. It was not. The
baseline had moved and the comparison was crossing four days of it.

All five runs below are paired on the same 74 episodes of
`datasets/go2_crm_merged/score_subset_index.json`, all on kyle-sbel.

| run | `mae_vx` |
|---|---|
| BASE, 2026-09-10 | 0.1541 |
| **BASE, 2026-09-15** | **0.1877** |
| fine-tuned `w_h15r0`, 2026-09-11 | 0.0918 |
| same policy, 2026-09-15, code at `0df6995` | 0.1671 |
| same policy, 2026-09-15, code at `d140ff4` | 0.1685 |

```
WITHIN-SEPTEMBER   -40.4%   67/74 won
WITHIN-TODAY       -10.2%   48/74 won
BASE drift         +21.8%   Sept 10 -> Sept 15
```

## What is controlled

- **The policy is identical.** `finetune_crm_w_h15r0/policy_ts.pt` is still on disk at
  sha256 `20c084424570341b`, matching the recipe's pin and the fresh reproduction.
- **The fine-tune is identical.** Stops at update 94, `||dW|| 1.002`, val `-1.438482`,
  exactly as documented.
- **The evaluation code is not the cause.** Scored at `0df6995` in a worktree (0.1671)
  and at `d140ff4` (0.1685). Two versions, same answer.
- **pychrono is unchanged** since 2026-05-07.
- **Same machine** throughout. The September fine-tuned file is `*_kyle-sbel.json`.

## The two findings

**1. The verdict harness is not stable across time, and nothing was watching for it.**
A 21.8% baseline shift makes every cross-session comparison in the ledger suspect
whenever the rows were not scored in the same session. The ledger already says rows
must not be mixed across machines; they must not be mixed across *time* either.

**2. The fine-tuned policy degraded 82% while BASE degraded 22%.** Whatever drifted,
the specialised policy is far more sensitive to it than the base policy. That is
consistent with the separately observed specialisation, and it is the more interesting
result: what fine-tuning buys is partly tuned to the exact conditions it was measured
under.

## Not yet identified

What drifted. pychrono and the repo are both excluded, so the candidate is outside
git: GPU driver, CUDA, or implicitly-resolved soil. Commit `465edc0` ("record the
effective soil, which the resolved config did not") indicates soil resolution was
already known to be under-recorded, which makes it the first place to look.

## Consequences

- **Every arm in the ledger needs re-scoring within one session** before the table can
  be read as one experiment again.
- **A BASE run belongs in every scoring session**, as a drift check, not as an
  economy to be skipped. The comparison here was nearly reported as a reproducibility
  failure because the baseline was four days old.
- The capacity transfer test now has a same-day baseline (0.1877) and a same-day
  control (0.1671). The 6x1024 arm still needs a GPU that fits it at batch 64.

## CAUSE IDENTIFIED (2026-09-15): the NVIDIA kernel module changed

The drift is not in the repo, the policy, the dataset or pychrono. It is the GPU driver.

```
2026-09-02 23:32   system boots, module 595.84 loaded          <- last reboot
2026-09-10 16:17   BASE scored          0.1541                 module 595.84
2026-09-11 06:11   unattended-upgrade installs nvidia 595.91.07
                   (packages only; the RUNNING module stays 595.84, which is
                    what later produced the NVML version mismatch)
2026-09-11 09:42   w_h15r0 scored       0.0918                 module 595.84
   ...             rmmod/modprobe to clear the NVML mismatch   module -> 595.91.07
2026-09-15         BASE scored          0.1877                 module 595.91.07
2026-09-15         SAME policy scored   0.1671                 module 595.91.07
```

`uptime -s` is still 2026-09-02 and `/proc/driver/nvidia/version` reads 595.91.07, so
the module changed mid-flight via the manual reload, with no reboot. Both September
runs are on the old module; both September-15 runs are on the new one.

| | 595.84 | 595.91.07 | change |
|---|---|---|---|
| BASE | 0.1541 | 0.1877 | +21.8% |
| fine-tuned `w_h15r0` | 0.0918 | 0.1671 | **+82.0%** |
| **measured effect** | **-40.4%** | **-10.2%** | |

**Both pairs are internally valid.** September compared two runs on 595.84; today
compares two runs on 595.91.07. Neither pair may be compared across the boundary.

**The fine-tuned policy is about 4x more sensitive to the driver change than the base
policy** (+82% vs +22%). A policy optimised inside a surrogate is specialised not only
to terrain and command family, as already documented, but to the numerical behaviour of
the simulator it was scored against. That is the substantive result here.

### Status of the claim

Timeline evidence, not a controlled test. The controlled test is to reload 595.84 and
re-score; it is a system-level change to kyle-sbel and would re-break NVML until
reloaded again, so it awaits Kyle's decision.

### What this changes

- **The ledger cannot be read as one experiment.** Its rows span 2026-09-07 to
  2026-09-12, straddling the 09-11 driver upgrade AND the later module reload.
- **Record the driver with every verdict.** The harness already stamps policy sha256;
  it should stamp `/proc/driver/nvidia/version` too. Without it a verdict cannot be
  placed on the correct side of a boundary like this one.
- **Score BASE in the same session as every arm.** This was nearly reported as a
  reproducibility failure because the baseline was five days and one driver old.

## RETRACTION (2026-09-15, same day): the driver attribution is NOT supported

The section above attributes the drift to the NVIDIA kernel module going 595.84 ->
595.91.07. **That attribution is withdrawn.** The timeline argument it rests on has a
hole that kills it.

**The hole.** The 2026-09-11 06:11 unattended-upgrade included `libnvidia-compute-595`,
which provides `libcuda.so`. If the running kernel module had remained 595.84 after that
point, every CUDA process would have failed with a version mismatch from 06:11 onward.
The `w_h15r0` verdict ran successfully at 09:42 that morning. Therefore the module was
almost certainly already reloaded before that verdict, and both September runs and both
2026-09-15 runs were on the same driver. The driver cannot then explain the difference.

**A second candidate that was missed.** `score_crm_tracking.py --concurrency` defaults
to **8**; the recipe specifies **4**. Every 2026-09-15 run here used 4. What the
September runs used is not recorded anywhere. Concurrency sets how many Chrono FSI sims
share the GPU at once, which on a non-deterministic SPH solver is a plausible systematic
effect, not merely a speed knob.

## What is actually established

**1. The verdict harness is non-deterministic per episode.** Same policy, same day, two
runs: identical row count on only 46/77 episodes, identical `mae_vx` on 9/77, median
per-episode relative difference 3.38%, max 46.4%.

**2. That noise averages out in aggregate.** The same two runs agree to **0.8%** on the
77-episode mean. So the aggregate statistic is stable within a day even though
individual episodes are not.

**3. The September-to-today shift is far larger than that noise.** Median per-episode
difference 34.9%, aggregate difference 21.8% on BASE. That is roughly 27x the same-day
aggregate noise, so something changed systematically.

**4. The cause is unidentified.** Excluded: the policy (sha256 identical), the fine-tune
(bit-identical, update 94 / dW 1.002 / val -1.438482), the evaluation code (scored at
both `0df6995` and `d140ff4`, agreeing to 0.8%), pychrono (unchanged since May), and the
machine. Remaining candidates include scorer concurrency, GPU driver, and thermal or
clock state. None is confirmed.

**5. The fine-tuned policy moved more than BASE** (+82% vs +21.8%). This is the one
substantive observation, and it survives not knowing the cause: whatever shifted, the
specialised policy was several times more sensitive to it. But with the cause unknown it
cannot yet be called numerical specialisation rather than, say, a policy that simply
operates nearer a failure boundary.

## Operationally, the fix does not depend on the cause

- **Score BASE in the same session, with the same flags, as every arm.** This alone
  would have prevented the entire episode.
- **Stamp the environment into every verdict**: driver version, concurrency, host. The
  harness already stamps policy sha256; none of the rest is recorded, which is why this
  could not be settled after the fact.
- **The ledger's rows are not mutually comparable** and need re-scoring in one session
  before that table can be read as one experiment again.
