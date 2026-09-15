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
