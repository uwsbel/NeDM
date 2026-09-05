# Fine-tuning a locomotion policy inside a frozen surrogate: step size, not horizon

**2026-09-05.** Four fine-tunes of the imported Go2 policy inside a frozen neural
surrogate, each evaluated in Chrono exactly once.

## CORRECTION, and it inverts the headline

**An earlier version of this document reported "v4 completes 17 of 43" as a success. It
is not one.** The baseline arm is the recorded episode, so its outcome was known and free
to report, and it was omitted:

```
  ON THE SAME 43 EPISODES        completed   fell   diverged   rows median
  BASELINE (original policy)            38      5          0          3884
  v4  (best fine-tune)                  17      4         22          2126
```

**The original policy completes 38 of 43. The best fine-tune completes 17.** v4 is a
**degradation** — merely a far less severe one than v1, v2 and v3, which complete zero.

**The honest finding is therefore:** *fine-tuning inside this surrogate degrades the
policy at every displacement tested, and the degradation shrinks as the step shrinks.*
**No improvement has been demonstrated — only a decreasing amount of harm.**

That raises the question this document must not dodge: **if smaller steps monotonically
approach the baseline, the limit as displacement goes to zero is simply the original
policy, and the method as configured has produced nothing.** Establishing improvement
would require a treated arm that beats 38/43, or a tracking gain measured on surviving
pairs — and the tracking criterion remains unscored at 20 pairs against a minimum of 30.

**Every survival number below carries the baseline beside it.** Reporting a raw treated
count as a result, without the arm it is supposed to be compared against, was the error.

## Result

```
  arm         ||dW||   objective        mean|a|  completed  fell  diverged  rows med  PAIRS
  BASELINE      0.000   (original)        1.033         38     5         0      3884    n/a
  v1            8.898   tracking only     5.244          0    43         0      3884      0
  v2           12.392   upstream 8        4.725          0     0        43       307      0
  v3           19.287   upstream 10       1.239          0     0        43       133      0
  v4            8.903   upstream 10       1.188         17     4        22      2126     20
```

**Read the BASELINE row first.** 38 of 43 completed. No fine-tune approaches it.

`||dW||` is the L2 distance in policy weight space from the baseline checkpoint, whose
norm is 1445.82 — so v4 moved **0.62%** and v3 **1.33%**.

**v3 and v4 are a controlled comparison.** Identical objective, identical surrogate,
identical harness, identical episodes, near-identical action statistics. **The only
difference is where training stopped.** Surviving pairs went from 0 to 20 and completions
from 0 to 17 — **against a baseline of 38.** The comparison is valid and the mechanism it
demonstrates is real; what it demonstrates is a reduction in harm, not a gain.

## What this establishes

**The damage is caused by over-optimisation inside a locally-valid model** — not by the
objective. The surrogate is certified over 0.1 s and only 0.1 s (gain 1.077
[1.007,1.152], corr 0.668 [0.583,0.738], cosine 0.962 [0.939,0.977], n=200 val, all PASS;
at 0.5 s its apparatus ratio is 1.12 and the measurement is swamped). Gradients through
such a model are usable **for small enough steps**, and the damage scales with distance
travelled.

**It resolves two readings that could not otherwise be separated.** *"A longer-horizon
surrogate permits larger trustworthy steps"* and *"smaller steps stay inside a
short-horizon model's validity"* are the same claim measured differently. A small-step
fine-tune survives on the **existing** 0.1 s surrogate, so **horizon was never the
binding constraint — step size was.**

**The failure path is characterised rather than avoided**, which is worth having even
though no fine-tune improved on the baseline: three objectives spanning tracking-only to 10 of 14
upstream reward terms all failed at large displacement, and the action-magnitude failure
mode was separately identified, fixed, and shown *not* to be what determined the outcome.

## What was fixed in advance, and what was not searched

- The verdict criterion, thresholds, cell and primary metric were declared before any
  fine-tune existed and **were not changed at any point**.
- **Each checkpoint was evaluated in Chrono exactly once.** v2's evaluation was deferred
  with its prediction recorded in advance, then run later; it was never re-run.
- v4's prediction — *"survives materially better than v3's 133 rows, toward v1's 3884"* —
  **was written down before the checkpoint existed.** Measured: 2126.
- **`||dW|| = 8.9 was chosen to match v1, not because it is optimal. Displacement was
  never searched.** A sweep of stopping points, each with its own Chrono evaluation, is
  exactly the fitting-to-the-verdict pattern the one-shot rule exists to prevent. **"We
  did not search" is a stronger claim than any optimum a search would have produced.**

**Stated confound:** displacement is entangled with objective for v1 and v2, which used
different rewards. It is **controlled only between v3 and v4**. Four points with one
controlled comparison is a finding; a swept curve would not have been.

## The verdict is INCOMPLETE, and stays that way

**20 surviving pairs against a declared minimum of 30.** No paired difference is
computed and **no claim is made about whether tracking improved.** The minimum was not
renegotiated after seeing the result.

Reaching 30 needs more eligible episodes than this host has in the cell (43), or a policy
surviving a higher fraction. The other machine holds 37 more, which would clear it
combined — but episodes replay bit-identically only on their collecting machine, so each
half must be scored where it was collected.

## Cost

```
  559,360 environment steps   =  3.11 h of simulated robot time at 50 Hz
  actual wall time            =  99 s (v1 budget) / ~400 s (v3, v4 budget)
  equivalent Chrono, RIGID    =  1.13 h    ->    41x
  equivalent Chrono, CRM SOIL = 23.9 h     ->   870x
```

Both measured from collection metadata on this plant. **Name the reference terrain
whenever quoting either** — soil is 21x more expensive than rigid here. And the 870x is
what these steps *would* cost in soil, not a soil experiment that was run: this work is
rigid-only.

## Scope

Rigid terrain. Val-split episodes collected on this machine. Open-loop gate, closed-loop
verdict. The surrogate omits 4 of the 14 upstream reward terms, including
`correct_base_height` at −10.0, the largest converged weight, because the 34-D state has
no `pos_z_m`.

## Amended reading, recorded rather than quietly dropped

**The half-noise surrogate route was aimed at the wrong variable.** It was designed to
buy a surrogate certified past 0.1 s so that branches could be longer. The displacement
test shows horizon was not the constraint, so a longer-horizon surrogate would not have
been the fix. It is still worth finishing for the stability/sensitivity trade-off it
measures directly — contraction buys rollout stability and costs action sensitivity,
corr 0.668 against 0.495 at 0.1 s — but it is no longer on the critical path, and it
should not be described as the remedy for this failure.

---

## v5 — the second controlled comparison, declared before the checkpoint exists

The confound stated above is that displacement is entangled with objective for v1 and v2;
only v3 against v4 is controlled. **v5 runs the comparison in the opposite direction:**

```
  v1   tracking-only objective,  ||dW||  8.9  ->  FELL, 3884 rows, 0 pairs
  v5   tracking-only objective,  ||dW|| 19.3  ->  ?
```

**PREDICTION, recorded now:** v5 **diverges rather than falls**, surviving on the order of
v3's 133 rows rather than v1's 3884. If displacement drives the failure mode largely
independent of objective, then moving v1's objective out to v3's distance should reproduce
v3's *mode* despite a completely different reward.

```
  diverges, short   -> displacement drives the mode independent of objective;
                       two controlled comparisons in opposite directions
  falls, long       -> the mode IS objective-linked, the v3/v4 result is specific to
                       that objective, and the finding narrows accordingly
```

**This is one declared point completing a 2x2, not a sweep.** It was chosen because it
closes the confound, not because it might look good, and the prediction is on the record
before the checkpoint exists. Displacement is still not being searched.

### v5 RESULT: the prediction was WRONG, and the finding narrows

```
  objective        ||dW||    completed  fell  diverged  rows med   pairs
  BASELINE              0           38     5         0      3884     n/a
  tracking (v1)      8.898           0    43         0      3884       0
  tracking (v5)     19.301           0    43         0      3884       1
  10-term  (v4)      8.903          17     4        22      2126      20
  10-term  (v3)     19.287           0     0        43       133       0
```

**I predicted v5 would diverge like v3 and survive ~133 rows. It fell like v1 and ran to
full length, 3884 rows.** The prediction was recorded in advance and is falsified.

**The 2x2 reads cleanly and it is not the reading I proposed:**

- **The failure MODE is objective-linked, not displacement-linked.** The tracking-only
  objective produces *falling* at both displacements, 8.9 and 19.3, with identical
  full-length episodes. The 10-term objective produces *divergence* at 19.3.
- **Displacement matters WITHIN the 10-term objective and not within the tracking one.**
  v3 against v4 is a 0-to-20-pair difference; v1 against v5 is a 0-to-1-pair difference
  across the same displacement change.

**So "damage scales with displacement largely independent of objective" is FALSIFIED.**
The correct, narrower claim: *for the 10-term upstream objective, constraining
displacement changes the outcome from total failure to partial survival.* That is the
v3/v4 controlled comparison and it stands. It does not generalise to the tracking
objective, where displacement changed almost nothing.

**What that costs the headline:** the mechanism is real but its scope is one objective,
not a general property of fine-tuning inside this surrogate. And since v4 still completes
17 against the baseline's 38, the overall result remains a reduction in harm within one
objective, not an improvement.
