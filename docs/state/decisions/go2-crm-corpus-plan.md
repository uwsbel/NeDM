# CRM corpus and surrogate: what the rigid experiment forces us to do differently

**2026-09-08.** The rigid programme cost a full day and produced one usable scientific
result plus a long list of defects. This is the checklist those defects imply, applied
to CRM **before** collection rather than after.

## Carried over from rigid, as binding requirements

**1. Select checkpoints on a metric that descends.** `checkpoint_metric` was
`rollout_sel`, which moves 27-46% between adjacent epochs; `best_val.pt` therefore held
epoch 1 for three of seven arms. Every CRM config sets `checkpoint_metric: val_loss`,
and the pipeline fine-tunes from `last.pt` regardless. **Verify the epoch inside any
checkpoint before using it.**

**2. The surrogate training run is the experimental unit, not the episode.** Two seeds
per arm cannot produce p below 0.333; four is the floor for reaching 0.05. Measured
between-seed sd on rigid was 12-14 points, so the standard error on an arm mean at n=4
is ~6.8 and only ~20-point effects are resolvable. **Budget four seeds per arm minimum
and expect to need 8-10 to resolve a 12-point effect.**

**3. Every treated arm needs a displacement-matched RANDOM control.** Three random
perturbations at the fine-tunes' own `||dW||` were flat on both evaluation cells
(96.5% vs base 96.1%; 21.5%/19.3% vs base 20.3%), which is the only reason the treated
numbers mean anything. **Build the control at the same `||dW||` and score it in the
same round.**

**4. Measure the baseline. Never take it from a selection rule.** "Base completes 0 of
these by construction" was asserted in a docstring and was false -- measured 20.3%.
Base's 96.1% on the val split was equally tautological because the split *is* the
episodes base cleared. **Score BASE on every cell, every time.**

**5. Verify transfers by content, not arrival.** Three fleet dataset transfers exited 0
while truncated, and the receiving gate checked only that the directory size had
stopped changing -- which a truncated file does instantly. **sha256 manifest at the
origin, verified on each receiver, promote from `.stage/` only after it passes.**

**6. Derive episode keys from path and refuse on collision.** The base-failure index
namespaced on scenario dir but not shard root: 522 episodes carried 416 distinct ids
and the paired evaluator silently kept one of each colliding pair. **Every index gets a
uniqueness check that can fail.**

**7. Read state from artifacts, never from process listings.** `pgrep -f`/`ps | grep`
patterns sent over ssh match the ssh command carrying them -- tailscaled's `be-child`
wrapper puts the whole command in the process table. This produced phantom "running"
readings three times and once killed the session mid-pipeline. **Poll `metrics.jsonl`
epochs and file existence.**

**8. A completion signal that cannot fail is not a check.** `best_val.pt` exists from
epoch 1, so "checkpoint exists" never meant "training finished" and fired a completion
watcher on three runs at epoch 3. **Gate on the logged epoch against `num_epochs`.**

**9. Clean up scratch.** The scorers left 16,352 temp episode dirs holding 67 GB on one
box. Fixed; the CRM scorers inherit it.

**10. Record the scoring host in every output filename.** Four boxes run four distinct
pychrono builds (a fifth on d33 if it comes up). Measured cross-machine effect is 0.2
points on the rate with a 0.93% paired flip rate -- tolerable for seed-level rates,
**not** tolerable for a per-episode paired test whose discordance is itself a handful
of episodes.

## New for CRM, from the pilot

**11. The primary metric is velocity-tracking error, not completion.** Nothing fell in
14 pilot episodes; the robot survives and fails to move, achieving 34-85% of commanded
velocity. A completion metric would score every arm, the baseline and the random
control at ~100%: a flat table that reads as a null and is actually a metric failure.

**12. The contact-conditioned arm must use force channels.** `foot_*_in_contact` is NaN
in every CRM row (475/475 measured) because the feet couple through FSI and Chrono's
contact system sees nothing. `foot_*_force_fz_n` is fully populated. A CRM arm D built
on the rigid preset would train on an all-NaN block and be a silent no-op.

**13. Collect on the WIDE command envelope.** The pilot drew |vx| <= 0.17 and still
showed a 15-66% tracking deficit; the deficit grows with speed, and `PARAM_RANGES_WIDE`
spans vx (-1.5, 2.0) against the narrow (-0.5, 0.5). **The narrow envelope understates
the very headroom the experiment exists to exploit.**

**14. Episode length is bounded by the bed, not by the budget.** Usable travel is ~5.7 m
from the spawn. At 16 s the pilot truncated 0 of 14, but the wide envelope will push
fast commands into the boundary. Duration is set to 12 s as the compromise, truncation
is *recorded per episode*, and truncated episodes remain valid for surrogate training
(transitions, not whole episodes) and for tracking error (which is defined on partial
episodes, unlike completion).

## Fleet

Four CUDA boxes collect. **d33 cannot**: it has an AMD 9070XT and Chrono's FSI/SPH is
CUDA-only. Its 24 cores and 46 GB make it a Chrono *scoring* box, which is pure CPU, so
it offloads evaluation from the machines doing collection and training. It is a fresh
install with no passwordless sudo, so its toolchain comes from conda rather than apt.
