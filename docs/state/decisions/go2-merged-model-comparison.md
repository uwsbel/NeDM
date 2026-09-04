# Merged-model comparison: which knobs are frozen

Written **before the merge**, while the level-3 result on the 152-episode CRM
model does not yet exist. dorm-pc cautioned that a merged model is not a
replacement for that result and should not be compared against it. The
coordinator's refinement is right: **the comparison is available if the right
things are held fixed**, and uninterpretable if anything else moves at the same
time. So the frozen list is decided now rather than after seeing which model
looks better.

## The comparison is available, and one thing makes it cleaner than expected

**THE CHRONO FLOOR DOES NOT INVOLVE THE DYNAMICS MODEL AT ALL.** It is Chrono
driven by the reference's own recorded commands. So the floors already measured —
primary 0.0655 m, least-moving 0.0439 m, val 0.0652 m — are *literally the same
numbers* for a merged model. Nothing to re-measure and nothing to drift.

**And the reference set is reusable.** `assign_split` is a function of
`episode_id`, and dorm-pc's episodes carry a different seed offset (1,000,000),
so no id of mine changes split when its half is added. Every episode in
`go2_flat_crm_ref40.npz` stays in train, so **the 8/8 training-overlap property of
the primary arm survives the merge**. This was the one knob the coordinator
doubted could be held; it can, provided the reference set is REUSED rather than
redrawn from the larger pool.

## Frozen. Changing any of these voids the comparison.

    reference set          go2_flat_crm_ref40.npz, REUSED, not redrawn
    primary indices        16, 17, 10, 19, 12, 13, 6, 15
    floors                 the already-measured values; not re-measured
    horizon / pre-roll     6.00 s, pre-roll 0.0
    Chrono build           source build, chrono-build/bin
    env config             go2_default_env_cfg unchanged, including all sigmas
                           and weights -- ESPECIALLY the ones now known to be
                           mis-balanced at convergence
    PPO config             seed 1, 2048 envs, 64 steps/env, 2000 iterations, K=16
    checkpoint rule        SELECTED + FINAL, both published
    verdict thresholds     unchanged, count governs
    checklist              unchanged

## The one varying input

CRM episode count **152 → 304**, entering only through the dynamics model. A
difference in the level-3 verdict is then attributable to CRM data quantity,
which is the kind of claim the paper's scaling appendix makes.

## What is NOT held fixed, and is inherent rather than a confound

The RL rollouts diverge regardless of seed, because the env's dynamics model
differs — that is the treatment, not a nuisance. And the merged dynamics model is
a different model in every weight; "more data" is the only *input* that changed,
not the only *thing* that changed.

## The temptation to name now

**Do not tune the merged run because it is the second one and we know more.** By
then we will know that the reward is state-dominated at convergence, that the
policy drifts after ~700 iterations, and that position_sigma is arguably
mis-scaled. Fixing any of that in the merged run is the right research move and
it destroys the comparison. If those fixes are worth making, they are worth a
THIRD run with its own registration — not a quiet improvement folded into the one
that is supposed to isolate data quantity.
