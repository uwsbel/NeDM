# Future work

The study is complete (see ../README.md). What remains, roughly in order of value.

## Would strengthen the paper

- **Leave-one-out ablations in the same surrogates.** The recipe's pieces were each
  justified by an ablation, but in different surrogates and at different times. Removing
  one piece at a time (2 s branches, rollout training, 1024 rollouts, the iteration-1000
  stop) in the same three surrogates, scored on one machine, would make the ladder
  additive.
- **A one-step surrogate at the old analytic recipe** (0.30 s, weight budget 1.0), to
  reproduce the first attempt's analytic result inside this pipeline.
- **2048 rollouts at iteration 1000** (only measured at the old stop).

## Would extend the method

- **Robustness beyond the corpus.** A corpus with harder pushes, keeping the push windows,
  with the push force as a model input (the `crm_perturb` preset exists).
- **The accuracy plateau.** Every surrogate sits near 0.47 rollout error at 2 s whatever
  is scaled or added; a longer context or a different objective is the next thing to try.
- **Re-fit the guard** against Chrono-scored labels for all logged runs (candidates:
  reward drop, value loss, fraction of branch-steps outside the corpus).

## Infrastructure

- **Checkpoint/resume in `finetune.py`.** Preemptible partitions restarted PPO runs from
  zero repeatedly; resumable runs would make euler's `research` partition usable.
- **The Chrono GPU fault** (`SphBceManager.cu:543`, about 1 episode in 720): contained,
  root cause unproven. Backport the upstream error-flag fix and add a bounds check that
  names the marker, then report upstream.
- **Paths shorter than 6 s** for the three held-out paths too wide for the soil bed, so the
  evaluation uses 40 of 40.

## Decided

- **No harder-push corpus for this paper** (Kyle, 2026-09-24): the claim is scoped to
  tracking, with push robustness held at the base policy's level.
