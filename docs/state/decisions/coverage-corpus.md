# A wider, more disturbed corpus makes the delivered policy worse

**Status:** scored 2026-09-20. Both cells, both selection rules.

## The 2x2

Volume is near-matched by design and the comparison is run under both selection rules, so
neither corpus size nor selection epoch can carry the result. Same 36-D state preset, same
architecture, same fine-tune budget, same box, same Chrono build.

```
                              vx        vy        wz      n
  baseline, val-selected   -39.5%    -10.9%    -20.1%    75
  baseline, roll-selected  -40.9%    -23.4%    -25.2%    74
  coverage, val-selected   -17.5%    +12.5%     -9.2%    74
  coverage, roll-selected  -30.2%    +13.6%    -16.1%    75
```

The coverage corpus loses in both cells, by 22.0 points under val selection and 10.7
under rollout selection, and both of its arms push LATERAL tracking above the policy they
started from.

Composition: the reference is 795 undisturbed episodes from one policy, 447,372 train
transitions. The coverage corpus is 422 episodes of which three quarters carry action
noise and body pushes, preprocessing to about 490,000 train transitions, so it has 9%
MORE data than the corpus it loses to. The dose ladder established that more transitions
help on this axis, which makes the volume explanation unavailable: coverage has more data
and delivers less policy.

## A second reading, which is the more useful one

Rollout selection is worth 1.4 points on the clean corpus and 12.7 points on the disturbed
one. Selection repairs far more when the corpus is worse, which matches the earlier
finding that its main effect is undoing off-axis damage rather than improving forward
tracking. A corpus that needs 12.7 points of repair was doing 12.7 points of damage.

## Why, and the test that would confirm it

The branch pool is drawn FROM THE CORPUS. A fine-tune starts each 15-step branch at a
recorded state, so the distribution of start states is the distribution of the corpus. On
a corpus that is three quarters disturbed, the optimiser is therefore asked to improve
behaviour predominantly at pushed and noise-perturbed states, while the verdict measures
undisturbed command tracking. That is a distribution mismatch between what was optimised
and what is scored, and it is introduced by the data rather than by the method.

Under that reading the coverage arms are not broken, they are optimised for a different
condition, and the honest claim is narrower than "disturbance hurts": coverage should
MATCH THE DEPLOYMENT DISTRIBUTION rather than maximise diversity.

The test is direct and not yet run: score the coverage-corpus policy on a DISTURBED
evaluation cell. If it beats the baseline-corpus policy there while losing on the clean
cell, the mechanism is confirmed and the result becomes a statement about matching rather
than a negative. If it loses on both, disturbance genuinely costs model quality and the
stronger claim stands.

## Bearing on random sampling in the wider framework

This is the case study's point of contact with excitation strategies that sample broadly.
Broad random coverage is the right default for a dynamics model intended to serve many
downstream tasks, because no single deployment distribution is known in advance. It is
the wrong default when the model exists to fine-tune ONE imported policy against ONE
evaluation condition, because then the deployment distribution IS known and the branch
pool inherits whatever the corpus contains. Nothing here argues against wide sampling in
general; it argues that this particular use of a learned model has a narrower requirement
than the general case, and that the requirement is matching, not breadth.

## The coverage result is confounded: the pushes were never shown to the model

Everything above rests on the disturbed corpus being a fair test of COVERAGE. It is not.
It is a test of coverage plus an unlogged input, and the second term is very likely the
whole effect.

The corpus is 70% pushed (28 of 40 episodes sampled carry non-zero `perturb_force_*`,
consistent with the design's "three quarters"). Those pushes are external forces applied
to the body. The model was trained on the 36-D `quadruped_crm_baseline` preset, which
contains **none of the six perturbation channels**. So for 70% of the data the true
dynamics were

    s_{t+1} = f(s_t, a_t, push_t)

while the model was fit to `f(s_t, a_t)` with `push_t` unobserved. Two transitions with
identical pose, joint state and action evolve differently according to a force the model
cannot see, so it cannot fit both and fits their average instead.

This repository had already written that argument down, twice, and applied it elsewhere.
`PERTURB_FIELDS` in `src/nedm/quadruped/dataset.py` says: "an unlogged disturbance is an
unexplained acceleration -- it widens coverage and makes the data unlearnable at the same
time. Logging it keeps 'is this a model input?' a modelling choice rather than a
collection one." The `quadruped_crm_payload` preset exists because the identical problem
was identified for carried mass. The reasoning was never carried across to perturbation,
even though the channels have been logged since the collector was written.

### What this does to the recorded conclusion

The claim was that a wider, more disturbed corpus delivers a worse policy, and the
mechanism offered was a distribution mismatch between the branch pool and the evaluation.
That mechanism is now the less likely of the two. An unobserved input explains the same
result more directly, and it also explains something the distribution story does not: the
horizon sweep shows the disturbed-corpus model is a far worse MULTI-STEP PREDICTOR --
0.166 against 0.034 at 0.30 s, crossing the predict-no-motion floor before 1 s where the
clean arms hold to about 3 s. Prediction quality on held-out rollouts has nothing to do
with which states the fine-tune starts from. It has everything to do with being asked to
predict an acceleration whose cause was withheld.

So: **"disturbance hurts" is not established.** What is established is that disturbance
hurts WHEN THE DISTURBANCE IS HIDDEN FROM THE MODEL, which is a statement about the state
definition, not about coverage.

### The test

`quadruped_crm_perturb` is added: the 36-D baseline plus the six logged channels, 42-D.
Reprocess the coverage corpus under it, retrain, fine-tune and score against the same
base. The raw episodes carrying populated channels are on north-ubuntu and a3-ubuntu.

If the 42-D arm recovers toward the clean corpus's -40.9%, the coverage conclusion is
overturned and becomes a much more useful one: a disturbed corpus is fine, and possibly
better, provided the disturbance is an input rather than noise. That would also make this
corpus usable for the broader-behaviour goal it was collected for, rather than evidence
against it.

If it does NOT recover, the distribution-mismatch reading survives and the coverage claim
stands on firmer ground than it does today.

Either way the current wcov numbers should not be quoted as a coverage result.


### CORRECTION: the reference corpus is pushed too, so this confound is common-mode

The section above assumed the reference corpus was clean. It is not. Its episode sidecars
record a push ladder: of 635 training episodes, 105 at 0 N and the rest spread over 24,
48, 72, 96 and 120 N, so 83% carry pushes. Per-row push-active rates are comparable
between the two corpora, about 3.6% for the baseline source and 3.1% for the coverage
source.

So the unobserved-input defect is real and it is COMMON-MODE. It cannot explain why the
coverage corpus delivers a worse policy, because the corpus it is being compared against
has the same defect at the same rate. The distribution reading demoted above is back to
being as plausible as it was, and the coverage question is open rather than explained.

What the defect does bear on is the HEADLINE result, which is trained on this same pushed
corpus. That is where the masking experiment belongs.
