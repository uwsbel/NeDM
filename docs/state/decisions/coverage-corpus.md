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
