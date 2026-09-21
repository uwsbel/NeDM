# Our CRM against the HMMWV study's CRM

**Updated:** 2026-09-20. Every HMMWV value below verified directly in
`configs/hmmwv_crm_eval.json` and `src/nedm/hmmwv_crm.py`; every quadruped value in
`src/nedm/quadruped/constants.py` and `src/nedm/quadruped/terrain.py`. The framework
paper's own statement of these numbers is `Journal/2026/neural-dynamics-model/sections/
case_study_hmmwv.tex:66-97`, and it matches the config.

Short answer: **same soil model, different soil.** Four parameters differ deliberately
and are now backed by measurement; four differ with no recorded reason on either side.

## Shared

Constitutive model (`ElasticMaterialProperties`, same seven fields assigned in the same
order), density 1700 kg/m^3, Poisson 0.3, `mu_fric_s`/`mu_fric_2` 0.8, `mu_I0` 0.04,
`average_diam` 0.005 m, `d0_multiplier` 1.0, RK2 integration, ARTIFICIAL_BILATERAL
viscosity, ADAMI boundaries, `shifting_method` NONE, CFD step 5e-4 s, settling 0.1 s
(`SetActiveDomainDelay` on their side, `SetFreeFlowDuration` on ours -- a Chrono rename,
same thing), open-top box convention, BARZILAIBORWEIN solver, 100 Hz recording. Neither
study uses `ConstructMovingPatch`.

## Deliberate differences, with measured justification

| | HMMWV | Go2 | why |
|---|---|---|---|
| Young's modulus | 1.0e6 Pa | 5.0e5 Pa | sinkage must be visible under 15 kg, not 2400 kg |
| cohesion | 5000 Pa | 2000 Pa | same |
| `initial_spacing` | 0.08 m | 0.02 m | foot sphere is r=0.025 m; at 0.08 the foot is smaller than one particle |
| `artificial_viscosity` | 0.5 | 2.0 | at 0.5 a landing Go2 enters an undamped limit cycle; measured dose-response in `docs/state/lessons/chrono-versions.md` |

The spacing difference is the deepest one and is not really a parameter choice. Their tire
spans about 11.8 particles; our foot spans about 2.2. **The Go2 foot is smaller than the
smoothing kernel it presses into; the HMMWV wheel is three times larger than its own.**
The two studies do not resolve the same length scales, and no amount of parameter matching
would change that.

`constants.py:125-127` already carries the right caveat: we match the soil MODEL, not the
soil PARAMETERS.

## The soil choice, measured rather than asserted

Until now `soft` was justified by a comment. It is now justified by a measurement:
`tracking_spread.py`, north, 8 replicates per cell, 3 s at 0.5 m/s, rigid as the control.

| soil | CRM tracking | gap vs rigid |
|---|---|---|
| `soft` (ours) | 78.3 +/- 3.5% | **18.7 +/- 3.5** |
| `hmmwv_reference` (theirs, byte-identical) | 85.0 +/- 8.2% | **12.0 +/- 8.2** |

Rigid is 97.0 +/- 0.1% in both, as it must be -- it does not touch the soil.

The gap is about 7 points smaller on their soil, but paired across offsets that is
t = 1.97 at n = 8, which is suggestive and not conclusive. **The decisive difference is
variance, not mean.** sd 8.2 against 3.5 is a 5.6x variance ratio, so detecting the same
fine-tuning improvement on the HMMWV soil would take 5.6x the replicates. Against the
replicate budget in `docs/EVALUATION.md` -- 8 replicates to resolve 2.4 points -- that is
the difference between a tractable evaluation and an unaffordable one.

The per-replicate numbers show the mechanism. On their soil: 74.5, 75.4, 81.0, 93.7, 85.4,
81.9, 92.7, 95.6. That reads as two clusters, with several replicates within a couple of
points of rigid-ground performance and a gap as low as 1.5 points. **On stiffer soil the
robot sometimes finds a gait that barely notices the terrain.** Physically interesting;
useless as a benchmark, because the phenomenon the study exists to measure intermittently
disappears.

**Decision: keep `soft`.** The caveat for the paper is now quantitative rather than
hand-waved: our soil yields roughly 7 more points of gap, and far more consistently, than
the framework paper's soil would.

## Unexplained divergences

None of these has a recorded reason on either side. Two are cases where the HMMWV
implementation departs from its own design document, which is what makes them worth
raising rather than simply adopting.

1. **`free_surface_threshold`: 2.0 theirs (`hmmwv_crm_eval.json:53`), 0.8 ours
   (`terrain.py:65`).** Their design document proposed 0.8
   (`docs/hmmwv_crm_data_collection_pipeline.md:212`) and the implementation shipped 2.0.
   0.8 is also what the demo-derived test uses. So ours matches the demo and theirs is the
   departure.

2. **`num_proximity_search_steps`: 4 theirs (`hmmwv_crm_eval.json:60`), unset ours.** Their
   design document proposed 1 and stated that it should be raised "only after comparing
   force, sinkage, and trajectory metrics against the baseline"
   (`docs/hmmwv_crm_data_collection_pipeline.md:244`). No record of that comparison exists
   in the repo. This controls neighbour-list staleness and therefore contact forces
   directly.

3. **MBS/CFD exchange: every step theirs (2000 Hz), every 4th step ours (500 Hz).**
   `collect.py:144` uses `dt = 4 * step`. A genuine difference in how tightly the
   multibody and fluid solvers are coupled, unexplained on both sides.

4. **Active domain: (2.0, 2.0, 1.0) theirs (`hmmwv_crm_eval.json:39`), (1.0, 1.0, 1.0)
   ours (`terrain.py:129`).** Ours was inherited from `demo_ROBOT_Viper_CRM`. Theirs is
   larger in x and y and smaller in z. Since `SetActiveDomain` centres a box on EACH FSI
   solid, and their FSI solids are four wheels while ours are four feet and four calves,
   the two are not directly comparable -- but neither value was calibrated. Ours is being
   calibrated now (`docs/COST.md`).

## One place the HMMWV side may be the one with a problem

**Their bed is 0.25 m deep. We measured that a CRM bed deeper than about 0.22 m heaves
upward and carries bodies on it** (`docs/state/lessons/chrono-versions.md`), which is why
ours is 0.20 m. That threshold was measured at our 0.02 m spacing and our robot mass, so
it may not manifest at 0.08 m spacing under a vehicle. **Uncertain, and worth checking
rather than assuming** -- if it does manifest, it affects the HMMWV corpus.

## What this does not resolve

None of the above explains the cross-machine discrepancy in `docs/EVALUATION.md`: the same
Go2 on the same `soft` soil tracks 70.2 +/- 3.4% on a3 and 79.2 +/- 2.4% on north, while
rigid agrees to 0.1 points on both. Soil parameters are identical across those runs, so
the soil is not the cause.
