# Contact mode: temporally separable, not amplitude separable

**Measured 2026-09-03. This is the methodological core of the case study.**

## Contact is not detectable by a force threshold on soil

| | flat | CRM |
|---|---|---|
| force histogram | **cleanly bimodal** — 42.4% of samples exactly 0.0 N, then a gap, then a loaded band to 185 N | **monotone decay, no local minimum, no gap**, and it goes *negative* to −12.9 N |

**There is no true zero on CRM because the foot never leaves the kernel.** The
standoff reappears here as the absence of an unloaded state.

**The decisive test was threshold sensitivity, not histogram shape.** Detected
switch rate against the rate implied by the gait's own spectral peak, where 1.0×
means every detected transition is real:

| | flat | CRM |
|---|---|---|
| plain threshold, 20 N | 1.02× | **1.66×** |
| Schmitt trigger, 5/60 N | 1.01× | **0.96×** |

On flat, duty cycle slides 0.51 → 0.36 across thresholds 1–40 N while switch rate
stays flat at 0.1455 — a genuine plateau, so the mode is threshold-independent. On
CRM **there is no plateau anywhere**; switch rate peaks mid-range, which is the
signature of a threshold cutting through noise and inventing transitions. **A
plain threshold on CRM manufactures 66% spurious contact events.**

**A binary mode is still the right abstraction — it needs hysteresis.** The mode
is *temporally* persistent even where it is not *amplitude* separable. That
distinction is the finding, and it is what a contact-conditioned model needs to
get right on deformable terrain.

## The trot was recovered without being told to look for one

Mode occupancy, extracted from force alone:

| | dominant modes | coverage |
|---|---|---|
| flat | `1001` (FL+RR) 33.6%, `0110` (FR+RL) 32.1% — **66% in two modes** | 10 of 16 |
| CRM | `0110` 28.7%, `1001` 20.7% — **49% in two modes** | **15 of 16** |

**The two diagonal pairs of a trot, dominating both terrains, recovered without
supplying a gait model.** Independent evidence the extraction is correct.

**And the occupancy differs by terrain**, which is a candidate signal in its own
right: flat concentrates in two clean diagonals, CRM spreads across nearly every
mode. **Soil compliance blurs the phases.** Contact mode is therefore not
independent of terrain, so conditioning on both is meaningful rather than
redundant.

## Contact mode is a derived function, not a stored column

`dataset.contact_mode(force_series)` — **not** materialised into the CSVs.

`foot_*_force_fz_n` is already in every row, so the mode is a pure function of
stored data. Writing it in would freeze **two tuned constants** (5 N and 60 N)
into the dataset where no consumer could revisit them without recollecting — and
they are tuned on one episode, at one spacing, for one gait.

Same rule as ungated slip and raw force, and it applies harder here because there
are two parameters rather than one. **Reversible at zero cost in both
directions**: columns can be added by a pass over the CSVs, never by re-simulating.

## Soil memory IS reachable — bigger feet flip the sign

Collision sphere 0.022 → 0.050 m, standing, training soil, each geometry compared
against **its own** rigid reference (a larger sphere shifts the origin-to-contact
offset, so raw sinkage is not comparable across geometries):

| geometry | radius | R / kernel support | rigid ref | CRM | penetration |
|---|---|---|---|---|---|
| stock | 0.022 | 0.55 | −0.0067 | −0.0296 | **−22.9 mm (floats)** |
| bigfoot | 0.050 | **1.25** | −0.0347 | −0.0296 | **+5.1 mm (penetrates)** |

**A sign reversal, not a magnitude change** — much harder to explain any other way.

**The CRM foot height is identical in both runs (−0.0296).** The kernel holds the
origin at the same absolute height regardless of radius; what changes is where
that height sits relative to true contact. That is the mechanism, confirmed
directly.

**Correction:** 0.050 m is **1.25×** the kernel support (2h = 0.040 m), not 2.5×
— that figure was the diameter ratio. **The flip occurs at a radius only slightly
above the support**, which is a stronger result than needing 2.5×.

Enlarged feet remain a **diagnostic only**; the policy was trained on stock
geometry. The isolated asset tree is symlinked and the shared URDF is untouched.

## Collection

200 rigid in **103 s** at 8 concurrent — 6.0×, matching the measurement, so rigid
volume is effectively free. CRM sequential at ~71 s each.

Spawn varies x ∈ [−1.5, 0.5], y ∈ [−0.3, 0.3], heading ∈ [−10°, 10°]. **y and
heading are bounded deliberately**: the sinkage probe's undisturbed control patch
sits at (0.0, 1.2), and a wider spread would walk the robot into its own
reference and corrupt `surface_disp`. Maximum lateral excursion is ~0.74 m against
that 1.2 m patch.

## The ablation needs a fifth cell: PERMUTED mode

**Added 2026-09-04, before any cell was run.** The four-cell design is confounded
by construction and cannot support the claim it was built to test.

At our data scale an added input tends to improve rollout **whether or not it
encodes real physics**, because we sit on the steep part of the data curve and
extra capacity absorbs variance. Every cell that carries mode also carries the
sixteen extra input dimensions that come with it. So "terrain+mode beats
terrain-only" is equally consistent with two worlds:

- contact mode carries physics, or
- contact mode carries nothing and the model benefited from the width.

This is the same confound as the rejected cross-product encoding — context tangled
with capacity — reappearing one level up, after we had already rejected it once in
its more obvious form.

### The control

A fifth cell with the mode code **permuted within each episode**: same width, same
sixteen dimensions, same one-hot structure, same everything, but the mode assigned
to each timestep carries no information about the actual contact state.

| cell | what it holds |
|---|---|
| terrain + mode | the claim |
| terrain + **permuted** mode | the same capacity, none of the signal |

Permute *within episode* rather than drawing from the pooled empirical
distribution: it preserves each episode's mode marginal exactly and destroys only
the temporal correspondence with the force series. Drawing from the pool would also
destroy the per-episode composition, adding a second difference to argue about.

**Seed the permutation and record the seed in the run config.** A control that
cannot be reproduced is a control nobody can check.

### It has a three-point null, which is what makes it interpretable

If capacity alone helps, permuted lands **between** terrain-only and terrain+mode.
If capacity is all there ever was, permuted lands **at** terrain+mode. If the
signal is real, real mode beats permuted.

Whichever way it falls, the result reads. And if permuted matches real mode, the
case study's central claim is not supported by this experiment — **a result, and a
much better one to find ourselves than to have a reviewer ask about.**

Cost: one training run, no new data. The permutation is a transform on the
precomputed per-timestep mode column, applied at load.
