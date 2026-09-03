# Open questions

Live and unresolved. Each says what would settle it. Delete an entry when it is
answered, and move the answer into
[`architecture.md`](architecture.md) or a `../lessons/` file.

**Updated:** 2026-09-02.

## Does a global pooled `z2` survive at 256²? **Measured: no.**

**Answered 2026-09-03 on synthetic data.** Localisation error in pixels on 256²
frames with a 15x7 px vehicle (0.16% of pixels) at a known position, 3-6 px
distractors, textured background. Encoder frozen, a probe fitted per
representation, held-out evaluation. Chance, meaning predict the image centre,
is **87.03 px**.

| Representation | px/cell | random init | after 400 steps of unweighted recon |
|---|---|---|---|
| stage 1 (128²) | 2 | 2.09 | 1.84 |
| stage 2 (64²) | 4 | 1.67 | 1.92 |
| stage 3 (32²) | 8 | 2.07 | 3.30 |
| stage 4 (16²) | 16 | 4.39 | 11.06 |
| stage 5 (8², the plan's map) | 32 | 7.11 | **84.64** |
| **pooled `z2` (128-D)** | — | **75.69** | **87.36** |

**Pooling is the cliff, not resolution.** At random initialisation every spatial
map localises to 1.7-7.1 px, degrading gracefully with resolution, while the
pooled `z2` lands at 75.7 px against 87.0 chance: distinguishable from chance
but useless, about an order of magnitude worse than the map it is pooled from.
This holds with **no training at all**, so it is a statement about the
architecture rather than about optimisation. The pre-declared fallback, keeping
a spatial map for the planner while `z2` serves the dynamics token, is therefore
the right call and is now evidenced rather than hedged.

**And unweighted reconstruction warm-up is worse than no warm-up.** Stage 5 goes
from 7.11 px at random init to 84.64 px, i.e. chance, after 400 steps of plain
4-channel reconstruction; stage 4 goes 4.39 to 11.06. The vehicle is 0.16% of
pixels, so an unweighted MSE has almost no incentive to represent it and the
deep layers spend capacity on the background that dominates the loss. **Training
without foreground weighting actively destroys the signal the probes exist to
measure**, which is direct empirical justification for plan §5 making the
foreground-weighted and vehicle-heatmap losses *mandatory* rather than optional.

**A correction worth keeping.** An earlier reading of this file argued the 8²
map "has already destroyed most of the localisation signal" because the vehicle
spans under half a cell. The measurement refutes the magnitude: a soft-argmax
readout over a smooth response field interpolates between cells and reaches
sub-cell precision, so stage 5 is degraded, not destroyed. The geometric
argument had the ordering right and the size wrong.

**What this cannot say.** Synthetic frames with an analytically placed bright
rectangle show the signal is *there to be found* and the architecture can
represent it. They do not show a real encoder trained on real Chrono frames will
find it: real vehicles are not colour-distinct from terrain, occlusion and
shadow exist, and these distractors are trivially separable. The random-init
figures are also a floor *for this readout*; a stronger probe might extract more
from `z2`. And the warm-up finding rests on 400 steps, though the mechanism
argues longer training makes it worse rather than better.

### The enforcement A/B is underpowered by construction, and the fix is cheaper

**Computed 2026-09-03, before the enforced arm finished, so the result is not
chosen after seeing it.**

The unenforced arm produced **1 collision in 100**. Ask what a 0/100 enforced arm
would prove:

| | |
|---|---|
| P(0/100) **if enforcement does nothing** | **0.366** |
| Fisher exact, 1/100 vs 0/100 | **p = 1.000** |
| Fisher exact, 1/100 vs 0/**1000** | p = 0.091 |

**A clean 0/100 is the single most likely outcome under the null.** It is not
weak evidence for enforcement; it is *no* evidence. And the arm cannot be
rescued by running longer — even a perfect 0/1000 against the existing baseline
still misses significance.

n per arm for 80% power at α=0.05, **assuming enforcement is perfect**:

| baseline collision rate | n per arm |
|---|---|
| 1% (what we have) | **1025** |
| 5% | 162 |
| 10% | 95 |
| 20% | **43** |
| 40% | 19 |

**So raising the baseline rate is both more powerful and cheaper.** At a 20%
baseline, 43 runs per arm settles it — fewer than half the 100 per arm we are
running now to learn nothing. The lever is scenario difficulty (tighter
corridors, denser obstacles, less initial clearance), not more repetitions.

**But the deeper error is the metric, not the sample size.** A binary
collided/didn't discards almost everything each episode knows. We already log
`min_centreline_clearance_m` and `min_asset_clearance_m` per episode — continuous
quantities to which **every** run contributes, where enforcement should shift the
whole distribution rather than only its extreme tail. At n=100 per arm a
two-sample test detects a shift of ~0.4 SD at 80% power, and the existing runs
already carry that data.

**Rank the arms by clearance distribution; report the collision count as a
descriptive footnote.** A rare binary event is the least informative function of
a continuous measurement we already have.

The honest statement if the enforced arm returns 0/100 is *"consistent with
enforcement working, and underpowered to demonstrate it"* — never "enforcement
works."

### What pooling keeps and what it destroys

**Second probe, occupancy/class masks, 2026-09-03.** Same generator, same
pyramid, 3 classes at 64² with priority max-pool so 3-6 px rocks survive
downsampling. Class fractions: background 0.9928, rock 0.0042, vehicle 0.0030.

| representation | rock IoU | vehicle IoU | vehicle recall | rock recall |
|---|---|---|---|---|
| stage 2 (64²) | **0.621** | **0.678** | 0.958 | 0.764 |
| stage 3 (32²) | 0.352 | 0.489 | 0.989 | 0.634 |
| stage 4 (16²) | 0.065 | 0.323 | 0.993 | 0.124 |
| stage 5 (8²) | 0.003 | 0.134 | 0.960 | 0.004 |
| **pooled z2** | **0.000** | 0.082 | **0.549** | **0.000** |

**Pooling keeps *whether*, destroys *where*.** Pooled `z2` reaches vehicle recall
0.549 at IoU 0.082: it detects a vehicle in about half of frames and then smears
the prediction over far more area than the vehicle occupies. A global latent
retains a weak *presence* signal and no usable extent or position, which is
consistent with the localisation arm rather than in tension with it.

**Rocks are the sharper result and the one that matters.** IoU and recall are
both **identically 0.000** at `z2`. Rock count barely varies between frames, so
"are there rocks" carries almost no information and everything distinguishing one
frame's rocks from another's is positional. Pooling destroys exactly that. So the
split is not "occupancy survives better than localisation" — it is that pooling
preserves the presence of one distinctive object and destroys everything about
small, numerous, positionally-defined ones. **For a planner-facing
representation, rocks are the class that matters, and they are the class that is
completely gone.**

**Do not lean on the vehicle number at all.** The generator places *exactly one*
vehicle in every frame, so a detector that always fires scores recall 1.0 at
near-zero IoU. Observed `z2` is recall 0.549 at IoU 0.082 — nearer that
degenerate behaviour than to detection, and in fact *worse at recall than a
constant predictor*. Against a class present in every frame, recall is not
evidence of anything. The rock row carries the result; the vehicle row is
included for completeness and should not be quoted as a capability.

**Probe capacity is ruled out by direction.** The 1×1 conv probes have C·3+3
parameters, so 99 at stage 1 rising to 1539 at stage 5: deeper stages get *more*
probe capacity and do *worse*. The `z2` probe is an MLP with ~6.4 M parameters,
about 4000× the stage-5 probe, and still loses. The asymmetry flatters `z2`.

**And the training arm reproduces the localisation finding independently.** 400
steps of unweighted reconstruction drives stage 5 and `z2` to exactly 0.000 on
both classes, worse than random initialisation. Two different probes, two
different quantities, same conclusion: **§5's foreground-weighted and heatmap
losses are mandatory, not advisory.**

Same caveat as the localisation arm. Synthetic masks say the signal is there to
be found; a real encoder on real Chrono frames faces occlusion, shadow, and rocks
that look like ground.

**This settles the probe bars (§14, and the open decision below).** A trained
encoder that cannot beat its own random initialisation on the localisation probe
has failed, whatever absolute threshold is chosen. That bar is anchored to a
measurement available today and, by construction, cannot be tuned after the
pilot results are seen.

## Original framing

Study 1's 64-D global latent worked on a 128² scene where the object covered 3%
of pixels. Study 3 has a 256² arena where the vehicle is ~15×7 px and small rocks
are 3–6 px. **Settles it:** the WP1 perception pilot's occupancy and localization
probes, run from both the global latent *and* the encoder's pre-pooling spatial
feature map, quantifying what pooling destroys. A fallback is pre-declared —
keep a low-res spatial map (or factored `z_layout`/`z_vehicle`) as the
planner-facing representation while the global `z2` continues to serve the
dynamics token.

## Can the vehicle marker be detected reliably enough for the localization probe?

6/10 layouts at 256². **Settles it:** either a larger marker footprint or
detection tuned on collection-light (55°) frames rather than probe-light (80°)
frames. Blocking for G1.

## Is the 16-token context long enough for a gaited system?

Only relevant if the quadruped case study proceeds. A gait cycle is ~0.3–0.5 s;
16 tokens at 50 Hz is 0.32 s. If phase is not inferable from context, the model
cannot predict touchdown. **Settles it:** decide before collection — feed the
controller phase/clock into the token, or lengthen the context. See
[`../progress/future-case-studies.md`](../progress/future-case-studies.md).

**Measured 2026-09-02, and it does not transfer.**
`scripts/quadruped_wp0_gait.py cycle` reports RoboSimian's shipped
`walking_cycle.txt` as 19,164 rows at dt 1 ms, 32 joints, **19.16 s per cycle**.
That is the period by construction, since `RS_Driver` replays the file on a
loop; the autocorrelation cross-check scatters from 5.9 to 18.9 s because one
period barely fits the record, so do not quote it.

19.16 s against a 0.32 s context is **60x short**, and covering it would need
~960 tokens. For RoboSimian, feeding the controller clock into the token is
therefore the only option.

**But RoboSimian is a slow statically-stable walker and the study targets a Go2
dynamic trot.** At the plan's 0.3-0.5 s that is 15 to 25 tokens at 50 Hz, so
lengthening the context stays entirely viable there, and 16 tokens is already
within a few of sufficient. The prototype measurement bounds the machinery, not
the answer. **Still open for Go2**, and it needs a Go2 gait period. That was blocked while
both boxes ran pychrono 9.0.0, where `ChParserURDF` fails to load. Under the
`nedm` environment (pychrono 10.0.0) the parser is exposed and verified, so
**this is now answerable**: import a Go2 URDF, run a trot, and measure the
period the same way `scripts/quadruped_wp0_gait.py cycle` does for RoboSimian.

## Where does the quadruped's seed controller come from?

You cannot train locomotion in Chrono + CRM (PPO needs ~10⁸ steps; CRM runs
below realtime) and random actions produce only collapse data. Three candidate
paths are ranked in
[`../progress/future-case-studies.md`](../progress/future-case-studies.md); the
recommendation is a scripted gait as a WP0-style vertical slice, with
RoboSimian's in-tree `walking_cycle.txt` as the prototype. **Settles it:** a
privileged scripted gait walking on rigid ground, then CRM, with zero learning.

## Does the `z1` gap close with more data, or is there a floor?

Study 1 measured `error ≈ data^-0.6` at 0.5 s while the matched state-only model
was saturated, and extrapolated that another ~5–6× data (~15 min of collection)
would bring NRD near the state-only floor. It was never run. Residual factors
named in the notes: multi-task loss competition and latent-drift feedback.
**Settles it:** one collection run and one training run — cheap, and it would
turn an extrapolation into a result.

## Is the figure and bibliography pipeline reproducible?

The manuscript's `\graphicspath` points at directories that are gitignored and
absent, filenames differ from what `scripts/figures/` emits
(`hmmwv_rl_reward.pdf` vs `hmmwv_rl_reward_L8.pdf`), and the shared `BibFiles/`
repo is not referenced anywhere. **Settles it:** check in the copy/rename step
and record where `BibFiles/` comes from. See
[`../machines/manuscript.md`](../machines/manuscript.md).
