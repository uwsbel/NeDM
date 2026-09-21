# What CRM collection costs, and which knob controls it

**Updated:** 2026-09-20. Measured on sbel, Chrono pin `698282895`, build hash `3b0bd530`,
soil `soft`, spacing 0.02 m, CFD step 5e-4 s, exchange multiplier 4.

The question this answers: SCM has an active domain that keeps cost off the size of the
terrain. Does CRM have the same, and can we use it to collect more cheaply?

Answer: yes, it is already on, and it is the only cost knob that matters. Patch size is
not.

## Patch length is nearly free

`quadruped/patch_cost.py`, 3.0 s of walking at 0.6 m/s commanded, after 1.5 s of warmup,
active domain held at the inherited 1.0 m:

| patch | SPH particles | ms/step | vs real time |
|---|---|---|---|
| 4 x 4 x 0.2 | 444,411 | 12.50 | 6.25x |
| 8 x 4 x 0.2 | 886,611 | 12.35 | 6.18x |
| 16 x 4 x 0.2 | 1,771,011 | 13.06 | 6.53x |

**Four times the particles costs 4.5% more per step.** Build time stays under 0.4 s
throughout, so a long bed is not expensive to create either.

This is not a surprise once the solver is read rather than guessed at. Every SPH kernel --
hashing, sorting, force evaluation, integration -- launches over `numExtendedParticles`,
the compacted active set, not over `numAllMarkers`
(`SphCollisionSystem.cu:350-411`, `SphFluidDynamics.cu:714-716`). Device arrays are
resized to the active set as well (`SphDataManager.cu:405-435`). The only per-step pass
over all markers is `UpdateActivityD` (`SphFluidDynamics.cu:284-287`), an O(N) position
test, and the 4.5% is essentially that plus the neighbour grid, which is still sized by
the full computational domain.

**Consequence for collection: stop sizing the patch to save money.** The bed should be
sized for the longest episode we want, with margin, because the marginal cost of extra
length is a few percent. The previous sizing rule traded episode length against cost on
the assumption that particles cost per step, and that assumption is wrong.

## The active domain is the whole cost

Same script, patch fixed at 8 x 4 x 0.2, varying only `SetActiveDomain`:

| active domain | ms/step | vs real time | relative |
|---|---|---|---|
| 0.5 m | 7.62 | 3.81x | 0.62x |
| 1.0 m (inherited) | 12.35 | 6.18x | 1.00x |
| 2.0 m | 28.21 | 14.11x | 2.28x |
| none | 73.90 | 36.95x | 5.98x |

A factor of 9.7 between the cheapest setting and no active domain at all, on a number that
was inherited from Chrono's Viper CRM demo and never calibrated for this robot. The
inherited 1.0 m is already buying a 6.0x speedup over the unapproximated solve.

The `none` row matters for a second reason: it still walks (1.29 m travelled, upright
0.999), so **the unapproximated solve is affordable enough to serve as the reference** for
calibration, at 37x real time. A candidate active domain can be checked against physics
with no active-domain approximation in it at all, rather than against a larger box that
might share the same bias.

Note what `SetActiveDomain(d)` actually means, because it is not one box around the robot:
the argument is a **full extent**, and the box is `+/- d/2` centred on **each FSI solid's
own origin**, all OR-ed together (`ChFsiFluidSystemSPH.cpp:282-287`,
`SphFluidDynamics.cu:181-217`). Our FSI solids are the four feet and four calves, so the
inherited 1.0 m is eight `+/-0.5 m` boxes whose union covers the robot and roughly half a
metre beyond it in every direction. The soil is only 0.2 m deep, so the z extent saturates
for every setting tested and the cost scales with the union's area, not its volume.

**This is not a free approximation, which is why it needs calibrating rather than
shrinking.** Outside the box a particle's velocity is zeroed every step
(`SphFluidDynamics.cu:245-246`), so momentum is destroyed at the face rather than
conserved, and the SPH sum is truncated at an envelope of only `2*h_multiplier*h`
(`SphDataManager.cu:578`), which is a free-surface-like artifact. A box chosen purely for
speed can stiffen the soil under the robot, and that biases the exact quantity this study
measures.

So the rule is the one already applied to artificial viscosity: **take the smallest value
that does not change the answer, and show that it does not.** That calibration is
`quadruped/active_domain_study.py`, run in two stages:

1. A cheap sweep against a 2.0 m reference to find the shape of the dependence and the
   smallest candidate that does not move the trajectory. Running now.
2. The candidate re-checked against `none` -- no active-domain approximation at all -- on
   the same cases. Stage 1 alone cannot separate "0.5 m agrees with 2.0 m" from "0.5 m and
   2.0 m share a bias," and at 37x real time the unapproximated reference is affordable.

Both stages are paired within a case and, crucially, measure the noise floor first by
running the reference configuration twice on identical inputs, because a paired difference
means nothing until you know what two identical GPU runs differ by.

Unpaired scatter is large enough to matter: three configurations that differ only in patch
size -- and so should be physically equivalent -- gave mean body-frame vx of 0.471, 0.426
and 0.362 m/s. That is +/-20%, which would swamp the effect being looked for. It comes
from the spawn sitting at a different absolute position on the particle lattice in each
run, not from the solver.

## The moving patch exists, and we should not use it

`CRMTerrain::ConstructMovingPatch(box_size, body, buffer_distance, shift_distance)` is the
direct SCM analogue: when the tracked body comes within `buffer_distance` of the front
boundary, particles behind it are relocated to the front and the computational domain is
translated. Particle count is invariant, the container walls and neighbour search are
handled internally, and `PatchMoved()` is informational only -- nothing is required of the
caller.

Three reasons it is the wrong tool here, in order of severity:

1. **It only moves in +x, hard-coded.** `CRMTerrain.h:64-65`: "The moving boundary is
   always assumed to be in the positive x direction," and `Synchronize` compares the
   sentinel's x against the front boundary with no travel-direction logic
   (`CRMTerrain.cpp:122-130`). Our command ranges are vx [-1.0, 1.5], vy [-1.0, 1.0],
   wz [-1.5, 1.5]. A robot that turns, strafes or reverses walks out of the patch
   sideways. A corpus deliberately built to cover lateral and rotational commands cannot
   use a mechanism that only follows one axis.

2. **Relocated soil is reset, not carried.** `SphParticleRelocator.cu:140-150` zeroes
   velocity, resets density to `rho0`, pressure to zero and the entire stress tensor to
   zero, snapping positions back onto the exact lattice. The robot would therefore always
   be stepping onto pristine, zero-stress, never-settled material -- a settling transient
   permanently co-located with the contact patch. For a study whose subject is foot-soil
   interaction that is a systematic artifact, not a detail.

3. **It is unexercised in the configuration we need.** Exactly one demo uses it
   (`demo_VEH_CRMTerrain_MovingPatch.cpp`) and that demo has no FSI bodies at all -- its
   tracked body is a plain `ChBody` on a prismatic joint. No demo or test in the tree
   combines a moving patch with an active domain, which is the combination a legged robot
   would require. There is also an unresolved index-bookkeeping asymmetry after the first
   shift (BCE markers shift by `(Ishift+1)*spacing` while the front and rear indices
   advance by `Ishift`, `CRMTerrain.cpp:155` against `:159-160`) that would need its own
   investigation before being trusted for many consecutive shifts.

Since patch length is nearly free, the moving patch would buy us memory we do not need at
the price of all three of those. **Use a long static bed and a calibrated active domain.**

## Settled: the bed does not meaningfully settle, so 0.1 s of free flow is enough

Measured on north, `settling_test.py`, three commands, total warmup held fixed at 2.5 s so
only the free-flow fraction varies.

| free flow | bulk compaction | surface | warmup wall |
|---|---|---|---|
| 0.1 s (inherited) | -0.000344 m | 0.000000 m | 13.2 s |
| 0.5 s | -0.000297 m | 0.000000 m | 22.9 s |
| 1.0 s | -0.000289 m | 0.000000 m | 34.8 s |
| 2.0 s | -0.000289 m | 0.000000 m | 59.0 s |

**Total bulk compaction is 0.3 mm on a 200 mm bed, about 0.15%, and it is complete within
0.5 s.** The difference between the inherited 0.1 s and a twenty-times-longer 2.0 s is
55 micrometres. There is essentially no settling to miss, so the worry that soil ahead of
the robot is frozen mid-settle and starts moving only when the robot arrives is not
supported: the soil is laid out at rest density and stays there.

The behavioural differences across arms -- mean vx moving by 0.01 to 0.07 m/s -- do not
converge as free flow lengthens and are within the chaotic scatter established in
`docs/EVALUATION.md` (sd 3.4 points of tracking, about 0.017 m/s at a 0.5 m/s command).
They are noise, which is what negligible settling predicts.

**Decision: keep `free_flow_duration` at 0.1 s.** A longer one costs 4.5x the warmup
(59 s against 13 s per episode, and that is pure overhead repeated on every episode of the
corpus) and buys 55 micrometres of compaction.

Two caveats recorded rather than buried. The surface percentile did not move at all in any
arm, which is a suspiciously round zero; bulk compaction is nonzero and varies with free
flow, so particle positions are genuinely live, but the surface statistic should not be
leaned on. And this was measured on one soil preset (`soft`) at one spacing; a denser or
deeper bed is not covered by it.

## Decided: the active domain is 0.5 m

Two independent 12-case ensembles on sbel, seeds 20260920 and 77, each against the
unapproximated solve, each with a null arm. Pooled, n = 21 after the gait gate:

| arm | mean_vx | speed | mean_z | cost |
|---|---|---|---|---|
| null (control) | 0.00000 | 0.00000 | 0.00000 | -- |
| **0.5 m** | +0.0044 (t 0.66) | -0.0082 (t -1.31) | +0.0012 (t 1.41) | **3.81x RT** |
| 1.0 m | +0.0005 (t 0.06) | -0.0109 (t -1.28) | -0.0003 (t -0.22) | 6.18x RT |

**Nothing reaches significance on either arm.** Neither box is distinguishable from the
unapproximated solve, and neither is distinguishable from the other. So the cheaper one
wins: 0.5 m runs at 3.81x real time against 6.18x, a 1.6x saving on every episode of
every corpus.

Resolution, stated rather than implied: a mean_vx bias above 0.013 m/s would have shown
at 2 se. Below that this measurement cannot see, and "no detectable bias" means exactly
that and not "no bias".

### The first run's significant result did not replicate

Run 1 reported `speed` for the 0.5 m box at -0.0182, t = -3.56, flagged as clear of the
null. Run 2 gave +0.0028, t = +0.26 -- opposite sign, nothing there. Pooled it is -0.0082
at t = -1.31.

That is the third result today that looked significant and did not survive replication,
after the machine effect and the bias-versus-speed correlation. All three came from small
samples of a chaotic system, and all three would have been believed if the replication had
not been run. The habit worth keeping is the cheap one: **a single ensemble is a
hypothesis, not a result**, and a second seed costs one job.

Worth noting what survived in the other direction: both arms and both runs put `speed`
slightly negative, four measurements with a consistent sign and none individually
significant. There may be a real speed bias of one to three percent that n = 21 cannot
resolve. It is not claimed here, but it is the reason to use the SAME active domain for
collection and for the Chrono evaluation -- a common-mode bias cancels in the paired
comparison the study actually reports, and an uncommon one does not.

### One number from these runs that should not be used

The ensembles' own ms/step figures disagree between runs by 2.3x on identical
configurations -- the null arm reads 73.6 ms/step in run 1 and 169.2 in run 2, same bed,
same settings. Something else was contending for the GPU. Cost numbers come from
`patch_cost.py`, which was run on an idle machine for that purpose; the ensemble's timings
are incidental and unreliable.
