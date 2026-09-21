# Lessons

Each entry is something that actually cost us, and the rule it produces. Add to this when
something goes wrong, not when it is fixed.

## Measurement

**A number that lives only in scrollback is not a result.** Throughput figures
(11,075 tr/s, 4.17 tr/s, 2,657x) were real measurements that had only ever been printed
to a terminal. An audit correctly reported them as nonexistent, because nothing in the
repository contained them. They reached a slide first.
→ Every presentable number has an artifact and a command that regenerates it.

**A number that sounds right is not therefore measured.** "3.1% error at 0.3 s, usable to
1-1.5 s" was written into a decision doc and a slide with no measurement behind it
anywhere. The later sweep put the true value at 0.034 at 0.30 s, so the figure was nearly
correct -- which is precisely why it survived unchallenged.
→ Plausibility is not provenance. Check the artifact, not the ring of the number.

**A constant in code is not a derivation.** "24.1 h of Chrono wall-clock" was a hardcoded
f-string literal inside the script that appeared to compute it, and a break-even of "about
four fine-tunes" was derived from it. The real figure is 37.0 h and 6.2 fine-tunes.
→ If a script prints a quantity, the script computes it.

**A metric can change meaning.** `max_val_batches` took a PREFIX of the validation split
until 2026-09-17, then a random sample. The prefix degraded with corpus size -- 4 of 8
command families at 88k windows, 1 of 8 at 2M -- which is the same axis the dose ladder
varies. Every `val_loss` comparison spanning that date is invalid, and the rho = -0.80
result has a confound aligned with its own independent variable.
→ Version metric definitions with the data they measure.

## Experiment design

**A confound that is checked and found immaterial can be quoted; one that is flagged and
left open cannot.** The selection comparison paired arms trained on two different hosts.
Repairing it cost one training run, one fine-tune and one verdict, and moved the result by
0.3 points. That 0.3 is what makes the finding citable.

**"Different machine" is not an explanation for a 30% difference.** It was accepted as one
for several hours. Two same-host runs later reproduced each other to 0.25%, and the real
cause was the metric change above.
→ An effect far larger than plausible nondeterminism has a mechanism. Find it.

**An unlogged disturbance makes data unlearnable.** 83% of the reference corpus carries
external pushes the 36-D state cannot see, so the model fits the average of "pushed" and
"not pushed", which is wrong for both. The repository had already written this argument
down for payload mass and built a preset for it; nobody carried it across to pushes.
→ Every cause of a state change is an input, an initial condition, or absent.

**A ladder must vary one thing.** The abstraction ladder was presented as "more channels
is worse", but the 40-D contact flags are thresholded foot forces the 48-D arm already
carries in full. Ordered by information added, the result is non-monotonic.
→ Check that the axis you name is the axis you varied.

## Implementation

**A flag accepted and ignored reads exactly like a working one.** `--target-dw` was
silently skipped on one code path. A checkpointing flag was set on a wrapper object that
never read it. Both produced runs that completed and reported numbers.
→ Every knob asserts it took effect, and fails loudly when it cannot.

**Deleting rows breaks sequences.** Training windows are built inside episodes as
`length - sequence_length + 1`. Removing rows from the middle produces windows that span a
temporal jump, silently.
→ Cut episodes into segments; never delete in place.

**Splitting is not free.** Window count is superlinear in episode length: a 1,475-row
episode gives 1,347 windows, but seven 190-row segments give 434.
→ Size the excitation schedule around the training window, not the other way round.

**Inherited conventions are liabilities.** The Chrono harness negates joint positions,
velocities and targets and no source records why. A sign flip bug followed.
→ Establish sign and ordering by round-trip test, never on faith.

**The plant filters your excitation.** Per-step i.i.d. noise on a position target is
rejected by the PD loop and the robot's inertia, producing action variance with no state
response -- which teaches the model that actions barely matter, the opposite of what
fine-tuning needs.
→ Excite inside the bandwidth the system responds to.

## Process

**Scoring is the bottleneck, not optimisation.** A fine-tune is about a minute; its Chrono
verdict is 25 to 90. The method collapses the cost of optimising and leaves the cost of
selecting untouched.

**The cheap proxy does not rank abstractions.** Rollout fidelity ranks data volumes at
rho +0.90 and ranks abstractions backwards. Selection within one state definition is
sound; selection between definitions is not.

## Silent failure modes found during the rebuild inventory

**A buffer sized before the data is read.** `preprocess.py` pre-sizes the output memmap
from `dataset_index.json`'s declared `rows` BEFORE any CSV is opened, while the actual
written length comes from the CSV. Any row-dropping scheme that does not also update the
index leaves uninitialised rows at the tail of a memmap -- no error, no warning, just
numbers that were never written being trained on.
→ Size from what was read, or verify the fill reached the allocation.

**A guard keyed on a filename.** `crm_verdict.py` refuses to pair score files across hosts,
which is correct and load-bearing. It extracts the host by parsing the FILENAME
(`..._<host>.json`). A file that does not follow that convention yields the same token on
both sides, the comparison passes, and the cross-host guard silently does nothing.
→ A guard that can pass vacuously is worse than no guard; key it on recorded content.

**A threshold hardcoded inside a metric.** The fall test `min_z < 0.20` is a literal inside
`summarise()`. Changing it changes what "fall" means with nothing recording that it moved.
→ Thresholds that define a metric belong in `metric_defs`, stamped into the manifest.

**RNG draw order is part of the contract.** The collector documents that the ORDER of
random draws is load-bearing for replay: multiplying a draw by zero still consumes it, and
reordering draws inside an existing corpus's seed changes every subsequent value. A
force-only episode once replayed with different forces because a torque draw moved.
→ Rewriting a sampler breaks replay of existing corpora even when the distribution is
identical. Version the sampler and never edit one in place.

## Removing a conda env can break binaries that RUNPATH into it

Standardising the fleet meant retiring sbel's redundant `nedm-src`, which held exactly one
package `nedm` lacked: the conda Chrono trap. Removing it immediately broke pychrono in
`nedm`, which had just been pointed at the source build.

The source build's RUNPATH is
`/home/kyle/Documents/sbel/chrono-build/lib:/home/kyle/miniconda3/envs/nedm-src/lib:` --
it resolves `libpython3.12.so.1.0` and `libtinyxml2.so.11` out of the env that was
deleted. Nothing in the package list showed that dependency, because it is a link-time
path baked into the ELF, not a declared requirement.

Fixed with a compat directory of symlinks at the old path, which is a shim rather than a
repair; the real fix is a rebuild with a corrected RUNPATH, or patchelf, which was not
installed. Recorded so the shim is not mistaken for the intended state.

→ Before removing an environment, check what RUNPATHs into it: `readelf -d <so> | grep
RUNPATH`. A package list does not show link-time paths.
→ `parsers` failing is not cosmetic: it is what loads the Go2 URDF.

## Standing up a Chrono scene: four settings that all fail the same way

Establishing the sign convention took six attempts, and every failed one looked like "the
policy cannot stand". None of them were the policy. Recorded because the symptom is
identical in all four cases and gives no hint which is wrong.

**Collision envelope and margin are GLOBAL defaults read at model construction.**
`ChCollisionModel.SetDefaultSuggestedEnvelope/Margin(0.0025)` must run before any body is
built. Setting them afterwards leaves the ground and the robot with whatever was in force
earlier.

**`apply_pd()` is a no-op on the position plant.** With `actuation="position"` the joint
is driven by a constraint and the policy's gains are never applied, so it is a different
controller than the one the policy was trained against. Use `actuation="torque"` and call
`apply_pd()` EVERY PHYSICS STEP, not every control step. `PD_KP/PD_KD` are 20.0/0.5, which
is exactly rl_sar's `rl_kp`/`rl_kd`.

**The URDF references meshes by relative path**, so the process must `chdir` to the URDF's
directory around construction or the collision geometry silently fails to load.

**Spawn must clear the FULLY EXTENDED leg, not the standing height.** This was the real
one. The parser starts every joint at zero, which is legs straight down and 0.42 m of
reach. Spawning at a plausible standing height of 0.38 m put the feet at z = -0.046
against a ground surface at +0.05, so the robot began the episode already penetrating the
floor, settled onto its trunk at 0.093, and stayed there with its legs through the ground.
Four contacts were reported the whole time, which made it look like collision was working.
`measure_leg_reach()` in terrain.py exists for exactly this.

The diagnostic that finally separated them was printing foot height at spawn, before any
step. A configuration that is already invalid at t=0 cannot be diagnosed from what happens
after t=0.

## A scale that applies to an observation the policy does not have

rl_sar's config carries `lin_vel_scale: 2.0` and `commands_scale: [1.0, 1.0, 1.0]`. The
first scales an OBSERVED base linear velocity -- which this policy does not have, and its
absence is why its observation is 45 wide rather than 48. Applying it to the command
doubles the command.

It was caught because the robot walked at 0.989 m/s when asked for 0.5, and 2.0 is not a
subtle factor. With `commands_scale` the tracking is 0.5 -> 0.473 and 1.0 -> 0.989.

The old policy's notes record the mirror-image mistake: yaw scaled by `lin_vel_scale`
instead of `ang_vel_scale`, invisible because the yaw command was identically zero. Both
are the same failure -- a scale applied to the wrong term -- and neither shows up as an
error, only as a number that is wrong by a clean factor.

## Two places computed where the soil was, and they disagreed

`build_crm` placed the SPH bed with `Construct(size, ChVector3d(patch_x/2 - 0.6, 0, 0))`.
The collector's far-end spawn rule placed the robot at `-sign(vx) * (patch_x/2 - margin)`,
which is the near edge **of a bed centred on the origin**. The bed was not centred on the
origin. For `patch_x = 8` it ran x [-0.600, +7.400], and a +x episode spawned at x = -3.500:
2.9 m off the front edge, in open air.

Neither number is wrong on its own. They were written at different times for different
reasons -- the 0.6 m offset so a robot starting at the origin has a little soil behind it,
the far-end spawn so an episode can use the whole bed instead of half -- and each is
correct against the convention its author had in mind. What makes this class of bug
expensive is that the two conventions never meet in one place, so there is nothing to read
that looks wrong.

**It would not have announced itself.** `MIN_BASE_Z_M = -0.5` does catch the free-fall, so
a +x episode truncates after roughly 50 rows, which is below `min_segment_rows = 256` and
therefore dropped entirely. But the bed's offset is in +x only, so an episode commanded in
**-x** spawns at +3.5, which IS on the bed, and collects normally. The corpus that comes
out is not empty and does not error. It is silently missing every forward-commanded
episode while keeping every backward one, on a command range of vx [-1.0, 1.5] that is
mostly forward. Gate 4 would pass, because held-out OOD is measured against the
distribution actually collected.

A corpus that is wrong in a way its own gates cannot see is worse than one that fails
loudly, and this one would have been found only after full-scale collection on hpcfund,
as "the policy cannot walk on CRM."

### What changed

`crm_patch_bounds()` in terrain.py is now the single source of truth, and the bed is
centred on the origin so the spawn rule and the bed share one convention. Three guards,
because the arithmetic being right today is not the same as it staying right:

- `assert_patch_where_expected()` compares the built terrain's `GetSPHBoundingBox()`
  against what `crm_patch_bounds` promised, so a Chrono convention change is caught at
  construction rather than inferred from bad data.
- `assert_spawn_on_patch()` checks the spawn against the built bed before any simulation
  time is spent.
- `validity.first_invalid(..., bed=...)` truncates at `off_bed`, which fires as the robot
  crosses the edge rather than half a second later when it has fallen far enough to trip
  `base_height`. Without it the rows between the edge and the fall are free-fall recorded
  as locomotion.

The general rule: **a derived quantity that two modules both need should be computed once
and imported, not recomputed from the same inputs.** Recomputation is how they drift, and
drift between two individually correct conventions produces no error anywhere.

### Chrono's Construct convention, since it is what made the offset easy to misread

In `ChFsiProblemCartesian::Construct(box_size, pos, side_flags)` the `pos` argument is the
**centre in x and y** but the **bottom in z**. Verified on the pinned build:
`Construct((8, 4, 0.2), (3.4, 0, 0))` yields x [-0.600, +7.400], y [-2.000, +2.000],
z [+0.000, +0.200].

### Found by accident

This surfaced from a cost benchmark, not from a correctness check. The benchmark varied
patch size and active-domain size and reported **bit-identical** dynamics for all five
configurations -- same `mean_z`, same `up`, same 0.029 m travelled. Identical output across
configurations that should differ is the signal; the fall-through was the explanation. A
sweep whose arms do not differ has told you something even when it has not told you what
you asked.

## A diagnostic that touches the resource it diagnoses can break what it checks

`doctor.py` verifies a host before a run. On hpcfund it made the run impossible.

The probe is a real GEMM, because `torch.cuda.is_available()` returning True is not
evidence the GPU computes -- which that host proves, since its GEMM dies with
`hipErrorFileNotFound`. But a failed HIP context does not stay inside the process that
created it. Run `doctor` as its own process, then collect, and Chrono's `Initialize()`
dies with `std::bad_alloc: hipErrorNoDevice` -- no device at all -- on a node where
`rocminfo` enumerates gfx90a, a plain `hipMalloc` succeeds, and a 6.6 M particle bed with
eight FSI bodies builds without complaint.

The fix is to not probe what the action will not use. Collection runs the policy on CPU
and never opens a GEMM; the registry already records that this host cannot train. The
probe still runs, and still fails hard, for `train` and `finetune`, which is where it was
always earning its keep.

### Ten jobs, and what actually found it

Four genuine defects were repaired on the way to this one, each of which really was
broken: the registry matched login nodes but not compute nodes, the registered Chrono
build had no `pychrono.parsers` and could not open a URDF, a failed GEMM was fatal for
actions that do not need a GPU, and the job script loaded no modules. Fixing each changed
nothing, because none was the blocker.

What worked was not the next hypothesis. It was **removing a variable**: one run with
`--skip-doctor` completed in 4m34s and named the culprit immediately. That experiment was
available from job three and would have cost one submission.

Three habits to keep from it:

- **Read the whole error, not the tail.** The doubled GEMM warning -- doctor running
  twice, once per process -- was in the full stderr for eight jobs. Tailing works until
  the informative part is not at the end.
- **A disconfirmation refutes the version you tested, not the family.** A single-process
  probe-then-Chrono test passed, and that was taken as clearing the probe entirely. The
  failing configuration was two processes, which is a different claim.
- **One-factor-at-a-time is blind to interactions.** A bare bed of 3.5 M particles passed,
  and the full `build_crm` at 4 x 4 passed, so both factors were cleared -- and the cell
  where they meet was never run. It turned out to be fine too, but the reasoning was not.

### The cost asymmetry that made this cheap

Eleven probes on `devel` and `mi2101x`, both at 0.1x charge, came to roughly 0.1 node-hours
against a 1500-hour shared allocation. The compute was free; the wall clock was not. The
expensive resource in a debugging session is the number of round trips, and every guess
that gets tested by rerunning the whole pipeline spends one to learn a single bit.

## The policy was never rolled; something that resembled it was

Both v1 fine-tunes made the policy worse in Chrono: analytic fell in 5 of 16 episodes and
left the corpus region (Gate 4, 29.4% outside); PPO stayed upright and in-corpus and still
tracked worse in 16 of 16 paired episodes (mae_vx +58%, t = +9.7). In the model, both had
improved. The cause was not the methods. It was that the thing being optimised inside the
NN-ROM was not the policy the robot runs, for three independent reasons:

1. **The corpus recorded the state after the action, not before it.** `collect.py`
   captured each row after `DoStepDynamics`. On a control row that one physics step is the
   PD kick from the target just set, so the recorded joint velocities were 0.91 of their
   own spread away from what the policy had observed, and the policy's output from a
   recorded row differed from its real output by 36% of its spread on rigid ground and
   ~60% on CRM. For training the NN-ROM this is harmless -- it is a consistent
   row-to-row map. For rolling a policy on those rows it is fatal: the observation
   already contains the consequence of the action it is supposed to choose.
2. **The policy ran at the record rate.** Rows are 100 Hz, the policy acts at 50 Hz, and
   the fine-tune called the policy on every model step. Inside the model it ran twice as
   fast as on the robot, fed the model an action stream that changed every row where every
   recorded one is held for two, and "15 steps = 0.30 s" was 0.15 s.
3. **The action history was shifted one row.** The rollout appended the policy's action to
   a window already ending in the recorded one, so the model saw state row j paired with
   action row j+1 -- a pairing it was never trained on -- from the first step.

**Why nothing caught it.** Every check that existed looked at the model (horizon profile,
floor, lottery) or at the outcome (Gate 4, the paired Chrono test). None asked whether the
closed loop inside the model was the closed loop on the robot. The paired test did catch
the consequence, which is what it is for; it cannot say why.

**The test that finds all three is one line of arithmetic:** rebuild the observation from
a recorded row, run the base policy on it, and compare with the output the policy actually
recorded at that row. A faithful loop agrees to float rounding (8.8e-07 against a spread of
1.55 on the fixed corpus). The broken one missed by 60%.
→ Before optimising a policy inside any surrogate, prove the surrogate loop reproduces the
policy's recorded behaviour at its start states. `finetune.py` now refuses to run unless
it does (`check_start_reproduction`), `obs_truth.py` checks the observation block by block
against a live Chrono run, and corpora are stamped `row_capture=pre_step` so the old ones
are refused rather than silently reused.
