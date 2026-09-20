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
