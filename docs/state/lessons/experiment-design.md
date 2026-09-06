# Lessons: experiment design

Failures where the *experiment* could not have answered its question, whatever
the code did. These are the expensive ones, because the compute is spent before
anyone notices, and the output looks like a result.

## Ask what it would look like if the thing did nothing

**Cost:** four instances in one day · **Found:** 2026-09-03 · **Applies to:** every gate, test, and A/B

The single question that caught all four:

> **What would this look like if the mechanism under test did nothing at all?**

If the answer is "the same, or close enough that I could not tell," the
experiment is not an experiment yet. It is cheap to ask, it takes one line of
arithmetic or one careful read, and on 2026-09-03 it paid four times:

| failure | what "did nothing" looked like |
|---|---|
| G0a gate passing a fallen robot | base-z criterion satisfied by a robot on its back |
| `AttachFsiSphSystem` empty body | returns `-1`, compiles, links, no error |
| CUDA silently disabled | configure exits 0, two warnings, wrong modules built |
| enforcement A/B on collisions | **P(0/100) = 0.366 under the null** |

All four report success while doing nothing. None is detectable from exit
status, and none was caught by testing harder — only by asking what the null
would produce.

**The corollary: establish what counts as a distinguishable outcome BEFORE
spending the compute.** Fix pass criteria before running. Compute detectability
before launching. A criterion chosen after seeing output is not a criterion.

## Choose the metric before worrying about the sample size

**Cost:** ~100 episodes scored on a readout that could not see the effect · **Found:** 2026-09-03 · **Applies to:** any rare-event evaluation

**Expected:** enforcement reduces collisions, so count collisions.
**Happened:** at a 1% baseline, a perfect intervention still yields a
null-looking 0/100 with probability 0.366, and Fisher on 1/100 vs 0/100 gives
p = 1.000. The experiment could not distinguish perfect from useless.

The fix was not more runs. It was **the same 100 episodes read differently**.
Measured on `kyle-N7-B650E` against the observed unenforced distribution
(n=100, mean 3.067 m, sd 1.146 m), 2000 sims per point, α=0.05 two-sided:

| shift | in SD | power |
|---|---|---|
| 0.00 m | 0.00 | **0.05** ← calibration check, lands where it must |
| 0.20 m | 0.17 | 0.37 |
| **0.40 m** | 0.35 | **0.80** |
| 0.75 m | 0.65 | **1.00** |

The expected effect is ~0.80 m. **So on the continuous readout the effect is
essentially certain to be detected; on the binary one it was essentially certain
to be missed.** Same episodes, same compute, same physics, roughly an order of
magnitude difference in what the experiment can see.

**A rare binary event is usually the least informative function of a continuous
measurement you already have.** Collided/didn't discards every episode that came
close, and "came close" is most of the signal.

**Validate the machinery against a known answer first.** The same code scored
against its own data returned `U=5000, z=0.000, p=1, Cliff's δ=+0.000` — the
degenerate answer it must give, which also reveals the bootstrap's resolution
(median difference CI ±0.53 m) before any real comparison is attempted. And the
0.00-shift row returning exactly 0.05 says the test is calibrated rather than
merely optimistic.

**Pre-register the expected effect, with its provenance.** The 0.80 m
expectation comes from 12 seeds of *planning geometry*, not physics, and the
driven trajectory is not the planned one. It is an order-of-magnitude
expectation, not a prediction. **The power curve does not depend on it** — which
is the property that makes the pre-registration honest rather than decorative.

## Measure the noise floor before you compare anything to it

**Cost:** none, applied before the cross-API comparison · **Found:** 2026-09-03 · **Applies to:** every A-vs-B on a stochastic simulator

The cross-API check asks whether a Go2 run under the source build matches one
under conda 10.0.0. **The obvious version of that question has no answer.** GPU
SPH is not bit-reproducible — atomic accumulation order varies between runs on
identical hardware — and the system has intermittent contact, so trajectories
diverge from arbitrarily small differences. Two runs of the *same* build differ.

So "do the APIs agree" is unanswerable until we know **how much a build
disagrees with itself.**

**Procedure: run the conda arm N times first, unchanged.** The spread of those
runs is the noise floor. Only then run the source arm, and ask whether it falls
*inside* that spread. A source-vs-conda difference smaller than conda-vs-conda
is agreement; one substantially larger is a real API difference.

Without the noise floor there is no criterion at all, and whatever difference
appears gets argued about after the fact — which is the failure this file
exists to prevent.

**This is the same move as scoring a dataset against itself** to check the
statistics machinery returns the degenerate answer, and the same move as
asking what the null would produce. Establish what "no effect" looks like
*with your own instrument, on your own hardware*, before interpreting an effect.

**Corollary: separate the variables.** Compare physics **headless** first, so an
API difference cannot be confounded with a rendering difference. Only then turn
the camera on. Two changes landing at once produce a difference nobody can
attribute.

## Check that A and B ran on the same population before attributing anything

**Cost:** would have shipped a causal claim the data does not support · **Found:** 2026-09-03 · **Applies to:** any A/B where the treatment can reject a sample

The enforcement A/B returned p = 3.65e-06, Cliff's δ +0.379, a +1.216 m median
shift with a CI excluding zero, and a fall from 45% to 9% below the bound. Every
statistic was correct and the pre-registered power analysis held.

**On the 60 episodes where both arms ran the identical layout, the paired
difference was bit-identical: 0 of 60 improved.** The treatment changed nothing
it was applied to. The whole effect came from the treatment *rejecting* samples,
which caused the sampler to draw replacements, so the two arms were scored on
different and unequally difficult populations.

**A treatment that can reject a sample silently redefines the population it is
evaluated on.** The between-arm comparison then measures selection, not effect,
and it does so while producing entirely respectable statistics.

**Always run the paired comparison on the shared subset**, and report how many
samples the arms actually share. Here 40 of 100 differed, which was itself the
signal. If the treatment can refuse, assume it is re-selecting until shown
otherwise.

## The check you add to catch silent failures can itself be silent

**Cost:** none, caught by a third arm · **Found:** 2026-09-03 · **Applies to:** every in-band success signal

All day the handle from `AttachFsiSphSystem` was our defence against the silent
no-op: `-1` means the OptiX branch was compiled out, `>= 0` means a real attach.
It was added precisely because a method that compiles, links and runs is not
evidence that it does anything.

`kyle-N7-B650E` then ran a third arm — attach succeeding, options left
default-constructed — in **C++, with the ordering already correct**:

| arm | handle | dark | bright |
|---|---|---|---|
| attached, options set | 0 | **53.8%** | 38.8% |
| attached, options default | **0** | **0.1%** | 94.1% |
| not attached | −1 | 0.1% | 93.9% |

**Handle `0` and a blank scene.** Two of the three arms are indistinguishable in
pixels while differing in handle, and two are indistinguishable in handle while
differing entirely in pixels.

So the handle separates *attached* from *not attached* and says nothing about
*rendering*. **We replaced a check that could not fail with a check that also
cannot fail in the case we had moved on to caring about.** Same shape as the
vacuous gate and the CUDA `exit 0`, one level further in — and this time inside
the instrument built to catch that shape.

**The rule: an in-band success signal reports that a call was made, not that it
did the thing you wanted.** Whenever you add one, ask the null question *about
the signal itself* — what would this return if the operation succeeded and
accomplished nothing? Here the answer was `0`, and only pixels tell the three
arms apart.

This also independently corroborates `kyle-sbel`'s run B, where populated-but-wrong
options returned handle 0 and drew nothing. Two languages, two machines, two
routes to the same conclusion.

## A watched process disappearing means it ENDED, not that it SUCCEEDED

**Cost:** ten minutes, and only because it was checked · **Found:** 2026-09-03 · **Applies to:** every background job we wait on

The rule written earlier — *poll the process that does the work, not the
launcher* — was followed correctly and **was not enough.** The lesson recurred
within ten minutes of being written down, in a form its own text did not cover.

A waiter reported `LATERAL COLLECTION FINISHED`. It had polled the collector
script's own PID, exactly as prescribed. Checking the *job* rather than the
signal:

- the output file was **empty** — not one progress line
- **one** episode directory existed, of fourteen
- the script had never printed its completion sentinel
- and episode 0's python child was **still running, orphaned**

```
while kill -0 <pid>; do sleep 120; done
```

**cannot distinguish "completed" from "killed", "crashed", or "OOM-ed".** All four
produce the same silence. The waiter reported the literal truth — that PID was
gone — and the inference drawn from it was wrong.

**So the rule has a second half: wait on a SUCCESS SENTINEL, not on an absence.**
The job prints something on successful completion; the waiter checks for that
string. Absence of a process is not evidence of success, and a monitor built on
absence reports success for every failure mode there is.

### And the launch method decided it

The 5-hour collection survived; this one did not. Same waiter design. The
difference:

| | launch | outcome |
|---|---|---|
| CRM collection | the tool's own tracked background mechanism | ran 5 h to completion |
| lateral re-collection | `nohup … &` inside a **foreground** call | killed when that call was torn down |

**`nohup` only ignores SIGHUP.** It does not survive the process group being
cleaned up. The python child was reparented and kept running, which is why one
episode was in flight with no parent — the tell that made the diagnosis
unambiguous.

### The pattern this completes

Every silent failure in this project has the same shape: **the check reports
success when the thing it checks did not happen.**

| check | what it reported | what was true |
|---|---|---|
| G0a gate | pass | robot on its back |
| `AttachFsiSphSystem` | handle ≥ 0 | nothing rendered |
| CMake | exit 0 | three modules silently disabled |
| first waiter | finished | the *launcher* finished |
| second waiter | finished | the job was **killed** |

The last two are the same bug at different depths, ten minutes apart, and the
second was invisible to the rule written for the first.
### A cleanup command can match itself

Twice in the same hour, a `pkill -f <pattern>` used to clear stale collectors
matched **the shell that was running the `pkill`**, because the pattern appeared
in that shell's own command line. Both times the command died with exit 144 and
the job it was supposed to be tidying up around was left in an unclear state.

The second time was ten minutes after writing the section above, which is the
point: the failure family is *a signal or an action aimed at the wrong process*,
and knowing the waiter version of it did not stop the killer version.

**Kill by resolved PID.** If a pattern must be used, match on something that
cannot appear in the killer — or check `pgrep -f` first and read what it would
have hit.

## A shared branch is a mutable dependency of every running job

**Cost:** one episode, caught within five minutes · **Found:** 2026-09-04 · **Applies to:** any long run on a box that touches git

A collection re-reads its own source on every episode. So **any git operation on
that working tree changes the running job**, and a rebase is enough — nobody has
to edit anything.

What happened: a correct fix (the validation-ratio default, from another machine)
was pushed to the shared branch. The collecting box rebased onto origin to publish
an unrelated tool. Git checked out `collect_go2_smoke.py`. **Eighteen seconds
later the next episode started on different code.**

**Nobody did anything wrong at the moment it happened.** The fix was right, the
push was right, the rebase was right, and the run had been going for hours before
either. The failure is in the *composition*, which is why neither party saw it
coming: each action was locally correct and reviewed as such.

This is the third instance in one session of a file changing under a live run —
after an actuation edit mid-batch and a mid-run seed change — and **the first
where the mutation came from someone else's good work arriving through a routine
git command.**

### Why it was catchable

The per-episode provenance sidecar had been running for five minutes when it
found it, and it found it by **digest, not by commit**. Commits moved constantly
and mean nothing on their own; the digest of the seven files an episode actually
reads is flat until behaviour changes. One episode out of 1,045 carried a
different digest.

**Record a content digest of the code an episode reads, per episode.** A commit
hash tells you what HEAD was; a digest tells you whether it mattered.

### Two ways to prevent it, in order of preference

1. **Run from a snapshot.** Copy the collection code to a run-scoped directory at
   launch and execute from there. The live tree then cannot reach a running job at
   all, and the box stays free to pull, rebase and push.
2. **Freeze git during a run.** Cheap, immediate, and the mitigation actually
   adopted here — but it makes every long run block the branch, which does not
   scale to two machines collecting for six hours.

### When a mid-run change does not require a restart

Only when the changed behaviour is confined to a field that is being overwritten
anyway. Here the diff reached exactly one thing — `assign_split`'s ratio — and the
repair pass recomputes every episode's split from its new id regardless. **After
the pass, an episode collected under either value is indistinguishable.**

That argument is legitimate and it is narrow. It holds *because* the differing
field is the one already scheduled for rewrite. Had the diff touched anything
else, the correct answer was to restart.

## Provenance defects are found by CONSUMERS, never by readers

**Cost:** five defects, all caught before training · **Found:** 2026-09-03/04 · **Applies to:** any dataset with metadata

Five metadata defects surfaced in one session. **Not one was found by reading the
collector.** Three came from writing the code that consumes the data, and two from
a second machine trying to reproduce a result.

| field | what it said | what was true |
|---|---|---|
| `policy` | a checkpoint that never ran | the imported one did |
| `checkpoint_path` | a path under `/tmp` | moved to durable storage |
| `git_commit` | commits that no longer resolve | rebased and reworded away |
| `patch_y` | correct | **invisible from where the decision was made** |
| `scenario_family` | `"constant_command"` for every episode | eight distinct families |

**Four were true when written and invalidated by change. One was never true.**

That distinction matters because the mitigations differ. A value invalidated by
change is caught by recording provenance at write time and re-verifying it later.
**A value that was false at its source survives any amount of faithful handling** —
a careful consolidation propagates it perfectly, a hash check confirms it was
copied exactly, and a rule like "keep everything addressable" preserves it intact.

### The one that reached checkpoint selection

`scenario_family` looked like a label. It is the key
`_select_rollout_episodes` buckets on (`trainer.py:797`) before round-robining
across families to choose the twelve episodes the deployed checkpoint is selected
on. **With one bucket the round-robin degenerates to "take the first twelve in
list order"** — and the output is still a rollout error over twelve episodes,
which is exactly what a correct one looks like.

### Three consumers, three failure modes, one field

`scenario_family` turned out to have three consumers, not one, and each degrades
differently when the field collapses to a single value:

| consumer | mechanism | degrades to |
|---|---|---|
| `trainer.py:796` | round-robin, `pop(0)` | the first twelve in list order |
| `build_combined_*:135` | shuffle within bucket | proportional to family frequency |
| `references.py:111` | seeded shuffle, `pop()` | a stable, reproducible wrong bank |

**The third is the worst, and it is worst because it is reproducible.** A seeded
shuffle over a single bucket returns the same plausible-looking, unstratified
reference set every time it runs. Rerunning it confirms it. The other two at least
have the decency to look arbitrary. It also accepts a `requested_families` list,
so an explicit request can be honoured exactly and mean nothing, drawn from a pool
where every episode claims the same family.

This is the argument for repairing the data rather than patching the consumer: a
consumer patch would have had to be correct in three modules, and the third is the
one nobody had read.

### A stop-condition must be checked where the property is CONSUMED

The coverage requirement had been set one layer too high: *"report the val split
per family before training."* A val split can cover every family perfectly and the
trainer can still select twelve rollout episodes from one of them.

**Check the property at the point of use, not the point of production.** Producing
coverage and consuming it are different guarantees, and only the second one is the
one anybody cares about.

### The practical rule

**Write the consumer early, even before the data exists.** Every defect here was
invisible to inspection and obvious to use. A synthetic dataset run through the
real pipeline would have surfaced most of them before a single real episode was
collected.

`scripts/collection/validate_go2_dataset.py` is that consumer, and it runs between
repair and preprocess.

### A sixth defect, in a worse category than the five

`terminated_near_boundary` was never written to the Go2 index at all. The RL
reference builder filters on it (`build_combined_flat_crm_rl_references.py:133`),
`.get()` returned `None`, `None` is falsy, and **a default-on exclusion was off
with nothing reporting that it was off.** The HMMWV collector does write the field;
ours diverged from the schema it was meant to match, in a key nobody reads until
the reference build.

The obvious rule — *audit every `.get()` with no default* — is wrong, and the
counter-example is four lines away in the same file. `preprocess.py:233` reads
`frames_path` with exactly that pattern and is harmless, because it checks the
absence and raises. The real condition is narrower:

> **A missing key is dangerous only where absence and a legitimate value collapse
> to the same branch.**

`None` is not a path, so `frames_path` cannot collapse. `None` and `False` are both
"keep this episode", so `terminated_near_boundary` collapses perfectly. **A wrong
value is at least present to be noticed; an absent one is not.**

The corollary is worth the discipline it costs: a field missing from a schema *for
no reason* is how the next one hides. `warmup_s` had no consumer and was added
anyway, on that argument.

### Watch the gate fail before trusting it

Run against synthetic pre- and post-repair datasets, the gate found three defects
and **two were in the gate itself**: it passed on zero selected episodes ("0 of 0
families covered" — true, useless), and it scored coverage against the families
that survived into the validation split rather than the families that exist,
reporting full coverage of an already-impoverished pool. That second one is the
produced-versus-consumed confusion again, one level below where it was first
found, *inside the check written to catch it*.

A gate must also refuse to emit PASS on input an earlier gate already failed —
otherwise a later check reads clean off data it never saw.

### Running a check is not evidence that the check ran

The strongest version of this, and the one that cost the most: a repair script's
dry run reported `csv_changed: []`, and that was quoted to me three times as
evidence the rewrite preserved the trajectory data. **The comparison was inside the
`if apply:` branch.** On a dry run it never executed, so the empty list meant "not
computed" and read as "nothing changed" — on a dataset it would have corrupted.

The five earlier instances of this shape (a G0a gate passing a robot on its back,
CMake's `exit 0`, the `AttachFsiSphSystem` handle, G8 on zero episodes, a waiter on
a vanished pid) were all caught by someone *reading* the checker. This one was
**run, three times, and its output quoted as a result.** Reading the checker is what
catches it; running it is what makes you stop reading it.

**The fix is a negative control inside the check, executed every run:**

```
data change detected:        True    (perturb a non-id column -> hash differs)
id change correctly ignored: True    (perturb the id column   -> hash identical)
```

Without the second line the comparison could pass by hashing nothing that matters.
A check that demonstrates it *can* fail, on every invocation, is a different object
from a check that merely has not failed yet.

## Ask what a command WRITES, not what you intended to edit

**Cost:** zero episodes, by a 22-second margin · **Found:** 2026-09-04

The rule was already written down and already being followed: *do not mutate
anything a running collection reads, and git HEAD counts, not just source files.*
The box that wrote that rule then ran `git pull --rebase` mid-collection to fetch a
script, and a rebase checks out files. It replaced the collector — which every
episode re-reads at spawn — with origin's version, which lacked six provenance keys
the run was recording.

**Zero episodes were affected, and that is luck, not margin.** The window fell
between two episodes that are ~2 minutes apart, and the in-flight one had already
imported the old file. Thirty seconds earlier and the batch would have split
silently.

**The margin was a property of the workload, not of the handling.** This collector
re-reads the file once per *episode*, so the exposed window is 22 seconds inside a
2-minute gap. A job that re-read per row, or whose episodes were seconds apart,
would have been caught by the same 22 seconds. Do not read "it was fine" as a
tolerance; it does not transfer to a faster job.

**The deliberate edits were never the risk.** That same box had checked an hour
earlier that `trainer.py` was safe to edit. The discipline was working exactly
where attention was pointed, and the hazard arrived through a command whose
file-writing is a *side effect* of its real purpose:

> It is not enough to ask "am I editing a file the run reads." Ask **"does this
> command write to that tree at all"** — and `pull`, `rebase`, `checkout`, `stash`
> and `merge` all do.

**A rule scoped to intent does not cover a danger scoped to effect.**

### The safe read, and a briefing failure worth more than the technique

`git fetch` updates refs only; `git show origin/<branch>:<path>` then reads a blob
without touching the worktree. That is the whole mitigation.

One box worked it out and reported it. **I passed it to that box and not to the
other**, then told the second box a file had been pushed, with no method attached.
Two machines running the same job under the same hazard, one holding the
mitigation. That is worse than nobody knowing it, because the fleet looked uniform
from where the second box sat. **When a mitigation is discovered by one node,
propagating it is the same task as recording it** — to every node that could lose
data, at the time it is learned, rather than to whoever is next in the conversation.

The affected box made the fair objection: the technique would have prevented that
instance, but the rule it already held would have prevented the whole class, so the
briefing gap is not the main lesson. What it does expose is structural. **In a fleet
coordinated through one hub, the hub holds the only complete view, and no leaf can
audit what another leaf was told.** From either box the fleet looked uniform. That
asymmetry is a standing property of the topology, not a one-off.

### Restoring

Restore by extracting the known-good file and **checking its hash against what the
run started with** before copying it in — never by editing it back, which can
produce a third version matching neither, with no way to tell.

## At small data scale, DELETION and PERMUTATION are different tests

**Found:** 2026-09-04, before any ablation ran · **Applies to:** every feature
ablation in this project, including the paper's own rule

The framework's rule for the reduced state is deletion: *a channel earns its place
only if removing it degrades rollout fidelity or closed-loop performance.* That
test is sound at the reference data scale. **It is confounded at ours.**

Deleting a channel removes information *and* width. Adding one supplies
information *and* width. On the steep part of the data curve, extra width absorbs
variance on its own, so a channel that carries no physics can still earn its place
by the deletion test, and a channel that carries physics can look unnecessary if
the model was capacity-limited rather than information-limited.

**The fix is to permute rather than delete.** Replace the channel's values with a
version that preserves its marginal distribution and destroys its correspondence
with everything else — permute within episode — and keep the dimensionality
identical. Then:

| comparison | isolates |
|---|---|
| present vs **deleted** | information **+ capacity**, tangled |
| present vs **permuted** | **information only** |

The three-point reading is what makes it interpretable: permuted lands *between*
the other two if capacity alone helps, *at* the full model if capacity was all
there ever was, and *below* the full model if the signal is real.

This applies to `quadruped_contact` (15-D) against `quadruped_full` (23-D) exactly
as it applies to the contact-mode context input — and it is why the pre-registered
warning about adding channels to fix a data shortfall is not paranoia. **A channel
added for capacity reasons passes the deletion test honestly.** Only permutation
separates it.

Permute with a recorded seed. A control that cannot be reproduced is a control
nobody can check.

### Permuting a STATE channel is harder than permuting a CONTEXT input

A context input is **exogenous**: supplied, never predicted, never fed back. Permute
it and nothing else about the model changes. That is dorm-pc's contact-mode cell,
and it is clean.

A state channel is **endogenous** — it is predicted, it is in the loss, and in
autoregressive rollout the model's own prediction for it becomes the next step's
input. Naive permutation breaks the control in two separate ways:

1. **The loss becomes incomparable.** `channel_weights` are mean-normalized to 1
   across channels (`trainer.py:524`), so a permuted channel is unpredictable, its
   per-channel term is large, and it dominates the aggregate. One-step loss then
   differs between cells for a reason that has nothing to do with the question.
2. **Train and rollout see different input distributions, in the permuted cell
   only.** During training that channel's inputs are permuted and jumpy; during
   rollout the model feeds back its own best prediction, which is smooth. The real
   cell has no such mismatch, so the permuted cell can lose for an artifact.

**The mechanism already exists and is better than permutation.**
`transformer_cfg.blind_state_fields` (`model.py:33`, applied at `model.py:123`,
logged at `trainer.py:427`, precedent in `configs/ablations/arm_transformer_8d_qdonly_v1.json`)
drops a channel from the token *before the input projection* while leaving **the
state and target layout untouched**. The network never sees the channel; it is
still predicted, still autoregressed, still in the loss with the same weights.

| control | input | output head | loss channel set |
|---|---|---|---|
| **blind** | narrows by 1 (−256 params) | **identical** | **identical** |
| permuted-exogenous | fixed | narrows by 1 (−257) | changes |

Blind is cleaner on the axis that matters most — an identical loss channel set
means the two cells' aggregate losses are built from the same terms — and it needs
**no new code**. Prefer it. Permutation remains the right control for an
*exogenous* input like the contact-mode code, which has no target to leave
untouched.

The residual capacity difference is ~256 of ~4.72 M backbone parameters
(6 layers × (4·256² + 2·256·1024)), i.e. **0.005%** — about four orders of
magnitude smaller than the capacity confound the control exists to remove. That
ratio is what makes the control sound rather than merely better.

**Compare on `rollout_sel`, not on aggregate one-step loss** — this caveat survives
either construction, because a blinded channel must still be predicted without
being seen, so its own one-step term is worse for a reason unrelated to the
question. The rollout metric is
pose-derived — `_integrate_pose` uses only `vel_body_x_mps`, `vel_body_y_mps` and
`yaw_rate_radps` — so a permuted non-pose channel affects it *only* through its
influence on predicting those three. That is exactly the causal path under test.
Report one-step loss restricted to the channels present in both cells, never the
aggregate.

## Normalisation hides the units it divided by

**Four instances in one night**, three of them mine. Every one was a ratio carried
across a boundary its denominator did not survive.

| ratio | denominator that differed | wrong by |
|---|---|---|
| "20% of their data" | episodes vs **transitions** (their flat episodes are 5,000 rows, ours 1,475) | flat is 0.35%, not 1.2% |
| "flat worse than CRM means a bug" | our flat is 0.35% of reference, our CRM 15.6% | criterion **inverted** |
| "expect 6–12% `errdist`" | path length: theirs 30–53 m, ours **1.0–1.2 m** | 25–52× |
| "1.9 body lengths" | HMMWV numerator ÷ **Go2** denominator | 1.9 should be 0.28 |

**A normalised quantity looks system-independent and is not.** That is the whole
trap: the ratio was doing exactly what it was designed to do *inside its scope*,
and dividing out the units is what made it look safe to carry outside.

### The defence is dimensional and takes one line

**Does each numerator divide by its own system's denominator?** Run it before
quoting any cross-system ratio. The fourth instance above was produced *in a
message whose thesis was this very point* — knowing the rule is not the same as
running the check.

### Plausibility is not confirmation

`1.30 / 0.7 = 1.86` rounded to "1.9 body lengths", which **looked like a plausible
vehicle-scale error**, and plausibility was read as confirmation. A wrong answer
inside the expected range is the one that survives review, because the check that
would catch it feels unnecessary once the number looks right.

### Quote the scope with the value

`trainer.py:770` calls `errdist` "the honest cross-domain comparison" — correct,
**across domains within one system**, which is what it was written for. We read it
as cross-*system*, and the word "honest" encouraged it. Nothing was stale, silent,
or wrong; a true statement was applied outside its conditions.

So report the scope *in* the number: *"errdist 0.14, normalised by ground-truth
path length pooled over the twelve selected episodes, mean_dist_m 1.24"* is a
sentence someone can carry to another system safely. *"errdist 0.14"* is not.

## A record can disagree with its check exactly where someone will look

**Seventh finding, 2026-09-04.** The Go2 collector tests the bed boundary on one
reference frame and logs the trajectory from another:

```
collect_go2_smoke.py:416   bp = base.GetPos()                  -> CENTRE OF GRAVITY
dataset.py:265             base.GetFrameRefToAbs().GetPos()    -> REFERENCE FRAME
```

They differ by a **body-frame longitudinal offset of 20.7 mm** — the COG sits that
far forward of the reference-frame origin along the body x-axis. Nothing in either
file says which frame the other uses.

**Found from data, not from reading.** Two boundary episodes stopped at logged
x = 0.1796 and 0.1808 against a bed edge of exactly 0.200. Since the check breaks
*before* recording, no row should exist below 0.200 — yet about twelve did. The
implied offsets, +20.4 mm and +19.2 mm, agree to 1.2 mm across approach speeds
differing by 70% and `t_switch` by 3 s. **A speed-dependent artefact cannot produce
a 1 mm spread; a fixed frame offset produces exactly that.**

**The first description was wrong, and nine episodes corrected it.** From two
x-edge episodes it read as "+20 mm in world x." That cannot be right, because the
offset also appears on the **y** edges. If it is one body-frame offset, the miss at
any edge is `|d · cos(angle between the body x-axis and that edge normal)|`, and
dividing it out should recover the same constant:

| | raw miss | implied offset |
|---|---|---|
| spread over 9 episodes, 4 families, 3 edges, 236° of heading | **8.5 mm** | **3.3 mm** |

Projection removes 62% of the scatter; mean 20.7 mm. And the falsifying case is
decisive: a world-x offset predicts **zero** miss at a y edge with the body pointing
along y, yet the four episodes at |yaw| 85–100° on a y edge show the **largest**
misses (19.7–22.4 mm). The residual 3.3 mm is about what a 10 ms sample interval at
0.2 m/s gives.

So the correct statement is: *the COG sits ~20.7 mm forward of REF along the body
x-axis; the boundary is tested on the COG and logged from REF; the discrepancy a
reader sees at any edge is that offset projected onto the edge normal.*

Not a data defect: terminations are real, trajectories self-consistent, effective
margin 0.78 m rather than 0.80, nothing to recollect.

### The shape worth remembering

**The 20 mm is present in every episode and legible in almost none.** Mid-episode
there is nothing to compare a position against, so the offset is invisible. It
becomes visible only at the boundary — *which is the one place a reader goes
looking for a geometric explanation.*

So the failure mode is not "a small constant error." It is **the record disagreeing
with the check precisely where someone will check it.** A future reader
reconstructing why an episode ended finds the robot 20 mm *inside* a line it
supposedly crossed, and has no way to tell which of the two numbers to distrust.

Same class as this project's earlier AuxRef visual-frame bug. The one-line rule:
**check the frame you log.**

### Coda: n=2 gives you a mechanism and lies about which parts are load-bearing

Two people made the same error inside an hour, on different findings:

- The frame offset read as **"+20 mm in world x"** from two x-edge episodes. Nine
  episodes across three edges showed it was body-frame, and the world-frame reading
  was *falsified* by the very cases the first sample happened to exclude.
- The `vel_step` spawn failure read as a **four-way conjunction** — forward spawn,
  `vx0` in the dead zone, strong negative `vx1`, early switch — from one episode
  where all four held. A second episode tracking forward at 0.155 m/s (2.5× the
  supposed threshold) exited anyway. **The spawn is sufficient; the dead zone only
  aggravates.** Exposure is ~half the family, not the 1-in-10 the conjunction
  implied.

Both times a small sample yielded a *correct* mechanism and a *wrong* account of
its necessary conditions — and in both, the sample's incidental features got
promoted to requirements. Inspecting few units beats a p-value (see above), but the
next question is always: **which of these conditions did I observe because it
matters, and which because it was there?**

## Correct arithmetic on an unchecked premise about what the system does

**Twice in one night, both mine.** Distinct from the denominator error: the numbers
were right and the *model of the system's behaviour* was not.

| claim | the arithmetic | the premise that was false |
|---|---|---|
| "the rollout has a valid start window" | sound | assumed a **random start**; `trainer.py:841` takes `states[:sequence_length]` |
| "spawn heading explains 57–71% of lateral drift" | sound — `E\|sin θ\|` over ±10° is 0.087 against a measured 0.123 | assumed the robot **walks along its initial heading**; the policy has a yaw-tracking reward and *corrects* the offset rather than integrating it |

Measured: regressing drift-per-metre on `|sin(heading)|` over 38 episodes gives
slope 0.177, intercept 0.090, **R² = 0.018** — heading explains **14%**, not 57–71%.
Drift is a plant property after all, at 0.090 m/m once the heading term is removed.

**The proposed remedy would have failed.** Narrowing the heading draw from ±10° to
±3° cuts drift by ~10%, not two thirds — and someone could have spent a recollection
on it.

**Both times the premise concerned what the system does *between* the numbers.**
`path × sin(θ)` is open-loop kinematics; a closed-loop controller with a heading
objective makes it wrong, and nothing in the arithmetic signals that. The check is
not dimensional — it is: *which component decides this, and have I read it?*

## A p-value can be a way of not looking, or a way of looking with the wrong instrument

Two failures of the same statistic in one night, in opposite directions.

**Too few units — a way of not looking.** `vel_step` vs `constant` was 3/38 against
0/38, p = 0.240. There were *three* anomalous episodes, fully instrumented. Reading
their parameters found a spawn bug; reading their trajectories turned it into an
arithmetic identity. **When the anomalous units are few and inspectable, inspect
them.**

**Wrong instrument — a way of looking badly.** Asking whether the drift bias was
systematic or per-episode, I proposed a **sign test**: ~35/3 means systematic,
~19/19 means per-episode. Over 237 episodes it came back **132/105, p = 0.091** —
neither outcome, and it would have been read as "per-episode, nothing to record."

The signed *mean* on the same data gives **+0.0292 ± 0.0118 m/m, 2.47σ.** A binomial
on signs **discards magnitude**, so it cannot see a small consistent lean sitting
inside large per-episode scatter. It is maximally robust and minimally sensitive:
the right tool when the alternative is *"all one way"*, the wrong one when the
alternative is *"slightly one way"* — and I had specified a binary rule for what
was a continuum.

The answer was **both, in a ratio**: 16% systematic, 84% per-episode.

### And scope the sample to the question, not to the conversation

I framed that test on the 38 CRM episodes because CRM was what we had been
discussing. The question was about **the controller**, which is identical on both
terrains — so the population was 237 episodes, and 199 of them were being discarded
for no reason but conversational momentum.

### The 16% answers a different question than it was used for

`mean/RMS = 15.9%` is a fair description of *how large the systematic component is
relative to typical drift*. It is **not** how much correcting the lean would remove:

| question | quantity | value |
|---|---|---|
| how big is the systematic part? | mean / RMS | **15.9%** |
| what does removing it buy? | 1 − sd/RMS | **1.3%** |
| how much mean-square does it hold? | mean² / (mean²+sd²) | 2.5% |

Removing a mean from a distribution leaves `sd`, and `sd` is already 98.7% of the
total RMS. **Correcting the controller lean removes ~1.3% of the drift, not 16%.**
The conclusion — not worth correcting — is unchanged and in fact stronger. One more
instance of a ratio computed for one purpose answering a different one.

## Identify a thing by what it IS, not by listing what it is not

The dataset gate collected episode metadata with `rglob("*.json")` minus a
**name-based exclusion list**. That is a closed-world assumption, and the world
grew: the repair pass kept `episodes/<id>.config.json` per episode (because
`patch_y_m` lives only there — 8.0 for `lateral` against 4.0 elsewhere). Those
files parse fine and yield `terrain_label=None`, so they registered as a
seventeenth `(terrain, command_family)` pair against sixteen `scenario_family`
values, and **G3 failed on a correct dataset.**

Fixed by identifying episode metadata **structurally — it carries an
`episode_id`.** That needs no update the next time the layout gains a sidecar
file; the exclusion list would have needed one every time.

**The false FAIL is the dangerous direction, and this is the instance that proves
it.** It pointed at `scenario_family` — the field that had *just* been repaired —
so acting on it would have meant "fixing" a correct repair while trusting the gate.
The only reason it didn't is that the operator counted the pairs independently
before touching anything.

A gate that fails on good data is worse than no gate, because it is believed.


## Part-whole correlation: a statistic whose value is fixed by its own construction

Correlating `floor` against `policy - floor` puts the SAME VARIABLE ON BOTH
SIDES. Its variance drives the correlation, and when the shared term is the
larger one the result is nearly predetermined: if `policy` were constant, then
`policy - floor = c - floor` and Spearman would be exactly -1 with no fact about
the policy in it at all.

Measured on the Go2 level-3 plumbing run: floor spanned 40x, policy 2.5x, and
rho came out -0.857 (exact permutation p = 0.0107 over all 40320 orderings). It
was about to be reported as "the policy corrects large errors and adds small
ones". What it actually said was "the policy's error is more uniform than the
replay's" -- true, worth knowing, and a much weaker claim.

**The fix is to put the shared term on ONE side only.** Regress `policy` on
`floor` and read the slope: 0 means the policy's error is independent of how hard
the reference is, 1 means it inherits the difficulty entirely. Thresholds can then
be set from the endpoints' MEANING rather than fitted to data, which is what the
ratio band lacked.

Two companions the slope needs:

- **Its p, and the power.** At n = 8, |rho| must exceed about 0.74 for p < 0.05.
  A small-n correlation quoted without its p is how a null becomes a finding.
- **The leave-one-out range.** With a 40x range in the predictor, one point
  carried 77.7% of Sxx. The estimate was leveraged (0.0936 to 0.1292) while the
  verdict was robust (all inside the registered band) -- different properties, and
  the report has to distinguish them.

Related family: a normalisation whose denominator varies across the sample
(`errdist` over families with 2.7x path lengths; the level-3 policy/floor ratio
over floors spanning 40x). Both are "the number you divided by is doing the work".


## A commit hash records where HEAD WAS, not what RAN

**Cost:** two datasets, both halves, one night · **Found:** 2026-09-04 · **Applies to:** any provenance stamp derived from a commit

Every collected episode stamps `git_commit`, and a repair pass derives
`collection_code_digest` from it with `git show <commit>:<file>`. That digest is a
claim about **what code produced this episode**. It is actually a claim about what
the commit contains, and the two differ **whenever the working tree is dirty,
restored, or checked out mid-run.**

**The ordinary case, which needs no accident.** *(Reported by `sbel-pc`; their
numbers, not independently checked here.)* The tree carried uncommitted changes
during collection, and the commit that introduced them landed **18 minutes after
the last episode finished**. So all 968 rigid episodes ran code NEWER than their
recorded commit. Proof: every episode carries `command_params`, the per-episode
amplitude draw, and the collector at the recorded commit contains **no occurrence
of that string at all** — a field in the data that the recorded code could not
have written. **Reproducible on that box** and marked unverified here at their
request: `git show 71c790d9:scripts/collection/collect_go2_smoke.py | grep
command_params` returns nothing, while every episode JSON carries the field.

**The exotic case, verified here on this disk.** A mid-run `git pull --rebase`
moved HEAD forward; the collector file was restored from a hash-verified
extraction but **HEAD stayed at the new commit**. So 46 of 152 CRM episodes ran
code OLDER than their recorded commit. Proven three independent ways, each a
contradiction rather than an inference:

| test | the recorded commit's code implies | the data shows |
|---|---|---|
| split | `--validation-ratio` defaults to 0.2, driver never passes it, so 6 val per 19 | all 46 `train`, which only 0.0 produces |
| assets | hardcodes a `DEFAULT_ASSETS` path absent on that box; raises `FileNotFoundError` on the URDF before simulating | all 46 completed |
| `git_tree` | imports `provenance.py`, whose `provenance()` writes a `git_tree` field | none of the 46 carry it |

`git_commit` and `git_tree` stayed **true in both cases** — HEAD really was there.
Only the digest's claim about *execution* was false, and it was false in **opposite
directions on the two halves**, which is why neither half could have found it by
looking at itself.

**Fix, in order of strength:**

1. **Record the running process's own state while it is alive.** A digest computed
   afterwards from a commit can only ever describe the commit.
2. **Freeze the mapping to a file.** Both halves' commits are now **orphaned** —
   reflog-only after rebases. They resolve today and **would not survive a
   `git gc --prune`**. Provenance that is derivable-in-principle can be
   perishable-in-fact; a committed sidecar is what converts one into the other.
3. **Prefer a field the data carries over one the filesystem carries.** The
   sidecar builder derives commits from **file mtime**, which no copy survives.
   Where episodes also recorded `git_commit` at collection time, the two agreed on
   all 152 — which validates the mtime method rather than merely trusting it.

**Say "consistent", not "clean".** An episode set that no test contradicts has
passed a **necessary and not sufficient** check: a tree differing from HEAD in
ways that add no new metadata key passes every test above unnoticed.

**Evidence:** `docs/state/provenance/go2_stratified_s1000000_commits.json`
(`caveat` block, 46 of 152 flagged); `scripts/collection/repair_go2_metadata.py`
`code_digest()`.


## State the SCOPE with the result, or the conclusion inherits one it never had

**Cost:** three false-but-reasoned conclusions in one day · **Found:** 2026-09-04 · **Applies to:** any check whose result gets restated in words

Three instances, three different operators, all the same shape. *(The first two
are reported second-hand; the third was made and traced here.)*

| the test that ran | the sentence it became | why the sentence was false |
|---|---|---|
| read a rename list in the source | "the CRM shim pairs A with B" | never checked against the *installed* module; refused to import on the only build the collection used |
| read code at **one** commit | "the collection cannot have run after `133427b`" | the frozen sidecar showed all 152 episodes ran under commits *containing* it |
| `grep LEG_ORDER src/nedm/quadruped/constants.py` | "no `LEG_ORDER` exists in the codebase" | it is in the **adjacent module**, `quadruped/dataset.py:73`, with a comment documenting the exact trap being rediscovered |

Each produced a **confident, well-reasoned, false** statement, and each then
**justified an action**: refusing to import, a wrong diagnosis, and adding a third
source of truth for a leg ordering that already had a canonical constant.

The common mechanism is that **the test's scope was never written down**, so the
conclusion silently inherited a scope the test never had. "Not in `constants.py`"
is true and nobody would have acted on it. "Not in the codebase" is what got acted
on, and it was never tested.

**Fix: state the scope in the same breath as the result** — "not in *this file*",
"at *this commit*", "in the *rename list*, not the installed module". A narrow true
statement is safe. A narrow test wearing a broad conclusion is not.

Same family as the padded-field `awk` split and the dry run that reported a
comparison it never made: in all of them the *check* was fine and the *restatement*
was wrong.


## When a defect is invariant under every available check, derivation is the only defence

**Cost:** near-miss, caught by review · **Found:** 2026-09-04 · **Applies to:** ordering constants, label maps, any index-to-name binding

Two orderings for the same four legs coexist: `LEG_ORDER` is `fl fr rl rr` and
`constants.FOOT_BODIES` is `FR FL RR RL`. Packing the 4-bit contact mode against
the wrong one **transposes the left/right bits**.

**Nothing downstream can see that.** The marginal mode distribution is
**unchanged under a relabelling**, so every summary statistic is identical. Even
the physics check passes: the trot signature is that the two dominant modes are
the diagonal pairs, and a left/right flip maps `0110` to `1001` — *the two
diagonals swap into each other*. The strongest validation available would have
confirmed a transposed channel.

**Someone tried to find a downstream check and could not.** *(Attempted by
`sbel-pc`; their numbers.)* Two candidate empirical tests for the labelling, over
1 s windows on 40 CRM episodes:

| candidate | result | why it fails |
|---|---|---|
| `corr(roll, left_load - right_load)` | **-0.083** | right sign, nowhere near decisive; gait-frequency load alternation dominates the variance and mean \|roll\| is only 0.034 rad |
| `corr(yaw_rate, left_slip - right_slip)` | **+0.019** | noise, wrong sign — slip is recorded as a **magnitude** (below) |

Both underpowered, and reported as such rather than quoting the one with the
agreeable sign. So "no check can see it" is a measured claim here, not a
rhetorical one.

**And the second one could never have worked, for a reason worth its own note.**
`foot_*_slip_mps` is `math.hypot(vel.x, vel.y)` — `quadruped/dataset.py:338`,
checked here rather than taken on report. It is a **magnitude**, so it discards
not just the left-right sign but **every directional component**. Any directional
diagnostic on this dataset — left-right asymmetry, fore-aft slip, lateral drift —
is dead on arrival, and **nothing in the column name says so**. That is a schema
change to argue for if contact work needs it, not something recoverable from what
is recorded. The logging is ungated on purpose, but ungated is not the same as
signed.

So the defence cannot be validation. It has to be **one source of truth**: derive
the field order from the canonical constant rather than restating it. A hardcoded
copy is correct on the day it is written and is a third ordering to reach for by
mistake thereafter.

**Where the risk actually lives: consumers, not the recorded data.** The collector
never associates by index. `capture_row` maps label to body *name*
(`LEG_TO_FOOT_BODY`), and the one `FOOT_BODIES`-ordered array in the path
(`soil_z` from `soilprobe.sample`) is converted to a **name-keyed dict** before
use. There is no positional association anywhere between the FR/FL/RR/RL body
list and the fl/fr/rl/rr column names, so the CSV columns are correct **by
construction**. The transposition hazard appears when a *downstream* consumer
re-derives an ordering — which is exactly what a preprocess step packing bits
does, and why it must import the constant rather than restate it.

**Generalises to:** any binding where the failure is a *permutation* of correct
values. Permutations preserve marginals, so aggregate checks are blind to them by
construction — this is not a gap in the checks, it is a property of the defect.

**Evidence:** `src/nedm/training/preprocess.py` `CONTACT_FORCE_FIELDS`;
`src/nedm/quadruped/dataset.py:73`.

## Two places a config diff cannot reach, and one rule for where to look

**A literal inside an inherited parent method.** `hmmwv_tracking_env.py`
hardcodes `pose_error[:, 0] / 10.0` in `_compute_observations`. Porting to the
Go2 we verified the training config key-for-key against the anchor and overrode
`default_env_cfg` wherever it differed. **Neither check could reach that
literal**, because it is not in any config. The instance turned out to be
neutralised, but the blind spot is not — a scale constant living in code that a
subclass inherits is invisible to exactly the two audits a careful port performs.

**And the rule that explains which inherited mismatches survive:**

    OBSERVATION path   empirical_normalization divides inherited scale
                       constants straight back out. The /10.0 was a ~600x raw
                       disparity for the Go2 and reached the network at unit
                       variance, weighted 1.84x the median channel.

    REWARD path        NOTHING normalises these. position_sigma_m 2.0 inherited
                       from a 30-53 m vehicle onto a 1.0-1.3 m robot made the
                       reward flat everywhere, and had to be solved by hand from
                       measured error percentiles.

So on the next port: **do not spend effort auditing observation scaling under
empirical normalisation; do audit every reward scale, because nothing normalises
those.** Cheap, correct, and it explains rather than merely records why one of two
inherited mismatches bit and the other did not. Generalisation due to the
coordinator.

## A correctly-established fact, and an unexamined claim about what follows

A distinct species from the arithmetic slips. Both instances tonight had the same
shape: the evidence was right and the inference from it was not.

- "`pose_error/10.0` exists and is a 600x disparity for the Go2" — TRUE, verified.
  "Therefore the network cannot act on position error" — FALSE, refuted by the
  running std and the first-layer weights.
- "All five CRM collection commits contain 133427b and pychrono has not changed" —
  TRUE, verified. "Therefore the shim change cannot be what broke CRM" — the
  reading was right but the conclusion drawn from it ("the environment changed
  under us") was wrong; the real cause was two Chrono builds selected by
  PYTHONPATH.

**An arithmetic slip announces itself to anyone who redoes the arithmetic. This
one survives any amount of rechecking of the part that was done**, because the
unverified step is invisible among verified ones. It is also the class that
survives review by other people, since reviewers check what they were shown.

The only defence found so far is to state the inference as a separate claim and
test it separately — reading the actor weights, not just the observation scale.


## When a comparison goes wrong, suspect the apparatus before the subject

**Cost:** three wrong answers in one day, one of them a discarded 220-target build · **Found:** 2026-09-04 · **Applies to:** any differential measurement — diffs, before/after, baseline vs change

Distinct from [state the scope with the result](#state-the-scope-with-the-result-or-the-conclusion-inherits-one-it-never-had),
which is about a test's *reach*. This is about **differential** measurement, where the
answer is a delta: if the machinery producing the delta is stale, misconfigured or
pointed at the wrong reference, it yields a **plausible, confident, wrong** number and
nothing about the output looks unusual.

Three instances, same shape. *(The first two are reported by the Chrono session and are
not verified here; the third is mine and was traced on this box.)*

| the comparison | what the artifact did | what it looked like |
|---|---|---|
| build against a **stale ninja file** | replayed an old failure | a **regression** introduced by the change |
| **two-dot** diff across a rebase | folded in upstream's movement | branch **content** the author never wrote |
| verifying "the OptiX path" with **OptiX compiled out** | measured Vulkan RT throughout | a **passing OptiX verification** |

In each the subject was fine and the instrument was not. And in each the wrong answer was
the *reassuring* one — a green verification, a clean attribution, a regression with an
obvious culprit — so nothing prompted a second look.

**Why this is worse than a failure.** A silent failure leaves you with no answer and you
go looking. A silent *wrong thing* leaves you with an answer, and answers do not prompt
investigation. The Chrono flag that started the third case is the pure form:
`-DCMAKE_CUDA_ARCHITECTURES=120` is **accepted without complaint** and then discarded
with `FORCE`, so the build succeeds while targeting the wrong architecture.

**Fix: assert the apparatus is what you think, from evidence outside the comparison.**
Not "it configured and the tests passed" but the cache variable, the linked libraries,
the merge base, the file timestamp:

```bash
grep -E "^CH_USE_SENSOR_OPTIX:" CMakeCache.txt      # the backend, not the intent
ldd bin/demo_SEN_camera | grep -E "nvrtc|cuda"      # what actually links
git diff upstream/main...HEAD -- <path>             # three-dot, against the merge base
```

The check must be **independent of the thing under test** — that is the whole content of
the rule. Verifying the OptiX build by running the OptiX tests is circular when the
question is whether OptiX is in the build at all.

**What exposed the third one was luck**, and worth naming as such: `demo_SEN_Gator` was
missing from `bin/`, and only because it happens to be gated behind the same `if()` as
the backend. There was no diagnostic designed to catch it. Do not rely on the next one
being similarly convenient.


## Four conventions for the same twelve values, and none of them is wrong

**Cost:** near-miss, caught before collecting 304 episodes · **Found:** 2026-09-04 · **Applies to:** any schema where several consumers index the same physical set

The Go2's twelve joints are ordered **four different ways** in one codebase, each
correct for its consumer:

| convention | order | who wants it |
|---|---|---|
| `MOTOR_NAMES` | RR RL FR FL | `robot.joint_pos()`, `joint_vel()`, `actuate()`, and `JOINT_ACTION_FIELDS` — the target columns in every CSV |
| `LEG_ORDER` | fl fr rl rr | the `foot_*` columns |
| `FOOT_BODIES` | FR FL RR RL | the body-name list |
| imported policy | FL FR RL RR | reached via `CHRONO_TO_IMPORTED` **and** `SIGN = -1.0` |

**This is not a naming problem to be cleaned up.** Four consumers legitimately want
four different orders; a codebase-wide "fix" would just pick one and break three.

**Two rules that actually work:**

**1. New columns take the order of the columns they must align with**, not the order
that seems canonical in the abstract. Adding 12 joint positions and 12 velocities, the
instruction was to use `LEG_ORDER` — reasonable, since that is the constant that guards
the `foot_*` columns against exactly this class of bug. It was wrong here: the joint
targets are in `MOTOR_NAMES` order, so `q`, `dq` and the previous action have to be too,
or they silently disagree with the action columns beside them in the same row. The rule
that avoided it is **"align with the neighbours", not "use the canonical constant"** —
because which constant is canonical depends on the column family.

**2. Log raw, and never bake a consumer's transform into the file.** The imported policy
needs `SIGN` and a permutation applied. Both stay in
[`imported_policy.py`](../../../src/nedm/quadruped/imported_policy.py); the CSV records
what the simulator reported, in the simulator's order. A dataset carrying one consumer's
convention is wrong for every other reader **and undiscoverable from the file**, since
sign-flipped joint angles look entirely plausible.

**Evidence:** `quadruped/dataset.py` (`JOINT_STATE_FIELDS`, `LEG_ORDER`),
`quadruped/constants.py` (`MOTOR_NAMES`, `FOOT_BODIES`), `imported_policy.py:224`.


## Some properties cannot be identified from symmetric data, and no test on it will say so

**Cost:** a reported misalignment that did not exist · **Found:** 2026-09-04 · **Applies to:** verifying any index/label assignment against periodic or symmetric motion

Checking that 12 new joint-position columns were matched to the right joints, against a
**constant forward walking** episode:

```
12x12 correlation, measured vs target : diagonal is argmax  0/12
12x12 RMS,         measured vs target : diagonal is argmin  2/12
                        both front hips appeared sign-inverted
```

Read literally that is a catastrophic misalignment. **It was an artifact of the gait.** A
trot moves diagonal pairs identically and mirrors left against right, so the twelve target
signals are phase-shifted near-copies of one another, and a measurement that lags its own
target legitimately resembles a different leg's target more closely.

**The data did not contain the information the test needed.** No amount of care with the
statistic would have fixed it — under a symmetric gait, leg identity is simply not
identifiable, and both a "confirmed" and a "refuted" verdict would have been unfounded.

**Fix: verify on a regime that breaks the symmetry.** Re-run on `pivot`, where the legs
must do genuinely different things: diagonal is RMS-argmin **8/12** with clear margins,
the four misses have narrow margins and sit on the worst-tracking joints, and the two
"inverted" hips resolve. Supported by an argument the numbers alone do not give: **a real
permutation would move a leg's hip, thigh and calf together, and would not dissolve when
the command changes.**

**The dangerous version of this is the reverse.** Had the symmetric episode happened to
return 12/12, it would have been recorded as confirmation — from data incapable of
confirming anything. Before trusting a check, ask whether the data could have produced
the opposite answer.

Related: [when a comparison goes wrong, suspect the apparatus](#when-a-comparison-goes-wrong-suspect-the-apparatus-before-the-subject).
There the instrument was broken; here the instrument was fine and the **regime** carried
no signal.

## When the codebase documents a trap, quote the documentation — recalling it is not enough

`dataset.py:105` carries a comment block enumerating **four** orderings of the
same twelve Go2 joints:

    MOTOR_NAMES   RR RL FR FL   joint_pos/joint_vel, actuate(), target columns
    LEG_ORDER     fl fr rl rr   the foot_* columns
    FOOT_BODIES   FR FL RR RL   the body-name list
    imported      FL FR RL RR   the policy, via CHRONO_TO_IMPORTED and SIGN

That block exists so nobody has to remember. It was written after an ordering bug
and it is exact.

An instruction to build a joint-space controller then arrived carrying "MOTOR_NAMES
is FR/FL/RR/RL" from memory — which is `FOOT_BODIES`, one line further down — **in
the same message that warned to verify orderings against the code rather than
against the instruction.** The warning was right and the claim beside it was wrong,
for the second time on the same point.

**A warning to check something, delivered alongside an unchecked claim about that
same thing, is worse than no warning: it lends the wrong value the authority of
the caution.** The reader who trusts the warning is exactly the reader who will
also trust the number next to it.

So: when the codebase documents a trap, the only acceptable form of the
instruction is a quotation with its file and line. Paraphrase from memory is how a
comment written to prevent an error becomes decoration beside a repetition of it.

(And the recipient undercounted too — reported "three orderings" when the block
says four. Both errors are the same kind: describing a document instead of reading
it.)

## An error identical across independent units is a shared constant, not a per-unit bug

Validating Go2 forward kinematics against Chrono, the foot position was wrong by
**0.0218 m on all four legs**. Being identical is the whole diagnosis: a sign
error, a leg-ordering error or a left-right transposition would differ BETWEEN
legs, because those bugs act on per-leg quantities. A constant common to all four
can only come from something shared.

It was the base frame: the base **body** origin sits at the COM, displaced
(0.02111, 0, −0.00537) from the base **link** frame that the URDF's hip offsets
are expressed in. Measured at spawn — before any dynamics, when the link frame is
exactly the spawn frame — rather than hardcoded. FK then matched to 0.00000 m.

The discriminating question is cheap and worth asking first: **does the error vary
across independent units, or not?** It separates "shared frame or constant" from
"per-unit sign or index" before any hunting starts.

The same run also produced a plausible-looking wrong answer worth naming. Link
lengths taken as distances between BODY origins in the rest pose gave thigh 0.2962
and calf 0.0982 — reasonable numbers that survive inspection. What killed them was
a physical consistency check, not a second look: they cannot reach a foot 0.426 m
below the hip. The URDF joint frames give L_thigh = L_calf = 0.2130, which extends
to exactly 0.426. **Check that a geometric quantity closes against an independent
measurement of the same geometry**; plausibility is not a test.

## Before running a verification step, state what result would constitute failure

If no reachable result would say "no", the step is decoration — and worse than no
step, because it produces the feeling of having checked.

The instance: a collection was gated on confirming that `--ground-size-m 200.0`
and `--perturb-peak-n` appeared in the first episode's log line. The driver logs
progress counts, not subprocess command lines. **That grep returns nothing whether
the design flag took effect or not.** It was handed over as the safeguard against
a provenance defect that had just been described in the same message, so passing
it would have meant signing off on a possibly-wrong dataset while believing it
verified.

What replaced it was falsifiable:

    168 columns against the old design's 69
    3889 rows against 1475
    duration_s 41.25 against 16
    plant_bed_m [-100, 100, -100, 100] against [-5, 5, -5, 5]
    travel x -1.85..40.28 -- 40 m, PHYSICALLY IMPOSSIBLE on the old 10 m bed
    cmd_vx 1.634, outside the old +/-0.5 envelope

Any one of the column or row counts could in principle be a coincidence. **Travel
exceeding the old bed size cannot be** — it is the one that could have come out
"no", and it is what makes the set a check rather than a tally.

Same family as "an n=8 correlation quoted without its p", and as "a comparison
that looked too clean", reaching the same conclusion from a third direction. This
one is the worst of the three because it was a check on PROVENANCE rather than on
a result: a wrong result gets argued about later, and wrong provenance is
undetectable later by construction.

The habit that generalises: **write down the failing outcome before running the
check.** If you cannot name one, you have not designed a check yet. The same test
applies to a clamp in place of an exception — `Unreachable` raised from the IK has
a failing outcome; silently clamping the acos domain does not.

## Chaos bounds prediction from an UNCERTAIN start, not from a KNOWN one

**Cost:** caught in review, before it became a reported "physical floor" · **Found:** 2026-09-04 · **Applies to:** any error floor claimed from sensitivity analysis

Measuring how fast two nearly-identical simulations separate is a good way to
characterise a system. Turning that separation into a floor under **model** error is
a different claim, and it does not follow.

**The two experiments are not measuring the same thing:**

| | initial condition | dynamics |
|---|---|---|
| twin separation | perturbed | **exact** |
| model rollout | **exact** | approximate |

A deterministic system given its exact state has **no floor from chaos at all** — a
perfect model reproduces it forever. Chaos supplies the **amplification**, not the
seed. So "the twins separate by 84% of travel at 5 s, therefore a model reporting 9%
at 10 s must be measuring something else" is invalid: the model starts where the
twins did not.

**The tempting error is to report a floor a better model is entitled to walk straight
through**, which is worse than reporting no floor, because it makes a real improvement
look impossible and a broken metric look vindicated.

**A weaker version does survive, and it is the useful one.** The model's initial
condition is not exact either — it is whatever the pipeline stores. Here
`training/preprocess.py` casts states, actions, targets and rollout to **float32**, so
a joint angle of order 1 rad reaches the model known to ~1e-7 rad. That IS an
uncertain start, so:

> the twin separation, seeded at the pipeline's own storage precision, is what a
> **perfect** model would produce. Model error should sit above it — not because a
> model cannot beat a twin, but because it **inherits the same uncertain start** and
> adds its own error on top.

**Seed the twin with the actual quantization, not a number near it.**
`x0_pert = np.float64(np.float32(x0))` reproduces exactly the error the pipeline
introduces. A flat perturbation is wrong per-channel — float32 is *relative*, so 1 rad
quantizes to ~6e-8 while 0.01 quantizes to ~6e-10, a spread of two orders of magnitude
that correlates with which channels matter. It also removes a free parameter: "we chose
1e-6, then 1e-7" invites the question of what 1e-8 would have shown.

**Find the coarsest step before assuming which one it is.** The floor is set by the
worst quantization on the path, not by the one you thought of. Checked here: the raw
CSVs are written by `csv.DictWriter` with no formatter, so Python emits full float64
round-trip text (16–17 digits, `repr(float) == text`). Had they gone through a
`%.6f`-style formatter the text would have been ~1e-6 absolute — **an order of
magnitude coarser than float32**, and the floor correspondingly larger. One `sed -n
'2p'` settles it and it changes what you perturb.

**Report the floor and the error together.** "9% against a 3% floor" says something
"9%" cannot, and it is only available in a system that amplifies storage precision to
something visible on the horizon of interest.



**Two independent reviewers proposed and relayed this check without either noticing.**
A mistake that survives two people is evidence about the **shape of the check**, not
about anyone's attention, which is why the rule has to be procedural rather than an
exhortation to be careful.

**The same principle from the other side.** A test written the same hour caught a bug
in *itself* on its first run -- it compared `JOINT_ACTION_FIELDS` against `MOTOR_NAMES`
without stripping the `joint_` prefix, and failed. That is what a reachable "no" looks
like: it can report failure, so it did, immediately, on its author. Each assertion in
`test/test_joint_orderings.py` was then fed a deliberate transposition and shown to
reject it, because a test never shown to fail is not yet evidence of anything.

**Corollary for the rationale, not just the test.** A passing test with a wrong reason
attached teaches the next reader the wrong thing while looking like it teaches the right
one. One assertion there was justified as catching a transposition the existing
assertion missed; measurement showed the existing one caught it too. The test was kept --
its value is independence from a hardcoded literal -- but the docstring was corrected,
because the reason is the part that gets reused.

## A guard whose absent input is also its failure signature must treat absence as failure

Not "a skip should fail" as a coding style. A design rule about which gates can
afford to skip.

`G11 design agreement` exists because a dataset's design — ground size,
perturbation, prewalk, slope — lives in the operator's ENVIRONMENT and appears in
no episode artifact. A half-set environment produces old-design episodes
indistinguishable from new-design ones, and pooling two roots collected under
different designs is undetectable afterwards. `run_manifest.json` is the stopgap;
G11 is what makes it load-bearing.

**And G11 skips when no manifest is found, and a skip does not fail the
validator.** `[SKIP]` prints without appending to `failures`, so the run returns 0.

**The disarming condition and the danger condition are the same condition.** An
operator who did not write a manifest is exactly the operator who may not have
set the environment either. So the gate turns itself off precisely when the thing
it guards against is most likely to have happened, and reports success while
doing it.

The test to apply when writing a gate: **ask what makes this check unable to run,
and then ask whether that same circumstance makes the defect more likely.** If
the answers coincide, absence must be reported as failure — or at minimum as
INCOMPLETE, never as a pass.

Contrast a gate that can afford to skip: `G10 rows` skips unless `--check-rows`
is passed. Its absent input is an explicit operator choice not to spend the time,
which is unrelated to whether the rows are correct. Skipping there is honest.

Related, from the same review: `effective_design` versus `resolved_design`. The
gate read one key and the manifest wrote the other, so G11 would have compared
`None` against a real design and either passed vacuously or flagged a spurious
mismatch — on the one gate whose whole purpose is preventing that pooling error.
**No amount of running the gate finds this**, because it returns 0 either way. The
fix points both names at the SAME dict object rather than duplicating the values,
since two keys holding two copies of one design is a future disagreement waiting
to happen.

## A check that fired once must be re-run in every condition it could apply to

**A boundary optimum is a self-advertising defect** — the number sits on the edge
of the grid and announces itself. **A wrongly-tuned optimum in an untested
condition just looks like a number.**

Tuning a proportional baseline on rigid ground, the optimum landed on the grid
corner. That was visible, so the grid was extended — twice, since it ran to the
boundary again. Then the same gains were used to evaluate on SOIL, and that
condition was never searched at all. Tuning on soil later gave an INTERIOR
optimum at a value 3.3x better than the one used, and the published conclusion —
that the learned policy beat the baseline on soil — was a tuning artifact and had
to be withdrawn from the status page.

The rule: **the conditions where a check stays quiet are indistinguishable from
the conditions where it was never run.** Having fired once is evidence the check
works, not evidence the other conditions are clean. So re-run it everywhere it
could apply, and record where it was run rather than only what it found.

Same shape as two other defects found the same day:

- a validator gate that SKIPS when its input is missing, printing `[SKIP]` and
  not failing — so the run reports success
- a verification grep for flags that the logs never contain, which returns
  nothing whether the design took effect or not

All three are **absence of a complaint being read as evidence of correctness.**
The self-advertising instance got fixed immediately; the silent ones each needed
someone to ask. That asymmetry is the thing to design against, because the defects
that announce themselves are the ones least likely to survive anyway.

## Stage outside the live path; move in only once the thing is characterised

A 1,762-episode dataset was tarred **directly into the Syncthing share**. Syncthing
began propagating as the file was written, so by the time the size was measured —
4.91 GB compressed, against the 0.108 GB transfer the authorisation had compared
it to — the artifact was already replicating to two other machines.

**The decision point passed before the decision could be made, and the cause was
the choice of where to write.** Nothing about the measurement was slow; the write
target removed the ability to reconsider.

Staging it outside the share and moving it in once characterised costs one `mv`
and preserves the choice. That is the whole rule:

**Write to a location with no side effects. Move it into the live path only after
the artifact is fully characterised.**

It applies to anything that starts acting the moment it exists — a dataset in a
sync share, a checkpoint in a directory something globs, a config in a path a
launcher reads, a file in a watched folder. Same class as a mid-run mutation: the
harm is not in the content but in the timing being taken out of your hands.

The related distinction, which is what made the problem visible at all:
**"authorised" and "informed" are different.** The content transferred was exactly
what was approved. The footprint was not what the approving sentence conveyed —
it said "mirroring what the other machine already does," and the other machine's
transfer was a fortieth of the size. An approval obtained on an accurate
description of *what* can still be uninformed about *how much*, and the person who
wrote the description is the one who owes the correction.

## A test that cannot reject is not evidence of agreement

An exact two-sided sign test on n = 5 has a minimum attainable p of 0.0625. Citing
p = 1.000 from it as "the halves agree" reports the test's own powerlessness as a
finding about the data. Before quoting a p-value, ask what the test *could* have
detected; if the answer is "nothing at these n", quote the effect size against the
sampling spread instead. Same shape as the recurring "check whose success path is
reachable without the thing being checked" — here the success path is reachable
without the halves agreeing.

## Compute the noise floor before choosing the threshold, not after

Setting a target effect first and discovering afterwards that it sits inside the
sampling noise produces a number nobody can interpret — the run is spent and the
result is unreadable. Bootstrap each candidate statistic at each candidate
granularity from the baseline itself, then pick the granularity so the target
exceeds the noise. For the Go2 fine-tune this ruled out the intended scoring cell
outright: a ±10-point criterion on the low-command ratio needs n ≈ 344 against 74
available.

## Pooling is variance reduction only when the parts agree

"Raise n by pooling across families so noise falls as 1/sqrt(n)" assumes one
population. Pooling averages down the within-group component and leaves the
between-group component intact, so where between exceeds within, the pooled
interval is *wider* than the sub-groups it came from — measured on Go2 backward-low,
between-family sd 45.5 pts against within-family 32.7, and the pooled half-width
(30.5 pts) worse than four of five constituent cells. Decompose before pooling.
Note this arose as the proposed *fix* for a family-composition error: the same
part-whole confusion reappeared inside its own remedy.

## Prefer the statistic without the denominator when the denominator is small

Seventh instance of this class. Backward-low tracking scored as achieved/commanded
divides by 0.023–0.18 m/s, so a fixed 0.05 m/s error reads as 47% at the median
command and 213% at the smallest; the resulting CI spans zero and cannot establish
even the sign. The same episodes scored as absolute velocity error give a CI far
from zero and pool legitimately in both bins. When a ratio's denominator is itself
the swept variable, most of the spread is manufactured by the division.

## Pairing can beat any n you could afford to collect

Running both arms on the same commands cancels everything explained by command
magnitude and family. Measured on the Go2 baseline that is 64% of the variance in
the low bin — more than quadrupling the effective n there, against a 4.6× shortfall
that no realistic collection would close. Check what a paired design removes before
concluding a measurement needs more data.

## A variance reduction is not a noise reduction

Removing 64% of the variance removes only 40% of the sd, and thresholds live in sd.
Worse, a variance decomposition measured on ONE arm's distribution is an upper bound
on what a paired design delivers: the structural component cancels only if both arms
respond to it identically, and whatever fails to cancel enters the difference twice,
at sqrt(2). On the Go2 low bin that is the difference between an 11.8% and a 16.7%
detectable effect — the gap between the two is larger than several of the effects
under discussion. Quote the detectable effect in the units of the claim, not the
fraction of variance explained.

## If two runs never repeat a condition, the irreducible noise is unmeasured

All 1,762 (family, command) combinations in the Go2 collection are distinct, so no
amount of reanalysis can separate treatment effect from run-to-run variability — the
data contains no replicate. That is invisible until you go looking for one, because
a large dataset feels like it must contain repeats. Deliberately replicating a
handful of conditions costs almost nothing at collection time and is the only way to
learn the floor afterwards. Here the retrofit probe is 64 episodes and 1.5 minutes,
which is the cheap case; on a slower plant it would not have been recoverable.

## NaN fails every comparison, so a divergence reads as whatever you test next

`not (lo <= x <= hi)` is True when x is NaN, so a diverged solver was recorded as a
bed-boundary exit, broke the step loop before the first row, and died three functions
away on `min()` over an empty list. 144 of 238 lost Go2 episodes at offset 3,000,000
took that route, and the error message named the symptom in a different file from the
cause. dorm-pc hit the same mechanism in its boundary flags, mislabelling 296
episodes. **Test finiteness explicitly and first, before any comparison whose False
branch means something else.** Two distinct symptoms, one silent and one fatal, from
one root cause.

## Python's hash() of a str is per-process randomised

Spawn jitter seeded from `random.Random(hash((family, json.dumps(params))))` drew a
different position on every invocation — measured -0.138, +0.421, +0.458 for the same
inputs. Within one process it is consistent, so a single collection looks
self-consistent and reproducible; it is not. This is invisible until a second run has
to match the first, and a paired evaluation generated that way would give the two
arms different realisations while every log said "same commands". Use `hashlib` for
any seed that must survive a process boundary.

## Difference of medians is not median of differences

On the same 49 pairs: difference of medians +0.0138 m/s (treatment looks harmful),
median of paired differences -0.0000 (no effect). Only the second has the shared
episode difficulty removed. If a design is paired, the statistic must be paired too;
pairing the data collection and then computing an unpaired summary discards the
entire benefit while appearing to keep it.

## A quantity measured on a simplified set does not transfer to the diverse one

Two correct measurements of the same correlation: r = 0.959 with spawn, heading,
tilt, prewalk and perturbation pinned so only the command varied, and r = 0.737 with
the realisation varying as the collection does. Sizing thresholds on the first would
have understated the noise by about 15%. Same shape as pooling across heterogeneous
families: the population you measured on has to be the population you will score on.

## A bright line on a noisy statistic rejects a null treatment at chance

"The wrong-way fraction must not increase" sounds strict and defensible. On a
statistic carrying +-14 points it rejects a treatment whose true effect is zero about
half the time — noise with a verdict attached. The fix is to keep the decision made
in advance (a regression is a failure, whatever the primary does) but test it for
significance, here McNemar on discordant pairs, which uses the pairing. Symptom to
watch for: a document containing both a bright line and a named hypothesis test for
the same quantity. That is two rules, and the bright line is usually the one written
first and never re-examined.

## Negative-control the acceptance criterion, not only the measurements

An acceptance criterion is a check like any other, and it deserves the same
reachable-failure test we demand of assertions: run it against a treatment that
ought to fail, before the real one exists. Random weight perturbation is a
known-null; putting it through the Go2 criterion showed the primary correctly not
firing (-0.0000 against a -0.020 threshold) and, after the anchor was fixed, the
anchor correctly not firing either (McNemar p = 0.227). The first version of the
anchor DID fire on that null, and it had been written into the document as expected
behaviour — the negative control is what exposed it. A criterion validated only by
the result it eventually produces is validated too late.

## When the noise scales with the effect, "the noise floor" is not a number

Two machines measured the paired sd of the same quantity as 0.0325 and 0.0667 and it
looked like a twofold contradiction. It was not: the paired difference is the
treatment-by-realisation interaction, so its variance grows with the size of the
treatment, and the two runs used perturbations differing 3.4x in effect. All three
measurements fit `paired_sd = 0.307 * shift^0.58` within 3% across both machines.

Two consequences. **Size the noise at the effect size you intend to detect** — using
the noise of a much larger treatment demands the criterion resolve a small effect
through a big effect's variance, and here that turned a 1.79-sigma threshold into
0.86. And **"detectable" becomes an implicit condition rather than a fixed number**:
E is detectable when `E >= 1.96 * 1.25 * sd(E) / sqrt(n)`, which has to be solved
rather than looked up. Quoting a single sd is what made two consistent measurements
look contradictory.

Fourth instance in one day of a specification difference presenting as a measurement
difference, after the selection predicate, the manifest vocabulary and the pinned-vs-
varied realisation. The tell each time was a control that AGREED while the headline
numbers did not — here pair loss at 25%, 23% and 22.4% predicted, which located the
difference in the treatment rather than the harness.

## A good method plus an unidentified parameter is still not a number

`E >= 1.96 * 1.25 * sd(E) / sqrt(n)` is the correct way to ask what is detectable
when noise scales with the effect. Solving it needs the exponent of that scaling, and
three points against two parameters gave a slope of 0.579 on one degree of freedom.
The central estimate got quoted as "the floor is 0.0050 and our threshold is four
times it", which reported a fit as though it were a measurement.

**Then both attempts to bound that claim were themselves invalid, in the same way:**
each varied the exponent while holding the fitted coefficient fixed, and in a log-log
fit those two are strongly correlated, so the pair stops describing the data at all —
one such combination predicted an sd of 0.193 where 0.0325 had been measured. Varying
one parameter of a correlated multi-parameter fit does not produce a confidence
range, it produces incoherent curves. One of those analyses also had the inequality's
direction backwards: `E >= k*E^b` becomes a CEILING when `b > 1`, satisfied by all
small effects, not an impossibility.

The fix is not to abandon the method but to separate it from the claim: keep the
inequality, drop the value, and note which quantities are anchored on measurements
instead. Here the threshold sat at the lowest measured point, so it survived the fit
being undetermined — a good reason to place thresholds ON measured points rather than
in the interpolated space between them.

## Prefer the construction with the fewest assumptions between data and claim

Three ways to put an interval on a median of 49 paired differences: the normal-theory
`1.96*1.25*sd/sqrt(n)`, which assumes normality and uses a non-robust sd to size a
robust statistic; the bootstrap, which assumes nothing but is known to run optimistic
for medians at small n because its sampling distribution is a coarse lattice of order
statistics; and the exact order-statistic interval, whose coverage is binomial and
which assumes nothing at all. Measured here, they gave 0.0114, 0.0096 and 0.0122 on
the same data — the normal formula wrong by 4x in one regime and by -8% in another,
which is worse than being conservative because the direction is unpredictable.

Where two constructions agree, the result is real: at the smaller treatment the exact
and bootstrap intervals matched to four decimals. Where they disagree, take the one
making fewer assumptions. A large resample count does not repair a biased estimator —
it just measures the wrong thing precisely.

## A control earns its keep by disagreeing, not by agreeing

All five same-name-different-thing collisions in this study were caught the same way:
**a control that AGREED while the headline numbers did not.** Pair loss matched three
ways (25%, 23%, 22.4% predicted) while two paired sds differed twofold, which located
the difference in the treatment rather than the harness. The per-family table matched
while the pooled medians differed by 26 points, which located that one in composition.

The general form: a control's diagnostic value is highest when it disagrees with the
thing it is controlling for, because agreement everywhere carries no information. A
control that always moves with the headline is not doing any work and should be
replaced by one that can come apart from it. Design controls that CAN disagree, and
report them even when they do not.

## State what a negative result means, in the same breath as the threshold

"FAIL" reads as "the thing does not work" unless the document says otherwise. It
means only that an effect of at least the threshold size was not demonstrated. At the
Go2 criterion's sizing a genuine 0.015 m/s improvement fails four times in five, and
an effect exactly at the threshold is a coin flip — a threshold is a decision
boundary, not a detection guarantee. Write the power at a few true effect sizes
beside the threshold, so the failure cannot be over-read later by someone who was not
in the conversation that set it.

## Two agents agreeing is not a control when they share the premise

Pair loss was modelled as `1 - 0.881^2 = 22.4%` because either arm could fail. The
coordinator derived the same figure independently, and the agreement is presumably
why neither of us examined it. It was wrong: in a design where the baseline arm is
an ALREADY-COLLECTED episode, that arm cannot fail — it survived collection and
passed the predicate by construction — so only the treated arm is at risk, and it
runs on specs pre-selected for having worked once. Measured loss was 5%, not 23%,
which moved the surviving-pair count from below the minimum to above it.

Independent derivation from a shared unexamined premise is not independent
confirmation. The check that would have caught it is the one that did: running the
thing and counting.

## A tool written to enforce a rule can rebuild the violation inside itself

The verdict harness printed the vacuity warning correctly — "smallest attainable
McNemar p = 0.0625, cannot reject" — and then its verdict logic counted the
non-firing anchor as a satisfied rule. That is the n = 5 sign-test error, rebuilt
inside the tool written to prevent it, one screen below the warning that names it.
Printing a caveat is not enforcing it: check that the control flow acts on the
warning, not merely that the warning exists.

## Surviving the pipeline is not evidence of being real

The verdict harness selected baseline episodes on the argument that a collected
episode "passed the predicate by construction". It had not: an episode can blow up to
absurd-but-FINITE values and clear every finiteness check in the pipeline. One
eligible episode carried a max joint angle of 137 rad and a joint target of 2e34
against a physical range of about +-3 rad. Finiteness is a much weaker property than
plausibility, and any filter built from `isfinite` alone inherits that weakness.

Check magnitudes against physical bounds, not just against NaN. Where the population
is bimodal — here p99 at 3.6 rad and the next value at 136 — the threshold is not a
judgement call, and saying so is what makes the exclusion defensible.

## The first sufficient explanation is where you stop looking

`rigid_constant_70` failed the harness's bit-identical replay check. The diagnosis —
a divergence guard I had added that raised where the original broke — was correct,
and the fix was right. But the same episode was ALSO physically absurd, and that
went unnoticed for another hour until a peer raised admissibility as a general
concern. An episode with a joint target of 2e34 failing to reproduce is not a
coincidence; the two defects were related, and finding the first is precisely what
stopped the search for the second.

**The general form: a CORRECT diagnosis is the most effective thing there is at
stopping a search.** A wrong one gets contradicted by the next piece of evidence and
forces a revisit. A right one satisfies you and closes the question, so a second
defect sitting behind it is never looked for and nothing ever prompts a second look.
Being right is the more dangerous case precisely because it is stable.

The check is cheap and belongs at the END of a diagnosis rather than being a new
technique: **ask whether the explanation accounts for the WHOLE symptom or merely
enough of it**, and whether the failing case is unusual in any other respect before
closing. The replay mismatch was fully explained by the guard regression, so nothing
pointed at 2e34 — one glance at that episode's joint magnitudes would have.

Pairs with [[two agents agreeing is not a control]]: both are a satisfying answer
suppressing further inquiry, approached from different directions — one where the
answer is your own, one where it is a collaborator's.

## Do not force two exclusions to agree until you know they answer the same question

Training exclusion and evaluation exclusion have opposed failure modes: excluding
aggressively from training protects the normalisation and the learned dynamics, while
excluding aggressively from evaluation costs representativeness, because a physically
real but difficult episode is exactly what should be scored. Two collaborators
excluding 14.3% and 0.7% is not necessarily an inconsistency to reconcile. Establish
what each filter is FOR before treating a numeric disagreement as a defect — and then
enforce consistency only where it actually bites, here across the two halves of the
evaluation set.

## A number embedded in an explanation goes stale silently

The verdict harness's FAIL message read "a true 0.015 m/s improvement fails about
four times in five." True at n = 33, wrong at n = 107 where it is nine in ten — and
it would have been read at the exact moment it mattered, in the message explaining a
negative result. Worse, it erred toward UNDERSTATING our own criterion, which is the
direction nobody audits. Compute such numbers from the run rather than writing them
into prose; a hardcoded figure inside an explanation has no test that fails when it
drifts.

## Label which side of a threshold a rate describes

A power table listing 0.010 through 0.030 against a 0.020 threshold shows numbers
falling with n below the threshold and rising above it. Both are the test improving:
above, the rate is power; below, it is the rate of passing an effect smaller than the
one declared meaningful. Unlabelled, the falling column reads as lost sensitivity.
Say which side is which in the table itself, not in the surrounding text.

## A simulation is not reproducible across machines until you have checked it is

Chrono episodes replay bit-identically on the box that produced them and not on the
other one: different build, different arithmetic in the last digits, and a chaotic
plant amplifies it into 147 differing columns by row 0. Separation was perfect by
machine, 14 of 14 foreign pairs differing and 6 of 6 native pairs identical.

Determinism is a property of a build, not of a simulator, and "it is deterministic"
is almost always shorthand for "it is deterministic here". Any design that replays
recorded episodes has to be stratified by the machine that recorded them. Ours was —
the baseline arm is the recorded file so it is unaffected, but the treated arm must
run where its baseline was collected, or the arms differ by build as well as by
treatment.

## A check that fires correctly but attributes wrongly still costs the time it saved

The replay check would have caught the cross-machine problem — as an unexplained
bit-identical failure. Someone would then have gone hunting for a collector
regression, which is precisely what happened with `rigid_constant_70`. Detecting a
fault and naming it are different services, and a guard that stops the run without
saying why hands the diagnosis back to whoever is least prepared to do it. Where the
cause is cheaply testable in advance — here, comparing the recorded `machine` field
to the hostname — check for it explicitly and say so, rather than letting a general
integrity check discover it as a mismatch.

## Check which implementation a claim is about before reasoning from it

I argued that upstream's joint-space reward terms could not transfer, because our
fine-tune acts in command space into a frozen low-level policy. That was true of
`Go2NeuralTrackingEnv`, the environment I had built, and false of the joint-space
environment the fine-tune actually uses — the repo contains both, and I reasoned from
the one I knew rather than checking which one was running. Had it been accepted it
would have removed real terms from the objective on a false premise.

The failure is not being wrong about the code; it is answering a question about
SYSTEM A using knowledge of SYSTEM B without noticing the substitution. Familiarity
with one implementation is what makes it feel unnecessary to check. When a repo holds
two designs for the same job, name the one you are reasoning about in the claim
itself, so the substitution has somewhere to become visible.

## A test whose wrong answers coincide is not a test

The first sign-convention check displaced all four Go2 hips by the same 0.35 rad.
Every implementation — correct, unsigned, and leg-mispaired — returned 1.4000 and
passed, because the two hips with default +0.1 and the two with -0.1 cancel exactly
under both wrong conventions. The test looked discriminating and did nothing.
Unequal displacements removed the cancellation.

Symmetric test inputs are the usual cause: symmetry in the case can annihilate
exactly the asymmetry the test is meant to expose. **Report the discrimination matrix,
not the verdict** — run each wrong implementation you can think of against every test
and check that each TEST is failed by something, not merely that the SUITE rejects
each variant. The suite here was sound as a whole while one of its tests was inert,
and only a per-test check made that visible.

## A story that explains the evidence elegantly is not thereby true

I reported that `hip_to_default` needs the recorded joint positions negated while
`dof_pos_limits` does not — one term referencing an imported constant, the other the
URDF, an elegant trap that "cuts both ways". It was wrong: the URDF calf range is
entirely negative and recorded calf positions are entirely positive, so both terms
need the same single negation. There is one convention shift, not two.

The claim survived because it sounded like the kind of thing that is true, and it was
stated more confidently than an unmeasured claim deserved. Symmetry and irony are not
evidence. Note also how it was caught: not by re-reading, but by gathering real joint
limits to BUILD a test, where the numbers refused to fit. Constructing the check found
the error in the specification before it examined any implementation.

## A suite validates implementations against its reference, not the reference

A green discrimination matrix says every candidate matches the reference encoded in
the harness. It says nothing about whether that reference is right: had the Go2
sign convention in it been wrong, the reference would have encoded the error and
every cell would still have read green. That convention rests on a separate check
entirely — measured joint ranges against the URDF — and the two validations answer
different questions.

Write the distinction where the matrix is READ, not only where it is documented. The
scope note belongs in the tool's own output, because the person most likely to
over-read a green matrix is the one looking at the output rather than the design doc.

## Sign tests cannot catch a form error

The reward-term suite asserted only that `dof_pos_limits` was ">0" inside the soft
band and "==0" at the nominal pose. Both hold for a SQUARED implementation as well as
the correct linear one, so adding that variant made the suite fail its own soundness
check. The fix was to assert a magnitude with a closed-form expectation: with every
joint at 0.48 of half-range from the midpoint, the excursion past the 0.45 soft limit
is exactly `0.03*r` per joint, so the correct total is `0.03*sum(range)` = 1.0849
against 0.1208 squared.

Predicates of the shape "is it positive" or "is it zero" test that something FIRED,
not that it computed the right thing. Where a closed form exists, assert the value.

## Being unsure is weakly correlated with being wrong

dorm-pc flagged two things it was uncertain about — `tracking_sigma = 0.25` and the
linear form of `dof_pos_limits` — and both were correct. The one real error, squaring
`hip_to_default` where upstream uses `torch.abs`, was in a term it did not flag. Its
confidence was calibrated in exactly the wrong direction, which is normal: doubt
attaches to things recalled as facts, while a formula that "obviously" looks like the
others gets transcribed without a second look.

Review the unflagged parts at least as carefully as the flagged ones. A reviewer who
concentrates on what the author was worried about is checking the author's model of
their own errors, not their errors.

## Instrument the input, not only the outcome

The first torque bracket returned 0.00% target-mode occupancy at every level. That is
a publishable-looking null: "torque does not excite these modes." It was wrong — the
trigger time was initialised to infinity unless the FORCE magnitude was positive, so
with force at zero and torque at 80 N·m nothing ever fired. I had updated the trigger
condition when adding torque and not the initialiser.

What caught it was a `|T| max` column included to check the channel rather than the
physics, reading 0.0 where it should have read 80. **A dead channel and a real null
produce the same outcome table**; only measuring the input distinguishes them. Log
what you applied, not just what happened.

Related: a partial fix reads exactly like a working one. When adding a parameter to an
existing mechanism, grep for every place the old parameter is tested — the trigger,
the initialiser, the guard, the summary — because the one you miss will fail silently
in the direction of doing nothing.

## A caveat you can measure is not a caveat

I reported that torque enriched three lateral contact modes but left FRONT worse than
the existing data, and listed "the probe was backward-only, so the comparison is not
like-for-like" as a caveat. It was not a caveat, it was an untested hypothesis with a
ten-minute experiment attached: FRONT means the rear feet are up, which is a nose-down
attitude, and walking direction plausibly biases which pitch unloads feet. Re-run
forward, FRONT went 0.30% to 1.15% and overtook the baseline.

Stating a limitation honestly is worth much less than testing it when the test is
cheap. Before writing "this may be an artifact of X", ask what it would cost to vary
X — and if the answer is minutes, the sentence should be a measurement instead.

## "It does not occur" and "we failed to collect it" need different responses

The Go2 REAR contact mode sat at 0.06% of transitions and looked like a coverage
deficit. Two mechanisms were tested: trunk torque reached 1.7x, and a gravity tilt
reached 50-150x — but every tilted episode fell within 0.9 s, so that enrichment was
the collapse trajectory. Bracketing the walkable range showed the gait breaks between
3° and 5°, and at 3° the mode is rarer than on flat ground. There is no setting at
which the robot both walks and rears.

So the mode is structurally absent for this policy on this plant, and 1,991 frames is
what the system produces rather than what collection missed. No episode count reaches
it, and a model that never predicts it is correct rather than deficient.

Before budgeting collection to close a coverage gap, test whether the configuration is
reachable at all. The distinction is invisible in a histogram — both cases look like a
small number — and only an attempt to produce the state on purpose separates them.

## Check WHEN a labelled event happened before treating the label as a population

428 of 1,762 Go2 episodes carry `fell: True`, and that was taken as 428 episodes of
falling-robot data. The median `fell_at_s` is 1.39 s against a recording start of
1.25-4.25 s: 94% of them collapsed during the stand-up ramp, before or barely after
recording began. They are failures to STAND, not falls while walking, and a robot that
collapsed during the ramp then lies still for the whole episode — which is why a peer
measured them as 89% airborne-or-collapsed and concluded falls contribute no contact
diversity. Both observations are the same fact seen from different sides.

Genuine loss-of-balance episodes numbered ten. A boolean flag says an event occurred,
not when, not whether the recording covers it, and not whether the label means the
same thing across the population it names.

## "n=10 of 428" and "n=10, which is the whole population" are different caveats

Reporting a 10.3x effect measured on 10 of 428 fall episodes invites the reading that
it is a 2% subsample and therefore a selection artifact. It is not: only 12 episodes
fall late enough to have any pre-fall history, so ten is very nearly the complete set
of genuine locomotion falls. The honest statement is "small in absolute terms, not a
biased slice", and it is much weaker than the fraction sounds.

When a filter removes most of a population, say WHY before quoting the survivors —
the exclusion reason determines whether the remainder is a sample or a census.

## Independent ranges compose into a magnitude nobody specified

Ground tilt was drawn as roll ~ U(−3,3) and pitch ~ U(−3,3), independently, and each
looked modest. The robot experiences the COMBINED tilt, which reaches 4.24° — inside
the 3–5° band where the gait was separately measured to collapse. A quarter of the
collection failed to stand as a result.

Nobody chose 4.24°; it is what two reasonable-looking ±3° ranges produce together.
When several axes of a disturbance are sampled independently, state the distribution
of the resulting MAGNITUDE, because that is the quantity the system responds to and
it is not what any single range says.

The corollary is that the fix is not always symmetric: here pitch drove falls
(corr +0.427) and roll did not (+0.057), so capping the combined magnitude — the
obvious response — would have sacrificed roll diversity for nothing. Decompose before
constraining.

## A "floor" is only a floor until you vary the thing you were not varying

An 11.9% episode-loss rate was measured to be flat against perturbation magnitude and
was therefore recorded — by me, repeatedly — as an irreducible solver-divergence
floor. It was flat against perturbation and steeply monotone against ground pitch:
0.6% in the lowest band, 22.3% in the highest. Capping pitch cut it to 4.2%.

Establishing that X does not cause a residual is not evidence the residual is
irreducible; it only removes X. The word "floor" asserts something much stronger than
the measurement supports, and once written it stops anyone looking — including its
author, who quoted it four times before testing a second variable against it.

## A derived channel inherits the assumptions of its derivation

Gravity-direction channels were reconstructed post-hoc from each episode's
quaternion. That computes the world-z axis in body frame, which equals the gravity
direction only when gravity points along world-z. These episodes apply terrain slope
by ROTATING GRAVITY on flat ground, so on every tilted episode the derived channel is
wrong by exactly the tilt — and wrong in the specific direction of asserting the
ground is level, which is the least likely error to be noticed.

A quantity the simulator SET should be logged, not reconstructed. Reconstruction
silently imports whatever the reconstructor assumed, and the assumption is invisible
in the resulting column, which looks like a measurement.

## Keep third-party source on disk

Two defects in this project were caught by checking a constant against upstream while
the source happened to be open for an unrelated review: a squared reward term that
should have been an absolute value, and a hip default of 0.0 where upstream has ±0.1.
Neither check was diligence; both cost one command because the source was local.

"Check your constants against upstream" is an exhortation nobody acts on. "Keep the
source on disk" is a one-time action that makes the check cost nothing, and the checks
then happen as a side effect of working nearby.

## Named quantities need an expiry note saying what was and was not varied

"11.9% solver-divergence floor" was measured against one variable, perturbation
magnitude, found flat, and named. The name then substituted for the measurement: it
was quoted four times without anyone testing a second variable, and the tilt
correlation that halved it was found only because someone was chasing an unrelated
question. Nothing about the word "floor" invites revisiting it.

The remedy is not vigilance about named quantities — vigilance does not scale and the
name is precisely what suppresses it. Record, beside any number given a name, WHAT
WAS VARIED WHEN IT WAS MEASURED. "11.9%, flat against perturbation 0–120 N, no other
variable tested" carries its own expiry; "floor" does not, and reads as a property of
the system rather than of one sweep.

## A placeholder that prints a plausible number is more dangerous than one that crashes

**Cost:** caught by an unrelated zero, one step before publication · **Found:** 2026-09-04 · **Applies to:** any scaffolded computation left in a reporting path

Building a per-episode "irreducible floor" column, the growth term was stubbed:

```python
floor = seed_sep * math.exp(rate * 0.0)      # exponent is zero
```

That is the seed separation multiplied by one, printed in a column headed **FLOOR**
alongside real measurements. It does not crash, does not warn, and produces numbers
of the right order of magnitude. **A reader — including its author an hour later —
has nothing to distinguish it from a computed result.**

**A stub that crashes is self-limiting; a stub that prints is load-bearing the moment
someone reads the column.** The failure mode is not writing the stub, which is normal
while building. It is that nothing about the output says it is one.

What actually exposed it was **unrelated**: a *different* bug made the seed separation
exactly `0.00e+00` on all twelve episodes, which was implausible enough to investigate.
Had the seed been correct, the FLOOR column would have carried plausible-looking
garbage into a report. **The check that caught it was luck, not design.**

**Practical rules:**

- **Make an unfinished computation fail, not evaluate.** `raise NotImplementedError`
  where the term belongs. If a partial result must be shown, name the column for what
  it is (`seed_sep_no_growth_model`), never for what it is meant to become.
- **Beware the multiply-by-one and the add-zero.** `exp(x * 0.0)`, `* 1.0`, `+ 0.0`
  and `if False:` all read as arithmetic and behave as deletions.
- **Report the stub rather than quietly fixing it**, if it reached anyone. The
  correction is cheap and the record of what was believed is not — someone may have
  already acted on the number.

Related: [when a comparison goes wrong, suspect the apparatus](#when-a-comparison-goes-wrong-suspect-the-apparatus-before-the-subject),
and the same session's rule that a check with no reachable failure is not a check. This
is that rule applied to a *value* rather than a test: **a number that cannot be wrong
is not a measurement.**


## Check a summary statistic against the quantities it came from

**Cost:** caught in one message, twice · **Found:** 2026-09-04 · **Applies to:** any derived figure quoted beside its inputs

A paired-design analysis reported **"81% variance reduction from pairing"** next to its
own inputs:

```
  sd baseline                 0.0511
  sd of the paired difference 0.0467
```

**A 9% reduction in sd cannot be an 81% reduction in variance.** The two numbers sat
one line apart and contradicted each other. The formula was
`1 - var(d)/(var_b + var_t)`, which is not the comparison a paired design makes; the
right one is the paired difference against the unpaired difference,
`sd(d)` against `sqrt(var_b + var_t)`, which gives **56%**.

**What caught it was not the value looking wrong** — 81% is entirely plausible for
pairing — **but its relation to the neighbours it was printed beside.** An isolated
summary statistic has no scale; the same number next to its inputs does.

**The practical form:** when reporting a derived figure, print the quantities it was
computed from beside it, and check the arithmetic in your head before sending. Ratios,
percentage reductions and normalised errors are where this bites, because they discard
the scale that would otherwise make an error visible.

**Same session, same shape, different direction:** an "improvement" of 4x in
`rollout_sel` sat beside `xy_rmse` values of 1.9045 m and 2.0374 m — the raw errors
said the model was slightly *worse*, and the ratio said four times better, because the
denominators differed by 3.7x. Printing raw and derived together is what made both
visible. See [normalisation hides the units it divided by](#normalisation-hides-the-units-it-divided-by).

## The same selection rule, run on two inputs, is not the same selection

**Cost:** three headline numbers, one launched experiment, and the justification for that
experiment's design · **Found:** 2026-09-05 · **Applies to:** any A/B between two models
whose state definitions differ

**Expected:** a scoring routine applied identically to two checkpoints compares them on
the same quantity. **Happened:** the action-sensitivity gate reported correlation
**0.876** at 0.5 s for the contact-conditioned model against **0.181** unconditioned, and
a **+0.793** transition-split effect. On a matched channel set the real numbers are
**0.310** and **+0.245** — the first is a gate *failure*, not a pass. **Cause:** one line.

```python
"body_vel": [i for i, f in enumerate(sf) if f.startswith("vel_body")]
```

`sf` is each model's *own* state-field list. The 34-channel model matched **two**
channels; the 40-channel model matched **three**. Both sides then computed a correct
score over different physical quantities. Every internal consistency check passed,
because nothing was internally inconsistent.

**Fix:** name channels explicitly, never select a scored family by prefix, and when two
artifacts are compared assert the selected index sets are **equal** — not that the same
rule ran on both.

```python
assert sel_a == sel_b, (sel_a, sel_b)   # the one line that would have caught it
```

**Why six other catches in the same session did not reach it.** Every one of them checked
a **value**: a threshold, a baseline, a scope, a premise, an arithmetic step. This defect
sat in the **selection**, upstream of every value, and it emitted values that were each
individually correct. Value-checking cannot find it, by construction.

**The second-order cost is larger than the first, and is the real lesson.** The retracted
0.876 was the *stated justification* for running the next fine-tune at 25-step branches.
Corrected, 0.5 s is a horizon that model **fails**; it passes only at 0.1 s. So that run
was not a pre-registered risk that materialised — it was **never licensed**, and its
0-of-43 failure is evidence about neither variable it changed. Distinguish these:

| | what it tells you |
|---|---|
| the risk materialised | the design was sound, the bet lost |
| the justification was void | **nothing**, and the pre-registration is not a defence |

A wrong number does not merely mislead a conclusion. It silently **authorises work**, and
that authorisation survives the number's retraction unless someone goes looking for it.

**The strongest available check is not a check.** Have a second party reimplement the
comparison from your written definition. Stating it precisely enough for someone else to
run is a stronger test than any amount of agreeing that you mean the same thing.

Related: [suspect the apparatus before the subject](#when-a-comparison-goes-wrong-suspect-the-apparatus-before-the-subject),
and [a commit hash records where HEAD WAS](#a-commit-hash-records-where-head-was-not-what-ran).

## A paper's headline number often does not cover the system you want it for

**Cost:** a case-study reframing built on a claim that was not established · **Found:** 2026-09-05
· **Applies to:** any decision made from a related-work claim

Three papers were read in full alongside their released code on the same day, to decide
what to build next. **All three disagreed with their own code in ways that changed the
design**, and in two of the three the headline result did not cover the system it was
being cited for.

| paper | what it is cited for | what it actually shows |
|---|---|---|
| NeRD | "1000-step stability on a quadruped" | 1000 steps is **Cartpole**, 2 DoF, contact-free. For the quadruped there is **no open-loop state error at any horizon** — only closed-loop agreement on a **saturating** reward |
| DHAL | "K~3 modes is optimal" | K=1 to K>=2 is real (2.5x); **K=2 vs 3 vs 4 is within seed noise**, 3 seeds, no numeric table |
| HALO | "handles legged contact events" | Uses **one foot**; the other's events are computed and discarded, code comment `# just choose the default first one for now` |

**Three checks, in the order that pays.**

1. **Which system carries the headline number?** A paper reporting six robots may have run
   its flagship metric on the easiest one. Find the table, not the abstract.
2. **Does the metric saturate where the result sits?** NeRD's ANYmal agreement is
   `exp(-((v_x-1)^2 + v_z^2))`, first-order insensitive at the peak a trained policy
   occupies. **The tell was in the same table:** its one non-saturating reward,
   `R = w_y + p_up`, shows **+17.21%** against **-0.02%** for the saturating ones. A
   metric that cannot move is not evidence that nothing moved.
3. **Read the loss, not the paragraph describing the loss.** DHAL credits a term with
   preventing mode collapse. It is *per-sample* entropy, minimized, and the collapsed
   solution attains its **global minimum**. The paragraph and the equation say opposite
   things and the equation is what ran.

**The generalisable form:** a related-work claim is a *measurement made on some system
under some metric*, and citing it is an implicit assertion that both transfer. State the
system and the metric when you cite it, the way a scope is stated with a result
([above](#state-the-scope-with-the-result-or-the-conclusion-inherits-one-it-never-had)),
and the mismatch becomes visible at the moment of citing rather than after the plan is
built on it.

**Cheapest sufficient check:** clone the code and grep for the mechanism. Noise injection
in NeRD is dead code with no call site; the multi-leg selection in HALO is one line with a
`for now` comment. Neither is discoverable from the PDF, both took minutes, and each one
would have changed a design decision on its own.
## A bound that is declared, logged, and never enforced

**Cost:** two full preprocess-and-train cycles · **Found:** 2026-09-05 · **Applies to:** any filter with a "too small to bother with" branch

A physical-admissibility filter declared its bounds up front, in the docstring, before
measuring what they would cut — the right order. It then failed to enforce them:

```
  declared joint-position bound   |q| <= 2.09 rad
  worst value in the "filtered"   158.15 rad
  samples outside the bound       45,057   (0.4999%)
```

The cause was a branch meant to be conservative:

```python
last = int(np.argmax(~ok))     # first inadmissible frame
if last < 50:
    continue                   # "too short to trim; a gate will drop it"
```

**No gate did.** An episode whose first violation came before frame 50 was skipped by
the trimmer entirely, so it kept **100% of its bad frames** — the exact opposite of the
declared criterion, and worse than having no filter, because the filter's existence was
what stopped anyone looking. The reasoning "something downstream handles this" named no
specific downstream check, and that is the tell.

**The bug was invisible in the filter's own output.** It reported `episodes trimmed 0,
frames removed 0` and looked like a clean no-op. The dropped-episode counter existed but
was never printed, so the one number that would have exposed it — 500 — was computed and
discarded. **A branch that takes an action needs a line in the report, or it is not
observable.** Every summary line should account for every path through the loop.

**How it was caught:** not by re-reading the filter, but by checking the declared bound
against the data it had supposedly filtered. `max |q|` should have been <= 2.09 by
construction. It was 158.15. **A postcondition that restates the declared guarantee is
worth more than any amount of re-reading the code that is supposed to provide it.**

## When the data falsifies the rationale but vindicates the threshold

**Cost:** none, caught before it propagated · **Found:** 2026-09-05 · **Applies to:** any filter whose docstring explains what it will cut

The same filter's docstring argued: diverged episodes are the integrator coming apart in
the *final* frames, the recovery behaviour worth keeping lies *before* the blow-up, so
trimming the tail preserves everything valuable. Measured over all 3,503 episodes:

```
  violating episodes                                  500
  episodes with a good prefix >= 50 frames            0
  episodes 100% inadmissible                          344 of 500
  median inadmissible fraction                        1.000
  episodes under 10% bad                              1 of 500
  status                                              diverged 499, fell 1
```

**Not one episode had the shape the rationale described.** There was no good prefix to
preserve anywhere in the population. The operation was never a trim; it was an exclusion,
and the docstring's story about the mechanism was simply wrong.

**The threshold was right and the reason for it was wrong, and those are independent.**
The bounds came from the URDF and from physics rather than from the data, so they
survived their rationale being falsified. Had they been tuned to produce a pleasing cut,
there would have been nothing left standing.

**The handling that matters:** the falsified premise was recorded in the script, with the
numbers that killed it, rather than being quietly rewritten to match the outcome. A
docstring silently edited to agree with its results is indistinguishable from one that
was right all along, and destroys the only evidence that a prediction was ever made.

**On the goalpost question.** The acceptance rule `r = std/((p99-p1)/4.65)`, declared as
`median < 1.5 and max < 3.0`, first FAILED at median 1.84 / max 31.17, then PASSED at
median 0.856 / max 1.384 after the exclusion. "The test failed, we changed the data, the
test passed" is exactly the shape of moving the goalposts, and needs the distinction made
explicitly rather than assumed: **the first failure was the criterion correctly detecting
that its own bound had never been applied** — demonstrated independently by `max |state|`
being 158.15 against a 2.09 bound before, and exactly 60.18 against the declared 60.2
bound after. The criterion was not weakened, the population was not chosen to pass it,
and the bound was not adjusted. Fixing a bug in *applying* a criterion validates the
criterion; changing the criterion because it failed does not. State which one it is.

## Do not infer a magnitude from a summary statistic when you can measure it

**Cost:** one retracted claim, caught in ~10 minutes · **Found:** 2026-09-05 · **Applies to:** any "therefore it must be" about numerical scale

From an action-normalisation `std` of `2.5e+32`, the inference was drawn that real
actions of order 1 rad "reach the network as numerically meaningless input" — plausible
arithmetic, and it was reported to a peer as a finding. Measuring it gave something else:

```
                        old cache      corrected cache
  raw |action| max      3.7e+33        3.77
  normalised |z| max    14.8           13.69
  normalised |z| MEDIAN 0.0132         0.5854
```

**Nothing was zeroed.** The true effect was ~44x attenuation of the action channel while
the state channels were normalised correctly — severe, badly conditioned, but not
destroyed. The inference had missed that the outlier actions inflating the `std` were
*also present in the raw data*, so both numerator and denominator moved.

**The difference changed a downstream decision**, which is the only reason it was worth
retracting: "meaningless inputs" implies prior results are void and should be discarded,
while "44x attenuated" implies they are degraded, retained as a record, and re-run. The
peer had already been told the stronger version and had amplified it to "void, discard".

**A second error rode along with the first, and it is the more general one.** The
retraction was accompanied by the reasoning *"a weak model that still showed the effect
is weak evidence for it rather than no evidence."* **That holds only when the artifact
pushes AGAINST the effect.** Here it pushes with it. Attenuating the action channel makes
the model lean on state history instead, which degrades SHORT-horizon prediction most —
that is where the action's influence is most direct — and matters least at long horizon:

```
  observed   wide model worse short-horizon, better long-horizon
  artifact   predicts worse short-horizon, little effect long-horizon
```

The artifact is a **sufficient explanation for the observation**, so the observation does
not discriminate between the substantive claim and the defect. It is **uninformative**
about the claim, not weak support for it. "Survived despite a handicap" is only an
argument once you have checked which way the handicap points, and that check is one line
of reasoning that is very easy to skip because the phrase sounds rigorous on its own.

**The rule:** when the data is on disk and the check is a few lines, measuring costs
minutes and inference costs a retraction. Reserve inference for what cannot be measured.
An arithmetic argument about scale is a *hypothesis about the data*, and should be
labelled as one until the data has been asked.

## Measure what the apparatus does before trusting what it measures

**Cost:** none — caught before the run · **Found:** 2026-09-05 · **Applies to:** any two-arm or paired comparison

A gate was built to ask whether a surrogate transmits action influence the way the
simulator does. Two arms, identical except for a perturbed policy; compare how far
apart they end up. Pre-registered thresholds, an apparatus check, a declared
INCOMPLETE branch. The design was sound and the measurement would have been wrong.

**Before running it, the arms were measured on their own.** The between-arm difference
in body velocity was flat across every horizon:

```
  0.05s 0.068   0.1s 0.064   0.2s 0.039   0.5s 0.065   1.0s 0.050   2.0s 0.074
```

First reading: saturation, the arms have decorrelated. **A control rejected that** —
matched arms differ by only ~9% of the difference between two *unrelated* episodes, so
they are nowhere near decorrelated.

**The real cause was structural.** A weight-perturbed policy differs from step 0, so by
the time the model's 128-step history window has filled, the two arms have been
diverging for 1.28 s. At the first row of the comparison window they already differed
by 0.068 m/s, while the model's two arms start at exactly 0 **by construction**. The
gain would have read near zero at short horizons for pure bookkeeping reasons.

**And the artifact pointed at the conclusion.** Near-zero gain reads as "the surrogate
is action-blind" — the exact FAIL the gate exists to detect. A pre-registered rule, a
confident negative, and an artifact that predicts it: unarguable after the fact. See
[when a comparison goes wrong, suspect the apparatus before the subject] and the
artifact-direction rule under
[do not infer a magnitude from a summary statistic](#do-not-infer-a-magnitude-from-a-summary-statistic-when-you-can-measure-it).

**The fix was a better experiment, not a correction factor:** branch the policy
mid-episode, so both arms are bit-identical until a known instant and differ only
after. Both sides then start from a difference of exactly zero at the same moment. The
branch is per-episode, at each episode's own recorded time at the window start —
row 0 is not t=0 and prewalk varies, so a constant would not have aligned. The swap
replaces **weights only**: replacing the policy object would reset its 5-step
observation history and inject a discontinuity unrelated to the change under study.

**Two further defects surfaced in the same pre-run pass, both in the checking code:**

- The self-test required the arms to be identical before the branch to within `1e-9`.
  They agree to `3e-8`, because one arm's gravity channels were *recorded* and the
  other's are *derived* by the same formula in float. **The check would have rejected
  every pair and reported INCOMPLETE** — a correct-looking null produced by the
  tolerance, not the data.
- The horizon filter was `if steps <= horizon`, with `steps` capped at the longest
  horizon. **The longest horizon therefore always reported n=0**, an empty row in a
  table whose other rows looked fine.

**The pattern across all three: none would have crashed, and each would have produced a
plausible number or a plausible absence.** A smoke run on a deliberately meaningless
input — an undertrained checkpoint, three episodes — found two of them, because the
point was to exercise the paths rather than to get an answer. **Run the apparatus on
something whose answer you do not care about, before running it on something you do.**

## Bit-reproducibility is a property of a machine, not of a dataset

**Cost:** one wrong claim to a collaborator, caught same session · **Found:** 2026-09-05 · **Applies to:** any replay-based or paired design over pooled data

A merged dataset was verified to replay bit-identically: five episodes, five command
families, all 164 physics columns identical on every row, from a spec RECONSTRUCTED
rather than read back. That is a strong check and it was reported as "this half
replays exactly."

**The five episodes were homogeneous in the one variable that turned out to matter.**
All five came from seed offset 2,000,000. The set also holds 1,510 episodes at offset
3,000,000 collected on a *different machine*, and those do not replay here at all:

```
  joint_rr_hip_target_rad   recorded -0.15132537   replayed -0.15141966   ~6e-5 relative
```

147 columns differ from row 0. The separation is clean rather than marginal: **14/14
from one machine differed, 6/6 from the other matched.**

**What caught it was a control built for a different question.** The gate's branch
self-test requires two arms to be bit-identical before a mid-episode branch — written
to detect a failed branch. Because both arms run the baseline checkpoint before the
branch, that test *is* a replay check, and it ran on 20 episodes rather than 5. It
partitioned them perfectly by collecting machine. A control designed for one question
answering a different one is worth engineering for deliberately.

**The consequences are asymmetric and both halves need saying:**

- **Training is unaffected.** The offsets are disjoint, so no spec appears twice and
  there are no contradictory samples; the model learns across a mixture of two very
  slightly different plants. Reporting only the alarming half would have caused a
  pointless re-collection.
- **Every replay-based design is constrained.** A paired experiment that re-runs one
  arm must run it on the machine that collected its baseline, or the arms differ by
  machine as well as by treatment — a second uncontrolled difference in a design whose
  whole claim is that only one thing changed. The fix is a stratified paired design,
  each machine treating its own episodes, **reported per machine as well as pooled** so
  a machine-by-treatment interaction stays visible.

**A 6e-5 relative difference is invisible at the step level and decisive over 40 s of
a chaotic plant.** The general form: "reproducible" without naming the machine, build
tree and library versions is an incomplete claim, and pooling data from two boxes
silently converts a reproducibility property into a per-row attribute nobody tracks.
See [an error identical across independent units is a shared constant] for the
inverse case, where uniformity was the signal.

## Two measurability questions, and catching one is a reason nobody asks the other

**Cost:** a verdict asserted on evidence that could not support it · **Found:** 2026-09-05 · **Applies to:** every threshold test

A gate tested three conditions against fixed thresholds. It carried a deliberate
**apparatus veto**: refuse to report a ratio when the instrument's own error exceeds the
signal it is measuring. That veto worked — it caught a horizon where the surrogate's
rollout error was 4.6x the between-arm signal, and returned INCOMPLETE instead of a
number.

**It asked the wrong measurability question of the deciding condition.** There are two:

```
  APPARATUS MEASURABLE     is the signal above the instrument's own noise      CHECKED
  STATISTIC DETERMINABLE   is n enough to separate observed from threshold     NOT ASKED
```

The verdict was driven by `corr >= 0.5`, a Pearson correlation, measured on **16
episodes**:

| | r | n | 95% CI | |
|---|---|---|---|---|
| as run | 0.143 | 16 | **[−0.380, +0.596]** | cannot reject 0.5 |
| re-run | 0.181 | 200 | [+0.043, +0.312] | rejects 0.5 |

**The observed 0.143 was not distinguishable from a passing 0.5.** A FAIL was reported
anyway. The conclusion later turned out to be right — but it was not established, and
being right is not the same as having measured.

**The pairing is the lesson, and so is the psychology.** Having built a visible guard
against one class of unmeasurability made the whole category feel handled. The veto's
existence was itself the reason nobody looked for the second question. **A guard against
one failure mode is not evidence about its neighbours, and is actively misleading about
them.**

**A correlation needs far more samples than a median does**, and the same n was used for
both without asking. In this gate: `gain` and `cosine` are medians and were adequately
determined at n=16 in one case; `corr` needed **n ≥ 60** to discriminate against its own
threshold. One sample size was chosen for three statistics with different requirements.

**The fix generalises:** decide every condition by its **interval**, not its point
estimate — PASS when the whole interval is inside the passing region, FAIL when it is
wholly outside, INDETERMINATE when it straddles the threshold. That makes "we could not
tell" a first-class outcome for each condition separately, rather than a property of the
run as a whole.

**And it repaid the scrutiny in both directions.** Re-measuring at n=200 confirmed the
FAIL *and* confirmed that `gain` had genuinely moved inside its band — a claim that had
been quoted as good news while resting on the same 16 samples as the failure. Checking
only the condition you dislike is not checking.

## A process check that matches its own invocation is not a process check

**Cost:** none, caught before acting · **Found:** 2026-09-05 · **Applies to:** any check whose pattern can match the checker

A chained job ran preprocess, then training, then a gate, sequentially. Mid-run:

```
  pgrep -f "nedm.training.trainer"  -> RUNNING      correct
  pgrep -f "gate_go2_action"        -> RUNNING      IMPOSSIBLE, the chain is sequential
```

Both cannot be true. The gate pattern matched **the chain script's own text** —
`chain_contact.sh` contains the string `gate_go2_action` in the command it will later
run. `ps` showed only two processes: the shell and the trainer.

**The check returned a plausible affirmative for a reason unrelated to what it was asked.**
Had the contradiction not been obvious — a sequential chain cannot run two stages at once
— it would have been believed, and the natural next action was to hunt for a duplicate
process that did not exist.

**Same shape as the other apparatus failures from this session** and it belongs beside
them: a dead torque channel that a `|T|max` column exposed; a joint-limit filter whose
`< 50` branch made it silently a no-op; a correlation tested at n=16 against a threshold
its interval could not exclude. In each case the instrument answered confidently about
something other than the question.

**The concrete fix:** match on the interpreter and script path rather than a substring
(`pgrep -f "python.*gate_go2_action"`), or read `ps` and exclude the orchestrator by PID.
**The general one: when a check can match the thing doing the checking, it will
eventually.** A contradiction between two checks is the cheapest possible signal that one
of them is measuring itself.

## Agreeing on a metric is not agreeing on what it selects

**Cost:** the headline result of a session, retracted after publication · **Found:** 2026-09-05 · **Applies to:** any metric defined over "a family of channels"

Two surrogates were compared on the same declared metric: action-response correlation on
**body velocity**. Both sides agreed the metric, repeatedly and carefully, across many
messages. The gate selected the family like this:

```python
  "body_vel": [i for i, f in enumerate(sf) if f.startswith("vel_body")]
```

**A 34-channel state has two matching channels. The 40-channel state added
`vel_body_z_mps` and has three.** So the headline comparison — **0.876 against 0.181** —
was three channels against two, and the difference was mostly the extra channel.

Corrected to an explicit two-channel set, the same gate on the same episodes gives
**0.310 against 0.181**: still an improvement, but it **fails** the 0.5 threshold where
0.876 passed it comfortably. Downstream, a claim that the model was "the first to pass all
three conditions" and "decisively off the trade-off curve" was withdrawn, and a
2.5-hour fine-tune had already been launched on branch lengths those numbers justified.

**Why six earlier catches in the same session missed it.** Every one of them checked a
*value*: a dead torque channel, a filter that silently no-opped, a correlation tested at
an n its interval could not support, a pooled veto that could not protect the family it
existed for. **This defect was in the SELECTION, not the value.** Every number was
correctly computed from the channels it was given.

**"Both models score body velocity" was true, and meant different things on each side.**

**It surfaced only because a collaborator asked for the computation written out by name**
— which channels, in which order, normalised by what — in order to implement it
independently. Prose agreement on a metric had survived a dozen exchanges; the request to
enumerate broke it immediately.

**The rules that follow:**

- **Name the channels. Never select a scored family by prefix, substring, or regex.** A
  pattern silently adapts to whatever a schema later contains, and schemas grow.
- **When two artefacts are compared, assert the selected sets are equal** — not just that
  the same selection rule ran on both.
- **Ask a collaborator to reimplement from your written definition.** The exercise of
  stating it precisely enough for someone else to run is a stronger check than any amount
  of agreeing that you mean the same thing.

## A test that runs a function is not a test that runs it at scale

**Cost:** a 32.9 GiB allocation on the first real run, after the test suite passed ·
**Found:** 2026-09-05 · **Applies to:** anything validated on synthetic data first

`knn_pred` in the W0 harness built an `(n_te, n_tr, d)` distance tensor. The synthetic
discrimination test has **240** test events. The rigid dataset has **43,224**. The test
exercised the code path and could not exercise the size, so it passed, and the function
died the first time it saw real data.

**The generalisable half is not "test at scale". It is what a passing test does to
attention:**

> A test's existence transfers confidence out of proportion to what it covers. "There is
> a test for it" is exactly what stops anyone looking.

**Two questions that cost nothing and would have caught it:**

1. **What is the largest input this will see, and how does cost grow?** Not "does it
   work" &mdash; `O(n_te x n_tr)` against 43k is a different program from the same
   expression against 240.
2. **What does this test NOT cover?** Write it next to the test. A synthetic fixture is
   chosen for the property under test and is silently unrepresentative in every other
   dimension, size first among them.

**Related shape, same session.** The pre-registered gait-band filter shipped with a check
that reports R^2 on the pairs it *dropped*, precisely so it could be caught doing nothing.
It fired on the first real run: dropped pairs scored 0.008 against kept pairs at -0.073,
so the band was laundering rather than cleaning, and the declared fallback to the
unfiltered number applied. **A check written so it can fail visibly caught its own
subject; a test written so it could only pass did not.** The difference is not rigour, it
is whether the failing outcome was made reachable
([above](#before-running-a-verification-step-state-what-result-would-constitute-failure)).


## A missing half can present as a shorter table

Pooling two machines' summaries printed the survivorship line for one stratum only,
because the two emitters named those keys differently. There was no error and no
warning — the table was simply one row shorter, which reads as "that stratum had
nothing to report" rather than "this code could not find it". Sixth specification
mismatch in this project, and the first to present as an absence rather than a
disagreement.

When merging records from independent producers, assert that every input contributed
to every output section. A section that silently accepts fewer inputs than it was
given is a check whose failure looks like data.

## Put the scope beside the verdict, not in a limitations section

The Go2 fine-tune result is "FAIL, both rules, n = 36" — for the backward-low command
cell on rigid terrain only. The sentence that will be quoted is "fine-tuning made it
worse", and it is true only with the scope attached. A limitations section at the end
is read by people who already have the conclusion; a scope sentence adjacent to the
verdict is read by everyone who reads the verdict.

## Diagnostics reported beside a pre-registered verdict undo the pre-registration

The pre-fall test declares one scoring family, one ratio and one bar before the
models exist. Reporting other shared families "as diagnostics" alongside the verdict
was offered and declined: with five verdict branches available, per-family numbers let
a reader take the verdict from whichever family looks best, and the declaration stops
constraining anything. The freedom pre-registration removes is exactly the freedom a
diagnostic table hands back.

If a second quantity is worth scoring, declare it now and say how the two combine.
"Reported for information" is not a neutral act when a verdict is in the same output.

## An abort beats a silent fall-back to a smaller set

Scoring fields are resolved by name in each model's own `state_fields`, and a model
missing any of them aborts naming the missing ones rather than scoring on those it
has. The fall-back is the tempting behaviour — it always produces a number — and it
produces one for a different quantity than the one declared, with nothing in the
output saying so. Four positional-ordering traps in this project argue for the same
handling anywhere a field set is assumed.

## One arm's performance across conditions does not test a two-arm claim

A conditioned surrogate scored corr 0.859 on windows containing a contact transition
and 0.934 on windows without, and that was read as falsifying the claim that
conditioning works by handling discontinuities. It does not. The claim is about the
DIFFERENCE between the conditioned and unconditioned models in each window; the
measurement is one model's absolute performance across windows. A model worse at
transitions is expected whether or not conditioning helped there, because transition
windows are harder — which is what the mechanism asserts.

The two worlds are numerically indistinguishable from the reported split: if the
unconditioned model scored 0.10 and 0.90 in the same windows, conditioning improved
transitions by 0.76 and the mechanism is strongly supported, with the conditioned
model still worse at transitions.

Whenever a claim is comparative, check the measurement has both arms in every cell.
The single-arm version is cheaper to compute and answers a different question, and
the difference is invisible in the numbers themselves.

## A pre-registered metric can be invalidated by facts about the system, not the answer

The pre-fall test declared absolute open-loop error as its metric. A peer then
reported that the conditioned model is ~2x worse at absolute prediction while far
better at action-response correlation — it became more accurate about how state
responds to actions and less about the state itself. The declared metric therefore
measures the axis the treatment regressed on.

That is a legitimate reason to amend a declaration and a dangerous one to act on
casually, because "the metric was wrong" is also what someone says after seeing a
result they dislike. The distinguishing test is WHAT the new information is about: a
fact about the system's behaviour, learned independently of the outcome, can justify
an amendment; a fact about the outcome cannot.

Either way, amend explicitly and before the result: state the old bar, the new bar,
and why the change was prompted. A silent substitution is indistinguishable from
fitting the instrument to the answer, even when it is not.

## The conservative-looking option can be the one that defeats the purpose

Amending the pre-fall test's metric from absolute error to correlation broke its
statistics: a correlation has no per-episode value, so the median and order-statistic
CI had nothing to operate on. An alternative was available — compute a PER-EPISODE
correlation, which preserves the original machinery untouched and looks like
minimising the change.

It was rejected because a per-episode correlation is a different quantity from the
pooled-across-windows correlation the gate scores, and matching the gate is the entire
reason the amendment exists. **Preserving the old statistics would have quietly
reverted the amendment while appearing to be the careful choice.**

When an amendment forces a second change, check whether the option that minimises
disruption also undoes the first change. "Smallest diff" and "still measuring the
intended thing" are different objectives and they diverge exactly when an amendment
was necessary.

## A low false-positive count is not a calibration measurement

One false positive in 40 synthetic nulls has a Wilson 95% interval of 0.4% to 12.9%.
It rules out gross miscalibration and says almost nothing else — 2.5%, 5% and 10% are
all consistent with it. Reporting "calibrated, slightly conservative" claims a
precision forty draws cannot deliver, and the phrase survives being quoted without
its n in a way the raw count does not.

Say what the check rules out rather than what it appears to confirm.

## Two metrics can share a name, be agreed between machines, and need different apparatus

The pre-fall test was amended to score "action-response correlation", the same
quantity the surrogate gate scores, and both machines agreed the change. Only when the
exact computation was requested did it emerge that the gate's metric is a PAIRED
COUNTERFACTUAL — the plant run twice under two action sequences — while the pre-fall
instrument does open-loop prediction against recorded truth. The second cannot produce
the inputs the first requires.

The mismatch was not in the formula. It was in **what the formula needs to exist**,
which no amount of agreeing on the formula would have surfaced. Asking for the
computation rather than implementing something that resembled it is what found it, and
it found it before days were spent rather than after.

When adopting a metric from another instrument, ask what data it consumes, not only
how it is calculated.

## A derivation from two measurements is not a third measurement

"Conditioning helps before a fall" now rests on composing two direct results: pre-fall
windows are transition-rich, and conditioning helps at transitions. That is a sound
inference and it is weaker than either input, because it assumes pre-fall transitions
behave like transitions generally — which neither measurement tests, and which a fall
could violate by involving several feet changing at once or modes rare elsewhere.

Label composed conclusions as composed. The failure mode is that a derivation acquires
the confidence of its inputs while carrying an assumption neither of them checked.

## A guard keyed off a different array from the one it guards is not a guard

**Cost:** a run where every component failed and the summary said nothing was wrong ·
**Found:** 2026-09-05 · **Applies to:** any validity check that recomputes its own subject

W0's harness withholds its verdict when most of the state does not vary, because R^2 sits
near zero on a constant component however good the model is. The check was written as:

```python
n_deg = int((spread < a.min_spread).sum())        # spread = TRAINING spread
```

while the NaNs it was counting are produced elsewhere, through **two** paths:

```python
ok = Yte.std(0) >= min_spread                      # HELD-OUT spread
return np.where(ok & (ss_tot > 1e-12), ..., np.nan)  # ...or no variance to explain
```

At a 0.02 s horizon the increments are small enough that the held-out condition fired for
**all 35 components** while the training spread did not. The table printed 35 dashes, the
medians were NaN, and the summary line read **"0 of 35 components unscoreable"** with the
withhold guard silent. A reader taking that line at face value concludes the run was fine.

**Fix: count the thing itself, never a proxy for it.**

```python
n_deg = int(np.isnan(res[first_key]).sum())   # the NaNs actually returned
```

Now the count and the table cannot disagree, because they read the same array.

**The general rule.** A validity check that *re-derives* its subject is a second
implementation of the same logic, and the two drift. Two implementations agreeing proves
nothing when only one of them is wired to the output; two implementations *disagreeing* is
invisible unless something compares them. **Have the check read the artefact it is
checking.**

**And a related trap the same run exposed:** `np.nanmedian` over an all-NaN array returns
NaN and prints as a number-shaped blank rather than raising. An aggregate that cannot
distinguish "no data" from "a value" will report the first as the second.

This is the third member of a family already in this file: the
[check that can itself be silent](#the-check-you-add-to-catch-silent-failures-can-itself-be-silent),
and [running a check is not evidence that the check ran](#running-a-check-is-not-evidence-that-the-check-ran).
**All three are cases where the instrument failed quietly and the subject took the blame.**

## Five instances of one class: a number correct about its scope, wrong about its meaning

**Found:** 2026-09-05, across a single session · **Applies to:** every summary statistic and
every name for a change

Each of these was arithmetically correct and each was read as answering a question it did
not answer. They are collected because **one is a mistake, five is a class**, and the
defence is the same in every case.

| # | the number or name | what it actually covered | what it was taken to mean |
|---|---|---|---|
| 1 | `body_vel` prefix match | 2 channels for one model, 3 for the other | "the same quantity on both" |
| 2 | "the collector seeding fix" | two `sha256` commits already on origin | a third, uncommitted, different fix |
| 3 | `grep grav_world_x_mps2` in the collector | that literal is in `dataset.py` | "the wiring is absent" |
| 4 | `grav_body_z`, increment sd 0.0014 | one channel at R^2 -60.3 | a 40-channel mean of -0.981 |
| 5 | W0 median R^2 0.393 at 0.29 s | 24 joint columns at 0.458 | "the floor for the body family", which is **0.125** |

**Instance 5 is the sharpest** because the conclusion would have survived review: comparing
the surrogate's 0.723 against 0.393 looks like a careful like-for-like, and it understates
the surrogate's advantage **threefold** because the denominator is a median over
heterogeneous channels dominated by the easy ones.

**Three defences, in the order they pay:**

1. **Report the subgroup you will act on, not the aggregate.** An aggregate over
   heterogeneous channels is a summary of nothing in particular. If a decision turns on
   body velocity, the number quoted beside that decision must be body velocity.
2. **Search for the mechanism, never the label.** `git log -S "<a distinctive literal>"`,
   scoped to the file the mechanism actually lives in. A negative result is only as good
   as its scope; a positive one carries its own.
3. **Assert equality of the selected sets**, never that the same rule ran on both sides.

**The tell they share:** in every case the number was *defensible* if challenged, because
it was correct. **What was wrong was the sentence it was placed in.** So the check is not
"is this number right" but **"what would have to be true for this number to answer the
question I am asking of it"** — and that is a question about scope, which no amount of
recomputation reaches.

Related: [part-whole correlation](#part-whole-correlation-a-statistic-whose-value-is-fixed-by-its-own-construction),
[state the scope with the result](#state-the-scope-with-the-result-or-the-conclusion-inherits-one-it-never-had),
and [the same selection rule](#the-same-selection-rule-run-on-two-inputs-is-not-the-same-selection).

## A defect that inflates its own variance is weighted out of the loss that would catch it

**Cost:** 2159 discontinuities that survived every check we ran · **Found:** 2026-09-05

`pitch_rad` wraps: range +/-pi, 2159 per-step jumps above 1 rad, the largest exactly 2*pi.
The surrogate is a **delta** model, so each wrap is a +-2*pi target to be fitted as an
ordinary real.

**The self-concealing part is the mechanism worth remembering:**

```
  the wraps inflate that channel's target_std   0.41543   (roll, undamaged: 0.00270)
  the loss is computed on NORMALISED targets
  => residuals there are divided by 154x more than anywhere else
  => the one channel with a pathology is the channel the loss weights least
```

*(First written as 23x, from `state_std`. The loss is on normalised **targets**, so the
figure that matters is `target_std` and the correct factor is **154x** — the first version
understated the defect sevenfold, in the direction that made it look less serious.
Unwrapping the channel drops its `target_std` from 0.34640 to 0.00305, a measured **113x**
rise in effective loss weight.)*

**A defect large enough to dominate a raw-units metric can be invisible to a normalised
one, precisely because it is large.** Normalisation by an empirical std is a
*data-dependent* weighting, so a channel that misbehaves buys itself a smaller weight.
Every training curve looked healthy.

**Two checks that would have caught it, neither of which is a loss:**

1. **Bound every angle channel by its own geometry before training.** A Cardan ZYX middle
   angle is bounded to +-pi/2; this one spanned +-pi. That is a one-line assertion on the
   dataset, not a diagnostic to run later.
2. **Histogram per-step increments per channel and look at the tail, in RAW units.** A
   target of exactly 2*pi is not a large residual, it is a different kind of object, and
   it is visible instantly at the top of a sorted list.

**The general form:** any weighting derived from the data can be *bought* by the pathology
it is meant to expose. Whenever a loss weight, a normaliser, or a threshold is estimated
from the same data it polices, ask what a defect would do to it — and check the quantity
in units the defect cannot rescale.

Related: [normalisation hides the units it divided by](#normalisation-hides-the-units-it-divided-by).

## A magnitude test standing in for a property test — inside the guard written to catch it

**Cost:** an assertion that fired on nine healthy channels · **Found:** 2026-09-05

After finding that a circular channel was being fitted as a real, we added an assertion so
an undeclared circular channel would fail loudly. **Its first version tested every channel
for a per-step jump above pi** and fired immediately on **nine joint velocity channels, up
to 16.85 rad/s** — impact transients, where a rate can jump by any amount and it says
nothing whatever about circularity.

> **Reading "large jump" as "wrapped" is the same conflation the guard exists to catch,
> one level up.**

Narrowed to `_rad` and not `_radps`. **A guard is not exempt from the class of error it
guards against**, and writing one is the moment you are most primed to commit that error,
because the pattern is fresh and every large number looks like the one you just found.

**The general form:** a magnitude is evidence about a *property* only under an assumption
about what the channel is. `|dx| > pi` means "wrapped" for an angle and means "a hard
impact" for a rate. **Test the property where it is declared — units, channel role,
declared topology — never a magnitude that correlates with it.**

## R^2 is unusable where a pathology inflates its own denominator

**Found:** 2026-09-05 · **Applies to:** any channel with rare high-variance episodes

Same channel, three episode selections:

| subset | R^2 | RMSE | n |
|---|---|---|---|
| all episodes | **+0.419** | 1.6159 | 59 |
| non-wrapping only | **-323.4** | 0.3333 | 49 |
| wrapping only | +0.371 | 3.8551 | 10 |

**Removing the pathological episodes made R^2 dramatically worse** — they were carrying
enormous target variance and inflating `ss_tot`, the denominator that flattered the score.
So the statistic is unusable here at *any* episode selection, and no filtering rescues it.

**Use normalised RMSE as the decision quantity instead:** `RMSE / sd(ground-truth
increment)`, per channel. On the clean 49 episodes that reads **18x the channel's own
variation** — immediately interpretable, comparable across channels, bounded below by
zero, and with no denominator a pathology can inflate. Keep R^2 alongside where it
behaves; do not decide on it.

This is the same denominator failure as the `grav` group scoring **-0.680** while
`grav_body_z` had RMSE 0.00069 against sd 0.446, and as the `body` group scoring 0.203
while `vel_body_x` was 0.822. **Three group-level conclusions in one session rested on it.**

## Two importable builds, selected by an environment variable, producing plausible numbers

**Cost:** an evening, a false "replay is not reproducible" finding, and a false consequence
drawn from it · **Found:** 2026-09-06

```
  /home/kyle/chrono-build/bin/pychrono/_core.so     md5 d1d0bd0a   Sep 3
  envs/nedm/.../site-packages/pychrono/_core.so     md5 8e9e3865   May 7
```

Collection pins the local build by setting `PYTHONPATH`; an ad-hoc replay that did not
imported the conda one. **Different floating point, a difference present at the first
step, amplified ~5000x over 41 s.** With the matching build: **0.000e+00** across 164
columns and 3994 rows.

**The failure mode is that it does not fail.** A missing library produces an ImportError; a
*different* library produces numbers, and they look like physics. Every downstream check
&mdash; digests, column comparisons, divergence plots &mdash; then measures the build
difference and reports it as a property of the simulation.

**The precondition nobody stated:** the digest check is meaningful only for episodes
replayed against **the same binary**, not merely on the same machine. Same-machine was
assumed and is not sufficient, because one machine can carry two.

**Fix, which detects rather than diagnoses:** record the **md5 of the `pychrono`
`_core.so` actually imported** into the episode metadata, alongside the perturbation peak,
prewalk and ground tilt. A mismatch is then caught at comparison time instead of costing
someone an evening. **Same argument as recording values rather than seeds, applied to the
simulator itself:** a seed plus a code version is a promise the code version can be found.

**How it was actually caught, which is the transferable part.** Not by inspecting
environments, but by an arithmetic argument about the *shape* of the divergence: the row-0
difference was **4.6e-05** against a float64 round-trip floor of **~1e-16**, eleven orders
apart, while the episode's own growth implied a ~5 s e-folding time. Reaching 4.6e-05 from
rounding in the 1.4 s before recording would need an e-folding time near 56 ms, a
hundredfold faster than the same episode shows later. **One system does not have two
Lyapunov times a hundredfold apart, so the difference predated recording.** Chaos amplifies
a difference; it cannot create one.

## When your misuse of a tool gives a wrong answer, "the tool is broken" is the expensive conclusion

**Found:** 2026-09-06 · **Applies to:** every negative result about shared tooling

The replay above was run by calling `arm_cmd()` — **half of a two-part helper** — and
executing the result directly:

```python
  cmd = g.arm_cmd(s, g.BASE_CKPT, out)
  r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)   # no env=
```

`run_arm()` is the half that supplies the environment. The gate's own single
`subprocess.run` does pass `env=env_for(s)`, so **no gate number was ever affected** —
but the conclusion drawn was "the gate sets no PYTHONPATH, so its apparatus check is
contaminated." A shared tool was declared defective on the strength of a private misuse.

**Two rules:**

1. **Before reporting a tool as broken, run the tool.** Not a piece of it, not a
   reimplementation of what it does — the entry point, as documented. Half a helper is a
   different program.
2. **Attribute to your own use before attributing to the tool**, because the two produce
   identical symptoms and only one of them costs other people their time.

**And the timing is the part worth remembering.** That consequence was asserted *inside a
retraction*. **A claim made in the same breath as a correction inherits the correction's
credibility** — the reader has just watched you be scrupulous, so the next sentence is
weighed less. It is exactly when a new claim needs the most scrutiny and is most likely to
receive the least.

## A threshold specified before its reference is measured is unanswerable

**Cost:** three instructions that could not be followed as written · **Found:** 2026-09-06
· **Applies to:** every briefing that sets a numeric criterion

Three separate instructions were pushed back on in one session. They share one shape:
**each specifies a NUMBER on a quantity whose scale had not yet been measured.**

| the instruction | why it was unanswerable |
|---|---|
| *"collect until the confound diagnostic plateaus"* | the statistic's behaviour under sampling was unknown — it declines with `n` for **any** population, so there is no plateau |
| *"keep the three weak channels above ~0.6"* | the metric's ceiling was unknown. Walking data scores **0.812** against itself, so 0.6 was 0.74x of a maximum nobody had measured |
| *"2.4M informative against 10.5M confounded"* | the denominator's composition was unchecked — the new collection's own recovery rows are confounded too, making it 13.1M |

**None is a reasoning error.** Each is the right *kind* of criterion. Each was
unanswerable because the reference value did not exist when the threshold was set.

**The fix, and it costs one sentence per briefing:**

> When a briefing sets a numeric threshold, either **name the reference it is relative
> to**, or **ask for the reference to be measured first** and set the threshold as a
> fraction of it.

*"Measure what the reference scores against itself, then take a fraction of that"* would
have prevented two of the three; asking what the denominator is composed of prevents the
third.

**This is [measure the noise floor](#measure-the-noise-floor-before-you-compare-anything-to-it)
applied to SPECIFICATION rather than to analysis**, and that is why it recurred despite
the lesson already being in this file. The habit had been trained on interpreting results
and not on writing instructions — the same discipline, one step earlier in the process,
where it is cheaper and where nobody was looking for it.

## Separate the measurement from what you claim it means, and verify them separately

**Found:** 2026-09-06 · **Applies to:** every reported result

Reviewing a session's worth of errors turned up a shape neither party had noticed:

> Almost none were measurement errors. Nearly all were **labels and attributions attached
> to correct numbers.**

| the number | correct | the sentence around it |
|---|---|---|
| 145 columns differ | yes | *"chaos"* — was a build mismatch |
| 4.6e-05 at row 0 | yes | *"environment difference"* — asserted as a cause after elimination |
| the gate's source | read correctly | *"contaminated"* — never checked |
| `grav` group -0.680 | yes | *"the error is in the gravity channels"* — was one near-constant channel |

**A number gets checked because it looks checkable. The sentence wrapped around it does
not, because it does not look like a claim.** And both arrive in the same breath, at the
same confidence, from a party who has just demonstrated care by producing the measurement.

**Two rules:**

1. **State the measurement and the interpretation as separate claims**, and treat the
   second as unverified until something independent supports it.
2. **In briefings, mark which claims are verified and which are believed.** An instruction
   that states facts and intentions at identical confidence gives the reader no way to
   allocate checking effort. Three errors propagated this way in one session — an
   asserted CLI flag that did not exist, a spec item that had changed, and a rate read off
   a stale docstring and passed on as a measurement.

Related: [when a comparison goes wrong, suspect the apparatus](#when-a-comparison-goes-wrong-suspect-the-apparatus-before-the-subject),
and [a residual after elimination is a hypothesis, not a conclusion](#when-your-misuse-of-a-tool-gives-a-wrong-answer-the-tool-is-broken-is-the-expensive-conclusion).
## A distribution can be inside every marginal band and nowhere near the joint region

Two instances in one collection, and neither is the familiar "an aggregate hid a
subgroup" — here every marginal is CORRECT and the conclusion still fails.

**Action scale.** The range was derived from the policy's own commanded offsets,
|target − stand| at p99.9 = 1.53 rad. Independent draws at that marginal extreme
drove joints past the URDF limit in 30 of 30 windows, because the policy's twelve
offsets are strongly correlated and independent samples are not. A marginal range is
an envelope, not a description of what lives inside it.

**State containment.** Median per-channel containment against the walking band was
0.997–1.000 — every channel inside its own p1–p99 — while JOINT containment across
all 33 channels at once was 0.03, against a walking-vs-walking ceiling of 0.812. The
state was inside every one-dimensional band and outside the walking region 97% of the
time.

Report the joint statistic whenever the question is "is this the same regime". The
per-channel view answers a different question and answers it reassuringly.

## Measure the ceiling before quoting a score against a reference

Joint containment across 33 channels at p1–p99 cannot reach 1.0 even for the
reference population: held-out walking scores 0.812 against walking. Quoting an
excitation score of 0.03 without that makes it read as "3% of an absolute standard"
when it means "0.04x of what the reference itself achieves". A coordinator on the
other end of the message said they would probably have over-reacted to the bare
number. Compute what the reference scores against itself, and quote the ratio.

## A statistic whose value depends on sample size cannot be a stopping rule for sampling

"Collect until the confound diagnostic plateaus" has no stopping point when the
diagnostic itself moves with n. The conditional/unconditional action variance fell
0.776 -> 0.696 on excitation data as rows accumulated — which reads exactly like the
confound returning under volume. Running the same instrument on POLICY data, where
the answer was already known, showed 0.185 -> 0.085 over the same range: both decline,
because denser sampling puts kNN neighbours closer together and shrinks the local
variance. The metric was moving, not the data.

The stopping rule and the statistic move together and nothing in the number
distinguishes them. **The control is what separates them, and it is only available by
running the instrument on a population whose answer you already have.**

Salvage rather than discard: effective rank does not depend on n in the same way and
saturated at 11.93 by 2,000 rows, which is a valid plateau and a finding in itself —
the confound broke with almost no data. Conditional variance stays meaningful as a
comparison at MATCHED n, which is how the headline was computed. A metric can be
unusable for one purpose and sound for another.

## Apply the discipline one step earlier: to instructions, not only to results

"Measure the noise floor before comparing anything to it" had been in this file for
days when three separate briefings in one session set a numeric threshold on a
quantity whose scale had not been measured — a stopping rule on a statistic whose
behaviour under sampling was unknown, a 0.6 containment threshold against a ceiling
that turned out to be 0.812, and a mixture ratio whose denominator contained rows of
the same kind it was being contrasted with.

The habit had been trained on INTERPRETING results and not on WRITING instructions,
which is one step earlier, cheaper, and where nobody was looking. A rule that lives
only at the analysis stage will be violated at the specification stage by the same
person who wrote it.

Operationally: when setting a threshold, name the reference it is relative to, or ask
for the reference to be measured first and express the threshold as a fraction of it.

## Record why and when a unit was dropped, before dropping it

Asked where in the episode the excitation collector's rejections occurred, the answer
was not merely unlogged: it had never existed. A rejected episode hit `continue` before
any row was written, so there was no artefact to re-analyse. That fails differently
from data that was collected and discarded — no amount of reprocessing reaches it, and
nothing in the pipeline distinguishes "this quantity is missing" from "this quantity is
uninteresting." The instrumentation had to be built and the arms re-run before the
question could be asked at all.

What it cost to skip and what it bought to add: three lines appending a rejection record
(`window`, `reason`, `row`, `phase`, `burst`, `rows_since_onset`) turned an unanswerable
question into a decisive one on the first re-run — all 41 rejections during recovery,
median 3 rows after the policy handover, none during perturbation. That inverted the
hypothesis and changed the collection design.

The discarded units are where the mechanism lives. A filter is a measurement of the
thing being filtered, so instrument it like one.

Related, and the reason this is not merely bookkeeping: excluding failures is a
survivorship filter over exactly the states the failures identify. Arm B's 13.7%
rejection rate was first waved off as tolerable "because those episodes are excluded" —
but the exclusions were concentrated on the handover states the study is about, so the
survivors were biased toward easy handovers. Judge an exclusion rule by WHAT it selects
against, not by how many units it removes.

## A wrong number written to disk outranks a wrong number printed to a log

The excitation collector printed `first tip at row median 332 of 10` — nonsense, because
the denominator used the burst length where the episode length belonged. Fixing the
printed line surfaced the same substitution in `summary.json`, where `"rows"` was
`kept * WINDOW_ROWS`: arm A's summary claimed 36,000 rows for a file holding 306,000, an
8.5x undercount, in a persisted artefact meant to be read by someone else later.

The printed line was self-evidently absurd and would have been caught by any reader. The
JSON field was plausible, so it would not have been. Four existing summaries were
recounted from their CSVs and stamped with a `rows_field_repaired` note.

Both are now derived from `rows_written`, accumulated as rows are actually written,
rather than recomputed from a formula that can drift from what the loop does. Prefer
counting what happened over recomputing what should have happened.

## Killing a shell does not kill the children it launched with `&`

A waiter shell (`until grep -q DONE log; do sleep 30; done; cmd_a & cmd_b &`) was
killed to cancel a queued collection. It had already fired, and the two backgrounded
children were reparented rather than killed, so "I cancelled it" was false while a
process kept writing to disk. It was caught 26 s in only because the process list was
checked afterwards; the next person finds it as a dataset that mysteriously grows.

Kill the process that does the work, not the shell that launched it — and after any
cancellation, re-list the processes and confirm the specific PIDs are gone. A cancel is
not done when the kill returns; it is done when the thing it was cancelling is absent.

## Be more suspicious of numbers that look right than of numbers that look wrong

One substitution produced two wrong numbers. The printed one — `first tip at row median
332 of 10` — was self-evidently absurd and was caught in minutes. The one written to
`summary.json` was an 8.5x undercount that looked entirely plausible and would have been
copied into a report by someone with no reason to doubt it.

The obviously wrong number is nearly harmless: it defends itself by being unbelievable.
The plausible one has no such defence, and attention flows naturally to the first. The
instinct this argues for — inspecting the numbers that raise no alarm — is not natural
and has to be deliberate. When a bug is found in one output, check every other output
derived from the same expression, especially the ones that look fine.

## A version is what a build claims; a hash is what it is

**Cost:** none yet, caught before the 5M-row set landed · **Found:** 2026-09-06 ·
**Applies to:** any dependency resolved at import time rather than pinned

Two Chrono builds live on kyle-sbel: a conda `pychrono` inside the env, and a source
build reached only through `PYTHONPATH`. The source build shadows the conda one when
the path is set. Omit the path and the import still succeeds, the run completes, and
the output looks entirely normal.

Measured, same seed and arguments, only `PYTHONPATH` differing:

```
  numeric columns compared     183
  columns differing            115
  largest delta                5.23 N   (foot_fl_force_fz_n)
  runs completed               2 of 2
  errors raised                0
```

**The wrong engine does not fail. It exits 0.** Compare the replay investigation,
which cost an evening but announced itself with 145 differing columns; this has the
same magnitude of divergence and no symptom at all. The only difference is an
environment variable that is easy to omit and invisible once omitted.

**A version is what a build claims about itself; the md5 is what it is.** Recording a
version would not have caught this — both builds report Chrono 10, so a recorded
version would have matched across a 115-column difference and certified the wrong
thing. That is worse than recording nothing, because it manufactures confidence. The
collector now fingerprints `_core.so` at startup, prints it, writes it to
`summary.json`, and stamps a short tag on **every row**, because rows get pooled
across collections and a summary does not travel with them.

The general rule: when a dependency is resolved at import time from a mutable search
path, record what was actually loaded, not what was requested. `/proc/<pid>/maps` on
a live process tells you which shared object is really mapped, and is stronger
evidence than the environment variable that was supposed to select it.

And the corollary that made this urgent rather than interesting: **provenance you did
not record cannot be recovered once the process exits.** Four running shards were
verifiable from `/proc`; the completed run's launcher was gone, so its build is
attested by a process listing someone happened to read, and two older diagnostics are
simply unknown. The window for recording provenance is while the thing is running.


## A docstring cannot state a rate that belongs to its callers

Three of us independently "fixed" the same stale line in `robot.py`, which said the PD
loop runs at 400 Hz. dorm-pc and I both argued the same thing: a docstring restating a
derived quantity goes stale the moment the config moves. I then wrote a fresher number,
500 Hz, which was the same defect refreshed; dorm-pc's version, which refuses to state
the rate at all, was taken instead.

**All three of us had the diagnosis wrong.** The line was never stale. Reading the
collectors afterwards:

```
  collect_go2_excitation.py    exchange 2.5e-3 s  ->  400 Hz
  collect_go2_smoke.py         exchange 2.0e-3 s  ->  500 Hz
```

`apply_pd` is shared, and the rate is a property of **whoever calls it**. The original
400 Hz was correct for one caller and wrong for the other, simultaneously, from the day
it was written. There is no number that fixes it — not a fresher one, not a range, not
a note about which config it came from. My 500 Hz would have been wrong for the very
caller I was working in.

Staleness is a claim going out of date. This is a claim that was never a property of
the thing it was written on. The second is worse, because updating it feels like
maintenance and cannot converge.

Before writing a fact into a docstring, ask whether it is a property of this function or
of the code that calls it. If it varies by caller, the docstring's job is to say where
the value comes from — here, `exchange_mult * step_size_s`, recorded per episode as
`simulation.exchange_step_s` — and nothing else.
