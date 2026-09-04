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

**The fix is to make the channel exogenous in the control cell:** supply it from
the permuted record as a context-style input at both training and rollout time, so
it is never predicted and never in the loss. Input width is then held exactly
fixed, which is the thing the control exists to hold, and only the output head
differs by one channel.

**Compare on `rollout_sel`, not on aggregate one-step loss.** The rollout metric is
pose-derived — `_integrate_pose` uses only `vel_body_x_mps`, `vel_body_y_mps` and
`yaw_rate_radps` — so a permuted non-pose channel affects it *only* through its
influence on predicting those three. That is exactly the causal path under test.
Report one-step loss restricted to the channels present in both cells, never the
aggregate.
