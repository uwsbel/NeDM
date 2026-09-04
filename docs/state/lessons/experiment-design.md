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
