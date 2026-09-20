# Quadruped pipeline: rebuild plan

**Status:** proposed, not started. One decision still open (see the end).

## What this is designed against

Not a general cleanup. Every rule below exists because a specific failure already cost us
something, and the structure is chosen so the same failure cannot recur silently.

| failure that happened | rule it produces |
|---|---|
| A number reached a slide with no measurement behind it, twice | every artifact carries provenance; a value without one is not quotable |
| 83% of the reference corpus carries pushes the model cannot see | excitation admissibility rule, below |
| A metric changed meaning mid-study and invalidated comparisons across a date | metric definitions versioned with the data they measure |
| A flag was accepted and silently ignored | every knob asserts it took effect |
| Diverged episodes entered the corpus | failure detection is a collection-time gate, not a later cleanup |
| The action is nearly redundant given the state, so the model need not learn the derivative we differentiate | action excitation on by default, and a Jacobian gate |

## 1. The admissibility rule

**An excitation is admissible if and only if it enters through a quantity the model reads,
or through the initial condition.**

Anything else leaves an unexplained acceleration in the data. The model cannot predict it,
so it fits the average of "pushed" and "not pushed", and that average is wrong for both.
The alternative -- adding a channel for it -- produces an input that is zero at
fine-tuning time and therefore buys nothing where it matters.

```
  excitation              enters via          admissible   exercised at fine-tune
  joint-target noise      the action           yes          yes, fully
  randomised initial s0   s0                   yes          yes, branches start there
  command variety         action via policy    yes          yes
  soil / payload          per-episode channel  yes*         set to deployment value
  mid-episode push        exogenous force      NO           channel would be zero
```

\* admissible only if the parameter is actually written as a channel and driven.

### Why action noise is not optional

Measured on the existing corpus: only ~4% of action variance survives conditioning on the
state, over an effectively rank-2 subspace of a 12-dimensional action. With `a = pi(s)`,
the action is redundant given the state, so a model can fit the data perfectly while
learning nothing about `d s' / d a`.

That derivative IS the product. Fine-tuning differentiates through it. A corpus without
action excitation cannot identify it, and no amount of extra data fixes that, because the
missing information was never collected.

So: **action noise default ON**, applied after the policy acts and before actuation, with
the APPLIED action logged. Logging the clean action would recreate the confound while
hiding it.

## 2. Pushes: kept, but the push window never enters the corpus

Pushes stay, because what they buy is state coverage that nominal walking never reaches.
The recovery after a shove is exactly the off-manifold behaviour we want the model to
know. What must not enter the corpus is the interval during which the force acts, because
that is the only part whose cause is unlogged.

**The recovery is admissible.** Once the force is off, the robot's post-push velocity and
pose are fully in the state, so `(s, a) -> s'` is explainable again. Nothing is hidden.

### The implementation trap

Deleting push rows in place does NOT work. Training windows are built inside episodes as
`valid = episode_length - sequence_length + 1`, so removing rows from the middle silently
produces windows that span a temporal jump -- a worse defect than the one being fixed, and
an invisible one.

**Episodes are SPLIT at each push into independent segments**, each with its own id and
length, and the push-active rows plus a small guard on either side are dropped. Each
segment is then contiguous by construction.

### Sizing, because splitting is expensive

Window count is superlinear in segment length. A 1,475-row episode yields 1,347 windows
whole; split into seven ~190-row segments it yields 434, about a third. So the push
schedule must be designed around the window length rather than inherited:

- target segment length >= 4x the training sequence length
- one or two pushes per episode, not a Poisson stream every 2 s
- equivalently: longer episodes, or pushes placed at planned times

**A push delivered during unrecorded warmup costs no segments at all** and produces the
same off-nominal starting state. That is the preferred form; recorded mid-episode pushes
are the variant that also yields a clean pre-push segment.

## 3. Failure detection as a collection gate

Divergence is detected at collection time and the episode is TRUNCATED at the first
failing row, keeping the valid prefix. Never quarantine a whole episode for a defect in
its last few rows; never let a defect through because it stayed finite.

Checks, all cheap, all applied per row:

```
  non-finite      any NaN or inf in any recorded column
  joint step      |dq| per control step above a physical bound
  joint range     q outside the actuator limits
  base velocity   |v| above a physical bound
  base height     below floor contact or implausibly high
  attitude        grav_body_z indicating inverted
  time            non-monotonic or duplicated timestamps
  solver          particle count change, or any solver status the backend exposes
```

The episode metadata records which check fired, at which row, and how many rows survived.
A corpus summary reports truncation rates per check, so a rising rate is visible before it
is quoted.

**Divergences are kept, separately.** Where the real simulator fails is information: it is
the boundary of the region a fine-tuned policy must be kept inside. They go to a
`failures/` set with the same provenance, excluded from dynamics training, available for
analysing what the optimiser pushed the system toward.

## 4. Corpus acceptance gates

A corpus is not "collected", it is ACCEPTED, and only if it passes. This replaces
judgement about whether data is "diverse enough" with measurement.

**Gate 1, action identifiability.** Fraction of action variance surviving conditioning on
the state, and the effective rank of that residual. Must be materially above the ~4% /
rank-2 that the current corpus shows, or the derivative is not identifiable.

**Gate 2, the Jacobian check.** This is new and, as far as I can tell, has never been run
here. Take recorded states, finite-difference Chrono by perturbing each joint target, and
compare against the trained model's `d s' / d a`. We have only ever validated predictions;
the derivative is what fine-tuning consumes and a model can predict well while having a
wrong gradient. Report per-channel correlation and relative error.

**Gate 3, rollout horizon.** err/dist against horizon on a fixed grid, with the
predict-no-motion floor at 1.0 as the reference. Already implemented.

**Gate 4, coverage.** Occupancy of the state manifold against the deployment distribution,
and command-family balance.

Gates 1 and 2 are the ones that matter for this method and neither is currently measured.

## 5. Repository layout

```
quadruped/
  collect.py            one corpus, one command
  train.py              one NeDM model from one corpus
  finetune.py           one policy in the model, --method {analytic,ppo}
  evaluate.py           fine-tuned vs base, in Chrono

  params/
    excitation.yaml     noise sigmas, push schedule, initial-state bounds
    presets.yaml        state/action channel sets
    training.yaml       architecture, optimiser, schedules
    machines.yaml       per-host GPU memory, Chrono build hash, what it may run
    transforms.py       frame conversions, derived channels

  docs/
    STATE.md            what is true right now
    QUEUE.md            what is next, in order
    LESSONS.md          what we learned and will not repeat
    RETRACTIONS.md      claims withdrawn, so they are not resurrected

  data/                 manifests only; bytes live on the artifact root
  models/               manifests only; NeDM models, base and fine-tuned policies
  results/              every run, timestamped, with provenance
  doctor.py             preflight: build hash, dataset integrity, preset match, memory
```

### Provenance, one schema everywhere

Every artifact in `data/`, `models/` and `results/` carries the same `meta.json`:

```
  run_id            timestamp + short hash, the primary key
  git_commit        and a dirty flag, refused for a release artifact
  command           exact argv
  host              and GPU model
  chrono_build      md5, because replay is not build-invariant
  inputs            content hashes of every upstream artifact
  versions          python, torch, chrono
  metric_defs       version of each metric computed, so a redefinition is visible
```

`metric_defs` exists because a metric silently changed meaning mid-study and invalidated
every comparison spanning that date. Versioning it makes that a visible mismatch instead
of a wrong number.

### Why data and models are manifests, not bytes

`.git` is already 502 MB with artifacts tracked. Corpora run to ~17 GB and a single model
checkpoint is 911 MB. Committing them makes the repo unusable, and LFS only moves the
problem. Manifests give full traceability -- hash, provenance, storage path -- which is
the actual requirement.

## 6. What gets reused

Fresh scripts, but not a rewrite of things that are already correct and hard-won:

- **Reuse as libraries:** `crm_verdict.py` (paired scoring, refuses cross-host and
  cross-build pairing), the contact/gravity derivations, the Chrono scene setup.
- **Rewrite:** collection orchestration, preprocessing, the excitation layer, the
  training entry point, fine-tuning entry point.
- **Drop:** the accumulated one-off ablation scripts.

## 7. Open decision

`origin/main` contains none of the Go2 code -- no `src/nedm/quadruped`, no collector, no
fine-tuner, no verdict tool. `kyle/locomotion` is 728 commits ahead and holds all of it.

So "branch off main" means porting the library pieces listed in section 6 before anything
runs. That is a real fresh start and is defensible; it is also a few days of work before
the first episode is collected.

Branching off `kyle/locomotion` gives a working base immediately, with the new structure
added alongside and the old scripts deleted as they are replaced.

**Recommendation:** branch off `kyle/locomotion`, build `quadruped/` as the only supported
path, delete the old scripts as each is superseded, then PR to main when it is clean. Same
end state, working code throughout, and no window where nothing runs.

## 8. Excitation ranges: both are currently far too narrow

### Push direction is a shallow disc, not a sphere

Current sampling:

```python
th  = rng.uniform(0, 2*pi)
_f  = [mag*cos(th), mag*sin(th), mag*rng.uniform(-0.3, 0.3)]
```

Azimuth is full, but elevation is capped at `atan(0.3)`, about **+/-16.7 degrees from
horizontal**. Every push is effectively sideways. The measured maxima show it: 53.6 and
51.8 N in x and y against 17.0 N in z.

So the corpus contains almost no vertical loading or unloading events -- nothing that
presses the robot into the soil or lifts it -- and those are exactly the transitions that
distinguish deformable terrain from rigid.

**Replace with uniform sampling on the sphere**, which is the only distribution with no
preferred direction:

```python
z   = rng.uniform(-1.0, 1.0)
th  = rng.uniform(0, 2*pi)
r   = math.sqrt(max(0.0, 1.0 - z*z))
dir = (r*math.cos(th), r*math.sin(th), z)      # unit, uniform over the sphere
_f  = [mag * c for c in dir]
```

Note the current vector is also not normalised: its magnitude varies between `mag` and
`1.044*mag` with elevation, so "peak N" does not mean what it says. Sampling a unit
direction and scaling fixes that too.

Magnitude should sweep a wide range rather than `uniform(0.25, 1.0) * peak`. Log-uniform
over roughly a decade gives even coverage across scales instead of concentrating near the
peak.

### Action injection needs a per-episode scale ladder

Currently one scalar `--action-noise-sigma-rad`, applied identically to all twelve joints,
and defaulting to zero. That gives a corpus at a single excitation level, or none.

**Sample sigma per episode, log-uniform across a wide range**, so one corpus spans scales
the way the dose ladder spans data volume. Optionally vary sigma per joint so the residual
action covariance is not rank-deficient by construction -- the current corpus is
effectively rank 2 in a 12-dimensional action, and identical noise on every joint would
not fix that.

There is a real ceiling: enough injection and the robot simply falls, and those episodes
truncate. That ceiling is empirical, so the first job on the new pipeline is a short
calibration sweep -- rising sigma against truncation rate and against Gate 1 action
identifiability -- and the operating range is set from it rather than guessed.

The two gates make this self-correcting: too little injection fails Gate 1, too much shows
up as a truncation-rate spike.

## 9. Where collection runs

**Data collection runs on hpcfund (preferred) or euler. Never on the desktops.**

hpcfund first, for reasons already measured and recorded there: a node is 1 MI210 plus 16
cores, so one CRM episode fills a node exactly with no packing logic and no idle GPU;
`mi2101x` yields 10 GPU-hours per charged node-hour, the best ratio on that machine; and
24 nodes give wide array concurrency. CRM collection is embarrassingly parallel and job
arrays give per-run isolation and restartability.

euler is the fallback, and is where training and fine-tuning run regardless, because
**hpcfund cannot train** -- torch fails a GEMM on MI210 across eight configurations
tested. So the split is fixed:

```
  collect      hpcfund mi2101x array   (preferred)  |  euler  (fallback)
  train        euler                                |  ---
  fine-tune    euler A100/H100                      |  ---
  evaluate     euler per-GPU sharding               |  hpcfund arrays
```

Desktops are excluded from collection entirely. Several of this study's
hardest-to-untangle problems came from corpora and verdicts produced on whichever box was
free, and replay is not machine-invariant, so `crm_verdict.py` already refuses to pair
across hosts. Collecting on one cluster with one pinned Chrono build removes that class of
problem at the source instead of detecting it later.

Enforcement, not convention: `collect.py` refuses to run on a host not listed as a
collection host in `params/machines.yaml`, and the Chrono build hash goes into the corpus
manifest so a mixed-build corpus is visible rather than silent.

## 8b. Action injection: what to inject, and why not white noise

The action is twelve JOINT POSITION TARGETS at 50 Hz, tracked by a PD controller. That
fact decides the scheme, because a position target is not applied directly -- it is
filtered by the controller and the robot's inertia before it becomes motion.

### Why i.i.d. per-step noise is the wrong default

Independent Gaussian noise on each control step has most of its energy above the
bandwidth the robot can follow. The PD loop and the leg inertia low-pass it, so the target
moves and the body largely does not.

That is worse than doing nothing. It produces action variance with no matching state
response, and a model fit to that data learns exactly the wrong lesson: that changing the
action barely changes the next state. Since `d s' / d a` is the quantity fine-tuning
consumes, injecting noise the plant rejects actively suppresses the thing we are trying to
identify.

### Default: temporally correlated noise, independent per joint

Ornstein-Uhlenbeck on each joint target -- a mean-reverting random walk -- with a
correlation time of roughly 0.1 to 0.2 s, which is 5 to 10 control steps at 50 Hz and
sits inside the band the robot actually follows.

This is the same thing as "inject a random RATE": integrating a random rate IS a random
walk, and mean reversion is what stops it drifting off and bounds it. So that intuition
is right; OU is the bounded form of it.

Per-joint independence matters. The residual action covariance in the current corpus is
effectively rank 2 in twelve dimensions, and one shared noise signal across joints would
leave it low-rank. Independent draws per joint are what make the excitation full-rank.

Per-episode sigma, log-uniform over a wide range, so one corpus spans excitation scales
the way the dose ladder spans data volume.

### Minority: sparse single-joint probes

On a fraction of episodes, roughly 10-20%, hold the policy and apply a deliberate
step or chirp to ONE joint target, or a small random subset, leaving the rest to the
policy.

This is the cleanest identifiability signal available: perturbing one input at a time and
recording the response is finite-differencing the Jacobian inside the data itself, and it
maps directly onto Gate 2, which compares the model's Jacobian against Chrono's. OU noise
covers the coordinated, multi-joint directions a policy actually explores; single-joint
probes pin down the individual columns. Neither alone does both, which is why the corpus
carries both.

This is also the "random direction on a random set of joints" idea, scoped: it is
excellent for identifiability and poor as a sole default, because real policy updates move
many joints together.

### Constraints

Injection is applied to the target AFTER the policy acts and BEFORE actuation, and the
INJECTED target is what gets logged. Targets are clipped to joint limits, so the
excitation never commands a configuration the actuator cannot hold. As with magnitude,
the usable ceiling is empirical and comes out of the calibration sweep.

## 10. Base policy: rl_sar `robot_lab/policy.pt`, memoryless

**Decision: adopt `robot_lab/policy.pt` from rl_sar. Not `himloco.pt`, and not the
existing imported policy.**

### What is wrong with what we have

The current policy's own documentation states it:

> THE POLICY IS STATEFUL. It is a concurrent teacher-student model: a 5-step observation
> history feeds a student_encoder to a 32-dim latent, and the actor consumes
> [obs 45, latent 32] = 77. Measured, it stabilises after exactly 5 calls on a repeated
> input. So it must be called once per control step in order, and RELOADED between
> episodes.

and separately:

> THE SIGN NEGATION IS INHERITED ON FAITH. The Chrono harness negates joint positions,
> velocities and targets, and no source we have records why.

Both are load-bearing defects for this method. A policy that stabilises only after five
calls means a 15-step branch may spend its first third with a mis-conditioned latent,
which is a third of every gradient. An unexplained sign transform between policy and
simulator has already produced one sign-flip bug.

### Why not himloco

himloco is not the plain MLP it is sometimes described as. It carries a 6-step
observation history and an internal velocity/latent estimator head, so structurally it is
the same class of architecture as the one being replaced.

The disqualifying point is narrower and worse: **that estimator was fit on rigid-terrain
dynamics.** On CRM it is out of distribution before fine-tuning even begins, so
fine-tuning would co-adapt an estimator and a policy simultaneously against a learned
model. Any resulting change could not be attributed to either. That is an unidentifiable
experiment, and this study is an evaluation, not a locomotion-performance paper.

### Why robot_lab

A plain rsl_rl actor MLP, Identity normaliser, `observations_history: []` -- memoryless.
The consequences for this pipeline are all simplifications:

```
  a = MLP(obs),  obs = f(propagated state, command)
```

- the gradient path through a branch is a single feedforward net, well defined from step 1
- branches need no observation history seeded from the corpus
- no warm-up transient, no reset between episodes, no internal buffer to desynchronise
- rsl_rl native, so the PPO comparison runs in the stack the policy was trained in
- published and standard, so no reviewer can attribute a result to encoder conditioning

### Adoption gates, asserted at load, not assumed

The lesson that a flag accepted and ignored reads exactly like a working one applies here.
`load_policy()` refuses to proceed unless:

```
  normaliser is Identity                  (else observations are silently rescaled)
  observations_history == []              (else the policy is not memoryless)
  state_dict has no encoder/estimator keys (else it is not a plain actor)
  obs dimension matches the built observation exactly
  action dimension == 12
```

Two further things are established by TEST rather than inherited:

**Sign and ordering.** The joint sign convention and channel ordering are fixed by an
action round-trip test against Chrono, not carried over on faith. A precedent exists in
this repository: `check_action_roundtrip.py`, committed as "the test that would have saved
three days".

**Observation construction.** robot_lab policies are trained under IsaacSim, so the
observation layout, scales and command convention come from robot_lab's own config and are
verified term by term against what the Chrono harness builds. The existing policy's notes
record a command-scale mismatch that was unobservable only because the yaw command
happened to be identically zero. That class of error is invisible until it is not.

### What this retires

The stateful-policy handling, the history buffer, the reset-per-episode requirement, the
5-call warm-up, the inherited sign negation, and the estimator confound. None of them need
to be modelled, worked around, or argued about in the new pipeline.
