# Agent context — start here

**Purpose.** Hand this file to any AI agent, on any machine, at the start of any
session. It should be enough to orient without reading the codebase first.

**Maintainer:** Kyle (`kasha2@wisc.edu`). **Branch:** `kyle/locomotion`.
**Last verified:** 2026-09-02.

Anything in this file that names a file, run, or number was true when written.
Verify before relying on it — §6 says how.

---

## 1. What this project is

NeDM / NRD builds **neural reduced dynamics** models: fast learned surrogates of
expensive Project Chrono scenes, used to train control policies that are then
validated back in Chrono.

The pipeline is the same in every study:

```
Chrono high-fidelity episodes  →  reduced state z1  →  causal transformer NRD
                                                            ↓ (frozen)
                              vectorized env  →  PPO policy  →  back to Chrono
```

The research claim is about **abstraction**, not about neural surrogates in
general: what should the model propagate, what can be supplied as an input, what
can be recovered analytically, and what can be omitted — such that the model is
both fast and good enough to train a transferable policy.

Two lines of work are live:

| Line | State | Where |
|---|---|---|
| **State-only NRD** (the published paper) | Done, submitted | `main` |
| **NRD + vision** (`z2` camera latent appended to `z1`) | Studies 1 done, Study 3 at WP0 | `nrd_vision` |

The manuscript is *Learning the Right Abstraction: Neural Reduced Dynamics for
Complex Robot Control* (Zhang and Negrut), arXiv:2608.19375v1. Source is in a
**different repo** — see [`docs/state/machines/manuscript.md`](docs/state/machines/manuscript.md).

---

## 2. Read these, in this order

Everything below is a link. Do not re-derive what is already written down.

**Orientation (read first, ~10 min):**

1. [`README.md`](README.md) — repo layout, environment, what each `scripts/`
   subfolder is for.
2. [`docs/state/progress/`](docs/state/progress/) — **current state of each
   workstream and the single next action for it.** This is the file that answers
   "where are we?".
3. [`docs/state/machines/`](docs/state/machines/) — the machines you can run
   on, how they are reached and how files move between them, and (under
   `reference/`) the boxes that hold the project's data but are **not**
   accessible.

**Depth, on demand:**

| Question | Doc |
|---|---|
| How was a published number produced? | [`docs/progress.md`](docs/progress.md) — the reproduction record |
| Which checkpoint is which? | [`docs/state/checkpoints/`](docs/state/checkpoints/), then [`docs/model_checkpoints.md`](docs/model_checkpoints.md) |
| Where is dataset X, how big, on which box? | [`docs/state/data/`](docs/state/data/) |
| Has someone already hit this bug? | [`docs/state/lessons/`](docs/state/lessons/) and [`.claude/lessons_learned.md`](.claude/lessons_learned.md) |
| Why was it built this way? | [`docs/state/decisions/`](docs/state/decisions/) |
| What is the vision plan? | [`docs/vision/NRD_overall_project_plan.md`](docs/vision/NRD_overall_project_plan.md) |
| What happened in Study 1 / Study 3? | `docs/vision/double_pen/*_implementation_notes.md`, `docs/vision/hmmwv_traverse/wp0*_implementation_notes.md` |

The `docs/vision/**/implementation_notes.md` files are the highest-signal
documents in the repo. They record what was *tried and failed*, not just what
shipped. Read the relevant one before touching a study.

---

## 2b. If you are working on the quadruped case study

It is the newest line of work and the one most likely to be mid-flight. **The
contribution is contact-mode conditioning** — a methodological extension the
manuscript names as future work — **not "a fourth vehicle."**

Read `docs/state/decisions/quadruped-case-study-plan.md` first; it links the
other three in dependency order.

Three findings that will otherwise cost you a day each:

- **The walking policy has no command input.** Not one command in the data — no
  channel to write to. It responds to a yaw command *incoherently*, so a
  path-follower bolted on will appear to half-work.
- **The foot is smaller than the SPH kernel support** (22 mm against 40 mm), so
  it floats 22.9 mm above true contact and never penetrates. Everything
  previously read as sinkage was surface deflection.
- **Contact cannot be detected by a force threshold on soil.** No bimodality, no
  plateau; a plain threshold invents 66% of its transitions. Use hysteresis.

## 3. House rules that are not obvious from the code

These are load-bearing. Violating them silently produces wrong results.

1. **Checkpoints are selected on open-loop rollout error, not one-step
   validation loss** (`checkpoint_metric: rollout_sel`). The file is still named
   `best_val.pt` but it is the rollout-selected epoch. The two metrics rank
   checkpoints differently.
2. **Two machines are reachable and both are provisioned.** `kyle-sbel` and
   `kyle-N7-B650E` both have the repo, pychrono, and working CUDA, on different
   torch builds by necessity (the 5070 Ti is Blackwell and needs cu130). Neither
   has `git-lfs`, so on both, every checkpoint under `artifacts/` is a pointer
   stub and loading one fails as an *unpickling* error. The project convention is that collection runs on `newton`
   (GPU rendering) or Euler (state-only), but **neither is reachable** — see
   [`docs/state/machines/reference/`](docs/state/machines/reference/). Do not
   plan a step that runs on a box outside
   [`docs/state/machines/`](docs/state/machines/); surface it as a blocker.
   Both reachable boxes are driven over Remote Control, under the policy in
   [`docs/state/machines/remote-control.md`](docs/state/machines/remote-control.md):
   agents do the work, a human installs software.
3. **Deliver code to other machines by commit + push, then `git pull --ff-only`
   there.** Never edit files directly on a remote box.
4. **`z2` must be normalized before use.** The encoder's LayerNorm'd latents
   share a huge constant component (raw pairwise cosine 0.9998 between arbitrary
   frames). Raw-cosine latent metrics read 1.000 and mean nothing. Use the
   model's `z2_mean`/`z2_std`.
5. **Reconstruction losses need foreground weighting.** A plain L1 autoencoder
   will erase the moving object entirely and reconstruct only the static
   background — this was verified, not hypothesized.
6. **One job at a time on a shared GPU box**, unless throughput has been measured
   under contention.
7. **`z2` is appended to `z1`, never substituted for it.** This is the
   architectural commitment of the whole vision line.

---

## 4. Layout of the state docs

`docs/state/` is the machine-flexible, always-current layer. `docs/` proper is
the paper's reproduction record and the study plans; those are stable documents
and should not be rewritten casually.

```
docs/state/
├── machines/     The boxes you can run on: paths, envs, constraints. Also
│                 how they are reached (remote-control.md) and how files move
│                 between them (file-sync.md). reference/ holds inaccessible
│                 boxes, for context only.
├── progress/     Current state + next action, one file per workstream.
├── lessons/      Hard-won findings. Gotchas that cost real time.
├── checkpoints/  Model registry: what exists, where, what it is for.
├── data/         Dataset registry: what exists, where, how big, how to get it.
└── decisions/    Decision log and open questions.
```

Each subfolder has a `README.md` stating what belongs in it and a template.

---

## 5. Bootstrapping on an unfamiliar machine

Nothing in `docs/state/` assumes a path. Set these once per shell:

```bash
export NEDM_ROOT=/path/to/NeDM      # this repo
export NEDM_PY=$(which python)      # the pychrono+torch interpreter
export PYTHONPATH=$NEDM_ROOT/src
```

Then identify the machine you are on and read its file:

```bash
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h "$NEDM_ROOT"
ls $NEDM_ROOT/docs/state/machines/
```

Today that is [`kyle-sbel.md`](docs/state/machines/kyle-sbel.md), the only
provisioned box, or [`kyle-N7-B650E.md`](docs/state/machines/kyle-N7-B650E.md),
which is reachable but has no repo and no verified interpreter yet. `docs/state/machines/reference/` documents boxes that
hold the project's data and runs but **cannot be logged into** — they are
context, not options.

If you gain access to a new machine and it has no file in
`docs/state/machines/`, **write one** before doing substantial work. Template is
in [`docs/state/machines/README.md`](docs/state/machines/README.md).

Environment: `conda env create -f environment.nedm.yml && conda activate nedm`
(env `nedm`, pychrono 10.0.0). `environment.yml` (pychrono 9.0.1) exists only for
the oldest datasets.

**Neither reachable box runs that environment.** There is no `nedm` env on
`kyle-sbel`; its `chrono` env carries pychrono **9.0.0**, older than both files
in this repo, installed from a local tarball rather than a channel.
`kyle-N7-B650E` runs its own builds. Do not assume a version from
`environment.nedm.yml`; read the machine's own file and use `$NEDM_PY`. Verified
2026-09-02.

---

## 6. Verify before you trust

This file and everything under `docs/state/` is a **snapshot**. Cheap checks:

```bash
cd "$NEDM_ROOT"
git log --oneline -10                       # what actually landed recently
git branch -a                               # which branches exist
git status -sb                              # uncommitted local work
ls artifacts/training_runs artifacts/rl_runs # what checkpoints are on THIS box
du -sh artifacts/* 2>/dev/null | sort -h     # what data is on THIS box
```

Artifacts are **per-machine**. Most runs named in `docs/state/checkpoints/` and
`docs/state/data/` exist only on machines listed under `machines/reference/`,
which are not reachable — so expect the `ls` above to come back mostly empty.
The registry records which box holds what; the `ls` records what you actually
have.

Git LFS holds the checkpoints. If `git lfs version` fails, checkpoint files are
pointer stubs, not weights — install `git-lfs`, then `git lfs install && git lfs pull`.

---

## 7. Keeping this current

The rot risk is real; these docs are only worth what they cost to maintain.
Minimum discipline:

- **After any session that changes state**, update the one `docs/state/progress/`
  file for that workstream. That is the required update; the rest are optional.
- **When something costs you more than an hour of confusion**, add a
  `docs/state/lessons/` entry. Say what was expected, what happened, and the fix.
- **When you produce a checkpoint or dataset someone else might want**, add a row
  to the relevant registry, including which machine it is on.
- **Do not restate detail here.** This file is an index and a set of rules. If it
  starts accumulating results, they belong in `docs/state/`.
- Date every claim. Prefer "verified 2026-09-02" over "currently".
