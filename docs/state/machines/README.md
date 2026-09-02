# Machines

The portability layer: nothing else in `docs/state/` hardcodes a path or an
interpreter.

## Machines you can actually run on

| File | Machine | Role |
|---|---|---|
| [`kyle-sbel.md`](kyle-sbel.md) | Kyle's box, RTX 3090 | **Everything.** Dev, docs, training, analysis, manuscript builds |

That is the whole list as of 2026-09-02. Plan work on the assumption that this
is the only compute available, and that anything needing another box is blocked
until access exists.

Also here: [`manuscript.md`](manuscript.md) — where the paper source lives and
how to build it locally (Tectonic, no sudo, works on this box today).

## Reference only — no access

[`reference/`](reference/) documents machines **you cannot log into**. They are
recorded because the existing pipeline, the storage rules, and the location of
every collected dataset assume them — not because they are options.

| File | Machine | Why it is documented |
|---|---|---|
| [`reference/newton.md`](reference/newton.md) | RTX 4090 collection box | Holds the raw frame stores and every Study 3 dataset |
| [`reference/workstation-5090.md`](reference/workstation-5090.md) | Harry's RTX 5090 desktop | Where the published training and eval runs happened |
| [`reference/euler.md`](reference/euler.md) | UW Euler cluster | Where the SLURM-scale state-only collections ran |

**Do not write a plan whose next step runs on one of these.** If a task needs
one, say so and stop — that is a blocker to escalate, not a step to attempt.

## Portable invocation

Every command elsewhere in `docs/state/` assumes these three variables:

```bash
export NEDM_ROOT=/path/to/NeDM
export NEDM_PY=/path/to/the/pychrono+torch/python
export PYTHONPATH=$NEDM_ROOT/src
```

Run scripts as `cd "$NEDM_ROOT" && "$NEDM_PY" -m ...` or
`"$NEDM_PY" scripts/<stage>/<script>.py`. Never `python` bare — the system
interpreter has no pychrono.

## How work and data move between boxes

Recorded so the conventions are legible, and so it is clear what is unavailable
from here.

1. **Code moves by git.** Commit and push; `git pull --ff-only` on the far side.
   Never edit files on a remote box directly.
2. **Data moves by rsync**, and only the trainable, compressed form — never raw
   frame dumps. `rsync -avP newton:NeDM/artifacts/<name> "$NEDM_ROOT/artifacts/"`
   is the project's pattern; it does not work from `kyle-sbel`, which cannot
   resolve `newton`.
3. **Collection runs where the renderer or the cores are**, never on a training
   box. This is a project rule, and separately it is moot here: no collection
   box is reachable.
4. **Record what you moved** in [`../data/`](../data/) so the next person knows
   which box holds which copy.

## Adding a machine

If you gain access to a box, write it up here — five minutes, saves the next
agent an hour.

```markdown
# <hostname>

**Verified:** <date> · **Owner:** <who> · **Role:** <one line>

| | |
|---|---|
| GPU | |
| CPU / RAM | |
| Disk (free) | |
| Repo path | |
| Interpreter | |
| Reachable from | |

## What this machine is for
## What it must not do
## Launch recipe
## Gotchas
```
