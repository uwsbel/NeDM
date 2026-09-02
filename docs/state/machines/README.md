# Machines

One file per machine, named by hostname. This is the portability layer: nothing
else in `docs/state/` should hardcode a path or an interpreter.

**If you are on a machine with no file here, write one before doing substantial
work.** Fill it from the template below — it takes five minutes and saves the
next agent an hour.

| File | Machine | Role |
|---|---|---|
| [`newton.md`](newton.md) | `newton` (RTX 4090) | Chrono collection + rendering. Home of raw frame stores. |
| [`workstation-5090.md`](workstation-5090.md) | Harry's RTX 5090 desktop | Training, eval, figures. Never raw collection. |
| [`kyle-sbel.md`](kyle-sbel.md) | Kyle's box (RTX 3090) | Dev, docs, light training. |
| [`euler.md`](euler.md) | UW Euler cluster | State-only CPU collection via SLURM arrays. |
| [`manuscript.md`](manuscript.md) | — | Where the paper source lives and how to build it. |

## Portable invocation

Every command written elsewhere in `docs/state/` assumes these three variables,
so the same command works on every box:

```bash
export NEDM_ROOT=/path/to/NeDM
export NEDM_PY=/path/to/the/pychrono+torch/python
export PYTHONPATH=$NEDM_ROOT/src
```

Run scripts as `cd "$NEDM_ROOT" && "$NEDM_PY" -m ...` or
`"$NEDM_PY" scripts/<stage>/<script>.py`. Never `python` bare — the system
interpreter has no pychrono on any of these boxes.

## Cross-machine rules

1. **Collection goes where the GPU renders** (`newton`) or where the cores are
   (Euler). Not on a training box.
2. **Code moves by git**, never by rsync or by editing on the remote:
   commit and push here, `git pull --ff-only` there.
3. **Data moves by rsync**, and only the trainable, compressed form:
   ```bash
   rsync -avP newton:NeDM/artifacts/<name> "$NEDM_ROOT/artifacts/"
   ```
   Check both sides first: `ssh newton "du -sh ~/NeDM/artifacts/<name>"` and
   `df -h "$NEDM_ROOT"`.
4. **Record what you moved** in `docs/state/data/` so the next person knows which
   box holds which copy.

## Template

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
