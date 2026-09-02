# kyle-N7-B650E

**Verified:** 2026-09-02 · **Owner:** Kyle · **Role:** Second compute box.
Reachable, **not yet provisioned** for this project.

| | |
|---|---|
| GPU | NVIDIA RTX 5070 Ti, 16 GB |
| CPU / RAM | 32 threads / 60 GB (55 GB available) |
| Disk (free) | 906 G total, **369 G free** (58% used, `/dev/nvme0n1p5`) |
| Repo path | **none yet**. NeDM is not checked out on this box |
| Interpreter | **unverified**. conda is present (`base` active); no pychrono env confirmed |
| Reachable from | the coordinator Mac, as `dorm-pc`, via [Remote Control](remote-control.md) |
| OS | Linux 6.17.0-20-generic |

Everything above except the last two rows was measured on the box on 2026-09-02.
The two blanks are real: this machine has hardware and access, and nothing else.

## What this machine is for

Nothing yet. It is documented because it is now **reachable**, which changes the
planning assumption recorded in [`kyle-sbel.md`](kyle-sbel.md) and in
`AGENT_CONTEXT.md` §3.

Once provisioned, the split against [`kyle-sbel.md`](kyle-sbel.md) is:

| | `kyle-sbel` | `kyle-N7-B650E` |
|---|---|---|
| VRAM | **24 GB** (RTX 3090) | 16 GB (RTX 5070 Ti) |
| CPU threads | 16 | **32** |
| RAM | 30 GB | **60 GB** |
| Free disk | **1.3 T** | 369 G |

So: VRAM-bound training and anything needing room for datasets stays on
`kyle-sbel`. CPU-bound work, parallel builds, and Chrono state-only collection
are the natural fit here, on twice the cores and twice the RAM.

Before planning real work on it, someone must clone the repo, stand up a
pychrono environment, verify `torch.cuda.is_available()` on Blackwell, and
replace the two unverified rows above. Until then, treat any step that runs here
as unproven rather than blocked.

## What it must not do

1. **Do not put datasets here.** 369 G free against `kyle-sbel`'s 1.3 T, and it
   is the box on the worse network path. See [`file-sync.md`](file-sync.md).
2. **Do not move large files between here and `kyle-sbel` directly.** That leg
   is always relayed and throughput-limited.
3. **Do not edit files here by hand.** Same rule as everywhere: commit, push,
   `git pull --ff-only`.

## Launch recipe

Not yet established. When the repo lands, the portable form applies:

```bash
export NEDM_ROOT=/home/kyle/<path once cloned>
export NEDM_PY=<pychrono+torch interpreter once built>
export PYTHONPATH=$NEDM_ROOT/src
```

The `claude-rc` unit currently runs with `WorkingDirectory=/home/kyle/sbel`,
which exists but is **empty and not a git repo**. Pointing it at the NeDM
checkout once one exists would be an improvement, and would enable the diff pane
on connected devices.

## Gotchas

1. **The RTX 5070 Ti is Blackwell.** Do not assume the torch build that works on
   `kyle-sbel`'s RTX 3090 works here. `torch 2.6.0+cu124` may need a newer CUDA
   or a nightly wheel for `sm_120`. Verify before budgeting any run.
2. **16 GB VRAM, not 24.** Batch sizes and model configs tuned on `kyle-sbel`
   will not transfer unchanged.
3. **This box is on a different network from `kyle-sbel`** and reaches it only
   over a public relay. See [`file-sync.md`](file-sync.md).
4. `~/sbel` on this machine is empty and unrelated to
   `/home/kyle/Documents/sbel` on `kyle-sbel`, despite the similar name. Do not
   assume they are two copies of one thing.
