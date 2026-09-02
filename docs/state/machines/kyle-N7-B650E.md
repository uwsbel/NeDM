# kyle-N7-B650E

**Verified:** 2026-09-02 · **Owner:** Kyle · **Role:** Second compute box.
Provisioned except for `git-lfs`.

| | |
|---|---|
| GPU | NVIDIA RTX 5070 Ti, 16 GB |
| CPU / RAM | 32 threads / 60 GB (55 GB available) |
| Disk (free) | 906 G total, **369 G free** (58% used, `/dev/nvme0n1p5`) |
| Repo path | `/home/kyle/sbel/NeDM`, branch `kyle/locomotion` at `272fcfa` |
| Interpreter | `/home/kyle/miniconda3/envs/chrono312/bin/python` (pychrono OK, torch 2.10.0.dev20251114+**cu130**, `cuda True`). `envs/chrono` (3.10.16) is equivalent |
| Reachable from | the coordinator Mac, as `dorm-pc`, via [Remote Control](remote-control.md) |
| OS | Linux 6.17.0-20-generic |

All rows measured on the box on 2026-09-02.

## What this machine is for

Real work, with one caveat: `git-lfs` is missing, so every checkpoint is a
pointer stub (gotcha 2). Fresh training and CPU-bound work are fine today;
resuming from a checkpoint is not.

The split against [`kyle-sbel.md`](kyle-sbel.md):

| | `kyle-sbel` | `kyle-N7-B650E` |
|---|---|---|
| VRAM | **24 GB** (RTX 3090) | 16 GB (RTX 5070 Ti) |
| CPU threads | 16 | **32** |
| RAM | 30 GB | **60 GB** |
| Free disk | **1.3 T** | 369 G |

So: VRAM-bound training and anything needing room for datasets stays on
`kyle-sbel`. CPU-bound work, parallel builds, and Chrono state-only collection
are the natural fit here, on twice the cores and twice the RAM.

Note the torch builds differ by box and this is not incidental: `kyle-sbel`
runs `2.6.0+cu124`, which does not cover this card. See gotcha 1.

## What it must not do

1. **Do not put datasets here.** 369 G free against `kyle-sbel`'s 1.3 T, and it
   is the box on the worse network path. See [`file-sync.md`](file-sync.md).
2. **Do not move large files between here and `kyle-sbel` directly.** That leg
   is always relayed and throughput-limited.
3. **Do not edit files here by hand.** Same rule as everywhere: commit, push,
   `git pull --ff-only`.

## Launch recipe

```bash
export NEDM_ROOT=/home/kyle/sbel/NeDM
export NEDM_PY=/home/kyle/miniconda3/envs/chrono312/bin/python
export PYTHONPATH=$NEDM_ROOT/src
```

Use the absolute interpreter path. **`conda` is not on `PATH` in a
non-interactive shell** on this box, so anything relying on `conda activate`
fails under the `claude-rc` unit, which runs non-interactively. Only
`/usr/bin/python3` is on the default `PATH`, and it has no pychrono.

## Gotchas

1. **The RTX 5070 Ti is Blackwell, so it needs a cu130 build.** This is solved
   here, not outstanding: the `chrono` and `chrono312` envs run
   `torch 2.10.0.dev20251114+cu130` and report `cuda True`. Do not "fix" them
   toward `kyle-sbel`'s `2.6.0+cu124`, which does not cover `sm_120`. The two
   boxes legitimately run different torch builds.
2. **`git-lfs` is not installed.** All 106 `.pt`/`.pth` files under `artifacts/`
   are 132-byte pointer stubs, 14,020 bytes for the lot. The failure mode is
   nasty: loading one raises an *unpickling* error, not a missing-file error.
   Same gotcha as `kyle-sbel`, same fix, and installing it needs Kyle since
   agents here cannot install software ([`remote-control.md`](remote-control.md)).
3. **16 GB VRAM, not 24.** Batch sizes and model configs tuned on `kyle-sbel`
   will not transfer unchanged.
4. **This box is on a different network from `kyle-sbel`** and reaches it only
   over a public relay, confirmed from both ends. See
   [`file-sync.md`](file-sync.md).
5. `~/sbel` here holds this box's NeDM checkout and is unrelated to
   `/home/kyle/Documents/sbel` on `kyle-sbel`, despite the similar name. They
   are separate clones, not two views of one thing.
