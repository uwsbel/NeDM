---
name: storage
description: Which machine runs what, and where large artifacts live. Use before planning data collection, launching anything that writes large artifacts, transferring datasets between machines, or deciding where a job should run.
---

# Compute and storage across the fleet

**Rewritten 2026-09-04.** The previous version described a `this desktop` /
`newton` / `Euler cluster` topology dated 2026-09-01 — the day before the current
fleet existed. None of those hosts are reachable and none of its rules applied to
the machines actually doing the work. It is superseded entirely.

## Machines

| Host | Role | GPU | Free disk | Workdir |
|---|---|---|---|---|
| **Kyles-MacBook-Pro** | coordinator | — (arm64) | — | `~/NeDM` |
| **`sbel-pc`** = kyle-sbel | collection, training, eval | RTX 3090, 24 GB, sm_86 | ~1.2 TB | `~/Documents/sbel` |
| **`dorm-pc`** = kyle-N7-B650E | collection, training, eval | RTX 5070 Ti, 16 GB, sm_120 | ~331 GB | `~/sbel` |
| **`kyle-B650M-D3HP`** | **cannot run our collectors** — see below | RTX 5060 Ti, 8 GB, sm_120 | ~392 GB | `~/sbel` |

Specs above are as reported by each machine on 2026-09-04, not inherited from an
earlier document. **The old table's GPU entries were wrong for both compute nodes
and nearly caused a correct metadata field to be overwritten with a plausible
false one** — if a spec here matters to a decision, have the machine confirm it.

### `kyle-B650M-D3HP` — surveyed 2026-09-04, not usable as a collector

12 logical cores and 30 GiB RAM, which would suit 8-concurrent CPU-bound collection
well. **The blocker is that it has no Python stack at all:**

- `import pychrono` **fails** — no conda, mamba, or virtualenv anywhere, and
  `CH_ENABLE_MODULE_PYTHON` is OFF in all four of its Chrono build trees, so it has
  never produced bindings.
- NeDM is not cloned there and the Go2 URDF assets are absent.
- Syncthing is not running and it does not share `sbel-shared`.

What it *does* have: Chrono 10.0.0 built from source in three worktrees, **C++
Vehicle module compiled**, Sensor with OptiX working. **FSI is OFF in all four
builds**, so granular soil is unavailable without a rebuild. Those are partial
builds — single demo targets, not full `ninja all`.

**CUDA, corrected:** `nvcc` on PATH is 11.5, which cannot compile for this card
(sm_120). But **CUDA 13.0 is also installed at `/usr/local/cuda-13.0`** and works —
reach it with `-DCUDAToolkit_ROOT=/usr/local/cuda-13.0` and
`-DCHRONO_CUDA_ARCHITECTURES=120`. The earlier "CUDA 11.5, cannot do GPU work"
record was wrong; it is "13.0 installed, 11.5 is the PATH default".

Making it a collector means a Python environment plus either installing `pychrono`
or rebuilding Chrono with the Python module — hours — plus cloning the repo,
fetching assets, and configuring Syncthing. **`sudo` there needs a password only
Kyle can supply.** Not worth it for rigid collection, which the two existing nodes
finish in about an hour. Revisit if granular collection becomes the bottleneck
again, and note that would need an FSI rebuild too.

## Where work runs

**Both compute nodes collect data.** `sbel-pc` and `dorm-pc` have each collected
Chrono episode datasets — 968 rigid + 152 CRM and 152 CRM + 304 rigid respectively.
There is no "collection machine" and no prohibition on collecting anywhere.

- **Rigid-body collection** is CPU-bound and parallelises well (8-concurrent measured).
- **CRM / granular collection** saturates one GPU and runs sequentially — roughly
  2 minutes of wall-clock per 16 s episode. It is the fleet's scarcest resource.
- **Training and policy rollout** are GPU-bound. Check with the machine before
  launching; a CRM collection and a training run will contend.

The coordinator does not run simulations.

## Dataset identity across machines

**Every collection must use a distinct `--seed-offset`.** Episode ids embed it, and
`preprocess.py` raises on duplicate ids across roots — which is the only thing that
makes a cross-machine merge safe.

Offsets used so far: `0` (sbel-pc), `1000000` (dorm-pc), `2000000` and `3000000`
(reserved for the joint-state collection). **Pick an unused one and record it.**

## Transferring datasets

Syncthing is the transfer path — the machines are on networks without inbound
ports, so there is no direct `ssh`/`rsync` between them.

- Shared folder: `~/sync/sbel`, three devices, `sendreceive`.
- `.stignore` excludes `*.mp4`, `*.exr`, `build/`, `__pycache__` and similar.
  **`.tar.gz` is not excluded** — archive a dataset and it syncs.
- **Send one archive, not a directory tree.** 460 loose files is 460 writes into a
  shared folder; a tarball is one.
- **Move the smaller half.** Direction is set by volume, not by which machine has
  space.
- Verify by `sha256` on both ends before trusting a transfer.

Writes into `~/sync/sbel` publish to every other device automatically, so they may
require explicit approval on some machines. That is a per-machine permission
matter, not a rule of this document.

## Large artifacts

Raw per-frame stores, uncompressed memmaps and video stay on the machine that
produced them. Move the processed cache or an archive, not the frame directory.
Check free space on the receiving machine first — `df -h`.
