# Harry's RTX 5090 workstation

**Verified:** 2026-09-01 (from `.claude/skills/storage/SKILL.md`) ·
**Owner:** Harry · **Role:** Training, evaluation, figures.

| | |
|---|---|
| GPU | NVIDIA RTX 5090, 32 GB |
| Disk (free) | 1 T total, ~700 G free after 2026-09-01 cleanup |
| Repo path | `/home/harry/NeDM` |
| Interpreter | conda env `nedm` |
| Reachable from | — (it is the box that reaches out to `newton` and Euler) |

## What this machine is for

Model training, RL, Chrono-backed evaluation, figure generation. It is the box
that drives `newton` over SSH.

## What it must not do

**No raw dataset collection.** Only trainable assets land here: compressed and
processed training stores, layout/camera manifests, checkpoints, small eval
summaries. Raw frame dumps, per-frame PNG directories, uncompressed memmaps and
video archives stay on `newton`.

## Disk budget

With ~700 G free, local `artifacts/` can hold ~400 G of training-ready stores;
keep ≥250 G headroom for checkpoints, caches and system growth.

Study 3 context: raw RGB-D frames are 25–49 GiB at pilot tier and
**122–488 GiB at full tier** if stored uncompressed. Full-tier raw must never
land here. The compressed episode-chunked store *is* the trainable asset
(28.8× ratio measured, so the full tier is ~17 GiB) — pulling that here for
training is the intended workflow.
