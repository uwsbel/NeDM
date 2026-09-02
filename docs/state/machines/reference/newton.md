# newton

> **NO ACCESS.** Kyle cannot log into this machine (verified 2026-09-02).
> Documented for context only — see [`README.md`](README.md). Work that requires
> this box is a blocker to escalate, not a step to attempt.

**Verified:** 2026-09-01 (from `.claude/skills/storage/SKILL.md` and the WP0c
notes) · **Role:** Chrono data collection, including Chrono::Sensor RGB-D
rendering. Home of the raw frame stores.

| | |
|---|---|
| GPU | NVIDIA RTX 4090, 24 GB |
| CPU | 32 cores |
| Disk (free) | 1.8 T total, ~630 G free; `artifacts/` already ~542 G |
| Repo path | `~/NeDM` |
| Interpreter | `~/anaconda3/envs/nedm/bin/python` (conda env `nedm`) |
| Reachable from | Harry's 5090 workstation via `ssh newton`. **Not** from `kyle-sbel` as of 2026-09-02. |

```bash
export NEDM_ROOT=~/NeDM
export NEDM_PY=~/anaconda3/envs/nedm/bin/python
export PYTHONPATH=$NEDM_ROOT/src
```

## What this machine is for

Everything that renders or writes raw per-frame data. All Study 3 RGB-D
collection happened here.

## Launch recipe

A plain `ssh newton 'nohup ... &'` **hangs the session**. Use:

```bash
ssh newton 'cd ~/NeDM && nohup env PYTHONPATH=src ~/anaconda3/envs/nedm/bin/python -u <script> \
  > /tmp/<name>.log 2>&1 < /dev/null & echo launched'
```

Deliver code by commit + push from your box, then `git pull --ff-only` here.
**Never edit files on newton directly.**

## Measured throughput

| Workload | Rate |
|---|---|
| Headless Chrono HMMWV episode | 15–25× slower than realtime |
| 20 Hz sim + RGB-D render, 256² | 1.30 frames/s ≈ 0.065× realtime |
| Traversal collection, 3 procs | ~124 s per 20 s episode, ~90% parallel efficiency (physics-bound; GPU contention negligible) |
| Study 3 pilot (200 × 20 s) | 6.5 h wall at 3 procs; p50 351 s, p95 414 s per episode |

Rendering adds little — **physics dominates**. Batch with ~12 workers for
headless CPU work. Measure GPU contention before going past ~3–6 render
processes.

## Gotchas

Disk is the binding constraint, not compute: `artifacts/` is already 542 G of
1.8 T. Check `du -sh` before starting a tier.
