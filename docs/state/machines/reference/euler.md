# Euler cluster (UW–Madison)

> **NO ACCESS.** Kyle cannot log into this machine (verified 2026-09-02).
> Documented for context only — see [`README.md`](README.md). Work that requires
> this box is a blocker to escalate, not a step to attempt.

**Verified:** 2026-09-01 (from `.claude/skills/storage/SKILL.md`) ·
**Role:** State-only, CPU-scale data collection via SLURM arrays.

| | |
|---|---|
| Hardware | CPU array |
| Storage | cluster storage |
| Job submission | SLURM array jobs |

## What this machine is for

Collections that only make sense at cluster scale and need **no rendering**.
`scripts/cluster/` holds the array jobs; the `create-euler-script` skill
(`.claude/skills/create-euler-script/SKILL.md`) generates new ones.

## What it must not do

Anything requiring a GPU renderer — no Chrono::Sensor, no RGB-D. Those go to
`newton`.

## Unverified

This file is transcribed from the storage skill and has **not** been checked
against the live cluster in this doc's lifetime. Confirm the account, partition,
module set, and scratch quota before submitting, and update this file with what
you find.
