# State

**Updated:** 2026-09-20 · **Branch:** `kyle/quadruped-pipeline` (off `kyle/locomotion`)

## Where this is

Rebuild, day zero. The plan is written (`PLAN.md`) and approved. No pipeline code exists
yet. Nothing is running.

## What is decided

| decision | value | why |
|---|---|---|
| base policy | rl_sar `robot_lab/policy.pt` | plain rsl_rl MLP, Identity normaliser, `observations_history: []` -- memoryless |
| excitation default | OU per-joint action injection ON, pushes ON | the action derivative is unidentifiable without injection |
| push handling | episode SPLIT at each push, active window dropped | the recovery is admissible, the force window is not |
| push direction | uniform over the sphere | current code is a +/-16.7 degree disc |
| collection host | hpcfund preferred, euler fallback | one node = one CRM episode; hpcfund cannot train |
| data/models in git | manifests only | `.git` is already 502 MB; one checkpoint is 911 MB |

## What is NOT decided

- Corpus size and episode length, pending the calibration sweep.
- OU sigma range and correlation time, same.
- Whether soil parameters vary within a corpus or are fixed per corpus.

## Inherited, not yet ported

`crm_verdict.py`, the Chrono scene setup, the gravity and contact derivations. Everything
else from the old tree is being replaced, not wrapped.

## Nothing is running

Confirmed idle: euler queue empty, sbel/north/a3 clear.
