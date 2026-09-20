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

## Fleet facts, probed 2026-09-20

- **NAS**: `/mnt/nas/Main`, 30 TB with 29 TB free, mounted on all four desktops. Artifact
  root for everything the clusters do not hold.
- **The conda Chrono trap is live on all four desktops.** They import
  `8e9e386546fe0b33`, not the pinned source build. See `STANDARD.md`.
- All four desktops have `chrono-src` at the correct pin `698282895`; only the import
  path is wrong.
- **euler default partition has zero nodes** -- an sbatch without `-p` goes nowhere.
- **hpcfund torch reports cuda_avail True and dies at the first nn.Linear.** Chrono only.
- numpy is split three ways and is ABI-locked to each host's pychrono. Do not unify.
- d33 (AMD 9070 XT, gfx1201, ROCm 7.2.4) is being given a ROCm torch; acceptance is a real
  GEMM plus backward pass, never `is_available()`.

## Nothing is running

Confirmed idle: euler queue empty, sbel/north/a3 clear.
