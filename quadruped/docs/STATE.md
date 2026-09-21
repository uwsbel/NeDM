# State

**Updated:** 2026-09-20 · **Branch:** `kyle/quadruped-pipeline` (off `kyle/locomotion`)

## Where this is

The policy walks, on both terrains, through the new pipeline. Measured 2026-09-21,
`walk_check.py`, 0.5 m/s commanded:

```
  rigid   mean_z 0.342   upright +1.000   vx +0.473   tracking 95%
  crm     mean_z 0.504   upright +0.999   vx +0.321   tracking 64%
```

Base sits 0.292 m above the rigid surface and 0.304 m above the soil surface, so it is
standing on the bed rather than sinking into it.

**Collection is verified end to end** on rigid: 3 episodes, 0% truncation, 98.6% of rows
and 3 segments per episode kept, 0 push-active rows surviving into any segment, manifest
written with the Chrono hash. The OU sigma range is now calibrated by measurement rather
than guessed, and at the chosen setting the corpus carries 29.2% identifiable action
variance over an effective rank of 8.96, against the previous corpus's 4% over rank 2.

**That 95% against 64% is the headroom the whole study is about**, reproduced from scratch
with a different base policy than the previous work used. Fine-tuning has something to
close.

Built and self-tested so far: `doctor.py`, `lib/policy.py`, `lib/validity.py`,
`lib/provenance.py`, `lib/excitation.py`, the full `params/` layer, `establish_sign.py`,
`walk_check.py`. Not yet built: `collect.py`, `train.py`, `finetune.py`, `evaluate.py`.

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

- Corpus size, pending the calibration sweep. Episode length is 20 s, set by the push
  segmentation arithmetic in `excitation.yaml`.
- OU sigma range and correlation time, same.
- Whether soil parameters vary within a corpus or are fixed per corpus.

## Inherited, not yet ported

`crm_verdict.py`, the Chrono scene setup, the gravity and contact derivations. Everything
else from the old tree is being replaced, not wrapped.

## Fleet facts, probed 2026-09-20

- **NAS**: `/mnt/nas/Main`, 30 TB with 29 TB free, mounted on all four desktops. Artifact
  root for everything the clusters do not hold.
- **The conda Chrono trap is RESOLVED.** All four desktops now import their pinned source
  build, verified by hash: sbel `3b0bd530`, north `d1d0bd0a`, a3 `cfbf8af6`,
  d33 `53102025`. The conda package is removed from every env.
- **Env name standardised to `nedm`** on all four; sbel's `nedm-src` retired.
- **d33 is now trainable**: torch 2.10.0+rocm7.0, verified by a real GEMM forward and
  backward, not by `is_available()`.
- All four desktops have `chrono-src` at the correct pin `698282895`; only the import
  path is wrong.
- **euler default partition has zero nodes** -- an sbatch without `-p` goes nowhere.
- **hpcfund torch reports cuda_avail True and dies at the first nn.Linear.** Chrono only.
- numpy is split three ways and is ABI-locked to each host's pychrono. Do not unify.
- d33 (AMD 9070 XT, gfx1201, ROCm 7.2.4) is being given a ROCm torch; acceptance is a real
  GEMM plus backward pass, never `is_available()`.

## Nothing is running

Confirmed idle: euler queue empty, sbel/north/a3 clear.
