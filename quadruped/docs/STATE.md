# State

**Updated:** 2026-09-20 (cost) · **Branch:** `kyle/quadruped-pipeline` (off `kyle/locomotion`)

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
`walk_check.py`, `collect.py`, `corpus_check.py`, `evaluate.py`, `patch_cost.py`,
`active_domain_study.py`. Not yet built: `train.py`, `finetune.py`, and Gate 3 in the
training path.

## The soil was not where the spawn rule thought it was

Found 2026-09-20 by a cost benchmark whose five configurations returned bit-identical
dynamics. `build_crm` centred the bed at `x = patch_x/2 - 0.6`, so an 8 m patch ran
x [-0.600, +7.400], while the collector spawned a +x episode at -3.500 -- the near edge of
a bed centred on the origin. The robot free-fell for the whole episode.

It would not have failed loudly. The offset is +x only, so backward-commanded episodes
spawn on the bed and collect normally while forward ones truncate below `min_segment_rows`
and vanish. The corpus would have been silently missing most of a vx [-1.0, 1.5] range
while passing every gate, and the symptom at full scale reads as "the policy cannot walk
on CRM". Fixed in `89c898b8`: `crm_patch_bounds()` is the single source of truth, the bed
is centred, and three guards were added -- built-bounds check, spawn-on-bed assertion, and
an `off_bed` validity check that fires at the edge rather than when `base_height` finally
trips.

**Blast radius: none, checked rather than assumed.** No CRM corpus had been collected yet
(`data/` holds only its README), so no data carries the fault. `walk_check.py` spawns at
the origin, which is on the bed under both the old convention and the new one, so the
rigid 95% and CRM 64% tracking figures stand. Only `collect.py`'s far-end spawn was
affected, and it had not yet been run on soil. The bug was caught one step before it
would have produced a corpus.

## Cost: patch length is nearly free

Measured 2026-09-20, `patch_cost.py`, full detail in `docs/COST.md`. Four times the
particles (444k -> 1.77M) costs 4.5% more per step, because every SPH kernel launches over
the compacted active set rather than over all markers. **The active domain is the only
cost knob**: 3.81x real time at 0.5 m, 6.18x at 1.0 m (the value inherited from Chrono's
Viper demo), 14.11x at 2.0 m, 36.95x with none at all.

That inherited 1.0 m is therefore worth a factor of 6, and is uncalibrated. It is also not
a free approximation -- outside the box a particle's velocity is zeroed every step -- so
`active_domain_study.py` is calibrating it against the unapproximated solve before it is
trusted.

The SCM-style moving patch exists (`ConstructMovingPatch`) and is the wrong tool: +x only
while our commands cover vy and wz, relocated soil is reset to zero stress and zero
velocity, and no demo combines it with an active domain. See `docs/COST.md`.

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
- Active-domain size, pending `active_domain_study.py`. This sets the cost of the entire
  corpus, so it is the last thing to settle before full-scale collection.
- Patch size, which is now nearly free and should be sized for the longest episode wanted
  rather than traded against cost.
- Whether soil ahead of the robot ever settles. `free_flow_duration` is 0.1 s and the
  active domain freezes everything outside it, so the robot may always be stepping onto
  settling rather than settled material. Untested, and systematic across the corpus if
  real.
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

## Running

sbel: `active_domain_study.py`, calibrating the active domain against a 2.0 m reference
with the noise floor measured from two identical runs per case. euler queue empty,
north/a3/d33 clear.
