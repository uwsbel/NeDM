# The CRM "verdict drift" was two pychrono builds, not drift

**Status: RESOLVED 2026-09-16. Everything below supersedes the earlier contents of this
file, which claimed a real drift and then blamed the NVIDIA driver. Both claims were
wrong. No drift occurred, no machine regressed, and the -40% headline stands.**

## What actually happened

kyle-sbel carries two pychrono installs and `PYTHONPATH` decides which one a verdict
runs against, silently:

| build | md5 | identity |
|---|---|---|
| source, `/home/kyle/Documents/sbel/chrono-build/bin` | `3b0bd530d06f54a4` | Chrono SHA `698282895`, the project pin |
| conda, `miniconda3/envs/nedm/.../pychrono` | `8e9e386546fe0b33` | released `pychrono-10.0.0` |

Every verdict scored on sbel on 2026-09-15 used the **conda** build, because the `nedm`
env was active and `NEDM_CHRONO_PYTHONPATH` was unset. The September verdicts and every
cluster verdict used the **source** build.

Scoring the identical policy (`w_h15r0`, sha256 `20c084424570341b`) under each:

| Chrono build | mae_vx |
|---|---|
| conda `pychrono-10.0.0` | 0.1685 |
| source, SHA `698282895` | **0.0788** |

A 53% swing with nothing else changed. That is the whole of the "drift".

## The claims this file previously made, and why each was wrong

**"BASE drifted 21.8% in five days."** BASE was 0.1541 in September under the source
build and 0.1877 on 2026-09-15 under the conda build. Two builds, not two eras.

**"The NVIDIA kernel module changed from 595.84 to 595.91.07."** The timeline was
suggestive and the argument was already withdrawn once: the 2026-09-11 upgrade included
`libnvidia-compute-595`, so a stale module would have broken CUDA outright rather than
subtly altering physics. The driver was never involved.

**"The fine-tuned policy is ~4x more sensitive to the environment than BASE."** The
asymmetry is real but it is a property of the two Chrono builds, not of drift: the
specialised policy is more sensitive to the *simulator implementation* it is scored in.
That remains an interesting observation and is the one thing worth keeping.

**"sbel has a machine-specific regression."** It does not. sbel reproduces the September
numbers as soon as the correct build is used.

## What confirmed it

- Same policy, same machine, same day, only the build swapped: 0.1685 -> 0.0788.
- euler, building from the same source pin `6982828`, scored BASE and `w_h15r0`
  **simultaneously** on four A100s in one allocation: **-39.6% on mae_vx, 65/75 episodes,
  sign p = 5.2e-11**, against the recipe's documented -40.5% / 67/74. The headline
  reproduces on independent hardware.
- The policy, the fine-tune, and the evaluation code were each excluded by direct test
  before the build was suspected: sha256 identical, stopping point identical
  (update 94, ||dW|| 1.002, val -1.438482), and two repo commits scored the same.

## What was put in place

`score_crm_tracking.py` now probes the pychrono its episode subprocess will resolve,
prints the path and md5, stamps `chrono_md5`/`chrono_path` into every record, and exits
FATAL when `NEDM_CHRONO_PYTHONPATH` is set but the import resolves elsewhere. The
canonical per-machine environment is in
[`../machines/scoring-environment.md`](../machines/scoring-environment.md).

## The lesson worth keeping

This trap was **already documented**, in a careful docstring in
`run_go2_finetune_verdict.py`, saying that an unset `NEDM_CHRONO_PYTHONPATH` silently
selects the conda build and breaks the bit-exact replay the paired design rests on. It
was read, understood, and fallen into regardless.

An environment that can be wrong silently will eventually be wrong. The fix is the guard,
not the prose. Two further habits that would each have caught this within minutes:

- **Score BASE in the same session as every arm.** A same-session BASE would have shown
  0.1877 immediately and pointed at the environment rather than at time.
- **Record the environment in the artifact.** The verdict already stamped the policy
  hash; had it also stamped the Chrono md5, this would have been a one-line diff instead
  of a day.
