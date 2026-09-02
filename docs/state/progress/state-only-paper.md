# State-only NRD — the published paper

**Status:** Submitted · **Updated:** 2026-09-02 · **Branch:** `main`

*Learning the Right Abstraction: Neural Reduced Dynamics for Complex Robot
Control* (Zhang and Negrut), arXiv:2608.19375v1 [cs.RO], 19 Aug 2026,
submitted to Elsevier. Source and build: [`../machines/manuscript.md`](../machines/manuscript.md).

## Content

Two case studies, three control tasks:

- **Study Case I** — terrain-aware HMMWV trajectory tracking on rigid, bumpy and
  deformable CRM terrain. 15-D reduced state (body motion, attitude, body rates,
  8-channel terramechanics block of tire normal loads and wheel speeds) plus a
  2-D terrain code. One policy trained inside the conditioned model beats both
  single-terrain specialists on all three terrains, including zero-shot bumpy.
- **Study Case II** — M113 tracked vehicle with a front-mounted 4-DOF arm.
  3-D planar state for base goal reaching (**100/100** at 0.75 m); 8-D
  joint-space state for arm end-effector reaching (**97/100** at 0.05 m, zero
  contacts, zero joint-limit violations), end-effector recovered by forward
  kinematics rather than learned.

NRD models advance ~4 orders of magnitude faster in simulated time than the
Chrono scenes they replace.

Full per-number reproduction record: [`docs/progress.md`](../../progress.md).
Datasets are on Hugging Face (`harryzhang1018/NeDM`, 70 GB) — see
[`../data/README.md`](../data/README.md).

## What is next

Nothing blocking. Two loose ends recorded in
[`../machines/manuscript.md`](../machines/manuscript.md): the shared `BibFiles/`
directory is missing so citations render `[?]`, and the figure copy/rename step
between `artifacts/analysis/manuscript_figs/` and the manuscript's `ImageArchive`
is not checked in anywhere.

## Structural gap a reviewer will find

Every contact in the paper is **continuous** — permanent tire contact, permanent
track contact, and an arm whose whole safety story is about *avoiding* contact.
There is no intermittent-contact system and no case where the reduced state must
carry something that is not a rigid-body coordinate. This is the gap the
proposed case studies in [`future-case-studies.md`](future-case-studies.md)
target.
