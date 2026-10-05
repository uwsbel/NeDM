# Minimal state and per-network inputs: plan (2026-10-04, 03:40 CDT, before results)

The user's ideas:
1. Minimal state with integrated positions:
   - the networks predict velocity and spin changes (the degrees of freedom: ball vx, vz, spin about y; pool vx, vy and 3 spins per ball);
   - positions are integrated from velocity.
2. Give each network the information it needs:
   - the collision network gets position-level inputs;
   - the contact network works at the impulse level (velocity and spin in and out).
3. Everything stays derived from the state only.

**Corrections recorded with the user.**
- The networks already see only these numbers: masked constant channels contribute
  nothing. The real change in idea 1 is integrating position instead of predicting it.
- An impact part-way through a 10 ms step moves the ball differently from the
  trapezoid of the old and new velocity: up to about 1 cm in pool and 9 cm for the
  ball. So one arm adds a contact-network position correction.

**Free steps, measured** (position error per step, p95):

| | Trapezoid of the true velocities | The core's learned position change |
|---|---|---|
| Pool | 1.3e-7 m | 1.9e-6 m |
| Ball | 6.4e-6 m | 3.7e-7 m |

Chrono's 0.5 ms steps shift the ball's flight from the exact formula.

**Common recipe (from the best state-only run so far):**
- joint rows for pool (routed for the ball, which has one body);
- history 1 unless stated;
- contact steps only (no far steps), no soft limit;
- exact labels, 160k Adam updates with cosine decay, then refinement (no futility rule);
- frozen version-2 cores.

## Arms (seed 61; pool decides, ball reported)

| Arm | Positions | Collision input | Contact input | Contact output |
|---|---|---|---|---|
| p0 (baseline; + seeds 62, 63) | predicted | full state | full state | position, velocity, spin |
| p1 | integrated | full state | full state | velocity, spin only (idea 1, literal) |
| p2 | integrated | full state | full state | velocity, spin + position correction |
| p3 | integrated | positions, current + previous step | full state | as p2 |
| p3b | integrated | positions, current step only | full state | as p2 |
| p4 | integrated | full state | velocity and spin only | as p2 |
| p5 | integrated | positions, current + previous | velocity and spin only | as p2 (idea 2, full) |

- Ball arms: b0, b1, b2, b4, b5 (same meanings).
- Later: the best pool arm trained with 4x the training shots (57,600 new shots collected
  tonight with a new seed, same launch grid, training split only). Same validation set.

## Decision and milestone

- Rank on the pool validation score.
- **Milestone:** a state-only pool model within 1.25x of the deployed pair-feature
  model on validation (score at most 0.0110; deployed 0.0088), then confirmed on a new
  fresh cohort with the same paired bootstrap as before.
- The ball must not get worse than its best state-only result (fresh path p95 0.93 mm,
  8-step Transformer) by more than 25 % for the same arm, or the arm is flagged.

## Fixes after the pre-launch review (04:05 CDT)

An independent review (two reviewers, each finding checked by a third) confirmed two defects.

1. **Integration rule for the ball.** Chrono advances the ball in semi-implicit Euler
   sub-steps of h = 0.125 ms, so the exact position change over a 10 ms step is
   0.5·dt·(v + v′) + 0.5·h·(v′ − v). The trapezoid alone was 6.1 µm off per free
   step, which builds a fixed drift of about 1 mm at the 1.7 s target even with
   perfect velocities.
   - **Fixed.** The helper reproduces Chrono's sub-steps to 1e-17 m.
   - The ball arms b1, b2, b4 and b5 were restarted (the first attempts are kept as `*_trapezoid_attempt`).
   - Pool: h = 0.0125 ms, the term is about 1e-7 m per step, so the running pool arms continue. They were trained with the trapezoid alone and are evaluated with the exact rule; the difference is negligible.
2. **Checkpoint tag.** Checkpoints with the new options now carry the tag
   `contact_graph_nrd_v4`, so an older copy of the code refuses them instead of
   rolling them out the old way. Certification will use a fresh code copy.

A third reported issue was refuted.

## Mid-round additions (04:15 CDT, after seeing 100k-update validation scores)

**Seen** (pool validation score at about 100k of 160k updates):

| Arm | Score at 100k |
|---|---|
| p0, seeds 61-63 | 0.025-0.027 |
| p2 | 0.025 |
| p1 | 0.10-0.11 |
| p3 | 0.043 |
| p3b | 11-90 (collision from positions alone: 50-320 false alarms and 6-23 misses per ~2,200 contacts per pair) |
| p4 | diverging (above 1) |
| p5 | diverging (above 1) |

Ball:
- b2 0.0033 against b0 0.0026 at 50k;
- b1 0.076, flat;
- b4 0.4-0.5;
- b5 exploded.

**Stopped:** p3b, p4 and p5 (clearly failed; best-so-far checkpoints kept).

**Added:**
- `d0` / `d2`: p0 / p2 plus the A-B difference of position and velocity as an extra input (6 numbers, derived from the state). Diagnostic, not a candidate.
- `p6`: p0 with 8-step history through causal Transformers (the best ball recipe), now without far steps.
- `x4_p0` / `x4_p2`: p0 / p2 trained on 4x the training shots (data/pool_train_x4: original + 57,600 new shots, same validation set).

## Fresh test for this round (fixed 05:45 CDT, before any checkpoint is chosen)

- New pool cohort: 4,800 shots (4 per cell of the 30 x 40 launch grid), seed 202610045, same physics and collector. It is collected now and not read until every checkpoint of this round is frozen.
- Compared on it, with the pool study's own evaluator and the paired bootstrap:
  - the deployed pair-feature model, single and as an ensemble of its three seeds;
  - the best state-only single model by validation;
  - the state-only ensemble of three seeds;
  - the integrated-position variant (the user's idea 1) as a single model and as an ensemble;
  - the A-B difference diagnostic.
- The headline is state-only against pair features, like for like: single vs single and ensemble vs ensemble.
