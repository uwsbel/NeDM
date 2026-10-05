# One contact NRD for the bouncing ball and pool: results (2026-10-02)

Pre-registration: `preregistration.md` (SHA recorded on AMD at 10:47 CDT).
Every change after it: `amendments.md` (10 entries; each one that touches a
reported claim was written before that system's fresh test data existed). Design: `design.md`. Architecture figure:
`artifacts/contact_nrd/architecture/unified_contact_nrd_architecture.png`.
Cluster root: `/work1/dannegrut/harry/experiments/contact_nrd_20261002`.

## Summary

- **One architecture, both systems.** A Transformer core plus exactly two
  contact networks: a collision network (on/off for every body pair) and a
  contact network (the change of a body's state caused by a partner, 9
  channels). Both contact networks are shared by every pair. The same code
  runs the ball (2 pairs) and pool (9 pairs). Fixed bodies are described by
  tokens, so a scene is data, not architecture.
- **Pool: beats the earlier best.** On 4,800 fresh shots it beats the
  11-network per-cushion model: B at 2 s is 1.68 against 1.85 mm (median)
  and 7.1 against 9.1 mm (p95). It also has fewer wrong events and fewer
  shots over 1 cm. All three seeds have the lower p95.
- **Ball: close to the earlier best, inconclusive by the pre-registered rule.**
  On 1,800 fresh shots the path error is 0.62 against 0.50 mm (median) and
  1.10 against 0.89 mm (p95) for the two-switch Transformer. The ratio's 95 %
  range, 1.20 to 1.27, straddles the 1.25 "matches" bar. The unified model
  has no shot over 1 cm; the earlier model has 34 (worst 463 mm). It beats
  the one-switch design. A physics-prior model is still the most precise
  (0.80 mm p95).
- **Headline (weakest cell): inconclusive.** One seed per reference, and the
  unified model still needs per-system settings (loss scales, rest anchoring,
  refinement schedule).
- **Targeting through the unified model works on both systems** (Chrono replay):
  - ball: 50 of 50 targets within 3 mm (median 0.59 mm);
  - pool: 99 of 100 within 1 cm (median 1.67 mm), as good as the earlier pool model.
  - With fixed starting positions and dense training grids, interpolating
    training launches is still as good or better on both systems.
- **Generalisation (exploratory).** On a mirrored pool layout never seen in
  training, the unified model stays at 1.7 mm median and 9.4 mm p95 at 2 s.
  Every earlier pool model fails (about 1 m). With the ball's wall moved,
  the unified model, given the new wall, is 36 mm off (median); the earlier
  models, which cannot be given it, are 0.7 m off.
- **Biggest lesson: exact collision labels.** Switching contact on one step
  early (the pre-registered recipe) breaks impulsive contacts. The on/off
  decision must carry the jump.

## The architecture (same code for both systems)

Three neural networks, whatever the number of bodies:

1. **Transformer core.** One set of weights, run on each moving body's state
   (position, velocity, spin; 9 numbers). It predicts the smooth 10 ms change
   (gravity, rolling, spin decay).
2. **Collision network.** One network shared by every body pair. For each pair
   it says on or off for the next 10 ms. With 2 bodies it gives 1 answer, with
   3 bodies 3 answers (A-B, A-C, B-C), and so on. Pairs of two fixed bodies are
   never asked.
3. **Contact network.** One network shared by every pair. For a pair that is
   on, it gives the change of body i's state caused by body j, in 9 channels
   (position, velocity, spin). If j also moves, the same network gives j's
   change.

Next state of body i = state + core change + sum over its pairs of
(on/off x contact change).

The minor improvements on the two-network idea, all arithmetic (no extra
networks, no contact rules):
- **Every partner is a body with a token** (sphere or plane, radius, point,
  normal, velocity, spin). A floor, a wall and a cushion are planes; a ball is
  a sphere. Moving the wall means changing its token, not retraining.
- **Pair frame.** Each pair's inputs are expressed in a frame tied to the pair
  (first axis toward the partner; the vertical from gravity). The contact
  network's output is rotated back. One network then sees the floor, the wall
  and the four cushions in the same orientation.
- **Pair type** (sphere-sphere or sphere-plane) gets its own input scaling and
  a learned scale-and-shift inside the one contact network.

Per system: ball 1 moving body, 2 pairs (floor, wall); pool 2 moving bodies,
9 pairs (A-B, and A and B with each of 4 cushions).

## What mattered most: exact collision labels

The pre-registered recipe widened the collision labels by one 10 ms step, so
the collision network switched on one step early. All three pre-registered
ball variants failed on validation (path p95 0.26 to 6.7 m). The reason is
physical: a bounce is a near-instant impulse. A step that ends just before
the impulse and one that just contains it start from almost the same state
but end very differently. With widened labels, the contact network has to
produce "no change" and "full bounce" from nearly identical inputs, and a
smooth network cannot. With exact labels, that jump moves into the on/off
decision, which can be sharp.

| Ball, seed 61, validation | Path p95 | Contact events right | Selection score |
|---|---|---|---|
| widened labels + 30 ms band (pre-registered) | 255 mm | 98.8 % | 0.596 |
| widened, band without the last 2 ms | 866 mm | 98 % | 2.04 |
| widened, no band | 6.7 m | 90 % | 14.7 |
| **exact labels** | **1.08 mm** | **100 %** | **0.0021** |

The same change helped pool (centerline frame, seed 61, B's error at 2 s on
target-worthy shots, p95, validation, at 20k updates): 22 mm with exact
labels (revised recipe) against 50 mm with widened labels.

Two other recipe faults were found and fixed the same way in both systems:
- an Adam phase after the core's L-BFGS undid it, so the best phase is now kept;
- the contact network's L-BFGS polish made no progress, so it was dropped.

## Ball (fresh cohort: 1,800 shots, collected after the ball freeze, seed 3106852939)

Deployed model: `v2_ball_exact_s61`, the validation-best seed of the chosen
variant. The same 1,800 shots and the bouncing-ball study's own evaluator are
used for every row.

| Model | Path median | Path p95 | Endpoint p95 | Worst endpoint | Shots over 1 cm | Bounce order right |
|---|---|---|---|---|---|---|
| **Unified (3 networks)** | 0.62 mm | 1.10 mm | 2.10 mm | 3.2 mm | **0** | 100 % |
| Two switches, two bounce networks (earlier best Transformer) | 0.50 | 0.89 | 1.56 | 463 mm | 34 | 100 % |
| One switch, shared bounce network (your earlier design) | 0.67 | 1.53 | 3.24 | 7.2 | 0 | 99.9 % |
| Transformer + analytic contact timing (physics prior) | 16.5 | 31.0 | 59.0 | | 1791 | 100 % |
| Analytic flight + small network (physics prior) | 0.48 | 0.80 | 1.34 | 2.3 | 0 | 100 % |
| Unified, first version of this session, pair frame (control) | 4.6 | 13.7 | 30.9 | | 843 | 99.9 % |
| Unified, first version, world frame (control) | 20.2 | 57.5 | 141 | | 1710 | 100 % |

**Pre-registered verdict against the earlier best Transformer (paired bootstrap):**
- Median ratio 1.23 (95 % range 1.20 to 1.27): **inconclusive**. The upper end is just above the 1.25 "matches" bar.
- p95 ratio 1.23 (1.18 to 1.28): **inconclusive**.
- Shots over 1 cm: 0 against 34 (exact paired test p = 1e-10), in the unified model's favour.
- Against the one-switch design: **beats** (median 0.93, 0.90 to 0.96; p95 0.72, 0.67 to 0.77).
- Against the analytic-flight physics prior: worse at p95 (1.38, 1.33 to 1.42).

**Seeds.** The chosen setup's three seeds reach path p95 1.10, 4.04 and
1.59 mm on the fresh shots. Two of the three seeds have some event or tail
failures (seed 62: 9 shots over 1 cm; seed 63: 45). The revised recipe
(cosine decay, 160k updates) gives 1.10, 4.29 and 1.33 mm. The deployed
number is the validation pick, not a typical seed.

**Targeting at 1.7 s** (50 seeded targets from the fresh shots; every launch
replayed in Chrono):

| Method | Within 3 mm | Median miss | Worst miss |
|---|---|---|---|
| **Levenberg-Marquardt on the unified model** | **50 / 50** | 0.59 mm | 1.89 mm |
| Gradient descent on the unified model | 50 / 50 | 0.51 mm | 1.79 mm |
| Local linear fit over the 12 nearest training launches | 49 / 50 | 0.39 mm | 5.06 mm |
| Nearest training launch | 0 / 50 | 17.5 mm | 46.2 mm |
| The target's own launch (Chrono repeatability check) | 50 / 50 | 0.0001 mm | |

With 5,400 densely gridded training launches, a local linear fit is a
strong competitor for the ball. The model's targeting is more consistent:
its worst miss is 1.9 mm against 5.1 mm.

Figures:
- `artifacts/contact_nrd/figures/ball_example_shots.png`: each model's worst
  fresh shot and two typical ones. The earlier model's worst shot misses the
  last floor contact and falls through the floor; the unified model is
  1.5 mm off on that shot.
- `artifacts/contact_nrd/figures/ball_collision_onoff.png`: on/off per pair
  over one shot, the unified model rolled out on its own, against Chrono.
- `artifacts/contact_nrd/ball_targeting/iterations_chrono_lm.mp4` and
  `iteration_vs_miss_lm.png`: the first 10 targets, Chrono replays of each
  saved iteration. All 10 are under 3 mm by iteration 15.

**Exploratory: wall moved from 5.0 m to 4.5 m** (200 shots; the unified
model is told the new wall position through its token; earlier models cannot
be told):

| Model | Path median | Path p95 | Endpoint median |
|---|---|---|---|
| Unified (deployed) | 36 mm | 67 mm | 93 mm |
| Earlier models (all four) | 695-701 mm | 754-767 mm | 954-984 mm |

The unified model goes most of the way but not to millimetres.

Follow-up (exploratory, after the freeze; amendment 9): a core that never
sees absolute position (free motion does not depend on it in either system).
- Validation: score 0.00227, the same as the deployed model.
- Fresh shots: path p95 1.15 mm against 1.10 mm.
- Moved wall: median path error 27 mm against 36 mm; p95 69 mm against 67 mm.

So absolute position in the core explains only part of the gap. The rest comes
from the wall impact itself. With the wall nearer, the ball meets it earlier
in its flight, while still rising faster:

| Vertical speed at the wall impact | Training shots | Moved-wall shots |
|---|---|---|
| median | 0.8 m/s | 1.8 m/s |
| p95 | 3.0 m/s | 3.8 m/s |
| max | 3.7 m/s | 4.4 m/s |

More than 5 % of the moved-wall impacts are faster than any impact in
training. That is extrapolation for the contact network, which no change of
coordinates removes. The fix is training data that covers the impact
conditions (or several wall positions), not a new architecture.

## Pool (fresh cohort: 4,800 shots, collected after the pool freeze, seed 4104008808)

Deployed model: `v2_pool_centerline_exactdecay_s61` (exact labels,
centerline pair frame, revised recipe; validation-best of three seeds). The
same 4,800 shots and the pool study's own evaluator are used for every row.
"Target-worthy" is the pool study's predeclared filter.

| Model | B at 2 s median | B at 2 s p95 | Target-worthy p95 | B path p95 | Events right | Shots over 1 cm at 2 s |
|---|---|---|---|---|---|---|
| **Unified (3 networks), deployed** | **1.68 mm** | **7.11 mm** | **6.93 mm** | **4.07 mm** | **99.35 %** | **136** |
| Unified, seed 62 / seed 63 | 1.67 / 1.72 | 8.76 / 8.64 | 8.48 / 7.65 | 4.80 / 4.83 | 99.2 / 99.1 % | 198 / 187 |
| Per-cushion model (earlier best; 11 networks) | 1.85 | 9.10 | 8.40 | 5.53 | 98.90 % | 209 |
| Shared-cushion model | 3.85 | 22.0 | 14.6 | 10.1 | 98.0 % | 814 |
| Literal port of the ball design | 48.4 | 356 | 184 | 142 | 77.3 % | 4484 |
| Transformer only (no contact networks) | 350 | 756 | 645 | 421 | 0 % | 4793 |
| Unified, registered recipe (widened labels) | 3.23 | 13.7 | 10.4 | 8.72 | 98.75 % | 403 |
| Unified, first version, pair frame (control) | 3.10 | 13.2 | 11.8 | 7.62 | 98.6 % | 378 |
| Unified, first version, world frame (control) | 13.4 | 48.1 | 50.5 | 27.3 | 95.4 % | 3103 |

**Pre-registered verdict against the per-cushion model (paired bootstrap):**
- B at 2 s, median: ratio 0.91 (95 % range 0.88 to 0.93), **beats**.
- B at 2 s, p95: ratio 0.78 (0.72 to 0.86), **beats**.
- B path: median 0.85 (0.82 to 0.87), p95 0.74 (0.67 to 0.82), both **beat**.
- Events wrong: 31 against 53 (exact paired test p = 0.004).
- Shots over 1 cm: 136 against 209 (p = 1e-5).
- Every other reference: beats, by large margins.
- Seeds 62 and 63 also have a lower p95 than the per-cushion model (8.8 and 8.6 against 9.1 mm). Their medians are the same as seed 61's.

**Targeting B to a point at t = 2 s** (100 targets from the fresh shots with
the pool study's filter, half with a B cushion hit before t; every chosen
launch replayed in Chrono):

| Method | Within 1 cm | Within 5 mm | Median miss | Worst miss |
|---|---|---|---|---|
| **Levenberg-Marquardt on the unified model** | **99 / 100** | 94 | 1.67 mm | 10.2 mm |
| Gradient descent on the unified model | 89 / 100 | 78 | 2.01 mm | 54.4 mm |
| Local linear fit over nearby training launches | 100 / 100 | 100 | 0.23 mm | 3.9 mm |
| Nearest training launch | 96 / 100 | 73 | 3.30 mm | 11.7 mm |
| The target's own launch (Chrono repeatability) | 100 / 100 | 100 | 0.003 mm | 0.43 mm |

This matches the earlier pool study's targeting with the per-cushion model
(99 / 100 within 1 cm, median 1.4 mm). The model is the only part of the
pipeline that changed. As in that study, the fixed starting positions and
the dense training grid (24,000 launches over 2 launch numbers) make
interpolation of training launches more precise than optimising through any
model. A model only pays off once the starting positions vary.

Video and figure:
- `artifacts/contact_nrd/pool_targeting/iterations_chrono_lm.mp4`: the first 10 targets, the Chrono replay of every 5th iteration.
- `artifacts/contact_nrd/pool_targeting/iteration_vs_loss_lm.png`: all 10 are under 1 cm by iteration 20.

Figures:
- `artifacts/contact_nrd/figures/pool_example_shots.png`: each model's worst
  fresh shot and two typical ones (table top view).
  - The unified model's worst shot (244 mm at 2 s) is a corner shot, where B
    meets two cushions almost at once; the per-cushion model is 44 mm off on it.
  - The per-cushion model's worst shot (323 mm) is a single-cushion shot; the
    unified model is 7.9 mm off on it.
- `artifacts/contact_nrd/figures/pool_collision_onoff.png`: all 9 pair
  switches over one shot, the unified model rolled out on its own.
  - It catches A-B, B with the +x cushion, A with the -y cushion and B with
    the +y cushion at Chrono's times.
  - It stays on for one extra 10 ms step after the last cushion hit. The step
    right after a contact is left out of the collision network's training
    negatives (the penalty-contact lag), so its on/off there is untrained.

**Exploratory: mirrored shots** (400 shots, collected before the freeze and
not read until after it; amendment 3). A starts at (+0.635, 0) instead of
(-0.635, 0) and is shot toward -x, so B leaves toward the -x cushion. No model
was trained on this layout. B's error, mm:

| Model | Path median | Path p95 | At 2 s median | At 2 s p95 |
|---|---|---|---|---|
| **Unified, deployed** | **0.93** | **6.1** | **1.72** | **9.4** |
| Unified, seeds 62 / 63 | 0.88 / 0.90 | 4.3 / 5.1 | 1.61 / 1.65 | 7.9 / 9.6 |
| Unified, first version, pair frame | 1.86 | 9.6 | 3.07 | 16.0 |
| Unified, first version, world frame | 562 | 928 | 850 | 1280 |
| Per-cushion model (earlier best) | 749 | 930 | 1026 | 1278 |
| Shared-cushion model | 749 | 930 | 1026 | 1278 |
| Literal port of the ball design | 731 | 910 | 989 | 1242 |
| Transformer only | 1117 | 1252 | 1463 | 1937 |

The unified model keeps nearly its in-range accuracy (validation at 2 s:
1.6 mm median, 7.0 mm p95). Every earlier model fails. The per-cushion and
shared-cushion models give identical errors, which suggests they never
register A hitting B from this side, so their B stays where it started. The
world-frame unified model fails too. The pair frame (one contact network that
sees every cushion and the other ball in the same orientation), together with
the core's rotation augmentation, is what carries the model to an unseen
layout.

## Limitations and deviations

- **Ten amendments.** The pre-registered recipe failed on the ball, and the
  chosen variants come from amendments (exact labels, revised schedule).
  Every amendment behind a reported claim was written before that system's
  fresh data existed, and the fresh cohorts were read once. Amendment 9 (the
  position-free core) came after the ball results and is labelled
  exploratory. Still, the recipe was tuned on
  validation after the pre-registration, and that counts against the
  strength of the claim.
- **Per-system settings remain:**
  - loss scales;
  - rest anchoring and rotation augmentation (pool on, ball off);
  - refinement schedule;
  - the ball uses constant-rate Adam for 80k updates, pool the cosine
    schedule for 160k. The two were within 3 % of each other on the ball.
- **Each system has its own weights.** One set of weights for both systems
  is not attempted.
- **Seed spread.**
  - Ball: one of three seeds (seed 62) is 4 times worse at p95 (4.0 mm).
  - Pool: the three seeds are close (p95 7.1, 8.8, 8.6 mm).
- **References are one seed each,** trained in their own studies.
- **Time guards.** Pool seeds 62 and 63 stopped their refinement at update
  1,300 of 2,000 (time guard). Ball `exact` seed 61 was stopped in its
  (useless) L-BFGS polish; its checkpoint is the end-of-Adam one.

## What I would do next

1. **One set of weights for both systems.** The contact networks already see
   only pair-frame quantities. The core would need to know whether a body
   rolls on cloth or flies; that information could come from its token.
2. **Wider impact conditions in training** (several wall positions or launch
   heights), so the ball generalises to moved walls. The architecture
   already accepts the new scene.
3. **Corner shots in pool** (two cushions within one step) are the unified
   model's worst case. More corner data, or a 5 ms step near contacts.
