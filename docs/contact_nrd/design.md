# One contact-NRD architecture for the bouncing ball and pool (design, 2026-10-02)

> This is the first design (version 1). The final model (version 2: gravity-tied
> pair frame, relative geometry only, per-pair-type conditioning, exact
> collision labels) and all results are in `results.md`; the changes and why
> they were made are in `amendments.md`.

## Goal

Replace the per-system contact modules with one architecture that has the same
structure for any set of bodies. It has a Transformer core plus exactly two
contact networks:

1. **Collision network.** For every pair of bodies, is that pair in contact
   during the next model step (on/off)?
   - 3 bodies give 3 pair entries (A-B, A-C, B-C); n bodies give n(n-1)/2.
   - Pairs of two fixed bodies are always off and are not evaluated.
2. **Contact network.** For a pair in contact, how much does the contact
   change each body's state? It has multiple output channels: position,
   velocity and spin changes.

The same code and hyperparameters are trained on each system. A single set of
weights trained on both systems is a stretch goal.

## Bodies

Every contact partner is a body:

| System | Moving bodies | Fixed bodies | Candidate pairs |
|---|---|---|---|
| Bouncing ball | ball (sphere, R 0.1 m) | floor (plane z = 0, normal +z), wall (plane x = 5, normal -x) | ball-floor, ball-wall |
| Pool | A, B (spheres, R 28.6 mm) | cushions at x = ±1.27, y = ±0.635 (planes, inward normals) | A-B, A-each cushion, B-each cushion (9) |

The pool cloth touches the balls all the time. It is part of smooth motion
(the core), not a collision entry.

Each body has one **token** of the same layout in both systems:
`[is_sphere, is_plane, radius, position (3), plane normal (3), velocity (3), angular velocity (3)]`.
- For a moving body, position, velocity and spin are its predicted state (world frame, 3D).
- For a fixed plane, position is a point on the plane, the normal faces into the play area, and velocity and spin are zero.

The token describes the body, as a simulator's scene description would. It is
not a contact rule: nothing computes distances or impact times for the network.

Moving-body state: `[p (3), v (3), w (3)]` = 9 numbers in world coordinates.
- Bouncing ball: y = vy = wx = wz = 0.
- Pool: z = R and vz = 0.

Channels that are constant in a system's training data are held fixed by a
data-derived mask.

## Model (one step = 10 ms, fed back)

```
for each moving body i:
    d_core_i = Transformer(history of body i)                          [9]  (shared weights)
for each candidate pair (i, j), with i moving:
    q_ij = pair features(token_i, token_j, p_j - p_i, v_j - v_i)
    g_ij = 1[ collision_net(q_ij) >= 0 ]                              on/off (1 network)
    d_ij = contact_net(q_ij)        change of body i due to j     [9]  (1 network)
    (if j also moves: d_ji = contact_net(q_ji) for body j; g uses the mean of both logits)
s_i(next) = s_i + d_core_i + sum_j g_ij * d_ij
```

Two contact networks in total, whatever the number of bodies; both are shared
over all pairs.

**Proposed refinement (ablation): pair frame.**
- Express each pair's vectors in a frame tied to the pair: the first axis is the partner plane's normal, or the direction to the partner sphere.
- Rotate the contact network's output back to the world frame.
- One network then sees every cushion (and the floor and wall) in the same orientation, so cushions share data instead of competing.
- This is a coordinate choice, not a contact rule. The world-frame version is the control.

## Training (the pool recipe, made system-agnostic)

1. **Core.** Each moving body's 10 ms steps without its own contact (and not
   right after one), from every 1 ms phase. Sampling is balanced over
   quantile bins of the velocity change, which picks out regimes such as
   sliding vs rolling generically. Optional rest anchoring and rotation
   augmentation are set per system: pool on, ball off, because gravity
   makes a ball in the air accelerate.
2. **Collision network.** Binary cross-entropy over pair samples: contact
   pairs, the same pair within ±50 ms of its contact (near negatives), and
   random pairs.
3. **Contact network.**
   - Training windows: windows with a contact, and windows up to 30 ms before or after one.
   - In the 30 ms band, the pair's switch is forced on and the target is the true change (about zero). The correction then fades out at the edge of the contact region (the decisive pool fix).
   - Every body's next state is predicted with the true or forced switches.
   - Optimiser: Adam for 80k steps at batch 4096, no L-BFGS.
4. **Rollout refinement** of the contact network: 1 s rollouts, 2,000 updates,
   learning rate 3e-6, with the checkpoint chosen on validation.

## Evaluation

- **Selection:** on validation only.
- **Fresh sealed cohorts for both systems:** collected new for this comparison and read once, after freezing.
- **Pool:**
  - B's error at t = 2.0 s (all shots and target-worthy shots), path error, and event counts.
  - Comparison with the per-cushion model, on the same new cohort.
  - Levenberg-Marquardt targeting on 100 targets with Chrono replay; data baselines.
- **Bouncing ball:**
  - Trajectory RMSE over the episode, endpoint error, and contact order.
  - Comparison with the earlier case-specific Transformer models (scalar switch, shared bounce net; two switches, two bounce nets) on the same cohort.
  - Targeting at 1.7 s with Chrono replay.
