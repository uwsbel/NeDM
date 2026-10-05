# Pool-ball NRD study — design (2026-10-02, draft before pilot)

Goal: port the bouncing-ball result (Transformer backbone + learned contact
switch + bounce network, all on the full state) to a two-ball pool system, then
use the trained model's gradients to choose ball A's launch `(vx, vy)` so that
ball B is at a target `(x, y)` after `t` seconds. All collection, training and
evaluation run on the AMD cluster.

## 1. Physical system in Chrono

| Item | Value | Reason |
|---|---|---|
| Ball radius, mass | 0.028575 m, 0.17 kg | standard pool ball |
| Table (cushion faces) | x in [-1.27, 1.27], y in [-0.635, 0.635] m | 9-ft table playing area |
| Ball A start (fixed) | (-0.635, 0) | head spot |
| Ball B start (fixed) | (0, 0) to be fixed by the pilot | centre spot; off-axis option if symmetric data is degenerate |
| Cushions | 4 fixed boxes, vertical faces, taller than the ball | contact at ball-centre height, no jump |
| Cloth (ball-floor) | sliding friction 0.2, rolling resistance length ~3e-4 m, spinning friction small, restitution 0 | slide-then-roll behaviour, slow rolling decay |
| Ball-ball | friction 0.06, restitution 0.95 | near-elastic, small throw |
| Ball-cushion | friction 0.2, restitution 0.8 | lossy rebound |
| Contact model | NSC (as in the ball study), per-pair material set in an add-contact callback | Chrono's default rule takes the minimum of the two materials, which cannot give cloth 0.2 and ball-ball 0.06 at once |
| Activation | ball-ball and ball-cushion contacts only when touching (ball study fix for premature impulses); floor contact kept persistent | the balls sit on the cloth the whole time |
| Physics step | 0.125 ms to start, checked by halving | |
| Recording | every 1 ms (state of both balls + per-pair contact forces) | lets training use any 1 ms phase of the 10 ms model grid |

Launch: A gets `(vx, vy)` with zero spin (centre-ball "stun" strike); B at
rest. A slides, picks up topspin, hits B; B slides then rolls, usually reaching
a cushion and coming back. No pockets.

Pilot checks before scaling: slide-to-roll speed ratio 5/7 and timing
`2 v0 / (7 mu g)`; rolling deceleration matches the rolling-resistance length;
ball-ball momentum and restitution; cushion restitution; balls stay on the
cloth (|z - R| and |vz| tiny); energy never increases; penetration small;
halving the physics step changes B's position by < 1-2 mm; wall-clock per
episode.

## 2. Model state and I/O

State per ball: `[x, y, vx, vy, wx, wy, wz]` (7). Full state `s` = A then B =
**14 numbers**. z and vz are dropped after the pilot shows they stay at R and 0.
Spin is needed: sliding vs rolling, follow after the hit, and cushion throw all
depend on it.

Model step: 10 ms (as in the ball study). Rollout to t = 2 s is 200 steps.

### Primary: literal port of the bouncing-ball winner (3 NNs)

```
s [14] -> Transformer (existing ContinuousTransformer) -> d_smooth [14]
s [14] -> contact MLP -> latent [32] -> p(any impulsive contact in next 10 ms) -> gate g in {0,1}
s [14] -> bounce NN (MLP + learned linear skip) -> d_contact [14]
s_next = s + d_smooth + g * d_contact          (fed back every step)
```

The Transformer learns everything smooth: rolling, sliding friction and the
slide-to-roll change, B sitting still. "Impulsive contact" means ball-ball or
any ball-cushion contact; the persistent cloth contact is not an event.

### Structured variant (5 NNs, uses the symmetry of the two balls)

```
s [14] -> Transformer -> d_smooth [14]
s [14] -> ball-ball contact MLP -> g_AB ; s [14] -> ball-ball bounce NN -> d_AB [14]
ball k's own state [7] -> cushion contact MLP (shared by A and B) -> g_k
ball k's own state [7] -> cushion bounce NN (shared by A and B) -> d_k [7]
s_next = s + d_smooth + g_AB d_AB + [g_A d_A ; g_B d_B]
```

The cushion networks see every A and B cushion hit, so they get twice the data.
They also generalise to more balls.

Controls: a Transformer with no contact branch (same training budget), plus
Transformer size/context changes.

No geometry, time-to-impact or physics formula enters inference. Contact labels
from Chrono supervise training only. Normalisation and data-fitted linear
initialisations use training transitions only (as in the ball study).

## 3. Data

Action sampling in polar form around the A->B line: speed `u` and aim angle
`theta`. A's path before impact is straight (zero-spin start), so the impact
offset is `b = D sin(theta)` and the cut angle is `asin(b / 2R)`. The angle
range keeps cut angles up to about 60-70 degrees, so every launch hits B. The
model and the optimiser still use `(vx, vy) = u (cos, sin)` of the aim
direction.

Stratified grid over (speed, angle) cells, with fixed per-cell
train/validation/test counts assigned before simulation (as in the ball
study). Target size about 20-25k episodes of 2.5 s (about 15 simulated hours).
After the models are frozen, a separate fresh test cohort is collected with a
new seed.

Quality checks: energy, penetration, z/vz, contact sequence per episode (A-B
hit exactly once is expected; second hits counted), physical-step
convergence, momentum at impact.

Diversity checks: coverage of the action grid, B's departure direction
histogram, number of B and A cushion hits by time t, scatter of B's position
at t (the reachable target set), and spin state of A at impact (sliding vs
rolling).

Training uses 10 ms transitions starting at every 1 ms phase, so each
collision gives about 10 distinct training transitions. Rollout evaluation
uses phase 0.

## 4. Targeting

Fixed t (about 2.0 s, chosen from the pilot so that B usually has hit a
cushion). Loss `L = |p_B(t) - target|^2`, differentiated through the frozen
NRD rollout with autograd. Projected gradient steps with backtracking (as in
the ball study), in normalised (speed, angle) coordinates. This is gradient
descent in `(vx, vy)` with a fixed rescaling, and the aim-angle range is about
50 times narrower than the speed range. Fixed multi-starts handle the several
launches that can reach one target. Targets: 10 Chrono positions from held-out
episodes. Every optimised launch is replayed in Chrono. Chrono is also checked
every 5 iterations to make the loss-versus-iteration video.

## 5. Metrics

Per episode, on the 10 ms grid: RMS position error of B and of A over the
rollout, B's error at t, end error. Also: the event sequence (A-B hit step,
cushion hit counts per ball), the contact switch scored on predicted rollout
states, and the tails (worst episode, count above 10 mm). Physical target
misses come from Chrono replay.
