# Pool NRD model: networks, inputs/outputs, and how they connect

State `s` has 14 numbers, ball A then ball B, each
`[x, y, vx, vy, wx, wy, wz]` (position, velocity, spin). One model call
advances 10 ms. The predicted state is fed back for the next call. Rollouts
from the launch state `[A at (-0.635, 0) with (vx, vy), no spin; B at rest at
(0, 0)]` reach t = 2.0 s in 200 calls. Code: `src/nedm/pool_ball/model.py`.

## Selected design (per-cushion contact modules): 11 networks in 3 roles

![Pool NRD architecture](../../artifacts/pool_ball/architecture/pool_nrd_architecture.png)

| # | Network | Input | Output | Role |
|---|---|---|---|---|
| 1 | Transformer backbone (shared `ContinuousTransformer`, 4 layers x 128, context 1), run once per ball with the same weights | that ball's 7 numbers `[7]` | that ball's smooth change `[7]` | rolling, sliding friction, slide-to-roll, rest |
| 2 | Ball-ball contact MLP (256-256-latent 32) | full state `[14]` | p(A-B contact in the next 10 ms) `[1]` | switch |
| 3 | Ball-ball bounce NN (3 x 512 tanh + learned linear skip) | full state `[14]` | correction `[14]` (both balls) | what the hit does |
| 4-7 | Cushion contact MLPs, one per cushion (xp, xm, yp, ym), each shared by A and B | that ball's 7 numbers `[7]` | p(that ball hits this cushion in the next 10 ms) `[1]` | switch |
| 8-11 | Cushion bounce NNs, one per cushion, each shared by A and B | that ball's 7 numbers `[7]` | correction to that ball `[7]` | what the rebound does |

```
s_next = s + [smooth(A) ; smooth(B)]                       (network 1)
           + g_AB * d_AB(s)                                (2 switches 3)
           + sum over balls k and cushions c of g_kc * [d_c(s_k) in ball k's slots]   (4-7 switch 8-11)
g = 1 if probability >= 0.5 else 0;  9 switches per step (1 ball-ball + 4 cushions x 2 balls)
```

The Transformer's output is anchored at rest: smooth change =
`h(s_k) - h(s_k with velocity and spin set to 0)`. A ball at rest (B before the
hit) therefore stays exactly at rest. Without this, any small leak gives B a
drift, and the cut angle magnifies it 15-50 times by t. The anchor uses no
physics formula; it is a property of the network's output.

An intermediate design used one cushion module shared by all four cushions
(5 networks). It is kept as a control: at p95 its sealed-cohort error of B at
t is 20 mm, against 9 mm for per-cushion modules.

## Literal port of the bouncing-ball winner (scalar_shared): 3 networks

| # | Network | Input | Output |
|---|---|---|---|
| 1 | Transformer backbone (per ball, as above; or one 14-number token in the "joint" control) | `[7]` per ball (or `[14]`) | smooth change `[14]` |
| 2 | Contact MLP | full state `[14]` | p(any A-B or cushion contact in the next 10 ms) `[1]` |
| 3 | Shared bounce NN | full state `[14]` | correction `[14]` |

`s_next = s + smooth(s) + g * d(s)`

## Controls

- **Joint core:** one 14-number token through the Transformer, as in the ball study.
- **Transformer only:** no contact branch; the Transformer learns the collisions too.
- **Capacity:** 2 layers x 64 against 4 x 128; context 8 against 1.

## What is learned and what is not

Nothing in inference uses table geometry, time-to-impact, a friction law or
Chrono's contact labels. The switches decide contact from the predicted state
alone. Contact labels supervise training only. Input/output normalisation and
data-fitted linear initialisations use training transitions only.

## Training (staged, validation-only selection)

1. **Transformer core.** Data-fitted linear readout, then 50k Adam steps and
   200 L-BFGS steps on 10 ms steps where that ball has no contact. Steps are
   sampled across sliding (35%), slide-to-roll inside the step (20%) and
   rolling (45%), with random rotations about the vertical axis (the cloth
   behaves the same in every direction). The core is then frozen.
2. **Switches.** Balanced batches: contact steps, steps up to 50 ms before and
   after a contact, and random free steps. 30k Adam steps plus 100 L-BFGS
   steps.
3. **Bounce networks.** A ridge-regularised, data-fitted linear skip, then
   80k Adam steps at batch 4096 (learning rate 1e-3 decaying to 1e-6). The
   steps are half contact steps and half "near" steps up to 30 ms before and
   after each contact, with that module's switch forced on and the true change
   (about zero) as the target. The true contact labels switch the responses
   on. No L-BFGS: it wrecked the rollouts.
4. **Refinement.** 300 updates of half-second rollouts, then 2,000 more
   updates of one-second rollouts of the bounce networks (learning rate
   3e-6); every 100 updates are scored by validation rollouts.
5. **Core refinement** (the selected model's core): the stage-1 core trained
   another 100k Adam steps at batch 4096, learning rate 1e-4 decaying.

Every 1 ms phase of the 10 ms step is used as a training step (the dynamics
do not depend on the clock), so each collision gives about 10 distinct
examples. Checkpoints are chosen by validation rollouts from launch: the p95 of
B's error at t on target-worthy validation shots, plus half the p95 of B's
whole-path error, plus event failures.
