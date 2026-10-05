# One connected bouncing-ball NRD model

State `s(t) = [x,z,vx,vz,omega_y]` has five components. Each model predicts one
next state after10 ms. Colored numbered boxes are network components; threshold,
multiplication, summation and feedback are operations around those components.
Transformer embeddings, output heads and learned linear skips stay inside their
network boxes.

The shared version has three NNs: Transformer, contact MLP, shared bounce NN.

![Shared bounce: three NNs forming one NRD model](../artifacts/analysis/bouncing_ball_architecture/shared-bounce-network.png)

| Network | Input | Output |
|---|---|---|
| Transformer, including output head | Current state `[5]` | Ordinary-flight delta `[5]` |
| Contact MLP, including probability head | Current state `[5]` | Any-contact probability `[1]` |
| Shared bounce NN, including learned linear skip | Current state `[5]` | Bounce correction `[5]` |

All three read the same current state in parallel. The contact probability becomes
a binary gate at0.5. It controls whether the bounce correction is added; it is not
passed into the shared bounce NN as an input. The shared NN learns different
responses from the state itself.

```text
g = 1[p_contact >= 0.5]
s(t+1) = s(t) + delta_flight(s(t)) + g * delta_bounce(s(t))
```

The two-bounce version has four NNs in three functional roles: Transformer,
contact MLP, and a bounce block containing separate ground/wall NNs.

![Two bounce networks: four NNs forming one NRD model](../artifacts/analysis/bouncing_ball_architecture/two-bounce-networks.png)

| Network | Input | Output |
|---|---|---|
| Transformer, including output head | Latest8 predicted states, including current `[8,5]` | Ordinary-flight delta `[5]` |
| Contact MLP and its learned linear skip | Current state `[5]` | Independent ground/wall probabilities `[2]` |
| Ground bounce NN | Current state `[5]` | Ground correction `[5]` |
| Wall bounce NN | Current state `[5]` | Wall correction `[5]` |

The contact branch's MLP is `5 -> 256 -> 256 -> latent32 -> logits2`; its learned
linear `5 -> 2` skip is added to those logits before sigmoid. Each probability is
thresholded independently. Each bounce map includes its own learned linear
`5 -> 5` skip.

```text
g_ground = 1[p_ground >= 0.5]
g_wall = 1[p_wall >= 0.5]
s(t+1) = s(t) + delta_flight(history)
          + g_ground * delta_ground(s(t))
          + g_wall * delta_wall(s(t))
```

With gates off, the update uses only current state plus ordinary-flight delta.
Both models return one5D state. That state feeds every branch at the next step;
the second model also shifts it into its8-state history. No analytical collision
test enters this inference.

These dimensions are per ball; batches add a leading `B`. The figures describe
the exact saved scalar and selected two-response architectures. They omit
normalization for clarity. Source: `src/nedm/bouncing_ball/transformer_temporal.py`.

Editable vector figures are alongside the PNGs with `.svg` extensions; layout
generation lives in `artifacts/analysis/bouncing_ball_architecture/draw_architecture.py`.
