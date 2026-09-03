# The quadruped needs a controller to be excited at all

**Written 2026-09-03 after reading the manuscript. This is the structural
difference between case study III and the three published ones.**

## What the paper does

**Excitation, §4 (HMMWV):**

> *"Data are generated with deliberately excited driver maneuvers rather than a
> single path-following controller, so the training distribution spans
> straight-line launch/brake, transient cornering, and combined
> longitudinal–lateral response. Six maneuver families are shared across regimes
> sampled over low, medium, and fast speed bands."*

**No controller.** Command sequences are scripted directly at the action level —
steering, throttle, braking — and the vehicle responds. Same for the arm and the
tracked vehicle.

**The policy is the output, not the input.** NRD is frozen, thousands of copies
are batched on one GPU, PPO trains a policy inside it, and the policy is returned
to Chrono for closed-loop validation. Throughput is what makes this practical.

## Why the quadruped is different: plant stability

**A car, an arm and a tracked vehicle are open-loop stable.** Command throttle
and the vehicle moves sensibly. Excitation happens *at the action level*, and no
controller is needed to keep the system in a usable state.

**A quadruped is open-loop unstable.** Command joint torques with no stabilising
loop and it collapses immediately. **It cannot be excited at the actuator level
at all.**

So the plant that gets excited must already contain a stabilising controller.

## The resolution: the policy is part of the plant

`model_2999` occupies the same architectural position as the HMMWV's
**powertrain** — the machinery that converts a command into motion. It is
infrastructure, not a competing answer to the research question.

| | HMMWV | Quadruped |
|---|---|---|
| plant | chassis + tires + **powertrain** | robot + **walking policy** |
| excitation enters at | steering / throttle / brake | **velocity command** |
| NRD learns | body + terramechanics response | body + foot-contact response |
| NRD-trained policy does | reference tracking | a task at the command level |

**The two policies are at different levels of a hierarchy.** The inner loop
walks; it cannot navigate, track a reference, or perform a task. The outer loop,
trained inside NRD, would do a task by issuing commands. That is ordinary
hierarchical legged control, and the circularity dissolves once they are not
confused for each other.

**This is why the command channel is load-bearing rather than a convenience.**
For the HMMWV, excitation is applied at the action level. For the quadruped it
*must* enter at the command level, because the action level is occupied by the
stabiliser. **No command channel means no excitation means no diverse dataset.**
See [`quadruped-command-channel.md`](quadruped-command-channel.md).

The six maneuver families become the template: sample commands over speed bands
and turn rates, which is exactly what command randomisation during retraining
would make possible.

## Falls: terminate and discard, following the paper's own practice

Every episode already *"discards an initial settling transient before recording"*,
and the traverse study terminates episodes on failure.

Falls should be handled the same way, and the paper says why in its limitations:

> *"the framework was not evaluated on explicit contact-mode transitions … These
> events introduce discontinuous and potentially multimodal dynamics that are
> difficult to represent reliably with a single continuous learned transition
> model."*

A falling quadruped generates exactly those transitions — body-ground contact
modes the model has no machinery for. **Terminate on fall, discard the episode
tail**, and keep the dataset on the locomotion manifold.

## And the case study answers a stated limitation

The same passage names the gap this case study fills. The arm is *"trained only on
free-space motion and its controller avoids contact through the safety shield"*;
the vehicle cases have only *aggregate* tire-terrain interaction. **A quadruped is
the framework's first intermittent-contact system**, with four contact modes
switching at 2-4 Hz.

That is the scientific case — and simultaneously the warning. The paper suggests
this territory *"will likely require event- or contact-conditioned neural
dynamics, in which a contact mode is predicted and the transition is conditioned
on it."* So case study III is not merely another system; it enters the regime the
authors flagged as needing new machinery. **A per-foot contact-mode input in the
context slot `c_t` is the obvious first thing to try**, and the framework already
has that slot — it is where the terrain one-hot goes.

## The reduced state, confirmed against the paper

HMMWV is **15-D**: 7 body-motion channels (`v_x, v_y, φ, θ, ω_x, ω_y, ω_z`) plus
an **8-channel terramechanics block** — 4 tire normal loads and 4 wheel speeds,
which together *"make slip and load transfer observable."*

Our collection schema matches by construction: the same 7 body channels, plus
per-foot vertical force and per-foot slip speed. **The design rule is deletion** —
*"a channel earns its place only if removing it degrades rollout fidelity or
closed-loop performance"* — so sinkage and surface displacement are collected as
candidates and justified by ablation, not by assertion.
