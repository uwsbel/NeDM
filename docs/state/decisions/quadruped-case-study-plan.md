# Quadruped case study: the plan to parity, and past it

**Written 2026-09-03. Supersedes ad-hoc direction in the session log.**

Goal: bring case study III (quadruped on CRM) to the same footing as the HMMWV
study, then remove the coverage limitation that parity alone leaves behind.

## Where this stands

**Working:** the Go2 walks on CRM and on rigid ground, the soil renders, runs are
bit-reproducible, and the per-foot forces and soil-height series are already
logged. See [`../machines/crm-rendering-handoff.md`](../machines/crm-rendering-handoff.md).

**The binding constraint:** the checkpoint has **no command input** — not one
command in the data, but no channel to write to. See
[`quadruped-command-channel.md`](quadruped-command-channel.md).

**Why that does not block parity.** The HMMWV model's action input is
`driver_steering / driver_throttle / driver_braking` — *low-level actuation*, not
waypoints. The quadruped analogue is the **12 joint targets** the policy already
emits each control step. So NRD's (state, control) → next-state contract is
satisfiable today.

**What it does cost is coverage.** With a constant command the actions are a
**limit cycle**: one gait, one speed, one heading. The model would learn that
slice very well and have seen nothing else.

## The contribution: contact-mode conditioning

**Not "a fourth system."** §7 of the manuscript names this as future work:

> *"Extending the method to contact-rich manipulation will likely require
> **event- or contact-conditioned neural dynamics, in which a contact mode is
> predicted and the transition is conditioned on it**."*

A quadruped is the natural testbed — four feet, sixteen modes, switching at
2-4 Hz, and the mode is directly **observable** from foot force. The framework
already has the slot: `c_t`, currently carrying the terrain one-hot.

**Why it is doable:** contact modes cycle regardless of the command. A trot at
constant velocity still traverses all-four-down, one diagonal, the other. So the
diversity this contribution needs is inherent in walking, and it is blocked by
none of the three constraints we hit:

| constraint | blocks this? |
|---|---|
| no command channel | **no** — modes cycle anyway |
| foot smaller than kernel | **no** — contact still switches |
| 20 mm spacing under-resolves the patch | **no** — no footprint depth is claimed |

**Tested by the paper's own deletion rule:** train with and without the
contact-mode context, and show removal degrades rollout fidelity.

**Scope:** validation levels 1 and 2 (one-step, open-loop rollout). Level 3,
closed-loop policy transfer, needs the command channel and is the follow-on.

## Stage 1 — parity (in progress, not blocked)

Mirror the HMMWV pipeline rather than building a parallel one.

| HMMWV | Quadruped |
|---|---|
| `driver_steering/throttle/braking` | 12 joint position targets |
| body vel x/y, roll, pitch, roll rate, pitch rate, yaw rate | **identical seven fields** |
| per-tire normal force ×4 | per-foot normal force ×4 |
| per-tire spindle omega ×4 | per-foot **sinkage** ×4 |
| terrain one-hot `['flat','crm']` | `rigid` and `training` soil |
| 75/25 train mix, huber loss, rollout eval at 5 s and 10 s | same |

**Superseded 2026-09-03 — see [`quadruped-contact-mode.md`](quadruped-contact-mode.md).**
The soil-memory justification does *not* hold at stock geometry: the foot (22 mm
radius) is smaller than the SPH kernel support (40 mm), so it floats 22.9 mm
above true contact and leaves no footprint. **The contribution is
contact-mode conditioning instead**, which is stronger and is not blocked by any
of it — contact modes cycle regardless of command, penetration, or spacing.

Soil memory becomes reachable with enlarged feet (50 mm radius flips penetration
to +5.1 mm), which is a *second*, later contribution rather than this one's
premise.

Steps: collection script → one episode's schema reviewed → volume → processed
dataset → training config → rollout eval.

**Stamp `command_constant: true` in every episode's metadata**, so nobody later
mistakes limit-cycle data for a command-conditioned dataset.

## Stage 2 — coverage WITHOUT retraining

**This is the part that does not need the command channel, and it is not a
consolation prize.** The HMMWV gets its diversity from *driver* variety — route
families, OU-noise meander, near-obstacle routes. With the command fixed, the
quadruped can get equivalent coverage from **environment** variety instead. That
is a legitimate substitution and it should be stated as one.

Roughly in order of value per unit effort:

**a. Terrain slope and heterogeneity.** On a grade the gait must adapt, so both
states and actions differ from the flat limit cycle without touching the policy.
Patches of differing soil within one episode do the same and additionally
exercise the transition the terramechanics channel is supposed to capture.

**b. External disturbances.** Push the base mid-episode. A dynamics model trained
only on nominal trajectories has never seen the states it will be *evaluated* in
once a controller deviates. **Recovery transients are the highest-information
data for a residual model** and are nearly free to generate.

**c. Soil parameter variation.** Eleven stiffness settings are already measured
(`chrono-build.md`) and each run is about two minutes. This is the cheapest real
axis available, and it is also the data a *continuously* conditioned model would
need — which the HMMWV study does not have at all.

**d. Robot mass and inertia.** Changes the dynamics the model must learn, with no
controller change.

**e. Initial conditions.** Spawn, heading, seed, initial joint perturbation.
Cheapest, and the weakest — it re-enters the same limit cycle after a transient.

Note (a), (b) and (c) all produce **off-nominal states**, which is precisely what
the current dataset lacks and what a residual dynamics model most needs.

## Stage 3 — the command channel

Only this unlocks trajectory following and the architectural parallel to
`ChPathFollowerDriver`.

0. **Importing is ruled out** — see
   [`quadruped-actuation.md`](quadruped-actuation.md). The nearest-neighbour
   candidate (Genesis Go2) has *our exact limitation* — its own command config is
   degenerate, `lin_vel_x_range [0.5, 0.5]`, which is where our hardcoded 0.5 came
   from. The only real candidate assumes PD torque control, and our joints are
   kinematic constraints. **The 35 h is not avoided by importing.**

1. **Retrain with command randomisation.** Preferred if affordable: same
   simulator, no transfer gap, and we specify the command space. Recipe is known
   to work here — train on rigid (fast), finetune on CRM (slow, brief). Needs a
   `commands` block with resampling, a yaw term replacing the current penalty,
   and relaxed termination. **Cost being priced; this decides the choice.**
2. **Import a command-conditioned policy.** Command randomisation is standard
   elsewhere. Carries sim-to-sim risk, now cheaply testable against two gates:
   `3.1100 m` rigid and `2.5623 m` CRM.
3. **Scripted parametric gait.** A command space by construction, no training.
   Weakest scientifically; a floor rather than a goal.

## Sequencing, and the honest trade

Stages 1 and 2 are **not blocked** and together give a dataset with real state
coverage on two terrains. Stage 3 is a training project of unknown cost.

**So do 1, then 2, and let 3 be decided by its price rather than by its appeal.**
A model trained on stage-2 data would be a genuine result — command-free, but
covering slopes, disturbances and soil variation, which is more than the HMMWV
study covers on the terrain axis.

**Opportunity cost, stated once.** The traverse study already works end to end and
has one open question — plan-space versus driven-space clearance — that is a day
of work. If a second complete case study is needed sooner than stage 3 can
deliver, that sequencing should be chosen deliberately.

## MEASURED: the terrain-conditioning justification, on the quadruped

The manuscript conditions on terrain because *"the same reduced state and action
evolve differently under physically distinct conditions."* For the HMMWV that is
asserted from tire physics. **On the quadruped it is now measured:**

| commanded forward velocity | achieved, rigid | achieved, CRM |
|---|---|---|
| 0.30 m/s | 0.030 | **0.145** — nearly 5× |
| 0.50 m/s | 0.337 | 0.347 — agree to 3% |

**Same policy, same command, same plant.** At low command the terrains differ by
a factor of five; at high command they agree. So the discrepancy is neither a
policy property nor a port bug — **a port bug would affect both terrains alike.**

**This is the argument for the terrain context input, as a measurement rather than
an analogy.** It also says the effect is *regime-dependent*: terrain matters most
in the low-command regime, which a context input can represent and a single
unconditioned model cannot.

**Mechanism not established.** Soil compliance may let a marching gait convert
into translation, or the CRM motion may be partly foot slip rather than clean
walking. The per-foot slip channel already logged would distinguish these. Open.

### And a dead zone that is worth learning

Below roughly 0.35 m/s commanded, the robot on rigid ground **does not translate**
— it marches in place. Inside its trained range, so a property of the plant rather
than an out-of-range request. Families renamed for behaviour
(`march_in_place_015`, `march_in_place_030`) so nothing implies a speed band.

Ordering *within* the dead zone is not claimed: 0.033 against 0.030 m/s is 3 mm/s
on a short window and both mean "not moving."

**Reverse walks and is mildly asymmetric** — 51% tracking backward against 59%
forward at the same magnitude. Another nonlinearity worth having; not leaning on
one episode for the ratio.

## Collection scale, chosen from the paper's own scaling curve

Appendix~D retrains on nested subsets at fixed compute, so unique-trajectory count
is the only variable:

| data | flat rollout err | CRM rollout err | S |
|---|---|---|---|
| **20%** | **4.2%** | **9.6%** | 6.9% |
| 80% | 3.9% | 4.6% | 4.3% |
| 100% | 3.7% | 5.4% | 4.6% |

**20% of their data already gives single-digit rollout error.** The curve is
monotonic 20→80 and the 100% point is within seed noise of 80. So a minimal repro
does not need their full scale — it needs enough unique trajectories to cover the
command families on both terrains.

**Record at 100 Hz, matching the HMMWV**, not at the 50 Hz control rate. The state
is available at physics rate and this doubles transitions per episode for free,
with stair-stepped actions between control steps — exactly what the plant does.

Target: **~600 episodes, 75/25**, 11 families balanced. ~470 rigid (minutes,
8-concurrent) and ~155 CRM (≈5.5 h sequential, the entire constraint), giving
roughly **1 M transitions** at 1600 rows per 16 s episode.
