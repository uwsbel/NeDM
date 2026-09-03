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
