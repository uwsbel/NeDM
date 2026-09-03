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

**The per-foot row is why this case study exists.** For a wheeled vehicle,
normal force and wheel speed are the terramechanics channel. For a quadruped it
is normal force and sinkage — and **the soil remembers**, so a footprint changes
the next stance. That hysteresis has no wheeled analogue.

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
