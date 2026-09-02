# Candidate case studies 3 and 4

**Status:** Under discussion, nothing built · **Updated:** 2026-09-02

Motivation: see the structural gap in
[`state-only-paper.md`](state-only-paper.md). Both existing case studies have
continuous contact and a fully-observed reduced state, and terrain conditioning
is a 2-D one-hot over three discrete terrains.

## III — Quadruped locomotion on CRM terrain *(proposed by the author; recommended)*

**Why it is the strongest candidate.** It is the first **intermittent-contact**
system: four contact modes switching at 2–4 Hz each, against an NRD that
predicts a residual `Δz_t` which is genuinely discontinuous at touchdown and
liftoff. On CRM the reduced-state question is also new — the foot analogue of
the paper's per-wheel terramechanics block is **sinkage plus penetration and
extraction force**, and the soil *remembers*: a footprint changes the next
stance. That is a hysteresis channel no current study has. Throughput value
peaks here too, since quadruped + CRM is plausibly the most expensive scene in
the lab.

**Proposed `z1` (~40-D):** body-frame `(vx, vy, vz)`, roll/pitch, body rates (3),
12 joint angles + 12 joint velocities, per-foot normal force (4), **per-foot
sinkage (4)**. The headline ablation is dropping the sinkage block — the direct
analogue of the paper's terramechanics ablation.

**The bootstrapping problem, which is the real risk.** You cannot train the
locomotion policy in Chrono + CRM: PPO needs ~10⁸ steps and CRM runs below
realtime — that is precisely the problem NRD exists to solve. And a
random-action quadruped falls in ~0.4 s, so the HMMWV trick of collecting from a
meandering driver gives a dataset that is 100% collapse dynamics. Ranked ways
out:

1. **Scripted gait, not RL — the WP0a move.** A trot with Raibert-style foot
   placement over PD joint tracking. Chrono already ships this pattern:
   `chrono_models/robot/robosimian/` with `RS_Driver`, and
   `data/robot/robosimian/actuation/walking_cycle.txt` plus `driving_`,
   `inchworming_` and `sculling_` start/cycle/stop triples. Gate on a privileged
   scripted gait walking on rigid ground, then CRM, with **zero learning**,
   before any model work.
2. **Rigid-ground PPO in Chrono → CRM collection driver.** The rigid policy will
   walk badly on soil; that is fine. The driver needs to be diverse and not
   instantly collapsing, not good — same logic as the 20% meander / 10%
   near-obstacle mixture in Study 3.
3. **Import a pretrained Go2 policy (Genesis / IsaacLab).** Highest risk, keep
   off the critical path. Sim-to-sim locomotion transfer is its own research
   problem (actuator models, contact solvers, URDF fidelity), and a policy
   trained on rigid ground with no soil randomization will likely not survive
   CRM.

**Platform.** Chrono has `ChParserURDF` (Python-exposed, with
`demo_PARSER_URDF_RoboSimian.cpp` as reference), so a Unitree Go2 URDF import is
supported — 12 DOF, universally legible, abundant reference controllers.
RoboSimian is in-tree with free gaits but is a 32-DOF wheel-on-limb dexterous
quadruped that reviewers will not recognize. Suggested: prototype on RoboSimian
to shake out CRM foot-contact machinery, run the study on Go2.

**Design gotcha to settle before collection.** A gait cycle is ~0.3–0.5 s; the
16-token context at 50 Hz is 0.32 s — right at the edge. The model may be unable
to infer gait phase from context alone and therefore unable to predict touchdown.
Either feed the controller phase/clock into the token or lengthen the context.

## IV — Excavation: bucket–soil interaction in CRM

Wheel loader or excavator arm digging a granular pile, with an ego depth camera
on the boom. Reuses CRM from Case Study I and the articulated arm from Case
Study II.

Makes "what to omit" unavoidable: you cannot propagate 10⁶ SPH particles, so the
reduced state must carry the pile's *shape* — and unlike terrain type, that
shape changes **irreversibly as a consequence of the policy's own actions**,
which kills the persistence and static-layout baselines by construction. It is
also the first case where `z2` carries state genuinely not recoverable from
`z1`, which is the master plan's Phase-2-onward requirement.

`z1` = arm joint state + bucket pose/twist + measured resistive force/torque.
`z2` = learned pile latent from ego depth. Task: fill to a target mass, or grade
to a target profile. Auxiliary head labels come free as a height-field residual
against the true SPH surface.

## Runners-up, and why they lost

- **Flexible-link arm** (ANCF links on Case Study II's arm). Cheap — reuses the
  entire pipeline — and uniquely offers a *classical* baseline (modal truncation
  at matched dimension) plus a clean **negative result**, which the paper lacks
  entirely. Keep as the swap-in if plant diversity is wanted over two CRM
  studies.
- **Continuous terrain-parameter conditioning** (rover on parameterized
  regolith, replacing the one-hot). Worth doing eventually but reads as "Case
  Study I with more terrains" — strengthens an existing claim rather than
  opening a new one.
- **Multi-vehicle interaction.** Needs a scene-graph encoder; different paper.
