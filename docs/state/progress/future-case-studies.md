# Candidate case studies 3 and 4

**Status:** Under discussion, nothing built · **Updated:** 2026-09-02

Motivation: see the structural gap in
[`state-only-paper.md`](state-only-paper.md). Both existing case studies have
continuous contact and a fully-observed reduced state, and terrain conditioning
is a 2-D one-hot over three discrete terrains.

## III — Quadruped locomotion on CRM terrain *(proposed by the author; recommended)*

> **Unblocked 2026-09-02 by the `nedm` environment.** Both blockers here were
> version, not hardware. Under pychrono 10.0.0 from the `projectchrono` channel
> (`conda env create -f environment.nedm.yml`), verified on `kyle-sbel`:
>
> - **`pychrono.fsi` is present with the CRM machinery by name**:
>   `PhysicsProblem_CRM`, `ChFsiProblemSPH`, `ChFsiSystemSPH`,
>   `RheologyCRM_MCC`, `RheologyCRM_MU_OF_I`, SPH integration schemes. So this
>   study is prototypable, and the **CRM realtime factor, which is the number
>   this section actually argues about and which nothing has ever measured, is
>   now measurable.** Presence is verified; **a CRM run is not**, and the
>   OptiX failure under this same build is a reminder that those differ. CRM is
>   SPH rather than ray tracing, so it should not share that fault, but that is
>   reasoning rather than evidence.
> - **`ChParserURDF` is exposed**, so the Go2 URDF path is open.
> - RoboSimian and all twelve actuation files ship in 10.0.0 too, so the
>   prototype and the study want the same environment.
>
> Under the older `envs/chrono` (pychrono 9.0.0) none of that holds. Use `nedm`.


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
meandering driver gives a dataset that is 100% collapse dynamics.

**Measured 2026-09-02, and it narrows this:** `scripts/quadruped_wp0_gait.py
walk` runs scripted RoboSimian on `kyle-sbel` at **roughly 1.22 to 1.25x
realtime**, depending on Chrono version (SMC + Bullet, 1 ms step, rigid ground,
no rendering). Six runs, n=3 per environment: 9.0.0 gives 1.2448-1.2516 and
10.0.0 gives 1.2111-1.2243. The ranges do not overlap, so the ~2.4% penalty on
10.0.0 is consistent rather than noise, but it is one machine, sequential, with
no thermal control, and 2.4% changes no decision here. Do not quote either
figure to four digits. A 10 s run reads 1.14x because startup is not amortized.

The gait is at steady state from the first cycle and **the physics is
version-independent**: `stride_length_m_mean` is **0.2001 m** in all four runs
across both environments, so **10.4 mm/s** stands regardless of Chrono version.
Sway peak-to-peak agrees to 0.13% and the callback speed to 0.04%.

Two measurement traps. The chassis sways +/-77 mm within a stride and moves
backward mid-stride, so anything measured between window endpoints rather than
between wrap events overstates stride by about 10% and reads sway as drift. And
**net lateral per cycle is not stable enough to gate on**: it is -0.0001 m on
9.0.0 and +0.0008 m on 10.0.0, a sign flip. Both mean "straight" at under 0.5%
of a stride, but a straightness criterion with a tight threshold would flip
between versions on integrator noise.

**CRM is now measured, 2026-09-02.** `scripts/crm_sensor_smoke.py` on
`kyle-sbel` at collection-grade resolution (0.08 m spacing and 5e-4 s step, both
matching `configs/hmmwv_crm_eval.json` exactly; 2 m patch, ~3.1k SPH particles):

| Configuration | Realtime factor | Steps/s |
|---|---|---|
| Uncoupled probe, no camera *(measurement artifact, see below)* | 0.679 | 1358 |
| **Coupled probe, no camera** | **0.478** | 956 |
| Coupled probe, with camera | 0.0865 | 173 |

**Quote the middle row.** Training a policy needs no camera in the loop, so
0.478x is the figure that bears on this section's argument: CRM runs roughly
**2x slower than realtime**, which supports the claim while being far less
damning than the rendered figure. Quoting 0.0865 would overstate the case by
5.5x.

Two things the spread shows. Coupling the probe costs about 30% (0.679 to
0.478), so the earlier uncoupled number really was an upper bound: it advanced
SPH with nothing in it, no BCE markers and no fluid-solid force computation.
And **rendering, not soil physics, dominates** when a camera is present, a 5.5x
slowdown on its own. `crm_build_seconds` is also not comparable across the
camera boundary, since the rendered run pays OptiX shader compilation there
(5.48 s versus 0.51 s).

Still optimistic against real collection, which carries roughly 2.5x the
particles (the config's active domain is [2, 2, 1] at 0.08 m) plus BCE markers
for four tires plus the vehicle's own multibody dynamics.

That 10.4 mm/s is worth holding onto separately from the throughput question:
RoboSimian is a slow statically-stable walker, and at 1 cm/s a traverse of any
useful length costs a lot of simulated time to collect. It is a machinery
prototype, not a data source. Scope the realtime figure honestly too. This bounds the *non-terrain* half of the loop only, and says nothing
about CRM, which is the terrain the "below realtime" claim is actually about and
which neither box can run. What it does say is that ranked option 2 below, a
rigid-ground PPO seed policy in Chrono, is worth costing properly rather than
assuming it is out of reach; the throughput objection applies to CRM, not to
rigid ground.

Ranked ways out:

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
