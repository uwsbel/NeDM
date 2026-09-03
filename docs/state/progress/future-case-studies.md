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
>
> **Superseded 2026-09-03: the CRM run is now verified, and so is the gait.**
> A Go2 under a ported RL policy walks **2.19 m in 8 s on CRM**, max tilt 7.3°
> against 6.6° on rigid ground — so the soil costs some stability and does not
> destroy it. The rigid-ground control walks 3.11 m in 8 s, which validates the
> whole port of the in-house checkpoint (joint reorder, sign flip, 45-dim
> observation). **The policy was trained in Chrono**, not imported from another
> engine — see the correction note below.
> The reasoning above was right: SPH does not share the OptiX fault.
>
> Two things were **both** required and neither alone sufficed: the `training`
> soil preset (cohesion 2000 rather than 5000) **and** `artificial_viscosity`
> 2.0. Do not carry 5.0 over from single-sphere box tests — it hard-crashes at
> domain scale, particles leaving the domain, core dump.
>
> **The "~4 cm differential sinkage" was never sinkage.** It was differential
> *foot height* — front feet at 0.131 m against rear at 0.245 m — measured from
> foot **body** positions, not from soil. That is the robot **pitching
> nose-down**, a body attitude. Corrected 2026-09-03; see the measured soil
> response below.
>
> **What remains blocked is only the rendering**, and its cause is now known
> precisely: `AttachFsiSphSystem` has no Python binding, and on the pinned
> commit its body is empty unless the build defines `CHRONO_HAS_OPTIX`. See
> [`../machines/chrono-build.md`](../machines/chrono-build.md). A source build
> is in progress on both boxes to close it.
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
stance. That is a hysteresis channel no current study has.
>
> **Measured 2026-09-03 and it does not hold at stock geometry.** The Go2 foot
> (22 mm radius) is *smaller than the SPH kernel support* (40 mm), so it floats
> 22.9 mm above true contact, never penetrates, and leaves no footprint. There is
> no hysteresis to learn. Enlarging the foot to 50 mm flips penetration to
> +5.1 mm, so the channel is reachable — but as a second contribution, not this
> one's premise. **The live justification is intermittent contact and
> contact-mode conditioning**; see
> [`../decisions/quadruped-contact-mode.md`](../decisions/quadruped-contact-mode.md). Throughput value
peaks here too, since quadruped + CRM is plausibly the most expensive scene in
the lab.

**Proposed `z1` (~40-D):** body-frame `(vx, vy, vz)`, roll/pitch, body rates (3),
12 joint angles + 12 joint velocities, per-foot normal force (4), **per-foot
sinkage (4)**. The headline ablation is dropping the sinkage block — the direct
analogue of the paper's terramechanics ablation.

**The bootstrapping problem is SOLVED, 2026-09-03, and the ranking below is
stale.** `scripts/quadruped_go2_crm.py` runs a Unitree Go2 on CRM soil in the
`nedm` environment, driven by `model_2999.pt` from
`uwsbel/sbel-reproducibility` 2025/multi-terrain-RL. That checkpoint was trained
on rigid ground in Chrono and **finetuned on CRM granular terrain**, which is
ranked option 2 below, listed there as untried.

First run, no tuning: 8 of 8 FSI bodies coupled (four feet, four calves), the
robot settles 5.9 cm and holds rather than sinking, and it **walks at 0.177 m/s**
against a hardcoded 0.5 m/s command, tracking roughly straight. A wrong
observation vector produces thrashing or backwards motion, so the ported
convention is right.

**Update 2026-09-03: the Go2 WALKS on rigid ground, and the port is validated.**
`--terrain rigid` reproduces the `ChBodyEasyBox` ground the policy was trained
on. Result: **8 s upright, 3.11 m travelled at 0.423 m/s** against a 0.5 m/s
command (85%), max tilt **6.6°**, body height steady at 0.35 m.

That validates the entire port end to end: observation convention, the
policy-frame reorder, the sign flip, the motor mapping, the 45-wide vector,
the actuation type. **So every CRM failure is CRM-specific**, not a porting bug.

The contrast is total and early:

| tilt at | rigid | CRM |
|---|---|---|
| 0.5 s | 0.3° | 3.0° |
| 1.0 s | **1.5°** | **11.8°** |
| 1.5 s | 2.4° | 29.8° |

**Cause found 2026-09-03: the CRM bed heaves.** A rigid sphere at rest on this
soil, zero drop, rises 0.12 m. Sharp onset between 0.20 and 0.25 m of terrain
depth, then saturating. Not the robot, not the policy, not the spawn, not the
step size, not the resolution. Full measurement in
[`../lessons/chrono-versions.md`](../lessons/chrono-versions.md).

That explains the run history: the early Go2 runs used depth 0.20, just below
onset, and showed a 7.8 cm "rebound"; the run at depth 0.30 fell earliest of
anything all day at 1.15 s, and is exactly the configuration with the largest
heave. **Enlarging the terrain from 3.0x1.6x0.20 to 8x4x0.3, which was adopted
from working code as an improvement, crossed the threshold.** Keep CRM beds at or
below ~0.15 m until this is understood.

**And the bounce is absent on rigid.** Same robot, same spawn logic, same policy,
comparable foot clearance: rigid recovers 2.5 cm as the legs take the load, CRM
rebounds **7.8 cm at a coefficient of restitution of 0.78**. Granular soil
returning three quarters of an impact velocity is not soil behaviour, and it is
now isolated as the open question. Candidates remaining are SPH resolution (the
25 mm foot spanned only 1.7 particle spacings at the original 0.03) and solver
iterations (60, taken from the rigid-ground skill, against 150 in SBEL's own CRM
playground).

**The lesson is the sequencing.** This control existed the whole time and would
have taken four minutes. Running it first would have skipped an afternoon spent
on spawn clearance, embedded particles and pose ramps, all of which assumed the
port might be at fault.

**Superseded below: the tumble was a spawn artifact, and a second failure is
underneath it.** The Go2's URDF rest pose extends the legs, so a base height
chosen as a constant put the feet *below* the soil surface and the BCE markers
took a launch impulse from the particle bed. Spawn height is now derived from
measured leg reach (0.421 m) with the foot margin expressed in SPH spacings.
Rise in the first 200 ms is then **exactly 0.0000**, and the robot settles into a
correct stance.

It still falls. Standing, doing nothing, it reaches **11.8° of tilt by t=1.0**
and pitches forward from there, front legs folding under. The stand and policy
runs are identical to three significant figures until t=1.0, so **whatever
topples it begins before the controller matters**. That is the open question,
and unlike the fall-time differences it is a large effect that survives the
solver noise.

*Superseded, retained for the record:* **it only walks for about two seconds.** A 6 s run shows the base pitching
~158° about Y between t=1 and t=3 and then lying inverted and motionless for the
rest of the window. The 0.177 m/s figure is from a 1 s run that stopped before
the tumble; `forward_travel_m` over 6 s is not gait distance, since the robot
reaches 0.441 m by t=2 and then slides back to 0.36 m while on its back.

So the honest statement is narrower than "solved": **the machinery works and the
controller does not yet survive on this soil.** The URDF loads, the FSI coupling
takes on all eight bodies, the observation convention is correct, and the policy
produces a real gait. Whether the tumble is the policy meeting soil it was not
finetuned against, the foot geometry (25 mm spheres), the 5.9 cm settle putting
the robot in a pose outside the training distribution, or the exchange interval,
is **not yet established**. A `--no-policy` stand-pose run separates the
controller from the physics and has not been done.

Ranked option 3 is therefore better than "highest risk, keep off the critical
path" but is not finished. It gets a walking Go2 onto CRM in an afternoon; it
does not yet get a Go2 that stays up.

> **Which CRM numbers are measurements and which are samples.** Five runs of one
> identical command, standing:
>
> | t | run 1 | run 2 | run 3 | run 4 | run 5 | spread |
> |---|---|---|---|---|---|---|
> | 0.50 s | 3.0° | 3.0° | 3.0° | 3.0° | 3.0° | **0.00** |
> | 1.00 s | 11.8° | 11.8° | 11.8° | 11.8° | 11.8° | **0.00** |
> | 1.25 s | 21.2° | 21.2° | 21.2° | 21.2° | 21.2° | **0.00** |
> | 2.00 s | 51.7° | 51.7° | 50.7° | 56.8° | 41.4° | 15.4 |
> | 8.00 s | 119.5° | 132.4° | 97.3° | 125.3° | 52.8° | 79.6 |
>
> **The first 1.25 s is bit-identical.** The solver is not noisy; it reproduces
> exactly until the robot loses balance, after which an inverted pendulum past
> its tipping point amplifies last-bit differences without bound. That is
> sensitive dependence, not a broken solver, and the distinction says precisely
> which figures can be trusted.
>
> **Reportable as measurements:** anything before ~1.25 s. Tilt at fixed early
> times, launch height, foot clearance, particle counts.
> **Samples, needing a median and range over N runs:** `fell_at_s`,
> `max_tilt_deg`, final base z, `forward_travel_m` (which varied by a factor of
> 2.2), and the gate verdict itself. **One of the five runs never fell at all**,
> so the same command yields both PASS and FAIL. `fell_at_s` cannot be a gate.
> Gate inside the reproducible window instead: tilt at t=1.0, or
> time-to-reach-15°.
>
> Realtime factors are a tight cluster across the day (0.291 / 0.267 / 0.273)
> and wall time does not depend on trajectory chaos, so they are approximately
> right, but each is still one sample.

**Measured CRM cost with a quadruped actually walking on it**, which is the
number this section has always been arguing about and which nothing previously
had: **0.291x realtime**, from one run. 3 x 1.6 m patch at 0.03 m spacing, 43,632 SPH
particles, 29,886 boundary BCE markers, 8 FSI-coupled bodies, no camera.
`rtf_cfd` 3.46 against `rtf_mbd` 0.543, so the multibody side is about 14% of
the cost rather than the ~0.04% a single-free-body measurement suggested.

That still supports "CRM runs below realtime", but **by 3.4x, not by the order
of magnitude** the earlier camera-inclusive figure implied. Note also that the
finer 0.03 m spacing did not cost what naive particle-count scaling predicts:
10x the particles of the single-probe smoke test for well under 2x the cost.

*Historical, retained because the reasoning still frames the alternatives:* You
cannot train the locomotion policy in Chrono + CRM: PPO needs ~10⁸ steps and CRM
runs below realtime — that is precisely the problem NRD exists to solve. And a
random-action quadruped falls in ~0.4 s, so the HMMWV trick of collecting from a
meandering driver gives a dataset that is 100% collapse dynamics.

**Measured 2026-09-02, and it narrows this:** the RoboSimian gait script (removed; see git history through `5ccd2fe`)
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

## Correction: the Go2 policy was trained in Chrono, not imported

**Recorded 2026-09-03 after Kyle caught it.**

`model_2999.pt` comes from `uwsbel/sbel-reproducibility` 2025/multi-terrain-RL:
**trained in Chrono on rigid ground and finetuned on CRM granular terrain**,
in-house. It has never touched another engine.

Earlier notes, variable names (`CHRONO_TO_GENESIS`, `GENESIS_DEFAULTS`) and a
summary page all described this as a "Genesis→Chrono port". **That was wrong.**
The word leaked from §Bootstrapping option 3 — *"Import a pretrained Go2 policy
(Genesis / IsaacLab)"* — which the plan **rejected** as highest-risk and kept off
the critical path. A rejected option's label attached itself to the thing that
replaced it, and then propagated into code identifiers, three doc lines and a
published page.

Kyle found it by noticing the claim was internally inconsistent: **Genesis has no
CRM**, so a Genesis-trained policy could not have been finetuned on granular
terrain. The contradiction was visible in the docs for a day.

The joint ordering `[FR, FL, RR, RL]` and the split thigh defaults are **RSL-RL's
convention**, taken from `chrono_crmenv.py`, which is authoritative because it
ships with the checkpoint. Identifiers renamed to `CHRONO_TO_POLICY` and
`POLICY_DEFAULTS`.

### And this explains the soil-preset dependency

The Go2 walks on the `training` preset (cohesion 2000) and falls on `eval`
(cohesion 5000). That was recorded as an unexplained sensitivity alongside the
`artificial_viscosity` finding.

**It is not a numerical mystery — it is a domain match.** `training` names the
soil parameters *the policy was finetuned on*. The preset dictionary was carried
over from the source repo with its names intact. A policy finetuned at cohesion
2000 surviving there and failing at 5000 is ordinary generalisation failure, and
the preset name said so all along.

**Consequence for the study:** results on `training` soil are in-distribution for
this controller. Any claim about CRM locomotion that depends on the policy
staying upright must say which preset it used, and `eval` is the honest test of
generalisation.
