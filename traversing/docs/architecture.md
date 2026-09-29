# Architecture

This describes the code at the experiment commit [`901d6c9`][commit]; [evidence.md](evidence.md) names the scripts. The
numbers are those of the deployed configurations.

## Two separate paths

```text
Planning and execution (milestones 1, 2 and 4)
  terrain map: one overhead depth image of the arena
  + candidate routes with speed profiles around the straight start-to-goal line
  + the last 2 s of observable vehicle state and applied controls (shared rigid/soil model only)
      -> risk network: a hazard at each of 96 stations along each route -> route failure probability,
         averaged over an ensemble of independently seeded networks
      -> route search: rank 256 candidates once | iterated sampling, 4 rounds x 64 | + gradient refinement
      -> chosen route and speed profile -> Chrono's stock PID path follower -> Chrono

Controller training (milestone 3)
  recorded drives: 17-number vehicle state, applied controls, local terrain heights, ground type
      -> NRD: a learned 50 ms dynamics model, then frozen
      -> PPO route tracker trained on short imagined drives inside the NRD
      -> exported NumPy actor, run in Chrono every 50 ms and compared with the PID follower
```

- **The planner does not use the NRD.** No planner entry point imports the dynamics-model code, and a planner never
  needs an NRD checkpoint. This was checked with a static import trace, not a runtime trace.
- **The planner learns the outcome of a route, not the dynamics.** A risk network trained on recorded drives predicts
  where a route will fail. No dynamics model is rolled out.
- **Every risk label describes drives by the PID follower.** Swapping in the learned tracker would change what the
  labels mean. The tracker has never driven a planner's route.

## Terrain map and route corridor

- **The map.**
  - Every single-goal planner reads one vehicle-free overhead depth image of the arena: the HMMWV planners, the
    shared history model, the multi-arena planners, the Gator and the Polaris.
  - The camera is 110 m above the arena centre and looks straight down with a 47 degree field of view. The image is
    rendered at 1024 x 1024 and used at 512 x 512, 0.187 m per pixel.
  - Heights are read through a flat-ground pixel lookup. Its height error (RMSE against Chrono's own terrain) is
    0.050 m on f104 and 0.059-0.094 m on the unseen arenas.
- **The corrected depth-to-map conversion.** It places each pixel along its measured ray into a 0.156 m grid. Its mean
  absolute height error against the authored heightmaps is 0.0071-0.0087 m, against 0.027-0.042 m for the flat lookup
  on the same six arenas. It is used only in the sensor-input study and the live navigation loop.
- **The corridor.** Each candidate route is resampled to 96 evenly spaced stations. Each station has 32 sideways
  samples over ±6 m, 0.387 m apart. Each sample carries five channels plus one constant plane:
  - height relative to the route start;
  - along-route grade and cross-slope, both clipped to ±2;
  - the commanded speed at that station;
  - a valid flag.

  The depth-reading model of the live navigation loop reads measured range, ray angle, commanded speed and the valid
  flag instead of height and slopes.
- **Route context.** Five numbers: goal offset in x and y, goal distance, start heading and route length. The network
  reads no vehicle state unless it has the history input.

## Risk network

- **Layers.**
  1. Four 3 x 3 convolution layers (32, 64, 64 and 96 channels). They downsample sideways only, so along-route
     resolution is kept.
  2. A sideways mean and maximum summarise each station.
  3. The route context and the station position are joined in.
  4. A one-dimensional convolution and a bidirectional GRU run along the 96 stations.
  5. The output is one hazard value per station.
- **Route risk.** P(fail) = 1 - exp(-Σ softplus(hazard)). Routes are ranked by the ensemble-mean route logit.
- **Training.**
  - Loss: a discrete-time survival loss at the station of the first unsafe event.
  - Schedule: 30 epochs, AdamW with a one-cycle learning rate peaking at 2e-3, weight decay 1e-4, batch 256.
  - Size: 256,161 trainable parameters, or 262,018 with the history input.
  - Ensembles: 5 independently seeded networks in every deployed planner, 3 in the live navigation loop.
- **Designs compared.** Nine alternatives at equal data (wider or deeper GRUs, an MLP, several transformers). None
  beat this network, except a transformer over route stations and history samples, which was better at a 1 s decision
  (758 vs 742 of 800 soil pairs) and worse at 0.5 s (754 vs 766).
- **Scores rank, they are not probabilities.** The predicted failure probability of the chosen routes is far below the
  observed rate: median 0.008 % against 0.33 % on a rigid-ground hazard test, and a mean of 0.0014 % (median 0.0012 %) against
  0.25 % for the Polaris.

## History input

Used only by the shared rigid/soil model (milestone 2).

- **Encoder.** A causal GRU (hidden size 32) reads the last 40 samples, 2 s at 20 Hz, plus a validity mask:
  - 12 observable signals: forward and sideways body speed, roll, pitch, the three body rates, four wheel speeds and
    engine speed;
  - the 3 applied controls: steering, throttle and brake.

  Tyre normal forces and engine torque are left out.
- **Output.** A linear layer and tanh give 16 numbers. They are computed once per decision, shared by every
  candidate, and joined with the 5 route-context numbers.
- **Training.**
  - Batches hold equal numbers of rigid and soil rows.
  - The whole window is blanked on 20 % of rows.
  - An auxiliary head predicts the ground type from the 16 numbers. It is a training loss only; the planner never
    receives a ground label.
- **The final model** was fitted on 217,814 rows (109,492 rigid, 108,322 soil):
  - recorded drives re-cut at many anchor points;
  - drives replayed part-way on both grounds and then continued along other routes;
  - decision rows cut 0.5, 1, 1.5 and 3 s after the start.

  At its 0.5 s decision only 10 of the 40 samples hold real motion.

## Candidate routes

- **Parameters.** A candidate is 7 numbers:
  - a sideways offset from the straight line, as the sum of three sine modes that are zero at both ends, with
    amplitudes capped by the curvature limit;
  - four speed knots within ±4 m/s of the base speed, clipped to 0.5-6 m/s, with 1.5 m/s² acceleration and a
    2 m/s² stopping cone into the goal.
- **Fixed anchors.** Nine anchors are always included: offsets 0, -4 and +4 m at 2, 4 and 6 m/s.
- **Validation.** A validator rejects routes that break the curvature limit (0.125 per metre), the speed and
  acceleration limits or the arena bounds. The live navigation loop also rejects routes that turn more than 45 degrees
  between consecutive points.
- **A rejected variant.** Starting every candidate at the vehicle's current speed made soil worse (81.8 % vs 83.9 %,
  paired p 0.02) and is not used.

## Route search

| Mode | What it does | Used by |
|---|---|---|
| One-shot ranking | Score 256 candidates, take the lowest risk | Rigid-ground f104 studies and missions, live navigation loop, first soil study |
| Iterated sampling | 4 rounds of 64; after each round, refit a Gaussian over the 7 route numbers to the best 15 % | Default since the second soil study |
| Gradient refinement | From the iterated-sampling pick and the 16 best first-round routes, up to 60 Adam steps on the ensemble-mean route logit through a differentiable corridor extraction, with curvature and arena penalties. Each start keeps its best step as judged by the most pessimistic ensemble member, and the pick is replaced only if that improves by at least 0.3 logit | Second soil study, milestone 2 final configuration, Polaris, some Gator and HMMWV checks |

- **Cost.** On an MI350X, 256 candidates take 0.42 s, of which the networks take 0.042 s. Gradient refinement takes
  2.6-3.4 s per decision on an RTX 5090.
- **Search matters only on soil.** On 200 f104 soil pairs of the second soil study, goals reached were:
  - one-shot ranking: 91.0 %;
  - iterated sampling: 98.5 %;
  - gradient refinement: 99.5 %.

  On rigid ground every mode is at or near the ceiling.
- **Naming.** The earlier forecaster stage used an MPPI optimiser. The current planners use the modes above; calling
  them "MPPI" is inaccurate.

## Path follower, physics and stop rules

- **Follower.** Chrono's stock path follower drives a Bezier path through the route's waypoints:
  - steering look-ahead 5 m, gains (0.8, 0, 0);
  - speed controller gains (0.6, 0.05, 0);
  - steering rate limited to 2 full-scale units per second;
  - desired speed refreshed every 50 ms from the nearest waypoint;
  - updated at every physics step.

  Single-goal studies plan once and never replan. The live navigation loop replans per waypoint or on a timer. It
  uses a re-implemented speed controller whose integral carries across route changes, with anti-windup.
- **Physics.** A 0.8 s braked settle before every drive.
  - Rigid ground: HMMWV with TMeasy tyres, 2 ms step.
  - Soil: Chrono CRM, 0.08 m particle spacing, a 0.24 m layer, density 1700 kg/m³, cohesion 5 kPa, friction 0.8,
    1 ms step. The boundary under the layer holds the soil but does not support a tyre, so a wheel that digs through
    ends the drive. Only the wheels touch the soil: the HMMWV's rigid tyre mesh, and for the Gator and Polaris plain
    cylinders calibrated to the HMMWV's settled sinkage.
- **Stop rules.** Goal radius, rollover, leaving the terrain, prolonged blockage and a 120 s horizon. On soil also a
  spinning wheel digging through the soil layer, which counts as goal not reached.

## Labels and training data

- **Label.**
  - A drive is **unsafe** unless it reached the goal without rolling back. Rolling back means at least 0.05 s in
    total backwards faster than 0.10 m/s with throttle above 0.3, or any moment backwards at 0.30 m/s or faster,
    ignoring the first 1 s of the drive (which starts after the 0.8 s braked settle).
  - The hazard is supervised at the station of the first unsafe event.
  - On soil the label is in practice "goal not reached", because stalled vehicles dig in rather than roll back. The
    Polaris study also counts the hull sitting inside the soil.
- **Rigid data on f104.**
  - 1,200 start/goal groups x 20 routes: 12 designed routes (3 sideways offsets x 4 speed profiles) and 8 from the
    planner's own route family.
  - The groups are split into 1,089 training, 56 validation and 55 test groups.
  - Earlier collection rounds on 1,500 more start/goal groups bring the rigid f104 total to 36,199 driven routes. The
    24,000 routes of the current set alone are 203 simulated hours.
- **Soil data on f104.** The same route ids, collected in batches by route index within each group: 15,235 drives (12-13 per group) and 91.5 simulated
  hours. Every soil drive has a rigid twin with the same id.
- **Other vehicles.** The Gator and the Polaris re-drove the same ids.
- **Exclusion.** A hard check in the trainer keeps the 800-pair f104 evaluation suite out of every training set.

## Live sensing path (milestone 1 only)

The whole mission runs in one Chrono rollout. At each decision the simulator renders one overhead depth frame with
the HMMWV in view. The frame is back-projected into the 0.156 m grid, and the footprint plus 1.5 m is marked unseen.
Corridors for 256 candidates are cut from that single frame and scored by the frozen three-network depth-reading
ensemble. The lowest-risk route goes to the follower, and nothing is remembered between decisions.

## Controller-training path (milestone 3 only)

- **NRD.**
  - Architecture: a causal transformer, 6 layers x 8 heads x width 256, 4.9 million parameters, with a 16-step
    (0.8 s) context at 50 ms.
  - Inputs per step:
    - the normalised 17-number state: body forward and sideways speed, roll, pitch, three body rates, four tyre
      normal forces, four wheel speeds, engine speed and engine torque;
    - a 64-number token from an 8 x 8 grid of terrain heights over 12 m x 12 m around the dead-reckoned pose, read
      from a static height grid, so soil deformation is not seen;
    - the command;
    - a ground-type one-hot.
  - Outputs: the state change and the engine power. The pose is integrated outside the network.
  - Training: 39,601 training episodes (24,280 rigid, 15,321 soil) for 30,000 steps with an 8-step rollout loss. The
    checkpoint used is from step 26,000.
- **PPO tracker.**
  - Training: 2,048 parallel imagined fragments of 1-3 s, started from recorded training drives, half on each ground.
    The NRD is told each fragment's ground type. 1,000 iterations after a warm start that imitates the recorded PID
    commands.
  - Observation, 158 numbers:
    - three tracking errors;
    - 10 look-ahead route points 1 m apart, 3 numbers each;
    - speed and yaw rate;
    - the current command;
    - the last 8 commands;
    - the last 8 values of 12 observable state signals.

    It has no tyre forces, no engine torque and no ground label.
  - Action: steering, throttle and brake, with the steering change clipped to 0.1 per step.
  - Reward: tracking terms, a command-change penalty, a throttle-and-brake overlap penalty and a progress term.
  - Deployment: the 512-256-128 network is exported to NumPy and matches the torch policy to 5e-7. In Chrono it runs
    every 50 ms, holding its command for the interval, without the NRD and without torch.

## Which pieces were run together

No experiment runs the latest pieces as one stack.

| Experiment | Map | Risk model | Search | Follower |
|---|---|---|---|---|
| Live navigation loop | fresh depth frame per decision, corrected conversion | frozen depth-reading ensemble, no history | one-shot | PID |
| Shared rigid/soil model | saved vehicle-free map, f104 only | shared history model | iterated sampling + gradient | PID |
| Multi-arena, Gator, Polaris | saved vehicle-free map per arena | per-vehicle, per-ground model without history | iterated sampling (+ gradient for the Polaris) | PID |
| Tracker benchmark | none, fixed designed routes | none | none | PID vs learned tracker |

An integrated live demonstration would need two things first: one map path, and risk labels re-collected (or
re-validated) under the learned tracker. Then it would need paired end-to-end Chrono runs.

## Three meanings of "HMMWV" in this repository

| Meaning | What it is | Status |
|---|---|---|
| The published study on main | HMMWV trajectory tracking: a 15-number reduced state at 100 Hz with a two-class terrain code, and PPO policies trained inside that NN-ROM and compared in Chrono on flat, CRM and bumpy ground | Unchanged; shares no model with this study |
| The earlier vision/NRD traversal work | An HMMWV under an overhead RGB-D camera. A dynamics model with a camera-derived terrain token imagined each candidate's drive to score time, energy and feasibility | Replaced; see [history.md](history.md) |
| This study | Direct learned route risk plus a conventional follower. A separate NRD-trained tracker is a research benchmark | Documented here |

"HMMWV NRD" is not an accurate label for all three.

[commit]: https://github.com/uwsbel/NeDM/tree/901d6c9423a16c0fafc3d60056065415d5a725f2
