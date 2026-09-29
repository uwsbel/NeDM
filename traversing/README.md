# Traversing: learned route-risk planning for off-road vehicles

Follow-on work to the NeDM paper, not part of it. A vehicle in Project Chrono has to reach a goal across generated
hill-and-crater terrain, on rigid ground and on CRM deformable soil. A neural network trained on thousands of
recorded Chrono drives predicts, for a candidate route and speed profile, where along the route the vehicle is likely
to fail: roll back on a climb, stall, dig into the soil or tip over. A route search picks the route with the lowest
predicted risk, and Chrono's stock PID path follower drives it. Separately, a low-level route tracker is trained with
PPO inside a learned neural reduced dynamics model (NRD) and compared with that PID follower.

This folder records the progress through 2026-09-28: what each milestone achieved, how it was measured, and what it
does not show. **The code, data and trained models are not on main yet.** They live on the experiment branch
`offroad_vehicles_v1` at commit [`901d6c9`][commit] and will follow in separate pull requests (see
[What comes next](#what-comes-next)).

## Milestones as of 2026-09-28

All results are closed-loop drives in Chrono. A drive that misses the goal counts as unsafe in every study except the
tracker comparison. Terms used below:

- **f104**: the 80 m x 80 m hill-and-crater arena where most training drives were recorded. The shared rigid/soil
  model, the tracker and the other-vehicle planners were trained on f104 alone; the depth-camera model of milestone 1
  also used three sibling arenas, and the planners of milestone 4a up to two more.
- **Unseen arenas**: eight arenas made by the same terrain generator with new seeds and never used in training.
- **Straight route**: the straight line to the goal at a fixed speed, with no model.

| # | Milestone | Status | Headline | Compared with | Scope |
|---|---|---|---|---|---|
| 1 | A Chrono depth camera in the planning loop | **Achieved in simulation** | Planning once per waypoint from a freshly rendered depth image completed **27/30** multi-waypoint missions (25 without a backward slide), including 16/16 on four arenas generated for the test (15 without a slide) | Replanning every 2 s, every 1 s, and every 1 s with the planning delay charged: 25, 25 and 22 of 30 completed (18, 19 and 19 without a slide). No model-free baseline was driven after the runner fix | Rigid ground only. A fixed overhead camera sees the whole arena, and the vehicle pose is exact |
| 2 | One risk model for rigid ground and soil, not told which one it is on | **Achieved on f104** | Deciding 0.5 s after the start, with gradient-refined routes: soil **780/800** (97.5 %), rigid **800/800** goals reached | The same kind of model deciding after a 3 s straight approach: 671/800 and 797/800 | Most of the soil gain comes from deciding earlier. The history input's own contribution was not shown. The 0.5 s decision was planned offline between two drive passes and switched in instantly (refinement takes about 2.6 s per decision), so it is not yet a live result. One arena, one soil setting, and the configuration was chosen on these same 800 pairs |
| 3 | A learned route tracker that beats PID | **Partial** | Rigid ground, on the 141 test routes the PID had completed on both grounds in earlier recordings: cross-track error **0.200 → 0.106 m**, speed error 0.464 → 0.348 m/s, 141/141 completed by both | Chrono's stock PID follower on the same routes | On soil it completes 129/141 against 140/141, with 12 unsafe drives against 1. The predeclared replacement test fails on both grounds. All routes are new start/goal pairs on f104; one trained policy |
| 4a | The HMMWV planner works on arenas it never trained on | **Achieved on rigid ground; at the 90 % line on soil** | Training on three arenas instead of f104 alone, with the same amount of data, on 8 unseen arenas: soil goal not reached **12.05 % → 9.70 %** (1,000 pairs); rigid unsafe at 2 m/s **8.325 % → 6.175 %** (2,000 pairs) | The f104-only planner | Same terrain generator, new seeds. The declared tests pass on both grounds, but the gains are small: on soil about twice the difference between two identically trained planners, and 4-5 of 8 unseen arenas stay below 90 % |
| 4b | The pipeline works for other vehicles | **Polaris achieved on soil; Gator not** | A Polaris planner trained on the Polaris's own drives reached the goal safely on **798/800** f104 pairs and **997/1,000** pairs on 8 unseen arenas | The Polaris driving the straight route at 6 m/s: 796/800 and 970/1,000 | Retrained for each vehicle, soil only. Mostly the Polaris's mobility: on f104 driving straight nearly matches the planner, whose own gain shows on the unseen arenas (+2.7 points). The Polaris used Chrono's stock driveline, which over-powers the wheels. The Gator's own planner reached the goal safely on 536/800 |

**Main finding: when the planner decides matters more than which model it uses.** After a 3 s straight approach the
vehicle is often already committed to a climb when the planner decides. Deciding 0.5 s after the start, while the
vehicle is still on the flat start pad, took the unchanged shared model from 671 to 753 of 800 soil pairs. Retraining
it with early-decision examples raised this to 766, and gradient refinement to 780. Deciding early mostly recovers what
planning from rest already achieved: with the same search, the retrained model deciding after 0.5 s ties the earlier
soil-only planner deciding at rest (766 vs 766). The final 14 pairs come from gradient refinement, which the planners
that decide at rest never received.

[docs/milestones.md](docs/milestones.md) gives the protocol, denominators, baselines, uncertainty and limits of
every headline.

### Against the 90 % bar

The bar: the planner in the loop gets the vehicle to the goal safely on at least 90 % of start/goal pairs. It was set
on 09-27, so every row except the Polaris rows is a post-hoc reading (the HMMWV f104 rows come from the studies of
09-21 to 09-24, the unseen-arena and Gator rows from 09-25/26). The HMMWV f104 rows were also measured on the pairs
used to choose the configuration. The Polaris study declared the bar before its drives.

| Vehicle | Ground | Test pairs | Reached safely | Bar |
|---|---|---|---|---|
| HMMWV | soil | f104, 800 | 779/800 (final shared model; its 0.5 s decision was planned offline and switched in instantly, not live) | met |
| HMMWV | rigid | f104, 800 | 799/800 (final shared model, same offline 0.5 s decision) | met |
| HMMWV | soil | 8 unseen arenas, 1,000 | 90.3 % (three arenas, same data; mean of two ensembles); 906/1,000 (all data); 87.9 % (f104 only) | at the line: 4-5 of 8 arenas below 90 % |
| HMMWV | rigid, 2 m/s | 8 unseen arenas, 2,000 | 93.8 % (three arenas, same data); 1,893/2,000 (all data); the three-arena planners stay at or above 90 % on every arena | met |
| Gator | soil | f104, 800 | 536/800 (own planner); 543/800 with gradient refinement | not met; mostly the vehicle stalling on climbs, but about half the failures were on routes the model rated at most 50 % risky |
| Gator | rigid | f104, 800 | 800/800 (the straight route at 6 m/s: 795/800 safely); from the Gator study's rigid read-out, not in the compact tables | met, at the ceiling |
| Polaris | soil | f104, 800 | 798/800 | met |
| Polaris | soil | 8 unseen arenas, 1,000 | 997/1,000; every arena 124 or 125 of 125 | met |
| Polaris | rigid | none | no rigid Polaris data was collected | not assessed |
| M113 | soil | 144-route smoke test only | no data collection, no planner | not assessed |

## How it works

Two separate paths. The route planner does not use the NRD, and the tracker has not been combined with the planner.

```text
Planning and execution (milestones 1, 2 and 4)
  terrain map from an overhead depth image of the arena
  + candidate routes with speed profiles
  + the last 2 s of observable vehicle state and applied controls (shared rigid/soil model only)
      -> risk network: a hazard at each of 96 stations along the route -> route failure probability
      -> route search: rank 256 candidates once, or iterated sampling (4 x 64), optionally gradient refinement
      -> chosen route and speed profile -> Chrono's stock PID path follower -> Chrono

Controller training (milestone 3)
  recorded drives: 17-number vehicle state, applied controls, local terrain heights, ground type
      -> NRD, a learned 50 ms dynamics model -> frozen
      -> PPO route tracker trained on short imagined drives inside the NRD
      -> exported NumPy actor, driven in Chrono against the PID follower
```

[docs/architecture.md](docs/architecture.md) describes each piece.

## What these results do not show

- **Onboard sensing.** Every map comes from a fixed camera 110 m above the arena that sees all of it, with ideal
  depth, and the planner uses the simulator's exact vehicle pose. Limited-range sensing was only tested offline.
- **A live early decision.** Milestone 2's routes were planned offline between two drive passes and assumed to take
  effect instantly 0.5 s after the start; planning with gradient refinement takes about 2.6 s. The final
  configuration was never driven from a standing start.
- **One integrated stack.** The pieces were validated in separate experiments:
  - the live depth-camera loop used an earlier depth-reading model without history;
  - the shared history model was tested only on f104;
  - the learned tracker never drove a planner's route.

  Every risk label describes drives by the PID follower.
- **Calibrated probabilities.** The networks rank routes well, but their predicted failure probabilities are far too
  low for the routes they pick. For the Polaris: about 0.0014 % predicted, 0.25 % observed.
- **New kinds of terrain or soil.** All arenas come from one hill-and-crater generator. All soil results use one
  setting: a 0.24 m particle layer over a rigid floor and no contact between the vehicle body and the soil. The HMMWV
  meets the soil with a rigid tyre mesh; the Gator and Polaris with plain cylinders calibrated to the HMMWV's sinkage.
- **An untouched confirmation set for milestone 2.** The same 800 f104 pairs were used by successive studies. At least
  25 planner configurations were driven on them while the final one was being chosen.

## Where things are

| What | Where |
|---|---|
| Milestone evidence, scope and limits | [docs/milestones.md](docs/milestones.md) |
| Architecture | [docs/architecture.md](docs/architecture.md) |
| History of the research line, including the earlier vision/NRD work | [docs/history.md](docs/history.md) |
| Claim → record map; code and commands behind each result; release status of data and models | [docs/evidence.md](docs/evidence.md) |
| Per-task outcomes behind every headline | [results/](results/README.md), about 1 MB of CSV |
| Code, reports, logs and plans at the time of each result | experiment commit [`901d6c9`][commit] |
| Datasets and trained models | not yet published; [docs/evidence.md](docs/evidence.md#release-status) lists what exists and where |

Recount the headline numbers from the per-task tables (every headline except the Gator's rigid-ground row, which
comes from that study's read-out). This needs Python 3.8 or newer and nothing else:

```bash
python traversing/scripts/recount_milestones.py               # per-arm counts and checks
python traversing/scripts/recount_milestones.py --check-only  # verdict only
```

## What comes next

Each step will be its own pull request, starting from main:

1. **This PR: documentation.** The milestones, architecture, history and evidence, plus the compact outcome tables
   and the recount script.
2. **Data and models.** Datasets and trained model files go to new Hugging Face repositories with pinned revisions.
   Their manifests, checksums and a download helper go in this folder. The paper's dataset
   ([harryzhang1018/NeDM](https://huggingface.co/datasets/harryzhang1018/NeDM)) stays as it is.
3. **The planning pipeline.** The minimal collection, training, planning and evaluation code behind milestones 1, 2
   and 4, checked against saved outputs.
4. **The NRD/PPO tracker benchmark** (milestone 3), with its positive rigid-ground result and its failed soil result.

None of these changes the published HMMWV, M113 base or arm code, configs, checkpoints or dataset instructions.

## Relation to the rest of the repository

- **The paper's study cases are unchanged.** They cover HMMWV trajectory tracking with a 15-number reduced state at
  100 Hz, M113 base goal reaching and arm reaching. The HMMWV work here is different: the planner uses no dynamics
  model at all, and the tracker's NRD uses a 17-number state at 50 ms.
- **The earlier vision/NRD traversal work** (an HMMWV under an overhead RGB-D camera, planning by imagined drives;
  numbered Study 3 in the NRD planning documents, not a paper study case) is where this study started. [docs/history.md](docs/history.md) summarizes it, including why its
  planning approach was replaced.
- **Kyle Sha's Go2 quadruped study** (Study 4 in the same numbering, branch `kyle/quadruped-pipeline`) is independent
  low-level-control work.

[commit]: https://github.com/uwsbel/NeDM/tree/901d6c9423a16c0fafc3d60056065415d5a725f2
