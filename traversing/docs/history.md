# History of the research line

How the study got from "plan by imagining drives with a learned dynamics model" to "learn route risk directly", and
what each stage found, including what did not work. Dates are 2026. Each stage's commit is on `origin`; every earlier
branch is an ancestor of the experiment commit [`901d6c9`][commit]. The full running record is the experiment
branch's [progress document][progress] ("Follow-on project" sections).

## Timeline

| Dates | Stage | Branch, commit | Question | Answer |
|---|---|---|---|---|
| 08-25/26 | Double pendulum with a camera (NRD Study 1) | `nrd_vision` [`82fb955`][c-82fb], [`69fbb2c`][c-69fb] | Can a learned camera latent be appended to the explicit state and predicted jointly? | Yes, as an architecture test. The camera was redundant by design, so it says nothing about vision helping control |
| 09-01..07 | Vision/NRD HMMWV traversal (NRD Study 3, work packages 0-9) | `nrd_vision` [`51839e1`][c-5183]; energy work [`6abf6ee`][c-6abf], [`4b6f2a1`][c-4b6f] | Can a world model combining camera and state choose feasible, efficient routes that simple methods misjudge? | **No for route selection.** Cheap terrain-profile predictors and a fastest-speed rule did as well or better |
| 09-08/09 | RGB-D forecaster with MPPI | `traverse_mppi` [`76aa590`][c-76aa] | Can a finite-horizon, route-conditioned forecaster with MPPI plan long traversals? | Only partly: 4/6 unseen arenas completed safely, against 3/6 without the image |
| 09-09..15 | Learned route risk on rigid ground, f104 | `traverse_mppi` [`dc366e2`][c-dc36] | Does a network trained on many recorded Chrono drives of one arena pick safe routes? | Yes at a fixed 2 m/s and in five-goal missions. On single f104 goals with speed free it never significantly beat driving straight at 6 m/s, and on new arenas it tied a hand-made rule |
| 09-16/17 | Continuous depth-camera navigation | `traverse_mppi` [`63f5c0c`][c-63f5]; corrected re-run on `crm_improve_v1` [`0d89977`][c-0d89] | Can the planner run on fresh depth frames inside one continuous rollout? | Yes: 27/30 missions. Faster replanning did not help (**milestone 1**) |
| 09-16/17 | First soil study | `traverse_mppi` [`c81c3f9`][c-c81c] | Does the pipeline work on CRM soil? | Soil-trained planner 91.0 % against 68.0 % for the rigid-trained one, 200 pairs |
| 09-17/18 | Second soil study | `traverse_mppi` [`aa32bd8`][c-aa32] | Better network? Better search? | Network: no. Search: iterated sampling 98.5 % and gradient refinement 99.5 % against one-shot 91.0 % |
| 09-21/22 | Shared rigid/soil model; learned tracker | `generalist_v1` [`db7a9e8`][c-db7a] | One model for both grounds without a label? A tracker trained inside an NRD that beats PID? | Shared model non-inferior after a 3 s approach but not from rest. The tracker followed routes more closely than PID on rigid ground, completed fewer routes on soil, and failed the predeclared replacement test on both grounds (**milestones 2, 3**) |
| 09-22/24 | Soil success from a moving start | `crm_improve_v1` [`4bc4438`][c-4bc4] | Why only 83.9 % on soil after the approach? | The approach itself. Deciding after 0.5 s plus gradient refinement: 97.5 % soil, 100 % rigid (**milestone 2**) |
| 09-25/26 | More training arenas; the Gator | `arena_gator_v1` [`051dfdb`][c-051d] | Do more arenas help on unseen ones? Does the pipeline work for the Gator? | A small gain on unseen arenas. The Gator's own planner reached 67.4 %, limited mostly by the vehicle and partly by an over-optimistic model (**milestone 4**) |
| 09-27/28 | Polaris and M113 on soil | `offroad_vehicles_v1` [`901d6c9`][commit] | Do they get through soil better than the Gator? Does a Polaris planner meet the 90 % bar? | Polaris yes on both. M113 only when re-geared, and costly (**milestone 4**) |

## The earlier planning approach and why it was replaced

**Double pendulum (Study 1).** A Chrono double pendulum with a camera tested whether a learned camera latent could be
appended to the explicit state and predicted jointly. A reaching policy trained inside the frozen model transferred to
Chrono with no gap: 87 % in the model and 87 % in Chrono on 100 held-out validation-bank pairs, with the task relaxed
to lower-half goals within 2 cm. A camera-only student matched it. The study plan itself states that the camera was
informationally redundant, so this does not show that vision improves control. Records: [implementation
notes][dp-notes], [reaching notes][dp-rl], [distillation notes][dp-distill], [study plan][dp-plan].

**Vision/NRD traversal (Study 3).** An HMMWV crossed small arenas under a fixed overhead RGB-D camera.

- **The model.** A learned dynamics model combined an explicit vehicle state (15 numbers, later 17 with engine speed
  and torque) with a camera-derived terrain token. That token became a patch of the overhead map around the imagined
  vehicle position. A tracker was trained inside the model.
- **Planning.** Candidate routes were chosen by imagining each drive and scoring time, energy and feasibility.
- **What worked:**
  - camera-only localisation on 32/32 routes;
  - camera-planned routes with zero contact in 622/622 Chrono runs;
  - a sampling planner that beat plain A* on 32 held-out layouts.
- **What did not.** The imagination-based selection did not beat cheap predictors:
  - On sealed arenas, a fastest-speed rule made 86/93 feasible selections, against 82/93 for the fine-tuned world
    model.
  - With the tracker in the loop, stall foresight stayed at AUC 0.65-0.71.
  - On four fresh sealed arenas, the imagination's stall head reached AUC 0.76. A simple terrain-profile predictor
    reached 0.81-0.82.
  - In the energy branch, a hand-written analytic work model chose cheaper routes than the imagination.
- **The conclusion recorded then:** on this terrain family, feasibility is a terrain-profile question. Records:
  [study plan][trav-plan] (sections 29-35), [work-package 4 notes][wp4].

**RGB-D forecaster with MPPI (transitional).** A finite-horizon forecaster was conditioned on the proposed route and
speed profile. From the overhead RGB-D image and recent vehicle history, it predicted 4-12 s of motion, work and event
probabilities, and an MPPI optimiser chose the route for the PID follower. On 36 generated 240 m arenas it safely
completed 4/6 unseen test arenas, against 3/6 for a matched model without the image. A failure investigation found
three separate problems:

- the forecast cannot see hazards beyond its 12 s horizon;
- it misses some hazards inside the horizon;
- MPPI can exploit a prediction error to prefer a physically worse route.

Direct route risk started the next day. Records: [results][fdm-results], [failure investigation][fdm-failure].

## The current approach, stage by stage

**Rigid ground on f104 (09-09..15).** The approach: overfit one 80 m x 80 m arena deliberately, with enough recorded
drives to learn where routes fail. Records: [first rigid-ground study log][n1-log], [second rigid-ground study
report][n2-report], [new-arena report][gen-v1-report], [sensor report][sensor-report].

- **The station-resolved hazard model** (09-10). Trained with roll-back labels and tested at a fixed 2 m/s against the
  previous risk head on 123 held-out groups, it cut failures from 15.4 % to 6.5 % and unsafe runs from 44.7 % to
  17.9 %.
- **A wider route proposal plus a retrained model** (09-12). On 300 hill/crater groups with speed free, against the
  first f104 route-risk planner, failures fell from 3.7 % to 0.3 % and unsafe runs from 9.7 % to 0.3 %. Changing only
  the route proposal took unsafe runs from 29 to 13; the retrained model on the same candidates took them to 2.
- **A result that did not hold.** On single f104 goals with speed free, the planner never significantly beat driving
  straight at 6 m/s on its own label (0.3 % vs 1.3 % unsafe on the hazard test); its only significant edge there was
  less body tilt. Its clear wins were at a fixed 2 m/s, in five-goal missions, and against the straight route on new
  arenas.
- **Five-goal missions.** 99 % completed, against 91 % for a hand-made terrain-and-speed rule and 90 % for the straight
  route.
- **Five new arenas.** The pre-registered test against the hand rule was null: 1.3 % against 1.8 % failed or slid,
  p = 0.42.
- **Depth input.** The corrected depth-to-map conversion cut the mean height error about 4-5 times (0.027-0.042 m to
  0.0071-0.0087 m on six arenas), but no planning gain was demonstrated.

**Continuous navigation (09-16/17).** See [milestone 1](milestones.md#1-a-chrono-depth-camera-in-the-planning-loop).
The first cluster campaign's arena exits all came from a runner defect: emergency routes could double back on
themselves. The defect was found in an audit of the drive videos and fixed, and the 120 drives were re-run.
[Report][nav-report].

**First soil study (09-16/17).** Records: [report][crm1-report].

- **Collection.** The same arena, follower and route pool, driven on Chrono CRM: 15,235 drives, 91.5 simulated hours,
  in 2 h 39 min on about 111 GPUs.
- **The result.** The soil-trained planner reached 91.0 % of 200 new pairs, against 68.0 % for the frozen
  rigid-trained planner and 66.5 % for the straight route.
- **Caveats.** The failures cluster at about 12 spots on 9 terrain features, so the honest p-value is about 0.003-0.008.
  "Held out" is interpolation: 99.9 % of the chosen route points lie within 1 m of a training route.

**Second soil study (09-17/18).** Records: [report][crm2-report].

- **Network design.** Nine designs at equal data; none beat the CNN-GRU.
- **Search.** Iterated sampling (98.5 %) and gradient refinement (99.5 %) beat one-shot ranking (91.0 %) on soil.
- **Vehicle speed as an input.** The model did not learn a real speed effect. Measured failure was flat in starting
  speed, while the models predicted it falling.
- **Energy.** An analytic energy term in the objective bought time, not energy.

**Shared model and learned tracker (09-21/22).** See milestones
[2](milestones.md#2-one-risk-model-for-rigid-ground-and-soil-and-the-decision-timing-finding) and
[3](milestones.md#3-a-learned-route-tracker-trained-inside-the-nrd-against-pid). [Report][gen-report].

**Early decision (09-22/24).** See [milestone 2](milestones.md#2-one-risk-model-for-rigid-ground-and-soil-and-the-decision-timing-finding).
Records: [report][ci-report], [rollout videos][ci-videos].

- **Refuted:** starting routes at the vehicle's current speed (worse); soil-heavy training batches (no effect);
  7,182 + 7,182 continuation drives from 3 s decision states (no offline gain).

**More arenas and the Gator (09-25/26), Polaris and M113 (09-27/28).** See
[milestone 4](milestones.md#4a-arenas-the-planner-never-trained-on-hmmwv). Records: [terrain and Gator report][ag-report],
[Polaris and M113 report][ov-report]. The Gator and M113 work was stopped on 09-28 to focus on the Polaris.

## Cost

| Stage | Billed cluster node-hours |
|---|---|
| First soil study | about 37 |
| Second soil study | 21.9 |
| Shared model and tracker | 29 |
| Early decision | 39.0 |
| More arenas and the Gator | 113.3 |
| Polaris and M113 | about 54 |

Earlier stages mostly did not record their billed node-hours. The second rigid-ground study reports about 4; the other
rigid-ground rounds, the live navigation study and the vision and forecaster stages give none.

## Naming

- **Studies.** Kyle Sha's documents number the NRD studies Study 1 (double pendulum), Study 3 (HMMWV traversal) and
  Study 4 (Go2 quadruped). This study grew out of Study 3, but its planner uses no NRD.
- **Arenas.** "f104" is seed 104 of the hill-and-crater generator. "g203", "g228" and similar names are other seeds
  of the same generator.
- **Early names on the experiment branch.** They called the rigid-ground studies "nights" and some planners "MPPI". The
  current planners use one-shot ranking, iterated sampling (cross-entropy method) or gradient refinement; MPPI
  belongs only to the forecaster stage.

[commit]: https://github.com/uwsbel/NeDM/commit/901d6c9423a16c0fafc3d60056065415d5a725f2
[progress]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/docs/progress.md
[c-82fb]: https://github.com/uwsbel/NeDM/commit/82fb955d694811fb77e9a1af618d62051608504e
[c-69fb]: https://github.com/uwsbel/NeDM/commit/69fbb2ce7855799e1dc8320521f5afd83f45082f
[c-5183]: https://github.com/uwsbel/NeDM/commit/51839e179a631ae93d5f25618dfb66ab33db6f9c
[c-6abf]: https://github.com/uwsbel/NeDM/commit/6abf6eecc22a86adff248f7b8c9c304db0e577a5
[c-4b6f]: https://github.com/uwsbel/NeDM/commit/4b6f2a173b9fc8a15b776c3f6134e349911de2e7
[c-76aa]: https://github.com/uwsbel/NeDM/commit/76aa5904e7c880659f73517062e231bb982df66f
[c-dc36]: https://github.com/uwsbel/NeDM/commit/dc366e2f11d4e8ac4ec41dd6ecc5773c6071cb1e
[c-63f5]: https://github.com/uwsbel/NeDM/commit/63f5c0c979ae0add5b64508b23c34a53d3a8917c
[c-0d89]: https://github.com/uwsbel/NeDM/commit/0d899772945c7fbafb2898498da780ad9149fb6b
[c-c81c]: https://github.com/uwsbel/NeDM/commit/c81c3f9f338e88e36467d588e56735b989488203
[c-aa32]: https://github.com/uwsbel/NeDM/commit/aa32bd8d38e1b0a5a4abd709298ccca5329dbf79
[c-db7a]: https://github.com/uwsbel/NeDM/commit/db7a9e864e7e873bc4106fd7da40e970973c11c5
[c-4bc4]: https://github.com/uwsbel/NeDM/commit/4bc44386db6390e767765744ef7bc7933a91c428
[c-051d]: https://github.com/uwsbel/NeDM/commit/051dfdba55b6ba0ef804984fc1ace3a028e3afc7
[dp-notes]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/docs/vision/double_pen/implementation_notes.md
[dp-rl]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/docs/vision/double_pen/rl_implementation_notes.md
[dp-distill]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/docs/vision/double_pen/distillation_implementation_notes.md
[n1-log]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/fdm_f104_50h_20260909/night_v1/LOG.md
[dp-plan]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/docs/vision/double_pen/NRD_double_pendulum_study_plan.md
[trav-plan]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/docs/vision/hmmwv_traverse/NRD_hmmwv_traversal_study_plan.md
[wp4]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/docs/vision/hmmwv_traverse/wp4_implementation_notes.md
[fdm-results]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/docs/vision/hmmwv_traverse/fdm_diverse_results_20260909.md
[fdm-failure]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/docs/vision/hmmwv_traverse/fdm_failure_investigation_20260909.md
[n2-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/fdm_f104_50h_20260909/night2_v1/REPORT.md
[gen-v1-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/fdm_f104_50h_20260909/gen_v1/REPORT.md
[sensor-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/REPORT.md
[nav-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/fdm_f104_50h_20260909/nav_v1/REPORT.md
[crm1-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/crm_f104_v1/REPORT.md
[crm2-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/crm_night2_v1/REPORT.md
[gen-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/generalist_20260921/REPORT.md
[ci-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/crm_improve_20260922/REPORT.md
[ci-videos]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/crm_improve_20260922/videos/README.md
[ag-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/arena_gator_20260925/REPORT.md
[ov-report]: https://github.com/uwsbel/NeDM/blob/901d6c9423a16c0fafc3d60056065415d5a725f2/artifacts/traverse/offroad_vehicles_20260927/REPORT.md
