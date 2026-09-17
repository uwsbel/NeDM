# Five scenarios, three routes each (f104 arena, night-2 planner)

Each scenario is one held-out start/goal pair on the hill-and-crater test set. For each, the planner
generated 256 candidate routes, scored every one with the deployed 5-seed risk model, and three were
then driven in Chrono: the model's best, a middling one, and the model's worst. Same model, same
candidate pool, three ranks -- so the videos are a direct read on whether the predicted ordering
matches the simulator.

Every filmed run reproduced its screening run exactly (status and elapsed time to 0.05 s).

| scenario | route | predicted risk | Chrono outcome | time | max tilt | backwards |
|---|---|---|---|---|---|---|
| 1 | planner's choice | 0.02% | goal reached | 10.3 s | 25 deg | 0.0 s |
| 1 | middling alternative | 12.10% | goal reached | 28.2 s | 28 deg | 8.3 s |
| 1 | worst-ranked | 100.00% | prolonged blockage terminated | 96.5 s | 13 deg | 32.9 s |
| 2 | planner's choice | 0.01% | goal reached | 19.4 s | 26 deg | 0.0 s |
| 2 | middling alternative | 10.58% | goal reached | 31.8 s | 23 deg | 0.0 s |
| 2 | worst-ranked | 100.00% | prolonged blockage terminated | 37.1 s | 23 deg | 5.1 s |
| 3 | planner's choice | 0.12% | goal reached | 19.8 s | 16 deg | 0.0 s |
| 3 | middling alternative | 9.92% | goal reached | 12.7 s | 30 deg | 0.0 s |
| 3 | worst-ranked | 100.00% | prolonged blockage terminated | 40.8 s | 14 deg | 10.2 s |
| 4 | planner's choice | 0.01% | goal reached | 11.8 s | 10 deg | 0.0 s |
| 4 | middling alternative | 9.95% | goal reached | 17.1 s | 25 deg | 0.0 s |
| 4 | worst-ranked | 100.00% | prolonged blockage terminated | 64.5 s | 26 deg | 15.0 s |
| 5 | planner's choice | 0.07% | goal reached | 10.1 s | 29 deg | 0.0 s |
| 5 | middling alternative | 10.05% | goal reached | 21.6 s | 33 deg | 0.0 s |
| 5 | worst-ranked | 100.00% | timeout | 120.0 s | 28 deg | 47.0 s |

## Files

`scenarios.png`  the five scenarios on the terrain map: dashed = proposed route, solid = driven path.

`s1_1_optimal.mp4`
`s1_2_suboptimal.mp4`
`s1_3_risky.mp4`
`s2_1_optimal.mp4`
`s2_2_suboptimal.mp4`
`s2_3_risky.mp4`
`s3_1_optimal.mp4`
`s3_2_suboptimal.mp4`
`s3_3_risky.mp4`
`s4_1_optimal.mp4`
`s4_2_suboptimal.mp4`
`s4_3_risky.mp4`
`s5_1_optimal.mp4`
`s5_2_suboptimal.mp4`
`s5_3_risky.mp4`

The green/amber/red ribbon on the ground in each video is the route the planner asked for; the
yellow disc is the 2.5 m goal circle. Clips of stalled runs are cut at 35 s -- the vehicle was still
stuck when the episode was terminated.

## How these were produced

1. `scripts/f104_demo_pick.py` -- score the cached 256-candidate pool of 28 hazard groups with
   `night2_v1/final/N2_s*.pt` and stage best / middling / worst per group.
2. Cluster screening (`demo_night2_v1`, 84 episodes): drive all three per group with the frozen
   collector `source_v1/scripts/collect_traverse_f104.py`; all three arms of a group share one array
   task because Chrono is only deterministic per node.
3. `scripts/f104_demo_select.py` -- keep the five scenarios where the trio tells three different
   stories, spread over the arena.
4. Cluster re-run with Chrono's Blender exporter attached to the collector's own observer
   (`demo_blend_v1`), which reproduces the recorded status and elapsed time exactly.
5. `scripts/f104_demo_render_blender.py` + `f104_demo_videos.py` -- render locally with Blender
   (EEVEE) and encode with ffmpeg captions.

## Caveats

- One arena. Every route in these videos was driven during training as part of some other episode;
  'held out' here means the start/goal pair, not the terrain.
- The model is a good ranker but badly calibrated: across the wider hazard test the median predicted
  risk was 0.008% against a realised 0.33%. Read the numbers as an ordering, not a probability.
- The routes are chosen once, before the run. There is no online replanning.
