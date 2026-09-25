# Rollout videos: soil and rigid drives of the decision-time study

These videos show how the vehicle drives from one start to one goal under six planner set-ups. Three start-goal pairs
are shown: two on soft soil and one on rigid ground. The set-ups differ in two ways: how long the vehicle drives
straight before the planner picks its route (3 s, 1 s or 0.5 s), and which risk model the planner uses (CNN-GRU or
transformer, with or without gradient refinement of the route).

There are two kinds of video.
- **Top-down comparisons** (`compare_*.mp4`) replay the drives exactly as the study recorded them. They are the
  reference.
- **3D chase-camera videos** (`chase_*.mp4`) are replays of three of the six drives, re-simulated on a different
  computer from the one that made the recordings. See "Can you trust the 3D replays?" below.

## The files

Durations are the playing time; each video holds its last frame at the end (2 s top-down, 3 s in 3D).

**Soil pair 0124** (start-goal pair `f104_crm_eval_group_0124`)
- `compare_soil.mp4`: top-down, all six planners side by side, recorded drives, real time. 17.4 s.
- `compare_soil_2x.mp4`: the same at double speed. 9.75 s.
- `chase_soil_H3s.mp4`: 3D replay of the CNN-GRU (earlier training) that decides after a 3 s approach. 15.7 s.
- `chase_soil_L1_X.mp4`: 3D replay of the transformer that decides after 1 s. 14.15 s.
- `chase_soil_HnG.mp4`: 3D replay of the final planner, the CNN-GRU with gradient refinement, deciding after 0.5 s. 16.85 s.
- `chase_soil_side_by_side.mp4`: those three 3D replays next to each other on one clock. 16.85 s.

**Soil pair 0500** (`f104_pair_group_0500`)
- `compare_soil2.mp4`: top-down, all six planners, recorded drives, real time. 36.05 s.
- `compare_soil2_2x.mp4`: the same at double speed. 19.05 s.
- `chase_soil2_H3s.mp4`: 3D replay of the CNN-GRU (earlier training), 3 s approach. 37.0 s.
- `chase_soil2_L1_X.mp4`: 3D replay of the transformer, 1 s. 11.6 s.
- `chase_soil2_HnG.mp4`: 3D replay of the final planner (CNN-GRU + gradient, 0.5 s). 10.55 s.
- `chase_soil2_side_by_side.mp4`: those three 3D replays on one clock. 37.0 s.

**Rigid pair 0011** (`f104_pair_group_0011`)
- `compare_rigid.mp4`: top-down, all six planners, recorded drives, real time. 27.85 s.
- `compare_rigid_2x.mp4`: the same at double speed. 14.95 s.
- `chase_rigid_H3s.mp4`: 3D replay of the CNN-GRU (earlier training), 3 s approach. 28.8 s.
- `chase_rigid_L0p5_X.mp4`: 3D replay of the transformer that decides after 0.5 s. 12.35 s.
- `chase_rigid_HnG.mp4`: 3D replay of the final planner (CNN-GRU + gradient, 0.5 s). 16.7 s.
- `chase_rigid_side_by_side.mp4`: those three 3D replays on one clock. 28.8 s.

In git: everything above except the three single rigid 3D videos (8-17 MB each), which stay on the workstation
(luffy) together with `chase_work/`; the side-by-side video contains the same three replays.

Other files:
- `compare_*_final.png`: the last frame of each top-down video.
- `compare_*_checks.json`: what the top-down script checked about its inputs.
- `manifest.json`: which recorded drive each panel shows.
- `chase_reproduction.json`: how each 3D replay compares with its recording.
- `VERIFY_*.md`: the checker reports.
- `EXPLAIN_soil2_old_protocol_stall.md`: why the old protocol gets stuck on soil pair 0500.
- `chase_work/`: the replays' raw frames and runs (working files, not for viewing; local only).

## What the file-name codes mean

| code | plain label | when it decides | risk model |
|---|---|---|---|
| `H3s` | CNN-GRU (earlier training), decides after a 3 s approach (old protocol) | after 3 s of straight driving | the previous study's CNN-GRU |
| `L1_Hn` | CNN-GRU, decides after 1 s | after 1 s | retrained CNN-GRU |
| `L1_X` | Transformer, decides after 1 s | after 1 s | transformer |
| `L0p5_Hn` | CNN-GRU, decides after 0.5 s | after 0.5 s | retrained CNN-GRU |
| `L0p5_X` | Transformer, decides after 0.5 s | after 0.5 s | transformer |
| `HnG` | CNN-GRU + gradient refinement, decides after 0.5 s (final) | after 0.5 s | retrained CNN-GRU; its route is then refined against the model's risk |

"Earlier training" means the previous study's CNN-GRU, trained on that study's data only. Every other CNN-GRU panel
uses the retrained model, which had about twice as many training examples (217,814 against 105,193): the same data
plus examples at the early decision points (0.5, 1 and 1.5 s), examples at the 3 s decision point, and a few
thousand examples from drives that were replayed part-way and then continued along different routes.

The old-protocol panel therefore differs from the others in two ways: it decides later, and it uses the earlier model.

## What to watch for

**Soil pair 0124: both the decision time and the model change the outcome at 1 s.**
- The old 3 s protocol bogs down on the climb (12.25 s).
- At 1 s the two models split: the CNN-GRU bogs down (12.75 s), but the transformer reaches the goal (12.85 s).
- Every planner that decides after 0.5 s reaches the goal (11.45-15.35 s). The final planner is the slowest of them
  (15.35 s).
- Watch the diamond in each cell (where the planner decided), then the dashed route it picked from there. The two
  failures stop about 13-16 m into a drive of about 35 m.

**Soil pair 0500: the old 3 s protocol stalls.**
- The old protocol gets stuck at 5.0 s (5.1 s in its 3D replay), 2 s after its decision. It stays stuck until the
  no-progress rule ends the drive at 34.0 s, the earliest time that rule can end a drive.
- Every planner that decides after 1 s or 0.5 s gets through (7.45-9.95 s).
- Why it stalls, in two sentences: when it decides at 3 s the vehicle is already about 1.5 m (front-left wheel) from
  the wall of a crater, and the chosen route asks for a sharp right turn at lower speed that it cannot make in that
  space, so the front-left wheel runs onto the wall and the rear-right wheel lifts off the ground. With open
  differentials, all the engine's torque then escapes through that airborne wheel, which spins while the two wheels
  carrying nearly all the weight stand still (the lightly loaded front-left wheel also spins, until 8.5 s); the
  vehicle stays at full throttle, without moving, for 29 s: grip is the limit, not soil depth or throttle.
- The planners that decide early make the same kind of right turn, but on the start pad, and pass 2-3 m south of
  the spot where the old protocol stalls.

**Rigid pair 0011: everyone arrives; the speed differs.**
- The two transformers arrive in 9.35 and 9.55 s, peaking near 7 m/s.
- The retrained CNN-GRU planners arrive in 13.7-16.9 s.
- The old 3 s protocol arrives in 25.8 s. After about 10 s it crawls at about 1 m/s.
- All six drive about 45 m.

## How to read the grids

**Top-down comparisons.**
- There is one cell per planner in a 2 x 3 grid, all on the same clock from the start of the drive.
- Every cell shows the same map crop, turned so that the start is on the left and the goal on the right. The 10 m bar
  gives the scale.
- Map marks:
  - grey line: the straight approach before the decision;
  - diamond: where the planner decided;
  - dashed coloured line: the route it planned at that moment;
  - solid coloured line: where the vehicle actually drove;
  - tan tint: ground steeper than 15 degrees;
  - thin lines: 1 m height contours;
  - circle: the goal. It fills with the planner's colour when the goal is reached. A red outline on the vehicle means
    the drive ended short of the goal.
- Above each cell: the time and forward speed while driving, then the outcome.
- The strip at the bottom plots every planner's forward speed over time. The short ticks above it mark when each
  planner decided. A dot marks a goal reached; a cross marks a drive that ended short of the goal.

**3D videos.**
- The camera follows the vehicle from behind.
- Markers in the scene: white spheres show the straight approach, orange spheres the planner's route (from the
  decision on), and the violet ring the goal.
- The header shows:
  - the planner;
  - time and speed;
  - on soil, the wheel sinkage against the bogged-down limit;
  - a phase line: approach, planner in control, or (red) stuck. "Stuck" means the vehicle has been below 0.3 m/s for
    2 s.
- The minimap (bottom right) shows the same marks from above.
- The footer and header say how this replay compares with its recording.
- The end card gives the replay's outcome, and also the recorded outcome when the two differ.
- On soil, "sinkage" is how far the deepest wheel has sunk below the untouched soil surface. A drive counts as bogged
  down when a wheel stays more than 0.30 m down (through the 0.24 m soil layer) for 0.25 s.

## Can you trust the 3D replays?

- **Top-down videos:** exact recordings of the study's drives. They are the reference; use them for any comparison.
- **3D videos:** the same drives re-simulated on a different computer. On soil the results can differ from the
  recording. On this computer the same replay gave the same result every time it was run, but that result is not
  always the recorded one.
  - **Rigid pair 0011:** reproduces: same outcomes and end times within 0.05 s, paths within 9-20 cm.
    - All three filmed replays end the same way at exactly the recorded times, on paths within 9-20 cm of the
      recordings (end points included).
    - Of the three planners without a 3D video, re-driven without a camera, two end 0.05 s later than recorded: the
      1 s transformer and the 0.5 s CNN-GRU.
  - **Soil pair 0124:** does not reproduce.
    - The 1 s transformer bogs down in the replay (11.15 s), though it reached the goal in the recording (12.85 s).
      Its 3D video says so on every frame.
    - The old protocol bogs down 0.45 s later and 4.1 m further back along the same path.
    - The final planner arrives 1.5 s earlier.
    - Treat these three videos as a picture of soft-soil driving, not as evidence for the comparison.
  - **Soil pair 0500:** reproduces.
    - All three replays end the same way as the recordings.
    - The old protocol is stuck at the same spot (within 11 cm), from 5.0 s in the recording and from 5.1 s in the
      replay; the drive ends at 34.0 s in both.
    - The transformer and the final planner arrive 0.55 s and 0.1 s later than recorded.
    - This pair was filmed because, of the five soil pairs screened, it was the only one whose six replays all
      ended the same way as recorded, within 1 s of the recorded times (at most 0.55 s off here). The screen stopped
      there.

## Changes in this render (2026-09-24)

- The old-protocol panels are labelled "CNN-GRU (earlier training)", with the footnote line explaining it.
- In the 3D videos, a replay that differs from its recording now says how:
  - the end-point gap when it is over 1 m;
  - "bogs down instead" for the opposite outcome;
  - a red "stuck" phase line, and an end card that gives when the vehicle got stuck.
- The side-by-side subtitles now name each difference and the top-down video that holds the recordings.
- The sinkage explanation is now in plain words. The on-screen text no longer names the machines. Times use one format
  everywhere (12.85 s, 12.7 s, 34.0 s).
- In `chase_reproduction.json`:
  - `max_distance_to_recorded_path_m` is now the distance to the recorded path line, not to the nearest recorded
    position;
  - the screening notes' wording is corrected.
- Wording round after the third check (same day):
  - The footnote under the old-protocol title is shorter and says the retrained model had about twice the data; it
    now speaks of the other CNN-GRU "planners", not "panels". The rigid top-down header says "retrained".
  - When a replay gets stuck at another time than its recording, the footer gives both times (soil pair 0500: 5.0 s
    recorded, 5.1 s replay).
  - The path distance of a replay that ends at the recorded time now includes the end points, so the rigid old
    protocol reads 20 cm (was 19 cm) and the rigid side-by-side "9-20 cm".
  - The screening section of `chase_reproduction.json` carries the current old-protocol title.
