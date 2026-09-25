# Data-accuracy check of the rollout videos, round 2 (2026-09-24)

**Verdict: pass.** I tried to find a place where a video disagrees with the recorded data and found none. Every
vehicle position, status line, speed trace, header sentence and decision marker I sampled matches the recorded
files, and so does every number in the reproduction file. Findings 1 and 2 are wording slips in
`chase_reproduction.json` that do not appear on screen. Finding 3 is a labelling caveat. Findings 4-6 are notes.

Scope:
- the second soil case (`soil2`, group f104_pair_group_0500): manifest entry, `compare_soil2.mp4`, `compare_soil2_2x.mp4`,
  `compare_soil2_final.png`, the four `chase_soil2_*` videos and their entries in `chase_reproduction.json`;
- a spot check of the re-rendered `compare_soil.mp4` and `compare_rigid.mp4`.

Nothing was re-simulated. Frames were extracted with ffmpeg into a scratch folder under /tmp, which I deleted
afterwards. The pixel checks rebuilt each figure's axes from `scripts/ci_video_compare.py` and, separately, from my
own implementation of the rotation, crop and axes boxes. The two mappings agree to 0.003 px.

## 1. Is the `soil2` manifest entry the right six drives?

Yes. All six run folders are the recorded cluster drives of f104_pair_group_0500 for the named set-ups. They were
looked up exactly the way the `soil` and `rigid` entries were, and I cross-checked them independently:

| set-up (cell title) | recorded run | index entry that points to it | decision frame |
|---|---|---|---|
| CNN-GRU, 3 s approach (old protocol) | generalist `a5/crm_pass2_runs/…__H_B` | `a5/run_index_crm_pass2.json` (also in `s2/idx/run_index_vs3s_crm.json`, `s4/run_index_s4.json`) `…__H_B` | 60 |
| CNN-GRU, 1 s | `s2/runs_crm/…__L1_Hn_B` | `s2/idx/run_index_crm.json` `…__L1_Hn_B` | 20 |
| Transformer, 1 s | `s2/runs_crm/…__L1_X_B` | same index, `…__L1_X_B` | 20 |
| CNN-GRU, 0.5 s | `s2/runs_crm/…__L0p5_Hn_B` | same index, `…__L0p5_Hn_B` | 10 |
| Transformer, 0.5 s | `s2/runs_crm/…__L0p5_X_B` | same index, `…__L0p5_X_B` | 10 |
| CNN-GRU + gradient, 0.5 s | `s4/runs/…__L0p5_HnG_B` | `s4/run_index_s4.json` `…__L0p5_HnG_G` (driven as `…_HnG_B`) | 10 |

For each of the six drives I checked the following:
- **Identity:** the outcome's case id is the group, and the status and elapsed time equal the manifest.
- **Decision frame:** it is 60/20/10 in three places that all agree: the outcome, `trajectory.npz` and the task row's `--branch-frame`.
- **Route file on the cluster:** it sits under the matching protocol folder (`generalist/a5/pass2`, `crm_improve/s2n/L1`,
  `crm_improve/s2n/L0p5` or `crm_improve/s4/L0p5`).
- **Route hashes:**
  - the local branch route file has the same file hash as `branch_route_sha256` and the same content hash as
    `branch_route_content_sha256`;
  - that content hash also equals the run index's sha256 for the arm;
  - the pick file names this route as the planner's pick (arm B; arm G for the gradient drive, whose metadata reads
    `planner_G_grad60_pessimistic`);
  - the approach route's file hash equals `prefix_route_sha256`.
- **Models:** the pick summaries confirm the models: joint transformer (`deploy_a3_haux_txjoint`) for the transformer
  cells, the GRU history model (`deploy_a1_haux_gru`) for the 1 s / 0.5 s / gradient CNN-GRU cells, and the K1 `H_deploy`
  ensemble for the 3 s cell (see finding 3).
- **Files unmodified:** `outcome.json` and `trajectory.npz` still carry the file hashes recorded in
  `episode_complete.json`. The case file's hash equals the recorded `case.json` hash.

I re-ran the same checks on the `soil` and `rigid` entries (12/12 pass). The rigid outcome stores the content hash
directly in `branch_route_sha256`.

## 2. `compare_soil2.mp4` and `compare_soil2_2x.mp4` against the recordings

Container: real-time 721 frames (680 steps + 1 + 40 hold); 2x 381 frames (341 + 40); both 1600x1000 at 20 fps.

Frames sampled:
- real time: 0, 10, 20, 60, 100, 149, 150, 161, 189, 191, 199, 200, 680 and 720 (last);
- 2x: 0, 5, 10, 30, 50, 74, 75, 81, 95, 96, 100, 340 and 380 (last).

These are the start, the 0.5/1/3 s decisions, 5.0 s, 10 s, 34 s, every arm's finish and the hold.

- **Vehicle position:** in each cell I found the vehicle as the arm-coloured blob that survives a 5x5 erosion (which
  removes the lines) and compared it with the recorded pose drawn the same way.
  - Across all 27 frames and 6 cells, the centroid error is at most 0.24 m and the heading error at most 2.8°.
  - The scale is 12.1 px/m. The recorded vehicle ends are shown frozen at their terminal pose.
  - Above 0.05 m, the error comes only from the decision diamond, which is drawn over the vehicle's rear. This covers
    the old-protocol cell from 5 s on (0.20-0.24 m; zoom viewed) and the 0.5 s cells at 1 s (0.17 m).
  - A 1 m shift of any track would be detected: the hit rate drops from 1.00 to about 0.10.
- **Lines:**
  - the decision diamond pixel carries the arm colour at the recorded decision pose (within 5/255);
  - the grey approach line lies on the recorded approach poses;
  - the driven track lies on the recorded poses (100 % of samples);
  - the dashed route lies on the branch route file (83-98 %, against 0-32 % when shifted 1 m).
- **Status lines** (viewed at 0, 0.5, 1, 3, 5, 7.4, 7.5, 8.05/8.1, 9.5, 9.6, 10 s and at the end, in both videos):
  - every "t s · speed v m/s" equals the recorded time and the forward speed `state[:,0]` at that step;
  - every finish text equals the outcome: goal reached in 9.55 / 8.05 / 9.45 / 9.95 / 7.45 s, and
    "stuck from 5.0 s; drive ended at 34.0 s (no progress)";
  - 34.0 − 5.0 s equals the outcome's `longest_consecutive_effortful_near_zero_speed_s` of 29.0 s.
- **Speed strip:**
  - I sampled every second recorded speed of every arm at 0, 0.5, 1, 3, 5, 10 and 34 s: 1,936 samples in each video.
    All carry the arm's colour, or lie under a later-drawn trace, except three distinct samples. Each of those sits
    2-3 steps before an end marker that covers it;
  - no future segment is drawn early;
  - the time cursor sits within 0.5 px of k × 50 ms.
- **Header sentence:** "The old 3 s approach gets stuck 2.0 s after its decision and stays stuck; every planner that
  decides after 1 s or 0.5 s gets through (7.45-9.95 s)." It is true:
  - decision at 3.0 s; speed below 0.3 m/s from 5.0 s to the end;
  - 0.078 m of movement after 5.0 s, with full throttle and full steering lock, and 95th-percentile wheel slip 814;
  - the five other drives all reached the goal, in 7.45-9.95 s.
  - "On the climb" is correctly not claimed: the vehicle stopped 0.11 m below the start on 1.8° ground.
  - The title reads "Soft soil (deformable ground)".
- **Still:** `compare_soil2_final.png` equals the last frame outside the playback label (blurred luma difference
  ≤ 5.4/255).

## 3. Spot check of the re-rendered `compare_soil.mp4` and `compare_rigid.mp4`

Frames checked: k = 150 and 200 plus the last frame (soil 347, rigid 556).
- **Vehicles:** within 0.07 m (soil) and 0.08 m (rigid) of the recorded pose; heading within 1.1°.
- **Speed strip:** matches. One sample in each video sits where two traces cross and has a chroma-blended colour.
- **Status lines:** the 10 s status lines equal the recorded speeds and outcomes.
- **Final frames:**
  - the finish texts equal the manifest: soil 12.25 / 12.75 / 12.85 / 11.45 / 12.7 / 15.35 s; rigid 25.8 / 16.9 / 9.55 / 15.3 / 9.35 / 13.7 s;
  - both header sentences are still true;
  - earlier fixes 3 (the "CRM" acronym) and 6 (the goal label covered by the vehicle) are applied;
  - the stills equal the last frames.

## 4. `chase_soil2_*` and `chase_reproduction.json`

- **Overlay data:** `overlay.json` has one row per local step (680 / 172 / 151 rows = frames = jpgs). Its time is
  frame × 50 ms. Its pose, forward speed and actions equal the local `run/trajectory.npz` exactly (difference 0.0).
  The camera-free overlay is identical to the filmed one.
- **On screen** (start, decision, decision + 5, mid, last step, hold, end card of each single video; side-by-side at
  0, 1, 3, 7.55, 8.6 and 34 s and the last frame): every time, speed and sinkage readout equals the overlay row.
  - The 1 s transformer and gradient videos carry "Recorded cluster run: goal reached in 8.05 s / 7.45 s; this local
    re-run differs (same outcome, 0.55 s / 0.10 s later)" on every frame, and their end cards show both outcomes.
  - The old-protocol video correctly shows no such line.
  - The side-by-side subtitle "end the same way … up to 0.55 s later" is true.
- **Reproduction statements:** I recomputed status, elapsed time, frame counts, position difference, terminal
  difference and distance to the recorded path with my own code, from the six local run folders listed. All values
  match exactly, and so do the `differs` flags:
  - 3 s: stuck at 34.00 s both, largest difference 0.108 m ("within 11 cm" is correct);
  - 1 s transformer: goal 8.60 s vs 8.05 s;
  - gradient: 7.55 s vs 7.45 s;
  - camera-free: the 1 s CNN-GRU is exact (9.55 s), the 0.5 s CNN-GRU is +0.05 s, the 0.5 s transformer is −0.20 s.
  - Filmed and camera-free local drives are bitwise identical in every array. They are also identical to the screening
    re-drives of the same group, so the local machine is deterministic.
  - I recomputed all 18 entries in the file (all three cases); every value matches.
- **Group screen table:**
  - all 30 screening rows (5 groups x 6 arms) recompute exactly from their local runs, including "matches",
    "all six", "filmed three" and the largest time difference;
  - the screening manifest entry for 0500 is identical to the videos manifest `soil2` entry;
  - tiers, candidate ranks (1-3 tier 0; 0093 = 13, 0500 = 22), success margins, stall times and the revised order
    (0093 then 0500) recompute from the recorded runs.

## Findings

1. **`chase_reproduction.json` → `soil_group_screen.revised_order_rule` (from `chase_work/screen/candidates.json`; source:
   docstring of `soil_candidates` in `scripts/ci_video_chase.py`, around line 207). Not on screen.**
   - Problem: the rule says it was "used after the first three screened groups" and that the local re-drives moved
     bogged-down end times "by -2.8 to +15.6 s". But `candidates.json` was written at 12:10:09, after groups 0356 and
     0244 (the last drive finished at 12:10:02) and before 0440 (12:11-12:15).
   - With 0440 the range is −2.8 to +23.6 s, which is what `order_note` in the same file says. The file therefore
     states two different ranges.
   - Fix: change the docstring to "after the first two screened groups" (or give the range −2.8 to +23.6 s), then
     rerun `soil-candidates`, `screen` and `repro`. Only the text changes.
2. **`chase_reproduction.json` → `soil_group_screen.order_note`. Not on screen.**
   - Problem: "in all three the recorded bogged-down failures did not reproduce (local end times -2.8 to +23.6 s off,
     or stuck instead of bogged)" leaves out one way of not reproducing. In group 0244 the 1 s CNN-GRU's recorded
     bog-down at 13.35 s reached the goal locally, at 12.60 s.
   - Fix: add "or reached the goal (0244, 1 s CNN-GRU)".
3. **All compare videos (cell titles and legend) and the 3D videos' titles, whole duration. A labelling caveat, not a
   data mismatch.**
   - Problem: "CNN-GRU" covers two different trained ensembles:
     - the 3 s cell drives the earlier generalist history model (`generalist_20260921/A_adapt/train/deploy_v1/H_deploy_s*.pt`);
     - the 1 s, 0.5 s and gradient CNN-GRU cells drive the retrained short-window-history model
       (`crm_improve_20260922/deploy_v1/deploy_a1_haux_gru_s*.pt`).
   - So the old-protocol cell differs from the others in model as well as in decision time. This matches the study's
     baseline ("after a 3 s approach (K1 protocol)" in REPORT.md), but a viewer will read "CNN-GRU" as one model.
     The report's own 1 s "history" figure (93.2 %) is the earlier model, whereas the video's 1 s CNN-GRU is the
     retrained one (92.8 %).
   - Fix: title the 3 s cell "CNN-GRU (earlier ensemble), 3 s approach (old protocol)". Alternatively, add one footnote
     line: "the 1 s and 0.5 s CNN-GRU planners use the retrained short-history model".
4. **Note, no fix needed: `compare_soil2*.mp4`, old-protocol cell, 5.0 s to the end.** The decision diamond covers the
   rear of the stuck vehicle, so the vehicle looks about 0.2 m further forward than its centre. The recorded pose is
   drawn correctly (checked by pixel and by zoom).
5. **Note, precision: `chase_reproduction.json` → `max_distance_to_recorded_path_m` (all entries).**
   - The value is the distance to the nearest recorded 50 ms position, not to the recorded path line. It is therefore
     an upper bound; for example, the 1 s CNN-GRU gives 0.079 m against 0.024 m to the line.
   - Nothing on screen uses this number: the footer's "within 11 cm" comes from the equal-time difference.
   - Fix (optional): name it `max_distance_to_recorded_positions_m`, or compute the distance to the segments.
6. **Housekeeping: `VERIFY_videos.md` (11:54) describes the previous render.** Its fixes 1, 3, 4 and 6 are now applied:
   - recorded outcome on the side-by-side cards;
   - no "CRM" acronym in the headers;
   - no "-0.0" speeds;
   - the goal label moved below the circle when the vehicle covers it.

   Add a line pointing to this file so readers do not act on the stale items.

## Also checked and fine

- **Wheel sinkage at the start:** the 3D readout shows 0.10 m at 0.00 s in all soil2 videos. This is the settled
  vehicle at rest. The collector's own maximum for the same drives is 0.095 m, and the old-protocol maximum of 0.22 m
  equals the collector's 0.217 m.
- **Local configuration:** the local and recorded outcome configurations differ only in results and in the route path
  string. The collector hash, the soil and driver settings, and both route hashes are equal.
