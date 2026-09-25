> **Stale (2026-09-24):** this report checks an earlier render; its fixes 1, 3, 4 and 6 are applied. The current checks are `VERIFY_videos_v2_data.md` (data) and `VERIFY_videos_v2_text.md` (on-screen text); the fixes made after them are listed in `README.md`.

# Verification of the rollout videos (2026-09-24)

Verdict: **pass**. The fixes listed at the end are recommended, and number 1 should be made before the soil 3D side-by-side is shown to anyone.

Scope: the 12 mp4 files and 2 stills in this folder, checked against `manifest.json`, the recorded `trajectory.npz` and
`outcome.json` of all 12 arms, the branch route files, `chase_reproduction.json` and the local re-drives in
`chase_work/`. Nothing was re-simulated for this check.

## Container facts (ffprobe)

| file | resolution | fps | frames | duration | size |
|---|---|---|---|---|---|
| compare_soil.mp4 | 1600x1000 | 20 | 348 | 17.40 s | 1.22 MB |
| compare_soil_2x.mp4 | 1600x1000 | 20 | 195 | 9.75 s | 0.60 MB |
| compare_rigid.mp4 | 1600x1000 | 20 | 557 | 27.85 s | 1.61 MB |
| compare_rigid_2x.mp4 | 1600x1000 | 20 | 299 | 14.95 s | 0.93 MB |
| chase_soil_H3s.mp4 | 1280x720 | 20 | 314 | 15.70 s | 0.92 MB |
| chase_soil_L1_X.mp4 | 1280x720 | 20 | 283 | 14.15 s | 1.05 MB |
| chase_soil_HnG.mp4 | 1280x720 | 20 | 337 | 16.85 s | 1.43 MB |
| chase_rigid_H3s.mp4 | 1280x720 | 20 | 576 | 28.80 s | 19.3 MB |
| chase_rigid_L0p5_X.mp4 | 1280x720 | 20 | 247 | 12.35 s | 7.85 MB |
| chase_rigid_HnG.mp4 | 1280x720 | 20 | 334 | 16.70 s | 11.1 MB |
| chase_soil_side_by_side.mp4 | 1920x624 | 20 | 337 | 16.85 s | 1.12 MB |
| chase_rigid_side_by_side.mp4 | 1920x624 | 20 | 576 | 28.80 s | 3.35 MB |

All files are H.264, yuv420p. The frame counts are exactly what they should be:
- top-down, real time: (longest recorded drive + 1) + a 40-frame hold (soil 307+1+40, rigid 516+1+40);
- top-down, 2x: every other step plus the last one, then the 40-frame hold;
- 3D: local rows + a 60-frame hold (for example, soil L1_X 223+60).

All durations and sizes match both builders' reports.

## Top-down comparisons of the recorded drives (compare_*.mp4, compare_*_final.png)

Frames viewed:
- soil real time: 0, 10, 20, 60, 150, 200, 244, 245, 254, 255, 306, 307 and 347;
- soil 2x: 0, 30, 122, 123, 154 and 194;
- rigid real time: 0, 10, 20, 60, 186, 187, 190, 191, 274, 516 and 556;
- rigid 2x: 0, 94, 258 and 298;
- both stills.

- **Inputs:**
  - status and elapsed time match the manifest for all 12 arms, and rows x 50 ms equals the elapsed time;
  - the branch route content hashes match the hashes stored in the outcomes (12/12, recomputed);
  - the pose at the decision frame equals the stored branch pose;
  - the final goal distance from the terminal pose equals the recorded value.
- **Vehicle position:** I rebuilt the figure's axes transforms and sampled each cell at the pixel of the recorded pose. At every frame listed above, the pixel carries the arm's colour (within 4/255) for all 6 cells, including:
  - the terminal pose in the last frame;
  - the failed arms frozen at their last recorded pose (soil 3 s old protocol at (-3.45, 11.49), soil 1 s CNN-GRU at (2.56, 9.25)).
- **Decision point:** the diamond sits at the recorded pose at 0.5 / 1 / 3 s, checked by pixel in the final frames. The planned route appears at the decision frame (frames 10, 20 and 60 viewed). In the old-protocol cells the approach line is grey.
- **Synchronisation:** every cell shows the same clock (for example, 12.25 s in all six at frame 245). A finished arm shows its frozen outcome and time.
- **Speed strip:** I sampled every third recorded speed of every arm on the final frame. All samples sit on the arm's colour, or under a later-drawn trace (1 unexplained sample out of 1,300, next to an end marker).
- **Outcome labels:** "goal reached in X s", or "bogged down at X s: wheels dug through the soil" with a red outline. These match the outcome files.
- **Header sentences:** both are true.
  - Soil: the two failed drives stopped 2.1-2.2 m above the start on 22-25° ground, which bears out "on the climb". The survivors took 11.45-15.35 s.
  - Rigid: every planner arrives; the transformers in 9.35 and 9.55 s, the other CNN-GRU planners in 13.7-16.9 s, the old protocol in 25.8 s.
- **Labels:** the cell titles are the manifest labels verbatim. The legend uses short plain forms. No arm abbreviations appear on screen. The one exception is the header acronym "CRM" (fix 3).
- **Stills:** they match the last video frame, without the playback label.

## 3D chase re-renders (chase_*.mp4)

Frames viewed:
- start, decision frame, decision + 5, a mid frame, last frame and end card of each of the six single videos;
- soil side-by-side: 0, 10, 20, 60, 150, 222, 223, 253, 254, 277 and 336;
- rigid side-by-side: 0, 10, 60, 186, 187, 274, 516 and 575.

- **Overlay data:** the rows in `overlay.json` equal the local trajectories exactly (pose and speed, 0.0 difference). There is one rendered frame per row, and the time shown is the frame index x 50 ms.
- **Local runs:**
  - they used the same branch routes (content hash equal to the recorded one) and the same decision frame;
  - their outcome configs differ from the recorded ones only in results (branch pose, slip and sinkage statistics);
  - the filmed and camera-free local drives are identical.
- **Decision:** the "Planner takes over at X s" banner and the orange route markers appear at exactly the decision frame. Before it, the phase line reads "Approach".
- **Reproduction flags are honest:**
  - rigid: same outcome and time for all three filmed arms, paths within 0.08-0.18 m;
  - soil: all three filmed arms "differ" (status or time). The 1 s transformer flips from goal reached in 12.85 s to bogged down at 11.15 s;
  - camera-free re-drives of the other arms: 3 of 6 soil outcomes flip, and rigid times agree within 0.05 s.
- **Where the flags show:** every single 3D video carries the recorded outcome in its footer on every frame and on its end card. For example, "Bogged down: wheels dug through the soil (at 11.15 s) / Recorded run: goal reached in 12.85 s".
- **Consequence:** the soil 3D videos do NOT show the recorded drives. The recorded drives are shown only by the top-down videos, which must stay the reference for the architecture comparison.

## Problems and fixes

1. **chase_soil_side_by_side.mp4, from 11.15 s to the end (frames 223-336), transformer panel; likewise from 12.70 s for the old-protocol panel and 13.85 s for the gradient panel.**
   - Problem: the large in-panel outcome card shows only the local outcome, "Bogged down: wheels dug through the soil (at 11.15 s)". The recorded "goal reached in 12.85 s" appears only in the small grey note below the panel. The subtitle says only "re-simulated locally", not that the outcomes differ.
   - Why it matters: a viewer comparing architectures will read the opposite of the recorded result.
   - Fix: in `side()`, add a second line "Recorded run: <recorded_outcome_text>" to the panel end card when `rep["differs"]`, as `compose()` already does. Add to the subtitle: "Local re-runs; their outcomes differ from the recorded drives (see compare_soil.mp4)."
   - Better fix (Builder B's own suggestion): screen other soil pairs with camera-free re-drives and film one whose three arms reproduce.
2. **chase_soil_H3s.mp4, chase_soil_L1_X.mp4 and chase_soil_HnG.mp4 (whole video).**
   - Problem: these are disclosed non-reproductions. The footer text is 15 px grey, and the header gives no hint until the end card.
   - Fix: add "local re-run; recorded outcome: ..." to the header panel's second line when the drive differs. At minimum, keep the top-down video as the one to show first.
3. **compare_soil.mp4, compare_soil_2x.mp4 and compare_soil_final.png, header at every frame: "Deformable soil (CRM)".**
   - Problem: CRM is an unexplained technical acronym on screen.
   - Fix: write "Deformable soil" or "Soft soil (deformable ground)", which also matches the 3D videos.
4. **Negative zero in the speed readouts.**
   - Problem: the readouts show "speed -0.0 m/s". Seen in chase_soil_L1_X.mp4 at 0.00 s, chase_soil_H3s.mp4 at 10.00 s, and the left panel of chase_soil_side_by_side.mp4 at 11.15 s.
   - Fix: set |v| < 0.05 to 0.0 before formatting, as `ci_video_compare.update()` does.
5. **Cosmetic, compare_*: the moving time cursor crosses the decision-band caption.**
   - Where: for example, compare_rigid_2x.mp4 at 9.4 s crosses "(0.5 s".
   - Fix: clip the cursor to the speed axes, or move the caption above the band.
6. **Cosmetic, compare_soil final frames (15.35 s onward): the gradient-refinement cell's vehicle outline covers the "goal" label.**
   - Fix: draw the label below the circle when a track ends above it.
7. **Report wording only; nothing on screen is affected.**
   - Builder A's "went over at about 3.7 m, or 2.9 m for the gradient-refined drive": the 0.5 s CNN-GRU also peaked at 2.8 m, and only the two transformers reached 3.6-3.7 m.
   - Builder B's "matches the particle surface to within 2.5 cm": 2.5 cm is the rms; the 99th percentile is 5.8 cm.
8. **Housekeeping.** `chase_work/` holds 259 MB, mostly raw frames. It can be deleted once the 3D videos are final, but after fixes 1-2 are applied, because re-composing needs the frames.

## Method notes

Scratch files: frames were extracted with ffmpeg and checked with small pixel-sampling scripts in /tmp, which I deleted afterwards. No files were created elsewhere in the repository.
