# Rollout videos: on-screen honesty and legibility check (v2, 2026-09-24)

Verdict: **pass**. Every required check holds on the current render. The fixes below would make the wording more precise. None of them corrects a false statement on screen. Fixes 1 and 2 matter most.

This report supersedes `VERIFY_videos.md` for on-screen text. That file describes the previous render: its fixes 1-6 are now applied, and it does not cover the soil2 videos.

## Container facts (ffprobe, frames counted by decoding)

All files: H.264, 20 fps.

| file | size (px) | frames | duration | MB |
|---|---|---|---|---|
| compare_soil.mp4 | 1600x1000 | 348 | 17.40 s | 1.22 |
| compare_soil_2x.mp4 | 1600x1000 | 195 | 9.75 s | 0.61 |
| compare_soil2.mp4 | 1600x1000 | 721 | 36.05 s | 1.48 |
| compare_soil2_2x.mp4 | 1600x1000 | 381 | 19.05 s | 0.86 |
| compare_rigid.mp4 | 1600x1000 | 557 | 27.85 s | 1.60 |
| compare_rigid_2x.mp4 | 1600x1000 | 299 | 14.95 s | 0.94 |
| chase_soil_H3s.mp4 | 1280x720 | 314 | 15.70 s | 0.89 |
| chase_soil_L1_X.mp4 | 1280x720 | 283 | 14.15 s | 1.04 |
| chase_soil_HnG.mp4 | 1280x720 | 337 | 16.85 s | 1.43 |
| chase_soil_side_by_side.mp4 | 1920x646 | 337 | 16.85 s | 1.13 |
| chase_soil2_H3s.mp4 | 1280x720 | 740 | 37.00 s | 2.06 |
| chase_soil2_L1_X.mp4 | 1280x720 | 232 | 11.60 s | 0.99 |
| chase_soil2_HnG.mp4 | 1280x720 | 211 | 10.55 s | 0.90 |
| chase_soil2_side_by_side.mp4 | 1920x646 | 740 | 37.00 s | 1.39 |
| chase_rigid_H3s.mp4 | 1280x720 | 576 | 28.80 s | 19.26 |
| chase_rigid_L0p5_X.mp4 | 1280x720 | 247 | 12.35 s | 7.85 |
| chase_rigid_HnG.mp4 | 1280x720 | 334 | 16.70 s | 11.05 |
| chase_rigid_side_by_side.mp4 | 1920x624 | 576 | 28.80 s | 3.35 |

The frame counts match the builders:
- Top-down, real time: the longest recorded drive + 1, plus a 40-frame hold (soil2: 680 + 1 + 40 = 721).
- Top-down, 2x: every other step, plus a 40-frame hold.
- 3D: the local rows, plus a 60-frame end card (soil2 old protocol: 680 + 60; soil2 1 s transformer: 172 + 60; soil2 gradient: 151 + 60).

## Frames viewed

- **Top-down:**
  - compare_soil: 0, 10, 20, 60, 245, 255, 307, 347.
  - compare_soil2: 0, 10, 20, 60, 100, 149, 161, 199, 400, 680, 720.
  - compare_rigid: 0, 10, 20, 60, 187, 274, 516, 556.
  - 2x videos: 0, the decision, the failure and the last frame.
  - The three `_final.png` stills.
- **3D single videos:** for each of the nine, the start, the decision frame, decision + 5, a mid frame, the last driven frame, the first end-card frame and the last frame.
- **3D side-by-side:** soil 13 frames, soil2 13 frames, rigid 10 frames, including every panel's end.
- **Automated checks:**
  - The marker bar of the "differs" note was sampled on every frame of all 12 3D videos.
  - All 2,734 3D speed, sinkage and time strings were rebuilt from `overlay.json`. All 4,946 top-down speed strings were rebuilt from the recorded `trajectory.npz`.
  - Every on-screen label, outcome, note and header sentence was rebuilt from the manifest, `chase_reproduction.json` and `compare_*_checks.json`, then scanned for arm codes, "CRM" and "-0.0".

## Required checks

| check | result |
|---|---|
| Plain language: no arm codes (H3s, L1_X, HnG, L0p5 ...), no "CRM" on screen | **pass**. None in any rebuilt string or viewed frame. Model names ("CNN-GRU", "Transformer") are what is being compared and are kept. Wording nits: fixes 4 and 5. |
| Soil title "Soft soil (deformable ground)" | **pass**. Seen in all soil and soil2 top-down videos, stills, single 3D headers and side-by-side titles. |
| No "-0.0 m/s" | **pass**. There are 0 cases in either set of rebuilt strings. There are genuine "-0.1"/"-0.2 m/s" readings: 57 rows in the 3D videos and 55 steps in the top-down videos, all while stuck or bogged down. See note 8. |
| Differing 3D re-drives say so in the header on every frame | **pass**. See the details below the table. |
| ... and on the end card | **pass**. Each differing card carries "Recorded run: ...". The first and last end-card frames are identical (mean pixel difference 0.04-0.8). |
| Soil side-by-side subtitle points to the top-down video | **pass**. The red line reads "...the recorded outcomes are shown in the top-down comparison video." It does not name the file (fix 3). |
| soil2 old-protocol wording and the 34 s end | **pass**. See the details below the table. |
| Overlaps and cut-offs | **pass**. See the details below the table. |
| soil2 header sentence true and fair | **pass**. See the details below the table. |

**Differs note in the header.** The note marker was checked on every frame:
- Red bar: chase_soil_L1_X (283/283 frames). This is the only outcome flip.
- Grey bar: chase_soil_H3s 314/314, chase_soil_HnG 337/337, chase_soil2_L1_X 232/232 and chase_soil2_HnG 211/211 (time differences only).
- No note bar: chase_soil2_H3s and the three rigid videos. Their re-drives match the recording (same outcome, same time, paths within 9-19 cm), and the footer says so.
- Side-by-side subtitle line: soil red 337/337 frames, soil2 grey 740/740, none on rigid.

**soil2 old protocol, top-down card "stuck from 5.0 s; drive ended at 34.0 s (no progress)".** Status is `prolonged_blockage_terminated`. The drive ended because the vehicle made no progress, not because it bogged down:
- Speed stayed below 0.3 m/s from 5.0 s. The vehicle moved 8 cm afterwards, at full throttle (mean 0.9996).
- The largest wheel sinkage was 0.175 m (local 0.217 m), under the 0.30 m bogged-down limit.
- The no-progress rule can fire no earlier than 24 s + 2 s confirmation + 8 s tail = 34.0 s. It fired exactly then.

**Overlaps and cut-offs.**
- The time cursor passes behind the caption's opaque box, with no crossing of the letters (checked at 3.0, 5.0, 9.35 and 9.4 s).
- The "goal" label is drawn below the circle in the soil gradient cell, and no final pose covers a goal label.
- The 3D footer ends at x <= 1008 px. The minimap panel starts at 1020. The widest footer (soil2 old protocol) ends at about 1001.
- The end card ends at y <= 374. The minimap panel starts at 382.
- The side-by-side info text ends at x0 + 462. The minimaps start at x0 + 480.
- The title-bar legend (x >= 1360) is clear of the title (ends at about 1110) and of the subtitle row.

**soil2 header sentence.** It reads: "The old 3 s approach gets stuck 2.0 s after its decision and stays stuck; every planner that decides after 1 s or 0.5 s gets through (7.45-9.95 s)."
- It is true: stuck at 5.0 s, decision at 3.0 s, and all five others reach the goal in 7.45-9.95 s.
- It makes no claim about the models. The 1 s CNN-GRU reaching the goal in 9.55 s is shown in its cell.
- The soil2 3D set films only the 1 s transformer, but no 3D text claims a difference between the models.

## Findings and fixes

1. **The old-protocol soil 3D re-run bogs down 4.1 m earlier than the recording, and the header hides it.**
   - Where: chase_soil_H3s.mp4, header note at every frame (0.00-15.65 s) and the end card from 12.70 s (frame 254). The same drive is in the left panel of chase_soil_side_by_side.mp4 from 12.70 s (frame 254).
   - Problem: the note reads "this local re-run differs (same outcome, 0.45 s later)". The local drive bogged down 9.0 m from the start (9.3 m driven). The recorded drive bogged down 13.1 m from the start (13.4 m driven). Both lie on the same line (within 0.07 m of the recorded path), but the local vehicle drove slower and dug in 4.1 m further back. A viewer comparing it with compare_soil.mp4 sees it bog down in a different place, while the header only admits a 0.45 s difference.
   - Fix: in `recorded_note()`, and in the side-by-side card, add the end-point gap when the outcome is the same and `terminal_position_diff_m` > 1 m. For example: "(same outcome, 0.45 s later and 4.1 m further back along the same path)".
2. **The soil2 old-protocol 3D video says "following its chosen route" through 29 s of standing still, and its end card leaves out when the vehicle got stuck.**
   - Where: chase_soil2_H3s.mp4 from 5.10 s to 34.00 s (frames 102-679), and the left panel of chase_soil2_side_by_side.mp4 over the same span. End cards from 34.00 s (frame 680).
   - Problem: the phase line keeps reading "Planner in control since 3 s, following its chosen route" while the vehicle stands at 0.0/-0.1 m/s at full throttle.
   - Problem: the end card reads "Stuck: no progress, drive ended at 34.00 s". That is accurate, but it drops the start of the stall, and its wording and precision differ from the top-down card ("stuck from 5.0 s; drive ended at 34.0 s (no progress)").
   - Fix: once the local speed has been below 0.3 m/s for 2 s, switch the phase line to "Stuck since 5.1 s: full throttle, no progress" (red). The local stall began at 5.10 s; the recorded one at 5.0 s.
   - Fix: make the end card "Stuck from 5.1 s; drive ended at 34.00 s (no progress)".
   - Optional: add the footer line "A drive that makes no progress is stopped at 34 s at the earliest."
3. **The one opposite-outcome re-run is flagged only as "differs".**
   - Where: chase_soil_L1_X.mp4, header at every frame, 0.00-14.15 s.
   - Problem: the header says "Recorded cluster run: goal reached in 12.85 s; this local re-run differs", but not how. The viewer learns that the re-run bogs down only from the end card at 11.15 s.
   - Fix: "...; this re-run bogs down instead (at 11.15 s)".
   - Same issue in the soil side-by-side subtitle (every frame), which says "Some local re-runs end differently". In fact one ends the opposite way, and the other two end 0.45 s later and 1.50 s earlier.
   - Fix for the subtitle: "The 1 s transformer bogs down in this re-run but reached the goal in the recording; the other two end 0.45 s later / 1.50 s earlier. The recorded drives are in compare_soil.mp4." Likewise, name compare_soil2.mp4 in the soil2 subtitle.
4. **The sinkage definition in the 3D footer is not plain language.**
   - Where: all six soil and soil2 single 3D videos, footer on every frame.
   - Problem: the footer reads "Sinkage = deepest wheel hub below tyre radius above the undisturbed surface, the measure the bogged-down rule uses." This is hard to parse.
   - Fix: "Sinkage = how far the lowest wheel has sunk below the untouched soil surface. A drive counts as bogged down when a wheel stays more than 0.30 m down (through the 0.24 m soil layer) for 0.25 s."
5. **"cluster" and "locally" are insider words.**
   - Where: the five differing single 3D headers ("Recorded cluster run: ...") and the soil and soil2 side-by-side subtitles ("recorded cluster drives"). Also "Re-simulated locally" in every 3D footer and panel note.
   - Problem: an outside viewer does not know that two computers are involved, which is the reason the re-runs differ.
   - Fix: for example, "Original recorded run: ...; this 3D replay was re-simulated on a different computer and differs (...)".
6. **Minor wording and precision.**
   - The top-down videos print "12.7 s" and "34.0 s"; the 3D videos print "12.70 s" and "34.00 s" for the same events.
   - Optional fix: use one format everywhere, for example two decimals in both.
7. **File names carry arm codes (not on screen).**
   - Where: chase_soil_L1_X.mp4, chase_rigid_L0p5_X.mp4 and the chase_*_H3s / _HnG files.
   - Problem: anyone who is sent the files sees the codes.
   - Optional fix: plain names such as chase_soil_transformer_1s.mp4, or a short README that maps names to set-ups.
8. **Note, no change needed.**
   - "-0.1 m/s" appears in the soil old-protocol and 1 s CNN-GRU cells and in the soil2 old-protocol cell and 3D readouts. Chase soil 1 s transformer also shows "-0.2 m/s".
   - These are real small backward body speeds (|vx| 0.05-0.19 m/s) while the wheels spin in place. They are not rounding artefacts.

## Method

- Frames were extracted with ffmpeg to /tmp and viewed at full resolution, or as crops and contact sheets.
- Strings were rebuilt with `scripts/ci_video_chase.py` (`speed_txt`, `recorded_note`) and `scripts/ci_video_compare.py` (`SHORT`, the `update()` speed rule), both imported without being modified.
- The termination rules were read from `scripts/gen_collect.py` (`StopPolicy`; 24 s / 2 s / 8 s defaults in `crm_collect.py`) and `scripts/crm_collect_ext.py` (sinkage above 0.30 m for 5 frames).
- The only file written in the repository is this report. The scratch files in /tmp were deleted.
