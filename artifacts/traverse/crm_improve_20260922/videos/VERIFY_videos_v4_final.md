# Final check of the round-3 wording fixes (2026-09-24)

VERDICT: pass

All nine fixes are on screen or in the files as the fixer's report describes them. Nothing else in the videos
changed: frame counts and fps are the same, and the readouts and vehicle positions I sampled match the saved data.
The README agrees with ffprobe and with the data files, and the folder has no stray files. The three notes at the
end are small and do not block.

Nothing was re-driven or re-rendered to disk. I extracted about 40 frames into /tmp and looked at them, and rendered
some top-down frames in memory only. The scratch folder has been deleted.

## The nine items

| # | fix | where I checked it | result |
|---|---|---|---|
| 1 | Soil pair 0500, old protocol: the footer gives both stuck times | chase_soil2_H3s.mp4 frames 400 and 739 (the end card); the left panel of chase_soil2_side_by_side.mp4 at frames 400 and 739 | The footer reads "Same outcome and end time as the original recorded run (stuck from 5.0 s there, 5.1 s here; drive ended at 34.0 s, no progress). Path within 11 cm of it." The header ("Stuck since 5.1 s") and the end card ("Stuck from 5.1 s", with "Original recorded run: stuck from 5.0 s") now agree with it. The README gives both times twice, under "What to watch for" and under "Can you trust the 3D replays?". |
| 2 | Rigid path distance includes the end points | chase_rigid_H3s.mp4 frames 100, 300 and 575; chase_rigid_side_by_side.mp4 frames 100 and 575 | The footer says "Path within 20 cm of it." The side-by-side subtitle says "on paths within 9-20 cm". In the data, both drives end at 25.8 s with the end points 0.1975 m apart, so the figure is now 0.1975 m, from the last frame (516). The other two rigid replays still read 9 cm and 14 cm, in their own videos (which were not re-rendered) and in the side-by-side. The soil pair 0500 old protocol stays at 11 cm, because its end-point gap (5.5 cm) is below its largest gap along the drive. |
| 3 | README rigid wording | README, "Can you trust the 3D replays?" | "reproduces: same outcomes and end times within 0.05 s, paths within 9-20 cm", with the two sub-points. The data agrees. The three filmed replays end at exactly 25.8, 9.35 and 13.7 s. Of the replays without a video, the 1 s transformer ends at 9.60 s against 9.55 s and the 0.5 s CNN-GRU at 15.35 s against 15.30 s. The 1 s CNN-GRU ends at the same 16.9 s. |
| 4 | "Earlier training" numbers | the two training records: generalist_20260921/A_adapt/train/deploy_v1/H_deploy.json and crm_improve_20260922/deploy_v1/deploy_a1_haux_gru.json | The earlier model was trained on the earlier data set only: 105,193 training rows. The retrained model has 217,814 (2.07 times as many). By file, those are 109,244 from the earlier data plus the part-way replays, 81,634 from the 0.5, 1 and 1.5 s decision points, and 26,936 from the 3 s decision point. The records list 4,051 of the rows as part-way-replay rows. 109,244 - 4,051 = 105,193, so it is the same earlier rows plus 4,051. The numbers add up exactly. The README paragraph and the shorter on-screen footnote are both true. |
| 5 | "planners", not "panels" | the old-protocol footnote in all three 3D old-protocol videos, all three side-by-sides and all three top-down videos | Every one reads "...the other CNN-GRU planners use the retrained model, trained on about twice the data." In manifest.json the label, short label and footnote of the old-protocol planner are identical across the three cases. The script's default footnote matches. |
| 6 | Rigid top-down header | compare_rigid.mp4 frame 200, compare_rigid_2x.mp4 frame 100, and compare_rigid_final.png (which equals the last video frame) | "the retrained CNN-GRU planners in 13.7-16.9 s". |
| 7 | README stall summary | the recorded drive's trajectory.npz (wheel spin and load columns, throttle, pose) | The fixer is right, and the previous checker's "up to 22 rad/s until 12 s" was wrong. The lightly loaded front-left wheel (1.4 kN on average) spins at up to 31.3 rad/s. It is last above 1 rad/s at 8.5 s and stays below 1 rad/s after 8.5 s. The two heavily loaded wheels (front-right 11.8 kN, rear-left 12.1 kN) never exceed 1.0 rad/s after 5 s. The airborne rear-right wheel spins at up to 174 rad/s. Throttle is at least 0.99 from 5.2 s to the end at 34.0 s, which is 28.8 s, so "29 s" holds. From 5.0 s to the end the vehicle moves 5.8 cm. |
| 8 | Screen rule wording | README; screen.json | "Ended the same way as recorded, within 1 s of the recorded times (at most 0.55 s off here)." screen.json uses a 1.0 s tolerance and records 0.55 s as the largest time difference for pair 0500. It is the only one of the five screened pairs where all six replays match. |
| 9 | Screening titles | chase_work/screen/manifest_screen.json, screen.json, and the screening section of chase_reproduction.json | All five screened cases carry the current old-protocol title (the screening manifest also gained the short label and the footnote). The screening section of chase_reproduction.json equals screen.json. The ranks and order are unchanged from the round-3 regression report: 0356, 0244 and 0440 are ranks 1-3 in tier 0, 0093 is rank 13 and 0500 is rank 22, and the screening order is 0356, 0244, 0440, 0093, 0500. The 0500 numbers equal the reproduction entries. |

## Nothing else changed

- **Containers (ffprobe with a full frame count).**
  - All 18 videos are H.264, yuv420p, 20 fps.
  - Frame counts equal the round-3 regression table:
    - top-down: 348/195 (soil), 721/381 (soil2), 557/299 (rigid);
    - single 3D videos: 314/283/337, 740/232/211, 576/247/334;
    - side-by-sides: 337, 740 and 576.
  - The only size change is the height of the side-by-sides: 668 px (soil) and 654 px (soil2, rigid), as reported.
  - Every duration in the README matches ffprobe.
- **3D readouts against the saved overlay rows.** I read the clock, speed and sinkage on sampled frames:
  - old protocol:
    - soil2 at frames 20, 400 and 739: 1.0, 20.0 and 34.0 s (end card), speeds 1.4, 0.0 and 0.0 m/s, sinkage 0.22 m;
    - rigid at frames 100, 300 and 575: 5.0, 15.0 and 25.8 s, speeds 2.7, 0.8 and 1.7 m/s;
    - soil at frames 200 and 313: 10.0 and 12.7 s, 0.0 and -0.1 m/s, sinkage 0.19 and 0.33 m;
  - side-by-sides:
    - rigid at 5.0 s: 2.7, 5.0 and 4.0 m/s;
    - soil at 5.0 s: 0.6, 2.6 and 1.7 m/s, sinkage 0.04, 0.00 and 0.00 m;
    - soil2 at 20.0 s: the two finished panels show their end times, 8.6 and 7.55 s, at 3.1 m/s.

  Every value equals its overlay row, formatted to the digits shown. On nine sampled frames of the three
  re-rendered 3D videos, the camera picture matches its own saved source frame best: a mean difference of 1.4-3.2/255,
  against 1.8-7.6/255 for the frames three steps away.
- **Top-down videos against the recorded poses.**
  - **Vehicle positions:** I rendered the decision frame, the middle frame and the last driven frame of each case in
    memory with the current script and manifest. In every cell of all three cases, the vehicle outline drawn at the
    last frame sits exactly on the recorded terminal pose (error 0.0 m), and the one at the middle frame sits exactly
    on the recorded pose at that step.
  - **Video against the fresh render:** the mean brightness difference is 1.05-1.12/255, with the 99.9th percentile
    at most 9.2/255. That holds for the real-time and 2x videos. As a control, the same video frame against the render
    four steps later differs by up to 95/255.
  - **Top-down readouts:**
    - rigid at 10.0 s: 2.3, 3.4, 3.6 and 4.1 m/s, plus the two finished transformers at 9.55 and 9.35 s;
    - soil at 7.5 s: 0.8, 0.6, 1.8, 2.9, 1.8 and 1.0 m/s;
    - soil2 at 15.0 s: the old protocol at 0.0 m/s.

    All equal the recorded speeds.
  - **Stills:** the three `_final.png` stills equal the last frames of both the real-time and 2x videos (blurred
    brightness difference at most 4/255), apart from the "playback" corner label, which the stills leave out by
    design.
- **Outcomes and times.** The outcomes and times in all three `compare_*_checks.json` files are unchanged from the
  round-3 regression report. The preview stills under chase_work/ equal their video frames (mean difference
  1.7-2.4/255).
- **3D videos not re-rendered.** The six other single 3D videos (13:08) still agree with the current
  chase_reproduction.json. For example, the rigid 0.5 s transformer footer says "9 cm" and the rigid final planner
  footer "14 cm", as in the data.
- **README against the data.**
  - Outcomes and times: all match the recorded drives.
  - Rigid ground: peak speed about 6.9 m/s for the transformers; the old protocol averages 1.04 m/s after 10 s; all six
    drives are 44.8-45.6 m long.
  - Soil pair 0124: the two failures stop at 13.4 and 16.1 m; the successful drives are 34-38 m long.
  - The file list covers every file in the folder.

## Stray files

- **Nothing left in the videos folder:** no `composed/` or `side_*` frame folders, no mp4 files under chase_work/,
  no `_check_` stills, and no backup or temporary files.
- **Files changed since the round-3 reports:** only the ones the fixer lists, plus the preview stills that the
  scripts always write.
- **Older working files:** chase_work/check/ and chase_work/dev/ date from the first render round (11:31-11:46), as
  the round-3 regression report noted. The README describes chase_work/ as working files.

## Notes (not blocking)

1. **README, rigid summary line.** "paths within 9-20 cm" is the range for the three filmed replays. Across all six
   rigid replays, the equal-time gap runs from 7 cm (the 1 s transformer without a video, 6.35 cm) to 20 cm. Every
   drive is still within 20 cm, so the line is not wrong. "paths within 20 cm" would avoid the question.
2. **A code comment in scripts/ci_video_chase.py (lines 98-101), not on screen.** It says "the same rows + 4,614
   branch-drive rows" and then "217,814 fit rows". The 4,614 counts all rows in the file (120,482 - 115,868). The
   training rows actually used are 4,051, which is what the README and the fixer's report give. Both numbers are
   right, but the comment mixes the two counts.
3. **A side effect of this check.** A syntax check I ran refreshed the git-ignored bytecode caches
   `scripts/__pycache__/ci_video_chase.cpython-312.pyc` and `ci_video_compare.cpython-312.pyc` (13:47). These are
   cache files only; no source, data or video file was touched. The later renders in memory ran with bytecode
   writing turned off.
