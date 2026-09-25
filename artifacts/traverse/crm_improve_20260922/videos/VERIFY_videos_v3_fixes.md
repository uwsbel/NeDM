# Check of the wording fixes (round 3, 2026-09-24)

VERDICT: pass

All four claimed fixes are on screen. Each one is true and in plain words, and README.md is accurate file by file.
The nine problems below are wording and precision fixes. None of them is a wrong outcome, time or model on
screen. Problems 1 and 2 are the only ones a viewer can see.

## What was checked

- **Nothing was re-driven or re-rendered.** 59 frames were extracted with ffmpeg into /tmp and viewed, then
  deleted. They cover every video's start, decision banner, stuck-line onset, last driven frame, end card and last
  frame.
- **On-screen strings.** 2,426 strings of the 3D videos were rebuilt with `ci_video_chase.py` (labels, footnote,
  header note, the phase line at every frame, footers, end cards, side-by-side subtitles). None contains
  "cluster", "local", "CRM", an arm code, "-0.0" or a time such as "12.70 s".
- **Durations and sizes (ffprobe).** Every value matches the README and the fixer's table. The side-by-side
  videos are now 1920x676 (soil) and 1920x670 (soil2, rigid). The three stills equal the last video frames
  (blurred difference at most 5/255).
- **Reproduction numbers.** `max_distance_to_recorded_path_m`, the equal-time differences and the terminal
  differences were recomputed with my own code for all nine entries of `chase_reproduction.json`. All match.
- **Stuck line.** It was recomputed from `overlay.json`. It first shows at 7.75 s (onset 5.75 s) in
  chase_soil_H3s, at 9.55 s (onset 7.55 s) in chase_soil_L1_X and at 7.1 s (onset 5.1 s) in chase_soil2_H3s.
  In all three the vehicle stays below 0.3 m/s for the rest of the drive at mean throttle 0.99-1.0 and moves
  0.30 / 0.19 / 0.01 m. "Full throttle, no progress" is true, and no other video shows the line.
- **Earliest stop.** The soil2 "stopped at 34.0 s at the earliest" equals `earliest_possible_stop_s` in the run's
  `collection_request.json`.

## The "earlier training" claim, checked independently

The ensemble of every pick was read from the pick folders' `summary.json`. Each pick was then traced to its drive:
the arm's route hash in the pick file equals the route that the manifest names and the outcome records.

| panel | pick folder | ensemble (5 members each) |
|---|---|---|
| 3 s, soil, soil2 and rigid | `generalist_20260921/A_adapt/a5/picks_{crm,rigid}_H_named` | `generalist_20260921/A_adapt/train/deploy_v1/H_deploy_s*.pt` |
| 1 s and 0.5 s CNN-GRU | `crm_improve_20260922/s2/idx/picks_{crm,rigid}_L{1,0p5}_Hn_named` | `crm_improve_20260922/deploy_v1/deploy_a1_haux_gru_s*.pt` |
| gradient, soil | `s4/idx/picks_crm_L0p5_HnG_named`, identical G routes to `s4/L0p5/picks_crm_HnG` and `s4/idx/picks_crm_L0p5_HnG` | `deploy_a1_haux_gru_s*.pt` |
| gradient, rigid | `s4/idx/picks_rigid_L0p5_HnG_named`, identical G route to `s4/L0p5/picks_rigid_HnG` | `deploy_a1_haux_gru_s*.pt` |
| transformers | `s2/idx/picks_*_L*_X_named` | `deploy_a3_haux_txjoint_s*.pt` |

- **Training data:**
  - `H_deploy.json` was trained on `mixed_reanchor.npz` only, with 105,193 fit rows.
  - `deploy_a1_haux_gru.json` was trained on `mixed_reanchor_plus_branch_both.npz`, `short_anchor.npz` and
    `anchor_k60.npz`, with 217,814 fit rows.
- **What the added files hold:**
  - `short_anchor.npz` holds decision frames 10/20/30 (0.5/1/1.5 s, 89,998 rows).
  - `anchor_k60.npz` holds only frame 60 (the 3 s decision point, 29,699 rows).
- **Model type.** Both models are CNN-GRUs: `ga_train.py` is the CNN-GRU trainer and `arch=gru` in the retrained
  model.

The on-screen claim is true. The fixer's report is wrong on one point: it calls `anchor_k60.npz` early-decision
data, but it is 3 s data (see problem 4).

## Problems

1. **chase_soil2_H3s.mp4: the footer disagrees with the header and end card.**
   - Where: the footer on every frame (0.0-37.0 s) and the same note in the left panel of
     chase_soil2_side_by_side.mp4 (all frames).
   - Problem: the footer reads "Same outcome and end time as the original recorded run: stuck from 5.0 s; drive
     ended at 34.0 s …". From 7.1 s the header says "Stuck since 5.1 s", and the 34.0 s end card says
     "Stuck from 5.1 s". A viewer reads "same outcome … 5.0 s" against "5.1 s".
   - Fix: in `repro()`, when `recorded_stuck_from_s` differs from `local_stuck_from_s`, write "Same outcome and end
     time as the original recorded run (stuck from 5.0 s there, 5.1 s here; drive ended at 34.0 s, no progress).
     Path within 11 cm of it." Then rerun repro, compose and side.
2. **Rigid old protocol: "within 19 cm" understates the gap.**
   - Where:
     - the chase_rigid_H3s.mp4 footer, every frame (0.0-28.8 s);
     - the chase_rigid_side_by_side.mp4 subtitle "within 9-19 cm" and left panel note, every frame;
     - README "Can you trust the 3D replays?".
   - Problem: both drives end at 25.8 s, and the end positions are 0.1975 m apart. The JSON's own
     `max_distance_to_recorded_path_m` is 0.197. The 19 cm comes from `max_position_diff_m` (0.1815), which leaves
     out the terminal pose, so it is not the upper bound the code comment claims.
   - Fix: in `compare_runs()`, include the terminal poses in `max_position_diff_m` when the elapsed times are
     equal. That gives "within 20 cm" and "9-20 cm"; update the README to match.
3. **README.md, "Rigid pair 0011: exact."**
   - Problem: the replays are not exact.
     - The paths differ by up to 20 cm (see problem 2).
     - Two of the unfilmed rigid arms' camera-free re-drives end 0.05 s later: the 1 s transformer at 9.60 s
       against 9.55 s, and the 0.5 s CNN-GRU at 15.35 s against 15.30 s.
   - Fix: "Rigid pair 0011: reproduces. All three filmed replays end the same way at the same times, on paths
     within 9-20 cm of the recordings."
4. **README.md, the "Earlier training" paragraph, and the fixer's report: incomplete.**
   - Problem: the retrained model did not add early-decision rows only. It also added 29,699 rows at the 3 s
     decision point (`anchor_k60.npz`) and the branch drives (`…_plus_branch_both.npz`), about twice the earlier
     model's rows. The on-screen footnote stays true.
   - Fix (README): "'Earlier training' means the previous study's CNN-GRU, trained on the earlier data set only.
     The retrained model added rows at early decision points (0.5, 1 and 1.5 s), rows at the 3 s decision point,
     and branch drives."
5. **"the other CNN-GRU panels" where there are no other panels.**
   - Where: the header footnote of chase_soil_H3s.mp4, chase_soil2_H3s.mp4 and chase_rigid_H3s.mp4 (every
     frame). In the soil and rigid side-by-sides there is only one other CNN-GRU panel.
   - Fix: in the three H3s `footnote` fields of manifest.json, write "…; the other CNN-GRU planners use the
     retrained model." Then re-render.
6. **compare_rigid.mp4, compare_rigid_2x.mp4 and compare_rigid_final.png: two words for one model.**
   - Where: the header on every frame.
   - Problem: the header says "the newer CNN-GRU planners", while the footnote directly below says "the retrained
     model".
   - Fix: `headline()` in `ci_video_compare.py` should say "the retrained CNN-GRU planners".
7. **README.md stall summary: the wheels do not all stand still for 29 s.**
   - Problem: it says "the three loaded wheels stand still at full throttle for 29 s". This copies the short answer
     in EXPLAIN_soil2_old_protocol_stall.md, but that file's own timeline and the recorded `trajectory.npz` differ:
     - the front-left wheel spins at up to 31 rad/s during 5.0-7.25 s and up to 22 rad/s until 12 s;
     - it carries only about 1.4 kN;
     - all three wheels are below 1 rad/s in 88 % of the 5-34 s frames.
   - Fix: "…which spins while the other three wheels (two of them carrying nearly all the weight) stand still; the
     vehicle stays at full throttle, without moving, for 29 s: …".
8. **README.md, "the only one whose six replays all matched their recordings": "matched" is undefined.**
   - Problem: the screen's rule is "same outcome, end time within 1 s". In 0500 the local end times differ from the
     recorded ones by 0-0.55 s.
   - Fix: "…the only one whose six replays all ended the same way as recorded, within 1 s of the recorded times…".
9. **chase_reproduction.json `soil_group_screen.groups[*].arms[*].label` (not on screen).**
   - Problem: it still carries the old 3 s title. The fixer reported this.
   - Fix (optional): give `chase_work/screen/manifest_screen.json` the new label for H3s, or add a line in the
     README saying that the screening section keeps the old titles.

## Not verifiable here

- Byte identity of the rest of manifest.json: only the soil2 entry could be compared, against
  `chase_work/screen/manifest_screen.json`. It differs only in the H3s label and the two new fields.
- That only text fields changed in `candidates.json` and `screen.json`: no earlier copies exist.
- The rigid still matching "a fresh render exactly".

Everything I did check agrees with the fixer's report, apart from its description of `anchor_k60.npz`
(problem 4).
