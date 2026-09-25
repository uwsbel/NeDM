# Regression check after the wording pass (round 3, 2026-09-24)

VERDICT: pass

The wording pass broke nothing I could find. All 18 videos open and have the expected frame counts at 20 fps, with
no black, flat or corrupt frames. The top-down videos still put every vehicle at its recorded final pose. Every
clock, speed and sinkage readout in the 3D videos equals the saved overlay row, on every frame. `manifest.json`
changed only in the label fields. The fixer left no scratch folders or temporary videos.

Nothing was re-driven. Frames were decoded by piping ffmpeg output straight into Python, so no frame files were
written. The only file I created is this report (see note 3).

## 1. Containers

I ran ffprobe on every file, counted frames by decoding, and ran a full `ffmpeg -f null` decode of every file.

- All 18 files are H.264, yuv420p, 20/1 fps, one video stream. The full decode reported 0 errors in every file.
- Frame counts and durations match the v2 table exactly. The only change is the height of the three side-by-side
  videos: soil 1920x676 (was 646), soil2 1920x670 (was 646), rigid 1920x670 (was 624).
- The expected counts were rebuilt from the data, not from the scripts' own logs:

| video | built from | expected | decoded |
|---|---|---|---|
| compare_soil / _2x | longest recorded drive 307 steps: 307+1+40 / 154+40 | 348 / 195 | 348 / 195 |
| compare_soil2 / _2x | 680 steps: 721 / 341+40 | 721 / 381 | 721 / 381 |
| compare_rigid / _2x | 516 steps: 557 / 259+40 | 557 / 299 | 557 / 299 |
| chase_soil_H3s / L1_X / HnG | 254 / 223 / 277 overlay rows + 60 | 314 / 283 / 337 | same |
| chase_soil2_H3s / L1_X / HnG | 680 / 172 / 151 rows + 60 | 740 / 232 / 211 | same |
| chase_rigid_H3s / L0p5_X / HnG | 516 / 187 / 274 rows + 60 | 576 / 247 / 334 | same |
| chase_*_side_by_side | longest of the three + 60 | 337 / 740 / 576 | same |

For every 3D arm, the overlay row count, the number of saved frame images and the local `trajectory.npz` length
agree. Each also equals the local elapsed time divided by 50 ms. The recorded trajectories agree with their outcome
times as well, and `compare_*_checks.json` states the same counts. The durations in README.md match ffprobe.

## 2. Black or corrupt frames

I checked every frame of all 18 videos for mean and spread of brightness, and for jumps between neighbouring frames.

- **Brightness:** the lowest frame mean is 156/255 and the lowest spread is 42. No frame is dark or flat.
- **Jumps:** none stand out. The largest frame-to-frame change in each video falls on the decision banner or on
  the first end-card frame.
- **3D camera picture:** I compared a region the overlays never cover against the saved source frame. I did this
  every 7th frame of each single video, every 9th frame of each side-by-side panel, and on each video's first,
  last-driven, first-card and last frames.
  - The video frame always matches its own source frame best. The largest mean difference in any one video is
    1.5-4.1/255.
  - The exceptions are 17 samples in the soil2 old-protocol side-by-side panel, all from its standing-still
    stretch (speed ≤ 0.02 m/s). There, the frames three steps away are identical to within 0.02/255, so the
    comparison cannot tell them apart.

## 3. Top-down videos: last frame against the recorded final poses

I used the axes mapping of `scripts/ci_video_compare.py`: its `build()` and `vehicle_poly()` functions, and each
cell's data-to-pixel transform. With these, I placed the vehicle footprint at the recorded `terminal_pose` from each
`trajectory.npz`. I then compared it with the arm-coloured pixels in:
- the last frame of each real-time video;
- the last frame of each 2x video;
- each `_final.png` still.

The measures were how much of the eroded footprint carries the arm colour, the centroid of the eroded colour blob,
and its long-axis direction.

| case | footprint coverage | the same footprint shifted 1 m | centroid error | heading error |
|---|---|---|---|---|
| soil (6 cells x 3 images) | 0.92-1.00 | 0.36-0.69 | ≤ 0.075 m | ≤ 1.5° |
| rigid | 1.00 | 0.41-0.80 | ≤ 0.096 m | ≤ 2.7° |
| soil2, the five successful cells | 1.00 | 0.53-0.64 | ≤ 0.041 m | ≤ 1.5° |
| soil2, old-protocol cell | 0.87-0.91 | 0.39-0.67 | 0.19-0.21 m | ≤ 2.2° |

- **Soil old-protocol cell:** its lower coverage (0.92) comes only from edge pixels under the thicker red outline
  that marks a failed drive.
- **Soil2 old-protocol cell:** the 0.2 m offset is the decision diamond drawn over the stuck vehicle's rear. This
  was already noted in v2 (data finding 4); I checked the missing pixels, and they lie on the diamond's dark rim.

I also rendered frames 0, 10, 20, 60, 61, K/3, K/2, K-1 and K, and the last hold frame, in memory with the current
script and the current manifest. I compared them with the real-time video, and six frames with each 2x video:
- the mean brightness difference is 1.05-1.12/255 and the 99.9th percentile ≤ 10/255;
- a control, video frame K/2 against the render four steps later, gives up to 95/255.

All three `_final.png` stills are pixel-identical to a fresh render (maximum difference 0). The finish texts,
header sentences, legend labels and footnote in `compare_*_checks.json` are unchanged from v2, apart from the new
footnote and the H3s labels.

## 4. 3D overlays against the saved overlay rows

I read the text on every frame of all 12 3D videos (8,233 panel frames) by template matching. For each
readout, every possible value was drawn in the video's font at the video's pixel position, and the value that fits
the frame best was taken as the reading:
- the clock: 801 values, 0-40 s;
- speed: -3.0 to 15.9 m/s;
- sinkage: 0.00-0.60 m.

The expected value came from `overlay.json`: the row's time, or the local end time on end-card frames, its speed
`vx` and its sinkage. It was formatted by my own rule, not by the script's.

| check | single videos (9) | side-by-side (3 x 3 panels) |
|---|---|---|
| clock wrong | 0 of 3,274 frames | 0 of 4,959 panel frames |
| speed wrong | 0 | 0 |
| sinkage wrong (soil) | 0 | 0 |
| global clock in the side-by-side subtitle wrong | - | 0 of 1,653 frames |
| best fit beats the next value by at least | 1.74x | 1.24x (global clock 1.06x) |

`overlay.json` still equals the local `run/trajectory.npz` exactly: time = frame × 50 ms, and `vx` = `state[:,0]`.

I also checked the new status line by the colour of its marker on every frame of the nine single videos. It shows
red "stuck" on exactly the frames where the overlay has been below 0.3 m/s for 2 s. Otherwise it shows grey
(approach) or orange (planner in control), with 0 mismatches:
- soil2 old protocol: from frame 142 (stall began at 5.10 s); the vehicle then moved 0.011 m at full throttle;
- soil 3 s: from frame 155 (5.75 s); it moved 0.301 m;
- soil 1 s transformer: from frame 191 (7.55 s); it moved 0.192 m.

The note marker under the header is still red only for the soil 1 s transformer and grey for the four other
differing replays; the four matching replays have none.

## 5. `manifest.json`

The videos folder is not tracked by git, so I used three other comparisons:
- **soil2:** I diffed the entry against the pre-fix copy in `chase_work/screen/manifest_screen.json` (key g0500),
  which v2 found identical to the old soil2 entry. Only the H3s arm differs:
  - `label` changed from "CNN-GRU, decides after a 3 s approach (old protocol)" to "CNN-GRU (earlier training),
    decides after a 3 s approach (old protocol)";
  - `short_label` and `footnote` were added.

  Every other value and the key order are unchanged.
- **soil:** I rebuilt the entry with the script's own lookup through the run indexes (`soil_case_entry`). It is
  field-for-field identical to the manifest, in the same key order.
- **rigid:** there is no pre-fix copy, so I checked every field against the recorded runs, all passing:
  - case id and status;
  - elapsed time and decision frame;
  - decision frame × 50 ms = approach time;
  - branch-route content hash, approach-route file hash (the outcome's `route_sha256`), and `have_traj`;
  - case and arena paths.
- **All three cases:**
  - the labels of the five other arms equal the pre-fix labels;
  - the H3s `label`, `short_label` and `footnote` are identical across the cases;
  - no other arm gained a key.

The statuses and times equal the v2 quotes: soil 12.25 / 12.75 / 12.85 / 11.45 / 12.7 / 15.35 s; rigid 25.8 / 16.9 /
9.55 / 15.3 / 9.35 / 13.7 s; soil2 34.0 / 9.55 / 8.05 / 9.45 / 9.95 / 7.45 s.

`chase_reproduction.json` also still holds. I recomputed all 18 entries with my own code: status, elapsed time,
frame counts, largest equal-time difference, end-point gap, distance to the recorded path line and stall onset
(recorded 5.00 s, local 5.10 s). All match, and so do the `differs` flags. The screening ranks and order are
unchanged: ranks 1-3 in tier 0, 0093 = 13, 0500 = 22, revised order 0093 then 0500.

## 6. Stray files

- **Nothing left behind:**
  - no `composed/` or `side_<case>/` frame folders;
  - no mp4 below `chase_work`;
  - no `compare_*_check_*.png` stills;
  - no temporary, partial or backup files.
- **Files changed since the v2 reports:** only the expected outputs (the 18 videos, 3 stills, 3 checks files,
  `manifest.json`, `chase_reproduction.json`, `README.md`, `VERIFY_videos.md`, and `screen.json` /
  `candidates.json`), plus the preview stills that `compose` and `side` always write:
  - `chase_work/<case>_<arm>/last_composed.jpg` and `decision_composed.jpg`;
  - `chase_work/side_<case>_last.jpg` and `side_<case>_t2s.jpg`.

  Each preview still equals its video frame (mean difference 1.6-3.5/255).

## Notes (no action needed)

1. `chase_work/check/` (5 stills) and `chase_work/dev/` (2 stills) date from 11:31-11:46, before the v2 reports.
   They are left over from the first render round, not from this pass. README.md describes `chase_work/` as
   working files.
2. `VERIFY_videos_v3_fixes.md` appeared at 13:26, while this check was running. It is another checker's report,
   not a leftover.
3. Importing `scripts/ci_video_compare.py` to reuse its axes mapping refreshed the git-ignored bytecode cache
   `scripts/__pycache__/ci_video_compare.cpython-312.pyc` (13:20). It is a cache file only; no source or data file
   was touched.
