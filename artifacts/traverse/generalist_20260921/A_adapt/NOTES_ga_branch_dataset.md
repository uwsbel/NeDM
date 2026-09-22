# Module note: branch-continuation rows for the shared risk model (PLAN v2 A4 "Labels", review finding 16)

Written 2026-09-21 23:40. Local machine only, CPU only; no cluster jobs; nothing under `crm_night2_v1`, `crm_f104_v1`
or `fdm_f104_50h_20260909` was touched.

## What was built

`scripts/ga_branch_dataset.py` turns the driven continuations of the A4 branch collection into training rows with
exactly the `mixed_reanchor.npz` keys (plus `cls` and `branch_run`), so `ga_train.py` can train on them alone or on the
merged file. One row per continuation run:

- The prefix (the recorded episode re-driven to the branch frame F) is used only for the history window: `hist` = the
  40 frames before the branch on the 12 observable state columns + the 3 actions, `hmask` = action frame >= 0. The
  window is re-derived from the run's own trajectory and asserted equal to the collector's `branch_hist`.
- Everything after the branch is projected onto the branch route only (the `branch_*` arrays of
  `command_reference.npz`, content-hashed and checked against `outcome.json`; the CRM layout's `branch_reference_*`
  keys and `branch_reached` flag are handled too). The corridor `X` is the 96-station tensor of the branch route from
  the branch pose; `ctx` = [state at F (17), goal - pose_F (2), its norm, yaw at F, branch route length].
- Grace window: the event clock starts at the first post-branch frame >= 1 m from the branch pose, or after 3 s,
  whichever comes first. The night-2 rules run on the frames from there: rollback = first frame with
  (vx < -0.10 and throttle > 0.3) or vx < -0.30; near-stop = first run of 20 frames with |vx| < 0.3 and throttle > 0.3.
- Labels: `fail` = outcome status is not goal_reached (goal reached from the branch, primary); `unsafe` = fail or any
  post-grace rollback / near-stop (secondary); `event_idx` = the branch-route station of the first event (furthest
  station reached for a failed drive without an explicit event), -1 when clean. One deliberate difference from the
  night-2 labeller: a near-stop after the grace window counts as unsafe even when the goal is reached without a
  rollback (night-2 ignored it); this changes 3 of 2,358 rows.
- `E` / `T` = first-arrival cumulative positive work and time after the branch per station (NaN beyond the furthest
  station reached), from `positive_work_kj_per_interval`.
- `privileged` (8) as in the mixed builder (window means of tyre loads, torque; CRM slip and spindle height from
  `crm_extra.npz`, zeros on rigid; is_crm).
- Metadata: `id` = `<run dir>@<world>`, `episode` = anchor episode, `group` / `split` / `cls` from the anchors file,
  `anchor_frame` = F, `vx_anchor`, `rem_m` = `route_len` = branch route length, `source` = 'branch', `profile` = -1,
  `domain` 0/1. Ids, groups and episodes are checked against the planner-suite blacklist; ids asserted unique; the key
  set and dtypes are checked against the mixed file's npz headers (nothing decompressed).
- `--merge <mixed npz> --merged-out <npz>` appends the branch rows to the mixed file (identical key sets asserted,
  `hist_cols` / `priv_names` asserted equal, ids unique after the merge); `cls` and `branch_run` are '' on the mixed
  rows.

## Commands run

```
PYTHONPATH=src:scripts python scripts/ga_branch_dataset.py \
  --runs artifacts/traverse/generalist_20260921/A_adapt/a4/rigid_runs \
  --anchors artifacts/traverse/generalist_20260921/A_adapt/a4/anchors/anchors_rigid.json --world rigid \
  --map-root artifacts/traverse/crm_f104_v1/map_root \
  --out artifacts/traverse/generalist_20260921/A_adapt/datasets/branch_rigid.npz            # 2 s with 8 workers
PYTHONPATH=src:scripts python scripts/ga_branch_dataset.py \
  --out artifacts/traverse/generalist_20260921/A_adapt/datasets/branch_rigid.npz \
  --merge artifacts/traverse/generalist_20260921/A_adapt/datasets/mixed_reanchor.npz \
  --merged-out artifacts/traverse/generalist_20260921/A_adapt/datasets/mixed_reanchor_plus_branch_rigid.npz   # 77 s
```

Files: `A_adapt/datasets/branch_rigid.npz` (42 MB, 2,358 rows), `branch_rigid_report.json`, `branch_rigid_build.log`,
`mixed_reanchor_plus_branch_rigid.npz` (2.14 GB, 118,226 rows = 115,868 mixed + 2,358 branch), `..._merge.json`.

## Rigid results (sync state at 23:31: 800 anchor dirs, 800 replays, 2,367 continuation dirs)

Input: 800 rigid anchors; 11 anchors skipped by the collector (no valid continuation after 192 draws: 4 with 0/3,
3 with 1/3, 4 with 2/3 valid); 8 anchors still without `branch_auto.json` (their job was running when the sync
happened): their finished continuations are included, 9 unfinished ones (all `__c2` but one) are dropped as
"incomplete". Result: 2,358 rows from 789 anchors, 789 groups. Every row passed the checks: branch route start = the
pose at F (max offset 0.000 m), replayed pose and vx at F equal to the recording (max 0.000), route hash checked on
all rows, full 40-step history on all rows (every rigid anchor has F >= 40).

Counts (rows / fail / unsafe / anchors):

| class | train | val | test |
|---|---|---|---|
| clean_moving | 1,260 / 145 / 309 / 420 | 90 / 11 / 23 / 30 | 90 / 6 / 21 / 30 |
| low_progress | 805 / 417 / 661 / 270 | 57 / 36 / 51 / 20 | 56 / 36 / 53 / 19 |

Rates: clean_moving fail 11.3 %, unsafe 24.5 %; low_progress fail 53.3 %, unsafe 83.3 %. Statuses: goal_reached
1,707, prolonged_blockage_terminated 570, timeout 70, terrain_bounds_exit 7, rollover 4. Unsafe-but-not-failed rows:
467 of 2,358 (19.8 %).

Label discordance among the 3 continuations of one anchor (anchors with all 3 present):

| class | anchors | fail labels differ | unsafe labels differ | all 3 fail | all 3 reach the goal |
|---|---|---|---|---|---|
| clean_moving | 480 | 115 (24.0 %) | 208 (43.3 %) | 8 | 357 |
| low_progress | 301 | 95 (31.6 %) | 62 (20.6 %) | 109 | 97 |

So the continuation matters: for a quarter of the clean anchors and a third of the low-progress anchors the
route/speed choice decides between reaching the goal and not; only 109 of 301 low-progress anchors fail on all three
continuations, so the low-progress rows are not "all unsafe by construction" any more (finding 16).

Failure by the continuation's base speed (fail / unsafe): clean_moving 2 m/s 0.192 / 0.394, 4 m/s 0.102 / 0.237,
6 m/s 0.044 / 0.104; low_progress 2 m/s 0.557 / 0.854, 4 m/s 0.503 / 0.831, 6 m/s 0.538 / 0.814. Among the
discordant anchors the failing continuation is the FASTER one in only 9 of 115 (clean) and 30 of 95 (low-progress)
cases: on rigid ground a slow continuation from a moving vehicle fails more often than a fast one.

Grace window: clean_moving rows leave the 1 m circle after 0.2 / 0.35 / 0.55 s (p10/50/90; all 1,440 rows end the
grace by distance). low_progress rows: 317 end it by distance, 601 hit the 3 s cap (p10/50/90 = 1.35 / 3.0 / 3.0 s).
First event: clean_moving rollback 308, near-stop 45, none 1,087 (rollbacks at 2.7 / 5.8 / 12.4 s after the branch,
event station p10/50/90 = 12 / 33 / 66); low_progress rollback 505, near-stop 260, none 153, with 650 of the 918
rows' first event within 1 s after the grace end and 69 % of the unsafe low-progress rows at station <= 2. In words:
the grace window moves the clock, but a vehicle that was bogging at the branch is usually still bogging 3 s later,
so for the low-progress class the event station carries little route information; the fail label (53 %) and the
per-anchor discordance are the informative parts.

Furthest station reached (fraction of the branch route) p10/50/90 = 0.02 / 0.90 / 0.95; the failed rows barely move.

Loader check: `ga_train.Data` (CPU, `--cond hist_rma`) constructs on `branch_rigid.npz` alone (train / val rows found,
history and privileged arrays picked up).

## Open issues

1. 8 anchors (9 continuations) were unfinished at sync time; re-run both commands above when the sync is complete
   (the build takes seconds; the merge 77 s and 2.1 GB of disk).
2. The merged file duplicates the 2.1 GB mixed file. A `--data` list in the trainer would avoid that; not done here
   (no existing repo file was edited).
3. `profile` is -1 on branch rows (the continuation's base speed 2/4/6 m/s is in the run's `branch_route.json` meta,
   not in the npz). If the trainer ever conditions on `profile`, add a key rather than overloading it.
4. The near-stop-counts-as-unsafe rule differs from the mixed rows' rule for 3 rows; harmless, but the merged file's
   `unsafe` is not one rule. The report lists the count (`ns_only_goal_reached_rows`).
5. Clean-moving continuations at 2 m/s fail four times as often as at 6 m/s. Whether this is the speed floor /
   follower braking at the swap or genuine low-speed blockage on rigid ground was not investigated; it affects how
   the history effect is read (speed is a confound within an anchor).
6. The CRM path (`<episode>__c<j>` dirs, `branch_reference_*` route keys, `branch_reached`, `crm_extra.npz`) is
   written but untested on real data; smoke it on the first CRM pass-2 run dir before building.
7. The 11 collector skips are all sampler rejections (no valid continuation within 192 draws); those anchors are lost
   for this world unless the sampler tolerance is relaxed.
