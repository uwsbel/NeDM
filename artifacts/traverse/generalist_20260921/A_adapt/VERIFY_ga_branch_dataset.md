# Verification of scripts/ga_branch_dataset.py (A4 branch labels, review finding 16)

Verifier run 2026-09-21, local machine, CPU only. Nothing under `crm_night2_v1`, `crm_f104_v1` or
`fdm_f104_50h_20260909` was touched; no repo file edited; no cluster job submitted. Scratch files in
`/tmp/gbd_verify/` (re-run output + three check scripts); the check code re-implements the projection, the event
rules and the grace window independently of the module and reads the route from `branch_route.json` rather than
from the npz the module uses.

## Verdict: PASS

The rows measure the continuation, not the prefix: the history window is the prefix, everything after the branch
is projected on the branch route, the labels come from the continuation's own outcome, and the held-out groups stay
held out. Details and the caveats that remain are below.

## Re-run

Same command as in the module note, output to `/tmp/gbd_verify/branch_rigid_rerun.npz`: 2,358 rows, 9 dropped
(incomplete), 11 anchors skipped by the collector, 8 anchors pending. Every array in the re-run is identical to the
shipped `A_adapt/datasets/branch_rigid.npz` (all keys, exact equality). The sync is unchanged since the module was
built (3,967 dirs; the 8 pending anchors still lack their `__c2` run, one also its `__c1`, which is exactly the 9
"incomplete" drops). The rerun is therefore not a new state; the build must be repeated once those 9 runs land.

## Hand checks on 5 continuation runs

Runs: `f104_v2_group_0002_route_00__a4__c1` (clean-moving, train, goal reached, no event),
`f104_v2_group_0063_route_01__a4__c0` (clean-moving, test, goal reached, rollback after the grace),
`f104_v2_group_0012_route_04__a4__c0` (low-progress, train, blockage termination, near-stop event),
`f104_v2_group_0038_route_08__a4__c0` (low-progress, val, goal reached, rollback at the grace end),
`f104_v2_group_0110_op_03__a4__c0` (low-progress, train, rollback inside the grace only, goal reached, row clean).

1. History window. Convention (stated by the code, the collector and the mixed builder alike): `hist[t] =
   [state[F-39+t][12 observable cols], action[F-40+t]]`, so the LAST row is `[state[F], action[F-1]]`: the state at
   the top of the branch frame (the result of the prefix's actions, before any branch action) and the last prefix
   action. This is the mixed file's convention with k = F. Verified on all 5 runs (first and last row against the
   raw trajectory; last row is not `state[F-1]`), and on all 2,358 rows against the collector's `branch_hist`
   (equal to float16 precision; the npz stores history as float16 like the mixed file). The swap happens at the top
   of frame F before the follower runs, so `action[F]` is the first branch action; on all 5 runs it is the swap
   transient (steer carried over by the clamp, throttle 0, brake 0). All rows have a full 40-frame window.
2. Corridor start. First branch waypoint at 0.0000 m from `pose[F]` on all 5 (module-wide maximum 0.0 m). `X`
   equals `station_tensor` of `branch_route.json` on all 5; route length equals `route_len` and `rem_m`; the branch
   route ends at the case goal (0.000 m).
3. Grace window. Recomputed independently: ends at the first post-branch frame >= 1 m from the branch pose or at
   60 frames. Events before it are ignored: run 5 has a rollback at 0.00 s after the swap (vx at F = -0.50 m/s),
   the grace ends at 2.25 s by distance, no event afterwards, the row is clean with `event_idx` -1. Runs 3 and 4
   show the other side of the rule: a stall or rollback that starts inside the grace and is still going at 3.0 s is
   stamped at the grace end (see caveat 1).
4. Event clock and station on the branch route. `time_to_event_s` = (event frame - F) x 0.05 on all 5 (my values
   match to 1e-6). `event_idx` matches my projection on the branch route (29, 0, 5); projecting the same frames on
   the ORIGINAL route would give 38, 28, 58, so the stations are unmistakably the branch route's.
5. Labels vs `outcome.json`. `fail` == (status != goal_reached) on all 5 and on all 2,358 rows; `unsafe` = fail or
   post-grace rollback/near-stop on all 5; `event_idx == -1` exactly when `unsafe == 0` on all rows; `ctx`, `E`,
   `T` equal my recomputation on all 5 (E/T first-arrival after the branch, NaN beyond the furthest station).

## Is the fail status decided by the continuation or inherited from the prefix?

The blockage stop rule needs a bounded 2 s window with throttle > 0.3 in all 40 frames, then 2 s confirmation and
an 8 s tail (12 s). The earliest `prolonged_blockage_terminated` among the 570 such rows is 12.05 s after the
branch (p10 14.2 s), so no termination clock carried over from the prefix: the recorded `action[F]` has throttle 0
(swap transient), which breaks any throttle window spanning the swap. The 70 `timeout` rows had 76.7 / 108.5 /
118.0 s (p10/50/90) left after the branch and none had less time than route length / base speed. The fail label
is the continuation's own.

## Leaks

Mixed file and branch file each have one split per group; every branch group exists in the mixed file with the
same split (0 disagreements); branch train groups vs mixed held-out groups: 0 overlaps, and the reverse: 0; row
split and group equal the anchors file; group is the episode's prefix; 0 id overlaps with the mixed file. 23 anchor
episodes are not in the mixed file (their groups are), harmless. Rows: train 2,065 / val 147 / test 146.

## Schema and merge

Branch npz keys = mixed keys + `cls`, `branch_run`; dtypes and trailing shapes identical for every shared key
(read from the npz headers). Merged file `mixed_reanchor_plus_branch_rigid.npz`: 118,226 = 115,868 + 2,358 rows on
every array, dtypes as the mixed file, ids unique, order = mixed rows then branch rows (id arrays equal), `cls` ''
on mixed rows, `ctx` and `hist` of the branch part equal the branch file, spot-checked mixed rows equal the mixed
file, `hist_cols` / `priv_names` equal, the small label arrays concatenated in order, no train/held-out group
overlap in the merged file.

## Caveats (none is a defect of the script; all affect how the rows should be read)

1. The grace window rarely changes a label. Over all rows, 711 have a rollback/near-stop starting inside the grace;
   in 531 of them the same event is still going at the grace end and is stamped there (event at 3.0 s, station 0-2);
   only 21 rows have a different `unsafe` label with vs without the grace. For low-progress rows `time_to_event_s`
   and `event_idx` therefore mostly encode "still stuck 3 s later", as the module note already says. The `fail`
   label and the per-anchor discordance are the informative parts for that class.
2. `unsafe` counts a post-grace near-stop even when the goal is reached (3 rows); the mixed rows do not. The
   merged file's `unsafe` is two rules. Known and reported by the module.
3. The recorded `action[F]` (throttle 0, brake 0) is the swap transient of the first substep, not what the follower
   drove for the rest of that 50 ms interval; it is inside the post-branch segment, not the history window, so the
   rows are unaffected, but anyone using `action[F]` as "the first branch command" should know.
4. The 8 pending anchors (9 runs) still need the rebuild + merge once synced; the rerun today reproduced the same
   state bit for bit.
5. CRM path untested on real data (no CRM pass-2 runs exist locally); the rigid path is what was verified here.
