# Module note: branch anchors for the moving-prefix collection (PLAN A4, stage "select")

Written 2026-09-21 18:10. Local CPU only (6 worker processes, 32 s wall for the full scan + selection); no cluster jobs,
no Chrono runs, no GPU. Nothing under `crm_f104_v1`, `fdm_f104_50h_20260909` or `crm_night2_v1` was modified.

## What was built

`scripts/ga_branch_anchors.py --stage select --out A_adapt/a4/anchors` chooses the 800 anchors per world from which the
A4 continuations will later be driven (an anchor = a recorded episode cut at a frame F). The continuation-sampling
stage is deliberately not in this script yet (it waits for the pending gc_control fix); `--stage continuations` exits
with a message saying so.

Command run:

```
PYTHONPATH=src:scripts python scripts/ga_branch_anchors.py --stage select \
    --out artifacts/traverse/generalist_20260921/A_adapt/a4/anchors --workers 6      # seed 20260921 (default)
```

Outputs in `A_adapt/a4/anchors/`:

| file | content |
|---|---|
| `anchors_crm.json`, `anchors_rigid.json` | 800 records each: episode, world, group, split, class, F (frame and seconds), onset frame and F-onset, recorded pose (x, y, yaw) and vx/throttle/brake/steer at F, lateral deviation, station, route length and remaining distance at F, goal (xy, radius, straight-line distance), route file path + sha256 (+ the sha256 the rigid outcome recorded), the recorded run dir and its `command_reference.npz`, recorded outcome status, sinkage numbers (CRM), pairing flags, index into the history file |
| `hist_crm.npz`, `hist_rigid.npz` | `hist (800,40,15) f32`: causal window ending at F, `hist[t] = [state[F-39+t][cols 0-6, 11-15], action[F-40+t]]` (same layout as `mixed_reanchor.npz`); `hmask (800,40)` True where the action frame `F-40+t >= 0` (partially masked when F < 40; every chosen anchor has F >= 40, so all windows are fully valid); `anchor_id`, `episode`, `F`, `hist_cols` |
| `summary.json` | the counts table, F distributions, survivor counts, onset-to-termination margins, pairing pools, selection bookkeeping, the rules used |
| `scan_crm.json`, `scan_rigid.json` | per-episode scan of all 15,235 CRM and 24,000 rigid recordings (reused by re-runs; `--rescan` rebuilds them) |

## How the anchors were chosen

Inputs: the raw recordings (CRM `crm_f104_v1/collect_v1/runs`, rigid `production_v3/runs` + `production_v4/runs`) for
every episode key in the B-cache manifest; group, split and status come from the manifest (the twin group split:
1,089 train / 56 val / 55 test groups). Every CRM episode has a rigid twin (15,235 of 24,000 rigid episodes), and the
two worlds drove byte-equal references (checked per anchor: the route json's waypoints and speeds equal the recorded
`command_reference.npz` for 800/800 anchors in both worlds; the rigid outcome's `route_sha256` equals the file's sha).
Route files: designed routes `fdm_f104_50h_20260909/cases_night2/cases/routes/<group>/route_KK.json`, planner proposals
`fdm_f104_50h_20260909/cases_night2_onpolicy/routes/<group>/op_KK.json`.

Rules (all applied to the recording of the world in question):

- clean-moving (60 %): F in {40, 80, 120} frames (2/4/6 s); at F the vehicle moves (vx > 1 m/s), is < 1 m from the
  route, is not parked, has >= 12 m of route left; and the prefix is clean, i.e. no stall onset and no rollback
  (vx < -0.1 under throttle, or vx < -0.3) before F. The "clean prefix" condition is my addition to the task text.
- low-progress (40 %): the episode has a stall onset = first run of 20 consecutive frames with |vx| < 0.3 and
  throttle > 0.3, searched from frame 20 (the night-2 labeller's rule, `f104_n2_dataset.first_run`). F in
  [onset-20, onset+10], F = onset-10 when possible, otherwise the feasible F nearest to onset-10 (earlier on ties).
  CRM anchors are pre-screened: the mean spindle height above the BMP ground (`crm_extra.npz: spindle_z_m -
  bmp_ground_z_m`, averaged over the 4 wheels) must drop by < 0.1 m between F-20 and F. Rigid keeps the same F rule
  without the screen. Not parked and >= 5 m of route left (my addition, so a continuation to the goal exists).
- Quotas: train 700 (420 clean-moving + 280 low-progress), held-out 100 split 50 val + 50 test (30 + 20 each), every
  anchor kept with its group's split. At most one anchor per episode and world, at most 2 per group and world;
  groups without an anchor are served first (round-robin over shuffled groups), so the 800 anchors sit in 800
  distinct groups. Deterministic seed 20260921.
- Pairing: candidates feasible for the class in BOTH worlds are used first (clean-moving: with a common F), so the same
  (episode, F) appears in both worlds where possible. Low-progress pairs share the episode; they share the frame only
  when the two worlds' windows overlap (`--lp-same-frame prefer`, default: the common F closest to both onset-10
  values; this can put a CRM cut up to 0.5 s after the onset, still inside the plan's window). Per-world fill from
  single-world candidates was never needed. The class agrees for all 800 episodes.
- The F distribution of clean-moving anchors is balanced on the fly (lowest running count wins, random tie-break).

## Result

Counts per world x split x class (identical episodes in both worlds; "same F" = the same frame in both worlds):

| world | split | class | n | quota | same F | groups |
|---|---|---|---|---|---|---|
| crm / rigid | train | low_progress | 280 | 280 | 56 | 280 |
| crm / rigid | train | clean_moving | 420 | 420 | 420 | 420 |
| crm / rigid | val | low_progress | 20 | 20 | 5 | 20 |
| crm / rigid | val | clean_moving | 30 | 30 | 30 | 30 |
| crm / rigid | test | low_progress | 20 | 20 | 4 | 20 |
| crm / rigid | test | clean_moving | 30 | 30 | 30 | 30 |

- Clean-moving F: 160 / 160 / 160 at 40 / 80 / 120 frames in both worlds. vx at F p10/50/90: CRM 1.6/2.5/4.0 m/s,
  rigid 1.7/3.0/5.9 m/s; remaining route p50 36 m (CRM) / 34 m (rigid).
- Low-progress F relative to the onset: CRM 255/320 at onset-10, rigid 265/320; the rest sit elsewhere in
  [onset-20, onset+10] because of the shared-frame preference (65 pairs) or the CRM screen. With
  `--lp-same-frame never` the same episodes are chosen and 302 (CRM) / 319 (rigid) cuts land at onset-10.
- Low-progress state at F (0.5 s before the stall run starts): vx p10/50/90 CRM 0.18/0.61/1.21 m/s, rigid
  -0.45/0.36/0.79 m/s; 70 (CRM) / 94 (rigid) of 320 already have |vx| < 0.3 at F. Recorded outcomes of the chosen
  low-progress episodes: CRM 259 soil breakthrough + 61 prolonged blockage; rigid 188 prolonged blockage, 115 goal
  reached (transient stalls the PID recovered from), 16 timeout, 1 bounds exit.
- CRM sinkage screen (survivor counts, before selection): train 9,339 episodes with a stall onset, 7,858 (84 %) pass
  the screen at onset-10, 9,131 (98 %) at some F in the window, in 1,073 of 1,089 groups; val 518 -> 437 / 505 (55
  groups); test 518 -> 426 / 507 (55 groups). Sinkage drop at onset-10 over all onsets p50/p90 = 0.031/0.128 m; over
  the chosen CRM low-progress anchors p50/p90 = 0.016/0.080 m (all < 0.1 by construction); spindle height above BMP at
  F p10/50/90 = 0.43/0.58/0.80 m.
- Onset-to-termination margin (onset to the last recorded frame): CRM all onsets p10/50/90 = 4.2/8.1/23.0 s (train),
  screen survivors 4.2/8.1/23.1 s, the chosen CRM low-progress anchors 5.0/8.8/19.4 s; rigid all onsets 11.6/25.2/67.8 s,
  chosen 11.5/25.4/62.5 s. CRM onset time p50 10.4 s (the plan review found 10.1 s on 669 breakthrough episodes).
- Paired pools (episodes in both worlds, train): 12,164 with a common clean-moving F; 3,895 with a stall onset in both
  worlds (5,444 stall only in CRM, 16 only in rigid), 3,796 feasible low-progress in both, only 742 with an
  overlapping F window: the onsets differ between worlds by p10/50/90 = 0.65/5.6/22 s, so a shared frame is possible
  for ~21 % of low-progress pairs.
- Cost hint for the later replay passes: total prefix length to replay is 6,933 simulated seconds in CRM (low-progress
  t_F p10/50/90 = 5.9/12.6/29.6 s, max 88 s; 3 CRM anchors have F > 1,200 frames) and 9,717 s rigid.

## Checks

- Asserted: no planner-suite id or group (`f104_crm_eval_group_*`, `f104_g1_test_group_*`, `f104_pair_group_*`) in the
  manifest or among the anchors; every group is an `f104_v2_group_*`; splits equal the manifest; <= 1 anchor per
  episode, <= 2 per group; every clean-moving anchor satisfies vx/deviation/remaining/not-parked; every CRM low-progress
  anchor passes the screen; `hmask` counts = min(F, 40); the last history step equals the state at F.
- Independent re-computation (separate code, 354 CRM + 60 rigid anchors incl. all CRM low-progress): recorded pose, vx,
  throttle at F; the history window rebuilt frame by frame; route sha256 and content; goal; the clean-prefix rule; the
  sinkage drop. All agree. The onset equals the first frame of the B-cache `stalled` mask on 317/320 CRM and 23/23
  rigid checked anchors; the 3 exceptions have their onset past frame 1,200, where the cache is cut.
- Smoke on 400 episodes per world with small quotas and the `--lp-same-frame never` variant (into /tmp, removed).

## Open issues / decisions for the next stage

1. Held-out allocation 50 val / 50 test and the 5 m minimum remaining route for low-progress anchors are my choices.
2. Rigid recordings came from arbitrary cluster nodes; the plan's pass-1 replay decides per anchor whether the class
   and pose recur (drop count reported there). The replay's "stalled vs moving" check should use the stall-onset rule
   on the replayed prefix, not the instantaneous state at F: by design F = onset-10 is 0.5 s before the stall run, and
   only 70 (CRM) / 94 (rigid) of the 320 low-progress anchors have |vx| < 0.3 at F.
3. Shared-frame low-progress pairs (65) put the CRM cut up to 10 frames after the onset; if the continuation stage
   prefers every cut at onset-10, re-run with `--lp-same-frame never` (same episodes, same seed).
4. Anchors with very long prefixes (3 CRM with F > 1,200 frames, up to 88 s) are the most expensive CRM replays; a
   `--max-F` cap is not implemented and would change the selection.
5. The two `scan_*.json` caches (22 MB) can be deleted once the anchors are frozen.
