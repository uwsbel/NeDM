# VERIFY: rigid results (E6b final, RESULTS_rigid.md), 2026-09-26

VERDICT: pass. Every number I recomputed from the raw drive files agrees with RESULTS_rigid.md. The integrity claims
hold: locks, routes, one node per group, shards, and labels. The declared conclusions stand: all four family tests
reject, the rigid gain is roll-backs only, and task B saturates. The prose overstated a few things. I fixed 16
passages in place (section 5). None of the fixes changes a declared decision.

K3 = this folder. My own code: `scripts/ag_vr_rigid_check.py` (sha256 `31f586aa87a84a81...`) and a read-only login-node
extraction `scripts/ag_vr_rigid_cluster_extract.py` (`2b899e5ce33a5180...`, run over ssh with the system python; no job,
nothing written on the cluster). Outputs: `verify_rigid/check.json`, `verify_rigid/cluster_runs.json`. My code does not
import ag_eval_index / ag_analyze / ga_analyze / f104_n2_analyze. Labels, clusters, bootstraps, Holm and McNemar are
written anew. The only shared library call is `TerrainMap.features`, which gives the feature positions. Run time is
1 min locally (plus about 1 min on the login node). No compute was billed: sacct shows only the soil stage-2 training
jobs 439361 / 439362 of another track since 09-26 00:00.

## 1. Drives, labels, index

- Task rows: 42,645 in the 5 task files (unseen 27,668 / f104 6,364 / f104B2 2,399 / held-out 4,143 / dev 2,071).
  There are 42,645 run folders in `e6/runs_rigid`. No row lacks a folder and no folder lacks a row. The mapping files
  carry the sha256 of their task files (all equal).
- Labels were recomputed for all 42,645 drives from `outcome.json` + `trajectory.npz`:
  - not reached = status is not goal_reached (and `goal_reached` agrees);
  - unsafe = not reached, or any 0.05 s interval after the 1 s settle (index >= 20) with body vx < -0.10 m/s and
    throttle > 0.3, or min vx <= -0.30 m/s.
  - No trajectory is shorter than the settle.
  - My labels disagree with the index (`e6/index/rigid_eval_v1.json`, 43,100 arm x group rows) on **0** rows, for both
    not reached and unsafe.
- Wording point, fixed: the rule trips on a single 0.05 s interval. The report said "more than 0.05 s".

## 2. Recomputed rates (HMMWV, fixed 2 m/s unless marked; % of groups)

Pooled over the 8 unseen arenas (2,000 groups). These equal RESULTS T1 to the printed digit:

| arm (code) | plain label | not reached | unsafe | backward-only |
|---|---|---|---|---|
| M1a_fx2 / M1b_fx2 / M1_fx2 | f104 only (ensemble a / b / mean) | 3.00 / 3.15 / 3.08 | 8.25 / 8.40 / 8.33 | 5.25 / 5.25 / 5.25 |
| M2_fx2 | two arenas, same total data | 3.50 | 6.60 | 3.10 |
| M3a_fx2 / M3b_fx2 / M3_fx2 | three arenas, same total data (a / b / mean) | 2.75 / 3.30 / 3.02 | 6.30 / 6.05 / 6.17 | 3.55 / 2.75 / 3.15 |
| A3_fx2 | three arenas, all data | 2.95 | 5.35 | 2.40 |
| straight2 | straight route at 2 m/s | 26.05 | 54.50 | 28.45 |
| speed free M1 / M2 / M3 / A3 / straight6 | as above, speed free | 0.62 / 0.25 / 0.33 / 0.30 / 1.60 | 1.25 / 0.60 / 0.65 / 0.70 / 2.10 | |

Per arena, goal reached / unsafe (%, 250 groups each), fixed 2 m/s. The task asked for 3 arenas; all 8 are shown:

| arena | f104 only a (M1a) | f104 only b (M1b) | three arenas same total a (M3a) | three arenas same total b (M3b) | three arenas all data (A3) | straight 2 m/s |
|---|---|---|---|---|---|---|
| g260 near | 98.8 / 1.6 | 98.4 / 4.4 | 99.2 / 1.2 | 98.4 / 2.0 | 98.4 / 2.4 | 78.8 / 39.2 |
| g271 near | 96.0 / 8.4 | 95.2 / 8.0 | 97.6 / 7.6 | 96.8 / 7.2 | 98.0 / 5.2 | 70.0 / 49.2 |
| g251 near | 98.8 / 6.0 | 98.4 / 7.6 | 96.4 / 7.2 | 98.0 / 5.2 | 97.6 / 5.2 | 71.6 / 68.8 |
| g247 near | 95.6 / 10.0 | 94.8 / 10.4 | 95.6 / 10.0 | 94.4 / 8.8 | 95.6 / 6.8 | 65.2 / 66.8 |
| g258 spread | 96.8 / 13.2 | 97.2 / 12.8 | 96.4 / 9.2 | 96.4 / 8.4 | 95.6 / 7.6 | 82.8 / 47.6 |
| g268 spread | 95.2 / 8.4 | 96.4 / 7.6 | 98.8 / 2.8 | 98.8 / 4.0 | 98.4 / 5.6 | 77.2 / 48.0 |
| g263 spread | 96.8 / 8.8 | 96.4 / 9.6 | 97.2 / 5.6 | 95.6 / 6.0 | 95.6 / 6.0 | 70.0 / 66.4 |
| g241 spread | 98.0 / 9.6 | 98.0 / 6.8 | 96.8 / 6.8 | 95.2 / 6.8 | 97.2 / 4.0 | 76.0 / 50.0 |

The unsafe values equal RESULTS T2 on every cell.

Pooled single ensembles, fixed 2 m/s unsafe. I used my own bootstraps (4,000 draws, other seeds) and exact McNemar:

| comparison | rates | difference, points | 95 % group interval | 90 % cluster interval (81 clusters) | groups better / worse | McNemar two-sided |
|---|---|---|---|---|---|---|
| three arenas same total a (M3a) vs f104 only a (M1a) | 6.30 vs 8.25 | -1.95 | [-3.20, -0.65] | [-3.40, -0.59] | 101 / 62 | 0.0028 |
| three arenas all data (A3) vs f104 only a (M1a) | 5.35 vs 8.25 | -2.90 | [-4.10, -1.65] | [-4.22, -1.67] | 110 / 52 | 6.1e-6 |
| three arenas all data (A3) vs three arenas same total a (M3a) | 5.35 vs 6.30 | -0.95 | [-2.00, +0.05] | [-1.85, -0.10] | 67 / 48 | 0.093 |
| M3b vs M1b; M3a vs M1b; M3b vs M1a; A3 vs M1b | | -2.35; -2.10; -2.20; -3.05 | | | | 0.00055; 0.0019; 0.0011; 8.9e-7 |

These agree with RESULTS 2.1 / T5 ("-1.95 to -2.35, p 0.0005-0.003; A3 -2.90 / -3.05, p 6e-6 / 9e-7").

Other sets, recomputed; every value equals the report:
- f104 in distribution (the declared 200 groups; my set equals `suites/f104_indist_200.json`): M1 / M2 / M3 / A3 /
  straight 2 m/s unsafe 0.50 / 2.00 / 0.50 / 0.50 / 58.0.
- Held-out g203 + g228: 11.33 / 8.00 / 3.50 / 1.67 / 58.0. Per arena g203 6.67 / 2.00 / 2.00 / 1.33 and g228
  16.00 / 14.00 / 5.00 / 2.00.
- Dev g217: 3.00 / 0.67 / 3.33 / 1.33.
- Gaps against f104: +7.82 [+6.5, +9.0], +4.60, +5.68, +4.85.
- Backward-only events of the fixed 2 m/s planner arms: median 5.15 / 5.5 / 6.9 / 5.4 / 5.8 / 5.65 s rolling back and
  peak 1.89 / 2.06 / 1.78 / 2.14 / 2.37 / 2.25 m/s (M1a, M1b, M2, M3a, M3b, A3). The report's ranges had left out the
  two-arena arm (fixed).
- Speed-free time ratio M3a / M1a 1.196 and A3 / M1a 1.065 (declared). The other pairing, M3b / M1b, is 1.109
  (post-hoc; added to the text).

**Task B, rigid (800 f104 groups, Gator):**
- Fixed 2 m/s unsafe: Gator-trained (G) 0.125 %, HMMWV-trained (H = M1a) 0.125 %, straight 2 m/s 15.5 %. Not reached:
  0 / 0 / 11.0 %.
- G vs H: 1 group better and 1 worse (McNemar 1.0), 95 % group interval [-0.375, +0.375].
- G vs straight 2 m/s: 123 / 0 groups.
- Speed free: G / H / straight 6 m/s on the Gator unsafe 0.00 / 0.375 / 0.625 %, not reached 0 / 0.125 / 0.375 %;
  H on the HMMWV 0.125 / 0.
- H's Gator picks equal the M1a HMMWV picks on 800 / 800 groups (speed free). The task B fixed 2 m/s H picks equal the
  task A M1a fixed 2 m/s picks on the 200 in-distribution groups.

## 3. Integrity

- **Pick locks predate the drives.** I recomputed every pick directory's lock (ga_planner scheme: sorted route file
  name + sha256): 172 directories, all equal to the `LOCK_*.json` entry and to the directory's own
  `PICKS_LOCKED.sha256`. In each set every route file is older than its lock, and the lock is older than the first
  drive. Drive start = `collection_request.json` mtime on the cluster, equal to the local `case.json` mtime and the
  shard's recorded start. All times are 09-25:

  | set | lock file | newest route | first drive start | last drive end |
  |---|---|---|---|---|
  | f104 | 13:40:28 | 13:40:04 | 13:41:34 | 14:03:56 |
  | held-out | 14:02:21 | 13:47:29 | 14:02:59 | 14:25:06 |
  | dev | 14:02:59 | 13:57:14 | 14:04:56 | 15:21:01 |
  | unseen | 14:05:29 | 14:05:09 | 14:09:06 | 15:15:09 |
  | f104 fixed 2 (B2) | 14:15:08 | 14:15:05 | 15:06:26 | 15:19:56 |

- **Spec times.**
  - `spec_rigid_v1.json` (e6ce0aaf...) is dated 09-25 13:26:17, before the first rigid evaluation drive (13:41:34).
  - `spec_rigid_v1_B2.json` (202e9908...) is dated 14:10:19, before its picks (14:15) and drives (15:06).
  - The add-on spec (8bc83203...) is dated 09-26 19:23:09. That is 3 min before the index (19:26:08), but my local
    birth times show the outcome files being copied 19:21:01-19:23:40. So "no unseen outcome seen" cannot be checked
    from files; the text now says so.
- **Routes driven = routes locked.**
  - For every one of the 42,645 drives, the sha256 of the cluster's `reference.json` equals `outcome.json`'s
    `route_sha256`.
  - For each of the 43,100 arm x group rows, the driven file has the same waypoints / speeds / stations / headings
    arrays (exact equality) as the arm's own locked pick.
  - The mapping's `route_sha256` equals my content hash of the arm's pick file on all 43,100 rows.
  - 455 rows share a drive with another arm of the same group (identical-route merges). 66 of them have a pick file
    that differs from the driven file only in the `meta` block, typically CEM picking the straight anchor. The
    collector records meta without using it (`gen_collect_ext.py` line 521).
  - No two different drives of a group carry the same route.
- **Fixed 2 m/s cap.** All 17,500 planner fixed 2 m/s routes and 3,450 straight 2 m/s routes have maximum speed
  2.0 m/s.
- **Hosts.** My regex extraction from `simulation_provenance.json` gives a host for all 42,645 runs, on 23 nodes.
  - Each of the 507 shards ran on exactly one node, and it matches the shard's done record.
  - Every done record shows rows = complete (42,645 total), no incomplete rows, no takeover, `moved_before` 0, no
    foreign completions, runner exit 0.
  - No group is split over shards within a task file, so every group is on one node within a file.
  - 712 f104 groups are on two nodes across files (speed-free f104 vs task B fixed 2 m/s). No declared, add-on or
    post-hoc contrast pairs arms across those two files (I checked the three specs).
- **Clusters.** My nearest-feature clusters (start-goal midpoint to the nearest `TerrainMap` feature) equal the
  index's on 2,000 / 2,000 unseen groups. There are 81 clusters.

## 4. Cluster bootstrap and Holm

- My cluster bootstrap resamples the 81 (arena, feature) clusters with replacement and takes the ratio of summed
  differences to summed groups.
  - P3 (three arenas same total vs f104 only, unsafe): -2.15 points, 90 % [-3.41, -0.99], one-sided p 0.00075.
    Tool: [-3.37, -1.01], p 0.0015.
  - P4 (three arenas all data vs f104 only): -2.98, 90 % [-4.13, -1.92], p 0.00025 (floor). Tool: [-4.1, -1.9],
    p 0.00025.
  - The p-values differ only by Monte Carlo noise at 4,000 draws.
- Holm (my step-down code) on the tool's four one-sided p (soil P1 0.00175, P2 0.00125 from `results_soil_v1.json`
  96facb25..., rigid 0.00150, 0.00025) gives 0.00375 / 0.00375 / 0.00375 / 0.00100. This equals
  `family_final_E6b.json` (9cb82ce8...). `family_v1_S2.json` differs from it only in `created`.
  - With my rigid p-values: 0.0025 / 0.0025 / 0.0022 / 0.0010.
  - Rigid-only run (soil at p = 1): 0.0045 / 0.0010. Both match the report.
  - All four reject in every variant. I did not recompute the soil p-values; that is the soil verifier's job.
- Declared secondary decisions checked against the rule (90 % cluster interval inside +-2 = "no meaningful
  difference", else "inconclusive"; one-sided p <= 0.05 = "improves", unadjusted). They are:
  - rigid fixed 2 m/s not reached M3 / A3 vs M1: no meaningful difference;
  - A3 vs M3: improves, inside the band;
  - seed floors: no meaningful difference;
  - f104 no-harm: no meaningful difference;
  - dev M3: no meaningful difference (9 clusters, group 90 % reaches +2.0);
  - dev A3: inconclusive;
  - speed free: improves, inside the band.

  The report states these correctly.

## 5. Wording fixed in RESULTS_rigid.md (in place; the pre-fix copy is `verify_rigid/RESULTS_rigid.before_verify.md`)

1. Label definition: "more than 0.05 s backwards under throttle" became "any recorded 0.05 s interval moving
   backwards faster than 0.10 m/s under throttle above 0.3" (the code's rule).
2. Added an interval convention paragraph. Brackets are 90 % cluster intervals unless marked. Per-arena and gap
   intervals are 95 % group intervals. Cluster counts: 81 unseen, 9 for each f104 set and dev, 19 held-out.
3. Headline 1:
   - "by 2-3 points" became "by about 2-3 points", adding that the M3 effect may be as small as 1 point.
   - "Better on 8 of 8 arenas" is now qualified: point estimates only. The per-arena 95 % intervals exclude zero on
     4 of 8 arenas for each model (PLAN 7.3 replaced the sign count with per-arena intervals).
4. Headline 3 rewritten.
   - "the second arena gives most of that" rested on point estimates. The two -> three step is inconclusive, and the
     two steps were never tested against each other.
   - A3 vs M3 is now labelled secondary, unadjusted, inside the 2-point band.
5. Headline 4: the 28 % / 38 % gap closure is marked as point estimates.
6. Headline 6: the "beats the straight route" criterion rests on the fixed 2 m/s drives. At speed free only 3 of 800
   groups separate G from the straight route.
7. Headline 7: "most of all at constant 2 m/s" was wrong for "not reached". The largest not-reached gap is on the
   planner-proposal routes (11.1 vs 33.7 %, 22.6 points, against 15.6 at constant 2 m/s). Also added the hull below
   the surface on 7.8 % of routes.
8. Section 1:
   - Drive window made exact (09-25 13:41-15:21, instead of "before 23:33").
   - Compute sentence updated: the soil stage-2 jobs of another track now bill.
   - Added the lock/spec timing line.
9. Section 2.1: a note that the bootstrap p-values move with the random draws, and that an independent re-run gives
   the same decisions.
10. Section 2.2: the backward-only descriptors now cover all six planner arms: 5.2-6.9 s, 1.8-2.4 m/s, 0-4 per arm
    through the throttle clause only. The earlier ranges had silently left out M2.
11. Section 2.3: named the arenas whose 95 % intervals exclude zero.
12. Section 2.5:
    - "Every planner beats the straight 6 m/s route" became the declared M1 contrast plus rates.
    - The 1.20 time cost is marked as the declared a / a pairing; b / b gives 1.11 (post-hoc).
13. Section 2.7: dev M3 "no meaningful difference" now carries the 9-cluster caveat (group 90 % interval
    [-1.7, +2.0]).
14. Section 2.8:
    - Added the M1a vs M1b offline difference (+0.0035 [+0.0009, +0.0061]), about half the M3 gain.
    - "Flat" became "rises little".
    - Added the design caveat. The f104 validation groups are all-strata, the test groups hill/crater. The
      like-for-like reference is g203 / g228 validation, 0.948 / 0.946.
15. Section 2.9: "their not reached (about 3 %) sits at the designed-route level" was misleading.
    - Only 6-13 of each arm's 55-70 not-reached drives fall in the 68 groups with no goal-reaching designed 2 m/s route
      (my recount from `test_designed_records.json`).
    - The rest are in groups where a designed 2 m/s route reaches the goal, so the ~3 % is not a feasibility floor.
    - Also verified: the 1,594 / 338 / 68 split and its rates (5.4 / 3.7 / 2.6 % and 20.3 / 15.8 / 15.7 %).
16. Section 3:
    - G vs straight 6 m/s at speed free: 3 and 5 groups, McNemar 0.25 / 0.06, 95 % cluster intervals reaching 0.
    - "Upper 95 % bound 0.00" replaced by +0.25 (group bootstrap). The cluster bootstrap is degenerate here: both
      discordant groups, f104_crm_eval_group_0163 and f104_pair_group_0555, are in cluster f104:3.
    - The fixed 2 m/s straight-route intervals are labelled 90 % cluster, 9 clusters.
    - The headroom criterion at speed free rests on 3 / 6-7 straight-route failures.
    - Collection caveat: the HMMWV drives are from other nodes.
    - The add-on spec timing statement (section 4.1) now matches the file times.

## 6. Not checked, or checked only for consistency

- Soil P1 / P2 were taken from `results_soil_v1.json` as they stand. Their verification belongs to the soil track.
- Task B collection table (T12 / 3.3): checked against `e6/collection/rigid_f104_gator_vs_hmmwv.json` (all numbers
  equal, 1,877 / 24,000 = 7.8 % hull below the surface). It was not recomputed from the 48,000 collection run folders.
- Offline AUC values (2.8, T13): read from `e6/offline/offline_unseen.md` and E5a. They were not re-scored; the E5a
  verifier reproduced the scorer.
- The appendix tables are generated by `ag_e6b_final_tables.py`. I checked T1, T2 and the family table cell by cell
  against my recount, and did not regenerate them. The wording fixes touch only the hand-written sections above the
  appendix. `RESULTS_rigid.json` is a numbers-only digest and was not changed.
