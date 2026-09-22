# Verification of fix round 1: ga_suite / ga_analyze / gb_track_analyze

Verifier run 2026-09-21 (after `VERIFY_ga_suite.md`; checks the fix reported in `NOTES_ga_suite.md` section 5 against
findings 2.1, 2.2 and 2.5). Nothing in the repo was modified; all scratch output went to `/tmp/verify_fix1/`. No cluster
submission, no simulator, no GPU. Protected directories (`crm_night2_v1`, `crm_f104_v1`, `fdm_f104_50h_20260909`) show
only the two pre-existing `nav_v1` modifications.

Verdict: **pass**. Every number in the fix report reproduces bit-for-bit, the suite files are byte-unchanged, the lock
verifies, and the polyline projection behaves correctly on hand-made cases. Three small remarks in section 3, none of
which blocks use of the tools.

## 1. Suite files unchanged and lock verified (finding 2.5)

| file | sha256 now | matches fix report |
|---|---|---|
| `suite/suite.json` | `64921952321fa4e20597bea8b0b4c1a61dad0ac2a763c6eb9351edfa30844c1a` | yes |
| `suite/tasks_crm.json` | `287737e388778187c4fe307f8c9b40e67c1862a6756576f8131f0308c18c8a80` | yes (before = after) |
| `suite/tasks_rigid.json` | `cc39cad584a6110744f4565bbe82249992a1372875099bffcb1215b79c68f26e` | yes |

File times untouched (suite.json 14:31, tasks 17:27; lock written 17:52). `tail -n +2 SUITE_LOCKED.sha256 | sha256sum -c`
gives OK for all three. Line 1 of the lock (`4327c110...910763`) was recomputed independently in plain Python
(sha256 over name bytes + content digest, names sorted) and is identical, so the scheme in `write_suite_lock` is what the
docstring says and matches `PICKS_LOCKED.sha256`. Code read: `--stage lock` only calls `write_suite_lock`; the merge stage
calls it last; nothing else is written.

## 2. Cross-track metric on the trajectory polyline (finding 2.1)

Hand-made route (L-shaped polyline (0,0)-(10,0)-(10,5)-(10,15), unequal sample spacing), `station_xtrack` with cap 6 m:

| station | expected | got |
|---|---|---|
| (4, 0) on the interior of segment 0 | 0 | 0.00000 |
| (10, 2.5) and (10, 12) on interior of segments 1, 2 | 0 | 0.00000 |
| (10, 0) exactly a vertex | 0 | 0.00000 |
| (4, 3), (7, 1) beside a segment (nearest sample would be 5.0 / 3.16) | 3.0 / 1.0 | 3.00000 / 1.00000 |
| (10, 20) and (13, 19) past the end | 5.0 (distance to the end point) | 5.00000 / 5.00000 |
| (-3, 4) before the start | 5.0 (distance to the first point) | 5.00000 |
| (20, 8) far off | 10.0, capped 6.0 | 10.00000, capped 6.000 |

Also: single-sample trajectory gives the point distance; duplicate samples (zero-length segments) give 0, 0, 1 as expected;
sub-sampling a straight path every 1/2/3/10 samples leaves the distance unchanged (0.7 / 0.2 m). Brute-force segment loop
on 50 random station/trajectory sets: max |vectorised - loop| = 3.6e-15 m, polyline <= nearest sample on every station, the
returned sample index equals the nearest-sample argmin everywhere. Heading is looked up at the nearest sample, as reported.

141-route feasible stratum re-derived from the tracking suite's recorded native-PID run directories with the new code
(Winsorised route mean averaged over routes, nearest sample -> polyline): rigid 0.2179 -> 0.1997 m (8.4 %; constant_2 3.3,
constant_4 6.7, constant_6 12.0, smooth 6.2 %), CRM 0.4516 -> 0.4418 m (2.2 %; 0.8 / 1.8 / 3.0 / 1.7 %); capped stations
0 -> 0, reached 12,435 -> 12,435 in both worlds. Identical to the fix report and to the original verification.

Self-tests re-run into `/tmp/verify_fix1/track/` and compared as whole JSON files with the stored `_fix1` outputs:
- identical copies, CRM and rigid: result files identical to `results_identical_fix1.json`; `xtrack_ratio.point`, p95, CI
  and group-clustered CI are exactly 1.0 (float equality), `degenerate = true`; native PID feasible 0.316 m (CRM) /
  0.147 m (rigid), infeasible 3.858 / 1.078, all 1.733 / 0.519.
- truncation (`--truncate 0.3 --truncate-arm policy`, `suite_one.json`, crm fixture): identical to
  `results_truncated_fix1.json`; frames 515 -> 154, reached 102 -> 39, capped 63, median and p95 6.0, Winsorised
  0.4256 -> 4.0546 m, ratio 9.5272; differs from the pre-fix stored file only in the cross-track fields, as expected.
- action bounds: equal -> "action bounds checked on 1 routes", file identical to `bounds/results_fix1.json`;
  mismatch -> `AssertionError` on the throttle bound, exit code 1.

## 3. Pick agreement by route geometry (finding 2.2)

Re-ran the exact command of note section 5.2 on the real A0 picks for both worlds (`/tmp/verify_fix1/ga_analyze/`). The
`agreement` block is identical to the stored `fix1/agreement_A0_<world>.json` (the only summary field that differs is
`arm_specs`, because I passed the same paths from a different working string; the numbers are the same):
- CRM-world picks (H stand-in = S_crm arm A, own = S_crm B, other = S_rigid B): n 800, all pairs with geometry 800/800;
  d(H, own) mean 1.939 m (median 1.303), d(H, other) 2.486 m (median 1.805); diff +0.547 m, one-sided p05 +0.433 m,
  CI95 [+0.411, +0.685] -> PASS; closer to own 485 / other 311 / ties 4 (60.6 %, exact sign p 7.5e-10); sha256-identical
  own 29 / other 0. Pair medians 1.303 / 1.917 / 1.805 / 1.914 / 1.466 / 1.452 m with identical counts 29 / 88 / 0 / 3 / 1 / 24.
- rigid-world picks: d(H, own) 1.994 m (median 1.299) vs 2.505 m (median 1.874); diff +0.512 m, p05 +0.390 m,
  CI95 [+0.367, +0.654] -> PASS; 492 / 304 / 4 (61.5 %, sign p 2.8e-11); identical own 27 / other 1.
The statistic is non-degenerate and points the expected way; my earlier medians (1.30 m own-model A vs B, 1.47 m S_crm-B
vs S_rigid-B) are reproduced. Code read: `route_dist` is the symmetrised mean nearest-waypoint distance on xy; the
bootstrap resamples groups with the shared rng after every other block, so the rest of the summary cannot move.

Night-2 reproduction re-run with the new code (`--cluster-ci`, 4 s): of the 19 summary blocks, only `world` (not passed)
differs from the stored `fix1/n2_repro.json`, and only `world` and `agreement` differ from the pre-fix `n2_repro.json`
and from my original verification run; PRIMARY B:A diff -7.5, CI [-12.0, -3.5], p 0.00073, one-sided p95 -4.0,
identical picks 5 unchanged. Cluster-CI file identical to the stored fix1 copy and to my original run (it differs from
night-2's own file only because that file holds contrasts E:A / F:A instead of E:B / F:B; the B:A, C:A, D:A blocks agree).
Geometric demo B:A:C: d(B, A) 1.727 m vs d(B, C) 1.435 m, diff -0.292, p05 -0.528 -> FAIL, closer to A 69/200; as the
fix report says, this is expected for two CEM siblings and is not a decision.

## 4. Remarks (none blocking)

1. `write_suite_lock` silently drops a file from the lock if it is missing (`present = [n for n in names if exists]`).
   The line-1 label lists which files were included, so a short lock is visible, but a warning or an assert on all three
   would be stricter. Current lock covers all three.
2. `route_metrics` now defines "reached" by the polyline distance (`reached = dmin < cap_m`) while the heading error
   is taken at the nearest sample. A station within 6 m of the path but more than 6 m from every sample now counts as
   reached; at 0.05 s sampling this only matters at the cap edge, and on the 141 feasible routes the reached count did
   not change. Worth one sentence in the docstring; no effect on the self-tests.
3. Finding 2.3 (rigid reference drives must run in the same array task as the later H/T/P arms) is recorded in the note
   as a launch rule and is not a code change; it was not exercised here and remains a condition on the A3 launch. The
   old self-test outputs (`results_identical.json` etc.) still hold the pre-fix numbers next to the `_fix1` files; the note
   marks which is current.

Scratch: `/tmp/verify_fix1/{track,ga_analyze}/`.
