# Verification: `scripts/ci_planner.py` and the S1 pick sets

2026-09-22 08:58-09:25, adversarial check of the S1 planner module (speed- and heading-continuous route candidates at a
moving branch). Read: `PLAN.md`, `NOTES_ci_planner.md`, `scripts/ci_planner.py`, and the code it wraps or feeds
(`ga_planner.py`, `f104_n2_iter.py`, `planner_arms.py`, `f104_n2_sampler.py`, `gen_planner.safe_validate` /
`validate_reference`, `ci_train.py` scoring functions, `ga_a5_pass2_tasks.py`, the branch mode and follower of
`crm_collect_ext.py` / `traverse_fdm_rgbd_diverse_chrono.py`). Every run of mine wrote to `/tmp/verify_ci_planner/`.
Nothing in K1 or the S1 sets was changed. No Chrono, no cluster, GPU use under 1 GB per process (at most 3 at once).

**Verdict: pass with issues.** The three S1 pick sets are correct and can be driven as they are. Nothing I found
changes a single S1 route. The issues are guards that are missing for later stages (standing starts in S4, some
little-used v0 paths) and bookkeeping.

## What was re-run and what came out

Environment for every command: `cd /home/harry/NeDM-traverse_mppi; PY=/home/harry/miniconda3/envs/nedm/bin/python;
export PYTHONPATH=src:scripts OMP_NUM_THREADS=6`; K1 = `artifacts/traverse/generalist_20260921/A_adapt`.

1. **Self-tests** (the checkpoints from the new trainer copied first):
   `cp -r .../s1/selftest/ci_ckpt /tmp/verify_ci_planner/selftest/; $PY scripts/ci_planner.py --selftest /tmp/verify_ci_planner/selftest`
   -> ALL PASSED in 59 s. Every row of T1, T1b, T2, T3, T3b and T4 is identical to the reported `s1/selftest/RESULTS.json`.
   T5 scorer difference 7.15e-7, 4 history encodings per decision (once per member), as reported. `--selftest-ci` alone
   also passes.
2. **Free family equals K1 on all 800 groups, not just 3** (the self-test T1 covers 3):
   `$PY scripts/ci_planner.py --family free ... --models "$K1/train/deploy_v1/H_deploy_s*.pt" --world crm --domain crm --arms B --poses $K1/a5/poses_crm.json --out /tmp/verify_ci_planner/free_H_800 --task-root $K1 --verify 0`
   -> all 800 route files byte-identical to `K1/a5/picks_crm_H/routes`. Same lock hash (`20f9426f2eff...`). z, pool
   index, route sha256, number evaluated and tries equal in 800/800 picks.
3. **Free family on standing starts** (case start pose, no poses file, arms A and B, the default `--verify 3` pool check on)
   against `ga_planner.py` itself, 5 groups (crm_eval 0005 / 0150, pair 0100 / 0333 / 0599): route files and lock
   identical. `tasks.json` differs only in the output directory name.
4. **Full rebuild of both H sets with the current file.** `ci_planner.py` was last changed at 08:41. H_cont had finished
   at 08:39 and H_conthead had started at 08:39, so both came from an earlier version of the file. The note's rebuild
   check covered 20 groups per set. All 800 groups were rebuilt into `/tmp`:
   H_cont lock `31c22e71...` and H_conthead lock `c6f4dbe4...` reproduced exactly, 800/800 route files byte-identical;
   `tasks.json` identical apart from the route directory. S'crm/cont_head was built after 08:41 (lock `0f7da72b...`).
5. **Independent checks of every S1 route file** (own script, `/tmp/verify_ci_planner/check_sets.py`, 3 x 800):
   first speed minus the frame-60 speed = 0.0 (max absolute value); acceleration within [-2.0000000000000107,
   1.5000000000000173] m/s^2; start heading at most 51.857 deg (H cont), 19.959 deg (H cont_head), 19.998 deg (S'crm
   cont_head); first waypoint 0.0 m from the poses-file pose; last waypoint within 4e-15 m of the case goal; speeds
   in [0.0, 6.0]; 0 invalid under the planner's validator; 0 content-hash mismatches against `tasks.json`; the frozen
   collector's route reader (`traverse_fdm_rgbd_diverse_chrono.read_route`) accepts all 2,400 files.
   Every pick file says v0 source `history_window`, v0 equal to the window value, cross-check difference 0.0.
6. **Decision state, window convention and v0 source** (all 800 CRM groups):
   the history window equals the rebuild `hist[t] = [state[k-39+t] on the 12 observable columns, action[k-40+t]]`
   at k = 60 with max |difference| 0.0. Frame 60 comes from `terminal_state`, because the pass-1 run stops after
   60 rows. All 40 rows are valid and the newest row is row 39. The poses-file pose equals the pass-1 `terminal_pose`
   (difference 0.0). The window's newest row, column 0 (`vel_body_x_mps`), equals `pass1_state.json` vx (difference
   0.0; range 1.20-3.53 m/s). v0 through `ga_planner.decision_for` + `decision_v0` on all four K1 poses files:
   `poses_crm.json` / `poses_rigid.json` -> `history_window` 800/800; the masked-history files -> `pass1_run` 800/800;
   every one equals pass-1 vx exactly. Case start pose -> 0 (`standing_start`). A pose override with no speed raises an error. A negative explicit v0 is clipped to 0.
7. **The vehicle really is at v0 when the new route takes over.** In the 778 K1 pass-2 H drives, the recorded
   `vx_at_branch_mps` equals pass-1 vx (max difference 0.0), the branch pose equals the poses-file pose (0.0), and the
   start offset is 0.0 m. The prefix replay is exact, so a route that starts at v0 starts at the vehicle's speed.
8. **Heading sign convention.** The base route's own start heading is at most 0.165 deg off the vehicle yaw (median
   0.017 deg, 800 groups). A flipped yaw sign or frame would show up here as large errors.
9. **The speed change never flips the validator's verdict**, re-tested on 40 other decisions (every 20th group from
   group 3, apart from the self-test's set) with v0 in {recorded, 0, 5.9, 8.0} m/s: 16,000 draws, verdict unchanged in all
   16,000. Every valid transformed route ends at 0 m/s, acceleration within [-2.0000000000000195, 1.5000000000000107].
   By construction the speed change can only move speeds inside [0, 6] while keeping the acceleration limits, and it
   leaves the geometry alone, so the verdict can only differ by floating-point noise.
10. **Python API.** `plan_decision()` equals the command-line picks for free, cont and cont_head on 3 groups each
    (crm_eval 0077, pair 0250, pair 0598): same route sha256 and z.
11. **Pass-2 rows.** `ga_a5_pass2_tasks.py --world crm --arms H_cont,H_conthead,Spcrm_conthead --picks-dir K2/s1
    --cluster-case-prefix generalist/suite/cases --cluster-approach-prefix generalist/a5/approach
    --cluster-route-prefix crm_improve/s1/pass2/routes_crm --out /tmp/verify_ci_planner/pass2/tasks_pass2_crm.json`
    -> 2,400 rows, 2,369 to drive (14 and 17 duplicates, as reported). Same keys as K1's `tasks_pass2_crm.json`.
    0 hash mismatches in the copied routes, and no id collides with a K1 row. The new rows get new episode seeds; in
    `crm_collect_ext.py` the seed is only used by the `pid_perturbed` mode, so branch drives are unaffected.
12. **Comparison tables.** Speed-step bins, heading bins, z means, P > 0.1 / P > 0.5 counts and the paired z changes
    (+0.013 / +0.204 / +0.301; higher in 56.5 / 69.0 / 67.2 %) of the note reproduced exactly from the pick files.

Contract checks against PLAN S1 and the K1 conventions: families as specified (floor = the 2 m/s^2 slowing ramp from
v0, cap = the 1.5 m/s^2 speed-up ramp from v0, start heading within 20 deg of the yaw); arms H/cont, H/cont_head,
S'crm/cont_head; CEM 4x64 with K1's seeds; frozen frame-60 states of K1 pass 1; picks hashed before driving; no
training, and the suite groups are used only for planning, as S1 intends. The planner does not touch the physics
settings; the CRM row format and relative paths are K1's (relative to the CRM root, as in K1).

## Problems found (most important first)

1. **Standing starts with cont / cont_head are not refused (matters for S4).** With the case start pose, v0 = 0
   (`standing_start`) and every cont / cont_head route starts at 0 m/s (T4 confirms). The frozen follower sets its
   target speed from the nearest waypoint among the next 60 (`nearest_index`, `traverse_fdm_rgbd_diverse_chrono.py:115-117`;
   `SetDesiredSpeed(speed[wp])`, `crm_collect_ext.py:218-220`). So a vehicle standing on waypoint 0 is told 0 m/s and
   will not move. The note describes this, but the code runs without a warning, and the saved route looks normal.
   Suggested fix: refuse `--family cont|cont_head` when v0 < 0.5 and no `--v0-min` is given (or make the ramp floor
   0.5 m/s for `standing_start`). Until then, the S4 standing-start check must use `--family free` or `--v0-min 0.5`.
   For S2 this is less acute: at frame 10 of the same approach the vehicle does 0.59-1.03 m/s (p5 0.73), at frame 20
   0.66-2.29 m/s. Those routes start slowly but not at zero.
2. **v0 rule for a `--history <trajectory file>` reads the wrong file** (`decision_v0`, `trajectory_vx(Path(path).parent, k)`).
   It loads `<folder>/trajectory.npz`, not the file given. Demonstrated: history = a copy of run 0000's trajectory
   under another name, with run 0001's `trajectory.npz` in the same folder -> v0 2.848 (run 0001) instead of 2.888
   (run 0000), without a warning. With no sibling file it fails loudly. There is also a one-frame gap when frame =
   number of rows: the window stops at row n-1 (ga_planner's behaviour) but v0 uses `terminal_state`. S1 does not use this path.
   Fix: load the given path.
3. **The encode-once check is empty for the new trainer's checkpoints.** For a `ci_train` ensemble, `summary.json`
   reports `members_with_history 0, encode_calls_total 0, expected_encode_calls 0` (T5 run), so ga_planner's
   once-per-decision check passes without testing anything. The right count (`n_encode 4`) is only in each
   `picks/<g>.json` history block, and only T5 checks it. Suggested: check `n_encode == number of history members`
   per group in `postprocess` and write the real numbers into `summary.json`. No score is affected (T5 matches
   `ci_train.score` to 7.2e-7).
4. **"The route still stops at the goal" needs a route at least v0^2 / (2 x 2.0) m long** (3.1 m at 3.53 m/s). On
   shorter routes the slowing floor keeps the speed above zero at the goal, and the validator still accepts the
   route. Demonstrated: 2.5 m route, v0 3.5 m/s -> every cont route ends at 1.5 m/s. It cannot happen in S1 (the
   shortest remaining route at a branch is 21.4 m), and the collector already commands 0 within 3 m of the end. It
   matters only if `candidates()` is used close to a goal.
5. **`run_s1_picks.sh` logs `rc=$?` after `$(date ...)`, so the logged rc is always 0** (demonstrated with `false`).
   It is the same pattern as K1's `run_a5_planning.sh`. The three sets are complete (800 picks, `summary.json` with
   the family block), so no set is affected.
6. Cosmetic: the note says the base route starts 0.03-0.06 deg off the yaw; measured over 800 groups it is median
   0.017, max 0.165 deg.

Not problems, but worth knowing: `cont` combined with `--fixed2` would replace the constant 2 m/s profile with ramps
from v0, and nothing prevents it (not used). `candidates()` does not check suite group ids; the stage that builds the
S3 data must keep the blacklist. The two caveats in the note (the new follower starts the branch with no throttle
memory, so expect a brief coast; the 20 deg limit raises the model's own risk estimate from 29 to 115 groups with
P > 0.5 for H) are stated correctly. Neither can be checked without driving.

## Files

- This note: `artifacts/traverse/crm_improve_20260922/VERIFY_ci_planner.md`
- Scratch (not kept): `/tmp/verify_ci_planner/` (self-test re-run, the 800-group free run, the two rebuilds, the pass-2
  dry run, the check scripts `check_sets.py`, `check_v0.py`, `check_verdict.py`, `check_short.py`, `check_api.py`)
