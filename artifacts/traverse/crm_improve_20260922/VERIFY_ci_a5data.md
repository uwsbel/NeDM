# Verification: `scripts/ci_a5data.py` (approach routes, pass-1 tasks, decision-state analysis, continuation tasks)

2026-09-22, adversarial check against PLAN.md and the conventions carried over from the previous effort (K1 =
`artifacts/traverse/generalist_20260921`). Nothing was submitted or cancelled, no Chrono run was made, and no repo file
was edited. All re-runs went to a scratch directory, `/tmp/vfy_a5`, so the module's own outputs under `a5data/` were not
touched. Cluster access was read-only (`ssh amd` for listings and hashes).

**Verdict: pass with issues.** I found no contract violation and no wrong number in the shipped task files. The history
window, the decision frame, the speed at the decision (v0), the sign conventions, the splits, the blacklist, the physics
config and the path formats are all correct. What is left are silent-failure risks in the analysis stage and a few
caveats for whoever uses the outputs. None of them changes the task files that exist today.

## 1. Re-runs (all reproduce)

```bash
cd /home/harry/NeDM-traverse_mppi; PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts OMP_NUM_THREADS=6
$PY scripts/ci_a5data.py --stage approach-routes --cases artifacts/traverse/generalist_20260921/A_adapt/suite/cases --out /tmp/vfy_a5   # 3.7 s
$PY scripts/ci_a5data.py --stage approach-routes --cases artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases --out /tmp/vfy_a5 # 4.6 s
$PY scripts/ci_a5data.py --stage pass1-tasks --out /tmp/vfy_a5
$PY scripts/ci_a5data.py --stage selftest-analyze --out /tmp/vfy_a5          # SELFTEST analyze OK
$PY scripts/ci_a5data.py --stage selftest-continuations --out /tmp/vfy_a5    # SELFTEST continuations OK (GPU < 0.2 GB)
```

- The regenerated approach routes are byte-identical to `a5data/approach_suite/` and `a5data/approach_twin/`
  (`diff -rq` finds no difference). The route summary numbers match the report: 800 and 1,200 straight lines, all
  chosen by tie. route_00 differs from the straight line in 0 groups (largest lateral distance 1.7e-14 m). The largest
  grade over the first 12 m is above 12 deg in 301 suite and 418 twin groups, and above 17 deg in 196 and 285. The twin
  split is 1,089 / 56 / 55.
- All 14 regenerated pass-1 task files have the same sha256 as the files in `a5data/tasks/`.
- Both self-tests pass with the reported numbers. At 3 s the pose, vx and sinkage differences against K1 are 0.0, and
  10/10 history windows are equal. Sinkage read at crm_extra row F-1 instead of row F differs by 0.0057 m at 0.5 s and
  0.0260 m at 1 s. Every continuation check is true.
- Merge-stage guards, tried by hand: three twin CRM length files merge into 3,600 rows with the same content as
  `tasks_pass1_twin_crm.json`. A suite file is refused ("suite groups"), mixed worlds are refused, and duplicate ids are
  refused.

## 2. Conventions checked

| item | finding |
|---|---|
| History window | Row t uses state[j] on the 12 observable columns and action[j-1], with j = F-39+t, valid iff j >= 1. This is state[F-39+t] with action[F-40+t], masked where the action frame is < 0. That matches the CRM collector's own `prefix_history` (crm_collect_ext.py:77-89), the rigid collector's copy, and `ga_branch_dataset.prefix_hist`. The column list [0-6, 11-15] is the same in ga_approach, ga_build_mixed and gc_control. At F = 10 / 20 the window has 10 / 20 valid rows. |
| Decision frame | A pass-1 run with `--horizon-s L` records rows 0..F-1. The terminal state is measured after F intervals, and the branch collectors swap at the top of frame F, so the terminal state and terminal pose are the right decision state. **Checked beyond the self-test's 5 groups:** in every K1 3 s group with both drives available locally (778 CRM, 790 rigid), the pass-1 terminal pose (x, y and yaw) and the 12 observable state columns equal row 60 of the K1 pass-2 branch drive exactly. So the terminal state really does equal the branch drive's row F, and in K1 it did so even for rigid drives that ran in different jobs. |
| v0 | v0 = state[F][0], the body-frame forward speed. It is the same value ci_planner takes from the last valid window row (cross-check 0.0). Negative v0 is clipped to 0 inside the family code. The speed step is the route's start speed minus v0, which is the plan's sign. The heading error is the wrapped difference headings[0] - yaw, with yaw from the collector pose (identical to the recorded row yaw, see above). |
| Label-free CEM | The K1 H members use `hist_aux` conditioning. Passing `domain=world` only matters for tag-conditioned members, so the picks do not see the world label. |
| Splits / blacklist | The twin split comes from `twin_crm.npz` (1,089 / 56 / 55) and is asserted equal to the case files. The suite patterns are asserted equal to `ga_build_mixed.BLACKLIST`. The suite guard in the continuation stage also covers `--all-groups`; only the self-test can lift it. |
| Physics | The launch templates use `CRM_CONFIG=configs/crm_main.json` relative to the CRM root. On the cluster that file has md5 76378bb5..., the same as the local `crm_f104_v1/configs/crm_main.json`, with `step_s` 0.001 (1 ms). The branch collector is `crm_collect_ext.py`; its name contains `crm_collect`, so the worker passes `--crm-config`. |
| Row formats | CRM worker: case and route are relative to the CRM root; the extra arguments come after the worker's own `--horizon-s 120`, so `L` wins; the branch route path is absolute (the collector resolves it against the working directory). Rigid runner: all paths absolute; `mode` becomes `--mode`; the shard is md5(group) % 6, so the six continuations of a group share a node. Both workers skip rows with run = false (the removed duplicates). The labeller reduces `<g>__p1_<len>__c<slot>` to `<g>__p1_<len>` correctly and finds everything it needs in the anchors (F, pose_F, vx_F, split, cls, remaining_m, world). |
| Case copies on the cluster | I re-hashed all 4,000 cluster files over ssh today (800 in `C/generalist/suite/cases`, 800 in `generalist_20260921/a5/cases`, 1,200 in `C/cases/night2`, 1,200 in `fdm_f104_50h_20260909/cases_night2_v1`): 0 mismatches against the local files. The `case_sha256` in every row of all 14 task files matches too. |
| Short horizons in the collectors | Neither collector's code after the drive loop needs more than 10 frames: its windows use `range(max(0, n-80+1))`, and a 0.5 s horizon passes the rigid "positive 50 ms multiple" check. |
| Costs | Confirmed from K1 outcome files: CRM pass 1 took 1.65 h of summed wall time (median 7.4 s per drive); CRM pass 2 took 77.9 h (4,652 drives, median 55.9 s, 24.0 simulated h). See issue 6 for what these sums leave out. |

## 3. Issues (most important first)

1. **Silent loss of groups when one world's runs are missing or only partly synced (medium).** The analysis set is the
   intersection of the "moving" groups of every world given, and a `--runs` directory that does not exist is treated as
   empty. Test: `--stage analyze --runs <5 CRM runs> /tmp/vfy_a5/does_not_exist --worlds crm rigid --length 0.5` exits 0
   with `n_analysis_set 0`. `--stage continuations --world crm` then exits 0 as well. It writes an empty
   `tasks_cont_twin_crm_L0p5.json` (0 rows), an anchors file, a report and a SHIP file. With a partial sync (for
   example, only half of the rigid pass 1 back) the CRM continuations silently lose the other half. The
   `excluded` list does not separate "missing in a world" from "not moving". Fix: assert that every world has the same
   group set (or list the missing groups apart), and refuse to write a task file with 0 rows.
2. **The default poses file can quietly drop suite groups at 0.5 s (medium-low; affects the S2 closed loop).**
   `poses_<world>.json` holds only groups moving in both worlds, and the notes' example for the suite closed loop uses
   it. In K1's rigid 3 s recordings, vx at frame 10 is 0.269 m/s in f104_pair_group_0189, 0.295 in _0348 and 0.304 in
   _0161. That is right at the 0.3 m/s cut, so the 0.5 s suite set would be 797-800 groups depending on the rigid node.
   The primary decision rule compares on the same 800 groups. For S2, plan from `poses_<world>_all.json`, or decide
   up front how excluded groups count. At 1 s no suite group is near the cut (smallest vx at frame 20: 0.36 m/s rigid,
   0.66 CRM).
3. **The pass-1 status is not checked (low).** A recording with exactly F rows is accepted as a decision state
   whatever its status. A drive stopped by the rollover or soil-breakthrough rule at frame F-1 would also have F rows.
   This is very unlikely in 0.5-3 s from the start pad. K1 had the same gap. Fix: require status `timeout` (horizon
   reached) for F-row recordings.
4. **The tolerances used for continuation routes are looser than the rigid collector's (low).** `validate_branch`
   uses a 0.5 m goal tolerance for both worlds, but the rigid collector refuses routes that end more than 0.25 m from
   the goal (gen_collect_ext.py:216). The production path also does not re-read routes with the frozen collector's
   route reader (only the self-test does). In practice every sampled and CEM route in the self-tests ends within
   1.8e-15 m of the goal, so today this would only ever fail loudly on the cluster, never silently.
5. **Some identity tests show less than they appear to (evidence strength, not a defect).**
   - Short-approach test B builds the "cut at F" copies with `terminal_state := state[F]`. So "cut equals full row F"
     for pose and vx holds by construction. The meaningful part of that test is the window check against two
     independent implementations (the planner's `history_from_trajectory` and the labeller's `prefix_hist`); the
     window indexing at F = 10 / 20 is covered by that.
   - No real 0.5 s or 1 s recording exists yet. The terminal-state semantics are verified only at F = 60, now on
     ~780-790 groups per world (above). The code path does not depend on F.
   - The 3 s test against K1's own analysis reuses ga_approach's functions, so it is a regression check rather than an
     independent one.
   - "CLI picks slot 5" compares ci_planner with itself. "CEM free equals K1 H pick" is a real like-for-like check:
     same state, same models, same seeds.
   - The labeller test uses K1's stripped local copies with a zero `crm_extra.npz`, so it checks the format only (the
     notes say so). Real runs do write `case.json`, `command_reference.npz` and `crm_extra.npz`, and the sync-back line
     brings them along.
6. **The cost figures leave out part of the per-drive time (low).** `wall_s` in outcome.json starts after the soil
   build (median 2.6 s in pass 1, 2.9 s in pass 2) and does not include process start-up. So the notes' phrase "set-up
   dominated" for pass 1 is not what the 1.65 h sum measures, and the real GPU time of 0.5 s and 1 s drives will be a
   larger multiple of their wall sums. The continuation estimate (about 120 GPU-hours, 15-18 billed node-hours per
   length) is dominated by physics and stands as a rough figure. Continuations after a 0.5 s approach also have about
   7 m more route left than after 3 s, so they will be somewhat longer drives.
7. **Prerequisites on the cluster (informational).** Checked today: `G2/source` exists but is empty (no
   `gen_array_g.sbatch`, `gen_runner_g.py`, `gen_collect_ext.py`, `crm_collect_ext.py`, `source_manifest.json` or
   assets), and `C/crm_improve/` does not exist yet. The rigid pass 1, not only the branch drives, needs a full source
   copy in `G2/source`, because the collector gets `--source-root $G2/source` and reads the arena from there.
   `SHIP_pass1.txt` mentions this only in a comment.
8. **Smaller points (low).** Merging anchor files does not check that they come from one world (the labeller's world
   assert catches it later). If analyze runs without `--approach-dir` and before `command_reference.npz` is synced,
   `remaining_m` is None, and the labeller's `float(an.get('remaining_m', nan))` then crashes; it is loud, not silent.
   Sinkage is read at row F for longer recordings and at row F-1 for F-row recordings (difference up to 0.026 m against
   the 0.1 m cut).

## 4. Points I agree with the module's open issues on

- The 0.3 m/s moving threshold below frame 60 is the module's own choice. In K1's CRM recordings 713/800 groups are at
  or below 1.0 m/s at frame 10 (median 0.89), and 18/800 at frame 20 (median 1.72). In rigid the figures are 127/800 and
  5/800. With 1.0 m/s the 0.5 s soil set would lose most groups, so a lower cut is needed. Issue 2 is its side effect.
- The K1 H ensemble never saw windows with only 10 or 20 valid rows, so its picks at those frames are proposals only.
  Rigid prefix replays can drift across nodes; the labeller reports it, and K1 A5 showed none.
- The labeller's disagreement summary assumes 3 continuations per anchor. Budget: choose the approach lengths before
  shipping.

## 5. Files

- Checked: `scripts/ci_a5data.py`, `NOTES_ci_a5data.md`, `a5data/approach_{suite,twin}_index.json`, `a5data/tasks/*`,
  `a5data/SHIP_pass1.txt`, `a5data/selftest/*/RESULTS.json`.
- Re-run outputs (scratch, not part of the repo): `/tmp/vfy_a5/` (logs `ar_*.log`, `p1.log`, `st_an.log`, `st_co.log`,
  `remote_sha.txt`, the empty-world probe in `t_empty/`).
