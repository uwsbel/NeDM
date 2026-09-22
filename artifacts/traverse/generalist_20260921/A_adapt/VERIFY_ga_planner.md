# Verification: `scripts/ga_planner.py` (plan A3 / A5 offline planning)

2026-09-21, adversarial verifier. Scope: PLAN.md contracts, `A_adapt/NOTES_ga_planner.md`, the code, the self-test
artefacts under `A_adapt/selftest/planner/`, and fresh re-runs of every self-test command into `/tmp/verify_ga_planner/`
(nothing under the artefact tree was modified; the stored `RESULTS.json` keeps its 14:39 mtime). No cluster submission,
no Chrono drive, no CRM run; every run here is GPU scoring only (peak 0.04 GB allocated in a full history-planner
process, RTX 5090 at 1.2 GB total during the checks).

Verdict: **pass with issues**. Every claimed number reproduces bit-for-bit, the contracts hold, and I found no
correctness bug in the module. The issues are one silent-failure usability trap in the reference-pick matching, one
stale open-issue statement in the note, and two integration points outside the module that the A3 launch must handle.

## 1. Reproduced (fresh runs, `/tmp/verify_ga_planner/`)

| test | what I re-ran | result |
|---|---|---|
| T1 | `ga_planner.py` twice + a fresh `planner_arms.py` on the first 3 CRM eval groups, CRM_N2 ensemble, arms A,B | all three runs produce the 6 route files byte-identical to the stored `t1_ref_planner_arms/routes`, `PICKS_LOCKED.sha256` equal, picks of run a == run b byte-identical; A z_mean -6.895, B -7.079, 3/3 = eval_v1 `crm` |
| T2 | rigid ensemble (`night2_v1/final/N2_s*.pt`) on the depth map, `--world rigid --task-root artifacts/traverse --cluster-prefix pair_v1/routes --cluster-case-prefix pair_v1/cases --extra-args '--mode native'` | routes, lock and `tasks_cluster.json` equal to the stored T2; shards {2,0,3} = md5(group) % 6, arena f104, extra `['--mode','native']` on every row; A -9.720, B -9.885 |
| T3 | random hist_aux checkpoints re-generated with `--write-random-ckpt` seeds 0/1 (and tag seed 3, none seed 4 as stored) | every state tensor and every metadata field equal to the stored checkpoints (deterministic scaffolding); planner run on 3 groups: routes equal to stored, `encode_calls_total` 6 = 2 members x 3 decisions, 18 score calls |
| T4 | `--pose-along-s 3` on group 0000 | pose [-28.660, 11.195, -1.505] at station 6.00 m, routes equal to stored, both picks start 0.0 m from the pose |
| T5 | `--from-run eval_v1/runs/f104_crm_eval_group_0000__crm --frame 60` | pose = recorded `pose[60]`, history 40/40 valid, picks/history blocks and routes equal to stored |
| T6 | tag checkpoint with `--domain crm` and `--domain rigid` on group 0001 | A z 4.099 vs 4.107 as claimed; routes equal to stored for both domains |
| T7 | `--poses t7_poses/poses.json` (two runs @60 + one explicit pose) | 3 groups, sources run@60 x2 + poses_file, pose/history blocks and routes equal to stored |
| T8 + all `checks.py` assertions | `checks.py` copied to /tmp (reads the stored dir, writes its RESULTS to /tmp) | exits 0; every block of the regenerated RESULTS.json equals the stored one (T1, T2, T3, T8 max logit/z diff 0.0, T4, T5) |

Extra checks beyond the self-tests:

- `--ref-picks` against the A0 suite picks (the untested open issue): with
  `--ref-picks A_adapt/suite/picks_crm_Scrm/picks --ref-arms A:A,B:B` on the 3 CRM eval groups the planner reports
  6/6 matches, `tasks_new_only.json` has run=false on all 6 rows with `ref_id = <g>__Scrm_<arm>`. Same on the rigid
  side: `picks_rigid_Srigid/picks` on two fresh `f104_pair_group_{0000,0599}` cases gives 4/4 matches and index/z_mean
  identical to the suite's picks (45/220 and 1/1). So ga_planner's rigid depth-map path equals `ga_suite.pick_pass`.
- One invocation per world is possible: `--cases A_adapt/suite/cases` (all 800 groups + `routes/<g>/route_00.json`,
  only `routes` is a non-group entry) plans fresh and reused groups together (checked on one group of each stratum,
  4/4 reference matches, `case = A_adapt/suite/cases/<g>.json` with `--task-root generalist_20260921`).
- History window edge cases (`history_from_trajectory`): k = n-1 gives 40 valid rows with the last row =
  `state[n-1]` / `action[n-2]`; k = 39 gives 39 valid rows (row 0 masked because `action[-1]` does not exist);
  k >= n masks the rows beyond the recording (and `decision_for` raises IndexError on `pose[k]` first, a loud failure).
  Rule identical to `ga_build_mixed.cut_episode` (`si = k-39+t`, `ai = k-40+t`, `ok = ai >= 0`).
- NaN parity: `standardise_history` == `ga_train.prep_hist` (|diff| 0.0) with NaN in both a valid and a masked step.
- float16 history: the builder stores windows as f16, the planner feeds f32; on the frame-60 window (engine speed up to
  176 rad/s, the largest channel) the z difference is <= 1.5e-5 per member. Negligible.
- `p_crm` sign: `ga_train` fits the dom head with `BCEWithLogits(out['dom'], domain)` where domain 1 = crm
  (`ga_build_mixed`: 0 rigid, 1 crm), and the planner's one-hot uses the same code (`DOMAIN_CODE` rigid 0, crm 1 =
  `np.eye(2)[domain]` in the trainer). Consistent.
- Geometry source: only `DS.init_map` is called; `GP.set_map` / `--rigid-arena` do not exist in the planner;
  `corridors_batched: true` in every summary (batched builder bit-equal to `gen_planner.corridors` on the depth map).
- Established-history context matches the training rows: `n2_reanchor_dataset` writes ctx cols 17-21 as
  `[goal - pose_k, |.|, yaw_k (recorded vehicle yaw), L_rem]`; the planner's override path uses `geom_ctx(pose[:2],
  goal, pose[2], L_candidate)`. Same convention (the training corridor starts at the route projection point, the
  planned candidates start at the pose; the anchor filter keeps that gap < 1 m, a dataset design choice).
- Contract loader: `hist_rma` students are saved by `ga_train.train_one` as `GAModel('hist_rma', cin, nctx=5)` (no
  `penc`, `henc`+`hz` present), which `check_contract`/`build_ga` accept; `rma_teacher` is rejected as intended.
- GPU determinism: two runs on the same GPU are byte-identical. The same route scored inside different batch
  compositions differs at the 1e-6 level (pair_group_0599: anchor 1 scored -9.818625 in arm A's 256-batch and
  -9.818631 in arm B's 64-batch), exactly as in `planner_arms.py`; picks stay reproducible on this GPU.

## 2. Problems

### P1 (minor, silent failure) Reference matching silently reports 0/0 when the picks layer is wrong

`--ref-picks A_adapt/suite/picks_crm_Scrm` (the pass directory, not its `picks/` child) is accepted (`ref_dir.is_dir()`
is true), no `<g>.json` is found for any group, the run ends with `reference check: 0/0` and **6 new drives instead of
0**. The same happens in the CRM world without `--ref-picks` on the pair_v1 cases (the default `eval_v1/picks` has no
`f104_pair_group_*`). Nothing fails, and a wave that ships `tasks_cluster.json` from it re-drives already-driven
routes (a budget error, not a correctness error).
Evidence: `/tmp/verify_ga_planner/refpicks_a0/summary.json` (`ref.checked 0`, `n_new_drives 6`) vs
`/tmp/verify_ga_planner/refpicks_a0b/summary.json` with `.../picks` (`checked 6, match 6, n_new_drives 0`).
Smallest fix (`scripts/ga_planner.py` `main`, after the group loop): if `ref_arms` is non-empty and
`ref_match['checked'] == 0`, print a loud `WARNING: no reference pick found for any of the N groups in <dir>` (or
`assert` with an opt-out flag); optionally, when `(ref_dir / 'picks').is_dir()` and `ref_dir` has no `*.json`, use the
child automatically.

### P2 (minor, note correctness) The open issue "trainer/builder hist_cols mismatch" is stale

`NOTES_ga_planner.md` and the report say `ga_train.Data` asserts `len(hist_cols) == 15` while the builder stores 12,
"one of them must change". `ga_train.py` line 256 already extends a 12-entry `hist_cols` with the 3 action columns
before the assertion (`if len(hc) == 12: hc = hc + HIST_ACTION_COLS`), so the mixed npz loads as is and the
checkpoint carries 15 entries equal to the planner's `HIST_COLS` (confirmed on the random checkpoints, which are the
trainer's field layout). Fix: delete that open issue from the note.

### P3 (minor, integration outside the module) Rigid rows carry `extra`/`episode_seed` that the existing runner drops

`scripts/gen_runner.py` (the only rigid runner in the tree) builds the collector command from `case`/`route` only: it
forwards neither `extra` (e.g. `--mode native`) nor `episode_seed`. PLAN R3's `gen_runner_g.py` with `GEN_COLLECTOR`
does not exist yet. The planner's rows are correct per the PLAN row contract; the rigid A3 launch must use a runner that
forwards `extra` (and, if wanted, `episode_seed`), otherwise the rigid drives silently run the plain collector.
`crm_worker.py` does forward `extra` and `episode_seed` and honours `run=false`, so the CRM side is fine.

### P4 (minor, usage) Paths in `tasks.json` become `../..` when `--out` is outside `--task-root`

`route = os.path.relpath(out/routes/..., task_root)`; for an out dir outside the root the row holds
`../../../../../tmp/...`. `tasks_cluster.json` rewrites the route (and optionally the case) path, so the shipped file is
fine, but `tasks.json`/`tasks_new_only.json` are then not usable locally. Keep `--out` under `--task-root` for A3/A5
(the note's example does), or assert `out` is inside `task_root` when `--task-root` is given.

## 3. Contract review (no violation found)

- Checkpoint contract: `check_contract` requires `cond, state, cin (=6), nctx (= 5 + 2 for tag), zdim, norm{mu,sd},
  ctx_mu/ctx_sd (5)`; warns on missing `hist_cols/hist_T/hist_mu/hist_sd/train_rows/split_hash` and on
  `hist_cols != HIST_COLS`. `ga_train.checkpoint_dict` writes all of them (`ctx_mu` is geometry-only, `hist_cols`
  12 + [0,1,2]). Forward signature `model(X, ctx, hist, hmask) -> {'haz'}` plus the `z=` shortcut; T8 shows the
  in-file `GANet` equals `ga_train.GAModel` on random weights for hist_aux / tag / none. The one-logit `dom` head and
  the `[hist | mask]` 16-channel GRU follow the trainer, not the brief's sketch, as the implementer states.
- Observation/history contract: 12 state columns [0-6, 11-15] + 3 actions, T = 40, causal, mask; startup all-masked
  -> the encoder's constant z (T3), the trainer's `--hist-drop` representation. Encoded once per member per decision
  (asserted in `summary.json`).
- Timing: row t = `state[k-39+t]` paired with `action[k-40+t]`, valid iff the action frame exists, equal to the A1
  builder; `pose[k]`, `state[k]` are read at the same frame (`trajectory.npz` pose = [x, y, yaw], dt 0.05).
- Task rows: `{id, group, case, route, run, tier, episode_seed, arms, sha256[, ref_id][, arena, shard][, extra]}`;
  rigid `arena`/`shard = md5(group) % 6` (all arms of a group on one shard, PLAN R20), CRM `extra` verbatim (the user
  supplies absolute paths). `tasks_cluster.json` = run=true rows with `file_sha256` of the route bytes.
- Geometry: OptiX static depth map in both worlds (`map_root/static_map_v1`); no heightmap path.
- Splits/blacklist: the planner builds no dataset; it plans whatever case dir it is given (the suites). Nothing to
  leak. Seeds: `IT.seed(group, tag)` with `planner_arms.arm_specs` tags (`crm_proposal` / `gen_night2` for A,
  `n2iter_cem4x64` for B in both worlds), identical to `ga_suite.DEPLOYED_TAG`.
- float16: corridors rounded to f16 before standardisation for both member kinds, as the deployed pools and the
  trainer's f16 `X`. No overflow risk (corridor channels are O(10)).

## 4. Notes for the A3 / A5 launch

- Plan each world once with `--cases A_adapt/suite/cases` (800 groups) and `--ref-picks
  A_adapt/suite/picks_<world>_<model>/picks --ref-arms B:B` (mind the `/picks`, P1); the CRM tasks file for the wave
  still needs to be merged with the other CRM rows of the wave (PLAN R20) by hand or by `ga_suite.merge`-style code;
  ga_planner has no merge step.
- `ref_id` points at the A0 route id (`<g>__Scrm_B`); in the merged suite that route may have been driven under an
  alias or reused from night 2 (`run_index_<world>.json`), so the analysis must resolve `ref_id` through the run index.
- On another GPU model the picks can differ from the stored ones at argmin ties (1e-6 batch effects); lock picks on
  the machine that plans and ship the locked routes, never re-plan on the cluster.
