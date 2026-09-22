# Module note: `scripts/ga_planner.py` (plan A3 / A5 offline planning)

2026-09-21. One new file, `scripts/ga_planner.py`; no existing repo file was edited. Self-test artefacts under
`A_adapt/selftest/planner/` (`checks.py` runs the checks and writes `RESULTS.json`).

## What it is

The planner_arms pick pass (one-shot 256 = arm A, CEM 4x64 = arm B, arms C-F available) for an ensemble of either kind:

- legacy N2 checkpoints (`CRM_N2_s*.pt`, `N2_s*.pt`; no `model_kind`): rebuilt with `gen_riskmodel.Net` exactly as
  `gen_planner.RiskModel` does and scored with the arithmetic of `f104_n2_iter.member_logits`;
- shared-model checkpoints with `model_kind='ga_train'` (contract in PLAN.md): rebuilt by an in-file network `GANet`
  that copies the deployable part of `ga_train.GAModel` (same state-dict keys: `front.cnn.*`, `lat`, `ctx.0`, `tconv.0`,
  `mix`, `head`; `henc` + `hz` for the history conditions; `dom` for `hist_aux`). `ga_train.py` is not imported by the
  planner; the self-test imports it once to prove both implementations agree.

Both worlds score from the OptiX static depth map (`--map-root`, `f104_n2_dataset.init_map`); `--world` only sets the
deployed rng tag (`crm_proposal` / `gen_night2`, `_fixed2` variants with `--fixed2`), the default reference picks and
whether task rows carry `arena`/`shard`. Arm tags and labels come from `planner_arms.arm_specs`, so seeds are identical.

Decision context, per member: `ctx = [(geom5 - ctx_mu) / ctx_sd | domain one-hot (rigid, crm) if cond == 'tag']` with
`--domain` (default = `--world`; override it for the cross-domain oracle control). History: a (40, 15) window
`[state cols 0-6, 11-15 | applied action]` with a validity mask, standardised with `hist_mu`/`hist_sd`, masked steps
zeroed, mask appended as channel 16, encoded ONCE per decision in `Scorer.__init__` (per member) and broadcast over every
candidate through the network's `z=` argument. At startup the window is all-masked (z = the encoder's constant startup
representation, the same one the trainer sees under `--hist-drop`). `summary.json` asserts `encode_calls_total ==
members_with_history x decisions`.

Start pose and base route: the case layout pose with `routes/<g>/route_00.json` (planner_arms behaviour), or an override:
`--pose-override x y yaw`, `--pose-along-s t` (the point t s along route_00 of the case), `--from-run <dir> --frame k`
(recorded pose at frame k, history cut there), or `--poses <json>` for a batch (`{group: {pose|run+frame, goal?,
history?}}`, plans those groups only). With an override (or `--goal`) the base route is `gen_planner.base_route(pose,
goal)` (Hermite with fallbacks) and every candidate of the night-2 family starts at the pose exactly (the lateral basis
is zero at both ends; the validator anchors at the pose). History windows follow the A1 builder rule: row t = state at
frame k-39+t paired with the action at frame k-40+t, valid iff that action frame is >= 0 (frame 0 -> all masked).

Outputs (planner_arms formats): `picks/<g>.json` (planner_arms fields + `pose`, `goal`, `source`, `history`
{T, n_valid, per-member z_norm and p_crm for hist_aux}, `start_dist_to_pose_m`), `routes/<g>__<arm>.json`
(content-deduplicated), `tasks.json` rows `{id, group, case, route, run, tier, episode_seed, arms, sha256[, ref_id]
[, arena, shard][, extra]}` (paths relative to `--task-root`; rigid: `arena` = `--arena-tag`, `shard = md5(group) %
--shards`, default 6 shards per PLAN R20), `tasks_new_only.json` (run=false where a reference pick has the same route:
by `route_sha256` when the reference entry has it, by pool index otherwise; `--ref-picks DIR --ref-arms A:crm,B:B`),
`summary.json`, `PICKS_LOCKED.sha256`, and with `--cluster-prefix P [--cluster-case-prefix Q] [--cluster-all]`
`tasks_cluster.json` with `route = P/<id>.json`, `case = Q/<group>.json`, plus `file_sha256` of the route file (the
row's `sha256` keeps the planner_arms meaning, the content hash of the route arrays). `--extra-args "..."` stores the
shlex-split list as `extra` on every row (crm_worker appends it to the collector command).

## How to run

    PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts
    # A3 startup picks, CRM world, a ga_train ensemble
    $PY scripts/ga_planner.py --cases artifacts/traverse/generalist_20260921/cases/pair_v1/cases \
        --map-root artifacts/traverse/crm_f104_v1/map_root --models '<dir>/H_s*.pt' --world crm --arms A,B \
        --out <out> --ref-picks <A0 picks dir> --ref-arms B:B --task-root <root> --cluster-prefix <routes on cluster>
    # rigid world: add --world rigid [--arena-tag f104 --shards 6]; tag oracle told the other domain: --domain rigid
    # A5 pass 2 (batch): --poses poses.json with {group: {"run": "<pass-1 run dir>", "frame": 60}}
    # test scaffolding: --write-random-ckpt path --cond hist_aux --seed 0

Latency on the 5090 (legacy 5-member ensemble): A 0.55 s/group, B 0.42 s/group; the random 2-member history ensembles
0.5-0.7 s/group.

## Self-tests (all under `selftest/planner/`, `checks.py` output in `RESULTS.json`)

- T1 `t1_ref_planner_arms` vs `t1_ga`: `scripts/planner_arms.py` and `ga_planner.py` on the first 3 groups of
  `crm_f104_v1/cases_eval/cases`, CRM ensemble, arms A,B. Picks identical for all 3 groups x 2 arms (route sha256,
  z_mean and pool index equal), the 6 route files byte-identical, `PICKS_LOCKED.sha256` equal, task rows equal
  (3 new drives, A = eval_v1 `crm` pick 3/3). A z_mean -6.895, B -7.079.
- T2 `t2_rigid`: rigid ensemble `fdm_f104_50h_20260909/night2_v1/final/N2_s*.pt` (the path in the brief,
  `gen_v1/models/`, does not exist) on the depth map, same 3 groups, `--world rigid`: rows carry `arena=f104`,
  `shard = md5(group) % 6` (all arms of a group on one shard), `extra=['--mode','native']`; `tasks_cluster.json` paths
  rewritten and `file_sha256` verified. A z_mean -9.720, B -9.885.
- T3 `t3_ga_random`: two random-weight `hist_aux` contract checkpoints (zdim 16), 3 groups: the code path runs, the
  history is encoded exactly 6 times (2 members x 3 decisions) while the scorer was called >= 2 times per group, the
  all-masked z equals the encoder's constant, picks written; on a real corridor batch with a partially valid window
  (15/40 steps) the precomputed-z path and the in-forward encode path give the same logits (max |dz| 0).
- T4 `t4_pose`: `--pose-along-s 3` on group 0000: pose [-28.66, 11.19, -1.505] at station 6.00 m of route_00; the
  full 256-candidate pool rebuilt from `base_route(pose, goal)` starts at the pose (max distance 0.0 m, bound 0.25 m);
  both picks and route files start there. Informational: `base_route(layout pose)` reproduces route_00 exactly.
- T5 `t5_from_run`: `--from-run crm_f104_v1/eval_v1/runs/f104_crm_eval_group_0000__crm --frame 60`: pose = the
  recorded pose at frame 60 (vx 1.31 m/s), window 40/40 valid and equal to `state[21..60]` / `action[20..59]`;
  frame 0 gives an all-masked window, frame 20 gives 20 valid steps; z_norm > 0.
- T6 `t6_tag_none`: random `tag` (nctx 7) and `none` checkpoints load and plan; the domain one-hot changes the logits
  (A z 4.0993 with crm vs 4.1067 with rigid).
- T7 `t7_poses`: a poses file with two recorded runs at frame 60 and one explicit pose plans exactly those 3 groups.
- T8: this loader/scorer vs `ga_train.load_ga_model` + `ga_train.score` / `encode_history` on 10 (cond, domain, member,
  window) cases: max |logit diff| 0, max |z diff| 0.

Command: `PYTHONPATH=src:scripts $PY artifacts/traverse/generalist_20260921/A_adapt/selftest/planner/checks.py`
(T1-T7 runs are the commands recorded in each subdir's `log.txt`).

## Known limits / things a reviewer must know

- The network definition is a copy of `ga_train.GAModel` made after that file appeared (14:32); T8 proves agreement on
  random weights. If the trainer's architecture changes (encoder width, zdim, keys), `GANet` and `build_ga` must follow;
  the loader errors on missing keys and warns on unused ones.
- Contract field `hist_cols`: the trainer writes 12 state columns + action columns [0, 1, 2]; `ga_build_mixed.py`
  stores only the 12 state columns in its npz (`hist_cols (12,)`) while `ga_train.Data` asserts 15 when the key exists.
  That is a trainer/builder mismatch, not the planner's; the planner only warns on `hist_cols` differences.
- Established-history windows come from a recorded trajectory (`--from-run` / `--poses`), so A5 pass 2 needs pass-1 run
  dirs synced locally (trajectory.npz only).
- `--rigid-arena` (heightmap) is deliberately not offered: PLAN convention, one geometry source (depth map) in both
  worlds. Rigid picks therefore differ from the night-2 heightmap picks by construction.
- Arms C-F run but were not self-tested here (planner_arms specs, unchanged).
- CEM is not guaranteed to beat one-shot on a single group (T4: B's pick z -2.36 vs A's -2.96 from a hard mid-route
  pose): different rng streams, argmin over what each arm evaluated. Expected, not a bug.
