# Scout map: original NeDM mixed-domain HMMWV dynamics, PPO tracker, Chrono evaluators

Scope: plan B "borrow the original domain conditioning / mixed-data training". Read-only; every
claim verified in code. Unprefixed paths are relative to `/home/harry/NeDM` (branch `nrd_vision`,
HEAD 4b6f2a173); "worktree" is `/home/harry/NeDM-traverse_mppi`.

## Which copy to build on

`diff -q` over every in-scope file: identical in both checkouts (trainer stack
`src/nedm/training/{trainer,model,model_transformer,dataset,preprocess,constants}.py`, all
`src/nedm/rl/*`, both scripts, both configs, all three docs) EXCEPT `src/nedm/hmmwv_crm.py`. The
worktree copy adds shims `SoilProperties|ElasticMaterialProperties` and `SetCrmSPH|SetElasticSPH`
(worktree `hmmwv_crm.py:111-125`) but still calls `SetActiveDomainDelay` (`:149`), which the cluster
FSI build lacks (memory: it has `SetFreeFlowDuration`). Only worktree `scripts/crm_collect.py:118-122`
handles both names. `scripts/training/train_hmmwv_dynamics.py` is an 18-line wrapper; the trainer is
`src/nedm/training/trainer.py`.

## (1) Domain conditioning, mixing, state preset, timing

- One-hot is appended PER TOKEN: `_build_tokens` concatenates `one_hot(terrain_id)` to every
  `[state_norm, action_norm]` token (`model.py:119-133`); a `(batch,)` id is expanded to
  `(batch, seq)` (`model.py:98-117`); `input_dim = 15 + 3 + 2 = 20` (`model.py:61-64`). The model
  raises when `terrain is None` (`model.py:126-128`): a hard integer id is mandatory at every call;
  no soft-probability input exists.
- Config keys: `terrain_conditioning.{enabled, terrains:["flat","crm"]}` (config lines 5-8;
  `trainer.py:266-278`); the vocabulary is written into checkpoint metadata (`trainer.py:274-277`)
  and restored by `load_frozen_dynamics` (`rl/dynamics.py:65-72`). Per-source id from the spec's
  `terrain` or `name` key (`trainer.py:470-490`).
- "mix25": `train_mix.datasets[].batch_fraction` 0.75/0.25 (config lines 9-24) ->
  `allocate_batch_sizes` gives 48 flat + 16 CRM rows per batch of 64 (`trainer.py:171-210, 314-315`);
  one `RandomSampler(replacement=True)` loader per source, sub-batches tagged with `terrain_ids` and
  concatenated (`trainer.py:113-135, 343-360`). Normalisation stays flat-only; the
  `model_normalization.equal_domain_combined` option (`trainer.py:507-521`) is NOT used in the onehot
  config (progress.md:344: combined input normalisation was the wrong lever).
- "rebal": `loss.channel_weight_mode = equal_domain_combined_std` -> `w_i = flat_std_i^2 /
  mean_d(std_{d,i}^2)`, mean-normalised, applied as `sqrt(w)` on the residual before Huber
  (`trainer.py:523-566`). Validation: flat loader (id 0) + CRM loader (id 1), `val_mixed_loss`
  0.5/0.5 (`trainer.py:364-414, 616-647`); checkpoint metric `rollout_sel` = 0.5 flat + 0.5 CRM
  distance-normalised open-loop xy error at 10 s, 12 episodes per domain (`trainer.py:649-794`).
- Preset `tire_normal_force_omega` = 7 body [vx, vy, roll, pitch, roll_rate, ang_vel_body_y,
  yaw_rate] + 4 `tire_*_force_wheel_fz_n` + 4 `tire_*_spindle_omega_radps` = 15-D
  (`constants.py:3-41, 53`). The arena 17-D is `tire_normal_force_omega_pt`: the two extra fields
  are `engine_motor_speed_radps` and `engine_motorshaft_torque_nm` (`constants.py:45-48, 54`; worktree
  `crm_collect.py:196`, `traverse_wp3_chrono_eval.py:300`). `capture_row` does not emit them (no
  engine fields in `hmmwv_data.py`); arena collectors add them by hand (`crm_collect.py:222-223`,
  `traverse_wp3_chrono_eval.py:475-476`), so the original evaluators' `_capture_state_pose_np`
  (`hmmwv_chrono_tracking_env.py:570-588`; CRM `:59-91`) would KeyError on a 17-D checkpoint.
- 10 ms: `dt_s` = raw collector `simulation.record_step_s` (`preprocess.py:41-53`), 0.01 in
  `configs/hmmwv_overfit_v1.json` and `configs/hmmwv_crm_eval.json`; targets are one-step deltas
  (`preprocess.py:231`). Sources must share `dt_s` and fields (`trainer.py:213-224`) and references
  must match the checkpoint (`hmmwv_tracking_env.py:220-237`), so 10 ms and 50 ms caches cannot be
  mixed without resampling.
- Action-repeat 5: `action_repeat: 5` (`hmmwv_tracking_env.py:36`), `step_dt = 0.05` (`:107`), one
  action held for 5 NN substeps (`:640-642`). The deployed run used `dynamics_context_steps: 16`
  (run `env_cfg.json`). The raw collector drives with `ChDataDriver` sampled at
  `driver_sample_step_s = 0.01` (`hmmwv_data.py:268-276, 554-556`) and logs after
  `driver.Synchronize` (`:610-635`): each logged action is the driver value at that instant.

## (2) PPO tracker

- Observation 231-D: 10 x (15 state + 3 action) normalised history, 15-D normalised state error,
  body-frame pose error (dx/10, dy/10, yaw/pi), 10 preview poses 50 ms apart (dx/20, dy/20, yaw/pi),
  last action (`hmmwv_tracking_env.py:518-528, 802-852`).
- Action: `tanh` -> dataset-mean centre + scale [1, 0.7, 0.5] -> clamp [-1,0,0]..[1,1,1]
  (`:665-667`); steering rate limit 0.1 per policy step in training and eval (`:630-637`;
  `hmmwv_chrono_tracking_env.py:600-607`).
- Reward: `exp(-(w_pos (e_pos/2)^2 + w_yaw (e_yaw/0.35)^2 + w_state mean((de/std)^2)))` minus
  action-rate and throttle*brake penalties (`:716-751`); deployed weights pos 2.0, yaw 1.6, state 0.2
  over [vx, vy, yaw_rate], action rate 0.2 (run `env_cfg.json`).
- References are pose-vs-TIME: `ref_step_buf` advances per NN substep (`:685`) and the reward reads
  `reference_poses[ref, step]` (`:700-704`). Sets are 1100-step (11 s) recorded segments with states,
  actions, poses (`rl/references.py:153-262`); the 40-ref training set carries
  `metadata["domains"]` in {flat, crm} (verified in the npz); eval sets are rest-start. Terminations:
  pos error > 20 m, |roll| > 0.6, |pitch| > 0.4, reference end, 180 steps (`:753-768`).
- PPO: rsl_rl `OnPolicyRunner`, MLP [512,256,128], adaptive KL 0.01, 64 steps x 2048 envs,
  empirical obs normalisation (`train_hmmwv_rl_tracking.py:203-248`); `terrain_mix` "flat:1,crm:1"
  with per-terrain reference sampling (`hmmwv_tracking_env.py:406-599`).
- Documented Chrono result (progress.md:105-131; `artifacts/rl_runs/chrono_eval_comparisons/
  onehot_policy_3x3_*.json`): closed-loop XY RMSE vs the timed reference, 20 refs per cell; mixture
  0.157 med / 0.184 mean (rigid flat), 0.180 / 0.249 (CRM), 0.149 / 0.229 (bumpy zero-shot);
  rigid-only on CRM 0.854 / 1.000. THERE IS NO PID BASELINE in the original study (grep "PID" over
  `docs/progress.md`, `docs/*.md` is empty; `ChPathFollowerDriver` appears only in traverse scripts).
  It is policy-vs-policy on pose-vs-time error, not cross-track error.

## (3) Chrono evaluators

- Action application: `_set_driver_action_np` writes `m_steering/m_throttle/m_braking` once per
  policy step (`hmmwv_chrono_tracking_env.py:405-409, 615`) and holds it (zero-order hold) for
  `action_repeat x chrono_steps_per_nn_step` solver steps of `terrain.Synchronize`,
  `hmmwv.Synchronize(t, driver_inputs, terrain)`, `terrain.Advance`, `hmmwv.Advance` (`:411-420`).
  `chrono_steps_per_nn_step = round(dt_s / step_size_s)`, step re-derived as `dt_s / n`
  (`:198-205`): overfit_v1 (0.002) -> 5 per NN step, 25 per policy step; CRM config (0.0005) -> 20
  and 100. State captured every 10 ms into the 128-row history (`:616-623`).
- Initialisation: `_create_sim` takes x, y, yaw from reference index 0 and
  `fwd_vel_mps = max(0, vx_ref)` (`:360-379`) via `SetInitPosition`/`SetInitFwdVel`
  (`hmmwv_data.py:301-307`); z from config (1.6 rigid, 0.7 CRM). Warm start replays reference
  actions for `pre_roll_time_s` (default 6 s) plus the 128-step context (`:322-333, 547-568`); the
  reported evals used `pre0` with rest-start references (folder names; progress.md:120-122).
- CRM subclass (`hmmwv_chrono_crm_tracking_env.py`): `_create_terrain` -> `configure_crm_terrain`
  (`hmmwv_crm.py:82-160`); `_advance_sim_steps` calls `terrain.Advance` only (`:47-57`); tyre
  channels from FSI spindle force (`hmmwv_crm.py:163-202`). Scene = homogeneous FLAT box
  150 x 150 x 0.25 m, spacing 0.08, RIGID_MESH tyres, fresh soil per reference
  (`configs/hmmwv_crm_eval.json`). Soil numbers equal the worktree collector's (density 1700,
  cohesion 5e3, friction 0.8, E 1e6, nu 0.3, mu_I0 0.04, diam 0.005; `crm_collect.py:40-41`) but SPH
  settings differ: d0 1.0 vs 1.2, free-surface 2.0 vs 0.8, shifting NONE vs PPST(3.0/1.0), depth 0.25
  vs 0.24, active-domain delay 0.1 vs 0.0, MBS threads 12 vs 4 (`crm_collect.py:36-46`). The
  collector builds soil from the arena BMP with height range (`crm_collect.py:123-131`); the evaluator
  has no heightmap path. Neither couples the chassis to soil.
- Worktree arena evaluators: `traverse_wp3_chrono_eval.py` is rigid-only, 50 ms control, PID
  re-synchronised every solver substep with a 2.0 full-scale/s steering clamp, action logged at
  `sub == 0` (`:457-478`), with a `manual` DriverInputs branch for neural policies (`:468-470`).
  `crm_collect.py` has the CRM arena scene but only the PID follower plus a stop-policy object
  (`:156-245`); no external-action branch.

## (4) Reusable vs obsolete for plan B

Reusable: the per-token one-hot and its metadata plumbing (`model.py:56-133`, `dynamics.py:65-72`);
the mixed-loader pattern (`allocate_batch_sizes`, tagged sub-batches); domain-rebalanced channel
weights and domain-balanced `rollout_sel`; per-env terrain allocation and per-terrain reference
sampling; the evaluator skeleton (`_create_terrain/_advance_sim_steps/_capture_state_pose_np`
overrides, CRM-owns-the-step rule, ZOH action application, rsl_rl policy loading, per-reference
summary JSON).

Obsolete: flat 240 m rigid patch and flat CRM box (`create_rigid_terrain` supports
`rigid_heightmap` but not the arena `scene.build_config` path); timed pose references and
pose-vs-time reward (`tracker_env.py:1-27` already ports the reward lessons to routes); 10 ms dt with
action-repeat 5 (arena is 50 ms, repeat 1, `nrd_model.DT_S = 0.05`); 128-token context; reference
warm start and `fwd_vel_mps` init (arena starts from rest after 0.8 s settle); the 231-D full-state
observation; `ChDataDriver` profiles; `WindowedHMMWVDataset` (contiguous 10 ms arrays vs the arena's
padded `(N, T, 17)` cache with `valid` masks, `nrd_data.py:39-61`).

## Gaps / unknowns

- The processed caches named in the onehot config are absent locally; the checkpoint
  `ablation_ofat/L8_H8_E256_ctx128/checkpoints/best_val.pt` (77 MB, metadata embedded) is present.
- `ChDataDriver` linear interpolation between 10 ms samples is assumed from Chrono, not verified here.
- Meaning of `min10` in the CRM eval folder name; the rsl_rl version pinned for `OnPolicyRunner`.
- Whether the worktree `hmmwv_crm.py` ever ran on the cluster (predicted failure at
  `SetActiveDomainDelay`).
- The arena NRD stack (`nrd_model.py`, `nrd_data.py`, `tracker_env.py`) has no domain field (grep for
  domain/terrain_id/crm is empty); CRM night-2 models were presumably per-domain, not tagged mixtures.

## What must change for the plan

1. Port the domain tag into `nedm.traverse.nrd_model`/`nrd_data` (a per-episode `domain` column
   beside `arena_idx`, one-hot concatenated to each 50 ms token) instead of reusing
   `HMMWVDynamicsModel`, whose dt/field validation rejects the 17-D 50 ms cache.
2. Plan A.5 / B.5 "no label at deployment" needs a soft or masked tag input; `_terrain_one_hot`
   accepts only integer ids and raises on `None`.
3. Do not cite the original study as evidence that PPO beats PID: it never ran a PID baseline and
   scores pose-vs-time XY RMSE. Build the PID comparison in the arena evaluator (follower vs manual
   branch of `traverse_wp3_chrono_eval.py`) and extend to CRM via an external-action branch in
   `crm_collect.py` or a heightmap `_create_terrain` in the CRM env.
4. Timing audit (B.2): the plan's statement is confirmed. The original evaluators' 50 ms ZOH already
   matches the target controller, but the original 10 ms caches sample an interpolated driver and
   cannot be turned into ZOH-50 ms transitions by striding.
5. Take the CRM scene from worktree `crm_collect.py:69-133` (cluster-proven shims, BMP soil) and
   align SPH settings to the collection recipe, not to `hmmwv_crm_eval.json`, so evaluation soil
   equals training soil.
