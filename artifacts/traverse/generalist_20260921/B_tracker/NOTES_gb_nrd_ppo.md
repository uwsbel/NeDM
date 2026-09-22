# Module notes: mixed-domain dynamics model trainer and PPO tracker (PLAN B4, B5)

Written 2026-09-21 (resumed after the usage-limit interruption; `gb_nrd_common.py` and `gb_train_nrd.py` existed as
partial files and were kept, fixed and tested; `gb_tracker_env.py` and `gb_train_tracker.py` are new).
Files: `scripts/gb_nrd_common.py`, `scripts/gb_train_nrd.py`, `scripts/gb_tracker_env.py`, `scripts/gb_train_tracker.py`.
Self-test artefacts: `artifacts/traverse/generalist_20260921/B_tracker/selftest/{nrd,tracker}/`.
Python: `/home/harry/miniconda3/envs/nedm/bin/python` with `PYTHONPATH=src:scripts` from the repo root.

## 1. What was built

### gb_nrd_common.py (shared model, cache loader, checkpoint format, synthetic cache)
* `GBNRDModel`: the WP2 map model's causal transformer (`nedm.training.model_transformer.ContinuousTransformer`, block
  16, default 6 layers / 8 heads / 256 embd, 4.9 M params) over per-frame tokens `[z1 (17, z-scored), crop token (64),
  action (3, z-scored), domain one-hot (2, only when cond='tag')]`. The crop token is `CropTokenizer`: `gb_crop.EgoCrop`
  on the Chrono-frame v2 grid at the pose (k x k relative heights / 2 m plus the validity flag) -> MLP(k*k+1 -> 128
  -> 64). Heads: delta of the normalised z1 (17) and normalised power (1). Pose integration is
  `nrd_model.integrate_pose` (yaw first, then the world velocity with the new yaw, 50 ms), never inside the network.
* Cache reader for the schema-3 contract (`read_manifest`, `select_keys`, `load_cache` -> last-row-padded arrays with a
  `valid` mask, `heldout_groups`, `check_group_split_consistency`); `fit_normalizer` (per-channel z-score on the
  recorded frames of the train split, both domains pooled).
* Checkpoint: `save_nrd` / `load_nrd(path, device, grid_path=None)`. Payload = the WP2 keys (`model`, `config`,
  `normalization`, `step`, `metrics`, `z1_dim`, `delta_scale`) plus `model_kind='gb_nrd'`, `crop_k`, `crop_half_m`,
  `cond`, `token_dim`, `grid_path`, `grid_sha256`, `domain_vocab` (`['rigid', 'crm']` = cache domain 0 / 1), and in
  `ckpt_last.pt` only `train_state` (optimizer, schedule, step, numpy/torch/cuda RNG, best, elapsed). `load_nrd`
  rebuilds the crop from the stored grid path (or an override) and refuses a grid whose sha256 differs.
* `make_synthetic_cache`: a schema-3 mini-cache for self-tests (both domains, straight 50 m routes at 0.5 m spacing,
  first-order speed response with a weaker/draggier deformable-soil variant, half the episodes stall for 100 frames,
  one 1 s brake tap per episode, `stalled`/`hold_ok` by the contract definitions, group split with 2 val + 1 test group).

### gb_train_nrd.py (fork of traverse_wp2_train_map.py)
* Data: one schema-3 cache; train = manifest `split_of == 'train'`, val = `'val'`, test never loaded; asserts no
  held-out group in the training set. Every batch draws `--domain-frac` (0.5) of its windows from the deformable-soil
  domain and the rest from rigid, windows uniform over the recorded frames within a domain.
* Channel weights (`--delta-scale`): `w_i = mean_j(s_j) / s_i` with `s_i^2` = the mean over domains of the per-domain
  variance of the normalised one-step delta, mean-normalised. This is the original mixed-domain trainer's
  `equal_domain_combined_std` rule (`/home/harry/NeDM/src/nedm/training/trainer.py`, `_build_channel_weights`:
  `w_i = ref_std_i^2 / mean_d std_{d,i}^2` applied as `sqrt(w)` on the residual before the Huber loss) written in this
  trainer's residual-scaling form, so no channel's loss is dominated by the domain with the larger delta variance.
* `hold_ok` weighting: each one-step target `z1[k] -> z1[k+1]` under `act[k]` weighs 1 when `hold_ok[k]` and
  `--hold-bad-weight` (0.25) otherwise, in the teacher-forced step loss and per step of the rollout loss.
* Rollout loss (`--rollout-steps` 8): state fed back, crop re-taken at the dead-reckoned pose every step (the
  imagination env's step), recorded actions.
* Resume: `ckpt_last.pt` is written every `--ckpt-every-min` (15) minutes and at every evaluation; `--resume` restores
  model, optimizer, step, RNG and continues the same cosine schedule (asserts `--lr/--min-lr/--warmup-steps/--steps`
  and the grid sha256 match). `--max-minutes` / `--stop-at-step` stop early after saving (4 h cap).
* Validation on the val groups (`evaluate`), per domain and pooled, teacher-forced and fed-back, horizon `--val-steps`
  60 frames, up to `--val-max-per-domain` windows per kind: random windows; stalled windows (all 60 predicted frames
  inside a run of cache `stalled` frames); moving windows (no stalled frame, mean |vx| > 0.5 m/s); brake-onset windows
  (first rollout action = the first frame with brake > 0.2 after >= 10 frames below 0.2). Per cell: signed and absolute
  along-track displacement error, relative displacement error, escape fraction (|predicted displacement| > 1 m), vx and
  yaw-rate response error (change over the window, relative), vx / yaw-rate trajectory MAE, normalised z1 MAE and pose /
  yaw error at 60 frames. Gate (PLAN B4, fed-back): `stalled_escape_frac < 0.20`, `moving_disp_rel_err < 0.25`,
  `brake_vx_resp_rel_err < 0.25`; printed as `gate[domain] pass=...` and written to `metrics.json` with `gate/pass/<dom>`.
  Model selection (`ckpt_best.pt`) = mean over domains of the fed-back random-window normalised z1 MAE at 60 frames (`sel`).
* Outputs: `config.json`, `train_log.jsonl`, `metrics.json` (last evaluation incl. `gpu_peak_gib`, `train_sps`),
  `ckpt_best.pt`, `ckpt_last.pt`, `run_state.json`.

### gb_tracker_env.py (fork of nedm/traverse/tracker_env.py)
* One static grid: the crop reads the grid of the NRD checkpoint (sha256-checked); no per-env map bank.
* Fragment bank (`FragmentBank`): the cache's train groups only (asserts no val/test group), both domains, flat on the
  GPU with per-episode offsets (the full train split, 17.8 M frames, is ~1.6 GB flat instead of ~4.4 GB padded); routes
  from the cache files; `active_end` = last frame > 3 m of arc from the route end (WP3 rule). `domain_frac` (0.5) of
  the resets go to deformable soil (`None` = the natural mix).
* Decision-frame convention (= `gc_control.PolicyObs`): a fragment starts at a decision frame s drawn uniformly from
  [0, active_end - len]; the policy sees state[s], pose[s] and the last held action act[s-1] (settle action (0, 0, 1)
  at s = 0) and its output becomes act[s]. The NRD context is frames s-15..s; frames before 0 are padded with frame 0's
  state and pose and the settle action. The 8-frame observation history is padded the same way.
* Observation 158-D = `PolicyObs` layout: [0:38] the WP3 block (errors to the nearest waypoint in the -2..+40 window,
  10 preview points at 1 m, vx/10, yaw rate, last action), [38:62] the last 8 held actions (oldest first),
  [62:158] the last 8 observable states (columns 0-6, 11-14, 15; oldest first; newest = current state; raw physical
  values). `check_against_policy_obs` rebuilds the observation with `gc_control.PolicyObs` from the same recording.
* Action squash: steer centre 0 scale 1, throttle/brake centre 0.5 scale 0.5, clamp [-1,0,0]..[1,1,1]; steering-rate
  clamp 0.1 per step vs the last action (also during training). `physical_to_policy(a, clip=0.99)` is the inverse.
* Reward: WP3 terms (exp(-(2 (e_ct/1)^2 + 0.8 (e_h/0.35)^2 + 0.5 (e_v/1)^2)) - 0.2 |da|^2 - 0.05 throttle*brake) plus
  0.5 x the along-track advance per step in metres (continuous station = nearest-waypoint station + e_along, per-step
  difference clipped to +-2 m). Terminations as WP3 (|e_ct| > 6 m, |roll| > 0.6, |pitch| > 0.4, non-finite);
  fragment end / route end bootstrap.
* Domain: the fragment's cache domain is passed to the NRD every step when the checkpoint is `cond='tag'`.
* `sample_imitation(n_resets)` returns (observation at a random decision frame, the recorded PID action at that frame).
* `policy_meta()` returns the obs layout, squash, NRD checkpoint sha256, cache manifest sha256, bank counts, env cfg.

### gb_train_tracker.py (fork of traverse_wp3_train_tracker.py)
* rsl_rl `OnPolicyRunner` + PPO with the WP3 block (2048 envs x 64 steps, 5 epochs x 8 minibatches, lr 3e-4 adaptive
  KL 0.01, entropy 3e-3, [512, 256, 128] ELU, init noise 0.7, empirical observation normalisation).
* PID-imitation warm start: `--imitation-samples` (131,072) observations from `sample_imitation`; the runner's own
  `obs_normalizer` and `critic_obs_normalizer` are fitted on them FIRST, then frozen for the fit; `--imitation-epochs` (2)
  epochs of MSE between the actor's pre-tanh output and the inverse-squashed PID actions (`--imitation-clip` 0.99 pulls the
  box edges inside: throttle 0 -> atanh(-0.99) = -2.65). The actor-vs-PID error (pre-tanh MSE, physical-action MAE per
  channel, per domain) is logged before, after the warm start (= PPO iteration 0) and after the iterations in `--mse-at`
  (default 10) into `imitation.json`.
* Exports: the runner's `save` is wrapped so that every `model_<it>.pt` (every `--save-interval` iterations and at the end
  of each learning segment) also writes `actor.npz` (`gc_control.export_torch_actor`: obs normaliser, MLP, squash) and
  `policy_meta.json` (obs layout, squash, NRD sha256, cache manifest sha256, iteration, completed iterations, imitation
  numbers, numpy check); `NumpyActor(actor.npz).act` is checked against the torch actor on 100 live observations and the
  run aborts if max |da| >= 1e-5. `model_init.pt` + export are written right after the warm start.
* `--resume model_<it>.pt` restores actor-critic, normalisers, optimizer and the completed-iteration count (from the
  checkpoint's `infos` for `model_init.pt`, from rsl_rl's `iter` + 1 for its own saves); no warm start on resume.
* `--smoke`: env-vs-PolicyObs check, scripted pure pursuit and a random policy (throughput, finiteness).

## 2. How to run (real data)

Cache: `artifacts/traverse/generalist_20260921/B_tracker/cache_v1` (schema 3, 39,235 episodes: train 35,601 / val 1,833 /
test 1,801; rigid 24,000, crm 15,235). Grid: `artifacts/traverse/crm_f104_v1/grids/arena_f104_50h_v1/grid.npz`.

NRD, tag arm (the PPO model), on the cluster (mi3501x, 4 h cap; resume in a second job if needed):
```
PYTHONPATH=src:scripts python scripts/gb_train_nrd.py \
  --cache artifacts/traverse/generalist_20260921/B_tracker/cache_v1 \
  --out artifacts/traverse/generalist_20260921/B_tracker/nrd_tag --cond tag --crop-k 8 --crop-half-m 6 \
  --steps 30000 --batch 256 --rollout-steps 8 --delta-scale --eval-every 2000 --val-max-per-domain 256 \
  --ckpt-every-min 15 --max-minutes 225
# continuation (same model/schedule arguments):
PYTHONPATH=src:scripts python scripts/gb_train_nrd.py ... (identical arguments) ... \
  --resume artifacts/traverse/generalist_20260921/B_tracker/nrd_tag/ckpt_last.pt
# notag arm: --cond notag --out .../nrd_notag ; 16x16 crop variant (fallback rule): --crop-k 16 --crop-half-m 4
```
`--steps` must be sized from the measured throughput (section 3) so that one 4 h job (or two with `--resume`) finishes
the schedule; `--max-minutes 225` leaves 15 min for the final evaluation and checkpoint.

PPO (local RTX 5090 or mi3501x; no Chrono, no CRM lock needed):
```
PYTHONPATH=src:scripts python scripts/gb_train_tracker.py \
  --out artifacts/traverse/generalist_20260921/B_tracker/ppo_v1 \
  --nrd artifacts/traverse/generalist_20260921/B_tracker/nrd_tag/ckpt_best.pt \
  --cache artifacts/traverse/generalist_20260921/B_tracker/cache_v1 \
  --num-envs 2048 --max-iterations 1000 --save-interval 100 --imitation-samples 131072 --seed 1
# resume: same arguments + --resume artifacts/.../ppo_v1/model_<it>.pt
# env sanity first (bank load, PolicyObs agreement on real routes, pure pursuit): add --smoke --smoke-steps 200
```
The collectors consume `ppo_v1/actor.npz` (`gc_control.NumpyActor.from_npz`, `PolicyObs.from_meta(route, actor.meta)`,
`hold_clip`); `policy_meta.json` holds the same meta (assert `action_center/scale/low/high` and `obs_layout` equal across
arms).

## 3. Self-tests (all local, RTX 5090, GPU peak <= 3.8 GiB; the real cache existed, so a real-data timing run was added)

All commands from the repo root with `PYTHONPATH=src:scripts` and `P=/home/harry/miniconda3/envs/nedm/bin/python`;
`S=artifacts/traverse/generalist_20260921/B_tracker/selftest`.

1. Model round trip + synthetic cache
   `$P scripts/gb_nrd_common.py --make-synthetic-cache $S/nrd/cache --selftest`
   40 episodes x 300 frames (20 rigid / 20 crm, 10 groups: 7 train / 2 val / 1 test = 28 / 8 / 4 episodes, 20 episodes
   with a stall, hold_ok 0.987). Model shapes tag/notag: token (5,16,64), delta (5,16,17), power (5,16,1); flipping the
   domain tag changes the output (True); save/load round trip max |diff| 0.0 for both arms.
2. NRD 200 steps (`$S/nrd/run200`)
   `$P scripts/gb_train_nrd.py --cache $S/nrd/cache --max-episodes 40 --steps 200 --eval-every 100 --n-layer 2 --n-embd 64 --n-head 4 --batch 64 --warmup-steps 20 --val-max-per-domain 64 --val-batches 4 --delta-scale --rollout-steps 8 --out $S/nrd/run200`
   loss 1.377 (step 1) -> 0.114 (100) -> 0.081 (200); z1 step loss 0.061 -> 0.018, rollout z1 0.523 -> 0.028;
   16 s wall; metrics.json has 481 numeric entries, none non-finite; gate cells filled for both domains (stalled n=8,
   moving n=61-63, brake n=4 per domain; the toy model fails the stalled and brake gates, as expected for 200 steps
   on synthetic data: stalled_escape 0.875 / 0.500, moving_disp_rel_err 0.020 / 0.024, brake_vx_resp 0.338 / 0.337).
3. Stop / resume (`$S/nrd/run_stop100`)
   same arguments with `--stop-at-step 100`, then the same arguments plus `--resume $S/nrd/run_stop100/ckpt_last.pt`.
   run_state: stop_at_step 100 -> resumed at step 100 -> finished 200/200; every metric of the resumed run equals the
   straight run's (max |diff| over the 481 entries = 0 except `wall_s`); losses at steps 100/200 identical (0.1138 / 0.0810).
   A bug found on the first attempt and fixed: the power-head target was sliced over the whole context+K window
   instead of the context frames (shape 23 vs 16); also the pooled validation cell now merges the per-domain rollout
   arrays instead of rolling out a second time.
4. Full-size NRD timing on the real cache (`$S/nrd/timing_real`)
   `$P scripts/gb_train_nrd.py --cache artifacts/traverse/generalist_20260921/B_tracker/cache_v1 --max-episodes 3000 --steps 300 --eval-every 300 --batch 256 --rollout-steps 8 --delta-scale --warmup-steps 10 --val-max-per-domain 256 --val-batches 2 --out $S/nrd/timing_real`
   3,000 train episodes (1,816 rigid / 1,184 crm) loaded in 4.9 s; 4.90 M params; 11.6 optimizer steps/s at batch 256
   with the 8-step rollout loss (2,969 windows/s, includes the final evaluation in the timer), GPU peak 3.77 GiB.
   30,000 steps therefore take ~45 min on the 5090; a 4 h mi3501x job has ample room (the cluster throughput is not
   measured). Note: the real val split offers only 53 deformable-soil brake-onset windows under the 60-frame horizon
   (`val/crm/brake/*/n` = 53 with `--val-max-per-domain 256`), which is why B3's hold-mode brake data is needed.
5. Tracker env smoke (`$S/tracker/smoke`, synthetic NRD `run200/ckpt_best.pt`, 64 envs)
   `$P scripts/gb_train_tracker.py --smoke --out $S/tracker/smoke --nrd $S/nrd/run200/ckpt_best.pt --cache $S/nrd/cache --num-envs 64 --smoke-steps 60`
   env observation vs `gc_control.PolicyObs` on 16 envs: max |diff| 1.6e-7 (base block, float32 vs float64), past
   actions and past states exactly 0; pure pursuit: cross-track 0.032 m, reward 0.648 finite, 0 failures; random
   policy: reward -0.031 finite, 8-13k env steps/s at 64 envs.
6. PPO 3 iterations with the warm start (`$S/tracker/ppo3`)
   `$P scripts/gb_train_tracker.py --out $S/tracker/ppo3 --nrd $S/nrd/run200/ckpt_best.pt --cache $S/nrd/cache --num-envs 64 --max-iterations 3 --num-steps-per-env 24 --num-mini-batches 2 --imitation-samples 2048 --mse-at 2 --save-interval 1 --logger none`
   imitation on 2,048 samples (977 rigid / 1,071 crm): pre-tanh MSE 2.291 -> 1.608 after 2 epochs, physical MAE per
   channel [0.050, 0.238, 0.416] -> [0.057, 0.154, 0.265]; after 2 PPO iterations 1.299 / [0.063, 0.083, 0.203]
   (the smoke logs at iteration 2 instead of 10 because it runs 3 iterations; the default `--mse-at 10` is used for
   real runs). actor.npz + policy_meta.json exported at every save (model_init, model_0..2); NumpyActor vs torch on 64
   live observations: max |da| 6.5e-8 .. 1.2e-7 (< 1e-5); rewards finite; run_state 3/3 iterations.
7. PPO resume (`$S/tracker/ppo3_resumed`)
   same env arguments with `--max-iterations 5 --resume $S/tracker/ppo3/model_2.pt`: "3 iterations completed,
   continuing at iteration 3", iterations 3 and 4 run, run_state 5/5, exports at iterations 3 and 4 (max |da| 9.4e-8).
   A bug found on the first attempt and fixed: the final explicit save happened after the iteration counter was bumped,
   so the resumed run skipped an iteration; the checkpoint now carries `completed_iterations` and the redundant save is gone.
8. Real-cache bank at 2,048 envs with an untrained full-size NRD (`$S/tracker/real_bank`)
   `$P scripts/gb_train_tracker.py --smoke --out $S/tracker/real_bank --nrd $S/nrd/fullsize_untrained.pt --cache artifacts/traverse/generalist_20260921/B_tracker/cache_v1 --max-bank-episodes 5000 --num-envs 2048 --smoke-steps 100`
   5,000 train episodes (3,126 rigid / 1,874 crm, 2.51 M frames) loaded in 10.0 s (=> ~70 s for the full 35,601), GPU
   peak 0.37 GiB; PolicyObs agreement on real routes 1.1e-7; 99k-110k env steps/s with the 4.9 M-param NRD
   (1000 PPO iterations x 64 x 2048 = 131 M steps => ~25 min of collection on the 5090). The tracking numbers of this
   run are meaningless (untrained NRD).

## 4. Known limits and notes for the reviewer

* The NRD gate numbers above are from a toy model on synthetic data; the real B4 gate needs the real cache plus the B3
  hold-mode brake-onset data (the current val split has 53 deformable-soil brake onsets under the window definition).
* Validation window definitions are mine where the plan left them open: 'stalled' = all 60 predicted frames inside a
  cache `stalled` run (context may start up to 16 frames before it); 'moving' = no stalled frame in the 76-frame window
  and mean |vx| > 0.5 m/s over the predicted frames; 'brake onset' = brake > 0.2 after >= 10 frames < 0.2, the rollout's
  first action at that frame; up to 256 windows per kind and domain (stride 10 frames for stalled/moving).
* Domain balance in PPO: `domain_frac 0.5` of the resets go to deformable soil (the plan does not specify; the natural
  mix is 61/39 rigid/crm; `--domain-frac -1` restores it).
* The NRD has never seen the frame-0-padded context (training windows lie inside recorded frames); fragments starting at
  s < 15 feed it the padded context exactly as the collectors' policy mode will at frame 0, so the env and deployment
  agree, but the model's behaviour there is unvalidated.
* Iteration bookkeeping with rsl_rl: `policy_meta.json` carries both `iteration` (rsl_rl's index of the last update)
  and `completed_iterations`; rsl_rl saves `model_<it>.pt` at `it % save_interval == 0` and at the end of every
  learning segment, so a run with `--mse-at 10` also leaves `model_9.pt`.
* Throughput on the AMD cluster is not measured; 11.6 steps/s (5090, batch 256, K = 8) is the only number.
* `gc_control.py` was complete when read (PolicyObs, NumpyActor, export_torch_actor); the env imports it directly, so
  the observation layout has one definition.


## 5. Fix round 1 (2026-09-21, after `VERIFY_gb_nrd_ppo.md`)

All seven verifier issues are fixed with the smallest change that closes each; no numeric path changed, and every
self-test of section 3 that runs on the synthetic mini-cache was re-run into the same directories (numbers below).
Nothing under `crm_night2_v1`, `crm_f104_v1` or `fdm_f104_50h_20260909` was touched; no cluster job; GPU peak of the
re-runs 0.16 GiB (NRD) / 0.04 GiB (tracker).

### What changed, per issue

1. (blocking) The documented PPO command crashed at the `model_init.pt` save with the default logger because rsl_rl sets
   `logger_type` only inside `learn()` and its `save()` reads it. Fix: `scripts/gb_train_tracker.py:319` sets
   `runner.logger_type` from `--logger` right after the runner is built (unconditionally; `learn()` re-derives it from the
   config). The `none`/`off` branch keeps its no-op writer. Limit: the pre-training export still needs a writer for
   `wandb`/`neptune` (not used here); `tensorboard` and `none` are covered.
2. (major) `--resume` accepted any change of the data / model / loss / optimiser arguments. Fix:
   `scripts/gb_train_nrd.py:62` lists the fixed arguments (`RESUME_FIXED`: cache, cond, crop_k, crop_half_m, domains,
   domain_frac, max_frames, max_episodes, context, batch, n_layer, n_head, n_embd, dropout, token_dim, rollout_steps,
   rollout_weight, progress_weight, context_noise, delta_scale, vx_weight, hold_bad_weight, weight_decay, grad_clip,
   seed; the verifier's list plus weight decay and gradient clip, which change the optimiser too);
   `scripts/gb_train_nrd.py:476-484` compares them with the checkpoint's `train_args` and exits on any
   difference (the message lists every differing argument with both values), and also refuses a cache whose manifest
   sha256 differs from the checkpoint's. The schedule and the grid were already checked. `config.json` of a resumed run
   is now written from the checkpoint's `train_args` plus `resumed_from`, with only the run-control arguments (`--out`,
   `--resume`, `--grid`, `--stop-at-step`, `--max-minutes`, `--ckpt-every-min`, `--eval-every`, `--val-*`,
   `--twin-split`) taken from the new command line (`scripts/gb_train_nrd.py:548-550`); later checkpoints
   carry the same merged record (`:570`).
3. (minor) The B4 gate could report `pass` with a missing cell. Fix: `scripts/gb_train_nrd.py:371`, `gate/pass/<dom>`
   is true only when all 3 cells exist and pass (`gate/n_cells/<dom>` still reports how many exist).
4. (minor) A `cond='notag'` NRD was accepted for PPO. Fix: `scripts/gb_train_tracker.py:305` exits after the env is
   built unless `--allow-notag` (`:64`) is given; the check applies to `--smoke` too.
5. (minor) Only the last export survived. Fix: `scripts/gb_train_tracker.py:182-184` also writes
   `actor_<it>.npz` and `policy_meta_<it>.json` next to every `model_<it>.pt` (`actor_init.npz` for the warm-start
   export); `actor.npz` / `policy_meta.json` remain the latest, and `run_state.json` names the per-checkpoint file.
6. (minor) `--max-minutes` started after data loading. Fix: `scripts/gb_train_nrd.py:435` takes the clock at the
   start of `main()` and the stop check (`:582`) uses it, so the budget covers the cache load, the normaliser fit and
   the channel weights; `run_state.json` records `load_s` and `process_wall_s` (`:638`). The throughput numbers
   (`train_sps`, `wall_s`) still count from the first optimizer step. With this, `--max-minutes 225` in a 240 min job leaves
   15 min for the final evaluation and the last checkpoint only; the recipe in section 2 keeps 225.
7. (minor, hardening) The readers trusted the manifest's split. Fix: `scripts/gb_nrd_common.py:104-115`
   (`twin_split_of_groups`, `check_split_against_twin`) read the group -> split map of the twin dataset
   (`crm_night2_v1/datasets/twin_crm.npz`, the plan's single source; `DEFAULT_TWIN_SPLIT` at `:60`) and raise when
   any cache group is unknown to it or carries a different split. The NRD trainer calls it after the group-consistency
   check (`scripts/gb_train_nrd.py:443`, result stored in `config.json` as `split_cross_check`) and so does the
   fragment bank (`scripts/gb_tracker_env.py:130`, result in `policy_meta.json` as `split_cross_check`). The file
   to use is the one the manifest's build report names, falling back to the repo-relative default on another machine;
   `--twin-split PATH` (trainer `:432`, tracker `:63`, env cfg key `twin_split` `:76`) overrides it
   and must exist, `--twin-split none` skips with a printed warning; a manifest flagged `synthetic` (the self-test
   cache) is skipped with a printed note. Reading the two arrays takes 0.01 s.

Also in this round (from `VERIFY_gb_crop.md` revision 2, consumer side): `scripts/gb_nrd_common.py:267` casts the
crop features to the token MLP's weight dtype, so a half-converted model works (the crop itself stays float32); under
autocast this is a no-op.

### Tests re-run (synthetic mini-cache, RTX 5090 shared with a foreign training job at 100 % utilisation)

1. NRD 200 steps (`$S/nrd/run200`, section 3 command): loss 1.3770 (step 1) -> 0.1138 (100) -> 0.0810 (200); sel 0.9350
   (100) -> 0.6519 (200), val_loss 0.0344; gate cells rigid/crm/all filled (stalled n=8, moving n=63/61, brake n=4);
   stalled_escape 0.875 / 0.500, moving_disp_rel_err 0.020 / 0.024, brake_vx_resp 0.338 / 0.337; `metrics.json` 483
   numeric entries, none non-finite; every entry equal to the pre-fix `metrics.json` (max |diff| 0 over 485 compared
   leaves incl. `best`, `wall_s` excluded; the two extra keys `gpu_peak_gib` 0.158 and `train_sps` were absent from
   the older file). Wall 0.3 min.
2. Stop at 100 + resume (`$S/nrd/run_stop100`): stop_at_step 100 -> "resumed ... at step 100" -> done 200/200; every
   metric equal to the straight run (max |diff| 0 over 486 numeric entries, `wall_s`/`train_sps` excluded); train losses
   at 100/200 identical (0.1138 / 0.0810); `config.json` carries `resumed_from` and `stop_at_step 0` (the new command
   line's run-control value), `ckpt_last.pt` `train_args.resumed_from` set.
   Negative checks (issue 2): resuming with `--batch 16 --rollout-steps 2 --hold-bad-weight 1.0 --cond notag --crop-k 16
   --n-layer 5` exits 1 listing all six differences; resuming without `--delta-scale` exits 1 (`delta_scale true vs
   false`); resuming with a different `--out` and `--val-batches 2` is accepted (exit 0).
3. Gate completeness (issue 3, scratch on `run200/ckpt_best.pt` with every threshold raised so all present cells pass):
   all three cells -> `gate/pass` true for rigid/crm/all with `n_cells` 3; brake windows removed -> `gate/pass` false
   for all three domains with `n_cells` 2.
4. `--max-minutes 0.01` (issue 6): stops at step 0 with `stop_reason max_minutes`, `load_s` 1.02 s, `process_wall_s`
   1.02 s (the old clock would have run 200 steps).
5. Tracker env smoke (`$S/tracker/smoke`, 64 envs, section 3 command): PolicyObs agreement 1.6e-7 base / 0 / 0; pure
   pursuit cross-track 0.0317 m, reward 0.6476, 0 failures; random reward -0.0308; all identical to the pre-fix
   `smoke.json`. Throughput 0.7k env steps/s at 64 envs under the foreign job's contention (8-13k idle in section 3;
   the step path is unchanged, only the bank load gained the 0.01 s split cross-check).
6. PPO 3 iterations WITH the warm start and the DEFAULT logger (`$S/tracker/ppo3`; the section 3 command without
   `--logger none`, which is exactly the crashing configuration of issue 1): runs to completion (exit 0), a tensorboard
   events file is written, `model_init.pt` + export happen right after the warm start; imitation on 2,048 samples
   (977 rigid / 1,071 crm) pre-tanh MSE 2.2911 -> 1.6077, physical MAE [0.0495, 0.2378, 0.4159] -> [0.0571, 0.1540, 0.2653];
   after 2 PPO iterations 1.2992 / [0.0631, 0.0833, 0.2026]; all identical to the pre-fix run. Exports at init, 0, 1, 2
   with numpy-vs-torch max |da| 6.98e-8, 1.15e-7, 6.52e-8, 9.29e-8 (< 1e-5). Issue 5: `actor_init/0/1/2.npz` and
   `policy_meta_init/0/1/2.json` present; `actor_2.npz` is byte-identical to `actor.npz` and `policy_meta_2.json`
   equals `policy_meta.json`.
7. PPO resume from `model_2.pt` (`$S/tracker/ppo3_resumed`, `--max-iterations 5`): "3 iterations completed, continuing
   at iteration 3"; iterations 3 and 4 run; run_state 5/5; exports `actor_3.npz`, `actor_4.npz` (max |da| 8.15e-8,
   9.40e-8).
8. actor.npz equality with `gc_control.NumpyActor`, independent of the runner's own check: `NumpyActor(actor_2.npz)` vs
   the torch actor loaded from `model_2.pt` on 64 live observations from a freshly built env (seed 7, after 5 random
   steps): max |da| 6.4e-8, pre-squash 1.7e-7.
9. Issue 4: a notag NRD (50 toy steps, `/tmp`) is refused by the tracker with exit 1 and the message naming
   `--allow-notag`; with the flag the env builds (`NRD cond=notag`).
10. Issue 7 on the real cache (readers only, `/tmp` outputs): the tracker bank (200 episodes) and the NRD trainer
   (40 episodes, 2 steps, tiny model) both print "1,200 cache groups agree with the twin split ... train 1089 / val 56 /
   test 55" and record it; PolicyObs agreement on real routes 1.8e-7. Scratch negatives: a manifest with one WHOLE val
   group (`f104_v2_group_0033`, 33 episodes) relabelled `train` passes the old group-consistency check and is now
   refused ("1 groups with a different split"); a group unknown to the twin split is refused; an explicit
   `--twin-split` path that does not exist raises instead of falling back; `--twin-split none` prints the skip; the
   synthetic cache prints "skipped: synthetic cache". `twin_rigid.npz` and `twin_crm.npz` carry the same 1,200-group map.

The section 2 recipes stand; the PPO one already passes no `--logger`, which now works. For the NRD recipe, note that a
continuation job must repeat every data / model / loss argument exactly (the trainer now enforces it) and that
`--max-minutes` includes the ~2-4 min of loading at full size.


## 6. Resumed session (2026-09-21, after the second usage-limit interruption)

State found: all four scripts complete with fix round 1 applied (section 5); the verifier's re-check of that round was
still pending. The four files compile; nothing outside them was changed. This session (a) closed one gap against the
brief, (b) re-ran every self-test, (c) diagnosed why the first real NRD job on the cluster died.

### 6a. Change: the numpy-vs-torch actor check now uses 100 real observations regardless of the env size

The export check compared `NumpyActor(actor.npz).act` with the torch actor on `env.obs_buf[:100]`, i.e. on only
`num_envs` observations when the run uses fewer than 100 envs (64 in the self-tests). Now (`scripts/gb_train_tracker.py`):
* `main()` draws a fixed pool of `--check-n` (100) real observations at random decision frames of the bank right after
  the env is built (`env.sample_imitation`), before the runner exists; every export checks the pool PLUS the live
  observation buffer (`numpy_actor_check(runner, env, npz, pool, n)`, rows capped at 4096) and still aborts the run when
  max |da| >= 1e-5. `policy_meta.json` -> `numpy_actor_check.n` reports the count (164 = 100 + 64 in the smokes).
* `--check-actor model_<it>.pt` (with the usual `--nrd/--cache/--num-envs`) loads that checkpoint into the runner,
  compares its `actor_<it>.npz` (the artefact the collectors load, sha256 recorded) against the torch actor on the pool
  plus the live buffer, writes `actor_check_<it>.json` to `--out`, and exits 0/1. Nothing is re-exported.
* Side effect: the pool draw advances the env's random stream before the imitation sampling, so the warm-start numbers
  of the smoke moved in the 4th digit versus section 5 (2.2905 vs 2.2911 pre-tanh MSE before the fit); nothing else
  in the training path changed.

### 6b. Self-tests re-run (RTX 5090 shared with two foreign risk-model trainings at 100 % utilisation / 13 GB; my
peak 0.16 GiB NRD toy, 0.04 GiB tracker, 3.77 GiB full-size NRD dry run)

Same commands as section 3 unless stated (`P=/home/harry/miniconda3/envs/nedm/bin/python`, `PYTHONPATH=src:scripts`,
`S=artifacts/traverse/generalist_20260921/B_tracker/selftest`).

1. `$P scripts/gb_nrd_common.py --make-synthetic-cache $S/nrd/cache --selftest`: 40 episodes regenerated (manifest sha256
   8db58091...), shapes (5,16,64)/(5,16,17)/(5,16,1) for tag and notag, tag changes the output, round trip 0.0.
2. NRD 200 steps (`$S/nrd/run200`): loss 1.3770 (1) -> 0.1138 (100) -> 0.0810 (200), sel 0.9350 -> 0.6519, val_loss
   0.0344; 483 numeric metrics, none non-finite; gate cells filled for rigid/crm/all (stalled n=8, moving 63/61,
   brake 4): identical to sections 3 and 5.
3. Stop at 100 + resume (`$S/nrd/run_stop100`): stop_reason `stop_at_step 100`, resumed "at step 100", finished 200/200;
   486 numeric metrics equal to the straight run (max |diff| 0, `wall_s`/`train_sps` excluded), losses at 100/101/200 =
   0.1138/0.1093/0.0810 as before; `config.json` carries `resumed_from`.
4. NEW: the exact cluster command line (`G/train/nrd_tag.sbatch` flags incl. explicit `--grid` and `--twin-split`) run
   locally on the real cache with `--max-episodes 200 --steps 4 --eval-every 4 --val-max-per-domain 16 --val-batches 1
   --warmup-steps 2` (`$S/nrd/cluster_cmd_dryrun`): twin-split cross-check "1200 cache groups agree ... train 1089 / val
   56 / test 55", 200 train / 200 val episodes loaded in 0.6 s, 4.90 M params, all gate cells present (crm brake n=9 in
   this subsample), GPU peak 3.77 GiB, `finished: true`. So the argument set of the sbatch is valid.
5. Tracker env smoke (`$S/tracker/smoke`, 64 envs): PolicyObs agreement 1.6e-7 / 0 / 0; pure pursuit cross-track
   0.0317 m, reward 0.6476, 0 failures; random reward -0.0308; identical to before. ~1.0k env steps/s under the
   foreign load (8-13k idle in section 3).
6. PPO 3 iterations, warm start, DEFAULT logger (`$S/tracker/ppo3`; section 3 command without `--logger none`): exit 0,
   tensorboard events written; imitation on 2,048 samples (986 rigid / 1,062 crm) pre-tanh MSE 2.2905 -> 1.6061,
   physical MAE [0.0494, 0.2378, 0.4157] -> [0.0573, 0.1548, 0.2655]; after 2 PPO iterations 1.3603 /
   [0.0354, 0.0917, 0.2142]; rewards finite; run_state 3/3. Exports at init/0/1/2 checked on 164 observations each:
   max |da| 9.3e-8, 7.9e-8, 1.1e-7, 1.4e-7 (< 1e-5). `actor_2.npz` byte-identical to `actor.npz`, `policy_meta_2.json`
   == `policy_meta.json`; the meta carries `obs_layout` (158-D: errors [0,3], preview [3,33], vx/yaw rate [33,35],
   last action [35,38], past actions [38,62], past states [62,158], state cols 0-6, 11-15), squash centre (0, 0.5, 0.5)
   scale (1, 0.5, 0.5) box [-1,0,0]..[1,1,1], steering rate 0.1, settle (0, 0, 1), `nrd_sha256`, `nrd.cond = tag`,
   `cache_manifest_sha256`, iteration / completed_iterations, the imitation record and the check result.
   (rsl_rl saves `model_<it>.pt` both at `it % save_interval == 0` and at the end of `learn()`; with `--save-interval 1`
   the same index is therefore exported twice in a row, harmless.)
7. PPO resume (`$S/tracker/ppo3_resumed`, `--max-iterations 5 --resume $S/tracker/ppo3/model_2.pt`): "3 iterations
   completed, continuing at iteration 3"; iterations 3 and 4 run; run_state 5/5; `actor_3.npz`, `actor_4.npz` exported
   (max |da| 9.2e-8, 1.9e-7 on 164 observations).
8. NEW `--check-actor` (`$S/tracker/ppo3_check`):
   `$P scripts/gb_train_tracker.py --out $S/tracker/ppo3_check --nrd $S/nrd/run200/ckpt_best.pt --cache $S/nrd/cache --num-envs 64 --check-actor $S/tracker/ppo3/model_2.pt --check-n 100`
   -> `actor_2.npz` vs `model_2.pt` on 164 observations: max |da| 7.2e-8 (pre-squash 2.0e-7), ok, exit 0;
   `actor_check_2.json` records the npz sha256.
9. NEW independent check without any trainer code (`$S/tracker/check_actor_independent.py`, kept as a self-test
   artefact): rebuilds `ActorCritic` + `EmpiricalNormalization` from `model_2.pt` with rsl_rl's own classes and
   `train_cfg.json`, draws 100 real observations from a fresh env (seed 777), applies the squash from
   `policy_meta_2.json`:
   `$P $S/tracker/check_actor_independent.py $S/tracker/ppo3/model_2.pt $S/tracker/ppo3/actor_2.npz 100`
   -> n = 100, max |da| 7.1e-8, ok (`$S/tracker/check_actor_independent.json`).

### 6c. The first real NRD job (cluster job 430699, `G/train/nrd_tag.sbatch`) never reached the trainer

`sacct`: FAILED after 1 s on k007-005-v3; the whole log is `/etc/profile: line 50: HISTCONTROL: unbound variable`.
The wrapper runs `set -uo pipefail` BEFORE `source /etc/profile`, and `/etc/profile` line 50 reads `$HISTCONTROL`,
which is unset in a batch shell, so bash aborts under `-u` before `python3.12 ... gb_train_nrd.py` is called. Not a
trainer defect: the same argument set runs locally (6b item 4), and `G/data` holds everything the command needs
(`cache_v1` with 39,238 files incl. `cache_manifest.json`, `grids/arena_f104_50h_v1/grid.npz` + `grid.json`,
`twin/twin_crm.npz`). Fix for whoever relaunches (I do not submit jobs): use `set -eo pipefail` like the repo's other
sbatch files (`scripts/crm_train.sbatch`, `gen_array_g.sbatch`), or move `set -u` below the `source` lines. The
wrapper's own resume logic (`[ -f $OUT/ckpt_last.pt ] && RESUME=...`) is compatible with the trainer's argument check
because the second job repeats the identical command line.

### 6d. Recipes (unchanged from section 2, plus the check)

After a PPO run: `... --check-actor <run>/model_<it>.pt --check-n 100` (same `--nrd/--cache/--num-envs` as the run)
before handing `actor_<it>.npz` to the collectors; both worlds' collectors must load the same npz and assert equal
`action_center/scale/low/high` and `obs_layout` across arms (`policy_meta.json`).
