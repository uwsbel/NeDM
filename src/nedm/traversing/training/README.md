# Traversing: training code

Training code for the models released with the traversing study, trimmed to what training uses. Run everything from
the repository root with the `nedm` environment. Relative input paths are the paths the release downloader
(`scripts/traversing/release/download_traversing_data.py`) restores, so the defaults work after a download. Each
entry point writes its outputs to `--out`.

| File | Contents |
|---|---|
| `state.py` | the 17-number vehicle state, the observable columns and the settle action |
| `dynamics_data.py` | dynamics cache loader, group split and twin check, normalisation, window batching |
| `dynamics_model.py` | NRD model (main's `nedm.core.training.model_transformer` + terrain-crop token), checkpoint save/load |
| `train_dynamics_model.py` | NRD training with the validation gates |
| `risk_model.py` | route-risk CNN-GRU with the history encoder, survival loss, route logit, checkpoint load and scoring |
| `risk_data.py` | risk dataset files -> rows, splits, suite-group exclusion, standardised tensors, balanced batches |
| `risk_metrics.py` | the offline metrics of the risk training record |
| `train_risk_model.py` | route-risk ensemble training |
| `tracking_env.py` | the tracker's imagined-drive environment: recorded route fragments driven inside the frozen NRD |
| `numpy_actor.py` | the tracking policy exported to NumPy (`actor.npz`) and its loader |
| `train_tracking_policy.py` | PPO tracker training: imitation warm start, rsl_rl PPO, NumPy export with the parity check |

## Dynamics model (NRD)

The learned 50 ms vehicle-dynamics model inside which the round-2 PPO tracker was trained (released as
`m3_tracker_learned_dynamics_model`, run `nrd_tag_v3`). It is a causal transformer (6 layers, 8 heads, width 256,
16-frame context, 4.9 M parameters) over the 17-number state, the controls, an 8 x 8 terrain crop around the
dead-reckoned pose and the ground type (rigid or soil). It predicts the next state change and the power.

**Inputs** (downloader restore paths):

| Input | Path | Release item |
|---|---|---|
| episode cache, 43,235 episodes (train 39,601 / val 1,833 / test 1,801) | `artifacts/traverse/generalist_20260921/B_tracker/cache_v3` | `tracker_dynamics_training_cache` |
| f104 terrain grid | `artifacts/traverse/crm_f104_v1/grids/arena_f104_50h_v1/grid.npz` | `f104_terrain_grid` |
| twin group split (cross-check only) | `artifacts/traverse/crm_night2_v1/datasets/twin_crm.npz` | `tracker_group_split_file` |

**Command.** The run was cluster job 430932 on one MI350X: 28 min for 30,000 steps plus about 1 min of loading. The
whole cache is held in host memory (the state array alone takes about 3 GB).

```bash
PYTHONPATH=src python -m nedm.traversing.training.train_dynamics_model --out runs/nrd_tag_v3 --cond tag --crop-k 8 --crop-half-m 6 \
    --steps 30000 --batch 256 --rollout-steps 8 --delta-scale --eval-every 2000 --val-max-per-domain 256 \
    --ckpt-every-min 15 --max-minutes 225
# after an interruption: the same command plus --resume runs/nrd_tag_v3/ckpt_last.pt
```

This maps 1:1 onto the recorded `scripts/gb_train_nrd.py` command at commit `901d6c9`. Every recorded option keeps its
name and meaning. `--cache`, `--grid` and `--twin-split` default to the restore paths above, so they can be omitted.
The remaining settings (lr 3e-4 to 3e-5 cosine after 1,000 warm-up steps, AdamW weight decay 0.1, clip 1, 16-frame
context, soil fraction 0.5, hold-bad weight 0.25, episodes cut at 1,200 frames, seed 20260921) are the defaults of
both scripts. Options that no released run used were removed, each at the value the run used: `--domains` (both),
`--progress-weight` (0), `--context-noise` (0), `--vx-weight` (1), `--dropout` (0), `--cond notag`, and the synthetic
cache and self-tests.

**Outputs:** `config.json`, `train_log.jsonl`, `metrics.json`, `ckpt_best.pt` (the model with the lowest validation
error), `ckpt_last.pt` (adds the optimizer and RNG states for `--resume`) and `run_state.json`. The checkpoint format
is unchanged (`model_kind` `gb_nrd`), so the released `ckpt_best.pt` loads with `dynamics_model.load_nrd`.

**Expected gates.** Every 2,000 steps the model rolls out 60 frames (3 s) with its own predictions fed back. It does
this on up to 256 windows per window kind and ground type from the val groups. The gates are: a stalled vehicle
must not slide more than 1 m in over 20 % of stalled windows; the displacement error on moving windows must stay
below 25 %; the speed-change error after brake onset must stay below 25 %. The released checkpoint (step 26,000 of
30,000) passes all of them:

| Gate (fed back, 60 frames) | Threshold | Rigid | Soil |
|---|---|---|---|
| stalled windows that slide more than 1 m | < 0.20 | 0.059 | 0.000 |
| displacement error, moving windows | < 0.25 | 0.088 | 0.041 |
| speed-change error after brake onset | < 0.25 | 0.202 | 0.122 |

The selection metric (mean fed-back normalised state error at 60 frames on random windows) is 0.2543. A retrain
reaches similar numbers but not identical bits: on a GPU, the backward pass of the terrain crop (`grid_sample`)
uses atomic additions, so it is not deterministic. On the CPU, training is deterministic and matches the original
script bit for bit.

**For the tracker.** The modules form the package `nedm.traversing.training`: with `src/` on `sys.path`, other code
imports them as `from nedm.traversing.training import dynamics_model`. The entry points need the same: run them from
the repository root with `PYTHONPATH=src`, as in the commands above.
`dynamics_model.load_nrd(path, device)` returns the frozen model, its normaliser and the checkpoint.
`model.token(pose)` gives the crop token and `integrate_pose` does the dead reckoning. `dynamics_data` provides the
manifest, split and held-out-group helpers. `state` provides `OBSERVABLE_COLS` and `SETTLE_ACTION`.

## Route-risk model

The network the planners use to score candidate routes. It is a CNN-GRU that reads a 96-station route corridor
(elevation, grade, cross slope, planned speed) and the route geometry, and gives a hazard at every station. The route
failure probability is 1 - exp(-sum softplus(hazard)). The final shared rigid/soil model (released as
`m2_shared_history_early_rows_final_model`, run `deploy_a1_haux_gru`) adds a history encoder. A causal GRU over the
last 2 s of observable state and applied controls (40 frames x 15 channels) gives a 16-number code z that joins the
geometry input. A small head on z predicts the ground type as an auxiliary loss (`--cond hist_aux`). The model is
never told which ground it is on. The ensemble has 5 seeds of 262,018 parameters each.

**Inputs** (downloader restore paths, the default `--ds`):

| Input | Path | Release item |
|---|---|---|
| re-anchored rows + 4,614 branch rows, 120,482 rows | `artifacts/traverse/generalist_20260921/A_adapt/datasets/mixed_reanchor_plus_branch_both.npz` | `shared_model_final_reanchored_plus_branch_rows` |
| early decision rows (frames 10, 20, 30 of every drive), 89,998 rows | `artifacts/traverse/crm_improve_20260922/datasets/short_anchor.npz` | `shared_model_final_short_anchor_rows` |
| decision rows at frame 60 (3 s), 29,699 rows | `artifacts/traverse/crm_improve_20260922/datasets/anchor_k60.npz` | `shared_model_final_anchor_k60_rows` |

**Command.** The run was cluster job 431491 on one MI350X: about 13.5 min per seed (25,500 steps) plus 42 s of
loading. All rows used for fitting or evaluation are held on the GPU as float32: 16.3 GB, with a peak of 17.5 GB while
training.

```bash
PYTHONPATH=src python -m nedm.traversing.training.train_risk_model --out runs/risk_final --tag deploy_a1_haux_gru \
    --arch gru --hist-enc gru --hist-window mask --ctx geom --cond hist_aux --domain-filter both \
    --crm-batch-frac 0.5 --hist-drop 0.2 --aux-weight 0.5 --split-eval val --mode deploy \
    --seeds 5 --seed0 0 --epochs 30 --bs 256
```

This is the recorded `scripts/ci_train.py` command at commit `901d6c9` (also written into `deploy_a1_haux_gru.json`),
option for option. The three `--ds` files are omitted because they are the defaults; pass them as repeated `--ds` to
use other copies. The GPU is picked with `--device` (default cuda when available), not with `CI_DEVICE`. The
remaining settings are the defaults of both scripts:

- AdamW with lr 2e-3 and weight decay 1e-4, a one-cycle schedule over 30 x (217,814 // 256) = 25,500 steps, gradient
  norm clipped at 5.
- Every batch has 128 rigid and 128 soil rows. The corridor, geometry and history are standardised on the fitted
  rows.
- Seeding uses `torch.manual_seed(seed)` plus a CPU generator for the batch order and the history drop.

`--mode` (deploy: every train row is fitted), `--split-eval` (val), `--arch`, `--hist-enc`, `--hist-window` and `--ctx`
each accept only the value every released model used. They exist so that recorded commands run unchanged. Options
that no released model used were removed, each at the value the runs used:

- `--mode holdout` and `--split-eval test` (used only by unreleased offline read-out runs)
- `--hist-T` (all 40 frames)
- `--data-frac-crm`, `--data-frac-rigid` (1) and `--data-seed`
- `--row-weight` (none), `--startup-only` (off), `--subsample` (off)
- `--lr`, `--wd` (the CNN-GRU defaults above)
- `--x-half`, `--x-host` (float32 corridor on the GPU), `--keep-all-rows`, `--strict-keys`, `--deterministic`,
  `--roundtrip-n` (2,048) and `--no-save`
- the transformer networks (`--arch txD_L`, `txjoint`), `--hist-enc tx`, `--ctx geom_vel` and `--cond tag|hist`

Rows from the evaluation suites (`f104_pair_group_*`, `f104_crm_eval_group_*`, the unseen-arena test and held-out
groups) must never be fitted; the script stops if any are.

**Outputs:**

- per seed, `<tag>_s<seed>.pt`, in the unchanged checkpoint format (`model_kind` `ci_train`, same keys), so the
  released checkpoints load with `risk_model.load_risk_model`
- `<tag>_logits.npz`, the member and ensemble route logits of every fitted and evaluated row
- `<tag>.json`, the arguments, row counts, split hash, and per-seed and ensemble metrics (see `risk_metrics.py`)

For scoring, `load_risk_model`, `encode_history` (once per decision) and `score` give the route logits of candidate
corridors.

**Expected numbers.** The released run fitted 217,814 rows (109,492 rigid and 108,322 soil). The split hash is
`b37fafe2`. The val split holds 5,680 rigid and 5,605 soil rows. Within-group AUC against unsafe (`W_unsafe`) is split
into established rows (decision after the vehicle has moved) and startup rows (decision at rest):

| Seed | Final loss | Rigid W_unsafe, established / startup | Soil W_unsafe, established / startup | Pooled AUC, rigid / soil |
|---|---|---|---|---|
| 0 | 0.328 | 0.972 / 0.963 | 0.984 / 0.961 | 0.972 / 0.980 |
| 1 | 0.347 | 0.979 / 0.965 | 0.983 / 0.965 | 0.978 / 0.981 |
| 2 | 0.340 | 0.971 / 0.958 | 0.982 / 0.968 | 0.973 / 0.980 |
| 3 | 0.422 | 0.973 / 0.947 | 0.985 / 0.972 | 0.973 / 0.983 |
| 4 | 0.403 | 0.975 / 0.961 | 0.985 / 0.972 | 0.974 / 0.983 |
| ensemble | | 0.982 / 0.968 | 0.990 / 0.979 | 0.981 / 0.988 |

A retrain matches these closely but not bit for bit, because GPU arithmetic differs between machines. On the RTX 5090
the first-step losses differ from the recorded ones by 2e-5 to 2e-4 relative.

**Other released risk models.** The same script, with one `--ds` file, `--cond none` and one ground type, trained
the single-ground planners of the arena and vehicle studies. Each recorded `--mode deploy` line (`e5/jobs/*.tsv` of
the arena study, the Polaris deploy record) maps 1:1; the `--mode holdout` lines there trained unreleased offline
read-out models and are not supported:

```bash
PYTHONPATH=src python -m nedm.traversing.training.train_risk_model --ds <file> --out runs/<tag> --tag <tag> --mode deploy --arch gru \
    --cond none --domain-filter <rigid|crm> --ctx geom --split-eval val --bs 256 --epochs 30 --seeds 5 \
    --seed0 <0|5> --roundtrip-check
```

| Tag (release item) | `--ds` (restore path under `artifacts/traverse/`) | Filter | Seeds | Fitted rows | Ensemble W_unsafe (val) |
|---|---|---|---|---|---|
| M1a_rigid_deploy / M1b_rigid_deploy (`m4a_rigid_f104_only_ens1/2`) | `arena_gator_20260925/e4/subsets/M1_f104_hmmwv_rigid.npz` | rigid | 0-4 / 5-9 | 84,787 | 0.986 / 0.987 |
| M2_rigid_deploy (`m4a_rigid_two_arenas_same_total`) | `arena_gator_20260925/e4/subsets/M2_hmmwv_rigid.npz` | rigid | 0-4 | 84,310 | 0.979 |
| M3a_rigid_deploy / M3b_rigid_deploy (`m4a_rigid_three_arenas_same_total_ens1/2`) | `arena_gator_20260925/e4/subsets/M3_hmmwv_rigid.npz` | rigid | 0-4 / 5-9 | 84,481 | 0.973 / 0.973 |
| A3_rigid_deploy (`m4a_rigid_three_arenas_all_data`) | `arena_gator_20260925/e4/subsets/A3_hmmwv_rigid.npz` | rigid | 0-4 | 250,490 | 0.985 |
| M1a_soil_deploy / M1b_soil_deploy (`m4a_soil_f104_only_ens1/2`) | `arena_gator_20260925/e4/soil_s1/subsets/M1_f104_hmmwv_soil.npz` | crm | 0-4 / 5-9 | 29,210 | 0.973 / 0.977 |
| M2_soil_deploy (`m4a_soil_two_arenas_same_total`) | `arena_gator_20260925/e4/soil_s1/subsets/M2_hmmwv_soil.npz` | crm | 0-4 | 29,048 | 0.950 |
| M3a_soil_deploy / M3b_soil_deploy (`m4a_soil_three_arenas_same_total_ens1/2`) | `arena_gator_20260925/e4/soil_s1/subsets/M3_hmmwv_soil.npz` | crm | 0-4 / 5-9 | 29,071 | 0.957 / 0.946 |
| A3_soil_deploy (`m4a_soil_three_arenas_all_data`) | `arena_gator_20260925/e4/soil_s1/subsets/A3_hmmwv_soil.npz` | crm | 0-4 | 57,898 | 0.959 |
| G_rigid_deploy (`m4b_gator_rigid_own_model`) | `arena_gator_20260925/e4/subsets/G_f104_gator_rigid.npz` | rigid | 0-4 | 84,922 | 0.974 |
| G_soil_deploy (`m4b_gator_soil_own_model_tiers0to6`) | `arena_gator_20260925/e4/soil_s1/subsets/G_f104_gator_soil.npz` | crm | 0-4 | 28,057 | 0.940 |
| G_full_soil_deploy (`m4b_gator_soil_own_model_full_data`) | `arena_gator_20260925/e4/soil_bf/subsets/G_full_f104_gator_soil.npz` | crm | 0-4 | 50,822 | 0.960 |
| H_full_soil_deploy (`m4b_hmmwv_soil_model_full_data`) | `arena_gator_20260925/e4/soil_bf/subsets/H_full_f104_hmmwv_soil.npz` | crm | 0-4 | 52,923 | 0.986 |
| polaris_full_soil_deploy (`m4b_polaris_soil_own_model_full_data`) | `offroad_vehicles_20260927/e4/subsets/polaris_full_f104_soil.npz` | crm | 0-4 | 52,021 | 0.984 |

W_unsafe is pooled over startup and established rows of the one ground type. Two groups of released risk models are
not trained by this script. The joint station/history transformer (`deploy_a3_haux_txjoint`) came from the same
experiment script with `--arch txjoint`, which was not ported; `load_risk_model` refuses it. The generalist-study
models (`H_deploy`, `P_deploy`, `T_deploy`, `Sp_*`) and the earlier single-ground planners came from other scripts.

**Equivalence with the original.** The following were checked against `scripts/ci_train.py` at `901d6c9`, with torch
2.12. Every comparison was bit for bit.

- All 90 released CNN-GRU checkpoints load with `load_risk_model` into identical weights. They give identical route
  logits and history codes on real rows, on CUDA and on the CPU. This held for per-row, shared, startup and
  precomputed history windows.
- Training was run side by side on real subsets with the same seeds. The cases were the recorded `deploy_a1` flags,
  the arena recipe with `--roundtrip-check` and seed0 5, soil only with `--cond none`, and CUDA with deterministic
  algorithms.
  The per-step losses, final parameters, checkpoints, logits and every metric are identical.
- One epoch on the full three files with the recorded flags (850 steps, CUDA, deterministic) is also identical. The
  port's data preparation reproduces the normalisation, split hash and row counts stored in the released checkpoints.

## PPO tracking policy

The round-2 learned route tracker (run `ppo_v2`). It is released as `m3_tracker_round2_numpy_actor` (`actor.npz` and
`policy_meta_999.json`, what the drives used) and `m3_tracker_round2_ppo_checkpoint` (`model_999.pt`, `env_cfg.json`,
`train_cfg.json`). The policy is an ELU MLP (512-256-128) from a 158-number observation to steering, throttle and
brake. It was trained with rsl_rl PPO inside the frozen NRD model: every environment replays a 1 to 3 s fragment of a
recorded f104 drive (half of the resets on soil) from the recorded 16-frame history and then drives on in the model.
The reward is a cross-track, heading and speed tracking term, small penalties on action changes and on throttle with
brake, and a progress bonus (`tracking_env.py` docstring). Before PPO, the actor is fitted to the recorded PID actions
on 131,072 samples (imitation warm start).

**Inputs** (downloader restore paths):

| Input | Path | Release item |
|---|---|---|
| NRD dynamics model | `artifacts/traverse/generalist_20260921/B_tracker/nrd_tag_v3/ckpt_best.pt` | `m3_tracker_learned_dynamics_model` |
| episode cache (fragments from the 39,601 train episodes) | `artifacts/traverse/generalist_20260921/B_tracker/cache_v3` | `tracker_dynamics_training_cache` |
| f104 terrain grid (sha256 checked against the NRD) | `artifacts/traverse/crm_f104_v1/grids/arena_f104_50h_v1/grid.npz` | `f104_terrain_grid` |
| twin group split (cross-check only) | `artifacts/traverse/crm_night2_v1/datasets/twin_crm.npz` | `tracker_group_split_file` |

**Command.** The run took 21.6 min on the workstation's RTX 5090, including 75 s to load the 19.4 M-frame bank onto
the GPU (2.1 GiB).

```bash
PYTHONPATH=src python -m nedm.traversing.training.train_tracking_policy --out runs/ppo_v2 --num-envs 2048 --max-iterations 1000 \
    --save-interval 100 --imitation-samples 131072 --seed 2 --speed-weight 1.5
# after an interruption: the same command plus --resume runs/ppo_v2/model_<it>.pt
```

This maps 1:1 onto the recorded command (`run_ppo_v2.sh`, `scripts/gb_train_tracker.py` at `901d6c9`). Every
recorded option keeps its name and meaning. `--nrd`, `--cache` and `--grid` default to the restore paths above, so
they can be omitted. With an `--out` folder named `ppo_v2`, the written `env_cfg.json` and `train_cfg.json` equal the
released ones. The rest of the recipe is the default of both scripts: PPO with 64 steps per environment, 5 epochs of
8 mini-batches, learning rate 3e-4 adapted to KL 0.01, entropy 0.003, initial noise 0.7, empirical observation
normalisation; imitation 2 epochs, batch 4,096, lr 1e-3. Removed: `--split` (always the train groups),
`--allow-notag` (only the ground-type-conditioned NRD loads) and `--smoke` / `--smoke-steps` (a scripted-controller
smoke test and a check against the deployed controller's observation code, neither part of training). Added:
`--check-npz`, to point `--check-actor` at the released `actor.npz`.

**Outputs:** `env_cfg.json`, `train_cfg.json`, `imitation.json`, `model_init.pt` (after the warm start),
`model_<it>.pt` (iterations 0, 9, 100, ..., 900 and 999; the save after iteration 9 ends the first learning segment,
after which the actor-vs-PID error is logged), `actor.npz` and `policy_meta.json` (latest export) with per-checkpoint
copies `actor_<it>.npz` and `policy_meta_<it>.json`, `run_state.json`, and rsl_rl's tensorboard events and `git/`
folder. `policy_meta.json` holds the observation layout (`obs_layout`: slices, 10 preview points at 1 m, 8-step
history of the 12 observable state columns), the action squash and bounds, the 0.1 steering-rate limit, the NRD
checkpoint and its sha256, the cache manifest sha256, the bank counts (rigid 24,280, soil 15,321), `env_cfg`, the
training arguments, the iteration, the imitation record and `numpy_actor_check`.

**Expected.** The warm start takes the pre-tanh error to the PID actions from 3.775 to 0.109. Every export checks
the NumPy actor against the torch actor on 100 fixed bank observations plus the 2,048 live ones; the run stops if the
largest action difference reaches 1e-5. The released export (iteration 999) records 5.0e-7 on 2,148 observations. To
repeat that check on the released checkpoint (loading the released `model_999.pt`, here or with `--resume`, needs a
visible CUDA device even with `--device cpu`: it stores CUDA tensors and rsl_rl loads it without `map_location`, as
in the original script):

```bash
PYTHONPATH=src python -m nedm.traversing.training.train_tracking_policy --out runs/ppo_v2_check --seed 2 --speed-weight 1.5 \
    --check-actor artifacts/traverse/generalist_20260921/B_tracker/ppo_v2/model_999.pt \
    --check-npz artifacts/traverse/generalist_20260921/B_tracker/ppo_v2/actor.npz
```

**Equivalence with the original.** Checked against `scripts/gb_train_tracker.py` at `901d6c9` with torch 2.12 and
rsl_rl 2.2.4; every comparison was bit for bit. The released `model_999.pt` loads into identical weights and gives
identical environment observations, rewards and resets, actor outputs and a re-export equal to the released
`actor.npz`, on CUDA and with `--device cpu`. Side-by-side training on a CPU subset (deterministic algorithms),
including a resume, gives identical losses, rollouts, checkpoints and exports. On the same RTX 5090, the ported trainer run with
the full `ppo_v2` command reproduces the released run exactly: every checkpoint, export and record of the 1,000
iterations is identical, and `model_999.pt` is byte-identical to the released file. GPU runs on other hardware or
library versions will differ in the last bits and then drift apart.
