# NeDM Project Progress

Reproduction record for *Learning the Right Abstraction: Neural Reduced Dynamics
for Complex Robot Control* (Zhang and Negrut). Every stage output the manuscript
reports is listed here with the artifact that produced it and the command that
regenerates it.

Last updated: 2026-08-07 — repo pruned to the manuscript's reproduction set. The
tracked artifact tree is now an allowlist in `.gitignore`; a paper artifact that
is missing a rule shows up in `git status` rather than staying silently
untracked.

**Scope of what is in git.** Checkpoints, run metadata, Chrono evaluation output
and reference sets are version controlled (~2 GB via LFS). Raw episode CSVs
(`artifacts/datasets/`, ~337 GB) and processed cache arrays
(`artifacts/training_datasets/`, ~73 GB) are local-only — regenerate them with
the collection and preprocessing scripts in the tables below.

## Status at a glance

| Paper section | Stage output | Headline | Artifact |
|---|---|---|---|
| Sec. IV-C | Terrain-conditioned HMMWV NN-ROM | flat 3.7% / CRM 5.4% open-loop 10 s err/dist, epoch 51 | `training_runs/ablation_ofat/L8_H8_E256_ctx128` |
| Sec. IV-E, App. B | Three tracking policies transferred to Chrono | generalist takes the lowest median **and** mean XY RMSE on all three terrains; 9/9 cells 20/20, zero early terminations | 3 × `rl_runs/…_ofatL8_…` |
| App. C | Architecture OFAT sweep, 14 configs | depth dominates; returns saturate past L8 | `training_runs/ablation_ofat/` |
| App. D | Training-data scaling, 20–100% | S falls 6.9% → 4.3% over 20–80% | `…/L8_H8_E256_ctx128_data{20,40,60,80}` |
| App. E | Reduced-state and context ablation | no one-hot doubles flat rollout error; no terramechanics costs CRM ~49% | `…_no_onehot`, `…_no_tireforce_omega` |
| Sec. V-C | Tracked-base NN-ROM | 5 s rollout 0.105 m XY / 5.85% err/dist, epoch 8 | `training_runs/tracked_transformer_v1` |
| Sec. V-C | Arm NN-ROM (8-D `[q, q̇]`) | 1.2 mm one-step EE, 1.2% EE drift at 2 s, epoch 76 | `training_runs/arm_transformer_8d_v1` |
| Sec. V-E, App. G | Tracked-base goal reaching in Chrono | **100/100** at 0.75 m, median closest approach 0.691 m | `rl_runs/tracked_goal_v2_far_rollsel_rom_20260721` |
| Sec. V-E, App. G | Arm end-effector reaching in Chrono | **97/100** at 0.05 m, zero contacts, zero joint-limit violations | `rl_runs_arm_goal_reach/…_8d_rom_20260727` |

Two model-selection rules hold everywhere and are worth stating once:

- **Checkpoints are selected on open-loop rollout error, not one-step loss.**
  `checkpoint_metric: rollout_sel` in every deployed config. The file is still
  named `best_val.pt`, but it is the rollout-selected epoch. The two metrics rank
  checkpoints differently: for `tracked_transformer_v1`, rollout picks epoch 8
  while one-step validation loss would pick epoch 36.
- **The RL environment queries the frozen ROM with a 16-step context**, not the
  full 128-step training context. Attention is quadratic in context length, so
  this is ~6.8× faster (≈5,080 → ≈35,000 policy-control steps/s) at no cost to
  tracking quality; the dynamics are close to Markovian at this scale.

---

## Study Case I — terrain-aware HMMWV

### Datasets

| Role | Raw | Processed cache | Scale |
|---|---|---|---|
| Train / val / in-domain test | `datasets/hmmwv_tire_rigid_300g_shards` (128 shards, 305 GB) | `training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1` | 26,124 train eps / 128.0 M transitions; 6,644 val / 32.5 M |
| Train / val / in-domain test | `datasets/hmmwv_crm_2000` (2,000 eps, 5.1 GB) | `training_datasets/hmmwv_crm_2000_normal_force_omega_seq_v1` | 1,582 train eps / 2.28 M; 418 val / 0.60 M |
| Zero-shot OOD test only | `datasets/hmmwv_bumpy_10g_shards` (8.7 GB) | — (20 eval references only) | never enters training, selection, normalization or reward tuning |
| App. E state ablation | (column slice of the above) | `…_300g_body7_seq_v1`, `…crm_2000_body7_seq_v1` | 7-D readout; built by column-slicing, not re-preprocessing |

Heightmaps for the bumpy regime live in `assets/bumpy_terrain/` (100 BMPs), and
the eval reproduces the exact per-episode patch each reference was recorded on.

Regenerate:

```bash
python scripts/prepare_hmmwv_tire300g_generation.py     # shard plan
sbatch scripts/cluster/collect_hmmwv_tire300g.sh        # collect (cluster only)
python scripts/build_hmmwv_training_dataset.py --help   # raw -> cache
python scripts/ablation_ofat/derive_state_subset_dataset.py   # body7 caches
```

`bash scripts/smoke_test_hmmwv_tire10g.sh` (and the `bumpy10g` / `crm` variants)
rehearse the whole path at small scale before committing cluster time.

### Reduced dynamics model

15-D state `[vx, vy, φ, θ, ωx, ωy, ωz, Fz×4, ω×4]`, 3-D driver action, 2-D
terrain one-hot → 20-D token. L8 / 8 heads / E256 / ctx128, 6.40 M parameters,
75/25 flat/CRM sub-batches, per-channel domain-rebalanced Huber loss,
domain-balanced rollout selection `S = ½E_rigid + ½E_CRM`.

- Config: `configs/ablation_ofat/L8_H8_E256_ctx128.json`
- Run: `artifacts/training_runs/ablation_ofat/L8_H8_E256_ctx128/`
- Selected epoch 51, S = 4.56% (flat 3.73%, CRM 5.38%)
- Anchor for the sweep (ablation model 10, trained once):
  `training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g_crm2000_mix25_rebal_rollout_onehot/`

Specialists for Table 4 are the data-mix arm of the same sweep:
`…_mix00` (flat-only, epoch 61) and `…_mix100` (CRM-only, epoch 54). Each is
close to the generalist in-domain but collapses off it — flat-only reaches 194%
rollout error on CRM, CRM-only 40% on rigid.

### Tracking policies

PPO over 2,048 vectorized copies of the frozen ROM. 231-D observation, 3 driver
commands with the steering channel rate-limited, 20 Hz control over 100 Hz
dynamics. Evaluated at iteration 1000.

| Policy | Run |
|---|---|
| Mixture generalist | `rl_runs/hmmwv_rl_15d_crm2000mix25_onehot_ofatL8_bestval51_flat20crm20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010/` |
| Rigid-only | `rl_runs/hmmwv_rl_15d_crm2000mix00_onehot_ofatL8_bestval61_rigid20_…_1000it/` |
| CRM-only | `rl_runs/hmmwv_rl_15d_crm2000mix100_onehot_ofatL8_bestval54_crmonly20_…_1000it/` |

References: 40 training refs (20 flat + 20 CRM, random mid-episode windows) in
`hmmwv_tire_normal_force_omega_flat_crm_train_refs_40_1100_randwin_seed20260623.npz`;
evaluation uses a **separate** held-out set of 20 rest-start references per
terrain. Rest-start matters — the Chrono warm start only works if reference
index 0 is at zero speed, so build eval refs with `--no-random-segment-start`.

Each run carries its three Chrono evaluations
(`chrono_eval_tracking_…`, `chrono_bumpy_eval_…`, `chrono_crm_eval_…`) and the
matching `eval_cfg_*` directories. The nine cells are collated into
`rl_runs/chrono_eval_comparisons/onehot_policy_3x3_chrono_xy_rmse_median_iqr_steerlim010_ofatL8_model1000.{csv,json,pdf,png}`.

Closed-loop XY RMSE (m), 20 references per cell, all 20/20:

| Terrain | Generalist | Rigid-only | CRM-only |
|---|---|---|---|
| Rigid flat | **0.157** med / **0.184** mean | 0.174 / 0.219 | 0.232 / 0.259 |
| CRM | **0.180** / **0.249** | 0.854 / 1.000 | 0.231 / 0.361 |
| Rigid bumpy (zero-shot) | **0.149** / **0.229** | 0.187 / 0.238 | 0.213 / 0.418 |

### Ablations

| Appendix | Runs | Ranking script |
|---|---|---|
| C — architecture OFAT (14 configs) | `ablation_ofat/{L2,L4,L12}_…`, `{L6_H4_E128, L6_H6_E192, L6_H12_E384, L6_H16_E512}_…`, `{L6_H16_E256, L6_H4_E256}_…`, `L6_H8_E256_ctx{32,64,256}`, `L8_H8_E256_ctx128`, + the anchor | `rank_stage_a.py`, `build_all_runs_table.py` → `all_runs_table.csv` |
| D — data quantity | `L8_H8_E256_ctx128_data{20,40,60,80}` (+ the 100% run) | `rank_data_quantity.py` → `l8_dataquantity_curve.csv` |
| E — reduced state and context | `L8_H8_E256_ctx128_no_onehot` (18-D), `…_no_tireforce_omega` (12-D in, 7-D out) | `rank_feature_ablation.py` → `l8_feature_ablation.csv` |

Judge these on `rollout_sel`, not `val_loss`: the 7-D readout's one-step loss is
computed over 7 channels instead of 15 and is not comparable across arms. The
open-loop column is, since it integrates `vx, vy, ωz`, which every variant keeps.

```bash
python scripts/ablation_ofat/gen_configs.py && python scripts/ablation_ofat/validate_configs.py
bash scripts/ablation_ofat/run_sweep.sh              # Stage A, tmux
bash scripts/ablation_ofat/run_l8_dataquantity_ablation.sh
bash scripts/ablation_ofat/run_l8_feature_ablation.sh
bash scripts/ablation_ofat/run_l8_chrono_eval_newton.sh   # 3-terrain closed loop
```

---

## Study Case II — M113 tracked vehicle with a 4-DOF arm

One Chrono scene, two control modes. Drive mode moves the base with the arm
welded at its home pose; reach mode holds the base and moves the arm. Each mode
has its own reduced state, ROM and policy.

### Drive mode

- Dataset: `configs/tracked_vehicle_drive_v2.json` → `datasets/tracked_vehicle_drive_v2_shards`
  (2,160 eps, 10 maneuver families) → `training_datasets/tracked_drive_v2_seq16_v1`
  (1.41 M train / 0.27 M val)
- ROM: 3-D `[vx, vy, r]`, 3-D action, 3L / 4H / E96 / ctx16, 0.34 M params,
  `configs/tracked_transformer_v1.json`, epoch 8
- Policy: `scripts/train_tracked_rl_goal.py`, 2,048 envs, 11-D obs, 10 Hz,
  iteration 1499 → `rl_runs/tracked_goal_v2_far_rollsel_rom_20260721/`
- Chrono: `chrono_benchmark_N100_seed12345/` — 100/100 at 0.75 m, median
  time-to-success 20.2 s, median path efficiency 0.959
- Route composition: `chrono_waypoints_fig8_bowtie/` — 8 goals chained in one
  rollout, 8/8, per-leg closest approach 0.46–0.69 m

One-step error is noise-limited here and nearly flat across epochs, so fidelity
is judged from the open-loop rollout, not the loss magnitude.

### Reach mode

- Dataset: `datasets/arm_dynamics_v3_home_reset_fulltraj_shards` (15,000 eps)
  → `training_datasets/arm_dyn_v3_8d_seq16_v1` (0.76 M train transitions)
- ROM: 8-D `[q, q̇]`, action = absolute `q_cmd`, 5L / 8H / E256 / ctx16,
  4.0 M params, `configs/arm_transformer_8d_v1.json`, epoch 76
- Policy: `scripts/train_arm_rl_reaching.py`, 4,096 envs, 26-D obs, 50 Hz,
  iteration 1499 → `rl_runs_arm_goal_reach/arm_reach_adaptivekl005_lr1e4_tol005_ep150_bonus150_sigma015_8d_rom_20260727/`
- Chrono: `chrono_reach_benchmark_N100_seed12345/` — 97/100 at 0.05 m, median
  reached error 4.17 cm, median convergence 0.9 s, **zero** contacts and **zero**
  joint-limit violations

The end-effector is **not** a learned channel: it is recovered as `FK(q)` from the
predicted joints, using the same batched forward kinematics that the safety
shield already evaluates each step. Geometry lives in
`artifacts/arm_geometry/arm_geometry_v1.json` (regenerate with
`scripts/extract_arm_geometry.py`); FK and the clearance shield are
`src/nedm/rl/arm_kinematics.py` and `arm_safety.py`.

Collection is restricted to free-space motion — episodes terminate on
arm–ground, arm–vehicle or arm–self contact — so the ROM has no notion of
contact and the shield is what keeps policy exploration inside the envelope it
was trained on.

The three remaining failures are all timeouts at deep lower-workspace goals
(target height down to −4.4 m in the arm-base frame; closest approach 6.4, 9.3
and 10.9 cm). That region is under-sampled by the collection, not a safety
failure.

### Benchmarks and figures

`benchmark_tracked_goal_chrono.py` and `benchmark_arm_reach_chrono.py` run one
goal per process and must be serialized against each other — repeated Chrono
scene re-creation in a single process crashes natively (stack smashing), which is
also why `eval_tracked_waypoints_chrono.py` swaps the active goal instead of
resetting.

---

## Regenerating the manuscript figures

All twelve scripts write into the manuscript image archive by default; pass
`--out`/`--out-dir` to redirect.

| Figure | Script |
|---|---|
| `hmmwv_cotrain_training.pdf` | `scripts/ablation_ofat/manuscript_figs/plot_l8_training_curves.py` |
| `hmmwv_rl_reward.pdf` | `scripts/ablation_ofat/manuscript_figs/plot_l8_rl_reward.py` |
| `hmmwv_policy_transfer_bars.pdf` | `scripts/ablation_ofat/manuscript_figs/plot_l8_policy_transfer_bars.py` |
| `hmmwv_policy_trajectories_grid.pdf` | `scripts/ablation_ofat/manuscript_figs/plot_l8_policy_trajectories_grid.py` |
| `tracked_arm_training.pdf` | `scripts/plot_tracked_arm_training.py` |
| `tracked_arm_rl_reward.pdf` | `scripts/plot_tracked_arm_rl_reward.py` |
| `tracked_stress_trajectories.pdf` | `scripts/plot_tracked_stress_trajectories.py` |
| `arm_stress_trajectories.pdf` | `scripts/plot_arm_stress_trajectories.py` |
| `arm_fk_boxes.pdf` | `scripts/plot_arm_fk_boxes.py` |
| imagery in `study-case-2.pdf` | `scripts/compose_tracked_arm_multiexposure.py` |

`fpp.pdf`, `hmmwv-nnrom.png` and the `study-case-2.pdf` layout are hand-drawn and
live only in the manuscript repo.

Appendix A throughput numbers come from `scripts/probe_sim_fps.py` (Chrono rows)
and the `Perf/total_fps` scalar in each PPO run's tfevents (NN-ROM rows). The
k=16 context claim comes from `scripts/bench_context_accuracy.py`; per-pass
inference cost from `scripts/bench_dynamics_inference.py`.

---

## Known gaps

1. **The CRM evaluation reference set is missing.** All three L8 runs' CRM evals
   point at
   `artifacts/rl_reference_sets/hmmwv_crm2000_val_refs_20_1100_rest_start_min10_seed20260623.npz`,
   which is not on the filesystem. The recorded results are intact, but the CRM
   column cannot be re-run until it is rebuilt with
   `scripts/build_crm_rl_references.py` from `datasets/hmmwv_crm_2000`
   (seed 20260623, `min10` displacement filter).
2. **Manuscript prose still describes the pre-correction reward run.**
   `plot_tracked_arm_rl_reward.py` read `rl_runs/tracked_goal_v2_far` while the
   100/100 Chrono result comes from `tracked_goal_v2_far_rollsel_rom_20260721`.
   Fixed and the figure regenerated on 2026-08-07; the curve is unchanged in
   shape (both runs plateau near 325). Three numbers in Sec. V-D-1 and Table 5
   came from the old run and are now wrong:
   - the transfer checkpoint is **iteration 1499**, not 1500 (there is no
     `model_1500.pt` in the transferred run);
   - that run was scheduled for **1,500** iterations, not 3,000 — 3,000 was the
     older run's `max_iterations`;
   - its wall-clock is 10.0 min, so the "≈10 min" claim still holds, as does
     "the arm run is ≈6× longer" (57 min / 10.0 min = 5.7×).

   Appendix A is unaffected: `Perf/total_fps` averages 163,022 over the correct
   run versus 163,170 over the old one, both ≈163,000.
3. **Dataset scale in the manuscript's Table 1.** It reports the flat set as
   ≈82k episodes / 329 M–81 M transitions, which describes the older
   `hmmwv_turn_300g` collection. The deployed model trains on
   `hmmwv_tire_rigid_300g_normal_force_omega_seq_v1`: 26,124 train episodes /
   128.0 M transitions, 6,644 val / 32.5 M, from 32,768 raw episodes.

## Open items

- **Arm lower workspace.** The three Chrono misses sit in a region the collection
  under-samples. Either collect more lower-workspace arm dynamics data or restrict
  the goal distribution to the covered upper/forward shell.
- **Base tolerance.** The policy stops on entering the 0.75 m region rather than
  homing onto the goal, so it hugs the radius (40/100 land in 0.70–0.75 m).
  Success at a tighter tolerance can be recomputed offline from the saved poses;
  reaching it may need reverse/differential track commands.
- **Single seed.** The architecture sweep, the data-scaling curve and both Study
  Case II policies are single-seed. The 80% data point edging out 100% on S is
  most likely seed noise, but only a repeat settles it.
- **Confounded arm comparison.** The 8-D model also dropped a layer (6L → 5L), so
  the open-loop win is not cleanly attributable to the state layout. An 8-D /
  6-layer run would separate them.
- **Contact-rich manipulation** is out of scope: the arm ROM is trained on
  free-space motion only and the shield avoids contact rather than modeling it.

---

## Superseded work

Kept as a record of what was tried. These runs and caches still exist locally but
are no longer version controlled.

| Line of work | Outcome |
|---|---|
| v01–v20 architecture sweeps, d005–d200 data scaling | Pre-dated the OFAT protocol; replaced by `ablation_ofat/` with rollout-based selection. |
| CRM-100 era (`crm100_*`: combnorm, crm40, vx3, scratch, rebal_rollout) | The limiter was CRM *data*, not batch weight — more CRM weight on the ~96k-row set just overfit. Resolved by collecting `hmmwv_crm_2000` (20×). Combined input normalization was the wrong lever: it de-centers the dominant flat domain. |
| Bumpy fine-tuning (`finetune_w_bumpy.py` and friends) | Fine-tuning the flat base on bumpy data was worse than the base. The failure was vx/omega longitudinal drift, not tire Fz; both Fz-feedback and WiSE-FT weight interpolation were refuted. Replaced by flat+CRM co-training plus rollout-based selection, which also generalizes to bumpy zero-shot. |
| Sequential flat → CRM fine-tune | Degrades the previously learned rigid behavior. Replaced by mixed 75/25 sub-batches. |
| 6-layer one-hot policy trio (`…_steerlim010` without `ofatL8`) | Superseded by the L8 backbone; same recipe, deeper dynamics. |
| 15-D arm ROM `[q, q̇, q_cmd, ee_base]` (`arm_transformer_full_v1`) | Learning `ee_base` as a channel is worse than recovering it by FK. Reach transfer 91/100. |
| 12-D arm ROM `[q, q̇, q_cmd]` (`arm_transformer_noee_v1`) | Dropping `ee_base` and using FK reached 97/100 and beat the 15-D channel by ~32% at multi-step. Superseded by the 8-D model, which treats `q_cmd` as the action rather than a state channel: same 97/100 transfer on a better open-loop ROM with 17% fewer parameters. |
| `tracked_goal_v1`, `arm_reach_fixedlr*`, `luffy_repro` | PPO tuning iterations before the reward and KL schedule settled. |
