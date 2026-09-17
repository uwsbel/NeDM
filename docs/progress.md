# NeDM Project Progress

Reproduction record for *Learning the Right Abstraction: Neural Reduced Dynamics
for Complex Robot Control* (Zhang and Negrut). Every stage output the manuscript
reports is listed here with the artifact that produced it and the command that
regenerates it.

Last updated: 2026-08-07 — repo pruned to the manuscript's reproduction set. The
tracked artifact tree is now an allowlist in `.gitignore`; a paper artifact that
is missing a rule shows up in `git status` rather than staying silently
untracked.

**Layout.** `scripts/` is organised by pipeline stage --- `collection/`,
`preprocess/`, `training/`, `ablations/`, `evaluation/`, `figures/`,
`throughput/`, plus `cluster/` for the SLURM array jobs. Every script in there
reproduces something this document records. The ablation artifacts and configs
keep their original `ablation_ofat` name, which is recorded inside run metadata.

**Scope of what is in git.** Checkpoints, run metadata, Chrono evaluation output
and reference sets are version controlled (~2 GB via LFS). Raw episode CSVs
(`artifacts/datasets/`, ~337 GB) and processed cache arrays
(`artifacts/training_datasets/`, ~73 GB) are not in git; the five datasets the
paper's models train on, and their four processed caches, are published on
Hugging Face at <https://huggingface.co/datasets/harryzhang1018/NeDM> (70 GB,
float32 Parquet + `.npy`; card in `docs/hf_dataset_card.md`). Fetch them with
`scripts/release/download_nedm_datasets.py` (`--processed` drops the caches into
`artifacts/training_datasets/`; `--rehydrate` rebuilds the per-episode CSV tree
under `artifacts/datasets/` so every script below runs unchanged) — or regenerate
them with the collection and preprocessing scripts in the tables below.

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
| Sec. V-E, App. G | Arm end-effector reaching in Chrono | **97/100** at 0.05 m, zero contacts, zero joint-limit violations | `rl_runs/…_8d_rom_20260727` |

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
python scripts/collection/prepare_hmmwv_tire300g_generation.py     # shard plan
sbatch scripts/cluster/collect_hmmwv_tire300g.sh        # collect (cluster only)
python scripts/preprocess/build_hmmwv_training_dataset.py --help   # raw -> cache
python scripts/ablations/derive_state_subset_dataset.py   # body7 caches
```

`bash scripts/collection/smoke_test_hmmwv_bumpy10g.sh` (and the `crm` variant) rehearse the
whole path at small scale before committing cluster time.

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
python scripts/ablations/gen_configs.py && python scripts/ablations/validate_configs.py
bash scripts/ablations/run_sweep.sh              # Stage A, tmux
bash scripts/ablations/run_l8_dataquantity_ablation.sh
bash scripts/ablations/run_l8_feature_ablation.sh
bash scripts/ablations/run_l8_chrono_eval_newton.sh   # 3-terrain closed loop
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
- Policy: `scripts/training/train_tracked_rl_goal.py`, 2,048 envs, 11-D obs, 10 Hz,
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
- Policy: `scripts/training/train_arm_rl_reaching.py`, 4,096 envs, 26-D obs, 50 Hz,
  iteration 1499 → `rl_runs/arm_reach_adaptivekl005_lr1e4_tol005_ep150_bonus150_sigma015_8d_rom_20260727/`
- Chrono: `chrono_reach_benchmark_N100_seed12345/` — 97/100 at 0.05 m, median
  reached error 4.17 cm, median convergence 0.9 s, **zero** contacts and **zero**
  joint-limit violations

The end-effector is **not** a learned channel: it is recovered as `FK(q)` from the
predicted joints, using the same batched forward kinematics that the safety
shield already evaluates each step. Geometry lives in
`artifacts/arm_geometry/arm_geometry_v1.json` (regenerate with
`scripts/preprocess/extract_arm_geometry.py`); FK and the clearance shield are
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
| `hmmwv_cotrain_training.pdf` | `scripts/figures/plot_l8_training_curves.py` |
| `hmmwv_rl_reward.pdf` | `scripts/figures/plot_l8_rl_reward.py` |
| `hmmwv_policy_transfer_bars.pdf` | `scripts/figures/plot_l8_policy_transfer_bars.py` |
| `hmmwv_policy_trajectories_grid.pdf` | `scripts/figures/plot_l8_policy_trajectories_grid.py` |
| `tracked_arm_training.pdf` | `scripts/figures/plot_tracked_arm_training.py` |
| `tracked_arm_rl_reward.pdf` | `scripts/figures/plot_tracked_arm_rl_reward.py` |
| `tracked_stress_trajectories.pdf` | `scripts/figures/plot_tracked_stress_trajectories.py` |
| `arm_stress_trajectories.pdf` | `scripts/figures/plot_arm_stress_trajectories.py` |
| `arm_fk_boxes.pdf` | `scripts/figures/plot_arm_fk_boxes.py` |
| imagery in `study-case-2.pdf` | `scripts/figures/compose_tracked_arm_multiexposure.py` |

`fpp.pdf`, `hmmwv-nnrom.png` and the `study-case-2.pdf` layout are hand-drawn and
live only in the manuscript repo.

Appendix A throughput numbers come from `scripts/throughput/probe_sim_fps.py` (Chrono rows)
and the `Perf/total_fps` scalar in each PPO run's tfevents (NN-ROM rows). The
k=16 context claim comes from `scripts/throughput/bench_context_accuracy.py`, and the
6.8x collection speedup from `scripts/throughput/sweep_env_context.py`.

---

## Known gaps

1. **The CRM evaluation reference set is missing.** All three L8 runs' CRM evals
   point at
   `artifacts/rl_reference_sets/hmmwv_crm2000_val_refs_20_1100_rest_start_min10_seed20260623.npz`,
   which is not on the filesystem. The recorded results are intact, but the CRM
   column cannot be re-run until it is rebuilt with
   `scripts/preprocess/build_crm_rl_references.py` from `datasets/hmmwv_crm_2000`
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

## Follow-on project: NRD (vision in the loop)

Beyond the manuscript's scope. Neural Reduced Dynamics appends a learned camera
latent z2 to the explicit state z1 and predicts both jointly; plans live in
`docs/vision/` and the first study (Chrono double pendulum with a Chrono::Sensor
camera) is implemented — collector `src/nedm/double_pendulum_data.py`, the
`src/nedm/nrd/` package, `configs/nrd/`, and
`scripts/evaluation/eval_nrd_dpend.py`. Status and gotchas:
`docs/vision/double_pen/implementation_notes.md`.

**Reaching RL inside the frozen NRD (2026-08-26).** `DPendNRDReachEnv`
(`src/nedm/rl/dpend_nrd_reach_env.py`, rsl_rl VecEnv, resets from a bank of
recorded 16-step `[z1, z2, a]` windows, decoder never called) plus
`scripts/training/train_dpend_nrd_rl_reach.py` and the paired NRD/Chrono
evaluator `scripts/evaluation/eval_dpend_nrd_rl_reach.py`. The task plan's
distance-shaped reward was exploitable (spinning past the 35 rad/s guard was
cheaper than surviving) and plateaued at 13–17 % even after charging failures;
the arm reach study's recipe (exponential reach reward, action-rate penalty,
success bonus, no termination charge; lower-half goals, 2 cm) trained a
state-only policy to 85 % in-NRD that transfers to Chrono with **no gap: 87 % /
87 %** on 100 held-out pairs (84 shared successes, closest-approach medians
15.2 / 15.3 mm). Interactive Chrono viewer with goal markers:
`scripts/evaluation/visualize_dpend_nrd_rl_chrono.py`. Notes:
`docs/vision/double_pen/rl_implementation_notes.md`.

**Teacher–student distillation to a camera-only policy (2026-08-26).** The z1
policy was distilled by online DAgger (`src/nedm/rl/dpend_distill.py`,
`scripts/training/distill_dpend_nrd_student.py`) into a student that observes
only four normalized camera latents 0.1 s apart plus the goal (258-D). On the
identical held-out NRD pairs the student matches the teacher on all three
seeds — **88 / 90 / 87 % vs 87 %** (unselected last checkpoints 88 / 86 / 85 %),
action MAE ≈ 0.02, no spin/OOD — passing every acceptance criterion of the plan.
Wired to the real plant (Chrono::Sensor frame → frozen encoder → z2 history →
student, `scripts/evaluation/visualize_dpend_student_chrono.py`, markers
verified invisible to the sensor) it reaches 9/10 consecutive goals, the same
miss as the teacher. Notes:
`docs/vision/double_pen/distillation_implementation_notes.md`.

---

## Follow-on project: learned terrain-risk route planner (f104 arena)

Beyond the manuscript's scope; worktree `traverse_mppi`. An HMMWV in Chrono must reach a goal across one fixed
80 x 80 m arena of hills and craters (arena "f104", `assets/traverse/arena_f104_50h_v1`). The work deliberately
overfits one arena first. Full records live next to the data: `artifacts/traverse/fdm_f104_50h_20260909/`
(`night_v1/LOG.md`, `night2_v1/{PLAN,LOG,REPORT}.md`, `gen_v1/{PLAN,LOG}.md`). The written records, result
summaries, figures, final checkpoints and mission definitions are in git (committed 2026-09-16); the bulk data
(per-run folders, tensors, rendered frames, most videos) stays local and on the cluster.

**Pipeline (inference).**

1. *Map.* One vehicle-free overhead depth image of the arena (`static_map_v1/`, 512 x 512, 0.187 m/px), or the
   arena heightmap encoded the same way (`scripts/gen_planner.set_map`; identical pick on 32/40 f104 pools, top-5 on
   40/40). There is no onboard perception.
2. *Route proposal* (`scripts/f104_n2_sampler.py`). 256 candidates from the straight start-to-goal route: 9 fixed
   routes (lateral offset 0/-4/+4 m x 2/4/6 m/s) plus random ones built from a 3-mode sine lateral offset (capped
   by the 0.125 /m curvature limit) and 4 speed knots with no forced slow-down except a 2 m/s^2 stopping cone at the
   goal. Candidates that break curvature, acceleration or arena limits are rejected.
3. *Risk model* (`scripts/gen_riskmodel.py`, weights `night2_v1/final/N2_s{0..4}.pt`, 256,677 parameters each,
   5-seed ensemble). Input per candidate: a 6 x 96 x 32 corridor (96 stations along the route x 32 lateral samples
   over +-6 m; channels height, along-path grade, cross-slope, commanded speed, valid mask, constant) plus 5 numbers
   (goal dx, dy, distance, start heading, route length). No vehicle state. Architecture: 4 conv layers that never
   pool along the route -> lateral mean+max -> per-station features + context + position -> Conv1d(k=5) ->
   BiGRU(64) -> hazard logit per station; P(unsafe) = 1 - exp(-sum softplus). Trained with a discrete-time survival
   loss on "unsafe" = did not reach goal OR slid backwards under throttle (after a 1 s settle).
4. *Selection.* argmin predicted risk (no time term). 0.4 s for the whole plan on the workstation GPU.
5. *Execution.* Chrono's stock `ChPathFollowerDriver` on a Bezier through the waypoints (steering PID look-ahead
   5 m gains 0.8/0/0, speed PI 0.6/0.05), 20 Hz, 2 ms physics. Route chosen once; no replanning (except the
   multi-goal missions below, which replan at each goal).

**Data.** 36,199 driven routes (about 200 h simulated) over 2,700 start/goal groups on f104; the deployed model
trained on 31,851. Waves: original 1,500 groups x 12 designed routes (66.4 h, `production_v2`), 1,200 new groups x 12
designed routes (`production_v3`), 9,309 routes drawn from the planner's own proposals (`production_v4`). Tensors:
`night2_v1/station_ds_all.npz`. Collection is cheap: the 66.4 h wave took 13.6 wall minutes on ~2,000 cluster workers.

**Milestones (Chrono, paired arms in one cluster job; unsafe = failed or slid back).**

| Date | Test | Result |
|---|---|---|
| 09-10 | 123 held-out groups, 2 m/s geometry-only planning | new station-preserving model: failed 15.4% -> 6.5%, unsafe 44.7% -> 17.9% |
| 09-12 | 300 hill/crater groups, speed free (`night2_v1/closed_haz`) | night-1 planner -> night-2 planner: failed 3.7% -> 0.3%, unsafe 9.7% -> 0.3% (28 vs 0, p < 1e-8) |
| 09-12 | same, speed fixed at 2 m/s, only the model differs | unsafe 4.3% -> 0.7% (12 vs 1, p = 0.003) |
| 09-12 | tilt added to the label (`N2T_s*`) | runs past 30 deg 22 -> 0 of 300 |
| 09-14 | 15 demo videos (`artifacts/f104_demo_v1`) | model's best / ~10% / worst route per scenario: 0/26, 4/26, 26/26 unsafe |

What the gains are made of (09-12 decomposition of the 29 unsafe control runs): 29 -> 13 by letting candidates
swing wider (old model), -> 2 by the new model on the same candidates, -> 1 by also letting them approach fast. The
biggest single factor is speed/momentum.

**Honest status after the 09-14 audit (9 agents + 3 verifiers).**

- The pipeline works, but with speed free the deployed planner never beat "always drive 6 m/s straight" on its own
  label on f104 (failed 1 vs 1, unsafe 1 vs 4, p = 0.375). Its wins over simple rules are at a fixed 2 m/s.
- Offline the model is more than a speed reader or a memorised map: same-speed ranking ~0.95 vs 0.88 for the best
  lookup; scored zero-shot on 6,633 routes from 24 other arenas it keeps within-start/goal ranking 0.90 (f104 bank
  0.94), 0.87 on mirror-image left/right detours where shape/speed rules score 0.5. But its confident tail does not
  transfer (catch rate at 5% false alarms 0.87 -> 0.51) and its lead over a hand terrain+speed rule is only
  +0.04-0.08.
- Known errors in the night-2 report: "the two models never picked the same route (523 groups)" is false (117 were
  identical); the 6 m/s baseline's worst tilt on the hazard set is 59.5 deg (a rollover), not 45.9.
- The model ranks well but is badly calibrated (median predicted 0.008% vs realised 0.33%): read scores as an ordering.

**Where things are.**

| What | Path |
|---|---|
| Arena, static map | `assets/traverse/arena_f104_50h_v1`, `artifacts/.../static_map_v1/` |
| Frozen collector + generator | `scripts/collect_traverse_f104.py`, `scripts/generate_traverse_f104_collection.py`; cluster copy `source_v1/` |
| Labels + tensors | `scripts/f104_n2_dataset.py` (cluster), `scripts/f104_n2_merge.py` -> `night2_v1/station_ds_all.npz` |
| Training | `scripts/f104_n2_train.py` (architecture x state sweep), `f104_n2_final.py` (scaling), `f104_n2_deploy.py` (ensemble) |
| Deployed model | `night2_v1/final/N2_s*.pt` (tilt-aware variant `N2T_s*`); night-1 control model `night_v1/final/H1_full_s*.pt` |
| Planner as one module | `scripts/gen_planner.py` (map, proposal, corridor, model, hand rule, plan) |
| Night-2 closed-loop tests | `scripts/f104_n2_{cand,pick,analyze,ext*,haz*,tilt*}.py`; results `night2_v1/closed*/results.json` |
| Demo videos | `scripts/f104_demo_*.py`; `artifacts/f104_demo_v1/` |
| Cluster campaign | `/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909` (`production_v2..v4`, test run dirs, `gen_v1/`) |

**Cleanup (2026-09-15).** Kept everything needed to explain and re-derive the results above; deleted what was
rebuildable or superseded. Manifests with sizes: `artifacts/.../CLEANUP_2026-09-15_{local.tsv,scripts.txt}` and the
cluster's `CLEANUP_2026-09-15_cluster.tsv`.
- Local 36 GB -> 4.8 GB: candidate caches (`testcand*`, night-1 `cand/`, 18 GB, rebuildable from md5 seeds),
  intermediate tensors (`station_ds_{fix,A1,A2,tilt}`, night-1 `station_ds`, feature caches), rendered frames and
  Blender exports (videos kept), the superseded first video set, early 09-10 risk-head dataset, unused start/goal pools,
  and the per-run rich telemetry of the local training-data copy.
- Cluster 52 GB -> 10 GB: Blender exports, the aborted first production attempt and pilots, and per-run rich
  telemetry everywhere (labels only need `trajectory.npz`, `outcome.json`, `command_reference.npz`, `case.json`; note
  `episode_complete.json` still lists the deleted files' hashes).
- 42 superseded or one-off f104 scripts removed (untracked; archived at
  `~/NeDM-archive/traverse_mppi_removed_f104_scripts_2026-09-15.tar.gz`, 62 KB). Kept scripts all import cleanly.
- Not touched (other research lines, for a separate decision): locally `artifacts/traverse/fdm_rgbd_*`,
  `fdm_diverse_v1_20260909` (~8 GB), `training_runs`, `rl_runs`; on the cluster `fdm_diverse_v1_20260909` (14 GB),
  `traverse_mppi_20260908` (7.2 GB), `state_ablation_20260908` (5.8 GB), `mppi_claude_20260908` (1.9 GB),
  `fdm_failure_investigation_20260909` (1.7 GB); the `~/NeDM-mppi-claude` worktree.

**Overnight 2026-09-15: longer tasks and new arenas** (`gen_v1/`, pre-registered in `gen_v1/PLAN.md`, report
`gen_v1/REPORT.md`, figure `gen_v1/results.png`; model frozen).

- *Five goals in a row* (one continuous simulation, replanning from the measured pose at each goal; 100 missions on
  f104, 100 on new arenas). f104: the model's planner completed **99%** of missions vs 91% for a hand terrain+speed
  rule and 90% for a 6 m/s straight line (10 vs 1, p = 0.012; 8 vs 0, p = 0.008), slid in 2% vs 7% / 11%, but took
  54 s vs 36 s / 30 s. New arenas: 94% / 89% / 94%, slides 7% / 15% / 16%, leaned past 30 deg 9% / 21% / 42%.
- *Five never-seen arenas* from the same terrain generator (closest 5 of 40 seeds), 200 hill/crater start/goals each.
  Single goal, failed or slid: model 1.3%, hand rule 1.8%, straight 6 m/s 4.5% (model vs rule p = 0.42 — the
  pre-registered primary test was null; vs straight p < 1e-5). At a fixed 2 m/s: 5.9% vs 13.6% vs 59.8% (model safer
  than the rule on all 5 arenas). Offline ranking on 9,000 routes collected there: 0.955 within a start/goal, 0.905 at
  matched speed (f104 held-out: 0.991 / 0.985; hand rule 0.902 / 0.778). The model is ~4 s slower per goal.
- *Data*: 15,639 labelled routes on the new arenas, tensors `gen_v1/station_ds_gen_v1.npz`, not yet used for training.
- Code: `scripts/gen_*.py` (planner module, cases, pools, missions, cluster runners, analyses, figures).

**Sensor channels as the network input, 2026-09-15** (`sensor_v1/`, pre-registered `PLAN.md`, `REPORT.md`).
One overhead RGB-D capture per arena (`scripts/sensor_capture_map.py`; the f104 re-capture is byte-identical to the
original), ten-channel corridors (`scripts/sensor_dataset.py`), same network/rows/schedule trained on the cluster
(`scripts/sensor_train.py`), planner support in `scripts/gen_planner.py` (`set_sensor_map`, `SensorRiskModel`).
- Raw depth (+ camera ray angle) matches the current model on f104 but is significantly worse on the five new arenas,
  replicated on two fresh 1,200-start/goal Chrono tests (2 m/s unsafe 9.4% vs 7.2%, then 9.3% vs 5.7%).
- Depth converted to height with the known camera intrinsics, without the hand-made slope channels (E0,
  `sensor_v1/final/E0_s*.pt`): not different from the current model (2 m/s 6.5% vs 5.7%, p = 0.30; speed free
  1.0% vs 0.75%, non-inferior), though the 2 m/s upper bound (+2.25) misses the declared +2.0 margin.
- Colour channels hurt transfer to new arenas offline (within-start/goal ranking 0.85-0.87 vs 0.94-0.95).

**Corrected depth->world pipeline and matched height/depth training, 2026-09-15** (`sensor_v2/`, `REPORT.md`;
prompted by the independent review in `artifacts/reviews/sensor_input_20260915/`).
- `scripts/sensor_map_v2.py` back-projects every depth pixel with the camera intrinsics into a metric world grid
  (the old sampler assumed flat ground). Height error against the authored heightmap falls from 0.027-0.044 m to
  0.0071-0.0087 m (0.0008 m per pixel against the terrain Chrono actually simulates); verified by two independent
  re-implementations against orientation, intrinsics, registration, aggregation and edge tests.
- Repo-level finding: `TerrainMap`, the privileged height oracle, is offset from Chrono's `RigidTerrain` by a pure
  511/512 radial scale (up to 0.078 m at the arena edge). Every previous "authored height" reference inherits it.
- Matched training (identical rows/labels/seeds/budget, split by whole arena, train f104+g228+g203+g217, held out
  g216+g231), primary metric route choice at matched speed: height + slopes 14.50% unsafe picks, height only 14.50%,
  absolute range + ray secant 14.92% (+0.42 [-0.42, +1.25]), relative range (the old depth arm) 15.58%
  (+1.08 [+0.25, +2.00]). Direct depth matches height once the geometry is corrected and absolute range is kept.
- Chrono pilot (200 held-out-arena start/goals, 1,066 drives): no difference resolvable - 2 m/s unsafe 6.0% height vs
  5.0% depth vs 7.5% old-depth, speed-free 0.5/0.0/0.5%, identical travel time. Geometric accuracy improved 4-10x; a
  planning gain is not demonstrated. Next: ~1,200 fixed-speed start/goals on genuinely new arenas.

**Vehicle-included depth input (pilot, 2026-09-15)** (`sensor_v2/VEHICLE_PILOT.md`). One overhead RGB-D frame per
planning decision with the HMMWV present after the settle (`scripts/vehicle_capture.py`), a footprint-based exclusion
zone at the measured pose (`scripts/vehicle_corridor.py`), corridor coordinates preserved and hidden ground never
filled. Measured: the vehicle plus its shadow reaches at most 0.50 m outside the bare footprint, so a 1.5 m margin
covers it; 5.2% of corridor samples become invalid. Vehicle-included + mask is identical to vehicle-free + mask
(0.0000 m, same pick 20/20). 63 drives: 0 unsafe and 0 failures in both conditions. On 1,200 labelled choices the mask
costs +0.33 points (height) / -0.09 (depth). Preprocessing works; retraining not justified yet.

**Continuous sensor-driven waypoint navigation (nav_v1, 2026-09-16)** (`nav_v1/REPORT.md`, `PLAN.md`, `LOG.md`,
`RUNNING.md`). One Chrono rollout per mission, the vehicle never reset: at every planning decision the simulator
renders one overhead depth frame **with the HMMWV in it**, back-projects it to the metric grid, masks the vehicle's
own footprint, builds 256 candidate corridors from that single frame and hands the lowest-risk route to the path
follower (`scripts/nav_runner.py`, `scripts/nav_online.py`). 30 missions on 10 arenas (6 development + 4 never seen:
g213, g204, g234, g223), 5-8 waypoints, 150-228 m; 120 rollouts, 5,043 decisions, 29.9 km driven.

- **It works.** One decision per waypoint completes 27/30 missions and 96.5% of waypoints, and **16/16 on the four
  arenas never seen before**. The random-pick control with the identical loop completes 6/30 against 24/30,
  reaches 45.2% of waypoints against 87.9%, takes 166 s a mission against 76 s, and slides backwards in 28 of 30.
- **Replanning every 1-2 s did not improve completion in the AMD campaign, but that comparison is confounded by a
  runner bug (found 09-16).** Plan-once completed 27/30, stalled on 3 and never left the terrain; every 2 s completed 23/30, every 1 s 24/30 and every 1 s with the planning delay charged 23/30, stalling on 2 each but with 4-5 runs each that drove off the arena. **All 14 of those exits followed a rescue route that doubled back on itself**: when no normal candidate was valid, the stand-in goal
  could sit directly behind the vehicle, and the route checker scored the resulting sharp reversal as curvature 0.
  Replanning asks for rescue routes far more often, so the exits landed on those arms. The earlier explanation
  (re-anchoring at the drifted pose with no boundary term) is withdrawn. Fix: routes that turn more than 45 deg
  between consecutive points are rejected; the fixed 120-run re-run is in progress on luffy
  (`nav_v1/local_luffy/`). Legs with a backward slide are the same across arms. Travel time favours replanning by 3-6 s per mission when planning
  is free and goes the other way (+3.1 s) once the measured planning delay is charged; every CI includes zero. The
  value of replanning is not demonstrated with the whole arena visible at every decision — which is the condition
  these runs are in.
- **Latency, separated from the simulator.** Per decision on a campaign node: 2.6 s (median) of Chrono's software depth render plus 1.5-1.8 s of planner. The planner alone is **0.42 s (2.4 Hz) on an MI350X** and 0.71 s on that
  node's CPU, and the risk network is only 42 ms of it; candidate generation and corridor extraction dominate.
  The `R1L` arm charges the measured algorithmic latency back to the simulation.
- **Rest vs moving needs no retraining.** The model reads no vehicle state, so a route scores the same parked or at
  6 m/s; 83% of decisions (every-2 s arm) are taken above 2 m/s and 10 slide events follow 802 periodic decisions
  taken while moving (1.2%, the same rate as at waypoints). The dangerous group is decisions taken when the vehicle
  has *already* dropped below 2 m/s (78 of 147 followed by a slide), which no route choice fixes.
- **Limited sensing range is where retraining pays.** Cropping the frame to a radius around the vehicle, measured on
  1,200 labelled choices: avoidable unsafe picks go 2.3% (whole arena) -> 5.7% (30 m) -> 10.0% (20 m) -> 11.8%
  (15 m) against 17.3% for random. Retraining the whole matched pipeline on 20 m corridors recovers 1.25-1.33 points
  of that (9.58 -> 8.33% depth, 10.00 -> 8.67% height): the loss is mostly missing terrain, not distribution shift.
- **Six infrastructure defects were found and fixed before the reported campaign**, each one hitting the replanning
  arms harder for reasons unrelated to planning (LOG.md has every measurement): the path follower's speed integrator
  resets on every route change; carrying it then needs anti-windup; a waypoint can be unreachable by every shape the
  frozen route builder tries; rescue arcs at the 8 m minimum radius are untrackable at 4 m/s; routes were allowed to
  touch the terrain edge; and a fixed planning margin cripples the planner once the vehicle is outside it. A
  seventh, the doubled-back rescue routes above, was found after the campaign.
- **Depth render latency** (`render_latency_v1/README.md`). The 2.6 s per frame on the cluster is Chrono's Vulkan
  ray tracing running on a CPU software driver (~1.9 s) plus Chrono rebuilding the whole ~800k-triangle scene every
  frame (~0.8 s). The same 1024x1024 depth frame takes 7 ms with OptiX on the workstation's RTX 5090 (user's
  source-built Chrono with the depth-FOV fix) and matches the cluster's depth to 5e-5 m.
- **Local re-run on luffy** (OptiX; `scripts/nav_local.py`, `scripts/nav_local_batch.py`; `nav_v1/LOG.md`). Per
  decision 0.9-1.1 s with six runs sharing the CPU, now dominated by candidate generation, not rendering. The first
  local re-run (before the fix, `local_luffy_v0_foldback/`; it had the same bug, so it is not independent confirmation) gave the same picture as the cluster: plan-once 27/30
  (16/16 unseen), every 2 s 25/30, every 1 s 24/30, delay-charged 23/30.

**Gaps in the committed record (checked 2026-09-16, before the first push of this branch).** What git holds for
this project is the written records, result summaries, figures, final checkpoints (LFS) and mission definitions.
These things are *not* checkable from git alone:

- *Start/goal definitions of the night-2 test groups* (1,007 groups): only the group IDs are committed
  (`night2_v1/*_test_groups.json`). The case files are local (`cases_{test,ext,haz}_final/`) and on the cluster;
  the 4,000-group pools they were drawn from were deleted on 09-15.
- *Removed scripts*: the 15 night-1 analysis scripts deleted in the 09-15 cleanup (including the night-1 checkpoint
  scorer and the only loaders of `night_v1/final/A1_full_s*.pt`) exist only in the archive tarball named above.
- *Numbers with no saved output file*: the 09-14 audit's offline ranking figures (0.95 vs 0.88, the 24-arena
  zero-shot scores); night-1 LOG steps 3, 8 and 11; night-2's fourth-audit re-runs; the sensor_v2 matched
  3-seed ensemble table (14.50 / 14.50 / 14.92 / 15.58%; only per-seed rows were saved, and `matched_H.json`
  holds seed 2 only); the vehicle-pilot mask cost (+0.33 / -0.09); the gen_v1 heightmap-vs-depth-map check
  (32/40 identical picks; the `gen_planner.py` docstring quotes different figures); the nav_v1 controller checks
  in its REPORT section 2; the nav_v1 leak re-check in its LOG (0.078 m, 6.9 m/s, 82%).
- *nav_v1 per-run files* (`main/runs/*/{decisions,routes}.json`, trajectories; local and cluster): only the
  per-run outcomes are committed, so the rescue-route extent behind the correction is recorded only in the saved
  output `videos/_superseded_v0_foldback/_audit/completeness_critic/c6_cusp_eval_stdout.txt`.
- *Videos and case files left out on purpose*: 12 of the 15 f104 demo videos (scenario 1 is committed), the
  per-route four-route MP4s of the RGB-D line, and the f104 collection `cases/cases.json` files that
  `fdm_f104_50h_collection_20260909.md` links.
- *Cluster job scripts*: the nav_v1 and night-2 test `.sbatch` files (and the night-2 runners) are copied as run into
  `scripts/`; they point at cluster paths and at the campaign's frozen code copies.
- *Cluster only*: nav_v1's random-pick control (`R1rand`) and the checkpoints retrained on 20 m corridors
  (`nav_v1/r20_choice.json` is the committed record); production_v3/v4 and the frozen `source_v1/`.
- *Earlier RGB-D / MPPI line* (`docs/vision/hmmwv_traverse/`): `protected_test_offline_final_v2/report.json`
  (10.3 MB, its SHA is in `complete.json`) and the matched blank-image checkpoint (AMD only) are not committed; the
  43.91% contact risk quoted for the smooth-hill straight route is only in the uncommitted `candidates.json`.
  `mppi_document_index_20260909.md` is a local-machine index: its links are absolute paths on the workstation and
  55 of them point at files that are not committed (it is left unedited because its hash is recorded in
  `mppi_documents_20260909/verification.json`).

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
