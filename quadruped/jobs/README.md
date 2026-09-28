# Job scripts behind the reported results

The exact scripts that produced the numbers in `../README.md`, `../docs/STATE.md` and on
the site, copied from the machines they ran on. Paths inside are machine-specific
(`/work1/dannegrut/kyle/qrun` on hpcfund, `/srv/home/kasha2/qrun` on euler, `~/qrc` on
the desktops); each script's header says what it was for. Corpus and scoring work that
must pair (one Chrono build) ran on one machine each; training and PPO are torch-only.

## hpcfund (AMD MI210/MI300X; Chrono HIP build c716f05e)

| Script | Produced |
|---|---|
| `collect_v2_array.sbatch`, `merge_v2.sbatch` | the normal corpus, go2_crm_v2 (24 shards x 50) |
| `train_v2_mi300.sbatch`, `train_ms100_mi210.sbatch` | standard surrogates (seeds 6-9, north): one-step, then 1 s rollout training |
| `train_v23.sbatch` | standard surrogates on the extended corpus (seeds 40-43) |
| `ft_data.sbatch` | the recipe (PPO, 1024 x 2 s, iteration 1000): ten-surrogate re-score, extended corpus |
| `ft_long.sbatch` | the stopping study (to iteration 3000, snapshots every 500) |
| `ft_envs.sbatch` | the rollout-count study (64-2048) |
| `ft_kick.sbatch` | disturbance training in the surrogate |
| `ft_seeds.sbatch` | PPO-seed test in seed 6's surrogate |
| `ft_risk.sbatch`, `ft_risk2.sbatch` | OOD penalty off; induced failures for the guard |
| `eval_paths.sbatch`, `eval_packed.sbatch` | Chrono scoring: 40 held-out paths; 16 straight |
| `eval_push.sbatch` | the push test (8 directions x 2, per force) |
| `cost_profile.sbatch` | the Chrono cost profile |
| `probe2.sbatch` | out-of-distribution probe along seed 6's checkpoints |

## euler (NVIDIA A100/H100)

| Script | Produced |
|---|---|
| `train_v2_a100.sbatch`, `ms100_a100.sbatch` | standard surrogates, euler seeds 2-5 |
| `rec1024_a100.sbatch`, `long_a100.sbatch` | recipe replication; stopping-study replication |
| `env2048_a100.sbatch` | 2048-rollout run |
| `scale_a100.sbatch`, `ft_scale_a100.sbatch` | model size and less-data surrogates, and their fine-tunes |
| `large_v23.sbatch` | 12x512 surrogate on the extended corpus + the four-cell accuracy sweep |
| `channels.sbatch` | the channel study (4 presets x 3 seeds, train + fine-tune) |
| `ft_rigid_h100.sbatch` | rigid-data control fine-tunes |
| `onestage.sbatch`, `ft_onestage.sbatch` | one-stage training, its accuracy sweep and fine-tunes |

## desktops (sbel: RTX 3090, Chrono 3b0bd530; d33: RX 9070 XT, Chrono 53102025)

| Script | Produced |
|---|---|
| `collect_rigid.sh`, `train_rigid.sh`, `ft_rigid.sh` | the rigid corpus (same seeds as v2) and rigid surrogates |
| `score_crm_sb.sh`, `score_rigid_sb.sh` | rigid-data control scoring on sbel (CRM and rigid paths) |
| `analytic_grid.sh`, `score_d33_now.sh`, `score_ppo_d33.sh` | the analytic-vs-PPO grid and its scoring on d33 |
