# NeDM — Neural Reduced Dynamics for Complex Robot Control

Code and artifacts for *Learning the Right Abstraction: Neural Reduced Dynamics
for Complex Robot Control* (Zhang and Negrut).

**Project page: <https://uwsbel.github.io/NeDM/>** — figures, and the nine
side-by-side Chrono rollouts behind the Study Case I result. Its source is
[`web/`](web/README.md).

High-fidelity Chrono trajectories are distilled into a task-specific neural
reduced dynamics model (NN-ROM); the model is frozen and replicated into a
vectorized environment where a control policy is trained with PPO; the trained
policy is then returned to the full Chrono system for closed-loop validation.

Two study cases instantiate the pipeline across three control tasks:

- **Study Case I** — terrain-aware HMMWV trajectory tracking on rigid, bumpy and
  deformable CRM terrain. A 15-D reduced state carries body motion plus a
  per-wheel terramechanics block; a two-class terrain code resolves the
  rigid-vs-CRM ambiguity. One policy trained inside the conditioned model beats
  both single-terrain specialists on all three terrains, including zero-shot
  bumpy.
- **Study Case II** — an M113 tracked vehicle with a front-mounted 4-DOF arm,
  driven by two independent tasks. A 3-D planar state serves base goal reaching
  (100/100 goals in Chrono at 0.75 m); an 8-D joint-space state serves arm
  end-effector reaching (97/100 at 0.05 m, zero contacts or joint-limit
  violations), with the end-effector recovered by forward kinematics rather than
  learned.

**[docs/progress.md](docs/progress.md) is the reproduction record** — every stage
output with the artifact that produced it and the command that regenerates it.
Start there.

## Environment

```bash
conda env create -f environment.nedm.yml
conda activate nedm
git lfs install && git lfs pull
```

`environment.nedm.yml` (env `nedm`, pychrono 10.0.0 from the `projectchrono`
channel) is what everything runs in: data collection, training, RL, and
Chrono-backed evaluation. `environment.yml` (env `tutorial`, pychrono 9.0.1) and
`environment.lock.yml` are retained for the earliest datasets, which were
collected under it.

Project Chrono itself is a local dependency, not vendored — collection configs
expect a checkout at `chrono/` and read `chrono/data` for vehicle assets.

## Layout

| Path | Contents |
|---|---|
| `src/nedm/` | Chrono scene builders and data collectors (`hmmwv_data`, `hmmwv_crm`, `arm_data`, `tracked_vehicle_data`) |
| `src/nedm/training/` | Preprocessing, the causal-transformer dynamics model, and the trainer with rollout-based checkpoint selection |
| `src/nedm/rl/` | Vectorized NN-ROM environments and their Chrono-backed twins, plus arm forward kinematics and the clearance shield |
| `src/arm_model/` | The 4-DOF gripper arm imported from SolidWorks |
| `configs/` | Collection and training configs |
| `artifacts/` | Checkpoints, run metadata and Chrono evaluation output (datasets are on Hugging Face, see below) |
| `test/` | Chrono validation harnesses for the tire-force channels (not a unit-test suite) |

`scripts/` is organised by pipeline stage, in the order you would run them:

| Path | Contents |
|---|---|
| `scripts/collection/` | Shard planners and Chrono collectors for the five datasets, plus their validators and small-scale smoke tests |
| `scripts/preprocess/` | Raw episodes → processed caches; RL reference-set builders; arm FK geometry extraction |
| `scripts/training/` | The dynamics trainer, the three PPO trainers, and the launchers holding each run's exact hyperparameters |
| `scripts/ablations/` | Config generation, sweep runners and ranking for Appendices C–E and the specialist comparison |
| `scripts/evaluation/` | Open-loop rollout eval, Chrono closed-loop transfer, and the seeded 100-goal benchmarks |
| `scripts/figures/` | The eleven generators behind the manuscript's plotted figures |
| `scripts/throughput/` | Chrono and NN-ROM throughput probes (Appendix A) and the context-truncation sweep |
| `scripts/cluster/` | SLURM array jobs for the collections that only run at cluster scale |
| `scripts/release/` | The Hugging Face dataset release: raw CSV → Parquet export, validation, upload, and the download/rehydrate helper |

Every script under `scripts/` reproduces something the paper reports; nothing else is
kept. The ablation *artifacts* and *configs* keep their original `ablation_ofat` name
because it is recorded inside the run metadata.

## Datasets

All five datasets the paper's dynamics models train on are published at
**<https://huggingface.co/datasets/harryzhang1018/NeDM>** (70 GB: every recorded channel as
float32 Parquet plus the four processed training caches; the dataset card documents schemas,
splits and provenance). Nothing needs to be re-collected:

```bash
conda activate nedm
# the exact .npy caches the deployed models trained on -> artifacts/training_datasets/
PYTHONPATH=src python scripts/release/download_nedm_datasets.py --dataset all --no-raw --processed
# a raw dataset as the collectors' per-episode CSV tree -> artifacts/datasets/ (preprocess etc. run unchanged)
PYTHONPATH=src python scripts/release/download_nedm_datasets.py --dataset tracked --rehydrate
```

`docs/hf_dataset_card.md` is the source of the Hub README; `scripts/release/export_hf_dataset.py`
+ `validate_hf_export.py` + `upload_hf_dataset.sh` regenerate and publish the release.

## Quick start

Collect a small dataset, build its cache, and train:

```bash
conda activate nedm
python scripts/collection/collect_hmmwv_dataset.py --config configs/hmmwv_overfit_v1.json
python scripts/preprocess/build_hmmwv_training_dataset.py --help
PYTHONPATH=src python scripts/training/train_hmmwv_dynamics.py \
  --config configs/hmmwv_transformer_v07_tire_normal_force_omega_300g_crm2000_mix25_rebal_rollout_onehot.json
```

The full flat collection is cluster-scale (~305 GB of CSV; download it from Hugging Face
instead, see above); `scripts/cluster/collect_hmmwv_tire300g.sh` is the job that produced it and
`scripts/collection/smoke_test_hmmwv_bumpy10g.sh` rehearses the same path at small scale.

Evaluate a trained policy back in Chrono:

```bash
PYTHONPATH=src python scripts/evaluation/eval_hmmwv_rl_chrono_tracking.py --help    # Study Case I
PYTHONPATH=src python scripts/evaluation/benchmark_tracked_goal_chrono.py --help    # Study Case II, base
PYTHONPATH=src python scripts/evaluation/benchmark_arm_reach_chrono.py --help       # Study Case II, arm
```

## Further reading

- [docs/model_checkpoints.md](docs/model_checkpoints.md) — the deployed checkpoints and how to load them
- [docs/data_collection_pipeline.md](docs/data_collection_pipeline.md), [docs/hmmwv_crm_data_collection_pipeline.md](docs/hmmwv_crm_data_collection_pipeline.md) — collection design
- [docs/hmmwv_training_pipeline.md](docs/hmmwv_training_pipeline.md), [docs/rl_tracking.md](docs/rl_tracking.md) — Study Case I
- [docs/arm-dyn-model.md](docs/arm-dyn-model.md), [docs/arm-reaching-rl-plan.md](docs/arm-reaching-rl-plan.md), [docs/tracked_vehicle_nn_rom_rl_plan.md](docs/tracked_vehicle_nn_rom_rl_plan.md) — Study Case II
- [docs/ablation_ofat_plan.md](docs/ablation_ofat_plan.md) — the ablation protocol
- [deep-research-report-vehicle.md](deep-research-report-vehicle.md) — background survey on learned vehicle dynamics
