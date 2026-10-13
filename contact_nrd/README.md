# Contact NRD: bouncing ball, pool, SO101 push-T and G1 locomotion

A neural reduced dynamics (NRD) model for systems whose contacts start and stop. The four study cases use the same
network design and the same training. The first three cases share one config file; only the data changes. G1 uses
the same design and training with `ridge` 0.1 and selects its checkpoint by a task score
([G1 differences](#g1-differences)).

| Study case | System | Folder |
|---|---|---|
| Bouncing ball | One ball bounces on a floor and hits a wall | [`bouncing_ball/`](bouncing_ball/README.md) |
| Pool | Ball A hits ball B; both balls hit the cushions | [`pool/`](pool/README.md) |
| SO101 push-T | A 5-joint SO101 arm pushes a T-shaped block on a table | [`so101_push_t/`](so101_push_t/README.md) |
| G1 locomotion | A frozen walking policy drives the 27-joint Unitree G1 humanoid on a flat floor | [`g1_locomotion/`](g1_locomotion/README.md) |

The earlier studies in this repository (HMMWV, M113 and its arm) have no contacts that start and stop. They need only
a model of smooth motion, as the core network is here.

## Design

| Network | Job | Runs for each 20 ms step | Output |
|---|---|---|---|
| Core | Motion without switching contacts (flight, rolling, sliding on the table, the arm under its command) | Once for each moving body | Change of the body's speeds |
| Collision | Does the pair touch in this step or in the step before? | Once for each contact pair | On or off |
| Contact | Effect of the contact on the pair | Once for each pair that is on | Change of the speeds of the pair's bodies |

- **One design for all three networks.** Each network is a Transformer encoder: 4 blocks, width 128, 4 heads,
  feed-forward 512, about 0.8 M parameters. The three networks share no weights.
- **Inputs.** The core sees the last 4 states of its own body. The collision and contact networks see the last 4
  states of every moving body. Each history is the current state and 3 step differences. Learned codes give the body
  kind, the fixed partner (floor, wall, cushion, ground) and the role in the pair. The joint command of the arm (or
  of the G1 robot) is one more token. There are no geometry features.
- **State.** The state holds the pose and the speeds of each moving body. The networks predict only speed changes.
  Linear speeds, joint speeds and the angular speeds of the T and the G1 base are averages over the last 20 ms step.
  The ball spin is the recorded value, because a ball state has no orientation.
- **Step.** speeds' = speeds + core change + sum of the contact changes of the pairs that the collision network
  switches on. The pose comes from integration, pose' = pose + 20 ms × speed', and goes into all three networks.
  An averaged speed still holds the effect of a contact in the step before. For this reason the switch covers that
  step too.

**Training** (`configs/contact_nrd/train.json` for the first three cases; G1: see [G1 differences](#g1-differences)):

1. Core: a linear fit on free-motion steps gives the start point.
2. Collision network: 30,000 updates alone, then frozen.
3. Core and contact network: 120,000 updates together, with three losses: the one-step core error, the one-step
   contact error and the rollout error over up to 1 s. Training adds noise to the history. After the first quarter
   of the updates, half of the contact examples use 1 to 3 of the model's own steps as history.
4. Selection: every 10,000 updates, free rollouts on up to 1,500 validation episodes (all 900 for the ball) give a
   score. The best checkpoint is kept.

### G1 differences

- **Config.** `configs/contact_nrd/train_g1_locomotion.json` is `train.json` with `"ridge": 0.1` (the ridge term of
  the core's linear start fit; the shared value is 1e-6). Give it with `--config`.
- **Selection.** The G1 data carry an `evaluation` block in `system.json` (`configs/contact_nrd/evaluation/`). The
  checks score rollouts from recorded histories at fixed horizons, contact events, a fall flag and non-finite
  rollouts with it, on the validation episodes of the families with `select: true`, instead of free rollouts.
- **Data items.** The G1 data give each episode a `family` and each moving body a `channel_mask` (the channels that
  the model predicts) ([`collection_contract.md`](collection_contract.md)). Data without these items train as before.

## Results

Released checkpoints on the test episodes. The test episodes come from a separate collection (Chrono; G1: MuJoCo) with
a new seed. Training and checkpoint selection did not use them; earlier design comparisons did. In the first three
cases each rollout starts from the episode's first recorded state and runs open loop (the arm gets its recorded
commands). G1 rollouts start from the recorded 4-state history every 0.5 s of each episode and run 1 s with the
recorded joint targets. Median / p95:

| Study case | Error | Seed 61 | Seed 62 |
|---|---|---|---|
| Bouncing ball (1,800 shots) | Ball position at 1.7 s | 0.90 / 2.09 mm | 0.90 / 2.09 mm |
| | Ball path, RMS up to 1.7 s | 0.60 / 1.08 mm | 0.61 / 1.10 mm |
| Pool (4,800 shots) | Ball B position at 2 s | 0.69 / 2.41 mm | 0.64 / 2.52 mm |
| SO101 push-T (1,000 episodes) | T position at 4 s | 0.43 / 2.70 mm | 0.37 / 3.07 mm |
| | T angle at 4 s | 0.23 / 1.61° | 0.20 / 1.61° |
| G1 locomotion (2,000 task episodes) | Root position after 1 s | 6.9 / 59.8 mm | 7.2 / 60.8 mm |
| | Root orientation after 1 s | 0.68 / 6.8° | 0.68 / 6.7° |

## Reproduce

Requirements: `torch` and `numpy`, plus `huggingface_hub` for the download. `requirements.txt` installs all three. No
simulator.

```bash
# 1. Download the case (train, test and the two released checkpoints) -> artifacts/contact_nrd/pool/
PYTHONPATH=src python -m nedm.contact_nrd.download --case pool
# 2. Train one seed on one GPU (G1: add --config configs/contact_nrd/train_g1_locomotion.json)
PYTHONPATH=src python -m nedm.contact_nrd.train --data artifacts/contact_nrd/pool/train --seed 61 \
    --output artifacts/contact_nrd/runs/pool_s61
# 3. Score the retrained model and a released checkpoint on the test episodes
PYTHONPATH=src python -m nedm.contact_nrd.test --data artifacts/contact_nrd/pool/test \
    --model retrained=artifacts/contact_nrd/runs/pool_s61 released=artifacts/contact_nrd/pool/models/seed61.pt
```

Use `--case bouncing_ball`, `--case so101_push_t` or `--case g1_locomotion` (and the same folder names) for the other
cases. `--case all` downloads all four cases, the 21.4 GB of G1 included.

| Study case | Download | Training time, one MI350X | GPU memory |
|---|---|---|---|
| Bouncing ball | 0.77 GB | about 4.5 h | about 12 GB |
| Pool | 5.9 GB | about 5.0 h | about 32 GB |
| SO101 push-T | 3.2 GB | about 5.9 h | about 24 GB |
| G1 locomotion | 21.4 GB | about 5.0 h | not logged; 69 GB in a larger run (see below) |

- The times come from 8 runs that shared one 8-GPU node. They do not include the work that a resume repeats.
- G1: two runs (seeds 61 and 62), each a 4 h job and a resume of about 1 h. Their peak memory was not logged. A
  later fine-tuning run from the seed-61 checkpoint (same batch sizes, 8 % more episodes, the collision network also
  in training; one MI300X) peaked at 69 GB GPU memory and 40 GB host memory.
- A run is longer than a 4 h job limit. If a job time limit stops a run, run the same command again. Training
  continues from the last check (every 10,000 updates).
- A new run with the same seed does not give the same digits. With `--deterministic`, two runs on the same machine
  do.

## Data

Hugging Face dataset [harryzhang1018/NeDM](https://huggingface.co/datasets/harryzhang1018/NeDM), folder `contact_nrd/`,
pinned in [`release_manifest.json`](release_manifest.json) (revision and SHA-256 of each file; the G1 case has its
own revision in `case_revisions`). Each case has:

| Path | Contents |
|---|---|
| `<case>/train/system.json` | Bodies (kind, size, fixed planes), contact pairs, record step, target body and time, simulator settings. The SO101 and G1 files also give the channel names and types; the G1 file also gives the families and the `evaluation` block |
| `<case>/train/unified_data.npz` | The states (ball and pool: one `states` array for all balls; SO101 and G1: one array for each moving body), the commands, the contact labels of each pair, episode lengths, splits (0 train, 1 validation, 2 the source campaign's own test episodes, not used). G1 also: `family` and diagnostics that training does not read |
| `<case>/test/` | The same files for the test episodes (split 2) |
| `<case>/models/seed61.pt`, `seed62.pt` | The released checkpoints |
| `g1_locomotion/{train,test}/manifest.json` | G1 only: the collection record (settings, seeds, package versions, model-file hashes) |
| `g1_locomotion/models/seed61_training/`, `seed62_training/` | G1 only: `run_config.json` (with the SHA-256 of the code and the data), `complete.json`, `train_log.jsonl`, `validation.json` of each run |
| `g1_locomotion/results/` | G1 only: the test output of both checkpoints (`sealed_test_seed61_seed62.json`, `.out`) |

The data of the first three cases come from [Project Chrono](https://projectchrono.org) simulations. The G1 data come
from MuJoCo 3.15.0 with the HOMIE walking policy of [OpenHomie](https://github.com/InternRobotics/OpenHomie) (commit
`33071f79`). OpenHomie has its own license; its files are not included. Each case README gives the
source campaign and its seed.

## Code

| File | Contents |
|---|---|
| `src/nedm/contact_nrd/model.py` | The three networks and the rollout |
| `src/nedm/contact_nrd/data.py` | Data loading, averaged speeds, the free-rollout metrics and the task metrics of an `evaluation` block (G1) |
| `src/nedm/contact_nrd/train.py` | Training (all stages, resume) |
| `src/nedm/contact_nrd/test.py` | Scores of one or more checkpoints on a test folder |
| `src/nedm/contact_nrd/download.py` | Download from Hugging Face and SHA-256 check |
| `configs/contact_nrd/evaluation/` | The G1 `evaluation` block and its README (every metric, term and scale) |
| `tests/contact_nrd/test_regression.py` | The first three cases train and test as with the release code `41be3ef` (needs a GPU and the released data; one logged diagnostic, `push_kept_fraction`, now counts the training pushes only) |
