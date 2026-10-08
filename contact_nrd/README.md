# Contact NRD: bouncing ball, pool and SO101 push-T

A neural reduced dynamics (NRD) model for systems whose contacts start and stop. The three study cases use the same
network design, the same training and the same config file. Only the data changes.

| Study case | System | Folder |
|---|---|---|
| Bouncing ball | One ball bounces on a floor and hits a wall | [`bouncing_ball/`](bouncing_ball/README.md) |
| Pool | Ball A hits ball B; both balls hit the cushions | [`pool/`](pool/README.md) |
| SO101 push-T | A 5-joint SO101 arm pushes a T-shaped block on a table | [`so101_push_t/`](so101_push_t/README.md) |

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
  kind, the fixed partner (floor, wall, cushion) and the role in the pair. The arm's joint command is one more token.
  There are no geometry features.
- **State.** The state holds the pose and the speeds of each moving body. The networks predict only speed changes.
  Linear speeds, joint speeds and the T's angular speed are averages over the last 20 ms step. The ball spin is the
  recorded value, because a ball state has no orientation.
- **Step.** speeds' = speeds + core change + sum of the contact changes of the pairs that the collision network
  switches on. The pose comes from integration, pose' = pose + 20 ms × speed', and goes into all three networks.
  An averaged speed still holds the effect of a contact in the step before. For this reason the switch covers that
  step too.

**Training** (`configs/contact_nrd/train.json`, one file for all three cases):

1. Core: a linear fit on free-motion steps gives the start point.
2. Collision network: 30,000 updates alone, then frozen.
3. Core and contact network: 120,000 updates together, with three losses: the one-step core error, the one-step
   contact error and the rollout error over up to 1 s. Training adds noise to the history. After the first quarter
   of the updates, half of the contact examples use 1 to 3 of the model's own steps as history.
4. Selection: every 10,000 updates, free rollouts on up to 1,500 validation episodes (all 900 for the ball) give a
   score. The best checkpoint is kept.

## Results

Released checkpoints on the test episodes. The test episodes come from a separate Chrono collection with a new seed.
Training and checkpoint selection did not use them; earlier design comparisons did. Each rollout starts from the
episode's first recorded state and runs open loop (the arm gets its recorded commands). Median / p95:

| Study case | Error | Seed 61 | Seed 62 |
|---|---|---|---|
| Bouncing ball (1,800 shots) | Ball position at 1.7 s | 0.90 / 2.09 mm | 0.90 / 2.09 mm |
| | Ball path, RMS up to 1.7 s | 0.60 / 1.08 mm | 0.61 / 1.10 mm |
| Pool (4,800 shots) | Ball B position at 2 s | 0.69 / 2.41 mm | 0.64 / 2.52 mm |
| SO101 push-T (1,000 episodes) | T position at 4 s | 0.43 / 2.70 mm | 0.37 / 3.07 mm |
| | T angle at 4 s | 0.23 / 1.61° | 0.20 / 1.61° |

## Reproduce

Requirements: `torch` and `numpy`, plus `huggingface_hub` for the download. `requirements.txt` installs all three. No
simulator.

```bash
# 1. Download the case (train, test and the two released checkpoints) -> artifacts/contact_nrd/pool/
PYTHONPATH=src python -m nedm.contact_nrd.download --case pool
# 2. Train one seed on one GPU
PYTHONPATH=src python -m nedm.contact_nrd.train --data artifacts/contact_nrd/pool/train --seed 61 \
    --output artifacts/contact_nrd/runs/pool_s61
# 3. Score the retrained model and a released checkpoint on the test episodes
PYTHONPATH=src python -m nedm.contact_nrd.test --data artifacts/contact_nrd/pool/test \
    --model retrained=artifacts/contact_nrd/runs/pool_s61 released=artifacts/contact_nrd/pool/models/seed61.pt
```

Use `--case bouncing_ball` or `--case so101_push_t` (and the same folder names) for the other cases.

| Study case | Download | Training time, one MI350X | GPU memory |
|---|---|---|---|
| Bouncing ball | 0.77 GB | about 4.5 h | about 12 GB |
| Pool | 5.9 GB | about 5.0 h | about 32 GB |
| SO101 push-T | 3.2 GB | about 5.9 h | about 24 GB |

- The times come from 8 runs that shared one 8-GPU node. They do not include the work that a resume repeats.
- A run is longer than a 4 h job limit. If a job time limit stops a run, run the same command again. Training
  continues from the last check (every 10,000 updates).
- A new run with the same seed does not give the same digits. With `--deterministic`, two runs on the same machine
  do.

## Data

Hugging Face dataset [harryzhang1018/NeDM](https://huggingface.co/datasets/harryzhang1018/NeDM), folder `contact_nrd/`,
pinned in [`release_manifest.json`](release_manifest.json) (revision and SHA-256 of each file). Each case has:

| Path | Contents |
|---|---|
| `<case>/train/system.json` | Bodies (kind, size, fixed planes), contact pairs, record step, target body and time, Chrono settings. The SO101 file also gives the channel names and types |
| `<case>/train/unified_data.npz` | The states (ball and pool: one `states` array for all balls; SO101: one array for each moving body), the arm commands, the contact labels of each pair, episode lengths, splits (0 train, 1 validation, 2 the source campaign's own test episodes, not used) |
| `<case>/test/` | The same two files for the test episodes (split 2) |
| `<case>/models/seed61.pt`, `seed62.pt` | The released checkpoints |

The data come from [Project Chrono](https://projectchrono.org) simulations. Each case README gives the source
campaign and its seed.

## Code

| File | Contents |
|---|---|
| `src/nedm/contact_nrd/model.py` | The three networks and the rollout |
| `src/nedm/contact_nrd/data.py` | Data loading, averaged speeds and the free-rollout metrics |
| `src/nedm/contact_nrd/train.py` | Training (all stages, resume) |
| `src/nedm/contact_nrd/test.py` | Scores of one or more checkpoints on a test folder |
| `src/nedm/contact_nrd/download.py` | Download from Hugging Face and SHA-256 check |
