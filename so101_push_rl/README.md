# SO101 push-T: one PPO policy for translation and rotation goals

A 5-joint SO101 arm pushes a T block (bar 100 × 25 mm, stem 25 × 75 mm) on a table to a goal pose (x, y, yaw). PPO
trains the policy inside the contact NRD, a learned simulator of the arm and the T
(see [`../contact_nrd/`](../contact_nrd/README.md)). Project Chrono then checks the policy. One policy reaches small
and large translation goals and small and large rotation goals.

## Results

Sealed test: 912 goals from start states that training and policy selection did not use. Chrono, deterministic policy.
A goal is reached when, at the end of the task time, the T is within 10 mm and 3° of the goal, the T is at rest (last
0.1 s) and no invalid event occurred (arm on the table, a link other than the finger on the T, T lifted or tilted,
T too close to the robot base).

| Goal type | Goal | Time | Earlier rotation policy (M6) | **C1f (selected)** | D2f (runner-up) |
|---|---|---|---|---|---|
| Small translation | 10–30 mm, yaw change ≤ 5° | 6 s | 0/192 (0 %) | **184/192 (96 %)** | 178/192 (93 %) |
| Large translation 1 | 40–60 mm, yaw change ≤ 5° | 6 s | 0/192 (0 %) | **165/192 (86 %)** | 168/192 (88 %) |
| Large translation 2 | 60–100 mm, yaw change ≤ 5° | 8 s | 0/128 (0 %) | **107/128 (84 %)** | 112/128 (88 %) |
| Small rotation | ±10–25°, ≤ 5 mm | 6 s | 1/128 (1 %) | **120/128 (94 %)** | 114/128 (89 %) |
| Large rotation | ±50° in place | 6 s | 76/144 (53 %) | **118/144 (82 %)** | 132/144 (92 %) |
| Combined (stretch goal, never trained) | 40–80 mm and ±20–45° | 8 s | 0/128 (0 %) | **46/128 (36 %)** | 39/128 (30 %) |

Validation (928 goals, fixed before training). "Trained starts" = the 8 start states that every curriculum stage
uses, with new goals. "New starts" = validation start states, which are not in the training set.

| Goal type | C1f, trained starts | C1f, new starts | D2f, trained starts | D2f, new starts |
|---|---|---|---|---|
| Small translation | 63/64 (98 %) | 121/128 (95 %) | 56/64 (88 %) | 119/128 (93 %) |
| Large translation 1 | 61/64 (95 %) | 122/128 (95 %) | 60/64 (94 %) | 119/128 (93 %) |
| Large translation 2 | 58/64 (91 %) | 56/64 (88 %) | 56/64 (88 %) | 51/64 (80 %) |
| Small rotation | 59/64 (92 %) | 52/64 (81 %) | 59/64 (92 %) | 59/64 (92 %) |
| Large rotation | 15/16 (94 %) | 123/144 (85 %) | 16/16 (100 %) | 133/144 (92 %) |
| Combined (stretch goal, never trained) | 28/64 (44 %) | 15/64 (23 %) | 27/64 (42 %) | 17/64 (27 %) |

- C1f was selected by a rule written before the test: the most successes on the trained-start validation rows of the
  five target goal types (C1f 256/272 = 94.1 %, D2f 247/272 = 90.8 %). The test was run once, after the selection.
- D2f is C1f plus 150 more updates with more rotation goals on every train start. It is better on +50° turns and
  worse on small translations.
- M6 is the best policy of the earlier rotation-only study (not released here).
- Median final error of C1f on the test: 3.0–5.4 mm and 0.6–1.2° on the five target goal types.

## How the policy was trained

| Step | Start | Curriculum | Updates | Output |
|---|---|---|---|---|
| B1 | new policy | `configs/curriculum_fresh.json`: 13 stages; small translation, small rotation and ±30° turns from the first stage (F1a), ±50° turns and large translation 1 from the 4th (F2a), large translation 2 from the 10th (F4a); the position tolerance goes from 25 mm to 10 mm | 400 | `policy/b1f_fresh_stage12/` |
| C1 | B1 (network weights only) | `configs/curriculum_cont.json`: the last B1 stage for 150 more updates, then one goal each of small translation, large translation 1 and 2, small rotation and a ±30–70° turn on all 2,048 train starts | 250 | `policy/c1f_selected/` = C1f |

- No stage of B1, C1 or D2 trains the combined goal type. Its rows in the result tables test generalisation only.
- PPO from rsl_rl 2.2.4 with 16,384 environments in the NRD, 48 decisions of 0.1 s per update, seed 1. The settings
  are in `configs/env.json` and in `reproduce.sh`.
- The policy gives a planar finger displacement and a gripper yaw change every 0.1 s. A decoder turns this into joint
  commands every 20 ms (inverse kinematics).
- The NRD is `rot_sc8_s61`. It has the contact NRD design (core, collision and contact Transformer networks, 2.4 M
  parameters). It is the last model of a chain: a model trained from the start on scripted Chrono pushes and turns,
  then 7 fine-tune rounds of 20,000 updates. Before each round, Chrono rollouts of earlier PPO policies were added to
  the data (10 checkpoints of 9 PPO runs in total). The data of the last round is released (`nrd_data/train/`). The
  fine-tune trainer is not in this repository. For each round, `provenance.nrd.fine_tune_chain` of the Hub copy of
  `release_manifest.json` gives the training config, the data and checkpoint SHA-256 and the episode counts.

## Reproduce

```bash
pip install rsl-rl-lib==2.2.4 tensorboard huggingface_hub   # next to a torch build for your GPU
export NRD_PYTHON=/path/to/python-with-pychrono               # pychrono 10.0.0

so101_push_rl/reproduce.sh smoke              # setup check: 2 PPO updates per run, 4 Chrono tasks
so101_push_rl/reproduce.sh eval               # released C1f on the 912 test goals in Chrono
so101_push_rl/reproduce.sh eval --split val --policy c1f_selected,d2f_runner_up
so101_push_rl/reproduce.sh full               # download, B1, C1, test eval of the new C1 and the released C1f
```

`reproduce.sh --help` lists the options (`--steps`, `--out`, `--data`, `--device`, `--workers`, `--rows`,
`--no-train-chrono`).
The header of the script names the NRD and every file the runs use. The code is in `scripts/so101_push_rl/` (trainer,
Chrono eval, score), `src/nedm/so101_push_rl/` (PPO environment in the NRD, Chrono controller) and
`src/nedm/so101_push/` (Chrono scene and step server); the NRD loader is `src/nedm/contact_nrd/model.py`.

| Step | Time | Machine |
|---|---|---|
| B1, 400 updates | 2 h 32 min | one AMD MI350X |
| C1, 250 updates | 1 h 34 min | one AMD MI350X |
| Chrono eval, 912 goals × 3 policies | 14 min | 120 CPU workers |
| B1 / C1 on an RTX 5090 | not measured | |

- Requirements: Python 3.10 or newer; `torch` (tested 2.10.0+rocm7.1 on an AMD MI350X and 2.12.0+cu130 on an RTX
  5090); `rsl-rl-lib` 2.2.4; `tensorboard`; `huggingface_hub` for the download. For the Chrono check: `pychrono` 10.0.0
  (conda channel `projectchrono`, Bullet collision; local build `py312h98ab86c_1187`) in the Python that `NRD_PYTHON`
  names; it can be the same Python. The released evaluations ran on a Chrono 10 build on the AMD cluster. This study
  does not use the repository's `requirements.txt`.
- Each Chrono episode runs in its own server process (`python -m nedm.so101_push.chrono_server`). The script writes the
  file that the server sources before it starts (`SO101_CHRONO_ENV`). Set `SO101_CHRONO_ENV` to your own file if your
  Chrono needs more setup. If you run `scripts/so101_push_rl/train_so101_planar_ppo.py` or
  `scripts/so101_push_rl/eval_so101_planar_chrono.py` without the script, set `SO101_CHRONO_ENV` yourself (for
  example to the `chrono_env.sh` that the script wrote in its output folder): the default path in the code is not in
  this repository.
- As in the original runs, the trainer runs Chrono checks in the background every 50 updates and at each stage end
  (48 validation tasks). Their results do not go back into training. They need `NRD_PYTHON`, also for short runs;
  `--no-train-chrono` turns them off and needs no Chrono.
- Chrono results can differ between machines. Last-bit float64 differences in the command decoder grow through the
  contacts, so the outcome of a borderline goal can flip: in a 28-episode check on two machines, 1 borderline goal
  flipped and the final T poses differed by up to 0.7 mm and 1°. Compare success rates, not single goals. On one
  machine the Chrono check is bitwise repeatable. A new PPO run also does not give the same digits as the released runs
  (GPU type, library versions).

## Data

Hugging Face dataset [harryzhang1018/NeDM](https://huggingface.co/datasets/harryzhang1018/NeDM), folder
`so101_push_rl/`, pinned in [`release_manifest.json`](release_manifest.json) (revision, bytes and SHA-256 of each file).

| Path | Contents |
|---|---|
| `nrd/rot_sc8_s61/best.pt` | The NRD that every PPO run used (next to it: `run_config.json` and `contact_scale.json`, its training record) |
| `nrd_data/train/` | The NRD's training data, 144,112 train / 7,828 validation episodes (15.8 GB) |
| `rl_inputs/tasks/` | Task tables (start, goal, time, level) for train, validation and test; the eval rows (`eval_ids_val.json`, `eval_ids_test.json`) |
| `rl_inputs/starts/`, `rl_inputs/banks/` | Start states and the recorded Chrono episodes they come from (train, validation, test) |
| `rl_inputs/stats/stats.pt` | Observation scales |
| `rl_inputs/provenance/` | How the task tables, starts, banks and scales were made |
| `policy/c1f_selected/`, `d2f_runner_up/`, `b1f_fresh_stage12/` | `policy.pt` and the run files (`env_cfg.json`, `train_cfg.json`, `curriculum.json`, `manifest.json`, learning curve) |
| `results/` | The original Chrono evaluations and their scores |

The data come from [Project Chrono](https://projectchrono.org) simulations.

The SO-101 robot model files in `assets/so101/` describe the open-source SO-101 arm (TheRobotStudio SO-ARM100 project).
They are a compact copy of a digital-twin description that was converted from the `SO-ARM101-USD.usd` model of the
[isaac-sim/Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop) repository.
