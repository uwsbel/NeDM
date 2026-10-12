# SO101 push-T: one PPO policy for translation and rotation goals

A 5-joint SO101 arm pushes a T block (bar 100 × 25 mm, stem 25 × 75 mm) on a table to a goal pose (x, y, yaw). PPO
trains the policy inside a contact NRD, a learned simulator of the arm and the T (design:
[`../contact_nrd/`](../contact_nrd/README.md)). Project Chrono then checks the policy. One policy reaches small and
large translation goals and small and large rotation goals.

As for `contact_nrd/`, this repository holds only the NRD side: the PPO environment in the NRD, the trainer and its
configs. The Chrono evaluations are released as data (`results/` on the Hub). The Chrono scene and the evaluation code
are not in this repository.

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

- The experiment's Chrono evaluation measured both tables on the AMD cluster, one run each (seed 0, the decoder
  settings of training). Every episode and score is in `results/` on the Hub: test = `results/test_M6_C1f_D2f/`,
  validation = `results/val_C1f_D1_D2/` (`score.md`; one row per goal and policy in `score.json` and
  `episodes.jsonl`).
- C1f was selected by a rule written before the test: the most successes on the trained-start validation rows of the
  five target goal types (C1f 256/272 = 94.1 %, D2f 247/272 = 90.8 %). The test was run once, after the selection.
- D2f is C1f plus 150 more updates with more rotation goals on every train start. It is better on +50° turns and
  worse on small translations.
- M6 is the best policy of the earlier rotation-only study (not released here).
- Median final error of C1f on the test: 3.0–5.4 mm and 0.6–1.2° on the five target goal types.
- Chrono results can differ between machines. Last-bit float64 differences in the command decoder grow through the
  contacts, so a borderline goal can flip: in a 28-episode check on two machines, 1 goal flipped and the final T
  poses differed by up to 0.7 mm and 1°. Compare success rates, not single goals.

## How the policy was trained

| Step | Start | Curriculum | Updates | Output |
|---|---|---|---|---|
| B1 | new policy | `configs/so101_push_rl/curriculum_fresh.json`: 13 stages; small translation, small rotation and ±30° turns from the first stage, ±50° turns and large translation 1 from the 4th, large translation 2 from the 10th; the position tolerance goes from 25 mm to 10 mm | 400 | `policy/b1f_fresh_stage12/` |
| C1 | B1 (network weights only) | `configs/so101_push_rl/curriculum_cont.json`: the last B1 stage for 150 more updates, then one goal each of small translation, large translation 1 and 2, small rotation and a ±30–70° turn on all 2,048 train starts | 250 | `policy/c1f_selected/` = C1f |

- No stage of B1, C1 or D2 trains the combined goal type. Its rows in the result tables test generalisation only.
- PPO from rsl_rl 2.2.4 with 16,384 environments in the NRD, 48 decisions of 0.1 s per update, seed 1. The settings
  are in `configs/so101_push_rl/env.json` and in `scripts/so101_push_rl/reproduce.sh`.
- The policy gives a planar finger displacement and a gripper yaw change every 0.1 s. A decoder turns this into joint
  commands every 20 ms (inverse kinematics).
- Every 25 updates the trainer scores the policy in the NRD on goals of the validation task table. These scores
  decide when a curriculum stage ends. The original runs also started Chrono checks in the background. Their results
  never went back into training. The trainer here has no such check; on one machine, PPO with and without the checks
  gives bitwise the same weights.

## The NRD

PPO trains inside one NRD checkpoint, `nrd/rot_sc8_s61/best.pt`. It has the design of
[`../contact_nrd/`](../contact_nrd/README.md), case SO101 push-T: the same three networks, state, channel masks, step
rule and file format. `nedm.contact_nrd.model.load_model` loads it. It is not the released
`contact_nrd/so101_push_t/models/seed61.pt`: it was trained on more data, which is released in `nrd_data/train/`.

| Data (`nrd_data/train/`) | Train / validation episodes |
|---|---|
| Scripted pushes, the same as `contact_nrd/so101_push_t/train/` | 51,200 / 1,000 |
| Scripted walks and turns of the T (4 s) | 39,320 / 2,500 |
| Chrono episodes of PPO policies (up to 8.6 s) | 53,592 / 4,328 |
| Total | 144,112 / 7,828 |

| Test, T at 4 s (median / p95) | `seed61.pt` | `rot_sc8_s61` |
|---|---|---|
| 1,000 scripted pushes (`contact_nrd/so101_push_t/test/`), position / angle | 0.43 / 2.70 mm, 0.23 / 1.61° | 0.44 / 2.35 mm, 0.25 / 1.45° |
| 2,048 new walks and turns of the T (not released), position / angle | 14.3 / 431 mm, 5.2 / 60° | 1.25 / 18.3 mm, 0.70 / 9.4° |

- The policy turns the T by up to ±50°. `seed61.pt` never saw walks and turns of the T and is 6 to 24 times less
  accurate on them, so this study trained its NRD on the larger data set.
- Both test sets were used to monitor the NRD while it was developed. They are not sealed tests.
- Training code and recipe: `python -m nedm.contact_nrd.train` with `configs/contact_nrd/train.json`
  ([`../contact_nrd/README.md`](../contact_nrd/README.md)). A model retrained this way from scratch on
  `nrd_data/train/` has not been compared with the released checkpoint.

## Reproduce

```bash
pip install rsl-rl-lib==2.2.4 tensorboard huggingface_hub   # next to a torch build for your GPU

scripts/so101_push_rl/reproduce.sh smoke     # setup check: 2 PPO updates per run, NRD validation summary
scripts/so101_push_rl/reproduce.sh full      # download, B1, C1, NRD validation at each stage end vs the released runs
scripts/so101_push_rl/reproduce.sh full --steps download --with-results   # also download the Chrono evaluations
```

`reproduce.sh --help` lists the options. The header of the script names the NRD and every file the runs use.

| Step | Time | Machine |
|---|---|---|
| B1, 400 updates | 2 h 32 min | one AMD MI350X |
| C1, 250 updates | 1 h 34 min | one AMD MI350X |
| Smoke (files already downloaded) | 1.5–3 min | one RTX 5090, shared with other jobs |

- Requirements: Python 3.10 or newer; `torch` (tested 2.10.0+rocm7.1 on an AMD MI350X and 2.12.0+cu130 on an RTX
  5090); `rsl-rl-lib` 2.2.4; `tensorboard`; `huggingface_hub` for the download. No Chrono. This study does not use
  the repository's `requirements.txt`.
- A new PPO run does not give the same digits as the released runs (GPU type, library versions). Compare the NRD
  validation that the summary step prints with the released runs.

| Code | Contents |
|---|---|
| `scripts/so101_push_rl/train_so101_planar_ppo.py` | PPO trainer: curriculum, NRD validation, checkpoints, resume |
| `scripts/so101_push_rl/reproduce.sh` | Download, B1, C1 and the summary |
| `src/nedm/so101_push_rl/` | The PPO environment in the NRD: tasks, action decoder, reward, validity rules, arm and T model |
| `configs/so101_push_rl/` | Environment, the two curricula, arm and T settings (`collector_v2.json`), fingertip surface points |
| `assets/so101/` | SO101 links, joints and collision shapes |
| `src/nedm/contact_nrd/model.py` | The NRD and its loader (from `contact_nrd/`) |

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
| `results/` | The Chrono evaluations behind the result tables: one folder per evaluation (`test_M6_C1f_D2f`, `val_C1f_D1_D2`, `val_M6_B1_B2`) with `episodes.jsonl`, `eval_plan.json`, `score.json`, `score.md` and `summary.json`; `TEST_DECLARATION.md` (the test rule, fixed before the test run) |

The Hub copy of `release_manifest.json` also holds the provenance. It was written for an earlier layout of this
release:

- It names `scripts/so101_push_rl/eval_so101_planar_chrono.py` and `scripts/so101_push_rl/score_eval.py`, the Chrono
  evaluation and its scorer. They are not in this repository.
- It names `configs/so101_push/collector_v2.json` and `configs/so101_push/collision_points_dense_2mm.npz`. They are now
  `configs/so101_push_rl/collector_v2.json` and `assets/so101/collision_points_dense_2mm.npz`, with the same SHA-256 (`884d3836…`, `47475303…`).
- The recorded command of each policy has `--chrono-every`, `--chrono-n` and `--chrono-workers`. The trainer here does
  not have these options.

The data come from [Project Chrono](https://projectchrono.org) simulations.

The SO-101 robot model files in `assets/so101/` describe the open-source SO-101 arm (TheRobotStudio SO-ARM100 project).
They are a compact copy of a digital-twin description that was converted from the `SO-ARM101-USD.usd` model of the
[isaac-sim/Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop) repository.
