# G1 footsteps: a PPO policy that puts the feet of a walking humanoid on given footprints

A frozen walking policy (HOMIE) drives the 27-joint Unitree G1 humanoid. The footstep policy sets HOMIE's velocity
command (forward speed, sideways speed, turn rate) every 20 ms, so that the feet land on a given row of footprints, on
a flat floor and on planks with gaps. PPO trains the policy only inside the G1 contact NRD, a learned simulator
([`../contact_nrd/g1_locomotion/`](../contact_nrd/g1_locomotion/README.md)). CPU MuJoCo then checks it on new courses.

Unlike `so101_push_rl/`, this study also holds the simulator side of its evaluation: the MuJoCo scene, the episode
runner and the evaluation script. With them, the MuJoCo tables can be run again. The data collector of the NRD is not
in this repository.

## Results

Development courses: 64 new courses per cell (course seeds 1000–1063), CPU MuJoCo, deterministic policy. Each episode
starts with the robot standing (1 s, HOMIE with a zero command); then the controller has 12 s. Two policies are
released, s1 and s2 (PPO seeds 1 and 2). The baseline H-FS is HOMIE with a hand-written controller that turns the next
footprints into a velocity command; its 4 parameters were tuned on other courses (36 settings, `results/hfs_tuning/`).

| Cell | Course (straight unless stated; half of the courses end with a stop, the others walk through the end) |
|---|---|
| F1 | Flat floor, 12–20 footprints; the step length (0.05–0.30 m) changes every 2–5 steps |
| F2 | Flat floor, 16–24 footprints, step length 0–0.34 m; half of the courses curve, 30 % change lanes |
| MP | 8 planks, 2 m wide, with gaps of 3–13 cm; 1–3 footprints on each plank; not a training level |
| MN | The same with planks 0.5 m wide; not a training level |
| ES | 4–7 planks, 0.5 m wide; every course ends with a stop, the toes 2–4 cm before the edge of the last platform |

Each value: s1 / s2 / H-FS. p: s1 / s2 against H-FS.

| Cell | Completion % | Clean completion % | Footprints within 5 cm % | Error p50–p95 (cm) | Falls | McNemar p (completion) |
|---|---|---|---|---|---|---|
| F1 | 96.9 / 98.4 / 28.1 | 96.9 / 98.4 / 28.1 | 100.0 / 99.9 / 42.3 | 1.3–2.9 / 1.3–3.0 / 5.7–11.5 | 0 / 0 / 0 | 1.1e-13 / 5.7e-14 |
| F2 | 92.2 / 98.4 / 15.6 | 92.2 / 98.4 / 15.6 | 98.4 / 98.9 / 37.1 | 1.7–3.9 / 1.7–3.7 / 6.2–13.4 | 0 / 0 / 0 | 3.6e-15 / 2.2e-16 |
| MP | 98.4 / 98.4 / 0.0 | 98.4 / 85.9 / 0.0 | 100.0 / 99.3 / 28.7 | 1.8–3.5 / 2.2–4.0 / 6.6–11.4 | 0 / 0 / 31 | 2.2e-19 / 2.2e-19 |
| MN | 100.0 / 96.9 / 0.0 | 98.4 / 89.1 / 0.0 | 99.8 / 99.1 / 26.3 | 1.8–3.4 / 2.2–4.0 / 6.7–13.2 | 0 / 0 / 31 | 1.1e-19 / 4.3e-19 |
| ES | 100.0 / 100.0 / 0.0 | 100.0 / 95.3 / 0.0 | 99.8 / 98.6 / 25.0 | 1.9–3.5 / 2.2–4.3 / 7.1–12.8 | 0 / 0 / 33 | 1.1e-19 / 1.1e-19 |

- **Completion:** the robot steps on every footprint in order with the correct foot, none more than 15 cm off, and on
  a stop course it then stands still for 0.5 s. A course fails at a step with the wrong foot, a step more than 15 cm
  off, more than 3 corrective steps, 1.5 s without a new footprint, a stop that does not settle in 2.5 s, a foot that
  moves more than 5 cm after the stop, or a fall.
- **Clean completion:** completion with no touchdown over a gap and no contact step with a sole point off the planks.
  On a flat floor it equals completion.
- **Footprints within 5 cm, error:** distance from the foot centre at touchdown to its footprint, over the footprints
  that the robot reached. **Falls:** a body part other than a foot touches the ground or a plank, a foot touches the
  pit floor, or the pelvis drops below 0.35 m or tilts more than 45°. **p:** two-sided exact McNemar test, paired on
  the same 64 courses.
- H-FS completes no course that a policy fails. On a flat floor it fails mostly by a step with the wrong foot (F1 33,
  F2 36 of 64); on the planks about half of its courses end in a fall (31–33 of 64). With "every footprint within 8 cm" added to
  completion, H-FS succeeds on 0 of 64 courses in every cell and the policies on 58–64 (p ≤ 6.9e-18).
- The policies do not fall. The pelvis stays within 7 cm of the course line (p95 of the largest distance); the largest
  heading error is 8–12° on a flat floor and 13–16° on the planks (p95).
- s1 is the policy of the videos and of the smoke run: its mean clean completion over the five cells is 97.2 %, that of
  s2 93.4 %.
- These development courses also guided choices during the study (the curriculum, the policy for the videos), so the
  numbers can be optimistic. The study fixed its final-test courses before training (the same five cells, other
  seeds). They stay sealed: this release does not run them, and no number here comes from them.

### NRD against MuJoCo

The evaluation also runs every episode inside the NRD, from the same state after the 1 s stand, once with each NRD
checkpoint (member 0 = `seed61.pt`, member 1 = `seed62.pt`). Completion: MuJoCo / NRD member 0 / NRD member 1.
Agreement: the share of courses where the NRD gives the same outcome as MuJoCo (member 0 / member 1).

| Cell | s1: completion % | s1: agreement % | s2: completion % | s2: agreement % |
|---|---|---|---|---|
| F1 | 96.9 / 100.0 / 100.0 | 96.9 / 96.9 | 98.4 / 100.0 / 100.0 | 98.4 / 98.4 |
| F2 | 92.2 / 96.9 / 100.0 | 92.2 / 92.2 | 98.4 / 98.4 / 98.4 | 96.9 / 96.9 |
| MP | 98.4 / 70.3 / 53.1 | 71.9 / 51.6 | 98.4 / 68.8 / 40.6 | 67.2 / 42.2 |
| MN | 100.0 / 81.2 / 62.5 | 81.2 / 62.5 | 96.9 / 65.6 / 48.4 | 65.6 / 51.6 |
| ES | 100.0 / 56.2 / 62.5 | 56.2 / 62.5 | 100.0 / 56.2 / 31.2 | 56.2 / 31.2 |
| All 320 | 97.5 / 80.9 / 75.6 | 79.7 / 73.1 | 98.4 / 77.8 / 63.8 | 76.9 / 64.1 |

- On a flat floor the NRD agrees with MuJoCo on 92–98 % of the courses. On the planks the NRD is pessimistic: almost
  all of its extra failures come from the support rule (a foot judged off a plank) on courses that MuJoCo completes.
- The footprint error is close in the two worlds: p95 3.5–3.9 cm in MuJoCo and 3.2–3.5 cm in the NRD (all cells).
- Over the 10 (policy, cell) pairs, the rank correlation (Spearman) of NRD and MuJoCo completion is −0.42 (bootstrap
  95 % interval −0.82 to 0.21). Over all 1,280 course pairs of both members, the agreement is 73.4 % (phi −0.01)
  (`results/dev/scatter_dev/scatter.json`). The policy learned in the NRD works in MuJoCo, but the NRD's own scores
  do not rank these cells as MuJoCo does. Use MuJoCo to score plank courses.

## How the policy was trained

| Stage | Training courses | Rules | Promotion (NRD validation) | Ended at update, s1 / s2 |
|---|---|---|---|---|
| S0 | F0 | Lenient scoring (of the scoring failures, only a stall counts), 8 cm tolerance | F0: completion ≥ 0.8, within 8 cm ≥ 0.8 | 50 / 50 |
| S1 | F0 30 %, F1 70 % | Lenient, 5 cm | F1: completion ≥ 0.7, within 5 cm ≥ 0.8 | 150 / 100 |
| S2 | F1 30 %, F2 70 % | Strict scoring (the failures above) | F2: completion ≥ 0.7, within 5 cm ≥ 0.8 | 225 / 200 |
| S3 | F2 30 %, P1 70 % | Planks: a foot with 2 of its 4 sole points off a plank fails | P1: completion ≥ 0.6 | 275 / 250 |
| S4 | F2 20 %, P1 30 %, P2 50 % | 1 sole point off a plank fails; 2 cm edge margin | P2: clean completion ≥ 0.6 | 450 / 525 |
| S5 | F2 30 %, P2 20 %, P3 35 %, ES 15 % | As S4 | F2: completion ≥ 0.75, within 5 cm ≥ 0.85; clean completion P2 ≥ 0.75, P3 ≥ 0.6; ES: completion ≥ 0.5 | 2,375 (rules met) / 2,525 (update limit; P3 0.43) |

- F0: flat floor, straight, 12–16 footprints, no stop. P1–P3: straight plank courses with 6–9 planks; the gaps grow
  from 3–7 cm (P1) to 5–13 cm (P3), and P3 also has planks 0.6 m and 0.4 m wide. Curriculum:
  `configs/g1_footstep_rl/footsteps_curriculum.json`.
- Every 25 updates the trainer runs the deterministic policy in the NRD on 128 validation courses of each level of the
  stage (64 with each member). A stage ends when all its rules hold (after a minimum number of updates) or at its
  update limit (S5: 600 to 2,000 updates). Training never uses MuJoCo.
- PPO from rsl_rl 2.2.4: 16,384 NRD environments, 32 steps of 20 ms per update, 5 epochs of 8 mini-batches, adaptive
  learning rate from 3e-4; actor and critic are MLPs 512-256-128 (ELU). An episode lasts up to 600 steps (12 s) from
  a standing state of the start bank, in one NRD member drawn for the episode. Settings and reward terms:
  `train_cfg.json` and `env_cfg.json` next to each policy.
- Besides the scoring rules, an episode in the NRD ends when a foot in contact has a sole point outside every plank
  (planks shrunk by the 2 cm edge margin, in two states in a row), the pelvis drops below 0.55 m or tilts more than
  30°, the legs come closer than 4 mm (a swept-volume check: the NRD does not model self-contact), or a joint leaves
  the range of the NRD training data by more than 0.05 rad.

| Run | Updates | Environment steps | Time | Machine |
|---|---|---|---|---|
| s1 (`fs_cmd_s1`) | 2,375 | 1.25 billion | 4.2 h (3 h 38 min, then a resume of 33 min) | AMD MI300X (two jobs) |
| s2 (`fs_cmd_s2`) | 2,525 | 1.32 billion | 3.1 h | AMD MI350X |

## Policy inputs and outputs

163 inputs → policy → 3 commands → frozen HOMIE → 12 leg joint targets (the 15 upper-body joints stay at the default
pose) → one 20 ms step of the NRD (training) or of MuJoCo (evaluation).

| Group | Values | Content |
|---|---|---|
| Body | 99 | Gravity direction and angular velocity in the pelvis frame (6); joint angles from the default pose (27); joint rates (27); HOMIE's 12 leg outputs for the last command (a preview that does not change HOMIE's memory); the previous action after tanh (15; the last 12 are 0 for this policy); the two foot contact switches (2); foot positions from the pelvis (6); pelvis velocity (3) and height (1) |
| Course line | 2 | Sideways offset and heading error of the pelvis against the line of the current footprint |
| Next 4 footprints | 4 × 14 | Position from the pelvis and from the foot that must land there (4); step length and width (2); yaw (1); left or right foot (1); free plank length in front and behind and free width to the left and right (4); stop flag (1); valid flag (1) |
| Status | 6 | Time since the last touchdown of each foot (2); footprints left (1); along and across error of the last placement (2); start window (1) |
| Outputs | 3 | Forward speed −0.5 to 1.2 m/s, sideways speed −0.5 to 0.5 m/s, turn rate −0.8 to 0.8 rad/s; each = low + (high − low) (tanh(a) + 1) / 2. Height command fixed at 0.74 m |

- Positions and velocities are in the pelvis heading frame. The joint rates are 20 ms averages, as in the NRD state, in
  both worlds. The critic sees the same 163 values.
- HOMIE runs as a torch copy of its `deploy.onnx` with its 6-frame history. H-FS drives the original ONNX model
  through onnxruntime with instantaneous joint rates (HOMIE's own input).

## The NRD and HOMIE

- PPO steps the two released G1 checkpoints `contact_nrd/g1_locomotion/models/seed61.pt` and `seed62.pt`, unchanged
  (`nedm.contact_nrd.model.load_model`). No footstep episode went back into NRD training.
- The NRD knows only a flat floor. On plank courses the planks exist only in the rules and in the observation. In
  MuJoCo they are boxes with their tops at z = 0, over a 0.6 m deep pit.
- HOMIE ([OpenHomie](https://github.com/InternRobotics/OpenHomie), commit `33071f79`) is not in this repository and
  not on the Hub. `scripts/g1_footstep_rl/fetch_homie.py` fetches it from that commit, checks 71 files by size and
  SHA-256 (`deploy.onnx`, the robot model `g1.xml` and its 49 meshes, licenses and code;
  `configs/g1_footstep_rl/homie_sources.json`) and writes `artifacts/g1_footstep_rl/homie/homie_weights.npz`
  (SHA-256 `9a4ade34…`, the value that both runs record). OpenHomie has its own license.

## Reproduce

```bash
pip install rsl-rl-lib==2.2.4 tensorboard huggingface_hub mujoco==3.15.0 onnx onnxruntime==1.30.0  # next to torch

scripts/g1_footstep_rl/reproduce.sh smoke   # setup check: 2 PPO updates per seed, MuJoCo + NRD on 2 courses per cell
scripts/g1_footstep_rl/reproduce.sh full    # download, HOMIE, both PPO runs, MuJoCo development evaluation, summary
scripts/g1_footstep_rl/reproduce.sh full --steps download,homie,eval,summary   # the released policies only, no PPO
```

`reproduce.sh --help` lists the options. The header of the script names the NRD and every file that the runs read.
All data go to `artifacts/` of the checkout. The NRD-against-MuJoCo correlation over cells (`scatter`) is not a step;
run it on two policy evaluation folders (repository root, `PYTHONPATH=src`):

```bash
python scripts/g1_footstep_rl/eval_footsteps_mujoco.py scatter <FS_cmd_s1 folder> <FS_cmd_s2 folder> --out <folder>
```

`videos/videos.json` on the Hub gives the command of the videos; `footstep_baselines.py tune` repeats the H-FS
parameter search.

| Step | Time | Machine |
|---|---|---|
| PPO s1 / s2 | 4.2 h / 3.1 h | AMD MI300X / MI350X; 5.5 GB GPU memory with 16,384 environments (RTX 5090) |
| `full --steps eval,summary`: H-FS (34 s) and both policies with the NRD part (about 4 min each) | 8.3 min | 8 CPU workers, AMD Ryzen 7 5800X |
| Smoke (files already downloaded) / H-FS parameter search | 1.5 min / 2.4 min | RTX 5090 and 8 CPU workers |
| Course banks / start bank (4.8 GB host memory) | 26 s / 24 s | 8 CPU workers / 8 threads |

- Requirements: Python 3.10 or newer; `torch` (tested 2.10.0+rocm7.1 on AMD MI300X and MI350X, 2.12.0+cu130 on an
  RTX 5090); `rsl-rl-lib` 2.2.4; `mujoco` 3.15.0; `onnx`; `onnxruntime` 1.30.0; numpy 2.4 (another numpy can write
  `homie_weights.npz` as other bytes; `fetch_homie.py` then stops); `huggingface_hub`; git for the HOMIE fetch. Videos
  also need `matplotlib`, `imageio`, ffmpeg and `MUJOCO_GL=egl`.
- On one machine this code gives bitwise the same PPO rollouts and updates as the experiment code (also after a
  resume) and the same MuJoCo episodes for both policies and H-FS. The bank builders give the released arrays bitwise.
- On other hardware, a PPO run does not give the same digits: compare the NRD validation at each stage end (the summary
  step prints it next to the released `done.json`). MuJoCo episodes repeat bitwise only with the CPU type and package
  versions of `results/dev/environment.json` (AMD Ryzen 7 5800X, numpy 2.4.6, torch 2.12.0, onnxruntime 1.30.0):
  elsewhere compare rates. The NRD part of the evaluation also needs the same number of torch threads
  (`--nrd-threads`, default 4, as in the released runs; `reproduce.sh` keeps the default).

| Code | Contents |
|---|---|
| `scripts/g1_footstep_rl/train_footsteps_ppo.py` | PPO trainer: curriculum, NRD validation, checkpoints, resume |
| `scripts/g1_footstep_rl/eval_footsteps_mujoco.py` | MuJoCo evaluation (`eval`, `nrd`, `report`, `video`, `scatter`, `parity`) |
| `scripts/g1_footstep_rl/footstep_baselines.py` | H-FS parameter search |
| `scripts/g1_footstep_rl/build_course_bank.py`, `build_start_bank.py` | The course banks and the start bank |
| `scripts/g1_footstep_rl/fetch_homie.py` | HOMIE fetch and checks |
| `scripts/g1_footstep_rl/reproduce.sh` | Download, HOMIE, PPO, MuJoCo evaluation, summary |
| `src/nedm/g1_footstep_rl/` | Courses and scorer, the PPO environment in the NRD, HOMIE copy, leg kinematics, MuJoCo scene and runner, H-FS (module table in `__init__.py`) |
| `configs/g1_footstep_rl/` | Environment config, curriculum, MuJoCo scene, HOMIE sources |
| `src/nedm/contact_nrd/model.py`, `data.py` | The NRD and its loader; 20 ms averaged rates (from `contact_nrd/`) |

## Data

Hugging Face dataset [harryzhang1018/NeDM](https://huggingface.co/datasets/harryzhang1018/NeDM), folder
`g1_footstep_rl/` (975 MB), pinned in [`release_manifest.json`](release_manifest.json) (revision, bytes and SHA-256 of
each file). The NRD checkpoints are in `contact_nrd/g1_locomotion/models/` (`../contact_nrd/release_manifest.json`).

| Path | Contents |
|---|---|
| `policy/fs_cmd_s1/`, `policy/fs_cmd_s2/` | `policy.pt` (the checkpoint at the end of S5) and the run files: `env_cfg.json`, `train_cfg.json`, `curriculum.json`, `manifest.json` (SHA-256 of every run input), `done.json` (stage ends, validation history), `val_history.json`, `learning_curve.csv` |
| `rl_inputs/start_bank/` | `bank_v1.npz`: 200,000 start states from the training split of the NRD data (140,000 nominal, 60,000 difficult), each with the NRD's 4-state history and HOMIE's 6-frame history; the runs use its nominal standing states. `bank_v1.json` (summary), `envelope_v1.json` (joint ranges of the training data) |
| `rl_inputs/courses/` | `course_bank_v4_train.npz` (200,000 training courses, 25,000 per level), `course_bank_v4_val.npz` (1,280 validation courses, 128 per level), `bank_report.json` |
| `results/dev/` | One folder per evaluation (`FS_cmd_s1_dev`, `FS_cmd_s2_dev`, `HFS_dev`): `rows.json` (one row per course), `steps.json` (one row per footprint), `summary.json`, `SUMMARY.md`, `meta.json`; the policy folders also hold `nrd_rows.json`, `nrd_steps.json` and `snapshots.npz` (the start states of the NRD part). `scatter_dev/scatter.json`; `environment.json` (versions, CPU, code commit) |
| `results/hfs_tuning/` | H-FS parameter search: 36 settings on 16 tuning courses each of F1, F2 and MN (`tune.json`, `tune_rows.json`, `best.json`) |
| `videos/` | `mixed_narrow_edge`, `rhythm` and `lane_change__compare.mp4`: s1 on top, H-FS below, on three fixed courses; `videos.json` (command, policy SHA-256, outcomes) |

- In the videos, s1 completes all three courses. H-FS falls after 4 of 20 footprints on `mixed_narrow_edge`, completes
  `rhythm` with 28 % of the footprints within 5 cm, and stops after 18 of 19 footprints on `lane_change`.
- The run files and the two stored policy evaluations are byte-exact copies of the experiment's files; their paths
  name the experiment machines. `manifest.json` names internal design notes (not released) and a leg correction that
  these runs switch off (`action.alpha` 0 in `env_cfg.json`). The release code gives both evaluations again bitwise;
  it added the paired tests against `HFS_dev` to their `summary.json` and `SUMMARY.md` (`report --baseline`).
- The Hub copy of `release_manifest.json` also holds the provenance: run sources, code version, jobs, input hashes.

## Limits

- Simulation only. One robot and one walking policy; flat floor and flat plank tops at the floor height; straight
  plank courses; standing starts.
- The NRD models the planks only as a rule on the foot positions, and it misjudges plank courses (above).
- All numbers are development results. The sealed final test of the study was not run.
