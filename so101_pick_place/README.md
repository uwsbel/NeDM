# SO101 pick and place: a camera policy trained only in the NRD, tested on the real arm

One camera policy (the "student") drives a real SO101 arm. It picks a 30 mm cube from the table and puts it on a
plate. Every training step ran inside the NRD, a learned simulator of the arm and the cube (a contact NRD of the
design in [`../contact_nrd/`](../contact_nrd/README.md)). The chain has four parts:

1. NRD: the arm and cube dynamics, learned from CPU MuJoCo episodes.
2. Teacher: a state policy (it sees the true cube pose), trained with PPO in the NRD.
3. Student: a camera policy, trained in the NRD with ray-traced images of a calibrated copy of the real scene. The
   student drives and the teacher gives the labels (online DAgger).
4. Real arm: the student runs with no further training.

No real image was a training sample. Real data set the simulator, the training worlds and the checks:

- the camera calibration and the look of the scene (from two LeRobot teleop recordings) and the table height;
- from S3 on, robot-like servo ranges in the student's training worlds (speed, lag and sag, set from robot timing
  logs), the wrist-roll limit of 98 deg (T5 / S4: from the motor range and two robot stalls) and the 10 mm lower
  table (T6 / S5: the measured real mat is 8-17 mm below the sim table at the cube and plate positions);
- a fitted servo model (used only in the sim checks) and the choice of the student for the robot test.

Each fine-tune round (S3, T5 / S4, T6 / S5) answered failures of the student before it on the robot. Section
"Real-data inputs" lists where each input came from.

This is follow-on work to the NeDM paper, not part of it. The study history (every stage commit named below) is the
branch `so101-pick-place-study` (tip `3599103704032ba67cb2736ff0aee501f2e537b9`). The data, checkpoints, robot logs
and videos are in the folder `so101_pick_place/` of the Hugging Face dataset (section "Data").

## Results on the real arm

Released student: run `v12_tb10_s1210`, checkpoint `it_01000` ("tb10"), SHA-256 `6458a4f791d10521...`. It was
trained for a fixed table 10 mm lower than the nominal sim table. D ("reference") is the previous robot student and
the start checkpoint of tb10: run `v8_roll_s801`, checkpoint `it_02000` (SHA-256 `d9a35bfc...`). "Placed" = the
operator typed `p` (cube on the plate). Intervals: Wilson 95 %. Source of the counts: the 19 session logs on the Hub
(`robot/logs/`), counted by `scripts/so101_pick_place/robot/build_results.py` (per-episode table:
[`results/robot_results.md`](results/robot_results.md)). Source of the cube and plate positions, the directions and
the end distances: the Hub file `results/robot/stress_episodes.json`, an image + kinematics estimate whose script
was not found (it is not released).

| Test (2026-10-27 unless stated) | Student | Placed | 95 % interval |
|---|---|---|---|
| Stress test, 6 sessions | tb10 | **36/50 = 72 %** | 58-83 % |
| Stress test, plate left of the cube (R->L) | tb10 | 19/23 = 83 % | 63-93 % |
| Stress test, plate right of the cube (L->R) | tb10 | 17/27 = 63 % | 44-78 % |
| Screening, 4 episodes | tb10 | 4/4 | 51-100 % |
| Screening, 7 students, 4 episodes each (D 8) | all | 22/32 = 69 % (note 1) | 51-82 % |
| All sessions 2026-10-24 to 10-27 | D | 19/28 = 68 % (note 1) | 49-82 % |

Stress test, per session (in time order): 8/10, 2/2, 5/8, 7/10, 8/10, 6/10.

- Protocol, as run (it was not fixed before the test): the plate stayed in one place in each of the first 5 sessions
  (sessions `150916` and `151136` share one place); in the last session it was moved twice (3 places). Plate centre
  (mm from the base pan axis / deg): `145520` 200 / +18; `150916` + `151136` 189 / +31; `151953` 188 / -19; `153045`
  177 / -29; `153943` ep 1-4 204 / -11, ep 5-8 154 / -15, ep 9-10 170 / +19. The cube was moved to many positions:
  177-289 mm from the base pan axis, -47 to +40 deg, cube yaw about 0. One start that stopped at the home pose before
  the policy ran is not counted (`eval_20261027_150916/ep_03`). The study's note `STRESS_tb10_20261027.md` says the
  plate moved once; the positions above show two moves.
- Placed cubes: distance from the cube's end position to the plate centre (image + kinematics estimate) p50 20 mm,
  max 29 mm; the plate is a square with 50 mm sides (25 mm from the centre to an edge).
- Cubes closer than 19 cm to the pan axis: 1/7 placed. Cubes 19-29 cm: 35/43. 48 of the 50 cubes were inside the
  training range 15-27 cm (35/48 placed).
- D per day: 6/8, 6/8, 2/4, 5/8.
- Note 1: in two of D's episodes the operator typed `p` with a note that the cube was off the plate or the mat
  (`eval_20261027_133039/ep_02` "GLP PLACE OFF MAT", `eval_20261024_205852/ep_02` "GLP BUT PLACE IN THE WRONG
  SPOT"). Without them: D 17/28, screening 21/32 (D 4/8). tb10's notes have no such case: 3 placed tb10 episodes are
  typed "GLP OFF" (end distance 22-29 mm) and 13 "GLP BIT OFF"; all count as placed.

| Stress-test failure (14 of 50) | Episodes | Direction |
|---|---|---|
| Held the cube at the plate, no release (carry stop) | 6 | 5 L->R, 1 R->L |
| Grasp failed | 5 | 3 R->L (3 of the last 4 episodes of session `151136`: ep 5, 7, 8), 2 L->R (far cubes, 260 and 278 mm) |
| Released short of the plate | 3 | 3 L->R |

Screening (2026-10-27 13:30-14:04; 4 episodes per student, D at the start and at the end; the names are those of
`results/sim_checks/LAB_PLAN_v13.md` on the Hub):

| Student | Placed |
|---|---|
| D (reference, 2 rounds) | 5/8 |
| tb10: table 10 mm lower, 1,000 steps (released) | 4/4 |
| visual fix, seed 1231, 500 steps (`visR500`) | 2/4 |
| visual fix, seed 1304, 750 steps (`visR_s1304 it750`) | 3/4 |
| visual fix + table 5 mm lower, seed 1301, 500 steps (`visRtb05_s1301 it500`) | 2/4 |
| table 5 mm lower, 1,500 steps (`tb05`) | 3/4 |
| real-look gripper, no randomisation, 250 steps (`visR0_250`) | 3/4 |

Only tb10 and D are released. The other five students are students of the v12 and v13 runs (2026-10-26/27).

### How tb10 was chosen

- The frozen sim rule of its round (`results/sim_checks/SELECTION_v12.md` on the Hub) passed no student. tb10 failed
  G1 (3.5 %, nominal sim table; expected for a student trained for a lower table) and the place line of G3 +3 mm
  (94 places, line 144.9) (`results/sim_checks/SELECTION_RESULT_v12.md`).
- At the user's request, a relaxed lab rule (`LAB_PICK_RULE_v13.md`) was written on 2026-10-27 at 09:30, before the
  new gate results of that day. The v12 gate results of all 24 checkpoints, tb10's among them, were known when it was
  written. Its G3 place line is 90, and students trained for a lower table are exempt from G1, so tb10 passed just
  above the line (94).
- Five students met the lab rule, which allowed four. tb10 was the optional fifth ("opt. A" in `LAB_PLAN_v13.md`) and
  ran after D's closing round. One screened student (`visR0_250`, "opt. B") did not meet the lab rule (grasps 171,
  line 172).
- The user then chose tb10 for the stress test. The 50 stress episodes are new episodes, but the choice used robot
  results, and the stress protocol was not fixed in advance.

## Sim checks of the released student

Frozen runner of the round (`check_eval.py` sha256 `f70011ab`), AMD MI350X, CPU MuJoCo with GPU renders. J224 = the
224 task layouts (cube and plate) of the robot sessions; on them the check counts grasps, places and carry stops
(the arm holds the cube at the plate and does not release it), with the measured table plane, a flat robot floor,
the fitted servo model and 40 s episodes. D's reference: G2 = mean of 3 D runs; G3 = one D run each. Source:
`results/sim_checks/v12_tb10_s1210_it1000.json`, `D_reference.json`.

| Check | tb10 | D | Rule | Result |
|---|---|---|---|---|
| G1: success on the nominal sim table, 200 tasks (%) | 3.5 | 99.0 | D - 3 | fail (expected: trained for a table 10 mm lower) |
| G2: J224, grasps / places / carry stops | 184 / 143 / 4 | 193 / 156 / 28.7 | D -10.5 / -13.9 / +12.1 | pass |
| G2 per direction, L->R (112 tasks): grasps / places, sim rule / places, robot rule / carry stops | 101 / 63 / 85 / 4 | 102.7 / 78 / 90.3 / 18.0 | report | |
| G2 per direction, R->L (112 tasks) | 83 / 80 / 83 / 0 | 90.3 / 78 / 85.7 / 10.7 | report | |
| G3: table plane -3 mm | 181 / 154 / 1 | 170 / 126 / 22 | D -12.9 / -17.1 / +14.9 | pass |
| G3: table plane +3 mm | 180 / 94 / 3 | 184 / 162 / 13 | same | fail (places) |
| Release check: grasp releases at the 68 real hold starts x 4 draws, 2 seed blocks | 0 / 0 | 0 / 0 | 2+ starts in both blocks vetoes | no veto |

- In-run evaluation of the run (CPU MuJoCo, training plane, decision 1,500; there is none at 1,000): 97 % lenient /
  92 % strict on `dev100` (100 fixed tasks with random cube starts), 98 % / 94 % on `home45_dev100` (100 tasks that
  start at the robot's HOME pose, wrist roll 45 deg) (`student/s5_v12_tb10_s1210/progress.jsonl`). Lenient = the cube
  released and at rest inside the 50 mm goal square, any yaw; strict = within 15 mm and 15 deg of the goal.
- By the sim place rule, G2 shows a weaker L->R side (63 vs 80 of 112), like the robot (63 % vs 83 %); by the looser
  robot place rule it does not (85 vs 83). The G2 row uses the sim rule (63 + 80 = 143).
- There is no sealed sim test of the student.

## The NRD

| Item | Value |
|---|---|
| Design | Contact NRD of `contact_nrd/` (stored name `contact_graph_nrd_v6`): core, collision and contact networks, width 128, 4 heads, FFN 512, 4 layers, history 4, step 20 ms, 2,424,735 parameters |
| State | Arm 12 (6 joint angles, 6 joint rates), cube 13 (position, quaternion, velocity, angular velocity); the rates are 20 ms step averages |
| Action | 6 joint position targets |
| Contact pairs | fixed jaw - cube, moving jaw - cube, cube - table |
| Base NRD `c0_s62` | From scratch, seed 62, on `production_plus` (37,500 scripted episodes); best check at update 100,000; 8.5 GPU-h (AMD MI350X) |
| Policy NRD `c1_s62` | `c0_s62` fine-tuned, 20,000 updates at lr 1e-4, on the 4 data parts merged (train rows of `policy_v3` x 2.5); best check at update 2,500; 1.6 GPU-h. Every teacher and student stage ran in it |

| Data part (Hub `nrd_data/`) | Episodes (train / validation) | Records | What |
|---|---|---|---|
| `production_plus` | 34,647 / 2,853 | 49.3 h | Scripted grasps, 12 families (nominal, off-centre, one-sided, near miss, weak closure, drop, push and 5 probe families) |
| `noisy_v1` | 9,000 / 1,000 | 15.6 h | Scripted plans with policy-like command noise |
| `expert_v2` | 24,684 / 2,079 | 30.9 h | A scripted expert through the study's action decoder |
| `policy_v3` | 17,749 / 1,482 | 25.4 h | 7 PPO policies of the study, rolled out in MuJoCo (`policies/` holds them) |

All parts: CPU MuJoCo 3.15.0, 0.5 ms physics step, records every 10 ms, 4 s episodes; 93,494 episodes, 121 h.

Released files. Main's `nedm.contact_nrd.model.load_model` cannot load the files the study used: they hold two
training-only geometry heads (config key `aux_geometry`, tensors `geo_collision.*`, `geo_contact.*`, 3,096
parameters) that the step never uses. The release has both:

| Hub file | SHA-256 (16) | Loads with main's code | Use |
|---|---|---|---|
| `nrd/c1_s62/best.pt` | `2d1a24926d463c54` | yes | the policy NRD for every step of this release |
| `nrd/c1_s62/original/best.pt` | `f45ee8be2c1bb1e6` | no (training code, commit `f6f05c21c`) | the file the study used |
| `nrd/c0_s62/best.pt` | `01c44cd0424d4f0b` | yes | the base NRD |
| `nrd/c0_s62/original/best.pt` | `c2b44e7f89e617d5` | no | the base NRD as trained; the `init_from` of the fine-tune |

The main-compatible file is the original with only the key and the 4 tensors removed
(`scripts/so101_pick_place/nrd/strip_ckpt.py`); every other entry is equal bit for bit. Check
(`scripts/so101_pick_place/nrd/equiv.py`, `equiv_compare.py`): the original in the training code and the new file in
main's code give bit-identical 100-step rollouts (from the first state and from the recorded history at step 100),
contact switches and task metrics on 256 validation episodes (64 of each data part, `subsets.py`), on CPU and on the
GPU. `reproduce.sh` step `nrd_strip` repeats the strip (bytes equal to the released file with torch 2.12) and this
check.

Sealed test: 3,050 scripted episodes (Hub `nrd_test/`, SHA-256 `657e3b47`, same collector, another master seed),
run once on 2026-10-27 with main's `python -m nedm.contact_nrd.test` unchanged (RTX 5090, 104 s). The 7 scored
families (`select: true`): 2,000 test episodes; validation = the 1,803 validation episodes of `production_plus` in
those families, scored by the same code on the same GPU. "1 s windows": 1 s rollouts from the recorded 4-state
history, started every 0.5 s. "4 s": one rollout from the first recorded state. Source: Hub `results/nrd/`; table by
`scripts/so101_pick_place/nrd/summarize.py`. `reproduce.sh` steps `nrd_test` (test columns) and `nrd_val` (validation
columns, needs `--with-nrd-data`) rebuild both.

| Metric | c0 validation | c0 test | c1 validation | c1 test |
|---|---|---|---|---|
| Cube position, 1 s windows, p50 / p95 (mm) | 0.16 / 3.26 | 0.16 / 3.48 | 0.21 / 4.04 | 0.22 / 4.13 |
| Cube angle, 1 s windows, p95 (deg) | 2.98 | 2.94 | 3.06 | 3.04 |
| Arm joint RMS, 1 s windows, p95 (mrad) | 3.61 | 3.63 | 2.31 | 2.37 |
| Cube position at 4 s, p50 / p95 (mm) | 1.20 / 8.57 | 1.20 / 9.41 | 1.34 / 10.42 | 1.37 / 11.03 |
| Wrong contact events per recorded event | 0.338 | 0.341 | 0.351 | 0.356 |
| Lift / drop disagreement (%) | 1.44 / 0.50 | 1.55 / 0.65 | 2.55 / 0.94 | 2.45 / 0.90 |
| Selection score (lower is better) | 11.77 | 12.58 | 13.27 | 13.68 |

| Family: cube position, 1 s windows, p95 (mm) | nominal | off_centre | one_sided | near_miss | weak_closure | drop | push |
|---|---|---|---|---|---|---|---|
| c0 validation / test | 2.14 / 2.16 | 3.04 / 3.14 | 2.51 / 2.34 | 9.94 / 10.03 | 13.08 / **37.14** | 4.41 / 5.56 | 9.08 / 7.47 |
| c1 validation / test | 2.81 / 2.95 | 4.75 / 4.40 | 2.88 / 3.18 | 10.95 / 9.49 | 46.72 / 47.13 | 4.86 / 5.46 | 9.88 / 8.39 |

- The test agrees with validation. One outlier: base NRD weak_closure goes from 13.1 to 37.1 mm (the p95 of only 125
  episodes).
- All 12 families (3,050 episodes): 1 s cube p95 3.84 mm (c0), 4.61 mm (c1).
- The policy NRD's own validation set (19 selected families of its 4 data parts, 4,922 episodes, with the expert,
  noisy-command and policy episodes): 1 s cube p95 8.40 mm, cube angle p95 4.08 deg, 4 s cube p95 14.37 mm (training
  record `nrd/c1_s62/validation.json`; the merged training file is not released, so no step rebuilds it). It has no
  test counterpart: the test set has only scripted families.
- Neither NRD was selected with the test set. The test set is not fully blind: it was run once before (2026-10-10)
  on an older NRD design, and the base NRD's config was made after that run.
- Closed loop in the policy NRD, 3 PPO policies trained outside the NRD (among them a MuJoCo Warp policy and a
  kinematic-world policy) reach 87.5 / 78.9 / 82.0 % on 256 tasks (in MuJoCo 96.5 / 93.0 / 96.9 %, study notes, not
  re-checked), and the scripted expert 91.3 % on 1,024 tasks.

## Teacher and student

| | Teacher (T2-T6) | Student |
|---|---|---|
| Inputs | Privileged state: arm joints and rates, true cube pose and velocity, goal | Two RGB images, 224 x 224: a crop (52, 103, 288, 288) of the external camera and the full wrist camera; joint angles and rates. Two ResNet18 encoders (ImageNet start in S0) |
| Output | Joint targets | A chunk of 10 joint targets; 5 are executed per query (every 100 ms) |
| Training world | The NRD, 8,192 worlds | The NRD, 1,024 worlds, with ray-traced images of the calibrated scene (Newton / Warp ray tracer; batches of 128 worlds) |
| Labels | PPO reward | The teacher's targets at the states the student reaches (online DAgger) |

What each robot-round fine-tune added (configs in `configs/so101_policy/`):

- S3: robot-like servos in 80 % of the student's worlds (speed 0.3-1 x, lag 20-150 ms, sag up to 1.5 deg), image
  lag of 1-4 frames, and teacher labels at the commanded speed.
- T5 / S4: a wrist-roll limit of 98 deg (the real servo range).
- T6: the action floor (the lowest target height the action layer allows) 10 mm higher, on the sim table.
- S5: one fixed flat table 10 mm below the sim table in every world (the images show it; the real mat is lower),
  the action floor flat in the student's observed frame, and T6 as the labelling teacher.

T0 is a decoder policy: it outputs commands of a scripted action decoder. T1 records 24,576 NRD episodes of T0
through the decoder; T2 learns joint targets from them by behaviour cloning.

## Training chain

Each teacher starts from the teacher above it; each student starts from the student above it. GPU: one AMD MI350X per
job (S3: MI300X; T1 and T2: one RTX 5090). Commits are on the branch `so101-pick-place-study`. A GPU re-run gives a
similar, not an equal, checkpoint. Re-run types:

- "same code": this branch runs the code of the stage (options added later are off).
- "probably same code": the configs are equal and later options are off, but the stage's commit is inferred and no
  bit-identity test covers the commits in between.
- "same recipe": the commit of the stage is known only from file times.
- "recipe only": the code of the stage is in no commit or was changed later.

| Stage | Run | Start | Labels | Length | Checkpoint (SHA-256) | GPU-h | Commit | Re-run |
|---|---|---|---|---|---|---|---|---|
| NRD c0 | `full_ns_rk_cc_geo_cs_s62_plus` | nothing | - | 100,000 of 120,000 updates | `best.pt` (`c2b44e7f`) | 8.5 | uncommitted (`provenance/code_v16_base_nrd.tar.gz` on the Hub) | recipe only |
| NRD c1 | `nrd_ft_c1_s62` | NRD c0 | - | update 2,500 of 20,000 | `best.pt` (`f45ee8be`) | 1.6 | `f6f05c21c` | same code (study branch) |
| T0 | `nC1_s62_b` | nothing (PPO) | - | 171 iterations, 448 M steps | `model_00171.pt` (`8069d5b3`) | 3.3 | `e205b801b` + NRD code of `f6f05c21c` | same code |
| T1 | expert e0 / e1 / e2 | T0 | - | 3 x 8,192 NRD episodes | `e0_n1.npz`, `e1_n2.npz`, `e2_n15.npz` | 0.6 | uncommitted (close to `6ea7ddb6e`) | recipe only |
| T2 | `bc_all` | nothing (BC) | T1 | 60,000 steps | `bc.pt` (`95df5ff7`) | 0.05 | uncommitted (close to `6ea7ddb6e`) | recipe only |
| T3 | `jt_ppo_b_v3` | T2 | - | 140 iterations | `model_00140.pt` (`41693858`) | 0.9 | about `6ea7ddb6e` (not exact) | recipe only |
| T4 | `jt_ppo_h_hold25` | T3 | - | 110 iterations, hold limit 2.5 s | `model_00110.pt` (`0c1b6142`) | 0.4 | `6ea7ddb6e` (by file times) | same recipe (commit by file times) |
| T5 | `tppo_roll_v8_s21` | T4 | - | 200 iterations, roll limit 98 deg | `model_00200.pt` (`c4ea842c`) | 0.8 | `46f74a797` | same code |
| T6 | `tppo_v12_fr10_s122` | T5 | - | 200 iterations, floor raised 10 mm | `model_00200.pt` (`069cd896`) | 0.8 | `24b71df62` | same code |
| S0 | `v2_start_s2` | ImageNet ResNet18 | T4 | 4,096 + 2 x 2,048 episodes; 10,000 + 2 x 4,000 steps | `r2/last.pt` (`3fe3fb5a`) | 0.5 | `b38677cf1` (inferred) | probably same code (commit inferred, no bit test) |
| S1 | `v2_online_s18` | S0 | T4 | 3,000 decisions | `it_03000.pt` (`8fe355c7`) | 0.8 | `b38677cf1` (inferred) | probably same code (commit inferred, no bit test) |
| S2 | `v2_online2_s28` | S1 | T4 | 3,000 decisions | `it_03000.pt` (`68d08d0b`) | 0.8 | `b38677cf1` (inferred) | probably same code (commit inferred, no bit test) |
| S3 | `v4_relabel_s63` ("C1") | S2 | T4 | 2,000 of 3,000 decisions | `it_02000.pt` (`ceebbd2f`) | 1.6 | `977863c9e` (inferred) | recipe only here (exact code: `977863c9e`) |
| S4 | `v8_roll_s801` ("D") | S3 | T5 | 2,000 decisions | `it_02000.pt` (`d9a35bfc`) | 0.6 | `46f74a797` | same code |
| S5 | `v12_tb10_s1210` ("tb10") | S4 | T6 | 1,000 of 1,500 decisions | `it_01000.pt` (`6458a4f7`) | 0.5 | `362b48801` | same code |

- Total about 21.7 GPU-h (NRD 10.0, teachers 6.9, students 4.8) plus the CPU MuJoCo collection of the NRD data
  (each scripted campaign about 15 min on 24 workers; `expert_v2` 1.4 h, `policy_v3` 1.0 h on one 128-core node).
- "Inferred" (S0-S3): the commit comes from the launch notes and from the saved config keys; the cluster copy of the
  code was overwritten later, so there is no hash proof.
- S3: commit `550a58cb6` (2026-10-23) changed the labels at the commanded speed (the first decision of an episode is
  skipped). This branch has that change; the exact S3 code is `977863c9e`. Recipe for an exact S3 re-run (not run): a
  git worktree of `977863c9e`, the `s3` command of `reproduce.sh` (`vision_viral.py online` with the config
  `configs/so101_policy/vision/viral_r2s_online_relabel_v4.json`, seed 63) run in that worktree, with the paths of
  that commit's configs (start banks, scene, calibration, NRD) changed to the downloaded files under
  `artifacts/so101_pick_place/`.
- T3 and T4 were planned for 400 and 200 iterations; the jobs were cancelled after the used checkpoint. PPO here has
  an adaptive learning rate, so `reproduce.sh` stops at 140 and 110. T0 stopped at its 3.3 h time limit after 171
  iterations; `reproduce.sh` stops at 171 iterations.
- NRD c0: `configs/so101_pick_place/nrd/base_c0.json` says seed 61; the run used seed 62 (`--seed 62` on its command
  line, as in `reproduce.sh`).
- The policy stages (`t0` ... `s5`) run in the released policy NRD. Only `chain` (or `--parents own`) after an own
  `nrd_train` + `nrd_strip` runs them in the own NRD (`OUT/nrd_main/c1_s62/best.pt`).
- The commands of every stage are in `scripts/so101_pick_place/reproduce.sh` (steps `t0` ... `s5`); the run records on
  the Hub (`args.json`, `meta.json`, `manifest.json`, `cmdline.txt`) hold the original command lines.

### Real-data inputs

The outputs are small files in git (`configs/so101_pick_place/`, the same bytes in the Hub folder `scene_calib/`).
Paths of committed code are on the study branch. The exact command lines of these steps were not recorded.

| Output | Real input | Code | Re-run |
|---|---|---|---|
| `calib_v1.json`: camera models and poses, joint map, table plane, mat, plate per episode | the 2 LeRobot teleop recordings (Hub `robot/lerobot_recordings/`) | `scripts/so101_policy/real2sim/calib_*.py` (13 files, commit `0338f2c3d`, committed 47 min after the file was written). Order: `calib_detect`, `calib_wrist_det`, `calib_segment`, `calib_lines`, `calib_jaw`, `calib_colors`, `calib_run.py full` / `holdout`, `calib_export`, `calib_finalize` (usage in each docstring) | recipe |
| `real_scene_v2.json`: the look of the scene (colours, light, tone, crops) | the same recordings + `calib_v1.json` | study scripts in no commit (not released) | no |
| `parts_override_v2.json`: real-look parts of the arm (the release check) | the same recordings | study scripts in no commit (not released) | no |
| `dyn_fit_d2q.json`: fitted servo model of the sim checks | servo recordings of the robot (not released) | study scripts in no commit (not released) | no |
| Table height and the 10 mm lower table (T6 / S5); robot-like servo, lag and sag ranges (S3 on); wrist-roll limit 98 deg (T5 / S4) | robot logs of 2026-10-22 to 10-26 (the Hub has the sessions from 10-24 on) | values set by hand from the logs (the reasons are in the `note` of each config); the study's measurement scripts are `scripts/so101_policy/real2sim/sysid/` (`run_all.sh`, `table_height.py`, `lag_fit.py`, ..., commit `6f7ee661b`) | recipe |
| `so101_follower_lerobot_calib.json` | LeRobot motor calibration of the lab robot (`lerobot-calibrate`) | LeRobot | another robot needs its own |

## Reproduce

```bash
# install the torch / torchvision build for your GPU first (https://pytorch.org; tested: 2.12.0+cu130, 2.10.0+rocm7.1)
pip install -r requirements-so101-pick-place.txt     # not requirements.txt: it pins torch 2.8.0

scripts/so101_pick_place/reproduce.sh smoke   # setup check, tiny sizes: about 5 min on one RTX 5090 (files local)
scripts/so101_pick_place/reproduce.sh full    # T6 and S5 from the released T5 / S4 (study: 0.8 + 0.5 GPU-h), the sim checks (21 runs), the bundle
scripts/so101_pick_place/reproduce.sh chain --with-nrd-data   # every stage from scratch (about 22 GPU-h)
scripts/so101_pick_place/reproduce.sh full --steps download,nrd_test                    # the sealed NRD test
scripts/so101_pick_place/reproduce.sh full --steps download,nrd_val --with-nrd-data     # the NRD validation columns
```

`reproduce.sh --help` lists the steps and options. The download step fetches 0.86 GB by default (NRD, test set,
teachers, students, RL inputs, results, bundle, LeRobot recordings) and checks the size and SHA-256 of every file;
`--with-nrd-data` adds 10.6 GB (needed by `nrd_val`, `nrd_train` and `t1`), `--with-expert-data` 0.83 GB,
`--with-robot` 1.15 GB. `scripts/so101_pick_place/download.py` does the download alone; `--source DIR` (or
`SO101_PP_HF_SOURCE`) takes the files from a local folder with the Hub layout instead of the Hub. All data go to
`artifacts/so101_pick_place/` of the checkout: every default path of the code points there
(`src/nedm/so101_policy/paths.py`). Run from the repository root.

The sealed NRD test with main's code alone:

```bash
PYTHONPATH=src python -m nedm.contact_nrd.test --data artifacts/so101_pick_place/nrd_test \
    --model c1_s62=artifacts/so101_pick_place/nrd/c1_s62 c0_s62=artifacts/so101_pick_place/nrd/c0_s62 --output nrd_test.json
```

| Step | Check | Result in the smoke run (fresh clone, RTX 5090, torch 2.12.0) |
|---|---|---|
| download | size + SHA-256 of 173 files | all OK |
| geometry | kin points and task geometry rebuilt from MuJoCo vs `rl_inputs/geometry/` | arrays and files equal |
| nrd | main's `load_model` on both NRDs, 2 s rollouts | load, 2,424,735 parameters, finite |
| nrd_strip | `strip_ckpt.py` on the released originals; original in the training code vs the new file in main's code | bytes equal to the released `best.pt` (both NRDs); rollouts, contact switches and metrics bit-identical |
| nrd_test | main's test code on 24 test episodes vs the released test | per-episode values equal to 2e-6 (c1) and 3e-3 (c0): other batch, float noise |
| nrd_collect | the collector regenerates 1 test episode per family | 12 of 12 bit for bit |
| t6 | 2 PPO iterations, 256 worlds | checkpoint bit-equal to the study code (`359910370`) with the original NRD file |
| s5 | 12 online decisions, 64 worlds, renderer | rows before the first training step equal to the study code; then the loss differs by GPU noise (0.0718 vs 0.0710 after the first update; two runs of the same study commit differ by as much) |
| simcheck | G2-type run (one G2 evaluation, cut to 4 J224 tasks and 300 steps) | every array equal to the frozen runner `f70011ab` with the code of `362b48801` (timing keys aside); metrics equal |
| bundle, fake | build, robot_test.sh check / eval 1 6 / replay of 3 recordings on a fake robot | replay outputs (joint inputs, images, action chunks) bit-equal to the released robot bundle |

- GPU runs do not repeat bit for bit across machines; compare rates (the `summary` step prints them next to the
  released records). CPU MuJoCo and the renderer repeat on one machine with the same package versions.
- GPU memory: every full-size stage ran on AMD MI350X / MI300X (192-288 GB); `full` was not run on a 32 GB card.
  T6 logged up to 7.8 GB of torch GPU memory (`gpu_mem_gb` in its `learning_curve.csv`); the student stages did not
  log it. On a smaller GPU, lower the render batch (`--render-worlds`) first.
- Teacher runs need `NEDM_SO101_DYN` unset (`reproduce.sh` unsets it); the sim checks set it to
  `configs/so101_pick_place/dyn_fit_d2q.json` (`run_gates.sh`).
- NRD retraining (step `nrd_train`) uses the training code of commit `f6f05c21c` (main's `train.py` lacks its options:
  push guard, history noise, loss scales, contact sampling, co-training, the geometry heads, `init_from`). The step
  makes a git worktree of that commit, writes the fine-tune config with an absolute `init_from` (the own base run with
  `--parents own`, else the released `nrd/c0_s62/original/best.pt`) and checks the merged data against the fine-tune's
  episode counts. The base NRD ran on uncommitted code (`provenance/` on the Hub); `f6f05c21c` is the closest commit.
  A re-trained NRD is in the training format: step `nrd_strip` makes its main-compatible file.
- `expert_v2` and `policy_v3` are not re-collected (no step). `policy_v3`, recipe (not run): commit `e205b801b`,
  `scripts/so101_policy/collect_round.py` `ptasks` -> `prun` (with the 7 policies of `nrd_data/policy_v3/policies/`)
  -> `label` -> `write --campaign policy_v3` (usage in its docstring). `expert_v2` is recipe only: its collector is
  in no commit (`provenance/round2_expert_v2_code.tar.gz` on the Hub). The collector of this branch regenerates the
  scripted parts bit for bit (checked on the first episode of every family: production 12/12, extra 4/4, noisy_v1
  4/4, test 12/12); `chain` collects the 4 scripted campaigns and compares their episode counts with the released
  manifests.
- Robot videos: `python3 scripts/so101_pick_place/robot/render_videos.py all --logs artifacts/so101_pick_place/robot/logs
  --results artifacts/so101_pick_place/results/robot/results.json --out <dir>` (needs `--with-robot`, opencv and an
  ffmpeg with libx264).

## Deploy on the robot

The robot computer (Jetson AGX Orin, conda env `lerobot`: torch, torchvision, numpy, opencv-python, pyarrow, LeRobot)
needs no MuJoCo, Warp or Newton: the bundle carries the kin points (the collision points of the arm). The study's
notes record LeRobot 0.6.2 on the Orin: the version string of LeRobot's GitHub main branch after the v0.6.1 tag, so a
source install whose commit was not recorded (not a PyPI release). PyPI `lerobot==0.6.1` ran the replay and the fake robot on the
workstation, not the real arm (`requirements-so101-pick-place.txt`, section 2).
[`scripts/so101_policy/real2sim/hw/README.md`](../scripts/so101_policy/real2sim/hw/README.md) holds study notes of
2026-10-24 (older students, the study's paths); the steps below are the current guide.

1. Take the tested folder `bundle/so101_hw_v13_tb10/` (or its `.tar.gz`) from the Hub, or build it:
   `reproduce.sh full --steps download,bundle,fake`. A bundle built from this branch differs from the tested folder in
   14 files (11 code modules with the changes of section "Code", 2 doc files with a header that marks them as study
   notes, the build time in `deploy.json`), has 2 small new modules (`paths.py`, `npz_rows.py`) and lacks 7 study
   modules that the robot never loads. Its replay outputs are bit-equal to those of the tested folder (smoke run).
   The Hub keeps no file modes: use the `.tar.gz`, or run `chmod +x robot_test.sh` in the downloaded folder.
2. Copy it to the robot computer; check `sed -n '4,$p' MANIFEST.txt | sha256sum -c --quiet` in the folder.
3. LeRobot motor calibration: the robot needs its own (`lerobot-calibrate`, in
   `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json`);
   `configs/so101_pick_place/so101_follower_lerobot_calib.json` is the lab robot's file (replay and fake robot only).
4. In the folder, in an interactive terminal: `./robot_test.sh check`, `DATA_ROOT=<folder with the Hub
   robot/lerobot_recordings/> ./robot_test.sh replay` (default `~/lerobot/data`), `dry 30`, `live 15`, then
   `eval 10 40` (10 episodes of 40 s). Logs: `~/so101_hw_runs/<mode>_<time>/` (the layout of `robot/logs/`).
5. Each episode: home move (type `MOVE`), place the cube and the plate, type `RUN`; type the result (`g` grasp,
   `l` lift, `p` placed, then a note), `r` back to home, `s` end. Stop keys: q, space, Esc, Ctrl+C.
6. Guard of the eval mode (`assets/deploy.json`): command floor -15 mm, measured floor -20 mm, joint and box limits,
   camera age, blocked joints.
7. Scene: cube and plate 15-27 cm from the shoulder pan axis, within +-75 deg of straight ahead, 6-30 cm apart (the
   training task area). On this robot, cubes closer than 19 cm failed 6 of 7 times.

The USB port and camera paths of the lab robot are written in `run_student_real.py` (`FOLLOWER_PORT`, `CAM_PATHS`);
another robot needs its own values and its own camera calibration (`configs/so101_pick_place/calib_v1.json` is
specific to this camera mount; section "Real-data inputs").

## Code

| Path | Contents |
|---|---|
| `src/nedm/so101_policy/` | Task, NRD worlds (`nrd_sim.py`), CPU MuJoCo worlds (`mujoco_env.py`), joint-target environment and teacher (`joint_env.py`, `joint_teacher.py`), decoder PPO (`vec_env.py`, `decoder.py`, `ppo.py`), kinematics, table rules; `paths.py` (data folder) |
| `src/nedm/so101_policy/vision/` | The student, its online DAgger loop and images (`viral.py`), the tray / ACT code it builds on, datasets |
| `src/nedm/render/` | The ray-traced renderer of the student's cameras (Newton / Warp), the calibrated camera models, the scene |
| `src/nedm/contact_nrd/collect/` | The MuJoCo collector of the NRD data and its checks (new sub-package of main's `contact_nrd`) |
| `scripts/so101_policy/vision_viral_teacher.py`, `vision_viral.py`, `train_ppo.py` | Teacher (tasks, expert, BC, PPO), student (`run`, `online`, `eval`), decoder PPO (T0) |
| `scripts/so101_policy/real2sim/hw/` | Robot runner, bundle builder, robot test launcher (the files of the robot bundle) |
| `scripts/so101_pick_place/` | `reproduce.sh`, `download.py`, sim checks (`check/`), NRD release checks (`nrd/`: strip, validation subsets, rollout comparison, tables), robot result tables and videos (`robot/`) |
| `configs/so101_policy/`, `configs/so101_pick_place/` | Stage configs; camera calibration, scene, servo fit, real-look parts, LeRobot motor calibration, teacher env configs, NRD configs |
| `configs/contact_nrd/collection/so101_grasp.json`, `contact_nrd/so101_grasp/` | Collector config; the pinned MuJoCo Menagerie fetch |

The code files are copies of commit `359910370` of `so101-pick-place-study` (= `2b6dc4d8d`, the robot bundle, for
every code file). Changes against the study code, all in the copied files (no main file changed):

| Change | Why | Check |
|---|---|---|
| `nrd_sim.py`: no derived / relative channels, pair preview or contact mask | runs on main's `ContactNRD` (this NRD has none of them) | teacher checkpoint bit-equal (smoke) |
| `mujoco_env.step_rates`: main's `data.step_average_rates` | the study's `_step_average` is not on main | bit-identical on 4,096 random states |
| `task.load_start_bank`: `npz_rows.npz_members` (new file) | `contact_nrd/rollouts.py` is not added | same function |
| Default data paths: `artifacts/so101_pick_place/...` (`paths.py`), interpreter of the MuJoCo workers = the running Python | the study read home-folder copies of the same files | same files (SHA-256) |
| Config paths: start banks, scene, NRD `init_from` -> `artifacts/so101_pick_place/...`, `configs/so101_pick_place/...` | the study configs named cluster or home paths | - |
| Robot scripts: workstation defaults of the calibration, LeRobot calibration, recordings, kin points, output folder (`~/so101_hw_runs`), bundle Python (`python3`) | the same rule; the robot computer gets its paths from `robot_test.sh` | fake replay equal to the tested bundle |
| Study options whose modules are not released fail with a clear "not part of this release" error: worlds `kin` / `mjwarp` (`train_ppo.py --world nrd` only), the FSM expert act (`ExpertAct`), the Madrona backend, the gripper-look / base add-on / background randomisation of the renderer; `check_eval.py` keeps only the released scene family | the release carries only the code of the released chain; no released config uses these options | smoke runs |
| Sim-check and robot-result scripts: paths are arguments; `check_eval.py` without its `build68` command; `build_results.py` reads the bundle records from the released `results.json` | ported from the study folders | same arrays / same tables |

`real_scene_v2.json` keeps the study's home path of the calibration inside it (its SHA-256 is pinned in every
student); commands pass `calib=...` (as every run did). Comments and docstrings of the copied code cite study notes
(`~/so101_policy/docs/...`, `~/so101_r2s_claude/...`) that are not released.

## Data

Hugging Face dataset [harryzhang1018/NeDM](https://huggingface.co/datasets/harryzhang1018/NeDM), folder
`so101_pick_place/` (503 files, 13.3 GB), pinned in [`release_manifest.json`](release_manifest.json) (revision,
bytes and SHA-256 of every file; the Hub copy also holds the provenance: commits, run records, sources).

| Path | Contents | Size |
|---|---|---|
| `nrd/c1_s62/`, `nrd/c0_s62/` | Main-compatible `best.pt`, the original in `original/`, training records (`run_config.json`, `train_log.jsonl`, `validation.json`, `complete.json`, `contact_scale.json`, `core_validation.json`) | 65 MB |
| `nrd_data/` | The 4 data parts (`unified_data.npz`, `system.json`, `manifest.json`); `policy_v3/policies/` | 10.6 GB |
| `nrd_test/` | The sealed scripted test set | 0.34 GB |
| `teacher/t0_*` ... `t6_*` | Teacher checkpoints and run records; `t1_expert_data/` (24,576 NRD episodes, 0.83 GB) | 0.91 GB |
| `student/s0_*` ... `s5_*` | Student checkpoints (S4 = D, S5 = tb10) and run records (`args.json`, `meta.json`, `curve.csv`, `progress.jsonl`) | 0.15 GB |
| `rl_inputs/` | Task sets (`tasks/`), start banks (`banks/`), geometry, the task sets of the sim checks (`check_tasks/`). Released inputs: the recipes of the task sets, mixes and start banks are in the docstrings of `vision_viral_teacher.py` (`tasks`, `mix`, `starts`) and `train_ppo.py` (`bank`); `geometry/` is rebuilt by the `geometry` step; the sim-check sets J224 and H68 (from the robot logs) have no rebuild step here | 28 MB |
| `results/` | `nrd/` (sealed test, validation re-score, tables), `sim_checks/` (gate results, selection rules), `robot/` (per-episode results, stress-test geometry, the study's test notes) | 27 MB |
| `robot/logs/` | 19 sessions: 6 stress (tb10), 8 screening, 5 of D; per episode `episode_live.json` + `.npz` (50 Hz joints and commands, guard events, each query with its action chunk and the two input images at 112 px) | 0.96 GB |
| `robot/videos/` | 50 stress-test videos (one per episode, real time) and 2 compilations (stress test, screening, 4x) | 65 MB |
| `robot/lerobot_recordings/` | 2 LeRobot teleop recordings (15 episodes): calibration input and replay input | 0.12 GB |
| `bundle/` | `so101_hw_v13_tb10/`: the folder installed on the robot computer (`MANIFEST.txt`), and its `.tar.gz` | 49 MB |
| `scene_calib/`, `provenance/` | The same bytes as `configs/so101_pick_place/`; code snapshots that are in no commit | 2 MB |

- The videos show the student's input images as logged on the robot (112 px, external | wrist, about 7.9 Hz; the
  student sees them at 224 px). There is no full-resolution or third-person video.
- The robot logs hold lab background images (`first_frames.npz`), the robot computer's user path and the USB
  adapter serial; the code holds the same serial.

## Limits

- One robot, one lab, one cube and one plate; the stress test took about 1 h. The camera calibration belongs to this
  camera mount.
- The outcome is typed by the operator. The stress protocol was not fixed in advance, and the student was chosen with
  robot results (section "How tb10 was chosen").
- The logs do not record a checkpoint hash: the student of each session is known from the bundle folder named in the
  logs and the study's local copy of that bundle; the files on the robot computer were not compared with
  `MANIFEST.txt`.
- The cube and plate positions, directions and end distances are an image + kinematics estimate whose script is not
  released.
- L->R is weaker than R->L (63 % vs 83 %). Cubes near the base fail (1/7 under 19 cm).
- The release detector of the earlier sim checks false-fires on tb10: on the 50 robot stress episodes it flags 19
  releases (tb10 carries the cube low; the traces show the jaw closed until the plate). A slightly higher table hurts
  placing in simulation (G3 +3 mm: 94 places).
- There is no sealed sim test of the student. The NRD test set is not fully blind and has only scripted families.
- No stage repeats bit for bit on a GPU. T1-T3, S3 (with this branch) and the base NRD are recipe only; the S0-S3
  commits are inferred. The scene look, the real-look parts and the servo fit cannot be made again from this release.
