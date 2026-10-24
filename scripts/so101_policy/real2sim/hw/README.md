# SO101 camera student on the real robot

Status 2026-10-24: eval sessions on the robot on 2026-10-22 / 23 (logs `~/so101_r2s_claude/robot_logs`); the
changes of 2026-10-24 (next section) ran only on the fake robot and in replay. First status (2026-10-22, before the
robot): student for the robot: `robot_v2_s28_it03000.pt` (AMD `runs/v2_online2_s28/ckpt/it_03000.pt`; cameras
`external` + `wrist`, 224 px, crops pinned in the checkpoint: external [52, 103, 288, 288], wrist [0, 0, 640, 480]).
It was chosen for the robot because it keeps 89-91 % success in CPU MuJoCo on the measured (lower, tilted) real table,
held-out tasks; the sim-best student (`final_v2_s26_it03000.pt`)
falls to 36-48 % there (`~/so101_r2s_claude/eval/SELECTION_RULE.md`). Checks done on luffy only (logs in
`~/so101_r2s_claude/hw_final/`, made with the sim-best student; the code path is the same): replay of 3 recordings
from the extracted bundle equals the worktree run exactly; dry and live loops on the fake robot; the HOME check from
the 15 recorded rest poses.

| File | What |
|---|---|
| `run_student_real.py` | modes `replay` (recording, no robot), `dry` (robot read only), `live` (moves the robot) |
| `hw_lib.py` | joint map, limits, image crop, action layer, hardware guard, joint paths, student wrapper, fake robot |
| `make_orin_bundle.sh` | builds the folder + `.tar.gz` for the Orin (sources, collision points, student, calib_v1.json, `assets/deploy.json`) |
| `robot_test.sh` | one command for each test on the Orin: check / replay / dry / live / live-tilt / eval (bundle root) |
| `tests/so101_policy/test_real2sim_hw.py` | 32 unit tests (CPU, about 15 s) |
| `tests/so101_policy/test_real2sim_hw_deploy.py` | 13 unit tests: anti-windup, floor mode, deploy.json, bundle, HOME, no two-student mode |

## Changes 2026-10-24 (read before the next robot day)

| Item | Before | Now |
|---|---|---|
| HOME (all bundles) | sim (0, -5, -15, 92, **45**, 30) deg | sim (0, -5, -15, 92, **0**, 30) deg = LeRobot (0.0, -3.2, -6.5, 93.1, 75.8, gripper 31.1). Sim tests with the real roll stop: robot-feasible success C1 59-66 % -> 87-88 %, s28 60-68 % -> 95 %. |
| Floor | one flat floor for every student | floor mode from the student checkpoint (section "Floor modes"): `flat` (C1, s28, roll-cap students: as before), `plane` (v8 table-trained students: floor above the calibrated mat) |
| Servo-error integral | wound up at a joint's range end (2 stalls "wrist_roll blocked" on 10-23) | anti-windup at the LeRobot-unit range (`GuardConfig.i_antiwindup`, on) |
| Two students in one episode (commit 39d2e5b72) | a second-student flag and bundle asset | removed (no policy stitching) |
| Bundle | student + calib | + `assets/deploy.json` (floor mode, table plane source, HOME, eval guard; `robot_test.sh` reads it) |

## Floor modes (the table floor; `run_student_real.py --floor-mode`, `assets/deploy.json`)

The action-layer floor must have the clearance above the real mat that the student was trained with (sysid
`~/so101_r2s_claude/v8/sysid/sysid_v8.json`, coupling rule). The real mat is 8.3-17.5 mm below the sim table at the
observed cube and plate positions (calibrated plane z = -4.7 mm - 36.4 x + 18.4 y, mm per m; +-3 mm). The bundle
script reads the mode from the checkpoint (`v8_online.table_bias` present = plane); live mode refuses another mode
(`--allow-floor-mismatch` overrides). The banner of every run prints the mode and these numbers.

| Student type | Floor mode | Eval settings (`robot_test.sh eval`) | Relative to the real mat at the observed positions |
|---|---|---|---|
| Trained on the sim table: C1, s28, roll-cap students | `flat` | `--floor-mode flat --table-z 0 --home 0,-5,-15,92,0,30 --amp 0 --guard '{"cmd_floor_m":-0.015,"hard_floor_m":-0.02,"i_time_s":0.2,"i_max_deg":3.0,"i_antiwindup":true}'` (floors as on 10-22 / 23) | floor +9.8 .. +19.0 mm; command stop -6.7 .. +2.5 mm; measured stop -11.7 .. -2.5 mm |
| Trained with the v8 table randomisation (checkpoint `v8_online.table_bias`) | `plane` | `--floor-mode plane --table-plane calib --home 0,-5,-15,92,0,30 --amp 0 --guard '{"cmd_floor_m":-0.002,"hard_floor_m":-0.005,"i_time_s":0.2,"i_max_deg":3.0,"i_antiwindup":true}'` | floor +1.5 mm (as in training: 1.5 mm above the true table); command stop -2.0 mm; measured stop -5.0 mm (everywhere on the plane) |

Plane mode measures the floor stops above the plane, so the flat eval stops (-15 / -20 mm) on top of the plane would
put them 15-20 mm into the mat (double lowering): live refuses plane-mode stops deeper than 5 mm (command) / 8 mm
(measured). The plane floor is not used for the flat students: they were trained with the floor at sim z 0, and
the plane would let them press 8-18 mm lower than in training. With the plane floor and the servo-error integral, a
student that pushes down on the plate or the mat can reach the command stop (2 mm below the mat); that ends the
episode ("command floor") and is the intended protection.

## Servo-error integral: anti-windup (2026-10-24)

On 10-23 two eval episodes ended with "stall: wrist_roll blocked" (eval_20261023_141614 ep 6, eval_20261023_144431
ep 1): the measured roll stayed at 100.69 sim deg (the real stop), the target 100.9-101.1 deg, the command clipped at
101.21 deg (LeRobot range end), and the integral correction grew to +3 deg; the stall check compares the measured q
with the command without the correction (101.21 - 3 = 98.2 deg): 2.5 deg >= 0.9 x lead band for 25 steps. Now
(`GuardConfig.i_antiwindup`, default on): on a joint whose target is at / beyond its LeRobot-unit range, or whose
corrected command would be clipped there, the error is not integrated, and the correction never pushes the command
past the range. Offline replay of the Guard on the two logged episodes (`~/so101_r2s_claude/v8/deploy/stall_replay.json`):
the old rule reproduces the logged commands exactly and stalls at the logged step; the new rule has no stall, the same
commands on the other 5 joints at every step, and the roll differs only after the first step at the range end (by up
to 0.65 / 0.37 deg, the removed wind-up; the integral stays <= 0.67 / 0.52 deg instead of 3.0). All 102 logged
robot episodes with the integral on (144,785 steps, `~/so101_r2s_claude/v8/deploy/stall_replay_all.json`): the old
rule reproduces 100 logs exactly (2 early live runs of 10-22 ran older guard code) and all 9 logged stalls (8 wrist
roll, 1 shoulder pan, every one a wind-up at the range end); the new rule: 0 stalls, commands changed only on the
pan (417 steps) and the roll (13,355 steps), always after the first step at the range end. The replay is open loop
(the logged q), so the later steps of a changed episode are not what the robot would have done. `{"i_antiwindup":
false}` in `--guard` gives the old rule. Training has the same rule as an option, off by default
(`joint_env.RobotDynConfig.i_antiwindup`).

## Closed gripper at the start (fixed 2026-10-22)

A closed real gripper (LeRobot < 3) is below the sim jaw limit; 9 of the 15 recordings start like that. The HOME and
return path checks now accept a measured end pose outside the sim range (the path stays between its ends). The
calibrated LeRobot range is still checked.

## 1. Build (luffy) and copy (you)

```bash
cd /home/harry/.t3/worktrees/NeDM-push-ppo/r2s-v8                         # branch claude/r2s-v8
scripts/so101_policy/real2sim/hw/make_orin_bundle.sh ~/so101_r2s_claude/ckpt/c1_relabel_s63_it02000.pt \
    ~/so101_r2s_claude/v8/deploy/bundles/so101_hw_c1                       # prints "floor mode flat"
scripts/so101_policy/real2sim/hw/make_orin_bundle.sh <v8 student .pt> \
    ~/so101_r2s_claude/v8/deploy/bundles/so101_hw_v8                       # prints "floor mode plane"
scp ~/so101_r2s_claude/v8/deploy/bundles/so101_hw_c1.tar.gz sbel@proxy:~/      # you do this
```

The script reads the checkpoint (Python with torch: `$PYTHON`, default the so101_vision env) and writes
`assets/deploy.json` (floor mode, table plane source, HOME, eval guard; `HOME_SIM_DEG=a,b,c,d,e,f` sets another
HOME). MANIFEST.txt: 3 header lines (built, student + floor mode, calib), then the sha256 of every file.

On the Orin: `cd ~ && tar xzf so101_hw_c1.tar.gz && cd so101_hw_c1 && sed -n '4,$p' MANIFEST.txt | sha256sum -c --quiet`
(no output = all files intact). Needs only the `lerobot` conda env (torch, numpy, cv2, pyarrow, LeRobot 0.6.2); no
mujoco. Run every command below from the bundle folder (e.g. `~/so101_hw_c1`), in an interactive terminal (desktop or `ssh -t`): the stop
keys q / space / Esc need a TTY, else only Ctrl+C stops. Do not pipe into `tee` (Ctrl+C also stops `tee`); the
`--out` folder keeps the log.

## 2. Commands on the Orin

Short form (`robot_test.sh`, in the bundle root; it activates `lerobot`, writes to `~/so101_hw_runs/<mode>_<time>`,
refuses dry / live without a TTY or with changed bundle files; `[s]` = the time limit; it refuses `--time-limit`,
`--q-offset-deg`, `--out` in the extra arguments, and `--floor-mode`, `--table-z`, `--table-plane`, `--home`: every
mode takes the floor mode, the table plane source and HOME from `assets/deploy.json`; the banner prints them). `run_student_real.py` dry / live refuses, just before it opens the
port and the cameras, while one of your processes has one of them open, or while a LeRobot robot tool
(`lerobot-teleoperate`, `-record`, `-calibrate`, ...) or another dry / live run of any user runs. The open files of
other users' processes (root, docker) are not visible: stop those yourself.

```bash
cd ~/so101_hw_c1
./robot_test.sh check        # files, deploy settings, env, port, cameras, calibration (opens no device)
./robot_test.sh replay       # a) below
./robot_test.sh dry          # b) below, 30 s
./robot_test.sh live         # c) below, 15 s  (live-tilt = c) + --q-offset-deg 0,-2.086,0,0,0)
./robot_test.sh eval [n] [s] # d) below: n episodes in a row (default 10 x 40 s)
```

d) Eval session (first robot day, 2026-10-22: grasp, lift and carry worked with these settings). `eval` = live with
the floor mode and HOME of `assets/deploy.json`, `--amp 0`, `--guard <eval_guard of deploy.json>` and
`--episodes-n n` (the table in "Floor modes": flat students `{"cmd_floor_m":-0.015,"hard_floor_m":-0.02,
"i_time_s":0.2,"i_max_deg":3.0,"i_antiwindup":true}`, plane students `{"cmd_floor_m":-0.002,"hard_floor_m":-0.005,
"i_time_s":0.2,"i_max_deg":3.0,"i_antiwindup":true}`). Why: for the flat students the real table is 8-18 mm below the
sim table (the default floor stops at -2 / -5 mm ended good grasps); without the integral correction the real servos
(P 16, gearbox friction) did not follow the student's 0.45 deg lift steps after the grasp and the arm stayed on the
table. Flow: MOVE (HOME) -> each episode: reset the scene, RUN,
the student (space ends the episode), type the result (g grasp, l lift, p placed, for example `glp`; a note after a
space), `r` = back to HOME for the next episode (the gripper opens), `s` = end -> menu (r return to the start pose,
t torque off). Logs: `<out>/ep_NN/episode_live.{npz,json}` (outcome in the json), `<out>/session.json`.

e) Joint-speed input gain (`--qd-gain g`, default 1 = off; 2026-10-22 evening). The student's step size follows the
measured joint speed; the real arm is about 5 x slower than the sim arm, so near the plate the steps shrink until the
arm holds still and never releases (10 of 12 eval episodes). `--qd-gain` multiplies only the student's qdot input (the
guard and the log keep the measured qdot). CPU MuJoCo, today's student, 12 real layouts x 4: robot-like dynamics
50 / 88 / 94 / 83 / 48 % at gain 1 / 2 / 3 / 4 / 5; nominal sim 98 / 90 / 75 % at 1 / 2 / 3 (more arm-table contacts).
Test plan: QDGAIN_TEST.md. Example: `./robot_test.sh eval 5 --qd-gain 2`.

Long form:

```bash
conda activate lerobot
cd ~/so101_hw_robot
S="--student assets/student.pt --calib assets/calib_v1.json"
F="--floor-mode flat --table-z 0 --home 0,-5,-15,92,0,30"      # plane students: --floor-mode plane --table-plane calib ...
# a) replay (no robot; Orin inference time; compare with luffy)
python scripts/so101_policy/real2sim/hw/run_student_real.py replay $S $F --data-root /home/sbel/lerobot/data \
    --episodes green_cube_red_plate:0,green_cube_red_plate:2,green_cube_red_plate_1:5 --out ~/so101_hw_runs/replay_orin
# b) dry (robot and cameras connected, NO motor writes, torque untouched; stop other LeRobot processes first)
python scripts/so101_policy/real2sim/hw/run_student_real.py dry $S $F --out ~/so101_hw_runs/dry_1 --time-limit 30
# c) live (moves the robot)
python scripts/so101_policy/real2sim/hw/run_student_real.py live --i-am-supervising $S $F \
    --out ~/so101_hw_runs/live_1 --time-limit 15
```

Defaults used by these commands: HOME = sim (0, -5, -15, 92, 0, 30) deg = LeRobot (0.0, -3.2, -6.5, 93.1, 75.8,
gripper 31.1) (`hw_lib.ROBOT_HOME_SIM_DEG`; before 2026-10-24 the wrist roll was 45 sim = 120.8 real deg); HOME speed
15 deg/s peak; `--floor-mode auto` (the student's: flat `--table-z 0` or plane `--table-plane calib`); `--configure
auto`; guard = `GuardConfig` defaults (table below). The run checks that `--calib` is the calibration the student was trained with (live refuses a mismatch).
Luffy replay of a) with the robot student: cube error before the grasp 25.3 / 21.1 / 28.9 mm, plate 15.2 / 12.8 /
12.1 mm (bf16 on another GPU: small differences are normal). About 16 mm of the cube error is height (next section).

## 3. Pre-run checklist (live)

1. Dry run first. Check: the printed q matches the arm; `first_frames.npz` / the logged thumbnails show the fixed
   camera as `external` and the gripper camera as `wrist` (a swap is not detected); the end of the run prints the
   model and step time (p50 / p95). Gripper >= 5 (above).
2. Arm in the folded rest pose. The HOME move takes about 10 s at <= 15 deg/s; the wrist roll turns to 75.8 deg real
   (sim 0; before 2026-10-24: 120.8 deg real).
3. Cube and plate: both 0.15-0.27 m from the shoulder pan axis (3.9 cm in front of the base origin), within +-75 deg
   of straight ahead, 6-30 cm apart, plate yaw within 30 deg of the cube yaw (mod 90). Place them after HOME, before
   you type `RUN`. (8 of the 15 recorded cubes were 0.28-0.34 m out: outside the training region.)
4. Table height: the real table is 4-17 mm below the sim table over that region (tilted about 2 deg; 10-18 mm at the
   15 recorded cubes). Keep `--table-z 0`: the gripper floor then stays 5 mm or more above the real table.
   CPU MuJoCo with this table (held-out tasks, 100 each): the robot student 89 % (normal starts) / 91 % (HOME).
   Optional correction `--q-offset-deg 0,-2.086,0,0,0` (shoulder_lift joint-map offset = the x tilt of the fitted
   table plane; correct only if the arm base is really tilted, not for a shoulder_lift calibration error): 90 / 94 %.
   A flat table 10 mm lower: 96 / 98 %. Level table: 98 / 98 %. If you can, make the table level with the arm base.
   `--table-z` is one flat height: a value below -0.005 lets the gripper reach the table at the high side.
   Students trained with the v8 table randomisation (checkpoint `v8_online.table_bias`; configs
   `configs/so101_policy/vision/viral_r2s_online_v8_*`) learned to go down to the real table: they run in floor mode
   plane (section "Floor modes"; the bundle and `--floor-mode auto` choose it): the floor and the guard's floor
   checks 1.5 mm / -5 mm / -2 mm above the calibrated table plane, never `--table-z 0` (that floor is ~13 mm above the
   real table at the cubes and holds the descent up) and never the flat eval stops on top of the plane.
5. Workspace clear, you at the robot, one hand near the arm, the other on the keyboard (q / space / Esc / Ctrl+C).
   The student may command up to 0.06 rad per 20 ms step (3 rad/s); the sim teacher's per-step change p99 is
   0.03-0.04 rad.
6. At the end the torque stays ON (holding). Menu: `r` slow return to the start pose, then `t` + `TORQUE OFF`
   with a hand under the arm. `h` (or Enter) exits holding.

## 4. What the live run does

1. Before any write: LeRobot calibration file present and equal to the motors' registers; 3 joint reads finite and
   inside the calibrated range; both cameras 640 x 480, frames younger than 200 ms.
2. `--configure auto`: only if a register differs from LeRobot's `configure` values, it asks for `CONFIGURE` (torque
   off for a moment; the arm must rest on its stops).
3. HOME: the joint path is checked (sim joint range, calibrated range, lowest gripper point, grasp-point box); you
   type `MOVE`; goal = present position, torque on, cosine move; the path waits while the arm lags > 0.25 rad and
   stops after 2 s of lag.
4. You place the cube and the plate and type `RUN`: the student loop, up to `--time-limit` s (sim episodes are 8 s).
5. End (time, key, guard): HOLD (arm goal = measured q, gripper = its last command), then the menu.

| Per control step (20 ms; a student query every 5 steps) | Value |
|---|---|
| Images | newest frame of each camera, crop of the checkpoint (`cfg.crops`), resize to 224 (INTER_AREA), RGB |
| Joints | LeRobot units <-> sim rad by `calib_v1.json` `joint_map`; qdot from the last 3 reads |
| Executed target | 1 of the 5 first chunk targets: change vs the previous command <= 0.06 rad (roll 0.08, jaw 0.5); action layer (lead band to the measured q, sim joint range, floor: lowest gripper point >= 1.5 mm + `--table-z`, or 1.5 mm above the mat plane in floor mode plane); eval: + the servo-error integral (<= 3 deg, anti-windup at the range); calibrated range - 3 deg; <= 10 deg (gripper 60) from the measured q |
| Stops (live) | command not finite, below the floor (-2 mm + table_z or the plane, and below the measured pose; eval: the deploy.json value) or outside the box; measured q > 5 deg outside the allowed range (calibrated - 3 deg); measured lowest gripper point < -5 mm + table_z or the plane (eval: deploy.json); grasp point outside x -0.10..0.45, y +-0.42, z -0.01..0.40 m or > 0.45 m from the pan axis; camera frame > 150 ms old (camera loss); an arm joint >= 90 % of its lead band for 0.5 s (blocked); read / write error; time limit; q / space / Esc / Ctrl+C |

## 5. What to record

- The `--out` folders (dry, live, replay): `episode_*.npz` / `.json`, `first_frames.npz`, `replay_*`.
- A phone video from the side, started before `RUN`.
- A photo of the scene and the cube / plate positions (x, y from the base, yaw) before `RUN`.
- Per run: end reason, reached the cube (yes / no), grasp height vs the cube, lift, place, manual stops.
- The timing lines at the end of the dry run (`latency ms_model`, `ms_step`, overruns).

## Not tested / open

- Nothing ran against the real bus or cameras (`LeRobotIO` was checked only against LeRobot's source).
- Timing: on luffy's busy shared GPU the model took 25 ms p50 / 43 ms p95, so every query step overran its 20 ms slot
  (46.7 Hz instead of 50). The Orin's time is unknown (dry run / replay a).
- Blank or wrong camera images are not detected (only the frame age). On the fake robot (constant grey images) the
  student moved the arm 87 deg (shoulder lift) and 127 deg (wrist roll) in 15 s with no stop.
- The jaw may open at the measured arm pose even if the moving jaw then goes lower; the command-floor stop (-2 mm)
  can end a good release. If so: `--guard '{"cmd_floor_m": -0.005}'`.
- Servo dynamics (STS3215, LeRobot P 16 / D 32) were not compared with the sim actuators. Replay images are H.264
  re-encodes, live images MJPG decodes. No undistortion (the student was trained on renders with the calibrated fisheye models).
