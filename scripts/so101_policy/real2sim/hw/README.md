# SO101 camera student on the real robot

Status 2026-10-22: **never run on the robot.** Student `final_v2_s26_it03000.pt` (cameras `external` + `wrist`,
224 px, crops pinned in the checkpoint: external [52, 103, 288, 288], wrist [0, 0, 640, 480]). Checks done on luffy
only (logs in `~/so101_r2s_claude/hw_final/`): replay of 3 recordings from the extracted bundle equals the worktree run
exactly; dry and live loops on the fake robot; the HOME check from the 15 recorded rest poses.

| File | What |
|---|---|
| `run_student_real.py` | modes `replay` (recording, no robot), `dry` (robot read only), `live` (moves the robot) |
| `hw_lib.py` | joint map, limits, image crop, action layer, hardware guard, joint paths, student wrapper, fake robot |
| `make_orin_bundle.sh` | builds the folder + `.tar.gz` for the Orin (sources, collision points, student, calib_v1.json) |
| `tests/so101_policy/test_real2sim_hw.py` | 21 unit tests (CPU, about 10 s) |

## Known problem: closed gripper at the start

If the gripper reads < 3 (LeRobot 0-100) at the start, the HOME check refuses the move ("path leaves the sim joint
range"): a closed real gripper is below the sim jaw limit. 9 of the 15 recordings start like that (gripper 0.6-1.4).
**Open the gripper by hand to 5 or more before the live run** (torque off; read the value in the dry run). The end-menu
return (`r`) to such a start pose is refused the same way.

## 1. Build (luffy) and copy (you)

```bash
cd /home/harry/.t3/worktrees/NeDM-push-ppo/t3-dded53d1
scripts/so101_policy/real2sim/hw/make_orin_bundle.sh ~/so101_r2s_claude/ckpt/final_v2_s26_it03000.pt \
    ~/so101_r2s_claude/hw_bundle/so101_hw_final
scp ~/so101_r2s_claude/hw_bundle/so101_hw_final.tar.gz sbel@proxy:~/          # you do this
```

On the Orin: `cd ~ && tar xzf so101_hw_final.tar.gz && cd so101_hw_final && sed -n '4,$p' MANIFEST.txt | sha256sum -c --quiet`
(no output = all files intact). Needs only the `lerobot` conda env (torch, numpy, cv2, pyarrow, LeRobot 0.6.2); no
mujoco. Run every command below from `~/so101_hw_final`, in an interactive terminal (desktop or `ssh -t`): the stop
keys q / space / Esc need a TTY, else only Ctrl+C stops. Do not pipe into `tee` (Ctrl+C also stops `tee`); the
`--out` folder keeps the log.

## 2. Commands on the Orin

```bash
conda activate lerobot
cd ~/so101_hw_final
S="--student assets/student.pt --calib assets/calib_v1.json"
# a) replay (no robot; Orin inference time; compare with luffy)
python scripts/so101_policy/real2sim/hw/run_student_real.py replay $S --data-root /home/sbel/lerobot/data \
    --episodes green_cube_red_plate:0,green_cube_red_plate:2,green_cube_red_plate_1:5 --out ~/so101_hw_runs/replay_orin
# b) dry (robot and cameras connected, NO motor writes, torque untouched; stop other LeRobot processes first)
python scripts/so101_policy/real2sim/hw/run_student_real.py dry $S --out ~/so101_hw_runs/dry_1 --time-limit 30
# c) live (moves the robot)
python scripts/so101_policy/real2sim/hw/run_student_real.py live --i-am-supervising $S \
    --out ~/so101_hw_runs/live_1 --time-limit 15
```

Defaults used by these commands: HOME = sim (0, -5, -15, 92, 45, 30) deg = LeRobot (0.0, -3.2, -6.5, 93.1, 120.8,
gripper 31.1); HOME speed 15 deg/s peak; `--table-z 0`; `--configure auto`; guard = `GuardConfig` defaults (table
below). Luffy replay of a): cube error before the grasp 23.9 / 22.7 / 23.0 mm, plate 9.6 / 9.2 / 4.6 mm (bf16 on
another GPU: small differences are normal).

## 3. Pre-run checklist (live)

1. Dry run first. Check: the printed q matches the arm; `first_frames.npz` / the logged thumbnails show the fixed
   camera as `external` and the gripper camera as `wrist` (a swap is not detected); the end of the run prints the
   model and step time (p50 / p95). Gripper >= 5 (above).
2. Arm in the folded rest pose. The HOME move takes 11-13 s at <= 15 deg/s; the wrist roll turns about 120 deg.
3. Cube and plate: both 0.15-0.27 m from the shoulder pan axis (3.9 cm in front of the base origin), within +-75 deg
   of straight ahead, 6-30 cm apart, plate yaw within 30 deg of the cube yaw (mod 90). Place them after HOME, before
   you type `RUN`. (8 of the 15 recorded cubes were 0.28-0.34 m out: outside the training region.)
4. Table height: the real table is 4-17 mm below the sim table over that region (tilted about 2 deg; 10-18 mm at the
   15 recorded cubes). Keep `--table-z 0` for the first run: the gripper floor then stays 5 mm or more above the real
   table. Expect the grasp about 15 mm too high (grasp point near the cube top face, a pinch or a miss).
   `--table-z` is one flat height: a value below -0.005 lets the gripper reach the table at the high side.
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
| Executed target | 1 of the 5 first chunk targets: change vs the previous command <= 0.06 rad (roll 0.08, jaw 0.5); action layer (lead band to the measured q, sim joint range, floor: lowest gripper point >= 1.5 mm + `--table-z`); calibrated range - 3 deg; <= 10 deg (gripper 60) from the measured q |
| Stops (live) | command not finite, below the floor (-2 mm + table_z, and below the measured pose) or outside the box; measured q > 5 deg outside the allowed range (calibrated - 3 deg); measured lowest gripper point < -5 mm + table_z; grasp point outside x -0.10..0.45, y +-0.42, z -0.01..0.40 m or > 0.45 m from the pan axis; camera frame > 150 ms old (camera loss); an arm joint >= 90 % of its lead band for 0.5 s (blocked); read / write error; time limit; q / space / Esc / Ctrl+C |

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
