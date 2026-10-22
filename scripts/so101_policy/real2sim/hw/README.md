# SO101 camera student on the real robot (draft)

Status 2026-10-21: **draft, never run on the robot.** Only the `replay` mode (2 real recordings) and the robot loop
on a fake servo (`--backend fake`, unit tests) were run, on luffy. Do the dry run first, then live.

| File | What |
|---|---|
| `run_student_real.py` | modes `replay`, `dry`, `live` |
| `hw_lib.py` | joint map, LeRobot-unit limits, image crop / resize, qdot estimate, action layer, hardware guard, joint paths, student wrapper, one control step (`Controller`), fake robot |
| `make_orin_bundle.sh` | builds the folder for the Orin (sources + collision points + student + calib) |
| `tests/so101_policy/test_real2sim_hw.py` | 21 unit tests (CPU, about 10 s) |

## What the script does in one control step (all modes)

| Item | Value |
|---|---|
| Control step | 20 ms (50 Hz). A student query every 5 steps (0.1 s). |
| Query inputs | the 2 camera images (newest frame), measured q and a qdot estimate (sim units) |
| Images | LeRobot camera frame 640 x 480 RGB (replay: the mp4 decoded BGR, then BGR -> RGB), the crop box of `calib_v1.json` `"student_view"` (else a centred 480 x 480 square), resize to the student's size (224) with `cv2.INTER_AREA`. No undistortion. |
| Joints | real (LeRobot deg, gripper 0-100) -> sim rad by `calib_v1.json` `joint_map`; targets back the same way |
| qdot | least-squares slope over the last 3 measured q (40 ms, lag about 20 ms) |
| Output | a chunk of 10 absolute targets; the first 5 are executed, one per step (jaw lead = checkpoint's, 0) |
| Each executed target | 1. change vs the previous command <= 0.06 rad (arm, roll 0.08) / 0.5 rad (jaw) per step; 2. the sim action layer (`joint_env.safety`: lead band to the measured q, joint range, table floor 1.5 mm from the kinematics); 3. LeRobot calibrated range minus 3 deg; 4. max 10 deg (gripper 60) from the measured q; 5. stop if the command is not finite, leaves the workspace box or the floor |
| Stop reasons (live) | measured q outside the calibrated range (+5 deg), measured gripper point 5 mm below the table, grasp point outside the box x -0.10..0.45, y +-0.42, z -0.01..0.40 m or > 0.45 m from the pan axis, a camera frame older than 150 ms, an arm joint held at >= 90 % of its lead band for 0.5 s (blocked), joint read / write error, time limit (15 s), keys q / space / Esc, Ctrl+C |
| After a stop | HOLD: arm goal = measured q, gripper goal = its last command (keeps the grip); torque stays on |

## Put it on the Orin (you do this; the script never copies anything to the robot machine)

On luffy:

```bash
cd /home/harry/.t3/worktrees/NeDM-push-ppo/t3-dded53d1
scripts/so101_policy/real2sim/hw/make_orin_bundle.sh <student.pt>        # -> ~/so101_r2s_claude/hw_bundle/so101_hw.tar.gz
scp ~/so101_r2s_claude/hw_bundle/so101_hw.tar.gz sbel@proxy:~/
```

On the Orin: `cd ~ && tar xzf so101_hw.tar.gz && cd so101_hw`. The bundle needs only the `lerobot` conda env
(torch, torchvision, numpy, cv2, pyarrow, LeRobot 0.6.2). No mujoco: the table floor uses `assets/kin_points_v1.npz`.

## Dry run (robot connected, no motor writes, torque untouched)

Stop any other LeRobot process first (it owns the serial port and the cameras). Run in an interactive terminal
(desktop terminal or `ssh -t`): the stop keys q / space / Esc need a TTY; without one only Ctrl+C stops.

```bash
conda activate lerobot
cd ~/so101_hw
python scripts/so101_policy/real2sim/hw/run_student_real.py dry \
    --student assets/student.pt --calib assets/calib_v1.json --out ~/so101_hw_runs/dry_1 --time-limit 30
```

- Follower port `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E115318-if00` (`--port`), cameras `wrist` =
  `/dev/v4l/by-path/platform-3610000.usb-usb-0:4.1.1:1.0-video-index0`, `external` =
  `/dev/v4l/by-id/usb-Xitech_USB_Camera_20250606105-video-index0` (`--cam-paths '{"wrist": ..., "external": ...}'`),
  640 x 480, 30 fps, MJPG, V4L2 backend (as the recordings).
- Dry mode opens the bus with LeRobot's handshake (it reads model numbers and firmware), checks that the motors'
  calibration registers equal `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json`, and
  never calls `robot.connect()` / `configure()` (they turn the torque off and on and write registers). Every writing
  method of the bus and robot is replaced by one that raises.
- With the torque off you can move the arm by hand and watch the printed status; the log has the student's chunk,
  executed targets, plate / cube predictions and image thumbnails (`episode_dry.npz`, `episode_dry.json`).
- **Use the dry run to measure the timing on the Orin**: the end of the run prints p50 / p95 / max of the joint read,
  the model and the whole step. A query step does the inference inside its 20 ms slot; on luffy (shared, busy GPU) the
  model took 13-21 ms (p50) and a query step 28 ms, so those steps overran.

## Live run (moves the robot)

Preconditions: a real `calib_v1.json` joint map (the script refuses the placeholder identity map), the dry run
looked right, the workspace is clear, you stand at the robot with a hand near the arm.

```bash
python scripts/so101_policy/real2sim/hw/run_student_real.py live --i-am-supervising \
    --student assets/student.pt --calib assets/calib_v1.json --out ~/so101_hw_runs/live_1 --time-limit 15
```

Sequence and checks:

1. Before any write: LeRobot calibration file present and equal to the motors' registers; 3 joint reads finite and
   inside the calibrated range; both cameras deliver 640 x 480 frames younger than 200 ms.
2. `--configure auto` (default): reads the registers that LeRobot's `configure` writes (P / I / D gains,
   acceleration, return delay, position mode, gripper torque / current limits). Only if one differs it asks you to type
   `CONFIGURE` (the arm must rest on its stops: configure turns the torque off for a moment), writes goal = present
   position, then runs LeRobot's `configure`. `--configure never` skips it; `always` forces it.
3. HOME: default sim pose (0, -5, -15, 92, 0, 30) deg (median start of the sim tasks; `--home`). The joint path from
   the present pose is checked: inside the sim joint range and the calibrated range, lowest gripper point not below
   min(2 cm, either end) - 1 mm, grasp point inside the box. You type `MOVE`; goal = present position, torque on, then
   the move with a cosine speed profile, peak 15 deg/s per joint (`--home-speed-deg-s`); the path waits while the arm
   lags > 0.25 rad and stops after 2 s of lag.
4. You place the cube and the plate and type `RUN`: the student loop, up to `--time-limit` s.
5. End (time, key, guard reason): HOLD, then a menu: `h` hold and exit (torque stays ON, the servos hold the goal),
   `r` slow return to the start pose (checked path, `MOVE`), `t` torque OFF (type `TORQUE OFF`; support the arm).

The robot class is used below LeRobot's `Robot` API: `SOFollowerRobotConfig(disable_torque_on_disconnect=False,
max_relative_target=None)`; the script applies its own max-relative-target clamp against the joint read of the same
step (LeRobot's option would cost a second bus read per step).

## Replay (offline, here)

```bash
~/miniconda3/envs/so101_vision/bin/python scripts/so101_policy/real2sim/hw/run_student_real.py replay \
    --student ~/so101_policy/runs/viral/final_student_o17_it03000.pt \
    --episodes green_cube_red_plate_1:0,green_cube_red_plate:2 --out ~/so101_r2s_claude/hw/replay_test
```

Per episode `<dataset>_ep<k>.npz` / `.json` (per step q, qdot, target, command, which rule acted, timings; per query
chunk, aux predictions, thumbnails; recorded leader action) and `replay_summary.json`. If `calib_v1.json` has
`episodes` entries, the summary compares the predicted cube position (before the grasp frame) and plate pose with them.

## Not tested / open

- Nothing here was run against the real bus or cameras: `LeRobotIO` (connect, register reads, `configure`, torque,
  goal writes, `read_latest` ages) was checked only by reading LeRobot 0.6.2's source on the Orin.
- Bus timing at 50 Hz (one sync read + one sync write per step) and the Orin's inference time are unknown.
- The placeholder identity joint map is wrong: in replay, the recorded grasps map to a grasp point up to 27 mm below
  the table, and the recorded wrist_flex (up to 101.7 deg) exceeds the sim range (92.1 deg).
- The recordings start from the folded rest pose (LeRobot about [0, -88, 84, 58-76, 0]); the sim tasks start near
  HOME. The student never saw a folded start; the HOME move bridges it.
- The sim action layer lets the jaw move at the measured arm pose even if the moving jaw then goes lower; the guard
  stops if a command puts the lowest gripper point 2 mm below the table and below the measured pose (for example a
  release with the jaw tip on the table). If that stops good releases, loosen it: `--guard '{"cmd_floor_m": -0.005}'`.
- Servo dynamics: the real STS3215 runs LeRobot's P 16 / D 32 position loop, the sim the MJCF actuators; not compared.
- Image path differences replay vs live: the recordings are H.264 (CRF 23) re-encodes of RGB frames; live frames are
  MJPG decodes. The fisheye distortion is not removed (the sim cameras must model it, or the student must be trained
  on undistorted views and `preprocess` extended).
