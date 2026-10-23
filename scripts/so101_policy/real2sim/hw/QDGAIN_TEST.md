# Robot test: joint-speed input gain (`--qd-gain`)

Written 2026-10-22 evening for an agent that guides the user at the SO101 robot (proxy, Jetson AGX Orin). The user runs
every command at the robot. The agent explains, reads the outputs and the logs, and keeps the rules below.

## 1. Why this test

| Item | Value |
|---|---|
| Robot result on 2026-10-22 (today's student, `eval`) | Grasp and lift 12 of 12; release 2 of 12. The arm holds the cube still 1-5 cm above the plate and never opens the gripper. |
| Cause (found in the sim) | The student sets its step size from the MEASURED joint speed. The real arm is about 5 times slower than the sim arm. So near the plate the steps get smaller and smaller, and the arm stops. |
| Sim proof (today's student, robot-like sim, 48 episodes) | Joint-speed input x 0 / 1 / 2 / 3 / 4 / 5: success 10 / 50 / 88 / 94 / 83 / 48 %. At 5 the arm hits the table. |
| Same, normal-speed sim | x 1 / 2 / 3: 98 / 90 / 75 %. A high gain makes a FAST arm too aggressive. The real arm is slow, so 2-3 should fit. |
| What `--qd-gain g` does | Multiplies only the student's joint-speed input by g. The safety guard and the logs keep the measured speed. All other settings are the same as `eval`. Default 1 = today's behaviour. |

The gripper decision itself does not change: the student must first get low enough. If the gain works, the arm keeps
moving down to the plate and the student releases as in the sim.

## 2. Rules (must)

1. The user runs all commands. The agent does not run `dry`, `live` or `eval`, and does not open the robot port or
   the cameras.
2. Never write motor EEPROM, never recalibrate, never run LeRobot tools (`lerobot-calibrate`, `lerobot-setup-motors`,
   `lerobot-teleoperate` ...) in this test.
3. Keep a hand near the arm. Stop keys: space, q, Esc or Ctrl+C. After a stop the arm holds (torque on).
4. Stop the test and lower the gain if the arm hits the table, presses the cube into the table, or oscillates.
5. Do not change files in `~/so101_hw_robot` (the launcher refuses changed files).

## 3. Before the first episode

1. `cd ~/so101_hw_robot && ./robot_test.sh check` must end with `CHECK OK`. The first line under "bundle files"
   must name git commit `see section 7` or later.
2. Start from the folded rest pose. In the `q real` line of the run output: the 2nd value (shoulder lift) must be
   between -88 and -100, the 3rd value (elbow) between 80 and 90. Else the run refuses (move the arm by hand with the
   torque off, then start again).
3. The wrist camera cable must have slack for a wrist roll of about 120 deg (HOME turns the wrist; the cable pulled
   out once on 10-22).
4. The cameras must stay in their USB ports (external = port 4.1.4, wrist = port 4.1.1). The script uses these paths.

## 4. Test plan

Use the layout that worked on 10-22 for most episodes (from behind the robot, looking where the arm points at HOME):
green cube 16 cm forward of the shoulder axis and 7 cm to the right; red mat 19 cm forward and 7 cm to the left; both
square to the forward direction. Also use the mirrored layout (cube left, mat right) and 2 other layouts inside
15-27 cm from the shoulder axis.

| Block | Command | Episodes | Layouts |
|---|---|---|---|
| A | `./robot_test.sh eval 4 --qd-gain 2` | 4 | 2 x working layout, 2 x mirrored |
| B | `./robot_test.sh eval 4 --qd-gain 2.5` | 4 | same 4 layouts |
| C | `./robot_test.sh eval 4 --qd-gain 3` | 4 | same 4 layouts, only if A and B had no table contact |
| D (optional) | `./robot_test.sh eval 4` (gain 1, reference) | 4 | same 4 layouts |

Episode limit: 40 s (the default of `eval`). A typical successful episode on 10-22 took about 22 s.

Each episode:
1. At HOME, place the cube and the mat, then type `RUN`.
2. Watch the arm. Push space when the cube is on the mat and the arm has moved away, or at a failure.
3. Type the result: `g` grasp, `l` lift, `p` placed on the mat, for example `glp`, or `-`. Add a short note after a
   space, for example `glp clean`, `gl hover 2cm`, `gl dropped high`, `g- slipped`, `table contact`.
4. Type `r`: the arm goes back to HOME (the gripper opens). Reset the scene.

At the end of a block: type `s`, then in the menu `r` (return to the start pose, slow), then `t` with a hand under the
arm (torque off), or `h` to keep holding.

## 5. What to watch for

| Sign | Meaning | Action |
|---|---|---|
| The arm goes down onto the mat and opens | The fix works | Continue |
| Hover 1-5 cm above the mat, no release (as on 10-22) | Gain too low | Next block (higher gain) |
| The gripper touches the table during the approach or the place | Gain too high | Stop; go back to the lower gain |
| Fast jerky motion or oscillation | Gain too high | Stop (space); lower the gain |
| "guard: command floor" or "measured floor" stop | The guard stopped a low command | Note it; if it repeats, lower the gain |
| Camera error / "frame ... old" stop | Cable or USB problem | Check the wrist cable; `ls /dev/v4l/by-path/` must show 4.1.1 and 4.1.4 |

## 6. What to send back

- The session folders: `~/so101_hw_runs/eval_<date>_<time>/` (each has `session.json` with the typed results and
  `ep_NN/episode_live.npz/json`).
- For each block: the gain, the typed results, and anything unusual.
- A phone video of 1-2 episodes per block helps.

## 7. Version

Bundle built from git commit noted in `~/so101_hw_robot/MANIFEST.txt` (line 1), which must include `--qd-gain`
(`./robot_test.sh eval 1 --qd-gain 2` prints the line `qdot input gain 2.0` after the joint map lines). Earlier bundles
reject `--qd-gain` with "unrecognized arguments".

## 8. Known issues from 10-22

| Issue | What to do |
|---|---|
| `preflight failed: measured q outside the calibrated range` | The rest pose is off. Use the ranges in section 3. |
| `HOME path refused` | Same. |
| The program ends and the arm stays stiff (torque on) | Run the program again and use the menu `t`, or: `conda activate lerobot` then `python -c "from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig as C; r=SOFollower(C(port='/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E115318-if00', id='so101_follower')); r.bus.connect(); r.bus.disconnect(disable_torque=True)"` with a hand under the arm. |
| `run` typed in lowercase | Refused: type `RUN`. The menu then appears. |
| Do not push `r` repeatedly after a stop near the table | Each hold lets the arm sag a little. Use `t` with a hand under the arm. |
