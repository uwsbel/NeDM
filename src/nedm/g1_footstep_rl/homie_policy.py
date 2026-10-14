"""The frozen HOMIE G1 walking policy (OpenHomie HomieDeploy/deploy.onnx) as a batched numpy/onnxruntime teacher.

Contract of deploy.onnx (input "input" [B, 456] float32 -> output "output" [B, 12] float32), checked against the
real-robot deploy code (HomieDeploy/g1_gym_deploy: lcm_agent.py, history_wrapper.py, deploy_policy.py,
cheetah_state_estimator.py), the training environment (HomieRL legged_gym legged_robot.py, g1_29dof_config.py,
legged_robot_config.py, helpers.PolicyExporterHIM) and the ONNX graph itself:

- Input = six 76-value frames, oldest first, newest last (the graph takes input[:, -76:] as the current frame and
  the whole 456 values for its velocity/latent estimator; no normalisation inside the graph). After a reset the
  history is all zeros (HistoryWrapper.reset); each control step appends the new frame.
- Frame: [vx, vy, wz commands x (2, 2, 0.5)] (3), height command in metres, unscaled (1; 0.74 = full height,
  trained range 0.24-0.74 and only with zero velocity commands below 0.74), pelvis angular velocity in the pelvis
  frame x 0.5 (3), gravity direction in the pelvis frame (3), joint angles minus the default angles (27), joint
  rates x 0.05 (27), the previous policy output after clipping (12). The IMU is imu_in_pelvis (fixed to the pelvis,
  no rotation). Training scales the yaw command by obs_scales.ang_vel = 0.5; the deploy code writes 0.25. In
  CPU MuJoCo (10 s runs, mean after 3 s; README yaw-scale table) 0.5 tracks (0.6 rad/s -> 0.57, 0.8 -> 0.73),
  0.25 does not (0.6 rad/s -> 0.00, inside the dead zone; 0.8 -> 0.41).
  The MuJoCo example (MujocoDeploy) copies unitree_rl_gym scales (angular velocity 0.25) and a height 0.34: both
  are wrong for this policy.
- Output: 12 leg actions, clipped to +-100; leg target = default + 0.25 x action. The 15 upper-body targets come
  from outside the policy (zero = default).
- Joint order (27, the MJCF and Isaac Gym order): left leg (hip pitch, roll, yaw, knee, ankle pitch, roll), right
  leg, waist yaw, left arm (shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw), right arm. Waist roll and
  pitch and the hand joints are fixed in this asset.
- Joint PD of training: legs kp (100, 100, 100, 150, 40, 40), kd (2, 2, 2, 4, 2, 2), explicit torques clipped to the
  URDF efforts; upper body Isaac Gym position drives kp (waist 300, shoulder 200, elbow 100, wrist 20), kd (5, 4,
  1, 0.5). Control period 20 ms (training: 5 ms physics x 4).
numpy and onnxruntime only (no torch).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")
N_JOINTS, N_LEG = 27, 12
DEFAULT_ANGLES = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2 + [0.0] * 15)
_ARM_KP, _ARM_KD = [200.0, 200.0, 200.0, 100.0, 20.0, 20.0, 20.0], [4.0, 4.0, 4.0, 1.0, 0.5, 0.5, 0.5]
KP = np.array([100.0, 100.0, 100.0, 150.0, 40.0, 40.0] * 2 + [300.0] + _ARM_KP * 2)
KD = np.array([2.0, 2.0, 2.0, 4.0, 2.0, 2.0] * 2 + [5.0] + _ARM_KD * 2)
TORQUE_LIMIT = np.array([88.0, 139.0, 88.0, 139.0, 50.0, 50.0] * 2 + [88.0] + [25.0] * 5 + [5.0] * 2
                        + [25.0] * 5 + [5.0] * 2)   # URDF efforts (training clips to these)
COMMAND_SCALE = np.array([2.0, 2.0, 0.5])
ANG_VEL_SCALE, DOF_VEL_SCALE, ACTION_SCALE, ACTION_CLIP = 0.5, 0.05, 0.25, 100.0
HISTORY, FRAME = 6, 76
FULL_HEIGHT = 0.74
COMMAND_RANGES = {"vx": (-0.8, 1.2), "vy": (-0.5, 0.5), "wz": (-0.8, 0.8), "height": (0.24, 0.74)}   # training


def projected_gravity(quat):
    """[B, 4] body quaternions (w, x, y, z) -> [B, 3] unit gravity direction in the body frame."""
    w, x, y, z = np.moveaxis(np.asarray(quat, np.float64), -1, 0)
    return -np.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)


def frame(command, quat, ang_vel_body, q, dq, last_action):
    """One observation frame [B, 76] float32. command [B, 4] = (vx, vy, wz, height) in the heading frame (m/s,
    rad/s, m); quat/ang_vel_body of the pelvis; q, dq [B, 27]; last_action [B, 12] (clipped policy output)."""
    command = np.asarray(command, np.float64)
    return np.concatenate((command[:, :3] * COMMAND_SCALE, command[:, 3:4], ang_vel_body * ANG_VEL_SCALE,
                           projected_gravity(quat), q - DEFAULT_ANGLES, dq * DOF_VEL_SCALE, last_action),
                          -1).astype(np.float32)


class HomiePolicy:
    """deploy.onnx with its observation history for a batch of environments. One onnxruntime call per control
    step for the whole batch."""

    def __init__(self, onnx_path, batch, sha256=None, threads=1):
        import onnxruntime as ort
        path = Path(onnx_path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha256 and digest != sha256:
            raise ValueError(f"{path}: sha256 {digest}, expected {sha256}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.sha256 = digest
        self.history = np.zeros((batch, HISTORY, FRAME), np.float32)
        self.last_action = np.zeros((batch, N_LEG), np.float32)

    def reset(self, ids):
        """Zero history and previous action (the deploy reset)."""
        self.history[ids] = 0.0
        self.last_action[ids] = 0.0

    def step(self, ids, command, quat, ang_vel_body, q, dq):
        """Append the frame of the current state to the histories of ids and run the policy on them.
        Returns (raw output [n, 12], clipped output [n, 12], leg targets [n, 12] in rad)."""
        f = frame(command, quat, ang_vel_body, q, dq, self.last_action[ids])
        self.history[ids] = np.concatenate((self.history[ids, 1:], f[:, None]), 1)
        raw = self.session.run(None, {"input": self.history[ids].reshape(len(ids), -1)})[0]
        act = np.clip(raw, -ACTION_CLIP, ACTION_CLIP)
        self.last_action[ids] = act
        return raw, act, DEFAULT_ANGLES[:N_LEG] + ACTION_SCALE * act
