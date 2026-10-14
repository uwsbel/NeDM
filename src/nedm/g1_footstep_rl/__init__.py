"""A Unitree G1 footstep policy trained inside the G1 contact NRD (a learned simulator) and checked in CPU MuJoCo.

This file imports only pathlib, so each module can be imported on its own: the torch modules need torch (and
rsl_rl for the environments), the MuJoCo modules need mujoco and onnxruntime.

| Module | Needs | Content |
|---|---|---|
| ``homie_policy`` | numpy, onnxruntime | The frozen HOMIE walking teacher (OpenHomie deploy.onnx): constants, observation frame, batched onnxruntime teacher |
| ``homie`` | torch | The same teacher as a torch module, its observation frame, and a batched teacher with its 6-frame history |
| ``fk`` | torch | Forward kinematics of the G1 legs: foot spheres, ankles and the swept leg distances of the self-contact rule |
| ``sim`` | torch | The batched NRD simulator: re-centred x, y, dense contact evaluation, several checkpoints (members) |
| ``bank`` | torch | Start states from recorded episodes: the NRD history and the teacher history at a control step |
| ``npz_rows`` | numpy, torch | Memory-mapped reads of chosen rows of an uncompressed unified_data.npz |
| ``planks_task`` | numpy | Plank layouts, the support rule and the plank metrics (shared by both worlds) |
| ``planks_env`` | torch | Shared parts of the NRD environment: config merge, start bank, HOMIE preview, path frame helpers |
| ``footsteps_task`` | numpy | The footstep-target task: courses, course banks, the numpy reference scorer |
| ``footsteps_env`` | torch, rsl_rl | The footstep task inside the NRD as a batched RSL-RL VecEnv |
| ``ppo_utils`` | torch | PPO additions for the trainer: split optimizer, std floor, time-out bootstrap, learning curve |
| ``g1_scene`` | numpy, mujoco | The MuJoCo scene of the G1 data collector (model, contact classes, record interval, state helpers) |
| ``mujoco_env`` | numpy, mujoco | The MuJoCo ground truth: the collector's scene with optional plank boxes, the HOMIE command policy |
| ``mujoco_policy`` | torch | Run files, action mapping and statistics shared by the MuJoCo evaluation |
| ``footsteps_mujoco`` | torch, rsl_rl, mujoco | The footstep task in MuJoCo: episode loop, policy runner, H-FS baseline, the same episodes inside the NRD, statistics, rendering |

State convention (67 channels, float64): pelvis x, y, z (world); quaternion wxyz (pelvis to world); 27 joint
angles; pelvis velocity (world); pelvis angular velocity (world); 27 joint rates. The rates are 20 ms step averages
(nedm.contact_nrd.data.step_average_rates). Action: 27 absolute joint position targets held for 20 ms, in the joint
order of nedm.g1_footstep_rl.homie_policy.JOINT_NAMES.

Comments that cite "spec" sections or decision ids (for example M1-M7, D15b) refer to the internal design notes of
the study, which are not released; the code and g1_footstep_rl/README.md define the task.

Data paths default to <repository>/artifacts/ (REPO_ROOT below); scripts/g1_footstep_rl/reproduce.sh puts the
released files there.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
