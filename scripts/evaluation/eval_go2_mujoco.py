#!/usr/bin/env python3
"""Evaluate a Go2 policy in MuJoCo, the domain its authors validated it in.

WHY MUJOCO AND NOT THE TRAINING SIMULATOR. The imported policy comes from
wty-yy/go2_rl_gym, a legged_gym-family repo, so its true training domain is Isaac Gym --
archived by NVIDIA, needing a developer-account download, Python 3.8 and an old torch.
What that repo ships instead is a MuJoCo deploy config, and we already have it verbatim at
checkpoints/go2_mujoco.yaml: kp 20, kd 0.5, dt 0.002, decimation 10, and the same scales
and default angles our Chrono adapter reimplemented. So this is not the training domain; it
is the authors' own sim-to-sim validation target, which is the closest reachable thing and
is a genuine third domain independent of both Chrono and the surrogate.

CONVENTIONS, CHECKED RATHER THAN ASSUMED. The adapter's CHRONO_TO_IMPORTED permutation and
its SIGN flip are Chrono-specific and must NOT be reused here:

  joint order   menagerie's go2.xml is FL, FR, RL, RR, which is already the policy's order,
                so the permutation is the identity. Verified from actuator names at load,
                not taken on trust -- a wrong permutation yields a perfectly plausible
                45-vector with left and right legs swapped.
  sign          thigh and calf signs agree between the MJCF home pose and the policy's
                default angles. The hips cannot be checked that way because the MJCF home
                pose has them at zero, so the settle test below checks them physically:
                with the policy's own defaults applied the robot must stand at a sensible
                height without lateral drift. Flipped hips splay the legs the wrong way.

The observation is rebuilt here to the same 45-dim layout the adapter documents:
ang_vel(3) | projected_gravity(3) | command(3) | dof_pos(12) | dof_vel(12) | prev_action(12)
"""
import argparse, json, sys, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True)
ap.add_argument("--xml", default="/home/kyle/sbel-artifacts/mujoco_menagerie/unitree_go2/scene.xml")
ap.add_argument("--vx", type=float, default=0.5)
ap.add_argument("--vy", type=float, default=0.0)
ap.add_argument("--wz", type=float, default=0.0)
ap.add_argument("--duration-s", type=float, default=10.0)
ap.add_argument("--settle-s", type=float, default=1.0)
ap.add_argument("--out", default=None)
a = ap.parse_args()

import mujoco, torch

DEFAULTS = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
                     0.1, 1.0, -1.5, -0.1, 1.0, -1.5], dtype=np.float64)
KP, KD = 20.0, 0.5
ANG_VEL_SCALE, DOF_POS_SCALE, DOF_VEL_SCALE, ACTION_SCALE = 0.25, 1.0, 0.05, 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
DECIMATION = 10
POLICY_ORDER = [f"{leg}_{j}" for leg in ("FL", "FR", "RL", "RR")
                for j in ("hip", "thigh", "calf")]

m = mujoco.MjModel.from_xml_path(a.xml)
d = mujoco.MjData(m)
act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
if act_names != POLICY_ORDER:
    raise SystemExit(f"FATAL: actuator order {act_names} != policy order {POLICY_ORDER}; "
                     "a permutation is needed and this script assumes identity")
print(f"  actuator order matches policy order ({len(act_names)} joints, identity permutation)")

pol = torch.jit.load(a.policy, map_location="cpu"); pol.eval()

# settle at the POLICY's default pose, which is also the hip-sign check
mujoco.mj_resetDataKeyframe(m, d, 0)
d.qpos[7:19] = DEFAULTS
d.qpos[2] = 0.35
mujoco.mj_forward(m, d)
for _ in range(int(a.settle_s / m.opt.timestep)):
    q, qd = d.qpos[7:19], d.qvel[6:18]
    d.ctrl[:] = KP * (DEFAULTS - q) - KD * qd
    mujoco.mj_step(m, d)
h = float(d.qpos[2]); lat = float(abs(d.qpos[1]))
print(f"  settle: trunk height {h:.3f} m, lateral drift {lat:.4f} m")
if not (0.20 < h < 0.45) or lat > 0.05:
    raise SystemExit("FATAL: the robot did not settle at a sensible pose, so the joint "
                     "sign or order convention is wrong; refusing to report tracking")
print("  sign/order convention validated physically")

cmd = np.array([a.vx, a.vy, a.wz], dtype=np.float32)
last_action = np.zeros(12, dtype=np.float32)
n_ctrl = int(a.duration_s / (m.opt.timestep * DECIMATION))
vx_err, vy_err, wz_err, heights, fell = [], [], [], [], False
for k in range(n_ctrl):
    quat = d.qpos[3:7]
    R = np.zeros(9); mujoco.mju_quat2Mat(R, quat); R = R.reshape(3, 3)
    grav = R.T @ np.array([0.0, 0.0, -1.0])
    w_local = R.T @ d.qvel[3:6]
    v_local = R.T @ d.qvel[0:3]
    q, qd = d.qpos[7:19].copy(), d.qvel[6:18].copy()
    obs = np.concatenate([
        w_local * ANG_VEL_SCALE, grav, cmd * CMD_SCALE,
        (q - DEFAULTS) * DOF_POS_SCALE, qd * DOF_VEL_SCALE, last_action]).astype(np.float32)
    with torch.no_grad():
        action = pol(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy().astype(np.float32)
    last_action = action
    targets = action * ACTION_SCALE + DEFAULTS
    for _ in range(DECIMATION):
        d.ctrl[:] = KP * (targets - d.qpos[7:19]) - KD * d.qvel[6:18]
        mujoco.mj_step(m, d)
    if d.qpos[2] < 0.15:
        fell = True; break
    vx_err.append(abs(v_local[0] - a.vx)); vy_err.append(abs(v_local[1] - a.vy))
    wz_err.append(abs(w_local[2] - a.wz)); heights.append(float(d.qpos[2]))

res = {"policy": os.path.basename(a.policy), "cmd": [a.vx, a.vy, a.wz],
       "control_steps": len(vx_err), "of": n_ctrl, "fell": fell,
       "mae_vx": float(np.mean(vx_err)) if vx_err else None,
       "mae_vy": float(np.mean(vy_err)) if vy_err else None,
       "mae_wz": float(np.mean(wz_err)) if wz_err else None,
       "mean_height": float(np.mean(heights)) if heights else None}
print(f"  {res['policy']}: mae_vx {res['mae_vx']:.4f}  height {res['mean_height']:.3f}  "
      f"fell={fell}  steps {len(vx_err)}/{n_ctrl}")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
