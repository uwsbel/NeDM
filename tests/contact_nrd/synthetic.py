"""Synthetic data directories in the two layouts of contact_nrd/collection_contract.md (not a test module).

The motion is smooth kinematics, not physics: poses are sums of sines, the rates their exact instantaneous
derivatives (free-body angular velocity in the world frame), contacts random on/off runs. Enough to check shapes,
conventions and that every training stage runs. ``write_dataset`` of the collection package writes the directory,
so the evaluation block of configs/contact_nrd/evaluation/<name>.json is embedded as the collectors' data will have.
"""
from __future__ import annotations

import numpy as np

from nedm.contact_nrd.collect import export

JOINTS_G1 = 27


def quat_mul(a, b):
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2), -1)


def quat_exp(rotvec):
    """Rotation vectors [..., 3] -> unit quaternions wxyz."""
    ang = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    axis = np.where(ang > 1e-12, rotvec / np.maximum(ang, 1e-12), 0.0)
    return np.concatenate((np.cos(ang / 2), np.sin(ang / 2) * axis), -1)


def euler_quat(yaw, pitch, roll):
    z, y, x = (np.zeros(yaw.shape + (3,)) for _ in range(3))
    z[..., 2], y[..., 1], x[..., 0] = yaw, pitch, roll
    return quat_mul(quat_mul(quat_exp(z), quat_exp(y)), quat_exp(x))


def world_rates(angles, t, h=1e-6):
    """Quaternion path q(t) = euler_quat(*angles(t)) -> q [T, 4] and world angular velocity [T, 3]."""
    q = euler_quat(*angles(t))
    q1, q0 = euler_quat(*angles(t + h)), euler_quat(*angles(t - h))
    d = quat_mul(q1, q0 * np.array([1.0, -1.0, -1.0, -1.0]))
    return q, 2 * d[:, 1:] / (2 * h) * np.sign(d[:, :1])


def sines(rng, t, n, amp, freq=(0.3, 2.0)):
    """n smooth signals [T, n] and their derivatives."""
    a = rng.uniform(0.3, 1.0, n) * amp
    f = rng.uniform(*freq, n) * 2 * np.pi
    p = rng.uniform(0, 2 * np.pi, n)
    return a * np.sin(f * t[:, None] + p), a * f * np.cos(f * t[:, None] + p)


def telegraph(rng, steps, on_s, off_s, dt, p_on=0.5):
    """Random on/off runs [steps] with mean durations on_s, off_s."""
    out, state, k = np.zeros(steps, np.uint8), rng.random() < p_on, 0
    while k < steps:
        run = max(1, int(rng.exponential((on_s if state else off_s) / dt)))
        out[k:k + run] = state
        k, state = k + run, not state
    return out


def held(x, per):
    """Hold every per-th record (zero-order hold of a command)."""
    return x[(np.arange(len(x)) // per) * per]


def g1_system(records):
    joints = [f"joint{k}" for k in range(JOINTS_G1)]
    types = (["position"] * 3 + ["quaternion"] * 4 + ["angle"] * JOINTS_G1 + ["velocity"] * 3 + ["angular_velocity"] * 3
             + ["angular_rate"] * JOINTS_G1)
    channels = (["x", "y", "z", "qw", "qx", "qy", "qz"] + [f"q_{j}" for j in joints] + ["vx", "vy", "vz", "wx", "wy", "wz"]
                + [f"qd_{j}" for j in joints])
    robot = dict(name="robot", kind="articulated", moving=True, state_dim=len(types), channels=channels, channel_types=types,
                 channel_mask=[1] * len(types), action_dim=JOINTS_G1, action_channels=[f"target_{j}" for j in joints],
                 joint_names=joints)
    return dict(name="g1_locomotion", record_step_s=0.01, model_step_s=0.02, control_step_s=0.02, target_time_s=2.0,
                records_per_episode=records, frame="world z up, ground at z = 0, SI, quaternions wxyz",
                bodies=[robot, dict(name="ground", kind="plane", moving=False, top_z_m=0.0)],
                pairs=[[0, 1], [0, 1]],
                pair_metadata=[dict(name="left_foot_ground", geoms=[["left_foot"], ["floor"]]),
                               dict(name="right_foot_ground", geoms=[["right_foot"], ["floor"]])],
                target_body=0, contact_rule="synthetic (no physics)",
                families=[dict(name="walk", group="nominal", select=True, note=""),
                          dict(name="perturbed", group="difficult", select=True, note=""),
                          dict(name="airborne", group="probe", select=False, note="")],
                terminal_reasons=["planned_end", "fall"], physics_config=dict(synthetic=True))


def g1_episode(rng, records, family):
    t = np.arange(records) * 0.01
    speed = rng.uniform(0.2, 0.6)
    pos = np.stack((speed * t, 0.05 * np.sin(1.3 * t), 0.75 + 0.002 * np.sin(9.0 * t)), -1)
    vel = np.stack((np.full_like(t, speed), 0.05 * 1.3 * np.cos(1.3 * t), 0.002 * 9.0 * np.cos(9.0 * t)), -1)
    yaw0, wy, ph = rng.uniform(-np.pi, np.pi), rng.uniform(-0.4, 0.4), rng.uniform(0, 6)
    q, w = world_rates(lambda s: (yaw0 + wy * s, 0.05 * np.sin(2 * s + ph), 0.04 * np.sin(3 * s)), t)
    q = export.quat_continuous(q)
    ang, rate = sines(rng, t, JOINTS_G1, 0.4)
    act = held(ang + 0.05 * np.sin(5 * t)[:, None], 2)
    state = np.concatenate((pos, q, ang, vel, w, rate), -1)
    if family == 2:
        contacts = np.zeros((records - 1, 2), np.uint8)
    else:
        phase = 2 * np.pi * 1.4 * t[:-1] + rng.uniform(0, 6)
        contacts = np.stack((np.sin(phase) > -0.3, np.sin(phase + np.pi) > -0.3), -1).astype(np.uint8)
    return dict(states={"robot": state}, action=act, contacts=contacts,
                per_interval=dict(contact_force=contacts * rng.uniform(50, 400, contacts.shape).astype(np.float32)))


def so101_system(records):
    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    arm = dict(name="arm", kind="articulated", moving=True, state_dim=12, channels=[f"q_{j}" for j in joints] + [f"qd_{j}" for j in joints],
               channel_types=["angle"] * 6 + ["angular_rate"] * 6, action_dim=6, action_channels=[f"target_{j}" for j in joints],
               joint_names=joints)
    cube_types = ["position"] * 3 + ["quaternion"] * 4 + ["velocity"] * 3 + ["angular_velocity"] * 3
    cube = dict(name="cube", kind="rigid", moving=True, state_dim=13, channels=["x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz"],
                channel_types=cube_types, channel_mask=[1] * 13)
    return dict(name="so101_grasp", record_step_s=0.01, model_step_s=0.02, control_step_s=0.02, target_time_s=2.0,
                records_per_episode=records, frame="world z up, table top at z = 0, SI, quaternions wxyz",
                bodies=[arm, cube, dict(name="table", kind="box", moving=False, top_z_m=0.0, size_m=[0.6, 0.6, 0.04])],
                pairs=[[0, 1], [0, 1], [1, 2]],
                pair_metadata=[dict(name="fixed_jaw_cube", geoms=[["fixed_jaw_pad"], ["cube"]]),
                               dict(name="moving_jaw_cube", geoms=[["moving_jaw_pad"], ["cube"]]),
                               dict(name="cube_table", geoms=[["cube"], ["table"]])],
                target_body=1, contact_rule="synthetic (no physics)",
                families=[dict(name="lift", group="nominal", select=True, note=""),
                          dict(name="slip", group="difficult", select=True, note=""),
                          dict(name="free_motion", group="probe", select=False, note="")],
                terminal_reasons=["planned_end", "unsupported_contact"], physics_config=dict(synthetic=True),
                phases=["idle", "approach", "descend", "close", "grip", "lift", "hold", "release"])


def so101_episode(rng, records, family):
    t = np.arange(records) * 0.01
    ang, rate = sines(rng, t, 6, 0.5)
    act = held(ang + 0.03 * np.sin(4 * t)[:, None], 2)
    lift = 0.03 * (1 - np.cos(np.clip(t - 1.0, 0, np.pi)))
    dlift = 0.03 * np.sin(np.clip(t - 1.0, 0, np.pi)) * ((t > 1.0) & (t < 1.0 + np.pi))
    x0, y0 = rng.uniform(0.15, 0.25), rng.uniform(-0.1, 0.1)
    pos = np.stack((x0 + 0.01 * np.sin(0.7 * t), y0 + 0.01 * np.sin(0.5 * t), 0.015 + lift), -1)
    vel = np.stack((0.007 * np.cos(0.7 * t), 0.005 * np.cos(0.5 * t), dlift), -1)
    yaw0 = rng.uniform(-np.pi, np.pi)
    q, w = world_rates(lambda s: (yaw0 + 0.3 * np.sin(s), 0.1 * np.sin(1.5 * s), 0.05 * np.sin(2.5 * s)), t)
    q = export.quat_continuous(q)
    cube = np.concatenate((pos, q, vel, w), -1)
    steps = records - 1
    if family == 2:
        contacts = np.zeros((steps, 3), np.uint8)
    else:
        contacts = np.stack([telegraph(rng, steps, 0.3, 0.4, 0.01) for _ in range(2)]
                            + [telegraph(rng, steps, 0.8, 0.3, 0.01, 0.8)], -1)
    phase = np.minimum(np.arange(records) // 25, 7).astype(np.int8)   # the controller phase: 0.25 s each, then release
    return dict(states={"arm": np.concatenate((ang, rate), -1), "cube": cube}, action=act, contacts=contacts,
                per_record=dict(phase=phase),
                per_interval=dict(contact_force=contacts * rng.uniform(1, 20, contacts.shape).astype(np.float32)))


def write(out, system_name, episodes=60, records=301, seed=0, shares=(0.5, 0.3, 0.2)):
    """Write a synthetic data directory of the given system (g1_locomotion or so101_grasp) -> system dict."""
    rng = np.random.default_rng(seed)
    system, episode = (g1_system, g1_episode) if system_name == "g1_locomotion" else (so101_system, so101_episode)
    system = system(records)
    counts = np.round(np.asarray(shares) * episodes).astype(int)
    family = np.repeat(np.arange(len(counts)), counts)
    eps, reasons = [], []
    for e, f in enumerate(family):
        length = records if f == 0 or rng.random() < 0.5 else int(rng.integers(records // 2, records))
        ep = episode(rng, records, int(f))
        ep["states"] = {k: v[:length] for k, v in ep["states"].items()}
        ep["action"] = ep["action"][:length]
        ep["contacts"] = ep["contacts"][:length - 1]
        ep["per_interval"] = {k: v[:length - 1] for k, v in ep["per_interval"].items()}
        ep["scalars"] = dict(episode_id=e, group_id=e, seed=seed * 100000 + e)
        reasons.append(0 if length == records else 1)
        eps.append(ep)
    arrays = export.pack_episodes(eps, system)
    arrays.update(family=family.astype(np.int16), terminal_reason=np.asarray(reasons, np.int8),
                  splits=export.assign_splits(family, arrays["group_id"], seed=seed))
    export.write_dataset(out, system, arrays, manifest=dict(synthetic=True))
    return system
