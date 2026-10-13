"""The MuJoCo scene of the G1 locomotion data (contact NRD case g1_locomotion): the frozen HOMIE policy walks the
27-joint Unitree G1 (OpenHomie g1.xml) on a flat plane in CPU MuJoCo.

This module is the data collector of that case cut down to its scene: the config loader, the model build with its
source checks, the contact classes, the state helpers and the record interval of one environment (Slot). It has no
episode plans, no workers and no data export. nedm.g1_footstep_rl.mujoco_env builds the evaluation scenes from it.

Scene (contact_nrd/collection_contract.md, G1 column):
- Bodies: 0 robot (moving, articulated), 1 ground (fixed plane, top at z = 0). Robot state, 67 channels: pelvis
  position (3), pelvis quaternion wxyz (4), 27 joint angles, pelvis velocity (3, world), pelvis angular velocity (3,
  world: MuJoCo's free-joint body-frame rate rotated), 27 joint rates. Joint order: homie_policy.JOINT_NAMES.
- action [27]: the joint position targets applied in the interval after each record: 12 leg targets and 15
  upper-body targets, held for the 20 ms control period.
- Pairs: left foot-ground, right foot-ground. A pair is on in record interval k if in any physics step inside it
  the pair's summed mj_contactForce normal force exceeds contact.threshold_n. The contact list read after mj_step
  belongs to the state at the start of that step: it is the force applied during the step.
- Records 10 ms, control 20 ms, physics physics.timestep_s. Floor and all other contacts: MuJoCo soft contacts with
  physics.solref and physics.solimp on every geom. Joint limits: the MJCF ranges with MuJoCo's default softness. The
  MJCF and every mesh it loads must match sources.mjcf_sha256 and sources.assets_sha256.
- End events of a record interval (Slot.interval): a contact of any other robot geom with the ground, a robot
  self-contact, the pelvis below termination.min_base_height_m, or a non-finite state (TERMINAL_REASONS).
The config is configs/g1_footstep_rl/g1_scene.json (the collection config of the released data with the vendor
directory under artifacts/); scripts/g1_footstep_rl/fetch_homie.py fetches the vendor files. numpy and mujoco only.
"""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from nedm.g1_footstep_rl import REPO_ROOT
from nedm.g1_footstep_rl.homie_policy import DEFAULT_ANGLES, JOINT_NAMES, KD, KP, N_JOINTS, N_LEG, TORQUE_LIMIT

DEFAULT_CONFIG = REPO_ROOT / "configs/g1_footstep_rl/g1_scene.json"
FAMILIES = [
    dict(name="nominal", group="nominal", select=True,
         note="random command schedules (stand, start, walk, turn, squat, stop), nominal upper body, random heading"),
    dict(name="difficult", group="difficult", select=True,
         note="abrupt commands, noise on the applied leg targets, perturbed start, moving upper body"),
    dict(name="probe_drop", group="probe", select=False,
         note="released 0.05-0.5 m above the ground from a walking state; flight, landing, teacher continues"),
    dict(name="probe_landing", group="probe", select=False,
         note="released 0.01-0.12 m above the ground from a walking state with a tilt; one foot lands first"),
]
FAMILY = {f["name"]: k for k, f in enumerate(FAMILIES)}
TERMINAL_REASONS = ["planned_end", "ground_contact", "self_contact", "low_base", "nonfinite"]
FLOOR, LEFT, RIGHT, ROBOT = 0, 1, 2, 3
LIMIT_JOINT = mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
CHANNELS = (["x", "y", "z", "qw", "qx", "qy", "qz"] + [n.removesuffix("_joint") for n in JOINT_NAMES]
            + ["vx", "vy", "vz", "wx", "wy", "wz"] + [n.removesuffix("_joint") + "_rate" for n in JOINT_NAMES])
CHANNEL_TYPES = (["position"] * 3 + ["quaternion"] * 4 + ["angle"] * N_JOINTS + ["velocity"] * 3
                 + ["angular_velocity"] * 3 + ["angular_rate"] * N_JOINTS)
STATE_DIM = len(CHANNELS)


# ---- configuration and scene ---------------------------------------------------------------------
def load_config(path=DEFAULT_CONFIG, overrides=()):
    """Read the JSON config; overrides are "a.b=value" strings (value parsed as JSON when possible)."""
    cfg = json.loads(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        node = cfg
        *head, last = key.split(".")
        for k in head:
            node = node[k]
        node[last] = value
    return cfg


def source_path(cfg, key):
    return REPO_ROOT / cfg["sources"]["dir"] / cfg["sources"][key]


@functools.lru_cache(maxsize=4)
def _model_files(mjcf):
    mjcf = Path(mjcf)
    if "<include" in mjcf.read_text():
        raise ValueError(f"{mjcf}: <include> is not supported by the asset check")
    spec = mujoco.MjSpec.from_file(str(mjcf))
    paths = [mjcf] + [mjcf.parent / spec.meshdir / m.file for m in spec.meshes if m.file]
    paths += [mjcf.parent / spec.texturedir / t.file for t in spec.textures if t.file]
    paths += [mjcf.parent / h.file for h in spec.hfields if h.file] + [mjcf.parent / s.file for s in spec.skins if s.file]
    return tuple(sorted(set(paths)))


def model_files(cfg):
    """{path relative to sources.dir: sha256} of every file the MJCF loads (the MJCF and its meshes) and their
    digest (sha256 of the sorted "sha256  path" lines), which config sources.assets_sha256 pins."""
    root = REPO_ROOT / cfg["sources"]["dir"]
    files = {str(p.relative_to(root)): sha256_file(p) for p in _model_files(str(source_path(cfg, "mjcf")))}
    lines = "".join(f"{h}  {p}\n" for p, h in sorted(files.items()))
    return files, hashlib.sha256(lines.encode()).hexdigest()



def build_model(cfg):
    """The vendor MJCF with this collector's changes (all listed in system.json physics_config):
    position servos with the training PD gains and URDF force limits instead of torque motors, named foot geoms
    (or a sole box, physics.foot), one friction, solref and solimp for every geom, no friction, damping or armature
    on the free joint, no sensors, the solver options of the config. The MJCF and every mesh it loads must match
    sources.mjcf_sha256 and sources.assets_sha256."""
    phys = cfg["physics"]
    path = source_path(cfg, "mjcf")
    want = cfg["sources"].get("mjcf_sha256")
    if want and sha256_file(path) != want:
        raise ValueError(f"{path}: sha256 differs from the config")
    want = cfg["sources"].get("assets_sha256")
    if want and model_files(cfg)[1] != want:
        raise ValueError(f"{path}: the files it loads differ from config sources.assets_sha256 (see model_files)")
    if phys["solref"][0] < 2 * phys["timestep_s"]:
        raise ValueError("physics.solref time constant below 2 physics steps: MuJoCo would raise it silently")
    spec = mujoco.MjSpec.from_file(str(path))
    for s in list(spec.sensors):
        spec.delete(s)
    for side in ("left", "right"):
        body = spec.body(f"{side}_ankle_roll_link")
        spheres = [g for g in body.geoms if g.type == mujoco.mjtGeom.mjGEOM_SPHERE]
        if phys["foot"] == "mjcf_spheres":
            for n, g in enumerate(spheres):
                g.name = f"{side}_foot_{n}"
        elif phys["foot"] == "sole_box":   # the bottom face of the URDF foot collision mesh (convex hull)
            for g in spheres:
                spec.delete(g)
            lo, hi = np.array(phys["sole_box_min"]), np.array(phys["sole_box_max"])
            body.add_geom(name=f"{side}_foot_0", type=mujoco.mjtGeom.mjGEOM_BOX, size=(hi - lo) / 2,
                          pos=(hi + lo) / 2, rgba=[0.2, 0.2, 0.2, 1])
        else:
            raise ValueError(f"unknown foot {phys['foot']}")
    for g in spec.geoms:
        g.friction, g.solref, g.solimp = phys["friction"], phys["solref"], phys["solimp"]
    for a in spec.actuators:
        j = JOINT_NAMES.index(a.target)
        a.name = JOINT_NAMES[j]
        a.set_to_position(kp=float(KP[j]), kv=float(KD[j]))
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-TORQUE_LIMIT[j], TORQUE_LIMIT[j]]
    for name in JOINT_NAMES:
        joint = spec.joint(name)
        if phys.get("joint_frictionloss") is not None:
            joint.frictionloss = phys["joint_frictionloss"]
        if phys.get("joint_damping") is not None:   # MuJoCo 3.15: polynomial coefficients, the first is linear
            joint.damping = np.r_[phys["joint_damping"], np.zeros(len(joint.damping) - 1)]
    # The MJCF joint default (frictionloss 0.1, damping 0.001, armature 0.01) also reaches the free joint: a force
    # between the pelvis and the world. Training has none; the base moves freely.
    base = next(j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE)
    base.frictionloss, base.armature = 0.0, 0.0
    base.damping = np.zeros_like(np.atleast_1d(base.damping))
    model = spec.compile()
    if model.dof_frictionloss[:6].any() or model.dof_damping[:6].any() or model.dof_armature[:6].any():
        raise ValueError("the free joint has friction, damping or armature")
    opt = model.opt
    opt.timestep = phys["timestep_s"]
    opt.integrator = getattr(mujoco.mjtIntegrator, "mjINT_" + phys["integrator"].upper())
    opt.solver = getattr(mujoco.mjtSolver, "mjSOL_" + phys["solver"].upper())
    opt.cone = getattr(mujoco.mjtCone, "mjCONE_" + phys["cone"].upper())
    opt.iterations, opt.ls_iterations = phys["iterations"], phys["ls_iterations"]
    opt.tolerance, opt.impratio = phys["tolerance"], phys["impratio"]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)]
    acts = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in range(model.nu)]
    if names != list(JOINT_NAMES) or acts != list(JOINT_NAMES) or model.nq != 7 + N_JOINTS:
        raise ValueError("joint or actuator order differs from homie_policy.JOINT_NAMES")
    return model


def geom_classes(model):
    """[ngeom] FLOOR / LEFT / RIGHT foot / other ROBOT geom (non-colliding geoms are -1)."""
    out = np.full(model.ngeom, -1, np.int64)
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        if name == "floor":
            out[g] = FLOOR
        elif name.startswith("left_foot"):
            out[g] = LEFT
        elif name.startswith("right_foot"):
            out[g] = RIGHT
        elif model.geom_bodyid[g] > 0:
            out[g] = ROBOT
    return out


def lowest_foot_z(model, data, feet):
    """Lowest point of the sphere or box foot geoms (world z)."""
    z = np.inf
    for g in feet:
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            z = min(z, data.geom_xpos[g, 2] - model.geom_size[g, 0])
        else:
            z = min(z, data.geom_xpos[g, 2] - np.abs(data.geom_xmat[g, 6:9]) @ model.geom_size[g])
    return z


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_mul(a, b):
    out = np.empty(4)
    mujoco.mju_mulQuat(out, np.asarray(a, np.float64), np.asarray(b, np.float64))
    return out


def set_state(model, data, pos, quat, q, vel=(0, 0, 0), ang_vel_world=(0, 0, 0), dq=None):
    """Place the robot; velocities in the world frame (the free joint stores the angular rate in the body frame)."""
    mujoco.mj_resetData(model, data)
    data.qpos[:3], data.qpos[3:7], data.qpos[7:] = pos, np.asarray(quat) / np.linalg.norm(quat), q
    data.qvel[:3] = vel
    w = np.empty(3)
    mujoco.mju_rotVecQuat(w, np.asarray(ang_vel_world, np.float64), data.qpos[3:7] * np.array([1, -1, -1, -1]))
    data.qvel[3:6] = w
    data.qvel[6:] = 0.0 if dq is None else dq
    mujoco.mj_forward(model, data)


def place_feet(model, data, feet, height):
    """Shift the robot vertically so its lowest foot point is at the given height."""
    data.qpos[2] += height - lowest_foot_z(model, data, feet)
    mujoco.mj_forward(model, data)


def robot_state(data):
    """[67] state of the contract order."""
    w = np.empty(3)
    mujoco.mju_rotVecQuat(w, data.qvel[3:6], data.qpos[3:7])
    return np.concatenate((data.qpos, data.qvel[:3], w, data.qvel[6:]))


def state_to_qpos_qvel(x):
    """Inverse of robot_state: [67] -> (qpos [34], qvel [33])."""
    x = np.asarray(x, np.float64)
    quat = x[3:7] / np.linalg.norm(x[3:7])
    w = np.empty(3)
    mujoco.mju_rotVecQuat(w, x[37:40], quat * np.array([1, -1, -1, -1]))
    return np.concatenate((x[:3], quat, x[7:34])), np.concatenate((x[34:37], w, x[40:]))


# ---- provenance -----------------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- one environment ------------------------------------------------------------------------------
class Slot:
    """One environment of a worker batch: an MjData, the running plan and its record buffers. free_model: the model
    without floor collisions (floor_free_model) that core-only branches step in from their branch record on."""

    def __init__(self, model, cfg, classes, feet, free_model=None):
        self.m, self.d = model, mujoco.MjData(model)
        self.floor_model, self.free_model = model, free_model
        self.cfg, self.classes, self.feet = cfg, classes, feet
        self.n_sub = round(cfg["steps"]["record_s"] / model.opt.timestep)
        if abs(self.n_sub * model.opt.timestep - cfg["steps"]["record_s"]) > 1e-12:
            raise ValueError("the physics step must divide the record step")
        self.per_ctrl = round(cfg["steps"]["control_s"] / cfg["steps"]["record_s"])
        self.threshold = cfg["contact"]["threshold_n"]
        self.min_z = cfg["termination"]["min_base_height_m"]
        self.plan = None
        self.branched = False
        self.f6 = np.zeros(6)

    def start(self, plan, policy, i):
        self.m, self.branched = self.floor_model, False   # every plan starts with the floor
        m, d, cfg = self.m, self.d, self.cfg
        self.plan, self.i = plan, i
        self.rng = np.random.default_rng([plan["seed"], 7])
        policy.reset([i])
        if plan["kind"] == "probe":
            snap = plan["snapshot"]
            mujoco.mj_resetData(m, d)
            d.qpos[:], d.qvel[:] = snap["qpos"], snap["qvel"]
            d.qpos[:2] = 0.0
            if plan["tilt"] is not None:
                d.qpos[3:7] = quat_mul(plan["tilt"], d.qpos[3:7])
            mujoco.mj_forward(m, d)
            place_feet(m, d, self.feet, plan["height"])
            policy.history[i], policy.last_action[i] = snap["history"], snap["last_action"]
        else:
            set_state(m, d, (0, 0, 1), yaw_quat(plan["yaw"]), plan["q0"], plan["vel0"], plan["ang_vel0"])
            place_feet(m, d, self.feet, cfg["start"]["foot_clearance_m"])
        n = round(plan["duration"] / cfg["steps"]["record_s"]) + 1
        self.n_rec = n
        self.rec = 0
        self.event = 0
        self.snap_next = 0
        self.snapshots = []
        self.badqacc = d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
        self.buf = None
        if plan["kind"] != "warmup":
            self.buf = dict(state=np.zeros((n, STATE_DIM)), action=np.zeros((n, N_JOINTS)), command=np.zeros((n, 4)),
                            policy_output=np.zeros((n, N_LEG)), contacts=np.zeros((n - 1, 2), np.uint8),
                            contact_force=np.zeros((n - 1, 2)), contact_steps=np.zeros((n - 1, 2), np.uint8),
                            torque=np.zeros((n - 1, N_JOINTS)), limit_steps=np.zeros(n - 1, np.uint8),
                            limit_joints=np.zeros(n - 1, np.int32))
            self.buf["state"][0] = robot_state(d)

    def command(self):
        c = self.plan["commands"]
        t = self.rec * self.cfg["steps"]["record_s"]
        return c[np.searchsorted(c[:, 0], t + 1e-9, side="right") - 1, 1:]

    def upper_targets(self):
        u = self.plan["upper"]
        if u is None:
            return DEFAULT_ANGLES[N_LEG:]
        t = self.rec * self.cfg["steps"]["record_s"]
        return np.array([np.interp(t, u[:, 0], u[:, 1 + j]) for j in range(15)])

    def interval(self, target):
        """Run one record interval; returns (event code, max pair force [2], steps on [2], mean torque [27],
        physics steps with a joint-limit force, bit mask of the joints with one: bit j = JOINT_NAMES[j]). Like the
        contact list, the constraint rows read after mj_step belong to the state at the start of the step."""
        m, d, classes = self.m, self.d, self.classes
        d.ctrl[:] = target
        fmax, on, tau = np.zeros(2), np.zeros(2, np.uint8), np.zeros(N_JOINTS)
        lim_steps, lim_mask = 0, 0
        for _ in range(self.n_sub):
            mujoco.mj_step(m, d)
            tau += d.qfrc_actuator[6:]
            f = np.zeros(2)
            n = d.ncon
            if n:
                gc = classes[d.contact.geom[:n]]
                lo, hi = gc.min(1), gc.max(1)
                if (lo >= LEFT).any():
                    return 2, fmax, on, tau, lim_steps, lim_mask
                if ((lo == FLOOR) & (hi == ROBOT)).any():
                    return 1, fmax, on, tau, lim_steps, lim_mask
                for c in np.flatnonzero(lo == FLOOR):
                    mujoco.mj_contactForce(m, d, c, self.f6)
                    f[hi[c] - 1] += self.f6[0]
            if d.nl:   # limit rows follow the equality and friction-loss rows
                a = d.ne + d.nf
                rows = slice(a, a + d.nl)
                act = (d.efc_type[rows] == LIMIT_JOINT) & (d.efc_force[rows] > 0)
                if act.any():
                    lim_steps += 1
                    for j in d.efc_id[rows][act]:
                        lim_mask |= 1 << (int(j) - 1)
            if d.qpos[2] < self.min_z and not self.branched:   # a core-only branch has no floor to fall on
                return 3, fmax, on, tau, lim_steps, lim_mask
            if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number != self.badqacc or not np.isfinite(d.qpos).all():
                return 4, fmax, on, tau, lim_steps, lim_mask
            fmax = np.maximum(fmax, f)
            on += f > self.threshold
        return 0, fmax, on, tau / self.n_sub, lim_steps, lim_mask

    def snapshot(self, policy):
        """Warm-up walks: keep the state and the policy memory (before this control step) at the planned times."""
        plan = self.plan
        if plan["kind"] == "warmup" and self.snap_next < len(plan["snapshots"]) \
                and self.rec * self.cfg["steps"]["record_s"] >= plan["snapshots"][self.snap_next]:
            self.snapshots.append(dict(qpos=self.d.qpos.copy(), qvel=self.d.qvel.copy(), command=self.command().copy(),
                                       history=policy.history[self.i].copy(), last_action=policy.last_action[self.i].copy()))
            self.snap_next += 1

    def advance(self, raw, leg_target):
        """One control step with the policy's leg targets; returns True when the plan has ended."""
        plan, b = self.plan, self.buf
        cmd = self.command()
        legs = leg_target
        if plan["leg_noise"] > 0:
            legs = legs + np.clip(self.rng.normal(0, plan["leg_noise"], N_LEG), -3 * plan["leg_noise"], 3 * plan["leg_noise"])
        target = np.concatenate((legs, self.upper_targets()))
        for _ in range(self.per_ctrl):
            k = self.rec
            if plan["kind"] == "core" and k == plan["branch_record"]:
                self.m, self.branched = self.free_model, True   # the same MjData steps on without the floor
            if b is not None:
                b["action"][k], b["command"][k], b["policy_output"][k] = target, cmd, raw
            event, fmax, on, tau, lim_steps, lim_mask = self.interval(target)
            if event:
                self.event = event
                return True
            if b is not None:
                b["contact_force"][k], b["contact_steps"][k], b["torque"][k] = fmax, on, tau
                b["limit_steps"][k], b["limit_joints"][k] = lim_steps, lim_mask
                b["contacts"][k] = on > 0
            self.rec += 1
            if b is not None:
                b["state"][self.rec] = robot_state(self.d)
            if self.rec == self.n_rec - 1:
                if b is not None:
                    b["action"][self.rec], b["command"][self.rec], b["policy_output"][self.rec] = target, cmd, raw
                return True
        return False
