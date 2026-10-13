"""MuJoCo ground truth for new G1 policies: the exact scene of the contact-NRD G1 collector
(nedm.g1_footstep_rl.g1_scene), optionally with a stepping-planks terrain, a step function that takes all
27 joint targets every 20 ms with the collector's contact labels and end rules, the 20 ms step-averaged state that
the NRD sees, the frozen HOMIE teacher and an offscreen video helper.

numpy + mujoco + onnxruntime; no torch.

Conventions (as in the NRD):
- state [67] float64: pelvis x, y, z (world, ground top at z = 0), quaternion wxyz (pelvis to world), 27 joint
  angles, pelvis velocity (world), pelvis angular velocity (world), 27 joint rates; joint order
  homie_policy.JOINT_NAMES.
- action [27]: absolute joint position targets, held for one 20 ms control period = 2 records of 10 ms = 20 physics
  steps of 1 ms (collector Slot.advance).
- state(): the instantaneous MuJoCo rates. avg_state(): the rates replaced by their 20 ms step averages over the
  last control period, exactly as nedm.contact_nrd.data.step_average_rates; at the first control step of an episode
  the recorded (instantaneous) rates are kept, as in the NRD data. A policy trained in the NRD must read avg_state().

Terrain (planks_task.Layout or None = the collector's flat floor):
- The floor plane moves down to z = -pit_depth (the pit bottom). One box geom per plank (start platform, planks,
  final platform): top face at z = 0, bottom at the pit floor, length along the path from the layout, |l| <=
  half_width across it. Same friction, solref and solimp as the floor (every geom gets the collector's values).
- All planks and the pit floor are ground (class FLOOR): a non-foot robot geom touching any of them ends the episode
  (event 1, collector rule), a foot touching a plank is a foot contact (labels as on the flat floor). A foot that
  touches the pit floor ends the episode with the new event 5 (foot_in_pit).

Validation (during development): flat floor + onnxruntime teacher with instantaneous rates is bit-identical to the
data collector on a nominal plan (states, actions, contact labels).
"""
from __future__ import annotations

import copy
import functools
from pathlib import Path

import mujoco
import numpy as np

from nedm.g1_footstep_rl import REPO_ROOT
from nedm.g1_footstep_rl import g1_scene as G
from nedm.g1_footstep_rl import homie_policy as H

FLOOR, LEFT, RIGHT, ROBOT = G.FLOOR, G.LEFT, G.RIGHT, G.ROBOT
EVENTS = list(G.TERMINAL_REASONS) + ["foot_in_pit"]     # 0 planned end, 1 ground contact, 2 self contact,
FOOT_IN_PIT = 5                                         # 3 low base, 4 non-finite, 5 a foot on the pit floor
HOMIE_WEIGHTS = REPO_ROOT / "artifacts/g1_footstep_rl/homie/homie_weights.npz"   # deploy.onnx weights (fetch_homie.py)
PLANK_RGBA = ([0.62, 0.45, 0.28, 1.0], [0.50, 0.36, 0.22, 1.0])   # alternate two wood tones (visible edges)
PLATFORM_RGBA = [0.45, 0.47, 0.50, 1.0]


# ---- scene -----------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=4)
def _check_sources(mjcf, mjcf_sha256, assets_sha256, cfg_json):
    """The collector's source checks (MJCF and mesh hashes), done once per process and config."""
    import json
    cfg = json.loads(cfg_json)
    if mjcf_sha256 and G.sha256_file(mjcf) != mjcf_sha256:
        raise ValueError(f"{mjcf}: sha256 differs from the config")
    if assets_sha256 and G.model_files(cfg)[1] != assets_sha256:
        raise ValueError(f"{mjcf}: the files it loads differ from config sources.assets_sha256 (see model_files)")
    return True


def build_scene(cfg, terrain=None):
    """G.build_model (g1_scene.py) with an optional plank terrain (planks_task.Layout). The flat case
    (terrain None) compiles the same model as G.build_model: the code below is a line-by-line copy with the terrain
    added to the spec before the per-geom contact settings (so the planks get the floor's friction, solref and
    solimp)."""
    import json
    phys = cfg["physics"]
    path = G.source_path(cfg, "mjcf")
    _check_sources(str(path), cfg["sources"].get("mjcf_sha256"), cfg["sources"].get("assets_sha256"),
                   json.dumps(cfg, sort_keys=True))
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
        elif phys["foot"] == "sole_box":
            for g in spheres:
                spec.delete(g)
            lo, hi = np.array(phys["sole_box_min"]), np.array(phys["sole_box_max"])
            body.add_geom(name=f"{side}_foot_0", type=mujoco.mjtGeom.mjGEOM_BOX, size=(hi - lo) / 2,
                          pos=(hi + lo) / 2, rgba=[0.2, 0.2, 0.2, 1])
        else:
            raise ValueError(f"unknown foot {phys['foot']}")
    if terrain is not None:
        add_planks(spec, terrain)
    for g in spec.geoms:
        g.friction, g.solref, g.solimp = phys["friction"], phys["solref"], phys["solimp"]
    for a in spec.actuators:
        j = H.JOINT_NAMES.index(a.target)
        a.name = H.JOINT_NAMES[j]
        a.set_to_position(kp=float(H.KP[j]), kv=float(H.KD[j]))
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-H.TORQUE_LIMIT[j], H.TORQUE_LIMIT[j]]
    for name in H.JOINT_NAMES:
        joint = spec.joint(name)
        if phys.get("joint_frictionloss") is not None:
            joint.frictionloss = phys["joint_frictionloss"]
        if phys.get("joint_damping") is not None:
            joint.damping = np.r_[phys["joint_damping"], np.zeros(len(joint.damping) - 1)]
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
    if names != list(H.JOINT_NAMES) or acts != list(H.JOINT_NAMES) or model.nq != 7 + H.N_JOINTS:
        raise ValueError("joint or actuator order differs from homie_policy.JOINT_NAMES")
    return model


def add_planks(spec, layout):
    """Move the floor plane to the pit bottom and add one box per plank (top at z = 0) to the world body."""
    floor = spec.geom("floor")
    floor.pos = [0.0, 0.0, -layout.pit_depth]
    quat = G.yaw_quat(layout.heading)
    last = len(layout.intervals) - 1
    for k, (a, b) in enumerate(layout.intervals):
        centre = layout.world_xy((a + b) / 2, 0.0)
        rgba = PLATFORM_RGBA if k in (0, last) else PLANK_RGBA[k % 2]
        spec.worldbody.add_geom(name=f"plank_{k:02d}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[(b - a) / 2, layout.half_width, layout.pit_depth / 2],
                                pos=[centre[0], centre[1], -layout.pit_depth / 2], quat=quat, rgba=rgba)


def geom_classes(model):
    """G.geom_classes with the planks as ground: [ngeom] FLOOR (floor or plank_*), LEFT / RIGHT foot, other ROBOT
    geom; -1 for geoms that collide with nothing."""
    out = np.full(model.ngeom, -1, np.int64)
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        if name == "floor" or name.startswith("plank_"):
            out[g] = FLOOR
        elif name.startswith("left_foot"):
            out[g] = LEFT
        elif name.startswith("right_foot"):
            out[g] = RIGHT
        elif model.geom_bodyid[g] > 0:
            out[g] = ROBOT
    return out


class TerrainSlot(G.Slot):
    """G.Slot whose record interval also ends the episode when a foot touches the pit floor (geom id `pit`; -1 on
    the flat floor, where interval() runs the collector's code unchanged)."""

    def __init__(self, model, cfg, classes, feet, pit=-1):
        super().__init__(model, cfg, classes, feet, None)
        self.pit = int(pit)
        self.body_of = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or "world"
                        for g in range(model.ngeom)]
        self.geom_name = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}" for g in range(model.ngeom)]
        self.event_contact = None

    def start(self, plan, policy, i):
        self.event_contact = None
        super().start(plan, policy, i)

    def _blame(self, rows):
        """Remember the first contact that caused an end event: (robot body, other geom or body, world position)."""
        c = int(rows[0])
        g1, g2 = (int(v) for v in self.d.contact.geom[c])
        if self.classes[g1] == FLOOR:
            g1, g2 = g2, g1
        other = self.geom_name[g2] if self.classes[g2] == FLOOR else self.body_of[g2]
        self.event_contact = (self.body_of[g1], other, self.d.contact.pos[c].round(4).tolist())

    def interval(self, target):
        """Copy of G.Slot.interval (g1_scene.py) plus the foot_in_pit check."""
        m, d, classes = self.m, self.d, self.classes
        d.ctrl[:] = target
        fmax, on, tau = np.zeros(2), np.zeros(2, np.uint8), np.zeros(H.N_JOINTS)
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
                    self._blame(np.flatnonzero(lo >= LEFT))
                    return 2, fmax, on, tau, lim_steps, lim_mask
                if ((lo == FLOOR) & (hi == ROBOT)).any():
                    self._blame(np.flatnonzero((lo == FLOOR) & (hi == ROBOT)))
                    return 1, fmax, on, tau, lim_steps, lim_mask
                if self.pit >= 0 and (((d.contact.geom[:n] == self.pit).any(1)) & (hi <= RIGHT)).any():
                    self._blame(np.flatnonzero((d.contact.geom[:n] == self.pit).any(1) & (hi <= RIGHT)))
                    return FOOT_IN_PIT, fmax, on, tau, lim_steps, lim_mask
                for c in np.flatnonzero(lo == FLOOR):
                    mujoco.mj_contactForce(m, d, c, self.f6)
                    f[hi[c] - 1] += self.f6[0]
            if d.nl:
                a = d.ne + d.nf
                rows = slice(a, a + d.nl)
                act = (d.efc_type[rows] == G.LIMIT_JOINT) & (d.efc_force[rows] > 0)
                if act.any():
                    lim_steps += 1
                    for j in d.efc_id[rows][act]:
                        lim_mask |= 1 << (int(j) - 1)
            if d.qpos[2] < self.min_z and not self.branched:
                return 3, fmax, on, tau, lim_steps, lim_mask
            if d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number != self.badqacc or not np.isfinite(d.qpos).all():
                return 4, fmax, on, tau, lim_steps, lim_mask
            fmax = np.maximum(fmax, f)
            on += f > self.threshold
        return 0, fmax, on, tau / self.n_sub, lim_steps, lim_mask


# ---- state helpers ---------------------------------------------------------------------------------------------
def step_average(prev, cur, dt=0.02):
    """[67] state `cur` with its rate channels replaced by the 20 ms step averages from `prev` (one control step
    earlier): numpy copy of nedm.contact_nrd.data._step_average (positions and joint angles: difference / dt;
    angular velocity: rotation vector of q * conj(q_prev) / dt, world frame, shortest rotation)."""
    prev, out = np.asarray(prev, np.float64), np.array(cur, np.float64)
    out[34:37] = (out[0:3] - prev[0:3]) / dt
    out[40:67] = (out[7:34] - prev[7:34]) / dt
    q1, q0 = out[3:7], prev[3:7]
    w0, v0 = q0[0], -q0[1:]
    w1, v1 = q1[0], q1[1:]
    w = w1 * w0 - np.dot(v1, v0)
    v = w1 * v0 + w0 * v1 + np.cross(v1, v0)
    if w < 0:
        w, v = -w, -v
    n = np.linalg.norm(v)
    out[37:40] = (v / max(n, 1e-12) * 2 * np.arctan2(n, w) if n > 1e-12 else 2 * v) / dt
    return out


def world_to_body(quat, v_world):
    """Rotate a world vector into the pelvis frame (inverse of mju_rotVecQuat body -> world)."""
    q = np.asarray(quat, np.float64)
    out = np.empty(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v_world, np.float64), q / np.linalg.norm(q) * np.array([1, -1, -1, -1]))
    return out


def tilt_deg(x):
    """Angle between the pelvis z axis and the world z axis (deg) of a [67] state."""
    q = np.asarray(x[3:7], np.float64)
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2) / (q @ q), -1, 1))))


def fallen(x):
    """Evaluation fall flag (configs/contact_nrd/evaluation/g1_locomotion.json): pelvis z < 0.35 m or tilt > 45 deg."""
    return bool(x[2] < 0.35 or tilt_deg(x) > 45.0)


def yaw_of(x):
    q = x[3:7]
    return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


# ---- teachers --------------------------------------------------------------------------------------------------
def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


class NumpyHomie:
    """deploy.onnx in numpy (weights from homie_weights.npz, the ONNX initializers) with HomiePolicy's history logic
    and interface."""
    EST, ACT = ("estimator.0", "estimator.2", "estimator.4"), ("actor.0", "actor.2", "actor.4", "actor.6")

    def __init__(self, batch, weights=HOMIE_WEIGHTS):
        with np.load(weights) as f:
            self.w = {k: f[k].astype(np.float32) for k in f.files}
        self.history = np.zeros((batch, H.HISTORY, H.FRAME), np.float32)
        self.last_action = np.zeros((batch, H.N_LEG), np.float32)

    def forward(self, x):
        h = np.asarray(x, np.float32)
        for k, name in enumerate(self.EST):
            h = h @ self.w[name + ".weight"].T + self.w[name + ".bias"]
            if k < len(self.EST) - 1:
                h = _elu(h)
        lat = h[:, 3:] / np.maximum(np.linalg.norm(h[:, 3:], axis=-1, keepdims=True), 1e-12)
        h = np.concatenate((np.asarray(x, np.float32)[:, -H.FRAME:], h[:, :3], lat), -1)
        for k, name in enumerate(self.ACT):
            h = h @ self.w[name + ".weight"].T + self.w[name + ".bias"]
            if k < len(self.ACT) - 1:
                h = _elu(h)
        return h

    def reset(self, ids):
        self.history[ids] = 0.0
        self.last_action[ids] = 0.0

    def step(self, ids, command, quat, ang_vel_body, q, dq):
        f = H.frame(command, quat, ang_vel_body, q, dq, self.last_action[ids])
        self.history[ids] = np.concatenate((self.history[ids, 1:], f[:, None]), 1)
        raw = self.forward(self.history[ids].reshape(len(ids), -1)).astype(np.float32)
        act = np.clip(raw, -H.ACTION_CLIP, H.ACTION_CLIP)
        self.last_action[ids] = act
        return raw, act, H.DEFAULT_ANGLES[:H.N_LEG] + H.ACTION_SCALE * act


class NoTeacher:
    """Placeholder for policies without the teacher (Slot.start calls reset)."""

    def reset(self, ids):
        pass


def make_teacher(kind, cfg, weights=HOMIE_WEIGHTS):
    if kind == "ort":
        return H.HomiePolicy(G.source_path(cfg, "policy"), 1, cfg["sources"]["policy_sha256"], threads=1)
    if kind == "numpy":
        return NumpyHomie(1, weights)
    if kind is None:
        return NoTeacher()
    raise ValueError(f"unknown teacher {kind!r} (ort, numpy or None)")


# ---- the environment ---------------------------------------------------------------------------------------------
class G1MujocoEnv:
    """One CPU MuJoCo environment of the G1 collector, with an optional plank terrain.

    env = G1MujocoEnv(terrain=layout, teacher="ort", avg_rates=True, duration_s=15.0)
    x = env.reset(pose=(x0, y0, yaw0))                 # standing default pose, lowest foot 2 mm above z = 0
    while True:                                         # one iteration = 20 ms
        raw, legs = env.teacher_action(command)         # (optional) the teacher; call once per control step
        target = np.concatenate((legs, env.default_upper()))     # or any policy(env.avg_state()) -> [27]
        done, event = env.step(target, raw, command)    # event: EVENTS index, 0 = planned end
        if done: break

    duration_s sets the planned episode length."""

    def __init__(self, cfg=None, terrain=None, teacher="ort", avg_rates=False, duration_s=10.0,
                 weights=HOMIE_WEIGHTS):
        self.cfg = copy.deepcopy(cfg or G.load_config())
        self.cfg["steps"]["task_duration_s"] = float(duration_s)
        self.duration = float(duration_s)
        self.terrain = terrain
        self.model = build_scene(self.cfg, terrain)
        self.classes = geom_classes(self.model)
        self.feet = np.flatnonzero((self.classes == LEFT) | (self.classes == RIGHT))
        self.foot_geoms = np.stack([np.flatnonzero(self.classes == c) for c in (LEFT, RIGHT)])   # [2, 4]
        pit = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor") if terrain is not None else -1
        self.slot = TerrainSlot(self.model, self.cfg, self.classes, self.feet, pit)
        self.teacher_kind = teacher
        self.teacher = make_teacher(teacher, self.cfg, weights)
        self.avg_rates = bool(avg_rates)
        self.kin = mujoco.MjData(self.model)      # forward kinematics of states (does not touch the simulation)

    # -- episode ------------------------------------------------------------------------------------------------
    @property
    def data(self):
        return self.slot.d

    @property
    def time(self):
        """Seconds since the episode start (record clock)."""
        return self.slot.rec * self.cfg["steps"]["record_s"]

    def reset(self, pose=(0.0, 0.0, 0.0), plan=None, family=0, seed=0, q0=None):
        """Start an episode: the collector's start (default joint angles or q0, zero velocities, lowest foot point
        start.foot_clearance_m = 2 mm above z = 0, teacher history zero) at world pose (x, y, yaw). A full collector
        plan (G.plan_task) can be given instead (validation); then pose is ignored."""
        if plan is None:
            x, y, yaw = (float(v) for v in pose)
            plan = dict(kind="task", family=int(family), seed=int(seed), duration=self.duration, yaw=yaw,
                        q0=H.DEFAULT_ANGLES.copy() if q0 is None else np.asarray(q0, np.float64), vel0=np.zeros(3),
                        ang_vel0=np.zeros(3), upper=None, leg_noise=0.0,
                        commands=np.array([(0.0, 0.0, 0.0, 0.0, H.FULL_HEIGHT)]), group=0)
        else:
            x = y = 0.0
        self.slot.start(plan, self.teacher, 0)
        if x or y:   # Slot.start places the pelvis at x = y = 0; move it (the floor and planks are at z = 0)
            d = self.slot.d
            d.qpos[:2] += (x, y)
            mujoco.mj_forward(self.model, d)
            self.slot.buf["state"][0] = G.robot_state(d)
        return self.state()

    def state(self):
        """[67] state with the instantaneous MuJoCo rates."""
        return G.robot_state(self.slot.d)

    def avg_state(self):
        """[67] state with 20 ms step-average rates over the last control period (NRD rates). At the first control
        step of an episode (no state 20 ms earlier) the instantaneous rates are kept, as in the NRD data."""
        s = self.slot
        if s.rec < s.per_ctrl:
            return self.state()
        return step_average(s.buf["state"][s.rec - s.per_ctrl], s.buf["state"][s.rec])

    def command(self):
        """The plan's command (vx, vy, wz, height) at the current record (validation plans)."""
        return self.slot.command()

    def default_upper(self):
        """The plan's upper-body targets (defaults = zeros unless the plan moves them)."""
        return self.slot.upper_targets()

    def teacher_action(self, command=None):
        """(raw [12], leg targets [12]) of the teacher for the current state and command (vx, vy, wz, height).
        Updates its 6-frame history: call exactly once per control step. avg_rates=True feeds it the 20 ms averaged
        joint rates and pelvis angular velocity (what it would see inside the NRD); False the instantaneous ones
        (the collector)."""
        d = self.slot.d
        cmd = self.slot.command() if command is None else np.asarray(command, np.float64)
        if self.avg_rates:
            x = self.avg_state()
            wb, dq = world_to_body(d.qpos[3:7], x[37:40]), x[40:67]
        else:
            wb, dq = d.qvel[3:6], d.qvel[6:]
        raw, _, legs = self.teacher.step([0], cmd[None], d.qpos[None, 3:7], np.asarray(wb)[None], d.qpos[None, 7:],
                                         np.asarray(dq)[None])
        return raw[0], legs[0]

    def step(self, target, raw=None, cmd=None):
        """Apply 27 joint targets for one 20 ms control period (2 records, 20 physics steps) with the collector's
        bookkeeping (Slot.advance without leg noise). raw [12] and cmd [4] are only recorded
        (extras policy_output and command). Returns (done, event); event indexes EVENTS (0 = planned end)."""
        s, b = self.slot, self.slot.buf
        target = np.asarray(target, np.float64)
        cmd = s.command() if cmd is None else np.asarray(cmd, np.float64)
        raw = np.zeros(H.N_LEG) if raw is None else raw
        for _ in range(s.per_ctrl):
            k = s.rec
            b["action"][k], b["command"][k], b["policy_output"][k] = target, cmd, raw
            event, fmax, on, tau, lim_steps, lim_mask = s.interval(target)
            if event:
                s.event = event
                return True, event
            b["contact_force"][k], b["contact_steps"][k], b["torque"][k] = fmax, on, tau
            b["limit_steps"][k], b["limit_joints"][k] = lim_steps, lim_mask
            b["contacts"][k] = on > 0
            s.rec += 1
            b["state"][s.rec] = G.robot_state(s.d)
            if s.rec == s.n_rec - 1:
                b["action"][s.rec], b["command"][s.rec], b["policy_output"][s.rec] = target, cmd, raw
                return True, 0
        return False, 0

    def step_contacts(self):
        """[2] bool: left / right foot touched the ground in the last control period (OR of its 2 record labels)."""
        s = self.slot
        a = max(0, s.rec - s.per_ctrl)
        return s.buf["contacts"][a:s.rec].any(0) if s.rec > a else np.zeros(2, bool)

    # -- kinematics -----------------------------------------------------------------------------------------------
    def spheres(self, x=None):
        """[2, 4, 3] world centres of the foot spheres (left, right; MJCF order = planks_task.FOOT_SPHERES) of the
        current state or of a [67] state x."""
        k = self.kin
        k.qpos[:] = self.slot.d.qpos if x is None else G.state_to_qpos_qvel(x)[0]
        mujoco.mj_kinematics(self.model, k)
        return k.geom_xpos[self.foot_geoms].copy()


# ---- video -----------------------------------------------------------------------------------------------------
def render_states(model, states, out, fps=25, record_s=0.01, size=(960, 544), heading=0.0, distance=3.2,
                  elevation=-18.0, side=True, label=None):
    """Write an mp4 of a recorded [T, 67] state sequence in `model` (e.g. G1MujocoEnv.model with its planks).
    The camera follows the pelvis; side=True looks at the robot from its right side (perpendicular to `heading`,
    rad), else from behind. label(t_index) -> optional text drawn in the top-left corner. Offscreen: set MUJOCO_GL
    (egl, or osmesa where EGL does not work) before mujoco is imported."""
    import imageio
    data = mujoco.MjData(model)
    vis = model.vis.global_          # offscreen buffer large enough (visual setting only, no effect on physics)
    vis.offwidth, vis.offheight = max(vis.offwidth, size[0]), max(vis.offheight, size[1])
    renderer = mujoco.Renderer(model, height=size[1], width=size[0])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation = distance, elevation
    cam.azimuth = np.degrees(heading) + (90.0 if side else 0.0)
    step = max(1, round(1 / fps / record_s))
    draw = None
    if label is not None:
        from PIL import Image, ImageDraw
        draw = (Image, ImageDraw)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out, fps=round(1 / (step * record_s)), codec="libx264", quality=7,
                            macro_block_size=8) as writer:
        for t in range(0, len(states), step):
            data.qpos[:], data.qvel[:] = G.state_to_qpos_qvel(states[t])
            mujoco.mj_forward(model, data)
            cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.45]
            renderer.update_scene(data, cam)
            img = renderer.render()
            if draw is not None:
                im = draw[0].fromarray(img)
                draw[1].Draw(im).text((10, 10), label(t), fill=(255, 255, 255))
                img = np.asarray(im)
            writer.append_data(img)
    renderer.close()


# ---- the stepping-planks task in MuJoCo ----------------------------------------------------------------------------
def make_planks_env(layout, teacher="ort", avg_rates=False, cfg=None, episode_s=None):
    """G1MujocoEnv on the layout's planks, planned length = the task's time limit (planks_task.EPISODE_S)."""
    from nedm.g1_footstep_rl import planks_task as P
    return G1MujocoEnv(cfg, layout, teacher, avg_rates, P.EPISODE_S if episode_s is None else episode_s)


def run_planks_episode(env, layout, policy, keep_states=False):
    """One planks episode in MuJoCo. The robot starts standing at the layout's origin and heading (path s = l = 0).

    policy(env, progress) -> (target [27], raw [12] or None, command [4] or None) is called once per 20 ms control
    step (it may call env.teacher_action itself; it must read env.avg_state() if it was trained in the NRD).
    progress is the planks_task.PlanksProgress of the episode (updated after every step).

    The episode stops at the first of: success (planks_task rule), a collector end event or foot_in_pit (outcome =
    its EVENTS name), the evaluation fall flag (pelvis z < 0.35 m or tilt > 45 deg; outcome "fall_flag"), or the
    time limit env.duration (outcome "timeout"). Returns a metrics dict (and the [T, 67] states if keep_states)."""
    from nedm.g1_footstep_rl import planks_task as P
    if env.terrain is not layout:
        raise ValueError("the env must be built on this layout (make_planks_env(layout))")
    env.reset(pose=(layout.origin[0], layout.origin[1], layout.heading))
    prog = P.PlanksProgress(layout, spheres0=env.spheres())
    outcome, last = None, np.ones(2, bool)        # standing start: both feet touched before the first step
    while outcome is None:
        target, raw, cmd = policy(env, prog)
        done, event = env.step(target, raw, cmd)
        if event:
            outcome = EVENTS[event]
            break
        x = env.state()
        raw_contact = env.step_contacts()
        success = prog.update(x[:2], env.spheres(), raw_contact | last)   # model label: step k or k - 1
        last = raw_contact
        # the fall flag is an end event: success must come before it, and the NRD env
        # also lets a failure win over a success in the same step
        if fallen(x):
            outcome = "fall_flag"
        elif success:
            outcome = "success"
        elif done:
            outcome = "timeout"
    m = prog.metrics()
    ok = outcome == "success"
    blame = env.slot.event_contact
    res = dict(level=layout.level, seed=layout.seed, outcome=outcome, success=ok,
               event_body=blame[0] if blame else None, event_other=blame[1] if blame else None,
               clean_success=bool(ok and m["unsupported_steps"] == 0 and m["touchdowns_over_gap"] == 0),
               fell=outcome not in ("success", "timeout"), end_s=round(env.time, 2),
               time_to_success_s=round(env.time - P.STAND_S, 2) if ok else None, **m)
    if keep_states:
        s = env.slot
        res["states"] = s.buf["state"][:s.rec + 1].copy()
    return res


class HomieCommandPolicy:
    """The HOMIE teacher's legs + default upper body under a velocity command from command_fn(env, progress) ->
    (vx, vy, wz, height). The command is zero during the standing start (planks_task.STAND_S)."""

    def __init__(self, command_fn, stand_s=None):
        from nedm.g1_footstep_rl import planks_task as P
        self.command_fn = command_fn
        self.stand_s = P.STAND_S if stand_s is None else stand_s

    def __call__(self, env, progress):
        if env.time < self.stand_s - 1e-9:
            cmd = np.array([0.0, 0.0, 0.0, H.FULL_HEIGHT])
        else:
            cmd = np.asarray(self.command_fn(env, progress), np.float64)
        raw, legs = env.teacher_action(cmd)
        return np.concatenate((legs, env.default_upper())), raw, cmd
