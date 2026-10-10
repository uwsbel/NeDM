"""Chrono scene and per-episode simulator / recorder for the SO-101 push-T digital twin.

Scene (world z up, table top at z = 0, SI units):
  * table: a fixed box whose top face is z = 0;
  * SO-101 arm: seven links built from the twinfactory description (mass, COM, inertia, joint frames;
    port of chrono_vla chrono_so101/env.py:_build_robot): ChBodyAuxRef bodies (REF = USD link frame),
    the 234 COACD convex hulls as collision shapes, the base fixed on the table;
    five ChLinkMotorRotationTorque joints (shoulder_pan .. wrist_roll); the jaw is locked closed to the
    gripper with a ChLinkLockLock (no gripper DOF);
  * T-shape: two boxes (bar and stem, plain ChBody at their centres) joined by a ChLinkLockLock.

Control: joint position commands q_cmd (5) held for one control step (zero-order hold, 50 Hz); at every
physics step an explicit PD law tau = kp (q_cmd - q) - kd qd, clipped to the effort limit, is written to the
motor torque functions (q, qd read from the motors at the start of the step).

Recording (every record step, after the physics step that ends at that time):
  arm    q (5), qd (5), TCP position (3), TCP point velocity (3), gripper angular velocity (3)  -> 19
  tshape COM of the two boxes (3), orientation of the bar = T orientation (wxyz, 4), COM velocity (3),
         mass-weighted angular velocity (3)                                                     -> 13
Per record interval (t, t+1], OR over the physics steps inside it (a pair touches at a step when one of
its contacts carries a positive normal force after that step; for arm-T pairs also when Chrono reports a
contact distance <= contact_touch_distance_m, so a resting contact whose force is zero at single solver steps
stays labelled):
  contacts_link [7] arm link (LINK_ORDER) touches either T box; t_table T touches the table;
  arm_table any arm link touches the table; max arm-T normal force (sum over contacts in a step).
Per record (geometry of the simulated body frames, clearance.ArmTClearance):
  finger_gap  signed distance between the fingertip boxes and the T boxes (exact below 20 mm, a lower bound
              above)
  link_gap    smallest signed distance of a convex-hull vertex (links base .. wrist and the upper finger
              parts) to the T boxes (a lower bound above 10 mm); these hull contacts are unreliable in Chrono 10
              Bullet, so the QA step rejects episodes where they come near the T
"""
from __future__ import annotations

import math
import time

import numpy as np

from nedm.so101_push.clearance import FINGER_LINKS, OTHER_LINKS, ArmTClearance
from nedm.so101_push.geometry import TShape
from nedm.so101_push.robot import ARM_JOINTS, ArmModel, link_collision_shapes, matrix_to_quat, rot_z

LINK_ORDER = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "jaw"]
TAG_TABLE, TAG_BAR, TAG_STEM = 100, 101, 102         # arm links use their index in LINK_ORDER as tag


def _V(chrono, p):
    return chrono.ChVector3d(float(p[0]), float(p[1]), float(p[2]))


def _Q(chrono, q):
    return chrono.ChQuaterniond(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _frame(chrono, T):
    return chrono.ChFramed(_V(chrono, T[:3, 3]), _Q(chrono, matrix_to_quat(T[:3, :3])))


def _contact_reporter(chrono, touch_distance: float):
    class Reporter(chrono.ReportContactCallback):
        """Pair flags from one ReportAllContacts pass. A pair touches when one of its contacts has a positive
        normal force; an arm-T pair also when its contact distance is <= touch_distance. arm_t_any: an arm-T
        contact exists inside the collision envelope (with or without force)."""

        def __init__(self):
            super().__init__()
            self.reset()

        def reset(self):
            self.arm_t = np.zeros(len(LINK_ORDER), dtype=bool)
            self.arm_table = False
            self.t_table = False
            self.arm_t_force = 0.0
            self.arm_t_any = False

        def OnReportContact(self, pA, pB, plane_coord, distance, eff_radius, react_forces, react_torques,
                            objA, objB, constraint_offset):
            fn = abs(react_forces.x)
            a = objA.GetPhysicsItem().GetTag()
            b = objB.GetPhysicsItem().GetTag()
            if a > b:
                a, b = b, a
            if b in (TAG_BAR, TAG_STEM) and a < len(LINK_ORDER):
                self.arm_t_any = True
                if fn > 0.0 or distance <= touch_distance:
                    self.arm_t[a] = True
                self.arm_t_force += fn
            elif fn <= 0.0:
                return True
            elif b == TAG_TABLE and a < len(LINK_ORDER):
                self.arm_table = True
            elif a == TAG_TABLE and b in (TAG_BAR, TAG_STEM):
                self.t_table = True
            return True

    return Reporter()


def _rotmat(q) -> np.ndarray:
    w, x, y, z = q.e0, q.e1, q.e2, q.e3
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class PushScene:
    def __init__(self, cfg: dict, model: ArmModel, tshape: TShape, q_init5, t_pose, clearance=None):
        import pychrono as chrono

        self.chrono = chrono
        self.cfg = cfg
        self.model = model
        self.tshape = tshape
        sim, mats, rob = cfg["simulation"], cfg["materials"], cfg["robot"]
        self.dt = float(sim["step_s"])
        sys_ = chrono.ChSystemNSC()
        sys_.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -float(sim["gravity_mps2"])))
        sys_.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        sys_.SetNumThreads(1, 1, 1)
        sys_.SetSolverType(getattr(chrono.ChSolver, "Type_" + sim["solver"]))
        solver = sys_.GetSolver().AsIterative()
        solver.SetMaxIterations(int(sim["solver_iterations"]))
        solver.SetTolerance(float(sim.get("solver_tolerance", 0.0)))
        if hasattr(solver, "EnableWarmStart"):
            solver.EnableWarmStart(bool(sim.get("warm_start", True)))
        sys_.SetMaxPenetrationRecoverySpeed(float(sim["max_penetration_recovery_mps"]))
        self.sys = sys_

        def material(friction):
            m = chrono.ChContactMaterialNSC()
            m.SetFriction(float(friction))
            m.SetRestitution(float(mats.get("restitution", 0.0)))
            return m
        self.mat_table = material(mats["table_friction"])
        self.mat_t = material(mats["tshape_friction"])
        self.mat_arm = material(mats["arm_friction"])
        env, mar = float(sim["collision_envelope_m"]), float(sim["collision_margin_m"])

        def collision_setup(body, family, disallow):
            cm = body.GetCollisionModel()
            cm.SetEnvelope(env)
            cm.SetSafeMargin(mar)
            cm.SetFamily(family)
            for f in disallow:
                cm.DisallowCollisionsWith(f)

        # ---- table (family 0)
        tab = cfg["table"]
        table = chrono.ChBody()
        table.SetName("table")
        table.SetTag(TAG_TABLE)
        table.SetFixed(True)
        sx, sy, sz = (float(v) for v in tab["size_m"])
        table.AddCollisionShape(chrono.ChCollisionShapeBox(self.mat_table, sx, sy, sz),
                                chrono.ChFramed(chrono.ChVector3d(float(tab["center_xy_m"][0]), float(tab["center_xy_m"][1]), -sz / 2)))
        table.EnableCollision(True)
        sys_.Add(table)
        collision_setup(table, 0, [])
        self.table = table

        # ---- arm (family 1, no self collision; the fixed base does not collide with the table)
        d = model.desc
        q_build = np.zeros(ARM_JOINTS)
        T0 = model.link_poses(q_build)
        self.links = {}
        for i, ln in enumerate(LINK_ORDER):
            L = d["links"][ln]
            b = chrono.ChBodyAuxRef()
            b.SetName(ln)
            b.SetTag(i)
            b.SetMass(float(L["mass"]))
            b.SetFrameCOMToRef(chrono.ChFramed(_V(chrono, L["com_link_frame"]), _Q(chrono, L["inertia_principal_axes_quat_wxyz"])))
            b.SetInertiaXX(_V(chrono, L["inertia_principal_diag"]))
            b.SetFrameRefToAbs(_frame(chrono, T0[ln]))
            for sh in link_collision_shapes(ln, rob):
                if sh["kind"] == "box":
                    full = 2 * sh["half"]
                    b.AddCollisionShape(chrono.ChCollisionShapeBox(self.mat_arm, *(float(v) for v in full)),
                                        chrono.ChFramed(_V(chrono, sh["center"]), _Q(chrono, matrix_to_quat(sh["rot"]))))
                    continue
                pts = chrono.vector_ChVector3d()
                for p in sh["points"]:
                    pts.append(_V(chrono, p))
                b.AddCollisionShape(chrono.ChCollisionShapeConvexHull(self.mat_arm, pts), chrono.ChFramed())
            b.EnableCollision(True)
            if ln == "base":
                b.SetFixed(True)
            sys_.Add(b)
            collision_setup(b, 1, [1, 0] if ln == "base" else [1])
            self.links[ln] = b
        self.arm_bodies = [self.links[ln] for ln in LINK_ORDER]

        # five torque motors (motor angle == twin joint angle: built at q = 0, child/parent/T0[parent] @ F0)
        self.motors, self.torque_funs = [], []
        for j in range(ARM_JOINTS):
            parent, child = self.links[model.parent[j]], self.links[model.child[j]]
            Fabs = T0[model.parent[j]] @ model.F0[j]
            mot = chrono.ChLinkMotorRotationTorque()
            fun = chrono.ChFunctionConst(0.0)
            mot.Initialize(child, parent, _frame(chrono, Fabs))
            mot.SetTorqueFunction(fun)
            mot.SetName(d["joints"][model.joints[j]]["name"])
            sys_.Add(mot)
            self.motors.append(mot)
            self.torque_funs.append(fun)
        # jaw locked to the gripper at the lock angle (the jaw body was placed at that angle by link_poses)
        jaw_j = model.joints.index("Jaw")
        lock = chrono.ChLinkLockLock()
        lock.Initialize(self.links["jaw"], self.links["gripper"], _frame(chrono, T0["gripper"] @ model.F0[jaw_j] @ rot_z(model.jaw_lock)))
        lock.SetName("jaw_lock")
        sys_.Add(lock)
        self.jaw_lock = lock

        # ---- T-shape (family 2, the two boxes do not collide with each other)
        ts = tshape
        self.bar = self._box("tshape_bar", TAG_BAR, ts.m_bar, ts.bar_inertia, (ts.bar_l, ts.bar_w, ts.height))
        self.stem = self._box("tshape_stem", TAG_STEM, ts.m_stem, ts.stem_inertia, (ts.stem_w, ts.stem_l, ts.height))
        for b in (self.bar, self.stem):
            sys_.Add(b)
            collision_setup(b, 2, [2])
        self.place_t(t_pose)
        tlock = chrono.ChLinkLockLock()
        com = np.r_[t_pose[0], t_pose[1], ts.height / 2]
        tlock.Initialize(self.stem, self.bar, chrono.ChFramed(_V(chrono, com), _Q(chrono, [math.cos(t_pose[2] / 2), 0, 0, math.sin(t_pose[2] / 2)])))
        tlock.SetName("tshape_lock")
        sys_.Add(tlock)
        self.t_lock = tlock
        self.stem_in_bar = ts.stem_center - ts.bar_center

        self.kp = np.array(cfg["controller"]["kp"], dtype=float)
        self.kd = np.array(cfg["controller"]["kd"], dtype=float)
        self.effort = float(cfg["controller"]["effort_nm"])
        self.cmd_limits = model.limits.copy()
        self.tau = np.zeros(ARM_JOINTS)
        self.reporter = _contact_reporter(chrono, float(sim.get("contact_touch_distance_m", -1.0)))
        self.contact_bodies = [b for ln, b in zip(LINK_ORDER, self.arm_bodies)]
        self.arm_t_near = False          # an arm-T contact was inside the envelope at the last report
        self.clear = clearance if clearance is not None else ArmTClearance(model, rob, tshape)
        self.t_half = {"bar": np.array([ts.bar_l, ts.bar_w, ts.height]) / 2,
                       "stem": np.array([ts.stem_w, ts.stem_l, ts.height]) / 2}
        self.set_arm_q(q_init5)

    def _box(self, name, tag, mass, inertia, size):
        chrono = self.chrono
        b = chrono.ChBody()
        b.SetName(name)
        b.SetTag(tag)
        b.SetMass(float(mass))
        b.SetInertiaXX(_V(chrono, inertia))
        b.AddCollisionShape(chrono.ChCollisionShapeBox(self.mat_t, *(float(s) for s in size)), chrono.ChFramed())
        b.EnableCollision(True)
        return b

    def add_visuals(self, goal_pose=None):
        """Visual shapes for rendering only (they do not change the dynamics): the table, every collision shape of the
        arm links (convex hulls as triangle meshes, fingertip boxes as boxes), the two T boxes and, if goal_pose
        (x, y, yaw) is given, a green 1 mm footprint of the goal T pose on the table."""
        from scipy.spatial import ConvexHull
        chrono = self.chrono
        col = lambda r, g, b: chrono.ChColor(float(r), float(g), float(b))
        tab = self.cfg["table"]
        sx, sy, sz = (float(v) for v in tab["size_m"])
        vt = chrono.ChVisualShapeBox(sx, sy, sz)
        vt.SetColor(col(0.45, 0.44, 0.42))
        self.table.AddVisualShape(vt, chrono.ChFramed(chrono.ChVector3d(float(tab["center_xy_m"][0]), float(tab["center_xy_m"][1]), -sz / 2)))
        rob = self.cfg["robot"]
        link_col = {"base": (0.25, 0.25, 0.28), "gripper": (0.95, 0.75, 0.15), "jaw": (0.95, 0.75, 0.15)}
        for ln, b in self.links.items():
            c = col(*link_col.get(ln, (0.93, 0.47, 0.12)))
            for sh in link_collision_shapes(ln, rob):
                if sh["kind"] == "box":
                    vs = chrono.ChVisualShapeBox(*(float(v) for v in 2 * sh["half"]))
                    vs.SetColor(c)
                    b.AddVisualShape(vs, chrono.ChFramed(_V(chrono, sh["center"]), _Q(chrono, matrix_to_quat(sh["rot"]))))
                    continue
                pts = np.asarray(sh["points"], float)
                hull = ConvexHull(pts)
                mesh = chrono.ChTriangleMeshConnected()
                for f, eq in zip(hull.simplices, hull.equations):
                    a, p1, p2 = pts[f]
                    if np.dot(np.cross(p1 - a, p2 - a), eq[:3]) < 0:          # outward winding
                        p1, p2 = p2, p1
                    mesh.AddTriangle(_V(chrono, a), _V(chrono, p1), _V(chrono, p2))
                vs = chrono.ChVisualShapeTriangleMesh()
                vs.SetMesh(mesh)
                vs.SetColor(c)
                b.AddVisualShape(vs, chrono.ChFramed())
        ts = self.tshape
        for b, size in ((self.bar, (ts.bar_l, ts.bar_w, ts.height)), (self.stem, (ts.stem_w, ts.stem_l, ts.height))):
            vs = chrono.ChVisualShapeBox(*(float(s) for s in size))
            vs.SetColor(col(0.20, 0.50, 1.00))
            b.AddVisualShape(vs, chrono.ChFramed())
        if goal_pose is not None:
            g = chrono.ChBody()
            g.SetName("goal_footprint")
            g.SetFixed(True)
            g.EnableCollision(False)
            qg = _Q(chrono, [math.cos(goal_pose[2] / 2), 0.0, 0.0, math.sin(goal_pose[2] / 2)])
            for c_, size in ((ts.bar_center, (ts.bar_l, ts.bar_w)), (ts.stem_center, (ts.stem_w, ts.stem_l))):
                w = ts.to_world(goal_pose, c_)
                vs = chrono.ChVisualShapeBox(float(size[0]), float(size[1]), 0.001)
                vs.SetColor(col(0.10, 0.90, 0.30))
                g.AddVisualShape(vs, chrono.ChFramed(chrono.ChVector3d(float(w[0]), float(w[1]), 0.0006), qg))
            self.sys.Add(g)

    # ------------------------------------------------------------------ state io
    def place_t(self, pose):
        chrono, ts = self.chrono, self.tshape
        q = [math.cos(pose[2] / 2), 0.0, 0.0, math.sin(pose[2] / 2)]
        for b, c in ((self.bar, ts.bar_center), (self.stem, ts.stem_center)):
            w = ts.to_world(pose, c)
            b.SetPos(chrono.ChVector3d(float(w[0]), float(w[1]), ts.height / 2))
            b.SetRot(_Q(chrono, q))
            b.SetPosDt(chrono.ChVector3d(0, 0, 0))
            b.SetAngVelParent(chrono.ChVector3d(0, 0, 0))

    def set_arm_q(self, q5):
        """Teleport the arm links to joint angles q5 at rest (legacy env.set_robot_joints)."""
        T = self.model.link_poses(q5)
        chrono = self.chrono
        for ln in LINK_ORDER:
            b = self.links[ln]
            b.SetFrameRefToAbs(_frame(chrono, T[ln]))
            b.SetPosDt(chrono.ChVector3d(0, 0, 0))
            b.SetAngVelParent(chrono.ChVector3d(0, 0, 0))
        self.sys.Update(True)

    def joint_q(self) -> np.ndarray:
        return np.array([m.GetMotorAngle() for m in self.motors])

    def joint_qd(self) -> np.ndarray:
        return np.array([m.GetMotorAngleDt() for m in self.motors])

    def arm_state(self) -> np.ndarray:
        g = self.links["gripper"]
        F = g.GetFrameRefToAbs()
        p = F.TransformPointLocalToParent(_V(self.chrono, self.model.tcp_local))
        c, v, w = g.GetPos(), g.GetPosDt(), g.GetAngVelParent()
        r = np.array([p.x - c.x, p.y - c.y, p.z - c.z])
        wv = np.array([w.x, w.y, w.z])
        vt = np.array([v.x, v.y, v.z]) + np.cross(wv, r)
        return np.r_[self.joint_q(), self.joint_qd(), p.x, p.y, p.z, vt, wv]

    def t_state(self) -> np.ndarray:
        mb, ms = self.tshape.m_bar, self.tshape.m_stem
        M = mb + ms
        pb, ps = self.bar.GetPos(), self.stem.GetPos()
        vb, vs = self.bar.GetPosDt(), self.stem.GetPosDt()
        wb, ws = self.bar.GetAngVelParent(), self.stem.GetAngVelParent()
        q = self.bar.GetRot()
        com = (mb * np.array([pb.x, pb.y, pb.z]) + ms * np.array([ps.x, ps.y, ps.z])) / M
        vel = (mb * np.array([vb.x, vb.y, vb.z]) + ms * np.array([vs.x, vs.y, vs.z])) / M
        om = (mb * np.array([wb.x, wb.y, wb.z]) + ms * np.array([ws.x, ws.y, ws.z])) / M
        return np.r_[com, q.e0, q.e1, q.e2, q.e3, vel, om]

    def t_planar_pose(self) -> np.ndarray:
        s = self.t_state()
        w, x, y, z = s[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([s[0], s[1], yaw])

    def link_frames(self) -> dict:
        out = {}
        for ln in LINK_ORDER:
            F = self.links[ln].GetFrameRefToAbs()
            T = np.eye(4)
            T[:3, :3] = _rotmat(F.GetRot())
            p = F.GetPos()
            T[:3, 3] = (p.x, p.y, p.z)
            out[ln] = T
        return out

    def t_box_list(self) -> list:
        out = []
        for b, half in ((self.bar, self.t_half["bar"]), (self.stem, self.t_half["stem"])):
            p = b.GetPos()
            out.append((np.array([p.x, p.y, p.z]), _rotmat(b.GetRot()), half))
        return out

    def gaps(self) -> tuple[float, float]:
        """(finger_gap, link_gap) in m from the simulated frames (see the module docstring)."""
        frames = self.link_frames()
        boxes = self.t_box_list()
        fg = self.clear.finger_gap(frames, boxes, cap=0.02)
        lg, _ = self.clear.hull_gap(frames, boxes, links=OTHER_LINKS + FINGER_LINKS, cap=0.01)
        return fg, lg

    def lock_errors(self) -> tuple[float, float]:
        """T-lock violation: stem centre offset from its nominal place in the bar frame (m) and the relative
        rotation angle (rad)."""
        Fb = self.bar.GetFrameCOMToAbs()
        ps = self.stem.GetPos()
        loc = Fb.TransformPointParentToLocal(ps)
        e = math.hypot(math.hypot(loc.x - self.stem_in_bar[0], loc.y - self.stem_in_bar[1]), loc.z)
        qb, qs = self.bar.GetRot(), self.stem.GetRot()
        dot = abs(qb.e0 * qs.e0 + qb.e1 * qs.e1 + qb.e2 * qs.e2 + qb.e3 * qs.e3)
        return e, 2 * math.acos(min(1.0, dot))

    # ------------------------------------------------------------------ stepping
    def apply_pd(self, q_cmd):
        q = self.joint_q()
        qd = self.joint_qd()
        qc = np.clip(q_cmd, self.cmd_limits[:, 0], self.cmd_limits[:, 1])
        tau = np.clip(self.kp * (qc - q) - self.kd * qd, -self.effort, self.effort)
        for f, t in zip(self.torque_funs, tau):
            f.SetConstant(float(t))
        self.tau = tau

    def step(self, q_cmd) -> tuple[np.ndarray, bool, bool, float]:
        """One physics step with the explicit PD law. Returns (arm-T flags per link, T-table, arm-table,
        arm-T normal force)."""
        self.apply_pd(q_cmd)
        self.sys.DoStepDynamics(self.dt)
        arm_touch = self.arm_t_near
        if not arm_touch:
            for b in self.contact_bodies:
                f = b.GetContactForce()
                if f.x != 0.0 or f.y != 0.0 or f.z != 0.0:
                    arm_touch = True
                    break
        if arm_touch:
            r = self.reporter
            r.reset()
            self.sys.GetContactContainer().ReportAllContacts(r)
            self.arm_t_near = r.arm_t_any
            return r.arm_t.copy(), r.t_table, r.arm_table, r.arm_t_force
        fb, fs = self.bar.GetContactForce(), self.stem.GetContactForce()
        t_table = (fb.x != 0.0 or fb.y != 0.0 or fb.z != 0.0) or (fs.x != 0.0 or fs.y != 0.0 or fs.z != 0.0)
        return np.zeros(len(LINK_ORDER), dtype=bool), t_table, False, 0.0


def run_episode(cfg: dict, model: ArmModel, tshape: TShape, policy, *, q_start, t_pose=None, duration_s=None,
                verbose=False, step_hook=None, clearance=None) -> tuple[dict, dict]:
    """Simulate one episode. `policy.command(k, time_s, obs)` returns q_cmd (5) for control step k; obs has
    'q', 'qd', 'arm' (19), 't_pose' (x, y, yaw) and 't' (13). Returns (arrays, meta)."""
    sim = cfg["simulation"]
    dt = float(sim["step_s"])
    rec = float(sim["record_step_s"])
    ctrl = float(sim["control_step_s"])
    duration = float(duration_s if duration_s is not None else sim["duration_s"])
    spr = int(round(rec / dt))                 # physics steps per record
    rpc = int(round(ctrl / rec))               # records per control step
    if abs(spr * dt - rec) > 1e-12 or abs(rpc * rec - ctrl) > 1e-12:
        raise ValueError("record step must be a multiple of the physics step and control step of the record step")
    n_rec = int(round(duration / rec)) + 1
    n_ctrl = (n_rec - 1) // rpc
    t_pose = np.asarray(t_pose if t_pose is not None else cfg["tshape"]["start_pose"], dtype=float)
    wall0 = time.perf_counter()
    scene = PushScene(cfg, model, tshape, q_start, t_pose, clearance=clearance)
    build_s = time.perf_counter() - wall0
    lo_lim, hi_lim = scene.cmd_limits[:, 0], scene.cmd_limits[:, 1]

    def command(k):
        # the stored action is the command the PD law applies (clipped to the joint limits)
        return np.clip(np.asarray(policy.command(k, k * ctrl, o), dtype=float), lo_lim, hi_lim)

    def obs():
        a = scene.arm_state()
        t = scene.t_state()
        return {"q": a[:5], "qd": a[5:10], "arm": a, "t": t, "t_pose": scene.t_planar_pose()}

    # settle (not recorded): hold the first command
    o = obs()
    q_cmd = command(0)
    settle_steps = int(round(float(sim["settle_s"]) / dt))
    settle_contact = False
    for _ in range(settle_steps):
        flags, _, _, _ = scene.step(q_cmd)
        settle_contact |= bool(flags.any())
    arm = np.zeros((n_rec, 19))
    tsh = np.zeros((n_rec, 13))
    action = np.zeros((n_rec, 5))
    tau = np.zeros((n_rec, 5))
    contacts_link = np.zeros((n_rec - 1, len(LINK_ORDER)), dtype=np.uint8)
    t_table = np.zeros(n_rec - 1, dtype=np.uint8)
    arm_table = np.zeros(n_rec - 1, dtype=np.uint8)
    force = np.zeros(n_rec - 1)
    lock_err = np.zeros((n_rec, 2))
    finger_gap = np.zeros(n_rec)
    link_gap = np.zeros(n_rec)
    o = obs()
    arm[0], tsh[0] = o["arm"], o["t"]
    lock_err[0] = scene.lock_errors()
    finger_gap[0], link_gap[0] = scene.gaps()
    gap_s = 0.0
    sim0 = time.perf_counter()
    r = 0
    for k in range(n_ctrl):
        if k > 0:
            q_cmd = command(k)
        for _ in range(rpc):
            action[r] = q_cmd
            fl = np.zeros(len(LINK_ORDER), dtype=bool)
            tt = at = False
            fmax = 0.0
            for _ in range(spr):
                f, a_tab, a_arm, fn = scene.step(q_cmd)
                if step_hook is not None:
                    step_hook(scene, r, f, fn)
                fl |= f
                tt |= a_tab
                at |= a_arm
                fmax = max(fmax, fn)
            contacts_link[r] = fl
            t_table[r] = tt
            arm_table[r] = at
            force[r] = fmax
            tau[r] = scene.tau
            r += 1
            o = obs()
            arm[r], tsh[r] = o["arm"], o["t"]
            lock_err[r] = scene.lock_errors()
            g0 = time.perf_counter()
            finger_gap[r], link_gap[r] = scene.gaps()
            gap_s += time.perf_counter() - g0
    action[r] = command(n_ctrl)
    tau[r] = scene.tau
    # quaternion sign continuity, w >= 0 at the first record
    qs = tsh[:, 3:7]
    if qs[0, 0] < 0:
        qs[0] *= -1
    for i in range(1, n_rec):
        if qs[i] @ qs[i - 1] < 0:
            qs[i] *= -1
    arrays = {"arm": arm, "tshape": tsh, "action": action, "contacts_link": contacts_link,
              "contacts": contacts_link.any(axis=1, keepdims=True).astype(np.uint8), "t_table": t_table,
              "arm_table": arm_table, "arm_t_force": force, "tau": tau, "lock_err": lock_err,
              "tip_pen": np.maximum(-finger_gap, 0.0), "finger_gap": finger_gap, "link_gap": link_gap}
    meta = {"build_s": build_s, "sim_s": time.perf_counter() - sim0, "wall_s": time.perf_counter() - wall0,
            "records": n_rec, "settle_arm_t_contact": settle_contact,
            "max_t_lock_err_m": float(lock_err[:, 0].max()), "max_t_lock_err_rad": float(lock_err[:, 1].max()),
            "max_tip_penetration_m": float(max(-finger_gap.min(), 0.0)), "min_finger_gap_m": float(finger_gap.min()),
            "min_link_gap_m": float(link_gap.min()), "gap_check_s": gap_s}
    return arrays, meta
