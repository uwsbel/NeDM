"""The "no learned dynamics" control world of the SO101 + cube study (spec_v2 B3; review_spec_claims item 1).

Same API as ``nrd_sim.GraspNrdSim``: ``reset(idx, start)``, ``step(joint_targets [B, 6]) -> WorldState``,
``set_cube(idx, pos, quat, v, w)``, ``observe()``. Batched torch, float64, one device. What it models:

- Arm and jaw: each joint follows its target with a discrete servo response fitted to the grasp data on the 20 ms
  grid (``ServoModel``, an ARX model per joint: next step change = coef . [u - q, last step change, u - last u,
  the step change before, 1, sin / cos of the summed pitch angles q1, q1 + q2, q1 + q2 + q3 (gravity)]). Joint
  targets are clipped to the collector's applied range (``kin.TLO``..``THI``) and rounded to float32, like the
  MuJoCo world.
- Cube held: when the jaw closes through the contact angle (``contact_jaw``, the 30 mm cube between the jaws) while
  the cube centre is inside the task's grasp box (``task.TaskConfig`` box at the close start), the cube attaches
  rigidly to the gripper: its centre snaps to ``grip_x`` along the closing axis (the data's firm-grip offset), keeps its
  jaw-width and height offsets, and its axes align with the gripper (yaw snapped to the nearest face); the jaw stops at
  ``grip_jaw`` (the measured firm-grip angle) while its command keeps closing (a stopped jaw has no servo velocity
  history).
- Cube free: at rest on the table. A release (the jaw opens past ``release_jaw``) lets the cube fall straight down
  under gravity, flat, with its yaw (no tumbling, no sliding) until it rests on the table. A cube that is not held
  ignores the arm (pushes and pass-throughs are left to the task's validity rules).
- WorldState: q, qd (20 ms step averages), cube pose / step-average v, w (world), contact = (held, held, cube on the
  table), arm_table = lowest arm collision point < 0 (the task geometry), invalid = non-finite state.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from nedm.so101_policy.kin import SO101Kin, mat_quat, quat_mat, rotz, yaw_of
from nedm.so101_policy.task import CUBE_HALF, CUBE_REST_Z, TaskConfig, TaskGeometry, wrap90
from nedm.so101_policy.world import CONTROL_DT, WorldState

N_FEAT = 11
# servo coefficients [6, N_FEAT] (rows = joints, columns = servo_features): least squares on one-step changes, then
# refined on 1 s open-loop rollouts (fit_rollout); 4,000 task-family episodes of production_plus split 0 (seed 0),
# ``calibrate_rules.py servo`` -> ~/so101_policy/runs/task/servo_v2.json. Validation (1,803 val_only task episodes):
# TCP error after 1 s open loop p50 / p95 / p99 0.27 / 1.08 / 2.58 mm (see build2_task.md).
DEFAULT_COEF = (
    (2.416768e-02, 2.034120e-01, 9.562960e-01, -1.789071e-01, 1.488592e-05, 7.914053e-07,
     5.871995e-06, -6.036553e-06, -1.032760e-05, -1.013414e-05, -1.344410e-05),
    (3.120069e-02, 6.681892e-01, 6.522404e-01, -3.506166e-01, -1.026469e-03, -6.359035e-05,
     2.528456e-04, -1.062476e-04, 5.003481e-04, 3.087500e-04, 9.851031e-05),
    (3.039624e-01, 4.114891e-01, 5.428754e-01, -2.085317e-01, 1.485099e-03, -7.676729e-05,
     3.823299e-04, -3.652555e-04, 9.366593e-04, -2.502050e-03, -1.107961e-03),
    (2.557571e-01, 2.674111e-01, 6.435725e-01, -1.313854e-01, 5.350935e-04, 1.888535e-05,
     -1.259437e-04, 1.503874e-04, -1.135518e-03, 6.648242e-04, 2.111747e-04),
    (3.518271e-01, 4.220943e-01, 4.788519e-01, -1.904455e-01, 1.812432e-05, -8.068272e-06,
     -9.833315e-07, -1.395998e-05, -2.850098e-05, -8.669269e-06, 1.260451e-05),
    (1.176075e-01, 8.312249e-01, 3.257187e-01, -2.919780e-01, -3.535046e-04, 1.985288e-04,
     -3.321374e-05, 1.761259e-05, 1.974914e-03, -1.661937e-03, 5.823905e-04),
)


def servo_features(q, d1, d2, u, u_prev):
    """ARX features [B, 6, N_FEAT] per joint: q, u, u_prev [B, 6] (rad), d1 / d2 = the last / second-last step change
    of q [B, 6] (rad per step)."""
    p1 = q[:, 1]
    p2 = p1 + q[:, 2]
    p3 = p2 + q[:, 3]
    grav = torch.stack((torch.sin(p1), torch.cos(p1), torch.sin(p2), torch.cos(p2), torch.sin(p3), torch.cos(p3)), -1)
    base = torch.stack((u - q, d1, u - u_prev, d2, torch.ones_like(q)), -1)          # [B, 6, 5]
    return torch.cat((base, grav[:, None, :].expand(-1, q.shape[1], -1)), -1)


class ServoModel:
    """next q = q + coef[j] . features (``servo_features``). coef [6, N_FEAT]."""

    def __init__(self, coef, device="cpu", dtype=torch.float64):
        self.coef = torch.as_tensor(coef, dtype=dtype, device=device)

    def step(self, q, d1, d2, u, u_prev):
        return q + (servo_features(q, d1, d2, u, u_prev) * self.coef).sum(-1)

    @staticmethod
    def fit(q, u, valid, jaw_free=None, ridge=1e-9):
        """Least squares on recorded grids: q [N, J, 6] measured angles, u [N, J, 6] targets held from grid point k to
        k + 1, valid [N, J] bool (grid point k + 1 recorded), jaw_free [N, J] bool (the jaw may be fitted at k; None =
        everywhere). Samples k >= 2 (two past step changes). -> coef [6, N_FEAT] float64."""
        q, u = q.to(torch.float64), u.to(torch.float64)
        d = q[:, 1:] - q[:, :-1]                                                    # d[:, k] = q[k + 1] - q[k]
        k = slice(2, q.shape[1] - 1)
        F = servo_features(q[:, k].reshape(-1, 6), d[:, 1:-1].reshape(-1, 6), d[:, :-2].reshape(-1, 6),
                           u[:, k].reshape(-1, 6), u[:, 1:-2].reshape(-1, 6))
        Y = d[:, 2:].reshape(-1, 6)
        ok = (valid[:, 3:] & valid[:, 2:-1]).reshape(-1)
        coef = torch.zeros(6, N_FEAT, dtype=torch.float64)
        for j in range(6):
            m = ok & (jaw_free[:, 2:-1].reshape(-1) if (j == 5 and jaw_free is not None) else True)
            X, y = F[m, j], Y[m, j]
            A = X.T @ X + ridge * torch.eye(N_FEAT, dtype=torch.float64) * len(X)
            coef[j] = torch.linalg.solve(A, X.T @ y)
        return coef


def fit_rollout(coef0, q, u, valid, jaw_free=None, horizon=50, every=10, iters=200, device="cpu", max_windows=60000,
                seed=0):
    """Refine servo coefficients on open-loop rollouts (simulation-error fit): from recorded grid point s (s >= 2)
    the model runs ``horizon`` steps on the recorded targets; loss = mean squared joint error over all steps (the jaw
    only in windows where it is free throughout). LBFGS from coef0. -> coef [6, N_FEAT] float64 (CPU)."""
    f = dict(dtype=torch.float64, device=device)
    q, u, valid = q.to(**f), u.to(**f), valid.to(device)
    free = torch.ones_like(valid) if jaw_free is None else jaw_free.to(device)
    J = q.shape[1]
    starts = torch.arange(2, J - horizon - 1, every, device=device)
    n_i, n_s = torch.meshgrid(torch.arange(q.shape[0], device=device), starts, indexing="ij")
    n_i, n_s = n_i.reshape(-1), n_s.reshape(-1)
    ok = valid[n_i, n_s + horizon]
    n_i, n_s = n_i[ok], n_s[ok]
    if len(n_i) > max_windows:
        pick = torch.randperm(len(n_i), generator=torch.Generator().manual_seed(seed))[:max_windows].to(device)
        n_i, n_s = n_i[pick], n_s[pick]
    steps = torch.arange(horizon + 1, device=device)
    Q = q[n_i[:, None], n_s[:, None] + steps]                                       # [W, H + 1, 6]
    U = u[n_i[:, None], n_s[:, None] - 1 + steps]                                   # U[:, t] = target before step t
    jaw_w = free[n_i[:, None], n_s[:, None] + steps[:-1]].all(1).to(q.dtype)        # [W]
    d1_0 = q[n_i, n_s] - q[n_i, n_s - 1]
    d2_0 = q[n_i, n_s - 1] - q[n_i, n_s - 2]
    scale = (Q[:, 1:] - Q[:, :-1]).std(dim=(0, 1)).clamp_min(1e-6)                  # per-joint normalisation
    coef = torch.nn.Parameter(torch.as_tensor(coef0, **f).clone())
    opt = torch.optim.LBFGS([coef], lr=0.5, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def loss_fn():
        opt.zero_grad()
        qq, d1, d2 = Q[:, 0], d1_0, d2_0
        loss = 0.0
        for t in range(horizon):
            qn = qq + (servo_features(qq, d1, d2, U[:, t + 1], U[:, t]) * coef).sum(-1)
            d2, d1, qq = d1, qn - qq, qn
            e = ((qq - Q[:, t + 1]) / scale) ** 2
            e = torch.cat((e[:, :5], e[:, 5:] * jaw_w[:, None]), -1)
            loss = loss + e.mean()
        loss = loss / horizon
        loss.backward()
        return loss

    opt.step(loss_fn)
    return coef.detach().cpu()


@dataclass
class KinWorldConfig:
    contact_jaw: float = 0.19        # rad, jaw angle at which the jaws touch a 30 mm cube (collector jaw_profile)
    grip_jaw: float = 0.167          # rad, firm-grip jaw angle (data median while gripped)
    grip_x: float = -0.0043          # m, cube centre along the closing axis in a firm grip (data median)
    release_jaw: float = 0.20        # rad, the jaw opens past this: the cube is released
    gravity: float = 9.81


class KinWorld:
    """B kinematic worlds on one device (module docstring). servo: ServoModel or coef [6, N_FEAT] (default
    ``DEFAULT_COEF``); task_cfg: the grasp box used for the capture (default ``TaskConfig()``)."""

    def __init__(self, num_envs, device="cpu", servo=None, cfg: KinWorldConfig | None = None,
                 task_cfg: TaskConfig | None = None, kin: SO101Kin | None = None, geom: TaskGeometry | None = None):
        self.B, self.dev = int(num_envs), torch.device(device)
        self.cfg, self.tcfg = cfg or KinWorldConfig(), task_cfg or TaskConfig()
        self.kin = kin or SO101Kin(self.dev)
        self.geom = geom if geom is not None else TaskGeometry.from_mujoco(self.dev)
        coef = servo.coef if isinstance(servo, ServoModel) else (DEFAULT_COEF if servo is None else servo)
        if coef is None:
            raise ValueError("no servo coefficients: pass servo= (ServoModel.fit) or set kin_world.DEFAULT_COEF")
        self.servo = ServoModel(coef, self.dev)
        f = dict(dtype=torch.float64, device=self.dev)
        B = self.B
        self.q = torch.zeros(B, 6, **f)
        self.d1 = torch.zeros(B, 6, **f)
        self.d2 = torch.zeros(B, 6, **f)
        self.u_prev = torch.zeros(B, 6, **f)
        self.qd = torch.zeros(B, 6, **f)
        self.cube_pos = torch.zeros(B, 3, **f)
        self.cube_rot = torch.eye(3, **f).repeat(B, 1, 1)
        self.cube_v = torch.zeros(B, 3, **f)
        self.cube_w = torch.zeros(B, 3, **f)
        self.fall_v = torch.zeros(B, **f)                       # m/s downwards while falling
        self.held = torch.zeros(B, dtype=torch.bool, device=self.dev)
        self.rel_pos = torch.zeros(B, 3, **f)                   # cube centre in the grasp frame while held
        self.rel_rot = torch.eye(3, **f).repeat(B, 1, 1)        # cube rotation in the gripper frame while held
        self.quat_prev = torch.zeros(B, 4, **f)
        self.quat_prev[:, 0] = 1

    # ---- resets ------------------------------------------------------------------------------------------------
    def _idx(self, idx):
        if idx is None:
            return torch.arange(self.B, device=self.dev)
        idx = torch.as_tensor(idx, device=self.dev)
        return torch.nonzero(idx).flatten() if idx.dtype == torch.bool else idx.long().reshape(-1)

    @staticmethod
    def _get(start, name):
        return start.get(name) if isinstance(start, dict) else getattr(start, name, None)

    @torch.no_grad()
    def reset(self, idx, start) -> WorldState:
        """Worlds idx start from a WorldState-like object / dict: q [n, 6], cube_pos [n, 3], cube_quat [n, 4] (w x y z),
        optional qd [n, 6] (the servo's last step change = qd dt; default 0). The cube is held if the jaw is at most
        2 deg above the contact angle and the cube is inside the held box (kept as it is), else free (on the table, or
        falling if above it); the last target = the start angles."""
        idx = self._idx(idx)
        f = dict(dtype=torch.float64, device=self.dev)
        q = torch.as_tensor(self._get(start, "q"), **f).reshape(-1, 6)
        qd = self._get(start, "qd")
        qd = torch.zeros_like(q) if qd is None else torch.as_tensor(qd, **f).reshape(-1, 6)
        self.q[idx], self.u_prev[idx], self.qd[idx] = q, q, qd
        self.d1[idx], self.d2[idx] = qd * CONTROL_DT, qd * CONTROL_DT
        quat = torch.as_tensor(self._get(start, "cube_quat"), **f).reshape(-1, 4)
        quat = quat / quat.norm(dim=-1, keepdim=True)
        self.cube_pos[idx] = torch.as_tensor(self._get(start, "cube_pos"), **f).reshape(-1, 3)
        self.cube_rot[idx] = quat_mat(quat)
        self.quat_prev[idx] = quat
        self.cube_v[idx], self.cube_w[idx], self.fall_v[idx] = 0.0, 0.0, 0.0
        # a start inside a grip (jaw at most the contact angle, cube in the held box) starts held, as it is
        tcp, rg = self.kin.fk(q)
        rel = (rg.transpose(-1, -2) @ (self.cube_pos[idx] - tcp)[..., None])[..., 0]
        t = self.tcfg
        held = ((q[:, 5] <= self.cfg.contact_jaw + 0.02) & (rel[:, 0].abs() <= t.held_x) & (rel[:, 1].abs() <= t.held_y)
                & (rel[:, 2] >= t.held_z[0]) & (rel[:, 2] <= t.held_z[1]))
        self.held[idx] = held
        self.rel_pos[idx] = rel
        self.rel_rot[idx] = rg.transpose(-1, -2) @ self.cube_rot[idx]
        return self.observe()

    @torch.no_grad()
    def set_cube(self, idx, pos, quat, v=None, w=None) -> WorldState:
        """Disturbance: the cube of worlds idx moves to (pos, quat) (released if it was held), at rest (v, w ignored:
        the kinematic cube rests or falls); the step-average rates restart at 0. Returns the WorldState of all worlds
        (as the other worlds do; review F4)."""
        idx = self._idx(idx)
        f = dict(dtype=torch.float64, device=self.dev)
        quat = torch.as_tensor(quat, **f).reshape(-1, 4)
        quat = quat / quat.norm(dim=-1, keepdim=True)
        self.cube_pos[idx] = torch.as_tensor(pos, **f).reshape(-1, 3)
        self.cube_rot[idx] = quat_mat(quat)
        self.quat_prev[idx] = quat
        self.cube_v[idx], self.cube_w[idx], self.fall_v[idx] = 0.0, 0.0, 0.0
        self.held[idx] = False
        return self.observe()

    # ---- one step ----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, joint_targets) -> WorldState:
        c = self.cfg
        # the applied target: float32(clip(target, TLO, THI)), as MuJoCo applies it (review F3)
        u = torch.minimum(torch.maximum(joint_targets.to(self.q), self.kin.tlo), self.kin.thi)
        u = u.to(torch.float32).to(self.q)
        q_new = self.servo.step(self.q, self.d1, self.d2, u, self.u_prev)
        jaw_free = q_new[:, 5]
        # capture: the jaw closes through the contact angle with the cube inside the grasp box
        tcp, rg = self.kin.fk(q_new)
        rel = (rg.transpose(-1, -2) @ (self.cube_pos - tcp)[..., None])[..., 0]
        t = self.tcfg
        in_box = ((rel[:, 0].abs() <= t.box_x) & (rel[:, 1].abs() <= t.box_y) & (rel[:, 2] >= t.box_z[0])
                  & (rel[:, 2] <= t.box_z[1]))
        grab = ~self.held & in_box & (jaw_free <= c.contact_jaw) & (self.q[:, 5] > c.contact_jaw - 0.02)
        release = self.held & (jaw_free > c.release_jaw)
        # new holds: snap x and the cube axes to the gripper (yaw to the nearest face)
        gy = yaw_of(rg)
        cy = yaw_of(self.cube_rot)
        snap = rotz(gy + wrap90(cy - gy))                                               # world rotation, flat
        snap_rel = rg.transpose(-1, -2) @ snap
        new_rel = torch.stack((torch.full_like(rel[:, 0], c.grip_x), rel[:, 1], rel[:, 2]), -1)
        self.rel_pos = torch.where(grab[:, None], new_rel, self.rel_pos)
        self.rel_rot = torch.where(grab[:, None, None], snap_rel, self.rel_rot)
        held = (self.held | grab) & ~release
        # the jaw stops on a held cube
        clamped = held & (jaw_free < c.grip_jaw)
        jaw = torch.where(clamped, torch.full_like(jaw_free, c.grip_jaw), jaw_free)
        q_new = torch.cat((q_new[:, :5], jaw[:, None]), -1)
        tcp, rg = self.kin.fk(q_new)
        pos_prev = self.cube_pos
        pos_h = tcp + (rg @ self.rel_pos[..., None])[..., 0]
        rot_h = rg @ self.rel_rot
        # free cube: fall straight down (flat, keep yaw) or rest
        flat = rotz(yaw_of(self.cube_rot))
        rest_z = torch.full_like(self.cube_pos[:, 2], CUBE_REST_Z)
        airborne = self.cube_pos[:, 2] > CUBE_REST_Z + 1e-9
        fall_v = torch.where(airborne | release, self.fall_v + c.gravity * CONTROL_DT, torch.zeros_like(self.fall_v))
        z = torch.maximum(self.cube_pos[:, 2] - fall_v * CONTROL_DT, rest_z)
        fall_v = torch.where(z <= rest_z, torch.zeros_like(fall_v), fall_v)
        pos_f = torch.cat((self.cube_pos[:, :2], z[:, None]), -1)
        rot_f = torch.where((release | airborne)[:, None, None], flat, self.cube_rot)
        self.cube_pos = torch.where(held[:, None], pos_h, pos_f)
        self.cube_rot = torch.where(held[:, None, None], rot_h, rot_f)
        self.fall_v = torch.where(held, torch.zeros_like(fall_v), fall_v)
        self.held = held
        # rates (20 ms step averages, the data definition)
        quat = mat_quat(self.cube_rot)
        quat = torch.where((quat * self.quat_prev).sum(-1, keepdim=True) < 0, -quat, quat)
        self.cube_v = (self.cube_pos - pos_prev) / CONTROL_DT
        self.cube_w = _rotvec_world(quat, self.quat_prev) / CONTROL_DT
        self.quat_prev = quat
        d = q_new - self.q
        self.qd = d / CONTROL_DT
        # a jaw stopped by the cube is at rest: its servo history restarts (else the ARX predicts a rebound)
        stop = torch.zeros_like(d, dtype=torch.bool)
        stop[:, 5] = clamped
        self.d2 = torch.where(stop, torch.zeros_like(d), self.d1)
        self.d1 = torch.where(stop, torch.zeros_like(d), d)
        self.q, self.u_prev = q_new, u
        return self.observe()

    @torch.no_grad()
    def observe(self) -> WorldState:
        bpos, brot = self.kin.body_poses(self.q)
        low = self.geom.lowest(bpos, brot)
        bottom = self.cube_pos[:, 2] - CUBE_HALF * self.cube_rot[:, 2, :].abs().sum(-1)
        finite = torch.isfinite(self.q).all(-1) & torch.isfinite(self.cube_pos).all(-1)
        return WorldState(q=self.q.clone(), qd=self.qd.clone(), cube_pos=self.cube_pos.clone(),
                          cube_quat=self.quat_prev.clone(), cube_v=self.cube_v.clone(), cube_w=self.cube_w.clone(),
                          contact=torch.stack((self.held, self.held, ~self.held & (bottom < 1e-3)), -1),
                          arm_table=low < 0, invalid=~finite)


def _rotvec_world(q1, q0):
    """World-frame rotation vector of q1 conj(q0) (shorter way)."""
    w0, v0 = q0[..., :1], -q0[..., 1:]
    w1, v1 = q1[..., :1], q1[..., 1:]
    w = w1 * w0 - (v1 * v0).sum(-1, keepdim=True)
    v = w1 * v0 + w0 * v1 + torch.cross(v1, v0, dim=-1)
    s = torch.where(w < 0, -1.0, 1.0)
    w, v = w * s, v * s
    n = v.norm(dim=-1, keepdim=True)
    return torch.where(n > 1e-12, v / n.clamp_min(1e-12) * 2 * torch.atan2(n, w), 2 * v)

