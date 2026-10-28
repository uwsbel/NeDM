"""Table height / tilt randomisation of the camera student's JOINT INPUT (real2sim v8, 2026-10-23;
``viral.OnlineConfig.table_bias``, off by default). Ported from the studentlab round C WIP (claude/so101-studentlab
e8efd92ea, ``table_bias.py``), redone: an exact joint map of every pose instead of one fitted constant offset per
episode, and a shear of the world instead of a rigid tilt (reasons below).

Why (``~/so101_studentlab_claude/diag/release_ep05``): the robot student reads its descent / stop height from the
JOINT input and the jaw-open decision from the WRIST image. The real table top is 12-17 mm lower than the sim table
relative to the arm where the cubes are (``calib_v1.json`` scene.table_plane: z = -4.7 mm - 36.4 x + 18.4 y, m; the
real goal plate is 3.9 mm thick against 1 mm in the scenes, so the real plate top is ~8.6 mm below the sim table top).
The student stops where the sim cube would touch, the real cube stays ~13 mm above the plate, the wrist image never
shows contact and the jaw stays shut. The training worlds (the learned simulator) have the table fixed at z = 0, so
the table cannot move in the training physics; the student's JOINT READING moves instead.

Model. A per-world table plane z = c0 + c1 x + c2 y (``table.py`` convention: m, arm base frame; c0 < 0 = lower than
the sim table). For the true sim joints q the student reads

    q_obs = Phi(q):   FK_pos(q_obs) = FK_pos(q) + plane(FK_xy(q)) e_z,   FK_rot(q_obs) = FK_rot(q)

= the joints of a real arm whose gripper is at the same place relative to the world (table, cube, plate) when that
world lies on the plane: under a lower table the reading is LOWER than the true gripper, so at a grasp the student's
joints are the real joints that reach a cube that much lower (``test_table_bias.py`` checks this against an
independent top-down IK). Only the three pitch joints change (shoulder_lift, elbow_flex, wrist_flex): their axes are
parallel and horizontal and the pan axis is vertical, so they move the gripper in the arm's vertical plane and turn
it only about their common axis; a vertical shift with the same gripper rotation is therefore exact (Gauss-Newton on
those three joints, ``ITERS`` = 8 iterations). Convergence, measured on 147,001 logged REAL robot states (q_sim of
``robot_logs`` eval_* / live_*, reach up to 0.33 m, wrist_flex near 92 deg at far reach): with a flat -20 mm plane, 20
mm at x 0.25 m with slope -40 mm/m, or the calibrated plane, every row is below 2e-16 m after 8 iterations (4: up to
2 mm, 0.1 % of the rows above 0.5 mm; 6: up to 3e-8 m; the 0.3 rad step guard slows the far-reach rows); with the
default draws 0.002 % of the rows (a plane HIGHER than the sim table at far x) have no image (residual ~1 mm, true
reach limit). Pan, wrist roll and the jaw are unchanged.

A SHEAR, not a rigid tilt: horizontal positions are kept, only the height of the world under the gripper changes
(the plane is evaluated at the grasp point's xy, so the offset follows the gripper over a tilted table). A rigid
rotation of the world about some pivot (the WIP: about the shoulder) also moves the joint-implied gripper sideways by
about tilt x (height above the pivot), ~4 mm at 2 deg, a lateral bias that the images contradict; the real cube's xy
relative to the arm is whatever it is, only its height differs. A 2 deg tilt of the cube itself is ignored.

Online loop (``viral.OnlineDAgger``): per world per episode with probability ``p`` a plane is drawn
(``TableBiasConfig``: the drop at ``x_ref_m``, slopes in x and y). Everything the student sees or learns from the
joints is in the observed frame, exactly (``viral.OnlineDAgger.table_frame``, one batched map per decision): its q
input Phi(q) (+ the ``q_bias`` draw on top; history = earlier inputs), its qdot input J_Phi(q) qdot (``observe_qd``,
analytic; exactly 0 stays 0), the cmd / last-command inputs Phi(executed targets) - Phi(q); the teacher's labels
Phi(label) and the buffer's q Phi(q), so the training targets are Phi(label) - Phi(q); the student's chunk is relative
to its reading, so the applied targets are Phi^-1(Phi(q) + delta) (``to_true``). Physics, the action layer, the
teacher's observation and its executed targets stay true; images and aux targets too unless ``table_render`` (below).
One constant offset per decision instead (b = Phi(q) - q for the chunk: applied q + delta, labels label - q) is
10-14 % off on a vertical move at a 14-20 mm drop (the offset changes along the move: ~1.0-1.4 mm on 10 mm,
``test_table_bias``), so the loop maps every pose; worlds without a plane are untouched bit for bit (``shift``
returns q itself for dz = 0). Near the reach limit
(arm almost straight) a pose may have no image: a target the student aims ~15 mm "higher" in the sim world than the
sim arm can reach at that gripper rotation (real robot states: 5.3 % of them have no sim preimage under the
calibrated plane, all at grasp-point reach >= 0.27 m; the training tasks' cubes and goals lie within 0.27 m).
Training data are exact: a world whose frame-map rows (q, the image state, labels, last executed targets) are not
all below ``EXACT`` adds no sample this decision (and a non-exact label does not complete the previous one).
Acting only: rows above ``TOL`` take the decision's constant offset b(q) (labels / executed targets + b, applied
targets - b); the loop logs the frame-map and applied-target shares separately (``tb_bad_frame``, ``tb_bad_applied``,
``tb_skip``; ``viral.OnlineDAgger.log_extra``). Online smoke (S family, init s28, 64 worlds, 400 decisions, the
consistent images): frame map 0 rows above TOL, 0 samples skipped; applied targets 0.35 % above TOL on average (2.6 %
in the first 50 decisions, where every episode starts together and s28 reaches 0.30-0.32 m out at 7-17 cm height
with the gripper pointing down; 0-0.13 % later).

Images (``OnlineConfig.table_render``, off by default): with the plain option the training renders keep the TRUE
arm and the sim-height world while the student reads Phi(q), a mismatch that never occurs on the robot (there the
arm in the image IS the reading) and that tells the drop. ``table_render`` draws the arm at the reading (Phi of the
image state, also with image lag) and the world on the plane (cube sheared by the plane height under it and tilted
with the plane; plate, floor, mat, walls, strip, ceiling on the plane as ``RealScene.on_plane``; ``SO101Scene``
``world_plane``), so the training images are the robot's images of a lower / tilted table; the aux cube target is
then the drawn cube (z + plane height).

Robot: the real joints ARE the observed frame (no offset at deploy). The robot's action-layer floor must follow the
real table: training keeps the floor 1.5 mm above the TRUE table, i.e. above the plane in the observed frame; the
robot's ``run_student_real.py --table-z 0`` (default) keeps the lowest gripper point 1.5 mm above z = 0 of the ARM
frame, ~13 mm above the real table at the cubes, and holds a lower descent up. Deploy a table-trained student with
``--table-plane calib`` (``hw_lib.ActionLayer(table_plane=)``: the floor and the guard's floor checks above the
calibrated plane; MuJoCo: ``vision_viral.py eval --floor table``), never with ``--table-z 0``. The robot script does
this by itself (2026-10-24): ``run_student_real.py --floor-mode auto`` (default) reads ``v8_online.table_bias`` from
the checkpoint (``hw_lib.student_floor_mode``), and live mode refuses a flat floor for such a student.

Round v12 (2026-10-26, off by default): ``TableBiasConfig.plane`` = one fixed plane in every world (no draw) and
``viral.OnlineConfig.student_floor`` "observed" = the online env's floor flat in the OBSERVED frame (the robot's flat
arm-frame floor; in the true frame raised by -plane at the grasp point, ``joint_env.JointEnvConfig.floor_raise``), so
the student deploys with the robot's flat floor (``hw_lib.student_floor_mode``: v8_online.student_floor observed ->
flat) and the labelling teacher is the one fine-tuned with the same floor raise.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

PITCH = (1, 2, 3)              # shoulder_lift, elbow_flex, wrist_flex (sim joint order)
ITERS = 8                      # Gauss-Newton iterations (module docstring: < 2e-16 m on all real robot states; 4 was
                               # not enough at far reach: up to 2 mm)
W_ROT = 0.1                    # m / rad: rotation weight of the fit (the collector's IK weight)
STEP_MAX = 0.3                 # rad: largest change of a joint in one iteration (a guard near the reach limit)
EXACT = 1e-9                   # m (+ 0.1 m / rad): a map row at or below this residual is exact (training data)
TOL = 5e-4                     # m (+ 0.1 m / rad): a map row above this residual has no image (acting: ``viral``
                               # fallback to the decision's constant offset)


@dataclass
class TableBiasConfig:
    """Per world per episode with probability ``p`` a table plane, else the sim table (b = 0):
    the table top at x = ``x_ref_m`` (y = 0) ``drop_mm`` ~ U(lo, hi) mm BELOW the sim table (positive = lower), its
    slope along x ``slope_x`` ~ U(lo, hi) mm per m (the measured real table: -36.4, i.e. it falls with x) and along y
    ``slope_y`` (real +18.4; off by default). Plane c = (-drop - slope_x x_ref, slope_x, slope_y) (m, ``table.py``).

    ``plane`` (real2sim round v12, 2026-10-26; () = off): ONE fixed plane (c0, c1, c2) (m, ``table.py``) for every
    world and episode, no random draw (``sample`` uses no random numbers); ``p`` must then be 1 (the default with a
    plane) and the ranges must not be given. (0, 0, 0) is allowed: every world on the sim table, the frame map the
    identity (``shift`` returns q itself)."""
    p: float = 0.7
    drop_mm: tuple = (0.0, 20.0)
    slope_x: tuple = (-40.0, 40.0)
    slope_y: tuple = (0.0, 0.0)
    x_ref_m: float = 0.25
    plane: tuple = ()

    RANGES = ("drop_mm", "slope_x", "slope_y")

    @staticmethod
    def from_dict(d) -> "TableBiasConfig":
        d = dict(d or {})
        bad = sorted(set(d) - set(TableBiasConfig.__dataclass_fields__) - {"note"})
        if bad:
            raise ValueError(f"unknown table_bias keys {bad}")
        d.pop("note", None)
        pl = d.get("plane")
        if pl is not None and not (isinstance(pl, (list, tuple)) and len(pl) == 0):   # one fixed plane (docstring)
            if not isinstance(pl, (list, tuple)) or len(pl) != 3 or \
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in pl):
                raise ValueError(f"table_bias plane {pl!r}: 3 numbers (c0, c1, c2), m")
            pl = tuple(float(v) + 0.0 for v in pl)
            if not all(math.isfinite(v) for v in pl) or abs(pl[0]) > 0.06 or max(abs(pl[1]), abs(pl[2])) > 0.2:
                raise ValueError(f"table_bias plane {list(pl)}: finite, |c0| <= 0.06 m, |c1|, |c2| <= 0.2 (m per m)")
            given = sorted(k for k in TableBiasConfig.RANGES if k in d)
            if given:
                raise ValueError(f"table_bias plane with {given}: a fixed plane has no ranges")
            if float(d.get("p", 1.0)) != 1.0:
                raise ValueError(f"table_bias plane with p {d['p']}: a fixed plane is in every world (p 1)")
            d["plane"], d["p"] = pl, 1.0
        else:
            d.pop("plane", None)
        for k in TableBiasConfig.RANGES:
            if k in d:
                v = tuple(float(x) for x in d[k])
                if len(v) != 2 or not v[0] <= v[1] or not all(math.isfinite(x) for x in v):
                    raise ValueError(f"table_bias {k} {list(d[k])}: (lo, hi) with lo <= hi")
                d[k] = v
        c = TableBiasConfig(**d)
        if not 0.0 <= float(c.p) <= 1.0:
            raise ValueError(f"table_bias p {c.p}: a probability in [0, 1]")
        if max(abs(v) for v in c.drop_mm) > 60.0 or max(abs(v) for k in ("slope_x", "slope_y")
                                                         for v in getattr(c, k)) > 200.0:
            raise ValueError(f"table_bias {c.to_dict()}: drop beyond 60 mm or slope beyond 200 mm/m (units: mm, "
                             "mm per m)")
        return c

    def to_dict(self) -> dict:
        d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}
        if not self.plane:                # off: the old record
            d.pop("plane")
        else:                             # a fixed plane: no ranges (unused)
            for k in self.RANGES:
                d.pop(k)
        return d


def plane_coef(drop_m, slope_x, slope_y, x_ref):
    """Drop (m, positive = lower) at (x_ref, 0) and slopes (m / m) -> plane coefficients [..., 3] (c0, c1, c2)."""
    drop_m = torch.as_tensor(drop_m)
    sx = torch.as_tensor(slope_x, dtype=drop_m.dtype, device=drop_m.device)
    sy = torch.as_tensor(slope_y, dtype=drop_m.dtype, device=drop_m.device)
    return torch.stack(torch.broadcast_tensors(-drop_m - sx * float(x_ref), sx, sy), -1)


def sample(n, cfg: TableBiasConfig, gen=None, device="cpu"):
    """n per-world draws -> (plane [n, 3] float64 (zeros where off), on [n] bool). Random numbers (one call each, on
    ``gen``): on, drop, slope_x, slope_y. A fixed ``cfg.plane``: that plane in every row, on everywhere, no random
    numbers."""
    if cfg.plane:
        pl = torch.tensor(cfg.plane, dtype=torch.float64, device=device)
        return pl.expand(n, 3).clone(), torch.ones(n, dtype=torch.bool, device=device)
    f = dict(device=device, generator=gen, dtype=torch.float64)
    on = torch.rand(n, **f) < float(cfg.p)

    def u(lo_hi, scale):
        lo, hi = (float(v) * scale for v in lo_hi)
        return lo + (hi - lo) * torch.rand(n, **f)

    drop, sx, sy = u(cfg.drop_mm, 1e-3), u(cfg.slope_x, 1e-3), u(cfg.slope_y, 1e-3)
    plane = plane_coef(drop, sx, sy, cfg.x_ref_m)
    return torch.where(on[:, None], plane, torch.zeros_like(plane)), on


def height(plane, xy):
    """Plane z [...] at xy [..., 2] (torch; plane [..., 3] or [3])."""
    plane = torch.as_tensor(plane, dtype=xy.dtype, device=xy.device)
    return plane[..., 0] + plane[..., 1] * xy[..., 0] + plane[..., 2] * xy[..., 1]


def shift(kin, q, dz, iters=ITERS, w_rot=W_ROT):
    """Joints that move the grasp point of poses q [B, 6] by dz [B] (m) along world z with the gripper rotation
    unchanged: Gauss-Newton on the pitch joints (``PITCH``) from q (pan, roll, jaw kept); the iterate with the lowest
    residual is returned (q itself included), so where it does not converge (near a straight arm, far outside the
    task poses) the shift degrades to a partial one (residual <= |dz|) instead of a diverged pose. -> (q' [B, 6] in
    q's dtype, residual [B]: |position error| (m) + w_rot |rotation error| (rad) of exact FK). Rows with dz = 0 (and
    rows with a non-finite q or dz) return q itself, bit for bit, residual 0."""
    from nedm.so101_policy.kin import rotvec_fast
    dt = q.dtype
    dz = torch.as_tensor(dz, dtype=kin.dtype, device=q.device).expand(q.shape[0])
    keep = (dz == 0) | ~torch.isfinite(dz) | ~torch.isfinite(q).all(-1)
    x = torch.where(keep[:, None], torch.zeros_like(q), q).to(kin.dtype)
    dz = torch.where(keep, torch.zeros_like(dz), dz)
    p0, r0 = kin.fk(x)
    tgt = p0 + torch.stack((torch.zeros_like(dz), torch.zeros_like(dz), dz), -1)

    def resid(y):
        p, r = kin.fk(y)
        e = (tgt - p).norm(dim=-1) + w_rot * rotvec_fast(r0 @ r.transpose(-1, -2)).norm(dim=-1)
        return torch.where(torch.isfinite(e), e, torch.full_like(e, math.inf))

    def step(y, d):
        return torch.cat((y[:, :1], y[:, 1:4] + d, y[:, 4:]), -1)

    J = list(PITCH)
    eye = torch.eye(len(J), dtype=kin.dtype, device=x.device)
    best, err = x, dz.abs()                                                            # q itself: residual |dz|
    for _ in range(int(iters)):
        jp, jr, p, r = kin.jacobian(x)
        a = torch.cat((jp[..., J], w_rot * jr[..., J]), -2)                                  # [B, 6, 3]
        rhs = torch.cat((tgt - p, w_rot * rotvec_fast(r0 @ r.transpose(-1, -2))), -1)[..., None]
        at = a.transpose(-1, -2)
        dx = torch.linalg.solve_ex(at @ a + 1e-12 * eye, at @ rhs, check_errors=False)[0][..., 0]
        dx = torch.where(torch.isfinite(dx), dx, torch.zeros_like(dx)).clamp(-STEP_MAX, STEP_MAX)
        x = step(x, dx)
        e = resid(x)
        better = e < err                                                               # keep the best iterate
        best, err = torch.where(better[:, None], x, best), torch.where(better, e, err)
    return torch.where(keep[:, None], q, best.to(dt)), torch.where(keep, torch.zeros_like(err), err)


def observe(kin, q, plane):
    """The student's reading q_obs = Phi(q) of true poses q [B, 6] in worlds whose table lies on ``plane`` [B, 3]
    (module docstring): the grasp point moved by the plane height under it, same rotation. -> (q_obs, residual)."""
    p, _ = kin.fk(q.to(kin.dtype))
    return shift(kin, q, height(torch.as_tensor(plane, dtype=kin.dtype, device=p.device), p[..., :2]))


def to_true(kin, u_obs, plane):
    """Exact inverse of ``observe``: the true joints whose reading is u_obs [B, 6] (a vertical shift keeps xy, so the
    plane is evaluated at FK_xy(u_obs)). -> (u, residual): the applied targets of the online loop."""
    p, _ = kin.fk(u_obs.to(kin.dtype))
    return shift(kin, u_obs, -height(torch.as_tensor(plane, dtype=kin.dtype, device=p.device), p[..., :2]))


def observe_qd(kin, q, qd, plane, q_obs=None):
    """The reading's velocity J_Phi(q) qd [B, 6] (rad/s) of true poses q [B, 6] moving at qd [B, 6], analytic (the time
    derivative of the map's definition, module docstring): Phi keeps pan, roll and the jaw, so their rates stay; the
    pitch rates d [B, 3] solve (least squares with the map's rotation weight; the system is consistent, so exact where
    ``q_obs`` = Phi(q) is)

        J_p,pitch(Phi(q)) d = J_p(q) qd + e_z (c1 v_x + c2 v_y) - J_p,other(Phi(q)) qd_other,   v = J_p(q) qd
        J_r,pitch(Phi(q)) d = J_r(q) qd - J_r,other(Phi(q)) qd_other

    (grasp-point velocity plus the plane's slope under the moving gripper; the gripper's angular velocity unchanged).
    ``q_obs``: Phi(q) if known (else computed). qd = 0 gives exactly 0. float64."""
    q, qd = q.to(kin.dtype), qd.to(kin.dtype)
    n = q.shape[0]
    pl = torch.as_tensor(plane, dtype=kin.dtype, device=q.device).expand(n, 3)
    if q_obs is None:
        q_obs = observe(kin, q, pl)[0]
    jp, jr, _, _ = kin.jacobian(q)
    jpo, jro, _, _ = kin.jacobian(q_obs.to(kin.dtype))
    J, O = list(PITCH), [0, 4]                                   # pitch joints, the other arm joints (pan, roll)
    v = (jp @ qd[:, :5, None])[..., 0]                           # grasp-point velocity [B, 3]
    w = (jr @ qd[:, :5, None])[..., 0]                           # gripper angular velocity [B, 3]
    vz = pl[:, 1] * v[:, 0] + pl[:, 2] * v[:, 1]                 # the plane height's rate under the gripper
    v = v + torch.stack((torch.zeros_like(vz), torch.zeros_like(vz), vz), -1)
    qo = qd[:, O, None]
    a = torch.cat((jpo[..., J], W_ROT * jro[..., J]), -2)                                     # [B, 6, 3]
    rhs = torch.cat((v - (jpo[..., O] @ qo)[..., 0], W_ROT * (w - (jro[..., O] @ qo)[..., 0])), -1)[..., None]
    at = a.transpose(-1, -2)
    eye = torch.eye(len(J), dtype=kin.dtype, device=q.device)
    d = torch.linalg.solve_ex(at @ a + 1e-15 * eye, at @ rhs, check_errors=False)[0][..., 0]
    out = torch.cat((qd[:, :1], d, qd[:, 4:]), -1)
    return torch.where((qd == 0).all(-1, keepdim=True), torch.zeros_like(out), out)


def is_on(d) -> bool:
    """``OnlineConfig.table_bias`` dict -> on? (empty or only a ``note``: off)."""
    return bool(set(dict(d or {})) - {"note"})
