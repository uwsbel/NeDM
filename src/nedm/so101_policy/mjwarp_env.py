"""Batched MuJoCo Warp worlds of the SO101 grasp collector scene: joint targets in, ``WorldState`` out (spec 2.7).

The GPU twin of ``mujoco_env.MujocoWorlds`` (same API). The model is the collector's
(``nedm.contact_nrd.collect.so101_grasp.Scene``: Menagerie SO101, table, 30 mm cube, 0.5 ms implicitfast, elliptic
cones, Newton solver, 100 iterations, 50 line-search iterations; MuJoCo Warp raises the solver tolerance from 1e-10
to 1e-6 because it runs in float32). One ``step`` = one 20 ms command = 40 physics steps, captured in ONE CUDA graph
together with the contact bookkeeping and the state output. Units SI, world frame z up, table top at z = 0, arm base
at the origin. Physics state is float32 (MuJoCo Warp); the WorldState is float64 computed from it.

WorldState definitions = the grasp NRD data (records stored float32, then ``data.step_average_rates``, 20 ms step):

- ``q``: joint angles at the end of the command. ``qd``: 20 ms step average (q(t) - q(t - 20 ms)) / 0.02.
- ``cube_pos``; ``cube_quat`` w x y z, the TRUE orientation (w >= 0 at the reset state, then sign continuous).
- ``cube_v`` = (p(t) - p(t - 20 ms)) / 0.02 and ``cube_w`` = rotation vector of q(t) q(t - 20 ms)^-1 / 0.02, world
  frame (``cube_rates="step_average"``, default, = ``data._step_average``). ``cube_rates="instant"``: the
  collector's raw record instead (MuJoCo velocity; spin rotated to the world frame). After ``reset`` / ``set_cube``
  the rates are the instantaneous ones (as given, float32-rounded), as the first record of a data episode.
- ``contact`` [fixed jaw-cube, moving jaw-cube, cube-table]: the pair's summed normal force (first efc row of each
  elliptic contact = mj_contactForce normal) above ``contact_threshold_n`` (0.01 N) in any of the 40 physics steps of
  the command = OR of the data's two 10 ms interval labels = ``data.physical_labels`` of the step. The NRD switch
  view (this step or the one before, ``data.model_labels``) is ``switch_contact``. After a reset / set_cube: the
  forces of one ``forward`` at that state.
- ``arm_table``: the same force rule for any arm geom (gripper, moving jaw, other links, base hulls) vs the table.
- ``invalid``: the other collector ends, checked as the collector does where possible: cube_other_link, arm_self,
  cube_base (force rule above); cube_off_table (cube centre x or y outside the table minus 5 cm, or z < -0.01 m, at
  the end of either 10 ms record); non-finite qpos / qvel at the end of the command. NOT computed on the GPU: the
  collector's visual-mesh overlap end (``cube_visual_overlap``; CPU-only signed distance grids) and MuJoCo's
  ``mjWARN_BADQACC``.

Flags are per step; physics always runs the whole command and never stops a world. ``end_reason`` [B] keeps the
FIRST end since the reset (``so101_grasp.TERMINAL`` index, 7 = numerical) and ``end_step`` the step that raised it
(inside a command: the earlier 10 ms record first; inside a record: contact kinds in TERMINAL order, then off table;
a non-finite state wins). Not bit-repeatable: MuJoCo Warp collects contacts with atomics, so identical worlds can
drift apart (measured in docs/build_mjwarp_env.md). ``overflow`` [B] is MuJoCo Warp's OverflowType bitmask since
the world's reset (e.g. 1024 = line-search iteration limit reached in some physics step).
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import torch
import warp as wp

from nedm.so101_policy.world import CONTROL_DT, JOINTS, WorldState

NCLASS = 7       # collector contact kinds: 3 pairs, arm-table, cube-other link, arm self, cube-base
NREC = 2         # 10 ms records per command
RAW = 25         # packed row: q 6, qd 6, cube pos 3, quat 4, v 3, w 3
NFLAG = 5        # contact 3, arm_table, invalid
NUMERICAL = 7    # end code after so101_grasp.TERMINAL ("end", arm_table, ..., cube_visual_overlap)
MJW_PINS = {"mujoco": "3.15.0", "mujoco_warp": "3.15.0", "warp": "1.18.0"}


# ---- kernels: float32 physics state in, float64 WorldState rows out --------------------------------------------
@wp.kernel
def _set_ctrl(targets: wp.array2d(dtype=wp.float64), tlo: wp.array(dtype=wp.float64), thi: wp.array(dtype=wp.float64),
              ctrl: wp.array2d(dtype=float)):
    """ctrl = float32(clip(target, tlo, thi)): the collector's applied value (so101_grasp.run_episode)."""
    w, j = wp.tid()
    ctrl[w, j] = float(wp.clamp(targets[w, j], tlo[j], thi[j]))


@wp.kernel
def _zero_forces(fsum: wp.array2d(dtype=float), fmax: wp.array2d(dtype=float), offtab: wp.array2d(dtype=wp.int32)):
    w, k = wp.tid()
    if k < 7:
        fsum[w, k] = 0.0
    fmax[w, k] = 0.0
    if k < 2:
        offtab[w, k] = 0


@wp.kernel
def _sum_contacts(nacon: wp.array(dtype=int), naconmax: int, geom: wp.array(dtype=wp.vec2i),
                  worldid: wp.array(dtype=int), efc_address: wp.array2d(dtype=int), efc_force: wp.array2d(dtype=float),
                  njmax: int, gcls: wp.array(dtype=wp.int32), kind: wp.array(dtype=wp.int32),
                  fsum: wp.array2d(dtype=float)):
    """Summed normal force per world and collector contact kind of the last physics step (Scene.contact_forces:
    efc_force of the contact's first efc row, contacts without efc rows count 0)."""
    c = wp.tid()
    if c >= wp.min(nacon[0], naconmax):
        return
    adr = efc_address[c, 0]
    if adr < 0 or adr >= njmax:
        return
    g = geom[c]
    if g[0] < 0 or g[1] < 0:
        return
    k = kind[gcls[g[0]] * 7 + gcls[g[1]]]
    if k < 0:
        return
    w = worldid[c]
    wp.atomic_add(fsum, w, k, efc_force[w, adr])


@wp.kernel
def _fold_forces(fsum: wp.array2d(dtype=float), fmax: wp.array2d(dtype=float), qpos: wp.array2d(dtype=float),
                 cq: int, rec: int, record_end: int, table_lo: wp.vec2, table_hi: wp.vec2,
                 offtab: wp.array2d(dtype=wp.int32)):
    """fmax[record] = max(fmax[record], fsum), fsum = 0; at the end of a 10 ms record the collector's off-table test."""
    w = wp.tid()
    for k in range(7):
        fmax[w, rec * 7 + k] = wp.max(fmax[w, rec * 7 + k], fsum[w, k])
        fsum[w, k] = 0.0
    if record_end != 0:
        x = qpos[w, cq]
        y = qpos[w, cq + 1]
        if x < table_lo[0] or x > table_hi[0] or y < table_lo[1] or y > table_hi[1] or qpos[w, cq + 2] < -0.01:
            offtab[w, rec] = 1


@wp.func
def _quat_mat(q0: wp.float64, q1: wp.float64, q2: wp.float64, q3: wp.float64):
    """Rotation matrix of a w x y z quaternion (mju_quat2Mat, no normalisation)."""
    two = wp.float64(2.0)
    return wp.mat33d(
        q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, two * (q1 * q2 - q0 * q3), two * (q1 * q3 + q0 * q2),
        two * (q1 * q2 + q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, two * (q2 * q3 - q0 * q1),
        two * (q1 * q3 - q0 * q2), two * (q2 * q3 + q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)


@wp.func
def _end_code(fmax: wp.array2d(dtype=float), offtab: wp.array2d(dtype=wp.int32), w: int, nrec: int, thr: float,
              finite: int):
    """First collector end of a command: numerical (7) wins; else per record (earlier first) arm_table 1,
    cube_other_link 2, arm_self 3, cube_base 4, then off table 5; 0 = none."""
    code = int(0)
    if finite == 0:
        code = 7
    for r in range(nrec):
        if code == 0:
            for k in range(3, 7):
                if code == 0 and fmax[w, r * 7 + k] > thr:
                    code = k - 2
            if code == 0 and offtab[w, r] != 0:
                code = 5
    return code


@wp.func
def _any_invalid(fmax: wp.array2d(dtype=float), offtab: wp.array2d(dtype=wp.int32), w: int, nrec: int, thr: float,
                 finite: int):
    """1 if any end other than arm-table happened in the command (any record): cube_other_link, arm_self,
    cube_base, off table, non-finite state."""
    bad = int(0)
    if finite == 0:
        bad = 1
    for r in range(nrec):
        if offtab[w, r] != 0:
            bad = 1
        for k in range(4, 7):
            if fmax[w, r * 7 + k] > thr:
                bad = 1
    return bad


@wp.kernel
def _finish(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), cq: int, cv: int,
            fmax: wp.array2d(dtype=float), offtab: wp.array2d(dtype=wp.int32), thr: float, inv_dt: wp.float64,
            instant: int, prev: wp.array2d(dtype=wp.float64), out: wp.array2d(dtype=wp.float64),
            flags: wp.array2d(dtype=wp.int32), code: wp.array(dtype=wp.int32)):
    """WorldState row after a command (module docstring); prev <- this command's pose (arm 6, cube pos 3, quat 4)."""
    w = wp.tid()
    finite = int(1)
    for j in range(6):
        q = wp.float64(qpos[w, j])
        out[w, j] = q
        out[w, 6 + j] = (q - prev[w, j]) * inv_dt
        prev[w, j] = q
        if not (wp.isfinite(qpos[w, j]) and wp.isfinite(qvel[w, j])):
            finite = 0
    for i in range(3):
        p = wp.float64(qpos[w, cq + i])
        out[w, 12 + i] = p
        if instant == 0:
            out[w, 19 + i] = (p - prev[w, 6 + i]) * inv_dt
        else:
            out[w, 19 + i] = wp.float64(qvel[w, cv + i])
        prev[w, 6 + i] = p
    for i in range(7):
        if not wp.isfinite(qpos[w, cq + i]):
            finite = 0
    for i in range(6):
        if not wp.isfinite(qvel[w, cv + i]):
            finite = 0
    a0 = prev[w, 9]
    a1 = prev[w, 10]
    a2 = prev[w, 11]
    a3 = prev[w, 12]
    b0 = wp.float64(qpos[w, cq + 3])
    b1 = wp.float64(qpos[w, cq + 4])
    b2 = wp.float64(qpos[w, cq + 5])
    b3 = wp.float64(qpos[w, cq + 6])
    if a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 < wp.float64(0.0):   # sign continuous (export.quat_continuous)
        b0 = -b0
        b1 = -b1
        b2 = -b2
        b3 = -b3
    out[w, 15] = b0
    out[w, 16] = b1
    out[w, 17] = b2
    out[w, 18] = b3
    if instant == 0:   # rotation vector of q1 * conj(q0) / dt (data._step_average)
        v0 = wp.vec3d(-a1, -a2, -a3)
        v1 = wp.vec3d(b1, b2, b3)
        rw = b0 * a0 - wp.dot(v1, v0)
        rv = b0 * v0 + a0 * v1 + wp.cross(v1, v0)
        if rw < wp.float64(0.0):
            rw = -rw
            rv = -rv
        n = wp.length(rv)
        rot = rv * wp.float64(2.0)
        if n > wp.float64(1e-12):
            rot = rv * (wp.float64(2.0) * wp.atan2(n, rw) / n)
        for i in range(3):
            out[w, 22 + i] = rot[i] * inv_dt
    else:              # world spin = R(q) @ body spin (Scene.state)
        om = _quat_mat(b0, b1, b2, b3) @ wp.vec3d(wp.float64(qvel[w, cv + 3]), wp.float64(qvel[w, cv + 4]),
                                                  wp.float64(qvel[w, cv + 5]))
        for i in range(3):
            out[w, 22 + i] = om[i]
    prev[w, 9] = b0
    prev[w, 10] = b1
    prev[w, 11] = b2
    prev[w, 12] = b3
    for k in range(4):
        flags[w, k] = wp.where(wp.max(fmax[w, k], fmax[w, 7 + k]) > thr, 1, 0)
    code[w] = _end_code(fmax, offtab, w, 2, thr, finite)
    flags[w, 4] = _any_invalid(fmax, offtab, w, 2, thr, finite)


@wp.kernel
def _place(mask: wp.array(dtype=wp.int32), start: wp.array2d(dtype=wp.float64), hold: wp.array2d(dtype=wp.float64),
           tlo: wp.array(dtype=wp.float64), thi: wp.array(dtype=wp.float64), cq: int, cv: int, arm: int,
           qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), ctrl: wp.array2d(dtype=float),
           prev: wp.array2d(dtype=wp.float64), out: wp.array2d(dtype=wp.float64)):
    """Write a packed start row into the masked worlds (arm != 0: arm, held targets and cube; else the cube only).
    The cube spin is given in the world frame (MuJoCo's free joint takes it in the body frame). The state row and the
    step-average reference get the float32 values the simulation holds (as the data's float32 records)."""
    w = wp.tid()
    if mask[w] == 0:
        return
    if arm != 0:
        for j in range(6):
            qpos[w, j] = float(start[w, j])
            qvel[w, j] = float(start[w, 6 + j])
            ctrl[w, j] = float(wp.clamp(hold[w, j], tlo[j], thi[j]))
            prev[w, j] = wp.float64(qpos[w, j])
            out[w, j] = wp.float64(qpos[w, j])
            out[w, 6 + j] = wp.float64(qvel[w, j])
    s = wp.float64(1.0)
    if start[w, 15] < wp.float64(0.0):   # w >= 0 at the first record (export.quat_continuous)
        s = wp.float64(-1.0)
    om = wp.transpose(_quat_mat(start[w, 15], start[w, 16], start[w, 17], start[w, 18])) @ wp.vec3d(
        start[w, 22], start[w, 23], start[w, 24])
    for i in range(3):
        qpos[w, cq + i] = float(start[w, 12 + i])
        qvel[w, cv + i] = float(start[w, 19 + i])
        qvel[w, cv + 3 + i] = float(om[i])
        prev[w, 6 + i] = wp.float64(qpos[w, cq + i])
        out[w, 12 + i] = wp.float64(qpos[w, cq + i])
        out[w, 19 + i] = wp.float64(float(start[w, 19 + i]))
        out[w, 22 + i] = wp.float64(float(start[w, 22 + i]))
    for i in range(4):
        qpos[w, cq + 3 + i] = float(start[w, 15 + i])
        prev[w, 9 + i] = s * wp.float64(qpos[w, cq + 3 + i])
        out[w, 15 + i] = s * wp.float64(qpos[w, cq + 3 + i])


@wp.kernel
def _flags_now(mask: wp.array(dtype=wp.int32), fsum: wp.array2d(dtype=float), qpos: wp.array2d(dtype=float),
               qvel: wp.array2d(dtype=float), cq: int, thr: float, table_lo: wp.vec2, table_hi: wp.vec2,
               fmax: wp.array2d(dtype=float), offtab: wp.array2d(dtype=wp.int32), flags: wp.array2d(dtype=wp.int32),
               code: wp.array(dtype=wp.int32)):
    """Flags and end code of the masked worlds from the forces of one forward call at the current state."""
    w = wp.tid()
    if mask[w] == 0:
        return
    for k in range(7):
        fmax[w, k] = fsum[w, k]
        fmax[w, 7 + k] = 0.0
        fsum[w, k] = 0.0
    x = qpos[w, cq]
    y = qpos[w, cq + 1]
    offtab[w, 1] = 0
    offtab[w, 0] = 0
    if x < table_lo[0] or x > table_hi[0] or y < table_lo[1] or y > table_hi[1] or qpos[w, cq + 2] < -0.01:
        offtab[w, 0] = 1
    finite = int(1)
    for i in range(qpos.shape[1]):
        if not wp.isfinite(qpos[w, i]):
            finite = 0
    for i in range(qvel.shape[1]):
        if not wp.isfinite(qvel[w, i]):
            finite = 0
    for k in range(4):
        flags[w, k] = wp.where(fmax[w, k] > thr, 1, 0)
    code[w] = _end_code(fmax, offtab, w, 1, thr, finite)
    flags[w, 4] = _any_invalid(fmax, offtab, w, 1, thr, finite)


@wp.kernel
def _keep_others(mask: wp.array(dtype=wp.int32), src: wp.array2d(dtype=float), dst: wp.array2d(dtype=float)):
    """dst[w] = src[w] for the worlds outside the mask (undo a warm-up for them)."""
    w, i = wp.tid()
    if mask[w] == 0:
        dst[w, i] = src[w, i]


@wp.kernel
def _instant_row(mask: wp.array(dtype=wp.int32), qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), cq: int,
                 cv: int, start: wp.array2d(dtype=wp.float64)):
    """Packed row of the simulation state with the instantaneous rates (Scene.state), masked worlds."""
    w = wp.tid()
    if mask[w] == 0:
        return
    for j in range(6):
        start[w, j] = wp.float64(qpos[w, j])
        start[w, 6 + j] = wp.float64(qvel[w, j])
    for i in range(7):
        start[w, 12 + i] = wp.float64(qpos[w, cq + i])
    om = _quat_mat(start[w, 15], start[w, 16], start[w, 17], start[w, 18]) @ wp.vec3d(
        wp.float64(qvel[w, cv + 3]), wp.float64(qvel[w, cv + 4]), wp.float64(qvel[w, cv + 5]))
    for i in range(3):
        start[w, 19 + i] = wp.float64(qvel[w, cv + i])
        start[w, 22 + i] = om[i]


# ---- helpers ------------------------------------------------------------------------------------------------------
def _get(start, name):
    return start.get(name) if isinstance(start, dict) else getattr(start, name, None)


def pack_start(start, n: int | None = None) -> torch.Tensor:
    """WorldState-like (attributes or dict keys q, qd, cube_pos, cube_quat, cube_v, cube_w; qd, cube_v, cube_w may be
    missing = 0) -> packed rows [n, 25] float64 (CPU)."""
    q = torch.as_tensor(_get(start, "q"), dtype=torch.float64)
    n = q.reshape(-1, JOINTS).shape[0] if n is None else n
    parts = []
    for k, dim in (("q", 6), ("qd", 6), ("cube_pos", 3), ("cube_quat", 4), ("cube_v", 3), ("cube_w", 3)):
        x = _get(start, k)
        parts.append(torch.zeros(n, dim, dtype=torch.float64) if x is None
                     else torch.as_tensor(x, dtype=torch.float64).detach().cpu().reshape(n, dim))
    return torch.cat(parts, 1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """R(q) v for w x y z quaternions [B, 4] (not normalised, as mju_quat2Mat) and vectors [B, 3]."""
    w, u = q[:, :1], q[:, 1:]
    return (w * w - (u * u).sum(1, keepdim=True)) * v + 2 * (u * v).sum(1, keepdim=True) * u + 2 * w * torch.cross(u, v, dim=1)


def collector_classes(scene):
    """(geom class [ngeom], contact kind table [7 * 7]) of the collector scene (Scene.cls, Scene.kind)."""
    return np.asarray(scene.cls, np.int32), np.asarray(scene.kind, np.int32).reshape(-1)


def end_names():
    from nedm.contact_nrd.collect.so101_grasp import TERMINAL
    return list(TERMINAL) + ["numerical"]


# ---- the worlds -----------------------------------------------------------------------------------------------------
class MjwarpWorlds:
    """``num_envs`` worlds of the collector scene in MuJoCo Warp on one CUDA device (API of ``MujocoWorlds``).

    ``reset(idx, start, warmup=False, spawn=False, targets=None)``: worlds at a WorldState-like start (warmup: the
    collector's 0.3 s warm-up first, arm held at ``start.q``, cube resting at its pose; spawn: cube parked during the
    warm-up, then placed with ``cube_v``, ``cube_w``); ``step(targets [B, 6])``: one 20 ms command;
    ``set_cube(idx, pos, quat, v, w)``: move the cube (disturbance). All return the WorldState of all worlds
    (float64, on ``device``). Attributes: ``end_reason`` / ``end_step`` [B], ``switch_contact`` [B, 3],
    ``contact_force`` [B, 7] (largest summed normal force per collector kind in the last command, N), ``applied``
    [B, 6] float32 targets of the last command, ``steps`` [B], ``overflow`` [B] (MuJoCo Warp OverflowType bits since
    the world's reset).

    ``nconmax`` contacts per world on average (one pool of nconmax * B), ``njmax`` constraint rows per world;
    ``graph=False`` runs the same kernels eagerly (debugging)."""

    def __init__(self, num_envs: int, device="cuda:0", nconmax: int = 48, njmax: int = 192,
                 cube_rates: str = "step_average", cfg: dict | None = None, physics: dict | None = None,
                 graph: bool = True):
        import mujoco_warp as mjw
        from nedm.contact_nrd.collect import so101_grasp as g

        if cube_rates not in ("step_average", "instant"):
            raise ValueError("cube_rates is 'step_average' or 'instant'")
        wp.config.quiet = True
        if hasattr(wp, "LOG_WARNING"):
            wp.config.log_level = wp.LOG_WARNING   # no "Module ... load" line per kernel
        wp.init()
        self.mjw = mjw
        self.B, self.device = int(num_envs), torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("MjwarpWorlds runs on a CUDA device")
        self.wdev = wp.get_device(f"cuda:{self.device.index or 0}")
        self.scene = s = g.Scene(cfg or g.load_config(), physics)
        m = s.m
        if list(s.qadr) != list(range(JOINTS)) or list(s.vadr) != list(range(JOINTS)) or list(s.act) != list(range(JOINTS)):
            raise ValueError("expected the arm joints and actuators at addresses 0-5")
        self.cq, self.cv, self.thr = int(s.cq), int(s.cv), float(s.thr)
        self.cube_rates, self.instant = cube_rates, int(cube_rates == "instant")
        self.spc = round(CONTROL_DT / m.opt.timestep)                 # physics steps per command (40)
        self.spr = round(s.cfg["record_step_s"] / m.opt.timestep)     # per 10 ms record (20)
        if self.spc != NREC * self.spr:
            raise ValueError("a command must be 2 records")
        self.nconmax, self.njmax, self.graph = nconmax, njmax, graph
        self.tlo_np, self.thi_np = np.asarray(s.tlo, np.float64), np.asarray(s.thi, np.float64)
        self.table_lo, self.table_hi = wp.vec2(*map(float, s.table_lo)), wp.vec2(*map(float, s.table_hi))
        cls, kind = collector_classes(s)
        B = self.B
        with wp.ScopedDevice(self.wdev):
            self.stream = wp.get_stream(self.wdev)
            self.m = mjw.put_model(m)
            self.m.opt.warn_overflow = 0          # overflow bits are still set in d.overflow; no printf per world
            self.d = mjw.make_data(m, nworld=B, nconmax=nconmax, njmax=njmax)
            self.gcls, self.kind = wp.array(cls, dtype=wp.int32), wp.array(kind, dtype=wp.int32)
            self.tlo, self.thi = wp.array(self.tlo_np, dtype=wp.float64), wp.array(self.thi_np, dtype=wp.float64)
            self.targets = wp.zeros((B, JOINTS), dtype=wp.float64)
            self.hold = wp.zeros((B, JOINTS), dtype=wp.float64)
            self.fsum = wp.zeros((B, NCLASS), dtype=float)
            self.fmax = wp.zeros((B, NREC * NCLASS), dtype=float)
            self.offtab = wp.zeros((B, NREC), dtype=wp.int32)
            self.prev = wp.zeros((B, 13), dtype=wp.float64)
            self.out = wp.zeros((B, RAW), dtype=wp.float64)
            self.flags = wp.zeros((B, NFLAG), dtype=wp.int32)
            self.code = wp.zeros(B, dtype=wp.int32)
            self.start = wp.zeros((B, RAW), dtype=wp.float64)
            self.mask = wp.zeros(B, dtype=wp.int32)
            self.mask_b = wp.zeros(B, dtype=bool)
        t = wp.to_torch
        self._targets_t, self._hold_t, self._out_t, self._flags_t = t(self.targets), t(self.hold), t(self.out), t(self.flags)
        self._start_t, self._mask_t, self._mask_bt, self._code_t = t(self.start), t(self.mask), t(self.mask_b), t(self.code)
        self._fmax_t, self._overflow_t = t(self.fmax), t(self.d.overflow)
        self._qpos_t, self._qvel_t, self._ctrl_t = t(self.d.qpos), t(self.d.qvel), t(self.d.ctrl)
        self._g_step = self._g_reset = self._g_bare = self._g_cube = self._g_place = None
        dev, B = self.device, self.B
        self.ready = torch.zeros(B, dtype=torch.bool, device=dev)
        self.end_reason = torch.zeros(B, dtype=torch.int8, device=dev)
        self.end_step = torch.full((B,), -1, dtype=torch.long, device=dev)
        self.steps = torch.zeros(B, dtype=torch.long, device=dev)
        self.switch_contact = torch.zeros(B, 3, dtype=torch.bool, device=dev)
        self._last_phys = torch.zeros(B, 3, dtype=torch.bool, device=dev)
        self.applied = torch.zeros(B, JOINTS, dtype=torch.float32, device=dev)

    # ---- graph bodies -----------------------------------------------------------------------------------------
    def _launch(self, kernel, dim, inputs):
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.wdev)

    def _sum(self):
        d = self.d
        self._launch(_sum_contacts, d.naconmax, [d.nacon, d.naconmax, d.contact.geom, d.contact.worldid,
                                                 d.contact.efc_address, d.efc.force, d.njmax, self.gcls, self.kind, self.fsum])

    def _body_step(self):
        d, m, B = self.d, self.m, self.B
        self._launch(_set_ctrl, (B, JOINTS), [self.targets, self.tlo, self.thi, d.ctrl])
        self._launch(_zero_forces, (B, NREC * NCLASS), [self.fsum, self.fmax, self.offtab])
        for k in range(self.spc):
            self.mjw.step(m, d)
            self._sum()
            self._launch(_fold_forces, B, [self.fsum, self.fmax, d.qpos, self.cq, k // self.spr,
                                           int((k + 1) % self.spr == 0), self.table_lo, self.table_hi, self.offtab])
        self._launch(_finish, B, [d.qpos, d.qvel, self.cq, self.cv, self.fmax, self.offtab, self.thr,
                                  wp.float64(1.0 / CONTROL_DT), self.instant, self.prev, self.out, self.flags, self.code])

    def _body_place(self, arm: int):
        d = self.d
        self._launch(_place, self.B, [self.mask, self.start, self.hold, self.tlo, self.thi, self.cq, self.cv, arm,
                                      d.qpos, d.qvel, d.ctrl, self.prev, self.out])
        self._body_flags()

    def _body_reset(self):
        self.mjw.reset_data(self.m, self.d, reset=self.mask_b)
        self._body_place(1)

    def _body_flags(self):
        d = self.d
        self.mjw.forward(self.m, d)        # contacts and forces of the state; does not change qpos/qvel/warm start
        self._launch(_zero_forces, (self.B, NREC * NCLASS), [self.fsum, self.fmax, self.offtab])
        self._sum()
        self._launch(_flags_now, self.B, [self.mask, self.fsum, d.qpos, d.qvel, self.cq, self.thr, self.table_lo,
                                          self.table_hi, self.fmax, self.offtab, self.flags, self.code])

    def _body_bare(self):
        for _ in range(self.spc):
            self.mjw.step(self.m, self.d)

    def _run(self, name, body):
        """Run a body as a CUDA graph (captured on first use; ``graph=False``: eagerly), ordered after the torch
        stream's pending work and before its later work."""
        ts = wp.stream_from_torch(self.device)
        self.stream.wait_stream(ts)
        with wp.ScopedDevice(self.wdev), wp.ScopedStream(self.stream):
            if not self.graph:
                body()
            else:
                g = getattr(self, name)
                if g is None:
                    with wp.ScopedCapture(device=self.wdev, stream=self.stream) as cap:
                        body()
                    g = cap.graph
                    setattr(self, name, g)
                wp.capture_launch(g, stream=self.stream)
        ts.wait_stream(self.stream)

    def _select(self, idx) -> torch.Tensor:
        """idx (None = all; indices or bool mask [B]) -> world indices [n] in the order of idx; sets the masks."""
        if idx is None:
            sel = torch.arange(self.B, device=self.device)
        else:
            x = torch.as_tensor(idx, device=self.device)
            sel = torch.nonzero(x).flatten() if x.dtype == torch.bool else x.long().reshape(-1)
            if len(torch.unique(sel)) != len(sel) or (len(sel) and (sel.min() < 0 or sel.max() >= self.B)):
                raise ValueError("world indices must be unique and inside 0 .. B-1")
        self._mask_t.zero_()
        self._mask_t[sel] = 1
        self._mask_bt.copy_(self._mask_t.bool())
        return sel

    def _after_place(self, sel, first: bool):
        code = self._code_t[sel].to(torch.int8)
        if first:
            self.end_reason[sel], self.end_step[sel] = code, torch.where(code > 0, 0, -1)
            self.steps[sel], self._last_phys[sel], self.switch_contact[sel] = 0, False, False
            self.ready[sel] = True
        else:
            new = (self.end_reason[sel] == 0) & (code > 0)
            self.end_reason[sel] = torch.where(new, code, self.end_reason[sel])
            self.end_step[sel] = torch.where(new, self.steps[sel], self.end_step[sel])

    # ---- API --------------------------------------------------------------------------------------------------
    def reset(self, idx=None, start=None, warmup: bool = False, spawn: bool = False, targets=None) -> WorldState:
        """Reset worlds idx (None = all; indices or bool mask [B]) to ``start`` (one row per selected world, in the
        order of idx). warmup=False: the exact state (rates as given = instantaneous; held targets ``targets``
        [n, 6], default start.q, matter for the reset-state contact forces only). warmup=True: Scene.reset (arm held
        at start.q, cube resting at its pose for 0.3 s; start rates ignored unless spawn: cube parked during the
        warm-up, then placed with cube_v, cube_w). Returns the state of all worlds."""
        if start is None:
            raise ValueError("reset needs a start state")
        if spawn and not warmup:
            raise ValueError("spawn needs warmup=True (the collector's probe start)")
        sel = self._select(idx)
        rows = pack_start(start, len(sel)).to(self.device)
        hold = rows[:, :6] if targets is None else torch.as_tensor(targets, dtype=torch.float64).to(self.device).reshape(-1, 6)
        cube = rows[:, 12:25].clone()
        if warmup:
            rows[:, 6:12] = 0.0
            rows[:, 19:25] = 0.0
            if spawn:
                rows[:, 12:15] = torch.as_tensor(self.scene.park, dtype=torch.float64, device=self.device)
                rows[:, 15:19] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64, device=self.device)
            hold = rows[:, :6]
        self._start_t[sel], self._hold_t[sel] = rows, hold
        self._run("_g_reset", self._body_reset)
        if warmup:
            self._warmup(sel, cube if spawn else None)
        self._after_place(sel, first=True)
        return self.state()

    def _warmup(self, sel, spawn_cube):
        """Collector warm-up (Scene.reset) of the selected worlds; the other worlds are put back afterwards."""
        d = self.d
        keep = [(a, wp.clone(a, device=self.wdev)) for a in (d.qpos, d.qvel, d.qacc_warmstart, d.ctrl)]
        time_t, ovf_t = wp.to_torch(d.time).clone(), self._overflow_t.clone()
        for _ in range(round(self.scene.cfg["warmup_s"] / CONTROL_DT)):
            self._run("_g_bare", self._body_bare)
        with wp.ScopedDevice(self.wdev), wp.ScopedStream(self.stream):
            for a, saved in keep:
                wp.launch(_keep_others, dim=a.shape, inputs=[self.mask, saved, a])
            wp.launch(_instant_row, dim=self.B, inputs=[self.mask, d.qpos, d.qvel, self.cq, self.cv, self.start])
        wp.synchronize_device(self.wdev)
        other = self._mask_t == 0
        tt = wp.to_torch(d.time)
        tt[other] = time_t[other]
        tt[~other] = 0.0
        self._overflow_t[other] = ovf_t[other]
        if spawn_cube is not None:
            self._start_t[sel, 12:25] = spawn_cube
        self._hold_t[sel] = self._ctrl_t[sel].double()
        self._run("_g_place", lambda: self._body_place(1))

    def set_cube(self, idx, pos, quat, v=None, w=None) -> WorldState:
        """Move the cube of worlds idx to pos [n, 3] (m), quat [n, 4] (w x y z, true orientation) with world velocity
        v and world spin w (default 0): a disturbance. The returned rows hold these (instantaneous) rates; the next
        step's step-average rates start from this pose (the jump itself is no rate). Flags are those of the new
        state (one forward)."""
        sel = self._select(idx)
        n = len(sel)
        f = lambda x, dim: (torch.zeros(n, dim, dtype=torch.float64) if x is None
                            else torch.as_tensor(x, dtype=torch.float64).reshape(n, dim)).to(self.device)
        row = self._out_t[sel].clone()
        row[:, 12:15], row[:, 15:19], row[:, 19:22], row[:, 22:25] = f(pos, 3), f(quat, 4), f(v, 3), f(w, 3)
        self._start_t[sel] = row
        self._run("_g_cube", lambda: self._body_place(0))
        self._after_place(sel, first=False)
        return self.state()

    def step(self, targets) -> WorldState:
        """One 20 ms command: targets [B, 6] absolute joint targets (rad, jaw last), clipped to the collector's
        applied range (joint limits minus 50 mrad) and cast to float32. Returns the new state of all worlds."""
        if not bool(self.ready.all()):
            raise RuntimeError("reset every world before stepping")
        targets = torch.as_tensor(targets)
        if targets.shape != (self.B, JOINTS):
            raise ValueError(f"targets must be [{self.B}, {JOINTS}]")
        self._targets_t.copy_(targets.to(device=self.device, dtype=torch.float64))
        self._run("_g_step", self._body_step)
        contact = self._flags_t[:, :3].bool()
        self.switch_contact = contact | self._last_phys
        self._last_phys = contact
        self.applied = self._ctrl_t.clone()
        self.steps += 1
        code = self._code_t.to(torch.int8)
        new = (self.end_reason == 0) & (code > 0)
        self.end_reason = torch.where(new, code, self.end_reason)
        self.end_step = torch.where(new, self.steps, self.end_step)
        return self.state()

    def state(self) -> WorldState:
        """Current WorldState of all worlds (copies, on ``device``)."""
        o, f = self._out_t, self._flags_t.bool()
        return WorldState(q=o[:, 0:6].clone(), qd=o[:, 6:12].clone(), cube_pos=o[:, 12:15].clone(),
                          cube_quat=o[:, 15:19].clone(), cube_v=o[:, 19:22].clone(), cube_w=o[:, 22:25].clone(),
                          contact=f[:, 0:3].clone(), arm_table=f[:, 3].clone(), invalid=f[:, 4].clone())

    def raw_state(self) -> torch.Tensor:
        """[B, 25] the collector's raw record of the current state: q, MuJoCo joint rates, cube pos, quat (as the
        state), MuJoCo cube velocity and spin rotated to the world frame (instantaneous rates; float32 physics)."""
        raw = self._out_t.clone()
        qv = self._qvel_t.double()
        raw[:, 6:12], raw[:, 19:22] = qv[:, 0:6], qv[:, self.cv:self.cv + 3]
        raw[:, 22:25] = quat_rotate(self._qpos_t[:, self.cq + 3:self.cq + 7].double(), qv[:, self.cv + 3:self.cv + 6])
        return raw

    @property
    def contact_force(self) -> torch.Tensor:
        """[B, 7] float64 largest summed normal force (N) per collector kind in the last command (after a reset or
        set_cube: at that state): fixed jaw-cube, moving jaw-cube, cube-table, arm-table, cube-other link, arm
        self, cube-base."""
        return self._fmax_t.reshape(self.B, NREC, NCLASS).amax(1).double()

    @property
    def overflow(self) -> torch.Tensor:
        """[B] int32 MuJoCo Warp overflow bitmask since each world's reset (mujoco_warp.OverflowType)."""
        return self._overflow_t.clone()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


MjwarpEnv = MjwarpWorlds


def state_fields() -> list[str]:
    return [f.name for f in fields(WorldState)]
