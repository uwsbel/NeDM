"""Batched CPU MuJoCo worlds of the SO101 grasp collector scene (spec 2.7): joint targets in, ``WorldState`` out.

The physics is the collector's (``nedm.contact_nrd.collect.so101_grasp.Scene``: Menagerie SO101, table, 30 mm cube,
0.5 ms implicitfast, elliptic cones, Newton solver, tolerance 1e-10). One ``step`` = one 20 ms command = 2 records of
20 physics steps, the command clipped to the collector's target range and cast to float32 (the applied value, as
``run_episode``). Worlds are split in contiguous blocks over worker processes (``workers`` > 0; numpy + mujoco only,
commands and arrays over pipes) or run in this process (``workers=0``). CPU MuJoCo is deterministic: the same starts
and commands give bit-identical states for any worker count. Units SI, world frame z up, table top z = 0, arm base at
the origin.

WorldState definitions = the grasp NRD data (``collect.so101_grasp.run_episode`` records, stored float32, then
``data.step_average_rates`` with the 20 ms model step; ``float32_records=False`` skips the float32 rounding):

- ``q``: joint angles (``qpos``) at the end of the command. ``qd``: 20 ms step average (q(t) - q(t - 20 ms)) / 0.02.
- ``cube_pos``; ``cube_quat`` w x y z: the TRUE orientation, unit (``export.quat_continuous``: w >= 0 at the reset
  record, then sign continuous over every 10 ms record).
- ``cube_v`` (p(t) - p(t - 20 ms)) / 0.02 and ``cube_w`` = rotation vector of q(t) q(t - 20 ms)^-1 / 0.02, both in
  the world frame (computed by ``data._step_average`` itself). After ``reset`` / ``set_cube`` the rates are the
  instantaneous ones (MuJoCo qvel, spin rotated to the world frame), as the first record of a data episode.
- ``contact`` [fixed jaw-cube, moving jaw-cube, cube-table]: the pair's summed normal force (first efc row of each
  elliptic contact = mj_contactForce normal) above ``contact_threshold_n`` (0.01 N) in any of the 40 physics steps of
  the command = OR of the data's two 10 ms interval labels = ``data.physical_labels`` of the step. The NRD switch
  view (this step or the one before, ``data.model_labels``) is ``switch_contact``. After a reset / set_cube: the
  forces of one ``mj_forward`` at that state.
- ``arm_table``: the same force rule for any arm geom (gripper, moving jaw, other links, base hulls) against the table.
- ``invalid``: any other collector end condition (``so101_grasp.TERMINAL``), checked as the collector does:
  cube_other_link (cube against shoulder / upper arm / lower arm / wrist / camera mount geoms), arm_self (arm-arm
  contact), cube_base (cube against the base hulls) - all with the force rule above; cube_off_table (cube centre x
  or y outside the table minus 5 cm, or z < -0.01 m, at the end of either 10 ms record); cube_visual_overlap (cube
  more than 1 mm inside the visual meshes of the gripper or the moving jaw beyond the collision penetration, at
  either record of the step); plus numerical failure (``mjWARN_BADQACC`` or non-finite qpos / qvel; the collector
  rejects such episodes).

Flags are per step (this command); physics always runs the whole command and never stops a world. ``end_reason``
[B] keeps the FIRST end since the reset (TERMINAL index, ``NUMERICAL`` = 7; collector precedence inside a record:
numerical, contact kinds in TERMINAL order, off table, overlap) and ``end_step`` the step that raised it; a collector
episode ends exactly there (the data's last grid state is the one before).

The visual-overlap grids need scipy once (``VisualOverlap``); they are cached as .npz (``overlap_cache_dir``) and
then load in any interpreter. Interpolation uses scipy's ``map_coordinates`` when present, else ``_map_linear``
(numpy trilinear, the same float32 result except within an ulp; only the 1 mm threshold decision depends on it).

    python -m nedm.so101_policy.mujoco_env overlap-cache     # build the overlap cache (needs scipy, about 2-6 s)
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

try:   # the worker side needs only numpy and mujoco (and runs in torch-free interpreters such as the env nrdsim)
    if os.environ.get("NEDM_MJ_WORKER") == "1":
        raise ImportError("worker process: no torch")
    import torch

    from nedm.so101_policy.world import CONTROL_DT, JOINTS, WorldState
except ImportError:
    torch = None
    CONTROL_DT, JOINTS = 0.02, 6

RAW = 25                      # raw record: q 6, qvel 6, cube pos 3, quat 4 (unit, sign continuous), v 3, w 3 (world)
NUMERICAL = 7                 # end code after so101_grasp.TERMINAL ("end", arm_table, ..., cube_visual_overlap)
ARM_TABLE = 1
CUBE_NAMES = ("x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz")
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SRC_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = Path(os.environ.get("NEDM_SO101_OVERLAP_CACHE", Path.home() / ".cache" / "nedm" / "so101_overlap"))
# bodies of the two data channels sets, as data._step_average reads them (system.json kinds and channel types)
_RATE_SYSTEM = {"bodies": [
    {"name": "arm", "kind": "articulated", "moving": True, "channel_types": ["angle"] * 6 + ["angular_rate"] * 6},
    {"name": "cube", "kind": "rigid", "moving": True,
     "channel_types": ["position"] * 3 + ["quaternion"] * 4 + ["velocity"] * 3 + ["angular_velocity"] * 3}]}
if 2 * 0.01 != CONTROL_DT:    # the data's step = stride x record step
    raise AssertionError("CONTROL_DT must equal 2 x 10 ms")


def end_names():
    from nedm.contact_nrd.collect.so101_grasp import TERMINAL
    return list(TERMINAL) + ["numerical"]


# ---- visual-overlap grids: cache and scipy-free interpolation ----------------------------------------------------
def _map_linear(input, coordinates, order=1, mode="nearest"):
    """scipy.ndimage.map_coordinates(input [nx, ny, nz], coordinates [3, N], order=1, mode="nearest") in numpy:
    trilinear interpolation, coordinates clamped to the grid; result in input's dtype."""
    if order != 1 or mode != "nearest":
        raise ValueError("only order=1, mode='nearest'")
    shape = np.asarray(input.shape)
    c = np.clip(np.asarray(coordinates, np.float64), 0.0, (shape - 1)[:, None])
    i0 = np.minimum(np.floor(c).astype(np.int64), np.maximum(shape - 2, 0)[:, None])
    t = c - i0
    out = np.zeros(c.shape[1])
    for a in (0, 1):
        wa = t[0] if a else 1.0 - t[0]
        for b in (0, 1):
            wb = t[1] if b else 1.0 - t[1]
            for e in (0, 1):
                we = t[2] if e else 1.0 - t[2]
                out += input[i0[0] + a, i0[1] + b, i0[2] + e] * (wa * wb * we)
    return out.astype(input.dtype)


def _overlap_key(s) -> str:
    """sha256 of everything the overlap grids depend on: the meshes and poses of the visual geoms of the two bodies,
    the overlap config and the source of VisualOverlap."""
    from nedm.contact_nrd.collect import so101_grasp as g
    m, h = s.m, hashlib.sha256()
    h.update(json.dumps(s.cfg["visual_overlap"], sort_keys=True).encode())
    h.update(inspect.getsource(g.VisualOverlap).encode())
    for b in s.overlap_bodies:
        for gid in np.flatnonzero(m.geom_bodyid == b):
            if m.geom_type[gid] != 7 or m.geom_contype[gid] or m.geom_conaffinity[gid]:   # 7 = mjGEOM_MESH
                continue
            k = m.geom_dataid[gid]
            v = m.mesh_vert[m.mesh_vertadr[k]: m.mesh_vertadr[k] + m.mesh_vertnum[k]]
            f = m.mesh_face[m.mesh_faceadr[k]: m.mesh_faceadr[k] + m.mesh_facenum[k]]
            for x in (v, f, m.geom_pos[gid], m.geom_quat[gid]):
                h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()[:24]


_FIELDS = ("lo", "hi", "h", "sdf", "cell", "ncell", "points", "start", "parity_disagreements")


def install_overlap(s, cache_dir=None, build=True):
    """Give Scene ``s`` its VisualOverlap grids from the cache (``cache_dir``, default $NEDM_SO101_OVERLAP_CACHE or
    ~/.cache/nedm/so101_overlap);
    if missing and ``build``: build them (scipy) and write the cache. -> path of the cache file."""
    from nedm.contact_nrd.collect import so101_grasp as g
    path = Path(cache_dir or DEFAULT_CACHE) / f"overlap_{_overlap_key(s)}.npz"
    try:
        from scipy.ndimage import map_coordinates
    except ImportError:
        map_coordinates = _map_linear
    if not path.exists():
        if not build:
            raise FileNotFoundError(path)
        try:
            import scipy  # noqa: F401
        except ImportError as e:
            raise RuntimeError(f"overlap cache {path} missing and no scipy here: run `python -m "
                               "nedm.so101_policy.mujoco_env overlap-cache` once with an interpreter that has scipy "
                               "(conda env nrdsim or kyle_madrona), or pass overlap=False") from e
        grids = s.overlap
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"{n}_{f}": np.asarray(getattr(o, f)) for n, o in enumerate(grids.values()) for f in _FIELDS}
        arrays["meshes"] = np.array(json.dumps([o.meshes for o in grids.values()]))
        tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(path)
    with np.load(path) as z:
        meshes = json.loads(str(z["meshes"]))
        grids = {}
        for n, b in enumerate(s.overlap_bodies):
            o = object.__new__(g.VisualOverlap)
            for f in _FIELDS:
                x = z[f"{n}_{f}"]
                setattr(o, f, x if x.ndim else x.item())
            o.map, o.meshes = map_coordinates, meshes[n]
            grids[b] = o
    s._overlap = grids
    return path


# ---- worker side: worlds of one block -----------------------------------------------------------------------------
class _Block:
    """n worlds in one process: one collector Scene (model, kinematics workspace ``dk``, overlap grids) and one MjData
    per world. All arrays numpy float64; the methods return dicts of arrays for the main process."""

    def __init__(self, n, overlap=True, cache_dir=None, overlap_margin=0.02, cpus=None, labels="exact"):
        if cpus:
            os.sched_setaffinity(0, cpus)
        import mujoco
        from nedm.contact_nrd.collect import so101_grasp as g
        self.mj, self.g = mujoco, g
        self.s = s = g.Scene(g.load_config())
        self.m = s.m
        self.data = [mujoco.MjData(s.m) for _ in range(n)]
        self.n = n
        self.kind = s.kind[s.cls[:, None], s.cls[None, :]]          # contact class of a geom pair, -1 none
        self.kind_list, self.ngeom = self.kind.ravel().tolist(), int(s.m.ngeom)
        self.thr = s.thr
        self.cube = s.cube_body
        self.prev_q = np.zeros((n, 4))                                # last recorded quaternion (sign continuity)
        self.nbad = np.zeros(n, np.int64)                             # BADQACC count at the last check
        self.overlap = bool(overlap)
        self.cache = install_overlap(s, cache_dir) if overlap else None
        self.margin = float(overlap_margin)
        if overlap:   # body-frame bounding boxes of the overlap grids, for the conservative prefilter
            self.boxes = [(b, o.lo - self.margin, o.hi + self.margin) for b, o in s._overlap.items()]
        self.badqacc = mujoco.mjtWarning.mjWARN_BADQACC
        if labels not in ("exact", "record"):
            raise ValueError("labels is 'exact' (forces after every physics step) or 'record' (once per 10 ms)")
        self.exact = labels == "exact"

    def info(self):
        return dict(pid=os.getpid(), cache=str(self.cache), n=self.n, python=sys.executable,
                    mujoco=self.mj.__version__, numpy=np.__version__, tlo=self.s.tlo.tolist(), thi=self.s.thi.tolist())

    # -- collector arithmetic --
    def _forces(self, d):
        """Summed normal force [7] by collector contact kind of the contacts in d (Scene.contact_forces()[0][:7])."""
        n = d.ncon
        c = d.contact
        a = c.efc_address[:n]
        f = np.where(a >= 0, d.efc_force[np.maximum(a, 0)], 0.0)
        k = self.kind[c.geom1[:n], c.geom2[:n]]
        ok = k >= 0
        return np.bincount(k[ok], weights=f[ok], minlength=7)

    def _forces_list(self, d):
        """_forces as a list, in a Python loop over the contacts: the same sums in the same order (bit-identical),
        about 3x cheaper for the few contacts of this scene (5.6 vs 14.8 us at 4 contacts, luffy)."""
        out = [0.0] * 7
        n = d.ncon
        if n:
            c, ef, kl, ng = d.contact, d.efc_force, self.kind_list, self.ngeom
            for (g1, g2), ad in zip(c.geom[:n].tolist(), c.efc_address[:n].tolist()):
                k = kl[g1 * ng + g2]
                if k >= 0 and ad >= 0:   # ad < 0: the collector adds 0.0
                    out[k] += float(ef[ad])
        return out

    def _near(self, d):
        """False if the cube is surely more than overlap_margin away from both overlap grids (bounding boxes in the
        body frames, poses of d: then visual_overlap is exactly 0)."""
        s = self.s
        rc, c = d.xmat[self.cube].reshape(3, 3), d.xpos[self.cube]
        for b, lo, hi in self.boxes:
            rb = d.xmat[b].reshape(3, 3)
            ext = s.half * np.abs(rb.T @ rc).sum(1)
            cb = (c - d.xpos[b]) @ rb
            if not (((cb + ext) < lo).any() or ((cb - ext) > hi).any()):
                return True
        return False

    def _overlap_now(self, d, fresh):
        """Visual overlap (m) of the current state of d. fresh: d holds the kinematics and contacts of its state
        (after mj_forward), as at the collector's record check; else they are recomputed in the workspace dk
        (Scene.overlap_at, the collector's last-record check)."""
        if not self.overlap or not self._near(d):
            return 0.0
        return self.s.visual_overlap(d) if fresh else self.s.overlap_at(d.qpos)

    def _record_end(self, d, i, bad, fresh):
        """End code after a record (0 = none), collector order; and the per-flag conditions."""
        s, g = self.s, self.g
        num = d.warning[self.badqacc].number != self.nbad[i] or not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        self.nbad[i] = d.warning[self.badqacc].number
        hit = bad > self.thr
        p = d.qpos[s.cq: s.cq + 3]
        off = bool((p[:2] < s.table_lo).any() or (p[:2] > s.table_hi).any() or p[2] < -0.01)
        ov = self._overlap_now(d, fresh) > s.overlap_thr
        code = NUMERICAL if num else 1 + int(np.argmax(hit)) if hit.any() else (
            g.TERMINAL.index("cube_off_table") if off else g.TERMINAL.index("cube_visual_overlap") if ov else 0)
        return code, bool(hit[0]), bool(num or hit[1:].any() or off or ov)

    def _record(self, d, i, first=False):
        """Raw record [25] of d (Scene.state, quaternion as export.quat_continuous)."""
        s = self.s
        s.d = d
        arm, cube = s.state()
        q = cube[3:7]
        q = q / np.linalg.norm(q[None], axis=-1, keepdims=True)[0]
        if (first and q[0] < 0) or (not first and np.dot(self.prev_q[i], q) < 0):
            q = -q
        self.prev_q[i] = q
        cube[3:7] = q
        return np.concatenate((arm, cube))

    def _now(self, d, i, first):
        """Outputs of a state without a step (after a reset or a cube move; d after mj_forward)."""
        f = self._forces(d) if d.ncon else np.zeros(7)
        if first:
            self.nbad[i] = 0
        code, at, inv = self._record_end(d, i, f[3:7], fresh=True)
        return self._record(d, i, first), f[:3] > self.thr, at, inv, code, f

    def _pack(self, rows):
        raw, lab, at, inv, code, force = zip(*rows)
        return dict(raw=np.array(raw), contact=np.array(lab, bool), arm_table=np.array(at, bool),
                    invalid=np.array(inv, bool), code=np.array(code, np.int8), force=np.array(force))

    # -- API --
    def reset(self, rows, start, ctrl=None, warmup=False, spawn=False):
        """Worlds ``rows`` to the packed starts [n, 25] (raw layout). warmup: Scene.reset (arm targets = q, cube
        resting at its pose, or parked and then placed with its v, w for spawn; 0.3 s; time 0). Else the exact
        state: qpos / qvel from the start (cube spin world -> body), targets ``ctrl`` (default q), mj_forward."""
        mj, s = self.mj, self.s
        out = []
        for k, i in enumerate(rows):
            d, x = self.data[i], start[k]
            s.d = d
            q = x[15:19] / np.linalg.norm(x[15:19])
            if warmup:
                s.reset(x[0:6], x[12:15], x[15:19], x[19:22], x[22:25], bool(spawn))   # quaternion as given
            else:
                mj.mj_resetData(s.m, d)
                d.qpos[s.qadr], d.qvel[s.vadr] = x[0:6], x[6:12]
                d.qpos[s.cq: s.cq + 3], d.qpos[s.cq + 3: s.cq + 7] = x[12:15], q
                d.qvel[s.cv: s.cv + 3] = x[19:22]
                d.qvel[s.cv + 3: s.cv + 6] = self.g.quat_mat(q).T @ x[22:25]
                u = x[0:6] if ctrl is None else ctrl[k]
                d.ctrl[s.act] = np.clip(u, s.tlo, s.thi).astype(np.float32)
                d.time = 0.0
                mj.mj_forward(s.m, d)
            out.append(self._now(d, i, first=True))
        return self._pack(out)

    def set_cube(self, rows, pose):
        """Move the cube of worlds ``rows``: pose [n, 13] = pos, quat (w x y z), world v, world w; mj_forward."""
        mj, s = self.mj, self.s
        out = []
        for k, i in enumerate(rows):
            d, x = self.data[i], pose[k]
            q = x[3:7] / np.linalg.norm(x[3:7])
            d.qpos[s.cq: s.cq + 3], d.qpos[s.cq + 3: s.cq + 7] = x[0:3], q
            d.qvel[s.cv: s.cv + 3] = x[7:10]
            d.qvel[s.cv + 3: s.cv + 6] = self.g.quat_mat(q).T @ x[10:13]
            mj.mj_forward(s.m, d)
            out.append(self._now(d, i, first=False))
        return self._pack(out)

    def step(self, targets):
        """One 20 ms command for every world of the block: targets [n, 6] -> raw records, flags, end codes."""
        mj, s, m = self.mj, self.s, self.m
        u_all = np.clip(targets, s.tlo, s.thi).astype(np.float32)   # stored and applied (run_episode)
        out = []
        for i, d in enumerate(self.data):
            u = u_all[i]
            lab, at, inv, code = np.zeros(3, bool), False, False, 0
            force = np.zeros(7)
            for r in range(s.per_ctrl):
                d.ctrl[s.act] = u
                if self.exact:
                    mx = [0.0] * 7   # largest summed force per kind over the record's physics steps
                    for _ in range(s.nsub):
                        mj.mj_step(m, d)
                        if d.ncon:
                            mx = [a if a >= b else b for a, b in zip(mx, self._forces_list(d))]
                else:   # same physics; forces of the last physics step of the record only
                    mj.mj_step(m, d, nstep=s.nsub)
                    mx = self._forces_list(d)
                mx = np.array(mx)
                fmax, bad = mx[:3], mx[3:7]
                lab |= fmax > self.thr
                force = np.maximum(force, np.concatenate((fmax, bad)))
                c, a, v = self._record_end(d, i, bad, fresh=False)
                code, at, inv = code or c, at or a, inv or v
                if r + 1 < s.per_ctrl:
                    self._record(d, i)   # the mid-step record: quaternion sign continuity only
            out.append((self._record(d, i), lab, at, inv, code, force))
        return self._pack(out)


def _enc(x):
    if isinstance(x, np.ndarray):
        return ("__nd__", x.dtype.str, x.shape, x.tobytes())
    if isinstance(x, dict):
        return {k: _enc(v) for k, v in x.items()}
    return x


def _dec(x):
    if isinstance(x, tuple) and len(x) == 4 and x[0] == "__nd__":
        return np.frombuffer(x[3], dtype=x[1]).reshape(x[2]).copy()
    if isinstance(x, dict):
        return {k: _dec(v) for k, v in x.items()}
    return x


def _serve():
    """Worker process: pickled (method, kwargs) on stdin -> pickled result on the original stdout (MuJoCo and
    other prints go to stderr)."""
    import faulthandler
    import traceback
    faulthandler.enable()   # a crash in MuJoCo prints the Python stack to stderr
    inp = sys.stdin.buffer
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    block = None
    while True:
        try:
            cmd, kw = pickle.load(inp)
        except EOFError:
            return
        if cmd == "close":
            return
        try:
            kw = _dec(kw)
            if cmd == "init":
                block = _Block(**kw)
                res = block.info()
            else:
                res = getattr(block, cmd)(**kw)
            msg = ("ok", _enc(res))
        except Exception:   # noqa: BLE001 - sent to the main process
            msg = ("error", traceback.format_exc())
        pickle.dump(msg, out, protocol=4)
        out.flush()


class _Proc:
    """A worker process running _serve()."""

    def __init__(self, python):
        code = f"import sys; sys.path.insert(0, {str(SRC_ROOT)!r}); from nedm.so101_policy.mujoco_env import _serve; _serve()"
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NEDM_MJ_WORKER="1")
        # -P: no current directory on sys.path (python >= 3.11; every interpreter of this study is 3.12)
        self.p = subprocess.Popen([python, "-P", "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)

    def send(self, cmd, **kw):
        pickle.dump((cmd, _enc(kw)), self.p.stdin, protocol=4)
        self.p.stdin.flush()

    def recv(self):
        try:
            status, res = pickle.load(self.p.stdout)
        except EOFError:
            raise RuntimeError(f"MuJoCo worker {self.p.pid} died (exit {self.p.poll()}); see its stderr") from None
        if status != "ok":
            raise RuntimeError(f"MuJoCo worker {self.p.pid} failed:\n{res}")
        return _dec(res)

    def close(self):
        if self.p.poll() is None:
            try:
                self.send("close")
                self.p.wait(timeout=10)
            except Exception:   # noqa: BLE001
                self.p.kill()
                self.p.wait()
        for f in (self.p.stdin, self.p.stdout):
            try:
                f.close()
            except Exception:   # noqa: BLE001
                pass


class _Local:
    """The same interface in this process (workers=0)."""

    def __init__(self):
        self.block, self.res = None, None

    def send(self, cmd, **kw):
        if cmd == "init":
            self.block = _Block(**kw)
            self.res = self.block.info()
        elif cmd != "close":
            self.res = getattr(self.block, cmd)(**kw)

    def recv(self):
        return self.res

    def close(self):
        pass


# ---- main side ----------------------------------------------------------------------------------------------------
def _get(start, name):
    return start.get(name) if isinstance(start, dict) else getattr(start, name, None)


def pack_start(start, n=None) -> np.ndarray:
    """WorldState-like (attributes or dict keys q, qd, cube_pos, cube_quat, cube_v, cube_w; qd, cube_v, cube_w may be
    missing = 0) -> raw rows [n, 25] float64 (numpy)."""
    q = np.asarray(_tonp(_get(start, "q")), np.float64)
    n = q.reshape(-1, JOINTS).shape[0] if n is None else n
    parts = []
    for k, dim in (("q", 6), ("qd", 6), ("cube_pos", 3), ("cube_quat", 4), ("cube_v", 3), ("cube_w", 3)):
        x = _get(start, k)
        parts.append(np.zeros((n, dim)) if x is None else np.asarray(_tonp(x), np.float64).reshape(n, dim))
    return np.concatenate(parts, 1)


def _tonp(x):
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().double().numpy()
    return x


def step_rates(prev: "torch.Tensor", cur: "torch.Tensor"):
    """20 ms step-average rates between two raw records [B, 25] (float64 tensors) by the data pipeline's own code
    (data._step_average): -> (qd [B, 6], cube_v [B, 3], cube_w [B, 3])."""
    from nedm.contact_nrd.data import _step_average
    st = torch.zeros(len(cur), 2, 2, 13, dtype=torch.float64)
    st[:, 0, 0, :12], st[:, 1, 0, :12] = prev[:, :12], cur[:, :12]
    st[:, 0, 1], st[:, 1, 1] = prev[:, 12:], cur[:, 12:]
    out = _step_average(st, {"stride": 1, "dt": CONTROL_DT, "system": _RATE_SYSTEM}, {"articulated", "rigid"})[:, 1]
    return out[:, 0, 6:12], out[:, 1, 7:10], out[:, 1, 10:13]


class MujocoWorlds:
    """``num_envs`` worlds of the collector scene in CPU MuJoCo, split over ``workers`` processes (0 = here).

    ``reset(idx, start, warmup=False)`` places worlds at a WorldState-like start (``warmup=True``: the collector's
    0.3 s warm-up first, arm held at ``start.q``, cube resting at its pose; ``spawn=True``: cube parked during the
    warm-up, then placed with ``cube_v``, ``cube_w``, as the collector's probes); ``step(targets [B, 6])`` runs one
    20 ms command; ``set_cube(idx, pos, quat, v, w)`` moves the cube (disturbance). All return the WorldState of all
    worlds (float64, on ``device``; module docstring for the definitions). Attributes: ``end_reason`` / ``end_step``
    [B] (first end since the reset, ``end_names()``), ``switch_contact`` [B, 3] (NRD switch view), ``contact_force``
    [B, 7] (largest summed normal force per collector kind in the last command, N), ``applied`` [B, 6] float32
    targets of the last command, ``steps`` [B] commands since the reset.

    ``python``: interpreter of the workers (default $NEDM_MJ_PYTHON or this one; needs numpy and mujoco, scipy only to
    build the overlap cache). ``overlap``: the visual-overlap end check. ``cpus``: CPU affinity of the workers (e.g.
    to keep off a faulty core; newton 2026-10-11: logical CPUs 10-11 crash, use [0..9, 12..31]). ``labels``:
    "exact" (contact forces after every 0.5 ms physics step, the data rule) or "record" (once per 10 ms record:
    the same physics and states, but contacts and force-based ends shorter than a record can be missed)."""

    def __init__(self, num_envs: int, workers: int = 0, device="cpu", overlap: bool = True,
                 float32_records: bool = True, python: str | None = None, overlap_cache_dir=None, cpus=None,
                 labels: str = "exact"):
        if torch is None:
            raise ImportError("MujocoWorlds needs torch in the main process")
        self.B, self.device, self.float32_records = int(num_envs), torch.device(device), bool(float32_records)
        nw = max(0, min(int(workers), self.B))
        cuts = np.linspace(0, self.B, (nw or 1) + 1).round().astype(int)
        self.blocks = [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:])]
        python = python or os.environ.get("NEDM_MJ_PYTHON") or sys.executable
        self.procs = [_Proc(python) for _ in self.blocks] if nw else [_Local()]
        kw = dict(overlap=overlap, cache_dir=None if overlap_cache_dir is None else str(overlap_cache_dir), labels=labels,
                  cpus=None if cpus is None or not nw else sorted(int(c) for c in cpus))
        try:   # the first worker builds the overlap cache if needed, then the others load it
            self.procs[0].send("init", n=self.blocks[0][1] - self.blocks[0][0], **kw)
            self.info = [self.procs[0].recv()]
            for p, (a, b) in zip(self.procs[1:], self.blocks[1:]):
                p.send("init", n=b - a, **kw)
            self.info += [p.recv() for p in self.procs[1:]]
        except BaseException:
            self.close()
            raise
        self.tlo, self.thi = np.array(self.info[0]["tlo"]), np.array(self.info[0]["thi"])   # applied-target range
        B, f64 = self.B, torch.float64
        self._raw = torch.zeros(B, RAW, dtype=f64)
        self._ws = WorldState(q=torch.zeros(B, 6, dtype=f64), qd=torch.zeros(B, 6, dtype=f64),
                              cube_pos=torch.zeros(B, 3, dtype=f64), cube_quat=torch.zeros(B, 4, dtype=f64),
                              cube_v=torch.zeros(B, 3, dtype=f64), cube_w=torch.zeros(B, 3, dtype=f64),
                              contact=torch.zeros(B, 3, dtype=torch.bool), arm_table=torch.zeros(B, dtype=torch.bool),
                              invalid=torch.zeros(B, dtype=torch.bool))
        self.ready = torch.zeros(B, dtype=torch.bool)
        self.end_reason = torch.zeros(B, dtype=torch.int8)
        self.end_step = torch.full((B,), -1, dtype=torch.long)
        self.steps = torch.zeros(B, dtype=torch.long)
        self.switch_contact = torch.zeros(B, 3, dtype=torch.bool)
        self._last_phys = torch.zeros(B, 3, dtype=torch.bool)
        self.contact_force = torch.zeros(B, 7, dtype=f64)
        self.applied = torch.zeros(B, 6, dtype=torch.float32)

    # -- plumbing --
    def _idx(self, idx) -> np.ndarray:
        if idx is None:
            return np.arange(self.B)
        x = np.asarray(_tonp(idx) if not (torch is not None and torch.is_tensor(idx)) else idx.cpu().numpy())
        return np.flatnonzero(x) if x.dtype == bool else x.astype(np.int64).reshape(-1)

    def _scatter(self, cmd, idx, arrays, **const):
        """Send cmd to the blocks owning idx (rows of ``arrays`` follow idx; ``const`` go as they are) -> (idx in
        block order, merged result)."""
        if len(np.unique(idx)) != len(idx) or (len(idx) and (idx.min() < 0 or idx.max() >= self.B)):
            raise ValueError("world indices must be unique and inside 0 .. B-1")
        order, calls = [], []
        for p, (a, b) in zip(self.procs, self.blocks):
            sel = np.flatnonzero((idx >= a) & (idx < b))
            if len(sel):
                p.send(cmd, rows=(idx[sel] - a).tolist(), **{k: (None if v is None else v[sel]) for k, v in arrays.items()},
                       **const)
                calls.append(p)
                order.append(sel)
        res = [p.recv() for p in calls]
        sel = np.concatenate(order)
        return idx[sel], {k: np.concatenate([r[k] for r in res]) for k in res[0]}

    def _round(self, raw: np.ndarray) -> torch.Tensor:
        x = raw.astype(np.float32) if self.float32_records else raw
        return torch.from_numpy(x.astype(np.float64))

    def _place(self, idx, res, first: bool):
        """State rows idx after a reset / cube move: instantaneous rates of the raw record."""
        t = torch.from_numpy(idx)
        raw = self._round(res["raw"])
        self._raw[t] = raw
        ws = self._ws
        ws.q[t], ws.qd[t], ws.cube_pos[t], ws.cube_quat[t] = raw[:, 0:6], raw[:, 6:12], raw[:, 12:15], raw[:, 15:19]
        ws.cube_v[t], ws.cube_w[t] = raw[:, 19:22], raw[:, 22:25]
        ws.contact[t] = torch.from_numpy(res["contact"])
        ws.arm_table[t], ws.invalid[t] = torch.from_numpy(res["arm_table"]), torch.from_numpy(res["invalid"])
        self.contact_force[t] = torch.from_numpy(res["force"])
        code = torch.from_numpy(res["code"])
        if first:
            self.end_reason[t], self.end_step[t] = code, torch.where(code > 0, 0, -1)
            self.steps[t], self._last_phys[t], self.switch_contact[t] = 0, False, False
            self.ready[t] = True
        else:
            new = (self.end_reason[t] == 0) & (code > 0)
            self.end_reason[t] = torch.where(new, code, self.end_reason[t])
            self.end_step[t] = torch.where(new, self.steps[t], self.end_step[t])

    # -- API --
    def reset(self, idx=None, start=None, warmup: bool = False, spawn: bool = False, targets=None) -> WorldState:
        """Reset worlds idx (None = all; indices or bool mask [B]) to ``start`` (one row per selected world).
        warmup=False: the exact state (rates as instantaneous MuJoCo velocities; held targets ``targets`` [n, 6],
        default start.q). warmup=True: Scene.reset (start rates ignored unless spawn). Returns the state of all
        worlds."""
        if start is None:
            raise ValueError("reset needs a start state")
        idx = self._idx(idx)
        rows = pack_start(start, len(idx))
        ctrl = None if targets is None else np.asarray(_tonp(targets), np.float64).reshape(len(idx), JOINTS)
        if spawn and not warmup:
            raise ValueError("spawn needs warmup=True (the collector's probe start)")
        order, res = self._scatter("reset", idx, dict(start=rows, ctrl=ctrl), warmup=bool(warmup), spawn=bool(spawn))
        self._place(order, res, first=True)
        return self.state()

    def set_cube(self, idx, pos, quat, v=None, w=None) -> WorldState:
        """Move the cube of worlds idx to pos [n, 3] (m), quat [n, 4] (w x y z, true orientation) with world velocity
        v and world spin w (default 0): a disturbance. The returned rows hold these (instantaneous) rates; the next
        step's step-average rates start from this pose (the jump itself is no rate). Flags and contacts are those of
        the new state (one mj_forward); a cube placed into the gripper meshes is ``invalid`` at once."""
        idx = self._idx(idx)
        n = len(idx)
        z = np.zeros((n, 3))
        pose = np.concatenate([np.asarray(_tonp(x), np.float64).reshape(n, k) if x is not None else z
                               for x, k in ((pos, 3), (quat, 4), (v, 3), (w, 3))], 1)
        order, res = self._scatter("set_cube", idx, dict(pose=pose))
        self._place(order, res, first=False)
        return self.state()

    def step(self, targets) -> WorldState:
        """One 20 ms command: targets [B, 6] absolute joint targets (rad, jaw last), clipped to the collector's
        applied range (joint limits minus 50 mrad) and cast to float32. Returns the new state of all worlds."""
        if not bool(self.ready.all()):
            raise RuntimeError("reset every world before stepping")
        u = np.asarray(_tonp(targets), np.float64).reshape(self.B, JOINTS)
        for p, (a, b) in zip(self.procs, self.blocks):
            p.send("step", targets=u[a:b])
        res = [p.recv() for p in self.procs]
        res = {k: np.concatenate([r[k] for r in res]) for k in res[0]}
        cur = self._round(res["raw"])
        qd, v, w = step_rates(self._raw, cur)
        self._raw = cur
        ws = self._ws
        ws.q, ws.qd, ws.cube_pos, ws.cube_quat = cur[:, 0:6].clone(), qd, cur[:, 12:15].clone(), cur[:, 15:19].clone()
        ws.cube_v, ws.cube_w = v, w
        ws.contact = torch.from_numpy(res["contact"])
        ws.arm_table, ws.invalid = torch.from_numpy(res["arm_table"]), torch.from_numpy(res["invalid"])
        self.switch_contact = ws.contact | self._last_phys
        self._last_phys = ws.contact.clone()
        self.contact_force = torch.from_numpy(res["force"])
        self.applied = torch.from_numpy(np.clip(u, self.tlo, self.thi).astype(np.float32))
        self.steps += 1
        code = torch.from_numpy(res["code"])
        new = (self.end_reason == 0) & (code > 0)
        self.end_reason = torch.where(new, code, self.end_reason)
        self.end_step = torch.where(new, self.steps, self.end_step)
        return self.state()

    def state(self) -> WorldState:
        """Current WorldState of all worlds (copies, on ``device``)."""
        return WorldState(**{f.name: getattr(self._ws, f.name).to(self.device, copy=True) for f in fields(WorldState)})

    def raw_state(self) -> torch.Tensor:
        """[B, 25] the collector's raw record of the current state (instantaneous rates, cube spin in the world
        frame; float32-rounded when float32_records)."""
        return self._raw.clone()

    def close(self):
        for p in getattr(self, "procs", []):
            p.close()
        self.procs = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:   # noqa: BLE001
            pass


_SCENE = None


def _scene():
    """One collector Scene of this process (kinematics, planning, limits; no overlap grids)."""
    global _SCENE
    if _SCENE is None:
        from nedm.contact_nrd.collect import so101_grasp as g
        _SCENE = g.Scene(g.load_config())
    return _SCENE


def task_starts(seed: int, n: int, spread_deg: float = 40.0) -> dict:
    """n collector task starts (so101_grasp.task_start: cube flat at rest on the annulus r 0.15-0.27 m, +-75 deg;
    collision-free top-down start pose 8-20 cm up within spread_deg of the cube direction); world k draws from
    numpy default_rng([seed, k]), so a start depends only on (seed, k). -> dict q [n, 6], cube_pos [n, 3],
    cube_quat [n, 4] (numpy float64) for reset(..., warmup=True)."""
    from nedm.contact_nrd.collect import so101_grasp as g
    s = _scene()
    q, pos, quat = np.zeros((n, 6)), np.zeros((n, 3)), np.zeros((n, 4))
    for k in range(n):
        rng = np.random.default_rng([int(seed), k])
        cube, cyaw, q0, _ = g.task_start(s, rng, spread_deg)
        c = g.resting_cube(cube, cyaw)
        q[k], pos[k], quat[k] = q0, c["pos"], c["quat"]
    return dict(q=q, cube_pos=pos, cube_quat=quat)


# ---- NRD layout ---------------------------------------------------------------------------------------------------
def _body_cols(model):
    """Moving bodies of the model -> (arm slot, cube slot, arm q cols, arm qd cols, cube cols) by channel name."""
    mov = [b for b in model.system["bodies"] if b["moving"]]
    ia = next(n for n, b in enumerate(mov) if b.get("kind") == "articulated")
    ic = next(n for n, b in enumerate(mov) if b.get("kind") == "rigid")
    ach, cch = mov[ia]["channels"], mov[ic]["channels"]
    return (ia, ic, [ach.index(f"q_{j}") for j in JOINT_NAMES], [ach.index(f"qd_{j}") for j in JOINT_NAMES],
            [cch.index(c) for c in CUBE_NAMES])


def nrd_state(ws: WorldState, model) -> torch.Tensor:
    """WorldState -> NRD state [B, D, S] float64 (on ws's device): the 12 + 13 base channels placed by channel name
    from model.system, then model.finish (unit quaternion, FK and relative channels if the model has them). The
    cube orientation is the given (true) one: the NRD sim canonicalises it, not this function."""
    ia, ic, qc, qdc, cc = _body_cols(model)
    B = ws.q.shape[0]
    x = torch.zeros(B, model.n_bodies, model.smax, dtype=torch.float64, device=ws.q.device)
    x[:, ia, qc], x[:, ia, qdc] = ws.q.double(), ws.qd.double()
    x[:, ic, cc] = torch.cat((ws.cube_pos, ws.cube_quat, ws.cube_v, ws.cube_w), -1).double()
    return model.finish(x)


def world_from_nrd(state: torch.Tensor, model) -> WorldState:
    """NRD state [B, D, S] -> WorldState (contact, arm_table, invalid False): a start for MujocoWorlds.reset."""
    ia, ic, qc, qdc, cc = _body_cols(model)
    c = state[:, ic, cc].double()
    B = state.shape[0]
    f = torch.zeros(B, dtype=torch.bool, device=state.device)
    return WorldState(q=state[:, ia, qc].double(), qd=state[:, ia, qdc].double(), cube_pos=c[:, 0:3],
                      cube_quat=c[:, 3:7], cube_v=c[:, 7:10], cube_w=c[:, 10:13],
                      contact=torch.zeros(B, 3, dtype=torch.bool, device=state.device), arm_table=f, invalid=f.clone())


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("overlap-cache", help="build the visual-overlap grid cache (needs scipy)")
    c.add_argument("--dir", default=None)
    a = p.parse_args(argv)
    if a.cmd == "overlap-cache":
        from nedm.contact_nrd.collect import so101_grasp as g
        s = g.Scene(g.load_config())
        print(install_overlap(s, a.dir))


if __name__ == "__main__":
    main()
