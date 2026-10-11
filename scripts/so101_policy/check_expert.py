"""Closed-loop check of the pick-and-place expert (``nedm.so101_policy.expert``) in CPU MuJoCo (the collector's physics)
and in the grasp NRD, on the same tasks; IK parity against the collector; command statistics of the decoder.

    PYTHONPATH=src python -P scripts/so101_policy/check_expert.py tasks --data DIR --out tasks.npz [--n 256] [--seed0 1000]
    PYTHONPATH=src python -P scripts/so101_policy/check_expert.py mujoco --tasks tasks.npz --out mj.npz [--procs 8]
    PYTHONPATH=src python -P scripts/so101_policy/check_expert.py nrd --tasks tasks.npz --ckpt CKPT --out nrd.npz [--device cuda]
    PYTHONPATH=src python -P scripts/so101_policy/check_expert.py overlap --mj mj.npz --out ov.npz [--procs 8]   # env with scipy
    PYTHONPATH=src python -P scripts/so101_policy/check_expert.py ikparity --out ik.json [--n 1000] [--procs 8]
    PYTHONPATH=src python -P scripts/so101_policy/check_expert.py smooth --data DIR --mj mj.npz --tasks tasks.npz --out sm.json
    PYTHONPATH=src python -P scripts/so101_policy/check_expert.py report --tasks tasks.npz --mj mj.npz --nrd A.npz B.npz \
        [--overlap ov.npz] --out report.json

Tasks (spec 2.6, written down here): task i uses numpy seed ``seed0 + i`` (validation range 1000-1999). Start = the first
recorded state of a random production_plus split-0 task-family episode (families 0-6; never the sealed test). Goal =
cube centre (x, y) on the workspace annulus (r 0.15-0.27 m about the pan axis, angle +-75 deg), 6-30 cm from the start
(rejection sampling), goal yaw = start yaw + U(-30, 30) deg (mod 90 deg). Horizon 400 steps (8 s).

Episode (both worlds, same harness): the expert acts every 20 ms on the true state; the episode ends at the first of
(a) the world's end condition (MuJoCo: the collector's contact ends arm-table / cube-other-link / arm-self / cube-base,
cube off the table, numerical failure; NRD: lowest collision point of gripper or moving jaw < 0.5 mm = arm-table,
non-finite state, cube off the table), (b) a drop: cube bottom > 30 mm above the table and no jaw contact in 2
consecutive steps, (c) the expert done for 5 steps (0.1 s), (d) 400 steps. Success at the end: no failure end, cube
resting (|v| < 5 mm/s), flat (tilt < 5 deg), xy error <= 15 mm and yaw error (mod 90) <= 15 deg (also 10 / 10), no jaw
contact, grasp point >= 30 mm above the cube top. Contacts: MuJoCo = pair force above 0.01 N in either 10 ms record of
the step (the collector's label); NRD = the collision switch of the step (logit >= 0; it covers this step and the one
before). The MuJoCo loop has no visual-overlap end (the collector's mesh check needs scipy); ``overlap`` replays the
recorded 10 ms states in an env with scipy and reports where the collector would have ended.

NRD stepping: ``ContactNRD.details`` (sparse) from a single-state history; after every step the cube orientation of
the whole history is relabelled by one of the 24 cube rotations (the one nearest a flat cube at yaw 45 deg, chosen on
the newest state; ~/so101_policy/scripts/critic/symmetry_canon.py rule) and ``finish`` recomputes the derived and
relative channels; the true orientation is tracked as canonical * conj(accumulated relabelling).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

PAN = np.array([0.0388, 0.0])
TASK_FAMILIES = range(7)
HORIZON = 400
DONE_HOLD = 5
DROP_Z = 0.03
GUARD_M = 0.0005
CUBE_HALF = 0.015


# ---- tasks ------------------------------------------------------------------------------------------------------
def annulus(rng):
    r, a = rng.uniform(0.15, 0.27), math.radians(rng.uniform(-75, 75))
    return PAN + r * np.array([math.cos(a), math.sin(a)])


def quat_yaw90(q):
    """Yaw (mod 90 deg) of a flat cube from w x y z (numpy)."""
    from nedm.so101_policy.expert import cube_yaw_tilt
    return cube_yaw_tilt(torch.as_tensor(np.asarray(q, np.float64)).reshape(-1, 4))[0].numpy()


def cmd_tasks(a):
    from nedm.contact_nrd.rollouts import npz_members
    f = npz_members(Path(a.data) / "unified_data.npz")
    fam, split = np.asarray(f["family"]), np.asarray(f["splits"])
    cand = np.flatnonzero((split == 0) & np.isin(fam, list(TASK_FAMILIES)))
    rows, goals, gyaw, seeds = [], [], [], []
    for i in range(a.n):
        seed = a.seed0 + i
        rng = np.random.default_rng(seed)
        row = int(rng.choice(cand))
        cube0 = np.asarray(f["cube"][row, 0], np.float64)
        for _ in range(1000):
            g = annulus(rng)
            if 0.06 <= np.linalg.norm(g - cube0[:2]) <= 0.30:
                break
        else:
            raise RuntimeError("no goal")
        y0 = float(quat_yaw90(cube0[3:7])[0])
        rows.append(row), goals.append(g), seeds.append(seed)
        gyaw.append((y0 + math.radians(rng.uniform(-30, 30))) % (math.pi / 2))
    rows = np.array(rows)
    order = np.argsort(rows)   # memmap reads in order
    arm0, cube0, act0 = (np.empty((a.n, k)) for k in (12, 13, 6))
    for j in order:
        arm0[j], cube0[j], act0[j] = f["arm"][rows[j], 0], f["cube"][rows[j], 0], f["action"][rows[j], 0]
    np.savez(a.out, rows=rows, seeds=np.array(seeds), family=fam[rows], arm0=arm0, cube0=cube0, act0=act0,
             goal_xy=np.array(goals), goal_yaw=np.array(gyaw), data=str(a.data))
    d = np.linalg.norm(np.array(goals) - cube0[:, :2], axis=1)
    print(f"{a.n} tasks -> {a.out}; families {np.bincount(fam[rows], minlength=7).tolist()}, goal distance "
          f"p50 {np.median(d) * 1e3:.0f} mm (min {d.min() * 1e3:.0f}, max {d.max() * 1e3:.0f})")


def load_tasks(path):
    t = np.load(path)
    return {k: t[k] for k in t.files}


# ---- shared episode harness ---------------------------------------------------------------------------------------
class Harness:
    """Expert + decoder over B worlds; end rules and per-step records (numpy)."""

    def __init__(self, tasks, idx, device="cpu"):
        from nedm.so101_policy.decoder import CommandDecoder
        from nedm.so101_policy.expert import PickPlaceExpert
        from nedm.so101_policy.kin import SO101Kin
        self.idx = np.asarray(idx)
        self.B = len(self.idx)
        self.kin = SO101Kin(device)
        self.dec = CommandDecoder(self.B, device, kin=self.kin)
        self.ex = PickPlaceExpert(self.dec)
        self.goal_xy = torch.as_tensor(tasks["goal_xy"][self.idx], device=device)
        self.goal_yaw = torch.as_tensor(tasks["goal_yaw"][self.idx], device=device)
        self.ended = np.zeros(self.B, bool)
        self.length = np.full(self.B, HORIZON, np.int64)
        self.reason = np.array(["timeout"] * self.B, dtype=object)
        self.done_run = np.zeros(self.B, np.int64)
        self.drop_run = np.zeros(self.B, np.int64)
        self.rec = {k: [] for k in ("q", "cube", "target", "phase", "contact", "action")}

    def reset(self, state):
        all_ = torch.arange(self.B, device=self.dec.dev)
        self.dec.reset(all_, state.q)
        self.ex.reset(all_, state, self.goal_xy, self.goal_yaw)
        self.last = state

    def act(self, state):
        a = self.ex.act(state)
        tgt = self.dec.step(a)
        self.rec["action"].append(a.cpu().numpy())
        self.rec["target"].append(tgt.cpu().numpy())
        return tgt

    def observe(self, t, state, world_end, world_reason):
        """After step t (0-based): record, then apply the end rules to worlds still running."""
        from nedm.so101_policy.expert import PH
        s = state.to("cpu")
        q, cube = s.q.numpy(), torch.cat((s.cube_pos, s.cube_quat), -1).numpy()
        contact = s.contact.numpy()
        self.rec["q"].append(q), self.rec["cube"].append(cube), self.rec["contact"].append(contact)
        self.rec["phase"].append(self.ex.phase.cpu().numpy())
        live = ~self.ended
        fail = live & world_end
        self.reason[fail] = world_reason[fail]
        up = (cube[:, 2] - CUBE_HALF > DROP_Z) & ~contact[:, 0] & ~contact[:, 1]
        self.drop_run = np.where(up, self.drop_run + 1, 0)
        drop = live & ~fail & (self.drop_run >= 2)
        self.reason[drop] = "dropped"
        done = self.ex.phase.cpu().numpy() == PH["done"]
        self.done_run = np.where(done, self.done_run + 1, 0)
        fin = live & ~fail & ~drop & (self.done_run >= DONE_HOLD)
        self.reason[fin] = "done"
        new = fail | drop | fin
        self.length[new] = t + 1
        self.ended |= new
        self.last = s
        return self.ended.all()

    def result(self):
        out = {k: np.stack(v, 1) for k, v in self.rec.items()}   # [B, T, ...]
        T = out["q"].shape[1]
        L = np.minimum(self.length, T)
        b = np.arange(self.B)
        fq, fc, fcon = out["q"][b, L - 1], out["cube"][b, L - 1], out["contact"][b, L - 1]
        prev = out["cube"][b, np.maximum(L - 2, 0)]
        out.update(length=L, reason=self.reason.astype(str), recoveries=self.ex.recoveries.cpu().numpy(),
                   final_q=fq, final_cube=fc, final_contact=fcon, final_cube_prev=prev, task=self.idx)
        return out


def success_metrics(res, tasks):
    """Per-task success and errors from a harness result."""
    from nedm.so101_policy.expert import cube_yaw_tilt, wrap90
    from nedm.so101_policy.kin import SO101Kin
    idx = res["task"]
    fc = torch.as_tensor(res["final_cube"])
    yaw, tilt = cube_yaw_tilt(fc[:, 3:7])
    gy = torch.as_tensor(tasks["goal_yaw"][idx])
    yaw_err = wrap90(yaw - gy).abs().numpy()
    xy_err = np.linalg.norm(res["final_cube"][:, :2] - tasks["goal_xy"][idx], axis=1)
    v = np.linalg.norm(res["final_cube"][:, :3] - res["final_cube_prev"][:, :3], axis=1) / 0.02
    gp, _ = SO101Kin().fk(torch.as_tensor(res["final_q"]))
    high = gp[:, 2].numpy() >= res["final_cube"][:, 2] + CUBE_HALF + 0.03
    jaw_free = ~res["final_contact"][:, 0] & ~res["final_contact"][:, 1]
    ok_end = ~np.isin(res["reason"], ["arm_table", "cube_other_link", "arm_self", "cube_base", "cube_off_table",
                                       "numerical", "invalid", "dropped"])
    rest = (v < 0.005) & (tilt.numpy() < math.radians(5))
    common = ok_end & rest & jaw_free & high
    s15 = common & (xy_err <= 0.015) & (yaw_err <= math.radians(15))
    s10 = common & (xy_err <= 0.010) & (yaw_err <= math.radians(10))
    # cause: the failure end, else the first unmet success rule (prefixed by "timeout:" when the expert never finished)
    cause = np.array(res["reason"], dtype=object)
    for j in np.flatnonzero(ok_end):
        if s15[j]:
            cause[j] = "success"
            continue
        why = ("place_error" if (xy_err[j] > 0.015 or yaw_err[j] > math.radians(15)) else "not_resting" if not rest[j]
               else "jaw_contact" if not jaw_free[j] else "gripper_low")
        cause[j] = ("timeout:" if res["reason"][j] == "timeout" else "") + why
    return dict(s15=s15, s10=s10, xy_err=xy_err, yaw_err=yaw_err, tilt=tilt.numpy(), v=v, cause=cause.astype(str))


# ---- MuJoCo world -------------------------------------------------------------------------------------------------
class MjWorlds:
    """B copies of the collector scene (one Scene, B MjData), stepped with the collector's 20 ms rule."""

    def __init__(self, n):
        import mujoco
        from nedm.contact_nrd.collect import so101_grasp as g
        self.mj, self.g = mujoco, g
        self.s = g.Scene(g.load_config())
        self.ds = [mujoco.MjData(self.s.m) for _ in range(n)]
        self.prev = None

    def reset(self, arm0, cube0, act0):
        """Restore the recorded first state (qpos, qvel, ctrl = the first recorded command); mj_forward."""
        s, mj = self.s, self.mj
        for d, a, c, u in zip(self.ds, arm0, cube0, act0):
            mj.mj_resetData(s.m, d)
            d.qpos[s.qadr], d.qvel[s.vadr] = a[:6], a[6:]
            d.qpos[s.cq: s.cq + 7] = c[:7]
            d.qvel[s.cv: s.cv + 3] = c[7:10]
            d.qvel[s.cv + 3: s.cv + 6] = self.g.quat_mat(c[3:7]).T @ c[10:13]
            d.ctrl[s.act] = np.clip(u, s.tlo, s.thi).astype(np.float32)
            mj.mj_forward(s.m, d)
        pose = self.poses()
        self.prev = pose
        return self.world_state(pose, pose, np.zeros((len(self.ds), 3), bool))

    def poses(self):
        s = self.s
        return np.stack([np.concatenate((d.qpos[s.qadr], d.qpos[s.cq: s.cq + 7])) for d in self.ds])

    def world_state(self, pose, prev, contact, end=None):
        from nedm.so101_policy.kin import quat_rotvec
        from nedm.so101_policy.world import WorldState
        t = lambda x: torch.as_tensor(np.asarray(x, np.float64))
        q, cp, cq = pose[:, :6], pose[:, 6:9], pose[:, 9:13]
        qp = prev[:, 9:13]
        sign = np.where((cq * qp).sum(1, keepdims=True) < 0, -1.0, 1.0)
        cq = cq * sign                                                     # sign continuous
        w_, x_, y_, z_ = (qp[:, k] for k in range(4))
        conj = np.stack((w_, -x_, -y_, -z_), 1)
        a, b = cq, conj                                                    # q * conj(q_prev), world-frame rotation
        dq = np.stack((a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1] - a[:, 2] * b[:, 2] - a[:, 3] * b[:, 3],
                       a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0] + a[:, 2] * b[:, 3] - a[:, 3] * b[:, 2],
                       a[:, 0] * b[:, 2] - a[:, 1] * b[:, 3] + a[:, 2] * b[:, 0] + a[:, 3] * b[:, 1],
                       a[:, 0] * b[:, 3] + a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1] + a[:, 3] * b[:, 0]), 1)
        w = quat_rotvec(t(dq)) / 0.02
        n = len(pose)
        end = np.zeros(n, bool) if end is None else end
        return WorldState(q=t(q), qd=t((q - prev[:, :6]) / 0.02), cube_pos=t(cp), cube_quat=t(cq),
                          cube_v=t((cp - prev[:, 6:9]) / 0.02), cube_w=w, contact=torch.as_tensor(contact),
                          arm_table=torch.zeros(n, dtype=torch.bool), invalid=torch.as_tensor(end))

    def step(self, targets, active):
        """targets [B, 6] (rad) -> (WorldState, end [B] bool, reason [B] str, qpos of the 2 records [B, 2, 13])."""
        s, mj, g = self.s, self.mj, self.g
        n = len(self.ds)
        contact = np.zeros((n, 3), bool)
        end = np.zeros(n, bool)
        reason = np.array([""] * n, dtype=object)
        recs = np.zeros((n, 2, 13))
        for i, d in enumerate(self.ds):
            if not active[i]:
                recs[i] = d.qpos[:13]
                continue
            u = np.clip(targets[i], s.tlo, s.thi).astype(np.float32)
            for r in range(s.per_ctrl):
                d.ctrl[s.act] = u
                fmax, bad = np.zeros(3), np.zeros(4)
                for _ in range(s.nsub):
                    mj.mj_step(s.m, d)
                    f, _ = s.contact_forces(d)
                    fmax, bad = np.maximum(fmax, f[:3]), np.maximum(bad, f[3:7])
                contact[i] |= fmax > s.thr
                recs[i, r] = d.qpos[:13]
                if d.warning[mj.mjtWarning.mjWARN_BADQACC].number or not np.isfinite(d.qpos).all():
                    end[i], reason[i] = True, "numerical"
                    break
                if (bad > s.thr).any():
                    end[i], reason[i] = True, g.TERMINAL[1 + int(np.argmax(bad > s.thr))]
                    break
                p = d.qpos[s.cq: s.cq + 3]
                if (p[:2] < s.table_lo).any() or (p[:2] > s.table_hi).any() or p[2] < -0.01:
                    end[i], reason[i] = True, "cube_off_table"
                    break
        pose = self.poses()
        st = self.world_state(pose, self.prev, contact, end)
        st.arm_table = torch.as_tensor(reason == "arm_table")
        self.prev = pose
        return st, end, reason, recs


def _mj_chunk(args):
    tasks_path, idx = args
    torch.set_num_threads(1)
    tasks = load_tasks(tasks_path)
    h = Harness(tasks, idx)
    w = MjWorlds(len(idx))
    st = w.reset(tasks["arm0"][idx], tasks["cube0"][idx], tasks["act0"][idx])
    h.reset(st)
    recs = []
    t0 = time.process_time()
    for t in range(HORIZON):
        tgt = h.act(st).numpy()
        st, end, reason, rq = w.step(tgt, ~h.ended)
        recs.append(rq)
        if h.observe(t, st, end, reason):
            break
    out = h.result()
    out["records_qpos"] = np.concatenate(recs, 1)        # [B, 2T, 13] qpos of every 10 ms record
    out["cpu_s"] = np.full(len(idx), (time.process_time() - t0) / len(idx))
    return out


def merge(parts):
    keys = parts[0].keys()
    out = {}
    for k in keys:
        arrs = [p[k] for p in parts]
        if arrs[0].ndim >= 2 and k not in ("final_q", "final_cube", "final_contact", "final_cube_prev"):
            T = max(a.shape[1] for a in arrs)
            arrs = [np.concatenate((a, np.repeat(a[:, -1:], T - a.shape[1], 1)), 1) if a.shape[1] < T else a
                    for a in arrs]
        out[k] = np.concatenate(arrs)
    order = np.argsort(out["task"])
    return {k: v[order] for k, v in out.items()}


def cmd_mujoco(a):
    tasks = load_tasks(a.tasks)
    n = len(tasks["rows"]) if a.n is None else a.n
    chunks = np.array_split(np.arange(n), max(1, math.ceil(n / a.chunk)))
    t0 = time.time()
    with ProcessPoolExecutor(a.procs) as ex:
        parts = list(ex.map(_mj_chunk, [(a.tasks, c) for c in chunks]))
    res = merge(parts)
    res["wall_s"] = np.array(time.time() - t0)
    np.savez_compressed(a.out, **res)
    m = success_metrics(res, tasks)
    print(f"MuJoCo: {n} tasks in {time.time() - t0:.0f} s; success 15/15 {m['s15'].mean() * 100:.1f} %, 10/10 "
          f"{m['s10'].mean() * 100:.1f} %; causes {dict(zip(*np.unique(m['cause'], return_counts=True)))}")


# ---- NRD world ----------------------------------------------------------------------------------------------------
def qmul(p, q):
    w1, x1, y1, z1 = p.unbind(-1)
    w2, x2, y2, z2 = q.unbind(-1)
    return torch.stack((w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2), -1)


def cube_group(dev):
    """The 24 cube rotations (w x y z), closed from 90 deg turns about x, y, z (symmetry_canon.cube_group)."""
    def ax(i):
        v = [0.0, 0.0, 0.0]
        v[i] = math.sin(math.pi / 4)
        return torch.tensor([math.cos(math.pi / 4)] + v, dtype=torch.float64, device=dev)
    gens = [ax(0), ax(1), ax(2)]
    G = [torch.tensor([1.0, 0, 0, 0], dtype=torch.float64, device=dev)]
    while True:
        new = []
        for g in G:
            for h in gens:
                q = qmul(g, h)
                if all(abs(float((q * x).sum())) < 1 - 1e-9 for x in G + new):
                    new.append(q)
        if not new:
            break
        G += new
    assert len(G) == 24
    return torch.stack(G)


class NrdWorlds:
    """B worlds of the grasp NRD (one checkpoint) with per-step cube canonicalisation."""

    def __init__(self, ckpt, device):
        from nedm.contact_nrd.model import load_model
        from nedm.so101_policy.kin import SO101Kin
        self.m, _ = load_model(ckpt, device)
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.dev = torch.device(device)
        self.kin = SO101Kin(device)
        mov = [b for b in self.m.system["bodies"] if b["moving"]]
        types = lambda b, t: [c for c, x in enumerate(b["channel_types"]) if x == t]
        self.q_c, self.qd_c = types(mov[0], "angle")[:6], types(mov[0], "angular_rate")[:6]
        self.p_c, self.quat_c = types(mov[1], "position"), types(mov[1], "quaternion")
        self.v_c, self.w_c = types(mov[1], "velocity"), types(mov[1], "angular_velocity")
        self.G = cube_group(self.dev)
        self.qref = torch.tensor([math.cos(math.pi / 8), 0, 0, math.sin(math.pi / 8)], dtype=torch.float64,
                                 device=self.dev)

    def _canon(self):
        """Relabel the cube orientation of the whole history (one cube rotation per world, chosen on the newest)."""
        qh = self.hist[:, :, 1, self.quat_c]                                     # [B, K, 4]
        qn = qh[:, -1]
        cand = qmul(qn[:, None, :].expand(-1, 24, -1), self.G[None].expand(len(qn), -1, -1))
        k = (cand * self.qref).sum(-1).abs().argmax(1)
        r = self.G[k]
        q2 = qmul(qh, r[:, None, :].expand_as(qh))
        q2 = q2 * torch.where(q2[:, -1:, :1] < 0, -1.0, 1.0)
        h = self.hist.clone()
        h[:, :, 1, self.quat_c] = q2
        B, K = h.shape[:2]
        h = self.m.finish(h.reshape(B * K, *h.shape[2:])).reshape(h.shape) * self.ok[:, :, None, None]
        self.hist = h
        self.racc = qmul(self.racc, r)
        self.relabels += (k != 0).long()

    def true_quat(self):
        r = self.racc
        q = qmul(self.hist[:, -1, 1, self.quat_c], torch.stack((r[:, 0], -r[:, 1], -r[:, 2], -r[:, 3]), -1))
        return q * torch.where(q[:, :1] < 0, -1.0, 1.0)

    def reset(self, arm0, cube0):
        from nedm.so101_policy.world import WorldState
        B = len(arm0)
        raw = torch.zeros(B, self.m.n_bodies, self.m.smax, dtype=torch.float64, device=self.dev)
        raw[:, 0, :12] = torch.as_tensor(arm0, device=self.dev)
        raw[:, 1, :13] = torch.as_tensor(cube0, device=self.dev)
        self.hist, self.ok = self.m.start(self.m.finish(raw))
        self.racc = torch.zeros(B, 4, dtype=torch.float64, device=self.dev)
        self.racc[:, 0] = 1
        self.relabels = torch.zeros(B, dtype=torch.long, device=self.dev)
        self._canon()
        z = torch.zeros(B, dtype=torch.bool, device=self.dev)
        return self._state(torch.zeros(B, 3, dtype=torch.bool, device=self.dev), z, z)

    def _state(self, contact, arm_table, invalid):
        from nedm.so101_policy.world import WorldState
        s = self.hist[:, -1]
        return WorldState(q=s[:, 0, self.q_c], qd=s[:, 0, self.qd_c], cube_pos=s[:, 1, self.p_c],
                          cube_quat=self.true_quat(), cube_v=s[:, 1, self.v_c], cube_w=s[:, 1, self.w_c],
                          contact=contact, arm_table=arm_table, invalid=invalid)

    @torch.no_grad()
    def step(self, targets):
        B = len(targets)
        act = torch.zeros(B, self.m.n_bodies, self.m.amax, dtype=torch.float64, device=self.dev)
        lo, hi = self.kin.tlo, self.kin.thi
        act[:, 0, :6] = torch.minimum(torch.maximum(targets.to(act), lo), hi).float().double()   # as MuJoCo applies
        nxt, info = self.m.details(self.hist, self.ok, act)
        self.hist, self.ok = self.m.advance(self.hist, self.ok, nxt)
        finite = torch.isfinite(nxt).all(-1).all(-1)
        self.hist = torch.where(finite[:, None, None, None], self.hist, torch.nan_to_num(self.hist))
        self._canon()
        s = self.hist[:, -1]
        p = s[:, 1, self.p_c]
        off = (p[:, 0] < -0.30) | (p[:, 0] > 0.80) | (p[:, 1].abs() > 0.55) | (p[:, 2] < -0.01)
        arm_table = self.kin.lowest_point(s[:, 0, self.q_c]) < GUARD_M
        st = self._state(info["gates"] > 0.5, arm_table, ~finite | off)
        reason = np.where(arm_table.cpu().numpy(), "arm_table",
                          np.where((~finite).cpu().numpy(), "invalid", np.where(off.cpu().numpy(), "cube_off_table", "")))
        return st, (arm_table | ~finite | off).cpu().numpy(), reason.astype(object)


def cmd_nrd(a):
    tasks = load_tasks(a.tasks)
    n = len(tasks["rows"]) if a.n is None else a.n
    idx = np.arange(n)
    dev = a.device
    t0 = time.time()
    w = NrdWorlds(a.ckpt, dev)
    h = Harness(tasks, idx, dev)
    st = w.reset(tasks["arm0"][idx], tasks["cube0"][idx])
    h.reset(st)
    for t in range(HORIZON):
        tgt = h.act(st)
        st, end, reason = w.step(tgt)
        if h.observe(t, st, end, reason):
            break
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    res = h.result()
    res["relabels"] = w.relabels.cpu().numpy()
    res["wall_s"] = np.array(time.time() - t0)
    res["ckpt"] = np.array(str(a.ckpt))
    np.savez_compressed(a.out, **res)
    m = success_metrics(res, tasks)
    print(f"NRD {Path(a.ckpt).name}: {n} tasks in {time.time() - t0:.0f} s; success 15/15 {m['s15'].mean() * 100:.1f} %,"
          f" 10/10 {m['s10'].mean() * 100:.1f} %; causes {dict(zip(*np.unique(m['cause'], return_counts=True)))}")


# ---- post-hoc visual overlap (needs scipy) ------------------------------------------------------------------------
def _ov_chunk(args):
    path, idx = args
    from nedm.contact_nrd.collect import so101_grasp as g
    s = g.Scene(g.load_config())
    f = np.load(path)
    rq, L = f["records_qpos"][idx], f["length"][idx]
    out = np.full(len(idx), -1, np.int64)
    depth = np.zeros(len(idx))
    for j in range(len(idx)):
        for r in range(2 * int(L[j])):
            ov = s.overlap_at(rq[j, r])
            depth[j] = max(depth[j], ov)
            if ov > s.overlap_thr and out[j] < 0:
                out[j] = r
    return out, depth


def cmd_overlap(a):
    n = len(np.load(a.mj)["length"])
    chunks = np.array_split(np.arange(n), a.procs * 4)
    with ProcessPoolExecutor(a.procs) as ex:
        parts = list(ex.map(_ov_chunk, [(a.mj, c) for c in chunks]))
    first = np.concatenate([p[0] for p in parts])
    depth = np.concatenate([p[1] for p in parts])
    np.savez(a.out, first_record=first, max_depth_m=depth)
    print(f"visual overlap > 1 mm in {(first >= 0).sum()} of {n} MuJoCo episodes; max depth p99 "
          f"{np.percentile(depth, 99) * 1e3:.2f} mm, max {depth.max() * 1e3:.2f} mm")


# ---- IK parity against the collector -------------------------------------------------------------------------------
def _ik_chunk(args):
    seed, n = args
    import mujoco  # noqa: F401
    from nedm.contact_nrd.collect import so101_grasp as g
    torch.set_num_threads(1)
    s = g.Scene(g.load_config())
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
        r, ang = rng.uniform(0.12, 0.30), math.radians(rng.uniform(-80, 80))
        pos = np.array([PAN[0] + r * math.cos(ang), PAN[1] + r * math.sin(ang), rng.uniform(0.015, 0.20)])
        yaw = ang + rng.uniform(-1.2, 1.2)
        hint = None
        if rng.random() < 0.5:   # half with a hint: a pose near the target, as Plan.cart / grasp_approach pass
            try:
                hint = s.solve(pos + rng.normal(0, 0.02, 3) * [1, 1, 0.5] + [0, 0, 0.01], g.topdown(yaw + rng.normal(0, 0.1)),
                               0.5, max_tilt=math.radians(45))
            except g.PlanError:
                continue
        try:
            q = s.solve(pos, g.topdown(yaw), 0.5, hint=hint, max_tilt=math.radians(45))
            ok = True
        except g.PlanError:
            q, ok = np.full(6, np.nan), False
        rows.append((pos, yaw, hint if hint is not None else np.full(6, np.nan), q, ok))
    return rows


def cmd_ikparity(a):
    from nedm.so101_policy.kin import SO101Kin, rotz, rotvec
    per = math.ceil(a.n / a.procs)
    with ProcessPoolExecutor(a.procs) as ex:
        parts = list(ex.map(_ik_chunk, [(a.seed + k, per) for k in range(a.procs)]))
    rows = [r for p in parts for r in p][: a.n]
    pos = torch.tensor(np.array([r[0] for r in rows]))
    yaw = torch.tensor(np.array([r[1] for r in rows]))
    hint = torch.tensor(np.array([r[2] for r in rows]))
    qref = np.array([r[3] for r in rows])
    okref = np.array([r[4] for r in rows])
    kin = SO101Kin()
    t0 = time.time()
    has = ~torch.isnan(hint[:, 0])
    q = torch.zeros(len(rows), 5, dtype=torch.float64)
    ok = torch.zeros(len(rows), dtype=torch.bool)
    ep = torch.zeros(len(rows), dtype=torch.float64)
    ea = torch.zeros(len(rows), dtype=torch.float64)
    for sel, h in ((has, hint[has]), (~has, None)):
        if bool(sel.any()):
            q[sel], ok[sel], ep[sel], ea[sel] = kin.ik_topdown(h, pos[sel], yaw[sel], max_tilt=math.radians(45))
    dt = time.time() - t0
    both = okref & ok.numpy()
    dq = np.abs(qref[both, :5] - q.numpy()[both]).max(1)
    p_ref, r_ref = kin.fk(torch.tensor(np.nan_to_num(qref[:, :5])))
    ep_ref = (pos - p_ref).norm(dim=-1).numpy()
    ea_ref = rotvec(rotz(yaw) @ r_ref.transpose(-1, -2)).norm(dim=-1).numpy()
    tilt = np.degrees(ea.numpy()[both])
    out = dict(n=len(rows), with_hint=int(has.sum()), solved_collector=int(okref.sum()), solved_torch=int(ok.sum()),
               solved_agree=int((okref == ok.numpy()).sum()), both_solved=int(both.sum()),
               dq_max_rad=float(dq.max()), dq_p99_rad=float(np.percentile(dq, 99)), dq_p50_rad=float(np.median(dq)),
               same_solution_1e6=int((dq < 1e-6).sum()), same_solution_1e3=int((dq < 1e-3).sum()),
               pos_residual_torch_mm=dict(p50=float(np.median(ep.numpy()[both]) * 1e3),
                                          p99=float(np.percentile(ep.numpy()[both], 99) * 1e3),
                                          max=float(ep.numpy()[both].max() * 1e3)),
               ang_residual_torch_deg=dict(p50=float(np.median(tilt)), p99=float(np.percentile(tilt, 99)),
                                           max=float(tilt.max())),
               pos_residual_collector_mm_max=float(ep_ref[both].max() * 1e3),
               ang_residual_collector_deg_max=float(np.degrees(ea_ref[both].max())),
               torch_cpu_s=dt)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# ---- command statistics ---------------------------------------------------------------------------------------------
def stats(x):
    x = np.abs(np.asarray(x, np.float64).ravel())
    return dict(n=int(x.size), p50=float(np.median(x)), p95=float(np.percentile(x, 95)),
                p99=float(np.percentile(x, 99)), max=float(x.max()))


def command_stats(cmds, lengths):
    """cmds [N, T, 6] joint targets per 20 ms step, lengths [N] valid steps -> per-step change and jerk statistics
    (arm joints pooled, jaw separate); jerk = third difference / dt^3 (rad/s^3)."""
    d1, j3 = {"arm": [], "jaw": []}, {"arm": [], "jaw": []}
    for c, L in zip(cmds, lengths):
        c = c[:L]
        if L < 4:
            continue
        d = np.diff(c, axis=0)
        j = np.diff(c, n=3, axis=0) / 0.02 ** 3
        d1["arm"].append(d[:, :5]), d1["jaw"].append(d[:, 5])
        j3["arm"].append(j[:, :5]), j3["jaw"].append(j[:, 5])
    return {f"{k}_step_rad": stats(np.concatenate([x.ravel() for x in v])) for k, v in d1.items()} | \
           {f"{k}_jerk_rad_s3": stats(np.concatenate([x.ravel() for x in v])) for k, v in j3.items()}


def cmd_smooth(a):
    from nedm.contact_nrd.rollouts import npz_members
    from nedm.so101_policy.decoder import CommandDecoder
    f = npz_members(Path(a.data) / "unified_data.npz")
    fam, split, L = np.asarray(f["family"]), np.asarray(f["splits"]), np.asarray(f["lengths"])
    rng = np.random.default_rng(a.seed)
    out = {}
    for name, fams in (("data_nominal", [0]), ("data_task_families", list(TASK_FAMILIES))):
        rows = np.sort(rng.choice(np.flatnonzero((split == 0) & np.isin(fam, fams)), a.data_eps, replace=False))
        cmds = np.stack([np.asarray(f["action"][r, ::2], np.float64) for r in rows])
        out[name] = command_stats(cmds, (L[rows] - 1) // 2 + 1)
    mj = np.load(a.mj)
    out["expert_mujoco"] = command_stats(mj["target"], mj["length"])
    # random actions through the decoder: OU noise (time constant 0.2 s, std 0.6) and i.i.d. uniform actions
    tasks = load_tasks(a.tasks)
    q0 = torch.as_tensor(tasks["arm0"][:, :6])
    for kind in ("ou", "uniform"):
        dec = CommandDecoder(len(q0))
        dec.reset(torch.arange(len(q0)), q0)
        g = torch.Generator().manual_seed(a.seed)
        x = torch.zeros(len(q0), 5, dtype=torch.float64)
        cmds = []
        for t in range(300):
            if kind == "ou":
                x = x + (-x * 0.02 / 0.2) + 0.6 * math.sqrt(2 * 0.02 / 0.2) * torch.randn(x.shape, generator=g, dtype=x.dtype)
            else:
                x = torch.rand(x.shape, generator=g, dtype=x.dtype) * 2 - 1
            cmds.append(dec.step(x).numpy())
        out[f"decoder_random_{kind}"] = command_stats(np.stack(cmds, 1), np.full(len(q0), 300))
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, {m: (round(s["p95"], 4), round(s["p99"], 4), round(s["max"], 4)) for m, s in v.items()})


# ---- report -----------------------------------------------------------------------------------------------------------
def cmd_report(a):
    tasks = load_tasks(a.tasks)
    mj = dict(np.load(a.mj, allow_pickle=False))
    worlds = {"mujoco": mj} | {Path(p).stem: dict(np.load(p, allow_pickle=False)) for p in a.nrd}
    rep = {}
    if a.overlap:
        ov = np.load(a.overlap)
        rep["mujoco_visual_overlap_episodes"] = int((ov["first_record"] >= 0).sum())
        rep["mujoco_visual_overlap_tasks"] = np.flatnonzero(ov["first_record"] >= 0).tolist()
    M = {}
    for name, r in worlds.items():
        m = success_metrics(r, tasks)
        M[name] = m
        s = m["s15"]
        rep[name] = dict(
            tasks=int(len(s)), success_15mm_15deg=float(s.mean()), success_10mm_10deg=float(m["s10"].mean()),
            causes={str(k): int(v) for k, v in zip(*np.unique(m["cause"], return_counts=True))},
            end_reasons={str(k): int(v) for k, v in zip(*np.unique(r["reason"], return_counts=True))},
            xy_err_mm=dict(p50=float(np.median(m["xy_err"]) * 1e3), p90=float(np.percentile(m["xy_err"], 90) * 1e3),
                           max=float(m["xy_err"].max() * 1e3)),
            yaw_err_deg=dict(p50=float(np.degrees(np.median(m["yaw_err"]))),
                             p90=float(np.degrees(np.percentile(m["yaw_err"], 90))),
                             max=float(np.degrees(m["yaw_err"].max()))),
            episode_s=dict(p50=float(np.median(r["length"]) * 0.02), max=float(r["length"].max() * 0.02)),
            recoveries=dict(tasks_with=int((r["recoveries"] > 0).sum()), total=int(r["recoveries"].sum())),
            success_by_family={int(f): float(s[tasks["family"][r["task"]] == f].mean())
                               for f in np.unique(tasks["family"][r["task"]])})
        if "wall_s" in r:
            rep[name]["wall_s"] = float(r["wall_s"])
    ref = M["mujoco"]
    for name in worlds:
        if name == "mujoco":
            continue
        m, r = M[name], worlds[name]
        a_, b_ = ref["s15"], m["s15"]
        dfin = np.linalg.norm(mj["final_cube"][:, :3] - r["final_cube"][:, :3], axis=1)
        both_ok = a_ & b_
        n01, n10 = int((~a_ & b_).sum()), int((a_ & ~b_).sum())
        rep[f"agreement_{name}_vs_mujoco"] = dict(
            both_success=int(both_ok.sum()), mujoco_only=n10, nrd_only=n01, both_fail=int((~a_ & ~b_).sum()),
            agree_frac=float((a_ == b_).mean()), mcnemar_exact_p=_mcnemar(n01, n10),
            cause_pairs={f"{x}|{y}": int(c) for (x, y), c in _pairs(ref["cause"], m["cause"]).items()},
            final_cube_diff_mm_both_success=dict(p50=float(np.median(dfin[both_ok]) * 1e3) if both_ok.any() else None,
                                                 p95=float(np.percentile(dfin[both_ok], 95) * 1e3) if both_ok.any() else None),
            length_diff_steps_both_success=dict(p50=float(np.median(np.abs(mj["length"] - r["length"])[both_ok]))
                                                if both_ok.any() else None),
            disagree_tasks=np.flatnonzero(a_ != b_).tolist())
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def _pairs(x, y):
    out = {}
    for a, b in zip(x, y):
        if a != b:
            out[(a, b)] = out.get((a, b), 0) + 1
    return out


def _mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tasks")
    p.add_argument("--data", required=True), p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=256), p.add_argument("--seed0", type=int, default=1000)
    p = sub.add_parser("mujoco")
    p.add_argument("--tasks", required=True), p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=8), p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--n", type=int)
    p = sub.add_parser("nrd")
    p.add_argument("--tasks", required=True), p.add_argument("--out", required=True), p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"), p.add_argument("--n", type=int)
    p = sub.add_parser("overlap")
    p.add_argument("--mj", required=True), p.add_argument("--out", required=True), p.add_argument("--procs", type=int, default=8)
    p = sub.add_parser("ikparity")
    p.add_argument("--out", required=True), p.add_argument("--n", type=int, default=1000)
    p.add_argument("--procs", type=int, default=8), p.add_argument("--seed", type=int, default=20261011)
    p = sub.add_parser("smooth")
    p.add_argument("--data", required=True), p.add_argument("--mj", required=True), p.add_argument("--tasks", required=True)
    p.add_argument("--out", required=True), p.add_argument("--data-eps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("report")
    p.add_argument("--tasks", required=True), p.add_argument("--mj", required=True), p.add_argument("--nrd", nargs="*", default=[])
    p.add_argument("--overlap"), p.add_argument("--out", required=True)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
