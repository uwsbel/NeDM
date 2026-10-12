"""Closed-loop check of the pick-and-place expert (``nedm.so101_policy.expert``, acting through ``DecoderV2``) in CPU
MuJoCo (the collector's physics) and in the grasp NRD (``nrd_sim.GraspNrdSim``), on the same tasks; IK parity against
the collector; command statistics of the decoder against the data.

    PYTHONPATH=src python -P scripts/so101_policy/check_expert.py tasks --data DIR --out tasks.npz [--n 256] [--seed0 1000]
    PYTHONPATH=src python -P scripts/so101_policy/check_expert.py mujoco --tasks tasks.npz --out mj.npz [--procs 4] \
        [--noise 0.3] [--expert-cfg '{"v_cruise": 0.15}'] [--decoder-cfg '{"a_max": 1.0}']
    PYTHONPATH=src python -P scripts/so101_policy/check_expert.py nrd --tasks tasks.npz --ckpt CKPT --out nrd.npz [--device cuda]
    PYTHONPATH=src python -P scripts/so101_policy/check_expert.py ikparity --out ik.json [--n 1000] [--procs 4]
    PYTHONPATH=src python -P scripts/so101_policy/check_expert.py smooth --data DIR --runs mj.npz [more.npz] --out sm.json
    PYTHONPATH=src python -P scripts/so101_policy/check_expert.py report --tasks tasks.npz --mj mj.npz --nrd A.npz B.npz \
        --out report.json

Tasks (spec 2.6, written down here): task i uses numpy seed ``seed0 + i`` (validation range 1000-1999). Start = the first
recorded state of a random production_plus split-0 task-family episode (families 0-6; never the sealed test). Goal =
cube centre (x, y) on the workspace annulus (r 0.15-0.27 m about the pan axis, angle +-75 deg), 6-30 cm from the start
(rejection sampling), goal yaw = start yaw + U(-30, 30) deg (mod 90 deg). Horizon 400 steps (8 s).

Episode (both worlds, same harness): every 5 steps (0.1 s) the expert reads the true state and gives a decoder action
(``--noise s``: Gaussian noise of std s added to the free-motion part a[0:4] of every action, the planned PPO
exploration; triggers and macros untouched); every 20 ms the decoder gives the joint targets (lead band against the
measured joint angles). The episode ends at the first of (a) the world's end condition (MuJoCo: the collector's
contact ends arm-table / cube-other-link / arm-self / cube-base, cube off the table, numerical failure; no visual-overlap
end; NRD: ``GraspNrdSim`` invalid flags and the arm-table guard = lowest gripper / moving-jaw collision point < 0.5 mm),
(b) a drop: lowest cube corner > 30 mm while the cube is outside the held box (geometric, no contact switch) for 2
steps, (c) the expert done for 5 steps, (d) 400 steps.

Success at the end (spec_v2 B2, geometric, no contact switch): no failure end, cube flat (tilt <= 5 deg), xy error <=
15 mm and yaw error (mod 90) <= 15 deg (also 10 / 10), speed < 5 mm/s, jaw >= 0.4 rad, TCP >= 3 cm above the cube top,
a firm lift happened (lowest cube corner >= 2 cm inside the held box for >= 0.2 s), and none of the post-hoc failures:
release_high (jaw opening starts with the lowest cube corner > 5 mm), hold_cap (held in the air > 3.5 s), lowest
gripper / jaw point < 0, cube tilted > 15 deg at the end. (The official rules are ``task.py``; these are the sanity
check's own, close to them.)
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
DECISION = 5
DONE_HOLD = 5
DROP_Z = 0.03
CUBE_HALF = 0.015
HELD_BOX = (0.015, 0.015, 0.02)
FAIL_ENDS = ("arm_table", "cube_other_link", "arm_self", "cube_base", "cube_off_table", "numerical", "invalid",
             "dropped")


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
def held_rel(kin, q, cube_pos):
    """Cube centre in the grasp frame [B, 3] (numpy in, numpy out)."""
    p, r = kin.fk(torch.as_tensor(q))
    rel = (r.transpose(-1, -2) @ (torch.as_tensor(cube_pos) - p)[..., None])[..., 0]
    return rel.numpy()


def bottom_np(cube):
    """Lowest cube corner height (m) from [..., 7] (pos, quat w x y z)."""
    from nedm.so101_policy.expert import cube_bottom
    c = torch.as_tensor(np.asarray(cube, np.float64)).reshape(-1, 7)
    return cube_bottom(c[:, :3], c[:, 3:7]).numpy().reshape(np.shape(cube)[:-1])


class Harness:
    """Expert + DecoderV2 over B worlds; end rules and per-step records (numpy)."""

    def __init__(self, tasks, idx, device="cpu", noise=0.0, noise_seed=0, expert_cfg=None, decoder_cfg=None):
        from dataclasses import replace
        from nedm.so101_policy.decoder import DecoderV2, DecoderV2Config
        from nedm.so101_policy.expert import ExpertConfig, PickPlaceExpert
        from nedm.so101_policy.kin import SO101Kin
        self.idx = np.asarray(idx)
        self.B = len(self.idx)
        self.kin = SO101Kin(device)
        self.kin_cpu = SO101Kin("cpu")
        self.dec = DecoderV2(self.B, device, kin=self.kin, cfg=replace(DecoderV2Config(), **(decoder_cfg or {})))
        self.ex = PickPlaceExpert(self.dec, replace(ExpertConfig(), **(expert_cfg or {})))
        self.dev = torch.device(device)
        self.goal_xy = torch.as_tensor(tasks["goal_xy"][self.idx], device=device)
        self.goal_yaw = torch.as_tensor(tasks["goal_yaw"][self.idx], device=device)
        self.seeds = torch.as_tensor(tasks["seeds"][self.idx], dtype=torch.long, device=device)
        self.noise = float(noise)
        self.gen = torch.Generator(device="cpu").manual_seed(int(noise_seed))
        self.ended = np.zeros(self.B, bool)
        self.length = np.full(self.B, HORIZON, np.int64)
        self.reason = np.array(["timeout"] * self.B, dtype=object)
        self.done_run = np.zeros(self.B, np.int64)
        self.drop_run = np.zeros(self.B, np.int64)
        self.rec = {k: [] for k in ("q", "cube", "target", "mode", "phase", "contact", "flags")}
        self.actions = []

    def reset(self, state):
        all_ = torch.arange(self.B, device=self.dev)
        self.dec.reset(all_, state.q, seed=self.seeds)
        self.ex.reset(all_, state, self.goal_xy, self.goal_yaw)

    def decide(self, state):
        from nedm.so101_policy.decoder import M
        a = self.ex.act(state)
        if self.noise > 0:
            eps = torch.randn(self.B, 4, generator=self.gen, dtype=torch.float64).to(a.device) * self.noise
            free = (self.dec.mode == M["free"]) | (self.dec.mode == M["hold"])
            a[:, :4] = torch.where(free[:, None], (a[:, :4] + eps).clamp(-1, 1), a[:, :4])
        self.dec.decide(a, state)
        self.actions.append(a.cpu().numpy())

    def act(self, state):
        from nedm.so101_policy.decoder import FLAGS
        tgt, info = self.dec.step(state.q)
        self.rec["target"].append(tgt.cpu().numpy())
        self.rec["flags"].append(np.stack([info[k].cpu().numpy() for k in FLAGS if k in info], -1))
        return tgt

    def observe(self, t, state, world_end, world_reason):
        """After step t (0-based): record, then apply the end rules to worlds still running."""
        from nedm.so101_policy.expert import PH
        s = state.to("cpu")
        q, cube = s.q.numpy(), torch.cat((s.cube_pos, s.cube_quat), -1).numpy()
        self.rec["q"].append(q), self.rec["cube"].append(cube), self.rec["contact"].append(s.contact.numpy())
        self.rec["phase"].append(self.ex.phase.cpu().numpy())
        self.rec["mode"].append(self.dec.mode.cpu().numpy())
        live = ~self.ended
        fail = live & world_end
        self.reason[fail] = world_reason[fail]
        rel = held_rel(self.kin_cpu, q, cube[:, :3])
        inbox = (np.abs(rel) < HELD_BOX).all(-1)
        up = (bottom_np(cube) > DROP_Z) & ~inbox
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
        return self.ended.all()

    def result(self):
        out = {k: np.stack(v, 1) for k, v in self.rec.items()}   # [B, T, ...]
        out["actions"] = np.stack(self.actions, 1)
        T = out["q"].shape[1]
        L = np.minimum(self.length, T)
        b = np.arange(self.B)
        out.update(length=L, reason=self.reason.astype(str), recoveries=self.ex.recoveries.cpu().numpy(),
                   final_q=out["q"][b, L - 1], final_cube=out["cube"][b, L - 1],
                   final_cube_prev=out["cube"][b, np.maximum(L - 2, 0)], task=self.idx,
                   counts=np.stack([v.cpu().numpy() for v in self.dec.counts.values()], -1),
                   count_names=np.array(list(self.dec.counts)))
        return out


def episode_checks(res):
    """Post-hoc per-episode checks over the recorded steps (numpy): firm lift, release height, held-in-air time,
    lowest point."""
    from nedm.so101_policy.decoder import M
    from nedm.so101_policy.kin import SO101Kin
    kin = SO101Kin()
    q, cube, mode, L = res["q"], res["cube"], res["mode"], res["length"]
    B, T = q.shape[:2]
    rel = held_rel(kin, q.reshape(-1, 6), cube.reshape(-1, 7)[:, :3]).reshape(B, T, 3)
    inbox = (np.abs(rel) < HELD_BOX).all(-1) & (q[..., 5] > 0.08) & (q[..., 5] < 0.35)
    bot = bottom_np(cube)
    valid = np.arange(T)[None] < L[:, None]
    lifted = inbox & (bot >= 0.02) & valid
    air = inbox & (bot > 0.005) & valid
    low = kin.lowest_point(torch.as_tensor(q.reshape(-1, 6))).numpy().reshape(B, T)
    out = dict(firm_lift=np.zeros(B, bool), held_air_s=np.zeros(B), release_bottom_mm=np.full(B, np.nan),
               lowest_mm=np.where(valid, low, np.inf).min(1) * 1e3)
    for i in range(B):
        run = best = 0
        for v in lifted[i]:
            run = run + 1 if v else 0
            best = max(best, run)
        out["firm_lift"][i] = best >= 10
        run = best = 0
        for v in air[i]:
            run = run + 1 if v else 0
            best = max(best, run)
        out["held_air_s"][i] = best * 0.02
        opens = np.flatnonzero((mode[i, 1:] == M["p_open"]) & (mode[i, :-1] != M["p_open"]) & valid[i, 1:]) + 1
        if len(opens):
            out["release_bottom_mm"][i] = bot[i, opens].max() * 1e3
    return out


def success_metrics(res, tasks):
    """Per-task success, errors and failure causes from a harness result (module docstring)."""
    from nedm.so101_policy.expert import cube_yaw_tilt, wrap90
    from nedm.so101_policy.kin import SO101Kin
    idx = res["task"]
    fc = torch.as_tensor(res["final_cube"])
    yaw, tilt = cube_yaw_tilt(fc[:, 3:7])
    gy = torch.as_tensor(tasks["goal_yaw"][idx])
    yaw_err = wrap90(yaw - gy).abs().numpy()
    tilt = tilt.numpy()
    xy_err = np.linalg.norm(res["final_cube"][:, :2] - tasks["goal_xy"][idx], axis=1)
    v = np.linalg.norm(res["final_cube"][:, :3] - res["final_cube_prev"][:, :3], axis=1) / 0.02
    gp, _ = SO101Kin().fk(torch.as_tensor(res["final_q"]))
    high = gp[:, 2].numpy() >= res["final_cube"][:, 2] + CUBE_HALF + 0.03
    jaw_open = res["final_q"][:, 5] >= 0.4
    chk = episode_checks(res)
    ok_end = ~np.isin(res["reason"], FAIL_ENDS)
    post = np.full(len(idx), "", dtype=object)
    post[chk["lowest_mm"] < 0] = "lowest_point"
    post[chk["held_air_s"] > 3.5] = "hold_cap"
    post[chk["release_bottom_mm"] > 5.0] = "release_high"
    post[(tilt > math.radians(15)) & ok_end] = "cube_tilted"
    rest = (v < 0.005) & (tilt < math.radians(5))
    common = ok_end & (post == "") & rest & jaw_open & high & chk["firm_lift"]
    s15 = common & (xy_err <= 0.015) & (yaw_err <= math.radians(15))
    s10 = common & (xy_err <= 0.010) & (yaw_err <= math.radians(10))
    cause = np.array(res["reason"], dtype=object)
    for j in range(len(idx)):
        if not ok_end[j]:
            continue
        if s15[j]:
            cause[j] = "success"
            continue
        why = (post[j] if post[j] else "no_lift" if not chk["firm_lift"][j]
               else "place_error" if (xy_err[j] > 0.015 or yaw_err[j] > math.radians(15))
               else "not_resting" if not rest[j] else "jaw_closed" if not jaw_open[j] else "gripper_low")
        cause[j] = ("timeout:" if res["reason"][j] == "timeout" else "") + why
    return dict(s15=s15, s10=s10, xy_err=xy_err, yaw_err=yaw_err, tilt=tilt, v=v, cause=cause.astype(str), **chk)


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
        """targets [B, 6] (rad) -> (WorldState, end [B] bool, reason [B] str)."""
        s, mj, g = self.s, self.mj, self.g
        n = len(self.ds)
        contact = np.zeros((n, 3), bool)
        end = np.zeros(n, bool)
        reason = np.array([""] * n, dtype=object)
        for i, d in enumerate(self.ds):
            if not active[i]:
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
        return st, end, reason


def _mj_chunk(args):
    tasks_path, idx, noise, noise_seed, ecfg, dcfg = args
    torch.set_num_threads(1)
    tasks = load_tasks(tasks_path)
    h = Harness(tasks, idx, noise=noise, noise_seed=noise_seed, expert_cfg=ecfg, decoder_cfg=dcfg)
    w = MjWorlds(len(idx))
    st = w.reset(tasks["arm0"][idx], tasks["cube0"][idx], tasks["act0"][idx])
    h.reset(st)
    t0 = time.process_time()
    for t in range(HORIZON):
        if t % DECISION == 0:
            h.decide(st)
        tgt = h.act(st).numpy()
        st, end, reason = w.step(tgt, ~h.ended)
        if h.observe(t, st, end, reason):
            break
    out = h.result()
    out["cpu_s"] = np.full(len(idx), (time.process_time() - t0) / len(idx))
    return out


def merge(parts):
    keys = parts[0].keys()
    out = {}
    for k in keys:
        arrs = [p[k] for p in parts]
        if k in ("count_names",):
            out[k] = arrs[0]
            continue
        if arrs[0].ndim >= 2 and k not in ("final_q", "final_cube", "final_cube_prev", "counts"):
            T = max(a.shape[1] for a in arrs)
            arrs = [np.concatenate((a, np.repeat(a[:, -1:], T - a.shape[1], 1)), 1) if a.shape[1] < T else a
                    for a in arrs]
        out[k] = np.concatenate(arrs)
    order = np.argsort(out["task"])
    return {k: (v[order] if k != "count_names" else v) for k, v in out.items()}


def summary(res, tasks, name):
    m = success_metrics(res, tasks)
    cnt = dict(zip(res["count_names"].tolist(), res["counts"].sum(0).tolist()))
    return (f"{name}: {len(res['task'])} tasks; success 15/15 {m['s15'].mean() * 100:.1f} %, 10/10 "
            f"{m['s10'].mean() * 100:.1f} %; causes {dict(zip(*np.unique(m['cause'], return_counts=True)))}; "
            f"decoder flags (world-steps) {cnt}")


def cmd_mujoco(a):
    tasks = load_tasks(a.tasks)
    n = len(tasks["rows"]) if a.n is None else a.n
    chunks = np.array_split(np.arange(n), max(1, math.ceil(n / a.chunk)))
    t0 = time.time()
    with ProcessPoolExecutor(a.procs) as ex:
        parts = list(ex.map(_mj_chunk, [(a.tasks, c, a.noise, a.noise_seed + k, json.loads(a.expert_cfg),
                                         json.loads(a.decoder_cfg)) for k, c in enumerate(chunks)]))
    res = merge(parts)
    res["wall_s"] = np.array(time.time() - t0)
    res["noise"] = np.array(a.noise)
    np.savez_compressed(a.out, **res)
    print(summary(res, tasks, f"MuJoCo (noise {a.noise})") + f"; wall {time.time() - t0:.0f} s")


# ---- NRD world ----------------------------------------------------------------------------------------------------
def cmd_nrd(a):
    from nedm.so101_policy.kin import SO101Kin
    from nedm.so101_policy.nrd_sim import GraspNrdSim, lowest_point_guard
    tasks = load_tasks(a.tasks)
    n = len(tasks["rows"]) if a.n is None else a.n
    idx = np.arange(n)
    dev = a.device
    t0 = time.time()
    sim = GraspNrdSim(a.ckpt, num_envs=n, device=dev, members="split",
                      arm_table_fn=lowest_point_guard(SO101Kin(dev)))
    h = Harness(tasks, idx, dev, noise=a.noise, noise_seed=a.noise_seed, expert_cfg=json.loads(a.expert_cfg),
                decoder_cfg=json.loads(a.decoder_cfg))
    t = lambda x: torch.as_tensor(x, dtype=torch.float64, device=dev)
    c0 = tasks["cube0"][idx]
    st = sim.reset(None, dict(q=t(tasks["arm0"][idx, :6]), qd=t(tasks["arm0"][idx, 6:]), cube_pos=t(c0[:, :3]),
                              cube_quat=t(c0[:, 3:7]), cube_v=t(c0[:, 7:10]), cube_w=t(c0[:, 10:13])))
    h.reset(st)
    for k in range(HORIZON):
        if k % DECISION == 0:
            h.decide(st)
        tgt = h.act(st)
        st = sim.step(tgt)
        end = (st.invalid | st.arm_table).cpu().numpy()
        reason = np.where(st.arm_table.cpu().numpy(), "arm_table", np.where(st.invalid.cpu().numpy(), "invalid", ""))
        if h.observe(k, st, end, reason.astype(object)):
            break
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    res = h.result()
    res["wall_s"] = np.array(time.time() - t0)
    res["ckpt"] = np.array(str(a.ckpt))
    res["noise"] = np.array(a.noise)
    np.savez_compressed(a.out, **res)
    print(summary(res, tasks, f"NRD {Path(a.ckpt).name} (noise {a.noise})") + f"; wall {time.time() - t0:.0f} s")


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
    from nedm.so101_policy.kin import SO101Kin, rotvec, rotz
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
JNAMES = ("pan", "lift", "elbow", "wflex", "wroll", "jaw")


def command_stats(cmds, q, lengths, grip=None):
    """cmds [N, T, 6] joint targets per 20 ms step, q [N, T, 6] measured joint angles at the same steps (or None),
    lengths [N], grip [N, T] bool (steps inside the grasp / release jaw phases: data close, grip, release; decoder
    g_close, g_grip, p_open) -> per joint |dcmd|, |d2cmd| p50 / p99 / max (rad per step), |cmd - q| p99.9 (lead), and
    the fraction of jaw-moving steps with the arm command still (|dcmd| < 1e-9 on all 5 arm joints), all and inside the
    grip phases."""
    d1, d2, lead = [], [], []
    still, jaw_moving, g_still, g_moving = 0, 0, 0, 0
    for i, L in enumerate(lengths):
        c = cmds[i, :L]
        if L < 3:
            continue
        a1, a2 = np.diff(c, axis=0), np.diff(c, n=2, axis=0)
        d1.append(np.abs(a1)), d2.append(np.abs(a2))
        jm = np.abs(a1[:, 5]) > 1e-6
        jaw_moving += int(jm.sum())
        still += int((np.abs(a1[jm, :5]).max(-1) < 1e-9).sum())
        if grip is not None:
            gm = jm & grip[i, 1:L]
            g_moving += int(gm.sum())
            g_still += int((np.abs(a1[gm, :5]).max(-1) < 1e-9).sum())
        if q is not None:
            lead.append(np.abs(c[1:] - q[i, :L - 1]))   # command of step k against the state it was sent from
    d1, d2 = np.concatenate(d1), np.concatenate(d2)
    out = {}
    for j, n in enumerate(JNAMES):
        out[n] = dict(d1_p50=float(np.median(d1[:, j])), d1_p99=float(np.percentile(d1[:, j], 99)),
                      d1_max=float(d1[:, j].max()), d2_p50=float(np.median(d2[:, j])),
                      d2_p99=float(np.percentile(d2[:, j], 99)), d2_max=float(d2[:, j].max()))
        if lead:
            out[n]["lead_p999"] = float(np.percentile(np.concatenate(lead)[:, j], 99.9))
    out["steps"] = int(len(d1))
    out["jaw_moving_steps"] = jaw_moving
    out["arm_still_while_jaw_moves"] = still / max(1, jaw_moving)
    out["grip_jaw_moving_steps"] = g_moving
    out["arm_still_while_jaw_moves_in_grip_phases"] = g_still / max(1, g_moving)
    return out


def cmd_smooth(a):
    from nedm.contact_nrd.rollouts import npz_members
    f = npz_members(Path(a.data) / "unified_data.npz")
    fam, L = np.asarray(f["family"]), np.asarray(f["lengths"])
    rows = np.flatnonzero(np.isin(fam, list(TASK_FAMILIES)))
    if a.data_eps and a.data_eps < len(rows):
        rows = np.sort(np.random.default_rng(a.seed).choice(rows, a.data_eps, replace=False))
    from nedm.contact_nrd.collect.so101_grasp import P
    from nedm.so101_policy.decoder import M
    cmds = np.stack([np.asarray(f["action"][r, ::2], np.float64) for r in rows])
    arm = np.stack([np.asarray(f["arm"][r, ::2, :6], np.float64) for r in rows])
    ph = np.stack([np.asarray(f["phase"][r, ::2]) for r in rows])
    out = {"data_task_families": command_stats(cmds, arm, (L[rows] - 1) // 2 + 1,
                                               np.isin(ph, [P["close"], P["grip"], P["release"]]))}
    out["data_task_families"]["episodes"] = int(len(rows))
    for path in a.runs:
        r = np.load(path)
        # mode[k] = the decoder mode after step k; command k belongs to the mode before it
        mode = np.concatenate((r["mode"][:, :1], r["mode"][:, :-1]), 1)
        grip = np.isin(mode, [M["g_close"], M["g_grip"], M["p_open"]])
        out[Path(path).stem] = command_stats(r["target"], r["q"], r["length"], grip)
        out[Path(path).stem]["episodes"] = int(len(r["length"]))
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, {n: (round(v[n]["d1_p99"], 4), round(v[n]["d2_p99"], 4)) for n in JNAMES},
              "arm still while jaw moves: all", round(v["arm_still_while_jaw_moves"], 4), "grip phases",
              round(v["arm_still_while_jaw_moves_in_grip_phases"], 4))


# ---- report -----------------------------------------------------------------------------------------------------------
def cmd_report(a):
    tasks = load_tasks(a.tasks)
    mj = dict(np.load(a.mj, allow_pickle=False))
    worlds = {"mujoco": mj} | {Path(p).stem: dict(np.load(p, allow_pickle=False)) for p in a.nrd}
    rep, M_ = {}, {}
    for name, r in worlds.items():
        m = success_metrics(r, tasks)
        M_[name] = m
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
            held_air_s=dict(p50=float(np.median(m["held_air_s"])), max=float(m["held_air_s"].max())),
            release_bottom_mm=dict(p50=float(np.nanmedian(m["release_bottom_mm"])),
                                   max=float(np.nanmax(m["release_bottom_mm"]))),
            lowest_mm_min=float(m["lowest_mm"].min()),
            recoveries=dict(tasks_with=int((r["recoveries"] > 0).sum()), total=int(r["recoveries"].sum())),
            decoder_flags_world_steps=dict(zip(r["count_names"].tolist(), r["counts"].sum(0).tolist())),
            success_by_family={int(f): float(s[tasks["family"][r["task"]] == f].mean())
                               for f in np.unique(tasks["family"][r["task"]])})
        if "wall_s" in r:
            rep[name]["wall_s"] = float(r["wall_s"])
    ref = M_["mujoco"]
    for name in worlds:
        if name == "mujoco":
            continue
        m, r = M_[name], worlds[name]
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
    p.add_argument("--procs", type=int, default=4), p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--n", type=int), p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--expert-cfg", default="{}"), p.add_argument("--decoder-cfg", default="{}")
    p = sub.add_parser("nrd")
    p.add_argument("--tasks", required=True), p.add_argument("--out", required=True), p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"), p.add_argument("--n", type=int)
    p.add_argument("--noise", type=float, default=0.0), p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--expert-cfg", default="{}"), p.add_argument("--decoder-cfg", default="{}")
    p = sub.add_parser("ikparity")
    p.add_argument("--out", required=True), p.add_argument("--n", type=int, default=1000)
    p.add_argument("--procs", type=int, default=4), p.add_argument("--seed", type=int, default=20261011)
    p = sub.add_parser("smooth")
    p.add_argument("--data", required=True), p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--out", required=True), p.add_argument("--data-eps", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("report")
    p.add_argument("--tasks", required=True), p.add_argument("--mj", required=True), p.add_argument("--nrd", nargs="*", default=[])
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
