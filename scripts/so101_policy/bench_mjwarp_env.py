"""Validation and speed of the MuJoCo Warp SO101 env (src/nedm/so101_policy/mjwarp_env.py) against CPU MuJoCo.

    # 1. episodes (validation split only; never the sealed test): first state, 20 ms commands, records, labels
    python -I scripts/so101_policy/bench_mjwarp_env.py prep --data ~/so101_nrd_fix/data/val_only --out EPS.npz
    # 2. CPU MuJoCo replay (collector Scene; nrdsim env: mujoco + numpy only), from the recorded first state
    PYTHONPATH=src python -P scripts/so101_policy/bench_mjwarp_env.py cpu --eps EPS.npz --out CPU.npz --workers 8
    # 3. GPU replay (luffy, conda env so101_mjwarp: torch 2.12 cu130 + mujoco 3.15 + mujoco_warp 3.15 + warp 1.18)
    PYTHONPATH=src python -P scripts/so101_policy/bench_mjwarp_env.py gpu --eps EPS.npz --out GPU.npz
    # 4. accuracy, contact-flag agreement, overflow -> JSON
    PYTHONPATH=src python -P scripts/so101_policy/bench_mjwarp_env.py compare --eps EPS.npz --cpu CPU.npz --gpu GPU.npz --out R.json
    # 5. speed: world i replays episode i % N; 300 env.step calls (graph launch + WorldState copy), capture excluded
    PYTHONPATH=src python -P scripts/so101_policy/bench_mjwarp_env.py speed --eps EPS.npz --nworld 4096 8192 16384 --bare --out S.json

Both replays start from the same state (recorded first state, qacc warm start 0, targets = clip(q)) and get the
same commands; after an episode's recorded end the last command is held (those steps are not scored).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

KINDS = ("fixed_jaw_cube", "moving_jaw_cube", "cube_table", "arm_table", "cube_other_link", "arm_self", "cube_base")
NCMD = 300


# ---- 1. episodes ------------------------------------------------------------------------------------------------
def prep(a):
    z = np.load(os.path.expanduser(a.data) + "/unified_data.npz")
    if (z["splits"] != 1).any() and not a.allow_train:
        raise SystemExit("prep reads the validation split only (splits == 1)")
    out = dict(arm=z["arm"][:, ::2], cube=z["cube"][:, ::2], contacts=z["contacts"])
    act = z["action"][:, 0:600:2].copy()
    L = z["lengths"]
    for e in np.flatnonzero(L < 601):   # hold the last applied command after the recorded end
        last = (int(L[e]) - 1) // 2
        act[e, last + 1:] = act[e, min(last, NCMD - 1)]
    out["action"] = act
    for k in ("lengths", "family", "terminal_reason", "splits", "episode_id", "outcome"):
        out[k] = z[k]
    np.savez(a.out, **out)
    print(json.dumps({k: list(v.shape) for k, v in out.items()}))


def valid_steps(L):
    """Number of scored 20 ms steps per episode: commands whose two records are inside the episode."""
    return (np.asarray(L) - 1) // 2


# ---- 2. CPU replay ----------------------------------------------------------------------------------------------
def _cpu_worker(job):
    import mujoco
    from nedm.contact_nrd.collect import so101_grasp as g
    path, rows = job
    E = np.load(path)
    s = g.Scene(g.load_config())
    m, d, cq, cv = s.m, s.d, s.cq, s.cv
    n = len(rows)
    qpos = np.zeros((n, NCMD + 1, m.nq))
    qvel = np.zeros((n, NCMD + 1, m.nv))
    fmax = np.zeros((n, NCMD, 7), np.float32)
    off = np.zeros((n, NCMD), bool)
    f0 = np.zeros((n, 7), np.float32)
    t0 = time.process_time()
    for i, e in enumerate(rows):
        arm, cube = E["arm"][e, 0].astype(np.float64), E["cube"][e, 0].astype(np.float64)
        act = E["action"][e]
        mujoco.mj_resetData(m, d)
        d.qpos[s.qadr], d.qvel[s.vadr] = arm[:6], arm[6:12]
        d.qpos[cq:cq + 3], d.qpos[cq + 3:cq + 7] = cube[0:3], cube[3:7]
        d.qvel[cv:cv + 3], d.qvel[cv + 3:cv + 6] = cube[7:10], g.quat_mat(cube[3:7]).T @ cube[10:13]
        d.ctrl[s.act] = np.clip(arm[:6], s.tlo, s.thi).astype(np.float32)
        mujoco.mj_forward(m, d)
        f0[i] = s.contact_forces()[0][:7]
        qpos[i, 0], qvel[i, 0] = d.qpos, d.qvel
        for k in range(NCMD):
            d.ctrl[s.act] = np.clip(act[k], s.tlo, s.thi).astype(np.float32)
            fm = np.zeros(7)
            for r in range(s.per_ctrl):
                for j in range(s.nsub):
                    mujoco.mj_step(m, d)
                    fm = np.maximum(fm, s.contact_forces()[0][:7])
                p = d.qpos[cq:cq + 3]
                off[i, k] |= bool((p[:2] < s.table_lo).any() or (p[:2] > s.table_hi).any() or p[2] < -0.01)
            fmax[i, k], qpos[i, k + 1], qvel[i, k + 1] = fm, d.qpos, d.qvel
    return rows, qpos, qvel, fmax, off, f0, time.process_time() - t0


def cpu(a):
    import mujoco
    E = np.load(a.eps)
    N = len(E["lengths"]) if a.limit is None else min(a.limit, len(E["lengths"]))
    chunks = [(a.eps, list(r)) for r in np.array_split(np.arange(N), max(1, N // a.chunk))]
    out = None
    t = time.time()
    cpu_s = 0.0
    with ProcessPoolExecutor(a.workers) as ex:
        for rows, qp, qv, fm, off, f0, ts in ex.map(_cpu_worker, chunks):
            if out is None:
                out = dict(qpos=np.zeros((N, *qp.shape[1:])), qvel=np.zeros((N, *qv.shape[1:]), np.float32),
                           fmax=np.zeros((N, NCMD, 7), np.float32), off=np.zeros((N, NCMD), bool),
                           f0=np.zeros((N, 7), np.float32))
            out["qpos"][rows], out["qvel"][rows], out["fmax"][rows], out["off"][rows], out["f0"][rows] = qp, qv, fm, off, f0
            cpu_s += ts
    wall = time.time() - t
    np.savez(a.out, **out, meta=json.dumps(dict(mujoco=mujoco.__version__, episodes=N, wall_s=wall, cpu_s=cpu_s,
                                                workers=a.workers, host=os.uname().nodename)))
    print(json.dumps(dict(episodes=N, wall_s=wall, cpu_s=cpu_s, ctrl_steps_per_cpu_s=N * NCMD / cpu_s)))


# ---- 3. GPU replay ----------------------------------------------------------------------------------------------
def _start(E, rows):
    arm, cube = E["arm"][rows, 0].astype(np.float64), E["cube"][rows, 0].astype(np.float64)
    return dict(q=arm[:, :6], qd=arm[:, 6:12], cube_pos=cube[:, 0:3], cube_quat=cube[:, 3:7], cube_v=cube[:, 7:10],
                cube_w=cube[:, 10:13])


def gpu(a):
    import torch
    from nedm.so101_policy.mjwarp_env import MjwarpWorlds
    E = np.load(a.eps)
    N = len(E["lengths"]) if a.limit is None else min(a.limit, len(E["lengths"]))
    rows = np.arange(N) % N
    if a.repeat > 1:   # the same episodes in several worlds (GPU repeatability)
        rows = np.tile(np.arange(N), a.repeat)
    B = len(rows)
    t = time.time()
    env = MjwarpWorlds(B, nconmax=a.nconmax, njmax=a.njmax, cube_rates=a.cube_rates)
    st = env.reset(None, _start(E, rows))
    torch.cuda.synchronize()
    t_init = time.time() - t
    act = torch.tensor(E["action"][rows], dtype=torch.float64, device=env.device)
    S = {k: [] for k in ("q", "qd", "cube_pos", "cube_quat", "cube_v", "cube_w", "contact", "arm_table", "invalid")}
    raw_q, fmax = [], []
    def keep(st):
        for k in S:
            S[k].append(getattr(st, k).cpu().numpy())
        raw_q.append(env._qpos_t.cpu().numpy().copy())
    keep(st)
    torch.cuda.synchronize()
    t = time.time()
    import warp as wp
    nacon, nefc = [], []
    for k in range(NCMD):
        st = env.step(act[:, k])
        keep(st)
        fmax.append(env.contact_force.cpu().numpy())
        nacon.append(int(wp.to_torch(env.d.nacon)[0]))          # last physics step of the command (a sample)
        nefc.append(wp.to_torch(env.d.nefc).cpu().numpy())
    torch.cuda.synchronize()
    wall = time.time() - t
    nefc = np.stack(nefc, 1)
    out = {k: np.stack(v, 1) for k, v in S.items()}
    out["raw_qpos"], out["fmax"] = np.stack(raw_q, 1), np.stack(fmax, 1)
    out["overflow"], out["rows"] = env.overflow.cpu().numpy(), rows
    out["nefc_sample"] = nefc.astype(np.int16)
    meta = dict(nworld=B, init_s=t_init, replay_wall_s=wall, nconmax=a.nconmax, njmax=a.njmax, cube_rates=a.cube_rates,
                nacon_total_max=max(nacon), nacon_per_world_mean_max=max(nacon) / B, nefc_world_max=int(nefc.max()),
                nefc_world_p999=float(np.percentile(nefc, 99.9)),
                gpu=torch.cuda.get_device_name(0), note="replay wall includes per-step device->host copies")
    np.savez(a.out, **out, meta=json.dumps(meta))
    print(json.dumps(meta))


# ---- 4. comparison ----------------------------------------------------------------------------------------------
def _pct(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return [round(float(v), 4) for v in np.percentile(x, [50, 95])] + [round(float(x.max()), 4)]


def step_average(q, dt=0.02):
    """Data definition (nedm.contact_nrd.data._step_average) on poses [N, T, 13] (arm 6, cube pos 3, quat 4):
    -> arm qd, cube v, cube w [N, T-1, 3 or 6] (float64)."""
    q = np.asarray(q, np.float64)
    qd = (q[:, 1:, :6] - q[:, :-1, :6]) / dt
    v = (q[:, 1:, 6:9] - q[:, :-1, 6:9]) / dt
    q1, q0 = q[:, 1:, 9:13], q[:, :-1, 9:13]
    w0, v0 = q0[..., :1], -q0[..., 1:]
    w1, v1 = q1[..., :1], q1[..., 1:]
    w = w1 * w0 - (v1 * v0).sum(-1, keepdims=True)
    vv = w1 * v0 + w0 * v1 + np.cross(v1, v0)
    sgn = np.where(w < 0, -1.0, 1.0)
    w, vv = w * sgn, vv * sgn
    n = np.linalg.norm(vv, axis=-1, keepdims=True)
    ang = 2 * np.arctan2(n, w)
    rot = np.where(n > 1e-12, vv / np.maximum(n, 1e-12) * ang, 2 * vv)
    return qd, v, rot / dt


def compare(a):
    E, C, G = np.load(a.eps), np.load(a.cpu), np.load(a.gpu)
    N = len(C["qpos"])
    rows = G["rows"][:N] if "rows" in G.files else np.arange(N)
    assert (rows == np.arange(N)).all()
    L, fam, term = E["lengths"][:N], E["family"][:N], E["terminal_reason"][:N]
    fams = [f["name"] for f in json.load(open(a.system))["families"]] if a.system else None
    nv = valid_steps(L)
    T = np.arange(NCMD)[None]
    valid = T < nv[:, None]                                       # [N, 300] scored steps
    cq = 6
    gpu_cube = G["cube_pos"][:N]                                  # [N, 301, 3]
    cpu_cube = C["qpos"][:, :, cq:cq + 3]
    rec_cube = E["cube"][:N, :, :3].astype(np.float64)
    full = L == 601
    task = fam <= 6
    res = dict(episodes=int(N), full_length=int(full.sum()), task_family=int(task.sum()))
    # cube position error at fixed times (full-length episodes)
    acc = {}
    for name, sel in (("task_full", task & full), ("nominal_full", (fam == 0) & full), ("all_full", full)):
        r = {}
        for ts in (1, 2, 4, 6):
            k = int(round(ts / 0.02))
            r[f"{ts}s"] = dict(gpu_vs_cpu=_pct(1e3 * np.linalg.norm(gpu_cube[sel, k] - cpu_cube[sel, k], axis=-1)),
                               gpu_vs_record=_pct(1e3 * np.linalg.norm(gpu_cube[sel, k] - rec_cube[sel, k], axis=-1)),
                               cpu_vs_record=_pct(1e3 * np.linalg.norm(cpu_cube[sel, k] - rec_cube[sel, k], axis=-1)))
        r["episodes"] = int(sel.sum())
        acc[name] = r
    res["cube_pos_error_mm_p50_p95_max"] = acc
    # per family at the last scored step (any length)
    per_fam = {}
    last = np.maximum(nv, 1)
    eg = 1e3 * np.linalg.norm(gpu_cube[np.arange(N), last] - cpu_cube[np.arange(N), last], axis=-1)
    for f in np.unique(fam):
        sel = fam == f
        per_fam[fams[f] if fams else int(f)] = dict(episodes=int(sel.sum()), end_err_gpu_vs_cpu=_pct(eg[sel]),
                                                    ends_early=int((L[sel] < 601).sum()))
    res["cube_end_error_mm_by_family"] = per_fam
    # contact flags per 20 ms step
    gflag = G["contact"][:N, 1:]                                  # [N, 300, 3] after each command
    cflag = C["fmax"][:, :, :3] > 0.01
    rlab = E["contacts"][:N].reshape(N, NCMD, 2, 3).any(2)
    fl = {}
    for p, name in enumerate(KINDS[:3]):
        v = valid
        fl[name] = dict(gpu_vs_cpu=float((gflag[..., p] == cflag[..., p])[v].mean() * 100),
                        gpu_vs_record=float((gflag[..., p] == rlab[..., p])[v].mean() * 100),
                        cpu_vs_record=float((cflag[..., p] == rlab[..., p])[v].mean() * 100),
                        on_fraction_cpu=float(cflag[..., p][v].mean() * 100),
                        gpu_on_cpu_off=int((gflag[..., p] & ~cflag[..., p])[v].sum()),
                        gpu_off_cpu_on=int((~gflag[..., p] & cflag[..., p])[v].sum()))
    for name, sel in (("nominal", fam == 0), ("task", task)):
        v = valid & sel[:, None]
        fl[f"all_three_equal_{name}_gpu_vs_cpu"] = float((gflag == cflag).all(-1)[v].mean() * 100)
    res["contact_flag_agreement_pct"] = fl
    # episode-level lift flag (evaluation rule: cube centre z > 25 mm with both jaws on and table off)
    def lifted(z, f):
        return ((z > 0.025) & f[..., 0] & f[..., 1] & ~f[..., 2] & valid).any(1)
    lg, lc = lifted(gpu_cube[:, 1:, 2], gflag), lifted(cpu_cube[:, 1:, 2], cflag)
    lr = lifted(rec_cube[:, 1:, 2], rlab)
    res["lift_flag"] = dict(gpu_vs_cpu_disagree=int((lg != lc).sum()), gpu_vs_record_disagree=int((lg != lr).sum()),
                            cpu_vs_record_disagree=int((lc != lr).sum()), lifted_cpu=int(lc.sum()), episodes=int(N))
    # arm-table and invalid (other unsupported contact or cube off table)
    cat = C["fmax"][:, :, 3] > 0.01
    cinv = (C["fmax"][:, :, 4:7] > 0.01).any(-1) | C["off"]
    gat, ginv = G["arm_table"][:N, 1:], G["invalid"][:N, 1:]
    full_v = np.ones_like(valid)
    res["arm_table"] = dict(step_agreement_pct_all_steps=float((gat == cat).mean() * 100),
                            episodes_any_cpu=int(cat.any(1).sum()), episodes_any_gpu=int(gat.any(1).sum()),
                            episodes_disagree=int((cat.any(1) != gat.any(1)).sum()),
                            first_step_equal=int(((np.argmax(cat, 1) == np.argmax(gat, 1)) & cat.any(1) & gat.any(1)).sum()),
                            recorded_arm_table_end=int((term == 1).sum()),
                            recorded_end_caught_cpu=int((cat.any(1) & (term == 1)).sum()),
                            recorded_end_caught_gpu=int((gat.any(1) & (term == 1)).sum()))
    res["invalid"] = dict(step_agreement_pct_all_steps=float((ginv == cinv).mean() * 100),
                          episodes_any_cpu=int(cinv.any(1).sum()), episodes_any_gpu=int(ginv.any(1).sum()),
                          episodes_disagree=int((cinv.any(1) != ginv.any(1)).sum()))
    # reset-state flags (one forward at the start state)
    c0 = C["f0"] > 0.01
    res["reset_flags_equal_pct"] = float((G["contact"][:N, 0] == c0[:, :3]).all(-1).mean() * 100)
    # WorldState definitions: GPU rates vs the data rule applied to the CPU replay poses (cast to float32 as stored)
    cp = np.concatenate([C["qpos"][:, :, :6], C["qpos"][:, :, cq:cq + 7]], -1).astype(np.float32).astype(np.float64)
    cp[..., 9:13] *= np.where(cp[..., 9:10] < 0, -1, 1)
    qd, v, w = step_average(cp)
    gq = np.concatenate([G["q"][:N], G["cube_pos"][:N], G["cube_quat"][:N]], -1)
    sqd, sv, sw = step_average(gq)
    fv = valid & full[:, None]
    res["state_definition"] = dict(
        gpu_qd_equals_own_step_average_max=float(np.abs(sqd - G["qd"][:N, 1:]).max()),
        gpu_cube_v_equals_own_step_average_max=float(np.abs(sv - G["cube_v"][:N, 1:]).max()),
        gpu_cube_w_equals_own_step_average_max=float(np.abs(sw - G["cube_w"][:N, 1:]).max()),
        gpu_vs_cpu_qd_abs_p50_p95_max=_pct(np.abs(G["qd"][:N, 1:] - qd)[fv].ravel()),
        gpu_vs_cpu_cube_v_mm_s_p50_p95_max=_pct(1e3 * np.linalg.norm(G["cube_v"][:N, 1:] - v, axis=-1)[fv]),
        gpu_vs_cpu_cube_w_abs_p50_p95_max=_pct(np.linalg.norm(G["cube_w"][:N, 1:] - w, axis=-1)[fv]),
        quat_sign_flips_gpu=int((np.einsum("ntk,ntk->nt", G["cube_quat"][:N, 1:], G["cube_quat"][:N, :-1]) < 0).sum()))
    # overflow
    ov = G["overflow"][:N]
    bits = {name: 1 << b for b, name in enumerate(("NEFC", "NJMAX_NNZ", "BROADPHASE", "NARROWPHASE", "CCD", "HFIELD",
                                                     "CONTACT_MATCH", "NVMAX", "EPA_HORIZON", "ITERATIONS", "LS_ITERATIONS"))}
    res["overflow_worlds_pct"] = dict(any=float((ov != 0).mean() * 100),
                                      **{k: float(((ov & b) != 0).mean() * 100) for k, b in bits.items() if ((ov & b) != 0).any()})
    hit = ov != 0
    if hit.any() and (~hit & full).any():
        k = NCMD
        e = 1e3 * np.linalg.norm(gpu_cube[:, k] - cpu_cube[:, k], axis=-1)
        res["cube_6s_err_mm_overflow_vs_clean"] = dict(overflow=_pct(e[hit & full]), clean=_pct(e[~hit & full]))
    res["gpu_meta"], res["cpu_meta"] = json.loads(str(G["meta"])), json.loads(str(C["meta"]))
    if a.gpu2:   # repeatability: a second GPU run of the same episodes
        G2 = np.load(a.gpu2)
        e = 1e3 * np.linalg.norm(G2["cube_pos"][:N, NCMD] - gpu_cube[:, NCMD], axis=-1)
        res["gpu_run_to_run_cube_6s_mm"] = dict(task_full=_pct(e[task & full]), identical_worlds=int((e == 0).sum()))
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


# ---- 5. speed ---------------------------------------------------------------------------------------------------
def speed(a):
    import torch
    import warp as wp
    from nedm.so101_policy.mjwarp_env import MjwarpWorlds
    E = np.load(a.eps)
    full = (E["family"] <= 6) & (E["lengths"] == 601)
    if a.pool == "task":
        sel = np.flatnonzero(full)
    elif a.pool == "nominal20":      # the critic's workload: 20 nominal full-length validation episodes
        sel = np.flatnonzero(full & (E["family"] == 0))[:20]
    else:
        sel = np.arange(len(E["lengths"]))
    res = dict(rows=[], gpu=torch.cuda.get_device_name(0), pool=a.pool, episodes_pool=int(len(sel)),
               load_avg=os.getloadavg())
    for B in a.nworld:
        rows = sel[np.arange(B) % len(sel)]
        free0 = torch.cuda.mem_get_info()[0]
        t = time.time()
        env = MjwarpWorlds(B, nconmax=a.nconmax, njmax=a.njmax)
        env.reset(None, _start(E, rows))
        act = torch.tensor(E["action"][rows], dtype=torch.float64, device=env.device)
        env.step(act[:, 0])            # capture + first launch
        if a.bare:
            env._run("_g_bare", env._body_bare)
        torch.cuda.synchronize()
        t_setup = time.time() - t
        walls, bares = [], []
        for rep in range(a.reps):      # env and bare runs interleaved (the GPU is shared: same conditions)
            env.reset(None, _start(E, rows))
            torch.cuda.synchronize()
            t = time.perf_counter()
            for k in range(NCMD):
                st = env.step(act[:, k])
            torch.cuda.synchronize()
            walls.append(time.perf_counter() - t)
            ov = env.overflow.cpu().numpy()
            nan = int((~torch.isfinite(st.cube_pos).all(1)).sum())
            if a.bare:                 # the 40 physics steps alone (no contact bookkeeping, no WorldState)
                env.reset(None, _start(E, rows))
                torch.cuda.synchronize()
                t = time.perf_counter()
                for k in range(NCMD):
                    env._targets_t.copy_(act[:, k])
                    with wp.ScopedDevice(env.wdev), wp.ScopedStream(env.stream):
                        wp.launch(_ctrl_only(), dim=(B, 6), inputs=[env.targets, env.tlo, env.thi, env.d.ctrl])
                    env._run("_g_bare", env._body_bare)
                torch.cuda.synchronize()
                bares.append(time.perf_counter() - t)
        row = dict(nworld=B, setup_s=t_setup, wall_s=walls, ctrl_steps_per_s=B * NCMD / min(walls),
                   ctrl_steps_per_s_median=B * NCMD / float(np.median(walls)),
                   physics_steps_per_s=B * NCMD * env.spc / min(walls), bare_wall_s=bares,
                   bare_ctrl_steps_per_s=B * NCMD / min(bares) if bares else None,
                   gpu_mem_GB=(free0 - torch.cuda.mem_get_info()[0]) / 1e9,
                   overflow_worlds_pct=float((ov != 0).mean() * 100),
                   overflow_bits={int(k): int(v) for k, v in zip(*np.unique(ov, return_counts=True))}, nan_worlds=nan)
        print(json.dumps(row), flush=True)
        res["rows"].append(row)
        del env, act, st
        torch.cuda.empty_cache()
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def _ctrl_only():
    from nedm.so101_policy.mjwarp_env import _set_ctrl
    return _set_ctrl


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-train", action="store_true")
    p = sub.add_parser("cpu")
    p.add_argument("--eps", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk", type=int, default=24)
    p.add_argument("--limit", type=int)
    p = sub.add_parser("gpu")
    p.add_argument("--eps", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--nconmax", type=int, default=48)
    p.add_argument("--njmax", type=int, default=192)
    p.add_argument("--cube-rates", default="step_average")
    p = sub.add_parser("compare")
    p.add_argument("--eps", required=True)
    p.add_argument("--cpu", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--gpu2")
    p.add_argument("--system", default=os.path.expanduser("~/so101_nrd_fix/data/val_only/system.json"))
    p.add_argument("--out", required=True)
    p = sub.add_parser("speed")
    p.add_argument("--eps", required=True)
    p.add_argument("--nworld", type=int, nargs="+", default=[4096, 8192, 16384])
    p.add_argument("--nconmax", type=int, default=48)
    p.add_argument("--njmax", type=int, default=192)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--bare", action="store_true")
    p.add_argument("--pool", default="task", choices=["task", "nominal20", "all"])
    p.add_argument("--out")
    a = ap.parse_args()
    globals()[a.cmd](a)


if __name__ == "__main__":
    main()
