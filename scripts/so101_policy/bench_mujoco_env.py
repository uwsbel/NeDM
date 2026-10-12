"""Checks and throughput of nedm.so101_policy.mujoco_env (batched CPU MuJoCo worlds of the SO101 grasp collector).

    PYTHONPATH=src python scripts/so101_policy/bench_mujoco_env.py extract --data ~/so101_nrd_fix/data/val_only \
        --out DIR [--per-family 8]
    PYTHONPATH=src python scripts/so101_policy/bench_mujoco_env.py replay --data DIR [--workers 8] [--out JSON] \
        [--labels exact|record] [--modes AB] [--python WORKER_PYTHON]
    PYTHONPATH=src python scripts/so101_policy/bench_mujoco_env.py throughput --data DIR --workers 1 8 24 \
        [--per-worker 16] [--steps 300] [--labels exact|record] [--no-overlap] [--python WORKER_PYTHON] [--out JSON]
    PYTHONPATH=src python scripts/so101_policy/bench_mujoco_env.py starts [--first 5000000] [--n 256] [--out JSON]

extract: a small data directory (system.json + unified_data.npz, the arrays the replay needs) of validation episodes
(split 1, never the sealed test): every episode with a terminal reason plus ``per-family`` evenly spread episodes of
each family.

replay: (A) collector-exact. Each episode's start is regenerated from its stored seed with the collector's planner
(production master seed 20261021; the plan whose warm-up state equals the stored first record), the env resets with
the collector warm-up and steps the stored commands (every second record). The WorldState sequence is compared with
the stored records as the NRD pipeline reads them (rollouts.load_rows, data.step_average_rates,
data.physical_labels and data.model_labels against the two contact views), the env's 10 ms records and interval labels
against the stored ones, together with the end step and reason. ``--labels record`` measures how often the cheaper
once-per-record force check changes labels or ends. (B) state start: the env resets exactly to the stored
first record (float32, no warm-up) and steps the same commands; the divergence from the record is reported.

throughput: worlds = workers x per-worker, each replays a nominal episode (collector warm-up from its first record's
pose, stored commands); control steps per wall second of the step calls (resets excluded), plus one core of bare
``mj_step(nstep=40)`` on the same episodes as the reference.

starts: wall time of ``sample_starts`` per start (collector sampler + 0.3 s warm-up + flags), one process.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from nedm.contact_nrd.collect import export
from nedm.contact_nrd.collect import so101_grasp as g
from nedm.contact_nrd.data import model_labels, physical_labels, step_average_rates
from nedm.contact_nrd.rollouts import load_rows, npz_members
from nedm.so101_policy.mujoco_env import MujocoWorlds, end_names, install_overlap, sample_starts

MASTER = 20261021   # production campaign (its validation split = val_only)
FIELDS = ("q", "qd", "cube_pos", "cube_quat", "cube_v", "cube_w")
KEEP = ("arm", "cube", "action", "contacts", "lengths", "splits", "family", "family_index", "seed", "terminal_reason",
        "plan_tries", "episode_id", "visual_overlap", "phase")


# ---- extract ------------------------------------------------------------------------------------------------------
def extract(data, out, per_family):
    a = npz_members(Path(data) / "unified_data.npz")
    fam, split, term = np.asarray(a["family"]), np.asarray(a["splits"]), np.asarray(a["terminal_reason"])
    if (split != 1).any():
        raise SystemExit("extract reads validation data only (split 1)")
    rows = set(np.flatnonzero(term != 0).tolist())
    for f in np.unique(fam):
        cand = np.flatnonzero((fam == f) & (term == 0))
        rows |= set(cand[((np.arange(per_family) + 0.5) * len(cand) / per_family).astype(int)].tolist())
    rows = np.array(sorted(rows))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(data) / "system.json", out / "system.json")
    np.savez(out / "unified_data.npz", source_row=rows, **{k: np.array(a[k][rows]) for k in KEEP})
    print(f"{len(rows)} episodes -> {out} (terminal reasons {np.bincount(term[rows], minlength=7).tolist()})")


# ---- regenerate the collector's start of a stored episode ----------------------------------------------------------
_S = None


def _plan_init():
    global _S
    _S = g.Scene(g.load_config())
    install_overlap(_S)   # run_episode (replans) and the jaw probes' planning need the overlap grids


def _first_record(s):
    arm, cube = s.state()
    cube[3:7] = export.quat_continuous(cube[None, 3:7])[0]
    return arm.astype(np.float32), cube.astype(np.float32)


def regenerate(job):
    """(family, family_index, seed, stored first arm [12], cube [13]) -> plan start (q0, cube pos, quat, vel, omega,
    spawn) whose collector warm-up gives the stored first record, and the number of plans tried."""
    fam, index, seed, arm0, cube0 = job
    s = _S
    cfg_fam = s.cfg["families"][fam]
    if g.episode_seed(MASTER, fam, index) != seed:
        raise ValueError(f"seed of family {fam} index {index} is not from master {MASTER}")
    rng = np.random.default_rng(seed)
    n_rec = int(cfg_fam["records"])
    n_ctrl = (n_rec - 1) // s.per_ctrl + 1
    tries = 0
    while tries < 200:
        tries += 1
        try:
            plan = g.FAMILIES[cfg_fam["name"]](s, rng, n_ctrl)
        except g.PlanError:
            continue
        c = plan["cube"]
        s.reset(plan["q0"], c["pos"], c["quat"], c.get("vel"), c.get("omega"), c["spawn"])
        arm, cube = _first_record(s)
        if np.array_equal(arm, arm0) and np.array_equal(cube, cube0):
            z = np.zeros(3)
            return dict(q=np.asarray(plan["q0"], float), pos=np.asarray(c["pos"], float), quat=np.asarray(c["quat"], float),
                        vel=np.asarray(c.get("vel", z), float), omega=np.asarray(c.get("omega", z), float),
                        spawn=bool(c["spawn"]), tries=tries)
        if not isinstance(g.run_episode(s, plan, n_rec), str):   # the collector kept this plan: not our episode
            raise RuntimeError(f"family {fam} index {index}: the kept plan does not give the stored first record")
    raise RuntimeError(f"family {fam} index {index}: no plan after {tries} tries")


# ---- replay -------------------------------------------------------------------------------------------------------
def _truth(root, rows):
    data = load_rows(root, rows, "cpu", 0.02)
    lengths = data["lengths"].numpy()
    step_average_rates(data, {"articulated", "rigid"})
    st = data["states"][:, ::2]                                       # model grid: records 0, 2, 4, ...
    truth = dict(q=st[:, :, 0, 0:6], qd=st[:, :, 0, 6:12], cube_pos=st[:, :, 1, 0:3], cube_quat=st[:, :, 1, 3:7],
                 cube_v=st[:, :, 1, 7:10], cube_w=st[:, :, 1, 10:13])
    phys = physical_labels(data["contacts"], 2)                       # [E, 300, 3]: pair touched in step n
    truth["gate"] = model_labels(data["contacts"], 2)                 # step n or n-1 (the NRD switch meaning)
    a = npz_members(Path(root) / "unified_data.npz")                  # the stored 10 ms records (float32) and labels
    truth["records"] = torch.from_numpy(np.concatenate((np.asarray(a["arm"])[rows], np.asarray(a["cube"])[rows]), -1)
                                        .astype(np.float64))
    truth["record_contact"] = torch.from_numpy(np.asarray(a["contacts"])[rows].astype(bool))
    return truth, phys, data["actions"][:, ::2, 0, :6], lengths


def _run(env, actions, steps, reset):
    """reset() then ``steps`` commands -> {field: [E, steps+1, ...]} (also ``gate`` = switch_contact, ``physical``,
    ``records`` [E, 2 steps, 25] and ``record_contact`` [E, 2 steps, 3] = records 1, 2, ... and intervals 0, 1, ...),
    end reason and step, wall time of the step calls."""
    ws = reset()
    seq = {f: [getattr(ws, f)] for f in FIELDS + ("contact", "arm_table", "invalid")}
    seq["gate"], seq["physical"], rec, rc = [env.switch_contact], [env.physical_contact], [], []
    t0 = time.perf_counter()
    for n in range(steps):
        ws = env.step(actions[:, n])
        for f in FIELDS + ("contact", "arm_table", "invalid"):
            seq[f].append(getattr(ws, f))
        seq["gate"].append(env.switch_contact)
        seq["physical"].append(env.physical_contact)
        rec.append(env.records)
        rc.append(env.record_contact)
    wall = time.perf_counter() - t0
    out = {f: torch.stack(v, 1) for f, v in seq.items()}
    out["records"], out["record_contact"] = torch.cat(rec, 1), torch.cat(rc, 1)
    return out, env.end_reason.clone(), env.end_step.clone(), wall


def compare(seq, end_reason, end_step, truth, phys, lengths, terminal, exact=True):
    """Per-episode comparison over the stored part (module docstring): grid states (max |diff| per field), the 10 ms
    records 1 .. L-1 (``records_max_abs``), physical and gate contact labels per step, interval labels per 10 ms,
    flags before the end, end step and reason."""
    out = []
    for e in range(len(lengths)):
        L = int(lengths[e])
        nv = (L - 1) // 2 + 1                                         # stored grid states 0 .. nv-1
        r = dict(valid_states=nv, terminal=int(terminal[e]))
        r["max_abs"] = {f: float((seq[f][e, :nv] - truth[f][e, :nv]).abs().max()) for f in FIELDS}
        r["contact_equal"] = bool(torch.equal(seq["physical"][e, 1:nv], phys[e, :nv - 1]))
        r["contact_mismatch_steps"] = int((seq["physical"][e, 1:nv] != phys[e, :nv - 1]).any(-1).sum())
        r["gate_equal"] = bool(torch.equal(seq["gate"][e, 1:nv], truth["gate"][e, :nv - 1]))
        r["records_max_abs"] = float((seq["records"][e, :L - 1] - truth["records"][e, 1:L]).abs().max()) if L > 1 else 0.
        r["record_contact_equal"] = bool(torch.equal(seq["record_contact"][e, :L - 1],
                                                     truth["record_contact"][e, :L - 1]))
        r["flags_before_end"] = int((seq["arm_table"][e, :nv] | seq["invalid"][e, :nv]).sum())
        er, es = int(end_reason[e]), int(end_step[e])
        if terminal[e]:
            r["end_ok"] = er == int(terminal[e]) and es == nv
        else:
            r["end_ok"] = er == 0 or es >= nv
        r["end"] = [er, es]
        if not exact:
            err = (seq["cube_pos"][e, :nv] - truth["cube_pos"][e, :nv]).norm(dim=-1)
            r["cube_err_mm"] = {f"{t}s": float(err[min(nv - 1, int(t / 0.02))]) * 1e3 for t in (0.5, 1, 2, 4, 6)}
            r["arm_err_mrad_max"] = float((seq["q"][e, :nv] - truth["q"][e, :nv]).abs().max()) * 1e3
        same = (r["contact_equal"] and r["gate_equal"] and r["record_contact_equal"] and r["end_ok"]
                and r["flags_before_end"] == 0)
        r["bit_exact"] = same and all(v == 0.0 for v in r["max_abs"].values()) and r["records_max_abs"] == 0.0
        r["exact_1e12"] = same and all(v <= 1e-12 for v in r["max_abs"].values()) and r["records_max_abs"] == 0.0
        out.append(r)
    return out


def replay(root, workers, python, out_path, plan_workers, labels="exact", modes=("A", "B")):
    root = Path(root)
    a = npz_members(root / "unified_data.npz")
    system = json.loads((root / "system.json").read_text())
    names = [f["name"] for f in system["families"]]
    cfg = g.load_config()
    for k, n in enumerate(names):
        if cfg["families"][k]["name"] != n:
            raise SystemExit(f"config family {k} is {cfg['families'][k]['name']}, the data's is {n}")
    E = len(a["lengths"])
    rows = np.arange(E)
    fam, idx, seed = np.asarray(a["family"]), np.asarray(a["family_index"]), np.asarray(a["seed"])
    term, plan_tries = np.asarray(a["terminal_reason"]), np.asarray(a["plan_tries"])
    jobs = [(int(fam[e]), int(idx[e]), int(seed[e]), np.array(a["arm"][e, 0]), np.array(a["cube"][e, 0])) for e in rows]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(plan_workers, initializer=_plan_init) as pool:
        plans = list(pool.map(regenerate, jobs, chunksize=4))
    t_plan = time.perf_counter() - t0
    truth, phys, actions, lengths = _truth(root, rows)
    steps = int(((lengths - 1) // 2).max())
    env = MujocoWorlds(E, workers=workers, python=python, labels=labels)
    spawn = np.array([p["spawn"] for p in plans])

    def reset_collector():
        for sp in (False, True):
            sel = np.flatnonzero(spawn == sp)
            if len(sel):
                env.reset(sel, dict(q=np.stack([plans[i]["q"] for i in sel]), cube_pos=np.stack([plans[i]["pos"] for i in sel]),
                                    cube_quat=np.stack([plans[i]["quat"] for i in sel]),
                                    cube_v=np.stack([plans[i]["vel"] for i in sel]),
                                    cube_w=np.stack([plans[i]["omega"] for i in sel])), warmup=True, spawn=bool(sp))
        return env.state()

    def reset_state():
        first = {f: truth[f][:, 0] for f in FIELDS}
        return env.reset(None, first, warmup=False)

    res = {}
    runs = (("A_collector_exact", reset_collector, True), ("B_state_start", reset_state, False))
    for name, fn, exact in [r for r in runs if r[0][0] in modes]:
        seq, er, es, wall = _run(env, actions, steps, fn)
        per = compare(seq, er, es, truth, phys, lengths, term, exact)
        groups = {}
        for e, r in enumerate(per):
            groups.setdefault(names[fam[e]], []).append(r)
        summ = {}
        for gname, rs in [("all", per)] + sorted(groups.items()):
            s = dict(episodes=len(rs), bit_exact=sum(r["bit_exact"] for r in rs), exact_1e12=sum(r["exact_1e12"] for r in rs),
                     contact_equal=sum(r["contact_equal"] for r in rs), gate_equal=sum(r["gate_equal"] for r in rs),
                     record_contact_equal=sum(r["record_contact_equal"] for r in rs),
                     end_ok=sum(r["end_ok"] for r in rs), max_abs={f: max(r["max_abs"][f] for r in rs) for f in FIELDS},
                     records_max_abs=max(r["records_max_abs"] for r in rs),
                     end_wrong=[[r["terminal"], r["end"]] for r in rs if not r["end_ok"]][:20],
                     terminal_episodes=sum(r["terminal"] != 0 for r in rs))
            if not exact:
                for t in ("0.5s", "1s", "2s", "4s", "6s"):
                    v = np.array([r["cube_err_mm"][t] for r in rs])
                    s[f"cube_err_mm_{t}"] = dict(p50=float(np.median(v)), p95=float(np.quantile(v, 0.95)), max=float(v.max()))
            summ[gname] = s
        res[name] = dict(summary=summ, wall_s=wall, steps=steps,
                         episodes=[dict(row=int(e), source_row=int(np.asarray(a["source_row"])[e]), family=names[fam[e]], **r)
                                   for e, r in enumerate(per)])
        print(name, json.dumps(summ["all"]), flush=True)
    env.close()
    info = dict(data=str(root), episodes=E, families={n: int((fam == k).sum()) for k, n in enumerate(names)},
                terminal_counts={n: int((term == k).sum()) for k, n in enumerate(end_names())},
                plan_tries_match=int(sum(int(p["tries"]) == int(t) for p, t in zip(plans, plan_tries))),
                plan_s=t_plan, workers=workers, labels=labels, host=os.uname().nodename, env=env.info[0])
    out = dict(info=info, **res)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=1))
    return out


# ---- throughput ---------------------------------------------------------------------------------------------------
def _nominal(root):
    a = npz_members(Path(root) / "unified_data.npz")
    system = json.loads((Path(root) / "system.json").read_text())
    k = [f["name"] for f in system["families"]].index("nominal")
    sel = np.flatnonzero((np.asarray(a["family"]) == k) & (np.asarray(a["lengths"]) == 601))
    return np.array(a["arm"][sel, 0]), np.array(a["cube"][sel, 0]), np.array(a["action"][sel, ::2][:, :300])


def bare_core(root, seconds=10.0):
    """One core: bare mj_step(nstep=40) replays of the nominal episodes (Scene.reset warm-up, stored commands)."""
    import mujoco
    arm, cube, act = _nominal(root)
    s = g.Scene(g.load_config())
    n, el, e = 0, 0.0, 0
    while el < seconds:
        i = e % len(arm)
        s.reset(arm[i, :6].astype(float), cube[i, :3].astype(float), cube[i, 3:7].astype(float))
        t0 = time.perf_counter()
        for u in act[i]:
            s.d.ctrl[s.act] = u
            mujoco.mj_step(s.m, s.d, nstep=40)
        el += time.perf_counter() - t0
        n += len(act[i])
        e += 1
    return n / el


def throughput(root, workers_list, per_worker, steps, overlap, python, out_path, bare_s, labels="exact", cpus=None):
    arm, cube, act = _nominal(root)
    res = dict(host=os.uname().nodename, cpus=os.cpu_count(), load_before=os.getloadavg(), per_worker=per_worker,
               steps=steps, overlap=overlap, labels=labels, affinity=sorted(os.sched_getaffinity(0)), runs=[])
    if bare_s:
        res["bare_mj_step_one_core"] = bare_core(root, bare_s)
        print(f"bare mj_step nstep=40, one core: {res['bare_mj_step_one_core']:.0f} steps/s", flush=True)
    for w in workers_list:
        B = w * per_worker
        k = np.arange(B) % len(arm)
        t0 = time.perf_counter()
        env = MujocoWorlds(B, workers=w, overlap=overlap, python=python, labels=labels, cpus=cpus)
        t_init = time.perf_counter() - t0
        t0 = time.perf_counter()
        env.reset(None, dict(q=arm[k, :6], cube_pos=cube[k, :3], cube_quat=cube[k, 3:7]), warmup=True)
        t_reset = time.perf_counter() - t0
        u = torch.as_tensor(act[k].astype(np.float64))
        t0 = time.perf_counter()
        for n in range(steps):
            env.step(u[:, n])
        wall = time.perf_counter() - t0
        env.close()
        r = dict(workers=w, worlds=B, init_s=t_init, reset_s=t_reset, step_wall_s=wall, steps_per_s=B * steps / wall,
                 per_worker_steps_per_s=B * steps / wall / max(w, 1), load=os.getloadavg())
        res["runs"].append(r)
        print(json.dumps(r), flush=True)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(res, indent=1))
    return res


def starts(first, n, out_path):
    """Time sample_starts on seeds first .. first+n-1 (in this process) -> ms per start, draws per start."""
    t0 = time.perf_counter()
    st = sample_starts(range(first, first + n))
    wall = time.perf_counter() - t0
    r = dict(host=os.uname().nodename, seeds=[first, first + n - 1], ms_per_start=1e3 * wall / n,
             tries=np.bincount(st["tries"]).tolist(), load=os.getloadavg(),
             cube_r_m=np.quantile(np.hypot(st["cube_pos"][:, 0] - 0.0388, st["cube_pos"][:, 1]), [0, 0.5, 1]).tolist())
    print(json.dumps(r), flush=True)
    if out_path:
        Path(out_path).write_text(json.dumps(r, indent=1))
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--data", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--per-family", type=int, default=8)
    r = sub.add_parser("replay")
    r.add_argument("--data", required=True)
    r.add_argument("--workers", type=int, default=0)
    r.add_argument("--plan-workers", type=int, default=4)
    r.add_argument("--python", default=None)
    r.add_argument("--out", default=None)
    r.add_argument("--labels", default="exact", choices=("exact", "record"))
    r.add_argument("--modes", default="AB", help="A = collector warm-up starts, B = exact state starts")
    t = sub.add_parser("throughput")
    t.add_argument("--data", required=True)
    t.add_argument("--workers", type=int, nargs="+", default=[1, 8])
    t.add_argument("--per-worker", type=int, default=16)
    t.add_argument("--steps", type=int, default=300)
    t.add_argument("--no-overlap", action="store_true")
    t.add_argument("--bare-seconds", type=float, default=10.0)
    t.add_argument("--python", default=None)
    t.add_argument("--labels", default="exact", choices=("exact", "record"))
    t.add_argument("--out", default=None)
    t.add_argument("--cpus", type=int, nargs="*", default=None, help="worker CPU affinity")
    s = sub.add_parser("starts")
    s.add_argument("--first", type=int, default=5_000_000)
    s.add_argument("--n", type=int, default=256)
    s.add_argument("--out", default=None)
    a = p.parse_args()
    if a.cmd == "extract":
        extract(a.data, a.out, a.per_family)
    elif a.cmd == "replay":
        replay(a.data, a.workers, a.python, a.out, a.plan_workers, a.labels, tuple(a.modes))
    elif a.cmd == "starts":
        starts(a.first, a.n, a.out)
    else:
        throughput(a.data, a.workers, a.per_worker, a.steps, not a.no_overlap, a.python, a.out, a.bare_seconds, a.labels,
                   a.cpus)


if __name__ == "__main__":
    main()
