"""VIRAL / DoorMan-style camera student of build9 (``nedm.so101_policy.vision.viral``; docs
~/so101_policy/docs/build9_viral_student.md). The teacher is the Markov joint teacher (``vision_viral_teacher.py``).

    export PYTHONPATH=src PY=~/miniconda3/envs/so101_vision/bin/python
    $PY -P scripts/so101_policy/vision_viral.py run --out RUN --jt JT.pt --tasks train.npz --eval-tasks dev100.npz \
        --config configs/so101_policy/vision/viral_v1.json --r0-tasks 2048 --rounds 4 --round-tasks 1024 \
        --teacher-share 0.5
    $PY -P scripts/so101_policy/vision_viral.py eval --student RUN/r4/last.pt --jt JT.pt --tasks dev100.npz \
        --world mujoco [--randomise] [--record-states] --out e.npz
    $PY -P scripts/so101_policy/vision_viral.py collect --jt JT.pt --tasks t.npz --world nrd --out d.npz [--student S]

``run`` (VIRAL's mixing per WHOLE episode, never inside one): round 0 = JT-driven episodes (``--r0-noise-frac`` of them
with noise ``--noise`` x the JT's PPO std on its normalised action); round i >= 1 = ``--round-tasks`` new tasks, the
share ``--teacher-share`` JT-driven (the same noise mix), the rest driven by the previous round's student with
randomised visuals (``--collect-randomise``); every visited decision labelled by the JT. After each round: train
(round 0 from ImageNet, later rounds fine-tune on the aggregate), evaluate on the development tasks in CPU MuJoCo (and
``--eval-worlds``), nominal visuals (+ randomised with ``--eval-randomise``). Files resume.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy import joint_env as JE
from nedm.so101_policy import joint_teacher as JT
from nedm.so101_policy.vision import viral as VI

NRD = "~/so101_policy/ckpt/nrd_c1_s62.pt"
MJ_PY = "~/miniconda3/envs/nrdsim/bin/python"


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def x(p):
    return str(Path(p).expanduser()) if p else p


def progress(out, **kw):
    with open(Path(out) / "progress.jsonl", "a") as f:
        f.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), **kw), default=float) + "\n")


def load_tasks(path, idx=None):
    from nedm.so101_policy.vision.demos import load_tasks as lt
    return lt(path, idx)


def subset(t, idx):
    n = len(t["seeds"])
    return {k: (v[idx] if np.ndim(v) and len(v) == n else v) for k, v in t.items()}


class EnvCache:
    def __init__(self, jt, device, nrd, workers, mj_python, horizon):
        self.c, self.jt, self.dev, self.nrd, self.w, self.mp, self.h = {}, jt, device, nrd, workers, mj_python, horizon

    def get(self, world, B):
        k = (world, B)
        if k not in self.c:
            self.c[k] = JT.make_env(world, B, self.jt.codec, self.dev, self.nrd, self.w, self.mp, self.h)
        return self.c[k]


@torch.no_grad()
def collect(tasks, world, jt, cache: EnvCache, src, noise, seed, batch, student=None, log=log):
    """Episodes on all tasks in batches -> concatenated record dict (``run_tasks`` layout)."""
    n = len(tasks["seeds"])
    parts, t0 = [], time.time()
    for s0 in range(0, n, batch):
        idx = np.arange(s0, min(n, s0 + batch))
        env = cache.get(world, len(idx))
        r = JT.run_tasks(env, subset(tasks, idx), jt, student, src[idx], noise[idx], seed, record=True)
        parts.append(r)
        s = np.asarray(r["src"]) == 1
        msg = f"  collect {world}: tasks {s0}-{idx[-1]}, lenient {r['lenient'].mean() * 100:.1f} %"
        if s.any():
            msg += f" (student-driven {r['lenient'][s].mean() * 100:.1f} % of {int(s.sum())}"
            msg += f", JT-driven {r['lenient'][~s].mean() * 100:.1f} %)" if (~s).any() else ")"
        log(msg + f", {time.time() - t0:.0f} s")
    return concat(parts)


def concat(parts):
    if len(parts) == 1:
        return parts[0]
    out, e0, r0 = {}, 0, 0
    for p in parts:
        q = dict(p)
        q["row_ep"] = p["row_ep"].astype(np.int64) + e0
        q["ep_row_start"] = p["ep_row_start"] + r0
        e0 += len(p["end"])
        r0 += len(p["row_ep"])
        for k, v in q.items():
            if np.ndim(v) == 0:
                continue
            out.setdefault(k, []).append(v)
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["wall_s"] = np.float64(sum(float(p["wall_s"]) for p in parts))
    return out


def evaluate(st, jt, tasks, world, cache, views, randomise=False, jaw_lead=None, record_states=False,
             full_horizon=False, device="cuda"):
    actor = VI.Actor(st, views, randomise=randomise, jaw_lead=jaw_lead, device=device)
    env = cache.get(world, len(tasks["seeds"]))
    r = JT.run_tasks(env, tasks, jt, actor, np.ones(len(tasks["seeds"]), np.int64), None, 0,
                     record=True, record_states=record_states, full_horizon=full_horizon)
    r["render_s"], r["model_s"] = np.float64(actor.t_render), np.float64(actor.t_model)
    r["summary"] = np.array(json.dumps(JT.summarize(r)))
    r["world"] = np.array(world)
    r["randomise"] = np.bool_(randomise)
    r["jaw_lead"] = np.int64(actor.jaw_lead)
    return r


def agree(r_a, r_b):
    a, b = np.asarray(r_a["lenient"]).astype(bool), np.asarray(r_b["lenient"]).astype(bool)
    return dict(both=int((a & b).sum()), a_only=int((a & ~b).sum()), b_only=int((~a & b).sum()),
                neither=int((~a & ~b).sum()), agreement=float((a == b).mean()))


def cmd_run(a):
    out = Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    if not (out / "args.json").exists():
        (out / "args.json").write_text(json.dumps({k: v for k, v in vars(a).items() if k != "fn"}, indent=1, default=str))
    jt = JE.JointTeacher(x(a.jt), a.device)
    cfg_d = json.loads(Path(a.config).read_text())
    for s in a.set or []:
        k, v = s.split("=", 1)
        cfg_d[k] = json.loads(v)
    cfg_d["seed"] = a.seed
    cfg = VI.ViralConfig.from_dict(cfg_d)
    tasks = load_tasks(a.tasks)
    from nedm.so101_policy.vision.teacher import check_seeds
    check_seeds(tasks["seeds"])
    ev = load_tasks(a.eval_tasks, np.arange(a.eval_n) if a.eval_n else None)
    for p_ in [a.eval_tasks] + list(a.guard_tasks or []):          # no training task may be an evaluation task
        hit = set(np.asarray(tasks["seeds"]).tolist()) & set(np.asarray(load_tasks(p_)["seeds"]).tolist())
        if hit:
            raise ValueError(f"{len(hit)} training task seeds are in {p_}")
    cache = EnvCache(jt, a.device, x(a.nrd_ckpt), a.eval_workers, x(a.mj_python), a.horizon)
    views = VI.DRViews(a.render_worlds, cfg.res, cfg.cameras, dr=VI.DRConfig.from_dict(cfg.dr), seed=a.seed)
    rows = [np.arange(0, a.r0_tasks)]
    for i in range(a.rounds):
        s0 = a.r0_tasks + i * a.round_tasks
        rows.append(np.arange(s0, s0 + a.round_tasks))
    if rows[-1][-1] >= len(tasks["seeds"]):
        raise ValueError(f"{len(tasks['seeds'])} tasks, need {rows[-1][-1] + 1}")
    files = [x(f) for f in (a.extra_data or [])]
    prev = x(a.init_from) if a.init_from else None
    for i in range(a.rounds + 1):
        f = out / "data" / f"r{i}.npz"
        t = subset(tasks, rows[i])
        n = len(rows[i])
        if not f.exists():
            t0 = time.time()
            rng = np.random.default_rng(a.seed * 1000 + i)
            share = 1.0 if i == 0 else a.teacher_share
            src = np.where(rng.random(n) < share, 0, 1).astype(np.int64)
            if prev is None:
                src[:] = 0
            noisy = (src == 0) & (rng.random(n) < a.r0_noise_frac)
            noise = np.where(noisy, a.noise, 0.0)
            student = None
            if (src == 1).any():
                st, _ = VI.load_student(prev, a.device)
                student = VI.Actor(st, views, randomise=a.collect_randomise, device=a.device)
            d = collect(t, a.world, jt, cache, src, noise, a.seed + 1000 * i, a.batch, student)
            d["meta"] = np.array(json.dumps(dict(round=i, world=a.world, jt=jt.info(), teacher_share=share,
                                                 noise=a.noise, student=str(prev) if student else "",
                                                 collect_randomise=a.collect_randomise, wall_s=time.time() - t0)))
            np.savez_compressed(f, **d)
            s = d["src"] == 1
            progress(out, stage=f"data r{i}", wall_s=time.time() - t0, rows=int(len(d["row_ep"])),
                     lenient_jt=float(d["lenient"][~s].mean()) if (~s).any() else None,
                     lenient_student=float(d["lenient"][s].mean()) if s.any() else None, n_student=int(s.sum()))
            del d, student
            torch.cuda.empty_cache()
        files.append(str(f))
        ck = out / f"r{i}" / "last.pt"
        if not ck.exists():
            t0 = time.time()
            data = VI.StudentSet(files, a.device, cfg.chunk, seed=a.seed)
            first = i == 0 and not a.init_from
            steps = a.train_steps0 if first else a.train_steps
            VI.train(data, views, cfg, out / f"r{i}", max_steps=steps, init_from=None if first else prev, log=log,
                     device=a.device, extra_meta=dict(argv=sys.argv))
            progress(out, stage=f"train r{i}", wall_s=time.time() - t0, steps=steps, rows=data.n_rows,
                     eps=data.n_eps)
            del data
            torch.cuda.empty_cache()
        prev = str(ck)
        if i % a.eval_every == 0 or i == a.rounds:
            st, _ = VI.load_student(ck, a.device)
            res = {}
            for w in a.eval_worlds:
                for rnd in ([False, True] if a.eval_randomise else [False]):
                    tag = f"{w}{'_dr' if rnd else ''}"
                    fe = out / f"r{i}" / f"eval_{tag}.npz"
                    if not fe.exists():
                        t0 = time.time()
                        r = evaluate(st, jt, ev, w, cache, views, rnd, device=a.device)
                        np.savez_compressed(fe, **r)
                        s = json.loads(str(r["summary"]))
                        progress(out, stage=f"eval r{i} {tag}", wall_s=time.time() - t0, **s)
                        log(f"r{i} eval {tag}: lenient {s['lenient'] * 100:.0f} % strict {s['strict'] * 100:.0f} % "
                            f"lifted {s['lifted'] * 100:.0f} % causes {s['causes_not_lenient']}")
                    res[tag] = dict(np.load(fe))
            if "mujoco" in res and "nrd" in res:
                progress(out, stage=f"agree r{i}", **agree(res["mujoco"], res["nrd"]))
            del st


def cmd_online(a):
    """VIRAL's online loop (``viral.OnlineDAgger``) from an initial student checkpoint; config overrides of the student
    (``--config`` keys that differ, e.g. qd_noise, jaw_lead, dr) are applied to the loaded student."""
    import csv
    out = Path(a.out)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({k: v for k, v in vars(a).items() if k != "fn"}, indent=1, default=str))
    jt = JE.JointTeacher(x(a.jt), a.device)
    st, ck = VI.load_student(x(a.init), a.device)
    if a.config:
        over = json.loads(Path(a.config).read_text())
        for s_ in a.set or []:
            k, v = s_.split("=", 1)
            over[k] = json.loads(v)
        from dataclasses import asdict as _ad
        cd = _ad(st.cfg)
        hist = int(over.get("hist", cd["hist"]))
        cd.update({k: v for k, v in over.items() if k in cd and k != "hist"})
        st.cfg = VI.ViralConfig.from_dict(cd)
        if hist != st.cfg.hist:
            st = VI.expand_hist(st, hist, a.device)
    oc = json.loads(Path(a.online_config).read_text()) if a.online_config else {}
    for s_ in a.oset or []:
        k, v = s_.split("=", 1)
        oc[k] = json.loads(v)
    ocfg = VI.OnlineConfig.from_dict(oc)
    tasks = load_tasks(a.tasks)
    from nedm.so101_policy.vision.teacher import check_seeds
    check_seeds(tasks["seeds"])
    ev = load_tasks(a.eval_tasks, np.arange(a.eval_n) if a.eval_n else None)
    for p_ in [a.eval_tasks] + list(a.guard_tasks or []):
        hit = set(np.asarray(tasks["seeds"]).tolist()) & set(np.asarray(load_tasks(p_)["seeds"]).tolist())
        if hit:
            raise ValueError(f"{len(hit)} training task seeds are in {p_}")
    dr = VI.DRConfig.from_dict(st.cfg.dr)
    views = VI.DRViews(ocfg.envs, st.cfg.res, st.cfg.cameras, dr=dr, seed=a.seed)
    eviews = VI.DRViews(a.render_worlds, st.cfg.res, st.cfg.cameras, dr=dr, seed=a.seed + 1)
    cache = EnvCache(jt, a.device, x(a.nrd_ckpt), a.eval_workers, x(a.mj_python), a.horizon)
    od = VI.OnlineDAgger(st, jt, tasks, x(a.nrd_ckpt), views, ocfg, a.device, a.seed, log)
    (out / "meta.json").write_text(json.dumps(dict(student_cfg=VI.asdict(st.cfg), online=VI.asdict(ocfg),
                                                   init=str(a.init), jt=jt.info(), argv=sys.argv), indent=1,
                                              default=str))
    f = open(out / "curve.csv", "a", newline="")
    wr = None
    t0 = time.time()
    acc, tacc, n_acc = {}, {}, 0
    min_fill = max(st.cfg.batch * 4, ocfg.envs)
    base_lr = [g_["lr"] for g_ in od.opt.param_groups]
    for it in range(1, ocfg.iterations + 1):
        od.it = it - 1
        if ocfg.lr_decay_start >= 0 and it > ocfg.lr_decay_start:
            u = (it - ocfg.lr_decay_start) / max(1, ocfg.iterations - ocfg.lr_decay_start)
            for g_, b_ in zip(od.opt.param_groups, base_lr):
                g_["lr"] = b_ * (1 - (1 - ocfg.lr_final_frac) * u)
        T = od.act()
        tt = time.time()
        info = od.train_steps() if od.buf.n >= min_fill else {}
        T["train"] = time.time() - tt
        for k, v in info.items():
            acc[k] = acc.get(k, 0.0) + v
        for k, v in T.items():
            tacc[k] = tacc.get(k, 0.0) + v
        n_acc += 1
        if it % ocfg.log_every == 0:
            s_ = od.stats
            row = dict(it=it, wall_min=round((time.time() - t0) / 60, 2), samples=od.samples, buffer=od.buf.n,
                       **{k: round(v / n_acc, 4) for k, v in acc.items()},
                       **{f"t_{k}": round(v / n_acc, 3) for k, v in tacc.items()},
                       ep_student=s_["ep_student"], len_student=round(s_["len_student"] / max(s_["ep_student"], 1), 3),
                       ep_teacher=s_["ep_teacher"], len_teacher=round(s_["len_teacher"] / max(s_["ep_teacher"], 1), 3))
            od.stats = {k: 0 for k in s_}
            acc, tacc, n_acc = {}, {}, 0
            if wr is None:
                wr = csv.DictWriter(f, fieldnames=list(row), extrasaction="ignore")
                wr.writeheader()
            wr.writerow(row)
            f.flush()
            log(json.dumps(row))
        if it % ocfg.save_every == 0 or it == ocfg.iterations:
            VI.save_student(st, out / "ckpt" / f"it_{it:05d}.pt", it=it, samples=od.samples)
            VI.save_student(st, out / "last.pt", it=it, samples=od.samples)
        if it % ocfg.eval_every == 0 or it == ocfg.iterations:
            st.eval()
            for rnd in ([False, True] if a.eval_randomise else [False]):
                tag = f"mujoco{'_dr' if rnd else ''}"
                te = time.time()
                r = evaluate(st, jt, ev, "mujoco", cache, eviews, rnd, device=a.device)
                np.savez_compressed(out / f"eval_it{it:05d}_{tag}.npz", **r)
                s_ = json.loads(str(r["summary"]))
                progress(out, stage=f"eval it {it} {tag}", wall_s=time.time() - te, samples=od.samples, **s_)
                log(f"it {it} eval {tag}: lenient {s_['lenient'] * 100:.0f} % strict {s_['strict'] * 100:.0f} % "
                    f"lifted {s_['lifted'] * 100:.0f} % causes {s_['causes_not_lenient']}")
            st.train()


def cmd_eval(a):
    jt = JE.JointTeacher(x(a.jt), a.device)
    st, _ = VI.load_student(a.student, a.device)
    t = load_tasks(a.tasks, np.arange(a.first, a.first + a.n) if a.n else None)
    cache = EnvCache(jt, a.device, x(a.nrd_ckpt), a.workers, x(a.mj_python), a.horizon)
    dr = VI.DRConfig.from_dict(st.cfg.dr)
    views = VI.DRViews(a.render_worlds, st.cfg.res, st.cfg.cameras, dr=dr, seed=a.seed)
    r = evaluate(st, jt, t, a.world, cache, views, a.randomise, a.jaw_lead, a.record_states, a.full_horizon, a.device)
    r["student"] = np.array(str(a.student))
    if a.out:
        np.savez_compressed(a.out, **r)
    log(f"student {a.world}{' randomised' if a.randomise else ''}: {str(r['summary'])}")


def cmd_collect(a):
    jt = JE.JointTeacher(x(a.jt), a.device)
    t = load_tasks(a.tasks, np.arange(a.first, a.first + a.n) if a.n else None)
    n = len(t["seeds"])
    cache = EnvCache(jt, a.device, x(a.nrd_ckpt), a.workers, x(a.mj_python), a.horizon)
    rng = np.random.default_rng(a.seed)
    student = None
    src = np.zeros(n, np.int64)
    if a.student:
        st, _ = VI.load_student(a.student, a.device)
        views = VI.DRViews(a.render_worlds, st.cfg.res, st.cfg.cameras, dr=VI.DRConfig.from_dict(st.cfg.dr), seed=a.seed)
        student = VI.Actor(st, views, randomise=a.randomise, device=a.device)
        src = np.where(rng.random(n) < a.teacher_share, 0, 1)
    noise = np.where((src == 0) & (rng.random(n) < a.noise_frac), a.noise, 0.0)
    d = collect(t, a.world, jt, cache, src, noise, a.seed, a.batch, student)
    np.savez_compressed(a.out, **d)
    log(f"-> {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--device", default="cuda")
        q.add_argument("--nrd-ckpt", default=NRD)
        q.add_argument("--jt", required=True)
        q.add_argument("--mj-python", default=MJ_PY)
        q.add_argument("--horizon", type=int, default=400)
        q.add_argument("--seed", type=int, default=0)
        q.add_argument("--render-worlds", type=int, default=256)

    q = sub.add_parser("run")
    common(q)
    q.add_argument("--out", required=True)
    q.add_argument("--tasks", required=True)
    q.add_argument("--eval-tasks", required=True)
    q.add_argument("--eval-n", type=int, default=0)
    q.add_argument("--guard-tasks", nargs="*", default=["~/so101_policy/runs/viral/tasks/fresh100.npz"])
    q.add_argument("--config", required=True)
    q.add_argument("--set", nargs="*")
    q.add_argument("--world", default="nrd")
    q.add_argument("--r0-tasks", type=int, default=2048)
    q.add_argument("--rounds", type=int, default=4)
    q.add_argument("--round-tasks", type=int, default=1024)
    q.add_argument("--teacher-share", type=float, default=0.5)
    q.add_argument("--r0-noise-frac", type=float, default=0.5)
    q.add_argument("--noise", type=float, default=1.0)
    q.add_argument("--collect-randomise", action="store_true")
    q.add_argument("--batch", type=int, default=1024)
    q.add_argument("--train-steps0", type=int, default=8000)
    q.add_argument("--train-steps", type=int, default=4000)
    q.add_argument("--init-from", default="")
    q.add_argument("--extra-data", nargs="*")
    q.add_argument("--eval-worlds", nargs="+", default=["mujoco"])
    q.add_argument("--eval-randomise", action="store_true")
    q.add_argument("--eval-every", type=int, default=1)
    q.add_argument("--eval-workers", type=int, default=8)
    q.set_defaults(fn=cmd_run)

    q = sub.add_parser("online")
    common(q)
    q.add_argument("--out", required=True)
    q.add_argument("--init", required=True)
    q.add_argument("--tasks", required=True)
    q.add_argument("--eval-tasks", required=True)
    q.add_argument("--eval-n", type=int, default=0)
    q.add_argument("--guard-tasks", nargs="*", default=["~/so101_policy/runs/viral/tasks/fresh100.npz"])
    q.add_argument("--config", default="")
    q.add_argument("--set", nargs="*")
    q.add_argument("--online-config", default="")
    q.add_argument("--oset", nargs="*")
    q.add_argument("--eval-randomise", action="store_true")
    q.add_argument("--eval-workers", type=int, default=8)
    q.set_defaults(fn=cmd_online)

    q = sub.add_parser("eval")
    common(q)
    q.add_argument("--student", required=True)
    q.add_argument("--tasks", required=True)
    q.add_argument("--world", default="mujoco")
    q.add_argument("--n", type=int, default=0)
    q.add_argument("--first", type=int, default=0)
    q.add_argument("--workers", type=int, default=8)
    q.add_argument("--randomise", action="store_true")
    q.add_argument("--jaw-lead", type=int, default=None)
    q.add_argument("--record-states", action="store_true")
    q.add_argument("--full-horizon", action="store_true")
    q.add_argument("--out", default="")
    q.set_defaults(fn=cmd_eval)

    q = sub.add_parser("collect")
    common(q)
    q.add_argument("--tasks", required=True)
    q.add_argument("--world", default="nrd")
    q.add_argument("--n", type=int, default=0)
    q.add_argument("--first", type=int, default=0)
    q.add_argument("--workers", type=int, default=8)
    q.add_argument("--student", default="")
    q.add_argument("--teacher-share", type=float, default=0.5)
    q.add_argument("--noise-frac", type=float, default=0.5)
    q.add_argument("--noise", type=float, default=1.0)
    q.add_argument("--randomise", action="store_true")
    q.add_argument("--batch", type=int, default=1024)
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_collect)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
