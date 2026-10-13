"""DAgger with a PPO teacher for the camera student (build5_vision.md): teacher rollouts (round 0), then rounds of
student rollouts (beta-mixed with the teacher) in the arm's own world, labels = the teacher's deterministic action at
every visited state, aggregate, train; evaluation in CPU MuJoCo after every round. One process holds the renderer
(Madrona), the teacher, the student and the world (stages resume from their files after a crash).

    source ~/so101_policy/runs/render/madrona_luffy/env_luffy.sh
    PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -P scripts/so101_policy/vision_dagger.py run \
        --out RUN --world mujoco|nrd [--nrd-ckpt ~/so101_policy/ckpt/nrd_r2_s62.pt] --teacher TEACHER.pt \
        --tasks ~/so101_policy/runs/vision2/tasks/tasks_dagger_8192.npz --config configs/so101_policy/vision/perstep_v1.json \
        --r0-tasks 2048 --r0-noise 0 1 --rounds 4 --round-tasks 1024 --betas 0.5 0.25 0 0 \
        --train-steps0 6000 --train-steps 3000 --eval-n 256 --workers 4 --mj-python ~/miniconda3/envs/nrdsim/bin/python

Files: RUN/data/r<i>.npz (``teacher.collect``), RUN/r<i>/{last.pt, learning_curve.csv, meta.json, train_summary.json}
(student after round i's data), RUN/r<i>/eval_<world>.npz (``teacher.evaluate``: ``eval_policy.py run`` format),
RUN/progress.jsonl (one line per finished stage: wall time, world steps, success).

Tasks: rows of the task file: round 0 = rows [0, r0_tasks), round i = the next ``round_tasks`` rows (seeds 510,000+;
refused if any seed is reserved). Evaluation: the first ``eval_n`` gate2 rows (development set; seeds 400000+).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy.vision import dataset as V
from nedm.so101_policy.vision import demos as D
from nedm.so101_policy.vision import teacher as TT

GATE2 = "~/so101_policy/data/expert_v2/tasks_gate2.npz"


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def progress(out, **kw):
    with open(Path(out) / "progress.jsonl", "a") as f:
        f.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), **kw), default=float) + "\n")


def summarize_eval(r):
    from nedm.so101_policy.task import ENDS
    s = r["success"].astype(bool)
    causes = {ENDS[int(k)]: int(v) for k, v in zip(*np.unique(r["end"][~s], return_counts=True))}
    out = dict(success=float(s.mean()), n=int(len(s)), success10=float(r["success10"].mean()),
               lifted=float(r["lifted"].mean()), causes=causes, wall_s=float(r["wall_s"]),
               world_steps=int(r["world_steps"]), render_s=float(r["render_s"]), model_s=float(r["model_s"]))
    if "lab_label" in r:                     # teacher vs student on the student's own states
        lab, ex, mode = r["lab_label"].clip(-3, 3), r["lab_executed"].clip(-3, 3), r["lab_mode"]
        m = TT.read_mask(mode, lab.shape[1]).numpy()
        out["label_mse"] = [float(((lab[:, j] - ex[:, j]) ** 2 * m[:, j]).sum() / max(1, m[:, j].sum()))
                            for j in range(lab.shape[1])]
        from nedm.so101_policy.decoder import M
        for name, code in (("free", M["free"]), ("hold", M["hold"])):
            sel = mode == code
            if sel.any():
                out[f"trig_agree_{name}"] = float(((lab[sel, 4] > 0) == (ex[sel, 4] > 0)).mean())
    return out


def build_student(kind, cfg_dict, data, views, out_dir, steps, init, seed):
    if kind == "perstep":
        from nedm.so101_policy.vision import perstep as P
        cfg = P.PerStepConfig.from_dict({**cfg_dict, "seed": seed})
        st = P.train(data, views, cfg, out_dir, max_steps=steps, init_from=init, log=log,
                     extra_meta=dict(argv=sys.argv))
        return st
    if kind == "act":
        from nedm.so101_policy.vision import act7 as A
        cfg = A.Act7Config.from_dict({**cfg_dict, "seed": seed})
        return A.train(data, views, cfg, out_dir, max_steps=steps, init_from=init, log=log,
                       extra_meta=dict(argv=sys.argv))
    raise ValueError(kind)


def load_student(kind, path, views, robot_idx, act_kw=None):
    if kind == "perstep":
        from nedm.so101_policy.vision import perstep as P
        st, _ = P.load_perstep(path)
        return P.PerStepAct(st, views, robot_idx)
    from nedm.so101_policy.vision import act7 as A
    st, _ = A.load_act7(path)
    return A.Act7Act(st, views, robot_idx, **(act_kw or {}))


def make_data(kind, paths, cfg_dict, seed):
    from nedm.so101_policy.vision import perstep as P
    chunk = int(cfg_dict.get("chunk", 1)) if kind == "act" else 1
    return P.DaggerSet(paths, "cuda", chunk=chunk, seed=seed, target_clip=float(cfg_dict.get("target_clip", 3.0)))


def cmd_run(a):
    out = Path(a.out).expanduser()
    (out / "data").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    cfg_dict = json.loads(Path(a.config).read_text()) if a.config else {}
    cfg_dict.update(json.loads(a.set) if a.set else {})
    kind = a.student
    (out / "args.json").write_text(json.dumps(vars(a), indent=1, default=str))
    teacher = TT.Teacher(a.teacher, "cuda")
    log(f"teacher {teacher.path} decoder {teacher.decoder_cfg if a.verbose else '(gate/latch/abort)'}")
    tasks_all = D.load_tasks(a.tasks)
    TT.check_seeds(tasks_all["seeds"])
    res, cams = int(cfg_dict.get("res", V.RES)), tuple(cfg_dict.get("cameras", V.CAMERAS))
    views = V.Views(a.render_worlds, res, cams)
    env_kw = dict(device="cuda", nrd_ckpt=a.nrd_ckpt, workers=a.workers, mj_python=a.mj_python, horizon=a.horizon,
                  members=a.members)
    eval_tasks = D.load_tasks(a.eval_tasks, np.arange(a.eval_n))
    act_kw = json.loads(a.act_kw) if a.act_kw else None
    n0 = a.r0_tasks
    paths = []
    prev = None
    betas = list(a.betas) + [a.betas[-1]] * max(0, a.rounds - len(a.betas))
    for i in range(a.rounds + 1):
        dpath = out / "data" / f"r{i}.npz"
        rows = np.arange(0, n0) if i == 0 else np.arange(n0 + (i - 1) * a.round_tasks, n0 + i * a.round_tasks)
        if rows[-1] >= len(tasks_all["seeds"]):
            raise SystemExit(f"task file too short for round {i}")
        if not dpath.exists():
            t = {k: (v[rows] if np.ndim(v) and len(v) == len(tasks_all["seeds"]) else v) for k, v in tasks_all.items()}
            t0 = time.time()
            if i == 0:
                parts = []
                noises = a.r0_noise
                cut = np.array_split(np.arange(len(rows)), len(noises))
                for nz, sel in zip(noises, cut):
                    tt = {k: (v[sel] if np.ndim(v) and len(v) == len(rows) else v) for k, v in t.items()}
                    parts.append(TT.collect(tt, a.world, teacher, None, 1.0, nz, a.seed, batch=a.batch, log=log,
                                            **env_kw))
                metas = [json.loads(str(p["meta"])) for p in parts]
                d = D.concat(parts)
                meta = dict(metas[0], noise=list(noises), wall_s=sum(m["wall_s"] for m in metas),
                            world_steps=sum(m["world_steps"] for m in metas), n_tasks=len(rows),
                            success=float(np.mean(d["ep_success"])), decisions=int(len(d["t"])),
                            seeds=[int(t["seeds"][0]), int(t["seeds"][-1])])
                d["meta"] = np.array(json.dumps(meta, default=str))
            else:
                sa = load_student(kind, prev, views, TT.robot_index(make_layout(teacher)), act_kw)
                d = TT.collect(t, a.world, teacher, sa, betas[i - 1], 0.0, a.seed + i, batch=a.batch, log=log,
                               meta_extra=dict(student=str(prev)), **env_kw)
                meta = json.loads(str(d["meta"]))
            np.savez_compressed(dpath, **d)
            progress(out, stage=f"collect r{i}", wall_s=time.time() - t0, world_steps=meta["world_steps"],
                     success=meta["success"], decisions=meta["decisions"], beta=1.0 if i == 0 else betas[i - 1],
                     student_s=meta.get("student_s", 0.0))
            log(f"round {i} data: {meta['decisions']} decisions, success {meta['success'] * 100:.1f} % "
                f"({time.time() - t0:.0f} s)")
        paths.append(str(dpath))
        spath = out / f"r{i}" / "last.pt"
        if not spath.exists():
            t0 = time.time()
            data = make_data(kind, paths, cfg_dict, a.seed)
            log(f"round {i} train: {len(data.train_rows)} rows ({data.n_train_eps} eps), val {len(data.val_rows)}")
            steps = a.train_steps0 if (i == 0 or not a.finetune) else a.train_steps
            build_student(kind, cfg_dict, data, views, out / f"r{i}", steps,
                          None if (i == 0 or not a.finetune) else prev, a.seed)
            del data
            torch.cuda.empty_cache()
            progress(out, stage=f"train r{i}", wall_s=time.time() - t0, steps=steps)
        prev = str(spath)
        if a.eval_every or i == a.rounds:
            for w in a.eval_worlds:
                epath = out / f"r{i}" / f"eval_{w}.npz"
                if epath.exists():
                    continue
                t0 = time.time()
                sa = load_student(kind, prev, views, TT.robot_index(make_layout(teacher)), act_kw)
                kw = dict(env_kw)
                if w == "mujoco":
                    kw["nrd_ckpt"] = None
                r = TT.evaluate(sa, eval_tasks, w, teacher, log=None, policy_path=prev, **kw)
                np.savez_compressed(epath, **r)
                sm = summarize_eval(r)
                progress(out, stage=f"eval r{i} {w}", **sm)
                log(f"round {i} eval {w}: success {sm['success'] * 100:.1f} %, lifted {sm['lifted'] * 100:.1f} %, "
                    f"causes {sm['causes']} ({time.time() - t0:.0f} s)")
                del sa
                torch.cuda.empty_cache()
    log("done")


def make_layout(teacher):
    from nedm.so101_policy import task as T
    from nedm.so101_policy.vec_env import obs_layout
    return obs_layout(T.TaskConfig(obs_rates=teacher.obs_rates))


def cmd_eval(a):
    """Evaluate one student checkpoint (``--student``) in one world."""
    teacher = TT.Teacher(a.teacher, "cuda")
    kind = a.student_kind
    ck = torch.load(Path(a.student).expanduser(), map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    views = V.Views(a.render_worlds, int(cfg.get("res", V.RES)), tuple(cfg.get("cameras", V.CAMERAS)))
    tasks = D.load_tasks(a.eval_tasks, np.arange(a.first, a.first + a.eval_n), final_test=a.final_test)
    sa = load_student(kind, a.student, views, TT.robot_index(make_layout(teacher)),
                      json.loads(a.act_kw) if a.act_kw else None)
    t0 = time.time()
    r = TT.evaluate(sa, tasks, a.world, teacher, device="cuda", nrd_ckpt=a.nrd_ckpt, workers=a.workers,
                    mj_python=a.mj_python, horizon=a.horizon, members=a.members, log=None, policy_path=a.student)
    np.savez_compressed(a.out, **r)
    sm = summarize_eval(r)
    log(f"eval {a.world} {a.student}: {json.dumps(sm)} ({time.time() - t0:.0f} s)")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def cmd_summary(a):
    """Markdown tables of runs (progress.jsonl + train summaries): per round the data, the training and the
    evaluations."""
    for run in a.runs:
        run = Path(run).expanduser()
        rows = [json.loads(x) for x in (run / "progress.jsonl").read_text().splitlines() if x.strip()]
        print(f"\n{run}\n")
        print("| round | data: episodes success / decisions / wall s / world steps / student s | train: steps / min | "
              "eval world: success (Wilson 95 %) / 10-10 / lifted / failures |")
        print("|---|---|---|---|")
        rounds = sorted({int(r["stage"].split()[1][1:]) for r in rows})
        for i in rounds:
            col = [r for r in rows if r["stage"] == f"collect r{i}"]
            data = (f"{col[-1]['success'] * 100:.1f} % / {col[-1]['decisions']} / {col[-1]['wall_s']:.0f} / "
                    f"{col[-1]['world_steps'] / 1e6:.2f} M / {col[-1].get('student_s', 0):.0f} (beta {col[-1]['beta']})"
                    if col else "shared")
            ts = run / f"r{i}" / "train_summary.json"
            tr = json.loads(ts.read_text()) if ts.exists() else None
            train = f"{tr['steps']} / {tr['wall_min']:.1f}" if tr else "-"
            ev = []
            for r in rows:
                if r["stage"].startswith(f"eval r{i} "):
                    k = int(round(r["success"] * r["n"]))
                    lo, hi = wilson(k, r["n"])
                    fails = ", ".join(f"{c} {v}" for c, v in sorted(r["causes"].items(), key=lambda kv: -kv[1]))
                    ev.append(f"{r['stage'].split()[2]}: **{r['success'] * 100:.1f} %** ({lo * 100:.1f}-{hi * 100:.1f}) / "
                              f"{r['success10'] * 100:.1f} % / {r['lifted'] * 100:.1f} % / {fails}")
            print(f"| r{i} | {data} | {train} | {'; '.join(ev) or '-'} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--out", required=True), p.add_argument("--world", choices=("mujoco", "nrd", "kin"), required=True)
    p.add_argument("--teacher", required=True), p.add_argument("--nrd-ckpt"), p.add_argument("--members", default="split")
    p.add_argument("--tasks", required=True), p.add_argument("--student", default="perstep", choices=("perstep", "act"))
    p.add_argument("--config"), p.add_argument("--set")
    p.add_argument("--r0-tasks", type=int, default=2048), p.add_argument("--r0-noise", type=float, nargs="+", default=[0.0, 1.0])
    p.add_argument("--rounds", type=int, default=4), p.add_argument("--round-tasks", type=int, default=1024)
    p.add_argument("--betas", type=float, nargs="+", default=[0.5, 0.25, 0.0, 0.0])
    p.add_argument("--train-steps0", type=int, default=6000), p.add_argument("--train-steps", type=int, default=3000)
    p.add_argument("--finetune", type=int, default=1, help="1: round i fine-tunes round i-1's student; 0: from scratch")
    p.add_argument("--eval-n", type=int, default=256), p.add_argument("--eval-tasks", default=GATE2)
    p.add_argument("--eval-worlds", nargs="+", default=["mujoco"]), p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--workers", type=int, default=4), p.add_argument("--mj-python")
    p.add_argument("--batch", type=int, default=256, help="worlds per env in the collection")
    p.add_argument("--render-worlds", type=int, default=256), p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--seed", type=int, default=0), p.add_argument("--threads", type=int, default=4)
    p.add_argument("--act-kw", help="JSON kwargs of the ACT actor (ensemble, trigger_latest)")
    p.add_argument("--verbose", action="store_true")
    p = sub.add_parser("summary")
    p.add_argument("runs", nargs="+")
    p = sub.add_parser("eval")
    p.add_argument("--student", required=True), p.add_argument("--student-kind", default="perstep", choices=("perstep", "act"))
    p.add_argument("--teacher", required=True), p.add_argument("--world", choices=("mujoco", "nrd", "kin"), required=True)
    p.add_argument("--nrd-ckpt"), p.add_argument("--members", default="split"), p.add_argument("--out", required=True)
    p.add_argument("--eval-n", type=int, default=256), p.add_argument("--first", type=int, default=0)
    p.add_argument("--eval-tasks", default=GATE2)
    p.add_argument("--workers", type=int, default=4), p.add_argument("--mj-python")
    p.add_argument("--render-worlds", type=int, default=256), p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--act-kw")
    p.add_argument("--final-test", action="store_true", help="allow the test seeds 900000+ (final test only)")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
