"""Tray student (build8_tray_student.md; ``nedm.so101_policy.vision.tray``): camera student with ONLY top + wrist images
and q / qdot as inputs and absolute joint targets as outputs; the goal is a tray drawn on the table.

    source ~/so101_policy/runs/render/madrona_luffy/env_luffy.sh; export PYTHONPATH=src
    PY=~/miniconda3/envs/so101_vision/bin/python
    $PY -P scripts/so101_policy/vision_tray.py tasks --stage A --n 4096 --out tasks_A_train.npz          # training seeds
    $PY -P scripts/so101_policy/vision_tray.py tasks --stage A --n 100 --dev --out tasks_A_dev100.npz     # development
    $PY -P scripts/so101_policy/vision_tray.py teacher --tasks tasks_A_dev100.npz --world mujoco --workers 8 \
        --mj-python ~/miniconda3/envs/nrdsim/bin/python --out teacher_A_mj.npz
    $PY -P scripts/so101_policy/vision_tray.py run --out RUN --world nrd --nrd-ckpt ~/so101_policy/ckpt/nrd_c1_s62.pt \
        --tasks tasks_A_train.npz --eval-tasks tasks_A_dev100.npz --config configs/so101_policy/vision/tray_perstep_v1.json \
        --mode dagger --r0-tasks 1024 --rounds 4 --round-tasks 512 --betas 0.5 0.25 0 0 --eval-worlds mujoco nrd
    $PY -P scripts/so101_policy/vision_tray.py eval --student RUN/r4/last.pt --tasks tasks_A_dev100.npz --world mujoco \
        --out e.npz [--record-states]
    $PY -P scripts/so101_policy/vision_tray.py summary RUN [RUN ...]
    $PY -P scripts/so101_policy/vision_tray.py frames --tasks tasks_A_dev100.npz --data RUN/data/r0.npz --out frames.png

``run``: round 0 = teacher episodes (``--r0-noise-frac`` of them with DART noise ``--noise`` x the teacher's std on the
move columns; labels = the clean teacher's joint targets); rounds 1.. = ``--mode dagger``: the student drives (beta =
probability of a teacher decision), labels by the re-synced shadow decoder; ``--mode bc``: more teacher episodes with
the same noise mix (the plain BC + noise arm). After each round: train (round 0 from scratch or ``--init-from``, later
rounds fine-tune on the aggregate), evaluate on the development tasks in every ``--eval-worlds`` world. Stages resume
from their files. ``--extra-data``: earlier stages' data files added to the aggregate (curriculum).
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
from nedm.so101_policy.vision import teacher as TT
from nedm.so101_policy.vision import tray as TR

TEACHER = "~/so101_policy/runs/c1ppo/ckpt/nC1_s62_b_model_00171.pt"
NRD = "~/so101_policy/ckpt/nrd_c1_s62.pt"
MJ_PY = "~/miniconda3/envs/nrdsim/bin/python"


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def progress(out, **kw):
    with open(Path(out) / "progress.jsonl", "a") as f:
        f.write(json.dumps(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), **kw), default=float) + "\n")


def x(p):
    return str(Path(p).expanduser()) if p else p


def cmd_tasks(a):
    t = TR.make_tasks(a.stage, a.n, a.first, a.dev)
    np.savez(a.out, **t)
    d = np.linalg.norm(t["goal_xy"] - t["state_cube_pos"][:, :2], axis=1)
    log(f"-> {a.out}; goal distance p50 {np.median(d) * 1e3:.0f} mm ({d.min() * 1e3:.0f}-{d.max() * 1e3:.0f})")


def cmd_teacher(a):
    te = TT.Teacher(x(a.teacher))
    t = TR.load_tasks(a.tasks, np.arange(a.first, a.first + a.n) if a.n else None)
    r = TR.evaluate(None, t, a.world, te, a.device, x(a.nrd_ckpt), a.workers, x(a.mj_python), a.horizon,
                    record_states=a.record_states, log=log, decoder=dec_over(a))
    np.savez_compressed(a.out, **r)
    log(f"teacher {a.world}: {r['summary']}")


def dec_over(a):
    """Teacher-side decoder overrides (JSON; default: ``tray.TEACHER_DECODER``)."""
    return None if not getattr(a, "decoder_over", None) else json.loads(a.decoder_over)


def make_views(n, cfg):
    return V.Views(n, cfg.res, cfg.cameras, goal="tray")


def cmd_eval(a):
    te = TT.Teacher(x(a.teacher))
    st, _ = TR.load_student(a.student, a.device)
    t = TR.load_tasks(a.tasks, np.arange(a.first, a.first + a.n) if a.n else None)
    views = make_views(a.render_worlds, st.cfg)
    actor = TR.TrayActor(st, views, a.device)
    r = TR.evaluate(actor, t, a.world, te, a.device, x(a.nrd_ckpt), a.workers, x(a.mj_python), a.horizon,
                    a.n_exec or st.cfg.n_exec, a.record_states, log, decoder=dec_over(a),
                    ensemble=a.ensemble if a.ensemble is not None else st.cfg.ensemble,
                    jaw_lead=a.jaw_lead if a.jaw_lead is not None else st.cfg.jaw_lead)
    r["student"] = np.array(str(a.student))
    np.savez_compressed(a.out, **r)
    log(f"student {a.world}: {r['summary']}")


def cmd_run(a):
    out = Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    args_path = out / "args.json"
    if not args_path.exists():
        args_path.write_text(json.dumps(vars(a), indent=1, default=str))
    te = TT.Teacher(x(a.teacher))
    cfg_d = json.loads(Path(a.config).read_text())
    for s in a.set or []:
        k, v = s.split("=", 1)
        cfg_d[k] = json.loads(v)
    cfg_d["seed"] = a.seed
    cfg = TR.TrayConfig.from_dict(cfg_d)
    tasks = TR.load_tasks(a.tasks)
    TT.check_seeds(tasks["seeds"])
    ev = TR.load_tasks(a.eval_tasks, np.arange(a.eval_n) if a.eval_n else None)
    class LazyViews:                    # the renderer (Madrona, ~9 GB of GPU memory) only when images are needed
        def __init__(self):
            self.v = None

        def __getattr__(self, k):
            if self.v is None:
                self.v = make_views(a.render_worlds, cfg)
            return getattr(self.v, k)

    views = LazyViews()
    cache = {}
    nrd = x(a.nrd_ckpt)
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
        t = TR.subset(tasks, rows[i])
        n = len(rows[i])
        if not f.exists():
            t0 = time.time()
            if i == 0 or a.mode == "bc":
                nz = np.zeros(n)
                nz[int(round(n * (1 - a.r0_noise_frac))):] = a.noise
                d = TR.collect(t, a.world, te, None, 1.0, nz, a.seed + 1000 * i, a.device, nrd, a.workers,
                               x(a.mj_python), a.horizon, a.batch, log, dict(round=i, mode=a.mode), tw_cache=cache,
                               decoder=dec_over(a))
            else:
                st, _ = TR.load_student(prev, a.device)
                actor = TR.TrayActor(st, views, a.device)
                beta = a.betas[i - 1]
                d = TR.collect(t, a.world, te, actor, beta, 0.0, a.seed + 1000 * i, a.device, nrd, a.workers,
                               x(a.mj_python), a.horizon, a.student_batch, log, dict(round=i, mode=a.mode,
                                                                                     student=str(prev)),
                               n_exec=cfg.n_exec, tw_cache=cache, decoder=dec_over(a), ensemble=cfg.ensemble,
                               jaw_lead=cfg.jaw_lead)
                meta = json.loads(str(d["meta"]))
                meta.update(student_render_s=actor.t_render, student_model_s=actor.t_model)
                d["meta"] = np.array(json.dumps(meta, default=str))
                del st, actor
            np.savez_compressed(f, **d)
            log(f"data r{i}: {json.dumps(TR.summarize(d))}")
            del d
            torch.cuda.empty_cache()
            m = TR.load(f)["meta"]
            progress(out, stage=f"data r{i}", wall_s=time.time() - t0, success=m["success"], rows=m["rows"],
                     world_steps=m["world_steps"], beta=m["beta"], noise=m["noise"],
                     student_s=m.get("student_render_s", 0.0) + m.get("student_model_s", 0.0))
        files.append(str(f))
        if a.data_only:                 # dagger: round 0 only (later rounds need a student); bc: every round
            log(f"data only: r{i} written")
            if a.mode != "bc":
                break
            continue
        ck = out / f"r{i}" / "last.pt"
        if not ck.exists():
            t0 = time.time()
            data = TR.TraySet(files, a.device, cfg.chunk, seed=a.seed, step_rows=cfg.step_rows)
            steps = a.train_steps0 if i == 0 else a.train_steps
            init = prev if (i > 0 or a.init_from) and a.finetune else None
            if i > 0 and not a.finetune:
                steps = a.train_steps0
            TR.train(data, views, cfg, out / f"r{i}", max_steps=steps, init_from=init, log=log, device=a.device,
                     extra_meta=dict(argv=sys.argv, files=files))
            progress(out, stage=f"train r{i}", wall_s=time.time() - t0, steps=steps, rows=data.n_rows,
                     init_from=str(init))
            del data
            torch.cuda.empty_cache()
        prev = str(ck)
        if i % a.eval_every == 0 or i == a.rounds:
            st, _ = TR.load_student(ck, a.device)
            for w in list(a.eval_worlds) + (list(a.eval_worlds_final or []) if i == a.rounds else []):
                fe = out / f"r{i}" / f"eval_{w}.npz"
                if fe.exists():
                    continue
                t0 = time.time()
                actor = TR.TrayActor(st, views, a.device)
                key = (w, len(ev["seeds"]))
                if key not in cache:
                    cache[key] = TR.TrayWorld(w, len(ev["seeds"]), te, a.device, nrd, a.eval_workers, x(a.mj_python),
                                              a.horizon, decoder=dec_over(a))
                r = TR.evaluate(actor, ev, w, te, a.device, nrd, a.eval_workers, x(a.mj_python), a.horizon, cfg.n_exec,
                                record_states=True, tw=cache[key], ensemble=cfg.ensemble, jaw_lead=cfg.jaw_lead)
                r["student"] = np.array(str(ck))
                np.savez_compressed(fe, **r)
                s = json.loads(str(r["summary"]))
                progress(out, stage=f"eval r{i} {w}", wall_s=time.time() - t0, **s)
                log(f"eval r{i} {w}: {json.dumps(s)}")
            del st
            torch.cuda.empty_cache()
    for tw in cache.values():
        tw.close()
    log("run done")


def cmd_summary(a):
    from nedm.so101_policy.task import ENDS   # noqa: F401
    for run in a.runs:
        p = Path(run)
        print(f"== {run}")
        for line in (p / "progress.jsonl").read_text().splitlines() if (p / "progress.jsonl").exists() else []:
            d = json.loads(line)
            if d["stage"].startswith("eval"):
                print(f"  {d['stage']}: success {d['success'] * 100:.1f} % (CI {d['ci95'][0] * 100:.1f}-"
                      f"{d['ci95'][1] * 100:.1f}), 10/10 {d['success10'] * 100:.1f} %, lifted {d['lifted'] * 100:.1f} %, "
                      f"causes {d['causes']}, {d['wall_s']:.0f} s")
            elif d["stage"].startswith("data"):
                print(f"  {d['stage']}: success {d['success'] * 100:.1f} %, rows {d['rows']}, beta {d['beta']}, "
                      f"noise {d['noise']:.2f}, {d['wall_s']:.0f} s, world steps {d['world_steps'] / 1e6:.2f} M")
            else:
                print(f"  {d['stage']}: {d['steps']} steps, {d['rows']} rows, {d['wall_s'] / 60:.1f} min")


def cmd_frames(a):
    """Top + wrist frames of a few states (rows of a data file, or the start states of a task file) -> one PNG."""
    import imageio.v2 as imageio
    cfg = TR.TrayConfig()
    views = make_views(16, cfg)
    if a.data:
        d = TR.load(a.data)
        ep = np.arange(a.episodes)
        rows = []
        for e in ep:
            r0, n = int(d["ep_row_start"][e]), int(d["ep_rows"][e])
            rows += [r0 + int(round(f * (n - 1))) for f in np.linspace(0, 1, a.per_episode)]
        rows = np.array(rows)
        q, cp, cq = d["row_q"][rows], d["row_cube_pos"][rows], d["row_cube_quat"][rows]
        e = d["row_ep"][rows]
        goal = np.concatenate([d["goal_xy"][e], d["goal_yaw"][e][:, None]], 1)
    else:
        t = TR.load_tasks(a.tasks)
        k = np.arange(min(a.episodes, len(t["seeds"])))
        q, cp, cq = t["state_q"][k], t["state_cube_pos"][k], t["state_cube_quat"][k]
        goal = np.concatenate([t["goal_xy"][k], t["goal_yaw"][k][:, None]], 1)
    f = lambda v: torch.as_tensor(np.asarray(v), dtype=torch.float32, device="cuda")      # noqa: E731
    imgs = []
    for s in range(0, len(q), 16):
        imgs.append(views.render(f(q[s:s + 16]), f(cp[s:s + 16]), f(cq[s:s + 16]), f(goal[s:s + 16])).cpu().numpy())
    rgb = np.concatenate(imgs)
    per = a.per_episode if a.data else 1
    tiles = [np.concatenate(list(rgb[i]), 1) for i in range(len(rgb))]       # top | wrist
    rows_img = [np.concatenate(tiles[r:r + per], 1) for r in range(0, len(tiles), per)]
    imageio.imwrite(a.out, np.concatenate(rows_img, 0))
    log(f"-> {a.out} ({len(rgb)} states)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tasks")
    p.add_argument("--stage", choices=tuple(TR.STAGES), required=True), p.add_argument("--n", type=int, required=True)
    p.add_argument("--first", type=int, default=0), p.add_argument("--dev", action="store_true")
    p.add_argument("--out", required=True)

    def common(p):
        p.add_argument("--teacher", default=TEACHER), p.add_argument("--nrd-ckpt", default=NRD)
        p.add_argument("--device", default="cuda"), p.add_argument("--workers", type=int, default=8)
        p.add_argument("--mj-python", default=MJ_PY), p.add_argument("--horizon", type=int, default=400)
        p.add_argument("--decoder-over", help="teacher decoder overrides (JSON); default tray.TEACHER_DECODER")

    p = sub.add_parser("teacher")
    common(p)
    p.add_argument("--tasks", required=True), p.add_argument("--world", choices=("mujoco", "nrd", "kin"), required=True)
    p.add_argument("--n", type=int, default=0), p.add_argument("--first", type=int, default=0)
    p.add_argument("--out", required=True), p.add_argument("--record-states", action="store_true")
    p = sub.add_parser("eval")
    common(p)
    p.add_argument("--student", required=True), p.add_argument("--tasks", required=True)
    p.add_argument("--world", choices=("mujoco", "nrd", "kin"), required=True)
    p.add_argument("--n", type=int, default=0), p.add_argument("--first", type=int, default=0)
    p.add_argument("--out", required=True), p.add_argument("--record-states", action="store_true")
    p.add_argument("--render-worlds", type=int, default=256)
    p.add_argument("--n-exec", type=int, default=0, help="query period in steps (0: the student's config)")
    p.add_argument("--ensemble", type=float, help="temporal ensembling coefficient (with --n-exec 1)")
    p.add_argument("--jaw-lead", type=int, help="jaw target this many steps ahead in the chunk")
    p = sub.add_parser("run")
    common(p)
    p.add_argument("--out", required=True), p.add_argument("--world", choices=("mujoco", "nrd", "kin"), required=True)
    p.add_argument("--tasks", required=True), p.add_argument("--eval-tasks", required=True)
    p.add_argument("--eval-n", type=int, default=0)
    p.add_argument("--config", required=True), p.add_argument("--set", action="append")
    p.add_argument("--mode", choices=("dagger", "bc"), default="dagger")
    p.add_argument("--r0-tasks", type=int, default=1024), p.add_argument("--r0-noise-frac", type=float, default=0.5)
    p.add_argument("--noise", type=float, default=1.0)
    p.add_argument("--rounds", type=int, default=4), p.add_argument("--round-tasks", type=int, default=512)
    p.add_argument("--betas", type=float, nargs="+", default=[0.5, 0.25, 0.0, 0.0])
    p.add_argument("--train-steps0", type=int, default=6000), p.add_argument("--train-steps", type=int, default=3000)
    p.add_argument("--finetune", type=int, default=1)
    p.add_argument("--init-from"), p.add_argument("--extra-data", nargs="*")
    p.add_argument("--eval-worlds", nargs="+", default=["mujoco"]), p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--eval-worlds-final", nargs="*", help="worlds evaluated only after the last round")
    p.add_argument("--eval-workers", type=int, default=8)
    p.add_argument("--batch", type=int, default=1024), p.add_argument("--student-batch", type=int, default=512)
    p.add_argument("--render-worlds", type=int, default=256), p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-only", action="store_true", help="stop after the round-0 data (no renderer, no training)")
    p = sub.add_parser("summary")
    p.add_argument("runs", nargs="+")
    p = sub.add_parser("frames")
    p.add_argument("--tasks"), p.add_argument("--data"), p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=4), p.add_argument("--per-episode", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(4)
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
