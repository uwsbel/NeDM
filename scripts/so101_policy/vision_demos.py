"""Expert demonstrations for the camera student (states only; ``nedm.so101_policy.vision.demos``).

    PYTHONPATH=src python -P scripts/so101_policy/vision_demos.py tasks --n 2048 --out tasks_demo.npz      # seeds 500000+
    PYTHONPATH=src python -P scripts/so101_policy/vision_demos.py collect --tasks tasks_demo.npz --world mujoco \
        --workers 4 --mj-python ~/miniconda3/envs/nrdsim/bin/python --out demos_mujoco.npz [--n 1200] [--first 0]
    PYTHONPATH=src python -P scripts/so101_policy/vision_demos.py collect --tasks tasks_demo.npz --world nrd \
        --ckpt ~/so101_policy/ckpt/nrd_final_s61.pt --device cuda --out demos_nrd_s61.npz
    PYTHONPATH=src python -P scripts/so101_policy/vision_demos.py noisy --tasks tasks_demo.npz --first 1280 --n 1280 \
        --world mujoco --workers 4 --mj-python ... --sigma 0.3 --out dart_mujoco.npz       # DART demos (env nedm works)
    PYTHONPATH=src python -P scripts/so101_policy/vision_demos.py summary demos_*.npz
    # DAgger (env so101_vision + env_luffy.sh): the student drives, the FSM expert labels (vision.rollout.collect_dagger)
    PYTHONPATH=src python -P scripts/so101_policy/vision_demos.py dagger --student RUN/last.pt --tasks tasks_demo.npz \
        --first 1280 --n 512 --world nrd --ckpt ~/so101_policy/ckpt/nrd_final_s61.pt [--beta 0.0] --out dagger_r1.npz

Decoder: ``--decoder-cfg`` JSON (DecoderV2Config overrides; default ``demos.DECODER_CFG`` = the CARRY macro, option b2),
stored in the file's meta and read again by training and evaluation.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from nedm.so101_policy.vision import demos as D


def cmd_tasks(a):
    t0 = time.time()
    t = D.make_tasks(a.n, start0=a.start0, master=a.master, first=a.first)
    np.savez(a.out, **t)
    d = np.linalg.norm(t["goal_xy"] - t["state_cube_pos"][:, :2], axis=1)
    print(f"{a.n} demo tasks (seeds {t['seeds'][0]}-{t['seeds'][-1]}) -> {a.out} in {time.time() - t0:.0f} s; goal "
          f"distance p50 {np.median(d) * 1e3:.0f} mm ({d.min() * 1e3:.0f}-{d.max() * 1e3:.0f})", flush=True)


def cmd_collect(a):
    torch.set_num_threads(a.threads)
    idx = np.arange(a.first, a.first + a.n) if a.n else None
    t = D.load_tasks(a.tasks, idx)
    dec = json.loads(a.decoder_cfg) if a.decoder_cfg else None
    ex = json.loads(a.expert_cfg) if a.expert_cfg else None
    out = D.collect(t, a.world, a.ckpt, a.device, a.workers, a.mj_python, dec, ex, a.horizon, a.batch)
    out["tasks_file"] = np.array(str(a.tasks))
    np.savez_compressed(a.out, **out)
    s = D.summary(D.load(a.out))
    print(json.dumps(s, indent=1), flush=True)


def cmd_dagger(a):
    from nedm.so101_policy.vision import dataset as V
    from nedm.so101_policy.vision import rollout as R
    from nedm.so101_policy.vision import student as S
    torch.set_num_threads(a.threads)
    t = D.load_tasks(a.tasks, np.arange(a.first, a.first + a.n))
    st, _ = S.load_student(a.student, a.device)
    views = V.Views(min(a.batch, len(t["seeds"])), st.cfg.res, st.cfg.cameras)
    out = R.collect_dagger(st, t, a.world, views, a.ckpt, a.device, a.workers, a.mj_python, a.horizon, a.beta,
                           json.loads(a.expert_cfg) if a.expert_cfg else None, a.seed, batch=a.batch,
                           harness=not a.no_harness)
    out["tasks_file"] = np.array(str(a.tasks))
    out["student"] = np.array(str(a.student))
    np.savez_compressed(a.out, **out)
    print(json.dumps(D.summary(D.load(a.out)), indent=1), flush=True)


def cmd_noisy(a):
    from nedm.so101_policy.vision import rollout as R
    torch.set_num_threads(a.threads)
    t = D.load_tasks(a.tasks, np.arange(a.first, a.first + a.n))
    out = R.collect_noisy_expert(t, a.world, a.ckpt, a.device, a.workers, a.mj_python, a.horizon, a.sigma,
                                 json.loads(a.expert_cfg) if a.expert_cfg else None, a.seed, batch=a.batch)
    out["tasks_file"] = np.array(str(a.tasks))
    np.savez_compressed(a.out, **out)
    print(json.dumps(D.summary(D.load(a.out)), indent=1), flush=True)


def cmd_summary(a):
    rows = {p: D.summary(D.load(p)) for p in a.files}
    print(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tasks")
    p.add_argument("--n", type=int, required=True), p.add_argument("--out", required=True)
    p.add_argument("--start0", type=int, default=D.DEMO_START0), p.add_argument("--master", type=int, default=D.DEMO_MASTER)
    p.add_argument("--first", type=int, default=0)
    p = sub.add_parser("collect")
    p.add_argument("--tasks", required=True), p.add_argument("--out", required=True)
    p.add_argument("--world", choices=("mujoco", "nrd", "kin"), required=True)
    p.add_argument("--ckpt"), p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=0), p.add_argument("--mj-python")
    p.add_argument("--n", type=int, default=0), p.add_argument("--first", type=int, default=0)
    p.add_argument("--batch", type=int, default=0, help="worlds per env (0 = all tasks at once)")
    p.add_argument("--horizon", type=int, default=400), p.add_argument("--threads", type=int, default=4)
    p.add_argument("--decoder-cfg"), p.add_argument("--expert-cfg")
    p = sub.add_parser("dagger")
    p.add_argument("--student", required=True), p.add_argument("--tasks", required=True), p.add_argument("--out", required=True)
    p.add_argument("--world", choices=("mujoco", "nrd", "kin"), required=True)
    p.add_argument("--ckpt"), p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=0), p.add_argument("--mj-python")
    p.add_argument("--n", type=int, required=True), p.add_argument("--first", type=int, default=0)
    p.add_argument("--beta", type=float, default=0.0), p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=256, help="worlds per env and renderer (GPU memory)")
    p.add_argument("--no-harness", action="store_true", help="the expert's own GRASP trigger only (rollout.ShadowExpert)")
    p.add_argument("--horizon", type=int, default=400), p.add_argument("--threads", type=int, default=4)
    p.add_argument("--expert-cfg")
    p = sub.add_parser("noisy", help="DART demos: the expert drives with approach noise, clean labels")
    p.add_argument("--tasks", required=True), p.add_argument("--out", required=True)
    p.add_argument("--world", choices=("mujoco", "nrd", "kin"), required=True)
    p.add_argument("--ckpt"), p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=0), p.add_argument("--mj-python")
    p.add_argument("--n", type=int, required=True), p.add_argument("--first", type=int, default=0)
    p.add_argument("--sigma", type=float, default=0.3), p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=0), p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--threads", type=int, default=4), p.add_argument("--expert-cfg")
    p = sub.add_parser("summary")
    p.add_argument("files", nargs="+")
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
