"""Closed-loop evaluation of a camera student (``vision.rollout``) on a fixed task set in one world.

    source ~/so101_policy/runs/render/madrona_luffy/env_luffy.sh
    PYTHONPATH=src ~/miniconda3/envs/so101_vision/bin/python -P scripts/so101_policy/vision_eval.py \
        --student RUN/last.pt --tasks ~/so101_policy/data/expert_v2/tasks_gate2.npz --n 256 --world mujoco \
        --workers 4 --mj-python ~/miniconda3/envs/nrdsim/bin/python --out eval_mj.npz
    ... --world nrd --ckpt ~/so101_policy/ckpt/nrd_final_s61.pt --out eval_nrd.npz
    PYTHONPATH=src python -P scripts/so101_policy/eval_policy.py report --runs eval_*.npz --ref eval_mj.npz --out r.json

Development task set: the first 256 rows of gate2 (seeds 400000-400255, family expert_base: collector starts, goals
6-30 cm). Never the test seeds 900000+ (refused).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy.vision import dataset as V
from nedm.so101_policy.vision import demos as D
from nedm.so101_policy.vision import rollout as R
from nedm.so101_policy.vision import student as S

ROOT = Path(__file__).resolve().parents[2]


def eval_policy_module():
    spec = importlib.util.spec_from_file_location("eval_policy", ROOT / "scripts" / "so101_policy" / "eval_policy.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SUM_KEYS = ("wall_s", "world_steps", "render_s", "model_s", "model_calls")


def merge_runs(parts):
    """Concatenate ``rollout.evaluate`` outputs of consecutive task chunks: per-episode arrays along axis 0; per-step
    records [B, T, ...] padded along T with their last step (episodes had ended by then); scalars: summed for
    ``SUM_KEYS``, else the first chunk's."""
    if len(parts) == 1:
        return parts[0]
    out = {}
    for k, v0 in parts[0].items():
        vs = [p[k] for p in parts]
        if np.ndim(v0) == 0:
            out[k] = np.asarray(sum(np.asarray(v) for v in vs)) if k in SUM_KEYS else v0
            continue
        if np.ndim(v0) >= 2 and k not in ("close_x", "first_fire"):
            T = max(v.shape[1] for v in vs)
            vs = [np.concatenate([v, np.repeat(v[:, -1:], T - v.shape[1], 1)], 1) if v.shape[1] < T else v for v in vs]
        out[k] = np.concatenate(vs, 0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", nargs="+", required=True, help="checkpoint(s); one output per checkpoint")
    ap.add_argument("--tasks", required=True), ap.add_argument("--out", nargs="+", required=True)
    ap.add_argument("--world", choices=("mujoco", "nrd", "kin"), required=True)
    ap.add_argument("--ckpt", help="NRD checkpoint (world nrd)"), ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=0), ap.add_argument("--mj-python")
    ap.add_argument("--n", type=int, default=256), ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=400), ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ensemble", default="mode", choices=("mode", "act", "none")), ap.add_argument("--chunk-exec", type=int, default=0)
    ap.add_argument("--ensemble-trigger", action="store_true", help="ensemble the trigger too (plain ACT)")
    ap.add_argument("--no-record", action="store_true"), ap.add_argument("--backend", default="madrona")
    ap.add_argument("--chunk", type=int, default=0, help="tasks per env and renderer (0 = all; GPU memory)")
    ap.add_argument("--trigger-thr", type=float, nargs=2, metavar=("GRASP", "PLACE"),
                    help="execution thresholds of the predicted trigger a[4] (GRASP side: mode free and the pre-close "
                         "GRASP phases; PLACE side: mode hold); default: the decoder's own > 0")
    a = ap.parse_args()
    if len(a.out) != len(a.student):
        raise SystemExit("one --out per --student")
    torch.set_num_threads(a.threads)
    E = eval_policy_module()
    views = None
    for sp, op in zip(a.student, a.out):
        st, ck = S.load_student(sp, a.device)
        chunk = a.chunk or a.n
        if views is None:
            views = V.Views(chunk, st.cfg.res, st.cfg.cameras, backend=a.backend)
        parts = []
        for c0 in range(0, a.n, chunk):
            tc = D.load_tasks(a.tasks, np.arange(a.first + c0, a.first + min(a.n, c0 + chunk)))
            parts.append(R.evaluate(st, tc, a.world, views, a.ckpt, a.device, a.workers, a.mj_python, a.horizon,
                                    not a.no_record, ensemble=a.ensemble, chunk_exec=a.chunk_exec, policy_path=sp,
                                    trigger_latest=not a.ensemble_trigger, trigger_thr=a.trigger_thr,
                                    log=lambda s: print(s, flush=True)))
        r = merge_runs(parts)
        r["student_step"] = np.int64(ck.get("step", -1))
        r["student_samples"] = np.int64(ck.get("samples", -1))
        if "mj_reason" not in r:            # --no-record: no per-step records (the collector end reasons are among them)
            r["mj_reason"] = np.zeros(len(r["end"]), np.int64)     # 0 = unknown (summary only)
        np.savez_compressed(op, **r)
        sm = E.summary(r)
        print(json.dumps(dict(student=sp, step=int(ck.get("step", -1)), out=op, success15=sm["success15"],
                              ci=sm["success15_ci"], success10=sm["success10"], lifted=sm["lifted"], causes=sm["causes"],
                              wall_s=sm["wall_s"], render_s=float(r["render_s"]), model_s=float(r["model_s"]))), flush=True)


if __name__ == "__main__":
    main()
