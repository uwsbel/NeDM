"""Teacher gate of the studentlab goal-jump arms (2026-10-22): the joint teacher (jt_h110) alone drives fixed tasks in
the NRD (or CPU MuJoCo) with goal jumps (``joint_env.GoalJumpConfig``) and without, same tasks, same hold cap. Reports
lenient / strict success, failure causes, the release xy error (cube centre to the CURRENT goal at the last release
after the lift: the first step the cube is no longer held) and the end xy error, for all episodes and for the
episodes with >= 1 jump.

    export PYTHONPATH=src PY=~/miniconda3/envs/so101_vision/bin/python
    $PY -P scripts/so101_policy/studentlab_teacher_gate.py --jt JT.pt --tasks train_mix45_16k.npz --n 1024 \
        --world nrd --hold-cap-s 6 --jump '{"p": 1.0}' --out gate_jump.json
    (--jump '' = no jumps; --world mujoco --workers 8 for CPU MuJoCo)
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
from nedm.so101_policy import task as T
from nedm.so101_policy.vision import viral as VI


def x(p):
    return str(Path(p).expanduser()) if p else p


class ReleaseLog:
    """After every 20 ms step (``viral.step_tap``): per world the cube xy error to the current goal at the last
    release after the firm lift (held -> not held), and that step."""

    def __init__(self, env):
        self.env = env
        B, dev = env.B, env.dev
        self.prev = torch.zeros(B, dtype=torch.bool, device=dev)
        self.err = torch.full((B,), float("nan"), dtype=torch.float64, device=dev)
        self.t = torch.full((B,), -1, dtype=torch.long, device=dev)
        self.n = torch.zeros(B, dtype=torch.long, device=dev)

    def __call__(self, ws):
        task = self.env.task
        F, tr = task.F, task.tr
        held = F["held"]
        rel = self.prev & ~held & tr.lifted & (tr.end == 0)
        if bool(rel.any()):
            e = (F["c"][:, :2] - task.goal_xy).norm(dim=-1)
            self.err = torch.where(rel, e, self.err)
            self.t = torch.where(rel, tr.t, self.t)
            self.n += rel.long()
        self.prev = held.clone()


def summarize(r, rel_err, rel_n, mask=None):
    m = np.ones(len(r["end"]), bool) if mask is None else mask
    if not m.any():
        return dict(n=0)
    le, s = np.asarray(r["lenient"]).astype(bool)[m], np.asarray(r["success"]).astype(bool)[m]
    causes = {T.ENDS[int(k)]: int(v) for k, v in zip(*np.unique(np.asarray(r["end"])[m][~le], return_counts=True))}
    ends = {T.ENDS[int(k)]: int(v) for k, v in zip(*np.unique(np.asarray(r["end"])[m], return_counts=True))}
    re = rel_err[m] * 1e3
    ee = np.asarray(r["end_xy"], np.float64)[m] * 1e3
    q = lambda v: [round(float(a), 2) for a in np.percentile(v, [50, 90])] if len(v) else None     # noqa: E731
    out = dict(n=int(m.sum()), lenient=round(float(le.mean()), 4), strict=round(float(s.mean()), 4),
               lifted=round(float(np.asarray(r["lifted"])[m].mean()), 4), causes_not_lenient=causes, ends=ends,
               released=int(np.isfinite(re).sum()), release_xy_mm_p50_p90=q(re[np.isfinite(re)]),
               release_xy_mm_p50_p90_lenient=q(re[np.isfinite(re) & le]),
               end_xy_mm_p50_p90=q(ee[np.isfinite(ee)]), releases_per_ep_mean=round(float(rel_n[m].mean()), 3),
               len_s_p50=round(float(np.median(np.asarray(r["length"])[m])) * 0.1, 2))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jt", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--world", default="nrd")
    p.add_argument("--nrd-ckpt", default="~/so101_policy/ckpt/nrd_c1_s62.pt")
    p.add_argument("--mj-python", default="~/miniconda3/envs/nrdsim/bin/python")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--hold-cap-s", type=float, default=None)
    p.add_argument("--jump", default="", help="GoalJumpConfig JSON ('' = no jumps)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    from nedm.so101_policy.vision.demos import load_tasks
    tasks = load_tasks(x(a.tasks), np.arange(a.first, a.first + a.n))
    jt = JE.JointTeacher(x(a.jt), a.device)
    gj = JE.goal_jump_config(a.jump) if a.jump else None
    parts, rel_err, rel_n = [], [], []
    t0 = time.time()
    env = None
    for s0 in range(0, a.n, a.batch):
        idx = np.arange(s0, min(a.n, s0 + a.batch))
        sub = {k: (v[idx] if np.ndim(v) and len(v) == a.n else v) for k, v in tasks.items()}
        if env is None or env.B != len(idx):
            env = JT.make_env(a.world, len(idx), jt.codec, a.device, x(a.nrd_ckpt), a.workers, x(a.mj_python),
                              a.horizon, hold_cap_s=a.hold_cap_s, dyn=False, goal_jump=gj)
            env.cfg.seed = a.seed
        rl = ReleaseLog(env)
        with VI.step_tap(env, rl):
            r = JT.run_tasks(env, sub, jt, None, np.zeros(len(idx), np.int64), None, a.seed)
        parts.append(r)
        rel_err.append(rl.err.cpu().numpy())
        rel_n.append(rl.n.cpu().numpy())
    r = {k: np.concatenate([q[k] for q in parts]) for k in parts[0] if np.ndim(parts[0][k])}
    rel_err, rel_n = np.concatenate(rel_err), np.concatenate(rel_n)
    out = dict(args=vars(a), argv=sys.argv, wall_s=round(time.time() - t0, 1), tcfg_hold_cap_s=env.tcfg.hold_cap_s,
               goal_jump=JE.asdict(gj) if gj is not None else None, all=summarize(r, rel_err, rel_n))
    if gj is not None:
        jm = np.asarray(r["goal_jumps"]) > 0
        d = np.asarray(r["goal_jump_d"], np.float64)
        out["jumped"] = summarize(r, rel_err, rel_n, jm)
        out["not_jumped"] = summarize(r, rel_err, rel_n, ~jm)
        out["jumped_past"] = summarize(r, rel_err, rel_n, jm & (d > 0))
        out["jumped_before"] = summarize(r, rel_err, rel_n, jm & (d < 0))
        out["jumps_per_ep"] = {int(k): int(v) for k, v in zip(*np.unique(r["goal_jumps"], return_counts=True))}
        out["last_jump_d_mm_p10_50_90"] = [round(float(v), 1) for v in np.percentile(d[jm] * 1e3, [10, 50, 90])] \
            if jm.any() else None
    print(json.dumps(out, indent=1, default=str), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=str))
        np.savez_compressed(Path(a.out).with_suffix(".npz"), release_xy=rel_err, releases=rel_n,
                            **{k: r[k] for k in ("end", "lenient", "success", "end_xy", "length", "lifted", "seeds")},
                            **({k: r[k] for k in ("goal_jumps", "goal_jump_d", "goal_jump_lat", "goal_jump_z", "goal_xy_final",
                                                  "goal_xy")} if gj is not None else {}))


if __name__ == "__main__":
    main(sys.argv[1:])
