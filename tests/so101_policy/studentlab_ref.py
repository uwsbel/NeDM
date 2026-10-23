"""Reference runs of the camera student's online loop and evaluation path on the CPU in the kinematic world, for the
studentlab tests (``test_studentlab.py``). Only the API that predates the studentlab change is used, so the same file
runs against the base commit's code (bit-identity check of the defaults):

    PYTHONPATH=<src> python tests/so101_policy/studentlab_ref.py --out ref.npz [--cfg '{"hist": 2}'] [--ocfg '{}']

``KinOnline``: ``viral.OnlineDAgger`` with the kinematic world (``JointEnvConfig`` patched to world "kin") and
``FakeViews`` (images = a deterministic function of the rendered state and goal, no renderer)."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy import joint_env as JE

JT_PATH = Path("~/so101_policy/runs/viral/jt/jt_h110.pt").expanduser()
TASKS = Path("~/so101_policy/runs/viral/tasks/dev100.npz").expanduser()
HAVE = JT_PATH.exists() and TASKS.exists()


class FakeViews:
    """``DRViews`` stand-in (sim-scene API): ``render`` -> uint8 [n, C, R, R, 3], a deterministic function of q, cube
    position and goal pose (and the per-world colours); records every call's goal poses (``goals``) and a call
    counter in pixel [0, 0, 0, 0] of every image (``calls``)."""

    def __init__(self, res=16, cameras=("external", "wrist"), nb=4):
        self.res, self.cameras, self.nb = res, tuple(cameras), nb
        self.real, self.fx, self.dr = None, None, None
        self.base = torch.linspace(0.2, 0.8, nb * 3).reshape(nb, 3)
        self.gen = torch.Generator().manual_seed(3)
        self.calls = 0
        self.goals = []

    def draw(self, n):
        C = len(self.cameras)
        u = lambda *s: torch.rand(n, *s, generator=self.gen)        # noqa: E731
        return dict(colors=u(self.nb, 3), cam_dpos=u(C, 3) * 0.01, cam_drot=u(C, 3) * 0.01)

    def render(self, q, cube_pos, cube_quat, goal_pose, dr=None, light=None):
        n, C, R = q.shape[0], len(self.cameras), self.res
        self.calls += 1
        self.goals.append(goal_pose.detach().clone().double())
        h = torch.arange(R, dtype=torch.float64)
        x = (q.double() * 400.0).sum(1) + cube_pos.double()[:, 0] * 900.0 + goal_pose.double()[:, 0] * 700.0 \
            + goal_pose.double()[:, 1] * 500.0                                                  # [n]
        if dr is not None:
            x = x + dr["colors"][:, 0, 0].double() * 50.0
        img = (x[:, None, None, None, None] + h[None, None, :, None, None] * 7 + h[None, None, None, :, None] * 3
               + torch.arange(C, dtype=torch.float64)[None, :, None, None, None] * 11
               + torch.arange(3, dtype=torch.float64)[None, None, None, None, :] * 5)
        img = torch.remainder(img.expand(n, C, R, R, 3), 256.0).to(torch.uint8).clone()
        img[:, :, 0, 0, 0] = self.calls % 256
        return img


def tiny_cfg(**over):
    from nedm.so101_policy.vision import viral as VI
    d = dict(res=16, keypoints=4, pool_dim=8, hidden=(32, 32), aux_hidden=16, amp=False, batch=8, cameras=["external",
                                                                                                          "wrist"],
             lag=0, lag_range=[0, 0], qd_noise=0.3)
    d.update(over)
    return VI.ViralConfig.from_dict(d)


def tiny_stats():
    from nedm.so101_policy.vision import viral as VI
    return dict(robot_mean=torch.zeros(12), robot_std=torch.ones(12), tgt_mean=torch.zeros(10, 6),
                tgt_std=torch.full((10, 6), 0.05), aux_mean=torch.zeros(VI.AUX_DIM), aux_std=torch.ones(VI.AUX_DIM))


def tiny_student(seed=0, **over):
    from nedm.so101_policy.vision import viral as VI
    torch.manual_seed(seed)
    return VI.ViralNet(tiny_cfg(**over), tiny_stats(), pretrained=False)


@contextlib.contextmanager
def kin_world():
    """``JE.JointEnvConfig`` with world "kin" (any caller, e.g. ``OnlineDAgger``); restored on exit."""
    orig = JE.JointEnvConfig

    class KinCfg(orig):
        def __post_init__(self):
            self.world = "kin"
            super().__post_init__()

    JE.JointEnvConfig = KinCfg
    try:
        yield
    finally:
        JE.JointEnvConfig = orig


def load(n=None):
    from nedm.so101_policy.vision.demos import load_tasks
    return load_tasks(str(TASKS), np.arange(n) if n else None)


def make_online(st, ocfg_over=None, seed=0, envs=16):
    """(OnlineDAgger in the kinematic world, its FakeViews)."""
    from nedm.so101_policy.vision import viral as VI
    jt = JE.JointTeacher(str(JT_PATH), "cpu")
    o = dict(envs=envs, buffer=64, steps_per_iter=2, horizon=60, lr=1e-3, lr_backbone=1e-3)
    o.update(ocfg_over or {})
    ocfg = VI.OnlineConfig.from_dict(o)
    views = FakeViews(st.cfg.res, st.cfg.cameras)
    with kin_world():
        od = VI.OnlineDAgger(st, jt, load(), "unused-nrd.pt", views, ocfg, "cpu", seed, lambda s: None)
    return od, views


def run_online(st, n_dec, ocfg_over=None, seed=0, envs=16, each=None):
    od, views = make_online(st, ocfg_over, seed, envs)
    info = []
    for it in range(n_dec):
        od.it = it
        if each is not None:
            each(od, "before")
        od.act()
        if each is not None:
            each(od, "after")
        if od.buf.n >= 16:
            info.append(od.train_steps())
    return od, views, info


def run_eval(st, n_tasks=8, jaw_lead=None):
    from nedm.so101_policy import joint_teacher as JT
    from nedm.so101_policy.vision import viral as VI
    jt = JE.JointTeacher(str(JT_PATH), "cpu")
    t = load(n_tasks)
    env = JT.make_env("kin", n_tasks, jt.codec, "cpu")
    actor = VI.Actor(st.eval(), FakeViews(st.cfg.res, st.cfg.cameras), device="cpu", jaw_lead=jaw_lead)
    if hasattr(actor, "bind"):
        actor.bind(env)
    tap = VI.step_tap(env, actor.push) if actor.needs_tap else contextlib.nullcontext()
    with tap:
        return JT.run_tasks(env, t, jt, actor, np.ones(n_tasks, np.int64), None, 0, record=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cfg", default="{}")
    p.add_argument("--ocfg", default="{}")
    p.add_argument("--n-dec", type=int, default=14)
    a = p.parse_args()
    torch.set_num_threads(1)
    st = tiny_student(0, **json.loads(a.cfg))
    od, views, info = run_online(st, a.n_dec, json.loads(a.ocfg))
    b = od.buf
    out = dict(rgb=b.rgb.numpy(), q=b.q.numpy(), robot=b.robot.numpy(), lab1=b.lab1.numpy(), lab2=b.lab2.numpy(),
               valid2=b.valid2.numpy(), aux=b.aux.numpy(), aug=b.aug.numpy(), ws_q=od.env.ws.q.numpy(),
               ws_cube=od.env.ws.cube_pos.numpy(), obs=od.env.obs.numpy(), samples=np.int64(od.samples),
               loss=np.array([v.get("loss", np.nan) for v in info], np.float64))
    for k, v in st.state_dict().items():
        out[f"w_{k}"] = v.detach().numpy()
    r = run_eval(tiny_student(1, **json.loads(a.cfg)))
    for k in ("end", "length", "lenient", "row_q", "row_lab", "row_obs"):
        out[f"eval_{k}"] = np.asarray(r[k])
    np.savez(a.out, **out)


if __name__ == "__main__":
    main()
