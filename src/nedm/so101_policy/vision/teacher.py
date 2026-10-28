"""PPO teacher + DAgger data for the camera student (build5_vision.md; round 2 of the camera-policy work).

The student acts through the SAME decoder as the PPO teacher (the teacher checkpoint's resolved DecoderV2Config:
``grasp_gate``, ``grasp_latch``, ``abort_action``, 7-dim pre-tanh action; ``configs/so101_policy/ppo/env_v2.json``), so
a GRASP trigger outside the gate box is ignored at no cost. The teacher is a ``ppo.Policy`` (deterministic pre-tanh
mean, the action space of the student) and is queried at ANY state from its own privileged observation (the env's
79-dim ``task.observation``); DAgger runs in the arm's own world (``nrd``, ``mujoco`` = CPU MuJoCo, ``kin``).

Per decision (0.1 s), BEFORE it is applied, ``Recorder`` stores:

| key | shape | meaning |
|---|---|---|
| obs | [D] | the env observation of the teacher (``task.observation``: privileged and robot-side entries) |
| q, cube_pos, cube_quat | [6], [3], [4] | render inputs (the world's state; TRUE cube orientation) |
| label | [7] | the teacher's deterministic pre-tanh action at this state |
| executed | [7] | the pre-tanh action actually sent (teacher, teacher + noise, or student) |
| src | [] int8 | who acted: 0 teacher, 1 teacher + noise, 2 student |
| mode | [] int8 | decoder mode at the decision start (``decoder.MODES``) |

Per episode: task seed, goal, outcome (``vec_env`` episode keys). Only the live decisions are stored (``demos.pack``
rule: the decision in which the task ended is the last one).

Student inputs (written down here): the cameras (``dataset.Views``) and the ROBOT-SIDE entries of the observation
(``ROBOT_KEYS``: joint angles and rates, TCP and gripper axes by forward kinematics, goal x, y, sin / cos 4 psi, the
decoder's macro phase and its 24-dim state). Not given: the cube pose / velocity / tilt, the cube in the grasp frame,
goal - cube, yaw error, ``lifted`` and ``hold_time`` (derived from the cube state).

Loss mask (``read_mask``; where the decoder reads a column at the decision start): mode free a[0:6] (a[5] = lift height
at a GRASP trigger), pre-close GRASP (g_brake, g_descend, g_settle) a[6] (the abort; latch on), mode hold a[0:5]
(PLACE trigger a[4]); the other macro phases read nothing.

Seeds (written down here): DAgger / demo task seeds 510,000+ (``DEMO_START0``; the vision demo range 500,000-599,999,
not the round-1 files 500,000-502,559); never the gate 1000-2023, gate2 400,000+ (development / selection rows) or the
test 900,000+ (``check_seeds``).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy.vision import demos as D

DEMO_START0 = 510_000
ROBOT_KEYS = ("q", "qd", "tcp", "grip_x", "grip_z", "goal_xy", "goal_yaw4", "macro", "decoder")
SRC = {"teacher": 0, "noisy": 1, "student": 2}
REC_KEYS = ("obs", "q", "cube_pos", "cube_quat", "label", "executed", "src", "mode")


def check_seeds(seeds):
    """Refuse training tasks with validation / test seeds (review_code_v3 N7: the R9 guard for the vision package)."""
    from nedm.so101_policy.vec_env import reserved_seed_hits
    h = reserved_seed_hits(seeds)
    if h:
        raise ValueError(f"{h} reserved (validation / test) seeds in a training task set")


def robot_index(layout: dict) -> torch.Tensor:
    """Indices of the robot-side observation entries (``ROBOT_KEYS``) in a ``vec_env.obs_layout``."""
    return torch.cat([torch.arange(layout[k].start, layout[k].stop) for k in ROBOT_KEYS])


def read_mask(mode, act_dim=7) -> torch.Tensor:
    """[N] decoder modes -> [N, act_dim] float mask of the columns the decoder reads (module docstring)."""
    from nedm.so101_policy.decoder import M
    mode = torch.as_tensor(mode).long()
    m = torch.zeros(mode.shape[0], act_dim)
    free = mode == M["free"]
    pre = (mode == M["g_brake"]) | (mode == M["g_descend"]) | (mode == M["g_settle"])
    hold = mode == M["hold"]
    m[free, :6] = 1.0
    m[hold, :5] = 1.0
    if act_dim > 6:
        m[pre, 6] = 1.0
    return m


def hash_normal(seed, counter, n, stream0=0):
    """Counter-based standard normals [B, n] of int64 seeds [B] (``gate_expert.hash_normal``)."""
    from nedm.so101_policy.decoder import hash_uniform
    cnt = torch.full_like(seed, int(counter))
    out = []
    for j in range(n):
        u1 = hash_uniform(seed, cnt, stream0 + 2 * j + 1)
        u2 = hash_uniform(seed, cnt, stream0 + 2 * j + 2)
        out.append(torch.sqrt(-2.0 * torch.log1p(-u1)) * torch.cos(2 * math.pi * u2))
    return torch.stack(out, -1)


class Teacher:
    """A PPO checkpoint (``ppo.load_policy(with_env=True)``) as the label source: ``act(obs)`` -> deterministic pre-tanh
    mean [B, A] (float32, on the obs device). ``decoder_cfg`` / ``obs_rates``: the training env's records (review R1)."""

    def __init__(self, path, device="cuda"):
        from nedm.so101_policy.ppo import load_policy, sha256
        self.path = str(Path(path).expanduser())
        self.pol, rec = load_policy(self.path, device, with_env=True)
        if rec is None:
            raise ValueError(f"{path}: no env record (needs a checkpoint written after 2026-10-11 23:00)")
        self.rec = rec
        self.dev = torch.device(device)
        self.decoder_cfg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in rec["decoder"].items()}
        self.obs_rates = bool(rec["task"]["obs_rates"])
        self.act_dim = int(self.pol.kwargs["act_dim"])
        self.obs_dim = int(self.pol.kwargs["obs_dim"])
        self.std = self.pol.std.detach().float()
        self.sha256 = sha256(self.path)
        self.files = {k: rec["env"][k] for k in ("geom_path", "kin_points")
                      if rec["env"].get(k) and Path(rec["env"][k]).expanduser().exists()}

    @torch.no_grad()
    def act(self, obs):
        return self.pol.act_inference(obs.to(self.dev, torch.float32)).float().to(obs.device)

    def info(self):
        return dict(path=self.path, sha256=self.sha256, decoder_cfg={k: v for k, v in self.decoder_cfg.items()},
                    obs_rates=self.obs_rates, act_dim=self.act_dim)


def make_env(world, B, teacher: Teacher, device="cuda", nrd_ckpt=None, workers=0, mj_python=None, horizon=400,
             members="split", record=False):
    """The evaluation env of ``eval_policy.run_policy`` for the teacher's decoder and observation: auto reset off,
    ``TaskConfig.evaluation()`` (15 mm / 15 deg success), overlap rules off in CPU MuJoCo only, 8 s horizon."""
    from nedm.so101_policy.vec_env import EnvConfig, PickPlaceVecEnv
    ck = [str(Path(c).expanduser()) for c in ([nrd_ckpt] if isinstance(nrd_ckpt, (str, Path)) else (nrd_ckpt or []))]
    cfg = EnvConfig(world=world, num_envs=B, device=device, ckpts=ck, members=members, auto_reset=False,
                    eval_rules=True, task=dict(horizon=int(horizon), obs_rates=teacher.obs_rates), mj_workers=workers,
                    mj_python=mj_python, starts="", nrd_cuda_graph=True, decoder=dict(teacher.decoder_cfg),
                    **teacher.files)
    env = PickPlaceVecEnv(cfg, record=record)
    if env.num_obs != teacher.obs_dim or env.num_actions != teacher.act_dim:
        raise RuntimeError(f"teacher {teacher.obs_dim} obs / {teacher.act_dim} actions vs env {env.num_obs} / "
                           f"{env.num_actions}")
    return env


class MixAct:
    """act_fn of ``vec_env.run_episodes``: label = the teacher's deterministic pre-tanh action at the current state;
    executed = the teacher's (with probability ``beta`` per world and decision, counter-based hash of the task seed,
    the decision index and ``seed``) or the student's (``student``: an object with ``reset(env)`` and
    ``__call__(obs, env) -> pre-tanh [B, A]``). ``noise`` > 0: the teacher's executed action gets Gaussian noise of
    noise x the teacher's own std on the move columns a[0:4] (DART-style; counter-based normals; not on the triggers,
    the lift height or the abort). Pre-tanh actions out (``squashed`` False)."""

    def __init__(self, teacher: Teacher, student=None, beta=1.0, noise=0.0, seed=0):
        if student is None and beta < 1.0:
            raise ValueError("beta < 1 needs a student")
        self.t, self.s, self.beta, self.noise, self.seed = teacher, student, float(beta), float(noise), int(seed)
        self.k, self.seeds = 0, None
        self.last_label = self.last_src = None

    def reset(self, env):
        if self.s is not None:
            self.s.reset(env)
        self.k = 0
        self.seeds = env.task_seed.clone().long() * 1_000_003 + 7919 * self.seed + 17

    @torch.no_grad()
    def __call__(self, obs, env):
        lab = self.t.act(obs.to(env.dev)).to(env.dev, torch.float64)
        a_t = lab
        src_t = torch.full((env.B,), SRC["teacher"], dtype=torch.int8, device=env.dev)
        if self.noise > 0:
            eps = hash_normal(self.seeds.to(env.dev), self.k, 4).to(lab)
            a_t = lab.clone()
            a_t[:, :4] = lab[:, :4] + self.noise * self.t.std[:4].to(lab) * eps
            src_t[:] = SRC["noisy"]
        if self.beta >= 1.0:
            a, src = a_t, src_t
        else:
            s = self.s(obs, env).to(env.dev, torch.float64)
            if self.beta <= 0.0:
                a, src = s, torch.full_like(src_t, SRC["student"])
            else:
                from nedm.so101_policy.decoder import hash_uniform
                u = hash_uniform(self.seeds.to(env.dev), torch.full_like(self.seeds, self.k).to(env.dev), 99)
                use_t = u < self.beta
                a = torch.where(use_t[:, None], a_t, s)
                src = torch.where(use_t, src_t, torch.full_like(src_t, SRC["student"]))
        self.k += 1
        self.last_label, self.last_src = lab, src
        return a, False


class Recorder:
    """Wraps a ``MixAct``: stores the state, the teacher's label and the executed action at every decision (module
    docstring)."""

    def __init__(self, act: MixAct):
        self.act = act
        self.rec = {k: [] for k in REC_KEYS}
        self.live = []

    def reset(self, env):
        self.act.reset(env)
        self.rec = {k: [] for k in REC_KEYS}
        self.live = []

    def __call__(self, obs, env):
        ws = env.ws
        mode = env.dec.mode.to(torch.int8).cpu()
        a, sq = self.act(obs, env)
        r = self.rec
        r["obs"].append(obs.float().cpu())
        r["q"].append(ws.q.float().cpu())
        r["cube_pos"].append(ws.cube_pos.float().cpu())
        r["cube_quat"].append(ws.cube_quat.float().cpu())
        r["label"].append(self.act.last_label.float().cpu())
        r["executed"].append(a.float().cpu())
        r["src"].append(self.act.last_src.cpu())
        r["mode"].append(mode)
        self.live.append((env.task.tr.end == 0).cpu())
        return a, sq

    def arrays(self):
        out = {k: torch.stack(v, 1).numpy() for k, v in self.rec.items()}
        out["live"] = torch.stack(self.live, 1).numpy()
        return out


def pack(per_dec, ep, tasks):
    """Live decisions of every episode -> flat arrays (``demos.pack`` layout with this module's keys)."""
    B = len(ep["end"])
    length = np.asarray(ep["length"], np.int64)
    live = per_dec["live"]
    for b in range(B):
        if not live[b, : length[b]].all():
            raise RuntimeError(f"episode {b}: live mask is not a prefix of length {length[b]}")
    rows_b = np.repeat(np.arange(B), length)
    rows_t = np.concatenate([np.arange(n) for n in length]) if B else np.zeros(0, np.int64)
    out = {k: per_dec[k][rows_b, rows_t] for k in REC_KEYS}
    out["ep_index"], out["t"] = rows_b.astype(np.int32), rows_t.astype(np.int16)
    out["ep_start"] = np.concatenate([[0], np.cumsum(length)[:-1]]).astype(np.int64)
    for k in D.EP_KEYS:
        out[f"ep_{k}"] = np.asarray(ep[k])
    out["ep_seed"] = np.asarray(tasks["seeds"], np.int64)
    out["ep_goal_xy"] = np.asarray(tasks["goal_xy"], np.float32)
    out["ep_goal_yaw"] = np.asarray(tasks["goal_yaw"], np.float32)
    return out


def _subset(tasks, idx):
    n = len(tasks["seeds"])
    return {k: (v[idx] if np.ndim(v) and len(v) == n else v) for k, v in tasks.items()}


@torch.no_grad()
def collect(tasks, world, teacher: Teacher, student=None, beta=1.0, noise=0.0, seed=0, device="cuda", nrd_ckpt=None,
            workers=0, mj_python=None, horizon=400, batch=256, members="split", log=print, meta_extra=None):
    """DAgger / teacher episodes on all tasks in one world -> packed dict (``pack``) + ``meta`` (json string)."""
    from nedm.so101_policy.vec_env import run_episodes
    check_seeds(tasks["seeds"])
    n = len(tasks["seeds"])
    batch = int(batch) or n
    t0, steps, parts = time.time(), 0, []
    t_student = 0.0
    for s0 in range(0, n, batch):
        idx = np.arange(s0, min(n, s0 + batch))
        t = _subset(tasks, idx)
        env = make_env(world, len(idx), teacher, device, nrd_ckpt, workers, mj_python, horizon, members)
        dev = env.dev
        act = MixAct(teacher, student, beta, noise, seed)
        rec = Recorder(act)
        g = lambda k: torch.as_tensor(t[k], dtype=torch.float64, device=dev)            # noqa: E731
        mj_start = dict(q=t["start_q"], cube_pos=t["start_cube_pos"], cube_quat=t["start_cube_quat"])
        ep = run_episodes(env, rec, D.start_state(t, dev), g("goal_xy"), g("goal_yaw"),
                          torch.as_tensor(t["seeds"], dtype=torch.long, device=dev),
                          mj_start=mj_start if world == "mujoco" else None, reset_hook=rec.reset, log=None)
        steps += env.world_steps
        env.close()
        parts.append(pack(rec.arrays(), ep, t))
        if student is not None and hasattr(student, "t_render"):
            t_student = student.t_render + student.t_model
        log(f"  collect {world} beta {beta} noise {noise}: tasks {s0}-{s0 + len(idx) - 1}, success "
            f"{ep['success'].mean() * 100:.1f} %, {time.time() - t0:.0f} s")
        del env, act, rec
    out = D.concat(parts)
    meta = dict(world=world, nrd_ckpt=str(nrd_ckpt), teacher=teacher.info(), n_tasks=n, beta=float(beta),
                noise=float(noise), seed=int(seed), horizon=int(horizon), batch=batch, wall_s=time.time() - t0,
                world_steps=int(steps), student_s=float(t_student), success=float(np.mean(out["ep_success"])),
                decisions=int(len(out["t"])), seeds=[int(tasks["seeds"][0]), int(tasks["seeds"][-1])],
                code=D.source_hashes(), **(meta_extra or {}))
    out["meta"] = np.array(json.dumps(meta, default=str))
    return out


@torch.no_grad()
def evaluate(student, tasks, world, teacher: Teacher, device="cuda", nrd_ckpt=None, workers=0, mj_python=None,
             horizon=400, record=True, members="split", log=print, policy_path="", with_labels=True):
    """The student (beta 0) on all tasks in one world -> dict in the ``eval_policy.py run`` format (+ timing, the
    teacher's labels on the student's states when ``with_labels``: agreement diagnostics)."""
    from nedm.so101_policy.vec_env import run_episodes
    B = len(tasks["seeds"])
    t0 = time.time()
    env = make_env(world, B, teacher, device, nrd_ckpt, workers, mj_python, horizon, members, record)
    dev = env.dev
    act = MixAct(teacher, student, beta=0.0)
    rec = Recorder(act) if with_labels else act
    g = lambda k: torch.as_tensor(tasks[k], dtype=torch.float64, device=dev)              # noqa: E731
    mj_start = dict(q=tasks["start_q"], cube_pos=tasks["start_cube_pos"], cube_quat=tasks["start_cube_quat"])
    r = run_episodes(env, rec, D.start_state(tasks, dev), g("goal_xy"), g("goal_yaw"),
                     torch.as_tensor(tasks["seeds"], dtype=torch.long, device=dev),
                     mj_start=mj_start if world == "mujoco" else None, reset_hook=rec.reset, log=log)
    out = dict(env.records() if record else {})
    resolved = env.resolved()
    steps = env.world_steps
    env.close()
    if with_labels:
        p = pack(rec.arrays(), r, tasks)
        out.update({f"lab_{k}": p[k] for k in ("label", "executed", "mode", "ep_index", "t")})
    out.update(end=r["end"], success=r["success"], success10=r["success10"], lifted=r["lifted"], end_xy=r["end_xy"],
               end_yaw=r["end_yaw"], t_end=r["steps"], ret=r["ret"], length=r["length"], seeds=np.asarray(tasks["seeds"]),
               box_log=np.zeros(B, bool), world=np.array(world), config=np.array("vision_dagger"), decoder=np.array("v2"),
               ckpt=np.array(str(nrd_ckpt)), wall_s=np.float64(time.time() - t0), device=np.array(str(dev)),
               world_steps=np.int64(steps), phase=np.zeros_like(out["mode"]) if record else np.zeros(0),
               label=np.zeros_like(out["mode"]) if record else np.zeros(0), phase_names=np.array(["policy"]),
               train_rules=np.bool_(False), assist=np.bool_(False), stochastic=np.bool_(False), noise_seed=np.int64(0),
               policy=np.array(str(policy_path)), env_resolved=np.array(json.dumps(resolved, default=str)),
               teacher=np.array(json.dumps(teacher.info(), default=str)),
               render_s=np.float64(getattr(student, "t_render", 0.0)), model_s=np.float64(getattr(student, "t_model", 0.0)),
               model_calls=np.int64(getattr(student, "calls", 0)))
    return out


def load(path):
    z = dict(np.load(Path(path).expanduser(), allow_pickle=False))
    z["meta"] = json.loads(str(z["meta"])) if "meta" in z else {}
    return z


def summary(d):
    """Per-file summary: episodes, success, failure causes, decisions, who acted, label statistics."""
    from nedm.so101_policy.task import ENDS
    end = d["ep_end"]
    s = d["ep_success"].astype(bool)
    causes = {ENDS[int(k)]: int(v) for k, v in zip(*np.unique(end[~s], return_counts=True))}
    src = {k: int((d["src"] == v).sum()) for k, v in SRC.items()}
    return dict(episodes=int(len(end)), success=float(s.mean()), lifted=float(d["ep_lifted"].mean()), failures=causes,
                decisions=int(len(d["t"])), src=src,
                meta={k: v for k, v in d["meta"].items() if k not in ("code",)} if isinstance(d.get("meta"), dict) else {})
