"""Expert demonstrations for the camera student, stored as STATES (the images are drawn on the fly in training).

The privileged FSM expert (``expert.PickPlaceExpert``) acts closed loop through ``DecoderV2`` in one world (``nrd`` =
``GraspNrdSim`` with one checkpoint, ``mujoco`` = CPU ``MujocoWorlds``, ``kin`` = ``KinWorld``), with the episode loop
of ``vec_env.run_episodes`` (the same loop as the gate and ``eval_policy.py``: evaluation rules, 8 s horizon, decoder
macro seeds = task seeds). Once per decision (0.1 s), BEFORE the decision is applied, the recorder stores:

| key | shape | meaning |
|---|---|---|
| q, qd | [6] | measured joint angles (rad) and 20 ms step-average rates (the world's state) |
| cube_pos, cube_quat | [3], [4] | cube centre (m) and TRUE orientation w x y z (what the renderer draws) |
| dec_obs | [24] | ``DecoderV2.observation()`` (target TCP, yaw sin / cos, target velocity, yaw rate, jaw command, armed, mode) |
| action | [6] | the expert's decoder action in [-1, 1] (``decoder`` module docstring) |
| mode | [] | decoder mode code (``decoder.MODES``) |

Per episode: the task (seed, goal x, y, yaw), the outcome of ``task.PickPlaceTask`` (end code, success 15 / 15 and
10 / 10, lifted, end errors, decisions, 20 ms steps). Decisions are stored for the live part of each episode only
(``length`` decisions: the decision in which the task ended is the last one). Failures are kept (``success`` False)
for the analysis; the dataset uses the successful episodes.

Task seeds (written down here): start seeds ``DEMO_START0`` + i (500,000-599,999 reserved for the vision demos; not
the gate 1000-2023, gate2 400,000+, campaign 300,000+, PPO 200,000+, test 900,000+ or a collector seed:
``mujoco_env.sample_starts`` refuses those); tasks from ``data_round.make_tasks`` with family expert_base only (start
spread 40 deg in 70 %, 60 deg in 30 %, goals ``task.sample_goals`` 6-30 cm, yaw +-30 deg), master seed
``DEMO_MASTER``: the same start and goal distribution as the gate2 expert_base rows that the evaluation uses
(``tasks_gate2.npz`` rows 0-306). The expert runs with ``ExpertConfig`` defaults (one speed, lift 5 cm); the task
options of the campaign (speed class, pause, lift) are not used.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

DEMO_START0 = 500_000
DEMO_MASTER = 20261301
DECODER_CFG = {"carry_macro": True}      # decoder option b2 (CARRY macro), build3_fixes.md: the default of this package
REC_KEYS = ("q", "qd", "cube_pos", "cube_quat", "dec_obs", "action", "mode")   # + "executed" for DAgger records
EP_KEYS = ("end", "success", "success10", "lifted", "end_xy", "end_yaw", "length", "steps")
REPO_ROOT = Path(__file__).resolve().parents[4]


def make_tasks(n, start0=DEMO_START0, master=DEMO_MASTER, first=0, log=print):
    """n demo tasks (module docstring) as a gate_expert.py task table (start, settled state, goal, seeds)."""
    from nedm.so101_policy import data_round as R
    if start0 + first + n > 600_000 or start0 < 500_000:
        raise ValueError("vision demo start seeds are 500,000-599,999")
    idx = np.arange(first, first + n)
    counts = np.zeros(len(R.FAMILIES), np.int64)
    counts[0] = first + n                                     # family expert_base only
    t = R.make_tasks(first + n, set_name="campaign", master=master, start0=start0, counts=counts, idx=idx, log=log)
    t["demo_master"] = np.int64(master)
    return t


def load_tasks(path, idx=None, final_test=False):
    """A task file (rows idx). Test seeds (900000+) are refused unless final_test (the final test, one evaluation per
    final policy)."""
    t = dict(np.load(Path(path).expanduser()))
    n = len(t["seeds"])
    if idx is not None:
        t = {k: (v[idx] if np.ndim(v) and len(v) == n else v) for k, v in t.items()}
    if (np.asarray(t["seeds"]) >= 900000).any() and not final_test:
        raise ValueError("test seeds (900000+) in a demo / development task set")
    return t


def start_state(t, device):
    from nedm.so101_policy.world import WorldState
    f = lambda k: torch.as_tensor(t[f"state_{k}"], dtype=torch.float64, device=device)    # noqa: E731
    n = len(t["seeds"])
    z = lambda *s: torch.zeros(*s, dtype=torch.bool, device=device)                      # noqa: E731
    return WorldState(q=f("q"), qd=f("qd"), cube_pos=f("cube_pos"), cube_quat=f("cube_quat"), cube_v=f("cube_v"),
                      cube_w=f("cube_w"), contact=z(n, 3), arm_table=z(n), invalid=z(n))


def make_env(world, B, device="cuda", ckpt=None, workers=0, mj_python=None, decoder_cfg=None, horizon=400,
             members="split", record=False):
    """The evaluation env of ``eval_policy.run_policy`` (auto reset off, ``TaskConfig.evaluation()``, overlap rules
    off in CPU MuJoCo only) with an explicit decoder config (review R1: the decoder options travel with the data)."""
    from nedm.so101_policy.vec_env import EnvConfig, PickPlaceVecEnv
    ck = [str(Path(c).expanduser()) for c in ([ckpt] if isinstance(ckpt, (str, Path)) else (ckpt or []))]
    cfg = EnvConfig(world=world, num_envs=B, device=device, ckpts=ck, members=members, auto_reset=False,
                    eval_rules=True, task=dict(horizon=int(horizon)), mj_workers=workers, mj_python=mj_python,
                    starts="", nrd_cuda_graph=True, decoder=dict(DECODER_CFG if decoder_cfg is None else decoder_cfg))
    return PickPlaceVecEnv(cfg, record=record)


class ExpertAct:
    """The FSM expert as an act_fn of ``vec_env.run_episodes`` (as ``eval_policy.ExpertAct``)."""

    def __init__(self, cfg=None):
        self.cfg, self.ex = dict(cfg or {}), None

    def reset(self, env):
        from nedm.so101_policy.expert import ExpertConfig, PickPlaceExpert
        self.ex = PickPlaceExpert(env.dec, replace(ExpertConfig(), **self.cfg))
        self.ex.reset(torch.arange(env.B, device=env.dev), env.ws, env.task.goal_xy.clone(), env.task.goal_yaw.clone())

    def __call__(self, obs, env):
        return self.ex.act(env.ws), True


class DecisionRecorder:
    """Wraps an act_fn (obs, env) -> (action, squashed): stores the state at every decision (module docstring) and the
    action actually sent (squashed actions are clamped to [-1, 1]; pre-tanh ones are stored after tanh). An act_fn with
    a ``last_label`` attribute (``rollout.DaggerAct``): ``action`` = that label, ``executed`` = the action sent."""

    def __init__(self, act):
        self.act = act
        self.keys = REC_KEYS + (("executed",) if hasattr(act, "last_label") else ())
        self.rec = {k: [] for k in self.keys}
        self.live = []

    def reset(self, env):
        if hasattr(self.act, "reset"):
            self.act.reset(env)
        self.rec = {k: [] for k in self.keys}
        self.live = []

    def __call__(self, obs, env):
        ws = env.ws
        dec_obs = env.dec.observation()
        a, sq = self.act(obs, env)
        a_sent = a.clamp(-1.0, 1.0) if sq else torch.tanh(a)
        r = self.rec
        r["q"].append(ws.q.float().cpu())
        r["qd"].append(ws.qd.float().cpu())
        r["cube_pos"].append(ws.cube_pos.float().cpu())
        r["cube_quat"].append(ws.cube_quat.float().cpu())
        r["dec_obs"].append(dec_obs.float().cpu())
        if "executed" in r:
            r["action"].append(self.act.last_label.float().cpu())
            r["executed"].append(a_sent.float().cpu())
        else:
            r["action"].append(a_sent.float().cpu())
        r["mode"].append(env.dec.mode.to(torch.int8).cpu())
        self.live.append((env.task.tr.end == 0).cpu())
        return a, sq

    def arrays(self):
        """-> dict of [B, K, ...] numpy arrays (K = decisions taken by the loop) and live [B, K]."""
        out = {k: torch.stack(v, 1).numpy() for k, v in self.rec.items()}
        out["live"] = torch.stack(self.live, 1).numpy()
        return out


def pack(per_dec, ep, tasks):
    """Flatten the live decisions of every episode: -> dict with per-decision arrays [N, ...], per-episode arrays [E],
    ``ep_start`` [E] (row of the first decision) and ``ep_index`` [N] / ``t`` [N] (episode, decision index)."""
    B = len(ep["end"])
    length = np.asarray(ep["length"], np.int64)
    live = per_dec["live"]
    K = live.shape[1]
    if (length > K).any():
        raise RuntimeError("episode longer than the recorded decisions")
    # the live part is the prefix of each episode
    for b in range(B):
        if not live[b, : length[b]].all():
            raise RuntimeError(f"episode {b}: live mask is not a prefix of length {length[b]}")
    rows_b = np.repeat(np.arange(B), length)
    rows_t = np.concatenate([np.arange(n) for n in length]) if B else np.zeros(0, np.int64)
    out = {k: per_dec[k][rows_b, rows_t] for k in REC_KEYS + ("executed",) if k in per_dec}
    out["ep_index"], out["t"] = rows_b.astype(np.int32), rows_t.astype(np.int16)
    out["ep_start"] = np.concatenate([[0], np.cumsum(length)[:-1]]).astype(np.int64)
    for k in EP_KEYS:
        out[f"ep_{k}"] = np.asarray(ep[k])
    out["ep_seed"] = np.asarray(tasks["seeds"], np.int64)
    out["ep_goal_xy"] = np.asarray(tasks["goal_xy"], np.float32)
    out["ep_goal_yaw"] = np.asarray(tasks["goal_yaw"], np.float32)
    return out


def source_hashes():
    files = ["src/nedm/so101_policy/" + f for f in ("decoder.py", "expert.py", "task.py", "vec_env.py", "kin.py",
                                                     "nrd_sim.py", "mujoco_env.py", "kin_world.py", "vision/demos.py")]
    return {f: hashlib.sha256((REPO_ROOT / f).read_bytes()).hexdigest()[:16] for f in files if (REPO_ROOT / f).exists()}


@torch.no_grad()
def collect(tasks, world, ckpt=None, device="cuda", workers=0, mj_python=None, decoder_cfg=None, expert_cfg=None,
            horizon=400, batch=0, log=print):
    """Expert episodes on all tasks in one world -> packed dict (``pack``) + meta. batch: worlds per env (0 = all)."""
    from nedm.so101_policy.vec_env import run_episodes
    n = len(tasks["seeds"])
    batch = int(batch) or n
    code = source_hashes()                   # the code at the start of the run
    parts, t0, steps = [], time.time(), 0
    for s in range(0, n, batch):
        idx = np.arange(s, min(n, s + batch))
        t = {k: (v[idx] if np.ndim(v) and len(v) == n else v) for k, v in tasks.items()}
        B = len(idx)
        env = make_env(world, B, device, ckpt, workers, mj_python, decoder_cfg, horizon)
        dev = env.dev
        rec = DecisionRecorder(ExpertAct(expert_cfg))
        g = lambda k: torch.as_tensor(t[k], dtype=torch.float64, device=dev)            # noqa: E731
        mj_start = dict(q=t["start_q"], cube_pos=t["start_cube_pos"], cube_quat=t["start_cube_quat"])
        ep = run_episodes(env, rec, start_state(t, dev), g("goal_xy"), g("goal_yaw"),
                          torch.as_tensor(t["seeds"], dtype=torch.long, device=dev),
                          mj_start=mj_start if world == "mujoco" else None, reset_hook=rec.reset, log=None)
        steps += env.world_steps
        env.close()
        parts.append(pack(rec.arrays(), ep, t))
        log(f"  {world}: tasks {s}-{s + B - 1} done, success {ep['success'].mean() * 100:.1f} %, "
            f"{time.time() - t0:.0f} s")
    out = concat(parts)
    meta = dict(world=world, ckpt=str(ckpt), n_tasks=n, wall_s=time.time() - t0, world_steps=int(steps),
                decoder_cfg=dict(DECODER_CFG if decoder_cfg is None else decoder_cfg), expert_cfg=dict(expert_cfg or {}),
                horizon=int(horizon), device=str(device), workers=int(workers), code=code,
                success=float(np.mean(out["ep_success"])), decisions=int(len(out["t"])))
    out["meta"] = np.array(json.dumps(meta))
    return out


def concat(parts):
    """Concatenate packed demo dicts (episode indices and starts shifted)."""
    if len(parts) == 1:
        return parts[0]
    out, e0, r0 = {}, 0, 0
    for p in parts:
        E, N = len(p["ep_end"]), len(p["t"])
        q = dict(p)
        q["ep_index"] = p["ep_index"] + e0
        q["ep_start"] = p["ep_start"] + r0
        for k, v in q.items():
            if k == "meta" or np.ndim(v) == 0:       # meta and scalar records (tasks_file, student): the first file's
                continue
            out.setdefault(k, []).append(v)
        e0, r0 = e0 + E, r0 + N
    out = {k: np.concatenate(v) for k, v in out.items()}
    for k, v in parts[0].items():
        if k == "meta" or np.ndim(v) == 0:
            out[k] = v
    return out


def load(path):
    """A demo file -> dict of numpy arrays + ``meta`` (dict)."""
    z = dict(np.load(Path(path).expanduser(), allow_pickle=False))
    z["meta"] = json.loads(str(z["meta"])) if "meta" in z else {}
    return z


def summary(d):
    """Per-file summary: episodes, success, failure causes, decisions per episode."""
    from nedm.so101_policy.task import ENDS
    end = d["ep_end"]
    s = d["ep_success"].astype(bool)
    causes = {ENDS[int(k)]: int(v) for k, v in zip(*np.unique(end[~s], return_counts=True))}
    L = d["ep_length"]
    return dict(episodes=int(len(end)), success=float(s.mean()), success10=float(d["ep_success10"].mean()),
                lifted=float(d["ep_lifted"].mean()), failures=causes, decisions=int(len(d["t"])),
                decisions_success=int(L[s].sum()), length_p50=float(np.median(L)), length_max=int(L.max()),
                end_xy_mm_p50=float(np.median(d["ep_end_xy"][s]) * 1e3) if s.any() else None,
                meta={k: v for k, v in d["meta"].items() if k != "code"} if isinstance(d.get("meta"), dict) else {})
