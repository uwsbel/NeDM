"""Joint-target task env and the Markov joint teacher of build9 (``~/so101_policy/docs/build9_viral_student.md``).

Why: the study's PPO teacher acts through ``decoder.DecoderV2`` (hidden macro phases, clocks, line plans that read the
true cube pose), so its label at a state that a camera student reached is ill-defined (build8: DAgger 36 -> 1 %). The
joint teacher ("JT") is Markov: its action at ANY state is a function of the privileged observation of that state.

Observation (``OBS_DIM`` = 47): ``task.observation`` without the decoder state and macro phase (q, qd, TCP, gripper axes,
cube position / yaw / tilt / v / w, cube in the grasp frame, relative yaw, goal, goal - cube, yaw error, the "lifted"
flag and the held-in-air time: the last two are task-tracker values owned by the env, as VIRAL's task stage).

Action (``ACT_DIM`` = 18): one decision every 0.1 s = the 5 joint targets of the next 5 control steps (20 ms) relative
to q at the decision start, D[j] = PHI[j] @ c, c = 3 coefficients per joint (``PHI``: 1, x, (3 x^2 - 1) / 2 over
x = -1 .. 1), normalised: c = mean + std * a (``Codec``; a in data units, no tanh). The env executes the targets
through the ACTION LAYER (``safety``, the same for the student): each arm target clamped to the decoder's lead band
around the MEASURED q of that step (``LEAD``, data p99.9), every target to the collector's target range, and the table
floor (``floor_guard``: the decoder's floor rule in joint space, lowest gripper / moving-jaw point >= 1.5 mm unless the
measured pose is already lower). Stateless: it needs only the measured q and the robot's kinematics, so the label stays
a function of the state.

``JointVecEnv``: B worlds (``vec_env.make_world``: nrd, kin, mjwarp, mujoco) + ``task.PickPlaceTask`` with build8's
macro-free rules (``tray.LOG_ONLY`` log only) + the lenient success tracker (``attach_lenient``). PPO interface of
``ppo.PPO`` (num_envs, num_obs, num_actions, steps_per_decision, reset, step -> obs, reward, done, extras with
action_mask / time_outs / final_obs / episode, gen, ep_count, resolved). ``step(a, targets=...)`` takes absolute
targets [B, 5, 6] instead (student-driven rows). Auto reset from the start bank + ``task.sample_goals`` (training), or
``reset_tasks`` (fixed tasks: evaluation, DAgger collection).

Lenient success (build9 sec. 1.3): released (not held, jaw >= open_jaw, TCP >= tcp_clear above the cube top), at rest
(speed < rest_speed, tilt <= flat_tilt, on the table), after a firm lift, the cube centre inside the goal block's square
(|x|, |y| <= ``BLOCK_HALF`` in the block frame, any yaw), no failure rule in that step; latched per episode. The task
still ends at strict success (15 mm / 15 deg), a failure or the time-out.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy import task as T
from nedm.so101_policy.world import WorldState

OBS_DIM = 47
STEPS = 5
NB = 3
ACT_DIM = NB * 6
ACTION_GROUPS = (("arm", NB * 5), ("jaw", NB))      # a = c.view(3, 6) flattened: [k0 j0..j5, k1 ..., k2 ...]
LEAD = (0.0445, 0.1235, 0.0552, 0.0559, 0.0476)      # decoder lead band (rad), |cmd - q| data p99.9
BLOCK_HALF = 0.025                                   # goal block half side (m): lenient success square
LOG_ONLY = ("grasp_box", "keep_out", "cube_slide")   # build8's macro-defined rules: log only
PP_DATA = Path.home() / "so101_nrd_fix" / "data" / "production_plus"
_X = (torch.arange(STEPS, dtype=torch.float64) - (STEPS - 1) / 2) / ((STEPS - 1) / 2)
PHI = torch.stack((torch.ones_like(_X), _X, (3 * _X ** 2 - 1) / 2), -1)        # [5, 3]


# ---- action codec ------------------------------------------------------------------------------------------------
class Codec:
    """Normalised coefficients a [B, 18] <-> relative targets D [B, 5, 6] (module docstring)."""

    def __init__(self, mean, std, device="cpu"):
        self.dev = torch.device(device)
        self.mean = torch.as_tensor(mean, dtype=torch.float64).reshape(NB, 6).to(self.dev)
        self.std = torch.as_tensor(std, dtype=torch.float64).reshape(NB, 6).to(self.dev)
        self.phi = PHI.to(self.dev)
        self.pinv = torch.linalg.pinv(PHI).to(self.dev)                          # [3, 5]

    @staticmethod
    def fit(D, floor=(1e-3, 1e-3, 1e-3)):
        """Codec from relative targets D [N, 5, 6] (float): mean / std of the least-squares coefficients."""
        D = torch.as_tensor(D, dtype=torch.float64)
        c = torch.einsum("kj,njd->nkd", torch.linalg.pinv(PHI), D)
        fl = torch.as_tensor(floor, dtype=torch.float64)[:, None]
        return Codec(c.mean(0), torch.maximum(c.std(0), fl))

    def to(self, device):
        return Codec(self.mean, self.std, device)

    def state(self):
        return dict(mean=self.mean.cpu().numpy(), std=self.std.cpu().numpy())

    def coefs(self, D):
        return torch.einsum("kj,njd->nkd", self.pinv.to(D), D)

    def encode(self, D):
        """[N, 5, 6] -> normalised a [N, 18]."""
        return ((self.coefs(D) - self.mean.to(D)) / self.std.to(D)).reshape(D.shape[0], ACT_DIM)

    def decode(self, a):
        """normalised a [N, 18] -> relative targets [N, 5, 6]."""
        c = self.mean.to(a) + self.std.to(a) * a.reshape(a.shape[0], NB, 6)
        return torch.einsum("jk,nkd->njd", self.phi.to(a), c)

    def targets(self, a, q):
        """normalised a [B, 18], q [B, 6] at the decision start -> absolute targets [B, 5, 6]."""
        return q[:, None, :].to(a) + self.decode(a)


FLOOR = 0.0015                                       # m, the decoder's low_min (lowest gripper / moving-jaw point)


def _low(kin, u, q):
    """Lowest gripper / moving-jaw collision point (m) of the arm pose u[:, :5] with the commanded and the measured jaw
    angle (the smaller)."""
    a = kin.lowest_point(u, exact=False)
    b = kin.lowest_point(torch.cat((u[:, :5], q[:, 5:6]), -1), exact=False)
    return torch.minimum(a, b)


def floor_guard(u, q, kin, floor=FLOOR, iters=8):
    """Table floor of the action layer (the decoder's floor rule in joint space): a target whose lowest gripper /
    moving-jaw point is below ``floor`` AND below that of the measured pose is pulled back toward the measured q along
    the straight joint line, u' = q + lam (u - q) with the largest lam (bisection) that keeps it at or above
    min(floor, the measured pose's lowest point). Stateless (robot kinematics + measured q). u, q [B, 6] -> [B, 6]."""
    lu = _low(kin, u, q)
    bad = lu < floor - 1e-9
    if not bool(bad.any()):
        return u
    lq = _low(kin, q, q)
    ref = torch.minimum(lq, torch.full_like(lq, floor))
    bad = bad & (lu < ref - 1e-9)
    if not bool(bad.any()):
        return u
    idx = bad.nonzero().flatten()
    ub, qb, rb = u[idx], q[idx], ref[idx]
    lo = torch.zeros(len(idx), dtype=u.dtype, device=u.device)
    hi = torch.ones_like(lo)
    d = torch.cat((ub[:, :5] - qb[:, :5], torch.zeros_like(ub[:, 5:])), -1)
    base = torch.cat((qb[:, :5], ub[:, 5:]), -1)
    for _ in range(iters):
        mid = (lo + hi) / 2
        ok = _low(kin, base + mid[:, None] * d, qb) >= rb - 1e-9
        lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid)
    out = u.clone()
    out[idx] = base + lo[:, None] * d
    return out


def safety(u, q, tlo, thi, lead=None, kin=None, floor=None):
    """The action layer of teacher AND student: arm targets within ``lead`` of the measured q, the table floor
    (``floor_guard`` with ``kin``, if ``floor`` is given), all targets within the collector's range. u, q [B, 6] ->
    [B, 6]."""
    if lead is not None:
        arm = torch.minimum(torch.maximum(u[:, :5], q[:, :5] - lead), q[:, :5] + lead)
        u = torch.cat((arm, u[:, 5:]), -1)
    u = torch.minimum(torch.maximum(u, tlo), thi)
    if kin is not None and floor is not None:
        u = floor_guard(u, q, kin, floor)
    return u


# ---- lenient success ---------------------------------------------------------------------------------------------
def lenient_now(F, cfg: T.TaskConfig, goal_xy, goal_yaw, lifted, half=BLOCK_HALF):
    """[B] bool: the strict success conditions with the position test replaced by 'cube centre inside the goal block's
    square of half side ``half``' and no yaw test."""
    d = F["c"][:, :2] - goal_xy
    c, s = torch.cos(goal_yaw), torch.sin(goal_yaw)
    x, y = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    return (lifted & ~F["held"] & F["on_table"] & (F["tilt"] <= cfg.flat_tilt) & (F["speed"] < cfg.rest_speed)
            & (x.abs() <= half) & (y.abs() <= half) & (F["jaw"] >= cfg.open_jaw)
            & (F["tcp"][:, 2] >= F["c"][:, 2] + T.CUBE_HALF + cfg.tcp_clear))


def attach_lenient(task: T.PickPlaceTask, half=BLOCK_HALF):
    """Wrap ``task.step`` / ``task.reset`` so that ``task.lenient`` [B] bool latches the lenient success of each episode
    (module docstring) and ``task.lenient_t`` [B] holds the step it first held (-1 never). Idempotent."""
    if getattr(task, "_lenient_attached", False):
        return task
    B, dev = task.B, task.dev
    task.lenient = torch.zeros(B, dtype=torch.bool, device=dev)
    task.lenient_t = torch.full((B,), -1, dtype=torch.long, device=dev)
    step0, reset0 = task.step, task.reset

    def step(state, macro=None, macro_goal=None):
        live = task.tr.end == 0
        out = step0(state, macro, macro_goal)
        info = out[3]
        ok = lenient_now(task.F, task.cfg, task.goal_xy, task.goal_yaw, task.tr.lifted, half) & live & ~info["fail"]
        new = ok & ~task.lenient
        task.lenient_t = torch.where(new, task.tr.t, task.lenient_t)
        task.lenient = task.lenient | ok
        return out

    def reset(idx, state, goal_xy, goal_yaw):
        reset0(idx, state, goal_xy, goal_yaw)
        task.lenient[idx] = False
        task.lenient_t[idx] = -1

    task.step, task.reset = step, reset
    task._lenient_attached = True
    return task


# ---- observation ----------------------------------------------------------------------------------------------------
def markov_obs(F, task: T.PickPlaceTask, idx=None):
    """[B, 47] float32 (module docstring) from task features F of the current state."""
    sel = (lambda x: x) if idx is None else (lambda x: x[idx])
    o = T.observation(F, sel(task.goal_xy), sel(task.goal_yaw), sel(task.tr.lifted), sel(task.tr.air_run), task.cfg,
                      None, None)
    return o[:, :OBS_DIM]


# ---- env ------------------------------------------------------------------------------------------------------------
@dataclass
class JointEnvConfig:
    world: str = "nrd"
    num_envs: int = 4096
    device: str = "cuda"
    seed: int = 0
    ckpts: list = field(default_factory=list)
    members: object = "split"
    nrd_cuda_graph: bool = True
    starts: str = str(PP_DATA)
    start_split: int = 0
    pool_size: int = 65536
    gamma: float = 0.99
    horizon: int = 400
    task: dict = field(default_factory=dict)          # TaskConfig overrides
    log_only: tuple = LOG_ONLY
    lead_clamp: bool = True
    floor_guard: bool = True                          # the table floor of the action layer (``floor_guard``)
    codec: str = ""                                   # .npz with mean / std (Codec)
    auto_reset: bool = True
    stagger: bool = True
    block_half: float = BLOCK_HALF
    geom_path: str = str(Path.home() / "so101_policy" / "ckpt" / "task_geom_v2.npz")
    kin_points: str = str(Path.home() / "so101_policy" / "ckpt" / "kin_points_v1.npz")
    mj_workers: int = 0
    mj_python: object = None
    mjwarp: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d):
        d = dict(d or {})
        known = {f.name for f in fields(JointEnvConfig)}
        bad = sorted(set(d) - known - {"note", "notes"})
        if bad:
            raise ValueError(f"unknown joint env config keys {bad}")
        if "log_only" in d:
            d["log_only"] = tuple(d["log_only"])
        return JointEnvConfig(**{k: v for k, v in d.items() if k in known})

    def to_dict(self):
        return copy.deepcopy(asdict(self))


def load_codec(path, device="cpu"):
    z = np.load(Path(path).expanduser())
    return Codec(z["mean"], z["std"], device)


class JointVecEnv:
    """B task worlds with joint-target actions (module docstring)."""

    def __init__(self, cfg: JointEnvConfig | dict, codec: Codec | None = None, share=None):
        from nedm.so101_policy import vec_env as VE
        self.cfg = cfg = JointEnvConfig.from_dict(cfg) if isinstance(cfg, dict) else cfg
        self.dev = dev = torch.device("cpu" if cfg.world == "mujoco" else cfg.device)
        self.B = B = int(cfg.num_envs)
        self.num_envs, self.num_actions, self.num_obs = B, ACT_DIM, OBS_DIM
        self.action_groups = [n for _, n in ACTION_GROUPS]
        self.action_group_names = [g for g, _ in ACTION_GROUPS]
        self.steps_per_decision = STEPS
        self.kin = share.kin if share is not None else VE.kinematics(cfg.kin_points, dev)
        self.geom = share.geom if share is not None else VE.task_geometry(cfg.geom_path, dev)
        self.tcfg = T.TaskConfig(**{"horizon": int(cfg.horizon), "obs_rates": True, "log_only": tuple(cfg.log_only),
                                    "gamma": float(cfg.gamma) ** (1.0 / STEPS), **dict(cfg.task)})
        self.max_episode_length = math.ceil(self.tcfg.horizon / STEPS)
        wcfg = VE.EnvConfig(world=cfg.world, num_envs=B, device=cfg.device, ckpts=list(cfg.ckpts), members=cfg.members,
                            nrd_cuda_graph=cfg.nrd_cuda_graph, starts="", mj_workers=cfg.mj_workers,
                            mj_python=cfg.mj_python, mjwarp=dict(cfg.mjwarp))
        self.world = VE.make_world(wcfg, B, dev, self.kin, self.geom, share)
        self.task = attach_lenient(T.PickPlaceTask(B, dev, kin=self.kin, geom=self.geom, cfg=self.tcfg,
                                                   overlap_rules=cfg.world != "mujoco"), cfg.block_half)
        if codec is None:
            if not cfg.codec:
                raise ValueError("JointVecEnv needs a codec (cfg.codec or the argument)")
            codec = load_codec(cfg.codec)
        self.codec = codec.to(dev)
        self.tlo, self.thi = self.kin.tlo.to(dev), self.kin.thi.to(dev)
        self.lead = torch.tensor(LEAD, dtype=torch.float64, device=dev) if cfg.lead_clamp else None
        self.gen = torch.Generator(dev).manual_seed(int(cfg.seed))
        if share is not None:
            self.bank = share.bank
        else:
            self.bank = VE.load_starts(cfg.starts, cfg.start_split, dev) if cfg.starts and cfg.world != "mujoco" else None
        self.pool = None
        f, li = dict(dtype=torch.float64, device=dev), dict(dtype=torch.long, device=dev)
        self.ep_count = torch.zeros(B, **li)
        self.ep_ret = torch.zeros(B, **f)
        self.ep_len = torch.zeros(B, **li)
        self.task_seed = torch.zeros(B, **li)
        self.ws = None
        self.obs = None
        self.world_steps = 0
        self.decisions = 0
        self.u_exe = None                 # [B, 5, 6] targets executed in the last decision (after the action layer)

    def resolved(self) -> dict:
        return dict(env=self.cfg.to_dict(), task=asdict(self.tcfg), codec={k: v.tolist() for k, v in
                                                                            self.codec.state().items()},
                    num_obs=self.num_obs, num_actions=self.num_actions,
                    action_groups=list(zip(self.action_group_names, self.action_groups)), kind="joint_env")

    # ---- tasks -------------------------------------------------------------------------------------------------
    def refresh_pool(self, n=None):
        if self.bank is None:
            raise RuntimeError("no start bank (world mujoco: use reset_tasks)")
        n = int(n or self.cfg.pool_size)
        N = self.bank["q"].shape[0]
        i = torch.randint(N, (n,), generator=self.gen, device=self.dev)
        yaw, _ = T.cube_yaw_tilt(self.bank["cube_quat"][i])
        gxy, gyaw = T.sample_goals(self.bank["cube_pos"][i, :2], yaw, self.gen)
        self.pool = dict(i=i, goal_xy=gxy, goal_yaw=gyaw)

    def _reset_rows(self, idx, start: WorldState, goal_xy, goal_yaw, seeds, mj_start=None):
        if self.cfg.world == "mujoco":
            ws = self.world.reset(idx.cpu(), mj_start, warmup=True)
        elif self.cfg.world == "mjwarp":
            ws = self.world.reset(idx, start, warmup=False)
        else:
            ws = self.world.reset(idx, start)
        self.ws = ws.to(self.dev)
        self.task.reset(idx, self.ws.index(idx), goal_xy, goal_yaw)
        self.task_seed[idx] = seeds
        self.ep_ret[idx], self.ep_len[idx] = 0.0, 0

    def _pool_reset(self, idx):
        from nedm.so101_policy.vec_env import start_rows
        if self.pool is None:
            self.refresh_pool()
        k = torch.randint(len(self.pool["i"]), (len(idx),), generator=self.gen, device=self.dev)
        i = self.pool["i"][k]
        s = (int(self.cfg.seed) * 0x9E3779B1 + self.ep_count[idx] * self.B + idx) & 0xFFFFFFFF
        self.ep_count[idx] += 1
        self._reset_rows(idx, start_rows(self.bank, i), self.pool["goal_xy"][k], self.pool["goal_yaw"][k], s)

    def reset(self):
        idx = torch.arange(self.B, device=self.dev)
        self._pool_reset(idx)
        if self.cfg.stagger and self.cfg.auto_reset:
            k = torch.randint(self.max_episode_length, (self.B,), generator=self.gen, device=self.dev)
            self.task.tr.t.copy_(k * STEPS)
        self.obs = self._observe()
        return self.obs

    def reset_tasks(self, idx, starts: WorldState, goal_xy, goal_yaw, seeds, mj_start=None, check_tol=1e-6):
        idx = torch.as_tensor(idx, dtype=torch.long, device=self.dev)
        st = starts.to(self.dev)
        self._reset_rows(idx, st, torch.as_tensor(goal_xy, dtype=torch.float64, device=self.dev),
                         torch.as_tensor(goal_yaw, dtype=torch.float64, device=self.dev),
                         torch.as_tensor(seeds, dtype=torch.long, device=self.dev), mj_start)
        if self.cfg.world == "mujoco" and check_tol is not None:
            got = self.ws.index(idx)
            dq = float((got.q - st.q).abs().max())
            dc = float((got.cube_pos - st.cube_pos).abs().max())
            if dq > check_tol or dc > check_tol:
                raise RuntimeError(f"MuJoCo warm-up start differs from the stored settled state ({dq:.2e}, {dc:.2e})")
        self.obs = self._observe()
        return self.obs

    # ---- observation ---------------------------------------------------------------------------------------------
    def _observe(self, F=None, idx=None):
        ws = self.ws if idx is None else self.ws.index(idx)
        if F is None:
            F = T.features(ws, self.kin, None, self.tcfg)
        return markov_obs(F, self.task, idx)

    def get_observations(self):
        return self.obs

    # ---- one decision --------------------------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, actions=None, targets=None, use_targets=None, step_hook=None):
        """actions [B, 18] normalised coefficients (``Codec``) and/or targets [B, 5, 6] absolute (``use_targets`` [B]
        bool: rows that take ``targets``; default all rows if only targets are given) -> (obs [B, 47], reward [B],
        done [B], extras). step_hook(env, j): called after each of the 5 world steps (records)."""
        task, B = self.task, self.B
        live0 = task.tr.end == 0
        q0 = self.ws.q.to(torch.float64)
        if actions is not None:
            a = actions.to(device=self.dev, dtype=torch.float64).clamp(-8.0, 8.0)
            u5 = self.codec.targets(a, q0)
        else:
            u5 = None
        if targets is not None:
            tg = targets.to(device=self.dev, dtype=torch.float64)
            if u5 is None:
                u5 = tg
            else:
                m = torch.ones(B, dtype=torch.bool, device=self.dev) if use_targets is None else use_targets.to(self.dev)
                u5 = torch.where(m[:, None, None], tg, u5)
        rew = torch.zeros(B, dtype=torch.float64, device=self.dev)
        info = None
        ex = []
        for j in range(STEPS):
            u = safety(u5[:, j], self.ws.q.to(torch.float64), self.tlo, self.thi, self.lead, self.kin,
                       FLOOR if self.cfg.floor_guard else None)
            ex.append(u)
            ws = self.world.step(u)
            self.ws = ws if self.cfg.world != "mujoco" else ws.to(self.dev)
            r, _, _, info = task.step(self.ws, None, None)
            rew += r
            if step_hook is not None:
                step_hook(self, j)
        self.u_exe = torch.stack(ex, 1)
        self.world_steps += B * STEPS
        self.decisions += B
        end = info["end"]
        done = live0 & (end != 0)
        time_out = done & (end == T.EC["timeout"])
        self.ep_ret += rew
        self.ep_len += live0.long()
        ep = dict(end=end.clone(), success=end == T.EC["success"], success10=info["success10"].clone(),
                  lifted=info["lifted"].clone(), end_xy=info["end_xy_err"].clone(), end_yaw=info["end_yaw_err"].clone(),
                  ret=self.ep_ret.clone(), length=self.ep_len.clone(), steps=info["t"].clone(),
                  lenient=task.lenient.clone(), lenient_t=task.lenient_t.clone())
        obs = self._observe(task.F)
        final_obs = obs
        if self.cfg.auto_reset:
            idx = done.nonzero().flatten()
            if len(idx):
                obs = obs.clone()
                self._pool_reset(idx)
                obs[idx] = self._observe(idx=idx)
        self.obs = obs
        mask = torch.ones(B, ACT_DIM, dtype=torch.bool, device=self.dev)
        extras = dict(time_outs=time_out, final_obs=final_obs, action_mask=mask, episode=ep, counts={},
                      world_steps=B * STEPS)
        return obs, rew.float(), done, extras

    def close(self):
        if hasattr(self.world, "close"):
            self.world.close()


# ---- the joint teacher ----------------------------------------------------------------------------------------------
class JointTeacher:
    """A JT checkpoint (``ppo.PPO.save`` / ``save_bc``): ``act(obs)`` -> deterministic normalised action [B, 18];
    ``targets(obs, q)`` -> absolute targets [B, 5, 6]; ``std`` the PPO action std."""

    def __init__(self, path, device="cuda"):
        from nedm.so101_policy.ppo import Policy, sha256
        self.path = str(Path(path).expanduser())
        d = torch.load(self.path, map_location=device, weights_only=False)
        self.pol = Policy(**d["policy_kwargs"]).to(device)
        self.pol.load_state_dict(d["policy"])
        self.pol.eval()
        rec = d.get("env") or {}
        self.codec = Codec(np.asarray(rec["codec"]["mean"]), np.asarray(rec["codec"]["std"]), device)
        self.env_rec = rec
        self.dev = torch.device(device)
        self.std = self.pol.std.detach().float()
        self.sha256 = sha256(self.path)
        self.iteration = int(d.get("iteration", 0))

    @torch.no_grad()
    def act(self, obs):
        return self.pol.act_inference(obs.to(self.dev, torch.float32)).to(obs.device)

    @torch.no_grad()
    def targets(self, obs, q):
        a = self.act(obs).to(q.device, torch.float64)
        return self.codec.to(q.device).targets(a, q.to(torch.float64))

    def info(self):
        return dict(path=self.path, sha256=self.sha256, iteration=self.iteration)


def save_bc(path, policy, codec: Codec, env_cfg: dict, extra=None):
    """A BC-trained ``ppo.Policy`` in the ``PPO.save`` layout (so ``JointTeacher`` and PPO resume read it)."""
    rec = dict(env=env_cfg, codec={k: v.tolist() for k, v in codec.state().items()}, kind="joint_env",
               num_obs=OBS_DIM, num_actions=ACT_DIM, action_groups=list(ACTION_GROUPS))
    torch.save(dict(policy=policy.state_dict(), policy_kwargs=policy.kwargs, iteration=0, env=rec,
                    extra=extra or {}), path)
