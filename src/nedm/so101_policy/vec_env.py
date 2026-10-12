"""Batched goal pick-and-place task env of the SO101 + cube policy study: world + DecoderV2 + PickPlaceTask +
observation + reward, one policy decision per 0.1 s (spec_v2 B1-B4). The same class trains PPO in every world and
evaluates any policy in every world, CPU MuJoCo included (``scripts/so101_policy/train_ppo.py``, ``eval_policy.py``).

Worlds (``WORLDS``; all take 6 joint targets per 20 ms and return a ``world.WorldState``):

| kind | class | device | training starts (``EnvConfig.starts``) |
|---|---|---|---|
| nrd | ``nrd_sim.GraspNrdSim`` (checkpoints ``ckpts``, ``members``) | CUDA (CPU works) | bank, exact resting history |
| kin | ``kin_world.KinWorld`` (kinematic control: no learned dynamics) | CUDA / CPU | bank |
| mjwarp | ``mjwarp_env.MjwarpWorlds`` (MuJoCo Warp, conda env ``so101_mjwarp``) | CUDA | bank, exact reset (``warmup=False``) |
| mujoco | ``mujoco_env.MujocoWorlds`` (CPU MuJoCo, the ground truth; ``mj_workers`` processes) | CPU | evaluation only (tasks with the collector warm-up) |

One ``step(actions)`` = one decision:

1. ``a = tanh(actions)`` (the policy's Gaussian lives in pre-tanh space; ``squashed=True`` passes a in [-1, 1] as it
   is, e.g. the FSM expert), ``DecoderV2.decide(a, state)``. Action (``decoder`` module docstring): a[0:3] TCP target
   displacement (x 35 mm), a[3] yaw change (x 0.2 rad), a[4] GRASP / PLACE trigger (> 0), a[5] GRASP lift height
   (3-12 cm). Action groups for the std (``ACTION_GROUPS``): move (4), trigger (1), lift (1).
2. ``decision_steps`` (5) x [``DecoderV2.step(q)`` -> joint targets, world step, ``PickPlaceTask.step(state, macro,
   macro_goal)``]. The decision reward = the sum of the 5 task rewards (worlds that ended inside the decision get 0 for
   the rest of it; their world keeps stepping until the reset at the decision end).
3. Done = the task ended in this decision (success, a failure rule, or the time-out at ``horizon`` = 400 steps = 80
   decisions); ``extras["time_outs"]`` marks time-outs (value bootstrap) and ``extras["final_obs"]`` holds the
   observation of the last state before the reset (all worlds; it equals ``obs`` for worlds that did not end).
4. ``auto_reset``: ended worlds get a new task from the task pool and a new decoder macro seed, then the observation.
   ``reset()`` staggers the first episodes (``stagger``: random start of the episode clock), so the 80-decision
   time-outs of all worlds do not fall in the same PPO iteration.

Reward (``task.PickPlaceTask``, P1-14): progress shaping, lift bonus, success bonus (terminal), time cost, failure =
``fail_cost`` + the discounted time cost of the remaining steps (``charge_remaining``). The task discounts that charge
per 20 ms step with ``TaskConfig.gamma``; the env sets it to ``gamma ** (1 / decision_steps)`` (``EnvConfig.gamma`` =
PPO's per-decision discount), so the charge equals PPO's discounted time cost of the remaining decisions to 0.5 %.

Action mask (``extras["action_mask"]`` [B, 6] bool; PPO drops the unused dims from the log-likelihood, review P1-10):
move dims where the decision drove free motion (decoder free / hold, or a GRASP abort; not on a trigger decision,
whose free motion is the brake), the trigger where it was read (free / hold, and the GRASP descent before the close,
where a[4] <= 0 aborts), the lift height where a GRASP started. Inside the other macro phases no dim is used. With
the decoder's CARRY macro (``decoder.carry_macro``, option b2): in mode hold at rest a[0:2] and a[3] set the CARRY
line (a[2] unused), during a CARRY line only a[5] (its abort) is read.

Observation: ``task.observation`` (the same function for every world; no contact flags) with the decoder state
(``DecoderV2.observation``, 24): ``task.obs_dim(cfg, 24)`` = 79 (73 with ``task.obs_rates`` False). Optional
Gaussian noise on the cube v / w entries (review F11; ``obs_rate_noise`` = (m/s, rad/s), training only).

Starts and goals (spec_v2 B4; written down here):

- ``starts`` = a start bank: the first recorded states of the task-family episodes of one split of a grasp dataset
  (``task.load_start_bank``: production_plus split 0 = 25,197 starts, cube flat at rest, z = 14.993 mm; never the
  sealed test), or an .npz from ``train_ppo.py bank`` (the same rows) or ``train_ppo.py starts`` (fresh collector
  starts: ``mujoco_env.sample_starts`` on new seeds, settled by the collector's 0.3 s warm-up; seeds 200000+ by
  default, never the validation 1000-2023, the test 900000+ or a collector seed). Both are the collector's task start
  distribution (arm at a top-down pose, cube on the annulus r 0.15-0.27 m, +-75 deg), so the two choices are
  comparable; the data bank is the default for ALL training worlds (NRD, kinematic, MuJoCo Warp: its exact reset of a
  recorded post-warm-up state), which keeps the arms on identical starts.
- Task pool: ``pool_size`` (start index, goal) pairs drawn once on the env's device (``task.sample_goals``: annulus,
  6-30 cm from the start cube, yaw +-30 deg; generator ``seed``), ``refresh_pool()`` draws a new pool (the trainer
  calls it between iterations: no host sync inside ``step``). A reset picks pool entries uniformly on the device.
- Decoder macro seeds (review F5): world slot i, its k-th episode -> ``(seed * 0x9E3779B1 + k * B + i) mod 2^32``,
  so durations differ between the episodes of a slot. Evaluation (``reset_tasks``) passes the task seed (the gate's
  rule: identical macro durations in every world).

Grasp assist (``grasp_assist``, off by default; a curriculum hook, NOT the study's rule set): a GRASP trigger whose
descent end (``brake_stop`` = ``DecoderV2.macro_goal`` before the decision, the decoder's own brake rule) would put
the cube outside the training grasp box (|x| <= ``box_x``, |y| <= ``box_y``, yaw <= ``close_yaw``, x 0.98) is
ignored (a[4] := -1) and charged ``grasp_assist_cost``, instead of ending the episode by the grasp-box rule; and a
started GRASP is not aborted (a[4] := +1 in g_brake / g_descend / g_settle; the trigger dim is then unused). Purpose:
PPO finds lifts. With the rule alone every early trigger fails and the policy stops triggering; with the box part alone
every in-box GRASP is aborted at the next decision by the random trigger (measured 2026-10-11 21:35, build3_ppo.md).
A later stage turns the assist off (review P0-8 keeps the rule as a termination for the final policy, and the policy
must then hold the trigger itself). ``extras["counts"]`` holds per-decision event flags (assist_blocked, assist_held,
grasp_started, place_started, aborted) for the learning curve.

Host syncs per decision: one ``nonzero`` for the reset indices (plus the decoder's and the worlds' own ``.any()``
checks). Everything else stays on the env's device.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import torch

from nedm.so101_policy import task as T
from nedm.so101_policy.decoder import DecoderV2, DecoderV2Config, M
from nedm.so101_policy.kin import SO101Kin
from nedm.so101_policy.world import WorldState

WORLDS = ("nrd", "kin", "mjwarp", "mujoco")
ACTION_DIM = 6
ACTION_GROUPS = (("move", 4), ("trigger", 1), ("lift", 1))
GEOM_NPZ = Path.home() / "so101_policy" / "ckpt" / "task_geom_v1.npz"
PP_DATA = Path.home() / "so101_nrd_fix" / "data" / "production_plus"
_M32 = 0xFFFFFFFF
EP_KEYS = ("end", "success", "success10", "lifted", "end_xy", "end_yaw", "ret", "length", "steps")


@dataclass
class EnvConfig:
    world: str = "nrd"
    num_envs: int = 4096
    device: str = "cuda"
    seed: int = 0                        # task pool, decoder macro seeds, observation noise
    ckpts: list = field(default_factory=list)   # NRD checkpoints (paths)
    members: object = "split"            # GraspNrdSim members
    nrd_cuda_graph: bool = True          # capture the NRD step in a CUDA graph (CUDA only)
    starts: str = str(PP_DATA)           # data dir (unified_data.npz) or .npz start bank (module docstring)
    start_split: int = 0
    pool_size: int = 65536
    gamma: float = 0.99                  # PPO discount per decision (task charge per step = gamma ** (1 / 5))
    task: dict = field(default_factory=dict)      # TaskConfig overrides
    eval_rules: bool = False             # TaskConfig.evaluation(): the grasp-box rule logs only
    overlap_rules: object = None         # None: on, except in CPU MuJoCo (real contacts + the collector's overlap end)
    decoder: dict = field(default_factory=dict)   # DecoderV2Config overrides
    obs_rate_noise: tuple = (0.0, 0.0)   # std of Gaussian noise on the observed cube v (m/s) and w (rad/s)
    auto_reset: bool = True
    stagger: bool = True                 # reset(): episode clocks start at random decisions (spread the time-outs)
    grasp_assist: bool = False           # exploration aid (curriculum, module docstring): out-of-box GRASP triggers ignored
    grasp_assist_cost: float = 0.05      # reward charged for an ignored trigger
    geom_path: str = str(GEOM_NPZ)       # TaskGeometry .npz (machines without the vendor model); missing -> MuJoCo
    mj_workers: int = 0                  # CPU MuJoCo worker processes
    mj_python: object = None             # interpreter of the CPU MuJoCo workers (luffy: conda nrdsim has scipy)
    mjwarp: dict = field(default_factory=dict)    # MjwarpWorlds kwargs (nconmax, njmax, graph, opt)

    @staticmethod
    def from_dict(d: dict | None) -> "EnvConfig":
        d = dict(d or {})
        known = {f.name for f in fields(EnvConfig)}
        bad = sorted(set(d) - known - {"note", "notes"})
        if bad:
            raise ValueError(f"unknown env config keys {bad}")
        return EnvConfig(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return copy.deepcopy(asdict(self))


def task_config(cfg: EnvConfig) -> T.TaskConfig:
    """The env's TaskConfig: training rules (or evaluation rules), the overrides, and the per-step charge discount."""
    over = dict(cfg.task)
    if "log_only" in over:
        over["log_only"] = tuple(over["log_only"])
    dec_steps = DecoderV2Config(**cfg.decoder).decision_steps if cfg.decoder else DecoderV2Config().decision_steps
    over.setdefault("gamma", float(cfg.gamma) ** (1.0 / dec_steps))
    return T.TaskConfig.evaluation(**over) if cfg.eval_rules else T.TaskConfig(**over)


def obs_layout(tcfg: T.TaskConfig, dec_dim: int = DecoderV2.OBS_DIM) -> dict:
    """name -> slice of ``task.observation`` (same order as there)."""
    parts = [("q", 6), ("qd", 6), ("tcp", 3), ("grip_x", 3), ("grip_z", 3), ("cube_pos", 3), ("cube_yaw4", 2),
             ("cube_tilt", 1)]
    if tcfg.obs_rates:
        parts += [("cube_v", 3), ("cube_w", 3)]
    parts += [("cube_in_grasp", 3), ("rel_yaw4", 2), ("goal_xy", 2), ("goal_yaw4", 2), ("goal_minus_cube", 2),
              ("yaw_err", 1), ("lifted", 1), ("hold_time", 1), ("macro", len(T.MACROS)), ("decoder", dec_dim)]
    out, k = {}, 0
    for n, w in parts:
        out[n] = slice(k, k + w)
        k += w
    return out


def task_geometry(path, device):
    path = Path(path) if path else None
    return T.TaskGeometry.load(path, device) if path is not None and path.exists() else T.TaskGeometry.from_mujoco(device)


def load_starts(spec, split=0, device="cpu") -> dict:
    """Start bank -> dict of float64 tensors q, qd, cube_pos, cube_quat, cube_v, cube_w [N, ...] on device. spec: a
    grasp data directory (``task.load_start_bank``, task families of ``split``), an .npz with these keys (``train_ppo.py
    bank`` / ``starts``), or a gate tasks file (``state_*`` keys). Refuses the sealed test."""
    import numpy as np
    p = Path(str(spec)).expanduser()
    if "test_sealed" in str(p):
        raise ValueError("the sealed test data must not be used for starts")
    keys = ("q", "qd", "cube_pos", "cube_quat", "cube_v", "cube_w")
    if p.is_dir():
        b = T.load_start_bank(p, split=split, device=device)
        return {k: b[k] for k in keys}
    f = np.load(p)
    pre = "state_" if f"state_q" in f.files else ""
    return {k: torch.as_tensor(f[pre + k], dtype=torch.float64, device=device) for k in keys}


def start_rows(bank: dict, i) -> WorldState:
    """WorldState of the bank rows i (flags False)."""
    n, dev = len(i), bank["q"].device
    z = torch.zeros(n, dtype=torch.bool, device=dev)
    return WorldState(q=bank["q"][i], qd=bank["qd"][i], cube_pos=bank["cube_pos"][i], cube_quat=bank["cube_quat"][i],
                      cube_v=bank["cube_v"][i], cube_w=bank["cube_w"][i], contact=torch.zeros(n, 3, dtype=torch.bool, device=dev),
                      arm_table=z, invalid=z.clone())


def make_world(cfg: EnvConfig, B: int, device, kin=None, geom=None, share=None):
    """World of kind cfg.world with B worlds. share: another env whose NRD models (loaded once) are reused."""
    kind = cfg.world
    if kind == "nrd":
        from nedm.so101_policy.nrd_sim import GraspNrdSim
        if share is not None and share.cfg.world == "nrd":
            ck = dict(zip(share.world.member_names, share.world.models))
            members = "split"
        else:
            if not cfg.ckpts:
                raise ValueError("world nrd needs ckpts")
            ck = {Path(p).stem: str(Path(p).expanduser()) for p in cfg.ckpts}
            members = cfg.members
        return GraspNrdSim(ck, num_envs=B, device=device, members=members,
                           cuda_graph=bool(cfg.nrd_cuda_graph) and torch.device(device).type == "cuda")
    if kind == "kin":
        from nedm.so101_policy.kin_world import KinWorld
        return KinWorld(B, device, kin=kin, geom=geom)
    if kind == "mjwarp":
        from nedm.so101_policy.mjwarp_env import MjwarpWorlds
        return MjwarpWorlds(B, device=str(device), **cfg.mjwarp)
    if kind == "mujoco":
        from nedm.so101_policy.mujoco_env import MujocoWorlds
        return MujocoWorlds(B, workers=cfg.mj_workers, python=cfg.mj_python)
    raise ValueError(f"world {kind!r}: one of {WORLDS}")


def brake_stop(dec: DecoderV2):
    """Where a GRASP / PLACE trigger at the next decision stops the decoder target (``DecoderV2.decide``'s brake rule;
    cosine blend): -> (xy [B, 2] m, yaw [B] rad) = ``macro_goal`` x, y, yaw of a GRASP started now (copy of
    ``scripts/so101_policy/gate_expert.brake_stop``; tested against the decoder)."""
    from nedm.so101_policy.decoder import CONTROL_DT as DT
    c = dec.cfg
    v0, w0 = dec.vel, dec.yrate
    rmap = dec._rate_map()
    sp, wr = v0.norm(dim=-1), w0.abs()
    qd0 = (rmap @ torch.cat((v0, w0[:, None]), -1)[..., None])[..., 0].abs()
    tj = (math.pi * qd0 * DT * DT / (2 * dec.ddq_max)).amax(-1)
    tb = torch.maximum(torch.maximum(torch.full_like(sp, dec.T_d), math.pi * sp / (2 * c.a_max)),
                       torch.maximum(math.pi * wr / (2 * c.alpha_max), tj))
    tb = torch.ceil(tb / DT - 1e-9) * DT
    tb = torch.where((sp < 1e-9) & (wr < 1e-9), torch.zeros_like(tb), tb)
    return dec.pos[:, :2] + v0[:, :2] * (tb / 2)[:, None], dec.yaw + w0 * tb / 2


class PickPlaceVecEnv:
    """B task worlds on one device (module docstring). ``reset() -> obs``; ``step(actions) -> (obs, reward, done,
    extras)``; ``reset_tasks(idx, starts, goal_xy, goal_yaw, seeds)`` for fixed evaluation tasks; ``set_tolerance``
    (curriculum). share: an env whose kinematics, geometry, start bank and NRD models are reused (validation env)."""

    def __init__(self, cfg: EnvConfig | dict, share: "PickPlaceVecEnv | None" = None, record: bool = False):
        self.cfg = cfg = EnvConfig.from_dict(cfg) if isinstance(cfg, dict) else cfg
        if cfg.world not in WORLDS:
            raise ValueError(f"world {cfg.world!r}: one of {WORLDS}")
        self.dev = dev = torch.device("cpu" if cfg.world == "mujoco" else cfg.device)
        self.B = B = int(cfg.num_envs)
        self.num_envs, self.num_actions = B, ACTION_DIM
        self.action_groups = [n for _, n in ACTION_GROUPS]
        self.kin = share.kin if share is not None else SO101Kin(dev)
        self.geom = share.geom if share is not None else task_geometry(cfg.geom_path, dev)
        self.tcfg = task_config(cfg)
        ov = cfg.overlap_rules if cfg.overlap_rules is not None else cfg.world != "mujoco"
        self.world = make_world(cfg, B, dev, self.kin, self.geom, share)
        self.dec = DecoderV2(B, dev, kin=self.kin, cfg=replace(DecoderV2Config(), **cfg.decoder), seed=cfg.seed)
        self.task = T.PickPlaceTask(B, dev, kin=self.kin, geom=self.geom, cfg=self.tcfg, overlap_rules=bool(ov))
        self.steps_per_decision = self.dec.cfg.decision_steps
        self.max_episode_length = math.ceil(self.tcfg.horizon / self.steps_per_decision)   # decisions
        self.layout = obs_layout(self.tcfg, DecoderV2.OBS_DIM)
        self.num_obs = T.obs_dim(self.tcfg, DecoderV2.OBS_DIM)
        self.gen = torch.Generator(dev).manual_seed(int(cfg.seed))
        if share is not None:
            self.bank = share.bank
        else:     # no bank for evaluation envs (starts "") and CPU MuJoCo (fixed tasks only)
            self.bank = load_starts(cfg.starts, cfg.start_split, dev) if cfg.starts and cfg.world != "mujoco" else None
        self.pool = None
        f, li = dict(dtype=torch.float64, device=dev), dict(dtype=torch.long, device=dev)
        self.ep_count = torch.zeros(B, **li)
        self.ep_ret = torch.zeros(B, **f)
        self.ep_len = torch.zeros(B, **li)
        self.task_seed = torch.zeros(B, **li)
        self.ws = None
        self.obs = None
        self.record = record
        self.rec = None
        self.world_steps = 0             # world (20 ms) steps of all worlds since construction
        self.decisions = 0

    # ---- tasks -------------------------------------------------------------------------------------------------
    def refresh_pool(self, n=None):
        """A new pool of (start index, goal_xy, goal_yaw) on the device (host syncs here, not in step)."""
        if self.bank is None:
            raise RuntimeError("no start bank (world mujoco: use reset_tasks)")
        n = int(n or self.cfg.pool_size)
        N = self.bank["q"].shape[0]
        i = torch.randint(N, (n,), generator=self.gen, device=self.dev)
        yaw, _ = T.cube_yaw_tilt(self.bank["cube_quat"][i])
        gxy, gyaw = T.sample_goals(self.bank["cube_pos"][i, :2], yaw, self.gen)
        self.pool = dict(i=i, goal_xy=gxy, goal_yaw=gyaw)

    def set_tolerance(self, tol_xy=None, tol_yaw=None):
        """Curriculum hook (tolerance ladder, review P1-13): success tolerance of the running and future episodes."""
        if tol_xy is not None:
            self.tcfg.tol_xy = float(tol_xy)
        if tol_yaw is not None:
            self.tcfg.tol_yaw = float(tol_yaw)

    def set_grasp_assist(self, on: bool, cost=None):
        """Curriculum hook: the grasp assist on / off (module docstring)."""
        self.cfg.grasp_assist = bool(on)
        if cost is not None:
            self.cfg.grasp_assist_cost = float(cost)

    def _assist(self, a):
        """Grasp assist: [B] bool of the GRASP triggers this decision that would fail the training grasp box."""
        dec, tc = self.dec, self.tcfg
        free = dec.mode == M["free"]
        jaw_ready = (dec.jaw_k >= dec.jaw_n) & ((dec.jaw_to - dec.cfg.jaw_open).abs() < 1e-9)
        trig = free & dec.armed & jaw_ready & (a[:, 4] > dec.cfg.trigger_thr)
        sxy, syaw = brake_stop(dec)
        cyaw, _ = T.cube_yaw_tilt(self.ws.cube_quat.to(torch.float64))
        d = self.ws.cube_pos[:, :2].to(torch.float64) - sxy
        c, s_ = torch.cos(syaw), torch.sin(syaw)
        x, y = c * d[:, 0] + s_ * d[:, 1], -s_ * d[:, 0] + c * d[:, 1]
        ok = (x.abs() <= 0.98 * tc.box_x) & (y.abs() <= 0.98 * tc.box_y) & (T.wrap90(syaw - cyaw).abs() <= 0.98 * tc.close_yaw)
        return trig & ~ok

    def _episode_seeds(self, idx):
        s = (int(self.cfg.seed) * 0x9E3779B1 + self.ep_count[idx] * self.B + idx) & _M32
        self.ep_count[idx] += 1
        return s

    # ---- resets ------------------------------------------------------------------------------------------------
    def _reset_rows(self, idx, start: WorldState, goal_xy, goal_yaw, seeds, mj_start=None):
        """World, decoder and task reset of worlds idx (long [n]); returns nothing (self.ws updated)."""
        if self.cfg.world == "mujoco":
            ws = self.world.reset(idx.cpu(), mj_start, warmup=True)
        elif self.cfg.world == "mjwarp":
            ws = self.world.reset(idx, start, warmup=False)
        else:
            ws = self.world.reset(idx, start)
        self.ws = ws.to(self.dev)
        rows = self.ws.index(idx)
        self.dec.reset(idx, rows.q, seed=seeds)
        self.task.reset(idx, rows, goal_xy, goal_yaw)
        self.task_seed[idx] = seeds
        self.ep_ret[idx], self.ep_len[idx] = 0.0, 0

    def _pool_reset(self, idx):
        if self.pool is None:
            self.refresh_pool()
        k = torch.randint(len(self.pool["i"]), (len(idx),), generator=self.gen, device=self.dev)
        i = self.pool["i"][k]
        self._reset_rows(idx, start_rows(self.bank, i), self.pool["goal_xy"][k], self.pool["goal_yaw"][k],
                         self._episode_seeds(idx))

    def reset(self):
        """Every world gets a pool task; with ``stagger`` its episode clock starts at a random decision in
        [0, horizon) (the task's step counter, a multiple of decision_steps: the first episodes are shorter, so the
        time-outs of all worlds do not fall on the same decision; RSL-RL's init_at_random_ep_len). -> observation
        [B, D] float32."""
        idx = torch.arange(self.B, device=self.dev)
        self._pool_reset(idx)
        if self.cfg.stagger and self.cfg.auto_reset:
            k = torch.randint(self.max_episode_length, (self.B,), generator=self.gen, device=self.dev)
            self.task.tr.t.copy_(k * self.steps_per_decision)
        self.obs = self._observe()
        self._rec_init()
        return self.obs

    def reset_tasks(self, idx, starts: WorldState, goal_xy, goal_yaw, seeds, mj_start=None, check_tol=1e-6):
        """Fixed tasks (evaluation): worlds idx (long [n]) start at the settled states ``starts`` with the goals and
        decoder seeds given. World mujoco: ``mj_start`` (dict q, cube_pos, cube_quat before the warm-up, as
        ``sample_starts``) is reset with the collector warm-up and must reproduce ``starts`` to ``check_tol``.
        -> observation [B, D]."""
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
        self._rec_init()
        return self.obs

    # ---- observation ---------------------------------------------------------------------------------------------
    def _observe(self, F=None, idx=None):
        """Observation of all worlds (F: task features of self.ws, else computed) or of the rows idx."""
        ws = self.ws if idx is None else self.ws.index(idx)
        if F is None:
            F = T.features(ws, self.kin, None, self.tcfg)
        sel = (lambda x: x) if idx is None else (lambda x: x[idx])
        obs = T.observation(F, sel(self.task.goal_xy), sel(self.task.goal_yaw), sel(self.task.tr.lifted),
                            sel(self.task.tr.air_run), self.tcfg, sel(self.dec.observation()), sel(self.dec.macro()))
        nv, nw = self.cfg.obs_rate_noise
        if (nv > 0 or nw > 0) and self.tcfg.obs_rates:
            n = obs.shape[0]
            obs[:, self.layout["cube_v"]] += float(nv) * torch.randn(n, 3, generator=self.gen, device=self.dev)
            obs[:, self.layout["cube_w"]] += float(nw) * torch.randn(n, 3, generator=self.gen, device=self.dev)
        return obs

    def get_observations(self):
        return self.obs

    # ---- one decision --------------------------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, actions, squashed=False):
        """actions [B, 6] (pre-tanh; squashed=True: already in [-1, 1]) -> (obs [B, D] float32, reward [B] float32,
        done [B] bool, extras): time_outs [B] bool, final_obs [B, D], action_mask [B, 6] bool, episode (dict of [B]
        tensors, valid where done: end code, success, success10, lifted, end_xy, end_yaw (m, rad), ret (decision
        reward sum), length (decisions), steps (20 ms steps)), world_steps (int)."""
        dec, task, B = self.dec, self.task, self.B
        live0 = task.tr.end == 0                      # running at the decision start
        a = actions.to(device=self.dev, dtype=torch.float64)
        a = a.clamp(-1.0, 1.0) if squashed else torch.tanh(a)
        blocked = torch.zeros(B, dtype=torch.bool, device=self.dev)
        held = torch.zeros(B, dtype=torch.bool, device=self.dev)
        if self.cfg.grasp_assist:
            blocked = self._assist(a) & live0
            held = (dec.mode == M["g_brake"]) | (dec.mode == M["g_descend"]) | (dec.mode == M["g_settle"])
            a = a.clone()
            a[:, 4] = torch.where(blocked, -1.0, torch.where(held, 1.0, a[:, 4]))   # no GRASP abort while assisted
        mode0 = dec.mode
        free0, hold0 = mode0 == M["free"], mode0 == M["hold"]
        pre0 = (mode0 == M["g_brake"]) | (mode0 == M["g_descend"]) | (mode0 == M["g_settle"])
        carry = bool(getattr(dec.cfg, "carry_macro", False))       # decoder option b2 (CARRY macro in mode hold)
        carry_on0 = dec.carry_on.clone() if carry else torch.zeros_like(hold0)
        moving0 = hold0 & (carry_on0 | dec.carry_brk) if carry else torch.zeros_like(hold0)
        dinfo = dec.decide(a, self.ws)
        started = dinfo["grasp_started"] | dinfo["place_started"]
        take_free = (free0 | dinfo["aborted"]) & ~started
        take_hold = hold0 & ~moving0 & ~started
        move = take_free | take_hold                   # x, y, yaw (CARRY: the line's end and turn)
        move_z = take_free | (take_hold if not carry else torch.zeros_like(take_hold))
        mask = torch.stack((move, move, move_z, move, free0 | hold0 | (pre0 & ~held),
                            dinfo["grasp_started"] | (hold0 & carry_on0)), -1)    # a[5]: lift height / CARRY abort
        rew = torch.zeros(B, dtype=torch.float64, device=self.dev)
        info = None
        for _ in range(self.steps_per_decision):
            u, _ = dec.step(self.ws.q)
            ws = self.world.step(u)
            self.ws = ws if self.cfg.world != "mujoco" else ws.to(self.dev)
            r, _, _, info = task.step(self.ws, dec.macro(), dec.macro_goal)
            rew += r
            if self.rec is not None:
                self._rec_step(u, info)
        if self.cfg.grasp_assist:
            rew = rew - self.cfg.grasp_assist_cost * blocked.to(rew.dtype)
        self.world_steps += B * self.steps_per_decision
        self.decisions += B
        end = info["end"]
        done = live0 & (end != 0)                     # ended in this decision
        time_out = done & (end == T.EC["timeout"])
        self.ep_ret += rew
        self.ep_len += live0.long()
        ep = dict(end=end.clone(), success=end == T.EC["success"], success10=info["success10"].clone(),
                  lifted=info["lifted"].clone(), end_xy=info["end_xy_err"].clone(), end_yaw=info["end_yaw_err"].clone(),
                  ret=self.ep_ret.clone(), length=self.ep_len.clone(), steps=info["t"].clone())
        obs = self._observe(task.F)
        final_obs = obs
        if self.cfg.auto_reset:
            idx = done.nonzero().flatten()
            if len(idx):
                obs = obs.clone()
                self._pool_reset(idx)
                obs[idx] = self._observe(idx=idx)
        self.obs = obs
        counts = dict(assist_blocked=blocked, assist_held=held & live0, grasp_started=dinfo["grasp_started"] & live0,
                      place_started=dinfo["place_started"] & live0, aborted=dinfo["aborted"] & live0)
        extras = dict(time_outs=time_out, final_obs=final_obs, action_mask=mask, episode=ep, counts=counts,
                      world_steps=B * self.steps_per_decision)
        return obs, rew.float(), done, extras

    # ---- per-step records (evaluation) -------------------------------------------------------------------------
    def _rec_init(self):
        if not self.record:
            return
        B, dev = self.B, self.dev
        self.rec = dict(q=[], cube=[], xg=[], u=[], mode=[], macro=[])
        self.rec_first_fire = torch.full((B, len(T.RULES)), -1, dtype=torch.long, device=dev)
        self.rec_lift_step = torch.full((B,), -1, dtype=torch.long, device=dev)
        self.rec_close_x = torch.full((B, 4), math.nan, dtype=torch.float64, device=dev)
        self.rec_mj_reason = torch.zeros(B, dtype=torch.long)
        self.rec_t = 0

    def _rec_step(self, u, info):
        t = self.rec_t
        live_before = info["t"] == t + 1
        F = self.task.F
        self.rec_first_fire = torch.where((self.rec_first_fire < 0) & info["flags"] & live_before[:, None], t,
                                          self.rec_first_fire)
        self.rec_lift_step = torch.where((self.rec_lift_step < 0) & info["lifted"], t, self.rec_lift_step)
        macro = self.dec.macro()
        cstart = (macro == T.MC["grasp_close"]) & torch.isnan(self.rec_close_x[:, 0]) & live_before
        self.rec_close_x = torch.where(cstart[:, None], torch.cat((F["xg"], F["yaw_mis"][:, None]), -1), self.rec_close_x)
        if self.cfg.world == "mujoco":
            new_end = (live_before & (info["end"] != 0)).cpu()
            self.rec_mj_reason = torch.where(new_end, self.world.end_reason.long(), self.rec_mj_reason)
        ws = self.ws
        self.rec["q"].append(ws.q.float().cpu())
        self.rec["cube"].append(torch.cat((ws.cube_pos, ws.cube_quat), -1).float().cpu())
        self.rec["xg"].append(F["xg"].float().cpu())
        self.rec["u"].append(u.float().cpu())
        self.rec["mode"].append(self.dec.mode.to(torch.int8).cpu())
        self.rec["macro"].append(macro.to(torch.int8).cpu())
        self.rec_t += 1

    def records(self) -> dict:
        """Per-step arrays [B, T, ...] (numpy) and per-episode markers of a recorded run (gate_expert run format)."""
        out = {k: torch.stack(v, 1).numpy() for k, v in self.rec.items()}
        out.update(first_fire=self.rec_first_fire.cpu().numpy(), lift_step=self.rec_lift_step.cpu().numpy(),
                   close_x=self.rec_close_x.cpu().numpy(), mj_reason=self.rec_mj_reason.numpy())
        return out

    def close(self):
        if hasattr(self.world, "close"):
            self.world.close()


# ---- fixed-task episodes (validation, evaluation) ---------------------------------------------------------------------
def policy_act_fn(policy):
    """A ``ppo.Policy`` -> act_fn: deterministic pre-tanh mean (the env applies tanh)."""
    def act(obs, env):
        return policy.act_inference(obs.to(next(policy.parameters()).device)).to(env.dev), False
    return act


@torch.no_grad()
def run_episodes(env: PickPlaceVecEnv, act_fn, starts: WorldState, goal_xy, goal_yaw, seeds, mj_start=None,
                 reset_hook=None, log=None) -> dict:
    """One episode per world on fixed tasks (env.num_envs of them; auto_reset must be off): act_fn(obs, env) ->
    (actions [B, 6], squashed) once per decision until every world has ended (the task horizon ends the rest).
    reset_hook(env): called after the reset (e.g. the expert's own reset). -> dict of numpy arrays [B]: end, success,
    success10, lifted, end_xy, end_yaw, ret, length (decisions), steps (20 ms steps, = task t at the end), seeds."""
    import numpy as np
    if env.cfg.auto_reset:
        raise ValueError("run_episodes needs an env with auto_reset off")
    obs = env.reset_tasks(torch.arange(env.B, device=env.dev), starts, goal_xy, goal_yaw, seeds, mj_start)
    if reset_hook is not None:
        reset_hook(env)
    keep = None
    for k in range(env.max_episode_length + 1):
        a, sq = act_fn(obs, env)
        obs, _, done, ex = env.step(a, squashed=sq)
        ep = ex["episode"]
        if keep is None:
            keep = {n: v.clone() for n, v in ep.items()}
        else:
            for n, v in ep.items():
                keep[n] = torch.where(done, v, keep[n])
        if log is not None and k % 20 == 19:
            log(f"  decision {k + 1}: ended {int((env.task.tr.end != 0).sum())}/{env.B}")
        if bool((env.task.tr.end != 0).all()):
            break
    out = {n: v.cpu().numpy() for n, v in keep.items()}
    out["seeds"] = np.asarray(torch.as_tensor(seeds).cpu())
    return out
