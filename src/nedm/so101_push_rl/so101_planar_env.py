"""SO-101 planar pusher in the frozen NRD (spec SPEC_lead_v1.md, part A): the policy moves the fingertip in the table
plane at the start context's push height and turns the T to a goal pose (large yaw changes).

RSL-RL 2.2.4 VecEnv. One env step = one decision = decision_steps (5) NRD steps of 20 ms (0.1 s). Actions a [3] (raw
latent, PPO keeps a): so101_planar_common.PlanarActionMap turns a into a planar displacement d (|d| <= 0.030 m, T frame at
decision time, rotated to the world) and a gripper yaw change dyaw (<= 0.3 rad); so101_planar_decoder.PlanarDecoder
blends the target velocity to d / 0.1 s (cosine blend), and every 20 ms makes q_cmd with the collector IK and command
law q_cmd = q_des + (g(q_des) + kd qdot_des) / kp, clipped to the joint limits and the recorded lead band around the
current joint angles. Yaw only away from the T: where the dense fingertip-T signed distance of the state at decision
time is below decoder.yaw_free_gap_m, the decision's yaw rate is 0 (decoder flag yaw_blocked).

Tasks (scripts/preprocess/make_so101_rotation_tasks.py): a recorded rest start (bank episode, frame j_a: history = the
recorded states j_a-3..j_a, previous command = the recorded command of step j_a-1, decoder target at rest at the start
file's p_line (x, y, push height z) with the start's gripper yaw and q_des), a goal T pose (x, y, yaw), a horizon (NRD
steps) and a level. Every task must start at frame j_a (checked at load).

Inner loop per decision (alive mask: an env that ends inside the decision is frozen for the remaining steps):
  q_cmd = decoder.step(q of the current state) -> NRD step (one of the two seeds per episode) -> geometry
  (so101_direct_common.DirectGeometry, dense fingertip points) -> validity -> reward -> success / deadline.
Reward per decision (gamma 1; sum over the NRD steps of the decision):
  progress  D_before - D_after, D = |xy - goal| / pos_scale + yaw_weight |wrap(yaw - goal yaw)| / yaw_scale_deg
  time      - time_cost per NRD step
  success   + success_bonus once (T inside pos_tol / yaw_tol and slow: planar speed < v_tol, |yaw rate| < w_tol_deg, for
            hold_steps consecutive NRD steps, no invalid event); the episode ends. success.per_level = {"<level>":
            {"pos_tol": m, "yaw_tol_deg": d}} (default None) sets the tolerances of the listed levels (per-env tensors
            tol_pos / tol_yaw_deg, set at reset and at a restore); other levels use the stage's pos_tol / yaw_tol_deg
  invalid   invalid_reward, and with charge_remaining also - time_cost x (horizon - t) (without drift, an invalid end is
            never cheaper than continuing); the episode ends; the progress of the invalid step is not paid. With
            drift_on_invalid (default false) also - drift_cost max(0, pe_last - free) / pos_scale x (horizon - t): the
            drift of the remaining steps at the error of the last valid state (pe_last = the state the progress term was
            last paid on; the invalid state itself may be non-finite or a kicked T), so the drift cost cannot make an
            invalid end cheaper than staying
  smooth    - w_smooth |d_t - d_{t-1}|^2 / d_max_m^2 per decision (d in the world frame, after the action map)
  drift     - drift_cost max(0, |xy - goal| - free) / pos_scale per NRD step (default 0 = off; 10-09: the +50 deg
            counter-clockwise turn left the T ~20 mm off and the telescoping progress term pays only the end state).
            drift_mode "abs" (default): free = drift_free_m; "regress": free = best_pe + drift_free_m, best_pe = the
            smallest position error of the episode's valid states so far (set to the current error at reset and at a
            restore), so a translation goal pays only for moving the T back away from the goal (in-place goals: = "abs")
  refused   - refuse_cost per decision in which the workspace clip, the joint-speed guard or ik_err fired
  contact   + contact_bonus once per contact run whose T displacement from the run start reaches contact_bonus_move_m
            while the run lasts (the finger touches at this step or at the step before), at most contact_bonus_max per
            episode (a contact run = consecutive NRD steps with fingertip-T signed distance <= touch_m). T motion after
            the finger has left (a coasting or kicked T, where the frozen NRDs are wrong) earns no bonus.
  coverage  (optional) w_coverage (coverage after - coverage before) per NRD step (telescoping; 0 = log only)
Deadline (t >= horizon): the episode ends, no bonus, no bootstrap (time_outs = 0).
Log only: /support/pen_log_step_share = share of the live NRD steps with the dense fingertip-T signed distance below
-log_penetration_m (spec A5 value 1.5 mm; the terminal penetration rule is loosened to 9 mm by the calibration, so this
shows whether the policy learns to push the finger deep into the NRD's T).
Validity (terminal): so101_planar_common.support_rules, every A5 rule on the NRD state and the FK geometry (non-finite,
joint limits, fingertip / link above the table, link-T gap, T height and tilt, gripper tilt, T speed and yaw rate, kick,
unexplained motion, penetration, keep-out; thresholds in the validity block, names of DEFAULT_VALIDITY). The env keeps
the ring buffers of the two window rules (so101_planar_common.SupportBuffers): the TCP planar speed of the last
kick_window_steps (5) NRD states and the dense fingertip-T signed distance of the last unexplained_window_steps (13)
states, current state last, filled with the start state's values at reset, updated only while the env is alive.
Observation: so101_planar_common.PlanarObs (139 values): DirectObs (obs.delta_max) + decoder target, T at rest
(t_at_rest with obs.rest_v_mps / rest_w_radps), time since the last fingertip contact, wrapped goal yaw error / 0.5 rad,
coverage of the goal T (obs.coverage_step_m grid).
Config layout: configs/rl/so101_planar/env_v1.json (action, decoder, obs, reward, success, validity, touch_m,
log_penetration_m).
Levels: every level id of level_weights must have rows in the tasks file (check_levels; ValueError at construction and at
set_level_weights). Logs per level of the tasks file: /level/<level>/success and /level/<level>/invalid = EMA (0.95) of
the share of the finished episodes of that level (NaN until the first one; reset when the level's weight becomes 0).

Snapshot / restore: snapshot(ids) -> dict "env.<attr>[.<key>]" / "dec.<attr>[.<key>]" / "buf.<attr>" -> tensor [len(ids),
...] (clones) of every per-env tensor of the env, its decoder and its support buffers (SNAP_ENV, SNAP_DECODER,
SNAP_BUFFERS); restore(ids, snap, rows) writes rows of a snapshot into env rows ids (out of place). Every attribute of the
three objects that holds a tensor or an array must be listed in SNAP_*, DERIVED_ENV or SHARED_* (checked at run time: a
new per-env attribute cannot be left out silently). DERIVED_ENV tensors (per-level success tolerances, best_pe) are not
in the snapshot (bank files of older code stay valid): restore sets them from the restored level / state and the current
config. A restored env continues bitwise like the source env when the batch holds the same states
(tests/rl/check_so101_planar_reset_bank.py (a); with drift_mode "regress" best_pe restarts at the restored error).
Reset bank (reverse curriculum; option reset_bank = {"path": npz of scripts/rot/make_reset_bank.py, "p": 0.5, "levels":
[9, 10], "min_left_decisions": 10, "skip_inside": true, "skip_doomed": true}, default None = off; unknown keys raise): in
training resets (reset_idx without task_ids; validation passes task_ids and is never affected), an env whose drawn task
level is in levels starts with probability p from a random bank entry of the same task (restore; the entry keeps its NRD
model, episode time t, success counter, contact runs and decoder state). Eligible entries (recomputed at every set_cfg):
remaining horizon (task horizon with horizon_min_steps) - t > min_left_decisions decisions, (skip_inside) T not already
inside the current success position / yaw tolerance, and (skip_doomed) not flagged hold_invalid by the builder. The file
is reloaded when its path, size or mtime changes. A restored episode: progress and coverage potentials recomputed for the
current config (no reward jump at the restore), horizon from the current config, from_bank True, episode returns, flags
and length counted from 0. Logs /episode/from_bank (finished episodes, EMA) and /reset/from_bank (share of the training
resets on the bank levels that came from the bank since the last set_cfg).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from rsl_rl.env import VecEnv

from nedm.contact_nrd.model import load_model
from nedm.so101_push_rl import so101_direct_common as D
from nedm.so101_push_rl import so101_planar_common as P
from nedm.so101_push_rl import so101_push_common as C
from nedm.so101_push_rl.so101_fk import ArmFK
from nedm.so101_push_rl.so101_goal_common import goal_error, potential, radians, settled_inside
from nedm.so101_push_rl.so101_planar_decoder import DEFAULT_CFG as DEC_DEFAULT
from nedm.so101_push_rl.so101_planar_decoder import STEP_FLAGS as DEC_STEP_FLAGS
from nedm.so101_push_rl.so101_planar_decoder import PlanarDecoder

# command lead band q_cmd - q (p0.5 / p99.5 over all phases, RL-train; direct study V10 / V14 / V39)
LEAD_LO = [-0.069, -0.101, -0.109, -0.086, -0.352]
LEAD_HI = [0.071, 0.077, 0.093, 0.122, 0.352]
STEP_FLAGS = DEC_STEP_FLAGS

# Per-env state copied by snapshot / restore (tensors or dicts of tensors with leading dimension num_envs). Every attribute
# of the env, the decoder and the support buffers that holds a tensor / array must be in one SNAP_* or SHARED_* tuple
# (SO101PlanarEnv._check_snap_spec raises otherwise).
SNAP_ENV = ("hist", "ok", "prev_cmd", "prev_inc", "prev_d", "task", "level", "goal", "horizon", "t", "model_id", "pot", "inside",
            "touched", "touching", "t_since_contact", "run_xy", "run_paid", "n_runs", "n_paid", "cov", "episode_length_buf",
            "from_bank", "obs_buf", "ep", "ep_flags", "geo", "lead_run")
SNAP_DECODER = ("p", "z", "yaw", "v0", "vn", "w0", "wn", "q_des_prev", "q_des_cur", "k", "k_dec", "counts")
SNAP_BUFFERS = ("tcp_speed", "finger_sd")
SNAP_OBJECTS = ("decoder", "buf")          # env attributes whose per-env state is listed above
# per-env tensors that are not snapshotted: restore() and reset_idx() set them from the level, the state and the config
DERIVED_ENV = ("tol_pos", "tol_yaw_deg", "best_pe")
# tensor-holding attributes that are not per-env state: models, kinematics, loaded data, constants, logging, step output
SHARED_ENV = ("models", "kin", "fk", "geom", "t_centres", "t_halves", "hist_norm", "stats_amap", "bank_states", "bank_cmds",
              "bank_first", "st_bank", "st_p_line", "st_yaw", "st_qdes_prev", "st_qdes_cur", "st_ja", "st_face", "tk_start",
              "tk_frame", "tk_goal", "tk_horizon", "tk_level", "task_w", "delta_max", "obs_builder", "amap", "done_stats",
              "extras", "rb", "rb_counts", "log_lv", "level_stats")
SHARED_DECODER = ("kin", "dq_max", "lead", "axis")
RESET_BANK_KEYS = ("path", "p", "levels", "min_left_decisions", "skip_inside", "skip_doomed")
SHARED_BUFFERS = ()
DRIFT_MODES = ("abs", "regress")
PER_LEVEL_KEYS = ("pos_tol", "yaw_tol_deg")


def _holds_tensor(v, depth=3):
    """True if v is (or holds, up to `depth` levels of dict / list / object attributes) a tensor, a module or an array."""
    if torch.is_tensor(v) or isinstance(v, (torch.nn.Module, np.ndarray)):
        return True
    if depth <= 0 or isinstance(v, (str, bytes, int, float, bool, type, torch.device, torch.dtype)) or v is None:
        return False
    if isinstance(v, dict):
        return any(_holds_tensor(x, depth - 1) for x in v.values())
    if isinstance(v, (list, tuple, set)):
        return any(_holds_tensor(x, depth - 1) for x in v)
    if hasattr(v, "__dict__"):
        return any(_holds_tensor(x, depth - 1) for x in vars(v).values())
    return False


def check_levels(task_levels, wanted, what="the tasks file"):
    """Raise ValueError if a level id of `wanted` has no row in task_levels (the level column of a tasks npz, or the
    npz path). A level id that is missing (e.g. a typo in a curriculum) would otherwise get no episodes."""
    if isinstance(task_levels, (str, Path)):
        what = str(task_levels)
        with np.load(task_levels) as T:
            task_levels = T["level"]
    have = set(np.asarray(task_levels.cpu() if torch.is_tensor(task_levels) else task_levels).astype(int).tolist())
    missing = sorted({int(lv) for lv in wanted} - have)
    if missing:
        raise ValueError(f"levels {missing} have no task in {what} (levels there: {sorted(have)})")


def level_tols(sc):
    """success block -> {level: (pos_tol m, yaw_tol_deg)} of success.per_level (levels not listed use the block's
    pos_tol / yaw_tol_deg; a listed level may set one of the two). Unknown keys raise."""
    out = {}
    for lv, d in (sc.get("per_level") or {}).items():
        unknown = sorted(set(d) - set(PER_LEVEL_KEYS))
        if unknown:
            raise ValueError(f"success.per_level[{lv}]: unknown keys {unknown} (known: {list(PER_LEVEL_KEYS)})")
        out[int(lv)] = (float(d.get("pos_tol", sc["pos_tol"])), float(d.get("yaw_tol_deg", sc["yaw_tol_deg"])))
    return out


def default_cfg():
    """Defaults; configs/rl/so101_planar/env_v1.json has the same layout (action + decoder = PlanarDecoder cfg)."""
    dec = DEC_DEFAULT
    return dict(
        num_envs=4096, device="cuda", seed=1, checkpoints=[], starts_path="", banks_path="", tasks_path="", stats_path="",
        collector_config="", dense_points="",
        action=dict(d_max_m=dec["d_max_m"], dyaw_max_rad=dec["dyaw_max_rad"], yaw_control=dec["yaw_control"], frame=dec["frame"]),
        # so101_planar_decoder.DEFAULT_CFG with the recorded lead band on (q_cmd - q, p0.5 / p99.5, RL-train)
        decoder=dict({k: dec[k] for k in ("decision_steps", "ctrl", "v_max", "ws_r_min", "ws_r_max", "ws_az_deg", "dq_max_rad",
                                          "ik_iters", "ik_err_flag_m", "yaw_free_gap_m")}, lead_lo=LEAD_LO, lead_hi=LEAD_HI),
        # delta_max: the direct env's scale of the previous command change (DirectObs [101:106)); T at rest: planar speed
        # < rest_v_mps and |yaw rate| < rest_w_radps; coverage grid step
        obs=dict(delta_max=[0.082, 0.068, 0.071, 0.079, 0.103], rest_v_mps=0.002, rest_w_radps=0.02, coverage_step_m=0.005),
        reward=dict(yaw_weight=1.0, pos_scale=0.010, yaw_scale_deg=5.0, time_cost=0.002, success_bonus=5.0, invalid_reward=-5.0,
                    charge_remaining=True, w_smooth=0.005, refuse_cost=0.05, contact_bonus=0.0, contact_bonus_move_m=0.003,
                    contact_bonus_max=3, w_coverage=0.0, drift_cost=0.0, drift_free_m=0.005, drift_mode="abs",
                    drift_on_invalid=False),
        # per_level: {"<level>": {"pos_tol", "yaw_tol_deg"}} (module docstring); None = the block's tolerances for every level
        success=dict(use_yaw=True, pos_tol=0.005, yaw_tol_deg=3.0, v_tol=0.005, w_tol_deg=5.0, hold_steps=5, per_level=None),
        # every A5 rule (so101_planar_common.support_rules; names and defaults there)
        validity=dict(P.DEFAULT_VALIDITY),
        # touch_m: fingertip contact = dense fingertip-T signed distance <= touch_m (contact runs, touched, contact timer).
        # 1.5 mm: the dense points are 2 mm apart, so a touching finger reads up to 1 mm (at 0.5 mm some recorded and
        # NRD pushes show no contact or split runs)
        touch_m=0.0015, level_weights=None, auto_reset=True,
        # log only (not a rule): share of NRD steps with penetration deeper than the spec A5 threshold
        log_penetration_m=0.0015,
        # reverse curriculum (module docstring): {"path", "p", "levels", "min_left_decisions", "skip_inside"}; None = off
        reset_bank=None,
    )


def merge(cfg):
    out = default_cfg()
    for k, v in (cfg or {}).items():
        out[k] = {**out[k], **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_stats(path, device):
    """Observation / action normalisation of the obs stats file (from so101_push_tracking_env, unchanged)."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    return C.ObsNorm.from_state_dict(d["obs_norm"]), C.ActionMap.from_state_dict(d["action_map"]).to(device), d


def _where(m, a, b):
    """torch.where with the [N] mask broadcast over the trailing dims of a / b."""
    return torch.where(m.view(-1, *([1] * (a.dim() - 1))), a, b)


class SO101PlanarEnv(VecEnv):
    def __init__(self, cfg=None, device=None):
        self.cfg = cfg = merge(cfg)
        self.device = dev = torch.device(device or cfg["device"])
        self.num_envs = N = int(cfg["num_envs"])
        self.num_actions = 3
        self.num_obs = P.OBS_DIM
        self.num_privileged_obs = None
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(int(cfg["seed"]))
        self.models = []
        for p in cfg["checkpoints"]:
            m, _ = load_model(Path(p) / "best.pt" if Path(p).is_dir() else p, dev)
            m.eval()
            for q in m.parameters():
                q.requires_grad_(False)
            assert abs(m.dt - 0.02) < 1e-12 and m.k == C.HISTORY and m.config["integrate"]["rule"] == "new"
            self.models.append(m)
        self.smax, self.amax = self.models[0].smax, self.models[0].amax
        from nedm.so101_push_rl.so101_kin_torch import So101Kin
        from nedm.so101_push.robot import load_config
        from nedm.so101_push.geometry import TShape
        ccfg = load_config(cfg["collector_config"])
        self.kin = So101Kin(ccfg, device=str(dev))
        self.fk = ArmFK(ccfg, TShape(ccfg["tshape"]), device=str(dev), dtype=torch.float64)
        if not cfg.get("dense_points"):
            raise ValueError("the planar env needs dense_points (fingertip-T signed distance of the support rules)")
        self.fk.load_dense(cfg["dense_points"])
        self.geom = D.DirectGeometry(self.kin, self.fk, dense=True)
        self.t_centres, self.t_halves = self.fk.t_centres[:, :2].to(torch.float64), self.fk.t_half[:, :2].to(torch.float64)
        self.hist_norm, self.stats_amap, _ = load_stats(cfg["stats_path"], dev)
        self._load()
        z = lambda *s, dt=torch.float64: torch.zeros(*s, dtype=dt, device=dev)
        self.hist = z(N, C.HISTORY, 2, self.smax)
        self.ok = z(N, C.HISTORY, dt=torch.bool)
        self.prev_cmd, self.prev_inc = z(N, 5), z(N, 5)
        self.prev_d = z(N, 2)
        self.task, self.level = z(N, dt=torch.long), z(N, dt=torch.long)
        self.goal = z(N, 3)
        self.horizon, self.t = z(N, dt=torch.long), z(N, dt=torch.long)
        self.model_id = z(N, dt=torch.long)
        self.pot = z(N)
        self.inside = z(N, dt=torch.long)
        self.tol_pos, self.tol_yaw_deg = z(N), z(N)    # success tolerances of the env's level (success.per_level)
        self.best_pe = z(N)                         # smallest position error of the episode so far (drift_mode "regress")
        self.lead_run = z(N, dt=torch.long)        # consecutive NRD steps with the command lead clipped (validity.lead_run_max)
        self.touched, self.touching = z(N, dt=torch.bool), z(N, dt=torch.bool)
        self.t_since_contact = z(N)
        self.run_xy, self.run_paid = z(N, 2), z(N, dt=torch.bool)
        self.n_runs, self.n_paid = z(N, dt=torch.long), z(N, dt=torch.long)
        self.cov = z(N)
        self.episode_length_buf = z(N, dt=torch.long)
        self.from_bank = z(N, dt=torch.bool)
        self.ep = {k: z(N) for k in ("reward", "progress", "smooth", "time", "bonus", "refused", "contact", "coverage", "invalid_r", "refusals",
                                     "drift")}
        self.ep_flags = {k: z(N) for k in STEP_FLAGS + ("yaw_blocked",)}
        self.done_stats = {}
        # per-level EMA of the finished episodes' success / invalid share (rows), levels of the tasks file (columns)
        self.log_lv = torch.as_tensor(self.levels, dtype=torch.long, device=dev)
        self.level_stats = torch.full((2, len(self.levels)), float("nan"), dtype=torch.float64, device=dev)
        self.geo = None
        self.extras = {}
        self.obs_buf = torch.zeros(N, P.OBS_DIM, device=dev)
        self.rb = None
        self._snap_sig = None
        self.set_cfg(cfg)
        self.set_level_weights(cfg.get("level_weights"))
        with torch.inference_mode():
            self.reset()

    def set_cfg(self, cfg):
        """(Re)build the parts that depend on the action / decoder / validity config (also at a curriculum stage change;
        a reset must follow)."""
        cfg = merge(cfg)
        if str(cfg["reward"].get("drift_mode") or "abs") not in DRIFT_MODES:
            raise ValueError(f"reward.drift_mode {cfg['reward'].get('drift_mode')!r} (known: {list(DRIFT_MODES)})")
        self.per_level = level_tols(cfg["success"])        # {level: (pos_tol, yaw_tol_deg)}
        self.cfg = cfg
        dev = self.device
        self.n_sub = int(cfg["decoder"]["decision_steps"])
        self.max_episode_length = int(-(-max(int(self.tk_horizon.max()), int(self.cfg.get("horizon_min_steps") or 0)) // self.n_sub))
        self.delta_max = torch.tensor(cfg["obs"]["delta_max"], dtype=torch.float64, device=dev)
        self.obs_builder = P.PlanarObs(self.hist_norm, self.stats_amap, self.delta_max)
        self.amap = P.PlanarActionMap(cfg["action"])
        self.dec_cfg = {**cfg["action"], **cfg["decoder"]}
        self.decoder = PlanarDecoder(self.kin, self.num_envs, dev, self.dec_cfg)
        self.buf = P.SupportBuffers(self.num_envs, cfg["validity"], dev)
        self.auto_reset = bool(cfg["auto_reset"])
        self._set_reset_bank(cfg.get("reset_bank"))

    # ------------------------------------------------------------------ data
    def _load(self):
        dev = self.device
        cfg = self.cfg
        S = np.load(cfg["starts_path"])
        B = np.load(cfg["banks_path"])
        T = np.load(cfg["tasks_path"])
        f = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)
        self.bank_states = f(B["states"])                       # [Nb, 201, 2, 13]
        self.bank_cmds = f(B["cmds"])                           # [Nb, 200, 5]
        self.bank_first = f(B["first_cmd"])
        self.st_bank = f(S["bank_index"], torch.long)
        self.st_p_line = f(S["p_line"])                         # [S, 3] (x, y, TCP push height)
        self.st_yaw = f(S["yaw"])                               # [S] gripper yaw
        self.st_qdes_prev, self.st_qdes_cur = f(S["q_des_prev"]), f(S["q_des_cur"])
        self.st_ja = f(S["j_a"], torch.long)
        self.st_face = f(S["face_id"], torch.long)
        self.tk_start = f(T["start_idx"], torch.long)
        self.tk_frame = f(T["start_frame"], torch.long)
        self.tk_goal = f(T["goal"])
        self.tk_horizon = f(T["horizon_steps"], torch.long)
        self.tk_level = f(T["level"], torch.long)
        if not bool((self.tk_frame == self.st_ja[self.tk_start]).all()):
            raise ValueError("planar env tasks must start at the rest frame j_a of their start")
        self.n_tasks = len(self.tk_start)
        self.levels = sorted(set(T["level"].tolist()))

    def set_level_weights(self, weights):
        """weights: dict level -> weight (curriculum); tasks are drawn uniformly inside a level. Every level id must have
        rows in the tasks file (ValueError otherwise)."""
        dev = self.device
        if not weights:
            self.task_w = torch.ones(self.n_tasks, dtype=torch.float64, device=dev)
        else:
            check_levels(self.tk_level, weights, f"{self.cfg['tasks_path']} (level_weights)")
            w = torch.zeros(self.n_tasks, dtype=torch.float64, device=dev)
            for lv, wl in weights.items():
                sel = self.tk_level == int(lv)
                if sel.any():
                    w[sel] = float(wl) / float(sel.sum())
            if not bool((w > 0).any()):
                raise ValueError(f"no task of levels {list(weights)} in {self.cfg['tasks_path']}")
            self.task_w = w
        self.cur_weights = weights
        # per-level logs: levels without weight now start again from NaN when they come back
        active = torch.as_tensor([bool((self.task_w[self.tk_level == lv] > 0).any()) for lv in self.levels], device=dev)
        self.level_stats = torch.where(active[None], self.level_stats, torch.full_like(self.level_stats, float("nan")))

    # ------------------------------------------------------------------ snapshot / restore
    def _check_snap_spec(self):
        """Every tensor-holding attribute of the env, its decoder and its support buffers must be classified (SNAP_* =
        per-env state, DERIVED_ENV = per-env, set by restore, SHARED_* = not per-env); the per-env ones must exist."""
        errs = []
        for name, obj, state, shared in (("env", self, SNAP_ENV + SNAP_OBJECTS + DERIVED_ENV, SHARED_ENV),
                                         ("decoder", self.decoder, SNAP_DECODER, SHARED_DECODER),
                                         ("support buffers", self.buf, SNAP_BUFFERS, SHARED_BUFFERS)):
            errs += [f"{name}.{a} (not classified)" for a, v in vars(obj).items()
                     if a not in state and a not in shared and _holds_tensor(v)]
            errs += [f"{name}.{a} (missing)" for a in state if not hasattr(obj, a)]
        if errs:
            raise AssertionError("snapshot spec of SO101PlanarEnv is out of date: " + ", ".join(errs)
                                 + "; add per-env state to SNAP_ENV / SNAP_DECODER / SNAP_BUFFERS / DERIVED_ENV, other data to SHARED_*")

    def _snap_fields(self):
        """[(key, holder, name, holder_is_dict)] of every per-env tensor (SNAP_*; dict attributes expanded per key)."""
        sig = (tuple(vars(self)), tuple(vars(self.decoder)), tuple(vars(self.buf)), id(self.decoder), id(self.buf))
        if sig != self._snap_sig:
            self._check_snap_spec()
            self._snap_sig = sig
        out = []
        for pre, obj, names in (("env", self, SNAP_ENV), ("dec", self.decoder, SNAP_DECODER), ("buf", self.buf, SNAP_BUFFERS)):
            for a in names:
                v = getattr(obj, a)
                items = [(f"{pre}.{a}.{k}", v, k, True, x) for k, x in v.items()] if isinstance(v, dict) else [(f"{pre}.{a}", obj, a, False, v)]
                for key, holder, nm, isd, x in items:
                    if not torch.is_tensor(x) or x.dim() == 0 or x.shape[0] != self.num_envs:
                        raise AssertionError(f"snapshot: {key} is not a per-env tensor (leading dimension {self.num_envs}): "
                                             f"{type(x).__name__} {tuple(x.shape) if torch.is_tensor(x) else ''}")
                    out.append((key, holder, nm, isd))
        return out

    def _ids_tensor(self, ids):
        if ids is None:
            return torch.arange(self.num_envs, device=self.device)
        ids = torch.as_tensor(ids, device=self.device)
        return ids.nonzero().flatten() if ids.dtype == torch.bool else ids.long().flatten()

    def snapshot(self, ids=None):
        """Copies of every per-env tensor of rows ids (None: all) -> dict key -> tensor [len(ids), ...] (env device)."""
        ids = self._ids_tensor(ids)
        out = {}
        for key, holder, nm, isd in self._snap_fields():
            v = holder[nm] if isd else getattr(holder, nm)
            out[key] = v[ids].clone()
        return out

    def restore(self, ids, snap, rows=None):
        """Write rows `rows` (None: 0..len(ids)-1) of a snapshot (dict of snapshot(); any device) into env rows ids. The
        keys must match this env's snapshot keys exactly. Out of place: tensors handed out earlier are not changed."""
        ids = self._ids_tensor(ids)
        fields = self._snap_fields()
        keys = {f[0] for f in fields}
        miss, extra = sorted(keys - set(snap)), sorted(set(snap) - keys)
        if miss or extra:
            raise KeyError(f"restore: snapshot keys do not match the env (missing {miss}, unknown {extra})")
        for key, holder, nm, isd in fields:
            src = snap[key]
            if rows is not None:
                src = src[torch.as_tensor(rows, device=src.device).long()]
            cur = holder[nm] if isd else getattr(holder, nm)
            if src.shape[0] != ids.numel() or tuple(src.shape[1:]) != tuple(cur.shape[1:]) or src.dtype != cur.dtype:
                raise ValueError(f"restore: {key} has shape {tuple(src.shape)} {src.dtype}, the env needs "
                                 f"[{ids.numel()}, {', '.join(map(str, cur.shape[1:]))}] {cur.dtype}")
            new = cur.clone()
            new[ids] = src.to(self.device)
            if isd:
                holder[nm] = new
            else:
                setattr(holder, nm, new)
        self._set_derived(ids, C.flat23(self.hist[ids, -1]))

    def _level_tols(self, level):
        """Success tolerances (pos_tol m, yaw_tol_deg) [n] float64 of levels [n]: success.per_level, else the stage's."""
        sc = self.cfg["success"]
        pt = torch.full(level.shape, float(sc["pos_tol"]), dtype=torch.float64, device=self.device)
        yt = torch.full(level.shape, float(sc["yaw_tol_deg"]), dtype=torch.float64, device=self.device)
        for lv, (p, y) in self.per_level.items():
            m = level == lv
            pt, yt = pt.masked_fill(m, p), yt.masked_fill(m, y)
        return pt, yt

    def _set_derived(self, ids, cur):
        """DERIVED_ENV of envs ids (current states cur [n, 23]): the level's success tolerances under the current config,
        best_pe = the current position error (out of place, like restore)."""
        pt, yt = self._level_tols(self.level[ids])
        for name, v in (("tol_pos", pt), ("tol_yaw_deg", yt), ("best_pe", goal_error(cur, self.goal[ids])[0])):
            new = getattr(self, name).clone()
            new[ids] = v
            setattr(self, name, new)

    # ------------------------------------------------------------------ reset bank (reverse curriculum)
    def _set_reset_bank(self, rb):
        """(Re)load the reset bank of cfg["reset_bank"] (the file only when its path, size or mtime changes) and recompute
        the eligible entries for the current config (module docstring)."""
        dev = self.device
        self.rb_counts = torch.zeros(2, dtype=torch.float64, device=dev)       # resets on the bank levels, from the bank
        if rb:
            unknown = sorted(set(rb) - set(RESET_BANK_KEYS))
            if unknown:
                raise ValueError(f"reset_bank: unknown keys {unknown} (known: {list(RESET_BANK_KEYS)})")
        if not rb or not rb.get("path") or float(rb.get("p", 0.5)) <= 0.0:
            self.rb = None
            return
        path = str(rb["path"])
        st = Path(path).stat()
        ident = (path, st.st_size, st.st_mtime_ns)          # a file rebuilt at the same path is reloaded
        if self.rb is not None and self.rb["ident"] == ident:
            data = self.rb["data"]
        else:
            Z = np.load(path, allow_pickle=False)
            data = dict(snap={k[5:]: torch.as_tensor(Z[k], device=dev) for k in Z.files if k.startswith("snap/")},
                        task=torch.as_tensor(Z["task"], device=dev).long(), t=torch.as_tensor(Z["t"], device=dev).long(),
                        level=torch.as_tensor(Z["level"], device=dev).long(),
                        doomed=(torch.as_tensor(Z["hold_invalid"], device=dev).bool() if "hold_invalid" in Z.files
                                else torch.zeros(len(Z["task"]), dtype=torch.bool, device=dev)))
            task = data["task"]
            if (len(task) == 0 or int(task.max()) >= self.n_tasks
                    or not torch.equal(self.tk_start[task], torch.as_tensor(Z["task_start"], device=dev).long())
                    or not torch.equal(self.tk_level[task], data["level"])
                    or not torch.allclose(self.tk_goal[task], torch.as_tensor(Z["task_goal"], device=dev, dtype=torch.float64), rtol=0, atol=1e-9)):
                raise ValueError(f"reset bank {path} is empty or was built for another tasks file than {self.cfg['tasks_path']}")
            for k in ("env.task", "env.t"):
                if not torch.equal(data["snap"][k].long(), data["task" if k == "env.task" else "t"]):
                    raise ValueError(f"reset bank {path}: {k} differs from the entry table")
            if int(data["snap"]["env.model_id"].max()) >= len(self.models):
                raise ValueError(f"reset bank {path} uses NRD model ids up to {int(data['snap']['env.model_id'].max())}, "
                                 f"the env has {len(self.models)} models")
            # the snapshot is only consistent with the dynamics / rule / observation config it was made with
            if "env_cfg" in Z.files:
                bc = merge(json.loads(str(Z["env_cfg"])))
                diff = [f"{b}.{k}" for b in ("action", "decoder", "obs", "validity") for k in sorted(set(bc[b]) | set(self.cfg[b]))
                        if bc[b].get(k) != self.cfg[b].get(k)]
                diff += [k for k in ("touch_m",) if bc[k] != self.cfg[k]]
                if [Path(p).name for p in bc["checkpoints"]] != [Path(p).name for p in self.cfg["checkpoints"]]:
                    diff.append("checkpoints")
                if diff:
                    print(f"[reset_bank] WARNING {path} was made with another env config: {', '.join(diff)}", flush=True)
        task, t, level, snap = data["task"], data["t"], data["level"], data["snap"]
        levels = sorted(int(x) for x in (rb.get("levels") or torch.unique(level).tolist()))
        lv_t = torch.as_tensor(levels, device=dev, dtype=torch.long)
        hms = int(self.cfg.get("horizon_min_steps") or 0)
        left = torch.clamp(self.tk_horizon[task], min=hms) - t
        min_left = int(rb.get("min_left_decisions", 10)) * self.n_sub
        sc = self.cfg["success"]
        pe, ye = goal_error(C.flat23(snap["env.hist"][:, -1]), self.tk_goal[task])
        pt, yt = self._level_tols(level)
        inside = pe < pt
        if sc["use_yaw"]:
            inside = inside & (ye.abs() < radians(yt))
        skip = inside if rb.get("skip_inside", True) else torch.zeros_like(inside)
        if rb.get("skip_doomed", True):                     # entries flagged by the builder's look-ahead (--keep-doomed)
            skip = skip | data["doomed"]
        elig = torch.isin(level, lv_t) & (left > min_left) & ~skip
        order = elig.nonzero().flatten()
        order = order[torch.argsort(task[order], stable=True)]
        count = torch.bincount(task[order], minlength=self.n_tasks)
        self.rb = dict(path=path, ident=ident, data=data, p=float(rb.get("p", 0.5)), levels=lv_t, order=order, count=count,
                       offset=torch.cumsum(count, 0) - count)
        print(f"[reset_bank] {path}: {len(task)} entries, {int(elig.sum())} eligible (levels {levels}, p {self.rb['p']}, "
              f"> {min_left} NRD steps left: {int((left > min_left).sum())}, inside the success tolerance: {int(inside.sum())}"
              f"{' skipped' if rb.get('skip_inside', True) else ' kept'})", flush=True)

    def _reset_from_bank(self, ids, tid):
        """Training reset: envs ids (fresh starts of tasks tid) of the bank levels start from a bank entry of the same
        task with probability p."""
        rb, dev, n = self.rb, self.device, len(ids)
        u = torch.rand(n, generator=self.gen, device=dev, dtype=torch.float64)
        v = torch.rand(n, generator=self.gen, device=dev, dtype=torch.float64)
        cnt, lv_ok = rb["count"][tid], torch.isin(self.tk_level[tid], rb["levels"])
        use = lv_ok & (cnt > 0) & (u < rb["p"])
        self.rb_counts += torch.stack((lv_ok.sum(), use.sum())).to(torch.float64)
        sel = use.nonzero().flatten()
        if len(sel) == 0:
            return
        j = torch.minimum((v[sel] * cnt[sel]).long(), cnt[sel] - 1)
        rows = rb["order"][rb["offset"][tid[sel]] + j]
        i2 = ids[sel]
        self.restore(i2, rb["data"]["snap"], rows)
        # horizon of the current config; returns, flags and length of the restored episode count from 0
        self.horizon[i2] = torch.clamp(self.tk_horizon[tid[sel]], min=int(self.cfg.get("horizon_min_steps") or 0))
        self.from_bank[i2] = True
        self.episode_length_buf[i2] = 0
        for x in list(self.ep.values()) + list(self.ep_flags.values()):
            x[i2] = 0.0
        # reward potentials (progress, coverage) and success counter of the restored state under the current reward /
        # success / obs config (equal to the snapshot values when the bank was made with the same config): the first
        # decision pays only the change from the restored state (restore set the level's tolerances and best_pe)
        cur = C.flat23(self.hist[i2, -1])
        rc, sc = self.cfg["reward"], self.cfg["success"]
        self.pot[i2] = potential(cur, self.goal[i2], rc["yaw_weight"], rc["pos_scale"], rc["yaw_scale_deg"])
        self.cov[i2] = self._coverage(cur, self.goal[i2])
        ins = settled_inside(cur, self.goal[i2], sc["use_yaw"], self.tol_pos[i2], self.tol_yaw_deg[i2], sc["v_tol"], sc["w_tol_deg"])
        self.inside[i2] = torch.where(ins, self.inside[i2], torch.zeros_like(self.inside[i2]))

    # ------------------------------------------------------------------ VecEnv
    @property
    def unwrapped(self):
        return self

    def get_observations(self):
        return self.obs_buf, {"observations": {"critic": self.obs_buf}}

    def get_privileged_observations(self):
        return None

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._compute_obs()
        return self.obs_buf, {"observations": {"critic": self.obs_buf}}

    def reset_idx(self, ids, task_ids=None, model_ids=None):
        n = len(ids)
        if n == 0:
            return
        dev = self.device
        tid = (torch.multinomial(self.task_w, n, replacement=True, generator=self.gen) if task_ids is None
               else torch.as_tensor(task_ids, device=dev).long())
        si = self.tk_start[tid]
        b = self.st_bank[si]
        f0 = self.tk_frame[tid]
        fr = f0[:, None] + torch.arange(-(C.HISTORY - 1), 1, device=dev)[None]
        ok = fr >= 0
        self.hist[ids] = self.bank_states[b[:, None], fr.clamp(min=0)] * ok[..., None, None]
        self.ok[ids] = ok
        c1 = torch.where((f0 >= 1)[:, None], self.bank_cmds[b, (f0 - 1).clamp(min=0)], self.bank_first[b])
        c2 = torch.where((f0 >= 2)[:, None], self.bank_cmds[b, (f0 - 2).clamp(min=0)], c1)
        self.prev_cmd[ids] = c1
        self.prev_inc[ids] = c1 - c2
        self.prev_d[ids] = 0.0
        self.task[ids] = tid
        self.level[ids] = self.tk_level[tid]
        self.goal[ids] = self.tk_goal[tid]
        # horizon_min_steps (stage option): episodes last at least this many NRD steps (e.g. 6 s for the 30 deg rungs)
        self.horizon[ids] = torch.clamp(self.tk_horizon[tid], min=int(self.cfg.get("horizon_min_steps") or 0))
        self.t[ids] = 0
        self.inside[ids] = 0
        self.lead_run[ids] = 0
        self.episode_length_buf[ids] = 0
        self.from_bank[ids] = False
        self.model_id[ids] = (torch.randint(0, len(self.models), (n,), generator=self.gen, device=dev) if model_ids is None
                              else torch.as_tensor(model_ids, device=dev).long())
        self.decoder.reset(ids, self.st_p_line[si], self.st_yaw[si], self.st_qdes_prev[si], self.st_qdes_cur[si], f0)
        cur = C.flat23(self.hist[ids, -1])
        g = self.geom(cur)
        rc = self.cfg["reward"]
        self.pot[ids] = potential(cur, self.goal[ids], rc["yaw_weight"], rc["pos_scale"], rc["yaw_scale_deg"])
        touch = g["finger_sd"] <= self.cfg["touch_m"]
        self.touched[ids] = touch
        self.touching[ids] = touch
        self.t_since_contact[ids] = torch.where(touch, torch.zeros_like(g["finger_sd"]), torch.full_like(g["finger_sd"], 10.0))
        self.run_xy[ids] = cur[:, 10:12]
        self.run_paid[ids] = touch
        self.n_runs[ids] = touch.long()
        self.n_paid[ids] = 0
        self._set_derived(ids, cur)
        self.buf.reset(ids, P.tcp_planar_speed(g), g["finger_sd"])
        self.cov[ids] = self._coverage(cur, self.goal[ids])
        for v in list(self.ep.values()) + list(self.ep_flags.values()):
            v[ids] = 0.0
        if self.geo is None:
            self.geo = {k: torch.zeros(self.num_envs, *v.shape[1:], dtype=v.dtype, device=dev) for k, v in g.items()}
        for k, v in g.items():
            self.geo[k][ids] = v
        if task_ids is None and self.rb is not None:      # training reset: reverse curriculum (validation never)
            self._reset_from_bank(torch.as_tensor(ids, device=dev).long(), tid)

    # ------------------------------------------------------------------ dynamics
    @torch.no_grad()
    def _model_step(self, cmd):
        act = torch.zeros(self.num_envs, 2, self.amax, dtype=torch.float64, device=self.device)
        act[:, 0, :5] = cmd
        nxt = torch.empty(self.num_envs, 2, self.smax, dtype=torch.float64, device=self.device)
        for i, m in enumerate(self.models):
            idx = torch.nonzero(self.model_id == i).flatten()
            if len(idx):
                nxt[idx] = m.details(self.hist[idx], self.ok[idx], act[idx])[0]
        return nxt

    def t_rest(self, s23):
        return P.t_at_rest(s23, self.cfg["obs"]["rest_v_mps"], self.cfg["obs"]["rest_w_radps"])

    def _coverage(self, s23, goal):
        return P.coverage(s23, goal, self.t_centres, self.t_halves, self.cfg["obs"]["coverage_step_m"]).to(torch.float64)

    def step(self, actions):
        a = actions.to(self.device).to(torch.float64)
        dev, N = self.device, self.num_envs
        cfg = self.cfg
        rc, sc, vc = cfg["reward"], cfg["success"], cfg["validity"]
        f64 = lambda x: x.to(torch.float64)
        # ---- decision
        d_world, dyaw, _ = self.amap(a, self.geo["t_yaw"])
        gap = float(self.dec_cfg.get("yaw_free_gap_m", 0.0) or 0.0)
        yaw_block = (self.geo["finger_sd"] < gap) if gap > 0 else None
        all_ids = torch.arange(N, device=dev)
        yb0 = self.decoder.counts["yaw_blocked"].clone() if "yaw_blocked" in self.decoder.counts else None
        self.decoder.decide(all_ids, d_world, dyaw, yaw_block)
        if yb0 is not None:
            self.ep_flags["yaw_blocked"] += f64(self.decoder.counts["yaw_blocked"] - yb0)
        smooth = -rc["w_smooth"] * ((d_world - self.prev_d) ** 2).sum(-1) / float(self.dec_cfg["d_max_m"]) ** 2
        self.prev_d = d_world
        # ---- inner loop
        alive = torch.ones(N, dtype=torch.bool, device=dev)
        zero = torch.zeros(N, dtype=torch.float64, device=dev)
        rew = zero.clone()
        parts = {k: zero.clone() for k in ("progress", "time", "bonus", "contact", "coverage", "invalid_r", "drift")}
        w_cov = float(rc.get("w_coverage", 0.0) or 0.0)
        w_drift, drift_free = float(rc.get("drift_cost", 0.0) or 0.0), float(rc.get("drift_free_m", 0.005))
        regress = str(rc.get("drift_mode") or "abs") == "regress"
        drift_inv = w_drift > 0 and bool(rc.get("drift_on_invalid", False))
        if drift_inv:
            parts["drift_invalid"] = zero.clone()            # log only (also inside invalid_r)
        flags = {k: torch.zeros(N, dtype=torch.bool, device=dev) for k in STEP_FLAGS}
        success, invalid, deadline = (torch.zeros(N, dtype=torch.bool, device=dev) for _ in range(3))
        reasons_any = {}
        pen_m = float(cfg.get("log_penetration_m", 0.0015))
        n_live, n_pen = zero.clone(), zero.clone()           # log only: live NRD steps, steps deeper than pen_m
        cmd_last = self.prev_cmd.clone()
        s = C.flat23(self.hist[:, -1])
        for _ in range(self.n_sub):
            cmd, info = self.decoder.step(self.geo["q"])
            for k in STEP_FLAGS:
                if k in info:
                    flags[k] |= info[k].bool() & alive
            nxt = self._model_step(cmd)
            self.hist = _where(alive, torch.cat((self.hist[:, 1:], nxt[:, None]), 1), self.hist)
            self.ok = _where(alive, torch.cat((self.ok[:, 1:], torch.ones_like(self.ok[:, :1])), 1), self.ok)
            s_new = C.flat23(nxt)
            g = self.geom(s_new)
            tcp_old, sd_old = self.buf.tcp_speed, self.buf.finger_sd
            self.buf.push(P.tcp_planar_speed(g), g["finger_sd"])
            self.buf.tcp_speed, self.buf.finger_sd = _where(alive, self.buf.tcp_speed, tcp_old), _where(alive, self.buf.finger_sd, sd_old)
            bad, reasons = P.support_rules(s_new, self.buf.bufs(), g, vc, self.kin)
            # sustained lead clip: the command ran at the edge of the recorded q_cmd - q band for lead_run_max NRD steps
            # in a row (the arm cannot follow; Chrono drops / tilts the gripper there, the NRD does not; 10-08 W1 analysis)
            # counted only at push height (TCP z < lead_run_z_m): in the collector data runs of >= 8 such steps occur in 0.5 %
            # of the episodes at TCP z < 25 mm (4.5 % of all steps are out of band, mostly fast air moves)
            lc = info.get("lead_clipped")
            if lc is not None:
                lc_low = lc.bool() & (g["tcp"][:, 2] < float(vc.get("lead_run_z_m", 0.025)))
                self.lead_run = torch.where(alive, torch.where(lc_low, self.lead_run + 1, torch.zeros_like(self.lead_run)), self.lead_run)
            lrm = vc.get("lead_run_max")
            if lrm:
                lr_bad = self.lead_run >= int(lrm)
                reasons = dict(reasons, lead_run=lr_bad)
                bad = bad | lr_bad
            n_live += f64(alive)
            n_pen += f64(alive & (g["finger_sd"] < -pen_m))
            for k, v in reasons.items():
                reasons_any[k] = reasons_any.get(k, torch.zeros_like(v)) | (v & alive)
            pot = potential(s_new, self.goal, rc["yaw_weight"], rc["pos_scale"], rc["yaw_scale_deg"])
            prog = self.pot - pot
            inside = settled_inside(s_new, self.goal, sc["use_yaw"], self.tol_pos, self.tol_yaw_deg, sc["v_tol"], sc["w_tol_deg"])
            self.inside = torch.where(alive, torch.where(inside, self.inside + 1, torch.zeros_like(self.inside)), self.inside)
            self.t = self.t + alive.long()
            inv_now = alive & bad
            succ_now = alive & ~bad & (self.inside >= sc["hold_steps"])
            dl = alive & ~inv_now & ~succ_now & (self.t >= self.horizon)
            # contact runs (fingertip-T signed distance <= touch_m) and the optional contact bonus
            touch = g["finger_sd"] <= cfg["touch_m"]
            new_run = alive & touch & ~self.touching
            self.run_xy = _where(new_run, s_new[:, 10:12], self.run_xy)
            self.run_paid = self.run_paid & ~new_run
            self.n_runs = self.n_runs + new_run.long()
            moved = (s_new[:, 10:12] - self.run_xy).norm(dim=-1) >= rc["contact_bonus_move_m"]
            in_run = touch | self.touching                       # touching now or at the step before (self.touching)
            pay = (alive & ~inv_now & in_run & (self.n_runs > 0) & ~self.run_paid & moved
                   & (self.n_paid < int(rc["contact_bonus_max"])))
            self.run_paid = self.run_paid | pay
            self.n_paid = self.n_paid + pay.long()
            cbonus = rc["contact_bonus"] * f64(pay)
            self.touching = torch.where(alive, touch, self.touching)
            self.touched = self.touched | (touch & alive)
            self.t_since_contact = torch.where(alive, torch.where(touch, zero, self.t_since_contact + 0.02), self.t_since_contact)
            # optional coverage reward: w_coverage (coverage after - coverage before), a telescoping term
            if w_cov > 0:
                cov_new = self._coverage(s_new, self.goal)
                rcov = torch.where(alive & ~inv_now, w_cov * (cov_new - self.cov), zero)
                self.cov = torch.where(alive, cov_new, self.cov)
            else:
                rcov = zero
            # optional drift cost: w_drift (T position error beyond drift_free_m) / pos_scale per NRD step, so a turn that
            # leaves the T off its goal position costs while it lasts, not only at the end (progress telescopes to the end state)
            if w_drift > 0:
                pe_now, _ = goal_error(s_new, self.goal)
                if regress:                    # beyond the best error of the episode's valid states (this one included)
                    self.best_pe = torch.where(alive & ~inv_now, torch.minimum(self.best_pe, pe_now), self.best_pe)
                    rdrift = torch.where(alive & ~inv_now, -w_drift * (pe_now - self.best_pe - drift_free).clamp(min=0.0) / rc["pos_scale"], zero)
                else:
                    rdrift = torch.where(alive & ~inv_now, -w_drift * (pe_now - drift_free).clamp(min=0.0) / rc["pos_scale"], zero)
            else:
                rdrift = zero
            # reward of this NRD step
            r = prog - rc["time_cost"] + rc["success_bonus"] * f64(succ_now) + cbonus + rcov + rdrift
            r_bad = torch.full_like(r, rc["invalid_reward"])
            if rc.get("charge_remaining", True):
                r_bad = r_bad - rc["time_cost"] * f64((self.horizon - self.t).clamp(min=0))
            if drift_inv:
                # drift of the remaining steps at the error of the last valid state s (best_pe excludes the invalid state)
                pe_last, _ = goal_error(s, self.goal)
                free = self.best_pe + drift_free if regress else drift_free
                d_inv = w_drift * (pe_last - free).clamp(min=0.0) / rc["pos_scale"] * f64((self.horizon - self.t).clamp(min=0))
                r_bad = r_bad - d_inv
                parts["drift_invalid"] -= torch.where(inv_now, d_inv, zero)
            r = torch.where(inv_now, r_bad, r)
            r = torch.where(alive, r, zero)
            rew += r
            ok_s = alive & ~inv_now
            parts["progress"] += torch.where(ok_s, prog, zero)
            parts["time"] -= rc["time_cost"] * f64(alive)
            parts["bonus"] += rc["success_bonus"] * f64(succ_now)
            parts["contact"] += cbonus
            parts["coverage"] += rcov
            parts["drift"] += rdrift
            parts["invalid_r"] += torch.where(inv_now, r_bad, zero)
            self.pot = torch.where(alive, pot, self.pot)
            self.prev_inc = _where(alive, cmd - self.prev_cmd, self.prev_inc)
            self.prev_cmd = _where(alive, cmd, self.prev_cmd)
            cmd_last = _where(alive, cmd, cmd_last)
            s = _where(alive, s_new, s)
            for k, v in g.items():
                self.geo[k] = _where(alive, v, self.geo[k])
            success |= succ_now
            invalid |= inv_now
            deadline |= dl
            alive = alive & ~succ_now & ~inv_now & ~dl
            if not bool(alive.any()):
                break
        refused = flags["ws_clip"] | flags["dq_scaled"] | flags["ik_err"]
        r_ref = -rc["refuse_cost"] * f64(refused)
        rew += smooth + r_ref
        self.episode_length_buf += 1
        self.cov = self._coverage(s, self.goal)
        dones = success | invalid | deadline
        for k, v in (("progress", parts["progress"]), ("smooth", smooth), ("time", parts["time"]), ("bonus", parts["bonus"]),
                     ("refused", r_ref), ("contact", parts["contact"]), ("coverage", parts["coverage"]), ("invalid_r", parts["invalid_r"]), ("reward", rew),
                     ("refusals", f64(refused)), ("drift", parts["drift"])):
            self.ep[k] += v
        for k in STEP_FLAGS:
            self.ep_flags[k] += f64(flags[k])
        pe, ye = goal_error(s, self.goal)
        extras = {"time_outs": torch.zeros(N, dtype=torch.bool, device=dev), "observations": {}}
        extras["log"] = self._log(rew, parts, smooth, r_ref, flags, pe, ye, success, invalid, deadline, dones, reasons_any)
        extras["log"]["/support/pen_log_step_share"] = n_pen.sum() / n_live.sum().clamp(min=1.0)
        extras["step_info"] = dict(success=success, invalid=invalid, deadline=deadline, reasons=reasons_any, cmd=cmd_last, state=s,
                                   pos_err=pe, yaw_err=ye.abs(), touched=self.touched.clone(), coverage=self.cov.clone(),
                                   contacts=self.n_runs.clone(), refusals=self.ep["refusals"].clone(), steps=self.t.clone(),
                                   returns={k: v.clone() for k, v in self.ep.items()},
                                   flags={k: v.clone() for k, v in self.ep_flags.items()})
        rew_out = rew.clone().to(torch.float32)
        self._compute_obs()
        extras["final_observations"] = self.obs_buf.clone()
        if self.auto_reset:
            ids = torch.nonzero(dones).flatten()
            if len(ids):
                self.reset_idx(ids)
                self._compute_obs()
        extras["observations"]["critic"] = self.obs_buf
        self.extras = extras
        return self.obs_buf, rew_out, dones.long(), extras

    def _log(self, rew, parts, smooth, r_ref, flags, pe, ye, success, invalid, deadline, dones, reasons):
        lg = {"/step/reward": rew.mean(), "/step/progress": parts["progress"].mean(), "/step/smooth": smooth.mean(),
              "/step/refused": r_ref.mean(), "/step/contact_bonus": parts["contact"].mean(), "/step/coverage_r": parts["coverage"].mean(),
              "/step/drift": parts["drift"].mean(),
              "/step/pos_err_mm": 1e3 * pe.mean(),
              "/step/yaw_err_deg": torch.rad2deg(ye.abs()).mean(), "/step/touched": self.touched.float().mean(),
              "/step/coverage": self.cov.mean(), "/step/t_at_rest": self.t_rest(C.flat23(self.hist[:, -1])).float().mean()}
        for k, v in flags.items():
            lg[f"/decoder/{k}"] = v.float().mean()
        for k, v in reasons.items():
            lg[f"/invalid/{k}"] = v.float().mean()
        if "drift_invalid" in parts:
            lg["/step/drift_invalid"] = parts["drift_invalid"].mean()
        ids = torch.nonzero(dones).flatten()
        if len(ids):
            # per level (self.level still holds the finished episodes' levels: the auto-reset comes after the log)
            hit = self.level[ids][:, None] == self.log_lv[None]             # [n, levels]
            cnt = hit.sum(0)
            share = torch.stack([(hit & x[ids][:, None]).sum(0).to(torch.float64) / cnt.clamp(min=1) for x in (success, invalid)])
            old = self.level_stats
            new = torch.where(torch.isnan(old), share, 0.95 * old + 0.05 * share)
            self.level_stats = torch.where((cnt > 0)[None], new, old)
            cur = dict(success=success[ids].float().mean(), invalid=invalid[ids].float().mean(), deadline=deadline[ids].float().mean(),
                       length=self.episode_length_buf[ids].float().mean(), ret=self.ep["reward"][ids].mean(),
                       touched=self.touched[ids].float().mean(), final_pos_mm=1e3 * pe[ids].mean(),
                       final_yaw_deg=torch.rad2deg(ye[ids].abs()).mean(), coverage=self.cov[ids].mean(),
                       contacts=self.n_runs[ids].float().mean(), refusals=self.ep["refusals"][ids].mean(),
                       yaw_blocked=self.ep_flags["yaw_blocked"][ids].mean(), lead_clipped=self.ep_flags["lead_clipped"][ids].mean(),
                       from_bank=self.from_bank[ids].float().mean())
            for k, v in cur.items():
                o = self.done_stats.get(k)
                self.done_stats[k] = v if o is None else 0.95 * o + 0.05 * v
        for k in ("success", "invalid", "deadline", "length", "ret", "touched", "final_pos_mm", "final_yaw_deg", "coverage", "contacts",
                  "refusals", "yaw_blocked", "lead_clipped", "from_bank"):
            lg[f"/episode/{k}"] = self.done_stats.get(k, torch.tensor(0.0, device=self.device))
        lg["/reset/from_bank"] = self.rb_counts[1] / self.rb_counts[0].clamp(min=1.0)
        for j, lv in enumerate(self.levels):
            lg[f"/level/{lv}/success"], lg[f"/level/{lv}/invalid"] = self.level_stats[0, j], self.level_stats[1, j]
        return lg

    def _compute_obs(self):
        h23 = C.flat23(self.hist)
        tl = (self.horizon - self.t).clamp(min=0).to(torch.float64)
        self.obs_buf = self.obs_builder(h23, self.ok, self.prev_cmd, self.prev_inc, self.goal, tl, self.horizon.to(torch.float64),
                                        self.geo, self.decoder.features(), self.t_since_contact, self.t_rest(h23[:, -1]), self.cov)
        return self.obs_buf

    def current_state23(self):
        return C.flat23(self.hist[:, -1])
