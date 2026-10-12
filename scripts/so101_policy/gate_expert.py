"""Expert agreement gate (review_spec_feasibility P1-9) and phase noise test (P0-4) of the SO101 + cube study.

The privileged FSM expert (``expert.PickPlaceExpert``) acts closed loop through decoder v2 (``decoder.DecoderV2``) on
the same tasks in CPU MuJoCo (``mujoco_env.MujocoWorlds``, the collector's physics = ground truth), in the grasp NRD
(``nrd_sim.GraspNrdSim``, one checkpoint per run) and in the kinematic control world (``kin_world.KinWorld``). Every
episode is scored by the official task rules (``task.PickPlaceTask``) with ``TaskConfig.evaluation()`` in ALL worlds
(the grasp-box rule logs, it does not end the episode: with the training rule the NRD would end episodes that MuJoCo
continues, a disagreement by construction; ``--train-rules`` switches); the link / jaw overlap rules run in the NRD and
the kinematic world and are off in MuJoCo, which has the real contacts and the collector's visual-overlap end.
Ablation: the 01:20 draft expert (per-step FSM, ``--expert-v1`` = commit 9c468acaf ``expert.py``, verbatim copy)
through decoder v1 (``DecoderV1``). Results and reading: ~/so101_policy/docs/gate_expert.md.

    PYTHONPATH=src python -P scripts/so101_policy/gate_expert.py tasks --out tasks.npz [--seed0 1000] [--n 1024]
    PYTHONPATH=src python -P scripts/so101_policy/gate_expert.py run --tasks tasks.npz --world mujoco|nrd|kin \
        --config base|approach|align|carry|retarget|place [--decoder v2|v1] [--ckpt CKPT] [--workers 4] --out run.npz
    PYTHONPATH=src python -P scripts/so101_policy/gate_expert.py replay --tasks tasks.npz --mj RUN.npz \
        --world nrd|kin|mujoco [--ckpt CKPT] --out replay.json      # a run's joint targets replayed open loop
    PYTHONPATH=src python -P scripts/so101_policy/gate_expert.py report --tasks tasks.npz --runs DIR --out report.json

Tasks (spec_v2 B4, written down here): task i uses start seed ``seed0 + i`` (1000-2023: the validation range; never
the test seeds 900000+ or a collector seed, ``mujoco_env.sample_starts`` refuses those): the collector's start sampler
and 0.3 s warm-up (MuJoCo resets bit-identically with ``warmup=True``, checked against the stored settled state; the
NRD and kinematic world start from that state). Goal: ``task.sample_goals`` (annulus, 6-30 cm from the start cube,
yaw = start yaw + U(-30, 30) deg mod 90) with a CPU generator seeded ``--goal-seed``; a second goal (same rule,
``GOAL2_SEP`` = 5-10 cm from the first) for re-targeting.

Episode (all worlds): every 5 steps (0.1 s) the expert gives a decoder action from the true state; every 20 ms step the
decoder gives joint targets (lead band against the measured angles), the world steps, the task rules run; an episode
ends at the task's first end (success, a rule, or the world's own end flag) or the horizon (configs base / retarget
400 steps = 8 s, noise configs 600 = 12 s: noisy alignments are slow and the NRD has no time input).

Configurations (``CONFIGS``). Noise = the planned PPO exploration: pre-tanh Gaussian std 0.3 on the free-motion
action a[0:4] (a' = tanh(atanh(a) + 0.3 eps)), drawn from a counter-based hash of (task seed, decision index), so a
task gets the same noise sequence in every world. It is added only on the decisions of ONE phase (``Latches``):

| Label | v2 decisions (decoder free / hold only) | v1 steps (ablation; per 20 ms step, no macros) |
|---|---|---|
| approach | expert approach, before the alignment latch | approach before the latch |
| align | expert approach after the decoder target came within ``R_LATCH`` (xy) of the cube (latched until the phase ends) | approach after the latch, descend, close, grip |
| carry | expert carry, before the place latch (config carry also re-targets the goal) | lift, carry before the latch |
| place | expert carry after the held cube came within ``R_LATCH`` (xy) of the goal | carry after the latch, lower, release |

Macros (GRASP, PLACE) never get noise. The expert's own triggers need 1.5 mm / 1 deg / 1 cm/s and never fire under
noise (the target jitters about 1 cm per decision), so in the noisy align / place phases the harness also triggers, as
a trained PPO policy would, when the macro would start inside a margin of the task's trigger box: GRASP when the
decoder's brake end (``brake_stop``, the decoder's own rule = ``macro_goal``) has the cube centre within ``GRASP_TOL``
(|x| along the closing axis, |y| across it) and the yaw within ``GRASP_YAW`` (expert abort distance raised for this
config, ``EXPERT_CFG``); PLACE when the predicted release point (brake end + the held cube's grasp-frame offset + the
expert's release shift) is within ``PLACE_TOL`` / ``PLACE_YAW`` of the goal. The expert's carry waypoint keeps the
current target height (no restoring term): noisy carry decisions plan back to the carry-start height instead. The v1
expert never 'arrives' under per-step noise: on noisy steps it gets looser tolerances (``V1_NOISY``) and the harness
advances its phases with the same tests (approach -> descend in the trigger box, descend -> close and lower -> release
at 3 mm, lift -> carry at 1 cm, carry -> lower by the place test). Re-targeting (configs carry and retarget): at the
``RETARGET_AT``-th decision of the first carry the goal changes to the second goal (task and expert).

Probe options of ``run`` (diagnostics, not gate configs): ``--goal-at-start --hold-s S`` (still hold in the air),
``--pause-s S`` (still pause before the carry), ``--expert-cfg`` / ``--task-cfg`` JSON overrides (e.g.
'{"hold_cap_s": 99}').

Outputs: per run an .npz (per-episode outcome, rule, timings, close-start offsets; per-step q, cube pose, grasp-frame
cube offset, commands, decoder mode, expert phase, noise label); ``report`` makes the 2 x 2 tables (MuJoCo vs each
other world: success 15 / 15, 10 / 10 and lift; also conditional on both lifted / both reaching PLACE), agreement,
phi, exact McNemar p, failure causes per rule (MuJoCo world ends by collector reason), per-phase timing, in-hand slip
during the carry, and the divergence point of disagreeing episodes (first step where the two cube positions differ by
more than ``DIVERGE_MM``) with the phase it falls in; ``replay`` gives the open-loop error of a world under another
run's commands at the GRASP trigger, firm lift, PLACE trigger and end.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

HORIZON = 400
DECISION = 5
SIGMA = 0.3                                   # pre-tanh exploration std (RF P1-10)
R_LATCH = 0.03                                # m, alignment / place-approach latch radius
GRASP_TOL = (0.0045, 0.010)                   # m, harness GRASP trigger: |x| (closing axis), |y| (jaw width) of the
GRASP_YAW = math.radians(2.5)                 #   cube at the brake end (task trigger box 5.5 / 12.5 mm, 3 deg)
PLACE_TOL = 0.008                             # m, harness PLACE trigger: predicted cube xy error ...
PLACE_YAW = math.radians(8.0)                 # ... and yaw error (mod 90) (success 15 mm / 15 deg)
RETARGET_AT = 3                               # decisions into the first carry
GOAL2_SEP = (0.05, 0.10)                      # m, second goal this far from the first
EXPERT_CFG = dict(align=dict(abort_off=0.013, yaw_ok=GRASP_YAW))   # per-config expert overrides (module docstring)
DIVERGE_MM = 5.0
GEOM_NPZ = Path.home() / "so101_policy" / "ckpt" / "task_geom_v1.npz"
EXPERT_V1 = Path.home() / "so101_policy" / "runs" / "gate" / "expert_v1.py"   # commit 9c468acaf expert.py, verbatim
V1_NOISY = dict(settle_tol=0.004, chain_tol=0.008, arm_tol=0.006, yaw_tol=math.radians(3.0), off_line=0.013)
NOISY_HORIZON = 600                           # 12 s: noisy alignments are slow; the NRD has no time input
CONFIGS = {"base": dict(noise=None, retarget=False, horizon=HORIZON),
           "approach": dict(noise="approach", retarget=False, horizon=NOISY_HORIZON),
           "align": dict(noise="align", retarget=False, horizon=NOISY_HORIZON),
           "carry": dict(noise="carry", retarget=True, horizon=NOISY_HORIZON),
           "retarget": dict(noise=None, retarget=True, horizon=HORIZON),
           "place": dict(noise="place", retarget=False, horizon=NOISY_HORIZON)}
LABELS = ("none", "approach", "align", "carry", "place")
LB = {n: k for k, n in enumerate(LABELS)}
WORLDS = ("mujoco", "nrd", "kin")
V1_MACRO = dict(approach="free", descend="grasp_descend", close="grasp_close", grip="grasp_grip", lift="grasp_lift",
                carry="free", lower="place_lower", release="place_open", retreat="place_retreat", done="free",
                recover="place_retreat")


# ---- noise, triggers, labels (pure functions; unit-tested) -----------------------------------------------------------
def hash_normal(seed, counter, n, stream0=0):
    """Counter-based standard normals [B, n] (Box-Muller on ``decoder.hash_uniform``) for int64 seeds [B] and one
    integer counter: the same numbers on every device and batch layout."""
    from nedm.so101_policy.decoder import hash_uniform
    cnt = torch.full_like(seed, int(counter))
    out = []
    for j in range(n):
        u1 = hash_uniform(seed, cnt, stream0 + 2 * j + 1)
        u2 = hash_uniform(seed, cnt, stream0 + 2 * j + 2)
        out.append(torch.sqrt(-2.0 * torch.log1p(-u1)) * torch.cos(2 * math.pi * u2))
    return torch.stack(out, -1)


def pretanh_noise(a, eps, sigma=SIGMA, mask=None):
    """a [B, k] in [-1, 1] -> tanh(atanh(a) + sigma eps) on the rows of mask [B] (a PPO tanh-Gaussian sample whose mean
    is the expert's action); |a| is clipped to 0.999 before atanh."""
    x = torch.atanh(a.clamp(-0.999, 0.999)) + sigma * eps.to(a)
    out = torch.tanh(x)
    return out if mask is None else torch.where(mask[:, None], out, a)


def brake_stop(dec):
    """Where a GRASP / PLACE trigger at this decision stops the target of a DecoderV2 (cosine blend): the decoder's
    brake rule (``DecoderV2.decide``): tb = max(T_d, pi |v| / (2 a_max), pi |w| / (2 alpha_max), joint bound) rounded
    up to the control step, 0 at rest; end = target + v tb / 2 -> (xy [B, 2] m, yaw [B] rad) = ``macro_goal`` x, y, yaw."""
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


def grasp_ready(stop_xy, stop_yaw, cube_xy, cube_yaw, tol=GRASP_TOL, tol_yaw=GRASP_YAW):
    """Harness GRASP trigger test [B] bool: cube centre within |x| <= tol[0] along the closing axis of the brake end
    and |y| <= tol[1] across it (the task's trigger box frame, ``task.goal_box_ok``), and its yaw (mod 90) within
    tol_yaw of the brake-end yaw."""
    from nedm.so101_policy.task import wrap90
    d = cube_xy - stop_xy
    c, s = torch.cos(stop_yaw), torch.sin(stop_yaw)
    x, y = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    return (x.abs() <= tol[0]) & (y.abs() <= tol[1]) & (wrap90(stop_yaw - cube_yaw).abs() <= tol_yaw)


def place_ready(stop_xy, stop_yaw, rel, grip_yaw, cube_yaw, goal_xy, goal_yaw, shift, tol=PLACE_TOL, tol_yaw=PLACE_YAW):
    """Harness PLACE trigger test [B] bool: the held cube (centre rel [B, 3] in the grasp frame, yaw cube_yaw; measured
    gripper yaw grip_yaw) released at the brake end (stop_xy, stop_yaw) after the expert's release shift along the
    closing axis lands within tol of goal_xy and within tol_yaw (mod 90) of goal_yaw."""
    from nedm.so101_policy.task import wrap90
    c, s = torch.cos(stop_yaw), torch.sin(stop_yaw)
    rx = rel[:, 0] + shift
    pred = stop_xy + torch.stack((c * rx - s * rel[:, 1], s * rx + c * rel[:, 1]), -1)
    pred_yaw = stop_yaw + wrap90(cube_yaw - grip_yaw)
    return ((pred - goal_xy).norm(dim=-1) <= tol) & (wrap90(pred_yaw - goal_yaw).abs() <= tol_yaw)


class Latches:
    """Per-world phase latches of the noise labels (alignment above the cube, approach to the goal)."""

    def __init__(self, B, device):
        self.align = torch.zeros(B, dtype=torch.bool, device=device)
        self.place = torch.zeros(B, dtype=torch.bool, device=device)

    def label(self, approach, carry, near_cube, near_goal, extra_align=None, extra_carry=None, extra_place=None):
        """Bool [B] inputs (the world is in the expert's approach / carry phase; target within the latch radius of the
        cube; held cube within it of the goal) -> label codes [B] (``LABELS``). The latches hold while the phase lasts
        and reset when it ends. extra_*: v1 phases that count as align / carry / place."""
        self.align = approach & (self.align | near_cube)
        self.place = carry & (self.place | near_goal)
        z = torch.zeros_like(approach)
        ea, ec, ep = (z if x is None else x for x in (extra_align, extra_carry, extra_place))
        lab = torch.zeros(approach.shape, dtype=torch.long, device=approach.device)
        lab = torch.where(approach & ~self.align, LB["approach"], lab)
        lab = torch.where((approach & self.align) | ea, LB["align"], lab)
        lab = torch.where((carry & ~self.place) | ec, LB["carry"], lab)
        lab = torch.where((carry & self.place) | ep, LB["place"], lab)
        return lab


# ---- statistics (pure numpy; unit-tested) ---------------------------------------------------------------------------
def mcnemar(b, c):
    """Exact two-sided McNemar p for the discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))


def table2x2(ref, other):
    """ref, other [N] bool (MuJoCo, other world) -> dict: n11 both, n10 ref only, n01 other only, n00 neither,
    agreement, phi (nan if a margin is empty), McNemar p, rates."""
    ref, other = np.asarray(ref, bool), np.asarray(other, bool)
    n11, n10 = int((ref & other).sum()), int((ref & ~other).sum())
    n01, n00 = int((~ref & other).sum()), int((~ref & ~other).sum())
    den = math.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    return dict(n=len(ref), n11=n11, n10=n10, n01=n01, n00=n00, agree=(n11 + n00) / max(1, len(ref)),
                phi=(n11 * n00 - n10 * n01) / den if den > 0 else float("nan"), mcnemar_p=mcnemar(n10, n01),
                rate_ref=float(ref.mean()) if len(ref) else float("nan"),
                rate_other=float(other.mean()) if len(ref) else float("nan"))


def wilson(k, n, z=1.96):
    """95 % Wilson interval of k / n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def first_divergence(cube_a, cube_b, length, thr_mm=DIVERGE_MM):
    """cube_a, cube_b [N, T, >= 3] positions (m), length [N] (common live steps) -> first step [N] where the cube centres
    differ by more than thr_mm (-1: never)."""
    d = np.linalg.norm(cube_a[..., :3] - cube_b[..., :3], axis=-1) * 1e3
    T = d.shape[1]
    live = np.arange(T)[None] < np.asarray(length)[:, None]
    hit = (d > thr_mm) & live
    return np.where(hit.any(1), hit.argmax(1), -1)


# ---- tasks -----------------------------------------------------------------------------------------------------------
def make_goals(cube_xy, cube_yaw, goal_seed, sep=GOAL2_SEP):
    """Goals of the tasks (``task.sample_goals``, CPU generator seeded goal_seed) and a second goal per task (same rule,
    sep[0]..sep[1] from the first; up to 400 draws) -> (goal_xy, goal_yaw, goal2_xy, goal2_yaw) tensors."""
    from nedm.so101_policy import task as T
    gen = torch.Generator().manual_seed(int(goal_seed))
    gxy, gyaw = T.sample_goals(cube_xy, cube_yaw, gen)
    g2, y2 = torch.full_like(gxy, math.nan), torch.full_like(gyaw, math.nan)
    todo = torch.ones(len(gxy), dtype=torch.bool)
    for _ in range(400):
        cxy, cyw = T.sample_goals(cube_xy, cube_yaw, gen)
        d = (cxy - gxy).norm(dim=-1)
        ok = todo & (d >= sep[0]) & (d <= sep[1])
        g2, y2 = torch.where(ok[:, None], cxy, g2), torch.where(ok, cyw, y2)
        todo &= ~ok
        if not bool(todo.any()):
            break
    else:
        raise RuntimeError("no second goal")
    return gxy, gyaw, g2, y2


def cmd_tasks(a):
    from nedm.so101_policy import task as T
    from nedm.so101_policy.mujoco_env import sample_starts
    seeds = np.arange(a.seed0, a.seed0 + a.n)
    if (seeds >= 900000).any():
        raise ValueError("900000+ are the test seeds")
    t0 = time.time()
    st = sample_starts(seeds)
    s = st["state"]
    cyaw, tilt = T.cube_yaw_tilt(s.cube_quat)
    gxy, gyaw, g2, y2 = make_goals(s.cube_pos[:, :2], cyaw, a.goal_seed)
    out = dict(seeds=seeds, start_q=st["q"], start_cube_pos=st["cube_pos"], start_cube_quat=st["cube_quat"],
               tries=st["tries"], goal_xy=gxy.numpy(), goal_yaw=gyaw.numpy(), goal2_xy=g2.numpy(), goal2_yaw=y2.numpy(),
               goal_seed=np.int64(a.goal_seed), **{f"state_{k}": getattr(s, k).numpy() for k in
                                                 ("q", "qd", "cube_pos", "cube_quat", "cube_v", "cube_w")})
    np.savez(a.out, **out)
    d = np.linalg.norm(out["goal_xy"] - out["state_cube_pos"][:, :2], axis=1)
    print(f"{a.n} tasks (seeds {seeds[0]}-{seeds[-1]}) -> {a.out} in {time.time() - t0:.0f} s; redraws {int((st['tries'] > 1).sum())};"
          f" goal distance p50 {np.median(d) * 1e3:.0f} mm ({d.min() * 1e3:.0f}-{d.max() * 1e3:.0f}); start tilt max "
          f"{math.degrees(float(tilt.max())):.2f} deg")


def load_tasks(path, idx=None):
    t = dict(np.load(path))
    return t if idx is None else {k: (v[idx] if np.ndim(v) and len(v) == len(t["seeds"]) else v) for k, v in t.items()}


def start_state(t, device):
    from nedm.so101_policy.world import WorldState
    f = lambda k: torch.as_tensor(t[f"state_{k}"], dtype=torch.float64, device=device)   # noqa: E731
    n = len(t["seeds"])
    z = lambda *s: torch.zeros(*s, dtype=torch.bool, device=device)                     # noqa: E731
    return WorldState(q=f("q"), qd=f("qd"), cube_pos=f("cube_pos"), cube_quat=f("cube_quat"), cube_v=f("cube_v"),
                      cube_w=f("cube_w"), contact=z(n, 3), arm_table=z(n), invalid=z(n))


# ---- one run ---------------------------------------------------------------------------------------------------------
def load_expert_v1(path=EXPERT_V1):
    import sys
    if "gate_expert_v1" not in sys.modules:   # dataclasses need the module registered while it runs
        spec = importlib.util.spec_from_file_location("gate_expert_v1", str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gate_expert_v1"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["gate_expert_v1"]


def make_world(kind, B, device, ckpt=None, workers=0, kin=None, geom=None, mj_python=None):
    if kind == "mujoco":
        from nedm.so101_policy.mujoco_env import MujocoWorlds
        return MujocoWorlds(B, workers=workers, python=mj_python)
    if kind == "nrd":
        from nedm.so101_policy.nrd_sim import GraspNrdSim
        return GraspNrdSim(ckpt, num_envs=B, device=device, members="split")
    if kind == "kin":
        from nedm.so101_policy.kin_world import KinWorld
        return KinWorld(B, device, kin=kin, geom=geom)
    raise ValueError(kind)


def task_geometry(device):
    from nedm.so101_policy.task import TaskGeometry
    return TaskGeometry.load(GEOM_NPZ, device) if GEOM_NPZ.exists() else TaskGeometry.from_mujoco(device)


@torch.no_grad()
def run_episodes(tasks, world_kind, config, decoder="v2", ckpt=None, device="cpu", workers=0, sigma=SIGMA,
                 noise_seed=0, expert_v1=EXPERT_V1, horizon=None, mj_python=None, eval_rules=True, goal_at_start=False,
                 hold_s=0.0, pause_s=0.0, expert_cfg=None, task_cfg=None, log=print):
    """Closed-loop expert episodes on all tasks of ``tasks`` (load_tasks dict) in one world -> dict of numpy arrays
    (module docstring). Probe options (v2): goal_at_start = the goal is the start cube pose (no carry motion);
    hold_s = no PLACE trigger before hold_s seconds of carry (a static hold in the air when goal_at_start); pause_s =
    zero free motion for the first pause_s of the carry (the collector's still 'hold' between the lift and the move);
    expert_cfg = ExpertConfig overrides (e.g. carry_profile="minjerk"); task_cfg = TaskConfig overrides (e.g.
    hold_cap_s=99 for the probes)."""
    from nedm.so101_policy import task as T
    from nedm.so101_policy.decoder import M, DecoderV1, DecoderV2
    from dataclasses import replace

    from nedm.so101_policy.expert import PH, PHASES, ExpertConfig, PickPlaceExpert
    from nedm.so101_policy.expert import cube_yaw_tilt as expert_yaw
    from nedm.so101_policy.kin import SO101Kin, yaw_of
    cfgd = CONFIGS[config]
    horizon = cfgd["horizon"] if horizon is None else int(horizon)
    dev = torch.device(device)
    B = len(tasks["seeds"])
    t0 = time.time()
    kin = SO101Kin(dev)
    geom = task_geometry(dev)
    world = make_world(world_kind, B, dev, ckpt, workers, kin, geom, mj_python)
    s0 = start_state(tasks, dev)
    if world_kind == "mujoco":
        ws = world.reset(None, dict(q=tasks["start_q"], cube_pos=tasks["start_cube_pos"],
                                    cube_quat=tasks["start_cube_quat"]), warmup=True)
        dq = (ws.q - s0.q).abs().max().item()
        dc = (ws.cube_pos - s0.cube_pos).abs().max().item()
        if dq > 1e-6 or dc > 1e-6:
            raise RuntimeError(f"MuJoCo warm-up start differs from the stored settled state ({dq:.2e}, {dc:.2e})")
        overlap = False
    else:
        ws = world.reset(None, s0)
        overlap = True
    kw = dict(horizon=horizon, **(task_cfg or {}))
    tcfg = T.TaskConfig.evaluation(**kw) if eval_rules else T.TaskConfig(**kw)
    ws = ws.to(dev)
    task = T.PickPlaceTask(B, dev, kin=kin, geom=geom, cfg=tcfg, overlap_rules=overlap)
    g = lambda k: torch.as_tensor(tasks[k], dtype=torch.float64, device=dev)              # noqa: E731
    goal_xy, goal_yaw, goal2_xy, goal2_yaw = g("goal_xy"), g("goal_yaw"), g("goal2_xy"), g("goal2_yaw")
    if goal_at_start:
        goal_xy, goal_yaw = s0.cube_pos[:, :2].clone(), T.cube_yaw_tilt(s0.cube_quat)[0]
    seeds = torch.as_tensor(tasks["seeds"], dtype=torch.long, device=dev)
    nseed = seeds * 7919 + 104729 * int(noise_seed) + 17
    all_ = torch.arange(B, device=dev)
    v2 = decoder == "v2"
    if v2:
        dec = DecoderV2(B, dev, kin=kin)
        dec.reset(all_, ws.q, seed=seeds)
        ex = PickPlaceExpert(dec, replace(ExpertConfig(), **{**EXPERT_CFG.get(config, {}), **(expert_cfg or {})}))
        phases = list(PHASES)
    else:
        mod = load_expert_v1(expert_v1)
        dec = DecoderV1(B, dev, kin=kin)
        dec.reset(all_, ws.q)
        ex = mod.PickPlaceExpert(dec)
        phases = list(mod.PHASES)
        base_tol = {k: float(getattr(ex.cfg, k)) for k in V1_NOISY}
        v1_macro = torch.tensor([T.MC[V1_MACRO[p]] for p in phases], dtype=torch.long, device=dev)
        P1 = mod.PH
    ex.reset(all_, ws, goal_xy, goal_yaw)
    task.reset(all_, ws, goal_xy, goal_yaw)
    lat = Latches(B, dev)
    noise_on = cfgd["noise"]
    carry_dec = torch.zeros(B, dtype=torch.long, device=dev)
    retarget_step = torch.full((B,), -1, dtype=torch.long, device=dev)
    noisy_n = torch.zeros(B, dtype=torch.long, device=dev)
    h_grasp = torch.zeros(B, dtype=torch.long, device=dev)
    h_place = torch.zeros(B, dtype=torch.long, device=dev)
    rec = {k: [] for k in ("q", "cube", "xg", "u", "mode", "phase", "label", "noisy", "macro")}
    first_fire = torch.full((B, len(T.RULES)), -1, dtype=torch.long, device=dev)
    lift_step = torch.full((B,), -1, dtype=torch.long, device=dev)
    close_x = torch.full((B, 4), math.nan, dtype=torch.float64, device=dev)
    mj_reason = torch.zeros(B, dtype=torch.long)
    info = None
    lab = torch.zeros(B, dtype=torch.long, device=dev)
    noisy = torch.zeros(B, dtype=torch.bool, device=dev)
    z_hold = torch.full((B,), math.nan, dtype=torch.float64, device=dev)   # target height at the carry start (v2)
    for t in range(horizon):
        decide = v2 and t % DECISION == 0
        if decide or not v2:
            k = t // DECISION if v2 else t
            a = ex.act(ws)
            ph = ex.phase
            cube_xy = ws.cube_pos[:, :2].to(torch.float64)
            if v2:
                free, hold = dec.mode == M["free"], dec.mode == M["hold"]
                approach, carry = (ph == PH["approach"]) & free, (ph == PH["carry"]) & hold
                lab = lat.label(approach, carry, (dec.pos[:, :2] - cube_xy).norm(dim=-1) < R_LATCH,
                                (cube_xy - ex.goal_xy).norm(dim=-1) < R_LATCH)
                z_hold = torch.where(carry, torch.where(torch.isnan(z_hold), dec.pos[:, 2], z_hold), math.nan)
            else:
                approach, carry = ph == P1["approach"], ph == P1["carry"]
                isin = lambda *n: torch.stack([ph == P1[x] for x in n]).any(0)              # noqa: E731
                lab = lat.label(approach, carry, (dec.pos[:, :2] - cube_xy).norm(dim=-1) < R_LATCH,
                                (cube_xy - ex.goal_xy).norm(dim=-1) < R_LATCH, extra_align=isin("descend", "close", "grip"),
                                extra_carry=isin("lift"), extra_place=isin("lower", "release"))
            noisy = (lab == LB[noise_on]) if noise_on else torch.zeros(B, dtype=torch.bool, device=dev)
            if v2:
                noisy = noisy & (free | hold)
                if noise_on in ("align", "place") and bool(noisy.any()):
                    sxy, syaw = brake_stop(dec)
                    cyaw, _ = expert_yaw(ws.cube_quat.to(torch.float64))
                    if noise_on == "align":
                        ok = (noisy & approach & dec.armed & (a[:, 4] <= 0)
                              & (ws.cube_v.to(torch.float64).norm(dim=-1) < ex.cfg.cube_still)
                              & grasp_ready(sxy, syaw, cube_xy, cyaw))
                        ex.phase = torch.where(ok, PH["grasp"], ex.phase)
                    else:
                        held, rel = ex._held(ws)
                        _, rg = kin.fk(ws.q.to(torch.float64))
                        ok = (noisy & carry & held & dec.armed & (a[:, 4] <= 0)
                              & place_ready(sxy, syaw, rel, yaw_of(rg), cyaw, ex.goal_xy, ex.goal_yaw,
                                            ex.cfg.release_shift))
                        ex.phase = torch.where(ok, PH["place"], ex.phase)
                    a[:, 4] = torch.where(ok, 1.0, a[:, 4])
                    (h_grasp if noise_on == "align" else h_place).add_(ok.long())
                    noisy = noisy & ~ok
                # the expert's carry waypoint keeps the CURRENT target height (no restoring term), so height noise
                # would random-walk the held cube into the table: noisy carry decisions plan back to the carry start
                zc = noisy & carry
                if bool(zc.any()):
                    vz = (z_hold - dec.pos[:, 2] - dec.vel[:, 2] * dec.T_d / 2) / dec.T_d
                    vz = vz.clamp(-ex.cfg.v_cruise, ex.cfg.v_cruise)
                    a[:, 2] = torch.where(zc, (vz * dec.T_d / dec.cfg.d_max).clamp(-1, 1), a[:, 2])
            else:
                for key, v in V1_NOISY.items():
                    setattr(ex.cfg, key, torch.where(noisy, v, base_tol[key]))
                if bool(noisy.any()):   # the v1 expert never 'arrives' under per-step noise: harness transitions
                    rest, ryaw = ex._rest()
                    q64 = ws.q.to(torch.float64)
                    gp, rg = kin.fk(q64)
                    cyaw, _ = expert_yaw(ws.cube_quat.to(torch.float64))
                    held1, rel1 = ex._held_parts(ws)
                    dz_rest, dz_tcp = (rest[:, 2] - ex.wp[:, 2]).abs(), (gp[:, 2] - ex.wp[:, 2]).abs()
                    moves = (
                        (noisy & (ph == P1["approach"]) & (dz_rest < 0.01) & grasp_ready(rest[:, :2], ryaw, cube_xy, cyaw)
                         & grasp_ready(dec.pos[:, :2], dec.yaw, cube_xy, cyaw)
                         & (ws.cube_v.to(torch.float64).norm(dim=-1) < 0.01), P1["descend"], h_grasp),
                        (noisy & (ph == P1["descend"]) & (dz_tcp < 0.003), P1["close"], None),
                        (noisy & (ph == P1["lift"]) & held1 & (dz_tcp < 0.01), P1["carry"], None),
                        (noisy & (ph == P1["carry"]) & held1
                         & place_ready(rest[:, :2], ryaw, rel1, yaw_of(rg), cyaw, ex.goal_xy, ex.goal_yaw,
                                       ex.cfg.release_shift), P1["lower"], h_place),
                        (noisy & (ph == P1["lower"]) & (dz_tcp < 0.003), P1["release"], None))
                    for m, new, cnt in moves:
                        if bool(m.any()):   # the expert's own phase-change side effects: timer, anchor
                            ex.phase = torch.where(m, new, ex.phase)
                            ex.timer = torch.where(m, torch.zeros_like(ex.timer), ex.timer)
                            ex.anchor = torch.where(m[:, None], rest, ex.anchor)
                            if cnt is not None:
                                cnt.add_(m.long())
            if noise_on:
                eps = hash_normal(nseed, k, 4)
                a[:, :4] = pretanh_noise(a[:, :4], eps, sigma, noisy)
                noisy_n += noisy.long()
            if v2 and max(hold_s, pause_s) > 0:   # probes: no PLACE before hold_s, no motion before pause_s of carry
                incarry = (dec.mode == M["hold"]) & ((ex.phase == PH["carry"]) | (ex.phase == PH["place"]))
                early = incarry & (carry_dec < round(max(hold_s, pause_s) / (DECISION * 0.02)))
                ex.phase = torch.where(early & (ex.phase == PH["place"]), PH["carry"], ex.phase)
                a[:, 4] = torch.where(early, -1.0, a[:, 4])
                still = incarry & (carry_dec < round((hold_s if goal_at_start else pause_s) / (DECISION * 0.02)))
                a[:, :4] = torch.where(still[:, None], 0.0, a[:, :4])
            in_carry = ex.phase == (PH["carry"] if v2 else P1["carry"])
            carry_dec = torch.where(in_carry & (retarget_step < 0), carry_dec + 1, carry_dec)
            if cfgd["retarget"]:
                rt = (carry_dec == RETARGET_AT) & (retarget_step < 0)
                if bool(rt.any()):
                    ix = rt.nonzero().flatten()
                    ex.goal_xy[ix], ex.goal_yaw[ix] = goal2_xy[ix], goal2_yaw[ix]
                    task.set_goal(ix, goal2_xy[ix], goal2_yaw[ix])
                    lat.place[ix] = False
                    retarget_step = torch.where(rt, t, retarget_step)
            if v2:
                dec.decide(a, ws)
        if v2:
            u, _ = dec.step(ws.q)
            macro, mgoal = dec.macro(), dec.macro_goal
            mode = dec.mode.clone()
        else:
            u = dec.step(a)
            macro, mgoal = v1_macro[ex.phase], None
            mode = torch.full((B,), -1, dtype=torch.long, device=dev)
        ws = world.step(u)
        if world_kind == "mujoco":
            ws = ws.to(dev)
        _, _, _, info = task.step(ws, macro, mgoal)
        live_before = info["t"] == t + 1                      # stepped this time (task t counts live steps)
        fl = info["flags"]
        first_fire = torch.where((first_fire < 0) & fl & live_before[:, None], t, first_fire)
        lift_step = torch.where((lift_step < 0) & info["lifted"], t, lift_step)
        cstart = (macro == T.MC["grasp_close"]) & torch.isnan(close_x[:, 0]) & live_before
        if bool(cstart.any()):    # first close step: cube centre in the grasp frame and yaw misalignment
            F = task.F
            close_x = torch.where(cstart[:, None], torch.cat((F["xg"], F["yaw_mis"][:, None]), -1), close_x)
        if world_kind == "mujoco":
            new_end = live_before & (info["end"] != 0)
            mj_reason = torch.where(new_end.cpu(), world.end_reason.long(), mj_reason)
        rec["q"].append(ws.q.float().cpu()), rec["u"].append(u.float().cpu())
        rec["cube"].append(torch.cat((ws.cube_pos, ws.cube_quat), -1).float().cpu())
        rec["xg"].append(task.F["xg"].float().cpu())
        rec["mode"].append(mode.to(torch.int8).cpu()), rec["phase"].append(ex.phase.to(torch.int8).cpu())
        rec["label"].append(lab.to(torch.int8).cpu()), rec["noisy"].append(noisy.cpu())
        rec["macro"].append(macro.to(torch.int8).cpu())
        if bool((info["end"] != 0).all()):
            break
        if t % 100 == 99:
            log(f"  t {t + 1}: ended {int((info['end'] != 0).sum())}/{B}, {time.time() - t0:.0f} s", flush=True)
    if world_kind == "mujoco":
        world.close()
    end = info["end"].cpu().numpy()
    out = {k: torch.stack(v, 1).numpy() for k, v in rec.items()}
    out.update(end=end, success=end == T.EC["success"], success10=info["success10"].cpu().numpy(),
               end_xy=info["end_xy_err"].cpu().numpy(), end_yaw=info["end_yaw_err"].cpu().numpy(),
               lifted=info["lifted"].cpu().numpy(), lift_step=lift_step.cpu().numpy(), t_end=info["t"].cpu().numpy(),
               box_log=info["grasp_box_log"].cpu().numpy(), first_fire=first_fire.cpu().numpy(),
               close_x=close_x.cpu().numpy(), mj_reason=mj_reason.numpy(), retarget_step=retarget_step.cpu().numpy(),
               noisy_n=noisy_n.cpu().numpy(), harness_grasp=h_grasp.cpu().numpy(), harness_place=h_place.cpu().numpy(),
               recoveries=ex.recoveries.cpu().numpy(), seeds=np.asarray(tasks["seeds"]),
               phase_names=np.array(phases), world=np.array(world_kind), config=np.array(config),
               decoder=np.array(decoder), ckpt=np.array(str(ckpt)), sigma=np.float64(sigma),
               noise_seed=np.int64(noise_seed), wall_s=np.float64(time.time() - t0), device=np.array(str(dev)))
    if v2:
        out["counts"] = np.stack([v.cpu().numpy() for v in dec.counts.values()], -1)
        out["count_names"] = np.array(list(dec.counts))
    return out


def summary_line(r):
    from nedm.so101_policy.task import ENDS
    e = r["end"]
    causes = {ENDS[k]: int((e == k).sum()) for k in np.unique(e)}
    return (f"{r['world']} {r['config']} {r['decoder']}: success 15/15 {r['success'].mean() * 100:.1f} %, 10/10 "
            f"{r['success10'].mean() * 100:.1f} %, lifted {r['lifted'].mean() * 100:.1f} %; ends {causes}; "
            f"wall {float(r['wall_s']):.0f} s")


def cmd_run(a):
    tasks = load_tasks(a.tasks)
    if a.n:
        tasks = load_tasks(a.tasks, np.arange(a.first, a.first + a.n))
    torch.set_num_threads(a.threads)
    r = run_episodes(tasks, a.world, a.config, a.decoder, a.ckpt, a.device, a.workers, a.sigma, a.noise_seed,
                     a.expert_v1, a.horizon, a.mj_python, eval_rules=not a.train_rules, goal_at_start=a.goal_at_start,
                     hold_s=a.hold_s, pause_s=a.pause_s, expert_cfg=json.loads(a.expert_cfg),
                     task_cfg=json.loads(a.task_cfg))
    np.savez_compressed(a.out, **r)
    print(summary_line(r), "->", a.out, flush=True)


# ---- open-loop replay of the MuJoCo commands (where does the NRD leave MuJoCo under identical commands?) -------------
@torch.no_grad()
def replay_commands(tasks, mj, world_kind="nrd", ckpt=None, device="cuda", thr_mm=DIVERGE_MM, workers=4,
                    mj_python=None):
    """The joint targets of a MuJoCo run (``mj``, run_episodes output) replayed open loop in another world from the
    same start states -> dict: cube [N, T, 7] (float32), q [N, T, 6], first step where the cube centre leaves MuJoCo's
    by more than thr_mm (-1 never; within MuJoCo's episode), and cube error (mm) at the episode events (first GRASP
    trigger, firm lift, first PLACE trigger, end)."""
    from nedm.so101_policy.kin import SO101Kin
    dev = torch.device(device)
    B = len(tasks["seeds"])
    world = make_world(world_kind, B, dev, ckpt, workers=workers, kin=SO101Kin(dev),
                       geom=task_geometry(dev) if world_kind == "kin" else None, mj_python=mj_python)
    if world_kind == "mujoco":
        ws = world.reset(None, dict(q=tasks["start_q"], cube_pos=tasks["start_cube_pos"],
                                    cube_quat=tasks["start_cube_quat"]), warmup=True)
    else:
        ws = world.reset(None, start_state(tasks, dev))
    T = mj["u"].shape[1]
    u = torch.as_tensor(mj["u"], dtype=torch.float64, device=dev)
    cube, q = [], []
    for t in range(T):
        ws = world.step(u[:, t])
        cube.append(torch.cat((ws.cube_pos, ws.cube_quat), -1).float().cpu())
        q.append(ws.q.float().cpu())
    cube, q = torch.stack(cube, 1).numpy(), torch.stack(q, 1).numpy()
    if world_kind == "mujoco":
        world.close()
    L = mj["t_end"]
    fd = first_divergence(mj["cube"], cube, L, thr_mm)
    pt = phase_times(mj)
    err = np.linalg.norm(mj["cube"][..., :3] - cube[..., :3], axis=-1) * 1e3
    ev = {}
    b = np.arange(B)
    for k in ("grasp_trigger_s", "lift_s", "place_trigger_s", "end_s"):
        st = np.round(pt[k] / 0.02).astype(float) - (1 if k == "end_s" else 0)
        ok = np.isfinite(st)
        e = np.full(B, np.nan)
        e[ok] = err[b[ok], np.clip(st[ok].astype(int), 0, T - 1)]
        ev[k] = e
    at = np.maximum(fd, 0)
    return dict(cube=cube, q=q, first_div=fd, err_at=ev, mode_at_div=np.where(fd >= 0, mj["mode"][b, at], -9),
                phase_at_div=np.where(fd >= 0, mj["phase"][b, at], -9))


def cmd_replay(a):
    from nedm.so101_policy.decoder import MODES
    tasks = load_tasks(a.tasks)
    mj = dict(np.load(a.mj))
    if not np.array_equal(mj["seeds"], tasks["seeds"]):
        tasks = load_tasks(a.tasks, np.searchsorted(tasks["seeds"], mj["seeds"]))
    r = replay_commands(tasks, mj, a.world, a.ckpt, a.device)
    fd = r["first_div"]
    names = list(mj["phase_names"])
    by = {}
    for i in np.flatnonzero(fd >= 0):
        m = int(r["mode_at_div"][i])
        key = MODES[m] if m >= 0 else names[int(r["phase_at_div"][i])]
        by[key] = by.get(key, 0) + 1
    out = dict(mj=str(a.mj), world=a.world, ckpt=str(a.ckpt), n=int(len(fd)), diverged=int((fd >= 0).sum()),
               diverged_by_mj_mode=dict(sorted(by.items(), key=lambda kv: -kv[1])),
               err_mm_at={k: dict(p50=float(np.nanmedian(v)), p90=float(np.nanpercentile(v, 90)),
                                  p95=float(np.nanpercentile(v, 95))) for k, v in r["err_at"].items()},
               first_div_s=dict(p50=float(np.median(fd[fd >= 0]) * 0.02) if (fd >= 0).any() else None))
    if "success" in mj:
        s = mj["success"]
        out["diverged_given_mj_success"] = int((fd[s] >= 0).sum())
    Path(a.out).write_text(json.dumps(out, indent=1))
    np.savez_compressed(Path(a.out).with_suffix(".npz"), first_div=fd,
                        **{f"err_{k}": v for k, v in r["err_at"].items()})
    print(json.dumps(out, indent=1))


# ---- report ----------------------------------------------------------------------------------------------------------
def run_name(world, config, decoder, ckpt_tag=""):
    return f"{decoder}_{config}_{world}{ckpt_tag}"


def phase_times(r):
    """Per-episode event steps and phase durations (s) from a run: GRASP trigger, firm lift, PLACE trigger, end;
    time in approach (free before the first GRASP), GRASP macros, carry (hold), PLACE macros."""
    from nedm.so101_policy.decoder import M
    L = r["t_end"]
    T = r["mode"].shape[1]
    live = np.arange(T)[None] < L[:, None]
    out = {}
    if (r["mode"] >= 0).any():
        mode = r["mode"]
        g = (mode >= M["g_brake"]) & (mode <= M["g_lift"]) & live
        p = (mode >= M["p_brake"]) & live
        first = lambda m: np.where(m.any(1), m.argmax(1), -1)                                # noqa: E731
        out["grasp_trigger_s"] = first(g) * 0.02
        out["place_trigger_s"] = first(p) * 0.02
        out["grasp_macro_s"] = g.sum(1) * 0.02
        out["carry_s"] = ((mode == M["hold"]) & live).sum(1) * 0.02
        out["place_macro_s"] = p.sum(1) * 0.02
    else:
        names = list(r["phase_names"])
        ph = r["phase"]
        sel = lambda *n: np.isin(ph, [names.index(x) for x in n]) & live                    # noqa: E731
        first = lambda m: np.where(m.any(1), m.argmax(1), -1)                                # noqa: E731
        out["grasp_trigger_s"] = first(sel("descend")) * 0.02
        out["place_trigger_s"] = first(sel("lower")) * 0.02
        out["grasp_macro_s"] = sel("descend", "close", "grip", "lift").sum(1) * 0.02
        out["carry_s"] = sel("carry").sum(1) * 0.02
        out["place_macro_s"] = sel("lower", "release", "retreat").sum(1) * 0.02
    out["lift_s"] = np.where(r["lift_step"] >= 0, r["lift_step"] * 0.02, np.nan)
    out["end_s"] = L * 0.02
    for k in ("grasp_trigger_s", "place_trigger_s"):
        out[k] = np.where(out[k] < 0, np.nan, out[k])
    return out


def inhand_slip(r):
    """Per-episode largest change of the cube centre in the grasp frame (x closing axis, y jaw width, z up; mm) during
    the first carry (v2: decoder mode hold; v1: expert phase carry) against its first carry step [N, 3] (nan: no
    carry)."""
    from nedm.so101_policy.decoder import M
    L = r["t_end"]
    T = r["mode"].shape[1]
    live = np.arange(T)[None] < L[:, None]
    if (r["mode"] >= 0).any():
        hold = (r["mode"] == M["hold"]) & live
    else:
        hold = (r["phase"] == list(r["phase_names"]).index("carry")) & live
    out = np.full((len(L), 3), np.nan)
    for i in np.flatnonzero(hold.any(1)):
        idx = np.flatnonzero(hold[i])
        run = idx[: np.argmax(np.diff(np.r_[idx, 10 ** 9]) > 1) + 1]          # the first contiguous carry
        x = r["xg"][i, run]
        out[i] = np.abs(x - x[0]).max(0) * 1e3
    return out


def cause_names(r):
    """Per-episode outcome name: task end code, MuJoCo world ends with the collector reason."""
    from nedm.so101_policy.mujoco_env import end_names
    from nedm.so101_policy.task import ENDS
    names = np.array([ENDS[k] for k in r["end"]], dtype=object)
    if str(r["world"]) == "mujoco":
        en = end_names()
        mask = names == "world_invalid"
        names[mask] = ["world_invalid:" + en[int(x)] for x in r["mj_reason"][mask]]
    return names.astype(str)


def divergence_rows(mj, ot, idx, phase_names_mj):
    """For episodes idx: first step (>= DIVERGE_MM cube difference), MuJoCo phase / mode there, in-hand offsets."""
    from nedm.so101_policy.decoder import MODES
    L = np.minimum(mj["t_end"], ot["t_end"])
    T = min(mj["cube"].shape[1], ot["cube"].shape[1])
    fd = first_divergence(mj["cube"][:, :T], ot["cube"][:, :T], np.minimum(L, T))
    rows = []
    for i in idx:
        s = int(fd[i])
        row = dict(task=int(i), seed=int(mj["seeds"][i]), mj=cause_names(mj)[i], other=cause_names(ot)[i],
                   diverge_step=s, mj_end_s=float(mj["t_end"][i] * 0.02), other_end_s=float(ot["t_end"][i] * 0.02))
        if s >= 0:
            m = int(mj["mode"][i, s])
            row.update(diverge_s=s * 0.02, mj_phase=phase_names_mj[int(mj["phase"][i, s])],
                       mj_mode=MODES[m] if m >= 0 else "", label=LABELS[int(mj["label"][i, s])],
                       xg_mj_mm=(mj["xg"][i, s] * 1e3).round(1).tolist(), xg_other_mm=(ot["xg"][i, s] * 1e3).round(1).tolist(),
                       cube_z_mj_mm=float(mj["cube"][i, s, 2] * 1e3), cube_z_other_mm=float(ot["cube"][i, s, 2] * 1e3),
                       jaw_mj=float(mj["q"][i, s, 5]), jaw_other=float(ot["q"][i, s, 5]))
        rows.append(row)
    return rows, fd


def cmd_report(a):
    from nedm.so101_policy.task import RULES
    tasks = load_tasks(a.tasks)
    runs = {}
    for p in sorted(Path(a.runs).glob("*.npz")):
        r = dict(np.load(p, allow_pickle=False))
        if "end" in r and "world" in r:          # run files only (not replay outputs)
            runs[p.stem] = r
    rep = dict(tasks=str(a.tasks), n_tasks=int(len(tasks["seeds"])), runs={}, agreement={}, gate={})
    for name, r in runs.items():
        n = len(r["end"])
        c = cause_names(r)
        pt = phase_times(r)
        ff = r["first_fire"]
        rep["runs"][name] = dict(
            world=str(r["world"]), config=str(r["config"]), decoder=str(r["decoder"]), ckpt=str(r["ckpt"]), n=n,
            success15=float(r["success"].mean()), success15_ci=wilson(int(r["success"].sum()), n),
            success10=float(r["success10"].mean()), lifted=float(r["lifted"].mean()),
            causes={k: int(v) for k, v in zip(*np.unique(c, return_counts=True))},
            rule_fired_any={RULES[j]: int((ff[:, j] >= 0).sum()) for j in range(len(RULES)) if (ff[:, j] >= 0).any()},
            grasp_box_logged=int(r["box_log"].sum()),
            end_xy_mm=dict(p50=float(np.nanmedian(r["end_xy"][r["success"]]) * 1e3) if r["success"].any() else None,
                           p95=float(np.nanpercentile(r["end_xy"][r["success"]], 95) * 1e3) if r["success"].any() else None),
            times_s={k: dict(p50=float(np.nanmedian(v)) if np.isfinite(v).any() else None,
                             p95=float(np.nanpercentile(v, 95)) if np.isfinite(v).any() else None)
                     for k, v in pt.items()},
            close_offset_mm=dict(zip(("x", "y", "z"), [dict(p50=float(np.nanmedian(np.abs(r["close_x"][:, j])) * 1e3),
                                                            max=float(np.nanmax(np.abs(r["close_x"][:, j])) * 1e3))
                                                       if np.isfinite(r["close_x"][:, j]).any() else None
                                                       for j in range(3)])),
            close_yaw_deg_max=float(np.degrees(np.nanmax(r["close_x"][:, 3]))) if np.isfinite(r["close_x"][:, 3]).any() else None,
            noisy_decisions=dict(mean=float(r["noisy_n"].mean()), episodes=int((r["noisy_n"] > 0).sum())),
            harness_triggers=dict(grasp=int(r["harness_grasp"].sum()), place=int(r["harness_place"].sum())),
            retargeted=int((r["retarget_step"] >= 0).sum()), recoveries=int(r["recoveries"].sum()),
            wall_s=float(r["wall_s"]))
        sl = inhand_slip(r)
        if np.isfinite(sl).any():
            rep["runs"][name]["inhand_slip_mm"] = {ax: dict(p50=float(np.nanmedian(sl[:, j])),
                                                            p95=float(np.nanpercentile(sl[:, j], 95)),
                                                            max=float(np.nanmax(sl[:, j])))
                                                   for j, ax in enumerate(("x", "y", "z"))}
        if "counts" in r:
            rep["runs"][name]["decoder_flags"] = dict(zip(r["count_names"].tolist(), r["counts"].sum(0).tolist()))
    # agreement of each non-MuJoCo run with the MuJoCo run of the same decoder and config
    for name, r in runs.items():
        if str(r["world"]) == "mujoco":
            continue
        ref = f"{r['decoder']}_{r['config']}_mujoco"
        if ref not in runs:
            continue
        mj = runs[ref]
        if not np.array_equal(mj["seeds"], r["seeds"]):
            raise ValueError(f"{name}: tasks differ from {ref}")
        pt_m, pt_o = phase_times(mj), phase_times(r)
        both = mj["success"] & r["success"]
        dis = np.flatnonzero(mj["success"] != r["success"])
        rows, fd = divergence_rows(mj, r, dis, list(mj["phase_names"]))
        lab_div = {}
        for row in rows:
            key = f"{row.get('mj_phase', 'never')}/{row.get('mj_mode', '')}"
            lab_div[key] = lab_div.get(key, 0) + 1
        bothlift = mj["lifted"] & r["lifted"]
        bothplace = np.isfinite(pt_m["place_trigger_s"]) & np.isfinite(pt_o["place_trigger_s"])
        rep["agreement"][name] = dict(
            ref=ref, success15=table2x2(mj["success"], r["success"]), lift=table2x2(mj["lifted"], r["lifted"]),
            success10=table2x2(mj["success10"], r["success10"]),
            success15_given_both_lifted=table2x2(mj["success"][bothlift], r["success"][bothlift]),
            success15_given_both_placed=table2x2(mj["success"][bothplace], r["success"][bothplace]),
            cause_pairs={f"{x} | {y}": int(v) for (x, y), v in _pairs(cause_names(mj), cause_names(r)).items()},
            timing_diff_s={k: dict(p50=float(np.nanmedian(pt_o[k][both] - pt_m[k][both])) if both.any() else None,
                                   p95_abs=float(np.nanpercentile(np.abs(pt_o[k][both] - pt_m[k][both]), 95))
                                   if both.any() else None) for k in pt_m},
            final_cube_diff_mm_both_success=_final_diff(mj, r, both),
            divergence_by_mj_phase=lab_div, divergence_examples=rows[: a.examples],
            disagree_tasks=dis.tolist())
    # gates
    for dec_ in ("v2", "v1"):
        base = f"{dec_}_base_mujoco"
        if base in runs:
            s = runs[base]["success"]
            rep["gate"][f"{dec_}_expert_mujoco_success"] = dict(value=float(s.mean()), pass_=bool(s.mean() >= 0.95),
                                                               ci=wilson(int(s.sum()), len(s)))
        for name, ag in rep["agreement"].items():
            if not name.startswith(dec_ + "_"):
                continue
            cfg = name.split("_")[1]
            if cfg == "base":
                rep["gate"][f"{name}_agreement"] = dict(value=ag["success15"]["agree"],
                                                        pass_=bool(ag["success15"]["agree"] >= 0.95))
            elif CONFIGS[cfg]["noise"] or CONFIGS[cfg]["retarget"]:
                rep["gate"][f"{name}_noise"] = dict(lift_agree=ag["lift"]["agree"], place_agree=ag["success15"]["agree"],
                                                    pass_=bool(ag["lift"]["agree"] >= 0.97
                                                               and ag["success15"]["agree"] >= 0.95))
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    print(markdown(rep))
    if a.md:
        Path(a.md).write_text(markdown(rep))


def _pairs(x, y):
    out = {}
    for p, q in zip(x, y):
        if p != q:
            out[(p, q)] = out.get((p, q), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _final_diff(mj, r, both):
    if not both.any():
        return None
    b = np.arange(len(both))
    fm = mj["cube"][b, np.maximum(mj["t_end"] - 1, 0), :3]
    fo = r["cube"][b, np.maximum(r["t_end"] - 1, 0), :3]
    d = np.linalg.norm(fm - fo, axis=1)[both] * 1e3
    return dict(p50=float(np.median(d)), p95=float(np.percentile(d, 95)), max=float(d.max()))


def markdown(rep):
    """Compact tables of a report dict."""
    L = ["| run | success 15/15 | 10/10 | lifted | main causes | wall s |", "|---|---|---|---|---|---|"]
    for name, r in rep["runs"].items():
        top = ", ".join(f"{k} {v}" for k, v in sorted(r["causes"].items(), key=lambda kv: -kv[1])[:5])
        L.append(f"| {name} | {r['success15'] * 100:.1f} % | {r['success10'] * 100:.1f} % | {r['lifted'] * 100:.1f} % | "
                 f"{top} | {r['wall_s']:.0f} |")
    L += ["", "| other vs MuJoCo | success: n11 n10 n01 n00 | agree | phi | McNemar p | lift: n11 n10 n01 n00 | lift agree |",
          "|---|---|---|---|---|---|---|"]
    for name, ag in rep["agreement"].items():
        s, l = ag["success15"], ag["lift"]
        L.append(f"| {name} | {s['n11']} {s['n10']} {s['n01']} {s['n00']} | {s['agree'] * 100:.1f} % | {s['phi']:.2f} | "
                 f"{s['mcnemar_p']:.2g} | {l['n11']} {l['n10']} {l['n01']} {l['n00']} | {l['agree'] * 100:.1f} % |")
    L += ["", "| gate | value | pass |", "|---|---|---|"]
    for k, v in rep["gate"].items():
        val = ", ".join(f"{kk} {vv * 100:.1f} %" for kk, vv in v.items() if kk not in ("pass_", "ci"))
        L.append(f"| {k} | {val} | {'PASS' if v['pass_'] else 'FAIL'} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tasks")
    p.add_argument("--out", required=True), p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--n", type=int, default=1024), p.add_argument("--goal-seed", type=int, default=20261012)
    p = sub.add_parser("run")
    p.add_argument("--tasks", required=True), p.add_argument("--out", required=True)
    p.add_argument("--world", choices=WORLDS, required=True), p.add_argument("--config", choices=list(CONFIGS), default="base")
    p.add_argument("--decoder", choices=("v2", "v1"), default="v2"), p.add_argument("--ckpt")
    p.add_argument("--device", default="cpu"), p.add_argument("--workers", type=int, default=0)
    p.add_argument("--n", type=int, default=0), p.add_argument("--first", type=int, default=0)
    p.add_argument("--sigma", type=float, default=SIGMA), p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--expert-v1", default=str(EXPERT_V1)), p.add_argument("--horizon", type=int)
    p.add_argument("--threads", type=int, default=4), p.add_argument("--mj-python")
    p.add_argument("--goal-at-start", action="store_true", help="probe: goal = start cube pose")
    p.add_argument("--hold-s", type=float, default=0.0, help="probe: no PLACE before this carry time (s)")
    p.add_argument("--pause-s", type=float, default=0.0, help="probe: no free motion in the first pause_s of the carry")
    p.add_argument("--expert-cfg", default="{}", help="probe: ExpertConfig overrides (JSON)")
    p.add_argument("--task-cfg", default="{}", help="probe: TaskConfig overrides (JSON)")
    p.add_argument("--train-rules", action="store_true", help="training rules (grasp box terminates) instead of the "
                   "evaluation rules in every world")
    p = sub.add_parser("replay")
    p.add_argument("--tasks", required=True), p.add_argument("--mj", required=True), p.add_argument("--out", required=True)
    p.add_argument("--world", choices=("nrd", "kin", "mujoco"), default="nrd"), p.add_argument("--ckpt")
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("report")
    p.add_argument("--tasks", required=True), p.add_argument("--runs", required=True), p.add_argument("--out", required=True)
    p.add_argument("--md"), p.add_argument("--examples", type=int, default=12)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
