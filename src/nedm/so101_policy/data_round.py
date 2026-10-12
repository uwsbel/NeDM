"""NRD data round 2 of the SO101 + cube policy study (campaign ``expert_v2``): MuJoCo episodes of the privileged FSM
expert acting through decoder v2, written in the grasp NRD's training-data contract (``contact_nrd/collect``).

Why: the expert agreement gate (``~/so101_policy/docs/gate_expert.md``) found that the frozen and the round-1 grasp NRD
lose the held cube in the expert's carries (decoder-v2 commands: 10 Hz decision rhythm, re-aiming, goal changes,
noise), a command family the collector data lack. This round adds such episodes, with every failure MuJoCo produces.

Pipeline (three passes; ``scripts/so101_policy/collect_round.py`` runs them):

1. ``make_tasks``: one task per parent episode, a deterministic function of (master seed, set, index): start = the
   collector's start sampler + 0.3 s warm-up (``mujoco_env.sample_starts``, start seed ``start0 + index``), goals
   (``task.sample_goals``), and the family's options (table ``FAMILIES``; ``TASK_KEYS``). The table has the keys of
   ``gate_expert.py tasks`` too, so the gate harness can run on it.
2. ``run_rollouts`` (torch, MuJoCo workers): the expert + decoder v2 closed loop in ``MujocoWorlds`` (collector physics,
   ``labels="exact"``), the gate harness rules for noise / retarget (``gate_expert.py`` functions, imported), the
   still pause / hold, the cube move. Logged per world: the applied commands (float32, one per 20 ms step, plus the
   next command after the last step), the collector phase of each command (``collector_phase``), the 10 ms records
   and interval labels the env provides, the cube move (step, pose, post-move raw record), the env's end, the task
   outcome (``task.PickPlaceTask``, ``TaskConfig.evaluation()``; metadata only, it never stops a world). A world stops
   at its first collector end (``MujocoWorlds.end_reason``), ``tail`` steps after the expert is done (decoder back in
   free motion), or at its family's horizon.
3. ``label_rollouts`` (numpy + mujoco only): each episode is simulated once more by the collector's own
   ``so101_grasp.run_episode`` from the same warm-up start with the logged commands and phases (``so101_grasp.play``):
   this gives every array of the contract exactly as the collector computes it (states, action, contacts, phase,
   cube_in_grasp, visual_overlap, contact force / impulse / depth, palm force, the end cut and terminal reason,
   ``outcome``). A cube move splits the episode into segments: segment 1 ends at the record before the move; the move is
   applied to the MuJoCo data as ``MujocoWorlds.set_cube`` does (free-joint qpos / qvel, ``mj_forward``) and segment 2
   runs on from that state (``run_episode`` with ``Scene.reset`` disabled), so no data episode contains the teleport.
   Every segment is checked against pass 2's input: states and interval labels bit-equal to the env's 10 ms records
   (quaternion sign of a later segment aside), the same end. CPU MuJoCo is deterministic, so they agree; a mismatch is
   counted and the episode dropped.
4. ``rows_of`` / ``write_round``: segments longer than ``RECORDS`` (601 = 6 s, the contract's records_per_episode;
   ``merge_data.py`` concatenates arrays and needs equal record counts) are cut into chunks at records 0, 492, 984, ...
   (``CHUNK_STEP``): each chunk overlaps the previous one by 109 records = a 1 s rollout (50 steps) + the 4-state
   history, so every window of the long segment is in some chunk with its full history and horizon. The first chunk
   keeps the segment's family; later chunks go to family ``expert_tail`` (select: false: they start mid-episode with a
   moving arm, a start checkpoint selection should not score). The full-length segments (no chunks) of the validation
   episodes and of the gate2 set go to separate data directories (records_per_episode = their longest episode) for
   whole-episode evaluation. Splits are drawn per parent episode (``export.assign_splits``, group = parent), so the
   chunks and segments of one episode share a split. system.json = the collector's (``so101_grasp.system_json``: its
   17 families incl. the noisy_v1 ones) + ``FAMILIES`` appended, so ``scripts/contact_nrd/merge_data.py`` merges
   production_plus + noisy_v1 + expert_v2. Per-row metadata that the contract has no array for (parent, segment,
   chunk, start record, task outcome, options) is in ``episodes.npz`` next to unified_data.npz (training ignores it).

Seeds (never the sealed test 20261034, its or any collector campaign's episode seeds, test seeds 900000+ or the gate
seeds 1000-2023; ``mujoco_env.sample_starts`` refuses reserved seeds): master seed ``MASTER_SEED`` (all option draws),
start seeds ``START0`` + index (campaign) and ``GATE2_START0`` + index (gate2 set).

This module imports torch only inside the closed-loop functions; the labelling and writing work in numpy-only
interpreters (conda env nrdsim on luffy: scipy gives the collector's exact visual-overlap interpolation).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = "expert_v2"
MASTER_SEED = 20261211          # new: not pilot 20261008, production 20261021, test 20261034, convergence 20261085,
                                # extra 20261123, noisy_v1 20261147, critic noise 20261011, gate goals 20261012
START0 = 300_000                # start seed of campaign parent i (sample_starts): 300,000-312,999
GATE2_START0 = 400_000          # start seed of gate2 parent i: 400,000-401,023
RECORDS = 601                   # records_per_episode of the NRD training data (6 s at 10 ms)
CHUNK_STEP = 492                # records between chunk starts (even: chunks start on the 20 ms command grid)
DECISION = 5                    # control steps per expert decision (decoder v2)
CONTROL_DT = 0.02
SET_IDS = {"campaign": 0, "gate2": 1}

# Families of the campaign (each its own name; group "expert", select true except the tail chunks). ``share`` = the
# fraction of the parents; horizon in 20 ms steps.
FAMILIES = (
    dict(name="expert_base", code="B", share=0.30, horizon=400,
         note="privileged FSM expert (so101_policy.expert) through decoder v2, closed loop in CPU MuJoCo (collector "
              "scene and labels): collector start sampler and 0.3 s warm-up, goal task.sample_goals (annulus, 6-30 cm, "
              "yaw +-30 deg), lift 3-12 cm, carry speed class 0.20 / 0.30 / 0.35 m/s, 30 % with a 0.2-0.5 s still pause "
              "before the carry; every MuJoCo failure kept (data_round.py)"),
    dict(name="expert_retarget", code="R", share=0.20, horizon=500,
         note="as expert_base; the goal changes during the carry, once (60 %) or twice (40 %): 2-10 decisions (0.1 s) "
              "into the carry to a goal 4-15 cm from the first, then 3-10 decisions later 4-12 cm from the second"),
    dict(name="expert_noise", code="N", share=0.22, horizon=600,
         note="as expert_base with pre-tanh Gaussian decision noise (std 0.15-0.3) on the free-motion action in one "
              "phase, gate_expert.py rules: approach 15 %, alignment above the cube 30 % (harness GRASP trigger), carry "
              "35 % (height kept; 30 % of them with a goal change), place approach 20 % (harness PLACE trigger)"),
    dict(name="expert_disturb", code="D", share=0.12, horizon=500,
         note="as expert_base; once during the approach (gripper 3-6 cm above the cube top, before the GRASP trigger) "
              "the cube is moved 2-6 cm and turned +-45 deg, flat, at rest, on the start annulus, >= 2 cm from the "
              "gripper (MujocoWorlds.set_cube); the data episode is cut at the move: one row before it, one after"),
    dict(name="expert_long", code="L", share=0.16, horizon=600,
         note="as expert_base with long carries: goal 20-45 cm from the start on the annulus (to the workspace "
              "edges), lift 6-12 cm, no PLACE before 1-2 s of carry (a still hold at the goal: time in the air up to "
              "about the task's 3.5 s cap)"),
)
TAIL = dict(name="expert_tail", code="T",
            note="later chunks of expert segments longer than 6 s (records_per_episode 601): records 492-1092, "
                 "984-1200 of the segment (overlap 109 records = 1 s rollout + 4-state history with the chunk before); "
                 "the parent family's code (B R N D L) starts the variant; select false (starts mid-episode)")
FAMILY_NAMES = tuple(f["name"] for f in FAMILIES)
NOISE_PHASES = ("approach", "align", "carry", "place")          # gate_expert.LABELS[1:]
NOISE_SHARE = (0.15, 0.30, 0.35, 0.20)
SPEED_CLASSES = (dict(v_cruise=0.20, a_plan=0.4, w_cruise=1.0, alpha_plan=2.0),
                 dict(v_cruise=0.30, a_plan=0.6, w_cruise=1.5, alpha_plan=3.0),     # ExpertConfig defaults
                 dict(v_cruise=0.35, a_plan=1.0, w_cruise=2.0, alpha_plan=4.0))
SPEED_SHARE = (0.25, 0.5, 0.25)
TASK_KEYS = ("seeds", "start_q", "start_cube_pos", "start_cube_quat", "tries", "goal_xy", "goal_yaw", "goal2_xy",
             "goal2_yaw", "goal3_xy", "goal3_yaw", "family", "index", "spread_deg", "speed", "lift_h", "pause_s",
             "hold_s", "rt1", "rt2", "noise", "sigma", "dist_r", "dist_cand", "tail", "horizon")

# ---- round 3 (campaign policy_v3): MuJoCo rollouts of PPO policies (section "round 3" at the end of this module) ----
P_CAMPAIGN = "policy_v3"
P_MASTER_SEED = 20261312        # new (see MASTER_SEED's list + expert_v2 20261211, vision DAgger tasks 20261302)
P_START0 = 600_000              # start seed of campaign parent i: 600,000-612,999
PGATE_START0 = 410_000          # start seed of pgate task j: 410,000-410,511 (inside vec_env.RESERVED_SEEDS'
                                # evaluation band 400,000-499,999: load_starts refuses them as training starts)
P_SET_IDS = {"campaign": 0, "pgate": 1}
POLICY_DIR = Path(os.environ.get("NEDM_SO101_POLICY_DIR", Path.home() / "so101_policy" / "runs" / "round3" / "policies"))
POLICY_FAMILIES = (
    dict(name="policy_mjwarp", code="W",
         note="MuJoCo-Warp-trained PPO policies (recipe R_dc: decoder v2 default carry + grasp_gate, grasp_latch, "
              "abort_action; build4_ppo2.md) acting through their own decoder config in CPU MuJoCo (collector scene and "
              "labels), evaluation rules; deterministic (1/3) or sampled from their own Gaussian at 0.5 x / 1.0 x std "
              "(counter-based noise); collector start sampler + 0.3 s warm-up, goals task.sample_goals; the episode "
              "ends at the first collector end or 10-30 steps after the task's end (success, failure rule, 8 s "
              "time-out); data_round.py round 3"),
    dict(name="policy_kin", code="K", note="as policy_mjwarp; kinematic-world-trained PPO policies (R_dc)"),
    dict(name="policy_nrd", code="Q",
         note="as policy_mjwarp; PPO policies trained in the round-2 grasp NRD s62 (R_dc, still in the 30 mm stage: "
              "they show where PPO exploits the NRD)"),
)
POLICY_TAIL = dict(name="policy_tail", code="T",
                   note="later chunks of policy segments longer than 6 s (records_per_episode 601), as expert_tail; the "
                        "parent family's code (W K Q) starts the variant; select false (starts mid-episode)")
P_FAMILY_NAMES = tuple(f["name"] for f in POLICY_FAMILIES)
# the policies (id, family index, checkpoint file in POLICY_DIR, parent share of the campaign, source)
POLICIES = (
    dict(id="w5a", family=0, file="w5a_w5_dc_00121.pt", share=0.14,
         source="~/so101_policy/runs/ppo2/mjwarp/w5_dc/ckpt/model_00121.pt (CPU MuJoCo 96.5 %, 256 gate2)"),
    dict(id="w5b", family=0, file="w5b_w5_dc_s2_00123.pt", share=0.14,
         source="~/so101_policy/runs/ppo2/mjwarp/w5_dc_s2/ckpt/model_00123.pt (last)"),
    dict(id="kCa", family=1, file="kCa_kC_00060.pt", share=0.14,
         source="$B/ppo2/runs/kin/kC_shape_kl02/ckpt/model_00060.pt (CPU MuJoCo 93.0 %, 256 gate2)"),
    dict(id="kCb", family=1, file="kCb_kC_s2_00059.pt", share=0.14, source="$B/ppo2/runs/kin/kC_s2/ckpt/model_00059.pt (last)"),
    dict(id="kCc", family=1, file="kCc_kC_s3_00060.pt", share=0.14, source="$B/ppo2/runs/kin/kC_s3/ckpt/model_00060.pt (last)"),
    dict(id="nRc", family=2, file="nRc_nR2_s62_c_00155.pt", share=0.15,
         source="$B/ppo2/runs/nrd/nR2_s62_c/ckpt/model_00155.pt (last)"),
    dict(id="nRa", family=2, file="nRa_nR2_s62_a_00160.pt", share=0.15,
         source="$B/ppo2/runs/nrd/nR2_s62_a/ckpt/model_00160.pt (last; CPU MuJoCo 10.9 %, 256 gate2)"),
)
POLICY_IDS = tuple(p["id"] for p in POLICIES)
NOISE_LEVELS = (0.0, 0.5, 1.0)  # x the policy's own std (campaign: 1/3 each); pgate: PGATE_NOISE
NOISE_CODES = ("d", "h", "f")
PGATE_NOISE = (0.0, 1.0)
P_TASK_KEYS = ("seeds", "start_q", "start_cube_pos", "start_cube_quat", "tries", "goal_xy", "goal_yaw", "family",
               "policy", "noise_scale", "noise_level", "index", "task", "spread_deg", "tail")
P_META_EXTRA = ("policy", "noise_scale", "task", "task_steps", "ret")


# ---- the gate harness (functions reused as they are) ----------------------------------------------------------------
def gate():
    """scripts/so101_policy/gate_expert.py as a module (noise, harness triggers, latches, goals)."""
    name = "so101_gate_expert"
    if name not in sys.modules:
        path = REPO_ROOT / "scripts" / "so101_policy" / "gate_expert.py"
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def source_hashes():
    """sha256 of the code that makes the data (decoder, expert, task, worlds, harness, collector)."""
    files = ["src/nedm/so101_policy/" + f for f in ("data_round.py", "decoder.py", "expert.py", "task.py", "kin.py",
                                                     "mujoco_env.py", "world.py")]
    files += ["scripts/so101_policy/gate_expert.py", "scripts/so101_policy/collect_round.py",
              "src/nedm/contact_nrd/collect/so101_grasp.py", "src/nedm/contact_nrd/collect/export.py",
              "src/nedm/contact_nrd/collect/validate.py"]
    out = {}
    for f in files:
        p = REPO_ROOT / f
        if p.exists():
            out[f] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ---- tasks ----------------------------------------------------------------------------------------------------------
def family_counts(n, shares=None):
    """Parents per family for n parents (largest remainder on ``share``)."""
    shares = np.array([f["share"] for f in FAMILIES] if shares is None else shares, float)
    raw = shares / shares.sum() * n
    k = np.floor(raw).astype(int)
    k[np.argsort(-(raw - k))[: n - k.sum()]] += 1
    return k


def task_rng(master, set_id, index):
    return np.random.default_rng([int(master), int(set_id), int(index)])


def draw_options(rng, family):
    """The option draws of one parent (family index) -> dict (keys of TASK_KEYS minus the start and goal arrays)."""
    code = FAMILIES[family]["code"]
    o = dict(spread_deg=60.0 if rng.random() < 0.3 else 40.0,
             speed=int(rng.choice(3, p=SPEED_SHARE)), lift_h=rng.uniform(0.03, 0.12), pause_s=0.0, hold_s=0.0,
             rt1=-1, rt2=-1, noise=0, sigma=0.0, dist_r=-1.0, dist_cand=np.zeros((8, 3)),
             tail=int(rng.integers(10, 31)), horizon=FAMILIES[family]["horizon"])
    if code == "B" and rng.random() < 0.3:
        o["pause_s"] = rng.uniform(0.2, 0.5)
    if code == "R":
        o["rt1"] = int(rng.integers(2, 11))
        o["rt2"] = int(rng.integers(3, 11)) if rng.random() < 0.4 else -1
    if code == "N":
        o["noise"] = 1 + int(rng.choice(4, p=NOISE_SHARE))                   # gate LABELS code
        o["sigma"] = rng.uniform(0.15, 0.3)
        if NOISE_PHASES[o["noise"] - 1] == "carry" and rng.random() < 0.3:
            o["rt1"] = int(rng.integers(2, 11))
    if code == "D":
        o["dist_r"] = rng.uniform(0.003, 0.03)                               # TCP within this xy distance of the cube
        o["dist_cand"] = np.stack((rng.uniform(0.02, 0.06, 8), rng.uniform(-math.pi, math.pi, 8),
                                   rng.uniform(-math.pi / 4, math.pi / 4, 8)), -1)   # move (m), direction, turn (rad)
    if code == "L":
        o["lift_h"] = rng.uniform(0.06, 0.12)
        o["hold_s"] = rng.uniform(1.0, 2.0)      # + lift and lowering: time in the air about 2.2-3.8 s (cap 3.5 s)
    return o


def make_tasks(n, set_name="campaign", master=MASTER_SEED, start0=None, counts=None, idx=None, log=print):
    """The task table of the parents idx (default all n) of a set of n parents (``SET_IDS``) -> dict of numpy arrays
    (``TASK_KEYS`` and the settled start state ``state_*``: the keys of gate_expert.py's task files). Parent i: family
    by index range (``counts``, default ``family_counts(n)``), start seed start0 + i, options and goals from
    ``task_rng(master, set, i)`` only, so any sharding of idx gives the same rows."""
    import torch

    from nedm.so101_policy import task as T
    start0 = (START0 if set_name == "campaign" else GATE2_START0) if start0 is None else int(start0)
    counts = family_counts(n) if counts is None else np.asarray(counts)
    idx = np.arange(n) if idx is None else np.asarray(idx, np.int64)
    fam = np.repeat(np.arange(len(FAMILIES)), counts)[idx]
    seeds = start0 + idx
    if (seeds >= 900000).any() or ((seeds >= 1000) & (seeds <= 2023)).any():
        raise ValueError("start seeds overlap the test (900000+) or gate (1000-2023) seeds")
    sid = SET_IDS[set_name]
    m = len(idx)
    opts = [draw_options(task_rng(master, sid, i), int(f)) for i, f in zip(idx, fam)]
    out = {k: np.array([o[k] for o in opts]) for k in opts[0]}
    t0 = time.time()
    q, cpos, cquat, tries, state = _starts(seeds, out["spread_deg"])
    log(f"{m} starts in {time.time() - t0:.0f} s")
    for k, v in state.items():
        out[f"state_{k}"] = v
    cyaw, _ = T.cube_yaw_tilt(torch.as_tensor(out["state_cube_quat"]))
    cxy = torch.as_tensor(out["state_cube_pos"][:, :2])
    goals = {k: np.full((m,) + s, np.nan) for k, s in (("goal_xy", (2,)), ("goal_yaw", ()), ("goal2_xy", (2,)),
                                                       ("goal2_yaw", ()), ("goal3_xy", (2,)), ("goal3_yaw", ()))}
    for r, i in enumerate(idx):
        gen = torch.Generator().manual_seed(int(task_rng(master, sid, i).integers(2 ** 62)) + 1)
        code = FAMILIES[int(fam[r])]["code"]
        c, y = cxy[r:r + 1], cyaw[r:r + 1]
        gxy, gyaw = T.sample_goals(c, y, gen, dist=(0.20, 0.45) if code == "L" else T.GOAL_DIST)
        goals["goal_xy"][r], goals["goal_yaw"][r] = gxy[0].numpy(), float(gyaw[0])
        for key, sep in (("goal2", (0.04, 0.15)), ("goal3", (0.04, 0.12))):
            ref = gxy if key == "goal2" else torch.as_tensor(goals["goal2_xy"][r:r + 1])
            if not torch.isfinite(ref).all():
                break
            for _ in range(400):
                cxy2, cy2 = T.sample_goals(c, y, gen)
                d = float((cxy2 - ref).norm())
                if sep[0] <= d <= sep[1]:
                    goals[f"{key}_xy"][r], goals[f"{key}_yaw"][r] = cxy2[0].numpy(), float(cy2[0])
                    break
    out.update(goals, seeds=seeds.astype(np.int64), start_q=q, start_cube_pos=cpos, start_cube_quat=cquat, tries=tries,
               family=fam.astype(np.int64), index=idx.astype(np.int64), master_seed=np.int64(master),
               set_id=np.int64(sid), goal_seed=np.int64(master), n_set=np.int64(n))
    return out


def _starts(seeds, spread_deg):
    """Collector task starts (``mujoco_env.sample_starts``, 0.3 s warm-up) of start seeds [m] with a spread [m] (deg)
    -> (q [m, 6], cube_pos [m, 3], cube_quat [m, 4] before the warm-up, tries [m], settled state {q, qd, cube_pos,
    cube_quat, cube_v, cube_w: [m, d]})."""
    from nedm.so101_policy.mujoco_env import sample_starts
    seeds, spread_deg = np.asarray(seeds), np.asarray(spread_deg)
    m = len(seeds)
    q, cpos, cquat, tries = np.zeros((m, 6)), np.zeros((m, 3)), np.zeros((m, 4)), np.zeros(m, np.int64)
    state = {k: np.zeros((m, d)) for k, d in (("q", 6), ("qd", 6), ("cube_pos", 3), ("cube_quat", 4), ("cube_v", 3),
                                              ("cube_w", 3))}
    for spread in np.unique(spread_deg):
        sel = np.flatnonzero(spread_deg == spread)
        st = sample_starts(seeds[sel], spread_deg=float(spread))
        q[sel], cpos[sel], cquat[sel], tries[sel] = st["q"], st["cube_pos"], st["cube_quat"], st["tries"]
        for k in state:
            state[k][sel] = getattr(st["state"], k).numpy()
    return q, cpos, cquat, tries, state


def concat_tasks(parts):
    """Concatenate task tables of one set (make_tasks shards), in the given order."""
    n = [len(p["seeds"]) for p in parts]
    out = {}
    for k, v in parts[0].items():
        out[k] = np.concatenate([p[k] for p in parts]) if np.ndim(v) and len(v) == n[0] else v
    return out


def subset(tasks, idx):
    """Rows idx of a task table (per-task arrays only)."""
    n = len(tasks["seeds"])
    return {k: (v[idx] if np.ndim(v) and len(v) == n else v) for k, v in tasks.items()}


# ---- collector phases of the decoder modes --------------------------------------------------------------------------
# decoder mode -> collector phase (so101_grasp.PHASES); free and hold depend on the expert / the motion (collector_phase)
MODE_PHASE = {"free": None, "g_brake": "approach", "g_descend": "descend", "g_settle": "descend", "g_close": "close",
              "g_grip": "grip", "g_lift": "lift", "hold": None, "p_brake": "move", "p_settle": "hold",
              "p_lower": "lower", "p_open": "release", "p_retreat": "retreat"}


def phase_table():
    """[n modes] collector phase codes (-1 where it depends: free, hold)."""
    from nedm.contact_nrd.collect.so101_grasp import P
    from nedm.so101_policy.decoder import MODES
    return np.array([-1 if MODE_PHASE[m] is None else P[MODE_PHASE[m]] for m in MODES], np.int8)


def collector_phase(mode, expert_done, moving):
    """Collector phase code [B] (int8 tensor) of the command of one step: decoder mode [B] (during the step), expert done
    [B] bool (free motion after the place: idle, else approach), target moving [B] bool (hold: move / hold)."""
    import torch

    from nedm.contact_nrd.collect.so101_grasp import P
    from nedm.so101_policy.decoder import M
    tab = torch.as_tensor(phase_table(), device=mode.device)
    ph = tab[mode.long()].long()
    ph = torch.where(mode == M["free"], torch.where(expert_done, P["idle"], P["approach"]), ph)
    ph = torch.where(mode == M["hold"], torch.where(moving, P["move"], P["hold"]), ph)
    return ph.to(torch.int8)


# ---- pass 1: the closed loop in CPU MuJoCo ----------------------------------------------------------------------------
def qz_mul(dyaw, quat):
    """Rotate orientations quat [n, 4] (w x y z) by dyaw [n] about the world z axis (left multiplication)."""
    c, s = np.cos(dyaw / 2), np.sin(dyaw / 2)
    w, x, y, z = quat.T
    out = np.stack((c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w), -1)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def clip_annulus(xy, r=(0.15, 0.27), deg=(-75.0, 75.0), axis=(0.0388, 0.0)):
    """Clip points [n, 2] to the start annulus about the pan axis (collector sample_cube region)."""
    d = xy - np.asarray(axis)
    rad = np.clip(np.linalg.norm(d, axis=-1), *r)
    ang = np.clip(np.arctan2(d[:, 1], d[:, 0]), *np.radians(deg))
    return np.asarray(axis) + np.stack((np.cos(ang), np.sin(ang)), -1) * rad[:, None]


def run_rollouts(tasks, workers=0, python=None, device="cpu", speed=None, decoder_overrides=None,
                 expert_overrides=None, cpus=None, log=print):
    """Pass 1 (module docstring) for every task of ``tasks`` (one speed class: ``SPEED_CLASSES[speed]``, default the
    class of the first task) -> dict of numpy arrays: start (``seeds``, ``start_q``, ``start_cube_pos``,
    ``start_cube_quat``), ``commands`` [B, T+1, 6] float32 (applied; row n_steps = the next command), ``phases``
    [B, T+1] int8, ``n_steps`` [B], ``records`` [B, 2T+1, 25] float32 (the env's 10 ms records; record 0 = the
    settled start), ``record_contact`` [B, 2T, 3], cube move (``dist_step`` [B] (-1 none; the move happens before
    command dist_step), ``dist_pose`` [B, 13], ``jump_raw`` [B, 25] the record after the move), env end
    (``env_end``, ``env_end_step``), task outcome and harness counts."""
    import torch
    from dataclasses import replace

    from nedm.so101_policy import task as T
    from nedm.so101_policy.decoder import M, DecoderV2, DecoderV2Config
    from nedm.so101_policy.expert import PH, ExpertConfig, PickPlaceExpert
    from nedm.so101_policy.expert import cube_yaw_tilt as expert_yaw
    from nedm.so101_policy.kin import SO101Kin, yaw_of
    from nedm.so101_policy.mujoco_env import REST_Z, MujocoWorlds
    G = gate()
    dev = torch.device(device)
    B = len(tasks["seeds"])
    speed = int(tasks["speed"][0]) if speed is None else int(speed)
    if (np.asarray(tasks["speed"]) != speed).any():
        raise ValueError("run_rollouts takes the tasks of one speed class (ExpertConfig is shared by the batch)")
    t0 = time.time()
    f64 = dict(dtype=torch.float64, device=dev)
    tt = lambda k, **kw: torch.as_tensor(np.asarray(tasks[k]), **{"device": dev, **kw})     # noqa: E731
    horizon = tt("horizon", dtype=torch.long)
    TMAX = int(horizon.max())
    kin = SO101Kin(dev)
    geom = G.task_geometry(dev)
    world = MujocoWorlds(B, workers=workers, python=python, cpus=cpus)
    try:
        ws = world.reset(None, dict(q=tasks["start_q"], cube_pos=tasks["start_cube_pos"],
                                    cube_quat=tasks["start_cube_quat"]), warmup=True).to(dev)
        s0 = G.start_state(tasks, dev)
        dq, dc = (ws.q - s0.q).abs().max().item(), (ws.cube_pos - s0.cube_pos).abs().max().item()
        if dq > 1e-6 or dc > 1e-6:
            raise RuntimeError(f"MuJoCo warm-up start differs from the stored settled state ({dq:.2e}, {dc:.2e})")
        tlo, thi = world.tlo, world.thi
        cmd = np.zeros((B, TMAX + 1, 6), np.float32)
        phs = np.zeros((B, TMAX + 1), np.int8)
        rec = np.zeros((B, 2 * TMAX + 1, 25), np.float32)
        rcon = np.zeros((B, 2 * TMAX, 3), bool)
        rec[:, 0] = world.raw_state().numpy().astype(np.float32)
        tcfg = T.TaskConfig.evaluation(horizon=TMAX)
        task = T.PickPlaceTask(B, dev, kin=kin, geom=geom, cfg=tcfg, overlap_rules=False)
        all_ = torch.arange(B, device=dev)
        seeds = tt("seeds", dtype=torch.long)
        dec = DecoderV2(B, dev, kin=kin, cfg=replace(DecoderV2Config(), **(decoder_overrides or {})))
        dec.reset(all_, ws.q, seed=seeds)
        ecfg = replace(ExpertConfig(), **{**SPEED_CLASSES[speed], **(expert_overrides or {})})
        noise = tt("noise", dtype=torch.long)
        align = noise == G.LB["align"]
        if bool(align.any()):   # the gate's align config: abort distance and yaw tolerance of the harness trigger
            al = G.EXPERT_CFG["align"]
            ecfg.abort_off = torch.where(align, torch.tensor(al["abort_off"], **f64), torch.tensor(ecfg.abort_off, **f64))
            ecfg.yaw_ok = torch.where(align, torch.tensor(al["yaw_ok"], **f64), torch.tensor(ecfg.yaw_ok, **f64))
        ex = PickPlaceExpert(dec, ecfg)
        cl = dec.cfg.lift_range
        ex.a_lift = (2 * (tt("lift_h", **f64) - cl[0]) / (cl[1] - cl[0]) - 1).clamp(-1, 1)
        goal_xy, goal_yaw = tt("goal_xy", **f64), tt("goal_yaw", **f64)
        goal2_xy, goal2_yaw = tt("goal2_xy", **f64), tt("goal2_yaw", **f64)
        goal3_xy, goal3_yaw = tt("goal3_xy", **f64), tt("goal3_yaw", **f64)
        ex.reset(all_, ws, goal_xy, goal_yaw)
        task.reset(all_, ws, goal_xy, goal_yaw)
        lat = G.Latches(B, dev)
        sigma = tt("sigma", **f64)
        nseed = seeds * 7919 + 17 + 104729 * (int(np.asarray(tasks.get("set_id", 0))) + 1)
        rt1, rt2 = tt("rt1", dtype=torch.long), tt("rt2", dtype=torch.long)
        rt1 = torch.where(torch.isfinite(goal2_xy).all(-1), rt1, -1)      # no second / third goal drawn: no change
        rt2 = torch.where(torch.isfinite(goal3_xy).all(-1), rt2, -1)
        pause_n = torch.round(tt("pause_s", **f64) / (DECISION * CONTROL_DT)).long()
        hold_n = torch.round(tt("hold_s", **f64) / (DECISION * CONTROL_DT)).long()
        tail = tt("tail", dtype=torch.long)
        dist_r = tt("dist_r", **f64)
        cand = np.asarray(tasks["dist_cand"], float)
        L = lambda: torch.zeros(B, dtype=torch.long, device=dev)                            # noqa: E731
        carry_n, rt_step1, rt_step2 = L(), L() - 1, L() - 1
        noisy_n, h_grasp, h_place = L(), L(), L()
        active = torch.ones(B, dtype=torch.bool, device=dev)
        n_steps, done_at = L() - 1, L() - 1
        dist_step = np.full(B, -1, np.int64)
        dist_pose, jump_raw = np.zeros((B, 13)), np.zeros((B, 25), np.float32)
        z_hold = torch.full((B,), math.nan, **f64)
        meta = dict(task_end=L(), success=torch.zeros(B, dtype=torch.bool, device=dev), success10=None, lifted=None,
                    end_xy=torch.full((B,), math.nan, **f64), end_yaw=torch.full((B,), math.nan, **f64),
                    recoveries=L())
        meta["success10"], meta["lifted"] = meta["success"].clone(), meta["success"].clone()
        info = None
        for t in range(TMAX + 1):
            if t % DECISION == 0:
                k = t // DECISION
                # cube move (expert_disturb): before the expert acts, so its decision sees the new pose
                cand_w = active & (dist_r > 0) & torch.as_tensor(dist_step < 0, device=dev) & \
                    (ex.phase == PH["approach"]) & (dec.mode == M["free"]) & (t >= 10)
                if bool(cand_w.any()):
                    p, _ = kin.fk(ws.q.to(**f64))
                    cube = ws.cube_pos.to(**f64)
                    dxy = (p[:, :2] - cube[:, :2]).norm(dim=-1)
                    dz = p[:, 2] - (cube[:, 2] + 0.015)
                    go = cand_w & (dxy < dist_r) & (dz >= 0.03) & (dz <= 0.06) & \
                        (ws.cube_v.to(**f64).norm(dim=-1) < 0.01) & (cube[:, 2] < REST_Z + 0.001)
                    ix = torch.nonzero(go).flatten().cpu().numpy()
                    if len(ix):
                        ws, moved = _move_cubes(world, ws, ix, cand, kin, dev)
                        for i, pose in moved.items():
                            dist_step[i], dist_pose[i] = t, pose
                        if moved:
                            raw = world.raw_state().numpy().astype(np.float32)
                            mv = np.array(sorted(moved))
                            jump_raw[mv] = raw[mv]
                        ws = ws.to(dev)
                a = ex.act(ws)
                ph = ex.phase
                cube_xy = ws.cube_pos[:, :2].to(**f64)
                free, hold = dec.mode == M["free"], dec.mode == M["hold"]
                approach, carry = (ph == PH["approach"]) & free, (ph == PH["carry"]) & hold
                lab = lat.label(approach, carry, (dec.pos[:, :2] - cube_xy).norm(dim=-1) < G.R_LATCH,
                                (cube_xy - ex.goal_xy).norm(dim=-1) < G.R_LATCH)
                z_hold = torch.where(carry, torch.where(torch.isnan(z_hold), dec.pos[:, 2], z_hold), math.nan)
                noisy = (noise > 0) & (lab == noise) & (free | hold) & active
                if bool(noisy.any()):
                    sxy, syaw = G.brake_stop(dec)
                    cyaw, _ = expert_yaw(ws.cube_quat.to(**f64))
                    na = noisy & (noise == G.LB["align"])
                    ok_a = (na & approach & dec.armed & (a[:, 4] <= 0)
                            & (ws.cube_v.to(**f64).norm(dim=-1) < ex.cfg.cube_still)
                            & G.grasp_ready(sxy, syaw, cube_xy, cyaw))
                    npl = noisy & (noise == G.LB["place"])
                    ok_p = torch.zeros_like(ok_a)
                    if bool(npl.any()):
                        held, rel = ex._held(ws)
                        _, rg = kin.fk(ws.q.to(**f64))
                        ok_p = (npl & carry & held & dec.armed & (a[:, 4] <= 0)
                                & G.place_ready(sxy, syaw, rel, yaw_of(rg), cyaw, ex.goal_xy, ex.goal_yaw,
                                                ex.cfg.release_shift))
                    ex.phase = torch.where(ok_a, PH["grasp"], torch.where(ok_p, PH["place"], ex.phase))
                    a[:, 4] = torch.where(ok_a | ok_p, 1.0, a[:, 4])
                    h_grasp += ok_a.long()
                    h_place += ok_p.long()
                    noisy = noisy & ~(ok_a | ok_p)
                    zc = noisy & carry          # the expert's carry keeps the current height: plan back to the start
                    if bool(zc.any()):
                        vz = (z_hold - dec.pos[:, 2] - dec.vel[:, 2] * dec.T_d / 2) / dec.T_d
                        vz = vz.clamp(-ex.cfg.v_cruise, ex.cfg.v_cruise)
                        a[:, 2] = torch.where(zc, (vz * dec.T_d / dec.cfg.d_max).clamp(-1, 1), a[:, 2])
                    eps = G.hash_normal(nseed, k, 4)
                    a[:, :4] = G.pretanh_noise(a[:, :4], eps, sigma[:, None], noisy)
                    noisy_n += noisy.long()
                # still pause before the carry (expert_base) / no PLACE before hold_s of carry (expert_long)
                incarry = (dec.mode == M["hold"]) & ((ex.phase == PH["carry"]) | (ex.phase == PH["place"]))
                early = incarry & (carry_n < hold_n)
                ex.phase = torch.where(early & (ex.phase == PH["place"]), PH["carry"], ex.phase)
                a[:, 4] = torch.where(early, -1.0, a[:, 4])
                still = incarry & (carry_n < pause_n)
                a[:, :4] = torch.where(still[:, None], 0.0, a[:, :4])
                in_carry = ex.phase == PH["carry"]
                carry_n = torch.where(in_carry, carry_n + 1, carry_n)
                # goal changes during the carry (expert_retarget; carry noise with a goal change)
                for which, fire, gxy, gyw in (
                        (1, (rt1 >= 0) & (rt_step1 < 0) & (carry_n == rt1), goal2_xy, goal2_yaw),
                        (2, (rt2 >= 0) & (rt_step1 >= 0) & (rt_step2 < 0) & (carry_n == rt1 + rt2), goal3_xy, goal3_yaw)):
                    fire = fire & active & in_carry
                    if bool(fire.any()):
                        ix = fire.nonzero().flatten()
                        ex.goal_xy[ix], ex.goal_yaw[ix] = gxy[ix], gyw[ix]
                        task.set_goal(ix, gxy[ix], gyw[ix])
                        lat.place[ix] = False
                        if which == 1:
                            rt_step1 = torch.where(fire, t, rt_step1)
                        else:
                            rt_step2 = torch.where(fire, t, rt_step2)
                dec.decide(a, ws)
            mode = dec.mode.clone()
            u, _ = dec.step(ws.q)
            moving = (dec.vel.norm(dim=-1) > 1e-6) | (dec.yrate.abs() > 1e-6)
            ph_t = collector_phase(mode, ex.phase == PH["done"], moving)
            un = u.detach().cpu().double().numpy()
            cmd[:, t] = np.clip(un, tlo, thi).astype(np.float32)     # the applied value (MujocoWorlds.step)
            phs[:, t] = ph_t.cpu().numpy()
            if not bool(active.any()) or t == TMAX:
                break
            ws = world.step(u).to(dev)
            if not np.array_equal(world.applied.numpy(), cmd[:, t]):
                raise RuntimeError("applied command differs from the logged one")
            r = world.records.numpy().astype(np.float32)
            rec[:, 2 * t + 1], rec[:, 2 * t + 2] = r[:, 0], r[:, 1]
            rcon[:, 2 * t: 2 * t + 2] = world.record_contact.numpy()
            _, _, _, info = task.step(ws, dec.macro(), dec.macro_goal)
            ended = active & (world.end_reason.to(dev) != 0)
            fin = active & (ex.phase == PH["done"]) & (dec.mode == M["free"])
            done_at = torch.where(fin & (done_at < 0), t + 1, done_at)
            stop = ended | (active & (done_at >= 0) & (t + 1 - done_at >= tail)) | (active & (t + 1 >= horizon))
            if bool(stop.any()):
                n_steps = torch.where(stop, t + 1, n_steps)
                meta["task_end"] = torch.where(stop, info["end"], meta["task_end"])
                for key, src in (("success", info["end"] == T.EC["success"]), ("success10", info["success10"]),
                                 ("lifted", info["lifted"]), ("end_xy", info["end_xy_err"]),
                                 ("end_yaw", info["end_yaw_err"]), ("recoveries", ex.recoveries)):
                    meta[key] = torch.where(stop, src, meta[key])
                active = active & ~stop
            if t % 100 == 99:
                log(f"  t {t + 1}: stopped {int((~active).sum())}/{B}, {time.time() - t0:.0f} s", flush=True)
        es = world.end_step.numpy()                    # the first collector end, if it came before the world stopped
        ns = n_steps.cpu().numpy()
        env_end = np.where((es >= 1) & (es <= ns), world.end_reason.numpy(), 0).astype(np.int8)
        env_end_step = np.where(env_end != 0, es, -1)
    finally:
        world.close()
    if (ns < 1).any():
        raise RuntimeError("a world never stopped")
    out = dict(seeds=np.asarray(tasks["seeds"]), start_q=np.asarray(tasks["start_q"]),
               start_cube_pos=np.asarray(tasks["start_cube_pos"]), start_cube_quat=np.asarray(tasks["start_cube_quat"]),
               family=np.asarray(tasks["family"]), index=np.asarray(tasks["index"]), commands=cmd, phases=phs,
               n_steps=ns, records=rec, record_contact=rcon, dist_step=dist_step, dist_pose=dist_pose,
               jump_raw=jump_raw, env_end=env_end, env_end_step=env_end_step,
               rt_step1=rt_step1.cpu().numpy(), rt_step2=rt_step2.cpu().numpy(), noisy_n=noisy_n.cpu().numpy(),
               harness_grasp=h_grasp.cpu().numpy(), harness_place=h_place.cpu().numpy(),
               speed=np.full(B, speed, np.int64), wall_s=np.float64(time.time() - t0),
               decoder_overrides=np.array(json.dumps(decoder_overrides or {})),
               expert_overrides=np.array(json.dumps(expert_overrides or {})))
    out.update({k: v.cpu().numpy() for k, v in meta.items()})
    for k in ("noise", "sigma", "rt1", "rt2", "pause_s", "hold_s", "lift_h", "dist_r", "spread_deg", "tries", "horizon"):
        out[k] = np.asarray(tasks[k])
    return out


def _move_cubes(world, ws, ix, cand, kin, dev):
    """Move the cubes of worlds ix (numpy) to the first valid candidate of ``cand`` [B, 8, 3] (move m, direction,
    turn): flat at rest at REST_Z, on the annulus (clipped), moved >= 2 cm, >= 2 cm from the gripper and jaw collision
    points (``kin.box_distance``). -> (new WorldState, {world: pose [13]})."""
    import torch

    from nedm.so101_policy.mujoco_env import REST_Z
    cpos = ws.cube_pos[ix].double().cpu().numpy()
    cq = ws.cube_quat[ix].double().cpu().numpy()
    q = ws.q[torch.as_tensor(ix)].double()
    chosen = {}
    for j in range(cand.shape[1]):
        todo = [n for n in range(len(ix)) if int(ix[n]) not in chosen]
        if not todo:
            break
        sel = np.array(todo)
        d, ang, dyaw = cand[ix[sel], j].T
        xy = clip_annulus(cpos[sel, :2] + d[:, None] * np.stack((np.cos(ang), np.sin(ang)), -1))
        moved = np.linalg.norm(xy - cpos[sel, :2], axis=-1) >= 0.02
        pos = np.concatenate((xy, np.full((len(sel), 1), REST_Z)), -1)
        quat = qz_mul(dyaw, cq[sel])
        clear = kin.box_distance(q[torch.as_tensor(sel)].to(kin.dtype), torch.as_tensor(pos, dtype=kin.dtype, device=q.device),
                                 torch.as_tensor(quat, dtype=kin.dtype, device=q.device), 0.015).cpu().numpy() >= 0.02
        for n in np.flatnonzero(moved & clear):
            chosen[int(ix[sel[n]])] = np.concatenate((pos[n], quat[n], np.zeros(6)))
    if chosen:
        rows = np.array(sorted(chosen))
        poses = np.stack([chosen[i] for i in rows])
        ws = world.set_cube(torch.as_tensor(rows), torch.as_tensor(poses[:, 0:3]), torch.as_tensor(poses[:, 3:7]))
    return ws, chosen


# ---- pass 2: the collector's own labels -------------------------------------------------------------------------------
_SCENE = None


def scene():
    """The collector Scene of this process with the visual-overlap grids (cache: mujoco_env.install_overlap)."""
    global _SCENE
    if _SCENE is None:
        from nedm.contact_nrd.collect import so101_grasp as g
        from nedm.so101_policy.mujoco_env import install_overlap
        s = g.Scene(g.load_config())
        install_overlap(s)
        _SCENE = s
    return _SCENE


def apply_cube_move(s, pose):
    """MujocoWorlds.set_cube on the collector Scene's MjData: free-joint qpos (pos, unit quat) and qvel (world v, spin
    to the body frame), mj_forward."""
    import mujoco

    from nedm.contact_nrd.collect import so101_grasp as g
    d = s.d
    x = np.asarray(pose, float)
    q = x[3:7] / np.linalg.norm(x[3:7])
    d.qpos[s.cq: s.cq + 3], d.qpos[s.cq + 3: s.cq + 7] = x[0:3], q
    d.qvel[s.cv: s.cv + 3] = x[7:10]
    d.qvel[s.cv + 3: s.cv + 6] = g.quat_mat(q).T @ x[10:13]
    mujoco.mj_forward(s.m, d)


def label_episode(s, ep, check=None):
    """One logged episode -> list of segment dicts (run_episode output + ``outcome`` + ``start_step``), the collector's
    labels (module docstring, pass 2). ep: q0 [6], cube_pos [3], cube_quat [4] (pre-settle), commands [S+1, 6]
    float32, phases [S+1] int8, n_steps S, events [(step, pose [13])] (cube moves). check: pass-1 logs (``records``
    [2S+1, 25] float32, ``record_contact`` [2S, 3], ``jump_raw`` {step: [25]}, ``env_end``) -> each segment gets
    ``check`` = dict(states, contacts, end) booleans. A numerical failure gives {"rejected": ...} as the last item."""
    from nedm.contact_nrd.collect import so101_grasp as g
    S = int(ep["n_steps"])
    events = sorted((int(t), np.asarray(p, float)) for t, p in ep.get("events", []))
    cuts = [0] + [t for t, _ in events] + [S]
    out = []
    for j in range(len(cuts) - 1):
        a, b = cuts[j], cuts[j + 1]
        cmd = np.asarray(ep["commands"][a: b + 1], np.float64)
        ph = np.asarray(ep["phases"][a: b + 1], np.int8)
        plan = dict(q0=np.asarray(ep["q0"], float), cube=dict(pos=np.asarray(ep["cube_pos"], float),
                                                              quat=np.asarray(ep["cube_quat"], float), spawn=False),
                    script=g.play(cmd, ph))
        if j == 0:
            r = g.run_episode(s, plan, 2 * (b - a) + 1)
        else:
            apply_cube_move(s, events[j - 1][1])
            s.reset = lambda *args, **kw: None      # segment 2 runs on from the moved state (no warm-up)
            try:
                r = g.run_episode(s, plan, 2 * (b - a) + 1)
            finally:
                del s.reset
        if r is None or isinstance(r, str):
            out.append(dict(rejected="numerical" if r is None else "ends_at_start", segment=j, start_step=a))
            break
        r["outcome"] = g.outcome(r, s.half, "expert")
        r["start_step"], r["segment"] = a, j
        if check is not None:
            r["check"] = _check_segment(r, a, b, j, j == len(cuts) - 2, check)
        out.append(r)
        if r["terminal"] != 0 or len(r["action"]) < 2 * (b - a) + 1:
            break
    return out


def _check_segment(r, a, b, j, last, check):
    """Pass 1 vs pass 2 of one segment (module docstring): states (float32 records), interval labels, the end."""
    L = len(r["action"])
    arm = r["states"]["arm"].astype(np.float32)
    cube = r["states"]["cube"].astype(np.float32)
    rec = np.asarray(check["records"])[2 * a: 2 * a + L].copy()
    if j > 0:   # the segment's first record is the state after the move; pass 1 logged it apart
        rec[0] = np.asarray(check["jump_raw"][a], np.float32)
    q1, q2 = rec[:, 15:19], cube[:, 3:7]
    sign = 1.0 if j == 0 or np.dot(q1[0], q2[0]) >= 0 else -1.0      # a later segment: w >= 0 at its own start
    states = bool(len(rec) == L and np.array_equal(rec[:, :12], arm) and np.array_equal(rec[:, 12:15], cube[:, :3])
                  and np.array_equal(sign * q1, q2) and np.array_equal(rec[:, 19:], cube[:, 7:]))
    contacts = bool(np.array_equal(np.asarray(check["record_contact"])[2 * a: 2 * a + L - 1], r["contacts"].astype(bool)))
    full = L == 2 * (b - a) + 1
    if last:    # the episode's end: the env's first collector end (0 = none) and the collector's terminal reason agree
        end = int(r["terminal"]) == int(check["env_end"]) and (r["terminal"] != 0 or full)
    else:
        end = r["terminal"] == 0 and full
    return dict(states=states, contacts=contacts, end=bool(end))


def episodes_of(rollouts, k):
    """Episode k of a pass-1 rollout file (dict) -> (ep, check) for label_episode."""
    S = int(rollouts["n_steps"][k])
    events = []
    if int(rollouts["dist_step"][k]) >= 0:
        events.append((int(rollouts["dist_step"][k]), rollouts["dist_pose"][k]))
    ep = dict(q0=rollouts["start_q"][k], cube_pos=rollouts["start_cube_pos"][k], cube_quat=rollouts["start_cube_quat"][k],
              commands=rollouts["commands"][k, : S + 1], phases=rollouts["phases"][k, : S + 1], n_steps=S, events=events)
    check = dict(records=rollouts["records"][k, : 2 * S + 1], record_contact=rollouts["record_contact"][k, : 2 * S],
                 jump_raw={int(t): rollouts["jump_raw"][k] for t, _ in events}, env_end=rollouts["env_end"][k])
    return ep, check


def _label_job(jobs):
    """Worker: [(k, ep, check)] -> [(k, segments)] (label_episode on this process's Scene)."""
    s = scene()
    return [(k, label_episode(s, ep, check)) for k, ep, check in jobs]


def label_rollouts(path, workers=1, chunk=8, pool=None, log=print):
    """Pass 2 for every episode of a pass-1 rollout file (.npz) on a process pool (``pool``, or a new one of
    ``workers`` spawned processes; each worker builds one Scene) -> list of segment lists (episode order)."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    with np.load(path) as f:
        ro = {key: f[key] for key in ("n_steps", "dist_step", "dist_pose", "start_q", "start_cube_pos",
                                       "start_cube_quat", "commands", "phases", "records", "record_contact",
                                       "jump_raw", "env_end")}
    n = len(ro["n_steps"])
    eps = [(k,) + episodes_of(ro, k) for k in range(n)]
    jobs = [eps[a: a + chunk] for a in range(0, n, chunk)]
    res = {}
    t0 = time.time()
    if pool is not None:
        for part in pool.map(_label_job, jobs):
            res.update(dict(part))
    elif workers <= 1:
        for j in jobs:
            res.update(dict(_label_job(j)))
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as own:
            for part in own.map(_label_job, jobs):
                res.update(dict(part))
    log(f"{path}: {n} episodes labelled in {time.time() - t0:.0f} s")
    return [res[k] for k in range(n)]


# ---- rows: segments -> chunks of the training contract -----------------------------------------------------------------
def chunk_starts(L, records=RECORDS, step=CHUNK_STEP):
    """Record offsets of the chunks of a segment of L records (one chunk if L <= records)."""
    starts = [0]
    while starts[-1] + records < L:
        starts.append(starts[-1] + step)
    return starts


def slice_segment(r, a, n):
    """Records a .. a+n-1 of a labelled segment as a pack_episodes episode dict (states, action, contacts,
    per_record, per_interval); the quaternions keep their signs (all flipped if w < 0 at the first record)."""
    arm = np.asarray(r["states"]["arm"][a: a + n])
    cube = np.asarray(r["states"]["cube"][a: a + n]).copy()
    if cube[0, 3] < 0:
        cube[:, 3:7] = -cube[:, 3:7]
    return dict(states=dict(arm=arm, cube=cube), action=np.asarray(r["action"][a: a + n]),
                contacts=np.asarray(r["contacts"][a: a + n - 1]),
                per_record={k: np.asarray(v[a: a + n]) for k, v in r["per_record"].items()},
                per_interval={k: np.asarray(v[a: a + n - 1]) for k, v in r["per_interval"].items()})


def variant_of(family, ro, k, segment, chunk):
    """Variant string (<= 16 characters): family code, option, segment / chunk."""
    code = FAMILIES[family]["code"]
    if code == "N":
        det = f"{('appr', 'align', 'carry', 'place')[int(ro['noise'][k]) - 1]}{round(100 * float(ro['sigma'][k]))}"
        det += "rt" if int(ro["rt1"][k]) >= 0 else ""
    elif code == "R":
        det = f"rt{1 + int(int(ro['rt2'][k]) >= 0)}"
    elif code == "D":
        det = ("pre", "post")[segment] if int(ro["dist_step"][k]) >= 0 else "none"
    elif code == "L":
        det = f"h{round(10 * float(ro['hold_s'][k]))}"
    else:
        det = "pause" if float(ro["pause_s"][k]) > 0 else "plain"
    v = f"{code}:{det}v{int(ro['speed'][k])}"
    return v + (f"/c{chunk}" if chunk else "")


def rows_of(ro, k, segments, family_offset, parent_id, chunked=True, records=RECORDS, step=CHUNK_STEP,
            n_families=None, variant=None, extra_meta=()):
    """Rows (pack_episodes episode dicts + per-row metadata) of episode k of a rollout dict. chunked: cut segments into
    ``records``-record chunks every ``step`` records (training folder; later chunks -> the campaign's tail family,
    index ``n_families`` after its first: expert_tail by default); else one row per segment. variant: the variant
    function (default ``variant_of``, the expert families); extra_meta: more per-episode keys of ro copied into the
    row's meta (round 3: policy, noise scale)."""
    n_families = len(FAMILIES) if n_families is None else int(n_families)
    variant = variant_of if variant is None else variant
    rows = []
    fam = int(ro["family"][k])
    for r in segments:
        if "rejected" in r:
            continue
        L = len(r["action"])
        starts = chunk_starts(L, records, step) if chunked else [0]
        for c, a in enumerate(starts):
            n = min(records, L - a) if chunked else L
            if n < 2:
                continue
            ep = slice_segment(r, a, n)
            last = a + n == L
            o = r["outcome"]
            f_idx = family_offset + (fam if c == 0 else n_families)
            ep["scalars"] = dict(
                family=np.int16(f_idx), episode_id=np.int64(-1), group_id=np.int64(parent_id),
                seed=np.int64(ro["seeds"][k]), terminal_reason=np.int8(r["terminal"] if last else 0),
                outcome=np.int8(o["outcome"]), max_lift_m=np.float32(o["max_lift_m"]),
                hold_slip_mm=np.float32(o["hold_slip_mm"]), carry_slip_mm=np.float32(o["carry_slip_mm"]),
                family_index=np.int64(ro["index"][k]), plan_tries=np.int16(ro["tries"][k]),
                unsupported_max_n=np.float32(np.asarray(r["other_force"]).max()),
                variant=variant(fam, ro, k, int(r["segment"]), c))
            ep["meta"] = dict(parent=int(parent_id), segment=int(r["segment"]), chunk=c, chunk_start=int(a),
                              segment_start_step=int(r["start_step"]), segment_records=int(L),
                              family=fam, task_end=int(ro["task_end"][k]), success=bool(ro["success"][k]),
                              success10=bool(ro["success10"][k]), lifted=bool(ro["lifted"][k]),
                              end_xy=float(ro["end_xy"][k]), end_yaw=float(ro["end_yaw"][k]),
                              n_steps=int(ro["n_steps"][k]), env_end=int(ro["env_end"][k]),
                              noise=int(ro["noise"][k]), sigma=float(ro["sigma"][k]), rt1=int(ro["rt1"][k]),
                              rt2=int(ro["rt2"][k]), rt_step1=int(ro["rt_step1"][k]), rt_step2=int(ro["rt_step2"][k]),
                              dist_step=int(ro["dist_step"][k]), speed=int(ro["speed"][k]),
                              recoveries=int(ro["recoveries"][k]), noisy_n=int(ro["noisy_n"][k]))
            for key in extra_meta:
                ep["meta"][key] = np.asarray(ro[key][k]).item()
            rows.append(ep)
    return rows


# ---- system.json, splits and the data directory -----------------------------------------------------------------------
def round_system(s=None, campaign=CAMPAIGN):
    """system.json: the collector's (all collector families, incl. noisy_v1's) + FAMILIES + the tail family; campaign
    policy_v3 (round 3) appends POLICY_FAMILIES + POLICY_TAIL after these, so merge_data.py's order rule holds for
    production_plus + noisy_v1 + expert_v2 + policy_v3."""
    from nedm.contact_nrd.collect import so101_grasp as g
    s = s or scene()
    system = g.system_json(s, s.cfg)
    for f in FAMILIES:
        system["families"].append(dict(name=f["name"], group="expert", select=True, note=f["note"]))
    system["families"].append(dict(name=TAIL["name"], group="expert", select=False, note=TAIL["note"]))
    if campaign == P_CAMPAIGN:
        for f in POLICY_FAMILIES:
            system["families"].append(dict(name=f["name"], group="policy", select=True, note=f["note"]))
        system["families"].append(dict(name=POLICY_TAIL["name"], group="policy", select=False, note=POLICY_TAIL["note"]))
    elif campaign != CAMPAIGN:
        raise ValueError(f"campaign {campaign!r}: {CAMPAIGN} or {P_CAMPAIGN}")
    return system


def family_offset(system, campaign=CAMPAIGN):
    names = [f["name"] for f in system["families"]]
    return names.index((P_FAMILY_NAMES if campaign == P_CAMPAIGN else FAMILY_NAMES)[0])


def parent_splits(family, parent, n_val, seed=MASTER_SEED):
    """Split per parent: family-stratified (export.assign_splits, groups = parents), about n_val validation parents
    in all (fractions (1 - n_val / n, n_val / n))."""
    from nedm.contact_nrd.collect import export
    n = len(family)
    frac = float(n_val) / n
    return export.assign_splits(np.asarray(family), np.asarray(parent), fractions=(1 - frac, frac), seed=seed)


META_KEYS = ("parent", "segment", "chunk", "chunk_start", "segment_start_step", "segment_records", "family", "task_end",
             "success", "success10", "lifted", "end_xy", "end_yaw", "n_steps", "env_end", "noise", "sigma", "rt1", "rt2",
             "rt_step1", "rt_step2", "dist_step", "speed", "recoveries", "noisy_n")


def write_round(out_dir, rows, splits, system, records, manifest, meta_keys=META_KEYS):
    """Pack rows (rows_of) with their splits [n rows] into a data directory (export.write_dataset: unified_data.npz,
    system.json, manifest.json) plus episodes.npz (per-row metadata, ``meta_keys``). records = records_per_episode."""
    from nedm.contact_nrd.collect import export
    from nedm.contact_nrd.collect import so101_grasp as g
    system = dict(system, records_per_episode=int(records))
    arrays = export.pack_episodes(rows, system)
    n = len(arrays["lengths"])
    arrays["episode_id"] = np.arange(n, dtype=np.int64)
    arrays["variant"] = arrays["variant"].astype("U16")
    arrays["splits"] = np.asarray(splits, np.int8)
    man = dict(manifest)
    man["summary"] = g.summarize(system, arrays)
    info = export.write_dataset(out_dir, system, arrays, man)
    meta = {k: np.array([r["meta"][k] for r in rows]) for k in meta_keys}
    np.savez(Path(out_dir) / "episodes.npz", **meta)
    return info


# ---- verification helpers -----------------------------------------------------------------------------------------------
STORED_KEYS = ("arm", "cube", "action", "contacts", "phase", "cube_in_grasp", "visual_overlap", "contact_force",
               "contact_impulse", "palm_force", "contact_depth")
STORED_SCALARS = ("terminal_reason", "outcome", "max_lift_m", "hold_slip_mm", "carry_slip_mm", "unsupported_max_n")


def compare_stored(stored, packed, i, j):
    """Row i of stored arrays vs row j of packed arrays over the valid records -> {key: bit-equal} (NaN == NaN)."""
    L = int(stored["lengths"][i])
    out = {"lengths": L == int(packed["lengths"][j])}
    for k in STORED_KEYS:
        n = L - 1 if k in ("contacts", "contact_force", "contact_impulse", "palm_force", "contact_depth") else L
        a, b = np.asarray(stored[k][i][:n]), np.asarray(packed[k][j][:n])
        out[k] = bool(a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))
    for k in STORED_SCALARS:
        a, b = np.asarray(stored[k][i]), np.asarray(packed[k][j])
        out[k] = bool(np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))
    return out


# ---- round 3: MuJoCo rollouts of PPO policies (campaign policy_v3) ------------------------------------------------------
# Design (report ~/so101_policy/docs/build6_round3.md, phase 0; written before the collection):
# - Actor: a PPO checkpoint (ppo.load_policy) acting through ITS OWN decoder config and observation, in the env that
#   eval_policy.py run builds for world mujoco (policy_env_config: eval_policy.checkpoint_env, evaluation rules,
#   default 15 mm / 15 deg, decoder macro seeds = task seeds). Pre-tanh action = mean + k std eps, k = the task's
#   noise_scale (0 = deterministic, bit for bit eval_policy's policy_act_fn), std = the checkpoint's own per-column std
#   (all columns, as PPO samples), eps = gate_expert.hash_normal(task seed * 7919 + NOISE_SALT, decision index).
# - Logged per world (pass 1, like run_rollouts): the applied float32 commands (one per 20 ms step + the next one), the
#   collector phase of each command (collector_phase: decoder mode during the step; free motion = approach until a
#   PLACE macro ran, then idle until the next GRASP), the env's 10 ms records and interval labels.
# - Episode end: the first collector end (MujocoWorlds.end_step), else the task's end step (success, a failure rule,
#   the time-out) + the task's tail (10-30 steps), whichever is first. The policy keeps acting in the tail.
# - Pass 2 and the writer are round 2's (label_rollouts, rows_of with POLICY_FAMILIES / policy_variant_of).
NOISE_SALT = 271_828


def policy_cells(n):
    """Parents per (policy, noise level) cell of a campaign of n parents, cells in the order (policy 0: levels 0, 1, 2;
    policy 1: ...) -> int array: parents per policy by largest remainder on ``share``, then per level equally (largest
    remainder)."""
    per = family_counts(n, [p["share"] for p in POLICIES])
    return np.concatenate([family_counts(int(m), np.ones(len(NOISE_LEVELS))) for m in per])


def policy_path(pid, policy_dir=None):
    """Checkpoint path of a policy (id string or index into POLICIES)."""
    i = POLICY_IDS.index(pid) if isinstance(pid, str) else int(pid)
    return Path(policy_dir or POLICY_DIR).expanduser() / POLICIES[i]["file"]


def check_policy_seeds(seeds, set_name):
    """Start seeds of a policy set: never the test (900000+), the gate (1000-2023) or the sealed-test master; campaign
    seeds outside the evaluation band 400,000-499,999, pgate seeds inside it."""
    s = np.asarray(seeds, np.int64)
    if (s >= 900000).any() or ((s >= 1000) & (s <= 2023)).any() or (s == 20261034).any():
        raise ValueError("start seeds overlap the test (900000+), the gate (1000-2023) or the sealed test master")
    band = (s >= 400000) & (s <= 499999)
    if set_name == "campaign" and band.any():
        raise ValueError("campaign start seeds inside the evaluation band 400,000-499,999")
    if set_name == "pgate" and not band.all():
        raise ValueError("pgate start seeds must lie inside the evaluation band 400,000-499,999")


def make_policy_tasks(n, set_name="campaign", master=P_MASTER_SEED, start0=None, idx=None, log=print):
    """Task table of the rows idx (default all n) of a policy set of n rows (``P_SET_IDS``) -> dict of numpy arrays
    (``P_TASK_KEYS`` + the settled start ``state_*``: also a gate_expert.py / eval_policy.py task file). Row i: start
    seed start0 + i (collector sampler + 0.3 s warm-up), spread 40 deg (60 deg with p 0.3), tail 10-30 steps and the
    goal (task.sample_goals) from task_rng(master, set, i) only, so any sharding of idx gives the same rows. Campaign
    rows get (policy, noise level) by index range (``policy_cells``); pgate rows get policy -1 (``expand_pgate``)."""
    import torch

    from nedm.so101_policy import task as T
    sid = P_SET_IDS[set_name]
    start0 = (P_START0 if set_name == "campaign" else PGATE_START0) if start0 is None else int(start0)
    idx = np.arange(n) if idx is None else np.asarray(idx, np.int64)
    seeds = start0 + idx
    check_policy_seeds(seeds, set_name)
    m = len(idx)
    spread, tail, gseed = np.zeros(m), np.zeros(m, np.int64), np.zeros(m, np.int64)
    for r, i in enumerate(idx):
        rng = task_rng(master, sid, i)
        spread[r] = 60.0 if rng.random() < 0.3 else 40.0
        tail[r] = int(rng.integers(10, 31))
        gseed[r] = int(rng.integers(2 ** 62)) + 1
    t0 = time.time()
    q, cpos, cquat, tries, state = _starts(seeds, spread)
    log(f"{m} starts in {time.time() - t0:.0f} s")
    out = {f"state_{k}": v for k, v in state.items()}
    cyaw, _ = T.cube_yaw_tilt(torch.as_tensor(out["state_cube_quat"]))
    cxy = torch.as_tensor(out["state_cube_pos"][:, :2])
    gxy, gyaw = np.zeros((m, 2)), np.zeros(m)
    for r in range(m):
        g_xy, g_yaw = T.sample_goals(cxy[r:r + 1], cyaw[r:r + 1], torch.Generator().manual_seed(int(gseed[r])))
        gxy[r], gyaw[r] = g_xy[0].numpy(), float(g_yaw[0])
    if set_name == "campaign":
        cell = np.repeat(np.arange(len(POLICIES) * len(NOISE_LEVELS)), policy_cells(n))[idx]
        pol, lev = cell // len(NOISE_LEVELS), cell % len(NOISE_LEVELS)
    else:
        pol, lev = np.full(m, -1, np.int64), np.zeros(m, np.int64)
    out.update(seeds=seeds.astype(np.int64), start_q=q, start_cube_pos=cpos, start_cube_quat=cquat, tries=tries,
               goal_xy=gxy, goal_yaw=gyaw, family=np.array([POLICIES[p]["family"] if p >= 0 else -1 for p in pol], np.int64),
               policy=pol.astype(np.int64), noise_level=lev.astype(np.int64),
               noise_scale=np.asarray(NOISE_LEVELS, np.float64)[lev] if set_name == "campaign" else np.zeros(m),
               index=idx.astype(np.int64), task=idx.astype(np.int64), spread_deg=spread, tail=tail, goal_seed=gseed,
               master_seed=np.int64(master), set_id=np.int64(sid), n_set=np.int64(n), campaign=np.array(P_CAMPAIGN))
    return out


def expand_pgate(tasks, noise=PGATE_NOISE):
    """pgate task table [n] -> episode table [len(POLICIES) * len(noise) * n]: every policy on every task at every noise
    scale; row index (parent id) = (policy * len(noise) + level) * n + task, ``task`` = the task row."""
    n = len(tasks["seeds"])
    parts = []
    for p, pol in enumerate(POLICIES):
        for j, s in enumerate(noise):
            t = {k: (np.array(v, copy=True) if np.ndim(v) and len(v) == n else v) for k, v in tasks.items()}
            t.update(policy=np.full(n, p, np.int64), family=np.full(n, pol["family"], np.int64),
                     noise_scale=np.full(n, float(s)), noise_level=np.full(n, NOISE_LEVELS.index(float(s)), np.int64),
                     index=(p * len(noise) + j) * n + np.asarray(tasks["task"], np.int64))
            parts.append(t)
    return concat_tasks(parts)


def _module(name, rel):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, str(REPO_ROOT / rel))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def eval_policy_module():
    """scripts/so101_policy/eval_policy.py as a module (checkpoint_env, run_policy)."""
    return _module("so101_eval_policy", "scripts/so101_policy/eval_policy.py")


def policy_env_config(policy, B, workers=0, python=None, files=None, horizon=None, log=print):
    """(vec_env.EnvConfig, checkpoint env record, geometry files used) of a policy checkpoint for B worlds of CPU
    MuJoCo: the config eval_policy.run_policy builds for world mujoco (the checkpoint's decoder and obs_rates,
    evaluation rules, no assist). files {geom_path, kin_points}: used where the checkpoint's own path does not exist
    on this machine (eval_policy then falls back to EnvConfig's defaults; the round-3 files are the same)."""
    from nedm.so101_policy.vec_env import EnvConfig
    E = eval_policy_module()
    dec_over, task_over, prov = E.checkpoint_env(str(policy), None, log)
    if horizon is not None:
        task_over["horizon"] = int(horizon)
    use = {}
    for k in ("geom_path", "kin_points"):
        p = (prov or {}).get("env", {}).get(k)
        if p and Path(p).expanduser().exists():
            use[k] = p
        elif files and files.get(k):
            use[k] = str(Path(files[k]).expanduser())
    cfg = EnvConfig(world="mujoco", num_envs=B, device="cpu", ckpts=[], members="split", auto_reset=False,
                    eval_rules=True, task=task_over, mj_workers=workers, mj_python=python, starts="",
                    nrd_cuda_graph=True, grasp_assist=False, decoder=dec_over, **use)
    return cfg, prov, use


def run_policy_rollouts(tasks, policy, workers=0, python=None, files=None, horizon=None, log=print):
    """Pass 1 of round 3 (section docstring) for every task of ``tasks`` (one policy; per-task ``noise_scale``) with the
    checkpoint ``policy`` -> dict of numpy arrays with run_rollouts' keys (label_rollouts / rows_of read them; the
    expert's option keys at neutral values) plus the policy keys: policy, noise_scale, noise_level, task, tail,
    task_steps (20 ms steps to the task's end), ret / length (decision reward sum, decisions), act_mean / act_exec
    [B, K, A] float32 (the policy's pre-tanh mean and the executed action per decision), decoder event counts while the
    task ran (grasp_started, place_started, aborted, gate_blocked), policy path / sha256, env_resolved."""
    import torch

    from nedm.so101_policy.decoder import MODES
    from nedm.so101_policy.ppo import load_policy, sha256
    from nedm.so101_policy.vec_env import PickPlaceVecEnv
    G = gate()
    B = len(tasks["seeds"])
    pols = np.unique(np.asarray(tasks["policy"]))
    if len(pols) != 1:
        raise ValueError(f"run_policy_rollouts takes the tasks of one policy (got {pols.tolist()})")
    t0 = time.time()
    cfg, prov, used = policy_env_config(policy, B, workers, python, files, horizon, log)
    env = PickPlaceVecEnv(cfg, record=False)
    try:
        dev = env.dev
        pol = load_policy(str(policy), device=dev)
        if pol.kwargs["obs_dim"] != env.num_obs or pol.kwargs["act_dim"] != env.num_actions:
            raise RuntimeError(f"policy {pol.kwargs['obs_dim']} obs / {pol.kwargs['act_dim']} actions vs env "
                               f"{env.num_obs} / {env.num_actions}")
        world, dec, task = env.world, env.dec, env.task
        D, A = int(dec.cfg.decision_steps), int(env.num_actions)
        H = int(env.tcfg.horizon)
        tail = np.asarray(tasks["tail"], np.int64)
        KMAX = (H + int(tail.max()) + 1) // D + 2
        TA = KMAX * D
        cmd = np.zeros((B, TA + 1, 6), np.float32)
        phs = np.zeros((B, TA + 1), np.int8)
        rec = np.zeros((B, 2 * TA + 1, 25), np.float32)
        rcon = np.zeros((B, 2 * TA, 3), bool)
        tlo, thi = world.tlo, world.thi
        mode_ids = torch.arange(len(MODES))
        g_modes = mode_ids[[m.startswith("g_") for m in MODES]]
        p_modes = mode_ids[[m.startswith("p_") for m in MODES]]
        st = dict(t=0, placed=torch.zeros(B, dtype=torch.bool))
        dec_step0, world_step0 = dec.step, world.step

        def dec_step(q):            # the decoder's command of step t and its collector phase
            t = st["t"]
            mode = dec.mode.clone()
            u, extra = dec_step0(q)
            moving = (dec.vel.norm(dim=-1) > 1e-6) | (dec.yrate.abs() > 1e-6)
            mc = mode.cpu().long()
            st["placed"] = torch.where(torch.isin(mc, p_modes), True,
                                       torch.where(torch.isin(mc, g_modes), False, st["placed"]))
            ph = collector_phase(mode, st["placed"].to(mode.device), moving)
            cmd[:, t] = np.clip(u.detach().cpu().double().numpy(), tlo, thi).astype(np.float32)
            phs[:, t] = ph.cpu().numpy()
            return u, extra

        def world_step(u):           # the env's 10 ms records of step t
            t = st["t"]
            ws = world_step0(u)
            if not np.array_equal(world.applied.numpy(), cmd[:, t]):
                raise RuntimeError("applied command differs from the logged one")
            r = world.records.numpy().astype(np.float32)
            rec[:, 2 * t + 1], rec[:, 2 * t + 2] = r[:, 0], r[:, 1]
            rcon[:, 2 * t: 2 * t + 2] = world.record_contact.numpy()
            st["t"] = t + 1
            return ws

        dec.step, world.step = dec_step, world_step
        starts = G.start_state(tasks, dev)
        g64 = lambda k: torch.as_tensor(np.asarray(tasks[k]), dtype=torch.float64, device=dev)      # noqa: E731
        seeds = torch.as_tensor(np.asarray(tasks["seeds"]), dtype=torch.long, device=dev)
        mj_start = dict(q=tasks["start_q"], cube_pos=tasks["start_cube_pos"], cube_quat=tasks["start_cube_quat"])
        obs = env.reset_tasks(torch.arange(B, device=dev), starts, g64("goal_xy"), g64("goal_yaw"), seeds, mj_start)
        rec[:, 0] = world.raw_state().numpy().astype(np.float32)
        scale = torch.as_tensor(np.asarray(tasks["noise_scale"], np.float64), dtype=torch.float32, device=dev)
        noisy = scale > 0
        nseed = seeds * 7919 + NOISE_SALT
        act_mean = np.zeros((B, KMAX, A), np.float32)
        act_exec = np.zeros((B, KMAX, A), np.float32)
        z = lambda: torch.zeros(B, dtype=torch.long, device=dev)                                   # noqa: E731
        counts = dict(grasp_started=z(), place_started=z(), aborted=z(), gate_blocked=z())
        noisy_n = z()
        keep = None
        INF = np.iinfo(np.int64).max // 4
        ns = None
        with torch.no_grad():
            for k in range(KMAX):
                live = task.tr.end == 0
                mu = pol.act_inference(obs.to(dev))
                std = pol.std_of(pol.ctx(obs)).detach().to(mu)
                eps = G.hash_normal(nseed, k, A).to(mu)
                a = torch.where(noisy[:, None], mu + scale[:, None].to(mu) * std * eps, mu)
                act_mean[:, k], act_exec[:, k] = mu.cpu().numpy(), a.cpu().numpy()
                noisy_n += (noisy & live).long()
                obs, _, done, ex = env.step(a)
                ep = ex["episode"]
                if keep is None:
                    keep = {n_: v.clone() for n_, v in ep.items()}
                else:
                    for n_, v in ep.items():
                        keep[n_] = torch.where(done, v, keep[n_])
                for c in counts:
                    counts[c] += ex["counts"][c].long()
                ended = (task.tr.end != 0).cpu().numpy()
                te = keep["steps"].cpu().numpy().astype(np.int64)
                es = world.end_step.numpy().astype(np.int64)
                ns = np.minimum(np.where(ended, te + tail, INF), np.where(es >= 1, es, INF))
                if ended.all() and st["t"] >= int(ns.max()) + 1:
                    break
                if k % 20 == 19:
                    log(f"  decision {k + 1}: task ended {int(ended.sum())}/{B}, {time.time() - t0:.0f} s", flush=True)
            else:
                raise RuntimeError(f"worlds did not stop within {KMAX} decisions")
        K = k + 1
        T_ = int(ns.max())
        es, er = world.end_step.numpy().astype(np.int64), world.end_reason.numpy()
        env_end = np.where((es >= 1) & (es <= ns), er, 0).astype(np.int8)
        env_end_step = np.where(env_end != 0, es, -1)
        resolved = json.dumps(env.resolved(), default=str)
        dec_json = json.dumps((prov or {}).get("decoder", {}), default=str, sort_keys=True)
    finally:
        env.close()
    if (ns < 1).any():
        raise RuntimeError("a world never stopped")
    from nedm.so101_policy import task as T
    kp = {n_: v.cpu().numpy() for n_, v in keep.items()}
    neg = lambda dt=np.int64: np.full(B, -1, dt)                                                   # noqa: E731
    out = dict(seeds=np.asarray(tasks["seeds"]), start_q=np.asarray(tasks["start_q"]),
               start_cube_pos=np.asarray(tasks["start_cube_pos"]), start_cube_quat=np.asarray(tasks["start_cube_quat"]),
               family=np.asarray(tasks["family"]), index=np.asarray(tasks["index"]), commands=cmd[:, : T_ + 1],
               phases=phs[:, : T_ + 1], n_steps=ns.astype(np.int64), records=rec[:, : 2 * T_ + 1],
               record_contact=rcon[:, : 2 * T_], dist_step=neg(), dist_pose=np.zeros((B, 13)),
               jump_raw=np.zeros((B, 25), np.float32), env_end=env_end, env_end_step=env_end_step, rt_step1=neg(),
               rt_step2=neg(), noisy_n=noisy_n.cpu().numpy(), harness_grasp=np.zeros(B, np.int64),
               harness_place=np.zeros(B, np.int64), speed=neg(), wall_s=np.float64(time.time() - t0),
               decoder_overrides=np.array(dec_json), expert_overrides=np.array("{}"),
               task_end=kp["end"], success=kp["end"] == T.EC["success"], success10=kp["success10"], lifted=kp["lifted"],
               end_xy=kp["end_xy"], end_yaw=kp["end_yaw"], recoveries=np.zeros(B, np.int64), noise=np.zeros(B, np.int64),
               sigma=np.zeros(B), rt1=neg(), rt2=neg(), pause_s=np.zeros(B), hold_s=np.zeros(B), lift_h=np.full(B, np.nan),
               dist_r=np.full(B, -1.0), spread_deg=np.asarray(tasks["spread_deg"]), tries=np.asarray(tasks["tries"]),
               horizon=np.full(B, H, np.int64), policy=np.asarray(tasks["policy"], np.int64),
               noise_scale=np.asarray(tasks["noise_scale"], np.float64),
               noise_level=np.asarray(tasks["noise_level"], np.int64), task=np.asarray(tasks["task"], np.int64),
               tail=tail, task_steps=kp["steps"].astype(np.int64), ret=kp["ret"], length=kp["length"].astype(np.int64),
               act_mean=act_mean[:, :K], act_exec=act_exec[:, :K], decisions=np.int64(K),
               policy_path=np.array(str(policy)), policy_sha256=np.array(sha256(policy)),
               env_resolved=np.array(resolved), geometry_files=np.array(json.dumps(used)), noise_salt=np.int64(NOISE_SALT),
               campaign=np.array(P_CAMPAIGN), task_end_names=np.array(json.dumps(list(T.ENDS))))
    out.update({k_: v.cpu().numpy() for k_, v in counts.items()})
    return out


def policy_variant_of(family, ro, k, segment, chunk):
    """Variant string of a policy row: family code, policy id, noise level (d 0, h 0.5, f 1.0 x std), chunk."""
    code = POLICY_FAMILIES[family]["code"]
    lev = NOISE_CODES[NOISE_LEVELS.index(float(ro["noise_scale"][k]))]
    v = f"{code}:{POLICY_IDS[int(ro['policy'][k])]}.{lev}"
    return v + (f"/c{chunk}" if chunk else "")


P_META_KEYS = META_KEYS + P_META_EXTRA


def policy_rows_of(ro, k, segments, family_offset, parent_id, chunked=True, records=RECORDS, step=CHUNK_STEP):
    """rows_of for a policy rollout file (POLICY_FAMILIES, policy_tail, policy_variant_of, P_META_EXTRA)."""
    return rows_of(ro, k, segments, family_offset, parent_id, chunked=chunked, records=records, step=step,
                   n_families=len(POLICY_FAMILIES), variant=policy_variant_of, extra_meta=P_META_EXTRA)
