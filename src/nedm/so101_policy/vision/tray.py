"""Tray student (build8_tray_student.md): a camera policy whose ONLY inputs are the top and wrist camera images and the
arm's joint angles q (6) and joint rates qdot (6), and whose output is a chunk of absolute joint targets (5 arm joints +
jaw) every 20 ms: deployable on a real SO101 without privileged information. The goal pose is shown as a TRAY drawn on
the table (``render.so101`` goal "tray"; visual only, no collision in MuJoCo or the NRD); the student reads it from the
images.

Teacher: a state PPO policy acting through decoder v2 (``teacher.Teacher``; its grasp gate and macro heights read the
TRUE cube pose: teacher side only). Labels = the joint targets that the teacher + its decoder send.

Joint-level episode loop (``TrayWorld.run``), one decision = 5 world steps of 20 ms, B worlds on one device:

1. At each decision boundary the LABEL decoder ``lab`` (a ``DecoderV2`` with the teacher's config) is made consistent
   with the world (below), the teacher decides from its 79-dim observation (world state + ``lab``'s state), ``lab``
   decides with the clean action and steps 5 times with the measured q: its 5 outputs are the labels of the decision.
2. Who drives the decision (per world): the clean teacher (``lab``'s outputs are executed), the noisy teacher (DART:
   a driver decoder ``drv`` decides with Gaussian noise of ``noise`` x the teacher's std on the move columns a[0:4];
   ``lab`` is re-copied from ``drv`` at every boundary, so the labels are the clean teacher's continuation from the
   noisy state), or the student (its chunk's first ``n_exec`` targets are executed). The mix is drawn per world and
   decision with probability ``beta`` for the teacher (counter-based hash, as ``teacher.MixAct``).
3. Re-sync after a student decision (``resync``): the label decoder's command state is set to the student's last two
   commands (q_cmd, step dq; in free / hold motion also the Cartesian target = FK of the command, its velocity from
   the last two commands, the IK warm start), the jaw schedule restarts from the student's jaw command if it differs
   from the shadow's by > 0.02 rad, and the mode is made consistent with the cube: a decoder that thinks it holds the
   cube (hold, p_brake, p_settle, p_lower) while the cube is not held goes to free (jaw opens); a decoder before the
   close (free, g_brake, g_descend, g_settle) while the student holds the cube goes to g_grip (then the GRASP lift by
   the teacher's current a[5]). Macro lines keep their plan (the labels pull the command back onto the line).
   A perfect follower is re-synced to (nearly) the shadow's own state (``tests``: labels change < 1e-3 rad).

Rules (``rule_config``): the evaluation rules of the study with the decoder-free macro view (``task.step(state,
macro=None)``) for BOTH teacher and student, so they are judged identically. The three rules that are defined by
the decoder's macro phases only log: ``keep_out`` (no low TCP near a resting cube outside a macro), ``grasp_box`` (at
the GRASP trigger / close) and ``cube_slide`` (a cube sliding on the table outside a macro: without the macro view it
fired on GRASP-descent nudges, 6 of 100 teacher episodes in CPU MuJoCo on stage A, all in g_descend; a pushed cube
must still be lifted and placed for success). Every other failure rule stays (release_high, unsupported, hold_cap,
cube_tilted, arm_self, moved_untouched, lowest_point, link / jaw overlap in the NRD, the world's own flags). Success = 15 mm / 15 deg, cube flat and at rest, released, jaw open, TCP 3 cm above
the cube top, after a firm lift (unchanged).

Records (``run(record=True)``): per decision row (boundary) the render inputs q, qd, cube pose (true orientation), who
drove (``src`` 0 teacher, 1 noisy teacher, 2 student), the label decoder's mode, ``cont`` (= the previous decision was
driven by the clean teacher: the label plan continues across the boundary) and the teacher's pre-tanh action; per 20
ms step the label ``u_lab`` [6] and the executed target ``u_exe`` [6]. Only live rows / steps are packed.

Student targets (``TraySet``): the chunk of the label targets u_lab[t .. t + K - 1] minus the measured q at the query
time t (a boundary), mean / std normalised per (step, joint); element j is valid if inside the episode and (j < 5, or
every boundary between t and t + j continues the clean teacher's plan: ``cont``).

Stages (``STAGES``, the user's curriculum): A = fixed cube spot + fixed tray spot (arm start from the collector's
start sampler), B = random cube + fixed tray, C = random cube + random tray (the PPO task distribution). Seeds
(written down here): training tasks A 530,000+, B 540,000+, C 550,000+; development evaluation sets A 560,000-560,099,
B 561,000-561,099, C 562,000-562,099 (the vision range 500,000-599,999; never gate 1000-2023, gate2 400,000+ or test
900,000+; ``teacher.check_seeds``).
"""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from nedm.so101_policy.vision import dataset as V

CONTROL_DT = 0.02
PAN_XY = (0.0388, 0.0)
LOG_ONLY = ("grasp_box", "keep_out", "cube_slide")
SRC = {"teacher": 0, "noisy": 1, "student": 2}


def polar(r, deg):
    a = math.radians(deg)
    return (PAN_XY[0] + r * math.cos(a), PAN_XY[1] + r * math.sin(a))


CUBE_A = (*polar(0.21, 30.0), math.radians(10.0))        # fixed cube spot of stage A: x, y (m), yaw (rad)
TRAY_AB = (*polar(0.21, -30.0), math.radians(25.0))      # fixed tray spot of stages A and B
STAGES = {
    "A": dict(cube="fixed", tray="fixed", train0=530_000, dev0=560_000),
    "B": dict(cube="random", tray="fixed", train0=540_000, dev0=561_000),
    "C": dict(cube="random", tray="random", train0=550_000, dev0=562_000),
}
# Teacher-side decoder changes of this build (written down here; build8 sec. 1b-1e). A student that sees only images, q
# and qdot cannot see the decoder's clocks: (v2) the place opening ramp kept the command below the cube-blocked jaw
# angle for a whole decision (no visible change: the student stayed with the jaw closed); (v3) the holds before the
# close (t_settle), after it (t_grip) and before the lowering (t_place_settle) are hidden waits and the minimum-jerk
# close is slow. TEACHER_DECODER = v3: no waits, jaw ramps at 20 rad/s (8 steps for a full close / open with the
# 0.09 rad step clip); round-0 students 54 / 60 % (seeds 0 / 1) on stage A. Fixing also the random grasp / place depths
# (v4: 14 %) and the random line durations (v5: 3 %) made behaviour cloning WORSE: the teacher's random draws spread
# the data around the nominal path (``V4_DECODER``, ``V5_DECODER`` kept for the record). The decoder's rest conditions
# and speed / acceleration rules stay; the PPO policy is not read inside the macros. Teacher with v3 / v4 / v5: stage A
# 100 / 100 in CPU MuJoCo and in the NRD.
TEACHER_DECODER = {"jaw_speed": 20.0, "t_open": (0.0, 0.0), "t_close": (0.0, 0.0), "t_settle": (0.0, 0.0),
                   "t_grip": (0.0, 0.0), "t_place_settle": (0.0, 0.0)}
V4_DECODER = {**TEACHER_DECODER, "grasp_dz": (0.001, 0.001), "place_dz": (0.0015, 0.0015)}
V5_DECODER = {**V4_DECODER, "t_descend": (0.375, 0.375), "t_lift": (0.575, 0.575), "t_lower": (0.525, 0.525),
              "t_retreat": (0.375, 0.375), "retreat": (0.075, 0.075)}
DEC_STATE = ("mode", "pos", "yaw", "vel", "yrate", "v0", "vn", "w0", "wn", "tau", "tb", "fv", "fa", "fw", "fwa", "lp0",
             "lp1", "lk", "ln", "jaw_from", "jaw_to", "jaw_k", "jaw_n", "q_des", "q_cmd", "dq", "low_off", "z_goal",
             "h", "z_ceil", "yaw_hold0", "goal", "tilt0", "t", "seed", "draws", "episode", "hv", "hw", "hA_d", "hA_w",
             "hA_T", "hA_tau", "hB_d", "hB_w", "hB_T", "hB_tau", "carry_on", "carry_brk", "ly0", "ly1", "ik_err",
             "armed")


def log_default(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


# ---- tasks ------------------------------------------------------------------------------------------------------------
def wrap90(a):
    q = math.pi / 2
    return -(((-a + q / 2) % q) - q / 2)


def make_tasks(stage, n, first=0, dev=False, log=log_default):
    """n tasks of a stage (module docstring), seeds = the stage's training (or ``dev``) start + first + i. Per seed:
    rng = default_rng(seed): the cube (A: fixed spot; B: the collector's ``sample_cube`` until it is 6-30 cm from the
    fixed tray and the needed turn |wrap90(tray yaw - cube yaw)| <= 30 deg; C: ``sample_cube``), the arm start
    (``Scene.sample_above`` within 40 deg (70 %) or 60 deg (30 %) of the cube direction, as the expert_base family),
    the collector's 0.3 s warm-up (rejected if it ends, touches the table or has a jaw on the cube), the goal (A, B: the
    fixed tray; C: ``task.sample_goals``, 6-30 cm, yaw +-30 deg). -> task table (gate_expert format: start_*, state_*,
    goal_xy, goal_yaw, seeds)."""
    from nedm.contact_nrd.collect import so101_grasp as g
    from nedm.so101_policy import task as T
    from nedm.so101_policy.mujoco_env import RAW, _Block, _SAMPLER, reserved_seeds
    st = STAGES[stage]
    seeds = np.arange(n, dtype=np.int64) + (st["dev0"] if dev else st["train0"]) + int(first)
    from nedm.so101_policy.vision.teacher import check_seeds
    check_seeds(seeds)
    bad = sorted(set(seeds.tolist()) & reserved_seeds())
    if bad:
        raise ValueError(f"seeds {bad[:5]} are collector campaign seeds")
    key = ("block", True)
    if key not in _SAMPLER:
        _SAMPLER[key] = _Block(1, overlap=True)
    blk = _SAMPLER[key]
    s = blk.s
    q, pos, quat, raw = np.zeros((n, 6)), np.zeros((n, 3)), np.zeros((n, 4)), np.zeros((n, RAW))
    gxy, gyaw, spread, tries = np.zeros((n, 2)), np.zeros(n), np.zeros(n), np.zeros(n, np.int64)
    t0 = time.time()
    for k, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        spread[k] = 40.0 if rng.uniform() < 0.7 else 60.0
        while True:
            tries[k] += 1
            if tries[k] > 500:
                raise RuntimeError(f"seed {seed}: no start")
            if st["cube"] == "fixed":
                cube, cyaw = np.array([CUBE_A[0], CUBE_A[1], s.half]), CUBE_A[2]
            else:
                cube, cyaw = s.sample_cube(rng)
                if st["tray"] == "fixed":
                    d = math.hypot(cube[0] - TRAY_AB[0], cube[1] - TRAY_AB[1])
                    if not (0.06 <= d <= 0.30) or abs(wrap90(TRAY_AB[2] - cyaw)) > math.radians(30.0):
                        continue
            a = math.degrees(math.atan2(cube[1] - PAN_XY[1], cube[0] - PAN_XY[0]))
            try:
                q0 = s.sample_above(rng, angle=(a - spread[k], a + spread[k]))
            except g.PlanError:
                continue
            if q0 is None:
                continue
            c = g.resting_cube(cube, cyaw)
            row = np.concatenate((q0, np.zeros(6), c["pos"], c["quat"], np.zeros(6)))[None]
            res = blk.reset([0], row, warmup=True)
            if res["code"][0] == 0 and not res["arm_table"][0] and not res["contact"][0, :2].any():
                break
        q[k], pos[k], quat[k], raw[k] = q0, c["pos"], c["quat"], res["raw"][0]
        if st["tray"] == "fixed":
            gxy[k], gyaw[k] = TRAY_AB[:2], TRAY_AB[2]
        else:
            gen = torch.Generator().manual_seed(int(seed) * 7 + 3)
            cy, _ = T.cube_yaw_tilt(torch.as_tensor(raw[k:k + 1, 15:19]).float().double())
            gg, gy = T.sample_goals(torch.as_tensor(raw[k:k + 1, 12:14]).float().double(), cy, gen)
            gxy[k], gyaw[k] = gg[0].numpy(), float(gy[0])
    r = raw.astype(np.float32).astype(np.float64)
    log(f"stage {stage}: {n} tasks (seeds {seeds[0]}-{seeds[-1]}) in {time.time() - t0:.0f} s, tries p50 "
        f"{np.median(tries):.0f}")
    out = dict(seeds=seeds, start_q=q, start_cube_pos=pos, start_cube_quat=quat, goal_xy=gxy, goal_yaw=gyaw,
               spread_deg=spread, tries=tries, stage=np.array(stage), dev=np.bool_(dev))
    for name, sl in (("q", slice(0, 6)), ("qd", slice(6, 12)), ("cube_pos", slice(12, 15)), ("cube_quat", slice(15, 19)),
                     ("cube_v", slice(19, 22)), ("cube_w", slice(22, 25))):
        out[f"state_{name}"] = r[:, sl]
    return out


def load_tasks(path, idx=None):
    from nedm.so101_policy.vision.demos import load_tasks as lt
    return lt(path, idx)


def subset(tasks, idx):
    n = len(tasks["seeds"])
    return {k: (v[idx] if np.ndim(v) and len(v) == n else v) for k, v in tasks.items()}


# ---- decoder state copy / re-sync ---------------------------------------------------------------------------------------
def dec_copy(dst, src, mask):
    """Rows ``mask`` [B] bool of every per-world state tensor of DecoderV2 ``src`` -> ``dst`` (new tensors)."""
    for n in DEC_STATE:
        a, b = getattr(dst, n), getattr(src, n)
        m = mask.view(-1, *([1] * (a.dim() - 1)))
        setattr(dst, n, torch.where(m, b, a))
    for k in dst.counts:
        dst.counts[k] = torch.where(mask, src.counts[k], dst.counts[k])


def settle_cfg():
    from nedm.so101_policy import task as T
    return T.TaskConfig()


def resync(dec, mask, u1, u2, F, a5=None, jaw_tol=0.02, jaw_done_tol=0.05, rest_tol=0.01, settle_bottom=0.004,
           settle_z=(-0.004, 0.0015)):
    """Make the label decoder ``dec`` consistent with a world that the student drove in the last decision (module
    docstring, point 3). mask [B] bool; u1, u2 [B, 6]: the last and the second last executed command; F: task
    features of the current state (``held``, ``bottom``); a5 [B]: the teacher's lift-height output in [-1, 1] (for a
    student grasp). -> dict of [B] bool: to_free, to_grip."""
    from nedm.so101_policy.decoder import CONTROL_DT as DT, M
    from nedm.so101_policy.kin import wrap, yaw_of
    if not bool(mask.any()):
        z = torch.zeros_like(mask)
        return dict(to_free=z, to_grip=z.clone(), snap=z.clone())
    c = dec.cfg
    f = dec.f
    u1, u2 = u1.to(**f), u2.to(**f)
    m = mask
    mode = dec.mode
    held = F["held"].to(m.device)
    holding = (mode == M["hold"]) | (mode == M["p_brake"]) | (mode == M["p_settle"]) | (mode == M["p_lower"])
    pre = (mode == M["free"]) | (mode == M["g_brake"]) | (mode == M["g_descend"]) | (mode == M["g_settle"])
    to_free = m & holding & ~held
    to_grip = m & pre & held
    new_mode = torch.where(to_free, M["free"], torch.where(to_grip, M["g_grip"], mode))
    fk = dec.kin.fk
    p1, r1 = fk(u1)
    p2, r2 = fk(u2)
    y1, y2 = yaw_of(r1), yaw_of(r2)
    # (a) delta re-sync (free / hold, no mode change): shift the shadow's command, IK target and Cartesian target by
    # the student's deviation from the shadow's own command, and its target velocity by the difference of the command
    # velocities; a perfect follower leaves the shadow unchanged
    keep = m & ((mode == M["free"]) | (mode == M["hold"])) & ~to_free & ~to_grip
    qc = dec.q_cmd
    qc2 = torch.cat((qc[:, :5] - dec.dq, qc[:, 5:]), -1)
    q_des_new = dec.q_des + (u1[:, :5] - qc[:, :5])
    pa, ra = fk(dec.q_des)
    pb, rb = fk(q_des_new)
    pc1, rc1 = fk(qc)
    pc2, rc2 = fk(qc2)
    pos_k = dec.pos + (pb - pa)
    yaw_k = dec.yaw + wrap(yaw_of(rb) - yaw_of(ra))
    vel_k = dec.vel + (p1 - p2) / DT - (pc1 - pc2) / DT
    yr_k = dec.yrate + wrap(y1 - y2) / DT - wrap(yaw_of(rc1) - yaw_of(rc2)) / DT
    # (b) absolute re-sync at a mode change (to_free: free motion from the student's command; to_grip: at rest)
    ab = to_free | to_grip
    yaw_a = dec.yaw + wrap(y1 - dec.yaw)
    vel_a = torch.where(to_grip[:, None], torch.zeros_like(p1), (p1 - p2) / DT)
    yr_a = torch.where(to_grip, torch.zeros_like(y1), wrap(y1 - y2) / DT)
    blendm = keep | to_free
    setm = keep | ab
    m1, m3 = setm, setm[:, None]
    pos_n = torch.where(ab[:, None], p1, pos_k)
    yaw_n = torch.where(ab, yaw_a, yaw_k)
    v_set = torch.where(ab[:, None], vel_a, vel_k)
    w_set = torch.where(ab, yr_a, yr_k)
    qd_n = torch.where(ab[:, None], u1[:, :5], q_des_new)
    dec.pos = torch.where(m3, pos_n, dec.pos)
    dec.yaw = torch.where(m1, yaw_n, dec.yaw)
    dec.vel = torch.where(m3, v_set, dec.vel)
    dec.yrate = torch.where(m1, w_set, dec.yrate)
    dec.q_des = torch.where(m3, qd_n, dec.q_des)
    ab3 = ab[:, None]
    dec.v0, dec.vn = torch.where(ab3, v_set, dec.v0), torch.where(ab3, v_set, dec.vn)
    dec.w0, dec.wn = torch.where(ab, w_set, dec.w0), torch.where(ab, w_set, dec.wn)
    dec.tau = torch.where(ab, torch.full_like(dec.tau, dec.T_d), dec.tau)
    dec.tb = torch.where(ab, torch.full_like(dec.tb, dec.T_d), dec.tb)
    # command state (every re-synced world, macros included: the limiter continues from the student's command)
    jaw_prev = dec.q_cmd[:, 5].clone()
    dec.q_cmd = torch.where(m[:, None], u1, dec.q_cmd)
    dec.dq = torch.where(m[:, None], u1[:, :5] - u2[:, :5], dec.dq)
    # macro lines (descent, lift, lowering, retreat): a student ahead of the shadow's line clock moves the clock to
    # the student's progress along the line (minimum-jerk inverse), so the labels never pull the student back
    from nedm.so101_policy.decoder import minjerk
    linem = m & ((mode == M["g_descend"]) | (mode == M["g_lift"]) | (mode == M["p_lower"]) | (mode == M["p_retreat"]))
    if bool(linem.any()):
        d = dec.lp1 - dec.lp0
        L2 = (d * d).sum(-1).clamp_min(1e-12)
        prog = (((p1 - dec.lp0) * d).sum(-1) / L2).clamp(0.0, 1.0)
        n_ = dec.ln.clamp_min(1.0)
        lo, hi = torch.zeros_like(prog), torch.ones_like(prog)
        for _ in range(30):                                   # s with minjerk(s) = prog
            mid = (lo + hi) / 2
            up_ = minjerk(mid) < prog
            lo, hi = torch.where(up_, mid, lo), torch.where(up_, hi, mid)
        k_new = torch.minimum(torch.round(hi * n_), n_)
        adv = linem & (L2 > 1e-8) & (k_new > dec.lk)
        dec.lk = torch.where(adv, k_new, dec.lk)
    # macros that wait for the command at rest (g_settle -> g_close, p_lower -> p_open; the decoder's rule is exact to
    # 1e-9 rad): with the line finished (its clock synced to the student's progress), the student counts as settled where the task geometry says the step is done (descent: cube centre in
    # the task's close box in x, y and yaw, and 4 mm below to 1.5 mm above the grasp point (the teacher descends to
    # 0-2 mm); lowering: lowest cube corner <= settle_bottom above the table) or
    # its command is within rest_tol of the shadow's IK target; the shadow then adopts the student's pose at rest and
    # the macro goes on (without this the shadow never reached the close: build8 sec. 1c)
    tc = settle_cfg()
    xg = F["xg"].to(m.device)
    in_box = ((xg[:, 0].abs() <= tc.box_x) & (xg[:, 1].abs() <= tc.box_y) & (xg[:, 2] >= settle_z[0])
              & (xg[:, 2] <= settle_z[1]) & (F["yaw_mis"].to(m.device) <= tc.close_yaw))
    near_table = F["bottom"].to(m.device) <= settle_bottom
    close_q = (u1[:, :5] - dec.q_des).abs().amax(-1) < rest_tol
    done = dec.lk >= dec.ln - 1e-9
    g_wait = (mode == M["g_descend"]) | (mode == M["g_settle"])
    snap = m & done & ((g_wait & (in_box | close_q)) | ((mode == M["p_lower"]) & (near_table | close_q)))
    if bool(snap.any()):
        s3 = snap[:, None]
        dec.q_des = torch.where(s3, u1[:, :5], dec.q_des)
        dec.q_cmd = torch.where(s3, u1, dec.q_cmd)
        dec.dq = torch.where(s3, torch.zeros_like(dec.dq), dec.dq)
        dec.pos = torch.where(s3, p1, dec.pos)
        dec.yaw = torch.where(snap, dec.yaw + wrap(y1 - dec.yaw), dec.yaw)
        dec.vel = torch.where(s3, torch.zeros_like(dec.vel), dec.vel)
        dec.yrate = torch.where(snap, torch.zeros_like(dec.yrate), dec.yrate)
        # a descent that is done: settle at once (g_settle with its hold finished); a lowering: its line is finished
        dec.mode = torch.where(snap & (mode == M["g_descend"]), M["g_settle"], dec.mode)
        mode = torch.where(snap & (mode == M["g_descend"]), M["g_settle"], mode)
        new_mode = torch.where(snap & (new_mode == M["g_descend"]), M["g_settle"], new_mode)
        dec.lk = torch.where(snap, dec.ln, dec.lk)
    # jaw: free / hold restart the schedule from the student's jaw if it left the shadow's; mode changes always
    goal = torch.where(new_mode == M["free"], torch.full_like(dec.yaw, c.jaw_open),
                       torch.where((new_mode == M["hold"]) | (new_mode == M["g_grip"]),
                                   torch.full_like(dec.yaw, c.jaw_closed), dec.jaw_to))
    jm = (blendm & ((u1[:, 5] - jaw_prev).abs() > jaw_tol)) | to_free | to_grip
    done = (goal - u1[:, 5]).abs() < jaw_done_tol
    n = torch.round(torch.clamp_min(1.875 * (goal - u1[:, 5]).abs() / c.jaw_speed, c.jaw_min_s) / DT).clamp_min(1)
    dec.jaw_from = torch.where(jm, torch.where(done, goal, u1[:, 5]), dec.jaw_from)
    dec.jaw_to = torch.where(jm, goal, dec.jaw_to)
    dec.jaw_k = torch.where(jm, torch.where(done, torch.ones_like(n), torch.zeros_like(n)), dec.jaw_k)
    dec.jaw_n = torch.where(jm, torch.where(done, torch.ones_like(n), n), dec.jaw_n)
    # mode changes
    if bool(to_free.any()):
        dec.z_ceil = torch.where(to_free, torch.full_like(dec.z_ceil, c.z_max), dec.z_ceil)
        dec.armed = dec.armed | to_free
        dec.lk = torch.where(to_free, torch.zeros_like(dec.lk), dec.lk)
        dec.ln = torch.where(to_free, torch.zeros_like(dec.ln), dec.ln)
    if bool(to_grip.any()):
        a5 = torch.zeros_like(dec.yaw) if a5 is None else a5.to(**f).clamp(-1, 1)
        h = c.lift_range[0] + (a5 + 1) / 2 * (c.lift_range[1] - c.lift_range[0])
        dec.h = torch.where(to_grip, h, dec.h)
        bottom = F["bottom"].to(**f)
        ceil = torch.clamp_max(p1[:, 2] + c.cube_bottom_max - bottom.clamp_min(0), c.z_max)
        dec.z_ceil = torch.where(to_grip, ceil, dec.z_ceil)
        dec.goal = torch.where(to_grip[:, None], torch.cat((p1, yaw_a[:, None]), -1), dec.goal)
        dec.armed = dec.armed & ~to_grip
        dec.lp0 = torch.where(to_grip[:, None], p1, dec.lp0)
        dec.lp1 = torch.where(to_grip[:, None], p1, dec.lp1)
        ln = torch.round(dec._draw(to_grip, c.t_grip) / DT)
        dec.ln = torch.where(to_grip, ln, dec.ln)
        dec.lk = torch.where(to_grip, torch.zeros_like(dec.lk), dec.lk)
    dec.mode = new_mode
    return dict(to_free=to_free, to_grip=to_grip, snap=snap)


# ---- the joint-level task world --------------------------------------------------------------------------------------
def rule_config(horizon=400, obs_rates=True, **kw):
    from nedm.so101_policy import task as T
    return T.TaskConfig(horizon=int(horizon), obs_rates=bool(obs_rates), log_only=LOG_ONLY, **kw)


class TrayWorld:
    """B task worlds of one kind (``nrd`` = GraspNrdSim, ``mujoco`` = CPU MuJoCo, ``kin``) + the teacher's label
    decoder + the task rules of this module (module docstring). ``run(tasks, ...)`` plays one episode per world."""

    def __init__(self, world, B, teacher, device="cuda", nrd_ckpt=None, workers=0, mj_python=None, horizon=400,
                 members="split", decoder=None):
        from nedm.so101_policy import task as T
        from nedm.so101_policy import vec_env as VE
        from nedm.so101_policy.decoder import DecoderV2, DecoderV2Config
        self.kind, self.B, self.teacher = world, int(B), teacher
        self.dev = torch.device("cpu" if world == "mujoco" else device)
        ck = [str(Path(c).expanduser()) for c in ([nrd_ckpt] if isinstance(nrd_ckpt, (str, Path)) else (nrd_ckpt or []))]
        cfg = VE.EnvConfig(world=world, num_envs=self.B, device=device, ckpts=ck, members=members, mj_workers=workers,
                           mj_python=mj_python, starts="", decoder=dict(teacher.decoder_cfg), **teacher.files)
        self.cfg = cfg
        self.kin = VE.kinematics(cfg.kin_points, self.dev)
        self.geom = VE.task_geometry(cfg.geom_path, self.dev)
        self.world = VE.make_world(cfg, self.B, self.dev, self.kin, self.geom)
        self.tcfg = rule_config(horizon, teacher.obs_rates)
        self.task = T.PickPlaceTask(self.B, self.dev, kin=self.kin, geom=self.geom, cfg=self.tcfg,
                                    overlap_rules=world != "mujoco")
        self.decoder_over = dict(TEACHER_DECODER if decoder is None else decoder)
        self.dcfg = replace(DecoderV2Config(), **{**teacher.decoder_cfg, **{k: tuple(v) if isinstance(v, list) else v
                                                                             for k, v in self.decoder_over.items()}})
        if self.dcfg.blend != "cosine" or self.dcfg.carry_macro:
            raise ValueError("the tray world supports the cosine blend without the CARRY macro")
        self.lab = DecoderV2(self.B, self.dev, kin=self.kin, cfg=self.dcfg)
        self.drv = None
        self.ws = None
        self.world_steps = 0
        self.nrd_ckpt = ck

    def close(self):
        if hasattr(self.world, "close"):
            self.world.close()

    def _reset(self, tasks):
        from nedm.so101_policy.vision.demos import start_state
        B, dev = self.B, self.dev
        if len(tasks["seeds"]) != B:
            raise ValueError(f"{len(tasks['seeds'])} tasks for {B} worlds")
        idx = torch.arange(B, device=dev)
        st = start_state(tasks, dev)
        if self.kind == "mujoco":
            mj = dict(q=tasks["start_q"], cube_pos=tasks["start_cube_pos"], cube_quat=tasks["start_cube_quat"])
            ws = self.world.reset(idx.cpu(), mj, warmup=True).to(dev)
            dq = float((ws.q - st.q).abs().max())
            dc = float((ws.cube_pos - st.cube_pos).abs().max())
            if dq > 1e-6 or dc > 1e-6:
                raise RuntimeError(f"MuJoCo warm-up start differs from the stored settled state ({dq:.2e}, {dc:.2e})")
        elif self.kind == "mjwarp":
            ws = self.world.reset(idx, st, warmup=False)
        else:
            ws = self.world.reset(idx, st)
        self.ws = ws.to(dev)
        seeds = torch.as_tensor(tasks["seeds"], dtype=torch.long, device=dev)
        self.seeds = seeds
        self.lab.reset(idx, self.ws.q, seed=seeds)
        if self.drv is not None:
            self.drv.reset(idx, self.ws.q, seed=seeds)
        g = lambda k: torch.as_tensor(tasks[k], dtype=torch.float64, device=dev)      # noqa: E731
        self.task.reset(idx, self.ws, g("goal_xy"), g("goal_yaw"))
        self.goal_pose = torch.cat((g("goal_xy"), g("goal_yaw")[:, None]), -1)

    def observation(self, F):
        from nedm.so101_policy import task as T
        tk = self.task
        return T.observation(F, tk.goal_xy, tk.goal_yaw, tk.tr.lifted, tk.tr.air_run, self.tcfg, self.lab.observation(),
                             self.lab.macro())

    @torch.no_grad()
    def run(self, tasks, student=None, beta=1.0, noise=None, seed=0, record=True, n_exec=5, record_states=False,
            log=None, ensemble=None, jaw_lead=0, full_horizon=False):
        """One episode per world on ``tasks`` (B rows). student: a ``TrayActor`` (or None: beta must be 1); beta: the
        probability that the teacher drives a decision (per world and decision); noise [B] (or float): DART noise
        scale of the teacher's move columns per world (beta 1 only); n_exec: executed targets per student query (<= 5,
        one query per decision). -> dict: per-episode arrays (end, success, success10, lifted, end_xy, end_yaw, steps,
        rows) and with ``record`` the packed rows / steps (module docstring); ``record_states``: per-step q, cube pose
        [B, S, ...] of all worlds (videos); ``ensemble`` (with n_exec 1): temporal ensembling of the overlapping chunk
        predictions for the current step, weight exp(ensemble x age in steps) (ACT's convention: > 0 favours the older
        predictions, < 0 the newer); ``jaw_lead``: the jaw target is taken that many steps ahead in the chunk(s) (the
        cube-blocked jaw hides the first steps of an opening ramp); ``full_horizon``: keep stepping every world (and
        querying the student) to the horizon after its task ended (videos; the outcome arrays still hold the task end)."""
        from nedm.so101_policy import task as T
        from nedm.so101_policy.decoder import DecoderV2, hash_uniform
        from nedm.so101_policy.vision.teacher import hash_normal
        te, B, dev = self.teacher, self.B, self.dev
        if student is None and beta < 1:
            raise ValueError("beta < 1 needs a student")
        nz = torch.zeros(B, dtype=torch.float64, device=dev) if noise is None else \
            torch.as_tensor(noise, dtype=torch.float64, device=dev).expand(B).clone()
        noisy_w = nz > 0
        if bool(noisy_w.any()) and beta < 1:
            raise ValueError("DART noise only in teacher-driven episodes (beta 1)")
        if bool(noisy_w.any()) and self.drv is None:
            self.drv = DecoderV2(B, dev, kin=self.kin, cfg=self.dcfg)
        n_exec = int(n_exec)                   # query period (steps): a new chunk every n_exec steps, its first n_exec
        if n_exec < 1:                         # targets executed (5 = one query per decision, at the boundaries)
            raise ValueError("n_exec must be >= 1")
        t0 = time.time()
        self._reset(tasks)
        if student is not None:
            student.reset(B)
        lab, drv, task, tc = self.lab, self.drv, self.task, self.tcfg
        any_noise = bool(noisy_w.any())
        steps_per = lab.cfg.decision_steps
        K = math.ceil(tc.horizon / steps_per)
        hseed = self.seeds * 1_000_003 + 7919 * int(seed) + 17
        std = te.std.to(dev, torch.float64)
        u1 = self.ws.q.clone()
        u2 = self.ws.q.clone()
        src_prev = torch.zeros(B, dtype=torch.long, device=dev)
        a_prev = torch.zeros(B, te.act_dim, dtype=torch.float64, device=dev)
        R = {k: [] for k in ("q", "qd", "cube_pos", "cube_quat", "src", "mode", "cont", "live", "a_pre", "to_free",
                             "to_grip", "snap")}
        S = {k: [] for k in ("u_lab", "u_exe", "live", "st_q", "st_qd", "st_cube_pos", "st_cube_quat")}
        SS = {k: [] for k in ("q", "cube")} if record_states else None
        info = None
        t_student = 0.0
        tlo, thi = self.kin.tlo.to(dev), self.kin.thi.to(dev)
        chunk, q_step, hist = None, 0, []
        for k in range(K + 1):
            live0 = task.tr.end == 0
            if not bool(live0.any()) and not full_horizon:
                break
            ws = self.ws
            F = T.features(ws, self.kin, None, tc)
            if any_noise:
                dec_copy(lab, drv, noisy_w)
            stud_prev = src_prev == SRC["student"]
            rs = resync(lab, stud_prev & live0, u1, u2, F, a_prev[:, 5] if a_prev.shape[1] > 5 else None)
            obs = self.observation(F)
            a_pre = te.act(obs.to(te.dev)).to(dev, torch.float64)
            a_clean = torch.tanh(a_pre)
            if beta >= 1.0:
                use_t = torch.ones(B, dtype=torch.bool, device=dev)
            elif beta <= 0.0:
                use_t = torch.zeros(B, dtype=torch.bool, device=dev)
            else:
                use_t = hash_uniform(hseed, torch.full_like(hseed, k), 99) < beta
            src = torch.where(use_t, torch.where(noisy_w, SRC["noisy"], SRC["teacher"]), SRC["student"]).long()
            cont = (src_prev == SRC["teacher"]) & (k > 0)
            if record:
                for name, v in (("q", ws.q), ("qd", ws.qd), ("cube_pos", ws.cube_pos), ("cube_quat", ws.cube_quat)):
                    R[name].append(v.float().cpu())
                R["src"].append(src.to(torch.int8).cpu())
                R["mode"].append(lab.mode.to(torch.int8).cpu())
                R["cont"].append(cont.cpu())
                R["live"].append(live0.cpu())
                R["a_pre"].append(a_pre.float().cpu())
                R["to_free"].append(rs["to_free"].cpu())
                R["to_grip"].append(rs["to_grip"].cpu())
                R["snap"].append(rs["snap"].cpu())
            if any_noise:
                eps = hash_normal(hseed, k, 4).to(a_pre)
                a_n = a_pre.clone()
                a_n[:, :4] = a_pre[:, :4] + nz[:, None] * std[:4] * eps
                drv.decide(torch.tanh(a_n), ws)
            lab.decide(a_clean, ws)
            stud = src == SRC["student"]
            for j in range(steps_per):
                gs = k * steps_per + j                       # 20 ms steps since the reset
                live_s = task.tr.end == 0
                if student is not None and bool((stud & (live_s | full_horizon)).any()) and (chunk is None or gs % n_exec == 0):
                    ts = time.time()
                    chunk = student(self.ws, self.goal_pose).to(dev, torch.float64)
                    chunk = torch.minimum(torch.maximum(chunk, tlo), thi)
                    q_step = gs
                    t_student += time.time() - ts
                if record:
                    for name, v in (("st_q", self.ws.q), ("st_qd", self.ws.qd), ("st_cube_pos", self.ws.cube_pos),
                                    ("st_cube_quat", self.ws.cube_quat)):
                        S[name].append(v.float().cpu())
                u_l, _ = lab.step(self.ws.q)
                u_e = u_l
                if any_noise:
                    u_d, _ = drv.step(self.ws.q)
                    u_e = torch.where(noisy_w[:, None], u_d, u_e)
                if chunk is not None and ensemble is not None and n_exec == 1:
                    if not hist or hist[-1][0] != q_step:
                        hist.append((q_step, chunk))
                        hist = [(t_, c_) for t_, c_ in hist if gs - t_ < c_.shape[1]]
                    num = torch.zeros_like(u_e)
                    den = 0.0
                    for t_, c_ in hist:
                        w_ = math.exp(float(ensemble) * (gs - t_))
                        kk = min(gs - t_ + int(jaw_lead), c_.shape[1] - 1)
                        num = num + w_ * torch.cat((c_[:, gs - t_, :5], c_[:, kk, 5:]), -1)
                        den += w_
                    u_e = torch.where(stud[:, None], num / den, u_e)
                elif chunk is not None:
                    jj = min(gs - q_step, chunk.shape[1] - 1)
                    kk = min(gs - q_step + int(jaw_lead), chunk.shape[1] - 1)
                    u_e = torch.where(stud[:, None], torch.cat((chunk[:, jj, :5], chunk[:, kk, 5:]), -1), u_e)
                w = self.world.step(u_e)
                self.ws = w if self.kind != "mujoco" else w.to(dev)
                _, _, _, info = task.step(self.ws, None, None)
                self.world_steps += B
                if record:
                    S["u_lab"].append(u_l.float().cpu())
                    S["u_exe"].append(u_e.float().cpu())
                    S["live"].append(live_s.cpu())
                if record_states:
                    SS["q"].append(self.ws.q.float().cpu())
                    SS["cube"].append(torch.cat((self.ws.cube_pos, self.ws.cube_quat), -1).float().cpu())
                u2, u1 = u1, u_e
            src_prev = src
            a_prev = a_clean
            if log is not None and k % 20 == 19:
                log(f"  decision {k + 1}: ended {int((task.tr.end != 0).sum())}/{B}")
        tr = task.tr
        cp = lambda v: v.detach().cpu().clone().numpy()                                    # noqa: E731
        out = dict(end=cp(tr.end), success=cp(tr.end == T.EC["success"]), success10=cp(tr.success10),
                   lifted=cp(tr.lifted), end_xy=cp(tr.end_xy), end_yaw=cp(tr.end_yaw), steps=cp(tr.t),
                   box_log=cp(tr.box_log), seeds=np.asarray(tasks["seeds"]),
                   goal_xy=np.asarray(tasks["goal_xy"], np.float32), goal_yaw=np.asarray(tasks["goal_yaw"], np.float32),
                   noise=nz.cpu().numpy(), wall_s=np.float64(time.time() - t0), student_s=np.float64(t_student))
        if self.kind == "mujoco":
            out["mj_reason"] = self.world.end_reason.clone().numpy().astype(np.int16)
        if record:
            out.update(pack({k: torch.stack(v, 1).numpy() for k, v in R.items()},
                            {k: torch.stack(v, 1).numpy() for k, v in S.items()}, out["steps"], steps_per))
        if record_states:
            out["state_q"] = torch.stack(SS["q"], 1).numpy()
            out["state_cube"] = torch.stack(SS["cube"], 1).numpy()
        return out


def pack(rows, steps, ep_steps, steps_per=5):
    """Live rows / steps of every episode -> flat arrays. rows: dict of [B, K, ...]; steps: dict of [B, S, ...];
    ep_steps [B] = live 20 ms steps of each episode (the step in which the task ended included)."""
    B = len(ep_steps)
    ep_steps = np.asarray(ep_steps, np.int64)
    n_rows = (ep_steps + steps_per - 1) // steps_per
    Kr, Ks = rows["live"].shape[1], steps["live"].shape[1]
    if (n_rows > Kr).any() or (ep_steps > Ks).any():
        raise RuntimeError("episode longer than the records")
    for b in range(B):
        if not rows["live"][b, : n_rows[b]].all() or not steps["live"][b, : ep_steps[b]].all():
            raise RuntimeError(f"episode {b}: live records are not a prefix")
    rb = np.repeat(np.arange(B), n_rows)
    rt = np.concatenate([np.arange(n) for n in n_rows]) if B else np.zeros(0, np.int64)
    sb = np.repeat(np.arange(B), ep_steps)
    stt = np.concatenate([np.arange(n) for n in ep_steps]) if B else np.zeros(0, np.int64)
    out = {f"row_{k}": v[rb, rt] for k, v in rows.items() if k != "live"}
    out.update({k: v[sb, stt] for k, v in steps.items() if k != "live"})
    out["row_ep"], out["row_t"] = rb.astype(np.int32), (rt * steps_per).astype(np.int16)
    out["ep_row_start"] = np.concatenate([[0], np.cumsum(n_rows)[:-1]]).astype(np.int64)
    out["ep_rows"] = n_rows
    out["ep_step_start"] = np.concatenate([[0], np.cumsum(ep_steps)[:-1]]).astype(np.int64)
    return out


EP_KEYS = ("end", "success", "success10", "lifted", "end_xy", "end_yaw", "steps", "box_log", "seeds", "goal_xy",
           "goal_yaw", "noise", "mj_reason", "ep_rows", "ep_row_start", "ep_step_start")


def concat(parts):
    """Concatenate ``run`` outputs (episode, row and step indices shifted)."""
    if len(parts) == 1:
        return parts[0]
    out, e0, r0, s0 = {}, 0, 0, 0
    for p in parts:
        E = len(p["end"])
        q = dict(p)
        if "row_ep" in p:
            q["row_ep"] = p["row_ep"] + e0
            q["ep_row_start"] = p["ep_row_start"] + r0
            q["ep_step_start"] = p["ep_step_start"] + s0
            r0 += len(p["row_ep"])
            s0 += len(p["u_lab"])
        e0 += E
        for k, v in q.items():
            if np.ndim(v) == 0:
                continue
            out.setdefault(k, []).append(v)
    out = {k: np.concatenate(v) for k, v in out.items()}
    for k, v in parts[0].items():
        if np.ndim(v) == 0:
            out[k] = np.asarray(sum(np.asarray(p[k]) for p in parts)) if k in ("wall_s", "student_s") else v
    return out


@torch.no_grad()
def collect(tasks, world, teacher, student=None, beta=1.0, noise=0.0, seed=0, device="cuda", nrd_ckpt=None,
            workers=0, mj_python=None, horizon=400, batch=512, log=log_default, meta_extra=None, n_exec=5,
            tw_cache=None, decoder=None, ensemble=None, jaw_lead=0):
    """Episodes on all tasks (batches of ``batch`` worlds) -> packed dict + ``meta`` (json). noise: a float (every
    episode) or an array per task."""
    from nedm.so101_policy.vision.teacher import check_seeds
    check_seeds(tasks["seeds"])
    n = len(tasks["seeds"])
    nzs = np.broadcast_to(np.asarray(noise, np.float64), (n,))
    parts, t0, steps = [], time.time(), 0
    for s0 in range(0, n, batch):
        idx = np.arange(s0, min(n, s0 + batch))
        t = subset(tasks, idx)
        key = (world, len(idx))
        if tw_cache is not None and key in tw_cache:
            tw = tw_cache[key]
        else:
            tw = TrayWorld(world, len(idx), teacher, device, nrd_ckpt, workers, mj_python, horizon, decoder=decoder)
            if tw_cache is not None:
                tw_cache[key] = tw
        w0 = tw.world_steps
        r = tw.run(t, student, beta, nzs[idx], seed, True, n_exec, ensemble=ensemble, jaw_lead=jaw_lead)
        steps += tw.world_steps - w0
        if tw_cache is None:
            tw.close()
        parts.append(r)
        log(f"  collect {world} beta {beta} noise {float(np.mean(nzs[idx])):.2f}: tasks {s0}-{idx[-1]}, success "
            f"{r['success'].mean() * 100:.1f} %, {time.time() - t0:.0f} s")
    out = concat(parts)
    meta = dict(world=world, nrd_ckpt=str(nrd_ckpt), teacher=teacher.info(), n_tasks=n, beta=float(beta),
                noise=float(np.mean(nzs)), seed=int(seed), horizon=int(horizon), batch=int(batch),
                wall_s=time.time() - t0, world_steps=int(steps), success=float(np.mean(out["success"])),
                rows=int(len(out["row_ep"])), seeds=[int(tasks["seeds"][0]), int(tasks["seeds"][-1])],
                n_exec=int(n_exec), rules=dict(log_only=list(LOG_ONLY), macro_view="none"),
                decoder_over={k: v for k, v in tw.decoder_over.items()}, **(meta_extra or {}))
    out["meta"] = np.array(json.dumps(meta, default=str))
    return out


def load(path):
    z = dict(np.load(Path(path).expanduser(), allow_pickle=False))
    z["meta"] = json.loads(str(z["meta"])) if "meta" in z else {}
    return z


def summarize(r):
    """Success, failure causes, lift rate, end errors of a ``run`` / ``collect`` / ``evaluate`` output."""
    from nedm.so101_policy.task import ENDS
    s = np.asarray(r["success"]).astype(bool)
    causes = {ENDS[int(k)]: int(v) for k, v in zip(*np.unique(np.asarray(r["end"])[~s], return_counts=True))}
    n = len(s)
    lo, hi = wilson(int(s.sum()), n)
    out = dict(n=n, success=float(s.mean()), ci95=[lo, hi], success10=float(np.mean(r["success10"])),
               lifted=float(np.mean(r["lifted"])), causes=causes)
    if s.any():
        out["end_xy_mm_p50_success"] = float(np.median(np.asarray(r["end_xy"])[s]) * 1e3)
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ---- dataset ---------------------------------------------------------------------------------------------------------
class TraySet:
    """Collected files (``collect``) on one device. A sample = one query state: render inputs, robot vector [q, qd]
    (12), target chunk [K, 6] = u_lab[t + j] - q_t with its validity mask (module docstring). ``step_rows``: every 20 ms
    step is a sample (needs the per-step states ``st_*`` of the files; for students queried more often than once per
    decision), else the decision boundaries only. Target j is valid inside the episode if every decision boundary
    between t and t + j continues the clean teacher's plan (``row_cont``). Validation: ``val_frac`` of the FIRST
    file's episodes (seeded): the same in every round."""

    def __init__(self, paths, device="cuda", chunk=10, val_frac=0.05, seed=0, steps_per=5, step_rows=False):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        files = [load(p) for p in paths]
        self.meta = [f["meta"] for f in files]
        keys = set.intersection(*[set(f) for f in files]) - {"meta"}
        d = concat([{k: v for k, v in f.items() if k in keys} for f in files])
        n_ep = [len(f["end"]) for f in files]
        E = len(d["end"])
        rng = np.random.default_rng(int(seed))
        first = rng.permutation(n_ep[0])
        val_eps = np.sort(first[: int(round(val_frac * n_ep[0]))])
        is_val = np.zeros(E, bool)
        is_val[val_eps] = True
        self.n_val_eps, self.n_train_eps = int(is_val.sum()), int(E - is_val.sum())
        self.chunk, self.dev, self.steps_per = int(chunk), torch.device(device), int(steps_per)
        self.step_rows = bool(step_rows)
        t = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=self.dev)   # noqa
        self.cont = t(d["row_cont"], torch.bool)
        self.ep_steps, self.ep_step_start = t(d["steps"], torch.long), t(d["ep_step_start"], torch.long)
        self.ep_rows, self.ep_row_start = t(d["ep_rows"], torch.long), t(d["ep_row_start"], torch.long)
        self.u_lab = t(d["u_lab"])
        self.goal_pose = torch.cat((t(d["goal_xy"]), t(d["goal_yaw"])[:, None]), -1)
        self.ep_success = t(d["success"], torch.bool)
        if self.step_rows:
            if "st_q" not in d:
                raise ValueError("step_rows needs files with per-step states (st_*)")
            steps = np.asarray(d["steps"], np.int64)
            s_ep = np.repeat(np.arange(E), steps)
            s_t = np.concatenate([np.arange(n) for n in steps])
            self.q, self.qd = t(d["st_q"]), t(d["st_qd"])
            self.cube_pos, self.cube_quat = t(d["st_cube_pos"]), t(d["st_cube_quat"])
            self.s_ep, self.s_t = t(s_ep, torch.long), t(s_t, torch.long)
            self.s_g = torch.arange(len(s_ep), device=self.dev)
        else:
            self.q, self.qd = t(d["row_q"]), t(d["row_qd"])
            self.cube_pos, self.cube_quat = t(d["row_cube_pos"]), t(d["row_cube_quat"])
            self.s_ep, self.s_t = t(d["row_ep"], torch.long), t(d["row_t"], torch.long)
            self.s_g = self.ep_step_start[self.s_ep] + self.s_t
        self.robot = torch.cat((self.q, self.qd), -1)
        row_val = torch.as_tensor(is_val, device=self.dev)[self.s_ep]
        self.train_rows = torch.nonzero(~row_val).flatten()
        self.val_rows = torch.nonzero(row_val).flatten()
        self.n_rows = len(self.q)

    def targets(self, rows):
        """-> (target [n, K, 6] = u_lab - q_t, valid [n, K] bool)."""
        e = self.s_ep[rows]
        K, sp = self.chunk, self.steps_per
        j = torch.arange(K, device=self.dev)
        t0 = self.s_t[rows]
        tloc = t0[:, None] + j[None]                                     # local step of target j
        inside = tloc < self.ep_steps[e][:, None]
        g = self.s_g[rows][:, None] + j[None]
        g = torch.minimum(g, (self.ep_step_start[e] + self.ep_steps[e] - 1)[:, None])
        tgt = self.u_lab[g] - self.q[rows][:, None, :]
        # boundaries crossed by target j: m = 1 .. (t0 % sp + j) // sp; boundary m = row t0 // sp + m of the episode
        crossed = (t0[:, None] % sp + j[None]) // sp
        nb = int(crossed.max()) if len(rows) else 0
        last_row = self.ep_row_start[e] + self.ep_rows[e] - 1
        base = self.ep_row_start[e] + t0 // sp
        ok = torch.ones(len(rows), K, dtype=torch.bool, device=self.dev)
        chain = torch.ones(len(rows), dtype=torch.bool, device=self.dev)
        for m in range(1, nb + 1):
            rr = base + m
            chain = chain & (self.cont[torch.minimum(rr, last_row)] | (rr > last_row))
            ok = ok & ~((crossed >= m) & ~chain[:, None])
        return tgt, inside & ok

    def batch(self, rows):
        e = self.s_ep[rows]
        tgt, valid = self.targets(rows)
        return dict(q=self.q[rows], cube_pos=self.cube_pos[rows], cube_quat=self.cube_quat[rows],
                    goal_pose=self.goal_pose[e], robot=self.robot[rows], target=tgt, valid=valid)

    def stats(self, floor=1e-3, robot_floor=1e-2, n=200_000, seed=0):
        g = torch.Generator(device=self.dev).manual_seed(seed)
        rows = self.train_rows[torch.randint(len(self.train_rows), (min(n, len(self.train_rows)),), device=self.dev,
                                             generator=g)]
        tgt, valid = self.targets(rows)
        v = valid[..., None].float()
        cnt = v.sum(0).clamp_min(1)
        mu = (tgt * v).sum(0) / cnt
        sd = (((tgt - mu) ** 2 * v).sum(0) / cnt).sqrt().clamp_min(floor)
        x = self.robot[rows]
        return dict(robot_mean=x.mean(0).cpu(), robot_std=x.std(0).clamp_min(robot_floor).cpu(), tgt_mean=mu.cpu(),
                    tgt_std=sd.cpu())

    def sample(self, n, gen=None, split="train"):
        rows = self.train_rows if split == "train" else self.val_rows
        return rows[torch.randint(len(rows), (int(n),), device=self.dev, generator=gen)]


# ---- students ----------------------------------------------------------------------------------------------------------
@dataclass
class TrayConfig:
    kind: str = "perstep"              # perstep | act
    chunk: int = 10                    # targets per query (20 ms each); the first n_exec are executed
    n_exec: int = 5                    # query period in 20 ms steps (5 = once per decision)
    step_rows: bool = False            # train on every step's state (needed for n_exec < 5), not only the boundaries
    ensemble: object = None            # execution with n_exec 1: temporal ensembling coefficient (None: off)
    jaw_lead: int = 0                  # execution: the jaw target taken this many steps ahead in the chunk(s)
    res: int = V.RES
    cameras: tuple = ("top", "wrist")
    robot_dim: int = 12                # 12: q (6) + qdot (6); 6: q only (the qdot ablation: no copycat input)
    qd_noise: float = 0.0              # training only: Gaussian noise on the normalised qdot inputs (std units)
    # per-step CNN
    encoder: str = "resnet18"
    stages: int = 3
    keypoints: int = 32
    pool_dim: int = 64
    hidden: tuple = (512, 512)
    # ACT (LeRobot)
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    kl_weight: float = 10.0
    dropout: float = 0.1
    # training
    batch: int = 128
    render_batch: int = 256
    lr: float = 3e-4
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    warmup: int = 300
    cosine: bool = True
    grad_clip: float = 10.0
    shift_px: int = 0
    amp: bool = True
    seed: int = 0

    @staticmethod
    def from_dict(d):
        d = dict(d or {})
        known = set(TrayConfig.__dataclass_fields__)
        bad = sorted(set(d) - known - {"note", "notes"})
        if bad:
            raise ValueError(f"unknown tray config keys {bad}")
        for k in ("cameras", "hidden"):
            if k in d:
                d[k] = tuple(d[k])
        return TrayConfig(**{k: v for k, v in d.items() if k in known})


STAT_KEYS = ("robot_mean", "robot_std", "tgt_mean", "tgt_std")


def robot_norm(st, robot):
    """Normalised robot vector: the first ``cfg.robot_dim`` entries of [q, qdot] (12: both; 6: q only); in training
    mode with ``cfg.qd_noise`` > 0, Gaussian noise of that std on the normalised qdot entries (against the copycat
    shortcut of behaviour cloning with rate inputs)."""
    k = st.cfg.robot_dim
    r = ((robot[:, :k] - st.robot_mean[:k]) / st.robot_std[:k]).clamp(-10, 10)
    if st.training and getattr(st.cfg, "qd_noise", 0.0) > 0 and k > 6:
        r = torch.cat((r[:, :6], r[:, 6:] + st.cfg.qd_noise * torch.randn_like(r[:, 6:])), -1)
    return r


class PerStepChunk(nn.Module):
    """Per camera ResNet18 (ImageNet, frozen BN, to layer3) -> spatial-softmax keypoints + pooled features; MLP over
    [cameras, normalised q / qdot] -> a normalised target chunk [K, 6]."""

    def __init__(self, cfg: TrayConfig, stats, pretrained=True):
        super().__init__()
        from nedm.so101_policy.vision.perstep import Encoder, PerStepConfig
        self.cfg = cfg
        pc = PerStepConfig(encoder=cfg.encoder, stages=cfg.stages, keypoints=cfg.keypoints, pool_dim=cfg.pool_dim)
        self.enc = nn.ModuleList([Encoder(pc, pretrained) for _ in cfg.cameras])
        d = sum(e.out_dim for e in self.enc) + cfg.robot_dim
        layers, k = [], d
        for h in cfg.hidden:
            layers += [nn.Linear(k, h), nn.ELU()]
            k = h
        layers.append(nn.Linear(k, cfg.chunk * 6))
        self.head = nn.Sequential(*layers)
        for k_, v in stats.items():
            self.register_buffer(k_, torch.as_tensor(v, dtype=torch.float32).clone())

    def forward(self, images, robot):
        feats = [e(images[:, i]) for i, e in enumerate(self.enc)]
        r = robot_norm(self, robot)
        return self.head(torch.cat(feats + [r.to(feats[0].dtype)], -1).float()).view(-1, self.cfg.chunk, 6)

    def loss(self, images, robot, tgt_n, valid):
        p = self(images, robot)
        e = (p.float() - tgt_n).abs() * valid[..., None]
        return e.sum() / (valid.sum() * 6).clamp_min(1), {}

    @torch.no_grad()
    def predict(self, images, robot):
        return self(images, robot).float()

    def backbone_params(self):
        return [p for e in self.enc if e.is_backbone for p in e.trunk.parameters() if p.requires_grad]


class ActChunk(nn.Module):
    """LeRobot ACT (0.6.1) on [q, qdot] (``observation.state``, 12) and the two cameras -> normalised chunk [K, 6]."""

    OBS_STATE, IMG = "observation.state", "observation.images."

    def __init__(self, cfg: TrayConfig, stats, device="cuda", pretrained=True):
        super().__init__()
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        self.cfg = cfg
        inp = {self.OBS_STATE: PolicyFeature(FeatureType.STATE, (cfg.robot_dim,))}
        for cam in cfg.cameras:
            inp[self.IMG + cam] = PolicyFeature(FeatureType.VISUAL, (3, cfg.res, cfg.res))
        out = {"action": PolicyFeature(FeatureType.ACTION, (6,))}
        ac = ACTConfig(input_features=inp, output_features=out, chunk_size=cfg.chunk, n_action_steps=1,
                       dim_model=cfg.dim_model, n_heads=cfg.n_heads, dim_feedforward=cfg.dim_feedforward,
                       n_encoder_layers=cfg.n_encoder_layers, n_decoder_layers=cfg.n_decoder_layers,
                       use_vae=cfg.use_vae, latent_dim=cfg.latent_dim, n_vae_encoder_layers=cfg.n_vae_encoder_layers,
                       kl_weight=cfg.kl_weight, dropout=cfg.dropout, temporal_ensemble_coeff=None,
                       pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1" if pretrained else None,
                       device=str(device))
        self.policy = ACTPolicy(ac)
        for k_, v in stats.items():
            self.register_buffer(k_, torch.as_tensor(v, dtype=torch.float32).clone())

    def _batch(self, images, robot, tgt_n=None, valid=None):
        b = {self.OBS_STATE: robot_norm(self, robot)}
        for i, cam in enumerate(self.cfg.cameras):
            b[self.IMG + cam] = images[:, i]
        if tgt_n is not None:
            b["action"] = tgt_n
            b["action_is_pad"] = ~valid
        return b

    def loss(self, images, robot, tgt_n, valid):
        from lerobot.utils.constants import OBS_IMAGES
        b = self._batch(images, robot, tgt_n, valid)
        b[OBS_IMAGES] = [b[k] for k in self.policy.config.image_features]
        hat, (mu, logs) = self.policy.model(b)
        m = valid[..., None].float()
        l1 = ((tgt_n - hat.float()).abs() * m).sum() / (m.sum() * 6).clamp_min(1)
        info = {"l1": float(l1.detach())}
        loss = l1
        if self.cfg.use_vae and logs is not None:
            kld = (-0.5 * (1 + logs - mu.pow(2) - logs.exp())).sum(-1).mean()
            info["kld"] = float(kld.detach())
            loss = loss + kld * self.cfg.kl_weight
        return loss, info

    @torch.no_grad()
    def predict(self, images, robot):
        from lerobot.utils.constants import OBS_IMAGES
        b = self._batch(images, robot)
        b[OBS_IMAGES] = [b[k] for k in self.policy.config.image_features]
        hat, _ = self.policy.model(b)
        return hat.float()

    def backbone_params(self):
        return [p for n, p in self.policy.named_parameters() if n.startswith("model.backbone") and p.requires_grad]


def build(cfg: TrayConfig, stats, device="cuda", pretrained=True):
    if cfg.kind == "perstep":
        return PerStepChunk(cfg, stats, pretrained).to(device)
    if cfg.kind == "act":
        return ActChunk(cfg, stats, device, pretrained).to(device)
    raise ValueError(f"student kind {cfg.kind!r}")


def save_student(st, path, **extra):
    torch.save(dict(kind=st.cfg.kind, cfg=asdict(st.cfg), state_dict=st.state_dict(),
                    stats={k: getattr(st, k).cpu() for k in STAT_KEYS}, **extra), path)


def load_student(path, device="cuda"):
    ck = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    cfg = TrayConfig.from_dict(ck["cfg"])
    st = build(cfg, ck["stats"], device, pretrained=False)
    st.load_state_dict(ck["state_dict"])
    return st.eval(), ck


def normalise(st, tgt):
    return (tgt - st.tgt_mean) / st.tgt_std


def denormalise(st, p):
    return p * st.tgt_std + st.tgt_mean


@torch.no_grad()
def val_metrics(st, data, views, rows, bs=256):
    """Mean absolute error (rad) of the de-normalised targets: first n_exec steps and the whole chunk, per joint."""
    e1, c1 = torch.zeros(6, device=data.dev), torch.zeros(6, device=data.dev)
    ea, ca = torch.zeros(6, device=data.dev), torch.zeros(6, device=data.dev)
    ne = st.cfg.n_exec
    for s in range(0, len(rows), bs):
        b = data.batch(rows[s:s + bs])
        img = V.to_input(views.render(b["q"], b["cube_pos"], b["cube_quat"], b["goal_pose"]))
        with torch.autocast(data.dev.type, dtype=torch.bfloat16, enabled=st.cfg.amp):
            p = denormalise(st, st.predict(img, b["robot"]))
        err = (p - b["target"]).abs() * b["valid"][..., None]
        e1 += err[:, :ne].sum((0, 1))
        c1 += b["valid"][:, :ne].sum()
        ea += err.sum((0, 1))
        ca += b["valid"].sum()
    out = {f"val_mae_exec_j{j}": float(e1[j] / c1[j].clamp_min(1)) for j in range(6)}
    out["val_mae_exec"] = float(e1.sum() / c1.sum().clamp_min(1))
    out["val_mae_chunk"] = float(ea.sum() / ca.sum().clamp_min(1))
    return out


def train(data: TraySet, views, cfg: TrayConfig, out_dir, max_steps=4000, init_from=None, log=log_default,
          val_n=2048, val_every=500, log_every=100, device="cuda", extra_meta=None):
    """Train (or fine-tune from ``init_from``: weights AND normalisation) a tray student. Writes out_dir/{meta.json,
    learning_curve.csv, train_summary.json, last.pt}. -> student (eval mode)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype_dev = torch.device(device).type
    cuda = dtype_dev == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    if data.chunk != cfg.chunk:
        raise ValueError(f"data chunk {data.chunk} != config chunk {cfg.chunk}")
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    if init_from:
        st, _ = load_student(init_from, device)
        st.cfg = cfg
    else:
        st = build(cfg, data.stats(), device, pretrained=True)
    st.train()
    bb = st.backbone_params()
    bb_ids = {id(p) for p in bb}
    rest = [p for p in st.parameters() if p.requires_grad and id(p) not in bb_ids]
    groups = [dict(params=rest, lr=cfg.lr)] + ([dict(params=bb, lr=cfg.lr_backbone)] if bb else [])
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    base = [g["lr"] for g in opt.param_groups]
    vgen = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    vrows = data.sample(min(val_n, max(1, len(data.val_rows))), vgen, "val") if len(data.val_rows) else None
    meta = dict(cfg=asdict(cfg), data_meta=data.meta, n_train_eps=data.n_train_eps, n_val_eps=data.n_val_eps,
                train_rows=len(data.train_rows), val_rows=len(data.val_rows), init_from=str(init_from or ""),
                **(extra_meta or {}))
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    f = open(out / "learning_curve.csv", "w", newline="")
    wr = None
    t0 = time.time()
    t_render = t_train = 0.0
    step, samples = 0, 0
    buf, pos = None, 0
    agg_loss, agg_n = 0.0, 0
    while step < max_steps:
        if buf is None or pos + cfg.batch > cfg.render_batch:
            tr = time.time()
            b = data.batch(data.sample(cfg.render_batch, gen))
            b["rgb"] = views.render(b["q"], b["cube_pos"], b["cube_quat"], b["goal_pose"])
            b["tgt_n"] = normalise(st, b["target"])
            sync()
            t_render += time.time() - tr
            buf, pos = b, 0
        tt = time.time()
        sl = slice(pos, pos + cfg.batch)
        pos += cfg.batch
        img = V.to_input(buf["rgb"][sl], cfg.shift_px, gen)
        frac = min(1.0, (step + 1) / max(1, cfg.warmup))
        if cfg.cosine and step >= cfg.warmup:
            u = (step - cfg.warmup) / max(1, max_steps - cfg.warmup)
            frac = 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * u))
        for g_, lr0 in zip(opt.param_groups, base):
            g_["lr"] = lr0 * frac
        with torch.autocast(dtype_dev, dtype=torch.bfloat16, enabled=cfg.amp):
            loss, _ = st.loss(img, buf["robot"][sl], buf["tgt_n"][sl], buf["valid"][sl])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(st.parameters(), cfg.grad_clip)
        opt.step()
        sync()
        t_train += time.time() - tt
        step += 1
        samples += cfg.batch
        agg_loss += float(loss.detach())
        agg_n += 1
        do_val = vrows is not None and (step % val_every == 0 or step == max_steps)
        if step % log_every == 0 or do_val:
            row = dict(step=step, samples=samples, wall_min=round((time.time() - t0) / 60, 3), loss=agg_loss / agg_n,
                       grad_norm=float(gn), lr=opt.param_groups[0]["lr"], render_s=round(t_render, 1),
                       train_s=round(t_train, 1), samples_per_s=round(samples / max(time.time() - t0, 1e-9), 1),
                       gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else 0.0)
            agg_loss, agg_n = 0.0, 0
            if do_val:
                st.eval()
                row.update(val_metrics(st, data, views, vrows))
                st.train()
            if wr is None:
                vk = ["val_mae_exec", "val_mae_chunk"] + [f"val_mae_exec_j{j}" for j in range(6)]
                wr = csv.DictWriter(f, fieldnames=[k for k in row if k not in vk] + vk, extrasaction="ignore")
                wr.writeheader()
            wr.writerow(row)
            f.flush()
            log(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}))
    st.eval()
    save_student(st, out / "last.pt", step=step, samples=samples, wall_min=(time.time() - t0) / 60)
    f.close()
    summ = dict(steps=step, samples=samples, wall_min=(time.time() - t0) / 60, render_s=t_render, train_s=t_train,
                views=views.views, views_per_s=views.views / max(views.seconds, 1e-9))
    (out / "train_summary.json").write_text(json.dumps(summ, indent=1))
    log(json.dumps(summ))
    return st


class TrayActor:
    """The student as the driver of ``TrayWorld.run``: renders the worlds' states (top + wrist with the tray), feeds
    [q, qdot] -> absolute joint targets [B, K, 6] (= q + de-normalised chunk)."""

    def __init__(self, student, views, device="cuda"):
        self.st, self.views, self.dev = student, views, torch.device(device)
        self.t_render = self.t_model = 0.0
        self.calls = 0

    def reset(self, B):
        pass

    @torch.no_grad()
    def __call__(self, ws, goal_pose):
        t0 = time.time()
        q = ws.q.to(self.dev, torch.float32)
        cp, cq = ws.cube_pos.to(self.dev, torch.float32), ws.cube_quat.to(self.dev, torch.float32)
        ok = torch.isfinite(q).all(-1) & torch.isfinite(cp).all(-1) & torch.isfinite(cq).all(-1)
        q = torch.where(ok[:, None], q, torch.zeros_like(q))
        cp = torch.where(ok[:, None], cp, torch.tensor([0.2, 0.0, 0.015], device=self.dev))
        cq = torch.where(ok[:, None], cq, torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.dev))
        rgb = self.views.render(q, cp, cq, goal_pose.to(self.dev, torch.float32))
        img = V.to_input(rgb)
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        robot = torch.cat((q, ws.qd.to(self.dev, torch.float32).nan_to_num()), -1)
        out = []
        for s in range(0, img.shape[0], 256):
            with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.st.cfg.amp):
                out.append(denormalise(self.st, self.st.predict(img[s:s + 256], robot[s:s + 256])))
        delta = torch.cat(out)
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        self.t_render += t1 - t0
        self.t_model += time.time() - t1
        self.calls += 1
        return (q[:, None, :] + delta).to(ws.q.device, torch.float64)


@torch.no_grad()
def evaluate(actor, tasks, world, teacher, device="cuda", nrd_ckpt=None, workers=0, mj_python=None, horizon=400,
             n_exec=5, record_states=False, log=None, tw=None, decoder=None, ensemble=None, jaw_lead=0,
             full_horizon=False):
    """The student (beta 0) or the teacher (actor None) on all tasks (one batch) -> ``run`` output + summary."""
    B = len(tasks["seeds"])
    own = tw is None
    if own:
        tw = TrayWorld(world, B, teacher, device, nrd_ckpt, workers, mj_python, horizon, decoder=decoder)
    w0 = tw.world_steps
    r = tw.run(tasks, actor, 0.0 if actor is not None else 1.0, None, 0, True, n_exec, record_states, log,
               ensemble=ensemble, jaw_lead=jaw_lead, full_horizon=full_horizon)
    r["ensemble"] = np.float64(np.nan if ensemble is None else ensemble)
    r["jaw_lead"] = np.int64(jaw_lead)
    r["world_steps"] = np.int64(tw.world_steps - w0)
    if own:
        tw.close()
    if actor is not None:
        r["render_s"], r["model_s"] = np.float64(actor.t_render), np.float64(actor.t_model)
    r["world"] = np.array(world)
    r["decoder_over"] = np.array(json.dumps(tw.decoder_over, default=str))
    r["summary"] = np.array(json.dumps(summarize(r)))
    return r
