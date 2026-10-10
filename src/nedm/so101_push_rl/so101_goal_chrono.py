"""Chrono episodes of the SO-101 push-T goal task: physical prefix replay, then a controller from the start frame.

One Chrono step server per episode (nedm.so101_push.chrono_server). The recorded commands of steps 0 .. j_a-1 are
replayed after the collector settle (no teleport into the start state); from frame j_a the controller acts every
20 ms on the MEASURED history (step-average rates on the 20 ms grid, as the NRD data). The controller is an object with
reset(ctx) and act(ctx) -> q_cmd [5] (the planar pusher policy: nedm.so101_push_rl.so101_planar_chrono.PlanarPolicy).

Validity in Chrono: arm-table contact, contact of a non-finger link with the T, T height / tilt outside the collector
QA bounds. Success as the learned environment: inside the tolerance and slow for hold_steps consecutive steps.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from nedm.so101_push_rl import so101_goal_common as G
from nedm.so101_push_rl.so101_push_chrono_tracking_env import FINGER_LINKS, ChronoServer, MeasuredHistory


def run_goal_episode(code_root, start, goal, horizon, controller, success_cfg=None, translation_only=True,
                     qa=None, extra_steps=0, goal_switch=None, render=None):
    """start: dict(q_start, first_cmd, prefix_cmds [j_a, 5], j_a, prev_cmd, p_line, dir_w, gripper_yaw, q_des_prev,
    q_des_cur); goal [3] (x, y, yaw) or None (goal generation). Runs the prefix, then the controller for horizon
    (+ extra_steps) steps. Returns arrays and per-step success flags."""
    sc = dict(pos_tol=0.005, yaw_tol_deg=3.0, v_tol=0.005, w_tol_deg=5.0, hold_steps=5, **(success_cfg or {}))
    qa = qa or {"t_z_tolerance_m": [0.002, 0.01], "max_t_tilt_deg": 15.0}
    srv = ChronoServer(code_root, start["q_start"], start["first_cmd"], render=render)
    try:
        mh = MeasuredHistory(srv.first)
        prefix = np.asarray(start["prefix_cmds"], float)
        for k in range(len(prefix)):
            mh.add(srv.step(prefix[k]))
        assert len(mh.frames) == int(start["j_a"]) + 1
        prev = np.asarray(start["prev_cmd"], float)
        ctx = dict(start=start, goal=np.zeros(3) if goal is None else np.asarray(goal, float), horizon=int(horizon), prev_cmd=prev)
        controller.reset(ctx)
        cmds, lat = [], []
        for t in range(int(horizon) + int(extra_steps)):
            hist, ok = mh.history()
            ctx.update(t=t, hist=hist, ok=ok, prev_cmd=prev)
            if goal_switch is not None and t == int(goal_switch[0]):
                ctx["goal"] = np.asarray(goal_switch[1], float)          # the controller sees the new goal from now on
            t0 = time.perf_counter()
            cmd = np.asarray(controller.act(ctx), float)
            lat.append(time.perf_counter() - t0)
            mh.add(srv.step(cmd))
            cmds.append(cmd)
            prev = cmd
    finally:
        srv.close()
    S = np.asarray(mh.frames)[int(start["j_a"]):]                    # frames j_a .. end (S[0] = start state)
    con = mh.contacts[int(start["j_a"]):]
    arm_table = np.array([c["arm_table"] for c in con])
    other = np.array([c["links"][[i for i in range(7) if i not in FINGER_LINKS]].any() for c in con])
    tz = S[1:, 12]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (S[1:, 14] ** 2 + S[1:, 15] ** 2), -1, 1)))
    t_bad = (tz < 0.015 - qa["t_z_tolerance_m"][0]) | (tz > 0.015 + qa["t_z_tolerance_m"][1]) | (tilt > qa["max_t_tilt_deg"])
    out = dict(states=S, cmds=np.asarray(cmds), arm_table=arm_table, other_links=other, t_bad=t_bad,
               invalid=arm_table | other | t_bad, latency_s=np.asarray(lat),
               interventions=int(getattr(controller, "interventions", 0)))
    if goal is not None:
        g_eval = np.asarray(goal if goal_switch is None else goal_switch[1], float)    # judged against the final goal
        out.update(goal_metrics(S, g_eval, int(horizon), out["invalid"], sc, translation_only))
    return out


def goal_metrics(S, goal, horizon, invalid, sc, translation_only):
    """Success: first step t (1..horizon) where the T was inside the tolerance and slow for hold_steps consecutive
    steps, with no invalid event up to t. Also final errors at the deadline."""
    s = torch.as_tensor(S[1:horizon + 1])
    g = torch.as_tensor(goal)[None].expand(len(s), 3)
    inside = G.settled_inside(s, g, not translation_only, sc["pos_tol"], sc["yaw_tol_deg"], sc["v_tol"], sc["w_tol_deg"]).numpy()
    run, t_succ = 0, -1
    for t in range(len(inside)):
        run = run + 1 if inside[t] else 0
        if run >= sc["hold_steps"]:
            t_succ = t + 1
            break
    inv_before = bool(invalid[: (t_succ if t_succ > 0 else horizon)].any())
    pe, ye = G.goal_error(s, g)
    return dict(success=bool(t_succ > 0 and not inv_before), success_step=int(t_succ), invalid_any=bool(invalid[:horizon].any()),
                final_pos_mm=float(1e3 * pe[-1]), final_yaw_deg=float(np.degrees(abs(float(ye[-1])))),
                min_pos_mm=float(1e3 * pe.min()), final_t_speed_mps=float(s[-1, 17:19].norm()),
                start_pos_mm=float(1e3 * np.linalg.norm(S[0, 10:12] - goal[:2])))
