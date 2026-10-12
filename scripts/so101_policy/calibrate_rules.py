"""Calibrate the geometric task rules (``nedm.so101_policy.task``) on recorded grasp data; fit the servo response of
the kinematic control world (``nedm.so101_policy.kin_world``).

    PYTHONPATH=src python -P scripts/so101_policy/calibrate_rules.py trace --data DIR --out trace.npz [--device cuda]
    PYTHONPATH=src python -P scripts/so101_policy/calibrate_rules.py stats --trace trace.npz --out stats.json
    PYTHONPATH=src python -P scripts/so101_policy/calibrate_rules.py rules --data DIR --out rules.json [--device cuda]
    PYTHONPATH=src python -P scripts/so101_policy/calibrate_rules.py terminal --data DIR [--data DIR2 --split 0] \
        --out terminal.json
    PYTHONPATH=src python -P scripts/so101_policy/calibrate_rules.py nrd --data DIR --out nrd.json [--device cuda]
    PYTHONPATH=src python -P scripts/so101_policy/calibrate_rules.py servo --train DIR --val DIR --out servo.json

Data: ``unified_data.npz`` of the grasp collector (10 ms records). Everything runs on the 20 ms control grid (records
0, 2, 4, ...; ``task.grid_from_records``): the state after control step j is record 2 j; its macro phase is the
collector phase of the command applied in that step (record 2 j - 2) mapped by ``task.COLLECTOR_MACRO``; its contact
labels (only used to SELECT records for statistics, never by a rule) are the OR of the intervals 2 j - 2 and 2 j - 1.
Use validation data (``~/so101_nrd_fix/data/val_only`` = production split 1) for thresholds; the training split only
adds terminal episodes for detection counts; never the sealed test.

- ``trace``: per-step rule inputs (``task.features``) of every episode -> npz.
- ``stats``: the distributions that set each threshold (per family).
- ``rules``: the task (all rules log-only, goal = the episode's final cube pose = hindsight) over every episode ->
  per-rule false fires (nominal, off-centre: per episode and per step), detection on known-bad windows, success
  with hindsight goals.
- ``terminal``: instantaneous rules (lowest point, link overlap, jaw overlap, base zone) at the last stored record of
  the episodes the collector ended (arm_table, cube_other_link, cube_visual_overlap, cube_base) and at that record
  extrapolated by one 10 ms record (the collector stores the record BEFORE the contact or overlap).
- ``nrd``: open-loop NRD rollouts (recorded commands) of near-miss / weak-closure / one-sided validation episodes:
  phantom lifts (NRD cube up while the MuJoCo cube stays on the table) as known-bad windows of the support rules.
- ``servo``: least-squares fit of the per-joint servo response, errors at 20 ms and 1 s on validation episodes.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy import task as T

FAM = ("nominal", "off_centre", "one_sided", "near_miss", "weak_closure", "drop", "push", "free_arm", "cube_flight",
       "cube_table", "jaw_fixed", "jaw_moving")
TERMINAL = ("end", "arm_table", "cube_other_link", "arm_self", "cube_base", "cube_off_table", "cube_visual_overlap")
GOOD = (0, 1)        # nominal, off-centre: false-fire set


def load(data_dir, rows=None, keys=("arm", "cube", "action", "contacts", "lengths", "phase", "family", "terminal_reason",
                                     "outcome", "variant", "splits", "cube_in_grasp", "visual_overlap")):
    from nedm.contact_nrd.rollouts import npz_members
    f = npz_members(Path(data_dir) / "unified_data.npz")
    sysj = json.loads((Path(data_dir) / "system.json").read_text())
    out = {}
    for k in keys:
        a = f[k]
        out[k] = np.asarray(a) if rows is None else np.asarray(a[np.sort(rows)] if a.ndim else a)
    out["phases"] = sysj["phases"]
    return out


def macro_grid(phase, phases):
    """Collector phase codes [N, T] -> macro codes of the step ending at each grid point [N, J]."""
    lut = np.zeros(len(phases), np.int64)
    for k, p in enumerate(phases):
        lut[k] = T.MC[T.COLLECTOR_MACRO[p]] if p in T.COLLECTOR_MACRO else 0
    ph = phase[:, ::2]
    prev = np.concatenate((ph[:, :1], ph[:, :-1]), 1)    # command applied in the step just taken
    return lut[prev]


def contact_grid(contacts):
    """Record-interval labels [N, T-1, 3] -> per grid step OR of the 2 intervals [N, J, 3] (j = 0: False)."""
    c = contacts.astype(bool)
    N, Ti, _ = c.shape
    J = (Ti + 1 + 1) // 2
    out = np.zeros((N, J, 3), bool)
    for j in range(1, J):
        out[:, j] = c[:, 2 * j - 2] | (c[:, 2 * j - 1] if 2 * j - 1 < Ti else False)
    return out


# ---- trace -----------------------------------------------------------------------------------------------------------
KEYS = ("x", "y", "z", "jaw", "bottom", "tilt", "vxy", "vz", "speed", "tcpz", "dxy", "yaw_mis", "hx", "hy", "lowest",
        "link_depth", "jaw_overlap", "cyaw", "gyaw", "held", "in_box", "on_table")


def trace_features(d, device, cfg=None, chunk=4096):
    """Per-step rule inputs [N, J] (float32 numpy) of all episodes of a loaded dataset."""
    cfg = cfg or T.TaskConfig()
    kin = T.SO101Kin(device)
    geom = T.TaskGeometry.from_mujoco(device)
    N = len(d["lengths"])
    grid = T.grid_from_records(d["arm"], d["cube"])
    J = grid["q"].shape[1]
    out = {k: np.zeros((N, J), np.float32) for k in KEYS}
    for s in range(0, N, chunk):
        sub = {k: v[s: s + chunk] for k, v in grid.items()}
        for j in range(J):
            F = T.features(T.grid_state(sub, j, device), kin, geom, cfg)
            vals = dict(x=F["xg"][:, 0], y=F["xg"][:, 1], z=F["xg"][:, 2], jaw=F["jaw"], bottom=F["bottom"],
                        tilt=F["tilt"], vxy=F["vxy"], vz=F["v"][:, 2], speed=F["speed"], tcpz=F["tcp"][:, 2],
                        dxy=F["dxy"], yaw_mis=F["yaw_mis"], hx=F["hx"], hy=F["hy"], lowest=F["lowest"],
                        link_depth=F["link_depth"], jaw_overlap=F["jaw_overlap"], cyaw=F["cyaw"], gyaw=F["gyaw"],
                        held=F["held"], in_box=F["in_box"], on_table=F["on_table"])
            for k in KEYS:
                out[k][s: s + chunk, j] = vals[k].float().cpu().numpy()
    out["valid"] = (2 * np.arange(J)[None, :]) < d["lengths"][:, None]
    out["macro"] = macro_grid(d["phase"], d["phases"])
    out["contact"] = contact_grid(d["contacts"])
    out["phase"] = d["phase"][:, ::2]
    for k in ("family", "terminal_reason", "outcome", "lengths"):
        out[k] = d[k]
    out["variant"] = d["variant"].astype(str)
    return out


def cmd_trace(a):
    t0 = time.time()
    d = load(a.data)
    tr = trace_features(d, a.device)
    np.savez_compressed(a.out, phases=np.array(d["phases"]), **tr)
    print(f"trace {len(d['lengths'])} episodes -> {a.out} ({time.time() - t0:.0f} s)")


# ---- statistics behind the thresholds ---------------------------------------------------------------------------------
def pct(x, q=(0, 0.5, 1, 50, 99, 99.5, 100)):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return {f"p{p:g}": round(float(np.percentile(x, p)), 6) for p in q} | {"n": int(len(x))} if len(x) else {"n": 0}


def runs(mask):
    """Longest run of True per row [N]."""
    best = np.zeros(mask.shape[0], np.int64)
    cur = np.zeros(mask.shape[0], np.int64)
    for j in range(mask.shape[1]):
        cur = np.where(mask[:, j], cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def cmd_stats(a):
    t = dict(np.load(a.trace, allow_pickle=False))
    phases = list(t["phases"])
    fam, valid, ph = t["family"], t["valid"], t["phase"]
    con = t["contact"]
    gripped = con[..., 0] & con[..., 1] & ~con[..., 2] & valid
    P = {p: k for k, p in enumerate(phases)}
    out = {}
    for k in range(7):
        e = fam == k
        g = gripped[e]
        o = {}
        o["gripped_x_mm"] = pct(np.abs(t["x"][e][g]) * 1e3)
        o["gripped_y_mm"] = pct(np.abs(t["y"][e][g]) * 1e3)
        o["gripped_z_mm"] = pct(t["z"][e][g] * 1e3)
        o["gripped_jaw"] = pct(t["jaw"][e][g])
        up = g & (t["bottom"][e] > 0.002)
        o["gripped_air_run_s_per_ep"] = pct(runs(up) * 0.02)
        geo_air = (t["held"][e] > 0.5) & (t["bottom"][e] > 0.002) & valid[e]
        o["held_air_run_s_per_ep"] = pct(runs(geo_air) * 0.02)
        o["held_frac_of_gripped_air"] = round(float((t["held"][e][up] > 0.5).mean()), 6) if up.any() else None
        # close start and trigger (first step of the close / descend phase)
        for name, key in (("close", "close"), ("descend", "descend")):
            first = []
            for i in np.flatnonzero(e):
                js = np.flatnonzero((ph[i] == P[key]) & valid[i])
                if len(js):
                    first.append((i, js[0]))
            if first:
                ii, jj = np.array(first).T
                o[f"{name}_start_x_mm"] = pct(np.abs(t["x"][ii, jj]) * 1e3)
                o[f"{name}_start_y_mm"] = pct(np.abs(t["y"][ii, jj]) * 1e3)
                o[f"{name}_start_z_mm"] = pct(t["z"][ii, jj] * 1e3)
                o[f"{name}_start_hx_mm"] = pct(np.abs(t["hx"][ii, jj]) * 1e3)
                o[f"{name}_start_hy_mm"] = pct(np.abs(t["hy"][ii, jj]) * 1e3)
                o[f"{name}_start_yaw_mis_deg"] = pct(np.degrees(t["yaw_mis"][ii, jj]))
                o[f"{name}_start_jaw"] = pct(t["jaw"][ii, jj])
                o[f"{name}_start_tcpz_mm"] = pct(t["tcpz"][ii, jj] * 1e3)
        # release: cube bottom at the first release step; at the first not-held step after holding
        rel = []
        for i in np.flatnonzero(e):
            js = np.flatnonzero((ph[i] == P["release"]) & valid[i])
            if len(js):
                rel.append(t["bottom"][i, js[0]])
        o["release_start_bottom_mm"] = pct(np.array(rel) * 1e3)
        held = (t["held"][e] > 0.5) & valid[e]
        drop_after = []
        for r in range(held.shape[0]):
            h = held[r]
            js = np.flatnonzero(h[:-1] & ~h[1:]) + 1
            for j in js:
                drop_after.append(t["bottom"][e][r, j])
        o["unheld_transition_bottom_mm"] = pct(np.array(drop_after) * 1e3)
        notheld_air = ~held & (t["bottom"][e] > 0.005) & valid[e]
        o["ep_frac_notheld_air_5mm"] = round(float(notheld_air.any(1).mean()), 6)
        # free motion near a resting cube
        free = (t["macro"][e] == 0) & valid[e]
        rest = (t["on_table"][e] > 0.5) & (t["held"][e] < 0.5)
        near = free & rest & (t["dxy"][e] < 0.05)
        o["free_near_cube_tcpz_mm_min_per_ep"] = pct(np.where(near, t["tcpz"][e], np.inf).min(1)[near.any(1)] * 1e3)
        o["free_near_cube_tcpz_mm_records"] = pct(t["tcpz"][e][near] * 1e3)
        # cube on the table, not held: tilt and slide
        tab = (t["on_table"][e] > 0.5) & ~held & valid[e]
        o["table_tilt_deg_records"] = pct(np.degrees(t["tilt"][e][tab]))
        o["table_tilt15_run_steps_per_ep"] = pct(runs(tab & (t["tilt"][e] > math.radians(15))))
        o["table_tilt5_run_steps_per_ep"] = pct(runs(tab & (t["tilt"][e] > math.radians(5))))
        outbox = tab & (t["in_box"][e] < 0.5)
        o["table_outbox_vxy_mm_s_records"] = pct(t["vxy"][e][outbox] * 1e3, (50, 99, 99.9, 100))
        o["table_outbox_vxy_mm_s_max_per_ep"] = pct(np.where(outbox, t["vxy"][e], 0).max(1) * 1e3)
        o["lowest_mm_min_per_ep"] = pct(np.where(valid[e], t["lowest"][e], np.inf).min(1) * 1e3)
        o["link_depth_mm_max_per_ep"] = pct(np.where(valid[e], t["link_depth"][e], -np.inf).max(1) * 1e3)
        o["jaw_overlap_mm_max_per_ep"] = pct(np.where(valid[e], t["jaw_overlap"][e], 0).max(1) * 1e3)
        lift = runs(held & (t["bottom"][e] >= 0.02))
        o["firm_lift_steps_per_ep"] = pct(lift)
        out[FAM[k]] = o
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, o in out.items():
        print(f"== {k}")
        for n, v in o.items():
            print(f"   {n}: {v}")


# ---- the task's rules over recorded episodes -------------------------------------------------------------------------
def run_rules(d, device, cfg=None):
    """Replay every episode through ``PickPlaceTask`` (all rules log-only, hindsight goal = the final cube pose,
    macros from the collector phases) -> per-episode arrays: first fire step per rule (-1 never), steps fired per rule,
    first success step (-1 never), trigger / close box results."""
    base = cfg or T.TaskConfig()
    cfg = T.TaskConfig(**{**base.__dict__, "log_only": T.RULES, "horizon": 10 ** 6})
    kin = T.SO101Kin(device)
    geom = T.TaskGeometry.from_mujoco(device)
    grid = T.grid_from_records(d["arm"], d["cube"])
    N, J = grid["q"].shape[:2]
    L = torch.as_tensor(d["lengths"], device=device)
    last = ((L - 1) // 2).clamp_min(0)
    fin = T.grid_state({k: v.to(device) for k, v in grid.items()}, 0)
    idx = torch.arange(N, device=device)
    fq = grid["cube_quat"].to(device)[idx, last]
    gyaw, _ = T.cube_yaw_tilt(fq / fq.norm(dim=-1, keepdim=True))
    gxy = grid["cube_pos"].to(device)[idx, last, :2]
    macro = torch.as_tensor(macro_grid(d["phase"], d["phases"]), device=device)
    task = T.PickPlaceTask(N, device, kin=kin, geom=geom, cfg=cfg)
    task.reset(idx, fin, gxy, gyaw)
    R = len(T.RULES)
    first = torch.full((N, R), -1, dtype=torch.long, device=device)
    count = torch.zeros(N, R, dtype=torch.long, device=device)
    succ = torch.full((N,), -1, dtype=torch.long, device=device)
    trig_bad = torch.zeros(N, dtype=torch.bool, device=device)
    close_bad = torch.zeros(N, dtype=torch.bool, device=device)
    trig_seen = torch.zeros(N, dtype=torch.bool, device=device)
    close_seen = torch.zeros(N, dtype=torch.bool, device=device)
    lifted = torch.zeros(N, dtype=torch.bool, device=device)
    prev_m = macro[:, 0]
    for j in range(1, J):
        valid = 2 * j < L
        st = T.grid_state({k: v.to(device) for k, v in grid.items()}, j)
        _, _, _, info = task.step(st, macro[:, j])
        fl = info["flags"] & valid[:, None]
        count += fl.long()
        first = torch.where((first < 0) & fl, j, first)
        succ = torch.where((succ < 0) & info["success"] & valid, j, succ)
        lifted |= info["lifted"] & valid
        F = task.F
        tg = valid & (prev_m == 0) & (macro[:, j] == T.MC["grasp_descend"])
        cl = valid & (prev_m != T.MC["grasp_close"]) & (macro[:, j] == T.MC["grasp_close"])
        trig_bad |= tg & ~T.trigger_box_ok(F, cfg)
        close_bad |= cl & ~T.close_box_ok(F, cfg)
        trig_seen |= tg
        close_seen |= cl
        prev_m = macro[:, j]
    c = lambda x: x.cpu().numpy()                                                   # noqa: E731
    return dict(first=c(first), count=c(count), success=c(succ), trig_bad=c(trig_bad), close_bad=c(close_bad),
                trig_seen=c(trig_seen), close_seen=c(close_seen), lifted=c(lifted),
                steps=c(((L - 1) // 2).clamp_min(0)))


def rate(m, sel):
    n = int(sel.sum())
    return dict(k=int((m & sel).sum()), n=n, pct=round(100.0 * float((m & sel).sum()) / max(n, 1), 3))


def cmd_rules(a):
    t0 = time.time()
    rows = None
    if a.split is not None:
        from nedm.contact_nrd.rollouts import npz_members
        rows = np.flatnonzero(np.asarray(npz_members(Path(a.data) / "unified_data.npz")["splits"]) == a.split)
    d = load(a.data, rows)
    names = [f["name"] for f in json.loads((Path(a.data) / "system.json").read_text())["families"]]
    if names[:12] != list(FAM):
        raise ValueError("expected the grasp collector's family order")
    r = run_rules(d, a.device)
    fam, var, out_ = d["family"], d["variant"].astype(str), d["outcome"]
    ep = r["first"] >= 0
    res = dict(data=str(a.data), episodes=int(len(fam)), config=T.TaskConfig().__dict__, per_rule={}, known_bad={})
    good = np.isin(fam, GOOD)
    steps = r["steps"]
    for k, rule in enumerate(T.RULES):
        row = {FAM[f]: rate(ep[:, k], fam == f) for f in range(12)}
        row.update({names[f]: rate(ep[:, k], fam == f) for f in set(int(x) for x in np.unique(fam)) if f >= 12})
        row["false_fires_episodes_nominal_offcentre"] = rate(ep[:, k], good)
        row["false_fires_steps_nominal_offcentre"] = dict(k=int(r["count"][good, k].sum()), n=int(steps[good].sum()),
                                                          pct=round(100.0 * r["count"][good, k].sum() / steps[good].sum(), 4))
        res["per_rule"][rule] = row
    any_rule = ep[:, [T.RULES.index(x) for x in T.RULES if x not in ("world_invalid", "world_arm_table")]].any(1)
    res["any_rule_false_fires_episodes_nominal_offcentre"] = rate(any_rule, good)
    R = {x: ep[:, T.RULES.index(x)] for x in T.RULES}
    kb = res["known_bad"]
    kb["grasp_box: near_miss episodes (trigger or close start)"] = rate(R["grasp_box"], fam == 3)
    kb["grasp_box at the trigger only: near_miss"] = rate(r["trig_bad"], (fam == 3) & r["trig_seen"])
    kb["grasp_box at the close start only: near_miss"] = rate(r["close_bad"], (fam == 3) & r["close_seen"])
    kb["release_high|unsupported: drop family"] = rate(R["release_high"] | R["unsupported"], fam == 5)
    kb["release_high|unsupported: weak_closure lifted (outcome held or dropped)"] = rate(
        R["release_high"] | R["unsupported"], (fam == 4) & np.isin(out_, (2, 3)))
    kb["release_high|unsupported: near_miss lifted (outcome held or dropped)"] = rate(
        R["release_high"] | R["unsupported"], (fam == 3) & np.isin(out_, (2, 3)))
    kb["cube_slide: push family"] = rate(R["cube_slide"], fam == 6)
    kb["cube_slide: one_sided push variant"] = rate(R["cube_slide"], (fam == 2) & (var == "push"))
    kb["keep_out: push family"] = rate(R["keep_out"], fam == 6)
    kb["keep_out: one_sided (all variants)"] = rate(R["keep_out"], fam == 2)
    kb["any rule: push family"] = rate(any_rule, fam == 6)
    kb["any rule: one_sided"] = rate(any_rule, fam == 2)
    kb["any rule: near_miss"] = rate(any_rule, fam == 3)
    kb["any rule: drop"] = rate(any_rule, fam == 5)
    kb["any rule: weak_closure"] = rate(any_rule, fam == 4)
    sc = {}
    for f in range(7):
        sel = fam == f
        sc[FAM[f]] = dict(success_hindsight=rate(r["success"] >= 0, sel), firm_lift=rate(r["lifted"], sel),
                          success_and_no_rule=rate((r["success"] >= 0) & ~any_rule, sel))
    res["success_hindsight_goal"] = sc
    extra = {}
    for f in sorted(set(int(x) for x in np.unique(fam)) - set(range(12))):
        sel = fam == f
        held = sel & (out_ == 2)
        extra[names[f]] = dict(episodes=int(sel.sum()), any_rule_all=rate(any_rule, sel),
                               any_rule_lifted_held=rate(any_rule, held), any_rule_other=rate(any_rule, sel & ~held),
                               per_rule_lifted_held={x: rate(R[x], held) for x in T.RULES if (R[x] & held).any()},
                               success_hindsight=rate(r["success"] >= 0, sel),
                               outcome={str(k): int(n) for k, n in zip(*np.unique(out_[sel], return_counts=True))})
    res["other_families"] = extra
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"{len(fam)} episodes, {res['seconds']} s -> {a.out}")
    print("rule              | false fires nominal+off-centre: episodes (steps) | fired per family (episode %)")
    for rule in T.RULES:
        row = res["per_rule"][rule]
        fe, fs = row["false_fires_episodes_nominal_offcentre"], row["false_fires_steps_nominal_offcentre"]
        fams = " ".join(f"{FAM[f][:6]} {row[FAM[f]]['pct']:.1f}" for f in range(12))
        print(f"{rule:17s} | {fe['k']}/{fe['n']} = {fe['pct']:.2f} % ({fs['pct']:.3f} %) | {fams}")
    print("any rule false fires:", res["any_rule_false_fires_episodes_nominal_offcentre"])
    for k_, v in kb.items():
        print(f"known bad  {k_}: {v['k']}/{v['n']} = {v['pct']:.1f} %")
    for f, v in sc.items():
        print(f"success (hindsight goal) {f}: {v}")
    for f, v in extra.items():
        print(f"{f}: {v}")


# ---- instantaneous rules at the collector's terminal records -------------------------------------------------------
def extrapolate(arm, cube, dt):
    """One record ahead from the recorded rates: q + qd dt, p + v dt, q_cube <- exp(w dt / 2) q_cube (world spin)."""
    arm2, cube2 = arm.copy(), cube.copy()
    arm2[..., :6] += arm[..., 6:] * dt
    cube2[..., :3] += cube[..., 7:10] * dt
    w = cube[..., 10:13] * dt
    th = np.linalg.norm(w, axis=-1, keepdims=True)
    ax = w / np.maximum(th, 1e-30)
    dq = np.concatenate((np.cos(th / 2), ax * np.sin(th / 2)), -1)
    q = cube[..., 3:7]
    a0, av, b0, bv = dq[..., :1], dq[..., 1:], q[..., :1], q[..., 1:]
    cube2[..., 3:7] = np.concatenate((a0 * b0 - (av * bv).sum(-1, keepdims=True),
                                      a0 * bv + b0 * av + np.cross(av, bv)), -1)
    return arm2, cube2


def instant_measures(arm, cube, device):
    """lowest (all arm collision geoms), lowest of gripper + moving jaw (kin.lowest_point = the NRD sim's guard), link
    depth, jaw overlap, base zone at single records arm [n, 12], cube [n, 13] -> dict of numpy [n]."""
    kin = T.SO101Kin(device)
    geom = T.TaskGeometry.from_mujoco(device)
    q = torch.as_tensor(arm[:, :6], dtype=torch.float64, device=device)
    c = torch.as_tensor(cube[:, :3], dtype=torch.float64, device=device)
    qu = torch.as_tensor(cube[:, 3:7], dtype=torch.float64, device=device)
    rc = T.quat_mat(qu / qu.norm(dim=-1, keepdim=True))
    bpos, brot = kin.body_poses(q)
    kg = T.SO101Kin(device, points={b: (geom.arrays["low_pts"][geom.arrays["low_body"] == T.BODIES.index(b)],
                                        geom.arrays["low_rad"][geom.arrays["low_body"] == T.BODIES.index(b)])
                                    for b in ("gripper", "moving_jaw_so101_v1")})
    out = dict(lowest=geom.lowest(bpos, brot), lowest_guard=kg.lowest_point(q), link=geom.link_depth(bpos, brot, c, rc),
               jaw=geom.jaw_overlap(bpos, brot, c, rc), base=geom.in_base_zone(c, T.cube_bottom(c, rc)))
    return {k: v.cpu().numpy() for k, v in out.items()}


def cmd_terminal(a):
    cfg = T.TaskConfig()
    res = {}
    for data, split in zip(a.data, a.split + [None] * (len(a.data) - len(a.split))):
        from nedm.contact_nrd.rollouts import npz_members
        f = npz_members(Path(data) / "unified_data.npz")
        term, sp = np.asarray(f["terminal_reason"]), np.asarray(f["splits"])
        rows = np.flatnonzero(np.isin(term, (1, 2, 4, 6)) & ((sp == split) if split is not None else True))
        d = load(data, rows, keys=("arm", "cube", "lengths", "family", "terminal_reason"))
        L = d["lengths"]
        i = np.arange(len(rows))
        arm, cube = d["arm"][i, L - 1].astype(np.float64), d["cube"][i, L - 1].astype(np.float64)
        arm2, cube2 = extrapolate(arm, cube, 0.01)
        m1, m2 = instant_measures(arm, cube, a.device), instant_measures(arm2, cube2, a.device)
        key = f"{data} split {split}"
        out = {}
        for code, name, fn in ((1, "arm_table", lambda m: m["lowest"] < cfg.lowest_min),
                               (1, "arm_table (guard: gripper + jaw < 0.5 mm, nrd_sim arm_table_fn)",
                                lambda m: m["lowest_guard"] < 0.0005),
                               (2, "cube_other_link", lambda m: m["link"] > cfg.link_depth),
                               (6, "cube_visual_overlap", lambda m: m["jaw"] > cfg.jaw_depth),
                               (4, "cube_base", lambda m: m["base"])):
            sel = d["terminal_reason"] == code
            fam = {FAM[k]: int(n) for k, n in zip(*np.unique(d["family"][sel], return_counts=True))}
            out[name] = dict(episodes=int(sel.sum()), families=fam, last_record=rate(fn(m1), sel),
                             extrapolated_10ms=rate(fn(m2), sel), either=rate(fn(m1) | fn(m2), sel))
        out["measures_extrapolated"] = {
            "lowest_mm_arm_table": pct(m2["lowest"][d["terminal_reason"] == 1] * 1e3),
            "link_depth_mm_cube_other_link": pct(m2["link"][d["terminal_reason"] == 2] * 1e3),
            "jaw_overlap_mm_visual_overlap": pct(m2["jaw"][d["terminal_reason"] == 6] * 1e3),
            "jaw_overlap_mm_visual_overlap_last_record": pct(m1["jaw"][d["terminal_reason"] == 6] * 1e3)}
        res[key] = out
        print(f"== {key}: {len(rows)} terminal episodes")
        for k, v in out.items():
            print(f"   {k}: {v}")
    Path(a.out).write_text(json.dumps(res, indent=1))


# ---- rules on open-loop NRD rollouts ---------------------------------------------------------------------------------
def nrd_rollout_rules(d, member, device, cfg=None, phantom_steps=5):
    """Open-loop NRD rollouts (recorded 20 ms commands, start = the first record) of every loaded episode with the
    task's rules (log-only, hindsight goal = the MuJoCo final cube pose, macros from the collector phases) -> per-episode
    first fire per rule on NRD states, success, and phantom-lift windows (NRD lowest cube corner > 5 mm while MuJoCo's
    < 2 mm for ``phantom_steps`` steps)."""
    from nedm.so101_policy.nrd_sim import GraspNrdSim
    base = cfg or T.TaskConfig()
    cfg = T.TaskConfig(**{**base.__dict__, "log_only": T.RULES, "horizon": 10 ** 6})
    N = len(d["lengths"])
    sim = GraspNrdSim(num_envs=N, device=device, members=member)
    grid = T.grid_from_records(d["arm"], d["cube"])
    J = grid["q"].shape[1]
    L = torch.as_tensor(d["lengths"], device=device)
    idx = torch.arange(N, device=device)
    st0 = T.grid_state({k: v.to(device) for k, v in grid.items()}, 0)
    sim.reset(idx, dict(q=st0.q, qd=st0.qd, cube_pos=st0.cube_pos, cube_quat=st0.cube_quat, cube_v=st0.cube_v,
                        cube_w=st0.cube_w))
    last = ((L - 1) // 2).clamp_min(0)
    fq = grid["cube_quat"].to(device)[idx, last]
    gyaw, _ = T.cube_yaw_tilt(fq / fq.norm(dim=-1, keepdim=True))
    gxy = grid["cube_pos"].to(device)[idx, last, :2]
    kin, geom = T.SO101Kin(device), T.TaskGeometry.from_mujoco(device)
    task = T.PickPlaceTask(N, device, kin=kin, geom=geom, cfg=cfg)
    task.reset(idx, sim.observe(), gxy, gyaw)
    macro = torch.as_tensor(macro_grid(d["phase"], d["phases"]), device=device)
    act = torch.as_tensor(d["action"][:, ::2], dtype=torch.float64, device=device)
    R = len(T.RULES)
    first = torch.full((N, R), -1, dtype=torch.long, device=device)
    succ = torch.full((N,), -1, dtype=torch.long, device=device)
    run = torch.zeros(N, dtype=torch.long, device=device)
    ph_first = torch.full((N,), -1, dtype=torch.long, device=device)
    mj_rc = T.quat_mat(grid["cube_quat"].to(device) / grid["cube_quat"].to(device).norm(dim=-1, keepdim=True))
    mj_bottom = T.cube_bottom(grid["cube_pos"].to(device).flatten(0, 1), mj_rc.flatten(0, 2).reshape(-1, 3, 3)).reshape(N, J)
    for j in range(1, J):
        valid = 2 * j < L
        st = sim.step(act[:, j - 1])
        _, _, _, info = task.step(st, macro[:, j])
        fl = info["flags"] & valid[:, None]
        first = torch.where((first < 0) & fl, j, first)
        succ = torch.where((succ < 0) & info["success"] & valid, j, succ)
        ph = valid & (task.F["bottom"] > 0.005) & (mj_bottom[:, j] < 0.002)
        run = torch.where(ph, run + 1, 0)
        ph_first = torch.where((ph_first < 0) & (run >= phantom_steps), j - phantom_steps + 1, ph_first)
    c = lambda x: x.cpu().numpy()                                                   # noqa: E731
    return dict(first=c(first), success=c(succ), phantom=c(ph_first))


def cmd_nrd(a):
    t0 = time.time()
    from nedm.contact_nrd.rollouts import npz_members
    f = npz_members(Path(a.data) / "unified_data.npz")
    rows = np.flatnonzero(np.isin(np.asarray(f["family"]), a.families))
    d = load(a.data, rows)
    fam = d["family"]
    res = dict(data=str(a.data), families=[FAM[k] for k in a.families], members={})
    sup = [T.RULES.index("release_high"), T.RULES.index("unsupported")]
    for member in a.members:
        r = nrd_rollout_rules(d, member, a.device)
        ep = r["first"] >= 0
        out = dict(per_rule={rule: {FAM[k]: rate(ep[:, i], fam == k) for k in a.families}
                             for i, rule in enumerate(T.RULES)})
        anyr = ep[:, [T.RULES.index(x) for x in T.RULES if x not in ("world_invalid", "world_arm_table")]].any(1)
        out["any_rule"] = {FAM[k]: rate(anyr, fam == k) for k in a.families}
        out["success_hindsight"] = {FAM[k]: rate(r["success"] >= 0, fam == k) for k in a.families}
        ph = r["phantom"] >= 0
        out["phantom_episodes"] = {FAM[k]: rate(ph, fam == k) for k in a.families}
        # support rules firing from 0.2 s before the phantom window starts (the rule may see the lift first)
        sup_first = np.where(ep[:, sup], r["first"][:, sup], 10 ** 9).min(1)
        caught = ph & (sup_first <= r["phantom"] + 30)
        out["phantom_caught_by_support_rules"] = {FAM[k]: rate(caught, ph & (fam == k)) for k in a.families}
        gb = r["first"][:, T.RULES.index("grasp_box")]
        out["phantom_caught_by_grasp_box_or_support"] = {
            FAM[k]: rate(caught | (ph & (gb >= 0) & (gb <= r["phantom"])), ph & (fam == k)) for k in a.families}
        res["members"][member] = out
        print(f"== {member}")
        for k, v in out.items():
            if k == "per_rule":
                for rule, row in v.items():
                    if any(x["k"] for x in row.values()):
                        print(f"   {rule}: " + ", ".join(f"{n} {x['k']}/{x['n']}" for n, x in row.items()))
            else:
                print(f"   {k}: " + ", ".join(f"{n} {x['k']}/{x['n']}" for n, x in v.items()))
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1))


# ---- servo response of the kinematic world ---------------------------------------------------------------------------
def servo_grid(d):
    """Loaded episodes -> (q [N, J, 6], u [N, J, 6] target held from grid point k to k + 1, valid [N, J] (grid point
    recorded), jaw_free [N, J] (no jaw-cube contact in the step k -> k + 1))."""
    q = torch.as_tensor(d["arm"][:, ::2, :6], dtype=torch.float64)
    u = torch.as_tensor(d["action"][:, ::2], dtype=torch.float64)
    J = q.shape[1]
    valid = torch.as_tensor((2 * np.arange(J))[None, :] < d["lengths"][:, None])
    c = d["contacts"].astype(bool)
    jc = c[..., 0] | c[..., 1]
    free = np.ones((len(q), J), bool)
    free[:, :-1] = ~(jc[:, 0::2][:, :J - 1] | np.pad(jc[:, 1::2], ((0, 0), (0, 1)))[:, :J - 1])
    return q, u, valid, torch.as_tensor(free)


def servo_eval(model, q, u, valid, free, kin, horizon=50, every=25):
    """One-step errors (rad) per joint and open-loop errors after ``horizon`` steps (joints; TCP position, m) of a
    ServoModel on recorded grids; the jaw only where it is free over the whole window."""
    d = q[:, 1:] - q[:, :-1]
    k = slice(2, q.shape[1] - 1)
    pred = model.step(q[:, k].reshape(-1, 6), d[:, 1:-1].reshape(-1, 6), d[:, :-2].reshape(-1, 6),
                      u[:, k].reshape(-1, 6), u[:, 1:-2].reshape(-1, 6))
    err1 = (pred - q[:, 3:].reshape(-1, 6)).abs()
    ok1 = (valid[:, 3:] & valid[:, 2:-1]).reshape(-1)
    out = {"one_step_mrad": {}, f"open_loop_{horizon * 20}ms_mrad": {}}
    for j in range(6):
        m = ok1 & (free[:, 2:-1].reshape(-1) if j == 5 else True)
        out["one_step_mrad"][str(j)] = pct(err1[m, j].numpy() * 1e3, (50, 95, 99))
    starts = list(range(2, q.shape[1] - horizon - 1, every))
    errs, tcp_err, jaw_ok, okw = [], [], [], []
    for s0 in starts:
        qq, d1, d2 = q[:, s0].clone(), q[:, s0] - q[:, s0 - 1], q[:, s0 - 1] - q[:, s0 - 2]
        up = u[:, s0 - 1]
        for t in range(horizon):
            qn = model.step(qq, d1, d2, u[:, s0 + t], up)
            d2, d1, qq, up = d1, qn - qq, qn, u[:, s0 + t]
        errs.append((qq - q[:, s0 + horizon]).abs())
        p1, _ = kin.fk(qq)
        p0, _ = kin.fk(q[:, s0 + horizon])
        tcp_err.append((p1 - p0).norm(dim=-1))
        okw.append(valid[:, s0 + horizon])
        jaw_ok.append(free[:, s0: s0 + horizon].all(1))
    E, P, W, JW = torch.stack(errs, 1), torch.stack(tcp_err, 1), torch.stack(okw, 1), torch.stack(jaw_ok, 1)
    for j in range(6):
        m = W & (JW if j == 5 else True)
        out[f"open_loop_{horizon * 20}ms_mrad"][str(j)] = pct(E[..., j][m].numpy() * 1e3, (50, 95, 99))
    out[f"open_loop_{horizon * 20}ms_tcp_mm"] = pct(P[W].numpy() * 1e3, (50, 95, 99))
    out["windows"] = int(W.sum())
    return out


def kinworld_windows(d, coef, device, horizon=50, every=25):
    """The full KinWorld from recorded states over ``horizon`` steps with the recorded commands: cube position error
    (mm) after the horizon against the record, all task-family windows and windows whose start is held."""
    from nedm.so101_policy.kin_world import KinWorld
    grid = T.grid_from_records(d["arm"], d["cube"])
    act = torch.as_tensor(d["action"][:, ::2], dtype=torch.float64, device=device)
    N, J = grid["q"].shape[:2]
    L = torch.as_tensor(d["lengths"], device=device)
    world = KinWorld(N, device, servo=coef)
    errs, held0, oks = [], [], []
    for s0 in range(2, J - horizon - 1, every):
        st = T.grid_state({k: v.to(device) for k, v in grid.items()}, s0)
        world.reset(None, st)
        world.d2 = (grid["q"][:, s0 - 1] - grid["q"][:, s0 - 2]).to(device)
        world.u_prev = act[:, s0 - 1]
        held0.append(world.held.clone())
        for t in range(horizon):
            ws = world.step(act[:, s0 + t])
        errs.append((ws.cube_pos - grid["cube_pos"][:, s0 + horizon].to(device)).norm(dim=-1))
        oks.append(2 * (s0 + horizon) < L)
    E, H, W = torch.stack(errs, 1).cpu(), torch.stack(held0, 1).cpu(), torch.stack(oks, 1).cpu()
    fam = torch.as_tensor(d["family"].astype(np.int64))[:, None].expand_as(W)
    out = dict(all=pct(E[W].numpy() * 1e3, (50, 95, 99)), held_start=pct(E[W & H].numpy() * 1e3, (50, 95, 99)),
               windows=int(W.sum()), held_windows=int((W & H).sum()))
    for k in range(7):
        out[FAM[k]] = pct(E[W & (fam == k)].numpy() * 1e3, (50, 95, 99))
    return out


def kinworld_episodes(d, coef, device):
    """Whole-episode open-loop replays in the KinWorld (recorded commands from the first record): per family the lift
    agreement with MuJoCo (cube lowest corner >= 2 cm at some step) and the final cube position error (mm)."""
    from nedm.so101_policy.kin_world import KinWorld
    grid = T.grid_from_records(d["arm"], d["cube"])
    act = torch.as_tensor(d["action"][:, ::2], dtype=torch.float64, device=device)
    N, J = grid["q"].shape[:2]
    L = torch.as_tensor(d["lengths"], device=device)
    last = ((L - 1) // 2).clamp_min(0)
    world = KinWorld(N, device, servo=coef)
    world.reset(None, T.grid_state({k: v.to(device) for k, v in grid.items()}, 0))
    up_k = torch.zeros(N, dtype=torch.bool, device=device)
    fin = torch.zeros(N, 3, dtype=torch.float64, device=device)
    for j in range(1, J):
        ws = world.step(act[:, j - 1])
        valid = j <= last
        up_k |= valid & (T.cube_bottom(ws.cube_pos, T.quat_mat(ws.cube_quat)) >= 0.02)
        fin = torch.where((j == last)[:, None], ws.cube_pos, fin)
    cq = grid["cube_quat"].to(device)
    mb = T.cube_bottom(grid["cube_pos"].to(device).flatten(0, 1),
                       T.quat_mat(cq / cq.norm(dim=-1, keepdim=True)).flatten(0, 0).reshape(-1, 3, 3)).reshape(N, J)
    jj = torch.arange(J, device=device)[None]
    up_m = ((mb >= 0.02) & (jj <= last[:, None])).any(1)
    err = (fin - grid["cube_pos"].to(device)[torch.arange(N, device=device), last]).norm(dim=-1).cpu().numpy()
    up_k, up_m = up_k.cpu().numpy(), up_m.cpu().numpy()
    out = {}
    for k in range(7):
        e = d["family"] == k
        out[FAM[k]] = dict(lift_agree=rate(up_k == up_m, e), lift_mujoco=int((up_m & e).sum()),
                           lift_kin=int((up_k & e).sum()), final_cube_err_mm=pct(err[e] * 1e3, (50, 95, 99)))
    return out


def cmd_servo(a):
    from nedm.contact_nrd.rollouts import npz_members
    from nedm.so101_policy.kin_world import ServoModel
    t0 = time.time()
    f = npz_members(Path(a.train) / "unified_data.npz")
    fam, sp, term = np.asarray(f["family"]), np.asarray(f["splits"]), np.asarray(f["terminal_reason"])
    cand = np.flatnonzero((sp == a.split) & np.isin(fam, list(range(7))) & (term == 0))
    rows = np.random.default_rng(a.seed).choice(cand, min(a.n, len(cand)), replace=False)
    dt = load(a.train, rows, keys=("arm", "action", "contacts", "lengths"))
    q, u, valid, free = servo_grid(dt)
    coef_ls = ServoModel.fit(q, u, valid, free)
    print(f"least-squares fit on {len(rows)} training episodes ({time.time() - t0:.0f} s)")
    if a.no_fit:
        from nedm.so101_policy.kin_world import DEFAULT_COEF
        coef = torch.as_tensor(DEFAULT_COEF, dtype=torch.float64)
    else:
        from nedm.so101_policy.kin_world import fit_rollout
        coef = fit_rollout(coef_ls, q, u, valid, free, horizon=a.horizon, device=a.device)
        print(f"rollout fit ({a.horizon} steps) done ({time.time() - t0:.0f} s)")
    fv = npz_members(Path(a.val) / "unified_data.npz")
    vrows = np.flatnonzero(np.isin(np.asarray(fv["family"]), list(range(7))))
    dv = load(a.val, vrows, keys=("arm", "cube", "action", "contacts", "lengths", "family"))
    qv, uv, vv, frv = servo_grid(dv)
    kin = T.SO101Kin()
    res = dict(train=str(a.train), split=a.split, train_episodes=int(len(rows)), seed=a.seed, coef=coef.tolist(),
               features=["u - q", "last step change", "u - last u", "step change before", "1", "sin p1", "cos p1",
                         "sin p2", "cos p2", "sin p3", "cos p3"],
               val=str(a.val), val_episodes=int(len(vrows)))
    model = ServoModel(coef)
    res["servo"] = servo_eval(model, qv, uv, vv, frv, kin)
    res["servo_least_squares_only"] = servo_eval(ServoModel(coef_ls), qv, uv, vv, frv, kin)
    res["coef_least_squares"] = coef_ls.tolist()
    # reference: the joint at its last target (no dynamics) and a pure 1-step lag
    ref = ServoModel(torch.zeros(6, coef.shape[1], dtype=torch.float64).index_put_((torch.arange(6), torch.zeros(6, dtype=torch.long)), torch.ones(6, dtype=torch.float64)))
    res["reference_q_equals_target"] = servo_eval(ref, qv, uv, vv, frv, kin)
    res["kin_world_cube_1s_mm"] = kinworld_windows(dv, coef, a.device)
    res["kin_world_episodes"] = kinworld_episodes(dv, coef, a.device)
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for name in ("servo", "servo_least_squares_only", "reference_q_equals_target"):
        r = res[name]
        print(name, "| one step p95 mrad:", [r["one_step_mrad"][str(j)]["p95"] for j in range(6)],
              "| 1 s p95 mrad:", [r["open_loop_1000ms_mrad"][str(j)]["p95"] for j in range(6)],
              "| 1 s TCP mm:", r["open_loop_1000ms_tcp_mm"])
    print("kin world cube 1 s:", json.dumps(res["kin_world_cube_1s_mm"]))
    for k, v in res["kin_world_episodes"].items():
        print("kin world whole episode", k, v)
    print("coef =", np.array2string(coef.numpy(), precision=6, separator=", ", max_line_width=200))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("trace")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    s.set_defaults(fn=cmd_trace)
    s = sub.add_parser("stats")
    s.add_argument("--trace", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_stats)
    s = sub.add_parser("rules")
    s.add_argument("--data", required=True)
    s.add_argument("--split", type=int, default=None)
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    s.set_defaults(fn=cmd_rules)
    s = sub.add_parser("terminal")
    s.add_argument("--data", action="append", required=True)
    s.add_argument("--split", action="append", type=int, default=[])
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    s.set_defaults(fn=cmd_terminal)
    s = sub.add_parser("nrd")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--families", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    s.add_argument("--members", nargs="+", default=["s61", "s62"])
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    s.set_defaults(fn=cmd_nrd)
    s = sub.add_parser("servo")
    s.add_argument("--train", required=True)
    s.add_argument("--split", type=int, default=0)
    s.add_argument("--n", type=int, default=4000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--horizon", type=int, default=50)
    s.add_argument("--no-fit", action="store_true", help="evaluate kin_world.DEFAULT_COEF instead of refitting")
    s.add_argument("--val", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    s.set_defaults(fn=cmd_servo)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
