"""Validation and throughput of nedm.so101_policy.nrd_sim.GraspNrdSim (study spec 2.2). Validation data only (split 1,
``~/so101_nrd_fix/data/val_only``); never the sealed test.

    PYTHONPATH=src python scripts/so101_policy/bench_nrd_sim.py MODE --out OUT.json [options]

| Mode | What it measures |
|---|---|
| ``windows`` | One step from recorded 4-state histories (starts 0.1 s + 0.5 k s, as the evaluation's windows) and the recorded command: the sim's dense step against ``ContactNRD.details``. Max abs difference of the next state, switches equal |
| ``parity`` | Whole episodes from the first recorded state with the recorded commands, canonicalisation off: the sim against ``nedm.contact_nrd.rollouts.rollouts`` (``ContactNRD.rollout``, sparse contact step). Max abs difference over the valid steps, per horizon, episodes above 1e-6, switch mismatches. The same for the reference run on the episodes in two halves (its own sensitivity to the batch), one sim step from the reference's own history at every step (teacher forced), and the start from a WorldState (``reset``: quaternion normalised, model.finish) |
| ``canon`` | Whole episodes of rollouts.rollouts (reference) and of the sim with canonicalisation off, on, and on/off from starts whose cube labelling is turned by a random cube rotation: cube position (mm) and angle (deg, the TRUE quaternion of WorldState against the record) errors at 1 s and 4 s, p50 / p95, paired bootstrap 95 % CI of the p95 difference, relabel counts |
| ``disturb`` | ``set_cube`` on approach-phase windows (nominal + off-centre): a no-op set_cube against the recorded window, and a cube moved far from the arm (truth: it rests) - NRD drift at 0.5 / 1 s, jaw switches (``mode_disturb``) |
| ``bench`` | steps/s of ``GraspNrdSim.step`` (incl. the WorldState copy) at the given world counts, members split over the checkpoints, canonicalisation on (eager, CUDA graph) and off (graph), and on + the arm-table guard (``lowest_point_guard``, graph); plus ``ContactNRD.details`` + ``advance`` of one member (sparse reference). The GPU may be shared: ``free_gb_before`` per row |

Episodes: the task families of the data (``select: true``: nominal, off_centre, one_sided, near_miss, weak_closure,
drop, push), ``--per-family`` evenly spread validation episodes each (rollouts.pick_rows), or all with 0.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from nedm.contact_nrd import rollouts as R  # noqa: E402
from nedm.contact_nrd.data import rotation_angle  # noqa: E402
from nedm.contact_nrd.model import load_model  # noqa: E402
from nedm.so101_policy import cube_symmetry as cs  # noqa: E402
from nedm.so101_policy.nrd_sim import STUDY_CKPTS, GraspNrdSim  # noqa: E402
from nedm.so101_policy.world import WorldState  # noqa: E402

DATA = Path.home() / "so101_nrd_fix" / "data" / "val_only"


def task_rows(data_dir, per_family):
    system = json.loads((Path(data_dir) / "system.json").read_text())
    fams = [f["name"] for f in system["families"] if f.get("select")]
    if per_family:
        return R.pick_rows(data_dir, per_family, fams, 0, 1, 2)
    arrays = R.npz_members(Path(data_dir) / "unified_data.npz")
    fam, split = np.asarray(arrays["family"]), np.asarray(arrays["splits"])
    sel = [k for k, f in enumerate(system["families"]) if f.get("select")]
    return np.flatnonzero(np.isin(fam, sel) & (split == 1)).tolist()


def ckpts(args):
    if not args.ckpt:
        return {k: str(v) for k, v in STUDY_CKPTS.items()}
    return dict(c.split("=", 1) for c in args.ckpt)


def env_info(dev):
    out = dict(host=platform.node(), torch=torch.__version__, device=str(dev), threads=torch.get_num_threads())
    if dev.type == "cuda":
        out["gpu"] = torch.cuda.get_device_name(dev)
    return out


def reference(model, data_dir, rows, dev):
    """rollouts.rollouts from the first state (whole episodes) -> dict of tensors on dev."""
    data = R.load_rows(data_dir, rows, dev, model.dt)
    out = R.rollouts(model, data, "first", 0, None, 2)
    keys = ("recorded", "predicted", "actions", "switches", "steps")
    return {k: torch.as_tensor(out[k], device=dev) for k in keys} | dict(family=data["family"], rows=rows)


def channel_groups(model):
    """{"body/channel_type": (body index, [channels])} of the model's moving bodies."""
    mov = [b for b in model.system["bodies"] if b["moving"]]
    return {f"{b['name']}/{t}": (n, [c for c, tt in enumerate(b["channel_types"]) if tt == t])
            for n, b in enumerate(mov) for t in dict.fromkeys(b["channel_types"])}


def run_sim(sim, start, actions, collect=True):
    """Step sim through actions [E, H, D, A] (arm row 0) after start(sim) -> states [E, H+1, D, S], true quats
    [E, H+1, 4], gates [E, H, P], relabel counts [E]."""
    start(sim)
    states, quats, gates = [sim.state.clone()], [sim.true_q.clone()], []
    nrel = torch.zeros(sim.B, dtype=torch.long, device=sim.device)
    for t in range(actions.shape[1]):
        sim.step(actions[:, t, sim.ia, :6])
        nrel += sim.relabelled.long()
        if collect:
            states.append(sim.state.clone())
        quats.append(sim.true_q.clone())
        gates.append(sim.gates.clone())
    return (torch.stack(states, 1) if collect else None), torch.stack(quats, 1), torch.stack(gates, 1), nrel


# ---- modes ------------------------------------------------------------------------------------------------------
def mode_windows(args, dev):
    rows = task_rows(args.data, args.per_family)
    out = dict(env=env_info(dev), episodes=len(rows), members={})
    for name, path in ckpts(args).items():
        model, _ = load_model(path, dev)
        data = R.load_rows(args.data, rows, dev, model.dt)
        R.step_average_rates(data, set(model.config["integrate"]["kinds"]))
        R.add_derived(data, model.config.get("derived"))
        R.add_relative(data, model.config.get("relative"))
        st, act, s = data["states"], data["actions"], data["stride"]
        n_grid = ((data["lengths"] - 1) // s).cpu()
        hs, oks, acts = [], [], []
        for start in range(5, int(n_grid.max()), 25):                       # 0.1 s + 0.5 k s on the 20 ms grid
            use = torch.nonzero(n_grid > start).flatten().to(dev)
            if not len(use):
                continue
            lag = torch.arange(1 - model.k, 1, device=dev) + start
            ok = (lag >= 0)[None].expand(len(use), -1)
            hs.append(st[use[:, None], (lag.clamp_min(0) * s)[None]] * ok[..., None, None])
            oks.append(ok)
            acts.append(act[use, start * s])
        h, ok, a = torch.cat(hs), torch.cat(oks), torch.cat(acts)
        n = len(h)
        groups = channel_groups(model)
        worst = {g: 0.0 for g in groups}
        worst_rel, above, gate_diff, gate_near = 0.0, 0, 0, 0
        for b in range(0, n, args.batch):
            sl = slice(b, min(n, b + args.batch))
            with torch.no_grad():
                ref, info = model.details(h[sl], ok[sl], a[sl])
            sim = GraspNrdSim({name: model}, num_envs=sl.stop - sl.start, device=dev, canonical=False)
            sim.reset_history(None, h[sl], ok[sl], finish=False)
            sim.step(a[sl][:, sim.ia, :6])
            d = (sim.state - ref).abs()
            for g, (body, cols) in groups.items():
                worst[g] = max(worst[g], float(d[:, body][:, cols].max()))
            worst_rel = max(worst_rel, float((d / (1.0 + ref.abs())).max()))
            above += int((d.amax(dim=(1, 2)) > 1e-6).sum())
            gate_diff += int((sim.gates != info["gates"]).sum())
            gate_near += int((info["logits"].abs() < 1e-4).sum())
        r = dict(windows=n, max_abs_diff=max(worst.values()), max_abs_diff_by_channel_type=worst,
                 max_diff_over_1_plus_abs=worst_rel, windows_above_1e6=above, switch_mismatches=gate_diff,
                 logits_within_1e4_of_0=gate_near)
        out["members"][name] = r
        print(name, json.dumps(r), flush=True)
    return out


HORIZONS = (1, 2, 5, 10, 25, 50, 100, 200, 300)


def divergence(st, pred, valid, H):
    """Max abs state difference [E, H+1] (valid steps only) -> summary."""
    d = torch.where(valid, (st - pred).abs().amax(dim=(2, 3)), torch.zeros(valid.shape, dtype=st.dtype, device=st.device))
    d = torch.nan_to_num(d, nan=float("inf"))
    first = torch.where((d > 1e-6).any(1), (d > 1e-6).float().argmax(1), torch.full(d.shape[:1], -1, device=d.device))
    return dict(max_abs_diff=float(d.max()), max_abs_diff_by_step={str(k): float(d[:, k].max()) for k in HORIZONS if k <= H},
                episodes_above_1e6=int((d.amax(1) > 1e-6).sum()), episodes_above_1e3=int((d.amax(1) > 1e-3).sum()),
                first_step_above_1e6_p50=float(first[first >= 0].float().median()) if (first >= 0).any() else None)


def mode_parity(args, dev):
    rows = task_rows(args.data, args.per_family)
    out = dict(env=env_info(dev), episodes=len(rows), members={})
    for name, path in ckpts(args).items():
        model, _ = load_model(path, dev)
        t0 = time.perf_counter()
        ref = reference(model, args.data, rows, dev)
        t_ref = time.perf_counter() - t0
        rec, pred, actions, steps = ref["recorded"], ref["predicted"], ref["actions"], ref["steps"]
        E, H = rec.shape[0], actions.shape[1]
        valid = torch.arange(H + 1, device=dev)[None] <= steps[:, None]                  # [E, H+1]
        res = dict(steps_total=int(steps.sum()), reference_s=t_ref)
        # the reference against itself on another batch composition: the episodes in two halves (its sparse contact
        # step then runs the contact network on another number of rows, which changes the float32 GEMM rounding)
        parts = set(args.parts)
        if "halves" in parts:
            half = len(rows) // 2
            split = torch.cat([reference(model, args.data, r, dev)["predicted"] for r in (rows[:half], rows[half:])])
            res["reference_two_halves"] = divergence(split, pred, valid, H)
            print(name, "reference vs reference (two halves)", json.dumps(res["reference_two_halves"]), flush=True)
        for graph in ([g for g in (False, True) if ("graph" if g else "eager") in parts] if dev.type == "cuda"
                      else [False] if "eager" in parts else []):
            sim = GraspNrdSim({name: model}, num_envs=E, device=dev, canonical=False, cuda_graph=graph)
            t0 = time.perf_counter()
            st, _, g, _ = run_sim(sim, lambda s: s.reset_state(None, rec[:, 0], finish=False), actions)
            t_sim = time.perf_counter() - t0
            gm = (g.bool() != ref["switches"]) & valid[:, 1:, None]
            r = dict(sim_s=t_sim, **divergence(st, pred, valid, H), switch_mismatch_steps=int(gm.any(-1).sum()),
                     episodes_with_switch_mismatch=int(gm.any(-1).any(-1).sum()))
            res["graph" if graph else "eager"] = r
            print(name, "graph" if graph else "eager", json.dumps(r), flush=True)
        # teacher-forced along the reference: one sim step from the reference's own 4-state history at every step
        if "teacher" not in parts:
            out["members"][name] = res
            continue
        sim = GraspNrdSim({name: model}, num_envs=E, device=dev, canonical=False)
        groups = channel_groups(model)
        z = torch.zeros(E, dtype=torch.float64, device=dev)
        worst, worst_rel, gmis = z.clone(), z.clone(), 0
        by_type = {g: 0.0 for g in groups}
        n_above = torch.zeros(E, dtype=torch.long, device=dev)
        self_worst, self_rel, half = 0.0, 0.0, E // 2
        for t in range(H):
            lag = torch.arange(t - model.k + 1, t + 1, device=dev)
            ok = (lag >= 0)[None].expand(E, -1)
            hist_t = pred[:, lag.clamp_min(0)] * ok[..., None, None]
            sim.reset_history(None, hist_t, ok, finish=False)
            sim.step(actions[:, t, sim.ia, :6])
            v = valid[:, t + 1]
            with torch.no_grad():   # scale: ContactNRD.details itself on the episodes in two halves (other GEMM shapes)
                own = torch.cat([model.details(hist_t[sl], ok[sl], actions[sl, t])[0] for sl in (slice(0, half), slice(half, E))])
            do = torch.nan_to_num((own - pred[:, t + 1]).abs(), nan=float("inf")) * v[:, None, None]
            self_worst = max(self_worst, float(do.max()))
            self_rel = max(self_rel, float((do / (1.0 + pred[:, t + 1].abs())).max()))
            dd = torch.nan_to_num((sim.state - pred[:, t + 1]).abs(), nan=float("inf")) * v[:, None, None]
            d = dd.amax(dim=(1, 2))
            worst = torch.maximum(worst, d)
            worst_rel = torch.maximum(worst_rel, (dd / (1.0 + pred[:, t + 1].abs())).amax(dim=(1, 2)))
            n_above += (d > 1e-6).long()
            for g, (body, cols) in groups.items():
                by_type[g] = max(by_type[g], float(dd[:, body][:, cols].max()))
            gmis += int(((sim.gates.bool() != ref["switches"][:, t]) & v[:, None]).sum())
        res["teacher_forced"] = dict(steps=int(valid[:, 1:].sum()), max_abs_diff=float(worst.max()),
                                     max_abs_diff_by_channel_type=by_type,
                                     max_diff_over_1_plus_abs=float(worst_rel.max()),
                                     p99_episode_max=float(torch.quantile(worst, 0.99)), switch_mismatches=gmis,
                                     episodes_above_1e6=int((worst > 1e-6).sum()),
                                     steps_above_1e6=int(n_above.sum()),
                                     details_two_halves_max_abs_diff=self_worst,
                                     details_two_halves_max_diff_over_1_plus_abs=self_rel)
        print(name, "teacher_forced", json.dumps(res["teacher_forced"]), flush=True)
        # start from a WorldState (reset: normalised quaternion, model.finish) instead of the recorded model state
        if "worldstate" not in parts:
            out["members"][name] = res
            continue
        s0 = rec[:, 0]
        ws = WorldState(q=s0[:, 0, 0:6], qd=s0[:, 0, 6:12], cube_pos=s0[:, 1, 0:3], cube_quat=s0[:, 1, 3:7],
                        cube_v=s0[:, 1, 7:10], cube_w=s0[:, 1, 10:13], contact=None, arm_table=None, invalid=None)
        sim = GraspNrdSim({name: model}, num_envs=E, device=dev, canonical=False, cuda_graph=dev.type == "cuda")
        st, _, _, _ = run_sim(sim, lambda s: s.reset(None, ws), actions)
        res["from_worldstate"] = dict(start_max_abs_diff=float((st[:, 0] - pred[:, 0]).abs().max()), **divergence(st, pred, valid, H))
        print(name, "from_worldstate", json.dumps(res["from_worldstate"]), flush=True)
        out["members"][name] = res
    return out


def cube_errors(pos, quat, rec, steps, at):
    """pos [E, H+1, 3], quat [E, H+1, 4] (true), recorded [E, H+1, D, S]: errors at step at for episodes long enough."""
    use = steps >= at
    e_pos = (pos[use, at] - rec[use, at, 1, 0:3]).norm(dim=-1) * 1e3
    e_ang = torch.rad2deg(rotation_angle(quat[use, at], rec[use, at, 1, 3:7]))
    return e_pos.cpu().numpy(), e_ang.cpu().numpy(), use.cpu().numpy()


def boot_p95_diff(a, b, n=2000, seed=0):
    """Paired bootstrap of p95(a) - p95(b) over episodes -> (estimate, lo, hi)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    d = np.percentile(a[idx], 95, axis=1) - np.percentile(b[idx], 95, axis=1)
    return float(np.percentile(a, 95) - np.percentile(b, 95)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def mode_canon(args, dev):
    rows = task_rows(args.data, args.per_family)
    out = dict(env=env_info(dev), episodes=len(rows), horizons_s=[1.0, 4.0], members={})
    gen = torch.Generator(device="cpu").manual_seed(20261011)
    for name, path in ckpts(args).items():
        model, _ = load_model(path, dev)
        ref = reference(model, args.data, rows, dev)
        rec, actions, steps = ref["recorded"], ref["actions"], ref["steps"]
        E = rec.shape[0]
        turn = torch.randint(1, cs.ORDER, (E,), generator=gen).to(dev)                     # never the identity
        G = cs.group(torch.float64, dev)
        s_turn = rec[:, 0].clone()
        s_turn[:, 1, 3:7] = cs.relabel(s_turn[:, 1, 3:7], turn, G)                         # same cube, other labels
        s_turn = model.finish(s_turn)
        rec_turn = rec.clone()
        rec_turn[:, :, 1, 3:7] = cs.qmul(rec[:, :, 1, 3:7], G[turn][:, None])
        runs = {}
        variants = dict(off=(False, rec[:, 0], rec), on=(True, rec[:, 0], rec),
                        on_turned=(True, s_turn, rec_turn), off_turned=(False, s_turn, rec_turn))
        res = dict(variants={})
        r = {}
        for hs, at in (("1s", 50), ("4s", 200)):                                       # rollouts.rollouts itself
            ep, ea, use = cube_errors(ref["predicted"][:, :, 1, 0:3], ref["predicted"][:, :, 1, 3:7], rec, steps, at)
            ep, ea = np.nan_to_num(ep, nan=1e9), np.nan_to_num(ea, nan=180.0)
            r[hs] = dict(n=int(use.sum()), pos_mm_p50=float(np.percentile(ep, 50)), pos_mm_p95=float(np.percentile(ep, 95)),
                         pos_mm_max=float(ep.max()), angle_deg_p50=float(np.percentile(ea, 50)), angle_deg_p95=float(np.percentile(ea, 95)))
            runs.setdefault("reference", dict(pos=ref["predicted"][:, :, 1, 0:3]))[hs] = (ep, ea)
        res["variants"]["reference"] = r
        print(name, "reference", json.dumps(r), flush=True)
        for v, (canon, s0, truth) in variants.items():
            sim = GraspNrdSim({name: model}, num_envs=E, device=dev, canonical=canon, cuda_graph=dev.type == "cuda")
            st, qt, g, nrel = run_sim(sim, lambda s: s.reset_state(None, s0, finish=False), actions)
            pos = st[:, :, 1, 0:3]
            runs[v] = dict(pos=pos, quat=qt)
            r = dict(episodes_relabelled=int((nrel > 0).sum()), relabels_total=int(nrel.sum()),
                     nonfinite_episodes=int((~torch.isfinite(st).all(-1).all(-1).all(-1)).sum()))
            for hs, at in (("1s", 50), ("4s", 200)):
                ep, ea, use = cube_errors(pos, qt, truth, steps, at)
                ep, ea = np.nan_to_num(ep, nan=1e9), np.nan_to_num(ea, nan=180.0)
                r[hs] = dict(n=int(use.sum()), pos_mm_p50=float(np.percentile(ep, 50)), pos_mm_p95=float(np.percentile(ep, 95)),
                             pos_mm_max=float(ep.max()), angle_deg_p50=float(np.percentile(ea, 50)),
                             angle_deg_p95=float(np.percentile(ea, 95)))
                runs[v][hs] = (ep, ea)
            res["variants"][v] = r
            print(name, v, json.dumps(r), flush=True)
        cmp = {}
        for a_, b_ in (("off", "reference"), ("on", "off"), ("on", "reference"), ("on_turned", "on"), ("off_turned", "off")):
            c = {}
            for hs in ("1s", "4s"):
                c[hs] = dict(pos_p95_diff_mm=boot_p95_diff(runs[a_][hs][0], runs[b_][hs][0]),
                             angle_p95_diff_deg=boot_p95_diff(runs[a_][hs][1], runs[b_][hs][1]))
            dp = (runs[a_]["pos"] - runs[b_]["pos"]).norm(dim=-1)
            dp = torch.where(torch.arange(dp.shape[1], device=dev)[None] <= steps[:, None], dp, torch.zeros_like(dp))
            c["max_pos_change_mm_valid_steps"] = float(np.nanmax(dp.cpu().numpy())) * 1e3
            cmp[f"{a_}_minus_{b_}"] = c
            print(name, a_, "-", b_, json.dumps(c), flush=True)
        res["paired"] = cmp
        out["members"][name] = res
    if len(out["members"]) == 2:   # the seed-to-seed difference as a scale of "noise"
        a_, b_ = list(out["members"])
        out["seed_gap_p95_off"] = {hs: out["members"][a_]["variants"]["off"][hs]["pos_mm_p95"]
                                   - out["members"][b_]["variants"]["off"][hs]["pos_mm_p95"] for hs in ("1s", "4s")}
    return out


def _quantiles(x, mask):
    x = x[mask].cpu().numpy()
    if not len(x):
        return None
    return dict(n=int(len(x)), p50=float(np.percentile(x, 50)), p95=float(np.percentile(x, 95)), max=float(x.max()))


def mode_disturb(args, dev):
    """set_cube on real approach-phase windows (nominal + off-centre validation episodes; cube resting, arm above it):
    (a) the recorded window as it is; (b) set_cube to the cube's own recorded pose (a no-op disturbance: must change
    nothing but the float32 data jitter); (c) the cube moved to a fresh annulus pose (flat, true yaw U(-180, 180) deg,
    data resting height) at least --clear m (xy) from the old cube and from the recorded grasp point over the whole window,
    with the arm continuing its recorded commands (it grasps at the old place). The truth for (c) is a cube at rest:
    every displacement is NRD error. Reference scale: the cube error of (a) against the record over the same window."""
    from nedm.so101_policy.kin import SO101Kin
    from nedm.so101_policy.nrd_sim import CUBE_REST_Z, ANNULUS, cube_rest_envelope
    system = json.loads((Path(args.data) / "system.json").read_text())
    arrays = R.npz_members(Path(args.data) / "unified_data.npz")
    fam, split = np.asarray(arrays["family"]), np.asarray(arrays["splits"])
    fams = [k for k, f in enumerate(system["families"]) if f["name"] in ("nominal", "off_centre")]
    rows = np.flatnonzero(np.isin(fam, fams) & (split == 1))
    if args.per_family:
        rows = rows[np.unique(np.linspace(0, len(rows) - 1, args.per_family).astype(np.int64))]
    phase_all = np.asarray(arrays["phase"][rows])
    approach = system["phases"].index("approach")
    kin = SO101Kin(dev)
    H = args.horizon
    out = dict(env=env_info(dev), episodes=int(len(rows)), horizon_steps=H, clear_m=args.clear, members={})
    for name, path in ckpts(args).items():
        model, _ = load_model(path, dev)
        data = R.load_rows(args.data, rows.tolist(), dev, model.dt)
        R.step_average_rates(data, set(model.config["integrate"]["kinds"]))
        R.add_derived(data, model.config.get("derived"))
        R.add_relative(data, model.config.get("relative"))
        st, act, s = data["states"], data["actions"], data["stride"]
        n_grid = ((data["lengths"] - 1) // s).cpu().numpy()
        t0 = np.full(len(rows), -1)
        for e in range(len(rows)):
            g = np.flatnonzero(phase_all[e, ::s][: n_grid[e] + 1] == approach)
            g = g[(g >= model.k - 1) & (g + H <= n_grid[e])]
            if len(g):
                t0[e] = g[len(g) // 2]
        use = np.flatnonzero(t0 >= 0)
        E = len(use)
        ue, tt = torch.as_tensor(use, device=dev), torch.as_tensor(t0[use], device=dev)
        lag = tt[:, None] + torch.arange(1 - model.k, 1, device=dev)[None]
        hist = st[ue[:, None], lag * s]                                         # [E, K, D, S] recorded window
        ok = torch.ones(E, model.k, dtype=torch.bool, device=dev)
        steps = tt[:, None] + torch.arange(H + 1, device=dev)[None]
        rec = st[ue[:, None], steps * s]                                        # [E, H+1, D, S]
        cmds = act[ue[:, None], steps[:, :H] * s, 0, :6]                       # [E, H, 6]
        rest_speed = hist[:, :, 1, 7:10].norm(dim=-1).amax(1)
        gp, _ = kin.fk(rec[:, :, 0, 0:6])                                       # recorded grasp point [E, H+1, 3]
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        new_pos = torch.zeros(E, 3, dtype=torch.float64)
        found = torch.zeros(E, dtype=torch.bool)
        old = rec[:, 0, 1, 0:2].cpu()
        gpx = gp[:, :, 0:2].cpu()
        for _ in range(50):                                                     # rejection sampling on the annulus
            r = ANNULUS["r_min"] + (ANNULUS["r_max"] - ANNULUS["r_min"]) * torch.rand(E, generator=gen, dtype=torch.float64)
            ang = torch.deg2rad((2 * torch.rand(E, generator=gen, dtype=torch.float64) - 1) * ANNULUS["angle_max_deg"])
            cand = torch.stack((ANNULUS["center_xy"][0] + r * torch.cos(ang), ANNULUS["center_xy"][1] + r * torch.sin(ang)), -1)
            far = ((cand - old).norm(dim=-1) >= args.clear) & ((gpx - cand[:, None]).norm(dim=-1).amin(1) >= args.clear)
            take = far & ~found
            new_pos[take, 0:2] = cand[take]
            found |= take
        new_pos[:, 2] = CUBE_REST_Z
        yaw = (2 * torch.rand(E, generator=gen, dtype=torch.float64) - 1) * math.pi        # true yaw, any of the 4 faces
        new_q = torch.stack((torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)), -1)
        new_pos, new_q, found = new_pos.to(dev), new_q.to(dev), found.to(dev)
        res = dict(windows=E, resting_history_max_speed=float(rest_speed.max()), moved_found=int(found.sum()),
                   moved_in_envelope=int(cube_rest_envelope(new_pos, new_q)[found].sum()))
        for canon in (False, True):
            key = "canon_on" if canon else "canon_off"
            sim = GraspNrdSim({name: model}, num_envs=E, device=dev, canonical=canon)
            runs = {}
            for v in ("recorded", "noop", "moved"):
                sim.reset_history(None, hist, ok, finish=False)
                h_rec = sim.hist.clone()
                if v == "noop":
                    sim.set_cube(None, rec[:, 0, 1, 0:3], rec[:, 0, 1, 3:7])
                    runs["noop_history_change"] = float((sim.hist - h_rec).abs().max())
                elif v == "moved":
                    sim.set_cube(None, new_pos, new_q)
                pos, quat, jaw, table = [sim.observe().cube_pos], [sim.true_q.clone()], [], []
                for t in range(H):
                    ws = sim.step(cmds[:, t])
                    pos.append(ws.cube_pos)
                    quat.append(ws.cube_quat)
                    jaw.append(ws.contact[:, :2].any(-1))
                    table.append(ws.contact[:, 2])
                runs[v] = (torch.stack(pos, 1), torch.stack(quat, 1), torch.stack(jaw, 1), torch.stack(table, 1))
            r = dict(noop_history_change=runs["noop_history_change"])
            pa, qa = runs["recorded"][:2]
            pb, qb = runs["noop"][:2]
            r["noop_vs_recorded_pos_mm_max"] = float((pb - pa).norm(dim=-1).max()) * 1e3
            r["noop_vs_recorded_pos_mm_max_first_step"] = float((pb[:, 1] - pa[:, 1]).norm(dim=-1).max()) * 1e3
            r["noop_vs_recorded_angle_deg_max"] = float(torch.rad2deg(rotation_angle(qb, qa)).max())
            all_ = torch.ones(E, dtype=torch.bool, device=dev)
            for hs, at in (("0.5s", H // 2), ("1s", H)):
                if at < 1:
                    continue
                r[f"recorded_err_pos_mm_{hs}"] = _quantiles((pa[:, at] - rec[:, at, 1, 0:3]).norm(dim=-1) * 1e3, all_)
                pm, qm = runs["moved"][:2]
                r[f"moved_drift_pos_mm_{hs}"] = _quantiles((pm[:, at] - new_pos).norm(dim=-1) * 1e3, found)
                r[f"moved_drift_angle_deg_{hs}"] = _quantiles(torch.rad2deg(rotation_angle(qm[:, at], new_q)), found)
            pm, qm, jm, tm = runs["moved"]
            r["moved_drift_pos_mm_window_max"] = _quantiles((pm - new_pos[:, None]).norm(dim=-1).amax(1) * 1e3, found)
            r["moved_worlds_with_jaw_switch"] = int((jm.any(1) & found).sum())
            r["moved_worlds_with_table_switch_off"] = int(((~tm).any(1) & found).sum())
            r["moved_nonfinite"] = int((~torch.isfinite(pm).all(-1).all(-1) & found).sum())
            ja = runs["recorded"][2]
            r["recorded_worlds_with_jaw_switch"] = int(ja.any(1).sum())
            # worst moved worlds: drift, first jaw switch, clearance of the recorded grasp point (xy and 3D) and the
            # jaw command over the window (closing on air at the old place?)
            wmax = torch.where(found, (pm - new_pos[:, None]).norm(dim=-1).amax(1), torch.zeros(E, dtype=torch.float64, device=dev))
            d3 = (gp - new_pos[:, None]).norm(dim=-1).amin(1)
            dxy = (gp[:, :, 0:2] - new_pos[:, None, 0:2]).norm(dim=-1).amin(1)
            first_jaw = torch.where(jm.any(1), jm.float().argmax(1), torch.full_like(wmax, -1, dtype=torch.long))
            worst = torch.argsort(wmax, descending=True)[:8]
            r["moved_worst"] = [dict(row=int(rows[use[int(i)]]), t0_step=int(tt[i]), drift_mm=float(wmax[i]) * 1e3,
                                     first_jaw_switch_step=int(first_jaw[i]), min_xy_clear_m=float(dxy[i]),
                                     min_3d_clear_m=float(d3[i]), jaw_cmd_min=float(cmds[i, :, 5].min()))
                                for i in worst.tolist()]
            jw = jm.any(1) & found
            r["moved_jaw_switch_worlds_jaw_cmd_closed"] = int((jw & (cmds[:, :, 5].amin(1) < 0.3)).sum())
            res[key] = r
            print(name, key, json.dumps(r), flush=True)
        out["members"][name] = res
    return out


def _oom(e) -> bool:
    """An out-of-memory error; on luffy (NVML broken) the allocator's OOM path raises an NVML assert instead."""
    return isinstance(e, torch.OutOfMemoryError) or "NVML" in str(e) or "out of memory" in str(e)


def mode_bench(args, dev):
    out = dict(env=env_info(dev), rows=[])
    if dev.type == "cuda":
        out["env"]["free_gb_at_start"] = torch.cuda.mem_get_info(dev)[0] / 1e9
    from nedm.so101_policy.kin import SO101Kin
    from nedm.so101_policy.nrd_sim import lowest_point_guard
    kin = SO101Kin(dev)
    named = ckpts(args)
    model, _ = load_model(next(iter(named.values())), dev)
    rows = task_rows(args.data, 30)
    data = R.load_rows(args.data, rows, dev, model.dt)
    R.step_average_rates(data, set(model.config["integrate"]["kinds"]))
    R.add_derived(data, model.config.get("derived"))
    R.add_relative(data, model.config.get("relative"))
    long_ = data["lengths"] >= 401
    st, act = data["states"][long_], data["actions"][long_]
    starts = st[:, 0:400:2].flatten(0, 1)                                     # recorded states as starts
    cmds = act[:, 0:400:2, 0, :6].flatten(0, 1)
    for n in args.sizes:
        pick = torch.arange(n, device=dev) % len(starts)
        s0, a0 = starts[pick], cmds[pick]
        row = dict(worlds=n)
        if dev.type == "cuda":
            row["free_gb_before"] = torch.cuda.mem_get_info(dev)[0] / 1e9
        variants = (((False, True, False), (True, True, False), (True, False, False), (True, True, True))
                    if dev.type == "cuda" else ((False, True, False), (False, False, False), (False, True, True)))
        for graph, canon, guard in variants:
            key = ("graph" if graph else "eager") + ("" if canon else "_canon_off") + ("_guard" if guard else "")
            try:
                sim = GraspNrdSim(named, num_envs=n, device=dev, canonical=canon, cuda_graph=graph,
                                  arm_table_fn=lowest_point_guard(kin) if guard else None)
                sim.reset_state(None, s0)
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(dev)
                t0 = time.perf_counter()
                for _ in range(args.warmup):
                    sim.step(a0)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                row[f"{key}_warmup_s"] = time.perf_counter() - t0
                ts = []
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    for _ in range(args.steps):
                        sim.step(a0)
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    ts.append((time.perf_counter() - t0) / args.steps)
                row[f"{key}_ms"] = float(np.median(ts)) * 1e3
                row[f"{key}_steps_per_s"] = n / float(np.median(ts))
                row[f"{key}_steps_per_s_best"] = n / float(np.min(ts))          # least contention on a shared GPU
                if dev.type == "cuda":
                    row[f"{key}_peak_gb"] = torch.cuda.max_memory_allocated(dev) / 1e9
                row[f"{key}_finite"] = bool(torch.isfinite(sim.state).all())
                del sim
            except RuntimeError as e:
                if not _oom(e):
                    raise
                row[f"{key}_error"] = "out of memory: " + str(e)[:120]
                sim = None
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        try:   # sparse reference: ContactNRD.details + advance, one member, same worlds
            h, ok = model.start(model.finish(s0))
            a = torch.zeros(n, model.n_bodies, model.amax, dtype=torch.float64, device=dev)
            a[:, 0, :6] = a0
            with torch.no_grad():
                for _ in range(args.warmup):
                    nxt, _ = model.details(h, ok, a)
                    h, ok = model.advance(h, ok, nxt)
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                ts = []
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    for _ in range(args.steps):
                        nxt, _ = model.details(h, ok, a)
                        h, ok = model.advance(h, ok, nxt)
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    ts.append((time.perf_counter() - t0) / args.steps)
            row["details_ms"] = float(np.median(ts)) * 1e3
            row["details_steps_per_s"] = n / float(np.median(ts))
            row["details_steps_per_s_best"] = n / float(np.min(ts))
        except RuntimeError as e:
            if not _oom(e):
                raise
            row["details_error"] = "out of memory: " + str(e)[:120]
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        out["rows"].append(row)
        print(json.dumps(row), flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("windows", "parity", "canon", "bench", "disturb"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--ckpt", nargs="+", metavar="NAME=PATH", help="default: STUDY_CKPTS (s61, s62)")
    p.add_argument("--per-family", type=int, default=0, help="validation episodes per task family (0 = all)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--batch", type=int, default=4096, help="windows: worlds per batch")
    p.add_argument("--sizes", type=int, nargs="+", default=[1024, 4096, 16384])
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--parts", nargs="+", default=["halves", "eager", "graph", "teacher", "worldstate"],
                   help="parity: which parts to run")
    p.add_argument("--horizon", type=int, default=50, help="disturb: steps after the disturbance (20 ms)")
    p.add_argument("--clear", type=float, default=0.08, help="disturb: min xy distance (m) of the moved cube")
    p.add_argument("--seed", type=int, default=20261011, help="disturb: seed of the new cube poses")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    t0 = time.perf_counter()
    out = dict(windows=mode_windows, parity=mode_parity, canon=mode_canon, bench=mode_bench,
               disturb=mode_disturb)[args.mode](args, dev)
    out["args"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    out["wall_s"] = time.perf_counter() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
