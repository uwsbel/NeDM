"""Chrono evaluation of planar pusher policies (nedm.so101_push_rl.so101_planar_chrono.PlanarPolicy) on rotation task tables
(recorded prefix replay to the start frame j_a, then the policy every 20 ms on the measured history, a decision every
5 steps). eval_so101_direct_chrono.py with PlanarPolicy; the same outputs, so so101_direct_replay_gap.py runs on them.

python scripts/so101_push_rl/eval_so101_planar_chrono.py --run <run> --ckpt <ckpt> --tasks rot_tasks_rl_val.npz \
    --starts rot_starts_rl_val.npz --banks rot_bank_rl_val.npz --levels 9 19 --n 48 --workers 16 --out <dir> \
    [--policy name=<run>:<ckpt> ...] [--baseline hold|random ... --env-cfg env.json] [--stage-env '{"success": {...}}']
    [--shard K N] [--task-ids 3 7 ...]

Success (field success) uses the success block of the policy's env config with the stage overrides (pos_tol, yaw_tol,
v_tol, w_tol, hold_steps; yaw is judged unless success.use_yaw is false; success.per_level sets pos_tol / yaw_tol of the
listed task levels, as in the env): first step where the T is inside and slow for
hold_steps consecutive steps, no invalid event before it. Invalid in Chrono: arm-table contact, non-finger link contact,
T height / tilt (run_goal_episode) and the keep-out rule on the measured T (T COM within validity.keepout_radius_m of
the pan axis). Also reported: success at 5 mm / 3 deg and at 10 mm / 5 deg (same rule), the env's other validity rules on the
measured states (logged, not terminal: steps flagged and the first flagged step), contact runs, coverage at the end,
decoder flags and the decision log.

Output: <out>/episodes.jsonl (one row per task and controller), <out>/arrays/task<k>.npz (<name>_states [H+1, 23],
<name>_cmds [H, 5], <name>_targets [H, 3] (decoder target x, y, yaw), <name>_actions [K, 3], <name>_decision_t [K]),
<out>/summary.json, <out>/meta.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

G_ = {}


def init_worker(code_root, policies, baselines, env_cfg_path, stage_env, std, seed):
    import importlib.util
    import torch
    torch.set_num_threads(1)
    from nedm.so101_push_rl import so101_planar_common as P
    from nedm.so101_push_rl.so101_fk import ArmFK
    from nedm.so101_push_rl.so101_kin_torch import So101Kin
    from nedm.so101_push_rl.so101_planar_env import load_stats, merge
    from nedm.so101_push.geometry import TShape
    from nedm.so101_push.robot import load_config
    spec = importlib.util.spec_from_file_location("tp", str(Path(code_root) / "scripts/so101_push_rl/train_so101_planar_ppo.py"))
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)
    G_["code"], G_["ctrls"] = code_root, []
    over = lambda ec: merge({**ec, **{k: ({**ec[k], **v} if isinstance(v, dict) and isinstance(ec.get(k), dict) else v) for k, v in stage_env.items()}})
    entries = [(name, *tp.load_actor(run, ckpt, "cpu"), None) for name, run, ckpt in policies]
    if baselines:
        ec0 = merge(json.loads(Path(env_cfg_path).read_text()))
        entries += [(f"planar_{b}", None, ec0, b) for b in baselines]
    for name, actor, ec, base in entries:
        ec = over(ec)
        ccfg = load_config(ec["collector_config"])
        kin = So101Kin(ccfg, device="cpu")
        fk = ArmFK(ccfg, TShape(ccfg["tshape"]), device="cpu", dtype=torch.float64)
        fk.load_dense(ec["dense_points"])
        hn, amap, _ = load_stats(ec["stats_path"], "cpu")
        obs = P.PlanarObs(hn, amap, torch.tensor(ec["obs"]["delta_max"], dtype=torch.float64))
        G_["ctrls"].append((name, actor, ec, kin, fk, obs, base))
    G_["std"], G_["seed"] = std, seed


def episode_metrics(res, item, ec, ctrl, kin, fk):
    """Planar-specific metrics of one Chrono episode (res = run_goal_episode output)."""
    import torch
    from nedm.so101_push_rl import so101_direct_common as D
    from nedm.so101_push_rl import so101_planar_common as P
    from nedm.so101_push_rl.so101_goal_chrono import goal_metrics
    from nedm.so101_push_rl.so101_planar_chrono import measured_rules
    from nedm.so101_push_rl.so101_planar_env import level_tols
    S, H = res["states"], int(item["horizon"])
    vc, sc = ec["validity"], ec["success"]
    s1 = torch.as_tensor(S[1:], dtype=torch.float64)
    keep = (P.keepout_distance(s1, kin) < vc["keepout_radius_m"]).numpy()
    invalid = np.asarray(res["invalid"]) | keep
    tonly = not bool(sc.get("use_yaw", True))
    sc_stage = {k: sc[k] for k in ("pos_tol", "yaw_tol_deg", "v_tol", "w_tol_deg", "hold_steps")}
    pl = level_tols(sc).get(int(item["level"]))          # success.per_level: the task's level tolerances
    if pl is not None:
        sc_stage.update(pos_tol=pl[0], yaw_tol_deg=pl[1])
    m = goal_metrics(S, np.asarray(item["goal"], float), H, invalid, sc_stage, tonly)
    m5 = goal_metrics(S, np.asarray(item["goal"], float), H, invalid, dict(sc_stage, pos_tol=0.005, yaw_tol_deg=3.0), False)
    m10 = goal_metrics(S, np.asarray(item["goal"], float), H, invalid, dict(sc_stage, pos_tol=0.010, yaw_tol_deg=5.0), False)
    bad, reasons = measured_rules(S, D.DirectGeometry(kin, fk, dense=True), vc, kin)
    rule_steps = {k: int(v.sum()) for k, v in reasons.items()}
    rule_first = {k: int(np.flatnonzero(v.numpy())[0]) + 1 for k, v in reasons.items() if bool(v.any())}
    geo = D.DirectGeometry(kin, fk, dense=True)(torch.as_tensor(S, dtype=torch.float64))
    touch = (geo["finger_sd"] <= ec["touch_m"]).numpy()
    runs = int((touch[1:] & ~touch[:-1]).sum() + int(touch[0]))
    goal = torch.as_tensor(np.asarray(item["goal"], float))[None]
    cov = float(P.coverage(s1[-1:], goal, fk.t_centres[:, :2].to(torch.float64), fk.t_half[:, :2].to(torch.float64),
                           ec["obs"]["coverage_step_m"])[0])
    yaw = np.unwrap(np.arctan2(2 * (S[:, 13] * S[:, 16] + S[:, 14] * S[:, 15]), 1 - 2 * (S[:, 15] ** 2 + S[:, 16] ** 2)))
    out = dict(m, success_5mm3deg=m5["success"], success_10mm5deg=m10["success"], keepout=bool(keep.any()),
               invalid_any=bool(invalid[:H].any()), rules_any=bool(bad.any()), rule_steps=rule_steps, rule_first_step=rule_first,
               contact_runs=runs, touched=bool(touch.any()), final_coverage=cov,
               yaw_change_deg=float(np.degrees(yaw[min(H, len(yaw) - 1)] - yaw[0])), max_abs_yaw_change_deg=float(np.degrees(np.abs(yaw - yaw[0]).max())),
               t_path_mm=float(1e3 * np.linalg.norm(np.diff(S[:, 10:12], axis=0), axis=1).sum()))
    if pl is not None:
        out["success_tol"] = [sc_stage["pos_tol"], sc_stage["yaw_tol_deg"]]
    out.update(ctrl.summary())
    return out


def run_item(item):
    from nedm.so101_push_rl.so101_goal_chrono import run_goal_episode
    from nedm.so101_push_rl.so101_planar_chrono import PlanarPolicy
    rows, arrays = [], {}
    for name, actor, ec, kin, fk, obs, base in G_["ctrls"]:
        ctrl = PlanarPolicy(name, actor, ec, kin, fk, obs, baseline=base, std=G_["std"], seed=G_["seed"] + int(item["task"]))
        t0 = time.time()
        try:
            # run_goal_episode's own metrics use its defaults (5 mm / 3 deg); episode_metrics recomputes them with the
            # stage's success block and the keep-out rule
            res = run_goal_episode(G_["code"], item["start"], item["goal"], item["horizon"], ctrl, translation_only=False)
            arrays[f"{name}_states"], arrays[f"{name}_cmds"] = res["states"], res["cmds"]
            arrays[f"{name}_targets"] = np.asarray(ctrl.targets)
            arrays[f"{name}_actions"] = np.asarray([d["a"] for d in ctrl.decisions])
            arrays[f"{name}_decision_t"] = np.asarray([d["t"] for d in ctrl.decisions])
            du = np.diff(np.vstack([np.asarray(item["start"]["prev_cmd"])[None], res["cmds"]]), axis=0)
            m = episode_metrics(res, item, ec, ctrl, kin, fk)
            m.update(arm_table=bool(res["arm_table"].any()), other_links=bool(res["other_links"].any()), t_bad=bool(res["t_bad"].any()),
                     cmd_change_rms_rad=float(np.sqrt((du ** 2).mean())), cmd_accel_rms_rad=float(np.sqrt((np.diff(du, axis=0) ** 2).mean())),
                     latency_ms_median=float(1e3 * np.median(res["latency_s"])), wall_s=time.time() - t0)
        except Exception as e:
            import traceback
            m = dict(error=repr(e)[:300], trace=traceback.format_exc()[-1500:])
        rows.append(dict(task=item["task"], level=item["level"], start=item["start_idx"], controller=name, goal=list(map(float, item["goal"])), **m))
    np.savez_compressed(Path(item["out"]) / "arrays" / f"task{item['task']:05d}.npz", **arrays)
    return rows


def summarise(rows):
    summ = {}
    for name in sorted({r["controller"] for r in rows}):
        R = [r for r in rows if r["controller"] == name and "error" not in r]
        if not R:
            summ[name] = dict(errors=sum(1 for r in rows if r["controller"] == name))
            continue
        q = lambda k, p: float(np.quantile([r[k] for r in R], p))
        mean = lambda k: float(np.mean([r[k] for r in R]))
        rules = sorted({k for r in R for k in r["rule_steps"]})
        flags = sorted({k for r in R for k in r["flags"]})
        summ[name] = dict(n=len(R), errors=sum(1 for r in rows if r["controller"] == name and "error" in r),
                          success=mean("success"), success_5mm3deg=mean("success_5mm3deg"), success_10mm5deg=mean("success_10mm5deg"),
                          invalid=mean("invalid_any"), arm_table=int(sum(r["arm_table"] for r in R)), other_links=int(sum(r["other_links"] for r in R)),
                          t_bad=int(sum(r["t_bad"] for r in R)), keepout=int(sum(r["keepout"] for r in R)),
                          final_pos_mm=dict(median=q("final_pos_mm", .5), p95=q("final_pos_mm", .95)),
                          final_yaw_deg=dict(median=q("final_yaw_deg", .5), p95=q("final_yaw_deg", .95)),
                          yaw_change_deg_median=q("yaw_change_deg", .5), final_coverage_median=q("final_coverage", .5),
                          contact_runs_mean=mean("contact_runs"), touched=mean("touched"),
                          rules_episode_share={k: float(np.mean([r["rule_steps"].get(k, 0) > 0 for r in R])) for k in rules},
                          decoder_flag_steps_mean={k: float(np.mean([r["flags"].get(k, 0) for r in R])) for k in flags},
                          interventions_mean=mean("interventions"), yaw_blocked_mean=mean("yaw_blocked"),
                          cmd_change_rms_rad=q("cmd_change_rms_rad", .5), cmd_accel_rms_rad=q("cmd_accel_rms_rad", .5),
                          latency_ms_median=q("latency_ms_median", .5), wall_s_median=q("wall_s", .5),
                          by_level={int(l): float(np.mean([r["success"] for r in R if r["level"] == l])) for l in sorted({r["level"] for r in R})})
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True); ap.add_argument("--starts", required=True); ap.add_argument("--banks", required=True)
    ap.add_argument("--run"); ap.add_argument("--ckpt"); ap.add_argument("--policy", action="append", default=[])
    ap.add_argument("--baseline", nargs="*", default=[], choices=["hold", "random"], help="decoder baselines (need --env-cfg)")
    ap.add_argument("--env-cfg", default=None, help="env config of the baselines"); ap.add_argument("--random-std", type=float, default=1.0)
    ap.add_argument("--levels", type=int, nargs="+", default=None); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--n-per-level", type=int, default=0, help="sample this many tasks from each level (instead of --n in total)")
    ap.add_argument("--task-ids", type=int, nargs="*", default=None, help="evaluate exactly these task rows")
    ap.add_argument("--stage-env", default="{}"); ap.add_argument("--shard", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--workers", type=int, default=os.cpu_count()); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--code-root", default=str(Path(__file__).resolve().parents[2])); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.baseline and not a.env_cfg:
        ap.error("--baseline needs --env-cfg")
    T, S, B = np.load(a.tasks), np.load(a.starts), np.load(a.banks)
    if a.task_ids is not None:
        idx = np.asarray(sorted(a.task_ids), int)
    else:
        idx = np.arange(len(T["start_idx"])) if a.levels is None else np.flatnonzero(np.isin(T["level"], a.levels))
        if a.n_per_level:
            rng = np.random.default_rng(a.seed)
            idx = np.sort(np.concatenate([rng.choice(idx[T["level"][idx] == lv], min(a.n_per_level, int((T["level"][idx] == lv).sum())), replace=False)
                                          for lv in sorted(set(T["level"][idx].tolist()))]))
        elif a.n and len(idx) > a.n:
            idx = np.sort(np.random.default_rng(a.seed).choice(idx, a.n, replace=False))
    idx = [int(t) for t in idx if t % a.shard[1] == a.shard[0]]
    policies = ([("policy", a.run, a.ckpt)] if a.run else []) + [(s.split("=", 1)[0], *s.split("=", 1)[1].split(":", 1)) for s in a.policy]
    if not policies and not a.baseline:
        ap.error("give --run/--ckpt, --policy or --baseline")
    out = Path(a.out)
    (out / "arrays").mkdir(parents=True, exist_ok=True)
    items = []
    for t in idx:
        si = int(T["start_idx"][t]); b = int(S["bank_index"][si]); f = int(T["start_frame"][t])
        assert f == int(S["j_a"][si]), "planar tasks start at the rest frame j_a"
        cmds, first = B["cmds"][b], B["first_cmd"][b]
        c1 = cmds[f - 1] if f >= 1 else first
        c2 = cmds[f - 2] if f >= 2 else c1
        start = dict(q_start=S["q_start"][si].tolist(), first_cmd=first.tolist(), prefix_cmds=cmds[:f], j_a=f, prev_cmd=c1, prev_inc=c1 - c2,
                     p_line=S["p_line"][si], gripper_yaw=float(S["yaw"][si]), q_des_prev=S["q_des_prev"][si], q_des_cur=S["q_des_cur"][si],
                     dir_w=S["d"][si], face=str(S["face"][si]))
        items.append(dict(task=t, level=int(T["level"][t]), start_idx=si, start=start, goal=T["goal"][t], horizon=max(int(T["horizon_steps"][t]), int(json.loads(a.stage_env).get("horizon_min_steps") or 0)), out=str(out)))
    (out / "meta.json").write_text(json.dumps(dict(vars(a), items=len(items), policies=policies, node=os.uname().nodename,
                                                   time=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
    rows = []
    init = (a.code_root, policies, a.baseline, a.env_cfg, json.loads(a.stage_env), a.random_std, a.seed)
    with ProcessPoolExecutor(a.workers, initializer=init_worker, initargs=init) as ex, open(out / "episodes.jsonl", "w") as f:
        for fu in as_completed([ex.submit(run_item, it) for it in items]):
            for r in fu.result():
                rows.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
    summ = summarise(rows)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
