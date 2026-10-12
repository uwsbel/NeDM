"""PPO for the SO101 + cube goal pick-and-place policy in any training world: the grasp NRD (``--world nrd``), the
kinematic control world (``kin``) or MuJoCo Warp (``mjwarp``, conda env so101_mjwarp). Env: ``nedm.so101_policy.
vec_env`` (decoder v2, task rules, observation, reward); algorithm: ``nedm.so101_policy.ppo`` (G1 PPO adapted).

    PYTHONPATH=src python -P scripts/so101_policy/train_ppo.py train --world nrd --ckpt ~/so101_policy/ckpt/nrd_r1_s61.pt \
        --out RUN [--env-cfg configs/so101_policy/ppo/env_v1.json] [--train-cfg configs/so101_policy/ppo/train_v1.json] \
        [--curriculum configs/so101_policy/ppo/curriculum_tol_v1.json] [--num-envs 4096] [--seed 1] [--members split] \
        [--max-world-steps 3e8] [--max-iterations N] [--max-hours 3.6] [--save-every 25] [--val-every 25] [--val-n 512] \
        [--set key=json ...] [--resume] [--init-from CKPT]
    PYTHONPATH=src python -P scripts/so101_policy/train_ppo.py bank --data ~/so101_nrd_fix/data/production_plus --split 0 \
        --out bank_pp_s0.npz                      # the data start bank as a small .npz (AMD, other machines)
    PYTHONPATH=src python -P scripts/so101_policy/train_ppo.py starts --out starts_fresh.npz --seed0 200000 --n 4096
                                                 # fresh collector starts (CPU MuJoCo sample_starts), a start bank

--set: "env.<key>=json" changes the env config (vec_env.EnvConfig; e.g. env.task='{"tol_xy": 0.03}'), anything else the
train config (ppo.train_cfg_default layout; e.g. num_steps_per_env=24 algorithm.entropy_coef=0.003). PPO's gamma is
also the env's gamma (the task's failure charge discount).

Checkpoints (spec_v2 B4: MuJoCo is not in the training loop of an NRD arm): every ``--save-every`` iterations
``ckpt/model_<it>.pt`` is kept (all of them), plus the final one at the budget = the primary checkpoint ("last at
budget"); ``last.pt`` (+ ``state.json``) is the resume point. Validation in the TRAINING world (deterministic policy,
``--val-n`` fixed tasks drawn from the start bank with ``--val-seed``, decoder seeds 10,000,000 + i) every
``--val-every`` iterations -> ``val_history.json``: the curriculum's promotion signal and the "training-world
validation selected" second row. Budget: ``--max-world-steps`` (20 ms world steps of the training envs), else
iterations / hours (this job's wall clock); a resumed run continues the counters.

Curriculum JSON (tolerance ladder, review P1-13; optional): {"stages": [{"name", "task": {tol_xy, tol_yaw (m, rad;
a key left out = the env config's value)}, "grasp_assist": bool (vec_env docstring), "grasp_assist_cost",
"val_assist": bool (validation with the assist; default = grasp_assist, so an assisted stage is promoted on assisted
validations; val_history.json marks them "assisted": a training-world-selected checkpoint must come from an
unassisted validation), "promote": validation success to move on (absent = last stage), "min_iters", "max_iters",
"std_reset": bool (raise the std to at least train_cfg std_reset at the promotion, default true)}, ...]}.

Outputs in --out: env_cfg.json, train_cfg.json, curriculum.json, manifest.json (argv, git HEAD, sha256 of the NRD
checkpoints and the start bank, torch, GPU), learning_curve.csv (one row per iteration; world_steps = 20 ms steps of
all training envs, gt_transitions = MuJoCo Warp steps incl. validation (0 in the NRD and kinematic worlds), wall_s =
training wall clock summed over jobs), val_history.json, ckpt/model_<it>.pt, last.pt, state.json, done.json /
paused.json, train.log (stdout copy when run with tee).
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from nedm.so101_policy import task as T
from nedm.so101_policy.ppo import (PPO, CurveWriter, Policy, merge, sha256, train_cfg_default, write_json)
from nedm.so101_policy.vec_env import (ACTION_GROUPS, EnvConfig, PickPlaceVecEnv, policy_act_fn, run_episodes,
                                       start_rows)

ROOT = Path(__file__).resolve().parents[2]
VAL_SEED0 = 10_000_000          # decoder macro seeds of the validation tasks
FRESH_SEED0 = 200_000           # default first seed of fresh training starts (never 1000-2023 or >= 900000)


def apply_set(tc, ec, items):
    """--set key=json: "env.<path>" into the env config, anything else into the train config."""
    for kv in items or []:
        k, v = kv.split("=", 1)
        node, ks = tc, k.split(".")
        if ks[0] == "env":
            node, ks = ec, ks[1:]
        for kk in ks[:-1]:
            node = node.setdefault(kk, {})
        node[ks[-1]] = json.loads(v)


def git_state():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
        return dict(head=head, dirty_files=[ln[3:] for ln in dirty.splitlines()][:200])
    except OSError:
        return {}


def val_tasks(env: PickPlaceVecEnv, n, seed):
    """n fixed validation tasks from the env's start bank (CPU generator seeded ``seed``): (starts WorldState,
    goal_xy, goal_yaw, decoder seeds)."""
    bank = {k: v.cpu() for k, v in env.bank.items()}
    gen = torch.Generator().manual_seed(int(seed))
    i, gxy, gyaw = T.make_tasks(bank, n, gen)
    return start_rows(bank, i), gxy, gyaw, torch.arange(VAL_SEED0, VAL_SEED0 + n, dtype=torch.long)


def summarise(r, tol_xy, tol_yaw):
    """Per-episode dict of run_episodes -> summary (success at the env's tolerance and at 15 / 10)."""
    s, n = r["success"], len(r["success"])
    ok15 = s & (r["end_xy"] <= 0.015 + 1e-12) & (r["end_yaw"] <= math.radians(15) + 1e-12)
    ends = {T.ENDS[k]: int(v) for k, v in zip(*np.unique(r["end"], return_counts=True))}
    return dict(n=n, success=float(s.mean()), success15=float(ok15.mean()), success10=float(r["success10"].mean()),
                lifted=float(r["lifted"].mean()), tol_xy=tol_xy, tol_yaw=tol_yaw, ends=ends,
                ep_return=float(r["ret"].mean()), ep_length=float(r["length"].mean()),
                end_xy_mm_success=dict(p50=float(np.median(r["end_xy"][s]) * 1e3), p95=float(np.percentile(r["end_xy"][s], 95) * 1e3))
                if s.any() else None)


def cmd_train(a):
    out = Path(a.out).expanduser()
    resume = a.resume and (out / "state.json").exists()
    if resume:
        ec = json.loads((out / "env_cfg.json").read_text())
        tc = json.loads((out / "train_cfg.json").read_text())
        cur = json.loads((out / "curriculum.json").read_text())
        state = json.loads((out / "state.json").read_text())
    else:
        ec = EnvConfig().to_dict()
        if a.env_cfg:
            merge(ec, {k: v for k, v in json.loads(Path(a.env_cfg).read_text()).items() if k not in ("note", "notes")})
        tc = train_cfg_default(a.seed)
        if a.train_cfg:
            merge(tc, {k: v for k, v in json.loads(Path(a.train_cfg).read_text()).items() if k not in ("note", "notes")})
        ec.update(world=a.world, num_envs=a.num_envs, device=a.device, seed=a.seed)
        if a.ckpt:
            ec["ckpts"] = [str(Path(p).expanduser()) for p in a.ckpt]
        if a.members:
            ec["members"] = a.members if len(a.members) > 1 else a.members[0]
        if a.starts:
            ec["starts"] = str(Path(a.starts).expanduser())
        tc["seed"] = a.seed
        apply_set(tc, ec, a.set)
        ec["gamma"] = tc["algorithm"]["gamma"]
        cur = json.loads(Path(a.curriculum).read_text()) if a.curriculum else {"stages": [{"name": "single"}]}
        state = dict(stage=0, stage_start=0, history=[], val_world_steps=0, jobs=[])
    EnvConfig.from_dict(ec)                                       # unknown keys raise before anything runs
    stages = cur["stages"]
    if a.init_from and not resume and not Path(a.init_from).is_file():
        raise FileNotFoundError(a.init_from)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    if not resume:
        if (out / "state.json").exists():
            raise FileExistsError(f"{out} holds a run; use --resume or a new --out")
        write_json(out / "env_cfg.json", ec)
        write_json(out / "train_cfg.json", tc)
        write_json(out / "curriculum.json", cur)
        sp = Path(ec["starts"]).expanduser()
        man = dict(argv=sys.argv, git=git_state(), host=socket.gethostname(), torch=torch.__version__,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   ckpts={p: sha256(p) for p in ec["ckpts"]} if ec["world"] == "nrd" else {},
                   starts=dict(path=str(sp), sha256=sha256(sp) if sp.is_file() else
                               (sha256(sp / "manifest.json") if (sp / "manifest.json").exists() else None),
                               split=ec["start_split"]),
                   method="PPO (ppo.py) on vec_env.PickPlaceVecEnv: decoder v2 actions, task.PickPlaceTask rules / reward",
                   init_from=dict(path=a.init_from, sha256=sha256(a.init_from)) if a.init_from else None)
        write_json(out / "manifest.json", man)
    torch.manual_seed(int(tc["seed"]))
    np.random.seed(int(tc["seed"]))
    t_job = time.time()
    env = PickPlaceVecEnv(EnvConfig.from_dict(ec))
    val_env = PickPlaceVecEnv(EnvConfig.from_dict(dict(ec, num_envs=a.val_n, auto_reset=False, obs_rate_noise=(0.0, 0.0))),
                              share=env)
    vt = val_tasks(env, a.val_n, a.val_seed)
    pk = dict(tc["policy"])
    pol = Policy(env.num_obs, env.num_actions, [n for _, n in ACTION_GROUPS], init_std=tc["init_std"], **pk).to(env.dev)
    ppo = PPO(env, pol, tc)
    gt_world = ec["world"] in ("mjwarp", "mujoco")
    if resume:
        ppo.load(out / "last.pt")
        print(f"[resume] {out / 'last.pt'} at iteration {ppo.iteration}, stage {state['stage']}, "
              f"{ppo.world_steps / 1e6:.1f} M world steps", flush=True)
    elif a.init_from:
        d = torch.load(a.init_from, map_location=env.dev, weights_only=False)
        pol.load_state_dict(d["policy"])
        ppo.clamp_std()
        print(f"[init] weights from {a.init_from}", flush=True)
    state["jobs"].append(dict(start=time.strftime("%Y-%m-%d %H:%M:%S"), argv=sys.argv, iteration=ppo.iteration,
                              host=socket.gethostname()))

    base_tol = (ec["task"].get("tol_xy", T.TaskConfig().tol_xy), ec["task"].get("tol_yaw", T.TaskConfig().tol_yaw))

    def apply_stage(k):
        over = stages[k].get("task", {})
        bad = sorted(set(over) - {"tol_xy", "tol_yaw"})
        if bad:
            raise ValueError(f"curriculum stage {k}: only tol_xy / tol_yaw can change during a run ({bad})")
        for e in (env, val_env):
            e.set_tolerance(over.get("tol_xy", base_tol[0]), over.get("tol_yaw", base_tol[1]))
        # the grasp assist (vec_env docstring) is a training aid; validation uses it only where the stage says so
        # ("val_assist", default = the stage's grasp_assist: the promotion signal of an assisted stage)
        on = bool(stages[k].get("grasp_assist", ec.get("grasp_assist", False)))
        env.set_grasp_assist(on, stages[k].get("grasp_assist_cost"))
        val_env.set_grasp_assist(bool(stages[k].get("val_assist", on)), stages[k].get("grasp_assist_cost"))

    apply_stage(state["stage"])
    curve = CurveWriter(out / "learning_curve.csv")
    print(f"[env] world {ec['world']}, {env.num_envs} envs, obs {env.num_obs}, actions {env.num_actions}, starts "
          f"{env.bank['q'].shape[0]} ({ec['starts']}), tol {env.tcfg.tol_xy * 1e3:.0f} mm / "
          f"{math.degrees(env.tcfg.tol_yaw):.0f} deg, setup {time.time() - t_job:.1f} s", flush=True)

    def validate():
        t0 = time.time()
        r = run_episodes(val_env, policy_act_fn(pol), *vt)
        val_env.world_steps = 0
        steps = int(r["steps"].sum())
        state["val_world_steps"] += steps
        sm = summarise(r, env.tcfg.tol_xy, env.tcfg.tol_yaw)
        sm.update(iteration=ppo.iteration, stage=state["stage"], stage_name=stages[state["stage"]].get("name"),
                  assisted=bool(val_env.cfg.grasp_assist), world_steps=ppo.world_steps, val_s=time.time() - t0,
                  val_world_steps=steps,
                  std=[float(x) for x in pol.std.detach().cpu()])
        state["history"].append(sm)
        write_json(out / "val_history.json", state["history"])
        print(f"[val] it {ppo.iteration} stage {state['stage']} success {sm['success']:.3f} (15/15 {sm['success15']:.3f},"
              f" 10/10 {sm['success10']:.3f}) lifted {sm['lifted']:.3f} return {sm['ep_return']:.2f} ends {sm['ends']} "
              f"| {sm['val_s']:.1f} s", flush=True)
        return sm

    def save_state():
        ppo.save(out / "last.pt", extra=dict(stage=state["stage"]))
        write_json(out / "state.json", state)

    def budget_left():
        if a.max_world_steps and ppo.world_steps >= a.max_world_steps:
            return False
        if a.max_iterations and ppo.iteration - it0 >= a.max_iterations:
            return False
        return time.time() - t_job < a.max_hours * 3600

    it0 = ppo.iteration
    if not resume and not a.no_initial_val:
        validate()
    while budget_left():
        row = ppo.iterate()
        k = state["stage"]
        row.update(stage=k, tol_xy_mm=env.tcfg.tol_xy * 1e3, tol_yaw_deg=math.degrees(env.tcfg.tol_yaw),
                   gt_transitions=(ppo.world_steps + state["val_world_steps"]) if gt_world else 0,
                   val_world_steps=state["val_world_steps"])
        curve.write(row)
        if ppo.iteration % a.print_every == 0 or ppo.iteration == it0 + 1:
            print(f"[train] it {ppo.iteration} {ppo.world_steps / 1e6:.1f} M steps, {row['world_steps_per_s']:.0f} "
                  f"steps/s (collect), {row['world_steps_per_s_total']:.0f} with update | reward/dec "
                  f"{row['reward_per_decision']:.4f} return {row['ep_return']:.2f} success {row['success']:.3f} lifted "
                  f"{row['lifted']:.3f} len {row['ep_length']:.1f} | kl {row['kl']:.4f} lr {row['lr']:.1e} std "
                  f"{row['std_move']:.2f}/{row['std_trigger']:.2f}/{row['std_lift']:.2f} | mem "
                  f"{row.get('gpu_mem_gb', 0):.1f} GB", flush=True)
        if a.pool_refresh and ppo.iteration % a.pool_refresh == 0:
            env.refresh_pool()
        if ppo.iteration % a.save_every == 0:
            ppo.save(out / "ckpt" / f"model_{ppo.iteration:05d}.pt", extra=dict(stage=k, world_steps=ppo.world_steps))
        if ppo.iteration % a.val_every == 0:
            sm = validate()
            s = stages[k]
            in_stage = ppo.iteration - state["stage_start"]
            ok = "promote" in s and sm["success"] >= s["promote"]
            if k + 1 < len(stages) and ((ok and in_stage >= s.get("min_iters", 0)) or in_stage >= s.get("max_iters", 10 ** 9)):
                reason = "promoted" if ok else "max_iters"
                ppo.save(out / "ckpt" / f"stage{k}_end_{ppo.iteration:05d}.pt", extra=dict(stage=k))
                state.setdefault("stage_ends", []).append(dict(stage=k, iteration=ppo.iteration, reason=reason,
                                                               success=sm["success"], world_steps=ppo.world_steps))
                state["stage"], state["stage_start"] = k + 1, ppo.iteration
                apply_stage(k + 1)
                if stages[k + 1].get("std_reset", True):
                    ppo.set_std(tc["std_reset"], at_least=True)
                print(f"[curriculum] stage {k} -> {k + 1} ({stages[k + 1].get('name')}) at it {ppo.iteration}: {reason}; "
                      f"tol {env.tcfg.tol_xy * 1e3:.0f} mm / {math.degrees(env.tcfg.tol_yaw):.0f} deg", flush=True)
            save_state()
    final = out / "ckpt" / f"model_{ppo.iteration:05d}.pt"
    if not final.exists():
        ppo.save(final, extra=dict(stage=state["stage"], world_steps=ppo.world_steps))
    save_state()
    budget_done = bool(a.max_world_steps) and ppo.world_steps >= a.max_world_steps
    summary = dict(iteration=ppo.iteration, world_steps=ppo.world_steps, decisions=ppo.decisions,
                   gt_transitions=(ppo.world_steps + state["val_world_steps"]) if gt_world else 0,
                   wall_s=ppo.wall, job_s=time.time() - t_job, final_checkpoint=str(final), stage=state["stage"],
                   last_val=state["history"][-1] if state["history"] else None)
    write_json(out / ("done.json" if budget_done or not a.max_world_steps else "paused.json"), summary)
    print(f"[done] {json.dumps({k: v for k, v in summary.items() if k != 'last_val'})}", flush=True)
    env.close()
    val_env.close()


def cmd_bank(a):
    b = T.load_start_bank(Path(a.data).expanduser(), split=a.split)
    out = {k: v.numpy() for k, v in b.items()}
    np.savez(a.out, source=np.array(str(a.data)), split=np.int64(a.split), **out)
    print(f"{len(out['q'])} starts (task families, split {a.split}) of {a.data} -> {a.out}")


def cmd_starts(a):
    from nedm.so101_policy.mujoco_env import sample_starts
    seeds = np.arange(a.seed0, a.seed0 + a.n)
    if ((seeds >= 1000) & (seeds <= 2023)).any() or (seeds >= 900000).any():
        raise ValueError("seeds 1000-2023 (validation / gate) and 900000+ (test) are not training starts")
    t0 = time.time()
    st = sample_starts(seeds)
    s = st["state"]
    np.savez(a.out, seed=seeds, start_q=st["q"], start_cube_pos=st["cube_pos"], start_cube_quat=st["cube_quat"],
             tries=st["tries"], **{k: getattr(s, k).numpy() for k in ("q", "qd", "cube_pos", "cube_quat", "cube_v", "cube_w")})
    print(f"{a.n} fresh starts (seeds {seeds[0]}-{seeds[-1]}) -> {a.out} in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--world", choices=("nrd", "kin", "mjwarp"), default="nrd")
    p.add_argument("--ckpt", nargs="*", help="NRD checkpoint(s)")
    p.add_argument("--members", nargs="*", help="GraspNrdSim members (default split)")
    p.add_argument("--starts", help="start bank: data dir or .npz (default: production_plus split 0)")
    p.add_argument("--env-cfg"), p.add_argument("--train-cfg"), p.add_argument("--curriculum")
    p.add_argument("--out", required=True), p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=4096), p.add_argument("--device", default="cuda")
    p.add_argument("--set", nargs="*"), p.add_argument("--resume", action="store_true")
    p.add_argument("--init-from", help="new run only: policy weights (and normaliser) of this checkpoint")
    p.add_argument("--max-world-steps", type=float, default=0, help="budget in 20 ms world steps (0: none)")
    p.add_argument("--max-iterations", type=int, default=0), p.add_argument("--max-hours", type=float, default=3.6)
    p.add_argument("--save-every", type=int, default=25), p.add_argument("--val-every", type=int, default=25)
    p.add_argument("--val-n", type=int, default=512), p.add_argument("--val-seed", type=int, default=20261211)
    p.add_argument("--pool-refresh", type=int, default=50, help="new task pool every N iterations (0: never)")
    p.add_argument("--print-every", type=int, default=1), p.add_argument("--no-initial-val", action="store_true")
    p.add_argument("--threads", type=int, default=0)
    p = sub.add_parser("bank")
    p.add_argument("--data", required=True), p.add_argument("--split", type=int, default=0), p.add_argument("--out", required=True)
    p = sub.add_parser("starts")
    p.add_argument("--out", required=True), p.add_argument("--seed0", type=int, default=FRESH_SEED0)
    p.add_argument("--n", type=int, default=4096)
    a = ap.parse_args()
    if getattr(a, "threads", 0):
        torch.set_num_threads(a.threads)
    if a.cmd == "train":
        a.max_world_steps = int(a.max_world_steps)
        if a.world == "nrd" and not a.ckpt and not a.resume:
            ap.error("--world nrd needs --ckpt")
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
