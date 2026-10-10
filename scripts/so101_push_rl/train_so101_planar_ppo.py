"""PPO for the SO-101 planar pusher (nedm.so101_push_rl.so101_planar_env; spec SPEC_lead_v1.md A8) with an automatic curriculum.
The direct-command trainer (train_so101_direct_ppo.py) with the env class, observation size and action size
parametrised (ENV_CLASS, OBS_DIM, NUM_ACTIONS below); one env step = one decision of 0.1 s.

python scripts/so101_push_rl/train_so101_planar_ppo.py train --env-cfg env_v1.json --curriculum curriculum_v1.json \
    --val-tasks rot_tasks_rl_val.npz --val-starts rot_starts_rl_val.npz --val-banks rot_bank_rl_val.npz --out run \
    [--seed 1] [--num-envs 16384] [--set key=json ...] [--resume] [--std-reset 0.3] [--start-stage K] [--max-iterations N]
    [--init-from other_run/stage10_end.pt]

Curriculum JSON: {"stages": [{"name", "levels": {level: weight}, "env": {overrides}, "val_levels": [..],
"promote_levels": [..] (default val_levels), "chrono_levels": [..] (default val_levels), "promote": success threshold,
"min_iters", "max_iters"}, ...]}. A stage's "env" may set or change "reset_bank" (reverse curriculum, see
nedm.so101_push_rl.so101_planar_env; the env reloads the bank at the stage change; validation never uses it).
Validation (deterministic actor mean, NRD, every model, up to --val-n tasks of the
stage's val_levels) every --val-every iterations; the stage advances when the validation success over promote_levels
reaches "promote" (after min_iters) or at max_iters. Chrono validation (scripts/so101_push_rl/eval_so101_planar_chrono.py)
of the current checkpoint on --chrono-n tasks of the stage's chrono_levels runs in the background every --chrono-every
iterations and at each promotion (--chrono-n 0: off). At a promotion the action noise std is raised to >= --std-reset.
Optional keys (absent: the rules above): "val_n_per_level" / "chrono_n_per_level" (curriculum top level or stage; the
stage wins): validation / Chrono checks draw this many tasks of EACH level (seed 0) instead of --val-n / --chrono-n
from their union. "promote_each" (stage): a number for every promote level or {level: number}; promotion then also
needs each listed level's validation success >= its number (a stage without "promote" uses only this rule).
Resume: the run directory keeps last.pt + curriculum_state.json; --resume continues (4 h job limit).
Warm start: --init-from <ckpt> makes a NEW run from that checkpoint's actor / critic / std weights only (no optimizer,
iteration, curriculum state or stage); the curriculum starts at --start-stage; --set std_floor=.. raises the std.
With --resume and an existing run directory, --init-from is ignored (the run continues from its last.pt).
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from nedm.so101_push_rl.so101_planar_common import OBS_DIM
from nedm.so101_push_rl.so101_planar_env import SO101PlanarEnv, check_levels, merge

ENV_CLASS, MERGE, NUM_ACTIONS = SO101PlanarEnv, merge, 3
EVAL_SCRIPT = Path(__file__).resolve().parent / "eval_so101_planar_chrono.py"


# run-directory helpers, moved unchanged from the SO-101 tracking trainer (experiment path scripts/training/train_so101_rl_tracking.py)
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def split_optimizer(alg, critic_lr):
    """Two Adam groups: actor (+ std) with PPO's adaptive rate, critic with a fixed rate. RSL-RL 2.2.4 uses one rate
    for both and adapts it to the actor KL; with a small action noise that rate collapses and starves the critic."""
    ac = alg.actor_critic
    actor_params = list(ac.actor.parameters()) + [ac.std]
    ids = {id(p) for p in actor_params}
    critic_params = [p for p in ac.parameters() if id(p) not in ids]
    opt = torch.optim.Adam([{"params": actor_params, "lr": alg.learning_rate}, {"params": critic_params, "lr": critic_lr}])
    orig_step = opt.step

    def step(*args, **kw):
        opt.param_groups[1]["lr"] = critic_lr
        return orig_step(*args, **kw)

    opt.step = step
    alg.optimizer = opt


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def patch_curve(runner, path):
    """Append one CSV row per iteration (means of the logged environment keys, losses, timing)."""
    orig = runner.log
    state = {"header": None}

    def log(locs, width=80, pad=35):
        orig(locs, width, pad)
        row = {"iteration": locs["it"], "collection_s": locs["collection_time"], "learn_s": locs["learn_time"],
               "value_loss": locs["mean_value_loss"], "surrogate_loss": locs["mean_surrogate_loss"],
               "entropy": locs["mean_entropy"], "action_std_mean": float(runner.alg.actor_critic.action_std.mean()),
               "learning_rate": runner.alg.learning_rate, "critic_lr": runner.alg.optimizer.param_groups[-1]["lr"]}
        if locs["ep_infos"]:
            for k in locs["ep_infos"][0]:
                vals = [float(torch.as_tensor(e[k]).float().mean()) for e in locs["ep_infos"] if k in e]
                row[k.strip("/").replace("/", ".")] = float(np.mean(vals))
        new = state["header"] is None
        if new:
            state["header"] = list(row)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=state["header"], extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)

    runner.log = log


def train_cfg_default(seed=1):
    return {
        "algorithm": {"class_name": "PPO", "clip_param": 0.2, "desired_kl": 0.01, "entropy_coef": 0.002, "gamma": 1.0,
                      "lam": 0.95, "learning_rate": 1e-4, "max_grad_norm": 1.0, "num_learning_epochs": 3,
                      "num_mini_batches": 32, "schedule": "adaptive", "use_clipped_value_loss": True, "value_loss_coef": 1.0},
        "policy": {"class_name": "ActorCritic", "activation": "elu", "actor_hidden_dims": [256, 256],
                   "critic_hidden_dims": [256, 256], "init_noise_std": 0.5},
        "num_steps_per_env": 48, "save_interval": 50, "empirical_normalization": False, "logger": "tensorboard", "seed": seed,
        "critic_lr": 3e-4, "actor_out_gain": 0.01,
    }


EXTRA = ("critic_lr", "actor_out_gain", "std_floor")


def apply_overrides(cfg, over):
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


def stage_opt(cur, s, key):
    """A per-stage option that may also be set once at the curriculum top level (the stage value wins)."""
    return s.get(key, cur.get(key))


def select_val_tasks(task_levels, levels, n_max=512, seed=0, n_per_level=0):
    """Task rows of the given levels: up to n_max rows of their union (seed `seed`), or with n_per_level > 0
    min(n_per_level, rows) rows of EACH level (one generator over the sorted levels, the same draw as the Chrono eval's
    --n-per-level). In the per-level mode a level without rows is an error."""
    idx = np.flatnonzero(np.isin(task_levels, levels))
    if n_per_level:
        rng, parts = np.random.default_rng(seed), []
        for lv in sorted({int(x) for x in levels}):
            rows = idx[task_levels[idx] == lv]
            if not len(rows):
                raise ValueError(f"validation level {lv} has no rows in the validation tasks file")
            parts.append(rng.choice(rows, min(int(n_per_level), len(rows)), replace=False))
        return np.sort(np.concatenate(parts))
    if len(idx) > n_max:
        idx = np.sort(np.random.default_rng(seed).choice(idx, n_max, replace=False))
    return idx


def each_thresholds(s):
    """Per-level promotion thresholds {level: success} of a stage's "promote_each" (one number = every promote level);
    None when the stage has no "promote_each"."""
    pe = s.get("promote_each")
    if pe is None:
        return None
    if isinstance(pe, dict):
        return {int(k): float(v) for k, v in pe.items()}
    return {int(lv): float(pe) for lv in s.get("promote_levels", s["val_levels"])}


def promote_ok(sm, s):
    """Promotion rule on a validation summary: pooled promote-level success >= "promote" (when the stage has it) and,
    with "promote_each", every listed level >= its threshold (summarise() decides that part)."""
    return ("promote" not in s or sm["promote_success"] >= s["promote"]) and sm.get("promote_each_ok", True)


def check_curriculum(cur, val_levels_present=None):
    """Fail at job start on bad values of the optional keys (val_n_per_level, chrono_n_per_level, promote_each) or on a
    stage without any promotion rule; with the levels of the validation tasks file, also on a val / chrono / promote_each
    level without rows in a stage that uses a per-level option. Returns warnings for missing levels of the other stages
    (old rule: such levels are skipped without a message), for promote_each with the union validation draw and for
    success.per_level levels that the stage does not use."""
    warn = []
    for key in ("val_n_per_level", "chrono_n_per_level"):
        v = cur.get(key)
        if v is not None and (not isinstance(v, int) or v < 0):
            raise ValueError(f"curriculum {key} must be an int >= 0, got {v!r}")
    for i, s in enumerate(cur["stages"]):
        name = f"stage {i} ({s.get('name')})"
        if "promote" not in s and s.get("promote_each") is None:          # "promote_each": null = no rule
            raise ValueError(f"{name}: needs 'promote' or 'promote_each'")
        for key in ("val_n_per_level", "chrono_n_per_level"):
            v = s.get(key)
            if v is not None and (not isinstance(v, int) or v < 0):
                raise ValueError(f"{name}: {key} must be an int >= 0, got {v!r}")
        th = each_thresholds(s)
        if th is not None:
            if not th:
                raise ValueError(f"{name}: promote_each lists no level")
            miss = sorted(set(th) - {int(x) for x in s["val_levels"]})
            if miss:
                raise ValueError(f"{name}: promote_each levels {miss} are not in val_levels")
            if any(not 0.0 <= v <= 1.0 for v in th.values()):
                raise ValueError(f"{name}: promote_each thresholds must be in [0, 1]")
            if not stage_opt(cur, s, "val_n_per_level"):
                warn.append(f"{name}: promote_each without val_n_per_level: the union draw of --val-n rows may leave out a "
                            f"listed level (the stage then cannot promote)")
        used = {int(x) for k in ("val_levels", "promote_levels", "chrono_levels") for x in s.get(k, [])} | {int(k) for k in s["levels"]}
        typo = sorted({int(k) for k in (((s.get("env") or {}).get("success") or {}).get("per_level") or {})} - used)
        if typo:
            warn.append(f"{name}: success.per_level levels {typo} are not used in this stage (typo?)")
        if val_levels_present is not None:
            have = {int(x) for x in val_levels_present}
            for key, lv in (("val_n_per_level", s["val_levels"]), ("chrono_n_per_level", s.get("chrono_levels", s["val_levels"])),
                            ("promote_each", th or [])):
                miss = sorted({int(x) for x in lv} - have)
                if miss and (th is not None if key == "promote_each" else stage_opt(cur, s, key)):
                    raise ValueError(f"{name}: {key} is set but levels {miss} have no rows in the validation tasks file")
            miss = sorted({int(x) for k in ("val_levels", "promote_levels", "chrono_levels") for x in s.get(k, [])} - have)
            if miss:
                warn.append(f"{name}: levels {miss} have no rows in the validation tasks file (skipped)")
    return warn


@torch.inference_mode()
def evaluate_nrd(env_cfg, actor, tasks_path, starts_path, banks_path, levels, device, n_max=512, seed=0, n_per_level=0):
    """Deterministic rollouts of up to n_max validation tasks of the given levels (or n_per_level tasks of each level)
    with every NRD model (always from the tasks' fresh starts: the training reset bank is switched off)."""
    T = np.load(tasks_path)
    idx = select_val_tasks(T["level"], levels, n_max, seed, n_per_level)
    cfg = MERGE(dict(env_cfg, tasks_path=str(tasks_path), starts_path=str(starts_path), banks_path=str(banks_path),
                     auto_reset=False, device=device, level_weights=None, reset_bank=None))
    jobs = [(int(t), m) for t in idx for m in range(len(cfg["checkpoints"]))]
    env = ENV_CLASS(dict(cfg, num_envs=len(jobs)), device)
    env.reset_idx(torch.arange(len(jobs), device=env.device), task_ids=[j[0] for j in jobs], model_ids=[j[1] for j in jobs])
    obs = env._compute_obs()
    n = len(jobs)
    done = torch.zeros(n, dtype=torch.bool, device=env.device)
    keys = ("success", "invalid", "steps", "pos", "yaw", "touched", "ret", "coverage", "contacts", "refusals")
    res = {k: torch.zeros(n, dtype=torch.float64, device=env.device) for k in keys}
    reasons = {}
    for _ in range(env.max_episode_length + 1):
        obs, r, d, ex = env.step(actor(obs))
        si = ex["step_info"]
        new = (d > 0) & ~done
        res["ret"] += torch.where(~done, r.double(), torch.zeros_like(res["ret"]))
        for k, v in (("success", si["success"]), ("invalid", si["invalid"]), ("touched", si["touched"]), ("pos", si["pos_err"]),
                     ("yaw", si["yaw_err"]), ("steps", si["steps"]), ("coverage", si["coverage"]), ("contacts", si["contacts"]),
                     ("refusals", si["refusals"])):
            res[k] = torch.where(new, v.double(), res[k])
        for k, v in si["reasons"].items():
            reasons[k] = torch.where(new, v, reasons.get(k, torch.zeros_like(v)))
        done |= d > 0
        if done.all():
            break
    lv = T["level"][[j[0] for j in jobs]]
    rows = []
    for i in range(n):
        rows.append(dict(task=jobs[i][0], model=jobs[i][1], level=int(lv[i]), success=bool(res["success"][i]), invalid=bool(res["invalid"][i]),
                         touched=bool(res["touched"][i]), final_pos_mm=float(1e3 * res["pos"][i]), final_yaw_deg=float(torch.rad2deg(res["yaw"][i])),
                         steps=int(res["steps"][i]), ret=float(res["ret"][i]), coverage=float(res["coverage"][i]),
                         contacts=int(res["contacts"][i]), refusals=int(res["refusals"][i]),
                         reasons=[k for k, v in reasons.items() if bool(v[i])]))
    return rows


def summarise(rows, promote_levels=None, promote_each=None):
    """Validation summary. promote_each {level: threshold} (each_thresholds) adds "promote_each" {level: {success,
    need, ok}} and "promote_each_ok" (False if a listed level has no rows)."""
    if not rows:
        return {}
    q = lambda k, p: float(np.quantile([r[k] for r in rows], p))
    out = dict(episodes=len(rows), success=float(np.mean([r["success"] for r in rows])), invalid=float(np.mean([r["invalid"] for r in rows])),
               touched=float(np.mean([r["touched"] for r in rows])), final_pos_mm=dict(median=q("final_pos_mm", .5), p95=q("final_pos_mm", .95)),
               final_yaw_deg=dict(median=q("final_yaw_deg", .5), p95=q("final_yaw_deg", .95)), steps_median=q("steps", .5),
               return_mean=float(np.mean([r["ret"] for r in rows])), coverage_median=q("coverage", .5),
               contacts_mean=float(np.mean([r["contacts"] for r in rows])), refusals_mean=float(np.mean([r["refusals"] for r in rows])))
    out["by_level"] = {int(l): float(np.mean([r["success"] for r in rows if r["level"] == l])) for l in sorted({r["level"] for r in rows})}
    out["by_level_invalid"] = {int(l): float(np.mean([r["invalid"] for r in rows if r["level"] == l])) for l in sorted({r["level"] for r in rows})}
    out["by_model"] = {int(m): float(np.mean([r["success"] for r in rows if r["model"] == m])) for m in sorted({r["model"] for r in rows})}
    rs = {}
    for r in rows:
        for k in r["reasons"]:
            rs[k] = rs.get(k, 0) + 1
    out["invalid_reasons"] = rs
    pl = [r for r in rows if promote_levels is None or r["level"] in promote_levels]
    out["promote_success"] = float(np.mean([r["success"] for r in pl])) if pl else 0.0
    if promote_each is not None:
        out["promote_each"] = {int(l): dict(success=out["by_level"].get(int(l)), need=float(v),
                                            ok=out["by_level"].get(int(l)) is not None and out["by_level"][int(l)] >= v)
                               for l, v in sorted(promote_each.items())}
        out["promote_each_ok"] = all(e["ok"] for e in out["promote_each"].values())
    return out


def make_actor_critic(num_obs, num_actions, tc, device):
    from rsl_rl.modules import ActorCritic
    p = tc["policy"]
    return ActorCritic(num_obs, num_obs, num_actions, actor_hidden_dims=p["actor_hidden_dims"],
                       critic_hidden_dims=p["critic_hidden_dims"], activation=p["activation"], init_noise_std=p["init_noise_std"]).to(device)


def load_actor(run, ckpt, device):
    tc = json.loads((Path(run) / "train_cfg.json").read_text())
    ec = MERGE(json.loads((Path(run) / "env_cfg.json").read_text()))
    ac = make_actor_critic(OBS_DIM, NUM_ACTIONS, tc, device)
    ac.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model_state_dict"])
    ac.eval()
    return (lambda o: ac.act_inference(o.to(torch.float32))), ec


def init_weights(ac, ckpt, device):
    """Warm start: load only the network (actor, critic, std; strict) of a trainer checkpoint into ac. The optimizer
    state, the checkpoint's iteration and any curriculum state of its run are NOT used. Returns the checkpoint iteration."""
    d = torch.load(ckpt, map_location=device, weights_only=False)
    ac.load_state_dict(d["model_state_dict"])
    return d.get("iter")


def chrono_cmd(a, cur, s, out, ck, tag):
    """Background Chrono check of checkpoint ck on the stage's chrono_levels: --chrono-n tasks of their union, or with
    "chrono_n_per_level" that many tasks of each level (the eval's --n-per-level; --n is then ignored)."""
    cmd = [sys.executable, str(EVAL_SCRIPT), "--run", str(out), "--ckpt", str(ck), "--tasks", a.val_tasks, "--starts", a.val_starts,
           "--banks", a.val_banks, "--levels", *[str(x) for x in s.get("chrono_levels", s["val_levels"])], "--n", str(a.chrono_n),
           "--stage-env", json.dumps(s.get("env", {})), "--workers", str(a.chrono_workers), "--out", str(out / f"chrono_val_{tag}")]
    npl = stage_opt(cur, s, "chrono_n_per_level")
    if npl:
        cmd += ["--n-per-level", str(int(npl))]
    return cmd


def cmd_train(a):
    from rsl_rl.runners import OnPolicyRunner
    out = Path(a.out)
    resume = a.resume and (out / "curriculum_state.json").exists()
    # checks before anything is written (a failed new run must not leave its run directory behind)
    cur_path = Path(a.from_run) / "curriculum.json" if a.from_run and not resume else Path(a.curriculum)
    with np.load(a.val_tasks) as z:
        for w in check_curriculum(json.loads(cur_path.read_text()), np.unique(z["level"])):
            print(f"[warn] {w}", flush=True)
    # training levels of EVERY stage (the env checks only the current stage, so a typo in a later stage would stop the run at
    # that stage change): the base tasks file of the run, or a stage's own env.tasks_path
    ec_path = out / "env_cfg.json" if resume else Path(a.from_run) / "env_cfg.json" if a.from_run else Path(a.env_cfg)
    tp0, tl = json.loads(ec_path.read_text())["tasks_path"], {}
    for i, s in enumerate(json.loads(cur_path.read_text())["stages"]):
        tp = (s.get("env") or {}).get("tasks_path", tp0)
        if tp not in tl:
            with np.load(tp) as z:
                tl[tp] = z["level"]
        check_levels(tl[tp], [int(k) for k in s["levels"]], f"{tp} (stage {i} '{s.get('name')}' levels)")
    if a.init_from and not resume and not Path(a.init_from).is_file():
        raise FileNotFoundError(f"--init-from {a.init_from}: no such checkpoint")
    if not resume:
        out.mkdir(parents=True, exist_ok=False)
    write_json(out / f"command_{time.strftime('%m%d_%H%M%S')}.json", dict(argv=sys.argv))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    cur = json.loads(Path(a.curriculum).read_text())
    if resume:
        ec = json.loads((out / "env_cfg.json").read_text())
        tc = json.loads((out / "train_cfg.json").read_text())
        state = json.loads((out / "curriculum_state.json").read_text())
    elif a.from_run:                                  # replicate a run exactly with another seed
        src = Path(a.from_run)
        ec = json.loads((src / "env_cfg.json").read_text())
        tc = json.loads((src / "train_cfg.json").read_text())
        ec.update(seed=a.seed)
        tc["seed"] = a.seed
        cur = json.loads((src / "curriculum.json").read_text())
        write_json(out / "env_cfg.json", ec)
        write_json(out / "train_cfg.json", tc)
        write_json(out / "curriculum.json", cur)
        state = dict(stage=a.start_stage, iteration=0, stage_start=0, history=[], transitions=0)
    else:
        ec = MERGE(json.loads(Path(a.env_cfg).read_text()))
        ec.update(device=a.device, seed=a.seed, num_envs=a.num_envs)
        tc = train_cfg_default(a.seed)
        for kv in a.set or []:
            k, v = kv.split("=", 1)
            node, ks = tc, k.split(".")
            if ks[0] == "env":
                node, ks = ec, ks[1:]
            for kk in ks[:-1]:
                node = node[kk]
            node[ks[-1]] = json.loads(v)
        write_json(out / "env_cfg.json", ec)
        write_json(out / "train_cfg.json", tc)
        write_json(out / "curriculum.json", cur)
        state = dict(stage=a.start_stage, iteration=0, stage_start=0, history=[], transitions=0)
    if not resume:
        man = dict(
            checkpoints={p: sha256(Path(p) / "best.pt" if Path(p).is_dir() else p) for p in ec["checkpoints"]},
            tasks=dict(path=ec["tasks_path"], sha256=sha256(ec["tasks_path"])), val_tasks=dict(path=a.val_tasks, sha256=sha256(a.val_tasks)),
            val_starts=dict(path=a.val_starts, sha256=sha256(a.val_starts)), replicate_of=a.from_run, start_stage=a.start_stage,
            method="PPO from scratch, planar pusher (decision 0.1 s: planar displacement + gripper yaw change, collector IK + command law)")
        if a.init_from:
            man["init_from"] = dict(path=str(a.init_from), sha256=sha256(a.init_from))
            man["method"] = man["method"].replace("PPO from scratch", "PPO warm start (--init-from: actor / critic / std weights)")
        write_json(out / "manifest.json", man)
    stages = cur["stages"]
    if state["stage"] >= len(stages):
        print("curriculum already complete", flush=True)
        return
    st = stages[state["stage"]]
    ec_stage = apply_overrides(copy.deepcopy(ec), st.get("env"))
    ec_stage["level_weights"] = st["levels"]
    env = ENV_CLASS(ec_stage, a.device)
    assert env.num_obs == OBS_DIM and env.num_actions == NUM_ACTIONS
    runner = OnPolicyRunner(env, copy.deepcopy({k: v for k, v in tc.items() if k not in EXTRA}), log_dir=str(out), device=a.device)
    runner.logger_type = tc["logger"]
    if resume:
        if a.init_from:
            print(f"[init] resume: --init-from {a.init_from} ignored, continuing from {out / 'last.pt'}", flush=True)
        runner.load(str(out / "last.pt"), load_optimizer=False)
        runner.current_learning_iteration = state["iteration"] + 1
    elif a.init_from:              # warm start of a new run: weights only; stage = --start-stage, iteration 0, fresh optimizer
        it_src = init_weights(runner.alg.actor_critic, a.init_from, a.device)
        print(f"[init] weights from {a.init_from} (its iteration {it_src}); curriculum stage {state['stage']}", flush=True)
    else:
        # untrained actor mean near 0 (latent) = small displacements; exploration comes from the noise std
        last = [m for m in runner.alg.actor_critic.actor if isinstance(m, torch.nn.Linear)][-1]
        with torch.no_grad():
            last.weight.mul_(float(tc.get("actor_out_gain", 0.01)))
            last.bias.zero_()
    runner.alg.learning_rate = tc["algorithm"]["learning_rate"]
    split_optimizer(runner.alg, tc["critic_lr"])
    patch_curve(runner, out / "learning_curve.csv")
    ac = runner.alg.actor_critic
    std_floor = float(tc.get("std_floor", 0.0))
    if std_floor > 0:          # exploration floor: the action noise std never falls below std_floor (clamped after every update)
        _update = runner.alg.update

        def _update_with_floor(*args, **kwargs):
            out_ = _update(*args, **kwargs)
            with torch.no_grad():
                ac.std.clamp_(min=std_floor)
            return out_
        runner.alg.update = _update_with_floor
        with torch.no_grad():
            ac.std.clamp_(min=std_floor)
        print(f"[std] floor {std_floor}", flush=True)
    t_start = time.time()
    chrono_jobs = []
    per_step = env.num_envs * tc["num_steps_per_env"]
    it_start = state["iteration"]

    def save_state():
        runner.save(str(out / "last.pt"))
        write_json(out / "curriculum_state.json", state)

    def validate(stage_idx):
        s = stages[stage_idx]
        ac.eval()
        ev = apply_overrides(copy.deepcopy(ec), s.get("env"))
        npl, each = stage_opt(cur, s, "val_n_per_level") or 0, each_thresholds(s)
        rows = evaluate_nrd(ev, lambda o: ac.act_inference(o), a.val_tasks, a.val_starts, a.val_banks, s["val_levels"], a.device, a.val_n,
                            n_per_level=npl)
        ac.train()
        sm = summarise(rows, s.get("promote_levels", s["val_levels"]), each)
        sm.update(iteration=state["iteration"], stage=stage_idx, stage_name=s["name"], transitions=state["transitions"],
                  wall_s=time.time() - t_start, action_std=[float(x) for x in ac.std.detach().cpu()])
        state["history"].append(sm)
        write_json(out / "val_history.json", state["history"])
        print(f"[val] it {state['iteration']} stage {stage_idx} ({s['name']}) success {sm['success']:.3f} promote-levels "
              f"{sm['promote_success']:.3f} invalid {sm['invalid']:.3f} touched {sm['touched']:.3f} pos med {sm['final_pos_mm']['median']:.2f} mm "
              f"yaw med {sm['final_yaw_deg']['median']:.2f} deg contacts {sm['contacts_mean']:.2f} by level {sm['by_level']} "
              f"reasons {sm['invalid_reasons']}", flush=True)
        if npl or each is not None:
            print(f"[val-level] it {state['iteration']} per level (n {npl or 'union'}): " + " ".join(
                f"L{lv} success {sm['by_level'][lv]:.3f} invalid {sm['by_level_invalid'][lv]:.3f}" for lv in sm["by_level"])
                + ("" if each is None else f" | promote_each ok {sm['promote_each_ok']} "
                   + " ".join(f"L{lv} {e['success'] if e['success'] is None else round(e['success'], 3)}>={e['need']}"
                              for lv, e in sm["promote_each"].items())), flush=True)
        return sm

    def chrono_val(tag):
        if a.chrono_n <= 0:
            return
        ck = out / f"chrono_ckpt_{tag}.pt"
        runner.save(str(ck))
        cmd = chrono_cmd(a, cur, stages[state["stage"]], out, ck, tag)
        log = open(out / f"chrono_val_{tag}.log", "w")
        chrono_jobs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS="1")))

    if not resume:
        validate(state["stage"])
    while state["stage"] < len(stages) and time.time() - t_start < a.max_hours * 3600:
        if a.max_iterations and state["iteration"] - it_start >= a.max_iterations:
            break
        s = stages[state["stage"]]
        n = a.val_every if not a.max_iterations else min(a.val_every, a.max_iterations - (state["iteration"] - it_start))
        t0 = time.time()
        runner.learn(n, init_at_random_ep_len=False)
        runner.current_learning_iteration += 1
        state["iteration"] += n
        state["transitions"] += n * per_step
        print(f"[train] it {state['iteration']} stage {state['stage']} ({n} in {time.time() - t0:.1f} s)", flush=True)
        sm = validate(state["stage"])
        in_stage = state["iteration"] - state["stage_start"]
        if a.chrono_every and state["iteration"] % a.chrono_every == 0:
            chrono_val(f"it{state['iteration']}")
        ok = promote_ok(sm, s)
        if (in_stage >= s.get("min_iters", 0) and ok) or in_stage >= s.get("max_iters", 10 ** 9):
            reason = "promoted" if ok else "max_iters"
            print(f"[curriculum] stage {state['stage']} ({s['name']}) ends at it {state['iteration']}: {reason}", flush=True)
            chrono_val(f"stage{state['stage']}_end")
            runner.save(str(out / f"stage{state['stage']}_end.pt"))
            state.setdefault("stage_ends", []).append(dict(stage=state["stage"], iteration=state["iteration"], reason=reason, val=sm))
            state["stage"] += 1
            state["stage_start"] = state["iteration"]
            if state["stage"] < len(stages):
                nxt = stages[state["stage"]]
                if a.std_reset > 0:                 # curriculum-aware exploration: restore the action noise for the new stage
                    with torch.no_grad():
                        ac.std.clamp_(min=a.std_reset)
                    print(f"[curriculum] action std reset to >= {a.std_reset}", flush=True)
                env.set_cfg(apply_overrides(copy.deepcopy(ec), nxt.get("env")))
                env.set_level_weights(nxt["levels"])
                with torch.inference_mode():
                    env.reset()
                validate(state["stage"])
        save_state()
    save_state()
    for p in chrono_jobs:
        p.wait()
    write_json(out / "done.json" if state["stage"] >= len(stages) else out / "paused.json", state)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--env-cfg", required=True); p.add_argument("--curriculum", required=True)
    p.add_argument("--val-tasks", required=True); p.add_argument("--val-starts", required=True); p.add_argument("--val-banks", required=True)
    p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=1); p.add_argument("--num-envs", type=int, default=16384)
    p.add_argument("--device", default="cuda"); p.add_argument("--set", nargs="*"); p.add_argument("--resume", action="store_true")
    p.add_argument("--val-every", type=int, default=25); p.add_argument("--val-n", type=int, default=512)
    p.add_argument("--chrono-every", type=int, default=50); p.add_argument("--chrono-n", type=int, default=48)
    p.add_argument("--chrono-workers", type=int, default=16); p.add_argument("--max-hours", type=float, default=3.6)
    p.add_argument("--max-iterations", type=int, default=0, help="stop after this many iterations in this job (0: no limit)")
    p.add_argument("--start-stage", type=int, default=0, help="first curriculum stage of a new run (pilots)")
    p.add_argument("--init-from", default=None, help="new run only: start from this checkpoint's network weights (actor, critic, std); "
                                                     "no optimizer / iteration / stage; ignored when --resume continues a run")
    p.add_argument("--from-run", default=None, help="copy env_cfg / train_cfg / curriculum of this run (new seed)")
    p.add_argument("--std-reset", type=float, default=0.3, help="at each promotion, raise the action noise std to at least this")
    a = ap.parse_args()
    dict(train=cmd_train)[a.cmd](a)


if __name__ == "__main__":
    main()
