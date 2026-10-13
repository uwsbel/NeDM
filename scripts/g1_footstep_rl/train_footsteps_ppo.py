"""PPO for the G1 footstep-target task inside the contact NRD (nedm.g1_footstep_rl.footsteps_env), with a level curriculum.

"spec N" refers to the internal design spec of the study (not released; see the package docstring of
nedm.g1_footstep_rl). The trainer has a split optimizer (actor adaptive lr, critic fixed lr), per-group action std
with a floor, std reset at a promotion, the time-out bootstrap, the command-bias start, resume, --init-from and the
learning-curve CSV; footstep validation metrics and promote_rules drive the curriculum.

    PYTHONPATH=src python scripts/g1_footstep_rl/train_footsteps_ppo.py train --env-cfg configs/g1_footstep_rl/footsteps_env.json \
        --curriculum configs/g1_footstep_rl/footsteps_curriculum.json --out runs/fs_cmd_s1 [--train-cfg ...json]
        [--seed 1] [--num-envs 16384] [--device cuda] [--set key=json ...] [--resume] [--init-from ckpt.pt]
        [--max-iterations N] [--max-hours 3.6] [--val-every 25] [--val-n-per-level 128] [--val-members 0 1]

--env-overrides f.json ...: merged into the env config (e.g. absolute data paths: checkpoints, homie_weights,
bank_path, course_bank, val_course_bank; scripts/g1_footstep_rl/reproduce.sh writes one). --set:
"env.<key>[.<sub>]=json" changes the env config (e.g. env.action.alpha=0, the command-only policy of the release);
any other key changes the train config.

Start policy (new runs): actor and critic MLP 512-256-128 ELU; the actor's last layer x actor_out_gain (0.01) and its
bias set so the mean action maps to command_init (vx 0.4, vy 0, wz 0) with zero leg residual; action std per group
init_std ([command, residual], pre-tanh), floor std_floor, raised to at least std_reset at each promotion. Train from
scratch (spec 10).

Curriculum JSON: {"stages": [{"name", "levels": {level: weight}, "env": {overrides}, "val_levels": [...],
"promote_rules": [{"level", "metric", "min"}, ...], "min_iters", "max_iters"}, ...]}. metric in METRICS:
completion, clean_completion, success (completion and every scored error <= the stage's score.tol), hit3, hit5, hit8
(share of the scored targets of the level within 3 / 5 / 8 cm, pooled over its validation episodes). A stage ends
when every rule holds (after min_iters) or at max_iters.

Validation (deterministic actor, every --val-every iterations; spec 10): for each val level the first
--val-n-per-level courses of the validation course bank (env val_course_bank: seeds 9,000-9,127 per level), each
course once, from a stand start of the start bank drawn with --val-seed, NRD member --val-members[i % len] (members 0
and 1: 64 each of 128). The validation env runs the stage's env config (rules, score, reward) with auto-reset off for
episode_steps steps.

Outputs in --out: env_cfg.json, train_cfg.json, curriculum.json, manifest.json (sha256 of the NRD checkpoints, the
start bank, the course banks and the HOMIE weights), learning_curve.csv, val_history.json, last.pt +
curriculum_state.json (resume), stage<k>_end.pt, model_<it>.pt (RSL-RL), TensorBoard events, done.json / paused.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from nedm.g1_footstep_rl.footsteps_env import LEVELS, G1FootstepsEnv, course_bank, merge_cfg
from nedm.g1_footstep_rl.ppo_utils import (install_std_floor, patch_bootstrap, patch_curve, set_std, sha256,
                                      split_optimizer, write_json)

EXTRA = ("critic_lr", "actor_out_gain", "init_std", "std_floor", "std_reset", "command_init")
METRICS = ("completion", "clean_completion", "success", "hit3", "hit5", "hit8")


def train_cfg_default(seed=1):
    return {
        "algorithm": {"class_name": "PPO", "clip_param": 0.2, "desired_kl": 0.01, "entropy_coef": 0.002, "gamma": 0.99,
                      "lam": 0.95, "learning_rate": 3e-4, "max_grad_norm": 1.0, "num_learning_epochs": 5,
                      "num_mini_batches": 8, "schedule": "adaptive", "use_clipped_value_loss": True, "value_loss_coef": 1.0},
        "policy": {"class_name": "ActorCritic", "activation": "elu", "actor_hidden_dims": [512, 256, 128],
                   "critic_hidden_dims": [512, 256, 128], "init_noise_std": 1.0},
        "num_steps_per_env": 32, "save_interval": 100, "empirical_normalization": False, "logger": "tensorboard",
        "seed": seed,
        "critic_lr": 3e-4, "actor_out_gain": 0.01,
        "init_std": [0.2, 0.5], "std_floor": [0.05, 0.15], "std_reset": [0.15, 0.3],   # [command, leg residual]
        "command_init": [0.4, 0.0, 0.0],
    }


def apply_overrides(cfg, over):
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


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


def groups_of(env, values):
    """Per-group values for the env's action groups ([3] command-only or [3, 12])."""
    if isinstance(values, (int, float)):
        return values
    return list(values)[:len(env.action_groups)]


def check_curriculum(cur, ec):
    """Every stage: known levels with courses in the banks, valid env overrides, promote_rules with known metrics."""
    for i, s in enumerate(cur["stages"]):
        name = f"stage {i} ({s.get('name')})"
        rules = s.get("promote_rules")
        if not isinstance(rules, list) or not rules:
            raise ValueError(f"{name}: needs a non-empty 'promote_rules' list")
        for r in rules:
            if set(r) != {"level", "metric", "min"} or r["metric"] not in METRICS:
                raise ValueError(f"{name}: bad rule {r} (keys level, metric, min; metric in {METRICS})")
            if r["level"] not in s.get("val_levels", []):
                raise ValueError(f"{name}: rule level {r['level']} is not a val level")
        for lv in list(s["levels"]) + list(s.get("val_levels", [])):
            if lv not in LEVELS:
                raise ValueError(f"{name}: unknown level {lv!r} (levels {LEVELS})")
        merge_cfg(apply_overrides(copy.deepcopy(ec), s.get("env")))     # unknown env keys raise here


# ---------------------------------------------------------------------------------------------------------- validation
def val_jobs(train_env, val_bank, levels, n_per_level, seed, members):
    """Fixed validation episodes: (level index, course row of the validation bank, start bank index, member) per
    episode; the first n_per_level courses of each level, starts drawn with seed, members alternating."""
    rng = np.random.default_rng(seed)
    jobs = []
    for lv in levels:
        rows = val_bank.rows_of(lv)[:n_per_level].cpu().numpy()
        if not len(rows):
            raise ValueError(f"no validation course of level {lv} in {val_bank.path}")
        starts = rng.choice(train_env.bank.n, len(rows), replace=train_env.bank.n < len(rows))
        for i, (r, s) in enumerate(zip(rows, starts)):
            jobs.append((LEVELS.index(lv), int(r), int(s), int(members[i % len(members)])))
    return jobs


@torch.inference_mode()
def evaluate_nrd(train_env, env_cfg, actor, levels, n_per_level, seed=0, members=(0, 1), device="cpu"):
    """Deterministic rollouts of the fixed validation episodes -> list of per-episode dicts (values at the first end
    of each episode: completion, failure or time-out)."""
    cfg = merge_cfg(dict(env_cfg, auto_reset=False, seed=seed, course_bank=env_cfg["val_course_bank"], course_bank_max=0,
                         members=sorted({*env_cfg.get("members", [0]), *members}),
                         level_weights={lv: 1.0 for lv in levels}))
    vb = course_bank(cfg["course_bank"], device)
    jobs = val_jobs(train_env, vb, levels, n_per_level, seed, members)
    cfg["num_envs"] = len(jobs)
    env = G1FootstepsEnv(cfg, device, share=train_env)
    env.reset_idx(torch.arange(len(jobs), device=env.device), starts=[j[2] for j in jobs], members=[j[3] for j in jobs],
                  courses=[j[1] for j in jobs])
    obs = env.observe()
    n = len(jobs)
    done = torch.zeros(n, dtype=torch.bool, device=env.device)
    keep = {}
    for _ in range(env.max_episode_length + 1):
        obs, _, d, ex = env.step(actor(obs))
        si = ex["step_info"]
        new = (d > 0) & ~done
        ep = si["episode"]
        flat = dict(completion=si["completed"], clean_completion=si["clean_completion"], success=si["success"],
                    finished=si["finished"], fail=si["fail"], time_out=si["time_out"], steps=si["steps"],
                    seed=si["course_seed"], **{f"reason_{k}": v for k, v in si["reasons"].items()},
                    **{k: ep[k] for k in ("scored", "hits3", "hits5", "hits8", "err_sum", "err_max", "pointer", "n",
                                          "max_abs_l", "max_abs_dpsi", "resteps", "resteps_free", "wrong_foot",
                                          "far_miss", "flickers", "touchdowns_over_gap", "unsupported_steps")},
                    ret=si["returns"]["reward"])
        for k, v in flat.items():
            keep[k] = torch.where(new, v.to(torch.float64), keep.get(k, torch.zeros(n, dtype=torch.float64, device=env.device)))
        done |= d > 0
        if bool(done.all()):
            break
    rows = []
    kp = {k: v.cpu().numpy() for k, v in keep.items()}
    for i, (lv, row, s, m) in enumerate(jobs):
        r = dict(level=LEVELS[lv], course_row=row, start=s, member=m, **{k: float(v[i]) for k, v in kp.items()})
        for k in ("completion", "clean_completion", "success", "finished", "fail", "time_out"):
            r[k] = bool(r[k])
        for x in (3, 5, 8):
            r[f"success{x}"] = r["completion"] and r["err_max"] <= x / 100
        r["reasons"] = [k[7:] for k in kp if k.startswith("reason_") and r[k] > 0]
        r = {k: v for k, v in r.items() if not k.startswith("reason_")}
        rows.append(r)
    return rows


def level_metrics(rs):
    """Metrics of the validation episodes of one level (hit rates pooled over the scored targets)."""
    mean = lambda k: float(np.mean([r[k] for r in rs])) if rs else float("nan")
    scored = sum(r["scored"] for r in rs)
    pooled = lambda k: float(sum(r[k] for r in rs) / scored) if scored else 0.0
    reasons = {}
    for r in rs:
        for k in r["reasons"]:
            reasons[k] = reasons.get(k, 0) + 1
    return dict(n=len(rs), completion=mean("completion"), clean_completion=mean("clean_completion"),
                success=mean("success"), success3=mean("success3"), success5=mean("success5"), success8=mean("success8"),
                hit3=pooled("hits3"), hit5=pooled("hits5"), hit8=pooled("hits8"),
                err_cm=100 * float(sum(r["err_sum"] for r in rs) / scored) if scored else float("nan"),
                fail=mean("fail"), time_out=mean("time_out"), finished=mean("finished"),
                targets_frac=float(np.mean([r["pointer"] / max(r["n"], 1) for r in rs])),
                max_abs_l_p95=float(np.percentile([r["max_abs_l"] for r in rs], 95)), resteps=mean("resteps"),
                wrong_foot=mean("wrong_foot"), flickers=mean("flickers"), scored=mean("scored"), ret=mean("ret"),
                fail_reasons=reasons)


def summarise(rows, rules):
    """Per-level metrics and the promotion check: every rule {level, metric, min} must hold."""
    if not rows:
        return {}
    mean = lambda k, rs: float(np.mean([r[k] for r in rs])) if rs else float("nan")
    out = dict(episodes=len(rows), completion=mean("completion", rows), clean_completion=mean("clean_completion", rows),
               success=mean("success", rows), fail=mean("fail", rows), time_out=mean("time_out", rows))
    out["by_level"] = {lv: level_metrics([r for r in rows if r["level"] == lv]) for lv in sorted({r["level"] for r in rows})}
    out["by_member"] = {int(m): mean("completion", [r for r in rows if r["member"] == m])
                        for m in sorted({r["member"] for r in rows})}
    checks = []
    for rule in rules or []:
        v = out["by_level"].get(rule["level"], {}).get(rule["metric"])
        checks.append(dict(rule, value=v, ok=v is not None and v >= float(rule["min"])))
    out["promote_rules"] = checks
    out["promote_ok"] = bool(checks) and all(c["ok"] for c in checks)
    return out


# ---------------------------------------------------------------------------------------------------------- training
def init_actor(ac, env, tc):
    """New run: actor last layer x actor_out_gain; its bias maps to command_init with zero residual; per-group std."""
    last = [m for m in ac.actor if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        last.weight.mul_(float(tc["actor_out_gain"]))
        last.bias.zero_()
        a0 = env.action_of_command(torch.tensor([tc["command_init"]], dtype=torch.float64))[0]
        last.bias.copy_(a0.to(last.bias))
    set_std(ac, groups_of(env, tc["init_std"]), env.action_groups)


def cmd_train(a):
    from rsl_rl.runners import OnPolicyRunner
    out = Path(a.out)
    resume = a.resume and (out / "curriculum_state.json").exists()
    if resume:
        ec = json.loads((out / "env_cfg.json").read_text())
        tc = json.loads((out / "train_cfg.json").read_text())
        cur = json.loads((out / "curriculum.json").read_text())
        state = json.loads((out / "curriculum_state.json").read_text())
    else:
        ec = merge_cfg(json.loads(Path(a.env_cfg).read_text()))
        ec.update(device=a.device, seed=a.seed, num_envs=a.num_envs)
        tc = train_cfg_default(a.seed)
        if a.train_cfg:
            apply_overrides(tc, json.loads(Path(a.train_cfg).read_text()))
        for f in a.env_overrides or []:               # e.g. absolute data paths (reproduce.sh)
            apply_overrides(ec, {k: v for k, v in json.loads(Path(f).read_text()).items() if k not in ("note", "notes")})
        apply_set(tc, ec, a.set)
        ec = merge_cfg(ec)
        cur = json.loads(Path(a.curriculum).read_text())
        state = dict(stage=a.start_stage, iteration=0, stage_start=0, history=[], transitions=0)
    check_curriculum(cur, ec)
    if a.init_from and not resume and not Path(a.init_from).is_file():
        raise FileNotFoundError(f"--init-from {a.init_from}: no such checkpoint")
    if not resume:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "env_cfg.json", ec)
        write_json(out / "train_cfg.json", tc)
        write_json(out / "curriculum.json", cur)
        man = dict(checkpoints={p: sha256(p) for p in ec["checkpoints"]}, bank=dict(path=ec["bank_path"], sha256=sha256(ec["bank_path"])),
                   course_bank=dict(path=ec["course_bank"], sha256=sha256(ec["course_bank"])),
                   val_course_bank=dict(path=ec["val_course_bank"], sha256=sha256(ec["val_course_bank"])),
                   homie_weights=None if not ec["homie_weights"] else dict(path=ec["homie_weights"], sha256=sha256(ec["homie_weights"])),
                   method="PPO in the G1 contact NRD: HOMIE command (tanh-mapped) + leg residual alpha tanh(r); footstep targets",
                   start_stage=a.start_stage, torch=torch.__version__, spec="g1_footstep_rl/README.md")
        if a.init_from:
            man["init_from"] = dict(path=str(a.init_from), sha256=sha256(a.init_from))
        write_json(out / "manifest.json", man)
    write_json(out / f"command_{time.strftime('%m%d_%H%M%S')}.json", dict(argv=sys.argv))
    torch.manual_seed(int(tc["seed"]))
    np.random.seed(int(tc["seed"]))
    stages = cur["stages"]
    if state["stage"] >= len(stages):
        print("curriculum already complete", flush=True)
        return
    st = stages[state["stage"]]
    env = G1FootstepsEnv(dict(apply_overrides(copy.deepcopy(ec), st.get("env")), level_weights=st["levels"]), a.device)
    print(f"[env] {env.num_envs} envs, obs {env.num_obs}, actions {env.num_actions}, bank {env.bank.summary}, "
          f"courses {env.courses.summary}", flush=True)
    runner = OnPolicyRunner(env, copy.deepcopy({k: v for k, v in tc.items() if k not in EXTRA}), log_dir=str(out), device=a.device)
    runner.logger_type = tc["logger"]
    ac = runner.alg.actor_critic
    runner.alg.learning_rate = float(state.get("learning_rate", tc["algorithm"]["learning_rate"]))
    split_optimizer(runner.alg, float(tc["critic_lr"]))
    if resume:
        runner.load(str(out / "last.pt"), load_optimizer=True)
        runner.current_learning_iteration = state["iteration"] + 1
        print(f"[resume] {out / 'last.pt'} at iteration {state['iteration']}, stage {state['stage']}", flush=True)
    elif a.init_from:
        d = torch.load(a.init_from, map_location=a.device, weights_only=False)
        ac.load_state_dict(d["model_state_dict"])
        print(f"[init] weights from {a.init_from} (its iteration {d.get('iter')}); stage {state['stage']}", flush=True)
    else:
        init_actor(ac, env, tc)
    install_std_floor(runner.alg, groups_of(env, tc["std_floor"]), env.action_groups)
    patch_bootstrap(runner)
    patch_curve(runner, out / "learning_curve.csv", extra=lambda: dict(stage=state["stage"]))
    per_step = env.num_envs * tc["num_steps_per_env"]
    t_start = time.time()
    it_start = state["iteration"]

    def save_state():
        state["learning_rate"] = float(runner.alg.learning_rate)
        runner.save(str(out / "last.pt"))
        write_json(out / "curriculum_state.json", state)

    def validate(stage_idx):
        s = stages[stage_idx]
        ac.eval()
        ev = apply_overrides(copy.deepcopy(ec), s.get("env"))
        t0 = time.time()
        rows = evaluate_nrd(env, ev, lambda o: ac.act_inference(o), s["val_levels"], a.val_n_per_level, a.val_seed,
                            a.val_members, a.device)
        ac.train()
        sm = summarise(rows, s["promote_rules"])
        sm.update(iteration=state["iteration"], stage=stage_idx, stage_name=s["name"], transitions=state["transitions"],
                  wall_s=time.time() - t_start, val_s=time.time() - t0, action_std=[float(x) for x in ac.std.detach().cpu()])
        state["history"].append(sm)
        write_json(out / "val_history.json", state["history"])
        lv = " | ".join(f"{k} compl {v['completion']:.3f} clean {v['clean_completion']:.3f} hit5 {v['hit5']:.3f} "
                        f"err {v['err_cm']:.1f} cm targets {v['targets_frac']:.2f} fail {v['fail']:.2f}"
                        for k, v in sm["by_level"].items())
        rules = " ".join(f"{c['level']}.{c['metric']} {c['value']:.3f}>={c['min']}{'' if c['ok'] else ' NO'}"
                         for c in sm["promote_rules"])
        print(f"[val] it {state['iteration']} stage {stage_idx} ({s['name']}) | {lv} | rules: {rules} | "
              f"{time.time() - t0:.1f} s", flush=True)
        return sm

    if not resume and not a.no_initial_val:
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
        dt = time.time() - t0
        print(f"[train] it {state['iteration']} stage {state['stage']} ({n} its in {dt:.1f} s, {n * per_step / dt:.0f} env steps/s "
              f"incl. updates)", flush=True)
        sm = validate(state["stage"])
        in_stage = state["iteration"] - state["stage_start"]
        ok = sm["promote_ok"]
        if (in_stage >= s.get("min_iters", 0) and ok) or in_stage >= s.get("max_iters", 10 ** 9):
            reason = "promoted" if ok else "max_iters"
            print(f"[curriculum] stage {state['stage']} ({s['name']}) ends at it {state['iteration']}: {reason}", flush=True)
            runner.save(str(out / f"stage{state['stage']}_end.pt"))
            state.setdefault("stage_ends", []).append(dict(stage=state["stage"], iteration=state["iteration"], reason=reason,
                                                           rules=sm["promote_rules"]))
            state["stage"] += 1
            state["stage_start"] = state["iteration"]
            if state["stage"] < len(stages):
                nxt = stages[state["stage"]]
                set_std(ac, groups_of(env, tc["std_reset"]), env.action_groups, at_least=True)
                env.set_cfg(dict(apply_overrides(copy.deepcopy(ec), nxt.get("env")), level_weights=nxt["levels"]))
                env.set_level_weights(nxt["levels"])
                env.reset()
                print(f"[curriculum] stage {state['stage']} ({nxt['name']}): std >= {tc['std_reset']}", flush=True)
                validate(state["stage"])
        save_state()
    save_state()
    write_json(out / "done.json" if state["stage"] >= len(stages) else out / "paused.json", state)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--env-cfg", required=False); p.add_argument("--curriculum", required=False)
    p.add_argument("--train-cfg", default=None, help="json overrides of the train config defaults")
    p.add_argument("--env-overrides", nargs="*", help="json files merged into the env config (e.g. cluster data paths)")
    p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=16384); p.add_argument("--device", default="cuda")
    p.add_argument("--set", nargs="*"); p.add_argument("--resume", action="store_true")
    p.add_argument("--init-from", default=None, help="new run only: actor / critic / std weights of this checkpoint")
    p.add_argument("--start-stage", type=int, default=0)
    p.add_argument("--val-every", type=int, default=25); p.add_argument("--val-n-per-level", type=int, default=128)
    p.add_argument("--val-seed", type=int, default=0); p.add_argument("--val-members", type=int, nargs="+", default=[0, 1])
    p.add_argument("--no-initial-val", action="store_true")
    p.add_argument("--max-iterations", type=int, default=0, help="stop after this many iterations in this job (0: no limit)")
    p.add_argument("--max-hours", type=float, default=3.6)
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0: torch default)")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if not a.resume and (not a.env_cfg or not a.curriculum):
        ap.error("a new run needs --env-cfg and --curriculum")
    dict(train=cmd_train)[a.cmd](a)


if __name__ == "__main__":
    main()
