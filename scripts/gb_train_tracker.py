#!/usr/bin/env python
"""PPO route tracker inside the mixed rigid / deformable-soil NRD (PLAN B5; fork of traverse_wp3_train_tracker.py).

Same rsl_rl runner and PPO block as WP3 on ``gb_tracker_env.GBTrackingEnv`` (one static grid, schema-3 fragment bank
from the train groups, 158-D ``gc_control.PolicyObs`` observation, full-box action squash, progress reward).  New:
  * PID-imitation warm start: ``--imitation-samples`` (observation, recorded PID action) pairs at random decision
    frames of the bank; the runner's empirical observation normaliser is fitted on them FIRST (then frozen for the
    fit), then ``--imitation-epochs`` epochs of MSE between the actor's pre-tanh output and the inverse-squashed PID
    actions; the actor-vs-PID error (pre-tanh MSE and physical-action MAE per channel) is logged after the warm
    start (PPO iteration 0) and after 10 PPO iterations, into ``imitation.json``.
  * exports: every rsl_rl save (``model_<it>.pt``, every ``--save-interval`` iterations and at the end) also writes
    ``actor.npz`` (``gc_control.export_torch_actor``: obs normaliser + MLP + squash, evaluated in numpy by the
    collectors) and ``policy_meta.json`` (obs layout, squash, NRD checkpoint sha256, cache manifest sha256, iteration),
    and checks ``NumpyActor.act`` against the torch actor on a fixed pool of ``--check-n`` (100) real observations
    (random decision frames of the bank, drawn once at start-up) plus the live observation buffer (max |da| < 1e-5).
    ``--check-actor model_<it>.pt`` re-runs that check for a saved checkpoint against its ``actor_<it>.npz`` and exits.
  * ``--resume model_<it>.pt`` restores actor-critic, normalisers, optimizer and iteration (no warm start).
  * ``--smoke`` runs the scripted pure-pursuit controller and a random policy, and the env-vs-PolicyObs check.
  * every export is also kept per checkpoint (``actor_<it>.npz`` / ``policy_meta_<it>.json`` next to ``model_<it>.pt``);
    a ``cond='notag'`` NRD is refused unless ``--allow-notag`` (PLAN B4: PPO uses the tag NRD); the bank cross-checks
    the manifest's split against the twin group split (``--twin-split``).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT / "src"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gc_control as GC  # noqa: E402
from gb_nrd_common import file_sha256  # noqa: E402
from gb_tracker_env import GBTrackingEnv, merge_env_cfg, pure_pursuit_actions  # noqa: E402


class NoOpSummaryWriter:
    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        return None

    def save_file(self, *args: Any, **kwargs: Any) -> None:
        return None


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nrd", required=True, help="gb_train_nrd.py checkpoint (cond=tag for PPO)")
    ap.add_argument("--cache", default="artifacts/traverse/generalist_20260921/B_tracker/cache_v1")
    ap.add_argument("--grid", default=None, help="override of the checkpoint's grid path (sha256 must match)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-bank-episodes", type=int, default=0)
    ap.add_argument("--min-frames", type=int, default=40)
    ap.add_argument("--domain-frac", type=float, default=0.5, help="fraction of resets on deformable soil; <0 = natural mix")
    ap.add_argument("--twin-split", default=None, help="twin group split npz to cross-check the cache's split against (default: the one the manifest names; 'none' skips)")
    ap.add_argument("--allow-notag", action="store_true", help="accept a cond='notag' NRD (PLAN B4: PPO uses the tag NRD)")
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--fragment-steps", type=int, nargs=2, default=[20, 60])
    ap.add_argument("--steering-rate-limit", type=float, default=0.1)
    ap.add_argument("--cross-track-sigma", type=float, default=1.0)
    ap.add_argument("--heading-sigma", type=float, default=0.35)
    ap.add_argument("--speed-sigma", type=float, default=1.0)
    ap.add_argument("--cross-track-weight", type=float, default=2.0)
    ap.add_argument("--heading-weight", type=float, default=0.8)
    ap.add_argument("--speed-weight", type=float, default=0.5)
    ap.add_argument("--action-rate-weight", type=float, default=0.2)
    ap.add_argument("--throttle-brake-weight", type=float, default=0.05)
    ap.add_argument("--progress-weight", type=float, default=0.5)
    ap.add_argument("--max-cross-track", type=float, default=6.0)
    # PPO (WP3 / HMMWV tracking study values)
    ap.add_argument("--max-iterations", type=int, default=1000)
    ap.add_argument("--num-steps-per-env", type=int, default=64)
    ap.add_argument("--num-learning-epochs", type=int, default=5)
    ap.add_argument("--num-mini-batches", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--desired-kl", type=float, default=0.01)
    ap.add_argument("--entropy-coef", type=float, default=0.003)
    ap.add_argument("--init-noise-std", type=float, default=0.7)
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256, 128])
    ap.add_argument("--save-interval", type=int, default=100)
    ap.add_argument("--logger", default="tensorboard")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # imitation warm start
    ap.add_argument("--imitation-samples", type=int, default=131072, help="0 disables the warm start")
    ap.add_argument("--imitation-epochs", type=int, default=2)
    ap.add_argument("--imitation-lr", type=float, default=1e-3)
    ap.add_argument("--imitation-batch", type=int, default=4096)
    ap.add_argument("--imitation-clip", type=float, default=0.99, help="inverse-squash clip of the box edges (atanh)")
    ap.add_argument("--mse-at", type=int, nargs="*", default=[10], help="PPO iterations after which the actor-vs-PID error is logged again")
    ap.add_argument("--resume", default="", help="model_<it>.pt of an interrupted run")
    ap.add_argument("--check-n", type=int, default=100, help="observations in the fixed pool of the numpy-vs-torch actor check")
    ap.add_argument("--check-actor", default="", help="model_<it>.pt: check its actor_<it>.npz against the torch actor on --check-n observations and exit")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-steps", type=int, default=60)
    return ap.parse_args(argv)


def env_cfg_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return merge_env_cfg({
        "num_envs": args.num_envs, "device": args.device, "nrd_checkpoint": args.nrd, "grid": args.grid,
        "cache": args.cache, "split": args.split, "max_bank_episodes": args.max_bank_episodes, "min_frames": args.min_frames,
        "domain_frac": None if args.domain_frac < 0 else float(args.domain_frac), "twin_split": args.twin_split,
        "fragment_steps_min": args.fragment_steps[0], "fragment_steps_max": args.fragment_steps[1],
        "steering_rate_limit": args.steering_rate_limit,
        "reward": {
            "cross_track_sigma_m": args.cross_track_sigma, "heading_sigma_rad": args.heading_sigma,
            "speed_sigma_mps": args.speed_sigma, "cross_track_weight": args.cross_track_weight,
            "heading_weight": args.heading_weight, "speed_weight": args.speed_weight,
            "action_rate_weight": args.action_rate_weight, "throttle_brake_weight": args.throttle_brake_weight,
            "progress_weight": args.progress_weight,
        },
        "termination": {"max_cross_track_m": args.max_cross_track},
        "seed": args.seed,
    })


def train_cfg_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "algorithm": {
            "class_name": "PPO", "clip_param": 0.2, "desired_kl": float(args.desired_kl),
            "entropy_coef": float(args.entropy_coef), "gamma": 0.99, "lam": 0.95,
            "learning_rate": float(args.learning_rate), "max_grad_norm": 1.0,
            "num_learning_epochs": int(args.num_learning_epochs), "num_mini_batches": int(args.num_mini_batches),
            "schedule": "adaptive", "use_clipped_value_loss": True, "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {"activation": "elu", "actor_hidden_dims": list(args.hidden_dims), "critic_hidden_dims": list(args.hidden_dims),
                   "init_noise_std": float(args.init_noise_std), "class_name": "ActorCritic"},
        "runner": {"checkpoint": -1, "experiment_name": Path(args.out).name, "load_run": -1, "log_interval": 1,
                   "max_iterations": int(args.max_iterations), "record_interval": -1, "resume": False, "resume_path": None, "run_name": ""},
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": int(args.num_steps_per_env),
        "save_interval": int(args.save_interval),
        "empirical_normalization": True,
        "logger": args.logger,
        "seed": int(args.seed),
    }


# =============================================================================================== exports
@torch.no_grad()
def numpy_actor_check(runner, env: GBTrackingEnv, npz_path: Path, pool: torch.Tensor | None = None, n: int = 100) -> dict:
    """NumpyActor(actor.npz).act vs the torch actor (normaliser in eval mode + act_inference + squash) on real
    observations: the fixed ``pool`` (``n`` observations at random decision frames of the bank, drawn once at start-up)
    plus the env's live observation buffer at the time of the save (capped at 4096 rows)."""
    actor = GC.NumpyActor.from_npz(npz_path)
    obs = env.obs_buf.clone() if pool is None else torch.cat([pool[:max(int(n), 0)], env.obs_buf])
    obs = obs[:4096]
    norm, ac = runner.obs_normalizer, runner.alg.actor_critic
    was_training = norm.training
    norm.eval()
    pre = ac.act_inference(norm(obs))
    if was_training:
        norm.train()
    ref = env._scale_policy_actions(pre).double().cpu().numpy()
    mine = actor.act(obs.double().cpu().numpy())
    d = float(np.abs(mine - ref).max())
    return {"n": int(obs.shape[0]), "max_abs_action_diff": d, "ok": bool(d < 1e-5),
            "max_abs_presquash_diff": float(np.abs(actor.mlp(obs.double().cpu().numpy()) - pre.double().cpu().numpy()).max())}


def make_runner_class(env: GBTrackingEnv, out: Path, meta_base: dict, state: dict, check_pool: torch.Tensor | None = None, check_n: int = 100):
    from rsl_rl.runners import OnPolicyRunner

    class ExportingRunner(OnPolicyRunner):
        def save(self, path: str, infos=None):
            super().save(path, infos)
            meta = dict(meta_base)
            completed = (int(infos["completed_iterations"]) if isinstance(infos, dict) and "completed_iterations" in infos
                         else int(self.current_learning_iteration) + 1)  # rsl_rl saves after iteration `it`'s update
            meta.update({"iteration": int(self.current_learning_iteration), "completed_iterations": completed, "checkpoint": str(Path(path).name),
                         "imitation": state.get("imitation"), "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")})
            npz = out / "actor.npz"
            GC.export_torch_actor(self.alg.actor_critic, self.obs_normalizer, meta, npz)
            chk = numpy_actor_check(self, env, npz, check_pool, check_n)
            meta["numpy_actor_check"] = chk
            (out / "policy_meta.json").write_text(json.dumps(meta, indent=2, default=GC._jsonable))
            tag = Path(path).stem.removeprefix("model_")  # model_<it>.pt -> actor_<it>.npz, model_init.pt -> actor_init.npz
            shutil.copyfile(npz, out / f"actor_{tag}.npz")
            (out / f"policy_meta_{tag}.json").write_text(json.dumps(meta, indent=2, default=GC._jsonable))
            state["last_export"] = {"iteration": meta["iteration"], "check": chk, "per_checkpoint": f"actor_{tag}.npz"}
            print(f"exported actor.npz + policy_meta.json at iteration {meta['iteration']}: numpy vs torch max|da| {chk['max_abs_action_diff']:.2e}", flush=True)
            if not chk["ok"]:
                raise RuntimeError(f"numpy actor differs from torch by {chk['max_abs_action_diff']:.3e} (>= 1e-5)")

    return ExportingRunner


# =============================================================================================== imitation
@torch.no_grad()
def actor_vs_pid(runner, env: GBTrackingEnv, data: dict, clip: float, batch: int = 8192) -> dict:
    """Pre-tanh MSE and physical-action errors of the deterministic actor against the recorded PID actions."""
    norm, ac = runner.obs_normalizer, runner.alg.actor_critic
    was_training = norm.training
    norm.eval()
    tgt_pre = env.physical_to_policy(data["action"], clip)
    se_pre, ae_phys, se_phys, n = torch.zeros(3, device=env.device), torch.zeros(3, device=env.device), torch.zeros(3, device=env.device), 0
    per_dom = {d: [torch.zeros(3, device=env.device), 0] for d in (0, 1)}
    for i in range(0, data["obs"].shape[0], batch):
        o = data["obs"][i:i + batch]
        pre = ac.act_inference(norm(o))
        phys = env._scale_policy_actions(pre)
        se_pre += ((pre - tgt_pre[i:i + batch]) ** 2).sum(0)
        err = phys - data["action"][i:i + batch]
        ae_phys += err.abs().sum(0); se_phys += (err ** 2).sum(0); n += o.shape[0]
        for d in (0, 1):
            m = data["domain"][i:i + batch] == d
            per_dom[d][0] += err[m].abs().sum(0); per_dom[d][1] += int(m.sum())
    if was_training:
        norm.train()
    f = lambda t: [float(v) for v in (t / max(n, 1)).cpu()]
    out = {"n": n, "pretanh_mse_per_channel": f(se_pre), "pretanh_mse": float(se_pre.sum() / max(3 * n, 1)),
           "phys_mae_per_channel": f(ae_phys), "phys_mse_per_channel": f(se_phys)}
    for d, name in ((0, "rigid"), (1, "crm")):
        if per_dom[d][1]:
            out[f"phys_mae_per_channel/{name}"] = [float(v) for v in (per_dom[d][0] / per_dom[d][1]).cpu()]
    return out


def imitation_warm_start(runner, env: GBTrackingEnv, args, log: dict) -> dict:
    """Fit the runner's obs normalisers on the imitation set, then MSE-fit the actor to the PID actions."""
    n_resets = max(1, math.ceil(args.imitation_samples / env.num_envs))
    t0 = time.time()
    data = env.sample_imitation(n_resets)
    N = int(data["obs"].shape[0])
    # 1. the runner's own empirical normaliser, fitted on the imitation observations first
    for nm in (runner.obs_normalizer, runner.critic_obs_normalizer):
        nm.train()
        for i in range(0, N, 8192):
            nm.update(data["obs"][i:i + 8192])
        nm.eval()
    before = actor_vs_pid(runner, env, data, args.imitation_clip)
    # 2. MSE on the pre-tanh actions
    ac = runner.alg.actor_critic
    opt = torch.optim.Adam(ac.actor.parameters(), lr=args.imitation_lr)
    tgt = env.physical_to_policy(data["action"], args.imitation_clip)
    obs_n = torch.cat([runner.obs_normalizer(data["obs"][i:i + 8192]) for i in range(0, N, 8192)])
    g = torch.Generator(device=env.device); g.manual_seed(args.seed)
    hist = []
    for ep in range(args.imitation_epochs):
        perm = torch.randperm(N, device=env.device, generator=g)
        tot, cnt = 0.0, 0
        for i in range(0, N, args.imitation_batch):
            idx = perm[i:i + args.imitation_batch]
            loss = torch.nn.functional.mse_loss(ac.actor(obs_n[idx]), tgt[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx); cnt += len(idx)
        hist.append(tot / max(cnt, 1))
    after = actor_vs_pid(runner, env, data, args.imitation_clip)
    rep = {"samples": N, "resets": n_resets, "epochs": args.imitation_epochs, "lr": args.imitation_lr, "clip": args.imitation_clip,
           "domain_counts": {"rigid": int((data["domain"] == 0).sum()), "crm": int((data["domain"] == 1).sum())},
           "epoch_mse": hist, "before": before, "after_warm_start_iter0": after, "wall_s": time.time() - t0}
    print(f"imitation: {N} samples, pre-tanh MSE {before['pretanh_mse']:.4f} -> {after['pretanh_mse']:.4f}; "
          f"physical MAE per channel {np.round(before['phys_mae_per_channel'], 4).tolist()} -> {np.round(after['phys_mae_per_channel'], 4).tolist()} "
          f"({time.time() - t0:.0f}s)", flush=True)
    log["imitation"] = rep
    return data


# =============================================================================================== smoke
@torch.no_grad()
def smoke(env: GBTrackingEnv, steps: int) -> dict[str, Any]:
    out: dict[str, Any] = {"policy_obs_check": env.check_against_policy_obs(n=16)}
    print("env obs vs gc_control.PolicyObs:", json.dumps(out["policy_obs_check"]), flush=True)
    for name in ("pure_pursuit", "random"):
        env.reset()
        ct, sp, rew, adv, fails, ends, n_done = [], [], [], [], 0, 0, 0
        torch.cuda.synchronize() if env.device.type == "cuda" else None
        t0 = time.time()
        for _ in range(steps):
            act = pure_pursuit_actions(env) if name == "pure_pursuit" else torch.randn(env.num_envs, 3, device=env.device)
            _, r, dones, extras = env.step(act)
            ct.append(extras["log"]["/tracking/cross_track_abs_m"].item()); sp.append(extras["log"]["/tracking/speed_err_abs_mps"].item())
            adv.append(extras["log"]["/tracking/advance_m"].item()); rew.append(r.mean().item())
            if "episode" in extras:
                k = int(dones.sum()); n_done += k
                fails += extras["episode"]["/episode/fail_rate"].item() * k; ends += extras["episode"]["/episode/route_end_rate"].item() * k
        torch.cuda.synchronize() if env.device.type == "cuda" else None
        dt = time.time() - t0
        out[name] = {"cross_track_abs_m": float(np.mean(ct)), "speed_err_abs_mps": float(np.mean(sp)), "advance_m_per_step": float(np.mean(adv)),
                     "reward": float(np.mean(rew)), "reward_finite": bool(np.isfinite(rew).all()), "done": n_done,
                     "fail_frac": fails / max(n_done, 1), "route_end_frac": ends / max(n_done, 1), "env_steps_per_s": steps * env.num_envs / dt}
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[name].items()}), flush=True)
    return out


# =============================================================================================== main
def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    env_cfg = env_cfg_from_args(args)
    train_cfg = train_cfg_from_args(args)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    env = GBTrackingEnv(env_cfg, device=args.device)
    print(f"env: {env.num_envs} envs, bank {env.bank.n_episodes} episodes {env.bank.counts()} ({env.bank.n_frames_total} frames, split {args.split}), "
          f"obs {env.num_obs}-D, context {env.context}, NRD cond={env.nrd_payload['cond']} crop {env.nrd_payload['crop_k']}x{env.nrd_payload['crop_k']}, "
          f"bank load {env.bank.load_s:.1f}s, total init {time.time() - t0:.1f}s; "
          f"GPU peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB" if torch.cuda.is_available() else "", flush=True)
    if env.nrd_payload["cond"] != "tag" and not args.allow_notag:
        raise SystemExit(f"NRD {args.nrd} is cond={env.nrd_payload['cond']!r}; PLAN B4 trains PPO in the tag NRD (pass --allow-notag to override)")
    (out / "env_cfg.json").write_text(json.dumps(env_cfg, indent=2))
    (out / "train_cfg.json").write_text(json.dumps(train_cfg, indent=2))
    meta_base = env.policy_meta()
    meta_base.update({"policy": {"activation": "elu", "actor_hidden_dims": list(args.hidden_dims)}, "train_args": vars(args),
                      "out": str(out)})
    if args.smoke:
        res = smoke(env, args.smoke_steps)
        (out / "smoke.json").write_text(json.dumps(res, indent=2))
        return 0
    # a fixed pool of real observations (random decision frames of the bank) for the numpy-vs-torch actor check at every export
    check_pool = env.sample_imitation(max(1, math.ceil(args.check_n / env.num_envs)))["obs"][:args.check_n].clone()
    state: dict = {}
    Runner = make_runner_class(env, out, meta_base, state, check_pool, args.check_n)
    runner = Runner(env, train_cfg, log_dir=str(out), device=args.device)
    # rsl_rl sets logger_type only inside learn() but save() reads it, and the warm-start export saves before learn()
    runner.logger_type = str(args.logger).lower()
    if runner.logger_type in {"none", "off"}:
        runner.writer = NoOpSummaryWriter()
    if args.check_actor:  # re-check a saved checkpoint's exported npz (the artefact the collectors load) and exit
        ck = Path(args.check_actor)
        tag = ck.stem.removeprefix("model_")
        npz = ck.parent / f"actor_{tag}.npz"
        if not npz.exists():
            raise SystemExit(f"{npz} not found next to {ck}")
        runner.load(str(ck))
        chk = numpy_actor_check(runner, env, npz, check_pool, args.check_n)
        chk.update({"checkpoint": str(ck), "npz": str(npz), "npz_sha256": file_sha256(npz)})
        (out / f"actor_check_{tag}.json").write_text(json.dumps(chk, indent=2))
        print(f"actor check {npz.name} vs {ck.name} on {chk['n']} observations: max|da| {chk['max_abs_action_diff']:.2e} "
              f"(pre-squash {chk['max_abs_presquash_diff']:.2e}) ok={chk['ok']}", flush=True)
        return 0 if chk["ok"] else 1
    done_iters = 0
    imit_data = None
    if args.resume:
        infos = runner.load(args.resume)
        # rsl_rl's own saves (model_<it>.pt) hold the weights AFTER iteration `it`'s update and infos=None; model_init.pt
        # records completed_iterations=0 explicitly
        done_iters = (int(infos["completed_iterations"]) if isinstance(infos, dict) and "completed_iterations" in infos
                      else int(runner.current_learning_iteration) + 1)
        runner.current_learning_iteration = done_iters
        print(f"resumed {args.resume}: {done_iters} iterations completed, continuing at iteration {done_iters}", flush=True)
        state["imitation"] = {"skipped": "resumed run"}
    elif args.imitation_samples > 0:
        imit_data = imitation_warm_start(runner, env, args, state)
        (out / "imitation.json").write_text(json.dumps(state["imitation"], indent=2))
    else:
        state["imitation"] = {"skipped": "disabled"}
    if not args.resume:
        runner.save(str(out / "model_init.pt"), infos={"completed_iterations": 0})  # iteration-0 export (warm-started or fresh)
    total = int(args.max_iterations)
    marks = sorted(m for m in args.mse_at if done_iters < m < total) if imit_data is not None else []
    seg_ends = marks + [total]
    for end in seg_ends:
        n = end - done_iters
        if n <= 0:
            continue
        runner.learn(num_learning_iterations=n, init_at_random_ep_len=False)  # ends with a save of model_<last it>.pt (+ export)
        done_iters = int(runner.current_learning_iteration) + 1
        runner.current_learning_iteration = done_iters  # the next segment continues at the next index
        if imit_data is not None and end in marks:
            m = actor_vs_pid(runner, env, imit_data, args.imitation_clip)
            state["imitation"][f"after_ppo_iter{end}"] = m
            (out / "imitation.json").write_text(json.dumps(state["imitation"], indent=2))
            print(f"actor vs PID after {end} PPO iterations: pre-tanh MSE {m['pretanh_mse']:.4f}, physical MAE {np.round(m['phys_mae_per_channel'], 4).tolist()}", flush=True)
    (out / "run_state.json").write_text(json.dumps({"iterations": done_iters, "max_iterations": total, "last_export": state.get("last_export"),
                                                    "wall_s": time.time() - t0}, indent=2))
    print(f"done: {done_iters} iterations, wall {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
