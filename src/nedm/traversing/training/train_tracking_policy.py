"""Train the PPO route tracker inside the frozen NRD dynamics model (``tracking_env.TrackingEnv``), rigid and soil.

1. Imitation warm start: ``--imitation-samples`` (observation, recorded PID action) pairs at random decision frames of
   the fragment bank. The runner's empirical observation normaliser is fitted on them first (then frozen for the
   fit), then ``--imitation-epochs`` epochs of MSE between the actor's pre-tanh output and the inverse-squashed PID
   actions. The actor-vs-PID error is logged before and after the warm start and after ``--mse-at`` PPO iterations
   (``imitation.json``).
2. PPO (rsl_rl ``OnPolicyRunner``, 2048 envs x 64 steps per iteration, 5 epochs x 8 mini-batches, adaptive lr from
   3e-4 at KL 0.01, entropy 0.003, ELU actor and critic 512-256-128, initial noise std 0.7).
3. Export: every save (``model_init.pt`` after the warm start, ``model_<it>.pt`` every ``--save-interval`` iterations
   and at the end) also writes ``actor.npz`` (``numpy_actor``) and ``policy_meta.json``, plus per-checkpoint copies
   ``actor_<it>.npz`` / ``policy_meta_<it>.json``. Each export is checked: ``NumpyActor.act`` against the torch actor
   on a fixed pool of ``--check-n`` real observations plus the live observation buffer; max |action difference|
   must be below 1e-5 (``policy_meta.json`` ``numpy_actor_check``).

    PYTHONPATH=src python -m nedm.traversing.training.train_tracking_policy --out runs/ppo_v2 --num-envs 2048 --max-iterations 1000 \\
        --save-interval 100 --imitation-samples 131072 --seed 2 --speed-weight 1.5
    # repeat the parity check of a saved checkpoint (e.g. the released model_999.pt against its actor.npz):
    PYTHONPATH=src python -m nedm.traversing.training.train_tracking_policy --out runs/ppo_v2_check --seed 2 --speed-weight 1.5 \\
        --check-actor artifacts/traverse/generalist_20260921/B_tracker/ppo_v2/model_999.pt \\
        --check-npz artifacts/traverse/generalist_20260921/B_tracker/ppo_v2/actor.npz
    # continue an interrupted run: the same arguments plus --resume runs/ppo_v2/model_<it>.pt

Outputs in ``--out``: env_cfg.json, train_cfg.json, imitation.json, model_*.pt, actor*.npz, policy_meta*.json,
run_state.json and the rsl_rl tensorboard events and git folder.
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

from nedm.traversing.training import dynamics_data as D
from nedm.traversing.training.numpy_actor import NumpyActor, export_torch_actor, jsonable
from nedm.traversing.training.tracking_env import DEFAULT_NRD, TrackingEnv, merge_env_cfg


class NoOpSummaryWriter:
    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        return None

    def save_file(self, *args: Any, **kwargs: Any) -> None:
        return None


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nrd", default=DEFAULT_NRD, help="NRD checkpoint (train_dynamics_model.py)")
    ap.add_argument("--cache", default=D.DEFAULT_CACHE)
    ap.add_argument("--grid", default=D.DEFAULT_GRID, help="the NRD's terrain grid (sha256 must match the checkpoint)")
    ap.add_argument("--twin-split", default=None, help="twin group split to cross-check the cache split against ('none' skips)")
    ap.add_argument("--max-bank-episodes", type=int, default=0, help="> 0: subsample the train episodes (smoke runs)")
    ap.add_argument("--min-frames", type=int, default=40)
    ap.add_argument("--domain-frac", type=float, default=0.5, help="fraction of resets on soil; < 0 = natural mix")
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
    # PPO
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
    ap.add_argument("--logger", default="tensorboard", help="rsl_rl logger; 'none' disables it")
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
    ap.add_argument("--check-n", type=int, default=100, help="observations in the fixed pool of the NumPy-vs-torch check")
    ap.add_argument("--check-actor", default="", help="model_<it>.pt: check its exported actor against the torch actor and exit")
    ap.add_argument("--check-npz", default="", help="the npz for --check-actor (default: actor_<it>.npz next to it)")
    return ap.parse_args(argv)


def env_cfg_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return merge_env_cfg({
        "num_envs": args.num_envs, "device": args.device, "nrd_checkpoint": args.nrd, "grid": args.grid,
        "cache": args.cache, "split": "train", "max_bank_episodes": args.max_bank_episodes, "min_frames": args.min_frames,
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


# ------------------------------------------------------------------------------------------ export
@torch.no_grad()
def numpy_actor_check(runner, env: TrackingEnv, npz_path: Path, pool: torch.Tensor, n: int = 100) -> dict:
    """``NumpyActor(npz).act`` vs the torch actor (normaliser in eval mode, ``act_inference``, squash) on the fixed
    ``pool`` of real observations plus the env's live observation buffer (at most 4096 rows in all)."""
    actor = NumpyActor.from_npz(npz_path)
    obs = torch.cat([pool[:max(int(n), 0)], env.obs_buf])[:4096]
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


def make_runner_class(env: TrackingEnv, out: Path, meta_base: dict, state: dict, check_pool: torch.Tensor, check_n: int = 100):
    from rsl_rl.runners import OnPolicyRunner

    class ExportingRunner(OnPolicyRunner):
        """rsl_rl runner whose every save also exports and checks the NumPy actor."""

        def save(self, path: str, infos=None):
            super().save(path, infos)
            meta = dict(meta_base)
            completed = (int(infos["completed_iterations"]) if isinstance(infos, dict) and "completed_iterations" in infos
                         else int(self.current_learning_iteration) + 1)  # rsl_rl saves after iteration `it`'s update
            meta.update({"iteration": int(self.current_learning_iteration), "completed_iterations": completed, "checkpoint": str(Path(path).name),
                         "imitation": state.get("imitation"), "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")})
            npz = out / "actor.npz"
            export_torch_actor(self.alg.actor_critic, self.obs_normalizer, meta, npz)
            chk = numpy_actor_check(self, env, npz, check_pool, check_n)
            meta["numpy_actor_check"] = chk
            (out / "policy_meta.json").write_text(json.dumps(meta, indent=2, default=jsonable))
            tag = Path(path).stem.removeprefix("model_")  # model_<it>.pt -> actor_<it>.npz, model_init.pt -> actor_init.npz
            shutil.copyfile(npz, out / f"actor_{tag}.npz")
            (out / f"policy_meta_{tag}.json").write_text(json.dumps(meta, indent=2, default=jsonable))
            state["last_export"] = {"iteration": meta["iteration"], "check": chk, "per_checkpoint": f"actor_{tag}.npz"}
            print(f"exported actor.npz + policy_meta.json at iteration {meta['iteration']}: numpy vs torch max|da| {chk['max_abs_action_diff']:.2e}", flush=True)
            if not chk["ok"]:
                raise RuntimeError(f"numpy actor differs from torch by {chk['max_abs_action_diff']:.3e} (>= 1e-5)")

    return ExportingRunner


# ------------------------------------------------------------------------------------------ imitation
@torch.no_grad()
def actor_vs_pid(runner, env: TrackingEnv, data: dict, clip: float, batch: int = 8192) -> dict:
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


def imitation_warm_start(runner, env: TrackingEnv, args, log: dict) -> dict:
    """Fit the runner's observation normalisers on the imitation set, then MSE-fit the actor to the PID actions."""
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


# ------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    env_cfg = env_cfg_from_args(args)
    train_cfg = train_cfg_from_args(args)
    out = D.resolve_path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    env = TrackingEnv(env_cfg, device=args.device)
    print(f"env: {env.num_envs} envs, bank {env.bank.n_episodes} episodes {env.bank.counts()} ({env.bank.n_frames_total} frames), "
          f"obs {env.num_obs}-D, context {env.context}, NRD crop {env.nrd_payload['crop_k']}x{env.nrd_payload['crop_k']}, "
          f"bank load {env.bank.load_s:.1f}s, total init {time.time() - t0:.1f}s", flush=True)
    (out / "env_cfg.json").write_text(json.dumps(env_cfg, indent=2))
    (out / "train_cfg.json").write_text(json.dumps(train_cfg, indent=2))
    meta_base = env.policy_meta()
    meta_base.update({"policy": {"activation": "elu", "actor_hidden_dims": list(args.hidden_dims)}, "train_args": vars(args),
                      "out": str(args.out)})
    # a fixed pool of real observations (random decision frames of the bank) for the NumPy-vs-torch check at every export
    check_pool = env.sample_imitation(max(1, math.ceil(args.check_n / env.num_envs)))["obs"][:args.check_n].clone()
    state: dict = {}
    Runner = make_runner_class(env, out, meta_base, state, check_pool, args.check_n)
    runner = Runner(env, train_cfg, log_dir=str(out), device=args.device)
    # rsl_rl sets logger_type only inside learn() but save() reads it, and the warm-start export saves before learn()
    runner.logger_type = str(args.logger).lower()
    if runner.logger_type in {"none", "off"}:
        runner.writer = NoOpSummaryWriter()
    if args.check_actor:  # re-check a saved checkpoint's exported npz and exit
        ck = D.resolve_path(args.check_actor)
        tag = ck.stem.removeprefix("model_")
        npz = D.resolve_path(args.check_npz) if args.check_npz else ck.parent / f"actor_{tag}.npz"
        if not npz.exists():
            raise SystemExit(f"{npz} not found (pass --check-npz)")
        runner.load(str(ck))
        chk = numpy_actor_check(runner, env, npz, check_pool, args.check_n)
        chk.update({"checkpoint": str(ck), "npz": str(npz), "npz_sha256": D.file_sha256(npz)})
        (out / f"actor_check_{tag}.json").write_text(json.dumps(chk, indent=2))
        print(f"actor check {npz.name} vs {ck.name} on {chk['n']} observations: max|da| {chk['max_abs_action_diff']:.2e} "
              f"(pre-squash {chk['max_abs_presquash_diff']:.2e}) ok={chk['ok']}", flush=True)
        return 0 if chk["ok"] else 1
    done_iters = 0
    imit_data = None
    if args.resume:
        infos = runner.load(str(D.resolve_path(args.resume)))
        # rsl_rl's own saves (model_<it>.pt) hold the weights after iteration `it`'s update and infos=None;
        # model_init.pt records completed_iterations=0
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
        runner.save(str(out / "model_init.pt"), infos={"completed_iterations": 0})  # iteration-0 export
    total = int(args.max_iterations)
    marks = sorted(m for m in args.mse_at if done_iters < m < total) if imit_data is not None else []
    for end in marks + [total]:  # learn() in segments so the actor-vs-PID error can be logged at the marks
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
