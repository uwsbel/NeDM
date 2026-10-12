"""Evaluate a saved policy (or the FSM expert) deterministically on a fixed task set in any world, and compare worlds.

    PYTHONPATH=src python -P scripts/so101_policy/eval_policy.py run --policy RUN/ckpt/model_00200.pt \
        --tasks ~/so101_policy/runs/gate/tasks_1024.npz --world mujoco [--workers 4 --mj-python ~/miniconda3/envs/nrdsim/bin/python] \
        [--n 64] [--first 0] --out eval_mujoco.npz
    ... run --policy CKPT --tasks TASKS --world nrd --ckpt ~/so101_policy/ckpt/nrd_r1_s61.pt --device cuda --out eval_nrd.npz
    ... run --policy CKPT --tasks TASKS --world kin|mjwarp --device cuda --out ...
    ... run --policy expert ...                                  # the FSM expert (privileged; sanity check of the harness)
    PYTHONPATH=src python -P scripts/so101_policy/eval_policy.py report --runs eval_*.npz --ref eval_mujoco.npz \
        --out report.json [--md report.md]

Tasks: a ``gate_expert.py tasks`` file (start seeds, the collector's start sampler + 0.3 s warm-up; goals by
``task.sample_goals``): CPU MuJoCo resets every world with the collector warm-up and checks the settled state against the
stored one (<= 1e-6); the other worlds start from the stored settled state. Never the test seeds (900000+) except for
the final, pre-registered test runs; never the sealed test data.

Episode (``vec_env.PickPlaceVecEnv`` with auto_reset off, ``vec_env.run_episodes``): the policy's deterministic action
(tanh of the pre-tanh mean) once per 0.1 s decision, decoder v2 per 20 ms step, the task rules after every step. Rules:
``TaskConfig.evaluation()`` in every world (the grasp-box rule logs, as in the gate; ``--train-rules`` = training
rules); the task's link / jaw overlap rules run in every world except CPU MuJoCo (real contacts and the collector's
visual-overlap end there). Decoder macro seeds = the task seeds (identical durations in every world).

Decoder and observation (review_code_v2 R1): a checkpoint written by ``train_ppo.py`` holds the training env's resolved
DecoderV2Config and TaskConfig (``PickPlaceVecEnv.resolved()``); the evaluation env uses that decoder config (every
field) and the training run's ``obs_rates``, with the evaluation rules and the default success tolerance (15 mm /
15 deg; 10 / 10 reported) in every world. ``--decoder-cfg JSON`` (DecoderV2Config overrides) sets the decoder of
``--policy expert``; for a checkpoint it must agree with the checkpoint's decoder (else the run stops). A checkpoint
without that record (before 2026-10-11 23:00) runs with the default decoder and a NOTE.

Output (.npz, the gate_expert ``run`` format, so its report functions apply): per episode end (``task.ENDS`` code),
success (15 mm / 15 deg), success10, lifted, end_xy / end_yaw, t_end (20 ms steps), ret / length (decision reward sum,
decisions), mj_reason (CPU MuJoCo collector end), first_fire / lift_step / close_x; per step q, cube (pos + quat), xg
(cube in the grasp frame), u (joint targets), mode (decoder), macro; wall_s, world_steps, policy path and sha256.

``--assist`` (diagnostic only): the training grasp assist of ``vec_env`` (out-of-box GRASP triggers ignored, the
trigger held through the descent) in the evaluated world, to compare worlds for policies of an assisted curriculum
stage; the output is marked ``assist``. ``--stochastic`` (diagnostic only): the policy sampled with its own std from
counter-based noise of (task seed, decision index), the same noise in every world (paired comparison of a policy
that still relies on exploration noise, e.g. for its triggers). Never for a reported success rate.

``report``: per run success (Wilson 95 % interval), 10 / 10, lifted, failure causes, end error, phase times, in-hand
slip; for each run against ``--ref`` (CPU MuJoCo): the 2 x 2 table (n11 n10 n01 n00), agreement, phi, exact McNemar p
(``gate_expert.table2x2``), lift agreement, cause pairs and the divergence point of disagreeing episodes.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("gate_expert", ROOT / "scripts" / "so101_policy" / "gate_expert.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)


class StochasticAct:
    """A ppo.Policy sampled with its own action std and counter-based noise (``gate_expert.hash_normal`` of the task
    seed and the decision index): the same noise sequence for a task in every world (paired, diagnostic)."""

    def __init__(self, policy, seeds, noise_seed=0):
        self.pol, self.k = policy, 0
        self.nseed = seeds.long() * 7919 + 104729 * int(noise_seed) + 31

    def __call__(self, obs, env):
        mu = self.pol.act_inference(obs.to(next(self.pol.parameters()).device)).to(env.dev)
        eps = G.hash_normal(self.nseed.to(env.dev), self.k, mu.shape[1]).to(mu)
        self.k += 1
        return mu + self.pol.std.detach().to(mu) * eps, False


class ExpertAct:
    """The FSM expert (expert.PickPlaceExpert on the env's decoder) as an act_fn of ``vec_env.run_episodes``."""

    def __init__(self, cfg=None):
        self.cfg, self.ex = cfg or {}, None

    def reset(self, env):
        from dataclasses import replace

        from nedm.so101_policy.expert import ExpertConfig, PickPlaceExpert
        self.ex = PickPlaceExpert(env.dec, replace(ExpertConfig(), **self.cfg))
        self.ex.reset(torch.arange(env.B, device=env.dev), env.ws, env.task.goal_xy.clone(), env.task.goal_yaw.clone())

    def __call__(self, obs, env):
        return self.ex.act(env.ws), True


def tasks_subset(path, n=0, first=0):
    t = G.load_tasks(path)
    if n:
        t = G.load_tasks(path, np.arange(first, first + n))
    if (np.asarray(t["seeds"]) >= 900000).any():
        print("NOTE: test seeds (900000+) in this task set", flush=True)
    return t


def checkpoint_env(policy, decoder_cfg=None, log=print):
    """(DecoderV2Config overrides, TaskConfig overrides, provenance dict or None) of the evaluation env for a policy
    checkpoint (module docstring, R1), or for the expert (decoder_cfg as given)."""
    if policy == "expert":
        return dict(decoder_cfg or {}), {}, None
    d = torch.load(policy, map_location="cpu", weights_only=False)
    prov = d.get("env")
    if prov is None:
        if decoder_cfg:
            raise ValueError(f"{policy}: no decoder record in the checkpoint; cannot check --decoder-cfg")
        if log:
            log(f"NOTE {policy}: checkpoint without the env record (before R1): default decoder and observation")
        return {}, {}, None
    dec = {k: (tuple(v) if isinstance(v, list) else v) for k, v in prov["decoder"].items()}
    if decoder_cfg:
        from dataclasses import asdict, replace

        from nedm.so101_policy.decoder import DecoderV2Config
        want = asdict(replace(DecoderV2Config(**dec), **decoder_cfg))
        if want != asdict(DecoderV2Config(**dec)):
            raise ValueError(f"--decoder-cfg {decoder_cfg} differs from the checkpoint's decoder")
    return dec, dict(obs_rates=bool(prov["task"]["obs_rates"])), prov


@torch.no_grad()
def run_policy(tasks, world, policy="expert", ckpt=None, device="cpu", workers=0, mj_python=None, train_rules=False,
               horizon=None, record=True, members="split", assist=False, stochastic=False, noise_seed=0, log=print,
               decoder_cfg=None):
    """Episodes of one policy on all tasks in one world -> dict of numpy arrays (module docstring). decoder_cfg:
    DecoderV2Config overrides (expert; a checkpoint's own decoder must agree)."""
    from nedm.so101_policy.ppo import load_policy, sha256
    from nedm.so101_policy.vec_env import EnvConfig, PickPlaceVecEnv, policy_act_fn, run_episodes
    B = len(tasks["seeds"])
    t0 = time.time()
    dec_over, task_over, prov = checkpoint_env(policy, decoder_cfg, log)
    if horizon is not None:
        task_over["horizon"] = int(horizon)
    files = {}                                # the checkpoint's geometry / kin points files where they exist here
    for k in ("geom_path", "kin_points"):
        p = (prov or {}).get("env", {}).get(k)
        if p and Path(p).expanduser().exists():
            files[k] = p
    cfg = EnvConfig(world=world, num_envs=B, device=device, ckpts=[str(Path(c).expanduser()) for c in (ckpt or [])],
                    members=members, auto_reset=False, eval_rules=not train_rules, task=task_over, mj_workers=workers,
                    mj_python=mj_python, starts="", nrd_cuda_graph=True, grasp_assist=bool(assist), decoder=dec_over,
                    **files)
    env = PickPlaceVecEnv(cfg, record=record)
    dev = env.dev
    starts = G.start_state(tasks, dev)
    g = lambda k: torch.as_tensor(tasks[k], dtype=torch.float64, device=dev)              # noqa: E731
    seeds = torch.as_tensor(tasks["seeds"], dtype=torch.long, device=dev)
    mj_start = dict(q=tasks["start_q"], cube_pos=tasks["start_cube_pos"], cube_quat=tasks["start_cube_quat"])
    if policy == "expert":
        act, hook, pinfo = ExpertAct(), None, dict(policy="expert")
        hook = act.reset
    else:
        pol = load_policy(policy, device=dev)
        if pol.kwargs["obs_dim"] != env.num_obs or pol.kwargs["act_dim"] != env.num_actions:
            raise RuntimeError(f"policy {pol.kwargs['obs_dim']} obs / {pol.kwargs['act_dim']} actions vs env "
                               f"{env.num_obs} / {env.num_actions}")
        act = StochasticAct(pol, seeds, noise_seed) if stochastic else policy_act_fn(pol)
        hook, pinfo = None, dict(policy=str(policy), policy_sha256=sha256(policy))
    r = run_episodes(env, act, starts, g("goal_xy"), g("goal_yaw"), seeds, mj_start=mj_start if world == "mujoco" else None,
                     reset_hook=hook, log=log)
    rec = env.records() if record else {}
    env.close()
    out = dict(rec)
    out.update(end=r["end"], success=r["success"], success10=r["success10"], lifted=r["lifted"], end_xy=r["end_xy"],
               end_yaw=r["end_yaw"], t_end=r["steps"], ret=r["ret"], length=r["length"], seeds=np.asarray(tasks["seeds"]),
               box_log=np.zeros(B, bool), world=np.array(world), config=np.array("policy"), decoder=np.array("v2"),
               ckpt=np.array(str(ckpt)), wall_s=np.float64(time.time() - t0), device=np.array(str(dev)),
               world_steps=np.int64(env.world_steps), phase=np.zeros_like(rec["mode"]) if record else np.zeros(0),
               label=np.zeros_like(rec["mode"]) if record else np.zeros(0), phase_names=np.array(["policy"]),
               train_rules=np.bool_(train_rules), assist=np.bool_(assist), stochastic=np.bool_(stochastic),
               noise_seed=np.int64(noise_seed), env_resolved=np.array(json.dumps(env.resolved(), default=str)),
               ckpt_env=np.array(json.dumps(prov, default=str) if prov is not None else ""),
               **{k: np.array(v) for k, v in pinfo.items()})
    return out


def summary(r):
    from nedm.so101_policy.task import ENDS
    n = len(r["end"])
    c = G.cause_names(r)
    s = r["success"]
    return dict(world=str(r["world"]) + (" ASSISTED" if bool(r.get("assist", False)) else "")
                + (" STOCHASTIC" if bool(r.get("stochastic", False)) else ""), ckpt=str(r["ckpt"]),
                policy=str(r.get("policy", "")), n=n,
                success15=float(s.mean()), success15_ci=G.wilson(int(s.sum()), n), success10=float(r["success10"].mean()),
                lifted=float(r["lifted"].mean()),
                causes={k: int(v) for k, v in zip(*np.unique(c, return_counts=True))},
                rule_fired_any={ENDS[3 + j]: int((r["first_fire"][:, j] >= 0).sum()) for j in range(r["first_fire"].shape[1])
                                if (r["first_fire"][:, j] >= 0).any()} if "first_fire" in r else {},
                end_xy_mm=dict(p50=float(np.nanmedian(r["end_xy"][s]) * 1e3), p95=float(np.nanpercentile(r["end_xy"][s], 95) * 1e3))
                if s.any() else None,
                end_s=dict(p50=float(np.median(r["t_end"]) * 0.02), p95=float(np.percentile(r["t_end"], 95) * 0.02)),
                ep_return=float(np.mean(r["ret"])) if "ret" in r else None, wall_s=float(r["wall_s"]),
                world_steps=int(r["world_steps"]) if "world_steps" in r else None)


def cmd_run(a):
    tasks = tasks_subset(a.tasks, a.n, a.first)
    torch.set_num_threads(a.threads)
    r = run_policy(tasks, a.world, a.policy, a.ckpt, a.device, a.workers, a.mj_python, a.train_rules, a.horizon,
                   not a.no_record, a.members[0] if a.members and len(a.members) == 1 else (a.members or "split"),
                   a.assist, a.stochastic, a.noise_seed, decoder_cfg=json.loads(a.decoder_cfg) if a.decoder_cfg else None)
    np.savez_compressed(a.out, **r)
    sm = summary(r)
    print(f"{a.world}{' (ASSISTED)' if a.assist else ''}{' (STOCHASTIC)' if a.stochastic else ''} {sm['policy']}: success 15/15 {sm['success15'] * 100:.1f} % ({sm['success15_ci'][0] * 100:.1f}-"
          f"{sm['success15_ci'][1] * 100:.1f}), 10/10 {sm['success10'] * 100:.1f} %, lifted {sm['lifted'] * 100:.1f} %; "
          f"causes {sm['causes']}; wall {sm['wall_s']:.0f} s -> {a.out}", flush=True)


def cmd_report(a):
    runs = {Path(p).stem: dict(np.load(p, allow_pickle=False)) for p in a.runs}
    rep = dict(runs={k: summary(r) for k, r in runs.items()}, agreement={})
    if a.ref:
        ref = dict(np.load(a.ref, allow_pickle=False))
        rep["ref"] = str(a.ref)
        for name, r in runs.items():
            if Path(a.ref).stem == name:
                continue
            if not np.array_equal(ref["seeds"], r["seeds"]):
                raise ValueError(f"{name}: tasks differ from the reference")
            dis = np.flatnonzero(ref["success"] != r["success"])
            ag = dict(success15=G.table2x2(ref["success"], r["success"]), success10=G.table2x2(ref["success10"], r["success10"]),
                      lift=G.table2x2(ref["lifted"], r["lifted"]),
                      cause_pairs={f"{x} | {y}": int(v) for (x, y), v in G._pairs(G.cause_names(ref), G.cause_names(r)).items()},
                      disagree_tasks=dis.tolist())
            if "cube" in ref and "cube" in r and len(ref["cube"]) and len(r["cube"]):
                rows, _ = G.divergence_rows(ref, r, dis, list(ref["phase_names"]))
                byp = {}
                for row in rows:
                    k = row.get("mj_mode", "never") or "never"
                    byp[k] = byp.get(k, 0) + 1
                ag["divergence_by_ref_mode"] = byp
                ag["divergence_examples"] = rows[: a.examples]
                both = ref["success"] & r["success"]
                ag["final_cube_diff_mm_both_success"] = G._final_diff(ref, r, both)
            rep["agreement"][name] = ag
    Path(a.out).write_text(json.dumps(rep, indent=1, default=float))
    md = markdown(rep)
    print(md)
    if a.md:
        Path(a.md).write_text(md)


def markdown(rep):
    L = ["| run | world | success 15/15 (95 % CI) | 10/10 | lifted | main causes | wall s |", "|---|---|---|---|---|---|---|"]
    for name, r in rep["runs"].items():
        top = ", ".join(f"{k} {v}" for k, v in sorted(r["causes"].items(), key=lambda kv: -kv[1])[:5])
        ci = r["success15_ci"]
        L.append(f"| {name} | {r['world']} | {r['success15'] * 100:.1f} % ({ci[0] * 100:.1f}-{ci[1] * 100:.1f}) | "
                 f"{r['success10'] * 100:.1f} % | {r['lifted'] * 100:.1f} % | {top} | {r['wall_s']:.0f} |")
    if rep["agreement"]:
        L += ["", f"Reference: {rep.get('ref')}", "",
              "| run vs ref | success: n11 n10 n01 n00 | agree | phi | McNemar p | lift: n11 n10 n01 n00 | lift agree |",
              "|---|---|---|---|---|---|---|"]
        for name, ag in rep["agreement"].items():
            s, l = ag["success15"], ag["lift"]
            phi = "nan" if not np.isfinite(s["phi"]) else f"{s['phi']:.2f}"
            L.append(f"| {name} | {s['n11']} {s['n10']} {s['n01']} {s['n00']} | {s['agree'] * 100:.1f} % | {phi} | "
                     f"{s['mcnemar_p']:.2g} | {l['n11']} {l['n10']} {l['n01']} {l['n00']} | {l['agree'] * 100:.1f} % |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--policy", required=True, help="a ppo checkpoint (.pt) or 'expert'")
    p.add_argument("--tasks", required=True), p.add_argument("--out", required=True)
    p.add_argument("--world", choices=("mujoco", "nrd", "kin", "mjwarp"), required=True)
    p.add_argument("--ckpt", nargs="*", help="NRD checkpoint(s) (world nrd)"), p.add_argument("--members", nargs="*")
    p.add_argument("--device", default="cpu"), p.add_argument("--workers", type=int, default=0)
    p.add_argument("--mj-python", help="interpreter of the CPU MuJoCo workers (luffy: ~/miniconda3/envs/nrdsim/bin/python)")
    p.add_argument("--n", type=int, default=0), p.add_argument("--first", type=int, default=0)
    p.add_argument("--horizon", type=int), p.add_argument("--threads", type=int, default=4)
    p.add_argument("--train-rules", action="store_true", help="training rules (grasp box terminates) in this world")
    p.add_argument("--no-record", action="store_true", help="no per-step arrays (smaller files; no divergence report)")
    p.add_argument("--assist", action="store_true", help="DIAGNOSTIC: the training grasp assist (vec_env) in this world, "
                   "for policies of an assisted curriculum stage; never for reported success")
    p.add_argument("--stochastic", action="store_true", help="DIAGNOSTIC: sample the policy (its std, counter-based noise "
                   "per task seed and decision: the same noise in every world)")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--decoder-cfg", help="DecoderV2Config overrides (JSON): the expert's decoder; for a checkpoint it "
                   "must equal the checkpoint's own decoder")
    p = sub.add_parser("report")
    p.add_argument("--runs", nargs="+", required=True), p.add_argument("--ref")
    p.add_argument("--out", required=True), p.add_argument("--md"), p.add_argument("--examples", type=int, default=12)
    a = ap.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
